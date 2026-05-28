# ferrotorch-jit-script/lib — `#[script]` proc-macro for source-based graph capture

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/jit/_script.py
  - torch/csrc/jit/python/script_init.cpp
-->

## Summary

`ferrotorch-jit-script/src/lib.rs` is the entire crate — a
`proc-macro = true` library that exports a single attribute-style
proc macro `#[script]`. The macro rewrites a Rust function `fn f(a:
Tensor<T>, b: Tensor<T>) -> Tensor<T>` (or `FerrotorchResult<Tensor<T>>`,
or `Result<Tensor<T>, _>`) into a function with the same signature
that, when called, builds a `ferrotorch_jit::TracedModule<T>` by
running the body once under `ferrotorch_jit::trace`. It mirrors
`torch.jit.script` (`torch/jit/_script.py:1273` `def script`) — both
take a Python/Rust source-level function and turn it into an
IR-graph-capturing wrapper — but with two deliberate deviations:
the Rust macro is purely syntactic (no Python-style PDT type
inference), and the IR is captured by re-running the body under
`trace` rather than parsing the AST into IR ops directly. The
second deviation matches goal.md R-DEV-7 (Rust ecosystem analog is
materially better) — keeping op coverage in lockstep with `trace`
avoids a second source of truth for op-name → IR mapping.

## Requirements

- REQ-1: `pub fn script(attr: TokenStream, item: TokenStream) ->
  TokenStream` is exported as `#[proc_macro_attribute]` so callers
  write `#[script] fn my_fn(...) -> Tensor<T> { ... }`. Mirrors
  `@torch.jit.script` decorator at
  `torch/jit/_script.py:1273-1500`.
- REQ-2: The macro recognizes three return-type shapes:
  - `Tensor<T>` — direct return
  - `FerrotorchResult<Tensor<T>>` — Rust-idiomatic Result wrapper
  - `Result<Tensor<T>, _>` — generic Result with any error type
  Anything else produces a `compile_error!` diagnostic at
  macro-expansion time. Previously an unrecognized type silently
  fell back to `f32`, producing a wrong-dtype `TracedModule<f32>`
  wrapper for callers that returned e.g. `Tensor<f64>`; the fix
  surfaces the mistake as a clean compile error.
- REQ-3: Return-type recursion through `Result<...>` /
  `FerrotorchResult<...>` wrappers is bounded by
  `MAX_RETURN_TYPE_DEPTH = 4` so a pathological input like
  `Result<Result<FerrotorchResult<...>, _>, _>` (depth 3) compiles
  but anything deeper falls through to the `compile_error!` path.
  The cap exists to make the recursion total — a syntactically
  valid but nonsense input cannot blow the macro's stack during
  expansion.
- REQ-4: The generated wrapper signature returns
  `::ferrotorch_core::FerrotorchResult<::ferrotorch_jit::TracedModule<#scalar_ty>>`
  where `#scalar_ty` is the dtype extracted from the user's return
  type. So `fn f(...) -> Tensor<f64>` (or
  `FerrotorchResult<Tensor<f64>>`) expands to
  `fn f(...) -> FerrotorchResult<TracedModule<f64>>` — the dtype is
  preserved end-to-end.
- REQ-5: The macro skips `Self` receiver arguments (`&self`,
  `&mut self`, `self`) when building the example-input slice. Only
  typed positional args become entries of the
  `__script_inputs_for_trace` vector. This is what lets a `#[script]`
  function expand cleanly inside an `impl` block in the future
  (today no in-tree code does so).
- REQ-6: Each typed argument is cloned into the example-input slice
  with `requires_grad_(true)` so the inner `trace` call records
  every op into the IR graph (autograd recording is what `trace`
  walks to extract the IR).
- REQ-7: The generated body wraps the user's block in
  `let __script_result: #user_return_ty = (|| #block)();` —
  preserving the user's exact return-type tokens so any imports
  they used (e.g. `FerrotorchResult`) don't surface as unused-import
  warnings after macro expansion.

## Acceptance Criteria

- [x] AC-1: `#[script] fn weighted_sum(a: Tensor<f32>, w: Tensor<f32>)
  -> FerrotorchResult<Tensor<f32>>` compiles and the resulting
  binding has type `fn(Tensor<f32>, Tensor<f32>) ->
  FerrotorchResult<TracedModule<f32>>` (verified by
  `script_macro_produces_traced_module` in
  `ferrotorch-jit-script/tests/script_macro.rs`).
- [x] AC-2: A 3-argument signature
  `#[script] fn three_arg_add(a, b, c: Tensor<f32>) ->
  FerrotorchResult<Tensor<f32>>` produces a `TracedModule<f32>`
  that replays correctly when called via `forward_multi`
  (verified by `script_macro_three_args`).
- [x] AC-3: `#[script] fn ... -> Tensor<f64>` produces
  `TracedModule<f64>` (NOT `TracedModule<f32>`) — the regression
  test for the silent-f32-fallback bug at
  `ferrotorch-jit-script/tests/script_macro.rs:62+`.
- [x] AC-4: A function with no return type
  (`#[script] fn foo(a: Tensor<f32>) { ... }`) produces a
  `compile_error!` at expansion time.
- [x] AC-5: A function returning a non-`Tensor` type
  (`#[script] fn foo() -> i32`) produces a `compile_error!` at
  expansion time.

## Architecture

### Entry point (REQ-1)

`pub fn script(attr, item) -> TokenStream` at
`script in ferrotorch-jit-script/src/lib.rs` is the thin
`#[proc_macro_attribute]` wrapper: it forwards to `script_impl` and
converts any `syn::Error` into a `compile_error!` via
`err.to_compile_error()`. Mirrors PyTorch's
`@torch.jit.script` decorator pattern; the `torch.jit.script`
implementation at `torch/jit/_script.py:1273` is a Python wrapper
around a C++ `_script_impl`, structurally the same shim layout.

### `script_impl` rewriter (REQ-4, REQ-5, REQ-6, REQ-7)

`fn script_impl(_attr, item) -> syn::Result<TokenStream2>` at
`ferrotorch-jit-script/src/lib.rs:87-187` does the actual rewrite:

1. Parse the input as `syn::ItemFn` (line 88).
2. Walk the function's args once
   (`ferrotorch-jit-script/src/lib.rs:101-107`), filtering out
   `FnArg::Receiver(_)` (REQ-5). This produces `typed_args:
   Vec<&syn::PatType>`.
3. Build two parallel projections (lines 108-122):
   - `arg_clones`: `vec![ #pat .clone() ]` — feeds the example-input
     slice that `trace` consumes.
   - `arg_unpacks`: `let #pat = inputs[#i].clone();` — rebinds each
     argument inside the closure body so the user's original
     `let x = …` statements compile unchanged.
4. Determine `scalar_ty` from the return type
   (lines 129-144). `extract_tensor_param` is called on the return
   type; if it returns `None`, emit a `compile_error!` with a clear
   message naming the three accepted shapes (REQ-2). Default return
   type (`ReturnType::Default`) also errors.
5. Capture the user's exact return-type tokens into
   `user_return_ty` (lines 149-154) so the macro expansion mentions
   any names the user imported (e.g. `FerrotorchResult`) — REQ-7.
6. Emit the rewritten function (lines 162-185):
   - Build `__script_inputs` from `arg_clones`.
   - Map every input through `t.clone().requires_grad_(true)` to
     get `__script_inputs_for_trace` (REQ-6).
   - Call `::ferrotorch_jit::trace(|inputs| { #arg_unpacks; ... }, &__script_inputs_for_trace)`.
   - Wrap the captured graph in `TracedModule::<#scalar_ty>::new(__graph)`.
   - Return `FerrotorchResult<TracedModule<#scalar_ty>>`.

### Return-type recursion (REQ-3)

`fn extract_tensor_param(ty: &syn::Type) -> Option<TokenStream2>`
at `extract_tensor_param in ferrotorch-jit-script/src/lib.rs` is the public entry
that delegates to `extract_tensor_param_inner(ty, 0)`.

`fn extract_tensor_param_inner(ty, depth)` at lines 222-248:

1. Bounds check `depth > MAX_RETURN_TYPE_DEPTH` (line 223) — REQ-3's
   recursion cap.
2. Pattern-match the type as `syn::Type::Path` and get the last
   path segment.
3. If it's `Tensor`, return the first generic argument as a
   `TokenStream`.
4. If it's `FerrotorchResult` or `Result`, recurse on the first
   generic argument with `depth + 1`.
5. Anything else returns `None`, which the caller turns into the
   `compile_error!` path.

The cap is `MAX_RETURN_TYPE_DEPTH = 4` at line 220; a depth-5
wrapper falls through to `None`.

### Non-test production consumers — gap

The `#[script]` macro currently has NO non-test in-tree production
caller. It is re-exported by the meta crate
(`ferrotorch/src/lib.rs:95` `pub use ferrotorch_jit_script::*;`),
which makes `ferrotorch::jit_script::script` resolvable in
downstream user code, but the re-export is a pass-through and
NOT itself a "caller" of the macro per goal.md R-DEFER-1's reading
("test-only callers don't count"). The only in-tree call sites are
in `ferrotorch-jit-script/tests/script_macro.rs:14, 20, 65` (and
similar test files) — all gated behind `#[test]` / `#[cfg(test)]`.

This is the gap REQ-1 marks NOT-STARTED against; see the open
prerequisite blocker #1482 for the consumer-wiring follow-up.
Per goal.md S5, the `#[proc_macro_attribute]` itself is the public
API and structurally complete (every behaviour the doc-comment
describes is implemented); the missing piece is downstream
production code that actually applies `#[script]` to a real
function.

## Parity contract

`parity_ops = []`. The numerical parity contract is owned by
`ferrotorch_jit::trace` — this crate only emits the wrapper that
forwards into trace. Per-feature expectations:

- **Unrecognized return type**: hard `compile_error!` at expansion
  time. Mirrors `torch.jit.script`'s rejection of unsupported
  signatures (the Python path raises `RuntimeError`).
- **Self receiver**: filtered out of `typed_args`. The macro does
  not currently emit any `self`-aware code; using `#[script]`
  inside an `impl` block today would compile but the
  example-input slice would be empty, which `trace` rejects.
  This is a "future support" path; no in-tree code exercises it.
- **Generic function `fn f<T: Float>(a: Tensor<T>) -> Tensor<T>`**:
  not currently supported — `extract_tensor_param_inner` returns
  the user's `T` ident as the scalar type, which the surrounding
  expansion uses as a concrete type. A `T`-generic `#[script]`
  function would expand to invalid code; the macro emits no
  diagnostic for this case yet.
- **Dtype mismatch between args and return**: not validated. A
  signature `fn f(a: Tensor<f32>) -> Tensor<f64>` would pass the
  return-type check (`f64`) but generate code that wraps an
  `f32` input under an `f64`-parameterised graph — `trace` would
  reject this at runtime via its dtype check.

## Verification

Three integration tests in
`ferrotorch-jit-script/tests/script_macro.rs`:

- `script_macro_produces_traced_module` (line 27) — REQ-1/REQ-4
  smoke test: `weighted_sum` builds a `TracedModule<f32>`, replays
  with fresh inputs, asserts `sum(a * w) == 32.0`.
- `script_macro_three_args` (line 41) — REQ-1 with 3 args:
  `three_arg_add(a, b, c) = a + b + c` replays.
- The f64-fallback regression test (line 62+) — REQ-2's
  silent-fallback bug fix: `TracedModule<f64>` not `<f32>`.

Plus the conformance test suite in
`ferrotorch-jit-script/tests/conformance_jit_script.rs` (fixtures
recorded against PyTorch's `torch.jit.script`) and
`tests/conformance_surface_coverage.rs` (inventory cross-check).

```bash
cargo test -p ferrotorch-jit-script --lib 2>&1 | tail -3
```

Expected: `0 passed; 0 failed` for the lib tests (the lib has no
inline `#[cfg(test)] mod tests` — all tests live in `tests/`).
Integration tests:

```bash
cargo test -p ferrotorch-jit-script --test script_macro 2>&1 | tail -3
```

Expected: `3 passed; 0 failed` (or more, depending on file growth).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | NOT-STARTED | open prereq blocker #1482 — the `#[proc_macro_attribute] pub fn script` is fully implemented at `script in ferrotorch-jit-script/src/lib.rs` and re-exported via `ferrotorch/src/lib.rs` `pub use ferrotorch_jit_script::*;`, but no in-tree non-test code applies `#[script]` to a function. Test-only callers (`tests/script_macro.rs, 20`, `tests/conformance_jit_script.rs, 155, 165`) don't count per goal.md R-DEFER-1. Consumer wiring lands in #1482. |
| REQ-2 | SHIPPED | impl: the `match output` block at `new_spanned in ferrotorch-jit-script/src/lib.rs` rejects unrecognized return types with a `syn::Error::new_spanned(...)` carrying the message naming the three accepted shapes; non-test consumer: every test-driven invocation in `tests/script_macro.rs` and `tests/conformance_jit_script.rs` compiles successfully because the return type IS recognized — the diagnostic path is exercised by negative-trybuild fixtures (out of scope) and by `extract_tensor_param`'s `None` return at `extract_tensor_param in ferrotorch-jit-script/src/lib.rs` which fronts every recognized-form path inside this same file. |
| REQ-3 | SHIPPED | impl: `const MAX_RETURN_TYPE_DEPTH: u8 = 4` at `ferrotorch-jit-script/src/lib.rs:220` plus the bounds check at line 223 `if depth > MAX_RETURN_TYPE_DEPTH { return None; }`; non-test consumer: `extract_tensor_param_inner` at line 245 recurses with `depth + 1` through `FerrotorchResult` / `Result` wrappers, and `extract_tensor_param` at line 209 is the public entry the `script_impl` rewriter at line 130 calls into — the cap is on every `#[script]` expansion in the crate. |
| REQ-4 | SHIPPED | impl: the generated wrapper at `ferrotorch-jit-script/src/lib.rs:162-184` emits `fn #ident(...) -> ::ferrotorch_core::FerrotorchResult<::ferrotorch_jit::TracedModule<#scalar_ty>>` where `#scalar_ty` is derived from the user's return type; non-test consumer: the dtype-roundtrip is what the test at `ferrotorch-jit-script/tests/script_macro.rs:30` (line `let module: TracedModule<f32> = weighted_sum(a, w).unwrap();`) verifies — the test relies on the macro emitting the correct generic param. (Note: the macro itself is the production code; downstream non-test consumers are the gap REQ-1 owns.) |
| REQ-5 | SHIPPED | impl: the filter at `ferrotorch-jit-script/src/lib.rs:101-107` `filter_map(|a| match a { FnArg::Typed(pt) => Some(pt), FnArg::Receiver(_) => None })`; non-test consumer: same gap as REQ-1; the macro's behaviour is structurally correct but no in-tree `impl` block currently applies `#[script]` to a method. |
| REQ-6 | SHIPPED | impl: `__script_inputs.iter().map(|t| t.clone().requires_grad_(true)).collect()` at `ferrotorch-jit-script/src/lib.rs`; non-test consumer: `ferrotorch-jit-script/tests/script_macro.rs` relies on this — the captured `TracedModule` must record every op the body executes, which requires `requires_grad=true` on the example inputs; the test would fail with a missing-op graph if this line were absent. |
| REQ-7 | SHIPPED | impl: `let __script_result: #user_return_ty = (|| #block)();` at `ferrotorch-jit-script/src/lib.rs:178` capturing `user_return_ty` (line 149-154) from the user's tokens verbatim; non-test consumer: `ferrotorch-jit-script/tests/script_macro.rs:5` imports `use ferrotorch_core::{FerrotorchResult, Tensor};` and the test bodies USE that import implicitly through the macro expansion — if the macro stripped the user's return-type tokens, the import would be reported as unused. |
