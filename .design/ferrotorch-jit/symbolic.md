# ferrotorch-jit — `symbolic` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/export/dynamic_shapes.py
  - torch/fx/experimental/symbolic_shapes.py
  - torch/_dynamo/symbolic_convert.py
-->

## Summary

`ferrotorch-jit/src/symbolic.rs` adds symbolic-shape (dynamic-batch)
support to traced modules. A normal trace records concrete input
shapes; re-running it with different shapes fails on hard-coded
dims. `SymbolicTracedModule` makes the trace polymorphic over a
declared set of symbolic dimensions (typically the batch axis) and
emits a runtime `Guard` that validates concrete input shapes before
execution. Mirrors `torch.export(model, args, dynamic_shapes={...})`
(`torch/export/dynamic_shapes.py:Dim`) and the symbolic-shape
machinery at
`torch/fx/experimental/symbolic_shapes.py:200-1200`.

## Requirements

- REQ-1: `pub struct ShapeSignature` declares the dynamic dimensions
  the trace is polymorphic over: `symbolic_dim(input_index, dim)`
  and `symbolic_dim_with_range(input_index, dim, min, max)`. Mirrors
  the dict-of-dim-name → `Dim(min, max)` argument that
  `torch.export.export(model, args, dynamic_shapes=...)` accepts.
- REQ-2: `pub struct SymbolicDim` is the per-dim record carrying
  the declared range (`min`, `max`).
- REQ-3: `pub struct Guard` is the runtime shape-check.
  `Guard::new(trace_shapes, signature)` constructs it;
  `Guard::check(inputs)` validates the concrete shapes against the
  signature, returning a descriptive error on violation.
- REQ-4: The guard validates: (a) input count matches, (b) per-input
  rank matches the trace-time rank, (c) static (non-symbolic) dims
  match exactly, (d) symbolic dims fall within their declared
  `[min, max]` range.
- REQ-5: `pub struct SymbolicTracedModule<T: Float>` wraps a
  `TracedModule<T>` plus its `Guard`. `forward_symbolic(inputs)`
  invokes the guard then delegates to the wrapped traced module's
  interpreter path.
- REQ-6: `pub fn patch_reshape_for_symbolic_dims(graph, signature,
  trace_shapes)` rewrites `Reshape` ops whose target shape literally
  equals a trace-time symbolic-dim value into `-1` (the
  infer-this-dim sentinel `ferrotorch_core::grad_fns::shape::reshape`
  already understands). Without this patch, a trace at `batch=4`
  would hard-code `[4, ...]` reshapes; the patch makes them
  polymorphic.
- REQ-7: `pub fn compile_symbolic<T, F>(f, example_inputs,
  signature) -> FerrotorchResult<SymbolicTracedModule<T>>` is the
  one-call entry point: trace, patch reshapes, build the guard,
  wrap in `SymbolicTracedModule`.

## Acceptance Criteria

- [x] AC-1: `ShapeSignature::new()` is empty;
  `.symbolic_dim(0, 0)` marks dim 0 of input 0 as symbolic;
  `.is_symbolic(0, 0)` returns `true`, `.is_symbolic(0, 1)` returns
  `false`.
- [x] AC-2: `Guard::check(&[wrong_input_count])` returns an error
  citing the count.
- [x] AC-3: `Guard::check(&[wrong_rank])` returns an error citing
  the rank.
- [x] AC-4: `Guard::check(&[symbolic_dim_outside_range])` returns
  an error citing the `[min, max]` range.
- [x] AC-5: `compile_symbolic(forward_fn, &[example], sig)` builds
  a `SymbolicTracedModule` that accepts inputs of different sizes
  along the symbolic dim.
- [x] AC-6: `patch_reshape_for_symbolic_dims` replaces literal
  `Reshape { shape: [4, 10] }` with `Reshape { shape: [-1, 10] }`
  for a trace at `batch=4`.

## Architecture

`ShapeSignature` is a builder-style struct carrying
`Vec<(input_index, Vec<SymbolicDim>)>`. The builder methods
(`symbolic_dim`, `symbolic_dim_with_range`) accumulate per-input
symbolic-dim lists, then `symbolic_dims_for(input_index)` /
`is_symbolic(input_index, dim)` query them.

`SymbolicDim { dim, min, max }` records the per-dim metadata. `min`
and `max` are `Option<usize>` so a fully-unconstrained dim is
representable.

`Guard` holds `trace_shapes: Vec<Vec<usize>>` and a clone of the
signature. `check(inputs: &[&[usize]])` walks each input's shape
against the corresponding trace shape; static dims must match
exactly, symbolic dims are validated against the declared range.

`SymbolicTracedModule<T>` holds the inner `TracedModule<T>` plus a
`Guard`. `forward_symbolic(inputs)` calls `guard.check(...)` then
`interpret(self.inner.graph(), inputs)` (`symbolic in symbolic.rs`).
The `inner()` accessor exposes the wrapped traced module for
serialisation and diagnostics; `guard()` exposes the guard.

`patch_reshape_for_symbolic_dims(graph, signature, trace_shapes)`
walks the IR nodes, finds any `IrOpKind::Reshape { shape }` whose
shape vector contains a literal that matches the trace-time value
of any symbolic dim, and replaces that entry with `-1`. The patched
graph is what `compile_symbolic` wraps.

`compile_symbolic(f, example_inputs, signature)` is the integrated
entry point: trace via `crate::trace::trace`, call
`patch_reshape_for_symbolic_dims`, build the guard from the
trace-time shapes + signature, wrap with `SymbolicTracedModule::new`.

### Non-test production consumers

- `pub use symbolic::{Guard, ShapeSignature, SymbolicDim,
  SymbolicTracedModule, compile_symbolic,
  patch_reshape_for_symbolic_dims}` at
  `ferrotorch-jit/src/lib.rs:113-115`.
- The `SymbolicTracedModule` and `compile_symbolic` paths are also
  used by the dynamic-shape export route in
  `ferrotorch-jit/src/export.rs::export_with_dynamic_shapes` (via
  the patch primitive); the resulting `ExportedProgram` carries
  `InputSpec`s with `DimSpec::Dynamic` entries.

## Parity contract

`parity_ops = []`. Symbolic shapes are structural; no numerics
host here. The wrapped interpreter handles execution.

Edge cases:

- **Symbolic dim outside `[min, max]`** — error with the actual
  value and declared range.
- **Mixed static/symbolic dims** — the same input can have some
  static dims and some symbolic ones; static dims are checked
  exactly.
- **Reshape with `-1` already** — the patch is a no-op; `-1` flows
  through unchanged.
- **Reshape with no symbolic-matching dim** — the patch is a
  no-op; the literal shape is preserved.

## Verification

Tests in `ferrotorch-jit/src/symbolic.rs` `mod tests` cover:
ShapeSignature builder, Guard check (happy/sad paths for each
violation), patch_reshape_for_symbolic_dims (positive + no-op
cases), compile_symbolic end-to-end with varying batch sizes.

Smoke command:

```bash
cargo test -p ferrotorch-jit --lib symbolic:: 2>&1 | tail -3
```

Expected: all tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct ShapeSignature` + `symbolic_dim` / `symbolic_dim_with_range` in `symbolic.rs`; non-test consumer: re-export at `lib.rs:113-115`; `pub fn compile_symbolic` accepts it as the third arg. |
| REQ-2 | SHIPPED | impl: `pub struct SymbolicDim` in `symbolic.rs`; non-test consumer: re-export at `lib.rs`; consumed by `ShapeSignature` internals. |
| REQ-3 | SHIPPED | impl: `pub struct Guard` + `Guard::new` + `Guard::check` in `symbolic.rs`; non-test consumer: `SymbolicTracedModule::forward_symbolic` invokes `self.guard.check(...)` at `symbolic in symbolic.rs`. |
| REQ-4 | SHIPPED | impl: the four-clause match inside `Guard::check` in `symbolic.rs` (input count, rank, static match, range); non-test consumer: `SymbolicTracedModule::forward_symbolic` is the production caller. |
| REQ-5 | SHIPPED | impl: `pub struct SymbolicTracedModule<T: Float>` + `pub fn new` + `pub fn forward_symbolic` in `symbolic.rs`; non-test consumer: re-export at `lib.rs`. |
| REQ-6 | SHIPPED | impl: `pub fn patch_reshape_for_symbolic_dims` in `symbolic.rs`; non-test consumer: `pub fn compile_symbolic` invokes it as the second step. |
| REQ-7 | SHIPPED | impl: `pub fn compile_symbolic<T, F>` in `symbolic.rs`; non-test consumer: re-export at `lib.rs`. |
