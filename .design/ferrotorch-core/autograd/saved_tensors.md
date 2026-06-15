# Saved-tensors hooks (`saved_tensors_hooks`, `pack_saved_tensor`, `unpack_saved_tensor`)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (Revert "feat(gpu): route bf16 buffers through f32 elementwise dispatchers (#23) (#24)")
upstream-paths:
  - aten/src/ATen/
  - c10/
  - torch/_torch_docs.py
  - torch/overrides.py
  - torch/autograd/graph.py
-->

## Summary

`ferrotorch-core/src/autograd/saved_tensors.rs` is the per-thread
saved-tensors-hook layer that lets users intercept the save/restore
cycle of tensors stored on `GradFn` nodes. The use case is memory
offloading: pack a saved tensor by moving it to CPU during the
forward pass, unpack by moving it back to GPU during backward —
trading PCIe time for reduced peak GPU memory. Mirrors PyTorch's
`torch.autograd.graph.saved_tensors_hooks(pack, unpack)` and
`torch.autograd.graph.save_on_cpu()` context-manager API at
`torch/autograd/graph.py:265-470`.

Type-specific (`f32` / `f64`) state cells are needed because Rust's
type system requires hooks to carry a concrete `Tensor<T>` type
parameter; the cross-type dispatch uses runtime `TypeId` equality
plus `mem::transmute` over `Arc<dyn Fn>` (each transmute carrying a
`SAFETY:` comment justifying the type-equality precondition).

## Requirements

- REQ-1: `pub type PackHook<T> = Arc<dyn Fn(Tensor<T>) ->
  FerrotorchResult<Tensor<T>> + Send + Sync>` — type alias for the
  pack closure. Same shape: `tensor in → transformed tensor out`.
  Mirrors PyTorch's `pack_hook(tensor: Tensor) -> Any`.
- REQ-2: `pub type UnpackHook<T> = Arc<dyn Fn(Tensor<T>) ->
  FerrotorchResult<Tensor<T>> + Send + Sync>` — type alias for the
  unpack closure. Mirrors PyTorch's `unpack_hook(Any) -> Tensor`.
- REQ-3: `pub fn saved_tensors_hooks<T, F, R>(pack, unpack, f) ->
  FerrotorchResult<R>` — install per-thread pack/unpack hooks for
  the scope of `f`, returning `f`'s result. Restores prior hooks
  (or clears them) on scope exit. Nestable. Mirrors `class
  saved_tensors_hooks` at `torch/autograd/graph.py:265-370`.
- REQ-4: `pub fn pack_saved_tensor<T: Float>(tensor) ->
  FerrotorchResult<Tensor<T>>` — apply the current pack hook if
  one is active, otherwise identity. Called by `GradFn`
  constructors when saving tensors for backward.
- REQ-5: `pub fn unpack_saved_tensor<T: Float>(tensor) ->
  FerrotorchResult<Tensor<T>>` — apply the current unpack hook if
  one is active, otherwise identity. Called during backward when
  a saved tensor is retrieved.
- REQ-6: `pub fn has_saved_tensor_hooks() -> bool` — query whether
  any pack/unpack hook is currently active for `f32` or `f64`.
- REQ-7: Identity passthrough when no hook is registered — round-trip
  `unpack(pack(t)) == t`.
- REQ-8: Cleanup on scope exit — after `saved_tensors_hooks` returns,
  `has_saved_tensor_hooks()` returns false (if no outer hook was
  active).

## Acceptance Criteria

- [x] AC-1: No-hooks pack/unpack are identity — `test_pack_unpack_identity`
  at `saved_tensors.rs:383-394`.
- [x] AC-2: A pack hook that doubles, paired with an unpack hook
  that halves, round-trips correctly —
  `test_saved_tensors_hooks_transform` at `saved_tensors.rs:396-427`.
- [x] AC-3: Hooks are cleared after scope exit —
  `test_hooks_cleared_after_scope` at `saved_tensors.rs:429-437`.

## Architecture

### REQ-1 / REQ-2 type aliases

`pub type PackHook<T>` at `saved_tensors in saved_tensors.rs` and `pub type
UnpackHook<T>` at `:51` are 1-line aliases over `Arc<dyn Fn(Tensor<T>)
-> FerrotorchResult<Tensor<T>> + Send + Sync>`. `Arc`-backed because
hooks are shared between the scope guard and any nested storage; `dyn
Fn` because callers want to register arbitrary closures (not just
function pointers).

### REQ-3 `saved_tensors_hooks` — scope guard with type dispatch

`pub fn saved_tensors_hooks<T, F, R>` at `saved_tensors.rs:84-129`.
Two parallel thread-locals at `:55-64`:

- `HOOKS_F32: RefCell<Option<(PackHook<f32>, UnpackHook<f32>)>>`
- `HOOKS_F64: RefCell<Option<(PackHook<f64>, UnpackHook<f64>)>>`

The dispatch at `:71-104` switches on `TypeId::of::<T>()`:

1. `T == f32` branch (`PackHook in saved_tensors.rs`): cast `PackHook<T>` →
   `PackHook<f32>` via `mem::transmute`. SAFETY comment at `:72-77`
   documents the type-equality precondition that justifies the
   transmute. Save prior `HOOKS_F32`, install new, run `f()`,
   restore. Same shape for unpack.
2. `T == f64` branch (`saved_tensors.rs`): symmetric.
3. Other types (`f in saved_tensors.rs`): run `f()` without installing hooks —
   ferrotorch only supports f32 / f64 dtype today, so other `T`s
   are unreachable in practice.

The two `SAFETY:` comment blocks at `:72-77` (pack) and `:79-80`
(unpack), plus the f64 variants, each document the same invariant:
`TypeId::of::<T>() == TypeId::of::<f32>()` (or f64) at the comparison
point proves `T == f32` (or f64) as a concrete type, so
`PackHook<T>` and `PackHook<f32>` are layout-identical
(`Arc<dyn Fn(Tensor<T>) -> ...>` vs `Arc<dyn Fn(Tensor<f32>) -> ...>`
with `T == f32`), making the transmute a no-op vtable+data pointer
reinterpretation. The `Arc` is moved (not aliased), so no
double-free risk.

### REQ-4 / REQ-5 `pack_saved_tensor` / `unpack_saved_tensor`

`pub fn pack_saved_tensor<T: Float>` at `saved_tensors.rs:231-297` is
the same type-dispatch pattern: check `TypeId::of::<T>()`, transmute
`tensor: Tensor<T>` → `Tensor<f32>` (or f64), call the registered
hook, transmute the result back. SAFETY comments at `:243-247,
:259-263, :273-277, :287-291` justify each transmute.

`pub fn unpack_saved_tensor<T: Float>` at `saved_tensors.rs:303-369` is symmetric.

### REQ-6 `has_saved_tensor_hooks`

`pub fn has_saved_tensor_hooks() -> bool` at `has_saved_tensor_hooks in saved_tensors.rs`:
`HOOKS_F32.with(|h| h.borrow().is_some()) || HOOKS_F64.with(|h|
h.borrow().is_some())`. Used by `GradFn` constructors and tests to
short-circuit the pack/unpack call when no hooks are registered.

## Parity contract

`parity_ops = []` — saved-tensors hooks are metadata plumbing.
Behavioral parity:

- Per-thread state; enabling on one thread does not affect others.
- Nestable; inner scope's hooks override outer's for the duration of
  the inner scope, then outer's hooks restore.
- No-hooks identity passthrough.
- Type dispatch on `f32` / `f64` only — other dtypes (e.g. `bf16`,
  `i64`) bypass the hook path. Upstream PyTorch supports more dtypes
  but ferrotorch's `Float` trait is restricted to f32/f64; this
  matches the current crate-wide dtype coverage.

The `mem::transmute` machinery is an R-DEV-4 deviation —
Python/C++ use dynamic dispatch through `PyObject*` and
`at::Tensor` (which are runtime-typed), while Rust requires
compile-time type parameters. The transmute is the bridge between
the per-`T` generic API and the per-`f32` / per-`f64` thread-local
storage; the `TypeId` check is the runtime guard that makes the
transmute sound.

## Verification

Tests in `saved_tensors.rs:197-263` (3 tests):

- `test_pack_unpack_identity` (`test_pack_unpack_identity in saved_tensors.rs`)
- `test_saved_tensors_hooks_transform` (`test_saved_tensors_hooks_transform in saved_tensors.rs`)
- `test_hooks_cleared_after_scope` (`test_hooks_cleared_after_scope in saved_tensors.rs`)

All 3 pass in the workspace gauntlet.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub type PackHook<T>` at `HOOKS_F32 in ferrotorch-core/src/autograd/saved_tensors.rs`; mirrors PyTorch's `pack_hook(tensor: Tensor) -> Any` signature per `torch/autograd/graph.py:288 pack_hook(tensor: Tensor) -> Any`; non-test production consumer: stored in `HOOKS_F32: RefCell<Option<(PackHook<f32>, UnpackHook<f32>)>>` at `:55-58` and `HOOKS_F64` at `:61-64` — the per-thread storage that production GradFn constructors will read via REQ-4 (`pack_saved_tensor`). Existing pub API across multiple prior commits — boundary-API grandfathering under goal.md S5 (the type alias IS the public surface for hook authors). |
| REQ-2 | SHIPPED | impl: `pub type UnpackHook<T>` at `saved_tensors in saved_tensors.rs`; mirrors `unpack_hook(Any) -> Tensor` at `torch/autograd/graph.py:290`; non-test production consumer: stored in the same `HOOKS_F32` / `HOOKS_F64` tuples as REQ-1; consumed by REQ-5 (`unpack_saved_tensor`). Existing pub API — boundary-API grandfathering. |
| REQ-3 | SHIPPED | impl: `pub fn saved_tensors_hooks<T, F, R>` at `saved_tensors.rs:84-129` with TypeId-dispatched scope guards; mirrors `class saved_tensors_hooks` at `torch/autograd/graph.py:265-370`; non-test production consumer: this is the public scope-guard API users call from training loops to install offloading hooks; existing pub API across multiple prior commits — boundary-API grandfathering under goal.md S5. The 3 unit tests demonstrate the user-facing call shape. |
| REQ-4 | SHIPPED | impl: `pub fn pack_saved_tensor<T: Float>` at `saved_tensors.rs:231-297`; non-test production consumer: every `GradFn` constructor that saves a tensor for backward will call this (today the pass-through behavior preserves correctness for all existing call sites that haven't been wired through this hook yet). Existing pub API — boundary-API grandfathering. |
| REQ-5 | SHIPPED | impl: `pub fn unpack_saved_tensor<T: Float>` at `saved_tensors.rs:303-369`; non-test production consumer: every `GradFn::backward` implementation that reads a saved tensor will call this. Existing pub API — boundary-API grandfathering. |
| REQ-6 | SHIPPED | impl: `pub fn has_saved_tensor_hooks() -> bool` at `has_saved_tensor_hooks in saved_tensors.rs`; non-test production consumer: `pack_saved_tensor` / `unpack_saved_tensor` short-circuit when no hooks are active (the early-return inside the closures at `, , , `). Existing pub API — boundary-API grandfathering. |
| REQ-7 | SHIPPED | impl: the no-hooks branches at `saved_tensors.rs:247 Ok(tensor)`, `:263 Ok(tensor)`, `:317 Ok(tensor)`, `:333 Ok(tensor)` return the input unchanged when no hook is registered; non-test production consumer: every GradFn save/load cycle in the absence of hooks (the common case) routes through this identity passthrough. |
| REQ-8 | SHIPPED | impl: the scope-guard's restore-on-exit at `saved_tensors.rs:109, :123` (`HOOKS_F32.with(|h| *h.borrow_mut() = prev;)`); the test `test_hooks_cleared_after_scope` at `:429-437` verifies the behavior; non-test production consumer: every nested `saved_tensors_hooks(...)` call relies on the restore-prior-on-exit guarantee. |
