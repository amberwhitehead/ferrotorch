# Autograd tensor hooks (`HookStorage`, `HookHandle`, `register_hook`, `register_post_accumulate_grad_hook`)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (Revert "feat(gpu): route bf16 buffers through f32 elementwise dispatchers (#23) (#24)")
upstream-paths:
  - torch/utils/hooks.py
  - torch/nn/modules/module.py
-->

## Summary

`ferrotorch-core/src/autograd/hooks.rs` is the per-tensor hook storage
layer used by `Tensor::register_hook` and
`Tensor::register_post_accumulate_grad_hook`. Each `TensorInner` carries
a `Mutex<HookStorage<T>>` containing two short `Vec`s — one of
`GradHook<T>` (called during backward with the gradient, may return a
replacement) and one of `PostAccumulateGradHook<T>` (called after leaf
gradient accumulation, may inspect but not modify). Hooks are
identified by a globally-unique `HookHandle(u64)` produced from a
process-wide `AtomicU64` counter. Mirrors PyTorch's
`torch.utils.hooks.RemovableHandle` and the hook-list machinery used by
`torch.Tensor.register_hook` and
`torch.Tensor.register_post_accumulate_grad_hook` (`torch/utils/hooks.py:10-260`).

## Requirements

- REQ-1: `HookHandle` — opaque, copyable, Hash-able u64 identifier
  produced from a process-wide `AtomicU64` counter. Two hooks on
  different tensors NEVER share a handle. Mirrors
  `RemovableHandle.id` at `torch/utils/hooks.py:10-71`.
- REQ-2: `HookStorage<T: Float>` — per-tensor container with two
  `Vec`s (`grad_hooks: Vec<GradHook<T>>`,
  `post_accumulate_hooks: Vec<PostAccumulateGradHook<T>>`). Short-vec
  storage with linear-scan removal is correct because hook counts
  per tensor are tiny (usually 0-2).
- REQ-3: `GradHook<T>` — wraps a
  `Box<dyn Fn(&Tensor<T>) -> Option<Tensor<T>> + Send + Sync>` plus
  the assigned `HookHandle`. Returning `None` means keep the
  original gradient; `Some(new_grad)` replaces it. Mirrors
  PyTorch's pattern at `torch/utils/hooks.py:13 — hook signature is
  hook(grad) -> Optional[Tensor]`.
- REQ-4: `PostAccumulateGradHook<T>` — wraps a
  `Box<dyn Fn(&Tensor<T>) + Send + Sync>` plus a `HookHandle`. Cannot
  return a replacement (matches upstream's post-accumulate semantic).
  Mirrors PyTorch's `register_post_accumulate_grad_hook` from
  `torch/_tensor.py`.
- REQ-5: `HookStorage::add_grad_hook` / `add_post_accumulate_hook`
  push a new hook into the appropriate vector and return the assigned
  `HookHandle`.
- REQ-6: `HookStorage::remove(handle)` — linear-scan both vectors,
  retain only entries whose `handle != handle`. Returns `true` if any
  hook was removed. Mirrors `RemovableHandle.remove` at
  `torch/utils/hooks.py:48-71`.
- REQ-7: `run_grad_hooks(hooks: &Mutex<HookStorage<T>>, grad:
  Tensor<T>) -> FerrotorchResult<Tensor<T>>` — invoked from the
  autograd engine; chains hooks in registration order, threading
  each hook's optional replacement to the next.
- REQ-8: `run_post_accumulate_hooks(hooks, tensor)` — invoked from
  the autograd engine AFTER `tensor.accumulate_grad` writes the leaf
  gradient. Loops the vec, calling each hook for side-effects.

## Acceptance Criteria

- [x] AC-1: Two consecutive `HookHandle::next()` calls yield distinct
  handles — `test_hook_handle_uniqueness in hooks.rs`.
- [x] AC-2: `add_grad_hook` + `remove` lifecycle works — adding then
  removing brings the storage back to empty, returning `true` from
  remove — `test_hook_storage_add_remove in hooks.rs`.
- [x] AC-3: `run_grad_hooks` on empty storage is identity passthrough
  — `test_run_grad_hooks_passthrough in hooks.rs`.
- [x] AC-4: A hook returning `Some(replacement)` replaces the
  gradient downstream — `test_run_grad_hooks_replace` at
  `hooks.rs:240-252`.
- [x] AC-5: Multiple chained hooks see the output of the previous
  hook — `test_run_grad_hooks_chain` at `hooks.rs:254-273`.
- [x] AC-6: Post-accumulate hooks fire on `run_post_accumulate_hooks`
  — `test_post_accumulate_hook_fires` at `hooks.rs:275-294`.
- [x] AC-7: `remove(fake_handle)` returns `false` cleanly — does NOT
  panic on a non-existent handle — `test_remove_nonexistent_handle`
  at `test_remove_nonexistent_handle in hooks.rs`.

## Architecture

### REQ-1 `HookHandle`

`pub struct HookHandle(u64)` at `HookHandle in hooks.rs` is a newtype around `u64`
with `Debug, Clone, Copy, PartialEq, Eq, Hash` derived. The
constructor `HookHandle::next()` at `hooks.rs:32-36` is `pub(crate)`
gated through the global `static NEXT_HOOK_ID: AtomicU64` at
`hooks.rs:23` — every `next()` fires a `fetch_add(1, Relaxed)` to
produce a fresh unique id. The `Hash` derive lets callers store hooks
in a `HashMap<HookHandle, _>` if needed.

### REQ-2 `HookStorage<T>`

`pub(crate) struct HookStorage<T: Float>` at `HookStorage in hooks.rs` carries
two `Vec`s. Crate-private — the public API for hook registration is
`Tensor::register_hook` (`ferrotorch-core/src/tensor.rs:933`) and
`Tensor::register_post_accumulate_grad_hook`
(`tensor.rs:483`), and the public API for removal is
`Tensor::remove_hook(handle)` (`tensor.rs:502`). The
`Mutex<HookStorage<T>>` is one field on `TensorInner` at
`TensorInner in tensor.rs` — every tensor allocation initializes it via
`HookStorage::new()` (zero-allocation: empty Vecs).

### REQ-3 / REQ-4 hook records

`pub(crate) struct GradHook<T>` at `GradHook in hooks.rs` and
`pub(crate) struct PostAccumulateGradHook<T>` at `PostAccumulateGradHook in hooks.rs` are
pair-of-field structs: a `HookHandle` for removal lookup and a
`Box<dyn Fn>` for the user-provided closure. The `dyn Fn` trait
objects carry `Send + Sync + 'static` bounds because the autograd
engine may dispatch hooks across worker threads in the parallel
backward path (see REQ-9 of the graph.md doc).

### REQ-5 / REQ-6 add/remove

`HookStorage::add_grad_hook<F>` at `hooks.rs:90-100` and
`add_post_accumulate_hook<F>` at `:103-113` are generic over the
closure type `F: Fn(...) + Send + Sync + 'static`, box the closure on
push, and return the assigned `HookHandle`. `HookStorage::remove` at
`:101-108` is a two-vec retain — short-circuit safe because the vecs
are tiny.

### REQ-7 / REQ-8 dispatch helpers

`pub(crate) fn run_grad_hooks` at `hooks.rs:140-180` locks the mutex
(returning `FerrotorchError::LockPoisoned` on poisoning) and walks the
`grad_hooks` vec, replacing `current` with the hook's return value
when `Some`. `pub(crate) fn run_post_accumulate_hooks` at
`:182-199` is the symmetric loop for side-effecting
post-accumulate hooks. Both are called from the autograd engine at
`graph.rs:175-193` (sequential) and `:515-407` (parallel).

## Parity contract

`parity_ops = []` — hook storage is per-tensor metadata, not a
tensor-valued op. Behavioral parity vs upstream:

- Hook registration is order-preserving — a hook registered first
  fires first.
- The handle returned by `register_*` can be passed to `remove_hook`
  to deregister. Removing a non-existent handle returns false (does
  not panic).
- Per-tensor hook lists are bounded only by available memory; the
  short-vec linear scan is correctness-equivalent to upstream's
  `OrderedDict[id, hook]` lookup, with the same `O(n)` registration
  / removal cost (acceptable for the n=0-2 hooks per tensor common
  case).
- The thread-safety substitution (`Mutex` vs Python GIL) is R-DEV-4
  permitted: PyTorch's hook-list mutation is GIL-protected; ferrotorch
  uses an explicit `Mutex` because Rust has no GIL.

## Verification

Tests in `hooks.rs:206-299` (7 tests):

- `test_hook_handle_uniqueness` (`test_hook_handle_uniqueness in hooks.rs`)
- `test_hook_storage_add_remove` (`test_hook_storage_add_remove in hooks.rs`)
- `test_run_grad_hooks_passthrough` (`test_run_grad_hooks_passthrough in hooks.rs`)
- `test_run_grad_hooks_replace` (`test_run_grad_hooks_replace in hooks.rs`)
- `test_run_grad_hooks_chain` (`test_run_grad_hooks_chain in hooks.rs`)
- `test_post_accumulate_hook_fires` (`test_post_accumulate_hook_fires in hooks.rs`)
- `test_remove_nonexistent_handle` (`test_remove_nonexistent_handle in hooks.rs`)

All 7 pass in the workspace gauntlet.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct HookHandle(u64)` at `HookHandle in ferrotorch-core/src/autograd/hooks.rs` with `static NEXT_HOOK_ID: AtomicU64` at `pub in ferrotorch-core/src/autograd/hooks.rs` and `HookHandle::next` factory at `next in ferrotorch-core/src/autograd/hooks.rs`; mirrors `class RemovableHandle` at `torch/utils/hooks.py:10-71`; non-test production consumer: `pub fn register_hook` at `register_hook in ferrotorch-core/src/tensor.rs` returns `HookHandle` and `pub fn remove_hook(handle: HookHandle)` at `remove_hook in tensor.rs` consumes it — the public Tensor API users call from `loss.backward()` workflows. Re-exported at `tensor in ferrotorch-core/src/lib.rs pub use autograd::hooks::HookHandle`. |
| REQ-2 | SHIPPED | impl: `pub(crate) struct HookStorage<T: Float>` at `HookStorage in hooks.rs` with `HookStorage::new` factory at `new in hooks.rs`; non-test production consumer: `TensorInner.hooks: Mutex<crate::autograd::hooks::HookStorage<T>>` at `new in ferrotorch-core/src/tensor.rs` and 7 `HookStorage::new()` call sites at `new in tensor.rs, , , , , , , , , ` (every tensor-construction path). |
| REQ-3 | SHIPPED | impl: `pub(crate) struct GradHook<T>` at `GradHook in hooks.rs` plus the `GradHookFn<T>` type alias at `GradHook in hooks.rs`; mirrors hook signature `hook(grad) -> Optional[Tensor]` at `torch/utils/hooks.py:13`; non-test production consumer: stored inside `HookStorage.grad_hooks: Vec<GradHook<T>>` per REQ-2; populated by `Tensor::register_hook` calling `HookStorage::add_grad_hook` at `add_grad_hook in tensor.rs`. |
| REQ-4 | SHIPPED | impl: `pub(crate) struct PostAccumulateGradHook<T>` at `PostAccumulateGradHook in hooks.rs` plus `PostAccumulateHookFn<T>` type alias at `PostAccumulateHookFn in hooks.rs`; mirrors `register_post_accumulate_grad_hook`; non-test production consumer: stored inside `HookStorage.post_accumulate_hooks: Vec<PostAccumulateGradHook<T>>` per REQ-2; populated by `Tensor::register_post_accumulate_grad_hook` at `register_post_accumulate_grad_hook in tensor.rs`. |
| REQ-5 | SHIPPED | impl: `HookStorage::add_grad_hook<F>` at `hooks.rs:90-100` and `add_post_accumulate_hook<F>` at `:103-113`; non-test production consumer: `Tensor::register_hook` at `tensor.rs:933-951` invokes `add_grad_hook`; `Tensor::register_post_accumulate_grad_hook` at `:953-977` invokes `add_post_accumulate_hook`. |
| REQ-6 | SHIPPED | impl: `HookStorage::remove(handle)` at `hooks.rs:116-123`; mirrors `RemovableHandle.remove` at `torch/utils/hooks.py:48-71`; non-test production consumer: `Tensor::remove_hook(handle)` at `ferrotorch-core/src/tensor.rs:979` invokes `HookStorage::remove` — the public deregistration API the user calls when they want to clear a temporary hook (e.g. visualization hooks during training). |
| REQ-7 | SHIPPED | impl: `pub(crate) fn run_grad_hooks` at `run_grad_hooks in hooks.rs`; non-test production consumer: `run_grad_hooks in ferrotorch-core/src/autograd/graph.rs let grad = run_grad_hooks(hooks, grad)?` inside the sequential backward dispatcher and `hooks in graph.rs` inside the parallel dispatcher — every user `loss.backward()` flows through this for any leaf with grad hooks. |
| REQ-8 | SHIPPED | impl: `pub(crate) fn run_post_accumulate_hooks` at `run_post_accumulate_hooks in hooks.rs`; non-test production consumer: `run_post_accumulate_hooks in ferrotorch-core/src/autograd/graph.rs` inside both the sequential and parallel dispatchers. |
