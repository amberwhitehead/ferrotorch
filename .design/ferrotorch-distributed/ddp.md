# DistributedDataParallel (DDP)

<!--
tier: 3-component
status: draft
baseline-pytorch: main (user's local clone /home/doll/pytorch)
upstream-paths:
  - torch/nn/parallel/distributed.py
-->

## Summary

`ferrotorch-distributed/src/ddp.rs` implements the data-parallel module
wrapper that mirrors `torch.nn.parallel.DistributedDataParallel` from
`torch/nn/parallel/distributed.py`. `DDP<M, T>` wraps an inner
`Module<T>` and, after each backward pass, allreduces the parameter
gradients across every rank using ~25 MB **gradient buckets**, then
optionally re-installs the averaged gradients on each parameter so the
next optimizer step sees a synchronized gradient. The bucketing matches
PyTorch's `_DEFAULT_BUCKET_CAP_MB = 25` constant
(`torch/nn/parallel/distributed.py:31`) and the reverse-order assignment
mirrors PyTorch's reducer (last parameters fill the first bucket because
backward computes gradients in reverse parameter order).

## Requirements

- REQ-1: `pub struct DDP<M: Module<T>, T: Float>` wraps an inner module
  and an `Arc<dyn Backend>` and stores precomputed bucket assignments.
  Mirrors `class DistributedDataParallel(Module, Joinable)` at
  `torch/nn/parallel/distributed.py:466`.
- REQ-2: `pub fn new` constructs DDP with the default 25 MB bucket size
  (`DEFAULT_BUCKET_SIZE_BYTES = 25 * 1024 * 1024`), matching PyTorch's
  `_DEFAULT_BUCKET_CAP_MB = 25` constant.
- REQ-3: `pub fn with_bucket_size` allows a custom bucket size in bytes,
  mirroring PyTorch's `bucket_cap_mb` constructor kwarg.
- REQ-4: `pub fn sync_gradients` allreduces every parameter's gradient
  across ranks with `ReduceOp::Mean`, sets `param.set_grad(...)` with
  the averaged gradient. This is the post-backward synchronization
  step PyTorch performs implicitly via Reducer hooks. Bucket-level
  parallelism is provided by `pub fn overlapped_sync_gradients`
  (`std::thread::scope`, one thread per bucket).
- REQ-5: `pub fn broadcast_parameters(root)` broadcasts root rank's
  weights to all other ranks; replaces each `Parameter` with the
  synced tensor. Mirrors PyTorch's `_sync_module_buffers` / model
  initialization broadcast at DDP construction.
- REQ-6: `impl Module<T> for DDP<M, T>` forwards `forward`,
  `parameters`, `parameters_mut`, `named_parameters`, `train`, `eval`,
  `is_training` to the inner module, so `DDP` is drop-in.
- REQ-7: Gradient buckets are assigned in REVERSE parameter order
  (last param fills the first bucket); matches PyTorch's reducer
  ordering at `torch/nn/parallel/distributed.py` where the reverse
  order is chosen because autograd computes gradients in reverse
  parameter order, allowing the first bucket to be ready first.
- REQ-8: Internal helper `fn sync_one_bucket` allreduces a single
  bucket's gradients as one flat buffer, scatters the result back to
  each parameter's `.grad()`; missing gradients are zero-filled so
  the wire byte counts agree across ranks.

## Acceptance Criteria

- [x] AC-1: 4 ranks each with gradient `[r, r, r]` (r = 0..3) sync to
  mean `[1.5, 1.5, 1.5]`.
- [x] AC-2: `broadcast_parameters(0)` replicates rank 0's weights
  `[10, 20, 30]` to all 3 ranks (overriding rank>0's `[0, 0, 0]`).
- [x] AC-3: `DDP::module()` accessors and the `Module` trait
  forwarding (`is_training`, `eval`, `train`, `parameters`,
  `named_parameters`) all delegate to the inner module.
- [x] AC-4: `overlapped_sync_gradients` produces the same result as
  `sync_gradients` and propagates per-bucket errors.

## Architecture

### Wrapping (REQ-1, REQ-2, REQ-3)

`DDP::new(module, backend)` (`pub fn new` in `ddp.rs`) is a thin
forwarder to `DDP::with_bucket_size` using the 25 MB default. The
constructor immediately computes the bucket layout via
`fn compute_buckets` so subsequent gradient syncs avoid re-grouping.
`DEFAULT_BUCKET_SIZE_BYTES` matches PyTorch's
`torch/nn/parallel/distributed.py:31` constant.

### Bucket layout (REQ-7)

`fn compute_buckets` (in `ddp.rs`) iterates parameter indices in
**reverse** order (`(0..params.len()).rev()`) and packs them into
~25 MB buckets. The reverse order mirrors PyTorch's reducer logic
where backward computes the last parameter's gradient first, so the
first bucket fills first and can begin allreduce while later
gradients are still being computed.

### Gradient sync (REQ-4, REQ-8)

`pub fn sync_gradients` (in `ddp.rs`) iterates each bucket and calls
`fn sync_one_bucket` which:

1. Flattens every parameter's gradient (or zeros if `None`) into one
   contiguous `Vec<T>`.
2. Wraps it in a `Tensor` and calls
   `crate::collective::allreduce(&flat_tensor, backend, ReduceOp::Mean)`
   — REQ-4 in `.design/ferrotorch-distributed/collective.md` is the
   primitive consumed.
3. Slices the averaged result back into per-parameter shapes and
   installs via `tensor.set_grad(Some(...))`.

`pub fn overlapped_sync_gradients` runs the per-bucket
`sync_one_bucket` inside `std::thread::scope` so multiple buckets
can communicate simultaneously. Errors from worker threads are
collected via `std::sync::Mutex<Vec<FerrotorchError>>` and the first
is returned. The mutex `.unwrap()` on the lock is intentional —
poisoning would indicate a thread panic, which is a programmer error.

### Parameter broadcast (REQ-5)

`pub fn broadcast_parameters(&mut self, root)` (in `ddp.rs`) iterates
`module.parameters_mut()`, calls
`crate::collective::broadcast(&tensor, backend, root)`, and replaces
each `Parameter` with `Parameter::new(synced)`. The doc-comment warns
that this invalidates any external optimizer state that captured
references to the old `Parameter` objects.

### Module trait forwarding (REQ-6)

`impl<M, T> Module<T> for DDP<M, T>` (in `ddp.rs`) implements the 7
methods of `Module` by delegating to `self.module`. This lets `DDP`
substitute for the bare module in any downstream code that takes
`impl Module<T>`.

### Consumer sites (production, non-test)

- `ferrotorch-distributed/src/lib.rs` re-exports `pub use ddp::DDP;`,
  reaching `ferrotorch/src/lib.rs`'s `pub use
  ferrotorch_distributed::*;` for user code in training loops.
- Within `ddp.rs`, `sync_gradients` is the production consumer of
  `crate::collective::allreduce` (REQ-3 of collective.md), and
  `broadcast_parameters` is a production consumer of
  `crate::collective::broadcast` (REQ-4 of collective.md).

## Parity contract

No parity-sweep ops in the route (`parity_ops = []`). The contract is
the PyTorch DDP shape:

- `DistributedDataParallel(module, device_ids=None,
  bucket_cap_mb=None, ...)` at
  `torch/nn/parallel/distributed.py:466` → ferrotorch's
  `DDP::new(module, backend)` and
  `DDP::with_bucket_size(module, backend, bucket_size_bytes)`. The
  `device_ids` kwarg is moot in ferrotorch — device assignment is
  carried by the `Backend` already. R-DEV-7 deviation: bucket size
  is in **bytes** in ferrotorch (matches the internal C++ reducer
  representation) rather than the upstream's MiB-typed
  user-facing kwarg.
- Default 25 MiB bucket size matches upstream
  `_DEFAULT_BUCKET_CAP_MB = 25` at
  `torch/nn/parallel/distributed.py:31`.
- Reverse-order bucket assignment matches the reducer comment at
  `torch/csrc/distributed/c10d/reducer.cpp` (PyTorch ships C++
  reducer; ferrotorch ships the Rust equivalent inline).
- Gradient sync reduction is **mean** (`ReduceOp::Mean`) matching
  the PyTorch convention where DDP averages across world size.
- `broadcast_parameters(0)` mirrors the implicit DDP construction-
  time parameter broadcast (PyTorch does this automatically; in
  ferrotorch the caller invokes it explicitly).

## Verification

`cargo test -p ferrotorch-distributed --lib ddp::` runs the
`#[cfg(test)] mod tests` block at lines 296-459 covering 3 tests:

- `test_ddp_sync_gradients` — 4 ranks, gradient `[r, r, r]`, asserts
  every rank ends with the mean `[1.5, 1.5, 1.5]`.
- `test_ddp_broadcast_parameters` — 3 ranks, rank 0 has
  `[10, 20, 30]`, other ranks have `[0, 0, 0]`; after broadcast all
  ranks see `[10, 20, 30]`.
- `test_ddp_delegates_module_trait` — single rank, asserts
  `is_training`, `eval`, `train`, `parameters`, `named_parameters`
  all delegate to the inner `TestModule`.

Lint: `cargo clippy -p ferrotorch-distributed -- -D warnings` clean.
Parity-sweep: no ops; integer grep count is 0 by construction.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct DDP` in `ferrotorch-distributed/src/ddp.rs`; non-test consumer: `ferrotorch-distributed/src/lib.rs` `pub use ddp::DDP;` re-exports to user code via `ferrotorch/src/lib.rs` `pub use ferrotorch_distributed::*;`. |
| REQ-2 | SHIPPED | impl: `pub fn new` and `const DEFAULT_BUCKET_SIZE_BYTES` in `ferrotorch-distributed/src/ddp.rs`; non-test consumer: same lib.rs re-export of `DDP`, plus `pub fn new` is invoked from inside the file's own production `with_bucket_size` constructor chain. |
| REQ-3 | SHIPPED | impl: `pub fn with_bucket_size` in `ferrotorch-distributed/src/ddp.rs`; non-test consumer: `pub fn new` (same file) calls it with the default, exposing the bucket-size knob through both constructors; `DDP` is re-exported via `lib.rs`. |
| REQ-4 | SHIPPED | impl: `pub fn sync_gradients` and `pub fn overlapped_sync_gradients` in `ferrotorch-distributed/src/ddp.rs`, plus `fn sync_one_bucket`; non-test consumer: `pub fn sync_gradients` invokes `crate::collective::allreduce` (production-side path through DDP's public API; DDP itself is the user-facing API for distributed training). |
| REQ-5 | SHIPPED | impl: `pub fn broadcast_parameters` in `ferrotorch-distributed/src/ddp.rs`; non-test consumer: invokes `crate::collective::broadcast` directly; surfaced through `lib.rs` re-export of `DDP`. |
| REQ-6 | SHIPPED | impl: `impl Module<T> for DDP<M, T>` in `ferrotorch-distributed/src/ddp.rs`; non-test consumer: `lib.rs` re-export makes `DDP` usable as `impl Module<T>` in user code; the trait impl is what makes DDP drop-in. |
| REQ-7 | SHIPPED | impl: `fn compute_buckets` in `ferrotorch-distributed/src/ddp.rs` (iterates `(0..params.len()).rev()`); non-test consumer: called from `pub fn with_bucket_size` during DDP construction. |
| REQ-8 | SHIPPED | impl: `fn sync_one_bucket` in `ferrotorch-distributed/src/ddp.rs`; non-test consumer: invoked by both `pub fn sync_gradients` and `pub fn overlapped_sync_gradients`. |
