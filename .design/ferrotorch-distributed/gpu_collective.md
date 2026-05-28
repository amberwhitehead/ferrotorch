# gpu_collective — GPU-aware collective dispatch with NCCL fast path

<!--
tier: 3-component
status: draft
baseline-pytorch: main (user's local clone /home/doll/pytorch)
upstream-paths:
  - torch/distributed/distributed_c10d.py
  - torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp
  - torch/csrc/distributed/c10d/ProcessGroupGloo.cpp
-->

## Summary

`ferrotorch-distributed/src/gpu_collective.rs` is the public
GPU-aware collective layer that dispatches between three paths:
(1) NCCL fast path when the `nccl` feature is on AND the backend
downcasts to `NcclBackend` via `Backend::as_nccl_backend()`,
(2) opt-in CPU host-round-trip "Gloo-equivalent slow path" when
`FERROTORCH_ENABLE_GPU_FALLBACK=1` is set in the environment
(emits a `tracing::warn!` per call), (3) error by default (PyTorch
parity — CUDA tensors passed to a non-NCCL collective raise
`RuntimeError` rather than silently host-round-trip). Mirrors the
role of PyTorch's `ProcessGroupNCCL::allreduce` /
`ProcessGroupGloo::allreduce` dispatch: NCCL backend → GPU-native;
Gloo backend → host round-trip with `c10d` `DeviceMismatch`
warning.

## Requirements

- REQ-1: `pub fn gpu_allreduce<T: GpuFloat>(tensor: &GpuTensor<T>,
  backend: &dyn Backend, op: ReduceOp) -> FerrotorchResult<GpuTensor<T>>`
  dispatches in priority order: (1) NCCL fast path if available,
  (2) CPU host round-trip if `FERROTORCH_ENABLE_GPU_FALLBACK=1`,
  (3) `Err(UnsupportedOp)` otherwise.
- REQ-2: `pub fn gpu_broadcast<T: GpuFloat>(tensor: &GpuTensor<T>,
  backend: &dyn Backend, root: usize) -> FerrotorchResult<GpuTensor<T>>`
  has the same dispatch order. Additionally rejects `root >=
  world_size` with `InvalidRank`.
- REQ-3: NCCL fast-path (`fn nccl_path_allreduce`, `fn
  nccl_path_broadcast`, gated `#[cfg(feature = "nccl")]`):
  1. Map `T` to `NcclDataType` via `fn nccl_dtype_of` (f32 →
     Float32, f64 → Float64, anything else → `UnsupportedOp`).
  2. D2D-clone the input tensor via `GpuTensor::try_clone` so
     the operation is non-destructive.
  3. Extract the raw CUDA device pointer via
     `GpuTensor::cu_device_ptr`.
  4. Call `NcclBackend::{allreduce_raw, broadcast_raw}` in-place
     on the clone's pointer.
  5. Synchronise the NCCL stream via `NcclBackend::synchronize`
     before returning so subsequent CPU reads are race-free.
- REQ-4: CPU fallback path (`fn cpu_path_allreduce`,
  `fn cpu_path_broadcast`):
  1. `tensor_to_cpu(tensor)` (device → host).
  2. Call the CPU collective (`crate::collective::allreduce` or
     `broadcast`) on the host tensor.
  3. `tensor_to_gpu(...)` (host → device on the same device the
     input came from).
- REQ-5: Fallback emits a `tracing::warn!` per call with
  `target = "ferrotorch::gpu_fallback"`, naming the collective
  and pointing the user at `FERROTORCH_ENABLE_GPU_FALLBACK`
  un-setting to make the call error instead.
- REQ-6: PyTorch parity default (no NCCL feature AND no
  `FERROTORCH_ENABLE_GPU_FALLBACK`): return
  `DistributedError::UnsupportedOp` with a message naming both
  remediation paths (`nccl` feature or env-var).
- REQ-7: Module is gated by `#[cfg(feature = "gpu")]` in
  `lib.rs` line 200.

## Acceptance Criteria

- [x] AC-1: Module compiles only with `--features=gpu`
  (`lib.rs` line 200 gate).
- [x] AC-2: `pub fn gpu_allreduce` and `pub fn gpu_broadcast`
  exist; signatures match REQ-1 / REQ-2.
- [x] AC-3: `Backend::as_nccl_backend` is invoked first (when
  the `nccl` feature is on) — verified by code inspection
  inside the `#[cfg(feature = "nccl")] if let Some(nccl) = ...`
  block.
- [x] AC-4: `std::env::var("FERROTORCH_ENABLE_GPU_FALLBACK").is_ok()`
  is the env-var check; fallback emits `tracing::warn!` with
  `target = "ferrotorch::gpu_fallback"`.
- [x] AC-5: Default-error message in `UnsupportedOp` names
  both the `nccl` feature and `FERROTORCH_ENABLE_GPU_FALLBACK=1`.
- [x] AC-6: `fn nccl_path_allreduce` / `_broadcast` D2D-clone
  the input (no in-place mutation of caller's tensor) and
  call `nccl.synchronize()` before returning.

## Architecture

The dispatch order in `pub fn gpu_allreduce` is:

```rust
#[cfg(feature = "nccl")]
if let Some(nccl) = backend.as_nccl_backend() {
    return nccl_path_allreduce(tensor, nccl, op);
}

if std::env::var("FERROTORCH_ENABLE_GPU_FALLBACK").is_ok() {
    tracing::warn!(target: "ferrotorch::gpu_fallback", ...);
    return cpu_path_allreduce(tensor, backend, op);
}

Err(DistributedError::UnsupportedOp { ... }.into())
```

The `Backend::as_nccl_backend` downcast hook returns `Some` ONLY
when `backend` is an `NcclBackend` (the trait's default impl
returns `None`; only `NcclBackend` and `HybridBackend` could
override — `HybridBackend` does NOT override since its
`as_nccl_backend` inherits the default `None`, so callers reach
the NCCL fast path by passing the inner `nccl: &NcclBackend` via
`HybridBackend::nccl()`). The hook is the one place the type-
erased trait dispatch can recover the NCCL-typed surface.

`fn nccl_dtype_of` uses `TypeId::of::<T>()` to discriminate f32
vs f64. The `GpuFloat` trait covers more types (f16, bf16) but
NCCL's dtype mapping only wires f32/f64 in this layer; other
dtypes return `UnsupportedOp` naming the dtype.

`fn nccl_path_allreduce` D2D-clones the input via
`GpuTensor::try_clone` (which performs a CUDA D2D copy with no
PCIe round-trip) so the input tensor is unmutated and the
output owns its storage. The `out.cu_device_ptr()` raw pointer
is used as BOTH `sendbuf` and `recvbuf` for the in-place NCCL
allreduce (which NCCL explicitly allows). After the NCCL call
returns (enqueues on `nccl.stream`), `nccl.synchronize()` blocks
until the kernel completes — necessary so subsequent CPU reads
via `out.cpu()` on the default compute stream don't race the
NCCL kernel.

`fn nccl_path_broadcast` mirrors the allreduce shape, plus a
`root: usize` bounds-check against `nccl.world_size()` (returns
`InvalidRank` on overflow OR on the `usize → i32` conversion
overflow path).

`fn cpu_path_allreduce` / `_broadcast` are the slow-path
helpers. They go through `tensor_to_cpu(tensor)` (D2H copy via
the GPU bridge), then the CPU `collective::allreduce` /
`broadcast`, then `tensor_to_gpu(&result, tensor.device())`
(H2D copy back to the same device).

### Consumer sites (production, non-test)

- `ferrotorch-distributed/src/lib.rs` — `#[cfg(feature =
  "gpu")] pub use gpu_collective::{gpu_allreduce, gpu_broadcast};`
  at line 238.
- `ferrotorch/src/lib.rs` — meta-crate re-export reaches
  every workspace user.
- `ferrotorch-distributed/src/ucc_backend.rs` —
  `UccBackend::gpu_allreduce` and `gpu_broadcast` invoke
  `gpu_collective::gpu_allreduce` / `gpu_broadcast` when the
  NCCL communicator is attached.

## Parity contract

No parity-sweep ops. The contract is the
`ProcessGroupNCCL`/`ProcessGroupGloo` dispatch:

- NCCL fast path ↔ `ProcessGroupNCCL::allreduce` on a CUDA
  tensor.
- CPU fallback path ↔ `ProcessGroupGloo::allreduce` on a CUDA
  tensor (PyTorch logs a "performance warning"; we use
  `tracing::warn!`).
- Default error ↔ PyTorch raises `RuntimeError` when a CUDA
  tensor reaches a non-CUDA-capable collective.

## Verification

`cargo test -p ferrotorch-distributed --features gpu --lib`
runs the in-file `tests` module. Six tests are `#[ignore]`-gated
because they were authored before the opt-in fallback default
and need either `FERROTORCH_ENABLE_GPU_FALLBACK=1` in the harness
or the real NCCL setup. The remaining tests under
`#[cfg(feature = "nccl")]` are also `#[ignore]`-gated (require
NCCL + a CUDA device).

The structural correctness of the dispatch wiring is verified at
compile time — the `#[cfg(feature = "nccl")]` block is present,
the `as_nccl_backend()` call site is present, and the env-var
fallback is present.

No parity-sweep ops; integer grep count is 0 by construction.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn gpu_allreduce` in `ferrotorch-distributed/src/gpu_collective.rs`; non-test consumer: `#[cfg(feature = "gpu")] pub use gpu_collective::{gpu_allreduce, gpu_broadcast};` in `ferrotorch-distributed/src/lib.rs` line 238; `ferrotorch-distributed/src/ucc_backend.rs` `UccBackend::gpu_allreduce` invokes `crate::gpu_collective::gpu_allreduce`. |
| REQ-2 | SHIPPED | impl: `pub fn gpu_broadcast` in `ferrotorch-distributed/src/gpu_collective.rs`; non-test consumer: `lib.rs` line 238 re-export; `ferrotorch-distributed/src/ucc_backend.rs` `UccBackend::gpu_broadcast` invokes it. |
| REQ-3 | SHIPPED | impl: `fn nccl_path_allreduce` and `fn nccl_path_broadcast` (both `#[cfg(feature = "nccl")]`) in `ferrotorch-distributed/src/gpu_collective.rs`; non-test consumer: `pub fn gpu_allreduce` / `pub fn gpu_broadcast` (same file) invoke them when `backend.as_nccl_backend()` returns `Some`. |
| REQ-4 | SHIPPED | impl: `fn cpu_path_allreduce` and `fn cpu_path_broadcast` in `ferrotorch-distributed/src/gpu_collective.rs`; non-test consumer: `pub fn gpu_allreduce` / `pub fn gpu_broadcast` (same file) invoke them when the env-var fallback is enabled. |
| REQ-5 | SHIPPED | impl: `tracing::warn!(target: "ferrotorch::gpu_fallback", collective = "allreduce" / "broadcast", ...)` inside `pub fn gpu_allreduce` and `pub fn gpu_broadcast` in `ferrotorch-distributed/src/gpu_collective.rs`; non-test consumer: implicit — every host-round-trip dispatch through these public fns emits the warn. |
| REQ-6 | SHIPPED | impl: `Err(DistributedError::UnsupportedOp { message: "gpu_allreduce requires the `nccl` feature ... Set FERROTORCH_ENABLE_GPU_FALLBACK=1 ..." })` in `pub fn gpu_allreduce` and `pub fn gpu_broadcast` in `ferrotorch-distributed/src/gpu_collective.rs`; non-test consumer: matches PyTorch parity behaviour; user-observable through the public dispatch. |
| REQ-7 | SHIPPED | impl: `#[cfg(feature = "gpu")] pub mod gpu_collective;` in `ferrotorch-distributed/src/lib.rs` line 200; non-test consumer: the gate prevents the module from being compiled on default builds where `GpuTensor` is not available; feature-on builds wire it through `lib.rs` line 238. |
