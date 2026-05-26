# Host-to-device and device-to-host transfers + zero-init allocators

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - aten/src/ATen/native/cuda/Copy.cu
  - aten/src/ATen/native/Copy.cpp
  - c10/cuda/CUDAFunctions.cpp
  - c10/cuda/CUDACachingAllocator.cpp
-->

## Summary

`ferrotorch-gpu/src/transfer.rs` is the foundational H2D / D2H /
allocate-zeros layer that every GPU kernel and tensor op in
ferrotorch-gpu routes through. Two transfer functions
(`cpu_to_gpu` / `gpu_to_cpu`), one pinned-memory variant
(`cpu_to_gpu_pinned`), and four dtype-specialised zero allocators
(`alloc_zeros<T>` / `alloc_zeros_f32` / `alloc_zeros_f64` /
`alloc_zeros_bf16`) — all integrated with the global allocator
pool in `crate::pool` for sub-microsecond pool hits. Mirrors
PyTorch's `at::native::copy` machinery and the underlying
`c10::cuda::CUDACachingAllocator` pool-then-driver allocation flow.

## Requirements

- REQ-1: `pub fn cpu_to_gpu<T: DeviceRepr>(data: &[T], device:
  &GpuDevice) -> GpuResult<CudaBuffer<T>>` — H2D transfer via
  `device.stream().clone_htod(data)`. Returns an unpooled
  `CudaBuffer` (alloc_len == len; no rounding).
- REQ-2: `pub fn gpu_to_cpu<T: DeviceRepr>(buffer: &CudaBuffer<T>,
  device: &GpuDevice) -> GpuResult<Vec<T>>` — D2H transfer via
  `device.stream().clone_dtoh(buffer.inner())`. Validates the
  buffer's device ordinal matches; truncates the result to the
  logical `len` (pool-rounded buffers have alloc_len > len).
- REQ-3: `pub fn alloc_zeros_f32(len, device) ->
  GpuResult<CudaBuffer<f32>>` — pool-aware zero-allocation. On a
  pool hit, reuses a cached `CudaSlice` with one `cuMemsetD8Async`;
  on a miss, allocates with the rounded length so the pool key
  matches on return.
- REQ-4: `pub fn alloc_zeros_f64` and (cuda-only)
  `pub fn alloc_zeros_bf16` — the f64 and bf16-as-u16 siblings of
  `alloc_zeros_f32`, with the same pool-aware shape.
- REQ-5: `pub fn alloc_zeros<T: DeviceRepr>(len, device) ->
  GpuResult<CudaBuffer<T>>` — generic dtype zero-alloc, used by
  kernels whose value-type is not known until method-dispatch time.
- REQ-6: `pub fn cpu_to_gpu_pinned<T: DeviceRepr>` — H2D via
  page-locked host memory for faster transfer on hosts that
  support it. Mirrors PyTorch's `pin_memory=True` path in
  DataLoader / tensor uploads.
- REQ-7: Non-CUDA stubs at lines 245-279 — every public symbol has
  a stub returning `GpuError::DeviceUnavailable` so dependent
  crates compile cleanly without `cuda`.
- REQ-8: Non-test production consumer wiring — 200+ call sites
  across `ferrotorch-gpu/src/`, `ferrotorch-core/src/` and the
  rest of the workspace.

## Acceptance Criteria

- [x] AC-1: `pub fn cpu_to_gpu` at line 19.
- [x] AC-2: `pub fn gpu_to_cpu` at line 40 with device-ordinal
  validation and result-truncation.
- [x] AC-3: `pub fn alloc_zeros_f32` at line 69 with the
  pool-then-driver flow (uses `crate::pool::round_len` +
  `crate::pool::pool_take` + `device.stream().alloc_zeros`).
- [x] AC-4: `pub fn alloc_zeros_f64` at line 141 and
  `pub fn alloc_zeros_bf16` at line 124 (cuda-only).
- [x] AC-5: `pub fn alloc_zeros<T>` at line 169.
- [x] AC-6: `pub fn cpu_to_gpu_pinned` at line 196 (cuda-only) /
  line 245 (non-cuda stub).
- [x] AC-7: Six non-CUDA stubs at lines 245-279 covering every
  cuda-feature `pub fn`.
- [x] AC-8: 200+ non-test consumer call sites verified via
  `grep -rn "transfer::cpu_to_gpu|transfer::gpu_to_cpu|transfer::alloc_zeros" ferrotorch-gpu/src ferrotorch-core/src`.

## Architecture

### Transfer functions (REQ-1, REQ-2, REQ-6)

`cpu_to_gpu<T>` (line 19):
1. Uploads via `device.stream().clone_htod(data)` — cudarc's H2D
   primitive that blocks on the stream until the copy completes.
2. Wraps the resulting `CudaSlice<T>` in a `CudaBuffer<T>` with
   `pool_fn = None` (unpooled — the buffer is freed via cudarc's
   `Drop` directly when the `CudaBuffer` is dropped).

`gpu_to_cpu<T>` (line 40):
1. Validates `buffer.device_ordinal() == device.ordinal()` —
   prevents a cross-device readback.
2. Downloads via `device.stream().clone_dtoh(buffer.inner())`.
3. Truncates the result vector to `buffer.len()` so pool-rounded
   buffers don't expose padding bytes to the caller.

`cpu_to_gpu_pinned<T>` (line 196): uses cudarc's pinned-memory
host allocation, then memcpy from the pinned region. Faster on
hosts where the OS supports `cudaHostAllocPortable`. Documented
at the upstream `aten/src/ATen/native/Copy.cpp:non_blocking` path.

### Pool-aware zero allocators (REQ-3, REQ-4, REQ-5)

`alloc_zeros_f32` (line 69) is the canonical shape:

1. Compute `rounded = crate::pool::round_len(len)` — the
   pool-size bucket (round up to the next power of 2 or similar).
2. Try `crate::pool::pool_take::<CudaSlice<f32>>(device.ordinal(),
   rounded, 4)` — checks the global allocator pool for a cached
   slice. On hit, reuses the existing slice (with its CUDA events
   still alive), `memset_zeros` it to clear stale data, wrap into
   a `CudaBuffer::new_pooled` and return.
3. On miss, allocate fresh via `device.stream().alloc_zeros::<f32>(rounded)`
   — using the rounded length so the slice's allocation size
   matches what `pool_take` will look for on a subsequent return.
4. Wrap into `CudaBuffer::new_pooled` and return.

The `memset_zeros` on the full rounded allocation (not just the
logical `len`) is intentional — it ensures no stale data from
previous uses leaks into the padding region (documented at
line 67 as "P10: intentional").

`alloc_zeros_f64` (line 141) is the f64 sibling with `dtype_bytes = 8`.

`alloc_zeros_bf16` (line 124, cuda-only) returns a `CudaSlice<u16>`
holding the bf16 bit-patterns.

`alloc_zeros<T: DeviceRepr>` (line 169) is the generic shape for
kernels whose value-type is determined by trait-dispatch (e.g.
`gather_int`'s value-type-erased return).

### Non-CUDA stubs (REQ-7)

Lines 245-279 (gated `#[cfg(not(feature = "cuda"))]`) re-define
each public symbol as a stub returning `GpuError::DeviceUnavailable`.
This lets the public API of ferrotorch-gpu compile cleanly without
CUDA, so downstream crates don't need conditional `use` imports.

### Non-test production consumers (REQ-8)

Almost every kernel-launcher file in `ferrotorch-gpu/src/` uses
this module:

- `kernels.rs` — `alloc_zeros_f32` / `alloc_zeros_f64` for every
  output buffer (hundreds of call sites).
- `roll.rs` — `alloc_zeros_f32(total, device)`.
- `group_norm.rs` — `alloc_zeros_f32(n, device)`.
- `flash_attention.rs`, `conv.rs`, `upsample.rs`, `blas.rs`,
  `cufft.rs`, `cusolver.rs`, `sparse.rs` — every dispatch
  allocates output via this module.
- `tensor_bridge.rs` — `cpu_to_gpu` and `gpu_to_cpu` are the
  primitives behind `tensor_to_gpu` / `tensor_to_cpu`.
- `backend_impl.rs` — every trait method that produces a new
  GPU buffer routes its allocation through `alloc_zeros*`.

ferrotorch-core's `Tensor::cuda()` and the dispatch through
`GpuBackend::cpu_to_gpu` / `gpu_to_cpu` trait methods (see
`impl GpuBackend for CudaBackend` in `backend_impl.rs`) ultimately
call into these free functions. Cross-crate consumers:
`pub fn tensor_to_gpu` in `ferrotorch-distributed/src/ucc_backend.rs`
uses `tensor_to_gpu` which internally uses `transfer::cpu_to_gpu`.

Total non-test call sites: 200+ across the workspace (counted via
`grep -rn "transfer::cpu_to_gpu|transfer::gpu_to_cpu|transfer::alloc_zeros" ferrotorch-gpu/src ferrotorch-core/src`).

## Parity contract

`parity_ops = []` for this route. Transfer is INFRASTRUCTURE — it
preserves byte-exact equivalence between the host and device sides
of every copy (cudarc's `clone_htod` / `clone_dtoh` are
`cudaMemcpy`-backed and the contract is byte-for-byte).

Edge cases preserved:

- **Empty transfer** (`data.len() == 0`): `cudaMemcpy` with size 0
  is a no-op; cudarc's `clone_htod` / `clone_dtoh` handle this
  cleanly. `alloc_zeros_*(0, device)` similarly returns an empty
  buffer without allocation.
- **Cross-device readback**: `gpu_to_cpu` validates the buffer's
  ordinal matches the provided device; mismatch returns
  `GpuError::DeviceMismatch`.
- **Pool-rounded length**: `alloc_zeros_*` allocates `rounded`
  elements but `CudaBuffer::len()` reports the logical `len`;
  `gpu_to_cpu` truncates to `len` so callers only see the
  meaningful data.
- **Stream synchronisation**: all transfers use the device's
  default stream; the `clone_htod` / `clone_dtoh` calls block
  until the copy completes, matching PyTorch's blocking-default
  behaviour. (Pinned-memory variant uses non-blocking transfers
  internally but still blocks until completion before returning.)
- **Zero-init contract**: `alloc_zeros_*` always returns a
  zero-filled buffer, even on pool hits (the `memset_zeros` after
  pool reuse clears any stale data from a previous user).

## Verification

Unit tests in `ferrotorch-gpu/src/transfer.rs` (gated `#[cfg(test)]
#[cfg(feature = "cuda")]`) exercise: H2D + D2H round-trip
preservation, device-ordinal mismatch rejection, pool hit / miss
zero-init, the pinned-memory path, and the empty-transfer corner.

Cross-cutting integration is implicit: every other test in the
crate that touches a `CudaBuffer<T>` indirectly exercises this
module.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-gpu --features cuda transfer:: 2>&1 | tail -3
```

Expected: ≥ 1 `test result: ok` line.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn cpu_to_gpu` in `transfer.rs`; non-test consumer: `pub fn tensor_to_gpu` in `tensor_bridge.rs` calls it; `impl GpuBackend::cpu_to_gpu for CudaBackend` in `backend_impl.rs` wraps it. |
| REQ-2 | SHIPPED | impl: `pub fn gpu_to_cpu` in `transfer.rs`; non-test consumer: `pub fn tensor_to_cpu` in `tensor_bridge.rs` calls it; `impl GpuBackend::gpu_to_cpu for CudaBackend` in `backend_impl.rs` wraps it. |
| REQ-3 | SHIPPED | impl: `pub fn alloc_zeros_f32` in `transfer.rs`; non-test consumer: every f32 kernel-launcher in the crate calls it (e.g. `pub fn gpu_roll_f32` in `roll.rs`, `pub fn gpu_group_norm_f32` in `group_norm.rs`, kernel launchers in `kernels.rs`). |
| REQ-4 | SHIPPED | impl: `pub fn alloc_zeros_f64` and `pub fn alloc_zeros_bf16` in `transfer.rs`; non-test consumer: f64 / bf16 kernel-launchers in `kernels.rs` and `bf16.rs`. |
| REQ-5 | SHIPPED | impl: `pub fn alloc_zeros<T>` in `transfer.rs`; non-test consumer: `impl GpuBackend::alloc_zeros for CudaBackend` in `backend_impl.rs` dispatches to it for dtype-generic allocations; re-exported via `pub use transfer::*` in `lib.rs`. |
| REQ-6 | SHIPPED | impl: `pub fn cpu_to_gpu_pinned` in `transfer.rs`; non-test consumer: `impl GpuBackend::cpu_to_gpu_pinned for CudaBackend` in `backend_impl.rs` wraps it; ferrotorch-data's DataLoader pinned-memory path uses it through that trait method. |
| REQ-7 | SHIPPED | impl: 6 non-CUDA stubs in `transfer.rs` (`cpu_to_gpu_pinned`, `cpu_to_gpu`, `gpu_to_cpu`, `alloc_zeros`, `alloc_zeros_f32`, `alloc_zeros_f64`); non-test consumer: workspace `--no-default-features` CI lane compiles cleanly. |
| REQ-8 | SHIPPED | impl: 200+ non-test call sites across the workspace (verified via grep of `transfer::cpu_to_gpu|transfer::gpu_to_cpu|transfer::alloc_zeros` in ferrotorch-gpu/src + ferrotorch-core/src); representative production consumers cited in REQs 1-6 evidence rows. |
