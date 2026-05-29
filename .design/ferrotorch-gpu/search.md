# GPU `searchsorted` / `bucketize` kernel

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - aten/src/ATen/native/cuda/Bucketization.cu
  - aten/src/ATen/native/BucketizationUtils.h
-->

## Summary

`ferrotorch-gpu/src/search.rs` provides the on-device binary-search kernel
that lowers `torch.searchsorted` / `torch.bucketize` for CUDA-resident
tensors. It mirrors `aten/src/ATen/native/cuda/Bucketization.cu`
(`searchsorted_cuda_kernel`, `lower_bound`, `upper_bound`) for the 1-D
boundaries case (`is_1d_boundaries = true`), which is the surface the
`ferrotorch-core` `searchsorted` / `bucketize` ops expose (1-D `boundaries`,
flat `values`, no `sorter`).

Each output element is an int64 insertion index (PyTorch returns
`ScalarType::Long` when `out_int32 == false`, which is the ferrotorch
default). The kernel runs one thread per value; each thread binary-searches
the entire boundary array.

## Boundary / tie semantics (the bug-prone part)

Matched byte-for-byte to upstream:

- `right == false` (PyTorch `side="left"`): `lower_bound` — first index `i`
  with `boundaries[i] >= val`. Upstream condition is `!(mid_val >= val)`
  advancing `start`. A value equal to a boundary lands ON that boundary's
  index (`seq[i-1] < v <= seq[i]`).
- `right == true` (PyTorch `side="right"`): `upper_bound` — first index `i`
  with `boundaries[i] > val`. Upstream condition is `!(mid_val > val)`
  advancing `start`. A value equal to a boundary lands AFTER it
  (`seq[i-1] <= v < seq[i]`).

This is the exact pair of half-open comparisons the CPU `partition_point`
path in `ferrotorch-core/src/ops/search.rs` already uses
(`partition_point(|b| *b < *v)` for left, `partition_point(|b| *b <= *v)` for
right), so the GPU and CPU paths agree bit-for-bit including the tie case
where a value equals a boundary.

## Requirements

- REQ-1: `gpu_searchsorted_f32` / `gpu_searchsorted_f64` — given a device
  `values` buffer of length `n_vals`, a device 1-D `boundaries` buffer of
  length `n_bounds`, and a `right: bool`, return a fresh device
  `CudaSlice<i64>` of length `n_vals` holding the per-value insertion index.
  One thread per value; serial binary search over `[0, n_bounds)`. Empty
  boundaries → every index is 0 (matches `partition_point` on an empty slice
  and upstream `start == end == 0`). Mirrors `searchsorted_cuda_kernel`.
- REQ-2: PTX template + ABI. A single `SEARCHSORTED_F32_PTX` /
  `SEARCHSORTED_F64_PTX` carrying ABI
  `(vals_ptr, bounds_ptr, out_ptr, n_vals, n_bounds, right)` where `right`
  is a `u32` flag (0 = lower_bound, 1 = upper_bound). Loaded via
  `module_cache::get_or_compile`.
- REQ-3: `GpuBackend::searchsorted_1d` trait surface in
  `ferrotorch-core/src/gpu_dispatch.rs` — `(values, boundaries, right)` →
  `DType::I64` `GpuBufferHandle`. Default impl returns `InvalidArgument`;
  the CUDA backend overrides. Output dtype tag is int64 (PyTorch
  `ScalarType::Long`).
- REQ-4: Dispatch wiring in `CudaBackendImpl::searchsorted_1d`
  (`ferrotorch-gpu/src/backend_impl.rs`) — `match dtype { F32, F64 }`,
  wrapping the resulting `CudaSlice<i64>` as an `I64` handle.
- REQ-5: Re-export + non-test production consumer. `pub use
  search::{gpu_searchsorted_f32, gpu_searchsorted_f64}` in
  `ferrotorch-gpu/src/lib.rs`; the production consumer is
  `ferrotorch-core/src/ops/search.rs::searchsorted`, which dispatches CUDA
  f32/f64 inputs through `GpuBackend::searchsorted_1d` and reads back ONLY
  the i64 result indices (the value/boundary data never leaves the device —
  no R-CODE-4 round trip).

## Acceptance Criteria

- [x] AC-1: `gpu_searchsorted_f32([1,3,5,7], [0,2,3,6,8], right=true)` →
  `[0,1,2,3,4]` (matches CPU `test_searchsorted_right`).
- [x] AC-2: `gpu_searchsorted_f32([1,3,5,7], [1,3,5,7], right=false)` →
  `[0,1,2,3]` (value exactly on a boundary, left side → boundary index).
- [x] AC-3: same values, `right=true` → `[1,2,3,4]` (value on a boundary,
  right side → after it). The right-vs-left tie divergence is the
  bug-prone case and is pinned by a dedicated GPU test.
- [x] AC-4: empty boundaries → all-zero output.
- [x] AC-5: the result buffer `is_cuda()` (lives on device); only the
  decoded indices are read to host.

## Architecture

`gpu_searchsorted_f32` (`search.rs`):
1. Validate device ordinals match across `values` and `boundaries`.
2. `n_vals == 0` → empty i64 buffer (no launch).
3. Range-check `n_vals` / `n_bounds` against `u32::MAX`.
4. `module_cache::get_or_compile(ctx, SEARCHSORTED_F32_PTX, ...)`.
5. Alloc `CudaSlice<i64>` of `n_vals`; launch one thread per value.
6. Each thread loops `lo=0, hi=n_bounds`; mid-value compare per `right`
   flag; writes `lo` (the converged insertion point) as `s64`.

`searchsorted_1d` (`backend_impl.rs`): unwrap f32/f64 value+boundary
buffers, call the matching `gpu_searchsorted_*`, wrap result via
`wrap_slice_i64`.

Non-test consumer (`ops/search.rs::searchsorted`): on CUDA f32/f64,
`backend.searchsorted_1d(values.gpu_handle()?, boundaries.gpu_handle()?,
right)?`, then `gpu_to_cpu` the i64 handle, decode 8-byte LE chunks to
`Vec<usize>`. `bucketize` inherits the GPU path through its existing
delegation to `searchsorted`.

## Parity contract

`parity_ops = []` — the `searchsorted` / `bucketize` route family has no
parity-sweep runner arm (a TEST-INFRASTRUCTURE gap tracked separately per
goal.md S5, not a REQ blocker). The numeric contract is byte-for-byte
parity with the CPU `partition_point` path, verified by GPU-vs-CPU tests
including the right=true/false tie cases.

## Verification

`cargo test -p ferrotorch-gpu --features cuda search` exercises the kernel
live on the device (skips cleanly when no GPU is present), asserting both
the index values and that the output buffer is device-resident.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | `pub fn gpu_searchsorted_f32` / `gpu_searchsorted_f64 in search.rs` mirror `lower_bound`/`upper_bound` in `aten/src/ATen/native/cuda/Bucketization.cu:26,44`; consumer `CudaBackendImpl::searchsorted_1d in backend_impl.rs` |
| REQ-2 | SHIPPED | `SEARCHSORTED_F32_PTX` / `SEARCHSORTED_F64_PTX in search.rs` carry the 6-arg ABI; launch site binds args in matching order |
| REQ-3 | SHIPPED | `fn searchsorted_1d in gpu_dispatch.rs` (`GpuBackend` trait method, `DType::I64` output); consumer `ops::search::searchsorted` GPU branch |
| REQ-4 | SHIPPED | `CudaBackendImpl::searchsorted_1d in backend_impl.rs` dispatches `match dtype { F32, F64 }` and wraps via `wrap_slice_i64` |
| REQ-5 | SHIPPED | `pub use search::{gpu_searchsorted_f32, gpu_searchsorted_f64} in lib.rs`; non-test consumer `ferrotorch_core::ops::search::searchsorted` CUDA f32/f64 branch (`searchsorted in ops/search.rs`) |
