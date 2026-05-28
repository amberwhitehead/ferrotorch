# cuSPARSE sparse primitives (SpMM, CSR ↔ CSC ↔ COO ↔ dense)

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - aten/src/ATen/native/sparse/cuda/SparseBlasImpl.cpp
  - aten/src/ATen/native/sparse/cuda/SparseBlasImpl.h
  - aten/src/ATen/native/sparse/cuda/SparseBlasLegacy.cpp
  - aten/src/ATen/native/sparse/cuda/SparseCUDABlas.cpp
  - aten/src/ATen/cuda/CUDASparse.h
  - aten/src/ATen/cuda/CUDASparseDescriptors.cpp
-->

## Summary

`ferrotorch-gpu/src/sparse.rs` is the cuSPARSE-backed sparse-primitive
layer used by `ferrotorch-core::sparse::SparseTensor`. It mirrors
PyTorch's `torch.sparse.mm`, `torch.Tensor.to_dense`, and
`torch.Tensor.to_sparse` GPU dispatch paths — all of which run on
cuSPARSE when the input lives on CUDA (per
`aten/src/ATen/native/sparse/cuda/SparseBlasImpl.cpp::spmm`).

The module covers:

- **SpMM** (sparse-dense matmul) for f32 / f64.
- **Sparse↔dense conversions** for CSR (both directions, f32 / f64).
- **Sparse-format conversions**: CSC↔dense, CSR↔CSC, COO↔CSR.

ferrotorch's `SparseTensor` stores indices/values on the host (COO);
the wrappers coalesce on the host, build a CSR triple, upload to
device, and dispatch `cusparseSpMM` with `CUSPARSE_SPMM_ALG_DEFAULT`.
The dense operand is already on device.

Handles are expensive to create; `CusparseHandle` is the RAII wrapper
that `CudaBackendImpl` caches one-per-device via `OnceLock`. The
handle is bound to the device's default stream via `cusparseSetStream`
before each call so subsequent `cusparseSpMM` enqueues on the same
stream as cuBLAS / kernel launches.

## Requirements

- REQ-1: `CusparseHandle` RAII wrapper — `pub struct CusparseHandle`
  (line 48) + `impl Drop` (line 81) that calls `cusparseDestroy`.
  Mirrors `aten/src/ATen/cuda/CuSparseHandlePool.cpp`'s
  thread-local handle reuse pattern (R-DEV-4: RAII replaces the
  manual destroy site).
- REQ-2: SpMM (CSR × dense → dense) — `pub fn gpu_spmm_csr_f32` /
  `gpu_spmm_csr_f64` dispatch `cusparseSpMM` with
  `CUSPARSE_SPMM_ALG_DEFAULT`. Mirrors `spmm` at
  `aten/src/ATen/native/sparse/cuda/SparseBlasImpl.cpp:528`.
- REQ-3: Sparse-to-dense (CSR → dense) — `pub fn
  gpu_sparse_to_dense_csr_f32` / `gpu_sparse_to_dense_csr_f64`.
  Used by `SparseTensor::to_dense_on`.
- REQ-4: Dense-to-sparse (dense → CSR) — `pub fn
  gpu_dense_to_sparse_csr_f32` / `gpu_dense_to_sparse_csr_f64`. Used
  by `SparseTensor::from_dense`.
- REQ-5: CSC-to-dense conversion — `pub fn gpu_csc_to_dense_f32` /
  `gpu_csc_to_dense_f64`.
- REQ-6: CSR↔CSC conversion — `pub fn gpu_csr_to_csc_f32` /
  `gpu_csr_to_csc_f64` (and the inverse via `cusparseCsr2cscEx2`).
- REQ-7: COO↔CSR conversion — `pub fn gpu_coo_to_csr_f32` /
  `gpu_coo_to_csr_f64` plus `gpu_csr_to_coo_f32` /
  `gpu_csr_to_coo_f64`.
- REQ-8: Error policy — every cuSPARSE call surfaces as
  `GpuError::Sparse(...)` on non-success; no silent CPU fallback,
  no unwrap/expect outside `#[cfg(test)]`. Matches
  `TORCH_CUDASPARSE_CHECK` pattern at
  `aten/src/ATen/native/sparse/cuda/SparseBlasImpl.cpp:583`.
- REQ-9: No-CUDA stubs — every cuda function has a
  `cfg(not(feature = "cuda"))` stub returning
  `GpuError::NoCudaFeature`.

## Acceptance Criteria

- [x] AC-1: All sparse functions have non-test consumers in
  `ferrotorch-gpu/src/backend_impl.rs` (the cuda backend's sparse
  dispatch arms at lines 4764-5014).
- [x] AC-2: `CusparseHandle` drop is exercised by RAII (every
  consumer constructs a handle scoped to one operation, or uses
  the cached handle on the backend).
- [x] AC-3: No-CUDA compile path — `cargo build -p ferrotorch-gpu
  --no-default-features` succeeds.

## Architecture

### `CusparseHandle` RAII (REQ-1)

`pub struct CusparseHandle in sparse.rs` (line 48) wraps the raw
`cusparseHandle_t`. `impl CusparseHandle in sparse.rs` (line 65)
provides constructor; `impl Drop for CusparseHandle in sparse.rs`
(line 81) calls `cusparseDestroy` exactly once at drop. The handle
is bound to the device's default stream via `cusparseSetStream`
before each call (documented in the module `//!` doc-comment).

Mirrors `aten/src/ATen/cuda/CuSparseHandlePool.cpp`'s thread-local
handle pattern. Per R-DEV-4 the RAII wrapper replaces upstream's
manual destroy callsites.

### SpMM (REQ-2)

`pub fn gpu_spmm_csr_f32 in sparse.rs` (line 137):
1. Take `&CusparseHandle` (passed by caller via the backend's
   cached handle) + CSR triple `(crow_indices, col_indices,
   values)` + dense buffer + dimensions.
2. Build sparse and dense descriptors via
   `cusparseCreateCsr` / `cusparseCreateDnMat`.
3. Query work-buffer size via `cusparseSpMM_bufferSize`.
4. Allocate work buffer + output buffer on device.
5. Call `cusparseSpMM` with `CUSPARSE_SPMM_ALG_DEFAULT`.
6. Destroy descriptors; release work buffer.
7. Return output as `CudaBuffer<f32>`.

Mirrors `aten/src/ATen/native/sparse/cuda/SparseBlasImpl.cpp:528-600`.
Consumer at `backend_impl.rs` (f32) and `backend_impl.rs` (f64).

### Sparse↔dense conversions (REQ-3, REQ-4)

`pub fn gpu_sparse_to_dense_csr_f32 in sparse.rs` (line 593) calls
`cusparseSparseToDense` with `CUSPARSE_SPARSETODENSE_ALG_DEFAULT`.
Consumer at `backend_impl.rs` (f32) and `backend_impl.rs` (f64).

`pub fn gpu_dense_to_sparse_csr_f32 in sparse.rs` (line 924) calls
`cusparseDenseToSparse_analysis` + `cusparseDenseToSparse_convert`.
Consumer at `backend_impl.rs` (f32) and `backend_impl.rs` (f64).

### Format conversions (REQ-5, REQ-6, REQ-7)

`pub fn gpu_csc_to_dense_f32 in sparse.rs` (line 1374): CSC dense
materialisation via the cuSPARSE generic API. Consumer at
`backend_impl.rs` (f32) and `backend_impl.rs` (f64).

`pub fn gpu_csr_to_csc_f32 in sparse.rs` (line 1700): wraps
`cusparseCsr2cscEx2`. Consumer at `backend_impl.rs` (f32) and
`backend_impl.rs` (f64).

`pub fn gpu_coo_to_csr_f32 in sparse.rs` (line 2067): wraps
`cusparseXcoo2csr`. Consumer at `backend_impl.rs` (f32) and
`backend_impl.rs` (f64).

`pub fn gpu_csr_to_coo_f32 in sparse.rs` (line 2164): wraps
`cusparseXcsr2coo`. Consumer at `backend_impl.rs` (f32) and
`backend_impl.rs` (f64).

### Error policy (REQ-8)

Every cuSPARSE call is wrapped to surface `GpuError::Sparse(code)`
on non-`CUSPARSE_STATUS_SUCCESS`. No unwrap / expect in production
code. Mirrors `TORCH_CUDASPARSE_CHECK` at upstream
`SparseBlasImpl.cpp:583,600`.

### No-CUDA stubs (REQ-9)

The trailing block has `#[cfg(not(feature = "cuda"))]` stubs for
every cuda function, returning `Err(GpuError::NoCudaFeature)`.
Crate compiles in both modes.

## Parity contract

`parity_ops = []` for this module. Reason: sparse ops are op-level
entries in `ferrotorch-core::sparse` (`sparse_mm`, `to_dense`,
`to_sparse`, etc.); the cuSPARSE dispatchers are reached
transitively when the GPU backend arm is selected.

Edge cases mirrored from upstream:

- **Empty CSR (nnz = 0)**: cuSPARSE returns success on empty
  descriptors; the wrapper returns an empty output buffer
  matching upstream's empty-sparse semantics.
- **All-zero values with non-empty indices**: explicit zeros are
  preserved (cuSPARSE doesn't coalesce on call). Matches
  upstream behaviour.
- **Mismatched dtype between sparse and dense**: caller is
  responsible; the wrapper accepts only matching dtypes via the
  per-typed `_f32` / `_f64` function selection.
- **Non-coalesced COO input**: ferrotorch's `SparseTensor`
  coalesces on host before upload (sort by `(row, col)`, sum
  duplicates) — documented in the module `//!` doc-comment.
- **`crow_indices` length mismatch**: caller validates; failure
  reaches cuSPARSE which surfaces as `Sparse(...)` error.

## Verification

This file has **0** in-module `#[test]` units. Integration coverage
is at the backend_impl level (`backend_impl.rs` dispatch
arms) plus op-level coverage in `ferrotorch-core/src/sparse.rs`. The
sparse path requires the `cuda` feature + a CUDA device, so it
cannot be exercised by default-feature CI.

Smoke commands:

```bash
cargo test -p ferrotorch-core --features cuda sparse:: 2>&1 | tail -3
cargo build -p ferrotorch-gpu --no-default-features 2>&1 | tail -3
```

Expected: core-side sparse tests pass; no-cuda compile succeeds.
`parity_ops = []` — no per-op parity-sweep applies here.

## REQ status table

Per S5 (existing pub-API grandfather): every sparse function is
consumed by `ferrotorch-gpu/src/backend_impl.rs` (the cuda backend's
sparse dispatch arms at lines 4764-5014). Those arms are reached
from `ferrotorch-core/src/sparse.rs` when a `SparseTensor` op
routes to GPU.

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct CusparseHandle in sparse.rs` (line 48), `impl CusparseHandle in sparse.rs` (line 65), `impl Drop for CusparseHandle in sparse.rs` (line 81). Non-test consumer: every SpMM / conversion entry point in this file takes `handle: &CusparseHandle` as the first parameter; the cuda backend at `backend_impl.rs,4824,4870,4906,4939,4969,4999` etc. passes the cached device-level handle through. |
| REQ-2 | SHIPPED | impl: `pub fn gpu_spmm_csr_f32 in sparse.rs` (line 137) and `pub fn gpu_spmm_csr_f64 in sparse.rs` (line 377) per upstream `aten/src/ATen/native/sparse/cuda/SparseBlasImpl.cpp:528::spmm`. Non-test consumer: `spmm in backend_impl.rs` (f32) and `spmm in backend_impl.rs` (f64). |
| REQ-3 | SHIPPED | impl: `pub fn gpu_sparse_to_dense_csr_f32 in sparse.rs` (line 593) and `pub fn gpu_sparse_to_dense_csr_f64 in sparse.rs` (line 763). Non-test consumer: `gpu_sparse_to_dense_csr_f64 in backend_impl.rs` (f32) and `backend_impl.rs` (f64) — the `SparseTensor::to_dense_on` GPU arm. |
| REQ-4 | SHIPPED | impl: `pub fn gpu_dense_to_sparse_csr_f32 in sparse.rs` (line 924) and `pub fn gpu_dense_to_sparse_csr_f64 in sparse.rs` (line 1148). Non-test consumer: `gpu_dense_to_sparse_csr_f64 in backend_impl.rs` (f32) and `backend_impl.rs` (f64) — the `SparseTensor::from_dense` GPU arm. |
| REQ-5 | SHIPPED | impl: `pub fn gpu_csc_to_dense_f32 in sparse.rs` (line 1374) and `pub fn gpu_csc_to_dense_f64 in sparse.rs` (line 1537). Non-test consumer: `gpu_csc_to_dense_f64 in backend_impl.rs` (f32) and `backend_impl.rs` (f64). |
| REQ-6 | SHIPPED | impl: `pub fn gpu_csr_to_csc_f32 in sparse.rs` (line 1700) and `pub fn gpu_csr_to_csc_f64 in sparse.rs` (line 1854). Non-test consumer: `gpu_csr_to_csc_f64 in backend_impl.rs` (f32) and `backend_impl.rs` (f64). |
| REQ-7 | SHIPPED | impl: `pub fn gpu_coo_to_csr_f32 in sparse.rs` (line 2067), `pub fn gpu_coo_to_csr_f64 in sparse.rs` (line 2088), `pub fn gpu_csr_to_coo_f32 in sparse.rs` (line 2164), `pub fn gpu_csr_to_coo_f64 in sparse.rs` (line 2192). Non-test consumer: `gpu_csr_to_coo_f64 in backend_impl.rs,4984,4999,5014`. |
| REQ-8 | SHIPPED | impl: every cuSPARSE call in `sparse.rs` is wrapped to return `Err(GpuError::Sparse(...))` on non-success; no unwrap/expect in production code outside `#[cfg(test)]`. Non-test consumer: every caller in `sparse in backend_impl.rs` uses `.map_err(Self::map_gpu_err)?` to propagate the structured error. |
| REQ-9 | SHIPPED | impl: `#[cfg(not(feature = "cuda"))]` stubs for every cuda function in `sparse.rs` return `Err(GpuError::NoCudaFeature)`. Non-test consumer: the same `backend_impl.rs` sparse arms under the no-cuda compile path. |
