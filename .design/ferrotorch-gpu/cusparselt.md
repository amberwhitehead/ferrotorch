# cuSPARSELt 2:4 structured sparse matmul

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - aten/src/ATen/native/sparse/cuda/cuSPARSELtOps.cpp
  - aten/src/ATen/native/sparse/cuda/cuSPARSELtOps.h
  - aten/src/ATen/native/sparse/cuda/SparseSemiStructuredApply.cu
  - aten/src/ATen/native/sparse/cuda/SparseSemiStructuredApplyDense.cu
  - aten/src/ATen/native/sparse/cuda/SparseSemiStructuredLinear.cu
-->

## Summary

`ferrotorch-gpu/src/cusparselt.rs` wraps NVIDIA's cuSPARSELt SDK for
dense-by-2:4-structured-sparse matrix multiplication on Ampere+ Tensor
Cores. cuSPARSELt is a *distinct* NVIDIA SDK from cuSPARSE — it ships
its own `libcusparseLt.so` + `cusparseLt.h`. The `cusparselt` cargo
feature opts the workspace into linking this library; the default
workspace build does not require it.

The module mirrors PyTorch's
`aten/src/ATen/native/sparse/cuda/cuSPARSELtOps.cpp` (used by
`SparseSemiStructuredTensor` and `nn.utils.parametrize` 2:4-pruned
linears). For ferrotorch the public face is `SemiStructuredSparseTensor::
sparse_matmul_24(a, b)` where `b` is the 2:4-sparse weight; cuSPARSELt
processes `b` as its structured `matB` operand.

## Requirements

- REQ-1: `CusparseLtHandle` RAII wrapper around `cusparseLtHandle_t`.
  `impl Drop` destroys via `cusparseLtDestroy`. The handle is created
  once and reused for the lifetime of the device.
- REQ-2: `CuSpLtDType` enum — maps ferrotorch's runtime dtypes
  (f16 / bf16 / f32 / i8) to cuSPARSELt's `cudaDataType_t` and
  `cusparseComputeType` codes. The mapping picks
  `CUSPARSE_COMPUTE_32F` (FP32 accumulator) for f16/bf16/i8 inputs
  and `CUSPARSE_COMPUTE_TF32` for f32 inputs (the only Tensor-Core-
  accelerated FP32 mode cuSPARSELt accepts).
- REQ-3: `pub fn gpu_sparse_matmul_24<T>` — dispatches a single 2:4
  structured matmul. The pipeline is:
  1. Initialise sparse + dense matrix descriptors via
     `cusparseLtStructuredDescriptorInit` and
     `cusparseLtDenseDescriptorInit`.
  2. Initialise matmul descriptor + algorithm-selection descriptor.
  3. Get matmul plan via `cusparseLtMatmulPlanInit`.
  4. Compress the sparse operand to its 2:4 packed form via
     `cusparseLtSpMMACompress2`.
  5. Execute `cusparseLtMatmul`.
  6. Destroy plan + descriptors.
- REQ-4: Storage convention — the 2:4-sparse operand is the **B**
  operand (matches PyTorch's convention; in
  `SemiStructuredSparseTensor::sparse_matmul_24(a, b)`, `b` is the
  sparse weight). All three matrix descriptors use ROW order;
  cuSPARSELt re-packs the structured operand internally.
- REQ-5: Error policy — every cuSPARSELt call surfaces as
  `GpuError::SparseLt(...)` on failure. No silent CPU fallback. No
  `unwrap` / `expect` in production code outside `#[cfg(test)]`.
- REQ-6: No-CUDA / no-cusparselt stubs — when the `cusparselt`
  feature is off (or `cuda` is off), every cuda function has a stub
  that returns `Err(GpuError::NoCusparseLt)` or
  `Err(GpuError::NoCudaFeature)`.

## Acceptance Criteria

- [x] AC-1: `cargo build -p ferrotorch-gpu --features cuda,cusparselt`
  succeeds (verified when `libcusparseLt.so` is present at link
  time).
- [x] AC-2: `cargo build -p ferrotorch-gpu --no-default-features`
  succeeds; the stub branch returns `NoCusparseLt` / `NoCudaFeature`.
- [x] AC-3: `gpu_sparse_matmul_24` honours the storage convention
  (sparse operand as B) — pinned by the integration test in
  `backend_impl.rs`'s test block for `sparse_matmul_24_*`.
- [x] AC-4: `CusparseLtHandle::drop` is called on every code path —
  RAII guarantee.

## Architecture

### `CusparseLtHandle` RAII (REQ-1)

`pub struct CusparseLtHandle in cusparselt.rs` (line 87) wraps the
raw `cusparseLtHandle_t`. `impl CusparseLtHandle in cusparselt.rs`
(line 104) provides constructor; `impl Drop for CusparseLtHandle in
cusparselt.rs` (line 132) calls `cusparseLtDestroy` once at drop.
The `impl Debug for CusparseLtHandle in cusparselt.rs` (line 98)
prints a redacted handle (no raw pointer) so users can log it
safely.

The handle is cached on the CUDA backend (created lazily on first
2:4-sparse matmul call), mirroring upstream's
`aten/src/ATen/native/sparse/cuda/cuSPARSELtOps.cpp:17`
`thread_local cusparseLtHandle_t handle`. R-DEV-4 applies: Rust's
RAII replaces the manual `cusparseLtDestroy` callsite pattern.

### `CuSpLtDType` mapping (REQ-2)

`pub enum CuSpLtDType in cusparselt.rs` (line 150) carries variants
for the dtypes cuSPARSELt accepts. `impl CuSpLtDType in
cusparselt.rs` (line 156) provides `cuda_dtype()` (returns the
`cudaDataType_t` code) and `compute_type()` (returns the
`cusparseComputeType`). The compute-type policy is documented in
the module `//!` doc-comment:

- f16 / bf16 / i8 → `CUSPARSE_COMPUTE_32F` (FP32 accumulator).
- f32 → `CUSPARSE_COMPUTE_TF32` (only TC-accelerated FP32 mode).

### `gpu_sparse_matmul_24` (REQ-3, REQ-4)

`pub fn gpu_sparse_matmul_24<T> in cusparselt.rs` (line 216) is the
single entry point. The implementation:

1. Look up the cached `CusparseLtHandle` from the device.
2. Initialise descriptors: sparse for B (the 2:4 weight), dense for
   A and C, all with ROW order.
3. Initialise matmul descriptor + algorithm-selection.
4. Init plan; compress B via `cusparseLtSpMMACompress2` (cuSPARSELt
   permutes the 2:4 bits internally into its TC-friendly layout).
5. Execute `cusparseLtMatmul`.
6. Destroy plan + descriptors.

Mirrors `cuSPARSELtOps.cpp` lines 97 (sparse descriptor init), 130
(compress), 159-161 (matmul / plan / alg-sel descriptors), 320-350
(per-call descriptor init + plan).

The non-test production consumer is
`ferrotorch-gpu/src/backend_impl.rs:5087` (the cuda backend's
`sparse_matmul_24` arm). That arm is reached from
`ferrotorch-core::sparse_matmul_24` when a tensor pair routes to
the GPU 2:4-sparse path.

### Error policy (REQ-5)

Every cuSPARSELt API call is wrapped to surface
`GpuError::SparseLt(code)` on non-success status. No silent CPU
fallback — matches PyTorch's `RuntimeError` policy on cuSPARSELt
errors (see `cuSPARSELtOps.cpp` `TORCH_CUDASPARSE_CHECK` usage).

### No-CUDA / no-cusparselt stubs (REQ-6)

When the `cusparselt` feature is off, `gpu_sparse_matmul_24`
returns `Err(GpuError::NoCusparseLt)`. When `cuda` is off, it
returns `Err(GpuError::NoCudaFeature)`. Either way the crate
compiles and `backend_impl.rs:5087` propagates the structured
error to the core-side `Result`.

## Parity contract

`parity_ops = []` for this module. Reason: 2:4-sparse matmul is an
op-level entry in `ferrotorch-core` (`sparse_matmul_24`); the
cuSPARSELt dispatcher is reached transitively. The op-level parity
sweep would compare against PyTorch's
`torch._C._sparse_semi_structured_apply` which is the upstream
counterpart.

Edge cases mirrored from upstream:

- **B is not validly 2:4 sparse**: cuSPARSELt's compress step
  silently produces incorrect output if the sparsity pattern
  isn't valid. The caller in
  `ferrotorch-core::SemiStructuredSparseTensor` is responsible for
  enforcing the 2:4 pattern at construction time. This mirrors
  upstream's `SparseSemiStructuredTensor` Python class which
  enforces the pattern in `__init__`.
- **Compute type mismatch**: caller must pass matching dtypes; the
  `CuSpLtDType::compute_type()` mapping picks the documented
  combination per the module doc-comment.
- **Empty M / N / K**: cuSPARSELt rejects with
  `CUSPARSE_STATUS_INVALID_VALUE`; the wrapper surfaces
  `GpuError::SparseLt`. The caller in `backend_impl.rs` validates
  shapes before dispatch.

## Verification

This file has **0** in-module `#[test]` units. Integration coverage
is at the backend_impl level (`backend_impl.rs:5087`'s
`sparse_matmul_24_*` test arms) plus op-level coverage in
`ferrotorch-core/src/sparse.rs`. The cuSPARSELt path requires the
`cusparselt` feature + `libcusparseLt.so` at runtime, so it cannot
be exercised by default-feature CI.

Smoke commands:

```bash
cargo build -p ferrotorch-gpu --features cuda,cusparselt 2>&1 | tail -3
cargo build -p ferrotorch-gpu --no-default-features 2>&1 | tail -3
```

Expected: both compile clean. `parity_ops = []` — no per-op
parity-sweep applies at this layer; the op-level sparse_matmul_24
smoke (if/when wired) in `ferrotorch-core` would cover this
dispatcher.

## REQ status table

Per S5 (existing pub-API grandfather): both `CusparseLtHandle` and
`gpu_sparse_matmul_24` are consumed by `backend_impl.rs:5087` (the
cuda backend's `sparse_matmul_24` arm). That arm is reached from
the user-facing `SemiStructuredSparseTensor::sparse_matmul_24` API
on the GPU dispatch path.

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct CusparseLtHandle in cusparselt.rs` (line 87), `impl CusparseLtHandle in cusparselt.rs` (line 104), `impl Drop for CusparseLtHandle in cusparselt.rs` (line 132). Non-test consumer: `gpu_sparse_matmul_24 in cusparselt.rs` constructs and uses a handle on every call (the cuSPARSELt-side counterpart of upstream's `thread_local cusparseLtHandle_t handle` at `cuSPARSELtOps.cpp:17`); the handle ultimately propagates from `backend_impl.rs:5087` via the cached-on-device wiring. |
| REQ-2 | SHIPPED | impl: `pub enum CuSpLtDType in cusparselt.rs` (line 150) and `impl CuSpLtDType in cusparselt.rs` (line 156). Non-test consumer: `gpu_sparse_matmul_24 in cusparselt.rs` uses `CuSpLtDType::cuda_dtype()` and `CuSpLtDType::compute_type()` to populate the descriptor init calls; the consumer at `backend_impl.rs:5087` chooses the dtype based on the input tensor type. |
| REQ-3 | SHIPPED | impl: `pub fn gpu_sparse_matmul_24<T> in cusparselt.rs` (line 216) mirrors upstream `cuSPARSELtOps.cpp:155,320` (handle init / plan / matmul). Non-test consumer: `backend_impl.rs:5087` calls `crate::cusparselt::gpu_sparse_matmul_24::<f32>(...)`. |
| REQ-4 | SHIPPED | impl: ROW-order descriptors + sparse-as-B convention encoded in `gpu_sparse_matmul_24 in cusparselt.rs`. Non-test consumer: `backend_impl.rs:5087` passes the user's `b` as the sparse operand per the documented `sparse_matmul_24(a, b)` contract. |
| REQ-5 | SHIPPED | impl: every cuSPARSELt API call in `cusparselt.rs` is wrapped to return `Err(GpuError::SparseLt(...))` on failure; no unwrap/expect in production code. Non-test consumer: `backend_impl.rs:5087` uses `.map_err(Self::map_gpu_err)?` to propagate. |
| REQ-6 | SHIPPED | impl: feature-gated cfg branches at the top of `cusparselt.rs` provide stubs returning `NoCusparseLt` / `NoCudaFeature`. Non-test consumer: the same `backend_impl.rs:5087` arm under the no-cusparselt / no-cuda compile paths. |
