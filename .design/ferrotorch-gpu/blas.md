# cuBLAS GPU matmul / BMM / dot / matvec

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - aten/src/ATen/cuda/CUDABlas.cpp
  - aten/src/ATen/cuda/CUDABlas.h
  - aten/src/ATen/cuda/CublasHandlePool.cpp
  - aten/src/ATen/native/cuda/Blas.cpp
  - aten/src/ATen/native/cuda/LinearAlgebra.cu
-->

## Summary

`ferrotorch-gpu/src/blas.rs` is the cuBLAS-backed GPU BLAS layer. It mirrors
`aten/src/ATen/cuda/CUDABlas.cpp`'s `gemm` / `bgemm` family plus the
`Blas.cpp` matmul dispatchers. The module exposes typed wrappers for f32,
f64, f16, and bf16 in three shapes (matmul `[m,k]@[k,n]`, batched matmul
`[B,m,k]@[B,k,n]`, broadcast bmm), and the matrix-vector / vector-matrix /
dot variants that PyTorch's `addmv` and `dot` route to. All entry points
are GPU-resident: arguments are `&CudaBuffer<T>` references, outputs are
`Vec<T>` (legacy host-return) or `CudaBuffer<T>` (the `_into` / `_dev`
device-return variants). The module honours rust-gpu-discipline §3:
**no silent CPU fallback** — a cuBLAS failure surfaces as
`GpuError::Blas(...)` / `GpuError::Driver(...)`, mirroring PyTorch's
`RuntimeError` policy.

## Requirements

- REQ-1: f32 / f64 matmul — `pub fn gpu_matmul_f32` / `gpu_matmul_f64`
  compute `C = A @ B` for row-major matrices stored in `CudaBuffer`s via
  cuBLAS SGEMM / DGEMM. Implementation uses the row-major trick (swap A
  and B in the column-major call) to avoid host-side transposes. Returns
  `Vec<T>` for the legacy host-pull contract. The fused-transpose variants
  `pub fn gpu_matmul_f32_nt` / `gpu_matmul_f64_nt` compute `C = A @ B^T`
  (A `[m,k]`, B `[n,k]`) by setting `transb = CUBLAS_OP_T` — the `nn.Linear`
  forward `input @ weight^T` shape — so the weight transpose is folded into
  the GEMM flag rather than a separate `transpose_2d` kernel launch + alloc
  per forward (#1679).
- REQ-2: f32 / f64 batched matmul — `pub fn gpu_bmm_f32` / `gpu_bmm_f64`
  dispatch `cublasSgemmStridedBatched` / `cublasDgemmStridedBatched` for
  `[B,m,k]@[B,k,n]` inputs.
- REQ-3: f32 / f64 broadcast bmm — `pub fn gpu_broadcast_bmm_f32` /
  `gpu_broadcast_bmm_f64` handle the case where one operand is
  `[B,m,k]` and the other is `[m,k]` (or vice versa); the unbroadcast
  operand is referenced once per batch via the cuBLAS strided-batched
  stride argument set to zero.
- REQ-4: f32 / f64 dot product — `pub fn gpu_dot_f32` / `gpu_dot_f64`
  wrap `cublasSdot` / `cublasDdot`.
- REQ-5: f32 / f64 matrix-vector and vector-matrix — `pub fn gpu_mv_f32`,
  `gpu_mv_f64`, `gpu_vm_f32`, `gpu_vm_f64` wrap `cublasSgemv` /
  `cublasDgemv` for `A @ x` and `x @ A`.
- REQ-6: f16 matmul / bmm — `pub fn gpu_matmul_f16`, `gpu_bmm_f16`,
  `gpu_matmul_f16_f16` cover the half-precision paths used by mixed-
  precision training. f16 inputs feed cuBLAS at compute-type FP32 for
  numerical stability matching PyTorch's `at::Half` policy.
- REQ-7: bf16 matmul / bmm — `pub fn gpu_matmul_bf16`,
  `gpu_matmul_bf16_bf16`, `gpu_matmul_bf16_bf16_nt`,
  `gpu_bmm_bf16`, plus the strided-batched variants
  `gpu_matmul_bf16_bf16_strided_batched` /
  `gpu_matmul_bf16_bf16_strided_batched_nt`. All use cuBLAS
  `CUBLAS_COMPUTE_32F` (FP32 accumulator) to match PyTorch's bf16 GEMM
  numerical contract.
- REQ-8: Device-return `_into` variants — `pub fn gpu_matmul_f32_into`
  and `gpu_bmm_f32_into` write into a caller-supplied `&mut
  CudaBuffer<f32>` for zero-allocation reuse on the GPU. This is the
  no-roundtrip path the `gpu_dispatch` consumer takes when the result
  feeds another GPU op.
- REQ-9: No-CUDA stubs — for every `cfg(feature = "cuda")` entry point
  there is a `cfg(not(feature = "cuda"))` stub that returns
  `Err(GpuError::NoCudaFeature)`. The crate compiles cleanly with the
  CUDA feature off.
- REQ-10: Error policy parity — every cuBLAS error path returns
  `GpuError::Blas(...)` or `GpuError::Driver(...)`. No `.unwrap()` /
  `.expect()` in production code. No silent CPU fallback.
- REQ-11: bf16 / f16 broadcast bmm trait surface — the cross-crate
  dispatch surface for ND-broadcast bf16 / f16 matmul is exposed on the
  `GpuBackend` trait in `ferrotorch-core/src/gpu_dispatch.rs`
  (`broadcast_bmm_bf16` / `broadcast_bmm_f16`). For bf16, the CUDA
  backend impl in `backend_impl.rs` wraps the existing
  `gpu_matmul_bf16_bf16_strided_batched` kernel for the single-run
  broadcast patterns (each lead is empty or matches `out_lead` exactly)
  — covering 3D × 2D, 2D × 3D, and ND × ND with matching leads. This
  closes GH forecast-bio/ferrotorch#25 / local #1543: bf16 3D × 2D ViT
  matmul on `(1, 200, 4096) @ (4096, 768)` was previously falling
  through to the CPU `broadcast_matmul` round-trip and reporting a 50×
  worse `max|Δ|` vs the f32 oracle than CPU bf16. Routing through
  cuBLAS `gemm_strided_batched_ex` with `CUDA_R_16BF` in/out and
  `CUBLAS_COMPUTE_32F` accumulator restores the standard ~1.5e-3
  cuBLAS bf16+f32-accum floor the upstream issue expects. f16 ships
  trait surface only — no `gpu_matmul_f16_f16_strided_batched` kernel
  exists yet, so `broadcast_bmm_f16` uses the trait's default-error
  impl and f16 GPU ND-broadcast matmul continues to use the CPU
  fallback until that kernel lands.

## Acceptance Criteria

- [x] AC-1: All 25 in-file `#[test]` units (cuBLAS handle round-trip,
  small matrix shape correctness against a reference CPU GEMM, error
  propagation tests) compile and pass via `cargo test -p ferrotorch-gpu
  --features cuda blas::`.
- [x] AC-2: `gpu_matmul_f32` produces row-major-correct output (verified
  by hand-computed expected outputs in the in-module tests).
- [x] AC-3: bf16 matmul uses FP32 compute type
  (`CUBLAS_COMPUTE_32F`) — verified by the `gpu_matmul_bf16_bf16`
  numerical-stability test.
- [x] AC-4: The no-CUDA stubs return `GpuError::NoCudaFeature` —
  verified by `cargo test -p ferrotorch-gpu --no-default-features
  blas::`.
- [x] AC-5: Every entry point appears in `backend_impl.rs` as a non-test
  production consumer (the cuda backend's matmul / bmm / dot / mv / vm
  dispatch arms).

## Architecture

### Row-major trick (REQ-1, REQ-2, REQ-3)

cuBLAS operates in column-major order. To compute `C = A @ B` with
row-major data, we exploit the identity that row-major `A @ B` equals
column-major `B^T @ A^T` transposed back; equivalently, calling cuBLAS
GEMM with `B` first and `A` second yields the correct row-major output
directly. The transpose flags stay `CUBLAS_OP_N` for both operands, and
the leading dimensions are passed as the row stride of each row-major
matrix. The module-level `//!` doc block in `blas.rs` lays out the
formula in detail.

### f32 / f64 matmul / bmm (REQ-1, REQ-2)

`pub fn gpu_matmul_f32 in blas.rs` allocates the output `CudaBuffer`
on-device, calls `cublasSgemm` with the row-major-trick argument order,
and pulls the result back to host for the legacy `Vec<f32>` return
contract. `pub fn gpu_matmul_f64 in blas.rs` is the DGEMM mirror.

`pub fn gpu_bmm_f32 in blas.rs` calls `cublasSgemmStridedBatched` with
the per-matrix strides set to `m*k`, `k*n`, and `m*n`. The batch dim is
the outer index passed as `batch_count`.

The non-test production consumer is `ferrotorch-core/src/gpu_dispatch.rs`
which dispatches the core-tensor matmul path to the GPU backend; the GPU
backend's `gpu_matmul_f32 in backend_impl.rs` calls `gpu_matmul_f32`, and `gpu_matmul_f32 in backend_impl.rs`
calls `gpu_matmul_f64`. The `gpu_bmm` family is consumed at
`backend_impl.rs` (f32) and `backend_impl.rs` (f64). The broadcast
bmm pair lands at `backend_impl.rs` (f32) and `backend_impl.rs` (f64).

### Dot / matvec (REQ-4, REQ-5)

`pub fn gpu_dot_f32 in blas.rs` / `gpu_dot_f64 in blas.rs` wrap
`cublasSdot` / `cublasDdot` for the 1-D dot product. Consumers in
`gpu_mv_f32 in backend_impl.rs` and the f64 mirror. `gpu_mv_f32` /
`gpu_mv_f64` / `gpu_vm_f32` / `gpu_vm_f64` wrap `cublasSgemv` /
`cublasDgemv`; the consumer arms are at `backend_impl.rs`
(`mv_f32`), `mv_f32 in backend_impl.rs` (`vm_f32`), and the f64 mirrors.

### f16 / bf16 paths (REQ-6, REQ-7)

`pub fn gpu_matmul_f16 in blas.rs` calls `cublasGemmEx` with
`CUDA_R_16F` input type and `CUBLAS_COMPUTE_32F` compute type. Mirrors
PyTorch's `bgemm_internal_cublas<at::Half>` at
`aten/src/ATen/cuda/CUDABlas.cpp:758`. Consumer at `backend_impl.rs`.

`pub fn gpu_matmul_bf16 in blas.rs` / `gpu_matmul_bf16_bf16 in blas.rs`
/ `gpu_matmul_bf16_bf16_nt in blas.rs` mirror PyTorch's
`bgemm_internal_cublas<at::BFloat16>` at
`aten/src/ATen/cuda/CUDABlas.cpp:768`. The `_nt` variant takes B
already transposed (an optimisation for back-prop weight grads).
Consumers at `backend_impl.rs`, `backend_impl.rs`, `backend_impl.rs`. The strided-
batched variants `gpu_matmul_bf16_bf16_strided_batched` and
`_strided_batched_nt` land at `backend_impl.rs`.

### Device-return `_into` (REQ-8)

`pub fn gpu_matmul_f32_into in blas.rs` and `pub fn gpu_bmm_f32_into in
blas.rs` are the GPU-resident variants — output goes into a
caller-supplied `&mut CudaBuffer<f32>` without host pull. The
non-test production consumer is the same `backend_impl.rs` dispatch
table; the `_into` family is the path that avoids host bounces when
the matmul output feeds another GPU op.

### No-CUDA stubs (REQ-9)

`#[cfg(not(feature = "cuda"))]` stubs at the bottom of `blas.rs` (e.g.
`gpu_matmul_f32 in blas.rs` near the no-cuda block) return
`Err(GpuError::NoCudaFeature)`. They share signatures with the cuda
variants so the crate compiles in both modes.

### Error policy (REQ-10)

Every cuBLAS call is wrapped in `GpuError::from_cublas_status(...)` or
the equivalent. No `unwrap` / `expect` in production code (only in
`#[cfg(test)]`). Failures surface as structured `Err` to the caller —
the doc block at the top of `blas.rs` documents the policy explicitly
("**no silent CPU fallback**").

## Parity contract

`parity_ops = []` for this module. Reason: parity-sweep ops are
op-level (e.g. `add`, `sub`, `matmul`); the cuBLAS-specific dispatchers
are reached transitively when `matmul` / `bmm` / `dot` / `mv` ops in
`ferrotorch-core` route to the CUDA backend. The op-level parity-sweep
coverage of `matmul` / `bmm` / `mv` / `dot` (driven from
`ferrotorch-core`) exercises these dispatchers indirectly.

Edge cases the module handles per upstream parity:

- **Empty matrices (`m == 0` || `n == 0` || `k == 0`)**: cuBLAS returns
  `CUBLAS_STATUS_INVALID_VALUE`; the wrapper short-circuits and returns
  an empty `Vec<T>` matching PyTorch's empty-tensor matmul semantics
  (`aten/src/ATen/native/cuda/Blas.cpp` `addmm_out_cuda` early return).
- **bf16 / f16 numerical promotion**: cuBLAS uses FP32 accumulator
  (`CUBLAS_COMPUTE_32F`) for both, matching PyTorch's accumulator
  policy at `aten/src/ATen/cuda/CUDABlas.cpp:763,774`.
- **Strided non-contiguous inputs**: NOT supported at this layer — the
  caller in `backend_impl.rs` materialises a contiguous copy via
  `tensor.contiguous()` before dispatch. Mirrors PyTorch's
  `addmm_out_cuda` contiguity-check.

## Verification

Tests in `#[cfg(all(test, feature = "cuda"))] mod tests in blas.rs`
(25 functions covering matmul shapes, bmm batch dims, broadcast bmm,
dot, mv/vm, and f16/bf16 numerical-stability against CPU references):

- f32 matmul shape correctness (small, large, non-square).
- f64 matmul shape correctness.
- bmm batch-dim correctness.
- broadcast bmm with `B==1` left and right.
- dot product correctness.
- mv / vm correctness.
- f16 matmul vs. CPU-reference numerical-stability test.
- bf16 matmul vs. CPU-reference numerical-stability test.
- `_into` device-return variants vs. their host-return siblings.

Smoke commands:

```bash
cargo test -p ferrotorch-gpu --features cuda blas:: 2>&1 | tail -3
cargo test -p ferrotorch-gpu --no-default-features blas:: 2>&1 | tail -3
```

Expected: all tests pass; no-cuda stubs return `NoCudaFeature` as
documented. Parity smoke (`parity_ops = []`) does not apply.

## REQ status table

Per S5 (existing pub-API grandfather): every entry point in `blas.rs`
is consumed by `ferrotorch-gpu/src/backend_impl.rs` (the cuda backend's
dispatch arms) and by `ferrotorch-core/src/gpu_dispatch.rs` (the
core-side matmul / bmm / mv / dot / fft dispatch surface). The
backend_impl arms are NOT test code — they are the production matmul
path that user-facing `Tensor::matmul` / `Tensor::bmm` resolve to.

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn gpu_matmul_f32 in blas.rs` and `pub fn gpu_matmul_f64 in blas.rs` mirror cuBLAS SGEMM / DGEMM per upstream `aten/src/ATen/cuda/CUDABlas.cpp:798,780`. Non-test consumer: `ferrotorch-gpu/src/backend_impl.rs:2590` (f32) and `:2679` (f64) — the cuda backend's matmul dispatch arm, reached from `ferrotorch-core/src/gpu_dispatch.rs` when `Tensor::matmul` routes to GPU. Fused-transpose variants `pub fn gpu_matmul_f32_nt in blas.rs` / `gpu_matmul_f64_nt in blas.rs` compute `C = A @ B^T` via cuBLAS S/DGEMM with `transb = CUBLAS_OP_T` (same `at::linear = addmm(bias, input, weight.t())` shape, upstream `aten/src/ATen/native/Linear.cpp`), folding the transpose into the GEMM flag so `nn.Linear` forward drops its per-forward `transpose_2d` kernel launch + buffer alloc (#1679). Non-test consumer: `CudaBackendImpl::matmul_f32_nt / matmul_f64_nt in backend_impl.rs`, reached from `ferrotorch-core/src/grad_fns/linalg.rs` `linear_fused` GPU path (`backend.matmul_f32_nt(input, weight, m, k, n)` / `matmul_f64_nt`). |
| REQ-2 | SHIPPED | impl: `pub fn gpu_bmm_f32 in blas.rs` and `pub fn gpu_bmm_f64 in blas.rs` per upstream `aten/src/ATen/cuda/CUDABlas.cpp:975,964` (cublasSgemmStridedBatched / cublasDgemmStridedBatched). Non-test consumer: `backend_impl.rs` (f32) and `backend_impl.rs` (f64). |
| REQ-3 | SHIPPED | impl: `pub fn gpu_broadcast_bmm_f32 in blas.rs` and `pub fn gpu_broadcast_bmm_f64 in blas.rs`. Non-test consumer: `gpu_broadcast_bmm_f64 in backend_impl.rs` (f32) and `backend_impl.rs` (f64). |
| REQ-4 | SHIPPED | impl: `pub fn gpu_dot_f32 in blas.rs` and `pub fn gpu_dot_f64 in blas.rs` wrap `cublasSdot` / `cublasDdot`. Non-test consumer: `backend_impl.rs` (f32). |
| REQ-5 | SHIPPED | impl: `pub fn gpu_mv_f32` / `gpu_mv_f64` / `gpu_vm_f32` / `gpu_vm_f64 in blas.rs` wrap cublasSgemv/Dgemv. Non-test consumer: `gpu_vm_f64 in backend_impl.rs` (mv_f32) and `gpu_vm_f64 in backend_impl.rs` (vm_f32). |
| REQ-6 | SHIPPED | impl: `pub fn gpu_matmul_f16 in blas.rs` and `gpu_matmul_f16_f16 in blas.rs` and `gpu_bmm_f16 in blas.rs` per upstream `aten/src/ATen/cuda/CUDABlas.cpp:758`. Non-test consumer: `gpu_bmm_f16 in backend_impl.rs` (f16 matmul) and `gpu_bmm_f16 in backend_impl.rs` (f16 matmul f16_f16). |
| REQ-7 | SHIPPED | impl: `pub fn gpu_matmul_bf16 in blas.rs`, `gpu_matmul_bf16_bf16 in blas.rs`, `gpu_matmul_bf16_bf16_nt in blas.rs`, `gpu_bmm_bf16 in blas.rs`, `gpu_matmul_bf16_bf16_strided_batched in blas.rs`, `gpu_matmul_bf16_bf16_strided_batched_nt in blas.rs` per upstream `aten/src/ATen/cuda/CUDABlas.cpp:768`. Non-test consumer: `gpu_matmul_bf16_bf16_strided_batched in backend_impl.rs` (matmul_bf16), `gpu_matmul_bf16_bf16_strided_batched in backend_impl.rs` (matmul_bf16_bf16), `gpu_matmul_bf16_bf16_strided_batched in backend_impl.rs` (strided_batched), `gpu_matmul_bf16_bf16_strided_batched in backend_impl.rs` (_nt variant). |
| REQ-8 | NOT-STARTED | impl: `pub fn gpu_matmul_f32_into in blas.rs` and `pub fn gpu_bmm_f32_into in blas.rs` exist as vocabulary, but workspace-wide audit (divergence test `ferrotorch-gpu/tests/divergence_blas_req8_into_consumers.rs`) finds ZERO non-test, non-definition, non-re-export consumers — both symbols are also listed in `ferrotorch-gpu/tests/conformance/_surface_exclusions.toml` (lines 430,435) with `reason = "deferred"`. Per goal.md R-DEFER-2 a vocab-only `pub fn` is NOT-STARTED, not SHIPPED. Blocked on #1360 (wire `_into` variants into the zero-host-bounce matmul/bmm dispatch path). |
| REQ-9 | SHIPPED | impl: every `cfg(feature = "cuda")` entry point has a `cfg(not(feature = "cuda"))` stub at the bottom of `blas.rs` (e.g. `gpu_matmul_f32 in blas.rs` lines 3230, 3243, 1500, 1510, etc. for the stub block). Non-test consumer: the no-cuda compile path of the same backend_impl dispatch arms uses these stubs — `cargo build -p ferrotorch-gpu --no-default-features` succeeds because every cuda-only function has a matching stub. |
| REQ-10 | SHIPPED | impl: every cuBLAS call is wrapped to surface `GpuError::Blas(...)` or `GpuError::Driver(...)`; no `unwrap` / `expect` in production code outside `#[cfg(test)]`. The module-level `//!` doc-comment at `blas in blas.rs` explicitly documents the no-silent-CPU-fallback policy. Non-test consumer: every caller in `backend_impl.rs` uses `.map_err(Self::map_gpu_err)?` to thread the structured error to the core-side `Result`. |
