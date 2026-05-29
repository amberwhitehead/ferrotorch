# cuSOLVER GPU linear algebra (SVD, QR, Cholesky, eig, solve)

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - aten/src/ATen/native/cuda/linalg/BatchLinearAlgebraLib.cpp
  - aten/src/ATen/native/cuda/linalg/BatchLinearAlgebraLibBlas.cpp
  - aten/src/ATen/native/cuda/linalg/BatchLinearAlgebra.cpp
  - aten/src/ATen/native/cuda/linalg/CUDASolver.cpp
  - aten/src/ATen/native/cuda/linalg/CUDASolver.h
-->

## Summary

`ferrotorch-gpu/src/cusolver.rs` is the cuSOLVER-backed GPU linear-algebra
layer. It mirrors `aten/src/ATen/native/cuda/linalg/BatchLinearAlgebraLib.cpp`'s
`svd_cusolver` / `cholesky_helper_cusolver` / `geqrf_cusolver` /
`lu_factor_looped_cusolver` / `linalg_eigh_cusolver` family. Each entry
point follows the cuSOLVER pattern:

1. Query workspace size via `*_bufferSize`.
2. Allocate workspace + output buffers on the device.
3. Call the cuSOLVER routine.
4. Check `devInfo` — non-zero means the operation failed (singular
   matrix, etc.) and the wrapper surfaces `GpuError::Solver(...)`.

The module exposes two flavours for most operations: host-return
(`gpu_svd_f32` returns `Vec<f32>` triples) and device-return
(`gpu_svd_f32_dev` returns `CudaBuffer<f32>` triples). The `_dev`
variants are the no-roundtrip path the cuda backend takes when the
result feeds another GPU op.

All cuSOLVER routines operate on **column-major** data (LAPACK
convention). The wrappers either transpose input/output on-device via
`crate::kernels::gpu_transpose_2d` or document the column-major
expectation in the function signature.

## Requirements

- REQ-1: SVD — `pub fn gpu_svd_f32` / `gpu_svd_f64` (host-return) and
  `gpu_svd_f32_dev` / `gpu_svd_f64_dev` (device-return). Mirrors
  `svd_cusolver` at `aten/src/ATen/native/cuda/linalg/BatchLinearAlgebraLib.cpp:643`.
- REQ-2: Cholesky factorisation — `pub fn gpu_cholesky_f32` /
  `gpu_cholesky_f64` (host) and `gpu_cholesky_f32_dev` /
  `gpu_cholesky_f64_dev` (device). Mirrors `cholesky_helper_cusolver`
  at `aten/src/ATen/native/cuda/linalg/BatchLinearAlgebraLib.cpp:800`.
- REQ-3: QR factorisation — `pub fn gpu_qr_f32` / `gpu_qr_f64` (host)
  and `gpu_qr_f32_dev` / `gpu_qr_f64_dev` (device). Mirrors
  `geqrf_cusolver` + `orgqr_helper_cusolver` at
  `aten/src/ATen/native/cuda/linalg/BatchLinearAlgebraLib.cpp:1025,1178`.
- REQ-4: Linear solve (LU + back-substitution) — `pub fn gpu_solve_f32`
  / `gpu_solve_f64` (host) and `gpu_solve_f32_dev` /
  `gpu_solve_f64_dev` (device). Computes `A @ x = b → x`.
- REQ-5: LU factorisation — `pub fn gpu_lu_factor_f32` /
  `gpu_lu_factor_f64`. Mirrors `lu_factor_looped_cusolver` at
  `aten/src/ATen/native/cuda/linalg/BatchLinearAlgebraLib.cpp:1647`.
- REQ-6: Least-squares — `pub fn gpu_lstsq_f32` / `gpu_lstsq_f64`.
  Computes `min ||A @ x - b||_2 → x` via cuSOLVER's
  `cusolverDnSgels` / `cusolverDnDgels`.
- REQ-7: Non-symmetric eigendecomposition — `pub fn gpu_eig_f32` /
  `gpu_eig_f64` (host) and `gpu_eig_f32_dev` / `gpu_eig_f64_dev`
  (device). Mirrors `linalg_eig_cusolver_xgeev` at
  `aten/src/ATen/native/cuda/linalg/BatchLinearAlgebraLib.cpp:1636`.
- REQ-8: Symmetric / Hermitian eigendecomposition — `pub fn
  gpu_eigh_f32` / `gpu_eigh_f64`, and the eigvals-only variants
  `gpu_eigvalsh_f32` / `gpu_eigvalsh_f64`. Mirrors
  `linalg_eigh_cusolver` at
  `aten/src/ATen/native/cuda/linalg/BatchLinearAlgebraLib.cpp:1509`.
- REQ-9: `DnParamsHandle` RAII wrapper for `cusolverDnParams_t` —
  `impl Drop` destroys the handle. Used by the eigenvalue and SVD
  paths that take an opaque params object.
- REQ-10: Singular-matrix error policy — every routine checks
  `devInfo` after the cuSOLVER call. Non-zero `devInfo` surfaces as
  `GpuError::Solver(...)` matching PyTorch's `torch._C._linalg_*`
  raising `RuntimeError` on rank-deficient / singular inputs.
- REQ-11: No-CUDA stubs — every cuda entry point has a
  `cfg(not(feature = "cuda"))` stub returning `GpuError::NoCudaFeature`.

## Acceptance Criteria

- [x] AC-1: All 17 in-module `#[test]` units pass under
  `cargo test -p ferrotorch-gpu --features cuda cusolver::`.
- [x] AC-2: SVD reconstructs `A = U @ diag(S) @ Vh` to within
  `1e-5` for f32 / `1e-12` for f64 — pinned by the SVD round-trip
  tests.
- [x] AC-3: Cholesky of a non-PSD matrix returns
  `GpuError::Solver(...)` with non-zero `devInfo` — pinned by the
  singular-matrix tests.
- [x] AC-4: `_dev` variants do not host-pull — verified by buffer-
  pointer identity tests.
- [x] AC-5: No-CUDA stubs return `NoCudaFeature`.

## Architecture

### SVD (REQ-1)

`pub fn gpu_svd_f32 in cusolver.rs` (host-return) computes the
thin/full SVD via `cusolverDnSgesvd` (legacy) or `cusolverDnXgesvd`
(64-bit). The device-return `pub fn gpu_svd_f32_dev in cusolver.rs`
keeps `(U, S, Vh)` on-device as three `CudaBuffer<f32>`s. Consumer
at `backend_impl.rs`: the cuda backend's `linalg_svd`
arm calls `gpu_svd_f32_dev` for the GPU-resident path.

The doc-comment at the head of the module documents the LAPACK
column-major contract: callers transpose row-major input before
calling. Mirrors upstream `svd_cusolver` at
`aten/src/ATen/native/cuda/linalg/BatchLinearAlgebraLib.cpp:643`.

### Cholesky (REQ-2)

`pub fn gpu_cholesky_f32 in cusolver.rs` calls `cusolverDnSpotrf`
(host-return, Vec<f32>); `gpu_cholesky_f32_dev` is the device-return
mirror. Returns the lower-triangular factor `L` such that `A = L @
L^T`. Non-test consumer at `backend_impl.rs:4227-4236`.

### QR (REQ-3)

`pub fn gpu_qr_f32 in cusolver.rs` and `gpu_qr_f32_dev in
cusolver.rs` call `cusolverDnSgeqrf` to compute Householder vectors,
then `cusolverDnSorgqr` to materialise `Q` explicitly. `R` is the
upper-triangle of the `geqrf` output. The device-return variant
uses on-device transposes via `crate::kernels::gpu_transpose_2d`
to convert between row-major (ferrotorch's storage) and column-major
(cuSOLVER's expectation). Non-test consumer at `backend_impl.rs`.

### Solve / LU / lstsq (REQ-4, REQ-5, REQ-6)

`pub fn gpu_solve_f32 in cusolver.rs` (host) and
`gpu_solve_f32_dev in cusolver.rs` (device) call
`cusolverDnSgetrf` (LU) then `cusolverDnSgetrs` (back-substitute).
Consumer at `backend_impl.rs`.

`pub fn gpu_lu_factor_f32 in cusolver.rs` returns `(LU, pivots)`
from `cusolverDnSgetrf`. Consumer at `backend_impl.rs`.

`pub fn gpu_lstsq_f32 in cusolver.rs` calls `cusolverDnSgels`.
Consumer at `gpu_lstsq_f32 in backend_impl.rs`.

### Eig (non-symmetric, REQ-7) and eigh (symmetric, REQ-8)

`pub fn gpu_eig_f32 in cusolver.rs` returns `(eigenvalues:
Vec<Complex32>, eigenvectors: Vec<Complex32>)` via
`cusolverDnXgeev`. The `_dev` variant lives on-device. Consumer at
`backend_impl.rs`.

`pub fn gpu_eigh_f32 in cusolver.rs` returns
`(eigenvalues: Vec<f32>, eigenvectors: Vec<f32>)` via
`cusolverDnSsyevd`. The `gpu_eigvalsh_f32 in cusolver.rs` variant
returns only eigenvalues. Consumers at `backend_impl.rs`.

### `DnParamsHandle` RAII (REQ-9)

`impl DnParamsHandle in cusolver.rs` (line 2837) wraps
`cusolverDnParams_t`; `impl Drop for DnParamsHandle in
cusolver.rs` (line 2869) calls `cusolverDnDestroyParams` on drop,
preventing leaks. This mirrors R-DEV-4 — Rust's RAII replaces the
manual `cusolverDnDestroyParams` callsites scattered through
upstream's eig/SVD paths.

### Error policy (REQ-10)

Each cuSOLVER call writes `devInfo` to a small device buffer; after
the call the wrapper pulls the value back to host and checks for
non-zero. Non-zero → `Err(GpuError::Solver { code: <devInfo> })`.
Mirrors `aten/src/ATen/native/cuda/linalg/BatchLinearAlgebraLib.cpp:800`
where Cholesky checks `infos.scalar_type() == kInt` and throws on
non-zero.

### No-CUDA stubs (REQ-11)

The trailing block (lines 5297-5386 region) has a
`#[cfg(not(feature = "cuda"))]` stub for every cuda function,
returning `Err(GpuError::NoCudaFeature)`. Crate compiles in both
modes.

## Parity contract

`parity_ops = []` for this module. Reason: linalg ops are op-level
entries in `ferrotorch-core/src/linalg.rs`'s parity surface
(`linalg_svd`, `linalg_cholesky`, `linalg_qr`, `linalg_solve`,
`linalg_lu`, `linalg_eig`, `linalg_eigh`, `linalg_lstsq`); the
cuSOLVER dispatchers are reached transitively when the GPU backend
arm is selected.

Edge cases mirrored from upstream:

- **Singular A in solve**: `devInfo > 0` → `GpuError::Solver(...)`
  matching PyTorch's `torch._C._linalg_solve` raising
  `LinAlgError("singular matrix")`.
- **Non-PSD in Cholesky**: `devInfo > 0` → `GpuError::Solver(...)`
  matching PyTorch's `torch.linalg.cholesky` raising
  `LinAlgError("non-positive-definite")`.
- **Rank-deficient in lstsq**: cuSOLVER's `gels` does NOT report
  rank — the wrapper does not attempt to compute rank either, the
  caller in `ferrotorch-core::linalg` falls back to SVD-based
  lstsq for the explicit-rank path.
- **Complex eigenvalues from real input in eig**: `cusolverDnXgeev`
  requires a HOMOGENEOUS datatype set (A, W, VL, VR, computeType all
  the same — there is no mixed real-A / complex-W combination), so
  the real input matrix is promoted to a complex column-major buffer
  (imag = 0) via `gpu_real_to_complex_f32` / `gpu_real_to_complex_f64`
  and every datatype is `CUDA_C_32F`/`CUDA_C_64F`. Eigenvalues come
  back as interleaved f32/f64 (real, imag) pairs; the caller in
  `ferrotorch-core::linalg` packs them into `Complex32` tensors
  matching torch's `eig` return type. (#1687)
- **Empty matrix (n=0)**: returns empty `Vec`s; consistent with
  upstream's empty-matrix linalg semantics.

## Verification

Tests in `#[cfg(all(test, feature = "cuda"))] mod tests in
cusolver.rs` (17 functions):

- SVD reconstruction f32 / f64 (small, large, non-square).
- Cholesky factorisation + reconstruction.
- QR Q-orthogonality + R-upper-triangular checks.
- Solve `A @ x = b` round-trip.
- LU factor + reconstruct.
- Lstsq overdetermined / square / underdetermined.
- Eigh symmetric reconstruction.
- Eigvalsh against numpy reference.
- `_dev` variants vs. their host-return siblings.
- Singular matrix `devInfo > 0` propagation.

Smoke commands:

```bash
cargo test -p ferrotorch-gpu --features cuda cusolver:: 2>&1 | tail -3
cargo build -p ferrotorch-gpu --no-default-features 2>&1 | tail -3
```

Expected: 17 tests pass; no-cuda compile succeeds. `parity_ops = []`
— no per-op parity-sweep applies here; the op-level linalg parity
smoke (`linalg_svd`, etc.) in `ferrotorch-core` covers these
dispatchers indirectly.

## REQ status table

Per S5 (existing pub-API grandfather): every cusolver function is
consumed by `ferrotorch-gpu/src/backend_impl.rs` (the cuda backend's
linalg dispatch arms). Those arms are reached from
`ferrotorch-core/src/linalg.rs` when a tensor's linalg op routes to
GPU.

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn gpu_svd_f32 in cusolver.rs`, `pub fn gpu_svd_f64 in cusolver.rs`, `pub fn gpu_svd_f32_dev in cusolver.rs`, `pub fn gpu_svd_f64_dev in cusolver.rs` per upstream `aten/src/ATen/native/cuda/linalg/BatchLinearAlgebraLib.cpp:643::svd_cusolver`. Non-test consumer: `backend_impl.rs` (f32_dev) and `backend_impl.rs` (f64_dev). |
| REQ-2 | SHIPPED | impl: `pub fn gpu_cholesky_f32 in cusolver.rs`, `pub fn gpu_cholesky_f64 in cusolver.rs`, `pub fn gpu_cholesky_f32_dev in cusolver.rs`, `pub fn gpu_cholesky_f64_dev in cusolver.rs` per upstream `aten/src/ATen/native/cuda/linalg/BatchLinearAlgebraLib.cpp:800::cholesky_helper_cusolver`. Non-test consumer: `backend_impl.rs` (f32_dev) and `backend_impl.rs` (f64_dev). |
| REQ-3 | SHIPPED | impl: `pub fn gpu_qr_f32 in cusolver.rs`, `pub fn gpu_qr_f64 in cusolver.rs`, `pub fn gpu_qr_f32_dev in cusolver.rs`, `pub fn gpu_qr_f64_dev in cusolver.rs` per upstream `aten/src/ATen/native/cuda/linalg/BatchLinearAlgebraLib.cpp:1025::geqrf_cusolver`. Non-test consumer: `backend_impl.rs` (f32_dev) and `backend_impl.rs` (f64_dev). |
| REQ-4 | SHIPPED | impl: `pub fn gpu_solve_f32 in cusolver.rs`, `pub fn gpu_solve_f64 in cusolver.rs`, `pub fn gpu_solve_f32_dev in cusolver.rs`, `pub fn gpu_solve_f64_dev in cusolver.rs`. Non-test consumer: `gpu_solve_f64_dev in backend_impl.rs` (f32_dev) and `backend_impl.rs` (f64_dev). |
| REQ-5 | SHIPPED | impl: `pub fn gpu_lu_factor_f32 in cusolver.rs` and `pub fn gpu_lu_factor_f64 in cusolver.rs` per upstream `aten/src/ATen/native/cuda/linalg/BatchLinearAlgebraLib.cpp:1647::lu_factor_looped_cusolver`. Non-test consumer: `backend_impl.rs` (f32) and `backend_impl.rs` (f64); also documented at `ferrotorch-core/src/linalg.rs` ("dispatches to the native `gpu_lu_factor` kernel"). |
| REQ-6 | SHIPPED | impl: `pub fn gpu_lstsq_f32 in cusolver.rs` and `pub fn gpu_lstsq_f64 in cusolver.rs`. Non-test consumer: `gpu_lstsq_f64 in backend_impl.rs` (f32) and `backend_impl.rs` (f64). |
| REQ-7 | SHIPPED | impl: `pub fn gpu_eig_f32 in cusolver.rs`, `pub fn gpu_eig_f64 in cusolver.rs`, `pub fn gpu_eig_f32_dev in cusolver.rs`, `pub fn gpu_eig_f64_dev in cusolver.rs` per upstream `aten/src/ATen/native/cuda/linalg/BatchLinearAlgebraLib.cpp:1636::linalg_eig_cusolver_xgeev`. cusolverDnXgeev's homogeneous datatype contract is honored by promoting the real col-major A to complex col-major (imag=0) via `pub fn gpu_real_to_complex_f32 in kernels.rs` / `pub fn gpu_real_to_complex_f64 in kernels.rs` and passing all-`CUDA_C_32F`/`CUDA_C_64F` (mirrors `aten/src/ATen/native/cuda/linalg/CUDASolver.cpp:1865-1931`, the `c10::complex<scalar_t>` xgeev specializations). Non-test consumer: `eig_f32 in backend_impl.rs` (calls `gpu_eig_f32`) and `eig_f64 in backend_impl.rs` (calls `gpu_eig_f64`). |
| REQ-8 | SHIPPED | impl: `pub fn gpu_eigh_f32 in cusolver.rs`, `pub fn gpu_eigh_f64 in cusolver.rs`, `pub fn gpu_eigvalsh_f32 in cusolver.rs`, `pub fn gpu_eigvalsh_f64 in cusolver.rs` per upstream `aten/src/ATen/native/cuda/linalg/BatchLinearAlgebraLib.cpp:1509::linalg_eigh_cusolver`. Non-test consumer: `backend_impl.rs` (all four arms). |
| REQ-9 | SHIPPED | impl: `impl DnParamsHandle in cusolver.rs` (line 2837) + `impl Drop for DnParamsHandle in cusolver.rs` (line 2869). Non-test consumer: every eig / SVD path inside `cusolver.rs` that takes a `cusolverDnParams_t` opaque allocates a `DnParamsHandle` and lets RAII drop it. |
| REQ-10 | SHIPPED | impl: every cuSOLVER call in `cusolver.rs` checks `devInfo` and returns `Err(GpuError::Solver(...))` on non-zero. Non-test consumer: every caller in `map_gpu_err in backend_impl.rs` uses `.map_err(Self::map_gpu_err)?` to propagate the structured error to core. |
| REQ-11 | SHIPPED | impl: `#[cfg(not(feature = "cuda"))]` stubs near the bottom of `cusolver.rs` (lines 5297-5386 region) return `Err(GpuError::NoCudaFeature)`. Non-test consumer: the same `backend_impl.rs` dispatch arms under the no-cuda compile path. |
