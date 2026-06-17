//! Backward functions for linear algebra operations.
//!
//! Each struct captures the forward inputs and implements `GradFn` to
//! compute VJPs (vector-Jacobian products) for reverse-mode autodiff.
//!
//! ## REQ status (per `.design/ferrotorch-core/grad_fns/linalg.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`mm`) | SHIPPED | `mm_differentiable` + `MmBackward` consumed by `Tensor::mm` in `methods.rs` and pervasively across `ferrotorch-nn` (attention, lora, rnn, functional); parity `24/24 passed`. |
//! | REQ-2 (`bmm`) | SHIPPED | `bmm_differentiable` + `BmmBackward` consumed by `Tensor::bmm`, by `flex_attention.rs`, and by `ferrotorch-nn/src/attention.rs`; parity `8/8 passed`. |
//! | REQ-3 (`matmul`) | SHIPPED | `matmul_differentiable` + `MatmulBackward` consumed by `Tensor::matmul`, `ferrotorch-vision/src/models/swin.rs`, `einsum.rs`, and the forward-AD primal; parity `120/120 passed` under matmul-family `rtol=1e-4` (closes #1347). |
//! | REQ-4 (`linalg.matmul`) | SHIPPED | aliased to REQ-3 by upstream design; `Tensor::matmul` covers both since the Python API surface is identical; parity `120/120 passed`. |
//! | REQ-5 (`addmm`) | SHIPPED | `AddmmBackward` + `addmm_differentiable` (VJP `dself=beta*grad`, `dmat1=alpha*grad@mat2^T`, `dmat2=alpha*mat1^T@grad` per `derivatives.yaml:256`); FD-verified `grad_fns::linalg::tests::addmm_public_forward_is_grad_aware_and_matches_fd`; non-test consumer: the grad-aware `crate::linalg::addmm` forward delegates here. Closes #1583. |
//! | REQ-6 (`addbmm`) | SHIPPED | `AddbmmBackward` + `addbmm_differentiable` (per `derivatives.yaml:238`); FD-verified `grad_fns::linalg::tests::addbmm_public_forward_is_grad_aware_and_matches_fd`; non-test consumer: the grad-aware `crate::linalg::addbmm` forward delegates here. Closes #1583. |
//! | REQ-7 (`baddbmm`) | SHIPPED | `BaddbmmBackward` + `baddbmm_differentiable` (per `derivatives.yaml:359`); FD-verified `grad_fns::linalg::tests::baddbmm_public_forward_is_grad_aware_and_matches_fd`; non-test consumer: the grad-aware `crate::linalg::baddbmm` forward delegates here. Closes #1583. |
//! | REQ-8 (`addmv`) | SHIPPED | `AddmvBackward` + `addmv_differentiable` (per `derivatives.yaml:267`); FD-verified `grad_fns::linalg::tests::addmv_public_forward_is_grad_aware_and_matches_fd`; non-test consumer: the grad-aware `crate::linalg::addmv` forward delegates here. Closes #1583. |
//! | REQ-9 (`addr`) | SHIPPED | `AddrBackward` + `addr_differentiable` (per `derivatives.yaml:273`); FD-verified `grad_fns::linalg::tests::addr_public_forward_is_grad_aware_and_matches_fd`; non-test consumer: the grad-aware `crate::linalg::addr` forward delegates here. Closes #1583. |
//! | REQ-10 (`linalg.solve`) | SHIPPED | `LinalgSolveBackward` + `solve_differentiable` (VJP `gB = A^-T @ gX`, `gA = -gB @ X^T` per `FunctionsManual.cpp:6160`); FD-verified `tests/divergence_linalg_grad_audit.rs:solve_backward_*`; non-test consumer `tools/parity-sweep/runner/src/main.rs` `"linalg.solve"` arm (parity 24/24 non-skipped, 0 failed). Blocker #1345. |
//! | REQ-11 (`linalg.svd`) | SHIPPED | `SvdBackwardU`/`SvdBackwardS`/`SvdBackwardV` + `svd_differentiable` (real reduced-SVD VJP, F-matrix `E[i,j]=S²[j]-S²[i]` + symmetrized core + rectangular `m!=n` projectors, split across the U/S/Vh outputs and accumulated into `A.grad`) per `FunctionsManual.cpp:3605` (E `3770-3777`, core `3767-3797`, projectors `3799-3815`); verified vs LIVE torch float64 by `grad_fns::linalg::tests::svd_backward_{square_3x3,tall_4x3,wide_3x4,s_only_square_3x3,s_only_tall_4x3}_matches_torch`; non-test consumer: the grad-aware `crate::linalg::svd` forward delegates here when grad is enabled. Gauge caveat mirrors eigh #1584. Blocker #1577. |
//! | REQ-12 (`linalg.eig`) | SHIPPED | `EigBackwardW`/`EigBackwardV` + `eig_differentiable` (non-Hermitian complex VJP on the `[.,2]` real/imag layout via the private `c_matmul`/`c_conj_transpose`/`c_inverse`/`c_solve`/`c_econj_gap` toolkit: `VhgV=V^H gV`, unit-norm tangent proj `-V^H(V·real(diag VhgV))`, `Econj[i,j]=conj(L_j)-conj(L_i)`, `gA=real(solve(V^H,(diag(gL)+VhgV/Econj)V^H))`, split across the L/V outputs and accumulated into `A.grad`) per `FunctionsManual.cpp:3820` (`handle_r_to_c` real-part `derivatives.yaml:1740`); verified vs LIVE torch 2.11.0 float64 by `grad_fns::linalg::tests::{eig_backward_real_3x3,eig_backward_complex_pair_2x2,eig_backward_v_only_complex_pair_2x2}_matches_torch` at `1e-6`; non-test consumer: the grad-aware `crate::linalg::eig` forward (which also unit-norm-normalizes ferray's eigenvector columns to match torch's contract) delegates here when grad is enabled. EXACT for diagonalizable A; phase-invariant-loss gauge note (R-DEV-1) in `EigBackwardShared`. Closes #1345. |
//! | REQ-13 (`linalg.eigh`) | NOT-STARTED | forward exists; no backward. Blocker #1345. |
//! | REQ-14 (`linalg.eigvals`) | SHIPPED | `EigvalsBackward` + `eigvals_differentiable` (non-Hermitian eigenvalues-only complex VJP `gA=real(solve(V^H, diag(gL) V^H))` — the `!gV.defined()` shortcut of `linalg_eig_backward`, on the `[.,2]` layout via the private complex toolkit) per `FunctionsManual.cpp:3857-3862` (`handle_r_to_c` real-part `derivatives.yaml:1740`); verified vs LIVE torch 2.11.0 float64 by `grad_fns::linalg::tests::{eigvals_backward_real_3x3,eigvals_backward_complex_pair_2x2}_matches_torch` at `1e-6`; non-test consumer: the grad-aware `crate::linalg::eigvals` forward delegates here when grad is enabled (eigenvectors from `crate::linalg::eig` under `no_grad`). EXACT for diagonalizable A. Closes #1345. |
//! | REQ-15 (`linalg.eigvalsh`) | NOT-STARTED | forward exists; no backward. Blocker #1345. |
//! | REQ-16 (`linalg.qr`) | SHIPPED | `QrBackwardQ`/`QrBackwardR` + `qr_differentiable` (reduced, m≥n; real `linalg_qr_backward` VJP split across the Q/R outputs, accumulated into `A.grad`) per `FunctionsManual.cpp:4166`; FD-verified `grad_fns::linalg::tests::qr_backward_matches_finite_difference_square` and `qr_backward_q_only_and_r_only`; non-test consumer: the grad-aware `crate::linalg::qr` forward delegates here when grad is enabled. Blocker #1345. |
//! | REQ-17 (`linalg.cholesky`) | SHIPPED | `CholeskyBackward` + `cholesky_differentiable` (Phi-symmetrisation VJP `L^{-T} Φ(tril(L^T gL)) L^{-1}`) per `FunctionsManual.cpp:2048`; FD-verified `grad_fns::linalg::tests::cholesky_backward_matches_finite_difference` (symmetric-FD + symmetry check); non-test consumer: the grad-aware `crate::linalg::cholesky` forward delegates here when grad is enabled. Blocker #1345. |
//! | REQ-18 (`linalg.inv`) | SHIPPED | `LinalgInvBackward` + `inv_differentiable` (VJP `dA = -Y^T @ grad @ Y^T` per `derivatives.yaml:917`); FD-verified `tests/divergence_linalg_grad_audit.rs:inv_backward_matches_finite_difference`; non-test consumer `tools/parity-sweep/runner/src/main.rs` `"linalg.inv"` arm. Blocker #1345. |
//! | REQ-19 (`linalg.pinv`) | SHIPPED | `PinvBackward` + `pinv_differentiable` (PyTorch `pinv_backward` full-rank algebraic VJP) with a resident CUDA f32/f64 branch built from tensor `mm`/transpose/add/sub/neg; CUDA forward composes `svd` + resident `amax`/compare/where/diag so tracked `pinv` does not detach or host-round-trip. FD/CPU-vs-CUDA verified in `tests/audit_core146_linalg_autograd.rs`. |
//! | REQ-20 (`linalg.det`) | SHIPPED | `LinalgDetBackward` + `det_differentiable` (forward stores only `A` + `det(A)`, then lazily computes the VJP so singular tracked forwards match PyTorch; ordinary singular backward follows the LU-perturbed zero branch, `create_graph` uses an adjugate/cofactor fallback) per `FunctionsManual.cpp:4373`; FD-verified `tests/divergence_linalg_grad_audit.rs:det_backward_matches_finite_difference` and singular parity in `tests/audit_core187_det_slogdet_singular.rs`; non-test consumer `tools/parity-sweep/runner/src/main.rs` `"linalg.det"` arm. Blocker #1345. |
//! | REQ-21 (`linalg.slogdet`) | SHIPPED | `SlogdetBackward` + `slogdet_differentiable` (forward stores only `A`, so singular tracked forwards return `(0, -inf)` like PyTorch; real-case VJP lazily computes `grad_logabsdet * inv(A)^T`, with nonsmooth singular fallback only in ordinary backward) per `FunctionsManual.cpp:4471` + `derivatives.yaml:559`; FD-verified `grad_fns::linalg::tests::slogdet_backward_matches_finite_difference` and singular forward parity in `tests/audit_core187_det_slogdet_singular.rs`; non-test consumer: the grad-aware `crate::linalg::slogdet` forward delegates here when grad is enabled. Blocker #1345. |
//! | REQ-22 (`linalg.lstsq`) | SHIPPED | `LstsqBackward` + `lstsq_differentiable`/`lstsq_solve_differentiable` mirror `FunctionsManual.cpp:4012` for both differentiable outputs (`solution`, `residuals`; integer rank/singular values non-diff). CUDA f32/f64 solution and residuals stay resident; tests cover solution VJP vs CPU and residual VJP vs finite difference. Explicit CPU/CUDA driver contract mirrors `torch.linalg.lstsq`. |
//! | REQ-23 (`linalg.norm`) | NOT-STARTED | forward exists; no backward. Blocker #1345. |
//! | REQ-24 (`linalg.matrix_rank`) | NOT-STARTED | forward exists; no backward. Blocker #1345. |
//! | REQ-25 (`linalg.cross`) | NOT-STARTED | forward exists; no backward. Blocker #1345. |
//! | REQ-26 (`linalg.householder_product`) | SHIPPED | `HouseholderProductBackward` + `householder_product_differentiable` (real reflector-recursion VJP — `tril(V,-1)` unit-diag, `sigma_j=tau_j/(tau_j‖input[:,j]‖²-1)`, `K=Q_full@grad^T`, per-`i` `update_grad` + `K←H_{i+1}^{-1}KH_i` advance, `grad_V=tril(-1)`) per `FunctionsManual.cpp:5544`; verified vs LIVE torch 2.11.0 float64 by `grad_fns::linalg::tests::householder_product_backward_{square_3x3,tall_4x3,tall_4x2}_matches_torch` (V.grad + tau.grad at `1e-9`); non-test consumer: the now-`[m,k]`-shaped grad-aware `crate::linalg::householder_product` forward delegates here (with `crate::linalg::householder_product_full` giving the `[m,m]` reconstruction). Residual #1345 = eig/eigvals (complex). |
//! | REQ-27 (`linalg.lu`) | SHIPPED | `LuBackwardL`/`LuBackwardU` + `lu_differentiable` (PyTorch square/wide/tall block formulas per `FunctionsManual.cpp:6854`); FD-verified for CPU wide and CUDA-resident against CPU in `tests/audit_core146_linalg_autograd.rs`; non-test consumer: grad-aware `crate::linalg::lu` delegates here when grad is enabled. |
//! | REQ-28 (`linalg.lu_factor`) | SHIPPED | `LuFactorBackward` + `lu_factor_differentiable` splits packed `LU` with `grad.narrow(-1,0,k)` / `grad.narrow(-2,0,k)` per `FunctionsManual.cpp:6960`; FD-verified for CPU tall and CUDA-resident against CPU in `tests/audit_core146_linalg_autograd.rs`; non-test consumer: grad-aware `crate::linalg::lu_factor` delegates here when grad is enabled. |
//! | REQ-29 (`trace`) | SHIPPED | `TraceBackward` + `trace_differentiable` (VJP `dA = grad * I` per `derivatives.yaml:1785`), forward `crate::linalg::trace`; FD-verified `tests/divergence_linalg_grad_audit.rs:trace_backward_matches_finite_difference`; non-test consumer `tools/parity-sweep/runner/src/main.rs` `"trace"` arm (parity 8/8, 0 failed). Blocker #1345. |
//! | REQ-30 (`diagonal`) | SHIPPED | `DiagonalBackward` + `diagonal_differentiable` (VJP scatters grad onto the offset-th diagonal per `derivatives.yaml:573` `diagonal_backward_symint`); FD-verified `grad_fns::linalg::tests::diagonal_public_forward_is_grad_aware_and_matches_fd`; non-test consumer: the now-grad-aware `crate::linalg::diagonal` forward delegates here. Closes #1583. |
//! | REQ-31 (`diag`) | SHIPPED | `DiagBackward` + `diag_differentiable` (adjoint of the 0/1 selection: gather for 1-D, scatter for 2-D); FD-verified `grad_fns::linalg::tests::diag_extract_public_forward_is_grad_aware_and_matches_fd` + `diag_construct_public_forward_is_grad_aware_and_matches_fd`; non-test consumer: the now-grad-aware `crate::ops::tensor_ops::diag` forward delegates here. Closes #1583. |
//! | REQ-32 (`tril`) | SHIPPED | `TriangularBackward` + `tril_differentiable` (VJP masks grad by the kept lower triangle per `derivatives.yaml:1805` `grad.tril_symint`); FD-verified `grad_fns::linalg::tests::tril_public_forward_is_grad_aware_and_matches_fd`; non-test consumer: the now-grad-aware `crate::ops::tensor_ops::tril` forward delegates here. Closes #1583. |
//! | REQ-33 (`triu`) | SHIPPED | `triu_differentiable` (sharing `TriangularBackward`; VJP masks grad by the kept upper triangle per `derivatives.yaml:1809` `grad.triu_symint`); FD-verified `grad_fns::linalg::tests::triu_public_forward_is_grad_aware_and_matches_fd`; non-test consumer: the now-grad-aware `crate::ops::tensor_ops::triu` forward delegates here. Closes #1583. |
//! | REQ-34 (`kron`) | SHIPPED | `KronBackward` + `kron_differentiable` (per-Kron-block VJP per `LinearAlgebra.cpp:3530` `kron`); FD-verified `grad_fns::linalg::tests::kron_public_forward_is_grad_aware_and_matches_fd`; non-test consumer: the new grad-aware `crate::linalg::kron` forward delegates here. Closes #1583. |
//! | REQ-35 (`outer`) | SHIPPED | `outer_differentiable` mirrors PyTorch's composite `self.reshape({m,1}) * vec2` graph (`LinearAlgebra.cpp:1337-1342`) instead of a CPU-only closed-form VJP; gradients flow through `MulBackward`/`ReshapeBackward` on CPU and CUDA. Forward `crate::linalg::outer`; FD-verified `tests/divergence_linalg_grad_audit.rs:outer_backward_matches_finite_difference`; non-test consumer `tools/parity-sweep/runner/src/main.rs` `"outer"` arm. |

use std::any::TypeId;
use std::sync::Arc;

use crate::autograd::autocast_ops::{AutocastCategory, autocast_guard};
use crate::autograd::no_grad::is_grad_enabled;
use crate::dtype::{DType, Element, Float};
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::gpu_dispatch::{GpuBackend, GpuBufferHandle, gpu_backend};
use crate::linalg as linalg_fwd;
use crate::ops::linalg::{self, mm, transpose};
use crate::storage::TensorStorage;
use crate::tensor::{GradFn, Tensor};

/// Type alias for a pair of optional tensor gradients (used by matmul backward).
type GradPair<T> = FerrotorchResult<(Option<Tensor<T>>, Option<Tensor<T>>)>;

/// Returns `true` if `T` is `f32`.
#[inline]
fn is_f32<T: Float>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f32>()
}

/// Returns `true` if `T` is `f64`.
#[inline]
fn is_f64<T: Float>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f64>()
}

/// Returns `true` if `T` is `bf16` (`half::bf16`).
///
/// Used by the broadcast-bmm dispatcher (#1543 / GH forecast-bio/ferrotorch#25)
/// to route 3D × 2D / 2D × 3D / ND × ND bf16 matmul through the cuBLAS
/// `gpu_matmul_bf16_bf16_strided_batched` kernel (bf16 in/out, FP32
/// accumulator — standard ~1.5e-3 floor) instead of the CPU `broadcast_matmul`
/// round-trip that was forcing a 50× precision regression on the ViT-shape
/// `(1, 200, 4096) @ (4096, 768)` matmul.
#[inline]
fn is_bf16<T: Float>() -> bool {
    TypeId::of::<T>() == TypeId::of::<half::bf16>()
}

#[inline]
fn is_f16<T: Float>() -> bool {
    TypeId::of::<T>() == TypeId::of::<half::f16>()
}

fn cuda_matmul_same_dtype<T: Float>(
    backend: &dyn GpuBackend,
    a: &GpuBufferHandle,
    b: &GpuBufferHandle,
    m: usize,
    k: usize,
    n: usize,
    op: &'static str,
) -> FerrotorchResult<GpuBufferHandle> {
    if is_f32::<T>() {
        backend.matmul_f32(a, b, m, k, n)
    } else if is_f64::<T>() {
        backend.matmul_f64(a, b, m, k, n)
    } else if is_bf16::<T>() {
        backend.matmul_bf16_bf16(a, b, m, k, n)
    } else if is_f16::<T>() {
        backend.matmul_f16_f16(a, b, m, k, n)
    } else {
        Err(FerrotorchError::NotImplementedOnCuda { op })
    }
}

fn cuda_matmul_nt_same_dtype<T: Float>(
    backend: &dyn GpuBackend,
    a: &GpuBufferHandle,
    b: &GpuBufferHandle,
    m: usize,
    k: usize,
    n: usize,
    op: &'static str,
) -> FerrotorchResult<GpuBufferHandle> {
    if is_f32::<T>() {
        backend.matmul_f32_nt(a, b, m, k, n)
    } else if is_f64::<T>() {
        backend.matmul_f64_nt(a, b, m, k, n)
    } else if is_bf16::<T>() {
        backend.matmul_bf16_bf16_nt(a, b, m, k, n)
    } else if is_f16::<T>() {
        backend.matmul_f16_f16_nt(a, b, m, k, n)
    } else {
        Err(FerrotorchError::NotImplementedOnCuda { op })
    }
}

fn cuda_transpose_2d_same_dtype<T: Float>(
    backend: &dyn GpuBackend,
    a: &GpuBufferHandle,
    m: usize,
    n: usize,
    op: &'static str,
) -> FerrotorchResult<GpuBufferHandle> {
    if is_f32::<T>() {
        backend.transpose_2d_f32(a, m, n)
    } else if is_f64::<T>() {
        backend.transpose_2d_f64(a, m, n)
    } else if is_bf16::<T>() {
        backend.transpose_2d_bf16(a, m, n)
    } else if is_f16::<T>() {
        backend.transpose_2d_f16(a, m, n)
    } else {
        Err(FerrotorchError::NotImplementedOnCuda { op })
    }
}

fn cuda_permute_0213_same_dtype<T: Float>(
    backend: &dyn GpuBackend,
    a: &GpuBufferHandle,
    d0: usize,
    d1: usize,
    d2: usize,
    d3: usize,
    op: &'static str,
) -> FerrotorchResult<GpuBufferHandle> {
    if is_f32::<T>() {
        backend.permute_0213_f32(a, d0, d1, d2, d3)
    } else if is_f64::<T>() {
        backend.permute_0213_f64(a, d0, d1, d2, d3)
    } else if is_bf16::<T>() {
        backend.permute_0213_bf16(a, d0, d1, d2, d3)
    } else if is_f16::<T>() {
        backend.permute_0213_f16(a, d0, d1, d2, d3)
    } else {
        Err(FerrotorchError::NotImplementedOnCuda { op })
    }
}

fn cuda_broadcast_add_same_dtype<T: Float>(
    backend: &dyn GpuBackend,
    a: &GpuBufferHandle,
    b: &GpuBufferHandle,
    a_shape: &[usize],
    b_shape: &[usize],
    out_shape: &[usize],
    op: &'static str,
) -> FerrotorchResult<GpuBufferHandle> {
    if is_f32::<T>() {
        backend.broadcast_add_f32(a, b, a_shape, b_shape, out_shape)
    } else if is_f64::<T>() {
        backend.broadcast_add_f64(a, b, a_shape, b_shape, out_shape)
    } else if is_bf16::<T>() {
        backend.broadcast_add_bf16(a, b, a_shape, b_shape, out_shape)
    } else if is_f16::<T>() {
        backend.broadcast_add_f16(a, b, a_shape, b_shape, out_shape)
    } else {
        Err(FerrotorchError::NotImplementedOnCuda { op })
    }
}

fn cuda_broadcast_mul_same_dtype<T: Float>(
    backend: &dyn GpuBackend,
    a: &GpuBufferHandle,
    b: &GpuBufferHandle,
    a_shape: &[usize],
    b_shape: &[usize],
    out_shape: &[usize],
    op: &'static str,
) -> FerrotorchResult<GpuBufferHandle> {
    if is_f32::<T>() {
        backend.broadcast_mul_f32(a, b, a_shape, b_shape, out_shape)
    } else if is_f64::<T>() {
        backend.broadcast_mul_f64(a, b, a_shape, b_shape, out_shape)
    } else if is_bf16::<T>() {
        backend.broadcast_mul_bf16(a, b, a_shape, b_shape, out_shape)
    } else if is_f16::<T>() {
        backend.broadcast_mul_f16(a, b, a_shape, b_shape, out_shape)
    } else {
        Err(FerrotorchError::NotImplementedOnCuda { op })
    }
}

fn cuda_sum_axis_same_dtype<T: Float>(
    backend: &dyn GpuBackend,
    a: &GpuBufferHandle,
    shape: &[usize],
    axis: usize,
    op: &'static str,
) -> FerrotorchResult<GpuBufferHandle> {
    if is_f32::<T>() {
        backend.sum_axis_f32(a, shape, axis)
    } else if is_f64::<T>() {
        backend.sum_axis_f64(a, shape, axis)
    } else if is_bf16::<T>() {
        backend.sum_axis_bf16_bf16(a, shape, axis)
    } else if is_f16::<T>() {
        backend.sum_axis_f16(a, shape, axis)
    } else {
        Err(FerrotorchError::NotImplementedOnCuda { op })
    }
}

fn ensure_same_device<T: Float>(expected: &Tensor<T>, got: &Tensor<T>) -> FerrotorchResult<()> {
    if expected.device() != got.device() {
        return Err(FerrotorchError::DeviceMismatch {
            expected: expected.device(),
            got: got.device(),
        });
    }
    Ok(())
}

fn ensure_rank<T: Float>(
    op: &str,
    name: &str,
    tensor: &Tensor<T>,
    expected: usize,
) -> FerrotorchResult<()> {
    if tensor.ndim() != expected {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "{op}: {name} must be {expected}-D, got shape {:?}",
                tensor.shape()
            ),
        });
    }
    Ok(())
}

fn validate_mm_operands<T: Float>(
    op: &str,
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> FerrotorchResult<(usize, usize, usize)> {
    ensure_same_device(a, b)?;
    ensure_rank(op, "mat1", a, 2)?;
    ensure_rank(op, "mat2", b, 2)?;

    let m = a.shape()[0];
    let k = a.shape()[1];
    let b_rows = b.shape()[0];
    let n = b.shape()[1];
    if k != b_rows {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "{op}: mat1 and mat2 shapes cannot be multiplied ({m}x{k} and {b_rows}x{n})"
            ),
        });
    }
    Ok((m, k, n))
}

fn validate_mm_bt_operands<T: Float>(
    op: &str,
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> FerrotorchResult<(usize, usize, usize)> {
    ensure_same_device(a, b)?;
    ensure_rank(op, "mat1", a, 2)?;
    ensure_rank(op, "mat2", b, 2)?;

    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[0];
    let b_k = b.shape()[1];
    if k != b_k {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "{op}: mat1 and mat2^T shapes cannot be multiplied ({m}x{k} and {n}x{b_k})"
            ),
        });
    }
    Ok((m, k, n))
}

fn validate_mv_operands<T: Float>(
    op: &str,
    a: &Tensor<T>,
    x: &Tensor<T>,
) -> FerrotorchResult<(usize, usize)> {
    ensure_same_device(a, x)?;
    ensure_rank(op, "mat", a, 2)?;
    ensure_rank(op, "vec", x, 1)?;

    let m = a.shape()[0];
    let k = a.shape()[1];
    let x_len = x.shape()[0];
    if x_len != k {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!("{op}: size mismatch, mat is ({m}x{k}), vec is ({x_len})"),
        });
    }
    Ok((m, k))
}

fn validate_dot_operands<T: Float>(
    op: &str,
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> FerrotorchResult<usize> {
    ensure_same_device(a, b)?;
    ensure_rank(op, "input", a, 1)?;
    ensure_rank(op, "other", b, 1)?;

    let n = a.shape()[0];
    let b_len = b.shape()[0];
    if n != b_len {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "{op}: inconsistent tensor size, expected {n} elements but got {b_len}"
            ),
        });
    }
    Ok(n)
}

fn validate_linear_fused_operands<T: Float>(
    input: &Tensor<T>,
    weight: &Tensor<T>,
    bias: Option<&Tensor<T>>,
) -> FerrotorchResult<(usize, usize, usize)> {
    ensure_same_device(input, weight)?;
    if let Some(b) = bias {
        ensure_same_device(input, b)?;
    }

    ensure_rank("linear_fused", "input", input, 2)?;
    ensure_rank("linear_fused", "weight", weight, 2)?;

    let m = input.shape()[0];
    let k = input.shape()[1];
    let n = weight.shape()[0];
    let weight_k = weight.shape()[1];
    if k != weight_k {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "linear_fused: input and weight shapes cannot be multiplied ({m}x{k} and {n}x{weight_k})"
            ),
        });
    }

    if let Some(b) = bias {
        ensure_rank("linear_fused", "bias", b, 1)?;
        let bias_len = b.shape()[0];
        if bias_len != n {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "linear_fused: bias length {bias_len} does not match out_features {n}"
                ),
            });
        }
    }

    Ok((m, k, n))
}

/// GPU-native matmul backward for all CUDA floating dtypes supported by `mm`.
/// dA = grad_C @ B^T, dB = A^T @ grad_C — all on GPU, no CPU roundtrip.
fn mm_backward_gpu<T: Float>(grad_output: &Tensor<T>, a: &Tensor<T>, b: &Tensor<T>) -> GradPair<T> {
    let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
    let go_h = grad_output.gpu_handle()?;
    let m = grad_output.shape()[0];
    let n = grad_output.shape()[1];

    let grad_a = if a.requires_grad() {
        let k = b.shape()[0];
        let b_h = b.gpu_handle()?;
        let result_h = cuda_matmul_nt_same_dtype::<T>(backend, go_h, b_h, m, n, k, "MmBackward")?;
        Some(Tensor::from_storage(
            TensorStorage::gpu(result_h),
            vec![m, k],
            false,
        )?)
    } else {
        None
    };

    let grad_b = if b.requires_grad() {
        let k = a.shape()[1];
        let a_h = a.gpu_handle()?;
        let at_h = cuda_transpose_2d_same_dtype::<T>(backend, a_h, m, k, "MmBackward")?;
        let result_h = cuda_matmul_same_dtype::<T>(backend, &at_h, go_h, k, m, n, "MmBackward")?;
        Some(Tensor::from_storage(
            TensorStorage::gpu(result_h),
            vec![k, n],
            false,
        )?)
    } else {
        None
    };

    Ok((grad_a, grad_b))
}

// ---------------------------------------------------------------------------
// MmBackward — C = A @ B  (2D x 2D)
// ---------------------------------------------------------------------------

/// Backward for matrix-matrix multiply: `C = mm(A, B)`.
///
/// VJP formulas:
/// - `dA = grad_C @ B^T`
/// - `dB = A^T @ grad_C`
#[derive(Debug)]
pub struct MmBackward<T: Float> {
    a: Tensor<T>,
    b: Tensor<T>,
}

impl<T: Float> MmBackward<T> {
    pub fn new(a: Tensor<T>, b: Tensor<T>) -> Self {
        Self { a, b }
    }
}

impl<T: Float> GradFn<T> for MmBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // GPU-native path for every dtype that `mm_differentiable` forwards
        // through cuBLAS. PyTorch keeps f16/bf16 VJPs resident and returns
        // same-dtype CUDA gradients.
        if grad_output.is_cuda() {
            let (ga, gb) = mm_backward_gpu(grad_output, &self.a, &self.b)?;
            return Ok(vec![ga, gb]);
        }

        // CPU path.
        let grad_a = if self.a.requires_grad() {
            let gc_data = grad_output.data()?;
            let b_data = self.b.data()?;
            let m = grad_output.shape()[0];
            let n = grad_output.shape()[1];
            let k = self.b.shape()[0];
            let result = crate::ops::linalg::mm_raw_bt(gc_data, b_data, m, n, k);
            Some(Tensor::from_storage(
                TensorStorage::cpu(result),
                vec![m, k],
                false,
            )?)
        } else {
            None
        };

        let grad_b = if self.b.requires_grad() {
            let a_data = self.a.data()?;
            let gc_data = grad_output.data()?;
            let m = self.a.shape()[0];
            let k = self.a.shape()[1];
            let n = grad_output.shape()[1];
            let result = crate::ops::linalg::mm_raw_at(a_data, gc_data, k, m, n);
            Some(Tensor::from_storage(
                TensorStorage::cpu(result),
                vec![k, n],
                false,
            )?)
        } else {
            None
        };

        Ok(vec![grad_a, grad_b])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.a, &self.b]
    }

    fn name(&self) -> &'static str {
        "MmBackward"
    }
}

// ---------------------------------------------------------------------------
// MvBackward — y = A @ x  (2D x 1D)
// ---------------------------------------------------------------------------

/// Backward for matrix-vector multiply: `y = mv(A, x)`.
///
/// VJP formulas:
/// - `dA = outer(grad_y, x)`   — i.e. `dA[i,j] = grad_y[i] * x[j]`
/// - `dx = A^T @ grad_y`
#[derive(Debug)]
pub struct MvBackward<T: Float> {
    a: Tensor<T>,
    x: Tensor<T>,
}

impl<T: Float> MvBackward<T> {
    pub fn new(a: Tensor<T>, x: Tensor<T>) -> Self {
        Self { a, x }
    }
}

impl<T: Float> GradFn<T> for MvBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // §3 GPU-native path (PyTorch parity: backward runs on same device as forward).
        // dA = outer(grad_y, x) = matmul(grad_y.reshape(m,1), x.reshape(1,k)).
        // dx = A^T @ grad_y = matmul(A^T, grad_y.reshape(m,1)).
        if grad_output.is_cuda() {
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let m = self.a.shape()[0];
            let k = self.a.shape()[1];

            let grad_a = if self.a.requires_grad() {
                let go_h = grad_output.gpu_handle()?;
                let x_h = self.x.gpu_handle()?;
                // outer(grad_y, x): treat grad_y as (m,1) and x as (1,k) → matmul gives (m,k).
                let result_h =
                    cuda_matmul_same_dtype::<T>(backend, go_h, x_h, m, 1, k, "MvBackward")?;
                Some(Tensor::from_storage(
                    TensorStorage::gpu(result_h),
                    vec![m, k],
                    false,
                )?)
            } else {
                None
            };

            let grad_x = if self.x.requires_grad() {
                let a_h = self.a.gpu_handle()?;
                let go_h = grad_output.gpu_handle()?;
                // dx = A^T @ grad_y: transpose A (m,k) → (k,m), then multiply
                // by grad_y treated as (m,1). The output storage is (k,1),
                // which is the same contiguous layout as the vector shape (k).
                let at_h = cuda_transpose_2d_same_dtype::<T>(backend, a_h, m, k, "MvBackward")?;
                let result_h =
                    cuda_matmul_same_dtype::<T>(backend, &at_h, go_h, k, m, 1, "MvBackward")?;
                Some(Tensor::from_storage(
                    TensorStorage::gpu(result_h),
                    vec![k],
                    false,
                )?)
            } else {
                None
            };

            return Ok(vec![grad_a, grad_x]);
        }

        // grad_output is shape (M,) — the upstream gradient on y.
        let grad_a = if self.a.requires_grad() {
            // dA = outer(grad_y, x): shape (M, K)
            let grad_data = grad_output.data()?;
            let x_data = self.x.data()?;
            let m = grad_data.len();
            let k = x_data.len();
            let mut outer = vec![<T as num_traits::Zero>::zero(); m * k];
            for i in 0..m {
                for j in 0..k {
                    outer[i * k + j] = grad_data[i] * x_data[j];
                }
            }
            Some(Tensor::from_storage(
                TensorStorage::cpu(outer),
                vec![m, k],
                false,
            )?)
        } else {
            None
        };

        let grad_x = if self.x.requires_grad() {
            // dx = A^T @ grad_y
            let at = transpose(&self.a)?;
            Some(linalg::mv(&at, grad_output)?)
        } else {
            None
        };

        Ok(vec![grad_a, grad_x])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.a, &self.x]
    }

    fn name(&self) -> &'static str {
        "MvBackward"
    }
}

// ---------------------------------------------------------------------------
// DotBackward — s = dot(a, b)  (1D x 1D -> scalar)
// ---------------------------------------------------------------------------

/// Backward for dot product: `s = dot(a, b)`.
///
/// VJP formulas:
/// - `da = grad_s * b`
/// - `db = grad_s * a`
#[derive(Debug)]
pub struct DotBackward<T: Float> {
    a: Tensor<T>,
    b: Tensor<T>,
}

impl<T: Float> DotBackward<T> {
    pub fn new(a: Tensor<T>, b: Tensor<T>) -> Self {
        Self { a, b }
    }
}

impl<T: Float> GradFn<T> for DotBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // §3 GPU-native path: da = grad_s * b, db = grad_s * a. Use the
        // broadcast-multiply kernels so even scalar cotangents stay resident.
        if grad_output.is_cuda() {
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let go_h = grad_output.gpu_handle()?;
            let scalar_shape: [usize; 0] = [];

            let grad_a = if self.a.requires_grad() {
                let b_h = self.b.gpu_handle()?;
                let result_h = cuda_broadcast_mul_same_dtype::<T>(
                    backend,
                    go_h,
                    b_h,
                    &scalar_shape,
                    self.b.shape(),
                    self.b.shape(),
                    "DotBackward",
                )?;
                Some(Tensor::from_storage(
                    TensorStorage::gpu(result_h),
                    self.a.shape().to_vec(),
                    false,
                )?)
            } else {
                None
            };

            let grad_b = if self.b.requires_grad() {
                let a_h = self.a.gpu_handle()?;
                let result_h = cuda_broadcast_mul_same_dtype::<T>(
                    backend,
                    go_h,
                    a_h,
                    &scalar_shape,
                    self.a.shape(),
                    self.a.shape(),
                    "DotBackward",
                )?;
                Some(Tensor::from_storage(
                    TensorStorage::gpu(result_h),
                    self.b.shape().to_vec(),
                    false,
                )?)
            } else {
                None
            };

            return Ok(vec![grad_a, grad_b]);
        }

        let s = grad_output.item()?;

        let grad_a = if self.a.requires_grad() {
            let b_data = self.b.data()?;
            let result: Vec<T> = b_data.iter().map(|&v| s * v).collect();
            Some(Tensor::from_storage(
                TensorStorage::cpu(result),
                self.a.shape().to_vec(),
                false,
            )?)
        } else {
            None
        };

        let grad_b = if self.b.requires_grad() {
            let a_data = self.a.data()?;
            let result: Vec<T> = a_data.iter().map(|&v| s * v).collect();
            Some(Tensor::from_storage(
                TensorStorage::cpu(result),
                self.b.shape().to_vec(),
                false,
            )?)
        } else {
            None
        };

        Ok(vec![grad_a, grad_b])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.a, &self.b]
    }

    fn name(&self) -> &'static str {
        "DotBackward"
    }
}

// ---------------------------------------------------------------------------
// batch_transpose — swap dims 1 and 2 of a 3D tensor
// ---------------------------------------------------------------------------

/// Transpose dims 1 and 2 of a 3D tensor: `[batch, r, c]` → `[batch, c, r]`.
///
/// This is a data rearrangement (not a view) that works on any device.
/// Used by `BmmBackward` to compute `bmm(grad_C, B^T)` and `bmm(A^T, grad_C)`.
fn batch_transpose<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    // Use permute + contiguous to transpose dims 1 and 2 entirely on-device.
    // This avoids the GPU→CPU→GPU roundtrip that dominated BmmBackward cost.
    input.permute(&[0, 2, 1])?.contiguous()
}

// ---------------------------------------------------------------------------
// BmmBackward — C[b] = A[b] @ B[b]  (3D batched matmul)
// ---------------------------------------------------------------------------

/// Backward for batched matrix-matrix multiply: `C = bmm(A, B)`.
///
/// VJP formulas (per batch element `b`):
/// - `dA[b] = grad_C[b] @ B[b]^T`
/// - `dB[b] = A[b]^T @ grad_C[b]`
#[derive(Debug)]
pub struct BmmBackward<T: Float> {
    a: Tensor<T>,
    b: Tensor<T>,
}

impl<T: Float> BmmBackward<T> {
    pub fn new(a: Tensor<T>, b: Tensor<T>) -> Self {
        Self { a, b }
    }
}

impl<T: Float> GradFn<T> for BmmBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // PyTorch approach: grad_A = bmm(grad_C, B^T), grad_B = bmm(A^T, grad_C)
        // where ^T transposes dims 1 and 2. Uses the same GPU-aware bmm path.
        let grad_a = if self.a.requires_grad() {
            let bt = batch_transpose(&self.b)?;
            Some(crate::autograd::no_grad::no_grad(|| bmm(grad_output, &bt))?)
        } else {
            None
        };

        let grad_b = if self.b.requires_grad() {
            let at = batch_transpose(&self.a)?;
            Some(crate::autograd::no_grad::no_grad(|| bmm(&at, grad_output))?)
        } else {
            None
        };

        Ok(vec![grad_a, grad_b])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.a, &self.b]
    }

    fn name(&self) -> &'static str {
        "BmmBackward"
    }
}

// ---------------------------------------------------------------------------
// MatmulBackward — dispatches based on input shapes
// ---------------------------------------------------------------------------

/// Backward for the general `matmul` dispatcher.
///
/// Internally delegates to `MmBackward`, `MvBackward`, `DotBackward`,
/// or the vm path depending on the operand ranks at forward time.
#[derive(Debug)]
pub struct MatmulBackward<T: Float> {
    a: Tensor<T>,
    b: Tensor<T>,
}

impl<T: Float> MatmulBackward<T> {
    pub fn new(a: Tensor<T>, b: Tensor<T>) -> Self {
        Self { a, b }
    }
}

impl<T: Float> GradFn<T> for MatmulBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        match (self.a.ndim(), self.b.ndim()) {
            (2, 2) => {
                let inner = MmBackward::new(self.a.clone(), self.b.clone());
                inner.backward(grad_output)
            }
            (2, 1) => {
                let inner = MvBackward::new(self.a.clone(), self.b.clone());
                inner.backward(grad_output)
            }
            (1, 1) => {
                let inner = DotBackward::new(self.a.clone(), self.b.clone());
                inner.backward(grad_output)
            }
            (1, 2) => {
                // vm: y = a @ B where a is (K,), B is (K,N), y is (N,)
                // §3 GPU-native path: da = B @ grad_y via GEMM; dB = outer(a, grad_y) via GEMM.
                if grad_output.is_cuda() {
                    let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
                    let k = self.a.numel();
                    let n = grad_output.numel();

                    let grad_a = if self.a.requires_grad() {
                        // da = B @ grad_y: treat grad_y as (1,N) and use the
                        // fused NT GEMM to produce contiguous (K,1) storage.
                        let b_h = self.b.gpu_handle()?;
                        let go_h = grad_output.gpu_handle()?;
                        let result_h = cuda_matmul_nt_same_dtype::<T>(
                            backend,
                            b_h,
                            go_h,
                            k,
                            n,
                            1,
                            "MatmulBackward(vm)",
                        )?;
                        Some(Tensor::from_storage(
                            TensorStorage::gpu(result_h),
                            vec![k],
                            false,
                        )?)
                    } else {
                        None
                    };

                    let grad_b = if self.b.requires_grad() {
                        // dB = outer(a, grad_y): a (K,) × grad_y (N,) → (K,N).
                        // Compose as matmul((K,1), (1,N)) = rank-1 mm.
                        let a_h = self.a.gpu_handle()?;
                        let go_h = grad_output.gpu_handle()?;
                        let result_h = cuda_matmul_same_dtype::<T>(
                            backend,
                            a_h,
                            go_h,
                            k,
                            1,
                            n,
                            "MatmulBackward(vm)",
                        )?;
                        Some(Tensor::from_storage(
                            TensorStorage::gpu(result_h),
                            vec![k, n],
                            false,
                        )?)
                    } else {
                        None
                    };

                    return Ok(vec![grad_a, grad_b]);
                }

                let grad_a = if self.a.requires_grad() {
                    Some(linalg::mv(&self.b, grad_output)?)
                } else {
                    None
                };

                let grad_b = if self.b.requires_grad() {
                    let a_data = self.a.data()?;
                    let grad_data = grad_output.data()?;
                    let k = a_data.len();
                    let n = grad_data.len();
                    let mut outer = vec![<T as num_traits::Zero>::zero(); k * n];
                    for ki in 0..k {
                        for ni in 0..n {
                            outer[ki * n + ni] = a_data[ki] * grad_data[ni];
                        }
                    }
                    Some(Tensor::from_storage(
                        TensorStorage::cpu(outer),
                        vec![k, n],
                        false,
                    )?)
                } else {
                    None
                };

                Ok(vec![grad_a, grad_b])
            }
            _ => {
                // Batched broadcast matmul backward.
                // For C = matmul(A, B) where shapes may broadcast:
                //   dA = matmul(grad_C, B^T)  — then sum-reduce broadcast dims
                //   dB = matmul(A^T, grad_C)  — then sum-reduce broadcast dims
                //
                // "Transpose" here means swapping the last two dims.
                // After computing the full broadcast gradient, we sum over
                // any dimensions that were broadcast (size-1 in original).
                broadcast_matmul_backward(&self.a, &self.b, grad_output)
            }
        }
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.a, &self.b]
    }

    fn name(&self) -> &'static str {
        "MatmulBackward"
    }
}

/// Backward pass for batched broadcast matmul.
///
/// Given forward: `C = matmul(A, B)` where A and B may have broadcast
/// batch dimensions, computes:
/// - `grad_A = matmul(grad_C, B_transposed)` summed over broadcast dims
/// - `grad_B = matmul(A_transposed, grad_C)` summed over broadcast dims
fn broadcast_matmul_backward<T: Float>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    grad_output: &Tensor<T>,
) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
    // CORE-186 (#1880): re-promote 1-D operands before the transpose/matmul
    // pipeline. The forward (`ops::linalg::broadcast_matmul`) promotes a 1-D
    // LHS to (1, K) and a 1-D RHS to (K, 1) and squeezes the promoted dim
    // from the output — mirroring torch's `matmul_impl` decomposition
    // (aten/src/ATen/native/LinearAlgebra.cpp), through which torch autograd
    // differentiates. The saved operands here are the ORIGINAL 1-D tensors
    // and `grad_output` carries the squeezed shape, so without re-promotion
    // the vector gradient broadcast-matmuls the squeezed cotangent against
    // every batch element (cross-batch contamination: the audit's [60, 96]
    // vs torch's [30, 48]) and the other operand's gradient errors outright
    // (`swap_last_two` rejects ndim < 2). Promote, run the matrix backward,
    // then squeeze the resulting gradient back to the leaf's 1-D shape.
    //
    // `view_reshape` is a pure metadata op on contiguous tensors (any
    // device) and never attaches autograd edges; whether a gradient is
    // needed is captured from the ORIGINAL leaves first.
    let need_a = a.requires_grad();
    let need_b = b.requires_grad();
    let a_is_1d = a.ndim() == 1;
    let b_is_1d = b.ndim() == 1;
    let a_p: Tensor<T> = if a_is_1d {
        a.view_reshape(vec![1, a.shape()[0]])?
    } else {
        a.clone()
    };
    let b_p: Tensor<T> = if b_is_1d {
        b.view_reshape(vec![b.shape()[0], 1])?
    } else {
        b.clone()
    };
    let g_p: Tensor<T> = if a_is_1d || b_is_1d {
        // Re-insert the squeezed output dims: the promoted output shape is
        // batch ++ [m, n]; the forward removed n (last dim) when b was 1-D
        // and m (second-to-last) when a was 1-D. Append the n=1 axis first,
        // then insert the m=1 axis before the last position.
        let mut g_shape = grad_output.shape().to_vec();
        if b_is_1d {
            g_shape.push(1);
        }
        if a_is_1d {
            let pos = g_shape.len() - 1;
            g_shape.insert(pos, 1);
        }
        grad_output.view_reshape(g_shape)?
    } else {
        grad_output.clone()
    };

    // Transpose last two dims of a tensor (swap matrix dims in batched tensor).
    //
    // §3 GPU-native: use `permute + contiguous` which already dispatches to GPU.
    // The permute axis vector is [0, 1, ..., nd-3, nd-1, nd-2].
    let swap_last_two = |t: &Tensor<T>| -> FerrotorchResult<Tensor<T>> {
        let shape = t.shape();
        let nd = shape.len();
        if nd < 2 {
            return Err(FerrotorchError::InvalidArgument {
                message: "Cannot transpose last two dims of tensor with ndim < 2".into(),
            });
        }
        if t.is_cuda() {
            // GPU path: permute last two dims, then make contiguous (copies on device).
            let mut perm: Vec<usize> = (0..nd).collect();
            perm.swap(nd - 2, nd - 1);
            return t.permute(&perm)?.contiguous();
        }
        let data = t.data()?;
        let rows = shape[nd - 2];
        let cols = shape[nd - 1];
        let mat_size = rows * cols;
        // CORE-139 (#1833): no `.max(1)` — an empty batch prefix already has
        // product 1; a zero-sized batch dim must skip the loop (the data
        // slice is empty, so a forced iteration would index out of bounds).
        let n_mats: usize = crate::shape::numel(&shape[..nd - 2]);
        let mut out = vec![<T as num_traits::Zero>::zero(); data.len()];
        for m in 0..n_mats {
            let off = m * mat_size;
            for i in 0..rows {
                for j in 0..cols {
                    out[off + j * rows + i] = data[off + i * cols + j];
                }
            }
        }
        let mut out_shape = shape.to_vec();
        out_shape[nd - 2] = cols;
        out_shape[nd - 1] = rows;
        Tensor::from_storage(TensorStorage::cpu(out), out_shape, false)
    };

    // Sum-reduce grad to match the original shape. This handles the case
    // where a dimension was size-1 (broadcast) in the original but expanded
    // in the gradient. We need to sum over those expanded dimensions.
    //
    // §3 GPU-native: iteratively call `sum_dim` (already GPU-aware for f32/f64)
    // to collapse each dimension that doesn't match the target. Leading dims
    // that were broadcast-expanded are collapsed with a final slice/reshape.
    let reduce_to_shape = |grad: Tensor<T>, target: &[usize]| -> FerrotorchResult<Tensor<T>> {
        let grad_shape = grad.shape().to_vec();
        if grad_shape == target {
            return Ok(grad);
        }

        if grad.is_cuda() {
            // GPU path: for each dim where grad_shape[d] > target[d], sum along that dim.
            // The dim alignment is: grad_shape and target have the same number of dims
            // OR grad has more leading dims (broadcast-expanded batch dims).
            let grad_nd = grad_shape.len();
            let target_nd = target.len();

            let mut current = grad;

            // Step 1: sum out any extra leading dims beyond target_nd.
            // After each sum_dim(0, keepdim=false), current loses its first dim.
            let extra_leading = grad_nd.saturating_sub(target_nd);
            for _ in 0..extra_leading {
                current = crate::grad_fns::reduction::sum_dim(&current, 0, false)?;
            }

            // Step 2: for each remaining dim that is size-1 in target but >1 in current,
            // sum along that dim (keepdim=true to preserve alignment).
            let cur_shape = current.shape().to_vec();
            for (d, (&cur_size, &tgt_size)) in cur_shape.iter().zip(target.iter()).enumerate() {
                if tgt_size == 1 && cur_size != 1 {
                    // sum_dim uses i64 dim index; d is already in-bounds after leading collapse.
                    current = crate::grad_fns::reduction::sum_dim(&current, d as i64, true)?;
                }
            }

            return Ok(current);
        }

        let grad_nd = grad_shape.len();
        let target_nd = target.len();
        let offset = grad_nd - target_nd;
        let grad_data = grad.data()?;

        // Compute target total size. CORE-139 (#1833): no `.max(1)` — a
        // scalar (empty) target already has product 1; a zero-sized target
        // dim must produce a zero-length buffer, not a spurious 1-element
        // one (`from_storage` would reject the length/shape mismatch).
        let target_size: usize = crate::shape::numel(target);
        let mut result = vec![<T as num_traits::Zero>::zero(); target_size];

        let grad_total: usize = crate::shape::numel(&grad_shape);

        // For each element in the gradient, compute which element in the
        // target it maps to, and accumulate.
        // Build stride tables for both shapes.
        let mut grad_strides = vec![1usize; grad_nd];
        for i in (0..grad_nd.saturating_sub(1)).rev() {
            grad_strides[i] = grad_strides[i + 1] * grad_shape[i + 1];
        }

        let mut target_strides = vec![1usize; target_nd];
        if target_nd > 0 {
            for i in (0..target_nd.saturating_sub(1)).rev() {
                target_strides[i] = target_strides[i + 1] * target[i + 1];
            }
        }

        for (flat, &grad_val) in grad_data.iter().enumerate().take(grad_total) {
            // Decompose flat index into grad multi-index.
            let mut remaining = flat;
            let mut target_flat = 0usize;
            for d in (0..grad_nd).rev() {
                let coord = remaining % grad_shape[d];
                remaining /= grad_shape[d];

                // Map to target dimension.
                if d >= offset {
                    let td = d - offset;
                    let target_coord = if target[td] == 1 { 0 } else { coord };
                    target_flat += target_coord * target_strides[td];
                }
                // If d < offset, this dimension doesn't exist in target — summed out.
            }
            result[target_flat] += grad_val;
        }

        Tensor::from_storage(TensorStorage::cpu(result), target.to_vec(), false)
    };

    let grad_a = if need_a {
        // grad_A = matmul(grad_C, B^T) reduced to A's (promoted) shape.
        let bt = swap_last_two(&b_p)?;
        let full_grad = linalg::matmul(&g_p, &bt)?;
        let reduced = reduce_to_shape(full_grad, a_p.shape())?;
        // Squeeze the promoted (1, K) gradient back to the 1-D leaf shape.
        Some(if a_is_1d {
            reduced.view_reshape(a.shape().to_vec())?
        } else {
            reduced
        })
    } else {
        None
    };

    let grad_b = if need_b {
        // grad_B = matmul(A^T, grad_C) reduced to B's (promoted) shape.
        let at = swap_last_two(&a_p)?;
        let full_grad = linalg::matmul(&at, &g_p)?;
        let reduced = reduce_to_shape(full_grad, b_p.shape())?;
        // Squeeze the promoted (K, 1) gradient back to the 1-D leaf shape.
        Some(if b_is_1d {
            reduced.view_reshape(b.shape().to_vec())?
        } else {
            reduced
        })
    } else {
        None
    };

    Ok(vec![grad_a, grad_b])
}

// ---------------------------------------------------------------------------
// Differentiable forward wrappers
// ---------------------------------------------------------------------------

/// Differentiable matrix-matrix multiply. If either input requires grad and
/// grad is enabled, attaches `MmBackward`.
pub fn mm_differentiable<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let (m, k, n) = validate_mm_operands("mm", a, b)?;

    // Materialize to exact packed storage before linalg ops. A narrow view can
    // be logically contiguous with offset 0 while still sharing a larger base
    // buffer; CUDA kernels receive only the raw handle and would see the base
    // length instead of the logical shape.
    let a = a.contiguous()?;
    let b = b.contiguous()?;

    if a.is_cuda() {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        // Dtype-aware GPU dispatch (#800 + #23): bf16 routes to
        // `matmul_bf16_bf16` (cuBLAS GemmEx, f32 accumulator).
        let handle: crate::gpu_dispatch::GpuBufferHandle = crate::dispatch_floating_dtype!(
            T,
            "mm",
            f32 => {
                if autocast_guard("mm") == Some(AutocastCategory::ReducedPrecision) {
                    backend.matmul_f16_f32(a.gpu_handle()?, b.gpu_handle()?, m, k, n)
                } else {
                    backend.matmul_f32(a.gpu_handle()?, b.gpu_handle()?, m, k, n)
                }
            },
            f64 => backend.matmul_f64(a.gpu_handle()?, b.gpu_handle()?, m, k, n),
            bf16 => backend.matmul_bf16_bf16(a.gpu_handle()?, b.gpu_handle()?, m, k, n),
            f16 => backend.matmul_f16_f16(a.gpu_handle()?, b.gpu_handle()?, m, k, n),
        )?;
        let storage = TensorStorage::gpu(handle);
        let shape = vec![m, n];

        if is_grad_enabled() && (a.requires_grad() || b.requires_grad()) {
            let grad_fn = Arc::new(MmBackward::new(a.clone(), b.clone()));
            Tensor::from_operation(storage, shape, grad_fn)
        } else {
            Tensor::from_storage(storage, shape, false)
        }
    } else {
        let a_data = a.data()?;
        let b_data = b.data()?;

        // Compute result directly — no intermediate Tensor allocation.
        let result_vec = linalg::mm_raw(a_data, b_data, m, k, n);
        let storage = TensorStorage::cpu(result_vec);
        let shape = vec![m, n];

        if is_grad_enabled() && (a.requires_grad() || b.requires_grad()) {
            let grad_fn = Arc::new(MmBackward::new(a.clone(), b.clone()));
            Tensor::from_operation(storage, shape, grad_fn)
        } else {
            Tensor::from_storage(storage, shape, false)
        }
    }
}

// ---------------------------------------------------------------------------
// MmBtBackward — C = A @ B^T  (fused transpose, no materialized B^T)
// ---------------------------------------------------------------------------

/// Backward for fused `C = A @ B^T` (B is NOT transposed in storage).
///
/// Forward: C[i,j] = sum_k A[i,k] * B[j,k]  (B is (N,K) row-major)
///
/// VJP:
/// - `dA = grad_C @ B`   (no transpose — B is already in the right layout)
/// - `dB = grad_C^T @ A` (which is the same as grad_C transposed times A)
#[derive(Debug)]
struct MmBtBackward<T: Float> {
    a: Tensor<T>, // (M, K)
    b: Tensor<T>, // (N, K) — original, not transposed
}

impl<T: Float> MmBtBackward<T> {
    fn new(a: Tensor<T>, b: Tensor<T>) -> Self {
        Self { a, b }
    }
}

impl<T: Float> GradFn<T> for MmBtBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // GPU-native path for f32/f64/bf16/f16. PyTorch keeps half/bfloat
        // attention and linear VJPs resident on CUDA; no CPU fallback here.
        if grad_output.is_cuda() {
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let go_h = grad_output.gpu_handle()?;
            let m = grad_output.shape()[0];
            let n = grad_output.shape()[1];

            let grad_a = if self.a.requires_grad() {
                let k = self.b.shape()[1];
                let b_h = self.b.gpu_handle()?;
                let result_h =
                    cuda_matmul_same_dtype::<T>(backend, go_h, b_h, m, n, k, "MmBtBackward")?;
                Some(Tensor::from_storage(
                    TensorStorage::gpu(result_h),
                    vec![m, k],
                    false,
                )?)
            } else {
                None
            };

            let grad_b = if self.b.requires_grad() {
                let k = self.a.shape()[1];
                let a_h = self.a.gpu_handle()?;
                let got_h = cuda_transpose_2d_same_dtype::<T>(backend, go_h, m, n, "MmBtBackward")?;
                let result_h =
                    cuda_matmul_same_dtype::<T>(backend, &got_h, a_h, n, m, k, "MmBtBackward")?;
                Some(Tensor::from_storage(
                    TensorStorage::gpu(result_h),
                    vec![n, k],
                    false,
                )?)
            } else {
                None
            };

            return Ok(vec![grad_a, grad_b]);
        }

        let grad_a = if self.a.requires_grad() {
            Some(mm(grad_output, &self.b)?)
        } else {
            None
        };

        let grad_b = if self.b.requires_grad() {
            let gc_data = grad_output.data()?;
            let a_data = self.a.data()?;
            let m = grad_output.shape()[0];
            let n = grad_output.shape()[1];
            let k = self.a.shape()[1];
            let result = crate::ops::linalg::mm_raw_at(gc_data, a_data, n, m, k);
            Some(Tensor::from_storage(
                TensorStorage::cpu(result),
                vec![n, k],
                false,
            )?)
        } else {
            None
        };

        Ok(vec![grad_a, grad_b])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.a, &self.b]
    }

    fn name(&self) -> &'static str {
        "MmBtBackward"
    }
}

/// Fused differentiable `A @ B^T`. Avoids materializing the transpose of B.
///
/// A: (M, K), B: (N, K) -> result: (M, N)
/// Linear layer uses this: input @ weight^T where weight is (out, in).
pub fn mm_bt_differentiable<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let (m, k, n) = validate_mm_bt_operands("mm_bt", a, b)?;
    let a = a.contiguous()?;
    let b = b.contiguous()?;

    // GPU path: fused-transpose matmul. This is the natural layout for
    // attention `Q @ K^T` and linear weights `[out, in]`, and it avoids a
    // temporary transpose for every dtype.
    if a.is_cuda() {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let handle = cuda_matmul_nt_same_dtype::<T>(
            backend,
            a.gpu_handle()?,
            b.gpu_handle()?,
            m,
            k,
            n,
            "mm_bt",
        )?;
        let storage = TensorStorage::gpu(handle);
        let shape = vec![m, n];

        return if is_grad_enabled() && (a.requires_grad() || b.requires_grad()) {
            let grad_fn = Arc::new(MmBtBackward::new(a.clone(), b.clone()));
            Tensor::from_operation(storage, shape, grad_fn)
        } else {
            Tensor::from_storage(storage, shape, false)
        };
    }

    let a_data = a.data()?;
    let b_data = b.data()?;
    let result_vec = linalg::mm_raw_bt(a_data, b_data, m, k, n);
    let storage = TensorStorage::cpu(result_vec);
    let shape = vec![m, n];

    if is_grad_enabled() && (a.requires_grad() || b.requires_grad()) {
        let grad_fn = Arc::new(MmBtBackward::new(a.clone(), b.clone()));
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Tensor::from_storage(storage, shape, false)
    }
}

// ---------------------------------------------------------------------------
// Fused Linear: C = A @ W^T + bias  (avoids intermediate tensors)
// ---------------------------------------------------------------------------

/// Backward for fused linear: C = A @ W^T + bias.
/// grad_A = grad_C @ W, grad_W = grad_C^T @ A, grad_bias = sum(grad_C, dim=0).
#[derive(Debug)]
struct LinearFusedBackward<T: Float> {
    input: Tensor<T>,  // (M, K)
    weight: Tensor<T>, // (N, K) — not transposed
    has_bias: bool,
    bias: Option<Tensor<T>>, // (N,)
}

impl<T: Float> GradFn<T> for LinearFusedBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let m = grad_output.shape()[0];
        let n = grad_output.shape()[1];

        // GPU-native path for f32/f64/bf16/f16 tensors.
        if grad_output.is_cuda() {
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let go_h = grad_output.gpu_handle()?;

            let grad_input = if self.input.requires_grad() {
                let k = self.weight.shape()[1];
                let w_h = self.weight.gpu_handle()?;
                let result_h = cuda_matmul_same_dtype::<T>(
                    backend,
                    go_h,
                    w_h,
                    m,
                    n,
                    k,
                    "LinearFusedBackward",
                )?;
                Some(Tensor::from_storage(
                    TensorStorage::gpu(result_h),
                    vec![m, k],
                    false,
                )?)
            } else {
                None
            };

            let grad_weight = if self.weight.requires_grad() {
                let k = self.input.shape()[1];
                let inp_h = self.input.gpu_handle()?;
                let got_h =
                    cuda_transpose_2d_same_dtype::<T>(backend, go_h, m, n, "LinearFusedBackward")?;
                let result_h = cuda_matmul_same_dtype::<T>(
                    backend,
                    &got_h,
                    inp_h,
                    n,
                    m,
                    k,
                    "LinearFusedBackward",
                )?;
                Some(Tensor::from_storage(
                    TensorStorage::gpu(result_h),
                    vec![n, k],
                    false,
                )?)
            } else {
                None
            };

            let grad_bias = if self.has_bias {
                if let Some(ref b) = self.bias {
                    if b.requires_grad() {
                        let go_shape = &[m, n];
                        let summed = cuda_sum_axis_same_dtype::<T>(
                            backend,
                            go_h,
                            go_shape,
                            0,
                            "LinearFusedBackward",
                        )?;
                        Some(Tensor::from_storage(
                            TensorStorage::gpu(summed),
                            vec![n],
                            false,
                        )?)
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };

            let mut grads = vec![grad_input, grad_weight];
            if self.bias.is_some() {
                grads.push(grad_bias);
            }
            return Ok(grads);
        }

        // CPU path.
        let gc_data = grad_output.data()?;

        let grad_input = if self.input.requires_grad() {
            let w_data = self.weight.data()?;
            let k = self.weight.shape()[1];
            let result = crate::ops::linalg::mm_raw(gc_data, w_data, m, n, k);
            Some(Tensor::from_storage(
                TensorStorage::cpu(result),
                vec![m, k],
                false,
            )?)
        } else {
            None
        };

        let grad_weight = if self.weight.requires_grad() {
            let a_data = self.input.data()?;
            let k = self.input.shape()[1];
            let result = crate::ops::linalg::mm_raw_at(gc_data, a_data, n, m, k);
            Some(Tensor::from_storage(
                TensorStorage::cpu(result),
                vec![n, k],
                false,
            )?)
        } else {
            None
        };

        let grad_bias = if self.has_bias {
            if let Some(ref b) = self.bias {
                if b.requires_grad() {
                    let zero = <T as num_traits::Zero>::zero();
                    let mut gb = vec![zero; n];
                    for i in 0..m {
                        let row = i * n;
                        for j in 0..n {
                            gb[j] += gc_data[row + j];
                        }
                    }
                    Some(Tensor::from_storage(
                        TensorStorage::cpu(gb),
                        vec![n],
                        false,
                    )?)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        // Return exactly as many gradients as inputs() returns.
        let mut grads = vec![grad_input, grad_weight];
        if self.bias.is_some() {
            grads.push(grad_bias);
        }
        Ok(grads)
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        let mut v = vec![&self.input, &self.weight];
        if let Some(ref b) = self.bias {
            v.push(b);
        }
        v
    }

    fn name(&self) -> &'static str {
        "LinearFusedBackward"
    }
}

/// Fused differentiable linear: output = input @ weight^T + bias.
/// Creates a single tensor (instead of 3) with a combined backward.
pub fn linear_fused<T: Float>(
    input: &Tensor<T>,
    weight: &Tensor<T>,
    bias: Option<&Tensor<T>>,
) -> FerrotorchResult<Tensor<T>> {
    let (m, k, n) = validate_linear_fused_operands(input, weight, bias)?;

    // GPU path: transpose weight, matmul, broadcast_add bias.
    if input.is_cuda() {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        // Dtype-aware GPU dispatch: every supported floating CUDA dtype has a
        // resident `input @ weight^T` path. f32 autocast keeps its existing
        // reduced-precision f32-output route; f16/bf16 tensors use resident
        // f16/bf16 output, matching PyTorch parameter/activation dtype.
        let mut result_handle = if is_f32::<T>() {
            // C = input @ weight^T. weight is row-major [n=out, k=in], input is
            // [m, k]. The fused-transpose matmul folds weight's transpose into
            // the cuBLAS `transb` flag, so we drop the per-forward
            // `transpose_2d_f32(weight)` kernel launch + buffer alloc (#1679).
            if autocast_guard("linear") == Some(AutocastCategory::ReducedPrecision) {
                // ReducedPrecision keeps the explicit-transpose +
                // f16-accumulate/f32-output path: there is no f32-input
                // autocast fused-transpose kernel, and `matmul_f16_f32` takes a
                // [k, n] right operand (the transposed weight).
                let wt_handle = backend.transpose_2d_f32(weight.gpu_handle()?, n, k)?;
                backend.matmul_f16_f32(input.gpu_handle()?, &wt_handle, m, k, n)?
            } else {
                backend.matmul_f32_nt(input.gpu_handle()?, weight.gpu_handle()?, m, k, n)?
            }
        } else if is_f64::<T>() {
            backend.matmul_f64_nt(input.gpu_handle()?, weight.gpu_handle()?, m, k, n)?
        } else if is_bf16::<T>() {
            backend.matmul_bf16_bf16_nt(input.gpu_handle()?, weight.gpu_handle()?, m, k, n)?
        } else if is_f16::<T>() {
            backend.matmul_f16_f16_nt(input.gpu_handle()?, weight.gpu_handle()?, m, k, n)?
        } else {
            return Err(FerrotorchError::NotImplementedOnCuda { op: "linear_fused" });
        };
        // Add bias if present (dtype-aware).
        if let Some(b) = bias {
            let out_shape = vec![m, n];
            let b_shape = vec![n];
            result_handle = cuda_broadcast_add_same_dtype::<T>(
                backend,
                &result_handle,
                b.gpu_handle()?,
                &out_shape,
                &b_shape,
                &out_shape,
                "linear_fused",
            )?;
        }
        let storage = TensorStorage::gpu(result_handle);
        let shape = vec![m, n];

        let needs_grad = is_grad_enabled()
            && (input.requires_grad()
                || weight.requires_grad()
                || bias.is_some_and(|b| b.requires_grad()));

        return if needs_grad {
            let grad_fn = Arc::new(LinearFusedBackward {
                input: input.clone(),
                weight: weight.clone(),
                has_bias: bias.is_some(),
                bias: bias.cloned(),
            });
            Tensor::from_operation(storage, shape, grad_fn)
        } else {
            Tensor::from_storage(storage, shape, false)
        };
    }

    let a_data = input.data()?;
    let w_data = weight.data()?;
    let mut result_vec = linalg::mm_raw_bt(a_data, w_data, m, k, n);

    // Fuse bias addition
    if let Some(b) = bias {
        let b_data = b.data()?;
        for i in 0..m {
            let row = i * n;
            for j in 0..n {
                result_vec[row + j] += b_data[j];
            }
        }
    }

    let storage = TensorStorage::cpu(result_vec);
    let shape = vec![m, n];

    let needs_grad = is_grad_enabled()
        && (input.requires_grad()
            || weight.requires_grad()
            || bias.is_some_and(|b| b.requires_grad()));

    if needs_grad {
        let grad_fn = Arc::new(LinearFusedBackward {
            input: input.clone(),
            weight: weight.clone(),
            has_bias: bias.is_some(),
            bias: bias.cloned(),
        });
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Tensor::from_storage(storage, shape, false)
    }
}

/// Differentiable matrix-vector multiply. Attaches `MvBackward` when needed.
pub fn mv_differentiable<T: Float>(a: &Tensor<T>, x: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let needs_grad = is_grad_enabled() && (a.requires_grad() || x.requires_grad());
    let (m, k) = validate_mv_operands("mv", a, x)?;

    // GPU path (#817): route CUDA inputs through cuBLAS Sgemv/Dgemv. Pre-fix
    // the function unconditionally called `.data()?` and surfaced as
    // `GpuTensorNotAccessible`. PyTorch's `torch.mv` works on CUDA for
    // f32 and f64 and so must ferrotorch's.
    if a.is_cuda()
        && a.device() == x.device()
        && let Some(backend) = gpu_backend()
    {
        // Materialise non-contiguous views (e.g. permute/transpose) so the
        // row-major-trick in cuBLAS sees contiguous strides.
        let a = if a.is_contiguous() {
            a.clone()
        } else {
            a.contiguous()?
        };
        let x = if x.is_contiguous() {
            x.clone()
        } else {
            x.contiguous()?
        };
        let handle = if is_f32::<T>() {
            backend.mv_f32(a.gpu_handle()?, x.gpu_handle()?, m, k)?
        } else if is_f64::<T>() {
            backend.mv_f64(a.gpu_handle()?, x.gpu_handle()?, m, k)?
        } else {
            // Reduced-precision CUDA mv has the same storage layout as
            // [m,k] @ [k,1]. Keep it resident through the dtype-specific GEMM.
            cuda_matmul_same_dtype::<T>(backend, a.gpu_handle()?, x.gpu_handle()?, m, k, 1, "mv")?
        };
        let storage = TensorStorage::gpu(handle);
        let shape = vec![m];
        return if needs_grad {
            let grad_fn = Arc::new(MvBackward::new(a, x));
            Tensor::from_operation(storage, shape, grad_fn)
        } else {
            Tensor::from_storage(storage, shape, false)
        };
    }

    // CPU path: compute mv directly from slices to avoid double-copy.
    let a_data = a.data()?;
    let x_data = x.data()?;
    let zero = <T as num_traits::Zero>::zero();

    let mut result_vec = vec![zero; m];
    // CORE-140 (#1834): f16/bf16 accumulate in the f32 opmath type with a
    // single rounding at the end, matching torch's CPU mv for Half/BFloat16
    // (opmath_type<Half> = float, aten/src/ATen/OpMathType.h).
    if crate::ops::linalg::is_reduced_precision::<T>() {
        for (i, result_elem) in result_vec.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            let row = i * k;
            for p in 0..k {
                acc += crate::ops::linalg::opmath_up(a_data[row + p])
                    * crate::ops::linalg::opmath_up(x_data[p]);
            }
            *result_elem = crate::ops::linalg::opmath_down(acc);
        }
    } else {
        for (i, result_elem) in result_vec.iter_mut().enumerate() {
            let mut acc = zero;
            let row = i * k;
            for p in 0..k {
                acc += a_data[row + p] * x_data[p];
            }
            *result_elem = acc;
        }
    }

    let storage = TensorStorage::cpu(result_vec);
    let shape = vec![m];

    if needs_grad {
        let grad_fn = Arc::new(MvBackward::new(a.clone(), x.clone()));
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Tensor::from_storage(storage, shape, false)
    }
}

/// Differentiable dot product. Attaches `DotBackward` when needed.
pub fn dot_differentiable<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let needs_grad = is_grad_enabled() && (a.requires_grad() || b.requires_grad());
    let n = validate_dot_operands("dot", a, b)?;

    // GPU path (#816): route CUDA inputs through cuBLAS Sdot/Ddot. Pre-fix
    // the function unconditionally called `.data()?` and surfaced as
    // `GpuTensorNotAccessible`. PyTorch's `torch.dot` works on CUDA for
    // f32 and f64 and so must ferrotorch's.
    if a.is_cuda()
        && a.device() == b.device()
        && let Some(backend) = gpu_backend()
    {
        let a = if a.is_contiguous() {
            a.clone()
        } else {
            a.contiguous()?
        };
        let b = if b.is_contiguous() {
            b.clone()
        } else {
            b.contiguous()?
        };
        let handle = if is_f32::<T>() {
            backend.dot_f32(a.gpu_handle()?, b.gpu_handle()?, n)?
        } else if is_f64::<T>() {
            backend.dot_f64(a.gpu_handle()?, b.gpu_handle()?, n)?
        } else {
            // Dot is [1,n] @ [n,1] with a scalar-shaped output.
            cuda_matmul_same_dtype::<T>(backend, a.gpu_handle()?, b.gpu_handle()?, 1, n, 1, "dot")?
        };
        let storage = TensorStorage::gpu(handle);
        let shape: Vec<usize> = vec![];
        return if needs_grad {
            let grad_fn = Arc::new(DotBackward::new(a, b));
            Tensor::from_operation(storage, shape, grad_fn)
        } else {
            Tensor::from_storage(storage, shape, false)
        };
    }

    let a_data = a.data()?;
    let b_data = b.data()?;
    // CORE-140 (#1834): f16/bf16 accumulate in the f32 opmath type with a
    // single rounding at the end, matching torch's CPU dot for Half/BFloat16
    // (opmath_type<Half> = float, aten/src/ATen/OpMathType.h). Storage-
    // precision accumulation drifts ~0.5 % at k = 128 and overflows to inf
    // when an f16 partial sum exceeds 65504.
    let result_val = if crate::ops::linalg::is_reduced_precision::<T>() {
        let acc = a_data
            .iter()
            .zip(b_data.iter())
            .fold(0.0f32, |acc, (&x, &y)| {
                acc + crate::ops::linalg::opmath_up(x) * crate::ops::linalg::opmath_up(y)
            });
        crate::ops::linalg::opmath_down(acc)
    } else {
        a_data
            .iter()
            .zip(b_data.iter())
            .fold(<T as num_traits::Zero>::zero(), |acc, (&x, &y)| acc + x * y)
    };

    let storage = TensorStorage::cpu(vec![result_val]);
    let shape = vec![];

    if needs_grad {
        let grad_fn = Arc::new(DotBackward::new(a.clone(), b.clone()));
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Tensor::from_storage(storage, shape, false)
    }
}

/// Differentiable batched matmul with `BmmBackward`.
///
/// Uses the GPU-aware `bmm()` for the forward pass (dispatches to cuBLAS on
/// GPU, CPU loops otherwise), then attaches `BmmBackward` for autograd.
pub fn bmm_differentiable<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    // Record autocast decision. f32 autocast routes through the mixed f16
    // Tensor Core path; real f16/bf16 tensors use same-dtype resident output.
    let _autocast_cat = autocast_guard("bmm");
    let result = bmm(a, b)?;

    if is_grad_enabled() && (a.requires_grad() || b.requires_grad()) {
        let grad_fn = Arc::new(BmmBackward::new(a.clone(), b.clone()));
        let (storage, shape) = result.into_storage_and_shape()?;
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Ok(result)
    }
}

/// Differentiable general matmul dispatcher. Attaches `MatmulBackward`
/// when needed. Supports all rank combinations including batched broadcast
/// matmul for ≥3D tensors.
pub fn matmul_differentiable<T: Float>(
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> FerrotorchResult<Tensor<T>> {
    if a.device() != b.device() {
        return Err(FerrotorchError::DeviceMismatch {
            expected: a.device(),
            got: b.device(),
        });
    }

    if a.is_cuda() && a.ndim() == 2 && b.ndim() == 2 {
        validate_mm_operands("matmul", a, b)?;
    }

    // Materialize to exact packed storage before linalg ops. This is needed
    // for CUDA raw-handle consumers even when shape/stride contiguity is true
    // but the tensor aliases a larger storage.
    let a = a.contiguous()?;
    let b = b.contiguous()?;

    // GPU dispatch for 1D x 2D vector-matrix product (#818). Pre-fix this
    // shape fell through to `linalg::matmul`, which calls `.data()?` and
    // surfaces as `GpuTensorNotAccessible` for CUDA tensors. PyTorch's
    // `torch.matmul(x_1d, B_2d)` works on CUDA, so this branch routes to
    // `vm_f{32,64}` (cuBLAS gemv with the OP_N transpose flag — no
    // materialised transpose).
    if a.is_cuda() && a.ndim() == 1 && b.ndim() == 2 {
        let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let k = a.shape()[0];
        let n = b.shape()[1];
        if k != b.shape()[0] {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!("matmul 1D x 2D: a is [{k}], b is {:?}", b.shape()),
            });
        }
        let handle = if is_f32::<T>() {
            backend.vm_f32(a.gpu_handle()?, b.gpu_handle()?, k, n)?
        } else if is_f64::<T>() {
            backend.vm_f64(a.gpu_handle()?, b.gpu_handle()?, k, n)?
        } else {
            // Reduced-precision CUDA vm is [1,k] @ [k,n]. The resulting
            // one-row storage is the same contiguous layout as the vector
            // output shape [n].
            cuda_matmul_same_dtype::<T>(
                backend,
                a.gpu_handle()?,
                b.gpu_handle()?,
                1,
                k,
                n,
                "matmul",
            )?
        };
        let storage = TensorStorage::gpu(handle);
        let shape = vec![n];
        return if is_grad_enabled() && (a.requires_grad() || b.requires_grad()) {
            let grad_fn = Arc::new(MatmulBackward::new(a.clone(), b.clone()));
            Tensor::from_operation(storage, shape, grad_fn)
        } else {
            Tensor::from_storage(storage, shape, false)
        };
    }

    if a.is_cuda() && a.ndim() == 2 && b.ndim() == 2 {
        let (m, k, n) = validate_mm_operands("matmul", &a, &b)?;
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        // Dtype-aware GPU dispatch (#800 + #23): bf16 now routes to
        // `matmul_bf16_bf16` (existing cuBLAS GemmEx path from #17).
        let handle: crate::gpu_dispatch::GpuBufferHandle = crate::dispatch_floating_dtype!(
            T,
            "matmul",
            f32 => {
                // When autocast says ReducedPrecision and inputs are f32 on
                // GPU, use the f16-accumulate path (falls back to f32 if no
                // kernel).
                if autocast_guard("matmul") == Some(AutocastCategory::ReducedPrecision) {
                    backend.matmul_f16_f32(a.gpu_handle()?, b.gpu_handle()?, m, k, n)
                } else {
                    backend.matmul_f32(a.gpu_handle()?, b.gpu_handle()?, m, k, n)
                }
            },
            f64 => backend.matmul_f64(a.gpu_handle()?, b.gpu_handle()?, m, k, n),
            bf16 => backend.matmul_bf16_bf16(a.gpu_handle()?, b.gpu_handle()?, m, k, n),
            f16 => backend.matmul_f16_f16(a.gpu_handle()?, b.gpu_handle()?, m, k, n),
        )?;
        let storage = TensorStorage::gpu(handle);
        let shape = vec![m, n];

        if is_grad_enabled() && (a.requires_grad() || b.requires_grad()) {
            let grad_fn = Arc::new(MatmulBackward::new(a.clone(), b.clone()));
            Tensor::from_operation(storage, shape, grad_fn)
        } else {
            Tensor::from_storage(storage, shape, false)
        }
    } else {
        // Dispatch to specialized paths that avoid double-copy.
        //
        // GPU shape coverage (#801): 2D x 2D is handled above; 3D x 3D with
        // matching batch dim routes to `bmm_differentiable` which has GPU
        // dispatch (#800). 1D x 1D / 2D x 1D / 1D x 2D vector cases
        // are handled by `dot_differentiable` / `mv_differentiable` and the
        // `vm_*` GPU branch above (#816 / #817 / #818).
        //
        // For all other rank combinations on CUDA — 4D bmm, 3D x 2D, 2D x 3D,
        // and arbitrary leading-dim broadcasts — route to `broadcast_bmm_*`
        // (cuBLAS gemmStridedBatched, stride=0 on broadcasted axes; #819).
        match (a.ndim(), b.ndim()) {
            (1, 1) => return dot_differentiable(&a, &b),
            (2, 1) => return mv_differentiable(&a, &b),
            (2, 2) => return mm_differentiable(&a, &b),
            (3, 3) if a.shape()[0] == b.shape()[0] => return bmm_differentiable(&a, &b),
            _ => {}
        }

        // GPU broadcast-bmm path (#819, #1543). Routes 4D bmm, 3D x 2D,
        // 2D x 3D, and leading-dim broadcasts to cuBLAS gemmStridedBatched.
        // PyTorch supports all of these on CUDA; pre-fix the f32/f64 case
        // surfaced as `GpuTensorNotAccessible`, and the bf16 case fell
        // through to the CPU `broadcast_matmul` round-trip (50× precision
        // regression on the ViT shape — see
        // `tests/divergence_gh25_gpu_bf16_matmul_precision.rs`).
        //
        if a.is_cuda()
            && a.ndim() >= 2
            && b.ndim() >= 2
            && (is_f32::<T>() || is_f64::<T>() || is_bf16::<T>() || is_f16::<T>())
        {
            let a_nd = a.ndim();
            let b_nd = b.ndim();
            let m = a.shape()[a_nd - 2];
            let k_a = a.shape()[a_nd - 1];
            let k_b = b.shape()[b_nd - 2];
            let n = b.shape()[b_nd - 1];
            if k_a != k_b {
                return Err(FerrotorchError::ShapeMismatch {
                    message: format!(
                        "matmul: inner dimensions mismatch: {:?} @ {:?}",
                        a.shape(),
                        b.shape()
                    ),
                });
            }
            let a_lead: Vec<usize> = a.shape()[..a_nd - 2].to_vec();
            let b_lead: Vec<usize> = b.shape()[..b_nd - 2].to_vec();
            // Broadcast leading shapes (numpy / PyTorch rules), including
            // 0-sized axes. PyTorch's rule: (a, b) compatible iff a == b
            // OR a == 1 OR b == 1; result = a if b == 1 else b.
            let max_len = a_lead.len().max(b_lead.len());
            let mut out_lead: Vec<usize> = Vec::with_capacity(max_len);
            for i in 0..max_len {
                let pa = max_len - a_lead.len();
                let pb = max_len - b_lead.len();
                let da = if i < pa { 1 } else { a_lead[i - pa] };
                let db = if i < pb { 1 } else { b_lead[i - pb] };
                if da == db || da == 1 || db == 1 {
                    let pick = if db == 1 { da } else { db };
                    out_lead.push(pick);
                } else {
                    return Err(FerrotorchError::ShapeMismatch {
                        message: format!(
                            "matmul: batch dims cannot be broadcast: {:?} vs {:?}",
                            a.shape(),
                            b.shape()
                        ),
                    });
                }
            }
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let handle = if is_f32::<T>() {
                backend.broadcast_bmm_f32(
                    a.gpu_handle()?,
                    b.gpu_handle()?,
                    &a_lead,
                    &b_lead,
                    &out_lead,
                    m,
                    k_a,
                    n,
                )?
            } else if is_f64::<T>() {
                backend.broadcast_bmm_f64(
                    a.gpu_handle()?,
                    b.gpu_handle()?,
                    &a_lead,
                    &b_lead,
                    &out_lead,
                    m,
                    k_a,
                    n,
                )?
            } else if is_bf16::<T>() {
                backend.broadcast_bmm_bf16(
                    a.gpu_handle()?,
                    b.gpu_handle()?,
                    &a_lead,
                    &b_lead,
                    &out_lead,
                    m,
                    k_a,
                    n,
                )?
            } else {
                backend.broadcast_bmm_f16(
                    a.gpu_handle()?,
                    b.gpu_handle()?,
                    &a_lead,
                    &b_lead,
                    &out_lead,
                    m,
                    k_a,
                    n,
                )?
            };
            let mut shape = out_lead;
            shape.push(m);
            shape.push(n);
            let storage = TensorStorage::gpu(handle);
            return if is_grad_enabled() && (a.requires_grad() || b.requires_grad()) {
                let grad_fn = Arc::new(MatmulBackward::new(a.clone(), b.clone()));
                Tensor::from_operation(storage, shape, grad_fn)
            } else {
                Tensor::from_storage(storage, shape, false)
            };
        }

        // Fallback for other shapes — still goes through linalg::matmul.
        let result = linalg::matmul(&a, &b)?;

        if is_grad_enabled() && (a.requires_grad() || b.requires_grad()) {
            let grad_fn = Arc::new(MatmulBackward::new(a.clone(), b.clone()));
            let (storage, shape) = result.into_storage_and_shape()?;
            Tensor::from_operation(storage, shape, grad_fn)
        } else {
            Ok(result)
        }
    }
}

// ===========================================================================
// bmm (batched matmul) — GPU-accelerated via strided batch SGEMM
// ===========================================================================

/// Batched matrix multiply: `C[i] = A[i] @ B[i]` for `i` in `0..batch`.
///
/// `a` shape: `[batch, m, k]`, `b` shape: `[batch, k, n]`.
/// Returns `[batch, m, n]`.
///
/// On GPU, dispatches to cuBLAS `SgemmStridedBatched`. On CPU, falls back
/// to per-batch `mm`.
pub fn bmm<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if a.ndim() != 3 || b.ndim() != 3 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "bmm requires 3-D tensors, got {:?} and {:?}",
                a.shape(),
                b.shape()
            ),
        });
    }
    if a.device() != b.device() {
        return Err(FerrotorchError::DeviceMismatch {
            expected: a.device(),
            got: b.device(),
        });
    }

    // Materialize non-contiguous views (e.g. from permute/transpose).
    let a = if a.is_contiguous() {
        a.clone()
    } else {
        a.contiguous()?
    };
    let b = if b.is_contiguous() {
        b.clone()
    } else {
        b.contiguous()?
    };

    let batch = a.shape()[0];
    let m = a.shape()[1];
    let k = a.shape()[2];
    let n = b.shape()[2];

    if b.shape()[0] != batch || b.shape()[1] != k {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!("bmm: a is [{batch},{m},{k}], b is {:?}", b.shape()),
        });
    }

    let out_shape = vec![batch, m, n];

    // GPU path.
    if a.is_cuda()
        && let Some(backend) = crate::gpu_dispatch::gpu_backend()
    {
        // Dtype-aware GPU dispatch (#800): the f32-only path returned
        // "GPU handle does not contain a CudaBuffer<f32>" for f64 inputs.
        // Forward must branch by dtype and keep CUDA tensors resident.
        let handle = if is_f32::<T>() {
            // Use f16 Tensor Core path when autocast selects ReducedPrecision.
            if autocast_guard("bmm") == Some(AutocastCategory::ReducedPrecision) {
                backend.bmm_f16_f32(a.gpu_handle()?, b.gpu_handle()?, batch, m, k, n)?
            } else {
                backend.bmm_f32(a.gpu_handle()?, b.gpu_handle()?, batch, m, k, n)?
            }
        } else if is_f64::<T>() {
            backend.bmm_f64(a.gpu_handle()?, b.gpu_handle()?, batch, m, k, n)?
        } else if is_bf16::<T>() {
            backend.bmm_bf16_bf16(a.gpu_handle()?, b.gpu_handle()?, batch, m, k, n)?
        } else if is_f16::<T>() {
            backend.bmm_f16_f16(a.gpu_handle()?, b.gpu_handle()?, batch, m, k, n)?
        } else {
            return Err(FerrotorchError::NotImplementedOnCuda { op: "bmm" });
        };
        return Tensor::from_storage(TensorStorage::gpu(handle), out_shape, false);
    }

    // CPU path: per-batch slab through the faer-backed `mm_raw` workhorse.
    // The earlier naive (i,j,p) triple-loop diverged from PyTorch's MKL
    // block-summation by ~1.5e-5 on f32 with k>=10 (verified 2026-05-26 on
    // op_db sample matmul seed=7 i=6); routing through `crate::ops::linalg::mm_raw`
    // consolidates accumulation behavior with the rest of the matmul family
    // (mm, broadcast_matmul). Byte-for-byte parity vs MKL still requires the
    // future-epic MKL/OpenBLAS FFI path; this commit acknowledges the
    // cross-BLAS f32 ULP reality by widening the matmul-family runner
    // tolerance to rtol=1e-4 (see `tools/parity-sweep/runner/src/main.rs`
    // `tolerance_for`).
    let a_data = a.data()?;
    let b_data = b.data()?;
    let slab = m * n;
    let a_stride = m * k;
    let b_stride = k * n;
    let mut out = vec![<T as num_traits::Zero>::zero(); batch * slab];

    for bi in 0..batch {
        let a_off = bi * a_stride;
        let b_off = bi * b_stride;
        let c_off = bi * slab;
        let a_slice = &a_data[a_off..a_off + a_stride];
        let b_slice = &b_data[b_off..b_off + b_stride];
        let c_slab = crate::ops::linalg::mm_raw(a_slice, b_slice, m, k, n);
        out[c_off..c_off + slab].copy_from_slice(&c_slab);
    }

    Tensor::from_storage(TensorStorage::cpu(out), out_shape, false)
}

// ===========================================================================
// permute_0213 — swap dims 1 and 2 of a 4D tensor
// ===========================================================================

/// Permute a 4-D tensor from `[d0, d1, d2, d3]` to `[d0, d2, d1, d3]`.
///
/// Primary use: reshape attention heads `[B, S, H, D_h]` → `[B, H, S, D_h]`.
/// On GPU, dispatches to a native PTX kernel. On CPU, does direct index mapping.
pub fn permute_0213<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if input.ndim() != 4 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("permute_0213 requires 4-D tensor, got {:?}", input.shape()),
        });
    }

    let d0 = input.shape()[0];
    let d1 = input.shape()[1];
    let d2 = input.shape()[2];
    let d3 = input.shape()[3];
    let out_shape = vec![d0, d2, d1, d3];

    // GPU path.
    if input.is_cuda()
        && let Some(backend) = crate::gpu_dispatch::gpu_backend()
    {
        let handle = cuda_permute_0213_same_dtype::<T>(
            backend,
            input.gpu_handle()?,
            d0,
            d1,
            d2,
            d3,
            "permute_0213",
        )?;
        return Tensor::from_storage(TensorStorage::gpu(handle), out_shape, false);
    }

    // CPU path.
    let data = input.data()?;
    let total = d0 * d1 * d2 * d3;
    let mut out = vec![<T as num_traits::Zero>::zero(); total];

    for i0 in 0..d0 {
        for i1 in 0..d1 {
            for i2 in 0..d2 {
                for i3 in 0..d3 {
                    let in_idx = ((i0 * d1 + i1) * d2 + i2) * d3 + i3;
                    let out_idx = ((i0 * d2 + i2) * d1 + i1) * d3 + i3;
                    out[out_idx] = data[in_idx];
                }
            }
        }
    }

    Tensor::from_storage(TensorStorage::cpu(out), out_shape, false)
}

// ===========================================================================
// Decomposition / reduction linalg ops with closed-form VJPs (#1345)
//
// These wrappers consume the forward implementations in `crate::linalg`
// (faer-backed) and attach a `GradFn` whose backward is a closed-form
// matrix differential. Each VJP below is grounded in a named PyTorch
// `file:line` and FD-verified in
// `tests/divergence_linalg_grad_audit.rs`.
// ===========================================================================

/// Helper: 2-D matrix transpose of a non-grad tensor (used by the matrix
/// VJPs below). Materialises a contiguous result on the input device.
fn mat_transpose<T: Float>(t: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    t.transpose(0, 1)?.contiguous()
}

// ---------------------------------------------------------------------------
// TraceBackward — s = trace(A) = sum_i A[i,i]   (2D -> scalar)
// ---------------------------------------------------------------------------

/// Backward for `s = trace(A)`.
///
/// VJP (`tools/autograd/derivatives.yaml:1785` `trace_backward_symint`):
/// `dA = grad_s * I` — i.e. `dA[i,j] = grad_s` on the main diagonal, else 0.
#[derive(Debug)]
pub struct TraceBackward<T: Float> {
    rows: usize,
    cols: usize,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Float> GradFn<T> for TraceBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if grad_output.is_cuda() {
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let handle = match <T as Element>::dtype() {
                DType::F32 => {
                    backend.trace_backward_f32(grad_output.gpu_handle()?, self.rows, self.cols)?
                }
                DType::F64 => {
                    backend.trace_backward_f64(grad_output.gpu_handle()?, self.rows, self.cols)?
                }
                DType::F16 | DType::BF16 => {
                    backend.trace_backward_u16(grad_output.gpu_handle()?, self.rows, self.cols)?
                }
                _ => {
                    return Err(FerrotorchError::NotImplementedOnCuda {
                        op: "trace backward",
                    });
                }
            };
            let grad_a = Tensor::from_storage(
                TensorStorage::gpu(handle),
                vec![self.rows, self.cols],
                false,
            )?;
            return Ok(vec![Some(grad_a)]);
        }

        let g: T = grad_output.item()?;
        let zero = <T as num_traits::Zero>::zero();
        let total =
            self.rows
                .checked_mul(self.cols)
                .ok_or_else(|| FerrotorchError::InvalidArgument {
                    message: format!(
                        "trace backward: shape [{}, {}] overflows storage size",
                        self.rows, self.cols
                    ),
                })?;
        let mut out = vec![zero; total];
        let k = self.rows.min(self.cols);
        for i in 0..k {
            out[i * self.cols + i] = g;
        }
        let grad_a =
            Tensor::from_storage(TensorStorage::cpu(out), vec![self.rows, self.cols], false)?;
        Ok(vec![Some(grad_a)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        // Trace's input is not retained (the VJP needs only the shape, which
        // is captured at forward time). The autograd graph edge is carried by
        // the leaf the differentiable wrapper passed to `from_operation`.
        vec![]
    }

    fn name(&self) -> &'static str {
        "TraceBackward"
    }
}

/// Differentiable `trace`. Attaches `TraceBackward` when grad is needed.
///
/// Forward computed under `no_grad`: `linalg_fwd::trace` (the public
/// `crate::linalg::trace` forward) delegates back here when grad is enabled,
/// so the guard prevents infinite re-entry.
pub fn trace_differentiable<T: Float>(a: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let result = crate::autograd::no_grad::no_grad(|| linalg_fwd::trace(a))?;
    if is_grad_enabled() && a.requires_grad() {
        let shape = a.shape();
        let grad_fn = Arc::new(TraceForward {
            input: a.clone(),
            inner: TraceBackward {
                rows: shape[0],
                cols: shape[1],
                _marker: std::marker::PhantomData,
            },
        });
        let (storage, shape) = result.into_storage_and_shape()?;
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Ok(result)
    }
}

/// `TraceBackward` retains only shape; this wrapper carries the input edge so
/// the graph connects back to the leaf `A`.
#[derive(Debug)]
struct TraceForward<T: Float> {
    input: Tensor<T>,
    inner: TraceBackward<T>,
}

impl<T: Float> GradFn<T> for TraceForward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        self.inner.backward(grad_output)
    }
    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }
    fn name(&self) -> &'static str {
        "TraceBackward"
    }
}

// ---------------------------------------------------------------------------
// OuterBackward — C = outer(a, b)   (1D x 1D -> 2D)
// ---------------------------------------------------------------------------

/// Backward for `C = outer(a, b)` where `C[i,j] = a[i] * b[j]`.
///
/// VJP (`tools/autograd/derivatives.yaml:275-276`, `vec1`/`vec2` of `addr`,
/// which is `outer` composed with `addmm`-style scaling):
/// - `da = grad_C @ b`     (mv: `[m,n] @ [n] -> [m]`)
/// - `db = grad_C^T @ a`   (mv: `[n,m] @ [m] -> [n]`)
#[derive(Debug)]
pub struct OuterBackward<T: Float> {
    a: Tensor<T>,
    b: Tensor<T>,
}

impl<T: Float> OuterBackward<T> {
    pub fn new(a: Tensor<T>, b: Tensor<T>) -> Self {
        Self { a, b }
    }
}

impl<T: Float> GradFn<T> for OuterBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let grad_a = if self.a.requires_grad() {
            // da = grad_C @ b
            Some(linalg::mv(grad_output, &self.b)?)
        } else {
            None
        };
        let grad_b = if self.b.requires_grad() {
            // db = grad_C^T @ a
            let gt = mat_transpose(grad_output)?;
            Some(linalg::mv(&gt, &self.a)?)
        } else {
            None
        };
        Ok(vec![grad_a, grad_b])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.a, &self.b]
    }

    fn name(&self) -> &'static str {
        "OuterBackward"
    }
}

/// Differentiable `outer`.
///
/// PyTorch implements `outer` as the composite
/// `self.reshape({self.size(0), 1}) * vec2`
/// (`aten/src/ATen/native/LinearAlgebra.cpp:1337-1342`). Returning the same
/// reshape/broadcast-mul graph is important for CUDA: the old closed-form
/// `OuterBackward` expressed the VJP through `ops::linalg::mv`, which is a
/// CPU leaf kernel and would force CUDA gradients into host-only data access.
pub fn outer_differentiable<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    linalg_fwd::outer_composite(a, b)
}

// ---------------------------------------------------------------------------
// LinalgInvBackward — Y = inv(A)   (2D square)
// ---------------------------------------------------------------------------

/// Backward for `Y = inv(A)`.
///
/// VJP (`tools/autograd/derivatives.yaml:917` `linalg_inv_ex`:
/// `inverse: -inv @ A_t @ inv`, transposed to VJP form):
/// `dA = -Y^T @ grad_Y @ Y^T`.
#[derive(Debug)]
pub struct LinalgInvBackward<T: Float> {
    /// The computed inverse `Y` (output), retained for the VJP.
    inv: Tensor<T>,
}

impl<T: Float> GradFn<T> for LinalgInvBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        crate::autograd::no_grad::no_grad(|| {
            // dA = -Y^T @ grad_Y @ Y^T
            let yt = mat_transpose(&self.inv)?;
            let tmp = yt.mm(grad_output)?; // Y^T @ grad
            let prod = tmp.mm(&yt)?; // (Y^T @ grad) @ Y^T
            Ok(vec![Some(prod.neg_t()?)])
        })
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        // The VJP closes over the retained inverse only; the graph edge to the
        // leaf `A` is carried by `InvForward` below.
        vec![]
    }

    fn name(&self) -> &'static str {
        "LinalgInvBackward"
    }
}

/// Carries the input edge for `inv` (the VJP itself only needs the output).
#[derive(Debug)]
struct InvForward<T: Float> {
    input: Tensor<T>,
    inner: LinalgInvBackward<T>,
}

impl<T: Float> GradFn<T> for InvForward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        self.inner.backward(grad_output)
    }
    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }
    fn name(&self) -> &'static str {
        "LinalgInvBackward"
    }
}

/// Differentiable `inv`. Attaches `LinalgInvBackward` when grad is needed.
///
/// Forward computed under `no_grad`: `linalg_fwd::inv` (the public
/// `crate::linalg::inv` forward) delegates back here when grad is enabled, so
/// the guard prevents infinite re-entry.
pub fn inv_differentiable<T: Float>(a: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let result = crate::autograd::no_grad::no_grad(|| linalg_fwd::inv(a))?;
    if is_grad_enabled() && a.requires_grad() {
        let grad_fn = Arc::new(InvForward {
            input: a.clone(),
            inner: LinalgInvBackward {
                inv: result.clone(),
            },
        });
        let (storage, shape) = result.into_storage_and_shape()?;
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// LinalgDetBackward — d = det(A)   (2D square -> scalar)
// ---------------------------------------------------------------------------

#[allow(
    clippy::float_cmp,
    reason = "matrix rank pivots intentionally test exact zero pivots to mirror LU singular handling"
)]
fn det_cofactor_minor<T: Float>(data: &[T], n: usize) -> T {
    if n == 0 {
        return <T as num_traits::One>::one();
    }
    if n == 1 {
        return data[0];
    }

    let zero = <T as num_traits::Zero>::zero();
    let mut a = data.to_vec();
    let mut sign = <T as num_traits::One>::one();
    let mut det = <T as num_traits::One>::one();

    for col in 0..n {
        let mut pivot_row = col;
        let mut pivot_abs = a[col * n + col].abs();
        for row in (col + 1)..n {
            let candidate = a[row * n + col].abs();
            if candidate > pivot_abs {
                pivot_abs = candidate;
                pivot_row = row;
            }
        }

        if pivot_abs == zero {
            return zero;
        }

        if pivot_row != col {
            for j in 0..n {
                a.swap(col * n + j, pivot_row * n + j);
            }
            sign = -sign;
        }

        let pivot = a[col * n + col];
        det = det * pivot;
        for row in (col + 1)..n {
            let factor = a[row * n + col] / pivot;
            a[row * n + col] = zero;
            for j in (col + 1)..n {
                a[row * n + j] = a[row * n + j] - factor * a[col * n + j];
            }
        }
    }

    sign * det
}

fn det_cofactor_matrix<T: Float>(data: &[T], n: usize) -> Vec<T> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![<T as num_traits::One>::one()];
    }

    let zero = <T as num_traits::Zero>::zero();
    let mut cofactors = vec![zero; n * n];
    let mut minor = Vec::with_capacity((n - 1) * (n - 1));
    for row in 0..n {
        for col in 0..n {
            minor.clear();
            for r in 0..n {
                if r == row {
                    continue;
                }
                for c in 0..n {
                    if c != col {
                        minor.push(data[r * n + c]);
                    }
                }
            }
            let sign = if (row + col) % 2 == 0 {
                <T as num_traits::One>::one()
            } else {
                -<T as num_traits::One>::one()
            };
            cofactors[row * n + col] = sign * det_cofactor_minor(&minor, n - 1);
        }
    }
    cofactors
}

/// Backward for `d = det(A)`.
///
/// VJP (`torch/csrc/autograd/FunctionsManual.cpp:4373` `linalg_det_backward`,
/// ordinary and higher-order branches):
/// - invertible: `dA = grad_d * det(A) * inv(A)^T`
/// - ordinary singular: PyTorch solves against a perturbed LU with RHS
///   `det(A) * grad`, which is zero for singular matrices except the explicit
///   1x1 special case.
/// - higher-order singular: PyTorch switches to the adjugate/cofactor formula.
#[derive(Debug)]
pub struct LinalgDetBackward<T: Float> {
    /// Retained input. Inverting during forward incorrectly rejects singular
    /// tracked inputs; PyTorch stores LU metadata and defers solve work to
    /// backward, so ferrotorch stores the input edge and computes lazily.
    input: Tensor<T>,
    /// Retained scalar determinant value.
    det: T,
}

impl<T: Float> GradFn<T> for LinalgDetBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let g: T = grad_output.item()?;
        let n = self.input.shape()[0];
        let grad_a = if n == 0 {
            Tensor::from_storage(TensorStorage::cpu(Vec::new()), vec![0, 0], false)?
        } else if n == 1 {
            Tensor::from_storage(TensorStorage::cpu(vec![g]), vec![1, 1], false)?
        } else {
            match crate::autograd::no_grad::no_grad(|| linalg_fwd::inv(&self.input)) {
                Ok(inv) => {
                    let inv_t = mat_transpose(&inv)?;
                    let scale = g * self.det;
                    let data = inv_t.data_vec()?;
                    let scaled: Vec<T> = data.iter().map(|&v| scale * v).collect();
                    Tensor::from_storage(TensorStorage::cpu(scaled), inv_t.shape().to_vec(), false)?
                }
                Err(_) if crate::autograd::higher_order::is_create_graph_enabled() => {
                    let data = self.input.data_vec()?;
                    let scaled: Vec<T> = det_cofactor_matrix(&data, n)
                        .into_iter()
                        .map(|v| g * v)
                        .collect();
                    Tensor::from_storage(TensorStorage::cpu(scaled), vec![n, n], false)?
                }
                Err(_) => Tensor::from_storage(
                    TensorStorage::cpu(vec![<T as num_traits::Zero>::zero(); n * n]),
                    vec![n, n],
                    false,
                )?,
            }
        };
        Ok(vec![Some(grad_a)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![]
    }

    fn name(&self) -> &'static str {
        "LinalgDetBackward"
    }
}

/// Carries the input edge for `det`.
#[derive(Debug)]
struct DetForward<T: Float> {
    input: Tensor<T>,
    inner: LinalgDetBackward<T>,
}

impl<T: Float> GradFn<T> for DetForward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        self.inner.backward(grad_output)
    }
    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }
    fn name(&self) -> &'static str {
        "LinalgDetBackward"
    }
}

/// Differentiable `det`. Attaches `LinalgDetBackward` when grad is needed.
///
/// Forward (and the VJP's internal `inv`) computed under `no_grad`:
/// `linalg_fwd::det` / `linalg_fwd::inv` (the public `crate::linalg::{det,inv}`
/// forwards) delegate back here when grad is enabled, so the guard prevents
/// infinite re-entry.
pub fn det_differentiable<T: Float>(a: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let result = crate::autograd::no_grad::no_grad(|| linalg_fwd::det(a))?;
    if is_grad_enabled() && a.requires_grad() {
        let det_val: T = result.item()?;
        let grad_fn = Arc::new(DetForward {
            input: a.clone(),
            inner: LinalgDetBackward {
                input: a.clone(),
                det: det_val,
            },
        });
        let (storage, shape) = result.into_storage_and_shape()?;
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// LinalgSolveBackward — X = solve(A, B)   (A: 2D square, B: [n] or [n,k])
// ---------------------------------------------------------------------------

/// Backward for `X = solve(A, B)` (i.e. `X = A^{-1} B`).
///
/// VJP (`torch/csrc/autograd/FunctionsManual.cpp:6160`
/// `linalg_solve_backward`, real case):
/// - `gB = A^{-T} @ gX`           (solve with `A^T`)
/// - `gA = -gB @ X^T`             (outer/matmul; vector RHS handled by
///   unsqueeze/squeeze to a column matrix)
#[derive(Debug)]
pub struct LinalgSolveBackward<T: Float> {
    a: Tensor<T>,
    /// The `B` graph edge — retained for gradient-slot ordering and the
    /// `requires_grad` check; the numeric VJP uses only `X`.
    b: Tensor<T>,
    /// Retained solution `X`.
    x: Tensor<T>,
    /// Whether `B` (and hence `X`) was a 1-D vector RHS.
    vector_rhs: bool,
}

impl<T: Float> GradFn<T> for LinalgSolveBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if grad_output.is_cuda() {
            return self.backward_cuda(grad_output);
        }

        // gB = A^{-T} @ gX, computed as solve(A^T, gX).
        let at = mat_transpose(&self.a)?;
        let gb = crate::autograd::no_grad::no_grad(|| linalg_fwd::solve(&at, grad_output))?;

        let grad_b = if self.b.requires_grad() {
            Some(gb.clone())
        } else {
            None
        };

        let grad_a = if self.a.requires_grad() {
            // gA = -gB @ X^T. Promote vector forms to column matrices first.
            let (gb_m, x_m) = if self.vector_rhs {
                let n = gb.shape()[0];
                let gb_col = Tensor::from_storage(
                    TensorStorage::cpu(gb.data()?.to_vec()),
                    vec![n, 1],
                    false,
                )?;
                let xn = self.x.shape()[0];
                let x_col = Tensor::from_storage(
                    TensorStorage::cpu(self.x.data()?.to_vec()),
                    vec![xn, 1],
                    false,
                )?;
                (gb_col, x_col)
            } else {
                (gb.clone(), self.x.clone())
            };
            let xt = mat_transpose(&x_m)?;
            let prod = mm(&gb_m, &xt)?;
            let data = prod.data()?;
            let neg: Vec<T> = data.iter().map(|&v| -v).collect();
            Some(Tensor::from_storage(
                TensorStorage::cpu(neg),
                prod.shape().to_vec(),
                false,
            )?)
        } else {
            None
        };

        Ok(vec![grad_a, grad_b])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.a, &self.b]
    }

    fn name(&self) -> &'static str {
        "LinalgSolveBackward"
    }
}

impl<T: Float> LinalgSolveBackward<T> {
    fn backward_cuda(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !(is_f32::<T>() || is_f64::<T>()) {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "LinalgSolveBackward",
            });
        }

        // gB = A^{-T} @ gX. Materialise the transposed view on device before
        // cuSOLVER, because the solve kernel consumes dense row-major storage.
        let at = crate::autograd::no_grad::no_grad(|| self.a.transpose(0, 1)?.contiguous())?;
        let gb = crate::autograd::no_grad::no_grad(|| linalg_fwd::solve(&at, grad_output))?;

        let grad_b = if self.b.requires_grad() {
            Some(gb.clone())
        } else {
            None
        };

        let grad_a = if self.a.requires_grad() {
            // gA = -gB @ X^T. `mm_bt` is the resident A @ B^T path for f32/f64.
            let (gb_m, x_m) = if self.vector_rhs {
                let n = gb.shape()[0];
                (
                    gb.view_reshape(vec![n, 1])?,
                    self.x.view_reshape(vec![self.x.shape()[0], 1])?,
                )
            } else {
                (gb.clone(), self.x.clone())
            };
            let prod = crate::autograd::no_grad::no_grad(|| gb_m.mm_bt(&x_m))?;
            Some(prod.neg_t()?)
        } else {
            None
        };

        Ok(vec![grad_a, grad_b])
    }
}

/// Differentiable `solve`. Attaches `LinalgSolveBackward` when grad is needed.
///
/// Forward computed under `no_grad`: `linalg_fwd::solve` (the public
/// `crate::linalg::solve` forward) delegates back here when grad is enabled, so
/// the guard prevents infinite re-entry.
pub fn solve_differentiable<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let result = crate::autograd::no_grad::no_grad(|| linalg_fwd::solve(a, b))?;
    if is_grad_enabled() && (a.requires_grad() || b.requires_grad()) {
        let grad_fn = Arc::new(LinalgSolveBackward {
            a: a.clone(),
            b: b.clone(),
            x: result.clone(),
            vector_rhs: b.ndim() == 1,
        });
        let (storage, shape) = result.into_storage_and_shape()?;
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// SolveTriangularBackward — X = triangular_solve(A, B)
// ---------------------------------------------------------------------------

/// Backward for `X = solve_triangular(A, B)`.
///
/// For the effective system `op(A) X = B`, PyTorch's VJP is the triangular
/// specialization of `linalg.solve`:
/// - `gB = op(A)^-T gX`
/// - `g op(A) = -gB X^T`
///
/// The gradient for `A` is then transposed back when the forward used
/// `transpose=true`, masked to the original declared triangle, and has a zero
/// diagonal for `unit_diagonal=true` because those entries are ignored by the
/// forward.
#[derive(Debug)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "stores torch.linalg.solve_triangular's boolean flags verbatim for backward"
)]
pub struct SolveTriangularBackward<T: Float> {
    a: Tensor<T>,
    b: Tensor<T>,
    x: Tensor<T>,
    upper: bool,
    transpose: bool,
    unit_diagonal: bool,
    vector_rhs: bool,
}

impl<T: Float> GradFn<T> for SolveTriangularBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if grad_output.is_cuda() {
            return self.backward_cuda(grad_output);
        }

        let n = self.a.shape()[0];
        let nrhs = if self.vector_rhs {
            1
        } else {
            self.b.shape()[1]
        };
        let go = from_cpu(grad_output.data_vec()?, grad_output.shape().to_vec())?;

        let grad_b_raw = crate::autograd::no_grad::no_grad(|| {
            linalg_fwd::solve_triangular(
                &self.a,
                &go,
                self.upper,
                !self.transpose,
                self.unit_diagonal,
            )
        })?;

        let grad_b = if self.b.requires_grad() {
            Some(grad_b_raw.clone())
        } else {
            None
        };

        let grad_a = if self.a.requires_grad() {
            let gb = if self.vector_rhs {
                from_cpu(grad_b_raw.data_vec()?, vec![n, 1])?
            } else {
                from_cpu(grad_b_raw.data_vec()?, vec![n, nrhs])?
            };
            let x = if self.vector_rhs {
                from_cpu(self.x.data_vec()?, vec![n, 1])?
            } else {
                from_cpu(self.x.data_vec()?, vec![n, nrhs])?
            };

            let gb_data = gb.data_vec()?;
            let x_data = x.data_vec()?;
            let mut grad_effective = mm_bt_rows(&gb_data, &x_data, n, nrhs, n);
            for v in &mut grad_effective {
                *v = -*v;
            }

            let grad_original = if self.transpose {
                transpose_rows(&grad_effective, n, n)
            } else {
                grad_effective
            };

            let zero = <T as num_traits::Zero>::zero();
            let mut masked = vec![zero; n * n];
            for r in 0..n {
                for c in 0..n {
                    if self.unit_diagonal && r == c {
                        continue;
                    }
                    let keep = if self.upper { c >= r } else { r >= c };
                    if keep {
                        masked[r * n + c] = grad_original[r * n + c];
                    }
                }
            }
            Some(from_cpu(masked, vec![n, n])?)
        } else {
            None
        };

        Ok(vec![grad_a, grad_b])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.a, &self.b]
    }

    fn name(&self) -> &'static str {
        "SolveTriangularBackward"
    }
}

impl<T: Float> SolveTriangularBackward<T> {
    fn backward_cuda(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !(is_f32::<T>() || is_f64::<T>()) {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "SolveTriangularBackward",
            });
        }

        let n = self.a.shape()[0];
        let grad_b_raw = crate::autograd::no_grad::no_grad(|| {
            linalg_fwd::solve_triangular(
                &self.a,
                grad_output,
                self.upper,
                !self.transpose,
                self.unit_diagonal,
            )
        })?;

        let grad_b = if self.b.requires_grad() {
            Some(grad_b_raw.clone())
        } else {
            None
        };

        let grad_a = if self.a.requires_grad() {
            let gb = if self.vector_rhs {
                grad_b_raw.view_reshape(vec![n, 1])?
            } else {
                grad_b_raw.clone()
            };
            let x = if self.vector_rhs {
                self.x.view_reshape(vec![n, 1])?
            } else {
                self.x.clone()
            };

            let grad_effective = crate::autograd::no_grad::no_grad(|| gb.mm_bt(&x))?.neg_t()?;
            let grad_original = if self.transpose {
                grad_effective.transpose(0, 1)?.contiguous()?
            } else {
                grad_effective
            };
            let diagonal = i64::from(self.unit_diagonal);
            let masked = crate::autograd::no_grad::no_grad(|| {
                if self.upper {
                    crate::ops::tensor_ops::triu(&grad_original, diagonal)
                } else {
                    crate::ops::tensor_ops::tril(&grad_original, -diagonal)
                }
            })?;
            Some(masked)
        } else {
            None
        };

        Ok(vec![grad_a, grad_b])
    }
}

/// Differentiable `solve_triangular`. CPU and CUDA match the public forward.
pub fn solve_triangular_differentiable<T: Float>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    upper: bool,
    transpose: bool,
    unit_diagonal: bool,
) -> FerrotorchResult<Tensor<T>> {
    let x = crate::autograd::no_grad::no_grad(|| {
        linalg_fwd::solve_triangular(a, b, upper, transpose, unit_diagonal)
    })?;
    if is_grad_enabled() && (a.requires_grad() || b.requires_grad()) {
        let grad_fn = Arc::new(SolveTriangularBackward {
            a: a.clone(),
            b: b.clone(),
            x: x.clone(),
            upper,
            transpose,
            unit_diagonal,
            vector_rhs: b.ndim() == 1,
        });
        let (storage, shape) = x.into_storage_and_shape()?;
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Ok(x)
    }
}

// ---------------------------------------------------------------------------
// MatrixExpBackward — Y = expm(A)
// ---------------------------------------------------------------------------

/// Backward for `Y = matrix_exp(A)`.
///
/// Uses the Frechet derivative adjoint identity:
/// `grad_A = L_exp(A^T, grad_Y)`, where `L_exp` is evaluated by the standard
/// block-matrix trick. The upper-right block of
/// `exp([[A^T, grad_Y], [0, A^T]])` is exactly the desired VJP.
#[derive(Debug)]
pub struct MatrixExpBackward<T: Float> {
    a: Tensor<T>,
}

impl<T: Float> GradFn<T> for MatrixExpBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if grad_output.is_cuda() {
            return self.backward_cuda(grad_output);
        }

        let n = self.a.shape()[0];
        if n == 0 {
            return Ok(vec![Some(from_cpu(Vec::new(), vec![0, 0])?)]);
        }

        let a_t = mat_transpose(&self.a)?;
        let a_t_data = a_t.data_vec()?;
        let g_data = grad_output.data_vec()?;
        let block_n = 2 * n;
        let zero = <T as num_traits::Zero>::zero();
        let mut block = vec![zero; block_n * block_n];
        for r in 0..n {
            for c in 0..n {
                block[r * block_n + c] = a_t_data[r * n + c];
                block[r * block_n + (n + c)] = g_data[r * n + c];
                block[(n + r) * block_n + (n + c)] = a_t_data[r * n + c];
            }
        }

        let block = from_cpu(block, vec![block_n, block_n])?;
        let exp_block = crate::autograd::no_grad::no_grad(|| linalg_fwd::matrix_exp(&block))?;
        let exp_data = exp_block.data_vec()?;
        let mut grad_a = vec![zero; n * n];
        for r in 0..n {
            for c in 0..n {
                grad_a[r * n + c] = exp_data[r * block_n + (n + c)];
            }
        }
        Ok(vec![Some(from_cpu(grad_a, vec![n, n])?)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.a]
    }

    fn name(&self) -> &'static str {
        "MatrixExpBackward"
    }
}

impl<T: Float> MatrixExpBackward<T> {
    fn backward_cuda(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !(is_f32::<T>() || is_f64::<T>()) {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "MatrixExpBackward",
            });
        }

        let n = self.a.shape()[0];
        if n == 0 {
            return Ok(vec![Some(crate::creation::zeros_like(&self.a)?)]);
        }

        let a_t = crate::autograd::no_grad::no_grad(|| self.a.transpose(0, 1)?.contiguous())?;
        let g = crate::autograd::no_grad::no_grad(|| grad_output.contiguous())?;
        let zero = crate::creation::zeros_like(&a_t)?;
        let top = crate::autograd::no_grad::no_grad(|| {
            crate::grad_fns::shape::cat(&[a_t.clone(), g], 1)
        })?;
        let bottom =
            crate::autograd::no_grad::no_grad(|| crate::grad_fns::shape::cat(&[zero, a_t], 1))?;
        let block =
            crate::autograd::no_grad::no_grad(|| crate::grad_fns::shape::cat(&[top, bottom], 0))?;
        let exp_block = crate::autograd::no_grad::no_grad(|| linalg_fwd::matrix_exp(&block))?;
        let grad_a = crate::autograd::no_grad::no_grad(|| {
            exp_block.narrow(0, 0, n)?.narrow(1, n, n)?.contiguous()
        })?;
        Ok(vec![Some(grad_a)])
    }
}

/// Differentiable `matrix_exp`. CPU and CUDA match the public forward.
pub fn matrix_exp_differentiable<T: Float>(a: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let y = crate::autograd::no_grad::no_grad(|| linalg_fwd::matrix_exp(a))?;
    if is_grad_enabled() && a.requires_grad() {
        let grad_fn = Arc::new(MatrixExpBackward { a: a.clone() });
        let (storage, shape) = y.into_storage_and_shape()?;
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Ok(y)
    }
}

// ---------------------------------------------------------------------------
// SlogdetBackward — (sign, logabsdet) = slogdet(A)   (2D square)
// ---------------------------------------------------------------------------

/// Backward for the differentiable output `logabsdet` of `slogdet(A)`.
///
/// VJP (`torch/csrc/autograd/FunctionsManual.cpp:4471` `slogdet_backward`,
/// real case — the formula `(g_abs - g_sign.conj()*sgn) * A^{-H}` collapses to
/// `g_abs * A^{-H}` because the real sign is locally constant, so
/// `grad_sign` contributes nothing):
/// `dA = grad_logabsdet * inv(A)^T`.
///
/// Per `tools/autograd/derivatives.yaml:559` `_linalg_slogdet`
/// (`output_differentiability: [True, True, False, False]`), the `sign`
/// output carries no real gradient; this node is attached to the `logabsdet`
/// output only.
#[derive(Debug)]
pub struct SlogdetBackward<T: Float> {
    /// Retained input. PyTorch's `_linalg_slogdet` stores LU metadata, not an
    /// eager inverse; singular tracked forwards must still produce `(0,-inf)`.
    input: Tensor<T>,
}

impl<T: Float> GradFn<T> for SlogdetBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // grad_output is the upstream gradient on `logabsdet` (a scalar).
        let g: T = grad_output.item()?;
        let grad_a = match crate::autograd::no_grad::no_grad(|| linalg_fwd::inv(&self.input)) {
            Ok(inv) => {
                let inv_t = mat_transpose(&inv)?;
                let data = inv_t.data_vec()?;
                let scaled: Vec<T> = data.iter().map(|&v| g * v).collect();
                Tensor::from_storage(TensorStorage::cpu(scaled), inv_t.shape().to_vec(), false)?
            }
            Err(err) if crate::autograd::higher_order::is_create_graph_enabled() => {
                // PyTorch's higher-order slogdet branch recomputes a solve and
                // raises on singular matrices; do not hide that nonsmooth case.
                return Err(err);
            }
            Err(_) => {
                let n = self.input.shape()[0];
                let data = self.input.data_vec()?;
                let det =
                    crate::autograd::no_grad::no_grad(|| linalg_fwd::det(&self.input))?.item()?;
                let scaled: Vec<T> = det_cofactor_matrix(&data, n)
                    .into_iter()
                    .map(|v| g * (v / det))
                    .collect();
                Tensor::from_storage(TensorStorage::cpu(scaled), vec![n, n], false)?
            }
        };
        Ok(vec![Some(grad_a)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![]
    }

    fn name(&self) -> &'static str {
        "SlogdetBackward"
    }
}

/// Carries the input edge for `slogdet`'s differentiable `logabsdet` output.
#[derive(Debug)]
struct SlogdetForward<T: Float> {
    input: Tensor<T>,
    inner: SlogdetBackward<T>,
}

impl<T: Float> GradFn<T> for SlogdetForward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        self.inner.backward(grad_output)
    }
    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }
    fn name(&self) -> &'static str {
        "SlogdetBackward"
    }
}

/// Differentiable `slogdet`. Returns `(sign, logabsdet)`; the `sign` output is
/// non-differentiable in the real case (locally constant), so only the
/// `logabsdet` output carries `SlogdetBackward`. Attaches it when grad is
/// needed.
pub fn slogdet_differentiable<T: Float>(a: &Tensor<T>) -> FerrotorchResult<(Tensor<T>, Tensor<T>)> {
    // Forward computed under `no_grad`: `linalg_fwd::slogdet` delegates back
    // here when grad is enabled, so the guard prevents infinite re-entry.
    let (sign, logabsdet) = crate::autograd::no_grad::no_grad(|| linalg_fwd::slogdet(a))?;
    if is_grad_enabled() && a.requires_grad() {
        let grad_fn = Arc::new(SlogdetForward {
            input: a.clone(),
            inner: SlogdetBackward { input: a.clone() },
        });
        let (storage, shape) = logabsdet.into_storage_and_shape()?;
        let logabsdet = Tensor::from_operation(storage, shape, grad_fn)?;
        Ok((sign, logabsdet))
    } else {
        Ok((sign, logabsdet))
    }
}

// ---------------------------------------------------------------------------
// CholeskyBackward — L = cholesky(A)   (2D square SPD, lower-triangular L)
// ---------------------------------------------------------------------------

/// Lower-triangular projection of an `n×n` row-major matrix (keep `c <= r`).
fn tril_cpu<T: Float>(x: &[T], n: usize) -> Vec<T> {
    let zero = <T as num_traits::Zero>::zero();
    let mut out = vec![zero; n * n];
    for r in 0..n {
        for c in 0..=r {
            out[r * n + c] = x[r * n + c];
        }
    }
    out
}

/// Backward for `L = cholesky(A)` (lower-triangular factor, `A = L @ L^T`).
///
/// VJP (`torch/csrc/autograd/FunctionsManual.cpp:2048` `cholesky_backward`,
/// real lower case):
/// 1. `P = tril(L^T @ gL)`                       (the `gA = L_.mH().matmul(gL_).tril()` step)
/// 2. `S = 0.5 * (P + strictly_lower(P)^T)`       (Phi-symmetrisation:
///    `0.5*(gA + gA.tril(-1).mH())`)
/// 3. `S = solve_triangular(L^T, S, upper=true, left=true)`   (`L^{-T} @ S`)
/// 4. `gA = solve_triangular(L,  S, upper=false, left=false)` (`S @ L^{-1}`)
///
/// The result is symmetric (not triangular), matching PyTorch.
#[derive(Debug)]
pub struct CholeskyBackward<T: Float> {
    /// Retained lower-triangular factor `L`.
    l: Tensor<T>,
}

impl<T: Float> GradFn<T> for CholeskyBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if grad_output.is_cuda() {
            return self.backward_cuda(grad_output);
        }

        let n = self.l.shape()[0];
        let l = self.l.data()?;
        let gl = grad_output.data()?;
        let zero = <T as num_traits::Zero>::zero();
        let half = T::from(0.5).unwrap();

        // Step 1: P = tril(L^T @ gL). L^T @ gL has entry [i,j] = sum_k L[k,i]*gL[k,j].
        let mut ltgl = vec![zero; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut acc = zero;
                for k in 0..n {
                    acc += l[k * n + i] * gl[k * n + j];
                }
                ltgl[i * n + j] = acc;
            }
        }
        let p = tril_cpu(&ltgl, n);

        // Step 2: S = 0.5 * (P + strictly_lower(P)^T). strictly_lower(P)[r,c] is
        // P[r,c] for c < r; its transpose contributes to the upper triangle.
        let mut s = vec![zero; n * n];
        for r in 0..n {
            for c in 0..n {
                let mut v = p[r * n + c];
                if c > r {
                    // upper triangle: strictly-lower(P)^T at [r,c] = P[c,r].
                    v += p[c * n + r];
                }
                s[r * n + c] = half * v;
            }
        }
        let s_t = Tensor::from_storage(TensorStorage::cpu(s), vec![n, n], false)?;

        // Step 3: S <- L^{-T} @ S  ≡ solve_triangular(L^T, S, upper=true, left=true).
        // The forward `solve_triangular` solves the LEFT system (A x = b) with
        // `transpose` folding A^T. With upper=false (L is lower) + transpose=true
        // we solve L^T x = S, i.e. x = L^{-T} S.
        let s = crate::autograd::no_grad::no_grad(|| {
            linalg_fwd::solve_triangular(&self.l, &s_t, false, true, false)
        })?;

        // Step 4: gA <- S @ L^{-1}  ≡ (L^{-T} @ S^T)^T. Solve_triangular only
        // does LEFT solves, so right-solve by transposing: S @ L^{-1} =
        // (L^{-T} @ S^T)^T = ((L^T)^{-1} S^T)^T.
        let s_tt = mat_transpose(&s)?;
        let right = crate::autograd::no_grad::no_grad(|| {
            linalg_fwd::solve_triangular(&self.l, &s_tt, false, true, false)
        })?;
        let grad_a = mat_transpose(&right)?;

        Ok(vec![Some(grad_a)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        // VJP closes over the retained `L`; the graph edge to leaf `A` is
        // carried by `CholeskyForward`.
        vec![]
    }

    fn name(&self) -> &'static str {
        "CholeskyBackward"
    }
}

impl<T: Float> CholeskyBackward<T> {
    fn backward_cuda(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !(is_f32::<T>() || is_f64::<T>()) {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "CholeskyBackward",
            });
        }

        let half = T::from(0.5).ok_or_else(|| FerrotorchError::InvalidArgument {
            message: "cholesky backward: 0.5 is not representable in dtype".into(),
        })?;

        let grad_a = crate::autograd::no_grad::no_grad(|| {
            // Step 1: P = tril(L^T @ gL).
            let lt = self.l.transpose(0, 1)?.contiguous()?;
            let ltgl = lt.mm(grad_output)?;
            let p = crate::ops::tensor_ops::tril(&ltgl, 0)?;

            // Step 2: S = 0.5 * (P + strictly_lower(P)^T).
            let strict_lower = crate::ops::tensor_ops::tril(&p, -1)?;
            let strict_lower_t = strict_lower.transpose(0, 1)?.contiguous()?;
            let sym = p.add_t(&strict_lower_t)?;
            let half_t = crate::creation::full_like(&sym, half)?;
            let s = sym.mul_t(&half_t)?;

            // Step 3: S <- L^{-T} @ S. Full cuSOLVER solve against the
            // triangular factor keeps the path resident without a CPU
            // triangular-solve fallback.
            let s = linalg_fwd::solve(&lt, &s)?;

            // Step 4: gA <- S @ L^{-1} = (L^{-T} @ S^T)^T.
            let s_t = s.transpose(0, 1)?.contiguous()?;
            let right = linalg_fwd::solve(&lt, &s_t)?;
            right.transpose(0, 1)?.contiguous()
        })?;

        Ok(vec![Some(grad_a)])
    }
}

/// Carries the input edge for `cholesky`.
#[derive(Debug)]
struct CholeskyForward<T: Float> {
    input: Tensor<T>,
    inner: CholeskyBackward<T>,
}

impl<T: Float> GradFn<T> for CholeskyForward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        self.inner.backward(grad_output)
    }
    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }
    fn name(&self) -> &'static str {
        "CholeskyBackward"
    }
}

/// Differentiable `cholesky`. Attaches `CholeskyBackward` (Phi-symmetrisation
/// VJP) when grad is needed. Lower-triangular factor only (`A = L @ L^T`).
pub fn cholesky_differentiable<T: Float>(a: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    // Compute the forward inside `no_grad`: `linalg_fwd::cholesky` itself
    // delegates back here when grad is enabled, so the guard prevents
    // infinite re-entry.
    let l = crate::autograd::no_grad::no_grad(|| linalg_fwd::cholesky(a))?;
    if is_grad_enabled() && a.requires_grad() {
        let grad_fn = Arc::new(CholeskyForward {
            input: a.clone(),
            inner: CholeskyBackward { l: l.clone() },
        });
        let (storage, shape) = l.into_storage_and_shape()?;
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Ok(l)
    }
}

// ---------------------------------------------------------------------------
// QrBackward — (Q, R) = qr(A, mode='reduced')   (2D, m >= n)
// ---------------------------------------------------------------------------

/// `syminvadj(X) = X + X^T` with the diagonal halved (real case of
/// `linalg_qr_backward`'s `syminvadj`). `x` is row-major `n×n`.
fn syminvadj_cpu<T: Float>(x: &[T], n: usize) -> Vec<T> {
    let zero = <T as num_traits::Zero>::zero();
    let half = T::from(0.5).unwrap();
    let mut out = vec![zero; n * n];
    for r in 0..n {
        for c in 0..n {
            let v = x[r * n + c] + x[c * n + r];
            out[r * n + c] = if r == c { half * v } else { v };
        }
    }
    out
}

/// Strict-upper + diagonal projection (`triu`, keep `c >= r`) of a row-major
/// `n×n` matrix.
fn triu_cpu<T: Float>(x: &[T], n: usize) -> Vec<T> {
    let zero = <T as num_traits::Zero>::zero();
    let mut out = vec![zero; n * n];
    for r in 0..n {
        for c in r..n {
            out[r * n + c] = x[r * n + c];
        }
    }
    out
}

fn symmetrize_cuda<T: Float>(x: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let half = T::from(0.5).ok_or_else(|| FerrotorchError::InvalidArgument {
        message: "symmetric linalg backward: 0.5 is not representable in dtype".into(),
    })?;
    let x_t = x.transpose(0, 1)?.contiguous()?;
    let sum = x.add_t(&x_t)?;
    let half_t = crate::creation::full_like(&sum, half)?;
    sum.mul_t(&half_t)
}

/// Shared backward for `(Q, R) = qr(A, 'reduced')`, real case `m >= n`.
///
/// VJP (`torch/csrc/autograd/FunctionsManual.cpp:4166` `linalg_qr_backward`,
/// `m >= n` branch):
/// `gA = (Q @ syminvadj(triu(M)) + gQ_term) @ R^{-T}`, where
/// `M = gR @ R^T - Q^T @ gQ` and the trailing `gQ` is added only when `gQ` is
/// defined. Implemented as a right triangular solve:
/// `gA = solve_triangular(R^T, rhs, upper=false, left=false)`.
///
/// Because ferrotorch's autograd engine drives one `grad_output` per node, the
/// jointly-linear `(gQ, gR)` VJP is split across two nodes — `QrBackwardQ`
/// (the `gQ`-only contribution, attached to the `Q` output) and `QrBackwardR`
/// (the `gR`-only contribution, attached to the `R` output). The engine
/// accumulates both partials into `A.grad`, reproducing the joint formula
/// (which is additive in `gQ` and `gR`). If a consumer uses only one output,
/// the other partial is simply absent — matching PyTorch's undefined-grad
/// (zero) semantics.
#[derive(Debug)]
struct QrBackwardShared<T: Float> {
    q: Tensor<T>,
    r: Tensor<T>,
}

impl<T: Float> QrBackwardShared<T> {
    fn reject_unsupported_cuda() -> FerrotorchResult<()> {
        if !(is_f32::<T>() || is_f64::<T>()) {
            return Err(FerrotorchError::NotImplementedOnCuda { op: "QrBackward" });
        }
        Ok(())
    }

    fn syminvadj_triu_cuda(x: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let upper = crate::ops::tensor_ops::triu(x, 0)?;
        let strict_upper = crate::ops::tensor_ops::triu(&upper, 1)?;
        let strict_upper_t = strict_upper.transpose(0, 1)?.contiguous()?;
        upper.add_t(&strict_upper_t)
    }

    fn finish_right_solve_cuda(&self, rhs: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        Self::reject_unsupported_cuda()?;
        let rhs_t = rhs.transpose(0, 1)?.contiguous()?;
        let y = linalg_fwd::solve(&self.r, &rhs_t)?;
        y.transpose(0, 1)?.contiguous()
    }

    /// Compute `gA = solve_triangular(R^T, rhs, upper=false, left=false)` where
    /// `rhs` is the per-branch right-hand-side matrix shaped `[m, n]`.
    fn finish_right_solve(
        &self,
        rhs: &Tensor<T>,
        m: usize,
        n: usize,
    ) -> FerrotorchResult<Tensor<T>> {
        // Right solve X @ R^{-T}: solve_triangular only does LEFT solves, so
        // X @ R^{-T} = (R^{-1} @ X^T)^T. R is upper-triangular [n,n]; we solve
        // R y = X^T (upper, no transpose) and transpose back.
        let rhs_t = mat_transpose(rhs)?; // [n, m]
        let y = crate::autograd::no_grad::no_grad(|| {
            linalg_fwd::solve_triangular(&self.r, &rhs_t, true, false, false)
        })?; // [n, m]
        let ga = mat_transpose(&y)?; // [m, n]
        debug_assert_eq!(ga.shape(), &[m, n]);
        Ok(ga)
    }

    /// `gQ`-only contribution to `gA` (set `gR = 0`):
    /// `M = -(Q^T @ gQ)`; `rhs = Q @ syminvadj(triu(M)) + gQ`.
    fn grad_a_from_gq(&self, gq: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if gq.is_cuda() {
            return crate::autograd::no_grad::no_grad(|| {
                Self::reject_unsupported_cuda()?;
                let qt = self.q.transpose(0, 1)?.contiguous()?;
                let mmat = qt.mm(gq)?.neg_t()?;
                let sym = Self::syminvadj_triu_cuda(&mmat)?;
                let rhs = self.q.mm(&sym)?.add_t(gq)?;
                self.finish_right_solve_cuda(&rhs)
            });
        }

        let m = self.q.shape()[0];
        let n = self.r.shape()[1];
        let q = self.q.data()?;
        let gqd = gq.data()?;
        let zero = <T as num_traits::Zero>::zero();

        // M = -(Q^T @ gQ): [n,n], M[i,j] = -sum_k Q[k,i]*gQ[k,j].
        let mut mmat = vec![zero; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut acc = zero;
                for k in 0..m {
                    acc += q[k * n + i] * gqd[k * n + j];
                }
                mmat[i * n + j] = -acc;
            }
        }
        let sym = syminvadj_cpu(&triu_cpu(&mmat, n), n); // [n,n]

        // rhs = Q @ sym + gQ: [m,n], (Q[m,n] @ sym[n,n]) + gQ[m,n].
        let mut rhs = vec![zero; m * n];
        for r in 0..m {
            for c in 0..n {
                let mut acc = zero;
                for k in 0..n {
                    acc += q[r * n + k] * sym[k * n + c];
                }
                rhs[r * n + c] = acc + gqd[r * n + c];
            }
        }
        let rhs = Tensor::from_storage(TensorStorage::cpu(rhs), vec![m, n], false)?;
        self.finish_right_solve(&rhs, m, n)
    }

    /// `gR`-only contribution to `gA` (set `gQ = 0`):
    /// `M = gR @ R^T`; `rhs = Q @ syminvadj(triu(M))`.
    fn grad_a_from_gr(&self, gr: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if gr.is_cuda() {
            return crate::autograd::no_grad::no_grad(|| {
                Self::reject_unsupported_cuda()?;
                let mmat = gr.mm_bt(&self.r)?;
                let sym = Self::syminvadj_triu_cuda(&mmat)?;
                let rhs = self.q.mm(&sym)?;
                self.finish_right_solve_cuda(&rhs)
            });
        }

        let m = self.q.shape()[0];
        let n = self.r.shape()[1];
        let q = self.q.data()?;
        let r = self.r.data()?;
        let grd = gr.data()?;
        let zero = <T as num_traits::Zero>::zero();

        // M = gR @ R^T: [n,n], M[i,j] = sum_k gR[i,k]*R[j,k].
        let mut mmat = vec![zero; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut acc = zero;
                for k in 0..n {
                    acc += grd[i * n + k] * r[j * n + k];
                }
                mmat[i * n + j] = acc;
            }
        }
        let sym = syminvadj_cpu(&triu_cpu(&mmat, n), n); // [n,n]

        // rhs = Q @ sym: [m,n].
        let mut rhs = vec![zero; m * n];
        for rr in 0..m {
            for c in 0..n {
                let mut acc = zero;
                for k in 0..n {
                    acc += q[rr * n + k] * sym[k * n + c];
                }
                rhs[rr * n + c] = acc;
            }
        }
        let rhs = Tensor::from_storage(TensorStorage::cpu(rhs), vec![m, n], false)?;
        self.finish_right_solve(&rhs, m, n)
    }
}

/// `gQ`-only QR backward node, attached to the `Q` output.
#[derive(Debug)]
struct QrBackwardQ<T: Float> {
    input: Tensor<T>,
    shared: QrBackwardShared<T>,
}

impl<T: Float> GradFn<T> for QrBackwardQ<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        Ok(vec![Some(self.shared.grad_a_from_gq(grad_output)?)])
    }
    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }
    fn name(&self) -> &'static str {
        "QrBackward"
    }
}

/// `gR`-only QR backward node, attached to the `R` output.
#[derive(Debug)]
struct QrBackwardR<T: Float> {
    input: Tensor<T>,
    shared: QrBackwardShared<T>,
}

impl<T: Float> GradFn<T> for QrBackwardR<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        Ok(vec![Some(self.shared.grad_a_from_gr(grad_output)?)])
    }
    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }
    fn name(&self) -> &'static str {
        "QrBackward"
    }
}

/// Differentiable `qr` (reduced mode, real, `m >= n`). Attaches the split
/// `QrBackwardQ` / `QrBackwardR` nodes (whose `A.grad` contributions the
/// autograd engine accumulates) when grad is needed.
///
/// Mirrors `torch.linalg.qr(A, mode='reduced')`. The `m < n` case is rejected
/// here: its VJP (`trilImInvAdjSkew`) is the separate research-grade branch of
/// `linalg_qr_backward` tracked under the hard-ops sub-blocker.
pub fn qr_differentiable<T: Float>(a: &Tensor<T>) -> FerrotorchResult<(Tensor<T>, Tensor<T>)> {
    // Forward under `no_grad`: `linalg_fwd::qr` delegates back here when grad
    // is enabled, so the guard prevents infinite re-entry.
    let (q, r) = crate::autograd::no_grad::no_grad(|| linalg_fwd::qr(a))?;
    let needs_grad = is_grad_enabled() && a.requires_grad();
    if !needs_grad {
        return Ok((q, r));
    }
    let m = q.shape()[0];
    let n = r.shape()[1];
    if m < n {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "qr backward (mode='reduced') is implemented for m >= n only; \
                 got A shape [{m}, {n}] (the m<n trilImInvAdjSkew branch is \
                 tracked under the hard-ops sub-blocker)"
            ),
        });
    }
    let q_node = Arc::new(QrBackwardQ {
        input: a.clone(),
        shared: QrBackwardShared {
            q: q.clone(),
            r: r.clone(),
        },
    });
    let r_node = Arc::new(QrBackwardR {
        input: a.clone(),
        shared: QrBackwardShared {
            q: q.clone(),
            r: r.clone(),
        },
    });
    let (q_storage, q_shape) = q.into_storage_and_shape()?;
    let (r_storage, r_shape) = r.into_storage_and_shape()?;
    let q = Tensor::from_operation(q_storage, q_shape, q_node)?;
    let r = Tensor::from_operation(r_storage, r_shape, r_node)?;
    Ok((q, r))
}

// ===========================================================================
// Research-grade decomposition backwards (#1577): eigvalsh / eigh / pinv /
// lstsq / lu / lu_factor + the clean linalg.cross / linalg.norm VJPs (#1345
// subset). Each is a closed-form / algebraic VJP grounded in a named PyTorch
// `file:line` and FD-verified in this file's `#[cfg(test)] mod tests`.
//
// Some CPU VJPs in this section operate on dense f32/f64 matrices via raw-slice
// GEMM helpers (`mm_rows`/`mm_bt_rows`/`mm_at_rows`) and the existing
// `mat_transpose` / `solve_triangular` / `linalg_fwd::*` forwards. CUDA-capable
// branches compose tensor operations directly so CUDA inputs remain resident.
// All forward recomputation is guarded by `no_grad`.
// ===========================================================================

/// Transpose a row-major `r×c` matrix into a `c×r` matrix.
fn transpose_rows<T: Float>(x: &[T], r: usize, c: usize) -> Vec<T> {
    let zero = <T as num_traits::Zero>::zero();
    let mut out = vec![zero; r * c];
    for i in 0..r {
        for j in 0..c {
            out[j * r + i] = x[i * c + j];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// EigvalshBackward — w = eigvalsh(A)  (symmetric A, eigenvalues only)
// ---------------------------------------------------------------------------

/// Backward for `w = eigvalsh(A)` (symmetric / Hermitian eigenvalues only).
///
/// VJP (`torch/csrc/autograd/FunctionsManual.cpp:3859` `linalg_eig_backward`,
/// Hermitian eigenvalues-only shortcut `at::matmul(V * gL.unsqueeze(-2),
/// V.mH())`): `gA = U @ diag(gw) @ U^T`. Because the eigenvalues of a symmetric
/// matrix are differentiable functions of `A` with NO eigenvector-gauge
/// freedom and NO degenerate-eigenvalue issue (the eigenvalue map is smooth),
/// this VJP is exact. PyTorch returns the gradient symmetrized (the UPLO
/// contract reads only one triangle of `A`); we symmetrize `0.5*(G + G^T)` to
/// match.
#[derive(Debug)]
pub struct EigvalshBackward<T: Float> {
    /// Eigenvector matrix `U` (columns are eigenvectors), retained for the VJP.
    u: Tensor<T>,
}

impl<T: Float> GradFn<T> for EigvalshBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if grad_output.is_cuda() {
            return self.backward_cuda(grad_output);
        }

        let n = self.u.shape()[0];
        let u = self.u.data()?;
        let gw = grad_output.data()?; // [n]
        // tmp = U @ diag(gw):  tmp[i,j] = U[i,j] * gw[j].
        let mut tmp = vec![<T as num_traits::Zero>::zero(); n * n];
        for i in 0..n {
            for j in 0..n {
                tmp[i * n + j] = u[i * n + j] * gw[j];
            }
        }
        // G = tmp @ U^T:  G[i,k] = sum_j tmp[i,j] * U[k,j].
        let g = mm_bt_rows(&tmp, u, n, n, n);
        // Symmetrize: 0.5*(G + G^T) — matches PyTorch's UPLO-triangle contract.
        let half = T::from(0.5).unwrap();
        let mut gsym = vec![<T as num_traits::Zero>::zero(); n * n];
        for i in 0..n {
            for j in 0..n {
                gsym[i * n + j] = half * (g[i * n + j] + g[j * n + i]);
            }
        }
        Ok(vec![Some(from_cpu(gsym, vec![n, n])?)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![]
    }

    fn name(&self) -> &'static str {
        "EigvalshBackward"
    }
}

impl<T: Float> EigvalshBackward<T> {
    fn backward_cuda(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !(is_f32::<T>() || is_f64::<T>()) {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "EigvalshBackward",
            });
        }

        let grad_a = crate::autograd::no_grad::no_grad(|| {
            let ret = crate::ops::tensor_ops::diag(grad_output, 0)?;
            let tmp = self.u.mm(&ret)?;
            let g = tmp.mm_bt(&self.u)?;
            symmetrize_cuda(&g)
        })?;
        Ok(vec![Some(grad_a)])
    }
}

/// Carries the input edge for `eigvalsh` (the VJP closes over the retained
/// eigenvectors only).
#[derive(Debug)]
struct EigvalshForward<T: Float> {
    input: Tensor<T>,
    inner: EigvalshBackward<T>,
}

impl<T: Float> GradFn<T> for EigvalshForward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        self.inner.backward(grad_output)
    }
    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }
    fn name(&self) -> &'static str {
        "EigvalshBackward"
    }
}

/// Differentiable `eigvalsh`. Attaches `EigvalshBackward` when grad is needed.
///
/// Forward (and the eigenvector solve the VJP needs) computed under `no_grad`:
/// `linalg_fwd::eigvalsh` delegates back here when grad is enabled, so the
/// guard prevents infinite re-entry. The eigenvectors `U` are obtained from
/// `linalg_fwd::eigh` (also under `no_grad`).
pub fn eigvalsh_differentiable<T: Float>(a: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let w = crate::autograd::no_grad::no_grad(|| linalg_fwd::eigvalsh(a))?;
    if is_grad_enabled() && a.requires_grad() {
        let (_w2, u) = crate::autograd::no_grad::no_grad(|| linalg_fwd::eigh(a))?;
        let grad_fn = Arc::new(EigvalshForward {
            input: a.clone(),
            inner: EigvalshBackward { u },
        });
        let (storage, shape) = w.into_storage_and_shape()?;
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Ok(w)
    }
}

// ---------------------------------------------------------------------------
// EighBackward — (w, U) = eigh(A)  (symmetric A, eigenvalues + eigenvectors)
// ---------------------------------------------------------------------------

/// Shared real symmetric eigh VJP, split across two single-output nodes
/// (`EighBackwardW` on the eigenvalues, `EighBackwardV` on the eigenvectors).
///
/// VJP (`torch/csrc/autograd/FunctionsManual.cpp:3882-3917`
/// `linalg_eig_backward`, Hermitian branch, real case):
/// 1. `VhgV = U^T @ gU`
/// 2. skew-symmetric projection `VhgV <- 0.5 * (VhgV - VhgV^T)`
/// 3. divide by `Econj_{ij} = w_j - w_i` off-diagonal, `1` on the diagonal
/// 4. write `gw` onto the diagonal (eigenvalue contribution)
/// 5. `gA = U @ ret @ U^T`
///
/// Because the two outputs `(w, U)` are jointly linear in `gA`, the engine
/// accumulates the `EighBackwardW` (`gU=0`) and `EighBackwardV` (`gw=0`)
/// partials into `A.grad` — the same split-node strategy QR uses. The result
/// is symmetric (PyTorch's UPLO contract); we symmetrize to match. This VJP is
/// EXACT on inputs with distinct eigenvalues; on degenerate inputs the `Econj`
/// off-diagonal `1/(w_j - w_i)` blows up exactly as PyTorch's does (PyTorch
/// does not special-case degeneracy in `linalg_eig_backward`, it simply
/// divides — the JVP/VJP are ill-defined at a degeneracy and the caller is
/// responsible for perturbing).
#[derive(Debug)]
struct EighBackwardShared<T: Float> {
    /// Eigenvalues `w` (ascending), retained for the `Econj` denominator.
    w: Tensor<T>,
    /// Eigenvector matrix `U`, retained for the conjugation `U @ ret @ U^T`.
    u: Tensor<T>,
}

impl<T: Float> EighBackwardShared<T> {
    fn reject_unsupported_cuda() -> FerrotorchResult<()> {
        if !(is_f32::<T>() || is_f64::<T>()) {
            return Err(FerrotorchError::NotImplementedOnCuda { op: "EighBackward" });
        }
        Ok(())
    }

    fn conjugate_cuda(&self, ret: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        Self::reject_unsupported_cuda()?;
        let tmp = self.u.mm(ret)?;
        let g = tmp.mm_bt(&self.u)?;
        symmetrize_cuda(&g)
    }

    /// `gA = U @ ret @ U^T`, where `ret` is the `n×n` middle factor, then
    /// symmetrize `0.5*(gA + gA^T)`.
    fn conjugate(&self, ret: &[T], n: usize) -> FerrotorchResult<Tensor<T>> {
        let u = self.u.data()?;
        let tmp = mm_rows(u, ret, n, n, n); // U @ ret
        let g = mm_bt_rows(&tmp, u, n, n, n); // (U @ ret) @ U^T
        let half = T::from(0.5).unwrap();
        let mut gsym = vec![<T as num_traits::Zero>::zero(); n * n];
        for i in 0..n {
            for j in 0..n {
                gsym[i * n + j] = half * (g[i * n + j] + g[j * n + i]);
            }
        }
        from_cpu(gsym, vec![n, n])
    }

    /// `gw`-only contribution: `ret = diag(gw)`, then conjugate.
    fn grad_a_from_gw(&self, gw: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if gw.is_cuda() {
            return crate::autograd::no_grad::no_grad(|| {
                let ret = crate::ops::tensor_ops::diag(gw, 0)?;
                self.conjugate_cuda(&ret)
            });
        }

        let n = self.u.shape()[0];
        let gwd = gw.data()?;
        let mut ret = vec![<T as num_traits::Zero>::zero(); n * n];
        for i in 0..n {
            ret[i * n + i] = gwd[i];
        }
        self.conjugate(&ret, n)
    }

    /// `gU`-only contribution: skew-project `U^T gU`, divide by `Econj`,
    /// then conjugate.
    fn grad_a_from_gu(&self, gu: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if gu.is_cuda() {
            return crate::autograd::no_grad::no_grad(|| {
                Self::reject_unsupported_cuda()?;
                let n = self.u.shape()[0];
                let half = T::from(0.5).ok_or_else(|| FerrotorchError::InvalidArgument {
                    message: "eigh backward: 0.5 is not representable in dtype".into(),
                })?;
                let u_t = self.u.transpose(0, 1)?.contiguous()?;
                let vhgv = u_t.mm(gu)?;
                let vhgv_t = vhgv.transpose(0, 1)?.contiguous()?;
                let skew_raw = vhgv.sub_t(&vhgv_t)?;
                let half_t = crate::creation::full_like(&skew_raw, half)?;
                let skew = skew_raw.mul_t(&half_t)?;

                let w_row = self.w.view_reshape(vec![1, n])?;
                let w_col = self.w.view_reshape(vec![n, 1])?;
                let gaps = w_row.sub_t(&w_col)?;
                let eye = crate::creation::eye::<T>(n)?.to(self.w.device())?;
                let denom = gaps.add_t(&eye)?;
                let ret = skew.div_t(&denom)?;
                self.conjugate_cuda(&ret)
            });
        }

        let n = self.u.shape()[0];
        let u = self.u.data()?;
        let gud = gu.data()?;
        let w = self.w.data()?;
        let zero = <T as num_traits::Zero>::zero();
        let half = T::from(0.5).unwrap();

        // VhgV = U^T @ gU:  [n,n], VhgV[i,j] = sum_k U[k,i]*gU[k,j].
        let vhgv = mm_at_rows(u, gud, n, n, n);
        // Skew-symmetric projection: 0.5*(VhgV - VhgV^T).
        let mut sk = vec![zero; n * n];
        for i in 0..n {
            for j in 0..n {
                sk[i * n + j] = half * (vhgv[i * n + j] - vhgv[j * n + i]);
            }
        }
        // Divide by Econj_{ij} = w_j - w_i off-diagonal, 1 on the diagonal.
        // The diagonal of `sk` is already 0 (skew), so dividing by 1 keeps it 0.
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    sk[i * n + j] = sk[i * n + j] / (w[j] - w[i]);
                }
            }
        }
        self.conjugate(&sk, n)
    }
}

/// `gw`-only eigh backward node, attached to the eigenvalues output.
#[derive(Debug)]
struct EighBackwardW<T: Float> {
    input: Tensor<T>,
    shared: EighBackwardShared<T>,
}

impl<T: Float> GradFn<T> for EighBackwardW<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        Ok(vec![Some(self.shared.grad_a_from_gw(grad_output)?)])
    }
    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }
    fn name(&self) -> &'static str {
        "EighBackward"
    }
}

/// `gU`-only eigh backward node, attached to the eigenvectors output.
#[derive(Debug)]
struct EighBackwardV<T: Float> {
    input: Tensor<T>,
    shared: EighBackwardShared<T>,
}

impl<T: Float> GradFn<T> for EighBackwardV<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        Ok(vec![Some(self.shared.grad_a_from_gu(grad_output)?)])
    }
    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }
    fn name(&self) -> &'static str {
        "EighBackward"
    }
}

/// Differentiable `eigh` (symmetric, real, distinct eigenvalues). Attaches the
/// split `EighBackwardW` / `EighBackwardV` nodes (whose `A.grad` contributions
/// the autograd engine accumulates) when grad is needed.
///
/// Forward computed under `no_grad` (re-entry guard). The eigenvector-gauge
/// freedom is real for symmetric matrices but the loss-invariance assumption
/// (the gradient lives in the skew-symmetric tangent space) means the VJP is
/// well-defined whenever the eigenvalues are distinct; on a degenerate input
/// the `Econj` denominator `1/(w_j-w_i)` diverges exactly as PyTorch's does.
pub fn eigh_differentiable<T: Float>(a: &Tensor<T>) -> FerrotorchResult<(Tensor<T>, Tensor<T>)> {
    let (w, u) = crate::autograd::no_grad::no_grad(|| linalg_fwd::eigh(a))?;
    let needs_grad = is_grad_enabled() && a.requires_grad();
    if !needs_grad {
        return Ok((w, u));
    }
    let w_node = Arc::new(EighBackwardW {
        input: a.clone(),
        shared: EighBackwardShared {
            w: w.clone(),
            u: u.clone(),
        },
    });
    let v_node = Arc::new(EighBackwardV {
        input: a.clone(),
        shared: EighBackwardShared {
            w: w.clone(),
            u: u.clone(),
        },
    });
    let (w_storage, w_shape) = w.into_storage_and_shape()?;
    let (u_storage, u_shape) = u.into_storage_and_shape()?;
    let w = Tensor::from_operation(w_storage, w_shape, w_node)?;
    let u = Tensor::from_operation(u_storage, u_shape, v_node)?;
    Ok((w, u))
}

// ---------------------------------------------------------------------------
// SvdBackward — (U, S, Vh) = svd(A, full_matrices=False)   (2D, reduced SVD)
// ---------------------------------------------------------------------------

/// Shared real reduced-SVD VJP, split across three single-output nodes
/// (`SvdBackwardU` on `U`, `SvdBackwardS` on `S`, `SvdBackwardV` on `Vh`).
///
/// For `A = U diag(S) Vh` with `A` `m×n`, `U` `m×k`, `S` `k`, `Vh` `k×n`,
/// `k = min(m, n)`, this mirrors `svd_backward` at
/// `torch/csrc/autograd/FunctionsManual.cpp:3605` (the REAL case, where
/// `skew(X) = X - X^T` and `^H` is the plain transpose):
///
/// - `UhgU = skew(U^T @ gU)` (`k×k`), `VhgV = skew(Vh @ gVh^T)` (`k×k`)
/// - `E[i,j] = S^2[j] - S^2[i]` off-diagonal, `1` on the diagonal
///   (`FunctionsManual.cpp:3770-3777`: `S2.unsqueeze(-2) - S2.unsqueeze(-1)`,
///   diagonal then `fill_(1)`)
/// - core `ret` — both gU & gVh:
///   `ret[i,j] = (UhgU[i,j]*S[j] + S[i]*VhgV[i,j]) / E[i,j]`; gU only:
///   `ret[i,j] = UhgU[i,j]/E[i,j] * S[j]`; gVh only:
///   `ret[i,j] = S[i] * VhgV[i,j]/E[i,j]`; then `ret += diag(gS)` when gS is
///   present (`FunctionsManual.cpp:3767-3797`).
/// - assembly (`FunctionsManual.cpp:3799-3815`) — for m > n & gU:
///   `gA = [U@ret + gU S^{-1} - U(U^T gU S^{-1})] @ Vh`; for m < n & gVh:
///   `gA = U @ [ret@Vh + S^{-1}gVh - (S^{-1}gVh Vh^T)Vh]`; else (square / no
///   projector): `gA = U @ ret @ Vh`.
///
/// The three outputs `(U, S, Vh)` are jointly linear in `gA`, so the engine
/// accumulates the `SvdBackwardU` (`gS=gVh=0`), `SvdBackwardS` (`gU=gVh=0`),
/// and `SvdBackwardV` (`gU=gS=0`) partials into `A.grad` — the same split-node
/// strategy QR (`QrBackwardQ`/`QrBackwardR`) and eigh
/// (`EighBackwardW`/`EighBackwardV`) use. Splitting the `gU`/`gS`/`gVh`
/// contributions reproduces exactly torch's `if gU.defined() ... else`
/// branching for the "only some outputs have gradients" case.
///
/// EXACT for inputs with DISTINCT singular values (and full rank, as torch
/// assumes — `FunctionsManual.cpp:3613-3615`). On a degenerate input the `E`
/// off-diagonal `1/(S^2[j]-S^2[i])` blows up exactly as torch's does (torch
/// does not special-case degeneracy; the JVP/VJP are ill-defined at a
/// repeated singular value). Like `eigh`, the SVD is gauge-free: `(U, V)` and
/// `(U·diag(±1), V·diag(±1))` are both valid reduced SVDs, so a loss must be
/// invariant under joint column sign flips for the gradient to be well-posed
/// (`FunctionsManual.cpp:3682-3698`); ferray's faer-backed forward emits its
/// own column signs (differing from torch's LAPACK signs), but the VJP is
/// sign-consistent, so on well-posed (gauge-invariant) losses `A.grad` matches
/// torch.
#[derive(Debug)]
struct SvdBackwardShared<T: Float> {
    /// Left singular vectors `U` (`m×k`), retained.
    u: Tensor<T>,
    /// Singular values `S` (`k`), retained.
    s: Tensor<T>,
    /// Right singular vectors (hermitian) `Vh` (`k×n`), retained.
    vh: Tensor<T>,
}

impl<T: Float> SvdBackwardShared<T> {
    fn m(&self) -> usize {
        self.u.shape()[0]
    }
    fn n(&self) -> usize {
        self.vh.shape()[1]
    }
    fn k(&self) -> usize {
        self.s.shape()[0]
    }

    fn reject_unsupported_cuda() -> FerrotorchResult<()> {
        if !(is_f32::<T>() || is_f64::<T>()) {
            return Err(FerrotorchError::NotImplementedOnCuda { op: "SvdBackward" });
        }
        Ok(())
    }

    fn e_matrix_cuda(&self) -> FerrotorchResult<Tensor<T>> {
        Self::reject_unsupported_cuda()?;
        let k = self.k();
        let s2 = self.s.mul_t(&self.s)?;
        let row = s2.view_reshape(vec![1, k])?;
        let col = s2.view_reshape(vec![k, 1])?;
        let gaps = row.sub_t(&col)?;
        let eye = crate::creation::eye::<T>(k)?.to(self.s.device())?;
        gaps.add_t(&eye)
    }

    fn conjugate_cuda(&self, ret: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        Self::reject_unsupported_cuda()?;
        let uret = self.u.mm(ret)?;
        uret.mm(&self.vh)
    }

    /// `E[i,j] = S^2[j] - S^2[i]` off-diagonal, `1` on the diagonal — the
    /// singular-value-gap denominator (`FunctionsManual.cpp:3770-3777`).
    fn e_matrix(&self, s: &[T]) -> Vec<T> {
        let k = self.k();
        let one = <T as num_traits::One>::one();
        let mut e = vec![one; k * k];
        for i in 0..k {
            for j in 0..k {
                if i != j {
                    e[i * k + j] = s[j] * s[j] - s[i] * s[i];
                }
            }
        }
        e
    }

    /// `gA = U @ ret @ Vh` (the square / no-projector assembly,
    /// `FunctionsManual.cpp:3811-3814`). `ret` is the `k×k` middle factor.
    fn conjugate(&self, ret: &[T]) -> FerrotorchResult<Tensor<T>> {
        let (m, n, k) = (self.m(), self.n(), self.k());
        let u = self.u.data()?; // [m,k]
        let vh = self.vh.data()?; // [k,n]
        let uret = mm_rows(u, ret, m, k, k); // [m,k]
        let ga = mm_rows(&uret, vh, m, k, n); // [m,n]
        from_cpu(ga, vec![m, n])
    }

    /// `gU`-only contribution. Core `ret_U[i,j] = UhgU[i,j]/E[i,j] * S[j]`
    /// plus, when `m > n`, the rectangular projector
    /// `(I_m - U U^T) gU S^{-1} V^T`.
    fn grad_a_from_gu(&self, gu: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if gu.is_cuda() {
            return crate::autograd::no_grad::no_grad(|| {
                Self::reject_unsupported_cuda()?;
                let (m, n, k) = (self.m(), self.n(), self.k());
                let u_t = self.u.transpose(0, 1)?.contiguous()?;
                let utgu = u_t.mm(gu)?;
                let utgu_t = utgu.transpose(0, 1)?.contiguous()?;
                let uhgu = utgu.sub_t(&utgu_t)?;
                let e = self.e_matrix_cuda()?;
                let s_row = self.s.view_reshape(vec![1, k])?;
                let ret = uhgu.div_t(&e)?.mul_t(&s_row)?;

                if m > n {
                    let uret = self.u.mm(&ret)?;
                    let gusinv = gu.div_t(&s_row)?;
                    let utgusinv = u_t.mm(&gusinv)?;
                    let proj = self.u.mm(&utgusinv)?;
                    let inner = uret.add_t(&gusinv)?.sub_t(&proj)?;
                    inner.mm(&self.vh)
                } else {
                    self.conjugate_cuda(&ret)
                }
            });
        }

        let (m, n, k) = (self.m(), self.n(), self.k());
        let zero = <T as num_traits::Zero>::zero();
        let u = self.u.data()?; // [m,k]
        let vh = self.vh.data()?; // [k,n]
        let s = self.s.data()?;
        let gud = gu.data()?; // [m,k]
        let e = self.e_matrix(s);

        // UhgU = skew(U^T @ gU) = U^T gU - (U^T gU)^T,  [k,k].
        let utgu = mm_at_rows(u, gud, k, m, k); // U^T @ gU, [k,k]
        let mut uhgu = vec![zero; k * k];
        for i in 0..k {
            for j in 0..k {
                uhgu[i * k + j] = utgu[i * k + j] - utgu[j * k + i];
            }
        }
        // ret[i,j] = UhgU[i,j]/E[i,j] * S[j].
        let mut ret = vec![zero; k * k];
        for i in 0..k {
            for j in 0..k {
                ret[i * k + j] = uhgu[i * k + j] / e[i * k + j] * s[j];
            }
        }

        if m > n {
            // gA = [U@ret + gU S^{-1} - U(U^T gU S^{-1})] @ Vh
            //      (FunctionsManual.cpp:3799-3804).
            let uret = mm_rows(u, &ret, m, k, k); // [m,k]
            // gUSinv[i,j] = gU[i,j] / S[j],  [m,k].
            let mut gusinv = vec![zero; m * k];
            for i in 0..m {
                for j in 0..k {
                    gusinv[i * k + j] = gud[i * k + j] / s[j];
                }
            }
            // U (U^T gUSinv): [m,k].
            let utgusinv = mm_at_rows(u, &gusinv, k, m, k); // [k,k]
            let proj = mm_rows(u, &utgusinv, m, k, k); // [m,k]
            let mut inner = vec![zero; m * k];
            for idx in 0..m * k {
                inner[idx] = uret[idx] + gusinv[idx] - proj[idx];
            }
            let ga = mm_rows(&inner, vh, m, k, n); // [m,n]
            from_cpu(ga, vec![m, n])
        } else {
            // m <= n: no gU projector; gA = U @ ret @ Vh.
            self.conjugate(&ret)
        }
    }

    /// `gS`-only contribution: `ret = diag(gS)`, then `gA = U @ ret @ Vh`
    /// (`FunctionsManual.cpp:3738-3741` svdvals optimisation / the diagonal
    /// fill at `3790-3791`).
    fn grad_a_from_gs(&self, gs: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if gs.is_cuda() {
            return crate::autograd::no_grad::no_grad(|| {
                let ret = crate::ops::tensor_ops::diag(gs, 0)?;
                self.conjugate_cuda(&ret)
            });
        }

        let k = self.k();
        let zero = <T as num_traits::Zero>::zero();
        let gsd = gs.data()?;
        let mut ret = vec![zero; k * k];
        for i in 0..k {
            ret[i * k + i] = gsd[i];
        }
        self.conjugate(&ret)
    }

    /// `gVh`-only contribution. Core `ret_V[i,j] = S[i] * VhgV[i,j]/E[i,j]`
    /// plus, when `m < n`, the rectangular projector
    /// `U S^{-1} (gV)^T (I_n - V V^T)`.
    fn grad_a_from_gvh(&self, gvh: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if gvh.is_cuda() {
            return crate::autograd::no_grad::no_grad(|| {
                Self::reject_unsupported_cuda()?;
                let (m, n, k) = (self.m(), self.n(), self.k());
                let vgvt = self.vh.mm_bt(gvh)?;
                let vgvt_t = vgvt.transpose(0, 1)?.contiguous()?;
                let vhgv = vgvt.sub_t(&vgvt_t)?;
                let e = self.e_matrix_cuda()?;
                let s_col = self.s.view_reshape(vec![k, 1])?;
                let ret = s_col.mul_t(&vhgv)?.div_t(&e)?;

                if m < n {
                    let retvh = ret.mm(&self.vh)?;
                    let sinvgvh = gvh.div_t(&s_col)?;
                    let sgvht = sinvgvh.mm_bt(&self.vh)?;
                    let proj = sgvht.mm(&self.vh)?;
                    let inner = retvh.add_t(&sinvgvh)?.sub_t(&proj)?;
                    self.u.mm(&inner)
                } else {
                    self.conjugate_cuda(&ret)
                }
            });
        }

        let (m, n, k) = (self.m(), self.n(), self.k());
        let zero = <T as num_traits::Zero>::zero();
        let u = self.u.data()?; // [m,k]
        let vh = self.vh.data()?; // [k,n]
        let s = self.s.data()?;
        let gvhd = gvh.data()?; // [k,n]
        let e = self.e_matrix(s);

        // VhgV = skew(Vh @ gVh^T) = Vh gVh^T - (Vh gVh^T)^T,  [k,k].
        let vgvt = mm_bt_rows(vh, gvhd, k, n, k); // Vh @ gVh^T, [k,k]
        let mut vhgv = vec![zero; k * k];
        for i in 0..k {
            for j in 0..k {
                vhgv[i * k + j] = vgvt[i * k + j] - vgvt[j * k + i];
            }
        }
        // ret[i,j] = S[i] * VhgV[i,j]/E[i,j].
        let mut ret = vec![zero; k * k];
        for i in 0..k {
            for j in 0..k {
                ret[i * k + j] = s[i] * vhgv[i * k + j] / e[i * k + j];
            }
        }

        if m < n {
            // gA = U @ [ret@Vh + S^{-1}gVh - (S^{-1}gVh Vh^T)Vh]
            //      (FunctionsManual.cpp:3805-3810).
            let retvh = mm_rows(&ret, vh, k, k, n); // [k,n]
            // SinvgVh[i,j] = gVh[i,j] / S[i],  [k,n].
            let mut sinvgvh = vec![zero; k * n];
            for i in 0..k {
                for j in 0..n {
                    sinvgvh[i * n + j] = gvhd[i * n + j] / s[i];
                }
            }
            // (SinvgVh @ Vh^T) @ Vh: [k,n].
            let sgvht = mm_bt_rows(&sinvgvh, vh, k, n, k); // [k,k]
            let proj = mm_rows(&sgvht, vh, k, k, n); // [k,n]
            let mut inner = vec![zero; k * n];
            for idx in 0..k * n {
                inner[idx] = retvh[idx] + sinvgvh[idx] - proj[idx];
            }
            let ga = mm_rows(u, &inner, m, k, n); // [m,n]
            from_cpu(ga, vec![m, n])
        } else {
            // m >= n: no gVh projector; gA = U @ ret @ Vh.
            self.conjugate(&ret)
        }
    }
}

/// `gU`-only svd backward node, attached to the `U` output.
#[derive(Debug)]
struct SvdBackwardU<T: Float> {
    input: Tensor<T>,
    shared: SvdBackwardShared<T>,
}

impl<T: Float> GradFn<T> for SvdBackwardU<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        Ok(vec![Some(self.shared.grad_a_from_gu(grad_output)?)])
    }
    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }
    fn name(&self) -> &'static str {
        "SvdBackward"
    }
}

/// `gS`-only svd backward node, attached to the `S` output.
#[derive(Debug)]
struct SvdBackwardS<T: Float> {
    input: Tensor<T>,
    shared: SvdBackwardShared<T>,
}

impl<T: Float> GradFn<T> for SvdBackwardS<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        Ok(vec![Some(self.shared.grad_a_from_gs(grad_output)?)])
    }
    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }
    fn name(&self) -> &'static str {
        "SvdBackward"
    }
}

/// `gVh`-only svd backward node, attached to the `Vh` output.
#[derive(Debug)]
struct SvdBackwardV<T: Float> {
    input: Tensor<T>,
    shared: SvdBackwardShared<T>,
}

impl<T: Float> GradFn<T> for SvdBackwardV<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        Ok(vec![Some(self.shared.grad_a_from_gvh(grad_output)?)])
    }
    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }
    fn name(&self) -> &'static str {
        "SvdBackward"
    }
}

/// Differentiable reduced `svd` (real, distinct singular values). Attaches the
/// split `SvdBackwardU` / `SvdBackwardS` / `SvdBackwardV` nodes (whose `A.grad`
/// contributions the autograd engine accumulates) when grad is needed.
///
/// Forward computed under `no_grad` (re-entry guard): `linalg_fwd::svd`
/// delegates back here when grad is enabled. Mirrors
/// `torch.linalg.svd(A, full_matrices=False)` / `torch.svd`. The rectangular
/// `m != n` projector terms are handled inside the U/V partials
/// (`grad_a_from_gu` for `m>n`, `grad_a_from_gvh` for `m<n`).
pub fn svd_differentiable<T: Float>(
    a: &Tensor<T>,
) -> FerrotorchResult<(Tensor<T>, Tensor<T>, Tensor<T>)> {
    let (u, s, vh) = crate::autograd::no_grad::no_grad(|| linalg_fwd::svd(a))?;
    let needs_grad = is_grad_enabled() && a.requires_grad();
    if !needs_grad {
        return Ok((u, s, vh));
    }
    let shared = || SvdBackwardShared {
        u: u.clone(),
        s: s.clone(),
        vh: vh.clone(),
    };
    let u_node = Arc::new(SvdBackwardU {
        input: a.clone(),
        shared: shared(),
    });
    let s_node = Arc::new(SvdBackwardS {
        input: a.clone(),
        shared: shared(),
    });
    let v_node = Arc::new(SvdBackwardV {
        input: a.clone(),
        shared: shared(),
    });
    let (u_storage, u_shape) = u.into_storage_and_shape()?;
    let (s_storage, s_shape) = s.into_storage_and_shape()?;
    let (vh_storage, vh_shape) = vh.into_storage_and_shape()?;
    let u = Tensor::from_operation(u_storage, u_shape, u_node)?;
    let s = Tensor::from_operation(s_storage, s_shape, s_node)?;
    let vh = Tensor::from_operation(vh_storage, vh_shape, v_node)?;
    Ok((u, s, vh))
}

// ---------------------------------------------------------------------------
// PinvBackward — P = pinv(A)  (Moore-Penrose pseudoinverse, 2D)
// ---------------------------------------------------------------------------

/// Backward for `P = pinv(A)` (Moore-Penrose pseudoinverse).
///
/// VJP (`torch/csrc/autograd/FunctionsManual.cpp:2175` `pinv_backward`), the
/// algebraic full-rank form expressed entirely through `pinvA`, `grad`, and
/// `A` (NO eigendecomposition, so NO degenerate-singular-value issue — the
/// formula is exact whenever `A` is full-rank). For `m <= n`:
///   `gA = -(pinvA K)^H + K pinvAh - (A pinvA)(K pinvAh)
///         + (pinvAh pinvA)(gradh - K A)`,  `K = gradh @ pinvA`
/// For `m > n` the symmetric dual form. `^H` is real transpose here.
#[derive(Debug)]
pub struct PinvBackward<T: Float> {
    /// Input `A` (`m×n`), retained.
    a: Tensor<T>,
    /// Pseudoinverse `P = pinv(A)` (`n×m`), retained.
    pinv: Tensor<T>,
}

impl<T: Float> PinvBackward<T> {
    fn compute(&self, grad: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if grad.is_cuda() {
            return self.compute_cuda(grad);
        }

        let m = self.a.shape()[0];
        let n = self.a.shape()[1];
        let a = self.a.data()?.to_vec(); // [m,n]
        let pa = self.pinv.data()?.to_vec(); // [n,m]
        let g = grad.data()?; // [n,m] (same shape as pinv)
        // pinvAh = pinvA^T  [m,n];  gradh = grad^T  [m,n].
        let pah = transpose_rows(&pa, n, m); // [m,n]
        let gh = transpose_rows(g, n, m); // [m,n]

        let neg = |v: &[T]| -> Vec<T> { v.iter().map(|&x| -x).collect() };
        let add =
            |x: &[T], y: &[T]| -> Vec<T> { x.iter().zip(y.iter()).map(|(&a, &b)| a + b).collect() };
        let sub =
            |x: &[T], y: &[T]| -> Vec<T> { x.iter().zip(y.iter()).map(|(&a, &b)| a - b).collect() };

        let out = if m <= n {
            // K = gradh @ pinvA:  [m,n]@[n,m] -> [m,m].
            let k = mm_rows(&gh, &pa, m, n, m);
            // KpinvAh = K @ pinvAh:  [m,m]@[m,n] -> [m,n].
            let kpah = mm_rows(&k, &pah, m, m, n);
            // -(pinvA @ K)^H:  pinvA@K = [n,m]@[m,m] -> [n,m]; ^H -> [m,n].
            let pak = mm_rows(&pa, &k, n, m, m); // [n,m]
            let neg_pak_h = neg(&transpose_rows(&pak, n, m)); // [m,n]
            // (A @ pinvA) @ KpinvAh:  A@pinvA=[m,n]@[n,m]->[m,m]; @[m,n]->[m,n].
            let apa = mm_rows(&a, &pa, m, n, m); // [m,m]
            let apa_kpah = mm_rows(&apa, &kpah, m, m, n); // [m,n]
            // (pinvAh @ pinvA) @ (gradh - K @ A):
            //   pinvAh@pinvA = [m,n]@[n,m]->[m,m]; KA=[m,m]@[m,n]->[m,n];
            //   gradh-KA=[m,n]; result [m,m]@[m,n]->[m,n].
            let pahpa = mm_rows(&pah, &pa, m, n, m); // [m,m]
            let ka = mm_rows(&k, &a, m, m, n); // [m,n]
            let gh_minus_ka = sub(&gh, &ka); // [m,n]
            let last = mm_rows(&pahpa, &gh_minus_ka, m, m, n); // [m,n]
            add(&add(&neg_pak_h, &kpah), &sub(&last, &apa_kpah)) // [m,n]
        } else {
            // m > n branch.
            // K = pinvA @ gradh:  [n,m]@[m,n] -> [n,n].
            let k = mm_rows(&pa, &gh, n, m, n);
            // pinvAhK = pinvAh @ K:  [m,n]@[n,n] -> [m,n].
            let pahk = mm_rows(&pah, &k, m, n, n);
            // -(K @ pinvA)^H:  K@pinvA = [n,n]@[n,m]->[n,m]; ^H -> [m,n].
            let kpa = mm_rows(&k, &pa, n, n, m); // [n,m]
            let neg_kpa_h = neg(&transpose_rows(&kpa, n, m)); // [m,n]
            // (gradh - A @ K) @ pinvA @ pinvAh:
            //   AK = [m,n]@[n,n]->[m,n]; gradh-AK=[m,n];
            //   (gradh-AK)@pinvA=[m,n]@[n,m]->[m,m]; @pinvAh=[m,m]@[m,n]->[m,n].
            let ak = mm_rows(&a, &k, m, n, n); // [m,n]
            let gh_minus_ak = sub(&gh, &ak); // [m,n]
            let t1 = mm_rows(&gh_minus_ak, &pa, m, n, m); // [m,m]
            let t2 = mm_rows(&t1, &pah, m, m, n); // [m,n]
            // - pinvAhK @ pinvA @ A:
            //   pinvAhK@pinvA = [m,n]@[n,m]->[m,m]; @A=[m,m]@[m,n]->[m,n].
            let p1 = mm_rows(&pahk, &pa, m, n, m); // [m,m]
            let p2 = mm_rows(&p1, &a, m, m, n); // [m,n]
            add(&add(&neg_kpa_h, &t2), &sub(&pahk, &p2)) // [m,n]
        };
        from_cpu(out, vec![m, n])
    }

    fn compute_cuda(&self, grad: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if !(is_f32::<T>() || is_f64::<T>()) {
            return Err(FerrotorchError::NotImplementedOnCuda { op: "PinvBackward" });
        }

        crate::autograd::no_grad::no_grad(|| {
            let m = self.a.shape()[0];
            let n = self.a.shape()[1];
            let pa = &self.pinv; // [n,m]
            let pah = pa.transpose(0, 1)?.contiguous()?; // [m,n]
            let gh = grad.transpose(0, 1)?.contiguous()?; // [m,n]

            if m <= n {
                // K = gradh @ pinvA: [m,n]@[n,m] -> [m,m].
                let k = gh.mm(pa)?;
                let kpah = k.mm(&pah)?;
                let pak = pa.mm(&k)?;
                let neg_pak_h = pak.transpose(0, 1)?.contiguous()?.neg_t()?;
                let apa = self.a.mm(pa)?;
                let apa_kpah = apa.mm(&kpah)?;
                let pahpa = pah.mm(pa)?;
                let ka = k.mm(&self.a)?;
                let gh_minus_ka = gh.sub_t(&ka)?;
                let last = pahpa.mm(&gh_minus_ka)?;
                neg_pak_h.add_t(&kpah)?.add_t(&last.sub_t(&apa_kpah)?)
            } else {
                // m > n branch.
                let k = pa.mm(&gh)?;
                let pahk = pah.mm(&k)?;
                let kpa = k.mm(pa)?;
                let neg_kpa_h = kpa.transpose(0, 1)?.contiguous()?.neg_t()?;
                let ak = self.a.mm(&k)?;
                let gh_minus_ak = gh.sub_t(&ak)?;
                let t1 = gh_minus_ak.mm(pa)?;
                let t2 = t1.mm(&pah)?;
                let p1 = pahk.mm(pa)?;
                let p2 = p1.mm(&self.a)?;
                neg_kpa_h.add_t(&t2)?.add_t(&pahk.sub_t(&p2)?)
            }
        })
    }
}

impl<T: Float> GradFn<T> for PinvBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        Ok(vec![Some(self.compute(grad_output)?)])
    }
    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.a]
    }
    fn name(&self) -> &'static str {
        "PinvBackward"
    }
}

/// Differentiable `pinv`. Attaches `PinvBackward` when grad is needed.
///
/// Forward computed under `no_grad` (re-entry guard).
pub fn pinv_differentiable<T: Float>(a: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let p = crate::autograd::no_grad::no_grad(|| linalg_fwd::pinv(a))?;
    if is_grad_enabled() && a.requires_grad() {
        let grad_fn = Arc::new(PinvBackward {
            a: a.clone(),
            pinv: p.clone(),
        });
        let (storage, shape) = p.into_storage_and_shape()?;
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Ok(p)
    }
}

// ---------------------------------------------------------------------------
// LstsqBackward — X = lstsq(A, B).solution  (full-rank least squares)
// ---------------------------------------------------------------------------

/// Which differentiable `torch.linalg.lstsq` output this node is attached to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LstsqBackwardOutput {
    Solution,
    Residuals,
}

/// Backward for the differentiable outputs of `lstsq(A, B)`.
///
/// VJP (`torch/csrc/autograd/FunctionsManual.cpp:4038-4050`
/// `linalg_lstsq_backward`). The `solution` output contributes:
///   `gB = pinv(A)^H @ gX`
///   `gA = pinv_backward(gX @ B^H, pinv(A), A)`
/// The `residuals` output contributes, when non-empty:
///   `R = A @ X - B`
///   `gA = 2 * (gL.unsqueeze(-2) * R) @ X^H`
///   `gB = -2 * gL.unsqueeze(-2) * R`
///
/// `rank` and `singular_values` are non-differentiable. Full-rank only
/// (`pinv_backward` is exact for full-rank `A`). `^H` is real transpose here.
/// Promotes a 1-D RHS to a column matrix for the matmuls.
#[derive(Debug)]
pub struct LstsqBackward<T: Float> {
    a: Tensor<T>,
    b: Tensor<T>,
    /// Retained pseudoinverse `pinv(A)` (`n×m`).
    pinv: Tensor<T>,
    /// Retained solution `X` (`n×nrhs` or `[n]` for vector RHS).
    solution: Tensor<T>,
    /// Whether `B` was a 1-D vector RHS (then `X` is 1-D too).
    vector_rhs: bool,
    output: LstsqBackwardOutput,
}

impl<T: Float> GradFn<T> for LstsqBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        crate::autograd::no_grad::no_grad(|| match self.output {
            LstsqBackwardOutput::Solution => self.backward_solution(grad_output),
            LstsqBackwardOutput::Residuals => self.backward_residuals(grad_output),
        })
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.a, &self.b]
    }

    fn name(&self) -> &'static str {
        "LstsqBackward"
    }
}

impl<T: Float> LstsqBackward<T> {
    fn matrix_rhs(&self) -> FerrotorchResult<Tensor<T>> {
        let m = self.a.shape()[0];
        if self.vector_rhs {
            self.b.view_reshape(vec![m, 1])
        } else {
            Ok(self.b.clone())
        }
    }

    fn matrix_solution(&self) -> FerrotorchResult<Tensor<T>> {
        let n = self.a.shape()[1];
        if self.vector_rhs {
            self.solution.view_reshape(vec![n, 1])
        } else {
            Ok(self.solution.clone())
        }
    }

    fn matrix_grad_solution(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let n = self.a.shape()[1];
        if self.vector_rhs {
            grad_output.view_reshape(vec![n, 1])
        } else {
            Ok(grad_output.clone())
        }
    }

    fn backward_solution(
        &self,
        grad_output: &Tensor<T>,
    ) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let m = self.a.shape()[0];
        let gx = if self.vector_rhs {
            self.matrix_grad_solution(grad_output)?
        } else {
            grad_output.clone()
        };
        let bmat = self.matrix_rhs()?;

        // gB = pinv(A)^H @ gX:  pinvAh = pinv^T [m,n]; gB = [m,n]@[n,nrhs].
        let grad_b = if self.b.requires_grad() {
            let pah = self.pinv.transpose(0, 1)?.contiguous()?; // [m,n]
            let gb = pah.mm(&gx)?; // [m,nrhs]
            let gb = if self.vector_rhs {
                gb.view_reshape(vec![m])?
            } else {
                gb
            };
            Some(gb)
        } else {
            None
        };

        // gA = pinv_backward(gX @ B^H, pinv, A).  gX@B^H = [n,nrhs]@[nrhs,m]->[n,m].
        let grad_a = if self.a.requires_grad() {
            let pinv_a_grad = gx.mm_bt(&bmat)?; // [n,m]
            let pb = PinvBackward {
                a: self.a.clone(),
                pinv: self.pinv.clone(),
            };
            Some(pb.compute(&pinv_a_grad)?)
        } else {
            None
        };

        Ok(vec![grad_a, grad_b])
    }

    fn backward_residuals(
        &self,
        grad_output: &Tensor<T>,
    ) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if grad_output.numel() == 0 {
            let grad_a = if self.a.requires_grad() {
                Some(crate::creation::zeros_like(&self.a)?)
            } else {
                None
            };
            let grad_b = if self.b.requires_grad() {
                Some(crate::creation::zeros_like(&self.b)?)
            } else {
                None
            };
            return Ok(vec![grad_a, grad_b]);
        }

        let nrhs = if self.vector_rhs {
            1
        } else {
            self.b.shape()[1]
        };
        let x = self.matrix_solution()?;
        let b = self.matrix_rhs()?;
        let residual = self.a.mm(&x)?.sub_t(&b)?;
        let gl = grad_output.view_reshape(vec![1, nrhs])?;
        let two = T::from(2.0).ok_or_else(|| FerrotorchError::InvalidArgument {
            message: "lstsq backward: 2.0 is not representable in dtype".into(),
        })?;
        let scale = crate::creation::full_like(&residual, two)?;
        let weighted = residual.mul_t(&gl)?.mul_t(&scale)?;

        let grad_a = if self.a.requires_grad() {
            Some(weighted.mm_bt(&x)?)
        } else {
            None
        };
        let grad_b = if self.b.requires_grad() {
            let gb = weighted.neg_t()?;
            let gb = if self.vector_rhs {
                gb.view_reshape(vec![self.b.shape()[0]])?
            } else {
                gb
            };
            Some(gb)
        } else {
            None
        };

        Ok(vec![grad_a, grad_b])
    }
}

/// Differentiable `lstsq`. Returns the 4-tuple `(solution, residuals, rank,
/// singular_values)`; `solution` and `residuals` are differentiable
/// (`output_differentiability: [True, True, False, False]` per
/// `derivatives.yaml:1056`). Attaches split `LstsqBackward` nodes when grad is
/// needed.
///
/// Forward computed under `no_grad` (re-entry guard).
#[allow(
    clippy::type_complexity,
    reason = "mirrors torch.linalg.lstsq's 4-tuple return"
)]
pub fn lstsq_differentiable<T: Float>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    rcond: Option<f64>,
    driver: Option<linalg_fwd::LstsqDriver>,
) -> FerrotorchResult<linalg_fwd::LstsqResult<T>> {
    let (sol, resid, rank, sv) =
        crate::autograd::no_grad::no_grad(|| linalg_fwd::lstsq_with_driver(a, b, rcond, driver))?;
    if is_grad_enabled() && (a.requires_grad() || b.requires_grad()) {
        let pinv = crate::autograd::no_grad::no_grad(|| linalg_fwd::pinv(a))?;
        let sol_grad_fn = Arc::new(LstsqBackward {
            a: a.clone(),
            b: b.clone(),
            pinv: pinv.clone(),
            solution: sol.clone(),
            vector_rhs: b.ndim() == 1,
            output: LstsqBackwardOutput::Solution,
        });
        let resid_grad_fn = Arc::new(LstsqBackward {
            a: a.clone(),
            b: b.clone(),
            pinv,
            solution: sol.clone(),
            vector_rhs: b.ndim() == 1,
            output: LstsqBackwardOutput::Residuals,
        });
        let (storage, shape) = sol.into_storage_and_shape()?;
        let sol = Tensor::from_operation(storage, shape, sol_grad_fn)?;
        let (storage, shape) = resid.into_storage_and_shape()?;
        let resid = Tensor::from_operation(storage, shape, resid_grad_fn)?;
        Ok((sol, resid, rank, sv))
    } else {
        Ok((sol, resid, rank, sv))
    }
}

/// Differentiable solution-only `lstsq_solve`, backed by the same PyTorch VJP
/// branch as `torch.linalg.lstsq(...).solution`.
pub fn lstsq_solve_differentiable<T: Float>(
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> FerrotorchResult<Tensor<T>> {
    let sol = crate::autograd::no_grad::no_grad(|| linalg_fwd::lstsq_solve(a, b))?;
    if is_grad_enabled() && (a.requires_grad() || b.requires_grad()) {
        let pinv = crate::autograd::no_grad::no_grad(|| linalg_fwd::pinv(a))?;
        let grad_fn = Arc::new(LstsqBackward {
            a: a.clone(),
            b: b.clone(),
            pinv,
            solution: sol.clone(),
            vector_rhs: b.ndim() == 1,
            output: LstsqBackwardOutput::Solution,
        });
        let (storage, shape) = sol.into_storage_and_shape()?;
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Ok(sol)
    }
}

// ---------------------------------------------------------------------------
// LuBackward — (P, L, U) = lu(A)  /  (LU, pivots) = lu_factor(A)
// ---------------------------------------------------------------------------

/// Shared LU VJP, grounded in
/// `torch/csrc/autograd/FunctionsManual.cpp:6854` `linalg_lu_backward`:
/// it handles the square, wide, and tall branches and composes tensor
/// operations so CUDA inputs stay resident.
///
/// The two outputs `L` and `U` are jointly linear in `gA`; the engine
/// accumulates `LuBackwardL` (`gU=0`) and `LuBackwardU` (`gL=0`) into `A.grad`.
/// `lu_factor` packs `L`/`U` into one matrix, so its single output carries
/// both partials through `LuFactorBackward`.
#[derive(Debug)]
struct LuBackwardShared<T: Float> {
    p: Tensor<T>,
    l: Tensor<T>,
    u: Tensor<T>,
}

impl<T: Float> LuBackwardShared<T> {
    fn transpose_2d(t: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        t.transpose(0, 1)?.contiguous()
    }

    fn zeros_like_shape(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        let numel = crate::shape::checked_numel(shape, "lu_backward")?;
        Tensor::from_storage(
            TensorStorage::on_device(
                vec![<T as num_traits::Zero>::zero(); numel],
                self.l.device(),
            )?,
            shape.to_vec(),
            false,
        )
    }

    fn add_optional(lhs: Option<Tensor<T>>, rhs: Tensor<T>) -> FerrotorchResult<Option<Tensor<T>>> {
        Ok(Some(if let Some(lhs) = lhs {
            lhs.add_t(&rhs)?
        } else {
            rhs
        }))
    }

    fn sub_optional(
        lhs: Option<Tensor<T>>,
        rhs: &Tensor<T>,
    ) -> FerrotorchResult<Option<Tensor<T>>> {
        let neg_rhs = rhs.neg_t()?;
        Self::add_optional(lhs, neg_rhs)
    }

    fn right_solve_triangular(
        a: &Tensor<T>,
        b: &Tensor<T>,
        upper: bool,
        unit_diagonal: bool,
    ) -> FerrotorchResult<Tensor<T>> {
        // PyTorch's formula uses `left=false` solves. This crate exposes only
        // left solves, so solve X @ A = B by transposing to
        // A^T @ X^T = B^T and transposing the result back.
        let b_t = Self::transpose_2d(b)?;
        let y = linalg_fwd::solve_triangular(a, &b_t, upper, true, unit_diagonal)?;
        Self::transpose_2d(&y)
    }

    fn cat(tensors: &[Tensor<T>], axis: isize) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::shape::cat(tensors, axis)
    }

    fn finish_with_pivot(&self, a_grad: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Saved P follows torch's A = P @ L @ U convention (CORE-144 / #1838),
        // so the final adjoint is exactly PyTorch's `P.matmul(A_grad)`.
        self.p.mm(a_grad)
    }

    fn grad_a(
        &self,
        gl: Option<&Tensor<T>>,
        gu: Option<&Tensor<T>>,
    ) -> FerrotorchResult<Tensor<T>> {
        crate::autograd::no_grad::no_grad(|| self.grad_a_impl(gl, gu))
    }

    fn grad_a_impl(
        &self,
        gl: Option<&Tensor<T>>,
        gu: Option<&Tensor<T>>,
    ) -> FerrotorchResult<Tensor<T>> {
        let m = self.l.shape()[0];
        let k = self.l.shape()[1];
        let n = self.u.shape()[1];
        if k == 0 {
            return self.zeros_like_shape(&[m, n]);
        }

        if m == n {
            return self.grad_a_square(gl, gu, m);
        }
        if m < n {
            self.grad_a_wide(gl, gu, m, n, k)
        } else {
            self.grad_a_tall(gl, gu, m, n, k)
        }
    }

    fn grad_a_square(
        &self,
        gl: Option<&Tensor<T>>,
        gu: Option<&Tensor<T>>,
        n: usize,
    ) -> FerrotorchResult<Tensor<T>> {
        let mut core: Option<Tensor<T>> = None;
        if let Some(gl) = gl {
            let lt = Self::transpose_2d(&self.l)?;
            core = Self::add_optional(core, crate::ops::tensor_ops::tril(&lt.mm(gl)?, -1)?)?;
        }
        if let Some(gu) = gu {
            let ut = Self::transpose_2d(&self.u)?;
            core = Self::add_optional(core, crate::ops::tensor_ops::triu(&gu.mm(&ut)?, 0)?)?;
        }
        let core = core.unwrap_or(self.zeros_like_shape(&[n, n])?);
        let ut = Self::transpose_2d(&self.u)?;
        let step1 = Self::right_solve_triangular(&ut, &core, false, false)?;
        let lt = Self::transpose_2d(&self.l)?;
        let step2 = linalg_fwd::solve_triangular(&lt, &step1, true, false, true)?;
        self.finish_with_pivot(&step2)
    }

    fn grad_a_wide(
        &self,
        gl: Option<&Tensor<T>>,
        gu: Option<&Tensor<T>>,
        m: usize,
        n: usize,
        k: usize,
    ) -> FerrotorchResult<Tensor<T>> {
        let mut core: Option<Tensor<T>> = None;
        if let Some(gl) = gl {
            let lt = Self::transpose_2d(&self.l)?;
            core = Self::add_optional(core, lt.mm(gl)?)?;
        }
        if let Some(gu) = gu {
            let gu_upper = crate::ops::tensor_ops::triu(gu, 0)?;
            let ut = Self::transpose_2d(&self.u)?;
            core = Self::sub_optional(core, &gu_upper.mm(&ut)?)?;
        }
        let core = core.unwrap_or(self.zeros_like_shape(&[k, k])?);
        let core_lower = crate::ops::tensor_ops::tril(&core, -1)?;
        let u1_t = Self::transpose_2d(&self.u.narrow(1, 0, k)?)?;
        let solved = Self::right_solve_triangular(&u1_t, &core_lower, false, false)?;

        let block = if let Some(gu) = gu {
            let gu1 = crate::ops::tensor_ops::triu(&gu.narrow(1, 0, k)?, 0)?;
            let left = solved.add_t(&gu1)?;
            let right = gu.narrow(1, k, n - k)?;
            Self::cat(&[left, right], 1)?
        } else {
            let right = self.zeros_like_shape(&[m, n - k])?;
            Self::cat(&[solved, right], 1)?
        };

        let lt = Self::transpose_2d(&self.l)?;
        let solved_l = linalg_fwd::solve_triangular(&lt, &block, true, false, true)?;
        self.finish_with_pivot(&solved_l)
    }

    fn grad_a_tall(
        &self,
        gl: Option<&Tensor<T>>,
        gu: Option<&Tensor<T>>,
        m: usize,
        n: usize,
        k: usize,
    ) -> FerrotorchResult<Tensor<T>> {
        let mut core: Option<Tensor<T>> = None;
        if let Some(gu) = gu {
            let ut = Self::transpose_2d(&self.u)?;
            core = Self::add_optional(core, gu.mm(&ut)?)?;
        }
        if let Some(gl) = gl {
            let gl_lower = crate::ops::tensor_ops::tril(gl, -1)?;
            let lt = Self::transpose_2d(&self.l)?;
            core = Self::sub_optional(core, &lt.mm(&gl_lower)?)?;
        }
        let core = core.unwrap_or(self.zeros_like_shape(&[k, k])?);
        let core_upper = crate::ops::tensor_ops::triu(&core, 0)?;
        let l1_t = Self::transpose_2d(&self.l.narrow(0, 0, k)?)?;
        let solved = linalg_fwd::solve_triangular(&l1_t, &core_upper, true, false, true)?;

        let block = if let Some(gl) = gl {
            let gl1 = crate::ops::tensor_ops::tril(&gl.narrow(0, 0, k)?, -1)?;
            let top = solved.add_t(&gl1)?;
            let bottom = gl.narrow(0, k, m - k)?;
            Self::cat(&[top, bottom], 0)?
        } else {
            let bottom = self.zeros_like_shape(&[m - k, n])?;
            Self::cat(&[solved, bottom], 0)?
        };

        let ut = Self::transpose_2d(&self.u)?;
        let solved_u = Self::right_solve_triangular(&ut, &block, false, false)?;
        self.finish_with_pivot(&solved_u)
    }
}

/// `gL`-only LU backward node, attached to the `L` output.
#[derive(Debug)]
struct LuBackwardL<T: Float> {
    input: Tensor<T>,
    shared: LuBackwardShared<T>,
}

impl<T: Float> GradFn<T> for LuBackwardL<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        Ok(vec![Some(self.shared.grad_a(Some(grad_output), None)?)])
    }
    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }
    fn name(&self) -> &'static str {
        "LuBackward"
    }
}

/// `gU`-only LU backward node, attached to the `U` output.
#[derive(Debug)]
struct LuBackwardU<T: Float> {
    input: Tensor<T>,
    shared: LuBackwardShared<T>,
}

impl<T: Float> GradFn<T> for LuBackwardU<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        Ok(vec![Some(self.shared.grad_a(None, Some(grad_output))?)])
    }
    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }
    fn name(&self) -> &'static str {
        "LuBackward"
    }
}

/// Differentiable `lu`. Returns `(P, L, U)`; `P` is a
/// non-differentiable permutation matrix (returned plain). Attaches the split
/// `LuBackwardL` / `LuBackwardU` nodes when grad is needed.
///
/// Forward computed under `no_grad` (re-entry guard). The VJP follows
/// PyTorch's square, wide, and tall branches.
pub fn lu_differentiable<T: Float>(
    a: &Tensor<T>,
) -> FerrotorchResult<(Tensor<T>, Tensor<T>, Tensor<T>)> {
    let (p, l, u) = crate::autograd::no_grad::no_grad(|| linalg_fwd::lu(a))?;
    let needs_grad = is_grad_enabled() && a.requires_grad();
    if !needs_grad {
        return Ok((p, l, u));
    }
    let l_node = Arc::new(LuBackwardL {
        input: a.clone(),
        shared: LuBackwardShared {
            p: p.clone(),
            l: l.clone(),
            u: u.clone(),
        },
    });
    let u_node = Arc::new(LuBackwardU {
        input: a.clone(),
        shared: LuBackwardShared {
            p: p.clone(),
            l: l.clone(),
            u: u.clone(),
        },
    });
    let (l_storage, l_shape) = l.into_storage_and_shape()?;
    let (u_storage, u_shape) = u.into_storage_and_shape()?;
    let l = Tensor::from_operation(l_storage, l_shape, l_node)?;
    let u = Tensor::from_operation(u_storage, u_shape, u_node)?;
    Ok((p, l, u))
}

/// Backward for the single packed `LU` output of `lu_factor(A)`.
///
/// The packed `LU` matrix holds `strict-lower(L)` (unit diagonal implicit) and
/// `upper(U)`. So the incoming `grad` decomposes as `gL = tril(grad, -1)` (with
/// unit diagonal → no diagonal contribution to `L`) and `gU = triu(grad)`. Per
/// `lu_factor_ex_backward` (`torch/csrc/autograd/FunctionsManual.cpp:6960`)
/// these are then fed jointly to `linalg_lu_backward`. We reuse
/// `LuBackwardShared::grad_a_combined` (the combined `m == n` formula), passing
/// the same `gL`/`gU` split.
#[derive(Debug)]
pub struct LuFactorBackward<T: Float> {
    input: Tensor<T>,
    shared: LuBackwardShared<T>,
}

impl<T: Float> GradFn<T> for LuFactorBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let k = self.shared.l.shape()[1];
        let gl = grad_output.narrow(1, 0, k)?;
        let gu = grad_output.narrow(0, 0, k)?;
        Ok(vec![Some(self.shared.grad_a(Some(&gl), Some(&gu))?)])
    }
    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }
    fn name(&self) -> &'static str {
        "LuFactorBackward"
    }
}

/// Differentiable `lu_factor`. Returns `(LU_packed, pivots)`;
/// the pivot `Vec<i32>` is non-differentiable. Attaches `LuFactorBackward` to
/// the packed `LU` output when grad is needed.
///
/// Forward computed under `no_grad` (re-entry guard); the `P`/`L`/`U` matrices
/// the VJP needs are unpacked from the same LU factorization.
pub fn lu_factor_differentiable<T: Float>(
    a: &Tensor<T>,
) -> FerrotorchResult<(Tensor<T>, Vec<i32>)> {
    let (lu, pivots) = crate::autograd::no_grad::no_grad(|| linalg_fwd::lu_factor(a))?;
    let needs_grad = is_grad_enabled() && a.requires_grad();
    if !needs_grad {
        return Ok((lu, pivots));
    }
    let (p, l, u) =
        crate::autograd::no_grad::no_grad(|| linalg_fwd::lu_unpack_from_factor(&lu, &pivots))?;
    let node = Arc::new(LuFactorBackward {
        input: a.clone(),
        shared: LuBackwardShared { p, l, u },
    });
    let (storage, shape) = lu.into_storage_and_shape()?;
    let lu = Tensor::from_operation(storage, shape, node)?;
    Ok((lu, pivots))
}

// ---------------------------------------------------------------------------
// CrossBackward — c = cross(a, b, dim)  (vector cross product)
// ---------------------------------------------------------------------------

/// Backward for `c = linalg.cross(a, b, dim)`.
///
/// VJP (`tools/autograd/derivatives.yaml:516-518` `linalg_cross`):
/// - `da = cross(b, grad, dim)`   (real case: `at::linalg_cross(other.conj(),
///   grad, dim)` with `conj` a no-op)
/// - `db = cross(grad, a, dim)`   (`at::linalg_cross(grad, self.conj(), dim)`)
///
/// Cross is bilinear, so the VJP is itself two cross products.
#[derive(Debug)]
pub struct CrossBackward<T: Float> {
    a: Tensor<T>,
    b: Tensor<T>,
    dim: i64,
}

impl<T: Float> GradFn<T> for CrossBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let grad_a = if self.a.requires_grad() {
            Some(crate::autograd::no_grad::no_grad(|| {
                let raw = linalg_fwd::cross(&self.b, grad_output, self.dim)?;
                crate::grad_fns::arithmetic::reduce_grad_to_shape(&raw, self.a.shape())
            })?)
        } else {
            None
        };
        let grad_b = if self.b.requires_grad() {
            Some(crate::autograd::no_grad::no_grad(|| {
                let raw = linalg_fwd::cross(grad_output, &self.a, self.dim)?;
                crate::grad_fns::arithmetic::reduce_grad_to_shape(&raw, self.b.shape())
            })?)
        } else {
            None
        };
        Ok(vec![grad_a, grad_b])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.a, &self.b]
    }

    fn name(&self) -> &'static str {
        "CrossBackward"
    }
}

/// Differentiable `cross`. Attaches `CrossBackward` when grad is needed.
///
/// Forward computed under `no_grad` (re-entry guard).
pub fn cross_differentiable<T: Float>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    dim: i64,
) -> FerrotorchResult<Tensor<T>> {
    let c = crate::autograd::no_grad::no_grad(|| linalg_fwd::cross(a, b, dim))?;
    if is_grad_enabled() && (a.requires_grad() || b.requires_grad()) {
        let grad_fn = Arc::new(CrossBackward {
            a: a.clone(),
            b: b.clone(),
            dim,
        });
        let (storage, shape) = c.into_storage_and_shape()?;
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Ok(c)
    }
}

// ---------------------------------------------------------------------------
// VectorNormBackward / MatrixNormBackward — Euclidean / Frobenius norm
// ---------------------------------------------------------------------------

/// Backward for the full-tensor `vector_norm` (every `ord`) and the
/// Frobenius `matrix_norm` (the `ord=2` instance over flattened entries).
///
/// VJP per `norm_backward` in `torch/csrc/autograd/FunctionsManual.cpp`
/// (`linalg_vector_norm_backward` dispatches to it), branch by `p`,
/// live-oracle-verified on torch 2.11.0+cu130 (CORE-047 / #1741):
///
/// - `p == 0`: count of nonzeros — torch returns an UNDEFINED gradient
///   (`if (p == 0.0) return {};`); the leaf's `.grad` stays `None` and the
///   contribution accumulates as zero in a wider graph (live probe:
///   `norm0(x).backward()` → `x.grad is None`; `(norm0(x)+x.sum())
///   .backward()` → `x.grad == ones`). Returned here as `vec![None]`.
/// - `p == 1`: `dx = g * sgn(x)` (`sgn(0) = 0`).
/// - `p == 2`: `dx = g * x / norm`, `masked_fill_(norm == 0, 0)` guard.
/// - `p == ±inf`: gradient routed to the extremal-`|x|` elements with ties
///   split EVENLY — `dx = sgn(x) * [|x| == norm] * g / count(|x| == norm)`
///   (live probe: `x=[3,-3,1]`, `ord=inf` → `[0.5, -0.5, 0]`). NaN inputs
///   count as ties iff the norm itself is NaN, per upstream's
///   `isnan().logical_and_(norm.isnan())`.
/// - `p < 1` (incl. negative p): `dx = sgn(x)*|x|^(p-1) * g * norm^(1-p)`,
///   with the `x == 0 → 0` subgradient mask (live probe: `x=[0,1,4]`,
///   `ord=0.5` → `[0, 3, 1.5]`; `x=[1,-2,0]`, `ord=-1` → forward 0,
///   gradient zeros via the `norm^(1-p) = 0` scale).
/// - `1 < p < 2`: `dx = sgn(x)*|x|^(p-1) * g / norm^(p-1)`; `|0|^(p-1)=0`
///   needs no mask; `norm == 0 → 0` guard.
/// - `p > 2`: `dx = x*|x|^(p-2) * g / norm^(p-1)`; `norm == 0 → 0` guard
///   (live probe: all-zero input, `ord=3` → zeros, never NaN).
#[derive(Debug)]
pub struct NormBackward<T: Float> {
    /// Input `x` (any shape), retained.
    x: Tensor<T>,
    /// Scalar norm tensor, retained on the forward device.
    norm: Tensor<T>,
    /// Norm order saved at forward time — selects the `norm_backward`
    /// branch above.
    ord: f64,
}

/// torch's `sgn` for real floats: `0` at `0` (both signs), `NaN` at `NaN`,
/// `±1` elsewhere. Distinct from `num_traits::Float::signum`, which maps
/// `±0 → ±1`.
fn sgn<T: Float>(v: T) -> T {
    if v == <T as num_traits::Zero>::zero() {
        <T as num_traits::Zero>::zero()
    } else {
        v.signum()
    }
}

#[allow(
    clippy::float_cmp,
    reason = "ord is a discrete dispatch selector (0/1/2/±inf are exact \
              sentinels, not computed values) and the |x| == norm tie test \
              mirrors upstream norm_backward's exact-equality `self.abs() \
              == norm` mask"
)]
impl<T: Float> GradFn<T> for NormBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let p = self.ord;
        // p == 0: undefined gradient per upstream `return {};` (doc above).
        if p == 0.0 {
            return Ok(vec![None]);
        }
        if self.x.is_cuda() {
            return crate::autograd::no_grad::no_grad(|| {
                Ok(vec![Some(norm_backward_tensor(
                    &self.x,
                    &self.norm,
                    grad_output,
                    p,
                )?)])
            });
        }
        let g: T = grad_output.item()?;
        let zero = <T as num_traits::Zero>::zero();
        let xd = self.x.data()?;
        let norm = self.norm.item()?;

        let dx: Vec<T> = if p == 2.0 {
            if norm == zero {
                // masked_fill_(norm == 0, 0): zero gradient at zero norm.
                vec![zero; xd.len()]
            } else {
                xd.iter().map(|&v| g * (v / norm)).collect()
            }
        } else if p == 1.0 {
            xd.iter().map(|&v| g * sgn(v)).collect()
        } else if p.is_infinite() {
            // Tie mask: |x| == norm, or NaN matching a NaN norm.
            let is_eq: Vec<bool> = xd
                .iter()
                .map(|&v| v.abs() == norm || (v.is_nan() && norm.is_nan()))
                .collect();
            let count = is_eq.iter().filter(|&&b| b).count();
            let count_t = <T as num_traits::NumCast>::from(count).ok_or_else(|| {
                FerrotorchError::InvalidArgument {
                    message: format!("norm backward: tie count {count} not representable"),
                }
            })?;
            let scale = g / count_t;
            xd.iter()
                .zip(is_eq)
                .map(|(&v, eq)| if eq { sgn(v) * scale } else { zero })
                .collect()
        } else {
            // Finite p ∉ {0, 1, 2}.
            let pt = <T as num_traits::NumCast>::from(p).ok_or_else(|| {
                FerrotorchError::InvalidArgument {
                    message: format!("norm backward: ord {p} not representable in dtype"),
                }
            })?;
            let one = <T as num_traits::One>::one();
            if p < 1.0 {
                // scale = g * norm^(1-p); norm == 0 → 0^(positive) = 0,
                // which zeroes the whole gradient (matches the live probes).
                let scale = g * norm.powf(one - pt);
                xd.iter()
                    .map(|&v| {
                        if v == zero {
                            // x == 0 subgradient mask (|0|^(p-1) diverges).
                            zero
                        } else {
                            sgn(v) * v.abs().powf(pt - one) * scale
                        }
                    })
                    .collect()
            } else if norm == zero {
                // p > 1 and norm == 0: every element is 0; torch returns
                // exact zeros (scale-inf mask), never NaN.
                vec![zero; xd.len()]
            } else if p < 2.0 {
                let scale = g / norm.powf(pt - one);
                xd.iter()
                    .map(|&v| sgn(v) * v.abs().powf(pt - one) * scale)
                    .collect()
            } else {
                let two = one + one;
                let scale = g / norm.powf(pt - one);
                xd.iter()
                    .map(|&v| v * v.abs().powf(pt - two) * scale)
                    .collect()
            }
        };
        Ok(vec![Some(from_cpu(dx, self.x.shape().to_vec())?)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.x]
    }

    fn name(&self) -> &'static str {
        "NormBackward"
    }
}

fn norm_sign_and_nonzero<T: Float>(
    x: &Tensor<T>,
) -> FerrotorchResult<(Tensor<T>, crate::bool_tensor::BoolTensor)> {
    let abs = x.abs_t()?;
    let zeros = crate::creation::zeros_like(&abs)?;
    let ones = crate::creation::ones_like(&abs)?;
    let nonzero = crate::bool_tensor::BoolTensor::ne(&abs, &zeros)?;
    let safe_abs = crate::grad_fns::comparison::where_bt(&nonzero, &abs, &ones)?;
    Ok((x.div_t(&safe_abs)?, nonzero))
}

#[allow(
    clippy::float_cmp,
    reason = "ord is a discrete dispatch selector matching upstream norm_backward"
)]
fn norm_backward_tensor<T: Float>(
    x: &Tensor<T>,
    norm: &Tensor<T>,
    grad_output: &Tensor<T>,
    p: f64,
) -> FerrotorchResult<Tensor<T>> {
    if x.is_cuda() && (is_f16::<T>() || is_bf16::<T>()) {
        let x32 = x.to_dtype::<f32>()?;
        let norm32 = norm.to_dtype::<f32>()?;
        let grad32 = grad_output.to_dtype::<f32>()?;
        return norm_backward_tensor(&x32, &norm32, &grad32, p)?.to_dtype::<T>();
    }

    let zero_x = crate::creation::zeros_like(x)?;
    if p == 1.0 {
        let (sign, _nonzero) = norm_sign_and_nonzero(x)?;
        return grad_output.mul_t(&sign);
    }
    if p == 2.0 {
        let norm_zero = crate::creation::zeros_like(norm)?;
        let norm_one = crate::creation::ones_like(norm)?;
        let norm_nonzero = crate::bool_tensor::BoolTensor::ne(norm, &norm_zero)?;
        let safe_norm = crate::grad_fns::comparison::where_bt(&norm_nonzero, norm, &norm_one)?;
        let raw = grad_output.mul_t(x)?.div_t(&safe_norm)?;
        return crate::grad_fns::comparison::where_bt(&norm_nonzero, &raw, &zero_x);
    }
    if p.is_infinite() {
        let abs = x.abs_t()?;
        let tie = crate::bool_tensor::BoolTensor::eq_t(&abs, norm)?;
        let tie_f = tie.to_float::<T>()?;
        let count = tie_f.count_nonzero_t()?.to_float::<T>()?;
        let (sign, _nonzero) = norm_sign_and_nonzero(x)?;
        return sign.mul_t(&tie_f)?.mul_t(grad_output)?.div_t(&count);
    }

    let (sign, nonzero) = norm_sign_and_nonzero(x)?;
    let abs = x.abs_t()?;
    if p < 1.0 {
        let ones = crate::creation::ones_like(&abs)?;
        let safe_abs = crate::grad_fns::comparison::where_bt(&nonzero, &abs, &ones)?;
        let scale = norm.pow_t(1.0 - p)?;
        let raw = sign
            .mul_t(&safe_abs.pow_t(p - 1.0)?)?
            .mul_t(grad_output)?
            .mul_t(&scale)?;
        return crate::grad_fns::comparison::where_bt(&nonzero, &raw, &zero_x);
    }

    let denom = norm.pow_t(p - 1.0)?;
    let norm_zero = crate::creation::zeros_like(norm)?;
    let norm_one = crate::creation::ones_like(norm)?;
    let norm_nonzero = crate::bool_tensor::BoolTensor::ne(norm, &norm_zero)?;
    let safe_denom = crate::grad_fns::comparison::where_bt(&norm_nonzero, &denom, &norm_one)?;
    let raw = if p < 2.0 {
        sign.mul_t(&abs.pow_t(p - 1.0)?)?
            .mul_t(grad_output)?
            .div_t(&safe_denom)?
    } else {
        x.mul_t(&abs.pow_t(p - 2.0)?)?
            .mul_t(grad_output)?
            .div_t(&safe_denom)?
    };
    crate::grad_fns::comparison::where_bt(&norm_nonzero, &raw, &zero_x)
}

/// Differentiable Frobenius `matrix_norm`. Attaches `NormBackward` when grad is
/// needed. Forward computed under `no_grad` (re-entry guard).
pub fn matrix_norm_differentiable<T: Float>(a: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let n = crate::autograd::no_grad::no_grad(|| linalg_fwd::matrix_norm(a))?;
    if is_grad_enabled() && a.requires_grad() {
        let grad_fn = Arc::new(NormBackward {
            x: a.clone(),
            norm: n.clone(),
            // Frobenius == 2-norm of the flattened entries.
            ord: 2.0,
        });
        let (storage, shape) = n.into_storage_and_shape()?;
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Ok(n)
    }
}

/// Differentiable `vector_norm` for EVERY accepted `ord` (CORE-047 /
/// #1741 — was `ord == 2.0` only, silently detaching the rest). Attaches
/// `NormBackward` when grad is needed; the per-`ord` VJP branches are
/// documented on [`NormBackward`]. Forward computed under `no_grad`
/// (re-entry guard).
pub fn vector_norm_differentiable<T: Float>(
    a: &Tensor<T>,
    ord: f64,
) -> FerrotorchResult<Tensor<T>> {
    let n = crate::autograd::no_grad::no_grad(|| linalg_fwd::vector_norm(a, ord))?;
    if is_grad_enabled() && a.requires_grad() {
        let grad_fn = Arc::new(NormBackward {
            x: a.clone(),
            norm: n.clone(),
            ord,
        });
        let (storage, shape) = n.into_storage_and_shape()?;
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Ok(n)
    }
}

// ===========================================================================
// Fused-affine family (#1344 / #1345): addmm / addmv / addr / addbmm /
// baddbmm + structural autograd: kron / diagonal / diag / tril / triu.
//
// Each forward is computed directly from raw CPU data (the underlying
// forward ops — `mm`, `mv`, `outer`, `bmm`, `tril`, `triu`, `diag`,
// `diagonal` — are CPU-only and error on CUDA), and each backward is a
// closed-form VJP grounded in a named PyTorch `file:line` and FD-verified in
// `tests/divergence_linalg_fused_audit.rs`.
// ===========================================================================

/// Sum-reduce `grad` (shape `grad_shape`) back onto `target` shape, handling
/// the numpy/PyTorch broadcast rules: leading dims present in `grad` but not
/// in `target` are summed out, and any `target` dim that is size-1 while the
/// corresponding `grad` dim is larger is summed with keepdim.
///
/// Used by the fused-affine `self`/bias gradient where op_db emits broadcast
/// `self` shapes (`[]`, `[1]`, `[1,1]`, `[m]`, `[m,n]`) for `addmm`/`addmv`/
/// `addr`/`addbmm`. Mirrors PyTorch's implicit `self` broadcast in
/// `TORCH_META_FUNC(addmm)` (`aten/src/ATen/native/LinearAlgebra.cpp:194`),
/// whose VJP `self: maybe_multiply(grad, beta)` is then reduced to `self`'s
/// shape by the autograd engine's `sum_to`.
fn reduce_grad_to_shape<T: Float>(grad: &[T], grad_shape: &[usize], target: &[usize]) -> Vec<T> {
    if grad_shape == target {
        return grad.to_vec();
    }
    let zero = <T as num_traits::Zero>::zero();
    let target_size: usize = crate::shape::numel(target).max(1);
    let mut out = vec![zero; target_size];

    let grad_nd = grad_shape.len();
    let target_nd = target.len();
    let offset = grad_nd - target_nd;

    let mut target_strides = vec![1usize; target_nd];
    for i in (0..target_nd.saturating_sub(1)).rev() {
        target_strides[i] = target_strides[i + 1] * target[i + 1];
    }

    let grad_total: usize = crate::shape::numel(grad_shape).max(1);
    for (flat, &g) in grad.iter().enumerate().take(grad_total) {
        let mut remaining = flat;
        let mut tgt_flat = 0usize;
        for d in (0..grad_nd).rev() {
            let coord = remaining % grad_shape[d];
            remaining /= grad_shape[d];
            if d >= offset {
                let td = d - offset;
                let tc = if target[td] == 1 { 0 } else { coord };
                tgt_flat += tc * target_strides[td];
            }
        }
        out[tgt_flat] += g;
    }
    out
}

/// `out[i,j] = sum_k a[i,k] * b[k,j]` — plain CPU GEMM on raw slices.
fn mm_rows<T: Float>(a: &[T], b: &[T], m: usize, k: usize, n: usize) -> Vec<T> {
    crate::ops::linalg::mm_raw(a, b, m, k, n)
}

/// `out[i,k] = sum_j a[i,j] * b[k,j]` — `a @ b^T` on raw slices.
fn mm_bt_rows<T: Float>(a: &[T], b: &[T], m: usize, n: usize, k: usize) -> Vec<T> {
    crate::ops::linalg::mm_raw_bt(a, b, m, n, k)
}

/// `out[i,j] = sum_k a[k,i] * b[k,j]` — `a^T @ b` on raw slices.
fn mm_at_rows<T: Float>(a: &[T], b: &[T], m: usize, k: usize, n: usize) -> Vec<T> {
    crate::ops::linalg::mm_raw_at(a, b, m, k, n)
}

#[inline]
fn scale_vec<T: Float>(v: &[T], s: T) -> Vec<T> {
    v.iter().map(|&x| s * x).collect()
}

#[inline]
fn from_cpu<T: Float>(data: Vec<T>, shape: Vec<usize>) -> FerrotorchResult<Tensor<T>> {
    Tensor::from_storage(TensorStorage::cpu(data), shape, false)
}

// ---------------------------------------------------------------------------
// AddmmBackward — C = beta*self + alpha*(mat1 @ mat2)   (2D)
// ---------------------------------------------------------------------------

/// Backward for `addmm`.
///
/// VJP (`tools/autograd/derivatives.yaml:256` `addmm`, with
/// `mm_mat1_backward`/`mm_mat2_backward` at
/// `torch/csrc/autograd/FunctionsManual.cpp:1486,1505`):
/// - `d_self = sum_to(beta * grad, self.shape)`
/// - `d_mat1 = alpha * (grad @ mat2^T)`
/// - `d_mat2 = alpha * (mat1^T @ grad)`
#[derive(Debug)]
pub struct AddmmBackward<T: Float> {
    bias: Tensor<T>,
    mat1: Tensor<T>,
    mat2: Tensor<T>,
    beta: T,
    alpha: T,
}

impl<T: Float> GradFn<T> for AddmmBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let m = grad_output.shape()[0];
        let n = grad_output.shape()[1];
        let g = grad_output.data()?;

        let grad_bias = if self.bias.requires_grad() {
            let scaled = scale_vec(g, self.beta);
            Some(from_cpu(
                reduce_grad_to_shape(&scaled, &[m, n], self.bias.shape()),
                self.bias.shape().to_vec(),
            )?)
        } else {
            None
        };

        let grad_mat1 = if self.mat1.requires_grad() {
            // d_mat1 = alpha * (grad @ mat2^T); mat2 is (k, n) so grad(m,n) @ mat2^T(n,k).
            let k = self.mat1.shape()[1];
            let m2 = self.mat2.data()?;
            let prod = mm_bt_rows(g, m2, m, n, k);
            Some(from_cpu(scale_vec(&prod, self.alpha), vec![m, k])?)
        } else {
            None
        };

        let grad_mat2 = if self.mat2.requires_grad() {
            // d_mat2 = alpha * (mat1^T @ grad); mat1 is (m, k).
            let k = self.mat1.shape()[1];
            let m1 = self.mat1.data()?;
            let prod = mm_at_rows(m1, g, k, m, n);
            Some(from_cpu(scale_vec(&prod, self.alpha), vec![k, n])?)
        } else {
            None
        };

        Ok(vec![grad_bias, grad_mat1, grad_mat2])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.bias, &self.mat1, &self.mat2]
    }

    fn name(&self) -> &'static str {
        "AddmmBackward"
    }
}

/// Differentiable `addmm(self, mat1, mat2, beta, alpha)` =
/// `beta*self + alpha*(mat1 @ mat2)`. Mirrors `TORCH_META_FUNC(addmm)` at
/// `aten/src/ATen/native/LinearAlgebra.cpp:194` (`self` is broadcast to the
/// `mat1 @ mat2` shape).
pub fn addmm_differentiable<T: Float>(
    bias: &Tensor<T>,
    mat1: &Tensor<T>,
    mat2: &Tensor<T>,
    beta: T,
    alpha: T,
) -> FerrotorchResult<Tensor<T>> {
    if mat1.ndim() != 2 || mat2.ndim() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "addmm: mat1/mat2 must be 2-D, got {:?} and {:?}",
                mat1.shape(),
                mat2.shape()
            ),
        });
    }
    let m = mat1.shape()[0];
    let k = mat1.shape()[1];
    if mat2.shape()[0] != k {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!("addmm: inner dims {:?} @ {:?}", mat1.shape(), mat2.shape()),
        });
    }
    let n = mat2.shape()[1];

    let m1 = mat1.data()?;
    let m2 = mat2.data()?;
    let prod = mm_rows(m1, m2, m, k, n);

    // out = beta*self_broadcast + alpha*prod.
    // When beta == 0 the self/bias term is DROPPED entirely (never read), so
    // nans/infs in self do not propagate — matches torch's
    // `aten/src/ATen/native/cpu/BlasKernel.cpp:161-162` (`if (beta == 0) c = alpha*dot;`)
    // and `aten/src/ATen/native/LinearAlgebra.cpp:1442` (self copied only when beta != 0).
    let mut out = vec![<T as num_traits::Zero>::zero(); m * n];
    if beta == <T as num_traits::Zero>::zero() {
        for i in 0..m * n {
            out[i] = alpha * prod[i];
        }
    } else {
        let bias_b = broadcast_data_to(bias, &[m, n])?;
        for i in 0..m * n {
            out[i] = beta * bias_b[i] + alpha * prod[i];
        }
    }
    let storage = TensorStorage::cpu(out);
    let shape = vec![m, n];

    if is_grad_enabled() && (bias.requires_grad() || mat1.requires_grad() || mat2.requires_grad()) {
        let grad_fn = Arc::new(AddmmBackward {
            bias: bias.clone(),
            mat1: mat1.clone(),
            mat2: mat2.clone(),
            beta,
            alpha,
        });
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Tensor::from_storage(storage, shape, false)
    }
}

/// Broadcast `t`'s data to `target` shape (numpy/PyTorch rules), returning a
/// flat row-major `Vec`. Used by the fused-affine forwards to materialise the
/// broadcast `self`/bias.
fn broadcast_data_to<T: Float>(t: &Tensor<T>, target: &[usize]) -> FerrotorchResult<Vec<T>> {
    let src = t.data()?;
    let src_shape = t.shape();
    if src_shape == target {
        return Ok(src.to_vec());
    }
    let target_size: usize = crate::shape::numel(target).max(1);
    let tnd = target.len();
    let snd = src_shape.len();
    if snd > tnd {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!("cannot broadcast {src_shape:?} to {target:?}"),
        });
    }
    let offset = tnd - snd;
    // Validate broadcast compatibility.
    for (d, &s) in src_shape.iter().enumerate() {
        let td = target[offset + d];
        if s != td && s != 1 {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!("cannot broadcast {src_shape:?} to {target:?}"),
            });
        }
    }
    let mut src_strides = vec![1usize; snd];
    for i in (0..snd.saturating_sub(1)).rev() {
        src_strides[i] = src_strides[i + 1] * src_shape[i + 1];
    }
    let mut tgt_strides = vec![1usize; tnd];
    for i in (0..tnd.saturating_sub(1)).rev() {
        tgt_strides[i] = tgt_strides[i + 1] * target[i + 1];
    }
    let mut out = vec![<T as num_traits::Zero>::zero(); target_size];
    for (flat, slot) in out.iter_mut().enumerate().take(target_size) {
        let mut rem = flat;
        let mut src_flat = 0usize;
        for d in 0..tnd {
            let coord = rem / tgt_strides[d];
            rem %= tgt_strides[d];
            if d >= offset {
                let sd = d - offset;
                let sc = if src_shape[sd] == 1 { 0 } else { coord };
                src_flat += sc * src_strides[sd];
            }
        }
        *slot = src[src_flat];
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// AddmvBackward — y = beta*self + alpha*(mat @ vec)   (2D × 1D)
// ---------------------------------------------------------------------------

/// Backward for `addmv`.
///
/// VJP (`tools/autograd/derivatives.yaml:267` `addmv`):
/// - `d_self = sum_to(beta * grad, self.shape)`
/// - `d_mat  = alpha * outer(grad, vec)`     (`grad.ger(vec)`)
/// - `d_vec  = alpha * (mat^T @ grad)`        (`mat.t().mv(grad)`)
#[derive(Debug)]
pub struct AddmvBackward<T: Float> {
    bias: Tensor<T>,
    mat: Tensor<T>,
    vec: Tensor<T>,
    beta: T,
    alpha: T,
}

impl<T: Float> GradFn<T> for AddmvBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let m = grad_output.shape()[0];
        let g = grad_output.data()?;
        let k = self.mat.shape()[1];

        let grad_bias = if self.bias.requires_grad() {
            let scaled = scale_vec(g, self.beta);
            Some(from_cpu(
                reduce_grad_to_shape(&scaled, &[m], self.bias.shape()),
                self.bias.shape().to_vec(),
            )?)
        } else {
            None
        };

        let grad_mat = if self.mat.requires_grad() {
            // d_mat = alpha * outer(grad, vec): (m, k) with out[i,j] = g[i]*vec[j].
            let v = self.vec.data()?;
            let mut out = vec![<T as num_traits::Zero>::zero(); m * k];
            for i in 0..m {
                let gi = self.alpha * g[i];
                for j in 0..k {
                    out[i * k + j] = gi * v[j];
                }
            }
            Some(from_cpu(out, vec![m, k])?)
        } else {
            None
        };

        let grad_vec = if self.vec.requires_grad() {
            // d_vec = alpha * (mat^T @ grad): mat is (m, k); out[j] = sum_i mat[i,j]*g[i].
            let mat = self.mat.data()?;
            let mut out = vec![<T as num_traits::Zero>::zero(); k];
            for i in 0..m {
                let gi = g[i];
                let row = i * k;
                for j in 0..k {
                    out[j] += mat[row + j] * gi;
                }
            }
            Some(from_cpu(scale_vec(&out, self.alpha), vec![k])?)
        } else {
            None
        };

        Ok(vec![grad_bias, grad_mat, grad_vec])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.bias, &self.mat, &self.vec]
    }

    fn name(&self) -> &'static str {
        "AddmvBackward"
    }
}

/// Differentiable `addmv(self, mat, vec, beta, alpha)` =
/// `beta*self + alpha*(mat @ vec)`. Mirrors `TORCH_META_FUNC(addmv)` at
/// `aten/src/ATen/native/Blas.cpp:40`.
pub fn addmv_differentiable<T: Float>(
    bias: &Tensor<T>,
    mat: &Tensor<T>,
    vec: &Tensor<T>,
    beta: T,
    alpha: T,
) -> FerrotorchResult<Tensor<T>> {
    if mat.ndim() != 2 || vec.ndim() != 1 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "addmv: mat must be 2-D, vec 1-D, got {:?} and {:?}",
                mat.shape(),
                vec.shape()
            ),
        });
    }
    let m = mat.shape()[0];
    let k = mat.shape()[1];
    if vec.shape()[0] != k {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!("addmv: {:?} @ {:?}", mat.shape(), vec.shape()),
        });
    }
    let mat_d = mat.data()?;
    let vec_d = vec.data()?;
    let mut prod = vec![<T as num_traits::Zero>::zero(); m];
    for (i, slot) in prod.iter_mut().enumerate() {
        let mut acc = <T as num_traits::Zero>::zero();
        let row = i * k;
        for j in 0..k {
            acc += mat_d[row + j] * vec_d[j];
        }
        *slot = acc;
    }
    // When beta == 0 the self term is DROPPED entirely (never read), so
    // nans/infs in self do not propagate — matches torch's
    // `aten/src/ATen/native/Blas.cpp:77-79,90` ("when beta==0, values in self
    // should be ignored ... self copied only when betaval != 0.0").
    let mut out = vec![<T as num_traits::Zero>::zero(); m];
    if beta == <T as num_traits::Zero>::zero() {
        for i in 0..m {
            out[i] = alpha * prod[i];
        }
    } else {
        let bias_b = broadcast_data_to(bias, &[m])?;
        for i in 0..m {
            out[i] = beta * bias_b[i] + alpha * prod[i];
        }
    }
    let storage = TensorStorage::cpu(out);
    let shape = vec![m];

    if is_grad_enabled() && (bias.requires_grad() || mat.requires_grad() || vec.requires_grad()) {
        let grad_fn = Arc::new(AddmvBackward {
            bias: bias.clone(),
            mat: mat.clone(),
            vec: vec.clone(),
            beta,
            alpha,
        });
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Tensor::from_storage(storage, shape, false)
    }
}

// ---------------------------------------------------------------------------
// AddrBackward — C = beta*self + alpha*outer(vec1, vec2)   (1D × 1D -> 2D)
// ---------------------------------------------------------------------------

/// Backward for `addr`.
///
/// VJP (`tools/autograd/derivatives.yaml:273` `addr`):
/// - `d_self = sum_to(beta * grad, self.shape)`
/// - `d_vec1 = alpha * (grad @ vec2)`     (`grad.mv(vec2)`)
/// - `d_vec2 = alpha * (grad^T @ vec1)`   (`grad.t().mv(vec1)`)
#[derive(Debug)]
pub struct AddrBackward<T: Float> {
    bias: Tensor<T>,
    vec1: Tensor<T>,
    vec2: Tensor<T>,
    beta: T,
    alpha: T,
}

impl<T: Float> GradFn<T> for AddrBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let m = grad_output.shape()[0];
        let n = grad_output.shape()[1];
        let g = grad_output.data()?;

        let grad_bias = if self.bias.requires_grad() {
            let scaled = scale_vec(g, self.beta);
            Some(from_cpu(
                reduce_grad_to_shape(&scaled, &[m, n], self.bias.shape()),
                self.bias.shape().to_vec(),
            )?)
        } else {
            None
        };

        let grad_vec1 = if self.vec1.requires_grad() {
            // d_vec1 = alpha * (grad @ vec2): out[i] = sum_j grad[i,j]*vec2[j].
            let v2 = self.vec2.data()?;
            let mut out = vec![<T as num_traits::Zero>::zero(); m];
            for i in 0..m {
                let mut acc = <T as num_traits::Zero>::zero();
                let row = i * n;
                for j in 0..n {
                    acc += g[row + j] * v2[j];
                }
                out[i] = self.alpha * acc;
            }
            Some(from_cpu(out, vec![m])?)
        } else {
            None
        };

        let grad_vec2 = if self.vec2.requires_grad() {
            // d_vec2 = alpha * (grad^T @ vec1): out[j] = sum_i grad[i,j]*vec1[i].
            let v1 = self.vec1.data()?;
            let mut out = vec![<T as num_traits::Zero>::zero(); n];
            for i in 0..m {
                let v1i = v1[i];
                let row = i * n;
                for j in 0..n {
                    out[j] += g[row + j] * v1i;
                }
            }
            Some(from_cpu(scale_vec(&out, self.alpha), vec![n])?)
        } else {
            None
        };

        Ok(vec![grad_bias, grad_vec1, grad_vec2])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.bias, &self.vec1, &self.vec2]
    }

    fn name(&self) -> &'static str {
        "AddrBackward"
    }
}

/// Differentiable `addr(self, vec1, vec2, beta, alpha)` =
/// `beta*self + alpha*outer(vec1, vec2)`. Mirrors `Tensor addr(...)` at
/// `aten/src/ATen/native/LinearAlgebra.cpp:1200`.
pub fn addr_differentiable<T: Float>(
    bias: &Tensor<T>,
    vec1: &Tensor<T>,
    vec2: &Tensor<T>,
    beta: T,
    alpha: T,
) -> FerrotorchResult<Tensor<T>> {
    if vec1.ndim() != 1 || vec2.ndim() != 1 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "addr: vec1/vec2 must be 1-D, got {:?} and {:?}",
                vec1.shape(),
                vec2.shape()
            ),
        });
    }
    let m = vec1.shape()[0];
    let n = vec2.shape()[0];
    let v1 = vec1.data()?;
    let v2 = vec2.data()?;
    // When beta == 0 the self term is DROPPED entirely (never read), so
    // nans/infs in self do not propagate — matches torch's
    // `aten/src/ATen/native/cpu/LinearAlgebraKernel.cpp:53-55,60`
    // ("when beta == 0, values in self should be ignored, nans and infs in self
    // should not propagate" + `return alpha_val * vec1_val * vec2_val;`).
    let mut out = vec![<T as num_traits::Zero>::zero(); m * n];
    if beta == <T as num_traits::Zero>::zero() {
        for i in 0..m {
            let av1 = alpha * v1[i];
            let row = i * n;
            for j in 0..n {
                out[row + j] = av1 * v2[j];
            }
        }
    } else {
        let bias_b = broadcast_data_to(bias, &[m, n])?;
        for i in 0..m {
            let av1 = alpha * v1[i];
            let row = i * n;
            for j in 0..n {
                out[row + j] = beta * bias_b[row + j] + av1 * v2[j];
            }
        }
    }
    let storage = TensorStorage::cpu(out);
    let shape = vec![m, n];

    if is_grad_enabled() && (bias.requires_grad() || vec1.requires_grad() || vec2.requires_grad()) {
        let grad_fn = Arc::new(AddrBackward {
            bias: bias.clone(),
            vec1: vec1.clone(),
            vec2: vec2.clone(),
            beta,
            alpha,
        });
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Tensor::from_storage(storage, shape, false)
    }
}

// ---------------------------------------------------------------------------
// BaddbmmBackward — C[b] = beta*self[b] + alpha*(batch1[b] @ batch2[b])  (3D)
// ---------------------------------------------------------------------------

/// Backward for `baddbmm`.
///
/// VJP (`tools/autograd/derivatives.yaml:359` `baddbmm`):
/// - `d_self   = sum_to(beta * grad, self.shape)`
/// - `d_batch1 = alpha * bmm(grad, batch2^T)`   per batch
/// - `d_batch2 = alpha * bmm(batch1^T, grad)`   per batch
#[derive(Debug)]
pub struct BaddbmmBackward<T: Float> {
    bias: Tensor<T>,
    batch1: Tensor<T>,
    batch2: Tensor<T>,
    beta: T,
    alpha: T,
}

impl<T: Float> GradFn<T> for BaddbmmBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let bsz = grad_output.shape()[0];
        let m = grad_output.shape()[1];
        let n = grad_output.shape()[2];
        let k = self.batch1.shape()[2];
        let g = grad_output.data()?;

        let grad_bias = if self.bias.requires_grad() {
            let scaled = scale_vec(g, self.beta);
            Some(from_cpu(
                reduce_grad_to_shape(&scaled, &[bsz, m, n], self.bias.shape()),
                self.bias.shape().to_vec(),
            )?)
        } else {
            None
        };

        let grad_b1 = if self.batch1.requires_grad() {
            let b2 = self.batch2.data()?;
            let mut out = vec![<T as num_traits::Zero>::zero(); bsz * m * k];
            for bi in 0..bsz {
                let g_off = bi * m * n;
                let b2_off = bi * k * n;
                let o_off = bi * m * k;
                // d_b1[b] = alpha * (grad[b] @ batch2[b]^T): grad(m,n) @ b2(k,n)^T.
                let slab = mm_bt_rows(
                    &g[g_off..g_off + m * n],
                    &b2[b2_off..b2_off + k * n],
                    m,
                    n,
                    k,
                );
                for (i, &v) in slab.iter().enumerate() {
                    out[o_off + i] = self.alpha * v;
                }
            }
            Some(from_cpu(out, vec![bsz, m, k])?)
        } else {
            None
        };

        let grad_b2 = if self.batch2.requires_grad() {
            let b1 = self.batch1.data()?;
            let mut out = vec![<T as num_traits::Zero>::zero(); bsz * k * n];
            for bi in 0..bsz {
                let g_off = bi * m * n;
                let b1_off = bi * m * k;
                let o_off = bi * k * n;
                // d_b2[b] = alpha * (batch1[b]^T @ grad[b]): b1(m,k)^T @ grad(m,n).
                let slab = mm_at_rows(
                    &b1[b1_off..b1_off + m * k],
                    &g[g_off..g_off + m * n],
                    k,
                    m,
                    n,
                );
                for (i, &v) in slab.iter().enumerate() {
                    out[o_off + i] = self.alpha * v;
                }
            }
            Some(from_cpu(out, vec![bsz, k, n])?)
        } else {
            None
        };

        Ok(vec![grad_bias, grad_b1, grad_b2])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.bias, &self.batch1, &self.batch2]
    }

    fn name(&self) -> &'static str {
        "BaddbmmBackward"
    }
}

/// Differentiable `baddbmm(self, batch1, batch2, beta, alpha)` =
/// `beta*self + alpha*bmm(batch1, batch2)`. Mirrors `TORCH_META_FUNC(baddbmm)`
/// at `aten/src/ATen/native/LinearAlgebra.cpp:340`.
pub fn baddbmm_differentiable<T: Float>(
    bias: &Tensor<T>,
    batch1: &Tensor<T>,
    batch2: &Tensor<T>,
    beta: T,
    alpha: T,
) -> FerrotorchResult<Tensor<T>> {
    if batch1.ndim() != 3 || batch2.ndim() != 3 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "baddbmm: batch1/batch2 must be 3-D, got {:?} and {:?}",
                batch1.shape(),
                batch2.shape()
            ),
        });
    }
    let bsz = batch1.shape()[0];
    let m = batch1.shape()[1];
    let k = batch1.shape()[2];
    let n = batch2.shape()[2];
    if batch2.shape()[0] != bsz || batch2.shape()[1] != k {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!("baddbmm: {:?} @ {:?}", batch1.shape(), batch2.shape()),
        });
    }
    let b1 = batch1.data()?;
    let b2 = batch2.data()?;
    let mut prod = vec![<T as num_traits::Zero>::zero(); bsz * m * n];
    for bi in 0..bsz {
        let a_off = bi * m * k;
        let b_off = bi * k * n;
        let c_off = bi * m * n;
        let slab = mm_rows(
            &b1[a_off..a_off + m * k],
            &b2[b_off..b_off + k * n],
            m,
            k,
            n,
        );
        prod[c_off..c_off + m * n].copy_from_slice(&slab);
    }
    // When beta == 0 the self term is DROPPED entirely (never read), so
    // nans/infs in self do not propagate — matches torch's
    // `aten/src/ATen/native/LinearAlgebra.cpp:1682-1684`
    // ("For beta == 0, the r's value will be ignored, especially for nan value.").
    let mut out = vec![<T as num_traits::Zero>::zero(); bsz * m * n];
    if beta == <T as num_traits::Zero>::zero() {
        for i in 0..out.len() {
            out[i] = alpha * prod[i];
        }
    } else {
        let bias_b = broadcast_data_to(bias, &[bsz, m, n])?;
        for i in 0..out.len() {
            out[i] = beta * bias_b[i] + alpha * prod[i];
        }
    }
    let storage = TensorStorage::cpu(out);
    let shape = vec![bsz, m, n];

    if is_grad_enabled()
        && (bias.requires_grad() || batch1.requires_grad() || batch2.requires_grad())
    {
        let grad_fn = Arc::new(BaddbmmBackward {
            bias: bias.clone(),
            batch1: batch1.clone(),
            batch2: batch2.clone(),
            beta,
            alpha,
        });
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Tensor::from_storage(storage, shape, false)
    }
}

// ---------------------------------------------------------------------------
// AddbmmBackward — C = beta*self + alpha*sum_b(batch1[b] @ batch2[b])  (2D out)
// ---------------------------------------------------------------------------

/// Backward for `addbmm`.
///
/// VJP (`tools/autograd/derivatives.yaml:238` `addbmm`):
/// - `d_self   = sum_to(beta * grad, self.shape)`
/// - `d_batch1[b] = alpha * (grad @ batch2[b]^T)`   (grad broadcast over batch)
/// - `d_batch2[b] = alpha * (batch1[b]^T @ grad)`
///
/// The forward sums the per-batch products, so the upstream `grad` (shape
/// `[m,n]`) is shared by every batch slab in the backward.
#[derive(Debug)]
pub struct AddbmmBackward<T: Float> {
    bias: Tensor<T>,
    batch1: Tensor<T>,
    batch2: Tensor<T>,
    beta: T,
    alpha: T,
}

impl<T: Float> GradFn<T> for AddbmmBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let m = grad_output.shape()[0];
        let n = grad_output.shape()[1];
        let bsz = self.batch1.shape()[0];
        let k = self.batch1.shape()[2];
        let g = grad_output.data()?;

        let grad_bias = if self.bias.requires_grad() {
            let scaled = scale_vec(g, self.beta);
            Some(from_cpu(
                reduce_grad_to_shape(&scaled, &[m, n], self.bias.shape()),
                self.bias.shape().to_vec(),
            )?)
        } else {
            None
        };

        let grad_b1 = if self.batch1.requires_grad() {
            let b2 = self.batch2.data()?;
            let mut out = vec![<T as num_traits::Zero>::zero(); bsz * m * k];
            for bi in 0..bsz {
                let b2_off = bi * k * n;
                let o_off = bi * m * k;
                // grad is shared (broadcast over batch): d_b1[b] = alpha*(grad @ b2[b]^T).
                let slab = mm_bt_rows(g, &b2[b2_off..b2_off + k * n], m, n, k);
                for (i, &v) in slab.iter().enumerate() {
                    out[o_off + i] = self.alpha * v;
                }
            }
            Some(from_cpu(out, vec![bsz, m, k])?)
        } else {
            None
        };

        let grad_b2 = if self.batch2.requires_grad() {
            let b1 = self.batch1.data()?;
            let mut out = vec![<T as num_traits::Zero>::zero(); bsz * k * n];
            for bi in 0..bsz {
                let b1_off = bi * m * k;
                let o_off = bi * k * n;
                let slab = mm_at_rows(&b1[b1_off..b1_off + m * k], g, k, m, n);
                for (i, &v) in slab.iter().enumerate() {
                    out[o_off + i] = self.alpha * v;
                }
            }
            Some(from_cpu(out, vec![bsz, k, n])?)
        } else {
            None
        };

        Ok(vec![grad_bias, grad_b1, grad_b2])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.bias, &self.batch1, &self.batch2]
    }

    fn name(&self) -> &'static str {
        "AddbmmBackward"
    }
}

/// Differentiable `addbmm(self, batch1, batch2, beta, alpha)` =
/// `beta*self + alpha*sum_b(batch1[b] @ batch2[b])`. Mirrors `Tensor addbmm(...)`
/// at `aten/src/ATen/native/LinearAlgebra.cpp:1615`.
pub fn addbmm_differentiable<T: Float>(
    bias: &Tensor<T>,
    batch1: &Tensor<T>,
    batch2: &Tensor<T>,
    beta: T,
    alpha: T,
) -> FerrotorchResult<Tensor<T>> {
    if batch1.ndim() != 3 || batch2.ndim() != 3 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "addbmm: batch1/batch2 must be 3-D, got {:?} and {:?}",
                batch1.shape(),
                batch2.shape()
            ),
        });
    }
    let bsz = batch1.shape()[0];
    let m = batch1.shape()[1];
    let k = batch1.shape()[2];
    let n = batch2.shape()[2];
    if batch2.shape()[0] != bsz || batch2.shape()[1] != k {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!("addbmm: {:?} @ {:?}", batch1.shape(), batch2.shape()),
        });
    }
    let b1 = batch1.data()?;
    let b2 = batch2.data()?;
    let mut acc = vec![<T as num_traits::Zero>::zero(); m * n];
    for bi in 0..bsz {
        let a_off = bi * m * k;
        let b_off = bi * k * n;
        let slab = mm_rows(
            &b1[a_off..a_off + m * k],
            &b2[b_off..b_off + k * n],
            m,
            k,
            n,
        );
        for (i, &v) in slab.iter().enumerate() {
            acc[i] += v;
        }
    }
    // When beta == 0 the self term is DROPPED entirely (never read), so
    // nans/infs in self do not propagate — matches torch's
    // `aten/src/ATen/native/LinearAlgebra.cpp:1682-1684`
    // ("For beta == 0, the r's value will be ignored, especially for nan value.").
    let mut out = vec![<T as num_traits::Zero>::zero(); m * n];
    if beta == <T as num_traits::Zero>::zero() {
        for i in 0..m * n {
            out[i] = alpha * acc[i];
        }
    } else {
        let bias_b = broadcast_data_to(bias, &[m, n])?;
        for i in 0..m * n {
            out[i] = beta * bias_b[i] + alpha * acc[i];
        }
    }
    let storage = TensorStorage::cpu(out);
    let shape = vec![m, n];

    if is_grad_enabled()
        && (bias.requires_grad() || batch1.requires_grad() || batch2.requires_grad())
    {
        let grad_fn = Arc::new(AddbmmBackward {
            bias: bias.clone(),
            batch1: batch1.clone(),
            batch2: batch2.clone(),
            beta,
            alpha,
        });
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Tensor::from_storage(storage, shape, false)
    }
}

// ---------------------------------------------------------------------------
// KronBackward — K = kron(A, B)  (2D × 2D)
// ---------------------------------------------------------------------------

/// Backward for the 2-D Kronecker product `K = kron(A, B)`.
///
/// Forward (2-D case of `Tensor kron(...)` at
/// `aten/src/ATen/native/LinearAlgebra.cpp:3530`, the reshape, broadcast-mul,
/// and view recipe `KronImpl::kron`): for `A` `(p, q)` and `B` `(r, s)`,
/// the result is `K[i*r + u, j*s + v] = A[i,j] * B[u,v]`, shape `(p*r, q*s)`.
///
/// Backward (adjoint of the bilinear product, equivalently the autograd of the
/// reshape/mul recipe):
/// - `dA[i,j] = sum_{u,v} grad[i*r+u, j*s+v] * B[u,v]`
/// - `dB[u,v] = sum_{i,j} grad[i*r+u, j*s+v] * A[i,j]`
#[derive(Debug)]
pub struct KronBackward<T: Float> {
    a: Tensor<T>,
    b: Tensor<T>,
}

impl<T: Float> GradFn<T> for KronBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let p = self.a.shape()[0];
        let q = self.a.shape()[1];
        let r = self.b.shape()[0];
        let s = self.b.shape()[1];
        let cols = q * s;
        let g = grad_output.data()?;
        let zero = <T as num_traits::Zero>::zero();

        let grad_a = if self.a.requires_grad() {
            let bd = self.b.data()?;
            let mut out = vec![zero; p * q];
            for i in 0..p {
                for j in 0..q {
                    let mut acc = zero;
                    for u in 0..r {
                        let grow = (i * r + u) * cols;
                        for v in 0..s {
                            acc += g[grow + j * s + v] * bd[u * s + v];
                        }
                    }
                    out[i * q + j] = acc;
                }
            }
            Some(from_cpu(out, vec![p, q])?)
        } else {
            None
        };

        let grad_b = if self.b.requires_grad() {
            let ad = self.a.data()?;
            let mut out = vec![zero; r * s];
            for u in 0..r {
                for v in 0..s {
                    let mut acc = zero;
                    for i in 0..p {
                        for j in 0..q {
                            acc += g[(i * r + u) * cols + j * s + v] * ad[i * q + j];
                        }
                    }
                    out[u * s + v] = acc;
                }
            }
            Some(from_cpu(out, vec![r, s])?)
        } else {
            None
        };

        Ok(vec![grad_a, grad_b])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.a, &self.b]
    }

    fn name(&self) -> &'static str {
        "KronBackward"
    }
}

/// Differentiable 2-D Kronecker product. Mirrors the 2-D specialisation of
/// `Tensor kron(const Tensor& self, const Tensor& other)` at
/// `aten/src/ATen/native/LinearAlgebra.cpp:3530`.
pub fn kron_differentiable<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if a.ndim() != 2 || b.ndim() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "kron: only 2-D × 2-D supported here, got {:?} and {:?}",
                a.shape(),
                b.shape()
            ),
        });
    }
    let p = a.shape()[0];
    let q = a.shape()[1];
    let r = b.shape()[0];
    let s = b.shape()[1];
    let rows = p * r;
    let cols = q * s;
    let ad = a.data()?;
    let bd = b.data()?;
    let mut out = vec![<T as num_traits::Zero>::zero(); rows * cols];
    for i in 0..p {
        for j in 0..q {
            let aij = ad[i * q + j];
            for u in 0..r {
                let orow = (i * r + u) * cols;
                for v in 0..s {
                    out[orow + j * s + v] = aij * bd[u * s + v];
                }
            }
        }
    }
    let storage = TensorStorage::cpu(out);
    let shape = vec![rows, cols];

    if is_grad_enabled() && (a.requires_grad() || b.requires_grad()) {
        let grad_fn = Arc::new(KronBackward {
            a: a.clone(),
            b: b.clone(),
        });
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Tensor::from_storage(storage, shape, false)
    }
}

// ---------------------------------------------------------------------------
// DiagonalBackward — d = diagonal(A, offset)  (2D -> 1D)
// ---------------------------------------------------------------------------

fn diag_scatter_to_shape<T: Float>(
    grad_output: &Tensor<T>,
    rows: usize,
    cols: usize,
    offset: i64,
) -> FerrotorchResult<Tensor<T>> {
    if grad_output.is_cuda() {
        let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let handle = match <T as Element>::dtype() {
            DType::F32 => {
                backend.diag_scatter_f32(grad_output.gpu_handle()?, rows, cols, offset)?
            }
            DType::F64 => {
                backend.diag_scatter_f64(grad_output.gpu_handle()?, rows, cols, offset)?
            }
            DType::F16 | DType::BF16 => {
                backend.diag_scatter_u16(grad_output.gpu_handle()?, rows, cols, offset)?
            }
            _ => {
                return Err(FerrotorchError::NotImplementedOnCuda {
                    op: "diag scatter backward",
                });
            }
        };
        return Tensor::from_storage(TensorStorage::gpu(handle), vec![rows, cols], false);
    }

    let g = grad_output.data_vec()?;
    let zero = <T as num_traits::Zero>::zero();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| FerrotorchError::InvalidArgument {
            message: format!("diagonal backward: shape [{rows}, {cols}] overflows storage size"),
        })?;
    let mut out = vec![zero; total];
    let (row_start, col_start) = if offset >= 0 {
        (0usize, offset as usize)
    } else {
        let row_start = usize::try_from(offset.unsigned_abs()).map_err(|_| {
            FerrotorchError::InvalidArgument {
                message: format!("diagonal backward: offset {offset} overflows usize"),
            }
        })?;
        (row_start, 0usize)
    };
    for (i, &gv) in g.iter().enumerate() {
        let r = row_start + i;
        let c = col_start + i;
        let idx = r
            .checked_mul(cols)
            .and_then(|base| base.checked_add(c))
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: "diagonal backward: diagonal index overflows storage size".into(),
            })?;
        if idx >= out.len() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "diagonal backward: grad length {} exceeds target diagonal in shape [{rows}, {cols}] at offset {offset}",
                    g.len()
                ),
            });
        }
        out[idx] = gv;
    }
    from_cpu(out, vec![rows, cols])
}

/// Backward for `diagonal(A, offset)`.
///
/// VJP (`tools/autograd/derivatives.yaml:572` `diagonal` →
/// `diagonal_backward_symint`): scatter `grad` (a 1-D vector) back onto the
/// `offset`-th diagonal of a zero matrix shaped like `A`.
#[derive(Debug)]
pub struct DiagonalBackward<T: Float> {
    rows: usize,
    cols: usize,
    offset: i64,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Float> GradFn<T> for DiagonalBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let grad_input = diag_scatter_to_shape(grad_output, self.rows, self.cols, self.offset)?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![]
    }

    fn name(&self) -> &'static str {
        "DiagonalBackward"
    }
}

/// Carries the input edge for `diagonal`.
#[derive(Debug)]
struct DiagonalForward<T: Float> {
    input: Tensor<T>,
    inner: DiagonalBackward<T>,
}

impl<T: Float> GradFn<T> for DiagonalForward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        self.inner.backward(grad_output)
    }
    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }
    fn name(&self) -> &'static str {
        "DiagonalBackward"
    }
}

/// Differentiable `diagonal(A, offset)`. Mirrors `Tensor linalg_diagonal(...)`
/// at `aten/src/ATen/native/LinearAlgebra.cpp:2215`.
pub fn diagonal_differentiable<T: Float>(
    a: &Tensor<T>,
    offset: i64,
) -> FerrotorchResult<Tensor<T>> {
    // Forward computed under `no_grad`: `linalg_fwd::diagonal` delegates back
    // here when grad is enabled, so the bare `no_grad` call prevents re-entry.
    let result = crate::autograd::no_grad::no_grad(|| linalg_fwd::diagonal(a, offset))?;
    if is_grad_enabled() && a.requires_grad() {
        let shape = a.shape();
        let grad_fn = Arc::new(DiagonalForward {
            input: a.clone(),
            inner: DiagonalBackward {
                rows: shape[0],
                cols: shape[1],
                offset,
                _marker: std::marker::PhantomData,
            },
        });
        let (storage, sh) = result.into_storage_and_shape()?;
        Tensor::from_operation(storage, sh, grad_fn)
    } else {
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// DiagBackward — d = diag(A, diagonal)  (1D->2D construct or 2D->1D extract)
// ---------------------------------------------------------------------------

/// Backward for `diag(A, diagonal)` — the adjoint of `torch.diag` (a pure
/// gather/scatter of elements onto/off the `diagonal`-th diagonal), so the
/// VJP simply applies the inverse selection (PyTorch derives this composite
/// gradient automatically; the adjoint of a 0/1 selection is its transpose).
#[derive(Debug)]
pub struct DiagBackward<T: Float> {
    /// `true` if forward was 1-D → 2-D (construct); `false` if 2-D → 1-D
    /// (extract).
    construct: bool,
    /// Input shape (1-D `[n]` or 2-D `[rows, cols]`).
    in_shape: Vec<usize>,
    diagonal: i64,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Float> GradFn<T> for DiagBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if self.construct {
            // Forward was 1-D -> 2-D diagonal matrix; grad is 2-D, grad_input
            // is the same diagonal extracted from grad (1-D). Reuse the public
            // structural kernel under no_grad so CUDA gradients stay resident.
            let grad_input = crate::autograd::no_grad::no_grad(|| {
                crate::ops::tensor_ops::diag(grad_output, self.diagonal)
            })?;
            Ok(vec![Some(grad_input)])
        } else {
            // Forward was 2-D -> 1-D extract; grad is 1-D, grad_input scatters
            // grad onto the `diagonal`-th diagonal of a zero matrix.
            let rows = self.in_shape[0];
            let cols = self.in_shape[1];
            let grad_input = diag_scatter_to_shape(grad_output, rows, cols, self.diagonal)?;
            Ok(vec![Some(grad_input)])
        }
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![]
    }

    fn name(&self) -> &'static str {
        "DiagBackward"
    }
}

/// Carries the input edge for `diag`.
#[derive(Debug)]
struct DiagForward<T: Float> {
    input: Tensor<T>,
    inner: DiagBackward<T>,
}

impl<T: Float> GradFn<T> for DiagForward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        self.inner.backward(grad_output)
    }
    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }
    fn name(&self) -> &'static str {
        "DiagBackward"
    }
}

/// Differentiable `diag(A, diagonal)`. Forward is
/// `crate::ops::tensor_ops::diag` (1-D → 2-D construct or 2-D → 1-D extract);
/// VJP is the adjoint selection.
pub fn diag_differentiable<T: Float>(a: &Tensor<T>, diagonal: i64) -> FerrotorchResult<Tensor<T>> {
    // Forward under `no_grad`: `crate::ops::tensor_ops::diag` delegates back
    // here when grad is enabled, so the bare `no_grad` call prevents re-entry.
    let result = crate::autograd::no_grad::no_grad(|| crate::ops::tensor_ops::diag(a, diagonal))?;
    if is_grad_enabled() && a.requires_grad() {
        let grad_fn = Arc::new(DiagForward {
            input: a.clone(),
            inner: DiagBackward {
                construct: a.ndim() == 1,
                in_shape: a.shape().to_vec(),
                diagonal,
                _marker: std::marker::PhantomData,
            },
        });
        let (storage, sh) = result.into_storage_and_shape()?;
        Tensor::from_operation(storage, sh, grad_fn)
    } else {
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// TrilBackward / TriuBackward — masked triangular zeroing  (2D)
// ---------------------------------------------------------------------------

/// Backward for `tril(A, diagonal)` / `triu(A, diagonal)`.
///
/// VJP (`tools/autograd/derivatives.yaml:1805,1809`:
/// `tril -> grad.tril_symint(diagonal)`, `triu -> grad.triu_symint(diagonal)`):
/// the same triangular mask applied to the upstream gradient. The mask runs over
/// the LAST TWO dims and is batched over all leading dims, so the gradient keeps
/// the full input shape (matching the now-batched forward + torch).
#[derive(Debug)]
pub struct TriangularBackward<T: Float> {
    diagonal: i64,
    /// `true` for `tril` (keep `c <= r + diag`), `false` for `triu`
    /// (keep `c >= r + diag`).
    lower: bool,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Float> GradFn<T> for TriangularBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let grad_input = crate::autograd::no_grad::no_grad(|| {
            if self.lower {
                crate::ops::tensor_ops::tril(grad_output, self.diagonal)
            } else {
                crate::ops::tensor_ops::triu(grad_output, self.diagonal)
            }
        })?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![]
    }

    fn name(&self) -> &'static str {
        if self.lower {
            "TrilBackward"
        } else {
            "TriuBackward"
        }
    }
}

/// Carries the input edge for `tril`/`triu`.
#[derive(Debug)]
struct TriangularForward<T: Float> {
    input: Tensor<T>,
    inner: TriangularBackward<T>,
}

impl<T: Float> GradFn<T> for TriangularForward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        self.inner.backward(grad_output)
    }
    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }
    fn name(&self) -> &'static str {
        self.inner.name()
    }
}

/// Differentiable `tril(A, diagonal)`. Forward is
/// `crate::ops::tensor_ops::tril` (2-D, lower-triangular zeroing).
pub fn tril_differentiable<T: Float>(a: &Tensor<T>, diagonal: i64) -> FerrotorchResult<Tensor<T>> {
    // Forward under `no_grad`: `crate::ops::tensor_ops::tril` delegates back
    // here when grad is enabled, so the bare `no_grad` call prevents re-entry.
    let result = crate::autograd::no_grad::no_grad(|| crate::ops::tensor_ops::tril(a, diagonal))?;
    if is_grad_enabled() && a.requires_grad() {
        let grad_fn = Arc::new(TriangularForward {
            input: a.clone(),
            inner: TriangularBackward {
                diagonal,
                lower: true,
                _marker: std::marker::PhantomData,
            },
        });
        let (storage, sh) = result.into_storage_and_shape()?;
        Tensor::from_operation(storage, sh, grad_fn)
    } else {
        Ok(result)
    }
}

/// Differentiable `triu(A, diagonal)`. Forward is
/// `crate::ops::tensor_ops::triu` (2-D, upper-triangular zeroing).
pub fn triu_differentiable<T: Float>(a: &Tensor<T>, diagonal: i64) -> FerrotorchResult<Tensor<T>> {
    // Forward under `no_grad`: `crate::ops::tensor_ops::triu` delegates back
    // here when grad is enabled, so the bare `no_grad` call prevents re-entry.
    let result = crate::autograd::no_grad::no_grad(|| crate::ops::tensor_ops::triu(a, diagonal))?;
    if is_grad_enabled() && a.requires_grad() {
        let grad_fn = Arc::new(TriangularForward {
            input: a.clone(),
            inner: TriangularBackward {
                diagonal,
                lower: false,
                _marker: std::marker::PhantomData,
            },
        });
        let (storage, sh) = result.into_storage_and_shape()?;
        Tensor::from_operation(storage, sh, grad_fn)
    } else {
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// HouseholderProductBackward — Q[:, :k] = householder_product(V, tau)
//   (real case; V is [m,k] with implicit unit diagonal + zeros above; tau [k])
// ---------------------------------------------------------------------------

/// Backward for `Q = householder_product(V, tau)` (real case).
///
/// `Q = H_0 H_1 ... H_{k-1}` where `H_i = I - tau_i v_i v_i^T` and `v_i` is the
/// `i`-th reflector (column `i` of `V` with implicit unit at row `i` and zeros
/// above). The public forward returns the first `k` columns of `Q`, matching
/// `torch.linalg.householder_product` (shape `[m, k]`).
///
/// The VJP mirrors `householder_product_backward` (real, `flip_order = false`)
/// at `torch/csrc/autograd/FunctionsManual.cpp:5544`. Given `grad` (shape
/// `[m, k]`):
/// 1. `input = tril(V, -1)` with unit diagonal (`FunctionsManual.cpp:5564-5565`).
/// 2. `sigma_j = tau_j / (tau_j * ||input[:, j]||^2 - 1)` so
///    `H(sigma_j) = H(tau_j)^{-1}` (`FunctionsManual.cpp:5574-5577`).
/// 3. `K = Q_full @ grad^T` where `Q_full` is the full `[m, m]` orthogonal
///    matrix (`grad` is zero-extended to `[m, m]`)
///    (`FunctionsManual.cpp:5579`).
/// 4. `K <- H_0^{-1} @ K` (`FunctionsManual.cpp:5638`), then for each `i` in
///    `0..k`: emit `grad_v_i`/`grad_tau_i` via `update_grad`
///    (`FunctionsManual.cpp:5593-5608`) and, when `i != k-1`, advance
///    `K <- H_{i+1}^{-1} @ K @ H_i` (`FunctionsManual.cpp:5701-5709`).
/// 5. `grad_V = tril(grad_V, -1)` (`FunctionsManual.cpp:5715`) — only the
///    strictly-lower part is active in the forward.
///
/// `Q_full` is retained because step 3 needs the full square reconstruction
/// (the public output is the truncated `[m, k]` slice).
#[derive(Debug)]
pub struct HouseholderProductBackward<T: Float> {
    /// Input reflector matrix `V` (`[m, k]`), retained for graph edges + the VJP.
    v: Tensor<T>,
    /// Input scalar coefficients `tau` (`[k]`).
    tau: Tensor<T>,
    /// Full `[m, m]` orthogonal product `Q_full = H_0 ... H_{k-1}`.
    q_full: Tensor<T>,
}

impl<T: Float> HouseholderProductBackward<T> {
    fn new(v: Tensor<T>, tau: Tensor<T>, q_full: Tensor<T>) -> Self {
        Self { v, tau, q_full }
    }
}

impl<T: Float> GradFn<T> for HouseholderProductBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let m = self.v.shape()[0];
        let k = self.v.shape()[1];

        let v_raw: Vec<f64> = self
            .v
            .data()?
            .iter()
            .map(|&x| x.to_f64().unwrap())
            .collect();
        let tau: Vec<f64> = self
            .tau
            .data()?
            .iter()
            .map(|&x| x.to_f64().unwrap())
            .collect();
        let q_full: Vec<f64> = self
            .q_full
            .data()?
            .iter()
            .map(|&x| x.to_f64().unwrap())
            .collect();
        let grad: Vec<f64> = grad_output
            .data()?
            .iter()
            .map(|&x| x.to_f64().unwrap())
            .collect();

        // Step 1: input = tril(V, -1) with unit diagonal (row-major [m,k]).
        // input[i,j] = V[i,j] for i>j, 1 for i==j, 0 for i<j.
        let mut input = vec![0.0f64; m * k];
        for i in 0..m {
            for j in 0..k {
                input[i * k + j] = match i.cmp(&j) {
                    std::cmp::Ordering::Equal => 1.0,
                    std::cmp::Ordering::Greater => v_raw[i * k + j],
                    std::cmp::Ordering::Less => 0.0,
                };
            }
        }

        // Step 2: sigma_j = tau_j / (tau_j * ||input[:, j]||^2 - 1).
        let mut sigma = vec![0.0f64; k];
        for j in 0..k {
            let mut norm_sq = 0.0f64;
            for i in 0..m {
                let e = input[i * k + j];
                norm_sq += e * e;
            }
            sigma[j] = tau[j] / (tau[j] * norm_sq - 1.0);
        }

        // Step 3: K = Q_full @ grad_full^T (row-major [m,m]). grad is [m,k];
        // grad_full zero-extends to [m,m]. K[r,c] = sum_p Qfull[r,p]*grad[c,p]
        // (= sum over the first k columns of grad, the rest are zero).
        let mut k_mat = vec![0.0f64; m * m];
        for r in 0..m {
            for c in 0..m {
                let mut acc = 0.0f64;
                for p in 0..k {
                    acc += q_full[r * m + p] * grad[c * k + p];
                }
                k_mat[r * m + c] = acc;
            }
        }

        // Helper: extract reflector column j of `input` as a full [m] vector.
        let reflector = |j: usize| -> Vec<f64> {
            let mut vj = vec![0.0f64; m];
            for i in 0..m {
                vj[i] = input[i * k + j];
            }
            vj
        };

        // Apply (I - t * vj * vj^T) from the LEFT: K <- K - t*vj*(vj^T K).
        // Mirrors apply_simple_transformation left branch, out-of-place,
        // condition_with_I=true (FunctionsManual.cpp:5524-5525).
        let apply_left = |k_mat: &mut Vec<f64>, vj: &[f64], t: f64| {
            // row vector w = vj^T K  (length m)
            let mut w = vec![0.0f64; m];
            for c in 0..m {
                let mut acc = 0.0f64;
                for i in 0..m {
                    acc += vj[i] * k_mat[i * m + c];
                }
                w[c] = acc;
            }
            for r in 0..m {
                let tv = t * vj[r];
                if tv == 0.0 {
                    continue;
                }
                for c in 0..m {
                    k_mat[r * m + c] -= tv * w[c];
                }
            }
        };

        // Apply (I - t * vj * vj^T) from the RIGHT: K <- K - t*(K vj)*vj^T.
        // Mirrors apply_simple_transformation right branch out-of-place
        // (FunctionsManual.cpp:5538-5539).
        let apply_right = |k_mat: &mut Vec<f64>, vj: &[f64], t: f64| {
            // column vector u = K vj  (length m)
            let mut u = vec![0.0f64; m];
            for r in 0..m {
                let mut acc = 0.0f64;
                for c in 0..m {
                    acc += k_mat[r * m + c] * vj[c];
                }
                u[r] = acc;
            }
            for r in 0..m {
                let tu = t * u[r];
                if tu == 0.0 {
                    continue;
                }
                for c in 0..m {
                    k_mat[r * m + c] -= tu * vj[c];
                }
            }
        };

        // Step 4a: K <- H_0^{-1} @ K  (left reflector with sigma_0).
        let v0 = reflector(0);
        apply_left(&mut k_mat, &v0, sigma[0]);

        // Step 4b: main loop. update_grad on K, then advance K.
        let mut grad_v = vec![0.0f64; m * k];
        let mut grad_tau = vec![0.0f64; k];
        for i in 0..k {
            let vi = reflector(i);
            let ti = tau[i];
            // update_grad (FunctionsManual.cpp:5593-5608), real case:
            //   v   = vi[i:]                       (length m-i)
            //   vHK = v^T @ K[i:, :]               (row vector, length m)
            //   Kv  = K[:, i:] @ v                 (column vector, length m)
            //   v_grad = -ti*vHK^T - ti*Kv         (length m)
            //   tau_grad = -(vHK[i:] @ v)          (scalar)
            // vHK[c] = sum_{r>=i} vi[r] * K[r,c]
            let mut vhk = vec![0.0f64; m];
            for c in 0..m {
                let mut acc = 0.0f64;
                for r in i..m {
                    acc += vi[r] * k_mat[r * m + c];
                }
                vhk[c] = acc;
            }
            // Kv[r] = sum_{c>=i} K[r,c] * vi[c]
            let mut kv = vec![0.0f64; m];
            for r in 0..m {
                let mut acc = 0.0f64;
                for c in i..m {
                    acc += k_mat[r * m + c] * vi[c];
                }
                kv[r] = acc;
            }
            // v_grad[r] = -ti*vhk[r] - ti*kv[r]  (vHK^T identified with vhk).
            for r in 0..m {
                grad_v[r * k + i] = -ti * vhk[r] - ti * kv[r];
            }
            // tau_grad = -(sum_{c>=i} vhk[c] * vi[c]).
            let mut tg = 0.0f64;
            for c in i..m {
                tg += vhk[c] * vi[c];
            }
            grad_tau[i] = -tg;

            // Advance: K <- H_{i+1}^{-1} @ K @ H_i  (FunctionsManual.cpp:5701-5709).
            if i != k - 1 {
                let v_next = reflector(i + 1);
                apply_left(&mut k_mat, &v_next, sigma[i + 1]);
                apply_right(&mut k_mat, &vi, ti);
            }
        }

        // Step 5: grad_V is strictly lower-triangular (forward only touches the
        // strict lower part). Zero the diagonal + upper part.
        for i in 0..m {
            for j in 0..k {
                if i <= j {
                    grad_v[i * k + j] = 0.0;
                }
            }
        }

        let grad_v_out: Vec<T> = grad_v.into_iter().map(|x| T::from(x).unwrap()).collect();
        let grad_tau_out: Vec<T> = grad_tau.into_iter().map(|x| T::from(x).unwrap()).collect();

        let grad_v_tensor = if self.v.requires_grad() {
            Some(Tensor::from_storage(
                TensorStorage::cpu(grad_v_out),
                vec![m, k],
                false,
            )?)
        } else {
            None
        };
        let grad_tau_tensor = if self.tau.requires_grad() {
            Some(Tensor::from_storage(
                TensorStorage::cpu(grad_tau_out),
                vec![k],
                false,
            )?)
        } else {
            None
        };

        Ok(vec![grad_v_tensor, grad_tau_tensor])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.v, &self.tau]
    }

    fn name(&self) -> &'static str {
        "HouseholderProductBackward"
    }
}

/// Differentiable `householder_product`. Attaches `HouseholderProductBackward`
/// (the real reflector-recursion VJP) when grad is needed.
///
/// Forward computed under `no_grad`: `linalg_fwd::householder_product` (the
/// public `crate::linalg::householder_product` forward) delegates back here
/// when grad is enabled, so the guard prevents infinite re-entry. The forward
/// returns the truncated `[m, k]` product (matching torch); the backward
/// reconstructs the full `[m, m]` `Q` from `(V, tau)` under `no_grad` for the
/// `K = Q_full @ grad^T` step.
pub fn householder_product_differentiable<T: Float>(
    v: &Tensor<T>,
    tau: &Tensor<T>,
) -> FerrotorchResult<Tensor<T>> {
    let result = crate::autograd::no_grad::no_grad(|| linalg_fwd::householder_product(v, tau))?;
    if is_grad_enabled() && (v.requires_grad() || tau.requires_grad()) {
        let q_full =
            crate::autograd::no_grad::no_grad(|| linalg_fwd::householder_product_full(v, tau))?;
        let grad_fn = Arc::new(HouseholderProductBackward::new(
            v.clone(),
            tau.clone(),
            q_full,
        ));
        let (storage, shape) = result.into_storage_and_shape()?;
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Ok(result)
    }
}

// ===========================================================================
// Complex linalg helpers (for eig / eigvals backward) — #1345
//
// `eig`/`eigvals` produce COMPLEX eigenvalues/eigenvectors for a non-symmetric
// real `A`. ferrotorch encodes a complex tensor as a trailing-dim-2 real tensor
// `[..., 2]` = `[re, im]` (matching `crate::fft`'s convention). The
// `linalg_eig_backward` VJP (`FunctionsManual.cpp:3820`) is entirely COMPLEX
// arithmetic, so this section provides a small private complex-matrix toolkit
// (matmul, conjugate-transpose, inverse-via-Gaussian-elimination, solve) on the
// flat `[re, im]` layout. This is BOUNDED plumbing for one op family — NOT a
// general complex-dtype subsystem.
//
// A complex `r×c` matrix is held as `Vec<(T, T)>` of length `r*c` in row-major
// order, `(re, im)` per element.
// ===========================================================================

/// Complex scalar add.
#[inline]
fn c_add<T: Float>(a: (T, T), b: (T, T)) -> (T, T) {
    (a.0 + b.0, a.1 + b.1)
}

/// Complex scalar subtract.
#[inline]
fn c_sub<T: Float>(a: (T, T), b: (T, T)) -> (T, T) {
    (a.0 - b.0, a.1 - b.1)
}

/// Complex scalar multiply `(a.re + i a.im)(b.re + i b.im)`.
#[inline]
fn c_mul<T: Float>(a: (T, T), b: (T, T)) -> (T, T) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

/// Complex scalar divide `a / b`.
#[inline]
fn c_div<T: Float>(a: (T, T), b: (T, T)) -> (T, T) {
    let denom = b.0 * b.0 + b.1 * b.1;
    (
        (a.0 * b.0 + a.1 * b.1) / denom,
        (a.1 * b.0 - a.0 * b.1) / denom,
    )
}

/// Complex conjugate.
#[inline]
fn c_conj<T: Float>(a: (T, T)) -> (T, T) {
    (a.0, -a.1)
}

/// Decode a flat `[.., 2]` real tensor slice into a `Vec<(T, T)>` of complex
/// elements. `data` must have even length `2 * count`.
fn complex_from_interleaved<T: Float>(data: &[T]) -> Vec<(T, T)> {
    data.chunks_exact(2).map(|c| (c[0], c[1])).collect()
}

/// Complex matrix multiply `C = A @ B` where `A` is `m×k`, `B` is `k×n`
/// (row-major complex), returning the `m×n` complex product.
fn c_matmul<T: Float>(a: &[(T, T)], b: &[(T, T)], m: usize, k: usize, n: usize) -> Vec<(T, T)> {
    let zero = <T as num_traits::Zero>::zero();
    let mut out = vec![(zero, zero); m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = (zero, zero);
            for p in 0..k {
                acc = c_add(acc, c_mul(a[i * k + p], b[p * n + j]));
            }
            out[i * n + j] = acc;
        }
    }
    out
}

/// Conjugate transpose `A^H` of an `r×c` complex matrix (`A^H` is `c×r`,
/// `A^H[j,i] = conj(A[i,j])`).
fn c_conj_transpose<T: Float>(a: &[(T, T)], r: usize, c: usize) -> Vec<(T, T)> {
    let zero = <T as num_traits::Zero>::zero();
    let mut out = vec![(zero, zero); c * r];
    for i in 0..r {
        for j in 0..c {
            out[j * r + i] = c_conj(a[i * c + j]);
        }
    }
    out
}

/// Invert an `n×n` complex matrix by Gauss-Jordan elimination with partial
/// pivoting (by magnitude). Returns `Err(SingularMatrix)` if no nonzero pivot
/// is found (the eig backward only inverts `V^H` for a diagonalizable `A`, so
/// `V` — hence `V^H` — is invertible by construction).
fn c_inverse<T: Float>(a: &[(T, T)], n: usize) -> FerrotorchResult<Vec<(T, T)>> {
    let zero = <T as num_traits::Zero>::zero();
    let one = <T as num_traits::One>::one();
    // Augmented [A | I], row-major, n×(2n).
    let w = 2 * n;
    let mut aug = vec![(zero, zero); n * w];
    for i in 0..n {
        for j in 0..n {
            aug[i * w + j] = a[i * n + j];
        }
        aug[i * w + n + i] = (one, zero);
    }
    for col in 0..n {
        // Partial pivot: row with largest |aug[row, col]|.
        let mut best_row = col;
        let mut best_mag = zero;
        for row in col..n {
            let e = aug[row * w + col];
            let mag = e.0 * e.0 + e.1 * e.1;
            if mag > best_mag {
                best_mag = mag;
                best_row = row;
            }
        }
        if best_mag == zero {
            return Err(FerrotorchError::InvalidArgument {
                message: "complex inverse: singular matrix (defective eig?)".into(),
            });
        }
        if best_row != col {
            for j in 0..w {
                aug.swap(col * w + j, best_row * w + j);
            }
        }
        // Normalize pivot row so the pivot becomes 1.
        let pivot = aug[col * w + col];
        for j in 0..w {
            aug[col * w + j] = c_div(aug[col * w + j], pivot);
        }
        // Eliminate the column in all other rows.
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row * w + col];
            if factor.0 == zero && factor.1 == zero {
                continue;
            }
            for j in 0..w {
                let sub = c_mul(factor, aug[col * w + j]);
                aug[row * w + j] = c_sub(aug[row * w + j], sub);
            }
        }
    }
    // Extract the right half (the inverse).
    let mut inv = vec![(zero, zero); n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = aug[i * w + n + j];
        }
    }
    Ok(inv)
}

/// Complex solve `X = M^{-1} @ B` for `M` `n×n` and `B` `n×c`, via explicit
/// complex inverse (small `n`; matches torch's `at::linalg_solve(V^H, ...)`).
fn c_solve<T: Float>(
    m: &[(T, T)],
    b: &[(T, T)],
    n: usize,
    c: usize,
) -> FerrotorchResult<Vec<(T, T)>> {
    let minv = c_inverse(m, n)?;
    Ok(c_matmul(&minv, b, n, n, c))
}

/// `Econj[i,j] = conj(L_j) - conj(L_i)` off-diagonal, `1` on the diagonal — the
/// eigenvalue-gap denominator of the non-Hermitian eig VJP
/// (`FunctionsManual.cpp:3893-3898`). `lc` is the length-`n` complex eigenvalue
/// vector.
fn c_econj_gap<T: Float>(lc: &[(T, T)], n: usize) -> Vec<(T, T)> {
    let zero = <T as num_traits::Zero>::zero();
    let one = <T as num_traits::One>::one();
    let mut e = vec![(one, zero); n * n];
    for i in 0..n {
        for j in 0..n {
            if i != j {
                e[i * n + j] = c_sub(c_conj(lc[j]), c_conj(lc[i]));
            }
        }
    }
    e
}

/// Take the REAL part of a complex matrix into a flat row-major real `Vec<T>`
/// (the `handle_r_to_c` step: for a real input `A`, `at::real(grad_A)` —
/// `FunctionsManual.cpp` `handle_r_to_c`, registered for `linalg_eig` at
/// `tools/autograd/derivatives.yaml:1740`).
fn complex_real_part<T: Float>(a: &[(T, T)]) -> Vec<T> {
    a.iter().map(|&(re, _im)| re).collect()
}

// ---------------------------------------------------------------------------
// EigvalsBackward — w = eigvals(A)  (non-symmetric A, eigenvalues only, COMPLEX)
// ---------------------------------------------------------------------------

/// Backward for `w = eigvals(A)` (non-symmetric A, complex eigenvalues only).
///
/// Mirrors the `linalg.eigvals` shortcut of `linalg_eig_backward`
/// (`torch/csrc/autograd/FunctionsManual.cpp:3857-3862`, the `!gV.defined()`,
/// non-Hermitian branch):
///
/// ```text
/// gA = linalg_solve(V^H, gL.unsqueeze(-1) * V^H)
/// ```
///
/// where `gL.unsqueeze(-1) * V^H == diag(gL) @ V^H` (broadcasting `gL` down the
/// rows), so `gA = V^{-H} @ diag(gL) @ V^H`. The complex cotangent `gL` is
/// reconstructed from the `[n,2]` real cotangent that flows into this node as
/// `gL[k] = grad_re[k] + i * grad_im[k]` — torch's conjugate-Wirtinger
/// convention for a real loss of a complex output (verified against LIVE torch:
/// `L.grad == cr + i*ci` for a loss `sum(re*cr) + sum(im*ci)`). Because `A` is
/// REAL, the returned gradient is `at::real(gA)`
/// (`handle_r_to_c`, `derivatives.yaml:1740`).
///
/// EXACT for DIAGONALIZABLE `A` (distinct eigenvalues ⇒ `V` invertible). On a
/// defective / repeated-eigenvalue input `V` is singular and `c_inverse`
/// returns `SingularMatrix` — torch likewise has no defined gradient there (it
/// divides through a degenerate `V`).
#[derive(Debug)]
pub struct EigvalsBackward<T: Float> {
    input: Tensor<T>,
    /// Eigenvector matrix `V`, encoded `[n,n,2]` (complex, row-major).
    v: Tensor<T>,
}

impl<T: Float> EigvalsBackward<T> {
    /// `gA = real(V^{-H} @ diag(gL) @ V^H)` from the `[n,2]` real cotangent.
    fn grad_a(&self, grad_output: &Tensor<T>, n: usize) -> FerrotorchResult<Tensor<T>> {
        let vc = complex_from_interleaved(self.v.data()?); // [n*n] complex
        let gl = complex_from_interleaved(grad_output.data()?); // [n] complex
        // V^H  (n×n).
        let vh = c_conj_transpose(&vc, n, n);
        // diag(gL) @ V^H:  scale row i of V^H by gL[i].
        let mut rhs = vec![
            (
                <T as num_traits::Zero>::zero(),
                <T as num_traits::Zero>::zero()
            );
            n * n
        ];
        for i in 0..n {
            for j in 0..n {
                rhs[i * n + j] = c_mul(gl[i], vh[i * n + j]);
            }
        }
        // gA = solve(V^H, rhs) = V^{-H} @ diag(gL) @ V^H.
        let ga = c_solve(&vh, &rhs, n, n)?;
        from_cpu(complex_real_part(&ga), vec![n, n])
    }
}

impl<T: Float> GradFn<T> for EigvalsBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let n = self.v.shape()[0];
        Ok(vec![Some(self.grad_a(grad_output, n)?)])
    }
    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }
    fn name(&self) -> &'static str {
        "EigvalsBackward"
    }
}

/// Differentiable `eigvals` (non-symmetric, diagonalizable). Attaches
/// `EigvalsBackward` when grad is needed. Forward computed under `no_grad`
/// (`linalg_fwd::eigvals` delegates back here when grad is enabled, so the guard
/// prevents infinite re-entry); the eigenvectors `V` the VJP needs come from
/// `linalg_fwd::eig` (also under `no_grad`).
pub fn eigvals_differentiable<T: Float>(a: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let w = crate::autograd::no_grad::no_grad(|| linalg_fwd::eigvals(a))?;
    if is_grad_enabled() && a.requires_grad() {
        let (_w2, v) = crate::autograd::no_grad::no_grad(|| linalg_fwd::eig(a))?;
        let grad_fn = Arc::new(EigvalsBackward {
            input: a.clone(),
            v,
        });
        let (storage, shape) = w.into_storage_and_shape()?;
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Ok(w)
    }
}

// ---------------------------------------------------------------------------
// EigBackward — (w, V) = eig(A)  (non-symmetric A, eigenvalues + eigenvectors)
// ---------------------------------------------------------------------------

/// Shared non-Hermitian eig VJP, split across two single-output nodes
/// (`EigBackwardW` on the eigenvalues, `EigBackwardV` on the eigenvectors).
///
/// Mirrors `linalg_eig_backward` (`torch/csrc/autograd/FunctionsManual.cpp:3820`,
/// the non-Hermitian general branch). For `A = V diag(L) V^{-1}`:
///
/// ```text
/// VhgV = V^H @ gV
/// VhgV <- VhgV - V^H @ (V * real(diag(VhgV)))        // unit-norm tangent proj
/// Econj[i,j] = conj(L_j) - conj(L_i) (i != j), 1 on the diagonal
/// ret = VhgV / Econj                                  // elementwise
/// ret.diagonal = gL                                   // eigenvalue contrib
/// gA = linalg_solve(V^H, ret @ V^H)                   // conjugate by V^{-H}
/// ```
///
/// (`FunctionsManual.cpp:3864-3920`). The cotangents `gL` (`[n,2]`) and `gV`
/// (`[n,n,2]`) are reconstructed as `re + i*im`. Because `A` is REAL the
/// returned gradient is `at::real(gA)` (`handle_r_to_c`,
/// `derivatives.yaml:1740`).
///
/// **Eigenvector gauge (R-DEV-1):** eig eigenvectors are scale-free — `V` and
/// `V diag(c)` for any nonzero complex `c` are both valid. torch normalizes to
/// unit-norm columns and the `-V^H V real(diag(VhgV))` projection handles the
/// norm constraint, but the PHASE `V_j -> V_j e^{i phi}` is a genuine gauge
/// freedom: torch asserts the loss is phase-invariant
/// (`FunctionsManual.cpp:3867-3879`, `imag(diag(V^H gV)) ≈ 0`). A well-posed
/// loss must therefore be phase-invariant (e.g. `sum(|V_ij|^2 * M)` — `|.|^2` is
/// unchanged by a per-column phase); for such losses `A.grad` matches torch even
/// though ferray's faer column gauge differs from LAPACK's.
///
/// The two outputs `(w, V)` are jointly linear in `gA`, so the engine
/// accumulates the `EigBackwardW` (`gV=0`) and `EigBackwardV` (`gL=0`) partials
/// into `A.grad` — the same split-node strategy `eigh` / `svd` / `qr` use.
///
/// EXACT for DIAGONALIZABLE `A` (distinct eigenvalues). On a defective input `V`
/// is singular (`c_inverse` ⇒ `SingularMatrix`) and on a repeated eigenvalue the
/// `Econj` off-diagonal `1/(conj(L_j)-conj(L_i))` diverges exactly as torch's
/// does (torch does not special-case degeneracy).
#[derive(Debug)]
struct EigBackwardShared<T: Float> {
    /// Eigenvalues `L`, encoded `[n,2]` (complex).
    l: Tensor<T>,
    /// Eigenvector matrix `V`, encoded `[n,n,2]` (complex, row-major).
    v: Tensor<T>,
}

impl<T: Float> EigBackwardShared<T> {
    fn n(&self) -> usize {
        self.v.shape()[0]
    }

    /// `gA = real(solve(V^H, ret @ V^H))` for a `n×n` complex middle factor
    /// `ret` (`FunctionsManual.cpp:3919`, non-Hermitian conjugation by `V^{-H}`).
    fn conjugate(&self, ret: &[(T, T)], n: usize) -> FerrotorchResult<Tensor<T>> {
        let vc = complex_from_interleaved(self.v.data()?);
        let vh = c_conj_transpose(&vc, n, n);
        let rhs = c_matmul(ret, &vh, n, n, n); // ret @ V^H
        let ga = c_solve(&vh, &rhs, n, n)?; // V^{-H} @ (ret @ V^H)
        from_cpu(complex_real_part(&ga), vec![n, n])
    }

    /// `gL`-only contribution: `ret = diag(gL)`, then conjugate.
    fn grad_a_from_gl(&self, gl: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let n = self.n();
        let zero = <T as num_traits::Zero>::zero();
        let glc = complex_from_interleaved(gl.data()?);
        let mut ret = vec![(zero, zero); n * n];
        for i in 0..n {
            ret[i * n + i] = glc[i];
        }
        self.conjugate(&ret, n)
    }

    /// `gV`-only contribution: build `VhgV`, GUARD that the loss is
    /// phase-invariant (`imag(diag(VhgV)) ≈ 0`, `FunctionsManual.cpp:3867-3879`),
    /// project onto the unit-norm tangent space, divide by `Econj`, then conjugate
    /// (`FunctionsManual.cpp:3864-3919`, `gL` undefined).
    fn grad_a_from_gv(&self, gv: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let n = self.n();
        let zero = <T as num_traits::Zero>::zero();
        let vc = complex_from_interleaved(self.v.data()?); // [n*n] complex
        let gvc = complex_from_interleaved(gv.data()?); // [n*n] complex
        let lc = complex_from_interleaved(self.l.data()?); // [n] complex

        // VhgV = V^H @ gV  (n×n).
        let vh = c_conj_transpose(&vc, n, n);
        let mut vhgv = c_matmul(&vh, &gvc, n, n, n);

        // Phase-invariance guard (FunctionsManual.cpp:3867-3879). Non-Hermitian
        // eigenvectors are defined only up to a per-column phase
        // `V_j -> V_j e^{i phi}`, so torch RAISES on a loss that is NOT
        // phase-invariant: it takes `diag_VhgV = diag(V^H gV)` (right after the
        // matmul, BEFORE the unit-norm projection) and checks that its imaginary
        // part is ~0 via `allclose(imag(diag_VhgV), zeros, rtol=1e-2, atol=1e-2)`.
        // For a real-V decomposition every imag(diag) is 0, so the guard never
        // fires for phase-invariant losses; it fires only when the loss reads the
        // gauge-dependent phase (e.g. `sum(V.real)`), where torch errors and we
        // must too rather than return a gauge-dependent (ill-defined) gradient.
        // allclose vs zeros: `|imag(diag)_i| <= atol + rtol*|0| = 1e-2`.
        let atol = T::from(1e-2).unwrap();
        let phase_tol_exceeded = (0..n).any(|i| vhgv[i * n + i].1.abs() > atol);
        if phase_tol_exceeded {
            return Err(FerrotorchError::InvalidArgument {
                message: "linalg_eig_backward: The eigenvectors in the complex \
                          case are specified up to multiplication by e^{i phi}. \
                          The specified loss function depends on this quantity, \
                          so it is ill-defined."
                    .to_string(),
            });
        }

        // Projection onto the tangent space at V^H V of unit-norm columns:
        //   VhgV <- VhgV - V^H @ (V * real(diag(VhgV)))
        // (FunctionsManual.cpp:3887-3889). `V * real(diag(VhgV))` scales column
        // j of V by the REAL scalar real(VhgV[j,j]).
        let mut v_scaled = vec![(zero, zero); n * n];
        for i in 0..n {
            for j in 0..n {
                let rj = vhgv[j * n + j].0; // real(diag(VhgV))[j]
                v_scaled[i * n + j] = (vc[i * n + j].0 * rj, vc[i * n + j].1 * rj);
            }
        }
        let correction = c_matmul(&vh, &v_scaled, n, n, n); // V^H @ (V * real(diag))
        for idx in 0..n * n {
            vhgv[idx] = c_sub(vhgv[idx], correction[idx]);
        }

        // ret = VhgV / Econj  (elementwise complex divide).
        let e = c_econj_gap(&lc, n);
        let mut ret = vec![(zero, zero); n * n];
        for idx in 0..n * n {
            ret[idx] = c_div(vhgv[idx], e[idx]);
        }
        // (gL undefined here — diagonal stays as the divide result, which for
        // the gV-only partial is `VhgV[i,i]/1 = VhgV[i,i]`. torch overwrites the
        // diagonal with gL only when gL is defined; for the split gV-only node
        // gL is zero so the diagonal carries the gV contribution as torch's
        // formula does when gL is the zero tensor — `ret.diagonal.copy_(0)`
        // would zero it, but torch only copies gL when `gL.defined()`, leaving
        // the divided diagonal in place. We mirror torch: leave it.)
        self.conjugate(&ret, n)
    }
}

/// `gL`-only eig backward node, attached to the eigenvalues output.
#[derive(Debug)]
struct EigBackwardW<T: Float> {
    input: Tensor<T>,
    shared: EigBackwardShared<T>,
}

impl<T: Float> GradFn<T> for EigBackwardW<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        Ok(vec![Some(self.shared.grad_a_from_gl(grad_output)?)])
    }
    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }
    fn name(&self) -> &'static str {
        "EigBackward"
    }
}

/// `gV`-only eig backward node, attached to the eigenvectors output.
#[derive(Debug)]
struct EigBackwardV<T: Float> {
    input: Tensor<T>,
    shared: EigBackwardShared<T>,
}

impl<T: Float> GradFn<T> for EigBackwardV<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        Ok(vec![Some(self.shared.grad_a_from_gv(grad_output)?)])
    }
    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }
    fn name(&self) -> &'static str {
        "EigBackward"
    }
}

/// Differentiable `eig` (non-symmetric, diagonalizable). Attaches the split
/// `EigBackwardW` / `EigBackwardV` nodes (whose `A.grad` contributions the
/// autograd engine accumulates) when grad is needed. Forward computed under
/// `no_grad` (re-entry guard).
///
/// Handles grad through `L` only (`gV` zero), `V` only (`gL` zero), or both —
/// the split-node strategy makes each output's partial independent. The complex
/// arithmetic runs on the `[n,2]`/`[n,n,2]` real encodings; the returned
/// `A.grad` is the REAL part (real `A`, per `handle_r_to_c`,
/// `derivatives.yaml:1740`). See `EigBackwardShared` for the gauge caveat.
pub fn eig_differentiable<T: Float>(a: &Tensor<T>) -> FerrotorchResult<(Tensor<T>, Tensor<T>)> {
    let (w, v) = crate::autograd::no_grad::no_grad(|| linalg_fwd::eig(a))?;
    let needs_grad = is_grad_enabled() && a.requires_grad();
    if !needs_grad {
        return Ok((w, v));
    }
    let w_node = Arc::new(EigBackwardW {
        input: a.clone(),
        shared: EigBackwardShared {
            l: w.clone(),
            v: v.clone(),
        },
    });
    let v_node = Arc::new(EigBackwardV {
        input: a.clone(),
        shared: EigBackwardShared {
            l: w.clone(),
            v: v.clone(),
        },
    });
    let (w_storage, w_shape) = w.into_storage_and_shape()?;
    let (v_storage, v_shape) = v.into_storage_and_shape()?;
    let w = Tensor::from_operation(w_storage, w_shape, w_node)?;
    let v = Tensor::from_operation(v_storage, v_shape, v_node)?;
    Ok((w, v))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::TensorStorage;

    /// Helper: create a leaf tensor with requires_grad.
    fn leaf(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
    }

    /// Helper: create a leaf tensor without requires_grad.
    fn no_grad_leaf(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
    }

    /// Assert two slices are element-wise close.
    fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "length mismatch: {} vs {}",
            actual.len(),
            expected.len()
        );
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < tol,
                "index {i}: {a} vs {e} (diff {})",
                (a - e).abs()
            );
        }
    }

    // -----------------------------------------------------------------------
    // mm backward
    // -----------------------------------------------------------------------

    #[test]
    fn test_mm_backward_both_grads() {
        // A = [[1, 2], [3, 4]]  (2x2)
        // B = [[5, 6], [7, 8]]  (2x2)
        // C = A @ B = [[19, 22], [43, 50]]
        //
        // To get a scalar loss: L = sum(C) = 19 + 22 + 43 + 50 = 134
        // dL/dC = [[1, 1], [1, 1]]
        //
        // dL/dA = dL/dC @ B^T = [[1,1],[1,1]] @ [[5,7],[6,8]] = [[11,15],[11,15]]
        // dL/dB = A^T @ dL/dC = [[1,3],[2,4]] @ [[1,1],[1,1]] = [[4,4],[6,6]]
        let a = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = leaf(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);

        let c = mm_differentiable(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);

        // Sum C to get a scalar for backward.
        let c_data = c.data().unwrap();
        let loss_val: f32 = c_data.iter().sum();

        // Build a SumBackward manually: dL/dC = ones_like(C).
        #[derive(Debug)]
        struct SumBackward<T: Float> {
            input: Tensor<T>,
        }
        impl<T: Float> GradFn<T> for SumBackward<T> {
            fn backward(
                &self,
                _grad_output: &Tensor<T>,
            ) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
                let ones = vec![<T as num_traits::One>::one(); self.input.numel()];
                let g = Tensor::from_storage(
                    TensorStorage::cpu(ones),
                    self.input.shape().to_vec(),
                    false,
                )?;
                Ok(vec![Some(g)])
            }
            fn inputs(&self) -> Vec<&Tensor<T>> {
                vec![&self.input]
            }
            fn name(&self) -> &'static str {
                "SumBackward"
            }
        }

        let loss = Tensor::from_operation(
            TensorStorage::cpu(vec![loss_val]),
            vec![],
            Arc::new(SumBackward { input: c }),
        )
        .unwrap();

        loss.backward().unwrap();

        let a_grad = a.grad().unwrap().expect("a should have grad");
        let b_grad = b.grad().unwrap().expect("b should have grad");

        assert_eq!(a_grad.shape(), &[2, 2]);
        assert_eq!(b_grad.shape(), &[2, 2]);

        // dL/dA = [[11, 15], [11, 15]]
        assert_close(a_grad.data().unwrap(), &[11.0, 15.0, 11.0, 15.0], 1e-5);
        // dL/dB = [[4, 4], [6, 6]]
        assert_close(b_grad.data().unwrap(), &[4.0, 4.0, 6.0, 6.0], 1e-5);
    }

    #[test]
    fn test_mm_backward_one_requires_grad() {
        // Only A requires grad, B does not.
        let a = leaf(&[1.0, 0.0, 0.0, 1.0], &[2, 2]); // identity
        let b = no_grad_leaf(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);

        let c = mm_differentiable(&a, &b).unwrap();
        assert!(c.grad_fn().is_some());

        // grad_output = ones(2,2)
        let grad_out = no_grad_leaf(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
        let grads = c.grad_fn().unwrap().backward(&grad_out).unwrap();

        // grad_a should be Some, grad_b should be None
        assert!(grads[0].is_some());
        assert!(grads[1].is_none());

        // dA = grad_C @ B^T = [[1,1],[1,1]] @ [[2,4],[3,5]] = [[5,9],[5,9]]
        let ga = grads[0].as_ref().unwrap();
        assert_close(ga.data().unwrap(), &[5.0, 9.0, 5.0, 9.0], 1e-5);
    }

    // -----------------------------------------------------------------------
    // dot backward
    // -----------------------------------------------------------------------

    #[test]
    fn test_dot_backward() {
        // a = [1, 2, 3], b = [4, 5, 6]
        // s = dot(a, b) = 4 + 10 + 18 = 32
        // ds/da = b = [4, 5, 6]
        // ds/db = a = [1, 2, 3]
        let a = leaf(&[1.0, 2.0, 3.0], &[3]);
        let b = leaf(&[4.0, 5.0, 6.0], &[3]);

        let s = dot_differentiable(&a, &b).unwrap();
        assert!(s.is_scalar());
        assert!((s.item().unwrap() - 32.0).abs() < 1e-5);

        s.backward().unwrap();

        let a_grad = a.grad().unwrap().expect("a should have grad");
        let b_grad = b.grad().unwrap().expect("b should have grad");

        assert_eq!(a_grad.shape(), &[3]);
        assert_eq!(b_grad.shape(), &[3]);
        assert_close(a_grad.data().unwrap(), &[4.0, 5.0, 6.0], 1e-5);
        assert_close(b_grad.data().unwrap(), &[1.0, 2.0, 3.0], 1e-5);
    }

    #[test]
    fn test_dot_backward_one_requires_grad() {
        let a = leaf(&[2.0, 3.0], &[2]);
        let b = no_grad_leaf(&[4.0, 5.0], &[2]);

        let s = dot_differentiable(&a, &b).unwrap();
        let grad_out = no_grad_leaf(&[1.0], &[]);
        let grads = s.grad_fn().unwrap().backward(&grad_out).unwrap();

        assert!(grads[0].is_some());
        assert!(grads[1].is_none());
        assert_close(
            grads[0].as_ref().unwrap().data().unwrap(),
            &[4.0, 5.0],
            1e-5,
        );
    }

    // -----------------------------------------------------------------------
    // mv backward
    // -----------------------------------------------------------------------

    #[test]
    fn test_mv_backward() {
        // A = [[1, 2], [3, 4]]  (2x2)
        // x = [5, 6]            (2,)
        // y = A @ x = [17, 39]
        //
        // Use L = sum(y) = 56, so dL/dy = [1, 1].
        // dA = outer([1,1], [5,6]) = [[5,6],[5,6]]
        // dx = A^T @ [1,1] = [[1,3],[2,4]] @ [1,1] = [4, 6]
        let a = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let x = leaf(&[5.0, 6.0], &[2]);

        let y = mv_differentiable(&a, &x).unwrap();
        assert_eq!(y.shape(), &[2]);

        // Build sum for scalar loss.
        let y_data = y.data().unwrap();
        let loss_val: f32 = y_data.iter().sum();

        #[derive(Debug)]
        struct SumBackward1D<T: Float> {
            input: Tensor<T>,
        }
        impl<T: Float> GradFn<T> for SumBackward1D<T> {
            fn backward(
                &self,
                _grad_output: &Tensor<T>,
            ) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
                let ones = vec![<T as num_traits::One>::one(); self.input.numel()];
                let g = Tensor::from_storage(
                    TensorStorage::cpu(ones),
                    self.input.shape().to_vec(),
                    false,
                )?;
                Ok(vec![Some(g)])
            }
            fn inputs(&self) -> Vec<&Tensor<T>> {
                vec![&self.input]
            }
            fn name(&self) -> &'static str {
                "SumBackward"
            }
        }

        let loss = Tensor::from_operation(
            TensorStorage::cpu(vec![loss_val]),
            vec![],
            Arc::new(SumBackward1D { input: y }),
        )
        .unwrap();

        loss.backward().unwrap();

        let a_grad = a.grad().unwrap().expect("a should have grad");
        let x_grad = x.grad().unwrap().expect("x should have grad");

        assert_eq!(a_grad.shape(), &[2, 2]);
        assert_eq!(x_grad.shape(), &[2]);

        // dA = outer([1,1], [5,6]) = [[5,6],[5,6]]
        assert_close(a_grad.data().unwrap(), &[5.0, 6.0, 5.0, 6.0], 1e-5);
        // dx = A^T @ [1,1] = [4, 6]
        assert_close(x_grad.data().unwrap(), &[4.0, 6.0], 1e-5);
    }

    // -----------------------------------------------------------------------
    // matmul backward (dispatch)
    // -----------------------------------------------------------------------

    #[test]
    fn test_matmul_backward_dispatches_to_dot() {
        // matmul(1D, 1D) should use DotBackward path.
        let a = leaf(&[1.0, 2.0], &[2]);
        let b = leaf(&[3.0, 4.0], &[2]);

        let s = matmul_differentiable(&a, &b).unwrap();
        assert!(s.is_scalar());
        assert!((s.item().unwrap() - 11.0).abs() < 1e-5);

        s.backward().unwrap();

        let a_grad = a.grad().unwrap().unwrap();
        let b_grad = b.grad().unwrap().unwrap();
        assert_close(a_grad.data().unwrap(), &[3.0, 4.0], 1e-5);
        assert_close(b_grad.data().unwrap(), &[1.0, 2.0], 1e-5);
    }

    #[test]
    fn test_matmul_backward_dispatches_to_mm() {
        let a = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = leaf(&[1.0, 0.0, 0.0, 1.0], &[2, 2]); // identity

        let c = matmul_differentiable(&a, &b).unwrap();

        // grad_output = ones
        let grad_out = no_grad_leaf(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
        let grads = c.grad_fn().unwrap().backward(&grad_out).unwrap();

        // dA = ones @ I^T = ones
        assert_close(
            grads[0].as_ref().unwrap().data().unwrap(),
            &[1.0, 1.0, 1.0, 1.0],
            1e-5,
        );
        // dB = A^T @ ones = [[1,3],[2,4]] @ [[1,1],[1,1]] = [[4,4],[6,6]]
        assert_close(
            grads[1].as_ref().unwrap().data().unwrap(),
            &[4.0, 4.0, 6.0, 6.0],
            1e-5,
        );
    }

    #[test]
    fn test_matmul_backward_vm() {
        // a = [1, 2] (K=2), B = [[3, 4, 5], [6, 7, 8]] (2x3)
        // y = a @ B = [1*3+2*6, 1*4+2*7, 1*5+2*8] = [15, 18, 21]
        //
        // dL/dy = [1, 1, 1]  (from sum)
        // da = B @ dL/dy = [[3,4,5],[6,7,8]] @ [1,1,1] = [12, 21]
        // dB = outer(a, dL/dy) = [[1,1,1],[2,2,2]]
        let a = leaf(&[1.0, 2.0], &[2]);
        let b = leaf(&[3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 3]);

        let y = matmul_differentiable(&a, &b).unwrap();
        assert_eq!(y.shape(), &[3]);

        // Build sum for scalar.
        let y_data = y.data().unwrap();
        let loss_val: f32 = y_data.iter().sum();

        #[derive(Debug)]
        struct SumBackwardVec<T: Float> {
            input: Tensor<T>,
        }
        impl<T: Float> GradFn<T> for SumBackwardVec<T> {
            fn backward(
                &self,
                _grad_output: &Tensor<T>,
            ) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
                let ones = vec![<T as num_traits::One>::one(); self.input.numel()];
                let g = Tensor::from_storage(
                    TensorStorage::cpu(ones),
                    self.input.shape().to_vec(),
                    false,
                )?;
                Ok(vec![Some(g)])
            }
            fn inputs(&self) -> Vec<&Tensor<T>> {
                vec![&self.input]
            }
            fn name(&self) -> &'static str {
                "SumBackward"
            }
        }

        let loss = Tensor::from_operation(
            TensorStorage::cpu(vec![loss_val]),
            vec![],
            Arc::new(SumBackwardVec { input: y }),
        )
        .unwrap();

        loss.backward().unwrap();

        let a_grad = a.grad().unwrap().expect("a should have grad");
        let b_grad = b.grad().unwrap().expect("b should have grad");

        assert_eq!(a_grad.shape(), &[2]);
        assert_eq!(b_grad.shape(), &[2, 3]);

        // da = B @ [1,1,1] = [12, 21]
        assert_close(a_grad.data().unwrap(), &[12.0, 21.0], 1e-5);
        // dB = outer([1,2], [1,1,1]) = [[1,1,1],[2,2,2]]
        assert_close(
            b_grad.data().unwrap(),
            &[1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            1e-5,
        );
    }

    // -----------------------------------------------------------------------
    // bmm backward
    // -----------------------------------------------------------------------

    #[test]
    fn test_bmm_backward_both_grads() {
        // Batch 0: A0=[[1,2],[3,4]], B0=[[5,6],[7,8]]
        //   C0 = [[19,22],[43,50]]
        // Batch 1: A1=[[1,0],[0,1]] (identity), B1=[[9,10],[11,12]]
        //   C1 = [[9,10],[11,12]]
        //
        // L = sum(C), dL/dC = ones(2,2,2)
        //
        // dA0 = ones(2,2) @ B0^T = [[1,1],[1,1]] @ [[5,7],[6,8]] = [[11,15],[11,15]]
        // dA1 = ones(2,2) @ B1^T = [[1,1],[1,1]] @ [[9,11],[10,12]] = [[19,23],[19,23]]
        //
        // dB0 = A0^T @ ones(2,2) = [[1,3],[2,4]] @ [[1,1],[1,1]] = [[4,4],[6,6]]
        // dB1 = A1^T @ ones(2,2) = [[1,0],[0,1]] @ [[1,1],[1,1]] = [[1,1],[1,1]]
        #[rustfmt::skip]
        let a = leaf(&[
            1.0, 2.0, 3.0, 4.0,   // batch 0
            1.0, 0.0, 0.0, 1.0,   // batch 1
        ], &[2, 2, 2]);
        #[rustfmt::skip]
        let b = leaf(&[
            5.0, 6.0, 7.0, 8.0,    // batch 0
            9.0, 10.0, 11.0, 12.0, // batch 1
        ], &[2, 2, 2]);

        let c = bmm_differentiable(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2, 2]);

        // Sum to scalar for backward.
        let c_data = c.data().unwrap();
        let loss_val: f32 = c_data.iter().sum();

        #[derive(Debug)]
        struct SumBackward3D<T: Float> {
            input: Tensor<T>,
        }
        impl<T: Float> GradFn<T> for SumBackward3D<T> {
            fn backward(
                &self,
                _grad_output: &Tensor<T>,
            ) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
                let ones = vec![<T as num_traits::One>::one(); self.input.numel()];
                let g = Tensor::from_storage(
                    TensorStorage::cpu(ones),
                    self.input.shape().to_vec(),
                    false,
                )?;
                Ok(vec![Some(g)])
            }
            fn inputs(&self) -> Vec<&Tensor<T>> {
                vec![&self.input]
            }
            fn name(&self) -> &'static str {
                "SumBackward"
            }
        }

        let loss = Tensor::from_operation(
            TensorStorage::cpu(vec![loss_val]),
            vec![],
            Arc::new(SumBackward3D { input: c }),
        )
        .unwrap();

        loss.backward().unwrap();

        let a_grad = a.grad().unwrap().expect("a should have grad");
        let b_grad = b.grad().unwrap().expect("b should have grad");

        assert_eq!(a_grad.shape(), &[2, 2, 2]);
        assert_eq!(b_grad.shape(), &[2, 2, 2]);

        #[rustfmt::skip]
        let expected_da: &[f32] = &[
            11.0, 15.0, 11.0, 15.0,  // batch 0
            19.0, 23.0, 19.0, 23.0,  // batch 1
        ];
        #[rustfmt::skip]
        let expected_db: &[f32] = &[
            4.0, 4.0, 6.0, 6.0,  // batch 0
            1.0, 1.0, 1.0, 1.0,  // batch 1
        ];
        assert_close(a_grad.data().unwrap(), expected_da, 1e-5);
        assert_close(b_grad.data().unwrap(), expected_db, 1e-5);
    }

    #[test]
    fn test_bmm_backward_batch_size_1() {
        // Single batch: should match mm backward exactly.
        // A=[[1,2],[3,4]], B=[[5,6],[7,8]]
        // dL/dC = ones(1,2,2)
        // dA = ones @ B^T = [[11,15],[11,15]]
        // dB = A^T @ ones = [[4,4],[6,6]]
        let a = leaf(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
        let b = leaf(&[5.0, 6.0, 7.0, 8.0], &[1, 2, 2]);

        let c = bmm_differentiable(&a, &b).unwrap();

        let grad_out = no_grad_leaf(&[1.0, 1.0, 1.0, 1.0], &[1, 2, 2]);
        let grads = c.grad_fn().unwrap().backward(&grad_out).unwrap();

        assert!(grads[0].is_some());
        assert!(grads[1].is_some());

        let ga = grads[0].as_ref().unwrap();
        let gb = grads[1].as_ref().unwrap();
        assert_eq!(ga.shape(), &[1, 2, 2]);
        assert_eq!(gb.shape(), &[1, 2, 2]);

        assert_close(ga.data().unwrap(), &[11.0, 15.0, 11.0, 15.0], 1e-5);
        assert_close(gb.data().unwrap(), &[4.0, 4.0, 6.0, 6.0], 1e-5);
    }

    #[test]
    fn test_bmm_backward_one_requires_grad() {
        // Only A requires grad.
        let a = leaf(&[1.0, 0.0, 0.0, 1.0], &[1, 2, 2]);
        let b = no_grad_leaf(&[2.0, 3.0, 4.0, 5.0], &[1, 2, 2]);

        let c = bmm_differentiable(&a, &b).unwrap();
        assert!(c.grad_fn().is_some());

        let grad_out = no_grad_leaf(&[1.0, 1.0, 1.0, 1.0], &[1, 2, 2]);
        let grads = c.grad_fn().unwrap().backward(&grad_out).unwrap();

        assert!(grads[0].is_some());
        assert!(grads[1].is_none());

        // dA = ones @ B^T = [[1,1],[1,1]] @ [[2,4],[3,5]] = [[5,9],[5,9]]
        let ga = grads[0].as_ref().unwrap();
        assert_close(ga.data().unwrap(), &[5.0, 9.0, 5.0, 9.0], 1e-5);
    }

    // -----------------------------------------------------------------------
    // no_grad disables backward tracking
    // -----------------------------------------------------------------------

    #[test]
    fn test_no_grad_skips_backward() {
        let a = leaf(&[1.0, 2.0, 3.0], &[3]);
        let b = leaf(&[4.0, 5.0, 6.0], &[3]);

        let s = crate::autograd::no_grad::no_grad(|| dot_differentiable(&a, &b).unwrap());

        // Should have no grad_fn because we were inside no_grad.
        assert!(s.grad_fn().is_none());
    }

    // -----------------------------------------------------------------------
    // broadcast matmul backward
    // -----------------------------------------------------------------------

    #[test]
    fn test_matmul_backward_3d_3d_numerical() {
        // Numerical gradient check for (2,2,3) @ (2,3,2).
        let eps = 1e-3f32;

        let a_data: Vec<f32> = (0..12).map(|i| (i as f32) * 0.1 + 0.1).collect();
        let b_data: Vec<f32> = (0..12).map(|i| (i as f32) * 0.1 + 0.5).collect();

        // Forward + backward.
        let a = leaf(&a_data, &[2, 2, 3]);
        let b = leaf(&b_data, &[2, 3, 2]);
        let c = matmul_differentiable(&a, &b).unwrap();
        let loss = crate::grad_fns::reduction::sum(&c).unwrap();
        loss.backward().unwrap();

        let analytic_a = a.grad().unwrap().unwrap().data().unwrap().to_vec();
        let analytic_b = b.grad().unwrap().unwrap().data().unwrap().to_vec();

        // Check each element of A numerically.
        for idx in 0..a_data.len() {
            let mut a_plus = a_data.clone();
            a_plus[idx] += eps;
            let mut a_minus = a_data.clone();
            a_minus[idx] -= eps;

            let loss_plus = crate::autograd::no_grad::no_grad(|| {
                let ap = no_grad_leaf(&a_plus, &[2, 2, 3]);
                let bp = no_grad_leaf(&b_data, &[2, 3, 2]);
                let c = linalg::matmul(&ap, &bp).unwrap();
                crate::grad_fns::reduction::sum(&c).unwrap().item().unwrap()
            });
            let loss_minus = crate::autograd::no_grad::no_grad(|| {
                let am = no_grad_leaf(&a_minus, &[2, 2, 3]);
                let bm = no_grad_leaf(&b_data, &[2, 3, 2]);
                let c = linalg::matmul(&am, &bm).unwrap();
                crate::grad_fns::reduction::sum(&c).unwrap().item().unwrap()
            });

            let numerical = (loss_plus - loss_minus) / (2.0 * eps);
            assert!(
                (numerical - analytic_a[idx]).abs() < 5e-2,
                "grad_a[{idx}]: numerical={numerical}, analytic={}, diff={}",
                analytic_a[idx],
                (numerical - analytic_a[idx]).abs()
            );
        }

        // Check each element of B numerically.
        for idx in 0..b_data.len() {
            let mut b_plus = b_data.clone();
            b_plus[idx] += eps;
            let mut b_minus = b_data.clone();
            b_minus[idx] -= eps;

            let loss_plus = crate::autograd::no_grad::no_grad(|| {
                let ap = no_grad_leaf(&a_data, &[2, 2, 3]);
                let bp = no_grad_leaf(&b_plus, &[2, 3, 2]);
                let c = linalg::matmul(&ap, &bp).unwrap();
                crate::grad_fns::reduction::sum(&c).unwrap().item().unwrap()
            });
            let loss_minus = crate::autograd::no_grad::no_grad(|| {
                let am = no_grad_leaf(&a_data, &[2, 2, 3]);
                let bm = no_grad_leaf(&b_minus, &[2, 3, 2]);
                let c = linalg::matmul(&am, &bm).unwrap();
                crate::grad_fns::reduction::sum(&c).unwrap().item().unwrap()
            });

            let numerical = (loss_plus - loss_minus) / (2.0 * eps);
            assert!(
                (numerical - analytic_b[idx]).abs() < 5e-2,
                "grad_b[{idx}]: numerical={numerical}, analytic={}, diff={}",
                analytic_b[idx],
                (numerical - analytic_b[idx]).abs()
            );
        }
    }

    #[test]
    fn test_matmul_backward_3d_2d_broadcast_numerical() {
        // (2,3,4) @ (4,2) — B broadcasts over batch dim.
        // Gradient for B must sum over the batch dimension.
        let eps = 1e-4f32;

        let a_data: Vec<f32> = (0..24).map(|i| (i as f32) * 0.05 + 0.1).collect();
        let b_data: Vec<f32> = (0..8).map(|i| (i as f32) * 0.1 + 0.2).collect();

        let a = leaf(&a_data, &[2, 3, 4]);
        let b = leaf(&b_data, &[4, 2]);
        let c = matmul_differentiable(&a, &b).unwrap();
        let loss = crate::grad_fns::reduction::sum(&c).unwrap();
        loss.backward().unwrap();

        let analytic_a = a.grad().unwrap().unwrap().data().unwrap().to_vec();
        let analytic_b = b.grad().unwrap().unwrap().data().unwrap().to_vec();

        // Grad shapes should match input shapes.
        assert_eq!(a.grad().unwrap().unwrap().shape(), &[2, 3, 4]);
        assert_eq!(b.grad().unwrap().unwrap().shape(), &[4, 2]);

        // Numerical check for B (the broadcast operand — most important).
        for idx in 0..b_data.len() {
            let mut b_plus = b_data.clone();
            b_plus[idx] += eps;
            let mut b_minus = b_data.clone();
            b_minus[idx] -= eps;

            let loss_plus = crate::autograd::no_grad::no_grad(|| {
                let ap = no_grad_leaf(&a_data, &[2, 3, 4]);
                let bp = no_grad_leaf(&b_plus, &[4, 2]);
                let c = linalg::matmul(&ap, &bp).unwrap();
                crate::grad_fns::reduction::sum(&c).unwrap().item().unwrap()
            });
            let loss_minus = crate::autograd::no_grad::no_grad(|| {
                let am = no_grad_leaf(&a_data, &[2, 3, 4]);
                let bm = no_grad_leaf(&b_minus, &[4, 2]);
                let c = linalg::matmul(&am, &bm).unwrap();
                crate::grad_fns::reduction::sum(&c).unwrap().item().unwrap()
            });

            let numerical = (loss_plus - loss_minus) / (2.0 * eps);
            assert!(
                (numerical - analytic_b[idx]).abs() < 1e-2,
                "grad_b[{idx}]: numerical={numerical}, analytic={}, diff={}",
                analytic_b[idx],
                (numerical - analytic_b[idx]).abs()
            );
        }

        // Spot-check A gradient too.
        for idx in 0..4 {
            let mut a_plus = a_data.clone();
            a_plus[idx] += eps;
            let mut a_minus = a_data.clone();
            a_minus[idx] -= eps;

            let loss_plus = crate::autograd::no_grad::no_grad(|| {
                let ap = no_grad_leaf(&a_plus, &[2, 3, 4]);
                let bp = no_grad_leaf(&b_data, &[4, 2]);
                let c = linalg::matmul(&ap, &bp).unwrap();
                crate::grad_fns::reduction::sum(&c).unwrap().item().unwrap()
            });
            let loss_minus = crate::autograd::no_grad::no_grad(|| {
                let am = no_grad_leaf(&a_minus, &[2, 3, 4]);
                let bm = no_grad_leaf(&b_data, &[4, 2]);
                let c = linalg::matmul(&am, &bm).unwrap();
                crate::grad_fns::reduction::sum(&c).unwrap().item().unwrap()
            });

            let numerical = (loss_plus - loss_minus) / (2.0 * eps);
            assert!(
                (numerical - analytic_a[idx]).abs() < 1e-2,
                "grad_a[{idx}]: numerical={numerical}, analytic={}, diff={}",
                analytic_a[idx],
                (numerical - analytic_a[idx]).abs()
            );
        }
    }

    #[test]
    fn test_matmul_backward_batch_broadcast_1_vs_n() {
        // (1,2,3) @ (2,3,2) — batch dim 1 broadcasts to 2.
        // grad_a must sum over the broadcast batch dimension.
        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data: Vec<f32> = (0..12).map(|i| (i as f32) + 1.0).collect();

        let a = leaf(&a_data, &[1, 2, 3]);
        let b = leaf(&b_data, &[2, 3, 2]);
        let c = matmul_differentiable(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2, 2]);

        let loss = crate::grad_fns::reduction::sum(&c).unwrap();
        loss.backward().unwrap();

        // Grad shapes must match original shapes, not broadcast shapes.
        assert_eq!(a.grad().unwrap().unwrap().shape(), &[1, 2, 3]);
        assert_eq!(b.grad().unwrap().unwrap().shape(), &[2, 3, 2]);
    }

    // -----------------------------------------------------------------------
    // Decomposition backward FD audits (#1345): slogdet / cholesky / qr.
    //
    // Each VJP is verified against a CENTRAL finite difference of the op's own
    // forward (R-CHAR-3: the reference is reconstructed from the forward at
    // perturbed inputs, not a cached oracle constant). f64 throughout for FD
    // accuracy.
    // -----------------------------------------------------------------------

    fn leaf64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
    }

    fn no_grad_leaf64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
    }

    /// Central finite-difference gradient of scalar `f(x)` wrt each element.
    fn fd_grad64<F>(x: &[f64], shape: &[usize], eps: f64, f: F) -> Vec<f64>
    where
        F: Fn(&Tensor<f64>) -> f64,
    {
        let mut g = vec![0.0; x.len()];
        for i in 0..x.len() {
            let mut xp = x.to_vec();
            xp[i] += eps;
            let mut xm = x.to_vec();
            xm[i] -= eps;
            let lp = f(&no_grad_leaf64(&xp, shape));
            let lm = f(&no_grad_leaf64(&xm, shape));
            g[i] = (lp - lm) / (2.0 * eps);
        }
        g
    }

    fn assert_grad_close64(analytic: &[f64], numeric: &[f64], tol: f64, label: &str) {
        assert_eq!(analytic.len(), numeric.len(), "{label}: length mismatch");
        for (i, (&a, &n)) in analytic.iter().zip(numeric.iter()).enumerate() {
            assert!(
                (a - n).abs() < tol,
                "{label} grad[{i}]: analytic={a}, numeric={n}, diff={}",
                (a - n).abs()
            );
        }
    }

    // slogdet — VJP: dA = grad_logabsdet * inv(A)^T
    //   (FunctionsManual.cpp:4471 slogdet_backward, real case).
    #[test]
    fn slogdet_backward_matches_finite_difference() {
        // Well-conditioned non-symmetric 3x3 with det far from 0.
        let a_data = vec![2.0, 1.0, 0.0, 0.5, 3.0, 1.0, 0.0, 1.0, 2.5];
        let shape = [3, 3];

        let a = leaf64(&a_data, &shape);
        let (sign, logabsdet) = slogdet_differentiable(&a).unwrap();
        // sign is non-differentiable and should carry no grad_fn.
        assert!(sign.grad_fn().is_none(), "slogdet sign must be non-grad");
        assert!(logabsdet.is_scalar());
        logabsdet.backward().unwrap();
        let analytic = a.grad().unwrap().unwrap().data().unwrap().to_vec();

        let numeric = fd_grad64(&a_data, &shape, 1e-6, |x| {
            // Forward `linalg::slogdet` on a no-grad leaf returns the plain
            // (sign, logabsdet); take logabsdet.
            let (_s, la) = linalg_fwd::slogdet(x).unwrap();
            la.item().unwrap()
        });

        assert_grad_close64(&analytic, &numeric, 1e-4, "slogdet vs FD");
    }

    // cholesky — Phi-symmetrisation VJP (FunctionsManual.cpp:2048).
    #[test]
    fn cholesky_backward_matches_finite_difference() {
        // SPD 3x3 (symmetric, positive-definite, well-conditioned).
        let a_data = vec![4.0, 1.0, 0.5, 1.0, 3.0, 0.8, 0.5, 0.8, 2.5];
        let n = 3usize;
        let shape = [n, n];

        let a = leaf64(&a_data, &shape);
        let l = cholesky_differentiable(&a).unwrap();
        assert_eq!(l.shape(), &[n, n]);
        // Scalar loss = sum(L); covers every entry of the lower factor.
        let loss = crate::grad_fns::reduction::sum(&l).unwrap();
        loss.backward().unwrap();
        let analytic = a.grad().unwrap().unwrap().data().unwrap().to_vec();

        // PyTorch's `cholesky_backward` returns a SYMMETRIC gradient: the
        // off-diagonal sensitivity is split 50/50 across `[i,j]` and `[j,i]`
        // (`gA = 0.5*(gA + gA.tril(-1).mH())`), and the forward reads only the
        // lower triangle. So an unconstrained entrywise FD reads 0 on the upper
        // triangle. The honest reference is a SYMMETRIC finite difference:
        // perturb `A[i,j]` and `A[j,i]` together for `i != j` (and `A[i,i]`
        // alone for the diagonal). For a symmetric gradient that symmetric FD
        // equals `analytic[i,j] + analytic[j,i]` off-diagonal and
        // `analytic[i,i]` on-diagonal.
        let f = |x: &[f64]| -> f64 {
            let t = no_grad_leaf64(x, &shape);
            let l = linalg_fwd::cholesky(&t).unwrap();
            crate::grad_fns::reduction::sum(&l).unwrap().item().unwrap()
        };
        let eps = 1e-6;
        for i in 0..n {
            for j in 0..=i {
                let mut xp = a_data.clone();
                let mut xm = a_data.clone();
                xp[i * n + j] += eps;
                xm[i * n + j] -= eps;
                if i != j {
                    xp[j * n + i] += eps;
                    xm[j * n + i] -= eps;
                }
                let sym_fd = (f(&xp) - f(&xm)) / (2.0 * eps);
                let analytic_sym = if i == j {
                    analytic[i * n + j]
                } else {
                    analytic[i * n + j] + analytic[j * n + i]
                };
                assert!(
                    (analytic_sym - sym_fd).abs() < 1e-4,
                    "cholesky vs symmetric-FD at ({i},{j}): analytic_sym={analytic_sym}, \
                     fd={sym_fd}, diff={}",
                    (analytic_sym - sym_fd).abs()
                );
            }
        }
        // Also confirm the analytic gradient is itself symmetric (PyTorch
        // contract), so the split above is well-defined.
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (analytic[i * n + j] - analytic[j * n + i]).abs() < 1e-9,
                    "cholesky grad must be symmetric at ({i},{j})"
                );
            }
        }
    }

    // qr (reduced, m>=n) — both Q and R grad paths combine into A.grad
    //   (FunctionsManual.cpp:4166 linalg_qr_backward, m>=n branch).
    #[test]
    fn qr_backward_matches_finite_difference_square() {
        // Well-conditioned non-symmetric 3x3.
        let a_data = vec![1.0, 2.0, 0.5, 0.3, 1.5, 2.0, 1.0, 0.2, 3.0];
        let shape = [3, 3];

        // Loss = sum(Q) + sum(R) exercises BOTH the gQ node and the gR node,
        // so A.grad accumulates both partials of the joint VJP.
        let a = leaf64(&a_data, &shape);
        let (q, r) = qr_differentiable(&a).unwrap();
        assert_eq!(q.shape(), &[3, 3]);
        assert_eq!(r.shape(), &[3, 3]);
        let loss = crate::grad_fns::reduction::sum(&q)
            .unwrap()
            .add_t(&crate::grad_fns::reduction::sum(&r).unwrap())
            .unwrap();
        loss.backward().unwrap();
        let analytic = a.grad().unwrap().unwrap().data().unwrap().to_vec();

        let numeric = fd_grad64(&a_data, &shape, 1e-6, |x| {
            let (q, r) = linalg_fwd::qr(x).unwrap();
            let sq: f64 = q.data().unwrap().iter().sum();
            let sr: f64 = r.data().unwrap().iter().sum();
            sq + sr
        });

        // QR sign convention is fixed by the forward (positive-diagonal R), so
        // the forward is smooth in A; FD and analytic agree.
        assert_grad_close64(&analytic, &numeric, 1e-4, "qr vs FD");
    }

    // qr — exercise the Q-only and R-only paths independently.
    #[test]
    fn qr_backward_q_only_and_r_only() {
        let a_data = vec![1.0, 2.0, 0.5, 0.3, 1.5, 2.0, 1.0, 0.2, 3.0];
        let shape = [3, 3];

        // Q-only: loss = sum(Q); only QrBackwardQ fires.
        let a = leaf64(&a_data, &shape);
        let (q, _r) = qr_differentiable(&a).unwrap();
        let loss = crate::grad_fns::reduction::sum(&q).unwrap();
        loss.backward().unwrap();
        let g_q_only = a.grad().unwrap().unwrap().data().unwrap().to_vec();
        let num_q = fd_grad64(&a_data, &shape, 1e-6, |x| {
            let (q, _r) = linalg_fwd::qr(x).unwrap();
            q.data().unwrap().iter().sum()
        });
        assert_grad_close64(&g_q_only, &num_q, 1e-4, "qr Q-only vs FD");

        // R-only: loss = sum(R); only QrBackwardR fires.
        let a2 = leaf64(&a_data, &shape);
        let (_q, r) = qr_differentiable(&a2).unwrap();
        let loss2 = crate::grad_fns::reduction::sum(&r).unwrap();
        loss2.backward().unwrap();
        let g_r_only = a2.grad().unwrap().unwrap().data().unwrap().to_vec();
        let num_r = fd_grad64(&a_data, &shape, 1e-6, |x| {
            let (_q, r) = linalg_fwd::qr(x).unwrap();
            r.data().unwrap().iter().sum()
        });
        assert_grad_close64(&g_r_only, &num_r, 1e-4, "qr R-only vs FD");
    }

    // -----------------------------------------------------------------------
    // Grad-aware-forward wiring audits (#1583): trace / outer / det / inv /
    // solve. Each test drives the *public production forward* `crate::linalg::X`
    // (aliased `linalg_fwd::X`) on a `requires_grad` leaf — the path a real
    // autograd user hits — then checks `input.grad` against a CENTRAL finite
    // difference of that same forward at perturbed (no-grad) inputs (R-CHAR-3:
    // reference reconstructed from the forward, not a cached constant). These
    // would all read `grad == None` before the forwards delegated to the
    // `*_differentiable` wrappers.
    // -----------------------------------------------------------------------

    // trace — VJP dA = grad * I (derivatives.yaml:1785 trace_backward_symint).
    #[test]
    fn trace_forward_is_grad_aware_and_matches_fd() {
        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.5];
        let shape = [3, 3];

        let a = leaf64(&a_data, &shape);
        // Drive the PUBLIC forward, not the wrapper directly.
        let s = linalg_fwd::trace(&a).unwrap();
        assert!(
            s.grad_fn().is_some(),
            "trace forward must attach a grad_fn when input requires_grad"
        );
        assert!(s.is_scalar());
        s.backward().unwrap();
        let analytic = a.grad().unwrap().unwrap().data().unwrap().to_vec();

        let numeric = fd_grad64(&a_data, &shape, 1e-6, |x| {
            linalg_fwd::trace(x).unwrap().item().unwrap()
        });
        assert_grad_close64(&analytic, &numeric, 1e-5, "trace forward vs FD");
    }

    // outer — VJP da = grad @ b, db = grad^T @ a (derivatives.yaml:275-276).
    #[test]
    fn outer_forward_is_grad_aware_and_matches_fd() {
        let a_data = vec![1.5, -2.0, 0.5];
        let b_data = vec![2.0, 1.0, -1.5, 3.0];
        let a_shape = [3usize];
        let b_shape = [4usize];

        let a = leaf64(&a_data, &a_shape);
        let b = leaf64(&b_data, &b_shape);
        let c = linalg_fwd::outer(&a, &b).unwrap();
        assert!(
            c.grad_fn().is_some(),
            "outer forward must attach a grad_fn when input requires_grad"
        );
        assert_eq!(c.shape(), &[3, 4]);
        // Scalar loss = sum(C) so both a.grad and b.grad are exercised.
        let loss = crate::grad_fns::reduction::sum(&c).unwrap();
        loss.backward().unwrap();
        let grad_a = a.grad().unwrap().unwrap().data().unwrap().to_vec();
        let grad_b = b.grad().unwrap().unwrap().data().unwrap().to_vec();

        // FD wrt a (b fixed as no-grad).
        let num_a = fd_grad64(&a_data, &a_shape, 1e-6, |x| {
            let bb = no_grad_leaf64(&b_data, &b_shape);
            let c = linalg_fwd::outer(x, &bb).unwrap();
            c.data().unwrap().iter().sum()
        });
        assert_grad_close64(&grad_a, &num_a, 1e-5, "outer da vs FD");

        // FD wrt b (a fixed as no-grad).
        let num_b = fd_grad64(&b_data, &b_shape, 1e-6, |x| {
            let aa = no_grad_leaf64(&a_data, &a_shape);
            let c = linalg_fwd::outer(&aa, x).unwrap();
            c.data().unwrap().iter().sum()
        });
        assert_grad_close64(&grad_b, &num_b, 1e-5, "outer db vs FD");
    }

    // det — VJP dA = grad * det(A) * inv(A)^T (FunctionsManual.cpp:4373).
    #[test]
    fn det_forward_is_grad_aware_and_matches_fd() {
        // Well-conditioned non-symmetric 3x3, det far from 0.
        let a_data = vec![2.0, 1.0, 0.0, 0.5, 3.0, 1.0, 0.0, 1.0, 2.5];
        let shape = [3, 3];

        let a = leaf64(&a_data, &shape);
        let d = linalg_fwd::det(&a).unwrap();
        assert!(
            d.grad_fn().is_some(),
            "det forward must attach a grad_fn when input requires_grad"
        );
        assert!(d.is_scalar());
        d.backward().unwrap();
        let analytic = a.grad().unwrap().unwrap().data().unwrap().to_vec();

        let numeric = fd_grad64(&a_data, &shape, 1e-6, |x| {
            linalg_fwd::det(x).unwrap().item().unwrap()
        });
        assert_grad_close64(&analytic, &numeric, 1e-4, "det forward vs FD");
    }

    // inv — VJP dA = -Y^T @ grad @ Y^T, Y = A^{-1} (derivatives.yaml:916).
    #[test]
    fn inv_forward_is_grad_aware_and_matches_fd() {
        let a_data = vec![2.0, 1.0, 0.0, 0.5, 3.0, 1.0, 0.0, 1.0, 2.5];
        let shape = [3, 3];

        let a = leaf64(&a_data, &shape);
        let y = linalg_fwd::inv(&a).unwrap();
        assert!(
            y.grad_fn().is_some(),
            "inv forward must attach a grad_fn when input requires_grad"
        );
        assert_eq!(y.shape(), &[3, 3]);
        // Scalar loss = sum(Y) covers every entry of the inverse.
        let loss = crate::grad_fns::reduction::sum(&y).unwrap();
        loss.backward().unwrap();
        let analytic = a.grad().unwrap().unwrap().data().unwrap().to_vec();

        let numeric = fd_grad64(&a_data, &shape, 1e-6, |x| {
            let y = linalg_fwd::inv(x).unwrap();
            y.data().unwrap().iter().sum()
        });
        assert_grad_close64(&analytic, &numeric, 1e-4, "inv forward vs FD");
    }

    // solve (matrix RHS) — VJP gA = -gB @ X^T, gB = A^{-T} @ gX
    //   (FunctionsManual.cpp:6160 linalg_solve_backward).
    #[test]
    fn solve_forward_is_grad_aware_and_matches_fd_matrix_rhs() {
        let a_data = vec![3.0, 1.0, 0.5, 1.0, 4.0, 1.5, 0.5, 1.5, 5.0];
        let b_data = vec![1.0, 2.0, -1.0, 0.5, 2.0, 1.0];
        let a_shape = [3usize, 3];
        let b_shape = [3usize, 2];

        let a = leaf64(&a_data, &a_shape);
        let b = no_grad_leaf64(&b_data, &b_shape);
        let x = linalg_fwd::solve(&a, &b).unwrap();
        assert!(
            x.grad_fn().is_some(),
            "solve forward must attach a grad_fn when A requires_grad"
        );
        assert_eq!(x.shape(), &[3, 2]);
        let loss = crate::grad_fns::reduction::sum(&x).unwrap();
        loss.backward().unwrap();
        let grad_a = a.grad().unwrap().unwrap().data().unwrap().to_vec();

        // FD wrt A (B fixed as no-grad).
        let num_a = fd_grad64(&a_data, &a_shape, 1e-6, |xa| {
            let bb = no_grad_leaf64(&b_data, &b_shape);
            let x = linalg_fwd::solve(xa, &bb).unwrap();
            x.data().unwrap().iter().sum()
        });
        assert_grad_close64(&grad_a, &num_a, 1e-3, "solve dA (matrix RHS) vs FD");
    }

    // solve (vector RHS) — exercises the unsqueeze/squeeze column-promotion
    // branch + both grad_A and grad_B slots.
    #[test]
    fn solve_forward_is_grad_aware_and_matches_fd_vector_rhs() {
        let a_data = vec![3.0, 1.0, 0.5, 1.0, 4.0, 1.5, 0.5, 1.5, 5.0];
        let b_data = vec![1.0, 2.0, -1.0];
        let a_shape = [3usize, 3];
        let b_shape = [3usize];

        // grad on both A and B.
        let a = leaf64(&a_data, &a_shape);
        let b = leaf64(&b_data, &b_shape);
        let x = linalg_fwd::solve(&a, &b).unwrap();
        assert!(x.grad_fn().is_some(), "solve (vec RHS) must attach grad_fn");
        assert_eq!(x.shape(), &[3]);
        let loss = crate::grad_fns::reduction::sum(&x).unwrap();
        loss.backward().unwrap();
        let grad_a = a.grad().unwrap().unwrap().data().unwrap().to_vec();
        let grad_b = b.grad().unwrap().unwrap().data().unwrap().to_vec();

        let num_a = fd_grad64(&a_data, &a_shape, 1e-6, |xa| {
            let bb = no_grad_leaf64(&b_data, &b_shape);
            let x = linalg_fwd::solve(xa, &bb).unwrap();
            x.data().unwrap().iter().sum()
        });
        assert_grad_close64(&grad_a, &num_a, 1e-3, "solve dA (vector RHS) vs FD");

        let num_b = fd_grad64(&b_data, &b_shape, 1e-6, |xb| {
            let aa = no_grad_leaf64(&a_data, &a_shape);
            let x = linalg_fwd::solve(&aa, xb).unwrap();
            x.data().unwrap().iter().sum()
        });
        assert_grad_close64(&grad_b, &num_b, 1e-4, "solve dB (vector RHS) vs FD");
    }

    // -----------------------------------------------------------------------
    // #1583 consumer-wiring FD tests: each drives the now-grad-aware PUBLIC
    // forward (not the wrapper directly) and checks A.grad vs central FD.
    // addmm/addbmm/baddbmm/addmv/addr forwards live in `crate::linalg`; the
    // structural diag/tril/triu forwards live in `crate::ops::tensor_ops`;
    // diagonal lives in `crate::linalg`.
    // -----------------------------------------------------------------------

    // addmm — VJP dself=beta*grad, dmat1=alpha*grad@mat2^T, dmat2=alpha*mat1^T@grad
    //   (derivatives.yaml:256 addmm; LinearAlgebra.cpp:194,1620).
    #[test]
    fn addmm_public_forward_is_grad_aware_and_matches_fd() {
        let self_d = vec![0.5, -1.0, 2.0, 1.5];
        let m1_d = vec![1.0, 2.0, -1.0, 0.5, 3.0, 1.0];
        let m2_d = vec![2.0, -1.0, 0.5, 1.0, 1.5, -0.5];
        let self_s = [2usize, 2];
        let m1_s = [2usize, 3];
        let m2_s = [3usize, 2];
        let (beta, alpha) = (0.75f64, 1.25f64);

        let s = leaf64(&self_d, &self_s);
        let m1 = leaf64(&m1_d, &m1_s);
        let m2 = leaf64(&m2_d, &m2_s);
        let c = linalg_fwd::addmm(&s, &m1, &m2, beta, alpha).unwrap();
        assert!(
            c.grad_fn().is_some(),
            "addmm public forward must attach a grad_fn"
        );
        assert_eq!(c.shape(), &[2, 2]);
        let loss = crate::grad_fns::reduction::sum(&c).unwrap();
        loss.backward().unwrap();
        let g_self = s.grad().unwrap().unwrap().data().unwrap().to_vec();
        let g_m1 = m1.grad().unwrap().unwrap().data().unwrap().to_vec();
        let g_m2 = m2.grad().unwrap().unwrap().data().unwrap().to_vec();

        let num_self = fd_grad64(&self_d, &self_s, 1e-6, |x| {
            let m1 = no_grad_leaf64(&m1_d, &m1_s);
            let m2 = no_grad_leaf64(&m2_d, &m2_s);
            linalg_fwd::addmm(x, &m1, &m2, beta, alpha)
                .unwrap()
                .data()
                .unwrap()
                .iter()
                .sum()
        });
        assert_grad_close64(&g_self, &num_self, 1e-5, "addmm dself vs FD");
        let num_m1 = fd_grad64(&m1_d, &m1_s, 1e-6, |x| {
            let s = no_grad_leaf64(&self_d, &self_s);
            let m2 = no_grad_leaf64(&m2_d, &m2_s);
            linalg_fwd::addmm(&s, x, &m2, beta, alpha)
                .unwrap()
                .data()
                .unwrap()
                .iter()
                .sum()
        });
        assert_grad_close64(&g_m1, &num_m1, 1e-5, "addmm dmat1 vs FD");
        let num_m2 = fd_grad64(&m2_d, &m2_s, 1e-6, |x| {
            let s = no_grad_leaf64(&self_d, &self_s);
            let m1 = no_grad_leaf64(&m1_d, &m1_s);
            linalg_fwd::addmm(&s, &m1, x, beta, alpha)
                .unwrap()
                .data()
                .unwrap()
                .iter()
                .sum()
        });
        assert_grad_close64(&g_m2, &num_m2, 1e-5, "addmm dmat2 vs FD");
    }

    // addmv — VJP dself=beta*grad, dmat=alpha*outer(grad,vec), dvec=alpha*mat^T@grad
    //   (derivatives.yaml:267 addmv; Blas.cpp:40,72).
    #[test]
    fn addmv_public_forward_is_grad_aware_and_matches_fd() {
        let self_d = vec![0.5, -1.0];
        let mat_d = vec![1.0, 2.0, -1.0, 0.5, 3.0, 1.0];
        let vec_d = vec![2.0, -1.0, 0.5];
        let self_s = [2usize];
        let mat_s = [2usize, 3];
        let vec_s = [3usize];
        let (beta, alpha) = (0.5f64, 2.0f64);

        let s = leaf64(&self_d, &self_s);
        let mat = leaf64(&mat_d, &mat_s);
        let v = leaf64(&vec_d, &vec_s);
        let c = linalg_fwd::addmv(&s, &mat, &v, beta, alpha).unwrap();
        assert!(
            c.grad_fn().is_some(),
            "addmv public forward must attach a grad_fn"
        );
        assert_eq!(c.shape(), &[2]);
        let loss = crate::grad_fns::reduction::sum(&c).unwrap();
        loss.backward().unwrap();
        let g_mat = mat.grad().unwrap().unwrap().data().unwrap().to_vec();
        let g_vec = v.grad().unwrap().unwrap().data().unwrap().to_vec();

        let num_mat = fd_grad64(&mat_d, &mat_s, 1e-6, |x| {
            let s = no_grad_leaf64(&self_d, &self_s);
            let v = no_grad_leaf64(&vec_d, &vec_s);
            linalg_fwd::addmv(&s, x, &v, beta, alpha)
                .unwrap()
                .data()
                .unwrap()
                .iter()
                .sum()
        });
        assert_grad_close64(&g_mat, &num_mat, 1e-5, "addmv dmat vs FD");
        let num_vec = fd_grad64(&vec_d, &vec_s, 1e-6, |x| {
            let s = no_grad_leaf64(&self_d, &self_s);
            let mat = no_grad_leaf64(&mat_d, &mat_s);
            linalg_fwd::addmv(&s, &mat, x, beta, alpha)
                .unwrap()
                .data()
                .unwrap()
                .iter()
                .sum()
        });
        assert_grad_close64(&g_vec, &num_vec, 1e-5, "addmv dvec vs FD");
    }

    // addr — VJP dself=beta*grad, dvec1=alpha*grad@vec2, dvec2=alpha*grad^T@vec1
    //   (derivatives.yaml:273 addr; LinearAlgebra.cpp:1200).
    #[test]
    fn addr_public_forward_is_grad_aware_and_matches_fd() {
        let self_d = vec![0.5, -1.0, 2.0, 1.5, 0.0, -0.5];
        let v1_d = vec![1.5, -2.0];
        let v2_d = vec![2.0, 1.0, -1.5];
        let self_s = [2usize, 3];
        let v1_s = [2usize];
        let v2_s = [3usize];
        let (beta, alpha) = (1.0f64, 0.5f64);

        let s = leaf64(&self_d, &self_s);
        let v1 = leaf64(&v1_d, &v1_s);
        let v2 = leaf64(&v2_d, &v2_s);
        let c = linalg_fwd::addr(&s, &v1, &v2, beta, alpha).unwrap();
        assert!(
            c.grad_fn().is_some(),
            "addr public forward must attach a grad_fn"
        );
        assert_eq!(c.shape(), &[2, 3]);
        let loss = crate::grad_fns::reduction::sum(&c).unwrap();
        loss.backward().unwrap();
        let g_v1 = v1.grad().unwrap().unwrap().data().unwrap().to_vec();
        let g_v2 = v2.grad().unwrap().unwrap().data().unwrap().to_vec();

        let num_v1 = fd_grad64(&v1_d, &v1_s, 1e-6, |x| {
            let s = no_grad_leaf64(&self_d, &self_s);
            let v2 = no_grad_leaf64(&v2_d, &v2_s);
            linalg_fwd::addr(&s, x, &v2, beta, alpha)
                .unwrap()
                .data()
                .unwrap()
                .iter()
                .sum()
        });
        assert_grad_close64(&g_v1, &num_v1, 1e-5, "addr dvec1 vs FD");
        let num_v2 = fd_grad64(&v2_d, &v2_s, 1e-6, |x| {
            let s = no_grad_leaf64(&self_d, &self_s);
            let v1 = no_grad_leaf64(&v1_d, &v1_s);
            linalg_fwd::addr(&s, &v1, x, beta, alpha)
                .unwrap()
                .data()
                .unwrap()
                .iter()
                .sum()
        });
        assert_grad_close64(&g_v2, &num_v2, 1e-5, "addr dvec2 vs FD");
    }

    // addbmm — VJP dself=beta*grad, dbatch1[b]=alpha*grad@batch2[b]^T,
    //   dbatch2[b]=alpha*batch1[b]^T@grad (derivatives.yaml:238 addbmm).
    #[test]
    fn addbmm_public_forward_is_grad_aware_and_matches_fd() {
        // 2 batches of [2,2] @ [2,2], self [2,2].
        let self_d = vec![0.5, -1.0, 2.0, 1.5];
        let b1_d = vec![1.0, 2.0, -1.0, 0.5, 0.5, -1.0, 2.0, 1.0];
        let b2_d = vec![2.0, -1.0, 0.5, 1.0, 1.0, 0.0, -0.5, 2.0];
        let self_s = [2usize, 2];
        let b_s = [2usize, 2, 2];
        let (beta, alpha) = (0.5f64, 1.5f64);

        let s = leaf64(&self_d, &self_s);
        let b1 = leaf64(&b1_d, &b_s);
        let b2 = leaf64(&b2_d, &b_s);
        let c = linalg_fwd::addbmm(&s, &b1, &b2, beta, alpha).unwrap();
        assert!(
            c.grad_fn().is_some(),
            "addbmm public forward must attach a grad_fn"
        );
        assert_eq!(c.shape(), &[2, 2]);
        let loss = crate::grad_fns::reduction::sum(&c).unwrap();
        loss.backward().unwrap();
        let g_b1 = b1.grad().unwrap().unwrap().data().unwrap().to_vec();

        let num_b1 = fd_grad64(&b1_d, &b_s, 1e-6, |x| {
            let s = no_grad_leaf64(&self_d, &self_s);
            let b2 = no_grad_leaf64(&b2_d, &b_s);
            linalg_fwd::addbmm(&s, x, &b2, beta, alpha)
                .unwrap()
                .data()
                .unwrap()
                .iter()
                .sum()
        });
        assert_grad_close64(&g_b1, &num_b1, 1e-5, "addbmm dbatch1 vs FD");
    }

    // baddbmm — per-batch addmm VJP (derivatives.yaml:359 baddbmm).
    #[test]
    fn baddbmm_public_forward_is_grad_aware_and_matches_fd() {
        let self_d = vec![0.5, -1.0, 2.0, 1.5, 0.0, 1.0, -0.5, 2.0];
        let b1_d = vec![1.0, 2.0, -1.0, 0.5, 0.5, -1.0, 2.0, 1.0];
        let b2_d = vec![2.0, -1.0, 0.5, 1.0, 1.0, 0.0, -0.5, 2.0];
        let s_s = [2usize, 2, 2];
        let (beta, alpha) = (1.0f64, 0.75f64);

        let s = leaf64(&self_d, &s_s);
        let b1 = leaf64(&b1_d, &s_s);
        let b2 = leaf64(&b2_d, &s_s);
        let c = linalg_fwd::baddbmm(&s, &b1, &b2, beta, alpha).unwrap();
        assert!(
            c.grad_fn().is_some(),
            "baddbmm public forward must attach a grad_fn"
        );
        assert_eq!(c.shape(), &[2, 2, 2]);
        let loss = crate::grad_fns::reduction::sum(&c).unwrap();
        loss.backward().unwrap();
        let g_b2 = b2.grad().unwrap().unwrap().data().unwrap().to_vec();

        let num_b2 = fd_grad64(&b2_d, &s_s, 1e-6, |x| {
            let s = no_grad_leaf64(&self_d, &s_s);
            let b1 = no_grad_leaf64(&b1_d, &s_s);
            linalg_fwd::baddbmm(&s, &b1, x, beta, alpha)
                .unwrap()
                .data()
                .unwrap()
                .iter()
                .sum()
        });
        assert_grad_close64(&g_b2, &num_b2, 1e-5, "baddbmm dbatch2 vs FD");
    }

    // kron — per-Kron-block VJP (LinearAlgebra.cpp:3530 kron).
    #[test]
    fn kron_public_forward_is_grad_aware_and_matches_fd() {
        let a_d = vec![1.0, 2.0, -1.0, 0.5];
        let b_d = vec![2.0, -1.0, 0.5, 1.0];
        let a_s = [2usize, 2];
        let b_s = [2usize, 2];

        let a = leaf64(&a_d, &a_s);
        let b = leaf64(&b_d, &b_s);
        let c = linalg_fwd::kron(&a, &b).unwrap();
        assert!(
            c.grad_fn().is_some(),
            "kron public forward must attach a grad_fn"
        );
        assert_eq!(c.shape(), &[4, 4]);
        let loss = crate::grad_fns::reduction::sum(&c).unwrap();
        loss.backward().unwrap();
        let g_a = a.grad().unwrap().unwrap().data().unwrap().to_vec();
        let g_b = b.grad().unwrap().unwrap().data().unwrap().to_vec();

        let num_a = fd_grad64(&a_d, &a_s, 1e-6, |x| {
            let b = no_grad_leaf64(&b_d, &b_s);
            linalg_fwd::kron(x, &b)
                .unwrap()
                .data()
                .unwrap()
                .iter()
                .sum()
        });
        assert_grad_close64(&g_a, &num_a, 1e-5, "kron dA vs FD");
        let num_b = fd_grad64(&b_d, &b_s, 1e-6, |x| {
            let a = no_grad_leaf64(&a_d, &a_s);
            linalg_fwd::kron(&a, x)
                .unwrap()
                .data()
                .unwrap()
                .iter()
                .sum()
        });
        assert_grad_close64(&g_b, &num_b, 1e-5, "kron dB vs FD");
    }

    // diagonal — VJP scatters grad onto the offset-th diagonal
    //   (derivatives.yaml:573 diagonal_backward_symint).
    #[test]
    fn diagonal_public_forward_is_grad_aware_and_matches_fd() {
        let a_d = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let a_s = [3usize, 3];
        let offset = 1i64;

        let a = leaf64(&a_d, &a_s);
        let d = linalg_fwd::diagonal(&a, offset).unwrap();
        assert!(
            d.grad_fn().is_some(),
            "diagonal public forward must attach a grad_fn"
        );
        let loss = crate::grad_fns::reduction::sum(&d).unwrap();
        loss.backward().unwrap();
        let g = a.grad().unwrap().unwrap().data().unwrap().to_vec();

        let num = fd_grad64(&a_d, &a_s, 1e-6, |x| {
            linalg_fwd::diagonal(x, offset)
                .unwrap()
                .data()
                .unwrap()
                .iter()
                .sum()
        });
        assert_grad_close64(&g, &num, 1e-5, "diagonal vs FD");
    }

    // diag (2-D extract) — VJP scatters grad onto the diagonal.
    #[test]
    fn diag_extract_public_forward_is_grad_aware_and_matches_fd() {
        let a_d = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let a_s = [3usize, 3];

        let a = leaf64(&a_d, &a_s);
        let d = crate::ops::tensor_ops::diag(&a, 0).unwrap();
        assert!(
            d.grad_fn().is_some(),
            "diag (extract) public forward must attach a grad_fn"
        );
        assert_eq!(d.shape(), &[3]);
        let loss = crate::grad_fns::reduction::sum(&d).unwrap();
        loss.backward().unwrap();
        let g = a.grad().unwrap().unwrap().data().unwrap().to_vec();

        let num = fd_grad64(&a_d, &a_s, 1e-6, |x| {
            crate::ops::tensor_ops::diag(x, 0)
                .unwrap()
                .data()
                .unwrap()
                .iter()
                .sum()
        });
        assert_grad_close64(&g, &num, 1e-5, "diag (extract) vs FD");
    }

    // diag (1-D construct) — VJP gathers grad's diagonal.
    #[test]
    fn diag_construct_public_forward_is_grad_aware_and_matches_fd() {
        let a_d = vec![1.0, 2.0, 3.0];
        let a_s = [3usize];

        let a = leaf64(&a_d, &a_s);
        let d = crate::ops::tensor_ops::diag(&a, 0).unwrap();
        assert!(
            d.grad_fn().is_some(),
            "diag (construct) public forward must attach a grad_fn"
        );
        assert_eq!(d.shape(), &[3, 3]);
        // Weighted loss so the gradient is not uniformly 1 (catches scatter bugs).
        let w = no_grad_leaf64(&[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0], &[3usize, 3]);
        let prod = crate::grad_fns::arithmetic::mul(&d, &w).unwrap();
        let loss = crate::grad_fns::reduction::sum(&prod).unwrap();
        loss.backward().unwrap();
        let g = a.grad().unwrap().unwrap().data().unwrap().to_vec();

        let num = fd_grad64(&a_d, &a_s, 1e-6, |x| {
            let dd = crate::ops::tensor_ops::diag(x, 0).unwrap();
            let wv = [1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0];
            dd.data()
                .unwrap()
                .iter()
                .zip(wv.iter())
                .map(|(a, b)| a * b)
                .sum()
        });
        assert_grad_close64(&g, &num, 1e-5, "diag (construct) vs FD");
    }

    // tril — VJP masks grad by the kept lower triangle (derivatives.yaml:1805).
    #[test]
    fn tril_public_forward_is_grad_aware_and_matches_fd() {
        let a_d = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let a_s = [3usize, 3];
        // Weighted loss to catch the mask: w has support on both triangles.
        let w = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let a = leaf64(&a_d, &a_s);
        let t = crate::ops::tensor_ops::tril(&a, 0).unwrap();
        assert!(
            t.grad_fn().is_some(),
            "tril public forward must attach a grad_fn"
        );
        let wt = no_grad_leaf64(&w, &a_s);
        let prod = crate::grad_fns::arithmetic::mul(&t, &wt).unwrap();
        let loss = crate::grad_fns::reduction::sum(&prod).unwrap();
        loss.backward().unwrap();
        let g = a.grad().unwrap().unwrap().data().unwrap().to_vec();

        let num = fd_grad64(&a_d, &a_s, 1e-6, |x| {
            let t = crate::ops::tensor_ops::tril(x, 0).unwrap();
            t.data()
                .unwrap()
                .iter()
                .zip(w.iter())
                .map(|(a, b)| a * b)
                .sum()
        });
        assert_grad_close64(&g, &num, 1e-5, "tril vs FD");
    }

    // triu — VJP masks grad by the kept upper triangle (derivatives.yaml:1809).
    #[test]
    fn triu_public_forward_is_grad_aware_and_matches_fd() {
        let a_d = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let a_s = [3usize, 3];
        let w = [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        let a = leaf64(&a_d, &a_s);
        let t = crate::ops::tensor_ops::triu(&a, 0).unwrap();
        assert!(
            t.grad_fn().is_some(),
            "triu public forward must attach a grad_fn"
        );
        let wt = no_grad_leaf64(&w, &a_s);
        let prod = crate::grad_fns::arithmetic::mul(&t, &wt).unwrap();
        let loss = crate::grad_fns::reduction::sum(&prod).unwrap();
        loss.backward().unwrap();
        let g = a.grad().unwrap().unwrap().data().unwrap().to_vec();

        let num = fd_grad64(&a_d, &a_s, 1e-6, |x| {
            let t = crate::ops::tensor_ops::triu(x, 0).unwrap();
            t.data()
                .unwrap()
                .iter()
                .zip(w.iter())
                .map(|(a, b)| a * b)
                .sum()
        });
        assert_grad_close64(&g, &num, 1e-5, "triu vs FD");
    }

    // -----------------------------------------------------------------------
    // #1577 research-grade decomposition VJPs. Each drives the now-grad-aware
    // PUBLIC forward and checks A.grad against central finite differences.
    // A symmetric input is built as `S = M + M^T` so the symmetric-eigh /
    // eigvalsh forward sees a genuinely symmetric matrix and the FD perturbs
    // the same symmetric forward (perturbing one entry off-symmetry still
    // probes the gradient torch returns, because torch's eigvalsh/eigh read a
    // single triangle and return a symmetrized gradient — we compare the
    // symmetrized analytic grad to the symmetrized FD grad).
    // -----------------------------------------------------------------------

    /// Weighted-sum-of-outputs loss gradient seed `w`, summed against the
    /// flattened output. Mirrors the triu test's weighting so the upstream
    /// gradient on the output is non-uniform (exercising off-diagonal terms).
    fn weighted_sum(out: &[f64], w: &[f64]) -> f64 {
        out.iter().zip(w.iter()).map(|(a, b)| a * b).sum()
    }

    // eigvalsh — VJP gA = U diag(gw) U^T symmetrized
    //   (FunctionsManual.cpp:3859 linalg_eig_backward Hermitian eigvals).
    #[test]
    fn eigvalsh_public_forward_is_grad_aware_and_matches_fd() {
        // Symmetric well-conditioned 3x3 with DISTINCT eigenvalues.
        let a_data = vec![4.0, 1.0, 0.5, 1.0, 3.0, 0.8, 0.5, 0.8, 2.0];
        let shape = [3usize, 3];
        // Non-uniform gradient seed on the 3 eigenvalues.
        let w = [0.7f64, -1.3, 2.1];

        let a = leaf64(&a_data, &shape);
        let lam = linalg_fwd::eigvalsh(&a).unwrap();
        assert!(
            lam.grad_fn().is_some(),
            "eigvalsh forward must attach a grad_fn"
        );
        assert_eq!(lam.shape(), &[3]);
        let wt = no_grad_leaf64(&w, &[3]);
        let prod = crate::grad_fns::arithmetic::mul(&lam, &wt).unwrap();
        let loss = crate::grad_fns::reduction::sum(&prod).unwrap();
        loss.backward().unwrap();
        let analytic = a.grad().unwrap().unwrap().data().unwrap().to_vec();

        let numeric = fd_grad64(&a_data, &shape, 1e-6, |x| {
            weighted_sum(linalg_fwd::eigvalsh(x).unwrap().data().unwrap(), &w)
        });
        // The analytic grad is symmetrized; symmetrize the FD grad to match
        // (FD perturbs each entry independently, breaking symmetry).
        let sym = |g: &[f64]| -> Vec<f64> {
            let mut out = vec![0.0; 9];
            for i in 0..3 {
                for j in 0..3 {
                    out[i * 3 + j] = 0.5 * (g[i * 3 + j] + g[j * 3 + i]);
                }
            }
            out
        };
        assert_grad_close64(&sym(&analytic), &sym(&numeric), 1e-4, "eigvalsh vs FD");
    }

    // eigh — F-matrix VJP with skew-symmetric projection
    //   (FunctionsManual.cpp:3882-3917 linalg_eig_backward Hermitian branch).
    #[test]
    fn eigh_public_forward_is_grad_aware_and_matches_fd() {
        // Symmetric well-conditioned 3x3 with DISTINCT eigenvalues (gaps ~1).
        let a_data = vec![4.0, 0.5, 0.3, 0.5, 2.5, 0.2, 0.3, 0.2, 1.0];
        let shape = [3usize, 3];
        // Non-uniform seeds: weight the eigenvalues AND the eigenvectors so the
        // F-matrix off-diagonal terms are exercised.
        let ww = [0.4f64, -0.9, 1.5]; // on eigenvalues w
        let wv = [0.2f64, -0.5, 0.7, 0.3, 0.1, -0.4, -0.6, 0.8, 0.25]; // on eigenvectors U (3x3)

        let a = leaf64(&a_data, &shape);
        let (w, u) = linalg_fwd::eigh(&a).unwrap();
        assert!(w.grad_fn().is_some(), "eigh w output must attach a grad_fn");
        assert!(u.grad_fn().is_some(), "eigh U output must attach a grad_fn");
        let wwt = no_grad_leaf64(&ww, &[3]);
        let wvt = no_grad_leaf64(&wv, &[3, 3]);
        let lw =
            crate::grad_fns::reduction::sum(&crate::grad_fns::arithmetic::mul(&w, &wwt).unwrap())
                .unwrap();
        let lv =
            crate::grad_fns::reduction::sum(&crate::grad_fns::arithmetic::mul(&u, &wvt).unwrap())
                .unwrap();
        let loss = crate::grad_fns::arithmetic::add(&lw, &lv).unwrap();
        loss.backward().unwrap();
        let analytic = a.grad().unwrap().unwrap().data().unwrap().to_vec();

        // FD: build the same loss = <ww, w> + <wv, U>. Eigenvector signs are
        // gauge-free; eigh returns a deterministic sign convention, so FD on
        // the same forward is consistent.
        let numeric = fd_grad64(&a_data, &shape, 1e-6, |x| {
            let (w, u) = linalg_fwd::eigh(x).unwrap();
            weighted_sum(w.data().unwrap(), &ww) + weighted_sum(u.data().unwrap(), &wv)
        });
        let sym = |g: &[f64]| -> Vec<f64> {
            let mut out = vec![0.0; 9];
            for i in 0..3 {
                for j in 0..3 {
                    out[i * 3 + j] = 0.5 * (g[i * 3 + j] + g[j * 3 + i]);
                }
            }
            out
        };
        assert_grad_close64(&sym(&analytic), &sym(&numeric), 2e-3, "eigh vs FD");
    }

    // pinv — algebraic Moore-Penrose VJP, both m<=n and m>n branches
    //   (FunctionsManual.cpp:2175 pinv_backward).
    #[test]
    fn pinv_public_forward_is_grad_aware_and_matches_fd_tall() {
        // Tall full-rank 4x2 (m > n).
        let a_data = vec![1.0, 0.5, 2.0, -1.0, 0.3, 1.5, -0.7, 2.0];
        let shape = [4usize, 2];
        let w: Vec<f64> = (0..8).map(|i| 0.3 + 0.2 * (i as f64)).collect(); // pinv is 2x4

        let a = leaf64(&a_data, &shape);
        let p = linalg_fwd::pinv(&a).unwrap();
        assert!(p.grad_fn().is_some(), "pinv forward must attach a grad_fn");
        assert_eq!(p.shape(), &[2, 4]);
        let wt = no_grad_leaf64(&w, &[2, 4]);
        let loss =
            crate::grad_fns::reduction::sum(&crate::grad_fns::arithmetic::mul(&p, &wt).unwrap())
                .unwrap();
        loss.backward().unwrap();
        let analytic = a.grad().unwrap().unwrap().data().unwrap().to_vec();

        let numeric = fd_grad64(&a_data, &shape, 1e-6, |x| {
            weighted_sum(linalg_fwd::pinv(x).unwrap().data().unwrap(), &w)
        });
        assert_grad_close64(&analytic, &numeric, 1e-3, "pinv (tall) vs FD");
    }

    #[test]
    fn pinv_public_forward_is_grad_aware_and_matches_fd_wide() {
        // Wide full-rank 2x4 (m < n) — exercises the m<=n branch.
        let a_data = vec![1.0, 0.5, 2.0, -1.0, 0.3, 1.5, -0.7, 2.0];
        let shape = [2usize, 4];
        let w: Vec<f64> = (0..8).map(|i| 0.2 - 0.15 * (i as f64)).collect(); // pinv is 4x2

        let a = leaf64(&a_data, &shape);
        let p = linalg_fwd::pinv(&a).unwrap();
        assert!(p.grad_fn().is_some(), "pinv forward must attach a grad_fn");
        assert_eq!(p.shape(), &[4, 2]);
        let wt = no_grad_leaf64(&w, &[4, 2]);
        let loss =
            crate::grad_fns::reduction::sum(&crate::grad_fns::arithmetic::mul(&p, &wt).unwrap())
                .unwrap();
        loss.backward().unwrap();
        let analytic = a.grad().unwrap().unwrap().data().unwrap().to_vec();

        let numeric = fd_grad64(&a_data, &shape, 1e-6, |x| {
            weighted_sum(linalg_fwd::pinv(x).unwrap().data().unwrap(), &w)
        });
        assert_grad_close64(&analytic, &numeric, 1e-3, "pinv (wide) vs FD");
    }

    // lstsq — solution-output VJP via pinv_backward
    //   (FunctionsManual.cpp:4038-4050 linalg_lstsq_backward).
    #[test]
    fn lstsq_public_forward_is_grad_aware_and_matches_fd() {
        // Overdetermined full-rank 4x2 system, matrix RHS 4x2.
        let a_data = vec![1.0, 0.5, 2.0, -1.0, 0.3, 1.5, -0.7, 2.0];
        let b_data = vec![1.0, -0.5, 0.8, 1.2, -0.3, 0.6, 2.0, -1.0];
        let a_s = [4usize, 2];
        let b_s = [4usize, 2];
        let w = [0.7f64, -1.1, 0.4, 1.3]; // solution is 2x2

        let a = leaf64(&a_data, &a_s);
        let b = leaf64(&b_data, &b_s);
        let (sol, _r, _rank, _sv) = linalg_fwd::lstsq(&a, &b, None).unwrap();
        assert!(
            sol.grad_fn().is_some(),
            "lstsq solution must attach a grad_fn"
        );
        assert_eq!(sol.shape(), &[2, 2]);
        let wt = no_grad_leaf64(&w, &[2, 2]);
        let loss =
            crate::grad_fns::reduction::sum(&crate::grad_fns::arithmetic::mul(&sol, &wt).unwrap())
                .unwrap();
        loss.backward().unwrap();
        let grad_a = a.grad().unwrap().unwrap().data().unwrap().to_vec();
        let grad_b = b.grad().unwrap().unwrap().data().unwrap().to_vec();

        let num_a = fd_grad64(&a_data, &a_s, 1e-6, |x| {
            let bb = no_grad_leaf64(&b_data, &b_s);
            let (s, _, _, _) = linalg_fwd::lstsq(x, &bb, None).unwrap();
            weighted_sum(s.data().unwrap(), &w)
        });
        assert_grad_close64(&grad_a, &num_a, 1e-3, "lstsq dA vs FD");

        let num_b = fd_grad64(&b_data, &b_s, 1e-6, |x| {
            let aa = no_grad_leaf64(&a_data, &a_s);
            let (s, _, _, _) = linalg_fwd::lstsq(&aa, x, None).unwrap();
            weighted_sum(s.data().unwrap(), &w)
        });
        assert_grad_close64(&grad_b, &num_b, 1e-4, "lstsq dB vs FD");
    }

    // lu — split (L,U) VJP, square case
    //   (FunctionsManual.cpp:6854 linalg_lu_backward m==n branch).
    #[test]
    fn lu_public_forward_is_grad_aware_and_matches_fd() {
        // Square 3x3 that REQUIRES a row pivot (col-0 max is row 2, value 7),
        // exercising the `P^T` adjoint in the VJP.
        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0];
        let shape = [3usize, 3];
        // Weight L and U separately so both split nodes contribute.
        let wl: Vec<f64> = (0..9).map(|i| 0.3 + 0.1 * (i as f64)).collect();
        let wu: Vec<f64> = (0..9).map(|i| -0.2 + 0.15 * (i as f64)).collect();

        let a = leaf64(&a_data, &shape);
        let (p, l, u) = linalg_fwd::lu(&a).unwrap();
        assert!(p.grad_fn().is_none(), "lu P output is non-differentiable");
        assert!(l.grad_fn().is_some(), "lu L output must attach a grad_fn");
        assert!(u.grad_fn().is_some(), "lu U output must attach a grad_fn");
        let wlt = no_grad_leaf64(&wl, &shape);
        let wut = no_grad_leaf64(&wu, &shape);
        let ll =
            crate::grad_fns::reduction::sum(&crate::grad_fns::arithmetic::mul(&l, &wlt).unwrap())
                .unwrap();
        let lu_loss =
            crate::grad_fns::reduction::sum(&crate::grad_fns::arithmetic::mul(&u, &wut).unwrap())
                .unwrap();
        let loss = crate::grad_fns::arithmetic::add(&ll, &lu_loss).unwrap();
        loss.backward().unwrap();
        let analytic = a.grad().unwrap().unwrap().data().unwrap().to_vec();

        let numeric = fd_grad64(&a_data, &shape, 1e-6, |x| {
            let (_p, l, u) = linalg_fwd::lu(x).unwrap();
            weighted_sum(l.data().unwrap(), &wl) + weighted_sum(u.data().unwrap(), &wu)
        });
        assert_grad_close64(&analytic, &numeric, 1e-3, "lu vs FD");
    }

    // lu — gradient on a NON-INVOLUTORY (3-cycle) pivot, pinned to LIVE
    // torch (CORE-144 / #1838). The FD test above is self-consistent under
    // EITHER permutation convention (forward and backward flip together),
    // so only a torch-pinned value can catch a convention mismatch in the
    // saved-P adjoint (`gA = P @ M`, FunctionsManual.cpp:6873).
    //
    // Oracle (live torch 2.11.0+cu130):
    //   A = torch.tensor([[0.5,4.,1.],[1.,0.25,3.],[6.,2.,0.5]],
    //                    dtype=torch.float64, requires_grad=True)
    //   P, L, U = torch.linalg.lu(A)            # ipiv 3-cycle [3,3,3]
    //   WL = torch.arange(1., 10., dtype=torch.float64).reshape(3, 3)
    //   WU = torch.arange(2., 11., dtype=torch.float64).reshape(3, 3)
    //   ((L*WL).sum() + (U*WU).sum()).backward()
    //   A.grad.flatten().tolist() ==
    //     [-1.9317895400126024, 5.991020793950851, 7.217391304347826,
    //       0.4710144927536232, -0.4130434782608696, 10.0,
    //       2.08248004620878, 2.5695888468809076, 1.7318840579710146]
    #[test]
    fn lu_backward_three_cycle_pivot_matches_torch_1838() {
        let a_data = vec![0.5, 4.0, 1.0, 1.0, 0.25, 3.0, 6.0, 2.0, 0.5];
        let shape = [3usize, 3];
        let wl: Vec<f64> = (1..10).map(|i| i as f64).collect();
        let wu: Vec<f64> = (2..11).map(|i| i as f64).collect();
        let expected = [
            -1.931_789_540_012_602_4,
            5.991_020_793_950_851,
            7.217_391_304_347_826,
            0.471_014_492_753_623_2,
            -0.413_043_478_260_869_6,
            10.0,
            2.082_480_046_208_78,
            2.569_588_846_880_907_6,
            1.731_884_057_971_014_6,
        ];

        let a = leaf64(&a_data, &shape);
        let (_p, l, u) = linalg_fwd::lu(&a).unwrap();
        let wlt = no_grad_leaf64(&wl, &shape);
        let wut = no_grad_leaf64(&wu, &shape);
        let ll =
            crate::grad_fns::reduction::sum(&crate::grad_fns::arithmetic::mul(&l, &wlt).unwrap())
                .unwrap();
        let lu_loss =
            crate::grad_fns::reduction::sum(&crate::grad_fns::arithmetic::mul(&u, &wut).unwrap())
                .unwrap();
        let loss = crate::grad_fns::arithmetic::add(&ll, &lu_loss).unwrap();
        loss.backward().unwrap();
        let analytic = a.grad().unwrap().unwrap().data().unwrap().to_vec();
        // f64 tolerance: values are O(10); the VJP chains two triangular
        // solves + one matmul over n = 3 (accumulation length ≤ 3), so
        // expected drift is within ~1e2 ULP ≈ 1e-13 relative; 1e-9 absolute
        // gives slack without admitting a convention error (which moves
        // entries by O(1)).
        assert_grad_close64(&analytic, &expected, 1e-9, "lu 3-cycle dA vs torch");
    }

    // lu_factor — packed-LU VJP via grad_a_combined
    //   (FunctionsManual.cpp:6960 lu_factor_ex_backward, m==n).
    #[test]
    fn lu_factor_public_forward_is_grad_aware_and_matches_fd() {
        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0];
        let shape = [3usize, 3];
        let w: Vec<f64> = (0..9).map(|i| 0.25 + 0.1 * (i as f64)).collect();

        let a = leaf64(&a_data, &shape);
        let (lu, _piv) = linalg_fwd::lu_factor(&a).unwrap();
        assert!(
            lu.grad_fn().is_some(),
            "lu_factor packed LU must attach a grad_fn"
        );
        assert_eq!(lu.shape(), &[3, 3]);
        let wt = no_grad_leaf64(&w, &shape);
        let loss =
            crate::grad_fns::reduction::sum(&crate::grad_fns::arithmetic::mul(&lu, &wt).unwrap())
                .unwrap();
        loss.backward().unwrap();
        let analytic = a.grad().unwrap().unwrap().data().unwrap().to_vec();

        let numeric = fd_grad64(&a_data, &shape, 1e-6, |x| {
            let (lu, _p) = linalg_fwd::lu_factor(x).unwrap();
            weighted_sum(lu.data().unwrap(), &w)
        });
        assert_grad_close64(&analytic, &numeric, 1e-3, "lu_factor vs FD");
    }

    // cross — bilinear VJP da = cross(b, grad), db = cross(grad, a)
    //   (derivatives.yaml:516-518 linalg_cross).
    #[test]
    fn cross_public_forward_is_grad_aware_and_matches_fd() {
        let a_data = vec![1.0, 2.0, -1.0];
        let b_data = vec![0.5, -1.5, 2.0];
        let shape = [3usize];
        let w = [0.8f64, -1.2, 0.6];

        let a = leaf64(&a_data, &shape);
        let b = leaf64(&b_data, &shape);
        let c = linalg_fwd::cross(&a, &b, -1).unwrap();
        assert!(c.grad_fn().is_some(), "cross forward must attach a grad_fn");
        assert_eq!(c.shape(), &[3]);
        let wt = no_grad_leaf64(&w, &shape);
        let loss =
            crate::grad_fns::reduction::sum(&crate::grad_fns::arithmetic::mul(&c, &wt).unwrap())
                .unwrap();
        loss.backward().unwrap();
        let grad_a = a.grad().unwrap().unwrap().data().unwrap().to_vec();
        let grad_b = b.grad().unwrap().unwrap().data().unwrap().to_vec();

        let num_a = fd_grad64(&a_data, &shape, 1e-6, |x| {
            let bb = no_grad_leaf64(&b_data, &shape);
            weighted_sum(linalg_fwd::cross(x, &bb, -1).unwrap().data().unwrap(), &w)
        });
        assert_grad_close64(&grad_a, &num_a, 1e-5, "cross dA vs FD");
        let num_b = fd_grad64(&b_data, &shape, 1e-6, |x| {
            let aa = no_grad_leaf64(&a_data, &shape);
            weighted_sum(linalg_fwd::cross(&aa, x, -1).unwrap().data().unwrap(), &w)
        });
        assert_grad_close64(&grad_b, &num_b, 1e-5, "cross dB vs FD");
    }

    // matrix_norm (Frobenius) — VJP dA = grad * A / ||A||_F
    //   (FunctionsManual.cpp:341 norm_backward p==2).
    #[test]
    fn matrix_norm_public_forward_is_grad_aware_and_matches_fd() {
        let a_data = vec![1.0, -2.0, 0.5, 3.0, -1.5, 2.0];
        let shape = [2usize, 3];

        let a = leaf64(&a_data, &shape);
        let nrm = linalg_fwd::matrix_norm(&a).unwrap();
        assert!(
            nrm.grad_fn().is_some(),
            "matrix_norm forward must attach a grad_fn"
        );
        assert!(nrm.is_scalar());
        nrm.backward().unwrap();
        let analytic = a.grad().unwrap().unwrap().data().unwrap().to_vec();

        let numeric = fd_grad64(&a_data, &shape, 1e-6, |x| {
            linalg_fwd::matrix_norm(x).unwrap().item().unwrap()
        });
        assert_grad_close64(&analytic, &numeric, 1e-5, "matrix_norm vs FD");
    }

    // vector_norm (p=2) — VJP dx = grad * x / ||x||_2
    //   (FunctionsManual.cpp:341 norm_backward p==2 via linalg_vector_norm_backward).
    #[test]
    fn vector_norm_public_forward_is_grad_aware_and_matches_fd() {
        let a_data = vec![3.0, -4.0, 1.0, 2.0];
        let shape = [4usize];

        let a = leaf64(&a_data, &shape);
        let nrm = linalg_fwd::vector_norm(&a, 2.0).unwrap();
        assert!(
            nrm.grad_fn().is_some(),
            "vector_norm(p=2) forward must attach a grad_fn"
        );
        assert!(nrm.is_scalar());
        nrm.backward().unwrap();
        let analytic = a.grad().unwrap().unwrap().data().unwrap().to_vec();

        let numeric = fd_grad64(&a_data, &shape, 1e-6, |x| {
            linalg_fwd::vector_norm(x, 2.0).unwrap().item().unwrap()
        });
        assert_grad_close64(&analytic, &numeric, 1e-5, "vector_norm(p=2) vs FD");
    }

    // -----------------------------------------------------------------------
    // svd backward (REQ-11, #1577) — A.grad vs LIVE torch 2.11.0+cu130 float64
    // for the reduced SVD `torch.linalg.svd(A, full_matrices=False)`.
    //
    // GAUGE FREEDOM (R-DEV-1, same situation eigh #1584 documents): `(U, V)`
    // and `(U·diag(±1), V·diag(±1))` are both valid reduced SVDs of `A`
    // (upstream: `FunctionsManual.cpp:3682-3698`). ferray's faer-backed
    // forward emits its OWN per-column signs, differing matrix-by-matrix from
    // torch's LAPACK signs. So the loss MUST be invariant under joint U/Vh
    // column sign flips for `A.grad` to be well-posed and comparable. We use
    //   L = sum((U*U)*MU) + sum((Vh*Vh)*MV) + sum(S*c)
    // — each `U_ij^2` and `Vh_ij^2` is unchanged under a column sign flip, and
    // `S` is gauge-free. The Python oracle (below) verifies maxdiff == 0 under
    // the sign flip, confirming the loss is gauge-invariant; both torch and
    // ferrotorch must then agree regardless of their differing sign
    // conventions. The MU/MV terms give BOTH `gU` and `gVh` nonzero so the
    // rectangular projector branches (`m>n` in `grad_a_from_gu`, `m<n` in
    // `grad_a_from_gvh`) are exercised for the tall/wide cases.
    //
    // R-CHAR-3 (a): every `torch = [...]` below is a LIVE torch float64 result.
    // Reproduce with (PYTHONPATH=~/.local/.../site-packages):
    //   import torch; torch.set_default_dtype(torch.float64)
    //   A = torch.tensor(<a_data>).reshape(shape).clone().requires_grad_(True)
    //   U,S,Vh = torch.linalg.svd(A, full_matrices=False)
    //   MU = torch.tensor(<mu>).reshape(U.shape)
    //   MV = torch.tensor(<mv>).reshape(Vh.shape)
    //   c  = torch.tensor(<c>).reshape(S.shape)
    //   (((U*U)*MU).sum() + ((Vh*Vh)*MV).sum() + (S*c).sum()).backward()
    //   A.grad.reshape(-1)
    // -----------------------------------------------------------------------

    /// Gauge-invariant SVD loss `sum((U*U)*MU) + sum((Vh*Vh)*MV) + sum(S*c)`
    /// driven through the PUBLIC grad-aware forward `linalg_fwd::svd`. Exercises
    /// all three split nodes (`SvdBackwardU`/`SvdBackwardS`/`SvdBackwardV`).
    fn svd_gauge_invariant_grad(
        a_data: &[f64],
        shape: &[usize],
        mu: &[f64],
        mv: &[f64],
        c: &[f64],
    ) -> Vec<f64> {
        let a = leaf64(a_data, shape);
        let (u, s, vh) = linalg_fwd::svd(&a).unwrap();
        assert!(
            u.grad_fn().is_some() && s.grad_fn().is_some() && vh.grad_fn().is_some(),
            "svd forward must attach grad_fns on all three outputs when input requires_grad"
        );
        let mu_t = no_grad_leaf64(mu, u.shape());
        let mv_t = no_grad_leaf64(mv, vh.shape());
        let c_t = no_grad_leaf64(c, s.shape());
        let usq = crate::grad_fns::arithmetic::mul(&u, &u).unwrap();
        let lu = crate::grad_fns::reduction::sum(
            &crate::grad_fns::arithmetic::mul(&usq, &mu_t).unwrap(),
        )
        .unwrap();
        let vsq = crate::grad_fns::arithmetic::mul(&vh, &vh).unwrap();
        let lv = crate::grad_fns::reduction::sum(
            &crate::grad_fns::arithmetic::mul(&vsq, &mv_t).unwrap(),
        )
        .unwrap();
        let ls =
            crate::grad_fns::reduction::sum(&crate::grad_fns::arithmetic::mul(&s, &c_t).unwrap())
                .unwrap();
        let loss = lu.add_t(&lv).unwrap().add_t(&ls).unwrap();
        loss.backward().unwrap();
        a.grad().unwrap().unwrap().data().unwrap().to_vec()
    }

    // (a) SQUARE 3x3, distinct singular values.
    #[test]
    fn svd_backward_square_3x3_matches_torch() {
        let a = [4.0, 0.5, 0.3, 0.2, 2.5, 0.1, 0.3, 0.15, 1.2];
        let mu = [0.2, -0.5, 0.7, 0.3, 0.1, -0.4, -0.6, 0.8, 0.25];
        let mv = [0.4, 0.1, -0.3, -0.2, 0.5, 0.6, 0.15, -0.7, 0.3];
        let c = [1.3, -0.7, 0.9];
        let g = svd_gauge_invariant_grad(&a, &[3, 3], &mu, &mv, &c);
        // LIVE torch.linalg.svd A.grad (R-CHAR-3 (a)), gauge-invariance verified.
        let torch = [
            1.291_872_488_158_285_7,
            0.254_925_342_453_013_6,
            0.010_080_726_167_581_882,
            0.268_367_455_984_650_4,
            -0.671_927_227_458_943_4,
            -0.184_516_208_432_730_06,
            0.035_875_446_437_466_67,
            -0.159_779_784_325_520_42,
            0.881_087_563_055_650_3,
        ];
        assert_grad_close64(&g, &torch, 1e-6, "svd square 3x3 A.grad vs torch");
    }

    // (b) TALL 4x3 (m > n) — exercises the `grad_a_from_gu` `m>n` projector.
    #[test]
    fn svd_backward_tall_4x3_matches_torch() {
        let a = [3.0, 0.4, 0.2, 0.1, 2.2, 0.3, 0.25, 0.1, 1.5, 0.6, 0.35, 0.4];
        let mu = [
            0.2, -0.5, 0.7, 0.3, 0.1, -0.4, -0.6, 0.8, 0.25, 0.5, -0.2, 0.9,
        ];
        let mv = [0.4, 0.1, -0.3, -0.2, 0.5, 0.6, 0.15, -0.7, 0.3];
        let c = [1.1, -0.6, 0.8];
        let g = svd_gauge_invariant_grad(&a, &[4, 3], &mu, &mv, &c);
        let torch = [
            1.197_882_050_858_392_5,
            0.228_300_430_582_179_41,
            0.022_954_471_103_677_15,
            0.188_714_949_098_595_6,
            -0.389_113_965_271_648_04,
            -0.855_548_366_417_448_8,
            0.101_652_565_998_676_11,
            -0.720_888_340_220_490_3,
            0.455_085_481_936_931_93,
            0.327_980_345_907_320_56,
            -0.247_945_299_633_792_36,
            0.126_868_877_585_850_53,
        ];
        assert_grad_close64(&g, &torch, 1e-6, "svd tall 4x3 A.grad vs torch");
    }

    // (c) WIDE 3x4 (m < n) — exercises the `grad_a_from_gvh` `m<n` projector.
    #[test]
    fn svd_backward_wide_3x4_matches_torch() {
        let a = [
            3.0, 0.4, 0.2, 0.5, 0.1, 2.2, 0.3, 0.15, 0.25, 0.1, 1.5, 0.35,
        ];
        let mu = [0.2, -0.5, 0.7, 0.3, 0.1, -0.4, -0.6, 0.8, 0.25];
        let mv = [
            0.4, 0.1, -0.3, 0.2, -0.2, 0.5, 0.6, -0.1, 0.15, -0.7, 0.3, 0.45,
        ];
        let c = [1.2, -0.5, 0.7];
        let g = svd_gauge_invariant_grad(&a, &[3, 4], &mu, &mv, &c);
        let torch = [
            1.320_835_083_725_02,
            0.155_998_234_431_780_49,
            0.025_100_247_386_388_476,
            0.213_629_738_873_760_5,
            0.184_679_120_382_554_57,
            -0.354_909_223_025_186_57,
            -0.733_393_775_260_972_9,
            -0.187_334_699_726_463_5,
            0.105_428_571_937_452_24,
            -0.656_612_751_778_449_9,
            0.419_718_259_999_602_1,
            0.090_484_769_926_790_86,
        ];
        assert_grad_close64(&g, &torch, 1e-6, "svd wide 3x4 A.grad vs torch");
    }

    /// (d) grad through S only (`gU = gVh = None`). Singular values are
    /// gauge-free and smooth in `A`, so this matches torch exactly with NO
    /// gauge caveat. Loss = `sum(S*c)`; only `SvdBackwardS` fires.
    fn svd_s_only_grad(a_data: &[f64], shape: &[usize], c: &[f64]) -> Vec<f64> {
        let a = leaf64(a_data, shape);
        let (_u, s, _vh) = linalg_fwd::svd(&a).unwrap();
        let c_t = no_grad_leaf64(c, s.shape());
        let loss =
            crate::grad_fns::reduction::sum(&crate::grad_fns::arithmetic::mul(&s, &c_t).unwrap())
                .unwrap();
        loss.backward().unwrap();
        a.grad().unwrap().unwrap().data().unwrap().to_vec()
    }

    #[test]
    fn svd_backward_s_only_square_3x3_matches_torch() {
        let a = [4.0, 0.5, 0.3, 0.2, 2.5, 0.1, 0.3, 0.15, 1.2];
        let c = [1.3, -0.7, 0.9];
        let g = svd_s_only_grad(&a, &[3, 3], &c);
        // LIVE torch.linalg.svd, loss = (S*c).sum() (R-CHAR-3 (a)).
        let torch = [
            1.194_931_571_680_387_4,
            0.448_845_297_858_673_9,
            0.054_245_497_692_962_86,
            0.420_930_628_143_998_4,
            -0.597_262_866_179_166_1,
            -0.061_889_902_447_841_184,
            0.062_108_766_246_365_944,
            -0.056_066_490_205_725_995,
            0.901_663_478_439_573_2,
        ];
        assert_grad_close64(&g, &torch, 1e-6, "svd S-only 3x3 A.grad vs torch");
    }

    #[test]
    fn svd_backward_s_only_tall_4x3_matches_torch() {
        let a = [3.0, 0.4, 0.2, 0.1, 2.2, 0.3, 0.25, 0.1, 1.5, 0.6, 0.35, 0.4];
        let c = [1.1, -0.6, 0.8];
        let g = svd_s_only_grad(&a, &[4, 3], &c);
        let torch = [
            0.868_920_516_856_988_9,
            0.548_143_680_162_881_7,
            0.120_098_994_041_652_57,
            0.505_486_391_476_610_9,
            -0.353_954_199_304_621,
            -0.230_198_242_257_246_6,
            0.130_796_388_653_387_1,
            -0.223_191_598_632_373_5,
            0.745_402_308_929_012_2,
            0.242_254_390_771_033_14,
            0.007_115_371_210_751_172,
            0.158_164_374_952_315_28,
        ];
        assert_grad_close64(&g, &torch, 1e-6, "svd S-only tall 4x3 A.grad vs torch");
    }

    // -----------------------------------------------------------------------
    // householder_product backward (REQ-26, #1345) — verified vs LIVE torch
    // 2.11.0 float64 (R-CHAR-3 (a)). Reproduce each oracle with:
    //   import torch; torch.set_default_dtype(torch.float64)
    //   V = torch.tensor(<v_data>).reshape(v_shape).requires_grad_(True)
    //   tau = torch.tensor(<tau_data>).requires_grad_(True)
    //   Q = torch.linalg.householder_product(V, tau)   # shape [m, k]
    //   Q.backward(torch.tensor(<g_data>).reshape(Q.shape))
    //   V.grad.reshape(-1); tau.grad.reshape(-1)
    // The ferrotorch side drives the PUBLIC grad-aware forward
    // `linalg_fwd::householder_product` through a `sum(Q * g)` loss (so
    // `dQ == g`), exercising `HouseholderProductBackward` end-to-end.
    // -----------------------------------------------------------------------

    /// Drive `linalg_fwd::householder_product` (the public forward) on
    /// grad-tracking `(V, tau)` leaves through the linear loss `sum(Q * g)`,
    /// returning `(V.grad, tau.grad)`.
    fn hh_grad(
        v_data: &[f64],
        v_shape: &[usize],
        tau_data: &[f64],
        g_data: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        let v = leaf64(v_data, v_shape);
        let tau = leaf64(tau_data, &[tau_data.len()]);
        let q = linalg_fwd::householder_product(&v, &tau).unwrap();
        let m = v_shape[0];
        let k = v_shape[1];
        assert_eq!(q.shape(), &[m, k], "torch returns the leading k columns");
        assert!(
            q.grad_fn().is_some(),
            "householder_product forward must attach a grad_fn when inputs require grad"
        );
        let g = no_grad_leaf64(g_data, &[m, k]);
        let loss =
            crate::grad_fns::reduction::sum(&crate::grad_fns::arithmetic::mul(&q, &g).unwrap())
                .unwrap();
        loss.backward().unwrap();
        let gv = v.grad().unwrap().unwrap().data().unwrap().to_vec();
        let gt = tau.grad().unwrap().unwrap().data().unwrap().to_vec();
        (gv, gt)
    }

    // (a) SQUARE 3x3, k=3.
    #[test]
    #[allow(
        clippy::excessive_precision,
        reason = "literals are verbatim LIVE torch float64 repr() oracle values; \
                  trailing digits beyond f64 precision are kept for provenance"
    )]
    fn householder_product_backward_square_3x3_matches_torch() {
        let v = [1.0, 0.2, 0.3, 0.5, 1.0, 0.1, 0.3, 0.15, 1.0];
        let tau = [0.4, 0.5, 0.6];
        let g = [0.2, -0.5, 0.7, 0.3, 0.1, -0.4, -0.6, 0.8, 0.25];
        let (gv, gt) = hh_grad(&v, &[3, 3], &tau, &g);
        // LIVE torch.linalg.householder_product V.grad / tau.grad.
        let torch_gv = [
            0.0,
            0.0,
            0.0,
            -0.063_616_000_000_000_034,
            0.0,
            0.0,
            0.059_570_000_000_000_04,
            -0.320_460_000_000_000_02,
            0.0,
        ];
        let torch_gt = [
            -0.181_823_749_999_999_98,
            -0.236_509_000_000_000_02,
            -0.217_588_749_999_999_94,
        ];
        assert_grad_close64(
            &gv,
            &torch_gv,
            1e-9,
            "householder square 3x3 V.grad vs torch",
        );
        assert_grad_close64(
            &gt,
            &torch_gt,
            1e-9,
            "householder square 3x3 tau.grad vs torch",
        );
    }

    // (b) TALL 4x3, k=3 (m > k == cols).
    #[test]
    #[allow(
        clippy::excessive_precision,
        reason = "literals are verbatim LIVE torch float64 repr() oracle values; \
                  trailing digits beyond f64 precision are kept for provenance"
    )]
    fn householder_product_backward_tall_4x3_matches_torch() {
        let v = [1.0, 0.2, 0.3, 0.5, 1.0, 0.1, 0.3, 0.15, 1.0, 0.6, 0.35, 0.4];
        let tau = [0.4, 0.5, 0.6];
        let g = [
            0.2, -0.5, 0.7, 0.3, 0.1, -0.4, -0.6, 0.8, 0.25, 0.4, -0.2, 0.9,
        ];
        let (gv, gt) = hh_grad(&v, &[4, 3], &tau, &g);
        let torch_gv = [
            0.0,
            0.0,
            0.0,
            -0.066_642_400_000_000_046,
            0.0,
            0.0,
            0.013_191_200_000_000_153,
            -0.341_559_599_999_999_63,
            0.0,
            -0.062_754_800_000_000_027,
            0.021_881_199_999_999_948,
            -0.419_784_150_000_000_02,
        ];
        let torch_gt = [
            -0.352_916_900_000_000_03,
            -0.258_881_520_000_000_09,
            -0.424_873_349_999_999_48,
        ];
        assert_grad_close64(&gv, &torch_gv, 1e-9, "householder tall 4x3 V.grad vs torch");
        assert_grad_close64(
            &gt,
            &torch_gt,
            1e-9,
            "householder tall 4x3 tau.grad vs torch",
        );
    }

    // (c) TALL 4x2, k=2 < m (truncated product — exercises k < cols active set).
    #[test]
    #[allow(
        clippy::excessive_precision,
        reason = "literals are verbatim LIVE torch float64 repr() oracle values; \
                  trailing digits beyond f64 precision are kept for provenance"
    )]
    fn householder_product_backward_tall_4x2_matches_torch() {
        let v = [1.0, 0.2, 0.5, 1.0, 0.3, 0.15, 0.6, 0.35];
        let tau = [0.4, 0.5];
        let g = [0.2, -0.5, 0.3, 0.1, -0.6, 0.8, 0.4, -0.2];
        let (gv, gt) = hh_grad(&v, &[4, 2], &tau, &g);
        let torch_gv = [
            0.0,
            0.0,
            -0.058_900_000_000_000_063,
            0.0,
            0.190_900_000_000_000_1,
            -0.419_799_999_999_999_84,
            -0.173_300_000_000_000_04,
            0.060_399_999_999_999_961,
        ];
        let torch_gt = [-0.369_575_000_000_000_04, -0.249_660_000_000_000_1];
        assert_grad_close64(&gv, &torch_gv, 1e-9, "householder tall 4x2 V.grad vs torch");
        assert_grad_close64(
            &gt,
            &torch_gt,
            1e-9,
            "householder tall 4x2 tau.grad vs torch",
        );
    }

    // V-only / tau-only grad paths: when only one input requires grad, the
    // backward returns `None` for the other (no spurious accumulation).
    #[test]
    fn householder_product_backward_single_input_grad() {
        let v_data = [1.0, 0.2, 0.3, 0.5, 1.0, 0.1, 0.3, 0.15, 1.0];
        let tau_data = [0.4, 0.5, 0.6];
        let g_data = [0.2, -0.5, 0.7, 0.3, 0.1, -0.4, -0.6, 0.8, 0.25];

        // tau-only: V is a no-grad leaf.
        let v = no_grad_leaf64(&v_data, &[3, 3]);
        let tau = leaf64(&tau_data, &[3]);
        let q = linalg_fwd::householder_product(&v, &tau).unwrap();
        let g = no_grad_leaf64(&g_data, &[3, 3]);
        let loss =
            crate::grad_fns::reduction::sum(&crate::grad_fns::arithmetic::mul(&q, &g).unwrap())
                .unwrap();
        loss.backward().unwrap();
        assert!(v.grad().unwrap().is_none(), "V must carry no grad");
        let gt = tau.grad().unwrap().unwrap().data().unwrap().to_vec();
        let torch_gt = [
            -0.181_823_749_999_999_98,
            -0.236_509_000_000_000_02,
            -0.217_588_749_999_999_94,
        ];
        assert_grad_close64(
            &gt,
            &torch_gt,
            1e-9,
            "householder tau-only tau.grad vs torch",
        );
    }

    // -----------------------------------------------------------------------
    // eigvals / eig backward (REQ-12 / REQ-14, #1345) — verified vs LIVE torch
    // 2.11.0 float64 (R-CHAR-3 (a)). The complex eigenvalues/eigenvectors are
    // encoded as trailing-dim-2 real tensors `[re, im]`; the downstream loss is
    // a REAL scalar of those real/imag parts, so the cotangent flowing into the
    // GradFn is the `[n,2]` / `[n,n,2]` real tensor torch's conjugate-Wirtinger
    // convention encodes as `re + i*im`. Reproduce each oracle with:
    //   import torch; torch.set_default_dtype(torch.float64)
    //   A = torch.tensor(<a>).reshape(n,n).clone().requires_grad_(True)
    //   # eigvals:
    //   L = torch.linalg.eigvals(A)
    //   ((L.real*cr).sum() + (L.imag*ci).sum()).backward()
    //   # eig (phase-invariant V loss + eigenvalue term):
    //   L,V = torch.linalg.eig(A)
    //   (((V.real**2+V.imag**2)*MR).sum()+(L.real*cr).sum()+(L.imag*ci).sum())
    //       .backward()
    //   A.grad.reshape(-1)
    // The ferrotorch side drives the PUBLIC grad-aware forwards
    // `linalg_fwd::eigvals` / `linalg_fwd::eig` through the matching loss,
    // exercising `EigvalsBackward` / `EigBackwardW` / `EigBackwardV`. The eig
    // loss is PHASE-INVARIANT (`|V_ij|^2` is unchanged by a per-column phase),
    // so `A.grad` is well-posed and comparable to torch even though ferray's
    // faer column phase may differ from LAPACK's (gauge note, R-DEV-1).
    // -----------------------------------------------------------------------

    /// Build a `[n,2]` weight tensor with re-slot `cr[k]`, im-slot `ci[k]`,
    /// element-wise multiply into the complex `[n,2]` eigenvalues `w` and sum:
    /// `sum_k (re(w_k)*cr_k + im(w_k)*ci_k)`.
    fn eigval_linear_loss(w: &Tensor<f64>, cr: &[f64], ci: &[f64]) -> Tensor<f64> {
        let n = cr.len();
        let mut wt = vec![0.0; n * 2];
        for k in 0..n {
            wt[2 * k] = cr[k];
            wt[2 * k + 1] = ci[k];
        }
        let wts = no_grad_leaf64(&wt, &[n, 2]);
        crate::grad_fns::reduction::sum(&crate::grad_fns::arithmetic::mul(w, &wts).unwrap())
            .unwrap()
    }

    /// Phase-invariant V loss `sum((re^2+im^2) * MR[i,j])` driven on the complex
    /// `[n,n,2]` eigenvectors `v`. `V*V` yields `[re^2, im^2]` per element; the
    /// weight tensor sets BOTH re/im slots to `MR[i,j]` so the sum collapses to
    /// `sum((re^2+im^2)*MR)`.
    fn eigvec_phase_invariant_loss(v: &Tensor<f64>, mr: &[f64], n: usize) -> Tensor<f64> {
        let mut wt = vec![0.0; n * n * 2];
        for idx in 0..n * n {
            wt[2 * idx] = mr[idx];
            wt[2 * idx + 1] = mr[idx];
        }
        let wts = no_grad_leaf64(&wt, &[n, n, 2]);
        let vsq = crate::grad_fns::arithmetic::mul(v, v).unwrap();
        crate::grad_fns::reduction::sum(&crate::grad_fns::arithmetic::mul(&vsq, &wts).unwrap())
            .unwrap()
    }

    /// Drive the PUBLIC grad-aware `linalg_fwd::eigvals` through the linear
    /// eigenvalue loss and return `A.grad`.
    fn eigvals_grad(a_data: &[f64], n: usize, cr: &[f64], ci: &[f64]) -> Vec<f64> {
        let a = leaf64(a_data, &[n, n]);
        let w = linalg_fwd::eigvals(&a).unwrap();
        assert!(
            w.grad_fn().is_some(),
            "eigvals forward must attach a grad_fn when input requires_grad"
        );
        let loss = eigval_linear_loss(&w, cr, ci);
        loss.backward().unwrap();
        a.grad().unwrap().unwrap().data().unwrap().to_vec()
    }

    /// Drive the PUBLIC grad-aware `linalg_fwd::eig` through the phase-invariant
    /// V loss + eigenvalue linear term and return `A.grad`. Exercises BOTH the
    /// `EigBackwardW` (`gL`) and `EigBackwardV` (`gV`) split nodes.
    fn eig_grad(a_data: &[f64], n: usize, mr: &[f64], cr: &[f64], ci: &[f64]) -> Vec<f64> {
        let a = leaf64(a_data, &[n, n]);
        let (w, v) = linalg_fwd::eig(&a).unwrap();
        assert!(
            w.grad_fn().is_some() && v.grad_fn().is_some(),
            "eig forward must attach grad_fns on both outputs when input requires_grad"
        );
        let lv = eigvec_phase_invariant_loss(&v, mr, n);
        let lw = eigval_linear_loss(&w, cr, ci);
        let loss = lv.add_t(&lw).unwrap();
        loss.backward().unwrap();
        a.grad().unwrap().unwrap().data().unwrap().to_vec()
    }

    // (a) EIGVALS — 3x3 REAL distinct eigenvalues (upper-triangular A, V real).
    #[test]
    fn eigvals_backward_real_3x3_matches_torch() {
        let a = [2.0, 0.5, 0.3, 0.0, 3.0, 0.4, 0.0, 0.0, 5.0];
        let g = eigvals_grad(&a, 3, &[1.3, -0.7, 0.9], &[0.4, 0.6, -0.2]);
        // LIVE torch.linalg.eigvals A.grad (R-CHAR-3 (a)). L = [2, 3, 5].
        let torch = [
            1.3,
            0.0,
            0.0,
            -1.0,
            -0.7,
            0.0,
            0.146_666_666_666_666_7,
            0.32,
            0.9,
        ];
        assert_grad_close64(&g, &torch, 1e-6, "eigvals real 3x3 A.grad vs torch");
    }

    // (b) EIGVALS — 2x2 COMPLEX-conjugate eigenvalue pair (V genuinely complex).
    //     A = [[1,-1],[1,1]] has eigenvalues 1 ± i. This is the essential
    //     complex-arithmetic case.
    #[test]
    fn eigvals_backward_complex_pair_2x2_matches_torch() {
        let a = [1.0, -1.0, 1.0, 1.0];
        let g = eigvals_grad(&a, 2, &[1.3, -0.7], &[0.4, 0.6]);
        // LIVE torch.linalg.eigvals A.grad (R-CHAR-3 (a)). L = [1+i, 1-i].
        let torch = [
            0.300_000_000_000_000_04,
            0.099_999_999_999_999_96,
            -0.099_999_999_999_999_96,
            0.300_000_000_000_000_04,
        ];
        assert_grad_close64(&g, &torch, 1e-6, "eigvals complex-pair 2x2 A.grad vs torch");
    }

    // (c) EIG — 3x3 REAL distinct eigenvalues, BOTH gL and gV active.
    #[test]
    fn eig_backward_real_3x3_matches_torch() {
        let a = [2.0, 0.5, 0.3, 0.0, 3.0, 0.4, 0.0, 0.0, 5.0];
        let mr = [0.2, -0.5, 0.7, 0.3, 0.1, -0.4, -0.6, 0.8, 0.25];
        let g = eig_grad(&a, 3, &mr, &[1.3, -0.7, 0.9], &[0.4, 0.6, -0.2]);
        // LIVE torch.linalg.eig A.grad (R-CHAR-3 (a)), phase-invariant V loss.
        let torch = [
            1.113_232_681_307_817_3,
            -0.376_150_978_038_274_1,
            0.039_245_109_808_629_34,
            -0.918_649_389_167_431_8,
            -0.529_974_083_751_147_4,
            -0.109_870_418_755_737_64,
            0.158_498_853_659_110_7,
            0.342_548_280_488_666_1,
            0.916_741_402_443_330_4,
        ];
        assert_grad_close64(&g, &torch, 1e-6, "eig real 3x3 A.grad vs torch");
    }

    // (d) EIG — 2x2 COMPLEX-conjugate eigenvalue pair, BOTH gL and gV active.
    //     The essential complex-eigenvector case.
    #[test]
    fn eig_backward_complex_pair_2x2_matches_torch() {
        let a = [1.0, -1.0, 1.0, 1.0];
        let mr = [0.5, -0.3, 0.2, 0.8];
        let g = eig_grad(&a, 2, &mr, &[1.3, -0.7], &[0.4, 0.6]);
        // LIVE torch.linalg.eig A.grad (R-CHAR-3 (a)), phase-invariant V loss.
        let torch = [
            0.300_000_000_000_000_04,
            0.3,
            0.099_999_999_999_999_96,
            0.300_000_000_000_000_04,
        ];
        assert_grad_close64(&g, &torch, 1e-6, "eig complex-pair 2x2 A.grad vs torch");
    }

    // (e) EIG — V-ONLY (gL = 0): drive ONLY the phase-invariant V loss so the
    //     `EigBackwardV` (`gV`) split node is exercised in isolation.
    #[test]
    fn eig_backward_v_only_complex_pair_2x2_matches_torch() {
        let a = [1.0, -1.0, 1.0, 1.0];
        let mr = [0.5, -0.3, 0.2, 0.8];
        let a_t = leaf64(&a, &[2, 2]);
        let (w, v) = linalg_fwd::eig(&a_t).unwrap();
        assert!(w.grad_fn().is_some() && v.grad_fn().is_some());
        let loss = eigvec_phase_invariant_loss(&v, &mr, 2);
        loss.backward().unwrap();
        let g = a_t.grad().unwrap().unwrap().data().unwrap().to_vec();
        // LIVE torch.linalg.eig A.grad with V-only loss (R-CHAR-3 (a)).
        let torch = [0.0, 0.2, 0.2, 0.0];
        assert_grad_close64(&g, &torch, 1e-6, "eig V-only 2x2 A.grad vs torch");
    }

    // (f) eig / eigvals attach NO grad_fn when grad is disabled (no_grad) or the
    //     input does not require grad — matching the forward-only contract.
    #[test]
    fn eig_eigvals_no_grad_when_input_is_plain() {
        let a = no_grad_leaf64(&[1.0, -1.0, 1.0, 1.0], &[2, 2]);
        let w = linalg_fwd::eigvals(&a).unwrap();
        assert!(
            w.grad_fn().is_none(),
            "eigvals on a non-requires_grad leaf must not attach a grad_fn"
        );
        let (w2, v2) = linalg_fwd::eig(&a).unwrap();
        assert!(
            w2.grad_fn().is_none() && v2.grad_fn().is_none(),
            "eig on a non-requires_grad leaf must not attach grad_fns"
        );
    }
}
