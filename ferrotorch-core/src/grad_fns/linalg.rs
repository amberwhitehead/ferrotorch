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
//! | REQ-5 (`addmm`) | NOT-STARTED | no `AddmmBackward` or `addmm_differentiable` in this file; the `linear_fused` shape covers only the `A @ W^T + bias` slice. Blocker #1345. |
//! | REQ-6 (`addbmm`) | NOT-STARTED | not present. Blocker #1345. |
//! | REQ-7 (`baddbmm`) | NOT-STARTED | not present. Blocker #1345. |
//! | REQ-8 (`addmv`) | NOT-STARTED | not present. Blocker #1345. |
//! | REQ-9 (`addr`) | NOT-STARTED | not present. Blocker #1345. |
//! | REQ-10 (`linalg.solve`) | SHIPPED | `LinalgSolveBackward` + `solve_differentiable` (VJP `gB = A^-T @ gX`, `gA = -gB @ X^T` per `FunctionsManual.cpp:6160`); FD-verified `tests/divergence_linalg_grad_audit.rs:solve_backward_*`; non-test consumer `tools/parity-sweep/runner/src/main.rs` `"linalg.solve"` arm (parity 24/24 non-skipped, 0 failed). Blocker #1345. |
//! | REQ-11 (`linalg.svd`) | NOT-STARTED | forward exists; no `LinalgSvdBackward`. Blocker #1345. |
//! | REQ-12 (`linalg.eig`) | NOT-STARTED | forward exists; no backward. Blocker #1345. |
//! | REQ-13 (`linalg.eigh`) | NOT-STARTED | forward exists; no backward. Blocker #1345. |
//! | REQ-14 (`linalg.eigvals`) | NOT-STARTED | forward exists; no backward. Blocker #1345. |
//! | REQ-15 (`linalg.eigvalsh`) | NOT-STARTED | forward exists; no backward. Blocker #1345. |
//! | REQ-16 (`linalg.qr`) | SHIPPED | `QrBackwardQ`/`QrBackwardR` + `qr_differentiable` (reduced, m≥n; real `linalg_qr_backward` VJP split across the Q/R outputs, accumulated into `A.grad`) per `FunctionsManual.cpp:4166`; FD-verified `grad_fns::linalg::tests::qr_backward_matches_finite_difference_square` and `qr_backward_q_only_and_r_only`; non-test consumer: the grad-aware `crate::linalg::qr` forward delegates here when grad is enabled. Blocker #1345. |
//! | REQ-17 (`linalg.cholesky`) | SHIPPED | `CholeskyBackward` + `cholesky_differentiable` (Phi-symmetrisation VJP `L^{-T} Φ(tril(L^T gL)) L^{-1}`) per `FunctionsManual.cpp:2048`; FD-verified `grad_fns::linalg::tests::cholesky_backward_matches_finite_difference` (symmetric-FD + symmetry check); non-test consumer: the grad-aware `crate::linalg::cholesky` forward delegates here when grad is enabled. Blocker #1345. |
//! | REQ-18 (`linalg.inv`) | SHIPPED | `LinalgInvBackward` + `inv_differentiable` (VJP `dA = -Y^T @ grad @ Y^T` per `derivatives.yaml:917`); FD-verified `tests/divergence_linalg_grad_audit.rs:inv_backward_matches_finite_difference`; non-test consumer `tools/parity-sweep/runner/src/main.rs` `"linalg.inv"` arm. Blocker #1345. |
//! | REQ-19 (`linalg.pinv`) | NOT-STARTED | forward exists; no backward. Blocker #1345. |
//! | REQ-20 (`linalg.det`) | SHIPPED | `LinalgDetBackward` + `det_differentiable` (VJP `dA = det * grad * inv(A)^T` per `FunctionsManual.cpp:4373` invertible branch); FD-verified `tests/divergence_linalg_grad_audit.rs:det_backward_matches_finite_difference`; non-test consumer `tools/parity-sweep/runner/src/main.rs` `"linalg.det"` arm. Blocker #1345. |
//! | REQ-21 (`linalg.slogdet`) | SHIPPED | `SlogdetBackward` + `slogdet_differentiable` (real-case VJP `dA = grad_logabsdet * inv(A)^T`, attached to the differentiable `logabsdet` output; `sign` is non-diff) per `FunctionsManual.cpp:4471` + `derivatives.yaml:559`; FD-verified `grad_fns::linalg::tests::slogdet_backward_matches_finite_difference`; non-test consumer: the grad-aware `crate::linalg::slogdet` forward delegates here when grad is enabled. Blocker #1345. |
//! | REQ-22 (`linalg.lstsq`) | NOT-STARTED | forward exists; no backward. Blocker #1345. |
//! | REQ-23 (`linalg.norm`) | NOT-STARTED | forward exists; no backward. Blocker #1345. |
//! | REQ-24 (`linalg.matrix_rank`) | NOT-STARTED | forward exists; no backward. Blocker #1345. |
//! | REQ-25 (`linalg.cross`) | NOT-STARTED | forward exists; no backward. Blocker #1345. |
//! | REQ-26 (`linalg.householder_product`) | NOT-STARTED | forward exists; no backward. Blocker #1345. |
//! | REQ-27 (`linalg.lu`) | NOT-STARTED | forward exists; no backward. Blocker #1345. |
//! | REQ-28 (`linalg.lu_factor`) | NOT-STARTED | forward exists; no backward. Blocker #1345. |
//! | REQ-29 (`trace`) | SHIPPED | `TraceBackward` + `trace_differentiable` (VJP `dA = grad * I` per `derivatives.yaml:1785`), forward `crate::linalg::trace`; FD-verified `tests/divergence_linalg_grad_audit.rs:trace_backward_matches_finite_difference`; non-test consumer `tools/parity-sweep/runner/src/main.rs` `"trace"` arm (parity 8/8, 0 failed). Blocker #1345. |
//! | REQ-30 (`diagonal`) | NOT-STARTED | forward-only `diagonal` exists in `crate::linalg`; no autograd. Blocker #1345. |
//! | REQ-31 (`diag`) | NOT-STARTED | forward-only `diag` exists in `ops::tensor_ops`; no autograd. Blocker #1345. |
//! | REQ-32 (`tril`) | NOT-STARTED | forward-only `tril` exists in `ops::tensor_ops`; no autograd. Blocker #1345. |
//! | REQ-33 (`triu`) | NOT-STARTED | forward-only `triu` exists in `ops::tensor_ops`; no autograd. Blocker #1345. |
//! | REQ-34 (`kron`) | NOT-STARTED | no `kron` anywhere. Blocker #1345. |
//! | REQ-35 (`outer`) | SHIPPED | `OuterBackward` + `outer_differentiable` (VJP `da = grad @ b`, `db = grad^T @ a` per `derivatives.yaml:275-276`), forward `crate::linalg::outer`; FD-verified `tests/divergence_linalg_grad_audit.rs:outer_backward_matches_finite_difference`; non-test consumer `tools/parity-sweep/runner/src/main.rs` `"outer"` arm (parity 8/8, 0 failed). Blocker #1345. |

use std::any::TypeId;
use std::sync::Arc;

use crate::autograd::autocast_ops::{AutocastCategory, autocast_guard};
use crate::autograd::no_grad::is_grad_enabled;
use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::gpu_dispatch::gpu_backend;
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

/// GPU-native matmul backward for f32 tensors.
/// dA = grad_C @ B^T, dB = A^T @ grad_C — all on GPU, no CPU roundtrip.
fn mm_backward_gpu<T: Float>(grad_output: &Tensor<T>, a: &Tensor<T>, b: &Tensor<T>) -> GradPair<T> {
    let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
    let go_h = grad_output.gpu_handle()?;
    let m = grad_output.shape()[0];
    let n = grad_output.shape()[1];
    let f64_path = is_f64::<T>();

    let grad_a = if a.requires_grad() {
        let k = b.shape()[0];
        let b_h = b.gpu_handle()?;
        let (bt_h, result_h) = if f64_path {
            let bt = backend.transpose_2d_f64(b_h, k, n)?;
            let r = backend.matmul_f64(go_h, &bt, m, n, k)?;
            (bt, r)
        } else {
            let bt = backend.transpose_2d_f32(b_h, k, n)?;
            let r = backend.matmul_f32(go_h, &bt, m, n, k)?;
            (bt, r)
        };
        let _ = bt_h;
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
        let (at_h, result_h) = if f64_path {
            let at = backend.transpose_2d_f64(a_h, m, k)?;
            let r = backend.matmul_f64(&at, go_h, k, m, n)?;
            (at, r)
        } else {
            let at = backend.transpose_2d_f32(a_h, m, k)?;
            let r = backend.matmul_f32(&at, go_h, k, m, n)?;
            (at, r)
        };
        let _ = at_h;
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
        // GPU-native path for f32/f64.
        if grad_output.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
            let (ga, gb) = mm_backward_gpu(grad_output, &self.a, &self.b)?;
            return Ok(vec![ga, gb]);
        }

        if grad_output.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda { op: "MmBackward" });
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
        // dA = outer(grad_y, x) = matmul(grad_y.reshape(m,1), x.reshape(1,k)) — rank-1 mm on GPU.
        // dx = A^T @ grad_y  — cuBLAS gemv with transpose flag via mv_f32/f64.
        if grad_output.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let m = self.a.shape()[0];
            let k = self.a.shape()[1];
            let f64_path = is_f64::<T>();

            let grad_a = if self.a.requires_grad() {
                let go_h = grad_output.gpu_handle()?;
                let x_h = self.x.gpu_handle()?;
                // outer(grad_y, x): treat grad_y as (m,1) and x as (1,k) → matmul gives (m,k).
                let result_h = if f64_path {
                    backend.matmul_f64(go_h, x_h, m, 1, k)?
                } else {
                    backend.matmul_f32(go_h, x_h, m, 1, k)?
                };
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
                // dx = A^T @ grad_y: transpose A (m,k) → (k,m), then mv((k,m), grad_y[m]) → (k,).
                let at_h = if f64_path {
                    backend.transpose_2d_f64(a_h, m, k)?
                } else {
                    backend.transpose_2d_f32(a_h, m, k)?
                };
                // mv_f32/f64(at, grad_y, rows=k, cols=m): y[k] = at[k,m] @ grad_y[m].
                let result_h = if f64_path {
                    backend.mv_f64(&at_h, go_h, k, m)?
                } else {
                    backend.mv_f32(&at_h, go_h, k, m)?
                };
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

        if grad_output.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda { op: "MvBackward" });
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
        // §3 GPU-native path: da = grad_s * b, db = grad_s * a — elementwise scale on GPU.
        // grad_s is a scalar (1-element buffer); scale_f32/f64 broadcasts it via PTX.
        if grad_output.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            // Extract scalar from the 1-element GPU buffer via D2H transfer.
            // `.item()` calls `.data()` which returns GpuTensorNotAccessible for CUDA
            // tensors; we must copy the 1-element scalar to CPU first.
            let s: T = grad_output.cpu()?.item()?;
            let f64_path = is_f64::<T>();

            let grad_a = if self.a.requires_grad() {
                let b_h = self.b.gpu_handle()?;
                let result_h = if f64_path {
                    let s64 = <T as num_traits::ToPrimitive>::to_f64(&s).unwrap();
                    backend.scale_f64(b_h, s64)?
                } else {
                    let s32 = <T as num_traits::ToPrimitive>::to_f32(&s).unwrap();
                    backend.scale_f32(b_h, s32)?
                };
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
                let result_h = if f64_path {
                    let s64 = <T as num_traits::ToPrimitive>::to_f64(&s).unwrap();
                    backend.scale_f64(a_h, s64)?
                } else {
                    let s32 = <T as num_traits::ToPrimitive>::to_f32(&s).unwrap();
                    backend.scale_f32(a_h, s32)?
                };
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

        if grad_output.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda { op: "DotBackward" });
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
                // §3 GPU-native path: da = mv(B, grad_y) via mv; dB = outer(a, grad_y) via matmul.
                if grad_output.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
                    let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
                    let k = self.a.numel();
                    let n = grad_output.numel();
                    let f64_path = is_f64::<T>();

                    let grad_a = if self.a.requires_grad() {
                        // da = B @ grad_y: B is (K,N), grad_y is (N,) → result (K,).
                        // mv_f32/f64(B, grad_y, rows=K, cols=N).
                        let b_h = self.b.gpu_handle()?;
                        let go_h = grad_output.gpu_handle()?;
                        let result_h = if f64_path {
                            backend.mv_f64(b_h, go_h, k, n)?
                        } else {
                            backend.mv_f32(b_h, go_h, k, n)?
                        };
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
                        let result_h = if f64_path {
                            backend.matmul_f64(a_h, go_h, k, 1, n)?
                        } else {
                            backend.matmul_f32(a_h, go_h, k, 1, n)?
                        };
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

                if grad_output.is_cuda() || self.a.is_cuda() || self.b.is_cuda() {
                    return Err(FerrotorchError::NotImplementedOnCuda {
                        op: "MatmulBackward(vm)",
                    });
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
        let n_mats: usize = shape[..nd - 2].iter().product::<usize>().max(1);
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

        // Compute target total size.
        let target_size: usize = target.iter().product::<usize>().max(1);
        let mut result = vec![<T as num_traits::Zero>::zero(); target_size];

        let grad_total: usize = grad_shape.iter().product::<usize>().max(1);

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

    let grad_a = if a.requires_grad() {
        // grad_A = matmul(grad_C, B^T) reduced to A's shape.
        let bt = swap_last_two(b)?;
        let full_grad = linalg::matmul(grad_output, &bt)?;
        Some(reduce_to_shape(full_grad, a.shape())?)
    } else {
        None
    };

    let grad_b = if b.requires_grad() {
        // grad_B = matmul(A^T, grad_C) reduced to B's shape.
        let at = swap_last_two(a)?;
        let full_grad = linalg::matmul(&at, grad_output)?;
        Some(reduce_to_shape(full_grad, b.shape())?)
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
    if a.device() != b.device() {
        return Err(FerrotorchError::DeviceMismatch {
            expected: a.device(),
            got: b.device(),
        });
    }

    // Materialize non-contiguous views before linalg ops.
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

    if a.is_cuda() {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let m = a.shape()[0];
        let k = a.shape()[1];
        let n = b.shape()[1];
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
        let m = a.shape()[0];
        let k = a.shape()[1];
        let n = b.shape()[1];

        if k != b.shape()[0] {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "mm: inner dimensions mismatch: ({},{}) @ ({},{})",
                    m,
                    k,
                    b.shape()[0],
                    n
                ),
            });
        }

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
        // GPU-native path for f32/f64.
        if grad_output.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let go_h = grad_output.gpu_handle()?;
            let m = grad_output.shape()[0];
            let n = grad_output.shape()[1];
            let f64_path = is_f64::<T>();

            let grad_a = if self.a.requires_grad() {
                let k = self.b.shape()[1];
                let b_h = self.b.gpu_handle()?;
                let result_h = if f64_path {
                    backend.matmul_f64(go_h, b_h, m, n, k)?
                } else {
                    backend.matmul_f32(go_h, b_h, m, n, k)?
                };
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
                let (got_h, result_h) = if f64_path {
                    let got = backend.transpose_2d_f64(go_h, m, n)?;
                    let r = backend.matmul_f64(&got, a_h, n, m, k)?;
                    (got, r)
                } else {
                    let got = backend.transpose_2d_f32(go_h, m, n)?;
                    let r = backend.matmul_f32(&got, a_h, n, m, k)?;
                    (got, r)
                };
                let _ = got_h;
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

        if grad_output.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda { op: "MmBtBackward" });
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
    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[0];

    if b.ndim() != 2 || b.shape()[1] != k {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "mm_bt: A is ({},{}) but B is {:?} (expected ({},{}))",
                m,
                k,
                b.shape(),
                n,
                k
            ),
        });
    }

    // GPU path: transpose B then matmul.
    if a.is_cuda() {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        // Dtype-aware GPU dispatch (#800): mirror the backward, which already
        // branches on `is_f64::<T>()`. The previous unconditional f32 path
        // returned "GPU handle does not contain a CudaBuffer<f32>" for f64.
        let handle = if is_f32::<T>() {
            let bt = backend.transpose_2d_f32(b.gpu_handle()?, n, k)?;
            backend.matmul_f32(a.gpu_handle()?, &bt, m, k, n)?
        } else if is_f64::<T>() {
            let bt = backend.transpose_2d_f64(b.gpu_handle()?, n, k)?;
            backend.matmul_f64(a.gpu_handle()?, &bt, m, k, n)?
        } else {
            return Err(FerrotorchError::NotImplementedOnCuda { op: "mm_bt" });
        };
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

        // GPU-native path for f32/f64 tensors.
        if grad_output.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let go_h = grad_output.gpu_handle()?;
            let f64_path = is_f64::<T>();

            let grad_input = if self.input.requires_grad() {
                let k = self.weight.shape()[1];
                let w_h = self.weight.gpu_handle()?;
                let result_h = if f64_path {
                    backend.matmul_f64(go_h, w_h, m, n, k)?
                } else {
                    backend.matmul_f32(go_h, w_h, m, n, k)?
                };
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
                let result_h = if f64_path {
                    let got_h = backend.transpose_2d_f64(go_h, m, n)?;
                    backend.matmul_f64(&got_h, inp_h, n, m, k)?
                } else {
                    let got_h = backend.transpose_2d_f32(go_h, m, n)?;
                    backend.matmul_f32(&got_h, inp_h, n, m, k)?
                };
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
                        let summed = if f64_path {
                            backend.sum_axis_f64(go_h, go_shape, 0)?
                        } else {
                            backend.sum_axis_f32(go_h, go_shape, 0)?
                        };
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

        if grad_output.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "LinearFusedBackward",
            });
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
    let m = input.shape()[0];
    let k = input.shape()[1];
    let n = weight.shape()[0];

    // GPU path: transpose weight, matmul, broadcast_add bias.
    if input.is_cuda() {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        // Dtype-aware GPU dispatch (#800): forward must branch on f32 vs. f64
        // just as the backward already does. Calling `*_f32` kernels on f64
        // handles surfaces "GPU handle does not contain a CudaBuffer<f32>".
        let mut result_handle = if is_f32::<T>() {
            // C = input @ weight^T
            let wt_handle = backend.transpose_2d_f32(weight.gpu_handle()?, n, k)?;
            // When autocast says ReducedPrecision and inputs are f32 on GPU,
            // use the f16-accumulate path (falls back to f32 if no kernel).
            if autocast_guard("linear") == Some(AutocastCategory::ReducedPrecision) {
                backend.matmul_f16_f32(input.gpu_handle()?, &wt_handle, m, k, n)?
            } else {
                backend.matmul_f32(input.gpu_handle()?, &wt_handle, m, k, n)?
            }
        } else if is_f64::<T>() {
            let wt_handle = backend.transpose_2d_f64(weight.gpu_handle()?, n, k)?;
            backend.matmul_f64(input.gpu_handle()?, &wt_handle, m, k, n)?
        } else {
            return Err(FerrotorchError::NotImplementedOnCuda { op: "linear_fused" });
        };
        // Add bias if present (dtype-aware).
        if let Some(b) = bias {
            let out_shape = vec![m, n];
            let b_shape = vec![n];
            result_handle = if is_f32::<T>() {
                backend.broadcast_add_f32(
                    &result_handle,
                    b.gpu_handle()?,
                    &out_shape,
                    &b_shape,
                    &out_shape,
                )?
            } else {
                backend.broadcast_add_f64(
                    &result_handle,
                    b.gpu_handle()?,
                    &out_shape,
                    &b_shape,
                    &out_shape,
                )?
            };
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
    let m = a.shape()[0];
    let k = a.shape()[1];

    // GPU path (#817): route CUDA inputs through cuBLAS Sgemv/Dgemv. Pre-fix
    // the function unconditionally called `.data()?` and surfaced as
    // `GpuTensorNotAccessible`. PyTorch's `torch.mv` works on CUDA for
    // f32 and f64 and so must ferrotorch's.
    if a.is_cuda() && a.device() == x.device() {
        if let Some(backend) = gpu_backend() {
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
                return Err(FerrotorchError::NotImplementedOnCuda { op: "mv" });
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
    }

    // CPU path: compute mv directly from slices to avoid double-copy.
    let a_data = a.data()?;
    let x_data = x.data()?;
    let zero = <T as num_traits::Zero>::zero();

    let mut result_vec = vec![zero; m];
    for (i, result_elem) in result_vec.iter_mut().enumerate() {
        let mut acc = zero;
        let row = i * k;
        for p in 0..k {
            acc += a_data[row + p] * x_data[p];
        }
        *result_elem = acc;
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

    // GPU path (#816): route CUDA inputs through cuBLAS Sdot/Ddot. Pre-fix
    // the function unconditionally called `.data()?` and surfaced as
    // `GpuTensorNotAccessible`. PyTorch's `torch.dot` works on CUDA for
    // f32 and f64 and so must ferrotorch's.
    if a.is_cuda() && a.device() == b.device() {
        if let Some(backend) = gpu_backend() {
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
            let n = a.shape().first().copied().unwrap_or(0);
            let handle = if is_f32::<T>() {
                backend.dot_f32(a.gpu_handle()?, b.gpu_handle()?, n)?
            } else if is_f64::<T>() {
                backend.dot_f64(a.gpu_handle()?, b.gpu_handle()?, n)?
            } else {
                return Err(FerrotorchError::NotImplementedOnCuda { op: "dot" });
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
    }

    let a_data = a.data()?;
    let b_data = b.data()?;
    let result_val = a_data
        .iter()
        .zip(b_data.iter())
        .fold(<T as num_traits::Zero>::zero(), |acc, (&x, &y)| acc + x * y);

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
    // Record autocast decision. Actual f16 dispatch for bmm will be added
    // when the batched f16 GEMM kernel lands; for now the guard ensures the
    // policy is tracked.
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

    // Materialize non-contiguous views before linalg ops.
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
            return Err(FerrotorchError::NotImplementedOnCuda { op: "matmul" });
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
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let m = a.shape()[0];
        let k = a.shape()[1];
        let n = b.shape()[1];
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
        // f32/f64 dispatch (#800). 1D x 1D / 2D x 1D / 1D x 2D vector cases
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
        // bf16 lands here for the "single-run" broadcast patterns only —
        // those where each lead is either empty (fully broadcast) or matches
        // `out_lead` exactly. That covers every shape the dispatcher routes
        // here today (3D × 2D, 2D × 3D, ND × ND with matching leads). For
        // less-uniform broadcasts the bf16 backend returns
        // `InvalidArgument`; we detect that and fall through to the CPU
        // path (same behaviour as today — no regression).
        if a.is_cuda()
            && a.ndim() >= 2
            && b.ndim() >= 2
            && (is_f32::<T>() || is_f64::<T>() || is_bf16::<T>())
        {
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
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
            // bf16 single-run guard: the bf16 backend only encodes broadcasts
            // where each lead is empty or matches `out_lead` exactly. For
            // anything else, skip the GPU path and let the CPU fallback
            // handle it — no regression vs. pre-fix behaviour.
            let bf16_skip = is_bf16::<T>() && !(a_lead.is_empty() || a_lead == out_lead)
                || is_bf16::<T>() && !(b_lead.is_empty() || b_lead == out_lead);
            if !bf16_skip {
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
                } else {
                    // bf16 path (#1543 / GH#25 fix). Routes through cuBLAS
                    // `gemm_strided_batched_ex` with CUDA_R_16BF in/out and
                    // CUBLAS_COMPUTE_32F accumulator — the standard
                    // ~1.5e-3 cuBLAS bf16+f32-accum floor that the
                    // upstream issue expects.
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
    if a.is_cuda() {
        if let Some(backend) = crate::gpu_dispatch::gpu_backend() {
            // Dtype-aware GPU dispatch (#800): the f32-only path returned
            // "GPU handle does not contain a CudaBuffer<f32>" for f64 inputs.
            // Forward must branch on `is_f64::<T>()` and use `bmm_f64` (cuBLAS
            // dgemm strided-batched) for f64 tensors.
            let handle = if is_f32::<T>() {
                // Use f16 Tensor Core path when autocast selects ReducedPrecision.
                if autocast_guard("bmm") == Some(AutocastCategory::ReducedPrecision) {
                    backend.bmm_f16_f32(a.gpu_handle()?, b.gpu_handle()?, batch, m, k, n)?
                } else {
                    backend.bmm_f32(a.gpu_handle()?, b.gpu_handle()?, batch, m, k, n)?
                }
            } else if is_f64::<T>() {
                backend.bmm_f64(a.gpu_handle()?, b.gpu_handle()?, batch, m, k, n)?
            } else {
                return Err(FerrotorchError::NotImplementedOnCuda { op: "bmm" });
            };
            return Tensor::from_storage(TensorStorage::gpu(handle), out_shape, false);
        }
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
    if input.is_cuda() {
        if let Some(backend) = crate::gpu_dispatch::gpu_backend() {
            let handle = backend.permute_0213_f32(input.gpu_handle()?, d0, d1, d2, d3)?;
            return Tensor::from_storage(TensorStorage::gpu(handle), out_shape, false);
        }
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
/// VJPs below). Always materialises a contiguous CPU result.
fn mat_transpose<T: Float>(t: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    transpose(t)
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
        let g: T = grad_output.item()?;
        let zero = <T as num_traits::Zero>::zero();
        let mut out = vec![zero; self.rows * self.cols];
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

/// Differentiable `outer`. Attaches `OuterBackward` when grad is needed.
///
/// Forward computed under `no_grad`: `linalg_fwd::outer` (the public
/// `crate::linalg::outer` forward) delegates back here when grad is enabled,
/// so the guard prevents infinite re-entry.
pub fn outer_differentiable<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let result = crate::autograd::no_grad::no_grad(|| linalg_fwd::outer(a, b))?;
    if is_grad_enabled() && (a.requires_grad() || b.requires_grad()) {
        let grad_fn = Arc::new(OuterBackward::new(a.clone(), b.clone()));
        let (storage, shape) = result.into_storage_and_shape()?;
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Ok(result)
    }
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
        // dA = -Y^T @ grad_Y @ Y^T
        let yt = mat_transpose(&self.inv)?;
        let tmp = mm(&yt, grad_output)?; // Y^T @ grad
        let prod = mm(&tmp, &yt)?; // (Y^T @ grad) @ Y^T
        let data = prod.data()?;
        let neg: Vec<T> = data.iter().map(|&v| -v).collect();
        let grad_a = Tensor::from_storage(TensorStorage::cpu(neg), prod.shape().to_vec(), false)?;
        Ok(vec![Some(grad_a)])
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

/// Backward for `d = det(A)`.
///
/// VJP (`torch/csrc/autograd/FunctionsManual.cpp:4373` `linalg_det_backward`,
/// invertible branch — the gradient solving `A^T G = det * grad * I`):
/// `dA = grad_d * det(A) * inv(A)^T`.
#[derive(Debug)]
pub struct LinalgDetBackward<T: Float> {
    /// Retained inverse-transpose of `A`.
    inv_t: Tensor<T>,
    /// Retained scalar determinant value.
    det: T,
}

impl<T: Float> GradFn<T> for LinalgDetBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let g: T = grad_output.item()?;
        let scale = g * self.det;
        let data = self.inv_t.data()?;
        let scaled: Vec<T> = data.iter().map(|&v| scale * v).collect();
        let grad_a = Tensor::from_storage(
            TensorStorage::cpu(scaled),
            self.inv_t.shape().to_vec(),
            false,
        )?;
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
        let inv = crate::autograd::no_grad::no_grad(|| linalg_fwd::inv(a))?;
        let inv_t = mat_transpose(&inv)?;
        let grad_fn = Arc::new(DetForward {
            input: a.clone(),
            inner: LinalgDetBackward {
                inv_t,
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
    /// Retained inverse-transpose of `A` (`inv(A)^T`).
    inv_t: Tensor<T>,
}

impl<T: Float> GradFn<T> for SlogdetBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // grad_output is the upstream gradient on `logabsdet` (a scalar).
        let g: T = grad_output.item()?;
        let data = self.inv_t.data()?;
        let scaled: Vec<T> = data.iter().map(|&v| g * v).collect();
        let grad_a = Tensor::from_storage(
            TensorStorage::cpu(scaled),
            self.inv_t.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_a)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        // VJP closes over the retained inverse-transpose only; the graph edge
        // to the leaf `A` is carried by `SlogdetForward`.
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
        let inv = crate::autograd::no_grad::no_grad(|| linalg_fwd::inv(a))?;
        let inv_t = mat_transpose(&inv)?;
        let grad_fn = Arc::new(SlogdetForward {
            input: a.clone(),
            inner: SlogdetBackward { inv_t },
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
    let target_size: usize = target.iter().product::<usize>().max(1);
    let mut out = vec![zero; target_size];

    let grad_nd = grad_shape.len();
    let target_nd = target.len();
    let offset = grad_nd - target_nd;

    let mut target_strides = vec![1usize; target_nd];
    for i in (0..target_nd.saturating_sub(1)).rev() {
        target_strides[i] = target_strides[i + 1] * target[i + 1];
    }

    let grad_total: usize = grad_shape.iter().product::<usize>().max(1);
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
    let bias_b = broadcast_data_to(bias, &[m, n])?;
    let mut out = vec![<T as num_traits::Zero>::zero(); m * n];
    for i in 0..m * n {
        out[i] = beta * bias_b[i] + alpha * prod[i];
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
    let target_size: usize = target.iter().product::<usize>().max(1);
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
    let bias_b = broadcast_data_to(bias, &[m])?;
    let mut out = vec![<T as num_traits::Zero>::zero(); m];
    for i in 0..m {
        out[i] = beta * bias_b[i] + alpha * prod[i];
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
    let bias_b = broadcast_data_to(bias, &[m, n])?;
    let mut out = vec![<T as num_traits::Zero>::zero(); m * n];
    for i in 0..m {
        let av1 = alpha * v1[i];
        let row = i * n;
        for j in 0..n {
            out[row + j] = beta * bias_b[row + j] + av1 * v2[j];
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
    let bias_b = broadcast_data_to(bias, &[bsz, m, n])?;
    let mut out = vec![<T as num_traits::Zero>::zero(); bsz * m * n];
    for i in 0..out.len() {
        out[i] = beta * bias_b[i] + alpha * prod[i];
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
    let bias_b = broadcast_data_to(bias, &[m, n])?;
    let mut out = vec![<T as num_traits::Zero>::zero(); m * n];
    for i in 0..m * n {
        out[i] = beta * bias_b[i] + alpha * acc[i];
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
        let g = grad_output.data()?;
        let zero = <T as num_traits::Zero>::zero();
        let mut out = vec![zero; self.rows * self.cols];
        let (row_start, col_start) = if self.offset >= 0 {
            (0usize, self.offset as usize)
        } else {
            ((-self.offset) as usize, 0usize)
        };
        for (i, &gv) in g.iter().enumerate() {
            let r = row_start + i;
            let c = col_start + i;
            out[r * self.cols + c] = gv;
        }
        Ok(vec![Some(from_cpu(out, vec![self.rows, self.cols])?)])
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
    let result = linalg_fwd::diagonal(a, offset)?;
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
        let g = grad_output.data()?;
        let zero = <T as num_traits::Zero>::zero();
        if self.construct {
            // Forward was 1-D -> 2-D diagonal matrix; grad is 2-D, grad_input
            // is the diagonal of grad (1-D).
            let n = self.in_shape[0];
            let offset = self.diagonal.unsigned_abs() as usize;
            let size = n + offset;
            let mut out = vec![zero; n];
            for (i, slot) in out.iter_mut().enumerate() {
                let (r, c) = if self.diagonal >= 0 {
                    (i, i + offset)
                } else {
                    (i + offset, i)
                };
                *slot = g[r * size + c];
            }
            Ok(vec![Some(from_cpu(out, vec![n])?)])
        } else {
            // Forward was 2-D -> 1-D extract; grad is 1-D, grad_input scatters
            // grad onto the `diagonal`-th diagonal of a zero matrix.
            let rows = self.in_shape[0];
            let cols = self.in_shape[1];
            let mut out = vec![zero; rows * cols];
            let (start_r, start_c) = if self.diagonal >= 0 {
                (0usize, self.diagonal as usize)
            } else {
                ((-self.diagonal) as usize, 0usize)
            };
            for (i, &gv) in g.iter().enumerate() {
                out[(start_r + i) * cols + (start_c + i)] = gv;
            }
            Ok(vec![Some(from_cpu(out, vec![rows, cols])?)])
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
    let result = crate::ops::tensor_ops::diag(a, diagonal)?;
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
/// the same triangular mask applied to the upstream gradient.
#[derive(Debug)]
pub struct TriangularBackward<T: Float> {
    rows: usize,
    cols: usize,
    diagonal: i64,
    /// `true` for `tril` (keep `c <= r + diag`), `false` for `triu`
    /// (keep `c >= r + diag`).
    lower: bool,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Float> GradFn<T> for TriangularBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let g = grad_output.data()?;
        let zero = <T as num_traits::Zero>::zero();
        let mut out = vec![zero; self.rows * self.cols];
        for r in 0..self.rows {
            for c in 0..self.cols {
                let keep = if self.lower {
                    (c as i64) <= (r as i64) + self.diagonal
                } else {
                    (c as i64) >= (r as i64) + self.diagonal
                };
                if keep {
                    out[r * self.cols + c] = g[r * self.cols + c];
                }
            }
        }
        Ok(vec![Some(from_cpu(out, vec![self.rows, self.cols])?)])
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
    let result = crate::ops::tensor_ops::tril(a, diagonal)?;
    if is_grad_enabled() && a.requires_grad() {
        let shape = a.shape();
        let grad_fn = Arc::new(TriangularForward {
            input: a.clone(),
            inner: TriangularBackward {
                rows: shape[0],
                cols: shape[1],
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
    let result = crate::ops::tensor_ops::triu(a, diagonal)?;
    if is_grad_enabled() && a.requires_grad() {
        let shape = a.shape();
        let grad_fn = Arc::new(TriangularForward {
            input: a.clone(),
            inner: TriangularBackward {
                rows: shape[0],
                cols: shape[1],
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
}
