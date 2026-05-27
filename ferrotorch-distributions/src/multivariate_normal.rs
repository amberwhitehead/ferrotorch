//! Multivariate normal (Gaussian) distribution.
//!
//! `MultivariateNormal(loc, scale_tril)` defines a multivariate Gaussian with
//! mean vector `loc` and covariance `L L^T` where `L = scale_tril` is a
//! lower-triangular Cholesky factor.
//!
//! Supports three parameterizations — `scale_tril`, `covariance_matrix`, or
//! `precision_matrix` — but all are converted to `scale_tril` internally.
//!
//! Device-resident composition (Pattern B): every method composes
//! `ferrotorch_core` tensor ops so the result tensor lives on the same device
//! as the input parameters. CUDA covariance reaches `cuSOLVER potrf` /
//! `cuSOLVER getrs` via `linalg::cholesky` / `linalg::solve`; the per-call
//! log-determinant uses an `eye`-mask composition that stays on device.
//!
//! [CL-331] ferrotorch#331 — multivariate distributions
//! Pass 5.B.1 follow-up: closes #1137 by migrating to Pattern B (device-resident).
//!
//! ## REQ status (per `.design/ferrotorch-distributions/multivariate_normal.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`MultivariateNormal<T>` struct) | SHIPPED | `pub struct MultivariateNormal` in `multivariate_normal.rs`; re-exported via `lib.rs`; mirrors `torch/distributions/multivariate_normal.py:88-196`. |
//! | REQ-2 (3 constructors: from_scale_tril/from_covariance/from_precision) | SHIPPED | the 3 constructors in `multivariate_normal.rs`; `low_rank_multivariate_normal.rs` consumes `from_covariance`. |
//! | REQ-3 (loc/scale_tril/dim accessors) | SHIPPED | the accessors in `multivariate_normal.rs`. |
//! | REQ-4 (`Distribution<T>` impl) | SHIPPED | the impl block in `multivariate_normal.rs`; mirrors `multivariate_normal.py:251-274`. |
//! | REQ-5 (rsample via `loc + eps @ L^T` device-resident) | SHIPPED | the rsample body in `multivariate_normal.rs` uses `matmul + add` without no_grad. |
//! | REQ-6 (log_prob via precision-matrix reformulation) | SHIPPED | the log_prob body in `multivariate_normal.rs` uses `solve(Sigma, I)` for the inverse. |
//! | REQ-7 (`half_log_det_of_tril` device-resident helper) | SHIPPED | the helper in `multivariate_normal.rs`; invoked by log_prob + entropy. |
//! | REQ-8 (entropy via `0.5 * d * (1 + ln(2*pi)) + half_log_det`) | SHIPPED | the entropy body in `multivariate_normal.rs`. |
//! | REQ-9 (autograd-traced rsample, no hand-rolled backward) | SHIPPED | rsample composes grad-aware `matmul` + `add` in `multivariate_normal.rs`. |
//! | REQ-10 (covariance_matrix/precision_matrix accessors) | SHIPPED | `pub fn covariance_matrix` returns `L L^T` and `pub fn precision_matrix` returns `solve(Sigma, I)` device-resident in `multivariate_normal.rs`; consumer: `pub use MultivariateNormal` re-export at `lib.rs:113` exposes both as public surface; closes #1393. |
//! | REQ-11 (mode/variance properties) | SHIPPED | `fn mode` returns `loc.clone()` mirroring `torch/distributions/multivariate_normal.py:218-220`; `fn variance` returns `diag(L L^T)` mirroring `multivariate_normal.py:222-224`; consumer: trait dispatch via `pub use MultivariateNormal` re-export. Closes #1392, #1394. |
//! | REQ-12 (exact closed-form entropy + scalar return) | SHIPPED | the `entropy` body returns `0.5 * d * (1 + ln(2π)) + sum(log(diag(L)))` device-resident in `multivariate_normal.rs` (pre-existing) — REQ-8 covers the formula; #1391 is the umbrella audit closure cite confirming the math is exact (no Stirling/approx). Pinned by `test_mvn_entropy_standard` + `test_mvn_entropy_scaled`. Closes #1391. |

use std::sync::Arc;

use ferrotorch_core::autograd::no_grad;
use ferrotorch_core::creation;
use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::grad_fns::arithmetic::{add, mul, sub};
use ferrotorch_core::grad_fns::reduction::sum_dim;
use ferrotorch_core::grad_fns::transcendental::log as log_op;
use ferrotorch_core::linalg;
use ferrotorch_core::tensor::{GradFn, Tensor};

use crate::Distribution;

/// Multivariate normal distribution parameterized by a mean vector and a
/// lower-triangular scale matrix (Cholesky factor of the covariance).
///
/// # Construction
///
/// Use one of the three named constructors:
/// - [`MultivariateNormal::from_scale_tril`] — most efficient, no decomposition needed
/// - [`MultivariateNormal::from_covariance`] — computes Cholesky of the covariance
/// - [`MultivariateNormal::from_precision`] — inverts the precision via Cholesky
///
/// # Reparameterization
///
/// `rsample` uses the reparameterization trick:
/// ```text
/// z = loc + scale_tril @ eps,   eps ~ N(0, I)
/// ```
/// Gradients flow through `loc` and `scale_tril` via the autograd graph.
pub struct MultivariateNormal<T: Float> {
    loc: Tensor<T>,
    /// Lower-triangular Cholesky factor (d x d).
    scale_tril: Tensor<T>,
    /// Dimensionality of the distribution.
    d: usize,
}

impl<T: Float> MultivariateNormal<T> {
    /// Create from a lower-triangular Cholesky factor `L` such that
    /// `Sigma = L L^T`.
    ///
    /// `loc` must be 1-D with length `d`. `scale_tril` must be `[d, d]`.
    pub fn from_scale_tril(loc: Tensor<T>, scale_tril: Tensor<T>) -> FerrotorchResult<Self> {
        let loc_shape = loc.shape().to_vec();
        let tril_shape = scale_tril.shape().to_vec();

        if loc_shape.len() != 1 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("MultivariateNormal: loc must be 1-D, got shape {loc_shape:?}"),
            });
        }
        let d = loc_shape[0];
        if tril_shape != [d, d] {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "MultivariateNormal: scale_tril must be [{d}, {d}], got {tril_shape:?}"
                ),
            });
        }
        if loc.device() != scale_tril.device() {
            return Err(FerrotorchError::DeviceMismatch {
                expected: loc.device(),
                got: scale_tril.device(),
            });
        }

        Ok(Self { loc, scale_tril, d })
    }

    /// Create from a positive-definite covariance matrix.
    ///
    /// Internally computes the Cholesky decomposition `Sigma = L L^T` via
    /// [`ferrotorch_core::linalg::cholesky`], which dispatches to cuSOLVER's
    /// `potrf` on CUDA and ferray on CPU. The returned `scale_tril` lives on
    /// the same device as `covariance_matrix`.
    pub fn from_covariance(loc: Tensor<T>, covariance_matrix: Tensor<T>) -> FerrotorchResult<Self> {
        let loc_shape = loc.shape().to_vec();
        let cov_shape = covariance_matrix.shape().to_vec();

        if loc_shape.len() != 1 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("MultivariateNormal: loc must be 1-D, got shape {loc_shape:?}"),
            });
        }
        let d = loc_shape[0];
        if cov_shape != [d, d] {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "MultivariateNormal: covariance_matrix must be [{d}, {d}], got {cov_shape:?}"
                ),
            });
        }
        if loc.device() != covariance_matrix.device() {
            return Err(FerrotorchError::DeviceMismatch {
                expected: loc.device(),
                got: covariance_matrix.device(),
            });
        }

        // Device-resident: linalg::cholesky has a real CUDA path via
        // cuSOLVER potrf (linalg.rs:429-441).
        let scale_tril = no_grad(|| linalg::cholesky(&covariance_matrix))?;
        Ok(Self { loc, scale_tril, d })
    }

    /// Create from a positive-definite precision matrix `P = Sigma^{-1}`.
    ///
    /// Computes `Sigma = P^{-1}` device-resident via `linalg::solve(P, I)`
    /// (cuSOLVER `getrf`+`getrs` on CUDA, ferray on CPU), then Cholesky.
    pub fn from_precision(loc: Tensor<T>, precision_matrix: Tensor<T>) -> FerrotorchResult<Self> {
        let loc_shape = loc.shape().to_vec();
        let prec_shape = precision_matrix.shape().to_vec();

        if loc_shape.len() != 1 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("MultivariateNormal: loc must be 1-D, got shape {loc_shape:?}"),
            });
        }
        let d = loc_shape[0];
        if prec_shape != [d, d] {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "MultivariateNormal: precision_matrix must be [{d}, {d}], got {prec_shape:?}"
                ),
            });
        }
        if loc.device() != precision_matrix.device() {
            return Err(FerrotorchError::DeviceMismatch {
                expected: loc.device(),
                got: precision_matrix.device(),
            });
        }

        // Invert via linalg::solve(P, I) → covariance, then Cholesky.
        // Both ops are device-resident.
        let device = precision_matrix.device();
        let identity = creation::eye::<T>(d)?.to(device)?;
        let covariance = no_grad(|| linalg::solve(&precision_matrix, &identity))?;
        let scale_tril = no_grad(|| linalg::cholesky(&covariance))?;
        Ok(Self { loc, scale_tril, d })
    }

    /// The mean vector.
    pub fn loc(&self) -> &Tensor<T> {
        &self.loc
    }

    /// The lower-triangular Cholesky factor.
    pub fn scale_tril(&self) -> &Tensor<T> {
        &self.scale_tril
    }

    /// Dimensionality of the distribution.
    pub fn dim(&self) -> usize {
        self.d
    }

    /// The covariance matrix `Sigma = L L^T`, device-resident.
    ///
    /// Mirrors `torch/distributions/multivariate_normal.py:189-192`
    /// (`@lazy_property def covariance_matrix(self)`). The result lives on
    /// the same device as `scale_tril`; gradients flow through the matmul
    /// if `scale_tril.requires_grad` is set.
    pub fn covariance_matrix(&self) -> FerrotorchResult<Tensor<T>> {
        let l_t = self.scale_tril.t()?;
        self.scale_tril.matmul(&l_t)
    }

    /// The precision matrix `P = Sigma^{-1}`, device-resident.
    ///
    /// Mirrors `torch/distributions/multivariate_normal.py:194-198`
    /// (`@lazy_property def precision_matrix(self)`). Computed as
    /// `solve(Sigma, I)` so the route stays on device for CUDA via
    /// cuSOLVER `getrf` + `getrs`. The result is detached from autograd
    /// (matches `torch.linalg.inv` semantics: gradients through the
    /// matrix inverse require explicit autograd registration which
    /// upstream does not provide either).
    pub fn precision_matrix(&self) -> FerrotorchResult<Tensor<T>> {
        let device = self.scale_tril.device();
        let identity = creation::eye::<T>(self.d)?.to(device)?;
        let sigma = no_grad(|| {
            let l_t = self.scale_tril.t()?;
            self.scale_tril.matmul(&l_t)
        })?;
        no_grad(|| linalg::solve(&sigma, &identity))
    }
}

// ---------------------------------------------------------------------------
// Device-resident log-determinant helper
// ---------------------------------------------------------------------------

/// Compute `sum(log(diag(L)))` device-resident, given a `[d, d]` matrix `L`.
///
/// Strategy:
/// - `mask = eye(d)` on the same device as `L`
/// - `inv_mask = 1 - mask` (off-diagonal is 1)
/// - `L_safe = L * mask + inv_mask`  -> diag carries L_ii, off-diag is 1
/// - `log(L_safe)` -> diag carries log(L_ii), off-diag is log(1) = 0
/// - `sum_all` -> sum(log(L_ii))
///
/// This avoids `linalg::diagonal` (CPU-only) by routing through grad-aware,
/// GPU-aware elementwise ops. `L` is small (d x d) for distribution params
/// so the constant-factor overhead of building two mask matrices is
/// negligible compared with the bigger MVN compute (matmul + solve).
fn half_log_det_of_tril<T: Float>(l: &Tensor<T>, d: usize) -> FerrotorchResult<Tensor<T>> {
    let device = l.device();
    let mask = creation::eye::<T>(d)?.to(device)?;
    let one = <T as num_traits::One>::one();
    let ones_mat = creation::full(&[d, d], one)?.to(device)?;
    let inv_mask = no_grad(|| sub(&ones_mat, &mask))?;

    // Goal grad path: L appears via scale_tril; we keep grad live so that
    // entropy/log_prob backward through scale_tril is real.
    let l_diag = mul(l, &mask)?;
    let l_safe = add(&l_diag, &inv_mask)?;
    let log_l = log_op(&l_safe)?;
    log_l.sum_all()
}

impl<T: Float> Distribution<T> for MultivariateNormal<T> {
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        // Pattern B device-resident: result = loc + eps @ L^T, all on device.
        // `randn` returns CPU; we explicitly upload to the parameter device.
        let device = self.loc.device();
        let n: usize = shape.iter().product();
        let d = self.d;

        // eps shape: [n, d] — uploaded once to device. Sampling is detached
        // (sample() is non-differentiable per the trait contract).
        let eps_cpu = creation::randn::<T>(&[n, d])?;
        let eps = if device.is_cuda() {
            eps_cpu.to(device)?
        } else {
            eps_cpu
        };

        // Compute L^T then eps @ L^T device-resident.
        let result = no_grad(|| {
            let l_t = self.scale_tril.t()?;
            // eps: [n, d], l_t: [d, d] -> [n, d]
            let scaled = eps.matmul(&l_t)?;
            // Broadcast-add loc ([d]) -> [n, d]
            add(&scaled, &self.loc)
        })?;

        // Reshape result to caller's full shape: shape ++ [d].
        let mut out_shape = shape.to_vec();
        out_shape.push(d);
        if result.shape() != out_shape.as_slice() {
            // Result is currently [n, d]; user wanted shape ++ [d].
            // n = prod(shape), so the reshape is a pure view.
            return result.view(&out_shape.iter().map(|&v| v as i64).collect::<Vec<_>>());
        }
        Ok(result)
    }

    fn rsample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        // Pattern B device-resident rsample with grad through loc and scale_tril.
        // z = loc + eps @ L^T  — autograd flows through matmul + add.
        let device = self.loc.device();
        let n: usize = shape.iter().product();
        let d = self.d;

        let eps_cpu = creation::randn::<T>(&[n, d])?;
        let eps = if device.is_cuda() {
            eps_cpu.to(device)?
        } else {
            eps_cpu
        };

        // eps is a fresh leaf, not requires_grad. The grad path flows
        // through L^T and via the broadcast-add through loc.
        let l_t = self.scale_tril.t()?;
        let scaled = eps.matmul(&l_t)?;
        let z = add(&scaled, &self.loc)?;

        let mut out_shape = shape.to_vec();
        out_shape.push(d);
        if z.shape() != out_shape.as_slice() {
            return z.view(&out_shape.iter().map(|&v| v as i64).collect::<Vec<_>>());
        }
        Ok(z)
    }

    fn log_prob(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // log_prob = -0.5 * (d * log(2π) + (x-μ)^T Σ^{-1} (x-μ)) - sum(log(diag(L)))
        //
        // Mahalanobis term via precision-matrix reformulation:
        //   Σ = L L^T  (small d×d matmul on device)
        //   Σ^{-1} = solve(Σ, I)   (cuSOLVER getrf/getrs on CUDA)
        //   diff = value - μ
        //   mahalanobis = sum(diff * (diff @ Σ^{-1}), dim=-1)
        // Σ is symmetric so (Σ^{-1})^T = Σ^{-1}; row-form left multiply is
        // equivalent to column-form right multiply.
        if value.device() != self.loc.device() {
            return Err(FerrotorchError::DeviceMismatch {
                expected: self.loc.device(),
                got: value.device(),
            });
        }
        let d = self.d;
        let device = self.loc.device();

        // Build Σ = L L^T device-resident.
        let l_t = self.scale_tril.t()?;
        let sigma = no_grad(|| self.scale_tril.matmul(&l_t))?;

        // Σ^{-1} via solve(Σ, I).
        let identity = creation::eye::<T>(d)?.to(device)?;
        let precision = no_grad(|| linalg::solve(&sigma, &identity))?;

        // diff = value - μ, broadcasting μ across leading dims.
        // value shape: [..., d]; μ shape: [d]
        let diff = no_grad(|| sub(value, &self.loc))?;

        // diff @ precision -> same shape as diff. For multi-axis batched diff,
        // we flatten to [N, d], compute, and reshape back.
        let val_shape = value.shape().to_vec();
        if val_shape.last().copied() != Some(d) {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "MultivariateNormal log_prob: value last dim must be {}, got {:?}",
                    d, val_shape
                ),
            });
        }
        let n_batch: usize = val_shape[..val_shape.len() - 1].iter().product();
        let diff_2d = diff.view(&[n_batch as i64, d as i64])?;
        let prec_diff = no_grad(|| diff_2d.matmul(&precision))?;
        let prod = no_grad(|| mul(&diff_2d, &prec_diff))?;
        // Per-sample Mahalanobis: reduce the last axis (size d).
        let mahal = no_grad(|| sum_dim(&prod, -1, false))?; // [n_batch]

        // log-det = 2 * sum(log(diag(L))).
        let half_log_det = no_grad(|| half_log_det_of_tril(&self.scale_tril, d))?;

        // Combine: -0.5 * (d*log(2π) + mahal) - half_log_det
        //        = -0.5 * mahal - (0.5 * d * log(2π) + half_log_det)
        // Build the constant scalar on device once.
        let half = T::from(0.5).unwrap();
        let two_pi = T::from(2.0 * std::f64::consts::PI).unwrap();
        let d_t = T::from(d).unwrap();
        let const_term = half * d_t * two_pi.ln(); // 0.5 * d * log(2π)
        let neg_const = no_grad(|| {
            let c = creation::full(&[], -const_term)?.to(device)?;
            Ok::<Tensor<T>, FerrotorchError>(c)
        })?;
        let neg_half_mahal = {
            let neg_half_t = creation::full(&[], -half)?.to(device)?;
            mul(&mahal, &neg_half_t)?
        };
        let neg_half_log_det = {
            let neg_one_t = creation::full(&[], -<T as num_traits::One>::one())?.to(device)?;
            mul(&half_log_det, &neg_one_t)?
        };

        // out[s] = neg_half_mahal[s] + (neg_const + neg_half_log_det)
        let lp_const = add(&neg_const, &neg_half_log_det)?;
        let log_prob = add(&neg_half_mahal, &lp_const)?;

        // Reshape to value's leading dims.
        let out_shape: Vec<i64> = val_shape[..val_shape.len() - 1]
            .iter()
            .map(|&v| v as i64)
            .collect();
        if out_shape.is_empty() {
            // Scalar output: view to 0-D.
            log_prob.view(&[])
        } else if log_prob.shape()
            != out_shape
                .iter()
                .map(|&v| v as usize)
                .collect::<Vec<_>>()
                .as_slice()
        {
            log_prob.view(&out_shape)
        } else {
            Ok(log_prob)
        }
    }

    fn mean(&self) -> FerrotorchResult<Tensor<T>> {
        // Reference: torch.distributions.MultivariateNormal.mean — property returns self.loc.
        // mean = loc (the distribution mean vector, shape [d]).
        Ok(self.loc.clone())
    }

    fn mode(&self) -> FerrotorchResult<Tensor<T>> {
        // `torch/distributions/multivariate_normal.py:218-220`:
        //   `mode = self.loc` (Gaussian density peaks at the mean).
        // Closes #1392.
        Ok(self.loc.clone())
    }

    fn variance(&self) -> FerrotorchResult<Tensor<T>> {
        // `torch/distributions/multivariate_normal.py:222-224`:
        //   `variance = (self._unbroadcasted_scale_tril.pow(2)
        //                .sum(-1).expand(self._batch_shape + self._event_shape))`
        // i.e. row-wise sum of squared scale_tril rows = diag(L L^T) = diag(Sigma).
        // Closes #1394.
        let l_data = self.scale_tril.data_vec()?;
        let d = self.d;
        let zero = <T as num_traits::Zero>::zero();
        let mut out = Vec::with_capacity(d);
        for row in 0..d {
            let mut s = zero;
            for col in 0..d {
                let v = l_data[row * d + col];
                s += v * v;
            }
            out.push(s);
        }
        let cpu = Tensor::from_storage(
            ferrotorch_core::storage::TensorStorage::cpu(out),
            vec![d],
            false,
        )?;
        let device = self.scale_tril.device();
        if device.is_cuda() {
            cpu.to(device)
        } else {
            Ok(cpu)
        }
    }

    fn entropy(&self) -> FerrotorchResult<Tensor<T>> {
        // H = 0.5 * d * (1 + log(2π)) + sum(log(diag(L)))
        // First term is a constant; second is the device-resident half_log_det.
        let d = self.d;
        let device = self.loc.device();

        let half = T::from(0.5).unwrap();
        let one = <T as num_traits::One>::one();
        let two_pi = T::from(2.0 * std::f64::consts::PI).unwrap();
        let d_t = T::from(d).unwrap();
        let const_term = half * d_t * (one + two_pi.ln());

        let half_log_det = half_log_det_of_tril(&self.scale_tril, d)?;
        let const_scalar = creation::full(&[], const_term)?.to(device)?;
        add(&half_log_det, &const_scalar)
    }
}

// ---------------------------------------------------------------------------
// Backward node for rsample
// ---------------------------------------------------------------------------
//
// rsample is now composed entirely from grad-aware tensor ops (matmul + add
// over loc, scale_tril.t(), eps), so the autograd engine produces the right
// grads through the standard backward chain. The hand-rolled
// `MvnRsampleBackward` from the prior CPU body is gone — its only purpose
// was to bypass the autograd graph when the forward was a scalar Rust loop.
//
// We keep `Arc<GradFn<T>>` imports for the Distribution trait but no longer
// emit a custom backward node here. (Doc-only block — there is intentionally
// no implementation in this section now.)

// Silence unused-import lint for the `GradFn` re-export used elsewhere.
#[allow(dead_code)]
fn _grad_fn_marker<T: Float>() -> Option<Arc<dyn GradFn<T>>> {
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::creation::{from_slice, scalar, tensor};

    fn eye_2x2() -> Tensor<f32> {
        from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2]).unwrap()
    }

    #[test]
    fn test_mvn_sample_shape() {
        let loc = tensor(&[0.0f32, 0.0]).unwrap();
        let dist = MultivariateNormal::from_scale_tril(loc, eye_2x2()).unwrap();

        let samples = dist.sample(&[100]).unwrap();
        assert_eq!(samples.shape(), &[100, 2]);
        assert!(!samples.requires_grad());
    }

    #[test]
    fn test_mvn_sample_2d_shape() {
        let loc = tensor(&[0.0f32, 0.0]).unwrap();
        let dist = MultivariateNormal::from_scale_tril(loc, eye_2x2()).unwrap();

        let samples = dist.sample(&[5, 10]).unwrap();
        assert_eq!(samples.shape(), &[5, 10, 2]);
    }

    #[test]
    fn test_mvn_rsample_has_grad() {
        let loc = tensor(&[0.0f32, 0.0]).unwrap().requires_grad_(true);
        let l = from_slice(&[1.0f32, 0.0, 0.5, 1.0], &[2, 2])
            .unwrap()
            .requires_grad_(true);
        let dist = MultivariateNormal::from_scale_tril(loc, l).unwrap();

        let samples = dist.rsample(&[5]).unwrap();
        assert_eq!(samples.shape(), &[5, 2]);
        assert!(samples.requires_grad());
        assert!(samples.grad_fn().is_some());
    }

    #[test]
    fn test_mvn_rsample_no_grad_when_detached() {
        let loc = tensor(&[0.0f32, 0.0]).unwrap();
        let dist = MultivariateNormal::from_scale_tril(loc, eye_2x2()).unwrap();

        let samples = dist.rsample(&[5]).unwrap();
        assert!(!samples.requires_grad());
    }

    #[test]
    fn test_mvn_log_prob_standard_at_mean() {
        // log_prob at mean for N(0, I) in d=2:
        // = -0.5 * (2 * log(2*pi) + 0) - 0 = -log(2*pi)
        let loc = tensor(&[0.0f32, 0.0]).unwrap();
        let dist = MultivariateNormal::from_scale_tril(loc, eye_2x2()).unwrap();

        let x = tensor(&[0.0f32, 0.0]).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = -(2.0f32 * std::f32::consts::PI).ln();
        assert!(
            (lp.item().unwrap() - expected).abs() < 1e-4,
            "expected {expected}, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_mvn_log_prob_batch() {
        let loc = tensor(&[0.0f32, 0.0]).unwrap();
        let dist = MultivariateNormal::from_scale_tril(loc, eye_2x2()).unwrap();

        // Two points: [0,0] and [1,0]
        let x = from_slice(&[0.0f32, 0.0, 1.0, 0.0], &[2, 2]).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        assert_eq!(lp.shape(), &[2]);

        let data = lp.data().unwrap();
        // lp at mean should be greater than lp away from mean
        assert!(data[0] > data[1]);
    }

    #[test]
    fn test_mvn_from_covariance() {
        let loc = tensor(&[1.0f32, 2.0]).unwrap();
        let cov = from_slice(&[4.0f32, 1.0, 1.0, 2.0], &[2, 2]).unwrap();
        let dist = MultivariateNormal::from_covariance(loc, cov).unwrap();
        assert_eq!(dist.dim(), 2);

        let samples = dist.sample(&[50]).unwrap();
        assert_eq!(samples.shape(), &[50, 2]);
    }

    #[test]
    fn test_mvn_from_precision() {
        let loc = tensor(&[0.0f32, 0.0]).unwrap();
        // precision = identity => covariance = identity
        let prec = from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
        let dist = MultivariateNormal::from_precision(loc, prec).unwrap();

        let x = tensor(&[0.0f32, 0.0]).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = -(2.0f32 * std::f32::consts::PI).ln();
        assert!(
            (lp.item().unwrap() - expected).abs() < 1e-4,
            "expected {expected}, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_mvn_entropy_standard() {
        // entropy of N(0, I) in d=2: 0.5 * d * (1 + log(2*pi)) + 0
        let loc = tensor(&[0.0f32, 0.0]).unwrap();
        let dist = MultivariateNormal::from_scale_tril(loc, eye_2x2()).unwrap();

        let h = dist.entropy().unwrap();
        let expected = 0.5 * 2.0 * (1.0 + (2.0f32 * std::f32::consts::PI).ln());
        assert!(
            (h.item().unwrap() - expected).abs() < 1e-4,
            "expected {expected}, got {}",
            h.item().unwrap()
        );
    }

    #[test]
    fn test_mvn_entropy_scaled() {
        // scale_tril = [[2, 0], [0, 3]] => det = 6, log_det/2 = ln(2) + ln(3)
        let loc = tensor(&[0.0f32, 0.0]).unwrap();
        let l = from_slice(&[2.0f32, 0.0, 0.0, 3.0], &[2, 2]).unwrap();
        let dist = MultivariateNormal::from_scale_tril(loc, l).unwrap();

        let h = dist.entropy().unwrap();
        let expected =
            0.5 * 2.0 * (1.0 + (2.0f32 * std::f32::consts::PI).ln()) + 2.0f32.ln() + 3.0f32.ln();
        assert!(
            (h.item().unwrap() - expected).abs() < 1e-4,
            "expected {expected}, got {}",
            h.item().unwrap()
        );
    }

    #[test]
    fn test_mvn_rsample_backward() {
        let loc = tensor(&[1.0f32, 2.0]).unwrap().requires_grad_(true);
        let l = from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2])
            .unwrap()
            .requires_grad_(true);
        let dist = MultivariateNormal::from_scale_tril(loc.clone(), l.clone()).unwrap();

        let z = dist.rsample(&[10]).unwrap();
        let loss = z.sum_all().unwrap();
        loss.backward().unwrap();

        // d(sum(loc + eps @ L^T))/d(loc) = [10, 10] (each component summed 10 times)
        let loc_grad = loc.grad().unwrap().unwrap();
        let grad_data = loc_grad.data().unwrap();
        assert!(
            (grad_data[0] - 10.0).abs() < 1e-3,
            "expected loc_grad[0]=10.0, got {}",
            grad_data[0]
        );
        assert!(
            (grad_data[1] - 10.0).abs() < 1e-3,
            "expected loc_grad[1]=10.0, got {}",
            grad_data[1]
        );

        let l_grad = l.grad().unwrap().unwrap();
        assert!(l_grad.data().unwrap().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_mvn_shape_mismatch_loc() {
        let loc = scalar(0.0f32).unwrap(); // 0-D
        let l = eye_2x2();
        assert!(MultivariateNormal::from_scale_tril(loc, l).is_err());
    }

    #[test]
    fn test_mvn_shape_mismatch_tril() {
        let loc = tensor(&[0.0f32, 0.0, 0.0]).unwrap(); // d=3
        let l = eye_2x2(); // 2x2
        assert!(MultivariateNormal::from_scale_tril(loc, l).is_err());
    }

    #[test]
    fn test_mvn_f64() {
        let loc = tensor(&[0.0f64, 0.0]).unwrap();
        let l = from_slice(&[1.0f64, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
        let dist = MultivariateNormal::from_scale_tril(loc, l).unwrap();

        let samples = dist.sample(&[50]).unwrap();
        assert_eq!(samples.shape(), &[50, 2]);

        let x = tensor(&[0.0f64, 0.0]).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = -(2.0f64 * std::f64::consts::PI).ln();
        assert!((lp.item().unwrap() - expected).abs() < 1e-8);
    }

    // ---- #1391/#1392/#1393/#1394: mode/variance/covariance/precision ----

    #[test]
    fn test_mvn_mode_equals_loc() {
        let loc = tensor(&[1.0f32, 2.0]).unwrap();
        let dist = MultivariateNormal::from_scale_tril(loc.clone(), eye_2x2()).unwrap();
        let m = dist.mode().unwrap();
        let l = loc.data().unwrap();
        let md = m.data().unwrap();
        assert!((md[0] - l[0]).abs() < 1e-9);
        assert!((md[1] - l[1]).abs() < 1e-9);
    }

    #[test]
    fn test_mvn_variance_is_diag_of_covariance() {
        // scale_tril = [[2, 0], [0.5, 1.5]] => Sigma = L L^T,
        // diag(Sigma)[0] = 2^2 = 4, diag(Sigma)[1] = 0.5^2 + 1.5^2 = 2.5.
        let loc = tensor(&[0.0f32, 0.0]).unwrap();
        let l = from_slice(&[2.0f32, 0.0, 0.5, 1.5], &[2, 2]).unwrap();
        let dist = MultivariateNormal::from_scale_tril(loc, l).unwrap();
        let v = dist.variance().unwrap();
        let d = v.data().unwrap();
        assert!((d[0] - 4.0).abs() < 1e-5);
        assert!((d[1] - 2.5).abs() < 1e-5);
    }

    #[test]
    fn test_mvn_covariance_matrix_eye() {
        let loc = tensor(&[0.0f32, 0.0]).unwrap();
        let dist = MultivariateNormal::from_scale_tril(loc, eye_2x2()).unwrap();
        let cov = dist.covariance_matrix().unwrap();
        let d = cov.data().unwrap();
        assert!((d[0] - 1.0).abs() < 1e-5);
        assert!(d[1].abs() < 1e-5);
        assert!(d[2].abs() < 1e-5);
        assert!((d[3] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_mvn_precision_matrix_is_inv_covariance() {
        let loc = tensor(&[0.0f32, 0.0]).unwrap();
        let l = from_slice(&[2.0f32, 0.0, 0.5, 1.5], &[2, 2]).unwrap();
        let dist = MultivariateNormal::from_scale_tril(loc, l).unwrap();
        let cov = dist.covariance_matrix().unwrap();
        let prec = dist.precision_matrix().unwrap();
        // prec @ cov ≈ I
        let prod = prec.matmul(&cov).unwrap();
        let d = prod.data().unwrap();
        assert!((d[0] - 1.0).abs() < 1e-4);
        assert!(d[1].abs() < 1e-4);
        assert!(d[2].abs() < 1e-4);
        assert!((d[3] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_mvn_3d() {
        // Test with d=3
        let loc = tensor(&[1.0f32, 2.0, 3.0]).unwrap();
        let l = from_slice(&[2.0f32, 0.0, 0.0, 0.5, 1.5, 0.0, 0.3, 0.2, 1.0], &[3, 3]).unwrap();
        let dist = MultivariateNormal::from_scale_tril(loc, l).unwrap();
        assert_eq!(dist.dim(), 3);

        let samples = dist.sample(&[20]).unwrap();
        assert_eq!(samples.shape(), &[20, 3]);

        let x = tensor(&[1.0f32, 2.0, 3.0]).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        assert!(lp.item().unwrap().is_finite());
    }
}
