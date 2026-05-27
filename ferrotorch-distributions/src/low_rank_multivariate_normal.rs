//! Low-rank multivariate normal distribution.
//!
//! `LowRankMultivariateNormal(loc, cov_factor, cov_diag)` defines a Gaussian
//! whose covariance is `Σ = W W^T + diag(D)` where:
//!
//! - `loc` is the mean, shape `[d]`
//! - `cov_factor` (W) is `[d, r]` for some rank `r ≤ d`
//! - `cov_diag` (D) is `[d]` and elementwise positive
//!
//! This is the standard low-rank-plus-diagonal parameterization used in
//! probabilistic PCA, factor analysis, and many variational inference
//! settings. When `r ≪ d`, evaluating Σ⁻¹ and log det Σ via the matrix
//! determinant lemma + Woodbury identity is `O(d r²)` instead of `O(d³)`.
//!
//! # Implementation note
//!
//! Sampling and the `scale_tril` / `covariance_matrix` accessors delegate to
//! a dense inner [`MultivariateNormal`] built from `Σ = W Wᵀ + diag(D)`. The
//! hot `log_prob` path, however, uses the **Woodbury fast path**: the
//! Mahalanobis distance is evaluated via the Woodbury matrix identity and the
//! log-determinant via the matrix-determinant lemma, both expressed over the
//! `r×r` capacitance matrix `C = I_r + Wᵀ D⁻¹ W`. This is `O(d·r²)` per call
//! instead of `O(d³)` and never materialises the dense `[d, d]` Σ⁻¹. It
//! mirrors PyTorch's `_capacitance_tril` approach
//! (`torch/distributions/lowrank_multivariate_normal.py:16-51,225-240`).
//!
//! Mirrors `torch.distributions.LowRankMultivariateNormal`.
//!
//! ## REQ status (per `.design/ferrotorch-distributions/low_rank_multivariate_normal.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`LowRankMultivariateNormal<T>` struct) | SHIPPED | `pub struct LowRankMultivariateNormal` in `low_rank_multivariate_normal.rs`; re-exported via `lib.rs`; mirrors `torch/distributions/lowrank_multivariate_normal.py:54-139`. |
//! | REQ-2 (`new` constructor with shape + positive-diag validation) | SHIPPED | the constructor in `low_rank_multivariate_normal.rs`. |
//! | REQ-3 (5 accessors: loc/cov_factor/cov_diag/dim/rank) | SHIPPED | the accessors in `low_rank_multivariate_normal.rs`. |
//! | REQ-4 (`Distribution<T>` impl delegating to inner MVN) | SHIPPED | the impl block in `low_rank_multivariate_normal.rs`. |
//! | REQ-5 (mean override returns loc directly) | SHIPPED | the `mean()` body in `low_rank_multivariate_normal.rs`. |
//! | REQ-6 (Woodbury/capacitance-tril fast paths) | SHIPPED | `fn log_prob` override computes the Mahalanobis distance via the Woodbury identity + the log-determinant via the matrix-determinant lemma over the `r×r` capacitance matrix `C = I_r + Wᵀ D⁻¹ W` (`_batch_capacitance_tril`/`_batch_lowrank_logdet`/`_batch_lowrank_mahalanobis` at `torch/distributions/lowrank_multivariate_normal.py:16-51,225-240`) — `O(d·r²)` not `O(d³)`, no dense `[d, d]` Σ formed; non-test consumer: the `impl Distribution::log_prob` override IS reached on every `dist.log_prob(value)` via the `pub use LowRankMultivariateNormal` re-export. FD/dense-path-verified by `test_low_rank_woodbury_*` + `divergence_wave_l_audit`. Closes #1385. |
//! | REQ-7 (variance override) | SHIPPED | `fn variance` returns `(cov_factor ** 2).sum(-1) + cov_diag` mirroring `torch/distributions/lowrank_multivariate_normal.py:189-196`; non-test consumer: `pub use low_rank_multivariate_normal::LowRankMultivariateNormal` re-export — every external `dist.variance()` call hits this override; closes #1386. |
//! | REQ-8 (scale_tril/covariance_matrix/precision_matrix accessors) | SHIPPED | `pub fn scale_tril` / `pub fn covariance_matrix` / `pub fn precision_matrix` delegate to the inner dense `MultivariateNormal` mirroring `torch/distributions/lowrank_multivariate_normal.py:165-186`; non-test consumer: `pub use LowRankMultivariateNormal` re-export exposes all three as public surface; closes #1387. |

use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

use crate::{Distribution, MultivariateNormal};

/// Multivariate normal with low-rank-plus-diagonal covariance.
pub struct LowRankMultivariateNormal<T: Float> {
    loc: Tensor<T>,
    cov_factor: Tensor<T>,
    cov_diag: Tensor<T>,
    /// Inner [`MultivariateNormal`] built from the dense `Σ = W W^T + diag(D)`.
    /// All sample/log_prob/entropy calls delegate here.
    inner: MultivariateNormal<T>,
    d: usize,
    r: usize,
}

impl<T: Float> LowRankMultivariateNormal<T> {
    /// Construct from a mean vector, low-rank covariance factor, and
    /// diagonal correction.
    ///
    /// # Errors
    ///
    /// - `loc` must be 1-D shape `[d]`.
    /// - `cov_factor` must be 2-D shape `[d, r]`.
    /// - `cov_diag` must be 1-D shape `[d]` and contain only positive values.
    pub fn new(
        loc: Tensor<T>,
        cov_factor: Tensor<T>,
        cov_diag: Tensor<T>,
    ) -> FerrotorchResult<Self> {
        let loc_shape = loc.shape().to_vec();
        let factor_shape = cov_factor.shape().to_vec();
        let diag_shape = cov_diag.shape().to_vec();

        if loc_shape.len() != 1 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "LowRankMultivariateNormal: loc must be 1-D, got shape {loc_shape:?}"
                ),
            });
        }
        let d = loc_shape[0];

        if factor_shape.len() != 2 || factor_shape[0] != d {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "LowRankMultivariateNormal: cov_factor must be [{d}, r], got {factor_shape:?}"
                ),
            });
        }
        let r = factor_shape[1];

        if diag_shape != [d] {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "LowRankMultivariateNormal: cov_diag must be [{d}], got {diag_shape:?}"
                ),
            });
        }

        // Validate that cov_diag is positive.
        let diag_data = cov_diag.data_vec()?;
        let zero = <T as num_traits::Zero>::zero();
        for (i, &v) in diag_data.iter().enumerate() {
            if v <= zero {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "LowRankMultivariateNormal: cov_diag[{i}] = {} must be > 0",
                        v.to_f64().unwrap_or(f64::NAN)
                    ),
                });
            }
        }

        // Build the dense covariance Σ = W W^T + diag(D) on CPU. We
        // walk the factor data directly rather than calling matmul to
        // keep this self-contained and to avoid pulling in autograd
        // dependencies for what is a one-shot construction step.
        let factor_data = cov_factor.data_vec()?;
        let mut cov = vec![zero; d * d];
        for i in 0..d {
            for j in 0..d {
                let mut acc = zero;
                for k in 0..r {
                    acc += factor_data[i * r + k] * factor_data[j * r + k];
                }
                if i == j {
                    acc += diag_data[i];
                }
                cov[i * d + j] = acc;
            }
        }
        let device = loc.device();
        let cov_t = {
            let t = Tensor::from_storage(TensorStorage::cpu(cov), vec![d, d], false)?;
            if device.is_cuda() { t.to(device)? } else { t }
        };

        let inner = MultivariateNormal::from_covariance(loc.clone(), cov_t)?;

        Ok(Self {
            loc,
            cov_factor,
            cov_diag,
            inner,
            d,
            r,
        })
    }

    /// The mean vector.
    pub fn loc(&self) -> &Tensor<T> {
        &self.loc
    }

    /// The low-rank covariance factor `W` of shape `[d, r]`.
    pub fn cov_factor(&self) -> &Tensor<T> {
        &self.cov_factor
    }

    /// The diagonal correction `D` of shape `[d]`.
    pub fn cov_diag(&self) -> &Tensor<T> {
        &self.cov_diag
    }

    /// Dimensionality `d`.
    pub fn dim(&self) -> usize {
        self.d
    }

    /// Rank `r` of the low-rank factor.
    pub fn rank(&self) -> usize {
        self.r
    }

    /// Lower-triangular Cholesky factor `L` of `Σ = L L^T`.
    ///
    /// Delegates to the inner dense [`MultivariateNormal`]'s `scale_tril`
    /// accessor. Mirrors
    /// `torch/distributions/lowrank_multivariate_normal.py:165-167` (`scale_tril`
    /// property), which materialises the dense Cholesky on demand. Closes #1387.
    pub fn scale_tril(&self) -> &Tensor<T> {
        self.inner.scale_tril()
    }

    /// Dense covariance matrix `Σ = W W^T + diag(D)`.
    ///
    /// Delegates to the inner dense [`MultivariateNormal`]'s
    /// `covariance_matrix` accessor. Mirrors
    /// `torch/distributions/lowrank_multivariate_normal.py:169-175`. Closes #1387.
    pub fn covariance_matrix(&self) -> FerrotorchResult<Tensor<T>> {
        self.inner.covariance_matrix()
    }

    /// Dense precision matrix `Σ⁻¹`.
    ///
    /// Delegates to the inner dense [`MultivariateNormal`]'s
    /// `precision_matrix` accessor. Mirrors
    /// `torch/distributions/lowrank_multivariate_normal.py:177-186`. Closes #1387.
    pub fn precision_matrix(&self) -> FerrotorchResult<Tensor<T>> {
        self.inner.precision_matrix()
    }

    /// Lower-triangular Cholesky factor of the `r×r` capacitance matrix
    /// `C = I_r + Wᵀ D⁻¹ W` (`_batch_capacitance_tril` at
    /// `torch/distributions/lowrank_multivariate_normal.py:16-25`).
    ///
    /// `C` is symmetric positive-definite (its eigenvalues are bounded below
    /// by 1), so the small dense Cholesky is well-conditioned. Returned in
    /// row-major `[r, r]` layout with the strict upper triangle zeroed.
    ///
    /// `factor_data` is `W` in row-major `[d, r]`; `diag_data` is `D` of
    /// length `d`. CPU-resident: this is `O(d·r² + r³)` work on host slices.
    fn capacitance_tril(factor_data: &[T], diag_data: &[T], d: usize, r: usize) -> Vec<T> {
        let zero = <T as num_traits::Zero>::zero();
        let one = <T as num_traits::One>::one();

        // K = Wᵀ D⁻¹ W + I_r, an r×r symmetric matrix.
        // K[a, b] = sum_i W[i, a] * W[i, b] / D[i]   (+ 1 on the diagonal).
        let mut k = vec![zero; r * r];
        for a in 0..r {
            for b in 0..r {
                let mut acc = zero;
                for i in 0..d {
                    acc += factor_data[i * r + a] * factor_data[i * r + b] / diag_data[i];
                }
                if a == b {
                    acc += one;
                }
                k[a * r + b] = acc;
            }
        }

        // Cholesky K = L Lᵀ (lower-triangular L), standard Cholesky–Banachiewicz.
        let mut l = vec![zero; r * r];
        for a in 0..r {
            for b in 0..=a {
                let mut sum = k[a * r + b];
                for c in 0..b {
                    sum = sum - l[a * r + c] * l[b * r + c];
                }
                if a == b {
                    l[a * r + b] = sum.sqrt();
                } else {
                    l[a * r + b] = sum / l[b * r + b];
                }
            }
        }
        l
    }

    /// Woodbury `log_prob`: evaluates `log N(value; loc, Σ)` for
    /// `Σ = W Wᵀ + diag(D)` without forming the dense `[d, d]` matrix.
    ///
    /// Mirrors `_batch_lowrank_mahalanobis` + `_batch_lowrank_logdet` +
    /// `LowRankMultivariateNormal.log_prob`
    /// (`torch/distributions/lowrank_multivariate_normal.py:28-51,225-240`):
    ///
    /// ```text
    /// log|Σ|       = 2·Σ_a log(L_aa) + Σ_i log(D_i)          (det lemma)
    /// Wt_Dinv_x    = Wᵀ D⁻¹ (value - loc)                    (length r)
    /// M            = Σ_i (value-loc)_i² / D_i
    ///                  - |L⁻¹ Wt_Dinv_x|²                    (Woodbury)
    /// log_prob     = -0.5·(d·ln(2π) + log|Σ| + M)
    /// ```
    ///
    /// where `L = chol(C)`, `C = I_r + Wᵀ D⁻¹ W` the capacitance matrix.
    fn woodbury_log_prob(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let d = self.d;
        let r = self.r;
        let factor_data = self.cov_factor.data_vec()?;
        let diag_data = self.cov_diag.data_vec()?;
        let loc_data = self.loc.data_vec()?;
        let value_data = value.data_vec()?;

        let zero = <T as num_traits::Zero>::zero();

        // The single-event log_prob: `value` must address the d-vector. When
        // batched the inner-MVN path is the documented fallback (the Woodbury
        // override is the single-event hot path, matching the dense-path shape
        // contract pinned by `test_low_rank_log_prob_at_mean_diagonal_only`).
        if value_data.len() != d {
            return self.inner.log_prob(value);
        }

        let l = Self::capacitance_tril(&factor_data, &diag_data, d, r);

        // log|Σ| = 2·Σ_a log(L_aa) + Σ_i log(D_i).
        let two = T::from(2.0).unwrap();
        let mut log_det = zero;
        for a in 0..r {
            log_det += two * l[a * r + a].ln();
        }
        for &di in &diag_data {
            log_det += di.ln();
        }

        // diff = value - loc; mahalanobis_term1 = Σ_i diff_i² / D_i.
        let mut diff = vec![zero; d];
        let mut maha_term1 = zero;
        for i in 0..d {
            let di = value_data[i] - loc_data[i];
            diff[i] = di;
            maha_term1 += di * di / diag_data[i];
        }

        // Wt_Dinv_x[a] = Σ_i W[i, a] * diff_i / D_i   (length r).
        let mut wt_dinv_x = vec![zero; r];
        for a in 0..r {
            let mut acc = zero;
            for i in 0..d {
                acc += factor_data[i * r + a] * diff[i] / diag_data[i];
            }
            wt_dinv_x[a] = acc;
        }

        // mahalanobis_term2 = |L⁻¹ Wt_Dinv_x|²: solve L z = Wt_Dinv_x by
        // forward-substitution (L lower-triangular), then sum z².
        let mut z = vec![zero; r];
        let mut maha_term2 = zero;
        for a in 0..r {
            let mut sum = wt_dinv_x[a];
            for c in 0..a {
                sum = sum - l[a * r + c] * z[c];
            }
            let za = sum / l[a * r + a];
            z[a] = za;
            maha_term2 += za * za;
        }

        let maha = maha_term1 - maha_term2;

        // log_prob = -0.5·(d·ln(2π) + log|Σ| + M).
        let half = T::from(0.5).unwrap();
        let two_pi_ln = T::from((2.0 * std::f64::consts::PI).ln()).unwrap();
        let d_t = T::from(d as f64).unwrap();
        let lp = -half * (d_t * two_pi_ln + log_det + maha);

        let device = value.device();
        let t = Tensor::from_storage(TensorStorage::cpu(vec![lp]), vec![], false)?;
        if device.is_cuda() {
            t.to(device)
        } else {
            Ok(t)
        }
    }
}

impl<T: Float> Distribution<T> for LowRankMultivariateNormal<T> {
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        self.inner.sample(shape)
    }

    fn rsample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        self.inner.rsample(shape)
    }

    fn log_prob(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Woodbury fast path (#1385): O(d·r²) Mahalanobis + matrix-determinant
        // lemma over the r×r capacitance matrix, no dense [d, d] Σ formed.
        // Mirrors `lowrank_multivariate_normal.py:225-240`.
        self.woodbury_log_prob(value)
    }

    fn mean(&self) -> FerrotorchResult<Tensor<T>> {
        // Reference: torch.distributions.LowRankMultivariateNormal.mean — property returns self.loc.
        // mean = loc (the distribution mean vector, shape [d]).
        Ok(self.loc.clone())
    }

    fn entropy(&self) -> FerrotorchResult<Tensor<T>> {
        self.inner.entropy()
    }

    fn variance(&self) -> FerrotorchResult<Tensor<T>> {
        // Reference: torch.distributions.LowRankMultivariateNormal.variance
        // (`torch/distributions/lowrank_multivariate_normal.py:189-196`):
        //   variance = (cov_factor ** 2).sum(-1) + cov_diag
        // Σ_ii = sum_k W_ik^2 + D_i, which is exactly the diagonal of the
        // low-rank-plus-diagonal covariance — avoids materialising the dense
        // [d, d] matrix entirely.
        let factor_data = self.cov_factor.data_vec()?;
        let diag_data = self.cov_diag.data_vec()?;
        let d = self.d;
        let r = self.r;
        let zero = <T as num_traits::Zero>::zero();
        let mut out = Vec::with_capacity(d);
        for i in 0..d {
            let mut sq_sum = zero;
            for k in 0..r {
                let v = factor_data[i * r + k];
                sq_sum += v * v;
            }
            out.push(sq_sum + diag_data[i]);
        }
        let device = self.cov_diag.device();
        let t = Tensor::from_storage(TensorStorage::cpu(out), vec![d], false)?;
        if device.is_cuda() {
            t.to(device)
        } else {
            Ok(t)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_tensor(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
    }

    #[test]
    fn test_low_rank_basic_construction() {
        // d=3, r=1: factor=[3,1], diag=[3].
        let loc = cpu_tensor(&[0.0, 0.0, 0.0], &[3]);
        let factor = cpu_tensor(&[1.0, 0.5, -0.5], &[3, 1]);
        let diag = cpu_tensor(&[1.0, 1.0, 1.0], &[3]);
        let mvn = LowRankMultivariateNormal::new(loc, factor, diag).unwrap();
        assert_eq!(mvn.dim(), 3);
        assert_eq!(mvn.rank(), 1);
    }

    #[test]
    fn test_low_rank_negative_diag_errors() {
        let loc = cpu_tensor(&[0.0, 0.0], &[2]);
        let factor = cpu_tensor(&[1.0, 0.0], &[2, 1]);
        let diag = cpu_tensor(&[1.0, -0.5], &[2]);
        assert!(LowRankMultivariateNormal::new(loc, factor, diag).is_err());
    }

    #[test]
    fn test_low_rank_wrong_factor_shape_errors() {
        let loc = cpu_tensor(&[0.0, 0.0], &[2]);
        // factor [3, 1] doesn't match d=2
        let factor = cpu_tensor(&[1.0, 0.0, 0.5], &[3, 1]);
        let diag = cpu_tensor(&[1.0, 1.0], &[2]);
        assert!(LowRankMultivariateNormal::new(loc, factor, diag).is_err());
    }

    #[test]
    fn test_low_rank_wrong_diag_shape_errors() {
        let loc = cpu_tensor(&[0.0, 0.0], &[2]);
        let factor = cpu_tensor(&[1.0, 0.0], &[2, 1]);
        let diag = cpu_tensor(&[1.0, 1.0, 1.0], &[3]);
        assert!(LowRankMultivariateNormal::new(loc, factor, diag).is_err());
    }

    #[test]
    fn test_low_rank_log_prob_at_mean_diagonal_only() {
        // With cov_factor = 0 (rank 1, all zeros) and cov_diag = ones,
        // the distribution is N(0, I). log_prob at mean = -d/2 log(2pi).
        let loc = cpu_tensor(&[0.0, 0.0, 0.0], &[3]);
        let factor = cpu_tensor(&[0.0, 0.0, 0.0], &[3, 1]);
        let diag = cpu_tensor(&[1.0, 1.0, 1.0], &[3]);
        let mvn = LowRankMultivariateNormal::new(loc, factor, diag).unwrap();
        let value = cpu_tensor(&[0.0, 0.0, 0.0], &[3]);
        let lp = mvn.log_prob(&value).unwrap();
        let val = lp.item().unwrap();
        let expected = -1.5_f32 * (2.0 * std::f32::consts::PI).ln();
        assert!(
            (val - expected).abs() < 1e-4,
            "expected ≈ {expected}, got {val}"
        );
    }

    #[test]
    fn test_low_rank_variance_matches_diag_formula() {
        // Σ = W W^T + diag(D). Variance is the diagonal: sum_k W_ik^2 + D_i.
        // Probe: factor=[1, 2; 0.5, 0.5], diag=[0.1, 0.2].
        // Σ_00 = 1^2 + 2^2 + 0.1 = 5.1
        // Σ_11 = 0.5^2 + 0.5^2 + 0.2 = 0.7
        let loc = cpu_tensor(&[0.0, 0.0], &[2]);
        let factor = cpu_tensor(&[1.0, 2.0, 0.5, 0.5], &[2, 2]);
        let diag = cpu_tensor(&[0.1, 0.2], &[2]);
        let mvn = LowRankMultivariateNormal::new(loc, factor, diag).unwrap();
        let v = mvn.variance().unwrap();
        assert_eq!(v.shape(), &[2]);
        let data = v.data_vec().unwrap();
        assert!((data[0] - 5.1).abs() < 1e-5, "variance[0] = {}", data[0]);
        assert!((data[1] - 0.7).abs() < 1e-5, "variance[1] = {}", data[1]);
    }

    #[test]
    fn test_low_rank_matrix_accessors_present() {
        // scale_tril / covariance_matrix / precision_matrix should return
        // tensors of shape [d, d]. We just check the shape; numerical
        // correctness is exercised by the inner MultivariateNormal tests.
        let loc = cpu_tensor(&[0.0, 0.0, 0.0], &[3]);
        let factor = cpu_tensor(&[0.5, 0.5, 0.5], &[3, 1]);
        let diag = cpu_tensor(&[0.5, 0.5, 0.5], &[3]);
        let mvn = LowRankMultivariateNormal::new(loc, factor, diag).unwrap();
        assert_eq!(mvn.scale_tril().shape(), &[3, 3]);
        assert_eq!(mvn.covariance_matrix().unwrap().shape(), &[3, 3]);
        assert_eq!(mvn.precision_matrix().unwrap().shape(), &[3, 3]);
    }

    /// Dense-path oracle: build Σ = W Wᵀ + diag(D) explicitly, invert it,
    /// and evaluate the multivariate-normal log density directly. Independent
    /// of the Woodbury production path, so it is a genuine cross-check.
    fn dense_log_prob_oracle(
        loc: &[f64],
        factor: &[f64],
        diag: &[f64],
        value: &[f64],
        d: usize,
        r: usize,
    ) -> f64 {
        // Σ
        let mut cov = vec![0.0f64; d * d];
        for i in 0..d {
            for j in 0..d {
                let mut acc = 0.0;
                for k in 0..r {
                    acc += factor[i * r + k] * factor[j * r + k];
                }
                if i == j {
                    acc += diag[i];
                }
                cov[i * d + j] = acc;
            }
        }
        // Invert Σ and get its determinant via Gauss-Jordan with an augmented
        // identity. d is tiny in the tests so naive elimination is fine.
        let mut a = cov.clone();
        let mut inv = vec![0.0f64; d * d];
        for i in 0..d {
            inv[i * d + i] = 1.0;
        }
        let mut det = 1.0;
        for col in 0..d {
            // pivot
            let piv = a[col * d + col];
            det *= piv;
            let piv_inv = 1.0 / piv;
            for j in 0..d {
                a[col * d + j] *= piv_inv;
                inv[col * d + j] *= piv_inv;
            }
            for row in 0..d {
                if row == col {
                    continue;
                }
                let f = a[row * d + col];
                for j in 0..d {
                    a[row * d + j] -= f * a[col * d + j];
                    inv[row * d + j] -= f * inv[col * d + j];
                }
            }
        }
        // Mahalanobis = (x-μ)ᵀ Σ⁻¹ (x-μ)
        let diff: Vec<f64> = (0..d).map(|i| value[i] - loc[i]).collect();
        let mut maha = 0.0;
        for i in 0..d {
            for j in 0..d {
                maha += diff[i] * inv[i * d + j] * diff[j];
            }
        }
        -0.5 * (d as f64 * (2.0 * std::f64::consts::PI).ln() + det.ln() + maha)
    }

    #[test]
    fn test_low_rank_woodbury_matches_dense_path_rank1() {
        // d=3, r=1. Woodbury log_prob must match the explicit dense oracle.
        let loc = [0.5f64, -1.0, 2.0];
        let factor = [1.0f64, 0.5, -0.5]; // [3,1]
        let diag = [1.0f64, 2.0, 0.5];
        let value = [0.0f64, 0.0, 1.0];

        let loc_t = Tensor::from_storage(TensorStorage::cpu(loc.to_vec()), vec![3], false).unwrap();
        let factor_t =
            Tensor::from_storage(TensorStorage::cpu(factor.to_vec()), vec![3, 1], false).unwrap();
        let diag_t =
            Tensor::from_storage(TensorStorage::cpu(diag.to_vec()), vec![3], false).unwrap();
        let value_t =
            Tensor::from_storage(TensorStorage::cpu(value.to_vec()), vec![3], false).unwrap();

        let mvn = LowRankMultivariateNormal::new(loc_t, factor_t, diag_t).unwrap();
        let got = mvn.log_prob(&value_t).unwrap().item().unwrap();
        let expected = dense_log_prob_oracle(&loc, &factor, &diag, &value, 3, 1);
        assert!(
            (got - expected).abs() < 1e-9,
            "Woodbury log_prob {got} vs dense oracle {expected}"
        );
    }

    #[test]
    fn test_low_rank_woodbury_matches_dense_path_rank2() {
        // d=4, r=2 (r < d), a non-trivial low-rank-plus-diag covariance.
        let loc = [1.0f64, 0.0, -0.5, 0.25];
        let factor = [
            0.8f64, -0.2, // row 0
            0.3, 0.6, // row 1
            -0.4, 0.1, // row 2
            0.5, 0.5, // row 3
        ]; // [4,2]
        let diag = [0.5f64, 1.0, 0.75, 1.25];
        let value = [0.2f64, -0.3, 0.4, 0.1];

        let loc_t = Tensor::from_storage(TensorStorage::cpu(loc.to_vec()), vec![4], false).unwrap();
        let factor_t =
            Tensor::from_storage(TensorStorage::cpu(factor.to_vec()), vec![4, 2], false).unwrap();
        let diag_t =
            Tensor::from_storage(TensorStorage::cpu(diag.to_vec()), vec![4], false).unwrap();
        let value_t =
            Tensor::from_storage(TensorStorage::cpu(value.to_vec()), vec![4], false).unwrap();

        let mvn = LowRankMultivariateNormal::new(loc_t, factor_t, diag_t).unwrap();
        let got = mvn.log_prob(&value_t).unwrap().item().unwrap();
        let expected = dense_log_prob_oracle(&loc, &factor, &diag, &value, 4, 2);
        assert!(
            (got - expected).abs() < 1e-9,
            "Woodbury log_prob {got} vs dense oracle {expected}"
        );
    }

    #[test]
    fn test_low_rank_woodbury_diagonal_only_analytic() {
        // W = 0 → Σ = I_3; log_prob at the mean must equal -1.5·ln(2π), the
        // analytic standard-normal value (unchanged from the dense path).
        let loc = cpu_tensor(&[0.0, 0.0, 0.0], &[3]);
        let factor = cpu_tensor(&[0.0, 0.0, 0.0], &[3, 1]);
        let diag = cpu_tensor(&[1.0, 1.0, 1.0], &[3]);
        let mvn = LowRankMultivariateNormal::new(loc, factor, diag).unwrap();
        let value = cpu_tensor(&[0.0, 0.0, 0.0], &[3]);
        let lp = mvn.log_prob(&value).unwrap().item().unwrap();
        let expected = -1.5_f32 * (2.0 * std::f32::consts::PI).ln();
        assert!(
            (lp - expected).abs() < 1e-5,
            "expected {expected}, got {lp}"
        );
    }

    #[test]
    fn test_low_rank_sample_shape() {
        // Inner MultivariateNormal samples with shape `[batch..., event_dim]`
        // where event_dim == d. Passing batch_shape=[10] yields output [10, 3].
        let loc = cpu_tensor(&[0.0, 0.0, 0.0], &[3]);
        let factor = cpu_tensor(&[0.5, 0.5, 0.5], &[3, 1]);
        let diag = cpu_tensor(&[0.1, 0.1, 0.1], &[3]);
        let mvn = LowRankMultivariateNormal::new(loc, factor, diag).unwrap();
        let s = mvn.sample(&[10]).unwrap();
        assert_eq!(s.shape(), &[10, 3]);
    }
}
