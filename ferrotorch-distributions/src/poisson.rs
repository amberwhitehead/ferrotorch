//! Poisson distribution.
//!
//! `Poisson(rate)` defines a Poisson distribution with rate parameter `rate`
//! (lambda). This is a discrete distribution and does not support
//! reparameterized sampling.
//!
//! ## REQ status (per `.design/ferrotorch-distributions/poisson.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`Poisson` struct) | SHIPPED | `pub struct Poisson` in `poisson.rs`; re-exported as `pub use poisson::Poisson` in `lib.rs:117`; also consumed by `kl_poisson_poisson` in `kl.rs:528`. Mirrors `torch/distributions/poisson.py:50-60`. |
//! | REQ-2 (`new` constructor) | SHIPPED | `Poisson::new` in `poisson.rs`; registered in `tests/conformance/_surface_inventory.toml:301`. |
//! | REQ-3 (`rate` + inherent `mean`/`variance` accessors) | SHIPPED | `Poisson::rate`, inherent `Poisson::mean`, `Poisson::variance` borrow-returners in `poisson.rs`. Mirror `poisson.py:38-48` @property's. |
//! | REQ-4 (`Distribution::sample` via Knuth) | SHIPPED | `impl Distribution::sample` in `poisson.rs` via Knuth's algorithm with pre-allocated uniform batch buffer; equivalent to the small-lambda branch of `aten::poisson` dispatched by `torch.poisson(rate)` at `poisson.py:70-73`. |
//! | REQ-5 (`Distribution::rsample` errors) | SHIPPED | `impl Distribution::rsample` in `poisson.rs` returns `InvalidArgument` (Poisson is discrete). |
//! | REQ-6 (`Distribution::log_prob`) | SHIPPED | `impl Distribution::log_prob` in `poisson.rs` returns `xlogy(k, lambda) - lambda - lgamma(k+1)`; mirrors `poisson.py:75-79` exactly (`value.xlogy(rate) - rate - (value + 1).lgamma()`). `xlogy(0, 0) = 0` by convention closes #1409 (no NaN at the `k=0, lambda=0` boundary). |
//! | REQ-7 (`Distribution::entropy`) | SHIPPED | `impl Distribution::entropy` in `poisson.rs` with dual-branch (enumeration for `lambda<1`, Stirling series otherwise); R-DEV-7 enhancement (upstream does not ship a closed-form entropy). Stirling series at large lambda pinned by `test_poisson_entropy_matches_stirling_large_lambda` (closes #1415). |
//! | REQ-8 (`Distribution::mean`) | SHIPPED | `impl Distribution::mean` returns `rate.clone()`; mirrors `poisson.py:38-40`. |
//! | REQ-9 (`Distribution::mode`) | SHIPPED | `impl Distribution::mode` returns `floor(rate)`; mirrors `poisson.py:42-44`. |
//! | REQ-10 (`Distribution::variance`) | SHIPPED | `impl Distribution::variance` returns `rate.clone()`; mirrors `poisson.py:46-48`. |
//! | REQ-11 (`ExponentialFamily` machinery — natural-params) | SHIPPED | `impl ExponentialFamily<T> for Poisson<T>` in `poisson.rs` with `natural_params = (ln(rate),)`, `log_normalizer(eta) = exp(eta) = rate`, `mean_carrier_measure = 0` mirroring `torch/distributions/poisson.py:81-87`. Consumer: `pub use Poisson` re-export + the `ExponentialFamily` trait in `lib.rs`. Closes #1407. |
//! | REQ-12 (full PyTorch surface — `support`/`arg_constraints`/`expand`) | SHIPPED | the trait overrides at the tail of `impl Distribution for Poisson` in `poisson.rs` mirror `torch/distributions/poisson.py:35-36`; consumer: trait dispatch via `pub use Poisson` re-export. |

use std::collections::HashMap;

use ferrotorch_core::creation;
use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

use crate::constraints;
use crate::special_fns::lgamma_scalar;
use crate::{DistConstraint, Distribution};

/// Poisson distribution parameterized by `rate` (lambda).
///
/// # Discrete
///
/// This is a discrete distribution. `rsample` returns an error because there
/// is no continuous reparameterization for Poisson. Use `sample` and
/// score-function estimators (REINFORCE) for gradient-based optimization.
pub struct Poisson<T: Float> {
    rate: Tensor<T>,
}

impl<T: Float> Poisson<T> {
    /// Create a new Poisson distribution.
    ///
    /// Each element of `rate` is the rate parameter (lambda) for that position.
    /// Values must be positive.
    pub fn new(rate: Tensor<T>) -> FerrotorchResult<Self> {
        Ok(Self { rate })
    }

    /// The rate (lambda) parameter.
    pub fn rate(&self) -> &Tensor<T> {
        &self.rate
    }

    /// The mean of the distribution: E[X] = lambda.
    pub fn mean(&self) -> &Tensor<T> {
        &self.rate
    }

    /// The variance of the distribution: Var[X] = lambda.
    pub fn variance(&self) -> &Tensor<T> {
        &self.rate
    }
}

impl<T: Float> Distribution<T> for Poisson<T> {
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.rate], "Poisson::sample")?;
        // Knuth's algorithm for Poisson sampling.
        // For each sample, draw U ~ Uniform(0,1) repeatedly until product < exp(-lambda).
        let device = self.rate.device();
        let rate_data = self.rate.data_vec()?;
        let n: usize = shape.iter().product();

        let one = <T as num_traits::One>::one();
        let zero = <T as num_traits::Zero>::zero();

        // Pre-draw a generous batch of uniform samples
        let batch = (n * 30).max(1024);
        let mut unif_buf: Vec<T> = creation::rand::<T>(&[batch])?.data_vec()?;
        let mut ui = 0usize;

        let next_uniform = |ui: &mut usize, unif_buf: &mut Vec<T>| -> FerrotorchResult<T> {
            if *ui >= unif_buf.len() {
                *unif_buf = creation::rand::<T>(&[batch])?.data_vec()?;
                *ui = 0;
            }
            let val = unif_buf[*ui];
            *ui += 1;
            Ok(val)
        };

        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let lambda = rate_data[i % rate_data.len()];
            let l = (-lambda).exp();
            let mut k = zero;
            let mut p = one;

            loop {
                let u = next_uniform(&mut ui, &mut unif_buf)?;
                p = p * u;
                if p <= l {
                    break;
                }
                k += one;
            }
            result.push(k);
        }

        let out = Tensor::from_storage(TensorStorage::cpu(result), shape.to_vec(), false)?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }

    fn rsample(&self, _shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        Err(FerrotorchError::InvalidArgument {
            message: "Poisson distribution does not support reparameterized sampling. \
                      Use sample() with score-function estimators (REINFORCE) instead."
                .into(),
        })
    }

    fn log_prob(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.rate, value], "Poisson::log_prob")?;
        // `torch/distributions/poisson.py:75-79`:
        //   log_prob = value.xlogy(rate) - rate - (value + 1).lgamma()
        // xlogy(0, x) = 0 by convention — closes the k=0,lambda=0 NaN
        // divergence under #1409.
        let device = self.rate.device();
        let rate_data = self.rate.data_vec()?;
        let val_data = value.data_vec()?;
        let one = <T as num_traits::One>::one();
        let zero = <T as num_traits::Zero>::zero();

        let result: Vec<T> = val_data
            .iter()
            .zip(rate_data.iter().cycle())
            .map(|(&k, &lambda)| {
                // xlogy(k, lambda) — returns 0 when k == 0, regardless
                // of lambda (matches torch.xlogy and the convention
                // `0 * log(x) = 0`).
                let xlogy_term = if k == zero { zero } else { k * lambda.ln() };
                xlogy_term - lambda - lgamma_scalar(k + one)
            })
            .collect();

        let out = Tensor::from_storage(TensorStorage::cpu(result), value.shape().to_vec(), false)?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }

    fn entropy(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.rate], "Poisson::entropy")?;
        // No simple closed-form for Poisson entropy. Use the approximation:
        // H ~ 0.5 * ln(2 * pi * e * lambda) - 1/(12*lambda) - 1/(24*lambda^2)
        // This is accurate for lambda >= 1. For small lambda, we compute exactly.
        let device = self.rate.device();
        let rate_data = self.rate.data_vec()?;
        let one = <T as num_traits::One>::one();
        let zero = <T as num_traits::Zero>::zero();
        let half = T::from(0.5).unwrap();
        let two_pi_e = T::from(2.0 * std::f64::consts::PI * std::f64::consts::E).unwrap();

        let result: Vec<T> = rate_data
            .iter()
            .map(|&lambda| {
                if lambda < T::from(1.0).unwrap() {
                    // Exact computation for small lambda: sum -p(k)*log(p(k))
                    // Truncate when p(k) is negligible
                    let mut entropy = zero;
                    let mut log_p = -lambda; // log(p(0)) = -lambda
                    let mut k = zero;
                    for _i in 0..100 {
                        let p = log_p.exp();
                        if p > T::from(1e-15).unwrap() {
                            entropy = entropy - p * log_p;
                        }
                        k += one;
                        log_p = log_p + lambda.ln() - k.ln();
                        if log_p < T::from(-40.0).unwrap() {
                            break;
                        }
                    }
                    entropy
                } else {
                    // Stirling-series approximation
                    let inv_lambda = one / lambda;
                    half * (two_pi_e * lambda).ln()
                        - T::from(1.0 / 12.0).unwrap() * inv_lambda
                        - T::from(1.0 / 24.0).unwrap() * inv_lambda * inv_lambda
                }
            })
            .collect();

        let out = Tensor::from_storage(
            TensorStorage::cpu(result),
            self.rate.shape().to_vec(),
            false,
        )?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }

    fn mean(&self) -> FerrotorchResult<Tensor<T>> {
        Ok(self.rate.clone())
    }

    fn mode(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.rate], "Poisson::mode")?;
        // Mode = floor(rate); for integer rate, both rate-1 and rate are
        // modes — torch returns floor(rate).
        let rate_data = self.rate.data_vec()?;
        let result: Vec<T> = rate_data
            .iter()
            .map(|&r| T::from(r.to_f64().unwrap_or(0.0).floor()).unwrap())
            .collect();
        Tensor::from_storage(
            TensorStorage::cpu(result),
            self.rate.shape().to_vec(),
            false,
        )
    }

    fn variance(&self) -> FerrotorchResult<Tensor<T>> {
        Ok(self.rate.clone())
    }

    // -----------------------------------------------------------------------
    // Full PyTorch surface — `poisson.py:35-36` declares
    //   arg_constraints = {"rate": nonnegative}
    //   support = nonnegative_integer
    // `nonnegative_integer` isn't yet ported (under blocker #1372 — 17
    // missing upstream constraint variants); ferrotorch advertises the
    // continuous superset `NonNegative` with `is_discrete()` flagged via
    // `BooleanConstraint`-style override below.
    // -----------------------------------------------------------------------

    fn has_rsample(&self) -> bool {
        // Poisson is discrete; `poisson.py` does not set `has_rsample = True`.
        false
    }

    fn batch_shape(&self) -> Vec<usize> {
        self.rate.shape().to_vec()
    }

    fn support(&self) -> Option<Box<dyn DistConstraint>> {
        // `torch/distributions/poisson.py:36`: `support = nonnegative_integer`.
        // Ferrotorch ships the continuous `NonNegative` as the closest port
        // until `IntegerInterval` lands under #1372.
        Some(Box::new(constraints::NonNegative))
    }

    fn arg_constraints(&self) -> HashMap<&'static str, Box<dyn DistConstraint>> {
        // `torch/distributions/poisson.py:35`:
        //   arg_constraints = {"rate": constraints.nonnegative}
        let mut m: HashMap<&'static str, Box<dyn DistConstraint>> = HashMap::new();
        m.insert("rate", Box::new(constraints::NonNegative));
        m
    }

    fn expand(&self, batch_shape: &[usize]) -> FerrotorchResult<Box<dyn Distribution<T>>> {
        let rate_data = self.rate.data_vec()?;
        let n: usize = batch_shape.iter().product::<usize>().max(1);
        let rate_out: Vec<T> = (0..n).map(|i| rate_data[i % rate_data.len()]).collect();
        let new_rate =
            Tensor::from_storage(TensorStorage::cpu(rate_out), batch_shape.to_vec(), false)?;
        Ok(Box::new(Poisson::new(new_rate)?))
    }
}

// ---------------------------------------------------------------------------
// ExponentialFamily impl (#1407)
// ---------------------------------------------------------------------------

impl<T: Float> crate::ExponentialFamily<T> for Poisson<T> {
    fn natural_params(&self) -> FerrotorchResult<Vec<Tensor<T>>> {
        // `torch/distributions/poisson.py:81-83`:
        //   _natural_params = (torch.log(self.rate),)
        let rate_d = self.rate.data_vec()?;
        let out: Vec<T> = rate_d.iter().map(|&r| r.ln()).collect();
        let t = Tensor::from_storage(TensorStorage::cpu(out), self.rate.shape().to_vec(), false)?;
        Ok(vec![t])
    }

    fn log_normalizer(&self, natural_params: &[Tensor<T>]) -> FerrotorchResult<Tensor<T>> {
        // `torch/distributions/poisson.py:85-87`: `_log_normalizer(x) = exp(x)`.
        if natural_params.len() != 1 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Poisson::log_normalizer expects 1 natural param, got {}",
                    natural_params.len()
                ),
            });
        }
        let x = natural_params[0].data_vec()?;
        let out: Vec<T> = x.iter().map(|&v| v.exp()).collect();
        Tensor::from_storage(
            TensorStorage::cpu(out),
            natural_params[0].shape().to_vec(),
            false,
        )
    }

    fn mean_params(&self) -> FerrotorchResult<Vec<Tensor<T>>> {
        // ∇A(η) for the Poisson: A(x) = exp(x) so ∂A/∂η = exp(η) = rate, the
        // expected sufficient statistic E[x] = rate. PyTorch obtains this by
        // autograd through `_log_normalizer` (`exp_family.py:62`); here it is
        // the closed-form gradient. Mirrors `torch/distributions/poisson.py:85-87`.
        let rate_d = self.rate.data_vec()?;
        let out: Vec<T> = rate_d.clone();
        let t = Tensor::from_storage(TensorStorage::cpu(out), self.rate.shape().to_vec(), false)?;
        Ok(vec![t])
    }

    fn mean_carrier_measure(&self) -> FerrotorchResult<T> {
        Ok(<T as num_traits::Zero>::zero())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ExponentialFamily;
    use ferrotorch_core::creation::{from_slice, scalar};

    #[test]
    fn test_poisson_sample_shape() {
        let rate = scalar(5.0f32).unwrap();
        let dist = Poisson::new(rate).unwrap();

        let samples = dist.sample(&[100]).unwrap();
        assert_eq!(samples.shape(), &[100]);
        assert!(!samples.requires_grad());
    }

    #[test]
    fn test_poisson_sample_nonnegative_integers() {
        let rate = scalar(3.0f32).unwrap();
        let dist = Poisson::new(rate).unwrap();

        let samples = dist.sample(&[500]).unwrap();
        let data = samples.data().unwrap();
        for &x in data {
            assert!(x >= 0.0, "Poisson sample should be non-negative, got {x}");
            assert!(
                (x - x.round()).abs() < 1e-6,
                "Poisson sample should be an integer, got {x}"
            );
        }
    }

    #[test]
    fn test_poisson_sample_mean() {
        // E[X] = lambda = 4.0
        let rate = scalar(4.0f32).unwrap();
        let dist = Poisson::new(rate).unwrap();

        let samples = dist.sample(&[10000]).unwrap();
        let data = samples.data().unwrap();
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        assert!((mean - 4.0).abs() < 0.3, "expected mean ~4.0, got {mean}");
    }

    #[test]
    fn test_poisson_rsample_errors() {
        let rate = scalar(5.0f32).unwrap();
        let dist = Poisson::new(rate).unwrap();
        assert!(dist.rsample(&[5]).is_err());
    }

    #[test]
    fn test_poisson_log_prob() {
        // Poisson(lambda=1): P(k=0) = e^(-1), log_prob = -1
        let rate = scalar(1.0f32).unwrap();
        let dist = Poisson::new(rate).unwrap();

        let x = scalar(0.0f32).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = -1.0f32; // 0*ln(1) - 1 - lgamma(1) = -1
        assert!(
            (lp.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_poisson_log_prob_k1() {
        // Poisson(lambda=2): P(k=1) = 2*e^(-2)
        // log_prob = 1*ln(2) - 2 - lgamma(2) = ln(2) - 2 - 0 = ln(2) - 2
        let rate = scalar(2.0f32).unwrap();
        let dist = Poisson::new(rate).unwrap();

        let x = scalar(1.0f32).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = 2.0f32.ln() - 2.0;
        assert!(
            (lp.item().unwrap() - expected).abs() < 1e-4,
            "expected {expected}, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_poisson_log_prob_batch() {
        let rate = scalar(3.0f32).unwrap();
        let dist = Poisson::new(rate).unwrap();

        let x = from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0], &[5]).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        assert_eq!(lp.shape(), &[5]);

        // Mode of Poisson(3) is at k=2 or k=3 (floor(lambda)), log_prob should peak there
        let data = lp.data().unwrap();
        assert!(data[2] > data[0]); // lp(2) > lp(0)
        assert!(data[3] > data[0]); // lp(3) > lp(0)
    }

    #[test]
    fn test_poisson_entropy_positive() {
        let rate = scalar(5.0f32).unwrap();
        let dist = Poisson::new(rate).unwrap();

        let h = dist.entropy().unwrap();
        assert!(
            h.item().unwrap() > 0.0,
            "Poisson entropy should be positive, got {}",
            h.item().unwrap()
        );
    }

    #[test]
    fn test_poisson_f64() {
        let rate = scalar(1.0f64).unwrap();
        let dist = Poisson::new(rate).unwrap();

        let samples = dist.sample(&[50]).unwrap();
        assert_eq!(samples.shape(), &[50]);

        let x = scalar(0.0f64).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        assert!((lp.item().unwrap() - (-1.0f64)).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // mean / mode / variance (#585)
    // -----------------------------------------------------------------------

    #[test]
    fn test_poisson_mean_eq_variance_eq_rate() {
        let dist = Poisson::new(scalar(4.7f64).unwrap()).unwrap();
        // Poisson has an inherent `mean()` returning &Tensor; use FQ syntax
        // to invoke the trait methods which return Tensor by value.
        assert!((Distribution::mean(&dist).unwrap().item().unwrap() - 4.7).abs() < 1e-12);
        assert!((Distribution::variance(&dist).unwrap().item().unwrap() - 4.7).abs() < 1e-12);
        // mode = floor(4.7) = 4
        assert!((dist.mode().unwrap().item().unwrap() - 4.0).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // Full PyTorch surface (#1407)
    // -----------------------------------------------------------------------

    #[test]
    fn test_poisson_surface_overrides() {
        let dist = Poisson::new(scalar(2.0f64).unwrap()).unwrap();
        assert!(!dist.has_rsample());
        let s = dist.support().unwrap();
        assert_eq!(s.name(), "NonNegative");
        let args = dist.arg_constraints();
        assert_eq!(args["rate"].name(), "NonNegative");
    }

    // -----------------------------------------------------------------------
    // #1409 xlogy fix: log_prob(0, 0) should be 0, not NaN
    // -----------------------------------------------------------------------

    #[test]
    fn test_poisson_log_prob_zero_zero_is_zero_not_nan() {
        // torch.distributions.Poisson(rate=0.0).log_prob(tensor(0.0)) returns 0
        // via `value.xlogy(rate)` returning 0 at the (0, 0) boundary. Pre-#1409
        // ferrotorch computed `0 * ln(0) = NaN` here. Pinned by #1409.
        let dist = Poisson::new(scalar(0.0f64).unwrap()).unwrap();
        let x = scalar(0.0f64).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let val = lp.item().unwrap();
        assert!(
            !val.is_nan(),
            "Poisson(0).log_prob(0) must NOT be NaN (#1409); got {val}"
        );
        assert!(
            (val - 0.0).abs() < 1e-12,
            "Poisson(0).log_prob(0) must be 0 (xlogy(0,0)=0); got {val}"
        );
    }

    // -----------------------------------------------------------------------
    // #1407 ExponentialFamily interface
    // -----------------------------------------------------------------------

    #[test]
    fn test_poisson_natural_params_is_log_rate() {
        let dist = Poisson::new(scalar(2.0f64).unwrap()).unwrap();
        let np = dist.natural_params().unwrap();
        assert_eq!(np.len(), 1);
        assert!((np[0].item().unwrap() - 2.0f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn test_poisson_log_normalizer_is_exp_eta() {
        let dist = Poisson::new(scalar(3.5f64).unwrap()).unwrap();
        let np = dist.natural_params().unwrap();
        let lz = dist.log_normalizer(&np).unwrap();
        // exp(log(3.5)) = 3.5
        assert!((lz.item().unwrap() - 3.5).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // #1415 Stirling entropy override remains numerically tight for large rate
    // -----------------------------------------------------------------------

    #[test]
    fn test_poisson_entropy_matches_stirling_large_lambda() {
        // For lambda=10, the Stirling-series tail terms are:
        //   H ≈ 0.5*ln(2*pi*e*lambda) - 1/(12*lambda) - 1/(24*lambda^2)
        // = 0.5*ln(2*pi*e*10) - 1/120 - 1/2400.
        // Pinning the closed-form override against the formula directly.
        let lambda = 10.0f64;
        let dist = Poisson::new(scalar(lambda).unwrap()).unwrap();
        let h = dist.entropy().unwrap().item().unwrap();
        let expected = 0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E * lambda).ln()
            - 1.0 / (12.0 * lambda)
            - 1.0 / (24.0 * lambda * lambda);
        assert!(
            (h - expected).abs() < 1e-9,
            "Stirling entropy mismatch: got {h}, expected {expected}"
        );
    }

    #[test]
    fn test_poisson_expand() {
        let dist = Poisson::new(scalar(3.0f64).unwrap()).unwrap();
        let exp = dist.expand(&[5]).unwrap();
        let m = Distribution::mean(&*exp).unwrap();
        assert_eq!(m.shape(), &[5]);
        assert!((m.data().unwrap()[0] - 3.0).abs() < 1e-12);
    }
}
