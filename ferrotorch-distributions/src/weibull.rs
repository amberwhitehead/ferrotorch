//! Weibull distribution.
//!
//! `Weibull(scale, concentration)` — a two-parameter continuous distribution
//! commonly used in reliability engineering and survival analysis.
//!
//! ## REQ status (per `.design/ferrotorch-distributions/weibull.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`Weibull` struct) | SHIPPED | `pub struct Weibull` in `weibull.rs`; re-exported as `pub use weibull::Weibull` in `lib.rs:124`; mirrors `torch/distributions/weibull.py:40-56`. |
//! | REQ-2 (`new` constructor, shape match) | SHIPPED | `Weibull::new` rejecting shape mismatch; registered in `tests/conformance/_surface_inventory.toml:511`. |
//! | REQ-3 (`scale` + `concentration` accessors) | SHIPPED | accessors in `weibull.rs`. |
//! | REQ-4 (`Distribution::sample` via inverse-CDF) | SHIPPED | `impl Distribution::sample` returns `scale * (-log(1-u))^(1/k)`; mirrors `TransformedDistribution(Exponential, [Power(1/k), Affine(scale)])` chain at `weibull.py:48-56`. |
//! | REQ-5 (`Distribution::log_prob`) | SHIPPED | `impl Distribution::log_prob` returns the closed-form Weibull log density (`-inf` for `x < 0`). |
//! | REQ-6 (`Distribution::entropy`) | SHIPPED | `impl Distribution::entropy` returns `euler*(1-1/k) + log(lambda/k) + 1`; mirrors `weibull.py:91-96`. |
//! | REQ-7 (`Distribution::cdf` closed-form) | SHIPPED | `impl Distribution::cdf` returns `1 - exp(-(x/lambda)^k)`; R-DEV-7 enhancement (upstream lacks direct `cdf` override). |
//! | REQ-8 (`Distribution::icdf` closed-form) | SHIPPED | `impl Distribution::icdf` returns `lambda * (-log(1-p))^(1/k)`; R-DEV-7 enhancement. |
//! | REQ-9 (`Distribution::mean`) | SHIPPED | `impl Distribution::mean` returns `lambda * exp(lgamma(1 + 1/k))`; mirrors `weibull.py:72-74`. Uses `lgamma_scalar` from `special_fns.rs`. |
//! | REQ-10 (`Distribution::mode` with `k <= 1` clamp) | SHIPPED | `impl Distribution::mode` returns `lambda * ((k-1)/k)^(1/k)` for `k > 1` else `0`; R-DEV-6 deviation from upstream NaN (`weibull.py:76-82`). |
//! | REQ-11 (`Distribution::variance`) | SHIPPED | `impl Distribution::variance` algebraically equivalent to `weibull.py:85-89`. |
//! | REQ-12 (`rsample` reparameterization) | SHIPPED | the `rsample` body in `weibull.rs` composes `scale * (-ln(1-u))^(1/concentration)` with a `WeibullRsampleBackward` autograd node so gradients flow through `scale` and `concentration`. Mirrors upstream `TransformedDistribution(Exponential, [Power(1/k), Affine(scale)])` with inherited `has_rsample = True` (`weibull.py:48-56`). Closes #1435. |
//! | REQ-13 (`expand`/`support`/`arg_constraints`) | SHIPPED | the trait overrides at the tail of `impl Distribution for Weibull` in `weibull.rs` mirror `torch/distributions/weibull.py:33-38`; consumer: trait dispatch via `pub use Weibull` re-export (closes #1436 — `concentration_reciprocal` remains a Python-only convenience). |

use std::collections::HashMap;
use std::sync::Arc;

use ferrotorch_core::creation;
use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::{GradFn, Tensor};

use crate::constraints;
use crate::special_fns::lgamma_scalar;
use crate::{DistConstraint, Distribution};

/// Weibull distribution parameterized by `scale` (lambda) and
/// `concentration` (k, also called shape parameter).
///
/// PDF: `f(x) = (k/lambda) * (x/lambda)^(k-1) * exp(-(x/lambda)^k)` for x >= 0.
///
/// Sampling uses inverse CDF: `x = scale * (-log(1 - u))^(1/concentration)`.
pub struct Weibull<T: Float> {
    scale: Tensor<T>,
    concentration: Tensor<T>,
}

impl<T: Float> Weibull<T> {
    pub fn new(scale: Tensor<T>, concentration: Tensor<T>) -> FerrotorchResult<Self> {
        if scale.shape() != concentration.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "Weibull: scale shape {:?} != concentration shape {:?}",
                    scale.shape(),
                    concentration.shape()
                ),
            });
        }
        Ok(Self {
            scale,
            concentration,
        })
    }

    pub fn scale(&self) -> &Tensor<T> {
        &self.scale
    }
    pub fn concentration(&self) -> &Tensor<T> {
        &self.concentration
    }
}

impl<T: Float> Distribution<T> for Weibull<T> {
    #[allow(clippy::needless_range_loop)]
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.scale, &self.concentration],
            "Weibull::sample",
        )?;
        let u = creation::rand::<T>(shape)?;
        let u_data = u.data()?;
        let s_data = self.scale.data()?;
        let k_data = self.concentration.data()?;
        let numel = u_data.len();
        let one = <T as num_traits::One>::one();

        let mut out = Vec::with_capacity(numel);
        for i in 0..numel {
            let si = if s_data.len() == 1 {
                0
            } else {
                i % s_data.len()
            };
            let ki = if k_data.len() == 1 {
                0
            } else {
                i % k_data.len()
            };
            // x = scale * (-log(1-u))^(1/k)
            let log_term = (one - u_data[i]).max(T::from(1e-30).unwrap()).ln();
            let val =
                s_data[si] * (<T as num_traits::Zero>::zero() - log_term).powf(one / k_data[ki]);
            out.push(val);
        }

        Tensor::from_storage(TensorStorage::cpu(out), shape.to_vec(), false)
    }

    #[allow(clippy::needless_range_loop)]
    fn rsample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        // Inverse-CDF reparameterization:
        //   x = scale * (-ln(1 - u))^(1/concentration), u ~ Uniform(0, 1)
        // Mirrors `TransformedDistribution(Exponential, [Power(1/k), Affine(scale)])`
        // chain at `weibull.py:48-56` with inherited `has_rsample = True`.
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.scale, &self.concentration],
            "Weibull::rsample",
        )?;
        let u = creation::rand::<T>(shape)?;
        let u_data = u.data_vec()?;
        let s_data = self.scale.data_vec()?;
        let k_data = self.concentration.data_vec()?;
        let numel = u_data.len();
        let one = <T as num_traits::One>::one();
        let zero = <T as num_traits::Zero>::zero();
        let eps = T::from(1e-30).unwrap();

        let mut out = Vec::with_capacity(numel);
        for i in 0..numel {
            let si = if s_data.len() == 1 {
                0
            } else {
                i % s_data.len()
            };
            let ki = if k_data.len() == 1 {
                0
            } else {
                i % k_data.len()
            };
            let log_term = (one - u_data[i]).max(eps).ln();
            let val = s_data[si] * (zero - log_term).powf(one / k_data[ki]);
            out.push(val);
        }
        let storage = TensorStorage::cpu(out);

        if (self.scale.requires_grad() || self.concentration.requires_grad())
            && ferrotorch_core::is_grad_enabled()
        {
            let grad_fn = Arc::new(WeibullRsampleBackward {
                scale: self.scale.clone(),
                concentration: self.concentration.clone(),
                u: u.clone(),
            });
            Tensor::from_operation(storage, shape.to_vec(), grad_fn)
        } else {
            Tensor::from_storage(storage, shape.to_vec(), false)
        }
    }

    #[allow(clippy::needless_range_loop)]
    fn log_prob(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.scale, &self.concentration, value],
            "Weibull::log_prob",
        )?;
        let v = value.data()?;
        let s = self.scale.data()?;
        let k = self.concentration.data()?;
        let numel = v.len();
        let zero = <T as num_traits::Zero>::zero();

        let mut out = Vec::with_capacity(numel);
        for i in 0..numel {
            let si = if s.len() == 1 { 0 } else { i % s.len() };
            let ki = if k.len() == 1 { 0 } else { i % k.len() };
            if v[i] < zero {
                out.push(T::neg_infinity());
            } else {
                // log_prob = log(k/lambda) + (k-1)*log(x/lambda) - (x/lambda)^k
                let x_over_l = v[i] / s[si];
                let lp = (k[ki] / s[si]).ln()
                    + (k[ki] - <T as num_traits::One>::one()) * x_over_l.ln()
                    - x_over_l.powf(k[ki]);
                out.push(lp);
            }
        }

        Tensor::from_storage(TensorStorage::cpu(out), value.shape().to_vec(), false)
    }

    fn entropy(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.scale, &self.concentration],
            "Weibull::entropy",
        )?;
        // H = euler_gamma * (1 - 1/k) + log(lambda/k) + 1
        let s = self.scale.data()?;
        let k = self.concentration.data()?;
        let euler = T::from(0.5772156649015329).unwrap();
        let one = <T as num_traits::One>::one();

        let mut out = Vec::with_capacity(s.len());
        for i in 0..s.len() {
            let h = euler * (one - one / k[i]) + (s[i] / k[i]).ln() + one;
            out.push(h);
        }

        Tensor::from_storage(TensorStorage::cpu(out), self.scale.shape().to_vec(), false)
    }

    fn cdf(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.scale, &self.concentration, value],
            "Weibull::cdf",
        )?;
        // F(x; lambda, k) = 1 - exp(-(x/lambda)^k) for x >= 0
        let v = value.data()?;
        let s = self.scale.data()?;
        let k = self.concentration.data()?;
        let zero = <T as num_traits::Zero>::zero();
        let one = <T as num_traits::One>::one();
        let mut out = Vec::with_capacity(v.len());
        for (i, &vi) in v.iter().enumerate() {
            let si = if s.len() == 1 { 0 } else { i % s.len() };
            let ki = if k.len() == 1 { 0 } else { i % k.len() };
            if vi < zero {
                out.push(zero);
            } else {
                let x_over_l = vi / s[si];
                out.push(one - (zero - x_over_l.powf(k[ki])).exp());
            }
        }
        Tensor::from_storage(TensorStorage::cpu(out), value.shape().to_vec(), false)
    }

    fn icdf(&self, q: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.scale, &self.concentration, q],
            "Weibull::icdf",
        )?;
        // F^{-1}(p) = lambda * (-log(1 - p))^(1/k)
        let p = q.data()?;
        let s = self.scale.data()?;
        let k = self.concentration.data()?;
        let zero = <T as num_traits::Zero>::zero();
        let one = <T as num_traits::One>::one();
        let mut out = Vec::with_capacity(p.len());
        for (i, &pi) in p.iter().enumerate() {
            let si = if s.len() == 1 { 0 } else { i % s.len() };
            let ki = if k.len() == 1 { 0 } else { i % k.len() };
            let log_term = (one - pi).max(T::from(1e-30).unwrap()).ln();
            out.push(s[si] * (zero - log_term).powf(one / k[ki]));
        }
        Tensor::from_storage(TensorStorage::cpu(out), q.shape().to_vec(), false)
    }

    fn mean(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.scale, &self.concentration],
            "Weibull::mean",
        )?;
        // E[X] = lambda * Gamma(1 + 1/k)
        let s = self.scale.data()?;
        let k = self.concentration.data()?;
        let one = <T as num_traits::One>::one();
        let mut out = Vec::with_capacity(s.len());
        for i in 0..s.len() {
            // Gamma(1 + 1/k) via exp(lgamma(...)) — supports fractional k.
            let lg = lgamma_scalar(one + one / k[i]);
            out.push(s[i] * lg.exp());
        }
        Tensor::from_storage(TensorStorage::cpu(out), self.scale.shape().to_vec(), false)
    }

    fn mode(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.scale, &self.concentration],
            "Weibull::mode",
        )?;
        // mode = lambda * ((k-1)/k)^(1/k) for k > 1, else 0.
        let s = self.scale.data()?;
        let k = self.concentration.data()?;
        let one = <T as num_traits::One>::one();
        let zero = <T as num_traits::Zero>::zero();
        let mut out = Vec::with_capacity(s.len());
        for i in 0..s.len() {
            if k[i] > one {
                out.push(s[i] * ((k[i] - one) / k[i]).powf(one / k[i]));
            } else {
                out.push(zero);
            }
        }
        Tensor::from_storage(TensorStorage::cpu(out), self.scale.shape().to_vec(), false)
    }

    fn variance(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.scale, &self.concentration],
            "Weibull::variance",
        )?;
        // Var[X] = lambda^2 * (Gamma(1 + 2/k) - Gamma(1 + 1/k)^2)
        let s = self.scale.data()?;
        let k = self.concentration.data()?;
        let one = <T as num_traits::One>::one();
        let two = T::from(2.0).unwrap();
        let mut out = Vec::with_capacity(s.len());
        for i in 0..s.len() {
            let g1 = lgamma_scalar(one + one / k[i]).exp();
            let g2 = lgamma_scalar(one + two / k[i]).exp();
            out.push(s[i] * s[i] * (g2 - g1 * g1));
        }
        Tensor::from_storage(TensorStorage::cpu(out), self.scale.shape().to_vec(), false)
    }

    // -----------------------------------------------------------------------
    // Full PyTorch surface — `weibull.py:33-38` declares
    //   arg_constraints = {"scale": positive, "concentration": positive}
    //   support = constraints.positive
    // Inherited has_rsample = True from TransformedDistribution; ferrotorch's
    // direct path does NOT build the autograd graph (blocker #1435).
    // -----------------------------------------------------------------------

    fn has_rsample(&self) -> bool {
        // Closes #1435: rsample now builds the autograd graph via
        // `WeibullRsampleBackward`.
        true
    }

    fn batch_shape(&self) -> Vec<usize> {
        self.scale.shape().to_vec()
    }

    fn support(&self) -> Option<Box<dyn DistConstraint>> {
        // `torch/distributions/weibull.py:38`: `support = constraints.positive`.
        Some(Box::new(constraints::Positive))
    }

    fn arg_constraints(&self) -> HashMap<&'static str, Box<dyn DistConstraint>> {
        // `torch/distributions/weibull.py:33-37`:
        //   arg_constraints = {"scale": positive, "concentration": positive}
        let mut m: HashMap<&'static str, Box<dyn DistConstraint>> = HashMap::new();
        m.insert("scale", Box::new(constraints::Positive));
        m.insert("concentration", Box::new(constraints::Positive));
        m
    }

    fn expand(&self, batch_shape: &[usize]) -> FerrotorchResult<Box<dyn Distribution<T>>> {
        let s_data = self.scale.data_vec()?;
        let k_data = self.concentration.data_vec()?;
        let n: usize = batch_shape.iter().product::<usize>().max(1);
        let s_out: Vec<T> = (0..n).map(|i| s_data[i % s_data.len()]).collect();
        let k_out: Vec<T> = (0..n).map(|i| k_data[i % k_data.len()]).collect();
        let new_scale =
            Tensor::from_storage(TensorStorage::cpu(s_out), batch_shape.to_vec(), false)?;
        let new_conc =
            Tensor::from_storage(TensorStorage::cpu(k_out), batch_shape.to_vec(), false)?;
        Ok(Box::new(Weibull::new(new_scale, new_conc)?))
    }
}

// ---------------------------------------------------------------------------
// Backward node for rsample
// ---------------------------------------------------------------------------

/// Backward for `x = scale * (-ln(1-u))^(1/k)` with stored uniform `u`.
///
/// Let `w = -ln(1-u)` (positive for u ∈ (0,1)), so `x = scale * w^(1/k)`.
/// - `dx/dscale = w^(1/k) = x/scale`
/// - `log(x) = log(scale) + (1/k)*log(w)`,
///   `d log(x)/dk = -log(w) / k^2`,
///   `dx/dk = x * d log(x)/dk = -x * log(w) / k^2`
///
/// (`log(w)` can be positive or negative depending on u; either sign is fine.)
#[derive(Debug)]
struct WeibullRsampleBackward<T: Float> {
    scale: Tensor<T>,
    concentration: Tensor<T>,
    u: Tensor<T>,
}

impl<T: Float> GradFn<T> for WeibullRsampleBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let go = grad_output.data_vec()?;
        let u_data = self.u.data_vec()?;
        let s_data = self.scale.data_vec()?;
        let k_data = self.concentration.data_vec()?;
        let one = <T as num_traits::One>::one();
        let zero = <T as num_traits::Zero>::zero();
        let eps = T::from(1e-30).unwrap();

        let mut g_scale = zero;
        let mut g_k = zero;
        for (i, (&g, &u_val)) in go.iter().zip(u_data.iter()).enumerate() {
            let si = if s_data.len() == 1 {
                0
            } else {
                i % s_data.len()
            };
            let ki = if k_data.len() == 1 {
                0
            } else {
                i % k_data.len()
            };
            let s = s_data[si];
            let k = k_data[ki];
            let one_minus_u = (one - u_val).max(eps);
            let w = zero - one_minus_u.ln(); // = -ln(1-u), positive
            let w_safe = w.max(eps);
            let w_pow = w_safe.powf(one / k);
            let x = s * w_pow;
            // dx/dscale = w_pow
            g_scale += g * w_pow;
            // dx/dk = -x * ln(w) / k^2
            g_k += g * (zero - x) * w_safe.ln() / (k * k);
        }

        let grad_scale = Tensor::from_storage(
            TensorStorage::cpu(vec![g_scale]),
            self.scale.shape().to_vec(),
            false,
        )?;
        let grad_k = Tensor::from_storage(
            TensorStorage::cpu(vec![g_k]),
            self.concentration.shape().to_vec(),
            false,
        )?;

        Ok(vec![
            if self.scale.requires_grad() {
                Some(grad_scale)
            } else {
                None
            },
            if self.concentration.requires_grad() {
                Some(grad_k)
            } else {
                None
            },
        ])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.scale, &self.concentration]
    }

    fn name(&self) -> &'static str {
        "WeibullRsampleBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar(v: f64) -> Tensor<f64> {
        Tensor::from_storage(TensorStorage::cpu(vec![v]), vec![1], false).unwrap()
    }

    #[test]
    fn test_weibull_sample_shape() {
        let d = Weibull::new(scalar(1.0), scalar(1.5)).unwrap();
        let s = d.sample(&[100]).unwrap();
        assert_eq!(s.shape(), &[100]);
        // All samples should be non-negative.
        for &v in s.data().unwrap() {
            assert!(v >= 0.0, "Weibull sample should be >= 0, got {v}");
        }
    }

    #[test]
    fn test_weibull_log_prob_negative() {
        let d = Weibull::new(scalar(1.0), scalar(2.0)).unwrap();
        let v = Tensor::from_storage(TensorStorage::cpu(vec![-1.0]), vec![1], false).unwrap();
        let lp = d.log_prob(&v).unwrap();
        assert!(lp.data().unwrap()[0].is_infinite() && lp.data().unwrap()[0] < 0.0);
    }

    #[test]
    fn test_weibull_entropy() {
        let d = Weibull::new(scalar(1.0), scalar(1.0)).unwrap();
        let h = d.entropy().unwrap();
        // For k=1, lambda=1: H = euler*(1-1/1) + ln(1/1) + 1 = 0 + 0 + 1 = 1.0
        assert!((h.data().unwrap()[0] - 1.0).abs() < 0.01);
    }

    // ---- properties (#608) ----

    #[test]
    fn test_weibull_cdf_at_scale_is_one_minus_e_inv() {
        // F(lambda; lambda, k) = 1 - exp(-1) for any k.
        let d = Weibull::new(scalar(2.0), scalar(3.0)).unwrap();
        let v = scalar(2.0);
        let c = d.cdf(&v).unwrap();
        let expected = 1.0 - (-1.0_f64).exp();
        assert!((c.data().unwrap()[0] - expected).abs() < 1e-9);
    }

    #[test]
    fn test_weibull_cdf_icdf_roundtrip() {
        let d = Weibull::new(scalar(1.5), scalar(2.5)).unwrap();
        for p in [0.1, 0.3, 0.7, 0.9] {
            let q = scalar(p);
            let x = d.icdf(&q).unwrap();
            let p2 = d.cdf(&x).unwrap();
            assert!((p2.data().unwrap()[0] - p).abs() < 1e-6, "p={p}");
        }
    }

    #[test]
    fn test_weibull_mean_k_one_equals_lambda() {
        // For k=1, mean = lambda * Gamma(2) = lambda * 1 = lambda.
        let d = Weibull::new(scalar(3.5), scalar(1.0)).unwrap();
        let m = d.mean().unwrap();
        assert!((m.data().unwrap()[0] - 3.5).abs() < 1e-9);
    }

    #[test]
    fn test_weibull_mode_k_below_one_is_zero() {
        // For k <= 1, mode = 0.
        let d = Weibull::new(scalar(2.0), scalar(0.7)).unwrap();
        let m = d.mode().unwrap();
        assert!(m.data().unwrap()[0].abs() < 1e-12);
    }

    #[test]
    fn test_weibull_variance_k_one_equals_lambda_sq() {
        // For k=1, Var = lambda^2 * (Gamma(3) - Gamma(2)^2) = lambda^2 * (2 - 1) = lambda^2.
        let d = Weibull::new(scalar(2.0), scalar(1.0)).unwrap();
        let v = d.variance().unwrap();
        assert!((v.data().unwrap()[0] - 4.0).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // Full PyTorch surface (#1436)
    // -----------------------------------------------------------------------

    #[test]
    fn test_weibull_surface_overrides() {
        let d = Weibull::new(scalar(1.0), scalar(2.0)).unwrap();
        assert!(d.has_rsample()); // closes #1435
        assert_eq!(d.support().unwrap().name(), "Positive");
        let args = d.arg_constraints();
        assert_eq!(args["scale"].name(), "Positive");
        assert_eq!(args["concentration"].name(), "Positive");
    }

    // ---- #1435 rsample reparameterized ----

    #[test]
    fn test_weibull_rsample_shape_and_nonnegative() {
        let d = Weibull::new(scalar(1.0), scalar(2.0)).unwrap();
        let r = d.rsample(&[100]).unwrap();
        assert_eq!(r.shape(), &[100]);
        for &v in r.data().unwrap() {
            assert!(v >= 0.0, "rsample must be >= 0, got {v}");
        }
    }

    #[test]
    fn test_weibull_rsample_has_grad() {
        let s = scalar(1.5).requires_grad_(true);
        let k = scalar(2.5).requires_grad_(true);
        let d = Weibull::new(s, k).unwrap();
        let r = d.rsample(&[5]).unwrap();
        assert!(r.requires_grad());
        assert!(r.grad_fn().is_some());
    }

    #[test]
    fn test_weibull_rsample_backward_finite() {
        let s = scalar(1.5).requires_grad_(true);
        let k = scalar(2.5).requires_grad_(true);
        let d = Weibull::new(s.clone(), k.clone()).unwrap();
        let r = d.rsample(&[10]).unwrap();
        let loss = r.sum_all().unwrap();
        loss.backward().unwrap();
        let gs = s.grad().unwrap().unwrap();
        let gk = k.grad().unwrap().unwrap();
        assert!(gs.item().unwrap().is_finite());
        assert!(gk.item().unwrap().is_finite());
    }

    #[test]
    fn test_weibull_expand() {
        let d = Weibull::new(scalar(1.0), scalar(2.0)).unwrap();
        let exp = d.expand(&[3]).unwrap();
        let m = exp.mean().unwrap();
        assert_eq!(m.shape(), &[3]);
    }
}
