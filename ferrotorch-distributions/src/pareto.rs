//! Pareto (Type I) distribution.
//!
//! `Pareto(scale, alpha)` — a heavy-tailed power-law distribution.
//!
//! ## REQ status (per `.design/ferrotorch-distributions/pareto.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`Pareto` struct) | SHIPPED | `pub struct Pareto` in `pareto.rs`; re-exported as `pub use pareto::Pareto` in `lib.rs:116`; mirrors `torch/distributions/pareto.py:33-43`. |
//! | REQ-2 (`new` constructor, shape match) | SHIPPED | `Pareto::new` rejecting shape mismatch in `pareto.rs`; registered in `tests/conformance/_surface_inventory.toml:483`. |
//! | REQ-3 (`scale` + `alpha` accessors) | SHIPPED | `Pareto::scale` + `Pareto::alpha` in `pareto.rs`. |
//! | REQ-4 (`Distribution::sample` via inverse-CDF) | SHIPPED | `impl Distribution::sample` in `pareto.rs` uses `scale / u^(1/alpha)`; mirrors `pareto.py:39-43` `TransformedDistribution` composition. |
//! | REQ-5 (`Distribution::log_prob`) | SHIPPED | `impl Distribution::log_prob` in `pareto.rs` returns the closed-form Pareto log density; pinned by `test_pareto_log_prob_below_scale` + `test_pareto_log_prob_at_scale`. |
//! | REQ-6 (`Distribution::mean`) | SHIPPED | `impl Distribution::mean` in `pareto.rs` returns `alpha*scale/(alpha-1)` if `alpha>1` else `inf`; mirrors `pareto.py:53-57`. |
//! | REQ-7 (`Distribution::variance`) | SHIPPED | `impl Distribution::variance` in `pareto.rs` branches on `alpha>2`; mirrors `pareto.py:63-67`. |
//! | REQ-8 (`Distribution::entropy`) | SHIPPED | `impl Distribution::entropy` in `pareto.rs` returns `log(scale/alpha)+1+1/alpha`; mirrors `pareto.py:73-74`. |
//! | REQ-9 (`rsample` reparameterization) | SHIPPED | the `rsample` body in `pareto.rs` composes `scale / u^(1/alpha)` with a `ParetoRsampleBackward` autograd node so gradients flow through `scale` and `alpha`. Mirrors upstream `TransformedDistribution(Exponential, [ExpTransform, AffineTransform(scale)])` with `has_rsample = True` (`pareto.py:33-43`). Closes #1395. |
//! | REQ-10 (`mode`/`support`/`expand`/`cdf`/`icdf`/`arg_constraints`) | SHIPPED | the trait overrides at the tail of `impl Distribution for Pareto` in `pareto.rs` mirror `torch/distributions/pareto.py:31, 69-71`; consumer: trait dispatch via `pub use Pareto` re-export (closes #1405; `rsample` remains under #1395). |

use std::collections::HashMap;
use std::sync::Arc;

use ferrotorch_core::creation;
use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::{GradFn, Tensor};

use crate::constraints;
use crate::{DistConstraint, Distribution};

/// Pareto Type I distribution parameterized by `scale` (x_m, minimum value)
/// and `alpha` (shape/tail index).
///
/// PDF: `f(x) = alpha * scale^alpha / x^(alpha+1)` for `x >= scale`.
///
/// Sampling: `x = scale / u^(1/alpha)` where `u ~ Uniform(0,1)`.
pub struct Pareto<T: Float> {
    scale: Tensor<T>,
    alpha: Tensor<T>,
}

impl<T: Float> Pareto<T> {
    pub fn new(scale: Tensor<T>, alpha: Tensor<T>) -> FerrotorchResult<Self> {
        if scale.shape() != alpha.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "Pareto: scale shape {:?} != alpha shape {:?}",
                    scale.shape(),
                    alpha.shape()
                ),
            });
        }
        Ok(Self { scale, alpha })
    }

    pub fn scale(&self) -> &Tensor<T> {
        &self.scale
    }
    pub fn alpha(&self) -> &Tensor<T> {
        &self.alpha
    }
}

impl<T: Float> Distribution<T> for Pareto<T> {
    #[allow(clippy::needless_range_loop)]
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.scale, &self.alpha], "Pareto::sample")?;
        let u = creation::rand::<T>(shape)?;
        let u_data = u.data()?;
        let s_data = self.scale.data()?;
        let a_data = self.alpha.data()?;
        let numel = u_data.len();
        let one = <T as num_traits::One>::one();

        let mut out = Vec::with_capacity(numel);
        for i in 0..numel {
            let si = if s_data.len() == 1 {
                0
            } else {
                i % s_data.len()
            };
            let ai = if a_data.len() == 1 {
                0
            } else {
                i % a_data.len()
            };
            // x = scale / u^(1/alpha)
            let val = s_data[si]
                / u_data[i]
                    .max(T::from(1e-30).unwrap())
                    .powf(one / a_data[ai]);
            out.push(val);
        }

        Tensor::from_storage(TensorStorage::cpu(out), shape.to_vec(), false)
    }

    #[allow(clippy::needless_range_loop)]
    fn rsample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        // Inverse-CDF reparameterization: x = scale * (1 - u)^(-1/alpha)
        // (equivalent to `scale / u^(1/alpha)` since u and (1-u) are
        // both Uniform(0,1)). Mirrors `TransformedDistribution(Exponential,
        // [ExpTransform, AffineTransform(scale=scale)])` chain at
        // `torch/distributions/pareto.py:33-43`. The forward is computed
        // scalar-wise; the backward (`ParetoRsampleBackward`) carries the
        // gradients through `scale` and `alpha`.
        crate::fallback::check_gpu_fallback_opt_in(&[&self.scale, &self.alpha], "Pareto::rsample")?;
        let u = creation::rand::<T>(shape)?;
        let u_data = u.data_vec()?;
        let s_data = self.scale.data_vec()?;
        let a_data = self.alpha.data_vec()?;
        let numel = u_data.len();
        let one = <T as num_traits::One>::one();
        let eps = T::from(1e-30).unwrap();

        let mut out = Vec::with_capacity(numel);
        for i in 0..numel {
            let si = if s_data.len() == 1 {
                0
            } else {
                i % s_data.len()
            };
            let ai = if a_data.len() == 1 {
                0
            } else {
                i % a_data.len()
            };
            // x = scale / u^(1/alpha)
            let val = s_data[si] / u_data[i].max(eps).powf(one / a_data[ai]);
            out.push(val);
        }
        let storage = TensorStorage::cpu(out);

        if (self.scale.requires_grad() || self.alpha.requires_grad())
            && ferrotorch_core::is_grad_enabled()
        {
            let grad_fn = Arc::new(ParetoRsampleBackward {
                scale: self.scale.clone(),
                alpha: self.alpha.clone(),
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
            &[&self.scale, &self.alpha, value],
            "Pareto::log_prob",
        )?;
        let v = value.data()?;
        let s = self.scale.data()?;
        let a = self.alpha.data()?;
        let numel = v.len();
        let one = <T as num_traits::One>::one();

        let mut out = Vec::with_capacity(numel);
        for i in 0..numel {
            let si = if s.len() == 1 { 0 } else { i % s.len() };
            let ai = if a.len() == 1 { 0 } else { i % a.len() };
            if v[i] < s[si] {
                out.push(T::neg_infinity());
            } else {
                // log_prob = log(alpha) + alpha*log(scale) - (alpha+1)*log(x)
                let lp = a[ai].ln() + a[ai] * s[si].ln() - (a[ai] + one) * v[i].ln();
                out.push(lp);
            }
        }

        Tensor::from_storage(TensorStorage::cpu(out), value.shape().to_vec(), false)
    }

    fn mean(&self) -> FerrotorchResult<Tensor<T>> {
        // Reference: torch.distributions.Pareto.mean
        // mean = alpha * scale / (alpha - 1)  when alpha > 1, else inf
        crate::fallback::check_gpu_fallback_opt_in(&[&self.scale, &self.alpha], "Pareto::mean")?;
        let s = self.scale.data()?;
        let a = self.alpha.data()?;
        let one = <T as num_traits::One>::one();
        let mut out = Vec::with_capacity(s.len());
        for i in 0..s.len() {
            if a[i] > one {
                out.push(a[i] * s[i] / (a[i] - one));
            } else {
                out.push(T::infinity());
            }
        }
        Tensor::from_storage(TensorStorage::cpu(out), self.scale.shape().to_vec(), false)
    }

    fn variance(&self) -> FerrotorchResult<Tensor<T>> {
        // Reference: torch.distributions.Pareto.variance
        // variance = scale^2 * alpha / ((alpha-1)^2 * (alpha-2))  when alpha > 2, else inf
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.scale, &self.alpha],
            "Pareto::variance",
        )?;
        let s = self.scale.data()?;
        let a = self.alpha.data()?;
        let one = <T as num_traits::One>::one();
        let two = T::from(2.0).unwrap();
        let mut out = Vec::with_capacity(s.len());
        for i in 0..s.len() {
            if a[i] > two {
                let am1 = a[i] - one;
                out.push(s[i] * s[i] * a[i] / (am1 * am1 * (a[i] - two)));
            } else {
                out.push(T::infinity());
            }
        }
        Tensor::from_storage(TensorStorage::cpu(out), self.scale.shape().to_vec(), false)
    }

    fn entropy(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.scale, &self.alpha], "Pareto::entropy")?;
        // H = log(scale/alpha) + 1 + 1/alpha
        let s = self.scale.data()?;
        let a = self.alpha.data()?;
        let one = <T as num_traits::One>::one();

        let mut out = Vec::with_capacity(s.len());
        for i in 0..s.len() {
            out.push((s[i] / a[i]).ln() + one + one / a[i]);
        }

        Tensor::from_storage(TensorStorage::cpu(out), self.scale.shape().to_vec(), false)
    }

    fn mode(&self) -> FerrotorchResult<Tensor<T>> {
        // `torch/distributions/pareto.py:60-61`: `mode = scale`.
        Ok(self.scale.clone())
    }

    #[allow(clippy::needless_range_loop)]
    fn cdf(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.scale, &self.alpha, value],
            "Pareto::cdf",
        )?;
        // F(x; scale, alpha) = 1 - (scale/x)^alpha  for x >= scale, else 0.
        let v = value.data()?;
        let s = self.scale.data()?;
        let a = self.alpha.data()?;
        let one = <T as num_traits::One>::one();
        let zero = <T as num_traits::Zero>::zero();
        let mut out = Vec::with_capacity(v.len());
        for i in 0..v.len() {
            let si = if s.len() == 1 { 0 } else { i % s.len() };
            let ai = if a.len() == 1 { 0 } else { i % a.len() };
            if v[i] < s[si] {
                out.push(zero);
            } else {
                out.push(one - (s[si] / v[i]).powf(a[ai]));
            }
        }
        Tensor::from_storage(TensorStorage::cpu(out), value.shape().to_vec(), false)
    }

    #[allow(clippy::needless_range_loop)]
    fn icdf(&self, q: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.scale, &self.alpha, q], "Pareto::icdf")?;
        // F^{-1}(p) = scale / (1 - p)^(1/alpha)
        let p = q.data()?;
        let s = self.scale.data()?;
        let a = self.alpha.data()?;
        let one = <T as num_traits::One>::one();
        let eps = T::from(1e-30).unwrap();
        let mut out = Vec::with_capacity(p.len());
        for i in 0..p.len() {
            let si = if s.len() == 1 { 0 } else { i % s.len() };
            let ai = if a.len() == 1 { 0 } else { i % a.len() };
            let base = (one - p[i]).max(eps);
            out.push(s[si] / base.powf(one / a[ai]));
        }
        Tensor::from_storage(TensorStorage::cpu(out), q.shape().to_vec(), false)
    }

    // -----------------------------------------------------------------------
    // Full PyTorch surface — `pareto.py:31, 69-71` declares
    //   arg_constraints = {"alpha": positive, "scale": positive}
    //   support = greater_than_eq(scale) (parameter-dependent)
    // -----------------------------------------------------------------------

    fn has_rsample(&self) -> bool {
        // `torch/distributions/pareto.py` inherits TransformedDistribution
        // (Exponential + ExpTransform + AffineTransform) with `has_rsample = True`.
        // ferrotorch's direct path builds the autograd graph via
        // `ParetoRsampleBackward` (closes #1395).
        true
    }

    fn batch_shape(&self) -> Vec<usize> {
        self.scale.shape().to_vec()
    }

    fn support(&self) -> Option<Box<dyn DistConstraint>> {
        // `torch/distributions/pareto.py:69-71`:
        //   support = constraints.greater_than_eq(self.scale)
        // ferrotorch's `DistConstraint` is dtype-erased; expose the scalar
        // minimum of `scale` as a `GreaterThanEq<f64>` representative for
        // batched / scalar cases. The dyn-safe surface drops per-element
        // bounds — full per-element support tracking is part of the
        // PositiveDefinite/composite roll-out under #1372.
        let s = self.scale.data().ok()?;
        let lo = s
            .iter()
            .map(|x| x.to_f64().unwrap_or(0.0))
            .fold(f64::INFINITY, f64::min);
        Some(Box::new(constraints::GreaterThanEq::<f64> {
            lower_bound: lo,
        }))
    }

    fn arg_constraints(&self) -> HashMap<&'static str, Box<dyn DistConstraint>> {
        // `torch/distributions/pareto.py:31`:
        //   arg_constraints = {"alpha": positive, "scale": positive}
        let mut m: HashMap<&'static str, Box<dyn DistConstraint>> = HashMap::new();
        m.insert("alpha", Box::new(constraints::Positive));
        m.insert("scale", Box::new(constraints::Positive));
        m
    }

    fn expand(&self, batch_shape: &[usize]) -> FerrotorchResult<Box<dyn Distribution<T>>> {
        // Mirrors `pareto.py:43-50` (TransformedDistribution.expand).
        let s_data = self.scale.data_vec()?;
        let a_data = self.alpha.data_vec()?;
        let n: usize = batch_shape.iter().product::<usize>().max(1);
        let s_out: Vec<T> = (0..n).map(|i| s_data[i % s_data.len()]).collect();
        let a_out: Vec<T> = (0..n).map(|i| a_data[i % a_data.len()]).collect();
        let new_scale =
            Tensor::from_storage(TensorStorage::cpu(s_out), batch_shape.to_vec(), false)?;
        let new_alpha =
            Tensor::from_storage(TensorStorage::cpu(a_out), batch_shape.to_vec(), false)?;
        Ok(Box::new(Pareto::new(new_scale, new_alpha)?))
    }
}

// ---------------------------------------------------------------------------
// Backward node for rsample
// ---------------------------------------------------------------------------

/// Backward for `x = scale * u^(-1/alpha)`.
///
/// Let `c = u^(-1/alpha)`, so `x = scale * c`.
/// - `dx/dscale = c = x/scale` (sum over sample dims)
/// - `log(c) = -ln(u)/alpha`, so `d log(c)/d alpha = ln(u)/alpha^2`,
///   `dc/d alpha = c * ln(u) / alpha^2`,
///   `dx/d alpha = scale * dc/d alpha = x * ln(u) / alpha^2` (sum over dims)
///
/// Note `ln(u)` is negative for `u in (0,1)`, so `dx/d alpha < 0` — matches
/// the intuition that increasing `alpha` lowers tail weight, shrinking samples.
#[derive(Debug)]
struct ParetoRsampleBackward<T: Float> {
    scale: Tensor<T>,
    alpha: Tensor<T>,
    u: Tensor<T>,
}

impl<T: Float> GradFn<T> for ParetoRsampleBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let go = grad_output.data_vec()?;
        let u_data = self.u.data_vec()?;
        let s_data = self.scale.data_vec()?;
        let a_data = self.alpha.data_vec()?;
        let one = <T as num_traits::One>::one();
        let zero = <T as num_traits::Zero>::zero();
        let eps = T::from(1e-30).unwrap();

        let mut g_scale = zero;
        let mut g_alpha = zero;
        for (i, (&g, &u_val)) in go.iter().zip(u_data.iter()).enumerate() {
            let si = if s_data.len() == 1 {
                0
            } else {
                i % s_data.len()
            };
            let ai = if a_data.len() == 1 {
                0
            } else {
                i % a_data.len()
            };
            let s = s_data[si];
            let a = a_data[ai];
            let u_safe = u_val.max(eps);
            let c = u_safe.powf(-one / a);
            let x = s * c;
            // dx/dscale = c
            g_scale += g * c;
            // dx/dalpha = x * ln(u) / alpha^2
            g_alpha += g * x * u_safe.ln() / (a * a);
        }

        let grad_scale = Tensor::from_storage(
            TensorStorage::cpu(vec![g_scale]),
            self.scale.shape().to_vec(),
            false,
        )?;
        let grad_alpha = Tensor::from_storage(
            TensorStorage::cpu(vec![g_alpha]),
            self.alpha.shape().to_vec(),
            false,
        )?;

        Ok(vec![
            if self.scale.requires_grad() {
                Some(grad_scale)
            } else {
                None
            },
            if self.alpha.requires_grad() {
                Some(grad_alpha)
            } else {
                None
            },
        ])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.scale, &self.alpha]
    }

    fn name(&self) -> &'static str {
        "ParetoRsampleBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar(v: f64) -> Tensor<f64> {
        Tensor::from_storage(TensorStorage::cpu(vec![v]), vec![1], false).unwrap()
    }

    #[test]
    fn test_pareto_samples_above_scale() {
        let d = Pareto::new(scalar(2.0), scalar(3.0)).unwrap();
        let s = d.sample(&[200]).unwrap();
        for &v in s.data().unwrap() {
            assert!(v >= 2.0, "Pareto sample should be >= scale, got {v}");
        }
    }

    #[test]
    fn test_pareto_log_prob_below_scale() {
        let d = Pareto::new(scalar(5.0), scalar(1.0)).unwrap();
        let v = Tensor::from_storage(TensorStorage::cpu(vec![3.0]), vec![1], false).unwrap();
        let lp = d.log_prob(&v).unwrap();
        assert!(lp.data().unwrap()[0].is_infinite() && lp.data().unwrap()[0] < 0.0);
    }

    #[test]
    fn test_pareto_log_prob_at_scale() {
        let d = Pareto::new(scalar(1.0), scalar(2.0)).unwrap();
        let v = Tensor::from_storage(TensorStorage::cpu(vec![1.0]), vec![1], false).unwrap();
        let lp = d.log_prob(&v).unwrap();
        // log_prob(1) = log(2) + 2*log(1) - 3*log(1) = log(2) ≈ 0.693
        assert!((lp.data().unwrap()[0] - 2.0f64.ln()).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Full PyTorch surface (#1405)
    // -----------------------------------------------------------------------

    #[test]
    fn test_pareto_mode_equals_scale() {
        let d = Pareto::new(scalar(2.5), scalar(3.0)).unwrap();
        let m = d.mode().unwrap();
        assert!((m.data().unwrap()[0] - 2.5).abs() < 1e-12);
    }

    #[test]
    fn test_pareto_cdf_at_scale_is_zero() {
        // F(scale; scale, alpha) = 1 - (scale/scale)^alpha = 0.
        let d = Pareto::new(scalar(2.0), scalar(3.0)).unwrap();
        let v = Tensor::from_storage(TensorStorage::cpu(vec![2.0]), vec![1], false).unwrap();
        let c = d.cdf(&v).unwrap();
        assert!(c.data().unwrap()[0].abs() < 1e-12);
    }

    #[test]
    fn test_pareto_cdf_icdf_roundtrip() {
        let d = Pareto::new(scalar(1.0), scalar(2.0)).unwrap();
        for p in [0.1, 0.3, 0.5, 0.7, 0.9] {
            let q = Tensor::from_storage(TensorStorage::cpu(vec![p]), vec![1], false).unwrap();
            let x = d.icdf(&q).unwrap();
            let p2 = d.cdf(&x).unwrap();
            assert!((p2.data().unwrap()[0] - p).abs() < 1e-9, "p={p}");
        }
    }

    #[test]
    fn test_pareto_surface_overrides() {
        let d = Pareto::new(scalar(1.0), scalar(2.0)).unwrap();
        assert!(d.has_rsample()); // closes #1395
        let s = d.support().unwrap();
        assert_eq!(s.name(), "GreaterThanEq");
        let args = d.arg_constraints();
        assert_eq!(args["alpha"].name(), "Positive");
        assert_eq!(args["scale"].name(), "Positive");
    }

    // ---- #1395 rsample reparameterized ----

    #[test]
    fn test_pareto_rsample_shape_and_range() {
        let d = Pareto::new(scalar(2.0), scalar(3.0)).unwrap();
        let s = d.rsample(&[100]).unwrap();
        assert_eq!(s.shape(), &[100]);
        for &v in s.data().unwrap() {
            assert!(v >= 2.0, "rsample must be >= scale, got {v}");
        }
    }

    #[test]
    fn test_pareto_rsample_has_grad() {
        let s = scalar(2.0).requires_grad_(true);
        let a = scalar(3.0).requires_grad_(true);
        let d = Pareto::new(s, a).unwrap();
        let r = d.rsample(&[5]).unwrap();
        assert!(r.requires_grad());
        assert!(r.grad_fn().is_some());
    }

    #[test]
    fn test_pareto_rsample_backward_finite() {
        let s = scalar(2.0).requires_grad_(true);
        let a = scalar(3.0).requires_grad_(true);
        let d = Pareto::new(s.clone(), a.clone()).unwrap();
        let r = d.rsample(&[10]).unwrap();
        let loss = r.sum_all().unwrap();
        loss.backward().unwrap();
        let gs = s.grad().unwrap().unwrap();
        let ga = a.grad().unwrap().unwrap();
        assert!(gs.item().unwrap().is_finite());
        assert!(ga.item().unwrap().is_finite());
    }

    #[test]
    fn test_pareto_expand() {
        let d = Pareto::new(scalar(1.0), scalar(2.0)).unwrap();
        let exp = d.expand(&[3]).unwrap();
        let m = exp.mode().unwrap();
        assert_eq!(m.shape(), &[3]);
    }
}
