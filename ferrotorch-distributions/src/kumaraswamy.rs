//! Kumaraswamy distribution.
//!
//! `Kumaraswamy(a, b)` — a two-parameter distribution on [0, 1], similar to
//! Beta but with a closed-form CDF and simpler sampling.
//!
//! ## REQ status (per `.design/ferrotorch-distributions/kumaraswamy.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`Kumaraswamy<T>` struct) | SHIPPED | `pub struct Kumaraswamy` in `kumaraswamy.rs`; re-exported via `lib.rs`; mirrors `torch/distributions/kumaraswamy.py:24-70`. |
//! | REQ-2 (`new` constructor with shape-mismatch rejection) | SHIPPED | the constructor in `kumaraswamy.rs`. |
//! | REQ-3 (`a` + `b` accessors) | SHIPPED | the accessors in `kumaraswamy.rs`. |
//! | REQ-4 (`Distribution<T>` impl with closed-form methods) | SHIPPED | the impl block in `kumaraswamy.rs`; mirrors `kumaraswamy.py:78-107`. |
//! | REQ-5 (inverse-CDF sampling) | SHIPPED | the sample body in `kumaraswamy.rs` invoking `(1 - (1-u)^(1/b))^(1/a)`. |
//! | REQ-6 (rsample) | SHIPPED | the `rsample` body in `kumaraswamy.rs` builds the inverse-CDF transform `(1 - (1-u)^(1/b))^(1/a)` with a `KumaraswamyRsampleBackward` autograd node so gradients flow through `a` and `b`. Mirrors upstream `has_rsample = True` (`kumaraswamy.py:48`). Consumer: trait dispatch via `pub use Kumaraswamy` re-export at `lib.rs:107`. Closes #1383. |
//! | REQ-7 (mode boundary returns NaN) | SHIPPED | the `mode` body returns `NaN` when `a < 1` or `b < 1` mirroring `kumaraswamy.py:89` (`log_mode[(concentration0 < 1) | (concentration1 < 1)] = nan`). Pinned by `test_kumaraswamy_mode_boundary_is_nan`. Closes #1384. |
//! | REQ-8 (`digamma` via shifted-asymptotic expansion) | SHIPPED | the entropy body invokes `digamma_scalar(b + 1)` from `special_fns`. |

use std::sync::Arc;

use ferrotorch_core::creation;
use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::{GradFn, Tensor};

use crate::Distribution;
use crate::special_fns::{digamma_scalar, lgamma_scalar};

/// Kumaraswamy distribution parameterized by concentration parameters `a` > 0
/// and `b` > 0.
///
/// PDF: `f(x) = a * b * x^(a-1) * (1 - x^a)^(b-1)` for `x in [0, 1]`.
///
/// CDF: `F(x) = 1 - (1 - x^a)^b`.
///
/// Sampling via inverse CDF: `x = (1 - (1-u)^(1/b))^(1/a)`.
pub struct Kumaraswamy<T: Float> {
    a: Tensor<T>,
    b: Tensor<T>,
}

impl<T: Float> Kumaraswamy<T> {
    pub fn new(a: Tensor<T>, b: Tensor<T>) -> FerrotorchResult<Self> {
        if a.shape() != b.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "Kumaraswamy: a shape {:?} != b shape {:?}",
                    a.shape(),
                    b.shape()
                ),
            });
        }
        Ok(Self { a, b })
    }

    pub fn a(&self) -> &Tensor<T> {
        &self.a
    }
    pub fn b(&self) -> &Tensor<T> {
        &self.b
    }
}

impl<T: Float> Distribution<T> for Kumaraswamy<T> {
    #[allow(clippy::needless_range_loop)]
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.a, &self.b], "Kumaraswamy::sample")?;
        let u = creation::rand::<T>(shape)?;
        let u_data = u.data()?;
        let a_data = self.a.data()?;
        let b_data = self.b.data()?;
        let numel = u_data.len();
        let one = <T as num_traits::One>::one();

        let mut out = Vec::with_capacity(numel);
        for i in 0..numel {
            let ai = if a_data.len() == 1 {
                0
            } else {
                i % a_data.len()
            };
            let bi = if b_data.len() == 1 {
                0
            } else {
                i % b_data.len()
            };
            // x = (1 - (1-u)^(1/b))^(1/a)
            let inner = (one - u_data[i]).powf(one / b_data[bi]);
            let val = (one - inner)
                .max(T::from(1e-30).unwrap())
                .powf(one / a_data[ai]);
            out.push(val);
        }

        Tensor::from_storage(TensorStorage::cpu(out), shape.to_vec(), false)
    }

    #[allow(clippy::needless_range_loop)]
    fn rsample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        // Inverse-CDF reparameterization: x = (1 - (1-u)^(1/b))^(1/a),
        // u ~ Uniform(0, 1). Mirrors `kumaraswamy.py:48,64-68` (the
        // `TransformedDistribution(Uniform, [PowerTransform(1/b),
        // AffineTransform(1,-1), PowerTransform(1/a)])` chain with
        // `has_rsample = True`).
        //
        // ferrotorch's grad-aware `pow(tensor, f64)` doesn't accept a
        // tensor exponent (the exponent here is `1/a` / `1/b`, both
        // tensors), so the forward is computed scalar-wise and the
        // backward is custom: see `KumaraswamyRsampleBackward` below.
        crate::fallback::check_gpu_fallback_opt_in(&[&self.a, &self.b], "Kumaraswamy::rsample")?;
        let u = creation::rand::<T>(shape)?;
        let u_data = u.data_vec()?;
        let a_data = self.a.data_vec()?;
        let b_data = self.b.data_vec()?;
        let numel = u_data.len();
        let one = <T as num_traits::One>::one();
        let eps_clamp = T::from(1e-30).unwrap();

        let mut out = Vec::with_capacity(numel);
        for i in 0..numel {
            let ai = if a_data.len() == 1 {
                0
            } else {
                i % a_data.len()
            };
            let bi = if b_data.len() == 1 {
                0
            } else {
                i % b_data.len()
            };
            // x = (1 - (1-u)^(1/b))^(1/a)
            let inner = (one - u_data[i]).max(eps_clamp).powf(one / b_data[bi]);
            let val = (one - inner).max(eps_clamp).powf(one / a_data[ai]);
            out.push(val);
        }
        let storage = TensorStorage::cpu(out);

        if (self.a.requires_grad() || self.b.requires_grad()) && ferrotorch_core::is_grad_enabled()
        {
            let grad_fn = Arc::new(KumaraswamyRsampleBackward {
                a: self.a.clone(),
                b: self.b.clone(),
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
            &[&self.a, &self.b, value],
            "Kumaraswamy::log_prob",
        )?;
        let v = value.data()?;
        let a = self.a.data()?;
        let b = self.b.data()?;
        let numel = v.len();
        let one = <T as num_traits::One>::one();
        let zero = <T as num_traits::Zero>::zero();

        let mut out = Vec::with_capacity(numel);
        for i in 0..numel {
            let ai = if a.len() == 1 { 0 } else { i % a.len() };
            let bi = if b.len() == 1 { 0 } else { i % b.len() };
            if v[i] <= zero || v[i] >= one {
                out.push(T::neg_infinity());
            } else {
                // log_prob = log(a) + log(b) + (a-1)*log(x) + (b-1)*log(1 - x^a)
                let lp = a[ai].ln()
                    + b[bi].ln()
                    + (a[ai] - one) * v[i].ln()
                    + (b[bi] - one) * (one - v[i].powf(a[ai])).ln();
                out.push(lp);
            }
        }

        Tensor::from_storage(TensorStorage::cpu(out), value.shape().to_vec(), false)
    }

    fn entropy(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.a, &self.b], "Kumaraswamy::entropy")?;
        // Derived from E[log f(X)] for Kumaraswamy(a, b):
        //   E[log X]       = (-euler_gamma - digamma(b+1)) / a
        //   E[log(1-X^a)]  = -1/b   (since X^a ~ Power(b), exact)
        //
        // H = -log(a) - log(b) - (a-1)*E[log X] - (b-1)*E[log(1-X^a)]
        //   = -log(a) - log(b) + (1 - 1/a)*(euler_gamma + digamma(b+1)) + (1 - 1/b)
        //
        // digamma(b+1) via shift-then-asymptotic expansion — valid for all b > 0
        // (integer or fractional). The previous integer-step loop diverged for
        // non-integer b in (0,1) because it stepped past the target argument.
        let a = self.a.data()?;
        let b = self.b.data()?;
        let one = <T as num_traits::One>::one();
        let euler = T::from(0.5772156649015329).unwrap();

        let mut out = Vec::with_capacity(a.len());
        for i in 0..a.len() {
            // digamma(b+1) via canonical shifted-asymptotic expansion in
            // special_fns. Valid for all b > 0 (integer or fractional).
            let digamma_b1 = digamma_scalar(b[i] + one);
            let h = (one - one / a[i]) * (euler + digamma_b1) + (one - one / b[i])
                - a[i].ln()
                - b[i].ln();
            out.push(h);
        }

        Tensor::from_storage(TensorStorage::cpu(out), self.a.shape().to_vec(), false)
    }

    fn cdf(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.a, &self.b, value], "Kumaraswamy::cdf")?;
        // F(x) = 1 - (1 - x^a)^b for x in [0, 1].
        let v = value.data()?;
        let a = self.a.data()?;
        let b = self.b.data()?;
        let zero = <T as num_traits::Zero>::zero();
        let one = <T as num_traits::One>::one();
        let mut out = Vec::with_capacity(v.len());
        for (i, &x) in v.iter().enumerate() {
            let ai = if a.len() == 1 { 0 } else { i % a.len() };
            let bi = if b.len() == 1 { 0 } else { i % b.len() };
            if x <= zero {
                out.push(zero);
            } else if x >= one {
                out.push(one);
            } else {
                out.push(one - (one - x.powf(a[ai])).powf(b[bi]));
            }
        }
        Tensor::from_storage(TensorStorage::cpu(out), value.shape().to_vec(), false)
    }

    fn icdf(&self, q: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.a, &self.b, q], "Kumaraswamy::icdf")?;
        // F^{-1}(p) = (1 - (1 - p)^(1/b))^(1/a)
        let p = q.data()?;
        let a = self.a.data()?;
        let b = self.b.data()?;
        let one = <T as num_traits::One>::one();
        let mut out = Vec::with_capacity(p.len());
        for (i, &pi) in p.iter().enumerate() {
            let ai = if a.len() == 1 { 0 } else { i % a.len() };
            let bi = if b.len() == 1 { 0 } else { i % b.len() };
            let inner = (one - pi).powf(one / b[bi]);
            out.push((one - inner).max(T::from(1e-30).unwrap()).powf(one / a[ai]));
        }
        Tensor::from_storage(TensorStorage::cpu(out), q.shape().to_vec(), false)
    }

    fn mean(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.a, &self.b], "Kumaraswamy::mean")?;
        // E[X] = b * Beta(1 + 1/a, b)
        // = b * Gamma(1+1/a) * Gamma(b) / Gamma(1+1/a+b)
        // = exp( ln(b) + lgamma(1+1/a) + lgamma(b) - lgamma(1+1/a+b) )
        let a = self.a.data()?;
        let b = self.b.data()?;
        let one = <T as num_traits::One>::one();
        let mut out = Vec::with_capacity(a.len());
        for i in 0..a.len() {
            let alpha = one + one / a[i];
            let lg = lgamma_scalar(alpha) + lgamma_scalar(b[i]) - lgamma_scalar(alpha + b[i]);
            out.push(b[i] * lg.exp());
        }
        Tensor::from_storage(TensorStorage::cpu(out), self.a.shape().to_vec(), false)
    }

    fn mode(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.a, &self.b], "Kumaraswamy::mode")?;
        // `torch/distributions/kumaraswamy.py:82-90`:
        //   log_mode = (1/b) * log1p(-b) - log1p(-a*b)
        //   log_mode[(b < 1) | (a < 1)] = nan
        //   return log_mode.exp()
        // ferrotorch keeps the algebraic form `((a-1)/(a*b - 1))^(1/a)`
        // for the well-defined branch and returns NaN otherwise to
        // mirror upstream's NaN-at-boundary contract (closes #1384).
        let a = self.a.data()?;
        let b = self.b.data()?;
        let one = <T as num_traits::One>::one();
        let zero = <T as num_traits::Zero>::zero();
        let nan = T::from(f64::NAN).unwrap();
        let mut out = Vec::with_capacity(a.len());
        for i in 0..a.len() {
            if a[i] < one || b[i] < one {
                // Upstream NaN-mask: kumaraswamy.py:89.
                out.push(nan);
            } else if a[i] > one && b[i] >= one {
                let denom = a[i] * b[i] - one;
                if denom > zero {
                    out.push(((a[i] - one) / denom).powf(one / a[i]));
                } else {
                    out.push(nan);
                }
            } else {
                // a == 1, b >= 1: mode is on the boundary (constant
                // density at the lower edge for a==1); upstream's NaN
                // mask only fires for a<1 or b<1, so a==1 is well-defined
                // — the analytic limit is `b^(-1/(b-1))` for b>1 and 0
                // for b==1. Use upstream's log-space form which collapses
                // gracefully: log_mode = (1/b)*log1p(-b) - log1p(-a*b).
                // For a==1, b==1: log1p(-1) = -inf, -inf*1 - log1p(-1) = NaN.
                // For a==1, b>1: numerator (1/b)*ln(1-b) is complex (1-b<0)
                // so upstream actually produces NaN here too via the log.
                // Defensive: return NaN to match upstream's
                // domain-of-validity contract.
                out.push(nan);
            }
        }
        Tensor::from_storage(TensorStorage::cpu(out), self.a.shape().to_vec(), false)
    }

    fn variance(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.a, &self.b], "Kumaraswamy::variance")?;
        // m1 = b * Beta(1 + 1/a, b)
        // m2 = b * Beta(1 + 2/a, b)
        // Var = m2 - m1^2
        let a = self.a.data()?;
        let b = self.b.data()?;
        let one = <T as num_traits::One>::one();
        let two = T::from(2.0).unwrap();
        let mut out = Vec::with_capacity(a.len());
        for i in 0..a.len() {
            let m1_log = lgamma_scalar(one + one / a[i]) + lgamma_scalar(b[i])
                - lgamma_scalar(one + one / a[i] + b[i]);
            let m1 = b[i] * m1_log.exp();
            let m2_log = lgamma_scalar(one + two / a[i]) + lgamma_scalar(b[i])
                - lgamma_scalar(one + two / a[i] + b[i]);
            let m2 = b[i] * m2_log.exp();
            out.push(m2 - m1 * m1);
        }
        Tensor::from_storage(TensorStorage::cpu(out), self.a.shape().to_vec(), false)
    }
}

// ---------------------------------------------------------------------------
// Backward node for rsample
// ---------------------------------------------------------------------------

/// Backward for `x = (1 - (1-u)^(1/b))^(1/a)` with stored uniform `u`.
///
/// Let `inner = (1-u)^(1/b)` and `x = (1 - inner)^(1/a)`. Then:
///
/// `dx/da = -x * ln(1 - inner) / a^2`
/// `dx/db = -(x^(1-a)) * (-(1-u)^(1/b) * ln(1-u) / b^2) / a`
///        = `x^(1-a) * (1-u)^(1/b) * ln(1-u) / (a * b^2)`
///
/// Equivalently, in log-form: `log(x) = (1/a) * log(1 - (1-u)^(1/b))`, so
/// `d(log x)/da = -(1/a^2) * log(1 - (1-u)^(1/b))` giving
/// `dx/da = x * d(log x)/da = -x * log(1 - inner) / a^2`. The b-derivative
/// is derived similarly via chain rule on `inner = (1-u)^(1/b)`.
#[derive(Debug)]
struct KumaraswamyRsampleBackward<T: Float> {
    a: Tensor<T>,
    b: Tensor<T>,
    u: Tensor<T>,
}

impl<T: Float> GradFn<T> for KumaraswamyRsampleBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let go = grad_output.data_vec()?;
        let u_data = self.u.data_vec()?;
        let a_data = self.a.data_vec()?;
        let b_data = self.b.data_vec()?;
        let one = <T as num_traits::One>::one();
        let zero = <T as num_traits::Zero>::zero();
        let eps_clamp = T::from(1e-30).unwrap();

        let mut grad_a_acc = zero;
        let mut grad_b_acc = zero;

        for (i, (&g, &u_val)) in go.iter().zip(u_data.iter()).enumerate() {
            let ai = if a_data.len() == 1 {
                0
            } else {
                i % a_data.len()
            };
            let bi = if b_data.len() == 1 {
                0
            } else {
                i % b_data.len()
            };
            let a = a_data[ai];
            let b = b_data[bi];
            let one_minus_u = (one - u_val).max(eps_clamp);
            // inner = (1-u)^(1/b); log_inner = (1/b) * ln(1-u)
            let log_one_minus_u = one_minus_u.ln();
            let inner = one_minus_u.powf(one / b);
            let one_minus_inner = (one - inner).max(eps_clamp);
            let log_omi = one_minus_inner.ln();
            // x = (1 - inner)^(1/a); log(x) = (1/a) * log(1 - inner)
            let x = one_minus_inner.powf(one / a);

            // d(log x)/da = -(1/a^2) * log(1 - inner)
            // dx/da = x * d(log x)/da
            let dx_da = x * (-(log_omi) / (a * a));

            // d inner/db = inner * d(log_inner)/db
            //            = inner * (-(1/b^2)) * ln(1-u)
            // d(log x)/db = (1/a) * d(log(1 - inner))/db
            //            = (1/a) * (-d inner/db)/(1 - inner)
            //            = (1/a) * (inner * (1/b^2) * ln(1-u)) / (1 - inner)
            // dx/db = x * d(log x)/db
            let dx_db = x * (inner * log_one_minus_u) / (a * b * b * one_minus_inner);
            // The sign: d inner/db has -(1/b^2)*ln(1-u); ln(1-u) is negative
            // for u in (0,1), so d inner/db = positive (since the two negs
            // cancel ... wait, -(1/b^2) * ln(1-u) where ln(1-u) < 0 yields
            // a positive d inner/db). And `dx/db = -(1/a)*d inner/db / (1-inner) * x`.
            // Reapply with explicit sign: d inner/db = -inner*ln(1-u)/b^2 (positive).
            // d(1-inner)/db = -d inner/db = inner*ln(1-u)/b^2 (negative).
            // d log(1-inner)/db = (1/(1-inner)) * inner*ln(1-u)/b^2 (negative).
            // d log x /db = (1/a) * inner * ln(1-u) / (b^2 * (1-inner)) (negative).
            // dx/db = x * d log x / db. ln(1-u) < 0, so dx/db < 0 — correct
            // (larger b shrinks `inner`, but increasing `b` makes inner larger
            // since (1-u)^(1/b) → 1 as b → inf, which then drives `(1-inner)`
            // smaller and x smaller). Numerically the dx_db computed above
            // already carries the correct sign because log_one_minus_u is
            // negative.

            grad_a_acc += g * dx_da;
            grad_b_acc += g * dx_db;
        }

        let grad_a = Tensor::from_storage(
            TensorStorage::cpu(vec![grad_a_acc]),
            self.a.shape().to_vec(),
            false,
        )?;
        let grad_b = Tensor::from_storage(
            TensorStorage::cpu(vec![grad_b_acc]),
            self.b.shape().to_vec(),
            false,
        )?;

        Ok(vec![
            if self.a.requires_grad() {
                Some(grad_a)
            } else {
                None
            },
            if self.b.requires_grad() {
                Some(grad_b)
            } else {
                None
            },
        ])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.a, &self.b]
    }

    fn name(&self) -> &'static str {
        "KumaraswamyRsampleBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar(v: f64) -> Tensor<f64> {
        Tensor::from_storage(TensorStorage::cpu(vec![v]), vec![1], false).unwrap()
    }

    #[test]
    fn test_kumaraswamy_sample_range() {
        let d = Kumaraswamy::new(scalar(2.0), scalar(5.0)).unwrap();
        let s = d.sample(&[500]).unwrap();
        for &v in s.data().unwrap() {
            assert!(
                v > 0.0 && v < 1.0,
                "Kumaraswamy sample should be in (0,1), got {v}"
            );
        }
    }

    #[test]
    fn test_kumaraswamy_log_prob_boundary() {
        let d = Kumaraswamy::new(scalar(1.0), scalar(1.0)).unwrap();
        let at_zero = Tensor::from_storage(TensorStorage::cpu(vec![0.0]), vec![1], false).unwrap();
        let at_one = Tensor::from_storage(TensorStorage::cpu(vec![1.0]), vec![1], false).unwrap();
        assert!(d.log_prob(&at_zero).unwrap().data().unwrap()[0].is_infinite());
        assert!(d.log_prob(&at_one).unwrap().data().unwrap()[0].is_infinite());
    }

    #[test]
    fn test_kumaraswamy_uniform_case() {
        // a=1, b=1 should be uniform — log_prob should be 0 everywhere in (0,1)
        let d = Kumaraswamy::new(scalar(1.0), scalar(1.0)).unwrap();
        let v = Tensor::from_storage(TensorStorage::cpu(vec![0.5]), vec![1], false).unwrap();
        let lp = d.log_prob(&v).unwrap().data().unwrap()[0];
        assert!(
            (lp - 0.0).abs() < 1e-6,
            "a=1,b=1 should be uniform, log_prob={lp}"
        );
    }

    // ---- properties (#608) ----

    #[test]
    fn test_kumaraswamy_cdf_unit_case_is_identity() {
        // a=1, b=1 -> Uniform(0,1), so F(x) = x.
        let d = Kumaraswamy::new(scalar(1.0), scalar(1.0)).unwrap();
        for x in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let v = Tensor::from_storage(TensorStorage::cpu(vec![x]), vec![1], false).unwrap();
            let c = d.cdf(&v).unwrap().data().unwrap()[0];
            assert!((c - x).abs() < 1e-9, "x={x}, cdf={c}");
        }
    }

    #[test]
    fn test_kumaraswamy_cdf_icdf_roundtrip() {
        let d = Kumaraswamy::new(scalar(2.0), scalar(3.0)).unwrap();
        for p in [0.1, 0.3, 0.7, 0.9] {
            let q = Tensor::from_storage(TensorStorage::cpu(vec![p]), vec![1], false).unwrap();
            let x = d.icdf(&q).unwrap();
            let p2 = d.cdf(&x).unwrap().data().unwrap()[0];
            assert!((p2 - p).abs() < 1e-6, "p={p}, recovered={p2}");
        }
    }

    #[test]
    fn test_kumaraswamy_mean_uniform_case_is_half() {
        // For a=1, b=1, mean should be 0.5.
        let d = Kumaraswamy::new(scalar(1.0), scalar(1.0)).unwrap();
        let m = d.mean().unwrap().data().unwrap()[0];
        assert!((m - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_kumaraswamy_variance_uniform_case_is_one_twelfth() {
        // For a=1, b=1, variance should be 1/12 ≈ 0.0833.
        let d = Kumaraswamy::new(scalar(1.0), scalar(1.0)).unwrap();
        let v = d.variance().unwrap().data().unwrap()[0];
        assert!((v - 1.0 / 12.0).abs() < 1e-6);
    }

    #[test]
    fn test_kumaraswamy_mode_well_defined_when_a_gt_one() {
        // a=2, b=2: mode = ((2-1)/(4-1))^(1/2) = sqrt(1/3) ≈ 0.5774
        let d = Kumaraswamy::new(scalar(2.0), scalar(2.0)).unwrap();
        let m = d.mode().unwrap().data().unwrap()[0];
        assert!((m - (1.0_f64 / 3.0).sqrt()).abs() < 1e-9);
    }

    // ---- #1384 mode boundary returns NaN ----

    #[test]
    fn test_kumaraswamy_mode_boundary_is_nan() {
        // a < 1 or b < 1: upstream `kumaraswamy.py:89` sets log_mode = NaN
        // via boolean-mask. Pinned to match.
        for (a, b) in [(0.5, 2.0), (2.0, 0.5), (0.5, 0.5)] {
            let d = Kumaraswamy::new(scalar(a), scalar(b)).unwrap();
            let m = d.mode().unwrap().data().unwrap()[0];
            assert!(m.is_nan(), "a={a}, b={b}: expected NaN, got {m}");
        }
    }

    // ---- #1383 rsample reparameterized ----

    #[test]
    fn test_kumaraswamy_rsample_shape_and_range() {
        let d = Kumaraswamy::new(scalar(2.0), scalar(5.0)).unwrap();
        let s = d.rsample(&[200]).unwrap();
        assert_eq!(s.shape(), &[200]);
        for &v in s.data().unwrap() {
            assert!(v > 0.0 && v < 1.0, "rsample must be in (0,1), got {v}");
        }
    }

    #[test]
    fn test_kumaraswamy_rsample_requires_grad_when_params_grad() {
        let a = scalar(2.0).requires_grad_(true);
        let b = scalar(3.0).requires_grad_(true);
        let d = Kumaraswamy::new(a, b).unwrap();
        let s = d.rsample(&[10]).unwrap();
        assert!(s.requires_grad());
        assert!(s.grad_fn().is_some());
    }

    #[test]
    fn test_kumaraswamy_rsample_backward_finite() {
        let a = scalar(2.0).requires_grad_(true);
        let b = scalar(3.0).requires_grad_(true);
        let d = Kumaraswamy::new(a.clone(), b.clone()).unwrap();
        let s = d.rsample(&[10]).unwrap();
        let loss = s.sum_all().unwrap();
        loss.backward().unwrap();
        let ga = a.grad().unwrap().unwrap();
        let gb = b.grad().unwrap().unwrap();
        assert!(ga.item().unwrap().is_finite());
        assert!(gb.item().unwrap().is_finite());
    }
}
