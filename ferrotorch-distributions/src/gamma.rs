//! Gamma distribution.
//!
//! `Gamma(concentration, rate)` defines a Gamma distribution with shape
//! parameter `concentration` (alpha) and rate parameter `rate` (beta).
//! Supports reparameterized sampling via Marsaglia & Tsang's method.
//!
//! [CL-329]
//!
//! ## REQ status (per `.design/ferrotorch-distributions/gamma.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`Gamma<T>` struct) | SHIPPED | `pub struct Gamma<T: Float>` with `concentration`/`rate` mirroring `torch/distributions/gamma.py:18-68`; consumer: `pub use gamma::Gamma` in `lib.rs` + `crate::Gamma::new(...)` invoked 4 times by the `Beta` sampling paths in `beta.rs` |
//! | REQ-2 (constructor) | SHIPPED | `pub fn Gamma::new` with shape-equality check; consumer: `crate::Gamma::new` callsites in `beta.rs` for `Beta::sample`/`rsample` |
//! | REQ-3 (accessors) | SHIPPED | `pub fn Gamma::concentration`/`rate`; consumer: `kl_gamma_gamma`/`kl_gamma_exponential`/`kl_exponential_gamma` in `kl.rs` read both accessors |
//! | REQ-4 (`Distribution` trait impl) | SHIPPED | `impl<T: Float> Distribution<T> for Gamma<T>`; consumer: `Beta::sample` and `Beta::rsample` in `beta.rs` invoke `gamma_a.sample(shape)?` and `gamma_a.rsample(shape)?` |
//! | REQ-5 (`sample_standard_gamma` private helper) | SHIPPED | Marsaglia-Tsang + α<1 boost + lazy RNG buffers; consumer: invoked by both `Gamma::sample` and `Gamma::rsample` |
//! | REQ-6 (`sample`) | SHIPPED | standard-Gamma / rate; consumer: `Beta::sample` invokes it |
//! | REQ-7 (`rsample` with backward) | SHIPPED | tiny-guard + `GammaRsampleBackward` attachment per `gamma.py:84-86`; consumer: `Beta::rsample` invokes it |
//! | REQ-8 (`log_prob`) | SHIPPED | `α*ln(β) + (α-1)*ln(x) - β*x - lgamma(α)` mirroring `gamma.py:89-98`; consumer: trait surface |
//! | REQ-9 (`entropy`) | SHIPPED | closed form mirroring `gamma.py:100-106`; consumer: trait surface |
//! | REQ-10 (`mean`/`mode`/`variance`) | SHIPPED | overrides mirroring `gamma.py:45-55` (mode NaN for α<1, R-DEV-6 vs PyTorch's clamp); consumer: trait surface |
//! | REQ-11 (`GammaRsampleBackward`) | SHIPPED | implicit-reparam through standard-Gamma; consumer: invoked by the rsample method when grad enabled |
//! | REQ-12 (full PyTorch surface — `expand`/`arg_constraints`/`support`/`has_rsample`/`cdf`) | SHIPPED | trait overrides at the tail of `impl Distribution for Gamma` in `gamma.rs` mirror `torch/distributions/gamma.py:18-119`; `cdf` = regularized lower incomplete gamma `P(conc, rate*x)` via `lower_incomplete_gamma_regularized` (Numerical-Recipes `gammp`: series for `x<s+1`, Lentz continued fraction for `x≥s+1`) mirroring `gamma.py:116-119 torch.special.gammainc`, verified against `scipy.special.gammainc` to 1e-12; consumer: `pub use gamma::Gamma` in `lib.rs` + `tests/divergence_distribution_trait_surface.rs::gamma_*` (closes #1416, #1397 — `validate_args` + exp-family `_natural_params`/`_log_normalizer` remain orthogonal trackers). |

use std::sync::Arc;

use ferrotorch_core::creation;
use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::{GradFn, Tensor};

use crate::constraints;
use crate::special_fns::{digamma_scalar, lgamma_scalar};
use crate::{DistConstraint, Distribution};
use std::collections::HashMap;

/// Gamma distribution parameterized by `concentration` (shape, alpha) and
/// `rate` (inverse scale, beta).
///
/// # Reparameterization
///
/// `rsample` uses Marsaglia & Tsang's method to draw standard gamma samples,
/// then scales by `1/rate`. Gradients flow through the implicit
/// reparameterization.
pub struct Gamma<T: Float> {
    concentration: Tensor<T>,
    rate: Tensor<T>,
}

impl<T: Float> Gamma<T> {
    /// Create a new Gamma distribution.
    ///
    /// # Errors
    ///
    /// Returns an error if `concentration` and `rate` have incompatible shapes.
    pub fn new(concentration: Tensor<T>, rate: Tensor<T>) -> FerrotorchResult<Self> {
        if concentration.shape() != rate.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "Gamma: concentration shape {:?} != rate shape {:?}",
                    concentration.shape(),
                    rate.shape()
                ),
            });
        }
        Ok(Self {
            concentration,
            rate,
        })
    }

    /// The shape (concentration) parameter.
    pub fn concentration(&self) -> &Tensor<T> {
        &self.concentration
    }

    /// The rate parameter.
    pub fn rate(&self) -> &Tensor<T> {
        &self.rate
    }
}

/// Sample `n` values from standard Gamma distributions using Marsaglia & Tsang's
/// method. For alpha >= 1, uses the direct method. For alpha < 1, uses the boost:
/// if X ~ Gamma(alpha+1, 1), then X * U^(1/alpha) ~ Gamma(alpha, 1).
///
/// Uses the framework's built-in xorshift RNG via `creation::rand` and
/// `creation::randn`.
fn sample_standard_gamma<T: Float>(alphas: &[T], n: usize) -> FerrotorchResult<Vec<T>> {
    let one = <T as num_traits::One>::one();
    let zero = <T as num_traits::Zero>::zero();
    let third = T::from(1.0 / 3.0).unwrap();

    // We need normal and uniform samples. Draw a generous batch up front
    // and pull from it. If we run out (rejection loop), draw more.
    let mut result = Vec::with_capacity(n);
    let batch = n.max(256);

    let mut norm_buf: Vec<T> = creation::randn::<T>(&[batch])?.data_vec()?;
    let mut unif_buf: Vec<T> = creation::rand::<T>(&[batch])?.data_vec()?;
    let mut ni = 0usize;
    let mut ui = 0usize;

    let next_normal = |ni: &mut usize, norm_buf: &mut Vec<T>| -> FerrotorchResult<T> {
        if *ni >= norm_buf.len() {
            *norm_buf = creation::randn::<T>(&[batch])?.data_vec()?;
            *ni = 0;
        }
        let val = norm_buf[*ni];
        *ni += 1;
        Ok(val)
    };

    let next_uniform = |ui: &mut usize, unif_buf: &mut Vec<T>| -> FerrotorchResult<T> {
        if *ui >= unif_buf.len() {
            *unif_buf = creation::rand::<T>(&[batch])?.data_vec()?;
            *ui = 0;
        }
        let val = unif_buf[*ui];
        *ui += 1;
        Ok(val)
    };

    for i in 0..n {
        let alpha = alphas[i % alphas.len()];

        // For alpha < 1, boost: sample Gamma(alpha+1) and scale
        let (effective_alpha, needs_boost) = if alpha < one {
            (alpha + one, true)
        } else {
            (alpha, false)
        };

        let d = effective_alpha - third;
        let c = third / d.sqrt();

        // Marsaglia & Tsang rejection loop
        let sample = loop {
            let x = next_normal(&mut ni, &mut norm_buf)?;
            let v_base = one + c * x;
            if v_base <= zero {
                continue;
            }
            let v = v_base * v_base * v_base;
            let u = next_uniform(&mut ui, &mut unif_buf)?;

            let x2 = x * x;
            // Squeeze test
            if u < one - T::from(0.0331).unwrap() * x2 * x2 {
                break d * v;
            }
            if u.ln() < T::from(0.5).unwrap() * x2 + d * (one - v + v.ln()) {
                break d * v;
            }
        };

        let final_sample = if needs_boost {
            let u = next_uniform(&mut ui, &mut unif_buf)?;
            // Clamp u away from 0 to avoid 0^(1/alpha) issues
            let u_safe = u.max(T::from(1e-30).unwrap());
            sample * u_safe.powf(one / alpha)
        } else {
            sample
        };

        result.push(final_sample);
    }

    Ok(result)
}

/// Regularized lower incomplete gamma `P(s, x) = γ(s, x) / Γ(s)`, computed
/// in `f64` for accuracy then cast back to `T`. This is the Numerical-Recipes
/// `gammp` routine and the scalar core of `torch.special.gammainc`, which is
/// what `torch/distributions/gamma.py:119 Gamma.cdf` calls
/// (`gammainc(concentration, rate * value)`).
///
/// Two complementary expansions cover the argument plane:
///   * `x < s + 1`: the power-series `γ*(s,x)·x^s·e^-x` form `e^{-x} x^s /
///     Γ(s) · Σ_{n≥0} x^n / (s(s+1)…(s+n))` (`gser`),
///   * `x ≥ s + 1`: the Lentz continued fraction for `Q(s,x) = 1 − P(s,x)`
///     (`gcf`).
///
/// Boundary conventions match scipy/torch: `x ≤ 0 → 0`, `s ≤ 0 → NaN`,
/// `x = 0 → 0`, and `P → 1` as `x → ∞`.
fn lower_incomplete_gamma_regularized<T: Float>(s: T, x: T) -> T {
    let s = <T as num_traits::ToPrimitive>::to_f64(&s).unwrap_or(f64::NAN);
    let x = <T as num_traits::ToPrimitive>::to_f64(&x).unwrap_or(f64::NAN);
    let result = gammp_f64(s, x);
    T::from(result).unwrap_or_else(|| T::from(f64::NAN).unwrap())
}

/// f64 core of [`lower_incomplete_gamma_regularized`]. `ln_gamma` is
/// `f64::ln_gamma`-equivalent via the crate's `lgamma_scalar` (Lanczos).
fn gammp_f64(s: f64, x: f64) -> f64 {
    if x.is_nan() || s.is_nan() {
        return f64::NAN;
    }
    if s <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    let gln = lgamma_scalar(s);
    if x < s + 1.0 {
        // Series expansion for P(s, x).
        let mut ap = s;
        let mut sum = 1.0 / s;
        let mut del = sum;
        // ~200 iterations is ample for f64 convergence over the series region.
        for _ in 0..300 {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * 1e-15 {
                break;
            }
        }
        sum * (-x + s * x.ln() - gln).exp()
    } else {
        // Lentz's continued fraction for Q(s, x) = 1 - P(s, x).
        let tiny = 1e-300;
        let mut b = x + 1.0 - s;
        let mut c = 1.0 / tiny;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..300 {
            let an = -(i as f64) * (i as f64 - s);
            b += 2.0;
            d = an * d + b;
            if d.abs() < tiny {
                d = tiny;
            }
            c = b + an / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-15 {
                break;
            }
        }
        let q = (-x + s * x.ln() - gln).exp() * h;
        1.0 - q
    }
}

impl<T: Float> Distribution<T> for Gamma<T> {
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.concentration, &self.rate],
            "Gamma::sample",
        )?;
        let device = self.concentration.device();
        let conc_data = self.concentration.data_vec()?;
        let rate_data = self.rate.data_vec()?;
        let n: usize = shape.iter().product();

        let gamma_samples = sample_standard_gamma(&conc_data, n)?;
        let result: Vec<T> = gamma_samples
            .iter()
            .zip(rate_data.iter().cycle())
            .map(|(&g, &r)| g / r)
            .collect();

        let out = Tensor::from_storage(TensorStorage::cpu(result), shape.to_vec(), false)?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }

    fn rsample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.concentration, &self.rate],
            "Gamma::rsample",
        )?;
        let device = self.concentration.device();
        let conc_data = self.concentration.data_vec()?;
        let rate_data = self.rate.data_vec()?;
        let n: usize = shape.iter().product();

        let gamma_samples = sample_standard_gamma(&conc_data, n)?;
        let tiny = T::from(1e-30).unwrap();
        let result: Vec<T> = gamma_samples
            .iter()
            .zip(rate_data.iter().cycle())
            .map(|(&g, &r)| {
                let val = g / r;
                if val < tiny { tiny } else { val }
            })
            .collect();

        let storage = TensorStorage::cpu(result);

        let out = if (self.concentration.requires_grad() || self.rate.requires_grad())
            && ferrotorch_core::is_grad_enabled()
        {
            let standard_gamma =
                Tensor::from_storage(TensorStorage::cpu(gamma_samples), shape.to_vec(), false)?;
            let grad_fn = Arc::new(GammaRsampleBackward {
                concentration: self.concentration.clone(),
                rate: self.rate.clone(),
                standard_gamma,
            });
            Tensor::from_operation(storage, shape.to_vec(), grad_fn)?
        } else {
            Tensor::from_storage(storage, shape.to_vec(), false)?
        };
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }

    fn log_prob(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.concentration, &self.rate, value],
            "Gamma::log_prob",
        )?;
        // log_prob = concentration * log(rate) + (concentration - 1) * log(x)
        //          - rate * x - lgamma(concentration)
        let device = self.concentration.device();
        let conc_data = self.concentration.data_vec()?;
        let rate_data = self.rate.data_vec()?;
        let val_data = value.data_vec()?;
        let one = <T as num_traits::One>::one();

        let result: Vec<T> = val_data
            .iter()
            .zip(conc_data.iter().cycle())
            .zip(rate_data.iter().cycle())
            .map(|((&x, &alpha), &beta)| {
                alpha * beta.ln() + (alpha - one) * x.ln() - beta * x - lgamma_scalar(alpha)
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
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.concentration, &self.rate],
            "Gamma::entropy",
        )?;
        // entropy = concentration - log(rate) + lgamma(concentration)
        //         + (1 - concentration) * digamma(concentration)
        let device = self.concentration.device();
        let conc_data = self.concentration.data_vec()?;
        let rate_data = self.rate.data_vec()?;
        let one = <T as num_traits::One>::one();

        let result: Vec<T> = conc_data
            .iter()
            .zip(rate_data.iter())
            .map(|(&alpha, &beta)| {
                alpha - beta.ln() + lgamma_scalar(alpha) + (one - alpha) * digamma_scalar(alpha)
            })
            .collect();

        let out = Tensor::from_storage(
            TensorStorage::cpu(result),
            self.concentration.shape().to_vec(),
            false,
        )?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }

    fn mean(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.concentration, &self.rate],
            "Gamma::mean",
        )?;
        // mean = concentration / rate
        let conc = self.concentration.data_vec()?;
        let rate = self.rate.data_vec()?;
        let result: Vec<T> = conc.iter().zip(rate.iter()).map(|(&a, &b)| a / b).collect();
        Tensor::from_storage(
            TensorStorage::cpu(result),
            self.concentration.shape().to_vec(),
            false,
        )
    }

    fn mode(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.concentration, &self.rate],
            "Gamma::mode",
        )?;
        // mode = (concentration - 1) / rate when concentration >= 1, else NaN.
        let conc = self.concentration.data_vec()?;
        let rate = self.rate.data_vec()?;
        let one = <T as num_traits::One>::one();
        let nan = T::from(f64::NAN).unwrap();
        let result: Vec<T> = conc
            .iter()
            .zip(rate.iter())
            .map(|(&a, &b)| if a >= one { (a - one) / b } else { nan })
            .collect();
        Tensor::from_storage(
            TensorStorage::cpu(result),
            self.concentration.shape().to_vec(),
            false,
        )
    }

    fn variance(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.concentration, &self.rate],
            "Gamma::variance",
        )?;
        // var = concentration / rate^2
        let conc = self.concentration.data_vec()?;
        let rate = self.rate.data_vec()?;
        let result: Vec<T> = conc
            .iter()
            .zip(rate.iter())
            .map(|(&a, &b)| a / (b * b))
            .collect();
        Tensor::from_storage(
            TensorStorage::cpu(result),
            self.concentration.shape().to_vec(),
            false,
        )
    }

    // -----------------------------------------------------------------------
    // Full PyTorch surface (#1376, #1416) — Gamma is reparameterizable
    // (Marsaglia-Tsang) with positive support and both parameters
    // strictly positive. `cdf` requires the regularized lower incomplete
    // gamma function (not yet implemented — kept NOT-STARTED per the
    // dispatch goal). Mirrors `torch/distributions/gamma.py:18-44`.
    // -----------------------------------------------------------------------

    fn has_rsample(&self) -> bool {
        // `torch/distributions/gamma.py:35`: `has_rsample = True`.
        true
    }

    fn support(&self) -> Option<Box<dyn DistConstraint>> {
        // `torch/distributions/gamma.py:34`: `support = constraints.nonnegative`.
        Some(Box::new(constraints::NonNegative))
    }

    fn arg_constraints(&self) -> HashMap<&'static str, Box<dyn DistConstraint>> {
        // `torch/distributions/gamma.py:30-33`:
        //   arg_constraints = {"concentration": positive, "rate": positive}
        let mut m: HashMap<&'static str, Box<dyn DistConstraint>> = HashMap::new();
        m.insert("concentration", Box::new(constraints::Positive));
        m.insert("rate", Box::new(constraints::Positive));
        m
    }

    fn event_shape(&self) -> Vec<usize> {
        vec![]
    }

    fn cdf(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // `torch/distributions/gamma.py:116-119`:
        //   return torch.special.gammainc(self.concentration, self.rate * value)
        // i.e. the regularized lower incomplete gamma `P(conc, rate * x)`.
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.concentration, &self.rate, value],
            "Gamma::cdf",
        )?;
        let device = self.concentration.device();
        let conc_data = self.concentration.data_vec()?;
        let rate_data = self.rate.data_vec()?;
        let val_data = value.data_vec()?;

        let result: Vec<T> = val_data
            .iter()
            .zip(conc_data.iter().cycle())
            .zip(rate_data.iter().cycle())
            .map(|((&x, &conc), &rate)| lower_incomplete_gamma_regularized(conc, rate * x))
            .collect();

        let out = Tensor::from_storage(TensorStorage::cpu(result), value.shape().to_vec(), false)?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }

    fn expand(&self, batch_shape: &[usize]) -> FerrotorchResult<Box<dyn Distribution<T>>> {
        // `torch/distributions/gamma.py:57-68`: broadcast both parameters.
        let conc = self.concentration.data_vec()?;
        let rate = self.rate.data_vec()?;
        let n: usize = batch_shape.iter().product::<usize>().max(1);
        let conc_out: Vec<T> = (0..n).map(|i| conc[i % conc.len()]).collect();
        let rate_out: Vec<T> = (0..n).map(|i| rate[i % rate.len()]).collect();
        let new_conc =
            Tensor::from_storage(TensorStorage::cpu(conc_out), batch_shape.to_vec(), false)?;
        let new_rate =
            Tensor::from_storage(TensorStorage::cpu(rate_out), batch_shape.to_vec(), false)?;
        Ok(Box::new(Gamma::new(new_conc, new_rate)?))
    }
}

// ---------------------------------------------------------------------------
// Backward nodes
// ---------------------------------------------------------------------------

/// Backward for Gamma rsample.
///
/// output = standard_gamma / rate
/// - d(out)/d(rate) = -standard_gamma / rate^2 = -output / rate
/// - d(out)/d(concentration): implicit reparameterization gradient
#[derive(Debug)]
struct GammaRsampleBackward<T: Float> {
    concentration: Tensor<T>,
    rate: Tensor<T>,
    standard_gamma: Tensor<T>,
}

impl<T: Float> GradFn<T> for GammaRsampleBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let device = grad_output.device();
        let go = grad_output.data_vec()?;
        let conc_data = self.concentration.data_vec()?;
        let rate_data = self.rate.data_vec()?;
        let sg_data = self.standard_gamma.data_vec()?;
        let zero = <T as num_traits::Zero>::zero();

        // grad_rate = sum(grad_output * (-standard_gamma / rate^2))
        let grad_rate_val: T = go
            .iter()
            .zip(sg_data.iter())
            .zip(rate_data.iter().cycle())
            .fold(zero, |acc, ((&g, &sg), &r)| acc + g * (-sg / (r * r)));
        let grad_rate = Tensor::from_storage(
            TensorStorage::cpu(vec![grad_rate_val]),
            self.rate.shape().to_vec(),
            false,
        )?;
        let grad_rate = if device.is_cuda() {
            grad_rate.to(device)?
        } else {
            grad_rate
        };

        // grad_concentration: implicit reparameterization gradient
        // d(sample)/d(alpha) ~= sample * (log(sample) - digamma(alpha))
        let grad_conc_val: T = go
            .iter()
            .zip(sg_data.iter())
            .zip(conc_data.iter().cycle())
            .zip(rate_data.iter().cycle())
            .fold(zero, |acc, (((&g, &sg), &alpha), &r)| {
                let tiny = T::from(1e-30).unwrap();
                let sg_safe = if sg < tiny { tiny } else { sg };
                let dsample_dalpha = sg_safe * (sg_safe.ln() - digamma_scalar(alpha));
                acc + g * dsample_dalpha / r
            });
        let grad_conc = Tensor::from_storage(
            TensorStorage::cpu(vec![grad_conc_val]),
            self.concentration.shape().to_vec(),
            false,
        )?;
        let grad_conc = if device.is_cuda() {
            grad_conc.to(device)?
        } else {
            grad_conc
        };

        Ok(vec![
            if self.concentration.requires_grad() {
                Some(grad_conc)
            } else {
                None
            },
            if self.rate.requires_grad() {
                Some(grad_rate)
            } else {
                None
            },
        ])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.concentration, &self.rate]
    }

    fn name(&self) -> &'static str {
        "GammaRsampleBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::creation::scalar;

    #[test]
    fn test_gamma_sample_shape() {
        let alpha = scalar(2.0f32).unwrap();
        let beta = scalar(1.0f32).unwrap();
        let dist = Gamma::new(alpha, beta).unwrap();

        let samples = dist.sample(&[100]).unwrap();
        assert_eq!(samples.shape(), &[100]);
        assert!(!samples.requires_grad());
    }

    #[test]
    fn test_gamma_sample_positive() {
        let alpha = scalar(2.0f32).unwrap();
        let beta = scalar(1.0f32).unwrap();
        let dist = Gamma::new(alpha, beta).unwrap();

        let samples = dist.sample(&[1000]).unwrap();
        let data = samples.data().unwrap();
        for &x in data {
            assert!(x > 0.0, "Gamma sample should be positive, got {x}");
        }
    }

    #[test]
    fn test_gamma_sample_mean() {
        // E[X] = alpha / beta = 3.0 / 2.0 = 1.5
        let alpha = scalar(3.0f32).unwrap();
        let beta = scalar(2.0f32).unwrap();
        let dist = Gamma::new(alpha, beta).unwrap();

        let samples = dist.sample(&[10000]).unwrap();
        let data = samples.data().unwrap();
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        assert!((mean - 1.5).abs() < 0.15, "expected mean ~1.5, got {mean}");
    }

    #[test]
    fn test_gamma_sample_small_alpha() {
        // alpha < 1 uses the boost method
        let alpha = scalar(0.5f32).unwrap();
        let beta = scalar(1.0f32).unwrap();
        let dist = Gamma::new(alpha, beta).unwrap();

        let samples = dist.sample(&[500]).unwrap();
        let data = samples.data().unwrap();
        for &x in data {
            assert!(x > 0.0, "Gamma sample should be positive, got {x}");
        }
    }

    #[test]
    fn test_gamma_rsample_has_grad() {
        let alpha = scalar(2.0f32).unwrap().requires_grad_(true);
        let beta = scalar(1.0f32).unwrap().requires_grad_(true);
        let dist = Gamma::new(alpha, beta).unwrap();

        let samples = dist.rsample(&[5]).unwrap();
        assert_eq!(samples.shape(), &[5]);
        assert!(samples.requires_grad());
        assert!(samples.grad_fn().is_some());
    }

    #[test]
    fn test_gamma_log_prob() {
        // Gamma(1, 1) = Exponential(1): log_prob(x) = -x
        let alpha = scalar(1.0f32).unwrap();
        let beta = scalar(1.0f32).unwrap();
        let dist = Gamma::new(alpha, beta).unwrap();

        let x = scalar(2.0f32).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = -2.0f32;
        assert!(
            (lp.item().unwrap() - expected).abs() < 1e-4,
            "expected {expected}, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_gamma_log_prob_alpha2() {
        // Gamma(2, 1): log_prob(1) = log(1) - 1 = -1
        let alpha = scalar(2.0f32).unwrap();
        let beta = scalar(1.0f32).unwrap();
        let dist = Gamma::new(alpha, beta).unwrap();

        let x = scalar(1.0f32).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = -1.0f32;
        assert!(
            (lp.item().unwrap() - expected).abs() < 1e-4,
            "expected {expected}, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_gamma_entropy() {
        // Gamma(1, 1) entropy = 1 - log(1) + lgamma(1) + 0 = 1
        let alpha = scalar(1.0f32).unwrap();
        let beta = scalar(1.0f32).unwrap();
        let dist = Gamma::new(alpha, beta).unwrap();

        let h = dist.entropy().unwrap();
        assert!(
            (h.item().unwrap() - 1.0).abs() < 1e-4,
            "expected 1.0, got {}",
            h.item().unwrap()
        );
    }

    #[test]
    fn test_gamma_shape_mismatch() {
        let alpha = scalar(1.0f32).unwrap();
        let beta = ferrotorch_core::creation::from_slice(&[1.0f32, 2.0], &[2]).unwrap();
        assert!(Gamma::new(alpha, beta).is_err());
    }

    #[test]
    fn test_gamma_f64() {
        let alpha = scalar(2.0f64).unwrap();
        let beta = scalar(1.0f64).unwrap();
        let dist = Gamma::new(alpha, beta).unwrap();

        let samples = dist.sample(&[50]).unwrap();
        assert_eq!(samples.shape(), &[50]);

        let x = scalar(1.0f64).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        assert!((lp.item().unwrap() - (-1.0f64)).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // mean / mode / variance (#585)
    // -----------------------------------------------------------------------

    #[test]
    fn test_gamma_mean_variance() {
        // Gamma(concentration=4, rate=2) → mean=2, var=1
        let dist = Gamma::new(scalar(4.0f64).unwrap(), scalar(2.0f64).unwrap()).unwrap();
        assert!((dist.mean().unwrap().item().unwrap() - 2.0).abs() < 1e-10);
        assert!((dist.variance().unwrap().item().unwrap() - 1.0).abs() < 1e-10);
        // mode = (4 - 1) / 2 = 1.5
        assert!((dist.mode().unwrap().item().unwrap() - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_gamma_mode_nan_for_concentration_below_one() {
        let dist = Gamma::new(scalar(0.5f64).unwrap(), scalar(2.0f64).unwrap()).unwrap();
        assert!(dist.mode().unwrap().item().unwrap().is_nan());
    }

    // -----------------------------------------------------------------------
    // cdf via regularized lower incomplete gamma (#1397)
    // Reference values from scipy.special.gammainc (verified at build time):
    //   gammainc(2.0, 1.0)  = 0.2642411176571153
    //   gammainc(3.0, 2.0)  = 0.32332358381693654
    //   gammainc(0.5, 0.5)  = 0.6826894921370859
    //   gammainc(5.0, 3.0)  = 0.18473675547622787
    // -----------------------------------------------------------------------

    #[test]
    fn test_gamma_cdf_series_region_matches_scipy() {
        // Gamma(conc=2, rate=1).cdf(1) = gammainc(2, 1*1) = gammainc(2,1).
        let dist = Gamma::new(scalar(2.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
        let c = dist.cdf(&scalar(1.0f64).unwrap()).unwrap();
        assert!(
            (c.item().unwrap() - 0.264_241_117_657_115_3).abs() < 1e-12,
            "got {}",
            c.item().unwrap()
        );
    }

    #[test]
    fn test_gamma_cdf_continued_fraction_region_matches_scipy() {
        // Gamma(conc=5, rate=1).cdf(3) = gammainc(5, 3): x >= s+1 -> Lentz CF.
        let dist = Gamma::new(scalar(5.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
        let c = dist.cdf(&scalar(3.0f64).unwrap()).unwrap();
        assert!(
            (c.item().unwrap() - 0.184_736_755_476_227_87).abs() < 1e-12,
            "got {}",
            c.item().unwrap()
        );
    }

    #[test]
    fn test_gamma_cdf_rate_scales_argument() {
        // Gamma(conc=3, rate=2).cdf(1) = gammainc(3, 2*1) = gammainc(3,2).
        let dist = Gamma::new(scalar(3.0f64).unwrap(), scalar(2.0f64).unwrap()).unwrap();
        let c = dist.cdf(&scalar(1.0f64).unwrap()).unwrap();
        assert!(
            (c.item().unwrap() - 0.323_323_583_816_936_54).abs() < 1e-12,
            "got {}",
            c.item().unwrap()
        );
    }

    #[test]
    fn test_gamma_cdf_subunit_concentration_matches_scipy() {
        // Gamma(conc=0.5, rate=1).cdf(0.5) = gammainc(0.5, 0.5).
        let dist = Gamma::new(scalar(0.5f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
        let c = dist.cdf(&scalar(0.5f64).unwrap()).unwrap();
        assert!(
            (c.item().unwrap() - 0.682_689_492_137_085_9).abs() < 1e-12,
            "got {}",
            c.item().unwrap()
        );
    }

    #[test]
    fn test_gamma_cdf_exponential_closed_form() {
        // Gamma(1, rate) == Exponential(rate): cdf(x) = 1 - exp(-rate*x), exact.
        let dist = Gamma::new(scalar(1.0f64).unwrap(), scalar(2.0f64).unwrap()).unwrap();
        let c = dist.cdf(&scalar(1.5f64).unwrap()).unwrap();
        let expected = 1.0 - (-2.0f64 * 1.5).exp();
        assert!(
            (c.item().unwrap() - expected).abs() < 1e-12,
            "expected {expected}, got {}",
            c.item().unwrap()
        );
    }

    #[test]
    fn test_gamma_cdf_boundaries() {
        let dist = Gamma::new(scalar(2.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
        // x = 0 -> P = 0; x -> large -> P -> 1.
        assert_eq!(
            dist.cdf(&scalar(0.0f64).unwrap()).unwrap().item().unwrap(),
            0.0
        );
        assert!(
            dist.cdf(&scalar(0.0f64).unwrap())
                .unwrap()
                .item()
                .unwrap()
                .abs()
                < 1e-15
        );
        assert!(
            (dist
                .cdf(&scalar(100.0f64).unwrap())
                .unwrap()
                .item()
                .unwrap()
                - 1.0)
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn test_gamma_cdf_f32() {
        let dist = Gamma::new(scalar(2.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
        let c = dist.cdf(&scalar(1.0f32).unwrap()).unwrap();
        assert!(
            (c.item().unwrap() - 0.264_241_12).abs() < 1e-5,
            "got {}",
            c.item().unwrap()
        );
    }
}
