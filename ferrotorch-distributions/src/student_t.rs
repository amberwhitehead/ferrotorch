//! Student's t-distribution.
//!
//! `StudentT(df, loc, scale)` defines a Student's t-distribution with `df`
//! degrees of freedom, location `loc`, and scale `scale`.
//! Supports reparameterized sampling.
//!
//! ## REQ status (per `.design/ferrotorch-distributions/student_t.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`StudentT` struct) | SHIPPED | `pub struct StudentT` in `student_t.rs`; re-exported as `pub use student_t::StudentT` in `lib.rs:120`; mirrors `torch/distributions/studentT.py:64-74`. |
//! | REQ-2 (`new` constructor, shape match) | SHIPPED | `StudentT::new` rejecting shape mismatch; registered in `tests/conformance/_surface_inventory.toml:315`. |
//! | REQ-3 (accessors `df`/`loc`/`scale`) | SHIPPED | accessors in `student_t.rs`. |
//! | REQ-4 (inherent `mean_value`/`variance_value`) | SHIPPED | inherent moment helpers in `student_t.rs` mirroring `studentT.py:42-62` with `df`-dependent branching. |
//! | REQ-5 (`Distribution::sample` via Normal/Chi2) | SHIPPED | `impl Distribution::sample` builds `loc + scale * Z * sqrt(df/Chi2)` with `sample_chi2` Marsaglia-Tsang Gamma sampler; mirrors `studentT.py:87-99`. |
//! | REQ-6 (`Distribution::rsample` differentiable through loc/scale) | SHIPPED | `impl Distribution::rsample` builds `Tensor::from_operation` with `StudentTRsampleBackward` autograd node capturing `df`/`loc`/`scale`/`z`/`chi2`; pinned by `test_student_t_rsample_{has_grad, backward}`. |
//! | REQ-7 (`Distribution::log_prob` closed-form) | SHIPPED | `impl Distribution::log_prob` returns the closed-form Student's-t log density; pinned by `test_student_t_log_prob_at_loc` (Cauchy edge), `test_student_t_log_prob_high_df_approaches_normal`. |
//! | REQ-8 (`Distribution::entropy` closed-form) | SHIPPED | `impl Distribution::entropy` uses `lgamma_scalar` + `digamma_scalar` from `special_fns.rs`; mirrors `studentT.py:114-127`. |
//! | REQ-9 (`df` gradient in backward node) | NOT-STARTED | blocker #1427 — `StudentTRsampleBackward::backward` returns `None` for the `df` slot. The correct df gradient needs the Chi2 implicit-reparameterisation gradient `d(sg)/d(alpha) = -(∂_alpha P(alpha,sg))/pdf(sg)` (PyTorch's `_standard_gamma_grad` series), which ferrotorch lacks. The repo's `GammaRsampleBackward` closed form `sg·(ln sg − digamma(alpha))` is NOT that ratio (FD/scipy-`gammaincinv`-verified discrepancy: `0.953` vs `-0.020` at `alpha=2.5,sg=2`), so shipping it would be a wrong gradient. loc/scale gradients are tractable and SHIPPED (REQ-6). #1427 stays open; the incorrect Gamma formula is a separate spillover finding. |
//! | REQ-10 (`expand`/`support`/`mode`/`mean`/`variance`/`arg_constraints`/`has_rsample`) | SHIPPED | the trait overrides at the tail of `impl Distribution for StudentT` in `student_t.rs` mirror `torch/distributions/studentT.py:34-50`; consumer: trait dispatch via `pub use StudentT` re-export (closes #1428; `cdf`/`icdf` require the regularized incomplete-beta function which is part of #1372 / not yet ported). |

use std::collections::HashMap;
use std::sync::Arc;

use ferrotorch_core::creation;
use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::{GradFn, Tensor};

use crate::constraints;
use crate::special_fns::{digamma_scalar, lgamma_scalar};
use crate::{DistConstraint, Distribution};

/// Student's t-distribution parameterized by `df` (degrees of freedom),
/// `loc` (location), and `scale`.
///
/// # Reparameterization
///
/// `rsample` uses the representation:
/// ```text
/// z ~ Normal(0, 1)
/// chi2 ~ Chi2(df)  (= Gamma(df/2, 1/2))
/// t = loc + scale * z * sqrt(df / chi2)
/// ```
/// Gradients flow through `loc` and `scale` via the autograd graph.
pub struct StudentT<T: Float> {
    df: Tensor<T>,
    loc: Tensor<T>,
    scale: Tensor<T>,
}

impl<T: Float> StudentT<T> {
    /// Create a new Student's t-distribution.
    ///
    /// # Errors
    ///
    /// Returns an error if `df`, `loc`, and `scale` have incompatible shapes.
    pub fn new(df: Tensor<T>, loc: Tensor<T>, scale: Tensor<T>) -> FerrotorchResult<Self> {
        if df.shape() != loc.shape() || df.shape() != scale.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "StudentT: df shape {:?}, loc shape {:?}, scale shape {:?} must all match",
                    df.shape(),
                    loc.shape(),
                    scale.shape()
                ),
            });
        }
        Ok(Self { df, loc, scale })
    }

    /// The degrees of freedom parameter.
    pub fn df(&self) -> &Tensor<T> {
        &self.df
    }

    /// The location parameter.
    pub fn loc(&self) -> &Tensor<T> {
        &self.loc
    }

    /// The scale parameter.
    pub fn scale(&self) -> &Tensor<T> {
        &self.scale
    }

    /// The mean of the distribution (defined for df > 1, equals loc).
    pub fn mean_value(&self) -> FerrotorchResult<Vec<T>> {
        let df_data = self.df.data_vec()?;
        let loc_data = self.loc.data_vec()?;
        let one = <T as num_traits::One>::one();
        Ok(df_data
            .iter()
            .zip(loc_data.iter())
            .map(|(&df, &loc)| if df > one { loc } else { T::nan() })
            .collect())
    }

    /// The variance of the distribution (defined for df > 2).
    /// Var[X] = scale^2 * df / (df - 2).
    pub fn variance_value(&self) -> FerrotorchResult<Vec<T>> {
        let df_data = self.df.data_vec()?;
        let scale_data = self.scale.data_vec()?;
        let two = T::from(2.0).unwrap();
        Ok(df_data
            .iter()
            .zip(scale_data.iter())
            .map(|(&df, &scale)| {
                if df > two {
                    scale * scale * df / (df - two)
                } else {
                    T::infinity()
                }
            })
            .collect())
    }
}

/// Sample from Chi-squared(df) = Gamma(df/2, 1/2) using Marsaglia & Tsang.
/// Returns `n` samples.
fn sample_chi2<T: Float>(df_values: &[T], n: usize) -> FerrotorchResult<Vec<T>> {
    let one = <T as num_traits::One>::one();
    let zero = <T as num_traits::Zero>::zero();
    let half = T::from(0.5).unwrap();
    let third = T::from(1.0 / 3.0).unwrap();

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

    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let df = df_values[i % df_values.len()];
        // Chi2(df) = Gamma(df/2, rate=1/2), scale = 2
        // So sample Gamma(df/2, 1) and multiply by 2
        let alpha = df * half;

        let (effective_alpha, needs_boost) = if alpha < one {
            (alpha + one, true)
        } else {
            (alpha, false)
        };

        let d = effective_alpha - third;
        let c = third / d.sqrt();

        let gamma_sample = loop {
            let x = next_normal(&mut ni, &mut norm_buf)?;
            let v_base = one + c * x;
            if v_base <= zero {
                continue;
            }
            let v = v_base * v_base * v_base;
            let u = next_uniform(&mut ui, &mut unif_buf)?;

            let x2 = x * x;
            if u < one - T::from(0.0331).unwrap() * x2 * x2 {
                break d * v;
            }
            if u.ln() < half * x2 + d * (one - v + v.ln()) {
                break d * v;
            }
        };

        let gamma_final = if needs_boost {
            let u = next_uniform(&mut ui, &mut unif_buf)?;
            let u_safe = u.max(T::from(1e-30).unwrap());
            gamma_sample * u_safe.powf(one / alpha)
        } else {
            gamma_sample
        };

        // Chi2 = Gamma(df/2, rate=1/2) = Gamma(df/2, 1) * 2
        let chi2_sample = gamma_final * T::from(2.0).unwrap();
        result.push(chi2_sample);
    }

    Ok(result)
}

impl<T: Float> Distribution<T> for StudentT<T> {
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.df, &self.loc, &self.scale],
            "StudentT::sample",
        )?;
        let device = self.loc.device();
        let df_data = self.df.data_vec()?;
        let loc_data = self.loc.data_vec()?;
        let scale_data = self.scale.data_vec()?;
        let n: usize = shape.iter().product();

        let z = creation::randn::<T>(shape)?;
        let z_data = z.data_vec()?;
        let chi2_samples = sample_chi2(&df_data, n)?;

        let result: Vec<T> = z_data
            .iter()
            .zip(chi2_samples.iter())
            .zip(df_data.iter().cycle())
            .zip(loc_data.iter().cycle())
            .zip(scale_data.iter().cycle())
            .map(|((((&z_val, &chi2), &df), &loc), &scale)| {
                loc + scale * z_val * (df / chi2).sqrt()
            })
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
            &[&self.df, &self.loc, &self.scale],
            "StudentT::rsample",
        )?;
        let device = self.loc.device();
        let df_data = self.df.data_vec()?;
        let loc_data = self.loc.data_vec()?;
        let scale_data = self.scale.data_vec()?;
        let n: usize = shape.iter().product();

        let z = creation::randn::<T>(shape)?;
        let z_data = z.data_vec()?;
        let chi2_samples = sample_chi2(&df_data, n)?;

        let result: Vec<T> = z_data
            .iter()
            .zip(chi2_samples.iter())
            .zip(df_data.iter().cycle())
            .zip(loc_data.iter().cycle())
            .zip(scale_data.iter().cycle())
            .map(|((((&z_val, &chi2), &df), &loc), &scale)| {
                loc + scale * z_val * (df / chi2).sqrt()
            })
            .collect();

        let storage = TensorStorage::cpu(result);

        let out = if (self.loc.requires_grad() || self.scale.requires_grad())
            && ferrotorch_core::is_grad_enabled()
        {
            let z_tensor = z.clone();
            let chi2_tensor =
                Tensor::from_storage(TensorStorage::cpu(chi2_samples), shape.to_vec(), false)?;
            let grad_fn = Arc::new(StudentTRsampleBackward {
                df: self.df.clone(),
                loc: self.loc.clone(),
                scale: self.scale.clone(),
                z: z_tensor,
                chi2: chi2_tensor,
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
            &[&self.df, &self.loc, &self.scale, value],
            "StudentT::log_prob",
        )?;
        // log_prob = lgamma((df+1)/2) - lgamma(df/2)
        //          - 0.5 * ln(df * pi) - ln(scale)
        //          - (df+1)/2 * ln(1 + ((x - loc)/scale)^2 / df)
        let device = self.loc.device();
        let df_data = self.df.data_vec()?;
        let loc_data = self.loc.data_vec()?;
        let scale_data = self.scale.data_vec()?;
        let val_data = value.data_vec()?;
        let half = T::from(0.5).unwrap();
        let one = <T as num_traits::One>::one();
        let pi = T::from(std::f64::consts::PI).unwrap();

        let result: Vec<T> = val_data
            .iter()
            .zip(df_data.iter().cycle())
            .zip(loc_data.iter().cycle())
            .zip(scale_data.iter().cycle())
            .map(|(((&x, &df), &loc), &scale)| {
                let y = (x - loc) / scale;
                lgamma_scalar((df + one) * half)
                    - lgamma_scalar(df * half)
                    - half * (df * pi).ln()
                    - scale.ln()
                    - (df + one) * half * (one + y * y / df).ln()
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
        crate::fallback::check_gpu_fallback_opt_in(&[&self.df, &self.scale], "StudentT::entropy")?;
        // entropy = (df + 1)/2 * (digamma((df+1)/2) - digamma(df/2))
        //         + ln(sqrt(df) * B(df/2, 1/2))
        //         + ln(scale)
        // where B(a, b) = Gamma(a)*Gamma(b) / Gamma(a+b)
        // Simplifying: ln(sqrt(df) * B(df/2, 1/2))
        //   = 0.5*ln(df) + lgamma(df/2) + 0.5*ln(pi) - lgamma((df+1)/2)
        //   = 0.5*ln(df) + lgamma(df/2) + 0.5*ln(pi) - lgamma((df+1)/2)
        let device = self.df.device();
        let df_data = self.df.data_vec()?;
        let scale_data = self.scale.data_vec()?;
        let half = T::from(0.5).unwrap();
        let one = <T as num_traits::One>::one();
        let pi = T::from(std::f64::consts::PI).unwrap();

        let result: Vec<T> = df_data
            .iter()
            .zip(scale_data.iter())
            .map(|(&df, &scale)| {
                let df_plus_1_half = (df + one) * half;
                let df_half = df * half;
                df_plus_1_half * (digamma_scalar(df_plus_1_half) - digamma_scalar(df_half))
                    + half * df.ln()
                    + lgamma_scalar(df_half)
                    + half * pi.ln()
                    - lgamma_scalar(df_plus_1_half)
                    + scale.ln()
            })
            .collect();

        let out =
            Tensor::from_storage(TensorStorage::cpu(result), self.df.shape().to_vec(), false)?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }

    fn mean(&self) -> FerrotorchResult<Tensor<T>> {
        // `torch/distributions/studentT.py:42-46`: mean = loc if df > 1 else nan.
        crate::fallback::check_gpu_fallback_opt_in(&[&self.df, &self.loc], "StudentT::mean")?;
        let data = self.mean_value()?;
        Tensor::from_storage(TensorStorage::cpu(data), self.loc.shape().to_vec(), false)
    }

    fn mode(&self) -> FerrotorchResult<Tensor<T>> {
        // `torch/distributions/studentT.py:48-50`: mode = loc.
        Ok(self.loc.clone())
    }

    fn variance(&self) -> FerrotorchResult<Tensor<T>> {
        // `torch/distributions/studentT.py:52-62`: scale^2 * df / (df - 2) if df > 2 else inf.
        crate::fallback::check_gpu_fallback_opt_in(&[&self.df, &self.scale], "StudentT::variance")?;
        let data = self.variance_value()?;
        Tensor::from_storage(TensorStorage::cpu(data), self.df.shape().to_vec(), false)
    }

    // -----------------------------------------------------------------------
    // Full PyTorch surface — `studentT.py:34-40` declares
    //   arg_constraints = {"df": positive, "loc": real, "scale": positive}
    //   support = constraints.real
    //   has_rsample = True
    // -----------------------------------------------------------------------

    fn has_rsample(&self) -> bool {
        // `torch/distributions/studentT.py:40`: `has_rsample = True`.
        true
    }

    fn batch_shape(&self) -> Vec<usize> {
        self.loc.shape().to_vec()
    }

    fn support(&self) -> Option<Box<dyn DistConstraint>> {
        // `torch/distributions/studentT.py:39`: `support = constraints.real`.
        Some(Box::new(constraints::Real))
    }

    fn arg_constraints(&self) -> HashMap<&'static str, Box<dyn DistConstraint>> {
        // `torch/distributions/studentT.py:34-38`:
        //   arg_constraints = {"df": positive, "loc": real, "scale": positive}
        let mut m: HashMap<&'static str, Box<dyn DistConstraint>> = HashMap::new();
        m.insert("df", Box::new(constraints::Positive));
        m.insert("loc", Box::new(constraints::Real));
        m.insert("scale", Box::new(constraints::Positive));
        m
    }

    fn expand(&self, batch_shape: &[usize]) -> FerrotorchResult<Box<dyn Distribution<T>>> {
        let df_data = self.df.data_vec()?;
        let loc_data = self.loc.data_vec()?;
        let scale_data = self.scale.data_vec()?;
        let n: usize = batch_shape.iter().product::<usize>().max(1);
        let df_out: Vec<T> = (0..n).map(|i| df_data[i % df_data.len()]).collect();
        let loc_out: Vec<T> = (0..n).map(|i| loc_data[i % loc_data.len()]).collect();
        let scale_out: Vec<T> = (0..n).map(|i| scale_data[i % scale_data.len()]).collect();
        let new_df = Tensor::from_storage(TensorStorage::cpu(df_out), batch_shape.to_vec(), false)?;
        let new_loc =
            Tensor::from_storage(TensorStorage::cpu(loc_out), batch_shape.to_vec(), false)?;
        let new_scale =
            Tensor::from_storage(TensorStorage::cpu(scale_out), batch_shape.to_vec(), false)?;
        Ok(Box::new(StudentT::new(new_df, new_loc, new_scale)?))
    }
}

// ---------------------------------------------------------------------------
// Backward nodes
// ---------------------------------------------------------------------------

/// Backward for StudentT rsample.
///
/// `output = loc + scale * z * sqrt(df / chi2)` where `chi2 = sg * 2` and
/// `sg ~ standard_gamma(df/2)` is the unscaled Gamma sample (so `sg = chi2/2`).
///
/// `d(out)/d(loc) = 1`; `d(out)/d(scale) = z*sqrt(df/chi2) = (out - loc)/scale`.
///
/// `d(out)/d(df)` is PARTIAL (returns `None`). Upstream `studentT.py:97-98`
/// flows the `df` gradient through `self._chi2.rsample()` (Chi2 = Gamma(df/2,
/// 1/2)) whose implicit-reparameterisation gradient is
/// `d(sg)/d(alpha) = -(∂_alpha P(alpha, sg)) / pdf(sg)` (PyTorch's
/// `_standard_gamma_grad` series). ferrotorch has no `_standard_gamma_grad`
/// primitive, and the repo's `GammaRsampleBackward` closed form
/// `sg·(ln sg − digamma(alpha))` does NOT equal that ratio (verified against
/// scipy `gammaincinv`: e.g. `alpha=2.5, sg=2` gives `0.953` vs the closed
/// form's `-0.020`). Shipping a wrong `df` gradient would be worse than none,
/// so the slot stays `None` until the implicit-Gamma grad primitive lands.
/// Blocker #1427 remains open for the `df` term; #1555 tracks the incorrect
/// `GammaRsampleBackward` formula.
#[derive(Debug)]
struct StudentTRsampleBackward<T: Float> {
    df: Tensor<T>,
    loc: Tensor<T>,
    scale: Tensor<T>,
    z: Tensor<T>,
    chi2: Tensor<T>,
}

impl<T: Float> GradFn<T> for StudentTRsampleBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let device = grad_output.device();
        let go = grad_output.data_vec()?;
        let z_data = self.z.data_vec()?;
        let chi2_data = self.chi2.data_vec()?;
        let df_data = self.df.data_vec()?;
        let zero = <T as num_traits::Zero>::zero();

        // grad_loc = sum(grad_output)
        let grad_loc_val: T = go.iter().copied().fold(zero, |acc, g| acc + g);
        let grad_loc = Tensor::from_storage(
            TensorStorage::cpu(vec![grad_loc_val]),
            self.loc.shape().to_vec(),
            false,
        )?;
        let grad_loc = if device.is_cuda() {
            grad_loc.to(device)?
        } else {
            grad_loc
        };

        // grad_scale = sum(grad_output * z * sqrt(df / chi2))
        let grad_scale_val: T = go
            .iter()
            .zip(z_data.iter())
            .zip(chi2_data.iter())
            .zip(df_data.iter().cycle())
            .fold(zero, |acc, (((&g, &z), &chi2), &df)| {
                acc + g * z * (df / chi2).sqrt()
            });
        let grad_scale = Tensor::from_storage(
            TensorStorage::cpu(vec![grad_scale_val]),
            self.scale.shape().to_vec(),
            false,
        )?;
        let grad_scale = if device.is_cuda() {
            grad_scale.to(device)?
        } else {
            grad_scale
        };

        // grad_df: PARTIAL — left as `None`. The correct df gradient needs the
        // implicit-Gamma reparameterisation gradient `d(sg)/d(alpha)` for the
        // Chi2 sample, which has no closed form in ferrotorch yet (PyTorch uses
        // a `_standard_gamma_grad` series). See the struct doc; blocker #1427
        // stays open for the df term.

        Ok(vec![
            None, // df gradient PARTIAL — see struct doc / #1427.
            if self.loc.requires_grad() {
                Some(grad_loc)
            } else {
                None
            },
            if self.scale.requires_grad() {
                Some(grad_scale)
            } else {
                None
            },
        ])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.df, &self.loc, &self.scale]
    }

    fn name(&self) -> &'static str {
        "StudentTRsampleBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::creation::{from_slice, scalar};

    #[test]
    fn test_student_t_sample_shape() {
        let df = scalar(5.0f32).unwrap();
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let dist = StudentT::new(df, loc, scale).unwrap();

        let samples = dist.sample(&[100]).unwrap();
        assert_eq!(samples.shape(), &[100]);
        assert!(!samples.requires_grad());
    }

    #[test]
    fn test_student_t_sample_mean() {
        // E[X] = loc = 2.0 for df > 1
        let df = scalar(10.0f32).unwrap();
        let loc = scalar(2.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let dist = StudentT::new(df, loc, scale).unwrap();

        let samples = dist.sample(&[10000]).unwrap();
        let data = samples.data().unwrap();
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        assert!((mean - 2.0).abs() < 0.2, "expected mean ~2.0, got {mean}");
    }

    #[test]
    fn test_student_t_rsample_has_grad() {
        let df = scalar(5.0f32).unwrap();
        let loc = scalar(0.0f32).unwrap().requires_grad_(true);
        let scale = scalar(1.0f32).unwrap().requires_grad_(true);
        let dist = StudentT::new(df, loc, scale).unwrap();

        let samples = dist.rsample(&[5]).unwrap();
        assert!(samples.requires_grad());
        assert!(samples.grad_fn().is_some());
    }

    #[test]
    fn test_student_t_log_prob_at_loc() {
        // StudentT(df=1, loc=0, scale=1) is the standard Cauchy distribution.
        // At x=0: log_prob = lgamma(1) - lgamma(0.5) - 0.5*ln(pi) - ln(1) - 1*ln(1)
        // = 0 - lgamma(0.5) - 0.5*ln(pi)
        // lgamma(0.5) = 0.5*ln(pi), so log_prob = -ln(pi)
        let df = scalar(1.0f32).unwrap();
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let dist = StudentT::new(df, loc, scale).unwrap();

        let x = scalar(0.0f32).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = -(std::f32::consts::PI).ln();
        assert!(
            (lp.item().unwrap() - expected).abs() < 1e-4,
            "expected {expected}, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_student_t_log_prob_symmetry() {
        let df = scalar(5.0f32).unwrap();
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let dist = StudentT::new(df, loc, scale).unwrap();

        let x = from_slice(&[-2.0, 2.0], &[2]).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let data = lp.data().unwrap();
        assert!(
            (data[0] - data[1]).abs() < 1e-5,
            "StudentT log_prob should be symmetric around loc"
        );
    }

    #[test]
    fn test_student_t_log_prob_high_df_approaches_normal() {
        // As df -> inf, StudentT -> Normal
        let df = scalar(10000.0f64).unwrap();
        let loc = scalar(0.0f64).unwrap();
        let scale = scalar(1.0f64).unwrap();
        let dist = StudentT::new(df, loc, scale).unwrap();

        let x = scalar(1.0f64).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        // Normal(0,1).log_prob(1) = -0.5 - 0.5*ln(2*pi)
        let expected = -0.5 - 0.5 * (2.0f64 * std::f64::consts::PI).ln();
        assert!(
            (lp.item().unwrap() - expected).abs() < 0.01,
            "expected ~{expected}, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_student_t_entropy_positive() {
        let df = scalar(5.0f32).unwrap();
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let dist = StudentT::new(df, loc, scale).unwrap();

        let h = dist.entropy().unwrap();
        assert!(
            h.item().unwrap() > 0.0,
            "StudentT entropy should be positive, got {}",
            h.item().unwrap()
        );
    }

    #[test]
    fn test_student_t_shape_mismatch() {
        let df = scalar(5.0f32).unwrap();
        let loc = scalar(0.0f32).unwrap();
        let scale = from_slice(&[1.0f32, 2.0], &[2]).unwrap();
        assert!(StudentT::new(df, loc, scale).is_err());
    }

    #[test]
    fn test_student_t_rsample_backward() {
        let df = scalar(5.0f32).unwrap();
        let loc = scalar(1.0f32).unwrap().requires_grad_(true);
        let scale = scalar(2.0f32).unwrap().requires_grad_(true);
        let dist = StudentT::new(df, loc.clone(), scale.clone()).unwrap();

        let z = dist.rsample(&[10]).unwrap();
        let loss = z.sum_all().unwrap();
        loss.backward().unwrap();

        let loc_grad = loc.grad().unwrap().unwrap();
        assert!(
            (loc_grad.item().unwrap() - 10.0).abs() < 1e-4,
            "expected loc_grad=10.0, got {}",
            loc_grad.item().unwrap()
        );

        let scale_grad = scale.grad().unwrap().unwrap();
        assert!(scale_grad.item().unwrap().is_finite());
    }

    #[test]
    fn test_student_t_f64() {
        let df = scalar(5.0f64).unwrap();
        let loc = scalar(0.0f64).unwrap();
        let scale = scalar(1.0f64).unwrap();
        let dist = StudentT::new(df, loc, scale).unwrap();

        let samples = dist.sample(&[50]).unwrap();
        assert_eq!(samples.shape(), &[50]);
    }

    // -----------------------------------------------------------------------
    // Full PyTorch surface (#1428)
    // -----------------------------------------------------------------------

    #[test]
    fn test_student_t_mean_mode_variance_df_gt_2() {
        // df=5 > 2 → mean=loc, mode=loc, var=scale^2 * df/(df-2)
        let dist = StudentT::new(
            scalar(5.0f64).unwrap(),
            scalar(2.5f64).unwrap(),
            scalar(1.0f64).unwrap(),
        )
        .unwrap();
        assert!((dist.mean().unwrap().item().unwrap() - 2.5).abs() < 1e-10);
        assert!((dist.mode().unwrap().item().unwrap() - 2.5).abs() < 1e-10);
        assert!((dist.variance().unwrap().item().unwrap() - 5.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_student_t_mean_df_le_1_is_nan() {
        // df=1 (Cauchy) → mean is undefined (NaN)
        let dist = StudentT::new(
            scalar(1.0f64).unwrap(),
            scalar(2.5f64).unwrap(),
            scalar(1.0f64).unwrap(),
        )
        .unwrap();
        assert!(dist.mean().unwrap().item().unwrap().is_nan());
    }

    #[test]
    fn test_student_t_variance_df_le_2_is_inf() {
        // df=2 → variance = inf
        let dist = StudentT::new(
            scalar(2.0f64).unwrap(),
            scalar(0.0f64).unwrap(),
            scalar(1.0f64).unwrap(),
        )
        .unwrap();
        assert!(dist.variance().unwrap().item().unwrap().is_infinite());
    }

    #[test]
    fn test_student_t_surface_overrides() {
        let dist = StudentT::new(
            scalar(5.0f64).unwrap(),
            scalar(0.0f64).unwrap(),
            scalar(1.0f64).unwrap(),
        )
        .unwrap();
        assert!(dist.has_rsample());
        assert_eq!(dist.support().unwrap().name(), "Real");
        let args = dist.arg_constraints();
        assert_eq!(args["df"].name(), "Positive");
        assert_eq!(args["loc"].name(), "Real");
        assert_eq!(args["scale"].name(), "Positive");
    }

    #[test]
    fn test_student_t_expand() {
        let dist = StudentT::new(
            scalar(5.0f64).unwrap(),
            scalar(0.0f64).unwrap(),
            scalar(1.0f64).unwrap(),
        )
        .unwrap();
        let exp = dist.expand(&[3]).unwrap();
        let m = exp.mode().unwrap();
        assert_eq!(m.shape(), &[3]);
    }

    // -----------------------------------------------------------------------
    // df gradient via Chi2 implicit reparameterization (#1427) — PARTIAL.
    //
    // The df gradient is NOT shipped: it requires the implicit-Gamma
    // reparameterisation gradient d(sg)/d(alpha) = -(d_alpha P(alpha,sg))/pdf(sg)
    // for the Chi2 sample, which ferrotorch lacks (PyTorch uses a
    // `_standard_gamma_grad` series). These tests (a) pin that the df slot is
    // currently `None`, and (b) DOCUMENT — as an independent scipy-grade FD
    // oracle — that the repo's `GammaRsampleBackward` closed form
    // `sg*(ln sg - digamma(alpha))` is NOT the correct implicit gradient, so it
    // must not be reused here. Both are non-tautological regression anchors.
    // -----------------------------------------------------------------------

    /// Independent reference for the regularized lower incomplete gamma
    /// `P(s, x)` (Numerical-Recipes `gammp`), used as an oracle for the
    /// implicit-Gamma gradient WITHOUT touching production code.
    fn gammp_ref(s: f64, x: f64) -> f64 {
        let gln = lgamma_scalar(s);
        if x < s + 1.0 {
            let mut ap = s;
            let mut sum = 1.0 / s;
            let mut del = sum;
            for _ in 0..500 {
                ap += 1.0;
                del *= x / ap;
                sum += del;
                if del.abs() < sum.abs() * 1e-15 {
                    break;
                }
            }
            sum * (-x + s * x.ln() - gln).exp()
        } else {
            let tiny = 1e-300;
            let mut b = x + 1.0 - s;
            let mut c = 1.0 / tiny;
            let mut d = 1.0 / b;
            let mut h = d;
            for i in 1..500 {
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
            1.0 - (-x + s * x.ln() - gln).exp() * h
        }
    }

    #[test]
    fn test_student_t_df_gradient_currently_none() {
        // Honest pin: with df.requires_grad and loc/scale NOT requiring grad,
        // rsample does not attach a node (only loc/scale drive the graph), so
        // there is no df gradient. This documents REQ-9 staying NOT-STARTED.
        let df = scalar(5.0f64).unwrap().requires_grad_(true);
        let loc = scalar(0.0f64).unwrap();
        let scale = scalar(1.0f64).unwrap();
        let dist = StudentT::new(df.clone(), loc, scale).unwrap();
        let s = dist.rsample(&[8]).unwrap();
        // No loc/scale grad requested -> no backward node attached at all.
        assert!(!s.requires_grad());
        assert!(df.grad().unwrap().is_none());
    }

    #[test]
    fn test_repo_gamma_implicit_grad_formula_is_incorrect() {
        // DOCUMENTS the spillover bug: the implicit reparameterization gradient
        // of a Gamma sample `sg` at shape `alpha` is, by the CDF-inversion
        // identity, d(sg)/d(alpha) = -(dP/dalpha) / pdf(sg). The repo's
        // GammaRsampleBackward / would-be StudentT df term uses the closed form
        // sg*(ln sg - digamma(alpha)), which does NOT match. We assert the
        // MISMATCH so this test fails loudly if someone "fixes" it by reusing
        // the wrong formula. Oracle: independent `gammp_ref` FD.
        let (alpha, sg) = (2.5_f64, 2.0_f64);
        let h = 1e-6;
        let dp_dalpha = (gammp_ref(alpha + h, sg) - gammp_ref(alpha - h, sg)) / (2.0 * h);
        let gln = lgamma_scalar(alpha);
        let pdf = ((alpha - 1.0) * sg.ln() - sg - gln).exp();
        let correct = -dp_dalpha / pdf; // ~ +0.953
        let repo_closed = sg * (sg.ln() - digamma_scalar(alpha)); // ~ -0.020
        assert!(
            correct > 0.9 && correct < 1.0,
            "oracle d(sg)/d(alpha) = {correct}"
        );
        assert!(
            (correct - repo_closed).abs() > 0.5,
            "the closed form should be demonstrably wrong: correct {correct} vs repo {repo_closed}"
        );
    }
}
