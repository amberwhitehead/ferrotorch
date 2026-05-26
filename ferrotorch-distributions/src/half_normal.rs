//! Half-Normal distribution.
//!
//! `HalfNormal(scale)` defines a half-normal distribution — the absolute value
//! of a `Normal(0, scale)` random variable. Supported on `[0, inf)`.
//! Supports reparameterized sampling.
//!
//! ## REQ status (per `.design/ferrotorch-distributions/half_normal.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`HalfNormal<T>` struct) | SHIPPED | `pub struct HalfNormal<T: Float>` with `scale` field mirroring `torch/distributions/half_normal.py:15-46`; consumer: `pub use half_normal::HalfNormal` in `lib.rs` |
//! | REQ-2 (constructor) | SHIPPED | `pub fn HalfNormal::new(scale)`; consumer: re-export |
//! | REQ-3 (accessors + utilities) | SHIPPED | `pub fn HalfNormal::scale`/`mean_value`/`variance_value`; consumer: `HalfNormal::mean`/`variance` (trait impl) invoke `self.mean_value()?` and `self.variance_value()?` |
//! | REQ-4 (`Distribution` trait impl) | SHIPPED | `impl<T: Float> Distribution<T> for HalfNormal<T>`; consumer: trait dispatch |
//! | REQ-5 (`sample`/`rsample`) | SHIPPED | `scale * |randn|` per `half_normal.py:15-27`; consumer: trait surface |
//! | REQ-6 (`log_prob` with support mask) | SHIPPED | closed-form with `x<0 → -inf` mirroring `half_normal.py:68-73`; consumer: trait surface |
//! | REQ-7 (`entropy`) | SHIPPED | `0.5*ln(π/2) + ln(scale) + 0.5` mirroring `half_normal.py:83-84`; consumer: trait surface |
//! | REQ-8 (`mean`/`mode`/`variance`) | SHIPPED | overrides mirroring `half_normal.py:56-66`; consumer: trait surface |
//! | REQ-9 (`HalfNormalRsampleBackward`) | SHIPPED | `sum(grad_output * |eps|)` backward; consumer: invoked by the rsample method when scale requires grad |
//! | REQ-10 (full PyTorch surface — `has_rsample`/`support`/`arg_constraints`/`expand`/`cdf`/`icdf`) | SHIPPED | the trait overrides at the tail of `impl Distribution for HalfNormal` in `half_normal.rs` mirror `torch/distributions/half_normal.py:33-36`; `cdf`/`icdf` dispatch to `ferrotorch_core::special::erf`/`erfinv`; consumer: trait dispatch via `pub use HalfNormal` re-export. |

use std::collections::HashMap;
use std::sync::Arc;

use ferrotorch_core::creation;
use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::FerrotorchResult;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::{GradFn, Tensor};

use crate::constraints;
use crate::{DistConstraint, Distribution};

/// Half-Normal distribution parameterized by `scale`.
///
/// If `X ~ Normal(0, scale)`, then `|X| ~ HalfNormal(scale)`.
///
/// # Reparameterization
///
/// `rsample` uses the reparameterization trick:
/// ```text
/// eps ~ N(0, 1)
/// z = scale * |eps|
/// ```
/// Gradients flow through `scale` via the autograd graph.
pub struct HalfNormal<T: Float> {
    scale: Tensor<T>,
}

impl<T: Float> HalfNormal<T> {
    /// Create a new Half-Normal distribution.
    pub fn new(scale: Tensor<T>) -> FerrotorchResult<Self> {
        Ok(Self { scale })
    }

    /// The scale parameter.
    pub fn scale(&self) -> &Tensor<T> {
        &self.scale
    }

    /// The mean of the distribution: E[X] = scale * sqrt(2/pi).
    pub fn mean_value(&self) -> FerrotorchResult<Vec<T>> {
        let scale_data = self.scale.data_vec()?;
        let sqrt_2_over_pi = T::from((2.0 / std::f64::consts::PI).sqrt()).unwrap();
        Ok(scale_data.iter().map(|&s| s * sqrt_2_over_pi).collect())
    }

    /// The variance of the distribution: Var[X] = scale^2 * (1 - 2/pi).
    pub fn variance_value(&self) -> FerrotorchResult<Vec<T>> {
        let scale_data = self.scale.data_vec()?;
        let one = <T as num_traits::One>::one();
        let two_over_pi = T::from(2.0 / std::f64::consts::PI).unwrap();
        Ok(scale_data
            .iter()
            .map(|&s| s * s * (one - two_over_pi))
            .collect())
    }
}

impl<T: Float> Distribution<T> for HalfNormal<T> {
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.scale], "HalfNormal::sample")?;
        let device = self.scale.device();
        let eps = creation::randn::<T>(shape)?;
        let eps_data = eps.data_vec()?;
        let scale_data = self.scale.data_vec()?;

        let result: Vec<T> = eps_data
            .iter()
            .zip(scale_data.iter().cycle())
            .map(|(&e, &s)| s * e.abs())
            .collect();

        let out = Tensor::from_storage(TensorStorage::cpu(result), shape.to_vec(), false)?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }

    fn rsample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.scale], "HalfNormal::rsample")?;
        let device = self.scale.device();
        let eps = creation::randn::<T>(shape)?;
        let eps_data = eps.data_vec()?;
        let scale_data = self.scale.data_vec()?;

        let result: Vec<T> = eps_data
            .iter()
            .zip(scale_data.iter().cycle())
            .map(|(&e, &s)| s * e.abs())
            .collect();

        let storage = TensorStorage::cpu(result);

        let out = if self.scale.requires_grad() && ferrotorch_core::is_grad_enabled() {
            let grad_fn = Arc::new(HalfNormalRsampleBackward {
                scale: self.scale.clone(),
                eps: eps.clone(),
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
        crate::fallback::check_gpu_fallback_opt_in(&[&self.scale, value], "HalfNormal::log_prob")?;
        // PDF = sqrt(2 / (pi * scale^2)) * exp(-x^2 / (2 * scale^2))  for x >= 0
        // log_prob = 0.5 * ln(2/pi) - ln(scale) - x^2 / (2 * scale^2)
        //
        // For x < 0, return -inf (log(0)).
        let device = self.scale.device();
        let scale_data = self.scale.data_vec()?;
        let val_data = value.data_vec()?;
        let half = T::from(0.5).unwrap();
        let two_over_pi = T::from(2.0 / std::f64::consts::PI).unwrap();
        let half_ln_2_over_pi = half * two_over_pi.ln();
        let zero = <T as num_traits::Zero>::zero();

        let result: Vec<T> = val_data
            .iter()
            .zip(scale_data.iter().cycle())
            .map(|(&x, &scale)| {
                if x < zero {
                    T::neg_infinity()
                } else {
                    half_ln_2_over_pi - scale.ln() - x * x / (T::from(2.0).unwrap() * scale * scale)
                }
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
        crate::fallback::check_gpu_fallback_opt_in(&[&self.scale], "HalfNormal::entropy")?;
        // entropy = 0.5 * ln(pi * scale^2 / 2) + 0.5
        //         = 0.5 * ln(pi/2) + ln(scale) + 0.5
        let device = self.scale.device();
        let scale_data = self.scale.data_vec()?;
        let half = T::from(0.5).unwrap();
        let pi_over_2 = T::from(std::f64::consts::PI / 2.0).unwrap();
        let half_ln_pi_over_2 = half * pi_over_2.ln();

        let result: Vec<T> = scale_data
            .iter()
            .map(|&scale| half_ln_pi_over_2 + scale.ln() + half)
            .collect();

        let out = Tensor::from_storage(
            TensorStorage::cpu(result),
            self.scale.shape().to_vec(),
            false,
        )?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }

    fn mean(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.scale], "HalfNormal::mean")?;
        let data = self.mean_value()?;
        Tensor::from_storage(TensorStorage::cpu(data), self.scale.shape().to_vec(), false)
    }

    fn mode(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.scale], "HalfNormal::mode")?;
        // Mode of HalfNormal is 0.
        let zero = <T as num_traits::Zero>::zero();
        let n: usize = self.scale.shape().iter().product();
        Tensor::from_storage(
            TensorStorage::cpu(vec![zero; n.max(1)]),
            self.scale.shape().to_vec(),
            false,
        )
    }

    fn variance(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.scale], "HalfNormal::variance")?;
        let data = self.variance_value()?;
        Tensor::from_storage(TensorStorage::cpu(data), self.scale.shape().to_vec(), false)
    }

    fn cdf(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.scale, value], "HalfNormal::cdf")?;
        // F(x) = erf(x / (scale * sqrt(2)))  for x >= 0, else 0.
        // (Equivalently 2 * Phi(x/scale) - 1.)
        let val_data = value.data_vec()?;
        let scale_data = self.scale.data_vec()?;
        let sqrt2 = T::from(std::f64::consts::SQRT_2).unwrap();
        let zero = <T as num_traits::Zero>::zero();

        let z: Vec<T> = val_data
            .iter()
            .zip(scale_data.iter().cycle())
            .map(|(&x, &s)| if x < zero { zero } else { x / (s * sqrt2) })
            .collect();
        let z_tensor = Tensor::from_storage(TensorStorage::cpu(z), value.shape().to_vec(), false)?;
        let erf_z = ferrotorch_core::special::erf(&z_tensor)?;
        // For x < 0 the z is 0 → erf(0) = 0, which matches the support floor.
        let erf_data = erf_z.data_vec()?;
        Tensor::from_storage(TensorStorage::cpu(erf_data), value.shape().to_vec(), false)
    }

    fn icdf(&self, q: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.scale, q], "HalfNormal::icdf")?;
        // F^{-1}(p) = scale * sqrt(2) * erfinv(p)  for p in [0, 1).
        let q_data = q.data_vec()?;
        let arg_tensor =
            Tensor::from_storage(TensorStorage::cpu(q_data), q.shape().to_vec(), false)?;
        let erfinv_arg = ferrotorch_core::special::erfinv(&arg_tensor)?;
        let erfinv_data = erfinv_arg.data_vec()?;
        let scale_data = self.scale.data_vec()?;
        let sqrt2 = T::from(std::f64::consts::SQRT_2).unwrap();
        let result: Vec<T> = erfinv_data
            .iter()
            .zip(scale_data.iter().cycle())
            .map(|(&e, &s)| s * sqrt2 * e)
            .collect();
        Tensor::from_storage(TensorStorage::cpu(result), q.shape().to_vec(), false)
    }

    // -----------------------------------------------------------------------
    // Full PyTorch surface — `half_normal.py:33-36` declares
    //   arg_constraints = {"scale": positive}
    //   support = constraints.nonnegative
    //   has_rsample = True
    // -----------------------------------------------------------------------

    fn has_rsample(&self) -> bool {
        // `torch/distributions/half_normal.py:36`: `has_rsample = True`.
        true
    }

    fn batch_shape(&self) -> Vec<usize> {
        self.scale.shape().to_vec()
    }

    fn support(&self) -> Option<Box<dyn DistConstraint>> {
        // `torch/distributions/half_normal.py:35`: `support = nonnegative`.
        Some(Box::new(constraints::NonNegative))
    }

    fn arg_constraints(&self) -> HashMap<&'static str, Box<dyn DistConstraint>> {
        // `torch/distributions/half_normal.py:33`:
        //   arg_constraints = {"scale": positive}
        let mut m: HashMap<&'static str, Box<dyn DistConstraint>> = HashMap::new();
        m.insert("scale", Box::new(constraints::Positive));
        m
    }

    fn expand(&self, batch_shape: &[usize]) -> FerrotorchResult<Box<dyn Distribution<T>>> {
        let scale_data = self.scale.data_vec()?;
        let n: usize = batch_shape.iter().product::<usize>().max(1);
        let scale_out: Vec<T> = (0..n).map(|i| scale_data[i % scale_data.len()]).collect();
        let new_scale =
            Tensor::from_storage(TensorStorage::cpu(scale_out), batch_shape.to_vec(), false)?;
        Ok(Box::new(HalfNormal::new(new_scale)?))
    }
}

// ---------------------------------------------------------------------------
// Backward nodes
// ---------------------------------------------------------------------------

/// Backward for `z = scale * |eps|`.
///
/// d(z)/d(scale) = |eps| (sum over sample dims)
#[derive(Debug)]
struct HalfNormalRsampleBackward<T: Float> {
    scale: Tensor<T>,
    eps: Tensor<T>,
}

impl<T: Float> GradFn<T> for HalfNormalRsampleBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let device = grad_output.device();
        let go = grad_output.data_vec()?;
        let eps_data = self.eps.data_vec()?;
        let zero = <T as num_traits::Zero>::zero();

        // grad_scale = sum(grad_output * |eps|)
        let grad_scale_val: T = go
            .iter()
            .zip(eps_data.iter())
            .fold(zero, |acc, (&g, &e)| acc + g * e.abs());
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

        Ok(vec![if self.scale.requires_grad() {
            Some(grad_scale)
        } else {
            None
        }])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.scale]
    }

    fn name(&self) -> &'static str {
        "HalfNormalRsampleBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::creation::scalar;

    #[test]
    fn test_half_normal_sample_shape() {
        let scale = scalar(1.0f32).unwrap();
        let dist = HalfNormal::new(scale).unwrap();

        let samples = dist.sample(&[100]).unwrap();
        assert_eq!(samples.shape(), &[100]);
        assert!(!samples.requires_grad());
    }

    #[test]
    fn test_half_normal_sample_nonnegative() {
        let scale = scalar(2.0f32).unwrap();
        let dist = HalfNormal::new(scale).unwrap();

        let samples = dist.sample(&[1000]).unwrap();
        let data = samples.data().unwrap();
        for &x in data {
            assert!(
                x >= 0.0,
                "HalfNormal sample should be non-negative, got {x}"
            );
        }
    }

    #[test]
    fn test_half_normal_sample_mean() {
        // E[X] = scale * sqrt(2/pi) ~ 1.0 * 0.7979 for scale=1
        let scale = scalar(1.0f32).unwrap();
        let dist = HalfNormal::new(scale).unwrap();

        let samples = dist.sample(&[10000]).unwrap();
        let data = samples.data().unwrap();
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        let expected = (2.0f32 / std::f32::consts::PI).sqrt();
        assert!(
            (mean - expected).abs() < 0.05,
            "expected mean ~{expected}, got {mean}"
        );
    }

    #[test]
    fn test_half_normal_rsample_has_grad() {
        let scale = scalar(1.0f32).unwrap().requires_grad_(true);
        let dist = HalfNormal::new(scale).unwrap();

        let samples = dist.rsample(&[5]).unwrap();
        assert!(samples.requires_grad());
        assert!(samples.grad_fn().is_some());
    }

    #[test]
    fn test_half_normal_log_prob_at_zero() {
        // HalfNormal(1) at x=0: log_prob = 0.5*ln(2/pi) - ln(1) - 0 = 0.5*ln(2/pi)
        let scale = scalar(1.0f32).unwrap();
        let dist = HalfNormal::new(scale).unwrap();

        let x = scalar(0.0f32).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = 0.5 * (2.0f32 / std::f32::consts::PI).ln();
        assert!(
            (lp.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_half_normal_log_prob_at_one() {
        // HalfNormal(1) at x=1: log_prob = 0.5*ln(2/pi) - 0 - 0.5 = 0.5*ln(2/pi) - 0.5
        let scale = scalar(1.0f32).unwrap();
        let dist = HalfNormal::new(scale).unwrap();

        let x = scalar(1.0f32).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = 0.5 * (2.0f32 / std::f32::consts::PI).ln() - 0.5;
        assert!(
            (lp.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_half_normal_log_prob_negative_is_neginf() {
        let scale = scalar(1.0f32).unwrap();
        let dist = HalfNormal::new(scale).unwrap();

        let x = scalar(-1.0f32).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        assert!(
            lp.item().unwrap().is_infinite() && lp.item().unwrap() < 0.0,
            "log_prob of negative value should be -inf, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_half_normal_log_prob_scale2() {
        // HalfNormal(2) at x=0: log_prob = 0.5*ln(2/pi) - ln(2) - 0
        let scale = scalar(2.0f32).unwrap();
        let dist = HalfNormal::new(scale).unwrap();

        let x = scalar(0.0f32).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = 0.5 * (2.0f32 / std::f32::consts::PI).ln() - 2.0f32.ln();
        assert!(
            (lp.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_half_normal_entropy() {
        // entropy = 0.5*ln(pi/2) + ln(scale) + 0.5
        // For scale=1: 0.5*ln(pi/2) + 0.5
        let scale = scalar(1.0f32).unwrap();
        let dist = HalfNormal::new(scale).unwrap();

        let h = dist.entropy().unwrap();
        let expected = 0.5 * (std::f32::consts::PI / 2.0).ln() + 0.5;
        assert!(
            (h.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            h.item().unwrap()
        );
    }

    #[test]
    fn test_half_normal_entropy_scale2() {
        let scale = scalar(2.0f32).unwrap();
        let dist = HalfNormal::new(scale).unwrap();

        let h = dist.entropy().unwrap();
        let expected = 0.5 * (std::f32::consts::PI / 2.0).ln() + 2.0f32.ln() + 0.5;
        assert!(
            (h.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            h.item().unwrap()
        );
    }

    #[test]
    fn test_half_normal_rsample_backward() {
        let scale = scalar(2.0f32).unwrap().requires_grad_(true);
        let dist = HalfNormal::new(scale.clone()).unwrap();

        let z = dist.rsample(&[10]).unwrap();
        let loss = z.sum_all().unwrap();
        loss.backward().unwrap();

        let scale_grad = scale.grad().unwrap().unwrap();
        assert!(scale_grad.item().unwrap().is_finite());
        // d(sum(scale*|eps|))/d(scale) = sum(|eps|) > 0
        assert!(
            scale_grad.item().unwrap() > 0.0,
            "expected positive scale_grad, got {}",
            scale_grad.item().unwrap()
        );
    }

    #[test]
    fn test_half_normal_f64() {
        let scale = scalar(1.0f64).unwrap();
        let dist = HalfNormal::new(scale).unwrap();

        let samples = dist.sample(&[50]).unwrap();
        assert_eq!(samples.shape(), &[50]);

        let x = scalar(0.0f64).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = 0.5 * (2.0f64 / std::f64::consts::PI).ln();
        assert!((lp.item().unwrap() - expected).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // mean / mode / variance (#585)
    // -----------------------------------------------------------------------

    #[test]
    fn test_half_normal_mean_mode_variance() {
        let dist = HalfNormal::new(scalar(1.0f64).unwrap()).unwrap();
        // mean = sqrt(2/pi)
        assert!(
            (dist.mean().unwrap().item().unwrap() - (2.0_f64 / std::f64::consts::PI).sqrt()).abs()
                < 1e-10
        );
        // mode = 0
        assert!(dist.mode().unwrap().item().unwrap().abs() < 1e-12);
        // var = 1 - 2/pi
        assert!(
            (dist.variance().unwrap().item().unwrap() - (1.0 - 2.0 / std::f64::consts::PI)).abs()
                < 1e-10
        );
    }

    // -----------------------------------------------------------------------
    // Full PyTorch surface (#1421)
    // -----------------------------------------------------------------------

    #[test]
    fn test_half_normal_cdf_at_zero_is_zero() {
        let dist = HalfNormal::new(scalar(1.0f64).unwrap()).unwrap();
        let x = scalar(0.0f64).unwrap();
        let c = dist.cdf(&x).unwrap();
        assert!(c.item().unwrap().abs() < 1e-10);
    }

    #[test]
    fn test_half_normal_cdf_icdf_roundtrip() {
        let dist = HalfNormal::new(scalar(1.0f64).unwrap()).unwrap();
        for p in [0.1, 0.3, 0.5, 0.7, 0.9] {
            let q = scalar(p).unwrap();
            let x = dist.icdf(&q).unwrap();
            let p2 = dist.cdf(&x).unwrap();
            assert!(
                (p2.item().unwrap() - p).abs() < 5e-3,
                "p={p}, recovered={}",
                p2.item().unwrap()
            );
        }
    }

    #[test]
    fn test_half_normal_surface_overrides() {
        let dist = HalfNormal::new(scalar(1.0f64).unwrap()).unwrap();
        assert!(dist.has_rsample());
        assert_eq!(dist.support().unwrap().name(), "NonNegative");
        let args = dist.arg_constraints();
        assert_eq!(args["scale"].name(), "Positive");
    }

    #[test]
    fn test_half_normal_expand() {
        let dist = HalfNormal::new(scalar(2.0f64).unwrap()).unwrap();
        let exp = dist.expand(&[3]).unwrap();
        let m = exp.mean().unwrap();
        assert_eq!(m.shape(), &[3]);
    }
}
