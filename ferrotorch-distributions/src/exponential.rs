//! Exponential distribution.
//!
//! `Exponential(rate)` defines an exponential distribution with rate parameter
//! `rate` (lambda). Supports reparameterized sampling via inverse CDF.
//!
//! [CL-329]
//!
//! ## REQ status (per `.design/ferrotorch-distributions/exponential.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`Exponential<T>` struct) | SHIPPED | `pub struct Exponential<T: Float>` with `rate` field mirroring `torch/distributions/exponential.py:14-58`; consumer: `pub use exponential::Exponential` in `lib.rs` |
//! | REQ-2 (constructor) | SHIPPED | `pub fn Exponential::new(rate)`; consumer: re-export |
//! | REQ-3 (`rate` accessor) | SHIPPED | `pub fn Exponential::rate(&self) -> &Tensor<T>`; consumer: `kl_exponential_exponential` / `kl_gamma_exponential` / `kl_exponential_gamma` in `kl.rs` read `.rate().data_vec()?` |
//! | REQ-4 (`Distribution` trait impl) | SHIPPED | `impl<T: Float> Distribution<T> for Exponential<T>`; consumer: trait dispatch + `kl.rs` arms |
//! | REQ-5 (`sample`/`rsample` inverse CDF) | SHIPPED | `-ln(u_safe)/rate` with `1e-30` guard mirroring `exponential.py:68-70`; consumer: trait surface |
//! | REQ-6 (`log_prob`) | SHIPPED | `ln(rate) - rate*x` mirroring `exponential.py:72-75`; consumer: trait surface |
//! | REQ-7 (`entropy`) | SHIPPED | `1 - ln(rate)` mirroring `exponential.py:85-86`; consumer: trait surface |
//! | REQ-8 (`cdf`/`icdf`) | SHIPPED | overrides mirroring `exponential.py:77-83`; consumer: trait surface |
//! | REQ-9 (`mean`/`mode`/`variance`) | SHIPPED | overrides mirroring `exponential.py:35-49`; consumer: trait surface |
//! | REQ-10 (`ExponentialRsampleBackward`) | SHIPPED | `d(z)/d(rate) = ln(u_safe)/rate²` backward; consumer: invoked by the rsample method |
//! | REQ-11 (full PyTorch surface — `expand`/`arg_constraints`/`support`/`has_rsample`) | SHIPPED | trait overrides at the tail of `impl Distribution for Exponential` in `exponential.rs` mirror `torch/distributions/exponential.py:14-49`; consumer: `tests/divergence_distribution_trait_surface.rs::exponential_*` pins every override (closes #1414 — `validate_args` + exp-family hooks remain orthogonal trackers). |

use std::sync::Arc;

use ferrotorch_core::creation;
use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::{GradFn, Tensor};

use crate::constraints;
use crate::{DistConstraint, Distribution};
use std::collections::HashMap;

/// Exponential distribution parameterized by `rate` (lambda).
///
/// # Reparameterization
///
/// `rsample` uses the inverse CDF (quantile) transform:
/// ```text
/// u ~ Uniform(0, 1)
/// z = -log(u) / rate
/// ```
/// Gradients flow through `rate` via the autograd graph.
pub struct Exponential<T: Float> {
    rate: Tensor<T>,
}

impl<T: Float> Exponential<T> {
    /// Create a new Exponential distribution.
    pub fn new(rate: Tensor<T>) -> FerrotorchResult<Self> {
        Ok(Self { rate })
    }

    /// The rate parameter.
    pub fn rate(&self) -> &Tensor<T> {
        &self.rate
    }
}

impl<T: Float> Distribution<T> for Exponential<T> {
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.rate], "Exponential::sample")?;
        let device = self.rate.device();
        let u = creation::rand::<T>(shape)?;
        let u_data = u.data_vec()?;
        let rate_data = self.rate.data_vec()?;

        let result: Vec<T> = u_data
            .iter()
            .zip(rate_data.iter().cycle())
            .map(|(&u_val, &r)| {
                // Clamp u away from 0 for numerical stability
                let u_safe = u_val.max(T::from(1e-30).unwrap());
                -u_safe.ln() / r
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
        crate::fallback::check_gpu_fallback_opt_in(&[&self.rate], "Exponential::rsample")?;
        let device = self.rate.device();
        let u = creation::rand::<T>(shape)?;
        let u_data = u.data_vec()?;
        let rate_data = self.rate.data_vec()?;

        let result: Vec<T> = u_data
            .iter()
            .zip(rate_data.iter().cycle())
            .map(|(&u_val, &r)| {
                let u_safe = u_val.max(T::from(1e-30).unwrap());
                -u_safe.ln() / r
            })
            .collect();

        let storage = TensorStorage::cpu(result);

        let out = if self.rate.requires_grad() && ferrotorch_core::is_grad_enabled() {
            let grad_fn = Arc::new(ExponentialRsampleBackward {
                rate: self.rate.clone(),
                u: u.clone(),
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
        crate::fallback::check_gpu_fallback_opt_in(&[&self.rate, value], "Exponential::log_prob")?;
        // log_prob = log(rate) - rate * x
        let device = self.rate.device();
        let rate_data = self.rate.data_vec()?;
        let val_data = value.data_vec()?;

        let result: Vec<T> = val_data
            .iter()
            .zip(rate_data.iter().cycle())
            .map(|(&x, &r)| r.ln() - r * x)
            .collect();

        let out = Tensor::from_storage(TensorStorage::cpu(result), value.shape().to_vec(), false)?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }

    fn entropy(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.rate], "Exponential::entropy")?;
        // entropy = 1 - log(rate)
        let device = self.rate.device();
        let rate_data = self.rate.data_vec()?;
        let one = <T as num_traits::One>::one();

        let result: Vec<T> = rate_data.iter().map(|&r| one - r.ln()).collect();

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

    fn cdf(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.rate, value], "Exponential::cdf")?;
        // cdf(x) = 1 - exp(-rate * x) for x >= 0; 0 for x < 0.
        let val = value.data_vec()?;
        let rate_data = self.rate.data_vec()?;
        let zero = <T as num_traits::Zero>::zero();
        let one = <T as num_traits::One>::one();
        let result: Vec<T> = val
            .iter()
            .zip(rate_data.iter().cycle())
            .map(
                |(&x, &r)| {
                    if x < zero { zero } else { one - (-r * x).exp() }
                },
            )
            .collect();
        Tensor::from_storage(TensorStorage::cpu(result), value.shape().to_vec(), false)
    }

    fn icdf(&self, q: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.rate, q], "Exponential::icdf")?;
        // icdf(p) = -ln(1 - p) / rate, for p in [0, 1).
        let q_data = q.data_vec()?;
        let rate_data = self.rate.data_vec()?;
        let one = <T as num_traits::One>::one();
        let result: Vec<T> = q_data
            .iter()
            .zip(rate_data.iter().cycle())
            .map(|(&p, &r)| -((one - p).ln()) / r)
            .collect();
        Tensor::from_storage(TensorStorage::cpu(result), q.shape().to_vec(), false)
    }

    fn mean(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.rate], "Exponential::mean")?;
        // 1 / rate
        let rate_data = self.rate.data_vec()?;
        let one = <T as num_traits::One>::one();
        let result: Vec<T> = rate_data.iter().map(|&r| one / r).collect();
        Tensor::from_storage(
            TensorStorage::cpu(result),
            self.rate.shape().to_vec(),
            false,
        )
    }

    fn mode(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.rate], "Exponential::mode")?;
        // Mode of exponential is 0.
        let zero = <T as num_traits::Zero>::zero();
        let n: usize = self.rate.shape().iter().product();
        Tensor::from_storage(
            TensorStorage::cpu(vec![zero; n.max(1)]),
            self.rate.shape().to_vec(),
            false,
        )
    }

    fn variance(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.rate], "Exponential::variance")?;
        // 1 / rate^2
        let rate_data = self.rate.data_vec()?;
        let one = <T as num_traits::One>::one();
        let result: Vec<T> = rate_data.iter().map(|&r| one / (r * r)).collect();
        Tensor::from_storage(
            TensorStorage::cpu(result),
            self.rate.shape().to_vec(),
            false,
        )
    }

    // -----------------------------------------------------------------------
    // Full PyTorch surface (#1376, #1414) — Exponential is continuous,
    // reparameterizable, with support `[0, inf)` and rate parameter
    // strictly positive. Mirrors `torch/distributions/exponential.py:14-30`.
    // -----------------------------------------------------------------------

    fn has_rsample(&self) -> bool {
        // `torch/distributions/exponential.py:25`: `has_rsample = True`.
        true
    }

    fn support(&self) -> Option<Box<dyn DistConstraint>> {
        // `torch/distributions/exponential.py:24`: `support = constraints.nonnegative`.
        Some(Box::new(constraints::NonNegative))
    }

    fn arg_constraints(&self) -> HashMap<&'static str, Box<dyn DistConstraint>> {
        // `torch/distributions/exponential.py:22`:
        //   arg_constraints = {"rate": constraints.positive}
        let mut m: HashMap<&'static str, Box<dyn DistConstraint>> = HashMap::new();
        m.insert("rate", Box::new(constraints::Positive));
        m
    }

    fn event_shape(&self) -> Vec<usize> {
        vec![]
    }

    fn expand(&self, batch_shape: &[usize]) -> FerrotorchResult<Box<dyn Distribution<T>>> {
        // `torch/distributions/exponential.py:43-49`: broadcast `rate`.
        let rate_data = self.rate.data_vec()?;
        let n: usize = batch_shape.iter().product::<usize>().max(1);
        let out: Vec<T> = (0..n).map(|i| rate_data[i % rate_data.len()]).collect();
        let new_rate = Tensor::from_storage(TensorStorage::cpu(out), batch_shape.to_vec(), false)?;
        Ok(Box::new(Exponential::new(new_rate)?))
    }
}

// ---------------------------------------------------------------------------
// Backward nodes
// ---------------------------------------------------------------------------

/// Backward for Exponential rsample: z = -log(u) / rate.
///
/// d(z)/d(rate) = log(u) / rate^2 = -z / rate
#[derive(Debug)]
struct ExponentialRsampleBackward<T: Float> {
    rate: Tensor<T>,
    u: Tensor<T>,
}

impl<T: Float> GradFn<T> for ExponentialRsampleBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let device = grad_output.device();
        let go = grad_output.data_vec()?;
        let u_data = self.u.data_vec()?;
        let rate_data = self.rate.data_vec()?;
        let zero = <T as num_traits::Zero>::zero();

        // d(z)/d(rate) = log(u) / rate^2
        let grad_rate_val: T = go
            .iter()
            .zip(u_data.iter())
            .zip(rate_data.iter().cycle())
            .fold(zero, |acc, ((&g, &u_val), &r)| {
                let u_safe = u_val.max(T::from(1e-30).unwrap());
                acc + g * u_safe.ln() / (r * r)
            });

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

        Ok(vec![if self.rate.requires_grad() {
            Some(grad_rate)
        } else {
            None
        }])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.rate]
    }

    fn name(&self) -> &'static str {
        "ExponentialRsampleBackward"
    }
}

// ---------------------------------------------------------------------------
// ExponentialFamily impl (#1575) — enables the generic Bregman KL fallback
// (`kl_expfamily_expfamily`, `torch/distributions/kl.py:282-300`).
// ---------------------------------------------------------------------------

impl<T: Float> crate::ExponentialFamily<T> for Exponential<T> {
    fn natural_params(&self) -> FerrotorchResult<Vec<Tensor<T>>> {
        // `torch/distributions/exponential.py:88-90`:
        //   _natural_params = (-self.rate,)
        let rate = self.rate.data_vec()?;
        let out: Vec<T> = rate.iter().map(|&r| -r).collect();
        let t = Tensor::from_storage(TensorStorage::cpu(out), self.rate.shape().to_vec(), false)?;
        Ok(vec![t])
    }

    fn log_normalizer(&self, natural_params: &[Tensor<T>]) -> FerrotorchResult<Tensor<T>> {
        // `torch/distributions/exponential.py:92-94`:
        //   _log_normalizer(x) = -log(-x)
        if natural_params.len() != 1 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Exponential::log_normalizer expects 1 natural param, got {}",
                    natural_params.len()
                ),
            });
        }
        let x = natural_params[0].data_vec()?;
        let out: Vec<T> = x.iter().map(|&v| -((-v).ln())).collect();
        Tensor::from_storage(
            TensorStorage::cpu(out),
            natural_params[0].shape().to_vec(),
            false,
        )
    }

    fn mean_params(&self) -> FerrotorchResult<Vec<Tensor<T>>> {
        // ∇A(η) for the Exponential (closed form; torch obtains it by autograd
        // through `_log_normalizer`, `exp_family.py:62`):
        //   A(x) = -log(-x);  ∂A/∂η = -1/x = 1/rate = E[X].
        let rate = self.rate.data_vec()?;
        let one = <T as num_traits::One>::one();
        let out: Vec<T> = rate.iter().map(|&r| one / r).collect();
        let t = Tensor::from_storage(TensorStorage::cpu(out), self.rate.shape().to_vec(), false)?;
        Ok(vec![t])
    }

    fn mean_carrier_measure(&self) -> FerrotorchResult<T> {
        // `torch/distributions/exponential.py:33`: `_mean_carrier_measure = 0`.
        Ok(<T as num_traits::Zero>::zero())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::creation::{from_slice, scalar};

    #[test]
    fn test_exponential_sample_shape() {
        let rate = scalar(1.0f32).unwrap();
        let dist = Exponential::new(rate).unwrap();

        let samples = dist.sample(&[100]).unwrap();
        assert_eq!(samples.shape(), &[100]);
        assert!(!samples.requires_grad());
    }

    #[test]
    fn test_exponential_sample_positive() {
        let rate = scalar(2.0f32).unwrap();
        let dist = Exponential::new(rate).unwrap();

        let samples = dist.sample(&[1000]).unwrap();
        let data = samples.data().unwrap();
        for &x in data {
            assert!(x > 0.0, "Exponential sample should be positive, got {x}");
        }
    }

    #[test]
    fn test_exponential_sample_mean() {
        // E[X] = 1/rate = 0.5
        let rate = scalar(2.0f32).unwrap();
        let dist = Exponential::new(rate).unwrap();

        let samples = dist.sample(&[10000]).unwrap();
        let data = samples.data().unwrap();
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        assert!((mean - 0.5).abs() < 0.05, "expected mean ~0.5, got {mean}");
    }

    #[test]
    fn test_exponential_rsample_has_grad() {
        let rate = scalar(1.0f32).unwrap().requires_grad_(true);
        let dist = Exponential::new(rate).unwrap();

        let samples = dist.rsample(&[5]).unwrap();
        assert!(samples.requires_grad());
        assert!(samples.grad_fn().is_some());
    }

    #[test]
    fn test_exponential_log_prob() {
        // Exp(1): log_prob(x) = -x
        let rate = scalar(1.0f32).unwrap();
        let dist = Exponential::new(rate).unwrap();

        let x = scalar(2.0f32).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = -2.0f32; // log(1) - 1*2 = -2
        assert!(
            (lp.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_exponential_log_prob_rate2() {
        // Exp(2): log_prob(1) = log(2) - 2
        let rate = scalar(2.0f32).unwrap();
        let dist = Exponential::new(rate).unwrap();

        let x = scalar(1.0f32).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = 2.0f32.ln() - 2.0;
        assert!(
            (lp.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_exponential_entropy() {
        // entropy = 1 - log(rate)
        let rate = scalar(2.0f32).unwrap();
        let dist = Exponential::new(rate).unwrap();

        let h = dist.entropy().unwrap();
        let expected = 1.0 - 2.0f32.ln();
        assert!(
            (h.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            h.item().unwrap()
        );
    }

    #[test]
    fn test_exponential_entropy_rate1() {
        // Exp(1): entropy = 1
        let rate = scalar(1.0f32).unwrap();
        let dist = Exponential::new(rate).unwrap();

        let h = dist.entropy().unwrap();
        assert!(
            (h.item().unwrap() - 1.0).abs() < 1e-5,
            "expected 1.0, got {}",
            h.item().unwrap()
        );
    }

    #[test]
    fn test_exponential_rsample_backward() {
        let rate = scalar(2.0f32).unwrap().requires_grad_(true);
        let dist = Exponential::new(rate.clone()).unwrap();

        let z = dist.rsample(&[10]).unwrap();
        let loss = z.sum_all().unwrap();
        loss.backward().unwrap();

        let rate_grad = rate.grad().unwrap().unwrap();
        assert!(rate_grad.item().unwrap().is_finite());
        // Gradient should be negative (increasing rate decreases samples)
        assert!(
            rate_grad.item().unwrap() < 0.0,
            "expected negative grad, got {}",
            rate_grad.item().unwrap()
        );
    }

    #[test]
    fn test_exponential_f64() {
        let rate = scalar(1.0f64).unwrap();
        let dist = Exponential::new(rate).unwrap();

        let samples = dist.sample(&[50]).unwrap();
        assert_eq!(samples.shape(), &[50]);

        let x = scalar(1.0f64).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        assert!((lp.item().unwrap() - (-1.0f64)).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // CDF / ICDF / mean / mode / variance / stddev (#585)
    // -----------------------------------------------------------------------

    #[test]
    fn test_exponential_mean_mode_variance() {
        // rate=2 → mean=0.5, mode=0, var=0.25
        let dist = Exponential::new(scalar(2.0f64).unwrap()).unwrap();
        assert!((dist.mean().unwrap().item().unwrap() - 0.5).abs() < 1e-10);
        assert!(dist.mode().unwrap().item().unwrap().abs() < 1e-12);
        assert!((dist.variance().unwrap().item().unwrap() - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_exponential_cdf() {
        let dist = Exponential::new(scalar(1.0f64).unwrap()).unwrap();
        // cdf(0) = 0; cdf(1) = 1 - 1/e
        let x = from_slice::<f64>(&[-1.0, 0.0, 1.0], &[3]).unwrap();
        let c = dist.cdf(&x).unwrap();
        let d = c.data().unwrap();
        assert!(d[0].abs() < 1e-12);
        assert!(d[1].abs() < 1e-12);
        assert!((d[2] - (1.0 - (-1.0_f64).exp())).abs() < 1e-10);
    }

    #[test]
    fn test_exponential_icdf_roundtrip() {
        let dist = Exponential::new(scalar(2.5f64).unwrap()).unwrap();
        for p in [0.1, 0.3, 0.5, 0.7, 0.9] {
            let q = scalar(p).unwrap();
            let x = dist.icdf(&q).unwrap();
            let p2 = dist.cdf(&x).unwrap();
            assert!((p2.item().unwrap() - p).abs() < 1e-10);
        }
    }
}
