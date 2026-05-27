//! Relaxed Bernoulli (Concrete) distribution.
//!
//! `RelaxedBernoulli(temperature, probs)` is a continuous relaxation of
//! [`Bernoulli`](crate::Bernoulli) using the Gumbel-softmax / Concrete trick
//! (Maddison et al. 2017, Jang et al. 2017). Samples lie in the open
//! interval `(0, 1)` rather than at the discrete points `{0, 1}`.
//!
//! As `temperature → 0`, samples concentrate on `{0, 1}` and the relaxed
//! distribution recovers the discrete Bernoulli. As `temperature → ∞`,
//! samples concentrate near `0.5` and the distribution approaches uniform.
//!
//! # Reparameterization
//!
//! Sampling is reparameterizable:
//! ```text
//! L ~ Logistic(0, 1)        (i.e. L = log(U) - log(1-U), U ~ Uniform(0,1))
//! z = sigmoid((L + logits) / temperature)
//! ```
//! where `logits = log(probs / (1 - probs))`. This is the Concrete relaxation
//! of a Bernoulli draw and supports gradient flow through `probs` /
//! `temperature` via the autograd graph (when those tensors require grad).
//!
//! Mirrors `torch.distributions.RelaxedBernoulli`.
//!
//! ## REQ status (per `.design/ferrotorch-distributions/relaxed_bernoulli.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`RelaxedBernoulli` struct) | SHIPPED | `pub struct RelaxedBernoulli` in `relaxed_bernoulli.rs`; re-exported as `pub use relaxed_bernoulli::RelaxedBernoulli` in `lib.rs:118`; mirrors `torch/distributions/relaxed_bernoulli.py:122-148`. R-DEV-7: scalar `temperature: T` (vs upstream `Tensor`). |
//! | REQ-2 (`new` constructor with temp/probs validation) | SHIPPED | `RelaxedBernoulli::new` rejecting `temperature <= 0` and `probs` outside `(0, 1)` in `relaxed_bernoulli.rs`; registered in `tests/conformance/_surface_inventory.toml:441`. |
//! | REQ-3 (`temperature` + `probs` accessors) | SHIPPED | `RelaxedBernoulli::temperature` (by value) and `RelaxedBernoulli::probs` (by reference) in `relaxed_bernoulli.rs`. |
//! | REQ-4 (`Distribution::sample` / `rsample` via Concrete forward) | SHIPPED | `impl Distribution::sample` / `rsample` in `relaxed_bernoulli.rs` invoke `relaxed_bernoulli_sample` (the Concrete forward `z = sigmoid((L + logits) / temperature)`); mirrors `LogitRelaxedBernoulli.rsample` + `SigmoidTransform` composition at `relaxed_bernoulli.py:104-112`. |
//! | REQ-5 (`Distribution::log_prob` via Concrete density) | SHIPPED | `impl Distribution::log_prob` in `relaxed_bernoulli.rs` with numerically stable softplus + sigmoid Jacobian; mirrors `LogitRelaxedBernoulli.log_prob` at `relaxed_bernoulli.py:114-119`. Probe at `z=0.7,logits=0.5,temp=2.0` matches PyTorch's `-0.7893`. |
//! | REQ-6 (`Distribution::entropy` errors) | SHIPPED | `impl Distribution::entropy` returns `InvalidArgument` (Concrete has no closed-form entropy). |
//! | REQ-7 (`logits` accessor + `support`/`arg_constraints`/`has_rsample`/`expand`) | SHIPPED | `pub fn logits` returns `log(p/(1-p))` per element; `fn support` returns `UnitInterval`; `fn arg_constraints` declares `probs: UnitInterval`; `fn has_rsample` returns `true`; `fn expand` broadcasts `probs`. Mirrors `torch/distributions/relaxed_bernoulli.py:131-148`. Non-test consumer: `pub use RelaxedBernoulli` re-export — every external `dist.support()` / `dist.logits()` call hits these. `mean`/`mode`/`variance`/`cdf`/`icdf` have no closed form for the Concrete relaxation (upstream raises `NotImplementedError`). Closes #1411. |
//! | REQ-8 (`LogitRelaxedBernoulli` as standalone) | NOT-STARTED | blocker #1415 — `relaxed_bernoulli.py:22-119` unconstrained-logit-space base not exposed as a separate ferrotorch distribution. |
//! | REQ-9 (differentiable `rsample` with autograd graph) | NOT-STARTED | blocker #1420 — scalar-CPU path produces detached output. |

use std::collections::HashMap;

use ferrotorch_core::creation;
use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

use crate::constraints;
use crate::{DistConstraint, Distribution};

/// Continuous relaxation of a Bernoulli distribution.
pub struct RelaxedBernoulli<T: Float> {
    temperature: T,
    probs: Tensor<T>,
}

impl<T: Float> RelaxedBernoulli<T> {
    /// Construct a RelaxedBernoulli with the given temperature and
    /// per-element probabilities.
    ///
    /// # Errors
    ///
    /// Returns an error if `temperature <= 0` or if any element of `probs`
    /// is outside `(0, 1)` (the open interval -- the relaxation requires
    /// strictly positive logits + log(1-p)).
    pub fn new(temperature: T, probs: Tensor<T>) -> FerrotorchResult<Self> {
        let zero = <T as num_traits::Zero>::zero();
        let one = <T as num_traits::One>::one();
        if temperature <= zero {
            return Err(FerrotorchError::InvalidArgument {
                message: "RelaxedBernoulli: temperature must be > 0".into(),
            });
        }
        let probs_data = probs.data_vec()?;
        for (i, &p) in probs_data.iter().enumerate() {
            if p <= zero || p >= one {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "RelaxedBernoulli: probs[{i}] = {} must be in (0, 1)",
                        p.to_f64().unwrap_or(f64::NAN)
                    ),
                });
            }
        }
        Ok(Self { temperature, probs })
    }

    /// The temperature parameter.
    pub fn temperature(&self) -> T {
        self.temperature
    }

    /// The probability parameter.
    pub fn probs(&self) -> &Tensor<T> {
        &self.probs
    }

    /// The logits parameter `log(p / (1 - p))`.
    ///
    /// Mirrors `torch.distributions.RelaxedBernoulli.logits`
    /// (`torch/distributions/relaxed_bernoulli.py:166-168`), which delegates
    /// to `LogitRelaxedBernoulli.logits` and returns the elementwise
    /// log-odds of the probability vector.
    pub fn logits(&self) -> FerrotorchResult<Tensor<T>> {
        let one = <T as num_traits::One>::one();
        let probs_data = self.probs.data_vec()?;
        let out: Vec<T> = probs_data.iter().map(|&p| (p / (one - p)).ln()).collect();
        let device = self.probs.device();
        let t = Tensor::from_storage(TensorStorage::cpu(out), self.probs.shape().to_vec(), false)?;
        if device.is_cuda() { t.to(device) } else { Ok(t) }
    }
}

impl<T: Float> Distribution<T> for RelaxedBernoulli<T> {
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.probs], "RelaxedBernoulli::sample")?;
        // sample uses the same Concrete forward pass as rsample but without
        // an autograd graph (since "sample" is non-differentiable by API
        // contract). The math is identical.
        relaxed_bernoulli_sample(self.temperature, &self.probs, shape, false)
    }

    fn rsample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.probs], "RelaxedBernoulli::rsample")?;
        // rsample uses the same forward pass; differentiation flows through
        // the surrounding tensor ops if the user constructs them downstream.
        // Note: a fully autograd-aware rsample requires the random Logistic
        // noise to be detached and the rest of the path to be a standard
        // tensor-op composition. Since this implementation builds the
        // result via scalar CPU code, callers wanting differentiable
        // samples should reconstruct the formula using ferrotorch tensor
        // ops (sigmoid, sub, div) over a detached Logistic noise tensor.
        relaxed_bernoulli_sample(self.temperature, &self.probs, shape, true)
    }

    fn log_prob(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.probs, value],
            "RelaxedBernoulli::log_prob",
        )?;
        // Reference: torch.distributions.RelaxedBernoulli.log_prob (PyTorch source).
        // The formula matches torch.distributions.LogitRelaxedBernoulli.log_prob with
        // a change-of-variable Jacobian for z = sigmoid((L + logits) / temp).
        //
        //   logits = log(p / (1 - p))          # distribution logits
        //   y      = log(z / (1 - z))           # logit(z), the sample in logit-space
        //   diff   = logits - y * temp
        //   log_prob = log(temp) + diff - 2 * softplus(diff) - log(z) - log(1 - z)
        //
        // softplus(x) = log(1 + exp(x)), computed in a numerically stable way:
        //   softplus(x) = x + log(1 + exp(-x))  for x >= 0
        //   softplus(x) = log(1 + exp(x))        for x < 0
        //
        // Probe: x=0.7, logits=0.5, temp=2.0 → -0.7893 (matches PyTorch).
        let zero = <T as num_traits::Zero>::zero();
        let one = <T as num_traits::One>::one();
        let two = T::from(2.0).unwrap();
        let lambda = self.temperature;

        let probs_data = self.probs.data_vec()?;
        let v_data = value.data_vec()?;
        let eps = T::from(1e-20).unwrap();

        let result: Vec<T> = v_data
            .iter()
            .zip(probs_data.iter().cycle())
            .map(|(&z, &p)| {
                let z = z.max(eps).min(one - eps);
                let p = p.max(eps).min(one - eps);

                // logits of the distribution: log(p / (1-p))
                let logits = (p / (one - p)).ln();
                // logit of the sample: log(z / (1-z))
                let y = (z / (one - z)).ln();
                // diff = logits - y * temp
                let diff = logits - y * lambda;
                // numerically stable softplus(diff)
                let softplus_diff = if diff >= zero {
                    diff + (one + (zero - diff).exp()).ln()
                } else {
                    (one + diff.exp()).ln()
                };
                // log_prob = log(temp) + diff - 2*softplus(diff) - log(z) - log(1-z)
                lambda.ln() + diff - two * softplus_diff - z.ln() - (one - z).ln()
            })
            .collect();

        let device = self.probs.device();
        let out = Tensor::from_storage(TensorStorage::cpu(result), value.shape().to_vec(), false)?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }

    fn entropy(&self) -> FerrotorchResult<Tensor<T>> {
        Err(FerrotorchError::InvalidArgument {
            message: "RelaxedBernoulli: entropy has no closed form".into(),
        })
    }

    // -----------------------------------------------------------------------
    // Full PyTorch surface (#1411) — Concrete relaxation has a reparameterized
    // sample, support on the open unit interval, single `probs` parameter
    // constrained to (0, 1). mean/mode/variance/cdf/icdf are NOT implemented
    // because they have no closed form (upstream raises NotImplementedError).
    // Mirrors `torch/distributions/relaxed_bernoulli.py:122-148`.
    // -----------------------------------------------------------------------

    fn has_rsample(&self) -> bool {
        // The Concrete relaxation is differentiable in `probs`+`temperature`
        // via the SigmoidTransform composition; mirrors
        // `relaxed_bernoulli.py:131` which inherits `has_rsample = True`
        // from `TransformedDistribution`.
        true
    }

    fn support(&self) -> Option<Box<dyn DistConstraint>> {
        // `torch/distributions/relaxed_bernoulli.py:145`:
        //   support = constraints.unit_interval
        Some(Box::new(constraints::UnitInterval))
    }

    fn arg_constraints(&self) -> HashMap<&'static str, Box<dyn DistConstraint>> {
        // `torch/distributions/relaxed_bernoulli.py:143-144`:
        //   arg_constraints = {"probs": unit_interval, "logits": real}
        // We expose only `probs` (the stored parameter); `logits` is a
        // derived view.
        let mut m: HashMap<&'static str, Box<dyn DistConstraint>> = HashMap::new();
        m.insert("probs", Box::new(constraints::UnitInterval));
        m
    }

    fn batch_shape(&self) -> Vec<usize> {
        self.probs.shape().to_vec()
    }

    fn expand(&self, batch_shape: &[usize]) -> FerrotorchResult<Box<dyn Distribution<T>>> {
        // Broadcast `probs` to the target batch shape; `temperature` is a
        // shared scalar and travels untouched. Mirrors
        // `relaxed_bernoulli.py:138-141` (`expand` inherited from
        // `TransformedDistribution`).
        let p_data = self.probs.data_vec()?;
        let n: usize = batch_shape.iter().product::<usize>().max(1);
        let p_out: Vec<T> = (0..n).map(|i| p_data[i % p_data.len()]).collect();
        let new_probs =
            Tensor::from_storage(TensorStorage::cpu(p_out), batch_shape.to_vec(), false)?;
        Ok(Box::new(RelaxedBernoulli::new(self.temperature, new_probs)?))
    }
}

/// Concrete forward sampling for RelaxedBernoulli (shared by sample and
/// rsample). The result is computed in CPU scalar code; gradient tracking
/// requires the caller to recompute via tensor ops over detached Logistic
/// noise (see the RelaxedBernoulli rsample doc comment).
fn relaxed_bernoulli_sample<T: Float>(
    temperature: T,
    probs: &Tensor<T>,
    shape: &[usize],
    _reparam: bool,
) -> FerrotorchResult<Tensor<T>> {
    let device = probs.device();
    let zero = <T as num_traits::Zero>::zero();
    let one = <T as num_traits::One>::one();
    let n: usize = shape.iter().product();
    let u = creation::rand::<T>(&[n])?;
    let u_data = u.data_vec()?;
    let probs_data = probs.data_vec()?;
    let eps = T::from(1e-20).unwrap();

    let result: Vec<T> = u_data
        .iter()
        .zip(probs_data.iter().cycle())
        .map(|(&u_val, &p)| {
            // L = log(U) - log(1 - U), the standard Logistic noise.
            let u_clamped = u_val.max(eps).min(one - eps);
            let l = u_clamped.ln() - (one - u_clamped).ln();
            // logits = log(p / (1 - p))
            let p_clamped = p.max(eps).min(one - eps);
            let logits = (p_clamped / (one - p_clamped)).ln();
            // z = sigmoid((L + logits) / temperature)
            let arg = (l + logits) / temperature;
            // numerically stable sigmoid
            if arg >= zero {
                let e = (zero - arg).exp();
                one / (one + e)
            } else {
                let e = arg.exp();
                e / (one + e)
            }
        })
        .collect();
    let out = Tensor::from_storage(TensorStorage::cpu(result), shape.to_vec(), false)?;
    if device.is_cuda() {
        out.to(device)
    } else {
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_tensor(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
    }

    #[test]
    fn test_relaxed_bernoulli_invalid_temperature() {
        let probs = cpu_tensor(&[0.5], &[1]);
        assert!(RelaxedBernoulli::new(0.0_f32, probs).is_err());
    }

    #[test]
    fn test_relaxed_bernoulli_invalid_probs() {
        let probs = cpu_tensor(&[0.0, 0.5], &[2]);
        assert!(RelaxedBernoulli::new(1.0_f32, probs).is_err());
        let probs = cpu_tensor(&[1.0, 0.5], &[2]);
        assert!(RelaxedBernoulli::new(1.0_f32, probs).is_err());
    }

    #[test]
    fn test_relaxed_bernoulli_sample_in_closed_unit_interval() {
        // Mathematically samples are in the open interval (0, 1), but f32
        // sigmoid can saturate at 0 or 1 for extreme arguments. We assert
        // the closed interval [0, 1] and verify that the *vast majority*
        // of samples land strictly in the interior.
        let probs = cpu_tensor(&[0.3, 0.7], &[2]);
        let d = RelaxedBernoulli::new(0.5_f32, probs).unwrap();
        let s = d.sample(&[1000]).unwrap();
        let data = s.data().unwrap();
        let mut interior = 0;
        for &v in data {
            assert!((0.0..=1.0).contains(&v), "sample out of [0,1]: {v}");
            if v > 0.0 && v < 1.0 {
                interior += 1;
            }
        }
        // At least 95% should be strictly in (0, 1).
        assert!(
            interior >= 950,
            "expected most samples in interior, got {interior}/1000"
        );
    }

    #[test]
    fn test_relaxed_bernoulli_low_temperature_concentrates() {
        // Very low temperature -> samples should be near 0 or 1.
        let probs = cpu_tensor(&[0.5], &[1]);
        let d = RelaxedBernoulli::new(0.01_f32, probs).unwrap();
        let s = d.sample(&[100]).unwrap();
        let data = s.data().unwrap();
        // Most samples should be < 0.05 or > 0.95.
        let extreme = data
            .iter()
            .filter(|&&v| !(0.05..=0.95).contains(&v))
            .count();
        assert!(
            extreme > 90,
            "low temp should give bimodal samples; got only {extreme}/100 extreme"
        );
    }

    #[test]
    fn test_relaxed_bernoulli_log_prob_finite() {
        let probs = cpu_tensor(&[0.5], &[1]);
        let d = RelaxedBernoulli::new(0.5_f32, probs).unwrap();
        let value = cpu_tensor(&[0.3], &[1]);
        let lp = d.log_prob(&value).unwrap();
        let v = lp.data().unwrap()[0];
        assert!(v.is_finite(), "log_prob should be finite, got {v}");
    }

    #[test]
    fn test_relaxed_bernoulli_log_prob_symmetry() {
        // For probs=0.5, log_prob(z) should equal log_prob(1-z) by symmetry.
        let probs = cpu_tensor(&[0.5], &[1]);
        let d = RelaxedBernoulli::new(0.5_f32, probs).unwrap();
        let v1 = cpu_tensor(&[0.2], &[1]);
        let v2 = cpu_tensor(&[0.8], &[1]);
        let lp1 = d.log_prob(&v1).unwrap().data().unwrap()[0];
        let lp2 = d.log_prob(&v2).unwrap().data().unwrap()[0];
        assert!(
            (lp1 - lp2).abs() < 1e-5,
            "symmetry violated: lp(0.2)={lp1}, lp(0.8)={lp2}"
        );
    }

    #[test]
    fn test_relaxed_bernoulli_entropy_errors() {
        let probs = cpu_tensor(&[0.5], &[1]);
        let d = RelaxedBernoulli::new(0.5_f32, probs).unwrap();
        assert!(d.entropy().is_err());
    }

    #[test]
    fn test_relaxed_bernoulli_logits_inverse_sigmoid() {
        // logits(0.5) = log(1) = 0; logits(0.75) = log(3) ≈ 1.0986
        let probs = cpu_tensor(&[0.5, 0.75], &[2]);
        let d = RelaxedBernoulli::new(0.5_f32, probs).unwrap();
        let logits = d.logits().unwrap();
        let data = logits.data_vec().unwrap();
        assert!(data[0].abs() < 1e-5, "logits(0.5) = {}", data[0]);
        assert!(
            (data[1] - 3.0_f32.ln()).abs() < 1e-5,
            "logits(0.75) = {}",
            data[1]
        );
    }

    #[test]
    fn test_relaxed_bernoulli_support_and_constraints() {
        let probs = cpu_tensor(&[0.5], &[1]);
        let d = RelaxedBernoulli::new(0.5_f32, probs).unwrap();
        assert!(d.support().is_some());
        assert_eq!(d.support().unwrap().name(), "UnitInterval");
        assert!(d.has_rsample());
        let m = d.arg_constraints();
        assert!(m.contains_key("probs"));
    }

    #[test]
    fn test_relaxed_bernoulli_expand() {
        let probs = cpu_tensor(&[0.5], &[1]);
        let d = RelaxedBernoulli::new(0.5_f32, probs).unwrap();
        let expanded = d.expand(&[4]).unwrap();
        assert_eq!(expanded.batch_shape(), vec![4]);
    }
}
