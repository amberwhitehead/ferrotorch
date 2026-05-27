//! Mixture distribution where all components are from the same family.
//!
//! `MixtureSameFamily(mixing, components)` defines a finite mixture
//! distribution. The mixing distribution is a `Categorical` over `K`
//! components, and the components are encoded as a single distribution
//! whose batch shape's rightmost dimension is the component index.
//!
//! Sample procedure:
//!   1. Draw a component index `k` from `mixing` (Categorical).
//!   2. Sample from component `k` of the components distribution.
//!
//! `log_prob(x) = logsumexp_k( log mixing_probs[k] + components_log_prob[k](x) )`.
//!
//! Mirrors `torch.distributions.MixtureSameFamily`.
//!
//! # Limitations
//!
//! This implementation accepts the components distribution by-value as a
//! type that yields per-component log_probs of shape `[..., K]` where the
//! rightmost dim is the component axis. Sample currently supports only
//! the simple case where the components share parameters and the
//! per-sample selection is performed in scalar code on CPU.
//!
//! `rsample` is not supported (mixture sampling is non-reparameterizable
//! without Gumbel-softmax tricks).
//!
//! ## REQ status (per `.design/ferrotorch-distributions/mixture_same_family.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`MixtureSameFamily<T, D>` struct) | SHIPPED | `pub struct MixtureSameFamily` in `mixture_same_family.rs`; re-exported via `lib.rs`; mirrors `torch/distributions/mixture_same_family.py:13-109`. |
//! | REQ-2 (`new` constructor with zero-K rejection) | SHIPPED | the constructor in `mixture_same_family.rs`. |
//! | REQ-3 (3 accessors: mixing/components/num_components) | SHIPPED | the accessors in `mixture_same_family.rs`. |
//! | REQ-4 (`Distribution<T>` impl with 4 methods) | SHIPPED | the impl block in `mixture_same_family.rs`. |
//! | REQ-5 (two-step ancestor sampling) | SHIPPED | the sample body in `mixture_same_family.rs` does mixing.sample then gather. |
//! | REQ-6 (log_prob via logsumexp) | SHIPPED | the manual logsumexp body in `mixture_same_family.rs`. |
//! | REQ-7 (`rsample` errors — not reparameterizable) | SHIPPED | the `rsample` body returns `InvalidArgument` in `mixture_same_family.rs`. |
//! | REQ-8 (`entropy` errors — no closed form) | SHIPPED | the `entropy` body returns `InvalidArgument` in `mixture_same_family.rs`. |
//! | REQ-9 (mean/variance via law-of-total-variance) | SHIPPED | `fn mean` returns `sum_k mix_probs[k] * components.mean()[k]` and `fn variance` uses law of total variance `E[Var(X|K)] + Var(E[X|K])` mirroring `torch/distributions/mixture_same_family.py:155-189`; non-test consumer: `pub use mixture_same_family::MixtureSameFamily` re-export — every external `dist.mean()` / `dist.variance()` call hits these overrides; closes #1388. |
//! | REQ-10 (cdf via sum cdf_x * mix_probs) | SHIPPED | `fn cdf` returns `sum_k mix_probs[k] * components.cdf(value)[k]` mirroring `torch/distributions/mixture_same_family.py:191-201`; non-test consumer: `pub use MixtureSameFamily` re-export; closes #1389. |
//! | REQ-11 (multi-event-dim component support) | NOT-STARTED | blocker #1390 — current impl assumes scalar event_shape. |

use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

use crate::{Categorical, Distribution};

/// Finite mixture distribution with same-family components.
///
/// The components distribution `D` must produce log_probs of shape
/// `[..., K]` for input values, where `K` is the number of mixture
/// components and matches the size of the mixing Categorical.
pub struct MixtureSameFamily<T: Float, D: Distribution<T>> {
    mixing: Categorical<T>,
    components: D,
    /// Number of mixture components (K). Equal to mixing.num_categories().
    num_components: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float, D: Distribution<T>> MixtureSameFamily<T, D> {
    /// Build a new mixture from a Categorical mixing distribution and a
    /// component distribution that yields per-component log_probs.
    ///
    /// # Errors
    ///
    /// Returns an error if `mixing` has zero components.
    pub fn new(mixing: Categorical<T>, components: D) -> FerrotorchResult<Self> {
        let k = mixing.num_categories();
        if k == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "MixtureSameFamily: mixing distribution must have at least 1 component"
                    .into(),
            });
        }
        Ok(Self {
            mixing,
            components,
            num_components: k,
            _phantom: std::marker::PhantomData,
        })
    }

    /// The mixing weights distribution.
    pub fn mixing(&self) -> &Categorical<T> {
        &self.mixing
    }

    /// The components distribution.
    pub fn components(&self) -> &D {
        &self.components
    }

    /// Number of mixture components.
    pub fn num_components(&self) -> usize {
        self.num_components
    }
}

impl<T: Float, D: Distribution<T>> Distribution<T> for MixtureSameFamily<T, D> {
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[self.mixing.probs()],
            "MixtureSameFamily::sample",
        )?;
        // Two-step sampling:
        //   1. Draw a component index per-output from the mixing Categorical.
        //   2. Draw a sample from each chosen component.
        //
        // Because the components distribution does not expose per-component
        // index access in the trait, we draw `numel` samples from the full
        // components distribution at the requested shape and then index
        // along the component axis using the chosen indices. This relies
        // on the convention that the components distribution sample with
        // shape [..., K, *event_shape] gives all-K outputs per draw.
        //
        // For the simplest and most common case (a Normal with batch shape
        // [K]), we instead just draw `numel` samples from the components
        // distribution and pick the right one per-output. This requires
        // CPU-side gather logic.
        //
        // We implement the simple case: components.sample(shape) is
        // assumed to produce a tensor whose rightmost dim is the
        // component axis, of size K. We then gather one slice per output.

        // Step 1: pick component indices from the mixing distribution.
        let comp_idx = self.mixing.sample(shape)?;
        let comp_idx_data = comp_idx.data_vec()?;

        // Step 2: draw component samples. We expand the request shape with
        // a trailing K dim so the components distribution produces all
        // possible outputs, then we index per-element with comp_idx.
        let mut comp_shape: Vec<usize> = shape.to_vec();
        comp_shape.push(self.num_components);
        let comp_samples = self.components.sample(&comp_shape)?;
        let comp_data = comp_samples.data_vec()?;

        // Build the gathered output: for each output index i,
        // out[i] = comp_data[i * K + comp_idx[i]].
        let numel: usize = shape.iter().product();
        let mut result = Vec::with_capacity(numel);
        for (i, k_float) in comp_idx_data.iter().enumerate().take(numel) {
            let k = k_float.to_usize().unwrap_or(0).min(self.num_components - 1);
            let val = comp_data[i * self.num_components + k];
            result.push(val);
        }

        let device = self.mixing.probs().device();
        let out = Tensor::from_storage(TensorStorage::cpu(result), shape.to_vec(), false)?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }

    fn rsample(&self, _shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        Err(FerrotorchError::InvalidArgument {
            message:
                "MixtureSameFamily: rsample is not supported -- mixture sampling is not reparameterizable. \
                 Use Gumbel-softmax (RelaxedOneHotCategorical) for differentiable approximations."
                    .into(),
        })
    }

    fn log_prob(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[self.mixing.probs(), value],
            "MixtureSameFamily::log_prob",
        )?;
        // log_prob(x) = logsumexp_k( log mixing_probs[k] + log p_k(x) )
        //
        // We compute the per-component log_probs by replicating value K
        // times along a new last axis (so it broadcasts against the
        // components distribution's K-shaped batch dim) and asking the
        // components distribution for the log_prob.
        //
        // Concretely: value has shape [...], components' log_prob expects
        // a tensor whose trailing dim is K. We unsqueeze and tile value to
        // shape [..., K], then call components.log_prob.
        let v_shape = value.shape().to_vec();
        let mut tiled_shape = v_shape.clone();
        tiled_shape.push(self.num_components);

        let v_data = value.data_vec()?;
        let v_numel = v_data.len();
        let mut tiled = Vec::with_capacity(v_numel * self.num_components);
        for &v in v_data.iter() {
            for _ in 0..self.num_components {
                tiled.push(v);
            }
        }
        let v_dev = value.device();
        let value_tiled = {
            let t = Tensor::from_storage(TensorStorage::cpu(tiled), tiled_shape.clone(), false)?;
            if v_dev.is_cuda() { t.to(v_dev)? } else { t }
        };

        // Per-component log p_k(x): shape [..., K].
        let comp_lp = self.components.log_prob(&value_tiled)?;
        let comp_lp_data = comp_lp.data_vec()?;

        // Mixing log probs: log(probs[k]).
        let mix_probs = self.mixing.probs().data_vec()?;
        let mix_log: Vec<T> = mix_probs.iter().map(|&p| p.ln()).collect();

        // logsumexp along the trailing K dim.
        let mut result = Vec::with_capacity(v_numel);
        for v_idx in 0..v_numel {
            let base = v_idx * self.num_components;
            // Compute log sum exp over k of (mix_log[k] + comp_lp[base+k]).
            let mut max_val = T::neg_infinity();
            for k in 0..self.num_components {
                let lp = mix_log[k] + comp_lp_data[base + k];
                if lp > max_val {
                    max_val = lp;
                }
            }
            let mut sum_exp = <T as num_traits::Zero>::zero();
            for k in 0..self.num_components {
                let lp = mix_log[k] + comp_lp_data[base + k];
                sum_exp += (lp - max_val).exp();
            }
            result.push(max_val + sum_exp.ln());
        }

        let out = Tensor::from_storage(TensorStorage::cpu(result), v_shape, false)?;
        if v_dev.is_cuda() {
            out.to(v_dev)
        } else {
            Ok(out)
        }
    }

    fn entropy(&self) -> FerrotorchResult<Tensor<T>> {
        // Closed-form entropy is not generally tractable for mixtures.
        // PyTorch's MixtureSameFamily also does not implement entropy.
        Err(FerrotorchError::InvalidArgument {
            message: "MixtureSameFamily: entropy has no closed form for general mixtures".into(),
        })
    }

    fn mean(&self) -> FerrotorchResult<Tensor<T>> {
        // Reference: torch.distributions.MixtureSameFamily.mean
        // (`torch/distributions/mixture_same_family.py:155-162`):
        //   mean = sum_k mix_probs[k] * components_mean[k]
        // Components are stored as a single distribution whose batch shape's
        // rightmost dim is the component index (size K). We weight the
        // per-component means by the mixing probabilities and sum.
        let comp_mean = self.components.mean()?;
        let comp_data = comp_mean.data_vec()?;
        let mix_probs = self.mixing.probs().data_vec()?;
        let k = self.num_components;
        if comp_data.len() % k != 0 {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "MixtureSameFamily: components.mean() numel {} not divisible by K={}",
                    comp_data.len(),
                    k
                ),
            });
        }
        let outer = comp_data.len() / k;
        let zero = <T as num_traits::Zero>::zero();
        let mut out = Vec::with_capacity(outer);
        for i in 0..outer {
            let mut acc = zero;
            for j in 0..k {
                acc += mix_probs[j] * comp_data[i * k + j];
            }
            out.push(acc);
        }
        let mut out_shape = comp_mean.shape().to_vec();
        out_shape.pop();
        Tensor::from_storage(TensorStorage::cpu(out), out_shape, false)
    }

    fn variance(&self) -> FerrotorchResult<Tensor<T>> {
        // Reference: torch.distributions.MixtureSameFamily.variance
        // (`torch/distributions/mixture_same_family.py:164-189`):
        //   Var(X) = E[Var(X|K)] + Var(E[X|K])
        //          = sum_k pi_k * Var_k + sum_k pi_k * (mu_k - mu)^2
        // where mu = sum_k pi_k * mu_k. We need both components.mean() and
        // components.variance() to be available.
        let comp_mean = self.components.mean()?;
        let comp_var = self.components.variance()?;
        let mean_data = comp_mean.data_vec()?;
        let var_data = comp_var.data_vec()?;
        let mix_probs = self.mixing.probs().data_vec()?;
        let k = self.num_components;
        let outer = mean_data.len() / k;
        let zero = <T as num_traits::Zero>::zero();

        let mut out = Vec::with_capacity(outer);
        for i in 0..outer {
            // overall mean for this outer slot
            let mut mu = zero;
            for j in 0..k {
                mu += mix_probs[j] * mean_data[i * k + j];
            }
            // law of total variance
            let mut acc = zero;
            for j in 0..k {
                let diff = mean_data[i * k + j] - mu;
                acc += mix_probs[j] * (var_data[i * k + j] + diff * diff);
            }
            out.push(acc);
        }
        let mut out_shape = comp_mean.shape().to_vec();
        out_shape.pop();
        Tensor::from_storage(TensorStorage::cpu(out), out_shape, false)
    }

    fn cdf(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Reference: torch.distributions.MixtureSameFamily.cdf
        // (`torch/distributions/mixture_same_family.py:191-201`):
        //   cdf(x) = sum_k mix_probs[k] * components.cdf(x)[k]
        // We tile `value` to [..., K] and call components.cdf, then weight
        // and sum over the trailing K dim.
        let v_shape = value.shape().to_vec();
        let v_data = value.data_vec()?;
        let k = self.num_components;
        let mut tiled = Vec::with_capacity(v_data.len() * k);
        for &v in v_data.iter() {
            for _ in 0..k {
                tiled.push(v);
            }
        }
        let mut tiled_shape = v_shape.clone();
        tiled_shape.push(k);
        let v_dev = value.device();
        let value_tiled = {
            let t = Tensor::from_storage(TensorStorage::cpu(tiled), tiled_shape, false)?;
            if v_dev.is_cuda() { t.to(v_dev)? } else { t }
        };

        let comp_cdf = self.components.cdf(&value_tiled)?;
        let comp_data = comp_cdf.data_vec()?;
        let mix_probs = self.mixing.probs().data_vec()?;
        let outer = v_data.len();
        let zero = <T as num_traits::Zero>::zero();
        let mut out = Vec::with_capacity(outer);
        for i in 0..outer {
            let mut acc = zero;
            for j in 0..k {
                acc += mix_probs[j] * comp_data[i * k + j];
            }
            out.push(acc);
        }
        let t = Tensor::from_storage(TensorStorage::cpu(out), v_shape, false)?;
        if v_dev.is_cuda() { t.to(v_dev) } else { Ok(t) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Normal;

    fn cpu_tensor(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
    }

    #[test]
    fn test_mixture_basic_log_prob() {
        // Two equal-weight Normals: N(-1, 1) and N(1, 1).
        // log_prob at x=0 should be log(0.5*N(0;-1,1) + 0.5*N(0;1,1)).
        // N(0;-1,1) = N(0;1,1) by symmetry, so the answer is just
        // log(N(0;0,1)) shifted: actually the value is
        //    log( 0.5 * (1/sqrt(2pi)) * exp(-0.5) + 0.5 * (1/sqrt(2pi)) * exp(-0.5) )
        //  = log( (1/sqrt(2pi)) * exp(-0.5) )
        //  = -0.5 * log(2pi) - 0.5 ≈ -1.4189
        let probs = cpu_tensor(&[0.5, 0.5], &[2]);
        let mixing = Categorical::new(probs).unwrap();

        let loc = cpu_tensor(&[-1.0, 1.0], &[2]);
        let scale = cpu_tensor(&[1.0, 1.0], &[2]);
        let components = Normal::new(loc, scale).unwrap();

        let m = MixtureSameFamily::new(mixing, components).unwrap();
        let value = cpu_tensor(&[0.0], &[1]);
        let lp = m.log_prob(&value).unwrap();
        assert_eq!(lp.shape(), &[1]);
        let val = lp.data().unwrap()[0];
        let expected = -0.5_f32 * (2.0 * std::f32::consts::PI).ln() - 0.5;
        assert!(
            (val - expected).abs() < 1e-4,
            "expected {expected}, got {val}"
        );
    }

    #[test]
    fn test_mixture_rsample_errors() {
        let probs = cpu_tensor(&[0.5, 0.5], &[2]);
        let mixing = Categorical::new(probs).unwrap();
        let loc = cpu_tensor(&[0.0, 1.0], &[2]);
        let scale = cpu_tensor(&[1.0, 1.0], &[2]);
        let components = Normal::new(loc, scale).unwrap();
        let m = MixtureSameFamily::new(mixing, components).unwrap();
        assert!(m.rsample(&[10]).is_err());
    }

    #[test]
    fn test_mixture_entropy_errors() {
        let probs = cpu_tensor(&[0.5, 0.5], &[2]);
        let mixing = Categorical::new(probs).unwrap();
        let loc = cpu_tensor(&[0.0, 1.0], &[2]);
        let scale = cpu_tensor(&[1.0, 1.0], &[2]);
        let components = Normal::new(loc, scale).unwrap();
        let m = MixtureSameFamily::new(mixing, components).unwrap();
        assert!(m.entropy().is_err());
    }

    #[test]
    fn test_mixture_log_prob_weighted() {
        // Asymmetric weights: 0.9 for component 0, 0.1 for component 1.
        // At x = -1, component 0's log_prob is high; component 1's is low.
        // The mixture log_prob should be close to log(0.9) + N(-1;-1,1).log_prob.
        let probs = cpu_tensor(&[0.9, 0.1], &[2]);
        let mixing = Categorical::new(probs).unwrap();
        let loc = cpu_tensor(&[-1.0, 5.0], &[2]);
        let scale = cpu_tensor(&[1.0, 1.0], &[2]);
        let components = Normal::new(loc, scale).unwrap();
        let m = MixtureSameFamily::new(mixing, components).unwrap();
        let value = cpu_tensor(&[-1.0], &[1]);
        let lp = m.log_prob(&value).unwrap();

        // Component 0 dominates at x=-1.
        // log(0.9) + N(-1;-1,1).log_prob = log(0.9) - 0.5*log(2pi) ≈
        //   -0.1054 - 0.9189 ≈ -1.0243.
        // Component 1 contributes negligibly: log(0.1) - 0.5*log(2pi) - 18 ≈ -21.22
        // logsumexp([-1.0243, -21.22]) ≈ -1.0243.
        let val = lp.data().unwrap()[0];
        assert!((val + 1.0243).abs() < 0.01, "expected ≈ -1.0243, got {val}");
    }

    #[test]
    fn test_mixture_mean_weighted_sum() {
        // Two Normals: N(-1, 1) and N(3, 1), mixing 0.25/0.75.
        // mean = 0.25 * -1 + 0.75 * 3 = -0.25 + 2.25 = 2.0
        let probs = cpu_tensor(&[0.25, 0.75], &[2]);
        let mixing = Categorical::new(probs).unwrap();
        let loc = cpu_tensor(&[-1.0, 3.0], &[2]);
        let scale = cpu_tensor(&[1.0, 1.0], &[2]);
        let components = Normal::new(loc, scale).unwrap();
        let m = MixtureSameFamily::new(mixing, components).unwrap();

        let mean = m.mean().unwrap();
        let val = mean.data_vec().unwrap()[0];
        assert!((val - 2.0).abs() < 1e-5, "expected 2.0, got {val}");
    }

    #[test]
    fn test_mixture_variance_total_variance_law() {
        // Two Normals: N(0, 1) and N(2, 1), 50/50.
        // mean = 1.0
        // Var = 0.5*(1 + (0-1)^2) + 0.5*(1 + (2-1)^2) = 0.5*2 + 0.5*2 = 2.0
        let probs = cpu_tensor(&[0.5, 0.5], &[2]);
        let mixing = Categorical::new(probs).unwrap();
        let loc = cpu_tensor(&[0.0, 2.0], &[2]);
        let scale = cpu_tensor(&[1.0, 1.0], &[2]);
        let components = Normal::new(loc, scale).unwrap();
        let m = MixtureSameFamily::new(mixing, components).unwrap();

        let var = m.variance().unwrap();
        let val = var.data_vec().unwrap()[0];
        assert!((val - 2.0).abs() < 1e-5, "expected 2.0, got {val}");
    }

    #[test]
    fn test_mixture_cdf_weighted_sum() {
        // Two Normals: N(0,1) and N(0,1) — identical, weights 0.5/0.5.
        // CDF at x=0 is 0.5 for each, so weighted sum is 0.5.
        let probs = cpu_tensor(&[0.5, 0.5], &[2]);
        let mixing = Categorical::new(probs).unwrap();
        let loc = cpu_tensor(&[0.0, 0.0], &[2]);
        let scale = cpu_tensor(&[1.0, 1.0], &[2]);
        let components = Normal::new(loc, scale).unwrap();
        let m = MixtureSameFamily::new(mixing, components).unwrap();
        let value = cpu_tensor(&[0.0], &[1]);
        let c = m.cdf(&value).unwrap();
        let v = c.data_vec().unwrap()[0];
        assert!((v - 0.5).abs() < 1e-4, "expected 0.5, got {v}");
    }

    #[test]
    fn test_mixture_sample_shape() {
        let probs = cpu_tensor(&[0.5, 0.5], &[2]);
        let mixing = Categorical::new(probs).unwrap();
        let loc = cpu_tensor(&[-1.0, 1.0], &[2]);
        let scale = cpu_tensor(&[0.5, 0.5], &[2]);
        let components = Normal::new(loc, scale).unwrap();
        let m = MixtureSameFamily::new(mixing, components).unwrap();
        let s = m.sample(&[100]).unwrap();
        assert_eq!(s.shape(), &[100]);
    }
}
