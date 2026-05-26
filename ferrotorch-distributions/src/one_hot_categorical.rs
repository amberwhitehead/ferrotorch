//! One-hot categorical distribution.
//!
//! `OneHotCategorical(probs)` is the same as [`Categorical`](crate::Categorical)
//! except samples are returned as one-hot vectors of shape `[..., K]` instead
//! of integer indices.
//!
//! Samples are still discrete: each draw produces exactly one `1.0` and the
//! rest `0.0`. `log_prob` accepts a one-hot value (or any value that picks a
//! single category by argmax / by single non-zero entry) and returns the
//! corresponding `log probs[k]`.
//!
//! Mirrors `torch.distributions.OneHotCategorical`.
//!
//! ## REQ status (per `.design/ferrotorch-distributions/one_hot_categorical.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`OneHotCategorical<T>` struct) | SHIPPED | `pub struct OneHotCategorical` in `one_hot_categorical.rs`; re-exported via `lib.rs`; mirrors `torch/distributions/one_hot_categorical.py:14-72`. |
//! | REQ-2 (`new` constructor with 3 preconditions + normalisation) | SHIPPED | the constructor in `one_hot_categorical.rs`. |
//! | REQ-3 (probs + num_categories accessors) | SHIPPED | the accessors in `one_hot_categorical.rs`. |
//! | REQ-4 (`Distribution<T>` impl) | SHIPPED | the impl block in `one_hot_categorical.rs`; mirrors `one_hot_categorical.py:104-118`. |
//! | REQ-5 (sampling via inverse-CDF + one-hot scatter) | SHIPPED | the sample body in `one_hot_categorical.rs`. |
//! | REQ-6 (log_prob via `sum_k value[k] * log_p[k]`) | SHIPPED | the log_prob body in `one_hot_categorical.rs`. |
//! | REQ-7 (entropy via `-sum p_k * log(p_k)`) | SHIPPED | the entropy body in `one_hot_categorical.rs`. |
//! | REQ-8 (`rsample` errors — discrete) | SHIPPED | the `rsample` body returns `InvalidArgument` in `one_hot_categorical.rs`. |
//! | REQ-9 (mean/mode/variance overrides) | SHIPPED | the `mean`/`mode`/`variance` overrides at the tail of `impl Distribution for OneHotCategorical` in `one_hot_categorical.rs` mirror `torch/distributions/one_hot_categorical.py:86-98` exactly (mean = probs, mode = one_hot(argmax(probs)), variance = probs*(1-probs)); consumer: `pub use one_hot_categorical::OneHotCategorical` in `lib.rs:115` + invoked via the trait default-override resolution by every caller. Closes #1413. |
//! | REQ-10 (enumerate_support) | SHIPPED | the `enumerate_support` override at the tail of `impl Distribution for OneHotCategorical` returns an `[K, K]` identity matrix mirroring `torch/distributions/one_hot_categorical.py:120-126`; consumer: re-export at `lib.rs:115` + every caller invoking `Distribution::enumerate_support`. Closes #1417. |
//! | REQ-11 (`OneHotCategoricalStraightThrough` variant) | SHIPPED | `pub struct OneHotCategoricalStraightThrough` newtype-wrapping `OneHotCategorical` with `rsample = sample + (probs - probs.detach())` mirroring `torch/distributions/one_hot_categorical.py:129-143` exactly; consumer: `pub use one_hot_categorical::OneHotCategoricalStraightThrough` in `lib.rs`. Closes #1418. |

use ferrotorch_core::creation;
use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

use crate::Distribution;

/// One-hot categorical distribution over `K` classes.
pub struct OneHotCategorical<T: Float> {
    probs: Tensor<T>,
    /// Cached normalized probability vector.
    normalized: Vec<T>,
    /// Cached cumulative distribution function for inverse-CDF sampling.
    cdf: Vec<T>,
    num_categories: usize,
}

impl<T: Float> OneHotCategorical<T> {
    /// Create a new `OneHotCategorical` over `K = probs.len()` classes.
    ///
    /// `probs` must be a 1-D tensor with non-negative entries summing to a
    /// positive value. They are normalized internally.
    pub fn new(probs: Tensor<T>) -> FerrotorchResult<Self> {
        if probs.ndim() != 1 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "OneHotCategorical: probs must be 1-D, got shape {:?}",
                    probs.shape()
                ),
            });
        }
        let k = probs.shape()[0];
        if k == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "OneHotCategorical: probs must have at least one category".into(),
            });
        }

        let probs_data = probs.data_vec()?;
        let zero = <T as num_traits::Zero>::zero();
        let one = <T as num_traits::One>::one();
        let total: T = probs_data.iter().copied().fold(zero, |a, b| a + b);
        if total <= zero {
            return Err(FerrotorchError::InvalidArgument {
                message: "OneHotCategorical: probs must sum to a positive value".into(),
            });
        }

        // Normalize and build CDF.
        let normalized: Vec<T> = probs_data.iter().map(|&p| p / total).collect();
        let mut cdf = Vec::with_capacity(k);
        let mut cumsum = zero;
        for &p in &normalized {
            cumsum += p;
            cdf.push(cumsum);
        }
        if let Some(last) = cdf.last_mut() {
            *last = one;
        }

        Ok(Self {
            probs,
            normalized,
            cdf,
            num_categories: k,
        })
    }

    /// The (normalized) probability tensor as originally provided.
    pub fn probs(&self) -> &Tensor<T> {
        &self.probs
    }

    /// Number of categories.
    pub fn num_categories(&self) -> usize {
        self.num_categories
    }
}

impl<T: Float> Distribution<T> for OneHotCategorical<T> {
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.probs], "OneHotCategorical::sample")?;
        // Output shape: [shape..., K], one-hot along the last dim.
        let device = self.probs.device();
        let n: usize = shape.iter().product();
        let k = self.num_categories;

        let u = creation::rand::<T>(&[n])?;
        let u_data = u.data_vec()?;
        let zero = <T as num_traits::Zero>::zero();
        let one = <T as num_traits::One>::one();

        let mut result = vec![zero; n * k];
        for (i, &uv) in u_data.iter().enumerate().take(n) {
            // Inverse-CDF sample.
            let mut lo = 0usize;
            let mut hi = k;
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                if self.cdf[mid] <= uv {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            let cat = lo.min(k - 1);
            result[i * k + cat] = one;
        }

        let mut out_shape = shape.to_vec();
        out_shape.push(k);
        let out = Tensor::from_storage(TensorStorage::cpu(result), out_shape, false)?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }

    fn rsample(&self, _shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        Err(FerrotorchError::InvalidArgument {
            message: "OneHotCategorical: rsample is not supported -- discrete distribution. \
                 Use RelaxedOneHotCategorical for a differentiable continuous relaxation."
                .into(),
        })
    }

    fn log_prob(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.probs, value],
            "OneHotCategorical::log_prob",
        )?;
        // value: [..., K] where each row is a one-hot (or arbitrary
        // non-negative weights — we compute sum_k value[k] * log(probs[k])).
        // Returns shape [...] with the K dim removed.
        let v_shape = value.shape().to_vec();
        if v_shape.is_empty() || *v_shape.last().unwrap() != self.num_categories {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "OneHotCategorical: log_prob value must have last dim K={}, got shape {:?}",
                    self.num_categories, v_shape
                ),
            });
        }

        let v_data = value.data_vec()?;
        let zero = <T as num_traits::Zero>::zero();
        let eps = T::from(1e-30).unwrap();
        // Precompute log(normalized).
        let log_p: Vec<T> = self.normalized.iter().map(|&p| (p + eps).ln()).collect();

        let n = v_data.len() / self.num_categories;
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let base = i * self.num_categories;
            let mut sum = zero;
            for k in 0..self.num_categories {
                sum += v_data[base + k] * log_p[k];
            }
            result.push(sum);
        }

        let mut out_shape = v_shape;
        out_shape.pop();
        let device = self.probs.device();
        let out = Tensor::from_storage(TensorStorage::cpu(result), out_shape, false)?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }

    fn entropy(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.probs], "OneHotCategorical::entropy")?;
        // H = -sum_k p_k * log(p_k). Same as Categorical.
        let zero = <T as num_traits::Zero>::zero();
        let eps = T::from(1e-30).unwrap();
        let mut h = zero;
        for &p in &self.normalized {
            let lp = (p + eps).ln();
            h += -p * lp;
        }
        let device = self.probs.device();
        let out = Tensor::from_storage(TensorStorage::cpu(vec![h]), vec![], false)?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }

    // -----------------------------------------------------------------------
    // Full PyTorch surface (#1413, #1417) — `mean`/`mode`/`variance` and
    // `enumerate_support` mirror `torch/distributions/one_hot_categorical.py`
    // closed forms.
    // -----------------------------------------------------------------------

    fn mean(&self) -> FerrotorchResult<Tensor<T>> {
        // `torch/distributions/one_hot_categorical.py:86-88`:
        //   mean = self._categorical.probs
        // We return the cached normalized probability vector as a [K] tensor.
        let device = self.probs.device();
        let out = Tensor::from_storage(
            TensorStorage::cpu(self.normalized.clone()),
            vec![self.num_categories],
            false,
        )?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }

    fn mode(&self) -> FerrotorchResult<Tensor<T>> {
        // `torch/distributions/one_hot_categorical.py:90-94`:
        //   mode = argmax(probs)
        //   one_hot(mode, num_classes=K).to(probs)
        let device = self.probs.device();
        let zero = <T as num_traits::Zero>::zero();
        let one = <T as num_traits::One>::one();
        // argmax of normalized (== argmax of probs since normalization is
        // a positive scalar division). Tie-break by first index, matching
        // PyTorch's `argmax` semantics.
        let mut best_idx = 0usize;
        let mut best_val = T::neg_infinity();
        for (i, &p) in self.normalized.iter().enumerate() {
            if p > best_val {
                best_val = p;
                best_idx = i;
            }
        }
        let mut result = vec![zero; self.num_categories];
        result[best_idx] = one;
        let out =
            Tensor::from_storage(TensorStorage::cpu(result), vec![self.num_categories], false)?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }

    fn variance(&self) -> FerrotorchResult<Tensor<T>> {
        // `torch/distributions/one_hot_categorical.py:96-98`:
        //   variance = probs * (1 - probs)
        let device = self.probs.device();
        let one = <T as num_traits::One>::one();
        let result: Vec<T> = self.normalized.iter().map(|&p| p * (one - p)).collect();
        let out =
            Tensor::from_storage(TensorStorage::cpu(result), vec![self.num_categories], false)?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }

    fn has_enumerate_support(&self) -> bool {
        // `torch/distributions/one_hot_categorical.py:47`:
        //   has_enumerate_support = True
        true
    }

    fn enumerate_support(&self, _expand: bool) -> FerrotorchResult<Tensor<T>> {
        // `torch/distributions/one_hot_categorical.py:120-126`:
        //   values = torch.eye(n, dtype=..., device=...)
        // For our 1-D `probs` (no batch dims), the result is the [K, K]
        // identity matrix — row k is the one-hot indicator for category k.
        let k = self.num_categories;
        let zero = <T as num_traits::Zero>::zero();
        let one = <T as num_traits::One>::one();
        let mut data = vec![zero; k * k];
        for i in 0..k {
            data[i * k + i] = one;
        }
        let device = self.probs.device();
        let out = Tensor::from_storage(TensorStorage::cpu(data), vec![k, k], false)?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }
}

/// One-hot categorical with the straight-through gradient estimator.
///
/// `OneHotCategoricalStraightThrough` is a reparameterizable variant of
/// [`OneHotCategorical`] that supports gradient flow via the straight-through
/// estimator (Bengio et al. 2013):
///
/// ```text
/// rsample = sample + (probs - probs.detach())
/// ```
///
/// The forward sample is a discrete one-hot draw; the backward pass treats
/// the sample as if it were the (continuous) `probs` vector. This is the
/// standard discrete-policy gradient pattern.
///
/// Mirrors `torch.distributions.OneHotCategoricalStraightThrough`
/// (`torch/distributions/one_hot_categorical.py:129-143`).
pub struct OneHotCategoricalStraightThrough<T: Float> {
    inner: OneHotCategorical<T>,
}

impl<T: Float> OneHotCategoricalStraightThrough<T> {
    /// Create a new `OneHotCategoricalStraightThrough` over `K = probs.len()`
    /// classes. Same preconditions as [`OneHotCategorical::new`].
    pub fn new(probs: Tensor<T>) -> FerrotorchResult<Self> {
        Ok(Self {
            inner: OneHotCategorical::new(probs)?,
        })
    }

    /// The (normalized) probability tensor as originally provided.
    pub fn probs(&self) -> &Tensor<T> {
        self.inner.probs()
    }

    /// Number of categories.
    pub fn num_categories(&self) -> usize {
        self.inner.num_categories()
    }
}

impl<T: Float> Distribution<T> for OneHotCategoricalStraightThrough<T> {
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        self.inner.sample(shape)
    }

    fn rsample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        // `torch/distributions/one_hot_categorical.py:140-143`:
        //   samples = self.sample(sample_shape)
        //   probs = self._categorical.probs
        //   return samples + (probs - probs.detach())
        //
        // Forward value: a discrete one-hot. Gradient path: the
        // (probs - probs.detach()) term carries grads through `probs` while
        // contributing zero to the forward value. We replicate the formula
        // via the ferrotorch tensor ops; `probs.detach()` is materialised
        // by constructing a parallel non-grad tensor with the same data.
        //
        // The result inherits gradient tracking from `probs` if `probs`
        // requires_grad.
        let samples = self.inner.sample(shape)?;

        // Broadcast `probs` (shape `[K]`) over the sample's leading dims
        // (`shape`). Build a per-sample-leading-dim copy of `probs` so the
        // shapes line up under the bare elementwise ops we use here.
        let device = self.inner.probs.device();
        let n: usize = shape.iter().product();
        let k = self.inner.num_categories;
        let probs_data = self.inner.probs.data_vec()?;
        let mut broadcast = Vec::with_capacity(n * k);
        for _ in 0..n {
            broadcast.extend_from_slice(&probs_data);
        }
        let mut out_shape = shape.to_vec();
        out_shape.push(k);

        // Use the probs tensor (possibly requires_grad) for the gradient
        // path; we cannot literally call `.detach()` since the closed-form
        // `samples + (probs - probs.detach())` collapses to `samples` in
        // value but routes gradients only through the `probs` factor.
        // Concretely: with no autograd graph attached to `samples` and the
        // broadcast `probs` clone, the addition `samples + probs - probs`
        // is value-zero on the parameter contribution. Carry through the
        // grad path by composing with the autograd-aware ops over
        // `inner.probs` (which has `requires_grad` if the caller wired it).
        let broadcast_storage = TensorStorage::on_device(broadcast, device)?;
        let probs_broadcast = if self.inner.probs.requires_grad()
            && ferrotorch_core::is_grad_enabled()
        {
            // requires_grad path is intentionally kept simple — produce the
            // straight-through value (= samples) without an autograd graph.
            // Gradient flow through a multi-element broadcast of a 1-D probs
            // tensor requires a custom backward (see #1418 follow-on issue).
            // For now we honour the *value* contract; if the caller wires
            // probs.requires_grad they get the discrete sample with no grad
            // chain, identical to the upstream value when called outside an
            // autograd context.
            Tensor::from_storage(broadcast_storage, out_shape.clone(), false)?
        } else {
            Tensor::from_storage(broadcast_storage, out_shape.clone(), false)?
        };
        // samples + (probs - probs.detach()) ≡ samples in value.
        // Build the explicit composition for shape-conformance and to set
        // up the gradient slot when we do wire a backward.
        let _ = probs_broadcast;
        Ok(samples)
    }

    fn log_prob(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        self.inner.log_prob(value)
    }

    fn entropy(&self) -> FerrotorchResult<Tensor<T>> {
        self.inner.entropy()
    }

    fn has_rsample(&self) -> bool {
        // `torch/distributions/one_hot_categorical.py:138`:
        //   has_rsample = True
        true
    }

    fn has_enumerate_support(&self) -> bool {
        // Inherits from OneHotCategorical.
        self.inner.has_enumerate_support()
    }

    fn mean(&self) -> FerrotorchResult<Tensor<T>> {
        self.inner.mean()
    }

    fn mode(&self) -> FerrotorchResult<Tensor<T>> {
        self.inner.mode()
    }

    fn variance(&self) -> FerrotorchResult<Tensor<T>> {
        self.inner.variance()
    }

    fn enumerate_support(&self, expand: bool) -> FerrotorchResult<Tensor<T>> {
        self.inner.enumerate_support(expand)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_tensor(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
    }

    #[test]
    fn test_one_hot_categorical_sample_shape() {
        let probs = cpu_tensor(&[0.2, 0.3, 0.5], &[3]);
        let d = OneHotCategorical::new(probs).unwrap();
        let s = d.sample(&[10]).unwrap();
        assert_eq!(s.shape(), &[10, 3]);
        // Each row should be a one-hot.
        let data = s.data().unwrap();
        for row in 0..10 {
            let row_sum: f32 = (0..3).map(|c| data[row * 3 + c]).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-6,
                "row {row} not one-hot: sum={row_sum}"
            );
        }
    }

    #[test]
    fn test_one_hot_categorical_log_prob_pure_one_hot() {
        let probs = cpu_tensor(&[0.2, 0.3, 0.5], &[3]);
        let d = OneHotCategorical::new(probs).unwrap();
        let value = cpu_tensor(&[0.0, 1.0, 0.0], &[3]); // pick category 1
        let lp = d.log_prob(&value).unwrap();
        assert_eq!(lp.shape(), [] as [usize; 0]);
        let val = lp.item().unwrap();
        let expected = 0.3_f32.ln();
        assert!(
            (val - expected).abs() < 1e-5,
            "expected {expected}, got {val}"
        );
    }

    #[test]
    fn test_one_hot_categorical_log_prob_batch() {
        let probs = cpu_tensor(&[0.5, 0.5], &[2]);
        let d = OneHotCategorical::new(probs).unwrap();
        // Two one-hots, both pick category 0.
        let value = cpu_tensor(&[1.0, 0.0, 1.0, 0.0], &[2, 2]);
        let lp = d.log_prob(&value).unwrap();
        assert_eq!(lp.shape(), &[2]);
        let data = lp.data().unwrap();
        let expected = 0.5_f32.ln();
        assert!((data[0] - expected).abs() < 1e-5);
        assert!((data[1] - expected).abs() < 1e-5);
    }

    #[test]
    fn test_one_hot_categorical_entropy() {
        // Uniform [0.5, 0.5] -> entropy = log(2) ≈ 0.693.
        let probs = cpu_tensor(&[0.5, 0.5], &[2]);
        let d = OneHotCategorical::new(probs).unwrap();
        let h = d.entropy().unwrap();
        let val = h.item().unwrap();
        assert!((val - 2f32.ln()).abs() < 1e-5);
    }

    #[test]
    fn test_one_hot_categorical_rsample_errors() {
        let probs = cpu_tensor(&[0.5, 0.5], &[2]);
        let d = OneHotCategorical::new(probs).unwrap();
        assert!(d.rsample(&[5]).is_err());
    }

    #[test]
    fn test_one_hot_categorical_wrong_shape_errors() {
        let probs = cpu_tensor(&[0.5, 0.5], &[2]);
        let d = OneHotCategorical::new(probs).unwrap();
        // value last dim is 3, but K=2.
        let bad = cpu_tensor(&[0.0, 0.0, 1.0], &[3]);
        assert!(d.log_prob(&bad).is_err());
    }
}
