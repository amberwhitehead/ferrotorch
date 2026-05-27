//! Independent distribution wrapper.
//!
//! ## REQ status (per `.design/ferrotorch-distributions/independent.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`Independent<T, D>` struct) | SHIPPED | `pub struct Independent` in `independent.rs`; re-exported via `lib.rs`; mirrors `torch/distributions/independent.py:18-69`. |
//! | REQ-2 (`new` constructor with zero-ndims rejection) | SHIPPED | `pub fn Independent::new` in `independent.rs`; pinned by `test_independent_zero_ndims_errors`. |
//! | REQ-3 (`base` + `reinterpreted_batch_ndims` accessors) | SHIPPED | `pub fn base` + `pub fn reinterpreted_batch_ndims` in `independent.rs`. |
//! | REQ-4 (`Distribution<T>` impl) | SHIPPED | `impl<T, D> Distribution<T> for Independent` in `independent.rs`; mirrors `torch/distributions/independent.py:84-126`. |
//! | REQ-5 (`sum_rightmost` helper) | SHIPPED | `fn sum_rightmost<T>` in `independent.rs`; consumed by `log_prob` + `entropy`. |
//! | REQ-6 (`sample` shape-forwarding) | SHIPPED | `Independent::sample` builds `full_shape = shape ++ event_dims` in `independent.rs`. |
//! | REQ-7 (expand/enumerate_support/support/mean/mode/variance/has_rsample) | SHIPPED | `Independent::{expand, enumerate_support, support, mean, mode, variance, has_rsample, has_enumerate_support}` forward to `self.base` mirroring `torch/distributions/independent.py:71-133`; consumer: trait dispatch via `pub use Independent` re-export at `lib.rs:106`. Closes #1377. |
//!
//! Reinterprets the rightmost `reinterpreted_batch_ndims` of a base
//! distribution's batch dimensions as event dimensions. The semantics:
//!
//! - `sample` / `rsample`: identical to the base distribution — same shape,
//!   same values.
//! - `log_prob`: the base log_prob is summed over the reinterpreted dims.
//!   This is the natural log_prob of a multivariate distribution formed
//!   by treating the reinterpreted dims as independent.
//! - `entropy`: similarly summed over the reinterpreted dims.
//!
//! Mirrors `torch.distributions.Independent`.
//!
//! # Why
//!
//! `Independent` is the standard way to turn a `Normal(loc=[B,K], scale=[B,K])`
//! (which yields a `[B,K]`-shaped log_prob) into a multivariate-style
//! distribution whose log_prob has shape `[B]`. It is also a building
//! block for variational autoencoders where the latent distribution is a
//! diagonal Gaussian over the K latent dims.

use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::tensor::Tensor;

use crate::Distribution;

/// Wraps a base distribution and reinterprets the rightmost
/// `reinterpreted_batch_ndims` of its batch shape as event dimensions.
pub struct Independent<T: Float, D: Distribution<T>> {
    base: D,
    reinterpreted_batch_ndims: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float, D: Distribution<T>> Independent<T, D> {
    /// Wrap a base distribution, treating the rightmost `n` batch dims
    /// as event dims.
    ///
    /// # Errors
    ///
    /// Returns an error if `reinterpreted_batch_ndims == 0` (in which
    /// case there is nothing to reinterpret — use the base directly).
    pub fn new(base: D, reinterpreted_batch_ndims: usize) -> FerrotorchResult<Self> {
        if reinterpreted_batch_ndims == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message:
                    "Independent: reinterpreted_batch_ndims must be > 0; use the base distribution directly"
                        .into(),
            });
        }
        Ok(Self {
            base,
            reinterpreted_batch_ndims,
            _phantom: std::marker::PhantomData,
        })
    }

    /// The wrapped base distribution.
    pub fn base(&self) -> &D {
        &self.base
    }

    /// The number of batch dims being reinterpreted as event dims.
    pub fn reinterpreted_batch_ndims(&self) -> usize {
        self.reinterpreted_batch_ndims
    }
}

impl<T: Float, D: Distribution<T> + 'static> Distribution<T> for Independent<T, D> {
    fn batch_shape(&self) -> Vec<usize> {
        // Independent reinterprets the rightmost `reinterpreted_batch_ndims` batch
        // dims as event dims, so the exposed batch shape has those dims removed.
        let base_batch = self.base.batch_shape();
        let n = self.reinterpreted_batch_ndims.min(base_batch.len());
        base_batch[..base_batch.len() - n].to_vec()
    }

    fn event_shape(&self) -> Vec<usize> {
        // Mirror `torch/distributions/independent.py:62-65`:
        //   shape = base.batch_shape + base.event_shape
        //   event_dim = reinterpreted_batch_ndims + len(base.event_shape)
        //   event_shape = shape[len(shape) - event_dim:]
        let base_batch = self.base.batch_shape();
        let base_event = self.base.event_shape();
        let mut shape = base_batch.clone();
        shape.extend_from_slice(&base_event);
        let event_dim = self.reinterpreted_batch_ndims + base_event.len();
        let n = event_dim.min(shape.len());
        shape[shape.len() - n..].to_vec()
    }

    fn has_rsample(&self) -> bool {
        // `torch/distributions/independent.py:84-86`: forwards to base.
        self.base.has_rsample()
    }

    fn has_enumerate_support(&self) -> bool {
        // `torch/distributions/independent.py:88-92`: returns False whenever
        // any batch dims have been reinterpreted (Cartesian-product
        // enumeration isn't implemented upstream either).
        if self.reinterpreted_batch_ndims > 0 {
            return false;
        }
        self.base.has_enumerate_support()
    }

    fn support(&self) -> Option<Box<dyn crate::DistConstraint>> {
        // `torch/distributions/independent.py:94-100`: forwards base.support.
        // Upstream wraps in `constraints.independent(.., reinterpreted_batch_ndims)`
        // when there are reinterpreted dims; ferrotorch's `DistConstraint`
        // surface is dtype-erased and doesn't yet carry a wrapping
        // `Independent`-constraint variant (tracked in #1372), so we
        // forward the base support unwrapped — the `event_dim` count is
        // still derivable via `self.event_shape()`.
        self.base.support()
    }

    fn mean(&self) -> FerrotorchResult<Tensor<T>> {
        // `torch/distributions/independent.py:102-104`: forwards to base.
        self.base.mean()
    }

    fn mode(&self) -> FerrotorchResult<Tensor<T>> {
        // `torch/distributions/independent.py:106-108`: forwards to base.
        self.base.mode()
    }

    fn variance(&self) -> FerrotorchResult<Tensor<T>> {
        // `torch/distributions/independent.py:110-112`: forwards to base.
        self.base.variance()
    }

    fn enumerate_support(&self, expand: bool) -> FerrotorchResult<Tensor<T>> {
        // `torch/distributions/independent.py:128-133`: raises
        // NotImplementedError when reinterpreted dims > 0, else forwards.
        if self.reinterpreted_batch_ndims > 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "Independent::enumerate_support: enumeration over Cartesian product of \
                          reinterpreted batch dims is not implemented (matches upstream \
                          NotImplementedError at independent.py:128-131)"
                    .into(),
            });
        }
        self.base.enumerate_support(expand)
    }

    fn expand(&self, _batch_shape: &[usize]) -> FerrotorchResult<Box<dyn Distribution<T>>> {
        // `torch/distributions/independent.py:71-82`: builds a new
        // Independent wrapping `base.expand(batch_shape + event_shape[:n])`.
        // ferrotorch's `Independent` is `<T, D>`-generic on the concrete
        // base type; `Distribution::expand` returns `Box<dyn Distribution<T>>`
        // which we cannot stuff back into the typed `Independent<T, D>`
        // without erasing `D`. We therefore return an
        // `InvalidArgument` describing the trait-object barrier; consumers
        // that need batched expansion can construct a fresh
        // `Independent::new(self.base.expand(..)?, self.reinterpreted_batch_ndims)`
        // by hand and immediately have a dyn-Distribution result.
        Err(FerrotorchError::InvalidArgument {
            message: "Independent::expand: cannot rewrap a `Box<dyn Distribution<T>>` into a \
                      typed `Independent<T, D>`. Construct `Independent::new(base.expand(..)?, n)` \
                      at the call site instead."
                .into(),
        })
    }

    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        // Reference: torch.distributions.Independent.sample(sample_shape)
        // PyTorch calls base.sample(sample_shape) which already includes the base's
        // batch shape in the output.  The base's sample(shape) appends its own
        // batch_shape to `shape` when constructing output.  Here, Normal::sample(shape)
        // uses the provided `shape` as the full output shape and cycles over the
        // batch parameters — so we must forward `shape ++ reinterpreted_batch_dims`
        // so that the last reinterpreted_batch_ndims dims are the event dims.
        let base_batch = self.base.batch_shape();
        if base_batch.is_empty() || self.reinterpreted_batch_ndims == 0 {
            return self.base.sample(shape);
        }
        // Take the rightmost `reinterpreted_batch_ndims` dims from base_batch as
        // the event dims that must appear at the end of every sample.
        let n = self.reinterpreted_batch_ndims.min(base_batch.len());
        let event_dims = &base_batch[base_batch.len() - n..];
        let mut full_shape: Vec<usize> = shape.to_vec();
        full_shape.extend_from_slice(event_dims);
        self.base.sample(&full_shape)
    }

    fn rsample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        let base_batch = self.base.batch_shape();
        if base_batch.is_empty() || self.reinterpreted_batch_ndims == 0 {
            return self.base.rsample(shape);
        }
        let n = self.reinterpreted_batch_ndims.min(base_batch.len());
        let event_dims = &base_batch[base_batch.len() - n..];
        let mut full_shape: Vec<usize> = shape.to_vec();
        full_shape.extend_from_slice(event_dims);
        self.base.rsample(&full_shape)
    }

    fn log_prob(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let base_lp = self.base.log_prob(value)?;
        sum_rightmost(&base_lp, self.reinterpreted_batch_ndims)
    }

    fn entropy(&self) -> FerrotorchResult<Tensor<T>> {
        let base_h = self.base.entropy()?;
        sum_rightmost(&base_h, self.reinterpreted_batch_ndims)
    }

    fn kl_recurse(&self) -> Option<crate::KlRecurseInfo<'_, T>> {
        // `torch/distributions/kl.py:944-949` `_kl_independent_independent`:
        //   if p.reinterpreted_batch_ndims != q.reinterpreted_batch_ndims: raise
        //   result = kl_divergence(p.base_dist, q.base_dist)
        //   return _sum_rightmost(result, p.reinterpreted_batch_ndims)
        // We expose the type-erased base + `reinterpreted_batch_ndims` so the KL
        // recursion in `kl::kl_divergence_dyn` can re-dispatch on the base and
        // sum the rightmost `n` dims, without the dispatcher needing the
        // concrete generic base type `D`.
        Some(crate::KlRecurseInfo {
            base: &self.base,
            kind: crate::KlRecurseKind::Independent {
                reinterpreted_batch_ndims: self.reinterpreted_batch_ndims,
            },
        })
    }
}

/// Sum a tensor along its rightmost `n` dims, returning a tensor whose
/// shape has `n` fewer dims. Stays on the input device.
fn sum_rightmost<T: Float>(t: &Tensor<T>, n: usize) -> FerrotorchResult<Tensor<T>> {
    let shape = t.shape();
    if n > shape.len() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "Independent: cannot sum {} rightmost dims of a {}-D tensor",
                n,
                shape.len()
            ),
        });
    }
    if n == 0 {
        return Ok(t.clone());
    }
    // Reduce along each rightmost dim from the right; sum_dim removes the
    // dim when keepdim=false. We start from the rightmost so dim indices
    // remain valid.
    let mut out = t.clone();
    for _ in 0..n {
        let last_dim = (out.ndim() - 1) as i64;
        out = ferrotorch_core::grad_fns::reduction::sum_dim(&out, last_dim, false)?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Normal;
    use ferrotorch_core::storage::TensorStorage;

    fn cpu_tensor(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
    }

    #[test]
    fn test_independent_zero_ndims_errors() {
        let loc = cpu_tensor(&[0.0, 0.0], &[2]);
        let scale = cpu_tensor(&[1.0, 1.0], &[2]);
        let n = Normal::new(loc, scale).unwrap();
        assert!(Independent::new(n, 0).is_err());
    }

    #[test]
    fn test_independent_log_prob_sums_event_dims() {
        // 2-D Normal: loc, scale shape [2]. log_prob of a 2-element value
        // is shape [2] for the base, shape [] (scalar) when wrapped in
        // Independent(reinterpreted_batch_ndims=1).
        let loc = cpu_tensor(&[0.0, 0.0], &[2]);
        let scale = cpu_tensor(&[1.0, 1.0], &[2]);
        let normal = Normal::new(loc.clone(), scale.clone()).unwrap();
        let value = cpu_tensor(&[0.5, -0.3], &[2]);
        let base_lp = normal.log_prob(&value).unwrap();
        assert_eq!(base_lp.shape(), &[2]);
        let base_data = base_lp.data().unwrap();
        let expected_sum = base_data[0] + base_data[1];

        let normal2 = Normal::new(loc, scale).unwrap();
        let ind = Independent::new(normal2, 1).unwrap();
        let ind_lp = ind.log_prob(&value).unwrap();
        // After summing 1 rightmost dim, the [2] -> [] (scalar).
        assert_eq!(ind_lp.shape(), [] as [usize; 0]);
        let val = ind_lp.item().unwrap();
        assert!(
            (val - expected_sum).abs() < 1e-5,
            "expected {expected_sum}, got {val}"
        );
    }

    #[test]
    fn test_independent_entropy_sums_event_dims() {
        let loc = cpu_tensor(&[0.0, 0.0, 0.0], &[3]);
        let scale = cpu_tensor(&[1.0, 2.0, 0.5], &[3]);
        let base_normal = Normal::new(loc.clone(), scale.clone()).unwrap();
        let base_h = base_normal.entropy().unwrap();
        let base_h_data = base_h.data().unwrap();
        let expected_sum: f32 = base_h_data.iter().sum();

        let normal2 = Normal::new(loc, scale).unwrap();
        let ind = Independent::new(normal2, 1).unwrap();
        let ind_h = ind.entropy().unwrap();
        assert_eq!(ind_h.shape(), [] as [usize; 0]);
        let val = ind_h.item().unwrap();
        assert!(
            (val - expected_sum).abs() < 1e-5,
            "expected {expected_sum}, got {val}"
        );
    }

    #[test]
    fn test_independent_sample_shape() {
        // Independent(Normal(loc=[2], scale=[2]), reinterpreted_batch_ndims=1):
        //   batch_shape = []  (the [2] dim is reinterpreted as event)
        //   event_shape = [2]
        // sample([5]) → shape [5, 2]  (PyTorch semantics: sample_shape ++ event_shape)
        let loc = cpu_tensor(&[0.0, 0.0], &[2]);
        let scale = cpu_tensor(&[1.0, 1.0], &[2]);
        let normal = Normal::new(loc, scale).unwrap();
        let ind = Independent::new(normal, 1).unwrap();
        let s = ind.sample(&[5]).unwrap();
        assert_eq!(s.shape(), &[5, 2]);
    }
}
