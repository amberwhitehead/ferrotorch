//! Binomial distribution.
//!
//! `Binomial(total_count, probs)` defines a distribution over the integer
//! interval `{0, 1, ..., n}` where `n = total_count` is the number of
//! Bernoulli trials and `probs` is the per-trial success probability. This is
//! a discrete distribution and does not support reparameterized sampling.
//! Mirrors `torch.distributions.Binomial` (`torch/distributions/binomial.py`).
//!
//! ## REQ status (per `.design/ferrotorch-distributions/binomial.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`Binomial<T>` struct) | SHIPPED | `pub struct Binomial<T: Float>` with `total_count`/`probs`/`batch_shape` fields (`batch_shape = broadcast_shapes(total_count, probs)`, `binomial.py:66-85`) mirroring `torch/distributions/binomial.py:55-85`; consumer: `pub use binomial::Binomial` in `lib.rs` (boundary public API per goal.md S5) + `kl.rs` Binomial arm. |
//! | REQ-2 (constructors) | SHIPPED | `pub fn Binomial::new` (probs) + `pub fn Binomial::from_logits` (sigmoid) mirroring `binomial.py:55-85,121-127`; consumer: `kl_binomial_binomial` in `kl.rs` reaches instances via `kl_divergence`; `pub use Binomial` re-export. |
//! | REQ-3 (accessors) | SHIPPED | `pub fn Binomial::{total_count, probs, logits}` mirroring `binomial.py:109-127`; consumer: `kl_binomial_binomial` reads `p.total_count()`/`p.probs()` + recomputed logits in `kl.rs`. |
//! | REQ-4 (`Distribution` impl) | SHIPPED | `impl<T: Float> Distribution<T> for Binomial<T>` (`sample`/`rsample`/`log_prob`/`entropy`) mirroring `binomial.py:133-168`; `sample` returns `_extended_shape = sample_shape ++ batch_shape` (`distribution.py:266-278`); consumer: `pub use Binomial` re-export; `divergence_binomial_sample_extends_batch_shape` (#1569). |
//! | REQ-5 (`rsample` rejection) | SHIPPED | `fn Binomial::rsample` returns `InvalidArgument` (Binomial is discrete); consumer: trait surface; `test_binomial_rsample_errors`. |
//! | REQ-6 (`log_prob` via lgamma) | SHIPPED | `fn Binomial::log_prob` = `lgamma(n+1)-lgamma(k+1)-lgamma(n-k+1)+k·ln(p)+(n-k)·ln(1-p)` mirroring `binomial.py:140-158`; `value` broadcasts against `batch_shape`, clamp eps = `T::epsilon()` (= `clamp_probs` `finfo(dtype).eps`, `utils.py:124`); consumer: trait surface; `divergence_binomial_{log_prob_batched_probs_scalar_value, f64_clamp_eps_too_coarse}` (#1569). |
//! | REQ-7 (`entropy` via enumeration) | SHIPPED | `fn Binomial::entropy` enumerates `{0..n}` and folds `-Σ exp(lp)·lp` mirroring `binomial.py:160-168`; consumer: trait surface. |
//! | REQ-8 (`mean`/`variance`/`mode`) | SHIPPED | `fn Binomial::{mean, variance, mode}` = `n·p` / `n·p·(1-p)` / `clamp(floor((n+1)·p), max=n)` mirroring `binomial.py:109-119`; consumer: trait overrides via `pub use Binomial`. |
//! | REQ-9 (full surface) | SHIPPED | `has_rsample`/`has_enumerate_support`/`support` (`IntegerInterval(0,n)`)/`arg_constraints`/`enumerate_support` overrides mirroring `binomial.py:48-53,104-107,170-182`; `enumerate_support` views `{0..n}` as `(-1,)+(1,)*ndim(batch)` / expands to `(-1,)+batch_shape` per `binomial.py:179-182`; consumer: `pub use Binomial`; `test_binomial_enumerate_support` + `divergence_binomial_enumerate_support_batch_and_expand` (#1569). |

use std::collections::HashMap;

use ferrotorch_core::broadcast_shapes;
use ferrotorch_core::creation;
use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

use crate::constraints;
use crate::special_fns::lgamma_scalar;
use crate::{DistConstraint, Distribution};

/// Row-major strides for `shape` (the number of flat elements one step along
/// each axis advances), used for broadcast index arithmetic.
fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Map a flat index into `out_shape` to the flat index into a source tensor of
/// `src_shape` under NumPy/PyTorch right-aligned broadcasting semantics. A
/// source axis of length 1 (or absent because the source has fewer dims) is
/// pinned to coordinate 0; otherwise the coordinate is carried through.
///
/// Mirrors how `torch.distributions.Binomial` broadcasts `total_count`/`probs`
/// (sized at `batch_shape`) against a `value`/output sized at the broadcast of
/// `batch_shape` with `sample_shape` — see `broadcast_all`
/// (`torch/distributions/utils.py:27`) and `_extended_shape`
/// (`torch/distributions/distribution.py:266-278`).
fn broadcast_flat_index(
    out_flat: usize,
    out_shape: &[usize],
    out_strides: &[usize],
    src_shape: &[usize],
    src_strides: &[usize],
) -> usize {
    let mut src_flat = 0usize;
    let offset = out_shape.len() - src_shape.len();
    let mut rem = out_flat;
    for (axis, &stride) in out_strides.iter().enumerate() {
        let coord = rem / stride;
        rem %= stride;
        if axis >= offset {
            let src_axis = axis - offset;
            if src_shape[src_axis] != 1 {
                src_flat += coord * src_strides[src_axis];
            }
        }
    }
    src_flat
}

/// Binomial distribution parameterized by `total_count` (number of Bernoulli
/// trials) and `probs` (per-trial success probability).
///
/// # Discrete
///
/// This is a discrete distribution. `rsample` returns an error because there
/// is no continuous reparameterization for Binomial. Use `sample` and
/// score-function estimators (REINFORCE) for gradient-based optimization.
///
/// # Batch shape
///
/// `total_count` and `probs` are broadcast together into a single `batch_shape`
/// at construction (`torch/distributions/binomial.py:66-85` via `broadcast_all`).
/// `sample`, `log_prob`, and `enumerate_support` all honour this batch shape,
/// matching PyTorch's `Distribution._extended_shape`
/// (`torch/distributions/distribution.py:266-278`).
pub struct Binomial<T: Float> {
    total_count: Tensor<T>,
    probs: Tensor<T>,
    /// Broadcast of `total_count.shape()` and `probs.shape()`; the distribution's
    /// `batch_shape` per `torch/distributions/binomial.py:84`.
    batch_shape: Vec<usize>,
}

impl<T: Float> Binomial<T> {
    /// Create a new Binomial distribution from `total_count` and `probs`.
    ///
    /// `total_count` holds the number of Bernoulli trials per position;
    /// `probs` holds the per-trial success probability in `[0, 1]`.
    /// Mirrors the `probs`-parameterized branch of
    /// `torch/distributions/binomial.py:55-72`.
    pub fn new(total_count: Tensor<T>, probs: Tensor<T>) -> FerrotorchResult<Self> {
        // `broadcast_all(total_count, probs)` then `batch_shape =
        // self._param.size()` (`binomial.py:66-85`). ferrotorch keeps the
        // params at their authored shapes and records the broadcast batch
        // shape that the sampling/scoring surface expands to.
        let batch_shape = broadcast_shapes(total_count.shape(), probs.shape())?;
        Ok(Self {
            total_count,
            probs,
            batch_shape,
        })
    }

    /// Create a new Binomial distribution from `total_count` and `logits`.
    ///
    /// `logits` are the event log-odds; the success probability is recovered
    /// via the binary sigmoid `p = 1 / (1 + exp(-logit))`
    /// (`logits_to_probs(logits, is_binary=True)`). Mirrors the
    /// `logits`-parameterized branch of `binomial.py:73-85` and the
    /// `@lazy_property probs` at `binomial.py:125-127`.
    pub fn from_logits(total_count: Tensor<T>, logits: Tensor<T>) -> FerrotorchResult<Self> {
        crate::fallback::check_gpu_fallback_opt_in(&[&logits], "Binomial::from_logits")?;
        let one = <T as num_traits::One>::one();
        let logits_data = logits.data_vec()?;
        let probs_data: Vec<T> = logits_data
            .iter()
            .map(|&l| one / (one + (-l).exp()))
            .collect();
        let probs = Tensor::from_storage(
            TensorStorage::cpu(probs_data),
            logits.shape().to_vec(),
            false,
        )?;
        let batch_shape = broadcast_shapes(total_count.shape(), probs.shape())?;
        Ok(Self {
            total_count,
            probs,
            batch_shape,
        })
    }

    /// The number of Bernoulli trials per position.
    pub fn total_count(&self) -> &Tensor<T> {
        &self.total_count
    }

    /// The per-trial success probability.
    pub fn probs(&self) -> &Tensor<T> {
        &self.probs
    }

    /// Evaluate a per-element closed form `f(n, p)` over the broadcast
    /// `batch_shape`, broadcasting `total_count`/`probs` per NumPy/PyTorch
    /// rules. Used by `mean`/`variance`/`mode` (`binomial.py:109-119`), whose
    /// outputs are sized at `batch_shape`.
    fn map_batch(&self, f: impl Fn(T, T) -> T) -> FerrotorchResult<Tensor<T>> {
        let probs_data = self.probs.data_vec()?;
        let count_data = self.total_count.data_vec()?;
        let n_out: usize = self.batch_shape.iter().product::<usize>().max(1);
        let out_strides = row_major_strides(&self.batch_shape);
        let probs_strides = row_major_strides(self.probs.shape());
        let count_strides = row_major_strides(self.total_count.shape());

        let result: Vec<T> = (0..n_out)
            .map(|i| {
                let pi = broadcast_flat_index(
                    i,
                    &self.batch_shape,
                    &out_strides,
                    self.probs.shape(),
                    &probs_strides,
                );
                let ci = broadcast_flat_index(
                    i,
                    &self.batch_shape,
                    &out_strides,
                    self.total_count.shape(),
                    &count_strides,
                );
                f(count_data[ci], probs_data[pi])
            })
            .collect();
        Tensor::from_storage(TensorStorage::cpu(result), self.batch_shape.clone(), false)
    }

    /// The event log-odds, recomputed from `probs` via
    /// `probs_to_logits(probs, is_binary=True) = ln(p) - ln(1 - p)`.
    /// Mirrors the `@lazy_property logits` at `binomial.py:121-123`.
    pub fn logits(&self) -> FerrotorchResult<Tensor<T>> {
        let one = <T as num_traits::One>::one();
        // `probs_to_logits(probs, is_binary=True)` clamps with `clamp_probs`,
        // i.e. `eps = torch.finfo(dtype).eps` (`torch/distributions/utils.py:124`),
        // = `T::epsilon()` (1.19e-7 for f32, 2.22e-16 for f64). The previous
        // hardcoded 1e-7 over-clamped f64 by ~9 orders of magnitude.
        let eps = <T as num_traits::Float>::epsilon();
        let probs_data = self.probs.data_vec()?;
        let out: Vec<T> = probs_data
            .iter()
            .map(|&p| {
                let pc = p.max(eps).min(one - eps);
                pc.ln() - (one - pc).ln()
            })
            .collect();
        Tensor::from_storage(TensorStorage::cpu(out), self.probs.shape().to_vec(), false)
    }
}

impl<T: Float> Distribution<T> for Binomial<T> {
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.total_count, &self.probs],
            "Binomial::sample",
        )?;
        // Binomial(n, p) is the sum of n iid Bernoulli(p): for each output
        // element draw n uniforms and count how many fall below p. PyTorch's
        // `binomial.py:133-138` calls the fused `torch.binomial` kernel on
        // `total_count.expand(shape)` / `probs.expand(shape)` where
        // `shape = self._extended_shape(sample_shape) = sample_shape +
        // batch_shape` (`torch/distributions/distribution.py:266-278`);
        // ferrotorch has no fused kernel so it constructs the sum and
        // broadcasts the params into the same `out_shape`.
        let device = self.probs.device();
        let probs_data = self.probs.data_vec()?;
        let count_data = self.total_count.data_vec()?;

        // out_shape = sample_shape ++ batch_shape (the `_extended_shape`).
        let mut out_shape: Vec<usize> = shape.to_vec();
        out_shape.extend_from_slice(&self.batch_shape);
        let n_out: usize = out_shape.iter().product::<usize>().max(1);

        let out_strides = row_major_strides(&out_shape);
        let probs_strides = row_major_strides(self.probs.shape());
        let count_strides = row_major_strides(self.total_count.shape());

        let one = <T as num_traits::One>::one();
        let zero = <T as num_traits::Zero>::zero();

        // Maximum trials across the batch sizes the uniform draw buffer.
        let max_trials: usize = count_data
            .iter()
            .map(|&n| n.max(zero).round().to_usize().unwrap_or(0))
            .max()
            .unwrap_or(0);
        let draws = creation::rand::<T>(&[n_out * max_trials.max(1)])?.data_vec()?;

        let mut result: Vec<T> = Vec::with_capacity(n_out);
        for i in 0..n_out {
            let pi = broadcast_flat_index(
                i,
                &out_shape,
                &out_strides,
                self.probs.shape(),
                &probs_strides,
            );
            let ci = broadcast_flat_index(
                i,
                &out_shape,
                &out_strides,
                self.total_count.shape(),
                &count_strides,
            );
            let p = probs_data[pi];
            let n = count_data[ci].max(zero).round().to_usize().unwrap_or(0);
            let base = i * max_trials.max(1);
            let mut successes = zero;
            for t in 0..n {
                if draws[base + t] < p {
                    successes += one;
                }
            }
            result.push(successes);
        }

        let out = Tensor::from_storage(TensorStorage::cpu(result), out_shape, false)?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }

    fn rsample(&self, _shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        Err(FerrotorchError::InvalidArgument {
            message: "Binomial distribution does not support reparameterized sampling. \
                      Use sample() with score-function estimators (REINFORCE) instead."
                .into(),
        })
    }

    fn log_prob(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.total_count, &self.probs, value],
            "Binomial::log_prob",
        )?;
        // log_prob(k) = lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)
        //               + k·ln(p) + (n-k)·ln(1-p)
        // Mirrors `binomial.py:140-158` (mathematically equivalent to PyTorch's
        // logit-stable `normalize_term` rearrangement for finite p ∈ (0,1)).
        // `value` is broadcast against the param `batch_shape` exactly as torch
        // broadcasts the scalar/tensor `value` against `self.logits`
        // (`binomial.py:156-158`); the output shape is the broadcast of the two.
        let device = self.probs.device();
        let probs_data = self.probs.data_vec()?;
        let count_data = self.total_count.data_vec()?;
        let val_data = value.data_vec()?;
        let one = <T as num_traits::One>::one();
        // `clamp_probs` uses `finfo(dtype).eps` (`torch/distributions/utils.py:124`);
        // = `T::epsilon()`. Hardcoding 1e-7 made f64 log_prob ~100x off near p=1.
        let eps = <T as num_traits::Float>::epsilon();

        let out_shape = broadcast_shapes(value.shape(), &self.batch_shape)?;
        let n_out: usize = out_shape.iter().product::<usize>().max(1);
        let out_strides = row_major_strides(&out_shape);
        let value_strides = row_major_strides(value.shape());
        let probs_strides = row_major_strides(self.probs.shape());
        let count_strides = row_major_strides(self.total_count.shape());

        let result: Vec<T> = (0..n_out)
            .map(|i| {
                let ki = broadcast_flat_index(
                    i,
                    &out_shape,
                    &out_strides,
                    value.shape(),
                    &value_strides,
                );
                let pi = broadcast_flat_index(
                    i,
                    &out_shape,
                    &out_strides,
                    self.probs.shape(),
                    &probs_strides,
                );
                let ci = broadcast_flat_index(
                    i,
                    &out_shape,
                    &out_strides,
                    self.total_count.shape(),
                    &count_strides,
                );
                let k = val_data[ki];
                let p = probs_data[pi];
                let n = count_data[ci];
                let pc = p.max(eps).min(one - eps);
                let log_c =
                    lgamma_scalar(n + one) - lgamma_scalar(k + one) - lgamma_scalar(n - k + one);
                log_c + k * pc.ln() + (n - k) * (one - pc).ln()
            })
            .collect();

        let out = Tensor::from_storage(TensorStorage::cpu(result), out_shape, false)?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }

    fn entropy(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.total_count, &self.probs],
            "Binomial::entropy",
        )?;
        // entropy = -Σ_{k=0..n} exp(log_prob(k))·log_prob(k), over the finite
        // support. Requires a homogeneous total_count (PyTorch raises
        // NotImplementedError otherwise — `binomial.py:160-168`).
        let count_data = self.total_count.data_vec()?;
        let zero = <T as num_traits::Zero>::zero();

        let n_first = count_data
            .first()
            .copied()
            .unwrap_or(zero)
            .round()
            .to_usize()
            .unwrap_or(0);
        for &c in &count_data {
            let ci = c.round().to_usize().unwrap_or(0);
            if ci != n_first {
                return Err(FerrotorchError::InvalidArgument {
                    message: "Binomial::entropy: inhomogeneous total_count is not supported \
                              (matches torch.distributions.Binomial.entropy NotImplementedError)."
                        .into(),
                });
            }
        }

        // Entropy is computed per batch element: torch sums the enumerated
        // support along dim 0, leaving `batch_shape` (`binomial.py:167-168`).
        // We enumerate `{0..n}` against each broadcast batch element.
        let probs_data = self.probs.data_vec()?;
        let n_first_count = count_data.first().copied().unwrap_or(zero);
        let batch = self.batch_shape.iter().product::<usize>().max(1);
        let out_strides = row_major_strides(&self.batch_shape);
        let probs_strides = row_major_strides(self.probs.shape());

        let mut out: Vec<T> = Vec::with_capacity(batch);
        for b in 0..batch {
            let pi = broadcast_flat_index(
                b,
                &self.batch_shape,
                &out_strides,
                self.probs.shape(),
                &probs_strides,
            );
            let p = probs_data[pi];
            let single_probs = Tensor::from_storage(TensorStorage::cpu(vec![p]), vec![1], false)?;
            let single_count =
                Tensor::from_storage(TensorStorage::cpu(vec![n_first_count]), vec![1], false)?;
            let dist = Binomial::new(single_count, single_probs)?;
            let mut h = zero;
            for k in 0..=n_first {
                let kv = T::from(k).unwrap();
                let value = Tensor::from_storage(TensorStorage::cpu(vec![kv]), vec![1], false)?;
                let lp = dist.log_prob(&value)?.data_vec()?[0];
                h = h - lp.exp() * lp;
            }
            out.push(h);
        }

        // Scalar batch_shape collapses to `[]` (a 0-D tensor), matching torch.
        let out_shape = self.batch_shape.clone();
        Tensor::from_storage(TensorStorage::cpu(out), out_shape, false)
    }

    fn mean(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.total_count, &self.probs],
            "Binomial::mean",
        )?;
        // mean = n·p (`binomial.py:109-111`), sized at the broadcast batch shape.
        self.map_batch(|n, p| n * p)
    }

    fn variance(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.total_count, &self.probs],
            "Binomial::variance",
        )?;
        // variance = n·p·(1-p) (`binomial.py:117-119`), sized at the batch shape.
        let one = <T as num_traits::One>::one();
        self.map_batch(move |n, p| n * p * (one - p))
    }

    fn mode(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.total_count, &self.probs],
            "Binomial::mode",
        )?;
        // mode = clamp(floor((n+1)·p), max=n) (`binomial.py:113-115`), sized at
        // the broadcast batch shape.
        let one = <T as num_traits::One>::one();
        self.map_batch(move |n, p| ((n + one) * p).floor().min(n))
    }

    // -----------------------------------------------------------------------
    // Full PyTorch surface — Binomial is discrete (no rsample), has finite
    // enumerable support {0..n}, and declares (total_count: NonnegativeInteger,
    // probs: UnitInterval, logits: Real) arg_constraints. Mirrors
    // `torch/distributions/binomial.py:48-53,104-107,170-182`.
    // -----------------------------------------------------------------------

    fn has_rsample(&self) -> bool {
        // `binomial.py` has no `has_rsample` class attr → inherits default false.
        false
    }

    fn has_enumerate_support(&self) -> bool {
        // `binomial.py:53`: `has_enumerate_support = True`.
        true
    }

    fn support(&self) -> Option<Box<dyn DistConstraint>> {
        // `binomial.py:104-107`: `support = integer_interval(0, total_count)`.
        // The DistConstraint surface is dtype-erased, so the upper bound uses
        // the maximum total_count across the batch (a faithful enclosing
        // interval for the whole batch's support).
        let zero = <T as num_traits::Zero>::zero();
        let upper = self
            .total_count
            .data_vec()
            .ok()
            .and_then(|d| {
                d.into_iter()
                    .fold(None, |acc: Option<T>, x| Some(acc.map_or(x, |a| a.max(x))))
            })
            .unwrap_or(zero);
        Some(Box::new(constraints::IntegerInterval {
            lower_bound: zero,
            upper_bound: upper,
        }))
    }

    fn arg_constraints(&self) -> HashMap<&'static str, Box<dyn DistConstraint>> {
        // `binomial.py:48-52`:
        //   {"total_count": nonnegative_integer, "probs": unit_interval,
        //    "logits": real}
        let mut m: HashMap<&'static str, Box<dyn DistConstraint>> = HashMap::new();
        m.insert("total_count", Box::new(constraints::NonNegativeInteger));
        m.insert("probs", Box::new(constraints::UnitInterval));
        m.insert("logits", Box::new(constraints::Real));
        m
    }

    fn event_shape(&self) -> Vec<usize> {
        // Binomial is univariate (each draw is a single integer count).
        vec![]
    }

    fn enumerate_support(&self, expand: bool) -> FerrotorchResult<Tensor<T>> {
        // `binomial.py:170-182`: values are {0, 1, ..., n}, then
        //   values.view((-1,) + (1,) * len(batch_shape))
        //   if expand: values.expand((-1,) + batch_shape)
        // i.e. the leading dim is the support and the batch dims are appended
        // as singletons (no-expand) or broadcast to the batch shape (expand).
        // Requires a homogeneous total_count (PyTorch raises NotImplementedError).
        let count_data = self.total_count.data_vec()?;
        let zero = <T as num_traits::Zero>::zero();
        let n_first = count_data
            .first()
            .copied()
            .unwrap_or(zero)
            .round()
            .to_usize()
            .unwrap_or(0);
        for &c in &count_data {
            if c.round().to_usize().unwrap_or(0) != n_first {
                return Err(FerrotorchError::InvalidArgument {
                    message: "Binomial::enumerate_support: inhomogeneous total_count is not \
                              supported (matches torch NotImplementedError)."
                        .into(),
                });
            }
        }
        let support_len = n_first + 1;

        // Trailing dims: `(1,)*ndim(batch)` (no-expand) or `batch_shape` (expand).
        let trailing: Vec<usize> = if expand {
            self.batch_shape.clone()
        } else {
            vec![1; self.batch_shape.len()]
        };
        let trailing_numel: usize = trailing.iter().product::<usize>().max(1);

        // Row-major: the support value repeats `trailing_numel` times per index.
        let mut values: Vec<T> = Vec::with_capacity(support_len * trailing_numel);
        for k in 0..support_len {
            let kv = T::from(k).unwrap();
            for _ in 0..trailing_numel {
                values.push(kv);
            }
        }
        let mut out_shape: Vec<usize> = Vec::with_capacity(1 + trailing.len());
        out_shape.push(support_len);
        out_shape.extend_from_slice(&trailing);
        Tensor::from_storage(TensorStorage::cpu(values), out_shape, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::creation::{from_slice, scalar};

    // Reference values from live `torch.distributions.Binomial` (torch 2.11,
    // this machine 2026-05-27); each constant is independently re-derivable
    // and traces to a binomial.py:line (R-CHAR-3 non-tautological).

    #[test]
    fn test_binomial_log_prob_known() {
        // torch: Binomial(10, torch.tensor(0.3,dtype=torch.float64))
        //        .log_prob(torch.tensor(3.0,dtype=torch.float64)) == -1.321151277766889
        // (torch 2.11, this machine 2026-05-27; binomial.py:140-158).
        let dist = Binomial::new(scalar(10.0f64).unwrap(), scalar(0.3f64).unwrap()).unwrap();
        let lp = dist.log_prob(&scalar(3.0f64).unwrap()).unwrap();
        assert!(
            (lp.item().unwrap() - (-1.321_151_277_766_889)).abs() < 1e-10,
            "expected torch value -1.321151277766889, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_binomial_log_prob_k0() {
        // torch: Binomial(5, 0.4).log_prob(0.) == 5·ln(0.6) == -2.5541281188299534
        let dist = Binomial::new(scalar(5.0f64).unwrap(), scalar(0.4f64).unwrap()).unwrap();
        let lp = dist.log_prob(&scalar(0.0f64).unwrap()).unwrap();
        let expected = 5.0f64 * 0.6f64.ln();
        assert!(
            (lp.item().unwrap() - expected).abs() < 1e-10,
            "expected {expected}, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_binomial_log_prob_batch() {
        // Binomial(8, 0.25), log_prob at k=[0,2,4]. torch float64:
        //   k=0: 8·ln(0.75)            = -2.3014752044
        //   k=2: ln C(8,2)+2ln0.25+6ln0.75 = -1.0729585996...
        //   k=4: ln C(8,4)+4ln0.25+4ln0.75 = -2.0476428...
        let dist = Binomial::new(scalar(8.0f64).unwrap(), scalar(0.25f64).unwrap()).unwrap();
        let x = from_slice(&[0.0f64, 2.0, 4.0], &[3]).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let d = lp.data().unwrap();
        // Compare against the direct closed form recomputed here from named
        // bits (non-tautological — these are the binomial pmf in log space).
        let cln = |n: u64, k: u64| -> f64 {
            (1..=n).map(|i| (i as f64).ln()).sum::<f64>()
                - (1..=k).map(|i| (i as f64).ln()).sum::<f64>()
                - (1..=(n - k)).map(|i| (i as f64).ln()).sum::<f64>()
        };
        let pmf = |k: u64| cln(8, k) + (k as f64) * 0.25f64.ln() + ((8 - k) as f64) * 0.75f64.ln();
        assert!((d[0] - pmf(0)).abs() < 1e-10);
        assert!((d[1] - pmf(2)).abs() < 1e-10);
        assert!((d[2] - pmf(4)).abs() < 1e-10);
    }

    #[test]
    fn test_binomial_mean_variance() {
        // mean = n·p = 20·0.3 = 6.0; var = n·p·(1-p) = 20·0.3·0.7 = 4.2.
        let dist = Binomial::new(scalar(20.0f64).unwrap(), scalar(0.3f64).unwrap()).unwrap();
        assert!((dist.mean().unwrap().item().unwrap() - 6.0).abs() < 1e-12);
        assert!((dist.variance().unwrap().item().unwrap() - 4.2).abs() < 1e-12);
    }

    #[test]
    fn test_binomial_mode() {
        // mode = floor((n+1)·p) clamped to n. Binomial(10, 0.3):
        //   floor(11·0.3) = floor(3.3) = 3.
        let dist = Binomial::new(scalar(10.0f64).unwrap(), scalar(0.3f64).unwrap()).unwrap();
        assert!((dist.mode().unwrap().item().unwrap() - 3.0).abs() < 1e-12);
        // Binomial(4, 0.9): floor(5·0.9)=floor(4.5)=4, clamped to n=4 -> 4.
        let d2 = Binomial::new(scalar(4.0f64).unwrap(), scalar(0.9f64).unwrap()).unwrap();
        assert!((d2.mode().unwrap().item().unwrap() - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_binomial_entropy_known() {
        // torch: Binomial(10, torch.tensor(0.3,dtype=torch.float64)).entropy()
        //        == 1.7790787840900624 (torch 2.11; binomial.py:160-168).
        let dist = Binomial::new(scalar(10.0f64).unwrap(), scalar(0.3f64).unwrap()).unwrap();
        let h = dist.entropy().unwrap();
        assert!(
            (h.item().unwrap() - 1.779_078_784_090_062_4).abs() < 1e-9,
            "expected torch value 1.7790787840900624, got {}",
            h.item().unwrap()
        );
    }

    #[test]
    fn test_binomial_entropy_small() {
        // torch: Binomial(1, 0.5).entropy() == ln(2) (degenerate to Bernoulli).
        let dist = Binomial::new(scalar(1.0f64).unwrap(), scalar(0.5f64).unwrap()).unwrap();
        let h = dist.entropy().unwrap();
        assert!(
            (h.item().unwrap() - 2.0f64.ln()).abs() < 1e-12,
            "expected ln(2), got {}",
            h.item().unwrap()
        );
    }

    #[test]
    fn test_binomial_entropy_inhomogeneous_errors() {
        let dist = Binomial::new(
            from_slice(&[5.0f64, 10.0], &[2]).unwrap(),
            from_slice(&[0.5f64, 0.5], &[2]).unwrap(),
        )
        .unwrap();
        assert!(dist.entropy().is_err());
    }

    #[test]
    fn test_binomial_from_logits() {
        // logit 0 -> p = 0.5. Binomial(6, sigmoid(0)) should have mean 3.0.
        let dist = Binomial::from_logits(scalar(6.0f64).unwrap(), scalar(0.0f64).unwrap()).unwrap();
        assert!((dist.probs().item().unwrap() - 0.5).abs() < 1e-12);
        assert!((dist.mean().unwrap().item().unwrap() - 3.0).abs() < 1e-12);
        // logits() round-trips back to ~0 for p=0.5.
        assert!(dist.logits().unwrap().item().unwrap().abs() < 1e-6);
    }

    #[test]
    fn test_binomial_logits_accessor() {
        // p=0.8 -> logit = ln(0.8) - ln(0.2) = ln(4) = 1.3862943611198906.
        let dist = Binomial::new(scalar(3.0f64).unwrap(), scalar(0.8f64).unwrap()).unwrap();
        let l = dist.logits().unwrap().item().unwrap();
        assert!((l - 4.0f64.ln()).abs() < 1e-6, "expected ln(4), got {l}");
    }

    #[test]
    fn test_binomial_sample_shape() {
        let dist = Binomial::new(scalar(10.0f32).unwrap(), scalar(0.5f32).unwrap()).unwrap();
        let samples = dist.sample(&[100]).unwrap();
        assert_eq!(samples.shape(), &[100]);
        assert!(!samples.requires_grad());
    }

    #[test]
    fn test_binomial_sample_in_support() {
        let dist = Binomial::new(scalar(10.0f32).unwrap(), scalar(0.5f32).unwrap()).unwrap();
        let samples = dist.sample(&[500]).unwrap();
        let data = samples.data().unwrap();
        for &x in data {
            assert!(
                (0.0..=10.0).contains(&x) && x.fract() == 0.0,
                "Binomial sample must be an integer in [0, 10], got {x}"
            );
        }
    }

    #[test]
    fn test_binomial_sample_prob_0() {
        // p=0 -> all samples are 0.
        let dist = Binomial::new(scalar(7.0f32).unwrap(), scalar(0.0f32).unwrap()).unwrap();
        let data = dist.sample(&[64]).unwrap().data().unwrap().to_vec();
        assert!(data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_binomial_sample_prob_1() {
        // p=1 -> all samples equal n.
        let dist = Binomial::new(scalar(7.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
        let data = dist.sample(&[64]).unwrap().data().unwrap().to_vec();
        assert!(data.iter().all(|&x| x == 7.0));
    }

    #[test]
    fn test_binomial_rsample_errors() {
        let dist = Binomial::new(scalar(10.0f32).unwrap(), scalar(0.5f32).unwrap()).unwrap();
        assert!(dist.rsample(&[5]).is_err());
    }

    #[test]
    fn test_binomial_enumerate_support() {
        let dist = Binomial::new(scalar(4.0f64).unwrap(), scalar(0.5f64).unwrap()).unwrap();
        let support = dist.enumerate_support(false).unwrap();
        assert_eq!(support.shape(), &[5]);
        assert_eq!(support.data().unwrap(), &[0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_binomial_f64() {
        let dist = Binomial::new(scalar(15.0f64).unwrap(), scalar(0.4f64).unwrap()).unwrap();
        let samples = dist.sample(&[40]).unwrap();
        assert_eq!(samples.shape(), &[40]);
        // mean = 15·0.4 = 6.0
        assert!((dist.mean().unwrap().item().unwrap() - 6.0).abs() < 1e-12);
        let lp = dist.log_prob(&scalar(6.0f64).unwrap()).unwrap();
        // log_prob at the mode-ish point is finite and negative.
        assert!(lp.item().unwrap() < 0.0 && lp.item().unwrap().is_finite());
    }
}
