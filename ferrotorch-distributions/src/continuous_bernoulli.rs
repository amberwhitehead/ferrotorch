//! Continuous Bernoulli distribution.
//!
//! `ContinuousBernoulli(probs)` is a **continuous** distribution supported on
//! the closed unit interval `[0, 1]`, parameterized by `probs` (in `(0, 1)`)
//! or `logits` (real-valued). Despite the names, `probs` is NOT a probability
//! and `logits` is NOT a log-odds — they are the natural parameter of an
//! exponential-family density whose normalizing constant `C(λ)` is the crux of
//! the distribution. The density is
//! `p(x; λ) = C(λ) · λ^x · (1-λ)^(1-x)` for `x ∈ [0, 1]`. Mirrors
//! `torch.distributions.ContinuousBernoulli`
//! (`torch/distributions/continuous_bernoulli.py`). See Loaiza-Ganem &
//! Cunningham, NeurIPS 2019 (arXiv:1907.06845).
//!
//! # The numerical-stability cutoff
//!
//! Every CB closed form is singular at `probs = 0.5` (the `λ/(2λ-1)`,
//! `1/(log1p(-λ)-log(λ))` factors hit `0/0`). PyTorch guards this with
//! `_lims = (0.499, 0.501)`: inside `(0.499, 0.501]` the exact formula is
//! replaced by a Taylor expansion about `0.5`, and the cut value `0.499` is
//! substituted before the exact branch so it never divides by zero. ferrotorch
//! matches `_lims` and every Taylor series byte-for-byte (R-DEV-1).
//!
//! ## REQ status (per `.design/ferrotorch-distributions/continuous_bernoulli.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`ContinuousBernoulli<T>` struct) | SHIPPED | `pub struct ContinuousBernoulli<T: Float>` with `probs`/`batch_shape` fields (`batch_shape = probs.size()`, `continuous_bernoulli.py:84-87`) mirroring `continuous_bernoulli.py:22-89`; consumer: `pub use continuous_bernoulli::ContinuousBernoulli` in `lib.rs` + the CB KL arms in `kl.rs`. |
//! | REQ-2 (constructors) | SHIPPED | `pub fn ContinuousBernoulli::new` (clamps via `clamp_probs`, `continuous_bernoulli.py:76`) + `pub fn ContinuousBernoulli::from_logits` (binary sigmoid then clamp, `continuous_bernoulli.py:77-89`); consumer: `kl_continuous_bernoulli_continuous_bernoulli` in `kl.rs` reaches instances via `kl_divergence`; `pub use` re-export. |
//! | REQ-3 (accessors) | SHIPPED | `pub fn ContinuousBernoulli::{probs, logits}` mirroring `continuous_bernoulli.py:164-174` (`logits = ln(p) - log1p(-p)`); consumer: `kl_beta_continuous_bernoulli` / `kl_uniform_continuous_bernoulli` read `q.probs()` in `kl.rs`. |
//! | REQ-4 (`Distribution` impl) | SHIPPED | `impl<T: Float> Distribution<T> for ContinuousBernoulli<T>` (`sample`/`rsample`/`log_prob`/`entropy`) mirroring `continuous_bernoulli.py:176-231`; `sample`/`rsample` return `icdf(u)` over `_extended_shape`; `log_prob = value·logits + log_norm` broadcasts `value` against `batch_shape`; consumer: trait surface via `pub use`; `test_cb_log_prob_*`. |
//! | REQ-5 (`mean`/`variance`/`entropy` with cutoff) | SHIPPED | `fn ContinuousBernoulli::{mean, variance, entropy}` with the `_lims = (0.499, 0.501)` Taylor cutoff mirroring `continuous_bernoulli.py:140-162,224-231`; consumer: trait overrides via `pub use` + CB KL formulas in `kl.rs`; `test_cb_{mean,variance,entropy}_*` incl. near-0.5. |
//! | REQ-6 (`cdf`/`icdf` with cutoff) | SHIPPED | `fn ContinuousBernoulli::{cdf, icdf}` with the cutoff mirroring `continuous_bernoulli.py:196-222`; consumer: `sample`/`rsample` call `icdf` (in-module production consumer); `test_cb_{cdf,icdf}_*`. |
//! | REQ-7 (full surface) | SHIPPED | `has_rsample`(`true`)/`support`(`UnitInterval`)/`arg_constraints`/`event_shape`/`batch_shape` overrides mirroring `continuous_bernoulli.py:49-53`; consumer: `pub use`; `test_cb_{support,arg_constraints,has_rsample}`. |
//! | REQ-8 (CB KL pairs) | SHIPPED | 13 CB KL pairs in `kl.rs` (6 finite + 7 `+inf`) — full evidence in `.design/ferrotorch-distributions/kl.md` REQ-7; consumer: each invoked by `fn kl_dispatch` reached via `pub fn kl_divergence`. |

use std::collections::HashMap;

use ferrotorch_core::creation;
use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::FerrotorchResult;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

use crate::constraints;
use crate::{DistConstraint, Distribution};

/// PyTorch's `_lims[0]` (`continuous_bernoulli.py:59`). Below-or-equal to this
/// the exact branch is used; inside `(LIM_LO, LIM_HI]` the Taylor branch is.
const LIM_LO: f64 = 0.499;
/// PyTorch's `_lims[1]` (`continuous_bernoulli.py:59`).
const LIM_HI: f64 = 0.501;

/// Row-major strides for `shape` (the number of flat elements one step along
/// each axis advances), used for broadcast index arithmetic. Mirrors the
/// helper in `geometric.rs` / `binomial.rs` (the FIXED batch-broadcast
/// references).
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
/// Mirrors how `torch.distributions.ContinuousBernoulli` broadcasts `probs`
/// (sized at `batch_shape`) against a `value`/output sized at the broadcast of
/// `batch_shape` with `sample_shape` — see `broadcast_all`
/// (`torch/distributions/utils.py:27`) and `_extended_shape`
/// (`torch/distributions/distribution.py:266-278`).
fn broadcast_flat_index(
    out_flat: usize,
    out_strides: &[usize],
    out_ndim: usize,
    src_shape: &[usize],
    src_strides: &[usize],
) -> usize {
    let mut src_flat = 0usize;
    let offset = out_ndim - src_shape.len();
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

// ---------------------------------------------------------------------------
// Scalar closed forms with the `_lims` cutoff (crate-visible so the CB KL
// formulas in `kl.rs` reuse them — the production consumers). Each mirrors a
// `torch.where(self._outside_unstable_region(), exact, taylor)` branch.
// ---------------------------------------------------------------------------

/// `clamp_probs(probs)` (`torch/distributions/utils.py:120-126`): clamp to
/// `[eps, 1-eps]` with `eps = finfo(dtype).eps` (= `T::epsilon()`).
pub(crate) fn clamp_probs<T: Float>(p: T) -> T {
    let one = <T as num_traits::One>::one();
    let eps = <T as num_traits::Float>::epsilon();
    p.max(eps).min(one - eps)
}

/// `_outside_unstable_region(p) = p <= 0.499 || p > 0.501`
/// (`continuous_bernoulli.py:108-111`).
pub(crate) fn outside_unstable_region<T: Float>(p: T) -> bool {
    let lim_lo = T::from(LIM_LO).unwrap();
    let lim_hi = T::from(LIM_HI).unwrap();
    p <= lim_lo || p > lim_hi
}

/// `_cut_probs(p)` (`continuous_bernoulli.py:113-118`): leave `p` untouched
/// outside the unstable band; inside, substitute the cut value `0.499` so the
/// exact branch never divides by zero.
fn cut_probs<T: Float>(p: T) -> T {
    if outside_unstable_region(p) {
        p
    } else {
        T::from(LIM_LO).unwrap()
    }
}

/// `_cont_bern_log_norm(p)` (`continuous_bernoulli.py:120-138`): the log
/// normalizing constant. Exact branch uses the cut probs; Taylor branch about
/// `0.5`. Reused by `log_prob`, `entropy`, and the CB KL formulas in `kl.rs`.
pub(crate) fn cont_bern_log_norm_scalar<T: Float>(p: T) -> T {
    let zero = <T as num_traits::Zero>::zero();
    let one = <T as num_traits::One>::one();
    let two = T::from(2.0).unwrap();
    let half = T::from(0.5).unwrap();
    let cut = cut_probs(p);
    // cut_probs_below_half = cut if cut <= 0.5 else 0; cut_probs_above_half =
    // cut if cut >= 0.5 else 1 (`continuous_bernoulli.py:123-128`).
    let cut_below = if cut <= half { cut } else { zero };
    let cut_above = if cut >= half { cut } else { one };
    // log(|log1p(-cut) - log(cut)|) - (cut<=0.5 ? log1p(-2·cut_below)
    //                                            : log(2·cut_above - 1)).
    let denom = if cut <= half {
        (-(two * cut_below)).ln_1p()
    } else {
        (two * cut_above - one).ln()
    };
    let log_norm = ((-cut).ln_1p() - cut.ln()).abs().ln() - denom;
    // Taylor: ln(2) + (4/3 + 104/45·x)·x, x = (p-0.5)².
    let x = (p - half) * (p - half);
    let c0 = T::from(4.0 / 3.0).unwrap();
    let c1 = T::from(104.0 / 45.0).unwrap();
    let taylor = T::from(std::f64::consts::LN_2).unwrap() + (c0 + c1 * x) * x;
    if outside_unstable_region(p) {
        log_norm
    } else {
        taylor
    }
}

/// CB `mean` per element (`continuous_bernoulli.py:140-148`).
pub(crate) fn mean_scalar<T: Float>(p: T) -> T {
    let one = <T as num_traits::One>::one();
    let two = T::from(2.0).unwrap();
    let half = T::from(0.5).unwrap();
    let cut = cut_probs(p);
    // mus = cut/(2·cut - 1) + 1/(log1p(-cut) - log(cut)).
    let mus = cut / (two * cut - one) + one / ((-cut).ln_1p() - cut.ln());
    // Taylor: 0.5 + (1/3 + 16/45·(p-0.5)²)·(p-0.5).
    let x = p - half;
    let c0 = T::from(1.0 / 3.0).unwrap();
    let c1 = T::from(16.0 / 45.0).unwrap();
    let taylor = half + (c0 + c1 * x * x) * x;
    if outside_unstable_region(p) {
        mus
    } else {
        taylor
    }
}

/// CB `variance` per element (`continuous_bernoulli.py:154-162`).
pub(crate) fn variance_scalar<T: Float>(p: T) -> T {
    let one = <T as num_traits::One>::one();
    let two = T::from(2.0).unwrap();
    let half = T::from(0.5).unwrap();
    let cut = cut_probs(p);
    // cut·(cut-1)/(1-2·cut)² + 1/(log1p(-cut) - log(cut))².
    let denom1 = (one - two * cut) * (one - two * cut);
    let lr = (-cut).ln_1p() - cut.ln();
    let vars = cut * (cut - one) / denom1 + one / (lr * lr);
    // Taylor: 1/12 - (1/15 - 128/945·x)·x, x = (p-0.5)².
    let x = (p - half) * (p - half);
    let c0 = T::from(1.0 / 12.0).unwrap();
    let c1 = T::from(1.0 / 15.0).unwrap();
    let c2 = T::from(128.0 / 945.0).unwrap();
    let taylor = c0 - (c1 - c2 * x) * x;
    if outside_unstable_region(p) {
        vars
    } else {
        taylor
    }
}

/// CB `entropy` per element (`continuous_bernoulli.py:224-231`):
/// `mean·(log1p(-p) - log(p)) - _cont_bern_log_norm() - log1p(-p)`.
pub(crate) fn entropy_scalar<T: Float>(p: T) -> T {
    let log_probs0 = (-p).ln_1p();
    let log_probs1 = p.ln();
    mean_scalar(p) * (log_probs0 - log_probs1) - cont_bern_log_norm_scalar(p) - log_probs0
}

/// CB `logits` per element: `probs_to_logits(p, is_binary=True) = ln(p) -
/// log1p(-p)` (`continuous_bernoulli.py:164-166`, `utils.py:135-137`), on the
/// clamped probs.
pub(crate) fn logits_scalar<T: Float>(p: T) -> T {
    let pc = clamp_probs(p);
    pc.ln() - (-pc).ln_1p()
}

/// CB `cdf` per element (`continuous_bernoulli.py:196-210`), clamped to `[0,1]`
/// at the support ends.
fn cdf_scalar<T: Float>(p: T, v: T) -> T {
    let zero = <T as num_traits::Zero>::zero();
    let one = <T as num_traits::One>::one();
    let two = T::from(2.0).unwrap();
    if v <= zero {
        return zero;
    }
    if v >= one {
        return one;
    }
    let cut = cut_probs(p);
    // (cut^v · (1-cut)^(1-v) + cut - 1)/(2·cut - 1); Taylor branch = v.
    if outside_unstable_region(p) {
        (cut.powf(v) * (one - cut).powf(one - v) + cut - one) / (two * cut - one)
    } else {
        v
    }
}

/// CB `icdf` per element (`continuous_bernoulli.py:212-222`).
fn icdf_scalar<T: Float>(p: T, v: T) -> T {
    let one = <T as num_traits::One>::one();
    let two = T::from(2.0).unwrap();
    if outside_unstable_region(p) {
        let cut = cut_probs(p);
        // (log1p(-cut + v·(2·cut - 1)) - log1p(-cut))/(log(cut) - log1p(-cut)).
        ((-cut + v * (two * cut - one)).ln_1p() - (-cut).ln_1p()) / (cut.ln() - (-cut).ln_1p())
    } else {
        v
    }
}

/// Continuous Bernoulli distribution parameterized by `probs` (the natural
/// parameter `λ ∈ (0, 1)`). Supported on the closed unit interval `[0, 1]`.
///
/// # Continuous + reparameterizable
///
/// CB is a continuous distribution with `has_rsample = true`; `rsample` is the
/// differentiable inverse-CDF map `icdf(Uniform[0,1])`.
///
/// # Batch shape
///
/// `probs` defines the `batch_shape` directly
/// (`torch/distributions/continuous_bernoulli.py:84-87`). `sample`/`rsample`
/// emit `_extended_shape = sample_shape ++ batch_shape`
/// (`torch/distributions/distribution.py:266-278`); `value` in `log_prob` is
/// right-aligned broadcast against `batch_shape`, mirroring
/// `broadcast_all(self.logits, value)` (`continuous_bernoulli.py:190`).
pub struct ContinuousBernoulli<T: Float> {
    /// The clamped natural parameter `λ` (clamped via `clamp_probs`,
    /// `continuous_bernoulli.py:76`).
    probs: Tensor<T>,
    /// The distribution's `batch_shape`, equal to `probs.shape()`
    /// (`continuous_bernoulli.py:84-87`).
    batch_shape: Vec<usize>,
}

impl<T: Float> ContinuousBernoulli<T> {
    /// Create a Continuous Bernoulli from `probs ∈ (0, 1)`.
    ///
    /// `probs` is clamped to `[eps, 1-eps]` via `clamp_probs`
    /// (`continuous_bernoulli.py:76`) before storage, matching PyTorch (which
    /// clamps for numerical stability near 0 and 1). Mirrors the
    /// `probs`-parameterized branch of `continuous_bernoulli.py:66-76`.
    pub fn new(probs: Tensor<T>) -> FerrotorchResult<Self> {
        crate::fallback::check_gpu_fallback_opt_in(&[&probs], "ContinuousBernoulli::new")?;
        let data = probs.data_vec()?;
        let clamped: Vec<T> = data.iter().map(|&p| clamp_probs(p)).collect();
        let shape = probs.shape().to_vec();
        let probs = Tensor::from_storage(TensorStorage::cpu(clamped), shape.clone(), false)?;
        Ok(Self {
            probs,
            batch_shape: shape,
        })
    }

    /// Create a Continuous Bernoulli from real-valued `logits`.
    ///
    /// The natural parameter is recovered via the binary sigmoid
    /// `λ = 1/(1+exp(-logit))` (`logits_to_probs(logits, is_binary=True)`,
    /// `utils.py:97-98`), then clamped via `clamp_probs`
    /// (the `@lazy_property probs` at `continuous_bernoulli.py:168-170`).
    /// Mirrors the `logits`-parameterized branch of
    /// `continuous_bernoulli.py:77-82`.
    pub fn from_logits(logits: Tensor<T>) -> FerrotorchResult<Self> {
        crate::fallback::check_gpu_fallback_opt_in(&[&logits], "ContinuousBernoulli::from_logits")?;
        let one = <T as num_traits::One>::one();
        let logits_data = logits.data_vec()?;
        let probs_data: Vec<T> = logits_data
            .iter()
            .map(|&l| clamp_probs(one / (one + (-l).exp())))
            .collect();
        let shape = logits.shape().to_vec();
        let probs = Tensor::from_storage(TensorStorage::cpu(probs_data), shape.clone(), false)?;
        Ok(Self {
            probs,
            batch_shape: shape,
        })
    }

    /// The clamped natural parameter `λ`.
    pub fn probs(&self) -> &Tensor<T> {
        &self.probs
    }

    /// The real-valued `logits = probs_to_logits(λ, is_binary=True) = ln(λ) -
    /// log1p(-λ)` (`continuous_bernoulli.py:164-166`).
    pub fn logits(&self) -> FerrotorchResult<Tensor<T>> {
        let out: Vec<T> = self
            .probs
            .data_vec()?
            .iter()
            .map(|&p| logits_scalar(p))
            .collect();
        Tensor::from_storage(TensorStorage::cpu(out), self.batch_shape.clone(), false)
    }

    /// Evaluate a per-element closed form `f(λ)` over the `batch_shape`. Used by
    /// `mean`/`variance` (`continuous_bernoulli.py:140-162`).
    fn map_batch(&self, f: impl Fn(T) -> T) -> FerrotorchResult<Tensor<T>> {
        let result: Vec<T> = self.probs.data_vec()?.iter().map(|&p| f(p)).collect();
        Tensor::from_storage(TensorStorage::cpu(result), self.batch_shape.clone(), false)
    }

    /// Broadcast a `value` tensor against `batch_shape` and apply `f(λ, v)`
    /// per element. Used by `log_prob`/`cdf`/`icdf`, mirroring
    /// `broadcast_all(self.logits, value)` (`continuous_bernoulli.py:190`).
    fn map_value(&self, value: &Tensor<T>, f: impl Fn(T, T) -> T) -> FerrotorchResult<Tensor<T>> {
        let probs_data = self.probs.data_vec()?;
        let val_data = value.data_vec()?;
        let out_shape = ferrotorch_core::broadcast_shapes(value.shape(), &self.batch_shape)?;
        let n_out: usize = out_shape.iter().product::<usize>().max(1);
        let out_strides = row_major_strides(&out_shape);
        let value_strides = row_major_strides(value.shape());
        let probs_strides = row_major_strides(self.probs.shape());

        let result: Vec<T> = (0..n_out)
            .map(|i| {
                let vi = broadcast_flat_index(
                    i,
                    &out_strides,
                    out_shape.len(),
                    value.shape(),
                    &value_strides,
                );
                let pi = broadcast_flat_index(
                    i,
                    &out_strides,
                    out_shape.len(),
                    self.probs.shape(),
                    &probs_strides,
                );
                f(probs_data[pi], val_data[vi])
            })
            .collect();
        Tensor::from_storage(TensorStorage::cpu(result), out_shape, false)
    }

    /// Draw `u ~ Uniform[0,1)` over `_extended_shape` and apply `icdf`
    /// element-wise (`continuous_bernoulli.py:176-185`). Shared by `sample`
    /// (no grad) and `rsample` (CB is reparameterizable).
    fn icdf_of_uniform(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        let device = self.probs.device();
        let probs_data = self.probs.data_vec()?;
        let mut out_shape: Vec<usize> = shape.to_vec();
        out_shape.extend_from_slice(&self.batch_shape);
        let n_out: usize = out_shape.iter().product::<usize>().max(1);
        let out_strides = row_major_strides(&out_shape);
        let probs_strides = row_major_strides(self.probs.shape());
        let u = creation::rand::<T>(&out_shape)?.data_vec()?;

        let result: Vec<T> = (0..n_out)
            .map(|i| {
                let pi = broadcast_flat_index(
                    i,
                    &out_strides,
                    out_shape.len(),
                    self.probs.shape(),
                    &probs_strides,
                );
                icdf_scalar(probs_data[pi], u[i])
            })
            .collect();
        let out = Tensor::from_storage(TensorStorage::cpu(result), out_shape, false)?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }
}

impl<T: Float> Distribution<T> for ContinuousBernoulli<T> {
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.probs], "ContinuousBernoulli::sample")?;
        // `sample` draws u ~ rand(shape) then returns icdf(u) under no_grad
        // (`continuous_bernoulli.py:176-180`).
        self.icdf_of_uniform(shape)
    }

    fn rsample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.probs], "ContinuousBernoulli::rsample")?;
        // `rsample` is the same inverse-CDF map but differentiable in `probs`
        // (`continuous_bernoulli.py:182-185`, `has_rsample = True`).
        self.icdf_of_uniform(shape)
    }

    fn log_prob(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.probs, value],
            "ContinuousBernoulli::log_prob",
        )?;
        // log_prob = -BCE_with_logits(logits, value) + _cont_bern_log_norm()
        //          = value·logits + log1p(-probs) + _cont_bern_log_norm()
        // since -BCE_with_logits(ℓ, v) = v·ℓ - log(1+exp(ℓ)) and
        // log(1+exp(ℓ)) = -log(1-λ) for ℓ = logit(λ). Equivalently the stable
        // BCE form `-(max(ℓ,0) - ℓ·v + log1p(exp(-|ℓ|)))`.
        // (`continuous_bernoulli.py:187-194`). `value` broadcasts against
        // `batch_shape`.
        let zero = <T as num_traits::Zero>::zero();
        self.map_value(value, |p, v| {
            let logit = logits_scalar(p);
            let abs_l = logit.abs();
            let max_l0 = if logit > zero { logit } else { zero };
            // -BCE_with_logits(logit, v):
            let neg_bce = -(max_l0 - logit * v + (-abs_l).exp().ln_1p());
            neg_bce + cont_bern_log_norm_scalar(p)
        })
    }

    fn entropy(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.probs], "ContinuousBernoulli::entropy")?;
        // entropy = mean·(log1p(-p) - log(p)) - _cont_bern_log_norm() -
        //           log1p(-p) (`continuous_bernoulli.py:224-231`).
        self.map_batch(entropy_scalar)
    }

    fn mean(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.probs], "ContinuousBernoulli::mean")?;
        // mean = λ/(2λ-1) + 1/(log1p(-λ)-log(λ)) with Taylor cutoff
        // (`continuous_bernoulli.py:140-148`).
        self.map_batch(mean_scalar)
    }

    fn variance(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.probs],
            "ContinuousBernoulli::variance",
        )?;
        // variance = λ(λ-1)/(1-2λ)² + 1/(log1p(-λ)-log(λ))² with Taylor cutoff
        // (`continuous_bernoulli.py:154-162`).
        self.map_batch(variance_scalar)
    }

    fn cdf(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.probs, value],
            "ContinuousBernoulli::cdf",
        )?;
        // (`continuous_bernoulli.py:196-210`); `value` broadcasts vs batch.
        self.map_value(value, cdf_scalar)
    }

    fn icdf(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.probs, value],
            "ContinuousBernoulli::icdf",
        )?;
        // (`continuous_bernoulli.py:212-222`); `value` broadcasts vs batch.
        self.map_value(value, icdf_scalar)
    }

    // -----------------------------------------------------------------------
    // Full PyTorch surface — CB is continuous + reparameterizable
    // (has_rsample = True), supported on the unit interval [0,1], and declares
    // (probs: unit_interval, logits: real) arg_constraints. Mirrors
    // `torch/distributions/continuous_bernoulli.py:49-53`.
    // -----------------------------------------------------------------------

    fn has_rsample(&self) -> bool {
        // `continuous_bernoulli.py:53`: `has_rsample = True`.
        true
    }

    fn support(&self) -> Option<Box<dyn DistConstraint>> {
        // `continuous_bernoulli.py:51`: `support = constraints.unit_interval`.
        Some(Box::new(constraints::UnitInterval))
    }

    fn arg_constraints(&self) -> HashMap<&'static str, Box<dyn DistConstraint>> {
        // `continuous_bernoulli.py:50`:
        //   {"probs": unit_interval, "logits": real}.
        let mut m: HashMap<&'static str, Box<dyn DistConstraint>> = HashMap::new();
        m.insert("probs", Box::new(constraints::UnitInterval));
        m.insert("logits", Box::new(constraints::Real));
        m
    }

    fn event_shape(&self) -> Vec<usize> {
        // CB is univariate (each draw is a single real in [0,1]).
        vec![]
    }

    fn batch_shape(&self) -> Vec<usize> {
        self.batch_shape.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::creation::{from_slice, scalar};

    // Reference values from live `torch.distributions.ContinuousBernoulli`
    // (torch 2.11.0+cu130, this machine 2026-05-27); each constant traces to a
    // continuous_bernoulli.py:line (R-CHAR-3 non-tautological).

    #[test]
    fn test_cb_mean_known() {
        // torch: ContinuousBernoulli(0.3).mean == 0.43022250114382865
        //        ContinuousBernoulli(0.7).mean == 0.5697774988561712
        //        (continuous_bernoulli.py:140-148).
        let d = ContinuousBernoulli::new(scalar(0.3f64).unwrap()).unwrap();
        assert!(
            (d.mean().unwrap().item().unwrap() - 0.430_222_501_143_828_65).abs() < 1e-12,
            "got {}",
            d.mean().unwrap().item().unwrap()
        );
        let d2 = ContinuousBernoulli::new(scalar(0.7f64).unwrap()).unwrap();
        assert!((d2.mean().unwrap().item().unwrap() - 0.569_777_498_856_171_2).abs() < 1e-12);
    }

    #[test]
    fn test_cb_mean_near_half_cutoff() {
        // The crux: at and near probs=0.5 the Taylor branch must engage.
        // torch: CB(0.5).mean == 0.5; CB(0.4999).mean == 0.4999666666663111;
        //        CB(0.5001).mean == 0.5000333333336889.
        let d = ContinuousBernoulli::new(scalar(0.5f64).unwrap()).unwrap();
        assert!((d.mean().unwrap().item().unwrap() - 0.5).abs() < 1e-15);
        let d_lo = ContinuousBernoulli::new(scalar(0.4999f64).unwrap()).unwrap();
        assert!(
            (d_lo.mean().unwrap().item().unwrap() - 0.499_966_666_666_311_1).abs() < 1e-12,
            "near-0.5 (below) got {}",
            d_lo.mean().unwrap().item().unwrap()
        );
        let d_hi = ContinuousBernoulli::new(scalar(0.5001f64).unwrap()).unwrap();
        assert!(
            (d_hi.mean().unwrap().item().unwrap() - 0.500_033_333_333_688_9).abs() < 1e-12,
            "near-0.5 (above) got {}",
            d_hi.mean().unwrap().item().unwrap()
        );
    }

    #[test]
    fn test_cb_variance_known_and_cutoff() {
        // torch: CB(0.3).variance == 0.08042515220619428;
        //        CB(0.5).variance == 0.08333333333333333 (Taylor: 1/12);
        //        CB(0.4999).variance == 0.08333333266666668.
        let d = ContinuousBernoulli::new(scalar(0.3f64).unwrap()).unwrap();
        assert!((d.variance().unwrap().item().unwrap() - 0.080_425_152_206_194_28).abs() < 1e-12);
        let d5 = ContinuousBernoulli::new(scalar(0.5f64).unwrap()).unwrap();
        assert!((d5.variance().unwrap().item().unwrap() - (1.0 / 12.0)).abs() < 1e-15);
        let dlo = ContinuousBernoulli::new(scalar(0.4999f64).unwrap()).unwrap();
        assert!(
            (dlo.variance().unwrap().item().unwrap() - 0.083_333_332_666_666_68).abs() < 1e-12,
            "got {}",
            dlo.variance().unwrap().item().unwrap()
        );
    }

    #[test]
    fn test_cb_entropy_known_and_cutoff() {
        // torch: CB(0.3).entropy() == -0.02938620223212912;
        //        CB(0.5).entropy() == 0.0 (Taylor);
        //        CB(0.1).entropy() == -0.17963111600005513.
        let d = ContinuousBernoulli::new(scalar(0.3f64).unwrap()).unwrap();
        assert!(
            (d.entropy().unwrap().item().unwrap() - (-0.029_386_202_232_129_12)).abs() < 1e-12,
            "got {}",
            d.entropy().unwrap().item().unwrap()
        );
        let d5 = ContinuousBernoulli::new(scalar(0.5f64).unwrap()).unwrap();
        assert!(d5.entropy().unwrap().item().unwrap().abs() < 1e-12);
        let d1 = ContinuousBernoulli::new(scalar(0.1f64).unwrap()).unwrap();
        assert!(
            (d1.entropy().unwrap().item().unwrap() - (-0.179_631_116_000_055_13)).abs() < 1e-12
        );
    }

    #[test]
    fn test_cb_log_prob_known() {
        // torch: CB(0.3).log_prob(x) for x in {0,0.25,0.5,0.75,1.0}
        //        == {0.3939128069417265, 0.1820883418449255,
        //            -0.029736123251875357, -0.24156058834867633,
        //            -0.4533850534454773} (continuous_bernoulli.py:187-194).
        let d = ContinuousBernoulli::new(scalar(0.3f64).unwrap()).unwrap();
        let expected = [
            (0.0f64, 0.393_912_806_941_726_5),
            (0.25, 0.182_088_341_844_925_5),
            (0.5, -0.029_736_123_251_875_357),
            (0.75, -0.241_560_588_348_676_33),
            (1.0, -0.453_385_053_445_477_3),
        ];
        for (v, want) in expected {
            let lp = d.log_prob(&scalar(v).unwrap()).unwrap().item().unwrap();
            assert!(
                (lp - want).abs() < 1e-12,
                "lp(0.3, {v}) = {lp}, want {want}"
            );
        }
    }

    #[test]
    fn test_cb_log_prob_at_half_is_flat() {
        // torch: CB(0.5) is the uniform on [0,1] → log_prob == 0 everywhere
        //        (continuous_bernoulli.py:187-194 with the Taylor log-norm).
        let d = ContinuousBernoulli::new(scalar(0.5f64).unwrap()).unwrap();
        for v in [0.0f64, 0.3, 0.5, 1.0] {
            let lp = d.log_prob(&scalar(v).unwrap()).unwrap().item().unwrap();
            assert!(lp.abs() < 1e-12, "CB(0.5).log_prob({v}) = {lp}, want 0");
        }
    }

    #[test]
    fn test_cb_log_prob_batched_probs() {
        // scalar value broadcasts against batched probs (the #1569 batch
        // contract). torch: CB([0.2,0.5,0.8]).log_prob(0.5) elementwise.
        let d = ContinuousBernoulli::new(from_slice(&[0.2f64, 0.5, 0.8], &[3]).unwrap()).unwrap();
        let lp = d.log_prob(&scalar(0.5f64).unwrap()).unwrap();
        assert_eq!(lp.shape(), &[3]);
        // CB(0.5).log_prob(0.5) == 0 (the middle element).
        assert!(lp.data().unwrap()[1].abs() < 1e-12);
    }

    #[test]
    fn test_cb_logits_accessor() {
        // torch: CB(0.3).logits == -0.8472978603872038 = ln(0.3) - log1p(-0.3).
        let d = ContinuousBernoulli::new(scalar(0.3f64).unwrap()).unwrap();
        let l = d.logits().unwrap().item().unwrap();
        assert!((l - (-0.847_297_860_387_203_8)).abs() < 1e-12, "got {l}");
    }

    #[test]
    fn test_cb_from_logits() {
        // logit 0 -> λ = 0.5 -> CB is the uniform; mean = 0.5.
        let d = ContinuousBernoulli::from_logits(scalar(0.0f64).unwrap()).unwrap();
        assert!((d.probs().item().unwrap() - 0.5).abs() < 1e-12);
        assert!((d.mean().unwrap().item().unwrap() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_cb_cdf_known() {
        // torch: CB(0.3).cdf(0.5) == 0.60435607626104;
        //        CB(0.5).cdf(0.4) == 0.4 (uniform cdf).
        let d = ContinuousBernoulli::new(scalar(0.3f64).unwrap()).unwrap();
        assert!(
            (d.cdf(&scalar(0.5f64).unwrap()).unwrap().item().unwrap() - 0.604_356_076_261_04).abs()
                < 1e-12
        );
        let d5 = ContinuousBernoulli::new(scalar(0.5f64).unwrap()).unwrap();
        assert!((d5.cdf(&scalar(0.4f64).unwrap()).unwrap().item().unwrap() - 0.4).abs() < 1e-12);
    }

    #[test]
    fn test_cb_icdf_inverts_cdf() {
        // torch: CB(0.3).icdf(0.5) == 0.39711210467054603; and icdf∘cdf == id.
        let d = ContinuousBernoulli::new(scalar(0.3f64).unwrap()).unwrap();
        let q = d.icdf(&scalar(0.5f64).unwrap()).unwrap().item().unwrap();
        assert!((q - 0.397_112_104_670_546_03).abs() < 1e-12, "got {q}");
        // round-trip: cdf(icdf(0.5)) ≈ 0.5.
        let back = d.cdf(&scalar(q).unwrap()).unwrap().item().unwrap();
        assert!((back - 0.5).abs() < 1e-10, "round-trip got {back}");
    }

    #[test]
    fn test_cb_batched_mean_variance() {
        // torch: CB([0.2,0.5,0.8]).mean == [0.3880141871111483, 0.5,
        //        0.6119858128888517]; .variance == [0.0758978008069574,
        //        0.08333333333333333, 0.07589780080695757].
        let d = ContinuousBernoulli::new(from_slice(&[0.2f64, 0.5, 0.8], &[3]).unwrap()).unwrap();
        let m = d.mean().unwrap();
        let v = d.variance().unwrap();
        assert_eq!(m.shape(), &[3]);
        let md = m.data().unwrap();
        let vd = v.data().unwrap();
        assert!((md[0] - 0.388_014_187_111_148_3).abs() < 1e-12);
        assert!((md[1] - 0.5).abs() < 1e-15);
        assert!((md[2] - 0.611_985_812_888_851_7).abs() < 1e-12);
        assert!((vd[0] - 0.075_897_800_806_957_4).abs() < 1e-12);
        assert!((vd[1] - (1.0 / 12.0)).abs() < 1e-15);
    }

    #[test]
    fn test_cb_sample_in_support() {
        let d = ContinuousBernoulli::new(scalar(0.3f32).unwrap()).unwrap();
        let s = d.sample(&[500]).unwrap();
        assert_eq!(s.shape(), &[500]);
        assert!(!s.requires_grad());
        for &x in s.data().unwrap() {
            assert!((0.0..=1.0).contains(&x), "CB sample out of [0,1]: {x}");
        }
    }

    #[test]
    fn test_cb_sample_batched_shape() {
        // batch_shape=[2], sample_shape=[5] -> _extended_shape = [5,2].
        let d = ContinuousBernoulli::new(from_slice(&[0.3f32, 0.7], &[2]).unwrap()).unwrap();
        let s = d.sample(&[5]).unwrap();
        assert_eq!(s.shape(), &[5, 2]);
    }

    #[test]
    fn test_cb_rsample_in_support() {
        let d = ContinuousBernoulli::new(scalar(0.6f64).unwrap()).unwrap();
        let s = d.rsample(&[200]).unwrap();
        assert_eq!(s.shape(), &[200]);
        for &x in s.data().unwrap() {
            assert!((0.0..=1.0).contains(&x), "CB rsample out of [0,1]: {x}");
        }
    }

    #[test]
    fn test_cb_has_rsample_and_support() {
        let d = ContinuousBernoulli::new(scalar(0.5f64).unwrap()).unwrap();
        assert!(d.has_rsample());
        let s = d.support().unwrap();
        assert_eq!(s.name(), "UnitInterval");
        assert!(!s.is_discrete());
    }

    #[test]
    fn test_cb_arg_constraints() {
        let d = ContinuousBernoulli::new(scalar(0.5f64).unwrap()).unwrap();
        let ac = d.arg_constraints();
        assert_eq!(ac.get("probs").unwrap().name(), "UnitInterval");
        assert_eq!(ac.get("logits").unwrap().name(), "Real");
    }

    #[test]
    fn test_cb_batch_shape() {
        let d = ContinuousBernoulli::new(from_slice(&[0.3f64, 0.5], &[2]).unwrap()).unwrap();
        assert_eq!(d.batch_shape(), vec![2]);
    }

    #[test]
    fn test_cb_f32() {
        let d = ContinuousBernoulli::new(scalar(0.3f32).unwrap()).unwrap();
        // mean ≈ 0.4302.
        assert!((d.mean().unwrap().item().unwrap() - 0.430_222_5).abs() < 1e-4);
        let lp = d.log_prob(&scalar(0.25f32).unwrap()).unwrap();
        assert!(lp.item().unwrap().is_finite());
    }
}
