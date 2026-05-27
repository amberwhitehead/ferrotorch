//! Geometric distribution.
//!
//! `Geometric(probs)` defines a distribution over the non-negative integers
//! `{0, 1, 2, ...}`, where the sampled value `k` is the **number of failures**
//! before the first success in a sequence of independent Bernoulli trials with
//! per-trial success probability `probs`. This convention matches
//! `torch.distributions.Geometric` (`torch/distributions/geometric.py:20-44`):
//! the `(k+1)`-th trial is the first success, so `P(X=k) = (1-p)^k · p`.
//!
//! It is a discrete distribution and does not support reparameterized sampling.
//!
//! ## REQ status (per `.design/ferrotorch-distributions/geometric.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`Geometric<T>` struct) | SHIPPED | `pub struct Geometric<T: Float>` with `probs`/`batch_shape` fields (`batch_shape = probs.shape()`, `geometric.py:68-74`) mirroring `torch/distributions/geometric.py:20-87`; consumer: `pub use geometric::Geometric` in `lib.rs` (boundary public API per goal.md S5) + `kl_geometric_geometric` arm in `kl.rs`. |
//! | REQ-2 (constructors) | SHIPPED | `pub fn Geometric::new` (probs) + `pub fn Geometric::from_logits` (binary sigmoid `p = 1/(1+exp(-l))` per `logits_to_probs(is_binary=True)` `utils.py:97-98`) mirroring `geometric.py:50-74`; consumer: `kl_geometric_geometric` in `kl.rs` reaches instances via `kl_divergence`; `pub use Geometric` re-export. |
//! | REQ-3 (accessors) | SHIPPED | `pub fn Geometric::{probs, logits}` mirroring the `@lazy_property` pair at `geometric.py:112-118` (`logits = probs_to_logits(probs, is_binary=True) = ln(p) - log1p(-p)`); consumer: `kl_geometric_geometric` reads `p.probs()` / `q.probs()` and recomputes `q.logits()` in `kl.rs`. |
//! | REQ-4 (`Distribution` impl) | SHIPPED | `impl<T: Float> Distribution<T> for Geometric<T>` (`sample`/`rsample`/`log_prob`/`entropy`) mirroring `geometric.py:120-144`; `sample` returns `_extended_shape = sample_shape ++ batch_shape` (`distribution.py:266-278`); consumer: trait surface via `pub use Geometric`; `test_geometric_sample_*`. |
//! | REQ-5 (`rsample` rejection) | SHIPPED | `fn Geometric::rsample` returns `InvalidArgument` (Geometric is discrete: `geometric.py` declares no `has_rsample`); consumer: trait surface; `test_geometric_rsample_errors`. |
//! | REQ-6 (`log_prob`) | SHIPPED | `fn Geometric::log_prob` = `k·log1p(-p) + ln(p)` with the `probs==1 & value==0 -> 0` clamp mirroring `geometric.py:132-138`; `value` broadcasts against `batch_shape` (right-aligned, NOT `cycle()`); consumer: trait surface; `test_geometric_log_prob_*` + `test_geometric_log_prob_batched_probs`. |
//! | REQ-7 (`entropy`) | SHIPPED | `fn Geometric::entropy` = `BCE_with_logits(logits, probs)/probs` mirroring `geometric.py:140-144`; consumer: trait surface; `test_geometric_entropy_known`. |
//! | REQ-8 (`mean`/`variance`/`mode`) | SHIPPED | `fn Geometric::{mean, variance, mode}` = `(1-p)/p` / `(1-p)/p²` / `0` mirroring `geometric.py:100-110,104-106`; consumer: trait overrides via `pub use Geometric`; `test_geometric_mean_variance`. |
//! | REQ-9 (full surface) | SHIPPED | `has_rsample`/`support` (`NonNegativeInteger`)/`arg_constraints` (`{probs: unit_interval, logits: real}`)/`event_shape` overrides mirroring `geometric.py:47-48`; consumer: `pub use Geometric`; `test_geometric_support` + `test_geometric_arg_constraints`. |

use std::collections::HashMap;

use ferrotorch_core::creation;
use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

use crate::constraints;
use crate::{DistConstraint, Distribution};

/// Row-major strides for `shape` (the number of flat elements one step along
/// each axis advances), used for broadcast index arithmetic. Mirrors the
/// helper in `binomial.rs` (the FIXED batch-broadcast reference).
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
/// Mirrors how `torch.distributions.Geometric` broadcasts `probs` (sized at
/// `batch_shape`) against a `value`/output sized at the broadcast of
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

/// Geometric distribution parameterized by `probs` (per-trial success
/// probability). The sampled value is the number of failures before the first
/// success — support `{0, 1, 2, ...}`.
///
/// # Discrete
///
/// This is a discrete distribution. `rsample` returns an error because there
/// is no continuous reparameterization for Geometric. Use `sample` and
/// score-function estimators (REINFORCE) for gradient-based optimization.
///
/// # Batch shape
///
/// `probs` defines the `batch_shape` directly (`torch/distributions/geometric.py:68-74`,
/// `batch_shape = probs.size()`). `sample` and `log_prob` honour this batch
/// shape, matching PyTorch's `Distribution._extended_shape`
/// (`torch/distributions/distribution.py:266-278`); `value` in `log_prob` is
/// right-aligned-broadcast against `batch_shape` exactly like
/// `broadcast_all(value, self.probs)` (`geometric.py:135`).
pub struct Geometric<T: Float> {
    probs: Tensor<T>,
    /// The distribution's `batch_shape`, equal to `probs.shape()` per
    /// `torch/distributions/geometric.py:74`.
    batch_shape: Vec<usize>,
}

impl<T: Float> Geometric<T> {
    /// Create a new Geometric distribution from `probs`.
    ///
    /// Each element of `probs` is the per-trial success probability in
    /// `(0, 1]`. Mirrors the `probs`-parameterized branch of
    /// `torch/distributions/geometric.py:60-62` (with `batch_shape =
    /// probs.size()` at `geometric.py:74`).
    pub fn new(probs: Tensor<T>) -> FerrotorchResult<Self> {
        let batch_shape = probs.shape().to_vec();
        Ok(Self { probs, batch_shape })
    }

    /// Create a new Geometric distribution from binary `logits`.
    ///
    /// `logits` are the event log-odds; the success probability is recovered
    /// via the binary sigmoid `p = 1 / (1 + exp(-logit))`
    /// (`logits_to_probs(logits, is_binary=True)` = `torch.sigmoid(logits)`,
    /// `torch/distributions/utils.py:97-98`). Mirrors the
    /// `logits`-parameterized branch of `geometric.py:63-67` + the
    /// `@lazy_property probs` at `geometric.py:116-118`.
    pub fn from_logits(logits: Tensor<T>) -> FerrotorchResult<Self> {
        crate::fallback::check_gpu_fallback_opt_in(&[&logits], "Geometric::from_logits")?;
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
        let batch_shape = probs.shape().to_vec();
        Ok(Self { probs, batch_shape })
    }

    /// The per-trial success probability.
    pub fn probs(&self) -> &Tensor<T> {
        &self.probs
    }

    /// The event log-odds, recomputed from `probs` via
    /// `probs_to_logits(probs, is_binary=True) = ln(p) - log1p(-p)`
    /// (`torch/distributions/utils.py:135-137`). Mirrors the
    /// `@lazy_property logits` at `geometric.py:112-114`.
    pub fn logits(&self) -> FerrotorchResult<Tensor<T>> {
        let one = <T as num_traits::One>::one();
        // `probs_to_logits` clamps with `clamp_probs`, i.e.
        // `eps = torch.finfo(dtype).eps` (`utils.py:124`) = `T::epsilon()`.
        let eps = <T as num_traits::Float>::epsilon();
        let probs_data = self.probs.data_vec()?;
        let out: Vec<T> = probs_data
            .iter()
            .map(|&p| {
                let pc = p.max(eps).min(one - eps);
                // ln(p) - log1p(-p); log1p(-pc) = ln(1-pc).
                pc.ln() - (-pc).ln_1p()
            })
            .collect();
        Tensor::from_storage(TensorStorage::cpu(out), self.probs.shape().to_vec(), false)
    }

    /// Evaluate a per-element closed form `f(p)` over the `batch_shape`. Used
    /// by `mean`/`variance`/`mode` (`geometric.py:100-110`), whose outputs are
    /// sized at `batch_shape`.
    fn map_batch(&self, f: impl Fn(T) -> T) -> FerrotorchResult<Tensor<T>> {
        let probs_data = self.probs.data_vec()?;
        let result: Vec<T> = probs_data.iter().map(|&p| f(p)).collect();
        Tensor::from_storage(TensorStorage::cpu(result), self.batch_shape.clone(), false)
    }
}

impl<T: Float> Distribution<T> for Geometric<T> {
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.probs], "Geometric::sample")?;
        // Inverse-CDF sampling: `(u.log() / (-p).log1p()).floor()` where `u ~
        // Uniform(tiny, 1)` (`geometric.py:120-130`). `(-p).log1p() = ln(1-p)`
        // is the log of the per-trial failure probability; flooring the ratio
        // gives the number of failures before the first success.
        //
        // `shape` is `_extended_shape(sample_shape) = sample_shape ++
        // batch_shape` (`distribution.py:266-278`); we draw uniforms over the
        // output and broadcast `probs` into it right-aligned.
        let device = self.probs.device();
        let probs_data = self.probs.data_vec()?;

        let mut out_shape: Vec<usize> = shape.to_vec();
        out_shape.extend_from_slice(&self.batch_shape);
        let n_out: usize = out_shape.iter().product::<usize>().max(1);

        let out_strides = row_major_strides(&out_shape);
        let probs_strides = row_major_strides(self.probs.shape());

        // `tiny = torch.finfo(dtype).tiny` (`geometric.py:122`) — the smallest
        // positive normal — keeps `u.log()` finite at the low end.
        let tiny = <T as num_traits::Float>::min_positive_value();
        let one = <T as num_traits::One>::one();
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
                let p = probs_data[pi];
                // `uniform_(tiny, 1)`: rescale rand() ∈ [0,1) onto [tiny, 1).
                let uu = (tiny + u[i] * (one - tiny)).max(tiny);
                (uu.ln() / (-p).ln_1p()).floor()
            })
            .collect();

        let out = Tensor::from_storage(TensorStorage::cpu(result), out_shape, false)?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }

    fn rsample(&self, _shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        Err(FerrotorchError::InvalidArgument {
            message: "Geometric distribution does not support reparameterized sampling. \
                      Use sample() with score-function estimators (REINFORCE) instead."
                .into(),
        })
    }

    fn log_prob(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.probs, value], "Geometric::log_prob")?;
        // log_prob(k) = k·log1p(-p) + ln(p) (`geometric.py:132-138`).
        // PyTorch additionally sets the term to 0 where `probs == 1 & value ==
        // 0` to avoid the `0·(-inf)` NaN in `value·log1p(-1)`; we reproduce that
        // by zeroing `log1p(-p)` for those elements. `value` is right-aligned
        // broadcast against `batch_shape`, mirroring `broadcast_all(value,
        // self.probs)` (`geometric.py:135`).
        let device = self.probs.device();
        let probs_data = self.probs.data_vec()?;
        let val_data = value.data_vec()?;
        let one = <T as num_traits::One>::one();
        let zero = <T as num_traits::Zero>::zero();

        let out_shape = ferrotorch_core::broadcast_shapes(value.shape(), &self.batch_shape)?;
        let n_out: usize = out_shape.iter().product::<usize>().max(1);
        let out_strides = row_major_strides(&out_shape);
        let value_strides = row_major_strides(value.shape());
        let probs_strides = row_major_strides(self.probs.shape());

        let result: Vec<T> = (0..n_out)
            .map(|i| {
                let ki = broadcast_flat_index(
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
                let k = val_data[ki];
                let p = probs_data[pi];
                // `probs[(probs == 1) & (value == 0)] = 0` -> the failure-log
                // term log1p(-p) is forced to 0 there (`geometric.py:137`).
                let log1p_neg_p = if p == one && k == zero {
                    zero
                } else {
                    (-p).ln_1p()
                };
                k * log1p_neg_p + p.ln()
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
        crate::fallback::check_gpu_fallback_opt_in(&[&self.probs], "Geometric::entropy")?;
        // entropy = binary_cross_entropy_with_logits(logits, probs) / probs
        // (`geometric.py:140-144`). With binary logit ℓ = ln(p) - log1p(-p)
        // and target t = p, BCE-with-logits per element is
        //   max(ℓ,0) - ℓ·t + log1p(exp(-|ℓ|))
        // (the numerically stable form torch uses internally). Dividing by p
        // recovers the closed-form Geometric entropy.
        let probs_data = self.probs.data_vec()?;
        let one = <T as num_traits::One>::one();
        let zero = <T as num_traits::Zero>::zero();
        let eps = <T as num_traits::Float>::epsilon();

        let result: Vec<T> = probs_data
            .iter()
            .map(|&p| {
                let pc = p.max(eps).min(one - eps);
                let logit = pc.ln() - (-pc).ln_1p();
                let abs_l = logit.abs();
                let max_l0 = if logit > zero { logit } else { zero };
                // stable BCE-with-logits: max(ℓ,0) - ℓ·t + log1p(exp(-|ℓ|)).
                let bce = max_l0 - logit * p + (-abs_l).exp().ln_1p();
                bce / p
            })
            .collect();

        Tensor::from_storage(TensorStorage::cpu(result), self.batch_shape.clone(), false)
    }

    fn mean(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.probs], "Geometric::mean")?;
        // mean = 1/p - 1 = (1-p)/p (`geometric.py:100-102`), sized at batch_shape.
        let one = <T as num_traits::One>::one();
        self.map_batch(move |p| one / p - one)
    }

    fn variance(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.probs], "Geometric::variance")?;
        // variance = (1/p - 1)/p = (1-p)/p² (`geometric.py:108-110`).
        let one = <T as num_traits::One>::one();
        self.map_batch(move |p| (one / p - one) / p)
    }

    fn mode(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.probs], "Geometric::mode")?;
        // mode = 0 always (`geometric.py:104-106`: `torch.zeros_like(probs)`).
        let zero = <T as num_traits::Zero>::zero();
        self.map_batch(move |_p| zero)
    }

    // -----------------------------------------------------------------------
    // Full PyTorch surface — Geometric is discrete (no rsample), has infinite
    // (non-enumerable) support {0,1,2,...}, and declares
    // (probs: UnitInterval, logits: Real) arg_constraints with
    // support = nonnegative_integer. Mirrors
    // `torch/distributions/geometric.py:46-48`.
    // -----------------------------------------------------------------------

    fn has_rsample(&self) -> bool {
        // `geometric.py` has no `has_rsample` class attr → inherits default false.
        false
    }

    fn support(&self) -> Option<Box<dyn DistConstraint>> {
        // `geometric.py:48`: `support = constraints.nonnegative_integer`.
        Some(Box::new(constraints::NonNegativeInteger))
    }

    fn arg_constraints(&self) -> HashMap<&'static str, Box<dyn DistConstraint>> {
        // `geometric.py:47`:
        //   {"probs": constraints.unit_interval, "logits": constraints.real}
        let mut m: HashMap<&'static str, Box<dyn DistConstraint>> = HashMap::new();
        m.insert("probs", Box::new(constraints::UnitInterval));
        m.insert("logits", Box::new(constraints::Real));
        m
    }

    fn event_shape(&self) -> Vec<usize> {
        // Geometric is univariate (each draw is a single integer count).
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

    // Reference values from live `torch.distributions.Geometric` (torch
    // 2.11.0+cu130, this machine 2026-05-27); each constant is independently
    // re-derivable and traces to a geometric.py:line (R-CHAR-3 non-tautological).

    #[test]
    fn test_geometric_log_prob_known() {
        // torch: Geometric(torch.tensor(0.3,dtype=torch.float64))
        //        .log_prob(torch.tensor(2.0,dtype=torch.float64))
        //        == -1.9173226922034008 (geometric.py:132-138).
        // Closed form: 2·ln(0.7) + ln(0.3).
        let dist = Geometric::new(scalar(0.3f64).unwrap()).unwrap();
        let lp = dist.log_prob(&scalar(2.0f64).unwrap()).unwrap();
        assert!(
            (lp.item().unwrap() - (-1.917_322_692_203_400_8)).abs() < 1e-12,
            "expected torch value -1.9173226922034008, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_geometric_log_prob_k0() {
        // Geometric(0.4).log_prob(0.) == ln(0.4) (no failures before success).
        let dist = Geometric::new(scalar(0.4f64).unwrap()).unwrap();
        let lp = dist.log_prob(&scalar(0.0f64).unwrap()).unwrap();
        assert!(
            (lp.item().unwrap() - 0.4f64.ln()).abs() < 1e-12,
            "expected ln(0.4), got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_geometric_log_prob_p1_k0() {
        // torch: Geometric(1.0).log_prob(0.) == 0.0 (the probs==1 & value==0
        // clamp at geometric.py:137 avoids the 0·(-inf) NaN).
        let dist = Geometric::new(scalar(1.0f64).unwrap()).unwrap();
        let lp = dist.log_prob(&scalar(0.0f64).unwrap()).unwrap();
        assert_eq!(lp.item().unwrap(), 0.0);
    }

    #[test]
    fn test_geometric_log_prob_batched_probs() {
        // torch: Geometric([0.3,0.5]).log_prob(2.0) (scalar value broadcasts
        //   against batched probs) == [-1.9173226922034008, -2.0794415416798357].
        // This is the batch-broadcast contract that the Binomial critic (#1569)
        // flagged Bernoulli for getting wrong with iter().cycle().
        let dist = Geometric::new(from_slice(&[0.3f64, 0.5], &[2]).unwrap()).unwrap();
        let lp = dist.log_prob(&scalar(2.0f64).unwrap()).unwrap();
        assert_eq!(lp.shape(), &[2]);
        let d = lp.data().unwrap();
        assert!(
            (d[0] - (-1.917_322_692_203_400_8)).abs() < 1e-12,
            "probs=0.3 element: expected -1.9173226922034008, got {}",
            d[0]
        );
        assert!(
            (d[1] - (-2.079_441_541_679_835_7)).abs() < 1e-12,
            "probs=0.5 element: expected -2.0794415416798357, got {}",
            d[1]
        );
    }

    #[test]
    fn test_geometric_log_prob_batched_value_and_probs() {
        // torch: Geometric([0.3,0.5]).log_prob([1.0,3.0])
        //   == [-1.5606477482646683, -2.772588722239781].
        let dist = Geometric::new(from_slice(&[0.3f64, 0.5], &[2]).unwrap()).unwrap();
        let lp = dist
            .log_prob(&from_slice(&[1.0f64, 3.0], &[2]).unwrap())
            .unwrap();
        let d = lp.data().unwrap();
        assert!(
            (d[0] - (-1.560_647_748_264_668_3)).abs() < 1e-12,
            "got {}",
            d[0]
        );
        assert!(
            (d[1] - (-2.772_588_722_239_781)).abs() < 1e-12,
            "got {}",
            d[1]
        );
    }

    #[test]
    fn test_geometric_mean_variance() {
        // torch: Geometric(0.3).mean == 2.3333333333333335;
        //        Geometric(0.3).variance == 7.777777777777779.
        // mean = (1-p)/p = 0.7/0.3; var = (1-p)/p² = 0.7/0.09.
        let dist = Geometric::new(scalar(0.3f64).unwrap()).unwrap();
        assert!(
            (dist.mean().unwrap().item().unwrap() - 2.333_333_333_333_333_5).abs() < 1e-12,
            "mean got {}",
            dist.mean().unwrap().item().unwrap()
        );
        assert!(
            (dist.variance().unwrap().item().unwrap() - 7.777_777_777_777_779).abs() < 1e-12,
            "variance got {}",
            dist.variance().unwrap().item().unwrap()
        );
    }

    #[test]
    fn test_geometric_mean_variance_batched() {
        // torch: Geometric([0.3,0.5]).mean == [2.3333333333333335, 1.0];
        //        .variance == [7.777777777777779, 2.0].
        let dist = Geometric::new(from_slice(&[0.3f64, 0.5], &[2]).unwrap()).unwrap();
        let m = dist.mean().unwrap();
        let v = dist.variance().unwrap();
        assert_eq!(m.shape(), &[2]);
        let md = m.data().unwrap();
        let vd = v.data().unwrap();
        assert!((md[0] - 2.333_333_333_333_333_5).abs() < 1e-12);
        assert!((md[1] - 1.0).abs() < 1e-12);
        assert!((vd[0] - 7.777_777_777_777_779).abs() < 1e-12);
        assert!((vd[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_geometric_mode() {
        // torch: Geometric(p).mode == 0 for any p (geometric.py:104-106).
        let dist = Geometric::new(scalar(0.3f64).unwrap()).unwrap();
        assert_eq!(dist.mode().unwrap().item().unwrap(), 0.0);
    }

    #[test]
    fn test_geometric_entropy_known() {
        // torch: Geometric(0.3).entropy() == 2.0362143401829784 (geometric.py:140-144).
        let dist = Geometric::new(scalar(0.3f64).unwrap()).unwrap();
        let h = dist.entropy().unwrap();
        assert!(
            (h.item().unwrap() - 2.036_214_340_182_978_4).abs() < 1e-10,
            "expected torch value 2.0362143401829784, got {}",
            h.item().unwrap()
        );
    }

    #[test]
    fn test_geometric_entropy_batched() {
        // torch: Geometric([0.3,0.5]).entropy()
        //   == [2.0362143401829784, 1.3862943611198906].
        let dist = Geometric::new(from_slice(&[0.3f64, 0.5], &[2]).unwrap()).unwrap();
        let h = dist.entropy().unwrap();
        assert_eq!(h.shape(), &[2]);
        let d = h.data().unwrap();
        assert!(
            (d[0] - 2.036_214_340_182_978_4).abs() < 1e-10,
            "got {}",
            d[0]
        );
        assert!(
            (d[1] - 1.386_294_361_119_890_6).abs() < 1e-10,
            "got {}",
            d[1]
        );
    }

    #[test]
    fn test_geometric_from_logits() {
        // torch: Geometric(logits=0.).probs == 0.5; .mean == 1.0.
        let dist = Geometric::from_logits(scalar(0.0f64).unwrap()).unwrap();
        assert!((dist.probs().item().unwrap() - 0.5).abs() < 1e-12);
        assert!((dist.mean().unwrap().item().unwrap() - 1.0).abs() < 1e-12);
        // logits() round-trips back to ~0 for p=0.5.
        assert!(dist.logits().unwrap().item().unwrap().abs() < 1e-9);
    }

    #[test]
    fn test_geometric_logits_accessor() {
        // torch: Geometric(0.8).logits == 1.3862943611198908.
        // logit = ln(0.8) - log1p(-0.8) = ln(0.8) - ln(0.2) = ln(4).
        let dist = Geometric::new(scalar(0.8f64).unwrap()).unwrap();
        let l = dist.logits().unwrap().item().unwrap();
        assert!(
            (l - 1.386_294_361_119_890_8).abs() < 1e-9,
            "expected torch value 1.3862943611198908 (= ln 4), got {l}"
        );
    }

    #[test]
    fn test_geometric_sample_shape() {
        let dist = Geometric::new(scalar(0.5f32).unwrap()).unwrap();
        let samples = dist.sample(&[100]).unwrap();
        assert_eq!(samples.shape(), &[100]);
        assert!(!samples.requires_grad());
    }

    #[test]
    fn test_geometric_sample_batched_shape() {
        // batch_shape=[2], sample_shape=[5] -> _extended_shape = [5,2].
        let dist = Geometric::new(from_slice(&[0.3f32, 0.7], &[2]).unwrap()).unwrap();
        let samples = dist.sample(&[5]).unwrap();
        assert_eq!(samples.shape(), &[5, 2]);
    }

    #[test]
    fn test_geometric_sample_in_support() {
        let dist = Geometric::new(scalar(0.5f32).unwrap()).unwrap();
        let samples = dist.sample(&[500]).unwrap();
        let data = samples.data().unwrap();
        for &x in data {
            assert!(
                x >= 0.0 && x.fract() == 0.0,
                "Geometric sample must be a non-negative integer, got {x}"
            );
        }
    }

    #[test]
    fn test_geometric_sample_prob_1() {
        // p=1 -> first trial always succeeds -> all samples are 0.
        let dist = Geometric::new(scalar(1.0f32).unwrap()).unwrap();
        let data = dist.sample(&[64]).unwrap().data().unwrap().to_vec();
        assert!(data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_geometric_rsample_errors() {
        let dist = Geometric::new(scalar(0.5f32).unwrap()).unwrap();
        assert!(dist.rsample(&[5]).is_err());
    }

    #[test]
    fn test_geometric_support() {
        let dist = Geometric::new(scalar(0.5f64).unwrap()).unwrap();
        let support = dist.support().unwrap();
        assert_eq!(support.name(), "NonNegativeInteger");
        assert!(support.is_discrete());
    }

    #[test]
    fn test_geometric_arg_constraints() {
        let dist = Geometric::new(scalar(0.5f64).unwrap()).unwrap();
        let ac = dist.arg_constraints();
        assert_eq!(ac.get("probs").unwrap().name(), "UnitInterval");
        assert_eq!(ac.get("logits").unwrap().name(), "Real");
    }

    #[test]
    fn test_geometric_batch_shape() {
        let dist = Geometric::new(from_slice(&[0.3f64, 0.5, 0.7], &[3]).unwrap()).unwrap();
        assert_eq!(dist.batch_shape(), vec![3]);
    }

    #[test]
    fn test_geometric_f32() {
        let dist = Geometric::new(scalar(0.4f32).unwrap()).unwrap();
        let samples = dist.sample(&[40]).unwrap();
        assert_eq!(samples.shape(), &[40]);
        // mean = (1-0.4)/0.4 = 1.5.
        assert!((dist.mean().unwrap().item().unwrap() - 1.5).abs() < 1e-5);
        let lp = dist.log_prob(&scalar(1.0f32).unwrap()).unwrap();
        assert!(lp.item().unwrap() < 0.0 && lp.item().unwrap().is_finite());
    }
}
