//! Relaxed one-hot categorical (Concrete) distribution.
//!
//! `RelaxedOneHotCategorical(temperature, probs)` is a continuous relaxation
//! of [`OneHotCategorical`](crate::OneHotCategorical) over the open
//! probability simplex via the Gumbel-softmax trick (Maddison et al. 2017,
//! Jang et al. 2017). Samples are points in the open `K-1` simplex, not
//! discrete one-hot vectors.
//!
//! As `temperature → 0`, samples concentrate on the corners of the simplex
//! and recover the discrete OneHotCategorical. As `temperature → ∞`,
//! samples approach the uniform distribution on the simplex.
//!
//! # Reparameterization
//!
//! Sampling is reparameterizable via Gumbel noise:
//! ```text
//! g_i ~ Gumbel(0, 1)            (i.e. g_i = -log(-log(U_i)), U_i ~ Uniform)
//! z_i = exp((log(probs_i) + g_i) / temperature)
//! z = z / sum_j z_j             (softmax over the K dimensions)
//! ```
//! Mirrors `torch.distributions.RelaxedOneHotCategorical`.
//!
//! ## REQ status (per `.design/ferrotorch-distributions/relaxed_one_hot_categorical.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`RelaxedOneHotCategorical` struct) | SHIPPED | `pub struct RelaxedOneHotCategorical` in `relaxed_one_hot_categorical.rs`; re-exported as `pub use relaxed_one_hot_categorical::RelaxedOneHotCategorical` in `lib.rs:119`; mirrors `torch/distributions/relaxed_categorical.py:109-135`. |
//! | REQ-2 (`new` constructor with simplex normalization) | SHIPPED | `RelaxedOneHotCategorical::new` validating + caching normalized probs in `relaxed_one_hot_categorical.rs`; registered in `tests/conformance/_surface_inventory.toml:455`. |
//! | REQ-3 (accessors `temperature`/`probs`/`num_categories`) | SHIPPED | accessors in `relaxed_one_hot_categorical.rs`; mirror `relaxed_categorical.py:153-163` @property delegations. |
//! | REQ-4 (`Distribution::sample` / `rsample` via Gumbel-softmax) | SHIPPED | `impl Distribution::sample` / `rsample` invoke `relaxed_one_hot_sample` (Gumbel-softmax forward) in `relaxed_one_hot_categorical.rs`; mirrors `ExpRelaxedCategorical.rsample` + `ExpTransform` composition at `relaxed_categorical.py:87-94`. |
//! | REQ-5 (`Distribution::log_prob` via Maddison eqn 26 + logsumexp) | SHIPPED | `impl Distribution::log_prob` in `relaxed_one_hot_categorical.rs` with logsumexp + last-K-dim collapse; rejects wrong-shape value. |
//! | REQ-6 (`Distribution::entropy` errors) | SHIPPED | `impl Distribution::entropy` returns `InvalidArgument` (Concrete has no closed-form entropy). |
//! | REQ-7 (`logits` accessor + `support`/`arg_constraints`/`has_rsample`/`expand`) | SHIPPED | `pub fn logits` returns `log(probs)` (normalised); `fn support` returns `Simplex`; `fn arg_constraints` declares `probs: Simplex`; `fn has_rsample` returns `true`; `fn expand` broadcasts `probs`. Mirrors `torch/distributions/relaxed_categorical.py:117-135`. Non-test consumer: `pub use RelaxedOneHotCategorical` re-export. `mean`/`mode`/`variance` have no closed form for the Concrete relaxation (upstream raises `NotImplementedError`). Closes #1422. |
//! | REQ-8 (`ExpRelaxedCategorical` as standalone) | SHIPPED | `pub struct ExpRelaxedCategorical<T>` — the log-space relaxed categorical whose `rsample` returns `scores - logsumexp(scores)` (a point in the log-simplex) per Gumbel-softmax; re-exported as `pub use relaxed_one_hot_categorical::ExpRelaxedCategorical` in `lib.rs`. Mirrors `torch/distributions/relaxed_categorical.py:17-106`. `RelaxedOneHotCategorical::rsample` consumes it by exponentiating its log-space draw (the `ExpTransform` composition), so the standalone type has a non-test production consumer in this module. Closes #1424. |
//! | REQ-9 (differentiable `rsample`) | SHIPPED | both `ExpRelaxedCategorical::rsample` and `RelaxedOneHotCategorical::rsample` build autograd nodes (`ExpRelaxedRsampleBackward` / `RelaxedOneHotRsampleBackward`) so gradients flow through `probs` (the only `Tensor` parameter). Gumbel-softmax gradient: `dz_i/dp_m = z_i(δ_im - z_m) / (temp * p_m)`. Mirrors `relaxed_categorical.py:87-94`. Consumer: `impl Distribution::rsample` is the production dispatch via `pub use RelaxedOneHotCategorical` at `lib.rs:119`. Closes #1425. |
//! | REQ-10 (batched `probs` with leading dims) | SHIPPED | `new` accepts `probs` of shape `[..., K]`; `normalized` is row-normalized per trailing-K group; `sample`/`rsample` emit `[...sample_shape, ...batch, K]`; `log_prob` collapses the trailing K dim leaving `[..., ...batch]`. Mirrors upstream's arbitrary leading batch dims (`relaxed_categorical.py:56-57` `event_shape = param_shape[-1:]`). Consumer: `impl Distribution` dispatch + `pub use` re-export. Closes #1426. |

use std::collections::HashMap;
use std::sync::Arc;

use ferrotorch_core::creation;
use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::{GradFn, Tensor};

use crate::constraints;
use crate::{DistConstraint, Distribution};

/// Continuous relaxation of a categorical distribution.
///
/// Supports batched parameters: `probs` may have shape `[..., K]` where the
/// trailing dim is the category axis and the leading dims form the batch.
pub struct RelaxedOneHotCategorical<T: Float> {
    temperature: T,
    probs: Tensor<T>,
    /// Cached per-row normalized probabilities, flat over all batch rows
    /// (length = `num_rows * num_categories`).
    normalized: Vec<T>,
    /// Number of categories (trailing dim of `probs`).
    num_categories: usize,
    /// Leading batch dims of `probs` (all dims except the trailing K).
    batch: Vec<usize>,
}

/// Validate temperature + probs and produce the per-row-normalized cache.
///
/// Shared by [`RelaxedOneHotCategorical`] and [`ExpRelaxedCategorical`].
/// Returns `(num_categories, batch_dims, normalized_flat)`.
fn validate_and_normalize<T: Float>(
    temperature: T,
    probs: &Tensor<T>,
    who: &str,
) -> FerrotorchResult<(usize, Vec<usize>, Vec<T>)> {
    let zero = <T as num_traits::Zero>::zero();
    if temperature <= zero {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("{who}: temperature must be > 0"),
        });
    }
    if probs.ndim() == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("{who}: probs must have at least 1 dim (trailing K), got scalar"),
        });
    }
    let shape = probs.shape().to_vec();
    let k = *shape.last().unwrap();
    if k == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("{who}: probs must have at least one category"),
        });
    }
    let batch: Vec<usize> = shape[..shape.len() - 1].to_vec();

    let probs_data = probs.data_vec()?;
    for (i, &p) in probs_data.iter().enumerate() {
        if p <= zero {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "{who}: probs[{i}] = {} must be > 0",
                    p.to_f64().unwrap_or(f64::NAN)
                ),
            });
        }
    }
    // Per-row normalization: each contiguous K-block sums to 1.
    let num_rows = probs_data.len() / k;
    let mut normalized = Vec::with_capacity(probs_data.len());
    for r in 0..num_rows {
        let base = r * k;
        let total: T = probs_data[base..base + k]
            .iter()
            .copied()
            .fold(zero, |a, b| a + b);
        for j in 0..k {
            normalized.push(probs_data[base + j] / total);
        }
    }
    Ok((k, batch, normalized))
}

impl<T: Float> RelaxedOneHotCategorical<T> {
    /// Construct a RelaxedOneHotCategorical with the given temperature
    /// and unnormalized class probabilities.
    ///
    /// `probs` may have shape `[..., K]`; the trailing dim is the category
    /// axis, leading dims are the batch. All entries must be strictly
    /// positive.
    pub fn new(temperature: T, probs: Tensor<T>) -> FerrotorchResult<Self> {
        let (k, batch, normalized) =
            validate_and_normalize(temperature, &probs, "RelaxedOneHotCategorical")?;
        Ok(Self {
            temperature,
            probs,
            normalized,
            num_categories: k,
            batch,
        })
    }

    /// Temperature parameter.
    pub fn temperature(&self) -> T {
        self.temperature
    }

    /// (Unnormalized) probabilities.
    pub fn probs(&self) -> &Tensor<T> {
        &self.probs
    }

    /// Number of categories.
    pub fn num_categories(&self) -> usize {
        self.num_categories
    }

    /// The logits parameter `log(normalized_probs)`.
    ///
    /// Mirrors `torch.distributions.RelaxedOneHotCategorical.logits`
    /// (`torch/distributions/relaxed_categorical.py:158-160`), which
    /// delegates to `ExpRelaxedCategorical.logits` (log-probabilities of the
    /// normalised distribution).
    pub fn logits(&self) -> FerrotorchResult<Tensor<T>> {
        let device = self.probs.device();
        let out: Vec<T> = self.normalized.iter().map(|&p| p.ln()).collect();
        let t = Tensor::from_storage(TensorStorage::cpu(out), self.probs.shape().to_vec(), false)?;
        if device.is_cuda() {
            t.to(device)
        } else {
            Ok(t)
        }
    }
}

impl<T: Float> Distribution<T> for RelaxedOneHotCategorical<T> {
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.probs],
            "RelaxedOneHotCategorical::sample",
        )?;
        // RelaxedOneHotCategorical = ExpTransform ∘ ExpRelaxedCategorical
        // (`relaxed_categorical.py:144-147`). Production-consume the standalone
        // `ExpRelaxedCategorical` (#1424): draw a log-simplex point, then map
        // through `exp` to land on the simplex. `exp(log-simplex)` sums to 1
        // exactly (up to f32 rounding) because the base draw is
        // `scores - logsumexp(scores)`.
        let base = ExpRelaxedCategorical {
            temperature: self.temperature,
            probs: self.probs.clone(),
            normalized: self.normalized.clone(),
            num_categories: self.num_categories,
            batch: self.batch.clone(),
        };
        let log_z = base.sample(shape)?;
        let log_z_data = log_z.data_vec()?;
        let z: Vec<T> = log_z_data.iter().map(|&v| v.exp()).collect();
        let device = self.probs.device();
        let out = Tensor::from_storage(TensorStorage::cpu(z), log_z.shape().to_vec(), false)?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }

    fn rsample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.probs],
            "RelaxedOneHotCategorical::rsample",
        )?;
        // Differentiable Gumbel-softmax draw: gradient flows through `probs`
        // via `RelaxedOneHotRsampleBackward` (the only `Tensor` parameter;
        // `temperature` is a scalar `T`). See #1425.
        relaxed_one_hot_sample(
            self.temperature,
            &self.normalized,
            self.num_categories,
            &self.batch,
            &self.probs,
            shape,
            true,
        )
    }

    fn log_prob(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.probs, value],
            "RelaxedOneHotCategorical::log_prob",
        )?;
        // Concrete log density on the simplex (Maddison et al. 2017, eqn 26):
        //
        //   log p(z; alpha, lambda) = log((K-1)!) + (K-1) * log(lambda)
        //       + sum_k ( log(alpha_k) - (lambda + 1) * log(z_k) )
        //       - K * log( sum_k alpha_k * z_k^(-lambda) )
        //
        // where alpha_k = probs_k (normalized) and lambda = temperature.
        //
        // We use logs throughout for numerical stability.
        let v_shape = value.shape().to_vec();
        if v_shape.is_empty() || *v_shape.last().unwrap() != self.num_categories {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "RelaxedOneHotCategorical: log_prob value last dim must be K={}, got shape {:?}",
                    self.num_categories, v_shape
                ),
            });
        }

        let lambda = self.temperature;
        let one = <T as num_traits::One>::one();
        let zero = <T as num_traits::Zero>::zero();
        let neg_lambda = zero - lambda;
        let k = self.num_categories;
        let kt = T::from(k).unwrap();

        // log((K-1)!)
        let mut log_fact_km1 = zero;
        for i in 1..k {
            log_fact_km1 += T::from(i).unwrap().ln();
        }
        let km1 = T::from((k as i64 - 1).max(0)).unwrap();
        let log_lambda = lambda.ln();
        // Per-row log(alpha): `normalized` is flat over `num_param_rows`
        // K-blocks (one per batch row). For unbatched probs there is a
        // single row, reused for every value row.
        let log_alpha: Vec<T> = self.normalized.iter().map(|&p| p.ln()).collect();
        let num_param_rows = self.normalized.len() / k;
        let constant = log_fact_km1 + km1 * log_lambda;

        let v_data = value.data_vec()?;
        let n = v_data.len() / k;
        let mut result = Vec::with_capacity(n);
        let eps = T::from(1e-20).unwrap();

        for i in 0..n {
            let base = i * k;
            // Pick the parameter row this value row aligns with. The trailing
            // batch dims of `value` are the rightmost-after-K dims, so cycling
            // `i % num_param_rows` aligns value rows with parameter rows for
            // both unbatched (rows=1) and batched value tensors.
            let alpha_base = (i % num_param_rows) * k;
            // sum_k log(alpha_k) - (lambda + 1) * log(z_k)
            let mut linear = zero;
            // logsumexp over k of log(alpha_k) - lambda * log(z_k)
            // (for the - K * log(sum) term)
            let mut max_lse = T::neg_infinity();
            let mut tmp = vec![zero; k];
            for j in 0..k {
                let z = v_data[base + j].max(eps);
                let log_z = z.ln();
                let la = log_alpha[alpha_base + j];
                linear += la - (lambda + one) * log_z;
                let t = la + neg_lambda * log_z;
                tmp[j] = t;
                if t > max_lse {
                    max_lse = t;
                }
            }
            let mut sum_exp = zero;
            for &t in &tmp {
                sum_exp += (t - max_lse).exp();
            }
            let lse = max_lse + sum_exp.ln();
            result.push(constant + linear - kt * lse);
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
        Err(FerrotorchError::InvalidArgument {
            message: "RelaxedOneHotCategorical: entropy has no closed form".into(),
        })
    }

    // -----------------------------------------------------------------------
    // Full PyTorch surface (#1422) — Concrete relaxation has a reparameterized
    // sample (Gumbel-softmax), support on the open `K-1` simplex, single
    // `probs` parameter constrained to the simplex. mean/mode/variance have
    // no closed form (upstream raises NotImplementedError). Mirrors
    // `torch/distributions/relaxed_categorical.py:117-135`.
    // -----------------------------------------------------------------------

    fn has_rsample(&self) -> bool {
        // Gumbel-softmax forward is differentiable in `probs`+`temperature`;
        // mirrors `relaxed_categorical.py:127` which inherits
        // `has_rsample = True` from `TransformedDistribution`.
        true
    }

    fn support(&self) -> Option<Box<dyn DistConstraint>> {
        // `torch/distributions/relaxed_categorical.py:135`:
        //   support = constraints.simplex
        Some(Box::new(constraints::Simplex))
    }

    fn arg_constraints(&self) -> HashMap<&'static str, Box<dyn DistConstraint>> {
        // `torch/distributions/relaxed_categorical.py:133-134`:
        //   arg_constraints = {"probs": simplex, "logits": real_vector}
        let mut m: HashMap<&'static str, Box<dyn DistConstraint>> = HashMap::new();
        m.insert("probs", Box::new(constraints::Simplex));
        m
    }

    fn batch_shape(&self) -> Vec<usize> {
        // Batch shape is the parameter shape with the trailing K dim
        // removed. For the current 1-D `probs` impl that is empty.
        let mut s = self.probs.shape().to_vec();
        s.pop();
        s
    }

    fn event_shape(&self) -> Vec<usize> {
        // Each draw is a length-K simplex point.
        vec![self.num_categories]
    }

    fn expand(&self, batch_shape: &[usize]) -> FerrotorchResult<Box<dyn Distribution<T>>> {
        // Broadcast the (already-row-normalized) per-category probabilities to
        // the requested batch shape, producing a `[...batch_shape, K]` probs
        // tensor. The trailing K-vector is replicated across every batch row.
        // Empty batch_shape is an identity expand.
        let k = self.num_categories;
        if batch_shape.is_empty() {
            return Ok(Box::new(RelaxedOneHotCategorical::new(
                self.temperature,
                self.probs.clone(),
            )?));
        }
        // Use the first parameter row's UNNORMALIZED probs as the template
        // (every row would normalize identically; we replicate row 0).
        let probs_data = self.probs.data_vec()?;
        let template = &probs_data[..k];
        let num_rows: usize = batch_shape.iter().product();
        let mut out = Vec::with_capacity(num_rows * k);
        for _ in 0..num_rows {
            out.extend_from_slice(template);
        }
        let mut out_shape = batch_shape.to_vec();
        out_shape.push(k);
        let new_probs = Tensor::from_storage(TensorStorage::cpu(out), out_shape, false)?;
        Ok(Box::new(RelaxedOneHotCategorical::new(
            self.temperature,
            new_probs,
        )?))
    }
}

/// The log-space relaxed categorical (ExpRelaxedCategorical).
///
/// `ExpRelaxedCategorical(temperature, probs)` is the distribution of
/// `log(z)` where `z ~ RelaxedOneHotCategorical(temperature, probs)`. Its
/// samples are points in the *log*-simplex (each row of the output is the
/// log of a point on the K-simplex; `exp(row).sum() == 1`). It is the base
/// distribution that `RelaxedOneHotCategorical` exponentiates via the
/// `ExpTransform`.
///
/// Mirrors `torch.distributions.relaxed_categorical.ExpRelaxedCategorical`
/// (`torch/distributions/relaxed_categorical.py:17-106`).
pub struct ExpRelaxedCategorical<T: Float> {
    temperature: T,
    probs: Tensor<T>,
    normalized: Vec<T>,
    num_categories: usize,
    batch: Vec<usize>,
}

impl<T: Float> ExpRelaxedCategorical<T> {
    /// Construct an ExpRelaxedCategorical with the given temperature and
    /// (possibly batched, shape `[..., K]`) unnormalized class probabilities.
    pub fn new(temperature: T, probs: Tensor<T>) -> FerrotorchResult<Self> {
        let (k, batch, normalized) =
            validate_and_normalize(temperature, &probs, "ExpRelaxedCategorical")?;
        Ok(Self {
            temperature,
            probs,
            normalized,
            num_categories: k,
            batch,
        })
    }

    /// Temperature parameter.
    pub fn temperature(&self) -> T {
        self.temperature
    }

    /// (Unnormalized) probabilities.
    pub fn probs(&self) -> &Tensor<T> {
        &self.probs
    }

    /// Number of categories.
    pub fn num_categories(&self) -> usize {
        self.num_categories
    }

    /// The logits parameter `log(normalized_probs)`. Mirrors
    /// `ExpRelaxedCategorical.logits` (`relaxed_categorical.py:80-81`).
    pub fn logits(&self) -> FerrotorchResult<Tensor<T>> {
        let device = self.probs.device();
        let out: Vec<T> = self.normalized.iter().map(|&p| p.ln()).collect();
        let t = Tensor::from_storage(TensorStorage::cpu(out), self.probs.shape().to_vec(), false)?;
        if device.is_cuda() {
            t.to(device)
        } else {
            Ok(t)
        }
    }
}

impl<T: Float> Distribution<T> for ExpRelaxedCategorical<T> {
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.probs],
            "ExpRelaxedCategorical::sample",
        )?;
        exp_relaxed_sample(
            self.temperature,
            &self.normalized,
            self.num_categories,
            &self.batch,
            &self.probs,
            shape,
            false,
        )
    }

    fn rsample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.probs],
            "ExpRelaxedCategorical::rsample",
        )?;
        // Log-space Gumbel-softmax: scores - logsumexp(scores). Differentiable
        // in `probs` via `ExpRelaxedRsampleBackward`. Mirrors
        // `relaxed_categorical.py:87-94`.
        exp_relaxed_sample(
            self.temperature,
            &self.normalized,
            self.num_categories,
            &self.batch,
            &self.probs,
            shape,
            true,
        )
    }

    fn log_prob(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.probs, value],
            "ExpRelaxedCategorical::log_prob",
        )?;
        // Mirrors `ExpRelaxedCategorical.log_prob` (`relaxed_categorical.py:96-106`):
        //   log_scale = lgamma(K) + (K - 1) * log(temperature)
        //   score = logits - value * temperature
        //   score = (score - logsumexp(score, dim=-1)).sum(-1)
        //   log_prob = score + log_scale
        // where `value` is a log-simplex point (log(z)) and `logits` are the
        // normalized log-probabilities.
        let v_shape = value.shape().to_vec();
        let k = self.num_categories;
        if v_shape.is_empty() || *v_shape.last().unwrap() != k {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "ExpRelaxedCategorical: log_prob value last dim must be K={k}, got shape {v_shape:?}"
                ),
            });
        }
        let lambda = self.temperature;
        let zero = <T as num_traits::Zero>::zero();
        // log_scale = log((K-1)!) + (K-1)*log(temperature)
        let mut log_fact_km1 = zero;
        for i in 1..k {
            log_fact_km1 += T::from(i).unwrap().ln();
        }
        let km1 = T::from((k as i64 - 1).max(0)).unwrap();
        let log_scale = log_fact_km1 + km1 * lambda.ln();

        let log_alpha: Vec<T> = self.normalized.iter().map(|&p| p.ln()).collect();
        let num_param_rows = self.normalized.len() / k;

        let v_data = value.data_vec()?;
        let n = v_data.len() / k;
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let base = i * k;
            let alpha_base = (i % num_param_rows) * k;
            // score_j = logits_j - value_j * temperature
            let mut score = vec![zero; k];
            let mut max_s = T::neg_infinity();
            for j in 0..k {
                let s = log_alpha[alpha_base + j] - v_data[base + j] * lambda;
                score[j] = s;
                if s > max_s {
                    max_s = s;
                }
            }
            let mut sum_exp = zero;
            for &s in &score {
                sum_exp += (s - max_s).exp();
            }
            let lse = max_s + sum_exp.ln();
            // (score - lse).sum(-1)
            let mut acc = zero;
            for &s in &score {
                acc += s - lse;
            }
            result.push(acc + log_scale);
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
        Err(FerrotorchError::InvalidArgument {
            message: "ExpRelaxedCategorical: entropy has no closed form".into(),
        })
    }

    fn has_rsample(&self) -> bool {
        // `relaxed_categorical.py:45`: has_rsample = True
        true
    }

    fn support(&self) -> Option<Box<dyn DistConstraint>> {
        // `relaxed_categorical.py:42-44`: support = constraints.real_vector
        // (the true support is a submanifold; upstream advertises real_vector).
        // ferrotorch has no `RealVector` constraint object; `Real` is the
        // closest available — the log-simplex draw is an unconstrained real
        // vector. (The only metadata difference from `real_vector` is
        // `event_dim`, which `DistConstraint` does not surface for this op.)
        Some(Box::new(constraints::Real))
    }

    fn arg_constraints(&self) -> HashMap<&'static str, Box<dyn DistConstraint>> {
        // `relaxed_categorical.py:41`: {"probs": simplex, "logits": real_vector}
        let mut m: HashMap<&'static str, Box<dyn DistConstraint>> = HashMap::new();
        m.insert("probs", Box::new(constraints::Simplex));
        m
    }

    fn batch_shape(&self) -> Vec<usize> {
        self.batch.clone()
    }

    fn event_shape(&self) -> Vec<usize> {
        vec![self.num_categories]
    }
}

/// Log-space Gumbel-softmax draw for `ExpRelaxedCategorical`: each output row
/// is `scores - logsumexp(scores)` (a log-simplex point). Output shape is
/// `[...shape, ...batch, K]`. When `reparam` is set + `probs` requires grad +
/// grad enabled, the output carries an `ExpRelaxedRsampleBackward` node.
#[allow(clippy::needless_range_loop)]
fn exp_relaxed_sample<T: Float>(
    temperature: T,
    normalized: &[T],
    k: usize,
    batch: &[usize],
    probs: &Tensor<T>,
    shape: &[usize],
    reparam: bool,
) -> FerrotorchResult<Tensor<T>> {
    let device = probs.device();
    let zero = <T as num_traits::Zero>::zero();
    let one = <T as num_traits::One>::one();
    let eps = T::from(1e-20).unwrap();
    let sample_rows: usize = shape.iter().product::<usize>().max(1);
    let batch_rows: usize = batch.iter().product::<usize>().max(1);
    let num_rows = sample_rows * batch_rows;
    let num_param_rows = normalized.len() / k;

    let u = creation::rand::<T>(&[num_rows * k])?;
    let u_data = u.data_vec()?;
    let log_alpha: Vec<T> = normalized.iter().map(|&p| (p + eps).ln()).collect();

    let mut result = Vec::with_capacity(num_rows * k);
    for i in 0..num_rows {
        let alpha_base = (i % num_param_rows) * k;
        // scores_j = (log_alpha_j + g_j) / temperature
        let mut scores = vec![zero; k];
        let mut max_s = T::neg_infinity();
        for j in 0..k {
            let u_val = u_data[i * k + j].max(eps).min(one - eps);
            let g = zero - (zero - u_val.ln()).ln();
            let s = (log_alpha[alpha_base + j] + g) / temperature;
            scores[j] = s;
            if s > max_s {
                max_s = s;
            }
        }
        // logsumexp(scores)
        let mut sum_exp = zero;
        for &s in &scores {
            sum_exp += (s - max_s).exp();
        }
        let lse = max_s + sum_exp.ln();
        for j in 0..k {
            result.push(scores[j] - lse);
        }
    }

    let mut out_shape = shape.to_vec();
    out_shape.extend_from_slice(batch);
    out_shape.push(k);

    let storage = TensorStorage::cpu(result);
    let out = if reparam && probs.requires_grad() && ferrotorch_core::is_grad_enabled() {
        let grad_fn = Arc::new(ExpRelaxedRsampleBackward {
            temperature,
            probs: probs.clone(),
            normalized: normalized.to_vec(),
            k,
            num_rows,
            u: u.clone(),
        });
        Tensor::from_operation(storage, out_shape, grad_fn)?
    } else {
        Tensor::from_storage(storage, out_shape, false)?
    };
    if device.is_cuda() {
        out.to(device)
    } else {
        Ok(out)
    }
}

/// Autograd node for `ExpRelaxedCategorical::rsample`.
///
/// Forward (per row): `y_i = s_i - logsumexp(s)` with
/// `s_j = (log(alpha_j) + g_j) / temp`, `g` detached Gumbel.
/// `ds_j/dp_m = δ_jm / (temp * p_m)` (the `-1/sum` normalization term cancels
/// after the `- logsumexp` recentering, exactly as in the softmax case).
/// With `softmax_m = exp(s_m)/Σ exp(s)`:
///
/// ```text
/// dy_i/dp_m = (δ_im - softmax_m) / (temp * p_m)
/// grad_p_m  = (1/(temp*p_m)) * (go_m - softmax_m * Σ_i go_i)
/// ```
///
/// `temperature` is a scalar `T` (no gradient); the single input is `probs`.
#[derive(Debug)]
struct ExpRelaxedRsampleBackward<T: Float> {
    temperature: T,
    probs: Tensor<T>,
    normalized: Vec<T>,
    k: usize,
    num_rows: usize,
    u: Tensor<T>,
}

impl<T: Float> GradFn<T> for ExpRelaxedRsampleBackward<T> {
    #[allow(clippy::needless_range_loop)]
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let go = grad_output.data_vec()?;
        let u_data = self.u.data_vec()?;
        let probs_data = self.probs.data_vec()?;
        let k = self.k;
        let zero = <T as num_traits::Zero>::zero();
        let one = <T as num_traits::One>::one();
        let eps = T::from(1e-20).unwrap();
        let num_param_rows = self.normalized.len() / k;
        let log_alpha: Vec<T> = self.normalized.iter().map(|&p| (p + eps).ln()).collect();
        let mut grad_p = vec![zero; probs_data.len()];

        for i in 0..self.num_rows {
            let alpha_base = (i % num_param_rows) * k;
            // softmax of the scores for this row.
            let mut sm = vec![zero; k];
            let mut max_s = T::neg_infinity();
            for j in 0..k {
                let u_val = u_data[i * k + j].max(eps).min(one - eps);
                let g = zero - (zero - u_val.ln()).ln();
                let s = (log_alpha[alpha_base + j] + g) / self.temperature;
                sm[j] = s;
                if s > max_s {
                    max_s = s;
                }
            }
            let mut sum_exp = zero;
            for j in 0..k {
                sm[j] = (sm[j] - max_s).exp();
                sum_exp += sm[j];
            }
            for j in 0..k {
                sm[j] = sm[j] / sum_exp;
            }
            // go_sum = Σ_i go_i for this row.
            let mut go_sum = zero;
            for j in 0..k {
                go_sum += go[i * k + j];
            }
            for m in 0..k {
                let p_m = probs_data[alpha_base + m].max(eps);
                let contrib = (go[i * k + m] - sm[m] * go_sum) / (self.temperature * p_m);
                grad_p[alpha_base + m] += contrib;
            }
        }

        let grad = Tensor::from_storage(
            TensorStorage::cpu(grad_p),
            self.probs.shape().to_vec(),
            false,
        )?;
        Ok(vec![if self.probs.requires_grad() {
            Some(grad)
        } else {
            None
        }])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.probs]
    }

    fn name(&self) -> &'static str {
        "ExpRelaxedRsampleBackward"
    }
}

/// Compute the flat output buffer of a Gumbel-softmax draw plus the detached
/// Uniform noise that produced it.
///
/// `normalized` is the per-row-normalized probability cache (flat over
/// `num_param_rows` K-blocks); `num_rows` is the total number of output rows
/// (sample rows × batch rows). Each output row `r` uses parameter row
/// `r % num_param_rows`. Returns `(samples_flat, u)` where `u` is the drawn
/// Uniform noise (length `num_rows * k`).
#[allow(clippy::needless_range_loop)]
fn gumbel_softmax_forward<T: Float>(
    temperature: T,
    normalized: &[T],
    k: usize,
    num_rows: usize,
) -> FerrotorchResult<(Vec<T>, Tensor<T>)> {
    let zero = <T as num_traits::Zero>::zero();
    let one = <T as num_traits::One>::one();
    let eps = T::from(1e-20).unwrap();
    let num_param_rows = normalized.len() / k;

    let u = creation::rand::<T>(&[num_rows * k])?;
    let u_data = u.data_vec()?;
    let log_alpha: Vec<T> = normalized.iter().map(|&p| (p + eps).ln()).collect();

    let mut result = Vec::with_capacity(num_rows * k);
    for i in 0..num_rows {
        let alpha_base = (i % num_param_rows) * k;
        let mut logits = vec![zero; k];
        let mut max_l = T::neg_infinity();
        for j in 0..k {
            let u_val = u_data[i * k + j].max(eps).min(one - eps);
            // Gumbel(0,1) = -log(-log(U))
            let g = zero - (zero - u_val.ln()).ln();
            let l = (log_alpha[alpha_base + j] + g) / temperature;
            logits[j] = l;
            if l > max_l {
                max_l = l;
            }
        }
        let mut sum_exp = zero;
        for j in 0..k {
            logits[j] = (logits[j] - max_l).exp();
            sum_exp += logits[j];
        }
        for j in 0..k {
            result.push(logits[j] / sum_exp);
        }
    }
    Ok((result, u))
}

/// Gumbel-softmax sampling for RelaxedOneHotCategorical (shared by sample and
/// rsample). Result lies on the open K-simplex. The output shape is
/// `[...shape, ...batch, K]` (matching upstream `_extended_shape`). When
/// `reparam` is set, `probs` requires grad, and grad is enabled, the output
/// carries a `RelaxedOneHotRsampleBackward` node.
fn relaxed_one_hot_sample<T: Float>(
    temperature: T,
    normalized: &[T],
    k: usize,
    batch: &[usize],
    probs: &Tensor<T>,
    shape: &[usize],
    reparam: bool,
) -> FerrotorchResult<Tensor<T>> {
    let device = probs.device();
    // Output is sample_shape ++ batch_shape ++ [K] (upstream _extended_shape).
    let sample_rows: usize = shape.iter().product();
    let batch_rows: usize = batch.iter().product::<usize>().max(1);
    let num_rows = sample_rows.max(1) * batch_rows;

    let (result, u) = gumbel_softmax_forward(temperature, normalized, k, num_rows)?;

    let mut out_shape = shape.to_vec();
    out_shape.extend_from_slice(batch);
    out_shape.push(k);

    let storage = TensorStorage::cpu(result);
    let out = if reparam && probs.requires_grad() && ferrotorch_core::is_grad_enabled() {
        let grad_fn = Arc::new(RelaxedOneHotRsampleBackward {
            temperature,
            probs: probs.clone(),
            normalized: normalized.to_vec(),
            k,
            num_rows,
            u: u.clone(),
        });
        Tensor::from_operation(storage, out_shape, grad_fn)?
    } else {
        Tensor::from_storage(storage, out_shape, false)?
    };
    if device.is_cuda() {
        out.to(device)
    } else {
        Ok(out)
    }
}

/// Autograd node for the differentiable Gumbel-softmax `rsample`.
///
/// Forward (per row): `z = softmax((log(alpha) + g) / temp)` with `g` the
/// detached Gumbel noise and `alpha` the normalized probabilities. The
/// gradient w.r.t. the UNNORMALIZED `probs_m` (the `1/sum` normalization term
/// cancels by softmax shift-invariance):
///
/// ```text
/// dz_i/dp_m = z_i (δ_im - z_m) / (temp * p_m)
/// grad_p_m  = (z_m / (temp * p_m)) * (go_m - Σ_i go_i z_i)
/// ```
///
/// `temperature` is a scalar `T` (no gradient); the single input is `probs`.
#[derive(Debug)]
struct RelaxedOneHotRsampleBackward<T: Float> {
    temperature: T,
    probs: Tensor<T>,
    normalized: Vec<T>,
    k: usize,
    num_rows: usize,
    u: Tensor<T>,
}

impl<T: Float> GradFn<T> for RelaxedOneHotRsampleBackward<T> {
    #[allow(clippy::needless_range_loop)]
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let go = grad_output.data_vec()?;
        let u_data = self.u.data_vec()?;
        let probs_data = self.probs.data_vec()?;
        let k = self.k;
        let zero = <T as num_traits::Zero>::zero();
        let one = <T as num_traits::One>::one();
        let eps = T::from(1e-20).unwrap();
        let num_param_rows = self.normalized.len() / k;

        let log_alpha: Vec<T> = self.normalized.iter().map(|&p| (p + eps).ln()).collect();
        let mut grad_p = vec![zero; probs_data.len()];

        for i in 0..self.num_rows {
            let alpha_base = (i % num_param_rows) * k;
            // Recompute the softmax z for this row from the detached noise.
            let mut z = vec![zero; k];
            let mut max_l = T::neg_infinity();
            for j in 0..k {
                let u_val = u_data[i * k + j].max(eps).min(one - eps);
                let g = zero - (zero - u_val.ln()).ln();
                let l = (log_alpha[alpha_base + j] + g) / self.temperature;
                z[j] = l;
                if l > max_l {
                    max_l = l;
                }
            }
            let mut sum_exp = zero;
            for j in 0..k {
                z[j] = (z[j] - max_l).exp();
                sum_exp += z[j];
            }
            for j in 0..k {
                z[j] = z[j] / sum_exp;
            }
            // dot = Σ_i go_i z_i for this row.
            let mut dot = zero;
            for j in 0..k {
                dot += go[i * k + j] * z[j];
            }
            // grad_p_m += (z_m / (temp * p_m)) * (go_m - dot)
            for m in 0..k {
                let p_m = probs_data[alpha_base + m].max(eps);
                let contrib = (z[m] / (self.temperature * p_m)) * (go[i * k + m] - dot);
                grad_p[alpha_base + m] += contrib;
            }
        }

        let grad = Tensor::from_storage(
            TensorStorage::cpu(grad_p),
            self.probs.shape().to_vec(),
            false,
        )?;
        Ok(vec![if self.probs.requires_grad() {
            Some(grad)
        } else {
            None
        }])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.probs]
    }

    fn name(&self) -> &'static str {
        "RelaxedOneHotRsampleBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_tensor(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
    }

    #[test]
    fn test_relaxed_one_hot_invalid_temperature() {
        let probs = cpu_tensor(&[0.5, 0.5], &[2]);
        assert!(RelaxedOneHotCategorical::new(0.0_f32, probs).is_err());
    }

    #[test]
    fn test_relaxed_one_hot_invalid_probs() {
        let probs = cpu_tensor(&[0.0, 0.5], &[2]);
        assert!(RelaxedOneHotCategorical::new(1.0_f32, probs).is_err());
    }

    #[test]
    fn test_relaxed_one_hot_sample_shape_and_simplex() {
        let probs = cpu_tensor(&[0.2, 0.3, 0.5], &[3]);
        let d = RelaxedOneHotCategorical::new(0.5_f32, probs).unwrap();
        let s = d.sample(&[100]).unwrap();
        assert_eq!(s.shape(), &[100, 3]);
        let data = s.data().unwrap();
        for row in 0..100 {
            let row_sum: f32 = (0..3).map(|c| data[row * 3 + c]).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-4,
                "row {row} not on simplex: sum={row_sum}"
            );
            for c in 0..3 {
                // Closed simplex [0, 1]: at finite precision a softmax
                // output can underflow to exactly 0.0 or saturate to 1.0
                // for low-prob / high-prob entries at temperature 0.5,
                // so the strict (0, 1) bound was occasionally flaky.
                let v = data[row * 3 + c];
                assert!(
                    (0.0..=1.0).contains(&v),
                    "row {row} col {c}: {v} not in [0, 1]"
                );
            }
        }
    }

    #[test]
    fn test_relaxed_one_hot_low_temperature_concentrates() {
        // At very low temperature, the largest probability should
        // dominate -- mode-collapse toward category 2 in this example.
        let probs = cpu_tensor(&[0.1, 0.1, 0.8], &[3]);
        let d = RelaxedOneHotCategorical::new(0.05_f32, probs).unwrap();
        let s = d.sample(&[200]).unwrap();
        let data = s.data().unwrap();
        let mut category_2_dominant = 0;
        for row in 0..200 {
            let r0 = data[row * 3];
            let r1 = data[row * 3 + 1];
            let r2 = data[row * 3 + 2];
            if r2 > r0 && r2 > r1 {
                category_2_dominant += 1;
            }
        }
        // With probs 0.8 on category 2 and a tiny temperature, we expect
        // category 2 to dominate in most draws (≥ 70%).
        assert!(
            category_2_dominant >= 140,
            "expected category 2 to dominate, only {category_2_dominant}/200"
        );
    }

    #[test]
    fn test_relaxed_one_hot_log_prob_finite() {
        let probs = cpu_tensor(&[0.3, 0.3, 0.4], &[3]);
        let d = RelaxedOneHotCategorical::new(0.5_f32, probs).unwrap();
        let value = cpu_tensor(&[0.2, 0.3, 0.5], &[3]);
        let lp = d.log_prob(&value).unwrap();
        assert_eq!(lp.shape(), [] as [usize; 0]);
        let v = lp.item().unwrap();
        assert!(v.is_finite(), "log_prob should be finite, got {v}");
    }

    #[test]
    fn test_relaxed_one_hot_log_prob_batch() {
        let probs = cpu_tensor(&[0.5, 0.5], &[2]);
        let d = RelaxedOneHotCategorical::new(0.5_f32, probs).unwrap();
        let value = cpu_tensor(&[0.3, 0.7, 0.5, 0.5], &[2, 2]);
        let lp = d.log_prob(&value).unwrap();
        assert_eq!(lp.shape(), &[2]);
        let data = lp.data().unwrap();
        for &v in data {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_relaxed_one_hot_entropy_errors() {
        let probs = cpu_tensor(&[0.5, 0.5], &[2]);
        let d = RelaxedOneHotCategorical::new(0.5_f32, probs).unwrap();
        assert!(d.entropy().is_err());
    }

    #[test]
    fn test_relaxed_one_hot_logits_equals_log_normalized() {
        // probs=[1, 1, 2] (unnormalized) -> normalized=[0.25, 0.25, 0.5]
        // logits = log(normalized).
        let probs = cpu_tensor(&[1.0, 1.0, 2.0], &[3]);
        let d = RelaxedOneHotCategorical::new(0.5_f32, probs).unwrap();
        let l = d.logits().unwrap();
        let data = l.data_vec().unwrap();
        assert!((data[0] - 0.25_f32.ln()).abs() < 1e-5);
        assert!((data[1] - 0.25_f32.ln()).abs() < 1e-5);
        assert!((data[2] - 0.5_f32.ln()).abs() < 1e-5);
    }

    #[test]
    fn test_relaxed_one_hot_support_simplex() {
        let probs = cpu_tensor(&[0.5, 0.5], &[2]);
        let d = RelaxedOneHotCategorical::new(0.5_f32, probs).unwrap();
        let s = d.support().unwrap();
        assert_eq!(s.name(), "Simplex");
        assert!(d.has_rsample());
        let m = d.arg_constraints();
        assert!(m.contains_key("probs"));
        assert_eq!(d.event_shape(), vec![2]);
    }

    #[test]
    fn test_relaxed_one_hot_log_prob_wrong_shape_errors() {
        let probs = cpu_tensor(&[0.5, 0.5], &[2]);
        let d = RelaxedOneHotCategorical::new(0.5_f32, probs).unwrap();
        let bad = cpu_tensor(&[0.3, 0.3, 0.4], &[3]);
        assert!(d.log_prob(&bad).is_err());
    }

    // --- #1425: differentiable rsample ---------------------------------------

    #[test]
    fn test_relaxed_one_hot_rsample_requires_grad_when_probs_grad() {
        let probs = cpu_tensor(&[0.2, 0.3, 0.5], &[3]).requires_grad_(true);
        let d = RelaxedOneHotCategorical::new(0.5_f32, probs).unwrap();
        let s = d.rsample(&[4]).unwrap();
        assert!(s.requires_grad());
        assert!(s.grad_fn().is_some());
    }

    #[test]
    fn test_relaxed_one_hot_sample_detached() {
        let probs = cpu_tensor(&[0.2, 0.3, 0.5], &[3]).requires_grad_(true);
        let d = RelaxedOneHotCategorical::new(0.5_f32, probs).unwrap();
        let s = d.sample(&[4]).unwrap();
        assert!(!s.requires_grad());
    }

    #[test]
    fn test_relaxed_one_hot_rsample_grad_flows_to_probs_finite() {
        // backward() through the Gumbel-softmax rsample populates probs.grad()
        // with finite values. Σ_m grad_p_m over a single row is
        // Σ_m (z_m/(temp*p_m))*(1 - Σ_i z_i) = 0 when grad_output is all-ones
        // and each output row sums to 1 — so we instead use a weighted loss
        // (sum of squares) to get a non-trivially-zero gradient.
        use ferrotorch_core::grad_fns::arithmetic::mul;
        let probs = cpu_tensor(&[0.2, 0.3, 0.5], &[3]).requires_grad_(true);
        let d = RelaxedOneHotCategorical::new(0.5_f32, probs.clone()).unwrap();
        let s = d.rsample(&[8]).unwrap();
        let sq = mul(&s, &s).unwrap();
        let loss = sq.sum_all().unwrap();
        loss.backward().unwrap();
        let g = probs.grad().unwrap().unwrap();
        let gd = g.data_vec().unwrap();
        assert_eq!(gd.len(), 3);
        for &v in &gd {
            assert!(v.is_finite(), "grad must be finite, got {v}");
        }
        // The gradient must not be all-zero (a non-degenerate loss touches probs).
        assert!(
            gd.iter().any(|&v| v.abs() > 1e-6),
            "expected a non-zero grad, got {gd:?}"
        );
    }

    // --- #1426: batched probs ------------------------------------------------

    #[test]
    fn test_relaxed_one_hot_batched_sample_shape() {
        // probs [B=2, K=3] → sample(&[S=4]) yields [4, 2, 3].
        let probs = cpu_tensor(&[0.2, 0.3, 0.5, 0.6, 0.3, 0.1], &[2, 3]);
        let d = RelaxedOneHotCategorical::new(0.5_f32, probs).unwrap();
        assert_eq!(d.batch_shape(), vec![2]);
        assert_eq!(d.event_shape(), vec![3]);
        let s = d.sample(&[4]).unwrap();
        assert_eq!(s.shape(), &[4, 2, 3]);
        // Each trailing-K row sums to ~1.
        let data = s.data().unwrap();
        for row in 0..(4 * 2) {
            let sum: f32 = (0..3).map(|c| data[row * 3 + c]).sum();
            assert!((sum - 1.0).abs() < 1e-4, "row {row} sum={sum}");
        }
    }

    #[test]
    fn test_relaxed_one_hot_batched_log_prob_shape() {
        // probs [2, 3], value [2, 3] → log_prob [2].
        let probs = cpu_tensor(&[0.2, 0.3, 0.5, 0.6, 0.3, 0.1], &[2, 3]);
        let d = RelaxedOneHotCategorical::new(0.5_f32, probs).unwrap();
        let value = cpu_tensor(&[0.25, 0.35, 0.4, 0.5, 0.3, 0.2], &[2, 3]);
        let lp = d.log_prob(&value).unwrap();
        assert_eq!(lp.shape(), &[2]);
        for &v in lp.data().unwrap() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_relaxed_one_hot_batched_rows_differ() {
        // With distinct rows, the per-row normalization must differ:
        // row 0 = [0.1, 0.9] → log_prob should weight category 1 heavily;
        // row 1 = [0.9, 0.1] → opposite. Evaluating each row's mode-ish point
        // should give different log_probs across rows.
        let probs = cpu_tensor(&[0.1, 0.9, 0.9, 0.1], &[2, 2]);
        let d = RelaxedOneHotCategorical::new(0.5_f32, probs).unwrap();
        // Same value applied to both rows.
        let value = cpu_tensor(&[0.2, 0.8, 0.2, 0.8], &[2, 2]);
        let lp = d.log_prob(&value).unwrap();
        let data = lp.data().unwrap();
        assert!(
            (data[0] - data[1]).abs() > 1e-3,
            "batched rows should differ: {data:?}"
        );
    }

    // --- #1424: ExpRelaxedCategorical standalone -----------------------------

    #[test]
    fn test_exp_relaxed_sample_is_log_simplex() {
        // Each row of a sample is log(z) for z on the simplex, so exp(row)
        // sums to 1.
        let probs = cpu_tensor(&[0.2, 0.3, 0.5], &[3]);
        let d = ExpRelaxedCategorical::new(0.5_f32, probs).unwrap();
        let s = d.sample(&[50]).unwrap();
        assert_eq!(s.shape(), &[50, 3]);
        let data = s.data().unwrap();
        for row in 0..50 {
            let sum: f32 = (0..3).map(|c| data[row * 3 + c].exp()).sum();
            assert!((sum - 1.0).abs() < 1e-4, "exp-row {row} sum={sum}");
        }
    }

    #[test]
    fn test_exp_relaxed_log_prob_finite_and_shape() {
        let probs = cpu_tensor(&[0.3, 0.3, 0.4], &[3]);
        let d = ExpRelaxedCategorical::new(0.5_f32, probs).unwrap();
        // A valid log-simplex point: log of [0.25, 0.35, 0.4].
        let value = cpu_tensor(&[0.25_f32.ln(), 0.35_f32.ln(), 0.4_f32.ln()], &[3]);
        let lp = d.log_prob(&value).unwrap();
        assert_eq!(lp.shape(), [] as [usize; 0]);
        assert!(lp.item().unwrap().is_finite());
        assert!(d.has_rsample());
        assert_eq!(d.support().unwrap().name(), "Real");
        assert_eq!(d.event_shape(), vec![3]);
        assert_eq!(d.num_categories(), 3);
    }

    #[test]
    fn test_exp_relaxed_rsample_grad_flows_to_probs() {
        use ferrotorch_core::grad_fns::arithmetic::mul;
        use ferrotorch_core::grad_fns::transcendental::exp;
        let probs = cpu_tensor(&[0.2, 0.3, 0.5], &[3]).requires_grad_(true);
        let d = ExpRelaxedCategorical::new(0.5_f32, probs.clone()).unwrap();
        let s = d.rsample(&[8]).unwrap();
        assert!(s.requires_grad());
        // exp(log-simplex) values are on the simplex; use a non-degenerate loss.
        let z = exp(&s).unwrap();
        let loss = mul(&z, &z).unwrap().sum_all().unwrap();
        loss.backward().unwrap();
        let g = probs.grad().unwrap().unwrap();
        let gd = g.data_vec().unwrap();
        for &v in &gd {
            assert!(v.is_finite(), "grad must be finite, got {v}");
        }
        assert!(gd.iter().any(|&v| v.abs() > 1e-6), "expected non-zero grad");
    }

    #[test]
    fn test_exp_relaxed_invalid_temperature() {
        let probs = cpu_tensor(&[0.5, 0.5], &[2]);
        assert!(ExpRelaxedCategorical::new(0.0_f32, probs).is_err());
    }
}
