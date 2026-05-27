//! Probability distributions for ferrotorch.
//!
//! This crate provides differentiable probability distributions following the
//! PyTorch `torch.distributions` API. Each distribution supports:
//!
//! - **`sample`** — draw samples (no gradient)
//! - **`rsample`** — reparameterized sampling (gradient flows through samples)
//! - **`log_prob`** — compute log-probability of a value
//! - **`entropy`** — compute the distribution's entropy
//!
//! # Distributions
//!
//! | Distribution | Parameters | Reparameterized |
//! |-------------|-----------|-----------------|
//! | [`Normal`] | `loc`, `scale` | Yes |
//! | [`Uniform`] | `low`, `high` | Yes |
//! | [`Bernoulli`] | `probs` | No (discrete) |
//! | [`Categorical`] | `probs` | No (discrete) |
//! | [`Beta`] | `concentration1`, `concentration0` | Yes |
//! | [`Gamma`] | `concentration`, `rate` | Yes |
//! | [`Exponential`] | `rate` | Yes |
//! | [`Laplace`] | `loc`, `scale` | Yes |
//! | [`Cauchy`] | `loc`, `scale` | Yes |
//! | [`Gumbel`] | `loc`, `scale` | Yes |
//! | [`HalfNormal`] | `scale` | Yes |
//! | [`LogNormal`] | `loc`, `scale` | Yes |
//! | [`Poisson`] | `rate` | No (discrete) |
//! | [`StudentT`] | `df`, `loc`, `scale` | Yes |
//! | [`MultivariateNormal`] | `loc`, `scale_tril` | Yes |
//! | [`LowRankMultivariateNormal`] | `loc`, `cov_factor`, `cov_diag` | Yes |
//! | [`Dirichlet`] | `concentration` | Yes |
//! | [`Multinomial`] | `total_count`, `probs` | No (discrete) |
//! | [`Independent`] | base distribution + `reinterpreted_batch_ndims` | inherits |
//! | [`MixtureSameFamily`] | mixing `Categorical` + components | No |
//! | [`OneHotCategorical`] | `probs` | No (discrete) |
//! | [`RelaxedBernoulli`] | `temperature`, `probs` | Yes (Concrete relaxation) |
//! | [`RelaxedOneHotCategorical`] | `temperature`, `probs` | Yes (Concrete relaxation) |
//! | [`ExpRelaxedCategorical`] | `temperature`, `probs` | Yes (log-simplex Concrete relaxation) |
//! | [`Pareto`] | `scale`, `alpha` | No (rsample not yet implemented) |
//! | [`Kumaraswamy`] | `concentration1`, `concentration0` | No (rsample not yet implemented) |
//! | [`VonMises`] | `loc`, `concentration` | No (rejection sampling) |
//! | [`Weibull`] | `scale`, `concentration` | No (rsample not yet implemented) |
//!
//! # Infrastructure
//!
//! - [`constraints`] — constraint objects for parameter and support validation
//! - [`transforms`] — bijective transforms with log-det-Jacobian computation
//! - [`kl`] — analytical KL divergence for same-family distribution pairs
//! - [`TransformedDistribution`](transforms::TransformedDistribution) — apply
//!   bijective transforms to a base distribution
//!
//! ## REQ status (per `.design/ferrotorch-distributions/lib.md`)
//!
//! Full evidence rows (impl + non-test production consumer + upstream cites)
//! live in the design doc; this synopsis is a one-line summary per REQ.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`Distribution<T>` trait: 4 required methods) | SHIPPED | `pub trait Distribution<T: Float>: Send + Sync` with `sample`/`rsample`/`log_prob`/`entropy` in `lib.rs` mirroring `torch/distributions/distribution.py:167-255`; consumers: `impl Distribution<T> for Normal<T>` in `normal.rs`, plus 25 other concrete `impl Distribution` sites across the crate |
//! | REQ-2 (default-implemented property methods) | SHIPPED | `batch_shape`/`cdf`/`icdf`/`mean`/`mode`/`variance`/`stddev` defaults in `pub trait Distribution` mirroring `torch/distributions/distribution.py:108-165`; consumers: `fn Independent::batch_shape` in `independent.rs` overrides the default; `fn TransformedDistribution::entropy` in `transforms.rs` invokes `self.base.mean()?` |
//! | REQ-3 (module tree + `pub use` re-exports) | SHIPPED | mod declarations + `pub use bernoulli::Bernoulli` through `pub use weibull::Weibull` block in `lib.rs` mirroring `torch/distributions/__init__.py:74-119`; consumers: `tests/conformance_distributions_*` use the re-exports; downstream crates import via `ferrotorch_distributions::{Normal, Bernoulli, ...}` |
//! | REQ-4 (`<T: Float>` generic parametrisation) | SHIPPED | `pub trait Distribution<T: Float>` with explicit `T: Float` on every method (R-DEV-7: monomorphise per-dtype); consumers: every concrete `pub struct Normal<T: Float>` / `Gamma<T: Float>` etc. with `impl<T: Float> Distribution<T>` — f32 and f64 both exercised by `*_f64` tests per family |
//! | REQ-5 (full PyTorch `Distribution` surface) | SHIPPED | `support` / `arg_constraints` / `has_rsample` / `has_enumerate_support` / `event_shape` / `expand` / `enumerate_support` / `perplexity` defaults landed on `pub trait Distribution` in `lib.rs` mirroring `torch/distributions/distribution.py:25-348`; consumers: `fn Normal::support` / `fn Normal::arg_constraints` in `normal.rs`, `fn Bernoulli::support` / `fn Bernoulli::enumerate_support` in `bernoulli.rs`, plus `fn Uniform::support` in `uniform.rs`, `fn Exponential::support` in `exponential.rs`, `fn Gamma::support` in `gamma.rs`, `fn Categorical::support` in `categorical.rs`; `Distribution::perplexity` default `exp(self.entropy()?)` consumed by every concrete distribution via the default impl |

mod bernoulli;
mod beta;
mod categorical;
mod cauchy;
pub mod constraints;
mod dirichlet;
mod exponential;
pub(crate) mod fallback;
mod gamma;
mod gumbel;
mod half_normal;
mod independent;
pub mod kl;
mod kumaraswamy;
mod laplace;
mod lognormal;
mod low_rank_multivariate_normal;
mod mixture_same_family;
mod multinomial;
mod multivariate_normal;
mod normal;
mod one_hot_categorical;
mod pareto;
mod poisson;
mod relaxed_bernoulli;
mod relaxed_one_hot_categorical;
pub(crate) mod special_fns;
mod student_t;
pub mod transforms;
mod uniform;
mod von_mises;
mod weibull;

pub use bernoulli::Bernoulli;
pub use beta::Beta;
pub use categorical::Categorical;
pub use cauchy::Cauchy;
pub use dirichlet::Dirichlet;
pub use exponential::Exponential;
pub use gamma::Gamma;
pub use gumbel::Gumbel;
pub use half_normal::HalfNormal;
pub use independent::Independent;
pub use kumaraswamy::Kumaraswamy;
pub use laplace::Laplace;
pub use lognormal::LogNormal;
pub use low_rank_multivariate_normal::LowRankMultivariateNormal;
pub use mixture_same_family::MixtureSameFamily;
pub use multinomial::Multinomial;
pub use multivariate_normal::MultivariateNormal;
pub use normal::Normal;
pub use one_hot_categorical::{OneHotCategorical, OneHotCategoricalStraightThrough};
pub use pareto::Pareto;
pub use poisson::Poisson;
pub use relaxed_bernoulli::RelaxedBernoulli;
pub use relaxed_one_hot_categorical::{ExpRelaxedCategorical, RelaxedOneHotCategorical};
pub use student_t::StudentT;
pub use transforms::{
    AbsTransform, AffineTransform, CatTransform, ComposeTransform, CorrCholeskyTransform,
    CumulativeDistributionTransform, ExpTransform, IndependentTransform, LowerCholeskyTransform,
    PowerTransform, ReshapeTransform, SigmoidTransform, SoftmaxTransform, SoftplusTransform,
    StackTransform, StickBreakingTransform, TanhTransform, Transform, TransformedDistribution,
};
pub use uniform::Uniform;
pub use von_mises::VonMises;
pub use weibull::Weibull;

use std::collections::HashMap;
use std::fmt::Debug;

use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::tensor::Tensor;

// ---------------------------------------------------------------------------
// Dyn-safe constraint object surface (REQ-5: support / arg_constraints)
// ---------------------------------------------------------------------------

/// Object-safe constraint descriptor exposed by `Distribution::support` and
/// `Distribution::arg_constraints`.
///
/// The full [`constraints::Constraint`] trait carries a generic
/// `check<T: Float>` method (R-DEV-7 monomorphisation) which forbids
/// trait-object use. `DistConstraint` exposes only the dtype-independent
/// metadata callers need to introspect a distribution's support:
/// human-readable name, discrete-or-continuous flag, and the number of
/// rightmost dims that together form an event.
///
/// Mirrors the subset of `torch.distributions.constraints.Constraint` that
/// is interrogable without a concrete tensor argument — see
/// `torch/distributions/constraints.py:80-106` (`Constraint.is_discrete`,
/// `Constraint.event_dim`).
pub trait DistConstraint: Send + Sync + Debug {
    /// Human-readable constraint name (e.g. `"Real"`, `"UnitInterval"`).
    fn name(&self) -> &'static str;

    /// Whether the constrained domain is discrete (`true`) or continuous
    /// (`false`). Defaults to `false`.
    fn is_discrete(&self) -> bool {
        false
    }

    /// Number of rightmost dimensions that together form a single event.
    /// Defaults to `0` (univariate).
    fn event_dim(&self) -> usize {
        0
    }
}

/// Blanket impl: every type that satisfies the non-generic surface of
/// [`constraints::Constraint`] *and* is `Debug + 'static` is a
/// [`DistConstraint`]. The blanket pulls `name`/`is_discrete`/`event_dim`
/// straight off the source trait — `check<T>` is intentionally *not* on
/// `DistConstraint` because it would re-introduce the generic method that
/// breaks dyn-compatibility.
impl<C> DistConstraint for C
where
    C: constraints::Constraint + Debug + 'static,
{
    fn name(&self) -> &'static str {
        <Self as constraints::Constraint>::name(self)
    }
    fn is_discrete(&self) -> bool {
        <Self as constraints::Constraint>::is_discrete(self)
    }
    fn event_dim(&self) -> usize {
        <Self as constraints::Constraint>::event_dim(self)
    }
}

/// A probability distribution over tensors.
///
/// This trait mirrors PyTorch's `torch.distributions.Distribution` base class.
/// Implementations define how to sample, compute log-probabilities, and
/// measure entropy.
///
/// # Type parameter
///
/// `T` must implement [`Float`] — currently `f32` or `f64`.
///
/// # `sample` vs `rsample`
///
/// - [`sample`](Distribution::sample) draws samples with no gradient. Use for
///   discrete distributions or when gradients through sampling are not needed.
/// - [`rsample`](Distribution::rsample) draws reparameterized samples. The
///   result has `requires_grad = true` and gradients flow back through the
///   sampling operation via the reparameterization trick. This is essential
///   for variational inference (VAE, etc.).
///
/// Distributions that cannot be reparameterized (e.g., [`Bernoulli`],
/// [`Categorical`]) return an error from `rsample`.
pub trait Distribution<T: Float>: Send + Sync {
    /// Draw samples from the distribution.
    ///
    /// The returned tensor has the given `shape` and `requires_grad = false`.
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>>;

    /// Draw reparameterized samples from the distribution.
    ///
    /// The returned tensor has `requires_grad = true` and gradients flow
    /// through the sampling operation back to the distribution parameters.
    ///
    /// Returns an error for distributions that cannot be reparameterized.
    fn rsample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>>;

    /// Compute the log-probability of `value` under the distribution.
    ///
    /// Returns a tensor with the same shape as `value`.
    fn log_prob(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>>;

    /// Compute the entropy of the distribution.
    ///
    /// Returns a scalar tensor (or a tensor matching the batch shape of the
    /// distribution parameters).
    fn entropy(&self) -> FerrotorchResult<Tensor<T>>;

    // -----------------------------------------------------------------------
    // Distribution properties (#585) — default implementations return
    // NotImplementedOnCuda-style errors. Concrete distributions override
    // what they can express in closed form.
    // -----------------------------------------------------------------------

    /// The batch shape of the distribution — the shape of parameter tensors
    /// (excluding event dims). Default returns an empty vec (scalar batch).
    ///
    /// Distributions with batched parameters (e.g. `Normal` with `loc` of
    /// shape `[B]`) override this to return `vec![B]`. Used by `Independent`
    /// to forward the correct sample shape to the base distribution.
    fn batch_shape(&self) -> Vec<usize> {
        vec![]
    }

    /// Cumulative distribution function: `P(X <= value)`. Default returns an
    /// `InvalidArgument` error for distributions without a closed-form CDF.
    fn cdf(&self, _value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        Err(FerrotorchError::InvalidArgument {
            message: "cdf not implemented for this distribution".into(),
        })
    }

    /// Inverse CDF (quantile function): the value `x` such that
    /// `P(X <= x) = q`. Default returns an `InvalidArgument` error.
    fn icdf(&self, _q: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        Err(FerrotorchError::InvalidArgument {
            message: "icdf not implemented for this distribution".into(),
        })
    }

    /// Distribution mean. Default returns an `InvalidArgument` error.
    fn mean(&self) -> FerrotorchResult<Tensor<T>> {
        Err(FerrotorchError::InvalidArgument {
            message: "mean not implemented for this distribution".into(),
        })
    }

    /// Distribution mode. Default returns an `InvalidArgument` error.
    fn mode(&self) -> FerrotorchResult<Tensor<T>> {
        Err(FerrotorchError::InvalidArgument {
            message: "mode not implemented for this distribution".into(),
        })
    }

    /// Distribution variance. Default returns an `InvalidArgument` error.
    fn variance(&self) -> FerrotorchResult<Tensor<T>> {
        Err(FerrotorchError::InvalidArgument {
            message: "variance not implemented for this distribution".into(),
        })
    }

    /// Distribution standard deviation. Default: `sqrt(variance)`.
    fn stddev(&self) -> FerrotorchResult<Tensor<T>> {
        let v = self.variance()?;
        let data = v.data_vec()?;
        let out: Vec<T> = data.iter().map(|x| x.sqrt()).collect();
        Tensor::from_storage(
            ferrotorch_core::storage::TensorStorage::cpu(out),
            v.shape().to_vec(),
            false,
        )
    }

    // -----------------------------------------------------------------------
    // Full PyTorch `Distribution` surface (#1376) — defaults return either
    // a structured `InvalidArgument` (for methods PyTorch raises
    // `NotImplementedError` on) or a sensible fallback (e.g. `perplexity =
    // exp(entropy)`). Concrete distributions override only what they can
    // express. Mirrors `torch/distributions/distribution.py:25-348`.
    // -----------------------------------------------------------------------

    /// Shape of a single sample (without batching). Default returns an
    /// empty vec, matching `torch/distributions/distribution.py:114-119`
    /// (`event_shape = torch.Size()` for univariate distributions).
    fn event_shape(&self) -> Vec<usize> {
        vec![]
    }

    /// Whether the distribution implements reparameterized sampling.
    ///
    /// Default: `false`. Continuous distributions with a closed-form
    /// reparameterization (Normal, Uniform, Exponential, Gamma, Beta,
    /// Laplace, Cauchy, …) override to return `true`. Mirrors the
    /// class-level `has_rsample = False` flag at
    /// `torch/distributions/distribution.py:25`.
    fn has_rsample(&self) -> bool {
        false
    }

    /// Whether the distribution implements `enumerate_support`.
    ///
    /// Default: `false`. Finite discrete distributions (Bernoulli,
    /// Categorical, OneHotCategorical) override to return `true`.
    /// Mirrors `torch/distributions/distribution.py:26`
    /// (`has_enumerate_support = False`).
    fn has_enumerate_support(&self) -> bool {
        false
    }

    /// The support of the distribution as a [`DistConstraint`] object
    /// (e.g. `Real`, `UnitInterval`, `Positive`).
    ///
    /// Default returns `None`. Concrete distributions override to advertise
    /// their support. Mirrors
    /// `torch/distributions/distribution.py:131-138` (the `support`
    /// property), where PyTorch raises `NotImplementedError` if a subclass
    /// has not declared support.
    fn support(&self) -> Option<Box<dyn DistConstraint>> {
        None
    }

    /// The argument constraints map: parameter-name → [`DistConstraint`].
    ///
    /// Default returns an empty map. Concrete distributions override to
    /// advertise the constraint each constructor argument must satisfy.
    /// Mirrors `torch/distributions/distribution.py:121-129` (the
    /// `arg_constraints` property).
    fn arg_constraints(&self) -> HashMap<&'static str, Box<dyn DistConstraint>> {
        HashMap::new()
    }

    /// Return a new distribution with batch dims expanded to `batch_shape`.
    ///
    /// Default returns `InvalidArgument`. Concrete distributions override
    /// by constructing a new instance whose parameters have been broadcast
    /// to the target shape (no allocation copy in PyTorch via
    /// `Tensor::expand`; ferrotorch CPU path materialises the broadcast for
    /// simplicity). Mirrors
    /// `torch/distributions/distribution.py:86-105` (`expand`).
    fn expand(&self, _batch_shape: &[usize]) -> FerrotorchResult<Box<dyn Distribution<T>>> {
        Err(FerrotorchError::InvalidArgument {
            message: "expand not implemented for this distribution".into(),
        })
    }

    /// Enumerate all values supported by a discrete distribution.
    ///
    /// Default returns `InvalidArgument`. Finite discrete distributions
    /// (Bernoulli, Categorical, OneHotCategorical) override. Mirrors
    /// `torch/distributions/distribution.py:224-246`
    /// (`enumerate_support`).
    ///
    /// When `expand` is `true`, the result is broadcast across the
    /// distribution's `batch_shape`; when `false`, the trailing batch
    /// dimensions are kept as singletons.
    fn enumerate_support(&self, _expand: bool) -> FerrotorchResult<Tensor<T>> {
        Err(FerrotorchError::InvalidArgument {
            message: "enumerate_support not implemented for this distribution".into(),
        })
    }

    /// Perplexity: `exp(entropy)`. Default body invokes `self.entropy()?`
    /// and exponentiates element-wise. Mirrors
    /// `torch/distributions/distribution.py:257-264`.
    fn perplexity(&self) -> FerrotorchResult<Tensor<T>> {
        let h = self.entropy()?;
        let data = h.data_vec()?;
        let out: Vec<T> = data.iter().map(|x| x.exp()).collect();
        Tensor::from_storage(
            ferrotorch_core::storage::TensorStorage::cpu(out),
            h.shape().to_vec(),
            false,
        )
    }
}

// ---------------------------------------------------------------------------
// ExponentialFamily trait (#1404, #1407)
// ---------------------------------------------------------------------------

/// Marker trait for distributions in the exponential family.
///
/// An exponential-family density has the canonical form
/// `p(x; θ) = exp(<t(x), η(θ)> − F(η) + k(x))` where:
/// - `η(θ)` are the **natural parameters** (returned by [`natural_params`](ExponentialFamily::natural_params))
/// - `F(η)` is the **log-normalizer** (returned by [`log_normalizer`](ExponentialFamily::log_normalizer))
/// - `k(x)` is the carrier measure (its expectation is
///   [`mean_carrier_measure`](ExponentialFamily::mean_carrier_measure))
///
/// Mirrors `torch.distributions.ExponentialFamily` (`torch/distributions/exp_family.py:11-66`).
/// Used by KL-divergence machinery and analytic entropy reasoning.
pub trait ExponentialFamily<T: Float>: Distribution<T> {
    /// The natural parameters `η(θ)` as a flat list of `Tensor<T>`.
    /// Mirrors `_natural_params` (`exp_family.py:32-38`).
    fn natural_params(&self) -> FerrotorchResult<Vec<Tensor<T>>>;

    /// The log-normalizer `F(η)` evaluated at the given natural-parameter
    /// tuple. The argument is the same shape/order as
    /// [`natural_params`](Self::natural_params) returns. Mirrors
    /// `_log_normalizer(*natural_params)` (`exp_family.py:40-45`).
    fn log_normalizer(&self, natural_params: &[Tensor<T>]) -> FerrotorchResult<Tensor<T>>;

    /// The expected carrier measure `E[k(X)]`. Returns 0 for most
    /// continuous families. Mirrors `_mean_carrier_measure`
    /// (`exp_family.py:47-53`).
    fn mean_carrier_measure(&self) -> FerrotorchResult<T> {
        Ok(<T as num_traits::Zero>::zero())
    }
}
