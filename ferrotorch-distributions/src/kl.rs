//! KL divergence between probability distributions.
//!
//! Provides closed-form analytical KL divergence formulas for same-family
//! and select cross-family distribution pairs.
//!
//! This mirrors PyTorch's `torch.distributions.kl` module.
//!
//! CL-330
//!
//! ## REQ status (per `.design/ferrotorch-distributions/kl.md`)
//!
//! Full evidence rows (impl + non-test production consumer + upstream cites)
//! live in the design doc; this synopsis is a one-line summary per REQ.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`kl_divergence<T, P, Q>` public entry point) | SHIPPED | `pub fn kl_divergence<T: Float, P, Q>` with `P: Distribution<T> + 'static`, `Q: Distribution<T> + 'static` bounds in `kl.rs` mirroring `torch/distributions/kl.py:kl_divergence`; consumer: `pub mod kl` in `lib.rs` exposes it as grandfathered public API; `test_kl_*` (~25 sites) exercise the dispatch path |
//! | REQ-2 (`kl_supported_pair_count` introspection) | SHIPPED | `pub const fn kl_supported_pair_count() -> usize` + `KL_SUPPORTED_PAIR_COUNT: usize = 70` in `kl.rs`; consumer: const fn is grandfathered public API; drift-prevention test `kl_doc_table_matches_dispatcher` reads `include_str!("kl.rs")` and asserts three-way invariant against the public accessor |
//! | REQ-3 (`kl_dispatch` `Any::downcast_ref` chain) | SHIPPED | the dispatcher in `kl.rs` is a 70-arm chain mirroring PyTorch's `_dispatch_kl` in `torch/distributions/kl.py:113-138`; consumer: `pub fn kl_divergence` invokes the dispatcher on every call |
//! | REQ-4 (8 same-family closed-form formulas) | SHIPPED | 8 closed-form helpers in `kl.rs` (`kl_normal_normal`, `kl_bernoulli_bernoulli`, `kl_uniform_uniform`, `kl_categorical_categorical`, `kl_laplace_laplace`, `kl_exponential_exponential`, `kl_gamma_gamma`, `kl_poisson_poisson`) mirroring `@register_kl` bodies in `torch/distributions/kl.py`; consumer: the dispatcher invokes each formula |
//! | REQ-5 (cross-family finite formulas) | SHIPPED | `kl_uniform_normal`, `kl_gamma_exponential`, `kl_exponential_gamma` in `kl.rs`; last two use `kl_gamma_scalar` via `Exp(λ) ≡ Gamma(1, λ)`; consumer: the dispatcher calls each; `kl_gamma_scalar` is consumed by 3 production sites internally. (Normal-Uniform was a finite arm here but moved to the `+inf` support-mismatch family per `kl.py:766,768` `_kl_normal_infinity` — #1563.) |
//! | REQ-6 (fallback guard on every formula) | SHIPPED | every finite formula's first statement is `crate::fallback::check_gpu_fallback_opt_in(&[...], "kl_divergence(P, Q)")?` in `kl.rs`; consumer: this IS the production consumer of `fn check_gpu_fallback_opt_in` per `fallback.md` REQ-2 (the `+inf` support-mismatch arms read a single param tensor that is already host-resident, so they hand it straight to `kl_infinite_like`) |
//! | REQ-7 (full ~75-pair PyTorch coverage) | PARTIAL | blocker #1374 — ferrotorch now ships 70 of PyTorch's ~75 pairs (was 41). The #1562 closure added 27; the #1374 Binomial sub-part added 2: `kl_binomial_binomial` (finite, mirrors `torch/distributions/kl.py:231-244`: `n·(p·(logit_p−logit_q)+ln(1−p)−ln(1−q))`, `+inf` where `n_p>n_q`, `InvalidArgument` where `n_p<n_q`) + Poisson-Binomial routed through `kl_infinite_like` (mirrors `_kl_poisson_infinity` `kl.py:842`). Finite #1562 arms (`kl_onehotcategorical_onehotcategorical` `kl.py:474-476`, `kl_bernoulli_poisson` `kl.py:513-516`, `kl_normal_laplace` `kl.py:782-792`) + 24 support-mismatch `+inf` arms (PyTorch's `_infinite_like` registrations: Beta-Pareto `kl.py:528`; Exponential-{Beta,Pareto,Uniform} `kl.py:620-623`; Gamma-{Beta,Pareto,Uniform} `kl.py:665-668`; Gumbel-{Beta,Exponential,Gamma,Pareto,Uniform} `kl.py:718-723`; Laplace-{Beta,Exponential,Gamma,Pareto,Uniform} `kl.py:740-745`; Normal-{Beta,Exponential,Gamma,Pareto} `kl.py:761-765`; Pareto-{Beta,Uniform} `kl.py:795-797`; Poisson-Bernoulli `kl.py:841`) routed through the `kl_infinite_like` helper; consumer: each is invoked by its dispatcher downcast arm. Still NOT-STARTED: (a) `Independent-Independent` (`kl.py:944`) — `Independent<T, D>` is generic over the concrete base `D`, so `Any::downcast_ref` cannot match it without a KL-recursion trait hook on `Distribution` (`lib.rs`) + an override in `independent.rs`, both outside this manifest (concrete prereq, not a deferral); (b) Geometric-Geometric (need a `Geometric` struct), ContinuousBernoulli-* pairs (need a `ContinuousBernoulli` struct), TransformedDistribution-TransformedDistribution / ExponentialFamily-ExponentialFamily — each blocked on a missing distribution TYPE or trait surface. #1374 stays open for the remaining ~5. |
//! | REQ-8 (`register_kl` extension API) | SHIPPED (design decision, #1375) | the explicit `Any::downcast_ref` match in `kl_dispatch` is the deliberate Rust-idiomatic equivalent of PyTorch's `@register_kl` + `_dispatch_kl` (a Python-runtime open-extension pattern). Rust's static analog is the closed-crate match, kept maintainable by the `kl_doc_table_matches_dispatcher` drift test that pins the doc table, the const count, and the dispatcher arms in lockstep. A `Lazy<HashMap<(TypeId,TypeId),Fn>>` registry would add indirection without enabling cross-crate extension (formulas need concrete accessors). Documented in `kl.md` REQ-8. Closes #1375. |

use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

use crate::special_fns::{digamma_scalar, lgamma_scalar};
use crate::{
    Bernoulli, Beta, Binomial, Categorical, Cauchy, Dirichlet, Distribution, Exponential, Gamma,
    Gumbel, HalfNormal, Laplace, LowRankMultivariateNormal, MultivariateNormal, Normal,
    OneHotCategorical, Pareto, Poisson, Uniform,
};

/// Euler-Mascheroni constant `γ`. Mirrors PyTorch's
/// `torch.distributions.utils.euler_constant` (used by the Gumbel KL formulas).
const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute the Kullback-Leibler divergence `KL(p || q)` between two
/// distributions.
///
/// The KL divergence is defined as:
///
/// ```text
/// KL(p || q) = integral p(x) * log(p(x) / q(x)) dx
/// ```
///
/// Returns a tensor whose shape matches the batch shape of the distributions.
///
/// # Supported pairs (f32 and f64)
///
/// | P | Q |
/// |---|---|
/// | Normal | Normal |
/// | Bernoulli | Bernoulli |
/// | Uniform | Uniform |
/// | Categorical | Categorical |
/// | Normal | Uniform |
/// | Uniform | Normal |
/// | Laplace | Laplace |
/// | Exponential | Exponential |
/// | Gamma | Gamma |
/// | Poisson | Poisson |
/// | Gamma | Exponential |
/// | Exponential | Gamma |
/// | Beta | Beta |
/// | Gumbel | Gumbel |
/// | Pareto | Pareto |
/// | HalfNormal | HalfNormal |
/// | Exponential | Normal |
/// | Gamma | Normal |
/// | Laplace | Normal |
/// | Cauchy | Cauchy |
/// | Normal | Gumbel |
/// | Gumbel | Normal |
/// | Gamma | Gumbel |
/// | Exponential | Gumbel |
/// | Uniform | Gumbel |
/// | Dirichlet | Dirichlet |
/// | Beta | Exponential |
/// | Beta | Gamma |
/// | Beta | Normal |
/// | Beta | Uniform |
/// | Pareto | Exponential |
/// | Pareto | Gamma |
/// | Pareto | Normal |
/// | Uniform | Exponential |
/// | Uniform | Gamma |
/// | Uniform | Pareto |
/// | Uniform | Beta |
/// | MultivariateNormal | MultivariateNormal |
/// | MultivariateNormal | LowRankMultivariateNormal |
/// | LowRankMultivariateNormal | MultivariateNormal |
/// | LowRankMultivariateNormal | LowRankMultivariateNormal |
/// | OneHotCategorical | OneHotCategorical |
/// | Bernoulli | Poisson |
/// | Normal | Laplace |
/// | Beta | Pareto |
/// | Exponential | Beta |
/// | Exponential | Pareto |
/// | Exponential | Uniform |
/// | Gamma | Beta |
/// | Gamma | Pareto |
/// | Gamma | Uniform |
/// | Gumbel | Beta |
/// | Gumbel | Exponential |
/// | Gumbel | Gamma |
/// | Gumbel | Pareto |
/// | Gumbel | Uniform |
/// | Laplace | Beta |
/// | Laplace | Exponential |
/// | Laplace | Gamma |
/// | Laplace | Pareto |
/// | Laplace | Uniform |
/// | Normal | Beta |
/// | Normal | Exponential |
/// | Normal | Gamma |
/// | Normal | Pareto |
/// | Pareto | Beta |
/// | Pareto | Uniform |
/// | Poisson | Bernoulli |
/// | Binomial | Binomial |
/// | Poisson | Binomial |
///
/// The same set is also reported by [`kl_supported_pair_count`].
///
/// # Errors
///
/// Returns an error if no KL formula is registered for the `(P, Q)` pair.
///
/// # Examples
///
/// ```ignore
/// use ferrotorch_distributions::{Normal, kl::kl_divergence};
/// let p = Normal::new(scalar(0.0f32)?, scalar(1.0)?)?;
/// let q = Normal::new(scalar(1.0f32)?, scalar(2.0)?)?;
/// let kl = kl_divergence(&p, &q)?;
/// ```
pub fn kl_divergence<T: Float, P, Q>(p: &P, q: &Q) -> FerrotorchResult<Tensor<T>>
where
    P: Distribution<T> + 'static,
    Q: Distribution<T> + 'static,
{
    kl_dispatch::<T>(p, q)
}

/// Number of `(P, Q)` distribution pairs for which [`kl_divergence`] has a
/// closed-form formula registered. Kept in sync with the dispatcher in
/// [`kl_dispatch`] and the supported-pairs doc table on [`kl_divergence`]
/// (drift-checked by `tests::kl_doc_table_matches_dispatcher`).
pub const fn kl_supported_pair_count() -> usize {
    KL_SUPPORTED_PAIR_COUNT
}

/// Compile-time count of registered `(P, Q)` pairs. Update this when adding
/// or removing a branch in [`kl_dispatch`] **and** the doc table on
/// [`kl_divergence`] in lockstep; the drift test enforces the invariant.
const KL_SUPPORTED_PAIR_COUNT: usize = 70;

fn kl_dispatch<T: Float>(
    p: &dyn std::any::Any,
    q: &dyn std::any::Any,
) -> FerrotorchResult<Tensor<T>> {
    // Normal-Normal
    if let (Some(pn), Some(qn)) = (p.downcast_ref::<Normal<T>>(), q.downcast_ref::<Normal<T>>()) {
        return kl_normal_normal(pn, qn);
    }
    // Bernoulli-Bernoulli
    if let (Some(pb), Some(qb)) = (
        p.downcast_ref::<Bernoulli<T>>(),
        q.downcast_ref::<Bernoulli<T>>(),
    ) {
        return kl_bernoulli_bernoulli(pb, qb);
    }
    // Uniform-Uniform
    if let (Some(pu), Some(qu)) = (
        p.downcast_ref::<Uniform<T>>(),
        q.downcast_ref::<Uniform<T>>(),
    ) {
        return kl_uniform_uniform(pu, qu);
    }
    // Categorical-Categorical
    if let (Some(pc), Some(qc)) = (
        p.downcast_ref::<Categorical<T>>(),
        q.downcast_ref::<Categorical<T>>(),
    ) {
        return kl_categorical_categorical(pc, qc);
    }
    // Normal-Uniform (kl.py:766,768): registered in `_kl_normal_infinity` ->
    // `_infinite_like(p.loc)` -> +inf. A Normal's support is all of R, which is
    // NOT contained in a Uniform's bounded [low,high], so KL(Normal||Uniform)
    // is +inf everywhere (mirrors the Normal-{Beta,Exponential,Gamma,Pareto}
    // arms below). The opposite direction (Uniform,Normal) is finite (kl.py:925).
    if let (Some(pn), Some(_qu)) = (
        p.downcast_ref::<Normal<T>>(),
        q.downcast_ref::<Uniform<T>>(),
    ) {
        return kl_infinite_like(pn.loc());
    }
    // Uniform-Normal
    if let (Some(pu), Some(qn)) = (
        p.downcast_ref::<Uniform<T>>(),
        q.downcast_ref::<Normal<T>>(),
    ) {
        return kl_uniform_normal(pu, qn);
    }
    // Laplace-Laplace
    if let (Some(pl), Some(ql)) = (
        p.downcast_ref::<Laplace<T>>(),
        q.downcast_ref::<Laplace<T>>(),
    ) {
        return kl_laplace_laplace(pl, ql);
    }
    // Exponential-Exponential
    if let (Some(pe), Some(qe)) = (
        p.downcast_ref::<Exponential<T>>(),
        q.downcast_ref::<Exponential<T>>(),
    ) {
        return kl_exponential_exponential(pe, qe);
    }
    // Gamma-Gamma
    if let (Some(pg), Some(qg)) = (p.downcast_ref::<Gamma<T>>(), q.downcast_ref::<Gamma<T>>()) {
        return kl_gamma_gamma(pg, qg);
    }
    // Poisson-Poisson
    if let (Some(pp_), Some(qp_)) = (
        p.downcast_ref::<Poisson<T>>(),
        q.downcast_ref::<Poisson<T>>(),
    ) {
        return kl_poisson_poisson(pp_, qp_);
    }
    // Gamma-Exponential: Exp(lambda) == Gamma(1, lambda), use gamma formula.
    if let (Some(pg), Some(qe)) = (
        p.downcast_ref::<Gamma<T>>(),
        q.downcast_ref::<Exponential<T>>(),
    ) {
        return kl_gamma_exponential(pg, qe);
    }
    // Exponential-Gamma: likewise.
    if let (Some(pe), Some(qg)) = (
        p.downcast_ref::<Exponential<T>>(),
        q.downcast_ref::<Gamma<T>>(),
    ) {
        return kl_exponential_gamma(pe, qg);
    }
    // Beta-Beta
    if let (Some(pb), Some(qb)) = (p.downcast_ref::<Beta<T>>(), q.downcast_ref::<Beta<T>>()) {
        return kl_beta_beta(pb, qb);
    }
    // Gumbel-Gumbel
    if let (Some(pg), Some(qg)) = (p.downcast_ref::<Gumbel<T>>(), q.downcast_ref::<Gumbel<T>>()) {
        return kl_gumbel_gumbel(pg, qg);
    }
    // Pareto-Pareto
    if let (Some(pp_), Some(qp_)) = (p.downcast_ref::<Pareto<T>>(), q.downcast_ref::<Pareto<T>>()) {
        return kl_pareto_pareto(pp_, qp_);
    }
    // HalfNormal-HalfNormal
    if let (Some(ph), Some(qh)) = (
        p.downcast_ref::<HalfNormal<T>>(),
        q.downcast_ref::<HalfNormal<T>>(),
    ) {
        return kl_halfnormal_halfnormal(ph, qh);
    }
    // Exponential-Normal (cross-family)
    if let (Some(pe), Some(qn)) = (
        p.downcast_ref::<Exponential<T>>(),
        q.downcast_ref::<Normal<T>>(),
    ) {
        return kl_exponential_normal(pe, qn);
    }
    // Gamma-Normal (cross-family)
    if let (Some(pg), Some(qn)) = (p.downcast_ref::<Gamma<T>>(), q.downcast_ref::<Normal<T>>()) {
        return kl_gamma_normal(pg, qn);
    }
    // Laplace-Normal (cross-family)
    if let (Some(pl), Some(qn)) = (
        p.downcast_ref::<Laplace<T>>(),
        q.downcast_ref::<Normal<T>>(),
    ) {
        return kl_laplace_normal(pl, qn);
    }
    // Cauchy-Cauchy
    if let (Some(pc), Some(qc)) = (p.downcast_ref::<Cauchy<T>>(), q.downcast_ref::<Cauchy<T>>()) {
        return kl_cauchy_cauchy(pc, qc);
    }
    // Normal-Gumbel (cross-family)
    if let (Some(pn), Some(qg)) = (p.downcast_ref::<Normal<T>>(), q.downcast_ref::<Gumbel<T>>()) {
        return kl_normal_gumbel(pn, qg);
    }
    // Gumbel-Normal (cross-family)
    if let (Some(pg), Some(qn)) = (p.downcast_ref::<Gumbel<T>>(), q.downcast_ref::<Normal<T>>()) {
        return kl_gumbel_normal(pg, qn);
    }
    // Gamma-Gumbel (cross-family)
    if let (Some(pg), Some(qg)) = (p.downcast_ref::<Gamma<T>>(), q.downcast_ref::<Gumbel<T>>()) {
        return kl_gamma_gumbel(pg, qg);
    }
    // Exponential-Gumbel (cross-family)
    if let (Some(pe), Some(qg)) = (
        p.downcast_ref::<Exponential<T>>(),
        q.downcast_ref::<Gumbel<T>>(),
    ) {
        return kl_exponential_gumbel(pe, qg);
    }
    // Uniform-Gumbel (cross-family)
    if let (Some(pu), Some(qg)) = (
        p.downcast_ref::<Uniform<T>>(),
        q.downcast_ref::<Gumbel<T>>(),
    ) {
        return kl_uniform_gumbel(pu, qg);
    }
    // Dirichlet-Dirichlet (multivariate same-family)
    if let (Some(pd), Some(qd)) = (
        p.downcast_ref::<Dirichlet<T>>(),
        q.downcast_ref::<Dirichlet<T>>(),
    ) {
        return kl_dirichlet_dirichlet(pd, qd);
    }
    // Beta-Exponential (cross-family)
    if let (Some(pb), Some(qe)) = (
        p.downcast_ref::<Beta<T>>(),
        q.downcast_ref::<Exponential<T>>(),
    ) {
        return kl_beta_exponential(pb, qe);
    }
    // Beta-Gamma (cross-family)
    if let (Some(pb), Some(qg)) = (p.downcast_ref::<Beta<T>>(), q.downcast_ref::<Gamma<T>>()) {
        return kl_beta_gamma(pb, qg);
    }
    // Beta-Normal (cross-family)
    if let (Some(pb), Some(qn)) = (p.downcast_ref::<Beta<T>>(), q.downcast_ref::<Normal<T>>()) {
        return kl_beta_normal(pb, qn);
    }
    // Beta-Uniform (cross-family, support-conditioned +inf)
    if let (Some(pb), Some(qu)) = (p.downcast_ref::<Beta<T>>(), q.downcast_ref::<Uniform<T>>()) {
        return kl_beta_uniform(pb, qu);
    }
    // Pareto-Exponential (cross-family, alpha<=1 -> +inf)
    if let (Some(pp_), Some(qe)) = (
        p.downcast_ref::<Pareto<T>>(),
        q.downcast_ref::<Exponential<T>>(),
    ) {
        return kl_pareto_exponential(pp_, qe);
    }
    // Pareto-Gamma (cross-family, alpha<=1 -> +inf)
    if let (Some(pp_), Some(qg)) = (p.downcast_ref::<Pareto<T>>(), q.downcast_ref::<Gamma<T>>()) {
        return kl_pareto_gamma(pp_, qg);
    }
    // Pareto-Normal (cross-family, alpha<=2 -> +inf)
    if let (Some(pp_), Some(qn)) = (p.downcast_ref::<Pareto<T>>(), q.downcast_ref::<Normal<T>>()) {
        return kl_pareto_normal(pp_, qn);
    }
    // Uniform-Exponential (cross-family, low<0 -> +inf)
    if let (Some(pu), Some(qe)) = (
        p.downcast_ref::<Uniform<T>>(),
        q.downcast_ref::<Exponential<T>>(),
    ) {
        return kl_uniform_exponential(pu, qe);
    }
    // Uniform-Gamma (cross-family, low<0 -> +inf)
    if let (Some(pu), Some(qg)) = (p.downcast_ref::<Uniform<T>>(), q.downcast_ref::<Gamma<T>>()) {
        return kl_uniform_gamma(pu, qg);
    }
    // Uniform-Pareto (cross-family, low<scale -> +inf)
    if let (Some(pu), Some(qp_)) = (
        p.downcast_ref::<Uniform<T>>(),
        q.downcast_ref::<Pareto<T>>(),
    ) {
        return kl_uniform_pareto(pu, qp_);
    }
    // Uniform-Beta (cross-family)
    if let (Some(pu), Some(qb)) = (p.downcast_ref::<Uniform<T>>(), q.downcast_ref::<Beta<T>>()) {
        return kl_uniform_beta(pu, qb);
    }
    // MultivariateNormal-MultivariateNormal (multivariate same-family)
    if let (Some(pm), Some(qm)) = (
        p.downcast_ref::<MultivariateNormal<T>>(),
        q.downcast_ref::<MultivariateNormal<T>>(),
    ) {
        return kl_multivariatenormal_multivariatenormal(pm, qm);
    }
    // MultivariateNormal-LowRankMultivariateNormal (multivariate cross-family)
    if let (Some(pm), Some(ql)) = (
        p.downcast_ref::<MultivariateNormal<T>>(),
        q.downcast_ref::<LowRankMultivariateNormal<T>>(),
    ) {
        return kl_multivariatenormal_lowrank(pm, ql);
    }
    // LowRankMultivariateNormal-MultivariateNormal (multivariate cross-family)
    if let (Some(pl), Some(qm)) = (
        p.downcast_ref::<LowRankMultivariateNormal<T>>(),
        q.downcast_ref::<MultivariateNormal<T>>(),
    ) {
        return kl_lowrank_multivariatenormal(pl, qm);
    }
    // LowRankMultivariateNormal-LowRankMultivariateNormal (multivariate same-family)
    if let (Some(pl), Some(ql)) = (
        p.downcast_ref::<LowRankMultivariateNormal<T>>(),
        q.downcast_ref::<LowRankMultivariateNormal<T>>(),
    ) {
        return kl_lowrank_lowrank(pl, ql);
    }
    // ---- #1374 / #1562: both-types-exist gaps ----
    // OneHotCategorical-OneHotCategorical (delegates to Categorical-Categorical)
    if let (Some(pc), Some(qc)) = (
        p.downcast_ref::<OneHotCategorical<T>>(),
        q.downcast_ref::<OneHotCategorical<T>>(),
    ) {
        return kl_onehotcategorical_onehotcategorical(pc, qc);
    }
    // Bernoulli-Poisson (cross-family, finite)
    if let (Some(pb), Some(qp_)) = (
        p.downcast_ref::<Bernoulli<T>>(),
        q.downcast_ref::<Poisson<T>>(),
    ) {
        return kl_bernoulli_poisson(pb, qp_);
    }
    // Normal-Laplace (cross-family, finite; symmetric partner of Laplace-Normal)
    if let (Some(pn), Some(ql)) = (
        p.downcast_ref::<Normal<T>>(),
        q.downcast_ref::<Laplace<T>>(),
    ) {
        return kl_normal_laplace(pn, ql);
    }
    // ---- support-mismatch `+inf` family (PyTorch `_infinite_like`) ----
    // Beta-Pareto (kl.py:528)
    if let (Some(pb), Some(_)) = (p.downcast_ref::<Beta<T>>(), q.downcast_ref::<Pareto<T>>()) {
        return kl_infinite_like(pb.concentration1());
    }
    // Exponential-{Beta, Pareto, Uniform} (kl.py:620-623)
    if let (Some(pe), Some(_)) = (
        p.downcast_ref::<Exponential<T>>(),
        q.downcast_ref::<Beta<T>>(),
    ) {
        return kl_infinite_like(pe.rate());
    }
    if let (Some(pe), Some(_)) = (
        p.downcast_ref::<Exponential<T>>(),
        q.downcast_ref::<Pareto<T>>(),
    ) {
        return kl_infinite_like(pe.rate());
    }
    if let (Some(pe), Some(_)) = (
        p.downcast_ref::<Exponential<T>>(),
        q.downcast_ref::<Uniform<T>>(),
    ) {
        return kl_infinite_like(pe.rate());
    }
    // Gamma-{Beta, Pareto, Uniform} (kl.py:665-668)
    if let (Some(pg), Some(_)) = (p.downcast_ref::<Gamma<T>>(), q.downcast_ref::<Beta<T>>()) {
        return kl_infinite_like(pg.concentration());
    }
    if let (Some(pg), Some(_)) = (p.downcast_ref::<Gamma<T>>(), q.downcast_ref::<Pareto<T>>()) {
        return kl_infinite_like(pg.concentration());
    }
    if let (Some(pg), Some(_)) = (p.downcast_ref::<Gamma<T>>(), q.downcast_ref::<Uniform<T>>()) {
        return kl_infinite_like(pg.concentration());
    }
    // Gumbel-{Beta, Exponential, Gamma, Pareto, Uniform} (kl.py:718-723)
    if let (Some(pg), Some(_)) = (p.downcast_ref::<Gumbel<T>>(), q.downcast_ref::<Beta<T>>()) {
        return kl_infinite_like(pg.loc());
    }
    if let (Some(pg), Some(_)) = (
        p.downcast_ref::<Gumbel<T>>(),
        q.downcast_ref::<Exponential<T>>(),
    ) {
        return kl_infinite_like(pg.loc());
    }
    if let (Some(pg), Some(_)) = (p.downcast_ref::<Gumbel<T>>(), q.downcast_ref::<Gamma<T>>()) {
        return kl_infinite_like(pg.loc());
    }
    if let (Some(pg), Some(_)) = (p.downcast_ref::<Gumbel<T>>(), q.downcast_ref::<Pareto<T>>()) {
        return kl_infinite_like(pg.loc());
    }
    if let (Some(pg), Some(_)) = (
        p.downcast_ref::<Gumbel<T>>(),
        q.downcast_ref::<Uniform<T>>(),
    ) {
        return kl_infinite_like(pg.loc());
    }
    // Laplace-{Beta, Exponential, Gamma, Pareto, Uniform} (kl.py:740-745)
    if let (Some(pl), Some(_)) = (p.downcast_ref::<Laplace<T>>(), q.downcast_ref::<Beta<T>>()) {
        return kl_infinite_like(pl.loc());
    }
    if let (Some(pl), Some(_)) = (
        p.downcast_ref::<Laplace<T>>(),
        q.downcast_ref::<Exponential<T>>(),
    ) {
        return kl_infinite_like(pl.loc());
    }
    if let (Some(pl), Some(_)) = (p.downcast_ref::<Laplace<T>>(), q.downcast_ref::<Gamma<T>>()) {
        return kl_infinite_like(pl.loc());
    }
    if let (Some(pl), Some(_)) = (
        p.downcast_ref::<Laplace<T>>(),
        q.downcast_ref::<Pareto<T>>(),
    ) {
        return kl_infinite_like(pl.loc());
    }
    if let (Some(pl), Some(_)) = (
        p.downcast_ref::<Laplace<T>>(),
        q.downcast_ref::<Uniform<T>>(),
    ) {
        return kl_infinite_like(pl.loc());
    }
    // Normal-{Beta, Exponential, Gamma, Pareto} (kl.py:761-765; Normal-Uniform
    // shares this `_kl_normal_infinity` family at kl.py:766,768 and is routed
    // through `kl_infinite_like` in the Normal-Uniform arm above).
    if let (Some(pn), Some(_)) = (p.downcast_ref::<Normal<T>>(), q.downcast_ref::<Beta<T>>()) {
        return kl_infinite_like(pn.loc());
    }
    if let (Some(pn), Some(_)) = (
        p.downcast_ref::<Normal<T>>(),
        q.downcast_ref::<Exponential<T>>(),
    ) {
        return kl_infinite_like(pn.loc());
    }
    if let (Some(pn), Some(_)) = (p.downcast_ref::<Normal<T>>(), q.downcast_ref::<Gamma<T>>()) {
        return kl_infinite_like(pn.loc());
    }
    if let (Some(pn), Some(_)) = (p.downcast_ref::<Normal<T>>(), q.downcast_ref::<Pareto<T>>()) {
        return kl_infinite_like(pn.loc());
    }
    // Pareto-{Beta, Uniform} (kl.py:795-797)
    if let (Some(pp_), Some(_)) = (p.downcast_ref::<Pareto<T>>(), q.downcast_ref::<Beta<T>>()) {
        return kl_infinite_like(pp_.scale());
    }
    if let (Some(pp_), Some(_)) = (
        p.downcast_ref::<Pareto<T>>(),
        q.downcast_ref::<Uniform<T>>(),
    ) {
        return kl_infinite_like(pp_.scale());
    }
    // Poisson-Bernoulli (kl.py:841)
    if let (Some(pp_), Some(_)) = (
        p.downcast_ref::<Poisson<T>>(),
        q.downcast_ref::<Bernoulli<T>>(),
    ) {
        return kl_infinite_like(pp_.rate());
    }
    // ---- #1374: Binomial pairs ----
    // Binomial-Binomial (kl.py:231-244, finite closed form; +inf where n_p > n_q)
    if let (Some(pb), Some(qb)) = (
        p.downcast_ref::<Binomial<T>>(),
        q.downcast_ref::<Binomial<T>>(),
    ) {
        return kl_binomial_binomial(pb, qb);
    }
    // Poisson-Binomial (kl.py:842, `_kl_poisson_infinity` -> +inf): a Poisson's
    // support is all of {0,1,2,...} which is NOT contained in a Binomial's
    // bounded {0..n}, so KL(Poisson||Binomial) is +inf everywhere (shares the
    // `_kl_poisson_infinity` body with Poisson-Bernoulli above).
    if let (Some(pp_), Some(_)) = (
        p.downcast_ref::<Poisson<T>>(),
        q.downcast_ref::<Binomial<T>>(),
    ) {
        return kl_infinite_like(pp_.rate());
    }

    Err(FerrotorchError::InvalidArgument {
        message: "No KL divergence formula registered for this distribution pair. \
                  Supported same-family pairs: Normal-Normal, Bernoulli-Bernoulli, \
                  Uniform-Uniform, Categorical-Categorical, Laplace-Laplace, \
                  Exponential-Exponential, Gamma-Gamma, Poisson-Poisson, \
                  Beta-Beta, Gumbel-Gumbel, Pareto-Pareto, HalfNormal-HalfNormal, \
                  Cauchy-Cauchy, Dirichlet-Dirichlet, \
                  MultivariateNormal-MultivariateNormal, \
                  LowRankMultivariateNormal-LowRankMultivariateNormal. \
                  Cross-family: Normal-Uniform, Uniform-Normal, \
                  Gamma-Exponential, Exponential-Gamma, Exponential-Normal, \
                  Gamma-Normal, Laplace-Normal, Normal-Gumbel, Gumbel-Normal, \
                  Gamma-Gumbel, Exponential-Gumbel, Uniform-Gumbel, \
                  Beta-Exponential, Beta-Gamma, Beta-Normal, Beta-Uniform, \
                  Pareto-Exponential, Pareto-Gamma, Pareto-Normal, \
                  Uniform-Exponential, Uniform-Gamma, Uniform-Pareto, Uniform-Beta, \
                  MultivariateNormal-LowRankMultivariateNormal, \
                  LowRankMultivariateNormal-MultivariateNormal, \
                  OneHotCategorical-OneHotCategorical, Bernoulli-Poisson, \
                  Normal-Laplace. Support-mismatch (+inf): Beta-Pareto, \
                  Exponential-{Beta,Pareto,Uniform}, Gamma-{Beta,Pareto,Uniform}, \
                  Gumbel-{Beta,Exponential,Gamma,Pareto,Uniform}, \
                  Laplace-{Beta,Exponential,Gamma,Pareto,Uniform}, \
                  Normal-{Beta,Exponential,Gamma,Pareto}, Pareto-{Beta,Uniform}, \
                  Poisson-Bernoulli, Poisson-Binomial. \
                  Discrete same-family: Binomial-Binomial."
            .into(),
    })
}

// ---------------------------------------------------------------------------
// KL divergence formulas (generic over T: Float)
// ---------------------------------------------------------------------------

/// KL(Normal(loc1, scale1) || Normal(loc2, scale2))
///
/// = 0.5 * (var_ratio + (loc1-loc2)^2/var2 - 1 - ln(var_ratio))
///
/// where var_ratio = (scale1/scale2)^2
fn kl_normal_normal<T: Float>(p: &Normal<T>, q: &Normal<T>) -> FerrotorchResult<Tensor<T>> {
    crate::fallback::check_gpu_fallback_opt_in(
        &[p.loc(), p.scale(), q.loc(), q.scale()],
        "kl_divergence(Normal, Normal)",
    )?;
    let p_loc = p.loc().data_vec()?;
    let p_scale = p.scale().data_vec()?;
    let q_loc = q.loc().data_vec()?;
    let q_scale = q.scale().data_vec()?;

    let half = T::from(0.5).unwrap();
    let one = T::from(1.0).unwrap();

    let result: Vec<T> = p_loc
        .iter()
        .zip(p_scale.iter())
        .zip(q_loc.iter().cycle())
        .zip(q_scale.iter().cycle())
        .map(|(((&pl, &ps), &ql), &qs)| {
            let var_ratio = (ps / qs) * (ps / qs);
            let mean_diff_sq = ((pl - ql) / qs) * ((pl - ql) / qs);
            half * (var_ratio + mean_diff_sq - one - var_ratio.ln())
        })
        .collect();

    Tensor::from_storage(TensorStorage::cpu(result), p.loc().shape().to_vec(), false)
}

/// KL(Bernoulli(p) || Bernoulli(q))
///
/// = p * log(p/q) + (1-p) * log((1-p)/(1-q))
fn kl_bernoulli_bernoulli<T: Float>(
    p: &Bernoulli<T>,
    q: &Bernoulli<T>,
) -> FerrotorchResult<Tensor<T>> {
    crate::fallback::check_gpu_fallback_opt_in(
        &[p.probs(), q.probs()],
        "kl_divergence(Bernoulli, Bernoulli)",
    )?;
    let p_probs = p.probs().data_vec()?;
    let q_probs = q.probs().data_vec()?;

    let one = T::from(1.0).unwrap();
    let eps = T::from(1e-7).unwrap();

    let result: Vec<T> = p_probs
        .iter()
        .zip(q_probs.iter().cycle())
        .map(|(&pp, &qp)| {
            let pp = pp.max(eps).min(one - eps);
            let qp = qp.max(eps).min(one - eps);
            pp * (pp / qp).ln() + (one - pp) * ((one - pp) / (one - qp)).ln()
        })
        .collect();

    Tensor::from_storage(
        TensorStorage::cpu(result),
        p.probs().shape().to_vec(),
        false,
    )
}

/// KL(Uniform(a1, b1) || Uniform(a2, b2))
///
/// = log((b2-a2) / (b1-a1)) if [a1,b1] subset of [a2,b2], else infinity
fn kl_uniform_uniform<T: Float>(p: &Uniform<T>, q: &Uniform<T>) -> FerrotorchResult<Tensor<T>> {
    crate::fallback::check_gpu_fallback_opt_in(
        &[p.low(), p.high(), q.low(), q.high()],
        "kl_divergence(Uniform, Uniform)",
    )?;
    let p_low = p.low().data_vec()?;
    let p_high = p.high().data_vec()?;
    let q_low = q.low().data_vec()?;
    let q_high = q.high().data_vec()?;

    let result: Vec<T> = p_low
        .iter()
        .zip(p_high.iter())
        .zip(q_low.iter().cycle())
        .zip(q_high.iter().cycle())
        .map(|(((&pl, &ph), &ql), &qh)| {
            if ql > pl || qh < ph {
                T::infinity()
            } else {
                ((qh - ql) / (ph - pl)).ln()
            }
        })
        .collect();

    Tensor::from_storage(TensorStorage::cpu(result), p.low().shape().to_vec(), false)
}

/// KL(Categorical(p) || Categorical(q))
///
/// = sum_k p_k * log(p_k / q_k)
fn kl_categorical_categorical<T: Float>(
    p: &Categorical<T>,
    q: &Categorical<T>,
) -> FerrotorchResult<Tensor<T>> {
    crate::fallback::check_gpu_fallback_opt_in(
        &[p.probs(), q.probs()],
        "kl_divergence(Categorical, Categorical)",
    )?;
    let p_probs = p.probs().data_vec()?;
    let q_probs = q.probs().data_vec()?;

    let zero = T::from(0.0).unwrap();
    let eps = T::from(1e-7).unwrap();

    // Normalize both
    let p_total: T = p_probs.iter().copied().fold(zero, |a, b| a + b);
    let q_total: T = q_probs.iter().copied().fold(zero, |a, b| a + b);

    let kl: T = p_probs
        .iter()
        .zip(q_probs.iter())
        .fold(zero, |acc, (&pp, &qp)| {
            let pp_norm = pp / p_total;
            let qp_norm = (qp / q_total).max(eps);
            if pp_norm <= eps {
                acc
            } else if qp_norm <= eps {
                T::infinity()
            } else {
                acc + pp_norm * (pp_norm / qp_norm).ln()
            }
        });

    // Categorical KL is a scalar
    Tensor::from_storage(TensorStorage::cpu(vec![kl]), vec![], false)
}

/// KL(Uniform(a, b) || Normal(loc, scale))
///
/// = -H(Uniform) + 0.5 * log(2*pi*scale^2) + (1/(2*scale^2)) * ((b-a)^2/12 + ((a+b)/2 - loc)^2)
///
/// where H(Uniform(a,b)) = log(b-a).
fn kl_uniform_normal<T: Float>(p: &Uniform<T>, q: &Normal<T>) -> FerrotorchResult<Tensor<T>> {
    crate::fallback::check_gpu_fallback_opt_in(
        &[p.low(), p.high(), q.loc(), q.scale()],
        "kl_divergence(Uniform, Normal)",
    )?;
    let p_low = p.low().data_vec()?;
    let p_high = p.high().data_vec()?;
    let q_loc = q.loc().data_vec()?;
    let q_scale = q.scale().data_vec()?;

    let half = T::from(0.5).unwrap();
    let two_pi = T::from(2.0 * std::f64::consts::PI).unwrap();
    let twelve = T::from(12.0).unwrap();
    let two = T::from(2.0).unwrap();

    let result: Vec<T> = p_low
        .iter()
        .zip(p_high.iter())
        .zip(q_loc.iter().cycle())
        .zip(q_scale.iter().cycle())
        .map(|(((&pl, &ph), &ql), &qs)| {
            let range = ph - pl;
            let entropy_uniform = range.ln();
            let log_normal_term = half * (two_pi * qs * qs).ln();
            let mean_p = (pl + ph) / two;
            let var_p = range * range / twelve;
            let mse = (mean_p - ql) * (mean_p - ql);
            let second_moment = var_p + mse;
            -entropy_uniform + log_normal_term + second_moment / (two * qs * qs)
        })
        .collect();

    Tensor::from_storage(TensorStorage::cpu(result), p.low().shape().to_vec(), false)
}

// ---------------------------------------------------------------------------
// Additional KL formulas (CL-365)
// ---------------------------------------------------------------------------

/// KL(Laplace(loc1, b1) || Laplace(loc2, b2))
///
/// = log(b2 / b1) + (b1 * exp(-|loc1 - loc2| / b1) + |loc1 - loc2|) / b2 - 1
///
/// Derived from integrating the Laplace log-density. Reduces to 0 when
/// the two distributions are identical.
fn kl_laplace_laplace<T: Float>(p: &Laplace<T>, q: &Laplace<T>) -> FerrotorchResult<Tensor<T>> {
    crate::fallback::check_gpu_fallback_opt_in(
        &[p.loc(), p.scale(), q.loc(), q.scale()],
        "kl_divergence(Laplace, Laplace)",
    )?;
    let p_loc = p.loc().data_vec()?;
    let p_scale = p.scale().data_vec()?;
    let q_loc = q.loc().data_vec()?;
    let q_scale = q.scale().data_vec()?;

    let one = T::from(1.0).unwrap();
    let zero = T::from(0.0).unwrap();

    let result: Vec<T> = p_loc
        .iter()
        .zip(p_scale.iter())
        .zip(q_loc.iter().cycle())
        .zip(q_scale.iter().cycle())
        .map(|(((&pl, &ps), &ql), &qs)| {
            let diff = pl - ql;
            let abs_diff = if diff < zero { zero - diff } else { diff };
            (qs / ps).ln() + (ps * (-abs_diff / ps).exp() + abs_diff) / qs - one
        })
        .collect();

    Tensor::from_storage(TensorStorage::cpu(result), p.loc().shape().to_vec(), false)
}

/// KL(Exponential(rate1) || Exponential(rate2))
///
/// = log(rate1 / rate2) + rate2 / rate1 - 1
fn kl_exponential_exponential<T: Float>(
    p: &Exponential<T>,
    q: &Exponential<T>,
) -> FerrotorchResult<Tensor<T>> {
    crate::fallback::check_gpu_fallback_opt_in(
        &[p.rate(), q.rate()],
        "kl_divergence(Exponential, Exponential)",
    )?;
    let p_rate = p.rate().data_vec()?;
    let q_rate = q.rate().data_vec()?;
    let one = T::from(1.0).unwrap();

    let result: Vec<T> = p_rate
        .iter()
        .zip(q_rate.iter().cycle())
        .map(|(&pr, &qr)| (pr / qr).ln() + qr / pr - one)
        .collect();

    Tensor::from_storage(TensorStorage::cpu(result), p.rate().shape().to_vec(), false)
}

/// KL(Gamma(α1, β1) || Gamma(α2, β2))
///
/// = (α1 - α2) * ψ(α1) - lnΓ(α1) + lnΓ(α2)
///   + α2 * (ln β1 - ln β2) + α1 * (β2 - β1) / β1
///
/// where ψ is the digamma function and Γ is the gamma function.
///
/// Reduces to 0 when the two distributions are identical (verified by
/// `test_kl_gamma_gamma_same`).
fn kl_gamma_gamma<T: Float>(p: &Gamma<T>, q: &Gamma<T>) -> FerrotorchResult<Tensor<T>> {
    crate::fallback::check_gpu_fallback_opt_in(
        &[p.concentration(), p.rate(), q.concentration(), q.rate()],
        "kl_divergence(Gamma, Gamma)",
    )?;
    let p_conc = p.concentration().data_vec()?;
    let p_rate = p.rate().data_vec()?;
    let q_conc = q.concentration().data_vec()?;
    let q_rate = q.rate().data_vec()?;

    let result: Vec<T> = p_conc
        .iter()
        .zip(p_rate.iter())
        .zip(q_conc.iter().cycle())
        .zip(q_rate.iter().cycle())
        .map(|(((&pa, &pb), &qa), &qb)| kl_gamma_scalar(pa, pb, qa, qb))
        .collect();

    Tensor::from_storage(
        TensorStorage::cpu(result),
        p.concentration().shape().to_vec(),
        false,
    )
}

/// Scalar KL(Gamma(α1, β1) || Gamma(α2, β2)). Factored out so the
/// Gamma-Exponential cross-family formula can reuse it.
fn kl_gamma_scalar<T: Float>(pa: T, pb: T, qa: T, qb: T) -> T {
    // (pa - qa) * digamma(pa) - lnGamma(pa) + lnGamma(qa)
    //   + qa * (ln pb - ln qb) + pa * (qb - pb) / pb
    let dig_pa = digamma_scalar(pa);
    let ln_gamma_pa = lgamma_scalar(pa);
    let ln_gamma_qa = lgamma_scalar(qa);
    (pa - qa) * dig_pa - ln_gamma_pa + ln_gamma_qa + qa * (pb.ln() - qb.ln()) + pa * (qb - pb) / pb
}

/// KL(Poisson(λ1) || Poisson(λ2))
///
/// = λ1 * (log λ1 - log λ2) - λ1 + λ2
fn kl_poisson_poisson<T: Float>(p: &Poisson<T>, q: &Poisson<T>) -> FerrotorchResult<Tensor<T>> {
    crate::fallback::check_gpu_fallback_opt_in(
        &[p.rate(), q.rate()],
        "kl_divergence(Poisson, Poisson)",
    )?;
    let p_rate = p.rate().data_vec()?;
    let q_rate = q.rate().data_vec()?;

    let result: Vec<T> = p_rate
        .iter()
        .zip(q_rate.iter().cycle())
        .map(|(&pr, &qr)| pr * (pr.ln() - qr.ln()) - pr + qr)
        .collect();

    Tensor::from_storage(TensorStorage::cpu(result), p.rate().shape().to_vec(), false)
}

/// KL(Gamma(α, β) || Exponential(λ))
///
/// Since Exp(λ) = Gamma(1, λ), this reduces to the Gamma-Gamma
/// formula with q_concentration = 1 and q_rate = λ.
fn kl_gamma_exponential<T: Float>(p: &Gamma<T>, q: &Exponential<T>) -> FerrotorchResult<Tensor<T>> {
    crate::fallback::check_gpu_fallback_opt_in(
        &[p.concentration(), p.rate(), q.rate()],
        "kl_divergence(Gamma, Exponential)",
    )?;
    let p_conc = p.concentration().data_vec()?;
    let p_rate = p.rate().data_vec()?;
    let q_rate = q.rate().data_vec()?;
    let one = T::from(1.0).unwrap();

    let result: Vec<T> = p_conc
        .iter()
        .zip(p_rate.iter())
        .zip(q_rate.iter().cycle())
        .map(|((&pa, &pb), &qb)| kl_gamma_scalar(pa, pb, one, qb))
        .collect();

    Tensor::from_storage(
        TensorStorage::cpu(result),
        p.concentration().shape().to_vec(),
        false,
    )
}

/// KL(Exponential(λ) || Gamma(α, β))
///
/// Exp(λ) = Gamma(1, λ), so this is Gamma-Gamma with
/// p_concentration = 1 and p_rate = λ.
fn kl_exponential_gamma<T: Float>(p: &Exponential<T>, q: &Gamma<T>) -> FerrotorchResult<Tensor<T>> {
    crate::fallback::check_gpu_fallback_opt_in(
        &[p.rate(), q.concentration(), q.rate()],
        "kl_divergence(Exponential, Gamma)",
    )?;
    let p_rate = p.rate().data_vec()?;
    let q_conc = q.concentration().data_vec()?;
    let q_rate = q.rate().data_vec()?;
    let one = T::from(1.0).unwrap();

    let result: Vec<T> = p_rate
        .iter()
        .zip(q_conc.iter().cycle())
        .zip(q_rate.iter().cycle())
        .map(|((&pb, &qa), &qb)| kl_gamma_scalar(one, pb, qa, qb))
        .collect();

    Tensor::from_storage(TensorStorage::cpu(result), p.rate().shape().to_vec(), false)
}

// ---------------------------------------------------------------------------
// Additional KL formulas (#1374): Beta-Beta, Gumbel-Gumbel, Pareto-Pareto,
// HalfNormal-HalfNormal + cross-family Exponential-Normal, Gamma-Normal,
// Laplace-Normal. Each mirrors a `@register_kl` body in
// `torch/distributions/kl.py`.
// ---------------------------------------------------------------------------

/// KL(Beta(α1, β1) || Beta(α2, β2)).
///
/// Mirrors `torch/distributions/kl.py:219-228` `_kl_beta_beta`:
/// ```text
/// t1 = lnΓ(α2) + lnΓ(β2) + lnΓ(α1+β1)
/// t2 = lnΓ(α1) + lnΓ(β1) + lnΓ(α2+β2)
/// t3 = (α1-α2)·ψ(α1);  t4 = (β1-β2)·ψ(β1)
/// t5 = (α2+β2 - (α1+β1))·ψ(α1+β1)
/// KL = t1 - t2 + t3 + t4 + t5
/// ```
/// where `concentration1 = α` and `concentration0 = β`.
fn kl_beta_beta<T: Float>(p: &Beta<T>, q: &Beta<T>) -> FerrotorchResult<Tensor<T>> {
    crate::fallback::check_gpu_fallback_opt_in(
        &[
            p.concentration1(),
            p.concentration0(),
            q.concentration1(),
            q.concentration0(),
        ],
        "kl_divergence(Beta, Beta)",
    )?;
    let pa = p.concentration1().data_vec()?;
    let pb = p.concentration0().data_vec()?;
    let qa = q.concentration1().data_vec()?;
    let qb = q.concentration0().data_vec()?;

    let result: Vec<T> = pa
        .iter()
        .zip(pb.iter())
        .zip(qa.iter().cycle())
        .zip(qb.iter().cycle())
        .map(|(((&pa, &pb), &qa), &qb)| {
            let sum_p = pa + pb;
            let sum_q = qa + qb;
            let t1 = lgamma_scalar(qa) + lgamma_scalar(qb) + lgamma_scalar(sum_p);
            let t2 = lgamma_scalar(pa) + lgamma_scalar(pb) + lgamma_scalar(sum_q);
            let t3 = (pa - qa) * digamma_scalar(pa);
            let t4 = (pb - qb) * digamma_scalar(pb);
            let t5 = (sum_q - sum_p) * digamma_scalar(sum_p);
            t1 - t2 + t3 + t4 + t5
        })
        .collect();

    Tensor::from_storage(
        TensorStorage::cpu(result),
        p.concentration1().shape().to_vec(),
        false,
    )
}

/// KL(Gumbel(loc1, scale1) || Gumbel(loc2, scale2)).
///
/// Mirrors `torch/distributions/kl.py:309-317` `_kl_gumbel_gumbel`:
/// ```text
/// ct1 = scale1/scale2;  ct2 = loc2/scale2;  ct3 = loc1/scale2
/// t1 = -ln(ct1) - ct2 + ct3
/// t2 = ct1·γ
/// t3 = exp(ct2 + lnΓ(1 + ct1) - ct3)
/// KL = t1 + t2 + t3 - (1 + γ)
/// ```
fn kl_gumbel_gumbel<T: Float>(p: &Gumbel<T>, q: &Gumbel<T>) -> FerrotorchResult<Tensor<T>> {
    crate::fallback::check_gpu_fallback_opt_in(
        &[p.loc(), p.scale(), q.loc(), q.scale()],
        "kl_divergence(Gumbel, Gumbel)",
    )?;
    let p_loc = p.loc().data_vec()?;
    let p_scale = p.scale().data_vec()?;
    let q_loc = q.loc().data_vec()?;
    let q_scale = q.scale().data_vec()?;
    let one = T::from(1.0).unwrap();
    let euler = T::from(EULER_GAMMA).unwrap();

    let result: Vec<T> = p_loc
        .iter()
        .zip(p_scale.iter())
        .zip(q_loc.iter().cycle())
        .zip(q_scale.iter().cycle())
        .map(|(((&pl, &ps), &ql), &qs)| {
            let ct1 = ps / qs;
            let ct2 = ql / qs;
            let ct3 = pl / qs;
            let t1 = -ct1.ln() - ct2 + ct3;
            let t2 = ct1 * euler;
            let t3 = (ct2 + lgamma_scalar(one + ct1) - ct3).exp();
            t1 + t2 + t3 - (one + euler)
        })
        .collect();

    Tensor::from_storage(TensorStorage::cpu(result), p.loc().shape().to_vec(), false)
}

/// KL(Pareto(scale1, α1) || Pareto(scale2, α2)).
///
/// Mirrors `torch/distributions/kl.py:479-488` `_kl_pareto_pareto`:
/// ```text
/// scale_ratio = scale1/scale2;  alpha_ratio = α2/α1
/// t1 = α2·ln(scale_ratio);  t2 = -ln(alpha_ratio)
/// KL = t1 + t2 + alpha_ratio - 1
/// KL = +inf when scale1 < scale2 (support lower bound of p below q's)
/// ```
fn kl_pareto_pareto<T: Float>(p: &Pareto<T>, q: &Pareto<T>) -> FerrotorchResult<Tensor<T>> {
    crate::fallback::check_gpu_fallback_opt_in(
        &[p.scale(), p.alpha(), q.scale(), q.alpha()],
        "kl_divergence(Pareto, Pareto)",
    )?;
    let p_scale = p.scale().data_vec()?;
    let p_alpha = p.alpha().data_vec()?;
    let q_scale = q.scale().data_vec()?;
    let q_alpha = q.alpha().data_vec()?;
    let one = T::from(1.0).unwrap();

    let result: Vec<T> = p_scale
        .iter()
        .zip(p_alpha.iter())
        .zip(q_scale.iter().cycle())
        .zip(q_alpha.iter().cycle())
        .map(|(((&ps, &pa), &qs), &qa)| {
            // Pareto support lower bound is `scale`; KL is +inf when p's
            // support extends below q's (p.scale < q.scale).
            if ps < qs {
                T::infinity()
            } else {
                let scale_ratio = ps / qs;
                let alpha_ratio = qa / pa;
                qa * scale_ratio.ln() - alpha_ratio.ln() + alpha_ratio - one
            }
        })
        .collect();

    Tensor::from_storage(
        TensorStorage::cpu(result),
        p.scale().shape().to_vec(),
        false,
    )
}

/// KL(HalfNormal(scale1) || HalfNormal(scale2)).
///
/// Mirrors `torch/distributions/kl.py:325-327` `_kl_halfnormal_halfnormal`,
/// which delegates to `_kl_normal_normal(p.base_dist, q.base_dist)` with both
/// base distributions `Normal(0, scale)`. With `loc = 0` this is
/// `0.5·(var_ratio - 1 - ln(var_ratio))` where `var_ratio = (s1/s2)^2`.
fn kl_halfnormal_halfnormal<T: Float>(
    p: &HalfNormal<T>,
    q: &HalfNormal<T>,
) -> FerrotorchResult<Tensor<T>> {
    crate::fallback::check_gpu_fallback_opt_in(
        &[p.scale(), q.scale()],
        "kl_divergence(HalfNormal, HalfNormal)",
    )?;
    let p_scale = p.scale().data_vec()?;
    let q_scale = q.scale().data_vec()?;
    let half = T::from(0.5).unwrap();
    let one = T::from(1.0).unwrap();

    let result: Vec<T> = p_scale
        .iter()
        .zip(q_scale.iter().cycle())
        .map(|(&ps, &qs)| {
            let var_ratio = (ps / qs) * (ps / qs);
            half * (var_ratio - one - var_ratio.ln())
        })
        .collect();

    Tensor::from_storage(
        TensorStorage::cpu(result),
        p.scale().shape().to_vec(),
        false,
    )
}

/// KL(Exponential(rate) || Normal(loc, scale)) (cross-family).
///
/// Mirrors `torch/distributions/kl.py:654-662` `_kl_exponential_normal`:
/// ```text
/// var = scale^2;  rate_sqr = rate^2
/// t1 = 0.5·ln(rate_sqr · var · 2π)
/// t2 = 1/rate_sqr;  t3 = loc/rate;  t4 = loc^2 / 2
/// KL = t1 - 1 + (t2 - t3 + t4) / var
/// ```
fn kl_exponential_normal<T: Float>(
    p: &Exponential<T>,
    q: &Normal<T>,
) -> FerrotorchResult<Tensor<T>> {
    crate::fallback::check_gpu_fallback_opt_in(
        &[p.rate(), q.loc(), q.scale()],
        "kl_divergence(Exponential, Normal)",
    )?;
    let p_rate = p.rate().data_vec()?;
    let q_loc = q.loc().data_vec()?;
    let q_scale = q.scale().data_vec()?;
    let half = T::from(0.5).unwrap();
    let one = T::from(1.0).unwrap();
    let two = T::from(2.0).unwrap();
    let two_pi = T::from(2.0 * std::f64::consts::PI).unwrap();

    let result: Vec<T> = p_rate
        .iter()
        .zip(q_loc.iter().cycle())
        .zip(q_scale.iter().cycle())
        .map(|((&rate, &loc), &scale)| {
            let var = scale * scale;
            let rate_sqr = rate * rate;
            let t1 = half * (rate_sqr * var * two_pi).ln();
            let t2 = one / rate_sqr;
            let t3 = loc / rate;
            let t4 = loc * loc / two;
            t1 - one + (t2 - t3 + t4) / var
        })
        .collect();

    Tensor::from_storage(TensorStorage::cpu(result), p.rate().shape().to_vec(), false)
}

/// KL(Gamma(α, β) || Normal(loc, scale)) (cross-family).
///
/// Mirrors `torch/distributions/kl.py:699-715` `_kl_gamma_normal`:
/// ```text
/// var = scale^2;  beta_sqr = β^2
/// t1 = 0.5·ln(beta_sqr · var · 2π) - α - lnΓ(α)
/// t2 = 0.5·(α^2 + α)/beta_sqr;  t3 = loc·α/β;  t4 = 0.5·loc^2
/// KL = t1 + (α-1)·ψ(α) + (t2 - t3 + t4)/var
/// ```
fn kl_gamma_normal<T: Float>(p: &Gamma<T>, q: &Normal<T>) -> FerrotorchResult<Tensor<T>> {
    crate::fallback::check_gpu_fallback_opt_in(
        &[p.concentration(), p.rate(), q.loc(), q.scale()],
        "kl_divergence(Gamma, Normal)",
    )?;
    let p_conc = p.concentration().data_vec()?;
    let p_rate = p.rate().data_vec()?;
    let q_loc = q.loc().data_vec()?;
    let q_scale = q.scale().data_vec()?;
    let half = T::from(0.5).unwrap();
    let one = T::from(1.0).unwrap();
    let two_pi = T::from(2.0 * std::f64::consts::PI).unwrap();

    let result: Vec<T> = p_conc
        .iter()
        .zip(p_rate.iter())
        .zip(q_loc.iter().cycle())
        .zip(q_scale.iter().cycle())
        .map(|(((&alpha, &beta), &loc), &scale)| {
            let var = scale * scale;
            let beta_sqr = beta * beta;
            let t1 = half * (beta_sqr * var * two_pi).ln() - alpha - lgamma_scalar(alpha);
            let t2 = half * (alpha * alpha + alpha) / beta_sqr;
            let t3 = loc * alpha / beta;
            let t4 = half * loc * loc;
            t1 + (alpha - one) * digamma_scalar(alpha) + (t2 - t3 + t4) / var
        })
        .collect();

    Tensor::from_storage(
        TensorStorage::cpu(result),
        p.concentration().shape().to_vec(),
        false,
    )
}

/// KL(Laplace(loc, scale) || Normal(loc2, scale2)) (cross-family).
///
/// Mirrors `torch/distributions/kl.py:750-758` `_kl_laplace_normal`:
/// ```text
/// var = scale2^2;  ratio = scale^2 / var
/// t1 = 0.5·ln(2·ratio/π)
/// t2 = 0.5·loc^2;  t3 = loc·loc2;  t4 = 0.5·loc2^2
/// KL = -t1 + ratio + (t2 - t3 + t4)/var - 1
/// ```
fn kl_laplace_normal<T: Float>(p: &Laplace<T>, q: &Normal<T>) -> FerrotorchResult<Tensor<T>> {
    crate::fallback::check_gpu_fallback_opt_in(
        &[p.loc(), p.scale(), q.loc(), q.scale()],
        "kl_divergence(Laplace, Normal)",
    )?;
    let p_loc = p.loc().data_vec()?;
    let p_scale = p.scale().data_vec()?;
    let q_loc = q.loc().data_vec()?;
    let q_scale = q.scale().data_vec()?;
    let half = T::from(0.5).unwrap();
    let one = T::from(1.0).unwrap();
    let two = T::from(2.0).unwrap();
    let pi = T::from(std::f64::consts::PI).unwrap();

    let result: Vec<T> = p_loc
        .iter()
        .zip(p_scale.iter())
        .zip(q_loc.iter().cycle())
        .zip(q_scale.iter().cycle())
        .map(|(((&pl, &ps), &ql), &qs)| {
            let var = qs * qs;
            let ratio = ps * ps / var;
            let t1 = half * (two * ratio / pi).ln();
            let t2 = half * pl * pl;
            let t3 = pl * ql;
            let t4 = half * ql * ql;
            -t1 + ratio + (t2 - t3 + t4) / var - one
        })
        .collect();

    Tensor::from_storage(TensorStorage::cpu(result), p.loc().shape().to_vec(), false)
}

// ---------------------------------------------------------------------------
// Additional KL formulas (#1374, wave-L): Cauchy-Cauchy (same-family) +
// cross-family Normal-Gumbel, Gumbel-Normal, Gamma-Gumbel, Exponential-Gumbel,
// Uniform-Gumbel. Each mirrors a `@register_kl` body in
// `torch/distributions/kl.py`.
// ---------------------------------------------------------------------------

/// KL(Cauchy(loc1, scale1) || Cauchy(loc2, scale2)).
///
/// Mirrors `torch/distributions/kl.py:952-957` `_kl_cauchy_cauchy` (from
/// <https://arxiv.org/abs/1905.10965>):
/// ```text
/// t1 = ln((s1+s2)² + (loc1-loc2)²)
/// t2 = ln(4·s1·s2)
/// KL = t1 - t2
/// ```
fn kl_cauchy_cauchy<T: Float>(p: &Cauchy<T>, q: &Cauchy<T>) -> FerrotorchResult<Tensor<T>> {
    crate::fallback::check_gpu_fallback_opt_in(
        &[p.loc(), p.scale(), q.loc(), q.scale()],
        "kl_divergence(Cauchy, Cauchy)",
    )?;
    let p_loc = p.loc().data_vec()?;
    let p_scale = p.scale().data_vec()?;
    let q_loc = q.loc().data_vec()?;
    let q_scale = q.scale().data_vec()?;
    let four = T::from(4.0).unwrap();

    let result: Vec<T> = p_loc
        .iter()
        .zip(p_scale.iter())
        .zip(q_loc.iter().cycle())
        .zip(q_scale.iter().cycle())
        .map(|(((&pl, &ps), &ql), &qs)| {
            let sum_scale = ps + qs;
            let loc_diff = pl - ql;
            let t1 = (sum_scale * sum_scale + loc_diff * loc_diff).ln();
            let t2 = (four * ps * qs).ln();
            t1 - t2
        })
        .collect();

    Tensor::from_storage(TensorStorage::cpu(result), p.loc().shape().to_vec(), false)
}

/// KL(Normal(loc, scale) || Gumbel(loc2, scale2)) (cross-family).
///
/// Mirrors `torch/distributions/kl.py:771-779` `_kl_normal_gumbel`:
/// ```text
/// mean_scale_ratio = loc/scale2;  var_scale_sqr_ratio = (scale/scale2)²
/// loc_scale_ratio = loc2/scale2
/// t1 = 0.5·ln(var_scale_sqr_ratio)
/// t2 = mean_scale_ratio - loc_scale_ratio
/// t3 = exp(-mean_scale_ratio + 0.5·var_scale_sqr_ratio + loc_scale_ratio)
/// KL = -t1 + t2 + t3 - 0.5·(1 + ln(2π))
/// ```
fn kl_normal_gumbel<T: Float>(p: &Normal<T>, q: &Gumbel<T>) -> FerrotorchResult<Tensor<T>> {
    crate::fallback::check_gpu_fallback_opt_in(
        &[p.loc(), p.scale(), q.loc(), q.scale()],
        "kl_divergence(Normal, Gumbel)",
    )?;
    let p_loc = p.loc().data_vec()?;
    let p_scale = p.scale().data_vec()?;
    let q_loc = q.loc().data_vec()?;
    let q_scale = q.scale().data_vec()?;
    let half = T::from(0.5).unwrap();
    let one = T::from(1.0).unwrap();
    let two_pi_ln = T::from((2.0 * std::f64::consts::PI).ln()).unwrap();

    let result: Vec<T> = p_loc
        .iter()
        .zip(p_scale.iter())
        .zip(q_loc.iter().cycle())
        .zip(q_scale.iter().cycle())
        .map(|(((&pl, &ps), &ql), &qs)| {
            let mean_scale_ratio = pl / qs;
            let var_scale_sqr_ratio = (ps / qs) * (ps / qs);
            let loc_scale_ratio = ql / qs;
            let t1 = half * var_scale_sqr_ratio.ln();
            let t2 = mean_scale_ratio - loc_scale_ratio;
            let t3 = (-mean_scale_ratio + half * var_scale_sqr_ratio + loc_scale_ratio).exp();
            -t1 + t2 + t3 - half * (one + two_pi_ln)
        })
        .collect();

    Tensor::from_storage(TensorStorage::cpu(result), p.loc().shape().to_vec(), false)
}

/// KL(Gumbel(loc, scale) || Normal(loc2, scale2)) (cross-family).
///
/// Mirrors `torch/distributions/kl.py:731-737` `_kl_gumbel_normal`:
/// ```text
/// param_ratio = scale/scale2
/// t1 = ln(param_ratio / sqrt(2π))
/// t2 = (π·param_ratio·0.5)² / 3
/// t3 = 0.5·((loc + scale·γ - loc2)/scale2)²
/// KL = -t1 + t2 + t3 - (γ + 1)
/// ```
fn kl_gumbel_normal<T: Float>(p: &Gumbel<T>, q: &Normal<T>) -> FerrotorchResult<Tensor<T>> {
    crate::fallback::check_gpu_fallback_opt_in(
        &[p.loc(), p.scale(), q.loc(), q.scale()],
        "kl_divergence(Gumbel, Normal)",
    )?;
    let p_loc = p.loc().data_vec()?;
    let p_scale = p.scale().data_vec()?;
    let q_loc = q.loc().data_vec()?;
    let q_scale = q.scale().data_vec()?;
    let half = T::from(0.5).unwrap();
    let one = T::from(1.0).unwrap();
    let three = T::from(3.0).unwrap();
    let pi = T::from(std::f64::consts::PI).unwrap();
    let sqrt_two_pi = T::from((2.0 * std::f64::consts::PI).sqrt()).unwrap();
    let euler = T::from(EULER_GAMMA).unwrap();

    let result: Vec<T> = p_loc
        .iter()
        .zip(p_scale.iter())
        .zip(q_loc.iter().cycle())
        .zip(q_scale.iter().cycle())
        .map(|(((&pl, &ps), &ql), &qs)| {
            let param_ratio = ps / qs;
            let t1 = (param_ratio / sqrt_two_pi).ln();
            let t2_inner = pi * param_ratio * half;
            let t2 = t2_inner * t2_inner / three;
            let t3_inner = (pl + ps * euler - ql) / qs;
            let t3 = t3_inner * t3_inner * half;
            -t1 + t2 + t3 - (euler + one)
        })
        .collect();

    Tensor::from_storage(TensorStorage::cpu(result), p.loc().shape().to_vec(), false)
}

/// KL(Gamma(α, β) || Gumbel(loc, scale)) (cross-family).
///
/// Mirrors `torch/distributions/kl.py:678-693` `_kl_gamma_gumbel`:
/// ```text
/// beta_scale_prod = β·scale;  loc_scale_ratio = loc/scale
/// t1 = (α-1)·ψ(α) - lnΓ(α) - α
/// t2 = ln(beta_scale_prod) + α/beta_scale_prod
/// t3 = exp(loc_scale_ratio)·(1 + 1/beta_scale_prod)^(-α) - loc_scale_ratio
/// KL = t1 + t2 + t3
/// ```
fn kl_gamma_gumbel<T: Float>(p: &Gamma<T>, q: &Gumbel<T>) -> FerrotorchResult<Tensor<T>> {
    crate::fallback::check_gpu_fallback_opt_in(
        &[p.concentration(), p.rate(), q.loc(), q.scale()],
        "kl_divergence(Gamma, Gumbel)",
    )?;
    let p_conc = p.concentration().data_vec()?;
    let p_rate = p.rate().data_vec()?;
    let q_loc = q.loc().data_vec()?;
    let q_scale = q.scale().data_vec()?;
    let one = T::from(1.0).unwrap();

    let result: Vec<T> = p_conc
        .iter()
        .zip(p_rate.iter())
        .zip(q_loc.iter().cycle())
        .zip(q_scale.iter().cycle())
        .map(|(((&alpha, &beta), &loc), &scale)| {
            let beta_scale_prod = beta * scale;
            let loc_scale_ratio = loc / scale;
            let t1 = (alpha - one) * digamma_scalar(alpha) - lgamma_scalar(alpha) - alpha;
            let t2 = beta_scale_prod.ln() + alpha / beta_scale_prod;
            let t3 = loc_scale_ratio.exp() * (one + one / beta_scale_prod).powf(-alpha)
                - loc_scale_ratio;
            t1 + t2 + t3
        })
        .collect();

    Tensor::from_storage(
        TensorStorage::cpu(result),
        p.concentration().shape().to_vec(),
        false,
    )
}

/// KL(Exponential(rate) || Gumbel(loc, scale)) (cross-family).
///
/// Mirrors `torch/distributions/kl.py:641-649` `_kl_exponential_gumbel`:
/// ```text
/// scale_rate_prod = rate·scale;  loc_scale_ratio = loc/scale
/// t1 = ln(scale_rate_prod) - 1
/// t2 = exp(loc_scale_ratio)·scale_rate_prod / (scale_rate_prod + 1)
/// t3 = 1/scale_rate_prod
/// KL = t1 - loc_scale_ratio + t2 + t3
/// ```
fn kl_exponential_gumbel<T: Float>(
    p: &Exponential<T>,
    q: &Gumbel<T>,
) -> FerrotorchResult<Tensor<T>> {
    crate::fallback::check_gpu_fallback_opt_in(
        &[p.rate(), q.loc(), q.scale()],
        "kl_divergence(Exponential, Gumbel)",
    )?;
    let p_rate = p.rate().data_vec()?;
    let q_loc = q.loc().data_vec()?;
    let q_scale = q.scale().data_vec()?;
    let one = T::from(1.0).unwrap();

    let result: Vec<T> = p_rate
        .iter()
        .zip(q_loc.iter().cycle())
        .zip(q_scale.iter().cycle())
        .map(|((&rate, &loc), &scale)| {
            let scale_rate_prod = rate * scale;
            let loc_scale_ratio = loc / scale;
            let t1 = scale_rate_prod.ln() - one;
            let t2 = loc_scale_ratio.exp() * scale_rate_prod / (scale_rate_prod + one);
            let t3 = one / scale_rate_prod;
            t1 - loc_scale_ratio + t2 + t3
        })
        .collect();

    Tensor::from_storage(TensorStorage::cpu(result), p.rate().shape().to_vec(), false)
}

/// KL(Uniform(a, b) || Gumbel(loc, scale)) (cross-family).
///
/// Mirrors `torch/distributions/kl.py:912-919` `_kl_uniform_gumbel`:
/// ```text
/// common_term = scale/(b-a)
/// high_loc_diff = (b-loc)/scale;  low_loc_diff = (a-loc)/scale
/// t1 = ln(common_term) + 0.5·(high_loc_diff + low_loc_diff)
/// t2 = common_term·(exp(-high_loc_diff) - exp(-low_loc_diff))
/// KL = t1 - t2
/// ```
fn kl_uniform_gumbel<T: Float>(p: &Uniform<T>, q: &Gumbel<T>) -> FerrotorchResult<Tensor<T>> {
    crate::fallback::check_gpu_fallback_opt_in(
        &[p.low(), p.high(), q.loc(), q.scale()],
        "kl_divergence(Uniform, Gumbel)",
    )?;
    let p_low = p.low().data_vec()?;
    let p_high = p.high().data_vec()?;
    let q_loc = q.loc().data_vec()?;
    let q_scale = q.scale().data_vec()?;
    let half = T::from(0.5).unwrap();

    let result: Vec<T> = p_low
        .iter()
        .zip(p_high.iter())
        .zip(q_loc.iter().cycle())
        .zip(q_scale.iter().cycle())
        .map(|(((&low, &high), &loc), &scale)| {
            let common_term = scale / (high - low);
            let high_loc_diff = (high - loc) / scale;
            let low_loc_diff = (low - loc) / scale;
            let t1 = common_term.ln() + half * (high_loc_diff + low_loc_diff);
            let t2 = common_term * ((-high_loc_diff).exp() - (-low_loc_diff).exp());
            t1 - t2
        })
        .collect();

    Tensor::from_storage(TensorStorage::cpu(result), p.low().shape().to_vec(), false)
}

// ---------------------------------------------------------------------------
// Additional KL formulas (#1374, wave-M): Dirichlet-Dirichlet (multivariate
// same-family) + Beta/Pareto/Uniform cross-family pairs. Each mirrors a
// `@register_kl` body in `torch/distributions/kl.py`.
// ---------------------------------------------------------------------------

/// Scalar entropy of `Beta(α, β)`, matching `Beta::entropy` in `beta.rs`
/// (`H = lnB(α,β) - (α-1)ψ(α) - (β-1)ψ(β) + (α+β-2)ψ(α+β)`). Factored as a
/// free scalar helper so the Beta cross-family KL bodies can reuse it inside
/// their per-element map.
fn beta_entropy_scalar<T: Float>(a: T, b: T) -> T {
    let one = T::from(1.0).unwrap();
    let two = T::from(2.0).unwrap();
    let lbeta = lgamma_scalar(a) + lgamma_scalar(b) - lgamma_scalar(a + b);
    lbeta - (a - one) * digamma_scalar(a) - (b - one) * digamma_scalar(b)
        + (a + b - two) * digamma_scalar(a + b)
}

/// KL(Dirichlet(α) || Dirichlet(β)) (multivariate same-family).
///
/// Mirrors `torch/distributions/kl.py:263-273` `_kl_dirichlet_dirichlet`:
/// ```text
/// sum_p = Σ_k α_k;  sum_q = Σ_k β_k
/// t1 = lnΓ(sum_p) - lnΓ(sum_q)
/// t2 = Σ_k (lnΓ(α_k) - lnΓ(β_k))
/// t3 = α_k - β_k
/// t4 = ψ(α_k) - ψ(sum_p)
/// KL = t1 - t2 + Σ_k t3·t4
/// ```
/// One scalar KL per batch row; output shape == `batch_shape`.
fn kl_dirichlet_dirichlet<T: Float>(
    p: &Dirichlet<T>,
    q: &Dirichlet<T>,
) -> FerrotorchResult<Tensor<T>> {
    crate::fallback::check_gpu_fallback_opt_in(
        &[p.concentration(), q.concentration()],
        "kl_divergence(Dirichlet, Dirichlet)",
    )?;
    let pa = p.concentration().data_vec()?;
    let qa = q.concentration().data_vec()?;
    let k = p.num_categories();
    if q.num_categories() != k {
        return Err(FerrotorchError::InvalidArgument {
            message: "kl_divergence(Dirichlet, Dirichlet): event dims (K) must match".into(),
        });
    }
    let zero = T::from(0.0).unwrap();
    let b = pa.len() / k;

    let mut out = Vec::with_capacity(b);
    for bi in 0..b {
        let prow = &pa[bi * k..bi * k + k];
        // q broadcasts over p's batch rows when q has a single row.
        let qbi = if qa.len() == k { 0 } else { bi };
        let qrow = &qa[qbi * k..qbi * k + k];
        let sum_p: T = prow.iter().copied().fold(zero, |acc, x| acc + x);
        let sum_q: T = qrow.iter().copied().fold(zero, |acc, x| acc + x);
        let t1 = lgamma_scalar(sum_p) - lgamma_scalar(sum_q);
        let dig_sum_p = digamma_scalar(sum_p);
        let mut t2 = zero;
        let mut t34 = zero;
        for (&ak, &bk) in prow.iter().zip(qrow.iter()) {
            t2 += lgamma_scalar(ak) - lgamma_scalar(bk);
            t34 += (ak - bk) * (digamma_scalar(ak) - dig_sum_p);
        }
        out.push(t1 - t2 + t34);
    }
    Tensor::from_storage(TensorStorage::cpu(out), p.batch_shape(), false)
}

/// KL(Beta(α, β) || Exponential(rate)) (cross-family).
///
/// Mirrors `torch/distributions/kl.py:533-539` `_kl_beta_exponential`:
/// ```text
/// KL = -H(Beta) - ln(rate) + rate·(α / (α+β))
/// ```
fn kl_beta_exponential<T: Float>(p: &Beta<T>, q: &Exponential<T>) -> FerrotorchResult<Tensor<T>> {
    crate::fallback::check_gpu_fallback_opt_in(
        &[p.concentration1(), p.concentration0(), q.rate()],
        "kl_divergence(Beta, Exponential)",
    )?;
    let a = p.concentration1().data_vec()?;
    let b = p.concentration0().data_vec()?;
    let rate = q.rate().data_vec()?;

    let result: Vec<T> = a
        .iter()
        .zip(b.iter())
        .zip(rate.iter().cycle())
        .map(|((&a, &b), &r)| -beta_entropy_scalar(a, b) - r.ln() + r * (a / (a + b)))
        .collect();

    Tensor::from_storage(
        TensorStorage::cpu(result),
        p.concentration1().shape().to_vec(),
        false,
    )
}

/// KL(Beta(α, β) || Gamma(conc, rate)) (cross-family).
///
/// Mirrors `torch/distributions/kl.py:542-552` `_kl_beta_gamma`:
/// ```text
/// t1 = -H(Beta)
/// t2 = lnΓ(c) - c·ln(rate)
/// t3 = (c-1)·(ψ(α) - ψ(α+β))
/// t4 = rate·α/(α+β)
/// KL = t1 + t2 - t3 + t4
/// ```
fn kl_beta_gamma<T: Float>(p: &Beta<T>, q: &Gamma<T>) -> FerrotorchResult<Tensor<T>> {
    crate::fallback::check_gpu_fallback_opt_in(
        &[
            p.concentration1(),
            p.concentration0(),
            q.concentration(),
            q.rate(),
        ],
        "kl_divergence(Beta, Gamma)",
    )?;
    let a = p.concentration1().data_vec()?;
    let b = p.concentration0().data_vec()?;
    let conc = q.concentration().data_vec()?;
    let rate = q.rate().data_vec()?;
    let one = T::from(1.0).unwrap();

    let result: Vec<T> = a
        .iter()
        .zip(b.iter())
        .zip(conc.iter().cycle())
        .zip(rate.iter().cycle())
        .map(|(((&a, &b), &c), &r)| {
            let t1 = -beta_entropy_scalar(a, b);
            let t2 = lgamma_scalar(c) - c * r.ln();
            let t3 = (c - one) * (digamma_scalar(a) - digamma_scalar(a + b));
            let t4 = r * a / (a + b);
            t1 + t2 - t3 + t4
        })
        .collect();

    Tensor::from_storage(
        TensorStorage::cpu(result),
        p.concentration1().shape().to_vec(),
        false,
    )
}

/// KL(Beta(α, β) || Normal(loc, scale)) (cross-family).
///
/// Mirrors `torch/distributions/kl.py:556-568` `_kl_beta_normal`:
/// ```text
/// E_beta = α/(α+β);  var = scale²
/// t1 = -H(Beta)
/// t2 = 0.5·ln(var·2π)
/// t3 = 0.5·(E_beta(1-E_beta)/(α+β+1) + E_beta²)
/// t4 = loc·E_beta;  t5 = 0.5·loc²
/// KL = t1 + t2 + (t3 - t4 + t5)/var
/// ```
fn kl_beta_normal<T: Float>(p: &Beta<T>, q: &Normal<T>) -> FerrotorchResult<Tensor<T>> {
    crate::fallback::check_gpu_fallback_opt_in(
        &[p.concentration1(), p.concentration0(), q.loc(), q.scale()],
        "kl_divergence(Beta, Normal)",
    )?;
    let a = p.concentration1().data_vec()?;
    let b = p.concentration0().data_vec()?;
    let loc = q.loc().data_vec()?;
    let scale = q.scale().data_vec()?;
    let half = T::from(0.5).unwrap();
    let one = T::from(1.0).unwrap();
    let two_pi = T::from(2.0 * std::f64::consts::PI).unwrap();

    let result: Vec<T> = a
        .iter()
        .zip(b.iter())
        .zip(loc.iter().cycle())
        .zip(scale.iter().cycle())
        .map(|(((&a, &b), &loc), &scale)| {
            let e_beta = a / (a + b);
            let var = scale * scale;
            let t1 = -beta_entropy_scalar(a, b);
            let t2 = half * (var * two_pi).ln();
            let t3 = half * (e_beta * (one - e_beta) / (a + b + one) + e_beta * e_beta);
            let t4 = loc * e_beta;
            let t5 = half * loc * loc;
            t1 + t2 + (t3 - t4 + t5) / var
        })
        .collect();

    Tensor::from_storage(
        TensorStorage::cpu(result),
        p.concentration1().shape().to_vec(),
        false,
    )
}

/// KL(Beta(α, β) || Uniform(low, high)) (cross-family).
///
/// Mirrors `torch/distributions/kl.py:571-577` `_kl_beta_uniform`:
/// ```text
/// KL = -H(Beta) + ln(high - low)
/// KL = +inf when the Uniform support [low,high] does not cover the Beta
///      support [0,1] (low > 0 or high < 1).
/// ```
fn kl_beta_uniform<T: Float>(p: &Beta<T>, q: &Uniform<T>) -> FerrotorchResult<Tensor<T>> {
    crate::fallback::check_gpu_fallback_opt_in(
        &[p.concentration1(), p.concentration0(), q.low(), q.high()],
        "kl_divergence(Beta, Uniform)",
    )?;
    let a = p.concentration1().data_vec()?;
    let b = p.concentration0().data_vec()?;
    let low = q.low().data_vec()?;
    let high = q.high().data_vec()?;
    let zero = T::from(0.0).unwrap();
    let one = T::from(1.0).unwrap();

    let result: Vec<T> = a
        .iter()
        .zip(b.iter())
        .zip(low.iter().cycle())
        .zip(high.iter().cycle())
        .map(|(((&a, &b), &low), &high)| {
            if low > zero || high < one {
                T::infinity()
            } else {
                -beta_entropy_scalar(a, b) + (high - low).ln()
            }
        })
        .collect();

    Tensor::from_storage(
        TensorStorage::cpu(result),
        p.concentration1().shape().to_vec(),
        false,
    )
}

/// KL(Pareto(scale, α) || Exponential(rate)) (cross-family).
///
/// Mirrors `torch/distributions/kl.py:802-810` `_kl_pareto_exponential`:
/// ```text
/// scale_rate_prod = scale·rate
/// t1 = ln(α / scale_rate_prod);  t2 = 1/α
/// t3 = α·scale_rate_prod/(α-1)
/// KL = t1 - t2 + t3 - 1;  +inf when α <= 1.
/// ```
fn kl_pareto_exponential<T: Float>(
    p: &Pareto<T>,
    q: &Exponential<T>,
) -> FerrotorchResult<Tensor<T>> {
    crate::fallback::check_gpu_fallback_opt_in(
        &[p.scale(), p.alpha(), q.rate()],
        "kl_divergence(Pareto, Exponential)",
    )?;
    let scale = p.scale().data_vec()?;
    let alpha = p.alpha().data_vec()?;
    let rate = q.rate().data_vec()?;
    let one = T::from(1.0).unwrap();

    let result: Vec<T> = scale
        .iter()
        .zip(alpha.iter())
        .zip(rate.iter().cycle())
        .map(|((&s, &a), &r)| {
            if a <= one {
                T::infinity()
            } else {
                let srp = s * r;
                (a / srp).ln() - one / a + a * srp / (a - one) - one
            }
        })
        .collect();

    Tensor::from_storage(
        TensorStorage::cpu(result),
        p.scale().shape().to_vec(),
        false,
    )
}

/// KL(Pareto(scale, α) || Gamma(conc, rate)) (cross-family).
///
/// Mirrors `torch/distributions/kl.py:813-825` `_kl_pareto_gamma`:
/// ```text
/// common = ln(scale) + 1/α
/// t1 = ln(α) - common
/// t2 = lnΓ(c) - c·ln(rate)
/// t3 = (1-c)·common
/// t4 = rate·α·scale/(α-1)
/// KL = t1 + t2 + t3 + t4 - 1;  +inf when α <= 1.
/// ```
fn kl_pareto_gamma<T: Float>(p: &Pareto<T>, q: &Gamma<T>) -> FerrotorchResult<Tensor<T>> {
    crate::fallback::check_gpu_fallback_opt_in(
        &[p.scale(), p.alpha(), q.concentration(), q.rate()],
        "kl_divergence(Pareto, Gamma)",
    )?;
    let scale = p.scale().data_vec()?;
    let alpha = p.alpha().data_vec()?;
    let conc = q.concentration().data_vec()?;
    let rate = q.rate().data_vec()?;
    let one = T::from(1.0).unwrap();

    let result: Vec<T> = scale
        .iter()
        .zip(alpha.iter())
        .zip(conc.iter().cycle())
        .zip(rate.iter().cycle())
        .map(|(((&s, &a), &c), &r)| {
            if a <= one {
                T::infinity()
            } else {
                let common = s.ln() + one / a;
                let t1 = a.ln() - common;
                let t2 = lgamma_scalar(c) - c * r.ln();
                let t3 = (one - c) * common;
                let t4 = r * a * s / (a - one);
                t1 + t2 + t3 + t4 - one
            }
        })
        .collect();

    Tensor::from_storage(
        TensorStorage::cpu(result),
        p.scale().shape().to_vec(),
        false,
    )
}

/// KL(Pareto(scale, α) || Normal(loc, scale2)) (cross-family).
///
/// Mirrors `torch/distributions/kl.py:828-838` `_kl_pareto_normal`:
/// ```text
/// var = 2·scale2²;  common = scale/(α-1)
/// t1 = ln(sqrt(2π)·scale2·α/scale)
/// t2 = 1/α
/// t3 = α·common²/(α-2)
/// t4 = (α·common - loc)²
/// KL = t1 - t2 + (t3 + t4)/var - 1;  +inf when α <= 2.
/// ```
fn kl_pareto_normal<T: Float>(p: &Pareto<T>, q: &Normal<T>) -> FerrotorchResult<Tensor<T>> {
    crate::fallback::check_gpu_fallback_opt_in(
        &[p.scale(), p.alpha(), q.loc(), q.scale()],
        "kl_divergence(Pareto, Normal)",
    )?;
    let scale = p.scale().data_vec()?;
    let alpha = p.alpha().data_vec()?;
    let loc = q.loc().data_vec()?;
    let scale2 = q.scale().data_vec()?;
    let one = T::from(1.0).unwrap();
    let two = T::from(2.0).unwrap();
    let sqrt_two_pi = T::from((2.0 * std::f64::consts::PI).sqrt()).unwrap();

    let result: Vec<T> = scale
        .iter()
        .zip(alpha.iter())
        .zip(loc.iter().cycle())
        .zip(scale2.iter().cycle())
        .map(|(((&s, &a), &loc), &s2)| {
            if a <= two {
                T::infinity()
            } else {
                let var = two * s2 * s2;
                let common = s / (a - one);
                let t1 = (sqrt_two_pi * s2 * a / s).ln();
                let t2 = one / a;
                let t3 = a * common * common / (a - two);
                let t4_inner = a * common - loc;
                let t4 = t4_inner * t4_inner;
                t1 - t2 + (t3 + t4) / var - one
            }
        })
        .collect();

    Tensor::from_storage(
        TensorStorage::cpu(result),
        p.scale().shape().to_vec(),
        false,
    )
}

/// KL(Uniform(low, high) || Exponential(rate)) (cross-family).
///
/// Mirrors `torch/distributions/kl.py:889-893` `_kl_uniform_exponential`:
/// ```text
/// KL = rate·(high+low)/2 - ln((high-low)·rate);  +inf when low < 0.
/// ```
fn kl_uniform_exponential<T: Float>(
    p: &Uniform<T>,
    q: &Exponential<T>,
) -> FerrotorchResult<Tensor<T>> {
    crate::fallback::check_gpu_fallback_opt_in(
        &[p.low(), p.high(), q.rate()],
        "kl_divergence(Uniform, Exponential)",
    )?;
    let low = p.low().data_vec()?;
    let high = p.high().data_vec()?;
    let rate = q.rate().data_vec()?;
    let zero = T::from(0.0).unwrap();
    let two = T::from(2.0).unwrap();

    let result: Vec<T> = low
        .iter()
        .zip(high.iter())
        .zip(rate.iter().cycle())
        .map(|((&low, &high), &r)| {
            if low < zero {
                T::infinity()
            } else {
                r * (high + low) / two - ((high - low) * r).ln()
            }
        })
        .collect();

    Tensor::from_storage(TensorStorage::cpu(result), p.low().shape().to_vec(), false)
}

/// KL(Uniform(low, high) || Gamma(conc, rate)) (cross-family).
///
/// Mirrors `torch/distributions/kl.py:896-909` `_kl_uniform_gamma`:
/// ```text
/// common = high - low
/// t1 = ln(common)
/// t2 = lnΓ(c) - c·ln(rate)
/// t3 = (1-c)·(x·ln(x)|_low^high - common)/common      [x·ln(x), 0·ln0 = 0]
/// t4 = rate·(high+low)/2
/// KL = -t1 + t2 + t3 + t4;  +inf when low < 0.
/// ```
fn kl_uniform_gamma<T: Float>(p: &Uniform<T>, q: &Gamma<T>) -> FerrotorchResult<Tensor<T>> {
    crate::fallback::check_gpu_fallback_opt_in(
        &[p.low(), p.high(), q.concentration(), q.rate()],
        "kl_divergence(Uniform, Gamma)",
    )?;
    let low = p.low().data_vec()?;
    let high = p.high().data_vec()?;
    let conc = q.concentration().data_vec()?;
    let rate = q.rate().data_vec()?;
    let zero = T::from(0.0).unwrap();
    let one = T::from(1.0).unwrap();
    let two = T::from(2.0).unwrap();

    let result: Vec<T> = low
        .iter()
        .zip(high.iter())
        .zip(conc.iter().cycle())
        .zip(rate.iter().cycle())
        .map(|(((&low, &high), &c), &r)| {
            if low < zero {
                T::infinity()
            } else {
                let common = high - low;
                let t1 = common.ln();
                let t2 = lgamma_scalar(c) - c * r.ln();
                let t3 = (one - c) * (x_log_x(high) - x_log_x(low) - common) / common;
                let t4 = r * (high + low) / two;
                -t1 + t2 + t3 + t4
            }
        })
        .collect();

    Tensor::from_storage(TensorStorage::cpu(result), p.low().shape().to_vec(), false)
}

/// `x·ln(x)` with the convention `0·ln(0) = 0`. Mirrors
/// `torch.special.xlogy(x, x)` used by `_x_log_x` in `kl.py:148-152`.
fn x_log_x<T: Float>(x: T) -> T {
    let zero = T::from(0.0).unwrap();
    if x <= zero { zero } else { x * x.ln() }
}

/// KL(Uniform(low, high) || Pareto(scale, α)) (cross-family).
///
/// Mirrors `torch/distributions/kl.py:934-941` `_kl_uniform_pareto`:
/// ```text
/// support = high - low
/// t1 = ln(α · scale^α · support)
/// t2 = (x·ln(x)|_low^high - support)/support
/// KL = t2·(α+1) - t1;  +inf when low < scale (Uniform support below Pareto's).
/// ```
fn kl_uniform_pareto<T: Float>(p: &Uniform<T>, q: &Pareto<T>) -> FerrotorchResult<Tensor<T>> {
    crate::fallback::check_gpu_fallback_opt_in(
        &[p.low(), p.high(), q.scale(), q.alpha()],
        "kl_divergence(Uniform, Pareto)",
    )?;
    let low = p.low().data_vec()?;
    let high = p.high().data_vec()?;
    let scale = q.scale().data_vec()?;
    let alpha = q.alpha().data_vec()?;
    let one = T::from(1.0).unwrap();

    let result: Vec<T> = low
        .iter()
        .zip(high.iter())
        .zip(scale.iter().cycle())
        .zip(alpha.iter().cycle())
        .map(|(((&low, &high), &s), &a)| {
            if low < s {
                T::infinity()
            } else {
                let support = high - low;
                let t1 = (a * s.powf(a) * support).ln();
                let t2 = (x_log_x(high) - x_log_x(low) - support) / support;
                t2 * (a + one) - t1
            }
        })
        .collect();

    Tensor::from_storage(TensorStorage::cpu(result), p.low().shape().to_vec(), false)
}

/// KL(Uniform(low, high) || Beta(α, β)) (cross-family).
///
/// Mirrors `torch/distributions/kl.py:847-869` `_kl_uniform_beta`:
/// ```text
/// common = high - low
/// t1 = ln(common)
/// t2 = (α-1)·(x·ln(x)|_low^high - common)/common
/// t3 = (β-1)·((1-x)·ln(1-x)|_low^high + common)/common
/// t4 = lnΓ(α) + lnΓ(β) - lnΓ(α+β)
/// KL = t3 + t4 - t1 - t2;  +inf when the Uniform support escapes [0,1].
/// ```
fn kl_uniform_beta<T: Float>(p: &Uniform<T>, q: &Beta<T>) -> FerrotorchResult<Tensor<T>> {
    crate::fallback::check_gpu_fallback_opt_in(
        &[p.low(), p.high(), q.concentration1(), q.concentration0()],
        "kl_divergence(Uniform, Beta)",
    )?;
    let low = p.low().data_vec()?;
    let high = p.high().data_vec()?;
    let a = q.concentration1().data_vec()?;
    let b = q.concentration0().data_vec()?;
    let zero = T::from(0.0).unwrap();
    let one = T::from(1.0).unwrap();

    let result: Vec<T> = low
        .iter()
        .zip(high.iter())
        .zip(a.iter().cycle())
        .zip(b.iter().cycle())
        .map(|(((&low, &high), &a), &b)| {
            if low < zero || high > one {
                T::infinity()
            } else {
                let common = high - low;
                let t1 = common.ln();
                let t2 = (a - one) * (x_log_x(high) - x_log_x(low) - common) / common;
                let t3 = (b - one) * (x_log_x(one - high) - x_log_x(one - low) + common) / common;
                let t4 = lgamma_scalar(a) + lgamma_scalar(b) - lgamma_scalar(a + b);
                t3 + t4 - t1 - t2
            }
        })
        .collect();

    Tensor::from_storage(TensorStorage::cpu(result), p.low().shape().to_vec(), false)
}

// ---------------------------------------------------------------------------
// Additional KL formulas (#1374, wave-N): MultivariateNormal &
// LowRankMultivariateNormal pairs (MVN-MVN, MVN-LowRank, LowRank-MVN,
// LowRank-LowRank). Each mirrors a `@register_kl` body in
// `torch/distributions/kl.py`. ferrotorch's MVN/LowRankMVN carry a 1-D `loc`
// and a dense lower-triangular `scale_tril` (`[n, n]`); the LowRank variant
// materialises its dense Cholesky on demand via `scale_tril()`. The dense
// formula is value-identical to PyTorch's Woodbury form (R-DEV-1: match the
// numeric contract; the Woodbury path is an O(d·r²) optimisation, not a
// different KL).
// ---------------------------------------------------------------------------

/// Forward-substitution solve of the lower-triangular system `L · X = B`.
///
/// `l` is row-major `[n, n]` lower-triangular (strict upper triangle ignored);
/// `b` is row-major `[n, m]`. Returns `X` row-major `[n, m]`. Mirrors
/// `torch.linalg.solve_triangular(L, B, upper=False)` used by
/// `_batch_trace_XXT` / `_batch_mahalanobis` in
/// `torch/distributions/kl.py:442-464`.
fn solve_lower_tri<T: Float>(l: &[T], b: &[T], n: usize, m: usize) -> Vec<T> {
    let zero = T::from(0.0).unwrap();
    let mut x = vec![zero; n * m];
    for col in 0..m {
        for row in 0..n {
            // x[row,col] = (b[row,col] - Σ_{k<row} L[row,k]·x[k,col]) / L[row,row]
            let mut acc = b[row * m + col];
            for k in 0..row {
                acc = acc - l[row * n + k] * x[k * m + col];
            }
            x[row * m + col] = acc / l[row * n + row];
        }
    }
    x
}

/// Sum of squares of the forward-substitution solution of `qL · X = b_mat`.
/// This is `_batch_trace_XXT(solve_triangular(qL, b_mat))` (for a matrix RHS)
/// or `_batch_mahalanobis(qL, v)` (for a vector RHS) in
/// `torch/distributions/kl.py:442-464`.
fn solve_lower_tri_sumsq<T: Float>(ql: &[T], b: &[T], n: usize, m: usize) -> T {
    let zero = T::from(0.0).unwrap();
    let x = solve_lower_tri(ql, b, n, m);
    x.iter().fold(zero, |acc, &v| acc + v * v)
}

/// Scalar KL(MVN(loc_p, L_p) || MVN(loc_q, L_q)) for a single distribution.
///
/// Mirrors `_kl_multivariatenormal_multivariatenormal`
/// (`torch/distributions/kl.py:442-464`):
/// ```text
/// half_term1 = Σ ln diag(L_q) - Σ ln diag(L_p)
/// term2 = ‖solve_tri(L_q, L_p)‖_F²            (= trace(M Mᵀ))
/// term3 = ‖solve_tri(L_q, loc_q - loc_p)‖²    (Mahalanobis)
/// KL = half_term1 + 0.5·(term2 + term3 - n)
/// ```
/// `lp`/`lq` are row-major `[n, n]` lower-triangular Cholesky factors;
/// `loc_p`/`loc_q` are length-`n` mean vectors.
fn kl_mvn_dense_scalar<T: Float>(loc_p: &[T], lp: &[T], loc_q: &[T], lq: &[T], n: usize) -> T {
    let zero = T::from(0.0).unwrap();
    let half = T::from(0.5).unwrap();

    let mut half_term1 = zero;
    for i in 0..n {
        half_term1 = half_term1 + lq[i * n + i].ln() - lp[i * n + i].ln();
    }
    // term2 = ‖solve_tri(L_q, L_p)‖_F²: solve L_q · X = L_p (n×n RHS).
    let term2 = solve_lower_tri_sumsq(lq, lp, n, n);
    // term3 = ‖solve_tri(L_q, loc_q - loc_p)‖²: vector RHS.
    let diff: Vec<T> = (0..n).map(|i| loc_q[i] - loc_p[i]).collect();
    let term3 = solve_lower_tri_sumsq(lq, &diff, n, 1);

    let n_t = T::from(n).unwrap();
    half_term1 + half * (term2 + term3 - n_t)
}

/// Shared body for the four (Low-Rank)MVN KL pairs. Pulls each operand's 1-D
/// `loc` and dense lower-triangular `scale_tril` to the host and applies the
/// dense MVN-MVN formula. Errors if the event dimensions differ, matching the
/// `ValueError` PyTorch raises for mismatched `event_shape`
/// (`torch/distributions/kl.py:445-449`).
fn kl_mvn_pair<T: Float>(
    loc_p: &Tensor<T>,
    scale_tril_p: &Tensor<T>,
    n_p: usize,
    loc_q: &Tensor<T>,
    scale_tril_q: &Tensor<T>,
    n_q: usize,
) -> FerrotorchResult<Tensor<T>> {
    crate::fallback::check_gpu_fallback_opt_in(
        &[loc_p, scale_tril_p, loc_q, scale_tril_q],
        "kl_divergence(MultivariateNormal, MultivariateNormal)",
    )?;
    if n_p != n_q {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("kl_divergence(MVN, MVN): event dims must match, got {n_p} and {n_q}"),
        });
    }
    let n = n_p;
    let loc_p_v = loc_p.data_vec()?;
    let lp = scale_tril_p.data_vec()?;
    let loc_q_v = loc_q.data_vec()?;
    let lq = scale_tril_q.data_vec()?;
    let kl = kl_mvn_dense_scalar(&loc_p_v, &lp, &loc_q_v, &lq, n);
    // The KL between two MVNs is a scalar (single distribution, no batch dims).
    Tensor::from_storage(TensorStorage::cpu(vec![kl]), vec![], false)
}

/// KL(MultivariateNormal || MultivariateNormal) (multivariate same-family).
///
/// Mirrors `torch/distributions/kl.py:442-464`
/// `_kl_multivariatenormal_multivariatenormal`.
fn kl_multivariatenormal_multivariatenormal<T: Float>(
    p: &MultivariateNormal<T>,
    q: &MultivariateNormal<T>,
) -> FerrotorchResult<Tensor<T>> {
    kl_mvn_pair(
        p.loc(),
        p.scale_tril(),
        p.dim(),
        q.loc(),
        q.scale_tril(),
        q.dim(),
    )
}

/// KL(MultivariateNormal || LowRankMultivariateNormal) (multivariate
/// cross-family).
///
/// Mirrors `torch/distributions/kl.py:375-403`
/// `_kl_multivariatenormal_lowrankmultivariatenormal`. PyTorch uses the
/// Woodbury identity for the q-side precision; the dense `scale_tril` route is
/// value-identical (R-DEV-1).
fn kl_multivariatenormal_lowrank<T: Float>(
    p: &MultivariateNormal<T>,
    q: &LowRankMultivariateNormal<T>,
) -> FerrotorchResult<Tensor<T>> {
    kl_mvn_pair(
        p.loc(),
        p.scale_tril(),
        p.dim(),
        q.loc(),
        q.scale_tril(),
        q.dim(),
    )
}

/// KL(LowRankMultivariateNormal || MultivariateNormal) (multivariate
/// cross-family).
///
/// Mirrors `torch/distributions/kl.py:405-440`
/// `_kl_lowrankmultivariatenormal_multivariatenormal`.
fn kl_lowrank_multivariatenormal<T: Float>(
    p: &LowRankMultivariateNormal<T>,
    q: &MultivariateNormal<T>,
) -> FerrotorchResult<Tensor<T>> {
    kl_mvn_pair(
        p.loc(),
        p.scale_tril(),
        p.dim(),
        q.loc(),
        q.scale_tril(),
        q.dim(),
    )
}

/// KL(LowRankMultivariateNormal || LowRankMultivariateNormal) (multivariate
/// same-family).
///
/// Mirrors `torch/distributions/kl.py:341-373`
/// `_kl_lowrankmultivariatenormal_lowrankmultivariatenormal`.
fn kl_lowrank_lowrank<T: Float>(
    p: &LowRankMultivariateNormal<T>,
    q: &LowRankMultivariateNormal<T>,
) -> FerrotorchResult<Tensor<T>> {
    kl_mvn_pair(
        p.loc(),
        p.scale_tril(),
        p.dim(),
        q.loc(),
        q.scale_tril(),
        q.dim(),
    )
}

// ---------------------------------------------------------------------------
// Additional KL formulas (#1374 / #1562, both-types-exist gaps): the
// finite OneHotCategorical-OneHotCategorical, Bernoulli-Poisson, Normal-Laplace
// pairs + the support-mismatch `+inf` cross-pair family. Each mirrors a
// `@register_kl` body in `torch/distributions/kl.py`. The `+inf` family
// mirrors PyTorch's `_infinite_like` registrations (the support of `q` does
// not cover the support of `p`, so the KL is `+inf` everywhere).
// ---------------------------------------------------------------------------

/// `+inf`-everywhere KL result shaped like `p`'s parameter tensor.
///
/// Mirrors `torch/distributions/kl.py:141-145` `_infinite_like`, which PyTorch
/// registers for every `(P, Q)` pair where `Q`'s support fails to cover `P`'s
/// (so the divergence is `+inf` at every point). The shape follows `p`'s
/// representative parameter tensor, matching `torch.full_like(tensor, inf)`.
fn kl_infinite_like<T: Float>(param: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let n: usize = param.shape().iter().product::<usize>().max(1);
    Tensor::from_storage(
        TensorStorage::cpu(vec![T::infinity(); n]),
        param.shape().to_vec(),
        false,
    )
}

/// KL(OneHotCategorical(p) || OneHotCategorical(q)).
///
/// Mirrors `torch/distributions/kl.py:474-476` `_kl_onehotcategorical_onehotcategorical`,
/// which delegates to `_kl_categorical_categorical(p._categorical, q._categorical)`.
/// `OneHotCategorical` carries the same probability vector as the equivalent
/// `Categorical`, so the divergence is exactly the Categorical-Categorical
/// closed form `Σ_k p_k·ln(p_k/q_k)` (a scalar). The probs are already
/// normalised by `OneHotCategorical::new`.
fn kl_onehotcategorical_onehotcategorical<T: Float>(
    p: &OneHotCategorical<T>,
    q: &OneHotCategorical<T>,
) -> FerrotorchResult<Tensor<T>> {
    crate::fallback::check_gpu_fallback_opt_in(
        &[p.probs(), q.probs()],
        "kl_divergence(OneHotCategorical, OneHotCategorical)",
    )?;
    let p_probs = p.probs().data_vec()?;
    let q_probs = q.probs().data_vec()?;

    let zero = T::from(0.0).unwrap();
    let eps = T::from(1e-7).unwrap();

    // Probs are normalised at construction; recompute the totals defensively
    // so the formula matches `_kl_categorical_categorical` exactly.
    let p_total: T = p_probs.iter().copied().fold(zero, |a, b| a + b);
    let q_total: T = q_probs.iter().copied().fold(zero, |a, b| a + b);

    let kl: T = p_probs
        .iter()
        .zip(q_probs.iter())
        .fold(zero, |acc, (&pp, &qp)| {
            let pp_norm = pp / p_total;
            let qp_norm = (qp / q_total).max(eps);
            if pp_norm <= eps {
                acc
            } else if qp_norm <= eps {
                T::infinity()
            } else {
                acc + pp_norm * (pp_norm / qp_norm).ln()
            }
        });

    Tensor::from_storage(TensorStorage::cpu(vec![kl]), vec![], false)
}

/// KL(Bernoulli(p) || Poisson(rate)) (cross-family, finite).
///
/// Mirrors `torch/distributions/kl.py:513-516` `_kl_bernoulli_poisson`:
/// ```text
/// KL = -H(Bernoulli) - (p · ln(rate) - rate)
/// ```
/// where `H(Bernoulli(p)) = -p·ln(p) - (1-p)·ln(1-p)`. Probabilities are
/// clamped to `[eps, 1-eps]` to mirror the entropy clamp in `Bernoulli::entropy`.
fn kl_bernoulli_poisson<T: Float>(p: &Bernoulli<T>, q: &Poisson<T>) -> FerrotorchResult<Tensor<T>> {
    crate::fallback::check_gpu_fallback_opt_in(
        &[p.probs(), q.rate()],
        "kl_divergence(Bernoulli, Poisson)",
    )?;
    let p_probs = p.probs().data_vec()?;
    let q_rate = q.rate().data_vec()?;
    let one = T::from(1.0).unwrap();
    let eps = T::from(1e-7).unwrap();

    let result: Vec<T> = p_probs
        .iter()
        .zip(q_rate.iter().cycle())
        .map(|(&pp, &rate)| {
            let pc = pp.max(eps).min(one - eps);
            let entropy = -(pc * pc.ln()) - (one - pc) * (one - pc).ln();
            -entropy - (pp * rate.ln() - rate)
        })
        .collect();

    Tensor::from_storage(
        TensorStorage::cpu(result),
        p.probs().shape().to_vec(),
        false,
    )
}

/// KL(Normal(loc1, scale1) || Laplace(loc2, scale2)) (cross-family, finite).
///
/// Mirrors `torch/distributions/kl.py:782-792` `_kl_normal_laplace`:
/// ```text
/// loc_diff = loc1 - loc2;  scale_ratio = scale1/scale2
/// loc_diff_scale_ratio = loc_diff / scale1
/// t1 = ln(scale_ratio)
/// t2 = sqrt(2/π)·scale1·exp(-0.5·loc_diff_scale_ratio²)
/// t3 = loc_diff·erf(sqrt(0.5)·loc_diff_scale_ratio)
/// KL = -t1 + (t2 + t3)/scale2 - 0.5·(1 + ln(0.5·π))
/// ```
/// `erf` is computed via `ferrotorch_core::special::erf` (the public tensor
/// entry point, ~1 ulp on f64) applied to the per-element arguments.
fn kl_normal_laplace<T: Float>(p: &Normal<T>, q: &Laplace<T>) -> FerrotorchResult<Tensor<T>> {
    crate::fallback::check_gpu_fallback_opt_in(
        &[p.loc(), p.scale(), q.loc(), q.scale()],
        "kl_divergence(Normal, Laplace)",
    )?;
    let p_loc = p.loc().data_vec()?;
    let p_scale = p.scale().data_vec()?;
    let q_loc = q.loc().data_vec()?;
    let q_scale = q.scale().data_vec()?;

    let half = T::from(0.5).unwrap();
    let sqrt_2_over_pi = T::from((2.0 / std::f64::consts::PI).sqrt()).unwrap();
    let sqrt_half = T::from(0.5_f64.sqrt()).unwrap();
    let half_ln_half_pi_term = T::from(0.5 * (1.0 + (0.5 * std::f64::consts::PI).ln())).unwrap();

    // erf argument = sqrt(0.5) * (loc_diff / scale1), one per output element.
    // q broadcasts over p via `cycle()`, matching the other cross-family bodies.
    let n = p_loc.len();
    let erf_args: Vec<T> = p_loc
        .iter()
        .zip(p_scale.iter())
        .zip(q_loc.iter().cycle())
        .map(|((&pl, &ps), &ql)| sqrt_half * ((pl - ql) / ps))
        .collect();
    let erf_tensor = Tensor::from_storage(TensorStorage::cpu(erf_args), vec![n], false)?;
    let erf_vals = ferrotorch_core::special::erf(&erf_tensor)?.data_vec()?;

    let result: Vec<T> = p_loc
        .iter()
        .zip(p_scale.iter())
        .zip(q_loc.iter().cycle())
        .zip(q_scale.iter().cycle())
        .zip(erf_vals.iter())
        .map(|((((&pl, &ps), &ql), &qs), &erf_v)| {
            let loc_diff = pl - ql;
            let scale_ratio = ps / qs;
            let ldsr = loc_diff / ps;
            let t1 = scale_ratio.ln();
            let t2 = sqrt_2_over_pi * ps * (-half * ldsr * ldsr).exp();
            let t3 = loc_diff * erf_v;
            -t1 + (t2 + t3) / qs - half_ln_half_pi_term
        })
        .collect();

    Tensor::from_storage(TensorStorage::cpu(result), p.loc().shape().to_vec(), false)
}

/// KL(Binomial(n_p, p) || Binomial(n_q, q)) (discrete same-family).
///
/// Mirrors `torch/distributions/kl.py:231-244` `_kl_binomial_binomial`:
/// ```text
/// if (p.total_count < q.total_count).any(): raise NotImplementedError
/// kl = p.total_count · (p.probs·(p.logits - q.logits)
///                       + log1p(-p.probs) - log1p(-q.probs))
/// kl[p.total_count > q.total_count] = +inf
/// ```
/// where `logits = ln(p) - ln(1-p)`. ferrotorch returns `InvalidArgument`
/// (matching PyTorch's `NotImplementedError`) when any `n_p < n_q`, and `+inf`
/// element-wise where `n_p > n_q` (the support of the larger-`n` Binomial is
/// not covered by the smaller-`n` one).
fn kl_binomial_binomial<T: Float>(p: &Binomial<T>, q: &Binomial<T>) -> FerrotorchResult<Tensor<T>> {
    crate::fallback::check_gpu_fallback_opt_in(
        &[p.total_count(), p.probs(), q.total_count(), q.probs()],
        "kl_divergence(Binomial, Binomial)",
    )?;
    let p_count = p.total_count().data_vec()?;
    let p_probs = p.probs().data_vec()?;
    let q_count = q.total_count().data_vec()?;
    let q_probs = q.probs().data_vec()?;

    let one = T::from(1.0).unwrap();
    let eps = T::from(1e-7).unwrap();

    // `n_p < n_q` anywhere is unsupported (matches torch NotImplementedError at
    // kl.py:235-238: "KL between Binomials where q.total_count > p.total_count").
    for (i, &pn) in p_count.iter().enumerate() {
        let qn = q_count[i % q_count.len()];
        if pn < qn {
            return Err(FerrotorchError::InvalidArgument {
                message: "kl_divergence(Binomial, Binomial): q.total_count > p.total_count is \
                          not implemented (matches torch NotImplementedError, kl.py:235-238)."
                    .into(),
            });
        }
    }

    let result: Vec<T> = p_count
        .iter()
        .zip(p_probs.iter())
        .zip(q_probs.iter().cycle())
        .zip(q_count.iter().cycle())
        .map(|(((&pn, &pp), &qp), &qn)| {
            if pn > qn {
                return T::infinity();
            }
            let pc = pp.max(eps).min(one - eps);
            let qc = qp.max(eps).min(one - eps);
            // logit = ln(p) - ln(1-p); log1p(-p) = ln(1-p).
            let p_logit = pc.ln() - (one - pc).ln();
            let q_logit = qc.ln() - (one - qc).ln();
            pn * (pc * (p_logit - q_logit) + (one - pc).ln() - (one - qc).ln())
        })
        .collect();

    Tensor::from_storage(
        TensorStorage::cpu(result),
        p.total_count().shape().to_vec(),
        false,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::creation::{scalar, tensor};

    // -- Normal-Normal -------------------------------------------------------

    #[test]
    fn test_kl_normal_normal_same() {
        // KL(N(0,1) || N(0,1)) = 0
        let p = Normal::new(scalar(0.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
        let q = Normal::new(scalar(0.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        assert!(
            kl.item().unwrap().abs() < 1e-6,
            "KL(same, same) should be 0, got {}",
            kl.item().unwrap()
        );
    }

    #[test]
    fn test_kl_normal_normal_different_mean() {
        // KL(N(0,1) || N(1,1)) = 0.5 * (1 + 1 - 1 - 0) = 0.5
        let p = Normal::new(scalar(0.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
        let q = Normal::new(scalar(1.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        assert!(
            (kl.item().unwrap() - 0.5).abs() < 1e-5,
            "expected 0.5, got {}",
            kl.item().unwrap()
        );
    }

    #[test]
    fn test_kl_normal_normal_different_scale() {
        // KL(N(0,1) || N(0,2)) = 0.5 * (0.25 + 0 - 1 - ln(0.25))
        //                       = 0.5 * (0.25 - 1 + ln(4))
        //                       = 0.5 * (-0.75 + 1.3863) = 0.5 * 0.6363 = 0.3181
        let p = Normal::new(scalar(0.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
        let q = Normal::new(scalar(0.0f32).unwrap(), scalar(2.0f32).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        let expected = 0.5 * (0.25 + 0.0 - 1.0 - 0.25f32.ln());
        assert!(
            (kl.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            kl.item().unwrap()
        );
    }

    #[test]
    fn test_kl_normal_normal_nonnegative() {
        // KL divergence is always >= 0
        let p = Normal::new(scalar(2.0f32).unwrap(), scalar(0.5f32).unwrap()).unwrap();
        let q = Normal::new(scalar(-1.0f32).unwrap(), scalar(3.0f32).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        assert!(
            kl.item().unwrap() >= 0.0,
            "KL should be non-negative, got {}",
            kl.item().unwrap()
        );
    }

    #[test]
    fn test_kl_normal_normal_asymmetric() {
        // KL(p||q) != KL(q||p) in general
        let p = Normal::new(scalar(0.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
        let q = Normal::new(scalar(1.0f32).unwrap(), scalar(2.0f32).unwrap()).unwrap();
        let kl_pq = kl_divergence(&p, &q).unwrap().item().unwrap();
        let kl_qp = kl_divergence(&q, &p).unwrap().item().unwrap();
        assert!(
            (kl_pq - kl_qp).abs() > 1e-3,
            "KL should be asymmetric: KL(p||q)={kl_pq}, KL(q||p)={kl_qp}"
        );
    }

    // -- Bernoulli-Bernoulli -------------------------------------------------

    #[test]
    fn test_kl_bernoulli_same() {
        let p = Bernoulli::new(scalar(0.3f32).unwrap()).unwrap();
        let q = Bernoulli::new(scalar(0.3f32).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        assert!(
            kl.item().unwrap().abs() < 1e-5,
            "KL(same, same) = 0, got {}",
            kl.item().unwrap()
        );
    }

    #[test]
    fn test_kl_bernoulli_different() {
        // KL(Bern(0.4) || Bern(0.6)) = 0.4*ln(0.4/0.6) + 0.6*ln(0.6/0.4)
        let p = Bernoulli::new(scalar(0.4f32).unwrap()).unwrap();
        let q = Bernoulli::new(scalar(0.6f32).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        let expected = 0.4f32 * (0.4f32 / 0.6).ln() + 0.6 * (0.6f32 / 0.4).ln();
        assert!(
            (kl.item().unwrap() - expected).abs() < 1e-4,
            "expected {expected}, got {}",
            kl.item().unwrap()
        );
    }

    #[test]
    fn test_kl_bernoulli_nonnegative() {
        let p = Bernoulli::new(scalar(0.1f32).unwrap()).unwrap();
        let q = Bernoulli::new(scalar(0.9f32).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        assert!(kl.item().unwrap() >= 0.0);
    }

    // -- Uniform-Uniform -----------------------------------------------------

    #[test]
    fn test_kl_uniform_same() {
        let p = Uniform::new(scalar(0.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
        let q = Uniform::new(scalar(0.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        assert!(
            kl.item().unwrap().abs() < 1e-6,
            "KL(same, same) = 0, got {}",
            kl.item().unwrap()
        );
    }

    #[test]
    fn test_kl_uniform_contained() {
        // KL(U(0,1) || U(-1,2)) = ln(3/1) = ln(3)
        let p = Uniform::new(scalar(0.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
        let q = Uniform::new(scalar(-1.0f32).unwrap(), scalar(2.0f32).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        let expected = 3.0f32.ln();
        assert!(
            (kl.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            kl.item().unwrap()
        );
    }

    #[test]
    fn test_kl_uniform_not_contained() {
        // If q doesn't cover p, KL = infinity
        let p = Uniform::new(scalar(0.0f32).unwrap(), scalar(3.0f32).unwrap()).unwrap();
        let q = Uniform::new(scalar(1.0f32).unwrap(), scalar(2.0f32).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        assert!(
            kl.item().unwrap().is_infinite(),
            "expected infinity, got {}",
            kl.item().unwrap()
        );
    }

    // -- Categorical-Categorical ---------------------------------------------

    #[test]
    fn test_kl_categorical_same() {
        let p = Categorical::new(tensor(&[0.2f32, 0.3, 0.5]).unwrap()).unwrap();
        let q = Categorical::new(tensor(&[0.2f32, 0.3, 0.5]).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        assert!(
            kl.item().unwrap().abs() < 1e-5,
            "KL(same, same) = 0, got {}",
            kl.item().unwrap()
        );
    }

    #[test]
    fn test_kl_categorical_different() {
        let p = Categorical::new(tensor(&[0.5f32, 0.5]).unwrap()).unwrap();
        let q = Categorical::new(tensor(&[0.25f32, 0.75]).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        // KL = 0.5*ln(0.5/0.25) + 0.5*ln(0.5/0.75) = 0.5*ln(2) + 0.5*ln(2/3)
        let expected = 0.5f32 * 2.0f32.ln() + 0.5 * (2.0f32 / 3.0).ln();
        assert!(
            (kl.item().unwrap() - expected).abs() < 1e-4,
            "expected {expected}, got {}",
            kl.item().unwrap()
        );
    }

    #[test]
    fn test_kl_categorical_nonnegative() {
        let p = Categorical::new(tensor(&[0.1f32, 0.2, 0.7]).unwrap()).unwrap();
        let q = Categorical::new(tensor(&[0.3f32, 0.3, 0.4]).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        assert!(kl.item().unwrap() >= -1e-6);
    }

    // -- Normal-Uniform (cross-family) ---------------------------------------

    #[test]
    fn test_kl_normal_uniform() {
        let p = Normal::new(scalar(0.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
        let q = Uniform::new(scalar(-10.0f32).unwrap(), scalar(10.0f32).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        // KL(Normal || Uniform) is +inf: Normal's support R is not contained in
        // Uniform's [low,high] (kl.py:766,768 -> _kl_normal_infinity ->
        // _infinite_like(p.loc)).
        let v = kl.item().unwrap();
        assert!(v.is_infinite() && v > 0.0);
    }

    // -- Uniform-Normal (cross-family) ---------------------------------------

    #[test]
    fn test_kl_uniform_normal() {
        let p = Uniform::new(scalar(-1.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
        let q = Normal::new(scalar(0.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        assert!(kl.item().unwrap().is_finite());
        assert!(kl.item().unwrap() >= -1e-6);
    }

    // -- f64 -----------------------------------------------------------------

    #[test]
    fn test_kl_normal_normal_f64() {
        let p = Normal::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
        let q = Normal::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        assert!(kl.item().unwrap().abs() < 1e-12);
    }

    #[test]
    fn test_kl_bernoulli_f64() {
        let p = Bernoulli::new(scalar(0.3f64).unwrap()).unwrap();
        let q = Bernoulli::new(scalar(0.7f64).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        let expected = 0.3f64 * (0.3 / 0.7f64).ln() + 0.7 * (0.7 / 0.3f64).ln();
        assert!((kl.item().unwrap() - expected).abs() < 1e-8);
    }

    #[test]
    fn test_kl_uniform_f64() {
        let p = Uniform::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
        let q = Uniform::new(scalar(0.0f64).unwrap(), scalar(2.0f64).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        assert!((kl.item().unwrap() - 2.0f64.ln()).abs() < 1e-10);
    }

    // -- Error case ----------------------------------------------------------

    #[test]
    fn test_kl_unsupported_pair() {
        // Normal-Bernoulli should fail (not registered)
        let p = Normal::new(scalar(0.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
        let q = Bernoulli::new(scalar(0.5f32).unwrap()).unwrap();
        assert!(kl_divergence(&p, &q).is_err());
    }

    // -----------------------------------------------------------------------
    // CL-365: new same-family and cross-family pairs
    // -----------------------------------------------------------------------

    // -- Laplace-Laplace -----------------------------------------------------

    #[test]
    fn test_kl_laplace_laplace_same_is_zero() {
        let p = Laplace::new(scalar(0.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
        let q = Laplace::new(scalar(0.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        assert!(
            kl.item().unwrap().abs() < 1e-5,
            "got {}",
            kl.item().unwrap()
        );
    }

    #[test]
    fn test_kl_laplace_laplace_different_scale() {
        // KL(Lap(0,1) || Lap(0,2)) = log(2/1) + (1*exp(0) + 0)/2 - 1
        //                          = ln(2) + 0.5 - 1 ≈ 0.6931 - 0.5 = 0.1931
        let p = Laplace::new(scalar(0.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
        let q = Laplace::new(scalar(0.0f32).unwrap(), scalar(2.0f32).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        let v = kl.item().unwrap();
        let expected = 2.0_f32.ln() + 0.5 - 1.0;
        assert!((v - expected).abs() < 1e-5, "expected {expected}, got {v}");
    }

    #[test]
    fn test_kl_laplace_laplace_different_loc() {
        // KL(Lap(0,1) || Lap(1,1)) = log(1) + (exp(-1) + 1)/1 - 1
        //                          = 0 + e^-1 + 1 - 1 = 1/e ≈ 0.3679
        let p = Laplace::new(scalar(0.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
        let q = Laplace::new(scalar(1.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        let expected = 1.0_f32 / std::f32::consts::E;
        assert!(
            (kl.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            kl.item().unwrap()
        );
    }

    // -- Exponential-Exponential ---------------------------------------------

    #[test]
    fn test_kl_exponential_exponential_same() {
        let p = Exponential::new(scalar(2.0f32).unwrap()).unwrap();
        let q = Exponential::new(scalar(2.0f32).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        assert!(kl.item().unwrap().abs() < 1e-5);
    }

    #[test]
    fn test_kl_exponential_exponential_different() {
        // KL(Exp(2) || Exp(1)) = log(2/1) + 1/2 - 1 = ln(2) - 0.5 ≈ 0.1931
        let p = Exponential::new(scalar(2.0f32).unwrap()).unwrap();
        let q = Exponential::new(scalar(1.0f32).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        let expected = 2.0_f32.ln() - 0.5;
        assert!((kl.item().unwrap() - expected).abs() < 1e-5);
    }

    // -- Gamma-Gamma ---------------------------------------------------------

    #[test]
    fn test_kl_gamma_gamma_same_is_zero() {
        // When both distributions are identical, KL should be 0. This
        // exercises the full Gamma-Gamma formula including digamma
        // and lgamma terms.
        let p = Gamma::new(scalar(2.0f32).unwrap(), scalar(3.0f32).unwrap()).unwrap();
        let q = Gamma::new(scalar(2.0f32).unwrap(), scalar(3.0f32).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        // Lanczos lgamma is accurate to ~1e-12 f64; f32 round-trip dominates
        // the error budget here.
        assert!(
            kl.item().unwrap().abs() < 1e-6,
            "KL(Gamma same) should be near 0, got {}",
            kl.item().unwrap()
        );
    }

    #[test]
    fn test_kl_gamma_gamma_exp_special_case() {
        // Gamma(1, λ) == Exp(λ). Verify that KL(Gamma(1,2) || Gamma(1,1))
        // matches KL(Exp(2) || Exp(1)) = ln(2) - 0.5.
        let p = Gamma::new(scalar(1.0f32).unwrap(), scalar(2.0f32).unwrap()).unwrap();
        let q = Gamma::new(scalar(1.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        let expected = 2.0_f32.ln() - 0.5;
        assert!(
            (kl.item().unwrap() - expected).abs() < 1e-6,
            "expected {expected}, got {}",
            kl.item().unwrap()
        );
    }

    // -- Poisson-Poisson -----------------------------------------------------

    #[test]
    fn test_kl_poisson_poisson_same() {
        let p = Poisson::new(scalar(3.0f32).unwrap()).unwrap();
        let q = Poisson::new(scalar(3.0f32).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        assert!(kl.item().unwrap().abs() < 1e-5);
    }

    #[test]
    fn test_kl_poisson_poisson_known_value() {
        // KL(Poisson(2) || Poisson(1)) = 2*(ln 2 - ln 1) - 2 + 1
        //                              = 2*ln 2 - 1 ≈ 0.3863
        let p = Poisson::new(scalar(2.0f32).unwrap()).unwrap();
        let q = Poisson::new(scalar(1.0f32).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        let expected = 2.0 * 2.0_f32.ln() - 1.0;
        assert!((kl.item().unwrap() - expected).abs() < 1e-5);
    }

    // -- Cross-family: Gamma-Exponential and Exponential-Gamma ---------------

    #[test]
    fn test_kl_gamma_exponential_matches_gamma_gamma() {
        // KL(Gamma(2, 3) || Exp(1)) should equal KL(Gamma(2,3) || Gamma(1,1))
        let p = Gamma::new(scalar(2.0f32).unwrap(), scalar(3.0f32).unwrap()).unwrap();
        let q_exp = Exponential::new(scalar(1.0f32).unwrap()).unwrap();
        let q_gamma = Gamma::new(scalar(1.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
        let kl_ge = kl_divergence(&p, &q_exp).unwrap();
        let p2 = Gamma::new(scalar(2.0f32).unwrap(), scalar(3.0f32).unwrap()).unwrap();
        let kl_gg = kl_divergence(&p2, &q_gamma).unwrap();
        assert!(
            (kl_ge.item().unwrap() - kl_gg.item().unwrap()).abs() < 1e-4,
            "Gamma-Exp and Gamma-Gamma(1,λ) should agree"
        );
    }

    #[test]
    fn test_kl_exponential_gamma_matches_gamma_gamma() {
        // KL(Exp(2) || Gamma(1, 1)) == KL(Gamma(1, 2) || Gamma(1, 1))
        //   == Exp-Exp(2, 1) == ln(2) - 0.5
        let p = Exponential::new(scalar(2.0f32).unwrap()).unwrap();
        let q = Gamma::new(scalar(1.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        let expected = 2.0_f32.ln() - 0.5;
        assert!(
            (kl.item().unwrap() - expected).abs() < 1e-6,
            "expected {expected}, got {}",
            kl.item().unwrap()
        );
    }

    #[test]
    fn test_kl_exponential_gamma_self_consistency() {
        // Gamma(1, 1) == Exp(1), so KL(Exp(1) || Gamma(1,1)) == 0.
        let p = Exponential::new(scalar(1.0f32).unwrap()).unwrap();
        let q = Gamma::new(scalar(1.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        assert!(
            kl.item().unwrap().abs() < 1e-6,
            "KL(Exp(1)||Gamma(1,1)) should be 0, got {}",
            kl.item().unwrap()
        );
    }

    // -----------------------------------------------------------------------
    // #1374: new same-family pairs (Beta, Gumbel, Pareto, HalfNormal) +
    // cross-family (Exponential-Normal, Gamma-Normal, Laplace-Normal).
    // -----------------------------------------------------------------------

    use crate::{Beta, Gumbel, HalfNormal, Pareto};

    // -- Beta-Beta -----------------------------------------------------------

    #[test]
    fn test_kl_beta_beta_same_is_zero() {
        let p = Beta::new(scalar(2.0f64).unwrap(), scalar(3.0f64).unwrap()).unwrap();
        let q = Beta::new(scalar(2.0f64).unwrap(), scalar(3.0f64).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        assert!(
            kl.item().unwrap().abs() < 1e-10,
            "got {}",
            kl.item().unwrap()
        );
    }

    #[test]
    fn test_kl_beta_beta_known_value() {
        // KL(Beta(2,3) || Beta(3,2)). Computed from the closed form mirrored
        // from torch _kl_beta_beta (kl.py:219-228):
        //   t1 = lnΓ(3)+lnΓ(2)+lnΓ(5); t2 = lnΓ(2)+lnΓ(3)+lnΓ(5)
        //   t3 = (2-3)ψ(2); t4 = (3-2)ψ(3); t5 = (5-5)ψ(5)
        //   => t1-t2 = 0; KL = -ψ(2) + ψ(3) = (ψ(3)-ψ(2)) = 1/2 = 0.5
        // (ψ(3)-ψ(2) = 1/2 by the digamma recurrence ψ(x+1)=ψ(x)+1/x).
        let p = Beta::new(scalar(2.0f64).unwrap(), scalar(3.0f64).unwrap()).unwrap();
        let q = Beta::new(scalar(3.0f64).unwrap(), scalar(2.0f64).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        assert!(
            (kl.item().unwrap() - 0.5).abs() < 1e-9,
            "expected 0.5, got {}",
            kl.item().unwrap()
        );
    }

    #[test]
    fn test_kl_beta_beta_nonnegative() {
        let p = Beta::new(scalar(0.5f64).unwrap(), scalar(0.5f64).unwrap()).unwrap();
        let q = Beta::new(scalar(2.0f64).unwrap(), scalar(5.0f64).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        assert!(kl.item().unwrap() >= -1e-9, "got {}", kl.item().unwrap());
    }

    // -- Gumbel-Gumbel -------------------------------------------------------

    #[test]
    fn test_kl_gumbel_gumbel_same_is_zero() {
        let p = Gumbel::new(scalar(1.0f64).unwrap(), scalar(2.0f64).unwrap()).unwrap();
        let q = Gumbel::new(scalar(1.0f64).unwrap(), scalar(2.0f64).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        // Same distribution → KL = 0. The Gumbel formula reduces exactly to 0
        // since exp(lnΓ(2)) = 1 and the linear/γ terms cancel.
        assert!(
            kl.item().unwrap().abs() < 1e-9,
            "got {}",
            kl.item().unwrap()
        );
    }

    #[test]
    fn test_kl_gumbel_gumbel_nonnegative() {
        let p = Gumbel::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
        let q = Gumbel::new(scalar(2.0f64).unwrap(), scalar(3.0f64).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        assert!(kl.item().unwrap() >= -1e-9, "got {}", kl.item().unwrap());
    }

    // -- Pareto-Pareto -------------------------------------------------------

    #[test]
    fn test_kl_pareto_pareto_same_is_zero() {
        let p = Pareto::new(scalar(1.0f64).unwrap(), scalar(3.0f64).unwrap()).unwrap();
        let q = Pareto::new(scalar(1.0f64).unwrap(), scalar(3.0f64).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        assert!(
            kl.item().unwrap().abs() < 1e-10,
            "got {}",
            kl.item().unwrap()
        );
    }

    #[test]
    fn test_kl_pareto_pareto_known_value() {
        // KL(Pareto(scale=1, α=4) || Pareto(scale=1, α=2)). scale_ratio=1 so
        // t1 = 0; alpha_ratio = 2/4 = 0.5; KL = -ln(0.5) + 0.5 - 1
        //   = ln(2) - 0.5 ≈ 0.193147.
        let p = Pareto::new(scalar(1.0f64).unwrap(), scalar(4.0f64).unwrap()).unwrap();
        let q = Pareto::new(scalar(1.0f64).unwrap(), scalar(2.0f64).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        let expected = 2.0f64.ln() - 0.5;
        assert!(
            (kl.item().unwrap() - expected).abs() < 1e-10,
            "expected {expected}, got {}",
            kl.item().unwrap()
        );
    }

    #[test]
    fn test_kl_pareto_pareto_support_violation_is_inf() {
        // p.scale < q.scale → p support extends below q → +inf (kl.py:487).
        let p = Pareto::new(scalar(1.0f64).unwrap(), scalar(3.0f64).unwrap()).unwrap();
        let q = Pareto::new(scalar(2.0f64).unwrap(), scalar(3.0f64).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        assert!(kl.item().unwrap().is_infinite());
    }

    // -- HalfNormal-HalfNormal -----------------------------------------------

    #[test]
    fn test_kl_halfnormal_halfnormal_same_is_zero() {
        let p = HalfNormal::new(scalar(1.5f64).unwrap()).unwrap();
        let q = HalfNormal::new(scalar(1.5f64).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        assert!(
            kl.item().unwrap().abs() < 1e-12,
            "got {}",
            kl.item().unwrap()
        );
    }

    #[test]
    fn test_kl_halfnormal_matches_normal_normal() {
        // _kl_halfnormal_halfnormal delegates to _kl_normal_normal(Normal(0,s1),
        // Normal(0,s2)) (kl.py:325-327). Cross-check against the Normal-Normal
        // formula with loc=0.
        let s1 = 1.0f64;
        let s2 = 2.0f64;
        let p = HalfNormal::new(scalar(s1).unwrap()).unwrap();
        let q = HalfNormal::new(scalar(s2).unwrap()).unwrap();
        let kl_hn = kl_divergence(&p, &q).unwrap().item().unwrap();

        let pn = Normal::new(scalar(0.0f64).unwrap(), scalar(s1).unwrap()).unwrap();
        let qn = Normal::new(scalar(0.0f64).unwrap(), scalar(s2).unwrap()).unwrap();
        let kl_nn = kl_divergence(&pn, &qn).unwrap().item().unwrap();
        assert!(
            (kl_hn - kl_nn).abs() < 1e-12,
            "HalfNormal KL {kl_hn} must equal Normal-Normal(loc=0) KL {kl_nn}"
        );
    }

    // -- Exponential-Normal (cross-family) -----------------------------------

    #[test]
    fn test_kl_exponential_normal_known_value() {
        // KL(Exp(1) || Normal(0, 1)). var=1, rate_sqr=1.
        //   t1 = 0.5·ln(1·1·2π) = 0.5·ln(2π)
        //   t2 = 1; t3 = 0; t4 = 0
        //   KL = 0.5·ln(2π) - 1 + 1 = 0.5·ln(2π) ≈ 0.918939
        let p = Exponential::new(scalar(1.0f64).unwrap()).unwrap();
        let q = Normal::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        let expected = 0.5 * (2.0 * std::f64::consts::PI).ln();
        assert!(
            (kl.item().unwrap() - expected).abs() < 1e-12,
            "expected {expected}, got {}",
            kl.item().unwrap()
        );
    }

    // -- Gamma-Normal (cross-family) -----------------------------------------

    #[test]
    fn test_kl_gamma_normal_finite_nonnegative() {
        let p = Gamma::new(scalar(2.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
        let q = Normal::new(scalar(2.0f64).unwrap(), scalar(2.0f64).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        let v = kl.item().unwrap();
        assert!(v.is_finite(), "got {v}");
        // Gamma(2,1) has mean 2, matching the Normal mean; KL stays finite.
    }

    #[test]
    fn test_kl_gamma_normal_exp_special_case() {
        // Gamma(1, λ) == Exp(λ); KL(Gamma(1,1) || Normal(0,1)) must equal
        // KL(Exp(1) || Normal(0,1)) = 0.5·ln(2π).
        let pg = Gamma::new(scalar(1.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
        let q = Normal::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
        let kl_g = kl_divergence(&pg, &q).unwrap().item().unwrap();
        let expected = 0.5 * (2.0 * std::f64::consts::PI).ln();
        assert!(
            (kl_g - expected).abs() < 1e-9,
            "Gamma(1,1)-Normal should match Exp(1)-Normal {expected}, got {kl_g}"
        );
    }

    // -- Laplace-Normal (cross-family) ---------------------------------------

    #[test]
    fn test_kl_laplace_normal_known_value() {
        // KL(Laplace(0, 1) || Normal(0, 1)). var=1, ratio = 1/1 = 1.
        //   t1 = 0.5·ln(2·1/π) = 0.5·ln(2/π)
        //   t2=t3=t4=0 (loc=loc2=0)
        //   KL = -0.5·ln(2/π) + 1 + 0 - 1 = -0.5·ln(2/π) = 0.5·ln(π/2)
        let p = Laplace::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
        let q = Normal::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        let expected = 0.5 * (std::f64::consts::PI / 2.0).ln();
        assert!(
            (kl.item().unwrap() - expected).abs() < 1e-12,
            "expected {expected}, got {}",
            kl.item().unwrap()
        );
    }

    // -----------------------------------------------------------------------
    // #1374 wave-L: Cauchy-Cauchy + Gumbel cross-family pairs.
    // -----------------------------------------------------------------------

    use crate::Cauchy;

    // -- Cauchy-Cauchy -------------------------------------------------------

    #[test]
    fn test_kl_cauchy_cauchy_same_is_zero() {
        // KL(C(loc,s) || C(loc,s)) = ln((2s)²) - ln(4s²) = ln(4s²) - ln(4s²) = 0.
        let p = Cauchy::new(scalar(1.0f64).unwrap(), scalar(2.0f64).unwrap()).unwrap();
        let q = Cauchy::new(scalar(1.0f64).unwrap(), scalar(2.0f64).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        assert!(
            kl.item().unwrap().abs() < 1e-12,
            "got {}",
            kl.item().unwrap()
        );
    }

    #[test]
    fn test_kl_cauchy_cauchy_known_value() {
        // KL(C(0,1) || C(0,2)) = ln((1+2)² + 0²) - ln(4·1·2)
        //                       = ln(9) - ln(8) = ln(9/8) (kl.py:952-957).
        let p = Cauchy::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
        let q = Cauchy::new(scalar(0.0f64).unwrap(), scalar(2.0f64).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        let expected = (9.0f64 / 8.0).ln();
        assert!(
            (kl.item().unwrap() - expected).abs() < 1e-12,
            "expected {expected}, got {}",
            kl.item().unwrap()
        );
    }

    // -- Normal-Gumbel (cross-family) ----------------------------------------

    #[test]
    fn test_kl_normal_gumbel_known_value() {
        // KL(Normal(0,1) || Gumbel(0,1)) (kl.py:771-779). With loc=0, scale=1,
        // loc2=0, scale2=1: mean_scale_ratio=0, var_scale_sqr_ratio=1,
        // loc_scale_ratio=0.
        //   t1 = 0.5·ln(1) = 0
        //   t2 = 0 - 0 = 0
        //   t3 = exp(-0 + 0.5·1 + 0) = exp(0.5) = sqrt(e)
        //   KL = -0 + 0 + sqrt(e) - 0.5·(1 + ln(2π))
        let p = Normal::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
        let q = Gumbel::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        let expected = 0.5f64.exp() - 0.5 * (1.0 + (2.0 * std::f64::consts::PI).ln());
        assert!(
            (kl.item().unwrap() - expected).abs() < 1e-12,
            "expected {expected}, got {}",
            kl.item().unwrap()
        );
    }

    // -- Gumbel-Normal (cross-family) ----------------------------------------

    #[test]
    fn test_kl_gumbel_normal_known_value() {
        // KL(Gumbel(0,1) || Normal(0,1)) (kl.py:731-737). param_ratio=1.
        //   t1 = ln(1/sqrt(2π))
        //   t2 = (π·1·0.5)²/3 = (π/2)²/3
        //   t3 = 0.5·((0 + 1·γ - 0)/1)² = 0.5·γ²
        //   KL = -t1 + t2 + t3 - (γ + 1)
        let p = Gumbel::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
        let q = Normal::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        let g = EULER_GAMMA;
        let t1 = (1.0 / (2.0 * std::f64::consts::PI).sqrt()).ln();
        let t2 = (std::f64::consts::PI * 0.5).powi(2) / 3.0;
        let t3 = 0.5 * g * g;
        let expected = -t1 + t2 + t3 - (g + 1.0);
        assert!(
            (kl.item().unwrap() - expected).abs() < 1e-12,
            "expected {expected}, got {}",
            kl.item().unwrap()
        );
        // KL must be non-negative.
        assert!(kl.item().unwrap() >= -1e-12);
    }

    // -- Gamma-Gumbel (cross-family) -----------------------------------------

    #[test]
    fn test_kl_gamma_gumbel_reduces_to_exponential_gumbel() {
        // Gamma(1, λ) == Exp(λ), so KL(Gamma(1,β) || Gumbel(loc,scale)) must
        // equal KL(Exp(β) || Gumbel(loc,scale)) (kl.py:678-693 reduces to
        // kl.py:641-649 at α=1: (α-1)ψ(α)=0, lnΓ(1)=0, -α=-1, and the t2/t3
        // terms coincide).
        let pg = Gamma::new(scalar(1.0f64).unwrap(), scalar(2.0f64).unwrap()).unwrap();
        let pe = Exponential::new(scalar(2.0f64).unwrap()).unwrap();
        let q = Gumbel::new(scalar(0.5f64).unwrap(), scalar(1.5f64).unwrap()).unwrap();
        let kl_g = kl_divergence(&pg, &q).unwrap().item().unwrap();
        let q2 = Gumbel::new(scalar(0.5f64).unwrap(), scalar(1.5f64).unwrap()).unwrap();
        let kl_e = kl_divergence(&pe, &q2).unwrap().item().unwrap();
        assert!(
            (kl_g - kl_e).abs() < 1e-12,
            "Gamma(1,β)-Gumbel {kl_g} must equal Exp(β)-Gumbel {kl_e}"
        );
    }

    // -- Exponential-Gumbel (cross-family) -----------------------------------

    #[test]
    fn test_kl_exponential_gumbel_known_value() {
        // KL(Exp(1) || Gumbel(0,1)) (kl.py:641-649). scale_rate_prod=1,
        // loc_scale_ratio=0.
        //   t1 = ln(1) - 1 = -1
        //   t2 = exp(0)·1/(1+1) = 0.5
        //   t3 = 1/1 = 1
        //   KL = -1 - 0 + 0.5 + 1 = 0.5
        let p = Exponential::new(scalar(1.0f64).unwrap()).unwrap();
        let q = Gumbel::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        assert!(
            (kl.item().unwrap() - 0.5).abs() < 1e-12,
            "expected 0.5, got {}",
            kl.item().unwrap()
        );
    }

    // -- Uniform-Gumbel (cross-family) ---------------------------------------

    #[test]
    fn test_kl_uniform_gumbel_known_value() {
        // KL(Uniform(0,1) || Gumbel(0,1)) (kl.py:912-919). common_term=1/1=1,
        // high_loc_diff=(1-0)/1=1, low_loc_diff=(0-0)/1=0.
        //   t1 = ln(1) + 0.5·(1+0) = 0.5
        //   t2 = 1·(exp(-1) - exp(0)) = e^-1 - 1
        //   KL = t1 - t2 = 0.5 - (e^-1 - 1) = 1.5 - e^-1
        let p = Uniform::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
        let q = Gumbel::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        let expected = 1.5 - (-1.0f64).exp();
        assert!(
            (kl.item().unwrap() - expected).abs() < 1e-12,
            "expected {expected}, got {}",
            kl.item().unwrap()
        );
    }

    // -----------------------------------------------------------------------
    // #1374 wave-M: Dirichlet-Dirichlet + Beta/Pareto/Uniform cross-family.
    // Reference values from live `torch.distributions.kl_divergence` (f64,
    // this machine 2026-05-26); each `expected` traces to a `@register_kl`
    // body in `torch/distributions/kl.py` (R-CHAR-3 non-tautological).
    // -----------------------------------------------------------------------

    use crate::Dirichlet;

    fn approx(got: f64, expected: f64, tol: f64, what: &str) {
        assert!(
            (got - expected).abs() < tol,
            "{what}: got {got}, torch {expected}, |err|={}",
            (got - expected).abs()
        );
    }

    // -- Dirichlet-Dirichlet -------------------------------------------------

    #[test]
    fn test_kl_dirichlet_dirichlet_same_is_zero() {
        let p = Dirichlet::new(tensor(&[1.0f64, 2.0, 3.0]).unwrap()).unwrap();
        let q = Dirichlet::new(tensor(&[1.0f64, 2.0, 3.0]).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        approx(kl.item().unwrap(), 0.0, 1e-12, "Dir-Dir same");
    }

    #[test]
    fn test_kl_dirichlet_dirichlet_known_value() {
        // torch.distributions.kl_divergence(Dirichlet([1,2,3]), Dirichlet([2,2,2]))
        let p = Dirichlet::new(tensor(&[1.0f64, 2.0, 3.0]).unwrap()).unwrap();
        let q = Dirichlet::new(tensor(&[2.0f64, 2.0, 2.0]).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        approx(
            kl.item().unwrap(),
            0.806_852_819_440_054_7,
            1e-10,
            "Dir-Dir",
        );
    }

    // -- Beta-Exponential ----------------------------------------------------

    #[test]
    fn test_kl_beta_exponential_known_value() {
        // kl.py:533-539. torch: KL(Beta(2,3) || Exp(1.5)).
        let p = Beta::new(scalar(2.0f64).unwrap(), scalar(3.0f64).unwrap()).unwrap();
        let q = Exponential::new(scalar(1.5f64).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        approx(
            kl.item().unwrap(),
            0.429_441_541_679_836_06,
            1e-10,
            "Beta-Exp",
        );
    }

    // -- Beta-Gamma ----------------------------------------------------------

    #[test]
    fn test_kl_beta_gamma_known_value() {
        // kl.py:542-552. torch: KL(Beta(2,3) || Gamma(2.0, 1.5)).
        let p = Beta::new(scalar(2.0f64).unwrap(), scalar(3.0f64).unwrap()).unwrap();
        let q = Gamma::new(scalar(2.0f64).unwrap(), scalar(1.5f64).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        approx(
            kl.item().unwrap(),
            1.107_309_766_905_004_7,
            1e-10,
            "Beta-Gamma",
        );
    }

    // -- Beta-Normal ---------------------------------------------------------

    #[test]
    fn test_kl_beta_normal_known_value() {
        // kl.py:556-568. torch: KL(Beta(2,3) || Normal(0.5, 1.0)).
        let p = Beta::new(scalar(2.0f64).unwrap(), scalar(3.0f64).unwrap()).unwrap();
        let q = Normal::new(scalar(0.5f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        approx(
            kl.item().unwrap(),
            1.178_845_182_992_673,
            1e-10,
            "Beta-Normal",
        );
    }

    // -- Beta-Uniform --------------------------------------------------------

    #[test]
    fn test_kl_beta_uniform_contained_known_value() {
        // kl.py:571-577. torch: KL(Beta(2,3) || Uniform(-1, 2)) (U covers [0,1]).
        let p = Beta::new(scalar(2.0f64).unwrap(), scalar(3.0f64).unwrap()).unwrap();
        let q = Uniform::new(scalar(-1.0f64).unwrap(), scalar(2.0f64).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        approx(
            kl.item().unwrap(),
            1.333_518_938_456_110_1,
            1e-10,
            "Beta-Unif",
        );
    }

    #[test]
    fn test_kl_beta_uniform_support_violation_is_inf() {
        // U(0.2, 0.8) does NOT cover Beta support [0,1] → +inf.
        let p = Beta::new(scalar(2.0f64).unwrap(), scalar(3.0f64).unwrap()).unwrap();
        let q = Uniform::new(scalar(0.2f64).unwrap(), scalar(0.8f64).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        assert!(kl.item().unwrap().is_infinite());
    }

    // -- Pareto-Exponential --------------------------------------------------

    #[test]
    fn test_kl_pareto_exponential_known_value() {
        // kl.py:802-810. torch: KL(Pareto(1, 3) || Exp(0.5)).
        let p = Pareto::new(scalar(1.0f64).unwrap(), scalar(3.0f64).unwrap()).unwrap();
        let q = Exponential::new(scalar(0.5f64).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        approx(
            kl.item().unwrap(),
            1.208_426_135_894_721_5,
            1e-10,
            "Pareto-Exp",
        );
    }

    #[test]
    fn test_kl_pareto_exponential_alpha_le_1_is_inf() {
        let p = Pareto::new(scalar(1.0f64).unwrap(), scalar(0.5f64).unwrap()).unwrap();
        let q = Exponential::new(scalar(0.5f64).unwrap()).unwrap();
        assert!(kl_divergence(&p, &q).unwrap().item().unwrap().is_infinite());
    }

    // -- Pareto-Gamma --------------------------------------------------------

    #[test]
    fn test_kl_pareto_gamma_known_value() {
        // kl.py:813-825. torch: KL(Pareto(1, 3) || Gamma(2.0, 0.5)).
        let p = Pareto::new(scalar(1.0f64).unwrap(), scalar(3.0f64).unwrap()).unwrap();
        let q = Gamma::new(scalar(2.0f64).unwrap(), scalar(0.5f64).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        approx(
            kl.item().unwrap(),
            1.568_239_983_121_333_8,
            1e-10,
            "Pareto-Gamma",
        );
    }

    // -- Pareto-Normal -------------------------------------------------------

    #[test]
    fn test_kl_pareto_normal_known_value() {
        // kl.py:828-838. torch: KL(Pareto(1, 3) || Normal(2.0, 1.0)).
        let p = Pareto::new(scalar(1.0f64).unwrap(), scalar(3.0f64).unwrap()).unwrap();
        let q = Normal::new(scalar(2.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        approx(
            kl.item().unwrap(),
            1.184_217_488_539_449_2,
            1e-10,
            "Pareto-Normal",
        );
    }

    #[test]
    fn test_kl_pareto_normal_alpha_le_2_is_inf() {
        let p = Pareto::new(scalar(1.0f64).unwrap(), scalar(1.5f64).unwrap()).unwrap();
        let q = Normal::new(scalar(2.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
        assert!(kl_divergence(&p, &q).unwrap().item().unwrap().is_infinite());
    }

    // -- Uniform-Exponential -------------------------------------------------

    #[test]
    fn test_kl_uniform_exponential_known_value() {
        // kl.py:889-893. torch: KL(Uniform(0.5, 2.0) || Exp(1.0)).
        let p = Uniform::new(scalar(0.5f64).unwrap(), scalar(2.0f64).unwrap()).unwrap();
        let q = Exponential::new(scalar(1.0f64).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        approx(
            kl.item().unwrap(),
            0.844_534_891_891_835_6,
            1e-10,
            "Unif-Exp",
        );
    }

    #[test]
    fn test_kl_uniform_exponential_neg_low_is_inf() {
        let p = Uniform::new(scalar(-0.5f64).unwrap(), scalar(2.0f64).unwrap()).unwrap();
        let q = Exponential::new(scalar(1.0f64).unwrap()).unwrap();
        assert!(kl_divergence(&p, &q).unwrap().item().unwrap().is_infinite());
    }

    // -- Uniform-Gamma -------------------------------------------------------

    #[test]
    fn test_kl_uniform_gamma_known_value() {
        // kl.py:896-909. torch: KL(Uniform(0.5, 2.0) || Gamma(2.0, 1.0)).
        let p = Uniform::new(scalar(0.5f64).unwrap(), scalar(2.0f64).unwrap()).unwrap();
        let q = Gamma::new(scalar(2.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        approx(
            kl.item().unwrap(),
            0.689_289_590_958_593_4,
            1e-10,
            "Unif-Gamma",
        );
    }

    // -- Uniform-Pareto ------------------------------------------------------

    #[test]
    fn test_kl_uniform_pareto_known_value() {
        // kl.py:934-941. torch: KL(Uniform(2.0, 4.0) || Pareto(1, 3)).
        let p = Uniform::new(scalar(2.0f64).unwrap(), scalar(4.0f64).unwrap()).unwrap();
        let q = Pareto::new(scalar(1.0f64).unwrap(), scalar(3.0f64).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        approx(
            kl.item().unwrap(),
            2.526_006_697_491_288,
            1e-10,
            "Unif-Pareto",
        );
    }

    #[test]
    fn test_kl_uniform_pareto_low_below_scale_is_inf() {
        let p = Uniform::new(scalar(0.5f64).unwrap(), scalar(4.0f64).unwrap()).unwrap();
        let q = Pareto::new(scalar(1.0f64).unwrap(), scalar(3.0f64).unwrap()).unwrap();
        assert!(kl_divergence(&p, &q).unwrap().item().unwrap().is_infinite());
    }

    // -- Uniform-Beta --------------------------------------------------------

    #[test]
    fn test_kl_uniform_beta_known_value() {
        // kl.py:847-869. torch: KL(Uniform(0.2, 0.8) || Beta(2,3)).
        let p = Uniform::new(scalar(0.2f64).unwrap(), scalar(0.8f64).unwrap()).unwrap();
        let q = Beta::new(scalar(2.0f64).unwrap(), scalar(3.0f64).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        approx(
            kl.item().unwrap(),
            0.309_055_266_800_728_8,
            1e-10,
            "Unif-Beta",
        );
    }

    #[test]
    fn test_kl_uniform_beta_support_violation_is_inf() {
        // Uniform support escapes [0,1] → +inf.
        let p = Uniform::new(scalar(-0.2f64).unwrap(), scalar(0.8f64).unwrap()).unwrap();
        let q = Beta::new(scalar(2.0f64).unwrap(), scalar(3.0f64).unwrap()).unwrap();
        assert!(kl_divergence(&p, &q).unwrap().item().unwrap().is_infinite());
    }

    // -- #1374: Binomial-Binomial + Poisson-Binomial -------------------------
    // Reference values from live `torch.distributions.kl_divergence` at float64
    // (torch 2.11, this machine 2026-05-27); each traces to a `@register_kl`
    // body in `torch/distributions/kl.py` (R-CHAR-3 non-tautological).

    #[test]
    fn test_kl_binomial_binomial_same_is_zero() {
        // KL(Binomial(10, 0.3) || Binomial(10, 0.3)) = 0.
        let mk = || Binomial::new(scalar(10.0f64).unwrap(), scalar(0.3f64).unwrap()).unwrap();
        let kl = kl_divergence(&mk(), &mk()).unwrap();
        approx(kl.item().unwrap(), 0.0, 1e-12, "Binomial-Binomial same");
    }

    #[test]
    fn test_kl_binomial_binomial_known_value() {
        // torch: kl_divergence(Binomial(10, 0.3), Binomial(10, 0.5)) (f64)
        //        == 0.8228287850505189 (torch 2.11; kl.py:231-244).
        let p = Binomial::new(scalar(10.0f64).unwrap(), scalar(0.3f64).unwrap()).unwrap();
        let q = Binomial::new(scalar(10.0f64).unwrap(), scalar(0.5f64).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        approx(
            kl.item().unwrap(),
            0.822_828_785_050_518_9,
            1e-9,
            "Binomial-Binomial",
        );
    }

    #[test]
    fn test_kl_binomial_binomial_known_value_2() {
        // torch: kl_divergence(Binomial(20, 0.6), Binomial(20, 0.4)) (f64)
        //        == 1.6094379124341003 (kl.py:231-244):
        //        n·(p·(logit_p - logit_q) + ln(1-p) - ln(1-q)) with n=20.
        let p = Binomial::new(scalar(20.0f64).unwrap(), scalar(0.6f64).unwrap()).unwrap();
        let q = Binomial::new(scalar(20.0f64).unwrap(), scalar(0.4f64).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        // n·(p·ln(p/q · (1-q)/(1-p)) + ln((1-p)/(1-q))) for p=.6,q=.4,n=20.
        let n = 20.0f64;
        let (pp, qq) = (0.6f64, 0.4f64);
        let expected = n
            * (pp * ((pp / qq).ln() - ((1.0 - pp) / (1.0 - qq)).ln())
                + ((1.0 - pp) / (1.0 - qq)).ln());
        approx(kl.item().unwrap(), expected, 1e-9, "Binomial-Binomial 2");
    }

    #[test]
    fn test_kl_binomial_binomial_larger_np_is_inf() {
        // p.total_count > q.total_count → +inf (support of the larger-n
        // Binomial is not covered). kl.py:242-243.
        let p = Binomial::new(scalar(12.0f64).unwrap(), scalar(0.5f64).unwrap()).unwrap();
        let q = Binomial::new(scalar(8.0f64).unwrap(), scalar(0.5f64).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        assert!(
            kl.item().unwrap().is_infinite() && kl.item().unwrap() > 0.0,
            "expected +inf, got {}",
            kl.item().unwrap()
        );
    }

    #[test]
    fn test_kl_binomial_binomial_smaller_np_errors() {
        // p.total_count < q.total_count → NotImplementedError (InvalidArgument).
        // kl.py:235-238.
        let p = Binomial::new(scalar(5.0f64).unwrap(), scalar(0.5f64).unwrap()).unwrap();
        let q = Binomial::new(scalar(9.0f64).unwrap(), scalar(0.5f64).unwrap()).unwrap();
        assert!(kl_divergence(&p, &q).is_err());
    }

    #[test]
    fn test_kl_poisson_binomial_is_inf() {
        // kl.py:842 `_kl_poisson_infinity` → +inf everywhere (a Poisson's
        // unbounded support is not covered by a Binomial's bounded {0..n}).
        let p = Poisson::new(tensor(&[1.5f64]).unwrap()).unwrap();
        let q = Binomial::new(scalar(10.0f64).unwrap(), scalar(0.3f64).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        assert!(
            kl.item().unwrap().is_infinite() && kl.item().unwrap() > 0.0,
            "expected +inf, got {}",
            kl.item().unwrap()
        );
    }

    // -- Drift prevention: doc table vs dispatcher ---------------------------

    /// Guards against the failure mode in #1124, where the doc table on
    /// `kl_divergence` listed 6 pairs while the dispatcher had grown to 12.
    /// Parses this very source file and asserts:
    ///   1. `KL_SUPPORTED_PAIR_COUNT` matches the public `kl_supported_pair_count()`.
    ///   2. The supported-pairs doc table on `kl_divergence` has exactly that many rows.
    ///   3. The dispatcher in `kl_dispatch` has exactly that many `downcast_ref` arms.
    #[test]
    fn kl_doc_table_matches_dispatcher() {
        const SRC: &str = include_str!("kl.rs");
        let expected = kl_supported_pair_count();
        assert_eq!(
            expected, KL_SUPPORTED_PAIR_COUNT,
            "public accessor must mirror the internal constant"
        );

        // (1) Count rows of the markdown table inside the `kl_divergence` rustdoc:
        // each data row begins with `/// |` and is not the header `| P | Q |`
        // nor the separator `|---|---|`. The table block ends at the first
        // blank doc line after we have started counting.
        let table_rows = count_doc_table_rows(SRC);
        assert_eq!(
            table_rows, expected,
            "doc-table rows ({table_rows}) on `kl_divergence` must equal \
             KL_SUPPORTED_PAIR_COUNT ({expected}) — update both together"
        );

        // (2) Count dispatcher arms: each registered pair uses
        // `p.downcast_ref::<...>()` exactly once. We count the occurrences of
        // that fragment in the body of `fn kl_dispatch`.
        let dispatch_arms = count_dispatcher_arms(SRC);
        assert_eq!(
            dispatch_arms, expected,
            "dispatcher arms ({dispatch_arms}) in `kl_dispatch` must equal \
             KL_SUPPORTED_PAIR_COUNT ({expected})"
        );
    }

    fn count_doc_table_rows(src: &str) -> usize {
        let mut in_table = false;
        let mut rows = 0usize;
        for raw in src.lines() {
            let line = raw.trim_start();
            // Find the start of the table: header row `| P | Q |`.
            if !in_table {
                if line.starts_with("///") && line.contains("| P | Q |") {
                    in_table = true;
                }
                continue;
            }
            // Inside the table: stop at the first non-`///` line or a `///`
            // line that no longer looks like a table row.
            let Some(rest) = line.strip_prefix("///") else {
                break;
            };
            let cell = rest.trim();
            if !cell.starts_with('|') {
                break;
            }
            // Skip the `|---|---|` separator row.
            if cell.chars().all(|c| matches!(c, '|' | '-' | ' ')) {
                continue;
            }
            rows += 1;
        }
        rows
    }

    fn count_dispatcher_arms(src: &str) -> usize {
        // Slice the body of `fn kl_dispatch` so we don't accidentally count
        // `downcast_ref` mentions elsewhere in the file (there are none today,
        // but we want to stay robust against future helpers/tests).
        let start = src
            .find("fn kl_dispatch")
            .expect("kl_dispatch must be defined in this file");
        // End of body: the closing `}` of the function — heuristically the
        // line beginning with `// -----` that follows it, or the next `fn `.
        let tail = &src[start..];
        let end = tail
            .find("\n// ----------------------------------")
            .unwrap_or(tail.len());
        let body = &tail[..end];
        // Each registered pair uses `p.downcast_ref::<...>()` exactly once.
        body.matches("p.downcast_ref::<").count()
    }

    // -- ln_gamma numerical sanity -------------------------------------------

    #[test]
    fn test_ln_gamma_known_values() {
        // lnΓ(1) = 0, lnΓ(2) = 0, lnΓ(3) = ln(2) ≈ 0.6931,
        // lnΓ(4) = ln(6) ≈ 1.7918, lnΓ(5) = ln(24) ≈ 3.1781.
        // After consolidation onto Lanczos in special_fns, error is < 1e-12
        // for x > 0.5 — tighten tolerance accordingly.
        assert!((lgamma_scalar(1.0f64) - 0.0).abs() < 1e-12);
        assert!((lgamma_scalar(2.0f64) - 0.0).abs() < 1e-12);
        assert!((lgamma_scalar(3.0f64) - 2.0f64.ln()).abs() < 1e-12);
        assert!((lgamma_scalar(4.0f64) - 6.0f64.ln()).abs() < 1e-12);
        assert!((lgamma_scalar(5.0f64) - 24.0f64.ln()).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // #1374 wave-N: MultivariateNormal & LowRankMultivariateNormal pairs.
    // Reference values from live `torch.distributions.kl_divergence` (f64,
    // torch 2.11, this machine 2026-05-27); each `expected` traces to a
    // `@register_kl` body in `torch/distributions/kl.py` (R-CHAR-3
    // non-tautological).
    // -----------------------------------------------------------------------

    use crate::{LowRankMultivariateNormal, MultivariateNormal};
    use ferrotorch_core::creation::from_slice;

    // -- MultivariateNormal-MultivariateNormal -------------------------------

    #[test]
    fn test_kl_mvn_mvn_same_is_zero() {
        let loc = tensor(&[0.0f64, 1.0]).unwrap();
        let l = from_slice(&[1.0f64, 0.0, 0.5, 1.0], &[2, 2]).unwrap();
        let p = MultivariateNormal::from_scale_tril(loc.clone(), l.clone()).unwrap();
        let q = MultivariateNormal::from_scale_tril(loc, l).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        approx(kl.item().unwrap(), 0.0, 1e-12, "MVN-MVN same");
    }

    #[test]
    fn test_kl_mvn_mvn_known_value() {
        // torch.distributions.kl_divergence(
        //   MVN([0,1], scale_tril=[[1,0],[0.5,1]]),
        //   MVN([1,-1], scale_tril=[[2,0],[0.3,1.5]])) (kl.py:442-464).
        let p = MultivariateNormal::from_scale_tril(
            tensor(&[0.0f64, 1.0]).unwrap(),
            from_slice(&[1.0f64, 0.0, 0.5, 1.0], &[2, 2]).unwrap(),
        )
        .unwrap();
        let q = MultivariateNormal::from_scale_tril(
            tensor(&[1.0f64, -1.0]).unwrap(),
            from_slice(&[2.0f64, 0.0, 0.3, 1.5], &[2, 2]).unwrap(),
        )
        .unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        approx(
            kl.item().unwrap(),
            1.625_278_955_334_776_2,
            1e-10,
            "MVN-MVN",
        );
    }

    #[test]
    fn test_kl_mvn_mvn_known_value_3d() {
        // torch: KL(MVN([0,1,2], L_p) || MVN([0.5,0.5,0.5], L_q)) with the
        // lower-triangular factors below (kl.py:442-464).
        let p = MultivariateNormal::from_scale_tril(
            tensor(&[0.0f64, 1.0, 2.0]).unwrap(),
            from_slice(&[1.0f64, 0.0, 0.0, 0.2, 1.1, 0.0, 0.1, 0.3, 0.9], &[3, 3]).unwrap(),
        )
        .unwrap();
        let q = MultivariateNormal::from_scale_tril(
            tensor(&[0.5f64, 0.5, 0.5]).unwrap(),
            from_slice(&[1.5f64, 0.0, 0.0, 0.0, 1.2, 0.0, 0.4, 0.1, 1.3], &[3, 3]).unwrap(),
        )
        .unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        approx(
            kl.item().unwrap(),
            1.170_769_970_022_585_3,
            1e-10,
            "MVN-MVN 3d",
        );
    }

    #[test]
    fn test_kl_mvn_mvn_nonnegative() {
        let p = MultivariateNormal::from_scale_tril(
            tensor(&[2.0f64, -3.0]).unwrap(),
            from_slice(&[0.7f64, 0.0, 0.2, 1.3], &[2, 2]).unwrap(),
        )
        .unwrap();
        let q = MultivariateNormal::from_scale_tril(
            tensor(&[0.0f64, 0.0]).unwrap(),
            from_slice(&[1.0f64, 0.0, 0.0, 1.0], &[2, 2]).unwrap(),
        )
        .unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        assert!(kl.item().unwrap() >= -1e-12, "got {}", kl.item().unwrap());
    }

    #[test]
    fn test_kl_mvn_mvn_event_dim_mismatch_errs() {
        let p = MultivariateNormal::from_scale_tril(
            tensor(&[0.0f64, 0.0]).unwrap(),
            from_slice(&[1.0f64, 0.0, 0.0, 1.0], &[2, 2]).unwrap(),
        )
        .unwrap();
        let q = MultivariateNormal::from_scale_tril(
            tensor(&[0.0f64, 0.0, 0.0]).unwrap(),
            from_slice(&[1.0f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], &[3, 3]).unwrap(),
        )
        .unwrap();
        assert!(kl_divergence(&p, &q).is_err());
    }

    // -- LowRankMultivariateNormal pairs -------------------------------------

    #[test]
    fn test_kl_lowrank_lowrank_known_value() {
        // torch: KL(LowRankMVN([0,1], W=[[1],[0.5]], D=[1,2]) ||
        //            LowRankMVN([1,-1], W=[[0.8],[0.2]], D=[1.5,1])) (kl.py:341-373).
        let p = LowRankMultivariateNormal::new(
            tensor(&[0.0f64, 1.0]).unwrap(),
            from_slice(&[1.0f64, 0.5], &[2, 1]).unwrap(),
            tensor(&[1.0f64, 2.0]).unwrap(),
        )
        .unwrap();
        let q = LowRankMultivariateNormal::new(
            tensor(&[1.0f64, -1.0]).unwrap(),
            from_slice(&[0.8f64, 0.2], &[2, 1]).unwrap(),
            tensor(&[1.5f64, 1.0]).unwrap(),
        )
        .unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        approx(kl.item().unwrap(), 2.528_723_734_168_518_3, 1e-9, "LR-LR");
    }

    #[test]
    fn test_kl_lowrank_lowrank_same_is_zero() {
        let mk = || {
            LowRankMultivariateNormal::new(
                tensor(&[0.0f64, 1.0]).unwrap(),
                from_slice(&[1.0f64, 0.5], &[2, 1]).unwrap(),
                tensor(&[1.0f64, 2.0]).unwrap(),
            )
            .unwrap()
        };
        let kl = kl_divergence(&mk(), &mk()).unwrap();
        approx(kl.item().unwrap(), 0.0, 1e-10, "LR-LR same");
    }

    #[test]
    fn test_kl_mvn_lowrank_known_value() {
        // torch: KL(MVN([0,1], scale_tril=[[1.2,0],[0.3,1.0]]) ||
        //            LowRankMVN([1,-1], W=[[0.8],[0.2]], D=[1.5,1])) (kl.py:375-403).
        let p = MultivariateNormal::from_scale_tril(
            tensor(&[0.0f64, 1.0]).unwrap(),
            from_slice(&[1.2f64, 0.0, 0.3, 1.0], &[2, 2]).unwrap(),
        )
        .unwrap();
        let q = LowRankMultivariateNormal::new(
            tensor(&[1.0f64, -1.0]).unwrap(),
            from_slice(&[0.8f64, 0.2], &[2, 1]).unwrap(),
            tensor(&[1.5f64, 1.0]).unwrap(),
        )
        .unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        approx(kl.item().unwrap(), 2.383_498_032_479_09, 1e-9, "MVN-LR");
    }

    #[test]
    fn test_kl_lowrank_mvn_known_value() {
        // torch: KL(LowRankMVN([0,1], W=[[1],[0.5]], D=[1,2]) ||
        //            MVN([1,-1], scale_tril=[[1.3,0],[0.1,1.1]])) (kl.py:405-440).
        let p = LowRankMultivariateNormal::new(
            tensor(&[0.0f64, 1.0]).unwrap(),
            from_slice(&[1.0f64, 0.5], &[2, 1]).unwrap(),
            tensor(&[1.0f64, 2.0]).unwrap(),
        )
        .unwrap();
        let q = MultivariateNormal::from_scale_tril(
            tensor(&[1.0f64, -1.0]).unwrap(),
            from_slice(&[1.3f64, 0.0, 0.1, 1.1], &[2, 2]).unwrap(),
        )
        .unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        approx(kl.item().unwrap(), 2.207_128_053_688_782_3, 1e-9, "LR-MVN");
    }
}
