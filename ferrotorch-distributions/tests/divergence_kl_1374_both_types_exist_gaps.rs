//! Divergence audit of commit `be2563bc3` (#1374, KL MVN + LowRankMVN pairs).
//!
//! The builder's commit message claims the ONLY remaining KL gaps are blocked
//! on missing distribution TYPES:
//!
//!   "NOT-STARTED (missing distribution TYPE, prereq named): Binomial-Binomial +
//!    Poisson-Binomial (need a `Binomial` struct), Geometric-Geometric (need a
//!    `Geometric` struct), ContinuousBernoulli-* pairs (need a `ContinuousBernoulli`
//!    struct), Bernoulli-Poisson ... Also deprioritised: the all-`+inf`
//!    degenerate-support cross-pairs"
//!
//! This is a divergence from the FULL SCOPE of #1374 ("full ~75-pair PyTorch
//! coverage"). Cross-referencing `torch/distributions/kl.py`'s `@register_kl`
//! list against the distribution types ferrotorch already exports
//! (`ferrotorch-distributions/src/lib.rs:98-125`) shows 28 registered pairs
//! where BOTH operand types ALREADY EXIST in ferrotorch yet are NOT in the
//! `kl_dispatch` chain. None of these is "blocked on a missing type"; several
//! are NOT `+inf`-degenerate (OneHotCategorical-OneHotCategorical,
//! Bernoulli-Poisson, Normal-Laplace are finite closed forms).
//!
//! Worse, the commit message MISCATEGORISES `Bernoulli-Poisson` as needing the
//! Bernoulli type — but `ferrotorch_distributions::Bernoulli` exists and is
//! used by the shipped Bernoulli-Bernoulli arm.
//!
//! Each test below constructs a pair whose KL torch computes (a finite value),
//! and asserts ferrotorch returns that value. Each FAILS today because
//! `kl_dispatch` falls through to the `InvalidArgument` "no formula registered"
//! arm for these pairs.
//!
//! Reference values from live `torch.distributions.kl_divergence` at float64
//! (torch 2.11.0, 2026-05-27); non-tautological per R-CHAR-3.
//!
//! Tracking: #1562 (#1374-completeness blocker). Marked `#[ignore]` because the
//! issue is now tracked; these tests stay red until the dispatch arms land.

use ferrotorch_core::creation::{scalar, tensor};
use ferrotorch_distributions::kl::kl_divergence;
use ferrotorch_distributions::{
    Bernoulli, Beta, Exponential, Gamma, Gumbel, Laplace, Normal, OneHotCategorical, Pareto,
    Poisson, Uniform,
};

fn item(t: ferrotorch_core::tensor::Tensor<f64>) -> f64 {
    t.item().unwrap()
}

/// Divergence: ferrotorch has no KL for `OneHotCategorical || OneHotCategorical`
/// even though BOTH types exist (`one_hot_categorical.rs`, exported in
/// `lib.rs:116`). torch registers it at
/// `pytorch torch/distributions/kl.py:474-476`:
///   `def _kl_onehotcategorical_onehotcategorical(p, q):
///        return _kl_categorical_categorical(p._categorical, q._categorical)`
/// This is a FINITE closed form (delegates to Categorical-Categorical, which
/// ferrotorch already ships), not a `+inf` degenerate pair, and not blocked on
/// any missing type.
/// Upstream returns 0.18609809382700085; ferrotorch returns
/// `Err(InvalidArgument "No KL divergence formula registered ...")`.
/// Tracking: #1562
#[test]
fn divergence_kl_onehotcategorical_onehotcategorical_missing() {
    let p = OneHotCategorical::new(tensor(&[0.2f64, 0.3, 0.5]).unwrap()).unwrap();
    let q = OneHotCategorical::new(tensor(&[0.1f64, 0.6, 0.3]).unwrap()).unwrap();
    let kl = kl_divergence(&p, &q).expect("OneHotCategorical-OneHotCategorical KL must exist");
    assert!(
        (item(kl) - 0.186_098_093_827_000_85).abs() < 1e-12,
        "expected torch value 0.18609809382700085"
    );
}

/// Divergence: ferrotorch has no KL for `Bernoulli || Poisson` even though both
/// types exist (`bernoulli.rs`, `poisson.rs`). The commit message wrongly says
/// this is "blocked on the Bernoulli type"; Bernoulli is already shipped.
/// torch registers it at `pytorch torch/distributions/kl.py:513-516`:
///   `def _kl_bernoulli_poisson(p, q):
///        return -p.entropy() - (p.probs * q.rate.log() - q.rate)`
/// Upstream returns 0.1961381811267262; ferrotorch returns
/// `Err(InvalidArgument)`.
/// Tracking: #1562
#[test]
fn divergence_kl_bernoulli_poisson_missing() {
    let p = Bernoulli::new(tensor(&[0.3f64]).unwrap()).unwrap();
    let q = Poisson::new(tensor(&[0.7f64]).unwrap()).unwrap();
    let kl =
        kl_divergence(&p, &q).expect("Bernoulli-Poisson KL must exist (Bernoulli type exists)");
    assert!(
        (item(kl) - 0.196_138_181_126_726_2).abs() < 1e-12,
        "expected torch value 0.1961381811267262"
    );
}

/// Divergence: ferrotorch has no KL for `Normal || Laplace` even though both
/// types exist (`normal.rs`, `laplace.rs`). torch registers a FINITE closed
/// form at `pytorch torch/distributions/kl.py:782-792`
/// (`_kl_normal_laplace`, uses `erf`). This is the symmetric partner of the
/// already-shipped `Laplace-Normal` arm — its absence is an oversight, not a
/// type or `+inf` blocker.
/// Upstream returns 0.32210248072031933; ferrotorch returns
/// `Err(InvalidArgument)`.
/// Tracking: #1562
#[test]
fn divergence_kl_normal_laplace_missing() {
    let p = Normal::new(tensor(&[0.5f64]).unwrap(), tensor(&[1.2f64]).unwrap()).unwrap();
    let q = Laplace::new(tensor(&[-0.3f64]).unwrap(), tensor(&[0.8f64]).unwrap()).unwrap();
    let kl = kl_divergence(&p, &q).expect("Normal-Laplace KL must exist");
    assert!(
        (item(kl) - 0.322_102_480_720_319_33).abs() < 1e-12,
        "expected torch value 0.32210248072031933"
    );
}

// ---------------------------------------------------------------------------
// Extended coverage (#1374 / #1562): additional finite-pair reference values +
// the support-mismatch `+inf` family. Reference values from PyTorch's
// `@register_kl` bodies in `torch/distributions/kl.py` (non-tautological per
// R-CHAR-3 — each traces to a kl.py:line + an independently re-derived float64
// constant via `torch.distributions.kl_divergence`, torch 2.11, 2026-05-27).
// ---------------------------------------------------------------------------

/// Second Bernoulli-Poisson point. torch `kl.py:513-516` returns
/// 0.5251049910180157 for `Bernoulli(0.6) || Poisson(1.4)`.
#[test]
fn divergence_kl_bernoulli_poisson_second_point() {
    let p = Bernoulli::new(tensor(&[0.6f64]).unwrap()).unwrap();
    let q = Poisson::new(tensor(&[1.4f64]).unwrap()).unwrap();
    let kl = item(kl_divergence(&p, &q).expect("Bernoulli-Poisson KL must exist"));
    assert!(
        (kl - 0.525_104_991_018_015_7).abs() < 1e-12,
        "expected torch value 0.5251049910180157, got {kl}"
    );
}

/// Second Normal-Laplace point: torch `kl.py:782-792` returns
/// 0.07209320815813802 for `Normal(0,1) || Laplace(0,1)`.
#[test]
fn divergence_kl_normal_laplace_standard_point() {
    let p = Normal::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
    let q = Laplace::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
    let kl = item(kl_divergence(&p, &q).expect("Normal-Laplace KL must exist"));
    assert!(
        (kl - 0.072_093_208_158_138_02).abs() < 1e-12,
        "expected torch value 0.07209320815813802, got {kl}"
    );
}

/// OneHotCategorical(p)||OneHotCategorical(p) is 0 (self-KL); the divergence
/// equals the Categorical-Categorical closed form (`kl.py:474-476`).
#[test]
fn divergence_kl_onehotcategorical_self_is_zero() {
    let p = OneHotCategorical::new(tensor(&[0.25f64, 0.25, 0.5]).unwrap()).unwrap();
    let kl = item(kl_divergence(&p, &p).expect("OHC-OHC self KL must exist"));
    assert!(kl.abs() < 1e-12, "self-KL must be 0, got {kl}");
}

/// Every `_infinite_like` registration (`kl.py:528,620-624,665-669,718-724,
/// 740-746,761-766,795-797,841-844`) is a support mismatch: torch returns
/// `+inf` everywhere because `q`'s support does not cover `p`'s. ferrotorch
/// must return `+inf` exactly (not an `Err`), matching `torch.full_like(.,inf)`.
#[test]
fn divergence_kl_support_mismatch_family_is_positive_infinity() {
    macro_rules! assert_inf {
        ($p:expr, $q:expr, $name:literal) => {{
            let kl = kl_divergence(&$p, &$q)
                .unwrap_or_else(|e| panic!("{} KL must be registered, got {:?}", $name, e));
            let v = item(kl);
            assert!(
                v.is_infinite() && v > 0.0,
                "{} KL must be +inf (support mismatch), got {}",
                $name,
                v
            );
        }};
    }

    let beta = || Beta::new(scalar(2.0f64).unwrap(), scalar(3.0f64).unwrap()).unwrap();
    let pareto = || Pareto::new(scalar(1.0f64).unwrap(), scalar(2.0f64).unwrap()).unwrap();
    let unif = || Uniform::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
    let expo = || Exponential::new(scalar(1.0f64).unwrap()).unwrap();
    let gamma = || Gamma::new(scalar(2.0f64).unwrap(), scalar(1.5f64).unwrap()).unwrap();
    let gumbel = || Gumbel::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
    let normal = || Normal::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
    let bern = || Bernoulli::new(tensor(&[0.4f64]).unwrap()).unwrap();
    let pois = || Poisson::new(tensor(&[0.9f64]).unwrap()).unwrap();

    assert_inf!(beta(), pareto(), "Beta-Pareto");
    assert_inf!(expo(), beta(), "Exponential-Beta");
    assert_inf!(expo(), pareto(), "Exponential-Pareto");
    assert_inf!(expo(), unif(), "Exponential-Uniform");
    assert_inf!(gamma(), beta(), "Gamma-Beta");
    assert_inf!(gamma(), pareto(), "Gamma-Pareto");
    assert_inf!(gamma(), unif(), "Gamma-Uniform");
    assert_inf!(gumbel(), beta(), "Gumbel-Beta");
    assert_inf!(gumbel(), expo(), "Gumbel-Exponential");
    assert_inf!(gumbel(), gamma(), "Gumbel-Gamma");
    assert_inf!(gumbel(), pareto(), "Gumbel-Pareto");
    assert_inf!(gumbel(), unif(), "Gumbel-Uniform");
    assert_inf!(
        Laplace::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap(),
        beta(),
        "Laplace-Beta"
    );
    assert_inf!(
        Laplace::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap(),
        expo(),
        "Laplace-Exponential"
    );
    assert_inf!(
        Laplace::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap(),
        gamma(),
        "Laplace-Gamma"
    );
    assert_inf!(
        Laplace::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap(),
        pareto(),
        "Laplace-Pareto"
    );
    assert_inf!(
        Laplace::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap(),
        unif(),
        "Laplace-Uniform"
    );
    assert_inf!(normal(), beta(), "Normal-Beta");
    assert_inf!(normal(), expo(), "Normal-Exponential");
    assert_inf!(normal(), gamma(), "Normal-Gamma");
    assert_inf!(normal(), pareto(), "Normal-Pareto");
    assert_inf!(pareto(), beta(), "Pareto-Beta");
    assert_inf!(pareto(), unif(), "Pareto-Uniform");
    assert_inf!(pois(), bern(), "Poisson-Bernoulli");
}
