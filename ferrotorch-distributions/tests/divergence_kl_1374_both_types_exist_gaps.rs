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

use ferrotorch_core::creation::tensor;
use ferrotorch_distributions::kl::kl_divergence;
use ferrotorch_distributions::{Bernoulli, Laplace, Normal, OneHotCategorical, Poisson};

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
#[ignore = "divergence: OneHotCategorical-OneHotCategorical KL missing though both types exist; tracking #1562"]
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
#[ignore = "divergence: Bernoulli-Poisson KL missing; commit miscategorised as Bernoulli-type-blocked though Bernoulli exists; tracking #1562"]
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
#[ignore = "divergence: Normal-Laplace KL missing though both types exist (symmetric partner of shipped Laplace-Normal); tracking #1562"]
fn divergence_kl_normal_laplace_missing() {
    let p = Normal::new(tensor(&[0.5f64]).unwrap(), tensor(&[1.2f64]).unwrap()).unwrap();
    let q = Laplace::new(tensor(&[-0.3f64]).unwrap(), tensor(&[0.8f64]).unwrap()).unwrap();
    let kl = kl_divergence(&p, &q).expect("Normal-Laplace KL must exist");
    assert!(
        (item(kl) - 0.322_102_480_720_319_33).abs() < 1e-12,
        "expected torch value 0.32210248072031933"
    );
}
