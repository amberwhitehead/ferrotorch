//! Critic re-audit of #1374/#1379 (commit 72312d32a), per the #1542 umbrella.
//!
//! The build's own wave-M audit cross-checks the 12 new KL pairs ONLY against a
//! loose Monte-Carlo estimate (tolerance 5e-2) and analytic identities. A
//! Monte-Carlo check at 5e-2 does NOT pin the closed-form constant: a small
//! sign/term error in a KL body produces a number that still lands inside the
//! O(1/sqrt(N)) MC band. This file pins each new pair to the EXACT
//! `torch.distributions.kl_divergence` value (computed in float64) to ~1e-5,
//! the tolerance the #1374 task specifies.
//!
//! R-CHAR-3: every expected constant below was produced by live-calling
//! PyTorch's `torch.distributions.kl_divergence` in float64 (NOT copied from
//! the ferrotorch side). The generating script is reproduced in each block's
//! comment so the oracle is auditable.
//!
//! NOTE on #1379 (trigamma/polygamma): `trigamma_scalar` and `polygamma_scalar`
//! are `pub(crate)` in `special_fns.rs` and `special_fns` is a `pub(crate)`
//! module, so neither is reachable from an integration test. They therefore
//! cannot be pinned to scipy from here; the only existing check (wave-M) is a
//! finite-difference-of-digamma internal-consistency test, not a scipy pin. The
//! gap is reported by the critic, not testable in this file.

use ferrotorch_core::creation::{scalar, tensor};
use ferrotorch_distributions::kl::kl_divergence;
use ferrotorch_distributions::{Beta, Dirichlet, Exponential, Gamma, Normal, Pareto, Uniform};

/// Assert `kl_divergence(p, q)` matches the live-torch float64 oracle.
macro_rules! assert_kl {
    ($p:expr, $q:expr, $expected:expr, $name:literal) => {{
        let kl = kl_divergence(&$p, &$q).unwrap().item().unwrap();
        let exp: f64 = $expected;
        assert!(
            (kl - exp).abs() < 1e-5,
            "{}: ferrotorch KL = {kl}, torch = {exp} (delta {})",
            $name,
            (kl - exp).abs()
        );
    }};
}

fn sc(x: f64) -> ferrotorch_core::Tensor<f64> {
    scalar(x).unwrap()
}

// ---------------------------------------------------------------------------
// Oracle (float64), reproducible:
//   import torch; from torch.distributions import *
//   torch.set_default_dtype(torch.float64)
//   kl = torch.distributions.kl_divergence
// ---------------------------------------------------------------------------

#[test]
fn critic_kl_dirichlet_dirichlet_value() {
    // kl(Dirichlet([2,3,4]), Dirichlet([1,1,1]))
    let p = Dirichlet::new(tensor(&[2.0f64, 3.0, 4.0]).unwrap()).unwrap();
    let q = Dirichlet::new(tensor(&[1.0f64, 1.0, 1.0]).unwrap()).unwrap();
    assert_kl!(p, q, 0.6194062152544486, "Dirichlet-Dirichlet");
}

#[test]
fn critic_kl_beta_exponential_value() {
    // kl(Beta(2,3), Exponential(1.5))
    let p = Beta::new(sc(2.0), sc(3.0)).unwrap();
    let q = Exponential::new(sc(1.5)).unwrap();
    assert_kl!(p, q, 0.42944154167983606, "Beta-Exponential");
}

#[test]
fn critic_kl_beta_gamma_value() {
    // kl(Beta(2,3), Gamma(conc=2, rate=1.5))
    let p = Beta::new(sc(2.0), sc(3.0)).unwrap();
    let q = Gamma::new(sc(2.0), sc(1.5)).unwrap();
    assert_kl!(p, q, 1.1073097669050047, "Beta-Gamma");
}

#[test]
fn critic_kl_beta_normal_value() {
    // kl(Beta(2,3), Normal(loc=0.5, scale=1))
    let p = Beta::new(sc(2.0), sc(3.0)).unwrap();
    let q = Normal::new(sc(0.5), sc(1.0)).unwrap();
    assert_kl!(p, q, 1.178845182992673, "Beta-Normal");
}

#[test]
fn critic_kl_beta_uniform_value() {
    // kl(Beta(2,3), Uniform(-1,2)) — support [-1,2] covers [0,1], finite.
    let p = Beta::new(sc(2.0), sc(3.0)).unwrap();
    let q = Uniform::new(sc(-1.0), sc(2.0)).unwrap();
    assert_kl!(p, q, 1.3335189384561101, "Beta-Uniform");
}

#[test]
fn critic_kl_beta_uniform_support_escape_is_inf() {
    // Uniform(0.1,0.9) does NOT cover Beta support [0,1] -> +inf.
    let p = Beta::new(sc(2.0), sc(3.0)).unwrap();
    let q = Uniform::new(sc(0.1), sc(0.9)).unwrap();
    let kl = kl_divergence(&p, &q).unwrap().item().unwrap();
    assert!(kl.is_infinite() && kl > 0.0, "Beta-Uniform escape must be +inf, got {kl}");
}

#[test]
fn critic_kl_pareto_exponential_value() {
    // kl(Pareto(scale=1, alpha=3), Exponential(0.5))
    let p = Pareto::new(sc(1.0), sc(3.0)).unwrap();
    let q = Exponential::new(sc(0.5)).unwrap();
    assert_kl!(p, q, 1.2084261358947215, "Pareto-Exponential");
}

#[test]
fn critic_kl_pareto_exponential_alpha_le_1_is_inf() {
    // alpha <= 1 -> +inf.
    let p = Pareto::new(sc(1.0), sc(0.5)).unwrap();
    let q = Exponential::new(sc(0.5)).unwrap();
    let kl = kl_divergence(&p, &q).unwrap().item().unwrap();
    assert!(kl.is_infinite() && kl > 0.0, "Pareto-Exp alpha<=1 must be +inf, got {kl}");
}

#[test]
fn critic_kl_pareto_gamma_value() {
    // kl(Pareto(scale=2, alpha=3), Gamma(conc=2, rate=1))
    let p = Pareto::new(sc(2.0), sc(3.0)).unwrap();
    let q = Gamma::new(sc(2.0), sc(1.0)).unwrap();
    assert_kl!(p, q, 1.0456512608815522, "Pareto-Gamma");
}

#[test]
fn critic_kl_pareto_normal_value() {
    // kl(Pareto(scale=1, alpha=3), Normal(loc=0, scale=2))
    let p = Pareto::new(sc(1.0), sc(3.0)).unwrap();
    let q = Normal::new(sc(0.0), sc(2.0)).unwrap();
    assert_kl!(p, q, 1.7523646690993941, "Pareto-Normal");
}

#[test]
fn critic_kl_pareto_normal_alpha_le_2_is_inf() {
    // alpha <= 2 -> +inf.
    let p = Pareto::new(sc(1.0), sc(1.5)).unwrap();
    let q = Normal::new(sc(0.0), sc(2.0)).unwrap();
    let kl = kl_divergence(&p, &q).unwrap().item().unwrap();
    assert!(kl.is_infinite() && kl > 0.0, "Pareto-Normal alpha<=2 must be +inf, got {kl}");
}

#[test]
fn critic_kl_uniform_exponential_value() {
    // kl(Uniform(0,2), Exponential(1.5))
    let p = Uniform::new(sc(0.0), sc(2.0)).unwrap();
    let q = Exponential::new(sc(1.5)).unwrap();
    assert_kl!(p, q, 0.4013877113318902, "Uniform-Exponential");
}

#[test]
fn critic_kl_uniform_exponential_low_lt_0_is_inf() {
    let p = Uniform::new(sc(-1.0), sc(2.0)).unwrap();
    let q = Exponential::new(sc(1.5)).unwrap();
    let kl = kl_divergence(&p, &q).unwrap().item().unwrap();
    assert!(kl.is_infinite() && kl > 0.0, "Uniform-Exp low<0 must be +inf, got {kl}");
}

#[test]
fn critic_kl_uniform_gamma_value() {
    // kl(Uniform(0.5,2), Gamma(conc=2, rate=1))
    let p = Uniform::new(sc(0.5), sc(2.0)).unwrap();
    let q = Gamma::new(sc(2.0), sc(1.0)).unwrap();
    assert_kl!(p, q, 0.6892895909585934, "Uniform-Gamma");
}

#[test]
fn critic_kl_uniform_pareto_value() {
    // kl(Uniform(2,4), Pareto(scale=1, alpha=3))
    let p = Uniform::new(sc(2.0), sc(4.0)).unwrap();
    let q = Pareto::new(sc(1.0), sc(3.0)).unwrap();
    assert_kl!(p, q, 2.526006697491288, "Uniform-Pareto");
}

#[test]
fn critic_kl_uniform_pareto_low_lt_scale_is_inf() {
    // Uniform(0.5,4) extends below Pareto support lower bound (scale=1) -> +inf.
    let p = Uniform::new(sc(0.5), sc(4.0)).unwrap();
    let q = Pareto::new(sc(1.0), sc(3.0)).unwrap();
    let kl = kl_divergence(&p, &q).unwrap().item().unwrap();
    assert!(kl.is_infinite() && kl > 0.0, "Uniform-Pareto low<scale must be +inf, got {kl}");
}

#[test]
fn critic_kl_uniform_beta_value() {
    // kl(Uniform(0.2,0.8), Beta(2,3)) — support inside [0,1], finite.
    let p = Uniform::new(sc(0.2), sc(0.8)).unwrap();
    let q = Beta::new(sc(2.0), sc(3.0)).unwrap();
    assert_kl!(p, q, 0.3090552668007288, "Uniform-Beta");
}

#[test]
fn critic_kl_uniform_beta_support_escape_is_inf() {
    // Uniform(-0.1,1) escapes Beta support [0,1] (low<0) -> +inf.
    let p = Uniform::new(sc(-0.1), sc(1.0)).unwrap();
    let q = Beta::new(sc(2.0), sc(3.0)).unwrap();
    let kl = kl_divergence(&p, &q).unwrap().item().unwrap();
    assert!(kl.is_infinite() && kl > 0.0, "Uniform-Beta escape must be +inf, got {kl}");
}
