//! #1575 — generic `ExponentialFamily-ExponentialFamily` KL (Bregman fallback).
//!
//! Mirrors PyTorch's `_kl_expfamily_expfamily` (`torch/distributions/kl.py:282-300`),
//! the generic `@register_kl(ExponentialFamily, ExponentialFamily)` fallback.
//!
//! R-CHAR-3: every expected constant below was produced by **live-calling**
//! PyTorch 2.11 in float64 (NOT copied from the ferrotorch side). The
//! generating script is reproduced in each block's comment so the oracle is
//! auditable:
//! ```text
//! import torch; torch.set_default_dtype(torch.float64)
//! from torch.distributions import Normal, Gamma, Exponential, Beta, \
//!     Poisson, Bernoulli, kl_divergence
//! from torch.distributions.kl import _kl_expfamily_expfamily
//! # kl_divergence(p,q) (specific arm) and _kl_expfamily_expfamily(p,q)
//! # (the Bregman fallback) agree to ~1e-10 for every pair below.
//! ```
//!
//! Two correctness claims are pinned:
//!   1. The ferrotorch Bregman KL (built from analytic `mean_params`) equals
//!      torch's `_kl_expfamily_expfamily` value (which torch builds via autograd
//!      through `_log_normalizer`).
//!   2. The Bregman KL equals the specific-pair closed-form KL for the same
//!      pair (so the generic fallback would be a drop-in for the specific arm).

use ferrotorch_core::creation::{scalar, tensor};
use ferrotorch_distributions::kl::kl_divergence;
use ferrotorch_distributions::{
    Bernoulli, Beta, Exponential, Gamma, Normal, Poisson, kl_expfamily_expfamily,
};

fn sc(x: f64) -> ferrotorch_core::Tensor<f64> {
    scalar(x).unwrap()
}

/// Bregman exp-family KL == torch's `_kl_expfamily_expfamily` value AND ==
/// ferrotorch's specific-pair `kl_divergence` (which mirrors the same torch
/// closed form). One assertion pins all three to within 1e-9.
macro_rules! assert_bregman_eq_specific_eq_torch {
    ($p:expr, $q:expr, $torch:expr, $name:literal) => {{
        let bregman = kl_expfamily_expfamily(&$p, &$q).unwrap().item().unwrap();
        let specific = kl_divergence(&$p, &$q).unwrap().item().unwrap();
        let t: f64 = $torch;
        assert!(
            (bregman - t).abs() < 1e-9,
            "{}: Bregman KL = {bregman}, torch = {t} (delta {})",
            $name,
            (bregman - t).abs()
        );
        assert!(
            (specific - t).abs() < 1e-9,
            "{}: specific KL = {specific}, torch = {t} (delta {})",
            $name,
            (specific - t).abs()
        );
        assert!(
            (bregman - specific).abs() < 1e-9,
            "{}: Bregman {bregman} must equal specific {specific}",
            $name
        );
    }};
}

#[test]
fn bregman_eq_specific_eq_torch_normal() {
    // kl_divergence(Normal(0,1), Normal(1,2)) = 0.4431471805599453
    assert_bregman_eq_specific_eq_torch!(
        Normal::new(sc(0.0), sc(1.0)).unwrap(),
        Normal::new(sc(1.0), sc(2.0)).unwrap(),
        0.4431471805599453,
        "Normal(0,1)->Normal(1,2)"
    );
}

#[test]
fn bregman_eq_specific_eq_torch_gamma() {
    // kl_divergence(Gamma(2,1.5), Gamma(3,0.5)) = 2.2328663781324742
    assert_bregman_eq_specific_eq_torch!(
        Gamma::new(sc(2.0), sc(1.5)).unwrap(),
        Gamma::new(sc(3.0), sc(0.5)).unwrap(),
        2.2328663781324742,
        "Gamma(2,1.5)->Gamma(3,0.5)"
    );
}

#[test]
fn bregman_eq_specific_eq_torch_exponential() {
    // kl_divergence(Exponential(1.5), Exponential(0.5)) = 0.43194562200144304
    assert_bregman_eq_specific_eq_torch!(
        Exponential::new(sc(1.5)).unwrap(),
        Exponential::new(sc(0.5)).unwrap(),
        0.43194562200144304,
        "Exponential(1.5)->Exponential(0.5)"
    );
}

#[test]
fn bregman_eq_specific_eq_torch_beta() {
    // kl_divergence(Beta(2,3), Beta(1,1)) = 0.2349066497879999
    assert_bregman_eq_specific_eq_torch!(
        Beta::new(sc(2.0), sc(3.0)).unwrap(),
        Beta::new(sc(1.0), sc(1.0)).unwrap(),
        0.2349066497879999,
        "Beta(2,3)->Beta(1,1)"
    );
}

#[test]
fn bregman_eq_specific_eq_torch_poisson() {
    // kl_divergence(Poisson(2), Poisson(3.5)) = 0.38076842412915446
    assert_bregman_eq_specific_eq_torch!(
        Poisson::new(sc(2.0)).unwrap(),
        Poisson::new(sc(3.5)).unwrap(),
        0.38076842412915446,
        "Poisson(2)->Poisson(3.5)"
    );
}

#[test]
fn bregman_eq_specific_eq_torch_bernoulli() {
    // kl_divergence(Bernoulli(0.3), Bernoulli(0.6)) = 0.1837868973868122
    assert_bregman_eq_specific_eq_torch!(
        Bernoulli::new(sc(0.3)).unwrap(),
        Bernoulli::new(sc(0.6)).unwrap(),
        0.1837868973868122,
        "Bernoulli(0.3)->Bernoulli(0.6)"
    );
}

#[test]
fn bregman_broadcast_p_vs_q_gamma() {
    // torch: Gamma([2,3],[1,1.5]) -> Gamma([1.5],[0.5])
    //   = [0.1303307007539063, 0.2181655174546746]
    let p = Gamma::new(
        tensor(&[2.0f64, 3.0]).unwrap(),
        tensor(&[1.0f64, 1.5]).unwrap(),
    )
    .unwrap();
    let q = Gamma::new(tensor(&[1.5f64]).unwrap(), tensor(&[0.5f64]).unwrap()).unwrap();
    let kl = kl_expfamily_expfamily(&p, &q).unwrap();
    assert_eq!(kl.shape(), &[2]);
    let v = kl.data_vec().unwrap();
    assert!((v[0] - 0.1303307007539063).abs() < 1e-9, "got {}", v[0]);
    assert!((v[1] - 0.2181655174546746).abs() < 1e-9, "got {}", v[1]);
}
