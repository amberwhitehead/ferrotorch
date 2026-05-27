//! Critic (acto-critic) independent re-audit of commit `3951534d0`
//! (ExponentialFamily Bregman KL, #1575 / #1374 closure).
//!
//! These tests are an INDEPENDENT verification gate, not a copy of the
//! builder's `exp_family.rs::tests`. The builder's tests only assert the final
//! Bregman KL equals torch; that check can be passed by a WRONG analytic
//! `mean_params` whose error cancels for the chosen pair. The crux of #1374's
//! correctness is that `mean_params()` IS the true `∇A(η)` — the gradient of
//! the distribution's own `log_normalizer` wrt its natural params.
//!
//! Strategy (mirrors the audit charter):
//!   1. FINITE DIFFERENCE on ferrotorch's own `log_normalizer` (central
//!      difference, perturbing each natural param) and assert it equals the
//!      analytic `mean_params()` to ~1e-5. This is the gradient-correctness
//!      check that catches a wrong closed form even when the KL still matches.
//!   2. ASYMMETRIC-pair Bregman KL == live-torch `kl_divergence` /
//!      `_kl_expfamily_expfamily`. Asymmetric (different params each side)
//!      exposes gradient-sign errors that a symmetric P==Q→0 pair hides.
//!      Oracle constants produced by live torch 2.11 f64 (R-CHAR-3) — see the
//!      reproduction block at the bottom of this file; NOT copied from the
//!      ferrotorch side.
//!   3. Cross-family `try_kl_expfamily` must return None (Poisson vs Bernoulli
//!      both have 1 natural param — a len-only guard would mis-fire).
//!   4. Full `kl_divergence` route shadows the fallback with the specific arm.
//!
//! Upstream sites mirrored:
//!   - `torch/distributions/exp_family.py:11-66` (ExponentialFamily base)
//!   - `torch/distributions/kl.py:282-297` (_kl_expfamily_expfamily)
//!   - per-distribution `_natural_params`/`_log_normalizer` (cited inline)
//!
//! All tests are EXPECTED TO PASS — they document that NO divergence was found
//! in the analytic gradients. A failure here means a `mean_params` is a wrong
//! gradient, which BLOCKS #1374's closure (orchestrator must reopen).

// The asymmetric-pair oracle table pairs a torch-sourced f64 with a boxed
// closure; the boxed-fn-tuple-slice type is intentional (one driver loop). The
// f64 oracle constants are copied verbatim from live torch (R-CHAR-3); trailing
// digits must stay so the value is traceable, even where the last digit is a 0.
#![allow(clippy::type_complexity, clippy::excessive_precision)]

use ferrotorch_core::creation::{scalar, tensor};
use ferrotorch_core::tensor::Tensor;
use ferrotorch_distributions::exp_family::{kl_expfamily_expfamily, try_kl_expfamily};
use ferrotorch_distributions::{
    Bernoulli, Beta, Distribution, Exponential, ExponentialFamily, Gamma, Normal, Poisson,
    kl::kl_divergence,
};

fn sc(x: f64) -> Tensor<f64> {
    scalar(x).unwrap()
}

/// Central finite-difference gradient of a distribution's own `log_normalizer`
/// wrt each natural param, evaluated at the distribution's own η. This is the
/// TRUE `∇A(η) = E[t(X)]`, computed WITHOUT reading `mean_params()`.
fn fd_grad_of_log_normalizer<D: ExponentialFamily<f64>>(d: &D, h: f64) -> Vec<f64> {
    let nps = d.natural_params().unwrap();
    let n = nps.len();
    let base: Vec<f64> = nps.iter().map(|t| t.item().unwrap()).collect();
    let mut grads = Vec::with_capacity(n);
    for i in 0..n {
        let mut up = base.clone();
        let mut dn = base.clone();
        up[i] += h;
        dn[i] -= h;
        let up_t: Vec<Tensor<f64>> = up.iter().map(|&v| scalar(v).unwrap()).collect();
        let dn_t: Vec<Tensor<f64>> = dn.iter().map(|&v| scalar(v).unwrap()).collect();
        let a_up = d.log_normalizer(&up_t).unwrap().item().unwrap();
        let a_dn = d.log_normalizer(&dn_t).unwrap().item().unwrap();
        grads.push((a_up - a_dn) / (2.0 * h));
    }
    grads
}

/// Assert `mean_params()` equals the finite-difference gradient of
/// `log_normalizer` to a tolerance the central-difference truncation allows.
fn assert_meanparams_is_grad<D: ExponentialFamily<f64>>(d: &D, h: f64, tol: f64, name: &str) {
    let analytic: Vec<f64> = d
        .mean_params()
        .unwrap()
        .iter()
        .map(|t| t.item().unwrap())
        .collect();
    let fd = fd_grad_of_log_normalizer(d, h);
    assert_eq!(
        analytic.len(),
        fd.len(),
        "{name}: mean_params arity {} != natural_params arity {}",
        analytic.len(),
        fd.len()
    );
    for (i, (&a, &g)) in analytic.iter().zip(fd.iter()).enumerate() {
        assert!(
            (a - g).abs() < tol,
            "{name}: mean_params[{i}] = {a} but FD(log_normalizer)[{i}] = {g} \
             (delta {}); analytic gradient diverges from ∇A(η)",
            (a - g).abs()
        );
    }
}

// ---------------------------------------------------------------------------
// (1) FD(log_normalizer) == analytic mean_params, per distribution.
//     h chosen per-family so central-difference truncation stays < tol.
// ---------------------------------------------------------------------------

#[test]
fn meanparams_is_grad_normal() {
    // normal.py:116-122: η=(μ/σ², -0.5/σ²); A=-0.25x²/y+0.5 log(-π/y).
    assert_meanparams_is_grad(
        &Normal::new(sc(0.5), sc(2.0)).unwrap(),
        1e-6,
        1e-4,
        "Normal(0.5,2)",
    );
    assert_meanparams_is_grad(
        &Normal::new(sc(-0.3), sc(0.7)).unwrap(),
        1e-6,
        1e-4,
        "Normal(-0.3,0.7)",
    );
}

#[test]
fn meanparams_is_grad_exponential() {
    // exponential.py:88-94: η=(-rate); A=-log(-x).
    assert_meanparams_is_grad(&Exponential::new(sc(0.5)).unwrap(), 1e-6, 1e-5, "Exp(0.5)");
    assert_meanparams_is_grad(&Exponential::new(sc(2.0)).unwrap(), 1e-6, 1e-5, "Exp(2.0)");
}

#[test]
fn meanparams_is_grad_gamma() {
    // gamma.py:109-114: η=(conc-1, -rate); A=lgamma(x+1)+(x+1)log(-1/y).
    assert_meanparams_is_grad(
        &Gamma::new(sc(3.0), sc(1.0)).unwrap(),
        1e-6,
        1e-4,
        "Gamma(3,1)",
    );
    assert_meanparams_is_grad(
        &Gamma::new(sc(2.0), sc(3.0)).unwrap(),
        1e-6,
        1e-4,
        "Gamma(2,3)",
    );
}

#[test]
fn meanparams_is_grad_beta() {
    // beta.py:112-118: η=(c1, c0); A=lgamma(x)+lgamma(y)-lgamma(x+y).
    assert_meanparams_is_grad(
        &Beta::new(sc(3.0), sc(2.0)).unwrap(),
        1e-6,
        1e-4,
        "Beta(3,2)",
    );
    assert_meanparams_is_grad(
        &Beta::new(sc(2.0), sc(5.0)).unwrap(),
        1e-6,
        1e-4,
        "Beta(2,5)",
    );
}

#[test]
fn meanparams_is_grad_poisson() {
    // poisson.py:81-87: η=(log rate); A=exp(x).
    assert_meanparams_is_grad(&Poisson::new(sc(3.5)).unwrap(), 1e-6, 1e-4, "Poisson(3.5)");
    assert_meanparams_is_grad(&Poisson::new(sc(2.0)).unwrap(), 1e-6, 1e-4, "Poisson(2.0)");
}

#[test]
fn meanparams_is_grad_bernoulli() {
    // bernoulli.py:139-145: η=(logit(p)); A=log1p(exp(x)).
    assert_meanparams_is_grad(
        &Bernoulli::new(sc(0.6)).unwrap(),
        1e-6,
        1e-5,
        "Bernoulli(0.6)",
    );
    assert_meanparams_is_grad(
        &Bernoulli::new(sc(0.3)).unwrap(),
        1e-6,
        1e-5,
        "Bernoulli(0.3)",
    );
}

// ---------------------------------------------------------------------------
// (2) ASYMMETRIC-pair Bregman KL == live torch kl_divergence.
//     Oracle constants from live torch 2.11 f64 (reproduction at file end).
// ---------------------------------------------------------------------------

fn bregman(p: &dyn ExponentialFamily<f64>, q: &dyn ExponentialFamily<f64>) -> f64 {
    kl_expfamily_expfamily(p, q).unwrap().item().unwrap()
}

#[test]
fn bregman_asymmetric_matches_torch() {
    // torch kl_divergence (== _kl_expfamily_expfamily, both to 1e-16):
    let cases: &[(&str, f64, Box<dyn Fn() -> f64>)] = &[
        (
            "Normal(0,1)||Normal(0.5,2)",
            0.3493971805599453,
            Box::new(|| {
                bregman(
                    &Normal::new(sc(0.0), sc(1.0)).unwrap(),
                    &Normal::new(sc(0.5), sc(2.0)).unwrap(),
                )
            }),
        ),
        (
            "Gamma(2,3)||Gamma(3,1)",
            2.2328663781324742,
            Box::new(|| {
                bregman(
                    &Gamma::new(sc(2.0), sc(3.0)).unwrap(),
                    &Gamma::new(sc(3.0), sc(1.0)).unwrap(),
                )
            }),
        ),
        (
            "Beta(2,5)||Beta(3,2)",
            1.2662907318741548,
            Box::new(|| {
                bregman(
                    &Beta::new(sc(2.0), sc(5.0)).unwrap(),
                    &Beta::new(sc(3.0), sc(2.0)).unwrap(),
                )
            }),
        ),
        (
            "Exp(1.5)||Exp(0.5)",
            0.4319456220014430,
            Box::new(|| {
                bregman(
                    &Exponential::new(sc(1.5)).unwrap(),
                    &Exponential::new(sc(0.5)).unwrap(),
                )
            }),
        ),
        (
            "Poisson(2)||Poisson(3.5)",
            0.3807684241291545,
            Box::new(|| {
                bregman(
                    &Poisson::new(sc(2.0)).unwrap(),
                    &Poisson::new(sc(3.5)).unwrap(),
                )
            }),
        ),
        (
            "Bernoulli(0.3)||Bernoulli(0.6)",
            0.1837868973868122,
            Box::new(|| {
                bregman(
                    &Bernoulli::new(sc(0.3)).unwrap(),
                    &Bernoulli::new(sc(0.6)).unwrap(),
                )
            }),
        ),
    ];
    for (name, torch, f) in cases {
        let got = f();
        assert!(
            (got - torch).abs() < 1e-9,
            "{name}: Bregman KL = {got}, torch = {torch} (delta {})",
            (got - torch).abs()
        );
    }
}

// ---------------------------------------------------------------------------
// (2b) Bregman KL == ferrotorch's specific same-family arm (kl_divergence),
//      for the SAME asymmetric pairs. The specific arm shadows the fallback,
//      so this confirms the two independent code paths agree.
// ---------------------------------------------------------------------------

#[test]
fn bregman_equals_specific_arm_asymmetric() {
    macro_rules! check {
        ($p:expr, $q:expr, $name:literal) => {{
            let p = $p;
            let q = $q;
            let breg = kl_expfamily_expfamily(&p, &q).unwrap().item().unwrap();
            let spec = kl_divergence(&p, &q).unwrap().item().unwrap();
            assert!(
                (breg - spec).abs() < 1e-9,
                "{}: Bregman {} != specific kl_divergence {} (delta {})",
                $name,
                breg,
                spec,
                (breg - spec).abs()
            );
        }};
    }
    check!(
        Normal::new(sc(0.0), sc(1.0)).unwrap(),
        Normal::new(sc(0.5), sc(2.0)).unwrap(),
        "Normal asym"
    );
    check!(
        Gamma::new(sc(2.0), sc(3.0)).unwrap(),
        Gamma::new(sc(3.0), sc(1.0)).unwrap(),
        "Gamma asym"
    );
    check!(
        Beta::new(sc(2.0), sc(5.0)).unwrap(),
        Beta::new(sc(3.0), sc(2.0)).unwrap(),
        "Beta asym"
    );
    check!(
        Exponential::new(sc(1.5)).unwrap(),
        Exponential::new(sc(0.5)).unwrap(),
        "Exp asym"
    );
    check!(
        Poisson::new(sc(2.0)).unwrap(),
        Poisson::new(sc(3.5)).unwrap(),
        "Poisson asym"
    );
    check!(
        Bernoulli::new(sc(0.3)).unwrap(),
        Bernoulli::new(sc(0.6)).unwrap(),
        "Bernoulli asym"
    );
}

// ---------------------------------------------------------------------------
// (3) Cross-family dispatch must NOT mis-fire. Poisson and Bernoulli BOTH have
//     exactly 1 natural param; a len-based guard inside kl_expfamily_expfamily
//     would happily compute garbage. try_kl_expfamily must return None so the
//     specific Poisson-Bernoulli arm (+inf, kl.py:841) wins.
// ---------------------------------------------------------------------------

#[test]
fn cross_family_same_arity_returns_none() {
    let pois: Poisson<f64> = Poisson::new(sc(2.0)).unwrap();
    let bern: Bernoulli<f64> = Bernoulli::new(sc(0.3)).unwrap();
    // Both single-param exp families, but different types -> None.
    let r = try_kl_expfamily::<f64>(
        &pois as &dyn Distribution<f64>,
        &bern as &dyn Distribution<f64>,
    )
    .unwrap();
    assert!(
        r.is_none(),
        "try_kl_expfamily(Poisson, Bernoulli) must return None (cross-family, \
         mirrors kl.py:284 NotImplementedError) — got Some(..)"
    );
    // Normal vs Gamma: both 2-param exp families, different types -> None.
    let n: Normal<f64> = Normal::new(sc(0.0), sc(1.0)).unwrap();
    let g: Gamma<f64> = Gamma::new(sc(2.0), sc(1.0)).unwrap();
    let r2 = try_kl_expfamily::<f64>(&n as &dyn Distribution<f64>, &g as &dyn Distribution<f64>)
        .unwrap();
    assert!(
        r2.is_none(),
        "try_kl_expfamily(Normal, Gamma) must return None (cross-family)"
    );
}

#[test]
fn cross_family_kl_divergence_is_specific_not_bregman() {
    // kl_divergence(Poisson(2)||Bernoulli(0.3)) == +inf (torch: support
    // mismatch arm, kl.py:841), NOT some finite garbage Bregman value.
    let pois: Poisson<f64> = Poisson::new(sc(2.0)).unwrap();
    let bern: Bernoulli<f64> = Bernoulli::new(sc(0.3)).unwrap();
    let kl = kl_divergence(&pois, &bern).unwrap().item().unwrap();
    assert!(
        kl.is_infinite() && kl > 0.0,
        "kl_divergence(Poisson||Bernoulli) must be +inf (specific arm), got {kl}"
    );
}

// ---------------------------------------------------------------------------
// (4) Broadcast p-vs-q in the Bregman fallback (p shape [2] vs q shape [1]
//     and the reverse). Oracle from live torch (reproduction at file end).
// ---------------------------------------------------------------------------

#[test]
fn bregman_broadcast_p2_q1_matches_torch() {
    // torch: Normal([0,1],[1,2]) || Normal([0.5],[1.5]) =
    //   [0.18324288588594217, 0.15676237199266352]
    let p = Normal::new(
        tensor(&[0.0f64, 1.0]).unwrap(),
        tensor(&[1.0f64, 2.0]).unwrap(),
    )
    .unwrap();
    let q = Normal::new(tensor(&[0.5f64]).unwrap(), tensor(&[1.5f64]).unwrap()).unwrap();
    let kl = kl_expfamily_expfamily(&p, &q).unwrap();
    assert_eq!(kl.shape(), &[2]);
    let v = kl.data_vec().unwrap();
    assert!((v[0] - 0.18324288588594217).abs() < 1e-9, "got {}", v[0]);
    assert!((v[1] - 0.15676237199266352).abs() < 1e-9, "got {}", v[1]);

    // Reverse broadcast: p shape [1] vs q shape [2].
    // torch: Normal([0.5],[1.5]) || Normal([0,1],[1,2]) =
    //   [0.3445348918918356, 0.1001820724517809]
    let p2 = Normal::new(tensor(&[0.5f64]).unwrap(), tensor(&[1.5f64]).unwrap()).unwrap();
    let q2 = Normal::new(
        tensor(&[0.0f64, 1.0]).unwrap(),
        tensor(&[1.0f64, 2.0]).unwrap(),
    )
    .unwrap();
    let kl2 = kl_expfamily_expfamily(&p2, &q2).unwrap();
    assert_eq!(kl2.shape(), &[2]);
    let v2 = kl2.data_vec().unwrap();
    assert!((v2[0] - 0.3445348918918356).abs() < 1e-9, "got {}", v2[0]);
    assert!((v2[1] - 0.1001820724517809).abs() < 1e-9, "got {}", v2[1]);
}

// ---------------------------------------------------------------------------
// Live-torch oracle reproduction (PyTorch 2.11, float64):
//
//   import torch; torch.set_default_dtype(torch.float64)
//   from torch.distributions import Normal, Gamma, Exponential, Beta, \
//       Poisson, Bernoulli, kl_divergence
//   t = lambda x: torch.tensor(float(x))
//   # (2) asymmetric pairs
//   kl_divergence(Normal(t(0),t(1)), Normal(t(.5),t(2)))  # 0.3493971805599453
//   kl_divergence(Gamma(t(2),t(3)),  Gamma(t(3),t(1)))    # 2.2328663781324742
//   kl_divergence(Beta(t(2),t(5)),   Beta(t(3),t(2)))     # 1.2662907318741548
//   kl_divergence(Exponential(t(1.5)),Exponential(t(.5))) # 0.4319456220014430
//   kl_divergence(Poisson(t(2)),     Poisson(t(3.5)))     # 0.3807684241291545
//   kl_divergence(Bernoulli(t(.3)),  Bernoulli(t(.6)))    # 0.1837868973868122
//   # (4) broadcast
//   kl_divergence(Normal(torch.tensor([0.,1.]),torch.tensor([1.,2.])),
//                 Normal(torch.tensor([0.5]), torch.tensor([1.5])))
//     # -> [0.18324288588594217, 0.15676237199266352]
//   kl_divergence(Normal(torch.tensor([0.5]), torch.tensor([1.5])),
//                 Normal(torch.tensor([0.,1.]),torch.tensor([1.,2.])))
//     # -> [0.3445348918918356, 0.1001820724517809]
// ---------------------------------------------------------------------------
