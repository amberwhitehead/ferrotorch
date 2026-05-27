//! Wave-K audit (#1542): the three distributions infrastructure deliverables
//! landed under #1427 / #1378 / #1374.
//!
//!   - #1427: `StudentT::rsample` now propagates a gradient to `df` via the
//!     pathwise Chi2 reparameterisation gradient `standard_gamma_grad_one`.
//!   - #1378: `TransformedDistribution::entropy` Monte-Carlo fallback so
//!     Sigmoid/Tanh/Softplus chains return a value instead of erroring.
//!   - #1374: 7 new KL pairs (Beta-Beta, Gumbel-Gumbel, Pareto-Pareto,
//!     HalfNormal-HalfNormal, Exponential-Normal, Gamma-Normal, Laplace-Normal).
//!
//! Reference values are constructed from PyTorch's closed-form `@register_kl`
//! bodies (`torch/distributions/kl.py`) and `studentT.py` so the asserts are
//! non-tautological (R-CHAR-3): each expected number traces to an upstream
//! file:line, an independent quadrature oracle, or a known analytic identity.

#![allow(clippy::approx_constant)]

use ferrotorch_core::creation::scalar;
use ferrotorch_distributions::kl::kl_divergence;
use ferrotorch_distributions::transforms::{
    AffineTransform, ExpTransform, SigmoidTransform, TanhTransform,
};
use ferrotorch_distributions::{
    Beta, Distribution, Exponential, Gamma, Gumbel, HalfNormal, Laplace, Normal, Pareto, StudentT,
    TransformedDistribution,
};

// ---------------------------------------------------------------------------
// #1427 — StudentT df gradient
// ---------------------------------------------------------------------------

/// The df gradient now attaches and backpropagates to a finite value. Prior to
/// #1427 the df slot was always `None` and rsample only attached a node when
/// loc/scale required grad.
#[test]
fn audit_1427_student_t_df_gradient_attaches_and_is_finite() {
    let df = scalar(5.0f64).unwrap().requires_grad_(true);
    let loc = scalar(0.0f64).unwrap();
    let scale = scalar(1.0f64).unwrap();
    let dist = StudentT::new(df.clone(), loc, scale).unwrap();

    let s = dist.rsample(&[16]).unwrap();
    assert!(
        s.requires_grad(),
        "rsample must attach a node when df requires grad"
    );
    s.sum_all().unwrap().backward().unwrap();

    let g = df
        .grad()
        .unwrap()
        .expect("df gradient must be populated post-#1427");
    assert!(
        g.item().unwrap().is_finite(),
        "df grad must be finite, got {}",
        g.item().unwrap()
    );
}

/// End-to-end FD sanity: the mean of many StudentT rsamples increases with df
/// for df < ~ where the scale-shrinking from heavier tails dominates is not
/// asserted; instead we FD-check the *expected sample magnitude* sensitivity to
/// df. As df grows the Chi2/df ratio concentrates at 1, so the t-sample
/// magnitude `|z|·sqrt(df/chi2)` shrinks toward `|z|`. We confirm the gradient
/// of `E[t^2]` w.r.t. df is negative (heavier tails at small df → larger
/// second moment), matching `Var = df/(df-2)` which decreases in df.
#[test]
fn audit_1427_df_gradient_sign_matches_variance_monotonicity() {
    // Var(StudentT(df)) = df/(df-2) is strictly decreasing in df for df>2,
    // so d/d(df) E[t^2] < 0. The backward of a t^2 loss must give negative
    // df gradient on average. Use a large sample to average out MC noise.
    let df = scalar(6.0f64).unwrap().requires_grad_(true);
    let loc = scalar(0.0f64).unwrap();
    let scale = scalar(1.0f64).unwrap();
    let dist = StudentT::new(df.clone(), loc, scale).unwrap();

    let s = dist.rsample(&[40_000]).unwrap();
    let sq = ferrotorch_core::grad_fns::arithmetic::mul(&s, &s).unwrap();
    sq.sum_all().unwrap().backward().unwrap();

    let g = df.grad().unwrap().unwrap().item().unwrap() / 40_000.0;
    assert!(
        g < 0.0,
        "d/d(df) E[t^2] should be negative (variance decreases in df), got {g}"
    );
}

// ---------------------------------------------------------------------------
// #1378 — TransformedDistribution entropy MC fallback
// ---------------------------------------------------------------------------

/// Numerical reference for `E_{X~Normal(0,1)}[ g(X) ]` via dense trapezoidal
/// quadrature — an independent oracle for the MC entropy estimator.
fn normal_expectation_quadrature(g: impl Fn(f64) -> f64) -> f64 {
    let (lo, hi, steps) = (-12.0_f64, 12.0_f64, 200_000usize);
    let dx = (hi - lo) / steps as f64;
    let norm = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
    let mut acc = 0.0;
    for i in 0..=steps {
        let x = lo + i as f64 * dx;
        let w = if i == 0 || i == steps { 0.5 } else { 1.0 };
        acc += w * norm * (-0.5 * x * x).exp() * g(x) * dx;
    }
    acc
}

#[test]
fn audit_1378_sigmoid_entropy_no_longer_errors_and_matches_quadrature() {
    let base = Normal::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
    let td = TransformedDistribution::new(Box::new(base), vec![Box::new(SigmoidTransform)]);
    let got = td
        .entropy()
        .expect("sigmoid entropy must return a value via MC fallback (#1378)");

    let base_ent = Normal::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap())
        .unwrap()
        .entropy()
        .unwrap()
        .item()
        .unwrap();
    // log|sigma'(x)| = -softplus(-x) - softplus(x).
    let softplus = |z: f64| (1.0 + z.exp()).ln();
    let contrib = normal_expectation_quadrature(|x| -softplus(-x) - softplus(x));
    let expected = base_ent + contrib;
    assert!(
        (got.item().unwrap() - expected).abs() < 3e-2,
        "sigmoid MC entropy {} vs quadrature {expected}",
        got.item().unwrap()
    );
}

#[test]
fn audit_1378_tanh_entropy_no_longer_errors_and_matches_quadrature() {
    let base = Normal::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
    let td = TransformedDistribution::new(Box::new(base), vec![Box::new(TanhTransform)]);
    let got = td
        .entropy()
        .expect("tanh entropy must return a value via MC fallback (#1378)");

    let base_ent = Normal::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap())
        .unwrap()
        .entropy()
        .unwrap()
        .item()
        .unwrap();
    // log|tanh'(x)| = log(1 - tanh(x)^2).
    let contrib = normal_expectation_quadrature(|x| (1.0 - x.tanh().powi(2)).ln());
    let expected = base_ent + contrib;
    assert!(
        (got.item().unwrap() - expected).abs() < 3e-2,
        "tanh MC entropy {} vs quadrature {expected}",
        got.item().unwrap()
    );
}

#[test]
fn audit_1378_exp_then_affine_entropy_matches_closed_form() {
    // [Exp, Affine(0,2)] over Normal(0,1): contribution E[x] + log2 = ln 2.
    let base = Normal::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
    let td = TransformedDistribution::new(
        Box::new(base),
        vec![
            Box::new(ExpTransform),
            Box::new(AffineTransform::new(0.0f64, 2.0)),
        ],
    );
    let got = td
        .entropy()
        .expect("exp-then-affine entropy via MC fallback (#1378)");
    let base_ent = Normal::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap())
        .unwrap()
        .entropy()
        .unwrap()
        .item()
        .unwrap();
    let expected = base_ent + 2.0_f64.ln();
    assert!(
        (got.item().unwrap() - expected).abs() < 3e-2,
        "exp-then-affine MC entropy {} vs reference {expected}",
        got.item().unwrap()
    );
}

// ---------------------------------------------------------------------------
// #1374 — new KL pairs (reference values from torch/distributions/kl.py)
// ---------------------------------------------------------------------------

#[test]
fn audit_1374_beta_beta_known_value() {
    // KL(Beta(2,3) || Beta(3,2)) = ψ(3) - ψ(2) = 1/2 (kl.py:219-228).
    let p = Beta::new(scalar(2.0f64).unwrap(), scalar(3.0f64).unwrap()).unwrap();
    let q = Beta::new(scalar(3.0f64).unwrap(), scalar(2.0f64).unwrap()).unwrap();
    let kl = kl_divergence(&p, &q).unwrap().item().unwrap();
    assert!((kl - 0.5).abs() < 1e-9, "expected 0.5, got {kl}");
}

#[test]
fn audit_1374_gumbel_gumbel_same_is_zero() {
    let p = Gumbel::new(scalar(1.0f64).unwrap(), scalar(2.0f64).unwrap()).unwrap();
    let q = Gumbel::new(scalar(1.0f64).unwrap(), scalar(2.0f64).unwrap()).unwrap();
    let kl = kl_divergence(&p, &q).unwrap().item().unwrap();
    assert!(kl.abs() < 1e-9, "KL(same) must be 0, got {kl}");
}

#[test]
fn audit_1374_pareto_pareto_known_value_and_support() {
    // KL(Pareto(1,4) || Pareto(1,2)) = ln 2 - 0.5 (kl.py:479-488).
    let p = Pareto::new(scalar(1.0f64).unwrap(), scalar(4.0f64).unwrap()).unwrap();
    let q = Pareto::new(scalar(1.0f64).unwrap(), scalar(2.0f64).unwrap()).unwrap();
    let kl = kl_divergence(&p, &q).unwrap().item().unwrap();
    assert!((kl - (2.0_f64.ln() - 0.5)).abs() < 1e-10, "got {kl}");

    // p.scale < q.scale → +inf (kl.py:487).
    let p2 = Pareto::new(scalar(1.0f64).unwrap(), scalar(3.0f64).unwrap()).unwrap();
    let q2 = Pareto::new(scalar(2.0f64).unwrap(), scalar(3.0f64).unwrap()).unwrap();
    assert!(
        kl_divergence(&p2, &q2)
            .unwrap()
            .item()
            .unwrap()
            .is_infinite()
    );
}

#[test]
fn audit_1374_halfnormal_matches_zero_mean_normal_normal() {
    // _kl_halfnormal_halfnormal delegates to Normal-Normal(loc=0) (kl.py:325-327).
    let p = HalfNormal::new(scalar(1.0f64).unwrap()).unwrap();
    let q = HalfNormal::new(scalar(2.0f64).unwrap()).unwrap();
    let hn = kl_divergence(&p, &q).unwrap().item().unwrap();
    let pn = Normal::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
    let qn = Normal::new(scalar(0.0f64).unwrap(), scalar(2.0f64).unwrap()).unwrap();
    let nn = kl_divergence(&pn, &qn).unwrap().item().unwrap();
    assert!(
        (hn - nn).abs() < 1e-12,
        "HalfNormal KL {hn} vs Normal-Normal {nn}"
    );
}

#[test]
fn audit_1374_exponential_normal_known_value() {
    // KL(Exp(1) || Normal(0,1)) = 0.5·ln(2π) (kl.py:654-662).
    let p = Exponential::new(scalar(1.0f64).unwrap()).unwrap();
    let q = Normal::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
    let kl = kl_divergence(&p, &q).unwrap().item().unwrap();
    let expected = 0.5 * (2.0 * std::f64::consts::PI).ln();
    assert!(
        (kl - expected).abs() < 1e-12,
        "expected {expected}, got {kl}"
    );
}

#[test]
fn audit_1374_gamma_normal_reduces_to_exponential_normal() {
    // Gamma(1,1) == Exp(1), so KL(Gamma(1,1)||Normal(0,1)) = 0.5·ln(2π)
    // (kl.py:699-715 reduces to kl.py:654-662 at α=1).
    let p = Gamma::new(scalar(1.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
    let q = Normal::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
    let kl = kl_divergence(&p, &q).unwrap().item().unwrap();
    let expected = 0.5 * (2.0 * std::f64::consts::PI).ln();
    assert!(
        (kl - expected).abs() < 1e-9,
        "expected {expected}, got {kl}"
    );
}

#[test]
fn audit_1374_laplace_normal_known_value() {
    // KL(Laplace(0,1) || Normal(0,1)) = 0.5·ln(π/2) (kl.py:750-758).
    let p = Laplace::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
    let q = Normal::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
    let kl = kl_divergence(&p, &q).unwrap().item().unwrap();
    let expected = 0.5 * (std::f64::consts::PI / 2.0).ln();
    assert!(
        (kl - expected).abs() < 1e-12,
        "expected {expected}, got {kl}"
    );
}
