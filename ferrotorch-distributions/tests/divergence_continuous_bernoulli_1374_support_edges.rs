//! Critic re-audit of commit c5cb6529d (#1374 CB sub-part) — support-edge
//! behavior: cdf clamping outside [0,1], log_prob extrapolation outside [0,1]
//! (ferrotorch does not validate, matching torch `validate_args=False`), and
//! the INCLUSIVE (ge/le) where-mask boundary at exactly low==0 / high==1.
//!
//! Expected values from live torch 2.11.0+cu130 (2026-05-27, validate_args=False);
//! traced to continuous_bernoulli.py:196-210 (cdf) / :187-194 (log_prob) and
//! kl.py:610-616 / :879-885 (where-mask). R-CHAR-3 non-tautological.

use ferrotorch_core::creation::scalar;
use ferrotorch_distributions::kl::kl_divergence;
use ferrotorch_distributions::{ContinuousBernoulli, Distribution, Uniform};

fn cb(p: f64) -> ContinuousBernoulli<f64> {
    ContinuousBernoulli::new(scalar(p).unwrap()).unwrap()
}
fn u(lo: f64, hi: f64) -> Uniform<f64> {
    Uniform::new(scalar(lo).unwrap(), scalar(hi).unwrap()).unwrap()
}

fn close(a: f64, b: f64, ctx: &str) {
    assert!(
        (a - b).abs() <= 1e-12,
        "{ctx}: ferrotorch={a:?} torch={b:?}"
    );
}

#[test]
fn divergence_cb_log_prob_outside_support() {
    // torch CB(0.3,validate_args=False).log_prob extrapolates the density:
    //   v=-0.5 -> 0.8175617371353283; v=1.5 -> -0.8770339836390793;
    //   v=2.0  -> -1.300682913832681.
    let d = cb(0.3);
    let cases = [
        (-0.5, 0.817_561_737_135_328_3),
        (1.5, -0.877_033_983_639_079_3),
        (2.0, -1.300_682_913_832_681),
    ];
    for (v, want) in cases {
        let got = d.log_prob(&scalar(v).unwrap()).unwrap().item().unwrap();
        close(got, want, &format!("log_prob(0.3,{v})"));
    }
}

#[test]
fn divergence_cb_cdf_clamps_outside_support() {
    // torch CB(0.3).cdf(-0.5)==0.0, cdf(1.5)==1.0 (clamped at the support ends).
    let d = cb(0.3);
    close(
        d.cdf(&scalar(-0.5f64).unwrap()).unwrap().item().unwrap(),
        0.0,
        "cdf(0.3,-0.5)",
    );
    close(
        d.cdf(&scalar(1.5f64).unwrap()).unwrap().item().unwrap(),
        1.0,
        "cdf(0.3,1.5)",
    );
}

#[test]
fn divergence_kl_cb_uniform_inclusive_boundary() {
    // INCLUSIVE: low==0 -> torch.ge(low,0)=True -> +inf; high==1 -> le(high,1)=True -> +inf.
    let lo0 = kl_divergence(&cb(0.4), &u(0.0, 2.0))
        .unwrap()
        .item()
        .unwrap();
    assert!(
        lo0.is_infinite() && lo0 > 0.0,
        "KL(CB,U(0,2)) low==0 must be +inf, got {lo0}"
    );
    let hi1 = kl_divergence(&cb(0.4), &u(-1.0, 1.0))
        .unwrap()
        .item()
        .unwrap();
    assert!(
        hi1.is_infinite() && hi1 > 0.0,
        "KL(CB,U(-1,1)) high==1 must be +inf, got {hi1}"
    );
}

#[test]
fn divergence_kl_uniform_cb_inclusive_boundary() {
    // INCLUSIVE: low==0 -> torch.le(low,0)=True -> +inf; high==1 -> ge(high,1)=True -> +inf.
    let lo0 = kl_divergence(&u(0.0, 0.8), &cb(0.4))
        .unwrap()
        .item()
        .unwrap();
    assert!(
        lo0.is_infinite() && lo0 > 0.0,
        "KL(U(0,0.8),CB) low==0 must be +inf, got {lo0}"
    );
    let hi1 = kl_divergence(&u(0.2, 1.0), &cb(0.4))
        .unwrap()
        .item()
        .unwrap();
    assert!(
        hi1.is_infinite() && hi1 > 0.0,
        "KL(U(0.2,1),CB) high==1 must be +inf, got {hi1}"
    );
}
