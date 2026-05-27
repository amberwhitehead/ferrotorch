//! Critic re-audit of commit c5cb6529d (#1374 CB sub-part) — the 7 `+inf`
//! support-mismatch KL pairs + the `from_logits` / f32 edges.
//!
//! Expected values from live torch 2.11.0+cu130 (2026-05-27); +inf
//! classifications confirmed against the `_infinite_like` registrations in
//! `torch/distributions/kl.py` (CB-Pareto :581; {Exp,Gamma,Gumbel,Laplace,
//! Normal,Pareto}-CB :621,666,719,741,762,796). R-CHAR-3 non-tautological.

use ferrotorch_distributions::kl::kl_divergence;
use ferrotorch_distributions::{
    ContinuousBernoulli, Distribution, Exponential, Gamma, Gumbel, Laplace, Normal, Pareto,
};
use ferrotorch_core::creation::scalar;

fn cb(p: f64) -> ContinuousBernoulli<f64> {
    ContinuousBernoulli::new(scalar(p).unwrap()).unwrap()
}

fn assert_pos_inf(v: f64, ctx: &str) {
    assert!(v.is_infinite() && v > 0.0, "{ctx}: expected +inf, got {v}");
}

#[test]
fn divergence_kl_cb_seven_infinite_pairs() {
    // CB-Pareto (kl.py:581-583).
    assert_pos_inf(
        kl_divergence(
            &cb(0.4),
            &Pareto::new(scalar(1.0f64).unwrap(), scalar(2.0f64).unwrap()).unwrap(),
        )
        .unwrap()
        .item()
        .unwrap(),
        "KL(CB(0.4),Pareto(1,2))",
    );
    // Exponential-CB (kl.py:621).
    assert_pos_inf(
        kl_divergence(&Exponential::new(scalar(1.5f64).unwrap()).unwrap(), &cb(0.4))
            .unwrap()
            .item()
            .unwrap(),
        "KL(Exp(1.5),CB(0.4))",
    );
    // Gamma-CB (kl.py:666).
    assert_pos_inf(
        kl_divergence(
            &Gamma::new(scalar(2.0f64).unwrap(), scalar(3.0f64).unwrap()).unwrap(),
            &cb(0.4),
        )
        .unwrap()
        .item()
        .unwrap(),
        "KL(Gamma(2,3),CB(0.4))",
    );
    // Gumbel-CB (kl.py:719).
    assert_pos_inf(
        kl_divergence(
            &Gumbel::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap(),
            &cb(0.4),
        )
        .unwrap()
        .item()
        .unwrap(),
        "KL(Gumbel(0,1),CB(0.4))",
    );
    // Laplace-CB (kl.py:741).
    assert_pos_inf(
        kl_divergence(
            &Laplace::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap(),
            &cb(0.4),
        )
        .unwrap()
        .item()
        .unwrap(),
        "KL(Laplace(0,1),CB(0.4))",
    );
    // Normal-CB (kl.py:762).
    assert_pos_inf(
        kl_divergence(
            &Normal::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap(),
            &cb(0.4),
        )
        .unwrap()
        .item()
        .unwrap(),
        "KL(Normal(0,1),CB(0.4))",
    );
    // Pareto-CB (kl.py:796).
    assert_pos_inf(
        kl_divergence(
            &Pareto::new(scalar(1.0f64).unwrap(), scalar(2.0f64).unwrap()).unwrap(),
            &cb(0.4),
        )
        .unwrap()
        .item()
        .unwrap(),
        "KL(Pareto(1,2),CB(0.4))",
    );
}

#[test]
fn divergence_cb_from_logits_band_and_extremes() {
    // torch: CB(logits=0).probs == 0.5 (the uniform), mean == 0.5.
    let d0 = ContinuousBernoulli::from_logits(scalar(0.0f64).unwrap()).unwrap();
    assert!((d0.probs().item().unwrap() - 0.5).abs() < 1e-15);
    assert!((d0.mean().unwrap().item().unwrap() - 0.5).abs() < 1e-12);
    // torch: CB(logits=5).probs == 0.9933071490757153, mean == 0.8067836549063048.
    let d5 = ContinuousBernoulli::from_logits(scalar(5.0f64).unwrap()).unwrap();
    assert!(
        (d5.probs().item().unwrap() - 0.993_307_149_075_715_3).abs() < 1e-12,
        "probs got {}",
        d5.probs().item().unwrap()
    );
    assert!(
        (d5.mean().unwrap().item().unwrap() - 0.806_783_654_906_304_8).abs() < 1e-12,
        "mean got {}",
        d5.mean().unwrap().item().unwrap()
    );
    // torch: CB(logits=-5).probs == 0.0066928509242848554, mean == 0.19321634509369578.
    let dn5 = ContinuousBernoulli::from_logits(scalar(-5.0f64).unwrap()).unwrap();
    assert!((dn5.probs().item().unwrap() - 0.006_692_850_924_284_855_4).abs() < 1e-12);
    assert!((dn5.mean().unwrap().item().unwrap() - 0.193_216_345_093_695_78).abs() < 1e-12);
}

#[test]
fn divergence_cb_f32_mean_and_log_prob() {
    // torch f32: CB(0.3).mean == 0.43022245168685913;
    //            CB(0.3).log_prob(0.25) == 0.18208837509155273.
    let d = ContinuousBernoulli::new(scalar(0.3f32).unwrap()).unwrap();
    assert!(
        (d.mean().unwrap().item().unwrap() - 0.430_222_45).abs() < 1e-6,
        "f32 mean got {}",
        d.mean().unwrap().item().unwrap()
    );
    let lp = d
        .log_prob(&scalar(0.25f32).unwrap())
        .unwrap()
        .item()
        .unwrap();
    assert!(
        (lp - 0.182_088_37).abs() < 1e-6,
        "f32 log_prob got {lp}"
    );
}
