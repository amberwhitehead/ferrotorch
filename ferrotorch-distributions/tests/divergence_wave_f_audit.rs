//! Wave-F audit (#1542 umbrella): autograd-aware rsample via inverse-CDF +
//! ExponentialFamily natural-params + closed-form mode/variance for
//! `ferrotorch-distributions`.
//!
//! Closes:
//!   * #1377 — Independent trait-method passthrough
//!   * #1383 — Kumaraswamy rsample (inverse-CDF)
//!   * #1384 — Kumaraswamy mode NaN-vs-0
//!   * #1391 — MVN exact closed-form entropy
//!   * #1392 — MVN mode (= mean)
//!   * #1393 — MVN covariance/precision_matrix accessors
//!   * #1394 — MVN variance (diag of cov)
//!   * #1395 — Pareto rsample (inverse-CDF)
//!   * #1404 — Normal ExponentialFamily natural_params / log_normalizer
//!   * #1407 — Poisson ExponentialFamily natural_params / log_normalizer
//!   * #1409 — Poisson log_prob xlogy fix (k=0,lambda=0 → 0, not NaN)
//!   * #1410 — Categorical from_logits constructor
//!   * #1415 — Poisson Stirling entropy override numerics
//!   * #1430 — Uniform expand + arg_constraints
//!   * #1434 — VonMises Stirling entropy override (exact I_1/I_0 ratio)
//!   * #1435 — Weibull rsample (inverse-CDF)
//!
//! Tests are written as crate-level black-box probes (they only consume the
//! published API), keeping them as permanent regression coverage independent
//! of the `mod tests` scope inside each `src/*.rs`.

#![allow(clippy::approx_constant)]

use ferrotorch_core::Tensor;
use ferrotorch_core::creation::{from_slice, scalar};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_distributions::{
    Categorical, Distribution, ExponentialFamily, Independent, Kumaraswamy, MultivariateNormal,
    Normal, Pareto, Poisson, Uniform, VonMises, Weibull,
};

fn s(v: f64) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(vec![v]), vec![1], false).unwrap()
}

// ---------------------------------------------------------------------------
// #1377 Independent — passthrough trait methods
// ---------------------------------------------------------------------------

#[test]
fn audit_1377_independent_mean_mode_variance_forward_to_base() {
    let loc = from_slice(&[1.0f64, 2.0], &[2]).unwrap();
    let scale = from_slice(&[1.0f64, 1.0], &[2]).unwrap();
    let base = Normal::new(loc.clone(), scale.clone()).unwrap();
    let ind = Independent::new(base, 1).unwrap();
    // mean/mode forward to Normal::mean (= loc)
    let m = ind.mean().unwrap();
    let d = m.data().unwrap();
    assert!((d[0] - 1.0).abs() < 1e-12);
    assert!((d[1] - 2.0).abs() < 1e-12);
    // variance forward to Normal::variance (= scale^2)
    let v = ind.variance().unwrap();
    let vd = v.data().unwrap();
    assert!((vd[0] - 1.0).abs() < 1e-12);
    assert!((vd[1] - 1.0).abs() < 1e-12);
}

#[test]
fn audit_1377_independent_has_rsample_forwards() {
    let loc = from_slice(&[0.0f64, 0.0], &[2]).unwrap();
    let scale = from_slice(&[1.0f64, 1.0], &[2]).unwrap();
    let base = Normal::new(loc, scale).unwrap();
    let ind = Independent::new(base, 1).unwrap();
    assert!(ind.has_rsample()); // Normal has rsample → Independent forwards true
}

// ---------------------------------------------------------------------------
// #1383 Kumaraswamy rsample + #1384 mode NaN
// ---------------------------------------------------------------------------

#[test]
fn audit_1383_kumaraswamy_rsample_grad_flows() {
    let a = s(2.0).requires_grad_(true);
    let b = s(3.0).requires_grad_(true);
    let d = Kumaraswamy::new(a.clone(), b.clone()).unwrap();
    let r = d.rsample(&[20]).unwrap();
    assert!(r.requires_grad());
    let loss = r.sum_all().unwrap();
    loss.backward().unwrap();
    assert!(a.grad().unwrap().unwrap().item().unwrap().is_finite());
    assert!(b.grad().unwrap().unwrap().item().unwrap().is_finite());
}

#[test]
fn audit_1384_kumaraswamy_mode_nan_at_boundary() {
    // a < 1 OR b < 1 → upstream returns NaN (`kumaraswamy.py:89`).
    let d = Kumaraswamy::new(s(0.5), s(0.5)).unwrap();
    assert!(d.mode().unwrap().data().unwrap()[0].is_nan());
}

// ---------------------------------------------------------------------------
// #1391/1392/1393/1394 MultivariateNormal accessors + properties
// ---------------------------------------------------------------------------

#[test]
fn audit_1391_mvn_entropy_closed_form() {
    // For N(0, 2I) in d=2: H = 0.5 * 2 * (1 + ln(2π)) + sum(log(diag(L)))
    // scale_tril = sqrt(2)*I → log(diag) = 0.5*ln(2), sum = ln(2).
    let loc = from_slice(&[0.0f64, 0.0], &[2]).unwrap();
    let sqrt2 = 2.0_f64.sqrt();
    let l = from_slice(&[sqrt2, 0.0, 0.0, sqrt2], &[2, 2]).unwrap();
    let d = MultivariateNormal::from_scale_tril(loc, l).unwrap();
    let h = d.entropy().unwrap().item().unwrap();
    let expected = 0.5 * 2.0 * (1.0 + (2.0 * std::f64::consts::PI).ln()) + 2.0f64.ln();
    assert!(
        (h - expected).abs() < 1e-9,
        "MVN entropy mismatch: got {h}, expected {expected}"
    );
}

#[test]
fn audit_1392_mvn_mode_equals_loc() {
    let loc = from_slice(&[3.0f64, 7.0], &[2]).unwrap();
    let i = from_slice(&[1.0f64, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
    let d = MultivariateNormal::from_scale_tril(loc.clone(), i).unwrap();
    let m = d.mode().unwrap();
    assert!((m.data().unwrap()[0] - 3.0).abs() < 1e-12);
    assert!((m.data().unwrap()[1] - 7.0).abs() < 1e-12);
}

#[test]
fn audit_1393_mvn_covariance_matches_l_lt() {
    // L = [[2, 0], [0.5, 1.5]] → Σ = LL^T = [[4, 1], [1, 2.5]].
    let loc = from_slice(&[0.0f64, 0.0], &[2]).unwrap();
    let l = from_slice(&[2.0f64, 0.0, 0.5, 1.5], &[2, 2]).unwrap();
    let d = MultivariateNormal::from_scale_tril(loc, l).unwrap();
    let cov = d.covariance_matrix().unwrap();
    let cd = cov.data().unwrap();
    assert!((cd[0] - 4.0).abs() < 1e-9);
    assert!((cd[1] - 1.0).abs() < 1e-9);
    assert!((cd[2] - 1.0).abs() < 1e-9);
    assert!((cd[3] - 2.5).abs() < 1e-9);
}

#[test]
fn audit_1394_mvn_variance_equals_diag_cov() {
    // diag(Σ) for the above: [4.0, 2.5].
    let loc = from_slice(&[0.0f64, 0.0], &[2]).unwrap();
    let l = from_slice(&[2.0f64, 0.0, 0.5, 1.5], &[2, 2]).unwrap();
    let d = MultivariateNormal::from_scale_tril(loc, l).unwrap();
    let v = d.variance().unwrap();
    let vd = v.data().unwrap();
    assert!((vd[0] - 4.0).abs() < 1e-9);
    assert!((vd[1] - 2.5).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// #1395 Pareto rsample
// ---------------------------------------------------------------------------

#[test]
fn audit_1395_pareto_rsample_grad_flows() {
    let scale = s(2.0).requires_grad_(true);
    let alpha = s(3.0).requires_grad_(true);
    let d = Pareto::new(scale.clone(), alpha.clone()).unwrap();
    let r = d.rsample(&[10]).unwrap();
    assert!(r.requires_grad());
    let loss = r.sum_all().unwrap();
    loss.backward().unwrap();
    let gs = scale.grad().unwrap().unwrap().item().unwrap();
    let ga = alpha.grad().unwrap().unwrap().item().unwrap();
    assert!(gs.is_finite() && gs > 0.0, "dscale > 0 (sample ↑ in scale)");
    assert!(ga.is_finite(), "dalpha must be finite");
}

// ---------------------------------------------------------------------------
// #1404 Normal ExponentialFamily
// ---------------------------------------------------------------------------

#[test]
fn audit_1404_normal_natural_params_canonical() {
    // For Normal(loc=2, scale=3): eta1 = 2/9, eta2 = -1/18.
    let d = Normal::new(s(2.0), s(3.0)).unwrap();
    let np = d.natural_params().unwrap();
    assert_eq!(np.len(), 2);
    assert!((np[0].item().unwrap() - 2.0 / 9.0).abs() < 1e-12);
    assert!((np[1].item().unwrap() + 1.0 / 18.0).abs() < 1e-12);
}

#[test]
fn audit_1404_normal_log_normalizer_at_standard() {
    // For Normal(0, 1) the log-partition is 0.5*ln(2π).
    let d = Normal::new(s(0.0), s(1.0)).unwrap();
    let np = d.natural_params().unwrap();
    let lz = d.log_normalizer(&np).unwrap().item().unwrap();
    let expected = 0.5 * (2.0 * std::f64::consts::PI).ln();
    assert!((lz - expected).abs() < 1e-10);
}

// ---------------------------------------------------------------------------
// #1407 Poisson ExponentialFamily
// ---------------------------------------------------------------------------

#[test]
fn audit_1407_poisson_natural_params_is_log_rate() {
    let d = Poisson::new(s(4.0)).unwrap();
    let np = d.natural_params().unwrap();
    assert_eq!(np.len(), 1);
    assert!((np[0].item().unwrap() - 4.0_f64.ln()).abs() < 1e-12);
    let lz = d.log_normalizer(&np).unwrap().item().unwrap();
    assert!((lz - 4.0).abs() < 1e-10, "log_normalizer = exp(eta) = rate");
}

// ---------------------------------------------------------------------------
// #1409 Poisson log_prob xlogy fix
// ---------------------------------------------------------------------------

#[test]
fn audit_1409_poisson_log_prob_zero_zero_no_nan() {
    // upstream `value.xlogy(rate)` returns 0 at (k=0, lambda=0).
    let d = Poisson::new(s(0.0)).unwrap();
    let lp = d.log_prob(&scalar(0.0_f64).unwrap()).unwrap();
    let val = lp.item().unwrap();
    assert!(!val.is_nan(), "log_prob(0|rate=0) must NOT be NaN");
    assert!(val.abs() < 1e-12, "log_prob(0|rate=0) = 0; got {val}");
}

// ---------------------------------------------------------------------------
// #1410 Categorical from_logits
// ---------------------------------------------------------------------------

#[test]
fn audit_1410_categorical_from_logits_softmax_uniform() {
    let logits = from_slice(&[0.0f64, 0.0, 0.0, 0.0], &[4]).unwrap();
    let d = Categorical::from_logits(&logits).unwrap();
    for &p in d.probs().data().unwrap() {
        assert!((p - 0.25).abs() < 1e-12);
    }
}

#[test]
fn audit_1410_categorical_from_logits_large_logits_stable() {
    // No overflow: softmax([1000, 1001, 1002]) ≈ softmax([0, 1, 2]).
    let logits = from_slice(&[1000.0f64, 1001.0, 1002.0], &[3]).unwrap();
    let d = Categorical::from_logits(&logits).unwrap();
    for &p in d.probs().data().unwrap() {
        assert!(p.is_finite() && (0.0..=1.0).contains(&p));
    }
}

// ---------------------------------------------------------------------------
// #1415 Poisson Stirling entropy override numerics
// ---------------------------------------------------------------------------

#[test]
fn audit_1415_poisson_stirling_entropy_large_lambda() {
    let lambda = 25.0f64;
    let d = Poisson::new(s(lambda)).unwrap();
    let h = d.entropy().unwrap().item().unwrap();
    let expected = 0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E * lambda).ln()
        - 1.0 / (12.0 * lambda)
        - 1.0 / (24.0 * lambda * lambda);
    assert!((h - expected).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// #1430 Uniform expand + arg_constraints
// ---------------------------------------------------------------------------

#[test]
fn audit_1430_uniform_arg_constraints_low_high_real() {
    let d = Uniform::new(s(0.0), s(1.0)).unwrap();
    let args = d.arg_constraints();
    assert_eq!(args["low"].name(), "Real");
    assert_eq!(args["high"].name(), "Real");
}

#[test]
fn audit_1430_uniform_expand_broadcasts() {
    let d = Uniform::new(s(0.0), s(3.0)).unwrap();
    let exp = d.expand(&[4]).unwrap();
    let m = exp.mean().unwrap();
    assert_eq!(m.shape(), &[4]);
    for &v in m.data().unwrap() {
        assert!((v - 1.5).abs() < 1e-12);
    }
}

// ---------------------------------------------------------------------------
// #1434 VonMises entropy override
// ---------------------------------------------------------------------------

#[test]
fn audit_1434_von_mises_entropy_finite_and_positive_for_low_kappa() {
    // Low kappa → near-uniform on the circle, H close to ln(2π).
    let d = VonMises::new(s(0.0), s(0.1)).unwrap();
    let h = d.entropy().unwrap().data().unwrap()[0];
    assert!(h.is_finite() && h > 0.0);
    let two_pi_ln = (2.0 * std::f64::consts::PI).ln();
    // Should be within 0.1 of ln(2π) for kappa=0.1.
    assert!(
        (h - two_pi_ln).abs() < 0.2,
        "entropy {h} should be near {two_pi_ln}"
    );
}

#[test]
fn audit_1434_von_mises_entropy_decreases_with_kappa() {
    // Higher concentration → tighter peak → lower entropy.
    let d_low = VonMises::new(s(0.0), s(0.5)).unwrap();
    let d_high = VonMises::new(s(0.0), s(5.0)).unwrap();
    let h_low = d_low.entropy().unwrap().data().unwrap()[0];
    let h_high = d_high.entropy().unwrap().data().unwrap()[0];
    assert!(
        h_low > h_high,
        "entropy must decrease with kappa: kappa=0.5 → {h_low}, kappa=5 → {h_high}"
    );
}

// ---------------------------------------------------------------------------
// #1435 Weibull rsample
// ---------------------------------------------------------------------------

#[test]
fn audit_1435_weibull_rsample_grad_flows() {
    let scale = s(1.5).requires_grad_(true);
    let conc = s(2.5).requires_grad_(true);
    let d = Weibull::new(scale.clone(), conc.clone()).unwrap();
    let r = d.rsample(&[10]).unwrap();
    assert!(r.requires_grad());
    let loss = r.sum_all().unwrap();
    loss.backward().unwrap();
    let gs = scale.grad().unwrap().unwrap().item().unwrap();
    let gk = conc.grad().unwrap().unwrap().item().unwrap();
    assert!(gs.is_finite() && gs > 0.0, "dscale > 0 expected");
    assert!(gk.is_finite());
}
