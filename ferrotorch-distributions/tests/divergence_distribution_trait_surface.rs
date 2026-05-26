//! Divergence tests for the Distribution trait surface extension (#1376).
//!
//! Closes the cross-cutting trait-surface gap: every concrete distribution
//! now exposes `has_rsample`, `support`, `arg_constraints`, `event_shape`,
//! `expand`, `enumerate_support` (where finite-discrete), and the default
//! `perplexity = exp(entropy)` body.
//!
//! These tests pin both the trait defaults (return `InvalidArgument` /
//! `None` / `false` for non-overriders) AND the concrete overrides on the
//! 6 distributions the dispatch covers (Normal, Bernoulli, Categorical,
//! Exponential, Gamma, Uniform).
//!
//! Reference: `/home/doll/pytorch/torch/distributions/distribution.py:25-264`
//! (the `class Distribution` base class surface).

use ferrotorch_core::creation::{from_slice, scalar, tensor};
use ferrotorch_distributions::{
    Bernoulli, Categorical, Distribution, Exponential, Gamma, Normal, Uniform,
};

// ===========================================================================
// has_rsample — class-level reparameterization flag.
// ===========================================================================

#[test]
fn normal_has_rsample_true() {
    // torch/distributions/normal.py:18 → `has_rsample = True`.
    let d = Normal::new(scalar(0.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
    assert!(d.has_rsample());
}

#[test]
fn bernoulli_has_rsample_false() {
    // torch/distributions/bernoulli.py:13-15 (no override) → defaults to False.
    let d = Bernoulli::new(scalar(0.5f32).unwrap()).unwrap();
    assert!(!d.has_rsample());
}

#[test]
fn categorical_has_rsample_false() {
    let d = Categorical::new(tensor(&[0.5f32, 0.5]).unwrap()).unwrap();
    assert!(!d.has_rsample());
}

#[test]
fn exponential_has_rsample_true() {
    // torch/distributions/exponential.py:25 → `has_rsample = True`.
    let d = Exponential::new(scalar(1.0f32).unwrap()).unwrap();
    assert!(d.has_rsample());
}

#[test]
fn gamma_has_rsample_true() {
    // torch/distributions/gamma.py:35 → `has_rsample = True`.
    let d = Gamma::new(scalar(2.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
    assert!(d.has_rsample());
}

#[test]
fn uniform_has_rsample_true() {
    // torch/distributions/uniform.py:32 → `has_rsample = True`.
    let d = Uniform::new(scalar(0.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
    assert!(d.has_rsample());
}

// ===========================================================================
// has_enumerate_support — finite-discrete distributions only.
// ===========================================================================

#[test]
fn normal_has_enumerate_support_false() {
    let d = Normal::new(scalar(0.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
    assert!(!d.has_enumerate_support());
}

#[test]
fn bernoulli_has_enumerate_support_true() {
    // torch/distributions/bernoulli.py:15 → `has_enumerate_support = True`.
    let d = Bernoulli::new(scalar(0.5f32).unwrap()).unwrap();
    assert!(d.has_enumerate_support());
}

#[test]
fn categorical_has_enumerate_support_true() {
    // torch/distributions/categorical.py:46 → `has_enumerate_support = True`.
    let d = Categorical::new(tensor(&[0.5f32, 0.5]).unwrap()).unwrap();
    assert!(d.has_enumerate_support());
}

// ===========================================================================
// support — DistConstraint descriptor.
// ===========================================================================

#[test]
fn normal_support_is_real() {
    // torch/distributions/normal.py:17 → `support = constraints.real`.
    let d = Normal::new(scalar(0.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
    let s = d.support().expect("Normal::support must be Some");
    assert_eq!(s.name(), "Real");
    assert!(!s.is_discrete());
    assert_eq!(s.event_dim(), 0);
}

#[test]
fn bernoulli_support_is_boolean() {
    // torch/distributions/bernoulli.py:14 → `support = constraints.boolean`.
    let d = Bernoulli::new(scalar(0.5f32).unwrap()).unwrap();
    let s = d.support().expect("Bernoulli::support must be Some");
    assert_eq!(s.name(), "Boolean");
    assert!(s.is_discrete());
}

#[test]
fn exponential_support_is_nonnegative() {
    // torch/distributions/exponential.py:24 → `support = constraints.nonnegative`.
    let d = Exponential::new(scalar(2.0f32).unwrap()).unwrap();
    let s = d.support().expect("Exponential::support must be Some");
    assert_eq!(s.name(), "NonNegative");
}

#[test]
fn gamma_support_is_nonnegative() {
    // torch/distributions/gamma.py:34 → `support = constraints.nonnegative`.
    let d = Gamma::new(scalar(2.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
    let s = d.support().expect("Gamma::support must be Some");
    assert_eq!(s.name(), "NonNegative");
}

#[test]
fn uniform_support_is_half_open_interval() {
    // torch/distributions/uniform.py:80-83 → `support = constraints.interval(low, high)`.
    let d = Uniform::new(scalar(2.0f32).unwrap(), scalar(5.0f32).unwrap()).unwrap();
    let s = d.support().expect("Uniform::support must be Some");
    assert_eq!(s.name(), "HalfOpenInterval");
}

#[test]
fn categorical_support_is_nonneg_discrete_proxy() {
    // ferrotorch returns NonNegative as a discrete-non-negative proxy until
    // #1372 ships `IntegerInterval`. PyTorch's
    // `categorical.py:165-167` is `integer_interval(0, K-1)`.
    let d = Categorical::new(tensor(&[0.25f32, 0.25, 0.25, 0.25]).unwrap()).unwrap();
    let s = d.support().expect("Categorical::support must be Some");
    assert_eq!(s.name(), "NonNegative");
}

// ===========================================================================
// arg_constraints — parameter-name → DistConstraint map.
// ===========================================================================

#[test]
fn normal_arg_constraints_loc_real_scale_positive() {
    // torch/distributions/normal.py:15-16:
    //   arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    let d = Normal::new(scalar(0.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
    let m = d.arg_constraints();
    assert_eq!(m.len(), 2);
    assert_eq!(m.get("loc").map(|c| c.name()), Some("Real"));
    assert_eq!(m.get("scale").map(|c| c.name()), Some("Positive"));
}

#[test]
fn bernoulli_arg_constraints_probs_unit_interval() {
    let d = Bernoulli::new(scalar(0.5f32).unwrap()).unwrap();
    let m = d.arg_constraints();
    assert_eq!(m.get("probs").map(|c| c.name()), Some("UnitInterval"));
}

#[test]
fn exponential_arg_constraints_rate_positive() {
    let d = Exponential::new(scalar(2.0f32).unwrap()).unwrap();
    let m = d.arg_constraints();
    assert_eq!(m.get("rate").map(|c| c.name()), Some("Positive"));
}

#[test]
fn gamma_arg_constraints_both_positive() {
    let d = Gamma::new(scalar(2.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
    let m = d.arg_constraints();
    assert_eq!(m.get("concentration").map(|c| c.name()), Some("Positive"));
    assert_eq!(m.get("rate").map(|c| c.name()), Some("Positive"));
}

#[test]
fn uniform_arg_constraints_low_high_real() {
    let d = Uniform::new(scalar(0.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
    let m = d.arg_constraints();
    // Both individual constraints are Real; the inter-parameter `low < high`
    // is left to the constructor (ferrotorch's R-DEV-4 path).
    assert_eq!(m.get("low").map(|c| c.name()), Some("Real"));
    assert_eq!(m.get("high").map(|c| c.name()), Some("Real"));
}

#[test]
fn categorical_arg_constraints_probs_simplex() {
    let d = Categorical::new(tensor(&[0.5f32, 0.5]).unwrap()).unwrap();
    let m = d.arg_constraints();
    let probs_c = m.get("probs").expect("probs constraint present");
    assert_eq!(probs_c.name(), "Simplex");
    assert_eq!(probs_c.event_dim(), 1);
}

// ===========================================================================
// event_shape — single-sample shape (empty for univariate).
// ===========================================================================

#[test]
fn univariate_distributions_have_empty_event_shape() {
    assert!(
        Normal::new(scalar(0.0f32).unwrap(), scalar(1.0f32).unwrap())
            .unwrap()
            .event_shape()
            .is_empty()
    );
    assert!(
        Bernoulli::new(scalar(0.5f32).unwrap())
            .unwrap()
            .event_shape()
            .is_empty()
    );
    assert!(
        Exponential::new(scalar(1.0f32).unwrap())
            .unwrap()
            .event_shape()
            .is_empty()
    );
    assert!(
        Gamma::new(scalar(2.0f32).unwrap(), scalar(1.0f32).unwrap())
            .unwrap()
            .event_shape()
            .is_empty()
    );
    assert!(
        Uniform::new(scalar(0.0f32).unwrap(), scalar(1.0f32).unwrap())
            .unwrap()
            .event_shape()
            .is_empty()
    );
}

// ===========================================================================
// expand — broadcast to a target batch_shape.
// ===========================================================================

#[test]
fn normal_expand_broadcasts_loc_and_scale() {
    let d = Normal::new(scalar(2.0f32).unwrap(), scalar(1.5f32).unwrap()).unwrap();
    let expanded = d.expand(&[4]).unwrap();
    // Sample shape must match the expanded batch_shape.
    let s = expanded.sample(&[4]).unwrap();
    assert_eq!(s.shape(), &[4]);
    // Mean of the expanded distribution should be a 4-vec of 2.0.
    let m = expanded.mean().unwrap();
    let data = m.data().unwrap();
    assert_eq!(data.len(), 4);
    for v in data {
        assert!((v - 2.0).abs() < 1e-6);
    }
}

#[test]
fn bernoulli_expand_broadcasts_probs() {
    let d = Bernoulli::new(scalar(0.3f32).unwrap()).unwrap();
    let expanded = d.expand(&[5]).unwrap();
    let m = expanded.mean().unwrap();
    let data = m.data().unwrap();
    assert_eq!(data.len(), 5);
    for v in data {
        assert!((v - 0.3).abs() < 1e-6);
    }
}

#[test]
fn exponential_expand_broadcasts_rate() {
    let d = Exponential::new(scalar(2.0f32).unwrap()).unwrap();
    let expanded = d.expand(&[3]).unwrap();
    let m = expanded.mean().unwrap();
    let data = m.data().unwrap();
    assert_eq!(data.len(), 3);
    for v in data {
        // mean = 1/rate = 0.5
        assert!((v - 0.5).abs() < 1e-6);
    }
}

#[test]
fn gamma_expand_broadcasts_both_parameters() {
    let d = Gamma::new(scalar(4.0f32).unwrap(), scalar(2.0f32).unwrap()).unwrap();
    let expanded = d.expand(&[2]).unwrap();
    let m = expanded.mean().unwrap();
    let data = m.data().unwrap();
    assert_eq!(data.len(), 2);
    for v in data {
        // mean = concentration / rate = 2.0
        assert!((v - 2.0).abs() < 1e-6);
    }
}

#[test]
fn uniform_expand_broadcasts_bounds() {
    let d = Uniform::new(scalar(0.0f32).unwrap(), scalar(4.0f32).unwrap()).unwrap();
    let expanded = d.expand(&[3]).unwrap();
    let m = expanded.mean().unwrap();
    let data = m.data().unwrap();
    assert_eq!(data.len(), 3);
    for v in data {
        assert!((v - 2.0).abs() < 1e-6);
    }
}

#[test]
fn categorical_expand_rejects_for_now() {
    // ferrotorch's Categorical is 1-D-probs-only; #1410 tracks N-D batched
    // probs. Until then, `expand` correctly surfaces the limitation rather
    // than silently producing an under-specified instance.
    let d = Categorical::new(tensor(&[0.5f32, 0.5]).unwrap()).unwrap();
    assert!(d.expand(&[3]).is_err());
}

// ===========================================================================
// enumerate_support — finite-discrete enumeration.
// ===========================================================================

#[test]
fn bernoulli_enumerate_support_yields_zero_one() {
    let d = Bernoulli::new(scalar(0.5f32).unwrap()).unwrap();
    let e = d.enumerate_support(false).unwrap();
    let data = e.data().unwrap();
    assert_eq!(data, &[0.0, 1.0]);
}

#[test]
fn categorical_enumerate_support_yields_0_to_k_minus_1() {
    let d = Categorical::new(tensor(&[0.1f32, 0.2, 0.3, 0.4]).unwrap()).unwrap();
    let e = d.enumerate_support(false).unwrap();
    let data = e.data().unwrap();
    assert_eq!(data, &[0.0, 1.0, 2.0, 3.0]);
}

#[test]
fn normal_enumerate_support_errors() {
    // Continuous distributions don't have an enumerable support.
    let d = Normal::new(scalar(0.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
    assert!(d.enumerate_support(false).is_err());
}

// ===========================================================================
// perplexity = exp(entropy) — default body.
// ===========================================================================

#[test]
fn normal_perplexity_equals_exp_entropy() {
    // Verifies the default `perplexity` body matches `exp(entropy)`
    // element-wise, per torch/distributions/distribution.py:257-264.
    let d = Normal::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
    let h = d.entropy().unwrap().item().unwrap();
    let p = d.perplexity().unwrap().item().unwrap();
    assert!((p - h.exp()).abs() < 1e-10);
}

#[test]
fn bernoulli_perplexity_equals_exp_entropy() {
    let d = Bernoulli::new(scalar(0.3f64).unwrap()).unwrap();
    let h = d.entropy().unwrap().item().unwrap();
    let p = d.perplexity().unwrap().item().unwrap();
    assert!((p - h.exp()).abs() < 1e-10);
}

#[test]
fn exponential_perplexity_equals_exp_entropy() {
    let d = Exponential::new(scalar(2.5f64).unwrap()).unwrap();
    let h = d.entropy().unwrap().item().unwrap();
    let p = d.perplexity().unwrap().item().unwrap();
    assert!((p - h.exp()).abs() < 1e-10);
}

// ===========================================================================
// Trait defaults — distributions that did NOT override still get sensible
// behavior (None / empty / err).
// ===========================================================================

#[test]
fn default_trait_methods_via_uncovered_distribution() {
    // We pick `Categorical` (covered for support/arg_constraints) and
    // verify that methods we explicitly left at trait-default still
    // behave as documented. Specifically:
    // - `mean()` returns InvalidArgument for Categorical (no scalar mean).
    let d = Categorical::new(tensor(&[0.5f32, 0.5]).unwrap()).unwrap();
    assert!(d.mean().is_err());
    assert!(d.variance().is_err());
}

// ===========================================================================
// Batched parameters via from_slice path.
// ===========================================================================

#[test]
fn normal_expand_from_batched_params() {
    // Start with a batched Normal, expand to a larger shape.
    let loc = from_slice(&[1.0f32, 2.0], &[2]).unwrap();
    let scale = from_slice(&[0.5f32, 1.5], &[2]).unwrap();
    let d = Normal::new(loc, scale).unwrap();
    let expanded = d.expand(&[6]).unwrap();
    let m = expanded.mean().unwrap();
    let data = m.data().unwrap();
    assert_eq!(data.len(), 6);
    // Cycling pattern: 1.0, 2.0, 1.0, 2.0, 1.0, 2.0
    for (i, v) in data.iter().enumerate() {
        let expected = if i % 2 == 0 { 1.0 } else { 2.0 };
        assert!((v - expected).abs() < 1e-6);
    }
}
