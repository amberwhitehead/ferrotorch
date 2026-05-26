//! Wave-D partial-landing audit — discriminator probes for the 3 distribution
//! files (`uniform.rs`, `von_mises.rs`, `one_hot_categorical.rs`) the
//! rate-limited builders DID touch.
//!
//! Each probe pins one upstream observable. PASS means the SHIPPED claim
//! is genuinely wired; FAIL means the claim is vocab-only or wrong.
//!
//! Upstream sites:
//!   - `/home/doll/pytorch/torch/distributions/uniform.py:46-47`
//!   - `/home/doll/pytorch/torch/distributions/von_mises.py:200-221`
//!   - `/home/doll/pytorch/torch/distributions/one_hot_categorical.py:86-126`
//!   - `/home/doll/pytorch/torch/distributions/one_hot_categorical.py:129-143`
//!     (StraightThrough variant)
//!
//! Tracking: issue #1542 (wave-D partial-landing audit). Blocker issues:
//! #1413 #1417 #1418 #1429 #1430 #1431 #1434.

use ferrotorch_core::creation::{scalar, tensor};
use ferrotorch_distributions::{Distribution, OneHotCategorical, Uniform, VonMises};

// NOTE on #1418 (`OneHotCategoricalStraightThrough`):
// `mod one_hot_categorical;` at `src/lib.rs:85` is a BARE `mod`
// declaration (private), and `pub use one_hot_categorical::OneHotCategorical;`
// at `src/lib.rs:115` re-exports ONLY the base distribution. The SHIPPED
// claim in `src/one_hot_categorical.rs:28` ("consumer: `pub use
// one_hot_categorical::OneHotCategoricalStraightThrough` in `lib.rs`") is
// FALSE — the type is UNREACHABLE from external crates. The compile-failure
// evidence below documents this:
//
//   error[E0603]: module `one_hot_categorical` is private
//     --> tests/divergence_wave_d_partial_audit.rs
//      |
//      | use ferrotorch_distributions::one_hot_categorical::OneHotCategoricalStraightThrough;
//      |                               ^^^^^^^^^^^^^^^^^^^ private module
//
// Therefore #1418 CANNOT be closed; the type is effectively dead code.

// ===========================================================================
// uniform.rs — #1429 mode-is-NaN
// ===========================================================================

/// Divergence probe: ferrotorch's `Uniform::mode` must return NaN, mirroring
/// `torch/distributions/uniform.py:46-47`:
/// ```python
/// @property
/// def mode(self) -> Tensor:
///     return nan * self.high
/// ```
#[test]
fn wave_d_uniform_mode_is_nan_1429() {
    let d = Uniform::new(scalar(0.0f64).unwrap(), scalar(4.0f64).unwrap()).unwrap();
    let m = d.mode().unwrap();
    let v = m.item().unwrap();
    assert!(
        v.is_nan(),
        "#1429: Uniform::mode must be NaN (got {v}); upstream `uniform.py:47` returns `nan * self.high`"
    );
}

/// Divergence probe: with positive high, `nan * high` is NaN (NOT signed
/// infinity, NOT 0). Pins NaN propagation contract through scalar multiply.
#[test]
fn wave_d_uniform_mode_is_nan_when_high_is_positive_1429() {
    let d = Uniform::new(scalar(-2.0f64).unwrap(), scalar(7.5f64).unwrap()).unwrap();
    let m = d.mode().unwrap();
    let v = m.item().unwrap();
    assert!(
        v.is_nan(),
        "#1429: Uniform::mode must be NaN regardless of high sign (got {v})"
    );
}

// ===========================================================================
// von_mises.rs — #1431 mode/variance via Bessel ratio
// ===========================================================================

/// Divergence probe: `VonMises::mode` must return `loc`, mirroring
/// `torch/distributions/von_mises.py:206-208`:
/// ```python
/// @property
/// def mode(self) -> Tensor:
///     return self.loc
/// ```
#[test]
fn wave_d_von_mises_mode_is_loc_1431() {
    let d = VonMises::new(scalar(0.7f64).unwrap(), scalar(5.0f64).unwrap()).unwrap();
    let m = d.mode().unwrap();
    let v = m.item().unwrap();
    assert!(
        (v - 0.7).abs() < 1e-12,
        "#1431: VonMises::mode must equal loc=0.7 (got {v}); upstream `von_mises.py:208`"
    );
}

/// Divergence probe: `VonMises::variance` at `kappa=5.0` must equal
/// `1 - I_1(5)/I_0(5) ≈ 0.10662`, mirroring `torch/distributions/von_mises.py:211-221`:
/// ```python
/// return (
///     1 - (
///         _log_modified_bessel_fn(self.concentration, order=1)
///         - _log_modified_bessel_fn(self.concentration, order=0)
///     ).exp()
/// )
/// ```
/// Reference value from scipy.special: I_1(5)/I_0(5) = 0.893383,
/// so variance = 1 - 0.893383 = 0.106617.
///
/// (Note: the user-provided `≈0.0103` figure in the dispatch brief is wrong.
/// The Bessel ratio I_1(5)/I_0(5) is 0.8934, not 0.9897. We pin to the
/// scipy-verified value 0.10662.)
#[test]
fn wave_d_von_mises_variance_kappa_5_1431() {
    let d = VonMises::new(scalar(0.0f64).unwrap(), scalar(5.0f64).unwrap()).unwrap();
    let var = d.variance().unwrap();
    let v = var.item().unwrap();
    // Abramowitz-Stegun polynomial coefficients should give a result within
    // ~5e-3 of the scipy reference.
    let expected = 0.106617_f64;
    assert!(
        (v - expected).abs() < 5e-3,
        "#1431: VonMises::variance(kappa=5) must be ~{expected} (got {v}); upstream `von_mises.py:218-219` I_1/I_0 ratio"
    );
}

/// Divergence probe: `VonMises::variance` at `kappa=1.0` must equal
/// `1 - I_1(1)/I_0(1) ≈ 0.55361`. Second kappa to triangulate the
/// polynomial fit.
#[test]
fn wave_d_von_mises_variance_kappa_1_1431() {
    let d = VonMises::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
    let var = d.variance().unwrap();
    let v = var.item().unwrap();
    let expected = 0.553610_f64;
    assert!(
        (v - expected).abs() < 5e-3,
        "#1431: VonMises::variance(kappa=1) must be ~{expected} (got {v})"
    );
}

/// Divergence probe: `VonMises::has_rsample` must be `false`, mirroring
/// `torch/distributions/von_mises.py:131`: `has_rsample = False`.
#[test]
fn wave_d_von_mises_has_rsample_false_1431() {
    let d = VonMises::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
    assert!(
        !d.has_rsample(),
        "#1431: VonMises has_rsample=False per von_mises.py:131"
    );
}

/// Divergence probe: `VonMises::expand` must broadcast loc & concentration
/// to the requested batch shape, mirroring `von_mises.py:190-197`. Probe by
/// expanding from `[1]` to `[3]` and checking the resulting batch_shape.
#[test]
fn wave_d_von_mises_expand_1431() {
    let d = VonMises::new(scalar(0.5f64).unwrap(), scalar(2.0f64).unwrap()).unwrap();
    let expanded = d.expand(&[3]).unwrap();
    let bs = expanded.batch_shape();
    assert_eq!(
        bs,
        vec![3],
        "#1431: VonMises::expand([3]) must yield batch_shape=[3] (got {bs:?})"
    );
}

// ===========================================================================
// one_hot_categorical.rs — #1413 mean/mode/variance + #1417 enumerate_support
// ===========================================================================

/// Divergence probe: `OneHotCategorical::mean` must equal `probs`,
/// mirroring `torch/distributions/one_hot_categorical.py:86-88`:
/// ```python
/// @property
/// def mean(self) -> Tensor:
///     return self._categorical.probs
/// ```
#[test]
fn wave_d_one_hot_cat_mean_equals_probs_1413() {
    let probs = tensor(&[0.2f64, 0.5, 0.3]).unwrap();
    let d = OneHotCategorical::new(probs).unwrap();
    let mean = d.mean().unwrap();
    let m = mean.data_vec().unwrap();
    assert_eq!(m.len(), 3, "#1413: mean must be [K=3]");
    let expected = [0.2, 0.5, 0.3];
    for (i, (&got, exp)) in m.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-12,
            "#1413: mean[{i}] = {got}, want {exp} (upstream one_hot_categorical.py:88)"
        );
    }
}

/// Divergence probe: `OneHotCategorical::mode` must return the one-hot
/// indicator of argmax(probs), mirroring
/// `torch/distributions/one_hot_categorical.py:90-94`.
#[test]
fn wave_d_one_hot_cat_mode_is_one_hot_argmax_1413() {
    let probs = tensor(&[0.2f64, 0.5, 0.3]).unwrap();
    let d = OneHotCategorical::new(probs).unwrap();
    let mode = d.mode().unwrap();
    let m = mode.data_vec().unwrap();
    assert_eq!(m.len(), 3, "#1413: mode must be [K=3]");
    let expected = [0.0, 1.0, 0.0]; // argmax = index 1
    for (i, (&got, exp)) in m.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-12,
            "#1413: mode[{i}] = {got}, want {exp} (one-hot of argmax)"
        );
    }
}

/// Divergence probe: `OneHotCategorical::variance` must equal
/// `probs * (1 - probs)` elementwise, mirroring
/// `torch/distributions/one_hot_categorical.py:96-98`.
#[test]
fn wave_d_one_hot_cat_variance_p_times_one_minus_p_1413() {
    let probs = tensor(&[0.2f64, 0.5, 0.3]).unwrap();
    let d = OneHotCategorical::new(probs).unwrap();
    let var = d.variance().unwrap();
    let v = var.data_vec().unwrap();
    assert_eq!(v.len(), 3, "#1413: variance must be [K=3]");
    let expected = [0.2 * 0.8, 0.5 * 0.5, 0.3 * 0.7];
    for (i, (&got, exp)) in v.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-12,
            "#1413: variance[{i}] = {got}, want {exp} (p*(1-p))"
        );
    }
}

/// Divergence probe: `OneHotCategorical::has_enumerate_support` must be
/// `true`, mirroring `torch/distributions/one_hot_categorical.py:47`:
/// `has_enumerate_support = True`.
#[test]
fn wave_d_one_hot_cat_has_enumerate_support_true_1417() {
    let d = OneHotCategorical::new(tensor(&[0.3f64, 0.7]).unwrap()).unwrap();
    assert!(
        d.has_enumerate_support(),
        "#1417: has_enumerate_support must be True per one_hot_categorical.py:47"
    );
}

/// Divergence probe: `OneHotCategorical::enumerate_support(false)` must
/// return the `[K, K]` identity matrix, mirroring
/// `torch/distributions/one_hot_categorical.py:120-126`. The k-th row is
/// the one-hot indicator for category k.
#[test]
fn wave_d_one_hot_cat_enumerate_support_is_eye_1417() {
    let d = OneHotCategorical::new(tensor(&[0.25f64, 0.25, 0.5]).unwrap()).unwrap();
    let support = d.enumerate_support(false).unwrap();
    let shape = support.shape();
    assert_eq!(
        shape,
        &[3, 3],
        "#1417: enumerate_support must be [K, K] = [3, 3], got {shape:?}"
    );
    let s = support.data_vec().unwrap();
    let expected = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    for (i, (&got, exp)) in s.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-12,
            "#1417: enumerate_support flat[{i}] = {got}, want {exp} (eye(3))"
        );
    }
}

// ===========================================================================
// #1418 OneHotCategoricalStraightThrough — UNREACHABLE
// ===========================================================================
// No runnable probe possible. See the module-level NOTE above. The type
// exists in `src/one_hot_categorical.rs:319+` but is gated behind a private
// `mod` and never `pub use`-re-exported. The SHIPPED claim is false.
//
// Additionally, even if the type WERE reachable, inspection of the impl
// shows the gradient path is dropped (`let _ = probs_broadcast;` returns
// `samples` only) — the straight-through estimator's entire purpose
// (gradient flow through probs) is not wired. So even unblocking the
// re-export would not close #1418.
