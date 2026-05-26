//! Constraint objects for distribution parameter and support validation.
//!
//! A constraint represents a region of valid values. Distributions use
//! constraints to declare the support of their samples and to validate
//! parameter domains.
//!
//! This mirrors PyTorch's `torch.distributions.constraints` module.
//!
//! CL-330
//!
//! ## REQ status (per `.design/ferrotorch-distributions/constraints.md`)
//!
//! Full evidence rows (impl + non-test production consumer + upstream cites)
//! live in the design doc; this synopsis is a one-line summary per REQ.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`Constraint` trait surface) | SHIPPED | `pub trait Constraint: Send + Sync` with `check<T: Float>`, default `is_discrete`/`event_dim`, required `name` in `constraints.rs` mirroring `torch/distributions/constraints.py:80-106`; consumer: `pub trait Constraint` is grandfathered public API (goal.md S5) + `pub use constraints` re-export in `lib.rs` |
//! | REQ-2 (`Real` + `real()` constructor) | SHIPPED | `pub struct Real` + `impl Constraint for Real` (rejects NaN) + `pub fn real()` in `constraints.rs` mirroring `torch/distributions/constraints.py:_Real`; consumer: `pub use constraints` re-export in `lib.rs` |
//! | REQ-3 (half-line `Positive`/`NonNegative`/`LessThan<T>`) | SHIPPED | `pub struct Positive`, `NonNegative`, `LessThan<T>` + constructors in `constraints.rs` mirroring upstream `_GreaterThan(0.)`/`_GreaterThanEq(0.)`/`_LessThan(...)`; consumer: `pub use constraints` re-export in `lib.rs` |
//! | REQ-4 (parametric `GreaterThan<T>` / `GreaterThanEq<T>`) | SHIPPED | `pub struct GreaterThan<T: Float>` + `GreaterThanEq<T: Float>` with `T::from(self.lower_bound).unwrap()` cross-dtype promotion in `constraints.rs` mirroring `_GreaterThan`/`_GreaterThanEq`; consumer: `pub use constraints` re-export in `lib.rs` |
//! | REQ-5 (interval constraints) | SHIPPED | `pub struct OpenInterval<T>`, `ClosedInterval<T>`, `HalfOpenInterval<T>` + constructors in `constraints.rs` mirroring `_Interval`/`_HalfOpenInterval`; consumer: `pub use constraints` re-export in `lib.rs` |
//! | REQ-6 (`UnitInterval`/`BooleanConstraint`) | SHIPPED | `pub struct UnitInterval`, `BooleanConstraint` + `unit_interval()`/`boolean()` constructors with `BooleanConstraint::is_discrete() -> true` in `constraints.rs` mirroring `unit_interval = _Interval(0., 1.)` / `boolean = _Boolean()`; consumer: `pub use constraints` re-export in `lib.rs` |
//! | REQ-7 (`Simplex` w/ `event_dim() -> 1`) | SHIPPED | `pub struct Simplex` with `event_dim() -> 1` override + `simplex()` constructor in `constraints.rs` mirroring `_Simplex`; consumer: `pub use constraints` re-export in `lib.rs`; scalar-only `check` is doc-commented limitation tracked under #1371 |
//! | REQ-8 (17 missing upstream constraint variants) | NOT-STARTED | blocker #1372 — `IntegerInterval`, `NonNegativeInteger`, `PositiveDefinite`, `PositiveSemiDefinite`, `Multinomial`, `OneHot`, `Symmetric`, `LowerCholesky`, `LowerTriangular`, `CorrCholesky`, `RealVector`, `Cat`, `Stack`, `Independent` composite, `_Dependent`/`is_dependent`, `MixtureSameFamilyConstraint` not ported (ferrotorch ships 11 of 28 upstream variants) |
//! | REQ-9 (concrete `arg_constraints` wiring) | NOT-STARTED | blocker #1371 — no concrete distribution declares `arg_constraints` or `support`; the Constraint trait + 11 impls have zero in-crate production consumers (only `tests/conformance_distributions_discrete.rs` exercises them, which is test-only per R-DOC-3); resolution requires `lib.md` REQ-5 / blocker #1376 to land first |

use ferrotorch_core::dtype::Float;

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// A constraint defines a region of validity for distribution values.
///
/// Constraints are used by distributions to describe:
/// - The **support** — the set of values that can be sampled.
/// - **Parameter domains** — valid ranges for distribution parameters.
///
/// # Examples
///
/// ```ignore
/// use ferrotorch_distributions::constraints;
///
/// let c = constraints::positive();
/// assert!(c.check(1.0f32));
/// assert!(!c.check(-1.0f32));
/// ```
pub trait Constraint: Send + Sync {
    /// Returns `true` if `value` satisfies this constraint.
    fn check<T: Float>(&self, value: T) -> bool;

    /// Whether the constrained space is discrete.
    ///
    /// Defaults to `false` (continuous).
    fn is_discrete(&self) -> bool {
        false
    }

    /// Number of rightmost dimensions that together define an event.
    ///
    /// Defaults to `0` (univariate).
    fn event_dim(&self) -> usize {
        0
    }

    /// Human-readable name for debugging.
    fn name(&self) -> &'static str;
}

// ---------------------------------------------------------------------------
// Concrete constraints
// ---------------------------------------------------------------------------

/// Constraint for the extended real line `(-inf, inf)`.
///
/// Accepts any finite or infinite value, but rejects NaN.
#[derive(Debug, Clone, Copy)]
pub struct Real;

impl Constraint for Real {
    fn check<T: Float>(&self, value: T) -> bool {
        // Matches PyTorch: rejects NaN.
        !value.is_nan()
    }

    fn name(&self) -> &'static str {
        "Real"
    }
}

/// Constraint for strictly positive reals `(0, inf)`.
#[derive(Debug, Clone, Copy)]
pub struct Positive;

impl Constraint for Positive {
    fn check<T: Float>(&self, value: T) -> bool {
        value > T::from(0.0).unwrap()
    }

    fn name(&self) -> &'static str {
        "Positive"
    }
}

/// Constraint for non-negative reals `[0, inf)`.
#[derive(Debug, Clone, Copy)]
pub struct NonNegative;

impl Constraint for NonNegative {
    fn check<T: Float>(&self, value: T) -> bool {
        value >= T::from(0.0).unwrap()
    }

    fn name(&self) -> &'static str {
        "NonNegative"
    }
}

/// Constraint for the closed unit interval `[0, 1]`.
#[derive(Debug, Clone, Copy)]
pub struct UnitInterval;

impl Constraint for UnitInterval {
    fn check<T: Float>(&self, value: T) -> bool {
        let zero = T::from(0.0).unwrap();
        let one = T::from(1.0).unwrap();
        value >= zero && value <= one
    }

    fn name(&self) -> &'static str {
        "UnitInterval"
    }
}

/// Constraint for the Boolean set `{0, 1}`.
#[derive(Debug, Clone, Copy)]
pub struct BooleanConstraint;

impl Constraint for BooleanConstraint {
    fn check<T: Float>(&self, value: T) -> bool {
        let zero = T::from(0.0).unwrap();
        let one = T::from(1.0).unwrap();
        value == zero || value == one
    }

    fn is_discrete(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "Boolean"
    }
}

/// Constraint for a half-open interval `(lower_bound, inf)`.
#[derive(Debug, Clone, Copy)]
pub struct GreaterThan<T: Float> {
    /// The exclusive lower bound.
    pub lower_bound: T,
}

impl<S: Float> Constraint for GreaterThan<S> {
    fn check<T: Float>(&self, value: T) -> bool {
        // Convert the bound to the check type for comparison.
        let bound = T::from(self.lower_bound).unwrap();
        value > bound
    }

    fn name(&self) -> &'static str {
        "GreaterThan"
    }
}

/// Constraint for a half-open interval `[lower_bound, inf)`.
#[derive(Debug, Clone, Copy)]
pub struct GreaterThanEq<T: Float> {
    /// The inclusive lower bound.
    pub lower_bound: T,
}

impl<S: Float> Constraint for GreaterThanEq<S> {
    fn check<T: Float>(&self, value: T) -> bool {
        let bound = T::from(self.lower_bound).unwrap();
        value >= bound
    }

    fn name(&self) -> &'static str {
        "GreaterThanEq"
    }
}

/// Constraint for an open interval `(lower_bound, upper_bound)`.
#[derive(Debug, Clone, Copy)]
pub struct OpenInterval<T: Float> {
    /// The exclusive lower bound.
    pub lower_bound: T,
    /// The exclusive upper bound.
    pub upper_bound: T,
}

impl<S: Float> Constraint for OpenInterval<S> {
    fn check<T: Float>(&self, value: T) -> bool {
        let lo = T::from(self.lower_bound).unwrap();
        let hi = T::from(self.upper_bound).unwrap();
        value > lo && value < hi
    }

    fn name(&self) -> &'static str {
        "OpenInterval"
    }
}

/// Constraint for a closed interval `[lower_bound, upper_bound]`.
#[derive(Debug, Clone, Copy)]
pub struct ClosedInterval<T: Float> {
    /// The inclusive lower bound.
    pub lower_bound: T,
    /// The inclusive upper bound.
    pub upper_bound: T,
}

impl<S: Float> Constraint for ClosedInterval<S> {
    fn check<T: Float>(&self, value: T) -> bool {
        let lo = T::from(self.lower_bound).unwrap();
        let hi = T::from(self.upper_bound).unwrap();
        value >= lo && value <= hi
    }

    fn name(&self) -> &'static str {
        "ClosedInterval"
    }
}

/// Constraint for a half-open interval `[lower_bound, upper_bound)`.
///
/// This matches PyTorch's `half_open_interval`.
#[derive(Debug, Clone, Copy)]
pub struct HalfOpenInterval<T: Float> {
    /// The inclusive lower bound.
    pub lower_bound: T,
    /// The exclusive upper bound.
    pub upper_bound: T,
}

impl<S: Float> Constraint for HalfOpenInterval<S> {
    fn check<T: Float>(&self, value: T) -> bool {
        let lo = T::from(self.lower_bound).unwrap();
        let hi = T::from(self.upper_bound).unwrap();
        value >= lo && value < hi
    }

    fn name(&self) -> &'static str {
        "HalfOpenInterval"
    }
}

/// Constraint for a half-open interval `(-inf, upper_bound)`.
#[derive(Debug, Clone, Copy)]
pub struct LessThan<T: Float> {
    /// The exclusive upper bound.
    pub upper_bound: T,
}

impl<S: Float> Constraint for LessThan<S> {
    fn check<T: Float>(&self, value: T) -> bool {
        let bound = T::from(self.upper_bound).unwrap();
        value < bound
    }

    fn name(&self) -> &'static str {
        "LessThan"
    }
}

/// Constraint for the probability simplex.
///
/// Values must be non-negative and the last dimension must sum to 1.
/// Since `check` here operates on a single scalar, full simplex validation
/// requires checking a whole vector. The scalar `check` verifies the
/// non-negativity part; callers must validate the sum-to-one property
/// separately when checking vectors.
#[derive(Debug, Clone, Copy)]
pub struct Simplex;

impl Constraint for Simplex {
    fn check<T: Float>(&self, value: T) -> bool {
        // Per-element: must be non-negative (full simplex check requires
        // sum-to-one over the event dimension, done by the caller).
        value >= T::from(0.0).unwrap()
    }

    fn event_dim(&self) -> usize {
        1
    }

    fn name(&self) -> &'static str {
        "Simplex"
    }
}

// ---------------------------------------------------------------------------
// Convenience constructors (matches PyTorch's module-level constants)
// ---------------------------------------------------------------------------

/// The real line constraint `(-inf, inf)`.
pub fn real() -> Real {
    Real
}

/// Strictly positive constraint `(0, inf)`.
pub fn positive() -> Positive {
    Positive
}

/// Non-negative constraint `[0, inf)`.
pub fn nonnegative() -> NonNegative {
    NonNegative
}

/// Unit interval constraint `[0, 1]`.
pub fn unit_interval() -> UnitInterval {
    UnitInterval
}

/// Boolean constraint `{0, 1}`.
pub fn boolean() -> BooleanConstraint {
    BooleanConstraint
}

/// Greater-than constraint `(lower_bound, inf)`.
pub fn greater_than<T: Float>(lower_bound: T) -> GreaterThan<T> {
    GreaterThan { lower_bound }
}

/// Greater-than-or-equal constraint `[lower_bound, inf)`.
pub fn greater_than_eq<T: Float>(lower_bound: T) -> GreaterThanEq<T> {
    GreaterThanEq { lower_bound }
}

/// Less-than constraint `(-inf, upper_bound)`.
pub fn less_than<T: Float>(upper_bound: T) -> LessThan<T> {
    LessThan { upper_bound }
}

/// Open interval constraint `(lower_bound, upper_bound)`.
pub fn open_interval<T: Float>(lower_bound: T, upper_bound: T) -> OpenInterval<T> {
    OpenInterval {
        lower_bound,
        upper_bound,
    }
}

/// Closed interval constraint `[lower_bound, upper_bound]`.
pub fn closed_interval<T: Float>(lower_bound: T, upper_bound: T) -> ClosedInterval<T> {
    ClosedInterval {
        lower_bound,
        upper_bound,
    }
}

/// Half-open interval constraint `[lower_bound, upper_bound)`.
pub fn half_open_interval<T: Float>(lower_bound: T, upper_bound: T) -> HalfOpenInterval<T> {
    HalfOpenInterval {
        lower_bound,
        upper_bound,
    }
}

/// Simplex constraint (non-negative, sum-to-one over event dimension).
pub fn simplex() -> Simplex {
    Simplex
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_real_accepts_finite() {
        let c = real();
        assert!(c.check(0.0f32));
        assert!(c.check(-1e30f32));
        assert!(c.check(1e30f32));
    }

    #[test]
    fn test_real_accepts_inf() {
        let c = real();
        assert!(c.check(f32::INFINITY));
        assert!(c.check(f32::NEG_INFINITY));
    }

    #[test]
    fn test_real_rejects_nan() {
        let c = real();
        assert!(!c.check(f32::NAN));
    }

    #[test]
    fn test_positive() {
        let c = positive();
        assert!(c.check(1e-7f32));
        assert!(c.check(100.0f32));
        assert!(!c.check(0.0f32));
        assert!(!c.check(-1.0f32));
    }

    #[test]
    fn test_nonnegative() {
        let c = nonnegative();
        assert!(c.check(0.0f32));
        assert!(c.check(1.0f32));
        assert!(!c.check(-1e-7f32));
    }

    #[test]
    fn test_unit_interval() {
        let c = unit_interval();
        assert!(c.check(0.0f32));
        assert!(c.check(0.5f32));
        assert!(c.check(1.0f32));
        assert!(!c.check(-0.01f32));
        assert!(!c.check(1.01f32));
    }

    #[test]
    fn test_boolean() {
        let c = boolean();
        assert!(c.check(0.0f32));
        assert!(c.check(1.0f32));
        assert!(!c.check(0.5f32));
        assert!(c.is_discrete());
    }

    #[test]
    fn test_greater_than() {
        let c = greater_than(5.0f32);
        assert!(c.check(5.1f32));
        assert!(!c.check(5.0f32));
        assert!(!c.check(4.9f32));
    }

    #[test]
    fn test_greater_than_eq() {
        let c = greater_than_eq(5.0f32);
        assert!(c.check(5.0f32));
        assert!(c.check(5.1f32));
        assert!(!c.check(4.9f32));
    }

    #[test]
    fn test_less_than() {
        let c = less_than(3.0f32);
        assert!(c.check(2.9f32));
        assert!(!c.check(3.0f32));
        assert!(!c.check(3.1f32));
    }

    #[test]
    fn test_open_interval() {
        let c = open_interval(0.0f32, 1.0f32);
        assert!(c.check(0.5f32));
        assert!(!c.check(0.0f32));
        assert!(!c.check(1.0f32));
    }

    #[test]
    fn test_closed_interval() {
        let c = closed_interval(0.0f32, 1.0f32);
        assert!(c.check(0.0f32));
        assert!(c.check(0.5f32));
        assert!(c.check(1.0f32));
        assert!(!c.check(-0.01f32));
        assert!(!c.check(1.01f32));
    }

    #[test]
    fn test_half_open_interval() {
        let c = half_open_interval(0.0f32, 1.0f32);
        assert!(c.check(0.0f32));
        assert!(c.check(0.5f32));
        assert!(!c.check(1.0f32));
        assert!(!c.check(-0.01f32));
    }

    #[test]
    fn test_simplex_nonneg() {
        let c = simplex();
        assert!(c.check(0.0f32));
        assert!(c.check(0.5f32));
        assert!(!c.check(-0.1f32));
        assert_eq!(c.event_dim(), 1);
    }

    #[test]
    fn test_constraint_traits() {
        // Verify default trait method values.
        let c = real();
        assert!(!c.is_discrete());
        assert_eq!(c.event_dim(), 0);
        assert_eq!(c.name(), "Real");
    }

    #[test]
    fn test_f64_constraints() {
        assert!(positive().check(1.0f64));
        assert!(!positive().check(-1.0f64));
        assert!(unit_interval().check(0.5f64));
        assert!(greater_than(0.0f64).check(0.001f64));
    }
}
