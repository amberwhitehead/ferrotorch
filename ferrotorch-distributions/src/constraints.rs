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
//! | REQ-7 (`Simplex` w/ `event_dim() -> 1` + full-vector `check_tensor`) | SHIPPED | `pub struct Simplex` with `event_dim() -> 1` + `check_tensor` override (`all(value >= 0, dim=-1) & ((value.sum(-1) - 1).abs() < 1e-6)`) + `simplex()` constructor in `constraints.rs` mirroring `torch/distributions/constraints.py:_Simplex.check`; consumer: `fn Dirichlet::log_prob` in `dirichlet.rs` validates the sample against `Simplex::check_tensor` (#1547). |
//! | REQ-8 (17 missing upstream constraint variants) | PARTIAL | wave-H #1372 — `IntegerInterval`/`NonNegativeInteger` added to `constraints.rs` mirroring `torch/distributions/constraints.py:_IntegerInterval` / `nonnegative_integer = _IntegerGreaterThan(0)`; consumer: `pub use constraints` re-export in `lib.rs`. Remaining 15 variants (`PositiveDefinite`, `PositiveSemiDefinite`, `Multinomial`, `OneHot`, `Symmetric`, `LowerCholesky`, `LowerTriangular`, `CorrCholesky`, `RealVector`, `Cat`, `Stack`, `Independent` composite, `_Dependent`/`is_dependent`, `MixtureSameFamilyConstraint`) remain under #1372 follow-up — they each need a paired concrete-distribution consumer (e.g. `MultivariateNormal.support = real_vector`) which is cross-cutting. |
//! | REQ-9 (concrete `arg_constraints` wiring) | SHIPPED | trait extension #1376 (commit `ff14fe66b`) added `support`/`arg_constraints` overrides to 20 concrete distributions (`normal.rs:296-308`, `bernoulli.rs`, `beta.rs`, `categorical.rs`, `cauchy.rs`, `exponential.rs`, `gamma.rs`, `gumbel.rs`, `half_normal.rs`, `independent.rs`, `laplace.rs`, `lognormal.rs`, `pareto.rs`, `poisson.rs`, `relaxed_bernoulli.rs`, `relaxed_one_hot_categorical.rs`, `student_t.rs`, `uniform.rs`, `von_mises.rs`, `weibull.rs`); consumer per #1376: production dispatch through `Distribution::support`/`arg_constraints` defaults exposed at `lib.rs:344-356`. Closes #1371. |

use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::FerrotorchResult;
use ferrotorch_core::tensor::Tensor;

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

    /// Returns `true` if the whole `value` tensor satisfies this constraint.
    ///
    /// Mirrors PyTorch's `Constraint.check(value)` which operates on a
    /// tensor and returns a boolean tensor
    /// (`torch/distributions/constraints.py:80-106`). The scalar
    /// [`check`](Constraint::check) method above is the ferrotorch-specific
    /// per-element predicate; this method is the tensor-level reduction.
    ///
    /// The default implementation applies the scalar `check` element-wise and
    /// AND-reduces — correct for every `event_dim() == 0` constraint. The
    /// [`Simplex`] constraint overrides it because its validity is a
    /// *vector* property (non-negativity AND sum-to-one over the last dim)
    /// that the scalar predicate cannot express.
    fn check_tensor<T: Float>(&self, value: &Tensor<T>) -> FerrotorchResult<bool> {
        let data = value.data_vec()?;
        Ok(data.iter().all(|&x| self.check(x)))
    }

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
/// A vector lies on the simplex iff every element is non-negative AND the
/// last dimension sums to 1. The scalar [`check`](Constraint::check) verifies
/// only the per-element non-negativity half; full validation (including
/// sum-to-one) is the tensor-level [`check_tensor`](Constraint::check_tensor)
/// override below, which mirrors upstream's vector `_Simplex.check`.
#[derive(Debug, Clone, Copy)]
pub struct Simplex;

impl Constraint for Simplex {
    fn check<T: Float>(&self, value: T) -> bool {
        // Per-element: must be non-negative. The sum-to-one half of the
        // simplex contract is a vector property — see `check_tensor`.
        value >= T::from(0.0).unwrap()
    }

    fn check_tensor<T: Float>(&self, value: &Tensor<T>) -> FerrotorchResult<bool> {
        // Mirrors `torch/distributions/constraints.py:_Simplex.check`:
        //   return torch.all(value >= 0, dim=-1) & ((value.sum(-1) - 1).abs() < 1e-6)
        // We reduce over the trailing event dim: split the flat buffer into
        // `n_rows` rows of length `k` (k = last dim) and require every row to
        // be non-negative and sum to 1 within tolerance. A row that fails
        // makes the whole tensor invalid (the `all` over batch rows).
        let shape = value.shape();
        let k = match shape.last().copied() {
            Some(k) if k > 0 => k,
            // A 0-D or trailing-zero tensor cannot lie on a simplex.
            _ => return Ok(false),
        };
        let data = value.data_vec()?;
        let zero = T::from(0.0).unwrap();
        let one = T::from(1.0).unwrap();
        // 1e-6 absolute tolerance, matching upstream's `< 1e-6`.
        let tol = T::from(1e-6).unwrap();
        for row in data.chunks(k) {
            let mut sum = zero;
            for &x in row {
                if x < zero {
                    return Ok(false);
                }
                sum += x;
            }
            if (sum - one).abs() >= tol {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn event_dim(&self) -> usize {
        1
    }

    fn name(&self) -> &'static str {
        "Simplex"
    }
}

/// Constraint for a closed integer interval `[lower_bound, upper_bound]`.
///
/// `check` accepts an `f32`/`f64` value and verifies that it is an integer
/// (i.e. `value.fract() == 0`) within the bounds. Mirrors
/// `torch/distributions/constraints.py:_IntegerInterval` (lines 354-376).
#[derive(Debug, Clone, Copy)]
pub struct IntegerInterval<T: Float> {
    /// The inclusive lower bound.
    pub lower_bound: T,
    /// The inclusive upper bound.
    pub upper_bound: T,
}

impl<S: Float> Constraint for IntegerInterval<S> {
    fn check<T: Float>(&self, value: T) -> bool {
        let lo = T::from(self.lower_bound).unwrap();
        let hi = T::from(self.upper_bound).unwrap();
        let zero = T::from(0.0).unwrap();
        value.fract() == zero && value >= lo && value <= hi
    }

    fn is_discrete(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "IntegerInterval"
    }
}

/// Constraint for non-negative integers `{0, 1, 2, ...}`.
///
/// Mirrors `torch/distributions/constraints.py:738` —
/// `nonnegative_integer = _IntegerGreaterThan(0)`.
#[derive(Debug, Clone, Copy)]
pub struct NonNegativeInteger;

impl Constraint for NonNegativeInteger {
    fn check<T: Float>(&self, value: T) -> bool {
        let zero = T::from(0.0).unwrap();
        value.fract() == zero && value >= zero
    }

    fn is_discrete(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "NonNegativeInteger"
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

/// Integer interval constraint `[lower_bound, upper_bound]`.
pub fn integer_interval<T: Float>(lower_bound: T, upper_bound: T) -> IntegerInterval<T> {
    IntegerInterval {
        lower_bound,
        upper_bound,
    }
}

/// Non-negative integer constraint `{0, 1, 2, ...}`.
pub fn nonnegative_integer() -> NonNegativeInteger {
    NonNegativeInteger
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
    fn test_integer_interval_accepts_integers_in_range() {
        let c = integer_interval(0.0f32, 5.0f32);
        assert!(c.check(0.0f32));
        assert!(c.check(3.0f32));
        assert!(c.check(5.0f32));
        assert!(!c.check(-1.0f32));
        assert!(!c.check(6.0f32));
        assert!(!c.check(2.5f32));
        assert!(c.is_discrete());
        assert_eq!(c.name(), "IntegerInterval");
    }

    #[test]
    fn test_nonnegative_integer() {
        let c = nonnegative_integer();
        assert!(c.check(0.0f32));
        assert!(c.check(42.0f32));
        assert!(!c.check(-1.0f32));
        assert!(!c.check(2.5f32));
        assert!(c.is_discrete());
        assert_eq!(c.name(), "NonNegativeInteger");
    }

    #[test]
    fn test_simplex_check_tensor_full_vector() {
        use ferrotorch_core::creation::{from_slice, tensor};
        let c = simplex();
        // Valid simplex vector (sum == 1, all >= 0).
        assert!(
            c.check_tensor(&tensor(&[0.2f32, 0.5, 0.3]).unwrap())
                .unwrap()
        );
        // Sum == 1.1 -> rejected (this is the sum-to-one half the scalar
        // `check` cannot catch; mirrors constraints.py:_Simplex.check).
        assert!(
            !c.check_tensor(&tensor(&[0.2f32, 0.5, 0.4]).unwrap())
                .unwrap()
        );
        // Negative element -> rejected.
        assert!(
            !c.check_tensor(&tensor(&[-0.1f32, 0.6, 0.5]).unwrap())
                .unwrap()
        );
        // Batched: both rows valid -> accepted; one bad row -> rejected.
        let ok = from_slice(&[0.5f32, 0.5, 0.25, 0.75], &[2, 2]).unwrap();
        assert!(c.check_tensor(&ok).unwrap());
        let bad = from_slice(&[0.5f32, 0.5, 0.25, 0.80], &[2, 2]).unwrap();
        assert!(!c.check_tensor(&bad).unwrap());
    }

    #[test]
    fn test_check_tensor_default_elementwise() {
        use ferrotorch_core::creation::tensor;
        // Default `check_tensor` AND-reduces the scalar `check`.
        let c = positive();
        assert!(
            c.check_tensor(&tensor(&[1.0f32, 2.0, 3.0]).unwrap())
                .unwrap()
        );
        assert!(
            !c.check_tensor(&tensor(&[1.0f32, -2.0, 3.0]).unwrap())
                .unwrap()
        );
    }

    #[test]
    fn test_f64_constraints() {
        assert!(positive().check(1.0f64));
        assert!(!positive().check(-1.0f64));
        assert!(unit_interval().check(0.5f64));
        assert!(greater_than(0.0f64).check(0.001f64));
    }
}
