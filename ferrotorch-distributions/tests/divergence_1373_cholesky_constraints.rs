//! Divergence: ferrotorch's `CorrCholesky` / `LowerCholesky` constraints
//! (shipped in commit `a2ab04347`, #1373) implement `check` as `!is_nan()`,
//! so their default `check_tensor` accepts ANY finite matrix — including ones
//! that are NOT valid Cholesky factors. This is the tautology bug the audit
//! charter calls out: a `Constraint::check` that always returns true.
//!
//! Upstream `torch.distributions.constraints.corr_cholesky` /
//! `.lower_cholesky` REJECT invalid matrices. Expected values OracleDerived
//! from live torch 2.11.0:
//!
//!   corr_cholesky.check([[1,0,0],[0.5,0.5,0],[0.3,0.3,0.3]]) == False
//!     (rows not unit Euclidean norm)
//!   lower_cholesky.check([[1,0.3],[0.5,2.0]]) == False  (nonzero upper tri)
//!   lower_cholesky.check([[1,0],[0.5,-2.0]]) == False  (negative diagonal)
//!
//! See `torch/distributions/constraints.py:_CorrCholesky.check` and
//! `_LowerCholesky.check`. Tracking: #1568.

use ferrotorch_core::creation::from_slice;
use ferrotorch_distributions::constraints::{Constraint, CorrCholesky, LowerCholesky};

/// Divergence: `CorrCholesky::check_tensor` accepts a matrix whose rows are not
/// unit-norm; torch `corr_cholesky.check` returns `False`.
/// Upstream `torch/distributions/constraints.py:_CorrCholesky.check`.
/// ferrotorch `constraints.rs:448-460` returns `!value.is_nan()`.
/// Tracking: #1568.
#[test]
fn divergence_corr_cholesky_constraint_rejects_non_unit_norm_rows() {
    let c = CorrCholesky;
    // Lower-triangular but rows are NOT unit Euclidean norm -> not a corr-chol.
    let bad = from_slice(&[1.0f32, 0.0, 0.0, 0.5, 0.5, 0.0, 0.3, 0.3, 0.3], &[3, 3]).unwrap();
    // torch returns False here.
    assert!(
        !c.check_tensor(&bad).unwrap(),
        "CorrCholesky::check_tensor accepted a non-unit-norm matrix; torch rejects it"
    );
}

/// Divergence: `LowerCholesky::check_tensor` accepts a matrix with a nonzero
/// upper-triangular entry; torch `lower_cholesky.check` returns `False`.
/// Upstream `torch/distributions/constraints.py:_LowerCholesky.check`.
/// ferrotorch `constraints.rs:473-485` returns `!value.is_nan()`.
/// Tracking: #1568.
#[test]
fn divergence_lower_cholesky_constraint_rejects_upper_triangle() {
    let c = LowerCholesky;
    // Nonzero (0,1) entry -> not lower-triangular.
    let bad = from_slice(&[1.0f32, 0.3, 0.5, 2.0], &[2, 2]).unwrap();
    assert!(
        !c.check_tensor(&bad).unwrap(),
        "LowerCholesky::check_tensor accepted a non-lower-triangular matrix; torch rejects it"
    );
}

/// Divergence: `LowerCholesky::check_tensor` accepts a matrix with a negative
/// diagonal entry; torch `lower_cholesky.check` returns `False`.
/// Upstream `torch/distributions/constraints.py:_LowerCholesky.check`.
/// Tracking: #1568.
#[test]
fn divergence_lower_cholesky_constraint_rejects_negative_diagonal() {
    let c = LowerCholesky;
    // Diagonal (1,1) = -2.0 -> not a valid Cholesky factor.
    let bad = from_slice(&[1.0f32, 0.0, 0.5, -2.0], &[2, 2]).unwrap();
    assert!(
        !c.check_tensor(&bad).unwrap(),
        "LowerCholesky::check_tensor accepted a negative-diagonal matrix; torch rejects it"
    );
}

/// Control: a genuinely valid corr-cholesky factor must be accepted by both
/// torch (True) and ferrotorch. Pins that the fix must not over-reject.
/// torch corr_cholesky.check([[1,0,0],[0.197..,0.980..,0],[-0.462..,0.536..,0.707..]]) == True
#[test]
#[allow(
    clippy::excessive_precision,
    reason = "literals are OracleDerived from live torch 2.11.0; truncating them \
              changes the pinned expected factor (R-CHAR-3 named typed bits)"
)]
fn corr_cholesky_constraint_accepts_valid_factor() {
    let c = CorrCholesky;
    let good = from_slice(
        &[
            1.0f32,
            0.0,
            0.0,
            0.19737533,
            0.98032802,
            0.0,
            -0.46211717,
            0.53596479,
            0.70653343,
        ],
        &[3, 3],
    )
    .unwrap();
    assert!(
        c.check_tensor(&good).unwrap(),
        "CorrCholesky::check_tensor rejected a valid corr-cholesky factor"
    );
}
