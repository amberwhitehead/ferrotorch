//! Red-then-green regression tests for audit finding CORE-172 (crosslink
//! #1866): `beta` computes `exp(ln|B|)` and therefore drops the sign for
//! negative arguments — `beta(-0.5, 1.5)` returned `+π` where the declared
//! oracle `scipy.special.beta` (cephes `beta.c`) returns `-π`.
//!
//! Oracle (R-ORACLE-1 path (b)) — live scipy 1.17.1 session, 2026-06-11,
//! this machine (scipy is the module's documented oracle for `beta`; torch
//! ships no `torch.special.beta`):
//!
//! ```python
//! >>> import scipy.special as sp
//! >>> sp.beta(-0.5, 1.5)   # -3.1415926535897927
//! >>> sp.beta(-1.5, 2.5)   # 3.1415926535897936
//! >>> sp.beta(-0.5, 0.5)   # -0.0
//! >>> sp.beta(-0.5, -0.5)  # 0.0
//! >>> sp.beta(-2.5, 0.5)   # -0.0
//! >>> sp.beta(-1.0, 0.5)   # inf
//! >>> sp.beta(-1.0, 2.0)   # inf
//! >>> sp.beta(-1.0, -2.0)  # inf
//! >>> sp.beta(-3.0, 2.0)   # 0.16666666666666666
//! >>> sp.beta(-3.0, 1.0)   # -0.3333333333333333
//! >>> sp.beta(-4.0, 2.0)   # 0.08333333333333333
//! >>> sp.beta(2.0, 3.0)    # 0.08333333333333333
//! ```
//!
//! Tolerance justification (R-ORACLE-5): finite non-zero pins use relative
//! error ≤ 1e-12 — the implementation composes three lgamma values, each
//! ~1e-15 relative for O(1) arguments, so 1e-12 gives ≥100× margin while
//! sitting 12 orders below the sign-flip divergence being pinned. Zero,
//! signed-zero, and infinity pins are exact (no tolerance applies).

use ferrotorch_core::special::beta;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

fn beta1(a: f64, b: f64) -> f64 {
    let ta = Tensor::from_storage(TensorStorage::cpu(vec![a]), vec![1], false).unwrap();
    let tb = Tensor::from_storage(TensorStorage::cpu(vec![b]), vec![1], false).unwrap();
    beta(&ta, &tb).unwrap().data().unwrap()[0]
}

fn assert_rel(got: f64, expected: f64, what: &str) {
    let rel = ((got - expected) / expected).abs();
    assert!(
        rel <= 1e-12,
        "{what}: got {got:e}, scipy oracle {expected:e}, rel err {rel:e} > 1e-12"
    );
}

#[test]
fn core172_beta_negative_noninteger_signed() {
    // The headline divergence: sgn(Γ(-0.5)) = -1 must survive.
    assert_rel(beta1(-0.5, 1.5), -std::f64::consts::PI, "beta(-0.5, 1.5)");
    // Even-floor negative arg: sign is +1 — guards against over-correcting.
    assert_rel(beta1(-1.5, 2.5), std::f64::consts::PI, "beta(-1.5, 2.5)");
}

#[test]
fn core172_beta_denominator_pole_signed_zero() {
    // Γ(a+b) pole ⇒ |B| = 0; the sign of the zero follows the numerator
    // sign product (cephes lgam_sgn semantics).
    let v = beta1(-0.5, 0.5); // scipy: -0.0
    assert!(
        v == 0.0 && v.is_sign_negative(),
        "beta(-0.5, 0.5) must be -0.0, got {v:?}"
    );
    let v = beta1(-2.5, 0.5); // scipy: -0.0
    assert!(
        v == 0.0 && v.is_sign_negative(),
        "beta(-2.5, 0.5) must be -0.0, got {v:?}"
    );
    let v = beta1(-0.5, -0.5); // scipy: 0.0
    assert!(
        v == 0.0 && v.is_sign_positive(),
        "beta(-0.5, -0.5) must be +0.0, got {v:?}"
    );
}

#[test]
fn core172_beta_negint_branch() {
    // Non-positive-integer arguments follow cephes beta_negint: +inf unless
    // the other operand is an integer with 1 - a - b > 0, in which case the
    // pole ratio has the finite limit (-1)^b · B(1-a-b, b).
    assert!(
        beta1(-1.0, 0.5).is_infinite() && beta1(-1.0, 0.5) > 0.0,
        "beta(-1, 0.5) must be +inf, got {}",
        beta1(-1.0, 0.5)
    );
    assert!(
        beta1(-1.0, 2.0).is_infinite() && beta1(-1.0, 2.0) > 0.0,
        "beta(-1, 2) must be +inf, got {}",
        beta1(-1.0, 2.0)
    );
    assert!(
        beta1(-1.0, -2.0).is_infinite() && beta1(-1.0, -2.0) > 0.0,
        "beta(-1, -2) must be +inf, got {}",
        beta1(-1.0, -2.0)
    );
    assert_rel(beta1(-3.0, 2.0), 1.0 / 6.0, "beta(-3, 2)");
    assert_rel(beta1(-3.0, 1.0), -1.0 / 3.0, "beta(-3, 1)");
    assert_rel(beta1(-4.0, 2.0), 1.0 / 12.0, "beta(-4, 2)");
}

#[test]
fn core172_beta_positive_args_unchanged() {
    assert_rel(beta1(2.0, 3.0), 1.0 / 12.0, "beta(2, 3)");
    assert_rel(beta1(0.5, 2.5), 1.1780972450961724, "beta(0.5, 2.5)"); // = exp(betaln) = 3π/8
}
