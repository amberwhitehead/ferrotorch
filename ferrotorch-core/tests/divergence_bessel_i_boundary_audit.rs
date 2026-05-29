//! Discriminator audit of the i0/i0e/i1/i1e A/B Chebyshev-set boundary at
//! |x| == 8 (the `x <= 8` switch in `aten/src/ATen/native/cuda/Math.cuh:505`
//! / `aten/src/ATen/native/Math.h:106`) and the large-x exp-scaled finiteness.
//!
//! The shipped `i_family_boundary_at_8_vs_torch` (in special.rs) only samples
//! 8.0 / 8.5 / 12.0. This audit pins the *transition* tightly: 7.99 (A-set),
//! 8.0 (A-set, the <= edge), 8.01 (B-set). A wrong split (`x < 8` instead of
//! `x <= 8`, or an off-by-one in the inverted-interval argument 32/x - 2) would
//! show as a discontinuity here that does not match torch.
//!
//! All expected values are LIVE torch 2.11.0+cu130 outputs (R-CHAR-3):
//!   python3 -c "import torch; print(torch.special.i0(
//!     torch.tensor([7.99,8.0,8.01], dtype=torch.float64)).tolist())"
//! Upstream: pytorch 2ec0222669f1bcd37b5670ce384f8608c033b158.

use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;
use ferrotorch_core::{i0, i0e, i1, i1e};

fn t(data: &[f64]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false)
        .expect("from_storage")
}

fn assert_rel(got: f64, want: f64, tol: f64, ctx: &str) {
    assert!(
        (got - want).abs() <= tol * (1.0 + want.abs()),
        "{ctx}: got {got} want {want} (tol {tol})"
    );
}

/// torch.special.i0/i0e/i1/i1e across the |x|==8 A/B Chebyshev-set boundary.
/// Live torch f64 (oracle above). The A-set covers x<=8; 8.01 crosses to B-set.
#[test]
fn divergence_bessel_boundary_at_8() {
    let xs = [7.99_f64, 8.0, 8.01];
    let input = t(&xs);

    let want_i0 = [423.58420367467363, 427.56411572180474, 431.58178605209235];
    let d = i0(&input).unwrap().data_vec().unwrap();
    for k in 0..3 {
        assert_rel(d[k], want_i0[k], 1e-11, &format!("i0(x={})", xs[k]));
    }

    let want_i0e = [0.14352476537933484, 0.1434317818568503, 0.1433389794111385];
    let d = i0e(&input).unwrap().data_vec().unwrap();
    for k in 0..3 {
        assert_rel(d[k], want_i0e[k], 1e-12, &format!("i0e(x={})", xs[k]));
    }

    let want_i1 = [396.11522620740124, 399.8731367825599, 403.66693999770416];
    let d = i1(&input).unwrap().data_vec().unwrap();
    for k in 0..3 {
        assert_rel(d[k], want_i1[k], 1e-11, &format!("i1(x={})", xs[k]));
    }

    let want_i1e = [
        0.13421733957828097,
        0.13414249329269812,
        0.13406776900984482,
    ];
    let d = i1e(&input).unwrap().data_vec().unwrap();
    for k in 0..3 {
        assert_rel(d[k], want_i1e[k], 1e-12, &format!("i1e(x={})", xs[k]));
    }
}

/// Negative-x odd/even at the boundary: i0/i0e EVEN, i1/i1e ODD.
/// Live torch f64: i1(-8) == -399.8731367825599 (negation of +8).
#[test]
fn divergence_bessel_boundary_at_8_negative() {
    let neg = t(&[-7.99_f64, -8.0, -8.01]);

    let want_i0 = [423.58420367467363, 427.56411572180474, 431.58178605209235];
    let d = i0(&neg).unwrap().data_vec().unwrap();
    for k in 0..3 {
        assert_rel(d[k], want_i0[k], 1e-11, "i0 even");
    }
    let want_i0e = [0.14352476537933484, 0.1434317818568503, 0.1433389794111385];
    let d = i0e(&neg).unwrap().data_vec().unwrap();
    for k in 0..3 {
        assert_rel(d[k], want_i0e[k], 1e-12, "i0e even");
    }

    let want_i1 = [-396.11522620740124, -399.8731367825599, -403.66693999770416];
    let d = i1(&neg).unwrap().data_vec().unwrap();
    for k in 0..3 {
        assert_rel(d[k], want_i1[k], 1e-11, "i1 odd");
    }
    let want_i1e = [
        -0.13421733957828097,
        -0.13414249329269812,
        -0.13406776900984482,
    ];
    let d = i1e(&neg).unwrap().data_vec().unwrap();
    for k in 0..3 {
        assert_rel(d[k], want_i1e[k], 1e-12, "i1e odd");
    }
}

/// Large-x: torch.special.i0(700) == 1.5295933476718735e+302 (FINITE, not inf);
/// i0e/i1e stay O(0.015). The shipped i_family_large_x_scaled_finite_vs_torch
/// only asserts `i0(700) > 1e300`; this pins the exact finite torch value so a
/// scaling-factor regression that overflowed to +inf would be caught.
/// Live torch f64 oracle (torch 2.11.0+cu130).
#[test]
fn divergence_bessel_large_x_i0_finite() {
    let input = t(&[700.0_f64, -700.0, 100.0]);

    let d = i0(&input).unwrap().data_vec().unwrap();
    assert!(
        d[0].is_finite(),
        "i0(700) must be FINITE (torch=1.53e302), got {}",
        d[0]
    );
    assert_rel(d[0], 1.5295933476718735e+302, 1e-11, "i0(700)");
    assert_rel(d[1], 1.5295933476718735e+302, 1e-11, "i0(-700) even");
    assert_rel(d[2], 1.0737517071310738e+42, 1e-11, "i0(100)");

    let d = i1(&input).unwrap().data_vec().unwrap();
    assert!(
        d[0].is_finite(),
        "i1(700) must be FINITE (torch=1.53e302), got {}",
        d[0]
    );
    assert_rel(d[0], 1.5285003902339006e+302, 1e-11, "i1(700)");
    assert_rel(d[1], -1.5285003902339006e+302, 1e-11, "i1(-700) odd");

    let d = i0e(&input).unwrap().data_vec().unwrap();
    assert_rel(d[0], 0.015081295651531355, 1e-12, "i0e(700)");
    assert_rel(d[2], 0.03994437929909668, 1e-12, "i0e(100)");

    let d = i1e(&input).unwrap().data_vec().unwrap();
    assert_rel(d[0], 0.015070519444716846, 1e-12, "i1e(700)");
    assert_rel(d[1], -0.015070519444716846, 1e-12, "i1e(-700) odd");
}

/// i1e(-inf) == -0.0 (SIGNED zero) in torch; i0e(+/-inf) == +0.0.
/// Verifies the odd-function sign flip is applied to the limiting zero.
#[test]
fn divergence_bessel_inf_signed_zero() {
    let input = t(&[f64::INFINITY, f64::NEG_INFINITY]);

    let d = i0(&input).unwrap().data_vec().unwrap();
    assert!(d[0].is_nan() && d[1].is_nan(), "i0(+/-inf) == NaN");
    let d = i1(&input).unwrap().data_vec().unwrap();
    assert!(d[0].is_nan() && d[1].is_nan(), "i1(+/-inf) == NaN");

    let d = i0e(&input).unwrap().data_vec().unwrap();
    assert_eq!(d[0], 0.0, "i0e(+inf) == 0");
    assert_eq!(d[1], 0.0, "i0e(-inf) == 0");

    let d = i1e(&input).unwrap().data_vec().unwrap();
    assert_eq!(d[0], 0.0, "i1e(+inf) == +0");
    // torch.special.i1e(-inf) == -0.0; ferrotorch applies the odd sign flip.
    assert_eq!(d[1], 0.0, "i1e(-inf) == -0 (numerically 0)");
    assert!(
        d[1].is_sign_negative(),
        "i1e(-inf) is -0.0 (signed), got {}",
        d[1]
    );
}
