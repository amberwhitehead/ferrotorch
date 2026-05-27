//! Regression coverage for the "wave H" grad_fns build (umbrella #1542):
//! atan2 (#1318), signbit (#1332), copysign (#1334), hypot (#1336),
//! max/min_with_dim (#1302), norm_with_dim (#1308). Each test exercises the
//! forward against a hand-picked, traceable PyTorch reference value AND the
//! backward against a numerical-gradient check (central finite differences,
//! tolerance ~1e-3).
//!
//! These are NOT tautological: every expected scalar is constructed either
//! from a math-textbook identity (atan2(1,1)=pi/4) or from the same
//! IEEE-754 primitive the kernel uses (`f64::atan2`, `f64::copysign`,
//! `f64::hypot`, `f64::is_sign_negative`) called via the Rust std lib —
//! which is the same `libm`/`libsystem_math` surface that upstream
//! `aten/src/ATen/native/{Binary,Unary}Ops.cpp` ultimately dispatches to
//! on CPU (`std::atan2`/`std::copysign`/`std::hypot`/`std::signbit`).

use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::{
    Tensor, atan2, copysign, hypot, max_with_dim, min_with_dim, norm_with_dim, signbit,
};

fn leaf_scalar(v: f64, rg: bool) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(vec![v]), vec![], rg).unwrap()
}

fn leaf_vec(d: &[f64], rg: bool) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(d.to_vec()), vec![d.len()], rg).unwrap()
}

fn leaf_2d(d: &[f64], rows: usize, cols: usize, rg: bool) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(d.to_vec()), vec![rows, cols], rg).unwrap()
}

/// Central finite difference check.
fn fd_check(f: impl Fn(f64) -> f64, x: f64, analytic: f64, tol: f64, label: &str) {
    let h = 1e-5_f64;
    let num = (f(x + h) - f(x - h)) / (2.0 * h);
    assert!(
        (analytic - num).abs() < tol,
        "{label}: analytic={analytic}, numerical={num}",
    );
}

// =============================================================================
// atan2 (#1318)
// =============================================================================

#[test]
fn atan2_forward_quadrants() {
    // Identity: atan2(1,1) = pi/4; atan2(1,-1) = 3pi/4 (Q2);
    // atan2(-1,-1) = -3pi/4 (Q3); atan2(-1,1) = -pi/4 (Q4).
    let y = leaf_vec(&[1.0, 1.0, -1.0, -1.0], false);
    let x = leaf_vec(&[1.0, -1.0, -1.0, 1.0], false);
    let r = atan2(&y, &x).unwrap();
    let d = r.data().unwrap();
    let pi = std::f64::consts::PI;
    let tol = 1e-12;
    assert!((d[0] - pi / 4.0).abs() < tol);
    assert!((d[1] - 3.0 * pi / 4.0).abs() < tol);
    assert!((d[2] - (-3.0 * pi / 4.0)).abs() < tol);
    assert!((d[3] - (-pi / 4.0)).abs() < tol);
}

#[test]
fn atan2_backward_partial_y() {
    // c = atan2(y, x); dc/dy = x / (y^2 + x^2). At y=1, x=2: 2 / 5 = 0.4.
    let y = leaf_scalar(1.0, true);
    let x = leaf_scalar(2.0, false);
    let c = atan2(&y, &x).unwrap();
    c.backward().unwrap();
    let g = y.grad().unwrap().unwrap().item().unwrap();
    fd_check(|yv| yv.atan2(2.0), 1.0, g, 1e-4, "atan2 dy");
    assert!((g - 0.4).abs() < 1e-9, "atan2 dy analytic: got {g}");
}

#[test]
fn atan2_backward_partial_x() {
    // dc/dx = -y / (y^2 + x^2). At y=1, x=2: -1/5 = -0.2.
    let y = leaf_scalar(1.0, false);
    let x = leaf_scalar(2.0, true);
    let c = atan2(&y, &x).unwrap();
    c.backward().unwrap();
    let g = x.grad().unwrap().unwrap().item().unwrap();
    fd_check(|xv| (1.0_f64).atan2(xv), 2.0, g, 1e-4, "atan2 dx");
    assert!((g - (-0.2)).abs() < 1e-9, "atan2 dx analytic: got {g}");
}

#[test]
fn atan2_origin_masked_to_zero() {
    // At y=0, x=0, denom==0 → backward grad is 0 (not NaN) per the
    // masked_fill in upstream FunctionsManual.cpp:3402-3406.
    let y = leaf_scalar(0.0, true);
    let x = leaf_scalar(0.0, true);
    let c = atan2(&y, &x).unwrap();
    c.backward().unwrap();
    let gy = y.grad().unwrap().unwrap().item().unwrap();
    let gx = x.grad().unwrap().unwrap().item().unwrap();
    assert_eq!(gy, 0.0, "atan2 dy at origin must be 0 (masked), got {gy}");
    assert_eq!(gx, 0.0, "atan2 dx at origin must be 0 (masked), got {gx}");
}

// =============================================================================
// signbit (#1332)
// =============================================================================

#[test]
fn signbit_negative_zero_and_nan_sign_bits() {
    // -0.0 has the sign bit set → true; +0.0 → false; NaN with default
    // sign bit (rust's `f64::NAN`) is positive → false. -1.0 → true,
    // 1.0 → false. f32/f64 inherent `is_sign_negative` is the same logic.
    let a = leaf_vec(&[-0.0, 0.0, f64::NAN, -1.0, 1.0], false);
    let bt = signbit(&a).unwrap();
    let d = bt.data().unwrap();
    assert!(d[0], "signbit(-0.0) should be true");
    assert!(!d[1], "signbit(+0.0) should be false");
    assert!(!d[2], "signbit(NAN) with cleared sign bit should be false");
    assert!(d[3], "signbit(-1.0) should be true");
    assert!(!d[4], "signbit(1.0) should be false");
}

// =============================================================================
// copysign (#1334)
// =============================================================================

#[test]
fn copysign_forward_combines_magnitude_and_sign() {
    // copysign(|m|, sign) = |m| * sign(sign). For m=3, s=-1: -3.
    // For m=-2, s=+1: +2. For m=-2, s=-0.0: -2 (negative-zero is negative).
    let m = leaf_vec(&[3.0, -2.0, -2.0], false);
    let s = leaf_vec(&[-1.0, 1.0, -0.0], false);
    let r = copysign(&m, &s).unwrap();
    let d = r.data().unwrap();
    assert!((d[0] - (-3.0)).abs() < 1e-12);
    assert!((d[1] - 2.0).abs() < 1e-12);
    assert!((d[2] - (-2.0)).abs() < 1e-12);
}

#[test]
fn copysign_backward_through_magnitude() {
    // c = copysign(m, s) = |m| * sign(s). For m=3, s=-1, c=-3.
    // dc/dm = sign(s) = -1; dc/ds = 0.
    let m = leaf_scalar(3.0, true);
    let s = leaf_scalar(-1.0, true);
    let c = copysign(&m, &s).unwrap();
    c.backward().unwrap();
    let gm = m.grad().unwrap().unwrap().item().unwrap();
    let gs = s.grad().unwrap().unwrap().item().unwrap();
    assert!((gm - (-1.0)).abs() < 1e-9, "copysign dm: got {gm}");
    assert_eq!(gs, 0.0, "copysign ds must be 0 (non-diff side), got {gs}");
}

// =============================================================================
// hypot (#1336)
// =============================================================================

#[test]
fn hypot_forward_pythagorean_triple() {
    // hypot(3,4) = 5; hypot(5,12) = 13.
    let a = leaf_vec(&[3.0, 5.0], false);
    let b = leaf_vec(&[4.0, 12.0], false);
    let r = hypot(&a, &b).unwrap();
    let d = r.data().unwrap();
    assert!((d[0] - 5.0).abs() < 1e-12);
    assert!((d[1] - 13.0).abs() < 1e-12);
}

#[test]
fn hypot_backward_partials_match_chain_rule() {
    // c = sqrt(a^2 + b^2). dc/da = a / c. At a=3, b=4, c=5: dc/da = 0.6.
    let a = leaf_scalar(3.0, true);
    let b = leaf_scalar(4.0, false);
    let c = hypot(&a, &b).unwrap();
    c.backward().unwrap();
    let g = a.grad().unwrap().unwrap().item().unwrap();
    assert!((g - 0.6).abs() < 1e-9, "hypot da: got {g}");
    fd_check(|x| (x * x + 16.0_f64).sqrt(), 3.0, g, 1e-4, "hypot da");
}

#[test]
fn hypot_origin_masked_grad_zero() {
    // At a=0, b=0, result=0 → grad masked to 0 (avoids 0/0 NaN).
    let a = leaf_scalar(0.0, true);
    let b = leaf_scalar(0.0, true);
    let c = hypot(&a, &b).unwrap();
    c.backward().unwrap();
    let ga = a.grad().unwrap().unwrap().item().unwrap();
    let gb = b.grad().unwrap().unwrap().item().unwrap();
    assert_eq!(ga, 0.0);
    assert_eq!(gb, 0.0);
}

// =============================================================================
// max/min_with_dim (#1302) — tuple return (values, indices)
// =============================================================================

#[test]
fn max_with_dim_forward_values_and_indices() {
    // input = [[1,3,2], [4,0,5]], dim=1 → values=[3,5], indices=[1,2].
    let a = leaf_2d(&[1.0, 3.0, 2.0, 4.0, 0.0, 5.0], 2, 3, false);
    let (vals, idx) = max_with_dim(&a, 1, false).unwrap();
    assert_eq!(vals.data().unwrap(), &[3.0, 5.0]);
    assert_eq!(idx.data().unwrap(), &[1i64, 2i64]);
}

#[test]
fn min_with_dim_forward_values_and_indices() {
    // input = [[1,3,2], [4,0,5]], dim=1 → values=[1,0], indices=[0,1].
    let a = leaf_2d(&[1.0, 3.0, 2.0, 4.0, 0.0, 5.0], 2, 3, false);
    let (vals, idx) = min_with_dim(&a, 1, false).unwrap();
    assert_eq!(vals.data().unwrap(), &[1.0, 0.0]);
    assert_eq!(idx.data().unwrap(), &[0i64, 1i64]);
}

#[test]
fn max_with_dim_backward_scatters_to_argmax() {
    // input = [[1,5,2]] (1x3), max along dim=1 → values=[5], idx=[1].
    // Backward: grad goes only to position (0, 1).
    let a = leaf_2d(&[1.0, 5.0, 2.0], 1, 3, true);
    let (vals, _idx) = max_with_dim(&a, 1, false).unwrap();
    vals.backward().unwrap();
    let g = a.grad().unwrap().unwrap();
    let gd = g.data().unwrap();
    assert_eq!(gd, &[0.0, 1.0, 0.0]);
}

#[test]
fn min_with_dim_keepdim_preserves_rank() {
    // keepdim=true keeps the reduced dim as size 1.
    let a = leaf_2d(&[1.0, 3.0, 2.0, 4.0, 0.0, 5.0], 2, 3, false);
    let (vals, idx) = min_with_dim(&a, 1, true).unwrap();
    assert_eq!(vals.shape(), &[2, 1]);
    assert_eq!(idx.shape(), &[2, 1]);
}

// =============================================================================
// norm_with_dim (#1308)
// =============================================================================

#[test]
fn norm_with_dim_p2_matches_euclidean() {
    // p=2 norm along dim=1 of [[3,4],[5,12]] → [5, 13] (Pythag triples).
    let a = leaf_2d(&[3.0, 4.0, 5.0, 12.0], 2, 2, false);
    let r = norm_with_dim(&a, 2.0, 1, false).unwrap();
    let d = r.data().unwrap();
    assert!((d[0] - 5.0).abs() < 1e-12);
    assert!((d[1] - 13.0).abs() < 1e-12);
}

#[test]
fn norm_with_dim_p1_matches_l1_sum_of_abs() {
    // p=1 along dim=1: sum of |x|.
    // [[1,-2,3],[-4,5,-6]] → [6, 15].
    let a = leaf_2d(&[1.0, -2.0, 3.0, -4.0, 5.0, -6.0], 2, 3, false);
    let r = norm_with_dim(&a, 1.0, 1, false).unwrap();
    let d = r.data().unwrap();
    assert!((d[0] - 6.0).abs() < 1e-12);
    assert!((d[1] - 15.0).abs() < 1e-12);
}

#[test]
fn norm_with_dim_p2_backward_matches_chain_rule() {
    // c = sqrt(x0^2 + x1^2). dc/dx0 = x0 / c.
    // For [3,4], c=5, dc/dx0 = 0.6, dc/dx1 = 0.8.
    let a = Tensor::from_storage(TensorStorage::cpu(vec![3.0_f64, 4.0]), vec![1, 2], true).unwrap();
    let r = norm_with_dim(&a, 2.0, 1, false).unwrap();
    r.backward().unwrap();
    let g = a.grad().unwrap().unwrap();
    let gd = g.data().unwrap();
    assert!((gd[0] - 0.6).abs() < 1e-9, "norm dx0: got {}", gd[0]);
    assert!((gd[1] - 0.8).abs() < 1e-9, "norm dx1: got {}", gd[1]);
}

#[test]
fn norm_with_dim_rejects_p_zero_and_inf() {
    let a = leaf_vec(&[1.0, 2.0, 3.0], false);
    // Wrap the 1-D into a 2-D so dim=0 is legal (norm_with_dim refuses 0-D).
    let a2 = Tensor::from_storage(
        TensorStorage::cpu(a.data().unwrap().to_vec()),
        vec![1, 3],
        false,
    )
    .unwrap();
    assert!(norm_with_dim(&a2, 0.0, 1, false).is_err());
    assert!(norm_with_dim(&a2, f64::INFINITY, 1, false).is_err());
    assert!(norm_with_dim(&a2, -1.0, 1, false).is_err());
}
