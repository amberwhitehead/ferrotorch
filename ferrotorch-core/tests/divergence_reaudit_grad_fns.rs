//! Adversarial re-audit of commit 2d27cccd0 (wave-H grad_fns: atan2 / signbit
//! / copysign / hypot / max_with_dim / min_with_dim / norm_with_dim) and
//! commit c013b5432 (scatter_value / where_ / where_bt).
//!
//! Independent critic pass requested under #1542. Expected values are derived
//! from math identities or the same IEEE-754 primitives upstream
//! `aten/src/ATen/native/{Binary,Unary}Ops.cpp` dispatch to on CPU
//! (std::atan2 / std::copysign / std::hypot). NEVER copied from ferrotorch
//! output (R-CHAR-3).

use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::{Tensor, atan2, copysign, hypot, max_with_dim, min_with_dim, norm_with_dim};

fn vec_t(d: &[f64]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(d.to_vec()), vec![d.len()], false).unwrap()
}
fn vec_g(d: &[f64]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(d.to_vec()), vec![d.len()], true).unwrap()
}
fn t2d(d: &[f64], r: usize, c: usize) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(d.to_vec()), vec![r, c], false).unwrap()
}

// ===========================================================================
// #1318 atan2(y, x)
// ===========================================================================

/// atan2(1,1) = pi/4 (math identity). Also verify argument order: torch
/// atan2(input=y, other=x) computes atan2(y, x), so atan2(1, 0) = pi/2 and
/// atan2(0, 1) = 0. A swapped (x,y) impl would give the opposite.
#[test]
fn atan2_value_and_arg_order() {
    let y = vec_t(&[1.0, 1.0, 0.0]);
    let x = vec_t(&[1.0, 0.0, 1.0]);
    let r = atan2(&y, &x).unwrap();
    let d = r.data().unwrap();
    let pi = std::f64::consts::PI;
    assert!((d[0] - pi / 4.0).abs() < 1e-12, "atan2(1,1) got {}", d[0]);
    assert!(
        (d[1] - pi / 2.0).abs() < 1e-12,
        "atan2(1,0) got {} want pi/2 — arg order swapped?",
        d[1]
    );
    assert!((d[2] - 0.0).abs() < 1e-12, "atan2(0,1) got {} want 0", d[2]);
}

/// atan2 backward: d/dy = x/(x^2+y^2), d/dx = -y/(x^2+y^2).
/// At (y=1, x=2): denom = 5, dy = 2/5 = 0.4, dx = -1/5 = -0.2.
#[test]
fn atan2_backward_gradients() {
    let y = vec_g(&[1.0]);
    let x = vec_g(&[2.0]);
    let r = atan2(&y, &x).unwrap();
    r.backward().unwrap();
    let gy = y.grad().unwrap().unwrap().data().unwrap()[0];
    let gx = x.grad().unwrap().unwrap().data().unwrap()[0];
    assert!((gy - 0.4).abs() < 1e-6, "atan2 d/dy got {gy} want 0.4");
    assert!((gx - (-0.2)).abs() < 1e-6, "atan2 d/dx got {gx} want -0.2");
}

// ===========================================================================
// #1334 copysign(magnitude, sign)
// ===========================================================================

/// copysign(3, -1) = -3; copysign(-3, 1) = 3; copysign(3, +0.0) = 3;
/// copysign(3, -0.0) = -3 (sign bit of -0.0). Matches std::copysign.
#[test]
fn copysign_values_including_signed_zero() {
    let mag = vec_t(&[3.0, -3.0, 3.0, 3.0]);
    let sgn = vec_t(&[-1.0, 1.0, 0.0, -0.0]);
    let r = copysign(&mag, &sgn).unwrap();
    let d = r.data().unwrap();
    assert_eq!(d[0], -3.0, "copysign(3,-1)");
    assert_eq!(d[1], 3.0, "copysign(-3,1)");
    assert_eq!(d[2], 3.0, "copysign(3,+0)");
    assert_eq!(d[3], -3.0, "copysign(3,-0) must honour -0.0 sign bit");
}

/// copysign backward: grad to magnitude is grad * sign(result)*sign(self).
/// copysign(3, -1) = -3, so d(out)/d(mag) = -1; grad to sign is 0.
#[test]
fn copysign_backward_magnitude_only() {
    let mag = vec_g(&[3.0]);
    let sgn = vec_g(&[-1.0]);
    let r = copysign(&mag, &sgn).unwrap();
    r.backward().unwrap();
    let gm = mag.grad().unwrap().unwrap().data().unwrap()[0];
    assert!(
        (gm - (-1.0)).abs() < 1e-9,
        "copysign d/dmag got {gm} want -1"
    );
    // sign input gradient must be zero (sign is non-diff in derivatives.yaml).
    if let Some(g) = sgn.grad().unwrap() {
        let v = g.data().unwrap()[0];
        assert!(
            v.abs() < 1e-12,
            "copysign grad to sign should be 0, got {v}"
        );
    }
}

// ===========================================================================
// #1336 hypot(x, y)
// ===========================================================================

/// hypot(3,4) = 5 (math identity). Backward: d/dx = x/hypot, d/dy = y/hypot.
/// At (3,4): d/dx = 3/5 = 0.6, d/dy = 4/5 = 0.8.
#[test]
fn hypot_value_and_backward() {
    let a = vec_g(&[3.0]);
    let b = vec_g(&[4.0]);
    let r = hypot(&a, &b).unwrap();
    assert!((r.data().unwrap()[0] - 5.0).abs() < 1e-12, "hypot(3,4)");
    r.backward().unwrap();
    let ga = a.grad().unwrap().unwrap().data().unwrap()[0];
    let gb = b.grad().unwrap().unwrap().data().unwrap()[0];
    assert!((ga - 0.6).abs() < 1e-6, "hypot d/dx got {ga} want 0.6");
    assert!((gb - 0.8).abs() < 1e-6, "hypot d/dy got {gb} want 0.8");
}

// ===========================================================================
// #1302 max_with_dim / min_with_dim
// ===========================================================================

/// max over dim=1 of [[1,5,3],[4,2,6]] -> values [5,6], indices [1,2].
/// min -> values [1,2], indices [0,1]. Backward scatters grad at argmax.
#[test]
fn max_with_dim_values_indices_and_backward() {
    let x = t2d(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], 2, 3);
    let (vals, idx) = max_with_dim(&x, 1, false).unwrap();
    let vd = vals.data().unwrap();
    assert!(
        (vd[0] - 5.0).abs() < 1e-12 && (vd[1] - 6.0).abs() < 1e-12,
        "max values {vd:?}"
    );
    let id = idx.data().unwrap();
    assert_eq!(id[0], 1, "argmax row0 should be col 1");
    assert_eq!(id[1], 2, "argmax row1 should be col 2");

    // min
    let (mvals, midx) = min_with_dim(&x, 1, false).unwrap();
    let mvd = mvals.data().unwrap();
    assert!(
        (mvd[0] - 1.0).abs() < 1e-12 && (mvd[1] - 2.0).abs() < 1e-12,
        "min values {mvd:?}"
    );
    let mid = midx.data().unwrap();
    assert_eq!(mid[0], 0, "argmin row0 should be col 0");
    assert_eq!(mid[1], 1, "argmin row1 should be col 1");

    // backward: grad of sum(max) w.r.t x is 1 at each argmax, 0 elsewhere.
    let xg = t2d(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], 2, 3);
    let xg = Tensor::from_storage(
        TensorStorage::cpu(xg.data().unwrap().to_vec()),
        vec![2, 3],
        true,
    )
    .unwrap();
    let (v, _) = max_with_dim(&xg, 1, false).unwrap();
    v.sum_all().unwrap().backward().unwrap();
    let g = xg.grad().unwrap().unwrap().data().unwrap().to_vec();
    // expected grad: [[0,1,0],[0,0,1]]
    let expected = [0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    for (i, (&got, &want)) in g.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-9,
            "max backward grad[{i}] got {got} want {want}"
        );
    }
}

// ===========================================================================
// #1308 norm_with_dim
// ===========================================================================

/// p=2 norm over dim=0 of [3,4] = 5 (math identity). Backward: d|x|_2/dx_i =
/// x_i/|x|. At [3,4]: [0.6, 0.8].
#[test]
fn norm_with_dim_value_and_backward() {
    let x = vec_g(&[3.0, 4.0]);
    let n = norm_with_dim(&x, 2.0, 0, false).unwrap();
    assert!(
        (n.data().unwrap()[0] - 5.0).abs() < 1e-12,
        "L2 norm of [3,4]"
    );
    n.backward().unwrap();
    let g = x.grad().unwrap().unwrap().data().unwrap().to_vec();
    assert!((g[0] - 0.6).abs() < 1e-6, "d|x|/dx0 got {} want 0.6", g[0]);
    assert!((g[1] - 0.8).abs() < 1e-6, "d|x|/dx1 got {} want 0.8", g[1]);
}

// ===========================================================================
// c013b5432 — scatter_value / where_ / where_bt
// ===========================================================================

/// scatter_value_t(dim=1, index, value=5.0) writes 5.0 at the indexed cols.
/// Base 2x3 zeros; index [[0],[2]] (shape [2,1]) -> out[0][0]=5, out[1][2]=5.
#[test]
fn scatter_value_writes_scalar() {
    let base = t2d(&[0.0; 6], 2, 3);
    // index tensor of shape [2,1]: row0 -> col0, row1 -> col2.
    let out = base.scatter_value_t(1, &[0, 2], &[2, 1], 5.0_f64).unwrap();
    let d = out.data().unwrap();
    // out laid out row-major [2,3]: positions (0,0)=idx0 and (1,2)=idx5.
    assert_eq!(d[0], 5.0, "scatter_value did not write 5.0 at (0,0)");
    assert_eq!(d[5], 5.0, "scatter_value did not write 5.0 at (1,2)");
    // everything else stays 0.
    for (i, &v) in d.iter().enumerate() {
        if i != 0 && i != 5 {
            assert_eq!(v, 0.0, "scatter_value clobbered position {i}");
        }
    }
}

/// where_t selects self where cond true, other where false.
/// cond=[true,false,true], self=[1,2,3], other=[10,20,30] -> [1,20,3].
#[test]
fn where_selects_by_condition() {
    let a = vec_t(&[1.0, 2.0, 3.0]);
    let b = vec_t(&[10.0, 20.0, 30.0]);
    let out = a.where_t(&[true, false, true], &b).unwrap();
    let d = out.data().unwrap();
    assert_eq!(d, &[1.0, 20.0, 3.0], "where_t selection wrong");
}
