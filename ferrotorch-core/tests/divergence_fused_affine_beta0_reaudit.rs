//! Re-audit of the #1598 fix (commit 1a43ddeee) — the fused-affine family
//! (`addmm`, `addmv`, `addr`, `addbmm`, `baddbmm`) `beta == 0`
//! NaN/Inf-suppression contract, probed adversarially BEYOND the original
//! 5 forward-NaN tests in `divergence_fused_affine_beta0_nan.rs`.
//!
//! The fixer branches on `beta == 0` in each forward and computes
//! `alpha * product` only (the `self`/bias buffer is never read). This file
//! pins the corners that branch could still get wrong:
//!
//!   1. beta == 0 AND alpha == 0, self = NaN, finite mats  -> all-zeros
//!      (torch drops self AND the product is multiplied by 0; result is +0,
//!      NOT NaN). `aten/src/ATen/native/cpu/LinearAlgebraKernel.cpp:60`
//!      (`return alpha_val * vec1_val * vec2_val;` with alpha_val == 0).
//!   2. beta == 0, self = +Inf / -Inf / mixed Inf/NaN       -> alpha*product
//!      (self FULLY ignored, no Inf/NaN leak).
//!   3. beta != 0 REGRESSION (beta=0.5, 1.0, 2.0, finite self) -> the normal
//!      `beta*self + alpha*product` path must be UNCHANGED by the fix.
//!   4. EXACT-zero detection: beta = 1e-30 (a TINY but NON-zero f32) with
//!      self = +Inf must take the NORMAL path => Inf in the output, NOT
//!      dropped. torch's `if (beta_val == zero_val)` is an exact `== 0`
//!      compare (`LinearAlgebraKernel.cpp:55`, `BlasKernel.cpp:161`,
//!      `LinearAlgebra.cpp:1683`), so 1e-30 is NOT the drop path.
//!   5. backward at beta == 0 with self = NaN requiring grad: self.grad must
//!      be all-zeros (NOT NaN). torch: `d_self = sum_to(beta*grad)`; at
//!      beta==0 that is exactly 0 (`tools/autograd/derivatives.yaml` addmm
//!      :256 / addmv :267 / addr :273). A NaN self that leaks into self.grad
//!      would be a divergence.
//!
//! ALL expected values below were produced by the LIVE torch 2.11.0+cu130
//! oracle (R-CHAR-3 — never copied from the ferrotorch side):
//!
//!   addmm beta0 alpha0 nan-self          = [0, 0, 0, 0]
//!   addr  beta0 alpha0 nan-self          = [0, 0, 0, 0, 0, 0]
//!   addmm beta0 {+Inf,-Inf,mixed}-self a1= [4, 5, 10, 11]
//!   addmm beta0.5 a2 finite              = [13, 20, 35, 42]
//!   addmm beta1 a1 finite                = [14, 25, 40, 51]
//!   addmm beta1e-30 inf-self a1          = [inf, inf, inf, inf]   (NORMAL path)
//!   addmv beta0 {nan,inf}-self a1        = [6, 15]
//!   addmv beta0 alpha0 nan-self          = [0, 0]
//!   addmv beta0.5 a2 finite              = [17, 40]
//!   addmv beta1e-30 inf-self             = [inf, inf]
//!   addr  beta0 {inf,mixed}-self a1      = [3, 4, 5, 6, 8, 10]
//!   addr  beta0.5 a2 finite              = [11, 13, 15, 17, 21, 25]
//!   addr  beta1e-30 inf-self             = [inf, inf, inf, inf, inf, inf]
//!   addbmm  beta0 {nan,inf}-self a1      = [6, 5, 10, 13]
//!   addbmm  beta0 alpha0 nan-self        = [0, 0, 0, 0]
//!   addbmm  beta0.5 a2 finite            = [17, 15, 25, 31]
//!   addbmm  beta1e-30 inf-self           = [inf, inf, inf, inf]
//!   baddbmm beta0 {nan,inf}-self a1      = [4,5,10,11, 2,0,0,2]
//!   baddbmm beta0 alpha0 nan-self        = [0,0,0,0, 0,0,0,0]
//!   baddbmm beta0.5 a2 finite            = [13,15,25,27, 9,5,5,9]
//!   baddbmm beta1e-30 inf-self           = [inf x8]
//!   addmm beta0 a2 self.grad (NaN self)  = [0, 0, 0, 0]   (m1.grad=[2,2,4,2,2,4], m2.grad=[10,10,14,14,18,18])
//!   addmv beta0 a3 self.grad (NaN self)  = [0, 0]         (mat.grad=[3x6], vec.grad=[15,21,27])
//!   addr  beta0 a1 self.grad (NaN self)  = [0, 0, 0, 0, 0, 0]  (v1.grad=[12,12], v2.grad=[3,3,3])
//!   addmm beta2 a1 self.grad (NaN self)  = [2, 2, 2, 2]   (beta!=0 path; finite)
//!
//! Tracking: #1598.

use ferrotorch_core::Tensor;
use ferrotorch_core::grad_fns::reduction::sum;
use ferrotorch_core::linalg;
use ferrotorch_core::storage::TensorStorage;

fn t(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}
fn tg(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}
fn grad_of(x: &Tensor<f32>) -> Vec<f32> {
    x.grad().unwrap().unwrap().data().unwrap().to_vec()
}

#[track_caller]
fn assert_exact(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        if e.is_infinite() {
            assert!(
                a.is_infinite() && a.signum() == e.signum(),
                "{label}: output[{i}] = {a}, torch = {e} (expected matching Inf)"
            );
        } else {
            assert!(
                a.is_finite(),
                "{label}: DIVERGENCE — output[{i}] = {a} is non-finite; torch = {e}"
            );
            let diff = (a - e).abs();
            assert!(
                diff <= 1e-4 + 1e-4 * e.abs(),
                "{label}: output[{i}] = {a} != torch {e} (diff={diff})"
            );
        }
    }
}

// ---------- shared finite operands ----------
fn mm_operands() -> (Tensor<f32>, Tensor<f32>) {
    (
        t(&[1., 2., 3., 4., 5., 6.], &[2, 3]),
        t(&[1., 0., 0., 1., 1., 1.], &[3, 2]),
    )
}
fn bmm_operands() -> (Tensor<f32>, Tensor<f32>) {
    let b1 = t(
        &[1., 2., 3., 4., 5., 6., 1., 0., 0., 0., 1., 0.],
        &[2, 2, 3],
    );
    let b2 = t(
        &[1., 0., 0., 1., 1., 1., 2., 0., 0., 2., 0., 0.],
        &[2, 3, 2],
    );
    (b1, b2)
}

// ===========================================================================
// 1. beta == 0 AND alpha == 0, self = NaN  -> all-zeros (NOT NaN)
// ===========================================================================
#[test]
fn addmm_beta0_alpha0_nan_self_all_zero() {
    let nan = f32::NAN;
    let (m1, m2) = mm_operands();
    let out = linalg::addmm(&t(&[nan; 4], &[2, 2]), &m1, &m2, 0.0, 0.0).unwrap();
    assert_exact(
        out.data().unwrap(),
        &[0., 0., 0., 0.],
        "addmm beta0 alpha0 nan-self",
    );
}

#[test]
fn addr_beta0_alpha0_nan_self_all_zero() {
    let nan = f32::NAN;
    let out = linalg::addr(
        &t(&[nan; 6], &[2, 3]),
        &t(&[1., 2.], &[2]),
        &t(&[3., 4., 5.], &[3]),
        0.0,
        0.0,
    )
    .unwrap();
    assert_exact(
        out.data().unwrap(),
        &[0., 0., 0., 0., 0., 0.],
        "addr beta0 alpha0 nan-self",
    );
}

#[test]
fn addmv_beta0_alpha0_nan_self_all_zero() {
    let nan = f32::NAN;
    let out = linalg::addmv(
        &t(&[nan; 2], &[2]),
        &t(&[1., 2., 3., 4., 5., 6.], &[2, 3]),
        &t(&[1., 1., 1.], &[3]),
        0.0,
        0.0,
    )
    .unwrap();
    assert_exact(
        out.data().unwrap(),
        &[0., 0.],
        "addmv beta0 alpha0 nan-self",
    );
}

#[test]
fn addbmm_beta0_alpha0_nan_self_all_zero() {
    let nan = f32::NAN;
    let (b1, b2) = bmm_operands();
    let out = linalg::addbmm(&t(&[nan; 4], &[2, 2]), &b1, &b2, 0.0, 0.0).unwrap();
    assert_exact(
        out.data().unwrap(),
        &[0., 0., 0., 0.],
        "addbmm beta0 alpha0 nan-self",
    );
}

#[test]
fn baddbmm_beta0_alpha0_nan_self_all_zero() {
    let nan = f32::NAN;
    let (b1, b2) = bmm_operands();
    let out = linalg::baddbmm(&t(&[nan; 8], &[2, 2, 2]), &b1, &b2, 0.0, 0.0).unwrap();
    assert_exact(
        out.data().unwrap(),
        &[0., 0., 0., 0., 0., 0., 0., 0.],
        "baddbmm beta0 alpha0 nan-self",
    );
}

// ===========================================================================
// 2. beta == 0, self = +Inf / -Inf / mixed  -> alpha*product (self ignored)
// ===========================================================================
#[test]
fn addmm_beta0_inf_self_dropped() {
    let inf = f32::INFINITY;
    let nan = f32::NAN;
    let (m1, m2) = mm_operands();
    let exp = [4., 5., 10., 11.];
    assert_exact(
        linalg::addmm(&t(&[inf; 4], &[2, 2]), &m1, &m2, 0.0, 1.0)
            .unwrap()
            .data()
            .unwrap(),
        &exp,
        "addmm beta0 +inf-self",
    );
    assert_exact(
        linalg::addmm(&t(&[-inf; 4], &[2, 2]), &m1, &m2, 0.0, 1.0)
            .unwrap()
            .data()
            .unwrap(),
        &exp,
        "addmm beta0 -inf-self",
    );
    assert_exact(
        linalg::addmm(&t(&[inf, -inf, nan, 0.0], &[2, 2]), &m1, &m2, 0.0, 1.0)
            .unwrap()
            .data()
            .unwrap(),
        &exp,
        "addmm beta0 mixed-self",
    );
}

#[test]
fn addmv_beta0_inf_self_dropped() {
    let inf = f32::INFINITY;
    let out = linalg::addmv(
        &t(&[inf, -inf], &[2]),
        &t(&[1., 2., 3., 4., 5., 6.], &[2, 3]),
        &t(&[1., 1., 1.], &[3]),
        0.0,
        1.0,
    )
    .unwrap();
    assert_exact(out.data().unwrap(), &[6., 15.], "addmv beta0 inf-self");
}

#[test]
fn addr_beta0_inf_self_dropped() {
    let inf = f32::INFINITY;
    let nan = f32::NAN;
    let exp = [3., 4., 5., 6., 8., 10.];
    let v1 = t(&[1., 2.], &[2]);
    let v2 = t(&[3., 4., 5.], &[3]);
    assert_exact(
        linalg::addr(&t(&[inf; 6], &[2, 3]), &v1, &v2, 0.0, 1.0)
            .unwrap()
            .data()
            .unwrap(),
        &exp,
        "addr beta0 inf-self",
    );
    assert_exact(
        linalg::addr(
            &t(&[inf, -inf, nan, 0.0, nan, inf], &[2, 3]),
            &v1,
            &v2,
            0.0,
            1.0,
        )
        .unwrap()
        .data()
        .unwrap(),
        &exp,
        "addr beta0 mixed-self",
    );
}

#[test]
fn addbmm_beta0_inf_self_dropped() {
    let inf = f32::INFINITY;
    let (b1, b2) = bmm_operands();
    let out = linalg::addbmm(&t(&[inf; 4], &[2, 2]), &b1, &b2, 0.0, 1.0).unwrap();
    assert_exact(
        out.data().unwrap(),
        &[6., 5., 10., 13.],
        "addbmm beta0 inf-self",
    );
}

#[test]
fn baddbmm_beta0_inf_self_dropped() {
    let inf = f32::INFINITY;
    let (b1, b2) = bmm_operands();
    let out = linalg::baddbmm(&t(&[inf; 8], &[2, 2, 2]), &b1, &b2, 0.0, 1.0).unwrap();
    assert_exact(
        out.data().unwrap(),
        &[4., 5., 10., 11., 2., 0., 0., 2.],
        "baddbmm beta0 inf-self",
    );
}

// ===========================================================================
// 3. beta != 0 REGRESSION — normal path must be UNCHANGED by the fix
// ===========================================================================
#[test]
fn addmm_beta_nonzero_regression() {
    let (m1, m2) = mm_operands();
    let s = t(&[10., 20., 30., 40.], &[2, 2]);
    assert_exact(
        linalg::addmm(&s, &m1, &m2, 0.5, 2.0)
            .unwrap()
            .data()
            .unwrap(),
        &[13., 20., 35., 42.],
        "addmm beta0.5 a2",
    );
    assert_exact(
        linalg::addmm(&s, &m1, &m2, 1.0, 1.0)
            .unwrap()
            .data()
            .unwrap(),
        &[14., 25., 40., 51.],
        "addmm beta1 a1",
    );
}

#[test]
fn addmv_beta_nonzero_regression() {
    let out = linalg::addmv(
        &t(&[10., 20.], &[2]),
        &t(&[1., 2., 3., 4., 5., 6.], &[2, 3]),
        &t(&[1., 1., 1.], &[3]),
        0.5,
        2.0,
    )
    .unwrap();
    assert_exact(out.data().unwrap(), &[17., 40.], "addmv beta0.5 a2");
}

#[test]
fn addr_beta_nonzero_regression() {
    let out = linalg::addr(
        &t(&[10.; 6], &[2, 3]),
        &t(&[1., 2.], &[2]),
        &t(&[3., 4., 5.], &[3]),
        0.5,
        2.0,
    )
    .unwrap();
    assert_exact(
        out.data().unwrap(),
        &[11., 13., 15., 17., 21., 25.],
        "addr beta0.5 a2",
    );
}

#[test]
fn addbmm_beta_nonzero_regression() {
    let (b1, b2) = bmm_operands();
    let out = linalg::addbmm(&t(&[10.; 4], &[2, 2]), &b1, &b2, 0.5, 2.0).unwrap();
    assert_exact(
        out.data().unwrap(),
        &[17., 15., 25., 31.],
        "addbmm beta0.5 a2",
    );
}

#[test]
fn baddbmm_beta_nonzero_regression() {
    let (b1, b2) = bmm_operands();
    let out = linalg::baddbmm(&t(&[10.; 8], &[2, 2, 2]), &b1, &b2, 0.5, 2.0).unwrap();
    assert_exact(
        out.data().unwrap(),
        &[13., 15., 25., 27., 9., 5., 5., 9.],
        "baddbmm beta0.5 a2",
    );
}

// ===========================================================================
// 4. EXACT-zero detection — beta = 1e-30 (non-zero) takes the NORMAL path:
//    self = +Inf must PROPAGATE (1e-30 * Inf = Inf), NOT be dropped.
// ===========================================================================
#[test]
fn addmm_beta_tiny_nonzero_takes_normal_path_inf() {
    let inf = f32::INFINITY;
    let (m1, m2) = mm_operands();
    let out = linalg::addmm(&t(&[inf; 4], &[2, 2]), &m1, &m2, 1e-30, 1.0).unwrap();
    assert_exact(
        out.data().unwrap(),
        &[inf, inf, inf, inf],
        "addmm beta1e-30 inf-self NORMAL path",
    );
}

#[test]
fn addmv_beta_tiny_nonzero_takes_normal_path_inf() {
    let inf = f32::INFINITY;
    let out = linalg::addmv(
        &t(&[inf; 2], &[2]),
        &t(&[1., 2., 3., 4., 5., 6.], &[2, 3]),
        &t(&[1., 1., 1.], &[3]),
        1e-30,
        1.0,
    )
    .unwrap();
    assert_exact(
        out.data().unwrap(),
        &[inf, inf],
        "addmv beta1e-30 inf-self NORMAL path",
    );
}

#[test]
fn addr_beta_tiny_nonzero_takes_normal_path_inf() {
    let inf = f32::INFINITY;
    let out = linalg::addr(
        &t(&[inf; 6], &[2, 3]),
        &t(&[1., 2.], &[2]),
        &t(&[3., 4., 5.], &[3]),
        1e-30,
        1.0,
    )
    .unwrap();
    assert_exact(
        out.data().unwrap(),
        &[inf, inf, inf, inf, inf, inf],
        "addr beta1e-30 inf-self NORMAL path",
    );
}

#[test]
fn addbmm_beta_tiny_nonzero_takes_normal_path_inf() {
    let inf = f32::INFINITY;
    let (b1, b2) = bmm_operands();
    let out = linalg::addbmm(&t(&[inf; 4], &[2, 2]), &b1, &b2, 1e-30, 1.0).unwrap();
    assert_exact(
        out.data().unwrap(),
        &[inf, inf, inf, inf],
        "addbmm beta1e-30 inf-self NORMAL path",
    );
}

#[test]
fn baddbmm_beta_tiny_nonzero_takes_normal_path_inf() {
    let inf = f32::INFINITY;
    let (b1, b2) = bmm_operands();
    let out = linalg::baddbmm(&t(&[inf; 8], &[2, 2, 2]), &b1, &b2, 1e-30, 1.0).unwrap();
    assert_exact(
        out.data().unwrap(),
        &[inf, inf, inf, inf, inf, inf, inf, inf],
        "baddbmm beta1e-30 inf-self NORMAL path",
    );
}

// ===========================================================================
// 5. backward at beta == 0 with NaN self requiring grad -> self.grad all-zero
// ===========================================================================
#[test]
fn addmm_beta0_backward_nan_self_grad_zero() {
    let nan = f32::NAN;
    let self_t = tg(&[nan; 4], &[2, 2]);
    let m1 = tg(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
    let m2 = tg(&[1., 0., 0., 1., 1., 1.], &[3, 2]);
    let out = linalg::addmm(&self_t, &m1, &m2, 0.0, 2.0).unwrap();
    let loss = sum(&out).unwrap();
    loss.backward().unwrap();
    assert_exact(
        &grad_of(&self_t),
        &[0., 0., 0., 0.],
        "addmm beta0 self.grad (NaN self)",
    );
    assert_exact(
        &grad_of(&m1),
        &[2., 2., 4., 2., 2., 4.],
        "addmm beta0 m1.grad",
    );
    assert_exact(
        &grad_of(&m2),
        &[10., 10., 14., 14., 18., 18.],
        "addmm beta0 m2.grad",
    );
}

#[test]
fn addmv_beta0_backward_nan_self_grad_zero() {
    let nan = f32::NAN;
    let self_t = tg(&[nan; 2], &[2]);
    let mat = tg(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
    let vec = tg(&[1., 1., 1.], &[3]);
    let out = linalg::addmv(&self_t, &mat, &vec, 0.0, 3.0).unwrap();
    let loss = sum(&out).unwrap();
    loss.backward().unwrap();
    assert_exact(
        &grad_of(&self_t),
        &[0., 0.],
        "addmv beta0 self.grad (NaN self)",
    );
    assert_exact(
        &grad_of(&mat),
        &[3., 3., 3., 3., 3., 3.],
        "addmv beta0 mat.grad",
    );
    assert_exact(&grad_of(&vec), &[15., 21., 27.], "addmv beta0 vec.grad");
}

#[test]
fn addr_beta0_backward_nan_self_grad_zero() {
    let nan = f32::NAN;
    let self_t = tg(&[nan; 6], &[2, 3]);
    let v1 = tg(&[1., 2.], &[2]);
    let v2 = tg(&[3., 4., 5.], &[3]);
    let out = linalg::addr(&self_t, &v1, &v2, 0.0, 1.0).unwrap();
    let loss = sum(&out).unwrap();
    loss.backward().unwrap();
    assert_exact(
        &grad_of(&self_t),
        &[0., 0., 0., 0., 0., 0.],
        "addr beta0 self.grad (NaN self)",
    );
    assert_exact(&grad_of(&v1), &[12., 12.], "addr beta0 v1.grad");
    assert_exact(&grad_of(&v2), &[3., 3., 3.], "addr beta0 v2.grad");
}

#[test]
fn addmm_beta_nonzero_backward_nan_self_grad_finite() {
    // beta != 0 path: self.grad = beta * grad = 2.0 (finite even with NaN self,
    // because grad w.r.t. self does NOT read self's value).
    let nan = f32::NAN;
    let self_t = tg(&[nan; 4], &[2, 2]);
    let m1 = tg(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
    let m2 = tg(&[1., 0., 0., 1., 1., 1.], &[3, 2]);
    let out = linalg::addmm(&self_t, &m1, &m2, 2.0, 1.0).unwrap();
    let loss = sum(&out).unwrap();
    loss.backward().unwrap();
    assert_exact(
        &grad_of(&self_t),
        &[2., 2., 2., 2.],
        "addmm beta2 self.grad (NaN self)",
    );
}
