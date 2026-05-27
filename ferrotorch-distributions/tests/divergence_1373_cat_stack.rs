//! Oracle parity for CatTransform / StackTransform (commit `a2ab04347`, #1373).
//! Expected values OracleDerived from live torch 2.11.0 (R-CHAR-3).
//!
//! CatTransform([Exp, Affine(1,2)], dim=0, lengths=[2,1]) on x=[0.1,0.2,0.3]:
//!   forward = [1.10517097, 1.22140276, 1.6]
//!   ldj     = [0.1, 0.2, 0.69314718]
//! StackTransform([Exp, Affine(0,3)], dim=0) on [[0.1,0.2],[0.3,0.4]]:
//!   forward = [[1.10517097,1.22140276],[0.9,1.2]]
//!   ldj     = [[0.1,0.2],[1.09861231,1.09861231]]

use ferrotorch_core::creation::from_slice;
use ferrotorch_distributions::{
    AffineTransform, CatTransform, ExpTransform, StackTransform, Transform,
};

fn approx(a: &[f32], b: &[f32], tol: f32, ctx: &str) {
    assert_eq!(a.len(), b.len(), "{ctx}: length {a:?} vs {b:?}");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!((x - y).abs() < tol, "{ctx}: idx {i} got {x} expected {y}");
    }
}

#[test]
#[allow(
    clippy::approx_constant,
    reason = "0.693_147_2 is the f32 PyTorch oracle ldj output ln(2) from the exp leg of the CatTransform, not std::f32::consts::LN_2; keep the printed torch value"
)]
fn divergence_cat_transform_forward_ldj() {
    let transforms: Vec<Box<dyn Transform<f32>>> = vec![
        Box::new(ExpTransform),
        Box::new(AffineTransform::new(1.0f32, 2.0f32)),
    ];
    let ct = CatTransform::new(transforms, 0, vec![2, 1]).unwrap();
    let x = from_slice(&[0.1f32, 0.2, 0.3], &[3]).unwrap();
    let y = ct.forward(&x).unwrap();
    approx(
        y.data().unwrap(),
        &[1.105_171_f32, 1.2214028, 1.6],
        1e-5,
        "Cat forward",
    );
    let ldj = ct.log_abs_det_jacobian(&x, &y).unwrap();
    approx(
        ldj.data().unwrap(),
        &[0.1f32, 0.2, 0.693_147_2],
        1e-5,
        "Cat ldj",
    );
}

#[test]
fn divergence_stack_transform_forward_ldj() {
    let transforms: Vec<Box<dyn Transform<f32>>> = vec![
        Box::new(ExpTransform),
        Box::new(AffineTransform::new(0.0f32, 3.0f32)),
    ];
    let st = StackTransform::new(transforms, 0);
    let x = from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[2, 2]).unwrap();
    let y = st.forward(&x).unwrap();
    approx(
        y.data().unwrap(),
        &[1.105_171_f32, 1.2214028, 0.9, 1.2],
        1e-5,
        "Stack forward",
    );
    let ldj = st.log_abs_det_jacobian(&x, &y).unwrap();
    approx(
        ldj.data().unwrap(),
        &[0.1f32, 0.2, 1.0986123, 1.0986123],
        1e-5,
        "Stack ldj",
    );
}
