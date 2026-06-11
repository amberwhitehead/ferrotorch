//! Red-then-green regression tests for audit finding CORE-175 (crosslink
//! #1869): `xlogy(0, NaN)` returns 0 — the NaN-first rule is missing. The
//! upstream kernel (`pytorch/aten/src/ATen/native/cpu/BinaryOpsKernel.cpp:
//! 1288-1300 xlogy_kernel`) checks `isnan(y)` BEFORE the `x == 0` shortcut.
//!
//! Oracle (R-ORACLE-1 path (b)) — live torch 2.11.0+cu130, 2026-06-11,
//! this machine:
//!
//! ```python
//! >>> t = lambda v: torch.tensor(v, dtype=torch.float64)
//! >>> torch.special.xlogy(t(0.0), t(float('nan'))).item()  # nan
//! >>> torch.special.xlogy(t(float('nan')), t(0.0)).item()  # nan
//! >>> torch.special.xlogy(t(2.0), t(float('nan'))).item()  # nan
//! >>> torch.special.xlogy(t(0.0), t(0.0)).item()           # 0.0
//! >>> torch.special.xlogy(t(0.0), t(float('inf'))).item()  # 0.0
//! ```
//!
//! Tolerance justification (R-ORACLE-5): all pins are exact (NaN-ness /
//! exact zero); the interior pin `xlogy(2, e) = 2` is exact in f64 because
//! `ln(E)` evaluates to exactly 1.0 for the f64 constant E (pre-existing
//! in-module test uses the same identity at 1e-14).

use ferrotorch_core::special::xlogy;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

fn t64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}
fn t32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

#[test]
fn core175_xlogy_nan_y_wins_over_zero_x_f64() {
    let x = t64(&[0.0, f64::NAN, 2.0], &[3]);
    let y = t64(&[f64::NAN, 0.0, f64::NAN], &[3]);
    let r = xlogy(&x, &y).unwrap();
    let d = r.data().unwrap();
    assert!(
        d[0].is_nan(),
        "xlogy(0, NaN) must be NaN per BinaryOpsKernel.cpp:1292 (isnan(y) first), got {}",
        d[0]
    );
    assert!(d[1].is_nan(), "xlogy(NaN, 0) must be NaN, got {}", d[1]);
    assert!(d[2].is_nan(), "xlogy(2, NaN) must be NaN, got {}", d[2]);
}

#[test]
fn core175_xlogy_nan_y_wins_over_zero_x_f32() {
    let x = t32(&[0.0], &[1]);
    let y = t32(&[f32::NAN], &[1]);
    let r = xlogy(&x, &y).unwrap();
    assert!(
        r.data().unwrap()[0].is_nan(),
        "xlogy f32 (0, NaN) must be NaN, got {}",
        r.data().unwrap()[0]
    );
}

#[test]
fn core175_xlogy_zero_x_non_nan_y_unchanged() {
    // The pre-existing contract rows: xlogy(0, y) = 0 for non-NaN y,
    // including y = 0 and y = inf (where x*ln(y) alone would be NaN).
    let x = t64(&[0.0, 0.0, 0.0], &[3]);
    let y = t64(&[1.0, 0.0, f64::INFINITY], &[3]);
    let r = xlogy(&x, &y).unwrap();
    let d = r.data().unwrap();
    for (i, v) in d.iter().enumerate() {
        assert!(*v == 0.0, "xlogy(0, non-NaN) lane {i} must stay 0, got {v}");
    }
    // Interior row untouched.
    let r = xlogy(&t64(&[2.0], &[1]), &t64(&[std::f64::consts::E], &[1])).unwrap();
    assert!(
        (r.data().unwrap()[0] - 2.0).abs() < 1e-14,
        "xlogy(2, e) moved: got {}",
        r.data().unwrap()[0]
    );
}
