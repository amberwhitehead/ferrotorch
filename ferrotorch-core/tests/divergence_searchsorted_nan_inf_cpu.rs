//! Re-audit of commit a930863f0 (#1645): searchsorted NaN -> len predicate flip.
//!
//! These tests pin the CPU `partition_point`-based `searchsorted` against the
//! LIVE torch 2.11.0+cu130 oracle for the tricky operands the predicate flip
//! (`<`/`<=` -> `!(>=)`/`!(>)`) could perturb. Every expected value below was
//! produced by live torch (see header of each test), NOT copied from
//! ferrotorch (R-CHAR-3). Upstream binary search:
//!   aten/src/ATen/native/cuda/Bucketization.cu:33 lower_bound `if (!(mid_val >= val))`
//!   aten/src/ATen/native/cuda/Bucketization.cu:51 upper_bound `if (!(mid_val > val))`
//!
//! Tracking: #1645, #1545

use ferrotorch_core::ops::search::searchsorted;
use ferrotorch_core::{Tensor, TensorStorage};

fn cpu_f32(data: &[f32]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false).unwrap()
}
fn cpu_f64(data: &[f64]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false).unwrap()
}

// ---------------------------------------------------------------------------
// NaN VALUE -> len, both sides, f32 + f64
// LIVE torch: searchsorted([1,3,5,7],[NaN,2], right=False/True) -> [4,1].
// ---------------------------------------------------------------------------
#[test]
fn searchsorted_cpu_nan_value_f32_left() {
    let b = cpu_f32(&[1.0, 3.0, 5.0, 7.0]);
    let v = cpu_f32(&[f32::NAN, 2.0]);
    assert_eq!(searchsorted(&b, &v, false).unwrap(), vec![4, 1]);
}
#[test]
fn searchsorted_cpu_nan_value_f32_right() {
    let b = cpu_f32(&[1.0, 3.0, 5.0, 7.0]);
    let v = cpu_f32(&[f32::NAN, 2.0]);
    assert_eq!(searchsorted(&b, &v, true).unwrap(), vec![4, 1]);
}
#[test]
fn searchsorted_cpu_nan_value_f64_both() {
    let b = cpu_f64(&[1.0, 3.0, 5.0, 7.0]);
    let v = cpu_f64(&[f64::NAN]);
    assert_eq!(searchsorted(&b, &v, false).unwrap(), vec![4]);
    assert_eq!(searchsorted(&b, &v, true).unwrap(), vec![4]);
}

// ---------------------------------------------------------------------------
// +/-inf VALUE (ordered, finite-handled): +inf -> len, -inf -> 0, both sides.
// LIVE torch: searchsorted([1,3,5,7],[inf,-inf], right=False/True) -> [4,0].
// ---------------------------------------------------------------------------
#[test]
fn searchsorted_cpu_inf_value_f32_left() {
    let b = cpu_f32(&[1.0, 3.0, 5.0, 7.0]);
    let v = cpu_f32(&[f32::INFINITY, f32::NEG_INFINITY]);
    assert_eq!(searchsorted(&b, &v, false).unwrap(), vec![4, 0]);
}
#[test]
fn searchsorted_cpu_inf_value_f32_right() {
    let b = cpu_f32(&[1.0, 3.0, 5.0, 7.0]);
    let v = cpu_f32(&[f32::INFINITY, f32::NEG_INFINITY]);
    assert_eq!(searchsorted(&b, &v, true).unwrap(), vec![4, 0]);
}
#[test]
fn searchsorted_cpu_inf_value_f64_both() {
    let b = cpu_f64(&[1.0, 3.0, 5.0, 7.0]);
    let v = cpu_f64(&[f64::INFINITY, f64::NEG_INFINITY]);
    assert_eq!(searchsorted(&b, &v, false).unwrap(), vec![4, 0]);
    assert_eq!(searchsorted(&b, &v, true).unwrap(), vec![4, 0]);
}

// ---------------------------------------------------------------------------
// NaN IN THE BOUNDARIES. torch treats boundaries as "sorted" with a trailing
// NaN; the binary search (lower_bound/upper_bound) is NOT a partition predicate
// here, so Rust `partition_point` could diverge from torch's explicit search.
// LIVE torch f32+f64 both sides: b=[1,3,5,7,NaN], v=[4,NaN,8] -> [2,5,5].
// ---------------------------------------------------------------------------
#[test]
fn searchsorted_cpu_nan_in_boundaries_f32_left() {
    let b = cpu_f32(&[1.0, 3.0, 5.0, 7.0, f32::NAN]);
    let v = cpu_f32(&[4.0, f32::NAN, 8.0]);
    assert_eq!(searchsorted(&b, &v, false).unwrap(), vec![2, 5, 5]);
}
#[test]
fn searchsorted_cpu_nan_in_boundaries_f32_right() {
    let b = cpu_f32(&[1.0, 3.0, 5.0, 7.0, f32::NAN]);
    let v = cpu_f32(&[4.0, f32::NAN, 8.0]);
    assert_eq!(searchsorted(&b, &v, true).unwrap(), vec![2, 5, 5]);
}
#[test]
fn searchsorted_cpu_nan_in_boundaries_f64_both() {
    let b = cpu_f64(&[1.0, 3.0, 5.0, 7.0, f64::NAN]);
    let v = cpu_f64(&[4.0, f64::NAN, 8.0]);
    assert_eq!(searchsorted(&b, &v, false).unwrap(), vec![2, 5, 5]);
    assert_eq!(searchsorted(&b, &v, true).unwrap(), vec![2, 5, 5]);
}

// ---------------------------------------------------------------------------
// FINITE REGRESSION (the predicate-flip risk). Must remain byte-exact.
// LIVE torch:
//   exact ties b=[1,3,5,7] v=[1,3,5,7] left -> [0,1,2,3], right -> [1,2,3,4]
//   duplicates b=[1,3,3,3,5] v=[3]      left -> [1],       right -> [4]
//   out-of-range b=[1,3,5,7] v=[0,8]    left/right -> [0,4]
// ---------------------------------------------------------------------------
#[test]
fn searchsorted_cpu_finite_ties_f32() {
    let b = cpu_f32(&[1.0, 3.0, 5.0, 7.0]);
    let v = cpu_f32(&[1.0, 3.0, 5.0, 7.0]);
    assert_eq!(searchsorted(&b, &v, false).unwrap(), vec![0, 1, 2, 3]);
    assert_eq!(searchsorted(&b, &v, true).unwrap(), vec![1, 2, 3, 4]);
}
#[test]
fn searchsorted_cpu_finite_ties_f64() {
    let b = cpu_f64(&[1.0, 3.0, 5.0, 7.0]);
    let v = cpu_f64(&[1.0, 3.0, 5.0, 7.0]);
    assert_eq!(searchsorted(&b, &v, false).unwrap(), vec![0, 1, 2, 3]);
    assert_eq!(searchsorted(&b, &v, true).unwrap(), vec![1, 2, 3, 4]);
}
#[test]
fn searchsorted_cpu_finite_duplicates() {
    let b = cpu_f32(&[1.0, 3.0, 3.0, 3.0, 5.0]);
    let v = cpu_f32(&[3.0]);
    assert_eq!(searchsorted(&b, &v, false).unwrap(), vec![1]);
    assert_eq!(searchsorted(&b, &v, true).unwrap(), vec![4]);
}
#[test]
fn searchsorted_cpu_finite_out_of_range() {
    let b = cpu_f64(&[1.0, 3.0, 5.0, 7.0]);
    let v = cpu_f64(&[0.0, 8.0]);
    assert_eq!(searchsorted(&b, &v, false).unwrap(), vec![0, 4]);
    assert_eq!(searchsorted(&b, &v, true).unwrap(), vec![0, 4]);
}
