//! Re-audit of commit a930863f0 (#1645): GPU searchsorted predicate flip
//! (setp.lt/le -> setp.ge/gt + not.pred). Confirms the new PTX advance logic
//! matches LIVE torch 2.11.0+cu130 not only for NaN values (covered in
//! divergence_searchsorted_nan.rs) but for the operands the flip could perturb:
//! +/-inf values, NaN-in-boundaries, and finite ties/dups/oob (regression).
//!
//! Upstream binary search:
//!   aten/src/ATen/native/cuda/Bucketization.cu:33 lower_bound `if (!(mid_val >= val))`
//!   aten/src/ATen/native/cuda/Bucketization.cu:51 upper_bound `if (!(mid_val > val))`
//!
//! Every expected value is from live torch (header per test), not ferrotorch
//! (R-CHAR-3). Tracking: #1645, #1545.

#![cfg(feature = "cuda")]

use ferrotorch_core::ops::search::searchsorted;
use ferrotorch_core::{Device, Tensor, TensorStorage};
use ferrotorch_gpu::init_cuda_backend;

fn ensure_cuda() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        init_cuda_backend().expect("CUDA backend init");
    });
}
fn gpu_f32(data: &[f32]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false)
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap()
}
fn gpu_f64(data: &[f64]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false)
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap()
}

// +/-inf VALUE. LIVE torch: b=[1,3,5,7], v=[inf,-inf] -> [4,0] both sides, f32+f64.
#[test]
fn gpu_searchsorted_inf_value_f32() {
    ensure_cuda();
    let b = gpu_f32(&[1.0, 3.0, 5.0, 7.0]);
    let v = gpu_f32(&[f32::INFINITY, f32::NEG_INFINITY]);
    assert_eq!(searchsorted(&b, &v, false).unwrap(), vec![4, 0]);
    assert_eq!(searchsorted(&b, &v, true).unwrap(), vec![4, 0]);
}
#[test]
fn gpu_searchsorted_inf_value_f64() {
    ensure_cuda();
    let b = gpu_f64(&[1.0, 3.0, 5.0, 7.0]);
    let v = gpu_f64(&[f64::INFINITY, f64::NEG_INFINITY]);
    assert_eq!(searchsorted(&b, &v, false).unwrap(), vec![4, 0]);
    assert_eq!(searchsorted(&b, &v, true).unwrap(), vec![4, 0]);
}

// NaN IN BOUNDARIES. LIVE torch: b=[1,3,5,7,NaN], v=[4,NaN,8] -> [2,5,5] both sides.
#[test]
fn gpu_searchsorted_nan_in_boundaries_f32() {
    ensure_cuda();
    let b = gpu_f32(&[1.0, 3.0, 5.0, 7.0, f32::NAN]);
    let v = gpu_f32(&[4.0, f32::NAN, 8.0]);
    assert_eq!(searchsorted(&b, &v, false).unwrap(), vec![2, 5, 5]);
    assert_eq!(searchsorted(&b, &v, true).unwrap(), vec![2, 5, 5]);
}
#[test]
fn gpu_searchsorted_nan_in_boundaries_f64() {
    ensure_cuda();
    let b = gpu_f64(&[1.0, 3.0, 5.0, 7.0, f64::NAN]);
    let v = gpu_f64(&[4.0, f64::NAN, 8.0]);
    assert_eq!(searchsorted(&b, &v, false).unwrap(), vec![2, 5, 5]);
    assert_eq!(searchsorted(&b, &v, true).unwrap(), vec![2, 5, 5]);
}

// FINITE REGRESSION (predicate-flip risk). LIVE torch values.
#[test]
fn gpu_searchsorted_finite_ties_f32() {
    ensure_cuda();
    let b = gpu_f32(&[1.0, 3.0, 5.0, 7.0]);
    let v = gpu_f32(&[1.0, 3.0, 5.0, 7.0]);
    assert_eq!(searchsorted(&b, &v, false).unwrap(), vec![0, 1, 2, 3]);
    assert_eq!(searchsorted(&b, &v, true).unwrap(), vec![1, 2, 3, 4]);
}
#[test]
fn gpu_searchsorted_finite_dup_oob_f64() {
    ensure_cuda();
    let bd = gpu_f64(&[1.0, 3.0, 3.0, 3.0, 5.0]);
    let v3 = gpu_f64(&[3.0]);
    assert_eq!(searchsorted(&bd, &v3, false).unwrap(), vec![1]);
    assert_eq!(searchsorted(&bd, &v3, true).unwrap(), vec![4]);
    let b = gpu_f64(&[1.0, 3.0, 5.0, 7.0]);
    let voob = gpu_f64(&[0.0, 8.0]);
    assert_eq!(searchsorted(&b, &voob, false).unwrap(), vec![0, 4]);
    assert_eq!(searchsorted(&b, &voob, true).unwrap(), vec![0, 4]);
}
