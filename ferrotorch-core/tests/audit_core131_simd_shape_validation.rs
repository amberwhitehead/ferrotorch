//! CORE-131 (#1825): same-shape SIMD binary helpers must reject mismatched
//! shapes before entering ferray's debug-asserting zip kernels.
//!
//! PyTorch `add.Tensor` / `mul.Tensor` route through TensorIterator
//! (`aten/src/ATen/native/BinaryOps.cpp`), so public arithmetic broadcasts
//! compatible shapes and rejects incompatible ones. Ferrotorch's `fast_*`
//! wrappers provide that PyTorch-style surface. The direct `simd_*` helpers
//! are lower-level same-shape kernels and must fail structurally, not panic
//! in debug or return partially initialized output in release.

use std::panic::{AssertUnwindSafe, catch_unwind};

use ferrotorch_core::ops::elementwise::{
    fast_add, fast_mul, simd_add_f32, simd_add_f64, simd_mul_f32, simd_mul_f64,
};
use ferrotorch_core::{FerrotorchError, Tensor, TensorStorage};

fn cpu_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

fn cpu_f64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

#[track_caller]
fn assert_shape_err_no_panic<T: std::fmt::Debug>(
    label: &str,
    call: impl FnOnce() -> Result<T, FerrotorchError>,
) {
    let outcome = catch_unwind(AssertUnwindSafe(call));
    match outcome {
        Ok(Err(FerrotorchError::ShapeMismatch { .. })) => {}
        Ok(Err(other)) => panic!("{label}: expected ShapeMismatch, got {other:?}"),
        Ok(Ok(value)) => panic!("{label}: expected ShapeMismatch, got Ok({value:?})"),
        Err(_) => panic!("{label}: panicked instead of returning ShapeMismatch"),
    }
}

#[test]
fn core131_simd_binary_rejects_non_broadcastable_mismatches() {
    let a32 = cpu_f32(&[1.5; 6], &[2, 3]);
    let b32 = cpu_f32(&[0.25; 2], &[2]);
    let a64 = cpu_f64(&[1.5; 6], &[2, 3]);
    let b64 = cpu_f64(&[0.25; 2], &[2]);

    assert_shape_err_no_panic("simd_add_f32 [2,3] [2]", || simd_add_f32(&a32, &b32));
    assert_shape_err_no_panic("simd_mul_f32 [2,3] [2]", || simd_mul_f32(&a32, &b32));
    assert_shape_err_no_panic("simd_add_f64 [2,3] [2]", || simd_add_f64(&a64, &b64));
    assert_shape_err_no_panic("simd_mul_f64 [2,3] [2]", || simd_mul_f64(&a64, &b64));
}

#[test]
fn core131_simd_binary_rejects_equal_numel_broadcastable_mismatches() {
    let a32 = cpu_f32(&[1.0, 2.0], &[2, 1]);
    let b32 = cpu_f32(&[10.0, 20.0], &[1, 2]);
    let a64 = cpu_f64(&[1.0, 2.0], &[2, 1]);
    let b64 = cpu_f64(&[10.0, 20.0], &[1, 2]);

    assert_shape_err_no_panic("simd_add_f32 [2,1] [1,2]", || simd_add_f32(&a32, &b32));
    assert_shape_err_no_panic("simd_mul_f32 [2,1] [1,2]", || simd_mul_f32(&a32, &b32));
    assert_shape_err_no_panic("simd_add_f64 [2,1] [1,2]", || simd_add_f64(&a64, &b64));
    assert_shape_err_no_panic("simd_mul_f64 [2,1] [1,2]", || simd_mul_f64(&a64, &b64));

    let fast_added = fast_add(&a32, &b32).unwrap();
    assert_eq!(fast_added.shape(), &[2, 2]);
    assert_eq!(fast_added.data_vec().unwrap(), vec![11.0, 21.0, 12.0, 22.0]);

    let fast_multiplied = fast_mul(&a32, &b32).unwrap();
    assert_eq!(fast_multiplied.shape(), &[2, 2]);
    assert_eq!(
        fast_multiplied.data_vec().unwrap(),
        vec![10.0, 20.0, 20.0, 40.0]
    );
}

#[test]
fn core131_simd_binary_same_shape_still_uses_kernel_surface() {
    let a32 = cpu_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b32 = cpu_f32(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);
    assert_eq!(
        simd_add_f32(&a32, &b32).unwrap().data_vec().unwrap(),
        vec![11.0, 22.0, 33.0, 44.0]
    );
    assert_eq!(
        simd_mul_f32(&a32, &b32).unwrap().data_vec().unwrap(),
        vec![10.0, 40.0, 90.0, 160.0]
    );

    let a64 = cpu_f64(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b64 = cpu_f64(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);
    assert_eq!(
        simd_add_f64(&a64, &b64).unwrap().data_vec().unwrap(),
        vec![11.0, 22.0, 33.0, 44.0]
    );
    assert_eq!(
        simd_mul_f64(&a64, &b64).unwrap().data_vec().unwrap(),
        vec![10.0, 40.0, 90.0, 160.0]
    );
}
