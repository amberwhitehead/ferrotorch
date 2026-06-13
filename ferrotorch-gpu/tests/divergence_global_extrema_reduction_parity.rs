//! Regression probes for global `amin`/`amax` parity.
//!
//! Live PyTorch 2.11.0+cu130 oracle:
//! - global extrema propagate NaNs;
//! - backward splits grad across equal extrema;
//! - NaN forward result produces all-NaN gradients;
//! - CUDA backward keeps the leaf gradient CUDA-resident.

#![cfg(feature = "cuda")]

use ferrotorch_core::{Device, Tensor, TensorStorage, backward};
use ferrotorch_gpu::init_cuda_backend;

fn ensure_cuda() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        init_cuda_backend().expect("CUDA backend init");
    });
}

fn cpu_f32(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
    Tensor::from_storage(
        TensorStorage::cpu(data.to_vec()),
        shape.to_vec(),
        requires_grad,
    )
    .expect("cpu f32 tensor")
}

fn cpu_f64(data: &[f64], shape: &[usize], requires_grad: bool) -> Tensor<f64> {
    Tensor::from_storage(
        TensorStorage::cpu(data.to_vec()),
        shape.to_vec(),
        requires_grad,
    )
    .expect("cpu f64 tensor")
}

fn host_f32(t: &Tensor<f32>) -> Vec<f32> {
    t.cpu().expect("cpu()").data().expect("data").to_vec()
}

fn host_f64(t: &Tensor<f64>) -> Vec<f64> {
    t.cpu().expect("cpu()").data().expect("data").to_vec()
}

fn grad_f32(t: &Tensor<f32>) -> Tensor<f32> {
    t.grad().expect("grad access").expect("grad must exist")
}

fn grad_f64(t: &Tensor<f64>) -> Tensor<f64> {
    t.grad().expect("grad access").expect("grad must exist")
}

fn cuda_leaf_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    cpu_f32(data, shape, false)
        .to(Device::Cuda(0))
        .expect("to cuda")
        .requires_grad_(true)
}

fn cuda_leaf_f64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    cpu_f64(data, shape, false)
        .to(Device::Cuda(0))
        .expect("to cuda")
        .requires_grad_(true)
}

fn assert_close_f32(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        if g.is_nan() && w.is_nan() {
            continue;
        }
        assert!((g - w).abs() <= 1e-6, "index {i}: got {g}, want {w}");
    }
}

fn assert_close_f64(got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        if g.is_nan() && w.is_nan() {
            continue;
        }
        assert!((g - w).abs() <= 1e-12, "index {i}: got {g}, want {w}");
    }
}

#[test]
fn cpu_global_extrema_propagate_nan_and_nan_backward() {
    let x = cpu_f32(&[1.0, f32::NAN, 3.0, f32::NAN], &[4], true);

    let y = x.amax().expect("amax");
    assert!(y.data().expect("amax data")[0].is_nan());
    backward(&y).expect("amax backward");
    assert!(host_f32(&grad_f32(&x)).iter().all(|v| v.is_nan()));

    let x = cpu_f64(&[1.0, f64::NAN, 3.0, f64::NAN], &[4], true);
    let y = x.amin().expect("amin");
    assert!(y.data().expect("amin data")[0].is_nan());
    backward(&y).expect("amin backward");
    assert!(host_f64(&grad_f64(&x)).iter().all(|v| v.is_nan()));
}

#[test]
fn cpu_global_extrema_split_tie_gradients() {
    let x = cpu_f32(&[3.0, 1.0, 3.0, 1.0], &[4], true);
    let y = x.amax().expect("amax");
    backward(&y).expect("amax backward");
    assert_close_f32(&host_f32(&grad_f32(&x)), &[0.5, 0.0, 0.5, 0.0]);

    let x = cpu_f64(&[3.0, 1.0, 3.0, 1.0], &[4], true);
    let y = x.amin().expect("amin");
    backward(&y).expect("amin backward");
    assert_close_f64(&host_f64(&grad_f64(&x)), &[0.0, 0.5, 0.0, 0.5]);
}

#[test]
fn cuda_global_extrema_propagate_nan_and_nan_backward_on_device() {
    ensure_cuda();

    let x = cuda_leaf_f32(&[1.0, f32::NAN, 3.0, f32::NAN], &[4]);
    let y = x.amax().expect("amax");
    assert!(y.is_cuda(), "forward result must stay CUDA-resident");
    assert!(host_f32(&y)[0].is_nan());
    backward(&y).expect("amax backward");
    let g = grad_f32(&x);
    assert!(g.is_cuda(), "leaf grad must stay CUDA-resident");
    assert!(host_f32(&g).iter().all(|v| v.is_nan()));

    let x = cuda_leaf_f64(&[1.0, f64::NAN, 3.0, f64::NAN], &[4]);
    let y = x.amin().expect("amin");
    assert!(y.is_cuda(), "f64 forward result must stay CUDA-resident");
    assert!(host_f64(&y)[0].is_nan());
    backward(&y).expect("amin backward");
    let g = grad_f64(&x);
    assert!(g.is_cuda(), "f64 leaf grad must stay CUDA-resident");
    assert!(host_f64(&g).iter().all(|v| v.is_nan()));
}

#[test]
fn cuda_global_extrema_split_tie_gradients_on_device() {
    ensure_cuda();

    let x = cuda_leaf_f32(&[3.0, 1.0, 3.0, 1.0], &[4]);
    backward(&x.amax().expect("amax")).expect("amax backward");
    let g = grad_f32(&x);
    assert!(g.is_cuda(), "f32 amax grad must stay CUDA-resident");
    assert_close_f32(&host_f32(&g), &[0.5, 0.0, 0.5, 0.0]);

    let x = cuda_leaf_f64(&[3.0, 1.0, 3.0, 1.0], &[4]);
    backward(&x.amin().expect("amin")).expect("amin backward");
    let g = grad_f64(&x);
    assert!(g.is_cuda(), "f64 amin grad must stay CUDA-resident");
    assert_close_f64(&host_f64(&g), &[0.0, 0.5, 0.0, 0.5]);
}

#[test]
fn cuda_offset_zero_narrowed_contiguous_view_reduces_logical_len_only() {
    ensure_cuda();

    let full = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false)
        .to(Device::Cuda(0))
        .expect("to cuda");
    let view = full.narrow(0, 0, 1).expect("narrow first row");
    assert!(view.is_cuda());
    assert!(view.is_contiguous());
    assert_eq!(view.storage_offset(), 0);
    assert_eq!(view.numel(), 3);
    assert_eq!(view.storage_len(), 6);

    let max = view.amax().expect("amax");
    let min = view.amin().expect("amin");
    assert!(max.is_cuda() && min.is_cuda());
    assert_close_f32(&host_f32(&max), &[3.0]);
    assert_close_f32(&host_f32(&min), &[1.0]);
}
