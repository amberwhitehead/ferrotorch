//! Regression probes for PyTorch-parity CUDA `std` / `var`.
//!
//! PyTorch 2.11.0+cu130 oracle:
//! - f32/f64 full and dim reductions run on CUDA;
//! - default correction over a single-element slice returns NaN;
//! - selected empty dimensions return NaN values in non-empty output slices;
//! - backward for `var` and `std` remains CUDA-resident.

#![cfg(feature = "cuda")]

use ferrotorch_core::grad_fns::reduction::{
    std_dim, std_with_correction, var_dim, var_with_correction,
};
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

fn host_f32(t: &Tensor<f32>) -> Vec<f32> {
    t.cpu().expect("cpu()").data().expect("data").to_vec()
}

fn host_f64(t: &Tensor<f64>) -> Vec<f64> {
    t.cpu().expect("cpu()").data().expect("data").to_vec()
}

fn assert_close_f32(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        if g.is_nan() && w.is_nan() {
            continue;
        }
        assert!((g - w).abs() <= 2e-6, "index {i}: got {g}, want {w}");
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
fn cuda_full_var_std_f32_forward_backward_stays_on_device() {
    ensure_cuda();

    let x = cuda_leaf_f32(&[1.0, 2.0, 3.0], &[3]);
    let y = var_with_correction(&x, 0.0).expect("var cuda f32");
    assert!(y.is_cuda());
    assert_close_f32(&host_f32(&y), &[2.0 / 3.0]);
    backward(&y).expect("var backward");
    let grad = x.grad().expect("grad access").expect("grad");
    assert!(grad.is_cuda(), "var grad must stay CUDA-resident");
    assert_close_f32(&host_f32(&grad), &[-2.0 / 3.0, 0.0, 2.0 / 3.0]);

    let x = cuda_leaf_f32(&[1.0, 2.0, 3.0], &[3]);
    let y = std_with_correction(&x, 0.0).expect("std cuda f32");
    assert!(y.is_cuda());
    assert_close_f32(&host_f32(&y), &[(2.0_f32 / 3.0).sqrt()]);
    backward(&y).expect("std backward");
    let grad = x.grad().expect("grad access").expect("grad");
    assert!(grad.is_cuda(), "std grad must stay CUDA-resident");
    assert_close_f32(&host_f32(&grad), &[-0.4082483, 0.0, 0.4082483]);
}

#[test]
fn cuda_dim_var_std_f64_keepdim_negative_dim() {
    ensure_cuda();
    let x = cuda_leaf_f64(&[1.0, 2.0, 3.0, 2.0, 4.0, 6.0], &[2, 3]);

    let y = var_dim(&x, -1, 1.0, true).expect("var_dim cuda f64");
    assert!(y.is_cuda());
    assert_eq!(y.shape(), &[2, 1]);
    assert_close_f64(&host_f64(&y), &[1.0, 4.0]);
    backward(&y.sum_all().expect("sum")).expect("var_dim backward");
    let grad = x.grad().expect("grad access").expect("grad");
    assert!(grad.is_cuda());
    assert_close_f64(&host_f64(&grad), &[-1.0, 0.0, 1.0, -2.0, 0.0, 2.0]);

    let x = cuda_leaf_f64(&[1.0, 2.0, 3.0, 2.0, 4.0, 6.0], &[2, 3]);
    let y = std_dim(&x, 1, 1.0, false).expect("std_dim cuda f64");
    assert!(y.is_cuda());
    assert_eq!(y.shape(), &[2]);
    assert_close_f64(&host_f64(&y), &[1.0, 2.0]);
    backward(&y.sum_all().expect("sum")).expect("std_dim backward");
    let grad = x.grad().expect("grad access").expect("grad");
    assert!(grad.is_cuda());
    assert_close_f64(&host_f64(&grad), &[-0.5, 0.0, 0.5, -0.5, 0.0, 0.5]);
}

#[test]
fn cuda_std_var_scalar_and_empty_axis_edges() {
    ensure_cuda();

    let scalar = cuda_leaf_f32(&[5.0], &[]);
    let y = var_dim(&scalar, 0, 0.0, true).expect("scalar var_dim correction 0");
    assert!(y.is_cuda());
    assert_eq!(y.shape(), &[]);
    assert_close_f32(&host_f32(&y), &[0.0]);
    let y = var_dim(&scalar, -1, 1.0, false).expect("scalar var_dim default correction");
    assert!(host_f32(&y)[0].is_nan());

    let empty = cpu_f32(&[], &[2, 0, 3], false)
        .to(Device::Cuda(0))
        .expect("empty to cuda");
    let y = std_dim(&empty, 1, 1.0, true).expect("empty selected axis std_dim");
    assert!(y.is_cuda());
    assert_eq!(y.shape(), &[2, 1, 3]);
    assert!(host_f32(&y).iter().all(|v| v.is_nan()));

    let empty_all = cpu_f64(&[], &[0], false)
        .to(Device::Cuda(0))
        .expect("empty full to cuda");
    let y = var_with_correction(&empty_all, -1.0).expect("empty full var");
    assert!(y.is_cuda());
    assert!(host_f64(&y)[0].is_nan());
}
