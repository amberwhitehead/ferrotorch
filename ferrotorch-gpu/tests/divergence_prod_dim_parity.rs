//! Regression probes for PyTorch-parity `prod(dim)`.
//!
//! PyTorch 2.11.0+cu130 oracle:
//! - `prod(dim)` returns the multiplicative identity `1` for zero-length
//!   selected dimensions;
//! - scalar dim 0/-1 is an identity;
//! - backward is `grad * product(slice except current index)`, covering
//!   no-zero, single-zero, and multi-zero slices;
//! - CUDA f32/f64 forward and backward remain device-resident.

#![cfg(feature = "cuda")]

use ferrotorch_core::grad_fns::reduction::prod_dim;
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

fn grad_f32(t: &Tensor<f32>) -> Tensor<f32> {
    t.grad().expect("grad access").expect("grad must exist")
}

fn grad_f64(t: &Tensor<f64>) -> Tensor<f64> {
    t.grad().expect("grad access").expect("grad must exist")
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
fn cpu_prod_dim_zero_cases_match_torch() {
    let x = cpu_f64(&[2.0, 3.0, 4.0], &[3], true);
    let y = prod_dim(&x, 0, false).expect("prod_dim no zero");
    assert_eq!(y.shape(), &[]);
    assert_close_f64(&host_f64(&y), &[24.0]);
    backward(&y).expect("backward");
    assert_close_f64(&host_f64(&grad_f64(&x)), &[12.0, 8.0, 6.0]);

    let x = cpu_f64(&[2.0, 0.0, 4.0], &[3], true);
    let y = prod_dim(&x, 0, false).expect("prod_dim single zero");
    assert_close_f64(&host_f64(&y), &[0.0]);
    backward(&y).expect("backward");
    assert_close_f64(&host_f64(&grad_f64(&x)), &[0.0, 8.0, 0.0]);

    let x = cpu_f64(&[0.0, 0.0, 4.0], &[3], true);
    let y = prod_dim(&x, 0, false).expect("prod_dim multi zero");
    assert_close_f64(&host_f64(&y), &[0.0]);
    backward(&y).expect("backward");
    assert_close_f64(&host_f64(&grad_f64(&x)), &[0.0, 0.0, 0.0]);
}

#[test]
fn cuda_prod_dim_f32_forward_backward_stays_on_device() {
    ensure_cuda();
    let x = cuda_leaf_f32(&[1.0, 2.0, 0.0, 3.0, 4.0, 5.0], &[2, 3]);

    let y = prod_dim(&x, 1, false).expect("prod_dim cuda f32");
    assert!(y.is_cuda(), "prod_dim output must stay CUDA-resident");
    assert_eq!(y.shape(), &[2]);
    assert_close_f32(&host_f32(&y), &[0.0, 60.0]);

    backward(&y.sum_all().expect("sum")).expect("backward");
    let grad = grad_f32(&x);
    assert!(grad.is_cuda(), "prod_dim grad must stay CUDA-resident");
    assert_close_f32(&host_f32(&grad), &[0.0, 0.0, 2.0, 20.0, 15.0, 12.0]);
}

#[test]
fn cuda_prod_dim_f64_keepdim_negative_dim() {
    ensure_cuda();
    let x = cuda_leaf_f64(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);

    let y = prod_dim(&x, -1, true).expect("prod_dim cuda f64");
    assert!(y.is_cuda());
    assert_eq!(y.shape(), &[2, 1]);
    assert_close_f64(&host_f64(&y), &[6.0, 20.0]);

    backward(&y.sum_all().expect("sum")).expect("backward");
    let grad = grad_f64(&x);
    assert!(grad.is_cuda());
    assert_close_f64(&host_f64(&grad), &[3.0, 2.0, 5.0, 4.0]);
}

#[test]
fn cuda_prod_dim_scalar_and_empty_axis_edges() {
    ensure_cuda();
    let scalar = cpu_f32(&[5.0], &[], false)
        .to(Device::Cuda(0))
        .expect("scalar to cuda")
        .requires_grad_(true);
    let y = prod_dim(&scalar, -1, true).expect("scalar prod_dim");
    assert!(y.is_cuda());
    assert_eq!(y.shape(), &[]);
    assert_close_f32(&host_f32(&y), &[5.0]);
    backward(&y).expect("scalar backward");
    let grad = grad_f32(&scalar);
    assert!(grad.is_cuda());
    assert_close_f32(&host_f32(&grad), &[1.0]);

    let empty = cpu_f32(&[], &[2, 0, 3], false)
        .to(Device::Cuda(0))
        .expect("empty to cuda")
        .requires_grad_(true);
    let y = prod_dim(&empty, 1, true).expect("empty selected axis prod_dim");
    assert!(y.is_cuda());
    assert_eq!(y.shape(), &[2, 1, 3]);
    assert_close_f32(&host_f32(&y), &[1.0; 6]);
    backward(&y.sum_all().expect("sum")).expect("empty backward");
    let grad = grad_f32(&empty);
    assert!(grad.is_cuda());
    assert_eq!(grad.shape(), &[2, 0, 3]);
    assert!(host_f32(&grad).is_empty());
}
