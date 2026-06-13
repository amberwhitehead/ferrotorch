//! Regression probes for PyTorch-parity `max(dim)` / `min(dim)`.
//!
//! PyTorch 2.11.0+cu130 oracle:
//! - tuple reductions return `(values, int64 indices)`;
//! - ties select the first index;
//! - the first NaN in a slice poisons the value and owns the gradient;
//! - scalar dim 0/-1 is an identity with index 0;
//! - empty selected dimensions error only when the output would be non-empty;
//! - CUDA f32/f64 forward and backward stay device-resident.

#![cfg(feature = "cuda")]

use ferrotorch_core::grad_fns::reduction::{max_with_dim, min_with_dim};
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

fn host_i64(t: &ferrotorch_core::IntTensor<i64>) -> Vec<i64> {
    t.to(Device::Cpu)
        .expect("indices to cpu")
        .data()
        .expect("indices data")
        .to_vec()
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
fn cpu_value_selecting_reduction_first_nan_and_tie_backward() {
    let x = cpu_f32(&[f32::NAN, 1.0, f32::NAN, 2.0, 2.0, 1.0], &[2, 3], true);
    let (values, indices) = max_with_dim(&x, 1, false).expect("max_with_dim cpu");
    assert_eq!(values.shape(), &[2]);
    assert_eq!(indices.shape(), &[2]);
    let vals = host_f32(&values);
    assert!(vals[0].is_nan());
    assert_close_f32(&vals[1..], &[2.0]);
    assert_eq!(host_i64(&indices), vec![0, 0]);

    backward(&values.sum_all().expect("sum")).expect("backward");
    assert_close_f32(&host_f32(&grad_f32(&x)), &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);

    let x = cpu_f32(&[1.0, f32::NAN, 2.0, 5.0, 5.0, 6.0], &[2, 3], true);
    let (values, indices) = min_with_dim(&x, 1, false).expect("min_with_dim cpu");
    let vals = host_f32(&values);
    assert!(vals[0].is_nan());
    assert_close_f32(&vals[1..], &[5.0]);
    assert_eq!(host_i64(&indices), vec![1, 0]);
    backward(&values.sum_all().expect("sum")).expect("backward");
    assert_close_f32(&host_f32(&grad_f32(&x)), &[0.0, 1.0, 0.0, 1.0, 0.0, 0.0]);
}

#[test]
fn cuda_max_with_dim_f32_forward_backward_stays_on_device() {
    ensure_cuda();
    let x = cuda_leaf_f32(&[f32::NAN, 1.0, f32::NAN, 2.0, 2.0, 1.0], &[2, 3]);

    let (values, indices) = max_with_dim(&x, 1, false).expect("max_with_dim cuda");
    assert!(values.is_cuda(), "values must stay CUDA-resident");
    assert!(indices.is_cuda(), "indices must stay CUDA-resident");
    assert_eq!(values.shape(), &[2]);
    assert_eq!(indices.shape(), &[2]);
    let vals = host_f32(&values);
    assert!(vals[0].is_nan());
    assert_close_f32(&vals[1..], &[2.0]);
    assert_eq!(host_i64(&indices), vec![0, 0]);

    backward(&values.sum_all().expect("sum")).expect("backward");
    let grad = grad_f32(&x);
    assert!(grad.is_cuda(), "grad must stay CUDA-resident");
    assert_close_f32(&host_f32(&grad), &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
}

#[test]
fn cuda_min_with_dim_f64_keepdim_negative_dim_backward() {
    ensure_cuda();
    let x = cuda_leaf_f64(&[4.0, 3.0, 3.0, 1.0, 2.0, 1.0], &[2, 3]);

    let (values, indices) = min_with_dim(&x, -1, true).expect("min_with_dim cuda f64");
    assert!(values.is_cuda());
    assert!(indices.is_cuda());
    assert_eq!(values.shape(), &[2, 1]);
    assert_eq!(indices.shape(), &[2, 1]);
    assert_close_f64(&host_f64(&values), &[3.0, 1.0]);
    assert_eq!(host_i64(&indices), vec![1, 0]);

    backward(&values.sum_all().expect("sum")).expect("backward");
    let grad = grad_f64(&x);
    assert!(grad.is_cuda());
    assert_close_f64(&host_f64(&grad), &[0.0, 1.0, 0.0, 1.0, 0.0, 0.0]);
}

#[test]
fn scalar_and_empty_dim_edges_match_torch() {
    ensure_cuda();
    let scalar = cpu_f32(&[5.0], &[], false)
        .to(Device::Cuda(0))
        .expect("scalar to cuda")
        .requires_grad_(true);
    let (values, indices) = max_with_dim(&scalar, 0, false).expect("scalar max");
    assert!(values.is_cuda());
    assert!(indices.is_cuda());
    assert_eq!(values.shape(), &[]);
    assert_eq!(indices.shape(), &[]);
    assert_close_f32(&host_f32(&values), &[5.0]);
    assert_eq!(host_i64(&indices), vec![0]);
    backward(&values).expect("scalar backward");
    let grad = grad_f32(&scalar);
    assert!(grad.is_cuda());
    assert_close_f32(&host_f32(&grad), &[1.0]);

    let empty = cpu_f32(&[], &[2, 0, 3], false)
        .to(Device::Cuda(0))
        .expect("empty to cuda");
    let (values, indices) = max_with_dim(&empty, 0, false).expect("zero-output max");
    assert!(values.is_cuda());
    assert!(indices.is_cuda());
    assert_eq!(values.shape(), &[0, 3]);
    assert_eq!(indices.shape(), &[0, 3]);
    assert!(host_f32(&values).is_empty());
    assert!(host_i64(&indices).is_empty());

    let err = min_with_dim(&empty, 1, false).expect_err("selected empty dim must error");
    assert!(
        format!("{err}").contains("non-zero size"),
        "unexpected error: {err}"
    );
}
