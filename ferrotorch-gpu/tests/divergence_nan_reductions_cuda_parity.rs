//! Regression probes for CUDA `nansum` / `nanmean` PyTorch parity.

#![cfg(feature = "cuda")]

use ferrotorch_core::grad_fns::reduction::{
    nanmean, nanmean_dim, nanmean_dims, nansum, nansum_dim, nansum_dims,
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
        assert!((g - w).abs() <= 1e-6, "idx {i}: got {g}, want {w}");
    }
}

fn assert_close_f64(got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        if g.is_nan() && w.is_nan() {
            continue;
        }
        assert!((g - w).abs() <= 1e-12, "idx {i}: got {g}, want {w}");
    }
}

#[test]
fn cuda_nansum_dim_f32_masks_backward() {
    ensure_cuda();
    let x = cuda_leaf_f32(&[1.0, f32::NAN, 3.0, f32::NAN, 5.0, 7.0], &[2, 3]);
    let y = nansum_dim(&x, 1, false).expect("nansum_dim");
    assert!(y.is_cuda());
    assert_eq!(y.shape(), &[2]);
    assert_close_f32(&host_f32(&y), &[4.0, 12.0]);

    backward(&y.sum_all().expect("sum")).expect("backward");
    let grad = x.grad().expect("grad access").expect("grad");
    assert!(grad.is_cuda());
    assert_close_f32(&host_f32(&grad), &[1.0, 0.0, 1.0, 0.0, 1.0, 1.0]);
}

#[test]
fn cuda_nanmean_dim_f32_counts_non_nan_and_matches_all_nan_grad() {
    ensure_cuda();
    let x = cuda_leaf_f32(&[1.0, f32::NAN, 3.0, f32::NAN, f32::NAN, f32::NAN], &[2, 3]);
    let y = nanmean_dim(&x, 1, false).expect("nanmean_dim");
    assert!(y.is_cuda());
    assert_close_f32(&host_f32(&y), &[2.0, f32::NAN]);

    backward(&y.sum_all().expect("sum")).expect("backward");
    let grad = x.grad().expect("grad access").expect("grad");
    assert!(grad.is_cuda());
    assert_close_f32(
        &host_f32(&grad),
        &[0.5, 0.0, 0.5, f32::NAN, f32::NAN, f32::NAN],
    );
}

#[test]
fn cuda_nan_reduction_dims_f32_non_adjacent_axes() {
    ensure_cuda();
    let x = cuda_leaf_f32(
        &[
            1.0,
            f32::NAN,
            3.0,
            4.0,
            5.0,
            6.0,
            f32::NAN,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
        ],
        &[2, 2, 3],
    );
    let y = nansum_dims(&x, &[0, -1], false).expect("nansum_dims");
    assert!(y.is_cuda());
    assert_eq!(y.shape(), &[2]);
    assert_close_f32(&host_f32(&y), &[21.0, 48.0]);

    let y = nanmean_dims(&x, &[0, 2], true).expect("nanmean_dims");
    assert!(y.is_cuda());
    assert_eq!(y.shape(), &[1, 2, 1]);
    assert_close_f32(&host_f32(&y), &[5.25, 8.0]);
}

#[test]
fn cuda_nan_reductions_full_f64_forward_backward_edges() {
    ensure_cuda();
    let x = cuda_leaf_f64(&[1.0, f64::NAN, 3.0], &[3]);
    let y = nansum(&x).expect("nansum");
    assert!(y.is_cuda());
    assert_eq!(y.shape(), &[]);
    assert_close_f64(&host_f64(&y), &[4.0]);
    backward(&y).expect("backward");
    let grad = x.grad().expect("grad access").expect("grad");
    assert_close_f64(&host_f64(&grad), &[1.0, 0.0, 1.0]);

    let x = cuda_leaf_f64(&[1.0, f64::NAN, 3.0], &[3]);
    let y = nanmean(&x).expect("nanmean");
    assert_close_f64(&host_f64(&y), &[2.0]);
    backward(&y).expect("backward");
    let grad = x.grad().expect("grad access").expect("grad");
    assert_close_f64(&host_f64(&grad), &[0.5, 0.0, 0.5]);

    let empty = cpu_f32(&[], &[0], false)
        .to(Device::Cuda(0))
        .expect("to cuda");
    assert_eq!(host_f32(&nansum(&empty).expect("empty nansum"))[0], 0.0);
    assert!(host_f32(&nanmean(&empty).expect("empty nanmean"))[0].is_nan());
}
