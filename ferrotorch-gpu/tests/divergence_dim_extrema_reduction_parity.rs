//! Regression probes for dim-keyed `amin`/`amax` parity.
//!
//! PyTorch 2.11.0+cu130 oracle:
//! - dim extrema propagate NaNs per reduced slice;
//! - backward uses `(grad / (result == input).sum(dim)) * (result == input)`;
//! - NaN-result slices produce NaN gradients for the whole slice;
//! - CUDA f32/f64 forward and backward remain device-resident.

#![cfg(feature = "cuda")]

use ferrotorch_core::grad_fns::reduction::{amax_dim, amin_dim};
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
fn cpu_dim_extrema_nan_slice_backward_matches_torch() {
    let x = cpu_f32(&[1.0, f32::NAN, 3.0, 2.0, 2.0, 1.0], &[2, 3], true);
    let y = amax_dim(&x, 1, false).expect("amax_dim");
    assert_eq!(y.shape(), &[2]);
    assert!(host_f32(&y)[0].is_nan());
    assert_close_f32(&host_f32(&y)[1..], &[2.0]);
    backward(&y.sum_all().expect("sum")).expect("backward");
    assert_close_f32(
        &host_f32(&grad_f32(&x)),
        &[f32::NAN, f32::NAN, f32::NAN, 0.5, 0.5, 0.0],
    );

    let x = cpu_f64(&[1.0, f64::NAN, 3.0, 2.0, 2.0, 1.0], &[2, 3], true);
    let y = amin_dim(&x, 1, false).expect("amin_dim");
    assert_eq!(y.shape(), &[2]);
    assert!(host_f64(&y)[0].is_nan());
    assert_close_f64(&host_f64(&y)[1..], &[1.0]);
    backward(&y.sum_all().expect("sum")).expect("backward");
    assert_close_f64(
        &host_f64(&grad_f64(&x)),
        &[f64::NAN, f64::NAN, f64::NAN, 0.0, 0.0, 1.0],
    );
}

#[test]
fn cuda_dim_extrema_f32_forward_backward_stays_on_device() {
    ensure_cuda();
    let x = cuda_leaf_f32(&[1.0, f32::NAN, 3.0, 2.0, 2.0, 1.0], &[2, 3]);

    let y = amax_dim(&x, 1, false).expect("amax_dim cuda");
    assert!(y.is_cuda(), "amax_dim output must stay CUDA-resident");
    assert_eq!(y.shape(), &[2]);
    let yh = host_f32(&y);
    assert!(yh[0].is_nan());
    assert_close_f32(&yh[1..], &[2.0]);

    backward(&y.sum_all().expect("sum")).expect("backward");
    let g = grad_f32(&x);
    assert!(g.is_cuda(), "amax_dim grad must stay CUDA-resident");
    assert_close_f32(
        &host_f32(&g),
        &[f32::NAN, f32::NAN, f32::NAN, 0.5, 0.5, 0.0],
    );
}

#[test]
fn cuda_dim_extrema_f64_forward_backward_stays_on_device() {
    ensure_cuda();
    let x = cuda_leaf_f64(&[1.0, f64::NAN, 3.0, 2.0, 2.0, 1.0], &[2, 3]);

    let y = amin_dim(&x, 1, false).expect("amin_dim cuda");
    assert!(y.is_cuda(), "amin_dim output must stay CUDA-resident");
    assert_eq!(y.shape(), &[2]);
    let yh = host_f64(&y);
    assert!(yh[0].is_nan());
    assert_close_f64(&yh[1..], &[1.0]);

    backward(&y.sum_all().expect("sum")).expect("backward");
    let g = grad_f64(&x);
    assert!(g.is_cuda(), "amin_dim grad must stay CUDA-resident");
    assert_close_f64(
        &host_f64(&g),
        &[f64::NAN, f64::NAN, f64::NAN, 0.0, 0.0, 1.0],
    );
}

#[test]
fn cuda_dim_extrema_keepdim_and_negative_dim() {
    ensure_cuda();
    let x = cuda_leaf_f32(&[1.0, 5.0, 3.0, 2.0, 5.0, 4.0], &[2, 3]);

    let y = amax_dim(&x, -1, true).expect("amax_dim keepdim");
    assert!(y.is_cuda());
    assert_eq!(y.shape(), &[2, 1]);
    assert_close_f32(&host_f32(&y), &[5.0, 5.0]);

    backward(&y.sum_all().expect("sum")).expect("backward");
    let g = grad_f32(&x);
    assert!(g.is_cuda());
    assert_close_f32(&host_f32(&g), &[0.0, 1.0, 0.0, 0.0, 1.0, 0.0]);
}
