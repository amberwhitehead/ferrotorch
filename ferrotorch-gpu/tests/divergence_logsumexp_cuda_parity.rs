//! Regression probes for CUDA `logsumexp` PyTorch parity.
//!
//! PyTorch 2.11.0+cu130 oracle:
//! - f32/f64 full and dim reductions run on CUDA;
//! - NaN propagates, +inf wins, all -inf and empty selected slices return -inf;
//! - scalar dim 0/-1 returns the scalar;
//! - backward is `grad * exp(input - result)` and stays CUDA-resident.

#![cfg(feature = "cuda")]

use ferrotorch_core::grad_fns::reduction::{logsumexp, logsumexp_dim};
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
        assert!((g - w).abs() <= 2e-5, "index {i}: got {g}, want {w}");
    }
}

fn assert_close_f64(got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        if g.is_nan() && w.is_nan() {
            continue;
        }
        assert!((g - w).abs() <= 2e-10, "index {i}: got {g}, want {w}");
    }
}

#[test]
fn cuda_logsumexp_full_f32_forward_backward_stays_on_device() {
    ensure_cuda();
    let x = cuda_leaf_f32(&[1.0, 2.0, 3.0], &[3]);
    let y = logsumexp(&x).expect("logsumexp f32");
    assert!(y.is_cuda());
    assert_eq!(y.shape(), &[]);
    assert_close_f32(&host_f32(&y), &[3.407606]);

    backward(&y).expect("backward");
    let grad = x.grad().expect("grad access").expect("grad");
    assert!(grad.is_cuda(), "logsumexp grad must stay CUDA-resident");
    assert_close_f32(&host_f32(&grad), &[0.09003057, 0.24472847, 0.66524096]);
}

#[test]
fn cuda_logsumexp_dim_f64_keepdim_and_backward() {
    ensure_cuda();
    let x = cuda_leaf_f64(&[1.0, 2.0, 3.0, 2.0, 4.0, 6.0], &[2, 3]);
    let y = logsumexp_dim(&x, -1, true).expect("logsumexp_dim f64");
    assert!(y.is_cuda());
    assert_eq!(y.shape(), &[2, 1]);
    assert_close_f64(&host_f64(&y), &[3.4076059644443806, 6.142931628499899]);

    backward(&y.sum_all().expect("sum")).expect("backward");
    let grad = x.grad().expect("grad access").expect("grad");
    assert!(grad.is_cuda());
    assert_close_f64(
        &host_f64(&grad),
        &[
            0.09003057317038046,
            0.24472847105479767,
            0.6652409557748218,
            0.015876239976466765,
            0.11731042782619837,
            0.8668133321973349,
        ],
    );
}

#[test]
fn cuda_logsumexp_dim_f32_squeezed_backward_stays_on_device() {
    ensure_cuda();
    let x = cuda_leaf_f32(&[1.0, 2.0, 3.0, 2.0, 4.0, 6.0], &[2, 3]);
    let y = logsumexp_dim(&x, 1, false).expect("logsumexp_dim squeezed f32");
    assert!(y.is_cuda());
    assert_eq!(y.shape(), &[2]);
    assert_close_f32(&host_f32(&y), &[3.407606, 6.142932]);

    backward(&y.sum_all().expect("sum")).expect("backward");
    let grad = x.grad().expect("grad access").expect("grad");
    assert!(grad.is_cuda());
    assert_close_f32(
        &host_f32(&grad),
        &[
            0.09003057, 0.24472847, 0.66524096, 0.01587624, 0.11731043, 0.86681336,
        ],
    );
}

#[test]
fn cuda_logsumexp_edges_match_torch() {
    ensure_cuda();

    let x = cpu_f32(&[f32::NAN, 1.0], &[2], false)
        .to(Device::Cuda(0))
        .expect("nan to cuda");
    assert!(host_f32(&logsumexp(&x).expect("nan logsumexp"))[0].is_nan());

    let x = cpu_f32(&[f32::INFINITY, 1.0], &[2], false)
        .to(Device::Cuda(0))
        .expect("inf to cuda");
    assert_eq!(
        host_f32(&logsumexp(&x).expect("inf logsumexp"))[0],
        f32::INFINITY
    );

    let x = cpu_f64(&[f64::NEG_INFINITY, f64::NEG_INFINITY], &[2], false)
        .to(Device::Cuda(0))
        .expect("-inf to cuda");
    assert_eq!(
        host_f64(&logsumexp(&x).expect("all -inf logsumexp"))[0],
        f64::NEG_INFINITY
    );

    let scalar = cpu_f32(&[3.0], &[], false)
        .to(Device::Cuda(0))
        .expect("scalar to cuda");
    let y = logsumexp_dim(&scalar, -1, true).expect("scalar dim");
    assert!(y.is_cuda());
    assert_eq!(y.shape(), &[]);
    assert_close_f32(&host_f32(&y), &[3.0]);

    let scalar = cuda_leaf_f32(&[3.0], &[]);
    let y = logsumexp_dim(&scalar, 0, false).expect("scalar dim backward");
    backward(&y).expect("scalar dim backward");
    let grad = scalar
        .grad()
        .expect("scalar grad access")
        .expect("scalar grad");
    assert!(grad.is_cuda());
    assert_eq!(grad.shape(), &[]);
    assert_close_f32(&host_f32(&grad), &[1.0]);

    let empty_axis = cpu_f32(&[], &[2, 0, 3], false)
        .to(Device::Cuda(0))
        .expect("empty axis to cuda");
    let y = logsumexp_dim(&empty_axis, 1, false).expect("empty selected axis");
    assert!(y.is_cuda());
    assert_eq!(y.shape(), &[2, 3]);
    assert!(host_f32(&y).iter().all(|v| *v == f32::NEG_INFINITY));

    let empty_full = cpu_f64(&[], &[0], false)
        .to(Device::Cuda(0))
        .expect("empty full to cuda");
    let y = logsumexp(&empty_full).expect("empty full");
    assert!(y.is_cuda());
    assert_eq!(host_f64(&y)[0], f64::NEG_INFINITY);

    let empty_full = cpu_f32(&[], &[0], false)
        .to(Device::Cuda(0))
        .expect("empty full f32 to cuda");
    let y = logsumexp(&empty_full).expect("empty full f32");
    assert!(y.is_cuda());
    assert_eq!(host_f32(&y)[0], f32::NEG_INFINITY);
}
