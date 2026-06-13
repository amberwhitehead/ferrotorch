//! Regression probes for f16/bf16 CUDA `std`/`var` PyTorch parity.
//!
//! PyTorch 2.11.0+cu130 oracle:
//! - f16/bf16 CUDA `std`/`var` return the same reduced dtype;
//! - reductions and backward stay CUDA-resident;
//! - internal reduction math is wider, but final values/gradients round back
//!   to the input dtype;
//! - empty full reductions return NaN and `correction >= n` propagates NaN.

#![cfg(feature = "cuda")]

use ferrotorch_core::grad_fns::reduction::{
    std_dim as ft_std_dim, std_dims as ft_std_dims, var as ft_var, var_dim as ft_var_dim,
    var_dims as ft_var_dims,
};
use ferrotorch_core::{Device, Tensor, TensorStorage, backward};
use ferrotorch_gpu::init_cuda_backend;
use half::{bf16, f16};

fn ensure_cuda() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        init_cuda_backend().expect("CUDA backend init");
    });
}

fn cpu_f16(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f16> {
    Tensor::from_storage(
        TensorStorage::cpu(data.iter().copied().map(f16::from_f32).collect()),
        shape.to_vec(),
        requires_grad,
    )
    .expect("cpu f16 tensor")
}

fn cpu_bf16(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<bf16> {
    Tensor::from_storage(
        TensorStorage::cpu(data.iter().copied().map(bf16::from_f32).collect()),
        shape.to_vec(),
        requires_grad,
    )
    .expect("cpu bf16 tensor")
}

fn cuda_leaf_f16(data: &[f32], shape: &[usize]) -> Tensor<f16> {
    cpu_f16(data, shape, false)
        .to(Device::Cuda(0))
        .expect("to cuda")
        .requires_grad_(true)
}

fn cuda_leaf_bf16(data: &[f32], shape: &[usize]) -> Tensor<bf16> {
    cpu_bf16(data, shape, false)
        .to(Device::Cuda(0))
        .expect("to cuda")
        .requires_grad_(true)
}

fn host_f16(t: &Tensor<f16>) -> Vec<f32> {
    t.cpu()
        .expect("cpu()")
        .data()
        .expect("data")
        .iter()
        .map(|v| v.to_f32())
        .collect()
}

fn host_bf16(t: &Tensor<bf16>) -> Vec<f32> {
    t.cpu()
        .expect("cpu()")
        .data()
        .expect("data")
        .iter()
        .map(|v| v.to_f32())
        .collect()
}

fn assert_close(got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        if g.is_nan() && w.is_nan() {
            continue;
        }
        assert!(
            (g - w).abs() <= tol,
            "index {i}: got {g}, want {w}, tol {tol}"
        );
    }
}

#[test]
fn cuda_var_dim_f16_forward_backward_stays_half_and_cuda() {
    ensure_cuda();
    let x = cuda_leaf_f16(&[1.0, 2.0, 3.0, 2.0, 4.0, 6.0], &[2, 3]);
    let y = ft_var_dim(&x, 1, 1.0, false).expect("f16 var_dim");
    assert!(y.is_cuda());
    assert_eq!(y.shape(), &[2]);
    assert_close(&host_f16(&y), &[1.0, 4.0], 0.0);

    backward(&y.sum_all().expect("sum")).expect("backward");
    let grad = x.grad().expect("grad access").expect("grad");
    assert!(grad.is_cuda());
    assert_close(&host_f16(&grad), &[-1.0, 0.0, 1.0, -2.0, 0.0, 2.0], 0.0);
}

#[test]
fn cuda_std_dim_bf16_forward_backward_stays_bfloat_and_cuda() {
    ensure_cuda();
    let x = cuda_leaf_bf16(&[1.0, 2.0, 3.0, 2.0, 4.0, 6.0], &[2, 3]);
    let y = ft_std_dim(&x, -1, 1.0, false).expect("bf16 std_dim");
    assert!(y.is_cuda());
    assert_eq!(y.shape(), &[2]);
    assert_close(&host_bf16(&y), &[1.0, 2.0], 0.0);

    backward(&y.sum_all().expect("sum")).expect("backward");
    let grad = x.grad().expect("grad access").expect("grad");
    assert!(grad.is_cuda());
    assert_close(&host_bf16(&grad), &[-0.5, 0.0, 0.5, -0.5, 0.0, 0.5], 0.0);
}

#[test]
fn cuda_var_dims_f16_uses_u16_strided_copy_and_stays_cuda() {
    ensure_cuda();
    let data: Vec<f32> = (0..24).map(|v| v as f32).collect();
    let x = cuda_leaf_f16(&data, &[2, 3, 4]);
    let y = ft_var_dims(&x, &[0, -1], 1.0, false).expect("f16 var_dims");
    assert!(y.is_cuda());
    assert_eq!(y.shape(), &[3]);
    assert_close(&host_f16(&y), &[42.5625, 42.5625, 42.5625], 0.0);
}

#[test]
fn cuda_std_dims_bf16_non_adjacent_axes_backward_stays_cuda() {
    ensure_cuda();
    let data: Vec<f32> = (0..24).map(|v| v as f32).collect();
    let x = cuda_leaf_bf16(&data, &[2, 3, 4]);
    let y = ft_std_dims(&x, &[1, 2], 1.0, true).expect("bf16 std_dims");
    assert!(y.is_cuda());
    assert_eq!(y.shape(), &[2, 1, 1]);
    assert_close(&host_bf16(&y), &[3.609375, 3.609375], 0.0);

    backward(&y.sum_all().expect("sum")).expect("std_dims backward");
    let grad = x.grad().expect("grad access").expect("grad");
    assert!(grad.is_cuda());
}

#[test]
fn cuda_full_var_half_edges_match_torch() {
    ensure_cuda();

    let x = cpu_f16(&[], &[0], false)
        .to(Device::Cuda(0))
        .expect("to cuda");
    let y = ft_var(&x, false).expect("empty f16 var");
    assert!(y.is_cuda());
    assert!(host_f16(&y)[0].is_nan());

    let x = cpu_bf16(&[], &[0], false)
        .to(Device::Cuda(0))
        .expect("to cuda");
    let y = ft_var(&x, false).expect("empty bf16 var");
    assert!(y.is_cuda());
    assert!(host_bf16(&y)[0].is_nan());

    let x = cuda_leaf_f16(&[3.0, 4.0], &[2, 1]);
    let y = ft_var_dim(&x, 1, 1.0, false).expect("denom zero var_dim");
    assert!(host_f16(&y).iter().all(|v| v.is_nan()));
    backward(&y.sum_all().expect("sum")).expect("backward denom zero");
    let grad = x.grad().expect("grad access").expect("grad");
    assert!(host_f16(&grad).iter().all(|v| v.is_nan()));

    let x = cuda_leaf_bf16(&[3.0, 4.0], &[2, 1]);
    let y = ft_var_dim(&x, 1, 1.0, false).expect("bf16 denom zero var_dim");
    assert!(host_bf16(&y).iter().all(|v| v.is_nan()));
    backward(&y.sum_all().expect("sum")).expect("bf16 backward denom zero");
    let grad = x.grad().expect("grad access").expect("grad");
    assert!(host_bf16(&grad).iter().all(|v| v.is_nan()));
}
