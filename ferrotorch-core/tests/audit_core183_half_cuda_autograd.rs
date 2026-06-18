#![cfg(feature = "gpu")]

//! CORE-183: reduced-precision CUDA autograd must not stop at forward.
//!
//! PyTorch 2.11.0+cu130 oracles on this host:
//! - `abs([-2, -0, 0, 3, nan]).backward(weight=[5,6,7,8,9])`
//!   gives `[-5, 0, 0, 8, 0]` for f16, bf16, and f32 on CUDA.
//! - broadcast add backward routes through `_grad_sum_to_size`, summing the
//!   upstream gradient over every expanded axis for f16/bf16 CUDA tensors.

use std::sync::Once;

use ferrotorch_core::creation::from_vec;
use ferrotorch_core::device::Device;
use ferrotorch_core::grad_fns::arithmetic::{abs, add, mul};
use ferrotorch_core::grad_fns::reduction::sum;
use ferrotorch_core::tensor::Tensor;
use half::{bf16, f16};

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for CORE-183 CUDA half tests");
    });
}

fn f16_cuda(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f16> {
    let values: Vec<f16> = data.iter().copied().map(f16::from_f32).collect();
    from_vec::<f16>(values, shape)
        .expect("f16 CPU tensor")
        .to(Device::Cuda(0))
        .expect("upload f16")
        .requires_grad_(requires_grad)
}

fn bf16_cuda(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<bf16> {
    let values: Vec<bf16> = data.iter().copied().map(bf16::from_f32).collect();
    from_vec::<bf16>(values, shape)
        .expect("bf16 CPU tensor")
        .to(Device::Cuda(0))
        .expect("upload bf16")
        .requires_grad_(requires_grad)
}

fn host_f16(t: &Tensor<f16>) -> Vec<f32> {
    assert_eq!(t.device(), Device::Cuda(0), "tensor must stay on CUDA");
    t.cpu()
        .expect("D2H f16")
        .data()
        .expect("f16 CPU data")
        .iter()
        .map(|v| v.to_f32())
        .collect()
}

fn host_bf16(t: &Tensor<bf16>) -> Vec<f32> {
    assert_eq!(t.device(), Device::Cuda(0), "tensor must stay on CUDA");
    t.cpu()
        .expect("D2H bf16")
        .data()
        .expect("bf16 CPU data")
        .iter()
        .map(|v| v.to_f32())
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (got - want).abs() <= tol,
            "{label}[{i}] got {got}, want {want}, tol {tol}"
        );
    }
}

fn assert_f16_grad(leaf: &Tensor<f16>, shape: &[usize], expected: &[f32], label: &str) {
    let grad = leaf
        .grad()
        .expect("f16 grad slot")
        .unwrap_or_else(|| panic!("{label}: no gradient reached f16 CUDA leaf"));
    assert_eq!(grad.device(), Device::Cuda(0), "{label}: grad device");
    assert_eq!(grad.shape(), shape, "{label}: grad shape");
    assert_close(&host_f16(&grad), expected, 1e-3, label);
}

fn assert_bf16_grad(leaf: &Tensor<bf16>, shape: &[usize], expected: &[f32], label: &str) {
    let grad = leaf
        .grad()
        .expect("bf16 grad slot")
        .unwrap_or_else(|| panic!("{label}: no gradient reached bf16 CUDA leaf"));
    assert_eq!(grad.device(), Device::Cuda(0), "{label}: grad device");
    assert_eq!(grad.shape(), shape, "{label}: grad shape");
    assert_close(&host_bf16(&grad), expected, 1e-2, label);
}

#[test]
fn f16_broadcast_add_backward_reduces_multiple_axes_on_cuda() {
    ensure_cuda_backend();

    let a = f16_cuda(&[1., 2., 3., 4., 5., 6.], &[2, 1, 3], true);
    let b = f16_cuda(&[10., 20., 30., 40.], &[1, 4, 1], true);
    let weights = f16_cuda(
        &[
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
            20., 21., 22., 23., 24.,
        ],
        &[2, 4, 3],
        false,
    );

    let out = add(&a, &b).expect("f16 CUDA broadcast add");
    assert_eq!(out.device(), Device::Cuda(0));
    assert_eq!(out.shape(), &[2, 4, 3]);
    let loss = sum(&mul(&out, &weights).expect("weighted f16 add output")).expect("f16 loss");
    loss.backward().expect("f16 broadcast add backward");

    assert_f16_grad(
        &a,
        &[2, 1, 3],
        &[22., 26., 30., 70., 74., 78.],
        "f16 a.grad",
    );
    assert_f16_grad(&b, &[1, 4, 1], &[48., 66., 84., 102.], "f16 b.grad");
}

#[test]
fn bf16_broadcast_add_backward_reduces_multiple_axes_on_cuda() {
    ensure_cuda_backend();

    let a = bf16_cuda(&[1., 2., 3., 4., 5., 6.], &[2, 1, 3], true);
    let b = bf16_cuda(&[10., 20., 30., 40.], &[1, 4, 1], true);
    let weights = bf16_cuda(
        &[
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
            20., 21., 22., 23., 24.,
        ],
        &[2, 4, 3],
        false,
    );

    let out = add(&a, &b).expect("bf16 CUDA broadcast add");
    assert_eq!(out.device(), Device::Cuda(0));
    assert_eq!(out.shape(), &[2, 4, 3]);
    let loss = sum(&mul(&out, &weights).expect("weighted bf16 add output")).expect("bf16 loss");
    loss.backward().expect("bf16 broadcast add backward");

    assert_bf16_grad(
        &a,
        &[2, 1, 3],
        &[22., 26., 30., 70., 74., 78.],
        "bf16 a.grad",
    );
    assert_bf16_grad(&b, &[1, 4, 1], &[48., 66., 84., 102.], "bf16 b.grad");
}

#[test]
fn f16_abs_backward_uses_cuda_sign_with_zero_and_nan_policy() {
    ensure_cuda_backend();

    let x = f16_cuda(&[-2., -0., 0., 3., f32::NAN], &[5], true);
    let weights = f16_cuda(&[5., 6., 7., 8., 9.], &[5], false);
    let loss = sum(&mul(&abs(&x).expect("f16 CUDA abs"), &weights).expect("weighted f16 abs"))
        .expect("f16 abs loss");
    loss.backward().expect("f16 abs backward");

    assert_f16_grad(&x, &[5], &[-5., 0., 0., 8., 0.], "f16 abs grad");
}

#[test]
fn bf16_abs_backward_uses_cuda_sign_with_zero_and_nan_policy() {
    ensure_cuda_backend();

    let x = bf16_cuda(&[-2., -0., 0., 3., f32::NAN], &[5], true);
    let weights = bf16_cuda(&[5., 6., 7., 8., 9.], &[5], false);
    let loss = sum(&mul(&abs(&x).expect("bf16 CUDA abs"), &weights).expect("weighted bf16 abs"))
        .expect("bf16 abs loss");
    loss.backward().expect("bf16 abs backward");

    assert_bf16_grad(&x, &[5], &[-5., 0., 0., 8., 0.], "bf16 abs grad");
}
