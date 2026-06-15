#![cfg(feature = "gpu")]

//! CUDA bf16 arithmetic parity for `sqrt`, `rsqrt`, and scalar-filled compose
//! paths.
//!
//! Live PyTorch 2.11.0+cu130 oracle on this machine:
//! - `torch.sqrt(torch.tensor([0,1,2,4,inf,nan], device="cuda",
//!   dtype=torch.bfloat16))` -> `[0, 1, 1.4140625, 2, inf, nan]`
//! - `torch.sqrt(torch.tensor([1,4,9], device="cuda", dtype=torch.bfloat16,
//!   requires_grad=True)).sum().backward()` -> grad
//!   `[0.5, 0.25, 0.1669921875]`, dtype bf16, device cuda
//! - `torch.rsqrt([1,4,9,inf,nan])` -> `[1, 0.5, 0.333984375, 0, nan]`
//! - `torch.rsqrt([1,4,9]).sum().backward()` -> grad
//!   `[-0.5, -0.0625, -0.0185546875]`, dtype bf16, device cuda

use std::sync::Once;

use ferrotorch_core::creation::from_vec;
use ferrotorch_core::device::Device;
use ferrotorch_core::grad_fns::arithmetic::{div, reciprocal, rsqrt, sqrt};
use ferrotorch_core::grad_fns::reduction::sum;
use ferrotorch_core::tensor::Tensor;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for bf16 sqrt CUDA tests");
    });
}

fn bf16_cuda(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<half::bf16> {
    let values: Vec<half::bf16> = data.iter().copied().map(half::bf16::from_f32).collect();
    from_vec::<half::bf16>(values, shape)
        .expect("bf16 CPU tensor")
        .to(Device::Cuda(0))
        .expect("upload bf16")
        .requires_grad_(requires_grad)
}

fn f16_cuda(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<half::f16> {
    let values: Vec<half::f16> = data.iter().copied().map(half::f16::from_f32).collect();
    from_vec::<half::f16>(values, shape)
        .expect("f16 CPU tensor")
        .to(Device::Cuda(0))
        .expect("upload f16")
        .requires_grad_(requires_grad)
}

fn host_bf16(t: &Tensor<half::bf16>) -> Vec<f32> {
    assert_eq!(
        t.device(),
        Device::Cuda(0),
        "result must stay CUDA-resident"
    );
    t.cpu()
        .expect("bf16 D2H")
        .data()
        .expect("bf16 CPU data")
        .iter()
        .map(|v| v.to_f32())
        .collect()
}

fn host_f16(t: &Tensor<half::f16>) -> Vec<f32> {
    assert_eq!(
        t.device(),
        Device::Cuda(0),
        "result must stay CUDA-resident"
    );
    t.cpu()
        .expect("f16 D2H")
        .data()
        .expect("f16 CPU data")
        .iter()
        .map(|v| v.to_f32())
        .collect()
}

fn assert_close_or_nan(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    for (i, (&got, &want)) in actual.iter().zip(expected.iter()).enumerate() {
        if want.is_nan() {
            assert!(got.is_nan(), "{label}[{i}] got {got}, want NaN");
        } else if want.is_infinite() {
            assert_eq!(got, want, "{label}[{i}] got {got}, want {want}");
        } else {
            assert!(
                (got - want).abs() <= tol,
                "{label}[{i}] got {got}, want {want}, tol {tol}"
            );
        }
    }
}

fn assert_grad_bf16(leaf: &Tensor<half::bf16>, expected: &[f32], tol: f32, label: &str) {
    let grad = leaf
        .grad()
        .expect("grad slot")
        .unwrap_or_else(|| panic!("{label}: no gradient reached CUDA bf16 leaf"));
    assert_eq!(grad.device(), Device::Cuda(0), "{label}: gradient device");
    assert_close_or_nan(&host_bf16(&grad), expected, tol, label);
}

fn assert_grad_f16(leaf: &Tensor<half::f16>, expected: &[f32], tol: f32, label: &str) {
    let grad = leaf
        .grad()
        .expect("grad slot")
        .unwrap_or_else(|| panic!("{label}: no gradient reached CUDA f16 leaf"));
    assert_eq!(grad.device(), Device::Cuda(0), "{label}: gradient device");
    assert_close_or_nan(&host_f16(&grad), expected, tol, label);
}

#[test]
fn bf16_cuda_sqrt_forward_special_values() {
    ensure_cuda_backend();

    let x = bf16_cuda(&[0.0, 1.0, 2.0, 4.0, f32::INFINITY, f32::NAN], &[6], false);
    assert_close_or_nan(
        &host_bf16(&x),
        &[0.0, 1.0, 2.0, 4.0, f32::INFINITY, f32::NAN],
        0.0,
        "bf16 CUDA input transport",
    );
    let y = sqrt(&x).expect("sqrt bf16 CUDA");

    assert_close_or_nan(
        &host_bf16(&y),
        &[0.0, 1.0, 1.4140625, 2.0, f32::INFINITY, f32::NAN],
        0.0,
        "sqrt bf16 CUDA special values",
    );
}

#[test]
fn bf16_cuda_sqrt_backward_stays_on_device() {
    ensure_cuda_backend();

    let x = bf16_cuda(&[1.0, 4.0, 9.0], &[3], true);
    sum(&sqrt(&x).expect("tracked sqrt bf16 CUDA"))
        .expect("sum")
        .backward()
        .expect("sqrt backward");

    assert_grad_bf16(&x, &[0.5, 0.25, 0.166_992_19], 0.0, "sqrt bf16 grad");
}

#[test]
fn bf16_cuda_rsqrt_forward_backward_matches_torch() {
    ensure_cuda_backend();

    let x = bf16_cuda(&[1.0, 4.0, 9.0, f32::INFINITY, f32::NAN], &[5], false);
    let y = rsqrt(&x).expect("rsqrt bf16 CUDA");
    assert_close_or_nan(
        &host_bf16(&y),
        &[1.0, 0.5, 0.333_984_38, 0.0, f32::NAN],
        0.0,
        "rsqrt bf16 CUDA forward",
    );

    let x = bf16_cuda(&[1.0, 4.0, 9.0], &[3], true);
    sum(&rsqrt(&x).expect("tracked rsqrt bf16 CUDA"))
        .expect("sum")
        .backward()
        .expect("rsqrt backward");
    assert_grad_bf16(&x, &[-0.5, -0.0625, -0.018_554_688], 0.0, "rsqrt bf16 grad");
}

#[test]
fn bf16_cuda_reciprocal_forward_backward_uses_device_fill() {
    ensure_cuda_backend();

    let x = bf16_cuda(&[1.0, 4.0, 8.0, f32::INFINITY, f32::NAN], &[5], false);
    let y = reciprocal(&x).expect("reciprocal bf16 CUDA");
    assert_close_or_nan(
        &host_bf16(&y),
        &[1.0, 0.25, 0.125, 0.0, f32::NAN],
        0.0,
        "reciprocal bf16 CUDA forward",
    );

    let x = bf16_cuda(&[1.0, 4.0, 8.0], &[3], true);
    sum(&reciprocal(&x).expect("tracked reciprocal bf16 CUDA"))
        .expect("sum")
        .backward()
        .expect("reciprocal backward");
    assert_grad_bf16(&x, &[-1.0, -0.0625, -0.015625], 0.0, "reciprocal bf16 grad");
}

#[test]
fn bf16_cuda_sqrt_empty_returns_empty_on_device() {
    ensure_cuda_backend();

    let x = bf16_cuda(&[], &[0], false);
    let y = sqrt(&x).expect("empty sqrt bf16 CUDA");

    assert_eq!(y.shape(), &[0]);
    assert_eq!(host_bf16(&y), Vec::<f32>::new());
}

#[test]
fn bf16_cuda_sqrt_accepts_noncontiguous_view() {
    ensure_cuda_backend();

    let base = bf16_cuda(&[1.0, 4.0, 9.0, 16.0, 25.0, 36.0], &[2, 3], false);
    let view = base.transpose(0, 1).expect("transpose view");
    let y = sqrt(&view).expect("sqrt bf16 CUDA non-contiguous view");

    assert_eq!(y.shape(), &[3, 2]);
    assert_close_or_nan(
        &host_bf16(&y),
        &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
        0.0,
        "sqrt bf16 CUDA non-contiguous view",
    );
}

#[test]
fn f16_cuda_sqrt_div_specials_and_backward_match_torch() {
    ensure_cuda_backend();

    let x = f16_cuda(&[0.0, 1.0, 2.0, 4.0, f32::INFINITY, f32::NAN], &[6], false);
    assert_close_or_nan(
        &host_f16(&sqrt(&x).expect("sqrt f16 CUDA")),
        &[0.0, 1.0, 1.4140625, 2.0, f32::INFINITY, f32::NAN],
        0.0,
        "sqrt f16 CUDA special values",
    );

    let x = f16_cuda(&[1.0, 4.0, 9.0, f32::INFINITY, f32::NAN], &[5], false);
    assert_close_or_nan(
        &host_f16(&rsqrt(&x).expect("rsqrt f16 CUDA")),
        &[1.0, 0.5, 0.33325195, 0.0, f32::NAN],
        0.0,
        "rsqrt f16 CUDA special values",
    );

    let x = f16_cuda(&[1.0, 4.0, 8.0, f32::INFINITY, f32::NAN], &[5], false);
    assert_close_or_nan(
        &host_f16(&reciprocal(&x).expect("reciprocal f16 CUDA")),
        &[1.0, 0.25, 0.125, 0.0, f32::NAN],
        0.0,
        "reciprocal f16 CUDA special values",
    );

    let x = f16_cuda(&[1.0, 4.0, 9.0], &[3], true);
    sum(&sqrt(&x).expect("tracked sqrt f16 CUDA"))
        .expect("sum")
        .backward()
        .expect("sqrt f16 backward");
    assert_grad_f16(&x, &[0.5, 0.25, 0.16662598], 0.0, "sqrt f16 grad");

    let x = f16_cuda(&[1.0, 4.0, 9.0], &[3], true);
    sum(&rsqrt(&x).expect("tracked rsqrt f16 CUDA"))
        .expect("sum")
        .backward()
        .expect("rsqrt f16 backward");
    assert_grad_f16(&x, &[-0.5, -0.0625, -0.018508911], 0.0, "rsqrt f16 grad");
}

#[test]
fn f16_bf16_cuda_broadcast_div_propagates_nan() {
    ensure_cuda_backend();

    let a = bf16_cuda(&[1.0, f32::NAN], &[2, 1], false);
    let b = bf16_cuda(&[1.0, 2.0, 4.0], &[1, 3], false);
    let y = div(&a, &b).expect("broadcast div bf16 CUDA");
    assert_eq!(y.shape(), &[2, 3]);
    assert_close_or_nan(
        &host_bf16(&y),
        &[1.0, 0.5, 0.25, f32::NAN, f32::NAN, f32::NAN],
        0.0,
        "broadcast div bf16 CUDA NaN",
    );

    let a = f16_cuda(&[1.0, f32::NAN], &[2, 1], false);
    let b = f16_cuda(&[1.0, 2.0, 4.0], &[1, 3], false);
    let y = div(&a, &b).expect("broadcast div f16 CUDA");
    assert_eq!(y.shape(), &[2, 3]);
    assert_close_or_nan(
        &host_f16(&y),
        &[1.0, 0.5, 0.25, f32::NAN, f32::NAN, f32::NAN],
        0.0,
        "broadcast div f16 CUDA NaN",
    );
}
