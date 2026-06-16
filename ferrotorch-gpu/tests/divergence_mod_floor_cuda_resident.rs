//! CUDA-resident parity probes for `remainder`, `fmod`, and `floor_divide`.
//!
//! These ops used to route CUDA tensors through `Tensor::data_vec()` host
//! staging in `ferrotorch-core::grad_fns::arithmetic`. PyTorch has native CUDA
//! kernels for f32/f64/f16/bf16:
//!
//! - `aten/src/ATen/native/cuda/BinaryRemainderKernel.cu`
//! - `aten/src/ATen/native/cuda/BinaryDivFloorKernel.cu`
//!
//! The tests below use torch 2.11.0+cu130 live-oracle values for signs,
//! NaN/Inf/zero edges, broadcasted shapes, and autograd quotient gradients.

#![cfg(feature = "cuda")]

use std::sync::Once;

use ferrotorch_core::grad_fns::{arithmetic, reduction};
use ferrotorch_core::{Device, Float, Tensor, TensorStorage};
use ferrotorch_gpu::init_cuda_backend;

fn ensure_cuda() -> bool {
    static INIT: Once = Once::new();
    static mut OK: bool = false;
    if ferrotorch_gpu::device::GpuDevice::new(0).is_err() {
        return false;
    }
    INIT.call_once(|| {
        init_cuda_backend().expect("CUDA backend init");
        unsafe { OK = true }
    });
    unsafe { OK }
}

fn cpu_tensor<T: Float>(data: &[T], shape: &[usize], requires_grad: bool) -> Tensor<T> {
    Tensor::from_storage(
        TensorStorage::cpu(data.to_vec()),
        shape.to_vec(),
        requires_grad,
    )
    .expect("cpu tensor")
}

fn cuda_tensor<T: Float>(data: &[T], shape: &[usize], requires_grad: bool) -> Tensor<T> {
    cpu_tensor(data, shape, false)
        .to(Device::Cuda(0))
        .expect("to cuda")
        .requires_grad_(requires_grad)
}

fn cpu_values_f32<T: Float>(t: &Tensor<T>) -> Vec<f32> {
    let cpu = t.cpu().expect("to cpu");
    cpu.data()
        .expect("cpu data")
        .iter()
        .map(|v| v.to_f32().expect("value representable as f32"))
        .collect()
}

fn grad_values_f32<T: Float>(t: &Tensor<T>) -> (Device, Vec<f32>) {
    let grad = t.grad().expect("grad access").expect("grad present");
    let device = grad.device();
    (device, cpu_values_f32(&grad))
}

fn assert_pattern(got: &[f32], want: &[f32], name: &str) {
    assert_eq!(got.len(), want.len(), "{name}: length mismatch");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        if w.is_nan() {
            assert!(g.is_nan(), "{name}[{i}] expected NaN, got {g}");
        } else if w.is_infinite() {
            assert_eq!(g, w, "{name}[{i}] expected {w}, got {g}");
        } else {
            assert!((g - w).abs() <= 1e-3, "{name}[{i}] expected {w}, got {g}");
        }
    }
}

fn run_forward_edge_case<T: Float>(data: &[T], divs: &[T], dtype_name: &str) {
    let a = cuda_tensor(data, &[data.len()], false);
    let b = cuda_tensor(divs, &[divs.len()], false);

    let r = arithmetic::remainder(&a, &b).expect("cuda remainder");
    let f = arithmetic::fmod(&a, &b).expect("cuda fmod");
    let q = arithmetic::floor_divide(&a, &b).expect("cuda floor_divide");

    assert_eq!(r.device(), Device::Cuda(0), "{dtype_name} remainder device");
    assert_eq!(f.device(), Device::Cuda(0), "{dtype_name} fmod device");
    assert_eq!(
        q.device(),
        Device::Cuda(0),
        "{dtype_name} floor_divide device"
    );

    assert_pattern(
        &cpu_values_f32(&r),
        &[
            1.0,
            2.0,
            -1.0,
            -2.0,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::INFINITY,
            5.0,
            -5.0,
            f32::NEG_INFINITY,
        ],
        &format!("{dtype_name} remainder"),
    );
    assert_pattern(
        &cpu_values_f32(&f),
        &[
            1.0,
            -1.0,
            2.0,
            -2.0,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            -5.0,
            5.0,
            -5.0,
            5.0,
        ],
        &format!("{dtype_name} fmod"),
    );
    assert_pattern(
        &cpu_values_f32(&q),
        &[
            2.0,
            -3.0,
            -2.0,
            1.0,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            -1.0,
            0.0,
            0.0,
            -1.0,
        ],
        &format!("{dtype_name} floor_divide"),
    );
}

#[test]
fn cuda_mod_floor_forward_edges_f32_f64() {
    if !ensure_cuda() {
        eprintln!("no CUDA device; skipping");
        return;
    }

    let data32 = [
        7.0,
        -7.0,
        5.0,
        -5.0,
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        -5.0,
        5.0,
        -5.0,
        5.0,
    ];
    let divs32 = [
        3.0,
        3.0,
        -3.0,
        -3.0,
        0.0,
        -0.0,
        3.0,
        3.0,
        3.0,
        f32::INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
    ];
    run_forward_edge_case(&data32, &divs32, "f32");

    let data64: Vec<f64> = data32.iter().map(|&x| x as f64).collect();
    let divs64: Vec<f64> = divs32.iter().map(|&x| x as f64).collect();
    run_forward_edge_case(&data64, &divs64, "f64");
}

#[test]
fn cuda_mod_floor_forward_edges_f16_bf16() {
    if !ensure_cuda() {
        eprintln!("no CUDA device; skipping");
        return;
    }

    let data = [
        7.0_f32,
        -7.0,
        5.0,
        -5.0,
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        -5.0,
        5.0,
        -5.0,
        5.0,
    ];
    let divs = [
        3.0_f32,
        3.0,
        -3.0,
        -3.0,
        0.0,
        -0.0,
        3.0,
        3.0,
        3.0,
        f32::INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
    ];

    let data_f16: Vec<half::f16> = data.iter().copied().map(half::f16::from_f32).collect();
    let divs_f16: Vec<half::f16> = divs.iter().copied().map(half::f16::from_f32).collect();
    run_forward_edge_case(&data_f16, &divs_f16, "f16");

    let data_bf16: Vec<half::bf16> = data.iter().copied().map(half::bf16::from_f32).collect();
    let divs_bf16: Vec<half::bf16> = divs.iter().copied().map(half::bf16::from_f32).collect();
    run_forward_edge_case(&data_bf16, &divs_bf16, "bf16");
}

#[test]
fn cuda_mod_floor_broadcasts_and_stays_on_device() {
    if !ensure_cuda() {
        eprintln!("no CUDA device; skipping");
        return;
    }

    let a = cuda_tensor(&[7.0_f32, -7.0, 5.0, -5.0, -7.0, 7.0], &[2, 3], false);
    let b = cuda_tensor(&[3.0_f32, 3.0, -3.0], &[3], false);

    let r = arithmetic::remainder(&a, &b).expect("broadcast remainder");
    let f = arithmetic::fmod(&a, &b).expect("broadcast fmod");
    let q = arithmetic::floor_divide(&a, &b).expect("broadcast floor_divide");

    assert_eq!(r.shape(), &[2, 3]);
    assert_eq!(f.shape(), &[2, 3]);
    assert_eq!(q.shape(), &[2, 3]);
    assert_eq!(r.device(), Device::Cuda(0));
    assert_eq!(f.device(), Device::Cuda(0));
    assert_eq!(q.device(), Device::Cuda(0));

    assert_pattern(
        &cpu_values_f32(&r),
        &[1.0, 2.0, -1.0, 1.0, 2.0, -2.0],
        "broadcast remainder",
    );
    assert_pattern(
        &cpu_values_f32(&f),
        &[1.0, -1.0, 2.0, -2.0, -1.0, 1.0],
        "broadcast fmod",
    );
    assert_pattern(
        &cpu_values_f32(&q),
        &[2.0, -3.0, -2.0, -2.0, -3.0, -3.0],
        "broadcast floor_divide",
    );
}

fn run_backward_case<T: Float>(data: &[T], divs: &[T], rem_bgrad: &[f32], fmod_bgrad: &[f32]) {
    let a = cuda_tensor(data, &[data.len()], true);
    let b = cuda_tensor(divs, &[divs.len()], true);
    let y = arithmetic::remainder(&a, &b).expect("remainder forward");
    let loss = reduction::sum(&y).expect("sum remainder");
    loss.backward().expect("remainder backward");
    let (a_dev, a_grad) = grad_values_f32(&a);
    let (b_dev, b_grad) = grad_values_f32(&b);
    assert_eq!(a_dev, Device::Cuda(0), "remainder a.grad device");
    assert_eq!(b_dev, Device::Cuda(0), "remainder b.grad device");
    assert_pattern(&a_grad, &[1.0, 1.0], "remainder a.grad");
    assert_pattern(&b_grad, rem_bgrad, "remainder b.grad");

    let a = cuda_tensor(data, &[data.len()], true);
    let b = cuda_tensor(divs, &[divs.len()], true);
    let y = arithmetic::fmod(&a, &b).expect("fmod forward");
    let loss = reduction::sum(&y).expect("sum fmod");
    loss.backward().expect("fmod backward");
    let (a_dev, a_grad) = grad_values_f32(&a);
    let (b_dev, b_grad) = grad_values_f32(&b);
    assert_eq!(a_dev, Device::Cuda(0), "fmod a.grad device");
    assert_eq!(b_dev, Device::Cuda(0), "fmod b.grad device");
    assert_pattern(&a_grad, &[1.0, 1.0], "fmod a.grad");
    assert_pattern(&b_grad, fmod_bgrad, "fmod b.grad");
}

#[test]
fn cuda_remainder_fmod_backward_uses_device_rounding_division() {
    if !ensure_cuda() {
        eprintln!("no CUDA device; skipping");
        return;
    }

    run_backward_case(&[-7.0_f32, 5.0], &[3.0_f32, -3.0], &[3.0, 2.0], &[2.0, 1.0]);
    run_backward_case(&[-7.0_f64, 5.0], &[3.0_f64, -3.0], &[3.0, 2.0], &[2.0, 1.0]);

    let a_f16: Vec<half::f16> = [-7.0_f32, 5.0]
        .into_iter()
        .map(half::f16::from_f32)
        .collect();
    let b_f16: Vec<half::f16> = [3.0_f32, -3.0]
        .into_iter()
        .map(half::f16::from_f32)
        .collect();
    run_backward_case(&a_f16, &b_f16, &[3.0, 2.0], &[2.0, 1.0]);

    let a_bf16: Vec<half::bf16> = [-7.0_f32, 5.0]
        .into_iter()
        .map(half::bf16::from_f32)
        .collect();
    let b_bf16: Vec<half::bf16> = [3.0_f32, -3.0]
        .into_iter()
        .map(half::bf16::from_f32)
        .collect();
    run_backward_case(&a_bf16, &b_bf16, &[3.0, 2.0], &[2.0, 1.0]);
}

#[test]
fn cuda_floor_divide_backward_matches_pytorch_not_implemented() {
    if !ensure_cuda() {
        eprintln!("no CUDA device; skipping");
        return;
    }

    let a = cuda_tensor(&[7.0_f32, -7.0], &[2], true);
    let b = cuda_tensor(&[3.0_f32, 3.0], &[2], false);
    let q = arithmetic::floor_divide(&a, &b).expect("floor_divide forward");
    assert_eq!(q.device(), Device::Cuda(0));
    let loss = reduction::sum(&q).expect("sum floor_divide");
    let err = loss
        .backward()
        .expect_err("floor_divide backward must error");
    assert!(
        err.to_string()
            .contains("derivative for floor_divide is not implemented"),
        "unexpected floor_divide backward error: {err}"
    );
}
