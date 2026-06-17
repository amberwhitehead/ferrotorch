//! Regression coverage for CORE #1991.
//!
//! `unary_map` is a CPU closure helper and must not pretend to execute on
//! CUDA. Named sin/cos, however, must route through resident CUDA kernels for
//! PyTorch's floating dtypes and keep outputs/grads device-resident.

#![cfg(feature = "cuda")]

use std::sync::OnceLock;

use ferrotorch_core::grad_fns::transcendental::{cos, sin};
use ferrotorch_core::ops::elementwise::{fast_cos, fast_sin, unary_map};
use ferrotorch_core::{Device, FerrotorchError, Tensor, TensorStorage, backward};
use ferrotorch_gpu::init_cuda_backend;
use half::{bf16, f16};

fn ensure_cuda() -> bool {
    static OK: OnceLock<bool> = OnceLock::new();
    *OK.get_or_init(|| {
        if ferrotorch_gpu::device::GpuDevice::new(0).is_err() {
            return false;
        }
        init_cuda_backend().is_ok()
    })
}

fn cpu_f32(data: &[f32]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false).unwrap()
}

fn cpu_f64(data: &[f64]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false).unwrap()
}

fn cpu_f16(data: &[f16]) -> Tensor<f16> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false).unwrap()
}

fn cpu_bf16(data: &[bf16]) -> Tensor<bf16> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false).unwrap()
}

fn host_f32(t: &Tensor<f32>) -> Vec<f32> {
    t.to(Device::Cpu).expect("to cpu").data().unwrap().to_vec()
}

fn host_f64(t: &Tensor<f64>) -> Vec<f64> {
    t.to(Device::Cpu).expect("to cpu").data().unwrap().to_vec()
}

fn host_f16(t: &Tensor<f16>) -> Vec<f16> {
    t.to(Device::Cpu).expect("to cpu").data().unwrap().to_vec()
}

fn host_bf16(t: &Tensor<bf16>) -> Vec<bf16> {
    t.to(Device::Cpu).expect("to cpu").data().unwrap().to_vec()
}

fn assert_close_f32(actual: f32, expected: f32, tol: f32, label: &str, i: usize) {
    if expected.is_nan() {
        assert!(actual.is_nan(), "{label}[{i}] expected NaN got {actual}");
    } else {
        let err = (actual - expected).abs();
        assert!(
            err <= tol * (1.0 + expected.abs()),
            "{label}[{i}] actual={actual} expected={expected} err={err}"
        );
    }
}

fn assert_close_f64(actual: f64, expected: f64, tol: f64, label: &str, i: usize) {
    if expected.is_nan() {
        assert!(actual.is_nan(), "{label}[{i}] expected NaN got {actual}");
    } else {
        let err = (actual - expected).abs();
        assert!(
            err <= tol * (1.0 + expected.abs()),
            "{label}[{i}] actual={actual} expected={expected} err={err}"
        );
    }
}

#[test]
fn unary_map_cuda_is_not_a_host_roundtrip() {
    if !ensure_cuda() {
        return;
    }
    let gpu = cpu_f32(&[-1.0, 0.0, 2.0]).to(Device::Cuda(0)).unwrap();
    let err = unary_map(&gpu, |x| x + 1.0).expect_err("generic CUDA closure must fail");
    assert!(
        matches!(err, FerrotorchError::NotImplementedOnCuda { op } if op == "unary_map"),
        "unexpected error: {err:?}"
    );
}

#[test]
fn sin_cos_cuda_f32_f64_forward_and_backward_stay_resident() {
    if !ensure_cuda() {
        return;
    }

    let xs32 = [
        -std::f32::consts::PI,
        -0.5,
        0.0,
        0.5,
        std::f32::consts::FRAC_PI_2,
        10.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    let gpu32 = cpu_f32(&xs32).to(Device::Cuda(0)).unwrap();
    let sin32 = sin(&gpu32).expect("sin f32 cuda");
    let cos32 = cos(&gpu32).expect("cos f32 cuda");
    assert!(sin32.is_cuda(), "sin f32 output must remain CUDA-resident");
    assert!(cos32.is_cuda(), "cos f32 output must remain CUDA-resident");
    let sin32_host = host_f32(&sin32);
    let cos32_host = host_f32(&cos32);
    for (i, (&x, (&s, &c))) in xs32
        .iter()
        .zip(sin32_host.iter().zip(cos32_host.iter()))
        .enumerate()
    {
        assert_close_f32(s, x.sin(), 2.0e-6, "sin32", i);
        assert_close_f32(c, x.cos(), 2.0e-6, "cos32", i);
    }

    let xs64 = [
        -std::f64::consts::PI,
        -0.5,
        0.0,
        0.5,
        std::f64::consts::FRAC_PI_2,
        10.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];
    let gpu64 = cpu_f64(&xs64).to(Device::Cuda(0)).unwrap();
    let sin64 = sin(&gpu64).expect("sin f64 cuda");
    let cos64 = cos(&gpu64).expect("cos f64 cuda");
    assert!(sin64.is_cuda(), "sin f64 output must remain CUDA-resident");
    assert!(cos64.is_cuda(), "cos f64 output must remain CUDA-resident");
    let sin64_host = host_f64(&sin64);
    let cos64_host = host_f64(&cos64);
    for (i, (&x, (&s, &c))) in xs64
        .iter()
        .zip(sin64_host.iter().zip(cos64_host.iter()))
        .enumerate()
    {
        assert_close_f64(s, x.sin(), 1.0e-12, "sin64", i);
        assert_close_f64(c, x.cos(), 1.0e-12, "cos64", i);
    }

    let leaf32 = cpu_f32(&[-0.7, 0.0, 1.25])
        .to(Device::Cuda(0))
        .unwrap()
        .requires_grad_(true);
    let loss32 = sin(&leaf32).unwrap().sum_all().unwrap();
    backward(&loss32).expect("sin f32 backward");
    let grad32 = leaf32.grad().unwrap().expect("grad32");
    assert!(grad32.is_cuda(), "sin f32 grad must remain CUDA-resident");
    let grad32_host = host_f32(&grad32);
    for (i, (&x, &g)) in [-0.7_f32, 0.0, 1.25]
        .iter()
        .zip(grad32_host.iter())
        .enumerate()
    {
        assert_close_f32(g, x.cos(), 2.0e-6, "sin32 grad", i);
    }

    let leaf64 = cpu_f64(&[-0.7, 0.0, 1.25])
        .to(Device::Cuda(0))
        .unwrap()
        .requires_grad_(true);
    let loss64 = cos(&leaf64).unwrap().sum_all().unwrap();
    backward(&loss64).expect("cos f64 backward");
    let grad64 = leaf64.grad().unwrap().expect("grad64");
    assert!(grad64.is_cuda(), "cos f64 grad must remain CUDA-resident");
    let grad64_host = host_f64(&grad64);
    for (i, (&x, &g)) in [-0.7_f64, 0.0, 1.25]
        .iter()
        .zip(grad64_host.iter())
        .enumerate()
    {
        assert_close_f64(g, -x.sin(), 1.0e-12, "cos64 grad", i);
    }
}

#[test]
fn sin_cos_cuda_half_and_bfloat_forward_stay_resident() {
    if !ensure_cuda() {
        return;
    }

    let xs = [-1.0_f32, -0.5, 0.0, 0.5, 1.0];
    let xs_f16: Vec<f16> = xs.iter().copied().map(f16::from_f32).collect();
    let gpu_f16 = cpu_f16(&xs_f16).to(Device::Cuda(0)).unwrap();
    let sin_f16 = fast_sin(&gpu_f16).expect("sin f16 cuda");
    let cos_f16 = fast_cos(&gpu_f16).expect("cos f16 cuda");
    assert!(
        sin_f16.is_cuda(),
        "sin f16 output must remain CUDA-resident"
    );
    assert!(
        cos_f16.is_cuda(),
        "cos f16 output must remain CUDA-resident"
    );
    let sin_f16_host = host_f16(&sin_f16);
    let cos_f16_host = host_f16(&cos_f16);
    for (i, (&x, (&s, &c))) in xs
        .iter()
        .zip(sin_f16_host.iter().zip(cos_f16_host.iter()))
        .enumerate()
    {
        assert_close_f32(s.to_f32(), x.sin(), 2.0e-3, "sin f16", i);
        assert_close_f32(c.to_f32(), x.cos(), 2.0e-3, "cos f16", i);
    }

    let xs_bf16: Vec<bf16> = xs.iter().copied().map(bf16::from_f32).collect();
    let gpu_bf16 = cpu_bf16(&xs_bf16).to(Device::Cuda(0)).unwrap();
    let sin_bf16 = fast_sin(&gpu_bf16).expect("sin bf16 cuda");
    let cos_bf16 = fast_cos(&gpu_bf16).expect("cos bf16 cuda");
    assert!(
        sin_bf16.is_cuda(),
        "sin bf16 output must remain CUDA-resident"
    );
    assert!(
        cos_bf16.is_cuda(),
        "cos bf16 output must remain CUDA-resident"
    );
    let sin_bf16_host = host_bf16(&sin_bf16);
    let cos_bf16_host = host_bf16(&cos_bf16);
    for (i, (&x, (&s, &c))) in xs
        .iter()
        .zip(sin_bf16_host.iter().zip(cos_bf16_host.iter()))
        .enumerate()
    {
        assert_close_f32(s.to_f32(), x.sin(), 1.0e-2, "sin bf16", i);
        assert_close_f32(c.to_f32(), x.cos(), 1.0e-2, "cos bf16", i);
    }
}
