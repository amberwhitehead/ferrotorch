//! End-to-end GPU dispatch test for the orthogonal-polynomial special
//! functions (#1545 / #1533).
//!
//! These exercise the FULL production path: a `Tensor<f32>` / `Tensor<f64>`
//! is moved to CUDA via `Tensor::to(Device::Cuda(0))`, then the public
//! `ferrotorch_core::special::*` op is called. The op's GPU branch dispatches
//! through the registered `GpuBackend` (init via `init_cuda_backend`) into the
//! PTX kernels in `ferrotorch-gpu/src/special.rs`. The test asserts:
//!
//!   1. the result tensor stays `is_cuda()` (NO CPU round trip — R-CODE-4),
//!   2. the on-device values match the CPU path of the SAME op element-wise.
//!
//! The CPU reference is the ferrotorch CPU path itself (the same public op on
//! a CPU tensor), so the assertion pins GPU/CPU agreement exactly, not a
//! re-derivation.

#![cfg(feature = "cuda")]

use std::sync::Once;

use ferrotorch_core::{Device, Tensor, TensorStorage, special};
use ferrotorch_gpu::init_cuda_backend;

fn ensure_cuda() -> bool {
    static INIT: Once = Once::new();
    static mut OK: bool = false;
    // Probe device availability first; skip gracefully on CUDA-less hosts.
    if ferrotorch_gpu::device::GpuDevice::new(0).is_err() {
        return false;
    }
    INIT.call_once(|| {
        init_cuda_backend().expect("CUDA backend init");
        // SAFETY: write happens inside `call_once`, single-threaded by Once.
        unsafe { OK = true }
    });
    // SAFETY: read after `call_once` completed; value is set once and never
    // mutated again.
    unsafe { OK }
}

fn cpu_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

fn cpu_f64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

/// Assert the GPU result of `op` matches the CPU result of the SAME op, and
/// that the GPU result stayed on device.
fn check_f32(op: impl Fn(&Tensor<f32>) -> Tensor<f32>, data: &[f32], label: &str) {
    let cpu_in = cpu_f32(data, &[data.len()]);
    let gpu_in = cpu_in.to(Device::Cuda(0)).expect("to GPU");
    let gpu_out = op(&gpu_in);
    assert!(
        gpu_out.is_cuda(),
        "{label}: GPU output must stay on device (no CPU round trip)"
    );
    let cpu_out = op(&cpu_in);
    let gpu_back = gpu_out.to(Device::Cpu).expect("back to CPU");
    let g = gpu_back.data().unwrap();
    let c = cpu_out.data().unwrap();
    for i in 0..data.len() {
        assert!(
            (g[i] - c[i]).abs() <= 1e-3 * (1.0 + c[i].abs()),
            "{label}: idx {i} x={} gpu={} cpu={}",
            data[i],
            g[i],
            c[i]
        );
    }
}

const XS: [f32; 7] = [-1.4, -0.7, -0.25, 0.0, 0.3, 0.8, 1.3];

#[test]
fn chebyshev_t_gpu_dispatch_stays_on_device_and_matches_cpu() {
    if !ensure_cuda() {
        return;
    }
    for n in 0..=10usize {
        check_f32(
            |t| special::chebyshev_polynomial_t(t, n).unwrap(),
            &XS,
            "chebyshev_polynomial_t",
        );
    }
}

#[test]
fn chebyshev_uvw_gpu_dispatch_matches_cpu() {
    if !ensure_cuda() {
        return;
    }
    for n in 0..=9usize {
        check_f32(
            |t| special::chebyshev_polynomial_u(t, n).unwrap(),
            &XS,
            "cheb_u",
        );
        check_f32(
            |t| special::chebyshev_polynomial_v(t, n).unwrap(),
            &XS,
            "cheb_v",
        );
        check_f32(
            |t| special::chebyshev_polynomial_w(t, n).unwrap(),
            &XS,
            "cheb_w",
        );
    }
}

#[test]
fn shifted_chebyshev_gpu_dispatch_matches_cpu() {
    if !ensure_cuda() {
        return;
    }
    let xs = [0.0f32, 0.2, 0.5, 0.75, 1.0];
    for n in 0..=8usize {
        check_f32(
            |t| special::shifted_chebyshev_polynomial_t(t, n).unwrap(),
            &xs,
            "shifted_cheb_t",
        );
        check_f32(
            |t| special::shifted_chebyshev_polynomial_u(t, n).unwrap(),
            &xs,
            "shifted_cheb_u",
        );
        check_f32(
            |t| special::shifted_chebyshev_polynomial_v(t, n).unwrap(),
            &xs,
            "shifted_cheb_v",
        );
        check_f32(
            |t| special::shifted_chebyshev_polynomial_w(t, n).unwrap(),
            &xs,
            "shifted_cheb_w",
        );
    }
}

#[test]
fn hermite_gpu_dispatch_matches_cpu() {
    if !ensure_cuda() {
        return;
    }
    for n in 0..=7usize {
        check_f32(
            |t| special::hermite_polynomial_h(t, n).unwrap(),
            &XS,
            "hermite_h",
        );
        check_f32(
            |t| special::hermite_polynomial_he(t, n).unwrap(),
            &XS,
            "hermite_he",
        );
    }
}

#[test]
fn laguerre_legendre_gpu_dispatch_matches_cpu() {
    if !ensure_cuda() {
        return;
    }
    for n in 0..=9usize {
        check_f32(
            |t| special::laguerre_polynomial_l(t, n).unwrap(),
            &XS,
            "laguerre_l",
        );
        check_f32(
            |t| special::legendre_polynomial_p(t, n).unwrap(),
            &XS,
            "legendre_p",
        );
    }
}

#[test]
fn legendre_f64_gpu_dispatch_stays_on_device_and_matches_cpu() {
    if !ensure_cuda() {
        return;
    }
    let xs = [-0.9f64, -0.4, -0.1, 0.2, 0.6, 0.95];
    for n in 0..=12usize {
        let cpu_in = cpu_f64(&xs, &[xs.len()]);
        let gpu_in = cpu_in.to(Device::Cuda(0)).expect("to GPU");
        let gpu_out = special::legendre_polynomial_p(&gpu_in, n).unwrap();
        assert!(
            gpu_out.is_cuda(),
            "legendre_f64: result must stay on device"
        );
        let cpu_out = special::legendre_polynomial_p(&cpu_in, n).unwrap();
        let gpu_back = gpu_out.to(Device::Cpu).unwrap();
        let g = gpu_back.data().unwrap();
        let c = cpu_out.data().unwrap();
        for i in 0..xs.len() {
            assert!(
                (g[i] - c[i]).abs() <= 1e-12 * (1.0 + c[i].abs()),
                "legendre_f64 n={n} idx {i}: gpu={} cpu={}",
                g[i],
                c[i]
            );
        }
    }
}
