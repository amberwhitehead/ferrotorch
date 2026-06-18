#![cfg(feature = "gpu")]

//! CORE-150: CUDA `eigh` must honor ferrotorch's stable eigenvector sign
//! contract, not return raw cuSOLVER gauge choices.
//!
//! PyTorch permits arbitrary signs for real Hermitian eigenvectors; ferrotorch
//! documents a stronger reproducibility contract by making the largest-absolute
//! entry in each eigenvector column non-negative. CPU already did that. These
//! tests require the CUDA path to do the same while remaining CUDA-resident.

use std::sync::Once;

use ferrotorch_core::Device;
use ferrotorch_core::creation::from_vec;
use ferrotorch_core::linalg::eigh;

static CUDA_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    CUDA_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend().expect("CUDA backend must initialize for CORE-150")
    });
}

fn assert_close_f32(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch actual={} expected={}",
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!((a - e).abs() <= tol, "{label}[{i}] actual={a} expected={e}");
    }
}

fn assert_close_f64(actual: &[f64], expected: &[f64], tol: f64, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch actual={} expected={}",
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!((a - e).abs() <= tol, "{label}[{i}] actual={a} expected={e}");
    }
}

fn assert_canonical_f32(q: &[f32], n: usize, label: &str) {
    assert_eq!(q.len(), n * n, "{label}: q length must be n*n");
    for col in 0..n {
        let mut best_row = 0usize;
        let mut best_abs = 0.0f32;
        for row in 0..n {
            let v = q[row * n + col].abs();
            if v > best_abs {
                best_abs = v;
                best_row = row;
            }
        }
        let pivot = q[best_row * n + col];
        assert!(
            pivot >= 0.0,
            "{label}: column {col} pivot row {best_row} is negative: {pivot}"
        );
    }
}

fn assert_canonical_f64(q: &[f64], n: usize, label: &str) {
    assert_eq!(q.len(), n * n, "{label}: q length must be n*n");
    for col in 0..n {
        let mut best_row = 0usize;
        let mut best_abs = 0.0f64;
        for row in 0..n {
            let v = q[row * n + col].abs();
            if v > best_abs {
                best_abs = v;
                best_row = row;
            }
        }
        let pivot = q[best_row * n + col];
        assert!(
            pivot >= 0.0,
            "{label}: column {col} pivot row {best_row} is negative: {pivot}"
        );
    }
}

#[test]
fn eigh_cuda_f32_uses_stable_sign_contract_and_stays_resident() {
    ensure_cuda_backend();
    let a_cpu = from_vec(vec![2.0f32, 1.0, 1.0, 2.0], &[2, 2]).expect("cpu f32");
    let a_gpu = a_cpu.clone().to(Device::Cuda(0)).expect("upload f32");

    let (w_cpu, q_cpu) = eigh(&a_cpu).expect("cpu eigh f32");
    let (w_gpu, q_gpu) = eigh(&a_gpu).expect("cuda eigh f32");

    assert_eq!(w_gpu.device(), Device::Cuda(0));
    assert_eq!(q_gpu.device(), Device::Cuda(0));
    assert_eq!(w_gpu.shape(), w_cpu.shape());
    assert_eq!(q_gpu.shape(), q_cpu.shape());

    let q_host = q_gpu
        .to(Device::Cpu)
        .expect("download q f32")
        .data_vec()
        .expect("q f32 values");
    let w_host = w_gpu
        .to(Device::Cpu)
        .expect("download w f32")
        .data_vec()
        .expect("w f32 values");
    let q_expected = q_cpu.data_vec().expect("cpu q f32");
    let w_expected = w_cpu.data_vec().expect("cpu w f32");

    assert_canonical_f32(&q_host, 2, "cuda f32 q");
    assert_close_f32(&w_host, &w_expected, 1e-5, "cuda f32 eigenvalues");
    assert_close_f32(&q_host, &q_expected, 1e-5, "cuda f32 eigenvectors");
}

#[test]
fn eigh_cuda_f64_uses_stable_sign_contract_and_stays_resident() {
    ensure_cuda_backend();
    let a_cpu = from_vec(vec![4.0f64, 1.0, 1.0, 3.0], &[2, 2]).expect("cpu f64");
    let a_gpu = a_cpu.clone().to(Device::Cuda(0)).expect("upload f64");

    let (w_cpu, q_cpu) = eigh(&a_cpu).expect("cpu eigh f64");
    let (w_gpu, q_gpu) = eigh(&a_gpu).expect("cuda eigh f64");

    assert_eq!(w_gpu.device(), Device::Cuda(0));
    assert_eq!(q_gpu.device(), Device::Cuda(0));
    assert_eq!(w_gpu.shape(), w_cpu.shape());
    assert_eq!(q_gpu.shape(), q_cpu.shape());

    let q_host = q_gpu
        .to(Device::Cpu)
        .expect("download q f64")
        .data_vec()
        .expect("q f64 values");
    let w_host = w_gpu
        .to(Device::Cpu)
        .expect("download w f64")
        .data_vec()
        .expect("w f64 values");
    let q_expected = q_cpu.data_vec().expect("cpu q f64");
    let w_expected = w_cpu.data_vec().expect("cpu w f64");

    assert_canonical_f64(&q_host, 2, "cuda f64 q");
    assert_close_f64(&w_host, &w_expected, 1e-10, "cuda f64 eigenvalues");
    assert_close_f64(&q_host, &q_expected, 1e-10, "cuda f64 eigenvectors");
}
