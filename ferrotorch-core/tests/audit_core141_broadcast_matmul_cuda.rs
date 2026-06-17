//! CORE-141 live CUDA probe: raw `ops::linalg::matmul` must dispatch
//! broadcast matmul to resident CUDA kernels, not only the differentiable
//! wrapper.

#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::creation::from_vec;
use ferrotorch_core::ops::linalg::matmul;
use ferrotorch_core::{Device, Tensor};

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for CORE-141 live CUDA probe");
    });
}

fn values_f32(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i as u32).wrapping_mul(1_664_525).wrapping_add(seed);
            ((x as f32) / (u32::MAX as f32)) * 2.0 - 1.0
        })
        .collect()
}

fn values_f64(n: usize, seed: u32) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let x = (i as u32).wrapping_mul(1_664_525).wrapping_add(seed);
            ((x as f64) / (u32::MAX as f64)) * 2.0 - 1.0
        })
        .collect()
}

fn read_back_f32(t: &Tensor<f32>) -> Vec<f32> {
    t.cpu().expect("D2H f32").data_vec().expect("read f32 data")
}

fn read_back_f64(t: &Tensor<f64>) -> Vec<f64> {
    t.cpu().expect("D2H f64").data_vec().expect("read f64 data")
}

fn assert_close_f32(got: &[f32], want: &[f32], label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: numel mismatch");
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let diff = (g - w).abs();
        assert!(
            diff <= 1.0e-4,
            "{label} elem {i}: got={g} want={w} diff={diff}"
        );
    }
}

fn assert_close_f64(got: &[f64], want: &[f64], label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: numel mismatch");
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let diff = (g - w).abs();
        assert!(
            diff <= 1.0e-10,
            "{label} elem {i}: got={g} want={w} diff={diff}"
        );
    }
}

#[test]
fn raw_ops_matmul_cuda_broadcast_f32_matches_cpu() {
    ensure_cuda_backend();

    let a_vals = values_f32(2 * 3 * 4, 0x141A);
    let b_vals = values_f32(2 * 5 * 4 * 6, 0x141B);
    let a_cpu = from_vec::<f32>(a_vals, &[2, 1, 3, 4]).expect("cpu A");
    let b_cpu = from_vec::<f32>(b_vals, &[2, 5, 4, 6]).expect("cpu B");
    let want = matmul(&a_cpu, &b_cpu).expect("CPU broadcast matmul reference");

    let a_gpu = a_cpu.to(Device::Cuda(0)).expect("upload A");
    let b_gpu = b_cpu.to(Device::Cuda(0)).expect("upload B");
    let got = matmul(&a_gpu, &b_gpu).expect("raw CUDA broadcast matmul");

    assert!(
        got.is_cuda(),
        "raw broadcast matmul result must stay on CUDA"
    );
    assert_eq!(got.shape(), &[2, 5, 3, 6]);
    assert_close_f32(
        &read_back_f32(&got),
        &want.data_vec().expect("CPU reference data"),
        "f32 broadcast",
    );
}

#[test]
fn raw_ops_matmul_cuda_1d_lhs_promotion_f64_matches_cpu() {
    ensure_cuda_backend();

    let a_vals = values_f64(4, 0x141C);
    let b_vals = values_f64(2 * 3 * 4 * 5, 0x141D);
    let a_cpu = from_vec::<f64>(a_vals, &[4]).expect("cpu vector");
    let b_cpu = from_vec::<f64>(b_vals, &[2, 3, 4, 5]).expect("cpu RHS");
    let want = matmul(&a_cpu, &b_cpu).expect("CPU 1D @ batched reference");

    let a_gpu = a_cpu.to(Device::Cuda(0)).expect("upload vector");
    let b_gpu = b_cpu.to(Device::Cuda(0)).expect("upload RHS");
    let got = matmul(&a_gpu, &b_gpu).expect("raw CUDA 1D @ batched RHS");

    assert!(got.is_cuda(), "1D promotion result must stay on CUDA");
    assert_eq!(got.shape(), &[2, 3, 5]);
    assert_close_f64(
        &read_back_f64(&got),
        &want.data_vec().expect("CPU reference data"),
        "f64 1D promotion",
    );
}

#[test]
fn raw_ops_matmul_cuda_zero_batch_stays_cuda_and_empty() {
    ensure_cuda_backend();

    let a_cpu = from_vec::<f32>(Vec::new(), &[0, 3, 4]).expect("empty CPU A");
    let b_cpu = from_vec::<f32>(values_f32(4 * 6, 0x141E), &[4, 6]).expect("CPU B");

    let a_gpu = a_cpu.to(Device::Cuda(0)).expect("upload empty A");
    let b_gpu = b_cpu.to(Device::Cuda(0)).expect("upload B");
    let got = matmul(&a_gpu, &b_gpu).expect("raw CUDA zero-batch broadcast matmul");

    assert!(got.is_cuda(), "zero-batch result must stay on CUDA");
    assert_eq!(got.shape(), &[0, 3, 6]);
    assert!(read_back_f32(&got).is_empty());
}
