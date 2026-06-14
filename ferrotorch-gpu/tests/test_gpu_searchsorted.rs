//! End-to-end GPU `searchsorted` / `bucketize` integration tests on RTX 3090.
//! (#1545)
//!
//! Exercises the full production consumer path: `ferrotorch_core::searchsorted`
//! / `ferrotorch_core::bucketize` on CUDA-resident tensors lower the binary
//! search on-device via `GpuBackend::searchsorted_1d`
//! (`ferrotorch_gpu::gpu_searchsorted_f32` / `_f64`), then read back ONLY the
//! int64 insertion indices. Each test:
//!   - confirms the input tensors are device-resident (`is_cuda()`), so the
//!     value/boundary data never round-trips through the host
//!   - asserts the resulting `Vec<usize>` matches the CPU `partition_point`
//!     oracle on the same data, INCLUDING the right=true/false boundary/tie
//!     cases where a value lands exactly on a boundary.

#![cfg(feature = "cuda")]

use ferrotorch_core::ops::search::{bucketize, searchsorted};
use ferrotorch_core::{Device, Tensor, TensorStorage};
use ferrotorch_gpu::init_cuda_backend;
use half::{bf16, f16};

fn ensure_cuda() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        init_cuda_backend().expect("CUDA backend init");
    });
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

/// CPU oracle with the exact half-open comparisons the kernel implements.
fn cpu_ref(bounds: &[f64], vals: &[f64], right: bool) -> Vec<usize> {
    vals.iter()
        .map(|&v| {
            if right {
                bounds.partition_point(|&b| b <= v)
            } else {
                bounds.partition_point(|&b| b < v)
            }
        })
        .collect()
}

#[test]
fn searchsorted_f32_gpu_matches_cpu_both_sides() {
    ensure_cuda();
    let bounds_h = [1.0f32, 3.0, 5.0, 7.0];
    let vals_h = [0.0f32, 2.0, 3.0, 6.0, 8.0];
    let bounds = cpu_f32(&bounds_h).to(Device::Cuda(0)).unwrap();
    let vals = cpu_f32(&vals_h).to(Device::Cuda(0)).unwrap();
    // The search runs on-device: inputs are CUDA-resident.
    assert!(bounds.is_cuda());
    assert!(vals.is_cuda());

    let bounds64: Vec<f64> = bounds_h.iter().map(|&x| x as f64).collect();
    let vals64: Vec<f64> = vals_h.iter().map(|&x| x as f64).collect();

    let right = searchsorted(&bounds, &vals, true).unwrap();
    assert_eq!(right, cpu_ref(&bounds64, &vals64, true));
    assert_eq!(right, vec![0, 1, 2, 3, 4]);

    let left = searchsorted(&bounds, &vals, false).unwrap();
    assert_eq!(left, cpu_ref(&bounds64, &vals64, false));
}

#[test]
fn searchsorted_f32_gpu_boundary_tie_left_vs_right() {
    // The bug-prone case: every value lands exactly ON a boundary.
    ensure_cuda();
    let bounds_h = [1.0f32, 3.0, 5.0, 7.0];
    let vals_h = [1.0f32, 3.0, 5.0, 7.0];
    let bounds = cpu_f32(&bounds_h).to(Device::Cuda(0)).unwrap();
    let vals = cpu_f32(&vals_h).to(Device::Cuda(0)).unwrap();
    assert!(bounds.is_cuda() && vals.is_cuda());

    // left  (side="left"): value-on-boundary -> that boundary's index.
    let left = searchsorted(&bounds, &vals, false).unwrap();
    assert_eq!(left, vec![0, 1, 2, 3]);
    // right (side="right"): value-on-boundary -> one past it.
    let right = searchsorted(&bounds, &vals, true).unwrap();
    assert_eq!(right, vec![1, 2, 3, 4]);
}

#[test]
fn searchsorted_f64_gpu_matches_cpu() {
    ensure_cuda();
    let bounds_h = [-2.5f64, 0.0, 0.0, 4.25, 9.0];
    let vals_h = [-3.0f64, -2.5, 0.0, 1.0, 9.0, 100.0];
    let bounds = cpu_f64(&bounds_h).to(Device::Cuda(0)).unwrap();
    let vals = cpu_f64(&vals_h).to(Device::Cuda(0)).unwrap();
    assert!(bounds.is_cuda() && vals.is_cuda());

    for right in [false, true] {
        let got = searchsorted(&bounds, &vals, right).unwrap();
        assert_eq!(got, cpu_ref(&bounds_h, &vals_h, right));
    }
}

#[test]
fn bucketize_f32_gpu_matches_cpu() {
    // bucketize(input, boundaries, right) == searchsorted(boundaries, input).
    ensure_cuda();
    let bounds_h = [0.0f32, 1.0, 2.0, 3.0];
    let input_h = [-0.5f32, 0.5, 1.5, 2.5, 3.5];
    let bounds = cpu_f32(&bounds_h).to(Device::Cuda(0)).unwrap();
    let input = cpu_f32(&input_h).to(Device::Cuda(0)).unwrap();
    assert!(bounds.is_cuda() && input.is_cuda());

    let bounds64: Vec<f64> = bounds_h.iter().map(|&x| x as f64).collect();
    let input64: Vec<f64> = input_h.iter().map(|&x| x as f64).collect();

    let got = bucketize(&input, &bounds, false).unwrap();
    assert_eq!(got, cpu_ref(&bounds64, &input64, false));
    assert_eq!(got, vec![0, 1, 2, 3, 4]);
}

#[test]
fn searchsorted_f32_gpu_equals_cpu_path_on_same_data() {
    // Cross-check: the GPU path and the pure-CPU path agree element-for-element
    // on a non-trivial dataset including duplicates and exact-boundary hits.
    ensure_cuda();
    let bounds_h = [-5.0f32, -1.0, -1.0, 0.0, 2.0, 2.0, 8.0];
    let vals_h = [-6.0f32, -5.0, -1.0, -0.5, 0.0, 2.0, 7.9, 8.0, 9.0];

    let bounds_cpu = cpu_f32(&bounds_h);
    let vals_cpu = cpu_f32(&vals_h);
    let bounds_gpu = bounds_cpu.clone().to(Device::Cuda(0)).unwrap();
    let vals_gpu = vals_cpu.clone().to(Device::Cuda(0)).unwrap();
    assert!(bounds_gpu.is_cuda() && vals_gpu.is_cuda());

    for right in [false, true] {
        let cpu = searchsorted(&bounds_cpu, &vals_cpu, right).unwrap();
        let gpu = searchsorted(&bounds_gpu, &vals_gpu, right).unwrap();
        assert_eq!(gpu, cpu, "right={right}");
    }
}

#[test]
fn searchsorted_f16_gpu_matches_torch_nan_inf_ties() {
    // Live PyTorch 2.11.0+cu130 CUDA oracle for dtype=torch.float16:
    // bounds=[-2,-1,0,0,2,inf], vals=[-inf,-1,0,1,2,nan,inf].
    ensure_cuda();
    let bounds_h = [
        f16::from_f32(-2.0),
        f16::from_f32(-1.0),
        f16::from_f32(0.0),
        f16::from_f32(0.0),
        f16::from_f32(2.0),
        f16::from_f32(f32::INFINITY),
    ];
    let vals_h = [
        f16::from_f32(f32::NEG_INFINITY),
        f16::from_f32(-1.0),
        f16::from_f32(0.0),
        f16::from_f32(1.0),
        f16::from_f32(2.0),
        f16::from_f32(f32::NAN),
        f16::from_f32(f32::INFINITY),
    ];
    let bounds = cpu_f16(&bounds_h).to(Device::Cuda(0)).unwrap();
    let vals = cpu_f16(&vals_h).to(Device::Cuda(0)).unwrap();
    assert!(bounds.is_cuda() && vals.is_cuda());

    let left = searchsorted(&bounds, &vals, false).unwrap();
    assert_eq!(left, vec![0, 1, 2, 4, 4, 6, 5]);

    let right = searchsorted(&bounds, &vals, true).unwrap();
    assert_eq!(right, vec![0, 2, 4, 4, 5, 6, 6]);
}

#[test]
fn bucketize_bf16_gpu_matches_torch_nan_inf_ties() {
    // Same live PyTorch 2.11.0+cu130 CUDA oracle for dtype=torch.bfloat16.
    ensure_cuda();
    let bounds_h = [
        bf16::from_f32(-2.0),
        bf16::from_f32(-1.0),
        bf16::from_f32(0.0),
        bf16::from_f32(0.0),
        bf16::from_f32(2.0),
        bf16::from_f32(f32::INFINITY),
    ];
    let vals_h = [
        bf16::from_f32(f32::NEG_INFINITY),
        bf16::from_f32(-1.0),
        bf16::from_f32(0.0),
        bf16::from_f32(1.0),
        bf16::from_f32(2.0),
        bf16::from_f32(f32::NAN),
        bf16::from_f32(f32::INFINITY),
    ];
    let bounds = cpu_bf16(&bounds_h).to(Device::Cuda(0)).unwrap();
    let vals = cpu_bf16(&vals_h).to(Device::Cuda(0)).unwrap();
    assert!(bounds.is_cuda() && vals.is_cuda());

    let left = bucketize(&vals, &bounds, false).unwrap();
    assert_eq!(left, vec![0, 1, 2, 4, 4, 6, 5]);

    let right = bucketize(&vals, &bounds, true).unwrap();
    assert_eq!(right, vec![0, 2, 4, 4, 5, 6, 6]);
}

#[test]
fn searchsorted_f16_gpu_empty_boundaries_returns_zeros() {
    ensure_cuda();
    let bounds = cpu_f16(&[]).to(Device::Cuda(0)).unwrap();
    let vals_h = [
        f16::from_f32(-3.0),
        f16::from_f32(0.0),
        f16::from_f32(f32::NAN),
    ];
    let vals = cpu_f16(&vals_h).to(Device::Cuda(0)).unwrap();
    assert!(bounds.is_cuda() && vals.is_cuda());

    assert_eq!(searchsorted(&bounds, &vals, false).unwrap(), vec![0, 0, 0]);
    assert_eq!(searchsorted(&bounds, &vals, true).unwrap(), vec![0, 0, 0]);
}
