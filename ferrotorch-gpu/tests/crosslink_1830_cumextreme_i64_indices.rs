//! Regression probes for CORE-136 / #1830.
//!
//! PyTorch returns `LongTensor` indices from `cummax` / `cummin`. The CUDA
//! kernels must therefore return real `CudaBuffer<i64>` indices, not f32-encoded
//! positions that lose integer precision above 2^24.

#![cfg(feature = "cuda")]

use std::sync::Once;

use ferrotorch_gpu::{GpuDevice, cpu_to_gpu, gpu_to_cpu, init_cuda_backend, kernels};

fn ensure_cuda() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        init_cuda_backend().expect("CUDA backend init");
    });
}

fn device() -> GpuDevice {
    ensure_cuda();
    GpuDevice::new(0).expect("CUDA device 0")
}

#[test]
fn cummax_f32_returns_i64_indices() {
    let dev = device();
    let input = [1.0_f32, 4.0, 3.0, 5.0, 2.0];
    let gpu_in = cpu_to_gpu(&input, &dev).expect("upload");
    let (values, indices) =
        kernels::gpu_cummax(&gpu_in, 1, input.len(), 1, &dev).expect("gpu_cummax");

    assert_eq!(
        gpu_to_cpu(&values, &dev).expect("download values"),
        vec![1.0, 4.0, 4.0, 5.0, 5.0],
    );
    assert_eq!(
        gpu_to_cpu(&indices, &dev).expect("download indices"),
        vec![0_i64, 1, 1, 3, 3],
    );
}

#[test]
fn cummin_f64_returns_i64_indices() {
    let dev = device();
    let input = [3.0_f64, -2.0, -1.0, -4.0, 7.0];
    let gpu_in = cpu_to_gpu(&input, &dev).expect("upload");
    let (values, indices) =
        kernels::gpu_cummin_f64(&gpu_in, 1, input.len(), 1, &dev).expect("gpu_cummin_f64");

    assert_eq!(
        gpu_to_cpu(&values, &dev).expect("download values"),
        vec![3.0, -2.0, -2.0, -4.0, -4.0],
    );
    assert_eq!(
        gpu_to_cpu(&indices, &dev).expect("download indices"),
        vec![0_i64, 1, 1, 3, 3],
    );
}

#[test]
#[ignore = "allocates large GPU buffers to prove the >2^24 index precision bug"]
fn cummax_f32_preserves_indices_above_f32_exact_range() {
    let dev = device();
    let target = 16_777_217_usize;
    let len = target + 2;
    let mut input = vec![0.0_f32; len];
    input[target] = 1.0;

    let gpu_in = cpu_to_gpu(&input, &dev).expect("upload large input");
    let (_values, indices) = kernels::gpu_cummax(&gpu_in, 1, len, 1, &dev).expect("gpu_cummax");
    let host_indices = gpu_to_cpu(&indices, &dev).expect("download large indices");

    assert_eq!(host_indices[target], target as i64);
    assert_eq!(host_indices[target + 1], target as i64);
}
