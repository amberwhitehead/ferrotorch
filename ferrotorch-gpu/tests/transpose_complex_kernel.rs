//! Correctness test for the complex transpose kernel (gpu_transpose_complex_f32
//! / _f64), which had NEVER run on hardware: TRANSPOSE_COMPLEX_F32_PTX failed
//! JIT with CUDA_ERROR_INVALID_PTX (a `%tid` register shadowed the builtin
//! special register), and the function returns Err(PtxCompileFailed) with no
//! CPU fallback — so the GPU complex-eig eigenvector repack (cusolver.rs) was
//! hard-broken. Fixed by renaming %tid -> %ltid. This pins the now-live kernel.
//!
//! Kernel semantics (per its doc + index math): a column-major -> row-major
//! repack of an n x n complex matrix. For flat complex index k in 0..n*n:
//!   out[k] == in[(k % n) * n + (k / n)]
//! where each complex element is two contiguous f32/f64 (re, im).
#![cfg(feature = "cuda")]

use ferrotorch_gpu::kernels::{gpu_transpose_complex_f32, gpu_transpose_complex_f64};
use ferrotorch_gpu::transfer::{cpu_to_gpu, gpu_to_cpu};
use ferrotorch_gpu::{GpuDevice, init_cuda_backend};

fn ensure_init() {
    if !ferrotorch_core::gpu_dispatch::has_gpu_backend() {
        init_cuda_backend().expect("init_cuda_backend");
    }
}

fn expected_repack(input: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0; input.len()];
    for k in 0..n * n {
        let in_idx = (k % n) * n + (k / n);
        out[2 * k] = input[2 * in_idx];
        out[2 * k + 1] = input[2 * in_idx + 1];
    }
    out
}

#[test]
fn transpose_complex_f32_repack_correct() {
    ensure_init();
    let dev = GpuDevice::new(0).expect("device");
    let n = 5;
    // input[2k]=re, input[2k+1]=im; distinct values per element
    let host: Vec<f32> = (0..2 * n * n).map(|i| i as f32 + 0.25).collect();
    let d_in = cpu_to_gpu(&host, &dev).expect("upload");
    let d_out = gpu_transpose_complex_f32(&d_in, n, &dev).expect("kernel must run on-device");
    let got = gpu_to_cpu(&d_out, &dev).expect("download");

    let host64: Vec<f64> = host.iter().map(|&x| x as f64).collect();
    let want = expected_repack(&host64, n);
    for k in 0..2 * n * n {
        assert!(
            (got[k] as f64 - want[k]).abs() < 1e-5,
            "f32 mismatch at {k}: got {}, want {}",
            got[k],
            want[k]
        );
    }
}

#[test]
fn transpose_complex_f64_repack_correct() {
    ensure_init();
    let dev = GpuDevice::new(0).expect("device");
    let n = 4;
    let host: Vec<f64> = (0..2 * n * n).map(|i| i as f64 * 1.5 - 3.0).collect();
    let d_in = cpu_to_gpu(&host, &dev).expect("upload");
    let d_out = gpu_transpose_complex_f64(&d_in, n, &dev).expect("kernel must run on-device");
    let got = gpu_to_cpu(&d_out, &dev).expect("download");

    let want = expected_repack(&host, n);
    for k in 0..2 * n * n {
        assert!(
            (got[k] - want[k]).abs() < 1e-12,
            "f64 mismatch at {k}: got {}, want {}",
            got[k],
            want[k]
        );
    }
}
