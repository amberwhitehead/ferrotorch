#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::fft::{FftNorm, fft_norm, ifft_norm, irfft_norm, rfft_norm};
use ferrotorch_core::grad_fns::fft::{irfft_differentiable, rfft_differentiable};
use ferrotorch_core::grad_fns::reduction::sum;
use ferrotorch_core::{Device, Tensor, TensorStorage};

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for CORE-1904 audit tests");
    });
}

fn tensor(data: &[f64], shape: &[usize], requires_grad: bool) -> Tensor<f64> {
    Tensor::from_storage(
        TensorStorage::cpu(data.to_vec()),
        shape.to_vec(),
        requires_grad,
    )
    .unwrap()
}

fn cuda_leaf(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    tensor(data, shape, true)
        .detach()
        .to(Device::Cuda(0))
        .expect("upload CUDA leaf")
        .requires_grad_(true)
}

fn to_vec(t: &Tensor<f64>) -> Vec<f64> {
    t.to(Device::Cpu)
        .expect("download result")
        .data_vec()
        .expect("logical data")
}

fn assert_close(actual: &[f64], expected: &[f64], label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch actual={} expected={}",
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        let allowed = 2e-9 * e.abs().max(1.0);
        assert!(
            diff <= allowed,
            "{label}: index {i} diff {diff:.3e} exceeds {allowed:.3e}; actual={a} expected={e}"
        );
    }
}

#[test]
fn cuda_c2c_norm_modes_match_cpu_reference() {
    ensure_cuda_backend();
    let shape = [2, 5, 2];
    let data = [
        0.25, -0.75, 1.5, 0.125, -0.5, 0.875, 2.0, -1.25, -1.5, 0.375, -0.2, 0.4, 0.6, -0.8, 1.0,
        -1.2, 1.4, -1.6, 1.8, -2.0,
    ];
    let cpu = tensor(&data, &shape, false);
    let gpu = cpu.to(Device::Cuda(0)).expect("upload");

    for norm in [FftNorm::Forward, FftNorm::Ortho] {
        let expected = fft_norm(&cpu, Some(7), None, norm).expect("cpu fft_norm");
        let actual = fft_norm(&gpu, Some(7), None, norm).expect("cuda fft_norm");
        assert_eq!(
            actual.device(),
            Device::Cuda(0),
            "fft output must stay on CUDA"
        );
        assert_close(&to_vec(&actual), &to_vec(&expected), "fft_norm CUDA");

        let expected = ifft_norm(&cpu, Some(7), None, norm).expect("cpu ifft_norm");
        let actual = ifft_norm(&gpu, Some(7), None, norm).expect("cuda ifft_norm");
        assert_eq!(
            actual.device(),
            Device::Cuda(0),
            "ifft output must stay on CUDA"
        );
        assert_close(&to_vec(&actual), &to_vec(&expected), "ifft_norm CUDA");
    }
}

#[test]
fn cuda_real_fft_resize_norm_modes_match_cpu_reference() {
    ensure_cuda_backend();
    let real_shape = [2, 6];
    let real = [
        -1.0, 0.25, 1.5, -0.75, 0.5, 2.0, 1.25, -1.5, 0.875, -0.375, 0.625, -0.125,
    ];
    let cpu = tensor(&real, &real_shape, false);
    let gpu = cpu.to(Device::Cuda(0)).expect("upload");

    for norm in [FftNorm::Forward, FftNorm::Ortho] {
        let expected = rfft_norm(&cpu, Some(10), None, norm).expect("cpu rfft_norm");
        let actual = rfft_norm(&gpu, Some(10), None, norm).expect("cuda rfft_norm");
        assert_eq!(
            actual.device(),
            Device::Cuda(0),
            "rfft output must stay on CUDA"
        );
        assert_close(&to_vec(&actual), &to_vec(&expected), "rfft_norm CUDA pad");

        let expected = rfft_norm(&cpu, Some(4), None, norm).expect("cpu rfft_norm");
        let actual = rfft_norm(&gpu, Some(4), None, norm).expect("cuda rfft_norm");
        assert_eq!(
            actual.device(),
            Device::Cuda(0),
            "rfft output must stay on CUDA"
        );
        assert_close(
            &to_vec(&actual),
            &to_vec(&expected),
            "rfft_norm CUDA truncate",
        );
    }
}

#[test]
fn cuda_irfft_resize_norm_modes_match_cpu_reference() {
    ensure_cuda_backend();
    let half_shape = [2, 6, 2];
    let half = [
        1.0, 0.0, 0.25, -0.5, -0.75, 0.125, 1.25, -1.5, 0.5, 0.875, -0.25, 0.0, -1.0, 0.0, 0.75,
        -0.625, -1.25, 0.375, 0.875, 1.125, -0.5, -0.25, 0.125, 0.0,
    ];
    let cpu = tensor(&half, &half_shape, false);
    let gpu = cpu.to(Device::Cuda(0)).expect("upload");

    for norm in [FftNorm::Forward, FftNorm::Ortho] {
        let expected = irfft_norm(&cpu, Some(8), None, norm).expect("cpu irfft_norm");
        let actual = irfft_norm(&gpu, Some(8), None, norm).expect("cuda irfft_norm");
        assert_eq!(
            actual.device(),
            Device::Cuda(0),
            "irfft output must stay on CUDA"
        );
        assert_close(
            &to_vec(&actual),
            &to_vec(&expected),
            "irfft_norm CUDA truncate",
        );

        let expected = irfft_norm(&cpu, Some(14), None, norm).expect("cpu irfft_norm");
        let actual = irfft_norm(&gpu, Some(14), None, norm).expect("cuda irfft_norm");
        assert_eq!(
            actual.device(),
            Device::Cuda(0),
            "irfft output must stay on CUDA"
        );
        assert_close(&to_vec(&actual), &to_vec(&expected), "irfft_norm CUDA pad");
    }
}

#[test]
fn cuda_rfft_backward_resizes_gradient_like_cpu_reference() {
    ensure_cuda_backend();
    let shape = [2, 6];
    let data = [
        -1.0, 0.25, 1.5, -0.75, 0.5, 2.0, 1.25, -1.5, 0.875, -0.375, 0.625, -0.125,
    ];

    for n in [4, 10] {
        let cpu = tensor(&data, &shape, true);
        let cpu_out = rfft_differentiable(&cpu, Some(n)).expect("cpu rfft diff");
        sum(&cpu_out)
            .expect("cpu loss")
            .backward()
            .expect("cpu backward");
        let cpu_grad = cpu.grad().unwrap().expect("cpu grad");

        let gpu = cuda_leaf(&data, &shape);
        let gpu_out = rfft_differentiable(&gpu, Some(n)).expect("cuda rfft diff");
        assert_eq!(
            gpu_out.device(),
            Device::Cuda(0),
            "forward must stay on CUDA"
        );
        sum(&gpu_out)
            .expect("cuda loss")
            .backward()
            .expect("cuda backward");
        let gpu_grad = gpu.grad().unwrap().expect("cuda grad");
        assert_eq!(
            gpu_grad.shape(),
            shape,
            "grad shape must match original input"
        );
        assert_eq!(gpu_grad.device(), Device::Cuda(0), "grad must stay on CUDA");
        assert_close(&to_vec(&gpu_grad), &to_vec(&cpu_grad), "rfft backward CUDA");
    }
}

#[test]
fn cuda_irfft_backward_resizes_gradient_like_cpu_reference() {
    ensure_cuda_backend();
    let shape = [2, 6, 2];
    let data = [
        1.0, 0.0, 0.25, -0.5, -0.75, 0.125, 1.25, -1.5, 0.5, 0.875, -0.25, 0.0, -1.0, 0.0, 0.75,
        -0.625, -1.25, 0.375, 0.875, 1.125, -0.5, -0.25, 0.125, 0.0,
    ];

    for n in [8, 14] {
        let cpu = tensor(&data, &shape, true);
        let cpu_out = irfft_differentiable(&cpu, Some(n)).expect("cpu irfft diff");
        sum(&cpu_out)
            .expect("cpu loss")
            .backward()
            .expect("cpu backward");
        let cpu_grad = cpu.grad().unwrap().expect("cpu grad");

        let gpu = cuda_leaf(&data, &shape);
        let gpu_out = irfft_differentiable(&gpu, Some(n)).expect("cuda irfft diff");
        assert_eq!(
            gpu_out.device(),
            Device::Cuda(0),
            "forward must stay on CUDA"
        );
        sum(&gpu_out)
            .expect("cuda loss")
            .backward()
            .expect("cuda backward");
        let gpu_grad = gpu.grad().unwrap().expect("cuda grad");
        assert_eq!(
            gpu_grad.shape(),
            shape,
            "grad shape must match original input"
        );
        assert_eq!(gpu_grad.device(), Device::Cuda(0), "grad must stay on CUDA");
        assert_close(
            &to_vec(&gpu_grad),
            &to_vec(&cpu_grad),
            "irfft backward CUDA",
        );
    }
}
