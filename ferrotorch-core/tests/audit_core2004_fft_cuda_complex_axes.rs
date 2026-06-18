#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::fft::{FftNorm, fft2_norm, fftn_norm, ifftn_norm};
use ferrotorch_core::grad_fns::fft::fftn_differentiable_norm;
use ferrotorch_core::grad_fns::reduction::sum;
use ferrotorch_core::{Device, Tensor, TensorStorage};

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for CORE-2004 audit tests");
    });
}

fn tensor_f64(data: &[f64], shape: &[usize], requires_grad: bool) -> Tensor<f64> {
    Tensor::from_storage(
        TensorStorage::cpu(data.to_vec()),
        shape.to_vec(),
        requires_grad,
    )
    .expect("cpu f64 tensor")
}

fn tensor_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .expect("cpu f32 tensor")
}

fn to_vec64(t: &Tensor<f64>) -> Vec<f64> {
    t.to(Device::Cpu)
        .expect("download f64 tensor")
        .data_vec()
        .expect("logical f64 data")
}

fn to_vec32(t: &Tensor<f32>) -> Vec<f32> {
    t.to(Device::Cpu)
        .expect("download f32 tensor")
        .data_vec()
        .expect("logical f32 data")
}

fn assert_close64(actual: &[f64], expected: &[f64], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let allowed = 2e-9 * e.abs().max(1.0);
        let diff = (a - e).abs();
        assert!(
            diff <= allowed,
            "{label}: index {i} diff {diff:.3e} exceeds {allowed:.3e}; actual={a} expected={e}"
        );
    }
}

fn assert_close32(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let allowed = 3e-4_f32 * e.abs().max(1.0);
        let diff = (a - e).abs();
        assert!(
            diff <= allowed,
            "{label}: index {i} diff {diff:.3e} exceeds {allowed:.3e}; actual={a} expected={e}"
        );
    }
}

fn complex_data64(len: usize) -> Vec<f64> {
    (0..len)
        .map(|i| {
            let base = (i as f64 + 1.0) * 0.125;
            if i % 2 == 0 {
                base.sin() + (i % 7) as f64 * 0.03125
            } else {
                -base.cos() + (i % 5) as f64 * 0.0625
            }
        })
        .collect()
}

fn complex_data32(len: usize) -> Vec<f32> {
    complex_data64(len).into_iter().map(|v| v as f32).collect()
}

#[test]
fn cuda_fftn_noninnermost_axis_norm_modes_match_cpu_reference_f64() {
    ensure_cuda_backend();

    let shape = [2usize, 3, 4, 2];
    let data = complex_data64(shape.iter().product());
    let cpu = tensor_f64(&data, &shape, false);
    let gpu = cpu.to(Device::Cuda(0)).expect("upload complex f64");
    let axes = [0isize];

    for norm in [FftNorm::Backward, FftNorm::Forward, FftNorm::Ortho] {
        let expected = fftn_norm(&cpu, None, Some(&axes), norm).expect("cpu fftn axes=0");
        let actual = fftn_norm(&gpu, None, Some(&axes), norm).expect("cuda fftn axes=0");
        assert_eq!(actual.device(), Device::Cuda(0), "fftn stays CUDA");
        assert_eq!(actual.shape(), expected.shape(), "fftn shape");
        assert_close64(&to_vec64(&actual), &to_vec64(&expected), "fftn axes=0");

        let expected = ifftn_norm(&cpu, None, Some(&axes), norm).expect("cpu ifftn axes=0");
        let actual = ifftn_norm(&gpu, None, Some(&axes), norm).expect("cuda ifftn axes=0");
        assert_eq!(actual.device(), Device::Cuda(0), "ifftn stays CUDA");
        assert_eq!(actual.shape(), expected.shape(), "ifftn shape");
        assert_close64(&to_vec64(&actual), &to_vec64(&expected), "ifftn axes=0");
    }
}

#[test]
fn cuda_fftn_resize_noncontiguous_axes_matches_cpu_reference_f32() {
    ensure_cuda_backend();

    let shape = [2usize, 3, 4, 2];
    let data = complex_data32(shape.iter().product());
    let cpu = tensor_f32(&data, &shape);
    let gpu = cpu.to(Device::Cuda(0)).expect("upload complex f32");
    let axes = [0isize, 2];
    let sizes = [3usize, 5];

    let expected = fftn_norm(&cpu, Some(&sizes), Some(&axes), FftNorm::Ortho)
        .expect("cpu fftn resize noncontiguous axes");
    let actual = fftn_norm(&gpu, Some(&sizes), Some(&axes), FftNorm::Ortho)
        .expect("cuda fftn resize noncontiguous axes");
    assert_eq!(actual.device(), Device::Cuda(0), "fftn resize stays CUDA");
    assert_eq!(actual.shape(), expected.shape(), "fftn resize shape");
    assert_close32(&to_vec32(&actual), &to_vec32(&expected), "fftn resize");

    let expected = ifftn_norm(&cpu, Some(&sizes), Some(&axes), FftNorm::Forward)
        .expect("cpu ifftn resize noncontiguous axes");
    let actual = ifftn_norm(&gpu, Some(&sizes), Some(&axes), FftNorm::Forward)
        .expect("cuda ifftn resize noncontiguous axes");
    assert_eq!(actual.device(), Device::Cuda(0), "ifftn resize stays CUDA");
    assert_eq!(actual.shape(), expected.shape(), "ifftn resize shape");
    assert_close32(&to_vec32(&actual), &to_vec32(&expected), "ifftn resize");
}

#[test]
fn cuda_fft2_batched_resize_norm_matches_cpu_reference_f64() {
    ensure_cuda_backend();

    let shape = [2usize, 3, 4, 2];
    let data = complex_data64(shape.iter().product());
    let cpu = tensor_f64(&data, &shape, false);
    let gpu = cpu.to(Device::Cuda(0)).expect("upload batched complex f64");
    let sizes = [4usize, 2];

    let expected =
        fft2_norm(&cpu, Some(&sizes), None, FftNorm::Forward).expect("cpu fft2 batched resize");
    let actual =
        fft2_norm(&gpu, Some(&sizes), None, FftNorm::Forward).expect("cuda fft2 batched resize");
    assert_eq!(actual.device(), Device::Cuda(0), "fft2 stays CUDA");
    assert_eq!(actual.shape(), expected.shape(), "fft2 shape");
    assert_close64(
        &to_vec64(&actual),
        &to_vec64(&expected),
        "fft2 batched resize",
    );
}

#[test]
fn cuda_fftn_noninnermost_backward_keeps_grad_cuda_matches_cpu_reference() {
    ensure_cuda_backend();

    let shape = [2usize, 3, 4, 2];
    let data = complex_data64(shape.iter().product());
    let axes = [0isize];

    let cpu = tensor_f64(&data, &shape, true);
    let cpu_out = fftn_differentiable_norm(&cpu, None, Some(&axes), FftNorm::Ortho)
        .expect("cpu differentiable fftn");
    sum(&cpu_out)
        .expect("cpu loss")
        .backward()
        .expect("cpu backward");
    let cpu_grad = cpu.grad().unwrap().expect("cpu grad");

    let gpu = tensor_f64(&data, &shape, false)
        .to(Device::Cuda(0))
        .expect("upload CUDA leaf")
        .requires_grad_(true);
    let gpu_out = fftn_differentiable_norm(&gpu, None, Some(&axes), FftNorm::Ortho)
        .expect("cuda differentiable fftn");
    assert_eq!(gpu_out.device(), Device::Cuda(0), "forward stays CUDA");
    sum(&gpu_out)
        .expect("cuda loss")
        .backward()
        .expect("cuda backward");
    let gpu_grad = gpu.grad().unwrap().expect("cuda grad");
    assert_eq!(gpu_grad.device(), Device::Cuda(0), "grad stays CUDA");
    assert_eq!(gpu_grad.shape(), cpu_grad.shape(), "grad shape");
    assert_close64(&to_vec64(&gpu_grad), &to_vec64(&cpu_grad), "fftn grad");
}
