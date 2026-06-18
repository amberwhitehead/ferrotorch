#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::fft::{
    FftNorm, hfft_norm, hfft2_norm, hfftn_norm, ihfft_norm, ihfft2_norm, ihfftn_norm, irfft_norm,
    irfft2_norm, irfftn_norm, rfft_norm, rfft2_norm, rfftn_norm,
};
use ferrotorch_core::{Device, Tensor, TensorStorage};

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for CORE-2003 audit tests");
    });
}

fn tensor64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

fn tensor32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

fn to_vec64(t: &Tensor<f64>) -> Vec<f64> {
    t.to(Device::Cpu)
        .expect("download f64 result")
        .data_vec()
        .expect("logical f64 data")
}

fn to_vec32(t: &Tensor<f32>) -> Vec<f32> {
    t.to(Device::Cpu)
        .expect("download f32 result")
        .data_vec()
        .expect("logical f32 data")
}

fn assert_close64(actual: &[f64], expected: &[f64], label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch actual={} expected={}",
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        let allowed = 2e-8 * e.abs().max(1.0);
        assert!(
            diff <= allowed,
            "{label}: index {i} diff {diff:.3e} exceeds {allowed:.3e}; actual={a} expected={e}"
        );
    }
}

fn assert_close32(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch actual={} expected={}",
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        let allowed = 8e-4 * e.abs().max(1.0);
        assert!(
            diff <= allowed,
            "{label}: index {i} diff {diff:.3e} exceeds {allowed:.3e}; actual={a} expected={e}"
        );
    }
}

#[test]
fn cuda_1d_real_fft_nonlast_axis_f64_matches_cpu_reference() {
    ensure_cuda_backend();
    let real = [
        -1.0, 0.25, 1.5, -0.75, 0.5, 2.0, 1.25, -1.5, 0.875, -0.375, 0.625, -0.125,
    ];
    let cpu = tensor64(&real, &[3, 4]);
    let gpu = cpu.to(Device::Cuda(0)).expect("upload real");

    for norm in [FftNorm::Forward, FftNorm::Ortho] {
        let expected = rfft_norm(&cpu, Some(5), Some(0), norm).expect("cpu rfft dim0");
        let actual = rfft_norm(&gpu, Some(5), Some(0), norm).expect("cuda rfft dim0");
        assert_eq!(actual.shape(), expected.shape(), "rfft dim0 shape");
        assert_eq!(actual.device(), Device::Cuda(0), "rfft dim0 stays CUDA");
        assert_close64(&to_vec64(&actual), &to_vec64(&expected), "rfft dim0");

        let expected = ihfft_norm(&cpu, Some(6), Some(0), norm).expect("cpu ihfft dim0");
        let actual = ihfft_norm(&gpu, Some(6), Some(0), norm).expect("cuda ihfft dim0");
        assert_eq!(actual.shape(), expected.shape(), "ihfft dim0 shape");
        assert_eq!(actual.device(), Device::Cuda(0), "ihfft dim0 stays CUDA");
        assert_close64(&to_vec64(&actual), &to_vec64(&expected), "ihfft dim0");
    }
}

#[test]
fn cuda_1d_c2r_nonlast_axis_f64_matches_cpu_reference() {
    ensure_cuda_backend();
    let complex = [
        1.0, 0.0, 0.25, -0.5, -0.75, 0.125, 1.25, -1.5, 0.5, 0.875, -0.25, 0.0, -1.0, 0.0, 0.75,
        -0.625, -1.25, 0.375, 0.875, 1.125, -0.5, -0.25, 0.125, 0.0,
    ];
    let cpu = tensor64(&complex, &[3, 4, 2]);
    let gpu = cpu.to(Device::Cuda(0)).expect("upload complex");

    for norm in [FftNorm::Forward, FftNorm::Ortho] {
        let expected = irfft_norm(&cpu, Some(5), Some(0), norm).expect("cpu irfft dim0");
        let actual = irfft_norm(&gpu, Some(5), Some(0), norm).expect("cuda irfft dim0");
        assert_eq!(actual.shape(), expected.shape(), "irfft dim0 shape");
        assert_eq!(actual.device(), Device::Cuda(0), "irfft dim0 stays CUDA");
        assert_close64(&to_vec64(&actual), &to_vec64(&expected), "irfft dim0");

        let expected = hfft_norm(&cpu, Some(6), Some(0), norm).expect("cpu hfft dim0");
        let actual = hfft_norm(&gpu, Some(6), Some(0), norm).expect("cuda hfft dim0");
        assert_eq!(actual.shape(), expected.shape(), "hfft dim0 shape");
        assert_eq!(actual.device(), Device::Cuda(0), "hfft dim0 stays CUDA");
        assert_close64(&to_vec64(&actual), &to_vec64(&expected), "hfft dim0");
    }
}

#[test]
fn cuda_nd_real_and_hermitian_axes_f64_match_cpu_reference() {
    ensure_cuda_backend();
    let data: Vec<f64> = (0..24)
        .map(|i| ((i as f64) * 0.25 - 2.0).sin() + (i % 5) as f64 * 0.125)
        .collect();
    let axes = [0, 2];
    let shape = [3, 5];
    let cpu = tensor64(&data, &[2, 3, 4]);
    let gpu = cpu.to(Device::Cuda(0)).expect("upload real nd");

    let default_expected =
        rfftn_norm(&cpu, None, None, FftNorm::Backward).expect("cpu default rfftn");
    let default_actual =
        rfftn_norm(&gpu, None, None, FftNorm::Backward).expect("cuda default rfftn");
    assert_eq!(
        default_actual.shape(),
        default_expected.shape(),
        "default rfftn shape"
    );
    assert_eq!(
        default_actual.device(),
        Device::Cuda(0),
        "default rfftn stays CUDA"
    );
    assert_close64(
        &to_vec64(&default_actual),
        &to_vec64(&default_expected),
        "default rfftn",
    );

    let default_irfft_expected =
        irfftn_norm(&default_expected, None, None, FftNorm::Backward).expect("cpu default irfftn");
    let default_irfft_actual =
        irfftn_norm(&default_actual, None, None, FftNorm::Backward).expect("cuda default irfftn");
    assert_eq!(
        default_irfft_actual.shape(),
        default_irfft_expected.shape(),
        "default irfftn shape"
    );
    assert_eq!(
        default_irfft_actual.device(),
        Device::Cuda(0),
        "default irfftn stays CUDA"
    );
    assert_close64(
        &to_vec64(&default_irfft_actual),
        &to_vec64(&default_irfft_expected),
        "default irfftn",
    );

    let expected =
        rfftn_norm(&cpu, Some(&shape), Some(&axes), FftNorm::Ortho).expect("cpu rfftn axes");
    let actual =
        rfftn_norm(&gpu, Some(&shape), Some(&axes), FftNorm::Ortho).expect("cuda rfftn axes");
    assert_eq!(actual.shape(), expected.shape(), "rfftn axes shape");
    assert_eq!(actual.device(), Device::Cuda(0), "rfftn axes stays CUDA");
    assert_close64(&to_vec64(&actual), &to_vec64(&expected), "rfftn axes");

    let spectrum_cpu =
        rfftn_norm(&cpu, Some(&shape), Some(&axes), FftNorm::Backward).expect("cpu spectrum");
    let spectrum_gpu = spectrum_cpu.to(Device::Cuda(0)).expect("upload spectrum");
    let expected = irfftn_norm(&spectrum_cpu, Some(&shape), Some(&axes), FftNorm::Ortho)
        .expect("cpu irfftn axes");
    let actual = irfftn_norm(&spectrum_gpu, Some(&shape), Some(&axes), FftNorm::Ortho)
        .expect("cuda irfftn axes");
    assert_eq!(actual.shape(), expected.shape(), "irfftn axes shape");
    assert_eq!(actual.device(), Device::Cuda(0), "irfftn axes stays CUDA");
    assert_close64(&to_vec64(&actual), &to_vec64(&expected), "irfftn axes");

    let expected =
        ihfftn_norm(&cpu, Some(&shape), Some(&axes), FftNorm::Forward).expect("cpu ihfftn axes");
    let actual =
        ihfftn_norm(&gpu, Some(&shape), Some(&axes), FftNorm::Forward).expect("cuda ihfftn axes");
    assert_eq!(actual.shape(), expected.shape(), "ihfftn axes shape");
    assert_eq!(actual.device(), Device::Cuda(0), "ihfftn axes stays CUDA");
    assert_close64(&to_vec64(&actual), &to_vec64(&expected), "ihfftn axes");

    let hermitian_cpu =
        ihfftn_norm(&cpu, Some(&shape), Some(&axes), FftNorm::Backward).expect("cpu hermitian");
    let hermitian_gpu = hermitian_cpu.to(Device::Cuda(0)).expect("upload hermitian");
    let expected = hfftn_norm(&hermitian_cpu, Some(&shape), Some(&axes), FftNorm::Ortho)
        .expect("cpu hfftn axes");
    let actual = hfftn_norm(&hermitian_gpu, Some(&shape), Some(&axes), FftNorm::Ortho)
        .expect("cuda hfftn axes");
    assert_eq!(actual.shape(), expected.shape(), "hfftn axes shape");
    assert_eq!(actual.device(), Device::Cuda(0), "hfftn axes stays CUDA");
    assert_close64(&to_vec64(&actual), &to_vec64(&expected), "hfftn axes");
}

#[test]
fn cuda_2d_aliases_nonlast_axes_f32_match_cpu_reference() {
    ensure_cuda_backend();
    let data: Vec<f32> = (0..24)
        .map(|i| ((i as f32) * 0.375 - 1.5).cos() + (i % 7) as f32 * 0.03125)
        .collect();
    let axes = [0, 2];
    let shape = [4, 6];
    let cpu = tensor32(&data, &[2, 3, 4]);
    let gpu = cpu.to(Device::Cuda(0)).expect("upload f32 real");

    let expected =
        rfft2_norm(&cpu, Some(&shape), Some(&axes), FftNorm::Forward).expect("cpu rfft2 axes");
    let actual =
        rfft2_norm(&gpu, Some(&shape), Some(&axes), FftNorm::Forward).expect("cuda rfft2 axes");
    assert_eq!(actual.shape(), expected.shape(), "rfft2 axes shape");
    assert_eq!(actual.device(), Device::Cuda(0), "rfft2 axes stays CUDA");
    assert_close32(&to_vec32(&actual), &to_vec32(&expected), "rfft2 axes");

    let spectrum_cpu =
        rfft2_norm(&cpu, Some(&shape), Some(&axes), FftNorm::Backward).expect("cpu spectrum");
    let spectrum_gpu = spectrum_cpu.to(Device::Cuda(0)).expect("upload spectrum");
    let expected = irfft2_norm(&spectrum_cpu, Some(&shape), Some(&axes), FftNorm::Ortho)
        .expect("cpu irfft2 axes");
    let actual = irfft2_norm(&spectrum_gpu, Some(&shape), Some(&axes), FftNorm::Ortho)
        .expect("cuda irfft2 axes");
    assert_eq!(actual.shape(), expected.shape(), "irfft2 axes shape");
    assert_eq!(actual.device(), Device::Cuda(0), "irfft2 axes stays CUDA");
    assert_close32(&to_vec32(&actual), &to_vec32(&expected), "irfft2 axes");

    let expected =
        ihfft2_norm(&cpu, Some(&shape), Some(&axes), FftNorm::Forward).expect("cpu ihfft2 axes");
    let actual =
        ihfft2_norm(&gpu, Some(&shape), Some(&axes), FftNorm::Forward).expect("cuda ihfft2 axes");
    assert_eq!(actual.shape(), expected.shape(), "ihfft2 axes shape");
    assert_eq!(actual.device(), Device::Cuda(0), "ihfft2 axes stays CUDA");
    assert_close32(&to_vec32(&actual), &to_vec32(&expected), "ihfft2 axes");

    let hermitian_cpu =
        ihfft2_norm(&cpu, Some(&shape), Some(&axes), FftNorm::Backward).expect("cpu hermitian");
    let hermitian_gpu = hermitian_cpu.to(Device::Cuda(0)).expect("upload hermitian");
    let expected = hfft2_norm(&hermitian_cpu, Some(&shape), Some(&axes), FftNorm::Ortho)
        .expect("cpu hfft2 axes");
    let actual = hfft2_norm(&hermitian_gpu, Some(&shape), Some(&axes), FftNorm::Ortho)
        .expect("cuda hfft2 axes");
    assert_eq!(actual.shape(), expected.shape(), "hfft2 axes shape");
    assert_eq!(actual.device(), Device::Cuda(0), "hfft2 axes stays CUDA");
    assert_close32(&to_vec32(&actual), &to_vec32(&expected), "hfft2 axes");
}
