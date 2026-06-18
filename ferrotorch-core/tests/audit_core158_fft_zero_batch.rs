//! Regression tests for CORE-158 / #1852.
//!
//! Live PyTorch 2.11.0+cu130 behavior separates two zero-sized FFT cases:
//! an empty transform axis with an explicit positive size is zero-padded and
//! transformed, while a zero-sized non-transform batch dimension reaches
//! MKL/cuFFT and raises a backend error. ferrotorch should not promote that
//! true zero batch to one transform, and it must not reject the valid
//! explicit-size empty-transform case.

use ferrotorch_core::fft::{
    FftNorm, fft_norm, fftn_norm, hfft_norm, ifft_norm, ifftn_norm, ihfft_norm, irfft_norm,
    rfft_norm,
};
use ferrotorch_core::{FerrotorchError, Tensor, TensorStorage};

fn empty_cpu(shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(Vec::<f64>::new()), shape.to_vec(), false)
        .expect("empty tensor shape must match empty storage")
}

fn assert_zero_batch_error<T: std::fmt::Debug>(result: Result<T, FerrotorchError>, op: &str) {
    match result {
        Err(FerrotorchError::InvalidArgument { message }) => {
            assert!(
                message.contains("zero-sized batch dimensions"),
                "{op}: expected zero-batch backend-parity error, got {message:?}"
            );
        }
        other => panic!("{op}: expected InvalidArgument zero-batch error, got {other:?}"),
    }
}

fn assert_all_zero(tensor: &Tensor<f64>, expected_shape: &[usize], op: &str) {
    assert_eq!(tensor.shape(), expected_shape, "{op}: output shape");
    let data = tensor.data_vec().expect("logical data");
    assert_eq!(
        data.len(),
        expected_shape.iter().product::<usize>(),
        "{op}: logical data length"
    );
    assert!(
        data.iter().all(|&value| value == 0.0),
        "{op}: zero-padded empty input should transform to exact zeros, got {data:?}"
    );
}

#[test]
fn cpu_zero_batch_nonzero_signal_rejects_like_torch_backends() {
    let complex = empty_cpu(&[0, 4, 2]);
    assert_zero_batch_error(fft_norm(&complex, Some(5), None, FftNorm::Backward), "fft");
    assert_zero_batch_error(
        ifft_norm(&complex, Some(5), None, FftNorm::Backward),
        "ifft",
    );
    assert_zero_batch_error(
        irfft_norm(&complex, Some(6), None, FftNorm::Backward),
        "irfft",
    );
    assert_zero_batch_error(
        hfft_norm(&complex, Some(6), None, FftNorm::Backward),
        "hfft",
    );

    let real = empty_cpu(&[0, 4]);
    assert_zero_batch_error(rfft_norm(&real, Some(6), None, FftNorm::Backward), "rfft");
    assert_zero_batch_error(ihfft_norm(&real, Some(6), None, FftNorm::Backward), "ihfft");
}

#[test]
fn cpu_empty_transform_axis_with_nonzero_batch_zero_pads_like_torch() {
    let complex = empty_cpu(&[2, 0, 2]);
    assert_all_zero(
        &fft_norm(&complex, Some(5), None, FftNorm::Backward).expect("fft zero-pad"),
        &[2, 5, 2],
        "fft",
    );
    assert_all_zero(
        &ifft_norm(&complex, Some(5), None, FftNorm::Backward).expect("ifft zero-pad"),
        &[2, 5, 2],
        "ifft",
    );
    assert_all_zero(
        &irfft_norm(&complex, Some(6), None, FftNorm::Backward).expect("irfft zero-pad"),
        &[2, 6],
        "irfft",
    );
    assert_all_zero(
        &hfft_norm(&complex, Some(6), None, FftNorm::Backward).expect("hfft zero-pad"),
        &[2, 6],
        "hfft",
    );

    let real = empty_cpu(&[2, 0]);
    assert_all_zero(
        &rfft_norm(&real, Some(6), None, FftNorm::Backward).expect("rfft zero-pad"),
        &[2, 4, 2],
        "rfft",
    );
    assert_all_zero(
        &ihfft_norm(&real, Some(6), None, FftNorm::Backward).expect("ihfft zero-pad"),
        &[2, 4, 2],
        "ihfft",
    );
}

#[test]
fn cpu_zero_batch_on_nonlast_axis_rejects_after_dim_resolution() {
    let complex = empty_cpu(&[4, 0, 2]);
    assert_zero_batch_error(
        fft_norm(&complex, Some(5), Some(0), FftNorm::Backward),
        "fft dim=0",
    );

    let real = empty_cpu(&[4, 0]);
    assert_zero_batch_error(
        rfft_norm(&real, Some(6), Some(0), FftNorm::Backward),
        "rfft dim=0",
    );
}

#[test]
fn cpu_c2c_empty_axes_identity_allows_zero_sized_tensors() {
    let complex = empty_cpu(&[0, 3, 2]);
    let fft = fftn_norm(&complex, None, Some(&[]), FftNorm::Backward).expect("fftn identity");
    assert_eq!(fft.shape(), &[0, 3, 2], "fftn identity shape");
    assert!(fft.data_vec().expect("fftn identity data").is_empty());

    let ifft = ifftn_norm(&complex, None, Some(&[]), FftNorm::Backward).expect("ifftn identity");
    assert_eq!(ifft.shape(), &[0, 3, 2], "ifftn identity shape");
    assert!(ifft.data_vec().expect("ifftn identity data").is_empty());
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use ferrotorch_core::Device;
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for CORE-158 audit tests");
        });
    }

    fn empty_cuda(shape: &[usize]) -> Tensor<f64> {
        empty_cpu(shape).to(Device::Cuda(0)).expect("upload empty")
    }

    fn assert_cuda_zeros(tensor: &Tensor<f64>, expected_shape: &[usize], op: &str) {
        assert_eq!(tensor.device(), Device::Cuda(0), "{op}: must stay CUDA");
        let cpu = tensor.to(Device::Cpu).expect("download result");
        assert_all_zero(&cpu, expected_shape, op);
    }

    #[test]
    fn cuda_zero_batch_nonzero_signal_rejects_before_cufft() {
        ensure_cuda_backend();

        let complex = empty_cuda(&[0, 4, 2]);
        assert_zero_batch_error(
            fft_norm(&complex, Some(5), None, FftNorm::Backward),
            "cuda fft",
        );
        assert_zero_batch_error(
            ifft_norm(&complex, Some(5), None, FftNorm::Backward),
            "cuda ifft",
        );
        assert_zero_batch_error(
            irfft_norm(&complex, Some(6), None, FftNorm::Backward),
            "cuda irfft",
        );
        assert_zero_batch_error(
            hfft_norm(&complex, Some(6), None, FftNorm::Backward),
            "cuda hfft",
        );

        let real = empty_cuda(&[0, 4]);
        assert_zero_batch_error(
            rfft_norm(&real, Some(6), None, FftNorm::Backward),
            "cuda rfft",
        );
        assert_zero_batch_error(
            ihfft_norm(&real, Some(6), None, FftNorm::Backward),
            "cuda ihfft",
        );
    }

    #[test]
    fn cuda_empty_transform_axis_with_nonzero_batch_zero_pads_on_device() {
        ensure_cuda_backend();

        let complex = empty_cuda(&[2, 0, 2]);
        assert_cuda_zeros(
            &fft_norm(&complex, Some(5), None, FftNorm::Backward).expect("cuda fft zero-pad"),
            &[2, 5, 2],
            "cuda fft",
        );
        assert_cuda_zeros(
            &ifft_norm(&complex, Some(5), None, FftNorm::Backward).expect("cuda ifft zero-pad"),
            &[2, 5, 2],
            "cuda ifft",
        );
        assert_cuda_zeros(
            &irfft_norm(&complex, Some(6), None, FftNorm::Backward).expect("cuda irfft zero-pad"),
            &[2, 6],
            "cuda irfft",
        );
        assert_cuda_zeros(
            &hfft_norm(&complex, Some(6), None, FftNorm::Backward).expect("cuda hfft zero-pad"),
            &[2, 6],
            "cuda hfft",
        );

        let real = empty_cuda(&[2, 0]);
        assert_cuda_zeros(
            &rfft_norm(&real, Some(6), None, FftNorm::Backward).expect("cuda rfft zero-pad"),
            &[2, 4, 2],
            "cuda rfft",
        );
        assert_cuda_zeros(
            &ihfft_norm(&real, Some(6), None, FftNorm::Backward).expect("cuda ihfft zero-pad"),
            &[2, 4, 2],
            "cuda ihfft",
        );
    }

    #[test]
    fn cuda_c2c_empty_axes_identity_allows_zero_sized_tensors() {
        ensure_cuda_backend();

        let complex = empty_cuda(&[0, 3, 2]);
        let fft =
            fftn_norm(&complex, None, Some(&[]), FftNorm::Backward).expect("cuda fftn identity");
        assert_eq!(fft.device(), Device::Cuda(0), "fftn identity stays CUDA");
        assert_eq!(fft.shape(), &[0, 3, 2], "fftn identity shape");

        let ifft =
            ifftn_norm(&complex, None, Some(&[]), FftNorm::Backward).expect("cuda ifftn identity");
        assert_eq!(ifft.device(), Device::Cuda(0), "ifftn identity stays CUDA");
        assert_eq!(ifft.shape(), &[0, 3, 2], "ifftn identity shape");
    }
}
