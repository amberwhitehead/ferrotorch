use ferrotorch_core::error::FerrotorchResult;
use ferrotorch_core::fft::{fftn, ifftn};
use ferrotorch_core::{Tensor, TensorStorage};

#[cfg(feature = "gpu")]
use std::sync::Once;

#[cfg(feature = "gpu")]
use ferrotorch_core::Device;

#[cfg(feature = "gpu")]
static GPU_INIT: Once = Once::new();

#[cfg(feature = "gpu")]
fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for CORE-159 audit tests");
    });
}

fn complex_f64_tensor() -> Tensor<f64> {
    let shape = vec![2usize, 3, 2];
    let data: Vec<f64> = (0..shape.iter().product::<usize>())
        .map(|i| {
            let v = i as f64 + 1.0;
            if i % 2 == 0 { v * 0.25 } else { -v * 0.125 }
        })
        .collect();
    Tensor::from_storage(TensorStorage::cpu(data), shape, false).expect("complex f64 tensor")
}

fn tensor_data(t: &Tensor<f64>) -> Vec<f64> {
    t.data_vec().expect("logical tensor data")
}

fn assert_err_contains<T>(label: &str, result: FerrotorchResult<T>, needle: &str) {
    match result {
        Ok(_) => panic!("{label}: expected error containing {needle:?}"),
        Err(err) => {
            let text = format!("{err}");
            assert!(
                text.contains(needle),
                "{label}: expected error containing {needle:?}, got {text}"
            );
        }
    }
}

fn assert_c2c_nd_axes_validation_matches_torch(x: &Tensor<f64>, device_label: &str) {
    // PyTorch 2.11.0+cu130 reference for complex logical shape [2, 3]:
    // - s=(2,3,4): RuntimeError "Got shape with 3 values..."
    // - dim=(0,1,2): IndexError "Dimension out of range..."
    // - dim=(0,1,0): RuntimeError "FFT dims must be unique"
    // - s=(2,), dim=(0,1): RuntimeError "dim and shape ... same length"
    assert_err_contains(
        &format!("{device_label} fftn overlong shape"),
        fftn(x, Some(&[2, 3, 4]), None),
        "Got shape with 3 values but input tensor only has 2 dimensions",
    );
    assert_err_contains(
        &format!("{device_label} ifftn overlong shape"),
        ifftn(x, Some(&[2, 3, 4]), None),
        "Got shape with 3 values but input tensor only has 2 dimensions",
    );

    assert_err_contains(
        &format!("{device_label} fftn overlong axes"),
        fftn(x, None, Some(&[0, 1, 2])),
        "Dimension out of range",
    );
    assert_err_contains(
        &format!("{device_label} ifftn overlong axes"),
        ifftn(x, None, Some(&[0, 1, 2])),
        "Dimension out of range",
    );

    assert_err_contains(
        &format!("{device_label} fftn duplicate overlong axes"),
        fftn(x, None, Some(&[0, 1, 0])),
        "FFT dims must be unique",
    );
    assert_err_contains(
        &format!("{device_label} ifftn duplicate overlong axes"),
        ifftn(x, None, Some(&[0, 1, 0])),
        "FFT dims must be unique",
    );

    assert_err_contains(
        &format!("{device_label} fftn shape dim length mismatch"),
        fftn(x, Some(&[2]), Some(&[0, 1])),
        "dim and shape arguments must have the same length",
    );
    assert_err_contains(
        &format!("{device_label} ifftn shape dim length mismatch"),
        ifftn(x, Some(&[2]), Some(&[0, 1])),
        "dim and shape arguments must have the same length",
    );
}

#[test]
fn cpu_fftn_ifftn_reject_overlong_shape_and_axes_without_panic() {
    let x = complex_f64_tensor();
    assert_c2c_nd_axes_validation_matches_torch(&x, "cpu");
}

#[test]
fn cpu_fftn_ifftn_empty_axes_are_identity() {
    let x = complex_f64_tensor();
    let expected = tensor_data(&x);

    let fft_identity = fftn(&x, None, Some(&[])).expect("fftn empty axes");
    assert_eq!(fft_identity.shape(), x.shape());
    assert_eq!(tensor_data(&fft_identity), expected);

    let ifft_identity = ifftn(&x, Some(&[]), None).expect("ifftn empty shape");
    assert_eq!(ifft_identity.shape(), x.shape());
    assert_eq!(tensor_data(&ifft_identity), expected);
}

#[cfg(feature = "gpu")]
#[test]
fn cuda_fftn_ifftn_reject_overlong_shape_and_axes_without_panic() {
    ensure_cuda_backend();
    let x = complex_f64_tensor()
        .to(Device::Cuda(0))
        .expect("upload complex f64");
    assert_c2c_nd_axes_validation_matches_torch(&x, "cuda");
}

#[cfg(feature = "gpu")]
#[test]
fn cuda_fftn_ifftn_empty_axes_are_cuda_resident_identity() {
    ensure_cuda_backend();
    let cpu = complex_f64_tensor();
    let expected = tensor_data(&cpu);
    let x = cpu.to(Device::Cuda(0)).expect("upload complex f64");

    let fft_identity = fftn(&x, None, Some(&[])).expect("cuda fftn empty axes");
    assert_eq!(fft_identity.device(), Device::Cuda(0));
    assert_eq!(fft_identity.shape(), x.shape());
    assert_eq!(
        tensor_data(&fft_identity.to(Device::Cpu).expect("download fft identity")),
        expected
    );

    let ifft_identity = ifftn(&x, Some(&[]), None).expect("cuda ifftn empty shape");
    assert_eq!(ifft_identity.device(), Device::Cuda(0));
    assert_eq!(ifft_identity.shape(), x.shape());
    assert_eq!(
        tensor_data(
            &ifft_identity
                .to(Device::Cpu)
                .expect("download ifft identity")
        ),
        expected
    );
}
