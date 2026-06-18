//! Regression tests for #2022: `torch.fft.fftfreq` / `rfftfreq` dtype and
//! device factory parity.
//!
//! The old helpers only returned CPU `Tensor<f64>`. PyTorch accepts dtype and
//! device options, including CUDA and meta. These tests pin the new
//! `*_on_device` API and, under the `gpu` feature, require CUDA outputs to be
//! resident on CUDA before any explicit readback.

use ferrotorch_core::{Device, FerrotorchError, fftfreq_on_device, rfftfreq_on_device};

fn assert_nan(value: f32, label: &str) {
    assert!(value.is_nan(), "{label}: expected NaN, got {value:?}");
}

fn assert_pos_inf(value: f32, label: &str) {
    assert!(
        value.is_infinite() && value.is_sign_positive(),
        "{label}: expected +inf, got {value:?}"
    );
}

fn assert_neg_inf(value: f32, label: &str) {
    assert!(
        value.is_infinite() && value.is_sign_negative(),
        "{label}: expected -inf, got {value:?}"
    );
}

#[test]
fn cpu_dtype_variants_match_torch_rounding() {
    let f32_full = fftfreq_on_device::<f32>(5, 0.3, Device::Cpu).expect("f32 fftfreq");
    assert_eq!(
        f32_full.data().unwrap(),
        &[0.0, 0.6666667, 1.3333334, -1.3333334, -0.6666667]
    );

    let f16_real = rfftfreq_on_device::<half::f16>(5, -2.0, Device::Cpu).expect("f16 rfftfreq");
    let f16: Vec<f32> = f16_real
        .data()
        .unwrap()
        .iter()
        .map(|v| v.to_f32())
        .collect();
    assert_eq!(f16, &[-0.0, -0.099975586, -0.19995117]);
    assert!(f16[0].is_sign_negative(), "f16 zero bin keeps -0.0");

    let bf16_full = fftfreq_on_device::<half::bf16>(5, -2.0, Device::Cpu).expect("bf16 fftfreq");
    let bf16: Vec<f32> = bf16_full
        .data()
        .unwrap()
        .iter()
        .map(|v| v.to_f32())
        .collect();
    assert_eq!(
        bf16,
        &[-0.0, -0.100097656, -0.20019531, 0.20019531, 0.100097656]
    );
    assert!(bf16[0].is_sign_negative(), "bf16 zero bin keeps -0.0");
}

#[test]
fn cpu_dtype_variants_preserve_edge_values() {
    let full = fftfreq_on_device::<f32>(4, -0.0, Device::Cpu).expect("f32 fftfreq d=-0");
    let full = full.data().unwrap();
    assert_nan(full[0], "fftfreq bin 0");
    assert_neg_inf(full[1], "fftfreq bin 1");
    assert_pos_inf(full[2], "fftfreq bin 2");
    assert_pos_inf(full[3], "fftfreq bin 3");

    let real = rfftfreq_on_device::<f32>(0, 1.0, Device::Cpu).expect("f32 rfftfreq n=0");
    assert_eq!(real.shape(), &[1]);
    assert_nan(real.data().unwrap()[0], "rfftfreq n=0 bin 0");
}

#[test]
fn meta_device_returns_shape_and_dtype_without_data() {
    let full = fftfreq_on_device::<f32>(5, -2.0, Device::Meta).expect("meta fftfreq");
    assert_eq!(full.shape(), &[5]);
    assert_eq!(full.device(), Device::Meta);
    assert!(full.data().is_err(), "meta tensor must not expose data");

    let real = rfftfreq_on_device::<half::bf16>(0, 1.0, Device::Meta).expect("meta rfftfreq");
    assert_eq!(real.shape(), &[1]);
    assert_eq!(real.device(), Device::Meta);
    assert!(real.data().is_err(), "meta tensor must not expose data");
}

#[test]
fn unsupported_accelerators_do_not_cpu_fallback() {
    let err = fftfreq_on_device::<f32>(4, 1.0, Device::Mps(0)).unwrap_err();
    assert!(
        matches!(err, FerrotorchError::InvalidArgument { .. }),
        "MPS must reject without CPU fallback, got {err:?}"
    );

    let err = rfftfreq_on_device::<f32>(4, 1.0, Device::Xpu(0)).unwrap_err();
    assert!(
        matches!(err, FerrotorchError::InvalidArgument { .. }),
        "XPU must reject without CPU fallback, got {err:?}"
    );
}

#[cfg(feature = "gpu")]
mod cuda {
    use std::sync::Once;

    use ferrotorch_core::{Device, Tensor, fftfreq_on_device, rfftfreq_on_device};

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for fftfreq dtype/device audit tests");
        });
    }

    fn cpu_values<T: ferrotorch_core::Float>(tensor: &Tensor<T>) -> Vec<T> {
        assert_eq!(
            tensor.device(),
            Device::Cuda(0),
            "factory output must stay CUDA"
        );
        tensor
            .to(Device::Cpu)
            .expect("download CUDA frequency tensor")
            .data_vec()
            .expect("CPU logical data")
    }

    #[test]
    fn cuda_f32_and_f64_factories_are_resident_and_match_torch_values() {
        ensure_cuda_backend();

        let full = fftfreq_on_device::<f32>(5, 0.3, Device::Cuda(0)).expect("cuda f32 fftfreq");
        let full = cpu_values(&full);
        assert_eq!(full, &[0.0, 0.6666667, 1.3333334, -1.3333334, -0.6666667]);

        let real = rfftfreq_on_device::<f64>(5, -2.0, Device::Cuda(0)).expect("cuda f64 rfftfreq");
        let real = cpu_values(&real);
        assert_eq!(real, &[-0.0, -0.1, -0.2]);
        assert!(real[0].is_sign_negative(), "f64 zero bin keeps -0.0");
    }

    #[test]
    fn cuda_reduced_dtype_factories_are_resident_and_dtype_correct() {
        ensure_cuda_backend();

        let f16 =
            rfftfreq_on_device::<half::f16>(5, -2.0, Device::Cuda(0)).expect("cuda f16 rfftfreq");
        let f16: Vec<f32> = cpu_values(&f16).iter().map(|v| v.to_f32()).collect();
        assert_eq!(f16, &[-0.0, -0.099975586, -0.19995117]);
        assert!(f16[0].is_sign_negative(), "f16 zero bin keeps -0.0");

        let bf16 =
            fftfreq_on_device::<half::bf16>(5, -2.0, Device::Cuda(0)).expect("cuda bf16 fftfreq");
        let bf16: Vec<f32> = cpu_values(&bf16).iter().map(|v| v.to_f32()).collect();
        assert_eq!(
            bf16,
            &[-0.0, -0.100097656, -0.20019531, 0.20019531, 0.100097656]
        );
        assert!(bf16[0].is_sign_negative(), "bf16 zero bin keeps -0.0");
    }

    #[test]
    fn cuda_zero_spacing_and_zero_length_match_torch_values() {
        ensure_cuda_backend();

        let full = fftfreq_on_device::<f32>(4, -0.0, Device::Cuda(0)).expect("cuda d=-0");
        let full = cpu_values(&full);
        assert!(full[0].is_nan(), "zero bin is NaN");
        assert!(
            full[1].is_infinite() && full[1].is_sign_negative(),
            "positive bin over -0.0 is -inf"
        );
        assert!(
            full[2].is_infinite() && full[2].is_sign_positive(),
            "negative bin over -0.0 is +inf"
        );
        assert!(
            full[3].is_infinite() && full[3].is_sign_positive(),
            "negative bin over -0.0 is +inf"
        );

        let empty = fftfreq_on_device::<f64>(0, 0.0, Device::Cuda(0)).expect("cuda fftfreq n=0");
        assert_eq!(empty.device(), Device::Cuda(0));
        assert!(cpu_values(&empty).is_empty());

        let real = rfftfreq_on_device::<f32>(0, 1.0, Device::Cuda(0)).expect("cuda rfftfreq n=0");
        let real = cpu_values(&real);
        assert_eq!(real.len(), 1);
        assert!(real[0].is_nan());
    }
}
