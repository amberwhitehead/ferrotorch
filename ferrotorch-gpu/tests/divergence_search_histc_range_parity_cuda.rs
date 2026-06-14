#![cfg(feature = "cuda")]

use ferrotorch_core::ops::search::{histc, meshgrid, searchsorted};
use ferrotorch_core::{Device, FerrotorchError, Tensor, TensorStorage};
use ferrotorch_gpu::init_cuda_backend;
use half::f16;

fn ensure_cuda() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        init_cuda_backend().expect("CUDA backend init");
    });
}

fn cpu_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).expect("cpu f32")
}

fn cpu_f64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).expect("cpu f64")
}

fn cpu_f16(data: &[f32], shape: &[usize]) -> Tensor<f16> {
    Tensor::from_storage(
        TensorStorage::cpu(data.iter().copied().map(f16::from_f32).collect()),
        shape.to_vec(),
        false,
    )
    .expect("cpu f16")
}

fn host_f32(t: &Tensor<f32>) -> Vec<f32> {
    t.cpu().expect("to cpu").data_vec().expect("host f32")
}

#[test]
fn cuda_histc_default_range_infers_from_narrowed_view_and_keeps_counts_resident() {
    ensure_cuda();
    let full = cpu_f32(&[99.0, 99.0, 3.0, 3.0, 3.0, 3.0, 99.0, 99.0], &[4, 2])
        .to(Device::Cuda(0))
        .expect("to cuda");
    let view = full.narrow(0, 1, 2).expect("narrow rows 1..3");
    assert_ne!(view.storage_offset(), 0);

    let out = histc(&view, 4, 0.0, 0.0).expect("histc default range");

    assert!(out.is_cuda(), "CUDA histc counts must stay resident");
    assert_eq!(host_f32(&out), vec![0.0, 0.0, 4.0, 0.0]);
}

#[test]
fn cuda_histc_default_range_nan_errors_like_torch() {
    ensure_cuda();
    let x = cpu_f32(&[f32::NAN, 1.0, 2.0], &[3])
        .to(Device::Cuda(0))
        .expect("to cuda");

    let err = histc(&x, 4, 0.0, 0.0).expect_err("NaN inferred range must error");

    assert!(
        matches!(err, FerrotorchError::InvalidArgument { .. }),
        "expected InvalidArgument for NaN inferred range, got {err:?}"
    );
    assert!(
        err.to_string().contains("not finite"),
        "error should name the non-finite inferred range, got {err}"
    );
}

#[test]
fn cuda_histc_default_range_infinite_f64_errors_like_torch() {
    ensure_cuda();
    let x = cpu_f64(&[f64::NEG_INFINITY, 1.0, 2.0], &[3])
        .to(Device::Cuda(0))
        .expect("to cuda");

    let err = histc(&x, 4, 0.0, 0.0).expect_err("infinite inferred range must error");

    assert!(
        matches!(err, FerrotorchError::InvalidArgument { .. }),
        "expected InvalidArgument for infinite inferred range, got {err:?}"
    );
    assert!(
        err.to_string().contains("not finite"),
        "error should name the non-finite inferred range, got {err}"
    );
}

#[test]
fn cuda_histc_f16_rejects_before_default_range_inference_like_torch() {
    ensure_cuda();
    let x = cpu_f16(&[1.0, 2.0, 3.0], &[3])
        .to(Device::Cuda(0))
        .expect("to cuda");

    let err = histc(&x, 4, 0.0, 0.0).expect_err("CUDA f16 histc is unsupported");

    assert!(
        matches!(err, FerrotorchError::NotImplementedOnCuda { op: "histc" }),
        "expected NotImplementedOnCuda for CUDA f16 histc, got {err:?}"
    );
}

#[test]
fn cuda_searchsorted_rejects_mixed_devices_like_torch() {
    ensure_cuda();
    let boundaries = cpu_f32(&[1.0, 2.0, 3.0], &[3]);
    let values = cpu_f32(&[1.5, 2.5], &[2])
        .to(Device::Cuda(0))
        .expect("to cuda");

    let err = searchsorted(&boundaries, &values, false)
        .expect_err("searchsorted must reject mixed CPU/CUDA inputs");

    assert!(
        matches!(err, FerrotorchError::DeviceMismatch { .. }),
        "expected DeviceMismatch for mixed-device searchsorted, got {err:?}"
    );
}

#[test]
fn cuda_meshgrid_rejects_mixed_devices_like_torch() {
    ensure_cuda();
    let x = cpu_f32(&[1.0, 2.0], &[2]);
    let y = cpu_f32(&[3.0, 4.0], &[2])
        .to(Device::Cuda(0))
        .expect("to cuda");

    let err = meshgrid(&[x, y]).expect_err("meshgrid must reject mixed CPU/CUDA inputs");

    assert!(
        matches!(err, FerrotorchError::DeviceMismatch { .. }),
        "expected DeviceMismatch for mixed-device meshgrid, got {err:?}"
    );
}
