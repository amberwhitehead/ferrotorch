#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::device::Device;
use ferrotorch_core::dtype::DType;
use ferrotorch_core::grad_fns::linalg::permute_0213;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for CORE-190 CUDA parity tests");
    });
}

fn input_values() -> Vec<f32> {
    (0..(2 * 3 * 4 * 5)).map(|v| v as f32).collect()
}

fn expected_0213() -> Vec<f32> {
    let (d0, d1, d2, d3) = (2, 3, 4, 5);
    let data = input_values();
    let mut out = vec![0.0; d0 * d1 * d2 * d3];
    for i0 in 0..d0 {
        for i1 in 0..d1 {
            for i2 in 0..d2 {
                for i3 in 0..d3 {
                    let in_idx = ((i0 * d1 + i1) * d2 + i2) * d3 + i3;
                    let out_idx = ((i0 * d2 + i2) * d1 + i1) * d3 + i3;
                    out[out_idx] = data[in_idx];
                }
            }
        }
    }
    out
}

fn assert_shape_device_dtype<T: ferrotorch_core::dtype::Float>(
    t: &Tensor<T>,
    dtype: DType,
    label: &str,
) {
    assert_eq!(t.shape(), &[2, 4, 3, 5], "{label}: shape");
    assert_eq!(t.device(), Device::Cuda(0), "{label}: device");
    assert_eq!(t.gpu_handle().unwrap().dtype(), dtype, "{label}: dtype");
}

fn assert_exact_f32(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    for (i, (&got, &want)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(got, want, "{label}: index {i}");
    }
}

#[test]
fn cuda_permute_0213_preserves_float_dtype_device_and_values() {
    ensure_cuda_backend();
    let expected = expected_0213();
    let a = Tensor::from_storage(TensorStorage::cpu(input_values()), vec![2, 3, 4, 5], false)
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();

    let y = permute_0213(&a).expect("f32 CUDA permute_0213");
    assert_shape_device_dtype(&y, DType::F32, "f32 output");
    let cpu = y.cpu().unwrap();
    let actual = cpu.data().unwrap();
    assert_exact_f32(actual, &expected, "f32 values");
}

#[test]
fn cuda_permute_0213_preserves_double_dtype_device_and_values() {
    ensure_cuda_backend();
    let expected: Vec<f64> = expected_0213().into_iter().map(f64::from).collect();
    let data: Vec<f64> = input_values().into_iter().map(f64::from).collect();
    let a = Tensor::from_storage(TensorStorage::cpu(data), vec![2, 3, 4, 5], false)
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();

    let y = permute_0213(&a).expect("f64 CUDA permute_0213");
    assert_shape_device_dtype(&y, DType::F64, "f64 output");
    assert_eq!(&y.cpu().unwrap().data().unwrap(), &expected, "f64 values");
}

#[test]
fn cuda_permute_0213_preserves_f16_dtype_device_and_values() {
    ensure_cuda_backend();
    let data: Vec<half::f16> = input_values()
        .into_iter()
        .map(half::f16::from_f32)
        .collect();
    let expected: Vec<f32> = expected_0213();
    let a = Tensor::from_storage(TensorStorage::cpu(data), vec![2, 3, 4, 5], false)
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();

    let y = permute_0213(&a).expect("f16 CUDA permute_0213");
    assert_shape_device_dtype(&y, DType::F16, "f16 output");
    let actual: Vec<f32> = y
        .cpu()
        .unwrap()
        .data()
        .unwrap()
        .iter()
        .map(|v| v.to_f32())
        .collect();
    assert_exact_f32(&actual, &expected, "f16 values");
}

#[test]
fn cuda_permute_0213_preserves_bf16_dtype_device_and_values() {
    ensure_cuda_backend();
    let data: Vec<half::bf16> = input_values()
        .into_iter()
        .map(half::bf16::from_f32)
        .collect();
    let expected: Vec<f32> = expected_0213();
    let a = Tensor::from_storage(TensorStorage::cpu(data), vec![2, 3, 4, 5], false)
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();

    let y = permute_0213(&a).expect("bf16 CUDA permute_0213");
    assert_shape_device_dtype(&y, DType::BF16, "bf16 output");
    let actual: Vec<f32> = y
        .cpu()
        .unwrap()
        .data()
        .unwrap()
        .iter()
        .map(|v| v.to_f32())
        .collect();
    assert_exact_f32(&actual, &expected, "bf16 values");
}
