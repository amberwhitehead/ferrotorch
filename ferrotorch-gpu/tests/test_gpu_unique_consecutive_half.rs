//! End-to-end CUDA `unique_consecutive` coverage for f16 / bf16.
//!
//! These tests exercise the production `ferrotorch_core::unique_consecutive`
//! path. The deduplicated values remain CUDA-resident with their original half
//! dtype tag; only the public host metadata vectors are materialized on CPU.

#![cfg(feature = "cuda")]

use ferrotorch_core::{Device, Tensor, TensorStorage, unique_consecutive};
use ferrotorch_gpu::init_cuda_backend;
use half::{bf16, f16};

fn ensure_cuda() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        init_cuda_backend().expect("CUDA backend init");
    });
}

fn cpu_f16(data: &[f16]) -> Tensor<f16> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false)
        .expect("cpu f16 tensor")
}

fn cpu_bf16(data: &[bf16]) -> Tensor<bf16> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false)
        .expect("cpu bf16 tensor")
}

fn read_f16(t: &Tensor<f16>) -> Vec<f32> {
    t.to(Device::Cpu)
        .expect("download f16 values")
        .data()
        .expect("host f16 data")
        .iter()
        .map(|v| v.to_f32())
        .collect()
}

fn read_bf16(t: &Tensor<bf16>) -> Vec<f32> {
    t.to(Device::Cpu)
        .expect("download bf16 values")
        .data()
        .expect("host bf16 data")
        .iter()
        .map(|v| v.to_f32())
        .collect()
}

#[test]
fn unique_consecutive_f16_cuda_nan_zero_runs_match_torch() {
    ensure_cuda();
    let data: Vec<f16> = [1.0_f32, 1.0, f32::NAN, f32::NAN, -0.0, 0.0, 2.0, 2.0, 1.0]
        .into_iter()
        .map(f16::from_f32)
        .collect();
    let input = cpu_f16(&data).to(Device::Cuda(0)).expect("upload f16");

    let (values, inverse, counts) =
        unique_consecutive(&input).expect("unique_consecutive f16 cuda");
    assert_eq!(values.device(), Device::Cuda(0));
    assert_eq!(values.shape(), &[6]);
    let host = read_f16(&values);
    assert_eq!(host.len(), 6);
    assert_eq!(host[0], 1.0);
    assert!(host[1].is_nan() && host[2].is_nan());
    assert_eq!(host[3], 0.0);
    assert_eq!(host[4], 2.0);
    assert_eq!(host[5], 1.0);
    assert_eq!(inverse, vec![0, 0, 1, 2, 3, 3, 4, 4, 5]);
    assert_eq!(counts, vec![2, 1, 1, 2, 2, 1]);
}

#[test]
fn unique_consecutive_bf16_cuda_runs_match_torch() {
    ensure_cuda();
    let data: Vec<bf16> = [3.0_f32, 3.0, -1.0, -1.0, -1.0, 4.0, 3.0]
        .into_iter()
        .map(bf16::from_f32)
        .collect();
    let input = cpu_bf16(&data).to(Device::Cuda(0)).expect("upload bf16");

    let (values, inverse, counts) =
        unique_consecutive(&input).expect("unique_consecutive bf16 cuda");
    assert_eq!(values.device(), Device::Cuda(0));
    assert_eq!(values.shape(), &[4]);
    assert_eq!(read_bf16(&values), vec![3.0, -1.0, 4.0, 3.0]);
    assert_eq!(inverse, vec![0, 0, 1, 1, 1, 2, 3]);
    assert_eq!(counts, vec![2, 3, 1, 1]);
}

#[test]
fn unique_consecutive_f16_cuda_empty_returns_empty_cuda_values() {
    ensure_cuda();
    let input = cpu_f16(&[]).to(Device::Cuda(0)).expect("upload empty f16");
    let (values, inverse, counts) =
        unique_consecutive(&input).expect("unique_consecutive empty f16 cuda");
    assert_eq!(values.device(), Device::Cuda(0));
    assert_eq!(values.shape(), &[0]);
    assert!(read_f16(&values).is_empty());
    assert!(inverse.is_empty());
    assert!(counts.is_empty());
}
