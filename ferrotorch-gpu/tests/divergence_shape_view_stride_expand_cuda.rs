//! CUDA shape/view/stride probes.
//!
//! These pin PyTorch-style metadata views on CUDA: `expand` must share GPU
//! storage and carry zero strides, and CUDA materialization of 2-byte float
//! strided views must use the device u16 strided-copy path.

#![cfg(feature = "cuda")]

use ferrotorch_core::grad_fns::shape::expand;
use ferrotorch_core::{Device, Tensor, TensorStorage, backward};
use ferrotorch_gpu::init_cuda_backend;
use half::{bf16, f16};

fn ensure_cuda() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        init_cuda_backend().expect("CUDA backend init");
    });
}

fn cpu_f32(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
    Tensor::from_storage(
        TensorStorage::cpu(data.to_vec()),
        shape.to_vec(),
        requires_grad,
    )
    .expect("cpu f32 tensor")
}

fn cpu_f16(data: &[f32], shape: &[usize]) -> Tensor<f16> {
    Tensor::from_storage(
        TensorStorage::cpu(data.iter().copied().map(f16::from_f32).collect()),
        shape.to_vec(),
        false,
    )
    .expect("cpu f16 tensor")
}

fn cpu_bf16(data: &[f32], shape: &[usize]) -> Tensor<bf16> {
    Tensor::from_storage(
        TensorStorage::cpu(data.iter().copied().map(bf16::from_f32).collect()),
        shape.to_vec(),
        false,
    )
    .expect("cpu bf16 tensor")
}

fn host_f32(t: &Tensor<f32>) -> Vec<f32> {
    t.cpu().expect("to cpu").data_vec().expect("host f32")
}

fn host_f16_bits(t: &Tensor<f16>) -> Vec<u16> {
    t.cpu()
        .expect("to cpu")
        .data_vec()
        .expect("host f16")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn host_bf16_bits(t: &Tensor<bf16>) -> Vec<u16> {
    t.cpu()
        .expect("to cpu")
        .data_vec()
        .expect("host bf16")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn f16_bits(data: &[f32]) -> Vec<u16> {
    data.iter()
        .copied()
        .map(f16::from_f32)
        .map(|v| v.to_bits())
        .collect()
}

fn bf16_bits(data: &[f32]) -> Vec<u16> {
    data.iter()
        .copied()
        .map(bf16::from_f32)
        .map(|v| v.to_bits())
        .collect()
}

#[test]
fn cuda_expand_f32_is_zero_stride_view_not_materialized_copy() {
    ensure_cuda();
    let x = cpu_f32(&[1.0, 2.0, 3.0], &[1, 3], false)
        .to(Device::Cuda(0))
        .expect("to cuda");
    let y = expand(&x, &[2, 3]).expect("expand");

    assert!(y.is_cuda());
    assert_eq!(y.shape(), &[2, 3]);
    assert_eq!(y.strides(), &[0isize, 1]);
    assert_eq!(y.storage_len(), x.storage_len());
    assert_eq!(host_f32(&y), &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
}

#[test]
fn cuda_expand_f32_backward_sums_without_host_fallback() {
    ensure_cuda();
    let x = cpu_f32(&[1.0, 2.0, 3.0], &[1, 3], false)
        .to(Device::Cuda(0))
        .expect("to cuda")
        .requires_grad_(true);
    let y = expand(&x, &[2, 3]).expect("expand");
    let loss = y
        .contiguous()
        .expect("materialize view")
        .sum_all()
        .expect("sum");

    backward(&loss).expect("backward");
    let grad = x.grad().expect("grad access").expect("grad");
    assert!(grad.is_cuda());
    assert_eq!(host_f32(&grad), &[2.0, 2.0, 2.0]);
}

#[test]
fn cuda_as_strided_copy_f16_uses_u16_strided_copy_path() {
    ensure_cuda();
    let x = cpu_f16(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5])
        .to(Device::Cuda(0))
        .expect("to cuda");

    let y = x
        .as_strided_copy(&[3, 3], &[1, 1], Some(0))
        .expect("f16 as_strided_copy");
    assert!(y.is_cuda());
    assert!(y.is_contiguous());
    assert_eq!(
        host_f16_bits(&y),
        f16_bits(&[1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0])
    );
}

#[test]
fn cuda_as_strided_copy_bf16_uses_u16_strided_copy_path_with_offset() {
    ensure_cuda();
    let x = cpu_bf16(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[6])
        .to(Device::Cuda(0))
        .expect("to cuda");

    let y = x
        .as_strided_copy(&[2, 2], &[2, 1], Some(1))
        .expect("bf16 as_strided_copy");
    assert!(y.is_cuda());
    assert!(y.is_contiguous());
    assert_eq!(host_bf16_bits(&y), bf16_bits(&[20.0, 30.0, 40.0, 50.0]));
}
