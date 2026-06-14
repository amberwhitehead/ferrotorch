//! Tensor view/materialization CUDA parity probes.
//!
//! PyTorch behavior this pins:
//! - `split` / `chunk` are view operations and work for f16/bf16/f64 on CUDA;
//! - non-contiguous f16/bf16 CUDA views read back in logical order;
//! - 2-byte CUDA concatenation dispatch distinguishes f16 from bf16 by dtype tag;
//! - memory-format materialization for half tensors stays resident and preserves
//!   logical values.

#![cfg(feature = "cuda")]

use ferrotorch_core::grad_fns::reduction::sum;
use ferrotorch_core::grad_fns::shape::cat;
use ferrotorch_core::{Device, MemoryFormat, Tensor, TensorStorage};
use ferrotorch_gpu::init_cuda_backend;
use half::{bf16, f16};
use std::sync::Arc;

fn ensure_cuda() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        init_cuda_backend().expect("CUDA backend init");
    });
}

fn cpu_f16(data: &[f32], shape: &[usize]) -> Tensor<f16> {
    Tensor::from_storage(
        TensorStorage::cpu(data.iter().copied().map(f16::from_f32).collect()),
        shape.to_vec(),
        false,
    )
    .expect("cpu f16 tensor")
}

fn cuda_leaf_f16(data: &[f32], shape: &[usize]) -> Tensor<f16> {
    cpu_f16(data, shape)
        .to(Device::Cuda(0))
        .expect("to cuda")
        .requires_grad_(true)
}

fn cpu_bf16(data: &[f32], shape: &[usize]) -> Tensor<bf16> {
    Tensor::from_storage(
        TensorStorage::cpu(data.iter().copied().map(bf16::from_f32).collect()),
        shape.to_vec(),
        false,
    )
    .expect("cpu bf16 tensor")
}

fn cpu_f64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .expect("cpu f64 tensor")
}

fn host_f16_bits(t: &Tensor<f16>) -> Vec<u16> {
    t.cpu()
        .expect("to cpu")
        .data()
        .expect("host f16")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn host_bf16_bits(t: &Tensor<bf16>) -> Vec<u16> {
    t.cpu()
        .expect("to cpu")
        .data()
        .expect("host bf16")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn host_f64(t: &Tensor<f64>) -> Vec<f64> {
    t.cpu().expect("to cpu").data().expect("host f64").to_vec()
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

fn shares_storage<T: ferrotorch_core::Float>(a: &Tensor<T>, b: &Tensor<T>) -> bool {
    Arc::ptr_eq(a.inner_storage_arc(), b.inner_storage_arc())
}

#[test]
fn cuda_split_f16_axis1_returns_views_and_reads_logical_order() {
    ensure_cuda();
    let data: Vec<f32> = (0..12).map(|v| v as f32).collect();
    let x = cpu_f16(&data, &[3, 4])
        .to(Device::Cuda(0))
        .expect("to cuda");

    let parts = x.split(&[1, 3], 1).expect("split f16");
    assert_eq!(parts.len(), 2);
    assert!(parts[0].is_cuda());
    assert!(shares_storage(&x, &parts[0]));
    assert!(shares_storage(&x, &parts[1]));
    assert_eq!(parts[0].shape(), &[3, 1]);
    assert_eq!(parts[1].shape(), &[3, 3]);

    assert_eq!(host_f16_bits(&parts[0]), f16_bits(&[0.0, 4.0, 8.0]));
    assert_eq!(
        host_f16_bits(&parts[1]),
        f16_bits(&[1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 9.0, 10.0, 11.0])
    );
}

#[test]
fn cuda_chunk_bf16_transpose_view_returns_views_and_reads_logical_order() {
    ensure_cuda();
    let data: Vec<f32> = (0..12).map(|v| v as f32).collect();
    let x = cpu_bf16(&data, &[3, 4])
        .to(Device::Cuda(0))
        .expect("to cuda");
    let xt = x.transpose(0, 1).expect("transpose");

    let chunks = xt.chunk(2, 0).expect("chunk bf16 transpose");
    assert_eq!(chunks.len(), 2);
    assert!(chunks[0].is_cuda());
    assert!(shares_storage(&xt, &chunks[0]));
    assert!(shares_storage(&xt, &chunks[1]));
    assert_eq!(chunks[0].shape(), &[2, 3]);
    assert_eq!(chunks[1].shape(), &[2, 3]);

    assert_eq!(
        host_bf16_bits(&chunks[0]),
        bf16_bits(&[0.0, 4.0, 8.0, 1.0, 5.0, 9.0])
    );
    assert_eq!(
        host_bf16_bits(&chunks[1]),
        bf16_bits(&[2.0, 6.0, 10.0, 3.0, 7.0, 11.0])
    );
}

#[test]
fn cuda_split_f64_transpose_view_returns_views_and_reads_logical_order() {
    ensure_cuda();
    let data: Vec<f64> = (0..12).map(|v| v as f64).collect();
    let x = cpu_f64(&data, &[3, 4])
        .to(Device::Cuda(0))
        .expect("to cuda");
    let xt = x.transpose(0, 1).expect("transpose");

    let parts = xt.split(&[1, 2], 1).expect("split f64 transpose");
    assert_eq!(parts.len(), 2);
    assert!(parts[0].is_cuda());
    assert!(shares_storage(&xt, &parts[0]));
    assert!(shares_storage(&xt, &parts[1]));
    assert_eq!(parts[0].shape(), &[4, 1]);
    assert_eq!(parts[1].shape(), &[4, 2]);

    assert_eq!(host_f64(&parts[0]), vec![0.0, 1.0, 2.0, 3.0]);
    assert_eq!(
        host_f64(&parts[1]),
        vec![4.0, 8.0, 5.0, 9.0, 6.0, 10.0, 7.0, 11.0]
    );
}

#[test]
fn cuda_cat_f16_uses_two_byte_path_without_bf16_mistagging() {
    ensure_cuda();
    let a = cpu_f16(&[1.0, 2.0], &[2])
        .to(Device::Cuda(0))
        .expect("a cuda");
    let b = cpu_f16(&[3.0, 4.0], &[2])
        .to(Device::Cuda(0))
        .expect("b cuda");

    let y = cat(&[a, b], 0).expect("cat f16");
    assert!(y.is_cuda());
    assert_eq!(host_f16_bits(&y), f16_bits(&[1.0, 2.0, 3.0, 4.0]));
}

#[test]
fn cuda_cat_f16_backward_splits_grad_without_host_fallback() {
    ensure_cuda();
    let a = cuda_leaf_f16(&[1.0, 2.0], &[2]);
    let b = cuda_leaf_f16(&[3.0, 4.0], &[2]);

    let y = cat(&[a.clone(), b.clone()], 0).expect("cat f16");
    let loss = sum(&y).expect("sum");
    loss.backward().expect("backward");

    let ga = a.grad().expect("grad result").expect("a grad");
    let gb = b.grad().expect("grad result").expect("b grad");
    assert!(ga.is_cuda());
    assert!(gb.is_cuda());
    assert_eq!(host_f16_bits(&ga), f16_bits(&[1.0, 1.0]));
    assert_eq!(host_f16_bits(&gb), f16_bits(&[1.0, 1.0]));
}

#[test]
fn cuda_channels_last_f16_materializes_and_reads_back_logical_values() {
    ensure_cuda();
    let data: Vec<f32> = (0..8).map(|v| v as f32).collect();
    let x = cpu_f16(&data, &[1, 2, 2, 2])
        .to(Device::Cuda(0))
        .expect("to cuda");

    let y = x
        .to_memory_format(MemoryFormat::ChannelsLast)
        .expect("channels_last f16");
    assert!(y.is_cuda());
    assert!(y.is_contiguous_for(MemoryFormat::ChannelsLast));
    assert_eq!(host_f16_bits(&y), f16_bits(&data));
}
