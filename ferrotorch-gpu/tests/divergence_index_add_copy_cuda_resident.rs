#![cfg(feature = "cuda")]

use ferrotorch_core::autograd::graph::backward;
use ferrotorch_core::grad_fns::indexing::{index_add, index_copy};
use ferrotorch_core::int_tensor::IntTensor;
use ferrotorch_core::{Device, Tensor, TensorStorage};
use ferrotorch_gpu::init_cuda_backend;
use std::sync::Once;

static INIT: Once = Once::new();

fn ensure_cuda() {
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
    .expect("cpu f32")
}

fn cpu_f64(data: &[f64], shape: &[usize], requires_grad: bool) -> Tensor<f64> {
    Tensor::from_storage(
        TensorStorage::cpu(data.to_vec()),
        shape.to_vec(),
        requires_grad,
    )
    .expect("cpu f64")
}

fn cuda_f32(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
    cpu_f32(data, shape, false)
        .to(Device::Cuda(0))
        .expect("to cuda")
        .requires_grad_(requires_grad)
}

fn cuda_f64(data: &[f64], shape: &[usize], requires_grad: bool) -> Tensor<f64> {
    cpu_f64(data, shape, false)
        .to(Device::Cuda(0))
        .expect("to cuda")
        .requires_grad_(requires_grad)
}

fn cuda_idx(data: &[i64], shape: &[usize]) -> IntTensor<i64> {
    IntTensor::from_vec(data.to_vec(), shape.to_vec())
        .expect("index")
        .to(Device::Cuda(0))
        .expect("index to cuda")
}

fn host_f32(t: &Tensor<f32>) -> Vec<f32> {
    t.cpu().expect("cpu").data_vec().expect("data").to_vec()
}

fn host_f64(t: &Tensor<f64>) -> Vec<f64> {
    t.cpu().expect("cpu").data_vec().expect("data").to_vec()
}

#[test]
fn cuda_index_add_f32_forward_backward_alpha_stays_resident() {
    ensure_cuda();
    let input = cuda_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
    let source = cuda_f32(&[10.0, 20.0, 30.0, 40.0], &[2, 2], true);
    let index = cuda_idx(&[2, 0], &[2]);

    let out = index_add(&input, 1, &index, &source, 2.5).expect("index_add cuda");
    assert!(out.is_cuda(), "forward must stay CUDA-resident");
    assert_eq!(host_f32(&out), vec![51.0, 2.0, 28.0, 104.0, 5.0, 81.0]);

    backward(&out.sum_all().expect("sum")).expect("index_add backward");
    let gi = input.grad().expect("grad access").expect("input grad");
    let gs = source.grad().expect("grad access").expect("source grad");
    assert!(
        gi.is_cuda() && gs.is_cuda(),
        "grads must stay CUDA-resident"
    );
    assert_eq!(host_f32(&gi), vec![1.0; 6]);
    assert_eq!(host_f32(&gs), vec![2.5; 4]);
}

#[test]
fn cuda_index_copy_f64_forward_backward_stays_resident() {
    ensure_cuda();
    let input = cuda_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
    let source = cuda_f64(&[10.0, 20.0, 30.0, 40.0], &[2, 2], true);
    let index = cuda_idx(&[2, 0], &[2]);

    let out = index_copy(&input, 1, &index, &source).expect("index_copy cuda");
    assert!(out.is_cuda(), "forward must stay CUDA-resident");
    assert_eq!(host_f64(&out), vec![20.0, 2.0, 10.0, 40.0, 5.0, 30.0]);

    backward(&out.sum_all().expect("sum")).expect("index_copy backward");
    let gi = input.grad().expect("grad access").expect("input grad");
    let gs = source.grad().expect("grad access").expect("source grad");
    assert!(
        gi.is_cuda() && gs.is_cuda(),
        "grads must stay CUDA-resident"
    );
    assert_eq!(host_f64(&gi), vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0]);
    assert_eq!(host_f64(&gs), vec![1.0; 4]);
}

#[test]
fn cuda_index_add_duplicate_indices_accumulate_and_empty_index_clones() {
    ensure_cuda();
    let input = cuda_f32(&[1.0, 2.0, 3.0], &[1, 3], false);
    let source = cuda_f32(&[10.0, 20.0], &[1, 2], false);
    let index = cuda_idx(&[1, 1], &[2]);

    let out = index_add(&input, 1, &index, &source, 1.0).expect("duplicate add");
    assert!(out.is_cuda());
    assert_eq!(host_f32(&out), vec![1.0, 32.0, 3.0]);

    let empty_index = cuda_idx(&[], &[0]);
    let empty_source = cuda_f32(&[], &[1, 0], false);
    let cloned = index_copy(&input, 1, &empty_index, &empty_source).expect("empty copy");
    assert!(cloned.is_cuda(), "empty index must not demote to CPU");
    assert_eq!(host_f32(&cloned), vec![1.0, 2.0, 3.0]);
}

#[test]
fn cuda_index_add_view_input_uses_logical_values_on_device() {
    ensure_cuda();
    let base = cuda_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
    let input = base.transpose(0, 1).expect("transpose"); // [[1,4],[2,5],[3,6]]
    assert!(input.is_cuda());
    assert!(!input.is_contiguous(), "probe must use a strided CUDA view");
    let source = cuda_f32(&[10.0, 20.0, 30.0], &[3, 1], false);
    let index = cuda_idx(&[1], &[1]);

    let out = index_add(&input, 1, &index, &source, 2.0).expect("view index_add");
    assert!(out.is_cuda());
    assert_eq!(host_f32(&out), vec![1.0, 24.0, 2.0, 45.0, 3.0, 66.0]);
}
