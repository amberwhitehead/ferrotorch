#![cfg(feature = "gpu")]

//! CUDA index bounds validation without full index-buffer CPU round trips.
//!
//! PyTorch CUDA validates `index_select` / `gather` indices on device. This
//! audit pins Ferrotorch's resident validator contract: valid CUDA paths stay
//! CUDA-resident, invalid indices return a structured error before the
//! unchecked copy kernel launches, and the CUDA context remains usable after
//! an invalid-index error.

use std::sync::Once;

use ferrotorch_core::device::Device;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::int_tensor::{IntElement, IntTensor};
use ferrotorch_core::{Tensor, TensorStorage};

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for resident index bounds tests");
    });
}

fn cpu_tensor(values: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(values.to_vec()), shape.to_vec(), false)
        .expect("CPU Tensor")
}

fn cuda_tensor(values: &[f32], shape: &[usize]) -> Tensor<f32> {
    cpu_tensor(values, shape)
        .to(Device::Cuda(0))
        .expect("upload Tensor")
}

fn cuda_int<I: IntElement>(values: &[i64], shape: &[usize]) -> IntTensor<I> {
    let typed: Vec<I> = values
        .iter()
        .map(|&v| I::try_from_i64(v).expect("test index fits dtype"))
        .collect();
    IntTensor::<I>::from_vec(typed, shape.to_vec())
        .expect("CPU IntTensor")
        .to(Device::Cuda(0))
        .expect("upload IntTensor")
}

fn cuda_values_f32(t: &Tensor<f32>) -> Vec<f32> {
    assert_eq!(
        t.device(),
        Device::Cuda(0),
        "result must stay CUDA-resident"
    );
    t.to(Device::Cpu)
        .expect("assertion readback")
        .data_vec()
        .expect("Tensor data")
}

fn cuda_values_i64(t: &IntTensor<i64>) -> Vec<i64> {
    assert_eq!(
        t.device(),
        Device::Cuda(0),
        "result must stay CUDA-resident"
    );
    t.to(Device::Cpu)
        .expect("assertion readback")
        .data()
        .expect("IntTensor data")
        .to_vec()
}

fn error_message<T>(result: FerrotorchResult<T>) -> String {
    match result {
        Err(FerrotorchError::InvalidArgument { message }) => message,
        Err(other) => panic!("expected InvalidArgument, got {other:?}"),
        Ok(_) => panic!("expected operation to reject invalid index"),
    }
}

#[test]
fn tensor_index_select_i32_cuda_rejects_negative_and_context_survives() {
    ensure_cuda_backend();
    let x = cuda_tensor(&[10.0, 11.0, 12.0, 20.0, 21.0, 22.0], &[2, 3]);
    let bad = cuda_int::<i32>(&[0, -1], &[2]);

    let message = error_message(x.index_select(1, &bad));
    assert!(message.contains("index_select: index -1 is out of bounds"));
    assert!(message.contains("dimension 1 with size 3"));
    assert!(message.contains("flat index position 1"));

    let good = cuda_int::<i32>(&[2, 0], &[2]);
    let out = x
        .index_select(1, &good)
        .expect("valid index_select after invalid index");
    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(cuda_values_f32(&out), vec![12.0, 10.0, 22.0, 20.0]);
}

#[test]
fn tensor_gather_i64_cuda_rejects_high_index_and_context_survives() {
    ensure_cuda_backend();
    let x = cuda_tensor(&[10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0], &[2, 4]);
    let bad = cuda_int::<i64>(&[0, 1, 4, 2], &[2, 2]);

    let message = error_message(x.gather(1, &bad));
    assert!(message.contains("gather: index 4 is out of bounds"));
    assert!(message.contains("dimension 1 with size 4"));
    assert!(message.contains("flat index position 2"));

    let good = cuda_int::<i64>(&[3, 0, 1, 2], &[2, 2]);
    let out = x
        .gather(1, &good)
        .expect("valid gather after invalid index");
    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(cuda_values_f32(&out), vec![13.0, 10.0, 21.0, 22.0]);
}

#[test]
fn inttensor_gather_nd_i64_cuda_uses_resident_validator() {
    ensure_cuda_backend();
    let x = IntTensor::<i64>::from_vec(vec![1, 2, 3, 4, 5, 6], vec![2, 3])
        .expect("CPU IntTensor")
        .to(Device::Cuda(0))
        .expect("upload values");
    let index = cuda_int::<i64>(&[2, 0], &[1, 2]);

    let out = x.gather(1, &index).expect("rank-aware IntTensor gather");
    assert_eq!(out.shape(), &[1, 2]);
    assert_eq!(cuda_values_i64(&out), vec![3, 1]);
}

#[test]
fn inttensor_index_select_i64_cuda_rejects_high_index() {
    ensure_cuda_backend();
    let x = IntTensor::<i64>::from_vec(vec![1, 2, 3, 4, 5, 6], vec![2, 3])
        .expect("CPU IntTensor")
        .to(Device::Cuda(0))
        .expect("upload values");
    let bad = cuda_int::<i64>(&[0, 3], &[2]);

    let message = error_message(x.index_select(1, &bad));
    assert!(message.contains("index_select: index 3 is out of bounds"));
    assert!(message.contains("dimension 1 with size 3"));
    assert!(message.contains("flat index position 1"));
}

#[test]
fn tracked_index_select_cuda_backward_keeps_saved_index_resident() {
    ensure_cuda_backend();
    let x = cuda_tensor(&[10.0, 11.0, 12.0, 20.0, 21.0, 22.0], &[2, 3]).requires_grad_(true);
    let index = cuda_int::<i32>(&[2, 0], &[2]);

    let out = x
        .index_select(1, &index)
        .expect("tracked CUDA index_select");
    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(cuda_values_f32(&out), vec![12.0, 10.0, 22.0, 20.0]);

    let grad_output = cuda_tensor(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let grads = out
        .grad_fn()
        .expect("tracked index_select must attach grad_fn")
        .backward(&grad_output)
        .expect("CUDA index_select backward");
    let grad_input = grads[0].as_ref().expect("input grad");
    assert_eq!(grad_input.shape(), &[2, 3]);
    assert_eq!(
        cuda_values_f32(grad_input),
        vec![1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    );
}

#[test]
fn tracked_gather_cuda_backward_keeps_saved_index_resident() {
    ensure_cuda_backend();
    let x = cuda_tensor(&[10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0], &[2, 4])
        .requires_grad_(true);
    let index = cuda_int::<i64>(&[3, 0, 1, 2], &[2, 2]);

    let out = x.gather(1, &index).expect("tracked CUDA gather");
    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(cuda_values_f32(&out), vec![13.0, 10.0, 21.0, 22.0]);

    let grad_output = cuda_tensor(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let grads = out
        .grad_fn()
        .expect("tracked gather must attach grad_fn")
        .backward(&grad_output)
        .expect("CUDA gather backward");
    let grad_input = grads[0].as_ref().expect("input grad");
    assert_eq!(grad_input.shape(), &[2, 4]);
    assert_eq!(
        cuda_values_f32(grad_input),
        vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    );
}
