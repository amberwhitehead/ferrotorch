#![cfg(feature = "cuda")]

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

fn cpu_f16(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f16> {
    Tensor::from_storage(
        TensorStorage::cpu(data.iter().copied().map(f16::from_f32).collect()),
        shape.to_vec(),
        requires_grad,
    )
    .expect("cpu f16 tensor")
}

fn cpu_bf16(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<bf16> {
    Tensor::from_storage(
        TensorStorage::cpu(data.iter().copied().map(bf16::from_f32).collect()),
        shape.to_vec(),
        requires_grad,
    )
    .expect("cpu bf16 tensor")
}

fn cuda_leaf_f16(data: &[f32], shape: &[usize]) -> Tensor<f16> {
    cpu_f16(data, shape, false)
        .to(Device::Cuda(0))
        .expect("to cuda")
        .requires_grad_(true)
}

fn cuda_leaf_bf16(data: &[f32], shape: &[usize]) -> Tensor<bf16> {
    cpu_bf16(data, shape, false)
        .to(Device::Cuda(0))
        .expect("to cuda")
        .requires_grad_(true)
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
fn cuda_f16_leaf_and_nonleaf_grad_accum_use_f16_add_kernel() {
    ensure_cuda();
    let x = cuda_leaf_f16(&[1.0, 2.0, 3.0], &[3]);

    let y = x.add_t(&x).expect("x + x");
    let z = y.add_t(&y).expect("y + y");
    let loss = z.sum_all().expect("sum");
    backward(&loss).expect("backward");

    let grad = x.grad().expect("grad access").expect("grad");
    assert!(
        grad.is_cuda(),
        "f16 accumulated grad must stay CUDA-resident"
    );
    assert_eq!(host_f16_bits(&grad), f16_bits(&[4.0, 4.0, 4.0]));
}

#[test]
fn cuda_bf16_leaf_and_nonleaf_grad_accum_use_bf16_add_kernel() {
    ensure_cuda();
    let x = cuda_leaf_bf16(&[1.0, 2.0, 3.0], &[3]);

    let y = x.add_t(&x).expect("x + x");
    let z = y.add_t(&y).expect("y + y");
    let loss = z.sum_all().expect("sum");
    backward(&loss).expect("backward");

    let grad = x.grad().expect("grad access").expect("grad");
    assert!(
        grad.is_cuda(),
        "bf16 accumulated grad must stay CUDA-resident"
    );
    assert_eq!(host_bf16_bits(&grad), bf16_bits(&[4.0, 4.0, 4.0]));
}

#[test]
fn cuda_update_storage_f16_subview_writes_in_place_with_u16_scatter() {
    ensure_cuda();
    let base = cpu_f16(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false)
        .to(Device::Cuda(0))
        .expect("to cuda");
    let view = base.narrow(0, 1, 1).expect("row view");
    assert_eq!(view.storage_offset(), 3);

    let replacement = cpu_f16(&[9.0, 10.0, 11.0], &[1, 3], false)
        .to(Device::Cuda(0))
        .expect("replacement to cuda");
    let (storage, shape) = replacement
        .into_storage_and_shape()
        .expect("replacement storage");
    assert_eq!(shape, vec![1, 3]);

    unsafe {
        view.update_storage(storage)
            .expect("f16 CUDA sub-view update_storage");
    }

    assert_eq!(
        host_f16_bits(&base),
        f16_bits(&[1.0, 2.0, 3.0, 9.0, 10.0, 11.0])
    );
}

#[test]
fn cuda_update_storage_bf16_subview_writes_in_place_with_u16_scatter() {
    ensure_cuda();
    let base = cpu_bf16(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false)
        .to(Device::Cuda(0))
        .expect("to cuda");
    let view = base.narrow(0, 1, 1).expect("row view");
    assert_eq!(view.storage_offset(), 3);

    let replacement = cpu_bf16(&[9.0, 10.0, 11.0], &[1, 3], false)
        .to(Device::Cuda(0))
        .expect("replacement to cuda");
    let (storage, shape) = replacement
        .into_storage_and_shape()
        .expect("replacement storage");
    assert_eq!(shape, vec![1, 3]);

    unsafe {
        view.update_storage(storage)
            .expect("bf16 CUDA sub-view update_storage");
    }

    assert_eq!(
        host_bf16_bits(&base),
        bf16_bits(&[1.0, 2.0, 3.0, 9.0, 10.0, 11.0])
    );
}
