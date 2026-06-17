#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::Tensor;
use ferrotorch_core::bool_tensor::BoolTensor;
use ferrotorch_core::device::Device;
use ferrotorch_core::grad_fns::indexing::{MaskedFillBackward, masked_fill_bt};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::GradFn;
use half::{bf16, f16};

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for CORE-1905 regression tests");
    });
}

fn cuda_bool(data: &[bool], shape: &[usize]) -> BoolTensor {
    BoolTensor::from_slice(data, shape)
        .expect("cpu bool tensor")
        .to(Device::Cuda(0))
        .expect("upload bool tensor")
}

fn cuda_f32(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .expect("cpu f32 tensor")
        .to(Device::Cuda(0))
        .expect("upload f32 tensor")
        .requires_grad_(requires_grad)
}

fn cuda_f64(data: &[f64], shape: &[usize], requires_grad: bool) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .expect("cpu f64 tensor")
        .to(Device::Cuda(0))
        .expect("upload f64 tensor")
        .requires_grad_(requires_grad)
}

fn cuda_f16(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f16> {
    let data: Vec<f16> = data.iter().copied().map(f16::from_f32).collect();
    Tensor::from_storage(TensorStorage::cpu(data), shape.to_vec(), false)
        .expect("cpu f16 tensor")
        .to(Device::Cuda(0))
        .expect("upload f16 tensor")
        .requires_grad_(requires_grad)
}

fn cuda_bf16(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<bf16> {
    let data: Vec<bf16> = data.iter().copied().map(bf16::from_f32).collect();
    Tensor::from_storage(TensorStorage::cpu(data), shape.to_vec(), false)
        .expect("cpu bf16 tensor")
        .to(Device::Cuda(0))
        .expect("upload bf16 tensor")
        .requires_grad_(requires_grad)
}

fn read_f32(t: &Tensor<f32>) -> Vec<f32> {
    t.cpu().expect("download f32").data_vec().expect("read f32")
}

fn read_f64(t: &Tensor<f64>) -> Vec<f64> {
    t.cpu().expect("download f64").data_vec().expect("read f64")
}

fn read_f16_as_f32(t: &Tensor<f16>) -> Vec<f32> {
    t.cpu()
        .expect("download f16")
        .data_vec()
        .expect("read f16")
        .iter()
        .map(|v| v.to_f32())
        .collect()
}

fn read_bf16_as_f32(t: &Tensor<bf16>) -> Vec<f32> {
    t.cpu()
        .expect("download bf16")
        .data_vec()
        .expect("read bf16")
        .iter()
        .map(|v| v.to_f32())
        .collect()
}

#[test]
fn masked_fill_backward_accepts_cuda_external_gradient_view() {
    ensure_cuda_backend();

    let input = cuda_f32(&[1.0, 2.0, 3.0], &[3], true);
    let mask = cuda_bool(&[true, false, true], &[3]);
    let out = masked_fill_bt(&input, &mask, -9.0).expect("masked_fill_bt");

    let backing = cuda_f32(&[10.0, 20.0, 30.0, 40.0, 50.0], &[5], false);
    let grad_view = backing.narrow(0, 1, 3).expect("narrow external grad");
    assert!(grad_view.is_cuda());
    assert!(grad_view.is_contiguous());
    assert_eq!(grad_view.storage_offset(), 1);
    assert_eq!(grad_view.storage_len(), 5);
    assert_eq!(grad_view.numel(), 3);

    out.backward_with_gradient(&grad_view)
        .expect("masked_fill backward with CUDA view gradient");

    let grad = input.grad().expect("grad result").expect("leaf grad");
    assert!(
        grad.is_cuda(),
        "masked_fill leaf grad must stay CUDA-resident"
    );
    assert_eq!(read_f32(&grad), vec![0.0, 30.0, 0.0]);
}

#[test]
fn masked_fill_backward_cuda_dtype_parity_is_resident() {
    ensure_cuda_backend();
    let mask = cuda_bool(&[false, true, false, true], &[4]);

    let f32_node = MaskedFillBackward {
        input: cuda_f32(&[1.0, 2.0, 3.0, 4.0], &[4], true),
        mask: mask.clone(),
    };
    let f32_grad = f32_node
        .backward(&cuda_f32(&[1.0, 2.0, 3.0, 4.0], &[4], false))
        .expect("f32 backward")
        .into_iter()
        .next()
        .expect("f32 grad slot")
        .expect("f32 grad");
    assert!(f32_grad.is_cuda());
    assert_eq!(read_f32(&f32_grad), vec![1.0, 0.0, 3.0, 0.0]);

    let f64_node = MaskedFillBackward {
        input: cuda_f64(&[1.0, 2.0, 3.0, 4.0], &[4], true),
        mask: mask.clone(),
    };
    let f64_grad = f64_node
        .backward(&cuda_f64(&[1.0, 2.0, 3.0, 4.0], &[4], false))
        .expect("f64 backward")
        .into_iter()
        .next()
        .expect("f64 grad slot")
        .expect("f64 grad");
    assert!(f64_grad.is_cuda());
    assert_eq!(read_f64(&f64_grad), vec![1.0, 0.0, 3.0, 0.0]);

    let f16_node = MaskedFillBackward {
        input: cuda_f16(&[1.0, 2.0, 3.0, 4.0], &[4], true),
        mask: mask.clone(),
    };
    let f16_grad = f16_node
        .backward(&cuda_f16(&[1.0, 2.0, 3.0, 4.0], &[4], false))
        .expect("f16 backward")
        .into_iter()
        .next()
        .expect("f16 grad slot")
        .expect("f16 grad");
    assert!(f16_grad.is_cuda());
    assert_eq!(read_f16_as_f32(&f16_grad), vec![1.0, 0.0, 3.0, 0.0]);

    let bf16_node = MaskedFillBackward {
        input: cuda_bf16(&[1.0, 2.0, 3.0, 4.0], &[4], true),
        mask,
    };
    let bf16_grad = bf16_node
        .backward(&cuda_bf16(&[1.0, 2.0, 3.0, 4.0], &[4], false))
        .expect("bf16 backward")
        .into_iter()
        .next()
        .expect("bf16 grad slot")
        .expect("bf16 grad");
    assert!(bf16_grad.is_cuda());
    assert_eq!(read_bf16_as_f32(&bf16_grad), vec![1.0, 0.0, 3.0, 0.0]);
}
