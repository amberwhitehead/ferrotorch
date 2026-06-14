//! End-to-end CUDA `topk` coverage for f16 / bf16.
//!
//! These tests exercise the production `ferrotorch_core::topk` path: CUDA
//! tensors dispatch through `GpuBackend::topk_1d`, values remain GPU-resident
//! with their original half dtype, and only the int64 indices are decoded to
//! the public `Vec<usize>`.

#![cfg(feature = "cuda")]

use ferrotorch_core::topk;
use ferrotorch_core::{Device, Tensor, TensorStorage};
use ferrotorch_gpu::init_cuda_backend;
use half::{bf16, f16};

fn ensure_cuda() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        init_cuda_backend().expect("CUDA backend init");
    });
}

fn cpu_f16(data: &[f16], shape: &[usize], requires_grad: bool) -> Tensor<f16> {
    Tensor::from_storage(
        TensorStorage::cpu(data.to_vec()),
        shape.to_vec(),
        requires_grad,
    )
    .unwrap()
}

fn cpu_bf16(data: &[bf16], shape: &[usize], requires_grad: bool) -> Tensor<bf16> {
    Tensor::from_storage(
        TensorStorage::cpu(data.to_vec()),
        shape.to_vec(),
        requires_grad,
    )
    .unwrap()
}

#[test]
fn topk_f16_cuda_nan_inf_values_resident() {
    // Live PyTorch 2.11.0+cu130 CUDA oracle for torch.float16:
    // topk([3,nan,inf,5,-inf,nan], k=6, largest=True)
    //   values [nan,nan,inf,5,3,-inf], indices are int64 on CUDA.
    ensure_cuda();
    let row_f32 = [
        3.0_f32,
        f32::NAN,
        f32::INFINITY,
        5.0,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    let row: Vec<f16> = row_f32.into_iter().map(f16::from_f32).collect();
    let input = cpu_f16(&row, &[6], true)
        .to(Device::Cuda(0))
        .expect("upload f16")
        .detach()
        .requires_grad_(true);

    let (values, indices) = topk(&input, 6, true).expect("topk f16 cuda");
    assert_eq!(values.device(), Device::Cuda(0));
    assert_eq!(values.shape(), &[6]);
    assert!(
        values.requires_grad(),
        "topk values must keep the backward edge for f16 inputs"
    );

    let host = values
        .to(Device::Cpu)
        .expect("read values")
        .data()
        .expect("host values")
        .iter()
        .map(|v| v.to_f32())
        .collect::<Vec<_>>();
    assert!(host[0].is_nan() && host[1].is_nan(), "{host:?}");
    assert_eq!(host[2], f32::INFINITY);
    assert_eq!(host[3], 5.0);
    assert_eq!(host[4], 3.0);
    assert_eq!(host[5], f32::NEG_INFINITY);
    assert_eq!(indices.len(), 6);
    assert!(row[indices[0]].is_nan());
    assert!(row[indices[1]].is_nan());
    assert_eq!(&indices[2..], &[2, 3, 0, 4]);
}

#[test]
fn topk_bf16_cuda_multirow_smallest_resident() {
    ensure_cuda();
    let data_f32 = [4.0_f32, 1.0, 3.0, 2.0, 0.0, -1.0, 5.0, 6.0];
    let data: Vec<bf16> = data_f32.into_iter().map(bf16::from_f32).collect();
    let input = cpu_bf16(&data, &[2, 4], false)
        .to(Device::Cuda(0))
        .expect("upload bf16");

    let (values, indices) = topk(&input, 2, false).expect("topk bf16 cuda");
    assert_eq!(values.device(), Device::Cuda(0));
    assert_eq!(values.shape(), &[2, 2]);
    let host = values
        .to(Device::Cpu)
        .expect("read values")
        .data()
        .expect("host values")
        .iter()
        .map(|v| v.to_f32())
        .collect::<Vec<_>>();
    assert_eq!(host, vec![1.0, 2.0, -1.0, 0.0]);
    assert_eq!(indices, vec![1, 3, 1, 0]);
}
