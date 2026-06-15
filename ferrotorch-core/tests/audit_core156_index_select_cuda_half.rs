#![cfg(feature = "gpu")]

//! CUDA `index_select` half/bfloat parity probes.
//!
//! PyTorch 2.11 oracle:
//! - CUDA `torch.index_select` supports `float16` and `bfloat16` values.
//! - CUDA indices must live on the same device as the CUDA input.
//! - duplicate indices accumulate gradients in backward.
//! - 0-D index tensors are accepted and produce a length-1 selected axis.

use std::sync::Once;

use ferrotorch_core::autograd::graph::backward;
use ferrotorch_core::creation::from_vec;
use ferrotorch_core::device::Device;
use ferrotorch_core::error::FerrotorchError;
use ferrotorch_core::grad_fns::indexing::{index_select_1d, index_select_dim};
use ferrotorch_core::int_tensor::IntTensor;
use ferrotorch_core::tensor::Tensor;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for index_select half probes");
    });
}

fn cuda_idx(vals: &[i64], shape: &[usize]) -> IntTensor<i64> {
    IntTensor::<i64>::from_vec(vals.to_vec(), shape.to_vec())
        .expect("CPU IntTensor")
        .to(Device::Cuda(0))
        .expect("upload IntTensor")
}

fn cuda_f16(data: &[f32], shape: &[usize]) -> Tensor<half::f16> {
    let values = data.iter().copied().map(half::f16::from_f32).collect();
    from_vec::<half::f16>(values, shape)
        .expect("f16 CPU tensor")
        .to(Device::Cuda(0))
        .expect("upload f16")
        .requires_grad_(true)
}

fn cuda_bf16(data: &[f32], shape: &[usize]) -> Tensor<half::bf16> {
    let values = data.iter().copied().map(half::bf16::from_f32).collect();
    from_vec::<half::bf16>(values, shape)
        .expect("bf16 CPU tensor")
        .to(Device::Cuda(0))
        .expect("upload bf16")
        .requires_grad_(true)
}

fn host_f16(t: &Tensor<half::f16>) -> Vec<f32> {
    assert_eq!(
        t.device(),
        Device::Cuda(0),
        "tensor must stay CUDA-resident until explicit readback"
    );
    t.cpu()
        .expect("D2H f16")
        .data_vec()
        .expect("f16 data")
        .iter()
        .map(|v| v.to_f32())
        .collect()
}

fn host_bf16(t: &Tensor<half::bf16>) -> Vec<f32> {
    assert_eq!(
        t.device(),
        Device::Cuda(0),
        "tensor must stay CUDA-resident until explicit readback"
    );
    t.cpu()
        .expect("D2H bf16")
        .data_vec()
        .expect("bf16 data")
        .iter()
        .map(|v| v.to_f32())
        .collect()
}

#[test]
fn tensor_index_select_cuda_f16_backward_accumulates_duplicates() {
    ensure_cuda_backend();
    let x = cuda_f16(
        &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
        &[3, 4],
    );
    let out = x
        .index_select(0, &cuda_idx(&[2, 0, 2], &[3]))
        .expect("CUDA f16 index_select");

    assert_eq!(out.shape(), &[3, 4]);
    assert_eq!(
        host_f16(&out),
        vec![
            8.0, 9.0, 10.0, 11.0, 0.0, 1.0, 2.0, 3.0, 8.0, 9.0, 10.0, 11.0
        ]
    );

    backward(&out.sum_all().expect("f16 sum")).expect("f16 backward");

    let grad = x.grad().expect("grad read").expect("leaf grad");
    assert_eq!(
        host_f16(&grad),
        vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0]
    );
}

#[test]
fn tensor_index_select_cuda_bf16_backward_accumulates_duplicates() {
    ensure_cuda_backend();
    let x = cuda_bf16(
        &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
        &[3, 4],
    );
    let out = x
        .index_select(0, &cuda_idx(&[2, 0, 2], &[3]))
        .expect("CUDA bf16 index_select");

    assert_eq!(out.shape(), &[3, 4]);
    assert_eq!(
        host_bf16(&out),
        vec![
            8.0, 9.0, 10.0, 11.0, 0.0, 1.0, 2.0, 3.0, 8.0, 9.0, 10.0, 11.0
        ]
    );

    backward(&out.sum_all().expect("bf16 sum")).expect("bf16 backward");

    let grad = x.grad().expect("grad read").expect("leaf grad");
    assert_eq!(
        host_bf16(&grad),
        vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0]
    );
}

#[test]
fn grad_fn_index_select_dim_cuda_bf16_accepts_scalar_index() {
    ensure_cuda_backend();
    let x = cuda_bf16(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let idx = cuda_idx(&[2], &[]);
    let out = index_select_dim(&x, 1, &idx).expect("CUDA bf16 scalar index_select_dim");

    assert_eq!(out.shape(), &[2, 1]);
    assert_eq!(host_bf16(&out), vec![3.0, 6.0]);

    backward(&out.sum_all().expect("bf16 scalar sum")).expect("bf16 scalar backward");

    let grad = x.grad().expect("grad read").expect("leaf grad");
    assert_eq!(host_bf16(&grad), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn grad_fn_index_select_1d_cuda_f16_backward_accumulates_duplicates() {
    ensure_cuda_backend();
    let x = cuda_f16(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let out = index_select_1d(&x, &[2, 0, 2]).expect("CUDA f16 index_select_1d");

    assert_eq!(out.shape(), &[3]);
    assert_eq!(host_f16(&out), vec![3.0, 1.0, 3.0]);

    backward(&out.sum_all().expect("f16 1-D sum")).expect("f16 1-D backward");

    let grad = x.grad().expect("grad read").expect("leaf grad");
    assert_eq!(host_f16(&grad), vec![1.0, 0.0, 2.0, 0.0]);
}

#[test]
fn grad_fn_index_select_dim_rejects_mixed_cuda_cpu_index() {
    ensure_cuda_backend();
    let x = cuda_f16(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let cpu_idx = IntTensor::<i64>::from_vec(vec![0], vec![1]).expect("CPU index");

    match index_select_dim(&x, 0, &cpu_idx) {
        Err(FerrotorchError::DeviceMismatch { expected, got }) => {
            assert_eq!(expected, Device::Cuda(0));
            assert_eq!(got, Device::Cpu);
        }
        Err(other) => panic!("expected DeviceMismatch, got {other:?}"),
        Ok(_) => panic!("expected mixed-device index_select_dim to fail"),
    }
}
