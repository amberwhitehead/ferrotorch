#![cfg(feature = "cuda")]

use ferrotorch_core::bool_tensor::BoolTensor;
use ferrotorch_core::grad_fns::indexing::{masked_fill, masked_scatter};
use ferrotorch_core::grad_fns::reduction::sum;
use ferrotorch_core::ops::indexing::{scatter, scatter_add, where_cond, where_cond_bt};
use ferrotorch_core::{Device, FerrotorchError, Tensor, TensorStorage};
use ferrotorch_gpu::init_cuda_backend;
use half::f16;

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

fn cpu_f16(data: &[f32], shape: &[usize]) -> Tensor<f16> {
    Tensor::from_storage(
        TensorStorage::cpu(data.iter().copied().map(f16::from_f32).collect()),
        shape.to_vec(),
        false,
    )
    .expect("cpu f16")
}

fn host_f64(t: &Tensor<f64>) -> Vec<f64> {
    t.cpu().expect("to cpu").data_vec().expect("host f64")
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

#[test]
fn cuda_host_mask_overload_uses_dtype_generic_f64_kernel_and_backpropagates() {
    ensure_cuda();
    let x = cpu_f64(&[0.0, 1.0, 2.0, 3.0], &[4], false)
        .to(Device::Cuda(0))
        .expect("to cuda")
        .requires_grad_(true);

    let y = masked_fill(&x, &[true, false, true, false], -7.25).expect("masked_fill");

    assert!(y.is_cuda());
    assert_eq!(host_f64(&y), vec![-7.25, 1.0, -7.25, 3.0]);

    sum(&y).expect("sum").backward().expect("backward");
    let grad = x.grad().expect("grad lookup").expect("x grad");
    assert!(grad.is_cuda());
    assert_eq!(host_f64(&grad), vec![0.0, 1.0, 0.0, 1.0]);
}

#[test]
fn cuda_host_mask_overload_uses_dtype_generic_f16_kernel() {
    ensure_cuda();
    let x = cpu_f16(&[0.0, 1.0, 2.0, 3.0], &[4])
        .to(Device::Cuda(0))
        .expect("to cuda");
    let fill = f16::from_f32(-2.5);

    let y = masked_fill(&x, &[false, true, true, false], fill).expect("masked_fill");

    assert!(y.is_cuda());
    assert_eq!(
        host_f16_bits(&y),
        vec![
            f16::from_f32(0.0).to_bits(),
            fill.to_bits(),
            fill.to_bits(),
            f16::from_f32(3.0).to_bits(),
        ]
    );
}

#[test]
fn cuda_booltensor_mask_must_match_input_device() {
    ensure_cuda();
    let x = cpu_f32(&[1.0, 2.0], &[2], false)
        .to(Device::Cuda(0))
        .expect("to cuda");
    let mask_cpu = BoolTensor::from_vec(vec![true, false], vec![2]).expect("mask");

    let err = x
        .masked_fill(&mask_cpu, -1.0)
        .expect_err("real mask tensor must be on input device");

    assert!(
        matches!(err, FerrotorchError::DeviceMismatch { .. }),
        "expected device mismatch, got {err:?}"
    );
}

#[test]
fn cuda_masked_scatter_f16_uses_resident_forward_kernel() {
    ensure_cuda();
    let x = cpu_f16(&[1.0, 2.0, 3.0, 4.0], &[4])
        .to(Device::Cuda(0))
        .expect("input to cuda");
    let source = cpu_f16(&[-1.0, -2.0], &[2])
        .to(Device::Cuda(0))
        .expect("source to cuda");
    let mask = BoolTensor::from_vec(vec![false, true, true, false], vec![4])
        .expect("mask")
        .to(Device::Cuda(0))
        .expect("mask to cuda");

    let out = masked_scatter(&x, &mask, &source).expect("masked_scatter");

    assert!(out.is_cuda());
    assert_eq!(
        host_f16_bits(&out),
        vec![
            f16::from_f32(1.0).to_bits(),
            f16::from_f32(-1.0).to_bits(),
            f16::from_f32(-2.0).to_bits(),
            f16::from_f32(4.0).to_bits(),
        ]
    );
}

#[test]
fn cuda_host_condition_overload_uses_dtype_generic_f64_where_and_backpropagates() {
    ensure_cuda();
    let x = cpu_f64(&[1.0, 2.0, 3.0, 4.0], &[4], false)
        .to(Device::Cuda(0))
        .expect("x to cuda")
        .requires_grad_(true);
    let y = cpu_f64(&[10.0, 20.0, 30.0, 40.0], &[4], false)
        .to(Device::Cuda(0))
        .expect("y to cuda")
        .requires_grad_(true);

    let out = where_cond(&[true, false, true, false], &x, &y).expect("where_cond");

    assert!(out.is_cuda());
    assert_eq!(host_f64(&out), vec![1.0, 20.0, 3.0, 40.0]);

    sum(&out).expect("sum").backward().expect("backward");
    assert_eq!(
        host_f64(&x.grad().expect("x grad lookup").expect("x grad")),
        vec![1.0, 0.0, 1.0, 0.0]
    );
    assert_eq!(
        host_f64(&y.grad().expect("y grad lookup").expect("y grad")),
        vec![0.0, 1.0, 0.0, 1.0]
    );
}

#[test]
fn cuda_host_condition_overload_uses_dtype_generic_f16_where() {
    ensure_cuda();
    let x = cpu_f16(&[1.0, 2.0, 3.0, 4.0], &[4])
        .to(Device::Cuda(0))
        .expect("x to cuda");
    let y = cpu_f16(&[10.0, 20.0, 30.0, 40.0], &[4])
        .to(Device::Cuda(0))
        .expect("y to cuda");

    let out = where_cond(&[false, true, true, false], &x, &y).expect("where_cond");

    assert!(out.is_cuda());
    assert_eq!(
        host_f16_bits(&out),
        vec![
            f16::from_f32(10.0).to_bits(),
            f16::from_f32(2.0).to_bits(),
            f16::from_f32(3.0).to_bits(),
            f16::from_f32(40.0).to_bits(),
        ]
    );
}

#[test]
fn tensor_where_t_host_mask_on_cuda_stays_resident_and_backpropagates() {
    ensure_cuda();
    let x = cpu_f64(&[1.0, 2.0, 3.0, 4.0], &[4], false)
        .to(Device::Cuda(0))
        .expect("x to cuda")
        .requires_grad_(true);
    let y = cpu_f64(&[10.0, 20.0, 30.0, 40.0], &[4], false)
        .to(Device::Cuda(0))
        .expect("y to cuda")
        .requires_grad_(true);

    let out = x
        .where_t(&[true, false, true, false], &y)
        .expect("Tensor::where_t");

    assert!(
        out.is_cuda(),
        "Tensor::where_t result must stay CUDA-resident"
    );
    assert_eq!(host_f64(&out), vec![1.0, 20.0, 3.0, 40.0]);

    sum(&out).expect("sum").backward().expect("backward");
    let gx = x.grad().expect("x grad lookup").expect("x grad");
    let gy = y.grad().expect("y grad lookup").expect("y grad");
    assert!(gx.is_cuda(), "x grad must stay CUDA-resident");
    assert!(gy.is_cuda(), "y grad must stay CUDA-resident");
    assert_eq!(host_f64(&gx), vec![1.0, 0.0, 1.0, 0.0]);
    assert_eq!(host_f64(&gy), vec![0.0, 1.0, 0.0, 1.0]);
}

#[test]
fn tensor_where_t_host_mask_broadcast_cuda_reduces_grads_resident() {
    ensure_cuda();
    let x = cpu_f32(&[1.0, 2.0, 3.0], &[1, 3], false)
        .to(Device::Cuda(0))
        .expect("x to cuda")
        .requires_grad_(true);
    let y = cpu_f32(&[10.0, 20.0], &[2, 1], false)
        .to(Device::Cuda(0))
        .expect("y to cuda")
        .requires_grad_(true);

    let out = x
        .where_t(&[true, true, true, false, false, false], &y)
        .expect("Tensor::where_t broadcast");

    assert!(
        out.is_cuda(),
        "Tensor::where_t broadcast result must stay CUDA-resident"
    );
    assert_eq!(out.shape(), &[2, 3]);
    assert_eq!(host_f32(&out), vec![1.0, 2.0, 3.0, 20.0, 20.0, 20.0]);

    sum(&out).expect("sum").backward().expect("backward");
    let gx = x.grad().expect("x grad lookup").expect("x grad");
    let gy = y.grad().expect("y grad lookup").expect("y grad");
    assert!(gx.is_cuda(), "broadcast x grad must stay CUDA-resident");
    assert!(gy.is_cuda(), "broadcast y grad must stay CUDA-resident");
    assert_eq!(gx.shape(), &[1, 3]);
    assert_eq!(gy.shape(), &[2, 1]);
    assert_eq!(host_f32(&gx), vec![1.0, 1.0, 1.0]);
    assert_eq!(host_f32(&gy), vec![0.0, 3.0]);
}

#[test]
fn tensor_where_bt_broadcast_cuda_condition_stays_resident_and_reduces_grads() {
    ensure_cuda();
    let x = cpu_f32(&[1.0, 2.0, 3.0], &[1, 3], false)
        .to(Device::Cuda(0))
        .expect("x to cuda")
        .requires_grad_(true);
    let y = cpu_f32(&[10.0], &[], false)
        .to(Device::Cuda(0))
        .expect("y to cuda")
        .requires_grad_(true);
    let cond = BoolTensor::from_vec(vec![true, false], vec![2, 1])
        .expect("cond cpu")
        .to(Device::Cuda(0))
        .expect("cond to cuda");

    let out = x.where_bt_t(&cond, &y).expect("Tensor::where_bt_t");

    assert!(
        out.is_cuda(),
        "Tensor::where_bt_t broadcast result must stay CUDA-resident"
    );
    assert_eq!(out.shape(), &[2, 3]);
    assert_eq!(host_f32(&out), vec![1.0, 2.0, 3.0, 10.0, 10.0, 10.0]);

    sum(&out).expect("sum").backward().expect("backward");
    let gx = x.grad().expect("x grad lookup").expect("x grad");
    let gy = y.grad().expect("y grad lookup").expect("y grad");
    assert!(gx.is_cuda(), "broadcast x grad must stay CUDA-resident");
    assert!(gy.is_cuda(), "broadcast y grad must stay CUDA-resident");
    assert_eq!(gx.shape(), &[1, 3]);
    assert!(gy.shape().is_empty());
    assert_eq!(host_f32(&gx), vec![1.0, 1.0, 1.0]);
    assert_eq!(host_f32(&gy), vec![3.0]);
}

#[test]
fn cuda_where_booltensor_condition_must_match_operand_device() {
    ensure_cuda();
    let x = cpu_f32(&[1.0, 2.0], &[2], false)
        .to(Device::Cuda(0))
        .expect("x to cuda");
    let y = cpu_f32(&[3.0, 4.0], &[2], false)
        .to(Device::Cuda(0))
        .expect("y to cuda");
    let cond_cpu = BoolTensor::from_vec(vec![true, false], vec![2]).expect("condition");

    let err =
        where_cond_bt(&cond_cpu, &x, &y).expect_err("real condition must be on operand device");

    assert!(
        matches!(err, FerrotorchError::DeviceMismatch { .. }),
        "expected device mismatch, got {err:?}"
    );
}

#[test]
fn cuda_scatter_tensor_src_must_match_input_device() {
    ensure_cuda();
    let input = cpu_f32(&[0.0, 0.0, 0.0], &[3], false);
    let src = cpu_f32(&[1.0, 2.0], &[2], false)
        .to(Device::Cuda(0))
        .expect("src to cuda");

    let err = scatter(&input, 0, &[0, 2], &[2], &src)
        .expect_err("scatter must reject mixed CPU/CUDA input/src");

    assert!(
        matches!(err, FerrotorchError::DeviceMismatch { .. }),
        "expected device mismatch, got {err:?}"
    );
}

#[test]
fn cuda_scatter_add_tensor_src_must_match_input_device() {
    ensure_cuda();
    let input = cpu_f32(&[0.0, 0.0, 0.0], &[3], false);
    let src = cpu_f32(&[1.0, 2.0], &[2], false)
        .to(Device::Cuda(0))
        .expect("src to cuda");

    let err = scatter_add(&input, 0, &[0, 2], &[2], &src)
        .expect_err("scatter_add must reject mixed CPU/CUDA input/src");

    assert!(
        matches!(err, FerrotorchError::DeviceMismatch { .. }),
        "expected device mismatch, got {err:?}"
    );
}
