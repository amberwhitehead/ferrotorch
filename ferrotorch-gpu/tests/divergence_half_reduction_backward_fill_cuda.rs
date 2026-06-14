//! Regression probes for CUDA half-precision reduction backward fills.
//!
//! PyTorch 2.11.0+cu130 oracle from this environment:
//! - `torch.float16` / `torch.bfloat16` CUDA tensors support `sum().backward()`
//!   and `mean().backward()` without host-side gradient construction.
//! - `sum` grad is all ones in the input dtype.
//! - `mean` grad is `1 / numel` rounded to the input dtype.

#![cfg(feature = "cuda")]

use std::sync::Once;

use ferrotorch_core::grad_fns::reduction::{mean, mean_dim, prod, prod_dim, sum, sum_dim};
use ferrotorch_core::{Device, Tensor, TensorStorage};
use ferrotorch_gpu::device::GpuDevice;
use ferrotorch_gpu::{bf16, f16, init_cuda_backend};
use half::{bf16 as Bf16, f16 as F16};

const LEN: usize = 6;
const SHAPE: &[usize] = &[2, 3];

fn ensure_cuda() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        init_cuda_backend().expect("CUDA backend init");
    });
}

fn device() -> GpuDevice {
    ensure_cuda();
    GpuDevice::new(0).expect("CUDA device 0")
}

fn cpu_f16(data: &[F16], shape: &[usize]) -> Tensor<F16> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .expect("cpu f16 tensor")
}

fn cpu_bf16(data: &[Bf16], shape: &[usize]) -> Tensor<Bf16> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .expect("cpu bf16 tensor")
}

fn cuda_leaf_f16() -> Tensor<F16> {
    let data = [1.0_f32, -2.0, 3.5, -4.25, 0.5, 8.0].map(F16::from_f32);
    cpu_f16(&data, SHAPE)
        .to(Device::Cuda(0))
        .expect("to cuda")
        .requires_grad_(true)
}

fn cuda_leaf_bf16() -> Tensor<Bf16> {
    let data = [1.0_f32, -2.0, 3.5, -4.25, 0.5, 8.0].map(Bf16::from_f32);
    cpu_bf16(&data, SHAPE)
        .to(Device::Cuda(0))
        .expect("to cuda")
        .requires_grad_(true)
}

fn cuda_leaf_f16_from_f32(data: &[f32]) -> Tensor<F16> {
    let data: Vec<F16> = data.iter().copied().map(F16::from_f32).collect();
    cpu_f16(&data, SHAPE)
        .to(Device::Cuda(0))
        .expect("to cuda")
        .requires_grad_(true)
}

fn cuda_leaf_bf16_from_f32(data: &[f32]) -> Tensor<Bf16> {
    let data: Vec<Bf16> = data.iter().copied().map(Bf16::from_f32).collect();
    cpu_bf16(&data, SHAPE)
        .to(Device::Cuda(0))
        .expect("to cuda")
        .requires_grad_(true)
}

fn host_f16_bits(t: &Tensor<F16>) -> Vec<u16> {
    t.cpu()
        .expect("cpu()")
        .data()
        .expect("data")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn f16_bits(data: &[f32]) -> Vec<u16> {
    data.iter()
        .copied()
        .map(F16::from_f32)
        .map(|v| v.to_bits())
        .collect()
}

fn bf16_bits(data: &[f32]) -> Vec<u16> {
    data.iter()
        .copied()
        .map(Bf16::from_f32)
        .map(|v| v.to_bits())
        .collect()
}

fn host_bf16_bits(t: &Tensor<Bf16>) -> Vec<u16> {
    t.cpu()
        .expect("cpu()")
        .data()
        .expect("data")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

#[test]
fn fill_f16_kernel_rounds_scalar_on_device() {
    let dev = device();
    let scalar = 1.0_f32 / 6.0_f32;
    let out = f16::gpu_fill_f16(LEN, scalar, &dev).expect("fill f16");
    let bits = dev.stream().clone_dtoh(&out).expect("read f16 bits");
    assert_eq!(bits, vec![F16::from_f32(scalar).to_bits(); LEN]);

    let empty = f16::gpu_fill_f16(0, scalar, &dev).expect("empty fill f16");
    assert_eq!(empty.len(), 0);
}

#[test]
fn fill_bf16_kernel_rounds_scalar_on_device() {
    let dev = device();
    let scalar = 1.0_f32 / 6.0_f32;
    let out = bf16::gpu_fill_bf16(LEN, scalar, &dev).expect("fill bf16");
    let bits = dev.stream().clone_dtoh(&out).expect("read bf16 bits");
    assert_eq!(bits, vec![Bf16::from_f32(scalar).to_bits(); LEN]);

    let empty = bf16::gpu_fill_bf16(0, scalar, &dev).expect("empty fill bf16");
    assert_eq!(empty.len(), 0);
}

#[test]
fn sum_backward_f16_and_bf16_grads_stay_cuda() {
    ensure_cuda();

    let x16 = cuda_leaf_f16();
    sum(&x16)
        .expect("sum f16")
        .backward()
        .expect("backward f16");
    let g16 = x16.grad().expect("grad result").expect("grad f16");
    assert!(g16.is_cuda(), "f16 sum grad must stay CUDA-resident");
    assert_eq!(host_f16_bits(&g16), vec![F16::from_f32(1.0).to_bits(); LEN]);

    let xb = cuda_leaf_bf16();
    sum(&xb)
        .expect("sum bf16")
        .backward()
        .expect("backward bf16");
    let gb = xb.grad().expect("grad result").expect("grad bf16");
    assert!(gb.is_cuda(), "bf16 sum grad must stay CUDA-resident");
    assert_eq!(
        host_bf16_bits(&gb),
        vec![Bf16::from_f32(1.0).to_bits(); LEN]
    );
}

#[test]
fn mean_backward_f16_and_bf16_grads_stay_cuda() {
    ensure_cuda();
    let scalar = 1.0_f32 / LEN as f32;

    let x16 = cuda_leaf_f16();
    mean(&x16)
        .expect("mean f16")
        .backward()
        .expect("backward f16");
    let g16 = x16.grad().expect("grad result").expect("grad f16");
    assert!(g16.is_cuda(), "f16 mean grad must stay CUDA-resident");
    assert_eq!(
        host_f16_bits(&g16),
        vec![F16::from_f32(scalar).to_bits(); LEN]
    );

    let xb = cuda_leaf_bf16();
    mean(&xb)
        .expect("mean bf16")
        .backward()
        .expect("backward bf16");
    let gb = xb.grad().expect("grad result").expect("grad bf16");
    assert!(gb.is_cuda(), "bf16 mean grad must stay CUDA-resident");
    assert_eq!(
        host_bf16_bits(&gb),
        vec![Bf16::from_f32(scalar).to_bits(); LEN]
    );
}

#[test]
fn sum_dim_backward_f16_and_bf16_grads_stay_cuda() {
    ensure_cuda();

    let x16 = cuda_leaf_f16();
    let y16 = sum_dim(&x16, 1, false).expect("sum_dim f16");
    sum(&y16)
        .expect("loss f16")
        .backward()
        .expect("backward f16");
    let g16 = x16.grad().expect("grad result").expect("grad f16");
    assert!(g16.is_cuda(), "f16 sum_dim grad must stay CUDA-resident");
    assert_eq!(host_f16_bits(&g16), vec![F16::from_f32(1.0).to_bits(); LEN]);

    let xb = cuda_leaf_bf16();
    let yb = sum_dim(&xb, 1, false).expect("sum_dim bf16");
    sum(&yb)
        .expect("loss bf16")
        .backward()
        .expect("backward bf16");
    let gb = xb.grad().expect("grad result").expect("grad bf16");
    assert!(gb.is_cuda(), "bf16 sum_dim grad must stay CUDA-resident");
    assert_eq!(
        host_bf16_bits(&gb),
        vec![Bf16::from_f32(1.0).to_bits(); LEN]
    );
}

#[test]
fn mean_dim_backward_f16_and_bf16_grads_stay_cuda() {
    ensure_cuda();
    let scalar = 1.0_f32 / 3.0_f32;

    let x16 = cuda_leaf_f16();
    let y16 = mean_dim(&x16, 1, false).expect("mean_dim f16");
    sum(&y16)
        .expect("loss f16")
        .backward()
        .expect("backward f16");
    let g16 = x16.grad().expect("grad result").expect("grad f16");
    assert!(g16.is_cuda(), "f16 mean_dim grad must stay CUDA-resident");
    assert_eq!(
        host_f16_bits(&g16),
        vec![F16::from_f32(scalar).to_bits(); LEN]
    );

    let xb = cuda_leaf_bf16();
    let yb = mean_dim(&xb, 1, false).expect("mean_dim bf16");
    sum(&yb)
        .expect("loss bf16")
        .backward()
        .expect("backward bf16");
    let gb = xb.grad().expect("grad result").expect("grad bf16");
    assert!(gb.is_cuda(), "bf16 mean_dim grad must stay CUDA-resident");
    assert_eq!(
        host_bf16_bits(&gb),
        vec![Bf16::from_f32(scalar).to_bits(); LEN]
    );
}

#[test]
fn prod_backward_f16_and_bf16_grads_stay_cuda() {
    ensure_cuda();
    let expected_grad = [119.0_f32, -59.5, 34.0, -28.0, 238.0, 14.875];

    let x16 = cuda_leaf_f16();
    let y16 = prod(&x16).expect("prod f16");
    assert!(y16.is_cuda(), "f16 prod output must stay CUDA-resident");
    assert_eq!(host_f16_bits(&y16), f16_bits(&[119.0]));
    y16.backward().expect("backward f16");
    let g16 = x16.grad().expect("grad result").expect("grad f16");
    assert!(g16.is_cuda(), "f16 prod grad must stay CUDA-resident");
    assert_eq!(host_f16_bits(&g16), f16_bits(&expected_grad));

    let xb = cuda_leaf_bf16();
    let yb = prod(&xb).expect("prod bf16");
    assert!(yb.is_cuda(), "bf16 prod output must stay CUDA-resident");
    assert_eq!(host_bf16_bits(&yb), bf16_bits(&[119.0]));
    yb.backward().expect("backward bf16");
    let gb = xb.grad().expect("grad result").expect("grad bf16");
    assert!(gb.is_cuda(), "bf16 prod grad must stay CUDA-resident");
    assert_eq!(host_bf16_bits(&gb), bf16_bits(&expected_grad));
}

#[test]
fn prod_dim_backward_f16_and_bf16_handles_zero_slices_on_cuda() {
    ensure_cuda();
    let values = [0.0_f32, 2.0, 3.0, 0.0, 0.0, 4.0];
    let expected_forward = [0.0_f32, 0.0];
    let expected_grad = [6.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0];

    let x16 = cuda_leaf_f16_from_f32(&values);
    let y16 = prod_dim(&x16, 1, false).expect("prod_dim f16");
    assert!(y16.is_cuda(), "f16 prod_dim output must stay CUDA-resident");
    assert_eq!(host_f16_bits(&y16), f16_bits(&expected_forward));
    sum(&y16)
        .expect("loss f16")
        .backward()
        .expect("backward f16");
    let g16 = x16.grad().expect("grad result").expect("grad f16");
    assert!(g16.is_cuda(), "f16 prod_dim grad must stay CUDA-resident");
    assert_eq!(host_f16_bits(&g16), f16_bits(&expected_grad));

    let xb = cuda_leaf_bf16_from_f32(&values);
    let yb = prod_dim(&xb, 1, false).expect("prod_dim bf16");
    assert!(yb.is_cuda(), "bf16 prod_dim output must stay CUDA-resident");
    assert_eq!(host_bf16_bits(&yb), bf16_bits(&expected_forward));
    sum(&yb)
        .expect("loss bf16")
        .backward()
        .expect("backward bf16");
    let gb = xb.grad().expect("grad result").expect("grad bf16");
    assert!(gb.is_cuda(), "bf16 prod_dim grad must stay CUDA-resident");
    assert_eq!(host_bf16_bits(&gb), bf16_bits(&expected_grad));
}

#[test]
fn prod_dim_backward_f16_and_bf16_nonzero_slices_on_cuda() {
    ensure_cuda();
    let expected_forward = [-7.0_f32, -17.0];
    let expected_grad = [-7.0_f32, 3.5, -2.0, 4.0, -34.0, -2.125];

    let x16 = cuda_leaf_f16();
    let y16 = prod_dim(&x16, 1, false).expect("prod_dim f16");
    assert!(y16.is_cuda(), "f16 prod_dim output must stay CUDA-resident");
    assert_eq!(host_f16_bits(&y16), f16_bits(&expected_forward));
    sum(&y16)
        .expect("loss f16")
        .backward()
        .expect("backward f16");
    let g16 = x16.grad().expect("grad result").expect("grad f16");
    assert!(g16.is_cuda(), "f16 prod_dim grad must stay CUDA-resident");
    assert_eq!(host_f16_bits(&g16), f16_bits(&expected_grad));

    let xb = cuda_leaf_bf16();
    let yb = prod_dim(&xb, 1, false).expect("prod_dim bf16");
    assert!(yb.is_cuda(), "bf16 prod_dim output must stay CUDA-resident");
    assert_eq!(host_bf16_bits(&yb), bf16_bits(&expected_forward));
    sum(&yb)
        .expect("loss bf16")
        .backward()
        .expect("backward bf16");
    let gb = xb.grad().expect("grad result").expect("grad bf16");
    assert!(gb.is_cuda(), "bf16 prod_dim grad must stay CUDA-resident");
    assert_eq!(host_bf16_bits(&gb), bf16_bits(&expected_grad));
}
