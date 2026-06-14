#![cfg(feature = "cuda")]

use ferrotorch_core::autograd::checkpoint::checkpoint;
use ferrotorch_core::autograd::graph::{backward_parallel, backward_with_grad};
use ferrotorch_core::autograd::higher_order::grad;
use ferrotorch_core::grad_fns::arithmetic::{add, mul};
use ferrotorch_core::grad_fns::reduction::sum;
use ferrotorch_core::{Device, FerrotorchError, FerrotorchResult, Tensor, TensorStorage};
use ferrotorch_gpu::init_cuda_backend;

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
    .expect("cpu tensor")
}

fn cuda_leaf(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    cpu_f32(data, shape, false)
        .to(Device::Cuda(0))
        .expect("to cuda")
        .requires_grad_(true)
}

fn host(t: &Tensor<f32>) -> Vec<f32> {
    t.cpu().expect("to cpu").data_vec().expect("host data")
}

#[test]
fn cuda_backward_with_grad_rejects_cpu_external_seed() {
    ensure_cuda();
    let x = cuda_leaf(&[2.0, 3.0], &[2]);
    let y = mul(&x, &x).expect("mul");
    let cpu_seed = cpu_f32(&[1.0, 1.0], &[2], false);

    let err = backward_with_grad(&y, Some(&cpu_seed)).expect_err("CPU seed must be rejected");

    assert!(
        matches!(err, FerrotorchError::DeviceMismatch { .. }),
        "expected device mismatch, got {err:?}"
    );
    assert!(
        x.grad().expect("grad lookup").is_none(),
        "rejected seed must not mutate CUDA leaf gradients"
    );
}

#[test]
fn cuda_backward_parallel_rejects_cpu_external_seed_before_fallback() {
    ensure_cuda();
    let x = cuda_leaf(&[2.0, 3.0], &[2]);
    let y = mul(&x, &x).expect("mul");
    let cpu_seed = cpu_f32(&[1.0, 1.0], &[2], false);

    let err = backward_parallel(&y, Some(&cpu_seed), 2).expect_err("CPU seed must be rejected");

    assert!(
        matches!(err, FerrotorchError::DeviceMismatch { .. }),
        "expected device mismatch, got {err:?}"
    );
    assert!(
        x.grad().expect("grad lookup").is_none(),
        "rejected seed must not mutate CUDA leaf gradients"
    );
}

#[test]
fn cuda_gradient_hook_rejects_cpu_replacement() {
    ensure_cuda();
    let x = cuda_leaf(&[2.0, 3.0], &[2]);
    let w = cpu_f32(&[4.0, 5.0], &[2], false)
        .to(Device::Cuda(0))
        .expect("w to cuda");
    x.register_hook(|_g| Some(cpu_f32(&[1.0, 1.0], &[2], false)))
        .expect("register hook");

    let y = sum(&mul(&x, &w).expect("mul")).expect("sum");
    let err = y.backward().expect_err("CPU hook replacement must fail");

    assert!(
        matches!(err, FerrotorchError::DeviceMismatch { .. }),
        "expected device mismatch, got {err:?}"
    );
    assert!(
        x.grad().expect("grad lookup").is_none(),
        "rejected hook replacement must not be accumulated on CUDA leaf"
    );
}

#[test]
fn cuda_checkpoint_backward_keeps_grad_on_cuda() {
    ensure_cuda();
    let x = cuda_leaf(&[2.0, 3.0], &[2]);
    let y = checkpoint(
        |input: &Tensor<f32>| -> FerrotorchResult<Tensor<f32>> { mul(input, input) },
        &x,
    )
    .expect("checkpoint");
    assert!(y.is_cuda());

    sum(&y).expect("sum").backward().expect("backward");

    let grad = x.grad().expect("grad lookup").expect("x grad");
    assert!(grad.is_cuda());
    assert_eq!(host(&grad), &[4.0, 6.0]);
}

#[test]
fn cuda_higher_order_grad_branch_accumulation_stays_on_cuda() {
    ensure_cuda();
    let x = cuda_leaf(&[2.0, 3.0], &[2]);
    let left = mul(&x, &x).expect("left");
    let right = mul(&x, &x).expect("right");
    let merged = add(&left, &right).expect("merge");
    let loss = sum(&merged).expect("sum");

    let grads = grad(&loss, &[&x], false, false).expect("grad");

    let gx = grads[0].as_ref().expect("x grad");
    assert!(gx.is_cuda());
    assert_eq!(host(gx), &[8.0, 12.0]);
}
