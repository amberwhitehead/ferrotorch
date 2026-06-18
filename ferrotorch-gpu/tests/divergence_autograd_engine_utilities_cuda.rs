#![cfg(feature = "cuda")]

use ferrotorch_core::autograd::checkpoint::checkpoint;
use ferrotorch_core::autograd::graph::{backward_parallel, backward_with_grad};
use ferrotorch_core::autograd::higher_order::{grad, hessian, jacobian};
use ferrotorch_core::grad_fns::arithmetic::{add, mul};
use ferrotorch_core::grad_fns::reduction::sum;
use ferrotorch_core::{
    Device, FerrotorchError, FerrotorchResult, Tensor, TensorStorage, manual_seed, rand_on_device,
};
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

fn cuda_checkpoint_rng_test_lock() -> std::sync::MutexGuard<'static, ()> {
    // These tests assert exact process-global CUDA RNG stream positions.
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().expect("CUDA checkpoint RNG test lock")
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
fn cuda_set_grad_rejects_cpu_grad_like_torch() {
    ensure_cuda();
    let x = cuda_leaf(&[2.0, 3.0], &[2]);
    let cpu_grad = cpu_f32(&[10.0, 20.0], &[2], false);

    let err = x
        .set_grad(Some(cpu_grad))
        .expect_err("CPU grad assignment to CUDA leaf must fail");

    assert!(
        matches!(err, FerrotorchError::DeviceMismatch { .. }),
        "expected device mismatch, got {err:?}"
    );
    assert!(
        x.grad().expect("grad lookup").is_none(),
        "rejected set_grad must not mutate CUDA leaf gradients"
    );
}

#[test]
fn cuda_set_grad_existing_cuda_grad_accumulates_on_device() {
    ensure_cuda();
    let x = cuda_leaf(&[2.0, 3.0], &[2]);
    let existing = cpu_f32(&[10.0, 20.0], &[2], false)
        .to(Device::Cuda(0))
        .expect("existing grad to cuda");
    x.set_grad(Some(existing)).expect("set cuda grad");

    let y = sum(&mul(&x, &x).expect("mul")).expect("sum");
    y.backward().expect("backward");

    let grad = x.grad().expect("grad lookup").expect("x grad");
    assert!(grad.is_cuda(), "accumulated grad must stay on CUDA");
    assert_eq!(host(&grad), &[14.0, 26.0]);
}

#[test]
fn cuda_checkpoint_backward_keeps_grad_on_cuda() {
    let _rng_lock = cuda_checkpoint_rng_test_lock();
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
fn cuda_checkpoint_preserves_rng_for_stochastic_recompute() {
    let _rng_lock = cuda_checkpoint_rng_test_lock();
    ensure_cuda();
    manual_seed(2026).unwrap();
    let x = cuda_leaf(&[1.0; 6], &[6]);

    let y = checkpoint(
        |input: &Tensor<f32>| -> FerrotorchResult<Tensor<f32>> {
            let mask = rand_on_device::<f32>(input.shape(), Device::Cuda(0))?;
            mul(input, &mask)
        },
        &x,
    )
    .expect("checkpoint");
    assert_eq!(y.device(), Device::Cuda(0));
    let forward_mask = host(&y);
    let after_forward = host(&rand_on_device::<f32>(&[4], Device::Cuda(0)).expect("after forward"));

    sum(&y).expect("sum").backward().expect("backward");

    let grad = x.grad().expect("grad lookup").expect("x grad");
    assert_eq!(grad.device(), Device::Cuda(0));
    assert_eq!(
        host(&grad),
        forward_mask,
        "CUDA checkpoint recompute must reuse the exact forward RNG mask"
    );
    let after_backward =
        host(&rand_on_device::<f32>(&[4], Device::Cuda(0)).expect("after backward"));

    manual_seed(2026).unwrap();
    let expected_forward = host(&rand_on_device::<f32>(&[6], Device::Cuda(0)).expect("forward"));
    let expected_after_forward =
        host(&rand_on_device::<f32>(&[4], Device::Cuda(0)).expect("expected after forward"));
    let expected_after_backward =
        host(&rand_on_device::<f32>(&[4], Device::Cuda(0)).expect("expected after backward"));

    assert_eq!(forward_mask, expected_forward);
    assert_eq!(after_forward, expected_after_forward);
    assert_eq!(
        after_backward, expected_after_backward,
        "CUDA checkpoint recompute must restore the caller RNG stream"
    );
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

#[test]
fn cuda_jacobian_square_stays_on_cuda_and_matches_torch() {
    ensure_cuda();
    let x = cuda_leaf(&[2.0, 3.0], &[2]);

    let jac = jacobian(
        |input: &Tensor<f32>| -> FerrotorchResult<Tensor<f32>> { mul(input, input) },
        &x,
    )
    .expect("jacobian");

    // torch 2.11.0+cu130:
    // torch.autograd.functional.jacobian(lambda z: z*z, tensor([2,3], cuda))
    // => device cuda:0, values [[4, 0], [0, 6]]
    assert_eq!(jac.device(), Device::Cuda(0));
    assert_eq!(jac.shape(), &[2, 2]);
    assert_eq!(host(&jac), &[4.0, 0.0, 0.0, 6.0]);
}

#[test]
fn cuda_jacobian_scalar_output_uses_pytorch_shape_on_cuda() {
    ensure_cuda();
    let x = cuda_leaf(&[2.0, 3.0], &[2]);

    let jac = jacobian(
        |input: &Tensor<f32>| -> FerrotorchResult<Tensor<f32>> {
            sum(&mul(input, input).expect("mul"))
        },
        &x,
    )
    .expect("jacobian");

    // torch 2.11.0+cu130:
    // torch.autograd.functional.jacobian(lambda z: (z*z).sum(), tensor([2,3], cuda))
    // => shape [2], device cuda:0, values [4, 6]
    assert_eq!(jac.device(), Device::Cuda(0));
    assert_eq!(jac.shape(), &[2]);
    assert_eq!(host(&jac), &[4.0, 6.0]);
}

#[test]
fn cuda_jacobian_noncontiguous_rank2_uses_output_plus_input_shape() {
    ensure_cuda();
    let base = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false)
        .to(Device::Cuda(0))
        .expect("to cuda");
    let x = base
        .transpose(0, 1)
        .expect("transpose")
        .requires_grad_(true);
    assert_eq!(x.shape(), &[3, 2]);
    assert!(
        !x.is_contiguous(),
        "probe must exercise non-contiguous CUDA input"
    );

    let jac = jacobian(
        |input: &Tensor<f32>| -> FerrotorchResult<Tensor<f32>> { mul(input, input) },
        &x,
    )
    .expect("jacobian");

    // torch 2.11.0+cu130 on base.reshape(2,3).t():
    // jacobian(lambda z: z*z, x).shape == [3, 2, 3, 2].
    // Logical input values are [1, 4, 2, 5, 3, 6], so the flattened
    // Jacobian is diagonal with [2, 8, 4, 10, 6, 12].
    assert_eq!(jac.device(), Device::Cuda(0));
    assert_eq!(jac.shape(), &[3, 2, 3, 2]);
    let got = host(&jac);
    let diag = [2.0, 8.0, 4.0, 10.0, 6.0, 12.0];
    let mut expected = vec![0.0f32; 36];
    for (i, &v) in diag.iter().enumerate() {
        expected[i * 6 + i] = v;
    }
    assert_eq!(got, expected);
}

#[test]
fn cuda_hessian_sum_square_stays_on_cuda_and_matches_torch() {
    ensure_cuda();
    let x = cuda_leaf(&[2.0, 3.0], &[2]);

    let hess = hessian(
        |input: &Tensor<f32>| -> FerrotorchResult<Tensor<f32>> {
            sum(&mul(input, input).expect("mul"))
        },
        &x,
    )
    .expect("hessian");

    // torch 2.11.0+cu130:
    // torch.autograd.functional.hessian(lambda z: (z*z).sum(), tensor([2,3], cuda))
    // => device cuda:0, values [[2, 0], [0, 2]]
    assert_eq!(hess.device(), Device::Cuda(0));
    assert_eq!(hess.shape(), &[2, 2]);
    assert_eq!(host(&hess), &[2.0, 0.0, 0.0, 2.0]);
}
