#![cfg(feature = "gpu")]

use std::sync::Arc;
use std::sync::Once;
use std::sync::atomic::{AtomicUsize, Ordering};

use ferrotorch_core::autograd::graph::backward_parallel;
use ferrotorch_core::grad_fns::arithmetic::mul;
use ferrotorch_core::{Device, FerrotorchError, FerrotorchResult, Tensor, TensorStorage};

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for CORE-038 external-gradient probes");
    });
}

fn cpu_leaf(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true)
        .expect("construct CPU leaf")
}

fn cpu_const(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .expect("construct CPU tensor")
}

fn cuda_leaf(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    ensure_cuda_backend();
    cpu_const(data, shape)
        .to(Device::Cuda(0))
        .expect("upload CUDA leaf")
        .detach()
        .requires_grad_(true)
}

fn cuda_const(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    ensure_cuda_backend();
    cpu_const(data, shape)
        .to(Device::Cuda(0))
        .expect("upload CUDA tensor")
}

fn assert_device_mismatch(err: FerrotorchError, expected: Device, got: Device) {
    match err {
        FerrotorchError::DeviceMismatch {
            expected: actual_expected,
            got: actual_got,
        } => {
            assert_eq!(actual_expected, expected);
            assert_eq!(actual_got, got);
        }
        other => panic!(
            "expected DeviceMismatch {{ expected: {expected:?}, got: {got:?} }}, got {other:?}"
        ),
    }
}

fn register_counting_hook(root: &Tensor<f32>) -> Arc<AtomicUsize> {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_hook = Arc::clone(&calls);
    root.register_hook(move |_grad| {
        calls_for_hook.fetch_add(1, Ordering::SeqCst);
        None
    })
    .expect("register root hook");
    calls
}

fn squared_root(input: &Tensor<f32>) -> FerrotorchResult<Tensor<f32>> {
    mul(input, input)
}

#[test]
fn cuda_root_rejects_cpu_external_gradient_before_hooks() {
    let x = cuda_leaf(&[2.0, 3.0], &[2]);
    let root = squared_root(&x).expect("x * x");
    let hook_calls = register_counting_hook(&root);
    let cpu_seed = cpu_const(&[1.0, 1.0], &[2]);

    let err = root
        .backward_with_gradient(&cpu_seed)
        .expect_err("CUDA root must reject CPU external gradient");

    assert_device_mismatch(err, Device::Cuda(0), Device::Cpu);
    assert_eq!(
        hook_calls.load(Ordering::SeqCst),
        0,
        "root hooks must not run after root-gradient device validation fails"
    );
    assert!(
        x.grad().expect("grad lookup").is_none(),
        "failed boundary validation must not reach leaf accumulation"
    );
}

#[test]
fn cpu_root_rejects_cuda_external_gradient_before_hooks() {
    let x = cpu_leaf(&[2.0, 3.0], &[2]);
    let root = squared_root(&x).expect("x * x");
    let hook_calls = register_counting_hook(&root);
    let cuda_seed = cuda_const(&[1.0, 1.0], &[2]);

    let err = root
        .backward_with_gradient(&cuda_seed)
        .expect_err("CPU root must reject CUDA external gradient");

    assert_device_mismatch(err, Device::Cpu, Device::Cuda(0));
    assert_eq!(
        hook_calls.load(Ordering::SeqCst),
        0,
        "root hooks must not run after root-gradient device validation fails"
    );
    assert!(
        x.grad().expect("grad lookup").is_none(),
        "failed boundary validation must not reach leaf accumulation"
    );
}

#[test]
fn parallel_backward_rejects_wrong_device_external_gradient_before_hooks() {
    let x = cuda_leaf(&[2.0, 3.0], &[2]);
    let root = squared_root(&x).expect("x * x");
    let hook_calls = register_counting_hook(&root);
    let cpu_seed = cpu_const(&[1.0, 1.0], &[2]);

    let err = backward_parallel(&root, Some(&cpu_seed), 4)
        .expect_err("parallel CUDA root must reject CPU external gradient");

    assert_device_mismatch(err, Device::Cuda(0), Device::Cpu);
    assert_eq!(
        hook_calls.load(Ordering::SeqCst),
        0,
        "parallel backward must validate root-gradient device before traversal"
    );
    assert!(
        x.grad().expect("grad lookup").is_none(),
        "failed boundary validation must not reach leaf accumulation"
    );
}

#[test]
fn cuda_root_accepts_cuda_external_gradient_and_keeps_leaf_grad_cuda() {
    let x = cuda_leaf(&[2.0, 3.0], &[2]);
    let root = squared_root(&x).expect("x * x");
    let cuda_seed = cuda_const(&[10.0, 20.0], &[2]);

    root.backward_with_gradient(&cuda_seed)
        .expect("same-device CUDA external gradient");

    let grad = x.grad().expect("grad lookup").expect("x grad");
    assert_eq!(grad.device(), Device::Cuda(0));
    assert_eq!(grad.shape(), &[2]);
    assert_eq!(
        grad.cpu().expect("read back CUDA grad").data_vec().unwrap(),
        vec![40.0, 120.0]
    );
}
