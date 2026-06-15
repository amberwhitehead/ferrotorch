//! CORE-020 (#1714) — `add_out` / `add_scaled_out` must reject tracking
//! source tensors under grad mode.
//!
//! PyTorch raises
//! `RuntimeError: functions with out=... arguments don't support automatic
//! differentiation, but one of the arguments requires grad` when any source
//! argument participates in autograd. The write is legal inside `no_grad`
//! because no graph edge is requested.

use ferrotorch_core::autograd::no_grad::no_grad;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::grad_fns::arithmetic::{add, add_out, add_scaled_out};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

fn cpu_tensor(data: Vec<f32>, shape: Vec<usize>, requires_grad: bool) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data), shape, requires_grad).unwrap()
}

fn assert_out_autograd_error(result: FerrotorchResult<()>) {
    let err = result.expect_err("out= with tracking source input must error");
    match &err {
        FerrotorchError::InvalidArgument { message } => assert!(
            message.contains("out=... arguments don't support automatic differentiation"),
            "unexpected InvalidArgument message: {message}"
        ),
        other => panic!("expected InvalidArgument, got {other:?}"),
    }
}

#[test]
fn add_out_rejects_left_leaf_requires_grad() {
    let out = cpu_tensor(vec![0.0, 0.0], vec![2], false);
    let a = cpu_tensor(vec![1.0, 2.0], vec![2], true);
    let b = cpu_tensor(vec![10.0, 20.0], vec![2], false);

    assert_out_autograd_error(add_out(&out, &a, &b));
    assert_eq!(out.data_vec().unwrap(), &[0.0, 0.0]);
}

#[test]
fn add_out_rejects_right_leaf_requires_grad() {
    let out = cpu_tensor(vec![0.0, 0.0], vec![2], false);
    let a = cpu_tensor(vec![1.0, 2.0], vec![2], false);
    let b = cpu_tensor(vec![10.0, 20.0], vec![2], true);

    assert_out_autograd_error(add_out(&out, &a, &b));
    assert_eq!(out.data_vec().unwrap(), &[0.0, 0.0]);
}

#[test]
fn add_scaled_out_rejects_left_nonleaf_source() {
    let leaf = cpu_tensor(vec![1.0, 2.0], vec![2], true);
    let plain = cpu_tensor(vec![10.0, 20.0], vec![2], false);
    let nonleaf = add(&leaf, &plain).unwrap();
    assert!(nonleaf.grad_fn().is_some());

    let out = cpu_tensor(vec![0.0, 0.0], vec![2], false);
    assert_out_autograd_error(add_scaled_out(&out, &nonleaf, &plain, 2.0));
    assert_eq!(out.data_vec().unwrap(), &[0.0, 0.0]);
}

#[test]
fn add_scaled_out_rejects_right_nonleaf_source() {
    let leaf = cpu_tensor(vec![1.0, 2.0], vec![2], true);
    let plain = cpu_tensor(vec![10.0, 20.0], vec![2], false);
    let nonleaf = add(&leaf, &plain).unwrap();
    assert!(nonleaf.grad_fn().is_some());

    let out = cpu_tensor(vec![0.0, 0.0], vec![2], false);
    assert_out_autograd_error(add_scaled_out(&out, &plain, &nonleaf, 0.5));
    assert_eq!(out.data_vec().unwrap(), &[0.0, 0.0]);
}

#[test]
fn add_out_allows_tracking_sources_inside_no_grad() {
    let out = cpu_tensor(vec![f32::NAN, f32::NAN], vec![2], false);
    let a = cpu_tensor(vec![1.0, 2.0], vec![2], true);
    let b = cpu_tensor(vec![10.0, 20.0], vec![2], false);

    no_grad(|| add_out(&out, &a, &b)).unwrap();

    assert_eq!(out.data_vec().unwrap(), &[11.0, 22.0]);
}

#[test]
fn add_out_rejects_tracking_source_before_resize_or_write() {
    let out = cpu_tensor(vec![123.0], vec![1], false);
    let a = cpu_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true);
    let b = cpu_tensor(vec![10.0, 20.0], vec![2], false);

    assert_out_autograd_error(add_out(&out, &a, &b));

    assert_eq!(out.shape(), &[1]);
    assert_eq!(out.data_vec().unwrap(), &[123.0]);
}

#[cfg(feature = "gpu")]
mod gpu {
    use std::sync::Once;

    use ferrotorch_core::Device;
    use ferrotorch_core::creation::from_vec;
    use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
    use ferrotorch_core::grad_fns::arithmetic::add_out;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for CORE-020 GPU probe");
        });
    }

    fn assert_out_autograd_error(result: FerrotorchResult<()>) {
        let err = result.expect_err("CUDA out= with tracking source input must error");
        match &err {
            FerrotorchError::InvalidArgument { message } => assert!(
                message.contains("out=... arguments don't support automatic differentiation"),
                "unexpected InvalidArgument message: {message}"
            ),
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn cuda_add_out_rejects_source_requires_grad_before_kernel_or_write() {
        ensure_cuda_backend();

        let out = from_vec::<f32>(vec![0.0, 0.0], &[2])
            .unwrap()
            .to(Device::Cuda(0))
            .unwrap();
        let a = from_vec::<f32>(vec![1.0, 2.0], &[2])
            .unwrap()
            .to(Device::Cuda(0))
            .unwrap()
            .requires_grad_(true);
        let b = from_vec::<f32>(vec![10.0, 20.0], &[2])
            .unwrap()
            .to(Device::Cuda(0))
            .unwrap();

        assert_out_autograd_error(add_out(&out, &a, &b));

        assert!(out.is_cuda(), "failed out= guard must not demote out");
        let host = out.to(Device::Cpu).unwrap();
        assert_eq!(host.data_vec().unwrap(), &[0.0, 0.0]);
    }
}
