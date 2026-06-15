//! CORE-1957: `cat` must preserve PyTorch's legacy handling of 1-D empty
//! tensors (`shape == [0]`).
//!
//! PyTorch skips exactly those legacy empties for dim wrapping and shape
//! compatibility, but still enforces same-device placement and still returns an
//! empty `[0]` gradient for skipped tracking inputs.

use ferrotorch_core::{FerrotorchError, Tensor, TensorStorage, cat};

fn t(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
    Tensor::from_storage(
        TensorStorage::cpu(data.to_vec()),
        shape.to_vec(),
        requires_grad,
    )
    .unwrap()
}

fn empty_1d(requires_grad: bool) -> Tensor<f32> {
    t(&[], &[0], requires_grad)
}

fn assert_data(t: &Tensor<f32>, shape: &[usize], expected: &[f32]) {
    assert_eq!(t.shape(), shape);
    assert_eq!(t.data_vec().unwrap(), expected);
}

#[test]
fn cat_skips_legacy_empty_for_dim_and_shape_checks() {
    let empty = empty_1d(false);
    let full = t(&[1.0; 25], &[5, 5], false);

    let out = cat(&[empty.clone(), full.clone()], 1).unwrap();
    assert_data(&out, &[5, 5], &[1.0; 25]);

    let out = cat(&[full.clone(), empty.clone()], 1).unwrap();
    assert_data(&out, &[5, 5], &[1.0; 25]);

    let out = cat(&[empty.clone(), full, empty], -1).unwrap();
    assert_data(&out, &[5, 5], &[1.0; 25]);
}

#[test]
fn cat_all_legacy_empty_accepts_any_dim_like_torch() {
    for dim in [-5, -1, 0, 1, 100] {
        let a = empty_1d(false);
        let b = empty_1d(false);
        let out = cat(&[a, b], dim).unwrap();
        assert_data(&out, &[0], &[]);
    }
}

#[test]
fn cat_does_not_skip_other_empty_shapes() {
    let empty_2d = t(&[], &[0, 2], false);
    let full = t(&[1.0; 25], &[5, 5], false);

    let err = cat(&[empty_2d, full], 1).expect_err("non-legacy empty shapes are not skipped");
    assert!(
        matches!(err, FerrotorchError::ShapeMismatch { .. }),
        "expected shape mismatch for non-legacy empty, got {err:?}"
    );
}

#[test]
fn cat_backward_returns_empty_grad_for_skipped_legacy_empty() {
    let empty = empty_1d(true);
    let full = t(&[1.0; 25], &[5, 5], true);
    let out = cat(&[empty.clone(), full.clone()], 1).unwrap();
    assert_eq!(out.shape(), &[5, 5]);

    let grad_seed: Vec<f32> = (0..25).map(|v| v as f32).collect();
    let grad = t(&grad_seed, &[5, 5], false);
    out.backward_with_gradient(&grad).unwrap();

    let empty_grad = empty.grad().unwrap().unwrap();
    assert_data(&empty_grad, &[0], &[]);
    let full_grad = full.grad().unwrap().unwrap();
    assert_data(&full_grad, &[5, 5], &grad_seed);
}

#[test]
fn cat_all_legacy_empty_backward_returns_empty_grads() {
    let a = empty_1d(true);
    let b = empty_1d(true);
    let out = cat(&[a.clone(), b.clone()], 100).unwrap();
    assert_data(&out, &[0], &[]);

    out.backward_with_gradient(&empty_1d(false)).unwrap();

    let a_grad = a.grad().unwrap().unwrap();
    assert_data(&a_grad, &[0], &[]);
    let b_grad = b.grad().unwrap().unwrap();
    assert_data(&b_grad, &[0], &[]);
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use ferrotorch_core::Device;
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for CORE-1957 probes");
        });
    }

    fn cuda(t: &Tensor<f32>) -> Tensor<f32> {
        t.to(Device::Cuda(0)).unwrap()
    }

    #[test]
    fn cuda_cat_skips_same_device_legacy_empty_without_host_demotion() {
        ensure_cuda_backend();
        let empty = cuda(&empty_1d(false));
        let full = cuda(&t(&[1.0; 25], &[5, 5], false));

        let out = cat(&[empty, full], 1).unwrap();

        assert_eq!(out.device(), Device::Cuda(0));
        assert_eq!(out.shape(), &[5, 5]);
        assert_eq!(out.cpu().unwrap().data_vec().unwrap(), vec![1.0; 25]);
    }

    #[test]
    fn cuda_cat_all_legacy_empty_accepts_any_dim_and_stays_cuda() {
        ensure_cuda_backend();

        for dim in [-5, -1, 0, 1, 100] {
            let a = cuda(&empty_1d(false));
            let b = cuda(&empty_1d(false));
            let out = cat(&[a, b], dim).unwrap();

            assert_eq!(out.device(), Device::Cuda(0));
            assert_eq!(out.shape(), &[0]);
            assert_eq!(out.cpu().unwrap().data_vec().unwrap(), Vec::<f32>::new());
        }
    }

    #[test]
    fn cuda_cat_rejects_mixed_device_legacy_empty() {
        ensure_cuda_backend();
        let cpu_empty = empty_1d(false);
        let cuda_empty = cuda(&empty_1d(false));
        let cpu_full = t(&[1.0; 25], &[5, 5], false);
        let cuda_full = cuda(&t(&[1.0; 25], &[5, 5], false));

        let err = cat(&[cpu_empty.clone(), cuda_full], 1).expect_err("mixed devices must reject");
        assert!(
            matches!(err, FerrotorchError::DeviceMismatch { .. }),
            "expected DeviceMismatch, got {err:?}"
        );

        let err = cat(&[cuda_empty.clone(), cpu_full], 1).expect_err("mixed devices must reject");
        assert!(
            matches!(err, FerrotorchError::DeviceMismatch { .. }),
            "expected DeviceMismatch, got {err:?}"
        );

        let err = cat(&[cpu_empty, cuda_empty], 0).expect_err("mixed devices must reject");
        assert!(
            matches!(err, FerrotorchError::DeviceMismatch { .. }),
            "expected DeviceMismatch, got {err:?}"
        );
    }

    #[test]
    fn cuda_cat_backward_returns_empty_cuda_grad_for_skipped_legacy_empty() {
        ensure_cuda_backend();
        let empty = cuda(&empty_1d(false)).requires_grad_(true);
        let full = cuda(&t(&[1.0; 25], &[5, 5], false)).requires_grad_(true);
        let out = cat(&[empty.clone(), full.clone()], 1).unwrap();

        let grad_seed: Vec<f32> = (0..25).map(|v| v as f32).collect();
        let grad = cuda(&t(&grad_seed, &[5, 5], false));
        out.backward_with_gradient(&grad).unwrap();

        let empty_grad = empty.grad().unwrap().unwrap();
        assert_eq!(empty_grad.device(), Device::Cuda(0));
        assert_eq!(empty_grad.shape(), &[0]);
        assert_eq!(
            empty_grad.cpu().unwrap().data_vec().unwrap(),
            Vec::<f32>::new()
        );

        let full_grad = full.grad().unwrap().unwrap();
        assert_eq!(full_grad.device(), Device::Cuda(0));
        assert_eq!(full_grad.shape(), &[5, 5]);
        assert_eq!(full_grad.cpu().unwrap().data_vec().unwrap(), grad_seed);
    }
}
