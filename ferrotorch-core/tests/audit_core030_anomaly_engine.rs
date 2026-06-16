use ferrotorch_core::autograd::graph::backward_parallel;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::{Tensor, detect_anomaly};

fn leaf(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true)
        .expect("construct leaf")
}

fn constant(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .expect("construct constant")
}

fn zero_like_device(t: &Tensor<f32>) -> Tensor<f32> {
    constant(&vec![0.0; t.numel()], t.shape())
        .to(t.device())
        .expect("move constant to tensor device")
}

fn nan_gradient_loss(x: &Tensor<f32>, add_depth: usize) -> Tensor<f32> {
    let zero = zero_like_device(x);
    let mut y = x.mul_t(&zero).expect("x * 0");
    for _ in 0..add_depth {
        let add_zero = zero_like_device(x);
        y = y.add_t(&add_zero).expect("add zero");
    }
    y.sqrt_t().expect("sqrt").sum_all().expect("sum")
}

#[test]
fn detect_anomaly_serial_reports_normal_op_nan_gradient_with_forward_trace() {
    let x = leaf(&[1.0, 2.0, 3.0], &[3]);

    let result = detect_anomaly(|| {
        let loss = nan_gradient_loss(&x, 0);
        loss.backward()
    });

    let err = result.expect_err("sqrt(x * 0) must produce a NaN gradient through MulBackward");
    let msg = err.to_string();
    assert!(
        msg.contains("Function 'MulBackward' returned nan values in its 0th output"),
        "wrong anomaly boundary/message: {msg}"
    );
    assert!(
        msg.contains("Forward-pass backtrace"),
        "missing captured forward provenance: {msg}"
    );
}

#[test]
fn detect_anomaly_parallel_reports_normal_op_nan_gradient_with_forward_trace() {
    let x = leaf(&[1.0, 2.0, 3.0], &[3]);

    let result = detect_anomaly(|| {
        let loss = nan_gradient_loss(&x, 8);
        backward_parallel(&loss, None, 4)
    });

    let err = result.expect_err("parallel engine must check backward outputs too");
    let msg = err.to_string();
    assert!(
        msg.contains("Function 'MulBackward' returned nan values in its 0th output"),
        "wrong anomaly boundary/message: {msg}"
    );
    assert!(
        msg.contains("Forward-pass backtrace"),
        "missing captured forward provenance: {msg}"
    );
}

#[test]
fn detect_anomaly_allows_inf_only_gradients_like_torch_check_nan() {
    let x = leaf(&[1.0, 2.0, 3.0], &[3]);
    let zero = constant(&[0.0, 0.0, 0.0], &[3]);

    detect_anomaly(|| {
        let loss = x.div_t(&zero).expect("x / 0").sum_all().expect("sum");
        loss.backward()
    })
    .expect("PyTorch detect_anomaly(check_nan=True) does not reject Inf-only gradients");

    let grad = x.grad().expect("grad access").expect("x.grad");
    let data = grad.data_vec().expect("grad data");
    assert!(
        data.iter().all(|v| v.is_infinite() && v.is_sign_positive()),
        "expected positive Inf gradient, got {data:?}"
    );
}

#[cfg(feature = "gpu")]
mod cuda {
    use super::nan_gradient_loss;
    use ferrotorch_core::storage::TensorStorage;
    use ferrotorch_core::{BoolTensor, Device, Tensor, detect_anomaly};
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for anomaly integration tests");
        });
    }

    fn cuda_leaf(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
            .expect("construct cpu tensor")
            .to(Device::Cuda(0))
            .expect("upload to cuda")
            .requires_grad_(true)
    }

    fn cuda_constant(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
            .expect("construct cpu tensor")
            .to(Device::Cuda(0))
            .expect("upload to cuda")
    }

    #[test]
    fn detect_anomaly_cuda_checks_nan_gradient_without_value_readback_fallback() {
        ensure_cuda_backend();
        let x = cuda_leaf(&[1.0, 2.0, 3.0], &[3]);

        let result = detect_anomaly(|| {
            let loss = nan_gradient_loss(&x, 0);
            loss.backward()
        });

        let err = match result {
            Err(err) => err,
            Ok(()) => {
                let grad_debug = x.grad().expect("grad access").map(|grad| {
                    grad.cpu()
                        .expect("grad readback")
                        .data_vec()
                        .expect("grad data")
                });
                panic!(
                    "CUDA backward output NaN must be detected before accumulation; x.grad={grad_debug:?}"
                );
            }
        };
        let msg = err.to_string();
        assert!(
            msg.contains("Function 'MulBackward' returned nan values in its 0th output"),
            "wrong CUDA anomaly boundary/message: {msg}"
        );
        assert!(
            msg.contains("Forward-pass backtrace"),
            "missing CUDA forward provenance: {msg}"
        );
        assert!(
            x.grad().expect("grad access").is_none(),
            "anomaly failure must stop before accumulating the bad CUDA gradient"
        );
    }

    #[test]
    fn detect_anomaly_cuda_allows_inf_only_gradients_like_torch_check_nan() {
        ensure_cuda_backend();
        let x = cuda_leaf(&[1.0, 2.0, 3.0], &[3]);
        let zero = cuda_constant(&[0.0, 0.0, 0.0], &[3]);

        detect_anomaly(|| {
            let loss = x.div_t(&zero).expect("x / 0").sum_all().expect("sum");
            loss.backward()
        })
        .expect("PyTorch detect_anomaly(check_nan=True) does not reject CUDA Inf-only gradients");

        let grad = x.grad().expect("grad access").expect("x.grad");
        assert_eq!(
            grad.device(),
            Device::Cuda(0),
            "grad must stay CUDA-resident"
        );
        let data = grad
            .cpu()
            .expect("grad readback")
            .data_vec()
            .expect("grad data");
        assert!(
            data.iter().all(|v| v.is_infinite() && v.is_sign_positive()),
            "expected positive Inf CUDA gradient, got {data:?}"
        );
    }

    #[test]
    fn cuda_float_ne_treats_nan_as_unequal_like_torch() {
        ensure_cuda_backend();
        let a = cuda_constant(&[f32::NAN, 1.0, f32::INFINITY, -0.0], &[4]);
        let b = cuda_constant(&[f32::NAN, 1.0, f32::INFINITY, 0.0], &[4]);

        let ne = BoolTensor::ne(&a, &b).expect("cuda ne");
        assert_eq!(
            ne.device(),
            Device::Cuda(0),
            "comparison result must stay CUDA-resident"
        );
        let ne_cpu = ne.to(Device::Cpu).expect("mask readback");
        assert_eq!(
            ne_cpu.data().expect("mask data"),
            &[true, false, false, false],
            "torch: NaN != NaN is true, equal finite/Inf/signed-zero lanes are false"
        );
    }
}
