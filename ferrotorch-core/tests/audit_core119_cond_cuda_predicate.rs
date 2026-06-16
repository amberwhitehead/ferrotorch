use ferrotorch_core::Tensor;
use ferrotorch_core::dtype::Float as FtFloat;
use ferrotorch_core::ops::higher_order::cond;
use ferrotorch_core::storage::TensorStorage;

fn cpu_tensor<T: FtFloat>(data: &[T], shape: &[usize], requires_grad: bool) -> Tensor<T> {
    Tensor::from_storage(
        TensorStorage::cpu(data.to_vec()),
        shape.to_vec(),
        requires_grad,
    )
    .expect("cpu tensor")
}

fn assert_close_f32(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch, actual={actual:?}, expected={expected:?}"
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "{label}[{i}]: expected {e}, got {a}; actual={actual:?}, expected={expected:?}"
        );
    }
}

fn assert_close_f64(actual: &[f64], expected: &[f64], tol: f64, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch, actual={actual:?}, expected={expected:?}"
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "{label}[{i}]: expected {e}, got {a}; actual={actual:?}, expected={expected:?}"
        );
    }
}

#[test]
fn cond_cpu_float_predicates_match_torch_nonzero_truthiness() {
    // PyTorch 2.11 oracle:
    // torch.cond(torch.tensor(v), true, false, operands) takes the true branch
    // for every nonzero scalar tensor, including 0.25, 0.5, -1.0, and NaN.
    let operand = cpu_tensor(&[1.0f32], &[1], false);
    for (pred_value, expected) in [
        (0.0f32, -9.0f32),
        (0.25f32, 11.0f32),
        (0.5f32, 11.0f32),
        (-1.0f32, 11.0f32),
        (f32::NAN, 11.0f32),
    ] {
        let pred = cpu_tensor(&[pred_value], &[], false);
        let out = cond(
            &pred,
            |_| vec![cpu_tensor(&[11.0f32], &[1], false)],
            |_| vec![cpu_tensor(&[-9.0f32], &[1], false)],
            std::slice::from_ref(&operand),
        )
        .expect("cond f32 predicate");
        assert_close_f32(
            &out[0].data_vec().expect("out"),
            &[expected],
            0.0,
            "f32 predicate output",
        );
    }

    let operand = cpu_tensor(&[1.0f64], &[1], false);
    for (pred_value, expected) in [
        (0.0f64, -9.0f64),
        (0.25f64, 11.0f64),
        (0.5f64, 11.0f64),
        (-1.0f64, 11.0f64),
        (f64::NAN, 11.0f64),
    ] {
        let pred = cpu_tensor(&[pred_value], &[], false);
        let out = cond(
            &pred,
            |_| vec![cpu_tensor(&[11.0f64], &[1], false)],
            |_| vec![cpu_tensor(&[-9.0f64], &[1], false)],
            std::slice::from_ref(&operand),
        )
        .expect("cond f64 predicate");
        assert_close_f64(
            &out[0].data_vec().expect("out"),
            &[expected],
            0.0,
            "f64 predicate output",
        );
    }
}

#[cfg(feature = "gpu")]
mod cuda {
    use super::{Tensor, assert_close_f32, assert_close_f64, cond, cpu_tensor};
    use ferrotorch_core::device::Device;
    use ferrotorch_core::grad_fns::arithmetic::{add, mul};
    use ferrotorch_core::grad_fns::reduction::sum;
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for CORE-119 probes");
        });
    }

    fn cuda_leaf_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        cpu_tensor(data, shape, false)
            .to(Device::Cuda(0))
            .expect("to cuda f32")
            .requires_grad_(true)
    }

    fn cuda_leaf_f64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
        cpu_tensor(data, shape, false)
            .to(Device::Cuda(0))
            .expect("to cuda f64")
            .requires_grad_(true)
    }

    fn read_cuda_f32(t: &Tensor<f32>, label: &str) -> Vec<f32> {
        assert!(t.is_cuda(), "{label}: tensor must be CUDA-resident");
        t.cpu()
            .expect("cuda to cpu")
            .data_vec()
            .expect("read cuda f32 tensor")
    }

    fn read_cuda_f64(t: &Tensor<f64>, label: &str) -> Vec<f64> {
        assert!(t.is_cuda(), "{label}: tensor must be CUDA-resident");
        t.cpu()
            .expect("cuda to cpu")
            .data_vec()
            .expect("read cuda f64 tensor")
    }

    #[test]
    fn cond_cuda_f32_predicate_selects_true_branch_and_keeps_graph_cuda() {
        ensure_cuda_backend();
        let pred = cpu_tensor(&[0.25f32], &[], false)
            .to(Device::Cuda(0))
            .expect("predicate to cuda");
        let x = cuda_leaf_f32(&[2.0, 3.0], &[2]);

        let out = cond(
            &pred,
            |ops| vec![mul(&ops[0], &ops[0]).expect("true mul")],
            |ops| vec![add(&ops[0], &ops[0]).expect("false add")],
            std::slice::from_ref(&x),
        )
        .expect("cond cuda true predicate");

        assert!(out[0].is_cuda(), "cond true output must stay CUDA");
        assert_close_f32(
            &read_cuda_f32(&out[0], "cond true output"),
            &[4.0, 9.0],
            1e-6,
            "cond true output",
        );

        sum(&out[0]).expect("sum").backward().expect("backward");
        let grad = x.grad().expect("grad lookup").expect("x grad");
        assert!(grad.is_cuda(), "cond true gradient must stay CUDA");
        assert_close_f32(
            &read_cuda_f32(&grad, "cond true grad"),
            &[4.0, 6.0],
            1e-6,
            "cond true grad",
        );
    }

    #[test]
    fn cond_cuda_zero_f32_predicate_selects_false_branch_and_keeps_graph_cuda() {
        ensure_cuda_backend();
        let pred = cpu_tensor(&[0.0f32], &[], false)
            .to(Device::Cuda(0))
            .expect("predicate to cuda");
        let x = cuda_leaf_f32(&[2.0, 3.0], &[2]);

        let out = cond(
            &pred,
            |ops| vec![mul(&ops[0], &ops[0]).expect("true mul")],
            |ops| vec![add(&ops[0], &ops[0]).expect("false add")],
            std::slice::from_ref(&x),
        )
        .expect("cond cuda false predicate");

        assert!(out[0].is_cuda(), "cond false output must stay CUDA");
        assert_close_f32(
            &read_cuda_f32(&out[0], "cond false output"),
            &[4.0, 6.0],
            1e-6,
            "cond false output",
        );

        sum(&out[0]).expect("sum").backward().expect("backward");
        let grad = x.grad().expect("grad lookup").expect("x grad");
        assert!(grad.is_cuda(), "cond false gradient must stay CUDA");
        assert_close_f32(
            &read_cuda_f32(&grad, "cond false grad"),
            &[2.0, 2.0],
            1e-6,
            "cond false grad",
        );
    }

    #[test]
    fn cond_cuda_f64_predicate_uses_nonzero_truthiness() {
        ensure_cuda_backend();
        let pred = cpu_tensor(&[0.5f64], &[], false)
            .to(Device::Cuda(0))
            .expect("predicate to cuda");
        let x = cuda_leaf_f64(&[2.0], &[1]);

        let out = cond(
            &pred,
            |ops| vec![mul(&ops[0], &ops[0]).expect("true mul")],
            |ops| vec![add(&ops[0], &ops[0]).expect("false add")],
            std::slice::from_ref(&x),
        )
        .expect("cond cuda f64 predicate");

        assert!(out[0].is_cuda(), "cond f64 output must stay CUDA");
        assert_close_f64(
            &read_cuda_f64(&out[0], "cond f64 output"),
            &[4.0],
            1e-12,
            "cond f64 output",
        );
    }

    #[test]
    fn cond_cuda_nan_predicate_is_true_like_torch() {
        ensure_cuda_backend();
        let pred = cpu_tensor(&[f32::NAN], &[], false)
            .to(Device::Cuda(0))
            .expect("predicate to cuda");
        let x = cuda_leaf_f32(&[2.0], &[1]);

        let out = cond(
            &pred,
            |ops| vec![mul(&ops[0], &ops[0]).expect("true mul")],
            |ops| vec![add(&ops[0], &ops[0]).expect("false add")],
            std::slice::from_ref(&x),
        )
        .expect("cond cuda nan predicate");

        assert!(out[0].is_cuda(), "cond NaN output must stay CUDA");
        assert_close_f32(
            &read_cuda_f32(&out[0], "cond NaN output"),
            &[4.0],
            1e-6,
            "cond NaN output",
        );
    }
}
