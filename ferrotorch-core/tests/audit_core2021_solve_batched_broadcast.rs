//! Regression tests for #2021: `torch.linalg.solve` supports batched `A` and
//! broadcasted matrix RHS batch dimensions. Vector RHS follows PyTorch's
//! stricter rule: `B` is vector-like only when it is 1-D or exactly
//! `A.shape[:-1]`; broadcastable-but-not-equal vector batches are errors.

use ferrotorch_core::grad_fns::reduction::sum;
use ferrotorch_core::linalg::{solve, solve_ex};
use ferrotorch_core::{FerrotorchError, Tensor, TensorStorage};

fn t(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
    Tensor::from_storage(
        TensorStorage::cpu(data.to_vec()),
        shape.to_vec(),
        requires_grad,
    )
    .expect("cpu tensor")
}

fn assert_close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch: actual={actual:?} expected={expected:?}"
    );
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= 1e-5,
            "{label}: mismatch at {i}: actual={a} expected={e}; full actual={actual:?}"
        );
    }
}

fn assert_invalid(result: ferrotorch_core::FerrotorchResult<Tensor<f32>>, needle: &str) {
    match result {
        Err(FerrotorchError::InvalidArgument { message })
        | Err(FerrotorchError::ShapeMismatch { message }) => {
            assert!(
                message.contains(needle),
                "error message {message:?} did not contain {needle:?}"
            );
        }
        Err(other) => panic!("expected structural shape error, got {other:?}"),
        Ok(tensor) => panic!("expected structural shape error, got Ok({tensor:?})"),
    }
}

fn batched_diag_a() -> Tensor<f32> {
    t(
        &[
            1.0, 0.0, 0.0, 2.0, //
            2.0, 0.0, 0.0, 4.0, //
            4.0, 0.0, 0.0, 8.0,
        ],
        &[3, 2, 2],
        false,
    )
}

#[test]
fn cpu_solve_broadcasts_matrix_rhs_batch_over_unbatched_a() {
    let a = t(&[2.0, 0.0, 0.0, 4.0], &[2, 2], false);
    let b = t(&[4.0, 8.0, 6.0, 12.0, 8.0, 16.0], &[3, 2, 1], false);

    let x = solve(&a, &b).expect("batched RHS solve");
    assert_eq!(x.shape(), &[3, 2, 1]);
    assert_close(x.data().unwrap(), &[2.0, 2.0, 3.0, 3.0, 4.0, 4.0], "x");
}

#[test]
fn cpu_solve_broadcasts_1d_vector_rhs_over_batched_a() {
    let a = batched_diag_a();
    let b = t(&[8.0, 16.0], &[2], false);

    let x = solve(&a, &b).expect("batched A vector RHS solve");
    assert_eq!(x.shape(), &[3, 2]);
    assert_close(x.data().unwrap(), &[8.0, 8.0, 4.0, 4.0, 2.0, 2.0], "x");
}

#[test]
fn cpu_solve_rejects_broadcastable_batched_vector_that_is_not_exact_vector_case() {
    let a1 = t(&[1.0, 0.0, 0.0, 1.0], &[1, 2, 2], false);
    let b3 = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], false);
    assert_invalid(solve(&a1, &b3), "incompatible shapes");

    let a3 = batched_diag_a();
    let b1 = t(&[1.0, 2.0], &[1, 2], false);
    assert_invalid(solve(&a3, &b1), "incompatible shapes");
}

#[test]
fn cpu_solve_broadcasted_rhs_backward_sums_over_expanded_batch() {
    let a = batched_diag_a();
    let b = t(&[8.0, 16.0], &[1, 2, 1], true);

    let x = solve(&a, &b).expect("batched solve");
    assert_eq!(x.shape(), &[3, 2, 1]);
    let loss = sum(&x).expect("sum loss");
    loss.backward().expect("backward");

    let grad_b = b.grad().unwrap().expect("B grad");
    assert_eq!(grad_b.shape(), &[1, 2, 1]);
    assert_close(grad_b.data().unwrap(), &[1.75, 0.875], "grad B");
}

#[test]
fn cpu_solve_zero_batch_preserves_shape_and_zero_gradients() {
    let a = t(&[], &[0, 2, 2], true);
    let b = t(&[3.0, 5.0], &[2], true);

    let x = solve(&a, &b).expect("zero batch solve");
    assert_eq!(x.shape(), &[0, 2]);
    assert!(x.data().unwrap().is_empty());

    let loss = sum(&x).expect("sum empty");
    loss.backward().expect("backward empty");

    let grad_a = a.grad().unwrap().expect("A grad");
    assert_eq!(grad_a.shape(), &[0, 2, 2]);
    assert!(grad_a.data().unwrap().is_empty());

    let grad_b = b.grad().unwrap().expect("B grad");
    assert_eq!(grad_b.shape(), &[2]);
    assert_close(grad_b.data().unwrap(), &[0.0, 0.0], "B grad");
}

#[test]
fn cpu_solve_ex_reports_info_for_each_a_batch() {
    let a = t(
        &[
            1.0, 0.0, 0.0, 2.0, //
            1.0, 0.0, 0.0, 0.0, //
            4.0, 0.0, 0.0, 8.0,
        ],
        &[3, 2, 2],
        false,
    );
    let b = t(&[8.0, 16.0], &[2], false);

    let (x, info) = solve_ex(&a, &b).expect("batched solve_ex");
    assert_eq!(x.shape(), &[3, 2]);
    assert_eq!(info.shape(), &[3]);
    assert_close(x.data().unwrap(), &[8.0, 8.0, 0.0, 0.0, 2.0, 2.0], "x");
    assert_close(info.data().unwrap(), &[0.0, 2.0, 0.0], "info");
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
                .expect("CUDA backend must initialize for batched solve tests");
        });
    }

    fn cuda(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
        let tensor = t(data, shape, false)
            .to(Device::Cuda(0))
            .expect("cpu to cuda");
        if requires_grad {
            tensor.requires_grad_(true)
        } else {
            tensor
        }
    }

    fn host_data(tensor: &Tensor<f32>) -> Vec<f32> {
        tensor
            .to(Device::Cpu)
            .expect("to cpu")
            .data()
            .expect("host data")
            .to_vec()
    }

    #[test]
    fn cuda_solve_broadcasts_matrix_rhs_batch_without_host_result() {
        ensure_cuda_backend();
        let a = cuda(&[2.0, 0.0, 0.0, 4.0], &[2, 2], false);
        let b = cuda(&[4.0, 8.0, 6.0, 12.0, 8.0, 16.0], &[3, 2, 1], false);

        let x = solve(&a, &b).expect("cuda batched RHS solve");
        assert_eq!(x.device(), Device::Cuda(0));
        assert_eq!(x.shape(), &[3, 2, 1]);
        assert_close(&host_data(&x), &[2.0, 2.0, 3.0, 3.0, 4.0, 4.0], "x");
    }

    #[test]
    fn cuda_solve_broadcasts_1d_vector_rhs_over_batched_a() {
        ensure_cuda_backend();
        let a = cuda(
            &[
                1.0, 0.0, 0.0, 2.0, //
                2.0, 0.0, 0.0, 4.0, //
                4.0, 0.0, 0.0, 8.0,
            ],
            &[3, 2, 2],
            false,
        );
        let b = cuda(&[8.0, 16.0], &[2], false);

        let x = solve(&a, &b).expect("cuda batched A vector RHS solve");
        assert_eq!(x.device(), Device::Cuda(0));
        assert_eq!(x.shape(), &[3, 2]);
        assert_close(&host_data(&x), &[8.0, 8.0, 4.0, 4.0, 2.0, 2.0], "x");
    }

    #[test]
    fn cuda_solve_zero_batch_returns_cuda_empty() {
        ensure_cuda_backend();
        let a = cuda(&[], &[0, 2, 2], false);
        let b = cuda(&[3.0, 5.0], &[2], false);

        let x = solve(&a, &b).expect("cuda zero batch solve");
        assert_eq!(x.device(), Device::Cuda(0));
        assert_eq!(x.shape(), &[0, 2]);
        assert!(host_data(&x).is_empty());
    }

    #[test]
    fn cuda_solve_broadcasted_rhs_backward_sums_over_expanded_batch() {
        ensure_cuda_backend();
        let a = cuda(
            &[
                1.0, 0.0, 0.0, 2.0, //
                2.0, 0.0, 0.0, 4.0, //
                4.0, 0.0, 0.0, 8.0,
            ],
            &[3, 2, 2],
            false,
        );
        let b = cuda(&[8.0, 16.0], &[1, 2, 1], true);

        let x = solve(&a, &b).expect("cuda batched solve");
        let loss = sum(&x).expect("sum loss");
        loss.backward().expect("cuda backward");

        let grad_b = b.grad().unwrap().expect("B grad");
        assert_eq!(grad_b.device(), Device::Cuda(0));
        assert_eq!(grad_b.shape(), &[1, 2, 1]);
        assert_close(&host_data(&grad_b), &[1.75, 0.875], "grad B");
    }

    #[test]
    fn cuda_solve_ex_reports_info_for_each_a_batch() {
        ensure_cuda_backend();
        let a = cuda(
            &[
                1.0, 0.0, 0.0, 2.0, //
                1.0, 0.0, 0.0, 0.0, //
                4.0, 0.0, 0.0, 8.0,
            ],
            &[3, 2, 2],
            false,
        );
        let b = cuda(&[8.0, 16.0], &[2], false);

        let (x, info) = solve_ex(&a, &b).expect("cuda batched solve_ex");
        assert_eq!(x.device(), Device::Cuda(0));
        assert_eq!(x.shape(), &[3, 2]);
        assert_eq!(info.device(), Device::Cuda(0));
        assert_eq!(info.shape(), &[3]);
        assert_close(&host_data(&x), &[8.0, 8.0, 0.0, 0.0, 2.0, 2.0], "x");
        assert_close(&host_data(&info), &[0.0, 2.0, 0.0], "info");
    }
}
