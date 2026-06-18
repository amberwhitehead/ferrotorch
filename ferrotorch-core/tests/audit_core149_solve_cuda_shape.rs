//! Regression tests for CORE-149 (#1843): `linalg::solve` must validate RHS
//! shape before CUDA dispatch and must pass packed logical buffers to cuSOLVER.
//!
//! PyTorch validates structural solve shapes on every device before the solver
//! is reached:
//!
//! ```python
//! >>> A = torch.eye(2, device="cuda")
//! >>> torch.linalg.solve(A, torch.ones(3, device="cuda"))
//! RuntimeError: linalg.solve: Incompatible shapes of A and B for the equation AX = B (2x2 and 3x1)
//! ```
//!
//! Ferrotorch currently exposes the unbatched subset documented by
//! `linalg::solve`: `A` is 2-D square and `B` is either `[n]` or `[n, nrhs]`.
//! Invalid shapes must return structured `InvalidArgument`, not reach backend
//! raw-buffer checks or accidentally solve a mislabeled 3-D RHS.

use ferrotorch_core::linalg::{solve, solve_ex};
use ferrotorch_core::{FerrotorchError, Tensor, TensorStorage};

fn cpu_f32(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
    Tensor::from_storage(
        TensorStorage::cpu(data.to_vec()),
        shape.to_vec(),
        requires_grad,
    )
    .expect("cpu tensor")
}

fn assert_invalid_tensor(result: ferrotorch_core::FerrotorchResult<Tensor<f32>>, needle: &str) {
    match result {
        Err(FerrotorchError::InvalidArgument { message }) => {
            assert!(
                message.contains(needle),
                "error message {message:?} did not contain {needle:?}"
            );
        }
        Err(other) => panic!("expected InvalidArgument, got {other:?}"),
        Ok(tensor) => panic!("expected InvalidArgument, got Ok({tensor:?})"),
    }
}

fn assert_invalid_solve_ex(
    result: ferrotorch_core::FerrotorchResult<(Tensor<f32>, Tensor<f32>)>,
    needle: &str,
) {
    match result {
        Err(FerrotorchError::InvalidArgument { message }) => {
            assert!(
                message.contains(needle),
                "error message {message:?} did not contain {needle:?}"
            );
        }
        Err(other) => panic!("expected InvalidArgument, got {other:?}"),
        Ok((x, info)) => panic!("expected InvalidArgument, got Ok(({x:?}, {info:?}))"),
    }
}

#[test]
fn cpu_solve_rejects_incompatible_rhs_length_before_factorization() {
    let a = cpu_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2], false);
    let b = cpu_f32(&[1.0, 2.0, 3.0], &[3], false);
    assert_invalid_tensor(solve(&a, &b), "incompatible shapes");
    assert_invalid_solve_ex(solve_ex(&a, &b), "incompatible shapes");
}

#[test]
fn cpu_solve_rejects_rank3_rhs_for_unbatched_contract() {
    let a = cpu_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2], false);
    let b = cpu_f32(&[1.0, 2.0], &[2, 1, 1], false);
    assert_invalid_tensor(solve(&a, &b), "b must be 1-D or 2-D");
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
                .expect("CUDA backend must initialize for solve shape tests");
        });
    }

    fn cuda_f32(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
        cpu_f32(data, shape, requires_grad)
            .to(Device::Cuda(0))
            .expect("cpu to cuda")
    }

    #[test]
    fn cuda_solve_rejects_incompatible_rhs_length_before_backend() {
        ensure_cuda_backend();
        let a = cuda_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2], false);
        let b = cuda_f32(&[1.0, 2.0, 3.0], &[3], false);

        assert_invalid_tensor(solve(&a, &b), "incompatible shapes");
        assert_invalid_solve_ex(solve_ex(&a, &b), "incompatible shapes");
    }

    #[test]
    fn cuda_solve_rejects_rank3_rhs_before_mislabeled_solve() {
        ensure_cuda_backend();
        let a = cuda_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2], false);
        let b = cuda_f32(&[1.0, 2.0], &[2, 1, 1], false);

        assert_invalid_tensor(solve(&a, &b), "b must be 1-D or 2-D");
    }

    #[test]
    fn cuda_solve_accepts_offset_views_by_packing_on_device() {
        ensure_cuda_backend();
        let a_base = cuda_f32(&[99.0, 99.0, 2.0, 0.0, 0.0, 2.0], &[3, 2], false);
        let a = a_base.narrow(0, 1, 2).expect("A offset view");
        assert_eq!(a.shape(), &[2, 2]);
        assert_eq!(a.storage_offset(), 2);
        assert_eq!(a.storage_len(), 6);

        let b_base = cuda_f32(&[-1.0, 4.0, 6.0], &[3], false);
        let b = b_base.narrow(0, 1, 2).expect("B offset view");
        assert_eq!(b.shape(), &[2]);
        assert_eq!(b.storage_offset(), 1);
        assert_eq!(b.storage_len(), 3);

        let x = solve(&a, &b).expect("solve offset views");
        assert_eq!(x.device(), Device::Cuda(0));
        assert_eq!(x.shape(), &[2]);
        let x_cpu = x.cpu().expect("read back solution");
        let values = x_cpu.data().expect("solution data");
        assert_eq!(values, &[2.0, 3.0]);
    }
}
