//! CORE-1990: torch.linalg.matrix_rank batch/hermitian/atol/rtol parity.
//!
//! PyTorch references:
//! - `/home/doll/pytorch/aten/src/ATen/native/LinearAlgebra.cpp`
//!   `get_atol_rtol`, `matrix_rank_impl`
//! - `/home/doll/pytorch/aten/src/ATen/native/native_functions.yaml`
//!   `linalg_matrix_rank.{atol_rtol_float,atol_rtol_tensor,tol_tensor}`

use ferrotorch_core::linalg::{
    matrix_rank, matrix_rank_atol_rtol, matrix_rank_atol_rtol_tensors, matrix_rank_tol_tensor,
};
use ferrotorch_core::{Device, Tensor, TensorStorage};
use half::f16;

fn tensor_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

fn tensor_f64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

fn tensor_f16(data: &[f32], shape: &[usize]) -> Tensor<f16> {
    Tensor::from_storage(
        TensorStorage::cpu(data.iter().copied().map(f16::from_f32).collect()),
        shape.to_vec(),
        false,
    )
    .unwrap()
}

#[test]
fn matrix_rank_cpu_accepts_batched_and_rectangular_inputs() {
    let a = tensor_f32(&[1.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 4.0], &[2, 2, 2]);
    let rank = matrix_rank(&a, None).unwrap();
    assert_eq!(rank.device(), Device::Cpu);
    assert_eq!(rank.shape(), &[2]);
    assert_eq!(rank.data().unwrap(), &[2, 1]);

    let rect = tensor_f64(
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 4.0, 6.0],
        &[2, 2, 3],
    );
    let rank = matrix_rank(&rect, None).unwrap();
    assert_eq!(rank.shape(), &[2]);
    assert_eq!(rank.data().unwrap(), &[2, 1]);
}

#[test]
fn matrix_rank_cpu_supports_hermitian_batch_and_empty_ordering() {
    let hermitian = tensor_f32(&[2.0, 0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 1.0], &[2, 2, 2]);
    let rank = matrix_rank_atol_rtol(&hermitian, None, None, true).unwrap();
    assert_eq!(rank.shape(), &[2]);
    assert_eq!(rank.data().unwrap(), &[1, 2]);

    let empty_nonsquare = tensor_f32(&[], &[2, 0, 3]);
    let rank = matrix_rank_atol_rtol(&empty_nonsquare, None, None, true).unwrap();
    assert_eq!(rank.shape(), &[2]);
    assert_eq!(rank.data().unwrap(), &[0, 0]);
}

#[test]
fn matrix_rank_cpu_matches_scalar_and_tensor_tolerance_broadcasting() {
    let a = tensor_f32(&[5.0, 0.0, 0.0, 1.0, 5.0, 0.0, 0.0, 3.0], &[2, 2, 2]);

    let scalar = matrix_rank_atol_rtol(&a, Some(2.0), None, false).unwrap();
    assert_eq!(scalar.shape(), &[2]);
    assert_eq!(scalar.data().unwrap(), &[1, 2]);

    let legacy = matrix_rank(&a, Some(2.0)).unwrap();
    assert_eq!(legacy.data().unwrap(), &[1, 2]);

    let tol = tensor_f32(&[2.0, 4.0], &[2]);
    let tensor_tol = matrix_rank_tol_tensor(&a, &tol, false).unwrap();
    assert_eq!(tensor_tol.shape(), &[2]);
    assert_eq!(tensor_tol.data().unwrap(), &[1, 1]);

    let atol = tensor_f32(&[2.0, 4.0], &[2, 1]);
    let broadcasted = matrix_rank_atol_rtol_tensors(&a, Some(&atol), None, false).unwrap();
    assert_eq!(broadcasted.shape(), &[2, 2]);
    assert_eq!(broadcasted.data().unwrap(), &[1, 2, 1, 1]);

    let atol64 = tensor_f64(&[2.0, 4.0], &[2, 1]);
    let mixed_atol = matrix_rank_atol_rtol_tensors(&a, Some(&atol64), None, false).unwrap();
    assert_eq!(mixed_atol.shape(), &[2, 2]);
    assert_eq!(mixed_atol.data().unwrap(), &[1, 2, 1, 1]);

    let b64 = tensor_f64(&[10.0, 0.0, 0.0, 1.0], &[2, 2]);
    let rtol32 = tensor_f32(&[0.2], &[]);
    let mixed_rtol = matrix_rank_atol_rtol_tensors(&b64, None, Some(&rtol32), false).unwrap();
    assert_eq!(mixed_rtol.shape(), &[]);
    assert_eq!(mixed_rtol.data().unwrap(), &[1]);
}

#[test]
fn matrix_rank_rejects_low_precision_and_rank_lt_two_like_torch() {
    let x16 = tensor_f16(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    assert!(
        matrix_rank(&x16, None)
            .unwrap_err()
            .to_string()
            .contains("requires f32 or f64")
    );

    let vector = tensor_f32(&[1.0, 2.0], &[2]);
    assert!(
        matrix_rank(&vector, None)
            .unwrap_err()
            .to_string()
            .contains("requires at least a 2-D tensor")
    );
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use ferrotorch_core::FerrotorchError;
    use std::sync::Once;

    static INIT: Once = Once::new();

    fn ensure_cuda() {
        INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend().expect("CUDA backend must initialize");
        });
    }

    fn cuda_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        ensure_cuda();
        tensor_f32(data, shape).to(Device::Cuda(0)).expect("upload")
    }

    fn cuda_f64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
        ensure_cuda();
        tensor_f64(data, shape).to(Device::Cuda(0)).expect("upload")
    }

    #[test]
    fn matrix_rank_cuda_batched_and_hermitian_stay_resident() {
        let a = cuda_f32(&[1.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 4.0], &[2, 2, 2]);
        let rank = matrix_rank(&a, None).unwrap();
        assert_eq!(rank.device(), Device::Cuda(0));
        assert_eq!(rank.shape(), &[2]);
        assert_eq!(rank.to(Device::Cpu).unwrap().data().unwrap(), &[2, 1]);

        let hermitian = cuda_f32(&[2.0, 0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 1.0], &[2, 2, 2]);
        let rank = matrix_rank_atol_rtol(&hermitian, None, None, true).unwrap();
        assert_eq!(rank.device(), Device::Cuda(0));
        assert_eq!(rank.to(Device::Cpu).unwrap().data().unwrap(), &[1, 2]);
    }

    #[test]
    fn matrix_rank_cuda_tensor_tolerance_broadcasts_on_device() {
        let a = cuda_f32(&[5.0, 0.0, 0.0, 1.0, 5.0, 0.0, 0.0, 3.0], &[2, 2, 2]);
        let atol = cuda_f32(&[2.0, 4.0], &[2, 1]);

        let rank = matrix_rank_atol_rtol_tensors(&a, Some(&atol), None, false).unwrap();
        assert_eq!(rank.device(), Device::Cuda(0));
        assert_eq!(rank.shape(), &[2, 2]);
        assert_eq!(rank.to(Device::Cpu).unwrap().data().unwrap(), &[1, 2, 1, 1]);

        let atol64 = cuda_f64(&[2.0, 4.0], &[2, 1]);
        let rank = matrix_rank_atol_rtol_tensors(&a, Some(&atol64), None, false).unwrap();
        assert_eq!(rank.device(), Device::Cuda(0));
        assert_eq!(rank.shape(), &[2, 2]);
        assert_eq!(rank.to(Device::Cpu).unwrap().data().unwrap(), &[1, 2, 1, 1]);
    }

    #[test]
    fn matrix_rank_cuda_rejects_cpu_tolerance_without_host_fallback() {
        let a = cuda_f32(&[5.0, 0.0, 0.0, 1.0], &[2, 2]);
        let cpu_tol = tensor_f32(&[2.0], &[1]);
        let err = matrix_rank_atol_rtol_tensors(&a, Some(&cpu_tol), None, false).unwrap_err();
        assert!(matches!(err, FerrotorchError::DeviceMismatch { .. }));
    }
}
