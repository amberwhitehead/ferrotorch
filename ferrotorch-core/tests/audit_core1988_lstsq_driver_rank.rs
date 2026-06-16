//! CORE-1988: `linalg.lstsq` driver and rank contract.
//!
//! PyTorch returns an integer rank tensor and intentionally varies residuals,
//! rank, and singular values by driver:
//! - CPU default is `gelsy`: scalar int64 rank, empty residuals/singular values.
//! - CPU `gels`: residuals for overdetermined systems, empty rank/singular values.
//! - CPU `gelsd`/`gelss`: scalar int64 rank, singular values, residuals only
//!   when overdetermined and full-rank.
//! - CUDA accepts only `gels`; default CUDA rank/singular values are empty.

use ferrotorch_core::device::Device;
use ferrotorch_core::error::FerrotorchError;
use ferrotorch_core::linalg::{self, LstsqDriver};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

fn t(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

fn assert_close(actual: &[f64], expected: &[f64], tol: f64, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch {} vs {}",
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "{label}[{i}]: actual={a}, expected={e}, diff={}",
            (a - e).abs()
        );
    }
}

#[test]
fn cpu_default_gelsy_returns_integer_rank_and_empty_metadata() {
    let a = t(&[0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0], &[4, 2]);
    let b = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2]);

    let (sol, residuals, rank, sv) = linalg::lstsq(&a, &b, None).unwrap();

    assert_close(
        &sol.data_vec().unwrap(),
        &[2.0, 2.0, 1.0, 2.0],
        1e-12,
        "gelsy solution",
    );
    assert_eq!(residuals.shape(), &[0]);
    assert_eq!(rank.shape(), &[] as &[usize]);
    assert_eq!(rank.data().unwrap(), &[2]);
    assert_eq!(rank.device(), Device::Cpu);
    assert_eq!(sv.shape(), &[0]);
}

#[test]
fn cpu_gels_returns_residuals_but_no_rank_or_singular_values() {
    let a = t(&[0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0], &[4, 2]);
    let b = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2]);

    let (sol, residuals, rank, sv) =
        linalg::lstsq_with_driver(&a, &b, None, Some(LstsqDriver::Gels)).unwrap();

    assert_close(
        &sol.data_vec().unwrap(),
        &[2.0, 2.0, 1.0, 2.0],
        1e-12,
        "gels solution",
    );
    assert_eq!(residuals.shape(), &[2]);
    assert!(
        residuals
            .data_vec()
            .unwrap()
            .iter()
            .all(|v| v.abs() < 1e-24)
    );
    assert_eq!(rank.shape(), &[0]);
    assert_eq!(rank.data().unwrap(), &[] as &[i64]);
    assert_eq!(sv.shape(), &[0]);
}

#[test]
fn cpu_gelsd_returns_rank_singular_values_and_full_rank_residuals() {
    let a = t(&[0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0], &[4, 2]);
    let b = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2]);

    let (sol, residuals, rank, sv) =
        linalg::lstsq_with_driver(&a, &b, None, Some(LstsqDriver::Gelsd)).unwrap();

    assert_close(
        &sol.data_vec().unwrap(),
        &[2.0, 2.0, 1.0, 2.0],
        1e-12,
        "gelsd solution",
    );
    assert_eq!(rank.shape(), &[] as &[usize]);
    assert_eq!(rank.data().unwrap(), &[2]);
    assert_eq!(residuals.shape(), &[2]);
    assert!(
        residuals
            .data_vec()
            .unwrap()
            .iter()
            .all(|v| v.abs() < 1e-24)
    );
    assert_close(
        &sv.data_vec().unwrap(),
        &[4.100030448168238, 1.090756766696107],
        1e-12,
        "gelsd singular values",
    );
}

#[test]
fn cpu_gelss_rank_deficient_omits_residuals_and_reports_zero_singular_value() {
    let a = t(&[1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0], &[4, 2]);
    let b = t(&[1.0, 2.0, 3.0, 4.0], &[4, 1]);

    let (sol, residuals, rank, sv) =
        linalg::lstsq_with_driver(&a, &b, None, Some(LstsqDriver::Gelss)).unwrap();

    assert_close(
        &sol.data_vec().unwrap(),
        &[1.0, 0.0],
        1e-12,
        "gelss solution",
    );
    assert_eq!(rank.shape(), &[] as &[usize]);
    assert_eq!(rank.data().unwrap(), &[1]);
    assert_eq!(residuals.shape(), &[0]);
    assert_close(
        &sv.data_vec().unwrap(),
        &[5.477225575051661, 0.0],
        1e-12,
        "gelss singular values",
    );
}

#[test]
fn cpu_gels_rank_deficient_reports_lapack_failure_instead_of_fake_rank() {
    let a = t(&[1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0], &[4, 2]);
    let b = t(&[1.0, 2.0, 3.0, 4.0], &[4, 1]);

    let err = linalg::lstsq_with_driver(&a, &b, None, Some(LstsqDriver::Gels)).unwrap_err();
    assert!(
        matches!(
            err,
            FerrotorchError::InvalidArgument { ref message }
                if message.contains("does not have full rank")
        ),
        "expected full-rank LAPACK failure, got {err:?}"
    );
}

#[cfg(feature = "gpu")]
fn init_cuda() {
    ferrotorch_gpu::init_cuda_backend().expect("CUDA backend");
}

#[cfg(feature = "gpu")]
#[test]
fn cuda_default_is_gels_and_rejects_other_drivers_without_host_fallback() {
    init_cuda();
    let a = t(&[0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0], &[4, 2])
        .to(Device::Cuda(0))
        .unwrap();
    let b = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2])
        .to(Device::Cuda(0))
        .unwrap();

    let (_sol, residuals, rank, sv) = linalg::lstsq(&a, &b, None).unwrap();
    assert_eq!(residuals.device(), Device::Cuda(0));
    assert_eq!(residuals.shape(), &[2]);
    assert_eq!(rank.device(), Device::Cuda(0));
    assert_eq!(rank.shape(), &[0]);
    assert_eq!(sv.device(), Device::Cuda(0));
    assert_eq!(sv.shape(), &[0]);

    let err = linalg::lstsq_with_driver(&a, &b, None, Some(LstsqDriver::Gelsy)).unwrap_err();
    assert!(
        matches!(
            err,
            FerrotorchError::InvalidArgument { ref message }
                if message.contains("`driver` other than `gels` is not supported on CUDA")
        ),
        "expected CUDA driver rejection, got {err:?}"
    );
}
