//! Red-then-green regression tests for audit finding CORE-077 (crosslink
//! #1771): sparse GPU dispatch truncates indices and pointers to 32 bits
//! without validation (CLASS-S — public representations store `usize`,
//! but every cuSPARSE dispatch site converted with unchecked `as u32`,
//! so a valid index above `u32::MAX` wraps to an unrelated coordinate
//! before reaching the backend).
//!
//! Observed at HEAD (red run, 2026-06-12, rev 74099dd19 + waves 1-4 of
//! this dispatch, `--features gpu`, RTX 3090): `CsrTensor::from_coo_on`
//! on a COO holding column index `2^33 + 1` returned
//! `Ok(([0, 1], [1]))` — `(2^33 + 1) as u32 == 1` wraparound (silent
//! metadata corruption, no error); `from_csr_on`/`to_csr_on` failed only
//! incidentally ("csr_to_csc_f32: shape mismatch, expected [2147483647],
//! got [8589934594]") instead of a structured index-limit error.
//!
//! Contract semantics (torch upstream): sparse indices are int64
//! (`aten/src/ATen/native/sparse/SparseTensor.cpp` — COO/CSR indices are
//! `kLong`); a backend lane that only supports 32-bit indices must
//! reject, never wrap. ferrotorch post-fix: every sparse GPU dispatch
//! site converts via a checked helper and returns a structured
//! `InvalidArgument` naming the offending quantity; CPU lanes keep
//! accepting 64-bit-class indices.

#![allow(clippy::unreadable_literal)]

use ferrotorch_core::{CooTensor, CsrTensor};

/// 2^33 + 1: wraps to 2 under `as u32`.
const BIG: usize = (1usize << 33) + 1;

/// CPU lanes must keep accepting >u32 indices (the metadata is `usize`;
/// only the 32-bit GPU descriptor lane is constrained).
#[test]
fn core077_cpu_paths_keep_large_indices() {
    let coo = CooTensor::<f32>::new(vec![0], vec![BIG], vec![7.0], 1, BIG + 1)
        .expect("COO with a >u32 column index is valid metadata");
    let csr = CsrTensor::from_coo(&coo).expect("CPU from_coo");
    assert_eq!(
        csr.col_indices(),
        &[BIG],
        "CPU conversion must not truncate"
    );
    assert_eq!(csr.row_ptrs(), &[0, 1]);
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use ferrotorch_core::{CscTensor, Device, FerrotorchError};
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for the GPU lane");
        });
    }

    /// `CsrTensor::from_coo_on(Cuda)` with a >u32 column index must
    /// return a structured error, not wrapped metadata. Pre-fix this
    /// returned `Ok` with the column index silently truncated.
    #[test]
    fn core077_gpu_from_coo_on_rejects_oversize_col_index() {
        ensure_cuda_backend();
        let coo = CooTensor::<f32>::new(vec![0], vec![BIG], vec![7.0], 1, BIG + 1)
            .expect("valid COO metadata");
        let r = CsrTensor::from_coo_on(&coo, Device::Cuda(0));
        match r {
            Err(FerrotorchError::InvalidArgument { ref message }) => {
                assert!(
                    message.contains("u32"),
                    "error must name the 32-bit index limit, got: {message}"
                );
            }
            other => panic!(
                "from_coo_on(Cuda) with col index {BIG} must reject (u32 wrap), got {:?}",
                other.map(|c| (c.row_ptrs().to_vec(), c.col_indices().to_vec()))
            ),
        }
    }

    /// `CscTensor::from_csr_on(Cuda)` with a >u32 column index in the
    /// source CSR must reject rather than wrap.
    #[test]
    fn core077_gpu_from_csr_on_rejects_oversize_col_index() {
        ensure_cuda_backend();
        let csr = CsrTensor::<f32>::new(vec![0, 1], vec![BIG], vec![7.0], 1, BIG + 1)
            .expect("valid CSR metadata");
        let r = CscTensor::from_csr_on(&csr, Device::Cuda(0));
        match r {
            Err(FerrotorchError::InvalidArgument { ref message }) => {
                assert!(message.contains("u32"), "got: {message}");
            }
            other => panic!(
                "from_csr_on(Cuda) with col index {BIG} must reject, got {:?}",
                other.map(|c| c.col_ptrs().len())
            ),
        }
    }

    /// `CscTensor::to_csr_on(Cuda)` with a >u32 row index must reject
    /// rather than wrap.
    #[test]
    fn core077_gpu_to_csr_on_rejects_oversize_row_index() {
        ensure_cuda_backend();
        let csc = CscTensor::<f32>::new(vec![0, 1], vec![BIG], vec![7.0], BIG + 1, 1)
            .expect("valid CSC metadata");
        let r = csc.to_csr_on(Device::Cuda(0));
        match r {
            Err(FerrotorchError::InvalidArgument { ref message }) => {
                assert!(message.contains("u32"), "got: {message}");
            }
            other => panic!(
                "to_csr_on(Cuda) with row index {BIG} must reject, got {:?}",
                other.map(|c| c.col_indices().to_vec())
            ),
        }
    }

    /// In-range indices keep flowing through the checked conversions:
    /// the small-fixture GPU round-trip must stay green.
    #[test]
    fn core077_gpu_checked_conversion_passes_small_indices() {
        ensure_cuda_backend();
        let coo =
            CooTensor::<f32>::new(vec![0, 1], vec![1, 0], vec![3.0, 4.0], 2, 2).expect("small COO");
        let csr = CsrTensor::from_coo_on(&coo, Device::Cuda(0)).expect("gpu from_coo_on");
        assert_eq!(csr.row_ptrs(), &[0, 1, 2]);
        assert_eq!(csr.col_indices(), &[1, 0]);
    }
}
