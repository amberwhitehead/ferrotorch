//! Red-then-green regression tests for audit finding CORE-073 (crosslink
//! #1767): CSC dense materialization overwrites duplicate entries instead
//! of summing them (CLASS-V — the CPU `to_dense_on` path assigned with
//! `=`, so the last duplicate silently won; every other sparse
//! representation in this module accumulates).
//!
//! Observed at HEAD (probe, 2026-06-12, rev 74099dd19):
//! `CscTensor::new([0,2,2], [0,0], [3.,4.], 2, 2).to_dense()` =
//! `[4.0, 0.0, 0.0, 0.0]` — last duplicate wins.
//!
//! torch oracle (live session, torch 2.11.0+cu130 — duplicates SUM):
//!
//! ```python
//! >>> c = torch.sparse_csc_tensor(torch.tensor([0,2,2]), torch.tensor([0,0]),
//! ...                             torch.tensor([3.,4.]), (2,2))
//! >>> c.to_dense()
//! tensor([[7., 0.],
//!         [0., 0.]])
//! >>> torch.sparse_coo_tensor(torch.tensor([[0,0],[0,0]]),
//! ...                         torch.tensor([3.,4.]), (2,2)).to_dense()
//! tensor([[7., 0.],
//!         [0., 0.]])   # same accumulation as COO
//! ```

use ferrotorch_core::CscTensor;

/// Duplicate row indices within a column accumulate on dense
/// materialization (torch oracle: 3 + 4 = 7).
#[test]
fn core073_csc_to_dense_sums_duplicates_f32() {
    let c = CscTensor::new(vec![0, 2, 2], vec![0, 0], vec![3.0f32, 4.0], 2, 2)
        .expect("structurally valid CSC with duplicate rows");
    let d = c.to_dense().expect("to_dense");
    let data = d.data().expect("data");
    assert!(
        (data[0] - 7.0).abs() < 1e-6,
        "duplicates must SUM (torch: 3 + 4 = 7), got {}",
        data[0]
    );
    assert!(data[1].abs() < 1e-6 && data[2].abs() < 1e-6 && data[3].abs() < 1e-6);
}

/// f64 lane, duplicates spread across distinct columns plus a triple
/// duplicate: col 0 rows [1,1,1] -> sums to 1+2+3=6 at (1,0); col 2
/// row 0 single entry.
#[test]
fn core073_csc_to_dense_sums_triple_duplicate_f64() {
    let c = CscTensor::new(
        vec![0, 3, 3, 4],
        vec![1, 1, 1, 0],
        vec![1.0f64, 2.0, 3.0, 9.0],
        2,
        3,
    )
    .expect("valid CSC");
    let d = c.to_dense().expect("to_dense");
    let data = d.data().expect("data");
    // Row-major [2, 3]: (1,0) = flat 3; (0,2) = flat 2.
    assert!(
        (data[3] - 6.0).abs() < 1e-12,
        "triple duplicate must sum to 6, got {}",
        data[3]
    );
    assert!((data[2] - 9.0).abs() < 1e-12);
    for &i in &[0usize, 1, 4, 5] {
        assert!(
            data[i].abs() < 1e-12,
            "elem {i} must stay 0, got {}",
            data[i]
        );
    }
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for the GPU lane");
        });
    }

    /// Direct-construction duplicate test on the CUDA lane: the cuSPARSE
    /// `to_dense_on(Cuda)` path must produce the same accumulated values
    /// as the CPU path (torch: CSC duplicates sum on CUDA too), and the
    /// result must stay on CUDA.
    #[test]
    fn core073_gpu_csc_to_dense_sums_duplicates() {
        ensure_cuda_backend();
        let c = CscTensor::new(vec![0, 2, 2], vec![0, 0], vec![3.0f32, 4.0], 2, 2)
            .expect("valid CSC with duplicate rows");
        let d = c
            .to_dense_on(ferrotorch_core::Device::Cuda(0))
            .expect("gpu to_dense_on");
        assert!(d.is_cuda(), "to_dense_on(Cuda) must stay on CUDA");
        let back = d.cpu().expect("gpu->cpu");
        let data = back.data().expect("data");
        assert!(
            (data[0] - 7.0).abs() < 1e-6,
            "CUDA lane duplicates must SUM (torch: 7), got {}",
            data[0]
        );
        assert!(data[1].abs() < 1e-6 && data[2].abs() < 1e-6 && data[3].abs() < 1e-6);
    }
}
