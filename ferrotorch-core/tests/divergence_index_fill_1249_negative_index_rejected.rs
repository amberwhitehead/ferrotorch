//! Divergence test: ferrotorch `grad_fns::indexing::index_fill` rejects
//! negative index values; upstream `torch.index_fill` accepts them with
//! standard Python-style wrap-around semantics.
//!
//! Builder claim in commit c3c1fd57c: "16 skips are for: ... negative
//! index values (ferrotorch IntTensor convention)."
//!
//! The claim that negative index rejection is a legitimate "narrower
//! contract" is a citation of an internal ferrotorch convention without
//! upstream support. Upstream `torch.index_fill` accepts negative indices
//! and applies `idx + dim_size` wrap-around (verified against live torch):
//!
//!     >>> x = torch.tensor([[1.,2.,3.],[4.,5.,6.]])
//!     >>> torch.index_fill(x, 1, torch.tensor([-1]), -1.0)
//!     tensor([[ 1.,  2., -1.],
//!             [ 4.,  5., -1.]])
//!     >>> torch.index_fill(x, 1, torch.tensor([-3]), -1.0)
//!     tensor([[-1.,  2.,  3.],
//!             [-1.,  5.,  6.]])
//!
//! Only out-of-range negatives (e.g. `-4` for a size-3 axis) raise
//! `IndexError`. ferrotorch instead hard-errors on every negative index at
//! `ferrotorch-core/src/grad_fns/indexing.rs:1517-1520`:
//!
//!     if i < 0 {
//!         return Err(FerrotorchError::InvalidArgument { ... });
//!     }
//!
//! This is a real user-visible API divergence — code that worked against
//! `torch.index_fill` breaks against `ferrotorch::index_fill`. Per goal.md
//! R-DEFER-3 there is no "acceptable drift"; this must be fixed.
//!
//! Tracking: blocker (filed by acto-critic).

use ferrotorch_core::grad_fns::indexing::index_fill;
use ferrotorch_core::int_tensor::IntTensor;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

fn cpu_f32(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data), shape, false).unwrap()
}

/// Upstream behavior (confirmed by live torch run): negative index wraps
/// modulo `dim_size`. Index `-1` along an axis of size 3 == index `2`.
#[test]
fn index_fill_negative_index_must_wrap_per_upstream_semantics() {
    // x = [[1,2,3],[4,5,6]]  along dim=1 of size 3
    let x = cpu_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let idx_neg_one: IntTensor<i64> = IntTensor::from_vec(vec![-1_i64], vec![1]).unwrap();
    let out = index_fill(&x, 1, &idx_neg_one, -1.0).expect(
        "torch.index_fill accepts negative indices with wrap-around per \
         TensorAdvancedIndexing.cpp:1917 + iterator-level bound check; \
         ferrotorch must mirror, not impose a narrower IntTensor convention",
    );
    let data = out.data().expect("data");
    // Upstream live-torch oracle:
    //   torch.index_fill(x, 1, tensor([-1]), -1.0)
    //     == tensor([[ 1.,  2., -1.], [ 4.,  5., -1.]])
    assert_eq!(
        data,
        &[1.0_f32, 2.0, -1.0, 4.0, 5.0, -1.0],
        "negative index -1 must wrap to dim_size-1 = 2"
    );
}

/// Out-of-range negative (`-4` for size-3 axis) must still raise — only
/// in-range negatives wrap. This guards against a fix that strips the
/// negative check entirely without restoring bounds for the wrapped index.
#[test]
fn index_fill_out_of_range_negative_index_must_still_error() {
    let x = cpu_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let idx_oob: IntTensor<i64> = IntTensor::from_vec(vec![-4_i64], vec![1]).unwrap();
    let result = index_fill(&x, 1, &idx_oob, -1.0);
    assert!(
        result.is_err(),
        "torch.index_fill raises IndexError on idx=-4 for size-3 axis; \
         ferrotorch must error too (but only on the OOB path, not on \
         in-range negatives like -1, -2, -3)"
    );
}
