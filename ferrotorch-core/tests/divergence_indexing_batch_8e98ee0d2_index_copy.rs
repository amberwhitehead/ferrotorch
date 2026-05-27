//! Divergences in commit `8e98ee0d2` (#1248 index_copy):
//!
//! D6: `pub fn index_copy` at `indexing.rs:2728-2868` SILENTLY WRAPS
//! negative index values via `idx_usize.push((i_raw + in_dim_size_i64) as
//! usize)` at lines 2807-2826. Upstream PyTorch REJECTS negative indices
//! with `index_copy_(): index -1 is out of bounds for dimension 0 with
//! size 4`. Live oracle:
//!
//!     >>> torch.index_copy(torch.tensor([1.,2.,3.,4.]), 0,
//!     ...                  torch.tensor([-1, -3]), torch.tensor([100., 200.]))
//!     RuntimeError: index_copy_(): index -1 is out of bounds for dimension 0
//!     with size 4
//!
//! Per upstream `aten/src/ATen/native/TensorAdvancedIndexing.cpp:1082
//! TORCH_IMPL_FUNC(index_copy_out)` → bounds check matches index_add's
//! kernel: negative values are rejected, not wrapped.
//!
//! This test fails against HEAD `8e98ee0d2`.

use ferrotorch_core::grad_fns::indexing::index_copy;
use ferrotorch_core::int_tensor::IntTensor;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

fn idx(d: Vec<i64>, s: Vec<usize>) -> IntTensor<i64> {
    IntTensor::from_vec(d, s).unwrap()
}

/// D6: negative index should ERROR upstream; ferrotorch silently wraps.
#[test]
fn divergence_index_copy_negative_index_silently_wraps_should_error() {
    let input = Tensor::from_storage(
        TensorStorage::cpu(vec![1.0_f32, 2.0, 3.0, 4.0]),
        vec![4],
        false,
    )
    .unwrap();
    let i = idx(vec![-1, -3], vec![2]);
    let source =
        Tensor::from_storage(TensorStorage::cpu(vec![100.0_f32, 200.0]), vec![2], false).unwrap();
    let res = index_copy(&input, 0, &i, &source);
    assert!(
        res.is_err(),
        "index_copy with negative index must error per upstream \
         `index_copy_(): index -1 is out of bounds for dimension 0 with \
         size 4`; ferrotorch silently wraps and returns a tensor"
    );
}

/// Sanity-pin: positive-idx baseline. Live oracle:
///   torch.index_copy(t([1,2,3,4]), 0, t([1,3]), t([100,200]))
///   -> tensor([1, 100, 3, 200])
#[test]
fn index_copy_positive_idx_baseline_pin() {
    let input = Tensor::from_storage(
        TensorStorage::cpu(vec![1.0_f32, 2.0, 3.0, 4.0]),
        vec![4],
        false,
    )
    .unwrap();
    let i = idx(vec![1, 3], vec![2]);
    let source =
        Tensor::from_storage(TensorStorage::cpu(vec![100.0_f32, 200.0]), vec![2], false).unwrap();
    let out = index_copy(&input, 0, &i, &source).unwrap();
    assert_eq!(out.data().unwrap(), &[1.0_f32, 100.0, 3.0, 200.0]);
}

/// D6b: index_copy with source LARGER than index along dim should ERROR
/// upstream (`index_copy_(): Number of indices ... should be equal to
/// source.size(dim)`); ferrotorch silently clamps via the same
/// `i.min(src_dim_size.saturating_sub(1))` pattern as index_add at line 2848.
#[test]
fn divergence_index_copy_source_larger_than_index_silently_truncates_should_error() {
    let input = Tensor::from_storage(
        TensorStorage::cpu(vec![1.0_f32, 2.0, 3.0, 4.0]),
        vec![4],
        false,
    )
    .unwrap();
    let i = idx(vec![0, 2], vec![2]);
    let source = Tensor::from_storage(
        TensorStorage::cpu(vec![100.0_f32, 200.0, 999.0]),
        vec![3],
        false,
    )
    .unwrap();
    let res = index_copy(&input, 0, &i, &source);
    assert!(
        res.is_err(),
        "index_copy with source.size(0)=3 and index.len()=2 must error per \
         upstream contract; ferrotorch silently truncates"
    );
}
