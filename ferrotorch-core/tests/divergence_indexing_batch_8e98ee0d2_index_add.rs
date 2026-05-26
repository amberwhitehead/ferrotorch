//! Divergences in commit `8e98ee0d2` (#1247 index_add):
//!
//! D3: `pub fn index_add` at `indexing.rs:2431-2597` SILENTLY WRAPS negative
//! index values via `idx_usize.push((i_raw + in_dim_size_i64) as usize)`
//! at lines 2522-2543. Upstream PyTorch REJECTS negative indices for
//! `index_add` with `IndexError: index out of range in self`. Live oracle:
//!
//!     >>> torch.index_add(torch.tensor([1.,2.,3.,4.]), 0,
//!     ...                 torch.tensor([-1, -2]), torch.tensor([10., 20.]))
//!     IndexError: index out of range in self
//!
//! Per upstream `aten/src/ATen/native/TensorAdvancedIndexing.cpp:1153
//! TORCH_IMPL_FUNC(index_add_cpu_out)` → the bounds check is
//! `TORCH_CHECK_INDEX(index_val >= 0 && index_val < self_dim_size, ...)`
//! at the kernel level. Negative indices are not wrapped.
//!
//! D4: `pub fn index_add` at `indexing.rs:2553-2582` SILENTLY CLAMPS or
//! TRUNCATES source-larger-than-index. Specifically the inner loop uses
//!     let i_clamped = i.min(src_dim_size.saturating_sub(1));
//! at line 2575, so any source element beyond `index.len()` along `dim` is
//! ignored, and any source element BEFORE that is reused (clamping if
//! src_dim_size < index.len()). Upstream PyTorch REJECTS this:
//!
//!     >>> torch.index_add(torch.tensor([1.,2.,3.,4.]), 0,
//!     ...                 torch.tensor([0, 2]), torch.tensor([10., 20., 99.]))
//!     RuntimeError: index_add_(): Number of indices (2) should be equal to
//!     source.size(dim): (3), for dim: 0
//!
//! Per upstream check around `TensorAdvancedIndexing.cpp:1260
//! TORCH_CHECK(source.dim() <= 1 || source.size(dim) == numIndices, ...)`.
//!
//! D5: `pub fn index_add` accepts 0-d source on a 1-D input. The 1-D branch
//! at `indexing.rs:2560-2566` sets `src_dim_size = 1` for 0-d source and
//! proceeds. Upstream PyTorch only allows 0-d source for 0-d input:
//!
//!     >>> torch.index_add(torch.tensor([1.,2.,3.,4.]), 0,
//!     ...                 torch.tensor([1]), torch.tensor(99.0))
//!     RuntimeError: source tensor shape must match self tensor shape,
//!     excluding the specified dimension. Got self.shape = [4]
//!     source.shape = []
//!
//! These tests fail against HEAD `8e98ee0d2`.

use ferrotorch_core::grad_fns::indexing::index_add;
use ferrotorch_core::int_tensor::IntTensor;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

fn idx(d: Vec<i64>, s: Vec<usize>) -> IntTensor<i64> {
    IntTensor::from_vec(d, s).unwrap()
}

/// D3: negative index should ERROR upstream (`IndexError: index out of range
/// in self`); ferrotorch silently wraps.
#[test]
fn divergence_index_add_negative_index_silently_wraps_should_error() {
    let input = Tensor::from_storage(
        TensorStorage::cpu(vec![1.0_f32, 2.0, 3.0, 4.0]),
        vec![4],
        false,
    )
    .unwrap();
    let i = idx(vec![-1, -2], vec![2]);
    let source =
        Tensor::from_storage(TensorStorage::cpu(vec![10.0_f32, 20.0]), vec![2], false).unwrap();
    let res = index_add(&input, 0, &i, &source, 1.0);
    assert!(
        res.is_err(),
        "index_add with negative index must error per upstream contract \
         (`IndexError: index out of range in self`); ferrotorch silently \
         wraps and returns Ok"
    );
}

/// D4: source-larger-than-index should ERROR upstream; ferrotorch silently
/// truncates/clamps.
#[test]
fn divergence_index_add_source_larger_than_index_silently_truncates_should_error() {
    let input = Tensor::from_storage(
        TensorStorage::cpu(vec![1.0_f32, 2.0, 3.0, 4.0]),
        vec![4],
        false,
    )
    .unwrap();
    let i = idx(vec![0, 2], vec![2]);
    // source has 3 elements; index has 2 -- upstream errors
    let source = Tensor::from_storage(
        TensorStorage::cpu(vec![10.0_f32, 20.0, 99.0]),
        vec![3],
        false,
    )
    .unwrap();
    let res = index_add(&input, 0, &i, &source, 1.0);
    assert!(
        res.is_err(),
        "index_add with source.size(0)=3 and index.len()=2 must error per \
         upstream check `index_add_(): Number of indices (2) should be \
         equal to source.size(dim): (3)`; ferrotorch silently truncates"
    );
}

/// D5: 0-d source on 1-D input should ERROR upstream; ferrotorch accepts.
#[test]
fn divergence_index_add_zero_d_source_on_1d_input_should_error() {
    let input = Tensor::from_storage(
        TensorStorage::cpu(vec![1.0_f32, 2.0, 3.0, 4.0]),
        vec![4],
        false,
    )
    .unwrap();
    let i = idx(vec![1], vec![1]);
    // 0-d source on 1-D input -- upstream errors with shape-mismatch
    let source = Tensor::from_storage(TensorStorage::cpu(vec![99.0_f32]), vec![], false).unwrap();
    let res = index_add(&input, 0, &i, &source, 1.0);
    assert!(
        res.is_err(),
        "index_add with 0-d source on 1-D input must error per upstream \
         `source tensor shape must match self tensor shape, excluding the \
         specified dimension. Got self.shape = [4] source.shape = []`; \
         ferrotorch accepts via the empty-shape branch at indexing.rs:2560"
    );
}

/// Sanity-pin: positive-idx + alpha works as expected (this PASSES today;
/// included as a regression pin). Live oracle:
///   torch.index_add(t([1,2,3,4]), 0, t([0,2]), t([10,20]), alpha=3.0)
///   -> tensor([31., 2., 63., 4.])
#[test]
fn index_add_alpha_positive_idx_baseline_pin() {
    let input = Tensor::from_storage(
        TensorStorage::cpu(vec![1.0_f32, 2.0, 3.0, 4.0]),
        vec![4],
        false,
    )
    .unwrap();
    let i = idx(vec![0, 2], vec![2]);
    let source =
        Tensor::from_storage(TensorStorage::cpu(vec![10.0_f32, 20.0]), vec![2], false).unwrap();
    let out = index_add(&input, 0, &i, &source, 3.0).unwrap();
    assert_eq!(out.data().unwrap(), &[31.0_f32, 2.0, 63.0, 4.0]);
}
