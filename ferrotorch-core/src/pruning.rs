//! ## REQ status (per `.design/ferrotorch-core/pruning.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | impl `magnitude_prune` (Python round-half-even prune count + torch CPU topk tie order — CORE-083 -> #1777, #1908); non-test consumer re-exported at `lib.rs:178` for `ferrotorch-nn` callers. |
//! | REQ-2 | SHIPPED | impl `apply_2_4_mask` (final-dim grouping, `InvalidArgument` when last dim % 4 != 0 — CORE-084 -> #1778; torch CPU topk in-block tie order — #1910); non-test consumer cross-checked at `sparse.rs` + re-exported at `lib.rs:178`. |
//! | REQ-3 | SHIPPED | impl `sparsity_ratio`; non-test consumer re-exported at `lib.rs:178`. |
//! | REQ-4 | SHIPPED | validation guard in `magnitude_prune`; non-test consumer part of pub-function contract. |
//! | REQ-5 | SHIPPED | differentiable mask multiplication via `apply_constant_mask` -> `grad_fns::arithmetic::mul` (`MulBackward` edge to the original parameter — CORE-082 -> #1776); non-test consumer sparse-finetune workflows. |

use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::storage::TensorStorage;
use crate::tensor::Tensor;

// ---------------------------------------------------------------------------
// torch CPU bottom-k selection-order port (CORE-083 -> #1777)
// ---------------------------------------------------------------------------
//
// `torch.nn.utils.prune.l1_unstructured` computes its mask via
// `torch.topk(torch.abs(t).view(-1), k=nparams_toprune, largest=False)` and
// scatters zeros at the returned indices
// (`pytorch/torch/nn/utils/prune.py:349-358`, `L1Unstructured.compute_mask`).
// CPU topk selects with `std::partial_sort` when `k * 64 <= dim_size`,
// otherwise `std::nth_element` at `k - 1`, using the NaN-aware comparator
// `(!isnan(x) && isnan(y)) || x < y` (NaN sorts LAST for `largest=False`) —
// `pytorch/aten/src/ATen/native/TopKImpl.h:44-88`. Which members of a
// magnitude TIE get pruned therefore follows libstdc++'s heap-select /
// introselect order, and matching torch bit-for-bit (the CLASS-V DoD)
// requires porting exactly those selection algorithms (GNU libstdc++
// `bits/stl_algo.h` `__introselect` and `bits/stl_heap.h` `__heap_select`
// family) over `(magnitude, index)` pairs.
//
// Only SET membership of the first `k` slots feeds the mask, so the final
// in-place orderings (`std::sort_heap` inside `partial_sort`, the
// `if (sorted) std::sort(..)` re-sort in `TopKImpl.h`) are omitted: they
// permute `queue[0..k]` but never change which elements are in it.
//
// Verified live on torch 2.11.0+cu130:
//   topk([1,1,1,1], k=2, largest=False).indices  -> [2, 3]
//   topk([1,2,2,3], k=2, largest=False).indices  -> [0, 1]
//   topk(|[2,-2,2,5,-2,7]|, k=3, largest=False).indices -> [2, 4, 0]
//   topk([1,1,1,1], k=1, largest=False).indices  -> [2]

/// `largest=False` comparator from `TopKImpl.h:78-79`:
/// `(!isnan(x) && isnan(y)) || (x.first < y.first)`. Values only; the index
/// component never participates in the comparison.
#[inline]
fn bottomk_lt<T: Float>(x: &(T, usize), y: &(T, usize)) -> bool {
    (!x.0.is_nan() && y.0.is_nan()) || x.0 < y.0
}

/// `std::__push_heap` (libstdc++ `bits/stl_heap.h`): sift `value` up from
/// `hole` toward `top` in the max-heap rooted at `top`.
fn sift_up<T: Float>(v: &mut [(T, usize)], mut hole: usize, top: usize, value: (T, usize)) {
    while hole > top {
        let parent = (hole - 1) / 2;
        if !bottomk_lt(&v[parent], &value) {
            break;
        }
        v[hole] = v[parent];
        hole = parent;
    }
    v[hole] = value;
}

/// `std::__adjust_heap` (libstdc++ `bits/stl_heap.h`): walk the hole down to
/// a leaf along the larger-child path, then sift `value` back up.
fn adjust_heap<T: Float>(v: &mut [(T, usize)], mut hole: usize, len: usize, value: (T, usize)) {
    let top = hole;
    let mut second = hole;
    while second < (len - 1) / 2 {
        second = 2 * (second + 1);
        if bottomk_lt(&v[second], &v[second - 1]) {
            second -= 1;
        }
        v[hole] = v[second];
        hole = second;
    }
    if len & 1 == 0 && second == (len - 2) / 2 {
        second = 2 * (second + 1);
        v[hole] = v[second - 1];
        hole = second - 1;
    }
    sift_up(v, hole, top, value);
}

/// `std::__make_heap` (libstdc++ `bits/stl_heap.h`) over `v[..len]`.
fn make_heap<T: Float>(v: &mut [(T, usize)], len: usize) {
    if len < 2 {
        return;
    }
    let mut parent = (len - 2) / 2;
    loop {
        let value = v[parent];
        adjust_heap(v, parent, len, value);
        if parent == 0 {
            return;
        }
        parent -= 1;
    }
}

/// `std::__heap_select(first, middle, last)` (libstdc++ `bits/stl_algo.h`)
/// over the subrange `v[..]` with `middle` relative to the slice start:
/// after the call, `v[..middle]` holds the `middle` smallest elements
/// (as a max-heap; internal order irrelevant to callers here).
fn heap_select<T: Float>(v: &mut [(T, usize)], middle: usize) {
    make_heap(v, middle);
    for i in middle..v.len() {
        if bottomk_lt(&v[i], &v[0]) {
            // `std::__pop_heap(first, middle, i)`.
            let value = v[i];
            v[i] = v[0];
            adjust_heap(&mut v[..middle], 0, middle, value);
        }
    }
}

/// `std::__insertion_sort` (libstdc++ `bits/stl_algo.h`) over the slice.
/// Stable for ties, exactly like the original.
fn insertion_sort<T: Float>(v: &mut [(T, usize)]) {
    for i in 1..v.len() {
        let value = v[i];
        if bottomk_lt(&value, &v[0]) {
            for j in (1..=i).rev() {
                v[j] = v[j - 1];
            }
            v[0] = value;
        } else {
            // `std::__unguarded_linear_insert`.
            let mut j = i;
            while bottomk_lt(&value, &v[j - 1]) {
                v[j] = v[j - 1];
                j -= 1;
            }
            v[j] = value;
        }
    }
}

/// `std::__move_median_to_first` (libstdc++ `bits/stl_algo.h`).
fn move_median_to_first<T: Float>(
    v: &mut [(T, usize)],
    result: usize,
    a: usize,
    b: usize,
    c: usize,
) {
    if bottomk_lt(&v[a], &v[b]) {
        if bottomk_lt(&v[b], &v[c]) {
            v.swap(result, b);
        } else if bottomk_lt(&v[a], &v[c]) {
            v.swap(result, c);
        } else {
            v.swap(result, a);
        }
    } else if bottomk_lt(&v[a], &v[c]) {
        v.swap(result, a);
    } else if bottomk_lt(&v[b], &v[c]) {
        v.swap(result, c);
    } else {
        v.swap(result, b);
    }
}

/// `std::__unguarded_partition(first, last, pivot)` (libstdc++
/// `bits/stl_algo.h`). The pivot element sits at index `pivot`, outside
/// `[first, last)`, so the inner scans need no bounds guards.
fn unguarded_partition<T: Float>(
    v: &mut [(T, usize)],
    mut first: usize,
    mut last: usize,
    pivot: usize,
) -> usize {
    loop {
        while bottomk_lt(&v[first], &v[pivot]) {
            first += 1;
        }
        last -= 1;
        while bottomk_lt(&v[pivot], &v[last]) {
            last -= 1;
        }
        if first >= last {
            return first;
        }
        v.swap(first, last);
        first += 1;
    }
}

/// `std::nth_element` == `std::__introselect` with `depth_limit = 2*__lg(n)`
/// (libstdc++ `bits/stl_algo.h`): median-of-3 quickselect with an
/// insertion-sort tail (ranges of <= 3) and a heap-select fallback when the
/// recursion-depth budget is exhausted.
fn nth_element<T: Float>(v: &mut [(T, usize)], nth: usize) {
    let len = v.len();
    if len == 0 || nth == len {
        return;
    }
    // `std::__lg(n) * 2`.
    let mut depth_limit = 2 * (usize::BITS - 1 - len.leading_zeros()) as usize;
    let mut first = 0usize;
    let mut last = len;
    while last - first > 3 {
        if depth_limit == 0 {
            // `std::__heap_select(first, nth + 1, last)` then
            // `std::iter_swap(first, nth)`.
            heap_select(&mut v[first..last], nth + 1 - first);
            v.swap(first, nth);
            return;
        }
        depth_limit -= 1;
        // `std::__unguarded_partition_pivot`.
        let mid = first + (last - first) / 2;
        move_median_to_first(v, first, first + 1, mid, last - 1);
        let cut = unguarded_partition(v, first + 1, last, first);
        if cut <= nth {
            first = cut;
        } else {
            last = cut;
        }
    }
    insertion_sort(&mut v[first..last]);
}

/// The index SET torch CPU `topk(values, k, largest=False)` selects —
/// `pytorch/aten/src/ATen/native/TopKImpl.h:44-88` ported faithfully so the
/// kept/pruned split among magnitude ties is bit-identical to torch.
///
/// Preconditions: `1 <= k <= values.len()`.
fn torch_cpu_bottomk_indices<T: Float>(values: &[T], k: usize) -> Vec<usize> {
    debug_assert!(k >= 1 && k <= values.len());
    let mut queue: Vec<(T, usize)> = values.iter().copied().zip(0..).collect();
    // `use_partial_sort = k * 64 <= n` (TopKImpl.h:44).
    if (k as u128) * 64 <= values.len() as u128 {
        // `std::partial_sort` = `__heap_select` + `__sort_heap`; the sort
        // only permutes the already-selected prefix.
        heap_select(&mut queue, k);
    } else {
        nth_element(&mut queue, k - 1);
    }
    queue[..k].iter().map(|&(_, i)| i).collect()
}

/// Unstructured magnitude pruning: zero out the smallest weights.
///
/// Given a weight tensor and a sparsity fraction in `[0, 1)`, zeroes
/// EXACTLY `round(sparsity * numel)` elements using Python's
/// round-half-to-even rule — the smallest by absolute value, with ties at
/// the cut split exactly as
/// `torch.nn.utils.prune.l1_unstructured` does (torch CPU
/// `topk(|w|, k, largest=False)` selection order; see
/// `torch_cpu_bottomk_indices`).
///
/// # Arguments
///
/// * `weights` - The weight tensor to prune.
/// * `sparsity` - Fraction of elements to zero out (e.g. 0.5 for 50% sparsity).
///
/// # Returns
///
/// `weights * mask` where `mask` is a constant 0/1 tensor — the same
/// differentiable masked-multiplication PyTorch's pruning parametrization
/// applies (`weight = weight_orig * weight_mask`,
/// `pytorch/torch/nn/utils/prune.py` `BasePruningMethod.apply_mask`).
/// When the input requires grad, backward delivers `grad * mask` to the
/// ORIGINAL parameter: exact zeros at pruned slots, pass-through at kept
/// slots (CORE-082 -> #1776).
pub fn magnitude_prune<T: Float>(
    weights: &Tensor<T>,
    sparsity: f64,
) -> FerrotorchResult<Tensor<T>> {
    if !(0.0..1.0).contains(&sparsity) {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("sparsity must be in [0, 1), got {sparsity}"),
        });
    }

    let data = weights.data()?;
    let numel = data.len();
    let n_prune = ((numel as f64) * sparsity).round_ties_even() as usize;

    let mut mask = vec![<T as num_traits::One>::one(); numel];
    if n_prune > 0 {
        let magnitudes: Vec<T> = data.iter().map(|&v| v.abs()).collect();
        for idx in torch_cpu_bottomk_indices(&magnitudes, n_prune) {
            mask[idx] = <T as num_traits::Zero>::zero();
        }
    }

    apply_constant_mask(weights, mask)
}

/// `weights * mask` with `mask` a constant (non-tracking) 0/1 tensor, via
/// the real `mul` op so a `MulBackward` edge connects the output to the
/// original parameter (torch's `weight = weight_orig * weight_mask`
/// parametrization). The multiplication also reproduces torch's value
/// semantics at pruned slots exactly: `(-w) * 0.0 == -0.0` (sign
/// preserved) and `NaN * 0.0 == NaN`.
fn apply_constant_mask<T: Float>(weights: &Tensor<T>, mask: Vec<T>) -> FerrotorchResult<Tensor<T>> {
    let mask_t = Tensor::from_storage(TensorStorage::cpu(mask), weights.shape().to_vec(), false)?;
    crate::grad_fns::arithmetic::mul(weights, &mask_t)
}

/// Apply 2:4 structured sparsity mask.
///
/// Groups of 4 are formed along the FINAL dimension only (the innermost-dim
/// layout semi-structured sparse kernels require; groups never span row
/// boundaries). For every group of 4, keeps the 2 with the largest
/// magnitude and zeros the other 2.
///
/// # Errors
///
/// Returns `InvalidArgument` when the final dimension is not a multiple of
/// 4 (including scalars). This matches the PyTorch oracle:
/// `torch.ao.pruning.WeightNormSparsifier(sparsity_level=1.0,
/// sparse_block_shape=(1,4), zeros_per_block=2)` rejects such shapes
/// (live torch 2.11.0+cu130: `AssertionError: mask shape
/// (torch.Size([2, 8])) must match x shape (torch.Size([2, 6]))`) —
/// CORE-084 -> #1778.
///
/// # Arguments
///
/// * `weights` - The weight tensor to apply the mask to.
///
/// # Returns
///
/// `weights * mask` with a real `MulBackward` edge (see `magnitude_prune`):
/// when the input requires grad, backward delivers `grad * mask` to the
/// original parameter (CORE-082 -> #1776).
pub fn apply_2_4_mask<T: Float>(weights: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let data = weights.data()?;
    let shape = weights.shape();

    let last_dim = *shape
        .last()
        .ok_or_else(|| FerrotorchError::InvalidArgument {
            message: "apply_2_4_mask: scalar (0-d) tensors cannot carry a 2:4 \
                  sparsity pattern"
                .to_string(),
        })?;
    if !last_dim.is_multiple_of(4) {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "apply_2_4_mask: final dimension must be a multiple of 4 for \
                 the 2:4 semi-structured layout, got shape {shape:?} (final \
                 dimension {last_dim}); PyTorch's WeightNormSparsifier \
                 (sparse_block_shape=(1,4)) rejects this shape too"
            ),
        });
    }

    // Build the constant 0/1 mask. The storage is contiguous row-major and
    // `last_dim % 4 == 0`, so 4-element groups taken row by row are exactly
    // the contiguous chunks of the flat buffer — but never crossing a row
    // boundary (CORE-084).
    let mut mask = vec![<T as num_traits::One>::one(); data.len()];
    let row_count = data.len().checked_div(last_dim).unwrap_or(0);
    for r in 0..row_count {
        let row = &data[r * last_dim..(r + 1) * last_dim];
        let mask_row = &mut mask[r * last_dim..(r + 1) * last_dim];
        for (group, mask_group) in row.chunks_exact(4).zip(mask_row.chunks_exact_mut(4)) {
            // WeightNormSparsifier's default norm is L2 (`w * w`) and it
            // zeroes `torch.topk(scores, k=2, largest=False).indices` inside
            // each sparse block.
            let scores: Vec<T> = group.iter().map(|&v| v * v).collect();
            for idx in torch_cpu_bottomk_indices(&scores, 2) {
                mask_group[idx] = <T as num_traits::Zero>::zero();
            }
        }
    }

    apply_constant_mask(weights, mask)
}

/// Compute the sparsity ratio of a tensor: fraction of exact zeros.
pub fn sparsity_ratio<T: Float>(tensor: &Tensor<T>) -> FerrotorchResult<f64> {
    let data = tensor.data()?;
    let zeros = data
        .iter()
        .filter(|&&v| v == <T as num_traits::Zero>::zero())
        .count();
    Ok(zeros as f64 / data.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tensor(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data), shape, false).unwrap()
    }

    fn make_tensor_rg(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data), shape, true).unwrap()
    }

    // --- magnitude_prune ---

    #[test]
    // reason: pruning is select-or-zero — kept slots hold the exact input
    // bit pattern (no arithmetic), pruned slots hold the exact zero bit
    // pattern. Equality is the right check.
    #[allow(clippy::float_cmp)]
    fn test_magnitude_prune_50_percent() {
        let t = make_tensor(vec![1.0, -4.0, 2.0, -3.0], vec![4]);
        let pruned = magnitude_prune(&t, 0.5).unwrap();
        let d = pruned.data().unwrap();

        // 50% of 4 = EXACTLY 2 elements pruned: the two smallest
        // magnitudes (|1| and |2|).
        assert_eq!(d[0], 0.0); // |1| pruned
        assert_eq!(d[1], -4.0); // |4| kept
        assert_eq!(d[2], 0.0); // |2| pruned
        assert_eq!(d[3], -3.0); // |3| kept
    }

    #[test]
    // reason: select-or-zero, exact bit patterns (see above).
    #[allow(clippy::float_cmp)]
    fn test_magnitude_prune_ties_prune_exact_count() {
        // CORE-083 (#1777) regression: ties at the cut must NOT all vanish.
        // Live torch 2.11.0+cu130 oracle (R-ORACLE-1b):
        //   >>> m.weight = nn.Parameter(torch.tensor([1.,1.,1.,1.]))
        //   >>> prune.l1_unstructured(m, "weight", 0.25)
        //   >>> m.weight
        //   tensor([1., 1., 0., 1.], ...)
        // (torch CPU topk(largest=False) k=1 picks index 2 among the ties.)
        let t = make_tensor(vec![1.0, 1.0, 1.0, 1.0], vec![4]);
        let pruned = magnitude_prune(&t, 0.25).unwrap();
        let d = pruned.data().unwrap();
        assert_eq!(d, &[1.0, 1.0, 0.0, 1.0]);

        // Live torch: prune.l1_unstructured([1,1,1,1], 0.5) -> [1,1,0,0]
        // (topk k=2 largest=False indices == [2, 3]).
        let pruned = magnitude_prune(&t, 0.5).unwrap();
        let d = pruned.data().unwrap();
        assert_eq!(d, &[1.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_magnitude_prune_zero_sparsity() {
        let t = make_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let pruned = magnitude_prune(&t, 0.0).unwrap();
        let d = pruned.data().unwrap();
        assert_eq!(d, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_magnitude_prune_count_rounds_half_to_even() {
        let t = make_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![4]);

        // torch.nn.utils.prune._compute_nparams_toprune uses Python round:
        // 0.125*4 = 0.5 -> 0, 0.375*4 = 1.5 -> 2,
        // 0.625*4 = 2.5 -> 2, 0.875*4 = 3.5 -> 4.
        let pruned = magnitude_prune(&t, 0.125).unwrap();
        assert_eq!(pruned.data().unwrap(), &[1.0, 2.0, 3.0, 4.0]);

        let pruned = magnitude_prune(&t, 0.375).unwrap();
        assert_eq!(pruned.data().unwrap(), &[0.0, 0.0, 3.0, 4.0]);

        let pruned = magnitude_prune(&t, 0.625).unwrap();
        assert_eq!(pruned.data().unwrap(), &[0.0, 0.0, 3.0, 4.0]);

        let pruned = magnitude_prune(&t, 0.875).unwrap();
        assert_eq!(pruned.data().unwrap(), &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_magnitude_prune_invalid_sparsity() {
        let t = make_tensor(vec![1.0], vec![1]);
        assert!(magnitude_prune(&t, 1.0).is_err());
        assert!(magnitude_prune(&t, -0.1).is_err());
    }

    // --- NaN edge case for pruning (Issue 11) ---

    #[test]
    fn test_magnitude_prune_nan_no_panic() {
        let t = make_tensor(vec![1.0, f32::NAN, 3.0, f32::NAN, 2.0, 4.0], vec![6]);
        // Should not panic even with NaN values.
        let result = magnitude_prune(&t, 0.5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_apply_2_4_mask_nan_no_panic() {
        let t = make_tensor(
            vec![1.0, f32::NAN, 3.0, f32::NAN, 2.0, 4.0, 0.5, 0.1],
            vec![8],
        );
        // Should not panic even with NaN values.
        let result = apply_2_4_mask(&t);
        assert!(result.is_ok());
    }

    // --- apply_2_4_mask ---

    #[test]
    // reason: 2:4 masking is select-or-zero — kept slots hold the exact
    // input bit pattern (no arithmetic), pruned slots hold exact zero. The
    // 0.9 and 0.8 literals on the RHS produce the same f32 bit pattern as
    // the corresponding input literals, so equality is the right check.
    #[allow(clippy::float_cmp)]
    fn test_apply_2_4_mask_basic() {
        let t = make_tensor(vec![1.0, -4.0, 2.0, -3.0, 0.5, 0.1, 0.9, 0.8], vec![8]);
        let masked = apply_2_4_mask(&t).unwrap();
        let d = masked.data().unwrap();

        // Group 0: [1, -4, 2, -3]. Magnitudes: [1, 4, 2, 3].
        // Smallest two: indices 0 (mag 1) and 2 (mag 2) -> zeroed.
        assert_eq!(d[0], 0.0);
        assert_eq!(d[1], -4.0);
        assert_eq!(d[2], 0.0);
        assert_eq!(d[3], -3.0);

        // Group 1: [0.5, 0.1, 0.9, 0.8]. Magnitudes: [0.5, 0.1, 0.9, 0.8].
        // Smallest two: indices 1 (mag 0.1) and 0 (mag 0.5) -> zeroed.
        assert_eq!(d[4], 0.0);
        assert_eq!(d[5], 0.0);
        assert_eq!(d[6], 0.9);
        assert_eq!(d[7], 0.8);
    }

    #[test]
    fn test_apply_2_4_mask_rejects_final_dim_not_multiple_of_4() {
        // CORE-084 (#1778) regression: torch's WeightNormSparsifier
        // (sparse_block_shape=(1,4)) REJECTS rows that are not a multiple
        // of 4 wide — live torch 2.11.0+cu130:
        //   AssertionError: mask shape (torch.Size([2, 8])) must match
        //   x shape (torch.Size([2, 6]))
        // ferrotorch must return a structured error, never flat-group
        // across row boundaries or leave a trailing remainder unchanged.
        let t2x6 = make_tensor(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            vec![2, 6],
        );
        assert!(matches!(
            apply_2_4_mask(&t2x6),
            Err(FerrotorchError::InvalidArgument { .. })
        ));

        let t6 = make_tensor(vec![1.0, -4.0, 2.0, -3.0, 0.5, 0.1], vec![6]);
        assert!(matches!(
            apply_2_4_mask(&t6),
            Err(FerrotorchError::InvalidArgument { .. })
        ));

        // Scalar (0-d): no final dimension to group along.
        let t0 = make_tensor(vec![1.0], vec![]);
        assert!(matches!(
            apply_2_4_mask(&t0),
            Err(FerrotorchError::InvalidArgument { .. })
        ));
    }

    #[test]
    // reason: select-or-zero, exact bit patterns (see test_apply_2_4_mask_basic).
    #[allow(clippy::float_cmp)]
    fn test_apply_2_4_mask_groups_along_final_dim() {
        // Valid [2, 4]: each row is one group; groups never mix rows.
        // Live torch 2.11.0+cu130 WeightNormSparsifier oracle:
        //   [2,4] of [1,-4,2,-3, 0.5,0.1,0.9,0.8]
        //   -> [0,-4,0,-3, 0,0,0.9,0.8]
        let t = make_tensor(vec![1.0, -4.0, 2.0, -3.0, 0.5, 0.1, 0.9, 0.8], vec![2, 4]);
        let masked = apply_2_4_mask(&t).unwrap();
        let d = masked.data().unwrap();
        assert_eq!(d[0], 0.0);
        assert_eq!(d[1], -4.0);
        assert_eq!(d[2], 0.0);
        assert_eq!(d[3], -3.0);
        assert_eq!(d[4], 0.0);
        assert_eq!(d[5], 0.0);
        assert_eq!(d[6], 0.9);
        assert_eq!(d[7], 0.8);
    }

    #[test]
    fn test_apply_2_4_mask_ties_match_weight_norm_sparsifier() {
        // Live torch 2.11.0+cu130 WeightNormSparsifier oracle:
        //   [2,2,2,2] -> [2,2,0,0]
        let t = make_tensor(vec![2.0, 2.0, 2.0, 2.0], vec![4]);
        let masked = apply_2_4_mask(&t).unwrap();
        assert_eq!(masked.data().unwrap(), &[2.0, 2.0, 0.0, 0.0]);

        //   [1,3,3,3] -> [0,3,0,3]
        let t = make_tensor(vec![1.0, 3.0, 3.0, 3.0], vec![4]);
        let masked = apply_2_4_mask(&t).unwrap();
        assert_eq!(masked.data().unwrap(), &[0.0, 3.0, 0.0, 3.0]);

        //   [-2,2,-2,2] -> [-2,2,-0,0]
        let t = make_tensor(vec![-2.0, 2.0, -2.0, 2.0], vec![4]);
        let masked = apply_2_4_mask(&t).unwrap();
        let d = masked.data().unwrap();
        let expected = [-2.0_f32, 2.0, -0.0, 0.0];
        for (actual, expected) in d.iter().zip(expected.iter()) {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn test_apply_2_4_mask_preserves_requires_grad() {
        let t = make_tensor_rg(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        assert!(t.requires_grad());

        let masked = apply_2_4_mask(&t).unwrap();
        assert!(
            masked.requires_grad(),
            "apply_2_4_mask must propagate requires_grad"
        );
    }

    // --- sparsity_ratio ---

    #[test]
    fn test_sparsity_ratio() {
        let t = make_tensor(vec![0.0, 1.0, 0.0, 2.0], vec![4]);
        let ratio = sparsity_ratio(&t).unwrap();
        assert!((ratio - 0.5).abs() < 1e-10);
    }
}
