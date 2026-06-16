//! ## REQ status (per `.design/ferrotorch-core/pruning.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | impl `magnitude_prune` (Python round-half-even prune count + torch CPU topk tie order — CORE-083 -> #1777, #1908); non-test consumer re-exported at `lib.rs:218` for `ferrotorch-nn` callers. |
//! | REQ-2 | SHIPPED | impl `apply_2_4_mask` (final-dim grouping, `InvalidArgument` when last dim % 4 != 0 — CORE-084 -> #1778; torch CPU topk in-block tie order — #1910); non-test consumer cross-checked at `sparse.rs` + re-exported at `lib.rs:218`. |
//! | REQ-3 | SHIPPED | impl `sparsity_ratio`; non-test consumer re-exported at `lib.rs:218`. |
//! | REQ-4 | SHIPPED | validation guard in `magnitude_prune`; non-test consumer part of pub-function contract. |
//! | REQ-5 | SHIPPED | differentiable mask multiplication via `apply_constant_mask` -> `grad_fns::arithmetic::mul` (`MulBackward` edge to the original parameter — CORE-082 -> #1776); non-test consumer sparse-finetune workflows. |

use crate::dtype::{DType, Float};
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::gpu_dispatch::{GpuBackend, GpuBufferHandle};
use crate::storage::TensorStorage;
use crate::tensor::Tensor;
use crate::torch_topk_cpu::torch_cpu_topk_indices;

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
// The shared helper in `torch_topk_cpu` ports that selection order for both
// public `topk` and pruning masks. Only SET membership of the first `k` slots
// feeds the pruning mask, so callers here request `sorted=false`; this may
// permute the selected prefix in the `nth_element` branch but never changes
// which elements are pruned.
//
// Verified live on torch 2.11.0+cu130:
//   topk([1,1,1,1], k=2, largest=False).indices  -> [2, 3]
//   topk([1,2,2,3], k=2, largest=False).indices  -> [0, 1]
//   topk(|[2,-2,2,5,-2,7]|, k=3, largest=False).indices -> [2, 4, 0]
//   topk([1,1,1,1], k=1, largest=False).indices  -> [2]

/// Unstructured magnitude pruning: zero out the smallest weights.
///
/// Given a weight tensor and a sparsity fraction in `[0, 1)`, zeroes
/// EXACTLY `round(sparsity * numel)` elements using Python's
/// round-half-to-even rule — the smallest by absolute value, with ties at
/// the cut split exactly as
/// `torch.nn.utils.prune.l1_unstructured` does (torch CPU
/// `topk(|w|, k, largest=False)` selection order; see
/// `torch_cpu_topk_indices`).
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

    if weights.is_cuda() {
        return magnitude_prune_cuda(weights, sparsity);
    }

    let data = weights.data()?;
    let numel = data.len();
    let n_prune = ((numel as f64) * sparsity).round_ties_even() as usize;

    let mut mask = vec![<T as num_traits::One>::one(); numel];
    if n_prune > 0 {
        let magnitudes: Vec<T> = data.iter().map(|&v| v.abs()).collect();
        for idx in torch_cpu_topk_indices(&magnitudes, n_prune, false, false) {
            mask[idx] = <T as num_traits::Zero>::zero();
        }
    }

    apply_constant_mask(weights, mask)
}

fn magnitude_prune_cuda<T: Float>(
    weights: &Tensor<T>,
    sparsity: f64,
) -> FerrotorchResult<Tensor<T>> {
    let numel = crate::shape::checked_numel(weights.shape(), "magnitude_prune")?;
    let n_prune = ((numel as f64) * sparsity).round_ties_even() as usize;
    let backend = crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
    let ordinal = weights.gpu_handle()?.device_ordinal();

    let mut mask = fill_cuda_mask::<T>(backend, numel, 1.0, ordinal)?;
    if n_prune > 0 {
        let weights_c = weights.contiguous()?;
        let scores = abs_handle_cuda::<T>(backend, weights_c.gpu_handle()?)?;
        let (_values, prune_indices) = backend.topk_nd(&scores, 1, numel, 1, n_prune, false)?;
        mask =
            scatter_zero_mask_cuda::<T>(backend, &mask, &prune_indices, &[numel], &[n_prune], 0)?;
    }

    apply_cuda_mask(weights, mask)
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

fn fill_cuda_mask<T: Float>(
    backend: &dyn GpuBackend,
    numel: usize,
    value: f32,
    ordinal: usize,
) -> FerrotorchResult<GpuBufferHandle> {
    match T::dtype() {
        DType::F32 => backend.fill_f32(numel, value, ordinal),
        DType::F64 => backend.fill_f64(numel, value as f64, ordinal),
        DType::F16 => backend.fill_f16(numel, value, ordinal),
        DType::BF16 => backend.fill_bf16_bf16(numel, value, ordinal),
        dtype => Err(FerrotorchError::InvalidArgument {
            message: format!("pruning CUDA mask: unsupported floating dtype {dtype}"),
        }),
    }
}

fn abs_handle_cuda<T: Float>(
    backend: &dyn GpuBackend,
    handle: &GpuBufferHandle,
) -> FerrotorchResult<GpuBufferHandle> {
    match T::dtype() {
        DType::F32 => backend.abs_f32(handle),
        DType::F64 => backend.abs_f64(handle),
        DType::F16 => backend.abs_f16(handle),
        DType::BF16 => backend.abs_bf16_bf16(handle),
        dtype => Err(FerrotorchError::InvalidArgument {
            message: format!("pruning CUDA abs: unsupported floating dtype {dtype}"),
        }),
    }
}

fn squared_scores_cuda<T: Float>(
    backend: &dyn GpuBackend,
    handle: &GpuBufferHandle,
) -> FerrotorchResult<GpuBufferHandle> {
    match T::dtype() {
        DType::F32 => backend.mul_f32(handle, handle),
        DType::F64 => backend.mul_f64(handle, handle),
        DType::F16 => backend.mul_f16(handle, handle),
        DType::BF16 => backend.mul_bf16_bf16(handle, handle),
        dtype => Err(FerrotorchError::InvalidArgument {
            message: format!("pruning CUDA squared scores: unsupported floating dtype {dtype}"),
        }),
    }
}

fn scatter_zero_mask_cuda<T: Float>(
    backend: &dyn GpuBackend,
    mask: &GpuBufferHandle,
    indices: &GpuBufferHandle,
    input_shape: &[usize],
    index_shape: &[usize],
    dim: usize,
) -> FerrotorchResult<GpuBufferHandle> {
    match T::dtype() {
        DType::F32 => {
            backend.scatter_value_nd_f32(mask, indices, 0.0, input_shape, index_shape, dim)
        }
        DType::F64 => {
            backend.scatter_value_nd_f64(mask, indices, 0.0, input_shape, index_shape, dim)
        }
        DType::F16 => {
            backend.scatter_value_nd_f16(mask, indices, 0.0, input_shape, index_shape, dim)
        }
        DType::BF16 => {
            backend.scatter_value_nd_bf16(mask, indices, 0.0, input_shape, index_shape, dim)
        }
        dtype => Err(FerrotorchError::InvalidArgument {
            message: format!("pruning CUDA scatter mask: unsupported floating dtype {dtype}"),
        }),
    }
}

fn apply_cuda_mask<T: Float>(
    weights: &Tensor<T>,
    mask: GpuBufferHandle,
) -> FerrotorchResult<Tensor<T>> {
    let mask_t = Tensor::from_storage(TensorStorage::gpu(mask), weights.shape().to_vec(), false)?;
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

    if weights.is_cuda() {
        return apply_2_4_mask_cuda(weights, last_dim);
    }

    let data = weights.data()?;

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
            for idx in torch_cpu_topk_indices(&scores, 2, false, false) {
                mask_group[idx] = <T as num_traits::Zero>::zero();
            }
        }
    }

    apply_constant_mask(weights, mask)
}

fn apply_2_4_mask_cuda<T: Float>(
    weights: &Tensor<T>,
    last_dim: usize,
) -> FerrotorchResult<Tensor<T>> {
    let numel = crate::shape::checked_numel(weights.shape(), "apply_2_4_mask")?;
    let groups_per_row = last_dim / 4;
    let row_count = numel.checked_div(last_dim).unwrap_or(0);
    let num_groups = row_count
        .checked_mul(groups_per_row)
        .ok_or_else(|| FerrotorchError::InvalidArgument {
            message: format!(
                "apply_2_4_mask: row_count {row_count} * groups_per_row {groups_per_row} overflows usize"
            ),
        })?;

    let backend = crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
    let ordinal = weights.gpu_handle()?.device_ordinal();
    let mut mask = fill_cuda_mask::<T>(backend, numel, 1.0, ordinal)?;
    if num_groups > 0 {
        let weights_c = weights.contiguous()?;
        let scores = squared_scores_cuda::<T>(backend, weights_c.gpu_handle()?)?;
        let (_values, prune_indices) = backend.topk_nd(&scores, num_groups, 4, 1, 2, false)?;
        mask = scatter_zero_mask_cuda::<T>(
            backend,
            &mask,
            &prune_indices,
            &[num_groups, 4],
            &[num_groups, 2],
            1,
        )?;
    }

    apply_cuda_mask(weights, mask)
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
