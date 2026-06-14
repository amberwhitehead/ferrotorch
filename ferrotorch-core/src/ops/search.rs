//! Searching and sorting tensor operations.
//!
//! - [`searchsorted`] — binary search a sorted tensor for insertion points
//! - [`bucketize`] — discretize values into bucket indices
//! - [`unique`] — return unique elements
//! - [`unique_consecutive`] — deduplicate consecutive elements
//!
//! ## REQ status (per `.design/ferrotorch-core/ops/search.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | `searchsorted` at `ops/search.rs:36`; consumer: re-export `ferrotorch_core::searchsorted` at `lib.rs:176`. CUDA f32/f64 lower on-device via `GpuBackend::searchsorted_1d` (#1545). |
//! | REQ-2 | SHIPPED | `bucketize` at `ops/search.rs:140`; consumer: re-export at `lib.rs:176`. Inherits the CUDA GPU path through its delegation to `searchsorted`. |
//! | REQ-3 | SHIPPED | `unique`; consumer: re-export `ferrotorch_core::unique`. CUDA f32/f64 lower the SORTED sort-by-key dedup on-device via `GpuBackend::unique_1d` (#1545); SORTED-unique values stay GPU-resident, only index/run metadata read back. NaN NOT collapsed (each distinct, sorted last), matching live torch 2.11. |
//! | REQ-4 | SHIPPED | `unique_consecutive` at `ops/search.rs:260`; consumer: re-export at `lib.rs:176`. CUDA f32/f64 lower the data-dependent run compaction on-device via `GpuBackend::unique_consecutive_1d` (#1545); values stay GPU-resident, only run-position metadata read back. |
//! | REQ-5 | SHIPPED | `histc`; consumer: re-export `ferrotorch_core::histc`. Out-of-range/NaN values are SKIPPED (not clamped), matching torch `SummaryOps.cu:92` (#1650); default `min==max` infers the range from data `aminmax`, widening all-equal data to `[v-1,v+1]` per `SummaryOps.cu:328-336` (#1652). CUDA f32/f64 accumulate the histogram on-device via `GpuBackend::histc_1d` (#1545); counts stay GPU-resident. |
//! | REQ-6 | SHIPPED | `meshgrid` (= `meshgrid_indexing(.., Ij)`) + `meshgrid_indexing(tensors, MeshIndexing)`; consumer: `meshgrid` delegates to `meshgrid_indexing`, both re-exported. `MeshIndexing::Xy` swaps the first two inputs+output grids per torch `TensorShape.cpp:4433-4438,4470-4472` (#1652). CUDA f32/f64 produce each axis grid on-device via `GpuBackend::meshgrid_grid` (#1545); grids stay GPU-resident. Grid `i` is differentiable w.r.t. coordinate tensor `i` via `MeshgridBackward` (CORE-110, #1804). |
//! | REQ-7 | SHIPPED | `topk` at `ops/search.rs:814`; consumer: re-export `ferrotorch_core::topk` at `lib.rs:176`. CUDA f32/f64 lower the k-selection on-device via `GpuBackend::topk_1d` (#1545); values stay GPU-resident, only int64 indices read back. Zero-width last dim short-circuits to shaped empty results (CORE-108, #1802); values carry `TopkBackward` for tracking inputs (CORE-109, #1803). |

use std::sync::Arc;

use crate::autograd::no_grad::is_grad_enabled;
use crate::dtype::Float;
use crate::dtype_dispatch::{is_f32, is_f64};
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::storage::TensorStorage;
use crate::tensor::{GradFn, Tensor};

/// Find insertion indices for `values` in a sorted 1-D `boundaries` tensor.
///
/// Returns a tensor of the same shape as `values` containing indices `i`
/// such that `boundaries[i-1] < value <= boundaries[i]` (right=true) or
/// `boundaries[i-1] <= value < boundaries[i]` (right=false).
///
/// Matches PyTorch's `torch.searchsorted`.
pub fn searchsorted<T: Float>(
    boundaries: &Tensor<T>,
    values: &Tensor<T>,
    right: bool,
) -> FerrotorchResult<Vec<usize>> {
    if boundaries.ndim() != 1 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "searchsorted: boundaries must be 1-D, got shape {:?}",
                boundaries.shape()
            ),
        });
    }
    if boundaries.device() != values.device() {
        return Err(FerrotorchError::DeviceMismatch {
            expected: boundaries.device(),
            got: values.device(),
        });
    }

    // GPU fast path (#1545): when both tensors are CUDA-resident f32/f64, run
    // the binary search on-device via `GpuBackend::searchsorted_1d` and read
    // back ONLY the int64 result indices. The value/boundary data never leaves
    // the device, so this is not a CPU<->GPU round trip (R-CODE-4): only the
    // freshly-computed indices are copied to host to satisfy this function's
    // `Vec<usize>` contract.
    if boundaries.is_cuda() && values.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        // #1658: normalise BOTH narrowed-offset CUDA operands to packed offset-0
        // buffers before the on-device binary search reads element 0.
        let values = values.contiguous()?;
        let boundaries = boundaries.contiguous()?;
        let idx_handle =
            backend.searchsorted_1d(values.gpu_handle()?, boundaries.gpu_handle()?, right)?;
        let bytes = backend.gpu_to_cpu(&idx_handle)?;
        // The handle is int64 (PyTorch `ScalarType::Long`); decode 8-byte
        // little-endian chunks into `usize` insertion indices.
        let n = values.numel();
        if bytes.len() < n * 8 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "searchsorted: GPU returned {} bytes, expected >= {} (8 per index)",
                    bytes.len(),
                    n * 8
                ),
            });
        }
        let result: Vec<usize> = bytes
            .chunks_exact(8)
            .take(n)
            .map(|c| {
                let mut buf = [0u8; 8];
                buf.copy_from_slice(c);
                i64::from_le_bytes(buf) as usize
            })
            .collect();
        return Ok(result);
    }

    if boundaries.is_cuda() || values.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "searchsorted" });
    }

    let bounds = boundaries.data()?;
    let vals = values.data_vec()?;

    // The negated comparison operators below are the intended NaN-handling
    // form copied byte-for-byte from upstream Bucketization.cu:33,51
    // (`!(mid_val >= val)` / `!(mid_val > val)`); they are NOT equivalent to
    // `<` / `<=` when `val` is NaN (NaN must advance to `len`, not stop at 0).
    #[allow(
        clippy::neg_cmp_op_on_partial_ord,
        reason = "matches pytorch Bucketization.cu:33,51 NaN advance semantics; \
                  `!(b >= v)` differs from `b < v` for NaN val (advances to len)"
    )]
    let result: Vec<usize> = vals
        .iter()
        .map(|v| {
            if right {
                // upper_bound: advance while `!(mid_val > val)`, mirroring
                // pytorch aten/src/ATen/native/cuda/Bucketization.cu:51
                // (`if (!(mid_val > val)) start = mid + 1;`). For a NaN `v`,
                // `b > NaN` is false so `!(b > NaN)` is true on every step ->
                // advance to `len`, matching torch (NaN -> len). The negated
                // form is REQUIRED: `*b <= *v` is false for NaN and would stop
                // at 0. Finite operands are unchanged: `!(b > v) == (b <= v)`.
                bounds.partition_point(|b| !(*b > *v))
            } else {
                // lower_bound: advance while `!(mid_val >= val)`, mirroring
                // pytorch aten/src/ATen/native/cuda/Bucketization.cu:33
                // (`if (!(mid_val >= val)) start = mid + 1;`). For a NaN `v`,
                // `b >= NaN` is false so `!(b >= NaN)` is true on every step ->
                // advance to `len`, matching torch (NaN -> len). The negated
                // form is REQUIRED: `*b < *v` is false for NaN and would stop
                // at 0. Finite operands are unchanged: `!(b >= v) == (b < v)`.
                bounds.partition_point(|b| !(*b >= *v))
            }
        })
        .collect();

    Ok(result)
}

/// Discretize `input` values into buckets defined by `boundaries`.
///
/// Returns a `Vec<usize>` of bucket indices. Equivalent to
/// `searchsorted(boundaries, input, right=false)`.
///
/// Matches PyTorch's `torch.bucketize`.
pub fn bucketize<T: Float>(
    input: &Tensor<T>,
    boundaries: &Tensor<T>,
    right: bool,
) -> FerrotorchResult<Vec<usize>> {
    searchsorted(boundaries, input, right)
}

/// Return the sorted unique elements of a 1-D tensor.
///
/// Returns `(unique_values, inverse_indices, counts)` where:
/// - `unique_values` — sorted tensor of unique elements
/// - `inverse_indices` — for each input element, its index in `unique_values`
/// - `counts` — how many times each unique element appears
///
/// Matches PyTorch's `torch.unique(sorted=True, return_inverse=True, return_counts=True)`.
pub fn unique<T: Float>(
    input: &Tensor<T>,
) -> FerrotorchResult<(Tensor<T>, Vec<usize>, Vec<usize>)> {
    // GPU fast path (#1545): for CUDA-resident f32/f64 the sort-by-key dedup
    // runs entirely on-device (bitonic sort carrying original indices →
    // run-flag → prefix-sum → compaction) via `GpuBackend::unique_1d`. The
    // SORTED-unique VALUE tensor stays GPU-resident (wrapped straight back into
    // a CUDA `Tensor`); only the derived index/run metadata is read back to
    // build the host `inverse` / `counts` vectors — which are host `Vec<usize>`
    // by this function's signature regardless. The value data never leaves the
    // device and returns, so this is NOT an R-CODE-4 round trip (mirrors
    // `unique_consecutive` / `searchsorted` reading back their integer metadata
    // while the values stay on device). The CUDA `unique` always sorts (no
    // device hashtable in thrust), matching `aten/src/ATen/native/cuda/Unique.cu`
    // `compute_unique` (sort-by-key → adjacent-difference inverse → run-length
    // counts). NaN sorts to the end, each NaN a distinct unique entry — verified
    // live on torch 2.11.0+cu130, matching the `setp.neu` run predicate.
    if input.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        // #1658: normalise a narrowed-offset CUDA view to a packed offset-0
        // buffer before the on-device sort reads element 0.
        let input = input.contiguous()?;
        let n = input.numel();
        let (values_handle, inverse, counts) = backend.unique_1d(input.gpu_handle()?, n)?;
        let out_len = values_handle.len();
        let values_tensor =
            Tensor::from_storage(TensorStorage::gpu(values_handle), vec![out_len], false)?;
        return Ok((values_tensor, inverse, counts));
    }

    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "unique" });
    }

    let data = input.data_vec()?;
    let n = data.len();

    if n == 0 {
        return Ok((
            Tensor::from_storage(TensorStorage::cpu(vec![]), vec![0], false)?,
            vec![],
            vec![],
        ));
    }

    // Sort indices by value. NaN must sort to the END (ranked as the maximum),
    // mirroring upstream `_unique_cpu`'s `input_flattened.sort()`
    // (`aten/src/ATen/native/Unique.cpp:185`), whose default ascending tensor
    // sort drives every NaN past every finite/inf value. The previous
    // `partial_cmp(..).unwrap_or(Equal)` treated NaN as Equal to its neighbours
    // so NaN never moved — corrupting the sorted order, inverse map, and counts.
    // `nan_is_max_cmp` uses `partial_cmp` for the non-NaN/non-NaN case, so
    // `-0.0` and `+0.0` stay equal+adjacent and dedup to ONE entry (torch), and
    // `-inf`/`+inf` order correctly (`-inf` first, `+inf` just before NaN).
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| nan_is_max_cmp(data[a], data[b]));

    // Extract unique values, inverse mapping, and counts.
    let mut unique_vals: Vec<T> = Vec::new();
    let mut inverse = vec![0usize; n];
    let mut counts: Vec<usize> = Vec::new();

    // The first sorted element is ALWAYS unique (its own first occurrence),
    // mirroring upstream `IsUnique<false>::operator()` which returns `true` for
    // `i == 0` unconditionally (`aten/src/ATen/native/Unique.cpp:128`). Seed it
    // with `counts = [1]` and `inverse[indices[0]] = 0`, then walk only
    // `indices[1..]`. Pre-seeding `counts = [0]` and re-iterating from index 0
    // (the old code) double-processed the seed: for a FINITE first element the
    // `val != last` test was false on iteration 0 so it merely incremented the
    // seed count to 1, but for a NaN first element `NaN != NaN == true` opened a
    // SECOND entry and left the seed's count at 0 — a spurious leading entry
    // (count 0) for all-NaN / single-NaN inputs (#1666).
    let mut current_unique_idx = 0;
    unique_vals.push(data[indices[0]]);
    counts.push(1);
    inverse[indices[0]] = 0;

    for &orig_idx in &indices[1..] {
        let val = data[orig_idx];
        if val != *unique_vals.last().unwrap() {
            unique_vals.push(val);
            counts.push(0);
            current_unique_idx += 1;
        }
        inverse[orig_idx] = current_unique_idx;
        counts[current_unique_idx] += 1;
    }

    let unique_len = unique_vals.len();
    let unique_tensor =
        Tensor::from_storage(TensorStorage::cpu(unique_vals), vec![unique_len], false)?;

    Ok((unique_tensor, inverse, counts))
}

/// Remove consecutive duplicate elements from a 1-D tensor.
///
/// Returns `(output, inverse_indices, counts)` where:
/// - `output` — tensor with consecutive duplicates removed
/// - `inverse_indices` — for each input element, its index in `output`
/// - `counts` — length of each run of consecutive equal elements
///
/// Matches PyTorch's `torch.unique_consecutive`.
pub fn unique_consecutive<T: Float>(
    input: &Tensor<T>,
) -> FerrotorchResult<(Tensor<T>, Vec<usize>, Vec<usize>)> {
    // GPU fast path (#1545): for CUDA-resident f32/f64 the run compaction runs
    // entirely on-device (run-flag → prefix-sum → scatter) via
    // `GpuBackend::unique_consecutive_1d`. The deduplicated VALUE tensor stays
    // GPU-resident (wrapped straight back into a CUDA `Tensor`); only the
    // derived run-position metadata is read back to build the host `inverse` /
    // `counts` vectors — which are host `Vec<usize>` by this function's
    // signature regardless. The value data never leaves the device and returns,
    // so this is NOT an R-CODE-4 round trip (mirrors `searchsorted` reading back
    // its i64 indices while the values stay on device).
    if input.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        // #1658: normalise a narrowed-offset CUDA view to a packed offset-0
        // buffer before the on-device run compaction reads element 0.
        let input = input.contiguous()?;
        let n = input.numel();
        let (values_handle, inverse, counts) =
            backend.unique_consecutive_1d(input.gpu_handle()?, n)?;
        let out_len = values_handle.len();
        let output_tensor =
            Tensor::from_storage(TensorStorage::gpu(values_handle), vec![out_len], false)?;
        return Ok((output_tensor, inverse, counts));
    }

    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda {
            op: "unique_consecutive",
        });
    }

    let data = input.data_vec()?;
    let n = data.len();

    if n == 0 {
        return Ok((
            Tensor::from_storage(TensorStorage::cpu(vec![]), vec![0], false)?,
            vec![],
            vec![],
        ));
    }

    let mut output: Vec<T> = vec![data[0]];
    let mut inverse = vec![0usize; n];
    let mut counts: Vec<usize> = vec![1];

    for i in 1..n {
        if data[i] == data[i - 1] {
            *counts.last_mut().unwrap() += 1;
        } else {
            output.push(data[i]);
            counts.push(1);
        }
        inverse[i] = output.len() - 1;
    }

    let out_len = output.len();
    let output_tensor = Tensor::from_storage(TensorStorage::cpu(output), vec![out_len], false)?;

    Ok((output_tensor, inverse, counts))
}

/// Histogram — count elements in equal-width bins.
///
/// `input` is flattened. Returns a 1-D tensor of `bins` counts.
///
/// When `min == max` (the default `torch.histc(x, bins)` call form passes
/// `min=0, max=0`) the range is inferred from the data's `aminmax()`; if the
/// inferred range is still degenerate (all-equal data) it is widened to
/// `[v-1, v+1]`. Mirrors `aten/src/ATen/native/cuda/SummaryOps.cu:328-336`.
///
/// Elements outside `[min, max]` (and `NaN`) are SKIPPED, not clamped, matching
/// torch's `if (bVal >= minvalue && bVal <= maxvalue)` guard
/// (`aten/src/ATen/native/cuda/SummaryOps.cu:92`).
///
/// Matches PyTorch's `torch.histc`.
#[allow(
    clippy::float_cmp,
    reason = "exact `==` mirrors upstream's `min == max` / `minvalue == maxvalue` \
              degenerate-range checks (aten/src/ATen/native/cuda/SummaryOps.cu:328,333); \
              the bit-exact comparison IS the upstream contract for range inference"
)]
pub fn histc<T: Float>(
    input: &Tensor<T>,
    bins: usize,
    min_val: f64,
    max_val: f64,
) -> FerrotorchResult<Tensor<T>> {
    if bins == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "histc: bins must be > 0".into(),
        });
    }
    if input.is_cuda() && !(is_f32::<T>() || is_f64::<T>()) {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "histc" });
    }

    // Default-range inference (#1652a): `torch.histc(x, bins)` defaults to
    // `min=max=0`. Upstream `_histc_*_template` recomputes the range from the
    // data's `aminmax()` when `min == max && numel > 0`, then — if the inferred
    // range is still degenerate (all-equal data) — widens it to `[v-1, v+1]`.
    // Mirrors `aten/src/ATen/native/cuda/SummaryOps.cu:328-336`:
    //   if (min == max && self.numel() > 0) { auto [mn,mx]=self.aminmax(); .. }
    //   if (minvalue == maxvalue) { minvalue -= 1; maxvalue += 1; }
    // This runs BEFORE the histogram device branch so the CPU and GPU paths
    // agree. CUDA f32/f64 infer via on-device min/max reductions and copy back
    // only the two scalar bounds, not the full input buffer.
    let (min_val, max_val) = if min_val == max_val {
        let numel = input.numel();
        if numel > 0 {
            let (mn, mx) = histc_infer_data_range(input)?;
            if !mn.is_finite() || !mx.is_finite() {
                return Err(histc_nonfinite_range_error(mn, mx));
            }
            if mn == mx {
                (mn - 1.0, mx + 1.0)
            } else {
                (mn, mx)
            }
        } else {
            // numel == 0: widen the equal bounds to a valid range (every bin
            // stays empty regardless).
            if !min_val.is_finite() || !max_val.is_finite() {
                return Err(histc_nonfinite_range_error(min_val, max_val));
            }
            (min_val - 1.0, max_val + 1.0)
        }
    } else if min_val > max_val {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("histc: min ({min_val}) must be <= max ({max_val})"),
        });
    } else {
        if !min_val.is_finite() || !max_val.is_finite() {
            return Err(histc_nonfinite_range_error(min_val, max_val));
        }
        (min_val, max_val)
    };

    // GPU fast path (#1545): for CUDA-resident f32/f64 inputs the histogram is
    // accumulated on-device via `GpuBackend::histc_1d` (one thread per input
    // element, `atomicAdd` into the bin). The resulting counts buffer stays
    // GPU-resident — it is wrapped straight back into a CUDA `Tensor` with no
    // host crossing (R-CODE-4: no CPU<->GPU round trip). Bin / range semantics
    // mirror `aten/src/ATen/native/cuda/SummaryOps.cu` getBin (`:41`), the
    // last-bin clamp (`:47-48`), and the `[min,max]` guard (`:92`).
    if input.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        // #1658: normalise a narrowed-offset CUDA view to a packed offset-0
        // buffer before the on-device histogram reads element 0.
        let input = input.contiguous()?;
        let counts_handle = backend.histc_1d(input.gpu_handle()?, bins, min_val, max_val)?;
        return Tensor::from_storage(TensorStorage::gpu(counts_handle), vec![bins], false);
    }

    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "histc" });
    }

    let data = input.data_vec()?;
    let mut counts = vec![<T as num_traits::Zero>::zero(); bins];
    let range = max_val - min_val;
    let bin_width = range / bins as f64;

    // SKIP out-of-range and NaN values, mirroring torch's
    // `if (bVal >= minvalue && bVal <= maxvalue)` guard
    // (`aten/src/ATen/native/cuda/SummaryOps.cu:92`). The previous code CLAMPED
    // out-of-range values into the boundary bins (counting them), which
    // diverged from torch and from ferrotorch's own GPU path (#1650). For an
    // in-range value the bin is `floor((v - min) / bin_width)` with the
    // top-edge value `v == max` falling in the last bin (getBin clamp,
    // `SummaryOps.cu:41,47-48`).
    for &v in &data {
        let f = match num_traits::ToPrimitive::to_f64(&v) {
            Some(f) => f,
            None => continue,
        };
        // NaN fails both comparisons -> skipped, matching torch.
        if !(f >= min_val && f <= max_val) {
            continue;
        }
        let idx = ((f - min_val) / bin_width) as usize;
        let idx = idx.min(bins - 1);
        counts[idx] += <T as num_traits::One>::one();
    }

    Tensor::from_storage(TensorStorage::cpu(counts), vec![bins], false)
}

/// Cartesian-indexing convention for [`meshgrid_indexing`].
///
/// Mirrors `torch.meshgrid`'s `indexing` keyword
/// (`aten/src/ATen/native/TensorShape.cpp:4433-4447`): only `"ij"` (matrix
/// indexing, the default) and `"xy"` (Cartesian indexing) are valid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshIndexing {
    /// Matrix indexing — the default. Output grid `i` varies along input axis
    /// `i`; grids have shape `[len0, len1, ..., lenN-1]`.
    Ij,
    /// Cartesian indexing — swaps the first two input tensors before building
    /// the grids and swaps the first two output grids back, exactly as
    /// `torch.meshgrid(*t, indexing='xy')`
    /// (`aten/src/ATen/native/TensorShape.cpp:4433-4438,4470-4472`).
    Xy,
}

/// Backward node for ONE [`meshgrid`] output grid (CORE-110, #1804).
///
/// Upstream `meshgrid` builds grid `i` as
/// `tensors[i].view(view_shape).expand(shape)`
/// (`aten/src/ATen/native/TensorShape.cpp:4462-4467`), so grid `i` connects
/// ONLY to coordinate tensor `i` and its backward is `ExpandBackward0`: the
/// grid gradient summed over every axis except the grid's own axis `dim`.
/// The `'xy'` convention needs no special casing here — the recursion in
/// [`meshgrid_indexing`] swaps tensor CLONES (which share autograd
/// identity), so each grid's node already references the right coordinate
/// tensor.
///
/// The reduction runs on host (the upstream gradient is read back via
/// `data_vec`) and the reduced gradient is uploaded back to the input's
/// device via `TensorStorage::on_device` — an explicit host round trip of
/// the GRADIENT only (R-LOUD-2; same pattern as `WhereBackward` in
/// `grad_fns/comparison.rs` and [`TopkBackward`]). The forward grids never
/// leave their device.
#[derive(Debug)]
pub struct MeshgridBackward<T: Float> {
    input: Tensor<T>,
    /// The axis this grid varies along (its index in the input list).
    dim: usize,
    /// Full grid shape `[len0, len1, ..., lenN-1]`.
    shapes: Vec<usize>,
}

impl<T: Float> GradFn<T> for MeshgridBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let go = grad_output.data_vec()?;
        let len = self.shapes[self.dim];
        let inner: usize = self.shapes[self.dim + 1..].iter().product();
        // grad[c] = sum of grad_output over every flat position whose
        // coordinate along `dim` is `c` — the inverse of the forward gather
        // `grid[flat] = input[(flat / inner) % len]`. An empty axis anywhere
        // makes `go` empty (total == 0), so the loop body (and its division
        // by `inner`) is never reached with `inner == 0`.
        let mut grad = vec![<T as num_traits::Zero>::zero(); len];
        for (flat, &g) in go.iter().enumerate() {
            grad[(flat / inner) % len] += g;
        }
        let grad_tensor = Tensor::from_storage(
            TensorStorage::on_device(grad, self.input.device())?,
            self.input.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_tensor)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "MeshgridBackward"
    }
}

/// Wrap one freshly computed meshgrid grid, attaching a
/// [`MeshgridBackward`] node when gradient tracking is enabled and THIS
/// grid's coordinate tensor tracks (torch: grid `i` carries
/// `ExpandBackward0` iff `tensors[i].requires_grad`; the other grids stay
/// honestly detached — R-LOUD-3).
fn meshgrid_grid_tensor<T: Float>(
    storage: TensorStorage<T>,
    shapes: &[usize],
    input: &Tensor<T>,
    dim: usize,
) -> FerrotorchResult<Tensor<T>> {
    if is_grad_enabled() && input.requires_grad() {
        Tensor::from_operation(
            storage,
            shapes.to_vec(),
            Arc::new(MeshgridBackward {
                input: input.clone(),
                dim,
                shapes: shapes.to_vec(),
            }),
        )
    } else {
        Tensor::from_storage(storage, shapes.to_vec(), false)
    }
}

/// Create coordinate grids from 1-D coordinate vectors.
///
/// Given N 1-D tensors, returns N tensors of shape `[len0, len1, ..., lenN-1]`
/// where each output tensor contains the coordinates for one axis.
///
/// Grid `i` is differentiable w.r.t. coordinate tensor `i`: for a
/// gradient-tracking input it carries a [`MeshgridBackward`] node that sums
/// the grid gradient over every other axis, matching torch's
/// `view(..).expand(..)` composition (CORE-110, #1804).
///
/// Matches PyTorch's `torch.meshgrid` with `indexing='ij'` (the default). For
/// `indexing='xy'` use [`meshgrid_indexing`].
pub fn meshgrid<T: Float>(tensors: &[Tensor<T>]) -> FerrotorchResult<Vec<Tensor<T>>> {
    meshgrid_indexing(tensors, MeshIndexing::Ij)
}

/// Create coordinate grids from 1-D coordinate vectors with an explicit
/// [`MeshIndexing`] convention.
///
/// For [`MeshIndexing::Ij`] this is identical to [`meshgrid`]. For
/// [`MeshIndexing::Xy`] the first two input tensors are swapped before the
/// grids are built and the first two output grids are swapped back, matching
/// `torch.meshgrid(*tensors, indexing='xy')`
/// (`aten/src/ATen/native/TensorShape.cpp:4433-4438` swap-in,
/// `:4470-4472` swap-out). The swap only happens when there are >= 2 inputs.
///
/// Matches PyTorch's `torch.meshgrid` with the `indexing` keyword.
pub fn meshgrid_indexing<T: Float>(
    tensors: &[Tensor<T>],
    indexing: MeshIndexing,
) -> FerrotorchResult<Vec<Tensor<T>>> {
    if tensors.is_empty() {
        return Ok(vec![]);
    }

    // 'xy' indexing swaps the first two tensors, builds the grids, then swaps
    // the first two output grids back. We can only swap when there are >= 2
    // tensors (`aten/src/ATen/native/TensorShape.cpp:4434-4438`). Building a
    // swapped slice and recursing through the 'ij' path keeps a single grid
    // implementation; the recursion uses `MeshIndexing::Ij` so it is one level
    // deep only.
    if indexing == MeshIndexing::Xy && tensors.len() >= 2 {
        let mut swapped: Vec<Tensor<T>> = Vec::with_capacity(tensors.len());
        swapped.push(tensors[1].clone());
        swapped.push(tensors[0].clone());
        swapped.extend(tensors[2..].iter().cloned());
        let mut grids = meshgrid_indexing(&swapped, MeshIndexing::Ij)?;
        // Swap the first two output grids back (`:4470-4472`).
        grids.swap(0, 1);
        return Ok(grids);
    }

    let device = tensors[0].device();
    for t in tensors {
        if t.ndim() != 1 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "meshgrid: all inputs must be 1-D, got shape {:?}",
                    t.shape()
                ),
            });
        }
        // Mixed devices, including different CUDA ordinals, are rejected
        // (matches upstream meshgrid's "all tensors to have the same device"
        // check at `aten/src/ATen/native/TensorShape.cpp:4396-4398`).
        if t.device() != device {
            return Err(FerrotorchError::DeviceMismatch {
                expected: device,
                got: t.device(),
            });
        }
    }
    let all_cuda = device.is_cuda();

    let shapes: Vec<usize> = tensors.iter().map(|t| t.shape()[0]).collect();
    let ndim = shapes.len();
    let total: usize = shapes.iter().product();

    // GPU fast path (#1545): when every input is CUDA-resident f32/f64, each
    // axis's broadcast grid is produced on-device via `GpuBackend::meshgrid_grid`
    // (a single gather `out[flat] = input[(flat/inner)%axis_len]`). Each grid
    // stays GPU-resident — wrapped straight back into a CUDA `Tensor` with no
    // host crossing (R-CODE-4: no CPU<->GPU round trip). Mirrors the
    // `view(view_shape).expand(shape)` decomposition at
    // `aten/src/ATen/native/TensorShape.cpp:4462-4467` (`indexing='ij'`).
    if all_cuda && (is_f32::<T>() || is_f64::<T>()) {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let mut result = Vec::with_capacity(ndim);
        for (dim, t) in tensors.iter().enumerate() {
            let inner: usize = shapes[dim + 1..].iter().product();
            // #1658: normalise a possibly narrowed-offset CUDA input to a packed
            // offset-0 buffer before the per-axis gather reads element 0. Kept
            // as a separate binding (NOT shadowing `t`) so the backward node
            // references the ORIGINAL input tensor for graph traversal.
            let t_c = t.contiguous()?;
            let grid_handle =
                backend.meshgrid_grid(t_c.gpu_handle()?, total, inner, shapes[dim])?;
            result.push(meshgrid_grid_tensor(
                TensorStorage::gpu(grid_handle),
                &shapes,
                t,
                dim,
            )?);
        }
        return Ok(result);
    }

    if all_cuda {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "meshgrid" });
    }

    let mut result = Vec::with_capacity(ndim);

    for (dim, t) in tensors.iter().enumerate() {
        let data = t.data()?;
        let mut grid = Vec::with_capacity(total);

        // Stride pattern: for dimension `dim`, the value repeats every
        // `product(shapes[dim+1..])` elements and cycles every
        // `product(shapes[..dim]) * product(shapes[dim+1..])` elements.
        let inner: usize = shapes[dim + 1..].iter().product();
        let outer_stride = shapes[dim] * inner;

        for flat in 0..total {
            let coord = (flat / inner) % shapes[dim];
            grid.push(data[coord]);
        }

        // Suppress unused variable warning.
        let _ = outer_stride;

        result.push(meshgrid_grid_tensor(
            TensorStorage::cpu(grid),
            &shapes,
            t,
            dim,
        )?);
    }

    Ok(result)
}

/// Total-order comparison ranking `NaN` as the MAXIMUM, matching torch's
/// sort/topk comparator (`aten/src/ATen/native/cuda/SortingCommon.cuh:47-60`,
/// `GTOp`/`LTOp` with `handleNaN=true`): a NaN `lhs` compares greater than a
/// non-NaN `rhs`, and any two NaNs compare equal.
///
/// Used by [`topk`] so that `largest=true` selects NaN-bearing elements first
/// and `largest=false` ranks NaN last (only picked after the finite values are
/// exhausted), byte-for-byte with `torch.topk`.
fn nan_is_max_cmp<T: Float>(lhs: T, rhs: T) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match (lhs.is_nan(), rhs.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater, // NaN ranks above any finite/inf
        (false, true) => Ordering::Less,
        (false, false) => lhs.partial_cmp(&rhs).unwrap_or(Ordering::Equal),
    }
}

fn histc_nonfinite_range_error(min_val: f64, max_val: f64) -> FerrotorchError {
    FerrotorchError::InvalidArgument {
        message: format!("histc: range of [{min_val}, {max_val}] is not finite"),
    }
}

fn histc_decode_gpu_scalar(bytes: &[u8], elem_size: usize, op: &str) -> FerrotorchResult<f64> {
    if bytes.len() < elem_size {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "{op}: GPU returned {} scalar bytes, expected >= {elem_size}",
                bytes.len()
            ),
        });
    }
    match elem_size {
        4 => {
            let mut buf = [0u8; 4];
            buf.copy_from_slice(&bytes[..4]);
            Ok(f32::from_le_bytes(buf) as f64)
        }
        8 => {
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&bytes[..8]);
            Ok(f64::from_le_bytes(buf))
        }
        _ => Err(FerrotorchError::InvalidArgument {
            message: format!("{op}: unsupported scalar byte width {elem_size}"),
        }),
    }
}

fn histc_infer_data_range<T: Float>(input: &Tensor<T>) -> FerrotorchResult<(f64, f64)> {
    if input.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let input = input.contiguous()?;
        let min_handle = if is_f32::<T>() {
            backend.min_f32(input.gpu_handle()?, input.numel())?
        } else {
            backend.min_f64(input.gpu_handle()?, input.numel())?
        };
        let max_handle = if is_f32::<T>() {
            backend.max_f32(input.gpu_handle()?, input.numel())?
        } else {
            backend.max_f64(input.gpu_handle()?, input.numel())?
        };
        let elem_size = if is_f32::<T>() { 4 } else { 8 };
        let mn = histc_decode_gpu_scalar(
            &backend.gpu_to_cpu(&min_handle)?,
            elem_size,
            "histc: min range inference",
        )?;
        let mx = histc_decode_gpu_scalar(
            &backend.gpu_to_cpu(&max_handle)?,
            elem_size,
            "histc: max range inference",
        )?;
        return Ok((mn, mx));
    }

    let data = input.data_vec()?;
    let mut mn = data[0];
    let mut mx = data[0];
    for &v in &data[1..] {
        if v.is_nan() || (!mn.is_nan() && v < mn) {
            mn = v;
        }
        if v.is_nan() || (!mx.is_nan() && v > mx) {
            mx = v;
        }
    }
    let mn =
        num_traits::ToPrimitive::to_f64(&mn).ok_or_else(|| FerrotorchError::InvalidArgument {
            message: "histc: inferred minimum is not representable as f64".into(),
        })?;
    let mx =
        num_traits::ToPrimitive::to_f64(&mx).ok_or_else(|| FerrotorchError::InvalidArgument {
            message: "histc: inferred maximum is not representable as f64".into(),
        })?;
    Ok((mn, mx))
}

/// Backward node for [`topk`] values (CORE-109, #1803).
///
/// Mirrors upstream `value_selecting_reduction_backward`
/// (`pytorch/torch/csrc/autograd/FunctionsManual.cpp`, used by
/// `derivatives.yaml`'s `topk` entry): the input gradient is
/// `zeros_like(input).scatter(dim, indices, grad)` — each value gradient
/// lands on the index `topk` SELECTED, so ties propagate to the returned
/// index only. Within a row the `k` selected indices are distinct, so
/// scatter-assign and scatter-add coincide.
///
/// The scatter itself runs on host (the gradient and saved indices are read
/// back via `data_vec`) and the resulting gradient is uploaded back to the
/// input's device via `TensorStorage::on_device` — an explicit host round
/// trip of the GRADIENT only (R-LOUD-2; same pattern as `WhereBackward` in
/// `grad_fns/comparison.rs`). The forward values never leave their device.
#[derive(Debug)]
pub struct TopkBackward<T: Float> {
    input: Tensor<T>,
    /// Row-major `[outer, k]` indices into the last dimension, exactly the
    /// `indices` half of the forward's return value.
    indices: Vec<usize>,
    outer: usize,
    k: usize,
    last_dim: usize,
}

impl<T: Float> GradFn<T> for TopkBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let go = grad_output.data_vec()?;
        let mut grad = vec![<T as num_traits::Zero>::zero(); self.input.numel()];
        for o in 0..self.outer {
            for j in 0..self.k {
                let idx = self.indices[o * self.k + j];
                grad[o * self.last_dim + idx] += go[o * self.k + j];
            }
        }
        let grad_tensor = Tensor::from_storage(
            TensorStorage::on_device(grad, self.input.device())?,
            self.input.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_tensor)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "TopkBackward"
    }
}

/// Wrap freshly computed `topk` values, attaching a [`TopkBackward`] node
/// when gradient tracking is enabled and the input tracks (torch: topk
/// values always carry `TopkBackward0` for a `requires_grad` input,
/// including `k == 0`). Otherwise the values are an honest non-tracking
/// tensor (R-LOUD-3: no bare `requires_grad` flag without a backward edge).
fn topk_values_tensor<T: Float>(
    storage: TensorStorage<T>,
    out_shape: Vec<usize>,
    input: &Tensor<T>,
    indices: &[usize],
    outer: usize,
    k: usize,
    last_dim: usize,
) -> FerrotorchResult<Tensor<T>> {
    if is_grad_enabled() && input.requires_grad() {
        Tensor::from_operation(
            storage,
            out_shape,
            Arc::new(TopkBackward {
                input: input.clone(),
                indices: indices.to_vec(),
                outer,
                k,
                last_dim,
            }),
        )
    } else {
        Tensor::from_storage(storage, out_shape, false)
    }
}

/// Return the `k` largest elements and their indices along the last dimension.
///
/// Input must be at least 1-D. Returns `(values, indices)` both with the
/// last dimension replaced by `k`.
///
/// For a gradient-tracking input the values tensor carries a
/// [`TopkBackward`] node that scatters value gradients to the selected
/// input indices, matching `torch.topk`'s `TopkBackward0` (CORE-109, #1803).
///
/// Matches PyTorch's `torch.topk`.
pub fn topk<T: Float>(
    input: &Tensor<T>,
    k: usize,
    largest: bool,
) -> FerrotorchResult<(Tensor<T>, Vec<usize>)> {
    if input.ndim() == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "topk: input must have at least 1 dimension".into(),
        });
    }

    let shape = input.shape();
    let last_dim = *shape.last().unwrap();
    if k > last_dim {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("topk: k ({k}) > last dimension size ({last_dim})"),
        });
    }

    // CORE-108 (#1802): a zero-width last dimension admits ONLY `k == 0`
    // (the `k > last_dim` check above already rejected `k > 0`). Both device
    // paths below compute `numel / last_dim`, an integer divide-by-zero, so
    // short-circuit to the correctly shaped empty result: torch preserves
    // every outer dimension and returns empty values + indices on the
    // input's device (`torch.topk(torch.empty(2,3,0), 0)` -> values/indices
    // shape [2,3,0]; CUDA input -> cuda:0 outputs; verified live on torch
    // 2.11.0+cu130). `TensorStorage::on_device` keeps a CUDA input's empty
    // values CUDA-resident (no silent CPU demotion, R-LOUD-1).
    if last_dim == 0 {
        let values = topk_values_tensor(
            TensorStorage::on_device(Vec::new(), input.device())?,
            shape.to_vec(),
            input,
            &[],
            0,
            0,
            0,
        )?;
        return Ok((values, Vec::new()));
    }

    // GPU fast path (#1545): for CUDA-resident f32/f64 inputs the k-selection
    // runs on-device via `GpuBackend::topk_1d` over the `[outer, last_dim]`
    // layout (the input is contiguous, so the last dim is the innermost run).
    // The VALUES tensor stays GPU-resident — it is wrapped straight back into a
    // CUDA `Tensor` with no host crossing — and ONLY the freshly-computed int64
    // indices are read to host to satisfy this function's `Vec<usize>` contract
    // (R-CODE-4: no CPU<->GPU round trip of the value data).
    if input.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        // #1658: normalise a narrowed-offset CUDA view to a packed offset-0
        // buffer before the on-device k-selection reads element 0 (the
        // `[outer, last_dim]` layout assumption now also guarantees offset 0).
        // Kept as a separate binding (NOT shadowing `input`) so the backward
        // node below references the ORIGINAL input tensor for graph
        // traversal to the leaf.
        let input_c = input.contiguous()?;
        let outer = input_c.numel() / last_dim;
        let (val_handle, idx_handle) =
            backend.topk_1d(input_c.gpu_handle()?, outer, last_dim, k, largest)?;
        let bytes = backend.gpu_to_cpu(&idx_handle)?;
        let n = outer * k;
        if bytes.len() < n * 8 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "topk: GPU returned {} bytes for indices, expected >= {} (8 per index)",
                    bytes.len(),
                    n * 8
                ),
            });
        }
        let out_indices: Vec<usize> = bytes
            .chunks_exact(8)
            .take(n)
            .map(|c| {
                let mut buf = [0u8; 8];
                buf.copy_from_slice(c);
                i64::from_le_bytes(buf) as usize
            })
            .collect();
        let mut out_shape = shape.to_vec();
        *out_shape.last_mut().unwrap() = k;
        let values = topk_values_tensor(
            crate::storage::TensorStorage::gpu(val_handle),
            out_shape,
            input,
            &out_indices,
            outer,
            k,
            last_dim,
        )?;
        return Ok((values, out_indices));
    }

    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "topk" });
    }

    let data = input.data_vec()?;
    let outer: usize = data.len() / last_dim;

    let mut out_values = Vec::with_capacity(outer * k);
    let mut out_indices = Vec::with_capacity(outer * k);

    for o in 0..outer {
        let slice = &data[o * last_dim..(o + 1) * last_dim];
        let mut idx: Vec<usize> = (0..last_dim).collect();

        // NaN-ordering mirrors torch's sort/topk comparator
        // (`aten/src/ATen/native/cuda/SortingCommon.cuh:47-60`, `GTOp`/`LTOp`
        // with `handleNaN=true`): NaN compares GREATER than every finite/inf
        // value. So `topk(largest=true)` selects NaN-bearing elements first
        // (`[NaN, NaN, 5, 3]`), and `topk(largest=false)` ranks NaN LAST and
        // only picks it once the finite values are exhausted (`[1,2,3,5,NaN,NaN]`
        // at k=numel). Verified live on torch 2.11.0+cu130 (RTX 3090). Replaces
        // the old `partial_cmp(..).unwrap_or(Equal)` which treated NaN as equal
        // to its neighbours, dropping NaN out of the top-k entirely.
        if largest {
            idx.sort_by(|&a, &b| nan_is_max_cmp(slice[b], slice[a]));
        } else {
            idx.sort_by(|&a, &b| nan_is_max_cmp(slice[a], slice[b]));
        }

        for &i in &idx[..k] {
            out_values.push(slice[i]);
            out_indices.push(i);
        }
    }

    let mut out_shape = shape.to_vec();
    *out_shape.last_mut().unwrap() = k;
    let values = topk_values_tensor(
        TensorStorage::cpu(out_values),
        out_shape,
        input,
        &out_indices,
        outer,
        k,
        last_dim,
    )?;

    Ok((values, out_indices))
}

#[cfg(test)]
#[allow(
    clippy::excessive_precision,
    clippy::float_cmp,
    reason = "oracle expected values from live torch 2.11; full precision intentional (rounds to dtype at compile time); float comparisons are deliberately exact byte-for-byte parity checks"
)]
mod tests {
    use super::*;

    fn tensor_1d(data: &[f32]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false).unwrap()
    }

    // --- searchsorted ---

    #[test]
    fn test_searchsorted_right() {
        let bounds = tensor_1d(&[1.0, 3.0, 5.0, 7.0]);
        let values = tensor_1d(&[0.0, 2.0, 3.0, 6.0, 8.0]);
        let result = searchsorted(&bounds, &values, true).unwrap();
        assert_eq!(result, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_searchsorted_left() {
        let bounds = tensor_1d(&[1.0, 3.0, 5.0, 7.0]);
        let values = tensor_1d(&[1.0, 3.0, 5.0, 7.0]);
        let result = searchsorted(&bounds, &values, false).unwrap();
        assert_eq!(result, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_searchsorted_empty_bounds() {
        let bounds = tensor_1d(&[]);
        let values = tensor_1d(&[1.0, 2.0]);
        let result = searchsorted(&bounds, &values, true).unwrap();
        assert_eq!(result, vec![0, 0]);
    }

    // --- bucketize ---

    #[test]
    fn test_bucketize() {
        let bounds = tensor_1d(&[0.0, 1.0, 2.0, 3.0]);
        let input = tensor_1d(&[-0.5, 0.5, 1.5, 2.5, 3.5]);
        let result = bucketize(&input, &bounds, false).unwrap();
        assert_eq!(result, vec![0, 1, 2, 3, 4]);
    }

    // --- unique ---

    #[test]
    // reason: round-trip bit-equality — unique() copies values without
    // arithmetic, and the inverse map is verified by index lookup. Both
    // sides hold the same bit pattern, so equality is the right check.
    #[allow(clippy::float_cmp)]
    fn test_unique_sorted() {
        let input = tensor_1d(&[3.0, 1.0, 2.0, 1.0, 3.0, 2.0]);
        let (unique, inverse, counts) = unique(&input).unwrap();
        let unique_data = unique.data().unwrap();
        assert_eq!(unique_data, &[1.0, 2.0, 3.0]);
        assert_eq!(counts, vec![2, 2, 2]);
        // Verify inverse: unique[inverse[i]] == input[i]
        let input_data = input.data().unwrap();
        for i in 0..6 {
            assert_eq!(unique_data[inverse[i]], input_data[i]);
        }
    }

    #[test]
    fn test_unique_empty() {
        let input = tensor_1d(&[]);
        let (unique, inverse, counts) = unique(&input).unwrap();
        assert_eq!(unique.numel(), 0);
        assert!(inverse.is_empty());
        assert!(counts.is_empty());
    }

    #[test]
    fn test_unique_all_same() {
        let input = tensor_1d(&[5.0, 5.0, 5.0]);
        let (unique, _inverse, counts) = unique(&input).unwrap();
        assert_eq!(unique.data().unwrap(), &[5.0]);
        assert_eq!(counts, vec![3]);
    }

    // --- unique_consecutive ---

    #[test]
    fn test_unique_consecutive_basic() {
        let input = tensor_1d(&[1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 1.0, 1.0]);
        let (output, inverse, counts) = unique_consecutive(&input).unwrap();
        let out_data = output.data().unwrap();
        assert_eq!(out_data, &[1.0, 2.0, 3.0, 1.0]);
        assert_eq!(counts, vec![2, 3, 1, 2]);
        assert_eq!(inverse, vec![0, 0, 1, 1, 1, 2, 3, 3]);
    }

    #[test]
    fn test_unique_consecutive_no_duplicates() {
        let input = tensor_1d(&[1.0, 2.0, 3.0]);
        let (output, _inverse, counts) = unique_consecutive(&input).unwrap();
        assert_eq!(output.data().unwrap(), &[1.0, 2.0, 3.0]);
        assert_eq!(counts, vec![1, 1, 1]);
    }

    #[test]
    fn test_unique_consecutive_empty() {
        let input = tensor_1d(&[]);
        let (output, inverse, counts) = unique_consecutive(&input).unwrap();
        assert_eq!(output.numel(), 0);
        assert!(inverse.is_empty());
        assert!(counts.is_empty());
    }

    // --- histc ---

    #[test]
    fn test_histc_basic() {
        let input = tensor_1d(&[0.5, 1.5, 2.5, 3.5, 1.5]);
        let hist = histc(&input, 4, 0.0, 4.0).unwrap();
        let data = hist.data().unwrap();
        assert_eq!(data, &[1.0, 2.0, 1.0, 1.0]);
    }

    #[test]
    fn test_histc_skips_out_of_range() {
        // torch.histc(tensor([-1, 5, 0.5]), bins=2, min=0, max=2) -> [1, 0]:
        // -1 (< min) and 5 (> max) are SKIPPED (not clamped); only 0.5 lands
        // in bin 0. Matches torch's `if (bVal >= minvalue && bVal <= maxvalue)`
        // guard (aten/src/ATen/native/cuda/SummaryOps.cu:92) and ferrotorch's
        // GPU histc path (#1650). Previously this asserted the wrong clamp
        // behavior ([2, 1]).
        let input = tensor_1d(&[-1.0, 5.0, 0.5]);
        let hist = histc(&input, 2, 0.0, 2.0).unwrap();
        let data = hist.data().unwrap();
        assert_eq!(data, &[1.0, 0.0]);
    }

    #[test]
    fn test_histc_skips_nan() {
        // NaN fails both `>= min` and `<= max`, so it is skipped like torch.
        let input = tensor_1d(&[0.5, f32::NAN, 1.5]);
        let hist = histc(&input, 2, 0.0, 2.0).unwrap();
        assert_eq!(hist.data().unwrap(), &[1.0, 1.0]);
    }

    #[test]
    fn test_histc_default_minmax_infers_range() {
        // torch.histc(tensor([1,2,3,4,5]), bins=4) passes min=max=0 -> range
        // inferred [1,5] -> [1,1,1,2] (SummaryOps.cu:328-331).
        let input = tensor_1d(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let hist = histc(&input, 4, 0.0, 0.0).unwrap();
        assert_eq!(hist.data().unwrap(), &[1.0, 1.0, 1.0, 2.0]);
    }

    #[test]
    fn test_histc_default_minmax_all_equal_widens() {
        // torch.histc(tensor([3,3,3]), bins=4) -> aminmax = [3,3], widened to
        // [2,4] (SummaryOps.cu:333-335) -> the three 3.0s land in bin 2.
        let input = tensor_1d(&[3.0, 3.0, 3.0]);
        let hist = histc(&input, 4, 0.0, 0.0).unwrap();
        assert_eq!(hist.data().unwrap(), &[0.0, 0.0, 3.0, 0.0]);
    }

    // --- meshgrid 'xy' ---

    #[test]
    fn test_meshgrid_xy() {
        // torch.meshgrid([1,2,3],[4,5], indexing='xy') -> grids of shape [2,3]
        // with grid0 = [1,2,3,1,2,3], grid1 = [4,4,4,5,5,5]
        // (TensorShape.cpp:4433-4438,4470-4472).
        let x = tensor_1d(&[1.0, 2.0, 3.0]);
        let y = tensor_1d(&[4.0, 5.0]);
        let grids = meshgrid_indexing(&[x, y], MeshIndexing::Xy).unwrap();
        assert_eq!(grids.len(), 2);
        assert_eq!(grids[0].shape(), &[2, 3]);
        assert_eq!(grids[1].shape(), &[2, 3]);
        assert_eq!(grids[0].data().unwrap(), &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        assert_eq!(grids[1].data().unwrap(), &[4.0, 4.0, 4.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_meshgrid_ij_default_unchanged() {
        // meshgrid(..) and meshgrid_indexing(.., Ij) agree (default preserved).
        let x = tensor_1d(&[1.0, 2.0, 3.0]);
        let y = tensor_1d(&[4.0, 5.0]);
        let a = meshgrid(&[x.clone(), y.clone()]).unwrap();
        let b = meshgrid_indexing(&[x, y], MeshIndexing::Ij).unwrap();
        assert_eq!(a[0].data().unwrap(), b[0].data().unwrap());
        assert_eq!(a[1].data().unwrap(), b[1].data().unwrap());
        assert_eq!(a[0].shape(), &[3, 2]);
    }

    // --- meshgrid ---

    #[test]
    fn test_meshgrid_2d() {
        let x = tensor_1d(&[1.0, 2.0, 3.0]);
        let y = tensor_1d(&[4.0, 5.0]);
        let grids = meshgrid(&[x, y]).unwrap();
        assert_eq!(grids.len(), 2);
        assert_eq!(grids[0].shape(), &[3, 2]);
        assert_eq!(grids[1].shape(), &[3, 2]);
        // grid_x should be [[1,1],[2,2],[3,3]]
        let gx = grids[0].data().unwrap();
        assert_eq!(gx, &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
        // grid_y should be [[4,5],[4,5],[4,5]]
        let gy = grids[1].data().unwrap();
        assert_eq!(gy, &[4.0, 5.0, 4.0, 5.0, 4.0, 5.0]);
    }

    // --- topk ---

    #[test]
    fn test_topk_largest() {
        let input = tensor_1d(&[3.0, 1.0, 4.0, 1.0, 5.0, 9.0]);
        let (values, indices) = topk(&input, 3, true).unwrap();
        let vdata = values.data().unwrap();
        assert_eq!(vdata, &[9.0, 5.0, 4.0]);
        assert_eq!(indices, vec![5, 4, 2]);
    }

    #[test]
    fn test_topk_smallest() {
        let input = tensor_1d(&[3.0, 1.0, 4.0, 1.0, 5.0]);
        let (values, indices) = topk(&input, 2, false).unwrap();
        let vdata = values.data().unwrap();
        assert_eq!(vdata, &[1.0, 1.0]);
        assert_eq!(indices, vec![1, 3]);
    }

    #[test]
    fn test_topk_k_exceeds_dim() {
        let input = tensor_1d(&[1.0, 2.0]);
        let result = topk(&input, 5, true);
        assert!(result.is_err());
    }

    /// NaN ordering matches torch's sort/topk comparator (`GTOp`/`LTOp`,
    /// `aten/src/ATen/native/cuda/SortingCommon.cuh:47-60`): NaN ranks as the
    /// MAXIMUM. Verified live on torch 2.11.0+cu130:
    ///   topk([3,NaN,1,5,NaN,2], k=4, largest=True) -> [NaN,NaN,5,3] idx [1,4,3,0]
    #[test]
    fn test_topk_largest_nan_is_top() {
        let input = tensor_1d(&[3.0, f32::NAN, 1.0, 5.0, f32::NAN, 2.0]);
        let (values, indices) = topk(&input, 4, true).unwrap();
        let vdata = values.data().unwrap();
        assert!(
            vdata[0].is_nan() && vdata[1].is_nan(),
            "NaNs first: {vdata:?}"
        );
        assert_eq!(vdata[2], 5.0);
        assert_eq!(vdata[3], 3.0);
        // Two NaNs in ascending original-index order, then the finite extrema.
        assert_eq!(indices, vec![1, 4, 3, 0]);
    }

    /// largest=False ranks NaN LAST under the same comparator. Verified live:
    ///   topk(.., k=4, largest=False) -> [1,2,3,5]              idx [2,5,0,3]
    ///   topk(.., k=6, largest=False) -> [1,2,3,5,NaN,NaN]      idx [2,5,0,3,1,4]
    #[test]
    fn test_topk_smallest_nan_is_last() {
        let input = tensor_1d(&[3.0, f32::NAN, 1.0, 5.0, f32::NAN, 2.0]);
        let (v4, i4) = topk(&input, 4, false).unwrap();
        let v4d = v4.data().unwrap();
        assert!(v4d.iter().all(|v| !v.is_nan()), "no NaN at k=4: {v4d:?}");
        assert_eq!(v4d, &[1.0, 2.0, 3.0, 5.0]);
        assert_eq!(i4, vec![2, 5, 0, 3]);

        let (v6, i6) = topk(&input, 6, false).unwrap();
        let v6d = v6.data().unwrap();
        assert_eq!(&v6d[..4], &[1.0, 2.0, 3.0, 5.0]);
        assert!(v6d[4].is_nan() && v6d[5].is_nan(), "NaNs last: {v6d:?}");
        assert_eq!(i6, vec![2, 5, 0, 3, 1, 4]);
    }
}
