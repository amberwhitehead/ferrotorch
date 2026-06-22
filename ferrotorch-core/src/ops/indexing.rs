//! Forward-pass implementations for N-D indexing operations.
//!
//! - `gather(input, dim, index)` — gather elements along an axis
//! - `scatter(input, dim, index, src)` — scatter src values into input
//! - `scatter_add(input, dim, index, src)` — scatter with addition
//! - `where_cond(condition, x, y)` — ternary selection
//!
//! `gather` / `scatter` / `scatter_value` / `scatter_add` have CUDA-resident
//! fast paths (f32/f64/f16/bf16) that dispatch through rank-aware `GpuBackend`
//! indexing entries backed by PTX kernels in `ferrotorch-gpu`; the host
//! `&[usize]` index is uploaded as a resident `i64` buffer and the result stays
//! GPU-resident.
//! `where_cond` (host-`&[bool]`) uploads the condition once for CUDA operands
//! and delegates to the same resident path as `where_cond_bt`; `masked_select`
//! has its own GPU-resident compaction path (#1185 / #1187).
//! Backward (gradient) functions live in `grad_fns::indexing`.
//!
//! ## REQ status (per `.design/ferrotorch-core/ops/indexing.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | `gather` at `ops/indexing.rs:112`; consumer: re-export `ferrotorch_core::gather` at `lib.rs:174` |
//! | REQ-2 | SHIPPED | `scatter` at `ops/indexing.rs:183` + scalar-src overload `scatter_value` at `ops/indexing.rs:306` (closes #1258 mirroring `aten/src/ATen/native/TensorAdvancedIndexing.cpp:2278`); consumer: re-export at `lib.rs:174`; non-test consumer for `scatter_value`: `Tensor::scatter_value_t` at `methods.rs:1166`. |
//! | REQ-3 | SHIPPED | `scatter_add` at `ops/indexing.rs:259`; consumer: `grad_fns::cumulative::cumsum_backward` at `grad_fns/cumulative.rs:503` invokes `ops::indexing::scatter_add` |
//! | REQ-4 | SHIPPED | `where_cond` at `ops/indexing.rs:334`; consumer: re-export at `lib.rs:174`; `where_cond_bt` CPU fallback at `:458` |
//! | REQ-5 | SHIPPED | `where_cond_bt` at `ops/indexing.rs:397`; consumer: `grad_fns::indexing::where_differentiable` at `grad_fns/indexing.rs:1845,1853` |
//! | REQ-6 | SHIPPED | `masked_select` at `ops/indexing.rs:478`; consumer: `tensor::Tensor::masked_select` at `tensor.rs:1146`; `grad_fns::indexing::masked_select_backward` at `grad_fns/indexing.rs:1823,1828` |
//! | REQ-7 | SHIPPED | grad-fn attachment (e.g. `gather` at `ops/indexing.rs:154-164`); consumer: every autograd-tracking caller |
//! | REQ-8 | SHIPPED | `validate_gather_shapes` at `ops/indexing.rs:66`; consumer: `gather`/`scatter`/`scatter_add` |

use std::sync::Arc;

use crate::autograd::no_grad::is_grad_enabled;
use crate::dtype::{DType, Float};
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::gpu_dispatch::GpuBufferHandle;
use crate::shape::normalize_axis;
use crate::storage::TensorStorage;
use crate::tensor::Tensor;

/// Upload a host `&[usize]` index slice to a GPU-resident `i64` buffer on
/// `ordinal` (PyTorch index tensors are `int64`). The CUDA indexing kernels
/// read the index with `ld.global.s64`, so the host indices are widened to
/// `i64` before the copy.
///
/// Precondition (CORE-125 / #1819): every caller runs
/// `validate_gather_shapes` (rank equality, exact `index.len() ==
/// product(index_shape)`, per-value bounds along `dim`) plus the PyTorch
/// non-dim shape constraints before this upload, so every uploaded index is
/// in-bounds along `dim` and the resident buffer covers exactly the compact
/// index coordinate space the rank-aware kernels read.
pub(crate) fn upload_index_i64(
    index: &[usize],
    ordinal: usize,
) -> FerrotorchResult<GpuBufferHandle> {
    let backend = crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
    let widened: Vec<i64> = index.iter().map(|&v| v as i64).collect();
    // SAFETY: `widened: Vec<i64>` is fully initialized and borrowed for the
    // duration of this call. `i64` has no padding/niches, so reading its
    // backing store as `&[u8]` of length `widened.len() * 8` (==
    // `widened.len() * size_of::<i64>()`) is sound and exactly covers the
    // allocation; the byte slice does not outlive `widened`.
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(widened.as_ptr().cast::<u8>(), widened.len() * 8) };
    backend.cpu_to_gpu(bytes, DType::I64, ordinal)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Whether at least one of two tensors requires grad (and grad is enabled).
#[inline]
fn needs_grad<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> bool {
    is_grad_enabled() && (a.requires_grad() || b.requires_grad())
}

/// Compute the flat index into a C-contiguous buffer from per-axis coordinates.
#[inline]
fn flat_index(coords: &[usize], shape: &[usize]) -> usize {
    let mut idx = 0;
    let mut stride = 1;
    for d in (0..shape.len()).rev() {
        idx += coords[d] * stride;
        stride *= shape[d];
    }
    idx
}

/// Increment a multi-dimensional coordinate vector in C-order (last axis
/// fastest). Returns `false` when the coordinate wraps past the last element.
#[inline]
fn increment_coords(coords: &mut [usize], shape: &[usize]) -> bool {
    for d in (0..shape.len()).rev() {
        coords[d] += 1;
        if coords[d] < shape[d] {
            return true;
        }
        coords[d] = 0;
    }
    false
}

/// Checked element count of a claimed `index_shape` — returns
/// `InvalidArgument` when the product overflows `usize` (CORE-007 class)
/// instead of wrapping in release builds.
fn checked_index_numel(index_shape: &[usize]) -> FerrotorchResult<usize> {
    index_shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| FerrotorchError::InvalidArgument {
            message: format!(
                "gather/scatter: index_shape {index_shape:?} element count overflows usize"
            ),
        })
}

/// Validate the flat `index` slice and its claimed `index_shape` against
/// `input` as ONE coherent logical tensor (CORE-125 / #1819):
///
///   - `index.ndim() == input.ndim()` — rank equality, mirroring
///     `gather_shape_check` / `scatter_shape_check` in
///     `aten/src/ATen/native/ScatterGatherChecks.h:41-124` (invoked by the
///     upstream meta functions BEFORE any kernel is selected,
///     `aten/src/ATen/native/TensorAdvancedIndexing.cpp:179,192`);
///   - `index.len() == product(index_shape)` with a CHECKED product — in
///     PyTorch the index is a real tensor so data length and shape agree by
///     construction; ferrotorch's flat-slice API must enforce the coherence
///     explicitly before ANY element count is derived from the shape;
///   - every index value is in-bounds for `input.shape()[dim]`.
///
/// Returns the validated index element count (== `index_data.len()`).
fn validate_gather_shapes(
    input_shape: &[usize],
    dim: usize,
    index_shape: &[usize],
    index_data: &[usize],
    axis_size: usize,
) -> FerrotorchResult<usize> {
    if input_shape.len() != index_shape.len() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "gather/scatter: input ndim ({}) must equal index ndim ({})",
                input_shape.len(),
                index_shape.len()
            ),
        });
    }
    // The host slice and its claimed shape are one logical tensor: their
    // element counts must agree exactly (CORE-125 metadata coherence).
    let index_numel = checked_index_numel(index_shape)?;
    if index_data.len() != index_numel {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "gather/scatter: index slice has {} elements but index_shape {:?} implies {}",
                index_data.len(),
                index_shape,
                index_numel
            ),
        });
    }
    // Validate index values are in-bounds along `dim`.
    for &v in index_data {
        if v >= axis_size {
            return Err(FerrotorchError::IndexOutOfBounds {
                index: v,
                axis: dim,
                size: axis_size,
            });
        }
    }
    Ok(index_numel)
}

/// CORE-126 (#1820): enforce PyTorch's self/input-side non-dim shape rule.
///
/// `gather_shape_check` requires `index.size(d) <= self.size(d)` for every
/// `d != dim`; `scatter_shape_check` applies the same self-side rule before
/// additionally checking `src`. The gather/scatter axis itself is governed by
/// per-value bounds, not by an index-size upper bound.
fn validate_index_fits_input_non_dim(
    op: &'static str,
    input_shape: &[usize],
    dim: usize,
    index_shape: &[usize],
) -> FerrotorchResult<()> {
    for (d, (&idx_d, &in_d)) in index_shape.iter().zip(input_shape).enumerate() {
        if d != dim && idx_d > in_d {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "{op}: expected index {index_shape:?} to be no larger than input \
                     {input_shape:?} apart from dimension {dim} \
                     (axis {d}: {idx_d} > {in_d})"
                ),
            });
        }
    }
    Ok(())
}

/// CORE-127 (#1821): validate `src` against `index` for the scatter family,
/// mirroring the upstream tensor-src rule in `scatter_shape_check`
/// (`aten/src/ATen/native/ScatterGatherChecks.h:67-124`), verified live on
/// `torch==2.11.0`:
///
///   - rank equality — "Index tensor must have the same number of dimensions
///     as src tensor";
///   - `index.size(d) <= src.size(d)` for ALL `d` — "Expected index [..] to
///     be ... no larger size than src [..]".
///
/// A larger `src` is legal: the consumed values are addressed by COORDINATE
/// (each index position maps to the same coordinates in `src`), never as a
/// flat prefix.
fn validate_scatter_src<T: Float>(
    op: &'static str,
    src: &Tensor<T>,
    index_shape: &[usize],
) -> FerrotorchResult<()> {
    if src.ndim() != index_shape.len() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "{op}: index tensor must have the same number of dimensions as src tensor \
                 (index_shape {index_shape:?} is rank {}, src shape {:?} is rank {})",
                index_shape.len(),
                src.shape(),
                src.ndim()
            ),
        });
    }
    for (d, (&idx_d, &src_d)) in index_shape.iter().zip(src.shape()).enumerate() {
        if idx_d > src_d {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "{op}: expected index {index_shape:?} to be no larger size than src {:?} \
                     (axis {d}: {idx_d} > {src_d})",
                    src.shape()
                ),
            });
        }
    }
    Ok(())
}

/// Coordinate-mapped flat `src` offset for the index position `coords`
/// (CORE-127 / #1821): PyTorch addresses `src` by the index element's
/// COORDINATES, so when `src` is larger than `index` the consumed region is
/// the per-axis prefix slab — never the flat prefix. `coords` are index-space
/// coordinates (already `< index_shape[d] <= src_shape[d]` per
/// [`validate_scatter_src`]); the offset is computed against `src`'s strides.
#[inline]
fn src_flat_offset(coords: &[usize], src_shape: &[usize], same_shape: bool, i: usize) -> usize {
    if same_shape {
        // index and src share a shape: flat order is identical.
        i
    } else {
        flat_index(coords, src_shape)
    }
}

/// Materialise the consumed prefix slab of `src` (shape = `index_shape`) as a
/// fresh contiguous ON-DEVICE buffer for the scatter-family CUDA kernels,
/// which read `src[t]` parallel to `index[t]` (CORE-127 / #1821). When the
/// shapes already match this is the identity. The narrowing + strided-copy
/// run on-device (no host round trip) and under `no_grad` — the slab feeds
/// the kernel only; autograd tracks the ORIGINAL `src` via the grad_fn.
fn cuda_src_prefix_slab<T: Float>(
    src: &Tensor<T>,
    index_shape: &[usize],
) -> FerrotorchResult<Tensor<T>> {
    if src.shape() == index_shape {
        return Ok(src.clone());
    }
    crate::autograd::no_grad::no_grad(|| {
        let mut slab = src.clone();
        for (d, &len) in index_shape.iter().enumerate() {
            if slab.shape()[d] != len {
                slab = slab.narrow(d, 0, len)?;
            }
        }
        slab.contiguous()
    })
}

// ---------------------------------------------------------------------------
// gather
// ---------------------------------------------------------------------------

/// Gather values from `input` along `dim` using `index`.
///
/// PyTorch semantics:
/// ```text
/// output[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
/// output[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
/// output[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
/// ```
///
/// The output has the same shape as `index`.
///
/// `index` is passed as a flat `&[usize]` slice with shape `index_shape`.
/// If `input.requires_grad()`, attaches a `GatherBackward` grad_fn.
pub fn gather<T: Float>(
    input: &Tensor<T>,
    dim: isize,
    index: &[usize],
    index_shape: &[usize],
) -> FerrotorchResult<Tensor<T>> {
    // PyTorch treats 0-D tensors as if they had `ensure_nonempty_dim(self.dim()) == 1`
    // for gather/scatter shape checks (see `ScatterGatherChecks.h:44`
    // `ensure_nonempty_dim`). Mirror that here: a 0-D input acts like a
    // 1-element tensor of shape `[1]` along the only valid axis (dim 0). When
    // index is also 0-D (rank-0 scalar index), promote it to shape `[1]` so
    // the ndim-equality validation succeeds; the output shape preserves the
    // caller's original `index_shape` (still `[]`) so 0-D in → 0-D out.
    let ndim = input.ndim();
    let effective_input_shape: Vec<usize> = if ndim == 0 {
        vec![1]
    } else {
        input.shape().to_vec()
    };
    let effective_ndim = effective_input_shape.len();
    let effective_index_shape: Vec<usize> = if ndim == 0 && index_shape.is_empty() {
        vec![1]
    } else {
        index_shape.to_vec()
    };
    let dim = normalize_axis(dim, effective_ndim)?;

    // PyTorch gather meta returns immediately for empty index tensors after
    // normalizing `dim` and sizing the output (`TensorAdvancedIndexing.cpp`
    // gather meta). That intentionally skips dtype and shape checks, so
    // `index_shape=[999, 0]` is legal for an input shaped `[2, 3]`. Preserve
    // that exact boundary here while still verifying the host slice coheres
    // with its claimed empty shape.
    let claimed_index_numel = checked_index_numel(index_shape)?;
    if claimed_index_numel == 0 {
        if !index.is_empty() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "gather: index slice has {} elements but index_shape {:?} implies 0",
                    index.len(),
                    index_shape
                ),
            });
        }
        let output_shape = index_shape.to_vec();
        let storage = if input.is_cuda() {
            let ordinal = match input.device() {
                crate::device::Device::Cuda(ordinal) => ordinal,
                _ => unreachable!(),
            };
            let backend =
                crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            TensorStorage::gpu(backend.alloc_zeros(0, T::dtype(), ordinal)?)
        } else {
            TensorStorage::cpu(Vec::new())
        };
        if input.requires_grad() && is_grad_enabled() {
            let grad_fn = Arc::new(crate::grad_fns::indexing::GatherBackward {
                input: input.clone(),
                dim,
                index: Vec::new(),
                index_cuda: None,
                index_shape: output_shape.clone(),
            });
            return Tensor::from_operation(storage, output_shape, grad_fn);
        }
        return Tensor::from_storage(storage, output_shape, false);
    }

    // CORE-125 (#1819): the host index slice and its claimed shape are
    // validated as ONE checked logical tensor (rank, exact length, value
    // bounds) BEFORE any CPU loop or CUDA upload/dispatch — mirroring
    // upstream, where `gather_shape_check` runs in the meta function before
    // any device kernel is selected
    // (`aten/src/ATen/native/TensorAdvancedIndexing.cpp:179`).
    let index_numel = validate_gather_shapes(
        &effective_input_shape,
        dim,
        &effective_index_shape,
        index,
        effective_input_shape[dim],
    )?;
    validate_index_fits_input_non_dim(
        "gather",
        &effective_input_shape,
        dim,
        &effective_index_shape,
    )?;

    // CUDA-resident fast path: `input` on a CUDA device. The
    // host `&[usize]` index is uploaded as a GPU-resident `i64` buffer; the
    // rank-aware gather kernel runs entirely on-device and the result stays
    // resident (no host round trip).
    if input.is_cuda() {
        match T::dtype() {
            DType::F32 | DType::F64 | DType::F16 | DType::BF16 => {
                // The rank-aware PTX kernel assumes a C-contiguous physical
                // buffer. A transposed/permuted CUDA view has logical shape !=
                // physical layout, so materialise to contiguous ON-DEVICE first
                // (strided_copy kernel — no host round trip).
                let original_input = input.clone();
                let input = input.contiguous()?;
                let input_handle = input.gpu_handle()?;
                let ordinal = input_handle.device_ordinal();
                let idx_handle = upload_index_i64(index, ordinal)?;
                let backend =
                    crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
                let h = backend.gather_intidx_nd(
                    input_handle,
                    &idx_handle,
                    &effective_input_shape,
                    &effective_index_shape,
                    dim,
                )?;
                let output_shape = index_shape.to_vec();
                let storage = TensorStorage::gpu(h);
                if original_input.requires_grad() && is_grad_enabled() {
                    let grad_fn = Arc::new(crate::grad_fns::indexing::GatherBackward {
                        input: original_input,
                        dim,
                        index: index.to_vec(),
                        index_cuda: None,
                        index_shape: index_shape.to_vec(),
                    });
                    return Tensor::from_operation(storage, output_shape, grad_fn);
                }
                return Tensor::from_storage(storage, output_shape, false);
            }
            _ => return Err(FerrotorchError::NotImplementedOnCuda { op: "gather" }),
        }
    }

    let input_shape: &[usize] = &effective_input_shape;

    let input_data = input.data_vec()?;
    let out_numel = index_numel;
    let mut output = vec![<T as num_traits::Zero>::zero(); out_numel];

    let mut coords = vec![0usize; effective_ndim];
    for out_flat in 0..out_numel {
        // Build source coordinates: same as output coords, but replace dim
        // with the index value.
        let idx_val = index[out_flat];
        let mut src_coords = coords.clone();
        src_coords[dim] = idx_val;
        let src_flat = flat_index(&src_coords, input_shape);
        output[out_flat] = input_data[src_flat];

        if out_flat + 1 < out_numel {
            increment_coords(&mut coords, &effective_index_shape);
        }
    }

    let output_shape = index_shape.to_vec();

    if input.requires_grad() && is_grad_enabled() {
        let grad_fn = Arc::new(crate::grad_fns::indexing::GatherBackward {
            input: input.clone(),
            dim,
            index: index.to_vec(),
            index_cuda: None,
            index_shape: index_shape.to_vec(),
        });
        Tensor::from_operation(TensorStorage::cpu(output), output_shape, grad_fn)
    } else {
        Tensor::from_storage(TensorStorage::cpu(output), output_shape, false)
    }
}

// ---------------------------------------------------------------------------
// scatter
// ---------------------------------------------------------------------------

/// Scatter `src` values into a clone of `input` along `dim` using `index`.
///
/// PyTorch semantics:
/// ```text
/// output = input.clone()
/// output[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
/// ```
///
/// The output has the same shape as `input`.
///
/// `index` and `src` are flat slices with shape `index_shape`.
/// If either `input` or `src` requires grad, attaches a `ScatterBackward`.
pub fn scatter<T: Float>(
    input: &Tensor<T>,
    dim: isize,
    index: &[usize],
    index_shape: &[usize],
    src: &Tensor<T>,
) -> FerrotorchResult<Tensor<T>> {
    let ndim = input.ndim();
    if ndim == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "scatter: input must have at least 1 dimension".into(),
        });
    }
    let dim = normalize_axis(dim, ndim)?;
    let input_shape = input.shape();

    // CORE-125 (#1819): validate the index metadata + values and the src
    // length BEFORE the CUDA fast path — mirroring upstream, where
    // `scatter_shape_check` runs in the meta function before any device
    // kernel is selected
    // (`aten/src/ATen/native/TensorAdvancedIndexing.cpp:192`).
    let index_numel =
        validate_gather_shapes(input_shape, dim, index_shape, index, input_shape[dim])?;
    validate_index_fits_input_non_dim("scatter", input_shape, dim, index_shape)?;
    // CORE-127 (#1821): per-axis src validation (rank equality +
    // `index.size(d) <= src.size(d)` for all d) replaces the numel-only gate
    // — the per-axis rule implies numel sufficiency and is what upstream
    // enforces (`scatter_shape_check`).
    validate_scatter_src("scatter", src, index_shape)?;

    // CUDA-resident fast path: `input` + `src` on the same CUDA device. The
    // host index uploads as a resident `i64` buffer; the result (a clone of
    // `input` with the scattered writes) stays GPU-resident.
    if input.is_cuda() || src.is_cuda() {
        if input.device() != src.device() {
            return Err(FerrotorchError::DeviceMismatch {
                expected: input.device(),
                got: src.device(),
            });
        }
        match T::dtype() {
            DType::F32 | DType::F64 | DType::F16 | DType::BF16 => {
                // The rank-aware scatter kernel reads `self`/`src` as
                // C-contiguous buffers. Materialise to contiguous ON-DEVICE
                // (strided_copy — no host round trip) so a transposed/permuted
                // view's physical buffer matches its logical shape.
                let original_input = input.clone();
                let input = input.contiguous()?;
                // CORE-127 (#1821): the kernel reads `src[t]` PARALLEL to
                // `index[t]`, which is coordinate-correct only when src and
                // index share a shape. For a per-axis-larger src, materialise
                // the consumed prefix slab (shape = index_shape) on-device
                // first — PyTorch addresses src by coordinate, never as a
                // flat prefix. The slab feeds the KERNEL only; autograd
                // stores the ORIGINAL `src` below so the graph edge reaches
                // the caller's leaf.
                let src_slab = cuda_src_prefix_slab(src, index_shape)?.contiguous()?;
                let input_shape: &[usize] = input.shape();
                let input_handle = input.gpu_handle()?;
                let ordinal = input_handle.device_ordinal();
                let idx_handle = upload_index_i64(index, ordinal)?;
                let backend =
                    crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
                let src_handle = src_slab.gpu_handle()?;
                let h = match T::dtype() {
                    DType::F32 => backend.scatter_nd_f32(
                        input_handle,
                        &idx_handle,
                        src_handle,
                        input_shape,
                        index_shape,
                        dim,
                    )?,
                    DType::F64 => backend.scatter_nd_f64(
                        input_handle,
                        &idx_handle,
                        src_handle,
                        input_shape,
                        index_shape,
                        dim,
                    )?,
                    DType::F16 => backend.scatter_nd_f16(
                        input_handle,
                        &idx_handle,
                        src_handle,
                        input_shape,
                        index_shape,
                        dim,
                    )?,
                    DType::BF16 => backend.scatter_nd_bf16(
                        input_handle,
                        &idx_handle,
                        src_handle,
                        input_shape,
                        index_shape,
                        dim,
                    )?,
                    _ => unreachable!(),
                };
                let output_shape = input_shape.to_vec();
                let storage = TensorStorage::gpu(h);
                if needs_grad(&original_input, src) {
                    let grad_fn = Arc::new(crate::grad_fns::indexing::ScatterBackward {
                        input: original_input,
                        // The ORIGINAL src (not the kernel slab): the graph
                        // edge must reach the caller's leaf, and the backward
                        // src-shape contract (CORE-127) checks against it.
                        src: src.clone(),
                        dim,
                        index: index.to_vec(),
                        index_shape: index_shape.to_vec(),
                    });
                    return Tensor::from_operation(storage, output_shape, grad_fn);
                }
                return Tensor::from_storage(storage, output_shape, false);
            }
            _ => return Err(FerrotorchError::NotImplementedOnCuda { op: "scatter" }),
        }
    }

    let mut output = input.data_vec()?;
    let src_data = src.data_vec()?;
    // CORE-127 (#1821): src is addressed by COORDINATE (PyTorch parity) —
    // for src.shape == index_shape that is the identical flat order; for a
    // per-axis-larger src the index coords map through src's strides.
    let src_shape = src.shape();
    let same_shape = src_shape == index_shape;

    let mut coords = vec![0usize; ndim];
    for i in 0..index_numel {
        let idx_val = index[i];
        let mut dst_coords = coords.clone();
        dst_coords[dim] = idx_val;
        let dst_flat = flat_index(&dst_coords, input_shape);
        output[dst_flat] = src_data[src_flat_offset(&coords, src_shape, same_shape, i)];

        if i + 1 < index_numel {
            increment_coords(&mut coords, index_shape);
        }
    }

    let output_shape = input_shape.to_vec();

    if needs_grad(input, src) {
        let grad_fn = Arc::new(crate::grad_fns::indexing::ScatterBackward {
            input: input.clone(),
            src: src.clone(),
            dim,
            index: index.to_vec(),
            index_shape: index_shape.to_vec(),
        });
        Tensor::from_operation(TensorStorage::cpu(output), output_shape, grad_fn)
    } else {
        Tensor::from_storage(TensorStorage::cpu(output), output_shape, false)
    }
}

// ---------------------------------------------------------------------------
// scatter_value (scalar-src overload — closes #1258)
// ---------------------------------------------------------------------------

/// Scatter a scalar `value` into a clone of `input` along `dim` at the
/// positions named by `index`. The `scatter.value` overload of PyTorch's
/// `scatter`.
///
/// PyTorch semantics (scalar-src):
/// ```text
/// output = input.clone()
/// output[index[i][j][k]][j][k] = value  # if dim == 0
/// ```
///
/// Mirrors upstream `Tensor& scatter_(int64_t dim, const Tensor& index,
/// const Scalar& value)` at
/// `aten/src/ATen/native/TensorAdvancedIndexing.cpp:2278`. Equivalent to
/// `scatter(input, dim, index, index_shape, full_like(index, value))` but
/// avoids the temporary `src` allocation.
///
/// Autograd note: the scalar `value` is not a differentiable input, so
/// gradients route only to `input` via a `ScatterValueBackward`-shaped path
/// — for now we route through the existing `ScatterBackward` by
/// materialising a `src` of zeros (the value-arm grad of `src` is
/// discarded anyway). When `input` does not require grad, no autograd node
/// is attached.
pub fn scatter_value<T: Float>(
    input: &Tensor<T>,
    dim: isize,
    index: &[usize],
    index_shape: &[usize],
    value: T,
) -> FerrotorchResult<Tensor<T>> {
    let ndim = input.ndim();
    if ndim == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "scatter_value: input must have at least 1 dimension".into(),
        });
    }
    let dim = normalize_axis(dim, ndim)?;
    let input_shape = input.shape();

    // CORE-125 (#1819): validate the index metadata + values BEFORE the CUDA
    // fast path (upstream runs `scatter_shape_check` in the meta function
    // before any device kernel is selected,
    // `aten/src/ATen/native/TensorAdvancedIndexing.cpp:192`).
    let index_numel =
        validate_gather_shapes(input_shape, dim, index_shape, index, input_shape[dim])?;
    validate_index_fits_input_non_dim("scatter_value", input_shape, dim, index_shape)?;

    // CUDA-resident fast path: `input` on a CUDA device. The host
    // index uploads as a resident `i64` buffer; the broadcast scalar `value`
    // is written at every named position by the on-device kernel and the
    // result stays resident.
    if input.is_cuda() {
        match T::dtype() {
            DType::F32 | DType::F64 | DType::F16 | DType::BF16 => {
                // Materialise `self` to contiguous ON-DEVICE (strided_copy — no
                // host round trip) so a transposed/permuted view's physical
                // buffer matches its logical shape.
                let original_input = input.clone();
                let input = input.contiguous()?;
                let input_shape: &[usize] = input.shape();
                let input_handle = input.gpu_handle()?;
                let ordinal = input_handle.device_ordinal();
                let idx_handle = upload_index_i64(index, ordinal)?;
                let backend =
                    crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
                let h = match T::dtype() {
                    DType::F32 => backend.scatter_value_nd_f32(
                        input_handle,
                        &idx_handle,
                        value.to_f32().ok_or(FerrotorchError::InvalidArgument {
                            message: "scatter_value: value not representable as f32".into(),
                        })?,
                        input_shape,
                        index_shape,
                        dim,
                    )?,
                    DType::F64 => backend.scatter_value_nd_f64(
                        input_handle,
                        &idx_handle,
                        value.to_f64().ok_or(FerrotorchError::InvalidArgument {
                            message: "scatter_value: value not representable as f64".into(),
                        })?,
                        input_shape,
                        index_shape,
                        dim,
                    )?,
                    DType::F16 => backend.scatter_value_nd_f16(
                        input_handle,
                        &idx_handle,
                        value.to_f32().ok_or(FerrotorchError::InvalidArgument {
                            message: "scatter_value: value not representable as f32".into(),
                        })?,
                        input_shape,
                        index_shape,
                        dim,
                    )?,
                    DType::BF16 => backend.scatter_value_nd_bf16(
                        input_handle,
                        &idx_handle,
                        value.to_f32().ok_or(FerrotorchError::InvalidArgument {
                            message: "scatter_value: value not representable as f32".into(),
                        })?,
                        input_shape,
                        index_shape,
                        dim,
                    )?,
                    _ => unreachable!(),
                };
                let output_shape = input_shape.to_vec();
                let storage = TensorStorage::gpu(h);
                if is_grad_enabled() && original_input.requires_grad() {
                    let zero = <T as num_traits::Zero>::zero();
                    let zeros_src = Tensor::from_storage(
                        TensorStorage::cpu(vec![zero; crate::shape::numel(index_shape)]),
                        index_shape.to_vec(),
                        false,
                    )?;
                    let grad_fn = Arc::new(crate::grad_fns::indexing::ScatterBackward {
                        input: original_input,
                        src: zeros_src,
                        dim,
                        index: index.to_vec(),
                        index_shape: index_shape.to_vec(),
                    });
                    return Tensor::from_operation(storage, output_shape, grad_fn);
                }
                return Tensor::from_storage(storage, output_shape, false);
            }
            _ => {
                return Err(FerrotorchError::NotImplementedOnCuda {
                    op: "scatter_value",
                });
            }
        }
    }

    let mut output = input.data_vec()?;

    let mut coords = vec![0usize; ndim];
    for i in 0..index_numel {
        let idx_val = index[i];
        let mut dst_coords = coords.clone();
        dst_coords[dim] = idx_val;
        let dst_flat = flat_index(&dst_coords, input_shape);
        output[dst_flat] = value;

        if i + 1 < index_numel {
            increment_coords(&mut coords, index_shape);
        }
    }

    let output_shape = input_shape.to_vec();

    if is_grad_enabled() && input.requires_grad() {
        // Route through ScatterBackward by passing a zeros `src` — the
        // value-arm has no `src` gradient (scalar is not differentiable),
        // and the `input` gradient is the standard scatter zero-out at the
        // written positions.
        let zero = <T as num_traits::Zero>::zero();
        let zeros_src = Tensor::from_storage(
            TensorStorage::cpu(vec![zero; index_numel]),
            index_shape.to_vec(),
            false,
        )?;
        let grad_fn = Arc::new(crate::grad_fns::indexing::ScatterBackward {
            input: input.clone(),
            src: zeros_src,
            dim,
            index: index.to_vec(),
            index_shape: index_shape.to_vec(),
        });
        Tensor::from_operation(TensorStorage::cpu(output), output_shape, grad_fn)
    } else {
        Tensor::from_storage(TensorStorage::cpu(output), output_shape, false)
    }
}

// ---------------------------------------------------------------------------
// scatter_add
// ---------------------------------------------------------------------------

/// Scatter-add `src` values into a clone of `input` along `dim`.
///
/// Like `scatter`, but uses addition instead of assignment:
/// ```text
/// output = input.clone()
/// output[index[i][j][k]][j][k] += src[i][j][k]  # if dim == 0
/// ```
pub fn scatter_add<T: Float>(
    input: &Tensor<T>,
    dim: isize,
    index: &[usize],
    index_shape: &[usize],
    src: &Tensor<T>,
) -> FerrotorchResult<Tensor<T>> {
    let ndim = input.ndim();
    if ndim == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "scatter_add: input must have at least 1 dimension".into(),
        });
    }
    let dim = normalize_axis(dim, ndim)?;
    let input_shape = input.shape();

    // CORE-125 (#1819): validate the index metadata + values and the src
    // length BEFORE the CUDA fast path (upstream runs `scatter_shape_check`
    // in the meta function before any device kernel is selected,
    // `aten/src/ATen/native/TensorAdvancedIndexing.cpp:192`).
    let index_numel =
        validate_gather_shapes(input_shape, dim, index_shape, index, input_shape[dim])?;
    validate_index_fits_input_non_dim("scatter_add", input_shape, dim, index_shape)?;
    // CORE-127 (#1821): per-axis src validation (rank equality +
    // `index.size(d) <= src.size(d)` for all d) replaces the numel-only gate.
    validate_scatter_src("scatter_add", src, index_shape)?;

    // CUDA-resident fast path: `input` + `src` on the same CUDA device. The
    // host index uploads as a resident `i64` buffer; the kernel accumulates
    // with an ATOMIC add so duplicate index values targeting the same output
    // slot sum correctly. The result stays GPU-resident.
    if input.is_cuda() || src.is_cuda() {
        if input.device() != src.device() {
            return Err(FerrotorchError::DeviceMismatch {
                expected: input.device(),
                got: src.device(),
            });
        }
        match T::dtype() {
            DType::F32 | DType::F64 | DType::F16 | DType::BF16 => {
                // The rank-aware scatter_add kernel reads `self`/`src` as
                // C-contiguous buffers. Materialise to contiguous ON-DEVICE
                // (strided_copy — no host round trip) so a transposed/permuted
                // view's physical buffer matches its logical shape. (A non-zero
                // `self` exposes this: an all-zeros buffer reads identically
                // either layout.)
                let original_input = input.clone();
                let input = input.contiguous()?;
                // CORE-127 (#1821): kernel consumption is parallel-flat; for
                // a per-axis-larger src materialise the coordinate-consumed
                // prefix slab (shape = index_shape) on-device. Autograd
                // stores the ORIGINAL src below.
                let src_slab = cuda_src_prefix_slab(src, index_shape)?.contiguous()?;
                let input_shape: &[usize] = input.shape();
                let input_handle = input.gpu_handle()?;
                let ordinal = input_handle.device_ordinal();
                let idx_handle = upload_index_i64(index, ordinal)?;
                let backend =
                    crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
                let src_handle = src_slab.gpu_handle()?;
                let h = match T::dtype() {
                    DType::F32 => backend.scatter_add_nd_f32(
                        input_handle,
                        &idx_handle,
                        src_handle,
                        input_shape,
                        index_shape,
                        dim,
                    )?,
                    DType::F64 => backend.scatter_add_nd_f64(
                        input_handle,
                        &idx_handle,
                        src_handle,
                        input_shape,
                        index_shape,
                        dim,
                    )?,
                    DType::F16 => backend.scatter_add_nd_f16(
                        input_handle,
                        &idx_handle,
                        src_handle,
                        input_shape,
                        index_shape,
                        dim,
                    )?,
                    DType::BF16 => backend.scatter_add_nd_bf16(
                        input_handle,
                        &idx_handle,
                        src_handle,
                        input_shape,
                        index_shape,
                        dim,
                    )?,
                    _ => unreachable!(),
                };
                let output_shape = input_shape.to_vec();
                let storage = TensorStorage::gpu(h);
                if needs_grad(&original_input, src) {
                    let grad_fn = Arc::new(crate::grad_fns::indexing::ScatterAddBackward {
                        input: original_input,
                        // The ORIGINAL src (not the kernel slab) — see scatter.
                        src: src.clone(),
                        dim,
                        index: index.to_vec(),
                        index_shape: index_shape.to_vec(),
                    });
                    return Tensor::from_operation(storage, output_shape, grad_fn);
                }
                return Tensor::from_storage(storage, output_shape, false);
            }
            _ => return Err(FerrotorchError::NotImplementedOnCuda { op: "scatter_add" }),
        }
    }

    let mut output = input.data_vec()?;
    let src_data = src.data_vec()?;
    // CORE-127 (#1821): coordinate-mapped src consumption (PyTorch parity).
    let src_shape = src.shape();
    let same_shape = src_shape == index_shape;

    let mut coords = vec![0usize; ndim];
    for i in 0..index_numel {
        let idx_val = index[i];
        let mut dst_coords = coords.clone();
        dst_coords[dim] = idx_val;
        let dst_flat = flat_index(&dst_coords, input_shape);
        output[dst_flat] += src_data[src_flat_offset(&coords, src_shape, same_shape, i)];

        if i + 1 < index_numel {
            increment_coords(&mut coords, index_shape);
        }
    }

    let output_shape = input_shape.to_vec();

    if needs_grad(input, src) {
        let grad_fn = Arc::new(crate::grad_fns::indexing::ScatterAddBackward {
            input: input.clone(),
            src: src.clone(),
            dim,
            index: index.to_vec(),
            index_shape: index_shape.to_vec(),
        });
        Tensor::from_operation(TensorStorage::cpu(output), output_shape, grad_fn)
    } else {
        Tensor::from_storage(TensorStorage::cpu(output), output_shape, false)
    }
}

// ---------------------------------------------------------------------------
// where_cond
// ---------------------------------------------------------------------------

/// Ternary selection: `output[i] = condition[i] ? x[i] : y[i]`.
///
/// `x` and `y` broadcast to their PyTorch common shape. Because `condition`
/// is a raw flat `&[bool]` slice with no shape metadata, it represents the
/// full flattened output mask and its length must equal the broadcast output
/// numel.
///
/// If either `x` or `y` requires grad, attaches a `WhereCondBackward`.
pub fn where_cond<T: Float>(
    condition: &[bool],
    x: &Tensor<T>,
    y: &Tensor<T>,
) -> FerrotorchResult<Tensor<T>> {
    if x.device() != y.device() {
        return Err(FerrotorchError::DeviceMismatch {
            expected: x.device(),
            got: y.device(),
        });
    }

    let output_shape = crate::shape::broadcast_shapes(x.shape(), y.shape()).map_err(|_| {
        FerrotorchError::ShapeMismatch {
            message: format!(
                "where_cond: x shape {:?} and y shape {:?} are not broadcast-compatible",
                x.shape(),
                y.shape()
            ),
        }
    })?;
    let output_numel = output_shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| FerrotorchError::InvalidArgument {
            message: format!(
                "where_cond: broadcast output shape {output_shape:?} element count overflows usize"
            ),
        })?;
    if condition.len() != output_numel {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "where_cond: condition length {} != broadcast output numel {} \
                 (broadcast shape {:?})",
                condition.len(),
                output_numel,
                output_shape
            ),
        });
    }

    let cond = crate::bool_tensor::BoolTensor::from_slice(condition, &output_shape)?;
    let cond = if x.is_cuda() {
        cond.to(x.device())?
    } else {
        cond
    };
    crate::grad_fns::indexing::where_cond_bcast(&cond, x, y)
}

fn where_cond_cpu<T: Float>(
    condition: &[bool],
    x: &Tensor<T>,
    y: &Tensor<T>,
) -> FerrotorchResult<Tensor<T>> {
    let x_data = x.data_vec()?;
    let y_data = y.data_vec()?;

    let output: Vec<T> = condition
        .iter()
        .zip(x_data.iter().zip(y_data.iter()))
        .map(|(&c, (&xv, &yv))| if c { xv } else { yv })
        .collect();

    let output_shape = x.shape().to_vec();

    if needs_grad(x, y) {
        // This entry point inherently has a host `&[bool]`; wrap it as a CPU
        // BoolTensor for storage. The backward struct now holds a BoolTensor
        // (CPU here; the resident `where_cond_bt` path stores a GPU one).
        let grad_fn = Arc::new(crate::grad_fns::indexing::WhereCondBackward {
            x: x.clone(),
            y: y.clone(),
            condition: crate::bool_tensor::BoolTensor::from_slice(condition, &output_shape)?,
        });
        Tensor::from_operation(TensorStorage::cpu(output), output_shape, grad_fn)
    } else {
        Tensor::from_storage(TensorStorage::cpu(output), output_shape, false)
    }
}

/// Shape-strict ternary selection taking a [`BoolTensor`] condition:
/// `output[i] = cond[i] ? x[i] : y[i]`.
///
/// All three tensors must share shape and device. When `cond`, `x`, and `y`
/// are CUDA-resident (same device), the select runs on the GPU through a real
/// PTX kernel dispatched on `x`'s dtype and the result stays GPU-resident — NO
/// host crossing (crosslink #1185 Phase 3c). Otherwise it uses the host bool
/// slice and CPU tensor data after the same-device check.
pub(crate) fn where_cond_bt_strict<T: Float>(
    cond: &crate::bool_tensor::BoolTensor,
    x: &Tensor<T>,
    y: &Tensor<T>,
) -> FerrotorchResult<Tensor<T>> {
    if x.shape() != y.shape() {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "where_cond_bt: x shape {:?} != y shape {:?}",
                x.shape(),
                y.shape()
            ),
        });
    }
    if cond.shape() != x.shape() {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "where_cond_bt: cond shape {:?} != x shape {:?}",
                cond.shape(),
                x.shape()
            ),
        });
    }
    if x.device() != y.device() {
        return Err(FerrotorchError::DeviceMismatch {
            expected: x.device(),
            got: y.device(),
        });
    }
    if x.device() != cond.device() {
        return Err(FerrotorchError::DeviceMismatch {
            expected: x.device(),
            got: cond.device(),
        });
    }

    // GPU-resident fast path: all three on the same CUDA device.
    if x.is_cuda() && y.is_cuda() && cond.is_cuda() {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        // #1660: normalise the narrowed-offset CUDA x/y operands to packed
        // offset-0 buffers before the select kernel reads element 0 (#1658
        // class). A row-narrowed view's BASE buffer is longer than `numel`, so
        // the kernel rejected the call ("where_cond: numel mismatch (cond 6,
        // x 8, y 6)"); `.contiguous()` materialises the logical view on-device
        // (strided_copy; cheap clone when already offset-0). The autograd
        // capture below stores the packed operands so the backward agrees.
        let x = x.contiguous()?;
        let y = y.contiguous()?;
        let h = backend.where_cond(cond.gpu_handle()?, x.gpu_handle()?, y.gpu_handle()?)?;
        let storage = TensorStorage::gpu(h);
        let output_shape = x.shape().to_vec();

        if needs_grad(&x, &y) {
            // Store the resident cond directly (cheap Arc/clone-on-storage) — the
            // backward routes through the resident `where_cond` VJP with NO host
            // crossing (crosslink #1187 Phase 3d). No `cond.to(Cpu)`.
            let grad_fn = Arc::new(crate::grad_fns::indexing::WhereCondBackward {
                x: x.clone(),
                y: y.clone(),
                condition: cond.clone(),
            });
            return Tensor::from_operation(storage, output_shape, grad_fn);
        }
        return Tensor::from_storage(storage, output_shape, false);
    }

    where_cond_cpu(cond.data()?, x, y)
}

/// Ternary selection taking a [`BoolTensor`] condition. Mirrors
/// `torch.where(cond, x, y)`: condition, `x`, and `y` broadcast to their common
/// shape before selecting, and CUDA operands stay on-device when all tensors
/// live on the same CUDA device.
pub fn where_cond_bt<T: Float>(
    cond: &crate::bool_tensor::BoolTensor,
    x: &Tensor<T>,
    y: &Tensor<T>,
) -> FerrotorchResult<Tensor<T>> {
    crate::grad_fns::indexing::where_cond_bcast(cond, x, y)
}

/// `masked_select(input, mask)` — return a 1-D tensor of the elements of
/// `input` where `mask` is true, in flat C-order. Mirrors
/// `torch.masked_select`: `input` and `mask` are broadcast to a common shape
/// before compaction.
///
/// On CUDA (input + mask resident, same device) this runs a GPU stream
/// compaction (crosslink #1185 Phase 3c): an on-device count of the true mask
/// bytes sizes the output, then a compaction kernel writes the kept elements —
/// the result stays GPU-resident. The single integer COUNT crosses to the host
/// to size the data-dependent output; that scalar is the result SHAPE, not a
/// data round-trip (PyTorch parity: a CUDA sync sizes `masked_select`'s
/// output).
///
/// `masked_select` IS differentiable (PyTorch parity). When `input` requires
/// grad and grad is enabled, the result carries a `MaskedSelectBackward` grad_fn
/// that scatters the compacted gradient back into a zeros tensor of
/// `input.numel()` at the selected positions. On the GPU path the backward stays
/// resident via the `masked_scatter` kernel (crosslink #1187 Phase 3d).
pub fn masked_select<T: Float>(
    input: &Tensor<T>,
    mask: &crate::bool_tensor::BoolTensor,
) -> FerrotorchResult<Tensor<T>> {
    if input.device() != mask.device() {
        return Err(FerrotorchError::DeviceMismatch {
            expected: input.device(),
            got: mask.device(),
        });
    }

    if input.shape() == mask.shape() {
        return masked_select_strict(input, mask);
    }

    let common = crate::shape::broadcast_shapes(input.shape(), mask.shape())?;
    let input_b = crate::grad_fns::shape::expand(input, &common)?;
    let mask_b = crate::grad_fns::indexing::broadcast_bool_tensor(mask, &common)?;
    masked_select_strict(&input_b, &mask_b)
}

/// Shape-strict masked-select implementation used after PyTorch-style
/// broadcasting has already produced equal-shape operands.
pub(crate) fn masked_select_strict<T: Float>(
    input: &Tensor<T>,
    mask: &crate::bool_tensor::BoolTensor,
) -> FerrotorchResult<Tensor<T>> {
    if mask.numel() != input.numel() {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "masked_select: mask numel {} != input numel {}",
                mask.numel(),
                input.numel()
            ),
        });
    }

    if input.is_cuda() && mask.is_cuda() {
        if input.device() != mask.device() {
            return Err(FerrotorchError::DeviceMismatch {
                expected: input.device(),
                got: mask.device(),
            });
        }
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        // #1660: normalise the narrowed-offset CUDA input to a packed offset-0
        // buffer before the compaction kernel reads element 0 (#1658 class). A
        // row-narrowed view's BASE buffer is longer than `numel`, which the
        // kernel rejected ("input numel 8 != mask numel 6"); `.contiguous()`
        // materialises the logical view on-device (strided_copy; cheap clone
        // when already offset-0). The backward capture below intentionally
        // keeps the original logical tensor so gradients route through views
        // and broadcast `ExpandBackward` nodes instead of stopping at this
        // packed kernel input.
        let original_input = input.clone();
        let packed_input = input.contiguous()?;
        let (handle, len) =
            backend.masked_select(packed_input.gpu_handle()?, mask.gpu_handle()?)?;
        let storage = TensorStorage::gpu(handle);

        // PyTorch parity: masked_select IS differentiable. Attach the backward
        // (scatter the compacted grad back into a zeros tensor at the true mask
        // positions). Store the resident mask directly — the backward stays
        // GPU-resident, NO host crossing (crosslink #1187 Phase 3d).
        if original_input.requires_grad() && is_grad_enabled() {
            let grad_fn = Arc::new(crate::grad_fns::indexing::MaskedSelectBackward {
                input: original_input,
                mask: mask.clone(),
            });
            return Tensor::from_operation(storage, vec![len], grad_fn);
        }
        return Tensor::from_storage(storage, vec![len], false);
    }

    // CPU (or mixed-residency) path: walk the host data + mask. `mask.data()?` /
    // `input.data_vec()` error on a GPU operand whose counterpart is on host,
    // which is the correct device-mismatch signal.
    let data = input.data_vec()?;
    let mask_h = mask.data()?;
    let out: Vec<T> = data
        .iter()
        .zip(mask_h.iter())
        .filter_map(|(&v, &m)| if m { Some(v) } else { None })
        .collect();
    let len = out.len();
    let storage = TensorStorage::cpu(out);

    if input.requires_grad() && is_grad_enabled() {
        let grad_fn = Arc::new(crate::grad_fns::indexing::MaskedSelectBackward {
            input: input.clone(),
            mask: mask.clone(),
        });
        Tensor::from_operation(storage, vec![len], grad_fn)
    } else {
        Tensor::from_storage(storage, vec![len], false)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::graph::backward;
    use crate::autograd::no_grad;
    use crate::storage::TensorStorage;
    use crate::tensor::GradFn;

    /// Create a leaf tensor from a flat slice and shape.
    fn leaf(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
        Tensor::from_storage(
            TensorStorage::cpu(data.to_vec()),
            shape.to_vec(),
            requires_grad,
        )
        .unwrap()
    }

    // -----------------------------------------------------------------------
    // gather forward
    // -----------------------------------------------------------------------

    #[test]
    fn test_gather_1d() {
        // input = [10, 20, 30, 40], gather along dim 0 with index [3, 0, 2]
        let input = leaf(&[10.0, 20.0, 30.0, 40.0], &[4], false);
        let index = &[3, 0, 2];
        let result = gather(&input, 0, index, &[3]).unwrap();
        assert_eq!(result.shape(), &[3]);
        assert_eq!(result.data().unwrap(), &[40.0, 10.0, 30.0]);
    }

    #[test]
    fn test_gather_2d_dim0() {
        // input = [[1, 2], [3, 4], [5, 6]]  shape [3, 2]
        // index = [[2, 0], [1, 1]]           shape [2, 2]
        // output[i][j] = input[index[i][j]][j]
        // output = [[5, 2], [3, 4]]
        let input = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], false);
        let index = &[2, 0, 1, 1];
        let result = gather(&input, 0, index, &[2, 2]).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.data().unwrap(), &[5.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_gather_2d_dim1() {
        // input = [[1, 2, 3], [4, 5, 6]]  shape [2, 3]
        // index = [[0, 2], [1, 0]]        shape [2, 2]
        // output[i][j] = input[i][index[i][j]]
        // output = [[1, 3], [5, 4]]
        let input = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        let index = &[0, 2, 1, 0];
        let result = gather(&input, 1, index, &[2, 2]).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.data().unwrap(), &[1.0, 3.0, 5.0, 4.0]);
    }

    #[test]
    fn test_gather_out_of_bounds() {
        let input = leaf(&[1.0, 2.0, 3.0], &[3], false);
        let result = gather(&input, 0, &[5], &[1]);
        assert!(result.is_err());
    }

    #[test]
    fn test_gather_ndim_mismatch() {
        // input is 2D, index is 1D
        let input = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2], false);
        let result = gather(&input, 0, &[0, 1], &[2]);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // scatter forward
    // -----------------------------------------------------------------------

    #[test]
    fn test_scatter_1d() {
        // input = [0, 0, 0, 0, 0], scatter src=[10, 20, 30] at index=[1, 3, 0]
        let input = leaf(&[0.0; 5], &[5], false);
        let src = leaf(&[10.0, 20.0, 30.0], &[3], false);
        let result = scatter(&input, 0, &[1, 3, 0], &[3], &src).unwrap();
        assert_eq!(result.data().unwrap(), &[30.0, 10.0, 0.0, 20.0, 0.0]);
    }

    #[test]
    fn test_scatter_2d_dim0() {
        // input = [[0,0],[0,0],[0,0]]  shape [3, 2]
        // src   = [[1,2]]              shape [1, 2]
        // index = [[2,0]]              shape [1, 2]
        // output[index[i][j]][j] = src[i][j]
        // output = [[0,2],[0,0],[1,0]]
        let input = leaf(&[0.0; 6], &[3, 2], false);
        let src = leaf(&[1.0, 2.0], &[1, 2], false);
        let result = scatter(&input, 0, &[2, 0], &[1, 2], &src).unwrap();
        assert_eq!(result.shape(), &[3, 2]);
        assert_eq!(result.data().unwrap(), &[0.0, 2.0, 0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_scatter_2d_dim1() {
        // input = [[0,0,0],[0,0,0]]  shape [2, 3]
        // src   = [[5],[6]]          shape [2, 1]
        // index = [[2],[0]]          shape [2, 1]
        // output[i][index[i][j]] = src[i][j]
        // output = [[0,0,5],[6,0,0]]
        let input = leaf(&[0.0; 6], &[2, 3], false);
        let src = leaf(&[5.0, 6.0], &[2, 1], false);
        let result = scatter(&input, 1, &[2, 0], &[2, 1], &src).unwrap();
        assert_eq!(result.data().unwrap(), &[0.0, 0.0, 5.0, 6.0, 0.0, 0.0]);
    }

    // -----------------------------------------------------------------------
    // scatter_add forward
    // -----------------------------------------------------------------------

    #[test]
    fn test_scatter_add_1d() {
        // input = [1, 2, 3], scatter_add src=[10, 20, 30] at index=[0, 2, 0]
        // output = [1+10+30, 2, 3+20] = [41, 2, 23]
        let input = leaf(&[1.0, 2.0, 3.0], &[3], false);
        let src = leaf(&[10.0, 20.0, 30.0], &[3], false);
        let result = scatter_add(&input, 0, &[0, 2, 0], &[3], &src).unwrap();
        assert_eq!(result.data().unwrap(), &[41.0, 2.0, 23.0]);
    }

    #[test]
    fn test_scatter_add_2d_dim0() {
        // input = [[0,0],[0,0]]  shape [2, 2]
        // src   = [[1,2],[3,4],[5,6]]  shape [3, 2]
        // index = [[0,1],[1,0],[0,0]]  shape [3, 2]
        //
        // output[index[i][j]][j] += src[i][j]
        // (0,0) += 1, (1,0) += 2
        // (1,0) += 3, (0,1) += 4
        // (0,0) += 5, (0,1) += 6
        // output = [[0+1+5, 0+4+6], [0+3, 0+2]] = [[6, 10], [3, 2]]
        let input = leaf(&[0.0; 4], &[2, 2], false);
        let src = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], false);
        let result = scatter_add(&input, 0, &[0, 1, 1, 0, 0, 0], &[3, 2], &src).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.data().unwrap(), &[6.0, 10.0, 3.0, 2.0]);
    }

    // -----------------------------------------------------------------------
    // where_cond forward
    // -----------------------------------------------------------------------

    #[test]
    fn test_where_cond_basic() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[4], false);
        let y = leaf(&[10.0, 20.0, 30.0, 40.0], &[4], false);
        let cond = [true, false, true, false];
        let result = where_cond(&cond, &x, &y).unwrap();
        assert_eq!(result.data().unwrap(), &[1.0, 20.0, 3.0, 40.0]);
    }

    #[test]
    fn test_where_cond_all_true() {
        let x = leaf(&[1.0, 2.0], &[2], false);
        let y = leaf(&[10.0, 20.0], &[2], false);
        let result = where_cond(&[true, true], &x, &y).unwrap();
        assert_eq!(result.data().unwrap(), &[1.0, 2.0]);
    }

    #[test]
    fn test_where_cond_all_false() {
        let x = leaf(&[1.0, 2.0], &[2], false);
        let y = leaf(&[10.0, 20.0], &[2], false);
        let result = where_cond(&[false, false], &x, &y).unwrap();
        assert_eq!(result.data().unwrap(), &[10.0, 20.0]);
    }

    #[test]
    fn test_where_cond_shape_mismatch() {
        let x = leaf(&[1.0, 2.0], &[2], false);
        let y = leaf(&[1.0, 2.0, 3.0], &[3], false);
        let result = where_cond(&[true, false], &x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_where_cond_cond_length_mismatch() {
        let x = leaf(&[1.0, 2.0], &[2], false);
        let y = leaf(&[10.0, 20.0], &[2], false);
        let result = where_cond(&[true], &x, &y);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // gather backward
    // -----------------------------------------------------------------------

    #[test]
    fn test_gather_backward_1d() {
        // input = [10, 20, 30], gather at [2, 0, 0] -> output = [30, 10, 10]
        // grad_output = [1, 1, 1]
        // grad_input: scatter_add of [1,1,1] at [2,0,0] into zeros(3)
        //   = [2, 0, 1]
        let input = leaf(&[10.0, 20.0, 30.0], &[3], true);
        let result = gather(&input, 0, &[2, 0, 0], &[3]).unwrap();

        assert!(result.requires_grad());

        let grad_output = leaf(&[1.0, 1.0, 1.0], &[3], false);
        let grad_fn = result.grad_fn().unwrap();
        let grads = grad_fn.backward(&grad_output).unwrap();
        let gi = grads[0].as_ref().unwrap();
        let gd = gi.data().unwrap();
        assert!((gd[0] - 2.0).abs() < 1e-6, "grad[0]={}, expected 2", gd[0]);
        assert!((gd[1] - 0.0).abs() < 1e-6, "grad[1]={}, expected 0", gd[1]);
        assert!((gd[2] - 1.0).abs() < 1e-6, "grad[2]={}, expected 1", gd[2]);
    }

    #[test]
    fn test_gather_backward_2d() {
        // input shape [2, 3], gather dim=1, index shape [2, 2]
        // input = [[1,2,3],[4,5,6]]
        // index = [[0, 2], [1, 0]]
        // output = [[1,3],[5,4]]
        //
        // grad_output = [[1,1],[1,1]]
        // grad_input: scatter_add along dim=1
        //   row 0: idx [0,2] -> [1, 0, 1]
        //   row 1: idx [1,0] -> [1, 1, 0]
        //   grad_input = [[1,0,1],[1,1,0]]
        let input = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
        let result = gather(&input, 1, &[0, 2, 1, 0], &[2, 2]).unwrap();

        let grad_output = leaf(&[1.0, 1.0, 1.0, 1.0], &[2, 2], false);
        let grad_fn = result.grad_fn().unwrap();
        let grads = grad_fn.backward(&grad_output).unwrap();
        let gi = grads[0].as_ref().unwrap();
        let gd = gi.data().unwrap();
        assert_eq!(gi.shape(), &[2, 3]);
        // row 0: [1, 0, 1]
        assert!((gd[0] - 1.0).abs() < 1e-6);
        assert!((gd[1] - 0.0).abs() < 1e-6);
        assert!((gd[2] - 1.0).abs() < 1e-6);
        // row 1: [1, 1, 0]
        assert!((gd[3] - 1.0).abs() < 1e-6);
        assert!((gd[4] - 1.0).abs() < 1e-6);
        assert!((gd[5] - 0.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // scatter backward
    // -----------------------------------------------------------------------

    #[test]
    fn test_scatter_backward_input() {
        // scatter zeros out the positions that were overwritten.
        // input = [1, 2, 3, 4, 5], scatter src at [1, 3]
        // grad wrt input: ones everywhere except positions 1 and 3
        // -> [1, 0, 1, 0, 1]
        let input = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5], true);
        let src = leaf(&[10.0, 20.0], &[2], false);
        let result = scatter(&input, 0, &[1, 3], &[2], &src).unwrap();

        let grad_output = leaf(&[1.0; 5], &[5], false);
        let grad_fn = result.grad_fn().unwrap();
        let grads = grad_fn.backward(&grad_output).unwrap();
        let gi = grads[0].as_ref().unwrap();
        let gd = gi.data().unwrap();
        assert_eq!(gd, &[1.0, 0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_scatter_backward_src() {
        // scatter grad wrt src is gather from grad_output at index positions.
        // input = [0, 0, 0], scatter src at [2, 0]
        // grad_output = [10, 20, 30]
        // grad_src = [grad_output[2], grad_output[0]] = [30, 10]
        let input = leaf(&[0.0; 3], &[3], false);
        let src = leaf(&[1.0, 2.0], &[2], true);
        let result = scatter(&input, 0, &[2, 0], &[2], &src).unwrap();

        let grad_output = leaf(&[10.0, 20.0, 30.0], &[3], false);
        let grad_fn = result.grad_fn().unwrap();
        let grads = grad_fn.backward(&grad_output).unwrap();

        // grads[0] is for input (not requiring grad -> None)
        assert!(grads[0].is_none());
        // grads[1] is for src
        let gs = grads[1].as_ref().unwrap();
        let gd = gs.data().unwrap();
        assert_eq!(gd, &[30.0, 10.0]);
    }

    // -----------------------------------------------------------------------
    // scatter_add backward
    // -----------------------------------------------------------------------

    #[test]
    fn test_scatter_add_backward_input() {
        // scatter_add backward for input is just grad_output (identity).
        let input = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let src = leaf(&[10.0, 20.0], &[2], false);
        let result = scatter_add(&input, 0, &[0, 2], &[2], &src).unwrap();

        let grad_output = leaf(&[5.0, 6.0, 7.0], &[3], false);
        let grad_fn = result.grad_fn().unwrap();
        let grads = grad_fn.backward(&grad_output).unwrap();
        let gi = grads[0].as_ref().unwrap();
        assert_eq!(gi.data().unwrap(), &[5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_scatter_add_backward_src() {
        // scatter_add backward for src is gather from grad_output.
        // index = [2, 0], grad_output = [5, 6, 7]
        // grad_src = [grad_output[2], grad_output[0]] = [7, 5]
        let input = leaf(&[1.0, 2.0, 3.0], &[3], false);
        let src = leaf(&[10.0, 20.0], &[2], true);
        let result = scatter_add(&input, 0, &[2, 0], &[2], &src).unwrap();

        let grad_output = leaf(&[5.0, 6.0, 7.0], &[3], false);
        let grad_fn = result.grad_fn().unwrap();
        let grads = grad_fn.backward(&grad_output).unwrap();

        assert!(grads[0].is_none());
        let gs = grads[1].as_ref().unwrap();
        assert_eq!(gs.data().unwrap(), &[7.0, 5.0]);
    }

    // -----------------------------------------------------------------------
    // where_cond backward
    // -----------------------------------------------------------------------

    #[test]
    fn test_where_cond_backward_x() {
        // where_cond grad for x: grad_output where condition is true, 0 otherwise.
        let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[4], true);
        let y = leaf(&[10.0, 20.0, 30.0, 40.0], &[4], false);
        let cond = [true, false, true, false];
        let result = where_cond(&cond, &x, &y).unwrap();

        let grad_output = leaf(&[1.0, 2.0, 3.0, 4.0], &[4], false);
        let grad_fn = result.grad_fn().unwrap();
        let grads = grad_fn.backward(&grad_output).unwrap();

        let gx = grads[0].as_ref().unwrap();
        assert_eq!(gx.data().unwrap(), &[1.0, 0.0, 3.0, 0.0]);
        assert!(grads[1].is_none());
    }

    #[test]
    fn test_where_cond_backward_y() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[4], false);
        let y = leaf(&[10.0, 20.0, 30.0, 40.0], &[4], true);
        let cond = [true, false, true, false];
        let result = where_cond(&cond, &x, &y).unwrap();

        let grad_output = leaf(&[1.0, 2.0, 3.0, 4.0], &[4], false);
        let grad_fn = result.grad_fn().unwrap();
        let grads = grad_fn.backward(&grad_output).unwrap();

        assert!(grads[0].is_none());
        let gy = grads[1].as_ref().unwrap();
        assert_eq!(gy.data().unwrap(), &[0.0, 2.0, 0.0, 4.0]);
    }

    #[test]
    fn test_where_cond_backward_both() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let y = leaf(&[10.0, 20.0, 30.0], &[3], true);
        let cond = [false, true, false];
        let result = where_cond(&cond, &x, &y).unwrap();

        let grad_output = leaf(&[5.0, 6.0, 7.0], &[3], false);
        let grad_fn = result.grad_fn().unwrap();
        let grads = grad_fn.backward(&grad_output).unwrap();

        let gx = grads[0].as_ref().unwrap();
        assert_eq!(gx.data().unwrap(), &[0.0, 6.0, 0.0]);
        let gy = grads[1].as_ref().unwrap();
        assert_eq!(gy.data().unwrap(), &[5.0, 0.0, 7.0]);
    }

    // -----------------------------------------------------------------------
    // no_grad context
    // -----------------------------------------------------------------------

    #[test]
    fn test_gather_no_grad() {
        let input = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let result = no_grad(|| gather(&input, 0, &[2, 0], &[2])).unwrap();
        assert!(!result.requires_grad());
        assert!(result.grad_fn().is_none());
    }

    #[test]
    fn test_where_cond_no_grad() {
        let x = leaf(&[1.0, 2.0], &[2], true);
        let y = leaf(&[3.0, 4.0], &[2], true);
        let result = no_grad(|| where_cond(&[true, false], &x, &y)).unwrap();
        assert!(!result.requires_grad());
    }

    // -----------------------------------------------------------------------
    // End-to-end backward through autograd
    // -----------------------------------------------------------------------

    #[test]
    fn test_gather_end_to_end_backward() {
        let input = leaf(&[10.0, 20.0, 30.0, 40.0], &[4], true);
        let gathered = gather(&input, 0, &[1, 3], &[2]).unwrap();

        // Sum to scalar via inline SumBackward.
        let data = gathered.data().unwrap();
        let total: f32 = data.iter().sum();

        #[derive(Debug)]
        struct SumBackward<T: Float> {
            input: Tensor<T>,
        }
        impl<T: Float> GradFn<T> for SumBackward<T> {
            fn backward(
                &self,
                grad_output: &Tensor<T>,
            ) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
                let go_val = grad_output.data()?[0];
                let grad = vec![go_val; self.input.numel()];
                let t = Tensor::from_storage(
                    TensorStorage::cpu(grad),
                    self.input.shape().to_vec(),
                    false,
                )?;
                Ok(vec![Some(t)])
            }
            fn inputs(&self) -> Vec<&Tensor<T>> {
                vec![&self.input]
            }
            fn name(&self) -> &'static str {
                "SumBackward"
            }
        }

        let loss = Tensor::from_operation(
            TensorStorage::cpu(vec![total]),
            vec![],
            Arc::new(SumBackward {
                input: gathered.clone(),
            }),
        )
        .unwrap();

        backward(&loss).unwrap();

        let grad = input.grad().unwrap().unwrap();
        let gd = grad.data().unwrap();
        // indices [1, 3]: grad = [0, 1, 0, 1]
        assert!((gd[0] - 0.0).abs() < 1e-6);
        assert!((gd[1] - 1.0).abs() < 1e-6);
        assert!((gd[2] - 0.0).abs() < 1e-6);
        assert!((gd[3] - 1.0).abs() < 1e-6);
    }
}
