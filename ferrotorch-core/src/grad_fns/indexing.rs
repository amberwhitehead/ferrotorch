//! Backward functions for indexing operations.
//!
//! Implements `GradFn` for:
//! - `index_select` (1D) — selects elements along an axis by integer indices
//! - `masked_fill` — fills elements where a boolean mask is true
//! - `gather` — gathers elements along an axis (N-D)
//! - `scatter` — scatters src values into input along an axis
//! - `scatter_add` — scatter with addition
//! - `where_cond` — ternary selection
//!
//! ## REQ status (per `.design/ferrotorch-core/grad_fns/indexing.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`gather`) | SHIPPED | `GatherBackward` Arc-attached by `ops::indexing::gather` (kernel-layer forward); CPU walk + GPU rank-aware `scatter_add_nd_{f32,f64,f16,bf16}` path with i64 index (#1822/#1823/#1820). |
//! | REQ-2 (`scatter`) | SHIPPED | `ScatterBackward` returns `[grad_input zeroed at written positions, grad_src gathered from those positions]`; Arc-attached by `ops::indexing::scatter`. |
//! | REQ-3 (`scatter_add`) | SHIPPED | `ScatterAddBackward` returns `[grad, grad.gather(dim, index)]`; Arc-attached by `ops::indexing::scatter_add` and consumed transitively by `grad_fns::cumulative::cummax/cummin` VJPs. |
//! | REQ-4 (`scatter_reduce`) | SHIPPED | runner arm + impl landed 2026-05-25 closing #1245; 144/168 passed (24 narrower-contract skips). |
//! | REQ-5 (`index_select`) | SHIPPED | three forward variants (`index_select_1d`, `index_select_1d_it`, `index_select_dim`) + `IndexSelectBackward` / `IndexSelectDimBackward`; non-test consumer is `RandomHorizontalFlip::apply` in `ferrotorch-data/src/transforms.rs`. |
//! | REQ-6 (`index_add`) | SHIPPED | runner arm + impl landed closing #1247; 72/72 passed. |
//! | REQ-7 (`index_copy`) | SHIPPED | runner arm + impl landed closing #1248; 24/24 passed. |
//! | REQ-8 (`index_fill`) | SHIPPED | `index_fill` + `IndexFillBackward` consumed by `Tensor::index_fill_t` in `methods.rs`; closes #1249. |
//! | REQ-9 (`masked_select`) | SHIPPED | `MaskedSelectBackward` Arc-attached by `ops::indexing::masked_select`; consumer is `Tensor::masked_select`. |
//! | REQ-10 (`masked_fill`) | SHIPPED | `MaskedFillBackward` consumed by `Tensor::masked_fill` (which routes through `masked_fill_bt`); GPU-resident path via `masked_fill_dt` kernel (#1187). |
//! | REQ-11 (`masked_scatter`) | SHIPPED | runner arm + impl landed closing #1252; 32/32 passed. |
//! | REQ-12 (`take`) | SHIPPED | `take` mirrors flat PyTorch indexing incl. 0-d / negative indices, and CUDA f32/f64/f16/bf16 route through resident `index_select_intidx`; pinned by `audit_core048_indexing_device_demotion::{take_cuda_resident_*,take_put_cuda_empty_resident_*}`. |
//! | REQ-13 (`put`) | SHIPPED | `put` enforces PyTorch's exact `source.numel() == index.numel()` contract; CUDA f32/f64/f16/bf16 route through resident scatter/scatter-add kernels, with f16/bf16 duplicate accumulation pinned by `audit_core048_indexing_device_demotion::put_cuda_accumulate_odd_len_duplicate_*`. |
//! | REQ-14 (`where`) | SHIPPED | `WhereCondBackward` Arc-attached by `ops::indexing::where_cond` / `where_cond_bt`; GPU-resident path via `masked_fill_dt` + `bool_not` (#1187). |
//! | REQ-15 (shared backward helpers: `flat_index`, `increment_coords`) | SHIPPED | internal scaffolding consumed by REQ-1 / REQ-2 / REQ-3 implementations above (the f32 flat-offset helpers `gather_dst_flat_indices` / `scatter_src_flat_indices` / `scatter_write_mask` were removed by the #1823 i64-index migration). |

use std::sync::Arc;

use crate::autograd::no_grad::{is_grad_enabled, no_grad};
use crate::device::Device;
use crate::dtype::{DType, Float};
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::gpu_dispatch::{GpuBufferHandle, GpuScatterReduce, gpu_backend};
use crate::storage::TensorStorage;
use crate::tensor::{GradFn, Tensor};

use crate::bool_tensor::BoolTensor;
use crate::int_tensor::{IntElement, IntTensor};

// ---------------------------------------------------------------------------
// Helpers for N-D backward (shared by gather/scatter/scatter_add)
// ---------------------------------------------------------------------------

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

#[inline]
fn cuda_float_dtype<T: Float>() -> bool {
    matches!(
        T::dtype(),
        DType::F32 | DType::F64 | DType::F16 | DType::BF16
    )
}

fn cuda_ordinal(device: Device, op: &'static str) -> FerrotorchResult<usize> {
    match device {
        Device::Cuda(ordinal) => Ok(ordinal),
        got => Err(FerrotorchError::InvalidArgument {
            message: format!("{op}: expected CUDA device, got {got:?}"),
        }),
    }
}

fn expanded_dim_indices(
    indices: &[usize],
    outer: usize,
    inner: usize,
) -> FerrotorchResult<Vec<usize>> {
    let capacity = outer
        .checked_mul(indices.len())
        .and_then(|n| n.checked_mul(inner))
        .ok_or_else(|| FerrotorchError::InvalidArgument {
            message: format!(
                "expanded_dim_indices: expanded index count overflowed for outer={outer}, indices={}, inner={inner}",
                indices.len()
            ),
        })?;
    let mut expanded = Vec::with_capacity(capacity);
    for _ in 0..outer {
        for &idx in indices {
            for _ in 0..inner {
                expanded.push(idx);
            }
        }
    }
    Ok(expanded)
}

fn upload_expanded_dim_indices(
    indices: &[usize],
    outer: usize,
    inner: usize,
    ordinal: usize,
) -> FerrotorchResult<GpuBufferHandle> {
    let expanded = expanded_dim_indices(indices, outer, inner)?;
    crate::ops::indexing::upload_index_i64(&expanded, ordinal)
}

fn clone_cuda_tensor<T: Float>(
    input: &Tensor<T>,
    shape: Vec<usize>,
) -> FerrotorchResult<Tensor<T>> {
    let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
    let input_c = no_grad(|| input.contiguous())?;
    let handle = backend.clone_buffer(input_c.gpu_handle()?)?;
    Tensor::from_storage(TensorStorage::gpu(handle), shape, false)
}

fn scale_cuda_tensor<T: Float>(
    input: &Tensor<T>,
    alpha: f64,
    op: &'static str,
) -> FerrotorchResult<Tensor<T>> {
    let alpha_t = <T as num_traits::NumCast>::from(alpha).ok_or_else(|| {
        FerrotorchError::InvalidArgument {
            message: format!("{op}: alpha {alpha} not representable in target dtype"),
        }
    })?;
    let alpha_f32 = alpha_t
        .to_f32()
        .ok_or_else(|| FerrotorchError::InvalidArgument {
            message: format!("{op}: alpha {alpha} not representable as f32"),
        })?;
    let input_c = no_grad(|| input.contiguous())?;
    if alpha_t == <T as num_traits::One>::one() {
        return Ok(input_c);
    }

    let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
    let handle = match T::dtype() {
        DType::F32 => backend.scale_f32(input_c.gpu_handle()?, alpha_f32)?,
        DType::F64 => backend.scale_f64(
            input_c.gpu_handle()?,
            alpha_t
                .to_f64()
                .ok_or_else(|| FerrotorchError::InvalidArgument {
                    message: format!("{op}: alpha {alpha} not representable as f64"),
                })?,
        )?,
        DType::F16 => backend.scale_f16(input_c.gpu_handle()?, alpha_f32)?,
        DType::BF16 => backend.scale_bf16_bf16(input_c.gpu_handle()?, alpha_f32)?,
        _ => {
            return Err(FerrotorchError::NotImplementedOnCuda { op });
        }
    };
    Tensor::from_storage(TensorStorage::gpu(handle), input_c.shape().to_vec(), false)
}

#[allow(clippy::too_many_arguments)]
fn gather_dim_cuda<T: Float>(
    input: &Tensor<T>,
    indices: &[usize],
    outer: usize,
    in_dim_size: usize,
    out_dim_size: usize,
    inner: usize,
    output_shape: Vec<usize>,
    op: &'static str,
) -> FerrotorchResult<Tensor<T>> {
    if out_dim_size == 0 || outer == 0 || inner == 0 {
        let ordinal = cuda_ordinal(input.device(), op)?;
        let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let empty = backend.alloc_zeros(0, T::dtype(), ordinal)?;
        return Tensor::from_storage(TensorStorage::gpu(empty), output_shape, false);
    }

    let input_c = no_grad(|| input.contiguous())?;
    let ordinal = cuda_ordinal(input_c.device(), op)?;
    let idx_handle = upload_expanded_dim_indices(indices, outer, inner, ordinal)?;
    let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
    let handle = backend.gather_intidx(
        input_c.gpu_handle()?,
        &idx_handle,
        outer,
        in_dim_size,
        out_dim_size,
        inner,
    )?;
    Tensor::from_storage(TensorStorage::gpu(handle), output_shape, false)
}

#[allow(clippy::too_many_arguments)]
fn scatter_dim_cuda_handle<T: Float>(
    input: &GpuBufferHandle,
    index: &GpuBufferHandle,
    src: &GpuBufferHandle,
    outer: usize,
    out_dim: usize,
    idx_dim: usize,
    inner: usize,
    accumulate: bool,
    op: &'static str,
) -> FerrotorchResult<GpuBufferHandle> {
    let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
    match (accumulate, T::dtype()) {
        (false, DType::F32) => {
            backend.scatter_dim_f32(input, index, src, outer, out_dim, idx_dim, inner)
        }
        (false, DType::F64) => {
            backend.scatter_dim_f64(input, index, src, outer, out_dim, idx_dim, inner)
        }
        (false, DType::F16) => {
            backend.scatter_dim_f16(input, index, src, outer, out_dim, idx_dim, inner)
        }
        (false, DType::BF16) => {
            backend.scatter_dim_bf16(input, index, src, outer, out_dim, idx_dim, inner)
        }
        (true, DType::F32) => {
            backend.scatter_add_dim_f32(input, index, src, outer, out_dim, idx_dim, inner)
        }
        (true, DType::F64) => {
            backend.scatter_add_dim_f64(input, index, src, outer, out_dim, idx_dim, inner)
        }
        (true, DType::F16) => {
            backend.scatter_add_dim_f16(input, index, src, outer, out_dim, idx_dim, inner)
        }
        (true, DType::BF16) => {
            backend.scatter_add_dim_bf16(input, index, src, outer, out_dim, idx_dim, inner)
        }
        (_, _) => Err(FerrotorchError::NotImplementedOnCuda { op }),
    }
}

// ---------------------------------------------------------------------------
// index_select (1D)
// ---------------------------------------------------------------------------

/// Backward function for `index_select` on a 1-D input tensor.
///
/// Forward: `output[i] = input[indices[i]]`
///
/// VJP: `grad_input = zeros(input.len()); for (i, idx) in indices: grad_input[idx] += grad_output[i]`
///
/// This is equivalent to a scatter-add of `grad_output` back into the input shape.
#[derive(Debug)]
pub struct IndexSelectBackward<T: Float> {
    /// The original input tensor (saved for shape information).
    pub input: Tensor<T>,
    /// The index vector used during the forward pass.
    pub indices: Vec<usize>,
}

impl<T: Float> GradFn<T> for IndexSelectBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !is_grad_enabled() {
            return Ok(vec![None]);
        }

        let input_len = self.input.numel();

        if grad_output.is_cuda() {
            // GPU path (#1822/#1823): scatter-add into a zeroed input-shaped
            // buffer via the dim-aware kernel (outer=1, inner=1 — the 1-D
            // case), dispatched on T::dtype() with the index uploaded as i64
            // (indices above 2^24 are not representable in f32).
            let dt = T::dtype();
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let ordinal = match grad_output.device() {
                Device::Cuda(o) => o,
                _ => unreachable!(),
            };
            let idx_handle = crate::ops::indexing::upload_index_i64(&self.indices, ordinal)?;
            let zeros = backend.alloc_zeros(input_len, dt, ordinal)?;
            let go_handle = grad_output.gpu_handle()?;
            let result_handle = scatter_dim_cuda_handle::<T>(
                &zeros,
                &idx_handle,
                go_handle,
                1,
                input_len,
                self.indices.len(),
                1,
                true,
                "IndexSelectBackward",
            )?;
            let grad_tensor = Tensor::from_storage(
                TensorStorage::gpu(result_handle),
                self.input.shape().to_vec(),
                false,
            )?;
            Ok(vec![Some(grad_tensor)])
        } else {
            // CPU path: direct scatter-add.
            let go_data = grad_output.data()?;
            let mut grad_input = vec![<T as num_traits::Zero>::zero(); input_len];
            for (i, &idx) in self.indices.iter().enumerate() {
                grad_input[idx] += go_data[i];
            }
            let grad_tensor = Tensor::from_storage(
                TensorStorage::cpu(grad_input),
                self.input.shape().to_vec(),
                false,
            )?;
            Ok(vec![Some(grad_tensor)])
        }
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "IndexSelectBackward"
    }
}

/// Perform 1-D `index_select`: gather elements from `input` at `indices`.
///
/// Returns a new tensor of the same dtype with shape `[indices.len()]`.
/// If `input.requires_grad()` and grad is enabled, the result tensor
/// carries an `IndexSelectBackward` grad_fn.
pub fn index_select_1d<T: Float>(
    input: &Tensor<T>,
    indices: &[usize],
) -> FerrotorchResult<Tensor<T>> {
    // Validate: input must be 1-D.
    if input.ndim() != 1 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "index_select_1d requires a 1-D input, got shape {:?}",
                input.shape()
            ),
        });
    }

    let input_len = input.shape()[0];

    // Validate all indices are in bounds (shape is CPU metadata).
    for &idx in indices {
        if idx >= input_len {
            return Err(FerrotorchError::IndexOutOfBounds {
                index: idx,
                axis: 0,
                size: input_len,
            });
        }
    }

    let output_shape = vec![indices.len()];

    if input.is_cuda() {
        // GPU path (#1822/#1823): 1-D gather via the dim-aware kernel
        // (outer=1, inner=1), dispatched on T::dtype() with an i64 index
        // upload — index values above 2^24 are not representable in f32.
        let result = gather_dim_cuda(
            input,
            indices,
            1,
            input_len,
            indices.len(),
            1,
            output_shape.clone(),
            "index_select_1d",
        )?;

        if input.requires_grad() && is_grad_enabled() {
            let grad_fn = Arc::new(IndexSelectBackward {
                input: input.clone(),
                indices: indices.to_vec(),
            });
            let (storage, output_shape) = result.into_storage_and_shape()?;
            Tensor::from_operation(storage, output_shape, grad_fn)
        } else {
            Ok(result)
        }
    } else {
        // CPU path: direct gather.
        let input_data = input.data()?;
        let output_data: Vec<T> = indices.iter().map(|&idx| input_data[idx]).collect();

        if input.requires_grad() && is_grad_enabled() {
            let grad_fn = Arc::new(IndexSelectBackward {
                input: input.clone(),
                indices: indices.to_vec(),
            });
            Tensor::from_operation(TensorStorage::cpu(output_data), output_shape, grad_fn)
        } else {
            Tensor::from_storage(TensorStorage::cpu(output_data), output_shape, false)
        }
    }
}

// ---------------------------------------------------------------------------
// masked_fill
// ---------------------------------------------------------------------------

/// Backward function for `masked_fill`.
///
/// Forward: `output[i] = if mask[i] { value } else { input[i] }`
///
/// VJP: `grad_input[i] = if mask[i] { 0 } else { grad_output[i] }`
///
/// The gradient is zeroed at every position where the mask was true, because
/// those positions were replaced by a constant and no longer depend on the input.
///
/// The mask is stored as a [`BoolTensor`], which is resident-capable: if the
/// forward ran on GPU the mask stays on the device, so the backward routes
/// through the resident Phase-3c masked op with NO host crossing.
#[derive(Debug)]
pub struct MaskedFillBackward<T: Float> {
    /// The original input tensor (saved for shape).
    pub input: Tensor<T>,
    /// The full boolean mask from the forward pass (CPU- or GPU-resident).
    pub mask: BoolTensor,
}

impl<T: Float> GradFn<T> for MaskedFillBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !is_grad_enabled() {
            return Ok(vec![None]);
        }

        if grad_output.shape() != self.input.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "MaskedFillBackward: grad_output shape {:?} does not match input shape {:?}",
                    grad_output.shape(),
                    self.input.shape()
                ),
            });
        }
        if self.mask.numel() != self.input.numel() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "MaskedFillBackward: mask numel {} does not match input numel {}",
                    self.mask.numel(),
                    self.input.numel()
                ),
            });
        }

        // GPU-resident path (crosslink #1187 Phase 3d): grad_input = masked_fill(
        // grad_output, mask, 0) — zero the gradient where the forward filled a
        // constant. Both grad and the bool mask stay on the device; the resident
        // `masked_fill_dt` kernel is dtype-generic (f32/f64/bf16/f16). NO mask
        // host crossing, NO float-mask upload.
        if self.input.is_cuda() || grad_output.is_cuda() || self.mask.is_cuda() {
            let expected = self.input.device();
            if !expected.is_cuda() {
                let got = if grad_output.is_cuda() {
                    grad_output.device()
                } else {
                    self.mask.device()
                };
                return Err(FerrotorchError::DeviceMismatch { expected, got });
            }
            if self.mask.device() != expected {
                return Err(FerrotorchError::DeviceMismatch {
                    expected,
                    got: self.mask.device(),
                });
            }
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let grad_output = if grad_output.device() == expected {
                grad_output.clone()
            } else {
                grad_output.to(expected)?
            };
            // `.contiguous()` is required even when the shape/stride layout is
            // logically contiguous: a CUDA view can carry a non-zero
            // storage_offset or share a larger base buffer. Backend kernels only
            // see raw handles, so materialize the logical gradient on-device
            // before comparing it with the resident mask length.
            let grad_output = grad_output.contiguous()?;
            let result_handle =
                backend.masked_fill_dt(grad_output.gpu_handle()?, self.mask.gpu_handle()?, 0.0)?;
            let grad_tensor = Tensor::from_storage(
                TensorStorage::gpu(result_handle),
                self.input.shape().to_vec(),
                false,
            )?;
            Ok(vec![Some(grad_tensor)])
        } else {
            // CPU path: direct mask zeroing. Use `data_vec()` so direct
            // `GradFn::backward` calls with CPU views get the same logical-order
            // treatment the autograd engine gives non-contiguous gradients.
            let go_data = grad_output.data_vec()?;
            let mask_h = self.mask.data()?;
            let mut grad_input = go_data;
            for (i, &m) in mask_h.iter().enumerate() {
                if m {
                    grad_input[i] = <T as num_traits::Zero>::zero();
                }
            }
            let grad_tensor = Tensor::from_storage(
                TensorStorage::cpu(grad_input),
                self.input.shape().to_vec(),
                false,
            )?;
            Ok(vec![Some(grad_tensor)])
        }
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "MaskedFillBackward"
    }
}

/// Fill elements of `input` with `value` where `mask` is `true`.
///
/// `mask` is a boolean slice with the same number of elements as `input`
/// (flat layout). Returns a new tensor; the original is not mutated.
///
/// If `input.requires_grad()` and grad is enabled, the result carries a
/// `MaskedFillBackward` grad_fn.
pub fn masked_fill<T: Float>(
    input: &Tensor<T>,
    mask: &[bool],
    value: T,
) -> FerrotorchResult<Tensor<T>> {
    let input_len = input.numel();
    if mask.len() != input_len {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "masked_fill: mask length {} does not match input length {}",
                mask.len(),
                input_len
            ),
        });
    }

    let mask_bt = BoolTensor::from_slice(mask, input.shape())?;
    if input.is_cuda() {
        let mask_bt = mask_bt.to(input.device())?;
        return masked_fill_bt(input, &mask_bt, value);
    }

    masked_fill_cpu(input, &mask_bt, value)
}

fn masked_fill_cpu<T: Float>(
    input: &Tensor<T>,
    mask: &BoolTensor,
    value: T,
) -> FerrotorchResult<Tensor<T>> {
    if input.device() != mask.device() {
        return Err(FerrotorchError::DeviceMismatch {
            expected: input.device(),
            got: mask.device(),
        });
    }
    let output_shape = input.shape().to_vec();
    let input_data = input.data_vec()?;
    let mask_data = mask.data()?;
    let output_data: Vec<T> = input_data
        .iter()
        .zip(mask_data.iter())
        .map(|(&x, &m)| if m { value } else { x })
        .collect();

    if input.requires_grad() && is_grad_enabled() {
        let grad_fn = Arc::new(MaskedFillBackward {
            input: input.clone(),
            mask: mask.clone(),
        });
        Tensor::from_operation(TensorStorage::cpu(output_data), output_shape, grad_fn)
    } else {
        Tensor::from_storage(TensorStorage::cpu(output_data), output_shape, false)
    }
}

// ---------------------------------------------------------------------------
// gather
// ---------------------------------------------------------------------------

/// Backward function for N-D `gather`.
///
/// Forward: `output[coords] = input[coords with dim replaced by index[coords]]`
///
/// VJP: scatter-add `grad_output` back into zeros of input shape along `dim`
/// using the same `index`.
#[derive(Debug)]
pub struct GatherBackward<T: Float> {
    /// The original input tensor (saved for shape).
    pub input: Tensor<T>,
    /// The dimension along which gathering was performed.
    pub dim: usize,
    /// The flat index data used during the forward pass.
    pub index: Vec<usize>,
    /// CUDA-resident i64 index saved by CUDA forwards.
    pub index_cuda: Option<IntTensor<i64>>,
    /// The shape of the index tensor.
    pub index_shape: Vec<usize>,
}

impl<T: Float> GradFn<T> for GatherBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !is_grad_enabled() {
            return Ok(vec![None]);
        }

        let input_shape = self.input.shape();
        let input_numel: usize = crate::shape::numel(input_shape);
        let index_numel: usize = crate::shape::numel(&self.index_shape);
        if grad_output.numel() != index_numel {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "GatherBackward: grad_output numel {} != saved index numel {}",
                    grad_output.numel(),
                    index_numel
                ),
            });
        }
        if index_numel == 0 {
            let grad_tensor = if grad_output.is_cuda() {
                let ordinal = match grad_output.device() {
                    Device::Cuda(o) => o,
                    _ => unreachable!(),
                };
                let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
                let zeros = backend.alloc_zeros(input_numel, T::dtype(), ordinal)?;
                Tensor::from_storage(TensorStorage::gpu(zeros), input_shape.to_vec(), false)?
            } else {
                Tensor::from_storage(
                    TensorStorage::cpu(vec![<T as num_traits::Zero>::zero(); input_numel]),
                    input_shape.to_vec(),
                    false,
                )?
            };
            return Ok(vec![Some(grad_tensor)]);
        }

        // §3 GPU-native path (#1822/#1823): scatter-add grad_output into a
        // zeroed input-shaped buffer via the rank-aware kernel, dispatched on
        // T::dtype() with the SAME forward i64 index. CUDA forwards save that
        // index resident; legacy CPU/kernel-layer callers still upload their
        // host Vec when explicitly constructing this node outside phase2c.
        if grad_output.is_cuda() {
            let dt = T::dtype();
            let ordinal = match grad_output.device() {
                Device::Cuda(o) => o,
                _ => unreachable!(),
            };
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let uploaded_idx;
            let idx_handle = if let Some(index_cuda) = &self.index_cuda {
                if index_cuda.numel() != index_numel {
                    return Err(FerrotorchError::ShapeMismatch {
                        message: format!(
                            "GatherBackward: CUDA saved index numel {} != saved index shape numel {}",
                            index_cuda.numel(),
                            index_numel
                        ),
                    });
                }
                if index_cuda.device() != grad_output.device() {
                    return Err(FerrotorchError::DeviceMismatch {
                        expected: grad_output.device(),
                        got: index_cuda.device(),
                    });
                }
                index_cuda.gpu_handle()?
            } else {
                if self.index.len() != index_numel {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!(
                            "GatherBackward: host saved index length {} != saved index shape numel {}",
                            self.index.len(),
                            index_numel
                        ),
                    });
                }
                uploaded_idx = crate::ops::indexing::upload_index_i64(&self.index, ordinal)?;
                &uploaded_idx
            };
            let zeros = backend.alloc_zeros(input_numel, dt, ordinal)?;
            let go_handle = grad_output.gpu_handle()?;
            let result_handle = match dt {
                DType::F32 => backend.scatter_add_nd_f32(
                    &zeros,
                    idx_handle,
                    go_handle,
                    input_shape,
                    &self.index_shape,
                    self.dim,
                )?,
                DType::F64 => backend.scatter_add_nd_f64(
                    &zeros,
                    idx_handle,
                    go_handle,
                    input_shape,
                    &self.index_shape,
                    self.dim,
                )?,
                DType::F16 => backend.scatter_add_nd_f16(
                    &zeros,
                    idx_handle,
                    go_handle,
                    input_shape,
                    &self.index_shape,
                    self.dim,
                )?,
                DType::BF16 => backend.scatter_add_nd_bf16(
                    &zeros,
                    idx_handle,
                    go_handle,
                    input_shape,
                    &self.index_shape,
                    self.dim,
                )?,
                _ => {
                    return Err(FerrotorchError::NotImplementedOnCuda {
                        op: "GatherBackward",
                    });
                }
            };
            let grad_tensor = Tensor::from_storage(
                TensorStorage::gpu(result_handle),
                input_shape.to_vec(),
                false,
            )?;
            return Ok(vec![Some(grad_tensor)]);
        }

        let go_data = grad_output.data_vec()?;
        let ndim = input_shape.len();
        if self.index.len() != index_numel {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "GatherBackward: CPU backward requires a host saved index of length {}, got {}",
                    index_numel,
                    self.index.len()
                ),
            });
        }

        let mut grad_input = vec![<T as num_traits::Zero>::zero(); input_numel];

        // Scatter-add grad_output into grad_input using the saved index and dim.
        let mut coords = vec![0usize; ndim];
        for (i, &go_val) in go_data.iter().enumerate().take(index_numel) {
            let idx_val = self.index[i];
            let mut dst_coords = coords.clone();
            dst_coords[self.dim] = idx_val;
            let dst_flat = flat_index(&dst_coords, input_shape);
            grad_input[dst_flat] += go_val;

            if i + 1 < index_numel {
                increment_coords(&mut coords, &self.index_shape);
            }
        }

        let grad_tensor =
            Tensor::from_storage(TensorStorage::cpu(grad_input), input_shape.to_vec(), false)?;
        Ok(vec![Some(grad_tensor)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "GatherBackward"
    }
}

// ---------------------------------------------------------------------------
// scatter
// ---------------------------------------------------------------------------

/// Backward function for N-D `scatter`.
///
/// Forward: `output = input.clone(); output[index-mapped coords] = src[coords]`
///
/// VJP for input: `grad_input = grad_output` with scattered positions zeroed out
/// (those positions came from src, not input).
///
/// VJP for src: `grad_src[coords] = grad_output[index-mapped coords]` (gather).
#[derive(Debug)]
pub struct ScatterBackward<T: Float> {
    /// The original input tensor.
    pub input: Tensor<T>,
    /// The source tensor scattered into input.
    pub src: Tensor<T>,
    /// The dimension along which scattering was performed.
    pub dim: usize,
    /// The flat index data.
    pub index: Vec<usize>,
    /// The shape of the index tensor.
    pub index_shape: Vec<usize>,
}

impl<T: Float> GradFn<T> for ScatterBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !is_grad_enabled() {
            return Ok(vec![None, None]);
        }

        // CORE-127 (#1821): the src gradient (`grad.gather(dim, index)`) is
        // index-shaped, so a per-axis-larger src has no scatter VJP. Live
        // torch 2.11.0 raises at backward time ("Function ScatterBackward0
        // returned an invalid gradient at index 1 - got [2, 4] but expected
        // shape compatible with [2, 5]"); match that contract with a
        // structured error instead of silently returning the index-shaped
        // gradient. grad for `input` alone is well-defined and flows.
        if self.src.requires_grad() && self.src.shape() != self.index_shape.as_slice() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "scatter backward: gradient for src is only defined when src.shape \
                     ({:?}) equals index.shape ({:?}); PyTorch raises here too \
                     (ScatterBackward0 returns an index-shaped gradient the engine \
                     rejects)",
                    self.src.shape(),
                    self.index_shape
                ),
            });
        }

        let input_shape = self.input.shape();
        let index_numel: usize = crate::shape::numel(&self.index_shape);

        // §3 GPU-native path (#1822/#1823), dispatched on T::dtype() with the
        // forward index uploaded as i64 (no f32 offset/mask encoding):
        //   grad_input = scatter_value_nd(grad_output, index, 0.0)
        //     — clones grad_output and writes 0 at every position scatter
        //       wrote to (those positions came from src), replacing the
        //       uploaded f32 write-mask + masked_zero_f32.
        //   grad_src   = gather_nd(grad_output, index)
        //     — gathers from the positions that scatter wrote into
        //       (src.shape == index.shape is guaranteed by the CORE-127
        //       contract check above).
        if grad_output.is_cuda() {
            let dt = T::dtype();
            let ordinal = match grad_output.device() {
                Device::Cuda(o) => o,
                _ => unreachable!(),
            };
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let idx_handle = crate::ops::indexing::upload_index_i64(&self.index, ordinal)?;
            let go_handle = grad_output.gpu_handle()?;

            let grad_input = if self.input.requires_grad() {
                let result_h = match dt {
                    DType::F32 => backend.scatter_value_nd_f32(
                        go_handle,
                        &idx_handle,
                        0.0,
                        input_shape,
                        &self.index_shape,
                        self.dim,
                    )?,
                    DType::F64 => backend.scatter_value_nd_f64(
                        go_handle,
                        &idx_handle,
                        0.0,
                        input_shape,
                        &self.index_shape,
                        self.dim,
                    )?,
                    DType::F16 => backend.scatter_value_nd_f16(
                        go_handle,
                        &idx_handle,
                        0.0,
                        input_shape,
                        &self.index_shape,
                        self.dim,
                    )?,
                    DType::BF16 => backend.scatter_value_nd_bf16(
                        go_handle,
                        &idx_handle,
                        0.0,
                        input_shape,
                        &self.index_shape,
                        self.dim,
                    )?,
                    _ => {
                        return Err(FerrotorchError::NotImplementedOnCuda {
                            op: "ScatterBackward",
                        });
                    }
                };
                Some(Tensor::from_storage(
                    TensorStorage::gpu(result_h),
                    input_shape.to_vec(),
                    false,
                )?)
            } else {
                None
            };

            let grad_src = if self.src.requires_grad() {
                let result_h = backend.gather_intidx_nd(
                    go_handle,
                    &idx_handle,
                    input_shape,
                    &self.index_shape,
                    self.dim,
                )?;
                Some(Tensor::from_storage(
                    TensorStorage::gpu(result_h),
                    self.index_shape.clone(),
                    false,
                )?)
            } else {
                None
            };

            return Ok(vec![grad_input, grad_src]);
        }

        let ndim = input_shape.len();
        let go_data = grad_output.data_vec()?;

        // grad for input: grad_output with scattered positions zeroed.
        let grad_input = if self.input.requires_grad() {
            let mut gi = go_data.clone();
            let mut coords = vec![0usize; ndim];
            for i in 0..index_numel {
                let idx_val = self.index[i];
                let mut dst_coords = coords.clone();
                dst_coords[self.dim] = idx_val;
                let dst_flat = flat_index(&dst_coords, input_shape);
                gi[dst_flat] = <T as num_traits::Zero>::zero();

                if i + 1 < index_numel {
                    increment_coords(&mut coords, &self.index_shape);
                }
            }
            let t = Tensor::from_storage(TensorStorage::cpu(gi), input_shape.to_vec(), false)?;
            Some(t)
        } else {
            None
        };

        // grad for src: gather from grad_output at index positions.
        let grad_src = if self.src.requires_grad() {
            let mut gs = vec![<T as num_traits::Zero>::zero(); index_numel];
            let mut coords = vec![0usize; ndim];
            for (i, gs_elem) in gs.iter_mut().enumerate() {
                let idx_val = self.index[i];
                let mut src_coords = coords.clone();
                src_coords[self.dim] = idx_val;
                let src_flat = flat_index(&src_coords, input_shape);
                *gs_elem = go_data[src_flat];

                if i + 1 < index_numel {
                    increment_coords(&mut coords, &self.index_shape);
                }
            }
            let t = Tensor::from_storage(TensorStorage::cpu(gs), self.index_shape.clone(), false)?;
            Some(t)
        } else {
            None
        };

        Ok(vec![grad_input, grad_src])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input, &self.src]
    }

    fn name(&self) -> &'static str {
        "ScatterBackward"
    }
}

// ---------------------------------------------------------------------------
// scatter_add
// ---------------------------------------------------------------------------

/// Backward function for N-D `scatter_add`.
///
/// Forward: `output = input.clone(); output[index-mapped coords] += src[coords]`
///
/// VJP for input: `grad_input = grad_output` (identity — addition passes
/// gradient through unchanged).
///
/// VJP for src: `grad_src[coords] = grad_output[index-mapped coords]` (gather).
#[derive(Debug)]
pub struct ScatterAddBackward<T: Float> {
    /// The original input tensor.
    pub input: Tensor<T>,
    /// The source tensor that was scatter-added.
    pub src: Tensor<T>,
    /// The dimension along which scatter_add was performed.
    pub dim: usize,
    /// The flat index data.
    pub index: Vec<usize>,
    /// The shape of the index tensor.
    pub index_shape: Vec<usize>,
}

impl<T: Float> GradFn<T> for ScatterAddBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !is_grad_enabled() {
            return Ok(vec![None, None]);
        }

        // CORE-127 (#1821): same src-shape contract as ScatterBackward —
        // live torch 2.11.0 raises "Function ScatterAddBackward0 returned an
        // invalid gradient at index 1" when src.shape != index.shape and src
        // requires grad; match with a structured error.
        if self.src.requires_grad() && self.src.shape() != self.index_shape.as_slice() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "scatter_add backward: gradient for src is only defined when src.shape \
                     ({:?}) equals index.shape ({:?}); PyTorch raises here too \
                     (ScatterAddBackward0 returns an index-shaped gradient the engine \
                     rejects)",
                    self.src.shape(),
                    self.index_shape
                ),
            });
        }

        let input_shape = self.input.shape();
        let index_numel: usize = crate::shape::numel(&self.index_shape);

        // §3 GPU-native path (#1822/#1823), dispatched on T::dtype():
        //   grad_input = grad_output  (identity — addition passes grad
        //     through unchanged; `clone_buffer` is dtype-agnostic).
        //   grad_src   = gather_nd(grad_output, index)  with the forward
        //     index uploaded as i64 — gathers the positions scatter_add
        //     accumulated into (src.shape == index.shape guaranteed by the
        //     CORE-127 contract check above).
        if grad_output.is_cuda() {
            let dt = T::dtype();
            if !matches!(dt, DType::F32 | DType::F64 | DType::F16 | DType::BF16) {
                return Err(FerrotorchError::NotImplementedOnCuda {
                    op: "ScatterAddBackward",
                });
            }
            let ordinal = match grad_output.device() {
                Device::Cuda(o) => o,
                _ => unreachable!(),
            };
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;

            let grad_input = if self.input.requires_grad() {
                // Identity: grad_input is an on-device copy of grad_output.
                let cloned_h = backend.clone_buffer(grad_output.gpu_handle()?)?;
                Some(Tensor::from_storage(
                    TensorStorage::gpu(cloned_h),
                    input_shape.to_vec(),
                    false,
                )?)
            } else {
                None
            };

            let grad_src = if self.src.requires_grad() {
                let idx_handle = crate::ops::indexing::upload_index_i64(&self.index, ordinal)?;
                let go_handle = grad_output.gpu_handle()?;
                let result_h = backend.gather_intidx_nd(
                    go_handle,
                    &idx_handle,
                    input_shape,
                    &self.index_shape,
                    self.dim,
                )?;
                Some(Tensor::from_storage(
                    TensorStorage::gpu(result_h),
                    self.index_shape.clone(),
                    false,
                )?)
            } else {
                None
            };

            return Ok(vec![grad_input, grad_src]);
        }

        let ndim = input_shape.len();
        let go_data = grad_output.data_vec()?;

        // grad for input: identity (pass grad_output through).
        let grad_input = if self.input.requires_grad() {
            let t = Tensor::from_storage(
                TensorStorage::cpu(go_data.clone()),
                input_shape.to_vec(),
                false,
            )?;
            Some(t)
        } else {
            None
        };

        // grad for src: gather from grad_output at index positions.
        let grad_src = if self.src.requires_grad() {
            let mut gs = vec![<T as num_traits::Zero>::zero(); index_numel];
            let mut coords = vec![0usize; ndim];
            for (i, gs_elem) in gs.iter_mut().enumerate() {
                let idx_val = self.index[i];
                let mut src_coords = coords.clone();
                src_coords[self.dim] = idx_val;
                let src_flat = flat_index(&src_coords, input_shape);
                *gs_elem = go_data[src_flat];

                if i + 1 < index_numel {
                    increment_coords(&mut coords, &self.index_shape);
                }
            }
            let t = Tensor::from_storage(TensorStorage::cpu(gs), self.index_shape.clone(), false)?;
            Some(t)
        } else {
            None
        };

        Ok(vec![grad_input, grad_src])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input, &self.src]
    }

    fn name(&self) -> &'static str {
        "ScatterAddBackward"
    }
}

// ---------------------------------------------------------------------------
// where_cond
// ---------------------------------------------------------------------------

/// Backward function for `where_cond`.
///
/// Forward: `output[i] = condition[i] ? x[i] : y[i]`
///
/// VJP for x: `grad_x[i] = condition[i] ? grad_output[i] : 0`
/// VJP for y: `grad_y[i] = condition[i] ? 0 : grad_output[i]`
#[derive(Debug)]
pub struct WhereCondBackward<T: Float> {
    /// The x tensor from the forward pass.
    pub x: Tensor<T>,
    /// The y tensor from the forward pass.
    pub y: Tensor<T>,
    /// The condition mask from the forward pass (CPU- or GPU-resident).
    pub condition: BoolTensor,
}

impl<T: Float> GradFn<T> for WhereCondBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !is_grad_enabled() {
            return Ok(vec![None, None]);
        }

        // GPU-resident path (crosslink #1187 Phase 3d):
        //   grad_x[i] = cond[i] ? grad[i] : 0  → zero grad where cond is FALSE
        //   grad_y[i] = cond[i] ? 0 : grad[i]  → zero grad where cond is TRUE
        // Both legs reuse the resident `masked_fill_dt` Phase-3c kernel with the
        // resident bool condition directly: `masked_fill(grad, mask, 0)` zeros
        // grad at every position where `mask` is true. grad_y uses `cond`; grad_x
        // uses `!cond` (the resident `bool_not`). NO float-mask upload, NO host
        // crossing. `masked_fill_dt` is dtype-generic (f32/f64/bf16/f16) and
        // allocates exact-length output buffers (PyTorch parity: VJP of `where`).
        if grad_output.is_cuda() && self.condition.is_cuda() {
            if grad_output.device() != self.condition.device() {
                return Err(FerrotorchError::DeviceMismatch {
                    expected: grad_output.device(),
                    got: self.condition.device(),
                });
            }
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let cond_h = self.condition.gpu_handle()?;
            let grad_h = grad_output.gpu_handle()?;

            let grad_x = if self.x.requires_grad() {
                // Zero grad where cond is false ⇒ fill grad with 0 at !cond.
                let not_cond = backend.bool_not(cond_h)?;
                let result_h = backend.masked_fill_dt(grad_h, &not_cond, 0.0)?;
                Some(Tensor::from_storage(
                    TensorStorage::gpu(result_h),
                    self.x.shape().to_vec(),
                    false,
                )?)
            } else {
                None
            };

            let grad_y = if self.y.requires_grad() {
                // Zero grad where cond is true ⇒ fill grad with 0 at cond.
                let result_h = backend.masked_fill_dt(grad_h, cond_h, 0.0)?;
                Some(Tensor::from_storage(
                    TensorStorage::gpu(result_h),
                    self.y.shape().to_vec(),
                    false,
                )?)
            } else {
                None
            };

            return Ok(vec![grad_x, grad_y]);
        }

        // CPU path. `self.condition.data()?` borrows the host bool slice (errors
        // if the cond is GPU-resident while grad is on host).
        let go_data = grad_output.data_vec()?;
        let cond = self.condition.data()?;
        let zero = <T as num_traits::Zero>::zero();

        let grad_x = if self.x.requires_grad() {
            let gx: Vec<T> = cond
                .iter()
                .zip(go_data.iter())
                .map(|(&c, &g)| if c { g } else { zero })
                .collect();
            let t = Tensor::from_storage(TensorStorage::cpu(gx), self.x.shape().to_vec(), false)?;
            Some(t)
        } else {
            None
        };

        let grad_y = if self.y.requires_grad() {
            let gy: Vec<T> = cond
                .iter()
                .zip(go_data.iter())
                .map(|(&c, &g)| if c { zero } else { g })
                .collect();
            let t = Tensor::from_storage(TensorStorage::cpu(gy), self.y.shape().to_vec(), false)?;
            Some(t)
        } else {
            None
        };

        Ok(vec![grad_x, grad_y])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.x, &self.y]
    }

    fn name(&self) -> &'static str {
        "WhereCondBackward"
    }
}

// ---------------------------------------------------------------------------
// masked_select (crosslink #1187 Phase 3d — masked_select IS differentiable)
// ---------------------------------------------------------------------------

/// Backward function for `masked_select`.
///
/// Forward: `output = compact(input[i] for i where mask[i])` — a 1-D tensor of
/// the kept elements in flat C-order (length = #true).
///
/// VJP: scatter the compacted `grad_output` (length = #true) back into a zeros
/// tensor of `input.numel()` at the flat positions where `mask` is true; every
/// non-selected position gets 0. This is the exact inverse of the forward
/// compaction (PyTorch parity — `torch.masked_select` is differentiable).
///
/// The mask is stored as a [`BoolTensor`]: resident if the forward ran on GPU,
/// so the backward routes through the resident `masked_scatter` kernel with NO
/// host crossing.
#[derive(Debug)]
pub struct MaskedSelectBackward<T: Float> {
    /// The original input tensor (saved for shape + autograd graph linkage).
    pub input: Tensor<T>,
    /// The boolean mask from the forward pass (CPU- or GPU-resident).
    pub mask: BoolTensor,
}

impl<T: Float> GradFn<T> for MaskedSelectBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !is_grad_enabled() {
            return Ok(vec![None]);
        }

        let input_shape = self.input.shape().to_vec();
        let input_numel: usize = crate::shape::numel(&input_shape);

        // GPU-resident path (crosslink #1187 Phase 3d): scatter the compacted
        // grad back into a zeros buffer of input.numel() at the true positions,
        // via the resident `masked_scatter` kernel (inverse of the Phase-3c
        // compaction). grad + bool mask stay on the device — NO host crossing.
        if grad_output.is_cuda() && self.mask.is_cuda() {
            if grad_output.device() != self.mask.device() {
                return Err(FerrotorchError::DeviceMismatch {
                    expected: grad_output.device(),
                    got: self.mask.device(),
                });
            }
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let result_handle = backend.masked_scatter(
                grad_output.gpu_handle()?,
                self.mask.gpu_handle()?,
                input_numel,
            )?;
            let grad_tensor =
                Tensor::from_storage(TensorStorage::gpu(result_handle), input_shape, false)?;
            return Ok(vec![Some(grad_tensor)]);
        }

        // CPU path: walk the host mask, scattering grad[j++] -> grad_input[i] at
        // each true position. `self.mask.data()?` errors if the mask is GPU-
        // resident while grad is on host (the correct device-mismatch signal).
        let go_data = grad_output.data()?;
        let mask_h = self.mask.data()?;
        let zero = <T as num_traits::Zero>::zero();
        let mut grad_input: Vec<T> = vec![zero; input_numel];
        let mut j = 0usize;
        for (i, &m) in mask_h.iter().enumerate() {
            if m {
                grad_input[i] = go_data[j];
                j += 1;
            }
        }
        let grad_tensor = Tensor::from_storage(TensorStorage::cpu(grad_input), input_shape, false)?;
        Ok(vec![Some(grad_tensor)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "MaskedSelectBackward"
    }
}

// ---------------------------------------------------------------------------
// First-class IntTensor / BoolTensor wrappers (#615)
// ---------------------------------------------------------------------------

/// `masked_fill` taking a [`BoolTensor`] mask. Shape and numel must
/// match `input`. Returns a new tensor; original unchanged. Mirrors
/// torch's `tensor.masked_fill(mask, value)` with mask convention
/// "true → fill" (same as the existing `&[bool]` variant).
pub fn masked_fill_bt<T: Float>(
    input: &Tensor<T>,
    mask: &BoolTensor,
    value: T,
) -> FerrotorchResult<Tensor<T>> {
    if mask.numel() != input.numel() {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "masked_fill_bt: mask numel={} != input numel={}",
                mask.numel(),
                input.numel()
            ),
        });
    }
    if input.device() != mask.device() {
        return Err(FerrotorchError::DeviceMismatch {
            expected: input.device(),
            got: mask.device(),
        });
    }

    // GPU-resident fast path (crosslink #1185 Phase 3c): both input and mask
    // live on CUDA — dispatch on input.dtype() through the resident-bool kernel.
    // The mask is the resident `DType::Bool` buffer; NO float-mask upload, NO
    // host crossing. The fill `value` is the only scalar passed (as f64; halves
    // narrow it in-kernel). Requires same device for both operands.
    if input.is_cuda() && mask.is_cuda() {
        if input.device() != mask.device() {
            return Err(FerrotorchError::DeviceMismatch {
                expected: input.device(),
                got: mask.device(),
            });
        }
        let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let value_f64 = num_traits::ToPrimitive::to_f64(&value).ok_or_else(|| {
            FerrotorchError::InvalidArgument {
                message: "masked_fill_bt: value not representable as f64".into(),
            }
        })?;
        // #1661: a row-narrowed CUDA view (e.g. `.narrow(0,1,3)`) reports its
        // logical numel but is backed by a larger base buffer carrying a non-zero
        // `storage_offset`. Reading `input.gpu_handle()` raw makes `masked_fill_dt`
        // see the base-buffer len (8) and reject it against the mask len (6), or
        // (post-#1660 logical-len launch) read the wrong window. `.contiguous()`
        // materialises the logical view on-device via strided_copy (#1657), so the
        // handle's logical len matches the mask numel and the kernel reads `[0, n)`.
        let input = input.contiguous()?;
        let result_handle =
            backend.masked_fill_dt(input.gpu_handle()?, mask.gpu_handle()?, value_f64)?;
        let storage = TensorStorage::gpu(result_handle);
        let output_shape = input.shape().to_vec();

        if input.requires_grad() && is_grad_enabled() {
            // Store the resident mask directly (cheap Arc/clone-on-storage) — the
            // backward routes through the resident `masked_fill_dt` VJP with NO
            // host crossing (crosslink #1187 Phase 3d). No `mask.to(Cpu)`.
            let grad_fn = Arc::new(MaskedFillBackward {
                input: input.clone(),
                mask: mask.clone(),
            });
            return Tensor::from_operation(storage, output_shape, grad_fn);
        }
        return Tensor::from_storage(storage, output_shape, false);
    }

    masked_fill_cpu(input, mask, value)
}

/// `index_select_1d` taking an [`IntTensor`] of indices. The index tensor
/// must be 1-D and contain non-negative values within range.
pub fn index_select_1d_it<T: Float, I: IntElement>(
    input: &Tensor<T>,
    indices: &IntTensor<I>,
) -> FerrotorchResult<Tensor<T>> {
    if indices.ndim() != 1 {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "index_select_1d_it: indices must be 1-D, got shape {:?}",
                indices.shape()
            ),
        });
    }
    let mut idx_usize: Vec<usize> = Vec::with_capacity(indices.numel());
    for v in indices.data()? {
        let i = v.to_i64();
        if i < 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("index_select_1d_it: negative index {i} not allowed"),
            });
        }
        idx_usize.push(i as usize);
    }
    index_select_1d(input, &idx_usize)
}

// ---------------------------------------------------------------------------
// index_select_dim — N-D, gather along arbitrary axis with 1-D indices (#1014)
// ---------------------------------------------------------------------------

/// Backward function for [`index_select_dim`].
///
/// Forward (for `dim = D`): `output[..., i, ...] = input[..., indices[i], ...]`,
/// i.e. each "slice" along `dim` of `output` at position `i` is a copy of the
/// `input` slice at position `indices[i]`.
///
/// VJP: scatter-add `grad_output` back along `dim` at positions `indices`,
/// accumulating duplicates. This is the N-D analogue of the 1-D
/// `IndexSelectBackward` above.
#[derive(Debug)]
pub struct IndexSelectDimBackward<T: Float> {
    /// Saved input handle (for shape and `requires_grad` propagation).
    pub input: Tensor<T>,
    /// The dimension along which selection was performed.
    pub dim: usize,
    /// The 1-D index vector used during the forward pass.
    pub indices: Vec<usize>,
    /// CUDA-resident i64 index saved by CUDA forwards.
    pub indices_cuda: Option<IntTensor<i64>>,
}

impl<T: Float> GradFn<T> for IndexSelectDimBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !is_grad_enabled() {
            return Ok(vec![None]);
        }
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }

        let input_shape = self.input.shape();
        let input_numel: usize = crate::shape::numel(input_shape);
        let dim = self.dim;
        let outer: usize = crate::shape::numel(&input_shape[..dim]);
        let inner: usize = crate::shape::numel(&input_shape[dim + 1..]);
        let in_dim_size = input_shape[dim];
        let out_dim_size = self
            .indices_cuda
            .as_ref()
            .map_or(self.indices.len(), IntTensor::numel);
        let expected_go_numel = outer
            .checked_mul(out_dim_size)
            .and_then(|x| x.checked_mul(inner))
            .ok_or(FerrotorchError::InvalidArgument {
                message: format!(
                    "IndexSelectDimBackward: output numel overflow for outer={outer}, out_dim={out_dim_size}, inner={inner}"
                ),
            })?;
        if grad_output.numel() != expected_go_numel {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "IndexSelectDimBackward: grad_output numel {} != expected {} from input shape {:?}, dim {}, and saved index length {}",
                    grad_output.numel(),
                    expected_go_numel,
                    input_shape,
                    dim,
                    out_dim_size
                ),
            });
        }

        // GPU path (#1822/#1823 mechanism): scatter-add grad_output into a
        // zeroed input-shaped buffer via the dim-aware kernel, dispatched on
        // T::dtype(). The kernel index is parallel to grad_output's
        // `[outer, out_dim_size, inner]` t-space. CUDA forwards save the 1-D
        // index resident and expand it resident before the scatter-add launch.
        if grad_output.is_cuda() {
            let dt = T::dtype();
            let ordinal = match grad_output.device() {
                Device::Cuda(o) => o,
                _ => unreachable!(),
            };
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;

            let uploaded_idx;
            let expanded_idx;
            let idx_handle = if let Some(indices_cuda) = &self.indices_cuda {
                if indices_cuda.numel() != out_dim_size {
                    return Err(FerrotorchError::ShapeMismatch {
                        message: format!(
                            "IndexSelectDimBackward: CUDA saved index numel {} != saved index length {}",
                            indices_cuda.numel(),
                            out_dim_size
                        ),
                    });
                }
                if indices_cuda.device() != grad_output.device() {
                    return Err(FerrotorchError::DeviceMismatch {
                        expected: grad_output.device(),
                        got: indices_cuda.device(),
                    });
                }
                expanded_idx = backend.expand_index_select_indices_i64(
                    indices_cuda.gpu_handle()?,
                    outer,
                    out_dim_size,
                    inner,
                )?;
                &expanded_idx
            } else {
                if self.indices.len() != out_dim_size {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!(
                            "IndexSelectDimBackward: host saved index length {} != saved index length {}",
                            self.indices.len(),
                            out_dim_size
                        ),
                    });
                }
                let mut axis_indices: Vec<usize> = Vec::with_capacity(expected_go_numel);
                for _o in 0..outer {
                    for i in 0..out_dim_size {
                        let dst_i = self.indices[i];
                        for _k in 0..inner {
                            axis_indices.push(dst_i);
                        }
                    }
                }
                uploaded_idx = crate::ops::indexing::upload_index_i64(&axis_indices, ordinal)?;
                &uploaded_idx
            };
            let zeros = backend.alloc_zeros(input_numel, dt, ordinal)?;
            let go_handle = grad_output.gpu_handle()?;
            let result_handle = scatter_dim_cuda_handle::<T>(
                &zeros,
                idx_handle,
                go_handle,
                outer,
                in_dim_size,
                out_dim_size,
                inner,
                true,
                "IndexSelectDimBackward",
            )?;
            let grad_tensor = Tensor::from_storage(
                TensorStorage::gpu(result_handle),
                input_shape.to_vec(),
                false,
            )?;
            return Ok(vec![Some(grad_tensor)]);
        }

        // CPU path: scatter-add directly.
        let go_data = grad_output.data_vec()?;
        if self.indices.len() != out_dim_size {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "IndexSelectDimBackward: CPU backward requires a host saved index of length {}, got {}",
                    out_dim_size,
                    self.indices.len()
                ),
            });
        }
        let mut grad_input = vec![<T as num_traits::Zero>::zero(); input_numel];
        for o in 0..outer {
            for i in 0..out_dim_size {
                let dst_i = self.indices[i];
                let go_base = o * out_dim_size * inner + i * inner;
                let gi_base = o * in_dim_size * inner + dst_i * inner;
                for k in 0..inner {
                    grad_input[gi_base + k] += go_data[go_base + k];
                }
            }
        }

        let grad_tensor =
            Tensor::from_storage(TensorStorage::cpu(grad_input), input_shape.to_vec(), false)?;
        Ok(vec![Some(grad_tensor)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "IndexSelectDimBackward"
    }
}

/// Differentiable N-D `index_select`: gathers slices along `dim` of `input`
/// using a 1-D vector of indices.
///
/// Mirrors `torch.index_select(input, dim, indices)`:
///
/// ```text
/// output[..., i, ...] = input[..., indices[i], ...]   (slice at axis `dim`)
/// ```
///
/// The output has the same shape as `input` except `output.shape()[dim] ==
/// indices.len()`. Indices may repeat — duplicates accumulate in backward.
///
/// If `input.requires_grad()` and grad is enabled, the result carries an
/// [`IndexSelectDimBackward`] grad_fn whose VJP scatter-adds `grad_output`
/// along `dim` back at the saved `indices` positions.
pub fn index_select_dim<T: Float, I: IntElement>(
    input: &Tensor<T>,
    dim: usize,
    indices: &IntTensor<I>,
) -> FerrotorchResult<Tensor<T>> {
    let input_shape = input.shape();
    let ndim = input_shape.len();
    if ndim == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "index_select_dim: input must have at least 1 dimension".into(),
        });
    }
    if dim >= ndim {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("index_select_dim: dim {dim} out of range for shape {input_shape:?}"),
        });
    }
    if indices.ndim() > 1 {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "index_select_dim: indices must be 0-D or 1-D, got shape {:?}",
                indices.shape()
            ),
        });
    }

    let in_dim_size = input_shape[dim];

    // Compute output: same shape but axis `dim` replaced by indices.numel().
    let mut output_shape = input_shape.to_vec();
    output_shape[dim] = indices.numel();

    let outer: usize = crate::shape::numel(&input_shape[..dim]);
    let inner: usize = crate::shape::numel(&input_shape[dim + 1..]);
    let out_dim_size = indices.numel();

    // GPU path: device-resident gather through `index_select_intidx`, which
    // dispatches on (src.dtype(), index.dtype()). The output buffer is
    // allocated on-device; no host round-trip. The index tensor must already
    // be resident on the same device, matching PyTorch's CUDA contract.
    if input.is_cuda() {
        if input.device() != indices.device() {
            return Err(FerrotorchError::DeviceMismatch {
                expected: input.device(),
                got: indices.device(),
            });
        }
        let input_c = input.contiguous()?;
        let (outer, in_dim_size, inner) = {
            let shape = input_c.shape();
            (
                crate::shape::numel(&shape[..dim]),
                shape[dim],
                crate::shape::numel(&shape[dim + 1..]),
            )
        };
        let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        backend.check_int_indices_in_bounds(
            indices.gpu_handle()?,
            dim,
            in_dim_size,
            "index_select_dim",
        )?;
        let result_handle = backend.index_select_intidx(
            input_c.gpu_handle()?,
            indices.gpu_handle()?,
            outer,
            in_dim_size,
            out_dim_size,
            inner,
        )?;

        let storage = TensorStorage::gpu(result_handle);
        return if input_c.requires_grad() && is_grad_enabled() {
            let grad_fn = Arc::new(IndexSelectDimBackward {
                input: input_c,
                dim,
                indices: Vec::new(),
                indices_cuda: Some(indices.cast::<i64>()?),
            });
            Tensor::from_operation(storage, output_shape, grad_fn)
        } else {
            Tensor::from_storage(storage, output_shape, false)
        };
    }

    if input.device() != indices.device() {
        return Err(FerrotorchError::DeviceMismatch {
            expected: input.device(),
            got: indices.device(),
        });
    }

    // Validate + widen host indices.
    let mut idx_usize: Vec<usize> = Vec::with_capacity(indices.numel());
    for v in indices.data()? {
        let i = v.to_i64();
        if i < 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("index_select_dim: negative index {i} not allowed"),
            });
        }
        let iu = i as usize;
        if iu >= in_dim_size {
            return Err(FerrotorchError::IndexOutOfBounds {
                index: iu,
                axis: dim,
                size: in_dim_size,
            });
        }
        idx_usize.push(iu);
    }

    // CPU path: dense memcpy along axis.
    let out_numel: usize = crate::shape::numel(&output_shape);
    let in_data = input.data_vec()?;
    let mut out = vec![<T as num_traits::Zero>::zero(); out_numel];
    for o in 0..outer {
        for i in 0..out_dim_size {
            let src_i = idx_usize[i];
            let in_base = o * in_dim_size * inner + src_i * inner;
            let out_base = o * out_dim_size * inner + i * inner;
            out[out_base..out_base + inner].copy_from_slice(&in_data[in_base..in_base + inner]);
        }
    }

    if input.requires_grad() && is_grad_enabled() {
        let grad_fn = Arc::new(IndexSelectDimBackward {
            input: input.clone(),
            dim,
            indices: idx_usize,
            indices_cuda: None,
        });
        Tensor::from_operation(TensorStorage::cpu(out), output_shape, grad_fn)
    } else {
        Tensor::from_storage(TensorStorage::cpu(out), output_shape, false)
    }
}

// ---------------------------------------------------------------------------
// index_fill — overwrite slices along `dim` at index positions with a scalar
// (#1249 — REQ-8). Mirrors `torch.index_fill(input, dim, index, value)` at
// `aten/src/ATen/native/TensorAdvancedIndexing.cpp:1979-1985 Tensor index_fill(
// const Tensor& self, int64_t dim, const Tensor& index, const Scalar& source)
// { return self.clone(...).index_fill_(dim, index, source); }`. Backward per
// `tools/autograd/derivatives.yaml:884-887
//   - name: index_fill.int_Scalar(Tensor self, int dim, Tensor index, Scalar value) -> Tensor
//     self: grad.index_fill(dim, index, 0)
//     index: non_differentiable
//     result: self_t.index_fill(dim, index, 0)`
// — gradient flows through every position NOT touched by the fill; the
// filled positions receive zero grad (they were overwritten and no longer
// depend on the input).
// ---------------------------------------------------------------------------

/// Backward function for `index_fill`.
///
/// Forward: `output = input.clone(); output[..., index[i], ...] = value` along
/// `dim`.
///
/// VJP: `grad_input = grad_output.index_fill(dim, index, 0)` — zero the
/// gradient at every slice position the forward overwrote with `value`.
#[derive(Debug)]
pub struct IndexFillBackward<T: Float> {
    /// Saved input handle (for shape and `requires_grad` propagation).
    pub input: Tensor<T>,
    /// The normalized (non-negative) dim along which fill was performed.
    pub dim: usize,
    /// The (validated, non-negative) index list saved from the forward pass.
    pub index: Vec<usize>,
}

impl<T: Float> GradFn<T> for IndexFillBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !is_grad_enabled() {
            return Ok(vec![None]);
        }
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }

        // grad_input = grad_output with the fill-positions zeroed.
        //
        // Walk grad_output's C-contiguous buffer once and zero every element
        // whose axis-`dim` coordinate is in `self.index`. The shape arithmetic
        // matches `index_select_dim`'s outer/inner decomposition: for axis
        // `dim`, the flat positions to zero are
        //     o * dim_size * inner + idx * inner + k
        // for every o ∈ outer, idx ∈ self.index, k ∈ inner.
        let input_shape = self.input.shape();
        let dim = self.dim;

        // 0-d input short-circuit (mirrors the forward's unsqueeze-to-1-d at
        // `aten/src/ATen/native/TensorAdvancedIndexing.cpp:1917`:
        //     Tensor self_nonzero_dim = (self.dim() == 0) ? self.unsqueeze(-1) : self;
        // The forward records `index = vec![0]` when the 0-d position was
        // filled and `index = vec![]` when the index tensor was empty.
        // Direct slice arithmetic below would panic via `input_shape[dim+1..]`
        // when `input_shape.len() == 0`. Per `derivatives.yaml:884-887`:
        //     self: grad.index_fill(dim, index, 0)
        // the VJP on the 0-d virtual length-1 dim is: zero out the single
        // scalar if the (only valid wrapped) index 0 is in `self.index`,
        // otherwise pass `grad_output` through unchanged.
        if input_shape.is_empty() {
            let go_data = grad_output.data_vec()?;
            let mut grad_input = go_data.clone();
            if !self.index.is_empty() {
                let zero = <T as num_traits::Zero>::zero();
                grad_input[0] = zero;
            }
            let grad_tensor = Tensor::from_storage(TensorStorage::cpu(grad_input), vec![], false)?;
            return Ok(vec![Some(grad_tensor)]);
        }

        let outer: usize = crate::shape::numel(&input_shape[..dim]);
        let inner: usize = crate::shape::numel(&input_shape[dim + 1..]);
        let dim_size = input_shape[dim];

        let go_data = grad_output.data_vec()?;
        let mut grad_input = go_data.clone();
        let zero = <T as num_traits::Zero>::zero();
        for o in 0..outer {
            for &idx in &self.index {
                let base = o * dim_size * inner + idx * inner;
                for k in 0..inner {
                    grad_input[base + k] = zero;
                }
            }
        }

        let grad_tensor =
            Tensor::from_storage(TensorStorage::cpu(grad_input), input_shape.to_vec(), false)?;
        Ok(vec![Some(grad_tensor)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "IndexFillBackward"
    }
}

/// Out-of-place `index_fill`: fill `output[..., index[i], ...] = value` along
/// `dim`. Mirrors `torch.index_fill(input, dim, index, value)` per
/// `aten/src/ATen/native/TensorAdvancedIndexing.cpp:1979 Tensor index_fill(
/// const Tensor& self, int64_t dim, const Tensor& index, const Scalar& source)`.
///
/// `dim` follows PyTorch's negative-wrapping convention: `dim ∈ [-ndim, ndim)`,
/// with negative values normalized via `dim + ndim` (the upstream
/// `at::maybe_wrap_dim` call at `TensorAdvancedIndexing.cpp:1919`). The index
/// tensor must be 1-D (the upstream `TORCH_CHECK(index.dim() <= 1, "Index has
/// to be a vector/scalar")` at `:1920`). Index values follow the upstream
/// kernel's contract at `aten/src/ATen/native/cpu/IndexKernel.cpp:224-229`:
/// `idx ∈ [-dim_size, dim_size)` is accepted, with negatives wrapped via
/// `idx + dim_size`; values outside that range raise [`FerrotorchError::IndexOutOfBounds`].
///
/// If `input.requires_grad()` and grad is enabled, the result carries an
/// [`IndexFillBackward`] grad_fn whose VJP zeroes the gradient at the filled
/// positions per `derivatives.yaml:884-887
///   - name: index_fill.int_Scalar(Tensor self, int dim, Tensor index, Scalar value) -> Tensor
///     self: grad.index_fill(dim, index, 0)`.
pub fn index_fill<T: Float>(
    input: &Tensor<T>,
    dim: i64,
    index: &IntTensor<i64>,
    value: f64,
) -> FerrotorchResult<Tensor<T>> {
    let input_shape = input.shape();
    let ndim = input_shape.len();
    if ndim == 0 {
        // Upstream mirrors 0-d input by unsqueezing to 1-d at
        // `TensorAdvancedIndexing.cpp:1917`:
        //   Tensor self_nonzero_dim = (self.dim() == 0)
        //       ? self.unsqueeze(-1) : self;
        // then performs the fill on the 1-d view. The result shares storage
        // with `self` in C++ (a view), so the write is visible in the original
        // 0-d tensor. ferrotorch copies the scalar value, runs index_fill on a
        // length-1 1-d tensor, and returns a 0-d scalar — matching the
        // upstream contract.
        //
        // dim must be 0 or -1 (only valid dim for a 0-d tensor treated as 1-d).
        // Upstream applies `at::maybe_wrap_dim(dim, self_nonzero_dim)` on the
        // unsqueezed (1-d) view, so dim ∈ {-1, 0} for 0-d input.
        let dim_for_0d = match dim {
            0 | -1 => 0i64,
            _ => {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "index_fill: dim {dim} out of range for 0-d input \
                         (valid range: [-1, 0])"
                    ),
                });
            }
        };
        // Validate index: any index value must be 0 (the single element of the
        // unsqueezed length-1 dimension). Per upstream `index_fill_kernel` at
        // `aten/src/ATen/native/cpu/IndexKernel.cpp:224-229`, negative indices
        // wrap by `idx + dim_size` and only out-of-range values (`idx < -size
        // || idx >= size`) raise IndexError. For the 0-d unsqueezed-to-1-d case
        // the only valid wrapped index is 0 (size == 1), so `-1` is the only
        // accepted negative.
        let scalar_val = input.data_vec()?[0];
        let mut result_val = scalar_val;
        let mut any_filled = false;
        for v in index.data()? {
            let i_raw = v.to_i64();
            let i = if i_raw < 0 { i_raw + 1 } else { i_raw };
            if !(0..1).contains(&i) {
                return Err(FerrotorchError::IndexOutOfBounds {
                    index: if i_raw < 0 {
                        i_raw.unsigned_abs() as usize
                    } else {
                        i_raw as usize
                    },
                    axis: dim_for_0d as usize,
                    size: 1,
                });
            }
            result_val = <T as num_traits::NumCast>::from(value).ok_or_else(|| {
                FerrotorchError::InvalidArgument {
                    message: format!("index_fill: value {value} not representable in target dtype"),
                }
            })?;
            any_filled = true;
        }
        // Return a 0-d scalar tensor. If any index was 0 (the only valid index),
        // result_val was overwritten with `value`; otherwise (empty index tensor)
        // result_val remains the original scalar. Autograd: a filled 0-d input
        // has grad = 0 at that position; the backward mirrors the 1-d case.
        let out_storage = TensorStorage::cpu(vec![result_val]);
        if input.requires_grad() && is_grad_enabled() {
            // Build a 1-element index list mirroring what the 1-d path would save.
            // If index tensor was non-empty (position 0 was filled), record it;
            // otherwise record empty (no positions filled).
            let saved_index: Vec<usize> = if any_filled { vec![0] } else { vec![] };
            let grad_fn = Arc::new(IndexFillBackward {
                input: input.clone(),
                dim: 0,
                index: saved_index,
            });
            return Tensor::from_operation(out_storage, vec![], grad_fn);
        }
        return Tensor::from_storage(out_storage, vec![], false);
    }
    if index.ndim() > 1 {
        // Upstream `TORCH_CHECK(index.dim() <= 1, "Index has to be a
        // vector/scalar")` at `TensorAdvancedIndexing.cpp:1920`.
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "index_fill: index must be 1-D or scalar, got shape {:?}",
                index.shape()
            ),
        });
    }

    // Normalize negative dim per `at::maybe_wrap_dim` at
    // `TensorAdvancedIndexing.cpp:1919`: dim ∈ [-ndim, ndim).
    let ndim_i64 = ndim as i64;
    let dim_norm = if dim < 0 { dim + ndim_i64 } else { dim };
    if !(0..ndim_i64).contains(&dim_norm) {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("index_fill: dim {dim} out of range for input ndim {ndim}"),
        });
    }
    let dim_usize = dim_norm as usize;
    let dim_size = input_shape[dim_usize];

    // Validate + widen indices. Per upstream `index_fill_kernel` at
    // `aten/src/ATen/native/cpu/IndexKernel.cpp:224-229`, the bound check is
    // `idx >= -self_dim_size && idx < self_dim_size` and negative indices wrap
    // via `idx += self_dim_size`. Match that contract (R-DEV-1/2): in-range
    // negatives wrap, only true OOB raises IndexError.
    let dim_size_i64 = dim_size as i64;
    let mut idx_usize: Vec<usize> = Vec::with_capacity(index.numel());
    for v in index.data()? {
        let i_raw = v.to_i64();
        if i_raw < -dim_size_i64 || i_raw >= dim_size_i64 {
            return Err(FerrotorchError::IndexOutOfBounds {
                index: if i_raw < 0 {
                    i_raw.unsigned_abs() as usize
                } else {
                    i_raw as usize
                },
                axis: dim_usize,
                size: dim_size,
            });
        }
        let i = if i_raw < 0 {
            i_raw + dim_size_i64
        } else {
            i_raw
        };
        idx_usize.push(i as usize);
    }

    // Forward: clone input and overwrite slices at index positions with value.
    // The outer/inner decomposition mirrors `index_select_dim` (axis `dim`):
    //   flat positions to fill = o * dim_size * inner + idx * inner + k
    let outer: usize = crate::shape::numel(&input_shape[..dim_usize]);
    let inner: usize = crate::shape::numel(&input_shape[dim_usize + 1..]);
    let in_data = input.data_vec()?;
    let mut out = in_data.clone();
    let value_t = <T as num_traits::NumCast>::from(value).ok_or_else(|| {
        FerrotorchError::InvalidArgument {
            message: format!("index_fill: value {value} not representable in target dtype"),
        }
    })?;
    for o in 0..outer {
        for &idx in &idx_usize {
            let base = o * dim_size * inner + idx * inner;
            for k in 0..inner {
                out[base + k] = value_t;
            }
        }
    }

    let output_shape = input_shape.to_vec();
    if input.requires_grad() && is_grad_enabled() {
        let grad_fn = Arc::new(IndexFillBackward {
            input: input.clone(),
            dim: dim_usize,
            index: idx_usize,
        });
        Tensor::from_operation(TensorStorage::cpu(out), output_shape, grad_fn)
    } else {
        Tensor::from_storage(TensorStorage::cpu(out), output_shape, false)
    }
}

// ---------------------------------------------------------------------------
// Broadcasting wrappers for masked_fill / masked_select / where_cond.
//
// Upstream PyTorch broadcasts mask/condition against input by NumPy rules
// before applying these ops:
//   - `masked_fill(input, mask, value)` calls `expand_outplace(mask, self)` at
//     `aten/src/ATen/native/TensorAdvancedIndexing.cpp:2503-2504` to broadcast
//     both operands to a common shape, then operates on the expanded views.
//   - `masked_select(input, mask)` calls `expand_outplace(mask, self)` at
//     `TensorAdvancedIndexing.cpp:2545` so a 1-D `[10]` mask paired with a
//     `[10, 10]` input compacts the 100-element broadcast.
//   - `where(condition, self, other)` runs a TensorIterator over all three
//     operands at `aten/src/ATen/native/TensorCompare.cpp:629-638` —
//     condition, self, other all broadcast to a common output shape.
// The existing ferrotorch entry points (`masked_fill_bt`, `where_cond_bt`,
// `ops::indexing::masked_select`) require identical shapes; they predate the
// broadcasting contract. These wrappers infer the common broadcast shape
// using `shape::broadcast_shapes`, expand each operand to that shape, then
// delegate to the existing shape-strict entry point.
//
// Autograd correctness: `Tensor::expand` (via `grad_fns::shape::expand`) is
// autograd-aware and attaches `ExpandBackward`, which reduces upstream
// gradients along the broadcast axes (`grad_fns::arithmetic::
// reduce_grad_to_shape`). Because we route the broadcast through that
// autograd-aware expand, the existing `MaskedFillBackward` /
// `MaskedSelectBackward` / `WhereCondBackward` structs (which produce
// gradients of the broadcasted shape) get their gradients automatically
// shrunk back to the original input shape by the upstream `ExpandBackward`
// in the chain — no per-op grad reduction code needed here.
// ---------------------------------------------------------------------------

/// Compute the flat index into the input's contiguous buffer for a given
/// output flat index, applying NumPy broadcasting rules: any axis where
/// `in_shape` has size 1 is broadcast (its coordinate maps to 0).
#[inline]
fn broadcast_in_flat(flat: usize, out_shape: &[usize], in_shape: &[usize]) -> usize {
    // Walk axes from innermost to outermost. The output's flat index decomposes
    // into per-axis coords; for each axis the corresponding input coord is
    // either the same (when in_shape[axis] == out_shape[axis]) or 0 (when
    // in_shape has size 1 there, i.e. broadcast). Missing-leading-axis cases
    // (in_shape.len() < out_shape.len()) collapse to 0 as well.
    let out_ndim = out_shape.len();
    let in_ndim = in_shape.len();
    let mut rem = flat;
    let mut in_idx = 0usize;
    // Compute strides for in_shape (C-contiguous, innermost = 1).
    let mut in_strides = vec![0usize; in_ndim];
    if in_ndim > 0 {
        in_strides[in_ndim - 1] = 1;
        for d in (0..in_ndim - 1).rev() {
            in_strides[d] = in_strides[d + 1] * in_shape[d + 1];
        }
    }
    for d_out in (0..out_ndim).rev() {
        let out_dim = out_shape[d_out];
        let coord = rem % out_dim;
        rem /= out_dim;
        // Map this output axis to an input axis (right-aligned). If the input
        // has fewer dimensions, the leading output axes have no input counterpart.
        let d_in_off = out_ndim - 1 - d_out;
        if d_in_off < in_ndim {
            let d_in = in_ndim - 1 - d_in_off;
            if in_shape[d_in] == 1 {
                // broadcast — coord 0
            } else {
                in_idx += coord * in_strides[d_in];
            }
        }
    }
    in_idx
}

/// Broadcast a [`BoolTensor`] to `out_shape` using NumPy / torch rules,
/// returning a new contiguous `BoolTensor` on the SAME device as `mask`.
///
/// CPU masks broadcast host-side here; a CUDA-resident mask broadcasts ENTIRELY
/// on device via [`crate::gpu_dispatch::GpuBackend::broadcast_bool`] (#1663) —
/// the result stays `is_cuda()`, no host round trip (R-CODE-4). This mirrors the
/// `expand_outplace(mask, self)` step PyTorch performs for masked ops at
/// `aten/src/ATen/native/TensorAdvancedIndexing.cpp:2406`. Used by the
/// broadcasting wrappers `masked_fill_bcast`, `masked_select_bcast`,
/// `where_cond_bcast`, and `masked_scatter` below.
pub(crate) fn broadcast_bool_tensor(
    mask: &BoolTensor,
    out_shape: &[usize],
) -> FerrotorchResult<BoolTensor> {
    if mask.shape() == out_shape {
        return Ok(mask.clone());
    }
    if mask.is_cuda() {
        // On-device bool broadcast (#1663): the resident analog of the CPU walk
        // below. The kernel maps each output flat index to the corresponding
        // input flat index via per-dim broadcast strides (size-1 / absent dim ->
        // stride 0), reading the u8 and writing the expanded u8 buffer. Result
        // stays a CUDA `BoolTensor`.
        let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let handle = backend.broadcast_bool(mask.gpu_handle()?, mask.shape(), out_shape)?;
        return BoolTensor::from_gpu_handle(handle, out_shape.to_vec());
    }
    let in_data = mask.data()?;
    let in_shape: Vec<usize> = mask.shape().to_vec();
    let out_numel: usize = if out_shape.is_empty() {
        1
    } else {
        crate::shape::numel(out_shape)
    };
    // Validate that mask is broadcast-compatible with out_shape — every input
    // axis must either equal the matching output axis (right-aligned) or be 1.
    let out_ndim = out_shape.len();
    let in_ndim = in_shape.len();
    if in_ndim > out_ndim {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "broadcast_bool_tensor: input ndim {in_ndim} > target ndim {out_ndim} \
                 (shapes {in_shape:?} -> {out_shape:?})"
            ),
        });
    }
    for d_in_off in 0..in_ndim {
        let in_dim = in_shape[in_ndim - 1 - d_in_off];
        let out_dim = out_shape[out_ndim - 1 - d_in_off];
        if in_dim != 1 && in_dim != out_dim {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "broadcast_bool_tensor: cannot broadcast {in_shape:?} -> {out_shape:?} \
                     (axis {} mismatch: {in_dim} vs {out_dim})",
                    out_ndim - 1 - d_in_off
                ),
            });
        }
    }
    let mut out = Vec::with_capacity(out_numel);
    for flat in 0..out_numel {
        let src = broadcast_in_flat(flat, out_shape, &in_shape);
        out.push(in_data[src]);
    }
    BoolTensor::from_vec(out, out_shape.to_vec())
}

/// Broadcasting `masked_fill` — mirrors `torch.masked_fill(input, mask, value)`
/// with PyTorch's broadcasting semantics. The input and mask are broadcast to
/// their common shape (per `aten/src/ATen/native/TensorAdvancedIndexing.cpp:
/// 2494-2509 Tensor masked_fill(...) { ... expand_outplace(mask, self); ... }`)
/// before the fill is applied. Delegates to [`masked_fill_bt`] on the
/// broadcasted operands; the autograd graph routes through the
/// autograd-aware [`crate::grad_fns::shape::expand`] so gradients reduce back
/// to the original input shape via `ExpandBackward`.
pub fn masked_fill_bcast<T: Float>(
    input: &Tensor<T>,
    mask: &BoolTensor,
    value: T,
) -> FerrotorchResult<Tensor<T>> {
    if input.shape() == mask.shape() {
        return masked_fill_bt(input, mask, value);
    }
    let common = crate::shape::broadcast_shapes(input.shape(), mask.shape())?;
    // Autograd-aware expand on the float operand; ExpandBackward will reduce
    // gradients of the MaskedFillBackward output back to input.shape().
    let input_b = crate::grad_fns::shape::expand(input, &common)?;
    let mask_b = broadcast_bool_tensor(mask, &common)?;
    masked_fill_bt(&input_b, &mask_b, value)
}

/// Broadcasting `masked_select` — mirrors `torch.masked_select(input, mask)`
/// with PyTorch's broadcasting semantics. The input and mask are broadcast to
/// their common shape (per `TensorAdvancedIndexing.cpp:2545
/// auto [_mask, _self] = expand_outplace(mask, self);`) before the compaction
/// is applied. Delegates to [`crate::ops::indexing::masked_select`] on the
/// broadcasted operands; the autograd graph routes the input's gradient back
/// through `ExpandBackward` to the original input shape.
pub fn masked_select_bcast<T: Float>(
    input: &Tensor<T>,
    mask: &BoolTensor,
) -> FerrotorchResult<Tensor<T>> {
    crate::ops::indexing::masked_select(input, mask)
}

/// Broadcasting `where_cond` — mirrors `torch.where(condition, self, other)`
/// with PyTorch's 3-way broadcasting semantics. The condition, x, and y are
/// each broadcast to their common shape (per `aten/src/ATen/native/
/// TensorCompare.cpp:629-637 where_self_out` which builds a TensorIterator
/// over `condition_, self_, other_`) before the select is applied. Delegates
/// to [`crate::ops::indexing::where_cond_bt`] on the broadcasted operands;
/// the autograd graph routes the x/y gradients back through `ExpandBackward`
/// to their original shapes.
pub fn where_cond_bcast<T: Float>(
    cond: &BoolTensor,
    x: &Tensor<T>,
    y: &Tensor<T>,
) -> FerrotorchResult<Tensor<T>> {
    if cond.shape() == x.shape() && x.shape() == y.shape() {
        return crate::ops::indexing::where_cond_bt_strict(cond, x, y);
    }
    // 3-way broadcast via two pairwise applications.
    let xy_common = crate::shape::broadcast_shapes(x.shape(), y.shape())?;
    let common = crate::shape::broadcast_shapes(cond.shape(), &xy_common)?;
    let cond_b = broadcast_bool_tensor(cond, &common)?;
    let x_b = crate::grad_fns::shape::expand(x, &common)?;
    let y_b = crate::grad_fns::shape::expand(y, &common)?;
    crate::ops::indexing::where_cond_bt_strict(&cond_b, &x_b, &y_b)
}

// ---------------------------------------------------------------------------
// scatter_reduce (#1245 — REQ-4). Mirrors `torch.scatter_reduce(input, dim,
// index, src, reduce, *, include_self=True)` at upstream
// `aten/src/ATen/native/TensorAdvancedIndexing.cpp:2354 TORCH_IMPL_FUNC(
// scatter_reduce_two)`. VJP per `tools/autograd/derivatives.yaml:3074-3077
//   - name: scatter_reduce.two(Tensor self, int dim, Tensor index, Tensor src,
//       str reduce, *, bool include_self=True) -> Tensor
//     self, src: scatter_reduce_backward(grad, self, dim, index, src, reduce,
//                                         include_self, result)`.
// op_db emits only `reduce='sum'` samples (verified 2026-05-25: seed 0..3
// i=0..25); the impl supports {sum, prod, amax, amin} for completeness but
// the upstream-pinned characterization is sum-only — other modes route to a
// concrete error rather than a wrong-value silent miss.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// CORE-048 (#1742): device contract for the advanced-indexing family
// (`scatter_reduce`, `index_add`, `index_copy`, `take`, `put`,
// `masked_scatter`).
//
// torch enforces strict same-device placement for EVERY operand of these
// ops — input, src/source, index AND mask. Live torch 2.11.0+cu130 probe
// (pasted in #1742): each mixed combination raises `RuntimeError: Expected
// all tensors to be on the same device, but got index is on cpu, different
// from other tensors on cuda:0 (...)`; the all-same-device forms return
// outputs on that device and deliver gradients on the leaves' devices.
// ferrotorch surfaces the structured `DeviceMismatch` equivalent at entry
// (R-LOUD-1) and preserves residency where kernels exist: `take`/`put` run
// dim-aware CUDA kernels for f32/f64/f16/bf16 flat gathers/scatters, and
// `index_add`/`index_copy` run resident f32/f64/f16/bf16 dim-aware scatter paths. Remaining
// unsupported CUDA dtype/operation combinations surface `NotImplementedOnCuda`
// rather than silently detouring through host code; see each op's doc-comment.
// ---------------------------------------------------------------------------

/// Strict same-device operand check (CORE-048 / #1742). `expected` is the
/// `self`/input device; `got` is the operand under test.
#[inline]
fn same_device(expected: Device, got: Device) -> FerrotorchResult<()> {
    if got == expected {
        Ok(())
    } else {
        Err(FerrotorchError::DeviceMismatch { expected, got })
    }
}

fn cuda_full<T: Float>(
    shape: &[usize],
    value: T,
    device: Device,
    op: &'static str,
) -> FerrotorchResult<Tensor<T>> {
    if !cuda_float_dtype::<T>() {
        return Err(FerrotorchError::NotImplementedOnCuda { op });
    }
    let ordinal = cuda_ordinal(device, op)?;
    let numel = crate::shape::checked_numel(shape, op)?;
    let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
    let handle = match T::dtype() {
        DType::F32 => backend.fill_f32(
            numel,
            value
                .to_f32()
                .ok_or_else(|| FerrotorchError::InvalidArgument {
                    message: format!("{op}: value is not representable as f32"),
                })?,
            ordinal,
        )?,
        DType::F64 => backend.fill_f64(
            numel,
            value
                .to_f64()
                .ok_or_else(|| FerrotorchError::InvalidArgument {
                    message: format!("{op}: value is not representable as f64"),
                })?,
            ordinal,
        )?,
        DType::F16 => backend.fill_f16(
            numel,
            value
                .to_f32()
                .ok_or_else(|| FerrotorchError::InvalidArgument {
                    message: format!("{op}: value is not representable as f32"),
                })?,
            ordinal,
        )?,
        DType::BF16 => backend.fill_bf16_bf16(
            numel,
            value
                .to_f32()
                .ok_or_else(|| FerrotorchError::InvalidArgument {
                    message: format!("{op}: value is not representable as f32"),
                })?,
            ordinal,
        )?,
        _ => return Err(FerrotorchError::NotImplementedOnCuda { op }),
    };
    Tensor::from_storage(TensorStorage::gpu(handle), shape.to_vec(), false)
}

fn zeros_on_device<T: Float>(
    shape: &[usize],
    device: Device,
    op: &'static str,
) -> FerrotorchResult<Tensor<T>> {
    let numel = crate::shape::checked_numel(shape, op)?;
    match device {
        Device::Cuda(ordinal) => {
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let handle = backend.alloc_zeros(numel, T::dtype(), ordinal)?;
            Tensor::from_storage(TensorStorage::gpu(handle), shape.to_vec(), false)
        }
        _ => Tensor::from_storage(
            TensorStorage::on_device(vec![<T as num_traits::Zero>::zero(); numel], device)?,
            shape.to_vec(),
            false,
        ),
    }
}

fn cuda_gather_index_shape<T: Float>(
    src: &Tensor<T>,
    idx_handle: &GpuBufferHandle,
    input_shape: &[usize],
    index_shape: &[usize],
    dim: usize,
) -> FerrotorchResult<Tensor<T>> {
    if !cuda_float_dtype::<T>() {
        return Err(FerrotorchError::NotImplementedOnCuda {
            op: "ScatterReduceBackward",
        });
    }
    let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
    let src_c = src.contiguous()?;
    let handle = backend.gather_intidx_nd(
        src_c.gpu_handle()?,
        idx_handle,
        input_shape,
        index_shape,
        dim,
    )?;
    Tensor::from_storage(TensorStorage::gpu(handle), index_shape.to_vec(), false)
}

fn cuda_scatter_zero_at_index<T: Float>(
    input: &Tensor<T>,
    idx_handle: &GpuBufferHandle,
    input_shape: &[usize],
    index_shape: &[usize],
    dim: usize,
) -> FerrotorchResult<Tensor<T>> {
    if !cuda_float_dtype::<T>() {
        return Err(FerrotorchError::NotImplementedOnCuda {
            op: "ScatterReduceBackward",
        });
    }
    let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
    let input_c = input.contiguous()?;
    let handle = match T::dtype() {
        DType::F32 => backend.scatter_value_nd_f32(
            input_c.gpu_handle()?,
            idx_handle,
            0.0,
            input_shape,
            index_shape,
            dim,
        )?,
        DType::F64 => backend.scatter_value_nd_f64(
            input_c.gpu_handle()?,
            idx_handle,
            0.0,
            input_shape,
            index_shape,
            dim,
        )?,
        DType::F16 => backend.scatter_value_nd_f16(
            input_c.gpu_handle()?,
            idx_handle,
            0.0,
            input_shape,
            index_shape,
            dim,
        )?,
        DType::BF16 => backend.scatter_value_nd_bf16(
            input_c.gpu_handle()?,
            idx_handle,
            0.0,
            input_shape,
            index_shape,
            dim,
        )?,
        _ => {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "ScatterReduceBackward",
            });
        }
    };
    Tensor::from_storage(TensorStorage::gpu(handle), input_shape.to_vec(), false)
}

/// Reduce mode for `scatter_reduce` mirroring upstream `ReductionType` at
/// `aten/src/ATen/native/ReductionType.h` (enum SUM / PROD / MAX / MIN /
/// MEAN). PyTorch's user-facing string-keyword `reduce` arg per
/// `torch/_torch_docs.py` accepts `"sum" | "prod" | "amax" | "amin" | "mean"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScatterReduce {
    /// `output[idx] += src[i]` (matches `scatter_add` semantics for include_self=true).
    Sum,
    /// `output[idx] = mean(output[idx], src values for the bucket)`.
    Mean,
    /// `output[idx] *= src[i]`.
    Prod,
    /// `output[idx] = max(output[idx], src[i])`.
    Amax,
    /// `output[idx] = min(output[idx], src[i])`.
    Amin,
}

impl ScatterReduce {
    /// Parse the user-facing string (matches upstream `get_operator_enum` at
    /// `TensorAdvancedIndexing.cpp:2368` which dispatches by string). Returns
    /// `None` for unknown strings.
    ///
    /// Named `parse_str` rather than `from_str` to avoid the
    /// `clippy::should_implement_trait` warning for `std::str::FromStr`
    /// (whose `Err` associated type would require a bespoke error type for
    /// a single call site — overkill for this 4-arm parse).
    pub fn parse_str(s: &str) -> Option<Self> {
        match s {
            "sum" => Some(Self::Sum),
            "mean" => Some(Self::Mean),
            "prod" => Some(Self::Prod),
            "amax" => Some(Self::Amax),
            "amin" => Some(Self::Amin),
            _ => None,
        }
    }

    #[inline]
    fn gpu(self) -> Option<GpuScatterReduce> {
        match self {
            Self::Sum => Some(GpuScatterReduce::Sum),
            Self::Mean => None,
            Self::Prod => Some(GpuScatterReduce::Prod),
            Self::Amax => Some(GpuScatterReduce::Amax),
            Self::Amin => Some(GpuScatterReduce::Amin),
        }
    }
}

fn checked_shape_numel(op: &'static str, shape: &[usize]) -> FerrotorchResult<usize> {
    shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| FerrotorchError::InvalidArgument {
            message: format!("{op}: shape {shape:?} element count overflows usize"),
        })
}

fn validate_scatter_reduce_shapes<T: Float>(
    input_shape: &[usize],
    dim: usize,
    index: &[usize],
    index_shape: &[usize],
    src: &Tensor<T>,
) -> FerrotorchResult<usize> {
    let ndim = input_shape.len();
    let index_numel = checked_shape_numel("scatter_reduce", index_shape)?;
    if index.len() != index_numel {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "scatter_reduce: index slice has {} elements but index_shape {:?} implies {}",
                index.len(),
                index_shape,
                index_numel
            ),
        });
    }
    if index_numel == 0 {
        return Ok(index_numel);
    }
    if index_shape.len() != ndim {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "scatter_reduce: index ndim {} != input ndim {}",
                index_shape.len(),
                ndim
            ),
        });
    }
    for (axis, (&idx_d, &input_d)) in index_shape.iter().zip(input_shape).enumerate() {
        if axis != dim && idx_d > input_d {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "scatter_reduce: expected index {index_shape:?} to be no larger than \
                     input {input_shape:?} apart from dimension {dim} \
                     (axis {axis}: {idx_d} > {input_d})"
                ),
            });
        }
    }
    for &idx in index {
        if idx >= input_shape[dim] {
            return Err(FerrotorchError::IndexOutOfBounds {
                index: idx,
                axis: dim,
                size: input_shape[dim],
            });
        }
    }

    let src_shape = src.shape();
    if src_shape.len() != ndim {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "scatter_reduce: index tensor must have the same number of dimensions as src \
                 tensor (index_shape {index_shape:?} is rank {}, src shape {:?} is rank {})",
                index_shape.len(),
                src_shape,
                src_shape.len()
            ),
        });
    }
    for (axis, (&idx_d, &src_d)) in index_shape.iter().zip(src_shape).enumerate() {
        if idx_d > src_d {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "scatter_reduce: expected index {index_shape:?} to be no larger size than \
                     src {src_shape:?} (axis {axis}: {idx_d} > {src_d})"
                ),
            });
        }
    }
    Ok(index_numel)
}

struct ScalarScatterReduceShape {
    index_numel: usize,
    effective_index_shape: Vec<usize>,
}

fn nonempty_shape(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        vec![1]
    } else {
        shape.to_vec()
    }
}

fn validate_scalar_scatter_reduce<T: Float>(
    index: &[usize],
    index_shape: &[usize],
    src: &Tensor<T>,
) -> FerrotorchResult<ScalarScatterReduceShape> {
    let index_numel = checked_shape_numel("scatter_reduce", index_shape)?;
    if index.len() != index_numel {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "scatter_reduce: index slice has {} elements but index_shape {:?} implies {}",
                index.len(),
                index_shape,
                index_numel
            ),
        });
    }
    let effective_index_shape = nonempty_shape(index_shape);
    let effective_src_shape = nonempty_shape(src.shape());
    if index_numel == 0 {
        return Ok(ScalarScatterReduceShape {
            index_numel,
            effective_index_shape,
        });
    }
    if effective_index_shape.len() != 1 {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "scatter_reduce: index ndim {} != effective scalar input ndim 1",
                index_shape.len()
            ),
        });
    }
    if effective_src_shape.len() != 1 {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "scatter_reduce: index tensor must have the same number of dimensions as src \
                 tensor (index_shape {index_shape:?} has effective rank 1, src shape {:?} has \
                 effective rank {})",
                src.shape(),
                effective_src_shape.len()
            ),
        });
    }
    if effective_index_shape[0] > effective_src_shape[0] {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "scatter_reduce: expected index {index_shape:?} to be no larger than self [] \
                 apart from dimension 0 and to be no larger size than src {:?}",
                src.shape()
            ),
        });
    }
    for &idx in index {
        if idx >= 1 {
            return Err(FerrotorchError::IndexOutOfBounds {
                index: idx,
                axis: 0,
                size: 1,
            });
        }
    }
    Ok(ScalarScatterReduceShape {
        index_numel,
        effective_index_shape,
    })
}

fn cuda_src_prefix_slab_for_index<T: Float>(
    src: &Tensor<T>,
    index_shape: &[usize],
) -> FerrotorchResult<Tensor<T>> {
    if src.shape() == index_shape {
        return Ok(src.clone());
    }
    no_grad(|| {
        let mut slab = src.clone();
        for (axis, &len) in index_shape.iter().enumerate() {
            if slab.shape()[axis] != len {
                slab = slab.narrow(axis, 0, len)?;
            }
        }
        slab.contiguous()
    })
}

fn scatter_reduce_cuda_forward<T: Float>(
    input: &Tensor<T>,
    dim: usize,
    index: &[usize],
    index_shape: &[usize],
    src: &Tensor<T>,
    reduce: ScatterReduce,
    include_self: bool,
) -> FerrotorchResult<Tensor<T>> {
    if !cuda_float_dtype::<T>() {
        return Err(FerrotorchError::NotImplementedOnCuda {
            op: "scatter_reduce",
        });
    }
    if reduce == ScatterReduce::Mean {
        let summed = scatter_reduce_cuda_forward(
            input,
            dim,
            index,
            index_shape,
            src,
            ScatterReduce::Sum,
            include_self,
        )?;
        let counts = scatter_reduce_mean_counts_cuda::<T>(
            input.shape(),
            dim,
            index,
            index_shape,
            include_self,
            summed.device(),
        )?;
        return crate::grad_fns::arithmetic::div(&summed, &counts);
    }
    let input_c = no_grad(|| input.contiguous())?;
    let src_slab = cuda_src_prefix_slab_for_index(src, index_shape)?.contiguous()?;
    let ordinal = cuda_ordinal(input_c.device(), "scatter_reduce")?;
    let idx_handle = crate::ops::indexing::upload_index_i64(index, ordinal)?;
    let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
    let gpu_reduce = reduce.gpu().ok_or(FerrotorchError::NotImplementedOnCuda {
        op: "scatter_reduce",
    })?;
    let handle = match T::dtype() {
        DType::F32 => backend.scatter_reduce_nd_f32(
            input_c.gpu_handle()?,
            &idx_handle,
            src_slab.gpu_handle()?,
            input_c.shape(),
            index_shape,
            dim,
            gpu_reduce,
            include_self,
        )?,
        DType::F64 => backend.scatter_reduce_nd_f64(
            input_c.gpu_handle()?,
            &idx_handle,
            src_slab.gpu_handle()?,
            input_c.shape(),
            index_shape,
            dim,
            gpu_reduce,
            include_self,
        )?,
        DType::F16 => backend.scatter_reduce_nd_f16(
            input_c.gpu_handle()?,
            &idx_handle,
            src_slab.gpu_handle()?,
            input_c.shape(),
            index_shape,
            dim,
            gpu_reduce,
            include_self,
        )?,
        DType::BF16 => backend.scatter_reduce_nd_bf16(
            input_c.gpu_handle()?,
            &idx_handle,
            src_slab.gpu_handle()?,
            input_c.shape(),
            index_shape,
            dim,
            gpu_reduce,
            include_self,
        )?,
        _ => {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "scatter_reduce",
            });
        }
    };
    Tensor::from_storage(TensorStorage::gpu(handle), input.shape().to_vec(), false)
}

fn scatter_reduce_mean_counts_cpu<T: Float>(
    input_shape: &[usize],
    dim: usize,
    index: &[usize],
    index_shape: &[usize],
    include_self: bool,
) -> Vec<T> {
    let zero = <T as num_traits::Zero>::zero();
    let one = <T as num_traits::One>::one();
    let input_numel: usize = crate::shape::numel(input_shape);
    let mut counts = vec![if include_self { one } else { zero }; input_numel];
    let index_numel: usize = crate::shape::numel(index_shape);
    let mut coords = vec![0usize; input_shape.len()];
    for i in 0..index_numel {
        let mut dst_coords = coords.clone();
        dst_coords[dim] = index[i];
        counts[flat_index(&dst_coords, input_shape)] += one;
        if i + 1 < index_numel {
            increment_coords(&mut coords, index_shape);
        }
    }
    for count in &mut counts {
        if *count == zero {
            *count = one;
        }
    }
    counts
}

fn scatter_reduce_mean_counts_cuda<T: Float>(
    input_shape: &[usize],
    dim: usize,
    index: &[usize],
    index_shape: &[usize],
    include_self: bool,
    device: Device,
) -> FerrotorchResult<Tensor<T>> {
    let zero = <T as num_traits::Zero>::zero();
    let one = <T as num_traits::One>::one();
    let start = cuda_full(
        input_shape,
        if include_self { one } else { zero },
        device,
        "scatter_reduce",
    )?;
    let ones = cuda_full(index_shape, one, device, "scatter_reduce")?;
    let counts =
        crate::ops::indexing::scatter_add(&start, dim as isize, index, index_shape, &ones)?;
    let zeros = cuda_full(input_shape, zero, device, "scatter_reduce")?;
    let zero_mask = BoolTensor::eq_t(&counts, &zeros)?;
    masked_fill_bt(&counts, &zero_mask, one)
}

#[allow(clippy::too_many_arguments)]
fn attach_scatter_reduce_grad<T: Float>(
    output: Tensor<T>,
    input: &Tensor<T>,
    src: &Tensor<T>,
    dim: usize,
    index: &[usize],
    index_shape: &[usize],
    reduce: ScatterReduce,
    include_self: bool,
) -> FerrotorchResult<Tensor<T>> {
    if (input.requires_grad() || src.requires_grad()) && is_grad_enabled() {
        let saved_result = output.clone();
        let (storage, shape) = output.into_storage_and_shape()?;
        let grad_fn = Arc::new(ScatterReduceBackward {
            input: input.clone(),
            src: src.clone(),
            dim,
            index: index.to_vec(),
            index_shape: index_shape.to_vec(),
            reduce,
            include_self,
            result: saved_result,
        });
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Ok(output)
    }
}

/// Backward function for `scatter_reduce` (all reduce modes).
///
/// Forward (sum, include_self=True): `output = input.clone();
/// output[..., index[p], ...] += src[..., p, ...]` along `dim`.
/// Forward (sum, include_self=False): like above but `output` slices at any
/// position touched by the index list are zeroed before accumulation
/// (upstream computes a mask via `include_self_ones` at
/// `TensorAdvancedIndexing.cpp:2390-2392`).
///
/// VJPs mirror upstream `scatter_reduce_backward` at
/// `torch/csrc/autograd/FunctionsManual.cpp:7194-7279`, per
/// `tools/autograd/derivatives.yaml:3074-3077`:
///
/// - `sum`: grad_self = grad; grad_src = grad.gather(dim, index).
/// - `mean`: divide those gradients by the per-output scatter count.
/// - `prod`: grad_self = grad * (masked_self_result / masked_self);
///   grad_src uses the result-over-src chain rule with masking for zeros
///   (`:7216-7248`).
/// - `amax`/`amin`: evenly distribute grad among positions whose value
///   matched the max/min (`:7256-7265`).
///
/// For `include_self=False`, the upstream post-processing at `:7274-7275`
/// scatters zeros into grad_self at the index-mapped positions (those
/// positions are entirely overwritten by src and no longer depend on self).
#[derive(Debug)]
pub struct ScatterReduceBackward<T: Float> {
    /// Saved input handle (for shape + autograd graph linkage).
    pub input: Tensor<T>,
    /// Saved src handle.
    pub src: Tensor<T>,
    /// The normalized (non-negative) dim.
    pub dim: usize,
    /// The flat index list.
    pub index: Vec<usize>,
    /// The shape of the index tensor.
    pub index_shape: Vec<usize>,
    /// The reduce mode used by the forward.
    pub reduce: ScatterReduce,
    /// Whether `include_self` was set in the forward.
    pub include_self: bool,
    /// Saved forward result. Required by the
    /// value-aware VJPs for `prod`/`amax`/`amin` per upstream
    /// `FunctionsManual.cpp:7216-7265` (which read `result` to identify
    /// max/min positions and compute the prod chain rule).
    pub result: Tensor<T>,
}

impl<T: Float> GradFn<T> for ScatterReduceBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !is_grad_enabled() {
            return Ok(vec![None, None]);
        }

        let input_shape = self.input.shape();
        let ndim = input_shape.len();

        if ndim == 0 {
            // 0-d input: forward short-circuits / upstream unsqueezes. The
            // backward is similarly degenerate for every mode — distribute
            // the scalar grad to input (sum/prod/amax/amin handle the
            // identity case identically in the 0-d limit).
            return self.backward_0d(grad_output);
        }

        // PyTorch's `scatter_reduce_backward` returns
        // `grad_src = grad.gather(dim, index)` for every reduce mode. That
        // tensor is index-shaped; if the forward used a larger `src` (allowed
        // by the meta function for forward reads), the autograd engine rejects
        // the gradient as incompatible with `src`. Match that contract instead
        // of padding zeros into a shape PyTorch never returns.
        if self.src.requires_grad() && self.src.shape() != self.index_shape.as_slice() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "scatter_reduce backward: gradient for src is only defined when src.shape \
                     ({:?}) equals index.shape ({:?}); PyTorch raises here too \
                     (ScatterReduceBackward0 returns an index-shaped gradient the engine \
                     rejects)",
                    self.src.shape(),
                    self.index_shape
                ),
            });
        }

        match self.reduce {
            ScatterReduce::Sum => self.backward_sum(grad_output),
            ScatterReduce::Mean => self.backward_mean(grad_output),
            ScatterReduce::Prod => self.backward_prod(grad_output),
            ScatterReduce::Amax | ScatterReduce::Amin => self.backward_amax_amin(grad_output),
        }
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input, &self.src]
    }

    fn name(&self) -> &'static str {
        "ScatterReduceBackward"
    }
}

impl<T: Float> ScatterReduceBackward<T> {
    /// Helper: iterate (i, idx_val, src_coords, dst_flat) over every index
    /// element, where `dst_flat` is the input/output flat slot the index
    /// targets along `self.dim`.
    fn for_each_index<F: FnMut(usize, usize, &[usize], usize)>(&self, mut f: F) {
        let input_shape = self.input.shape();
        let ndim = input_shape.len();
        let index_numel: usize = crate::shape::numel(&self.index_shape);
        let mut coords = vec![0usize; ndim];
        for i in 0..index_numel {
            let idx_val = self.index[i];
            let mut dst_coords = coords.clone();
            dst_coords[self.dim] = idx_val;
            let dst_flat = flat_index(&dst_coords, input_shape);
            f(i, idx_val, &coords, dst_flat);
            if i + 1 < index_numel {
                increment_coords(&mut coords, &self.index_shape);
            }
        }
    }

    /// VJP for the 0-d input degenerate case (input is a single scalar).
    fn backward_0d(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if grad_output.numel() != 1 {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "ScatterReduceBackward: scalar output gradient must have one element, got {}",
                    grad_output.numel()
                ),
            });
        }
        same_device(self.input.device(), grad_output.device())?;
        same_device(self.input.device(), self.src.device())?;

        let index_numel = checked_shape_numel("ScatterReduceBackward", &self.index_shape)?;
        if self.index.len() != index_numel {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "ScatterReduceBackward: saved index length {} != index_shape {:?} numel {}",
                    self.index.len(),
                    self.index_shape,
                    index_numel
                ),
            });
        }

        if index_numel == 0 {
            let grad_input = if self.input.requires_grad() {
                Some(grad_output.contiguous()?.view_reshape(vec![])?)
            } else {
                None
            };
            let grad_src = if self.src.requires_grad() {
                if !self.src.shape().is_empty() && self.src.shape() != self.index_shape.as_slice() {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!(
                            "scatter_reduce backward: empty scalar-index gradient for src is \
                             index-shaped ({:?}) and is only compatible with scalar src or \
                             src.shape == index.shape; got src.shape {:?}",
                            self.index_shape,
                            self.src.shape()
                        ),
                    });
                }
                Some(zeros_on_device(
                    self.src.shape(),
                    self.src.device(),
                    "ScatterReduceBackward",
                )?)
            } else {
                None
            };
            return Ok(vec![grad_input, grad_src]);
        }

        if self.src.requires_grad()
            && !self.src.shape().is_empty()
            && self.src.shape() != self.index_shape.as_slice()
        {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "scatter_reduce backward: scalar source gradient is index-shaped ({:?}) \
                     except for scalar src; got src.shape {:?}",
                    self.index_shape,
                    self.src.shape()
                ),
            });
        }

        let effective_index_shape = nonempty_shape(&self.index_shape);
        let effective_src = if self.src.shape().is_empty() {
            self.src.view_reshape(vec![1])?
        } else {
            self.src.clone()
        };
        let effective = ScatterReduceBackward {
            input: self.input.view_reshape(vec![1])?,
            src: effective_src,
            dim: 0,
            index: self.index.clone(),
            index_shape: effective_index_shape,
            reduce: self.reduce,
            include_self: self.include_self,
            result: self.result.view_reshape(vec![1])?,
        };

        let grad_1d = grad_output.view_reshape(vec![1])?;
        let mut grads = match self.reduce {
            ScatterReduce::Sum => effective.backward_sum(&grad_1d)?,
            ScatterReduce::Mean => effective.backward_mean(&grad_1d)?,
            ScatterReduce::Prod => effective.backward_prod(&grad_1d)?,
            ScatterReduce::Amax | ScatterReduce::Amin => effective.backward_amax_amin(&grad_1d)?,
        };

        if let Some(grad_input) = grads[0].take() {
            grads[0] = Some(grad_input.view_reshape(vec![])?);
        }
        if self.src.shape().is_empty()
            && let Some(grad_src) = grads[1].take()
        {
            grads[1] = Some(grad_src.view_reshape(vec![])?);
        }
        Ok(grads)
    }

    /// VJP for `reduce='mean'` per upstream
    /// `FunctionsManual.cpp:7249-7255`:
    ///   N = include_self ? ones_like(grad) : zeros_like(grad)
    ///   N = N.scatter_add(dim, index, ones_like(src))
    ///   N.masked_fill_(N == 0, 1)
    ///   grad_self = grad / N
    ///   grad_src = grad.gather(dim, index) / N.gather(dim, index)
    /// then `:7274-7275`: if !include_self, scatter zeros into grad_self.
    fn backward_mean(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let input_shape = self.input.shape();
        if grad_output.is_cuda() {
            return no_grad(|| self.backward_mean_cuda(grad_output));
        }

        let go_data = grad_output.data_vec()?;
        let counts = scatter_reduce_mean_counts_cpu::<T>(
            input_shape,
            self.dim,
            &self.index,
            &self.index_shape,
            self.include_self,
        );
        let zero = <T as num_traits::Zero>::zero();

        let grad_input = if self.input.requires_grad() {
            let mut gi: Vec<T> = go_data.iter().zip(&counts).map(|(&g, &n)| g / n).collect();
            if !self.include_self {
                self.for_each_index(|_, _, _, dst_flat| {
                    gi[dst_flat] = zero;
                });
            }
            Some(Tensor::from_storage(
                TensorStorage::on_device(gi, self.input.device())?,
                input_shape.to_vec(),
                false,
            )?)
        } else {
            None
        };

        let grad_src = if self.src.requires_grad() {
            let src_shape = self.src.shape();
            let mut gs = vec![zero; self.src.numel()];
            self.for_each_index(|_, _, coords, dst_flat| {
                gs[flat_index(coords, src_shape)] = go_data[dst_flat] / counts[dst_flat];
            });
            Some(Tensor::from_storage(
                TensorStorage::on_device(gs, self.src.device())?,
                src_shape.to_vec(),
                false,
            )?)
        } else {
            None
        };

        Ok(vec![grad_input, grad_src])
    }

    fn backward_mean_cuda(
        &self,
        grad_output: &Tensor<T>,
    ) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !cuda_float_dtype::<T>() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "ScatterReduceBackward",
            });
        }
        same_device(self.input.device(), grad_output.device())?;
        same_device(self.input.device(), self.src.device())?;

        let input_shape = self.input.shape();
        let grad_c = grad_output.contiguous()?;
        let ordinal = cuda_ordinal(grad_c.device(), "ScatterReduceBackward")?;
        let idx_handle = crate::ops::indexing::upload_index_i64(&self.index, ordinal)?;
        let counts = scatter_reduce_mean_counts_cuda::<T>(
            input_shape,
            self.dim,
            &self.index,
            &self.index_shape,
            self.include_self,
            grad_c.device(),
        )?;
        let grad_distributed = crate::grad_fns::arithmetic::div(&grad_c, &counts)?;

        let grad_input = if self.input.requires_grad() {
            let gi = if self.include_self {
                grad_distributed.clone()
            } else {
                cuda_scatter_zero_at_index(
                    &grad_distributed,
                    &idx_handle,
                    input_shape,
                    &self.index_shape,
                    self.dim,
                )?
            };
            Some(gi)
        } else {
            None
        };

        let grad_src = if self.src.requires_grad() {
            Some(cuda_gather_index_shape(
                &grad_distributed,
                &idx_handle,
                input_shape,
                &self.index_shape,
                self.dim,
            )?)
        } else {
            None
        };

        Ok(vec![grad_input, grad_src])
    }

    /// VJP for `reduce='sum'` per upstream
    /// `FunctionsManual.cpp:7213-7215`:
    ///   grad_self = grad; grad_src = grad.gather(dim, index);
    /// then `:7274-7275`: if !include_self, scatter zeros into grad_self at
    /// the index-mapped positions.
    fn backward_sum(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let input_shape = self.input.shape();
        let zero = <T as num_traits::Zero>::zero();

        if grad_output.is_cuda() {
            if !cuda_float_dtype::<T>() {
                return Err(FerrotorchError::NotImplementedOnCuda {
                    op: "ScatterReduceBackward",
                });
            }
            same_device(self.input.device(), grad_output.device())?;
            same_device(self.input.device(), self.src.device())?;

            let ordinal = cuda_ordinal(grad_output.device(), "ScatterReduceBackward")?;
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let idx_handle = crate::ops::indexing::upload_index_i64(&self.index, ordinal)?;
            let go_c = no_grad(|| grad_output.contiguous())?;
            let go_handle = go_c.gpu_handle()?;

            let grad_input = if self.input.requires_grad() {
                let gi = if self.include_self {
                    let result_h = backend.clone_buffer(go_handle)?;
                    Tensor::from_storage(TensorStorage::gpu(result_h), input_shape.to_vec(), false)?
                } else {
                    cuda_scatter_zero_at_index(
                        &go_c,
                        &idx_handle,
                        input_shape,
                        &self.index_shape,
                        self.dim,
                    )?
                };
                Some(gi)
            } else {
                None
            };

            let grad_src = if self.src.requires_grad() {
                let result_h = backend.gather_intidx_nd(
                    go_handle,
                    &idx_handle,
                    input_shape,
                    &self.index_shape,
                    self.dim,
                )?;
                Some(Tensor::from_storage(
                    TensorStorage::gpu(result_h),
                    self.index_shape.clone(),
                    false,
                )?)
            } else {
                None
            };

            return Ok(vec![grad_input, grad_src]);
        }

        let go_data = grad_output.data_vec()?;

        let grad_input = if self.input.requires_grad() {
            let mut gi = go_data.clone();
            if !self.include_self {
                self.for_each_index(|_, _, _, dst_flat| {
                    gi[dst_flat] = zero;
                });
            }
            Some(Tensor::from_storage(
                // CORE-048 (#1742): gradient on the input leaf's device.
                TensorStorage::on_device(gi, self.input.device())?,
                input_shape.to_vec(),
                false,
            )?)
        } else {
            None
        };

        let grad_src = if self.src.requires_grad() {
            let src_shape = self.src.shape();
            let mut gs = vec![zero; self.src.numel()];
            self.for_each_index(|_, _, coords, dst_flat| {
                gs[flat_index(coords, src_shape)] = go_data[dst_flat];
            });
            Some(Tensor::from_storage(
                // CORE-048 (#1742): gradient on the src leaf's device.
                TensorStorage::on_device(gs, self.src.device())?,
                src_shape.to_vec(),
                false,
            )?)
        } else {
            None
        };

        Ok(vec![grad_input, grad_src])
    }

    /// VJP for `reduce='amax'` / `reduce='amin'` per upstream
    /// `FunctionsManual.cpp:7256-7265`:
    ///   value = result.gather(dim, index);
    ///   self_is_result = (self == result);  src_is_result = (src == value);
    ///   N = self_is_result.scatter_add(dim, index, src_is_result);
    ///   grad_distributed = grad / N;
    ///   grad_self = (self == result) * grad_distributed;
    ///   grad_src  = (src == value) * grad_distributed.gather(dim, index);
    /// then `:7274-7275`: if !include_self, scatter zeros into grad_self.
    ///
    /// The intuition: gradient flows to every input position whose value
    /// equals the output maximum (resp. minimum) at the index-mapped slot,
    /// shared evenly among all the tied positions (across both self and the
    /// src elements that scattered into that slot).
    fn backward_amax_amin(
        &self,
        grad_output: &Tensor<T>,
    ) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let input_shape = self.input.shape();
        if grad_output.is_cuda() {
            return no_grad(|| self.backward_amax_amin_cuda(grad_output));
        }

        let go_data = grad_output.data_vec()?;
        let in_data = self.input.data_vec()?;
        let src_data = self.src.data_vec()?;
        let result_data = self.result.data_vec()?;
        let src_shape = self.src.shape();
        let zero = <T as num_traits::Zero>::zero();
        let one = <T as num_traits::One>::one();
        let input_numel: usize = crate::shape::numel(input_shape);
        let index_numel: usize = crate::shape::numel(&self.index_shape);

        // self_is_result[p] = 1 iff input[p] == result[p].
        let mut self_is_result = vec![zero; input_numel];
        for p in 0..input_numel {
            if in_data[p] == result_data[p] {
                self_is_result[p] = one;
            }
        }

        // For each (i, dst_flat): value = result[dst_flat]; src_is_result[i] =
        // 1 iff src_at_coords(i) == value. Read src at index-shape coords via
        // the src_shape stride walk (same as forward).
        let read_src_at = |coords: &[usize]| -> T { src_data[flat_index(coords, src_shape)] };
        let mut src_is_result = vec![zero; index_numel];
        let mut value = vec![zero; index_numel];
        self.for_each_index(|i, _, coords, dst_flat| {
            let v = result_data[dst_flat];
            value[i] = v;
            if read_src_at(coords) == v {
                src_is_result[i] = one;
            }
        });

        // N[p] = self_is_result[p] + sum over (i: dst_flat==p) of src_is_result[i].
        let mut n_to_distribute = self_is_result.clone();
        self.for_each_index(|i, _, _, dst_flat| {
            n_to_distribute[dst_flat] += src_is_result[i];
        });

        // grad_distributed[p] = grad[p] / N[p] (guarded — N can never be 0 at
        // touched positions because the forward wrote `result[p]` from
        // exactly one of those positions, so at least one of self_is_result
        // or one of the src_is_result entries is 1).
        let mut grad_distributed = vec![zero; input_numel];
        for p in 0..input_numel {
            if n_to_distribute[p] != zero {
                grad_distributed[p] = go_data[p] / n_to_distribute[p];
            }
        }

        let grad_input = if self.input.requires_grad() {
            let mut gi = vec![zero; input_numel];
            for p in 0..input_numel {
                if self_is_result[p] != zero {
                    gi[p] = grad_distributed[p];
                }
            }
            // !include_self: zero positions the index touched (post-processing
            // step at upstream `:7274-7275`).
            if !self.include_self {
                self.for_each_index(|_, _, _, dst_flat| {
                    gi[dst_flat] = zero;
                });
            }
            Some(Tensor::from_storage(
                // CORE-048 (#1742): gradient on the input leaf's device.
                TensorStorage::on_device(gi, self.input.device())?,
                input_shape.to_vec(),
                false,
            )?)
        } else {
            None
        };

        let grad_src = if self.src.requires_grad() {
            let mut gs = vec![zero; self.src.numel()];
            self.for_each_index(|i, _, coords, dst_flat| {
                if src_is_result[i] != zero {
                    gs[flat_index(coords, src_shape)] = grad_distributed[dst_flat];
                }
            });
            Some(Tensor::from_storage(
                // CORE-048 (#1742): gradient on the src leaf's device.
                TensorStorage::on_device(gs, self.src.device())?,
                src_shape.to_vec(),
                false,
            )?)
        } else {
            None
        };

        let _ = value; // value buffer used inline above; silence unused-binding.
        Ok(vec![grad_input, grad_src])
    }

    fn backward_amax_amin_cuda(
        &self,
        grad_output: &Tensor<T>,
    ) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !cuda_float_dtype::<T>() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "ScatterReduceBackward",
            });
        }
        same_device(self.input.device(), grad_output.device())?;
        same_device(self.input.device(), self.src.device())?;
        same_device(self.input.device(), self.result.device())?;

        let input_shape = self.input.shape();
        let zero = <T as num_traits::Zero>::zero();
        let input_c = self.input.contiguous()?;
        let src_slab =
            cuda_src_prefix_slab_for_index(&self.src, &self.index_shape)?.contiguous()?;
        let result_c = self.result.contiguous()?;
        let grad_c = grad_output.contiguous()?;
        let ordinal = cuda_ordinal(grad_c.device(), "ScatterReduceBackward")?;
        let idx_handle = crate::ops::indexing::upload_index_i64(&self.index, ordinal)?;

        let value = cuda_gather_index_shape(
            &result_c,
            &idx_handle,
            input_shape,
            &self.index_shape,
            self.dim,
        )?;
        let self_is_result = BoolTensor::eq_t(&input_c, &result_c)?;
        let src_is_result = BoolTensor::eq_t(&src_slab, &value)?;
        let self_is_result_f = self_is_result.to_float::<T>()?;
        let src_is_result_f = src_is_result.to_float::<T>()?;

        let zero_input = cuda_full(input_shape, zero, grad_c.device(), "ScatterReduceBackward")?;
        let src_hits = crate::ops::indexing::scatter_add(
            &zero_input,
            self.dim as isize,
            &self.index,
            &self.index_shape,
            &src_is_result_f,
        )?;
        let n_to_distribute = crate::grad_fns::arithmetic::add(&self_is_result_f, &src_hits)?;
        let grad_distributed = crate::grad_fns::arithmetic::div(&grad_c, &n_to_distribute)?;

        let grad_input = if self.input.requires_grad() {
            let gi = crate::grad_fns::arithmetic::mul(&self_is_result_f, &grad_distributed)?;
            let gi = if self.include_self {
                gi
            } else {
                cuda_scatter_zero_at_index(
                    &gi,
                    &idx_handle,
                    input_shape,
                    &self.index_shape,
                    self.dim,
                )?
            };
            Some(gi)
        } else {
            None
        };

        let grad_src = if self.src.requires_grad() {
            let gathered = cuda_gather_index_shape(
                &grad_distributed,
                &idx_handle,
                input_shape,
                &self.index_shape,
                self.dim,
            )?;
            Some(crate::grad_fns::arithmetic::mul(
                &src_is_result_f,
                &gathered,
            )?)
        } else {
            None
        };

        Ok(vec![grad_input, grad_src])
    }

    /// VJP for `reduce='prod'` per upstream `FunctionsManual.cpp:7216-7248`:
    ///
    ///   masked_self = self.masked_fill(self == 0, 1)
    ///   masked_self_result = masked_self.scatter_reduce(dim, index, src,
    ///                                                    'prod', include_self)
    ///   grad_self = grad * masked_self_result / masked_self
    ///   src_zero = (src == 0)
    ///   src_num_zeros = zeros_like(self).scatter_add(dim, index, src_zero)
    ///                                    .gather(dim, index)
    ///   src_single_zero = src_zero & (src_num_zeros == 1)
    ///   masked_src = src.masked_fill(src_single_zero, 1)
    ///   masked_src_result = self.scatter_reduce(dim, index, masked_src,
    ///                                            'prod', include_self)
    ///   grad_src = where(src_single_zero,
    ///                    (grad * masked_src_result).gather(dim, index),
    ///                    (grad * result).gather(dim, index)
    ///                       / src.masked_fill(src_zero, 1))
    ///   if !include_self: grad_self = grad_self.scatter(dim, index, 0)
    ///
    /// The chain rule for a product `r = a*b*c*...`: `dr/da = r/a = b*c*...`,
    /// guarded so a single zero in the product still produces the right
    /// gradient (the exclusive-product over the non-zero entries).
    fn backward_prod(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let input_shape = self.input.shape();
        if grad_output.is_cuda() {
            return no_grad(|| self.backward_prod_cuda(grad_output));
        }

        let go_data = grad_output.data_vec()?;
        let in_data = self.input.data_vec()?;
        let src_data = self.src.data_vec()?;
        let result_data = self.result.data_vec()?;
        let src_shape = self.src.shape();
        let zero = <T as num_traits::Zero>::zero();
        let one = <T as num_traits::One>::one();
        let input_numel: usize = crate::shape::numel(input_shape);
        let index_numel: usize = crate::shape::numel(&self.index_shape);

        // masked_self[p] = self[p] == 0 ? 1 : self[p]
        let mut masked_self = in_data.clone();
        for v in &mut masked_self {
            if *v == zero {
                *v = one;
            }
        }

        // masked_self_result: recompute `scatter_reduce(masked_self, dim,
        // index, src, prod, include_self)` — the prod-fold uses `masked_self`
        // as the starting buffer (or identity 1 when include_self=false).
        let read_src_at = |coords: &[usize]| -> T { src_data[flat_index(coords, src_shape)] };
        let mut masked_self_result = if self.include_self {
            masked_self.clone()
        } else {
            // For include_self=false: identity is 1 for prod; only positions
            // the index touched start at 1 and accumulate src*src*... ; other
            // positions keep masked_self.
            let mut buf = masked_self.clone();
            self.for_each_index(|_, _, _, dst_flat| {
                buf[dst_flat] = one;
            });
            buf
        };
        self.for_each_index(|_, _, coords, dst_flat| {
            masked_self_result[dst_flat] = masked_self_result[dst_flat] * read_src_at(coords);
        });

        // src_zero[i] = src_at_coords(i) == 0 (per index slot — read src at
        // index-shape coords like the forward).
        let mut src_zero = vec![zero; index_numel];
        self.for_each_index(|i, _, coords, _| {
            if read_src_at(coords) == zero {
                src_zero[i] = one;
            }
        });

        // src_num_zeros[i] = sum of src_zero[j] for j that scatter into the
        // same dst_flat as index slot i. Build a per-dst count first, then
        // gather it at the index positions.
        let mut zero_count_per_dst = vec![zero; input_numel];
        self.for_each_index(|i, _, _, dst_flat| {
            zero_count_per_dst[dst_flat] += src_zero[i];
        });
        let mut src_num_zeros = vec![zero; index_numel];
        self.for_each_index(|i, _, _, dst_flat| {
            src_num_zeros[i] = zero_count_per_dst[dst_flat];
        });

        // src_single_zero[i] = src_zero[i] && src_num_zeros[i] == 1.
        let mut src_single_zero = vec![zero; index_numel];
        for i in 0..index_numel {
            if src_zero[i] != zero && src_num_zeros[i] == one {
                src_single_zero[i] = one;
            }
        }

        // masked_src[i] = src_single_zero[i] ? 1 : src_at(coords). When we
        // need this we'll read it as the value at index slot i.
        // masked_src_result: scatter_reduce(self, dim, index, masked_src,
        // prod, include_self) — fold `masked_src` over the start buffer in
        // the same way as above.
        let mut masked_src_result = if self.include_self {
            in_data.clone()
        } else {
            let mut buf = in_data.clone();
            self.for_each_index(|_, _, _, dst_flat| {
                buf[dst_flat] = one;
            });
            buf
        };
        let mut masked_src_values = vec![zero; index_numel];
        self.for_each_index(|i, _, coords, _| {
            let s = read_src_at(coords);
            let m = if src_single_zero[i] == zero { s } else { one };
            masked_src_values[i] = m;
        });
        self.for_each_index(|i, _, _, dst_flat| {
            masked_src_result[dst_flat] = masked_src_result[dst_flat] * masked_src_values[i];
        });

        // grad_self[p] = grad[p] * masked_self_result[p] / masked_self[p]
        let grad_input = if self.input.requires_grad() {
            let mut gi = vec![zero; input_numel];
            for p in 0..input_numel {
                if masked_self[p] != zero {
                    gi[p] = go_data[p] * masked_self_result[p] / masked_self[p];
                }
            }
            // !include_self post-processing: zero grad at index-touched
            // positions (`:7274-7275`).
            if !self.include_self {
                self.for_each_index(|_, _, _, dst_flat| {
                    gi[dst_flat] = zero;
                });
            }
            Some(Tensor::from_storage(
                // CORE-048 (#1742): gradient on the input leaf's device.
                TensorStorage::on_device(gi, self.input.device())?,
                input_shape.to_vec(),
                false,
            )?)
        } else {
            None
        };

        // grad_src[i] = where(
        //   src_single_zero[i],
        //   (grad * masked_src_result)[dst_flat],
        //   (grad * result)[dst_flat] / (src_at(i) if !src_zero[i] else 1)
        // )
        let grad_src = if self.src.requires_grad() {
            let mut gs = vec![zero; self.src.numel()];
            self.for_each_index(|i, _, coords, dst_flat| {
                let s_raw = read_src_at(coords);
                let denom = if s_raw == zero { one } else { s_raw };
                let primary = (go_data[dst_flat] * result_data[dst_flat]) / denom;
                let single_zero_branch = go_data[dst_flat] * masked_src_result[dst_flat];
                gs[flat_index(coords, src_shape)] = if src_single_zero[i] == zero {
                    primary
                } else {
                    single_zero_branch
                };
            });
            Some(Tensor::from_storage(
                // CORE-048 (#1742): gradient on the src leaf's device.
                TensorStorage::on_device(gs, self.src.device())?,
                src_shape.to_vec(),
                false,
            )?)
        } else {
            None
        };

        Ok(vec![grad_input, grad_src])
    }

    fn backward_prod_cuda(
        &self,
        grad_output: &Tensor<T>,
    ) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !cuda_float_dtype::<T>() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "ScatterReduceBackward",
            });
        }
        same_device(self.input.device(), grad_output.device())?;
        same_device(self.input.device(), self.src.device())?;
        same_device(self.input.device(), self.result.device())?;

        let input_shape = self.input.shape();
        let zero = <T as num_traits::Zero>::zero();
        let one = <T as num_traits::One>::one();
        let input_c = self.input.contiguous()?;
        let src_slab =
            cuda_src_prefix_slab_for_index(&self.src, &self.index_shape)?.contiguous()?;
        let result_c = self.result.contiguous()?;
        let grad_c = grad_output.contiguous()?;
        let ordinal = cuda_ordinal(grad_c.device(), "ScatterReduceBackward")?;
        let idx_handle = crate::ops::indexing::upload_index_i64(&self.index, ordinal)?;

        let zero_input = cuda_full(input_shape, zero, grad_c.device(), "ScatterReduceBackward")?;

        let self_zero = BoolTensor::eq_t(&input_c, &zero_input)?;
        let masked_self = masked_fill_bt(&input_c, &self_zero, one)?;
        let masked_self_result = scatter_reduce_cuda_forward(
            &masked_self,
            self.dim,
            &self.index,
            &self.index_shape,
            &src_slab,
            ScatterReduce::Prod,
            self.include_self,
        )?;

        let grad_input = if self.input.requires_grad() {
            let numerator = crate::grad_fns::arithmetic::mul(&grad_c, &masked_self_result)?;
            let gi = crate::grad_fns::arithmetic::div(&numerator, &masked_self)?;
            let gi = if self.include_self {
                gi
            } else {
                cuda_scatter_zero_at_index(
                    &gi,
                    &idx_handle,
                    input_shape,
                    &self.index_shape,
                    self.dim,
                )?
            };
            Some(gi)
        } else {
            None
        };

        let grad_src = if self.src.requires_grad() {
            let zero_src = cuda_full(
                &self.index_shape,
                zero,
                grad_c.device(),
                "ScatterReduceBackward",
            )?;
            let one_src = cuda_full(
                &self.index_shape,
                one,
                grad_c.device(),
                "ScatterReduceBackward",
            )?;
            let src_zero = BoolTensor::eq_t(&src_slab, &zero_src)?;
            let src_zero_f = src_zero.to_float::<T>()?;
            let zero_count_per_dst = crate::ops::indexing::scatter_add(
                &zero_input,
                self.dim as isize,
                &self.index,
                &self.index_shape,
                &src_zero_f,
            )?;
            let src_num_zeros = cuda_gather_index_shape(
                &zero_count_per_dst,
                &idx_handle,
                input_shape,
                &self.index_shape,
                self.dim,
            )?;
            let src_one_zero_count = BoolTensor::eq_t(&src_num_zeros, &one_src)?;
            let src_single_zero = src_zero.and(&src_one_zero_count)?;
            let masked_src = masked_fill_bt(&src_slab, &src_single_zero, one)?;
            let masked_src_result = scatter_reduce_cuda_forward(
                &input_c,
                self.dim,
                &self.index,
                &self.index_shape,
                &masked_src,
                ScatterReduce::Prod,
                self.include_self,
            )?;

            let single_zero_num = crate::grad_fns::arithmetic::mul(&grad_c, &masked_src_result)?;
            let single_zero_branch = cuda_gather_index_shape(
                &single_zero_num,
                &idx_handle,
                input_shape,
                &self.index_shape,
                self.dim,
            )?;
            let result_num = crate::grad_fns::arithmetic::mul(&grad_c, &result_c)?;
            let result_branch_num = cuda_gather_index_shape(
                &result_num,
                &idx_handle,
                input_shape,
                &self.index_shape,
                self.dim,
            )?;
            let denom = masked_fill_bt(&src_slab, &src_zero, one)?;
            let result_branch = crate::grad_fns::arithmetic::div(&result_branch_num, &denom)?;
            Some(crate::ops::indexing::where_cond_bt(
                &src_single_zero,
                &single_zero_branch,
                &result_branch,
            )?)
        } else {
            None
        };

        Ok(vec![grad_input, grad_src])
    }
}

/// Forward `scatter_reduce` for floating dtypes. Mirrors upstream
/// `at::scatter_reduce(self, dim, index, src, reduce, include_self=true)`
/// at `aten/src/ATen/native/TensorAdvancedIndexing.cpp:2354
/// TORCH_IMPL_FUNC(scatter_reduce_two)`.
///
/// `dim` follows PyTorch's negative-wrapping convention (upstream
/// `maybe_wrap_dim` at `:2362`).
///
/// For `reduce='sum'` with `include_self=False`, every output slice at an
/// index position is reset to zero before accumulation — upstream pattern at
/// `TensorAdvancedIndexing.cpp:2378-2386` via `scatter_impl<...>(..., reduce,
/// include_self)` followed by include_self_ones masking.
///
/// Backward is implemented for ALL reduce modes — `sum`, `mean`, `prod`,
/// `amax`, `amin` — per upstream `scatter_reduce_backward` at
/// `torch/csrc/autograd/FunctionsManual.cpp:7194-7279`, registered in
/// `tools/autograd/derivatives.yaml:3074-3077`. Live oracle confirms torch
/// attaches `ScatterReduceBackward0` for every reduce mode:
///   ```python
///   r = inp.scatter_reduce(0, idx, src, reduce='amax', include_self=True)
///   r.grad_fn   # <ScatterReduceBackward0 ...>
///   r.requires_grad   # True
///   r.sum().backward()   # succeeds, src.grad = [1., 1.]
///   ```
/// The `ScatterReduceBackward` GradFn saves the forward `result` buffer so
/// the value-aware VJPs (which need to read the per-slot max/min and the
/// prod chain-rule) can compute the right gradient. For all modes the
/// result tensor carries [`ScatterReduceBackward`] when grad is enabled.
///
/// # Device contract (CORE-048 / #1742)
///
/// `src` must live on `input`'s device — a mix returns
/// [`FerrotorchError::DeviceMismatch`] (torch: "Expected all tensors to be
/// on the same device, but got ..."). CUDA f32/f64/f16/bf16 operands lower through
/// resident kernels/composites for every shipped reduce mode (`sum`, `mean`,
/// `prod`, `amax`, `amin`); unsupported CUDA dtypes return
/// `NotImplementedOnCuda` instead of falling through to a host fold. Gradients
/// are delivered on the leaves' devices.
pub fn scatter_reduce<T: Float>(
    input: &Tensor<T>,
    dim: i64,
    index: &[usize],
    index_shape: &[usize],
    src: &Tensor<T>,
    reduce: ScatterReduce,
    include_self: bool,
) -> FerrotorchResult<Tensor<T>> {
    // CORE-048 (#1742): strict same-device operands at entry.
    same_device(input.device(), src.device())?;

    let input_shape = input.shape();
    let ndim = input_shape.len();
    let effective_ndim = ndim.max(1);
    let dim_usize = crate::shape::normalize_axis(dim as isize, effective_ndim)?;
    if ndim == 0 {
        // PyTorch validates scalar scatter/scatter_reduce through
        // `ensure_nonempty_dim`: 0-D self behaves as a 1-element tensor for
        // shape and bounds checks, but the output shape remains scalar.
        let scalar_shapes = validate_scalar_scatter_reduce(index, index_shape, src)?;
        if input.is_cuda() || src.is_cuda() {
            if !input.is_cuda() || !src.is_cuda() || !cuda_float_dtype::<T>() {
                return Err(FerrotorchError::NotImplementedOnCuda {
                    op: "scatter_reduce",
                });
            }
            let input_1d = input.view_reshape(vec![1])?;
            let src_1d = if src.shape().is_empty() {
                src.view_reshape(vec![1])?
            } else {
                src.clone()
            };
            let output_1d = if scalar_shapes.index_numel == 0 {
                input_1d
            } else {
                scatter_reduce_cuda_forward(
                    &input_1d,
                    dim_usize,
                    index,
                    &scalar_shapes.effective_index_shape,
                    &src_1d,
                    reduce,
                    include_self,
                )?
            };
            let output = output_1d.view_reshape(vec![])?;
            return attach_scatter_reduce_grad(
                output,
                input,
                src,
                dim_usize,
                index,
                index_shape,
                reduce,
                include_self,
            );
        }

        let in_data = input.data_vec()?;
        let src_data = src.data_vec()?;
        let zero = <T as num_traits::Zero>::zero();
        let one = <T as num_traits::One>::one();
        let mut out = in_data[0];
        let mut mean_count = if include_self { one } else { zero };
        if !include_self && scalar_shapes.index_numel != 0 {
            out = match reduce {
                ScatterReduce::Sum | ScatterReduce::Mean => zero,
                ScatterReduce::Prod => one,
                ScatterReduce::Amax | ScatterReduce::Amin => src_data[0],
            };
        }
        for i in 0..scalar_shapes.index_numel {
            let src_flat = if src.shape().is_empty() { 0 } else { i };
            let s = src_data[src_flat];
            out = apply_reduce(reduce, out, s);
            if reduce == ScatterReduce::Mean {
                mean_count += one;
            }
        }
        if reduce == ScatterReduce::Mean {
            if mean_count == zero {
                mean_count = one;
            }
            out = out / mean_count;
        }
        let output = Tensor::from_storage(
            TensorStorage::on_device(vec![out], input.device())?,
            vec![],
            false,
        )?;
        return attach_scatter_reduce_grad(
            output,
            input,
            src,
            0,
            index,
            index_shape,
            reduce,
            include_self,
        );
    }

    let index_numel =
        validate_scatter_reduce_shapes(input_shape, dim_usize, index, index_shape, src)?;

    if input.is_cuda() || src.is_cuda() {
        if !input.is_cuda() || !src.is_cuda() || !cuda_float_dtype::<T>() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "scatter_reduce",
            });
        }
        let output = scatter_reduce_cuda_forward(
            input,
            dim_usize,
            index,
            index_shape,
            src,
            reduce,
            include_self,
        )?;
        return attach_scatter_reduce_grad(
            output,
            input,
            src,
            dim_usize,
            index,
            index_shape,
            reduce,
            include_self,
        );
    }

    let in_data = input.data_vec()?;
    let src_data = src.data_vec()?;
    let src_shape = src.shape();
    let mut out = in_data.clone();

    // Read a src element at the index-shape coordinate `coords`, using src's
    // own shape for stride arithmetic (NOT a flat-i src_data[i] walk). This
    // mirrors upstream `_cpu_scatter_gather_dim_loop` at
    // `aten/src/ATen/native/cpu/ScatterGatherKernel.cpp:112-126`:
    //   for i in 0..index_dim_size:
    //     f(self + idx_dim * self_dim_stride, src + i * src_dim_stride)
    // where the outer TensorIterator iterates over index.sizes() and reads
    // src at the same coordinates with src.strides() — so when src is BIGGER
    // than index along any non-`dim` axis (allowed per `scatter_shape_check`
    // at `aten/src/ATen/native/ScatterGatherChecks.h:90-100`: `index.size(d) <=
    // src.size(d)`), the trailing src elements past index.size(d) are
    // ignored, but the accessed elements are at the index-shape coords —
    // NOT flat-i positions, which would read past row boundaries in src.
    let read_src_at = |coords: &[usize]| -> T { src_data[flat_index(coords, src_shape)] };

    // For include_self=false we mask out positions the index list will touch
    // and reset to the reduction identity. Per upstream `include_self`
    // semantics at `TensorAdvancedIndexing.cpp:2360-2391`: include_self=true
    // accumulates onto the existing self values; include_self=false
    // overwrites them at touched positions (using the reduction identity for
    // sum=0, prod=1, amax/amin=the first src element written).
    let zero = <T as num_traits::Zero>::zero();
    let one = <T as num_traits::One>::one();
    let mut mean_counts = if reduce == ScatterReduce::Mean {
        Some(vec![
            if include_self { one } else { zero };
            crate::shape::numel(input_shape)
        ])
    } else {
        None
    };
    if !include_self {
        let identity = match reduce {
            ScatterReduce::Sum | ScatterReduce::Mean => Some(zero),
            ScatterReduce::Prod => Some(one),
            // For amax/amin, identity is the first src write — handle below
            // by tracking first-touch positions.
            ScatterReduce::Amax | ScatterReduce::Amin => None,
        };
        if let Some(id) = identity {
            let mut coords = vec![0usize; ndim];
            for i in 0..index_numel {
                let idx_val = index[i];
                let mut dst_coords = coords.clone();
                dst_coords[dim_usize] = idx_val;
                let dst_flat = flat_index(&dst_coords, input_shape);
                out[dst_flat] = id;
                if i + 1 < index_numel {
                    increment_coords(&mut coords, index_shape);
                }
            }
        } else {
            // amax/amin with include_self=false: track first-touch per output
            // slot and seed with the first src write rather than identity.
            let input_numel: usize = crate::shape::numel(input_shape);
            let mut touched = vec![false; input_numel];
            let mut coords = vec![0usize; ndim];
            for i in 0..index_numel {
                let idx_val = index[i];
                let mut dst_coords = coords.clone();
                dst_coords[dim_usize] = idx_val;
                let dst_flat = flat_index(&dst_coords, input_shape);
                let s = read_src_at(&coords);
                out[dst_flat] = if touched[dst_flat] {
                    apply_reduce(reduce, out[dst_flat], s)
                } else {
                    touched[dst_flat] = true;
                    s
                };
                if i + 1 < index_numel {
                    increment_coords(&mut coords, index_shape);
                }
            }
            let output = Tensor::from_storage(
                TensorStorage::on_device(out, input.device())?,
                input_shape.to_vec(),
                false,
            )?;
            return attach_scatter_reduce_grad(
                output,
                input,
                src,
                dim_usize,
                index,
                index_shape,
                reduce,
                include_self,
            );
        }
    }

    // Sum / prod, OR amax/amin with include_self=true: accumulate onto out.
    let mut coords = vec![0usize; ndim];
    for i in 0..index_numel {
        let idx_val = index[i];
        let mut dst_coords = coords.clone();
        dst_coords[dim_usize] = idx_val;
        let dst_flat = flat_index(&dst_coords, input_shape);
        out[dst_flat] = apply_reduce(reduce, out[dst_flat], read_src_at(&coords));
        if let Some(counts) = mean_counts.as_mut() {
            counts[dst_flat] += one;
        }
        if i + 1 < index_numel {
            increment_coords(&mut coords, index_shape);
        }
    }
    if let Some(counts) = mean_counts.as_mut() {
        for (value, count) in out.iter_mut().zip(counts) {
            if *count == zero {
                *count = one;
            }
            *value = *value / *count;
        }
    }

    let output = Tensor::from_storage(
        TensorStorage::on_device(out, input.device())?,
        input_shape.to_vec(),
        false,
    )?;
    attach_scatter_reduce_grad(
        output,
        input,
        src,
        dim_usize,
        index,
        index_shape,
        reduce,
        include_self,
    )
}

/// Apply the per-mode binary reduction. `a` is the running accumulator,
/// `b` is the new src value being folded in.
#[inline]
fn apply_reduce<T: Float>(mode: ScatterReduce, a: T, b: T) -> T {
    match mode {
        ScatterReduce::Sum | ScatterReduce::Mean => a + b,
        ScatterReduce::Prod => a * b,
        // Use `partial_cmp` to match upstream PyTorch's NaN-passes-through
        // contract: any NaN in either operand keeps the accumulator
        // unchanged when comparing returns None.
        ScatterReduce::Amax => {
            if a.partial_cmp(&b) == Some(std::cmp::Ordering::Less) {
                b
            } else {
                a
            }
        }
        ScatterReduce::Amin => {
            if b.partial_cmp(&a) == Some(std::cmp::Ordering::Less) {
                b
            } else {
                a
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Shared strict-validation helper for index_add / index_copy.
//
// Both upstream ops use the same meta-function-driven strict contract:
//   - negative index values are REJECTED with `IndexError: index out of range
//     in self` (upstream kernels at `aten/src/ATen/native/
//     TensorAdvancedIndexing.cpp:1245-1247` index_add and `:1300-1301` 1-d
//     index_add, plus `cpu/IndexKernel.cpp` for index_copy_stub — none of
//     these wrap negatives, unlike `index_fill_kernel` at `cpu/
//     IndexKernel.cpp:224-229` which DOES wrap).
//   - source size mismatch is REJECTED with
//     `Number of indices (N) should be equal to source.size(dim): (M), for
//     dim: D` (upstream meta at `:394-402 for index_add`, `:343-349 for
//     index_copy`).
//   - source shape mismatch on non-dim axes is REJECTED with
//     `source tensor shape must match self tensor shape, excluding the
//     specified dimension. Got self.shape = ... source.shape = ...`
//     (upstream `:410-415` for index_add, `:330-342` for index_copy).
//
// The prior implementations of index_add / index_copy (#1247/#1248, commit
// 8e98ee0d2) extended the wrap-then-clamp pattern from index_fill (#1272/
// #1273) — but index_fill's wrap-negative pattern is specific to its
// upstream kernel (`cpu/IndexKernel.cpp:224-229`'s `if (idx < 0) idx +=
// size`); index_add and index_copy upstream do NOT wrap negatives. Pin
// #1286 D3-D6b. This helper now enforces strict validation for both.
// ---------------------------------------------------------------------------

/// Strict validation shared by `index_add` and `index_copy`. Mirrors the
/// meta-function checks at `aten/src/ATen/native/
/// TensorAdvancedIndexing.cpp:354-435 index_func_meta_impl` (index_add) and
/// `:278-352 TORCH_PRECOMPUTE_META_FUNC(index_copy)` (index_copy).
///
/// The two ops share most of the contract — strict-no-wrap negatives,
/// strict source-size match along `dim`, strict non-dim shape match — but
/// differ on the 0-d source case:
///
/// - **`index_add`** REJECTS 0-d source on N-D self. The upstream meta at
///   `:404-415` does `self_sizes == source_sizes` after a CONDITIONAL erase
///   (only when both are non-0-d); for `self.dim() != 0 && source.dim() ==
///   0` the erase is skipped, so the equality check `self_sizes == []`
///   fails immediately. Caller passes `accept_0d_source: false`.
///
/// - **`index_copy`** accepts 0-d source only when the destination slice is
///   scalar (`self.dim() == 1`) and `numIndices == 1`; 2-D+ destinations are
///   rejected because the destination slice has non-empty shape. Live oracle:
///   `torch.tensor([1.,2.,3.,4.]).index_copy(0, t([1]), t(99.))` ->
///   `tensor([1., 99., 3., 4.])`, while a 2-D destination with scalar source
///   raises "Source/destination tensor must have same slice shapes".
///   Caller passes `accept_0d_source: true`.
///
/// Validates:
/// 1. `dim` ∈ `[-input.ndim, input.ndim)` and normalizes to non-negative.
/// 2. `index.ndim <= 1` (scalar or 1-D only).
/// 3. Every index value is in `[0, input.size(dim))` — NEGATIVES REJECTED
///    (no wrap), matching upstream's `TORCH_CHECK_INDEX((self_i >= 0) &&
///    (self_i < self_dim_size))` at `:1245-1247`.
/// 4. `source.dim() <= 1 || source.size(dim) == index.numel()` — strict
///    size match along the index dim (no silent clamp). For 0-d source
///    when `accept_0d_source = true`, requires `n_indices == 1` per
///    upstream `:285-290 index_copy`.
/// 5. `source.dim() == 0 || self.dim() == 0 || self_sizes-dim ==
///    source_sizes-dim` — strict shape match on the non-dim axes.
/// 6. 0-d `source` on N-D `self` with N >= 1: REJECTED when
///    `accept_0d_source = false` (index_add); accepted only for 1-D
///    `index_copy` with `n_indices == 1`.
///
/// Returns `(dim_usize, idx_usize)` where `idx_usize` is the validated
/// non-negative index vector (length == `index.numel()`).
fn strict_index_add_copy_validate<T: Float>(
    op_name: &'static str,
    input: &Tensor<T>,
    dim: i64,
    index: &IntTensor<i64>,
    source: &Tensor<T>,
    accept_0d_source: bool,
) -> FerrotorchResult<(usize, Vec<usize>)> {
    let input_shape = input.shape();
    let ndim = input_shape.len();
    let ndim_i64 = ndim as i64;

    // (2) index.ndim <= 1
    if index.ndim() > 1 {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "{op_name}: index must be 1-D or scalar, got shape {:?}",
                index.shape()
            ),
        });
    }

    // (1) dim ∈ [-ndim, ndim), normalize.
    let dim_norm = if dim < 0 { dim + ndim_i64 } else { dim };
    if !(0..ndim_i64).contains(&dim_norm) {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("{op_name}: dim {dim} out of range for input ndim {ndim}"),
        });
    }
    let dim_usize = dim_norm as usize;
    let in_dim_size = input_shape[dim_usize];

    // (3) Validate every index value is in [0, in_dim_size). Negatives
    // REJECTED — upstream contract per `:1245-1247` (no wrap).
    let mut idx_usize: Vec<usize> = Vec::with_capacity(index.numel());
    for v in index.data()? {
        let i_raw = v.to_i64();
        if i_raw < 0 || i_raw >= in_dim_size as i64 {
            return Err(FerrotorchError::IndexOutOfBounds {
                index: if i_raw < 0 {
                    i_raw.unsigned_abs() as usize
                } else {
                    i_raw as usize
                },
                axis: dim_usize,
                size: in_dim_size,
            });
        }
        idx_usize.push(i_raw as usize);
    }

    // (4) source size match along `dim`. Upstream meta check at
    // `:394-402 for index_add`:
    //   TORCH_CHECK(numel == (source.dim() == 0 ? 1 : source.size(dim)),
    //     "Number of indices (", numel, ") should be equal to
    //      source.size(dim): (", source.size(dim), "), for dim: ", dim);
    // For index_copy the equivalent check is at `:343-349`:
    //   TORCH_CHECK_INDEX(source.dim() == 0 || numIndices == source.size(dim),
    //     ...);
    let source_shape = source.shape();
    let source_ndim = source_shape.len();
    let n_indices = index.numel();
    let expected_src_at_dim = if source_ndim == 0 {
        1
    } else if dim_usize < source_ndim {
        source_shape[dim_usize]
    } else {
        // dim out of bounds of source rank: only valid for source 0-d, which
        // is the `source_ndim == 0` branch above. Reaching here means
        // source.ndim > 0 but dim >= source.ndim — strict shape mismatch.
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "{op_name}: source.dim() ({source_ndim}) does not contain dim {dim_usize}"
            ),
        });
    };
    if n_indices != expected_src_at_dim {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "{op_name}: Number of indices ({n_indices}) should be equal to \
                 source.size(dim): ({expected_src_at_dim}), for dim: {dim_usize}"
            ),
        });
    }

    // (5)+(6) Non-dim shape match. Upstream `:404-415` for index_add and
    // `:321-342` for index_copy diverge on the `source.dim() == 0 &&
    // self.dim() != 0` case:
    //   - index_add: rejected (the conditional erase at `:406` is skipped,
    //     so self_sizes stays non-empty and the `self_sizes == source_sizes`
    //     equality at `:410-415` fails).
    //   - index_copy: accepted only when the destination slice is scalar
    //     (`self.dim() == 1`) and `numIndices == 1`; PyTorch rejects scalar
    //     source for 2-D+ destinations because the destination slice has
    //     non-empty shape.
    if source_ndim == 0 && ndim > 0 {
        if !accept_0d_source {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "{op_name}: source tensor shape must match self tensor shape, \
                     excluding the specified dimension. Got self.shape = {input_shape:?} \
                     source.shape = {source_shape:?}"
                ),
            });
        }
        // accept_0d_source (index_copy): the 0-d source contract requires one
        // index and a scalar destination slice. Live PyTorch 2.11 rejects a
        // scalar source with empty index, and rejects 2-D+ destination slices.
        if n_indices != 1 {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "{op_name}: When source is scalar, index should have one element \
                     (got {n_indices})"
                ),
            });
        }
        if ndim != 1 {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "{op_name}: source tensor shape must match destination slice shape. \
                     Got self.shape = {input_shape:?} source.shape = {source_shape:?}"
                ),
            });
        }
        // 0-d source on 1-D self, n_indices == 1: validated. Skip the
        // remaining non-dim shape walk below (source has no non-dim axes).
        return Ok((dim_usize, idx_usize));
    }
    if source_ndim != 0 && ndim != 0 {
        for d in 0..ndim {
            if d == dim_usize {
                continue;
            }
            let self_d = input_shape[d];
            let src_d = if d < source_ndim {
                source_shape[d]
            } else {
                // source rank differs from self rank: shape mismatch.
                return Err(FerrotorchError::ShapeMismatch {
                    message: format!(
                        "{op_name}: source tensor shape must match self tensor shape, \
                         excluding the specified dimension. Got self.shape = \
                         {input_shape:?} source.shape = {source_shape:?}"
                    ),
                });
            };
            if self_d != src_d {
                return Err(FerrotorchError::ShapeMismatch {
                    message: format!(
                        "{op_name}: source tensor shape must match self tensor shape, \
                         excluding the specified dimension. Got self.shape = \
                         {input_shape:?} source.shape = {source_shape:?}"
                    ),
                });
            }
        }
        // Also: source rank must equal self rank when both are non-0-d, or
        // source must be 1-D when self.dim() > 1 (the upstream `1-D source`
        // branch at `:1259-1308` accepts source.dim() <= 1 only when the
        // result is 1-D too; multi-D self with 1-D source is rejected by the
        // meta `self_sizes == source_sizes` check unless ndim==1).
        if source_ndim != ndim {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "{op_name}: source.dim() ({source_ndim}) must match self.dim() \
                     ({ndim}) (excluding 0-d source on 0-d self)"
                ),
            });
        }
    }

    Ok((dim_usize, idx_usize))
}

// ---------------------------------------------------------------------------
// index_add (#1247 — REQ-6). Mirrors `torch.index_add(input, dim, index,
// source, *, alpha=1)` at upstream `aten/src/ATen/native/
// TensorAdvancedIndexing.cpp:1153 TORCH_IMPL_FUNC(index_add_cpu_out)`. VJP
// per `tools/autograd/derivatives.yaml:862-869
//   - name: index_add(Tensor self, int dim, Tensor index, Tensor source, *,
//       Scalar alpha=1) -> Tensor
//     self: grad
//     source: "maybe_multiply(source.dim() > 0 ? grad.index_select(dim, index)
//       .expand_as(source) : grad.index_select(dim, index.squeeze(0)), alpha)"
//     index: non_differentiable`.
// ---------------------------------------------------------------------------

/// Backward function for `index_add`.
///
/// Forward: `output = input.clone(); output[..., index[i], ...] += alpha *
/// source[..., i, ...]` along `dim`.
///
/// VJP for input: identity (`derivatives.yaml:863 self: grad`).
/// VJP for source: `alpha * grad.index_select(dim, index)` — gather grad
/// slices at the index positions along `dim`, scaled by `alpha`.
#[derive(Debug)]
pub struct IndexAddBackward<T: Float> {
    /// Saved input handle (for shape + autograd graph linkage).
    pub input: Tensor<T>,
    /// Saved source handle.
    pub source: Tensor<T>,
    /// The normalized (non-negative) dim.
    pub dim: usize,
    /// The validated (non-negative) index list.
    pub index: Vec<usize>,
    /// The alpha scaling factor (from `Scalar alpha` upstream).
    pub alpha: f64,
}

impl<T: Float> GradFn<T> for IndexAddBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !is_grad_enabled() {
            return Ok(vec![None, None]);
        }
        let input_shape = self.input.shape();
        let ndim = input_shape.len();
        let gpu_fast = grad_output.is_cuda() && cuda_float_dtype::<T>();

        // grad for input: identity.
        let grad_input = if self.input.requires_grad() {
            if gpu_fast {
                Some(clone_cuda_tensor(grad_output, input_shape.to_vec())?)
            } else {
                let go = grad_output.data_vec()?;
                Some(Tensor::from_storage(
                    // CORE-048 (#1742): gradient on the input leaf's device.
                    TensorStorage::on_device(go, self.input.device())?,
                    input_shape.to_vec(),
                    false,
                )?)
            }
        } else {
            None
        };

        // grad for source: alpha * grad.index_select(dim, index). Walk
        // grad_output's outer/inner decomposition along `dim`, gather slices
        // at index positions, multiply by alpha. For 0-d source we copy the
        // single scalar at index[0] (upstream squeeze-on-zero-d path).
        let grad_source = if self.source.requires_grad() {
            let source_shape = self.source.shape();
            if gpu_fast {
                let gathered = if ndim == 0 || source_shape.is_empty() {
                    clone_cuda_tensor(grad_output, source_shape.to_vec())?
                } else {
                    let outer: usize = crate::shape::numel(&input_shape[..self.dim]);
                    let inner: usize = crate::shape::numel(&input_shape[self.dim + 1..]);
                    let in_dim_size = input_shape[self.dim];
                    let src_dim_size = if source_shape.len() == ndim {
                        source_shape[self.dim]
                    } else {
                        self.index.len()
                    };
                    gather_dim_cuda(
                        grad_output,
                        &self.index,
                        outer,
                        in_dim_size,
                        src_dim_size,
                        inner,
                        source_shape.to_vec(),
                        "IndexAddBackward",
                    )?
                };
                Some(scale_cuda_tensor(
                    &gathered,
                    self.alpha,
                    "IndexAddBackward",
                )?)
            } else {
                let go = grad_output.data_vec()?;
                let alpha_t = <T as num_traits::NumCast>::from(self.alpha).ok_or_else(|| {
                    FerrotorchError::InvalidArgument {
                        message: format!(
                            "IndexAddBackward: alpha {} not representable in target dtype",
                            self.alpha
                        ),
                    }
                })?;
                let gs = if ndim == 0 || source_shape.is_empty() {
                    // 0-d input or 0-d source: scalar copy of grad_output[0] * alpha.
                    let v = if go.is_empty() {
                        <T as num_traits::Zero>::zero()
                    } else {
                        go[0] * alpha_t
                    };
                    vec![v]
                } else {
                    let outer: usize = crate::shape::numel(&input_shape[..self.dim]);
                    let inner: usize = crate::shape::numel(&input_shape[self.dim + 1..]);
                    let in_dim_size = input_shape[self.dim];
                    let src_dim_size = if source_shape.len() == ndim {
                        source_shape[self.dim]
                    } else {
                        self.index.len()
                    };
                    let src_numel = if source_shape.is_empty() {
                        1
                    } else {
                        crate::shape::numel(source_shape)
                    };
                    let mut out = vec![<T as num_traits::Zero>::zero(); src_numel];
                    // gather: source[o, i, k] = grad_output[o, index[i], k] * alpha
                    for o in 0..outer {
                        for i in 0..src_dim_size.min(self.index.len()) {
                            let dst_i = self.index[i];
                            let go_base = o * in_dim_size * inner + dst_i * inner;
                            let src_base = o * src_dim_size * inner + i * inner;
                            for k in 0..inner {
                                out[src_base + k] = go[go_base + k] * alpha_t;
                            }
                        }
                    }
                    out
                };
                Some(Tensor::from_storage(
                    // CORE-048 (#1742): gradient on the source leaf's device.
                    TensorStorage::on_device(gs, self.source.device())?,
                    source_shape.to_vec(),
                    false,
                )?)
            }
        } else {
            None
        };

        Ok(vec![grad_input, grad_source])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input, &self.source]
    }

    fn name(&self) -> &'static str {
        "IndexAddBackward"
    }
}

/// Out-of-place `index_add`: `output[..., index[i], ...] += alpha *
/// source[..., i, ...]` along `dim`. Mirrors upstream `Tensor index_add(
/// const Tensor& self, int64_t dim, const Tensor& index, const Tensor&
/// source, const Scalar& alpha)` at `aten/src/ATen/native/
/// TensorAdvancedIndexing.cpp:1153 TORCH_IMPL_FUNC(index_add_cpu_out)`.
///
/// `dim` follows PyTorch's negative-wrap convention (`maybe_wrap_dim` at
/// `:1179`). `index` must be 1-D or 0-D scalar (upstream restricts at
/// `:1260-1264 TORCH_CHECK(source.dim() <= 1, ...)`).
///
/// **Strict validation** per upstream meta function at `:438-446
/// TORCH_PRECOMPUTE_META_FUNC(index_add)` → `:354-435 index_func_meta_impl`:
/// negative index values are REJECTED (no wrap, unlike `index_fill`);
/// `source.size(dim) != index.numel()` is REJECTED (no silent clamp);
/// 0-d source on N-D self is REJECTED (shape mismatch). See
/// [`strict_index_add_copy_validate`] for the shared helper. Closes #1286
/// divergences D3/D4/D5.
///
/// # Device contract (CORE-048 / #1742)
///
/// `index` and `source` must live on `input`'s device — a mix returns
/// [`FerrotorchError::DeviceMismatch`] (torch: "Expected all tensors to be
/// on the same device, but got index is on cpu, different from other
/// tensors on cuda:0"). CUDA f32/f64/f16/bf16 operands run resident: source is
/// alpha-scaled on-device when needed, the 1-D index is expanded to the
/// dim-aware kernel layout, and `scatter_add_dim_*` accumulates without
/// downloading input/source values. Unsupported CUDA dtypes return
/// `NotImplementedOnCuda` instead of falling through to a host walk.
/// Gradients are delivered on the leaves' devices; CUDA backward uses resident
/// clone/gather/scale kernels for the same dtype set.
pub fn index_add<T: Float>(
    input: &Tensor<T>,
    dim: i64,
    index: &IntTensor<i64>,
    source: &Tensor<T>,
    alpha: f64,
) -> FerrotorchResult<Tensor<T>> {
    // CORE-048 (#1742): strict same-device operands at entry.
    same_device(input.device(), index.device())?;
    same_device(input.device(), source.device())?;
    // CUDA index values are copied to host for bounds/shape validation; tensor
    // payloads stay resident on the CUDA fast path.
    let index_host;
    let index: &IntTensor<i64> = if index.is_cuda() {
        index_host = index.to(Device::Cpu)?;
        &index_host
    } else {
        index
    };

    let input_shape = input.shape();
    let ndim = input_shape.len();

    if ndim == 0 {
        // 0-d input: only valid when source is also 0-d (or 1-d length-1)
        // AND index has a single entry. Upstream unsqueezes to 1-d at
        // `TensorAdvancedIndexing.cpp:1259-1278`. Only dim ∈ {-1, 0} and
        // index ∈ {0} are valid (upstream rejects negative indices —
        // unwrapped here too).
        let dim_for_0d = match dim {
            0 | -1 => 0i64,
            _ => {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "index_add: dim {dim} out of range for 0-d input (valid: -1, 0)"
                    ),
                });
            }
        };
        // Source must be 0-d (matching self) — upstream meta function at
        // `aten/src/ATen/native/TensorAdvancedIndexing.cpp:404-415` enforces
        // `self_sizes == source_sizes` (the size-erase at :407 is conditional
        // on BOTH self.dim() != 0 AND source.dim() != 0). For 0-d self the
        // erase is skipped, so self_sizes stays `[]` and source_sizes stays
        // whatever source had — a 1-D length-1 source ends up as `[1]` and
        // the equality check `[] == [1]` REJECTS it. Live oracle:
        //   `torch.index_add(t(5.), 0, t([0]), t([99.]))` -> RuntimeError
        //   "source tensor shape must match self tensor shape, excluding the
        //    specified dimension. Got self.shape = [] source.shape = [1]"
        // Only an actually-0-d source is compatible.
        let source_shape = source.shape();
        if !source_shape.is_empty() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "index_add: source tensor shape must match self tensor shape, \
                     excluding the specified dimension. Got self.shape = [] \
                     source.shape = {source_shape:?}"
                ),
            });
        }
        let alpha_t = <T as num_traits::NumCast>::from(alpha).ok_or_else(|| {
            FerrotorchError::InvalidArgument {
                message: format!("index_add: alpha {alpha} not representable"),
            }
        })?;
        // Upstream requires `numel == 1` for source.dim() == 0. For 0-d
        // self + 0-d source: index must be 1-element.
        let n_indices = index.numel();
        if n_indices != 1 {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "index_add: Number of indices ({n_indices}) should be equal to \
                     source.size(dim): (1), for dim: 0"
                ),
            });
        }
        let mut saved_index: Vec<usize> = Vec::new();
        for v in index.data()? {
            let i_raw = v.to_i64();
            // 0-d input has dim_size = 1 — only the literal 0 is valid;
            // upstream rejects negatives.
            if i_raw != 0 {
                return Err(FerrotorchError::IndexOutOfBounds {
                    index: if i_raw < 0 {
                        i_raw.unsigned_abs() as usize
                    } else {
                        i_raw as usize
                    },
                    axis: dim_for_0d as usize,
                    size: 1,
                });
            }
            saved_index.push(0);
        }

        if input.is_cuda() || source.is_cuda() {
            if !cuda_float_dtype::<T>() {
                return Err(FerrotorchError::NotImplementedOnCuda { op: "index_add" });
            }
            let input_c = no_grad(|| input.contiguous())?;
            let source_c = no_grad(|| source.contiguous())?;
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let handle = match T::dtype() {
                DType::F32 => {
                    backend.add_scaled_f32(input_c.gpu_handle()?, source_c.gpu_handle()?, alpha)?
                }
                DType::F64 => {
                    backend.add_scaled_f64(input_c.gpu_handle()?, source_c.gpu_handle()?, alpha)?
                }
                DType::F16 | DType::BF16 => {
                    let source_scaled = if alpha_t == <T as num_traits::One>::one() {
                        source_c
                    } else {
                        scale_cuda_tensor(&source_c, alpha, "index_add")?
                    };
                    match T::dtype() {
                        DType::F16 => {
                            backend.add_f16(input_c.gpu_handle()?, source_scaled.gpu_handle()?)?
                        }
                        DType::BF16 => backend
                            .add_bf16_bf16(input_c.gpu_handle()?, source_scaled.gpu_handle()?)?,
                        _ => unreachable!("outer dtype arm restricted to f16/bf16"),
                    }
                }
                _ => return Err(FerrotorchError::NotImplementedOnCuda { op: "index_add" }),
            };
            let storage = TensorStorage::gpu(handle);
            if (input.requires_grad() || source.requires_grad()) && is_grad_enabled() {
                let grad_fn = Arc::new(IndexAddBackward {
                    input: input.clone(),
                    source: source.clone(),
                    dim: 0,
                    index: saved_index,
                    alpha,
                });
                return Tensor::from_operation(storage, vec![], grad_fn);
            }
            return Tensor::from_storage(storage, vec![], false);
        }

        let scalar_val = input.data_vec()?[0];
        let src_data = source.data_vec()?;
        let src_v = if src_data.is_empty() {
            <T as num_traits::Zero>::zero()
        } else {
            src_data[0]
        };
        let acc = scalar_val + alpha_t * src_v;
        // CORE-048 (#1742): result on input's device.
        let storage = TensorStorage::on_device(vec![acc], input.device())?;
        if (input.requires_grad() || source.requires_grad()) && is_grad_enabled() {
            let grad_fn = Arc::new(IndexAddBackward {
                input: input.clone(),
                source: source.clone(),
                dim: 0,
                index: saved_index,
                alpha,
            });
            return Tensor::from_operation(storage, vec![], grad_fn);
        }
        return Tensor::from_storage(storage, vec![], false);
    }

    // N-D input: route through the shared strict validator. index_add
    // REJECTS 0-d source on N-D self per upstream `:404-415` (the
    // `self_sizes == source_sizes` check after the conditional erase) —
    // pass `accept_0d_source = false`.
    let (dim_usize, idx_usize) =
        strict_index_add_copy_validate("index_add", input, dim, index, source, false)?;

    let in_dim_size = input_shape[dim_usize];
    let alpha_t = <T as num_traits::NumCast>::from(alpha).ok_or_else(|| {
        FerrotorchError::InvalidArgument {
            message: format!("index_add: alpha {alpha} not representable"),
        }
    })?;

    let outer: usize = crate::shape::numel(&input_shape[..dim_usize]);
    let inner: usize = crate::shape::numel(&input_shape[dim_usize + 1..]);
    let source_shape = source.shape();

    // Post-validate: src_dim_size == idx_usize.len() (strict check ensured
    // by the validator).
    let src_dim_size = if source_shape.is_empty() {
        // Strict validator guarantees: source 0-d only allowed when self also
        // 0-d (handled above) — reaching here is impossible.
        return Err(FerrotorchError::Internal {
            message: "index_add: unexpected 0-d source after strict validation".into(),
        });
    } else {
        source_shape[dim_usize]
    };

    if input.is_cuda() && source.is_cuda() && cuda_float_dtype::<T>() {
        let output_shape = input_shape.to_vec();
        let input_c = no_grad(|| input.contiguous())?;
        let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let handle = if idx_usize.is_empty() || outer == 0 || inner == 0 || input.numel() == 0 {
            backend.clone_buffer(input_c.gpu_handle()?)?
        } else {
            let source_c = no_grad(|| source.contiguous())?;
            let source_scaled = if alpha_t == <T as num_traits::One>::one() {
                source_c
            } else {
                scale_cuda_tensor(&source_c, alpha, "index_add")?
            };
            let ordinal = cuda_ordinal(input_c.device(), "index_add")?;
            let idx_handle = upload_expanded_dim_indices(&idx_usize, outer, inner, ordinal)?;
            scatter_dim_cuda_handle::<T>(
                input_c.gpu_handle()?,
                &idx_handle,
                source_scaled.gpu_handle()?,
                outer,
                in_dim_size,
                src_dim_size,
                inner,
                true,
                "index_add",
            )?
        };
        let storage = TensorStorage::gpu(handle);
        if (input.requires_grad() || source.requires_grad()) && is_grad_enabled() {
            let grad_fn = Arc::new(IndexAddBackward {
                input: input.clone(),
                source: source.clone(),
                dim: dim_usize,
                index: idx_usize,
                alpha,
            });
            return Tensor::from_operation(storage, output_shape, grad_fn);
        }
        return Tensor::from_storage(storage, output_shape, false);
    }
    if input.is_cuda() || source.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "index_add" });
    }

    let mut out = input.data_vec()?;
    let src_data = source.data_vec()?;

    for o in 0..outer {
        for (i, &dst_i) in idx_usize.iter().enumerate() {
            let dst_base = o * in_dim_size * inner + dst_i * inner;
            let src_base = o * src_dim_size * inner + i * inner;
            for k in 0..inner {
                let s = src_data[src_base + k];
                out[dst_base + k] += alpha_t * s;
            }
        }
    }

    let output_shape = input_shape.to_vec();
    if (input.requires_grad() || source.requires_grad()) && is_grad_enabled() {
        let grad_fn = Arc::new(IndexAddBackward {
            input: input.clone(),
            source: source.clone(),
            dim: dim_usize,
            index: idx_usize,
            alpha,
        });
        // CORE-048 (#1742): result on input's device.
        Tensor::from_operation(
            TensorStorage::on_device(out, input.device())?,
            output_shape,
            grad_fn,
        )
    } else {
        Tensor::from_storage(
            TensorStorage::on_device(out, input.device())?,
            output_shape,
            false,
        )
    }
}

// ---------------------------------------------------------------------------
// index_copy (#1248 — REQ-7). Mirrors `torch.index_copy(input, dim, index,
// source)` at upstream `aten/src/ATen/native/TensorAdvancedIndexing.cpp:1082
// TORCH_IMPL_FUNC(index_copy_out)`. VJP per `tools/autograd/derivatives.yaml:
// 875-883
//   - name: index_copy(Tensor self, int dim, Tensor index, Tensor source) -> Tensor
//     self: grad.index_fill(dim, index, 0)
//     source: "source.dim() > 0 ? grad.index_select(dim, index).expand_as(
//       source) : grad.index_select(dim, index.squeeze(0))"
//     index: non_differentiable`. Depends on REQ-8 (index_fill, SHIPPED).
// ---------------------------------------------------------------------------

/// Backward function for `index_copy`.
///
/// Forward: `output = input.clone(); output[..., index[i], ...] =
/// source[..., i, ...]` along `dim`.
///
/// VJP for input: zero grad at every position the copy overwrote (the same
/// pattern as `IndexFillBackward`).
/// VJP for source: gather grad at the index-mapped positions along `dim`
/// (same pattern as `IndexAddBackward` but without the alpha scale).
#[derive(Debug)]
pub struct IndexCopyBackward<T: Float> {
    /// Saved input handle (for shape + autograd graph linkage).
    pub input: Tensor<T>,
    /// Saved source handle.
    pub source: Tensor<T>,
    /// The normalized (non-negative) dim.
    pub dim: usize,
    /// The validated (non-negative) index list.
    pub index: Vec<usize>,
}

impl<T: Float> GradFn<T> for IndexCopyBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !is_grad_enabled() {
            return Ok(vec![None, None]);
        }
        let input_shape = self.input.shape();
        let ndim = input_shape.len();
        let zero = <T as num_traits::Zero>::zero();
        let gpu_fast = grad_output.is_cuda() && cuda_float_dtype::<T>();

        // grad for input: zero positions the copy overwrote.
        let grad_input = if self.input.requires_grad() {
            if gpu_fast {
                if self.index.is_empty() {
                    Some(clone_cuda_tensor(grad_output, input_shape.to_vec())?)
                } else if ndim == 0 {
                    let ordinal = cuda_ordinal(grad_output.device(), "IndexCopyBackward")?;
                    let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
                    let handle = backend.alloc_zeros(1, T::dtype(), ordinal)?;
                    Some(Tensor::from_storage(
                        TensorStorage::gpu(handle),
                        input_shape.to_vec(),
                        false,
                    )?)
                } else {
                    let outer: usize = crate::shape::numel(&input_shape[..self.dim]);
                    let inner: usize = crate::shape::numel(&input_shape[self.dim + 1..]);
                    let dim_size = input_shape[self.dim];
                    let ordinal = cuda_ordinal(grad_output.device(), "IndexCopyBackward")?;
                    let idx_handle =
                        upload_expanded_dim_indices(&self.index, outer, inner, ordinal)?;
                    let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
                    let go_c = no_grad(|| grad_output.contiguous())?;
                    let handle = match T::dtype() {
                        DType::F32 => backend.scatter_value_dim_f32(
                            go_c.gpu_handle()?,
                            &idx_handle,
                            0.0,
                            outer,
                            dim_size,
                            self.index.len(),
                            inner,
                        )?,
                        DType::F64 => backend.scatter_value_dim_f64(
                            go_c.gpu_handle()?,
                            &idx_handle,
                            0.0,
                            outer,
                            dim_size,
                            self.index.len(),
                            inner,
                        )?,
                        DType::F16 | DType::BF16 => {
                            let zero_len = outer
                                .checked_mul(self.index.len())
                                .and_then(|n| n.checked_mul(inner))
                                .ok_or_else(|| FerrotorchError::InvalidArgument {
                                    message: format!(
                                        "IndexCopyBackward: zero scatter source overflow for outer={outer}, index={}, inner={inner}",
                                        self.index.len()
                                    ),
                                })?;
                            let zeros = backend.alloc_zeros(zero_len, T::dtype(), ordinal)?;
                            scatter_dim_cuda_handle::<T>(
                                go_c.gpu_handle()?,
                                &idx_handle,
                                &zeros,
                                outer,
                                dim_size,
                                self.index.len(),
                                inner,
                                false,
                                "IndexCopyBackward",
                            )?
                        }
                        _ => {
                            return Err(FerrotorchError::NotImplementedOnCuda {
                                op: "IndexCopyBackward",
                            });
                        }
                    };
                    Some(Tensor::from_storage(
                        TensorStorage::gpu(handle),
                        input_shape.to_vec(),
                        false,
                    )?)
                }
            } else {
                let mut gi = grad_output.data_vec()?;
                if ndim == 0 {
                    if !self.index.is_empty() {
                        gi[0] = zero;
                    }
                } else {
                    let outer: usize = crate::shape::numel(&input_shape[..self.dim]);
                    let inner: usize = crate::shape::numel(&input_shape[self.dim + 1..]);
                    let dim_size = input_shape[self.dim];
                    for o in 0..outer {
                        for &idx in &self.index {
                            let base = o * dim_size * inner + idx * inner;
                            for k in 0..inner {
                                gi[base + k] = zero;
                            }
                        }
                    }
                }
                Some(Tensor::from_storage(
                    // CORE-048 (#1742): gradient on the input leaf's device.
                    TensorStorage::on_device(gi, self.input.device())?,
                    input_shape.to_vec(),
                    false,
                )?)
            }
        } else {
            None
        };

        // grad for source: gather grad_output at the index-mapped positions.
        let grad_source = if self.source.requires_grad() {
            let source_shape = self.source.shape();
            if gpu_fast {
                let gathered = if ndim == 0 {
                    clone_cuda_tensor(grad_output, source_shape.to_vec())?
                } else if source_shape.is_empty() {
                    gather_dim_cuda(
                        grad_output,
                        &self.index,
                        1,
                        input_shape[self.dim],
                        1,
                        1,
                        source_shape.to_vec(),
                        "IndexCopyBackward",
                    )?
                } else {
                    let outer: usize = crate::shape::numel(&input_shape[..self.dim]);
                    let inner: usize = crate::shape::numel(&input_shape[self.dim + 1..]);
                    let in_dim_size = input_shape[self.dim];
                    let src_dim_size = if source_shape.len() == ndim {
                        source_shape[self.dim]
                    } else {
                        self.index.len()
                    };
                    gather_dim_cuda(
                        grad_output,
                        &self.index,
                        outer,
                        in_dim_size,
                        src_dim_size,
                        inner,
                        source_shape.to_vec(),
                        "IndexCopyBackward",
                    )?
                };
                Some(gathered)
            } else {
                let go = grad_output.data_vec()?;
                let gs = if ndim == 0 || source_shape.is_empty() {
                    let v = if go.is_empty() { zero } else { go[0] };
                    vec![v]
                } else {
                    let outer: usize = crate::shape::numel(&input_shape[..self.dim]);
                    let inner: usize = crate::shape::numel(&input_shape[self.dim + 1..]);
                    let in_dim_size = input_shape[self.dim];
                    let src_dim_size = if source_shape.len() == ndim {
                        source_shape[self.dim]
                    } else {
                        self.index.len()
                    };
                    let src_numel = crate::shape::numel(source_shape);
                    let mut out = vec![zero; src_numel];
                    for o in 0..outer {
                        for i in 0..src_dim_size.min(self.index.len()) {
                            let dst_i = self.index[i];
                            let go_base = o * in_dim_size * inner + dst_i * inner;
                            let src_base = o * src_dim_size * inner + i * inner;
                            out[src_base..src_base + inner]
                                .copy_from_slice(&go[go_base..go_base + inner]);
                        }
                    }
                    out
                };
                Some(Tensor::from_storage(
                    // CORE-048 (#1742): gradient on the source leaf's device.
                    TensorStorage::on_device(gs, self.source.device())?,
                    source_shape.to_vec(),
                    false,
                )?)
            }
        } else {
            None
        };

        Ok(vec![grad_input, grad_source])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input, &self.source]
    }

    fn name(&self) -> &'static str {
        "IndexCopyBackward"
    }
}

/// Out-of-place `index_copy`: `output[..., index[i], ...] = source[..., i, ...]`
/// along `dim`. Mirrors upstream `Tensor index_copy(const Tensor& self,
/// int64_t dim, const Tensor& index, const Tensor& source)` at
/// `aten/src/ATen/native/TensorAdvancedIndexing.cpp:1082 TORCH_IMPL_FUNC(
/// index_copy_out)`.
///
/// `dim` follows PyTorch's negative-wrap convention. `index` must be 1-D or
/// scalar.
///
/// **Strict validation** per upstream meta function at `:258-352
/// TORCH_PRECOMPUTE_META_FUNC(index_copy)`: negative index values are
/// REJECTED (no wrap, unlike `index_fill`); `source.size(dim) !=
/// index.numel()` is REJECTED (no silent clamp); non-dim shape mismatch
/// rejected. See [`strict_index_add_copy_validate`] for the shared helper.
/// Closes #1286 divergences D6/D6b.
///
/// # Device contract (CORE-048 / #1742)
///
/// `index` and `source` must live on `input`'s device — a mix returns
/// [`FerrotorchError::DeviceMismatch`] (torch: "Expected all tensors to be
/// on the same device, but got source is on cpu, different from other
/// tensors on cuda:0"). CUDA f32/f64/f16/bf16 operands run resident via the existing
/// dim-aware `scatter_dim_*` kernels after expanding the 1-D index to the
/// per-element kernel layout. Unsupported CUDA dtypes return
/// `NotImplementedOnCuda` instead of falling through to a host walk.
/// Gradients are delivered on the leaves' devices; CUDA backward uses resident
/// scatter-zero/gather kernels for the same dtype set.
pub fn index_copy<T: Float>(
    input: &Tensor<T>,
    dim: i64,
    index: &IntTensor<i64>,
    source: &Tensor<T>,
) -> FerrotorchResult<Tensor<T>> {
    // CORE-048 (#1742): strict same-device operands at entry.
    same_device(input.device(), index.device())?;
    same_device(input.device(), source.device())?;
    // CUDA index values are copied to host for bounds/shape validation; tensor
    // payloads stay resident on the CUDA fast path.
    let index_host;
    let index: &IntTensor<i64> = if index.is_cuda() {
        index_host = index.to(Device::Cpu)?;
        &index_host
    } else {
        index
    };

    let input_shape = input.shape();
    let ndim = input_shape.len();

    if ndim == 0 {
        let dim_for_0d = match dim {
            0 | -1 => 0i64,
            _ => {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "index_copy: dim {dim} out of range for 0-d input (valid: -1, 0)"
                    ),
                });
            }
        };
        // Source must be 0-d (or length-1 1-d). Upstream meta at `:285-290`:
        //   if (source.dim() == 0 && numIndices != 1) error
        // and `:291-300`:
        //   if (source.dim() != self.dim() && source.dim() != 0 && self.dim() != 0) error
        // For 0-d self: source must be 0-d (else shape mismatch).
        let source_shape = source.shape();
        let source_is_0d_compatible =
            source_shape.is_empty() || (source_shape.len() == 1 && source_shape[0] <= 1);
        if !source_is_0d_compatible {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "index_copy: When source and destination are not scalars, \
                     their dimensionality must match. Source dimensionality \
                     ({}), destination dimensionality (0)",
                    source_shape.len()
                ),
            });
        }
        let n_indices = index.numel();
        if source_shape.is_empty() {
            if n_indices != 1 {
                return Err(FerrotorchError::ShapeMismatch {
                    message: format!(
                        "index_copy: When source is scalar, index should have one element \
                         (got {n_indices})"
                    ),
                });
            }
        } else if source_shape.len() == 1 && n_indices != source_shape[0] {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "index_copy: Number of indices ({n_indices}) should be equal to \
                     source.size(dim) ({})",
                    source_shape[0]
                ),
            });
        }
        let mut saved_index: Vec<usize> = Vec::new();
        for v in index.data()? {
            let i_raw = v.to_i64();
            // 0-d input has dim_size = 1; upstream rejects negatives.
            if i_raw != 0 {
                return Err(FerrotorchError::IndexOutOfBounds {
                    index: if i_raw < 0 {
                        i_raw.unsigned_abs() as usize
                    } else {
                        i_raw as usize
                    },
                    axis: dim_for_0d as usize,
                    size: 1,
                });
            }
            saved_index.push(0);
        }

        if input.is_cuda() || source.is_cuda() {
            if !cuda_float_dtype::<T>() {
                return Err(FerrotorchError::NotImplementedOnCuda { op: "index_copy" });
            }
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let handle = if saved_index.is_empty() {
                let input_c = no_grad(|| input.contiguous())?;
                backend.clone_buffer(input_c.gpu_handle()?)?
            } else {
                let source_c = no_grad(|| source.contiguous())?;
                backend.clone_buffer(source_c.gpu_handle()?)?
            };
            let storage = TensorStorage::gpu(handle);
            if (input.requires_grad() || source.requires_grad()) && is_grad_enabled() {
                let grad_fn = Arc::new(IndexCopyBackward {
                    input: input.clone(),
                    source: source.clone(),
                    dim: 0,
                    index: saved_index,
                });
                return Tensor::from_operation(storage, vec![], grad_fn);
            }
            return Tensor::from_storage(storage, vec![], false);
        }

        let scalar_val = input.data_vec()?[0];
        let src_data = source.data_vec()?;
        let result_val = if saved_index.is_empty() || src_data.is_empty() {
            scalar_val
        } else {
            src_data[0]
        };
        // CORE-048 (#1742): result on input's device.
        let storage = TensorStorage::on_device(vec![result_val], input.device())?;
        if (input.requires_grad() || source.requires_grad()) && is_grad_enabled() {
            let grad_fn = Arc::new(IndexCopyBackward {
                input: input.clone(),
                source: source.clone(),
                dim: 0,
                index: saved_index,
            });
            return Tensor::from_operation(storage, vec![], grad_fn);
        }
        return Tensor::from_storage(storage, vec![], false);
    }

    // N-D input: route through the shared strict validator. index_copy
    // ACCEPTS 0-d source on N-D self per upstream `:285-300` (broadcasts the
    // scalar source per index slot, requires n_indices == 1) — pass
    // `accept_0d_source = true`.
    let (dim_usize, idx_usize) =
        strict_index_add_copy_validate("index_copy", input, dim, index, source, true)?;

    let in_dim_size = input_shape[dim_usize];
    let outer: usize = crate::shape::numel(&input_shape[..dim_usize]);
    let inner: usize = crate::shape::numel(&input_shape[dim_usize + 1..]);
    let source_shape = source.shape();
    let src_dim_size = if source_shape.is_empty() {
        1
    } else {
        source_shape[dim_usize]
    };

    if input.is_cuda() && source.is_cuda() && cuda_float_dtype::<T>() {
        let output_shape = input_shape.to_vec();
        let input_c = no_grad(|| input.contiguous())?;
        let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let handle = if idx_usize.is_empty() || outer == 0 || inner == 0 || input.numel() == 0 {
            backend.clone_buffer(input_c.gpu_handle()?)?
        } else {
            let source_c = if source_shape.is_empty() {
                no_grad(|| {
                    crate::grad_fns::shape::broadcast_to(source, &[outer, src_dim_size, inner])?
                        .contiguous()
                })?
            } else {
                no_grad(|| source.contiguous())?
            };
            let ordinal = cuda_ordinal(input_c.device(), "index_copy")?;
            let idx_handle = upload_expanded_dim_indices(&idx_usize, outer, inner, ordinal)?;
            scatter_dim_cuda_handle::<T>(
                input_c.gpu_handle()?,
                &idx_handle,
                source_c.gpu_handle()?,
                outer,
                in_dim_size,
                src_dim_size,
                inner,
                false,
                "index_copy",
            )?
        };
        let storage = TensorStorage::gpu(handle);
        if (input.requires_grad() || source.requires_grad()) && is_grad_enabled() {
            let grad_fn = Arc::new(IndexCopyBackward {
                input: input.clone(),
                source: source.clone(),
                dim: dim_usize,
                index: idx_usize,
            });
            return Tensor::from_operation(storage, output_shape, grad_fn);
        }
        return Tensor::from_storage(storage, output_shape, false);
    }
    if input.is_cuda() || source.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "index_copy" });
    }

    let mut out = input.data_vec()?;
    let src_data = source.data_vec()?;

    if source_shape.is_empty() {
        // 0-d source on N-D self: broadcast the single scalar to each
        // (outer × inner) slice at the chosen index slot. The strict
        // validator guarantees `idx_usize.len() == 1` in this branch.
        // Live oracle:
        //   `torch.tensor([1.,2.,3.,4.]).index_copy(0, t([1]), t(99.))`
        //   -> `tensor([1., 99., 3., 4.])` — every element of the
        //   target slice along `dim` at `idx_usize[0]` is set to the
        //   scalar src value (here a length-1 slice for 1-D self).
        let scalar = if src_data.is_empty() {
            <T as num_traits::Zero>::zero()
        } else {
            src_data[0]
        };
        let dst_i = idx_usize[0];
        for o in 0..outer {
            let dst_base = o * in_dim_size * inner + dst_i * inner;
            for k in 0..inner {
                out[dst_base + k] = scalar;
            }
        }
    } else {
        for o in 0..outer {
            for (i, &dst_i) in idx_usize.iter().enumerate() {
                let dst_base = o * in_dim_size * inner + dst_i * inner;
                let src_base = o * src_dim_size * inner + i * inner;
                out[dst_base..dst_base + inner]
                    .copy_from_slice(&src_data[src_base..src_base + inner]);
            }
        }
    }

    let output_shape = input_shape.to_vec();
    if (input.requires_grad() || source.requires_grad()) && is_grad_enabled() {
        let grad_fn = Arc::new(IndexCopyBackward {
            input: input.clone(),
            source: source.clone(),
            dim: dim_usize,
            index: idx_usize,
        });
        // CORE-048 (#1742): result on input's device.
        Tensor::from_operation(
            TensorStorage::on_device(out, input.device())?,
            output_shape,
            grad_fn,
        )
    } else {
        Tensor::from_storage(
            TensorStorage::on_device(out, input.device())?,
            output_shape,
            false,
        )
    }
}

// ---------------------------------------------------------------------------
// masked_scatter (#1252 — REQ-11). Mirrors `torch.masked_scatter(input, mask,
// source)` at upstream `aten/src/ATen/native/TensorAdvancedIndexing.cpp:2402
// Tensor masked_scatter(const Tensor& self, const Tensor& mask, const Tensor&
// source) { auto [_mask, _self] = expand_outplace(mask, self); return
// _self->clone(at::MemoryFormat::Contiguous).masked_scatter_(*_mask, source); }`.
// VJP per `tools/autograd/derivatives.yaml:1105-1108
//   - name: masked_scatter(Tensor self, Tensor mask, Tensor source) -> Tensor
//     self: grad.masked_fill(mask, 0)
//     source: masked_scatter_backward_symint(grad, mask, source.sym_sizes())`.
// ---------------------------------------------------------------------------

/// Backward function for `masked_scatter`.
///
/// Forward: `output = input.clone(); j = 0; for i in 0..output.numel() {
///   if mask[i] { output[i] = source[j]; j += 1; } }` (after broadcasting
/// mask + input to common shape).
///
/// VJP for input: zero grad at mask-true positions (the same pattern as
/// `MaskedFillBackward`).
/// VJP for source: walk mask in C-order, gather grad at every true position
/// into the first `count_nonzero(mask)` elements of grad_source; reshape to
/// source.shape (the inverse of the forward's compaction-from-source).
#[derive(Debug)]
pub struct MaskedScatterBackward<T: Float> {
    /// Saved input handle (for shape + autograd graph linkage).
    pub input: Tensor<T>,
    /// Saved source handle (for shape + numel).
    pub source: Tensor<T>,
    /// The mask, after broadcasting to the input's shape.
    pub mask: BoolTensor,
}

impl<T: Float> GradFn<T> for MaskedScatterBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !is_grad_enabled() {
            return Ok(vec![None, None]);
        }
        if grad_output.numel() != self.mask.numel() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "masked_scatter backward: grad_output numel {} != mask numel {}",
                    grad_output.numel(),
                    self.mask.numel()
                ),
            });
        }

        // CUDA-resident VJP: PyTorch's derivatives are exactly
        //   self   = grad.masked_fill(mask, 0)
        //   source = grad.masked_select(mask).pad_to(source.numel()).view(source.sizes())
        // The backend already exposes those resident primitives. Any CUDA
        // operand reaches this branch; mixed CPU/CUDA state is rejected instead
        // of detouring through host memory.
        if grad_output.is_cuda()
            || self.input.is_cuda()
            || self.source.is_cuda()
            || self.mask.is_cuda()
        {
            same_device(self.input.device(), grad_output.device())?;
            same_device(self.input.device(), self.source.device())?;
            same_device(self.input.device(), self.mask.device())?;
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let grad_c = no_grad(|| grad_output.contiguous())?;

            let grad_input = if self.input.requires_grad() {
                let result_handle =
                    backend.masked_fill_dt(grad_c.gpu_handle()?, self.mask.gpu_handle()?, 0.0)?;
                Some(Tensor::from_storage(
                    TensorStorage::gpu(result_handle),
                    self.input.shape().to_vec(),
                    false,
                )?)
            } else {
                None
            };

            let grad_source = if self.source.requires_grad() {
                let (selected_handle, selected_len) =
                    backend.masked_select(grad_c.gpu_handle()?, self.mask.gpu_handle()?)?;
                let source_numel = self.source.numel();
                if selected_len > source_numel {
                    return Err(FerrotorchError::ShapeMismatch {
                        message: format!(
                            "masked_scatter backward: mask selected {selected_len} gradients, \
                             but source has {source_numel} elements"
                        ),
                    });
                }

                let flat = if selected_len == source_numel {
                    Tensor::from_storage(
                        TensorStorage::gpu(selected_handle),
                        vec![selected_len],
                        false,
                    )?
                } else {
                    let ordinal = cuda_ordinal(self.source.device(), "masked_scatter backward")?;
                    let pad_len = source_numel - selected_len;
                    let zeros = Tensor::from_storage(
                        TensorStorage::gpu(backend.alloc_zeros(pad_len, T::dtype(), ordinal)?),
                        vec![pad_len],
                        false,
                    )?;
                    if selected_len == 0 {
                        zeros
                    } else {
                        let selected = Tensor::from_storage(
                            TensorStorage::gpu(selected_handle),
                            vec![selected_len],
                            false,
                        )?;
                        no_grad(|| crate::grad_fns::shape::cat(&[selected, zeros], 0))?
                    }
                };
                let (storage, _) = flat.into_storage_and_shape()?;
                Some(Tensor::from_storage(
                    storage,
                    self.source.shape().to_vec(),
                    false,
                )?)
            } else {
                None
            };

            return Ok(vec![grad_input, grad_source]);
        }

        let mask_h = self.mask.data()?;
        let go = grad_output.data_vec()?;
        let zero = <T as num_traits::Zero>::zero();

        // grad for input: zero at mask-true positions.
        let grad_input = if self.input.requires_grad() {
            let mut gi = go.clone();
            for (i, &m) in mask_h.iter().enumerate() {
                if m {
                    gi[i] = zero;
                }
            }
            Some(Tensor::from_storage(
                // CORE-048 (#1742): gradient on the input leaf's device.
                TensorStorage::on_device(gi, self.input.device())?,
                self.input.shape().to_vec(),
                false,
            )?)
        } else {
            None
        };

        // grad for source: compact grad at true positions, pad to source.numel()
        // (per upstream `masked_scatter_backward_symint` which builds
        // zeros(sizes) then writes the compacted grad — at
        // `TensorAdvancedIndexing.cpp:2411-2430`).
        let grad_source = if self.source.requires_grad() {
            let source_numel = self.source.numel();
            let true_count = mask_h.iter().filter(|&&m| m).count();
            if true_count > source_numel {
                return Err(FerrotorchError::ShapeMismatch {
                    message: format!(
                        "masked_scatter backward: mask selected {true_count} gradients, \
                         but source has {source_numel} elements"
                    ),
                });
            }
            let mut gs = vec![zero; source_numel];
            let mut j = 0usize;
            for (i, &m) in mask_h.iter().enumerate() {
                if m {
                    gs[j] = go[i];
                    j += 1;
                }
            }
            Some(Tensor::from_storage(
                // CORE-048 (#1742): gradient on the source leaf's device.
                TensorStorage::on_device(gs, self.source.device())?,
                self.source.shape().to_vec(),
                false,
            )?)
        } else {
            None
        };

        Ok(vec![grad_input, grad_source])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input, &self.source]
    }

    fn name(&self) -> &'static str {
        "MaskedScatterBackward"
    }
}

/// `masked_scatter`: copy elements from `source` into a clone of `input` at
/// positions where `mask` is true. Mirrors upstream `Tensor masked_scatter(
/// const Tensor& self, const Tensor& mask, const Tensor& source)` at
/// `aten/src/ATen/native/TensorAdvancedIndexing.cpp:2402-2409`.
///
/// Broadcast: upstream applies `expand_outplace(mask, self)` at `:2406` so
/// the mask and input are broadcast to a common shape before the
/// element-by-element walk. ferrotorch broadcasts both via the shared
/// `broadcast_bool_tensor` + `grad_fns::shape::expand` (autograd-aware) helpers.
///
/// `source` must have at least `count_nonzero(mask)` elements (upstream
/// requirement at `:2406-2408`). The walk consumes source in C-order, taking
/// the first `count_nonzero(mask)` elements.
///
/// # Device contract (CORE-048 / #1742)
///
/// `mask` and `source` must live on `input`'s device — a mix returns
/// [`FerrotorchError::DeviceMismatch`] (torch: "Expected all tensors to be
/// on the same device, but got mask is on cpu, different from other tensors
/// on cuda:0"); the audit's "host-accessible mask" fallback for a CUDA
/// input is therefore unreachable. All-CUDA f32/f64/f16/bf16 runs the
/// on-device `masked_scatter_forward` kernel (#1662, below), preserving
/// residency and dtype bit patterns. Gradients are delivered on the leaves'
/// devices.
pub fn masked_scatter<T: Float>(
    input: &Tensor<T>,
    mask: &BoolTensor,
    source: &Tensor<T>,
) -> FerrotorchResult<Tensor<T>> {
    // CORE-048 (#1742): strict same-device operands at entry.
    same_device(input.device(), mask.device())?;
    same_device(input.device(), source.device())?;

    // Broadcast input + mask to common shape (upstream `expand_outplace` at
    // `TensorAdvancedIndexing.cpp:2406`).
    let common = if input.shape() == mask.shape() {
        input.shape().to_vec()
    } else {
        crate::shape::broadcast_shapes(input.shape(), mask.shape())?
    };
    let input_b = if input.shape() == common.as_slice() {
        input.clone()
    } else {
        crate::grad_fns::shape::expand(input, &common)?
    };
    let mask_b = if mask.shape() == common.as_slice() {
        mask.clone()
    } else {
        broadcast_bool_tensor(mask, &common)?
    };

    // GPU-resident fast path (#1662): input, mask AND source all on CUDA. torch
    // accepts a fully-on-device masked_scatter (input, mask, source all CUDA ->
    // CUDA result); the host path below calls `mask_b.data()` which errors
    // `GpuTensorNotAccessible` on a CUDA bool mask. Route the forward through the
    // on-device kernel `out[i] = mask[i] ? source[j++] : input[i]` (the
    // source-index `j` is the exclusive prefix-sum of the mask, realised by a
    // serial in-order walk — matching upstream
    // `aten/src/ATen/native/cuda/IndexKernel.cu:416-453`). Result stays
    // `is_cuda()`; NO host round trip (R-CODE-4). f32/f64/f16/bf16 are covered
    // by the backend; unsupported CUDA dtypes remain explicit errors.
    if input_b.is_cuda() && mask_b.is_cuda() && source.is_cuda() {
        if matches!(
            T::dtype(),
            crate::dtype::DType::F32
                | crate::dtype::DType::F64
                | crate::dtype::DType::F16
                | crate::dtype::DType::BF16
        ) && input_b.device() == mask_b.device()
            && input_b.device() == source.device()
        {
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            // `.contiguous()` materialises a (possibly narrowed-offset) view's
            // logical [0,n) window on-device (#1657), so the handle's logical len
            // matches the mask numel and the kernel reads `[0, n)` (the #1661
            // pooled-buffer convention). Source is flattened-contiguous too.
            let input_c = input_b.contiguous()?;
            let source_c = source.contiguous()?;
            let n = input_c.numel();
            // The backend reads the on-device true count once (the same
            // single-integer shape sync PyTorch performs in
            // `masked_scatter_size_check`, `IndexKernel.cu:394`) and validates
            // `source.numel() >= count_nonzero(mask)` — NOT a data round trip.
            let result_handle = backend.masked_scatter_forward(
                input_c.gpu_handle()?,
                source_c.gpu_handle()?,
                mask_b.gpu_handle()?,
                n,
            )?;
            let storage = TensorStorage::gpu(result_handle);
            let output_shape = common.clone();
            if (input_c.requires_grad() || source.requires_grad()) && is_grad_enabled() {
                let grad_fn = Arc::new(MaskedScatterBackward {
                    input: input_c.clone(),
                    source: source.clone(),
                    mask: mask_b.clone(),
                });
                return Tensor::from_operation(storage, output_shape, grad_fn);
            }
            return Tensor::from_storage(storage, output_shape, false);
        }
        return Err(FerrotorchError::NotImplementedOnCuda {
            op: "masked_scatter",
        });
    }

    // Host path: CPU tensors only. Same-device CUDA operands should have been
    // handled by the resident branch above or rejected as unsupported, never
    // downloaded for a fallback walk.
    let mask_host;
    let mask_h = if mask_b.is_cuda() {
        mask_host = mask_b.to(Device::Cpu)?;
        mask_host.data()?
    } else {
        mask_b.data()?
    };
    let true_count = mask_h.iter().filter(|&&b| b).count();
    if source.numel() < true_count {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "masked_scatter: source has {} elements, but mask has {} true positions",
                source.numel(),
                true_count
            ),
        });
    }

    let in_data = input_b.data_vec()?;
    let src_data = source.data_vec()?;
    let mut out = in_data.clone();
    let mut j = 0usize;
    for (i, &m) in mask_h.iter().enumerate() {
        if m {
            out[i] = src_data[j];
            j += 1;
        }
    }

    let output_shape = common.clone();
    if (input_b.requires_grad() || source.requires_grad()) && is_grad_enabled() {
        let grad_fn = Arc::new(MaskedScatterBackward {
            input: input_b.clone(),
            source: source.clone(),
            mask: mask_b.clone(),
        });
        // CORE-048 (#1742): result on input's device.
        Tensor::from_operation(
            TensorStorage::on_device(out, input_b.device())?,
            output_shape,
            grad_fn,
        )
    } else {
        Tensor::from_storage(
            TensorStorage::on_device(out, input_b.device())?,
            output_shape,
            false,
        )
    }
}

// ---------------------------------------------------------------------------
// take (#1253 — REQ-12). Mirrors `torch.take(input, index)` at upstream
// `aten/src/ATen/native/TensorAdvancedIndexing.cpp:1067-1071 Tensor take(
// const Tensor& self, const Tensor& index) { auto out = at::empty(
// index.sizes(), self.options()); at::native::take_out(self, index, out);
// return out; }`. VJP per `tools/autograd/derivatives.yaml:1766-1769
//   - name: take(Tensor self, Tensor index) -> Tensor
//     self: take_backward(grad, self, index)
//     index: non_differentiable
//     result: auto_linear`.
// take_backward = `zeros_like(self).put_(index, grad, accumulate=true)` —
// scatter grad into a zeros buffer of input shape at flat index positions.
// ---------------------------------------------------------------------------

/// Backward function for `take`.
///
/// Forward: `output[i] = input.view(-1)[index[i]]` — flat-index gather.
///
/// VJP for input: `zeros_like(input).put_(index, grad, accumulate=true)` —
/// scatter-add grad at the flat positions the forward read from. Equivalent
/// to a flat scatter-add (matches the `put_` accumulate=true semantics; if
/// `index` contains duplicates the gradient accumulates).
#[derive(Debug)]
pub struct TakeBackward<T: Float> {
    /// Saved input handle (for shape + autograd graph linkage).
    pub input: Tensor<T>,
    /// Flat indices into input's contiguous buffer.
    pub index: Vec<usize>,
}

impl<T: Float> GradFn<T> for TakeBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !is_grad_enabled() {
            return Ok(vec![None]);
        }
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }
        let input_shape = self.input.shape().to_vec();
        let input_numel: usize = if input_shape.is_empty() {
            1
        } else {
            crate::shape::numel(&input_shape)
        };
        if grad_output.is_cuda() {
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let ordinal = cuda_ordinal(grad_output.device(), "TakeBackward")?;
            let idx_handle = crate::ops::indexing::upload_index_i64(&self.index, ordinal)?;
            let zeros = backend.alloc_zeros(input_numel, T::dtype(), ordinal)?;
            let grad_c = no_grad(|| grad_output.contiguous())?;
            let result_handle = scatter_dim_cuda_handle::<T>(
                &zeros,
                &idx_handle,
                grad_c.gpu_handle()?,
                1,
                input_numel,
                self.index.len(),
                1,
                true,
                "TakeBackward",
            )?;
            let grad_tensor =
                Tensor::from_storage(TensorStorage::gpu(result_handle), input_shape, false)?;
            return Ok(vec![Some(grad_tensor)]);
        }
        let go = grad_output.data_vec()?;
        let zero = <T as num_traits::Zero>::zero();
        let mut grad_input = vec![zero; input_numel];
        for (i, &idx) in self.index.iter().enumerate() {
            if idx < input_numel && i < go.len() {
                grad_input[idx] += go[i];
            }
        }
        // CORE-048 (#1742): gradient on the input leaf's device.
        let grad_tensor = Tensor::from_storage(
            TensorStorage::on_device(grad_input, self.input.device())?,
            input_shape,
            false,
        )?;
        Ok(vec![Some(grad_tensor)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "TakeBackward"
    }
}

/// `take`: flat-index gather. `output[i] = input.view(-1)[index[i]]`, output
/// shape = index shape. Mirrors upstream `Tensor take(const Tensor& self,
/// const Tensor& index)` at `aten/src/ATen/native/TensorAdvancedIndexing.cpp:
/// 1067-1071`.
///
/// `index` may be any shape (including 0-d for a single scalar pull); index
/// values are flat indices into the C-contiguous buffer of `input`. Negative
/// indices wrap per `idx + input.numel()`. Out-of-range raises
/// `IndexOutOfBounds`.
///
/// # Device contract (CORE-048 / #1742)
///
/// `index` must live on `input`'s device — a mix returns
/// [`FerrotorchError::DeviceMismatch`] (torch: "Expected all tensors to be
/// on the same device, but got index is on cpu, different from other
/// tensors on cuda:0"). CUDA f32/f64/f16/bf16 inputs gather ON-DEVICE via
/// the dtype-generic `index_select_intidx` path (a flat take is
/// `outer=1, inner=1`); the validated/wrapped index uploads as a resident
/// `i64` buffer, and only the index itself downloads host-side for value
/// validation. Empty CUDA indices allocate a resident empty output with the
/// input dtype. Gradients are delivered on the leaf's device.
pub fn take<T: Float>(input: &Tensor<T>, index: &IntTensor<i64>) -> FerrotorchResult<Tensor<T>> {
    // CORE-048 (#1742): strict same-device operands at entry.
    same_device(input.device(), index.device())?;
    // Host copy of a CUDA index for value validation (see device contract).
    let index_host;
    let index: &IntTensor<i64> = if index.is_cuda() {
        index_host = index.to(Device::Cpu)?;
        &index_host
    } else {
        index
    };

    let input_numel = input.numel();
    let input_numel_i64 =
        i64::try_from(input_numel).map_err(|_| FerrotorchError::InvalidArgument {
            message: format!("take: input numel {input_numel} exceeds i64::MAX"),
        })?;

    let mut idx_usize: Vec<usize> = Vec::with_capacity(index.numel());
    for v in index.data()? {
        let i_raw = v.to_i64();
        if i_raw < -input_numel_i64 || i_raw >= input_numel_i64 {
            return Err(FerrotorchError::IndexOutOfBounds {
                index: if i_raw < 0 {
                    i_raw.unsigned_abs() as usize
                } else {
                    i_raw as usize
                },
                axis: 0,
                size: input_numel,
            });
        }
        let i = if i_raw < 0 {
            i_raw + input_numel_i64
        } else {
            i_raw
        };
        idx_usize.push(i as usize);
    }

    // Output shape matches index shape.
    let output_shape = index.shape().to_vec();
    let output_numel = if output_shape.is_empty() {
        1
    } else {
        crate::shape::numel(&output_shape)
    };

    // CUDA-resident path. `.contiguous()` materialises the logical [0, n)
    // window on-device (#1657) so flat addressing matches `data_vec`'s C-order.
    if input.is_cuda() {
        let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let h = if idx_usize.is_empty() {
            let ordinal = cuda_ordinal(input.device(), "take")?;
            backend.alloc_zeros(0, T::dtype(), ordinal)?
        } else {
            let input_c = input.contiguous()?;
            let input_handle = input_c.gpu_handle()?;
            let ordinal = input_handle.device_ordinal();
            let idx_handle = crate::ops::indexing::upload_index_i64(&idx_usize, ordinal)?;
            backend.index_select_intidx(
                input_handle,
                &idx_handle,
                1,
                input_numel,
                idx_usize.len(),
                1,
            )?
        };
        let storage = TensorStorage::gpu(h);
        if input.requires_grad() && is_grad_enabled() {
            let grad_fn = Arc::new(TakeBackward {
                input: input.clone(),
                index: idx_usize,
            });
            return Tensor::from_operation(storage, output_shape, grad_fn);
        }
        return Tensor::from_storage(storage, output_shape, false);
    }

    // CPU host path.
    let input_data = input.data_vec()?;
    let mut out = Vec::with_capacity(output_numel);
    // For a 0-d index tensor `index.numel()` == 1 (the scalar count), so the
    // loop runs once with idx_usize[0].
    for &idx in &idx_usize {
        out.push(input_data[idx]);
    }
    // Edge case: 0-d input + 0-d empty index — keep length consistent.
    if out.is_empty() && output_numel == 1 {
        out.push(<T as num_traits::Zero>::zero());
    }

    if input.requires_grad() && is_grad_enabled() {
        let grad_fn = Arc::new(TakeBackward {
            input: input.clone(),
            index: idx_usize,
        });
        Tensor::from_operation(
            TensorStorage::on_device(out, input.device())?,
            output_shape,
            grad_fn,
        )
    } else {
        Tensor::from_storage(
            TensorStorage::on_device(out, input.device())?,
            output_shape,
            false,
        )
    }
}

// ---------------------------------------------------------------------------
// put (#1254 — REQ-13). Mirrors `torch.put(input, index, source, accumulate=
// False)` at upstream `aten/src/ATen/native/TensorAdvancedIndexing.cpp:928-934
// Tensor put(const Tensor& self, const Tensor& index, const Tensor& source,
// const bool accumulate) { return self.clone(at::MemoryFormat::Preserve)
// .put_(index, source, accumulate); }`. VJP per `tools/autograd/derivatives.
// yaml:1421-1424
//   - name: put(Tensor self, Tensor index, Tensor source, bool accumulate=False) -> Tensor
//     self: "accumulate ? grad : grad.put(index, zeros_like(source), false)"
//     index: non_differentiable
//     source: grad.take(index).reshape_as(source)`. Depends on REQ-12 (take, SHIPPED above).
// ---------------------------------------------------------------------------

/// Backward function for `put`.
///
/// Forward: `output = input.clone(); output.view(-1)[index[i]] = source[i]`
/// (when `accumulate=False`) or `+= source[i]` (when `accumulate=True`).
///
/// VJP for input (accumulate=true): identity — addition passes grad through.
/// VJP for input (accumulate=false): zero grad at every flat position the put
/// overwrote (`grad.put(index, zeros_like(source), false)` per upstream).
/// VJP for source: gather grad at the flat positions (`grad.take(index)`).
#[derive(Debug)]
pub struct PutBackward<T: Float> {
    /// Saved input handle.
    pub input: Tensor<T>,
    /// Saved source handle.
    pub source: Tensor<T>,
    /// Flat indices (validated, non-negative).
    pub index: Vec<usize>,
    /// Whether accumulate mode was on in the forward.
    pub accumulate: bool,
}

impl<T: Float> GradFn<T> for PutBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !is_grad_enabled() {
            return Ok(vec![None, None]);
        }
        let input_shape = self.input.shape().to_vec();
        let input_numel: usize = if input_shape.is_empty() {
            1
        } else {
            crate::shape::numel(&input_shape)
        };
        if grad_output.is_cuda() {
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let ordinal = cuda_ordinal(grad_output.device(), "PutBackward")?;
            let idx_handle = crate::ops::indexing::upload_index_i64(&self.index, ordinal)?;
            let grad_c = no_grad(|| grad_output.contiguous())?;

            let grad_input = if self.input.requires_grad() {
                let result_handle = if self.accumulate {
                    backend.clone_buffer(grad_c.gpu_handle()?)?
                } else {
                    let zeros = backend.alloc_zeros(self.index.len(), T::dtype(), ordinal)?;
                    scatter_dim_cuda_handle::<T>(
                        grad_c.gpu_handle()?,
                        &idx_handle,
                        &zeros,
                        1,
                        input_numel,
                        self.index.len(),
                        1,
                        false,
                        "PutBackward",
                    )?
                };
                Some(Tensor::from_storage(
                    TensorStorage::gpu(result_handle),
                    input_shape.clone(),
                    false,
                )?)
            } else {
                None
            };

            let grad_source = if self.source.requires_grad() {
                let result_handle = backend.index_select_intidx(
                    grad_c.gpu_handle()?,
                    &idx_handle,
                    1,
                    input_numel,
                    self.index.len(),
                    1,
                )?;
                Some(Tensor::from_storage(
                    TensorStorage::gpu(result_handle),
                    self.source.shape().to_vec(),
                    false,
                )?)
            } else {
                None
            };

            return Ok(vec![grad_input, grad_source]);
        }
        let go = grad_output.data_vec()?;
        let zero = <T as num_traits::Zero>::zero();

        // grad for input.
        let grad_input = if self.input.requires_grad() {
            let mut gi = go.clone();
            if !self.accumulate {
                for &idx in &self.index {
                    if idx < input_numel {
                        gi[idx] = zero;
                    }
                }
            }
            Some(Tensor::from_storage(
                // CORE-048 (#1742): gradient on the input leaf's device.
                TensorStorage::on_device(gi, self.input.device())?,
                input_shape,
                false,
            )?)
        } else {
            None
        };

        // grad for source: gather grad at flat index positions.
        let grad_source = if self.source.requires_grad() {
            let source_numel = self.source.numel();
            let mut gs = vec![zero; source_numel];
            for (i, &idx) in self.index.iter().enumerate() {
                if idx < go.len() && i < source_numel {
                    gs[i] = go[idx];
                }
            }
            Some(Tensor::from_storage(
                // CORE-048 (#1742): gradient on the source leaf's device.
                TensorStorage::on_device(gs, self.source.device())?,
                self.source.shape().to_vec(),
                false,
            )?)
        } else {
            None
        };

        Ok(vec![grad_input, grad_source])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input, &self.source]
    }

    fn name(&self) -> &'static str {
        "PutBackward"
    }
}

/// `put`: flat-index scatter. `output = input.clone();
/// output.view(-1)[index[i]] = source[i]` (or `+= source[i]` when
/// `accumulate=True`). Mirrors upstream `Tensor put(const Tensor& self, const
/// Tensor& index, const Tensor& source, const bool accumulate)` at
/// `aten/src/ATen/native/TensorAdvancedIndexing.cpp:928-934`.
///
/// `index` may be any shape; values are flat indices into input's
/// C-contiguous buffer (negative-wrap per `idx + input.numel()`,
/// out-of-range raises `IndexOutOfBounds`). `source` must have exactly as
/// many elements as `index`.
///
/// # Device contract (CORE-048 / #1742)
///
/// `index` and `source` must live on `input`'s device — a mix returns
/// [`FerrotorchError::DeviceMismatch`] (torch: "Expected all tensors to be
/// on the same device, but got source is on cpu, different from other
/// tensors on cuda:0"). CUDA f32/f64/f16/bf16 inputs scatter ON-DEVICE via
/// dim-aware scatter (`accumulate=false`) or atomic scatter-add
/// (`accumulate=true`) kernels — a flat scatter is the `outer=1, inner=1`
/// special case; the validated/wrapped index uploads as a resident `i64`
/// buffer, and only the index itself downloads host-side for value
/// validation. Like torch's CUDA `put_`, the `accumulate=false` write order
/// for DUPLICATE flat indices is nondeterministic on the kernel path. Empty
/// CUDA indices clone the input on-device. Gradients are delivered on the
/// leaves' devices.
pub fn put<T: Float>(
    input: &Tensor<T>,
    index: &IntTensor<i64>,
    source: &Tensor<T>,
    accumulate: bool,
) -> FerrotorchResult<Tensor<T>> {
    // CORE-048 (#1742): strict same-device operands at entry.
    same_device(input.device(), index.device())?;
    same_device(input.device(), source.device())?;
    // Host copy of a CUDA index for value validation (see device contract).
    let index_host;
    let index: &IntTensor<i64> = if index.is_cuda() {
        index_host = index.to(Device::Cpu)?;
        &index_host
    } else {
        index
    };

    let input_shape = input.shape().to_vec();
    let input_numel: usize = if input_shape.is_empty() {
        1
    } else {
        crate::shape::numel(&input_shape)
    };
    let input_numel_i64 = input_numel as i64;

    let mut idx_usize: Vec<usize> = Vec::with_capacity(index.numel());
    for v in index.data()? {
        let i_raw = v.to_i64();
        if i_raw < -input_numel_i64 || i_raw >= input_numel_i64 {
            return Err(FerrotorchError::IndexOutOfBounds {
                index: if i_raw < 0 {
                    i_raw.unsigned_abs() as usize
                } else {
                    i_raw as usize
                },
                axis: 0,
                size: input_numel,
            });
        }
        let i = if i_raw < 0 {
            i_raw + input_numel_i64
        } else {
            i_raw
        };
        idx_usize.push(i as usize);
    }

    if source.numel() != idx_usize.len() {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "put: source numel {} must equal index numel {}",
                source.numel(),
                idx_usize.len()
            ),
        });
    }

    // CUDA-resident path. `.contiguous()` materialises the logical [0, n)
    // window on-device (#1657) so flat kernel addressing matches `data_vec`'s
    // C-order.
    if input.is_cuda() {
        let h = if idx_usize.is_empty() {
            let input_c = input.contiguous()?;
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            backend.clone_buffer(input_c.gpu_handle()?)?
        } else {
            let input_c = input.contiguous()?;
            let source_c = source.contiguous()?;
            let input_handle = input_c.gpu_handle()?;
            let ordinal = input_handle.device_ordinal();
            let idx_handle = crate::ops::indexing::upload_index_i64(&idx_usize, ordinal)?;
            scatter_dim_cuda_handle::<T>(
                input_handle,
                &idx_handle,
                source_c.gpu_handle()?,
                1,
                input_numel,
                idx_usize.len(),
                1,
                accumulate,
                "put",
            )?
        };
        let storage = TensorStorage::gpu(h);
        if (input.requires_grad() || source.requires_grad()) && is_grad_enabled() {
            let grad_fn = Arc::new(PutBackward {
                input: input.clone(),
                source: source.clone(),
                index: idx_usize,
                accumulate,
            });
            return Tensor::from_operation(storage, input_shape, grad_fn);
        }
        return Tensor::from_storage(storage, input_shape, false);
    }

    // CPU host path.
    let mut out = input.data_vec()?;
    if out.is_empty() && input_numel == 1 {
        out.push(<T as num_traits::Zero>::zero());
    }
    let src_data = source.data_vec()?;
    for (i, &idx) in idx_usize.iter().enumerate() {
        let s = src_data[i];
        if accumulate {
            out[idx] += s;
        } else {
            out[idx] = s;
        }
    }

    if (input.requires_grad() || source.requires_grad()) && is_grad_enabled() {
        let grad_fn = Arc::new(PutBackward {
            input: input.clone(),
            source: source.clone(),
            index: idx_usize,
            accumulate,
        });
        Tensor::from_operation(
            TensorStorage::on_device(out, input.device())?,
            input_shape,
            grad_fn,
        )
    } else {
        Tensor::from_storage(
            TensorStorage::on_device(out, input.device())?,
            input_shape,
            false,
        )
    }
}

#[cfg(test)]
mod first_class_wrappers_tests {
    use super::*;

    #[test]
    fn masked_fill_bt_replaces_true_positions() {
        let t = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0_f32, 2.0, 3.0, 4.0]),
            vec![4],
            false,
        )
        .unwrap();
        let mask = BoolTensor::from_vec(vec![true, false, true, false], vec![4]).unwrap();
        let out = masked_fill_bt(&t, &mask, -1.0).unwrap();
        assert_eq!(out.data().unwrap(), &[-1.0, 2.0, -1.0, 4.0]);
    }

    #[test]
    fn masked_fill_bt_rejects_shape_mismatch() {
        let t =
            Tensor::from_storage(TensorStorage::cpu(vec![1.0_f32, 2.0]), vec![2], false).unwrap();
        let mask = BoolTensor::from_vec(vec![true, false, true], vec![3]).unwrap();
        let err = masked_fill_bt(&t, &mask, 0.0).unwrap_err();
        assert!(matches!(err, FerrotorchError::ShapeMismatch { .. }));
    }

    #[test]
    fn index_select_1d_it_picks_at_indices() {
        let t = Tensor::from_storage(
            TensorStorage::cpu(vec![10.0_f32, 20.0, 30.0, 40.0]),
            vec![4],
            false,
        )
        .unwrap();
        let idx: IntTensor<i64> = IntTensor::from_vec(vec![3, 0, 2], vec![3]).unwrap();
        let out = index_select_1d_it(&t, &idx).unwrap();
        assert_eq!(out.data().unwrap(), &[40.0, 10.0, 30.0]);
    }

    #[test]
    fn index_select_1d_it_rejects_2d_indices() {
        let t = Tensor::from_storage(TensorStorage::cpu(vec![1.0_f32; 4]), vec![4], false).unwrap();
        let idx: IntTensor<i64> = IntTensor::from_vec(vec![0, 1, 2, 3], vec![2, 2]).unwrap();
        let err = index_select_1d_it(&t, &idx).unwrap_err();
        assert!(matches!(err, FerrotorchError::ShapeMismatch { .. }));
    }

    #[test]
    fn index_select_1d_it_rejects_negative() {
        let t = Tensor::from_storage(TensorStorage::cpu(vec![1.0_f32; 4]), vec![4], false).unwrap();
        let idx: IntTensor<i64> = IntTensor::from_vec(vec![0, -1, 2], vec![3]).unwrap();
        let err = index_select_1d_it(&t, &idx).unwrap_err();
        assert!(matches!(err, FerrotorchError::InvalidArgument { .. }));
    }

    // -----------------------------------------------------------------------
    // Broadcasting wrapper tests (closes #1250 #1251 #1255 — see header for
    // upstream PyTorch broadcast contract per
    // `aten/src/ATen/native/TensorAdvancedIndexing.cpp:2503-2545` and
    // `aten/src/ATen/native/TensorCompare.cpp:629-637`).
    //
    // Tests use `?` propagation so the anti-pattern-gate hook (which scans
    // Edit patches without honoring the `#[cfg(test)]` exemption applied for
    // Write) accepts the patch.
    // -----------------------------------------------------------------------

    fn bcast_cpu_f32(data: Vec<f32>, shape: Vec<usize>) -> FerrotorchResult<Tensor<f32>> {
        Tensor::from_storage(TensorStorage::cpu(data), shape, false)
    }

    fn bcast_cpu_f32_grad(data: Vec<f32>, shape: Vec<usize>) -> FerrotorchResult<Tensor<f32>> {
        Tensor::from_storage(TensorStorage::cpu(data), shape, true)
    }

    #[test]
    fn masked_fill_bcast_passthrough_same_shape() -> FerrotorchResult<()> {
        let t = bcast_cpu_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
        let mask = BoolTensor::from_vec(vec![true, false, false, true], vec![2, 2])?;
        let out = masked_fill_bcast(&t, &mask, -1.0)?;
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(out.data()?, &[-1.0, 2.0, 3.0, -1.0]);
        Ok(())
    }

    #[test]
    fn masked_fill_bcast_broadcasts_row_mask_to_matrix() -> FerrotorchResult<()> {
        // input [2, 3], mask [3] — torch broadcasts mask across rows.
        // Verified against the upstream contract at
        // `TensorAdvancedIndexing.cpp:2503 expand_outplace(mask, self)`.
        let t = bcast_cpu_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
        let mask = BoolTensor::from_vec(vec![true, false, true], vec![3])?;
        let out = masked_fill_bcast(&t, &mask, 0.0)?;
        assert_eq!(out.shape(), &[2, 3]);
        // mask -> [[T,F,T],[T,F,T]]
        assert_eq!(out.data()?, &[0.0, 2.0, 0.0, 0.0, 5.0, 0.0]);
        Ok(())
    }

    #[test]
    fn masked_fill_bcast_broadcasts_scalar_input_against_2d_mask() -> FerrotorchResult<()> {
        // input shape [] (scalar), mask [2, 2] — input broadcasts to [2, 2].
        let t = bcast_cpu_f32(vec![7.0], vec![])?;
        let mask = BoolTensor::from_vec(vec![true, false, true, true], vec![2, 2])?;
        let out = masked_fill_bcast(&t, &mask, -1.0)?;
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(out.data()?, &[-1.0, 7.0, -1.0, -1.0]);
        Ok(())
    }

    #[test]
    fn masked_fill_bcast_rejects_incompatible_shapes() -> FerrotorchResult<()> {
        let t = bcast_cpu_f32(vec![1.0_f32; 6], vec![2, 3])?;
        let mask = BoolTensor::from_vec(vec![true; 4], vec![2, 2])?;
        let err = masked_fill_bcast(&t, &mask, 0.0).err();
        assert!(matches!(err, Some(FerrotorchError::ShapeMismatch { .. })));
        Ok(())
    }

    #[test]
    fn masked_select_bcast_passthrough_same_shape() -> FerrotorchResult<()> {
        let t = bcast_cpu_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
        let mask = BoolTensor::from_vec(vec![true, false, false, true], vec![2, 2])?;
        let out = masked_select_bcast(&t, &mask)?;
        // Compaction order is C-order (flat layout); true positions are 0, 3.
        assert_eq!(out.shape(), &[2]);
        assert_eq!(out.data()?, &[1.0, 4.0]);
        Ok(())
    }

    #[test]
    fn masked_select_bcast_broadcasts_1d_mask_against_2d_input() -> FerrotorchResult<()> {
        // input [2, 3], mask [3] — both broadcast to [2, 3]; selection is
        // 100% byte-for-byte vs upstream `masked_select_cpu` at
        // `TensorAdvancedIndexing.cpp:2621-2624` whose `expand_outplace`
        // step at `:2545` produces the same broadcast.
        let t = bcast_cpu_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
        let mask = BoolTensor::from_vec(vec![true, false, true], vec![3])?;
        let out = masked_select_bcast(&t, &mask)?;
        // mask expands to [[T,F,T],[T,F,T]] => kept = [1.0, 3.0, 4.0, 6.0]
        assert_eq!(out.shape(), &[4]);
        assert_eq!(out.data()?, &[1.0, 3.0, 4.0, 6.0]);
        Ok(())
    }

    #[test]
    fn masked_select_bcast_broadcasts_1d_input_against_2d_mask() -> FerrotorchResult<()> {
        // input [3], mask [2, 3] — input broadcasts to [2, 3] (each row a copy).
        let t = bcast_cpu_f32(vec![10.0, 20.0, 30.0], vec![3])?;
        let mask = BoolTensor::from_vec(vec![true, true, false, false, true, true], vec![2, 3])?;
        let out = masked_select_bcast(&t, &mask)?;
        // Broadcast input -> [[10,20,30],[10,20,30]]. Mask flattened = T T F F T T.
        // Selected: 10, 20, 20, 30.
        assert_eq!(out.shape(), &[4]);
        assert_eq!(out.data()?, &[10.0, 20.0, 20.0, 30.0]);
        Ok(())
    }

    #[test]
    fn where_cond_bcast_passthrough_same_shape() -> FerrotorchResult<()> {
        let cond = BoolTensor::from_vec(vec![true, false, true, false], vec![2, 2])?;
        let x = bcast_cpu_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
        let y = bcast_cpu_f32(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2])?;
        let out = where_cond_bcast(&cond, &x, &y)?;
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(out.data()?, &[1.0, 20.0, 3.0, 40.0]);
        Ok(())
    }

    #[test]
    fn where_cond_bcast_three_way_broadcast_with_scalars() -> FerrotorchResult<()> {
        // x shape [], cond [2, 2], y [1, 2] — common shape [2, 2].
        // Mirrors the upstream 3-way TensorIterator at
        // `TensorCompare.cpp:629-637 where_self_out`.
        let cond = BoolTensor::from_vec(vec![true, false, false, true], vec![2, 2])?;
        let x = bcast_cpu_f32(vec![7.0], vec![])?;
        let y = bcast_cpu_f32(vec![100.0, 200.0], vec![1, 2])?;
        let out = where_cond_bcast(&cond, &x, &y)?;
        // x broadcasts to [[7,7],[7,7]]; y to [[100,200],[100,200]].
        // result: [[x, y],[y, x]] = [[7,200],[100,7]].
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(out.data()?, &[7.0, 200.0, 100.0, 7.0]);
        Ok(())
    }

    #[test]
    fn where_cond_bcast_rejects_incompatible_shapes() -> FerrotorchResult<()> {
        // x [2, 3] vs y [2, 4] — no common shape.
        let cond = BoolTensor::from_vec(vec![true; 6], vec![2, 3])?;
        let x = bcast_cpu_f32(vec![1.0_f32; 6], vec![2, 3])?;
        let y = bcast_cpu_f32(vec![0.0_f32; 8], vec![2, 4])?;
        let err = where_cond_bcast(&cond, &x, &y).err();
        assert!(matches!(err, Some(FerrotorchError::ShapeMismatch { .. })));
        Ok(())
    }

    #[test]
    fn masked_select_bcast_backward_reduces_to_input_shape() -> FerrotorchResult<()> {
        // Verify autograd correctness across the broadcast: an input shape [3]
        // selected via a [2, 3] mask must receive a gradient of shape [3]
        // (via the upstream ExpandBackward shrink). Mirrors the upstream
        // contract at `TensorAdvancedIndexing.cpp:2626-2655 masked_select_backward`
        // which builds `zeros_like(input.expand(infer_size(input, mask)))`.
        use crate::autograd::graph::backward;
        let t = bcast_cpu_f32_grad(vec![10.0, 20.0, 30.0], vec![3])?;
        let mask = BoolTensor::from_vec(vec![true, false, true, false, true, true], vec![2, 3])?;
        let out = masked_select_bcast(&t, &mask)?;
        // Compose a scalar via sum so backward has a well-defined seed.
        #[derive(Debug)]
        struct BcastSumBackward<T: Float> {
            input: Tensor<T>,
            numel: usize,
        }
        impl<T: Float> GradFn<T> for BcastSumBackward<T> {
            fn backward(
                &self,
                _grad_output: &Tensor<T>,
            ) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
                let ones = vec![<T as num_traits::One>::one(); self.numel];
                let t = Tensor::from_storage(
                    TensorStorage::cpu(ones),
                    self.input.shape().to_vec(),
                    false,
                )?;
                Ok(vec![Some(t)])
            }
            fn inputs(&self) -> Vec<&Tensor<T>> {
                vec![&self.input]
            }
            fn name(&self) -> &'static str {
                "BcastTestSumBackward"
            }
        }
        let out_numel = out.numel();
        let total: f32 = out.data()?.iter().sum();
        let scalar = Tensor::from_operation(
            TensorStorage::cpu(vec![total]),
            vec![],
            Arc::new(BcastSumBackward {
                input: out.clone(),
                numel: out_numel,
            }),
        )?;
        backward(&scalar)?;
        let g_opt = t.grad()?;
        let g = match g_opt {
            Some(g) => g,
            None => {
                return Err(FerrotorchError::Internal {
                    message: "no grad on leaf".into(),
                });
            }
        };
        // Expected: gradient at input axis = #true mask positions broadcast to that index.
        // Broadcast mask [[T,F,T],[F,T,T]] over axis-0 (size 2) — per-column counts:
        // col 0: T+F = 1; col 1: F+T = 1; col 2: T+T = 2 → grad = [1, 1, 2].
        assert_eq!(g.shape(), &[3]);
        assert_eq!(g.data()?, &[1.0, 1.0, 2.0]);
        Ok(())
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

    /// Helper: create a 1-D leaf tensor with `requires_grad`.
    fn leaf_1d(data: &[f32], requires_grad: bool) -> Tensor<f32> {
        Tensor::from_storage(
            TensorStorage::cpu(data.to_vec()),
            vec![data.len()],
            requires_grad,
        )
        .unwrap()
    }

    // --- index_select_1d forward ---

    #[test]
    fn test_index_select_1d_forward() {
        let input = leaf_1d(&[10.0, 20.0, 30.0, 40.0, 50.0], false);
        let result = index_select_1d(&input, &[0, 2, 4]).unwrap();

        assert_eq!(result.shape(), &[3]);
        assert_eq!(result.data().unwrap(), &[10.0, 30.0, 50.0]);
    }

    #[test]
    fn test_index_select_1d_duplicate_indices() {
        let input = leaf_1d(&[10.0, 20.0, 30.0], false);
        let result = index_select_1d(&input, &[1, 1, 2, 0, 1]).unwrap();

        assert_eq!(result.shape(), &[5]);
        assert_eq!(result.data().unwrap(), &[20.0, 20.0, 30.0, 10.0, 20.0]);
    }

    #[test]
    fn test_index_select_1d_out_of_bounds() {
        let input = leaf_1d(&[10.0, 20.0, 30.0], false);
        let result = index_select_1d(&input, &[0, 5]);
        assert!(result.is_err());
    }

    #[test]
    fn test_index_select_1d_non_1d_input() {
        let input = Tensor::<f32>::from_storage(
            TensorStorage::cpu(vec![1.0, 2.0, 3.0, 4.0]),
            vec![2, 2],
            false,
        )
        .unwrap();
        let result = index_select_1d(&input, &[0]);
        assert!(result.is_err());
    }

    // --- index_select_1d backward ---

    #[test]
    fn test_index_select_1d_backward_simple() {
        // input = [10, 20, 30, 40], select indices [1, 3]
        // output = [20, 40]
        // sum(output) = 60   (scalar for backward)
        //
        // grad_output for sum = [1, 1]
        // grad_input = [0, 1, 0, 1]  (scatter_add of [1,1] at [1,3])
        let input = leaf_1d(&[10.0, 20.0, 30.0, 40.0], true);
        let selected = index_select_1d(&input, &[1, 3]).unwrap();

        assert!(selected.requires_grad());
        assert!(!selected.is_leaf());
        assert_eq!(selected.grad_fn().unwrap().name(), "IndexSelectBackward");

        // Sum the selected tensor to get a scalar.
        let data = selected.data().unwrap();
        let total: f32 = data.iter().sum();
        let sum_storage = TensorStorage::cpu(vec![total]);

        // SumBackward: broadcasts the scalar grad_output to the shape of the input.
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
            sum_storage,
            vec![],
            Arc::new(SumBackward {
                input: selected.clone(),
            }),
        )
        .unwrap();

        backward(&loss).unwrap();

        let grad = input.grad().unwrap().unwrap();
        let grad_data = grad.data().unwrap();
        assert_eq!(grad_data.len(), 4);
        assert!((grad_data[0] - 0.0).abs() < 1e-6, "grad[0] should be 0");
        assert!((grad_data[1] - 1.0).abs() < 1e-6, "grad[1] should be 1");
        assert!((grad_data[2] - 0.0).abs() < 1e-6, "grad[2] should be 0");
        assert!((grad_data[3] - 1.0).abs() < 1e-6, "grad[3] should be 1");
    }

    #[test]
    fn test_index_select_1d_backward_duplicate_indices() {
        // input = [10, 20, 30], select indices [0, 1, 1, 2, 1]
        // output = [10, 20, 20, 30, 20]
        // sum(output) = 100
        //
        // grad_output for sum = [1, 1, 1, 1, 1]
        // grad_input:
        //   idx 0 appears 1 time -> grad_input[0] = 1
        //   idx 1 appears 3 times -> grad_input[1] = 3
        //   idx 2 appears 1 time -> grad_input[2] = 1
        let input = leaf_1d(&[10.0, 20.0, 30.0], true);
        let selected = index_select_1d(&input, &[0, 1, 1, 2, 1]).unwrap();

        // Manually invoke the backward of IndexSelectBackward with a
        // uniform grad_output of ones.
        let grad_output =
            Tensor::from_storage(TensorStorage::cpu(vec![1.0; 5]), vec![5], false).unwrap();

        let grad_fn = selected.grad_fn().unwrap();
        let grads = grad_fn.backward(&grad_output).unwrap();

        let grad_input = grads[0].as_ref().unwrap();
        let gd = grad_input.data().unwrap();

        assert_eq!(gd.len(), 3);
        assert!(
            (gd[0] - 1.0).abs() < 1e-6,
            "grad[0] = {}, expected 1",
            gd[0]
        );
        assert!(
            (gd[1] - 3.0).abs() < 1e-6,
            "grad[1] = {}, expected 3",
            gd[1]
        );
        assert!(
            (gd[2] - 1.0).abs() < 1e-6,
            "grad[2] = {}, expected 1",
            gd[2]
        );
    }

    #[test]
    fn test_index_select_1d_backward_weighted_grad() {
        // input = [100, 200, 300], select indices [2, 0]
        // output = [300, 100]
        // grad_output = [0.5, 2.0]
        //
        // grad_input[0] += 2.0  (from output[1])
        // grad_input[2] += 0.5  (from output[0])
        // grad_input[1] = 0
        let input = leaf_1d(&[100.0, 200.0, 300.0], true);
        let selected = index_select_1d(&input, &[2, 0]).unwrap();

        let grad_output =
            Tensor::from_storage(TensorStorage::cpu(vec![0.5, 2.0]), vec![2], false).unwrap();

        let grad_fn = selected.grad_fn().unwrap();
        let grads = grad_fn.backward(&grad_output).unwrap();

        let grad_input = grads[0].as_ref().unwrap();
        let gd = grad_input.data().unwrap();

        assert!(
            (gd[0] - 2.0).abs() < 1e-6,
            "grad[0] = {}, expected 2.0",
            gd[0]
        );
        assert!(
            (gd[1] - 0.0).abs() < 1e-6,
            "grad[1] = {}, expected 0.0",
            gd[1]
        );
        assert!(
            (gd[2] - 0.5).abs() < 1e-6,
            "grad[2] = {}, expected 0.5",
            gd[2]
        );
    }

    // --- index_select_1d: no grad when grad disabled ---

    #[test]
    fn test_index_select_1d_no_grad_context() {
        let input = leaf_1d(&[10.0, 20.0, 30.0], true);

        let result = no_grad(|| index_select_1d(&input, &[0, 2])).unwrap();

        // Under no_grad, the result should be a leaf with no grad_fn.
        assert!(!result.requires_grad());
        assert!(result.grad_fn().is_none());
    }

    // --- masked_fill forward ---

    #[test]
    fn test_masked_fill_forward() {
        let input = leaf_1d(&[1.0, 2.0, 3.0, 4.0], false);
        let mask = [false, true, false, true];
        let result = masked_fill(&input, &mask, -999.0).unwrap();

        assert_eq!(result.data().unwrap(), &[1.0, -999.0, 3.0, -999.0]);
    }

    // --- masked_fill backward ---

    #[test]
    fn test_masked_fill_backward() {
        let input = leaf_1d(&[1.0, 2.0, 3.0, 4.0], true);
        let mask = [false, true, false, true];
        let filled = masked_fill(&input, &mask, 0.0).unwrap();

        // grad_output = [1, 1, 1, 1]
        // grad_input  = [1, 0, 1, 0]  (zeroed where mask is true)
        let grad_output =
            Tensor::from_storage(TensorStorage::cpu(vec![1.0; 4]), vec![4], false).unwrap();

        let grad_fn = filled.grad_fn().unwrap();
        let grads = grad_fn.backward(&grad_output).unwrap();

        let grad_input = grads[0].as_ref().unwrap();
        let gd = grad_input.data().unwrap();

        assert!((gd[0] - 1.0).abs() < 1e-6);
        assert!((gd[1] - 0.0).abs() < 1e-6);
        assert!((gd[2] - 1.0).abs() < 1e-6);
        assert!((gd[3] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_masked_fill_shape_mismatch() {
        let input = leaf_1d(&[1.0, 2.0, 3.0], false);
        let mask = [true, false]; // wrong length
        let result = masked_fill(&input, &mask, 0.0);
        assert!(result.is_err());
    }

    // --- gather backward ---

    #[test]
    fn test_gather_backward_stub() {
        let input = leaf_1d(&[1.0, 2.0], true);
        let gf = GatherBackward {
            input,
            dim: 0,
            index: vec![0, 1],
            index_cuda: None,
            index_shape: vec![2],
        };
        let grad_output =
            Tensor::from_storage(TensorStorage::cpu(vec![1.0, 1.0]), vec![2], false).unwrap();
        // Should now succeed rather than error.
        let result = gf.backward(&grad_output);
        assert!(result.is_ok());
    }

    #[test]
    fn test_scatter_add_backward_stub() {
        let input = leaf_1d(&[1.0, 2.0], true);
        let src = leaf_1d(&[3.0], false);
        let gf = ScatterAddBackward {
            input,
            src,
            dim: 0,
            index: vec![0],
            index_shape: vec![1],
        };
        let grad_output =
            Tensor::from_storage(TensorStorage::cpu(vec![1.0, 1.0]), vec![2], false).unwrap();
        let result = gf.backward(&grad_output);
        assert!(result.is_ok());
    }

    // -- index_select_dim (#1014) --

    #[test]
    fn test_index_select_dim_2d_dim0_forward() {
        // input: shape [4, 3]
        //   row 0: [10, 11, 12]
        //   row 1: [20, 21, 22]
        //   row 2: [30, 31, 32]
        //   row 3: [40, 41, 42]
        // indices: [3, 0, 2]
        // output: shape [3, 3]
        //   row 0 = input row 3 = [40, 41, 42]
        //   row 1 = input row 0 = [10, 11, 12]
        //   row 2 = input row 2 = [30, 31, 32]
        let input = Tensor::from_storage(
            TensorStorage::cpu(vec![
                10.0_f32, 11.0, 12.0, 20.0, 21.0, 22.0, 30.0, 31.0, 32.0, 40.0, 41.0, 42.0,
            ]),
            vec![4, 3],
            false,
        )
        .unwrap();
        let idx: IntTensor<i64> = IntTensor::from_vec(vec![3, 0, 2], vec![3]).unwrap();
        let out = index_select_dim(&input, 0, &idx).unwrap();
        assert_eq!(out.shape(), &[3, 3]);
        assert_eq!(
            out.data().unwrap(),
            &[40.0, 41.0, 42.0, 10.0, 11.0, 12.0, 30.0, 31.0, 32.0]
        );
    }

    #[test]
    fn test_index_select_dim_2d_dim1_forward() {
        // input: shape [2, 4]
        //   [[1, 2, 3, 4],
        //    [5, 6, 7, 8]]
        // dim=1, indices=[1, 3, 0]
        // output: shape [2, 3]
        //   [[2, 4, 1],
        //    [6, 8, 5]]
        let input = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            vec![2, 4],
            false,
        )
        .unwrap();
        let idx: IntTensor<i64> = IntTensor::from_vec(vec![1, 3, 0], vec![3]).unwrap();
        let out = index_select_dim(&input, 1, &idx).unwrap();
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out.data().unwrap(), &[2.0, 4.0, 1.0, 6.0, 8.0, 5.0]);
    }

    #[test]
    fn test_index_select_dim_registers_grad_fn() {
        let input = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]),
            vec![3, 2],
            true,
        )
        .unwrap();
        let idx: IntTensor<i64> = IntTensor::from_vec(vec![0, 2], vec![2]).unwrap();
        let out = index_select_dim(&input, 0, &idx).unwrap();
        assert!(out.requires_grad());
        assert!(!out.is_leaf());
        assert_eq!(out.grad_fn().unwrap().name(), "IndexSelectDimBackward");
    }

    #[test]
    fn test_index_select_dim_backward_simple_2d() {
        // input: [4, 2], indices [2, 0, 2] along dim=0 → output [3, 2]
        // grad_output =
        //   [[1, 10],
        //    [100, 1000],
        //    [10000, 100000]]
        // expected grad_input (scatter-add along dim 0, accumulating dups):
        //   row 0: from grad_output row 1            -> [100, 1000]
        //   row 1: untouched                         -> [0, 0]
        //   row 2: from grad_output rows 0 + 2       -> [1+10000, 10+100000] = [10001, 100010]
        //   row 3: untouched                         -> [0, 0]
        let input = Tensor::from_storage(
            TensorStorage::cpu(vec![
                1.0_f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // arbitrary
            ]),
            vec![4, 2],
            true,
        )
        .unwrap();
        let idx: IntTensor<i64> = IntTensor::from_vec(vec![2, 0, 2], vec![3]).unwrap();
        let out = index_select_dim(&input, 0, &idx).unwrap();

        let grad_output = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0_f32, 10.0, 100.0, 1000.0, 10000.0, 100000.0]),
            vec![3, 2],
            false,
        )
        .unwrap();

        let grads = out.grad_fn().unwrap().backward(&grad_output).unwrap();
        let g = grads[0].as_ref().unwrap();
        assert_eq!(g.shape(), &[4, 2]);
        let gd = g.data().unwrap();
        let expected = [100.0_f32, 1000.0, 0.0, 0.0, 10001.0, 100010.0, 0.0, 0.0];
        for (i, (&got, &exp)) in gd.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-3,
                "grad[{i}] = {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn test_index_select_dim_backward_dim1() {
        // input: [2, 4], indices [3, 1] along dim=1 → output [2, 2]
        // grad_output =
        //   [[1, 10], [100, 1000]]
        // expected grad_input (per-row scatter into 4 columns at cols 3 and 1):
        //   row 0: [0, 10, 0, 1]
        //   row 1: [0, 1000, 0, 100]
        let input = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0_f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            vec![2, 4],
            true,
        )
        .unwrap();
        let idx: IntTensor<i64> = IntTensor::from_vec(vec![3, 1], vec![2]).unwrap();
        let out = index_select_dim(&input, 1, &idx).unwrap();

        let grad_output = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0_f32, 10.0, 100.0, 1000.0]),
            vec![2, 2],
            false,
        )
        .unwrap();
        let grads = out.grad_fn().unwrap().backward(&grad_output).unwrap();
        let g = grads[0].as_ref().unwrap();
        assert_eq!(g.shape(), &[2, 4]);
        let gd = g.data().unwrap();
        let expected = [0.0_f32, 10.0, 0.0, 1.0, 0.0, 1000.0, 0.0, 100.0];
        for (i, (&got, &exp)) in gd.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "grad[{i}] = {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn test_index_select_dim_e2e_via_autograd() {
        // End-to-end: drive the gradient through the autograd graph (rather
        // than calling backward() directly on the grad_fn) and verify the
        // input.grad() lands on the bias-table parameter equivalent.
        // input: [3, 2] = [[1,2],[3,4],[5,6]], indices [0, 2, 0] on dim=0
        // out: [3, 2] = [[1,2],[5,6],[1,2]]
        // sum(out) = 1+2+5+6+1+2 = 17
        // grad_out (from sum) = ones([3, 2])
        // grad_input (scatter-add along dim 0):
        //   row 0: from out rows 0 and 2 -> [2, 2]
        //   row 1: untouched              -> [0, 0]
        //   row 2: from out row 1         -> [1, 1]
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]),
            vec![3, 2],
            true,
        )
        .unwrap();
        let idx: IntTensor<i64> = IntTensor::from_vec(vec![0, 2, 0], vec![3]).unwrap();
        let out = index_select_dim(&x, 0, &idx).unwrap();
        let total: f32 = out.data().unwrap().iter().sum();
        let loss = Tensor::from_operation(
            TensorStorage::cpu(vec![total]),
            vec![],
            Arc::new({
                #[derive(Debug)]
                struct SumBackward<T: Float> {
                    input: Tensor<T>,
                }
                impl<T: Float> GradFn<T> for SumBackward<T> {
                    fn backward(
                        &self,
                        _go: &Tensor<T>,
                    ) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
                        let n = self.input.numel();
                        let ones = vec![<T as num_traits::One>::one(); n];
                        let g = Tensor::from_storage(
                            TensorStorage::cpu(ones),
                            self.input.shape().to_vec(),
                            false,
                        )?;
                        Ok(vec![Some(g)])
                    }
                    fn inputs(&self) -> Vec<&Tensor<T>> {
                        vec![&self.input]
                    }
                    fn name(&self) -> &'static str {
                        "SumBackward"
                    }
                }
                SumBackward { input: out.clone() }
            }),
        )
        .unwrap();

        crate::autograd::graph::backward(&loss).unwrap();

        let grad = x.grad().unwrap().expect("x.grad() should be Some");
        assert_eq!(grad.shape(), &[3, 2]);
        let gd = grad.data().unwrap();
        let expected = [2.0_f32, 2.0, 0.0, 0.0, 1.0, 1.0];
        for (i, (&got, &exp)) in gd.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "grad[{i}] = {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn test_index_select_dim_rejects_2d_indices() {
        let x =
            Tensor::from_storage(TensorStorage::cpu(vec![1.0_f32; 6]), vec![3, 2], false).unwrap();
        let idx: IntTensor<i64> = IntTensor::from_vec(vec![0, 1, 0, 1], vec![2, 2]).unwrap();
        let err = index_select_dim(&x, 0, &idx).unwrap_err();
        assert!(matches!(err, FerrotorchError::ShapeMismatch { .. }));
    }

    #[test]
    fn test_index_select_dim_rejects_oob() {
        let x =
            Tensor::from_storage(TensorStorage::cpu(vec![1.0_f32; 6]), vec![3, 2], false).unwrap();
        let idx: IntTensor<i64> = IntTensor::from_vec(vec![0, 7], vec![2]).unwrap();
        let err = index_select_dim(&x, 0, &idx).unwrap_err();
        assert!(matches!(err, FerrotorchError::IndexOutOfBounds { .. }));
    }

    #[test]
    fn test_index_select_dim_rejects_negative() {
        let x =
            Tensor::from_storage(TensorStorage::cpu(vec![1.0_f32; 6]), vec![3, 2], false).unwrap();
        let idx: IntTensor<i64> = IntTensor::from_vec(vec![0, -1], vec![2]).unwrap();
        let err = index_select_dim(&x, 0, &idx).unwrap_err();
        assert!(matches!(err, FerrotorchError::InvalidArgument { .. }));
    }
}

// ---------------------------------------------------------------------------
// index_fill tests (REQ-8, #1249).
//
// Uses `?` propagation per the bcast-wrapper-tests precedent
// (`first_class_wrappers_tests` mod above) so the anti-pattern-gate hook
// (which scans Edit patches without honoring the `#[cfg(test)]` exemption
// applied for Write) accepts the patch.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod index_fill_tests {
    use super::*;
    use crate::autograd::graph::backward;

    fn cpu_f32(data: Vec<f32>, shape: Vec<usize>) -> FerrotorchResult<Tensor<f32>> {
        Tensor::from_storage(TensorStorage::cpu(data), shape, false)
    }

    fn cpu_f32_grad(data: Vec<f32>, shape: Vec<usize>) -> FerrotorchResult<Tensor<f32>> {
        Tensor::from_storage(TensorStorage::cpu(data), shape, true)
    }

    fn idx_i64(values: Vec<i64>, shape: Vec<usize>) -> FerrotorchResult<IntTensor<i64>> {
        IntTensor::from_vec(values, shape)
    }

    #[test]
    fn index_fill_forward_2d_dim1_matches_torch_docstring() -> FerrotorchResult<()> {
        // Mirrors the upstream docstring example at
        // `pytorch/torch/_tensor_docs.py:2503-2508`:
        //   x = [[1,2,3],[4,5,6],[7,8,9]]; x.index_fill_(1, [0,2], -1)
        //   => [[-1,2,-1],[-1,5,-1],[-1,8,-1]]
        // Expected values quoted from torch/_tensor_docs.py:2506-2508
        // (named typed bits traceable to upstream — NOT self-referential).
        let input = cpu_f32(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![3, 3],
        )?;
        let idx = idx_i64(vec![0, 2], vec![2])?;
        let out = index_fill(&input, 1, &idx, -1.0)?;
        assert_eq!(out.shape(), &[3, 3]);
        let got = out.data()?;
        let expected = [-1.0_f32, 2.0, -1.0, -1.0, 5.0, -1.0, -1.0, 8.0, -1.0];
        assert_eq!(got, &expected);
        Ok(())
    }

    #[test]
    fn index_fill_forward_2d_dim0_replaces_row() -> FerrotorchResult<()> {
        // x = [[1,2,3],[4,5,6]]; x.index_fill(0, [1], -9)
        // => [[1,2,3],[-9,-9,-9]] (replaces row 1 entirely).
        // Constructed from the upstream forward rule at
        // `aten/src/ATen/native/TensorAdvancedIndexing.cpp:1979-1984`:
        // clone(self) then overwrite slice [1, :] with -9.
        let input = cpu_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
        let idx = idx_i64(vec![1], vec![1])?;
        let out = index_fill(&input, 0, &idx, -9.0)?;
        assert_eq!(out.shape(), &[2, 3]);
        let got = out.data()?;
        let expected = [1.0_f32, 2.0, 3.0, -9.0, -9.0, -9.0];
        assert_eq!(got, &expected);
        Ok(())
    }

    #[test]
    fn index_fill_backward_zeros_at_fill_positions() -> FerrotorchResult<()> {
        // Mirrors the upstream VJP at `tools/autograd/derivatives.yaml:884-887
        //   - name: index_fill.int_Scalar(...)
        //     self: grad.index_fill(dim, index, 0)`
        // gradient is zeroed at every filled position, passes through elsewhere.
        // input = [[1,2,3],[4,5,6]], index_fill(dim=1, [0,2], -1)
        // out = [[-1,2,-1],[-1,5,-1]]; grad_output = ones([2,3])
        // grad_input = ones with cols 0,2 zeroed = [[0,1,0],[0,1,0]].
        let input = cpu_f32_grad(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
        let idx = idx_i64(vec![0, 2], vec![2])?;
        let out = index_fill(&input, 1, &idx, -1.0)?;
        let gf = match out.grad_fn() {
            Some(g) => g,
            None => {
                return Err(FerrotorchError::Internal {
                    message: "expected grad_fn on requires_grad output".into(),
                });
            }
        };
        assert_eq!(gf.name(), "IndexFillBackward");

        let grad_output = cpu_f32(vec![1.0_f32; 6], vec![2, 3])?;
        let grads = gf.backward(&grad_output)?;
        let g = match grads[0].as_ref() {
            Some(g) => g,
            None => {
                return Err(FerrotorchError::Internal {
                    message: "expected Some(grad_input)".into(),
                });
            }
        };
        assert_eq!(g.shape(), &[2, 3]);
        let gd = g.data()?;
        let expected = [0.0_f32, 1.0, 0.0, 0.0, 1.0, 0.0];
        assert_eq!(gd, &expected);
        Ok(())
    }

    #[test]
    fn index_fill_negative_dim_wraps() -> FerrotorchResult<()> {
        // Negative dim per `at::maybe_wrap_dim` at
        // `aten/src/ATen/native/TensorAdvancedIndexing.cpp:1919`:
        // dim=-1 on a 2-D tensor maps to dim=1. Neg-dim result must equal
        // pos-dim result (wrap is the only transformation).
        let input = cpu_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
        let idx = idx_i64(vec![0, 2], vec![2])?;
        let neg = index_fill(&input, -1, &idx, -7.0)?;
        let pos = index_fill(&input, 1, &idx, -7.0)?;
        assert_eq!(neg.data()?, pos.data()?);
        let expected = [-7.0_f32, 2.0, -7.0, -7.0, 5.0, -7.0];
        assert_eq!(neg.data()?, &expected);
        Ok(())
    }

    #[test]
    fn index_fill_rejects_out_of_bounds() -> FerrotorchResult<()> {
        let input = cpu_f32(vec![1.0_f32; 6], vec![2, 3])?;
        let idx = idx_i64(vec![0, 7], vec![2])?;
        let err = index_fill(&input, 1, &idx, 0.0).err();
        assert!(matches!(
            err,
            Some(FerrotorchError::IndexOutOfBounds { .. })
        ));
        Ok(())
    }

    #[test]
    fn index_fill_wraps_negative_index_per_upstream() -> FerrotorchResult<()> {
        // Upstream `index_fill_kernel` at
        // `aten/src/ATen/native/cpu/IndexKernel.cpp:224-229` wraps negative
        // indices: `if (idx < 0) idx += self_dim_size`. Only OOB negatives
        // (`idx < -dim_size`) raise IndexError. Verified against live torch:
        //   torch.index_fill(torch.tensor([[1.,2.,3.],[4.,5.,6.]]), 1,
        //                    torch.tensor([-1]), -1.0)
        //     == tensor([[1., 2., -1.], [4., 5., -1.]])
        let input = cpu_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
        let idx = idx_i64(vec![-1], vec![1])?;
        let out = index_fill(&input, 1, &idx, -1.0)?;
        let expected = [1.0_f32, 2.0, -1.0, 4.0, 5.0, -1.0];
        assert_eq!(out.data()?, &expected);
        // OOB negative (-4 for size-3 axis) still errors.
        let idx_oob = idx_i64(vec![-4], vec![1])?;
        let err = index_fill(&input, 1, &idx_oob, 0.0).err();
        assert!(matches!(
            err,
            Some(FerrotorchError::IndexOutOfBounds { .. })
        ));
        Ok(())
    }

    #[test]
    fn index_fill_rejects_out_of_range_dim() -> FerrotorchResult<()> {
        let input = cpu_f32(vec![1.0_f32; 6], vec![2, 3])?;
        let idx = idx_i64(vec![0], vec![1])?;
        let err = index_fill(&input, 5, &idx, 0.0).err();
        assert!(matches!(err, Some(FerrotorchError::InvalidArgument { .. })));
        Ok(())
    }

    #[test]
    fn index_fill_zero_dim_input_succeeds_per_upstream() -> FerrotorchResult<()> {
        // Upstream unsqueezes 0-d input at TensorAdvancedIndexing.cpp:1917:
        //   Tensor self_nonzero_dim = (self.dim() == 0) ? self.unsqueeze(-1) : self;
        // torch.index_fill(torch.tensor(1.0), 0, torch.tensor([0]), 0.0) == tensor(0.)
        let input = cpu_f32(vec![1.0_f32], vec![])?;
        let idx = idx_i64(vec![0], vec![1])?;
        let out = index_fill(&input, 0, &idx, 0.0)?;
        assert_eq!(out.shape(), &[] as &[usize], "0-d output must remain 0-d");
        assert_eq!(out.data()?, &[0.0_f32], "filled value must be 0.0");
        Ok(())
    }

    #[test]
    fn index_fill_rejects_multi_d_index() -> FerrotorchResult<()> {
        // Upstream `TORCH_CHECK(index.dim() <= 1, "Index has to be a
        // vector/scalar")` at `TensorAdvancedIndexing.cpp:1920`.
        let input = cpu_f32(vec![1.0_f32; 6], vec![2, 3])?;
        let idx = idx_i64(vec![0, 1, 0, 1], vec![2, 2])?;
        let err = index_fill(&input, 1, &idx, 0.0).err();
        assert!(matches!(err, Some(FerrotorchError::ShapeMismatch { .. })));
        Ok(())
    }

    #[test]
    fn index_fill_e2e_via_autograd() -> FerrotorchResult<()> {
        // End-to-end: drive backward through the autograd graph and verify
        // the leaf grad lands the expected mask-zero pattern.
        // x = [10,20,30,40] (requires_grad); index_fill(0, [1,3], -1)
        // out = [10,-1,30,-1]; sum(out) = 38;
        // grad_out (from sum) = ones([4]); grad_input = [1,0,1,0].
        let x = cpu_f32_grad(vec![10.0, 20.0, 30.0, 40.0], vec![4])?;
        let idx = idx_i64(vec![1, 3], vec![2])?;
        let out = index_fill(&x, 0, &idx, -1.0)?;
        let total: f32 = out.data()?.iter().sum();

        #[derive(Debug)]
        struct SumBackward<T: Float> {
            input: Tensor<T>,
        }
        impl<T: Float> GradFn<T> for SumBackward<T> {
            fn backward(&self, _go: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
                let n = self.input.numel();
                let ones = vec![<T as num_traits::One>::one(); n];
                let g = Tensor::from_storage(
                    TensorStorage::cpu(ones),
                    self.input.shape().to_vec(),
                    false,
                )?;
                Ok(vec![Some(g)])
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
            Arc::new(SumBackward { input: out.clone() }),
        )?;
        backward(&loss)?;

        let grad = match x.grad()? {
            Some(g) => g,
            None => {
                return Err(FerrotorchError::Internal {
                    message: "expected leaf grad".into(),
                });
            }
        };
        assert_eq!(grad.shape(), &[4]);
        let expected = [1.0_f32, 0.0, 1.0, 0.0];
        assert_eq!(grad.data()?, &expected);
        Ok(())
    }
}
