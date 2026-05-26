//! Backward functions for indexing operations.
//!
//! Implements `GradFn` for:
//! - `index_select` (1D) — selects elements along an axis by integer indices
//! - `masked_fill` — fills elements where a boolean mask is true
//! - `gather` — gathers elements along an axis (N-D)
//! - `scatter` — scatters src values into input along an axis
//! - `scatter_add` — scatter with addition
//! - `where_cond` — ternary selection

use std::sync::Arc;

use crate::autograd::no_grad::is_grad_enabled;
use crate::device::Device;
use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::gpu_dispatch::gpu_backend;
use crate::storage::TensorStorage;
use crate::tensor::{GradFn, Tensor};

use crate::bool_tensor::BoolTensor;
use crate::int_tensor::{IntElement, IntTensor};

/// Upload a CPU `&[f32]` slice to a GPU buffer on the given device ordinal.
fn upload_f32_to_gpu(
    data: &[f32],
    ordinal: usize,
) -> FerrotorchResult<crate::gpu_dispatch::GpuBufferHandle> {
    let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
    // SAFETY: `data: &[f32]` is borrowed for the duration of this function
    // and is fully initialized (f32 has no padding, no niches). Reading its
    // bytes as &[u8] of length `data.len() * 4` (== `data.len() *
    // size_of::<f32>()`) is sound and matches the actual byte size of the
    // underlying allocation; the resulting slice does not outlive `data`.
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    backend.cpu_to_gpu(bytes, crate::dtype::DType::F32, ordinal)
}

/// For `ScatterBackward` grad_input: build a flat boolean mask (1.0 at positions
/// overwritten by scatter, 0.0 elsewhere) in the input's flat space.
fn scatter_write_mask(
    index: &[usize],
    index_shape: &[usize],
    input_shape: &[usize],
    dim: usize,
) -> Vec<f32> {
    let input_numel: usize = input_shape.iter().product();
    let index_numel: usize = index_shape.iter().product();
    let mut mask = vec![0.0f32; input_numel];
    let ndim = input_shape.len();
    let mut coords = vec![0usize; ndim];
    for i in 0..index_numel {
        let idx_val = index[i];
        let mut dst_coords = coords.clone();
        dst_coords[dim] = idx_val;
        let dst_flat = flat_index(&dst_coords, input_shape);
        mask[dst_flat] = 1.0;
        if i + 1 < index_numel {
            increment_coords(&mut coords, index_shape);
        }
    }
    mask
}

/// For `GatherBackward`: compute flat destination indices (into input space)
/// for each element of the index tensor — i.e. the same flat positions that
/// `gather` read from, so scatter-add routes gradients back there.
fn gather_dst_flat_indices(
    index: &[usize],
    index_shape: &[usize],
    input_shape: &[usize],
    dim: usize,
) -> Vec<f32> {
    let ndim = input_shape.len();
    let index_numel: usize = index_shape.iter().product();
    let mut result = Vec::with_capacity(index_numel);
    let mut coords = vec![0usize; ndim];
    for i in 0..index_numel {
        let idx_val = index[i];
        // The destination in input space: same coords as the index position
        // but with `dim` replaced by idx_val.
        let mut dst_coords = coords.clone();
        dst_coords[dim] = idx_val;
        result.push(flat_index(&dst_coords, input_shape) as f32);
        if i + 1 < index_numel {
            increment_coords(&mut coords, index_shape);
        }
    }
    result
}

/// For scatter/scatter_add backward grad_src: the source gradient comes from
/// gathering grad_output at the index-mapped positions in input space — the
/// inverse of what scatter wrote. Returns flat indices into grad_output space.
fn scatter_src_flat_indices(
    index: &[usize],
    index_shape: &[usize],
    input_shape: &[usize],
    dim: usize,
) -> Vec<f32> {
    // Same computation as gather_dst_flat_indices: for each position in the
    // index tensor, the source flat index in grad_output (= input) is the same
    // flat location that was overwritten during scatter.
    gather_dst_flat_indices(index, index_shape, input_shape, dim)
}

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
            // GPU path: scatter-add via GPU kernel.
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let ordinal = match grad_output.device() {
                Device::Cuda(o) => o,
                _ => unreachable!(),
            };
            let indices_f32: Vec<f32> = self.indices.iter().map(|&i| i as f32).collect();
            let idx_handle = upload_f32_to_gpu(&indices_f32, ordinal)?;
            let result_handle =
                backend.scatter_add_1d_f32(grad_output.gpu_handle()?, &idx_handle, input_len)?;
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
        // GPU path: gather via GPU kernel.
        let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let ordinal = match input.device() {
            Device::Cuda(o) => o,
            _ => unreachable!(),
        };
        let indices_f32: Vec<f32> = indices.iter().map(|&i| i as f32).collect();
        let idx_handle = upload_f32_to_gpu(&indices_f32, ordinal)?;
        let result_handle = backend.index_select_1d_f32(input.gpu_handle()?, &idx_handle)?;
        let storage = TensorStorage::gpu(result_handle);

        if input.requires_grad() && is_grad_enabled() {
            let grad_fn = Arc::new(IndexSelectBackward {
                input: input.clone(),
                indices: indices.to_vec(),
            });
            Tensor::from_operation(storage, output_shape, grad_fn)
        } else {
            Tensor::from_storage(storage, output_shape, false)
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

        // GPU-resident path (crosslink #1187 Phase 3d): grad_input = masked_fill(
        // grad_output, mask, 0) — zero the gradient where the forward filled a
        // constant. Both grad and the bool mask stay on the device; the resident
        // `masked_fill_dt` kernel is dtype-generic (f32/f64/bf16/f16). NO mask
        // host crossing, NO float-mask upload.
        if grad_output.is_cuda() && self.mask.is_cuda() {
            if grad_output.device() != self.mask.device() {
                return Err(FerrotorchError::DeviceMismatch {
                    expected: grad_output.device(),
                    got: self.mask.device(),
                });
            }
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let result_handle =
                backend.masked_fill_dt(grad_output.gpu_handle()?, self.mask.gpu_handle()?, 0.0)?;
            let grad_tensor = Tensor::from_storage(
                TensorStorage::gpu(result_handle),
                self.input.shape().to_vec(),
                false,
            )?;
            Ok(vec![Some(grad_tensor)])
        } else {
            // CPU path: direct mask zeroing. `self.mask.data()?` borrows the host
            // bool slice (errors if the mask is GPU-resident while grad is on
            // host — the correct device-mismatch signal).
            let go_data = grad_output.data()?;
            let mask_h = self.mask.data()?;
            let mut grad_input: Vec<T> = go_data.to_vec();
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

    let output_shape = input.shape().to_vec();

    if input.is_cuda() {
        // GPU path: masked-fill via GPU kernel.
        let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let ordinal = match input.device() {
            Device::Cuda(o) => o,
            _ => unreachable!(),
        };
        let mask_f32: Vec<f32> = mask.iter().map(|&m| if m { 1.0 } else { 0.0 }).collect();
        let mask_handle = upload_f32_to_gpu(&mask_f32, ordinal)?;
        // value must be f32 for the GPU kernel.
        let value_f32: f32 = num_traits::ToPrimitive::to_f32(&value).unwrap_or(0.0);
        let result_handle =
            backend.masked_fill_f32(input.gpu_handle()?, &mask_handle, value_f32)?;
        let storage = TensorStorage::gpu(result_handle);

        if input.requires_grad() && is_grad_enabled() {
            // This entry point inherently has a host `&[bool]`; wrap it as a CPU
            // BoolTensor for storage. The backward struct now holds a BoolTensor
            // (CPU here; the resident `masked_fill_bt` path stores a GPU one).
            let grad_fn = Arc::new(MaskedFillBackward {
                input: input.clone(),
                mask: BoolTensor::from_slice(mask, &output_shape)?,
            });
            Tensor::from_operation(storage, output_shape, grad_fn)
        } else {
            Tensor::from_storage(storage, output_shape, false)
        }
    } else {
        // CPU path: direct masked fill.
        let input_data = input.data()?;
        let output_data: Vec<T> = input_data
            .iter()
            .zip(mask.iter())
            .map(|(&x, &m)| if m { value } else { x })
            .collect();

        if input.requires_grad() && is_grad_enabled() {
            let grad_fn = Arc::new(MaskedFillBackward {
                input: input.clone(),
                mask: BoolTensor::from_slice(mask, &output_shape)?,
            });
            Tensor::from_operation(TensorStorage::cpu(output_data), output_shape, grad_fn)
        } else {
            Tensor::from_storage(TensorStorage::cpu(output_data), output_shape, false)
        }
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
    /// The shape of the index tensor.
    pub index_shape: Vec<usize>,
}

impl<T: Float> GradFn<T> for GatherBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !is_grad_enabled() {
            return Ok(vec![None]);
        }

        let input_shape = self.input.shape();
        let input_numel: usize = input_shape.iter().product();

        // §3 GPU-native path: flatten grad_output, compute flat dst indices CPU-side
        // (the index tensor is always CPU-resident), scatter-add via existing 1-D kernel.
        if grad_output.is_cuda() {
            let ordinal = match grad_output.device() {
                Device::Cuda(o) => o,
                _ => unreachable!(),
            };
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let dst_indices =
                gather_dst_flat_indices(&self.index, &self.index_shape, input_shape, self.dim);
            let idx_handle = upload_f32_to_gpu(&dst_indices, ordinal)?;
            // scatter_add_1d_f32 treats grad_output as a flat 1-D buffer and
            // accumulates each element at its flat destination index.
            let result_handle =
                backend.scatter_add_1d_f32(grad_output.gpu_handle()?, &idx_handle, input_numel)?;
            let grad_tensor = Tensor::from_storage(
                TensorStorage::gpu(result_handle),
                input_shape.to_vec(),
                false,
            )?;
            return Ok(vec![Some(grad_tensor)]);
        }

        let go_data = grad_output.data_vec()?;
        let ndim = input_shape.len();
        let index_numel: usize = self.index_shape.iter().product();

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

        let input_shape = self.input.shape();
        let index_numel: usize = self.index_shape.iter().product();

        // §3 GPU-native path:
        //   grad_input = masked_zero_f32(grad_output, write_mask)
        //     — zeros at every position scatter wrote to (those positions came from src).
        //   grad_src   = index_select_1d_f32(flat(grad_output), scatter_src_indices)
        //     — gathers from the flat positions that scatter wrote into.
        if grad_output.is_cuda() {
            let ordinal = match grad_output.device() {
                Device::Cuda(o) => o,
                _ => unreachable!(),
            };
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;

            let grad_input = if self.input.requires_grad() {
                // Build a 1.0/0.0 mask for the written positions, upload, zero them out.
                let mask_f32 =
                    scatter_write_mask(&self.index, &self.index_shape, input_shape, self.dim);
                let mask_handle = upload_f32_to_gpu(&mask_f32, ordinal)?;
                let result_h = backend.masked_zero_f32(grad_output.gpu_handle()?, &mask_handle)?;
                Some(Tensor::from_storage(
                    TensorStorage::gpu(result_h),
                    input_shape.to_vec(),
                    false,
                )?)
            } else {
                None
            };

            let grad_src = if self.src.requires_grad() {
                // Gather grad_output at the flat positions that scatter wrote into.
                let src_indices =
                    scatter_src_flat_indices(&self.index, &self.index_shape, input_shape, self.dim);
                let idx_handle = upload_f32_to_gpu(&src_indices, ordinal)?;
                let result_h =
                    backend.index_select_1d_f32(grad_output.gpu_handle()?, &idx_handle)?;
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

        let input_shape = self.input.shape();
        let index_numel: usize = self.index_shape.iter().product();

        // §3 GPU-native path:
        //   grad_input = grad_output  (identity — addition passes grad through unchanged).
        //   grad_src   = index_select_1d_f32(flat(grad_output), scatter_src_indices)
        //     — gathers the positions that scatter_add accumulated into.
        if grad_output.is_cuda() {
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
                let src_indices =
                    scatter_src_flat_indices(&self.index, &self.index_shape, input_shape, self.dim);
                let idx_handle = upload_f32_to_gpu(&src_indices, ordinal)?;
                let result_h =
                    backend.index_select_1d_f32(grad_output.gpu_handle()?, &idx_handle)?;
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

        if grad_output.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "scatter_add backward",
            });
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
        let input_numel: usize = input_shape.iter().product();

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
        let value_f64 = num_traits::ToPrimitive::to_f64(&value).unwrap_or(0.0);
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

    // CPU (or mixed-residency) path: delegate to the host `&[bool]` variant,
    // which itself handles a CUDA `input` with a host mask (legacy float-mask
    // upload). `mask.data()?` errors if the mask is on GPU but input is not,
    // which is the correct device-mismatch signal.
    masked_fill(input, mask.data()?, value)
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
        let input_numel: usize = input_shape.iter().product();
        let dim = self.dim;
        let outer: usize = input_shape[..dim].iter().product();
        let inner: usize = input_shape[dim + 1..].iter().product();
        let in_dim_size = input_shape[dim];
        let out_dim_size = self.indices.len();

        // GPU path: scatter-add via the existing 1-D kernel. We compute the
        // flat destination index in input-space for every element of
        // grad_output (which is dense, in C-order, with shape replacing
        // `dim` by `out_dim_size`), upload, and reuse
        // `scatter_add_1d_{f32,f64}`. f64 inputs now reach this path
        // via #1098 (CUDA forward for `index_select_dim`); fall back to
        // CPU only for non-{f32,f64} floats so we never silently demote
        // an in-graph CUDA buffer.
        if grad_output.is_cuda() {
            use std::any::TypeId;
            let is_t_f32 = TypeId::of::<T>() == TypeId::of::<f32>();
            let is_t_f64 = TypeId::of::<T>() == TypeId::of::<f64>();
            if is_t_f32 || is_t_f64 {
                let ordinal = match grad_output.device() {
                    Device::Cuda(o) => o,
                    _ => unreachable!(),
                };
                let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;

                // Build flat destination indices, one per grad_output element.
                //
                // For grad_output with C-contiguous layout
                //   [outer, out_dim_size, inner]
                // and target buffer (= input space) with layout
                //   [outer, in_dim_size, inner]
                // a grad_output element at flat position
                //   o * out_dim_size * inner + i * inner + k
                // maps to flat dst position
                //   o * in_dim_size * inner + indices[i] * inner + k
                let go_numel = outer * out_dim_size * inner;
                let mut dst_indices: Vec<f32> = Vec::with_capacity(go_numel);
                for o in 0..outer {
                    for i in 0..out_dim_size {
                        let dst_i = self.indices[i];
                        let base = o * in_dim_size * inner + dst_i * inner;
                        for k in 0..inner {
                            dst_indices.push((base + k) as f32);
                        }
                    }
                }

                let idx_handle = upload_f32_to_gpu(&dst_indices, ordinal)?;
                let result_handle = if is_t_f32 {
                    backend.scatter_add_1d_f32(
                        grad_output.gpu_handle()?,
                        &idx_handle,
                        input_numel,
                    )?
                } else {
                    backend.scatter_add_1d_f64(
                        grad_output.gpu_handle()?,
                        &idx_handle,
                        input_numel,
                    )?
                };
                let grad_tensor = Tensor::from_storage(
                    TensorStorage::gpu(result_handle),
                    input_shape.to_vec(),
                    false,
                )?;
                return Ok(vec![Some(grad_tensor)]);
            }
            // Unsupported float dtype on CUDA: surface explicitly.
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "IndexSelectDimBackward",
            });
        }

        // CPU path: scatter-add directly.
        let go_data = grad_output.data_vec()?;
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
    if indices.ndim() != 1 {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "index_select_dim: indices must be 1-D, got shape {:?}",
                indices.shape()
            ),
        });
    }

    let in_dim_size = input_shape[dim];
    // Validate + widen indices.
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

    // Compute output: same shape but axis `dim` replaced by indices.len().
    let mut output_shape = input_shape.to_vec();
    output_shape[dim] = idx_usize.len();

    let outer: usize = input_shape[..dim].iter().product();
    let inner: usize = input_shape[dim + 1..].iter().product();
    let out_dim_size = idx_usize.len();

    // GPU path: route via TypeId to the f32/f64 device-resident gather
    // kernel. The output buffer is allocated on-device; no host
    // round-trip. Indices are f32-encoded (backend convention shared
    // with `index_select_1d_f32`, `scatter_add_1d_f32`, etc.).
    if input.is_cuda() {
        use std::any::TypeId;
        let is_t_f32 = TypeId::of::<T>() == TypeId::of::<f32>();
        let is_t_f64 = TypeId::of::<T>() == TypeId::of::<f64>();
        if is_t_f32 || is_t_f64 {
            let ordinal = match input.device() {
                Device::Cuda(o) => o,
                _ => unreachable!("input.is_cuda() but device() not Cuda"),
            };
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            // Upload indices as f32 (the established encoding for
            // index buffers across the GPU dispatch surface).
            let indices_f32: Vec<f32> = idx_usize.iter().map(|&u| u as f32).collect();
            let idx_handle = upload_f32_to_gpu(&indices_f32, ordinal)?;

            let result_handle = if is_t_f32 {
                backend.index_select_dim_f32(
                    input.gpu_handle()?,
                    &idx_handle,
                    outer,
                    in_dim_size,
                    out_dim_size,
                    inner,
                )?
            } else {
                backend.index_select_dim_f64(
                    input.gpu_handle()?,
                    &idx_handle,
                    outer,
                    in_dim_size,
                    out_dim_size,
                    inner,
                )?
            };

            let storage = TensorStorage::gpu(result_handle);
            return if input.requires_grad() && is_grad_enabled() {
                let grad_fn = Arc::new(IndexSelectDimBackward {
                    input: input.clone(),
                    dim,
                    indices: idx_usize,
                });
                Tensor::from_operation(storage, output_shape, grad_fn)
            } else {
                Tensor::from_storage(storage, output_shape, false)
            };
        }
        // Non-f32/f64 floats (e.g., bf16) still surface explicit
        // NotImplementedOnCuda — preserves the "no silent fallback"
        // contract for unsupported dtypes.
        return Err(FerrotorchError::NotImplementedOnCuda {
            op: "index_select_dim",
        });
    }

    // CPU path: dense memcpy along axis.
    let out_numel: usize = output_shape.iter().product();
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
        let outer: usize = input_shape[..dim].iter().product();
        let inner: usize = input_shape[dim + 1..].iter().product();
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
    let outer: usize = input_shape[..dim_usize].iter().product();
    let inner: usize = input_shape[dim_usize + 1..].iter().product();
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

/// Broadcast a CPU-resident [`BoolTensor`] to `out_shape` using NumPy rules,
/// returning a new contiguous CPU `BoolTensor`. Errors if the mask is GPU-
/// resident (no resident broadcast kernel exists for `DType::Bool`; the
/// runner-side production consumer feeds CPU tensors). Used by the
/// broadcasting wrappers `masked_fill_bcast`, `masked_select_bcast`, and
/// `where_cond_bcast` below to mirror PyTorch's `expand_outplace` step.
fn broadcast_bool_tensor(mask: &BoolTensor, out_shape: &[usize]) -> FerrotorchResult<BoolTensor> {
    if mask.shape() == out_shape {
        return Ok(mask.clone());
    }
    if mask.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda {
            op: "broadcast_bool_tensor",
        });
    }
    let in_data = mask.data()?;
    let in_shape: Vec<usize> = mask.shape().to_vec();
    let out_numel: usize = if out_shape.is_empty() {
        1
    } else {
        out_shape.iter().product()
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
    if input.shape() == mask.shape() {
        return crate::ops::indexing::masked_select(input, mask);
    }
    let common = crate::shape::broadcast_shapes(input.shape(), mask.shape())?;
    let input_b = crate::grad_fns::shape::expand(input, &common)?;
    let mask_b = broadcast_bool_tensor(mask, &common)?;
    crate::ops::indexing::masked_select(&input_b, &mask_b)
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
        return crate::ops::indexing::where_cond_bt(cond, x, y);
    }
    // 3-way broadcast via two pairwise applications.
    let xy_common = crate::shape::broadcast_shapes(x.shape(), y.shape())?;
    let common = crate::shape::broadcast_shapes(cond.shape(), &xy_common)?;
    let cond_b = broadcast_bool_tensor(cond, &common)?;
    let x_b = crate::grad_fns::shape::expand(x, &common)?;
    let y_b = crate::grad_fns::shape::expand(y, &common)?;
    crate::ops::indexing::where_cond_bt(&cond_b, &x_b, &y_b)
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
