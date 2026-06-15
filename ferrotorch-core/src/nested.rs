//! `NestedTensor` and `PackedNestedTensor` — ragged (jagged) tensors that
//! mirror `torch.nested.nested_tensor` (`aten/src/ATen/native/nested/`) +
//! the jagged-layout NJT (`torch/nested/_internal/nested_tensor.py`).
//!
//! ## REQ status (per `.design/ferrotorch-core/nested.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (NestedTensor::new) | SHIPPED | `NestedTensor::new` validates ndim + non-ragged shape parity + single-device invariant (CORE-070/#1764); consumer `lib.rs` `pub use nested::{NestedTensor, ...}` — R-DEFER-1 S5 grandfathering (#806, #291) |
//! | REQ-2 (accessors) | SHIPPED | `num_components`, `ragged_dim`, `tensors`, `ndim`, `consistent_shape`, `ragged_lengths`; consumer `lib.rs` re-export + internal GPU fast-path uses |
//! | REQ-3 (to_padded) | SHIPPED | `to_padded`: differentiable cat/unsqueeze composition when grad is tracked (CORE-066/#1760), GPU fast path `try_to_padded_gpu`, CPU logical-view path (`data_vec`, CORE-070/#1764); consumer `lib.rs` re-export — R-DEFER-1 S5 grandfathering |
//! | REQ-4 (from_padded) | SHIPPED | `from_padded`: differentiable narrow→contiguous→reshape when the source tracks grads (CORE-066/#1760), GPU fast path `try_from_padded_gpu`, CPU logical-view path; consumer `lib.rs` re-export |
//! | REQ-5 (nested SDPA) | SHIPPED | `pub fn nested_scaled_dot_product_attention<T: Float>`: differentiable `attention_component_composite` for grad-tracking inputs (CORE-066/#1760) and for CUDA components the flash kernel declines (CORE-067/#1761); flash dispatch `try_flash_attention_gpu_component`; consumer `lib.rs` re-export |
//! | REQ-6 (PackedNestedTensor) | SHIPPED | `pub struct PackedNestedTensor<T: Float>` with stored `lengths` (CORE-069/#1763); every constructor routes through `validate_packed_layout` (CORE-068/#1762); `mean_per_component` NaN on empty (CORE-071/#1765); `from_nested` rejects grad-tracking components loudly (R-LOUD-3); consumer `lib.rs` re-export — R-DEFER-1 S5 grandfathering (#291) |
//! | REQ-7 (structured errors) | SHIPPED | `InvalidArgument`/`ShapeMismatch`/`DeviceMismatch` at multiple sites; no `panic!` in production paths; consumers propagate via `?` |

use crate::device::Device;
use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::storage::TensorStorage;
use crate::tensor::Tensor;

/// A nested (ragged) tensor — a collection of tensors with differing sizes
/// along one dimension (the "ragged" dimension).
///
/// This is the ferrotorch equivalent of PyTorch's `torch.nested.nested_tensor`.
/// Each component tensor may have a different size along the ragged dimension,
/// but all other dimensions must match.
///
/// # Example
///
/// A batch of sequences with different lengths:
///
/// ```text
/// NestedTensor {
///     tensors: [
///         Tensor([3, 8]),  // sequence length 3, hidden dim 8
///         Tensor([5, 8]),  // sequence length 5, hidden dim 8
///         Tensor([2, 8]),  // sequence length 2, hidden dim 8
///     ],
///     ragged_dim: 0,       // dimension 0 varies across components
/// }
/// ```
#[derive(Debug, Clone)]
pub struct NestedTensor<T: Float> {
    /// The component tensors. All must have the same number of dimensions
    /// and identical sizes on every axis except `ragged_dim`.
    tensors: Vec<Tensor<T>>,
    /// Which dimension is ragged (varies in length across components).
    ragged_dim: usize,
}

impl<T: Float> NestedTensor<T> {
    /// Create a nested tensor from a list of component tensors.
    ///
    /// All tensors must have the same number of dimensions, and identical sizes
    /// on every axis except `ragged_dim`.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `tensors` is empty
    /// - Tensors have differing numbers of dimensions
    /// - Tensors have mismatched sizes on non-ragged dimensions
    /// - Tensors live on different devices (CORE-070 / #1764 — a mixed
    ///   CPU/CUDA component list would make every later op fail with a
    ///   confusing data-access error; torch's `nested_tensor` likewise
    ///   places all components on one device)
    /// - `ragged_dim` is out of range
    pub fn new(tensors: Vec<Tensor<T>>, ragged_dim: usize) -> FerrotorchResult<Self> {
        if tensors.is_empty() {
            return Err(FerrotorchError::InvalidArgument {
                message: "NestedTensor requires at least one component tensor".into(),
            });
        }

        let ndim = tensors[0].ndim();
        if ragged_dim >= ndim {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("ragged_dim {ragged_dim} out of range for {ndim}-D tensors"),
            });
        }

        // CORE-070 (#1764): single-device invariant. Enforced at
        // construction so mixed-device lists fail HERE with a structured
        // DeviceMismatch instead of later inside to_padded/attention with
        // an opaque GPU-data-access error.
        let device = tensors[0].device();
        for t in tensors.iter().skip(1) {
            if t.device() != device {
                return Err(FerrotorchError::DeviceMismatch {
                    expected: device,
                    got: t.device(),
                });
            }
        }

        for (i, t) in tensors.iter().enumerate().skip(1) {
            if t.ndim() != ndim {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "tensor {} has {} dims but tensor 0 has {} dims",
                        i,
                        t.ndim(),
                        ndim
                    ),
                });
            }
            for d in 0..ndim {
                if d != ragged_dim && t.shape()[d] != tensors[0].shape()[d] {
                    return Err(FerrotorchError::ShapeMismatch {
                        message: format!(
                            "tensor {} has size {} on dim {} but tensor 0 has size {} \
                             (only dim {} may differ)",
                            i,
                            t.shape()[d],
                            d,
                            tensors[0].shape()[d],
                            ragged_dim,
                        ),
                    });
                }
            }
        }

        Ok(Self {
            tensors,
            ragged_dim,
        })
    }

    /// Number of component tensors.
    #[inline]
    pub fn num_components(&self) -> usize {
        self.tensors.len()
    }

    /// The ragged dimension index.
    #[inline]
    pub fn ragged_dim(&self) -> usize {
        self.ragged_dim
    }

    /// References to the component tensors.
    #[inline]
    pub fn tensors(&self) -> &[Tensor<T>] {
        &self.tensors
    }

    /// Number of dimensions of each component tensor.
    #[inline]
    pub fn ndim(&self) -> usize {
        self.tensors[0].ndim()
    }

    /// The size of non-ragged dimensions (taken from the first component).
    pub fn consistent_shape(&self) -> Vec<usize> {
        self.tensors[0].shape().to_vec()
    }

    /// The lengths along the ragged dimension for each component.
    pub fn ragged_lengths(&self) -> Vec<usize> {
        self.tensors
            .iter()
            .map(|t| t.shape()[self.ragged_dim])
            .collect()
    }

    /// Convert to a padded dense tensor.
    ///
    /// Pads each component along the ragged dimension to the maximum length,
    /// filling missing positions with `pad_value`. The result has an extra
    /// leading batch dimension.
    ///
    /// For a nested tensor with components of shape `[L_i, D]` and
    /// `ragged_dim=0`, the output is `[batch, max_L, D]`.
    ///
    /// # GPU dispatch (P4 of #806)
    ///
    /// When every component is on the same CUDA device and `T` is `f32`
    /// or `f64`, the padded tensor is materialized **entirely on-device**:
    ///
    /// 1. Allocate the output buffer pre-filled with `pad_value` via the
    ///    existing `fill_f{32,64}` GPU primitive.
    /// 2. For each component, dispatch `strided_scatter_f{32,64}` to write
    ///    its values into the corresponding slice of the output buffer.
    ///
    /// This composes from the existing GPU primitives (`fill`, the
    /// `strided_copy`/`strided_scatter` cluster from #802 / CL-496); no new
    /// `GpuBackend` trait surface is added. PyTorch parity:
    /// `torch.nested.to_padded_tensor` on a CUDA nested tensor produces a
    /// CUDA padded tensor without host bounce.
    ///
    /// Mixed-device or unsupported-dtype components fall through to the
    /// CPU path which materializes via `tensor.data()?` (so callers get
    /// `GpuTensorNotAccessible` rather than silent corruption).
    pub fn to_padded(&self, pad_value: T) -> FerrotorchResult<Tensor<T>> {
        let batch = self.tensors.len();
        let ndim = self.ndim();
        let max_len = self
            .tensors
            .iter()
            .map(|t| t.shape()[self.ragged_dim])
            .max()
            .unwrap_or(0);

        // Build output shape: [batch, d0, d1, ..., d_{ndim-1}] where
        // d_{ragged_dim} = max_len.
        let mut out_shape = Vec::with_capacity(ndim + 1);
        out_shape.push(batch);
        for d in 0..ndim {
            if d == self.ragged_dim {
                out_shape.push(max_len);
            } else {
                out_shape.push(self.tensors[0].shape()[d]);
            }
        }

        // Compute output strides as isize (row-major). Used by both paths;
        // the GPU path passes them straight to `strided_scatter`, the CPU
        // path indexes into a flat Vec.
        let full_ndim = ndim + 1;
        let mut out_strides_i: Vec<isize> = vec![0; full_ndim];
        if full_ndim > 0 {
            out_strides_i[full_ndim - 1] = 1;
            for d in (0..full_ndim - 1).rev() {
                out_strides_i[d] = out_strides_i[d + 1] * out_shape[d + 1] as isize;
            }
        }

        // CORE-066 (#1760): graph-preserving path when autograd is live.
        // torch.nested.to_padded_tensor is differentiable (live torch
        // 2.11.0+cu130: padded.requires_grad == True with component grads
        // flowing back to the leaves); the fast paths below build detached
        // outputs, so grad-tracking components route through the
        // differentiable cat/unsqueeze composition instead.
        if crate::autograd::no_grad::is_grad_enabled()
            && self.tensors.iter().any(|t| t.requires_grad())
        {
            return self.to_padded_differentiable(pad_value, max_len);
        }

        // GPU fast path — every component on the same CUDA device, dtype
        // is f32 or f64, every component is rank-≤8 (the strided_scatter
        // kernel's hard cap). Composes from `fill_f{32,64}` +
        // `strided_scatter_f{32,64}`.
        if let Some(out) = self.try_to_padded_gpu(pad_value, &out_shape, &out_strides_i)? {
            return Ok(out);
        }

        // CPU path — original semantics preserved verbatim.
        let numel: usize = crate::shape::numel(&out_shape);
        let mut data = vec![pad_value; numel];

        // Convert isize strides to usize for index math (CPU path).
        let out_strides: Vec<usize> = out_strides_i.iter().map(|&s| s as usize).collect();

        for (b, t) in self.tensors.iter().enumerate() {
            // CORE-070 (#1764): materialize the LOGICAL view — `data_vec`
            // walks strides/offset, so valid non-contiguous component views
            // (transpose, narrow) pad correctly. CUDA components that fall
            // through to this CPU path (unsupported dtype, rank > 8, no
            // backend) still error loudly — no silent host demotion
            // (R-LOUD-1).
            if t.is_cuda() {
                return Err(FerrotorchError::GpuTensorNotAccessible);
            }
            let t_data = t.data_vec()?;
            let t_shape = t.shape();

            // Compute strides for this component tensor (row-major).
            let mut t_strides = vec![0usize; ndim];
            if ndim > 0 {
                t_strides[ndim - 1] = 1;
                for d in (0..ndim - 1).rev() {
                    t_strides[d] = t_strides[d + 1] * t_shape[d + 1];
                }
            }

            let t_numel: usize = crate::shape::numel(t_shape);
            for (flat, &val) in t_data.iter().enumerate().take(t_numel) {
                // Convert flat index to multi-dim coords in the component.
                let mut remaining = flat;
                let mut out_flat = b * out_strides[0];
                for d in 0..ndim {
                    let coord = remaining / t_strides[d];
                    remaining %= t_strides[d];
                    out_flat += coord * out_strides[d + 1];
                }
                data[out_flat] = val;
            }
        }

        Tensor::from_storage(TensorStorage::cpu(data), out_shape, false)
    }

    /// Graph-preserving `to_padded` (CORE-066 / #1760): pads each
    /// component along the ragged dim by concatenating a CONSTANT
    /// pad-value filler (requires_grad = false — pad slots are constants,
    /// so the cat backward routes the cotangent only into the component),
    /// then stacks via `unsqueeze(0)` + `cat(axis 0)`. Every primitive
    /// (cat / unsqueeze) is differentiable and device-aware, so CUDA
    /// components stay on-device and gradients flow back to the original
    /// component leaves — torch parity with the differentiable
    /// `torch.nested.to_padded_tensor`.
    fn to_padded_differentiable(
        &self,
        pad_value: T,
        max_len: usize,
    ) -> FerrotorchResult<Tensor<T>> {
        let device = self.tensors[0].device();
        let mut rows = Vec::with_capacity(self.tensors.len());
        for t in &self.tensors {
            let len = t.shape()[self.ragged_dim];
            let padded_comp = if len == max_len {
                t.clone()
            } else {
                let mut pad_shape = t.shape().to_vec();
                pad_shape[self.ragged_dim] = max_len - len;
                // Constant filler on the components' device; never tracked.
                let filler = crate::creation::full::<T>(&pad_shape, pad_value)?.to(device)?;
                crate::grad_fns::shape::cat(&[t.clone(), filler], self.ragged_dim as isize)?
            };
            rows.push(crate::grad_fns::shape::unsqueeze(&padded_comp, 0)?);
        }
        crate::grad_fns::shape::cat(&rows, 0)
    }

    /// GPU fast path for [`to_padded`]. Returns `Ok(None)` when any
    /// precondition fails (component not on CUDA, mixed devices,
    /// unsupported dtype, no backend installed) so the caller falls
    /// through to the CPU path.
    ///
    /// All work happens on-device:
    ///  - one `fill_f{32,64}` allocates the padded buffer pre-loaded with
    ///    `pad_value`,
    ///  - one `strided_scatter_f{32,64}` per component writes its values
    ///    into the corresponding slot.
    ///
    /// Each component is materialised via `.contiguous()` before the
    /// scatter, which itself routes through the GPU `strided_copy_*`
    /// kernel for non-contiguous inputs (#802) — the upshot is that
    /// stride views (narrow / permute) of CUDA components stay on device
    /// throughout.
    fn try_to_padded_gpu(
        &self,
        pad_value: T,
        out_shape: &[usize],
        out_strides_i: &[isize],
    ) -> FerrotorchResult<Option<Tensor<T>>> {
        use std::any::TypeId;

        let is_f32 = TypeId::of::<T>() == TypeId::of::<f32>();
        let is_f64 = TypeId::of::<T>() == TypeId::of::<f64>();
        if !is_f32 && !is_f64 {
            return Ok(None);
        }

        // All components must live on the same CUDA device.
        let ordinal = match self.tensors[0].device() {
            Device::Cuda(o) => o,
            _ => return Ok(None),
        };
        for t in &self.tensors {
            match t.device() {
                Device::Cuda(o) if o == ordinal => {}
                Device::Cuda(o) => {
                    return Err(FerrotorchError::DeviceMismatch {
                        expected: Device::Cuda(ordinal),
                        got: Device::Cuda(o),
                    });
                }
                _ => return Ok(None),
            }
        }

        // Rank-≤8 cap: strided_scatter / strided_copy reject ranks above
        // that. Output rank is `ndim() + 1`.
        if out_shape.len() > 8 {
            return Ok(None);
        }

        let backend = match crate::gpu_dispatch::gpu_backend() {
            Some(b) => b,
            None => return Ok(None),
        };

        let numel: usize = crate::shape::numel(out_shape);

        // Step 1: allocate padded output buffer pre-filled with pad_value.
        let mut out_handle = if is_f32 {
            let pad_f32 = <T as num_traits::ToPrimitive>::to_f32(&pad_value).ok_or_else(|| {
                FerrotorchError::InvalidArgument {
                    message: "to_padded: pad_value not representable as f32".into(),
                }
            })?;
            backend.fill_f32(numel, pad_f32, ordinal)?
        } else {
            let pad_f64 = <T as num_traits::ToPrimitive>::to_f64(&pad_value).ok_or_else(|| {
                FerrotorchError::InvalidArgument {
                    message: "to_padded: pad_value not representable as f64".into(),
                }
            })?;
            backend.fill_f64(numel, pad_f64, ordinal)?
        };

        // Step 2: scatter each component into its slot.
        // The slot view shares the output buffer's strides on the
        // component dims (everything after the leading batch axis); the
        // offset is `b * out_strides[0]` (the stride of the batch dim).
        //
        // Each component is materialised into a fresh contiguous CUDA
        // buffer via `strided_copy_f{32,64}` first. Plain `.contiguous()`
        // is insufficient because it short-circuits on stride-contiguous
        // tensors but leaves a non-zero `storage_offset` (e.g. a
        // `narrow(0, k, n)` view), and we pass the raw buffer handle to
        // `strided_scatter` which has no offset parameter for the source.
        // `strided_copy_*` always produces a fresh `[0..numel)` buffer
        // that exactly matches the view's logical extent — which is
        // what `strided_scatter` expects.
        let comp_strides_out: Vec<isize> = out_strides_i[1..].to_vec();
        let batch_stride = out_strides_i[0] as usize;

        for (b, t) in self.tensors.iter().enumerate() {
            let src_buf = t.gpu_handle()?;
            let comp_shape = t.shape().to_vec();
            let comp_view_strides = t.strides().to_vec();
            let comp_view_offset = t.storage_offset();
            let scatter_offset = b * batch_stride;

            if is_f32 {
                let materialised = backend.strided_copy_f32(
                    src_buf,
                    &comp_shape,
                    &comp_view_strides,
                    comp_view_offset,
                )?;
                backend.strided_scatter_f32(
                    &materialised,
                    &mut out_handle,
                    &comp_shape,
                    &comp_strides_out,
                    scatter_offset,
                )?;
            } else {
                let materialised = backend.strided_copy_f64(
                    src_buf,
                    &comp_shape,
                    &comp_view_strides,
                    comp_view_offset,
                )?;
                backend.strided_scatter_f64(
                    &materialised,
                    &mut out_handle,
                    &comp_shape,
                    &comp_strides_out,
                    scatter_offset,
                )?;
            }
        }

        let storage = TensorStorage::gpu(out_handle);
        Tensor::from_storage(storage, out_shape.to_vec(), false).map(Some)
    }

    /// Reconstruct a nested tensor from a padded dense tensor and per-component
    /// lengths along the ragged dimension.
    ///
    /// This is the inverse of [`to_padded`](Self::to_padded). The first
    /// dimension of `tensor` is the batch dimension.
    ///
    /// # Arguments
    ///
    /// * `tensor` - Padded tensor with shape `[batch, d0, d1, ..., d_{ndim-1}]`.
    /// * `lengths` - Length of each component along the ragged dimension.
    ///   Must have length equal to the batch dimension.
    /// * `ragged_dim` - Which dimension (in the component tensors) is ragged.
    ///
    /// # GPU dispatch (P4 of #806)
    ///
    /// When `tensor` is on CUDA and `T` is `f32` or `f64`, each component
    /// is sliced out **on-device** by walking through `narrow` (a
    /// zero-copy stride view) and then `.contiguous()`, which dispatches
    /// to the existing `strided_copy_f{32,64}` GPU kernel from #802 /
    /// CL-496. The resulting components are CUDA tensors — no host
    /// bounce. PyTorch parity: `torch.nested.nested_tensor` over slices
    /// of a CUDA padded tensor produces CUDA components.
    pub fn from_padded(
        tensor: &Tensor<T>,
        lengths: &[usize],
        ragged_dim: usize,
    ) -> FerrotorchResult<Self> {
        let full_shape = tensor.shape();
        if full_shape.is_empty() {
            return Err(FerrotorchError::InvalidArgument {
                message: "from_padded requires at least a batch dimension".into(),
            });
        }

        let batch = full_shape[0];
        if lengths.len() != batch {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "lengths has {} entries but batch dimension is {}",
                    lengths.len(),
                    batch
                ),
            });
        }

        let comp_ndim = full_shape.len() - 1; // number of dims in each component
        if ragged_dim >= comp_ndim {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "ragged_dim {ragged_dim} out of range for {comp_ndim}-D component tensors"
                ),
            });
        }

        // Per-component length must not exceed the padded extent on the
        // ragged axis (otherwise the narrow would walk off the buffer).
        let max_len = full_shape[ragged_dim + 1];
        for (i, &len) in lengths.iter().enumerate() {
            if len > max_len {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "from_padded: lengths[{i}] = {len} exceeds padded extent \
                         {max_len} on ragged dim {ragged_dim}"
                    ),
                });
            }
        }

        // CORE-066 (#1760): graph-preserving path when autograd is live.
        // Each component is sliced out of the padded source with the
        // DIFFERENTIABLE narrow → contiguous → reshape chain (all
        // device-aware), so the components stay connected to the padded
        // tensor's graph: backward scatters each component's cotangent
        // into its slot of the padded source and leaves zeros in the pad
        // region (live torch oracle in tests/audit_core066_nested_autograd.rs).
        if crate::autograd::no_grad::is_grad_enabled() && tensor.requires_grad() {
            let mut tensors = Vec::with_capacity(batch);
            for (b, &len_b) in lengths.iter().enumerate() {
                let batch_view = tensor.narrow(0, b, 1)?;
                let ragged_view = if len_b == full_shape[ragged_dim + 1] {
                    batch_view
                } else {
                    batch_view.narrow(ragged_dim + 1, 0, len_b)?
                };
                // Collapse the leading 1-batch axis. `contiguous` and
                // `reshape` both attach backward edges.
                let comp_shape: Vec<isize> = (0..comp_ndim)
                    .map(|d| {
                        if d == ragged_dim {
                            len_b as isize
                        } else {
                            full_shape[d + 1] as isize
                        }
                    })
                    .collect();
                let comp =
                    crate::grad_fns::shape::reshape(&ragged_view.contiguous()?, &comp_shape)?;
                tensors.push(comp);
            }
            return Self::new(tensors, ragged_dim);
        }

        // GPU fast path — CUDA padded tensor + f32/f64 dtype. Composes
        // from `narrow` (zero-copy stride view) + `.contiguous()` (which
        // routes through `strided_copy_*` for non-contiguous CUDA views).
        if tensor.is_cuda()
            && let Some(nested) = Self::try_from_padded_gpu(tensor, lengths, ragged_dim)?
        {
            return Ok(nested);
        }

        // CORE-070 (#1764): materialize the LOGICAL view so valid
        // non-contiguous padded sources (e.g. transpose views) slice
        // correctly; CUDA tensors that the GPU fast path declined still
        // error loudly (no silent host demotion, R-LOUD-1).
        if tensor.is_cuda() {
            return Err(FerrotorchError::GpuTensorNotAccessible);
        }
        let padded_data = tensor.data_vec()?;

        // Strides for the full padded tensor (row-major).
        let full_ndim = full_shape.len();
        let mut full_strides = vec![0usize; full_ndim];
        if full_ndim > 0 {
            full_strides[full_ndim - 1] = 1;
            for d in (0..full_ndim - 1).rev() {
                full_strides[d] = full_strides[d + 1] * full_shape[d + 1];
            }
        }

        let mut tensors = Vec::with_capacity(batch);
        for (b, &len_b) in lengths.iter().enumerate().take(batch) {
            // Build component shape: same as full_shape[1..] but with
            // ragged_dim replaced by len_b.
            let mut comp_shape = Vec::with_capacity(comp_ndim);
            for d in 0..comp_ndim {
                if d == ragged_dim {
                    comp_shape.push(len_b);
                } else {
                    comp_shape.push(full_shape[d + 1]);
                }
            }

            // Compute strides for the component (row-major).
            let mut comp_strides = vec![0usize; comp_ndim];
            if comp_ndim > 0 {
                comp_strides[comp_ndim - 1] = 1;
                for d in (0..comp_ndim - 1).rev() {
                    comp_strides[d] = comp_strides[d + 1] * comp_shape[d + 1];
                }
            }

            let comp_numel: usize = crate::shape::numel(&comp_shape);
            let mut comp_data = Vec::with_capacity(comp_numel);

            for flat in 0..comp_numel {
                // Convert flat index to multi-dim coords in the component.
                let mut remaining = flat;
                let mut full_flat = b * full_strides[0];
                for d in 0..comp_ndim {
                    let coord = remaining.checked_div(comp_strides[d]).unwrap_or(0);
                    if comp_strides[d] > 0 {
                        remaining %= comp_strides[d];
                    }
                    full_flat += coord * full_strides[d + 1];
                }
                comp_data.push(padded_data[full_flat]);
            }

            tensors.push(Tensor::from_storage(
                TensorStorage::cpu(comp_data),
                comp_shape,
                false,
            )?);
        }

        Self::new(tensors, ragged_dim)
    }

    /// GPU fast path for [`from_padded`]. Returns `Ok(None)` when the
    /// dtype isn't f32/f64 so the caller falls through to the CPU path
    /// (which will surface `GpuTensorNotAccessible` for an unsupported
    /// CUDA dtype rather than silently corrupting).
    ///
    /// For each component `b`:
    ///  1. `narrow(0, b, 1)` → batch-slice view (zero-copy stride view),
    ///     then `narrow(ragged_dim+1, 0, lengths[b])` → ragged-slice view.
    ///  2. `view_reshape` collapses the leading 1-batch axis (composes
    ///     through `.contiguous()` on the GPU strided view).
    ///  3. `.contiguous()` materialises the (possibly non-contiguous)
    ///     CUDA view into a fresh contiguous CUDA buffer via the existing
    ///     `strided_copy_f{32,64}` kernel — never bounces through host.
    fn try_from_padded_gpu(
        tensor: &Tensor<T>,
        lengths: &[usize],
        ragged_dim: usize,
    ) -> FerrotorchResult<Option<Self>> {
        use std::any::TypeId;

        let is_f32 = TypeId::of::<T>() == TypeId::of::<f32>();
        let is_f64 = TypeId::of::<T>() == TypeId::of::<f64>();
        if !is_f32 && !is_f64 {
            return Ok(None);
        }

        let full_shape = tensor.shape().to_vec();
        let comp_ndim = full_shape.len() - 1;

        let mut tensors = Vec::with_capacity(lengths.len());
        for (b, &len_b) in lengths.iter().enumerate() {
            // narrow on the batch axis: shape [1, d0, ..., d_{r}=max_len, ...]
            let batch_view = tensor.narrow(0, b, 1)?;

            // narrow on the ragged axis (now at index `ragged_dim + 1` of
            // the padded tensor): shape [1, d0, ..., d_{r}=len_b, ...]
            let ragged_view = if len_b == full_shape[ragged_dim + 1] {
                batch_view
            } else {
                batch_view.narrow(ragged_dim + 1, 0, len_b)?
            };

            // Materialise the (potentially non-contiguous) view into a
            // fresh contiguous CUDA buffer. `contiguous()` dispatches to
            // `strided_copy_f{32,64}` on CUDA — entirely on-device.
            let materialised = ragged_view.contiguous()?;

            // Drop the leading 1-batch axis so the component shape is
            // `[d0, ..., d_{r}=len_b, ...]`. Build a fresh tensor that
            // re-wraps the same GPU storage with the trimmed shape — the
            // materialised tensor is already contiguous and its data
            // layout already matches the trimmed shape (size 1 at the
            // front contributes nothing to the linear index).
            let comp_shape: Vec<usize> = (0..comp_ndim)
                .map(|d| {
                    if d == ragged_dim {
                        len_b
                    } else {
                        full_shape[d + 1]
                    }
                })
                .collect();

            // Sanity check: the linear extent must match the materialised
            // tensor's numel (size-1 leading dim trivially preserves it).
            let comp_numel: usize = crate::shape::numel(&comp_shape);
            debug_assert_eq!(comp_numel, materialised.numel());

            // Rewrap into the trimmed shape via `view_reshape` (no copy:
            // contiguous storage, same numel).
            let comp = materialised.view_reshape(comp_shape)?;
            tensors.push(comp);
        }

        Self::new(tensors, ragged_dim).map(Some)
    }
}

// --- Attention ---

/// Row-wise softmax in place.
///
/// Each row of `data` (of width `cols`) is independently softmax'd.
/// For numerical stability, the maximum value is subtracted before
/// exponentiation.
///
/// When all values in a row are `-inf` (producing a sum of zero after
/// exponentiation), the row is filled with NaN to match PyTorch semantics.
fn softmax_rows_inplace<T: Float>(data: &mut [T], rows: usize, cols: usize) {
    for r in 0..rows {
        let row = &mut data[r * cols..(r + 1) * cols];

        // Numerical stability: subtract row max.
        let max_val = row
            .iter()
            .copied()
            .fold(<T as num_traits::Float>::neg_infinity(), |a, b| {
                if b > a { b } else { a }
            });

        let mut sum = <T as num_traits::Zero>::zero();
        for val in row.iter_mut() {
            *val = (*val - max_val).exp();
            sum += *val;
        }

        if sum == <T as num_traits::Zero>::zero() {
            // All inputs were -inf; produce NaN to match PyTorch.
            for val in row.iter_mut() {
                *val = <T as num_traits::Float>::nan();
            }
        } else {
            for val in row.iter_mut() {
                *val = *val / sum;
            }
        }
    }
}

/// Scaled dot-product attention over nested tensors.
///
/// Implements the standard multi-head attention formula:
///
/// ```text
/// Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
/// ```
///
/// Each component in the nested tensor is processed independently (they may
/// have different sequence lengths). This is generic over `T: Float` so it
/// works with both f32 and f64.
///
/// # Arguments
///
/// * `query` - Nested tensor of shape `[seq_q, d_k]` per component.
/// * `key` - Nested tensor of shape `[seq_k, d_k]` per component.
/// * `value` - Nested tensor of shape `[seq_k, d_v]` per component.
///
/// # Returns
///
/// A nested tensor of shape `[seq_q, d_v]` per component.
pub fn nested_scaled_dot_product_attention<T: Float>(
    query: &NestedTensor<T>,
    key: &NestedTensor<T>,
    value: &NestedTensor<T>,
) -> FerrotorchResult<NestedTensor<T>> {
    let n = query.num_components();
    if key.num_components() != n || value.num_components() != n {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "query has {} components but key has {} and value has {}",
                n,
                key.num_components(),
                value.num_components()
            ),
        });
    }

    let mut outputs = Vec::with_capacity(n);

    for i in 0..n {
        let q = &query.tensors()[i];
        let k = &key.tensors()[i];
        let v = &value.tensors()[i];

        if q.ndim() != 2 || k.ndim() != 2 || v.ndim() != 2 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "attention requires 2-D tensors, component {} has dims ({}, {}, {})",
                    i,
                    q.ndim(),
                    k.ndim(),
                    v.ndim()
                ),
            });
        }

        let seq_q = q.shape()[0];
        let d_k = q.shape()[1];
        let seq_k = k.shape()[0];
        let d_k2 = k.shape()[1];
        let seq_k2 = v.shape()[0];
        let d_v = v.shape()[1];

        if d_k != d_k2 {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!("component {i}: query d_k={d_k} but key d_k={d_k2}"),
            });
        }
        if seq_k != seq_k2 {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!("component {i}: key seq_len={seq_k} but value seq_len={seq_k2}"),
            });
        }

        // CORE-066 (#1760): the flash kernel and the scalar CPU loop both
        // build detached outputs. When autograd is live and any of
        // q/k/v tracks gradients, route through the differentiable
        // composite (`mm_bt` → broadcast `mul` → `softmax` → `matmul`),
        // which is device-aware (CUDA components stay on-device) and
        // attaches real backward edges — torch parity with the
        // differentiable `F.scaled_dot_product_attention`.
        if crate::autograd::no_grad::is_grad_enabled()
            && (q.requires_grad() || k.requires_grad() || v.requires_grad())
        {
            outputs.push(attention_component_composite(q, k, v, d_k)?);
            continue;
        }

        // GPU FlashAttention forward dispatch. Per #806, when the
        // component lives on CUDA and falls within the kernel's regime
        // (d_k <= 128 and d_v <= 128), route to the on-device tiled
        // online-softmax kernel via the registered `GpuBackend`. If the
        // backend declines (unsupported dtype, shape, etc.), fall
        // through to the composite path below -- never CPU detour.
        if try_flash_attention_gpu_component::<T>(q, k, v, seq_q, seq_k, d_k, d_v, &mut outputs, i)?
        {
            continue;
        }

        // CORE-067 (#1761): the flash kernel declined (head dim > 128,
        // unsupported dtype, no backend) but the component lives on CUDA.
        // The scalar loop below is CPU-only; run the device-aware
        // composite (`mm_bt` → broadcast `mul` → `softmax` → `matmul`)
        // instead — the result stays on CUDA. Mixed-device q/k/v surface
        // a structured DeviceMismatch from the primitives. Never a silent
        // host bounce (R-LOUD-1).
        if q.is_cuda() || k.is_cuda() || v.is_cuda() {
            outputs.push(attention_component_composite(q, k, v, d_k)?);
            continue;
        }
        // CORE-070 (#1764): materialize LOGICAL views (`data_vec` walks
        // strides/offset) so valid non-contiguous q/k/v views run.
        let q_data = q.data_vec()?;
        let k_data = k.data_vec()?;
        let v_data = v.data_vec()?;

        let scale = T::from(d_k).unwrap().sqrt().recip();

        // Compute Q @ K^T: [seq_q, seq_k]
        let mut scores = vec![<T as num_traits::Zero>::zero(); seq_q * seq_k];
        for qi in 0..seq_q {
            for ki in 0..seq_k {
                let mut dot = <T as num_traits::Zero>::zero();
                for di in 0..d_k {
                    dot += q_data[qi * d_k + di] * k_data[ki * d_k + di];
                }
                scores[qi * seq_k + ki] = dot * scale;
            }
        }

        // Softmax over each row.
        softmax_rows_inplace(&mut scores, seq_q, seq_k);

        // Multiply by V: [seq_q, d_v]
        let mut out = vec![<T as num_traits::Zero>::zero(); seq_q * d_v];
        for qi in 0..seq_q {
            for dvi in 0..d_v {
                let mut acc = <T as num_traits::Zero>::zero();
                for ki in 0..seq_k {
                    acc += scores[qi * seq_k + ki] * v_data[ki * d_v + dvi];
                }
                out[qi * d_v + dvi] = acc;
            }
        }

        outputs.push(Tensor::from_storage(
            TensorStorage::cpu(out),
            vec![seq_q, d_v],
            false,
        )?);
    }

    NestedTensor::new(outputs, query.ragged_dim())
}

/// Composite scaled-dot-product attention for one nested component,
/// assembled from the dispatched differentiable primitives:
///
/// ```text
/// out = softmax(q @ k^T / sqrt(d_k)) @ v
/// ```
///
/// - `mm_bt` (fused `q @ k^T`), broadcast `mul` by the constant
///   `1/sqrt(d_k)`, row-wise `softmax` (last dim), and `matmul` are all
///   device-aware — CUDA inputs execute on CUDA — and all attach
///   backward edges, so this is BOTH the autograd path for CORE-066
///   (#1760) and the device-correct fallback for CUDA shapes outside the
///   flash-kernel regime (CORE-067 / #1761).
fn attention_component_composite<T: Float>(
    q: &Tensor<T>,
    k: &Tensor<T>,
    v: &Tensor<T>,
    d_k: usize,
) -> FerrotorchResult<Tensor<T>> {
    let scores = q.mm_bt(k)?; // [seq_q, seq_k]
    let scale = T::from(d_k)
        .ok_or_else(|| FerrotorchError::InvalidArgument {
            message: format!("attention: d_k {d_k} not representable in the tensor dtype"),
        })?
        .sqrt()
        .recip();
    // Constant scale factor (never tracked) on the inputs' device;
    // broadcast [seq_q, seq_k] * [1].
    let scale_t =
        Tensor::from_storage(TensorStorage::cpu(vec![scale]), vec![1], false)?.to(q.device())?;
    let scaled = crate::grad_fns::arithmetic::mul(&scores, &scale_t)?;
    let weights = scaled.softmax()?; // row-wise over the last dim
    weights.matmul(v)
}

/// GPU dispatch for one nested SDPA component. Returns `Ok(true)` when the
/// FlashAttention kernel handled the component (and pushed the result into
/// `outputs`); `Ok(false)` when the caller must fall through to the CPU /
/// composite path (e.g. tensors are not on CUDA, dtype is unsupported,
/// `d > 128`, or no GPU backend is registered).
///
/// Falling back to the GPU composite path (`bmm + softmax_rows + bmm`) for
/// shapes outside the kernel regime is filed as a follow-up; today we only
/// re-route to the existing CPU loop.
#[allow(clippy::too_many_arguments)]
fn try_flash_attention_gpu_component<T: Float>(
    q: &Tensor<T>,
    k: &Tensor<T>,
    v: &Tensor<T>,
    seq_q: usize,
    seq_k: usize,
    d_k: usize,
    d_v: usize,
    outputs: &mut Vec<Tensor<T>>,
    component_idx: usize,
) -> FerrotorchResult<bool> {
    use std::any::TypeId;

    let is_f32 = TypeId::of::<T>() == TypeId::of::<f32>();
    let is_f64 = TypeId::of::<T>() == TypeId::of::<f64>();
    if !is_f32 && !is_f64 {
        return Ok(false);
    }

    // All three components must be on the same CUDA device.
    let ordinal = match q.device() {
        Device::Cuda(o) => o,
        _ => return Ok(false),
    };
    match k.device() {
        Device::Cuda(o) if o == ordinal => {}
        Device::Cuda(o) => {
            return Err(FerrotorchError::DeviceMismatch {
                expected: Device::Cuda(ordinal),
                got: Device::Cuda(o),
            });
        }
        _ => return Ok(false),
    }
    match v.device() {
        Device::Cuda(o) if o == ordinal => {}
        Device::Cuda(o) => {
            return Err(FerrotorchError::DeviceMismatch {
                expected: Device::Cuda(ordinal),
                got: Device::Cuda(o),
            });
        }
        _ => return Ok(false),
    }

    // Kernel regime: d_head <= 128, scalar-fma path.
    if d_k > 128 || d_v > 128 {
        return Ok(false);
    }

    // Empty seq_q -> a [0, d_v] tensor on the same device.
    if seq_q == 0 {
        // Build a zero-length GPU tensor so the result remains on CUDA.
        let backend = match crate::gpu_dispatch::gpu_backend() {
            Some(b) => b,
            None => return Ok(false),
        };
        let handle = if is_f32 {
            backend.fill_f32(0, 0.0, ordinal)?
        } else {
            backend.fill_f64(0, 0.0, ordinal)?
        };
        outputs.push(Tensor::from_storage(
            TensorStorage::gpu(handle),
            vec![0, d_v],
            false,
        )?);
        let _ = component_idx;
        return Ok(true);
    }

    let backend = match crate::gpu_dispatch::gpu_backend() {
        Some(b) => b,
        None => return Ok(false),
    };

    // Components must be contiguous (the kernel walks row-major).
    let q_h = q.contiguous()?;
    let k_h = k.contiguous()?;
    let v_h = v.contiguous()?;

    let q_handle = q_h.gpu_handle()?;
    let k_handle = k_h.gpu_handle()?;
    let v_handle = v_h.gpu_handle()?;

    let scale_t = T::from(d_k).unwrap().sqrt().recip();

    let out_handle = if is_f32 {
        let scale_f32 = <T as num_traits::ToPrimitive>::to_f32(&scale_t).ok_or_else(|| {
            FerrotorchError::InvalidArgument {
                message: "flash_attention: scale not representable as f32".into(),
            }
        })?;
        backend.flash_attention_forward_f32(
            q_handle, k_handle, v_handle, seq_q, seq_k, d_k, d_v, scale_f32,
        )?
    } else {
        let scale_f64 = <T as num_traits::ToPrimitive>::to_f64(&scale_t).ok_or_else(|| {
            FerrotorchError::InvalidArgument {
                message: "flash_attention: scale not representable as f64".into(),
            }
        })?;
        backend.flash_attention_forward_f64(
            q_handle, k_handle, v_handle, seq_q, seq_k, d_k, d_v, scale_f64,
        )?
    };

    outputs.push(Tensor::from_storage(
        TensorStorage::gpu(out_handle),
        vec![seq_q, d_v],
        false,
    )?);

    Ok(true)
}

// ===========================================================================
// PackedNestedTensor — packed flat storage + offsets layout. CL-291.
// ===========================================================================

/// A nested (jagged) tensor stored as **one contiguous flat buffer**
/// with an offsets array marking the start of each component.
///
/// This is the efficient storage layout for nested tensors: bulk
/// elementwise ops operate on the whole flat buffer at once without
/// touching the offsets, per-sequence reductions walk the offsets
/// once, and conversion to/from padded dense tensors uses a single
/// linear scan. The companion [`NestedTensor`] list-of-tensors layout
/// is better for ergonomic per-component access; choose based on the
/// workload.
///
/// Every component has the same shape on the **tail** dimensions
/// (everything except `ragged_dim`), and the ragged dimension is
/// always the **leading** dim (dim 0) within each component. That
/// restriction keeps the offsets 1-D and the flat layout unambiguous.
///
/// # Example
///
/// ```ignore
/// // Batch of three sequences with lengths 3, 5, 2 and hidden dim 4.
/// let seqs = vec![
///     vec![1.0, 2.0, 3.0, 4.0,
///          5.0, 6.0, 7.0, 8.0,
///          9.0, 10.0, 11.0, 12.0],    // len=3 → 3*4 = 12 values
///     // ... sequence 1
///     // ... sequence 2
/// ];
/// let lengths = vec![3, 5, 2];
/// let tail_shape = vec![4];
/// let pnt = PackedNestedTensor::from_sequences(seqs, &lengths, &tail_shape)?;
/// assert_eq!(pnt.num_components(), 3);
/// assert_eq!(pnt.offsets(), &[0, 12, 32, 40]);
/// ```
///
/// # Layout invariants
///
/// - `offsets.len() == num_components + 1`
/// - `lengths.len() == num_components`
/// - `offsets[0] == 0`
/// - `offsets[i+1] - offsets[i] == lengths[i] * tail_numel`
/// - `offsets[num_components] == data.len()`
///
/// where `tail_numel = product(tail_shape)` — the ACTUAL product: `1`
/// for an empty tail (scalar tail), `0` when any tail dim is zero
/// (CORE-069 / #1763). Per-component lengths are carried explicitly
/// because element offsets degenerate (all equal) when
/// `tail_numel == 0` and the ragged lengths would be unrecoverable;
/// torch's jagged-layout NJT never loses them because its `_offsets`
/// count ragged-dim rows rather than flat elements
/// (`torch/nested/_internal/nested_tensor.py`).
#[derive(Debug, Clone)]
pub struct PackedNestedTensor<T: Float> {
    /// Flat concatenation of every component's data, in component
    /// order. For a nested tensor with components of shape
    /// `[L_i] + tail_shape`, component `i` occupies the slice
    /// `data[offsets[i] .. offsets[i+1]]`.
    data: Vec<T>,
    /// Length-`num_components + 1` offsets array. `offsets[i]` is
    /// the start of component `i` in `data`.
    offsets: Vec<usize>,
    /// Per-component ragged-dim lengths (`num_components` entries).
    /// Authoritative for [`Self::length`]; NOT derivable from
    /// `offsets` when the tail contains a zero dim.
    lengths: Vec<usize>,
    /// Shape of each component's tail (everything after the ragged
    /// dim). For a 1-D ragged sequence the tail is empty; for
    /// `[L, D]` components with ragged dim 0, the tail is `[D]`.
    tail_shape: Vec<usize>,
}

/// Centralized validation of the packed-layout invariants documented on
/// [`PackedNestedTensor`] (CORE-068 / #1762). Every constructor routes
/// through this check so no path can build a layout that violates:
///
/// - `offsets` is non-empty
/// - `offsets[0] == 0` (a nonzero first offset would silently discard a
///   data prefix)
/// - `offsets` is monotonically non-decreasing
/// - `offsets[num_components] == data_len`
/// - `lengths.len() == offsets.len() - 1`
/// - every component extent `offsets[i+1] - offsets[i]` equals
///   `lengths[i] * tail_numel` exactly (a non-divisible extent would
///   truncate `length()` and let `to_nested()` silently lose elements)
///
/// `tail_numel` is the ACTUAL tail product (CORE-069 / #1763): `1` for an
/// empty tail, `0` when any tail dim is zero. Mirrors the torch
/// jagged-layout NJT offsets contract
/// (`torch/nested/_internal/nested_tensor.py` — `_offsets[0] == 0`,
/// final offset addressing the full `_values` extent).
fn validate_packed_layout(
    data_len: usize,
    offsets: &[usize],
    lengths: &[usize],
    tail_shape: &[usize],
) -> FerrotorchResult<()> {
    if offsets.is_empty() {
        return Err(FerrotorchError::InvalidArgument {
            message: "PackedNestedTensor: offsets must be non-empty".into(),
        });
    }
    if offsets[0] != 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "PackedNestedTensor: offsets[0] must be 0 (got {}); a nonzero first \
                 offset silently discards the data prefix [0..{})",
                offsets[0], offsets[0]
            ),
        });
    }
    for w in offsets.windows(2) {
        if w[1] < w[0] {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("PackedNestedTensor: offsets not monotonic: {offsets:?}"),
            });
        }
    }
    let last = *offsets.last().unwrap();
    if last != data_len {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!("PackedNestedTensor: final offset {last} != data length {data_len}"),
        });
    }
    if lengths.len() + 1 != offsets.len() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "PackedNestedTensor: lengths has {} entries but offsets has {} \
                 (expected lengths.len() + 1 == offsets.len())",
                lengths.len(),
                offsets.len()
            ),
        });
    }
    let tail_numel: usize = crate::shape::numel(tail_shape);
    for (i, w) in offsets.windows(2).enumerate() {
        let extent = w[1] - w[0];
        let expected =
            lengths[i]
                .checked_mul(tail_numel)
                .ok_or_else(|| FerrotorchError::InvalidArgument {
                    message: format!(
                        "PackedNestedTensor: length overflow in component {i} \
                     (length={}, tail_numel={tail_numel})",
                        lengths[i]
                    ),
                })?;
        if extent != expected {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "PackedNestedTensor: component {i} extent {extent} != \
                     lengths[{i}] * tail_numel = {} * {tail_numel} = {expected} \
                     (tail shape {tail_shape:?}); a non-divisible extent would \
                     truncate length() and silently lose elements",
                    lengths[i]
                ),
            });
        }
    }
    Ok(())
}

impl<T: Float> PackedNestedTensor<T> {
    /// Create a packed nested tensor from a list of per-component
    /// flat data buffers, their ragged-dim lengths, and the shared
    /// tail shape.
    ///
    /// `sequences[i]` must contain exactly `lengths[i] * tail_numel`
    /// values in row-major order (outer dim ragged, tail dims
    /// c-contiguous).
    ///
    /// # Errors
    ///
    /// - `sequences.len() != lengths.len()`
    /// - Any `sequences[i].len() != lengths[i] * tail_numel`
    /// - Empty input (`sequences.is_empty()`)
    pub fn from_sequences(
        sequences: Vec<Vec<T>>,
        lengths: &[usize],
        tail_shape: &[usize],
    ) -> FerrotorchResult<Self> {
        if sequences.is_empty() {
            return Err(FerrotorchError::InvalidArgument {
                message: "PackedNestedTensor requires at least one sequence".into(),
            });
        }
        if sequences.len() != lengths.len() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "PackedNestedTensor: sequences has {} entries but lengths has {}",
                    sequences.len(),
                    lengths.len()
                ),
            });
        }
        // CORE-069 (#1763): the ACTUAL tail product. An empty tail is a
        // scalar tail (empty product = 1); a tail containing a zero dim
        // means every row holds ZERO elements — conflating the two via
        // `.max(1)` accepted phantom data for `[L, 0]`-shaped components.
        let tail_numel: usize = crate::shape::numel(tail_shape);

        let mut total = 0usize;
        for (i, seq) in sequences.iter().enumerate() {
            let expected = lengths[i].checked_mul(tail_numel).ok_or_else(|| {
                FerrotorchError::InvalidArgument {
                    message: format!(
                        "PackedNestedTensor: length overflow in component {i} \
                         (length={}, tail_numel={tail_numel})",
                        lengths[i]
                    ),
                }
            })?;
            if seq.len() != expected {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "PackedNestedTensor: sequence {i} has {} elements but expected \
                         lengths[{i}] * tail_numel = {}*{} = {}",
                        seq.len(),
                        lengths[i],
                        tail_numel,
                        expected
                    ),
                });
            }
            total += expected;
        }

        let mut data = Vec::with_capacity(total);
        let mut offsets = Vec::with_capacity(sequences.len() + 1);
        offsets.push(0);
        for seq in sequences {
            data.extend(seq);
            offsets.push(data.len());
        }

        // CORE-068 (#1762): every constructor funnels through the
        // centralized layout validation.
        validate_packed_layout(data.len(), &offsets, lengths, tail_shape)?;

        Ok(Self {
            data,
            offsets,
            lengths: lengths.to_vec(),
            tail_shape: tail_shape.to_vec(),
        })
    }

    /// Create a packed nested tensor from the component tensors of
    /// a [`NestedTensor`]. `ragged_dim` must be 0 — the packed
    /// layout requires the ragged dim to lead.
    ///
    /// # Errors
    ///
    /// - `nested.ragged_dim() != 0`
    /// - `nested.num_components() == 0`
    pub fn from_nested(nested: &NestedTensor<T>) -> FerrotorchResult<Self> {
        if nested.ragged_dim() != 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "PackedNestedTensor::from_nested requires ragged_dim == 0, got {}",
                    nested.ragged_dim()
                ),
            });
        }
        let comps = nested.tensors();
        if comps.is_empty() {
            return Err(FerrotorchError::InvalidArgument {
                message: "PackedNestedTensor::from_nested: no components".into(),
            });
        }
        // Tail shape = shape without the ragged (leading) dim.
        let tail_shape: Vec<usize> = comps[0].shape()[1..].to_vec();
        let lengths: Vec<usize> = comps.iter().map(|t| t.shape()[0]).collect();

        let mut sequences: Vec<Vec<T>> = Vec::with_capacity(comps.len());
        for t in comps {
            // CORE-066 (#1760, R-LOUD-3): the packed layout stores raw
            // values and drops autograd graphs by design. A grad-tracking
            // component must not be silently detached — error loudly;
            // callers detach() explicitly or stay on the NestedTensor
            // components-list layout for autograd work.
            if crate::autograd::no_grad::is_grad_enabled() && t.requires_grad() {
                return Err(FerrotorchError::InvalidArgument {
                    message: "PackedNestedTensor::from_nested: component tracks \
                              gradients, but the packed layout stores raw values and \
                              would silently sever its graph; detach() the component \
                              explicitly or keep the NestedTensor layout"
                        .into(),
                });
            }
            // CORE-070 (#1764): materialize each component's LOGICAL view
            // (`data_vec` walks strides/offset) so non-contiguous component
            // views pack correctly. `PackedNestedTensor` storage is
            // CPU-resident by construction; CUDA components error loudly
            // rather than silently bouncing to host (R-LOUD-1).
            if t.is_cuda() {
                return Err(FerrotorchError::GpuTensorNotAccessible);
            }
            sequences.push(t.data_vec()?);
        }
        Self::from_sequences(sequences, &lengths, &tail_shape)
    }

    /// Convert back into a component-list [`NestedTensor`] (ragged
    /// dim 0). Each component becomes a fresh [`Tensor<T>`] holding
    /// its slice of the packed data.
    pub fn to_nested(&self) -> FerrotorchResult<NestedTensor<T>> {
        let n = self.num_components();
        let mut tensors = Vec::with_capacity(n);
        for i in 0..n {
            let len = self.length(i);
            let mut shape = vec![len];
            shape.extend_from_slice(&self.tail_shape);
            let slice = self.component_slice(i).to_vec();
            tensors.push(Tensor::from_storage(
                TensorStorage::cpu(slice),
                shape,
                false,
            )?);
        }
        NestedTensor::new(tensors, 0)
    }

    /// Number of component sequences.
    #[inline]
    pub fn num_components(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// The `offsets` array. Always `num_components + 1` long.
    #[inline]
    pub fn offsets(&self) -> &[usize] {
        &self.offsets
    }

    /// Shared tail shape (non-ragged dims).
    #[inline]
    pub fn tail_shape(&self) -> &[usize] {
        &self.tail_shape
    }

    /// The flat packed data buffer.
    #[inline]
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Length (along the ragged dim) of component `i`.
    ///
    /// Authoritative from the stored `lengths` (CORE-069 / #1763): with a
    /// zero-containing tail the element offsets degenerate (all equal) and
    /// the ragged lengths cannot be recomputed from them.
    ///
    /// # Panics
    ///
    /// Panics if `i >= num_components()`.
    #[inline]
    pub fn length(&self, i: usize) -> usize {
        self.lengths[i]
    }

    /// Total number of elements in the packed buffer (sum of every
    /// component's element count).
    #[inline]
    pub fn total_numel(&self) -> usize {
        self.data.len()
    }

    /// Borrow the slice of the packed data holding component `i`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= num_components()`.
    pub fn component_slice(&self, i: usize) -> &[T] {
        &self.data[self.offsets[i]..self.offsets[i + 1]]
    }

    /// Materialise the flat packed buffer as a 1-D CPU [`Tensor<T>`] of
    /// shape `[total_numel]`, with no autograd hookup. P8 of #806.
    ///
    /// This is the bridge between `PackedNestedTensor`'s `Vec<T>` storage
    /// and the device-aware `Tensor<T>` API. Callers wanting on-device
    /// element-wise composition (`Tensor + Tensor` etc.) call this, then
    /// `.to(Device::Cuda(0))`, do their dispatched arithmetic on the flat
    /// buffer (offsets are layout metadata only — they don't influence the
    /// bulk arithmetic), and round-trip back via [`Self::from_data_tensor`].
    ///
    /// PyTorch parity (`rust-gpu-discipline` §3): the analog is
    /// `torch.nested.nested_tensor(...).values()` on the
    /// `jagged`/`packed` layout, which exposes the underlying contiguous
    /// buffer for element-wise composition.
    pub fn data_to_tensor(&self) -> FerrotorchResult<Tensor<T>> {
        Tensor::from_storage(
            TensorStorage::cpu(self.data.clone()),
            vec![self.data.len()],
            false,
        )
    }

    /// Build a `PackedNestedTensor` from a flat 1-D `Tensor<T>` plus
    /// offsets and tail shape. Inverse of [`Self::data_to_tensor`].
    ///
    /// If `tensor` lives on CUDA, its data is read back to host (this
    /// reconstructs the CPU-resident `Vec<T>` storage that
    /// `PackedNestedTensor` owns by construction). The expected workflow is
    /// "upload, compose, download once" — not per-call round-trips.
    ///
    /// # Errors
    ///
    /// - `tensor` is not the documented flat 1-D tensor (`ndim != 1`)
    /// - `tensor.shape() != [offsets[N]]` (length must match the offsets'
    ///   final entry)
    /// - `offsets` violates the packed-layout invariants: empty, first
    ///   entry nonzero, not monotonically non-decreasing, or any component
    ///   extent not divisible by `product(tail_shape)` (CORE-068 / #1762).
    pub fn from_data_tensor(
        tensor: &Tensor<T>,
        offsets: Vec<usize>,
        tail_shape: Vec<usize>,
    ) -> FerrotorchResult<Self> {
        if tensor.ndim() != 1 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "PackedNestedTensor::from_data_tensor: data tensor must be the \
                     flat 1-D buffer produced by data_to_tensor (got ndim {} with \
                     shape {:?})",
                    tensor.ndim(),
                    tensor.shape()
                ),
            });
        }
        // CORE-069 (#1763): with a zero-containing tail every component
        // spans zero elements, so the per-component ragged lengths are NOT
        // derivable from element offsets. The honest contract is a
        // structured error pointing at the constructor that carries them.
        let tail_numel: usize = crate::shape::numel(&tail_shape);
        if tail_numel == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "PackedNestedTensor::from_data_tensor: tail shape {tail_shape:?} \
                     contains a zero dim, so ragged lengths are not derivable from \
                     element offsets; use from_sequences (which carries lengths \
                     explicitly) instead"
                ),
            });
        }
        // Derive lengths from the extents (divisibility is enforced by the
        // CORE-068 validation below; use checked_sub so a non-monotonic
        // offsets array cannot underflow before validation runs).
        let mut lengths = Vec::with_capacity(offsets.len().saturating_sub(1));
        for w in offsets.windows(2) {
            let extent =
                w[1].checked_sub(w[0])
                    .ok_or_else(|| FerrotorchError::InvalidArgument {
                        message: format!(
                            "PackedNestedTensor::from_data_tensor: offsets not monotonic: \
                         {offsets:?}"
                        ),
                    })?;
            if extent % tail_numel != 0 {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "PackedNestedTensor::from_data_tensor: component extent {extent} \
                         is not divisible by tail_numel {tail_numel} (tail shape \
                         {tail_shape:?}); length() would truncate and silently lose \
                         elements"
                    ),
                });
            }
            lengths.push(extent / tail_numel);
        }
        // CORE-068 (#1762): full layout validation — offsets[0] == 0,
        // monotonic, final entry == numel, extents match lengths*tail_numel.
        validate_packed_layout(tensor.numel(), &offsets, &lengths, &tail_shape)?;
        let host = if tensor.is_cuda() {
            tensor.cpu()?.data()?.to_vec()
        } else {
            tensor.data()?.to_vec()
        };
        Ok(Self {
            data: host,
            offsets,
            lengths,
            tail_shape,
        })
    }

    /// Elementwise map that applies `f` to every value in the
    /// packed buffer and returns a new `PackedNestedTensor` with
    /// the same offsets and tail shape.
    ///
    /// This is the workhorse for implementing nested-level
    /// elementwise ops (relu, neg, abs, etc.) without writing per-
    /// component loops. CL-291.
    ///
    /// # GPU policy (P8 of #806)
    ///
    /// `map` is **CPU-only by design** because Rust closures cannot be
    /// JIT-compiled into PTX kernels. This is the canonical
    /// `rust-gpu-discipline` §3 composite-implicit-autograd shape: callers
    /// who want GPU execution decompose their `map(|v| ...)` into a sequence
    /// of dispatched primitives (e.g. `Tensor::relu`, `Tensor::sigmoid`,
    /// `Tensor::neg`), each of which routes correctly to the device the
    /// inputs live on. PyTorch's analog is identical — `torch.Tensor.apply_`
    /// is documented as CPU-only and users are directed to the dispatched
    /// element-wise ops for CUDA.
    pub fn map(&self, f: impl Fn(T) -> T) -> Self {
        let data: Vec<T> = self.data.iter().copied().map(f).collect();
        Self {
            data,
            offsets: self.offsets.clone(),
            lengths: self.lengths.clone(),
            tail_shape: self.tail_shape.clone(),
        }
    }

    /// Elementwise addition of two packed nested tensors. Both must
    /// have identical offsets and tail shape.
    ///
    /// # Errors
    ///
    /// Returns an error if the offsets or tail shapes don't match.
    ///
    /// # GPU policy (P8 of #806)
    ///
    /// `PackedNestedTensor` stores the flat buffer in `Vec<T>` (CPU-resident
    /// by definition). The GPU lane is a `rust-gpu-discipline` §3
    /// composite-implicit-autograd: callers materialise the flat buffer as
    /// a `Tensor<T>` on CUDA via `data_to_tensor` (P8), use the dispatched
    /// `Tensor + Tensor` (which routes to `add_f32` / `add_f64` /
    /// `broadcast_add_*` on-device), and re-pack via `from_data_tensor`. The
    /// offsets and tail shape are layout metadata only — they don't
    /// influence the bulk arithmetic, which operates on the flat buffer.
    pub fn add(&self, other: &Self) -> FerrotorchResult<Self> {
        self.zip_with(other, "add", |a, b| a + b)
    }

    /// Elementwise subtraction. See [`Self::add`] for the §3 GPU lane.
    pub fn sub(&self, other: &Self) -> FerrotorchResult<Self> {
        self.zip_with(other, "sub", |a, b| a - b)
    }

    /// Elementwise multiplication. See [`Self::add`] for the §3 GPU lane.
    pub fn mul(&self, other: &Self) -> FerrotorchResult<Self> {
        self.zip_with(other, "mul", |a, b| a * b)
    }

    /// Elementwise division. See [`Self::add`] for the §3 GPU lane.
    pub fn div(&self, other: &Self) -> FerrotorchResult<Self> {
        self.zip_with(other, "div", |a, b| a / b)
    }

    /// Shared implementation for elementwise binary ops. Validates
    /// that `self` and `other` have matching layouts before
    /// applying `f` over the packed data.
    fn zip_with(
        &self,
        other: &Self,
        op_name: &'static str,
        f: impl Fn(T, T) -> T,
    ) -> FerrotorchResult<Self> {
        if self.offsets != other.offsets {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "PackedNestedTensor::{op_name}: offsets mismatch \
                     ({:?} vs {:?})",
                    self.offsets, other.offsets
                ),
            });
        }
        if self.tail_shape != other.tail_shape {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "PackedNestedTensor::{op_name}: tail shape mismatch \
                     ({:?} vs {:?})",
                    self.tail_shape, other.tail_shape
                ),
            });
        }
        // CORE-069 (#1763): with a zero-containing tail, equal offsets do
        // NOT imply equal ragged lengths (all offsets degenerate to 0), so
        // lengths are compared explicitly.
        if self.lengths != other.lengths {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "PackedNestedTensor::{op_name}: ragged lengths mismatch \
                     ({:?} vs {:?})",
                    self.lengths, other.lengths
                ),
            });
        }
        let data: Vec<T> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| f(a, b))
            .collect();
        Ok(Self {
            data,
            offsets: self.offsets.clone(),
            lengths: self.lengths.clone(),
            tail_shape: self.tail_shape.clone(),
        })
    }

    /// Per-component sum of every element. Returns a 1-D vec with
    /// `num_components` entries. Tail dims are summed into a single
    /// scalar per component. CL-291.
    pub fn sum_per_component(&self) -> Vec<T> {
        let mut out = Vec::with_capacity(self.num_components());
        for i in 0..self.num_components() {
            let slice = self.component_slice(i);
            let mut acc = <T as num_traits::Zero>::zero();
            for &v in slice {
                acc += v;
            }
            out.push(acc);
        }
        out
    }

    /// Per-component mean of every element. Returns NaN for empty
    /// components (CORE-071 / #1765): a mean over zero elements is
    /// undefined, and torch's floating reductions agree —
    /// `torch.tensor([]).mean()` is `nan` (live session, torch
    /// 2.11.0+cu130). A fabricated finite `0` would silently bias
    /// downstream aggregation. CL-291.
    pub fn mean_per_component(&self) -> Vec<T> {
        let mut out = Vec::with_capacity(self.num_components());
        for i in 0..self.num_components() {
            let slice = self.component_slice(i);
            if slice.is_empty() {
                out.push(<T as num_traits::Float>::nan());
                continue;
            }
            let mut acc = <T as num_traits::Zero>::zero();
            for &v in slice {
                acc += v;
            }
            let n = T::from(slice.len()).unwrap();
            out.push(acc / n);
        }
        out
    }

    /// Convert to a padded dense tensor. The output shape is
    /// `[num_components, max_length] + tail_shape`; positions
    /// beyond each component's ragged length are filled with
    /// `pad_value`. CL-291.
    pub fn to_padded(&self, pad_value: T) -> FerrotorchResult<Tensor<T>> {
        let n = self.num_components();
        let mut max_len = 0usize;
        for i in 0..n {
            max_len = max_len.max(self.length(i));
        }
        // CORE-069 (#1763): actual tail product — a zero-containing tail
        // yields a zero row stride and an all-pad-free [n, max_len, ..0..]
        // output with numel 0 (torch jagged oracle: to_padded of
        // [zeros(3,0), zeros(2,0)] has shape (2, 3, 0)).
        let tail_numel: usize = crate::shape::numel(&self.tail_shape);
        let row_stride = max_len * tail_numel;

        let mut out = vec![pad_value; n * row_stride];
        for i in 0..n {
            let dst_base = i * row_stride;
            let slice = self.component_slice(i);
            out[dst_base..dst_base + slice.len()].copy_from_slice(slice);
        }

        let mut shape = vec![n, max_len];
        shape.extend_from_slice(&self.tail_shape);
        Tensor::from_storage(TensorStorage::cpu(out), shape, false)
    }

    /// Reconstruct a packed nested tensor from a padded dense
    /// tensor + per-component lengths. Inverse of [`to_padded`].
    ///
    /// # Errors
    ///
    /// - `tensor.ndim() < 2` — must have at least batch and
    ///   sequence dims.
    /// - `lengths.len() != tensor.shape()[0]`
    /// - Any `lengths[i] > tensor.shape()[1]` (would walk off the
    ///   padded row).
    pub fn from_padded(tensor: &Tensor<T>, lengths: &[usize]) -> FerrotorchResult<Self> {
        let shape = tensor.shape();
        if shape.len() < 2 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "PackedNestedTensor::from_padded: tensor must have at least \
                     2 dims (batch, sequence), got {shape:?}"
                ),
            });
        }
        let n = shape[0];
        let max_len = shape[1];
        let tail_shape: Vec<usize> = shape[2..].to_vec();
        // CORE-069 (#1763): actual tail product (0 for zero-containing
        // tails) — see `to_padded`.
        let tail_numel: usize = crate::shape::numel(&tail_shape);

        if lengths.len() != n {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "PackedNestedTensor::from_padded: lengths has {} entries but \
                     batch dim is {}",
                    lengths.len(),
                    n
                ),
            });
        }
        for (i, &len) in lengths.iter().enumerate() {
            if len > max_len {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "PackedNestedTensor::from_padded: lengths[{i}] = {len} \
                         exceeds max_len = {max_len}"
                    ),
                });
            }
        }

        // CORE-070 (#1764): materialize the LOGICAL view (strided walk) so
        // non-contiguous padded sources pack correctly; CUDA sources error
        // loudly (packed storage is CPU-resident by construction).
        if tensor.is_cuda() {
            return Err(FerrotorchError::GpuTensorNotAccessible);
        }
        let padded = tensor.data_vec()?;
        let row_stride = max_len * tail_numel;

        let mut data = Vec::with_capacity(lengths.iter().sum::<usize>() * tail_numel);
        let mut offsets = Vec::with_capacity(n + 1);
        offsets.push(0);
        for (i, &len) in lengths.iter().enumerate() {
            let src_base = i * row_stride;
            let src_end = src_base + len * tail_numel;
            data.extend_from_slice(&padded[src_base..src_end]);
            offsets.push(data.len());
        }

        // CORE-068 (#1762): every constructor funnels through the
        // centralized layout validation.
        validate_packed_layout(data.len(), &offsets, lengths, &tail_shape)?;

        Ok(Self {
            data,
            offsets,
            lengths: lengths.to_vec(),
            tail_shape,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tensor(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data), shape, false).unwrap()
    }

    fn make_tensor_f64(data: Vec<f64>, shape: Vec<usize>) -> Tensor<f64> {
        Tensor::from_storage(TensorStorage::cpu(data), shape, false).unwrap()
    }

    // --- NestedTensor construction ---

    #[test]
    fn test_nested_construction() {
        let t1 = make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
        let t2 = make_tensor(vec![7.0, 8.0, 9.0, 10.0], vec![2, 2]);

        let nt = NestedTensor::new(vec![t1, t2], 0).unwrap();

        assert_eq!(nt.num_components(), 2);
        assert_eq!(nt.ragged_dim(), 0);
        assert_eq!(nt.ndim(), 2);
        assert_eq!(nt.ragged_lengths(), vec![3, 2]);
    }

    #[test]
    fn test_nested_rejects_empty() {
        let result = NestedTensor::<f32>::new(vec![], 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_nested_rejects_shape_mismatch() {
        let t1 = make_tensor(vec![1.0; 6], vec![3, 2]);
        let t2 = make_tensor(vec![1.0; 6], vec![2, 3]); // dim 1 differs

        let result = NestedTensor::new(vec![t1, t2], 0);
        assert!(result.is_err());
    }

    // --- to_padded / from_padded round-trip (ragged_dim=0) ---

    #[test]
    fn test_to_padded_from_padded_ragged_dim_0() {
        let t1 = make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
        let t2 = make_tensor(vec![7.0, 8.0, 9.0, 10.0], vec![2, 2]);

        let nt = NestedTensor::new(vec![t1, t2], 0).unwrap();
        let padded = nt.to_padded(0.0).unwrap();

        assert_eq!(padded.shape(), &[2, 3, 2]); // batch=2, max_len=3, d=2

        let lengths = nt.ragged_lengths();
        let reconstructed = NestedTensor::from_padded(&padded, &lengths, 0).unwrap();

        assert_eq!(reconstructed.num_components(), 2);

        let r0 = reconstructed.tensors()[0].data().unwrap();
        assert_eq!(r0, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let r1 = reconstructed.tensors()[1].data().unwrap();
        assert_eq!(r1, &[7.0, 8.0, 9.0, 10.0]);
    }

    // --- from_padded round-trip for ragged_dim != 0 ---

    #[test]
    // reason: to_padded copies component values verbatim and writes the literal
    // pad value (0.0) into pad slots — no float arithmetic, so bitwise equality
    // with the source literals is the correct assertion.
    #[allow(clippy::float_cmp)]
    fn test_from_padded_round_trip_ragged_dim_1() {
        // Component tensors: shape [2, L_i] where dim 1 is ragged.
        // t1: [2, 3], t2: [2, 2]
        let t1 = make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let t2 = make_tensor(vec![7.0, 8.0, 9.0, 10.0], vec![2, 2]);

        let nt = NestedTensor::new(vec![t1, t2], 1).unwrap();

        // to_padded: output shape [batch=2, 2, max_col=3]
        let padded = nt.to_padded(0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 3]);

        let padded_data = padded.data().unwrap();
        // batch 0: [[1,2,3],[4,5,6]]
        assert_eq!(padded_data[0], 1.0);
        assert_eq!(padded_data[1], 2.0);
        assert_eq!(padded_data[2], 3.0);
        assert_eq!(padded_data[3], 4.0);
        assert_eq!(padded_data[4], 5.0);
        assert_eq!(padded_data[5], 6.0);
        // batch 1: [[7,8,0],[9,10,0]]
        assert_eq!(padded_data[6], 7.0);
        assert_eq!(padded_data[7], 8.0);
        assert_eq!(padded_data[8], 0.0); // pad
        assert_eq!(padded_data[9], 9.0);
        assert_eq!(padded_data[10], 10.0);
        assert_eq!(padded_data[11], 0.0); // pad

        // Reconstruct from padded.
        let lengths = nt.ragged_lengths();
        assert_eq!(lengths, vec![3, 2]);
        let reconstructed = NestedTensor::from_padded(&padded, &lengths, 1).unwrap();

        assert_eq!(reconstructed.num_components(), 2);
        assert_eq!(reconstructed.tensors()[0].shape(), &[2, 3]);
        assert_eq!(reconstructed.tensors()[1].shape(), &[2, 2]);

        let r0 = reconstructed.tensors()[0].data().unwrap();
        assert_eq!(r0, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let r1 = reconstructed.tensors()[1].data().unwrap();
        assert_eq!(r1, &[7.0, 8.0, 9.0, 10.0]);
    }

    // --- scaled dot-product attention ---

    #[test]
    fn test_sdpa_basic() {
        // Single component: Q=[2,4], K=[3,4], V=[3,5]
        let q = make_tensor(vec![1.0; 8], vec![2, 4]);
        let k = make_tensor(vec![1.0; 12], vec![3, 4]);
        let v = make_tensor(vec![1.0; 15], vec![3, 5]);

        let qn = NestedTensor::new(vec![q], 0).unwrap();
        let kn = NestedTensor::new(vec![k], 0).unwrap();
        let vn = NestedTensor::new(vec![v], 0).unwrap();

        let result = nested_scaled_dot_product_attention(&qn, &kn, &vn).unwrap();

        assert_eq!(result.num_components(), 1);
        assert_eq!(result.tensors()[0].shape(), &[2, 5]);

        // With uniform values, softmax should produce uniform weights,
        // and the output should be close to 1.0 everywhere.
        let out = result.tensors()[0].data().unwrap();
        for &val in out {
            assert!((val - 1.0).abs() < 1e-5, "expected ~1.0, got {val}");
        }
    }

    #[test]
    fn test_sdpa_f64() {
        // Verify it works with f64.
        let q = make_tensor_f64(vec![1.0; 8], vec![2, 4]);
        let k = make_tensor_f64(vec![1.0; 12], vec![3, 4]);
        let v = make_tensor_f64(vec![1.0; 15], vec![3, 5]);

        let qn = NestedTensor::new(vec![q], 0).unwrap();
        let kn = NestedTensor::new(vec![k], 0).unwrap();
        let vn = NestedTensor::new(vec![v], 0).unwrap();

        let result = nested_scaled_dot_product_attention(&qn, &kn, &vn).unwrap();

        assert_eq!(result.num_components(), 1);
        assert_eq!(result.tensors()[0].shape(), &[2, 5]);
    }

    // --- softmax degenerate case: all -inf -> NaN ---

    #[test]
    fn test_softmax_all_neg_inf_produces_nan() {
        let mut data = vec![f32::NEG_INFINITY; 6];
        softmax_rows_inplace(&mut data, 2, 3);

        for val in &data {
            assert!(val.is_nan(), "expected NaN for all -inf input, got {val}");
        }
    }

    #[test]
    fn test_softmax_normal_case() {
        let mut data = vec![1.0f32, 2.0, 3.0];
        softmax_rows_inplace(&mut data, 1, 3);

        let sum: f32 = data.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "softmax should sum to 1, got {sum}"
        );
        assert!(data[0] < data[1]);
        assert!(data[1] < data[2]);
    }

    // ────────────────────────────────────────────────────────────────
    // CL-291: PackedNestedTensor tests
    // ────────────────────────────────────────────────────────────────

    #[test]
    fn packed_from_sequences_1d() {
        // Three 1-D sequences with empty tail shape.
        let seqs = vec![
            vec![1.0f32, 2.0, 3.0],
            vec![4.0, 5.0, 6.0, 7.0, 8.0],
            vec![9.0, 10.0],
        ];
        let lengths = vec![3usize, 5, 2];
        let pnt = PackedNestedTensor::from_sequences(seqs, &lengths, &[]).unwrap();

        assert_eq!(pnt.num_components(), 3);
        assert_eq!(pnt.offsets(), &[0, 3, 8, 10]);
        assert_eq!(pnt.total_numel(), 10);
        assert_eq!(pnt.length(0), 3);
        assert_eq!(pnt.length(1), 5);
        assert_eq!(pnt.length(2), 2);
        assert_eq!(pnt.component_slice(0), &[1.0, 2.0, 3.0]);
        assert_eq!(pnt.component_slice(1), &[4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(pnt.component_slice(2), &[9.0, 10.0]);
    }

    #[test]
    fn packed_from_sequences_with_tail_shape() {
        // Two 2-D sequences, ragged on dim 0, tail shape [4].
        let seqs = vec![
            vec![
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ], // len=3
            vec![13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0], // len=2
        ];
        let lengths = vec![3usize, 2];
        let tail = vec![4usize];
        let pnt = PackedNestedTensor::from_sequences(seqs, &lengths, &tail).unwrap();

        assert_eq!(pnt.num_components(), 2);
        assert_eq!(pnt.offsets(), &[0, 12, 20]);
        assert_eq!(pnt.length(0), 3);
        assert_eq!(pnt.length(1), 2);
        assert_eq!(pnt.tail_shape(), &[4]);
    }

    #[test]
    fn packed_rejects_empty_sequences_list() {
        let result = PackedNestedTensor::<f32>::from_sequences(vec![], &[], &[]);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("at least one sequence"));
    }

    #[test]
    fn packed_rejects_mismatched_sequence_length() {
        let seqs = vec![vec![1.0f32, 2.0, 3.0]]; // 3 elements
        let lengths = vec![2usize]; // expects 2*tail_numel=2
        let result = PackedNestedTensor::from_sequences(seqs, &lengths, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn packed_rejects_mismatched_sequences_vs_lengths() {
        let seqs = vec![vec![1.0f32, 2.0]];
        let lengths = vec![2usize, 3];
        let result = PackedNestedTensor::from_sequences(seqs, &lengths, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn packed_map_applies_fn_to_every_element() {
        let pnt = PackedNestedTensor::from_sequences(
            vec![vec![1.0f32, -2.0, 3.0], vec![-4.0, 5.0]],
            &[3usize, 2],
            &[],
        )
        .unwrap();
        // ReLU via map.
        let relu = pnt.map(|x: f32| x.max(0.0));
        assert_eq!(relu.data(), &[1.0, 0.0, 3.0, 0.0, 5.0]);
        // Offsets preserved.
        assert_eq!(relu.offsets(), pnt.offsets());
    }

    #[test]
    fn packed_add_sub_mul_div() {
        let a = PackedNestedTensor::from_sequences(
            vec![vec![10.0f32, 20.0, 30.0], vec![40.0, 50.0]],
            &[3usize, 2],
            &[],
        )
        .unwrap();
        let b = PackedNestedTensor::from_sequences(
            vec![vec![1.0f32, 2.0, 3.0], vec![4.0, 5.0]],
            &[3usize, 2],
            &[],
        )
        .unwrap();

        assert_eq!(a.add(&b).unwrap().data(), &[11.0, 22.0, 33.0, 44.0, 55.0]);
        assert_eq!(a.sub(&b).unwrap().data(), &[9.0, 18.0, 27.0, 36.0, 45.0]);
        assert_eq!(a.mul(&b).unwrap().data(), &[10.0, 40.0, 90.0, 160.0, 250.0]);
        assert_eq!(a.div(&b).unwrap().data(), &[10.0, 10.0, 10.0, 10.0, 10.0]);
    }

    #[test]
    fn packed_add_rejects_mismatched_offsets() {
        let a = PackedNestedTensor::from_sequences(vec![vec![1.0f32, 2.0, 3.0]], &[3usize], &[])
            .unwrap();
        let b =
            PackedNestedTensor::from_sequences(vec![vec![1.0f32, 2.0]], &[2usize], &[]).unwrap();
        let result = a.add(&b);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("offsets mismatch"));
    }

    #[test]
    fn packed_add_rejects_mismatched_tail_shape() {
        let a = PackedNestedTensor::from_sequences(
            vec![vec![1.0f32, 2.0, 3.0, 4.0]],
            &[2usize],
            &[2], // tail [2]
        )
        .unwrap();
        let b = PackedNestedTensor::from_sequences(
            vec![vec![1.0f32, 2.0, 3.0, 4.0]],
            &[4usize],
            &[], // tail []
        )
        .unwrap();
        // Offsets match (both [0, 4]) but tail shapes differ.
        let result = a.add(&b);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("tail shape mismatch"));
    }

    #[test]
    fn packed_sum_per_component() {
        let pnt = PackedNestedTensor::from_sequences(
            vec![
                vec![1.0f32, 2.0, 3.0],       // sum = 6
                vec![10.0, 20.0, 30.0, 40.0], // sum = 100
                vec![5.0],                    // sum = 5
            ],
            &[3usize, 4, 1],
            &[],
        )
        .unwrap();
        let sums = pnt.sum_per_component();
        assert_eq!(sums, vec![6.0, 100.0, 5.0]);
    }

    #[test]
    fn packed_mean_per_component() {
        let pnt = PackedNestedTensor::from_sequences(
            vec![
                vec![2.0f32, 4.0, 6.0],       // mean = 4
                vec![10.0, 20.0, 30.0, 40.0], // mean = 25
                vec![7.0],                    // mean = 7
            ],
            &[3usize, 4, 1],
            &[],
        )
        .unwrap();
        let means = pnt.mean_per_component();
        assert_eq!(means, vec![4.0, 25.0, 7.0]);
    }

    #[test]
    // reason: 1.5 == 3.0 / 2.0 is exact in binary; bitwise equality is the
    // right assertion for the non-empty component.
    #[allow(clippy::float_cmp)]
    fn packed_mean_empty_component_is_nan() {
        // CORE-071 (#1765): a mean over zero elements is undefined; torch
        // oracle (live session, torch 2.11.0+cu130):
        //   >>> torch.tensor([]).mean().item()
        //   nan
        let pnt =
            PackedNestedTensor::from_sequences(vec![vec![1.0f32, 2.0], vec![]], &[2usize, 0], &[])
                .unwrap();
        let means = pnt.mean_per_component();
        assert_eq!(means[0], 1.5);
        assert!(
            means[1].is_nan(),
            "empty component mean must be NaN (torch parity), got {}",
            means[1]
        );
    }

    #[test]
    fn packed_to_padded_pads_with_value() {
        let pnt = PackedNestedTensor::from_sequences(
            vec![
                vec![1.0f32, 2.0, 3.0],
                vec![4.0, 5.0],
                vec![6.0, 7.0, 8.0, 9.0],
            ],
            &[3usize, 2, 4],
            &[],
        )
        .unwrap();

        let padded = pnt.to_padded(-1.0).unwrap();
        // [3 components × 4 max_len] = 12 elements
        assert_eq!(padded.shape(), &[3, 4]);
        let data = padded.data().unwrap();
        assert_eq!(
            data,
            &[
                1.0, 2.0, 3.0, -1.0, // first: [1,2,3] + pad
                4.0, 5.0, -1.0, -1.0, // second: [4,5] + pad
                6.0, 7.0, 8.0, 9.0, // third: [6,7,8,9]
            ]
        );
    }

    #[test]
    fn packed_to_padded_with_tail_shape() {
        // Two components of shape [L, 2], lengths 2 and 1.
        let pnt = PackedNestedTensor::from_sequences(
            vec![
                vec![1.0f32, 2.0, 3.0, 4.0], // 2 rows of 2
                vec![5.0, 6.0],              // 1 row of 2
            ],
            &[2usize, 1],
            &[2],
        )
        .unwrap();

        let padded = pnt.to_padded(0.0).unwrap();
        // [2 components, 2 max_len, 2 tail] = 8 elements
        assert_eq!(padded.shape(), &[2, 2, 2]);
        let data = padded.data().unwrap();
        assert_eq!(
            data,
            &[
                1.0, 2.0, 3.0, 4.0, // first: [[1,2],[3,4]]
                5.0, 6.0, 0.0, 0.0, // second: [[5,6], pad]
            ]
        );
    }

    #[test]
    fn packed_from_padded_inverse_of_to_padded() {
        // Roundtrip: pack → pad → unpack.
        let orig = PackedNestedTensor::from_sequences(
            vec![
                vec![1.0f32, 2.0, 3.0],
                vec![4.0, 5.0],
                vec![6.0, 7.0, 8.0, 9.0],
            ],
            &[3usize, 2, 4],
            &[],
        )
        .unwrap();

        let padded = orig.to_padded(-99.0).unwrap();
        let recovered = PackedNestedTensor::from_padded(&padded, &[3, 2, 4]).unwrap();

        assert_eq!(recovered.offsets(), orig.offsets());
        assert_eq!(recovered.data(), orig.data());
    }

    #[test]
    fn packed_from_padded_rejects_length_exceeding_max_len() {
        let data: Vec<f32> = vec![0.0; 12];
        let t = make_tensor(data, vec![3, 4]);
        let result = PackedNestedTensor::from_padded(&t, &[3, 5, 2]); // 5 > 4
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("exceeds max_len"));
    }

    #[test]
    fn packed_from_padded_rejects_lengths_count_mismatch() {
        let t = make_tensor(vec![0.0f32; 12], vec![3, 4]);
        let result = PackedNestedTensor::from_padded(&t, &[3, 4]); // 2 lengths for batch 3
        assert!(result.is_err());
    }

    #[test]
    fn packed_from_nested_and_back() {
        // Roundtrip from list-of-tensors NestedTensor to packed and back.
        let t1 = make_tensor(vec![1.0, 2.0, 3.0], vec![3]);
        let t2 = make_tensor(vec![4.0, 5.0], vec![2]);
        let nested = NestedTensor::new(vec![t1, t2], 0).unwrap();

        let packed = PackedNestedTensor::from_nested(&nested).unwrap();
        assert_eq!(packed.num_components(), 2);
        assert_eq!(packed.data(), &[1.0, 2.0, 3.0, 4.0, 5.0]);

        let round_trip = packed.to_nested().unwrap();
        assert_eq!(round_trip.num_components(), 2);
        assert_eq!(round_trip.tensors()[0].shape(), &[3]);
        assert_eq!(round_trip.tensors()[1].shape(), &[2]);
        assert_eq!(round_trip.tensors()[0].data().unwrap(), &[1.0, 2.0, 3.0]);
        assert_eq!(round_trip.tensors()[1].data().unwrap(), &[4.0, 5.0]);
    }

    #[test]
    fn packed_from_nested_rejects_non_zero_ragged_dim() {
        // The packed layout requires ragged_dim == 0.
        let t1 = make_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let t2 = make_tensor(vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0], vec![2, 3]);
        let nested = NestedTensor::new(vec![t1, t2], 1).unwrap();

        let result = PackedNestedTensor::from_nested(&nested);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("ragged_dim == 0"));
    }

    #[test]
    fn packed_f64_works_like_f32() {
        let pnt = PackedNestedTensor::from_sequences(
            vec![vec![1.0f64, 2.0], vec![3.0, 4.0, 5.0]],
            &[2usize, 3],
            &[],
        )
        .unwrap();
        assert_eq!(pnt.sum_per_component(), vec![3.0, 12.0]);
        let doubled = pnt.map(|x: f64| x * 2.0);
        assert_eq!(doubled.data(), &[2.0, 4.0, 6.0, 8.0, 10.0]);

        // Silence the unused `make_tensor_f64` warning by exercising it too.
        let dense = make_tensor_f64(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert_eq!(dense.shape(), &[2, 2]);
    }
}
