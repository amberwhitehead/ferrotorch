//! `as_strided` family — direct stride manipulation on tensors.
//!
//! Mirrors `torch.Tensor.as_strided`, `torch.as_strided_copy`, and
//! `torch.as_strided_scatter`. Strides are given in *element* units (matching
//! torch and ferrotorch's existing `Tensor::strides`), not bytes. A stride
//! may be zero (broadcast-style replication) or negative (reverse iteration).
//!
//! # Operations
//!
//! - [`as_strided`] returns a zero-copy view with the requested
//!   shape, strides, and storage offset. Works on any device because it is
//!   pure metadata — no kernels are dispatched.
//! - [`as_strided_copy`] materialises that view into a new contiguous
//!   tensor. CPU and CUDA paths exist (CUDA dispatches to the existing
//!   `strided_copy_f32`/`strided_copy_f64` GPU kernels). Other devices
//!   error with [`FerrotorchError::NotImplementedOnCuda`] until a kernel
//!   lands.
//! - [`as_strided_scatter`] is the inverse of `as_strided`: returns a copy
//!   of `self` with the strided positions overwritten by `src`. CPU only
//!   today; CUDA support is tracked separately.
//!
//! # Autograd
//!
//! All three operations are differentiable (CORE-058/059/060,
//! #1752/#1753/#1754):
//!
//! - [`as_strided`] / [`as_strided_copy`] attach [`AsStridedBackward`],
//!   which implements torch's full `as_strided_backward` algorithm
//!   (`torch/csrc/autograd/FunctionsManual.cpp`, baseline `2ec0222669`):
//!   scatter-ADD the upstream gradient into a base buffer spanning the
//!   input and output geometries (overlapping view positions SUM, not
//!   overwrite), divide by the visit count when the input geometry itself
//!   may overlap, then gather back out through the input's own
//!   (shape, strides, offset) geometry. Offsets are handled as deltas
//!   against the shared minimum reachable offset, so offset / transposed /
//!   chained input views are correct, and negative strides (which torch
//!   rejects in the forward) size the base buffer from the signed
//!   reachable span.
//! - [`as_strided_scatter`] attaches [`AsStridedScatterBackward`]:
//!   `d/d src` is the upstream gradient gathered at the view geometry
//!   (matches torch); `d/d self` is the upstream gradient with the view
//!   region zeroed — the finite-difference Jacobian. torch 2.11.0's
//!   analytic formula returns the opposite masking and fails its own
//!   `gradcheck`; the deliberate divergence is tracked in #1959.
//!
//! # Safety
//!
//! Like torch, this is **not** safe under in-place mutation when the
//! requested strides cause overlapping memory accesses. Reads always
//! return well-defined values (since storage is initialised), but
//! `tensor.as_strided(...)`-then-`add_(...)`-style writes against
//! overlapping views are undefined behaviour and produce torch-equivalent
//! "unexpected results". Bounds are always validated; overlap is not
//! rejected.
//!
//! # GPU discipline
//!
//! - View construction is metadata-only and shares the same `Arc<Storage>`
//!   on every device. No silent device transfer.
//! - `as_strided_copy` on CUDA dispatches to the dedicated GPU kernel; it
//!   does not bounce data through host memory.
//! - `as_strided_scatter` on CUDA dispatches through `strided_copy_*` +
//!   `strided_scatter_*`; no host bounce.
//! - `AsStridedBackward` on CUDA keeps the gradient data on device
//!   end-to-end: non-overlapping output geometries use `strided_scatter_*`
//!   into a device zeros buffer; overlapping geometries use the i64-index
//!   `scatter_add_dim_*` atomic-add kernels. Only host-COMPUTED constants
//!   move host→device (the freshly built flat-index buffer, a zeros/ones
//!   fill) — exactly what torch's `index_add_`-based backward materialises
//!   as an `arange` index tensor.
//!
//! ## REQ status (per `.design/ferrotorch-core/stride_tricks.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | impl `Tensor::as_strided`; non-test consumer `crate::einsum`. |
//! | REQ-2 | SHIPPED | impl `Tensor::as_strided_copy` (CUDA via `strided_copy_*`, CPU walk); non-test consumer `crate::einsum`. |
//! | REQ-3 | SHIPPED | impl `Tensor::as_strided_scatter`; non-test consumer `AsStridedScatterBackward::backward` (base-grad masking). |
//! | REQ-4 | SHIPPED | impl `validate_bounds` + `stride_extent`; non-test consumer invoked from `as_strided` / `as_strided_scatter`. |
//! | REQ-5 | SHIPPED | impl `AsStridedBackward` + `GradFn::backward`; non-test consumer `Tensor::as_strided` attach path. |
//! | REQ-6 | SHIPPED | negative-stride support via `stride_extent`; non-test consumer reverse views. |
//! | REQ-7 | SHIPPED | zero-stride broadcast support; non-test consumer `Tensor::expand` family. |
//! | REQ-8 | SHIPPED | free-function wrappers `as_strided`, `as_strided_copy`, `as_strided_scatter`; non-test consumer downstream functional-style call sites. |

use std::sync::Arc;

use crate::dtype::{DType, Float};
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::gpu_dispatch::GpuBufferHandle;
use crate::storage::TensorStorage;
use crate::tensor::{GradFn, Tensor};

// ---------------------------------------------------------------------------
// Public free functions (mirroring torch.as_strided / torch.as_strided_copy)
// ---------------------------------------------------------------------------

/// Zero-copy strided view; see [`Tensor::as_strided`] for full docs.
pub fn as_strided<T: Float>(
    input: &Tensor<T>,
    size: &[usize],
    stride: &[isize],
    storage_offset: Option<usize>,
) -> FerrotorchResult<Tensor<T>> {
    input.as_strided(size, stride, storage_offset)
}

/// Materialised strided copy; see [`Tensor::as_strided_copy`] for full docs.
pub fn as_strided_copy<T: Float>(
    input: &Tensor<T>,
    size: &[usize],
    stride: &[isize],
    storage_offset: Option<usize>,
) -> FerrotorchResult<Tensor<T>> {
    input.as_strided_copy(size, stride, storage_offset)
}

/// Inverse of `as_strided`; see [`Tensor::as_strided_scatter`] for full docs.
pub fn as_strided_scatter<T: Float>(
    input: &Tensor<T>,
    src: &Tensor<T>,
    size: &[usize],
    stride: &[isize],
    storage_offset: Option<usize>,
) -> FerrotorchResult<Tensor<T>> {
    input.as_strided_scatter(src, size, stride, storage_offset)
}

// ---------------------------------------------------------------------------
// Internal: bounds validation
// ---------------------------------------------------------------------------

/// Compute the smallest and largest element offsets reachable by walking
/// every position in a strided view.
///
/// Returns `(min_offset, max_offset)` in element units, both inclusive.
/// For an empty view (`shape` contains a 0) returns `(0, 0)` to signal
/// "no positions reached" — the caller should treat that as trivially
/// in-bounds at any `storage_offset`.
fn stride_extent(shape: &[usize], stride: &[isize]) -> (i64, i64) {
    if shape.contains(&0) {
        return (0, 0);
    }
    let mut min_off: i64 = 0;
    let mut max_off: i64 = 0;
    for (&dim, &s) in shape.iter().zip(stride.iter()) {
        if dim == 0 {
            continue;
        }
        let last = (dim as i64 - 1) * s as i64;
        if last >= 0 {
            max_off += last;
        } else {
            min_off += last;
        }
    }
    (min_off, max_off)
}

/// Validate that the requested view fits within `storage_len`.
///
/// Returns `Ok(())` if every reachable offset (including the zero-position
/// origin at `storage_offset`) lies inside `[0, storage_len)`.
///
/// `pub(crate)`: also the bounds gate for the in-place view-region writes
/// in `Tensor::update_storage` / `Tensor::update_data` (#1938).
pub(crate) fn validate_bounds(
    op: &'static str,
    shape: &[usize],
    stride: &[isize],
    storage_offset: usize,
    storage_len: usize,
) -> FerrotorchResult<()> {
    if shape.len() != stride.len() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "{op}: shape and stride must have the same length (got {} vs {})",
                shape.len(),
                stride.len()
            ),
        });
    }

    // Empty view (any dim is zero) — nothing to read or write.
    if shape.contains(&0) {
        // Zero-element views are valid even at storage_offset == storage_len.
        if storage_offset > storage_len {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "{op}: storage_offset {storage_offset} > storage length {storage_len}"
                ),
            });
        }
        return Ok(());
    }

    let (min_off, max_off) = stride_extent(shape, stride);
    let lo = storage_offset as i64 + min_off;
    let hi = storage_offset as i64 + max_off;
    if lo < 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "{op}: storage_offset {storage_offset} with strides {stride:?} reaches negative \
                 offset {lo} (out of bounds)"
            ),
        });
    }
    if hi >= storage_len as i64 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "{op}: storage_offset {storage_offset} with shape {shape:?} and strides \
                 {stride:?} reaches offset {hi}, beyond storage length {storage_len}"
            ),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Internal: strided-geometry helpers shared by the backward passes
// ---------------------------------------------------------------------------

/// Conservative memory-overlap test for a strided geometry: `false` only
/// when the layout provably maps distinct logical positions to distinct
/// storage offsets. Mirrors `_maybe_overlapping_memory` in
/// `torch/csrc/autograd/FunctionsManual.cpp` (sort the size>1 dims by
/// stride; each stride must exceed the maximum offset addressable by all
/// smaller-stride dims). Extensions over torch (which never sees them
/// here): a zero stride on a size>1 dim always overlaps; a negative stride
/// is conservatively treated as maybe-overlapping — correctness is
/// preserved because the divide-by-count step computes exact multiplicity.
fn maybe_overlapping_memory(shape: &[usize], stride: &[isize]) -> bool {
    let mut dims: Vec<(usize, isize)> = shape
        .iter()
        .zip(stride.iter())
        .filter(|&(&sz, _)| sz > 1)
        .map(|(&sz, &st)| (sz, st))
        .collect();
    if dims.iter().any(|&(_, st)| st <= 0) {
        return true;
    }
    dims.sort_by_key(|&(_, st)| st);
    let mut max_reach: i64 = 0;
    for (sz, st) in dims {
        if (st as i64) <= max_reach {
            return true;
        }
        max_reach += (st as i64) * (sz as i64 - 1);
    }
    false
}

/// Walk every logical position of a strided geometry in C order (last axis
/// fastest), calling `f(linear_index, flat_storage_offset)`. `base_offset`
/// is added to every flat offset; strides may be zero or negative. The
/// caller guarantees the geometry was bounds-validated against the buffer
/// the flat offsets index into.
fn for_each_strided_offset(
    shape: &[usize],
    stride: &[isize],
    base_offset: i64,
    mut f: impl FnMut(usize, i64),
) {
    let numel: usize = shape.iter().product();
    if numel == 0 {
        return;
    }
    let ndim = shape.len();
    let mut indices = vec![0usize; ndim];
    let mut flat = base_offset;
    for i in 0..numel {
        f(i, flat);
        for d in (0..ndim).rev() {
            indices[d] += 1;
            flat += stride[d] as i64;
            if indices[d] < shape[d] {
                break;
            }
            flat -= (shape[d] as i64) * (stride[d] as i64);
            indices[d] = 0;
        }
    }
}

/// Signed inclusive range `[lo, hi]` of storage offsets reachable by a
/// strided geometry rooted at `offset`. Caller guarantees the shape is
/// non-empty.
fn reachable_span(shape: &[usize], stride: &[isize], offset: usize) -> (i64, i64) {
    let (min_off, max_off) = stride_extent(shape, stride);
    (offset as i64 + min_off, offset as i64 + max_off)
}

/// Materialise `t` into a tensor whose backing buffer is EXACTLY its
/// logical C-order elements (offset 0, `storage_len == numel`). Tensors
/// already in that form are returned as cheap clones. Device-preserving:
/// CUDA tensors gather via the `strided_copy_*` kernels (no host bounce).
///
/// The kernel-facing backward paths need this exactness because they read
/// the raw buffer linearly (`scatter_add_1d_*`) or re-interpret offsets
/// against a fresh logical buffer.
fn exact_contiguous<T: Float>(t: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if t.is_contiguous() && t.storage_offset() == 0 && t.storage_len() == t.numel() {
        return Ok(t.clone());
    }
    if t.is_cuda() {
        let storage = materialize_strided_cuda(t)?;
        return Tensor::from_storage(storage, t.shape().to_vec(), false);
    }
    let data = t.data_vec()?;
    Tensor::from_storage(TensorStorage::cpu(data), t.shape().to_vec(), false)
}

/// Upload `len` copies of `value` (an f32/f64 scalar fill, dtype-tagged per
/// `T`) to a device buffer on `ordinal`. Used for the zeros scatter base
/// and the ones multiplicity source of the CUDA backward — host-computed
/// constants, not tensor-data round trips.
fn upload_scalar_fill<T: Float>(
    value: f64,
    len: usize,
    ordinal: usize,
) -> FerrotorchResult<GpuBufferHandle> {
    use std::any::TypeId;

    let backend = crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        let mut bytes = Vec::with_capacity(len * 4);
        for _ in 0..len {
            bytes.extend_from_slice(&(value as f32).to_ne_bytes());
        }
        backend.cpu_to_gpu(&bytes, DType::F32, ordinal)
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        let mut bytes = Vec::with_capacity(len * 8);
        for _ in 0..len {
            bytes.extend_from_slice(&value.to_ne_bytes());
        }
        backend.cpu_to_gpu(&bytes, DType::F64, ordinal)
    } else {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "as_strided backward (non-f32/f64 dtype)",
        })
    }
}

// ---------------------------------------------------------------------------
// Tensor methods (the impl block in tensor.rs is closed; we use a separate
// inherent impl here)
// ---------------------------------------------------------------------------

impl<T: Float> Tensor<T> {
    /// Build a zero-copy view with the given shape, strides (element units),
    /// and storage offset. If `storage_offset` is `None`, the input's
    /// existing offset is used.
    ///
    /// Equivalent to `torch.Tensor.as_strided(size, stride, storage_offset)`.
    /// Works on any device — no data movement.
    ///
    /// Validates that every reachable offset stays inside the underlying
    /// storage. **Does not** reject overlapping views: those are useful for
    /// constructing Toeplitz matrices, sliding windows, broadcast views,
    /// etc. As in torch, in-place writes against an overlapping view have
    /// undefined behaviour.
    pub fn as_strided(
        &self,
        size: &[usize],
        stride: &[isize],
        storage_offset: Option<usize>,
    ) -> FerrotorchResult<Tensor<T>> {
        let offset = storage_offset.unwrap_or_else(|| self.storage_offset());
        let storage_len = self.storage_len();
        validate_bounds("as_strided", size, stride, offset, storage_len)?;

        // No-grad fast path: pure metadata change, zero-copy on every device.
        if !crate::autograd::no_grad::is_grad_enabled() || !self.requires_grad() {
            return Ok(self.stride_view(size.to_vec(), stride.to_vec(), offset));
        }

        // Grad path: attach AsStridedBackward so autograd scatters the
        // upstream grad back into the original shape on backward.
        let grad_fn = Arc::new(AsStridedBackward::new(
            self.clone(),
            size.to_vec(),
            stride.to_vec(),
            offset,
        ));
        Ok(self.stride_view_operation(size.to_vec(), stride.to_vec(), offset, grad_fn))
    }

    /// Materialised strided copy: returns a new contiguous tensor whose
    /// values are the elements that `as_strided(size, stride, offset)` would
    /// read.
    ///
    /// On CUDA tensors this dispatches to the existing `strided_copy_f32`
    /// / `strided_copy_f64` GPU kernels (no host bounce). On CPU it walks
    /// the multi-index. On other devices (e.g. XPU) it returns
    /// [`FerrotorchError::NotImplementedOnCuda`] — install a kernel before
    /// using this on those devices.
    ///
    /// Differentiable like `torch.as_strided_copy` (CORE-060 / #1754): when
    /// grad is enabled and `self` tracks gradients, the output carries
    /// [`AsStridedBackward`] — the copy is the identity on values, so its
    /// VJP equals the view's VJP (torch generates `AsStridedBackward0_copy`
    /// from the same derivative entry).
    pub fn as_strided_copy(
        &self,
        size: &[usize],
        stride: &[isize],
        storage_offset: Option<usize>,
    ) -> FerrotorchResult<Tensor<T>> {
        let offset = storage_offset.unwrap_or_else(|| self.storage_offset());
        validate_bounds("as_strided_copy", size, stride, offset, self.storage_len())?;

        // Materialise from a plain metadata view (no intermediate grad
        // node; the backward node attaches to the OUTPUT below,
        // referencing `self` directly).
        let view = self.stride_view(size.to_vec(), stride.to_vec(), offset);
        let storage = if view.is_cuda() {
            // Direct GPU strided_copy dispatch; `view.data_vec()` would
            // bounce through host first, which violates GPU discipline.
            materialize_strided_cuda(&view)?
        } else {
            // `data_vec` already understands non-contiguous CPU layouts.
            TensorStorage::cpu(view.data_vec()?)
        };

        if crate::autograd::no_grad::is_grad_enabled() && self.requires_grad() {
            let grad_fn = Arc::new(AsStridedBackward::new(
                self.clone(),
                size.to_vec(),
                stride.to_vec(),
                offset,
            ));
            Tensor::from_operation(storage, size.to_vec(), grad_fn)
        } else {
            Tensor::from_storage(storage, size.to_vec(), false)
        }
    }

    /// Inverse of [`as_strided`]: return a copy of `self` with `src` written
    /// into the strided positions described by `(size, stride, offset)`.
    /// Positions outside that view retain `self`'s values.
    ///
    /// Equivalent to `torch.as_strided_scatter`. The CUDA path
    /// dispatches through the GPU backend (via the
    /// `strided_copy` + `strided_scatter` kernels) — no host bounce.
    ///
    /// Differentiable w.r.t. BOTH `self` and `src` (CORE-060 / #1754) via
    /// [`AsStridedScatterBackward`]. The `src` gradient (a gather at the
    /// view geometry) matches torch; the `self` gradient pins the
    /// finite-difference Jacobian, deliberately diverging from torch
    /// 2.11.0's self-inconsistent analytic formula — see #1959.
    ///
    /// Like torch, the gradients assume the view geometry does not overlap
    /// itself; the forward's last-write-wins resolution of overlapping
    /// destinations is not reflected in the `src` gather.
    pub fn as_strided_scatter(
        &self,
        src: &Tensor<T>,
        size: &[usize],
        stride: &[isize],
        storage_offset: Option<usize>,
    ) -> FerrotorchResult<Tensor<T>> {
        let offset = storage_offset.unwrap_or(0);
        let storage_len = self.numel();
        validate_bounds("as_strided_scatter", size, stride, offset, storage_len)?;

        if size.len() != src.shape().len() || size.iter().zip(src.shape()).any(|(a, b)| a != b) {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "as_strided_scatter: src shape {:?} does not match requested view shape {size:?}",
                    src.shape()
                ),
            });
        }

        if self.is_cuda() != src.is_cuda() {
            return Err(FerrotorchError::DeviceMismatch {
                expected: self.device(),
                got: src.device(),
            });
        }

        let storage = if self.is_cuda() {
            scatter_on_cuda(self, src, size, stride, offset)?
        } else {
            // CPU path: start from a contiguous copy of self, walk src in
            // C-order and write into the strided positions.
            let mut buf = self.data_vec()?;
            let src_data = src.data_vec()?;
            // Bounds were validated; every flat offset is in [0, storage_len).
            for_each_strided_offset(size, stride, offset as i64, |src_i, flat| {
                buf[flat as usize] = src_data[src_i];
            });
            TensorStorage::cpu(buf)
        };

        if crate::autograd::no_grad::is_grad_enabled()
            && (self.requires_grad() || src.requires_grad())
        {
            let grad_fn = Arc::new(AsStridedScatterBackward::new(
                self.clone(),
                src.clone(),
                size.to_vec(),
                stride.to_vec(),
                offset,
            ));
            Tensor::from_operation(storage, self.shape().to_vec(), grad_fn)
        } else {
            Tensor::from_storage(storage, self.shape().to_vec(), false)
        }
    }
}

// ---------------------------------------------------------------------------
// Autograd: AsStridedBackward
// ---------------------------------------------------------------------------

/// VJP for `as_strided(input, size, stride, offset)` and
/// `as_strided_copy(input, size, stride, offset)`.
///
/// Implements torch's full `as_strided_backward` algorithm
/// (`torch/csrc/autograd/FunctionsManual.cpp`, baseline `2ec0222669`;
/// derivative entry in `tools/autograd/derivatives.yaml: as_strided`):
///
/// 1. Allocate a zero "base" buffer spanning the union of the reachable
///    storage ranges of the INPUT geometry (`input.shape/strides/offset`)
///    and the OUTPUT geometry (`size/stride/storage_offset`). Offsets are
///    rebased as deltas against the shared minimum reachable offset
///    (torch's `shared_offset` trick), so nonzero-offset input views
///    (narrow, chained `as_strided`) are handled, and negative strides
///    (rejected by torch's forward but supported here) size the buffer
///    from the signed span (CORE-059 / #1753).
/// 2. Scatter-ADD `grad_output` into the base at the output geometry —
///    overlapping view positions (sliding windows, zero strides) SUM their
///    upstream gradients instead of keeping the last write
///    (CORE-058 / #1752).
/// 3. If the input geometry itself may overlap (a chained overlapping
///    view), divide each base cell by its visit count, mirroring torch's
///    `index_add_`-of-ones step.
/// 4. Gather the base back out through the input geometry into a
///    contiguous tensor of `input.shape()`.
///
/// Device-preserving: the CPU path is pure `Vec` walks; the CUDA path
/// composes `strided_scatter_*` (non-overlapping outputs),
/// `scatter_add_dim_*` (overlapping outputs; i64 indices, atomic adds),
/// `div`, and `strided_copy_*` so the gradient data never bounces through
/// the host.
#[derive(Debug)]
pub struct AsStridedBackward<T: Float> {
    input: Tensor<T>,
    size: Vec<usize>,
    stride: Vec<isize>,
    storage_offset: usize,
}

impl<T: Float> AsStridedBackward<T> {
    pub fn new(
        input: Tensor<T>,
        size: Vec<usize>,
        stride: Vec<isize>,
        storage_offset: usize,
    ) -> Self {
        Self {
            input,
            size,
            stride,
            storage_offset,
        }
    }

    /// CPU half of [`GradFn::backward`]: plain `Vec` walks, no unsafe.
    fn backward_cpu(
        &self,
        grad_output: &Tensor<T>,
        base_len: usize,
        out_eff: i64,
        inp_eff: i64,
        inp_overlap: bool,
    ) -> FerrotorchResult<Tensor<T>> {
        let inp_shape = self.input.shape();
        let inp_strides = self.input.strides();
        let gdata = grad_output.data_vec()?; // logical C-order, any layout
        let zero = <T as num_traits::Zero>::zero();
        let one = <T as num_traits::One>::one();

        // Step 2: scatter-ADD at the output geometry.
        let mut base = vec![zero; base_len];
        for_each_strided_offset(&self.size, &self.stride, out_eff, |i, flat| {
            base[flat as usize] += gdata[i];
        });

        // Step 3: divide by visit count where the input geometry overlaps.
        if inp_overlap {
            let mut count = vec![zero; base_len];
            for_each_strided_offset(inp_shape, inp_strides, inp_eff, |_, flat| {
                count[flat as usize] += one;
            });
            for (b, c) in base.iter_mut().zip(count.iter()) {
                // Cells with count 0 are never gathered below; skipping the
                // division avoids materialising torch's 0/0 NaNs.
                if *c > one {
                    *b = *b / *c;
                }
            }
        }

        // Step 4: gather through the input geometry.
        let mut out = vec![zero; self.input.numel()];
        for_each_strided_offset(inp_shape, inp_strides, inp_eff, |i, flat| {
            out[i] = base[flat as usize];
        });
        Tensor::from_storage(TensorStorage::cpu(out), inp_shape.to_vec(), false)
    }

    /// CUDA half of [`GradFn::backward`]: gradient data stays on device.
    fn backward_cuda(
        &self,
        grad_output: &Tensor<T>,
        base_len: usize,
        out_eff: i64,
        inp_eff: i64,
        inp_overlap: bool,
    ) -> FerrotorchResult<Tensor<T>> {
        use std::any::TypeId;

        let is_f32 = TypeId::of::<T>() == TypeId::of::<f32>();
        let is_f64 = TypeId::of::<T>() == TypeId::of::<f64>();
        if !is_f32 && !is_f64 {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "as_strided backward (non-f32/f64 dtype)",
            });
        }
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;

        // `scatter_add_1d_*` reads its grad buffer linearly and
        // `strided_scatter_*` consumes a contiguous src, so the upstream
        // grad must be exactly its logical elements.
        let grad = exact_contiguous(grad_output)?;
        let grad_buf = grad
            .storage()
            .gpu_handle()
            .ok_or(FerrotorchError::DeviceUnavailable)?;
        let ordinal = grad_buf.device_ordinal();

        let inp_shape = self.input.shape().to_vec();
        let inp_strides = self.input.strides().to_vec();

        // Step 2: scatter the upstream grad into the base buffer.
        let out_overlap = maybe_overlapping_memory(&self.size, &self.stride);
        let base_t: Tensor<T> = if out_overlap {
            // Aliasing destinations: atomic scatter-add via the modern
            // i64-index `scatter_add_dim_*` kernels (sm_60+ f64 atomics),
            // decomposed as [outer=1, out_dim=base_len, inner=1]: for every
            // t in [0, numel), base[idx[t]] += grad[t] on top of the zeros
            // base. (The legacy `scatter_add_1d_*` arms were rejected: f32-
            // encoded indices round above 2^24 and the f64 PTX transform
            // fails to JIT — see the #1960 thread.)
            let mut idx = vec![0usize; grad.numel()];
            for_each_strided_offset(&self.size, &self.stride, out_eff, |i, flat| {
                // flat ∈ [0, base_len) by construction (bounds-validated
                // geometry rebased to the shared minimum offset).
                idx[i] = flat as usize;
            });
            let idx_h = crate::ops::indexing::upload_index_i64(&idx, ordinal)?;
            let zeros_h = upload_scalar_fill::<T>(0.0, base_len, ordinal)?;
            let base_h = if is_f32 {
                backend.scatter_add_dim_f32(
                    &zeros_h,
                    &idx_h,
                    grad_buf,
                    1,
                    base_len,
                    idx.len(),
                    1,
                )?
            } else {
                backend.scatter_add_dim_f64(
                    &zeros_h,
                    &idx_h,
                    grad_buf,
                    1,
                    base_len,
                    idx.len(),
                    1,
                )?
            };
            Tensor::from_storage(TensorStorage::gpu(base_h), vec![base_len], false)?
        } else {
            // Distinct destinations: overwrite-scatter into zeros is
            // exactly scatter-add, with no index buffer needed.
            let mut base_h = upload_scalar_fill::<T>(0.0, base_len, ordinal)?;
            if is_f32 {
                backend.strided_scatter_f32(
                    grad_buf,
                    &mut base_h,
                    &self.size,
                    &self.stride,
                    out_eff as usize,
                )?;
            } else {
                backend.strided_scatter_f64(
                    grad_buf,
                    &mut base_h,
                    &self.size,
                    &self.stride,
                    out_eff as usize,
                )?;
            }
            Tensor::from_storage(TensorStorage::gpu(base_h), vec![base_len], false)?
        };

        // Step 3: divide by visit count where the input geometry overlaps.
        let base_t = if inp_overlap {
            let inp_numel = self.input.numel();
            let mut idx = vec![0usize; inp_numel];
            for_each_strided_offset(&inp_shape, &inp_strides, inp_eff, |i, flat| {
                idx[i] = flat as usize;
            });
            let idx_h = crate::ops::indexing::upload_index_i64(&idx, ordinal)?;
            let zeros_h = upload_scalar_fill::<T>(0.0, base_len, ordinal)?;
            let ones_h = upload_scalar_fill::<T>(1.0, inp_numel, ordinal)?;
            let count_h = if is_f32 {
                backend.scatter_add_dim_f32(&zeros_h, &idx_h, &ones_h, 1, base_len, inp_numel, 1)?
            } else {
                backend.scatter_add_dim_f64(&zeros_h, &idx_h, &ones_h, 1, base_len, inp_numel, 1)?
            };
            let count_t = Tensor::from_storage(TensorStorage::gpu(count_h), vec![base_len], false)?;
            // Cells with count 0 become 0/0 = NaN, exactly like torch's
            // `storage.div_(count)` ("this will give nan outside visible
            // range"); the gather below never reads them.
            base_t.div_t(&count_t)?
        } else {
            base_t
        };

        // Step 4: gather through the input geometry.
        let base_buf = base_t
            .storage()
            .gpu_handle()
            .ok_or(FerrotorchError::DeviceUnavailable)?;
        let out_h = if is_f32 {
            backend.strided_copy_f32(base_buf, &inp_shape, &inp_strides, inp_eff as usize)?
        } else {
            backend.strided_copy_f64(base_buf, &inp_shape, &inp_strides, inp_eff as usize)?
        };
        Tensor::from_storage(TensorStorage::gpu(out_h), inp_shape, false)
    }
}

impl<T: Float> GradFn<T> for AsStridedBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }

        let inp_shape = self.input.shape();
        let out_numel: usize = self.size.iter().product();

        // Empty output view (or empty input): nothing was read, so the
        // gradient is identically zero (torch returns
        // `at::zeros(input_geometry.sizes())` for size-0 dims).
        if out_numel == 0 || self.input.numel() == 0 {
            let zeros = crate::creation::zeros::<T>(inp_shape)?;
            let zeros = if self.input.is_cuda() {
                zeros.to(self.input.device())?
            } else {
                zeros
            };
            return Ok(vec![Some(zeros)]);
        }

        // Step 1: base-buffer geometry. Both spans were validated against
        // the SAME storage in their forwards, so the union is finite and
        // every effective offset is non-negative.
        let (in_lo, in_hi) =
            reachable_span(inp_shape, self.input.strides(), self.input.storage_offset());
        let (out_lo, out_hi) = reachable_span(&self.size, &self.stride, self.storage_offset);
        let shared = in_lo.min(out_lo);
        let base_len = (in_hi.max(out_hi) - shared + 1) as usize;
        let inp_eff = self.input.storage_offset() as i64 - shared;
        let out_eff = self.storage_offset as i64 - shared;
        let inp_overlap = maybe_overlapping_memory(inp_shape, self.input.strides());

        let grad_input = if grad_output.is_cuda() {
            self.backward_cuda(grad_output, base_len, out_eff, inp_eff, inp_overlap)?
        } else {
            self.backward_cpu(grad_output, base_len, out_eff, inp_eff, inp_overlap)?
        };
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "AsStridedBackward"
    }
}

// ---------------------------------------------------------------------------
// Autograd: AsStridedScatterBackward
// ---------------------------------------------------------------------------

/// VJP for `as_strided_scatter(input, src, size, stride, offset)`.
///
/// Forward: `out = input.clone(); out.as_strided(size, stride, offset)
/// .copy_(src)` — view positions come from `src`, everything else passes
/// `input` through.
///
/// - `d/d src`: gather `grad_output` at the view geometry (matches torch's
///   `grad.as_strided(size, stride, storage_offset)` derivative and the
///   numerical Jacobian).
/// - `d/d input`: `grad_output` with the view region ZEROED — the
///   finite-difference Jacobian. **Deliberate divergence (#1959):** torch
///   2.11.0's `as_strided_scatter_backward`
///   (`torch/csrc/autograd/FunctionsManual.cpp:3366-3389` at baseline
///   `2ec0222669`) returns the OPPOSITE masking (zeros except the view
///   region) and fails `torch.autograd.gradcheck` against its own
///   numerical Jacobian.
///
/// Like torch, both formulas assume the view geometry does not overlap
/// itself.
#[derive(Debug)]
pub struct AsStridedScatterBackward<T: Float> {
    input: Tensor<T>,
    src: Tensor<T>,
    size: Vec<usize>,
    stride: Vec<isize>,
    storage_offset: usize,
}

impl<T: Float> AsStridedScatterBackward<T> {
    pub fn new(
        input: Tensor<T>,
        src: Tensor<T>,
        size: Vec<usize>,
        stride: Vec<isize>,
        storage_offset: usize,
    ) -> Self {
        Self {
            input,
            src,
            size,
            stride,
            storage_offset,
        }
    }
}

impl<T: Float> GradFn<T> for AsStridedScatterBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // The (size, stride, offset) triple is interpreted against the
        // logical C-order buffer of the forward output (the forward
        // validated it against `self.numel()`), so rebase the upstream
        // grad to exactly that layout before gathering/masking.
        let grad = exact_contiguous(grad_output)?;

        let grad_input = if self.input.requires_grad() {
            // Pass-through grad with the scattered region zeroed: reuse the
            // forward scatter with a zeros src (device-preserving on CUDA;
            // `grad` never requires grad here, so no graph is attached).
            let zeros_src = crate::creation::zeros::<T>(&self.size)?;
            let zeros_src = if grad.is_cuda() {
                zeros_src.to(grad.device())?
            } else {
                zeros_src
            };
            Some(grad.as_strided_scatter(
                &zeros_src,
                &self.size,
                &self.stride,
                Some(self.storage_offset),
            )?)
        } else {
            None
        };

        let grad_src = if self.src.requires_grad() {
            Some(grad.as_strided_copy(&self.size, &self.stride, Some(self.storage_offset))?)
        } else {
            None
        };

        Ok(vec![grad_input, grad_src])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input, &self.src]
    }

    fn name(&self) -> &'static str {
        "AsStridedScatterBackward"
    }
}

// ---------------------------------------------------------------------------
// CUDA strided copy dispatch
// ---------------------------------------------------------------------------

/// Materialise an `as_strided` view living on CUDA into a contiguous CUDA
/// storage. Dispatches to the existing `strided_copy_{f32,f64}` GPU kernels
/// via the `GpuBackend` dispatcher.
///
/// Never bounces data through host memory. Returns the bare storage so
/// callers decide whether to attach autograd metadata (CORE-060 / #1754).
fn materialize_strided_cuda<T: Float>(view: &Tensor<T>) -> FerrotorchResult<TensorStorage<T>> {
    use std::any::TypeId;

    let backend = crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
    let storage = view.storage();
    let buf = storage
        .gpu_handle()
        .ok_or(FerrotorchError::DeviceUnavailable)?;
    let out_shape = view.shape().to_vec();
    let stride = view.strides().to_vec();
    let offset = view.storage_offset();

    let new_handle = if TypeId::of::<T>() == TypeId::of::<f32>() {
        backend.strided_copy_f32(buf, &out_shape, &stride, offset)?
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        backend.strided_copy_f64(buf, &out_shape, &stride, offset)?
    } else {
        return Err(FerrotorchError::NotImplementedOnCuda {
            op: "as_strided_copy",
        });
    };
    Ok(TensorStorage::gpu(new_handle))
}

/// CUDA path for `as_strided_scatter`. Mirrors the CPU implementation
/// shape-for-shape:
///
/// 1. Materialise `self` into a fresh contiguous GPU buffer of length
///    `numel(self)` using `strided_copy_*` (no host bounce).
/// 2. Run `strided_scatter_*` to overwrite the strided positions with
///    values from `src`.
/// 3. Return the resulting contiguous storage (the caller wraps it with
///    `self.shape()` and the autograd node when needed, CORE-060 / #1754).
///
/// f32 and f64 are supported. Other dtypes (`bf16`) on CUDA fall back
/// with `NotImplementedOnCuda`. There is no `.to(Cpu)` shortcut anywhere
/// — the data stays on device end-to-end (per `/rust-gpu-discipline`).
fn scatter_on_cuda<T: Float>(
    base: &Tensor<T>,
    src: &Tensor<T>,
    size: &[usize],
    stride: &[isize],
    offset: usize,
) -> FerrotorchResult<TensorStorage<T>> {
    use std::any::TypeId;

    let backend = crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
    let base_buf = base
        .storage()
        .gpu_handle()
        .ok_or(FerrotorchError::DeviceUnavailable)?;
    let src_buf = src
        .storage()
        .gpu_handle()
        .ok_or(FerrotorchError::DeviceUnavailable)?;

    let out_shape = base.shape().to_vec();
    let base_strides = base.strides().to_vec();
    let base_offset = base.storage_offset();

    // Step 1: clone `base` into a fresh contiguous GPU buffer. This
    // mirrors the CPU path's `let mut buf = self.data_vec()?;` line,
    // and as a side effect the resulting tensor is contiguous regardless
    // of `base`'s stride pattern — same shape result as the CPU path.
    let mut dst_handle = if TypeId::of::<T>() == TypeId::of::<f32>() {
        backend.strided_copy_f32(base_buf, &out_shape, &base_strides, base_offset)?
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        backend.strided_copy_f64(base_buf, &out_shape, &base_strides, base_offset)?
    } else {
        return Err(FerrotorchError::NotImplementedOnCuda {
            op: "as_strided_scatter",
        });
    };

    // Step 2: scatter src into dst at (size, stride, offset).
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        backend.strided_scatter_f32(src_buf, &mut dst_handle, size, stride, offset)?;
    } else {
        backend.strided_scatter_f64(src_buf, &mut dst_handle, size, stride, offset)?;
    }

    Ok(TensorStorage::gpu(dst_handle))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::creation::{tensor, zeros};
    use crate::storage::TensorStorage;

    fn t(data: &[f64], shape: &[usize]) -> Tensor<f64> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
    }

    // -----------------------------------------------------------------------
    // as_strided: zero-copy view tests
    // -----------------------------------------------------------------------

    #[test]
    fn as_strided_reshape_to_2x3() {
        // 1-D length-6 → 2x3 contiguous.
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]);
        let v = a.as_strided(&[2, 3], &[3, 1], None).unwrap();
        assert_eq!(v.shape(), &[2, 3]);
        assert_eq!(v.data_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn as_strided_overlapping_sliding_window() {
        // Sliding window of length 3 over [1..6]: shape [3, 3], stride [1, 1].
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5]);
        let v = a.as_strided(&[3, 3], &[1, 1], None).unwrap();
        assert_eq!(v.shape(), &[3, 3]);
        // Each row is a 3-window:
        assert_eq!(
            v.data_vec().unwrap(),
            vec![1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0]
        );
    }

    #[test]
    fn as_strided_negative_stride_reverses() {
        // Reverse a 1-D tensor: start at the end, stride -1.
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[4]);
        // storage_offset = 3 (last element), stride = -1, size = 4.
        let v = a.as_strided(&[4], &[-1], Some(3)).unwrap();
        assert_eq!(v.data_vec().unwrap(), vec![4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn as_strided_zero_stride_broadcast() {
        // Stride 0: every position reads the same element (broadcast).
        let a = t(&[7.0, 8.0, 9.0], &[3]);
        let v = a.as_strided(&[5], &[0], Some(1)).unwrap();
        assert_eq!(v.data_vec().unwrap(), vec![8.0, 8.0, 8.0, 8.0, 8.0]);
    }

    // -----------------------------------------------------------------------
    // as_strided: bounds validation
    // -----------------------------------------------------------------------

    #[test]
    fn as_strided_rejects_out_of_bounds() {
        let a = t(&[1.0, 2.0, 3.0], &[3]);
        // shape [4], stride [1] would reach offset 3 — out of bounds.
        let err = a.as_strided(&[4], &[1], Some(0)).unwrap_err();
        assert!(
            matches!(err, FerrotorchError::InvalidArgument { .. }),
            "expected InvalidArgument, got {err:?}"
        );
    }

    #[test]
    fn as_strided_rejects_negative_reach() {
        let a = t(&[1.0, 2.0, 3.0], &[3]);
        // stride -1 from offset 1 reaches -1 on the second step.
        let err = a.as_strided(&[3], &[-1], Some(1)).unwrap_err();
        assert!(matches!(err, FerrotorchError::InvalidArgument { .. }));
    }

    #[test]
    fn as_strided_rejects_size_stride_length_mismatch() {
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let err = a.as_strided(&[2, 2], &[1], None).unwrap_err();
        assert!(matches!(err, FerrotorchError::InvalidArgument { .. }));
    }

    #[test]
    fn as_strided_zero_size_dim_is_valid() {
        // Empty view: shape [0, 5] with any strides is in-bounds.
        let a = t(&[1.0, 2.0, 3.0], &[3]);
        let v = a.as_strided(&[0, 5], &[100, 100], Some(0)).unwrap();
        assert_eq!(v.shape(), &[0, 5]);
        assert_eq!(v.data_vec().unwrap(), Vec::<f64>::new());
    }

    // -----------------------------------------------------------------------
    // as_strided shares storage with input (zero-copy)
    // -----------------------------------------------------------------------

    #[test]
    fn as_strided_shares_storage() {
        // Verify the view points at the same Arc<Storage> by checking that
        // building the view succeeds with a small storage offset and the
        // storage length stays the same.
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]);
        let v = a.as_strided(&[3], &[2], Some(0)).unwrap();
        // [1, 3, 5]
        assert_eq!(v.data_vec().unwrap(), vec![1.0, 3.0, 5.0]);
        // The underlying storage length matches `a`'s storage length.
        assert_eq!(v.storage().len(), a.storage().len());
    }

    // -----------------------------------------------------------------------
    // as_strided_copy: materialised, contiguous output
    // -----------------------------------------------------------------------

    #[test]
    fn as_strided_copy_makes_contiguous_2x3() {
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]);
        let copy = a.as_strided_copy(&[2, 3], &[3, 1], None).unwrap();
        assert_eq!(copy.shape(), &[2, 3]);
        assert!(copy.is_contiguous());
        assert_eq!(copy.data_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn as_strided_copy_collects_overlapping_window() {
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5]);
        let copy = a.as_strided_copy(&[3, 3], &[1, 1], None).unwrap();
        assert!(copy.is_contiguous());
        assert_eq!(
            copy.data_vec().unwrap(),
            vec![1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0]
        );
    }

    // -----------------------------------------------------------------------
    // as_strided_scatter: write at strided positions
    // -----------------------------------------------------------------------

    #[test]
    fn as_strided_scatter_writes_into_view_positions() {
        let dst = t(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[6]);
        let src = t(&[10.0, 20.0, 30.0], &[3]);
        // Write src into positions 0, 2, 4 of dst.
        let out = dst.as_strided_scatter(&src, &[3], &[2], Some(0)).unwrap();
        assert_eq!(
            out.data_vec().unwrap(),
            vec![10.0, 0.0, 20.0, 0.0, 30.0, 0.0]
        );
    }

    #[test]
    fn as_strided_scatter_preserves_non_view_positions() {
        let dst = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]);
        let src = t(&[100.0, 200.0], &[2]);
        // Write src into positions 1, 3.
        let out = dst.as_strided_scatter(&src, &[2], &[2], Some(1)).unwrap();
        assert_eq!(
            out.data_vec().unwrap(),
            vec![1.0, 100.0, 3.0, 200.0, 5.0, 6.0]
        );
    }

    #[test]
    fn as_strided_scatter_2d_view_into_1d_dst() {
        // dst is length 6; scatter a 2x3 source via [3, 1] strides starting at 0.
        let dst = zeros::<f64>(&[6]).unwrap();
        let src = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let out = dst
            .as_strided_scatter(&src, &[2, 3], &[3, 1], Some(0))
            .unwrap();
        assert_eq!(out.data_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn as_strided_scatter_rejects_shape_mismatch() {
        let dst = zeros::<f64>(&[5]).unwrap();
        let src = t(&[1.0, 2.0, 3.0], &[3]);
        // Requested view is [2] but src is [3] → mismatch.
        let err = dst
            .as_strided_scatter(&src, &[2], &[1], Some(0))
            .unwrap_err();
        assert!(matches!(err, FerrotorchError::ShapeMismatch { .. }));
    }

    // -----------------------------------------------------------------------
    // Autograd: as_strided then sum should yield correct gradients.
    // -----------------------------------------------------------------------

    #[test]
    fn as_strided_backward_scatters_into_input_shape() {
        use crate::autograd::backward;

        // input: [a, b, c, d, e, f] with requires_grad
        let input = tensor(&[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let input = input.requires_grad_(true);
        // View as 2x3 (contiguous reshape via as_strided).
        let view = input.as_strided(&[2, 3], &[3, 1], None).unwrap();
        // Sum to scalar so backward returns ones at every view position.
        let s = view.sum_all().unwrap();
        backward(&s).unwrap();
        let g = input.grad().unwrap().expect("input should have a gradient");
        // Each input element appears exactly once in the view, so the
        // gradient should be all ones.
        assert_eq!(g.data_vec().unwrap(), vec![1.0; 6]);
    }

    #[test]
    fn as_strided_backward_overlapping_view_sums_gradients() {
        use crate::autograd::backward;

        // Sliding window: each input element appears in multiple view
        // positions, so autograd must SUM the upstream gradient over every
        // aliasing position (CORE-058 / #1752 — the previous expectation
        // pinned the buggy last-write-wins scatter as `[1.0; 5]` and
        // claimed torch parity).
        //
        // torch oracle (live, 2.11.0+cu130):
        // ```python
        // x = torch.arange(1., 6., dtype=torch.float64, requires_grad=True)
        // x.as_strided([3,3],[1,1],0).sum().backward()
        // x.grad  # tensor([1., 2., 3., 2., 1.], dtype=torch.float64)
        // ```
        //
        // The view is non-contiguous, and `sum_all` requires contiguous
        // input today, so materialise via `.contiguous()` first; this
        // chains `AsStridedBackward` <- `ContiguousBackward` <- sum, which
        // exercises the as_strided VJP under composition.
        let input = tensor(&[1.0_f64, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let input = input.requires_grad_(true);
        let view = input.as_strided(&[3, 3], &[1, 1], None).unwrap();
        let contig = view.contiguous().unwrap();
        let s = contig.sum_all().unwrap();
        backward(&s).unwrap();
        let g = input.grad().unwrap().expect("input should have a gradient");
        assert_eq!(g.data_vec().unwrap(), vec![1.0, 2.0, 3.0, 2.0, 1.0]);
    }
}
