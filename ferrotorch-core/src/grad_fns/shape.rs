//! GradFn backward implementations for shape-manipulation operations.
//!
//! Shape ops (reshape, squeeze, unsqueeze, flatten, transpose, expand, cat)
//! are essentially bookkeeping — the data moves around but is never scaled.
//! Their VJPs either reinterpret the gradient buffer under the original
//! shape, transpose it, or split/sum along axes.
//!
//! ## REQ status (per `.design/ferrotorch-core/grad_fns/shape.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`view`) | SHIPPED | `Tensor::view_t` in `methods.rs` delegates to `grad_fns::shape::reshape`; parity-sweep `view` reports `56/56 passed`. |
//! | REQ-2 (`reshape`) | SHIPPED | `reshape` + `ReshapeBackward` consumed by `Tensor::reshape_t`, `flex_attention.rs`, `einsum.rs`; parity `56/56 passed`. |
//! | REQ-3 (`flatten`) | SHIPPED | `flatten` + `FlattenBackward` consumed by `Tensor::flatten_t` and `Tensor::flatten` (method body in `tensor.rs`); parity `48/48 passed`. |
//! | REQ-4 (`unflatten`) | SHIPPED | `unflatten` here (reshape a single dim into `sizes`, inheriting `ReshapeBackward`) consumed by `Tensor::unflatten_t` in `methods.rs`; lib tests `test_unflatten_*`. Closes #1342 REQ-4. |
//! | REQ-5 (`squeeze`) | SHIPPED | `squeeze` + `SqueezeBackward` consumed by `Tensor::squeeze_t` and `einsum.rs`; runner-arm gap tracked under #1340. |
//! | REQ-6 (`unsqueeze`) | SHIPPED | `unsqueeze` + `UnsqueezeBackward` consumed by `Tensor::unsqueeze_t`, `einsum.rs`, and `grad_fns::indexing` broadcast prep; runner-arm gap #1340. |
//! | REQ-7 (`permute`) | SHIPPED | `permute_t` + `PermuteBackward` live in `methods.rs`; `TransposeBackward` and `transpose_2d` here delegate through it; consumers include `Tensor::permute` + pervasive einsum/vmap callers; runner-arm gap #1340. |
//! | REQ-8 (`transpose` / 2-D `t`) | SHIPPED | `transpose_2d` + `TransposeBackward` here, plus `Tensor::transpose` in `methods.rs` building a swap perm; consumer is `Tensor::t`; runner-arm gap #1340. |
//! | REQ-9 (`swapaxes`) | SHIPPED | `swapaxes` here (literal transpose alias) consumed by `Tensor::swapaxes` in `methods.rs`; lib tests `test_swapaxes_equals_transpose`, `test_swapaxes_backward_reaches_leaf`. Closes #1342 REQ-9. |
//! | REQ-10 (`swapdims`) | SHIPPED | `swapdims` here (literal transpose alias) consumed by `Tensor::swapdims` in `methods.rs`; lib test `test_swapdims_equals_transpose`. Closes #1342 REQ-10. |
//! | REQ-11 (`expand`) | SHIPPED | `expand` + `ExpandBackward` consume the shared `arithmetic::reduce_grad_to_shape`; forward is a PyTorch-parity zero-stride metadata view on CPU and CUDA; consumed by `grad_fns::indexing` broadcast prep and `einsum.rs`; runner-arm gap #1340. |
//! | REQ-12 (`expand_as`) | SHIPPED | `expand_as` here (delegates to `expand` with `other.shape()`, inheriting `ExpandBackward`) consumed by `Tensor::expand_as_t` in `methods.rs`; lib tests `test_expand_as_equals_expand`, `test_expand_as_backward_sums_broadcast_axes`. Closes #1342 REQ-12. |
//! | REQ-13 (`repeat`) | SHIPPED | `repeat` here (cat-composition tile) consumed by `Tensor::repeat_t`; lib tests `test_repeat_*`. Closes #1342 REQ-13. |
//! | REQ-14 (`repeat_interleave`) | SHIPPED | `repeat_interleave` + `RepeatInterleaveBackward` here consumed by `Tensor::repeat_interleave_t`; lib tests `test_repeat_interleave_*`. Closes #1342 REQ-14. |
//! | REQ-15 (`cat`) | SHIPPED | `cat` + `CatBackward` with byte-width-dispatched `strided_cat` GPU fast path; consumers in `flex_attention.rs` and `lib.rs` re-export; runner-arm gap #1340. |
//! | REQ-16 (`stack`) | SHIPPED | `vmap::stack` is the pub-API surface (grandfathered per S5); autograd inherited from `unsqueeze + cat`; runner-arm gap #1340. |
//! | REQ-17 (`vstack`) | SHIPPED | `vstack` here (`atleast_2d` + `cat(_,0)`) consumed by `Tensor::vstack_t`; lib tests `test_vstack_*`. Closes #1342 REQ-17. |
//! | REQ-18 (`hstack`) | SHIPPED | `hstack` here (`atleast_1d` + rank-dispatched `cat`) consumed by `Tensor::hstack_t`; lib tests `test_hstack_*`. Closes #1342 REQ-18. |
//! | REQ-19 (`dstack`) | SHIPPED | `dstack` here (`atleast_3d` + `cat(_,2)`) consumed by `Tensor::dstack_t`; lib test `test_dstack_1d_inputs`. Closes #1342 REQ-19. |
//! | REQ-20 (`column_stack`) | SHIPPED | `column_stack` here (reshape ≤1-D → `(numel,1)` + `hstack`) consumed by `Tensor::column_stack_t`; lib test `test_column_stack_1d_inputs`. Closes #1342 REQ-20. |
//! | REQ-21 (`split`) | SHIPPED | `SplitBackward` here is consumed by `methods::split_t` per the explicit `use crate::grad_fns::shape::SplitBackward`; runner-arm gap #1340. |
//! | REQ-22 (`chunk`) | SHIPPED | `methods::chunk_t` shares the `SplitBackward` machinery from this file; runner-arm gap #1340. |
//! | REQ-23 (`tensor_split`) | SHIPPED | `tensor_split` here (`narrow` per section, boundaries clamped) consumed by `Tensor::tensor_split_t`; lib tests `test_tensor_split_*`. Closes #1342 REQ-23. |
//! | REQ-24 (`narrow`) | SHIPPED | `narrow_t` + `NarrowBackward` live in `methods.rs`; consumer is `Tensor::narrow`; runner-arm gap #1340. |
//! | REQ-25 (`unbind`) | SHIPPED | `unbind` here (`narrow` + `squeeze` per index) consumed by `Tensor::unbind_t`; lib tests `test_unbind_*`. Closes #1342 REQ-25. |
//! | REQ-26 (`broadcast_tensors`) | SHIPPED | `broadcast_tensors` here (`broadcast_shapes` fold + per-input `expand`) consumed by `lib.rs` crate-root re-export; lib test `test_broadcast_tensors_common_shape`. Closes #1342 REQ-26. |
//! | REQ-27 (`broadcast_to`) | SHIPPED | `broadcast_to` here (literal `expand` alias) consumed by `Tensor::broadcast_to_t`; lib test `test_broadcast_to_equals_expand`. Closes #1342 REQ-27. |
//! | REQ-28 (`broadcast_shapes`) | SHIPPED | `broadcast_shapes` lives in `crate::shape` (sister utility module); consumed across `meta_propagate.rs`, `ops/elementwise.rs`, `grad_fns/indexing.rs`, `grad_fns/arithmetic.rs`; runner-arm gap #1340. |
//! | REQ-29 (`movedim`) | SHIPPED | `movedim` here (computed full perm → `permute_t`) consumed by `Tensor::movedim_t`; lib tests `test_movedim_*`. Closes #1342 REQ-29. |
//! | REQ-30 (`moveaxis`) | SHIPPED | `moveaxis` here (literal `movedim` alias) consumed by `Tensor::moveaxis_t`; lib test `test_moveaxis_equals_movedim`. Closes #1342 REQ-30. |
//! | REQ-31 (`tile`) | SHIPPED | `tile` here (left-pad reps → `repeat`) consumed by `Tensor::tile_t`; lib test `test_tile_pads_reps`. Closes #1342 REQ-31. |
//! | REQ-32 (`roll`) | SHIPPED | `RollBackward` here is consumed by `ops::tensor_ops::roll` (CUDA + CPU forward arms both attach the backward fn); upstream is `TensorTransformations.cpp:110` (route's upstream list is incomplete for this op); runner-arm gap #1340. |
//! | REQ-33 (`rot90`) | SHIPPED | `rot90` here (`k mod 4` switch over `flip`+`transpose`) consumed by `Tensor::rot90_t`; lib tests `test_rot90_*`. Closes #1342 REQ-33. |
//! | REQ-34 (`flip`) | SHIPPED | `flip` + `FlipBackward` + `flip_cpu_inner` here (CPU index-reversal; flip is its own inverse) consumed by `Tensor::flip_t`; lib tests `test_flip_*`. Closes #1342 REQ-34. |
//! | REQ-35 (`fliplr`) | SHIPPED | `fliplr` here (≥2-D check + `flip({1})`) consumed by `Tensor::fliplr_t`; lib test `test_fliplr_equals_flip_dim1`. Closes #1342 REQ-35. |
//! | REQ-36 (`flipud`) | SHIPPED | `flipud` here (≥1-D check + `flip({0})`) consumed by `Tensor::flipud_t`; lib test `test_flipud_equals_flip_dim0`. Closes #1342 REQ-36. |

use std::any::TypeId;
use std::sync::Arc;

use crate::autograd::no_grad::is_grad_enabled;
use crate::device::Device;
use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::storage::TensorStorage;
use crate::tensor::{GradFn, Tensor};

/// Returns `true` if `T` is `f32`.
#[inline]
fn is_f32<T: Float>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f32>()
}

// ---------------------------------------------------------------------------
// GPU-aware helper
// ---------------------------------------------------------------------------

/// Verify a tensor is on CPU, returning it with its device tag.
///
/// Shape ops don't have native GPU kernels yet. Instead of silently
/// downloading from GPU (which hides a costly roundtrip), we error
/// immediately so the caller can move data explicitly.
#[inline]
fn ensure_cpu<T: Float>(input: &Tensor<T>) -> FerrotorchResult<(Tensor<T>, Device)> {
    let device = input.device();
    if input.is_cuda() {
        return Err(crate::error::FerrotorchError::NotImplementedOnCuda {
            op: "shape backward",
        });
    }
    Ok((input.clone(), device))
}

/// Move a tensor to the given device if it isn't already there.
#[inline]
fn restore_device<T: Float>(tensor: Tensor<T>, device: Device) -> FerrotorchResult<Tensor<T>> {
    if device.is_cuda() {
        tensor.to(device)
    } else {
        Ok(tensor)
    }
}

// ---------------------------------------------------------------------------
// ReshapeBackward
// ---------------------------------------------------------------------------

/// Backward for `reshape(x, new_shape)`.
///
/// VJP: `grad_input = reshape(grad_output, original_shape)`.
/// The data is identical — we just reinterpret the flat buffer.
#[derive(Debug)]
pub struct ReshapeBackward<T: Float> {
    input: Tensor<T>,
    /// The shape of `input` before the reshape.
    input_shape: Vec<usize>,
}

impl<T: Float> ReshapeBackward<T> {
    pub fn new(input: Tensor<T>, input_shape: Vec<usize>) -> Self {
        Self { input, input_shape }
    }
}

impl<T: Float> GradFn<T> for ReshapeBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }
        // Reshape is a pure metadata change — zero-copy view on any device.
        let grad_input = grad_output.view_reshape(self.input_shape.clone())?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "ReshapeBackward"
    }
}

/// Reshape a tensor to `new_shape`, preserving the computation graph.
///
/// The product of `new_shape` must equal `input.numel()`. Exactly one
/// dimension may be `-1`, in which case it is inferred.
pub fn reshape<T: Float>(input: &Tensor<T>, new_shape: &[isize]) -> FerrotorchResult<Tensor<T>> {
    let numel = input.numel();
    let resolved = resolve_shape(new_shape, numel)?;

    // No-grad fast path: zero-copy view reshape (works on any device).
    if !is_grad_enabled() || !input.requires_grad() {
        return input.view_reshape(resolved);
    }

    // Grad path: zero-copy view with grad_fn attached (works on any device).
    let grad_fn = Arc::new(ReshapeBackward::new(input.clone(), input.shape().to_vec()));
    input.view_operation(resolved, grad_fn)
}

// ---------------------------------------------------------------------------
// FlattenBackward
// ---------------------------------------------------------------------------

/// Backward for `flatten(x)`.
///
/// VJP: `grad_input = reshape(grad_output, original_shape)`.
#[derive(Debug)]
pub struct FlattenBackward<T: Float> {
    input: Tensor<T>,
    input_shape: Vec<usize>,
}

impl<T: Float> FlattenBackward<T> {
    pub fn new(input: Tensor<T>, input_shape: Vec<usize>) -> Self {
        Self { input, input_shape }
    }
}

impl<T: Float> GradFn<T> for FlattenBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }
        // Unflatten is a pure metadata change — zero-copy view on any device.
        let grad_input = grad_output.view_reshape(self.input_shape.clone())?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "FlattenBackward"
    }
}

/// Flatten a tensor to 1-D.
pub fn flatten<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let numel = input.numel();

    if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(FlattenBackward::new(input.clone(), input.shape().to_vec()));
        input.view_operation(vec![numel], grad_fn)
    } else {
        input.view_reshape(vec![numel])
    }
}

// ---------------------------------------------------------------------------
// SqueezeBackward
// ---------------------------------------------------------------------------

/// Backward for `squeeze(x, axis)`.
///
/// VJP: `grad_input = unsqueeze(grad_output, axis)` — insert the removed
/// size-1 dimension back.
#[derive(Debug)]
pub struct SqueezeBackward<T: Float> {
    input: Tensor<T>,
    /// The axis that was squeezed (after normalization).
    axis: usize,
}

impl<T: Float> SqueezeBackward<T> {
    pub fn new(input: Tensor<T>, axis: usize) -> Self {
        Self { input, axis }
    }
}

impl<T: Float> GradFn<T> for SqueezeBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }
        // Re-insert the size-1 dimension at `self.axis` — pure metadata change.
        let mut new_shape = grad_output.shape().to_vec();
        new_shape.insert(self.axis, 1);
        let grad_input = grad_output.view_reshape(new_shape)?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "SqueezeBackward"
    }
}

/// Remove a size-1 dimension at `axis`.
pub fn squeeze<T: Float>(input: &Tensor<T>, axis: isize) -> FerrotorchResult<Tensor<T>> {
    let norm_axis = crate::shape::normalize_axis(axis, input.ndim())?;

    if input.shape()[norm_axis] != 1 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "squeeze: dimension {} has size {}, expected 1",
                norm_axis,
                input.shape()[norm_axis]
            ),
        });
    }

    let mut new_shape = input.shape().to_vec();
    new_shape.remove(norm_axis);

    // No-grad fast path: zero-copy view reshape.
    if !is_grad_enabled() || !input.requires_grad() {
        return input.view_reshape(new_shape);
    }

    let grad_fn = Arc::new(SqueezeBackward::new(input.clone(), norm_axis));
    input.view_operation(new_shape, grad_fn)
}

// ---------------------------------------------------------------------------
// UnsqueezeBackward
// ---------------------------------------------------------------------------

/// Backward for `unsqueeze(x, axis)`.
///
/// VJP: `grad_input = squeeze(grad_output, axis)` — remove the inserted
/// size-1 dimension.
#[derive(Debug)]
pub struct UnsqueezeBackward<T: Float> {
    input: Tensor<T>,
    /// The axis that was unsqueezed.
    axis: usize,
}

impl<T: Float> UnsqueezeBackward<T> {
    pub fn new(input: Tensor<T>, axis: usize) -> Self {
        Self { input, axis }
    }
}

impl<T: Float> GradFn<T> for UnsqueezeBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }
        // Remove the size-1 dimension at `self.axis` — pure metadata change.
        let mut new_shape = grad_output.shape().to_vec();
        new_shape.remove(self.axis);
        let grad_input = grad_output.view_reshape(new_shape)?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "UnsqueezeBackward"
    }
}

/// Insert a size-1 dimension at `axis`.
///
/// `axis` may be in the range `[-(ndim+1), ndim]` (inclusive on both ends),
/// following PyTorch semantics where a new dimension is inserted *before*
/// the given position.
pub fn unsqueeze<T: Float>(input: &Tensor<T>, axis: isize) -> FerrotorchResult<Tensor<T>> {
    // For unsqueeze, the valid range is [-(ndim+1), ndim].
    let ndim = input.ndim();
    let new_ndim = ndim + 1;
    let ndim_i = new_ndim as isize;

    if axis >= ndim_i || axis < -ndim_i {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "unsqueeze: axis {axis} is out of bounds for tensor with {ndim} dimensions (new ndim = {new_ndim})"
            ),
        });
    }

    let norm_axis = if axis < 0 {
        (ndim_i + axis) as usize
    } else {
        axis as usize
    };

    let mut new_shape = input.shape().to_vec();
    new_shape.insert(norm_axis, 1);

    // No-grad fast path: zero-copy view reshape.
    if !is_grad_enabled() || !input.requires_grad() {
        return input.view_reshape(new_shape);
    }

    let grad_fn = Arc::new(UnsqueezeBackward::new(input.clone(), norm_axis));
    input.view_operation(new_shape, grad_fn)
}

// ---------------------------------------------------------------------------
// TransposeBackward
// ---------------------------------------------------------------------------

/// Backward for `transpose_2d(x)` (2-D transpose).
///
/// VJP: `grad_input = transpose(grad_output)`.
#[derive(Debug)]
pub struct TransposeBackward<T: Float> {
    input: Tensor<T>,
}

impl<T: Float> TransposeBackward<T> {
    pub fn new(input: Tensor<T>) -> Self {
        Self { input }
    }
}

impl<T: Float> GradFn<T> for TransposeBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }
        // Zero-copy stride swap: transpose is its own inverse.
        let grad_input = crate::methods::permute_t(grad_output, &[1, 0])?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "TransposeBackward"
    }
}

/// Transpose a tensor of rank ≤ 2, preserving the computation graph.
/// Backs `Tensor::t()`.
///
/// Mirrors `torch.t` (`aten/src/ATen/native/TensorShape.cpp` `t`):
/// 0-D and 1-D tensors are returned as is (alias — same storage, same
/// autograd node), rank > 2 errors with torch's contract message, and
/// rank 2 transposes (CORE-153 / #1847).
///
/// CPU path: zero-copy O(1) stride swap — shares storage with the input.
/// GPU path: runs a transpose kernel (data copy on device).
pub fn transpose_2d<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let ndim = input.ndim();
    if ndim > 2 {
        // torch: "t() expects a tensor with <= 2 dimensions, but self is 3D"
        return Err(FerrotorchError::InvalidArgument {
            message: format!("t() expects a tensor with <= 2 dimensions, but self is {ndim}D"),
        });
    }
    if ndim < 2 {
        // torch.t pass-through: `self.transpose(0, 0)` — the identity.
        return Ok(input.clone());
    }

    // Zero-copy stride swap — works on both CPU and GPU.
    crate::methods::permute_t(input, &[1, 0])
}

// ---------------------------------------------------------------------------
// ExpandBackward
// ---------------------------------------------------------------------------

/// Backward for `expand(x, new_shape)`.
///
/// VJP: sum along every axis where the input dimension was 1 (and the
/// output dimension was > 1), or where the input had fewer dimensions
/// (implicit leading 1s).
#[derive(Debug)]
pub struct ExpandBackward<T: Float> {
    input: Tensor<T>,
    input_shape: Vec<usize>,
}

impl<T: Float> ExpandBackward<T> {
    pub fn new(input: Tensor<T>, input_shape: Vec<usize>) -> Self {
        Self { input, input_shape }
    }
}

impl<T: Float> GradFn<T> for ExpandBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }

        // reduce_grad_to_shape sums over broadcast dimensions (leading dims
        // not in target + size-1 dims) and works natively on GPU via
        // sum_axis_f32/f64 — no CPU roundtrip.
        let grad_input = super::arithmetic::reduce_grad_to_shape(grad_output, &self.input_shape)?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "ExpandBackward"
    }
}

/// Broadcast (expand) a tensor to `new_shape`.
///
/// Only size-1 dimensions can be expanded. This follows PyTorch's
/// `Tensor.expand()` semantics.
pub fn expand<T: Float>(input: &Tensor<T>, new_shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
    let in_shape = input.shape();
    let out_ndim = new_shape.len();
    let in_ndim = in_shape.len();

    if out_ndim < in_ndim {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "expand: target shape {new_shape:?} has fewer dimensions than input {in_shape:?}"
            ),
        });
    }

    // Validate that non-1 dimensions match.
    for i in 0..in_ndim {
        let in_dim = in_shape[in_ndim - 1 - i];
        let out_dim = new_shape[out_ndim - 1 - i];
        if in_dim != 1 && in_dim != out_dim {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "expand: cannot expand dimension {} from {} to {}",
                    in_ndim - 1 - i,
                    in_dim,
                    out_dim
                ),
            });
        }
    }

    // PyTorch `expand` is a metadata-only view. Dimensions that are actually
    // expanded get stride 0; unchanged dimensions preserve the source stride.
    // Leading synthetic size-1 dimensions use contiguous-style strides when
    // they remain size 1, matching ATen's `inferExpandGeometry`.
    let mut out_strides = vec![0isize; out_ndim];
    let in_strides = input.strides();
    let mut next_stride = 1isize;
    for out_dim in (0..out_ndim).rev() {
        let in_pos = out_dim as isize - (out_ndim as isize - in_ndim as isize);
        let (source_size, source_stride) = if in_pos >= 0 {
            let idx = in_pos as usize;
            (in_shape[idx], in_strides[idx])
        } else {
            (1, next_stride)
        };
        let target_size = new_shape[out_dim];
        out_strides[out_dim] = if source_size == target_size {
            source_stride
        } else {
            0
        };
        next_stride = new_shape[out_dim] as isize * out_strides[out_dim];
    }

    let offset = input.storage_offset();
    if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(ExpandBackward::new(input.clone(), in_shape.to_vec()));
        input.try_stride_view_operation(new_shape.to_vec(), out_strides, offset, grad_fn)
    } else {
        input.try_stride_view(new_shape.to_vec(), out_strides, offset)
    }
}

/// `expand_as(input, other)` — broadcast `input` to the shape of `other`.
///
/// Mirrors upstream `aten/src/ATen/native/TensorShape.cpp:1374 Tensor
/// expand_as(const Tensor& self, const Tensor& other) { return
/// self.expand_symint(other.sym_sizes()); }` — a literal one-line delegation
/// to `expand` with `other`'s sizes. Autograd is inherited from `expand`'s
/// `ExpandBackward` (sum-reduces over broadcast axes back to `input`'s shape).
pub fn expand_as<T: Float>(input: &Tensor<T>, other: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    expand(input, other.shape())
}

/// `unflatten(input, dim, sizes)` — reshape a single dimension `dim` into the
/// multiple sizes `sizes`, leaving every other dimension untouched.
///
/// Mirrors upstream `aten/src/ATen/native/TensorShape.cpp:4350 Tensor
/// unflatten_symint(const Tensor& self, int64_t dim, SymIntArrayRef sizes)`,
/// which delegates to `unflatten_impl` at `:4305`: it `maybe_wrap_dim`s the
/// dim, requires `sizes` be non-empty, `infer_size_dv`-resolves a single `-1`
/// slot against `self.sym_size(dim)`, then `view_symint`s the spliced shape.
///
/// `sizes` accepts at most one `-1` inference slot; the product of the
/// remaining entries must divide the size of `dim`. Autograd is inherited from
/// `reshape`'s `ReshapeBackward` (the op is a pure metadata change).
pub fn unflatten<T: Float>(
    input: &Tensor<T>,
    dim: isize,
    sizes: &[isize],
) -> FerrotorchResult<Tensor<T>> {
    if sizes.is_empty() {
        return Err(FerrotorchError::InvalidArgument {
            message: "unflatten: sizes must be non-empty".into(),
        });
    }
    let norm_dim = crate::shape::normalize_axis(dim, input.ndim())?;
    let old_shape = input.shape();
    let dim_size = old_shape[norm_dim];

    // Splice `sizes` in place of `old_shape[norm_dim]`, then let `reshape`'s
    // `resolve_shape` validate the product and resolve any single `-1` against
    // the *full* numel. We pre-substitute `dim_size` for a `-1` slot so the
    // inference is local to `dim` (upstream resolves against `self.size(dim)`,
    // not the whole tensor) and emit a clear unflatten-specific error.
    let resolved_sizes = resolve_unflatten_sizes(sizes, dim_size)?;

    let mut new_shape: Vec<isize> = Vec::with_capacity(old_shape.len() + sizes.len() - 1);
    new_shape.extend(old_shape[..norm_dim].iter().map(|&d| d as isize));
    new_shape.extend(resolved_sizes.iter().map(|&d| d as isize));
    new_shape.extend(old_shape[norm_dim + 1..].iter().map(|&d| d as isize));

    reshape(input, &new_shape)
}

/// Resolve the `sizes` argument of `unflatten` against `dim_size`, handling a
/// single `-1` inference slot exactly as upstream's `infer_size_dv` does
/// (`aten/src/ATen/native/TensorShape.cpp:4322`).
fn resolve_unflatten_sizes(sizes: &[isize], dim_size: usize) -> FerrotorchResult<Vec<usize>> {
    let mut inferred_idx: Option<usize> = None;
    let mut product: usize = 1;
    for (i, &s) in sizes.iter().enumerate() {
        if s == -1 {
            if inferred_idx.is_some() {
                return Err(FerrotorchError::InvalidArgument {
                    message: "unflatten: only one dimension can be -1".into(),
                });
            }
            inferred_idx = Some(i);
        } else if s < 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("unflatten: invalid size {s}"),
            });
        } else {
            product *= s as usize;
        }
    }

    let mut out: Vec<usize> = sizes.iter().map(|&s| s.max(0) as usize).collect();
    if let Some(idx) = inferred_idx {
        if product == 0 || !dim_size.is_multiple_of(product) {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "unflatten: cannot infer -1 slot for dim of size {dim_size} from {sizes:?}"
                ),
            });
        }
        out[idx] = dim_size / product;
    } else if product != dim_size {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "unflatten: provided sizes {sizes:?} (product {product}) do not match dim size {dim_size}"
            ),
        });
    }
    Ok(out)
}

/// `swapaxes(input, axis0, axis1)` — swap two axes; a literal alias of
/// `transpose`.
///
/// Mirrors upstream `aten/src/ATen/native/TensorShape.cpp:4776 Tensor
/// swapaxes(const Tensor& self, int64_t axis0, int64_t axis1) { return
/// self.transpose(axis0, axis1); }`. Zero-copy stride swap; autograd inherited
/// from `Tensor::transpose`'s `PermuteBackward`.
pub fn swapaxes<T: Float>(
    input: &Tensor<T>,
    axis0: usize,
    axis1: usize,
) -> FerrotorchResult<Tensor<T>> {
    input.transpose(axis0, axis1)
}

/// `swapdims(input, dim0, dim1)` — swap two dims; a literal alias of
/// `transpose`.
///
/// Mirrors upstream `aten/src/ATen/native/TensorShape.cpp:4784 Tensor
/// swapdims(const Tensor& self, int64_t dim0, int64_t dim1) { return
/// self.transpose(dim0, dim1); }`. Identical behavior to `swapaxes` — both are
/// the NumPy / array-API spellings of `transpose`.
pub fn swapdims<T: Float>(
    input: &Tensor<T>,
    dim0: usize,
    dim1: usize,
) -> FerrotorchResult<Tensor<T>> {
    input.transpose(dim0, dim1)
}

// ---------------------------------------------------------------------------
// FlipBackward — backward for flip (REQ-34)
// ---------------------------------------------------------------------------

/// Backward for `flip(x, dims)` (reverse element order along `dims`).
///
/// VJP: `flip` is a permutation (its own inverse), so the Jacobian is the
/// corresponding permutation matrix and the VJP re-applies the SAME flip to
/// the incoming gradient: `grad_input = flip(grad_output, dims)`.
#[derive(Debug)]
pub struct FlipBackward<T: Float> {
    input: Tensor<T>,
    /// The (already normalized) dims that were reversed in the forward pass.
    dims: Vec<usize>,
}

impl<T: Float> FlipBackward<T> {
    pub fn new(input: Tensor<T>, dims: Vec<usize>) -> Self {
        Self { input, dims }
    }
}

impl<T: Float> GradFn<T> for FlipBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }
        // Flip is its own inverse: re-flip the incoming gradient along the
        // same dims to reverse the permutation. Re-use the host kernel so the
        // grad tensor is unconditionally a leaf (no nested grad_fn).
        let (cpu_go, device) = ensure_cpu(grad_output)?;
        let go_data = cpu_go.data_vec()?;
        let shape = cpu_go.shape();
        let flipped = flip_cpu_inner(&go_data, shape, &self.dims);
        let grad_tensor = Tensor::from_storage(TensorStorage::cpu(flipped), shape.to_vec(), false)?;
        Ok(vec![Some(restore_device(grad_tensor, device)?)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "FlipBackward"
    }
}

/// Reverse the order of elements along each axis in `dims`.
///
/// Mirrors upstream `aten/src/ATen/native/TensorTransformations.cpp:36 Tensor
/// flip(const Tensor& self, IntArrayRef dims)` — wraps + de-duplicates the
/// dims (a repeated dim is an error here), then materializes a copy with each
/// listed axis reversed. Backward re-applies the same flip (`FlipBackward`,
/// flip is its own inverse).
pub fn flip<T: Float>(input: &Tensor<T>, dims: &[isize]) -> FerrotorchResult<Tensor<T>> {
    let ndim = input.ndim();
    // Normalize + validate dims, rejecting duplicates (upstream
    // `dim_list_to_bitset` sets one bit per dim and errors on a repeat).
    let mut norm: Vec<usize> = Vec::with_capacity(dims.len());
    for &d in dims {
        let nd = crate::shape::normalize_axis(d, ndim)?;
        if norm.contains(&nd) {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("flip: dim {nd} appears multiple times in the list of dims"),
            });
        }
        norm.push(nd);
    }

    if input.is_cuda() {
        return Err(crate::error::FerrotorchError::NotImplementedOnCuda { op: "flip" });
    }

    // `data_vec` gathers a (possibly strided) view into logical C-order, so
    // `flip_cpu_inner`'s C-contiguous-stride assumption holds even when the
    // input is a non-materialized view (e.g. a transpose produced by `rot90`).
    let in_data = input.data_vec()?;
    let shape = input.shape();
    let out_data = flip_cpu_inner(&in_data, shape, &norm);

    if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(FlipBackward::new(input.clone(), norm));
        Tensor::from_operation(TensorStorage::cpu(out_data), shape.to_vec(), grad_fn)
    } else {
        Tensor::from_storage(TensorStorage::cpu(out_data), shape.to_vec(), false)
    }
}

/// CPU flip kernel shared by `flip` (forward) and `FlipBackward` (backward).
///
/// Produces `out[i0,…] = data[j0,…]` where the coordinate along every axis in
/// `dims` is reversed (`jk = size_k - 1 - ik`) and unchanged otherwise.
/// `data` is assumed C-contiguous in `shape`.
fn flip_cpu_inner<T: Float>(data: &[T], shape: &[usize], dims: &[usize]) -> Vec<T> {
    let numel = data.len();
    let strides = crate::shape::c_contiguous_strides(shape);
    let ndim = shape.len();
    let mut out = vec![<T as num_traits::Zero>::zero(); numel];

    for out_flat in 0..numel {
        // Decompose the output flat index into coords, mapping each flipped
        // axis to its source coordinate, then recompose into the source flat
        // index (same C-contiguous strides for input and output).
        let mut rem = out_flat;
        let mut src_flat = 0usize;
        for d in 0..ndim {
            let stride = strides[d] as usize;
            let coord = rem / stride;
            rem %= stride;
            let src_coord = if dims.contains(&d) {
                shape[d] - 1 - coord
            } else {
                coord
            };
            src_flat += src_coord * stride;
        }
        out[out_flat] = data[src_flat];
    }
    out
}

/// `fliplr(input)` — flip a (≥2-D) tensor left-to-right (along dim 1).
///
/// Mirrors upstream `aten/src/ATen/native/TensorTransformations.cpp:180 Tensor
/// fliplr(const Tensor& self) { ... return self.flip({1}); }`. Autograd is
/// inherited from `flip`'s `FlipBackward`.
pub fn fliplr<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if input.ndim() < 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("fliplr: input must be >= 2-D, got {}-D", input.ndim()),
        });
    }
    flip(input, &[1])
}

/// `flipud(input)` — flip a (≥1-D) tensor up-to-down (along dim 0).
///
/// Mirrors upstream `aten/src/ATen/native/TensorTransformations.cpp:186 Tensor
/// flipud(const Tensor& self) { ... return self.flip({0}); }`. Autograd is
/// inherited from `flip`'s `FlipBackward`.
pub fn flipud<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if input.ndim() < 1 {
        return Err(FerrotorchError::InvalidArgument {
            message: "flipud: input must be >= 1-D, got 0-D".into(),
        });
    }
    flip(input, &[0])
}

/// `rot90(input, k, dims)` — rotate a tensor 90° `k` times in the plane
/// spanned by `dims`.
///
/// Mirrors upstream `aten/src/ATen/native/TensorTransformations.cpp:134 Tensor
/// rot90(const Tensor& self, int64_t k, IntArrayRef dims)`: `k` is reduced mod
/// 4, then `k==1 → flip({dims[1]}).transpose(dims[0],dims[1])`,
/// `k==2 → flip(dims)`, `k==3 → flip({dims[0]}).transpose(dims[0],dims[1])`,
/// `k==0 → clone`. Autograd is inherited from the `flip` + `transpose`
/// composition (`FlipBackward` + `PermuteBackward`).
pub fn rot90<T: Float>(input: &Tensor<T>, k: i64, dims: &[isize]) -> FerrotorchResult<Tensor<T>> {
    let ndim = input.ndim();
    if dims.len() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "rot90: expected exactly 2 rotation dims, got {}",
                dims.len()
            ),
        });
    }
    if ndim < 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("rot90: expected total dims >= 2, got {ndim}"),
        });
    }
    let d0 = crate::shape::normalize_axis(dims[0], ndim)?;
    let d1 = crate::shape::normalize_axis(dims[1], ndim)?;
    if d0 == d1 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("rot90: rotation dims must differ, got dim0 = {d0}, dim1 = {d1}"),
        });
    }

    // Handle modulo with negative k: reduce to {0,1,2,3}.
    let kk = k.rem_euclid(4) as u8;
    match kk {
        1 => flip(input, &[d1 as isize])?.transpose(d0, d1),
        2 => flip(input, &[d0 as isize, d1 as isize]),
        3 => flip(input, &[d0 as isize])?.transpose(d0, d1),
        _ => Ok(input.clone()),
    }
}

/// `movedim(input, source, destination)` — reposition the dims listed in
/// `source` to the indices listed in `destination`.
///
/// Mirrors upstream `aten/src/ATen/native/TensorShape.cpp:4657 Tensor
/// movedim(const Tensor& self, IntArrayRef src, IntArrayRef dst)`: it
/// `maybe_wrap_dim`s + de-duplicates both lists, then assembles a full
/// permutation (the listed dims land at their targets; the remaining dims
/// fill the leftover slots in their original relative order) and `permute`s.
/// Autograd is inherited from `permute_t`'s `PermuteBackward`.
pub fn movedim<T: Float>(
    input: &Tensor<T>,
    source: &[isize],
    destination: &[isize],
) -> FerrotorchResult<Tensor<T>> {
    if source.len() != destination.len() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "movedim: source ({} dims) and destination ({} dims) must match in length",
                source.len(),
                destination.len()
            ),
        });
    }
    let ndim = input.ndim();

    let norm_src: Vec<usize> = source
        .iter()
        .map(|&d| crate::shape::normalize_axis(d, ndim))
        .collect::<FerrotorchResult<_>>()?;
    let norm_dst: Vec<usize> = destination
        .iter()
        .map(|&d| crate::shape::normalize_axis(d, ndim))
        .collect::<FerrotorchResult<_>>()?;

    let has_dup = |v: &[usize]| {
        let mut s = v.to_vec();
        s.sort_unstable();
        s.windows(2).any(|w| w[0] == w[1])
    };
    if has_dup(&norm_src) {
        return Err(FerrotorchError::InvalidArgument {
            message: "movedim: repeated dim in `source`".into(),
        });
    }
    if has_dup(&norm_dst) {
        return Err(FerrotorchError::InvalidArgument {
            message: "movedim: repeated dim in `destination`".into(),
        });
    }

    if ndim == 0 {
        return Ok(input.clone());
    }

    // `order[new_pos] = old_pos`. Mark the explicitly-placed dims, then fill
    // the remaining target slots with the leftover source dims in order.
    let sentinel = usize::MAX;
    let mut order = vec![sentinel; ndim];
    let mut src_used = vec![false; ndim];
    for i in 0..norm_src.len() {
        order[norm_dst[i]] = norm_src[i];
        src_used[norm_src[i]] = true;
    }
    let mut leftover_src = (0..ndim).filter(|d| !src_used[*d]);
    for slot in &mut order {
        if *slot == sentinel {
            *slot = leftover_src
                .next()
                .expect("movedim: leftover dim accounting");
        }
    }

    crate::methods::permute_t(input, &order)
}

/// `moveaxis(input, source, destination)` — a literal alias of `movedim`.
///
/// Mirrors upstream `aten/src/ATen/native/TensorShape.cpp:4768 Tensor
/// moveaxis(const Tensor& self, IntArrayRef src, IntArrayRef dst) { return
/// at::movedim(self, src, dst); }`. Autograd inherited via `movedim`.
pub fn moveaxis<T: Float>(
    input: &Tensor<T>,
    source: &[isize],
    destination: &[isize],
) -> FerrotorchResult<Tensor<T>> {
    movedim(input, source, destination)
}

/// `broadcast_to(input, shape)` — broadcast `input` to `shape`; a literal
/// alias of `expand`.
///
/// Mirrors upstream `aten/src/ATen/native/TensorShape.cpp:652 Tensor
/// broadcast_to_symint(const Tensor& self, SymIntArrayRef size) { return
/// self.expand_symint(size); }`. Autograd inherited from `expand`'s
/// `ExpandBackward`.
pub fn broadcast_to<T: Float>(input: &Tensor<T>, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
    expand(input, shape)
}

/// `broadcast_tensors(tensors)` — expand every input to their common
/// broadcast shape.
///
/// Mirrors upstream `aten/src/ATen/native/TensorShape.cpp:656
/// std::vector<Tensor> broadcast_tensors(TensorList tensors) { return
/// expand_outplace(tensors); }`. The common shape is the right-aligned NumPy
/// broadcast of all inputs (via `crate::shape::broadcast_shapes`); each input
/// is then `expand`ed to it, so autograd is inherited per-input from
/// `ExpandBackward`.
pub fn broadcast_tensors<T: Float>(tensors: &[Tensor<T>]) -> FerrotorchResult<Vec<Tensor<T>>> {
    if tensors.is_empty() {
        return Err(FerrotorchError::InvalidArgument {
            message: "broadcast_tensors: empty tensor list".into(),
        });
    }
    let mut common: Vec<usize> = tensors[0].shape().to_vec();
    for t in &tensors[1..] {
        common = crate::shape::broadcast_shapes(&common, t.shape())?;
    }
    tensors.iter().map(|t| expand(t, &common)).collect()
}

/// `repeat(input, repeats)` — tile `input` `repeats[i]` times along each axis.
///
/// Mirrors upstream `aten/src/ATen/native/TensorShape.cpp:1909 Tensor
/// repeat(const Tensor& self, IntArrayRef repeats)`: `repeats.size()` must be
/// `>= self.dim()`; leading new dims are prepended (treated as size-1 inputs);
/// the result size along axis `i` is `input_size[i] * repeats[i]`. We assemble
/// the tile by repeated `cat` of the (optionally leading-unsqueezed) input
/// along each axis, so autograd is inherited from `cat`'s `CatBackward`
/// (gradient of a tile is the sum of the per-copy gradients).
pub fn repeat<T: Float>(input: &Tensor<T>, repeats: &[isize]) -> FerrotorchResult<Tensor<T>> {
    let in_ndim = input.ndim();
    if repeats.len() < in_ndim {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "repeat: number of repeat dims ({}) cannot be smaller than tensor dims ({})",
                repeats.len(),
                in_ndim
            ),
        });
    }
    for &r in repeats {
        if r < 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("repeat: repeat value {r} must be non-negative"),
            });
        }
    }

    // Prepend leading size-1 dims so the input matches `repeats.len()`.
    let num_new = repeats.len() - in_ndim;
    let mut cur = if num_new > 0 {
        let mut padded: Vec<isize> = vec![1; num_new];
        padded.extend(input.shape().iter().map(|&d| d as isize));
        reshape(input, &padded)?
    } else {
        input.clone()
    };

    // Tile axis by axis: `cat` `r` copies of `cur` along axis `ax`. A 0 repeat
    // collapses that axis to size 0 (matches the upstream empty-tensor path).
    for (ax, &r) in repeats.iter().enumerate() {
        let r = r as usize;
        if r == 1 {
            continue;
        }
        if r == 0 {
            // A zero count collapses this axis to size zero (torch:
            // `input_size[i] * repeats[i]`). `reshape` cannot do this — it
            // requires the element count to stay unchanged, so reshaping the
            // non-empty `cur` to a zero-sized shape was a guaranteed
            // ShapeMismatch (CORE-054 / #1748). `narrow(ax, 0, 0)` yields a
            // genuine zero-size tensor with the computed output shape on the
            // input's device, and its `NarrowBackward` scatters the (empty)
            // upstream gradient into zeros of the input shape — exactly the
            // zero gradient torch's `RepeatBackward0` produces for a zero
            // count.
            cur = cur.narrow(ax, 0, 0)?;
            continue;
        }
        let copies = vec![cur.clone(); r];
        cur = cat(&copies, ax as isize)?;
    }
    Ok(cur)
}

/// `tile(input, reps)` — NumPy-style tile.
///
/// Mirrors upstream `aten/src/ATen/native/TensorShape.cpp:1971 Tensor
/// tile_symint(const Tensor& self, SymIntArrayRef reps)`: when `reps` is
/// shorter than `self.dim()` it is left-padded with 1s (so a 4-D tensor with
/// `reps=(2,2)` is treated as `(1,1,2,2)`), then delegates to `repeat`.
/// Autograd inherited from `repeat`'s `cat` composition.
pub fn tile<T: Float>(input: &Tensor<T>, reps: &[isize]) -> FerrotorchResult<Tensor<T>> {
    let in_ndim = input.ndim();
    if reps.len() < in_ndim {
        let pad = in_ndim - reps.len();
        let mut padded: Vec<isize> = vec![1; pad];
        padded.extend_from_slice(reps);
        repeat(input, &padded)
    } else {
        repeat(input, reps)
    }
}

/// `unbind(input, dim)` — split `input` into `size(dim)` slices, removing
/// `dim` from each.
///
/// Mirrors upstream `aten/src/ATen/native/TensorShape.cpp:4367
/// std::vector<Tensor> unbind(const Tensor& self, int64_t dim)`: returns one
/// `select(dim, i)` per index `i`. We compose `narrow(dim, i, 1)` +
/// `squeeze(dim)` (both autograd-aware) so each output slice inherits a
/// `NarrowBackward` + `SqueezeBackward` chain that scatters its gradient back
/// into the correct slice of `input`.
pub fn unbind<T: Float>(input: &Tensor<T>, dim: isize) -> FerrotorchResult<Vec<Tensor<T>>> {
    let ndim = input.ndim();
    if ndim == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "unbind: cannot unbind a 0-D tensor".into(),
        });
    }
    let norm_dim = crate::shape::normalize_axis(dim, ndim)?;
    let size = input.shape()[norm_dim];
    let mut out = Vec::with_capacity(size);
    for i in 0..size {
        let slice = crate::methods::narrow_t(input, norm_dim, i, 1)?;
        out.push(squeeze(&slice, norm_dim as isize)?);
    }
    Ok(out)
}

/// `tensor_split(input, indices, dim)` — split `input` at the given integer
/// `indices` along `dim` (the indices form section boundaries).
///
/// Mirrors upstream `aten/src/ATen/native/TensorShape.cpp:1167
/// tensor_split` (the indices form, `_tensor_split_indices` at `:1130`):
/// section `j` spans `[indices[j-1], indices[j])` along `dim` (with implicit
/// `0` and `size(dim)` endpoints), so `n` indices yield `n+1` sections and an
/// out-of-order / out-of-range index clamps to the valid range. Each section
/// is a `narrow` view, inheriting `NarrowBackward`.
pub fn tensor_split<T: Float>(
    input: &Tensor<T>,
    indices: &[usize],
    dim: isize,
) -> FerrotorchResult<Vec<Tensor<T>>> {
    let ndim = input.ndim();
    if ndim == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "tensor_split: expected at least a 1-dimensional tensor".into(),
        });
    }
    let norm_dim = crate::shape::normalize_axis(dim, ndim)?;
    let dim_size = input.shape()[norm_dim];

    let mut out = Vec::with_capacity(indices.len() + 1);
    let mut start = 0usize;
    for &idx in indices {
        // Upstream clamps each boundary to [start, dim_size] so the section is
        // never negative-length and never overruns the axis.
        let end = idx.clamp(start, dim_size);
        out.push(crate::methods::narrow_t(
            input,
            norm_dim,
            start,
            end - start,
        )?);
        start = end;
    }
    out.push(crate::methods::narrow_t(
        input,
        norm_dim,
        start,
        dim_size - start,
    )?);
    Ok(out)
}

// ---------------------------------------------------------------------------
// RepeatInterleaveBackward — backward for repeat_interleave (REQ-14)
// ---------------------------------------------------------------------------

/// Backward for `repeat_interleave(x, repeats, dim)`.
///
/// Forward duplicates each index `i` along `dim` `repeats` times
/// contiguously. The op is a (non-square) selection matrix whose VJP sums the
/// `repeats` consecutive output gradient slices that came from each input
/// index back onto that index.
#[derive(Debug)]
pub struct RepeatInterleaveBackward<T: Float> {
    input: Tensor<T>,
    /// The (already normalized) dim along which elements were repeated.
    dim: usize,
    /// The (constant) per-element repeat count.
    repeats: usize,
}

impl<T: Float> RepeatInterleaveBackward<T> {
    pub fn new(input: Tensor<T>, dim: usize, repeats: usize) -> Self {
        Self {
            input,
            dim,
            repeats,
        }
    }
}

impl<T: Float> GradFn<T> for RepeatInterleaveBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }
        let (cpu_go, device) = ensure_cpu(grad_output)?;
        let go_data = cpu_go.data()?;
        let in_shape = self.input.shape();
        let dim = self.dim;
        let dim_size = in_shape[dim];

        let outer: usize = in_shape[..dim].iter().product();
        let inner: usize = if dim + 1 < in_shape.len() {
            in_shape[dim + 1..].iter().product()
        } else {
            1
        };
        let out_dim_size = dim_size * self.repeats;

        let in_numel: usize = in_shape.iter().product();
        let mut grad = vec![<T as num_traits::Zero>::zero(); in_numel];

        // For each input index `d`, sum the `repeats` consecutive output rows
        // `[d*repeats .. (d+1)*repeats)` back onto it.
        for o in 0..outer {
            for d in 0..dim_size {
                for r in 0..self.repeats {
                    let od = d * self.repeats + r;
                    let src_base = o * out_dim_size * inner + od * inner;
                    let dst_base = o * dim_size * inner + d * inner;
                    for i in 0..inner {
                        grad[dst_base + i] += go_data[src_base + i];
                    }
                }
            }
        }

        let grad_tensor = Tensor::from_storage(TensorStorage::cpu(grad), in_shape.to_vec(), false)?;
        Ok(vec![Some(restore_device(grad_tensor, device)?)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "RepeatInterleaveBackward"
    }
}

/// `repeat_interleave(input, repeats, dim)` — repeat each element `repeats`
/// times consecutively along `dim`.
///
/// Mirrors `torch.repeat_interleave(input, repeats, dim)` for the scalar
/// `repeats` form (`aten/src/ATen/native/TensorShape.cpp` `repeat_interleave`
/// family): unlike `repeat`/`tile` (which tile whole blocks), this interleaves
/// — `[a, b]` with `repeats=2` along dim 0 becomes `[a, a, b, b]`. Backward
/// (`RepeatInterleaveBackward`) sums the `repeats` consecutive output slices
/// of each input index back onto it.
pub fn repeat_interleave<T: Float>(
    input: &Tensor<T>,
    repeats: usize,
    dim: isize,
) -> FerrotorchResult<Tensor<T>> {
    let ndim = input.ndim();
    if ndim == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "repeat_interleave: cannot repeat a 0-D tensor along a dim".into(),
        });
    }
    let norm_dim = crate::shape::normalize_axis(dim, ndim)?;

    if input.is_cuda() {
        return Err(crate::error::FerrotorchError::NotImplementedOnCuda {
            op: "repeat_interleave",
        });
    }

    // Gather logical C-order so a strided view input is handled correctly.
    let in_data = input.data_vec()?;
    let in_shape = input.shape();
    let dim_size = in_shape[norm_dim];
    let outer: usize = in_shape[..norm_dim].iter().product();
    let inner: usize = if norm_dim + 1 < ndim {
        in_shape[norm_dim + 1..].iter().product()
    } else {
        1
    };

    let out_dim_size = dim_size * repeats;
    let out_numel = outer * out_dim_size * inner;
    let mut out_data = Vec::with_capacity(out_numel);
    for o in 0..outer {
        for d in 0..dim_size {
            let src_base = o * dim_size * inner + d * inner;
            for _ in 0..repeats {
                out_data.extend_from_slice(&in_data[src_base..src_base + inner]);
            }
        }
    }

    let mut out_shape = in_shape.to_vec();
    out_shape[norm_dim] = out_dim_size;

    if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(RepeatInterleaveBackward::new(
            input.clone(),
            norm_dim,
            repeats,
        ));
        Tensor::from_operation(TensorStorage::cpu(out_data), out_shape, grad_fn)
    } else {
        Tensor::from_storage(TensorStorage::cpu(out_data), out_shape, false)
    }
}

/// `vstack(tensors)` — stack tensors row-wise (along dim 0 after promoting
/// each to ≥2-D).
///
/// Mirrors upstream `aten/src/ATen/native/TensorShape.cpp:3532 Tensor
/// vstack(TensorList tensors)`: `atleast_2d` each input then `cat(_, 0)`.
/// Autograd inherited from `reshape`/`unsqueeze` + `cat`.
pub fn vstack<T: Float>(tensors: &[Tensor<T>]) -> FerrotorchResult<Tensor<T>> {
    if tensors.is_empty() {
        return Err(FerrotorchError::InvalidArgument {
            message: "vstack: empty tensor list".into(),
        });
    }
    let promoted: Vec<Tensor<T>> = tensors
        .iter()
        .map(atleast_2d)
        .collect::<FerrotorchResult<_>>()?;
    cat(&promoted, 0)
}

/// `hstack(tensors)` — stack tensors column-wise.
///
/// Mirrors upstream `aten/src/ATen/native/TensorShape.cpp:3514 Tensor
/// hstack(TensorList tensors)`: `atleast_1d` each input; if the (promoted)
/// inputs are 1-D, `cat(_, 0)`, otherwise `cat(_, 1)`. Autograd inherited from
/// `cat`.
pub fn hstack<T: Float>(tensors: &[Tensor<T>]) -> FerrotorchResult<Tensor<T>> {
    if tensors.is_empty() {
        return Err(FerrotorchError::InvalidArgument {
            message: "hstack: empty tensor list".into(),
        });
    }
    let promoted: Vec<Tensor<T>> = tensors
        .iter()
        .map(atleast_1d)
        .collect::<FerrotorchResult<_>>()?;
    // 1-D (promoted) inputs cat along axis 0; otherwise along axis 1.
    let axis: isize = isize::from(promoted[0].ndim() != 1);
    cat(&promoted, axis)
}

/// `dstack(tensors)` — stack tensors depth-wise (along dim 2 after promoting
/// each to ≥3-D).
///
/// Mirrors upstream `aten/src/ATen/native/TensorShape.cpp:3544 Tensor
/// dstack(TensorList tensors)`: `atleast_3d` each input then `cat(_, 2)`.
/// Autograd inherited from `reshape`/`unsqueeze` + `cat`.
pub fn dstack<T: Float>(tensors: &[Tensor<T>]) -> FerrotorchResult<Tensor<T>> {
    if tensors.is_empty() {
        return Err(FerrotorchError::InvalidArgument {
            message: "dstack: empty tensor list".into(),
        });
    }
    let promoted: Vec<Tensor<T>> = tensors
        .iter()
        .map(atleast_3d)
        .collect::<FerrotorchResult<_>>()?;
    cat(&promoted, 2)
}

/// `column_stack(tensors)` — stack 1-D/0-D tensors as columns of a 2-D matrix.
///
/// Mirrors upstream `aten/src/ATen/native/TensorShape.cpp:3628 Tensor
/// column_stack(TensorList tensors)`: reshape each ≤1-D input to `(numel, 1)`,
/// leave ≥2-D inputs as-is, then `hstack`. Autograd inherited from
/// `reshape` + `cat`.
pub fn column_stack<T: Float>(tensors: &[Tensor<T>]) -> FerrotorchResult<Tensor<T>> {
    if tensors.is_empty() {
        return Err(FerrotorchError::InvalidArgument {
            message: "column_stack: empty tensor list".into(),
        });
    }
    let reshaped: Vec<Tensor<T>> = tensors
        .iter()
        .map(|t| {
            if t.ndim() <= 1 {
                reshape(t, &[t.numel() as isize, 1])
            } else {
                Ok(t.clone())
            }
        })
        .collect::<FerrotorchResult<_>>()?;
    hstack(&reshaped)
}

/// `atleast_1d(input)` — view `input` as ≥1-D, reshaping a 0-D scalar to `[1]`.
///
/// Mirrors upstream `aten/src/ATen/native/TensorTransformations.cpp:192`.
fn atleast_1d<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if input.ndim() == 0 {
        reshape(input, &[1])
    } else {
        Ok(input.clone())
    }
}

/// `atleast_2d(input)` — view `input` as ≥2-D.
///
/// Mirrors upstream `aten/src/ATen/native/TensorTransformations.cpp:211`:
/// 0-D → `[1, 1]`, 1-D → `unsqueeze(0)`, else unchanged.
fn atleast_2d<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    match input.ndim() {
        0 => reshape(input, &[1, 1]),
        1 => unsqueeze(input, 0),
        _ => Ok(input.clone()),
    }
}

/// `atleast_3d(input)` — view `input` as ≥3-D.
///
/// Mirrors upstream `aten/src/ATen/native/TensorTransformations.cpp:233`:
/// 0-D → `[1, 1, 1]`, 1-D → `unsqueeze(0).unsqueeze(-1)`, 2-D →
/// `unsqueeze(-1)`, else unchanged.
fn atleast_3d<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    match input.ndim() {
        0 => reshape(input, &[1, 1, 1]),
        1 => unsqueeze(&unsqueeze(input, 0)?, -1),
        2 => unsqueeze(input, -1),
        _ => Ok(input.clone()),
    }
}

// ---------------------------------------------------------------------------
// CatBackward
// ---------------------------------------------------------------------------

/// Backward for `cat(tensors, axis)`.
///
/// VJP: split `grad_output` along `axis` at the original sizes, yielding
/// one gradient per input tensor.
#[derive(Debug)]
pub struct CatBackward<T: Float> {
    inputs: Vec<Tensor<T>>,
    axis: usize,
    /// The size of each input along `axis`.
    split_sizes: Vec<usize>,
}

impl<T: Float> CatBackward<T> {
    pub fn new(inputs: Vec<Tensor<T>>, axis: usize, split_sizes: Vec<usize>) -> Self {
        Self {
            inputs,
            axis,
            split_sizes,
        }
    }
}

impl<T: Float> GradFn<T> for CatBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // PyTorch's cat backward slices the incoming gradient. Use the same
        // metadata-only split views here: dtype/device/layout generic, no CUDA
        // split kernel, no CPU fallback. The returned gradients are leaf-like
        // view tensors, so build them under no_grad.
        let chunks = crate::autograd::no_grad::no_grad(|| {
            crate::methods::split_t(grad_output, &self.split_sizes, self.axis)
        })?;
        Ok(self
            .inputs
            .iter()
            .zip(chunks)
            .map(|(input, chunk)| {
                if input.requires_grad() {
                    Some(chunk)
                } else {
                    None
                }
            })
            .collect())
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        self.inputs.iter().collect()
    }

    fn name(&self) -> &'static str {
        "CatBackward"
    }
}

// ---------------------------------------------------------------------------
// SplitBackward — backward for split/chunk
// ---------------------------------------------------------------------------

/// Backward for a single chunk produced by `split`.
///
/// Each chunk gets its own `SplitBackward`. The VJP zero-pads the incoming
/// gradient into the original tensor's shape at the correct offset along
/// the split dimension.
#[derive(Debug)]
pub struct SplitBackward<T: Float> {
    /// The original unsplit tensor (needed for shape and requires_grad).
    input: Tensor<T>,
    /// The split dimension.
    dim: usize,
    /// Offset of this chunk along `dim` in the original tensor.
    offset: usize,
    /// Size of this chunk along `dim`.
    chunk_size: usize,
}

impl<T: Float> SplitBackward<T> {
    pub fn new(input: Tensor<T>, dim: usize, offset: usize, chunk_size: usize) -> Self {
        Self {
            input,
            dim,
            offset,
            chunk_size,
        }
    }
}

impl<T: Float> GradFn<T> for SplitBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }

        // GPU fast path: allocate zeros on GPU, strided_cat the gradient into it.
        // Dtype-generic dispatch mirrors PyTorch's `aten::cat_out_cuda` — the
        // host computes `elem_size = size_of::<T>()` once and passes it to the
        // backend, which routes to the matching byte-width copy kernel
        // (`bf16`/`f16` = 2, `f32` = 4, `f64` = 8).
        let elem_size = std::mem::size_of::<T>();
        if grad_output.is_cuda()
            && matches!(elem_size, 2 | 4 | 8)
            && let Some(backend) = crate::gpu_dispatch::gpu_backend()
        {
            let orig_shape = self.input.shape();
            let ndim = orig_shape.len();
            let inner: usize = if self.dim + 1 < ndim {
                orig_shape[self.dim + 1..].iter().product()
            } else {
                1
            };
            let total_along_dim = orig_shape[self.dim];
            let orig_numel: usize = orig_shape.iter().product();
            let device_ord = grad_output.gpu_handle()?.device_ordinal();

            let mut zeros_handle = backend.alloc_zeros(orig_numel, T::dtype(), device_ord)?;

            let go_handle = grad_output.gpu_handle()?;
            let chunk_numel = grad_output.numel();
            backend.strided_cat(
                go_handle,
                &mut zeros_handle,
                total_along_dim,
                self.offset,
                self.chunk_size,
                inner,
                chunk_numel,
                elem_size,
            )?;

            let grad_tensor =
                Tensor::from_storage(TensorStorage::gpu(zeros_handle), orig_shape.to_vec(), false)?;
            return Ok(vec![Some(grad_tensor)]);
        }

        // CPU path (also serves as fallback for non-f32 or missing backend).
        let (cpu_go, device) = ensure_cpu(grad_output)?;
        let grad_data = cpu_go.data()?;
        let orig_shape = self.input.shape();
        let ndim = orig_shape.len();

        let outer: usize = orig_shape[..self.dim].iter().product();
        let inner: usize = if self.dim + 1 < ndim {
            orig_shape[self.dim + 1..].iter().product()
        } else {
            1
        };
        let total_along_dim = orig_shape[self.dim];

        // Build a zero tensor with the original shape, then copy the gradient
        // into the correct slice.
        let orig_numel: usize = orig_shape.iter().product();
        let mut result = vec![<T as num_traits::Zero>::zero(); orig_numel];

        for o in 0..outer {
            let dst_start = o * total_along_dim * inner + self.offset * inner;
            let src_start = o * self.chunk_size * inner;
            let row_len = self.chunk_size * inner;
            result[dst_start..dst_start + row_len]
                .copy_from_slice(&grad_data[src_start..src_start + row_len]);
        }

        let grad_tensor =
            Tensor::from_storage(TensorStorage::cpu(result), orig_shape.to_vec(), false)?;
        Ok(vec![Some(restore_device(grad_tensor, device)?)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "SplitBackward"
    }
}

/// Concatenate tensors along an axis.
///
/// All tensors must have the same shape except along `axis`.
pub fn cat<T: Float>(tensors: &[Tensor<T>], axis: isize) -> FerrotorchResult<Tensor<T>> {
    if tensors.is_empty() {
        return Err(FerrotorchError::InvalidArgument {
            message: "cat: empty tensor list".into(),
        });
    }

    let ndim = tensors[0].ndim();
    if ndim == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "cat: cannot concatenate scalar (0-D) tensors".into(),
        });
    }

    let norm_axis = crate::shape::normalize_axis(axis, ndim)?;

    // Validate shapes: all dims except `axis` must match.
    for (i, t) in tensors.iter().enumerate().skip(1) {
        if t.ndim() != ndim {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!("cat: tensor {} has {} dims, expected {}", i, t.ndim(), ndim),
            });
        }
        for d in 0..ndim {
            if d != norm_axis && t.shape()[d] != tensors[0].shape()[d] {
                return Err(FerrotorchError::ShapeMismatch {
                    message: format!(
                        "cat: tensor {} has shape {:?}, incompatible with {:?} on axis {}",
                        i,
                        t.shape(),
                        tensors[0].shape(),
                        d
                    ),
                });
            }
        }
    }

    // Build output shape.
    let mut out_shape = tensors[0].shape().to_vec();
    let split_sizes: Vec<usize> = tensors.iter().map(|t| t.shape()[norm_axis]).collect();
    let total_along_axis: usize = split_sizes.iter().sum();
    out_shape[norm_axis] = total_along_axis;

    let device = tensors[0].device();

    // Device validation up front (CORE-055 / #1749): torch requires all cat
    // inputs on one device ("Expected all tensors to be on the same
    // device"). Without this check the failure mode depended on which path
    // happened to choke on the foreign tensor (raw `gpu_handle()` error on
    // the GPU path, `data()` error on the CPU path).
    for t in tensors.iter().skip(1) {
        if t.device() != device {
            return Err(FerrotorchError::DeviceMismatch {
                expected: device,
                got: t.device(),
            });
        }
    }

    // View-geometry staging (CORE-055 / #1749 — same `gpu_handle()`-drops-
    // view-geometry class as #1657 / #1845): both consumers below require a
    // packed offset-0 buffer. The CUDA `strided_cat` kernel receives the raw
    // base handle plus logical `numel` — its trait signature carries no
    // source strides and no storage offset — so a transposed / permuted /
    // narrowed / offset view would silently concatenate the WRONG values.
    // The CPU loop reads `data()`, which rejects non-contiguous tensors.
    // Any non-conforming input is materialized ONCE via `contiguous()`
    // (device-aware: on-device `strided_copy` per the #1657 fix) under
    // `no_grad` — the staged copies are pure data sources; `CatBackward`
    // still attaches to the ORIGINAL inputs, so gradients reach the original
    // views' leaves through their own backward chains.
    let staged: Vec<Tensor<T>> = tensors
        .iter()
        .map(|t| {
            if t.is_contiguous() && t.storage_offset() == 0 {
                Ok(t.clone())
            } else {
                crate::autograd::no_grad::no_grad(|| t.contiguous())
            }
        })
        .collect::<FerrotorchResult<_>>()?;

    // GPU fast path: allocate output on GPU, then strided_cat each input — no CPU
    // download needed. Mirrors `aten::cat_out_cuda` (PyTorch): host computes
    // `elem_size` once, backend dispatches by element width into a pure-memcpy
    // kernel — no per-dtype trait method explosion.
    let elem_size = std::mem::size_of::<T>();
    if device.is_cuda()
        && matches!(elem_size, 2 | 4 | 8)
        && let Some(backend) = crate::gpu_dispatch::gpu_backend()
    {
        let inner: usize = if norm_axis + 1 < ndim {
            out_shape[norm_axis + 1..].iter().product()
        } else {
            1
        };
        let out_numel: usize = out_shape.iter().product();
        let device_ord = staged[0].gpu_handle()?.device_ordinal();

        let mut out_handle = backend.alloc_zeros(out_numel, T::dtype(), device_ord)?;

        let mut offset = 0usize;
        // Read from the STAGED (packed, offset-0) copies — `strided_cat`
        // sees no view geometry (CORE-055 / #1749).
        for t in &staged {
            let t_axis_size = t.shape()[norm_axis];
            let t_numel = t.numel();
            let t_handle = t.gpu_handle()?;
            backend.strided_cat(
                t_handle,
                &mut out_handle,
                total_along_axis,
                offset,
                t_axis_size,
                inner,
                t_numel,
                elem_size,
            )?;
            offset += t_axis_size;
        }

        let any_requires_grad = tensors.iter().any(|t| t.requires_grad());
        let storage = TensorStorage::gpu(out_handle);

        return if is_grad_enabled() && any_requires_grad {
            let grad_fn = Arc::new(CatBackward::new(tensors.to_vec(), norm_axis, split_sizes));
            Tensor::from_operation(storage, out_shape, grad_fn)
        } else {
            Tensor::from_storage(storage, out_shape, false)
        };
    }

    // CPU path — GPU tensors with an `elem_size` not in {2, 4, 8} (e.g.,
    // hypothetical future complex types) have no GPU kernel, error out
    // rather than silently spilling to host.
    if device.is_cuda() {
        return Err(crate::error::FerrotorchError::NotImplementedOnCuda { op: "cat" });
    }
    // Read from the STAGED copies — `data()` rejects non-contiguous views
    // (CORE-055 / #1749).
    let cpu_tensors: Vec<Tensor<T>> = staged;

    // Compute strides for the interleaved copy.
    let outer: usize = out_shape[..norm_axis].iter().product();
    let inner: usize = if norm_axis + 1 < ndim {
        out_shape[norm_axis + 1..].iter().product()
    } else {
        1
    };

    let out_numel: usize = out_shape.iter().product();
    let mut out_data = vec![<T as num_traits::Zero>::zero(); out_numel];

    let mut offset = 0usize;
    for t in &cpu_tensors {
        let t_data = t.data()?;
        let t_axis_size = t.shape()[norm_axis];
        for o in 0..outer {
            let src_start = o * t_axis_size * inner;
            let dst_start = o * total_along_axis * inner + offset;
            let row_len = t_axis_size * inner;
            out_data[dst_start..dst_start + row_len]
                .copy_from_slice(&t_data[src_start..src_start + row_len]);
        }
        offset += t_axis_size * inner;
    }

    let any_requires_grad = tensors.iter().any(|t| t.requires_grad());

    if is_grad_enabled() && any_requires_grad {
        let storage = if device.is_cuda() {
            let tmp = Tensor::from_storage(TensorStorage::cpu(out_data), out_shape.clone(), false)?;
            let gpu_tmp = tmp.to(device)?;
            gpu_tmp.into_storage_and_shape()?.0
        } else {
            TensorStorage::cpu(out_data)
        };
        let grad_fn = Arc::new(CatBackward::new(tensors.to_vec(), norm_axis, split_sizes));
        Tensor::from_operation(storage, out_shape, grad_fn)
    } else {
        let result = Tensor::from_storage(TensorStorage::cpu(out_data), out_shape, false)?;
        restore_device(result, device)
    }
}

// ---------------------------------------------------------------------------
// RollBackward
// ---------------------------------------------------------------------------

/// Backward for `roll(x, shifts, dim)` (cyclic shift along one axis).
///
/// Forward: `output[..., (d + shifts) mod n, ...] = input[..., d, ...]`.
///
/// VJP: cyclic shift is a permutation, so its Jacobian is the corresponding
/// permutation matrix and the VJP is the inverse permutation:
///   `grad_input = roll(grad_output, -shifts, dim)`
///
/// We replay the forward kernel with `-shifts` against `grad_output` instead
/// of calling back into `crate::ops::tensor_ops::roll` so that the resulting
/// grad tensor is unconditionally a leaf (no nested grad_fn) and so we can
/// reuse the already-validated CPU shift loop.
#[derive(Debug)]
pub struct RollBackward<T: Float> {
    /// Saved input handle (for shape and `requires_grad` propagation).
    input: Tensor<T>,
    /// The original (un-normalized) shift used in the forward pass. The
    /// backward applies `-shifts`.
    shifts: i64,
    /// The axis along which the forward roll was performed.
    dim: usize,
}

impl<T: Float> RollBackward<T> {
    pub fn new(input: Tensor<T>, shifts: i64, dim: usize) -> Self {
        Self { input, shifts, dim }
    }
}

impl<T: Float> GradFn<T> for RollBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !is_grad_enabled() {
            return Ok(vec![None]);
        }
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }

        let shape = self.input.shape();
        let dim_size = shape[self.dim] as i64;

        // Inverse shift: forward used `shifts`, backward uses `-shifts`.
        // We re-normalize against the (positive) dim_size.
        let shift_norm = if dim_size == 0 {
            0
        } else {
            (((-self.shifts) % dim_size) + dim_size) % dim_size
        };

        // GPU fast path: when the grad arrives on CUDA, reuse the same
        // `roll_f32` kernel that powers the forward pass. The op is its
        // own VJP up to negating the shift; the kernel doesn't care
        // whether the caller is forward or backward.
        if grad_output.is_cuda() {
            if is_f32::<T>()
                && let Some(backend) = crate::gpu_dispatch::gpu_backend()
            {
                if shift_norm == 0 {
                    // Inverse shift collapses to identity — clone the
                    // grad tensor's storage so the upstream input
                    // receives a leaf-grad with no grad_fn (matches
                    // the CPU branch below).
                    let grad_handle = backend.clone_buffer(grad_output.gpu_handle()?)?;
                    let grad_tensor = Tensor::from_storage(
                        TensorStorage::gpu(grad_handle),
                        shape.to_vec(),
                        false,
                    )?;
                    return Ok(vec![Some(grad_tensor)]);
                }
                let outer: usize = shape[..self.dim].iter().product();
                let inner: usize = shape[self.dim + 1..].iter().product();
                let handle = backend.roll_f32(
                    grad_output.gpu_handle()?,
                    outer,
                    shape[self.dim],
                    inner,
                    shift_norm as usize,
                )?;
                let grad_tensor =
                    Tensor::from_storage(TensorStorage::gpu(handle), shape.to_vec(), false)?;
                return Ok(vec![Some(grad_tensor)]);
            }
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "roll backward",
            });
        }

        let go_data = grad_output.data_vec()?;

        let grad = if shift_norm == 0 {
            // Inverse shift collapses to identity (e.g. shifts ≡ 0 mod n).
            go_data
        } else {
            crate::ops::tensor_ops::roll_cpu_inner(&go_data, shape, shift_norm as usize, self.dim)
        };

        let grad_tensor = Tensor::from_storage(TensorStorage::cpu(grad), shape.to_vec(), false)?;
        Ok(vec![Some(grad_tensor)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "RollBackward"
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Resolve a shape specification that may contain exactly one `-1`.
///
/// Returns the fully-resolved `Vec<usize>`.
fn resolve_shape(shape: &[isize], numel: usize) -> FerrotorchResult<Vec<usize>> {
    let mut inferred_idx: Option<usize> = None;
    let mut product: usize = 1;

    for (i, &dim) in shape.iter().enumerate() {
        if dim == -1 {
            if inferred_idx.is_some() {
                return Err(FerrotorchError::InvalidArgument {
                    message: "reshape: only one dimension can be -1".into(),
                });
            }
            inferred_idx = Some(i);
        } else if dim < 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("reshape: invalid dimension {dim}"),
            });
        } else {
            product *= dim as usize;
        }
    }

    let mut result: Vec<usize> = shape.iter().map(|&d| d as usize).collect();

    if let Some(idx) = inferred_idx {
        if product == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "reshape: cannot infer dimension with zero-size dimensions".into(),
            });
        }
        if !numel.is_multiple_of(product) {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "reshape: cannot reshape tensor of {numel} elements into shape {shape:?}"
                ),
            });
        }
        result[idx] = numel / product;
    } else if product != numel {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "reshape: cannot reshape tensor of {numel} elements into shape {shape:?}"
            ),
        });
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::backward;

    /// Helper: create a leaf tensor.
    fn leaf(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
        Tensor::from_storage(
            TensorStorage::cpu(data.to_vec()),
            shape.to_vec(),
            requires_grad,
        )
        .unwrap()
    }

    /// A trivial SumBackward for testing: broadcasts ones back to input shape.
    #[derive(Debug)]
    struct SumBackward<T: Float> {
        input: Tensor<T>,
    }

    impl<T: Float> GradFn<T> for SumBackward<T> {
        fn backward(&self, _grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
            let n = self.input.numel();
            let ones = vec![<T as num_traits::One>::one(); n];
            let g =
                Tensor::from_storage(TensorStorage::cpu(ones), self.input.shape().to_vec(), false)?;
            Ok(vec![Some(g)])
        }

        fn inputs(&self) -> Vec<&Tensor<T>> {
            vec![&self.input]
        }

        fn name(&self) -> &'static str {
            "SumBackward"
        }
    }

    /// Helper: wrap a tensor in sum-to-scalar so backward() can be called.
    fn sum_to_scalar(t: &Tensor<f32>) -> Tensor<f32> {
        let data = t.data().unwrap();
        let total: f32 = data.iter().sum();
        Tensor::from_operation(
            TensorStorage::cpu(vec![total]),
            vec![],
            Arc::new(SumBackward { input: t.clone() }),
        )
        .unwrap()
    }

    // -- reshape --

    #[test]
    fn test_reshape_forward() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        let y = reshape(&x, &[3, 2]).unwrap();
        assert_eq!(y.shape(), &[3, 2]);
        assert_eq!(y.data().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_reshape_infer_dim() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6], false);
        let y = reshape(&x, &[2, -1]).unwrap();
        assert_eq!(y.shape(), &[2, 3]);
    }

    #[test]
    fn test_reshape_backward() {
        // x: [2,3] -> reshape to [3,2] -> sum -> scalar -> backward
        // grad_output at reshape is ones([3,2]), backward produces ones([2,3]).
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
        let y = reshape(&x, &[3, 2]).unwrap();
        let loss = sum_to_scalar(&y);

        backward(&loss).unwrap();

        let grad = x.grad().unwrap().expect("x should have a gradient");
        assert_eq!(grad.shape(), &[2, 3]);
        for &v in grad.data().unwrap() {
            assert!((v - 1.0).abs() < 1e-6, "expected 1.0, got {v}");
        }
    }

    #[test]
    fn test_reshape_shape_mismatch() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], false);
        assert!(reshape(&x, &[2, 2]).is_err());
    }

    // -- flatten --

    #[test]
    fn test_flatten_forward() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2], false);
        let y = flatten(&x).unwrap();
        assert_eq!(y.shape(), &[4]);
        assert_eq!(y.data().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_flatten_backward() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
        let y = flatten(&x).unwrap();
        let loss = sum_to_scalar(&y);

        backward(&loss).unwrap();

        let grad = x.grad().unwrap().expect("x should have a gradient");
        assert_eq!(grad.shape(), &[2, 3]);
    }

    // -- squeeze / unsqueeze --

    #[test]
    fn test_squeeze_forward() {
        let x = leaf(&[1.0, 2.0, 3.0], &[1, 3], false);
        let y = squeeze(&x, 0).unwrap();
        assert_eq!(y.shape(), &[3]);
    }

    #[test]
    fn test_squeeze_non_one_error() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], false);
        assert!(squeeze(&x, 0).is_err());
    }

    #[test]
    fn test_unsqueeze_forward() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], false);
        let y = unsqueeze(&x, 0).unwrap();
        assert_eq!(y.shape(), &[1, 3]);

        let z = unsqueeze(&x, -1).unwrap();
        assert_eq!(z.shape(), &[3, 1]);
    }

    #[test]
    fn test_squeeze_unsqueeze_roundtrip() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let y = unsqueeze(&x, 1).unwrap();
        assert_eq!(y.shape(), &[3, 1]);
        let z = squeeze(&y, 1).unwrap();
        assert_eq!(z.shape(), &[3]);
        assert_eq!(z.data().unwrap(), &[1.0, 2.0, 3.0]);
    }

    // -- transpose --

    #[test]
    fn test_transpose_2d_forward() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        let y = transpose_2d(&x).unwrap();
        assert_eq!(y.shape(), &[3, 2]);
        // Transpose is now a zero-copy stride swap; use data_vec for logical order.
        assert_eq!(y.data_vec().unwrap(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    // -- cat --

    #[test]
    fn test_cat_forward_axis0() {
        let a = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2], false);
        let b = leaf(&[5.0, 6.0], &[1, 2], false);
        let c = cat(&[a, b], 0).unwrap();
        assert_eq!(c.shape(), &[3, 2]);
        assert_eq!(c.data().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_cat_forward_axis1() {
        let a = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2], false);
        let b = leaf(&[5.0, 6.0], &[2, 1], false);
        let c = cat(&[a, b], 1).unwrap();
        assert_eq!(c.shape(), &[2, 3]);
        assert_eq!(c.data().unwrap(), &[1.0, 2.0, 5.0, 3.0, 4.0, 6.0]);
    }

    #[test]
    fn test_cat_backward_axis0() {
        let a = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2], true);
        let b = leaf(&[5.0, 6.0], &[1, 2], true);
        let c = cat(&[a.clone(), b.clone()], 0).unwrap();
        let loss = sum_to_scalar(&c);

        backward(&loss).unwrap();

        let a_grad = a.grad().unwrap().expect("a should have gradient");
        assert_eq!(a_grad.shape(), &[2, 2]);
        for &v in a_grad.data().unwrap() {
            assert!((v - 1.0).abs() < 1e-6);
        }

        let b_grad = b.grad().unwrap().expect("b should have gradient");
        assert_eq!(b_grad.shape(), &[1, 2]);
        for &v in b_grad.data().unwrap() {
            assert!((v - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_cat_backward_axis1() {
        let a = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2], true);
        let b = leaf(&[5.0, 6.0], &[2, 1], true);
        let c = cat(&[a.clone(), b.clone()], 1).unwrap();
        let loss = sum_to_scalar(&c);

        backward(&loss).unwrap();

        let a_grad = a.grad().unwrap().expect("a should have gradient");
        assert_eq!(a_grad.shape(), &[2, 2]);
        for &v in a_grad.data().unwrap() {
            assert!((v - 1.0).abs() < 1e-6);
        }

        let b_grad = b.grad().unwrap().expect("b should have gradient");
        assert_eq!(b_grad.shape(), &[2, 1]);
        for &v in b_grad.data().unwrap() {
            assert!((v - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_cat_backward_mixed_requires_grad() {
        let a = leaf(&[1.0, 2.0], &[2], true);
        let b = leaf(&[3.0, 4.0], &[2], false);
        let c = cat(&[a.clone(), b.clone()], 0).unwrap();
        let loss = sum_to_scalar(&c);

        backward(&loss).unwrap();

        let a_grad = a.grad().unwrap().expect("a should have gradient");
        assert_eq!(a_grad.shape(), &[2]);
        for &v in a_grad.data().unwrap() {
            assert!((v - 1.0).abs() < 1e-6);
        }

        assert!(b.grad().unwrap().is_none());
    }

    #[test]
    fn test_cat_empty_error() {
        let result: FerrotorchResult<Tensor<f32>> = cat(&[], 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_cat_1d() {
        let a = leaf(&[1.0, 2.0], &[2], false);
        let b = leaf(&[3.0, 4.0, 5.0], &[3], false);
        let c = cat(&[a, b], 0).unwrap();
        assert_eq!(c.shape(), &[5]);
        assert_eq!(c.data().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    // -- no_grad --

    #[test]
    fn test_reshape_no_grad() {
        crate::autograd::no_grad(|| {
            let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[4], true);
            let y = reshape(&x, &[2, 2]).unwrap();
            assert!(y.grad_fn().is_none());
        });
    }

    // -- resolve_shape helper --

    #[test]
    fn test_resolve_shape_basic() {
        assert_eq!(resolve_shape(&[2, 3], 6).unwrap(), vec![2, 3]);
    }

    #[test]
    fn test_resolve_shape_infer() {
        assert_eq!(resolve_shape(&[2, -1], 6).unwrap(), vec![2, 3]);
        assert_eq!(resolve_shape(&[-1, 2], 6).unwrap(), vec![3, 2]);
        assert_eq!(resolve_shape(&[-1], 6).unwrap(), vec![6]);
    }

    #[test]
    fn test_resolve_shape_multiple_infer_error() {
        assert!(resolve_shape(&[-1, -1], 6).is_err());
    }

    #[test]
    fn test_resolve_shape_mismatch() {
        assert!(resolve_shape(&[2, 2], 6).is_err());
    }

    // -- graph preservation through shape ops --
    //
    // Shape ops must produce non-leaf tensors with grad_fn when the input
    // requires_grad. This is critical on GPU where `Tensor::to()` creates
    // detached leaf tensors — a `restore_device(from_operation(...))` pattern
    // would sever the graph.

    #[test]
    fn test_squeeze_preserves_grad_fn() {
        let x = leaf(&[1.0, 2.0, 3.0], &[1, 3], true);
        let y = squeeze(&x, 0).unwrap();
        assert!(y.grad_fn().is_some(), "squeeze must attach a grad_fn");
        assert!(!y.is_leaf(), "squeeze output must be non-leaf");
        assert!(y.requires_grad(), "squeeze output must require grad");
    }

    #[test]
    fn test_unsqueeze_preserves_grad_fn() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let y = unsqueeze(&x, 0).unwrap();
        assert!(y.grad_fn().is_some(), "unsqueeze must attach a grad_fn");
        assert!(!y.is_leaf(), "unsqueeze output must be non-leaf");
        assert!(y.requires_grad(), "unsqueeze output must require grad");
    }

    #[test]
    fn test_flatten_preserves_grad_fn() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2], true);
        let y = flatten(&x).unwrap();
        assert!(y.grad_fn().is_some(), "flatten must attach a grad_fn");
        assert!(!y.is_leaf(), "flatten output must be non-leaf");
        assert!(y.requires_grad(), "flatten output must require grad");
    }

    #[test]
    fn test_squeeze_backward_reaches_leaf() {
        // Simulates the goodness_from_output pattern: x -> pow -> mm -> squeeze -> loss.
        // The squeeze backward must propagate gradients back to x.
        let x = leaf(&[1.0, 2.0, 3.0], &[3, 1], true);
        let squeezed = squeeze(&x, 1).unwrap();
        let loss = sum_to_scalar(&squeezed);

        backward(&loss).unwrap();

        let grad = x
            .grad()
            .unwrap()
            .expect("squeeze must propagate gradients to leaf input");
        assert_eq!(grad.shape(), &[3, 1]);
        for &v in grad.data().unwrap() {
            assert!((v - 1.0).abs() < 1e-6, "expected gradient 1.0, got {v}");
        }
    }

    #[test]
    fn test_unsqueeze_backward_reaches_leaf() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let unsqueezed = unsqueeze(&x, 1).unwrap();
        let loss = sum_to_scalar(&unsqueezed);

        backward(&loss).unwrap();

        let grad = x
            .grad()
            .unwrap()
            .expect("unsqueeze must propagate gradients to leaf input");
        assert_eq!(grad.shape(), &[3]);
        for &v in grad.data().unwrap() {
            assert!((v - 1.0).abs() < 1e-6, "expected gradient 1.0, got {v}");
        }
    }

    #[test]
    fn test_squeeze_in_longer_chain() {
        // Mirrors the FF loss computation graph: leaf -> op -> squeeze -> op -> scalar.
        // Backward must reach the original leaf through the squeeze node.
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], true);

        // Multiply x by a constant (creates an intermediate with grad_fn).
        let two = leaf(&[2.0; 6], &[3, 2], false);
        let scaled = crate::grad_fns::arithmetic::mul(&x, &two).unwrap();

        // Sum columns via matmul with ones to get [3, 1].
        let ones = leaf(&[1.0, 1.0], &[2, 1], false);
        let row_sums = crate::grad_fns::linalg::mm_differentiable(&scaled, &ones).unwrap();

        // Squeeze to [3] — this is the operation that previously severed the graph on GPU.
        let squeezed = squeeze(&row_sums, 1).unwrap();
        assert!(squeezed.grad_fn().is_some(), "squeeze must preserve graph");

        let loss = sum_to_scalar(&squeezed);
        backward(&loss).unwrap();

        let grad = x
            .grad()
            .unwrap()
            .expect("backward through squeeze in a longer chain must reach leaf parameters");
        assert_eq!(grad.shape(), &[3, 2]);
        // d(loss)/d(x) = 2.0 * ones (from the scaling and sum).
        for &v in grad.data().unwrap() {
            assert!((v - 2.0).abs() < 1e-6, "expected gradient 2.0, got {v}");
        }
    }

    #[test]
    fn test_shape_ops_share_storage_with_input() {
        // view_operation must share storage — no data copy.
        // This catches the old ensure_cpu/restore_device pattern which
        // allocated new storage even on CPU.
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
        let flat = flatten(&x).unwrap();

        assert_eq!(flat.data().unwrap(), x.data().unwrap());
        assert_eq!(flat.shape(), &[6]);
        // Pointer equality: must be the same Arc, not a copy.
        assert!(
            flat.shares_storage(&x),
            "flatten should share storage with input (zero-copy)"
        );

        let orig = leaf(&[1.0, 2.0, 3.0], &[1, 3], true);
        let sq2 = squeeze(&orig, 0).unwrap();
        assert!(
            sq2.shares_storage(&orig),
            "squeeze should share storage with input (zero-copy)"
        );

        let orig3 = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let us = unsqueeze(&orig3, 0).unwrap();
        assert!(
            us.shares_storage(&orig3),
            "unsqueeze should share storage with input (zero-copy)"
        );
    }

    #[test]
    fn test_squeeze_no_grad_is_view() {
        // Without requires_grad, squeeze should be a cheap view_reshape.
        let x = leaf(&[1.0, 2.0, 3.0], &[1, 3], false);
        let y = squeeze(&x, 0).unwrap();
        assert!(y.grad_fn().is_none());
        assert_eq!(y.shape(), &[3]);
        assert_eq!(y.data().unwrap(), &[1.0, 2.0, 3.0]);
    }

    // -- roll backward (#1014) --

    #[test]
    fn test_roll_forward_registers_grad_fn() {
        // Forward must attach RollBackward when input requires_grad.
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5], true);
        let y = crate::ops::tensor_ops::roll(&x, 2, 0).unwrap();
        // shifts=2 along dim 0 of size 5: [1,2,3,4,5] -> [4,5,1,2,3]
        assert_eq!(y.data().unwrap(), &[4.0, 5.0, 1.0, 2.0, 3.0]);
        assert!(y.requires_grad());
        assert!(!y.is_leaf());
        assert_eq!(y.grad_fn().unwrap().name(), "RollBackward");
    }

    #[test]
    fn test_roll_zero_shift_early_return() {
        // shifts ≡ 0 mod dim_size collapses to identity. The result is
        // a clone of the input; if the input is a leaf with requires_grad,
        // the clone is also a leaf — there's nothing to backprop through
        // that wasn't already trivial.
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let y = crate::ops::tensor_ops::roll(&x, 0, 0).unwrap();
        assert_eq!(y.data().unwrap(), &[1.0, 2.0, 3.0]);
        // shifts=3 mod 3 = 0 → also identity.
        let y2 = crate::ops::tensor_ops::roll(&x, 3, 0).unwrap();
        assert_eq!(y2.data().unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_roll_backward_simple_1d_hand_computed() {
        // x = [10, 20, 30, 40, 50], requires_grad
        // y = roll(x, 2, 0) = [40, 50, 10, 20, 30]
        // loss = sum(y * w) where w = [1, 2, 3, 4, 5]
        //
        // dy/dx_i is a permutation: y_j = x_{(j - 2) mod 5}, so the
        // backward maps grad_y[j] back to x[(j - 2) mod 5]. With
        // grad_y = w = [1,2,3,4,5], the expected grad_x equals
        // roll(grad_y, -2, 0) = [3, 4, 5, 1, 2].
        let x = leaf(&[10.0, 20.0, 30.0, 40.0, 50.0], &[5], true);
        let y = crate::ops::tensor_ops::roll(&x, 2, 0).unwrap();

        // Use a custom WeightedSumBackward: loss = sum(y[i] * w[i]),
        // grad_y = w. This gives a non-uniform grad_y, which exposes
        // the permutation direction (uniform ones would be invariant
        // under any roll and could not detect a sign error).
        #[derive(Debug)]
        struct WeightedSumBackward<T: Float> {
            input: Tensor<T>,
            weights: Vec<T>,
        }
        impl<T: Float> GradFn<T> for WeightedSumBackward<T> {
            fn backward(
                &self,
                _grad_output: &Tensor<T>,
            ) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
                let g = Tensor::from_storage(
                    TensorStorage::cpu(self.weights.clone()),
                    self.input.shape().to_vec(),
                    false,
                )?;
                Ok(vec![Some(g)])
            }
            fn inputs(&self) -> Vec<&Tensor<T>> {
                vec![&self.input]
            }
            fn name(&self) -> &'static str {
                "WeightedSumBackward"
            }
        }

        let w = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let total: f32 = y
            .data()
            .unwrap()
            .iter()
            .zip(w.iter())
            .map(|(yi, wi)| yi * wi)
            .sum();
        let loss = Tensor::from_operation(
            TensorStorage::cpu(vec![total]),
            vec![],
            Arc::new(WeightedSumBackward {
                input: y.clone(),
                weights: w,
            }),
        )
        .unwrap();

        backward(&loss).unwrap();

        let grad = x.grad().unwrap().expect("x should have a gradient");
        let gd = grad.data().unwrap();
        // Expected: roll([1,2,3,4,5], -2, 0) = [3, 4, 5, 1, 2]
        let expected = [3.0, 4.0, 5.0, 1.0, 2.0];
        for (i, (&g, &e)) in gd.iter().zip(expected.iter()).enumerate() {
            assert!((g - e).abs() < 1e-6, "grad[{i}] = {g}, expected {e}");
        }
    }

    #[test]
    fn test_roll_backward_negative_shift_2d() {
        // x: shape [2, 3], data [[1,2,3],[4,5,6]]
        // y = roll(x, -1, 1) shifts each row left by 1:
        //   row 0: [2, 3, 1]
        //   row 1: [5, 6, 4]
        // grad_y = [[1, 10, 100], [1000, 10000, 100000]]
        // backward: grad_x = roll(grad_y, +1, 1):
        //   row 0: [100, 1, 10]
        //   row 1: [100000, 1000, 10000]
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
        let y = crate::ops::tensor_ops::roll(&x, -1, 1).unwrap();
        assert_eq!(y.data().unwrap(), &[2.0, 3.0, 1.0, 5.0, 6.0, 4.0]);

        // Apply RollBackward directly with a hand-built grad_output.
        let grad_output = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0_f32, 10.0, 100.0, 1000.0, 10000.0, 100000.0]),
            vec![2, 3],
            false,
        )
        .unwrap();
        let grad_fn = y.grad_fn().expect("y must carry RollBackward");
        let grads = grad_fn.backward(&grad_output).unwrap();
        let g = grads[0].as_ref().expect("grad must be Some");
        let gd = g.data().unwrap();
        let expected = [100.0_f32, 1.0, 10.0, 100000.0, 1000.0, 10000.0];
        for (i, (&got, &exp)) in gd.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "grad[{i}] = {got}, expected {exp}"
            );
        }
    }

    // -- unflatten (REQ-4, #1342) --

    #[test]
    fn test_unflatten_forward() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        // Unflatten dim 1 (size 3) into [3] is a no-op-ish reshape preserving
        // values; unflatten dim 0 (size 2) into [2, 1] splices a singleton.
        let y = unflatten(&x, 0, &[2, 1]).unwrap();
        assert_eq!(y.shape(), &[2, 1, 3]);
        assert_eq!(y.data_vec().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_unflatten_infer_slot() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6], false);
        // dim 0 size 6 -> [2, -1] resolves the -1 to 3 (local to the dim).
        let y = unflatten(&x, 0, &[2, -1]).unwrap();
        assert_eq!(y.shape(), &[2, 3]);
        assert_eq!(y.data_vec().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_unflatten_negative_dim_and_middle_splice() {
        let x = leaf(
            &(0..24).map(|v| v as f32).collect::<Vec<_>>(),
            &[2, 12, 1],
            false,
        );
        // dim=-2 normalizes to 1 (size 12) -> [3, 4]; outer/inner dims kept.
        let y = unflatten(&x, -2, &[3, 4]).unwrap();
        assert_eq!(y.shape(), &[2, 3, 4, 1]);
    }

    #[test]
    fn test_unflatten_empty_sizes_errors() {
        let x = leaf(&[1.0, 2.0], &[2], false);
        assert!(unflatten(&x, 0, &[]).is_err());
    }

    #[test]
    fn test_unflatten_product_mismatch_errors() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6], false);
        // 2 * 4 = 8 != 6.
        assert!(unflatten(&x, 0, &[2, 4]).is_err());
    }

    #[test]
    fn test_unflatten_backward_reaches_leaf() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6], true);
        let y = unflatten(&x, 0, &[2, 3]).unwrap();
        let loss = sum_to_scalar(&y);
        backward(&loss).unwrap();
        let g = x.grad().unwrap().expect("x should have gradient");
        assert_eq!(g.shape(), &[6]);
        for &v in g.data().unwrap() {
            assert!((v - 1.0).abs() < 1e-6);
        }
    }

    // -- swapaxes / swapdims (REQ-9, REQ-10, #1342) --

    #[test]
    fn test_swapaxes_equals_transpose() {
        // R-CHAR-3: swapaxes is upstream-defined as `self.transpose(a, b)`
        // (TensorShape.cpp:4776-4778). Assert byte-equality with transpose.
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        let via_swap = swapaxes(&x, 0, 1).unwrap();
        let via_transpose = x.transpose(0, 1).unwrap();
        assert_eq!(via_swap.shape(), via_transpose.shape());
        assert_eq!(
            via_swap.data_vec().unwrap(),
            via_transpose.data_vec().unwrap()
        );
        assert_eq!(via_swap.shape(), &[3, 2]);
        assert_eq!(
            via_swap.data_vec().unwrap(),
            &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
    }

    #[test]
    fn test_swapdims_equals_transpose() {
        // swapdims is upstream-defined as `self.transpose(d0, d1)`
        // (TensorShape.cpp:4784-4786).
        let x = leaf(
            &(0..24).map(|v| v as f32).collect::<Vec<_>>(),
            &[2, 3, 4],
            false,
        );
        let via_swap = swapdims(&x, 0, 2).unwrap();
        let via_transpose = x.transpose(0, 2).unwrap();
        assert_eq!(via_swap.shape(), &[4, 3, 2]);
        assert_eq!(
            via_swap.data_vec().unwrap(),
            via_transpose.data_vec().unwrap()
        );
    }

    #[test]
    fn test_swapaxes_backward_reaches_leaf() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
        // swapaxes yields a stride view; materialize before the host-side
        // sum_to_scalar helper which reads contiguous data.
        let y = crate::methods::contiguous_t(&swapaxes(&x, 0, 1).unwrap()).unwrap();
        let loss = sum_to_scalar(&y);
        backward(&loss).unwrap();
        let g = x.grad().unwrap().expect("x should have gradient");
        assert_eq!(g.shape(), &[2, 3]);
        for &v in g.data().unwrap() {
            assert!((v - 1.0).abs() < 1e-6);
        }
    }

    // -- expand_as (REQ-12, #1342) --

    #[test]
    fn test_expand_as_equals_expand() {
        // expand_as is upstream-defined as `self.expand(other.sizes())`
        // (TensorShape.cpp:1374-1376).
        let x = leaf(&[1.0, 2.0, 3.0], &[1, 3], false);
        let other = leaf(&[0.0; 12], &[4, 3], false);
        let via_expand_as = expand_as(&x, &other).unwrap();
        let via_expand = expand(&x, &[4, 3]).unwrap();
        assert_eq!(via_expand_as.shape(), &[4, 3]);
        assert_eq!(
            via_expand_as.data_vec().unwrap(),
            via_expand.data_vec().unwrap()
        );
        assert_eq!(
            via_expand_as.data_vec().unwrap(),
            &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn test_expand_as_backward_sums_broadcast_axes() {
        let x = leaf(&[1.0, 2.0, 3.0], &[1, 3], true);
        let other = leaf(&[0.0; 12], &[4, 3], false);
        let y = expand_as(&x, &other).unwrap();
        let loss = sum_to_scalar(&y);
        backward(&loss).unwrap();
        let g = x.grad().unwrap().expect("x should have gradient");
        // The size-1 axis was broadcast to 4, so each element accumulates 4×1.
        assert_eq!(g.shape(), &[1, 3]);
        for &v in g.data().unwrap() {
            assert!((v - 4.0).abs() < 1e-6, "expected 4.0, got {v}");
        }
    }

    // -- flip / fliplr / flipud (REQ-34/35/36, #1342) --

    #[test]
    fn test_flip_forward_1d() {
        // torch.flip([1,2,3,4], [0]) == [4,3,2,1] (TensorTransformations.cpp:36).
        let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[4], false);
        let y = flip(&x, &[0]).unwrap();
        assert_eq!(y.shape(), &[4]);
        assert_eq!(y.data_vec().unwrap(), &[4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_flip_forward_2d_both_dims() {
        // x = [[1,2,3],[4,5,6]]; flip both dims reverses rows AND columns:
        // torch.flip(x, [0,1]) == [[6,5,4],[3,2,1]].
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        let y = flip(&x, &[0, 1]).unwrap();
        assert_eq!(y.shape(), &[2, 3]);
        assert_eq!(y.data_vec().unwrap(), &[6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_flip_forward_2d_single_dim() {
        // flip(x, [1]) reverses each row: [[3,2,1],[6,5,4]].
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        let y = flip(&x, &[1]).unwrap();
        assert_eq!(y.data_vec().unwrap(), &[3.0, 2.0, 1.0, 6.0, 5.0, 4.0]);
    }

    #[test]
    fn test_flip_rejects_duplicate_dim() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2], false);
        assert!(flip(&x, &[0, 0]).is_err());
    }

    #[test]
    fn test_flip_backward_is_self_inverse() {
        // flip is a permutation; the VJP re-flips. With a non-uniform upstream
        // gradient (via the row-distinct weights), grad_x = flip(grad_y, dims).
        // x=[10,20,30], y=flip(x,[0])=[30,20,10]; grad_y=[1,2,3] (weights) ⇒
        // grad_x = flip([1,2,3],[0]) = [3,2,1].
        let x = leaf(&[10.0, 20.0, 30.0], &[3], true);
        let y = flip(&x, &[0]).unwrap();
        assert_eq!(y.data_vec().unwrap(), &[30.0, 20.0, 10.0]);

        #[derive(Debug)]
        struct WSum {
            input: Tensor<f32>,
            w: Vec<f32>,
        }
        impl GradFn<f32> for WSum {
            fn backward(&self, _g: &Tensor<f32>) -> FerrotorchResult<Vec<Option<Tensor<f32>>>> {
                Ok(vec![Some(
                    Tensor::from_storage(
                        TensorStorage::cpu(self.w.clone()),
                        self.input.shape().to_vec(),
                        false,
                    )
                    .unwrap(),
                )])
            }
            fn inputs(&self) -> Vec<&Tensor<f32>> {
                vec![&self.input]
            }
            fn name(&self) -> &'static str {
                "WSum"
            }
        }
        let w = vec![1.0_f32, 2.0, 3.0];
        let total: f32 = y.data().unwrap().iter().zip(&w).map(|(a, b)| a * b).sum();
        let loss = Tensor::from_operation(
            TensorStorage::cpu(vec![total]),
            vec![],
            Arc::new(WSum {
                input: y.clone(),
                w,
            }),
        )
        .unwrap();
        backward(&loss).unwrap();
        let g = x.grad().unwrap().expect("x should have gradient");
        assert_eq!(g.data().unwrap(), &[3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_fliplr_equals_flip_dim1() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        assert_eq!(
            fliplr(&x).unwrap().data_vec().unwrap(),
            flip(&x, &[1]).unwrap().data_vec().unwrap()
        );
        assert!(fliplr(&leaf(&[1.0, 2.0], &[2], false)).is_err());
    }

    #[test]
    fn test_flipud_equals_flip_dim0() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        assert_eq!(
            flipud(&x).unwrap().data_vec().unwrap(),
            flip(&x, &[0]).unwrap().data_vec().unwrap()
        );
    }

    // -- rot90 (REQ-33, #1342) --

    #[test]
    fn test_rot90_k1() {
        // torch.rot90 of [[1,2],[3,4]] by k=1 in dims (0,1) ==
        // flip({1}).transpose(0,1) = [[2,4],[1,3]] (TensorTransformations.cpp:170).
        let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2], false);
        let y = crate::methods::contiguous_t(&rot90(&x, 1, &[0, 1]).unwrap()).unwrap();
        assert_eq!(y.shape(), &[2, 2]);
        assert_eq!(y.data_vec().unwrap(), &[2.0, 4.0, 1.0, 3.0]);
    }

    #[test]
    fn test_rot90_k2_is_flip_both() {
        // k=2 ⇒ flip(dims) = flip both ⇒ [[4,3],[2,1]].
        let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2], false);
        let y = rot90(&x, 2, &[0, 1]).unwrap();
        assert_eq!(y.data_vec().unwrap(), &[4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_rot90_k0_and_k4_identity() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2], false);
        assert_eq!(
            rot90(&x, 0, &[0, 1]).unwrap().data_vec().unwrap(),
            &[1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(
            rot90(&x, 4, &[0, 1]).unwrap().data_vec().unwrap(),
            &[1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn test_rot90_negative_k() {
        // k=-1 ≡ 3 mod 4 ⇒ flip({0}).transpose(0,1) = [[3,1],[4,2]].
        let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2], false);
        let y = crate::methods::contiguous_t(&rot90(&x, -1, &[0, 1]).unwrap()).unwrap();
        assert_eq!(y.data_vec().unwrap(), &[3.0, 1.0, 4.0, 2.0]);
    }

    #[test]
    fn test_rot90_backward_reaches_leaf() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2], true);
        let y = crate::methods::contiguous_t(&rot90(&x, 1, &[0, 1]).unwrap()).unwrap();
        let loss = sum_to_scalar(&y);
        backward(&loss).unwrap();
        let g = x.grad().unwrap().expect("x should have gradient");
        assert_eq!(g.shape(), &[2, 2]);
        for &v in g.data().unwrap() {
            assert!((v - 1.0).abs() < 1e-6);
        }
    }

    // -- movedim / moveaxis (REQ-29/30, #1342) --

    #[test]
    fn test_movedim_single() {
        // x shape [2,3,4]; movedim(0, 2) -> permute [1,2,0] -> shape [3,4,2].
        let x = leaf(
            &(0..24).map(|v| v as f32).collect::<Vec<_>>(),
            &[2, 3, 4],
            false,
        );
        let y = movedim(&x, &[0], &[2]).unwrap();
        assert_eq!(y.shape(), &[3, 4, 2]);
        // Equivalent to permute([1,2,0]).
        let viap = crate::methods::permute_t(&x, &[1, 2, 0]).unwrap();
        assert_eq!(
            crate::methods::contiguous_t(&y)
                .unwrap()
                .data_vec()
                .unwrap(),
            crate::methods::contiguous_t(&viap)
                .unwrap()
                .data_vec()
                .unwrap()
        );
    }

    #[test]
    fn test_movedim_multi() {
        // src=[0,1] dst=[2,4] on a 5-D tensor reproduces the upstream worked
        // example: order = [2,3,0,4,1] (TensorShape.cpp:4756).
        let x = leaf(&vec![0.0; 2 * 3 * 4 * 5 * 6], &[2, 3, 4, 5, 6], false);
        let y = movedim(&x, &[0, 1], &[2, 4]).unwrap();
        let viap = crate::methods::permute_t(&x, &[2, 3, 0, 4, 1]).unwrap();
        assert_eq!(y.shape(), viap.shape());
        assert_eq!(y.shape(), &[4, 5, 2, 6, 3]);
    }

    #[test]
    fn test_moveaxis_equals_movedim() {
        let x = leaf(
            &(0..24).map(|v| v as f32).collect::<Vec<_>>(),
            &[2, 3, 4],
            false,
        );
        assert_eq!(
            moveaxis(&x, &[2], &[0]).unwrap().shape(),
            movedim(&x, &[2], &[0]).unwrap().shape()
        );
    }

    #[test]
    fn test_movedim_backward_reaches_leaf() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
        let y = crate::methods::contiguous_t(&movedim(&x, &[0], &[1]).unwrap()).unwrap();
        let loss = sum_to_scalar(&y);
        backward(&loss).unwrap();
        let g = x.grad().unwrap().expect("x should have gradient");
        assert_eq!(g.shape(), &[2, 3]);
        for &v in g.data().unwrap() {
            assert!((v - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_movedim_rejects_repeated_dim() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2], false);
        assert!(movedim(&x, &[0, 0], &[0, 1]).is_err());
        assert!(movedim(&x, &[0, 1], &[1, 1]).is_err());
        assert!(movedim(&x, &[0], &[0, 1]).is_err());
    }

    // -- broadcast_to / broadcast_tensors (REQ-27/26, #1342) --

    #[test]
    fn test_broadcast_to_equals_expand() {
        let x = leaf(&[1.0, 2.0, 3.0], &[1, 3], false);
        let y = broadcast_to(&x, &[2, 3]).unwrap();
        let e = expand(&x, &[2, 3]).unwrap();
        assert_eq!(y.shape(), &[2, 3]);
        assert_eq!(y.data_vec().unwrap(), e.data_vec().unwrap());
        assert_eq!(y.data_vec().unwrap(), &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_broadcast_tensors_common_shape() {
        // [3,1] and [1,4] broadcast to [3,4].
        let a = leaf(&[1.0, 2.0, 3.0], &[3, 1], false);
        let b = leaf(&[10.0, 20.0, 30.0, 40.0], &[1, 4], false);
        let out = broadcast_tensors(&[a, b]).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].shape(), &[3, 4]);
        assert_eq!(out[1].shape(), &[3, 4]);
        // a expands by repeating each row element across the 4 columns.
        assert_eq!(
            out[0].data_vec().unwrap(),
            &[1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0]
        );
    }

    // -- repeat / tile (REQ-13/31, #1342) --

    #[test]
    fn test_repeat_1d() {
        // torch.tensor([1,2,3]).repeat(2) == [1,2,3,1,2,3] (TensorShape.cpp:1909).
        let x = leaf(&[1.0, 2.0, 3.0], &[3], false);
        let y = repeat(&x, &[2]).unwrap();
        assert_eq!(y.shape(), &[6]);
        assert_eq!(y.data_vec().unwrap(), &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_repeat_2d_with_new_leading_dim() {
        // [1,2] with repeats [2,2] -> shape [2,4]: each row is [1,2,1,2],
        // two such rows. (input promoted from [2] to [1,2] then tiled.)
        let x = leaf(&[1.0, 2.0], &[2], false);
        let y = repeat(&x, &[2, 2]).unwrap();
        assert_eq!(y.shape(), &[2, 4]);
        assert_eq!(
            y.data_vec().unwrap(),
            &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
        );
    }

    #[test]
    fn test_repeat_rejects_too_few_dims() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2], false);
        assert!(repeat(&x, &[2]).is_err());
    }

    #[test]
    fn test_tile_pads_reps() {
        // tile of a [2,2] tensor with reps (2,) is treated as (1,2):
        // tile each row twice horizontally. [[1,2],[3,4]] -> [[1,2,1,2],[3,4,3,4]].
        let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2], false);
        let y = tile(&x, &[2]).unwrap();
        assert_eq!(y.shape(), &[2, 4]);
        assert_eq!(
            y.data_vec().unwrap(),
            &[1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]
        );
    }

    #[test]
    fn test_repeat_backward_accumulates() {
        // grad of a tile is the sum over copies. x=[1,2], repeat 3 along dim 0
        // -> [1,2,1,2,1,2]; d(sum)/dx = [3,3] (each element copied 3×).
        let x = leaf(&[1.0, 2.0], &[2], true);
        let y = repeat(&x, &[3]).unwrap();
        assert_eq!(y.shape(), &[6]);
        let loss = sum_to_scalar(&y);
        backward(&loss).unwrap();
        let g = x.grad().unwrap().expect("x should have gradient");
        assert_eq!(g.shape(), &[2]);
        for &v in g.data().unwrap() {
            assert!((v - 3.0).abs() < 1e-6, "expected 3.0, got {v}");
        }
    }

    // -- repeat_interleave (REQ-14, #1342) --

    #[test]
    fn test_repeat_interleave_1d() {
        // torch.repeat_interleave([1,2,3], 2) == [1,1,2,2,3,3].
        let x = leaf(&[1.0, 2.0, 3.0], &[3], false);
        let y = repeat_interleave(&x, 2, 0).unwrap();
        assert_eq!(y.shape(), &[6]);
        assert_eq!(y.data_vec().unwrap(), &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
    }

    #[test]
    fn test_repeat_interleave_2d_dim1() {
        // x=[[1,2],[3,4]]; repeat_interleave(2, dim=1) duplicates each column
        // in place: [[1,1,2,2],[3,3,4,4]].
        let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2], false);
        let y = repeat_interleave(&x, 2, 1).unwrap();
        assert_eq!(y.shape(), &[2, 4]);
        assert_eq!(
            y.data_vec().unwrap(),
            &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]
        );
    }

    #[test]
    fn test_repeat_interleave_differs_from_repeat() {
        // interleave: [a,a,b,b]; repeat/tile: [a,b,a,b] — distinct orderings.
        let x = leaf(&[1.0, 2.0], &[2], false);
        assert_eq!(
            repeat_interleave(&x, 2, 0).unwrap().data_vec().unwrap(),
            &[1.0, 1.0, 2.0, 2.0]
        );
        assert_eq!(
            repeat(&x, &[2]).unwrap().data_vec().unwrap(),
            &[1.0, 2.0, 1.0, 2.0]
        );
    }

    #[test]
    fn test_repeat_interleave_backward_sums_segments() {
        // x=[1,2], interleave 3 -> [1,1,1,2,2,2]; d(sum)/dx = [3,3].
        let x = leaf(&[1.0, 2.0], &[2], true);
        let y = repeat_interleave(&x, 3, 0).unwrap();
        let loss = sum_to_scalar(&y);
        backward(&loss).unwrap();
        let g = x.grad().unwrap().expect("x should have gradient");
        assert_eq!(g.shape(), &[2]);
        for &v in g.data().unwrap() {
            assert!((v - 3.0).abs() < 1e-6, "expected 3.0, got {v}");
        }
    }

    // -- unbind (REQ-25, #1342) --

    #[test]
    fn test_unbind_dim0() {
        // x=[[1,2,3],[4,5,6]]; unbind(0) -> [[1,2,3], [4,5,6]] (each 1-D).
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        let parts = unbind(&x, 0).unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].shape(), &[3]);
        assert_eq!(
            crate::methods::contiguous_t(&parts[0])
                .unwrap()
                .data_vec()
                .unwrap(),
            &[1.0, 2.0, 3.0]
        );
        assert_eq!(
            crate::methods::contiguous_t(&parts[1])
                .unwrap()
                .data_vec()
                .unwrap(),
            &[4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn test_unbind_dim1() {
        // unbind(1) gives 3 column slices, each shape [2].
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        let parts = unbind(&x, 1).unwrap();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].shape(), &[2]);
        assert_eq!(
            crate::methods::contiguous_t(&parts[1])
                .unwrap()
                .data_vec()
                .unwrap(),
            &[2.0, 5.0]
        );
    }

    #[test]
    fn test_unbind_backward_scatters() {
        // Backprop through one slice scatters its grad into the right row.
        let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2], true);
        let parts = unbind(&x, 0).unwrap();
        // loss = sum(parts[1]); grad should be [[0,0],[1,1]].
        let loss = sum_to_scalar(&crate::methods::contiguous_t(&parts[1]).unwrap());
        backward(&loss).unwrap();
        let g = x.grad().unwrap().expect("x should have gradient");
        assert_eq!(g.data().unwrap(), &[0.0, 0.0, 1.0, 1.0]);
    }

    // -- tensor_split (REQ-23, #1342) --

    #[test]
    fn test_tensor_split_indices() {
        // x = 0..6 along dim 0; indices [2,4] -> [0:2], [2:4], [4:6].
        let x = leaf(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[6], false);
        let parts = tensor_split(&x, &[2, 4], 0).unwrap();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].data_vec().unwrap(), &[0.0, 1.0]);
        assert_eq!(parts[1].data_vec().unwrap(), &[2.0, 3.0]);
        assert_eq!(parts[2].data_vec().unwrap(), &[4.0, 5.0]);
    }

    #[test]
    fn test_tensor_split_empty_section() {
        // Equal indices yield a zero-length middle section.
        let x = leaf(&[0.0, 1.0, 2.0, 3.0], &[4], false);
        let parts = tensor_split(&x, &[2, 2], 0).unwrap();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].shape(), &[2]);
        assert_eq!(parts[1].shape(), &[0]);
        assert_eq!(parts[2].shape(), &[2]);
    }

    #[test]
    fn test_tensor_split_backward() {
        let x = leaf(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[6], true);
        let parts = tensor_split(&x, &[2, 4], 0).unwrap();
        // Backprop only through the middle section.
        let loss = sum_to_scalar(&crate::methods::contiguous_t(&parts[1]).unwrap());
        backward(&loss).unwrap();
        let g = x.grad().unwrap().expect("x should have gradient");
        assert_eq!(g.data().unwrap(), &[0.0, 0.0, 1.0, 1.0, 0.0, 0.0]);
    }

    // -- vstack / hstack / dstack / column_stack (REQ-17/18/19/20, #1342) --

    #[test]
    fn test_vstack_1d_inputs() {
        // vstack of two 1-D [3] tensors -> [2,3] (each promoted to [1,3]).
        let a = leaf(&[1.0, 2.0, 3.0], &[3], false);
        let b = leaf(&[4.0, 5.0, 6.0], &[3], false);
        let y = vstack(&[a, b]).unwrap();
        assert_eq!(y.shape(), &[2, 3]);
        assert_eq!(y.data_vec().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_hstack_1d_inputs() {
        // hstack of 1-D tensors concatenates along dim 0.
        let a = leaf(&[1.0, 2.0], &[2], false);
        let b = leaf(&[3.0, 4.0, 5.0], &[3], false);
        let y = hstack(&[a, b]).unwrap();
        assert_eq!(y.shape(), &[5]);
        assert_eq!(y.data_vec().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_hstack_2d_inputs() {
        // hstack of 2-D tensors concatenates along dim 1.
        let a = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2], false);
        let b = leaf(&[5.0, 6.0], &[2, 1], false);
        let y = hstack(&[a, b]).unwrap();
        assert_eq!(y.shape(), &[2, 3]);
        assert_eq!(y.data_vec().unwrap(), &[1.0, 2.0, 5.0, 3.0, 4.0, 6.0]);
    }

    #[test]
    fn test_dstack_1d_inputs() {
        // dstack promotes 1-D [3] to [1,3,1] then cats dim 2 -> [1,3,2].
        let a = leaf(&[1.0, 2.0, 3.0], &[3], false);
        let b = leaf(&[4.0, 5.0, 6.0], &[3], false);
        let y = dstack(&[a, b]).unwrap();
        assert_eq!(y.shape(), &[1, 3, 2]);
        assert_eq!(y.data_vec().unwrap(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_column_stack_1d_inputs() {
        // column_stack of two 1-D [3] tensors -> [3,2] (each reshaped to [3,1]).
        let a = leaf(&[1.0, 2.0, 3.0], &[3], false);
        let b = leaf(&[4.0, 5.0, 6.0], &[3], false);
        let y = column_stack(&[a, b]).unwrap();
        assert_eq!(y.shape(), &[3, 2]);
        assert_eq!(y.data_vec().unwrap(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_vstack_backward() {
        let a = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let b = leaf(&[4.0, 5.0, 6.0], &[3], true);
        let y = vstack(&[a.clone(), b.clone()]).unwrap();
        let loss = sum_to_scalar(&y);
        backward(&loss).unwrap();
        for t in [&a, &b] {
            let g = t.grad().unwrap().expect("should have gradient");
            assert_eq!(g.shape(), &[3]);
            for &v in g.data().unwrap() {
                assert!((v - 1.0).abs() < 1e-6);
            }
        }
    }
}
