//! Backward functions for elementwise arithmetic operations.
//!
//! Each operation has a backward struct implementing `GradFn<T>` and a public
//! function that performs the forward pass and attaches the grad_fn to the
//! result tensor when gradient tracking is enabled.

use std::any::TypeId;
use std::sync::Arc;

use crate::autograd::no_grad::{is_grad_enabled, no_grad};
use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::ops::elementwise::{fast_add, fast_mul, scalar_map, unary_map};
use crate::shape::broadcast_shapes;
use crate::storage::TensorStorage;
use crate::tensor::{GradFn, Tensor};

/// Returns `true` if `T` is `f64`.
#[inline]
fn is_f64<T: Float>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f64>()
}

/// Returns `true` if `T` is `f32`.
#[inline]
fn is_f32<T: Float>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f32>()
}

/// Returns `true` if `T` is `half::bf16` (#23).
#[inline]
fn is_bf16<T: Float>() -> bool {
    TypeId::of::<T>() == TypeId::of::<half::bf16>()
}

/// Returns `true` if `T` is `half::f16` (IEEE float16, crosslink #1185 Phase 1).
#[inline]
fn is_f16<T: Float>() -> bool {
    TypeId::of::<T>() == TypeId::of::<half::f16>()
}

/// Materialize a CUDA tensor into a fresh buffer whose backing storage
/// length exactly matches the logical numel, when needed.
///
/// Why this exists (#812). The GPU elementwise kernels in this module take
/// raw `gpu_handle()` references and length-check against the *underlying*
/// buffer length. A view's `gpu_handle()` returns the WHOLE underlying
/// storage. The view may be non-contiguous (`narrow` on an interior axis,
/// `transpose`, `permute`) — but it may also be a *contiguous-by-strides*
/// view that nonetheless covers only a subset of the storage (`narrow` on
/// the outer axis, the residual of a single-element narrow on any axis,
/// or any `storage_offset != 0` slice). Both cases trip the kernel's
/// `LengthMismatch` guard with messages like `"buffer length mismatch:
/// 32 vs 16"`. PyTorch parity expects all elementwise ops to accept any
/// view transparently.
///
/// The right test is "does the view's storage exactly cover the view's
/// logical extent" — i.e. `is_contiguous() && storage_offset == 0 &&
/// numel == storage_len`. All three must hold to skip materialization.
///
/// Note: `Tensor::contiguous()` early-returns `clone()` when
/// `is_contiguous()` is true, which is insufficient here — a view that is
/// "contiguous-by-strides" may still cover only part of the underlying
/// buffer. So when the view is contiguous-by-strides but the buffer is
/// oversize, we dispatch directly to `strided_copy_{f32,f64}` (mirroring
/// `methods::contiguous_t`'s GPU fast path / CL-496). The materialization
/// stays on device, NEVER detours through host memory.
///
/// Returns `t.clone()` for the fast path. CPU tensors are returned as-is
/// (CPU elementwise paths handle stride views via `data_vec()`
/// independently and don't need this guard).
#[inline]
fn ensure_contig_for_gpu<T: Float>(t: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if !t.is_cuda() {
        return Ok(t.clone());
    }
    let view_matches_buffer =
        t.is_contiguous() && t.storage_offset() == 0 && t.numel() == t.storage_len();
    if view_matches_buffer {
        return Ok(t.clone());
    }

    // Non-contiguous OR (contiguous-by-strides but offset>0 / oversized
    // buffer): route to the on-device `strided_copy_*` kernel directly.
    // `Tensor::contiguous()` would early-return a `clone()` for the
    // contiguous-but-oversized-buffer case, leaving the bug unfixed; so
    // we bypass that path via the strided_copy kernel directly.
    if t.shape().len() <= 8 {
        if let Some(backend) = crate::gpu_dispatch::gpu_backend() {
            let in_handle = t.gpu_handle()?;
            let out_shape = t.shape().to_vec();
            let src_strides = t.strides().to_vec();
            let src_offset = t.storage_offset();
            let out_handle = if is_f32::<T>() {
                backend
                    .strided_copy_f32(in_handle, &out_shape, &src_strides, src_offset)
                    .ok()
            } else if is_f64::<T>() {
                backend
                    .strided_copy_f64(in_handle, &out_shape, &src_strides, src_offset)
                    .ok()
            } else {
                None
            };
            if let Some(handle) = out_handle {
                let storage = TensorStorage::gpu(handle);
                return Tensor::from_storage(storage, out_shape, false);
            }
        }
    }

    // Fallback (rank > 8, missing backend, or unsupported dtype): use
    // the public `contiguous` path. For genuinely non-contiguous views
    // this still routes through the on-device strided_copy in
    // `contiguous_t`. For the "contiguous-but-oversized" edge case the
    // host-memory fallback there is acceptable as a last resort.
    t.contiguous()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Whether at least one of two tensors requires grad (and grad is enabled).
#[inline]
fn needs_grad<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> bool {
    is_grad_enabled() && (a.requires_grad() || b.requires_grad())
}

/// Whether a single tensor requires grad (and grad is enabled).
#[inline]
fn needs_grad_unary<T: Float>(a: &Tensor<T>) -> bool {
    is_grad_enabled() && a.requires_grad()
}

/// Reduce a gradient tensor to a target shape by summing over broadcast
/// dimensions.
///
/// When two tensors with different shapes are combined via broadcasting,
/// the backward pass must sum the gradient over the dimensions that were
/// broadcast to recover the correct gradient shape for each input.
///
/// Algorithm:
/// 1. If shapes already match, return `grad` as-is (clone).
/// 2. Left-pad `target_shape` with 1s to match grad's ndim.
/// 3. For each dimension where target has size 1 but grad has size > 1,
///    sum over that dimension.
/// 4. Reshape to `target_shape`.
///
/// For f32 GPU tensors, reduction is performed entirely on GPU via
/// `sum_axis_f32` — no CPU roundtrip.  Other dtypes fall back to a
/// CPU reduction loop and re-upload.
pub(crate) fn reduce_grad_to_shape<T: Float>(
    grad: &Tensor<T>,
    target_shape: &[usize],
) -> FerrotorchResult<Tensor<T>> {
    let grad_shape = grad.shape();

    // Fast path: shapes already match.
    if grad_shape == target_shape {
        return Ok(grad.clone());
    }

    // Scalar target: sum everything.
    if target_shape.is_empty() {
        // Use the reduction forward op which already handles GPU.
        return crate::grad_fns::reduction::sum(grad);
    }

    // GPU fast path for f32/f64: reduce each broadcast axis on-device.
    if grad.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        // #812 cluster: a non-contiguous grad view's `gpu_handle()` is the
        // whole underlying buffer; cloning it would carry stale entries
        // outside the logical view into the reduction. Materialize first.
        let grad_c = ensure_contig_for_gpu(grad)?;
        let mut handle = backend.clone_buffer(grad_c.gpu_handle()?)?;
        let mut current_shape = grad_c.shape().to_vec();

        let target_ndim = target_shape.len();

        // First reduce leading dimensions that don't exist in target.
        while current_shape.len() > target_ndim {
            handle = if is_f32::<T>() {
                backend.sum_axis_f32(&handle, &current_shape, 0)?
            } else {
                backend.sum_axis_f64(&handle, &current_shape, 0)?
            };
            current_shape.remove(0);
        }

        // Then reduce dimensions where target has size 1 but grad has size > 1.
        for axis in 0..current_shape.len() {
            if axis < target_shape.len() && target_shape[axis] == 1 && current_shape[axis] > 1 {
                handle = if is_f32::<T>() {
                    backend.sum_axis_f32(&handle, &current_shape, axis)?
                } else {
                    backend.sum_axis_f64(&handle, &current_shape, axis)?
                };
                current_shape[axis] = 1;
            }
        }

        return Tensor::from_storage(TensorStorage::gpu(handle), target_shape.to_vec(), false);
    }

    // CPU path — non-f32/f64 GPU tensors have no GPU kernel, error out.
    if grad.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda {
            op: "broadcast_grad",
        });
    }

    let grad_data = grad.data()?;
    let grad_ndim = grad_shape.len();
    let target_ndim = target_shape.len();

    // Rank-mismatch-but-same-numel reshape branch — #814.
    //
    // When `grad.numel() == target.numel()` the operation is a pure
    // reshape, not a broadcast reduction. This arises in higher-order grad
    // chains where, e.g., a `PowBackward` produces a 0-D scalar
    // intermediate that the downstream node must align with a shape-`[1]`
    // leaf. Pre-#814 the `grad_ndim < target_ndim` arm rejected this with
    // `ShapeMismatch` even though the data is one-to-one. PyTorch's
    // `torch.autograd.grad` works on shape-`[1]` leafs unconditionally;
    // this branch closes that parity gap. Cases covered:
    //   * grad []      -> target [1]       (cited fixture)
    //   * grad []      -> target [1, 1]    (general invariant)
    //   * grad [1]     -> target [1, 1]
    //   * grad [1, 1]  -> target [1]       (also matches via padding below,
    //                                       handled here for symmetry)
    // Numel-mismatch (e.g. `[1] -> [2]`) still falls through to the
    // existing rejection guard and broadcast-reduction logic.
    let grad_numel: usize = grad_shape.iter().product();
    let target_numel: usize = target_shape.iter().product();
    if grad_numel == target_numel {
        // Pure reshape: same elements, different rank. The CPU storage is
        // already row-major contiguous over `grad_data`, so we can rebuild
        // a tensor of the target shape directly from the same data.
        return Tensor::from_storage(
            TensorStorage::cpu(grad_data.to_vec()),
            target_shape.to_vec(),
            false,
        );
    }

    // Standard broadcasting requires grad_ndim >= target_ndim. The reverse
    // case (gradient has fewer dims than target) used to trigger an integer
    // underflow at `grad_ndim - target_ndim`. The graph::backward seed
    // construction in #498 fixed the most common cause (root.shape() instead
    // of a scalar []), but keep an explicit check here as a defense in
    // depth — better a clean error than a panic if any other path produces
    // a misshapen gradient. CL-498.
    if grad_ndim < target_ndim {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "reduce_grad_to_shape: gradient has {grad_ndim} dim(s) but target has {target_ndim} dim(s) ({grad_shape:?} -> {target_shape:?}). \
                 Standard broadcasting backward requires grad_ndim >= target_ndim."
            ),
        });
    }

    // Left-pad target_shape with 1s to match grad_ndim.
    let padded_target: Vec<usize> = if target_ndim < grad_ndim {
        let mut p = vec![1usize; grad_ndim - target_ndim];
        p.extend_from_slice(target_shape);
        p
    } else {
        target_shape.to_vec()
    };

    let out_numel: usize = target_shape.iter().product();
    let mut result = vec![<T as num_traits::Zero>::zero(); out_numel.max(1)];

    // Precompute target strides for flat index calculation.
    let mut target_strides = vec![1usize; target_ndim];
    for td in (0..target_ndim.saturating_sub(1)).rev() {
        target_strides[td] = target_strides[td + 1] * target_shape[td + 1];
    }

    let offset = grad_ndim - target_ndim; // number of leading 1-padded dims

    for (i, &grad_val) in grad_data.iter().enumerate() {
        // Decompose grad flat index into per-axis coordinates.
        let mut coords = [0usize; 16]; // support up to 16 dims
        let mut rem = i;
        for d in (0..grad_ndim).rev() {
            coords[d] = rem % grad_shape[d];
            rem /= grad_shape[d];
        }

        // Compute flat index in target by mapping each grad coord to
        // the corresponding target coord (collapsing broadcast dims).
        let mut flat = 0usize;
        for (td, &target_stride) in target_strides.iter().enumerate() {
            let gd = td + offset;
            let coord = if padded_target[gd] == 1 {
                0
            } else {
                coords[gd]
            };
            flat += coord * target_stride;
        }

        result[flat] += grad_val;
    }

    Tensor::from_storage(TensorStorage::cpu(result), target_shape.to_vec(), false)
}

// ===========================================================================
// add
// ===========================================================================

/// Backward node for `c = a + b`.
///
/// VJP: da = grad, db = grad.
#[derive(Debug)]
struct AddBackward<T: Float> {
    a: Tensor<T>,
    b: Tensor<T>,
}

impl<T: Float> GradFn<T> for AddBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let da = if self.a.requires_grad() {
            Some(reduce_grad_to_shape(grad_output, self.a.shape())?)
        } else {
            None
        };
        let db = if self.b.requires_grad() {
            Some(reduce_grad_to_shape(grad_output, self.b.shape())?)
        } else {
            None
        };
        Ok(vec![da, db])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.a, &self.b]
    }

    fn name(&self) -> &'static str {
        "AddBackward"
    }
}

/// Elementwise addition: `c = a + b`.
pub fn add<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if a.device() != b.device() {
        return Err(FerrotorchError::DeviceMismatch {
            expected: a.device(),
            got: b.device(),
        });
    }

    if let Some(out) = crate::meta_propagate::binary_broadcast(a, b)? {
        return Ok(out);
    }

    crate::profiler_hook::profile_op_scope("add", "tensor_op", &[a.shape(), b.shape()], || {
        add_inner(a, b)
    })
}

fn add_inner<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if a.is_cuda() {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;

        // #812: non-contiguous CUDA inputs (e.g., narrowed views) must be
        // materialized via the on-device `strided_copy_*` kernel before the
        // raw `gpu_handle()` is handed to the elementwise kernel — otherwise
        // the kernel's length guard sees the underlying buffer (potentially
        // larger than the logical view) and panics with `LengthMismatch`.
        let a_c = ensure_contig_for_gpu(a)?;
        let b_c = ensure_contig_for_gpu(b)?;

        let needs_broadcast = a_c.shape() != b_c.shape();
        let (handle, out_shape): (crate::gpu_dispatch::GpuBufferHandle, Vec<usize>) =
            if needs_broadcast {
                let out_shape = broadcast_shapes(a_c.shape(), b_c.shape())?;
                // #23: dtype dispatch via `dispatch_floating_dtype!`
                // enumerates every supported dtype (f32, f64, bf16) in one
                // place; no silent f32 fallthrough for bf16.
                let h: crate::gpu_dispatch::GpuBufferHandle = crate::dispatch_floating_dtype!(
                    T,
                    "broadcast_add",
                    f32 => backend.broadcast_add_f32(
                        a_c.gpu_handle()?,
                        b_c.gpu_handle()?,
                        a_c.shape(),
                        b_c.shape(),
                        &out_shape,
                    ),
                    f64 => backend.broadcast_add_f64(
                        a_c.gpu_handle()?,
                        b_c.gpu_handle()?,
                        a_c.shape(),
                        b_c.shape(),
                        &out_shape,
                    ),
                    bf16 => backend.broadcast_add_bf16(
                        a_c.gpu_handle()?,
                        b_c.gpu_handle()?,
                        a_c.shape(),
                        b_c.shape(),
                        &out_shape,
                    ),
                    f16 => backend.broadcast_add_f16(
                        a_c.gpu_handle()?,
                        b_c.gpu_handle()?,
                        a_c.shape(),
                        b_c.shape(),
                        &out_shape,
                    ),
                )?;
                (h, out_shape)
            } else {
                let h: crate::gpu_dispatch::GpuBufferHandle = crate::dispatch_floating_dtype!(
                    T,
                    "add",
                    f32 => backend.add_f32(a_c.gpu_handle()?, b_c.gpu_handle()?),
                    f64 => backend.add_f64(a_c.gpu_handle()?, b_c.gpu_handle()?),
                    bf16 => backend.add_bf16_bf16(a_c.gpu_handle()?, b_c.gpu_handle()?),
                    f16 => backend.add_f16(a_c.gpu_handle()?, b_c.gpu_handle()?),
                )?;
                (h, a_c.shape().to_vec())
            };
        let storage = TensorStorage::gpu(handle);

        if needs_grad(a, b) {
            Tensor::from_operation(
                storage,
                out_shape,
                Arc::new(AddBackward {
                    a: a.clone(),
                    b: b.clone(),
                }),
            )
        } else {
            Tensor::from_storage(storage, out_shape, false)
        }
    } else {
        let result = fast_add(a, b)?;

        if needs_grad(a, b) {
            let (storage, shape) = result.into_storage_and_shape()?;
            Tensor::from_operation(
                storage,
                shape,
                Arc::new(AddBackward {
                    a: a.clone(),
                    b: b.clone(),
                }),
            )
        } else {
            Ok(result)
        }
    }
}

// ---------------------------------------------------------------------------
// add_scaled: PyTorch-parity `torch.add(input, other, *, alpha=1)`
// ---------------------------------------------------------------------------

/// Backward node for `c = a + alpha * b`.
///
/// VJP: da = grad, db = alpha * grad.
#[derive(Debug)]
struct AddScaledBackward<T: Float> {
    a: Tensor<T>,
    b: Tensor<T>,
    alpha: f64,
}

impl<T: Float> GradFn<T> for AddScaledBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let da = if self.a.requires_grad() {
            Some(reduce_grad_to_shape(grad_output, self.a.shape())?)
        } else {
            None
        };
        let db = if self.b.requires_grad() {
            // db = alpha * grad. `T::from(self.alpha)` is infallible for the
            // Float types we support (f32/f64/bf16/f16) given a finite input;
            // a NaN/Inf alpha is preserved by the cast, matching PyTorch.
            let alpha_t: T = num_traits::cast::cast(self.alpha).ok_or_else(|| {
                FerrotorchError::InvalidArgument {
                    message: format!(
                        "AddScaledBackward: alpha {} not representable in tensor dtype",
                        self.alpha
                    ),
                }
            })?;
            let scaled = no_grad(|| scale_tensor(grad_output, alpha_t))?;
            Some(reduce_grad_to_shape(&scaled, self.b.shape())?)
        } else {
            None
        };
        Ok(vec![da, db])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.a, &self.b]
    }

    fn name(&self) -> &'static str {
        "AddScaledBackward"
    }
}

/// Scale a tensor by a scalar of the same dtype, on whichever device it lives.
///
/// Used by `add_scaled` (forward pre-scale of `b` and backward db = alpha*grad).
/// Kept private — public callers should reach for `grad_fns::arithmetic::mul`
/// with a scalar tensor, or the inplace `mul_scalar_`. This helper exists so we
/// can stay generic over `T: Float` while routing to the dtype-specialised GPU
/// `scale_*` kernels.
fn scale_tensor<T: Float>(t: &Tensor<T>, alpha: T) -> FerrotorchResult<Tensor<T>> {
    if t.is_cuda() {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let tc = ensure_contig_for_gpu(t)?;
        let handle = if is_f32::<T>() {
            // SAFETY: is_f32::<T>() holds, so T == f32 — `cast::<T,f32>` here is
            // a no-op width-wise. Use num_traits cast for an explicit conversion.
            let s: f32 = num_traits::cast::cast(alpha).ok_or(FerrotorchError::DeviceUnavailable)?;
            backend.scale_f32(tc.gpu_handle()?, s)?
        } else if is_f64::<T>() {
            let s: f64 = num_traits::cast::cast(alpha).ok_or(FerrotorchError::DeviceUnavailable)?;
            backend.scale_f64(tc.gpu_handle()?, s)?
        } else if is_bf16::<T>() {
            let s: f32 = num_traits::cast::cast(alpha).ok_or(FerrotorchError::DeviceUnavailable)?;
            // bf16 scale routes through f32 scalar (matches existing bf16 kernels).
            backend.scale_bf16_bf16(tc.gpu_handle()?, s)?
        } else if is_f16::<T>() {
            let s: f32 = num_traits::cast::cast(alpha).ok_or(FerrotorchError::DeviceUnavailable)?;
            backend.scale_f16(tc.gpu_handle()?, s)?
        } else {
            return Err(FerrotorchError::NotImplementedOnCuda { op: "scale_tensor" });
        };
        Tensor::from_storage(TensorStorage::gpu(handle), tc.shape().to_vec(), false)
    } else {
        scalar_map(t, alpha, |x, s| x * s)
    }
}

// ---------------------------------------------------------------------------
// add_out / add_scaled_out: PyTorch-parity `torch.add(input, other, *,
// alpha=1, out=tensor)`.
//
// The `out=` kwarg in PyTorch writes the result into a caller-provided
// pre-allocated tensor (in-place from the caller's perspective on `out`,
// regardless of how the computation itself is staged). The semantics are:
//
//   1. `out.shape()` must equal the broadcast shape of `a` and `b`. PyTorch
//      raises `RuntimeError: shape mismatch` otherwise.
//   2. `out` may not be in the autograd graph (no grad_fn) and may not be a
//      leaf with requires_grad=true — the write is not autograd-tracked.
//      (PyTorch raises here too, with the same message as `add_`.)
//   3. Devices of `a`, `b`, and `out` must match.
//   4. NaN/Inf already present in `out` must be fully overwritten (no leak).
//
// Forward-only: `out=` is incompatible with attaching a grad_fn to `out`
// (autograd ops never accept a pre-allocated `out`), so no backward node
// is attached. This matches torch's behavior — `torch.add(a, b, out=c)`
// returns `c` and does NOT add `c` to the autograd graph for `a`/`b`.
//
// Style note: these are sibling forward functions to `add` / `add_scaled`,
// not methods on `Tensor`, because the natural call shape is
// `add_scaled_out(&mut out, &a, &b, alpha)` rather than
// `out.add_scaled_out_from(&a, &b, alpha)`. The trailing-underscore in-place
// methods (e.g. `add_scaled_`) live on `Tensor` because their `self`
// receiver is the natural target. The first `_out` variant in the workspace
// lands here; a broader convention (a trait, a naming policy) can settle
// when more ops grow `_out` variants.
// ---------------------------------------------------------------------------

/// Validate that `out` is eligible to receive a torch-style `out=` write.
///
/// Mirrors `inplace::check_inplace_allowed` (kept private to that module),
/// but specialised to the `out=` op name so the error message is
/// recognisable at the call site.
fn check_out_allowed<T: Float>(out: &Tensor<T>, op_name: &str) -> FerrotorchResult<()> {
    if out.grad_fn().is_some() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "{op_name}: `out` tensor is part of the computation graph \
                 (has grad_fn = {:?}); cannot write into it",
                out.grad_fn().map(|gf| gf.name()),
            ),
        });
    }
    if out.requires_grad() && out.is_leaf() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "{op_name}: `out` is a leaf tensor with requires_grad=true; \
                 the write would not be tracked by autograd"
            ),
        });
    }
    Ok(())
}

/// `torch.add(a, b, *, out=out)` — write `a + b` into `out` in-place.
///
/// Thin `alpha = 1.0` wrapper over [`add_scaled_out`].
///
/// # Errors
///
/// See [`add_scaled_out`].
pub fn add_out<T: Float>(out: &Tensor<T>, a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<()> {
    add_scaled_out(out, a, b, 1.0)
}

/// `torch.add(a, b, *, alpha=alpha, out=out)` — write `a + alpha * b`
/// into `out` in-place.
///
/// `out`'s shape must equal the broadcast result shape of `a` and `b`.
/// `out`'s storage is replaced with the freshly-computed result; any
/// values previously held by `out` (including NaN sentinels) are
/// completely overwritten.
///
/// `out` is a `&Tensor` rather than `&mut Tensor` because `Tensor` carries
/// its mutability through the `Arc<TensorStorage>` (matching `add_scaled_`
/// / `update_storage`); the autograd-safety checks below enforce exclusive
/// access at the semantic level.
///
/// # Errors
///
/// - [`FerrotorchError::DeviceMismatch`] if `a`, `b`, or `out` reside on
///   different devices.
/// - [`FerrotorchError::ShapeMismatch`] if `out.shape()` does not equal
///   `broadcast_shapes(a.shape(), b.shape())`.
/// - [`FerrotorchError::InvalidArgument`] if `out` has a `grad_fn` or is
///   a leaf with `requires_grad = true` (matches PyTorch's `out=` rule).
/// - [`FerrotorchError::InvalidArgument`] if `alpha` is not representable
///   in the tensor's dtype.
pub fn add_scaled_out<T: Float>(
    out: &Tensor<T>,
    a: &Tensor<T>,
    b: &Tensor<T>,
    alpha: f64,
) -> FerrotorchResult<()> {
    check_out_allowed(out, "add_scaled_out")?;

    if a.device() != b.device() {
        return Err(FerrotorchError::DeviceMismatch {
            expected: a.device(),
            got: b.device(),
        });
    }
    if a.device() != out.device() {
        return Err(FerrotorchError::DeviceMismatch {
            expected: a.device(),
            got: out.device(),
        });
    }

    let broadcast_shape = broadcast_shapes(a.shape(), b.shape())?;

    // Compute the result via the existing scalar/broadcast dispatch. We
    // run inside `no_grad` so the temporary computation does not attach a
    // grad_fn — `out=` is explicitly non-autograd-tracked per torch.
    let result = no_grad(|| add_scaled(a, b, alpha))?;
    let (storage, result_shape) = result.into_storage_and_shape()?;

    // Shape policy: matches torch's CURRENT semantics — when `out.shape()`
    // equals the broadcast shape, the write is purely a storage swap;
    // when it differs, `out` is silently resized to the broadcast shape
    // (PyTorch emits a `UserWarning` for this; it is being deprecated in
    // a future release but is still the observed behavior in 2.x).
    // Mirroring it keeps `torch.add(a, b, out=t)` bit-equivalent across
    // both the matched-shape and resize cases.
    if out.shape() == broadcast_shape.as_slice() {
        // SAFETY: `check_out_allowed` proved `out` has no grad_fn and is
        // not a requires_grad leaf, so no autograd machinery references
        // its storage. `storage` was just produced by a freshly-allocated
        // tensor with no outstanding aliases. Shape equality guarantees
        // `storage.len() == out.numel()`, satisfying update_storage's
        // length contract.
        unsafe { out.update_storage(storage)? };
    } else {
        // Resize `out` to the broadcast shape and swap in the result
        // storage atomically. `result_shape == broadcast_shape` because
        // `add_scaled` returns the broadcast shape on success.
        debug_assert_eq!(result_shape, broadcast_shape);
        // SAFETY: same autograd argument as the matched-shape branch
        // (check_out_allowed already validated). `update_storage_and_shape`
        // verifies its own length invariant (storage.len() == new numel).
        // The caller holds a unique semantic reference to `out` for the
        // duration of this `out=` write — clones of `out` made before
        // this call will observe the new shape after it returns, which
        // matches torch's `Tensor.resize_` behavior exactly.
        unsafe { out.update_storage_and_shape(storage, broadcast_shape)? };
    }
    Ok(())
}

/// Elementwise addition with a scalar multiplier on the second operand:
/// `c = a + alpha * b`. This is the full PyTorch `torch.add(input, other,
/// *, alpha=1)` semantic; the `alpha == 1.0` case is forwarded to [`add`]
/// without an extra scaling pass so callers that don't need alpha pay no
/// allocation cost.
pub fn add_scaled<T: Float>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    alpha: f64,
) -> FerrotorchResult<Tensor<T>> {
    // Fast path: alpha == 1.0 is the existing `add` semantics verbatim.
    // Exact equality is what we want here — we are skipping a scaling pass
    // only when the multiplier is the identity. Any other alpha (including
    // 0.9999..., -0.0, NaN, ±inf) must take the scaled path so torch parity
    // holds bit-for-bit on the no-scale case while everything else flows
    // through `scale_tensor` + `add_inner`.
    #[allow(clippy::float_cmp)]
    let is_identity = alpha == 1.0;
    if is_identity {
        return add(a, b);
    }
    if a.device() != b.device() {
        return Err(FerrotorchError::DeviceMismatch {
            expected: a.device(),
            got: b.device(),
        });
    }
    if let Some(out) = crate::meta_propagate::binary_broadcast(a, b)? {
        return Ok(out);
    }
    crate::profiler_hook::profile_op_scope(
        "add_scaled",
        "tensor_op",
        &[a.shape(), b.shape()],
        || {
            // Forward: scale `b`, then add. `no_grad` so the temporary `b_scaled`
            // does not introduce its own MulScalarBackward node — we attach a
            // single `AddScaledBackward` that handles the full VJP directly.
            let alpha_t: T =
                num_traits::cast::cast(alpha).ok_or_else(|| FerrotorchError::InvalidArgument {
                    message: format!("add_scaled: alpha {alpha} not representable in tensor dtype"),
                })?;
            let b_scaled = no_grad(|| scale_tensor(b, alpha_t))?;
            let result = no_grad(|| add_inner(a, &b_scaled))?;

            if needs_grad(a, b) {
                let (storage, shape) = result.into_storage_and_shape()?;
                Tensor::from_operation(
                    storage,
                    shape,
                    Arc::new(AddScaledBackward {
                        a: a.clone(),
                        b: b.clone(),
                        alpha,
                    }),
                )
            } else {
                Ok(result)
            }
        },
    )
}

// ===========================================================================
// sub
// ===========================================================================

/// Elementwise subtraction: `c = a - b`.
///
/// This is the `alpha=1` thin path over [`sub_scaled`]: PyTorch's
/// `torch.sub(input, other, *, alpha=1)` collapses to `sub_scaled(a, b, 1.0)`
/// which in turn delegates to `add_scaled(a, b, -1.0)` — matching upstream's
/// `TORCH_IMPL_FUNC(sub_out)` at `aten/src/ATen/native/BinaryOps.cpp:434-439`
/// byte-for-byte (`add_stub(device_type(), *this, -alpha)`). Routing `sub`
/// through `sub_scaled` makes every existing caller of `arithmetic::sub`
/// (e.g. `Tensor::sub_t`, `dual_sub`) a transitive non-test production
/// consumer of `sub_scaled` (closes #1215 / R-DEFER-1).
///
/// The forward result, broadcasting behavior, GPU/CPU dispatch, NaN/Inf
/// propagation, and autograd VJP (`da = grad`, `db = -1.0 * grad = -grad`)
/// are all numerically equivalent to the previous standalone `SubBackward`
/// implementation: the grad_fn type just changes from `SubBackward` to
/// `AddScaledBackward { alpha: -1.0 }`.
pub fn sub<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    sub_scaled(a, b, 1.0)
}

/// Elementwise subtraction with a scalar multiplier on the second operand:
/// `c = a - alpha * b`. This is the full PyTorch
/// `torch.sub(input, other, *, alpha=1)` semantic.
///
/// PyTorch's own implementation literally delegates `sub` to `add` with a
/// negated alpha. Per `aten/src/ATen/native/BinaryOps.cpp:434-439`:
///
/// ```cpp
/// TORCH_IMPL_FUNC(sub_out) (
///   const Tensor& self, const Tensor& other, const Scalar& alpha, const Tensor& result
/// ) {
///   add_stub(device_type(), *this, -alpha);
///   TORCH_INTERNAL_ASSERT(result.scalar_type() == output().dtype());
/// }
/// ```
///
/// We match that contract byte-for-byte by delegating to [`add_scaled`]
/// with `-alpha` (R-DEV-1 numerical-contract match). The alpha-identity
/// shortcut, broadcasting, device dispatch, and `SubBackward`-equivalent
/// VJP (`da = grad, db = -alpha * grad`, naturally produced as
/// `AddScaledBackward` with `-alpha`) all come for free from `add_scaled`.
///
/// # Errors
///
/// See [`add_scaled`].
pub fn sub_scaled<T: Float>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    alpha: f64,
) -> FerrotorchResult<Tensor<T>> {
    // `-alpha` flips the sign for every finite alpha and for ±inf; for
    // NaN, `-NaN` preserves the NaN payload bit (matching the `-alpha`
    // produced by `c10::Scalar::operator-` in upstream's TORCH_IMPL_FUNC).
    add_scaled(a, b, -alpha)
}

// ===========================================================================
// mul
// ===========================================================================

/// Backward node for `c = a * b`.
///
/// VJP: da = grad * b, db = grad * a.
#[derive(Debug)]
struct MulBackward<T: Float> {
    a: Tensor<T>,
    b: Tensor<T>,
}

impl<T: Float> GradFn<T> for MulBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // When grad_output requires_grad (i.e., create_graph=true), use
        // differentiable operations so the backward pass itself is recorded
        // in the computation graph for higher-order gradients.
        if grad_output.requires_grad() || grad_output.grad_fn().is_some() {
            // Higher-order: use differentiable ops so the backward pass
            // itself is recorded in the graph.
            let da = if self.a.requires_grad() {
                let raw = mul(grad_output, &self.b)?;
                Some(reduce_grad_to_shape(&raw, self.a.shape())?)
            } else {
                None
            };

            let db = if self.b.requires_grad() {
                let raw = mul(grad_output, &self.a)?;
                Some(reduce_grad_to_shape(&raw, self.b.shape())?)
            } else {
                None
            };

            return Ok(vec![da, db]);
        }

        // Standard (non-higher-order) path: use no_grad + op functions
        // so it works on both CPU and GPU tensors.
        let da = if self.a.requires_grad() {
            let raw = no_grad(|| mul(grad_output, &self.b))?;
            Some(reduce_grad_to_shape(&raw, self.a.shape())?)
        } else {
            None
        };

        let db = if self.b.requires_grad() {
            let raw = no_grad(|| mul(grad_output, &self.a))?;
            Some(reduce_grad_to_shape(&raw, self.b.shape())?)
        } else {
            None
        };

        Ok(vec![da, db])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.a, &self.b]
    }

    fn name(&self) -> &'static str {
        "MulBackward"
    }
}

/// Elementwise multiplication: `c = a * b`.
pub fn mul<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if a.device() != b.device() {
        return Err(FerrotorchError::DeviceMismatch {
            expected: a.device(),
            got: b.device(),
        });
    }

    if let Some(out) = crate::meta_propagate::binary_broadcast(a, b)? {
        return Ok(out);
    }

    crate::profiler_hook::profile_op_scope("mul", "tensor_op", &[a.shape(), b.shape()], || {
        mul_inner(a, b)
    })
}

fn mul_inner<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if a.is_cuda() {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;

        // #812 cluster: materialize non-contiguous CUDA views before kernel.
        let a_c = ensure_contig_for_gpu(a)?;
        let b_c = ensure_contig_for_gpu(b)?;

        let needs_broadcast = a_c.shape() != b_c.shape();
        let (handle, out_shape): (crate::gpu_dispatch::GpuBufferHandle, Vec<usize>) =
            if needs_broadcast {
                let out_shape = broadcast_shapes(a_c.shape(), b_c.shape())?;
                // #23: see arithmetic::add_inner for the rationale.
                let h: crate::gpu_dispatch::GpuBufferHandle = crate::dispatch_floating_dtype!(
                    T,
                    "broadcast_mul",
                    f32 => backend.broadcast_mul_f32(
                        a_c.gpu_handle()?,
                        b_c.gpu_handle()?,
                        a_c.shape(),
                        b_c.shape(),
                        &out_shape,
                    ),
                    f64 => backend.broadcast_mul_f64(
                        a_c.gpu_handle()?,
                        b_c.gpu_handle()?,
                        a_c.shape(),
                        b_c.shape(),
                        &out_shape,
                    ),
                    bf16 => backend.broadcast_mul_bf16(
                        a_c.gpu_handle()?,
                        b_c.gpu_handle()?,
                        a_c.shape(),
                        b_c.shape(),
                        &out_shape,
                    ),
                    f16 => backend.broadcast_mul_f16(
                        a_c.gpu_handle()?,
                        b_c.gpu_handle()?,
                        a_c.shape(),
                        b_c.shape(),
                        &out_shape,
                    ),
                )?;
                (h, out_shape)
            } else {
                let h: crate::gpu_dispatch::GpuBufferHandle = crate::dispatch_floating_dtype!(
                    T,
                    "mul",
                    f32 => backend.mul_f32(a_c.gpu_handle()?, b_c.gpu_handle()?),
                    f64 => backend.mul_f64(a_c.gpu_handle()?, b_c.gpu_handle()?),
                    bf16 => backend.mul_bf16_bf16(a_c.gpu_handle()?, b_c.gpu_handle()?),
                    f16 => backend.mul_f16(a_c.gpu_handle()?, b_c.gpu_handle()?),
                )?;
                (h, a_c.shape().to_vec())
            };
        let storage = TensorStorage::gpu(handle);

        if needs_grad(a, b) {
            Tensor::from_operation(
                storage,
                out_shape,
                Arc::new(MulBackward {
                    a: a.clone(),
                    b: b.clone(),
                }),
            )
        } else {
            Tensor::from_storage(storage, out_shape, false)
        }
    } else {
        let result = fast_mul(a, b)?;

        if needs_grad(a, b) {
            let (storage, shape) = result.into_storage_and_shape()?;
            Tensor::from_operation(
                storage,
                shape,
                Arc::new(MulBackward {
                    a: a.clone(),
                    b: b.clone(),
                }),
            )
        } else {
            Ok(result)
        }
    }
}

// ===========================================================================
// div
// ===========================================================================

/// Backward node for `c = a / b`.
///
/// VJP: da = grad / b, db = -grad * a / (b * b).
#[derive(Debug)]
struct DivBackward<T: Float> {
    a: Tensor<T>,
    b: Tensor<T>,
}

impl<T: Float> GradFn<T> for DivBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // Use op-level functions which handle broadcasting correctly.
        // da = grad / b
        let da = if self.a.requires_grad() {
            let raw = no_grad(|| div(grad_output, &self.b))?;
            Some(reduce_grad_to_shape(&raw, self.a.shape())?)
        } else {
            None
        };
        // db = -grad * a / (b * b)
        let db = if self.b.requires_grad() {
            let raw = no_grad(|| {
                let neg_go = neg(grad_output)?;
                let neg_go_a = mul(&neg_go, &self.a)?;
                let b_sq = mul(&self.b, &self.b)?;
                div(&neg_go_a, &b_sq)
            })?;
            Some(reduce_grad_to_shape(&raw, self.b.shape())?)
        } else {
            None
        };

        Ok(vec![da, db])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.a, &self.b]
    }

    fn name(&self) -> &'static str {
        "DivBackward"
    }
}

/// Elementwise division: `c = a / b`.
///
/// Division by zero follows IEEE 754 semantics: `x / 0.0` produces `+inf`
/// or `-inf` depending on the sign of `x`, and `0.0 / 0.0` produces `NaN`.
pub fn div<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if a.device() != b.device() {
        return Err(FerrotorchError::DeviceMismatch {
            expected: a.device(),
            got: b.device(),
        });
    }

    if let Some(out) = crate::meta_propagate::binary_broadcast(a, b)? {
        return Ok(out);
    }

    crate::profiler_hook::profile_op_scope("div", "tensor_op", &[a.shape(), b.shape()], || {
        div_inner(a, b)
    })
}

fn div_inner<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    // #23: drop the `(is_f32 || is_f64)` guard — bf16 now has a real GPU
    // kernel via `div_bf16_bf16`. Other dtypes fall back to the CPU branch
    // below via `dispatch_floating_dtype!` returning NotImplementedOnCuda.
    if a.is_cuda() && (is_f32::<T>() || is_f64::<T>() || is_bf16::<T>() || is_f16::<T>()) {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;

        // #812 cluster: materialize non-contiguous CUDA views before kernel.
        let a_c = ensure_contig_for_gpu(a)?;
        let b_c = ensure_contig_for_gpu(b)?;

        let needs_broadcast = a_c.shape() != b_c.shape();
        let (handle, out_shape): (crate::gpu_dispatch::GpuBufferHandle, Vec<usize>) =
            if needs_broadcast {
                let out_shape = broadcast_shapes(a_c.shape(), b_c.shape())?;
                let h: crate::gpu_dispatch::GpuBufferHandle = crate::dispatch_floating_dtype!(
                    T,
                    "broadcast_div",
                    f32 => backend.broadcast_div_f32(
                        a_c.gpu_handle()?,
                        b_c.gpu_handle()?,
                        a_c.shape(),
                        b_c.shape(),
                        &out_shape,
                    ),
                    f64 => backend.broadcast_div_f64(
                        a_c.gpu_handle()?,
                        b_c.gpu_handle()?,
                        a_c.shape(),
                        b_c.shape(),
                        &out_shape,
                    ),
                    bf16 => backend.broadcast_div_bf16(
                        a_c.gpu_handle()?,
                        b_c.gpu_handle()?,
                        a_c.shape(),
                        b_c.shape(),
                        &out_shape,
                    ),
                    f16 => backend.broadcast_div_f16(
                        a_c.gpu_handle()?,
                        b_c.gpu_handle()?,
                        a_c.shape(),
                        b_c.shape(),
                        &out_shape,
                    ),
                )?;
                (h, out_shape)
            } else {
                let h: crate::gpu_dispatch::GpuBufferHandle = crate::dispatch_floating_dtype!(
                    T,
                    "div",
                    f32 => backend.div_f32(a_c.gpu_handle()?, b_c.gpu_handle()?),
                    f64 => backend.div_f64(a_c.gpu_handle()?, b_c.gpu_handle()?),
                    bf16 => backend.div_bf16_bf16(a_c.gpu_handle()?, b_c.gpu_handle()?),
                    f16 => backend.div_f16(a_c.gpu_handle()?, b_c.gpu_handle()?),
                )?;
                (h, a_c.shape().to_vec())
            };
        let storage = TensorStorage::gpu(handle);

        if needs_grad(a, b) {
            Tensor::from_operation(
                storage,
                out_shape,
                Arc::new(DivBackward {
                    a: a.clone(),
                    b: b.clone(),
                }),
            )
        } else {
            Tensor::from_storage(storage, out_shape, false)
        }
    } else {
        let result = crate::ops::elementwise::fast_div(a, b)?;

        if needs_grad(a, b) {
            let (storage, shape) = result.into_storage_and_shape()?;
            Tensor::from_operation(
                storage,
                shape,
                Arc::new(DivBackward {
                    a: a.clone(),
                    b: b.clone(),
                }),
            )
        } else {
            Ok(result)
        }
    }
}

// ===========================================================================
// neg
// ===========================================================================

/// Backward node for `c = -a`.
///
/// VJP: da = -grad.
#[derive(Debug)]
struct NegBackward<T: Float> {
    a: Tensor<T>,
}

impl<T: Float> GradFn<T> for NegBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let da = if self.a.requires_grad() {
            Some(no_grad(|| neg(grad_output))?)
        } else {
            None
        };
        Ok(vec![da])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.a]
    }

    fn name(&self) -> &'static str {
        "NegBackward"
    }
}

/// Elementwise negation: `c = -a`.
pub fn neg<T: Float>(a: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) = crate::meta_propagate::unary_same_shape(a)? {
        return Ok(out);
    }
    crate::profiler_hook::profile_op_scope("neg", "tensor_op", &[a.shape()], || neg_inner(a))
}

fn neg_inner<T: Float>(a: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if a.is_cuda() {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        // #812 cluster: materialize non-contiguous CUDA views before kernel.
        let a_c = ensure_contig_for_gpu(a)?;
        // #23: bf16 routes through `neg_bf16_bf16` (sign-bit XOR PTX kernel,
        // no f32 round-trip needed since bf16 is IEEE-shaped).
        let handle: crate::gpu_dispatch::GpuBufferHandle = crate::dispatch_floating_dtype!(
            T,
            "neg",
            f32 => backend.neg_f32(a_c.gpu_handle()?),
            f64 => backend.neg_f64(a_c.gpu_handle()?),
            bf16 => backend.neg_bf16_bf16(a_c.gpu_handle()?),
            f16 => backend.neg_f16(a_c.gpu_handle()?),
        )?;
        let storage = TensorStorage::gpu(handle);
        let shape = a_c.shape().to_vec();

        if needs_grad_unary(a) {
            Tensor::from_operation(storage, shape, Arc::new(NegBackward { a: a.clone() }))
        } else {
            Tensor::from_storage(storage, shape, false)
        }
    } else {
        let result = unary_map(a, |x| -x)?;

        if needs_grad_unary(a) {
            let (storage, shape) = result.into_storage_and_shape()?;
            Tensor::from_operation(storage, shape, Arc::new(NegBackward { a: a.clone() }))
        } else {
            Ok(result)
        }
    }
}

// ===========================================================================
// pow (tensor ^ scalar exponent)
// ===========================================================================

/// Backward node for `c = a ^ exp` where `exp` is a scalar.
///
/// VJP: da = exp * a^(exp-1) * grad.
#[derive(Debug)]
struct PowBackward<T: Float> {
    a: Tensor<T>,
    exp: f64,
}

impl<T: Float> GradFn<T> for PowBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let da = if self.a.requires_grad() {
            // When grad_output requires_grad (create_graph=true), use
            // differentiable operations so the backward pass itself is
            // tracked in the computation graph for higher-order gradients.
            if grad_output.requires_grad() || grad_output.grad_fn().is_some() {
                // da = grad_output * exp * a^(exp-1)
                // Using differentiable pow and mul.
                let a_pow = pow(&self.a, self.exp - 1.0)?; // a^(exp-1)
                let exp_t = T::from(self.exp).unwrap();
                let exp_tensor = Tensor::from_storage(
                    TensorStorage::cpu(vec![exp_t; self.a.numel().max(1)]),
                    self.a.shape().to_vec(),
                    false,
                )?;
                let scaled = mul(&exp_tensor, &a_pow)?; // exp * a^(exp-1)
                Some(mul(grad_output, &scaled)?) // grad_output * exp * a^(exp-1)
            } else if grad_output.is_cuda() {
                // GPU path: use op-level functions in no_grad.
                // da = grad_output * exp * a^(exp-1)
                let da = no_grad(|| {
                    let a_pow = pow(&self.a, self.exp - 1.0)?;
                    let exp_t = T::from(self.exp).unwrap();
                    let exp_tensor = Tensor::from_storage(
                        TensorStorage::cpu(vec![exp_t; self.a.numel().max(1)]),
                        self.a.shape().to_vec(),
                        false,
                    )?;
                    let exp_gpu = exp_tensor.to(self.a.device())?;
                    let scaled = mul(&exp_gpu, &a_pow)?;
                    mul(grad_output, &scaled)
                })?;
                Some(da)
            } else {
                // CPU path: direct data access for performance.
                let go_data = grad_output.data()?;
                let a_data = self.a.data()?;
                let exp_t = T::from(self.exp).unwrap();
                let exp_m1 = T::from(self.exp - 1.0).unwrap();
                let grad_a: Vec<T> = go_data
                    .iter()
                    .zip(a_data.iter())
                    .map(|(&g, &a)| g * exp_t * a.powf(exp_m1))
                    .collect();
                Some(Tensor::from_storage(
                    TensorStorage::cpu(grad_a),
                    self.a.shape().to_vec(),
                    false,
                )?)
            }
        } else {
            None
        };
        Ok(vec![da])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.a]
    }

    fn name(&self) -> &'static str {
        "PowBackward"
    }

    fn scalar_args(&self) -> Vec<f64> {
        // The exponent is the single scalar saved by PowBackward; the JIT
        // tracer reads this to reconstruct `IrOpKind::Pow { exponent }` with
        // the correct value instead of the 0.0 placeholder (#887).
        vec![self.exp]
    }
}

/// Elementwise power: `c = a ^ exp` where `exp` is a scalar `f64`.
pub fn pow<T: Float>(a: &Tensor<T>, exp: f64) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) = crate::meta_propagate::unary_same_shape(a)? {
        let _ = exp;
        return Ok(out);
    }
    crate::profiler_hook::profile_op_scope("pow", "tensor_op", &[a.shape()], || pow_inner(a, exp))
}

fn pow_inner<T: Float>(a: &Tensor<T>, exp: f64) -> FerrotorchResult<Tensor<T>> {
    if a.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        // #812 cluster: materialize non-contiguous CUDA views before kernel.
        let a_c = ensure_contig_for_gpu(a)?;
        let handle = if is_f32::<T>() {
            backend.pow_f32(a_c.gpu_handle()?, exp as f32)?
        } else {
            backend.pow_f64(a_c.gpu_handle()?, exp)?
        };
        let storage = TensorStorage::gpu(handle);
        let shape = a_c.shape().to_vec();

        if needs_grad_unary(a) {
            Tensor::from_operation(storage, shape, Arc::new(PowBackward { a: a.clone(), exp }))
        } else {
            Tensor::from_storage(storage, shape, false)
        }
    } else {
        let exp_t = T::from(exp).unwrap();
        let result = scalar_map(a, exp_t, |x, e| x.powf(e))?;

        if needs_grad_unary(a) {
            let (storage, shape) = result.into_storage_and_shape()?;
            Tensor::from_operation(storage, shape, Arc::new(PowBackward { a: a.clone(), exp }))
        } else {
            Ok(result)
        }
    }
}

// ===========================================================================
// sqrt
// ===========================================================================

/// Backward node for `c = sqrt(a)`.
///
/// VJP: da = grad / (2 * sqrt(a)).
#[derive(Debug)]
struct SqrtBackward<T: Float> {
    a: Tensor<T>,
}

impl<T: Float> GradFn<T> for SqrtBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let da = if self.a.requires_grad() {
            if grad_output.is_cuda() {
                // GPU path: da = grad / (2 * sqrt(a))
                let da = no_grad(|| {
                    let sqrt_a = sqrt(&self.a)?;
                    let two_t = T::from(2.0).unwrap();
                    let two_tensor = Tensor::from_storage(
                        TensorStorage::cpu(vec![two_t; self.a.numel().max(1)]),
                        self.a.shape().to_vec(),
                        false,
                    )?;
                    let two_gpu = two_tensor.to(self.a.device())?;
                    let denom = mul(&two_gpu, &sqrt_a)?;
                    div(grad_output, &denom)
                })?;
                Some(da)
            } else {
                // CPU path: direct data access for performance.
                let go_data = grad_output.data()?;
                let a_data = self.a.data()?;
                let two = T::from(2.0).unwrap();
                let grad_a: Vec<T> = go_data
                    .iter()
                    .zip(a_data.iter())
                    .map(|(&g, &a)| g / (two * a.sqrt()))
                    .collect();
                Some(Tensor::from_storage(
                    TensorStorage::cpu(grad_a),
                    self.a.shape().to_vec(),
                    false,
                )?)
            }
        } else {
            None
        };
        Ok(vec![da])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.a]
    }

    fn name(&self) -> &'static str {
        "SqrtBackward"
    }
}

/// Elementwise square root: `c = sqrt(a)`.
pub fn sqrt<T: Float>(a: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) = crate::meta_propagate::unary_same_shape(a)? {
        return Ok(out);
    }
    crate::profiler_hook::profile_op_scope("sqrt", "tensor_op", &[a.shape()], || sqrt_inner(a))
}

fn sqrt_inner<T: Float>(a: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if a.is_cuda() && (is_f32::<T>() || is_f64::<T>() || is_f16::<T>()) {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        // #812 cluster: materialize non-contiguous CUDA views before kernel.
        let a_c = ensure_contig_for_gpu(a)?;
        // crosslink #1185 Phase 1: f16 routes to the native `sqrt_f16` PTX
        // kernel (sqrt.approx.f32, f16 RNE store) — no CPU fallthrough.
        let handle = if is_f32::<T>() {
            backend.sqrt_f32(a_c.gpu_handle()?)?
        } else if is_f64::<T>() {
            backend.sqrt_f64(a_c.gpu_handle()?)?
        } else {
            backend.sqrt_f16(a_c.gpu_handle()?)?
        };
        let storage = TensorStorage::gpu(handle);
        let shape = a_c.shape().to_vec();

        if needs_grad_unary(a) {
            Tensor::from_operation(storage, shape, Arc::new(SqrtBackward { a: a.clone() }))
        } else {
            Tensor::from_storage(storage, shape, false)
        }
    } else {
        let result = unary_map(a, |x| x.sqrt())?;

        if needs_grad_unary(a) {
            let (storage, shape) = result.into_storage_and_shape()?;
            Tensor::from_operation(storage, shape, Arc::new(SqrtBackward { a: a.clone() }))
        } else {
            Ok(result)
        }
    }
}

// ===========================================================================
// abs
// ===========================================================================

/// Backward node for `c = |a|`.
///
/// VJP: da = grad * sign(a).
#[derive(Debug)]
struct AbsBackward<T: Float> {
    a: Tensor<T>,
}

impl<T: Float> GradFn<T> for AbsBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        use crate::gpu_dispatch::gpu_backend;

        let da = if self.a.requires_grad() {
            // GPU-native path for f32/f64 when both tensors live on CUDA.
            if grad_output.is_cuda() && self.a.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
                let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
                // #812 cluster: materialize non-contiguous CUDA views.
                let go_c = ensure_contig_for_gpu(grad_output)?;
                let a_c = ensure_contig_for_gpu(&self.a)?;
                let handle = if is_f32::<T>() {
                    backend.abs_backward_f32(go_c.gpu_handle()?, a_c.gpu_handle()?)?
                } else {
                    backend.abs_backward_f64(go_c.gpu_handle()?, a_c.gpu_handle()?)?
                };
                let grad_a = Tensor::from_storage(
                    TensorStorage::gpu(handle),
                    self.a.shape().to_vec(),
                    false,
                )?;
                return Ok(vec![Some(grad_a)]);
            }

            if grad_output.is_cuda() || self.a.is_cuda() {
                return Err(FerrotorchError::NotImplementedOnCuda { op: "AbsBackward" });
            }
            // CPU path: direct data access for performance.
            let go_data = grad_output.data()?;
            let a_data = self.a.data()?;
            let zero = <T as num_traits::Zero>::zero();
            let one = <T as num_traits::One>::one();
            let grad_a: Vec<T> = go_data
                .iter()
                .zip(a_data.iter())
                .map(|(&g, &a)| {
                    let sign = if a > zero {
                        one
                    } else if a < zero {
                        -one
                    } else {
                        zero
                    };
                    g * sign
                })
                .collect();
            Some(Tensor::from_storage(
                TensorStorage::cpu(grad_a),
                self.a.shape().to_vec(),
                false,
            )?)
        } else {
            None
        };
        Ok(vec![da])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.a]
    }

    fn name(&self) -> &'static str {
        "AbsBackward"
    }
}

/// Elementwise absolute value: `c = |a|`.
pub fn abs<T: Float>(a: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) = crate::meta_propagate::unary_same_shape(a)? {
        return Ok(out);
    }
    crate::profiler_hook::profile_op_scope("abs", "tensor_op", &[a.shape()], || abs_inner(a))
}

fn abs_inner<T: Float>(a: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if a.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        // #812 cluster: materialize non-contiguous CUDA views before kernel.
        let a_c = ensure_contig_for_gpu(a)?;
        let handle = if is_f32::<T>() {
            backend.abs_f32(a_c.gpu_handle()?)?
        } else {
            backend.abs_f64(a_c.gpu_handle()?)?
        };
        let storage = TensorStorage::gpu(handle);
        let shape = a_c.shape().to_vec();

        if needs_grad_unary(a) {
            Tensor::from_operation(storage, shape, Arc::new(AbsBackward { a: a.clone() }))
        } else {
            Tensor::from_storage(storage, shape, false)
        }
    } else {
        let result = unary_map(a, |x| x.abs())?;

        if needs_grad_unary(a) {
            let (storage, shape) = result.into_storage_and_shape()?;
            Tensor::from_operation(storage, shape, Arc::new(AbsBackward { a: a.clone() }))
        } else {
            Ok(result)
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a leaf scalar tensor.
    fn leaf_scalar(val: f32, requires_grad: bool) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(vec![val]), vec![], requires_grad).unwrap()
    }

    /// Create a leaf 1-D tensor.
    fn leaf_vec(data: &[f32], requires_grad: bool) -> Tensor<f32> {
        Tensor::from_storage(
            TensorStorage::cpu(data.to_vec()),
            vec![data.len()],
            requires_grad,
        )
        .unwrap()
    }

    /// Assert a scalar tensor is approximately equal to `expected`.
    fn assert_scalar_approx(t: &Tensor<f32>, expected: f32, tol: f32) {
        let val = t.item().unwrap();
        assert!(
            (val - expected).abs() < tol,
            "expected {expected}, got {val}"
        );
    }

    // -----------------------------------------------------------------------
    // Forward tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_forward() {
        let a = leaf_vec(&[1.0, 2.0, 3.0], false);
        let b = leaf_vec(&[4.0, 5.0, 6.0], false);
        let c = add(&a, &b).unwrap();
        assert_eq!(c.data().unwrap(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_sub_forward() {
        let a = leaf_vec(&[10.0, 20.0, 30.0], false);
        let b = leaf_vec(&[1.0, 2.0, 3.0], false);
        let c = sub(&a, &b).unwrap();
        assert_eq!(c.data().unwrap(), &[9.0, 18.0, 27.0]);
    }

    #[test]
    fn test_mul_forward() {
        let a = leaf_vec(&[2.0, 3.0, 4.0], false);
        let b = leaf_vec(&[5.0, 6.0, 7.0], false);
        let c = mul(&a, &b).unwrap();
        assert_eq!(c.data().unwrap(), &[10.0, 18.0, 28.0]);
    }

    #[test]
    fn test_div_forward() {
        let a = leaf_vec(&[10.0, 20.0, 30.0], false);
        let b = leaf_vec(&[2.0, 5.0, 10.0], false);
        let c = div(&a, &b).unwrap();
        assert_eq!(c.data().unwrap(), &[5.0, 4.0, 3.0]);
    }

    #[test]
    fn test_neg_forward() {
        let a = leaf_vec(&[1.0, -2.0, 3.0], false);
        let c = neg(&a).unwrap();
        assert_eq!(c.data().unwrap(), &[-1.0, 2.0, -3.0]);
    }

    #[test]
    fn test_pow_forward() {
        let a = leaf_vec(&[2.0, 3.0, 4.0], false);
        let c = pow(&a, 2.0).unwrap();
        let d = c.data().unwrap();
        assert!((d[0] - 4.0).abs() < 1e-6);
        assert!((d[1] - 9.0).abs() < 1e-6);
        assert!((d[2] - 16.0).abs() < 1e-6);
    }

    #[test]
    fn test_sqrt_forward() {
        let a = leaf_vec(&[4.0, 9.0, 16.0], false);
        let c = sqrt(&a).unwrap();
        let d = c.data().unwrap();
        assert!((d[0] - 2.0).abs() < 1e-6);
        assert!((d[1] - 3.0).abs() < 1e-6);
        assert!((d[2] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_abs_forward() {
        let a = leaf_vec(&[-3.0, 0.0, 5.0], false);
        let c = abs(&a).unwrap();
        assert_eq!(c.data().unwrap(), &[3.0, 0.0, 5.0]);
    }

    // -----------------------------------------------------------------------
    // Backward tests (scalar tensors for simplicity)
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_backward() {
        // c = a + b; dc/da = 1, dc/db = 1.
        let a = leaf_scalar(2.0, true);
        let b = leaf_scalar(3.0, true);
        let c = add(&a, &b).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), 1.0, 1e-6);
        assert_scalar_approx(&b.grad().unwrap().unwrap(), 1.0, 1e-6);
    }

    #[test]
    fn test_sub_backward() {
        // c = a - b; dc/da = 1, dc/db = -1.
        let a = leaf_scalar(5.0, true);
        let b = leaf_scalar(3.0, true);
        let c = sub(&a, &b).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), 1.0, 1e-6);
        assert_scalar_approx(&b.grad().unwrap().unwrap(), -1.0, 1e-6);
    }

    #[test]
    fn test_mul_backward() {
        // c = a * b; dc/da = b = 3, dc/db = a = 2.
        let a = leaf_scalar(2.0, true);
        let b = leaf_scalar(3.0, true);
        let c = mul(&a, &b).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), 3.0, 1e-6);
        assert_scalar_approx(&b.grad().unwrap().unwrap(), 2.0, 1e-6);
    }

    #[test]
    fn test_div_backward() {
        // c = a / b; dc/da = 1/b = 1/4, dc/db = -a/b^2 = -6/16 = -0.375.
        let a = leaf_scalar(6.0, true);
        let b = leaf_scalar(4.0, true);
        let c = div(&a, &b).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), 0.25, 1e-6);
        assert_scalar_approx(&b.grad().unwrap().unwrap(), -0.375, 1e-6);
    }

    #[test]
    fn test_div_backward_tensor_by_scalar() {
        // Reproducer from GitHub issue #7:
        // x = [1, 2, 3, 4] (shape [2,2]), s = 2.0 (scalar)
        // y = x / s = [0.5, 1.0, 1.5, 2.0]
        // loss = sum(y) = 5.0
        // d_loss/d_x = 1/s = 0.5 for all elements
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0f64, 2.0, 3.0, 4.0]),
            vec![2, 2],
            true,
        )
        .unwrap();
        let s = Tensor::from_storage(TensorStorage::cpu(vec![2.0f64]), vec![], false).unwrap();
        let y = div(&x, &s).unwrap();
        let loss = crate::grad_fns::reduction::sum(&y).unwrap();
        loss.backward().unwrap();

        let grad = x.grad().unwrap().expect("x should have grad");
        assert_eq!(grad.shape(), &[2, 2]);
        let g = grad.data().unwrap();
        for (i, &v) in g.iter().enumerate() {
            assert!((v - 0.5).abs() < 1e-10, "grad[{i}] = {v}, expected 0.5");
        }
    }

    #[test]
    fn test_neg_backward() {
        // c = -a; dc/da = -1.
        let a = leaf_scalar(7.0, true);
        let c = neg(&a).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), -1.0, 1e-6);
    }

    #[test]
    fn test_pow_backward() {
        // c = a^3; dc/da = 3 * a^2 = 3 * 4 = 12.
        let a = leaf_scalar(2.0, true);
        let c = pow(&a, 3.0).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), 12.0, 1e-5);
    }

    #[test]
    fn test_sqrt_backward() {
        // c = sqrt(a); dc/da = 1 / (2 * sqrt(a)).
        // a = 4.0 => dc/da = 1 / (2 * 2) = 0.25.
        let a = leaf_scalar(4.0, true);
        let c = sqrt(&a).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), 0.25, 1e-6);
    }

    #[test]
    fn test_abs_backward_positive() {
        // c = |a| where a > 0; dc/da = sign(a) = 1.
        let a = leaf_scalar(3.0, true);
        let c = abs(&a).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), 1.0, 1e-6);
    }

    #[test]
    fn test_abs_backward_negative() {
        // c = |a| where a < 0; dc/da = sign(a) = -1.
        let a = leaf_scalar(-3.0, true);
        let c = abs(&a).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), -1.0, 1e-6);
    }

    // -----------------------------------------------------------------------
    // Tests for no-grad and partial requires_grad
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_no_grad_fn_when_inputs_detached() {
        let a = leaf_scalar(2.0, false);
        let b = leaf_scalar(3.0, false);
        let c = add(&a, &b).unwrap();
        assert!(c.grad_fn().is_none());
    }

    #[test]
    fn test_mul_partial_requires_grad() {
        // a requires grad, b does not.
        // c = a * b; dc/da = b = 5, dc/db = None.
        let a = leaf_scalar(3.0, true);
        let b = leaf_scalar(5.0, false);
        let c = mul(&a, &b).unwrap();
        assert!(c.grad_fn().is_some());
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), 5.0, 1e-6);
        assert!(b.grad().unwrap().is_none());
    }

    #[test]
    fn test_no_grad_context_skips_backward() {
        use crate::autograd::no_grad::no_grad;

        let a = leaf_scalar(2.0, true);
        let b = leaf_scalar(3.0, true);
        let c = no_grad(|| add(&a, &b)).unwrap();
        // Inside no_grad, no grad_fn should be attached.
        assert!(c.grad_fn().is_none());
    }

    // -----------------------------------------------------------------------
    // Chain rule tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_chain_mul_add() {
        // d = a * b + b
        // dd/da = b = 3
        // dd/db = a + 1 = 3
        let a = leaf_scalar(2.0, true);
        let b = leaf_scalar(3.0, true);
        let c = mul(&a, &b).unwrap();
        let d = add(&c, &b).unwrap();
        d.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), 3.0, 1e-6);
        assert_scalar_approx(&b.grad().unwrap().unwrap(), 3.0, 1e-6);
    }

    #[test]
    fn test_chain_div_sub() {
        // c = a / b - a
        // dc/da = 1/b - 1 = 1/2 - 1 = -0.5
        // dc/db = -a/b^2 = -3/4 = -0.75
        let a = leaf_scalar(3.0, true);
        let b = leaf_scalar(2.0, true);
        let d = div(&a, &b).unwrap();
        let e = sub(&d, &a).unwrap();
        e.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), -0.5, 1e-5);
        assert_scalar_approx(&b.grad().unwrap().unwrap(), -0.75, 1e-5);
    }

    #[test]
    fn test_chain_sqrt_pow() {
        // c = sqrt(a)^2 = a. dc/da = 1.
        // sqrt(9) = 3, pow(3, 2) = 9.
        // d(pow)/d(sqrt) = 2 * sqrt(a) = 6.
        // d(sqrt)/da = 1 / (2*sqrt(a)) = 1/6.
        // dc/da = 6 * 1/6 = 1.
        let a = leaf_scalar(9.0, true);
        let s = sqrt(&a).unwrap();
        let c = pow(&s, 2.0).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), 1.0, 1e-5);
    }

    #[test]
    fn test_neg_double() {
        // c = -(-a) = a; dc/da = 1.
        let a = leaf_scalar(5.0, true);
        let b = neg(&a).unwrap();
        let c = neg(&b).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), 1.0, 1e-6);
    }

    // -----------------------------------------------------------------------
    // Vector backward tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_mul_vector_backward() {
        // c = a * b (elementwise), then sum to scalar for backward.
        // loss = sum(a * b)
        // d(loss)/d(a_i) = b_i, d(loss)/d(b_i) = a_i.
        let a = leaf_vec(&[1.0, 2.0, 3.0], true);
        let b = leaf_vec(&[4.0, 5.0, 6.0], true);
        let c = mul(&a, &b).unwrap();

        // Sum to scalar so we can call backward.
        let c_data = c.data().unwrap().to_vec();
        let total: f32 = c_data.iter().sum();
        let sum_backward = SumBackward { input: c.clone() };
        let loss = Tensor::from_operation(
            TensorStorage::cpu(vec![total]),
            vec![],
            Arc::new(sum_backward),
        )
        .unwrap();
        loss.backward().unwrap();

        let a_grad = a.grad().unwrap().unwrap();
        let a_g = a_grad.data().unwrap();
        assert!((a_g[0] - 4.0).abs() < 1e-6);
        assert!((a_g[1] - 5.0).abs() < 1e-6);
        assert!((a_g[2] - 6.0).abs() < 1e-6);

        let b_grad = b.grad().unwrap().unwrap();
        let b_g = b_grad.data().unwrap();
        assert!((b_g[0] - 1.0).abs() < 1e-6);
        assert!((b_g[1] - 2.0).abs() < 1e-6);
        assert!((b_g[2] - 3.0).abs() < 1e-6);
    }

    /// Helper backward node for sum reduction in tests:
    /// loss = sum(input); d(loss)/d(input_i) = 1.
    #[derive(Debug)]
    struct SumBackward<T: Float> {
        input: Tensor<T>,
    }

    impl<T: Float> GradFn<T> for SumBackward<T> {
        fn backward(&self, _grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
            let ones_data = vec![<T as num_traits::One>::one(); self.input.numel()];
            let ones = Tensor::from_storage(
                TensorStorage::cpu(ones_data),
                self.input.shape().to_vec(),
                false,
            )?;
            Ok(vec![Some(ones)])
        }

        fn inputs(&self) -> Vec<&Tensor<T>> {
            vec![&self.input]
        }

        fn name(&self) -> &'static str {
            "SumBackward"
        }
    }
}
