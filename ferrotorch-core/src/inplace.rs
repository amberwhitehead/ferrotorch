//! In-place tensor operations following PyTorch's trailing-underscore convention.
//!
//! These methods mutate the tensor's underlying storage through
//! [`Tensor::data_vec()`] + [`Tensor::update_data()`], which is
//! device-transparent (works on both CPU and GPU tensors). The
//! `update_data()` / `update_storage()` calls mutate through the storage's
//! interior mutability (`UnsafeCell`-backed, CORE-001 / #1695) — never
//! through a `&mut` manufactured behind the aliased `Arc` — so mutating
//! while clones/views alias the same storage is UB-free for every
//! sequenced access pattern. What each call site discharges of the
//! `update_*` safety contract:
//!
//! - **No conflicting borrow created here:** the fresh values are computed
//!   into an owned `Vec` *before* the write; no `&[T]` into the destination
//!   buffer is live across the `update_*` call within these methods.
//! - **Forwarded to the crate-level documented contract** (module doc of
//!   [`crate::storage`], `Tensor::data()` docs): callers must not *use* a
//!   previously obtained `&[T]` across an in-place op on an aliasing
//!   handle, and cross-thread access requires external synchronization.
//!
//! ## REQ status (per `.design/ferrotorch-core/inplace.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`add_scalar_`) | NOT-STARTED | impl + in-file tests pass, but no non-test production consumer in `ferrotorch-{core,nn,optim,...}/src/**/*.rs`. Blocker #1205. |
//! | REQ-2 (`mul_scalar_`) | NOT-STARTED | impl + tests pass; no non-test consumer. Blocker #1206. |
//! | REQ-3 (`fill_`) | NOT-STARTED | impl + tests pass; natural caller is `ferrotorch-nn::init::constant_` which builds storage directly. Blocker #1207. |
//! | REQ-4 (`zero_`) | NOT-STARTED | delegates to `self.fill_(T::zero())`; no non-test consumer. Blocker #1208. |
//! | REQ-5 (`add_`) | NOT-STARTED | single-line wrapper over `add_scaled_(other, 1.0)`; no non-test consumer (natural caller is `optim::sgd`). Blocker #1209. |
//! | REQ-6 (`add_scaled_`) | NOT-STARTED | load-bearing impl with GPU + CPU + broadcast paths; the only non-test invocation is the parity-sweep runner's dispatch table (test-side per R-DEFER-1). Blocker #1210. |
//! | REQ-7 (`sub_`) | NOT-STARTED | shape-strict, no `alpha` kwarg, no broadcasting. Blocker #1211. |
//! | REQ-8 (`mul_`) | NOT-STARTED | shape-strict; no broadcasting; no non-test consumer. Blocker #1212. |
//! | REQ-9 (`div_`) | NOT-STARTED | shape-strict; missing `rounding_mode` kwarg; no non-test consumer. Blocker #1213. |
//! | REQ-10 (`clamp_`) | NOT-STARTED | `clamp_` delegates to `clamp_opt_`, which supports PyTorch scalar optional-bound semantics, live-torch one-sided NaN behavior, dedicated `clamp_min_`/`clamp_max_`, `min > max`, and f32/f64/f16/bf16 CUDA-resident two-bound/one-sided paths; no non-test consumer. Blocker #1214. |
//! | REQ-11 (`sub_scaled_`) | SHIPPED | `Tensor::sub_scaled_` delegates to `self.add_scaled_(other, -alpha)` mirroring upstream's `TORCH_IMPL_FUNC(sub_out) { add_stub(device_type(), *this, -alpha); }`; the out-of-place sibling `arithmetic::sub_scaled` is the symmetric production consumer that establishes torch's `sub(alpha=k)` parity across both surfaces; parity-sweep `[sub] 88/88 passed (0 skipped, 0 failed)` (closes #1192). |
//!
//! # Autograd safety
//!
//! Scalar/fill/clamp in-place operations are forward-only and reject tensors
//! already participating in autograd. Binary tensor in-place operations follow
//! PyTorch's graph-rebasing behavior for legal mutations: a non-tracking or
//! non-leaf destination combined with a tracking source receives a `grad_fn`
//! and propagates gradients through the old destination value and source.
//! Leaf tensors with `requires_grad = true` are still rejected before mutation,
//! matching PyTorch's leaf in-place rule.

use std::any::TypeId;
use std::sync::Arc;

use crate::autograd::no_grad::{is_grad_enabled, no_grad};
use crate::device::Device;
use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::storage::TensorStorage;
use crate::tensor::{GradFn, Tensor};

/// Validate that an in-place operation is safe to perform on `tensor`.
///
/// Returns `Ok(())` if the tensor is eligible, or an error describing why
/// the operation was rejected.
fn check_inplace_allowed<T: Float>(tensor: &Tensor<T>, op_name: &str) -> FerrotorchResult<()> {
    if tensor.grad_fn().is_some() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "in-place operation '{op_name}' not allowed on a tensor that is \
                 part of the computation graph (has grad_fn = {:?})",
                tensor.grad_fn().map(|gf| gf.name()),
            ),
        });
    }

    if tensor.requires_grad() && tensor.is_leaf() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "in-place operation '{op_name}' not allowed on a leaf tensor \
                 with requires_grad=true (the modification would not be tracked \
                 by autograd)",
            ),
        });
    }

    Ok(())
}

fn check_binary_inplace_allowed<T: Float>(
    tensor: &Tensor<T>,
    op_name: &str,
) -> FerrotorchResult<()> {
    if tensor.requires_grad() && tensor.is_leaf() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "in-place operation '{op_name}' not allowed on a leaf tensor \
                 with requires_grad=true (the modification would not be tracked \
                 by autograd)",
            ),
        });
    }

    Ok(())
}

#[inline]
fn needs_binary_inplace_autograd<T: Float>(lhs: &Tensor<T>, rhs: &Tensor<T>) -> bool {
    is_grad_enabled() && (lhs.requires_grad() || rhs.requires_grad())
}

fn finish_tracked_binary_inplace<'a, T: Float>(
    target: &'a Tensor<T>,
    result: Tensor<T>,
    op_name: &'static str,
) -> FerrotorchResult<&'a Tensor<T>> {
    if result.shape() != target.shape() {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "{op_name}: broadcast result {:?} does not match self.shape() {:?} \
                 — in-place operation cannot resize the target tensor",
                result.shape(),
                target.shape(),
            ),
        });
    }

    let requires_grad = result.requires_grad();
    let is_leaf = result.is_leaf();
    let grad_fn = result.grad_fn();
    let forward_backtrace = result.forward_backtrace();
    let (storage, _shape) = result.into_storage_and_shape()?;
    // SAFETY: caller has already validated in-place eligibility. The result
    // storage has the same shape/device as target and is freshly materialized.
    unsafe { target.update_storage(storage)? };
    target.replace_autograd_metadata(requires_grad, is_leaf, grad_fn, forward_backtrace)?;
    Ok(target)
}

fn zeros_like_grad<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if input.is_cuda() {
        let Device::Cuda(ordinal) = input.device() else {
            unreachable!("input.is_cuda() implies Device::Cuda")
        };
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let handle = backend.alloc_zeros(
            input.numel(),
            <T as crate::dtype::Element>::dtype(),
            ordinal,
        )?;
        Tensor::from_storage(TensorStorage::gpu(handle), input.shape().to_vec(), false)
    } else {
        Tensor::from_storage(
            TensorStorage::cpu(vec![<T as num_traits::Zero>::zero(); input.numel()]),
            input.shape().to_vec(),
            false,
        )
    }
}

fn cuda_div_rounding_forward<T: Float>(
    lhs: &Tensor<T>,
    rhs: &Tensor<T>,
    rounding_mode: &str,
) -> FerrotorchResult<Option<Tensor<T>>> {
    if !(lhs.is_cuda() || rhs.is_cuda()) {
        return Ok(None);
    }
    if lhs.device() != rhs.device() {
        return Err(FerrotorchError::DeviceMismatch {
            expected: lhs.device(),
            got: rhs.device(),
        });
    }
    if !(lhs.is_cuda() && rhs.is_cuda()) {
        return Err(FerrotorchError::DeviceMismatch {
            expected: lhs.device(),
            got: rhs.device(),
        });
    }

    let out_shape = crate::shape::broadcast_shapes(lhs.shape(), rhs.shape())?;
    let backend = crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
    let handle = if is_f32::<T>() {
        if lhs.shape() == rhs.shape() {
            backend.div_rounding_f32(lhs.gpu_handle()?, rhs.gpu_handle()?, rounding_mode)?
        } else {
            backend.broadcast_div_rounding_f32(
                lhs.gpu_handle()?,
                rhs.gpu_handle()?,
                lhs.shape(),
                rhs.shape(),
                &out_shape,
                rounding_mode,
            )?
        }
    } else if is_f64::<T>() {
        if lhs.shape() == rhs.shape() {
            backend.div_rounding_f64(lhs.gpu_handle()?, rhs.gpu_handle()?, rounding_mode)?
        } else {
            backend.broadcast_div_rounding_f64(
                lhs.gpu_handle()?,
                rhs.gpu_handle()?,
                lhs.shape(),
                rhs.shape(),
                &out_shape,
                rounding_mode,
            )?
        }
    } else {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "div_rounding" });
    };

    Tensor::from_storage(TensorStorage::gpu(handle), out_shape, false).map(Some)
}

fn div_rounding_forward<T: Float>(
    lhs: &Tensor<T>,
    rhs: &Tensor<T>,
    rounding_mode: &str,
) -> FerrotorchResult<Tensor<T>> {
    if let Some(result) = cuda_div_rounding_forward(lhs, rhs, rounding_mode)? {
        return Ok(result);
    }

    match rounding_mode {
        "floor" => no_grad(|| crate::grad_fns::arithmetic::floor_divide(lhs, rhs)),
        "trunc" => {
            let result =
                no_grad(|| crate::grad_fns::arithmetic::div(lhs, rhs)).map_err(|e| match e {
                    FerrotorchError::ShapeMismatch { message } => FerrotorchError::ShapeMismatch {
                        message: format!("div_rounding_: {message}"),
                    },
                    other => other,
                })?;
            let data: Vec<T> = result.data_vec()?.into_iter().map(|x| x.trunc()).collect();
            Tensor::from_storage(
                TensorStorage::on_device(data, result.device())?,
                result.shape().to_vec(),
                false,
            )
        }
        other => Err(FerrotorchError::InvalidArgument {
            message: format!(
                "div_rounding_: expected rounding_mode to be one of 'trunc' or 'floor' \
                 but found '{other}'"
            ),
        }),
    }
}

#[derive(Debug)]
struct RoundedDivInplaceBackward<T: Float> {
    lhs: Tensor<T>,
    rhs: Tensor<T>,
}

impl<T: Float> GradFn<T> for RoundedDivInplaceBackward<T> {
    fn backward(&self, _grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let lhs_grad = if self.lhs.requires_grad() {
            Some(zeros_like_grad(&self.lhs)?)
        } else {
            None
        };
        let rhs_grad = if self.rhs.requires_grad() {
            Some(zeros_like_grad(&self.rhs)?)
        } else {
            None
        };
        Ok(vec![lhs_grad, rhs_grad])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.lhs, &self.rhs]
    }

    fn name(&self) -> &'static str {
        "DivBackward2"
    }
}

#[inline]
fn is_f32<T: Float>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f32>()
}

#[inline]
fn is_f64<T: Float>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f64>()
}

#[inline]
fn is_bf16<T: Float>() -> bool {
    TypeId::of::<T>() == TypeId::of::<half::bf16>()
}

#[inline]
fn is_f16<T: Float>() -> bool {
    TypeId::of::<T>() == TypeId::of::<half::f16>()
}

#[inline]
fn clamp_scalar_pair<T: Float>(x: T, min: T, max: T) -> T {
    let mut y = x;
    if y < min {
        y = min;
    }
    if y > max {
        y = max;
    }
    y
}

fn scalar_to_f32<T: Float>(value: T, op_name: &str) -> FerrotorchResult<f32> {
    value
        .to_f32()
        .ok_or_else(|| FerrotorchError::InvalidArgument {
            message: format!("{op_name}: scalar {value:?} cannot be represented as f32"),
        })
}

fn scalar_to_f64<T: Float>(value: T, op_name: &str) -> FerrotorchResult<f64> {
    value
        .to_f64()
        .ok_or_else(|| FerrotorchError::InvalidArgument {
            message: format!("{op_name}: scalar {value:?} cannot be represented as f64"),
        })
}

fn cuda_fill_storage<T: Float>(
    tensor: &Tensor<T>,
    value: T,
    op_name: &'static str,
) -> FerrotorchResult<Option<TensorStorage<T>>> {
    if !tensor.is_cuda() {
        return Ok(None);
    }
    let Device::Cuda(ordinal) = tensor.device() else {
        return Ok(None);
    };
    let backend = crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
    let handle = if is_f32::<T>() {
        backend.fill_f32(tensor.numel(), scalar_to_f32(value, op_name)?, ordinal)?
    } else if is_f64::<T>() {
        backend.fill_f64(tensor.numel(), scalar_to_f64(value, op_name)?, ordinal)?
    } else if is_bf16::<T>() {
        backend.fill_bf16_bf16(tensor.numel(), scalar_to_f32(value, op_name)?, ordinal)?
    } else if is_f16::<T>() {
        backend.fill_f16(tensor.numel(), scalar_to_f32(value, op_name)?, ordinal)?
    } else {
        return Err(FerrotorchError::NotImplementedOnCuda { op: op_name });
    };
    Ok(Some(TensorStorage::gpu(handle)))
}

fn cuda_clamp_pair_storage<T: Float>(
    tensor: &Tensor<T>,
    min: T,
    max: T,
    op_name: &'static str,
) -> FerrotorchResult<Option<TensorStorage<T>>> {
    if !tensor.is_cuda() {
        return Ok(None);
    }
    let backend = crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
    let handle = if is_f32::<T>() {
        backend.clamp_f32(
            tensor.gpu_handle()?,
            scalar_to_f32(min, op_name)?,
            scalar_to_f32(max, op_name)?,
        )?
    } else if is_f64::<T>() {
        backend.clamp_f64(
            tensor.gpu_handle()?,
            scalar_to_f64(min, op_name)?,
            scalar_to_f64(max, op_name)?,
        )?
    } else if is_f16::<T>() {
        backend.clamp_f16(
            tensor.gpu_handle()?,
            scalar_to_f32(min, op_name)?,
            scalar_to_f32(max, op_name)?,
        )?
    } else if is_bf16::<T>() {
        backend.clamp_bf16_bf16(
            tensor.gpu_handle()?,
            scalar_to_f32(min, op_name)?,
            scalar_to_f32(max, op_name)?,
        )?
    } else {
        return Err(FerrotorchError::NotImplementedOnCuda { op: op_name });
    };
    Ok(Some(TensorStorage::gpu(handle)))
}

fn fill_inplace_after_allowed<T: Float>(
    tensor: &Tensor<T>,
    value: T,
    op_name: &'static str,
) -> FerrotorchResult<()> {
    if let Some(storage) = cuda_fill_storage(tensor, value, op_name)? {
        // SAFETY: callers invoke this helper only after `check_inplace_allowed`;
        // storage has exactly tensor.numel() elements on tensor.device().
        unsafe { tensor.update_storage(storage)? };
    } else if !tensor.is_cuda() {
        let data = vec![value; tensor.numel()];
        // SAFETY: callers invoke this helper only after `check_inplace_allowed`;
        // `data` has exactly tensor.numel() elements.
        unsafe { tensor.update_data(&data)? };
    } else {
        return Err(FerrotorchError::NotImplementedOnCuda { op: op_name });
    }
    Ok(())
}

impl<T: Float> Tensor<T> {
    /// Add a scalar to every element in-place: `self += value`.
    ///
    /// Returns `&Self` for method chaining. Follows PyTorch's `Tensor.add_()`
    /// semantics — the trailing underscore denotes mutation.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor is part of the computation graph or is a
    /// leaf with `requires_grad = true`.
    pub fn add_scalar_(&self, value: T) -> FerrotorchResult<&Self> {
        check_inplace_allowed(self, "add_scalar_")?;

        let mut data = self.data_vec()?;
        for x in &mut data {
            *x += value;
        }
        // SAFETY: check_inplace_allowed ensures this tensor is not part of the
        // computation graph and does not require grad, so no concurrent access.
        unsafe { self.update_data(&data)? };

        Ok(self)
    }

    /// Multiply every element by a scalar in-place: `self *= value`.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor is part of the computation graph or is a
    /// leaf with `requires_grad = true`.
    pub fn mul_scalar_(&self, value: T) -> FerrotorchResult<&Self> {
        check_inplace_allowed(self, "mul_scalar_")?;

        let mut data = self.data_vec()?;
        for x in &mut data {
            *x = *x * value;
        }
        // SAFETY: check_inplace_allowed ensures this tensor is not part of the
        // computation graph and does not require grad, so no concurrent access.
        unsafe { self.update_data(&data)? };

        Ok(self)
    }

    /// Fill every element with `value` in-place.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor is part of the computation graph or is a
    /// leaf with `requires_grad = true`.
    pub fn fill_(&self, value: T) -> FerrotorchResult<&Self> {
        check_inplace_allowed(self, "fill_")?;

        let new_data = vec![value; self.numel()];
        // SAFETY: check_inplace_allowed ensures this tensor is not part of the
        // computation graph and does not require grad, so no concurrent access.
        unsafe { self.update_data(&new_data)? };

        Ok(self)
    }

    /// Zero all elements in-place: `self = 0`.
    ///
    /// Equivalent to `self.fill_(T::zero())`.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor is part of the computation graph or is a
    /// leaf with `requires_grad = true`.
    pub fn zero_(&self) -> FerrotorchResult<&Self> {
        self.fill_(<T as num_traits::Zero>::zero())
    }

    /// Add another tensor elementwise in-place: `self += other`.
    ///
    /// Equivalent to PyTorch's `Tensor.add_(other)` — i.e. `add_scaled_`
    /// with `alpha = 1.0`. `other` may be broadcast to `self.shape()` as
    /// long as the broadcast result equals `self.shape()` (PyTorch
    /// invariant for all in-place ops).
    ///
    /// For GPU f32 tensors on the same-shape fast path, uses the GPU add
    /// kernel and swaps the storage (no CPU round-trip).
    ///
    /// # Errors
    ///
    /// Returns an error if `other` cannot be broadcast to `self.shape()`
    /// (or if doing so would change `self.shape()`), or if `self` is a leaf
    /// tensor with `requires_grad = true`.
    pub fn add_(&self, other: &Tensor<T>) -> FerrotorchResult<&Self> {
        self.add_scaled_(other, 1.0)
    }

    /// In-place version of `torch.add(input, other, *, alpha)`:
    /// `self = self + alpha * other`.
    ///
    /// `other` may be broadcast to `self.shape()` (PyTorch parity); the
    /// broadcast result must equal `self.shape()` — an in-place op cannot
    /// change the tensor's shape. The fast same-shape, `alpha == 1.0`
    /// path uses the GPU add kernel directly when applicable; broadcast
    /// or scaled paths route through `grad_fns::arithmetic::add_scaled`
    /// (which itself dispatches CPU/GPU + broadcasting) and swap the
    /// resulting storage in.
    ///
    /// # Errors
    ///
    /// Returns an error if shapes are not broadcast-compatible, if the
    /// broadcast result differs from `self.shape()`, or if `self` is a leaf
    /// tensor with `requires_grad = true`.
    pub fn add_scaled_(&self, other: &Tensor<T>, alpha: f64) -> FerrotorchResult<&Self> {
        check_binary_inplace_allowed(self, "add_scaled_")?;

        if needs_binary_inplace_autograd(self, other) {
            let lhs = self.autograd_snapshot()?;
            let result = crate::grad_fns::arithmetic::add_scaled(&lhs, other, alpha).map_err(
                |e| match e {
                    FerrotorchError::ShapeMismatch { message } => FerrotorchError::ShapeMismatch {
                        message: format!("add_scaled_: {message}"),
                    },
                    other => other,
                },
            )?;
            return finish_tracked_binary_inplace(self, result, "add_scaled_");
        }

        // Same-shape, alpha == 1.0 fast path: keep the GPU storage-swap
        // and SIMD CPU path that the previous `add_` had. Any other shape
        // or alpha goes through the full broadcast/scale dispatch below.
        #[allow(clippy::float_cmp)]
        let is_identity_alpha = alpha == 1.0;
        if is_identity_alpha && self.shape() == other.shape() {
            // GPU f32 fast path.
            if self.is_cuda()
                && other.is_cuda()
                && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
                && let Some(backend) = crate::gpu_dispatch::gpu_backend()
            {
                let sum_handle = backend.add_f32(self.gpu_handle()?, other.gpu_handle()?)?;
                let storage = crate::storage::TensorStorage::gpu(sum_handle);
                // SAFETY: `check_binary_inplace_allowed` rejected requires-grad
                // leaves, this no-grad fast path only runs when no binary
                // autograd rebasing is needed, and the new storage has the same
                // shape as `self`.
                unsafe { self.update_storage(storage)? };
                return Ok(self);
            }

            let mut data = self.data_vec()?;
            let other_data = other.data_vec()?;
            for (a, &b) in data.iter_mut().zip(other_data.iter()) {
                *a += b;
            }
            // SAFETY: `check_binary_inplace_allowed` rejected requires-grad
            // leaves and this fast path only runs when no binary autograd
            // rebasing is needed. `data` has exactly `self.numel()` elements.
            unsafe { self.update_data(&data)? };
            return Ok(self);
        }

        // Broadcast / scaled path. `add_scaled` already handles CPU and GPU,
        // broadcasting via `binary_map` / `broadcast_add_*`, and dtype
        // dispatch. We materialize the result into a fresh tensor, then swap
        // its storage into `self` — but only if the broadcast shape equals
        // `self.shape()` (in-place ops cannot resize `self`).
        let result = crate::grad_fns::arithmetic::add_scaled(self, other, alpha).map_err(|e| {
            // Re-shape errors come out of `broadcast_shapes`; surface them
            // under the `add_scaled_` op name for caller clarity.
            match e {
                FerrotorchError::ShapeMismatch { message } => FerrotorchError::ShapeMismatch {
                    message: format!("add_scaled_: {message}"),
                },
                other => other,
            }
        })?;
        if result.shape() != self.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "add_scaled_: broadcast result {:?} does not match self.shape() {:?} \
                     — in-place add cannot resize the target tensor",
                    result.shape(),
                    self.shape(),
                ),
            });
        }

        // Swap storage. Take the storage out of `result` rather than
        // copying it through CPU. `into_storage_and_shape` consumes the
        // Tensor and yields its TensorStorage.
        let (storage, _shape) = result.into_storage_and_shape()?;
        // SAFETY: `check_binary_inplace_allowed` rejected requires-grad leaves.
        // `storage` was just produced from a freshly-allocated tensor with no
        // aliases and has the same numel because `result.shape() == self.shape()`.
        unsafe { self.update_storage(storage)? };
        Ok(self)
    }

    /// In-place version of `torch.sub(input, other, *, alpha)`:
    /// `self = self - alpha * other`.
    ///
    /// Delegates to [`Tensor::add_scaled_`] with `-alpha`. PyTorch's own
    /// `sub_out` at `aten/src/ATen/native/BinaryOps.cpp:434-439` does the
    /// same: `add_stub(device_type(), *this, -alpha)`. This is the
    /// in-place sibling of [`crate::grad_fns::arithmetic::sub_scaled`]
    /// and the non-test production consumer of that out-of-place entry
    /// point (it invokes `add_scaled_`, which routes through
    /// `arithmetic::add_scaled`; `sub_scaled` is the symmetric forward
    /// caller wired through the parity-sweep `"sub"` dispatch arm).
    ///
    /// `other` may be broadcast to `self.shape()`; the broadcast result
    /// must equal `self.shape()` — an in-place op cannot resize the
    /// target tensor (PyTorch invariant for all `_` ops).
    ///
    /// # Errors
    ///
    /// Returns an error if shapes are not broadcast-compatible, if the
    /// broadcast result differs from `self.shape()`, or if `self` is a leaf
    /// tensor with `requires_grad = true`.
    pub fn sub_scaled_(&self, other: &Tensor<T>, alpha: f64) -> FerrotorchResult<&Self> {
        // PyTorch parity: `sub_out` literally calls `add_stub` with
        // negated alpha. Delegate to `add_scaled_(other, -alpha)` and
        // inherit its broadcast / GPU fast path / shape-strict in-place
        // semantics for free. Errors surface under the `add_scaled_` op
        // name in the error message; that is acceptable since this is
        // a thin alias and the caller's stack trace pinpoints `sub_scaled_`.
        self.add_scaled_(other, -alpha)
    }

    /// Subtract another tensor elementwise in-place: `self -= other`.
    ///
    /// Equivalent to PyTorch's `Tensor.sub_(other)` — i.e. `sub_scaled_`
    /// with `alpha = 1.0`. Mirrors upstream's
    /// `aten/src/ATen/native/BinaryOps.cpp:434-439`
    /// `TORCH_IMPL_FUNC(sub_out) { add_stub(device_type(), *this, -alpha); }`
    /// with `alpha = 1.0`, i.e. `self += -1.0 * other == self -= other`.
    /// Delegating here gives `sub_scaled_` a non-test production consumer
    /// transitively for free (every caller of `sub_` becomes a caller of
    /// `sub_scaled_`), and brings `sub_` to PyTorch parity with the
    /// `sub_(other, *, alpha=1)` docstring at `torch/_tensor_docs.py:5113`
    /// (broadcasting from `add_scaled_` is inherited; in-place ops cannot
    /// resize `self`).
    ///
    /// # Errors
    ///
    /// Returns an error if `other` cannot be broadcast to `self.shape()`
    /// (or if doing so would change `self.shape()`), or if `self` is a leaf
    /// tensor with `requires_grad = true`.
    pub fn sub_(&self, other: &Tensor<T>) -> FerrotorchResult<&Self> {
        self.sub_scaled_(other, 1.0)
    }

    /// Multiply another tensor elementwise in-place: `self *= other`.
    ///
    /// `other` may be broadcast to `self.shape()` (PyTorch parity for
    /// `Tensor.mul_(other)` — `aten/src/ATen/native/BinaryOps.cpp:441
    /// TORCH_IMPL_FUNC(mul_out)` inherits broadcasting via `TensorIterator`);
    /// the broadcast result must equal `self.shape()` — an in-place op
    /// cannot resize the target tensor.
    ///
    /// The same-shape, both-on-CUDA, `T == f32` path takes the GPU `mul_f32`
    /// kernel and swaps the storage (no CPU round-trip). Anything else
    /// (broadcasting or non-f32 or CPU) routes through
    /// `grad_fns::arithmetic::mul` (which itself handles CPU + GPU broadcasting
    /// via `binary_broadcast` / `broadcast_mul_*`) and swaps the resulting
    /// storage in.
    ///
    /// # Errors
    ///
    /// Returns an error if shapes are not broadcast-compatible, if the
    /// broadcast result differs from `self.shape()`, or if `self` is a leaf
    /// tensor with `requires_grad = true`.
    pub fn mul_(&self, other: &Tensor<T>) -> FerrotorchResult<&Self> {
        check_binary_inplace_allowed(self, "mul_")?;

        if needs_binary_inplace_autograd(self, other) {
            let lhs = self.autograd_snapshot()?;
            let result = crate::grad_fns::arithmetic::mul(&lhs, other).map_err(|e| match e {
                FerrotorchError::ShapeMismatch { message } => FerrotorchError::ShapeMismatch {
                    message: format!("mul_: {message}"),
                },
                other => other,
            })?;
            return finish_tracked_binary_inplace(self, result, "mul_");
        }

        // Same-shape fast paths (preserve previous behavior).
        if self.shape() == other.shape() {
            if self.is_cuda()
                && other.is_cuda()
                && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
                && let Some(backend) = crate::gpu_dispatch::gpu_backend()
            {
                let handle = backend.mul_f32(self.gpu_handle()?, other.gpu_handle()?)?;
                let storage = crate::storage::TensorStorage::gpu(handle);
                // SAFETY: `check_binary_inplace_allowed` rejected requires-grad
                // leaves, this no-grad fast path only runs when no binary
                // autograd rebasing is needed, and the new storage has the same
                // shape as `self`.
                unsafe { self.update_storage(storage)? };
                return Ok(self);
            }

            let mut data = self.data_vec()?;
            let other_data = other.data_vec()?;
            for (a, &b) in data.iter_mut().zip(other_data.iter()) {
                *a = *a * b;
            }
            // SAFETY: `check_binary_inplace_allowed` rejected requires-grad
            // leaves and this fast path only runs when no binary autograd
            // rebasing is needed. `data` has exactly `self.numel()` elements.
            unsafe { self.update_data(&data)? };
            return Ok(self);
        }

        // Broadcast path. `arithmetic::mul` handles broadcast shape inference
        // and CPU/GPU dispatch via `meta_propagate::binary_broadcast` and
        // `broadcast_mul_*` kernels. We then check the broadcast result
        // matches `self.shape()` — in-place mul cannot resize the target
        // (PyTorch invariant for all `_` ops).
        let result = crate::grad_fns::arithmetic::mul(self, other).map_err(|e| match e {
            FerrotorchError::ShapeMismatch { message } => FerrotorchError::ShapeMismatch {
                message: format!("mul_: {message}"),
            },
            other => other,
        })?;
        if result.shape() != self.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "mul_: broadcast result {:?} does not match self.shape() {:?} \
                     — in-place mul cannot resize the target tensor",
                    result.shape(),
                    self.shape(),
                ),
            });
        }
        let (storage, _shape) = result.into_storage_and_shape()?;
        // SAFETY: `check_binary_inplace_allowed` rejected requires-grad leaves.
        // `storage` was just produced from a freshly-allocated tensor with no
        // aliases and has the same numel because `result.shape() == self.shape()`.
        unsafe { self.update_storage(storage)? };
        Ok(self)
    }

    /// Divide by another tensor elementwise in-place: `self /= other`.
    ///
    /// `other` may be broadcast to `self.shape()` (PyTorch parity for
    /// `Tensor.div_(other)` — `aten/src/ATen/native/BinaryOps.cpp:447
    /// TORCH_IMPL_FUNC(div_out)` inherits broadcasting via `TensorIterator`);
    /// the broadcast result must equal `self.shape()` — an in-place op
    /// cannot resize the target tensor.
    ///
    /// The same-shape, both-on-CUDA, `T == f32` path takes the GPU `div_f32`
    /// kernel and swaps the storage (no CPU round-trip). Anything else routes
    /// through `grad_fns::arithmetic::div`.
    ///
    /// True-division semantics (PyTorch parity, no rounding). For
    /// floor / trunc rounding modes use [`Tensor::div_rounding_`].
    ///
    /// # Errors
    ///
    /// Returns an error if shapes are not broadcast-compatible, if the
    /// broadcast result differs from `self.shape()`, or if `self` is a leaf
    /// tensor with `requires_grad = true`.
    pub fn div_(&self, other: &Tensor<T>) -> FerrotorchResult<&Self> {
        check_binary_inplace_allowed(self, "div_")?;

        if needs_binary_inplace_autograd(self, other) {
            let lhs = self.autograd_snapshot()?;
            let result = crate::grad_fns::arithmetic::div(&lhs, other).map_err(|e| match e {
                FerrotorchError::ShapeMismatch { message } => FerrotorchError::ShapeMismatch {
                    message: format!("div_: {message}"),
                },
                other => other,
            })?;
            return finish_tracked_binary_inplace(self, result, "div_");
        }

        if self.shape() == other.shape() {
            if self.is_cuda()
                && other.is_cuda()
                && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
                && let Some(backend) = crate::gpu_dispatch::gpu_backend()
            {
                let handle = backend.div_f32(self.gpu_handle()?, other.gpu_handle()?)?;
                let storage = crate::storage::TensorStorage::gpu(handle);
                // SAFETY: `check_binary_inplace_allowed` rejected requires-grad
                // leaves, this no-grad fast path only runs when no binary
                // autograd rebasing is needed, and the new storage has the same
                // shape as `self`.
                unsafe { self.update_storage(storage)? };
                return Ok(self);
            }

            let mut data = self.data_vec()?;
            let other_data = other.data_vec()?;
            for (a, &b) in data.iter_mut().zip(other_data.iter()) {
                *a = *a / b;
            }
            // SAFETY: `check_binary_inplace_allowed` rejected requires-grad
            // leaves and this fast path only runs when no binary autograd
            // rebasing is needed. `data` has exactly `self.numel()` elements.
            unsafe { self.update_data(&data)? };
            return Ok(self);
        }

        // Broadcast path.
        let result = crate::grad_fns::arithmetic::div(self, other).map_err(|e| match e {
            FerrotorchError::ShapeMismatch { message } => FerrotorchError::ShapeMismatch {
                message: format!("div_: {message}"),
            },
            other => other,
        })?;
        if result.shape() != self.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "div_: broadcast result {:?} does not match self.shape() {:?} \
                     — in-place div cannot resize the target tensor",
                    result.shape(),
                    self.shape(),
                ),
            });
        }
        let (storage, _shape) = result.into_storage_and_shape()?;
        // SAFETY: see `mul_` broadcast-path SAFETY.
        unsafe { self.update_storage(storage)? };
        Ok(self)
    }

    /// In-place division with a `rounding_mode` kwarg, mirroring
    /// `torch.Tensor.div_(other, *, rounding_mode=...)` per
    /// `torch/_tensor_docs.py:1746` and `aten/src/ATen/native/BinaryOps.cpp:176`
    /// `TORCH_META_FUNC2(div, Tensor_mode)`.
    ///
    /// Accepted modes:
    ///
    /// - `"trunc"` — `self = (self / other).trunc()` (rounds toward zero).
    /// - `"floor"` — `self = (self / other).floor()` (rounds toward negative infinity).
    ///
    /// For true-division (no rounding), use [`Tensor::div_`] directly. Any other
    /// `mode` string returns `InvalidArgument` matching upstream:
    ///
    /// > `div expected rounding_mode to be one of None, 'trunc', or 'floor' but found '...'`
    /// > (`BinaryOps.cpp:186`)
    ///
    /// Broadcasting follows `div_` semantics — `other` may broadcast to
    /// `self.shape()` and the broadcast result must equal `self.shape()`.
    ///
    /// # Errors
    ///
    /// Returns an error if `mode` is unrecognized, if shapes are not
    /// broadcast-compatible, or if `self` is a leaf tensor with
    /// `requires_grad = true`.
    pub fn div_rounding_(&self, other: &Tensor<T>, rounding_mode: &str) -> FerrotorchResult<&Self> {
        check_binary_inplace_allowed(self, "div_rounding_")?;
        match rounding_mode {
            "trunc" | "floor" => {}
            other_mode => {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "div_rounding_: expected rounding_mode to be one of 'trunc' or 'floor' \
                         but found '{other_mode}'"
                    ),
                });
            }
        }

        let lhs = if needs_binary_inplace_autograd(self, other) {
            Some(self.autograd_snapshot()?)
        } else {
            None
        };
        let dividend = lhs.as_ref().unwrap_or(self);

        let result = div_rounding_forward(dividend, other, rounding_mode).map_err(|e| match e {
            FerrotorchError::ShapeMismatch { message } => FerrotorchError::ShapeMismatch {
                message: format!("div_rounding_: {message}"),
            },
            other => other,
        })?;
        if result.shape() != self.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "div_rounding_: broadcast result {:?} does not match self.shape() {:?} \
                     — in-place div cannot resize the target tensor",
                    result.shape(),
                    self.shape(),
                ),
            });
        }
        let result = if let Some(lhs) = lhs {
            let (storage, shape) = result.into_storage_and_shape()?;
            Tensor::from_operation(
                storage,
                shape,
                Arc::new(RoundedDivInplaceBackward {
                    lhs,
                    rhs: other.clone(),
                }),
            )?
        } else {
            result
        };
        finish_tracked_binary_inplace(self, result, "div_rounding_")
    }

    /// Clamp every element in-place using PyTorch's scalar two-bound kernel.
    ///
    /// Each element `x` is replaced with `min(max(x, min), max)`, matching
    /// PyTorch's `Tensor.clamp_()` scalar path. PyTorch does not reject
    /// `min > max`; that expression naturally returns `max` for every
    /// non-NaN input.
    ///
    /// This is the both-bounds-required overload; for the
    /// `(Option<T>, Option<T>)` overload that mirrors torch's
    /// `clamp_(min=None, max=None)` see [`Tensor::clamp_opt_`].
    ///
    /// # Errors
    ///
    /// - Returns an error if the tensor is part of the computation graph or is
    ///   a leaf with `requires_grad = true`.
    pub fn clamp_(&self, min: T, max: T) -> FerrotorchResult<&Self> {
        self.clamp_opt_(Some(min), Some(max))
    }

    /// Clamp with optional bounds — `Tensor.clamp_(min=None, max=None)` parity.
    ///
    /// Mirrors `torch.Tensor.clamp_(min=None, max=None) -> Tensor` per
    /// `torch/_tensor_docs.py:1141` and the structured kernel
    /// `TORCH_IMPL_FUNC(clamp_out)` at
    /// `aten/src/ATen/native/TensorCompare.cpp:831`. Either bound may be
    /// `None`:
    ///
    /// - `clamp_opt_(Some(lo), Some(hi))` — equivalent to `clamp_(lo, hi)`.
    /// - `clamp_opt_(Some(lo), None)` — `clamp_min_` (lower bound only).
    /// - `clamp_opt_(None, Some(hi))` — `clamp_max_` (upper bound only).
    /// - `clamp_opt_(None, None)` — rejected with `InvalidArgument`
    ///   matching upstream "torch.clamp: At least one of 'min' or 'max' must
    ///   not be None" (`TensorCompare.cpp:106`).
    ///
    /// NaN-bound parity is intentionally split to match live PyTorch:
    /// two-bound `clamp_(min=nan, max=...)` / `clamp_(..., max=nan)` fills
    /// the tensor with NaN, while one-sided `clamp_(min=nan)` or
    /// `clamp_(max=nan)` leaves forward values unchanged. The dedicated
    /// `clamp_min_(nan)` / `clamp_max_(nan)` wrappers fill with NaN.
    ///
    /// Per-element NaN inputs propagate (matching the kernel's
    /// `std::min(std::max(a, min), max)` semantics — when `a` is NaN, both
    /// comparisons evaluate false in this implementation and `a` is left
    /// unchanged, which propagates NaN through).
    ///
    /// # Errors
    ///
    /// - Returns an error if both `min` and `max` are `None`.
    /// - Returns an error if the tensor is part of the computation graph or
    ///   is a leaf with `requires_grad = true`.
    pub fn clamp_opt_(&self, min: Option<T>, max: Option<T>) -> FerrotorchResult<&Self> {
        if min.is_none() && max.is_none() {
            return Err(FerrotorchError::InvalidArgument {
                message: "clamp_opt_: at least one of 'min' or 'max' must not be None".into(),
            });
        }

        check_inplace_allowed(self, "clamp_opt_")?;

        let min_is_nan = min.is_some_and(num_traits::Float::is_nan);
        let max_is_nan = max.is_some_and(num_traits::Float::is_nan);
        if min.is_some() && max.is_some() && (min_is_nan || max_is_nan) {
            let nan = <T as num_traits::Float>::nan();
            fill_inplace_after_allowed(self, nan, "clamp_opt_")?;
            return Ok(self);
        }

        match (min, max) {
            (Some(lo), Some(hi)) => {
                if let Some(storage) = cuda_clamp_pair_storage(self, lo, hi, "clamp_opt_")? {
                    // SAFETY: check_inplace_allowed above ensures `self` is
                    // not in the autograd graph and not a requires_grad leaf;
                    // update_storage enforces same device, same numel, and
                    // view-aware region writes.
                    unsafe { self.update_storage(storage)? };
                    return Ok(self);
                }
                let mut data = self.data_vec()?;
                for x in &mut data {
                    // NaN inputs propagate: `*x < lo` and `*x > hi` are both
                    // false when `*x` is NaN, leaving `*x` unchanged.
                    *x = clamp_scalar_pair(*x, lo, hi);
                }
                // SAFETY: check_inplace_allowed above ensures `self` is not in
                // the autograd graph and not a requires_grad leaf; satisfies
                // update_data's exclusive-access contract.
                unsafe { self.update_data(&data)? };
            }
            (Some(lo), None) => {
                if num_traits::Float::is_nan(lo) {
                    return Ok(self);
                }
                if let Some(storage) = cuda_clamp_pair_storage(
                    self,
                    lo,
                    <T as num_traits::Float>::infinity(),
                    "clamp_opt_",
                )? {
                    // SAFETY: same as the both-bounds CUDA branch above.
                    unsafe { self.update_storage(storage)? };
                    return Ok(self);
                }
                let mut data = self.data_vec()?;
                for x in &mut data {
                    if *x < lo {
                        *x = lo;
                    }
                }
                // SAFETY: same as the both-bounds CPU branch above.
                unsafe { self.update_data(&data)? };
            }
            (None, Some(hi)) => {
                if num_traits::Float::is_nan(hi) {
                    return Ok(self);
                }
                if let Some(storage) = cuda_clamp_pair_storage(
                    self,
                    <T as num_traits::Float>::neg_infinity(),
                    hi,
                    "clamp_opt_",
                )? {
                    // SAFETY: same as the both-bounds CUDA branch above.
                    unsafe { self.update_storage(storage)? };
                    return Ok(self);
                }
                let mut data = self.data_vec()?;
                for x in &mut data {
                    if *x > hi {
                        *x = hi;
                    }
                }
                // SAFETY: same as the both-bounds CPU branch above.
                unsafe { self.update_data(&data)? };
            }
            (None, None) => {
                return Err(FerrotorchError::InvalidArgument {
                    message: "clamp_opt_: at least one of 'min' or 'max' must not be None".into(),
                });
            }
        }
        Ok(self)
    }

    /// In-place scalar `torch.clamp_min_(min)` parity. Unlike
    /// `clamp_opt_(Some(NaN), None)`, PyTorch's dedicated one-sided API fills
    /// the tensor with NaN when `min` is NaN.
    pub fn clamp_min_(&self, min: T) -> FerrotorchResult<&Self> {
        if num_traits::Float::is_nan(min) {
            check_inplace_allowed(self, "clamp_min_")?;
            fill_inplace_after_allowed(self, <T as num_traits::Float>::nan(), "clamp_min_")?;
            Ok(self)
        } else {
            self.clamp_opt_(Some(min), None)
        }
    }

    /// In-place scalar `torch.clamp_max_(max)` parity. Unlike
    /// `clamp_opt_(None, Some(NaN))`, PyTorch's dedicated one-sided API fills
    /// the tensor with NaN when `max` is NaN.
    pub fn clamp_max_(&self, max: T) -> FerrotorchResult<&Self> {
        if num_traits::Float::is_nan(max) {
            check_inplace_allowed(self, "clamp_max_")?;
            fill_inplace_after_allowed(self, <T as num_traits::Float>::nan(), "clamp_max_")?;
            Ok(self)
        } else {
            self.clamp_opt_(None, Some(max))
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::storage::TensorStorage;
    use crate::tensor::Tensor;

    // -----------------------------------------------------------------------
    // add_scalar_
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_scalar_basic() {
        let t = Tensor::from_storage(TensorStorage::cpu(vec![1.0f32, 2.0, 3.0]), vec![3], false)
            .unwrap();

        t.add_scalar_(10.0).unwrap();

        let data = t.data().unwrap();
        assert_eq!(data, &[11.0, 12.0, 13.0]);
    }

    #[test]
    fn test_add_scalar_negative() {
        let t =
            Tensor::from_storage(TensorStorage::cpu(vec![5.0f64, 10.0]), vec![2], false).unwrap();

        t.add_scalar_(-3.0).unwrap();

        let data = t.data().unwrap();
        assert!((data[0] - 2.0).abs() < 1e-10);
        assert!((data[1] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_add_scalar_chaining() {
        let t =
            Tensor::from_storage(TensorStorage::cpu(vec![0.0f32; 4]), vec![2, 2], false).unwrap();

        t.add_scalar_(1.0).unwrap().add_scalar_(2.0).unwrap();

        let data = t.data().unwrap();
        assert_eq!(data, &[3.0, 3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_add_scalar_rejects_requires_grad_leaf() {
        let t =
            Tensor::<f32>::from_storage(TensorStorage::cpu(vec![1.0, 2.0]), vec![2], true).unwrap();

        let err = t.add_scalar_(1.0).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("requires_grad=true"), "got: {msg}");
    }

    // -----------------------------------------------------------------------
    // mul_scalar_
    // -----------------------------------------------------------------------

    #[test]
    fn test_mul_scalar_basic() {
        let t = Tensor::from_storage(TensorStorage::cpu(vec![2.0f32, 3.0, 4.0]), vec![3], false)
            .unwrap();

        t.mul_scalar_(0.5).unwrap();

        let data = t.data().unwrap();
        assert_eq!(data, &[1.0, 1.5, 2.0]);
    }

    #[test]
    fn test_mul_scalar_zero() {
        let t = Tensor::from_storage(
            TensorStorage::cpu(vec![42.0f64, -7.0, 100.0]),
            vec![3],
            false,
        )
        .unwrap();

        t.mul_scalar_(0.0).unwrap();

        let data = t.data().unwrap();
        assert_eq!(data, &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_mul_scalar_rejects_requires_grad_leaf() {
        let t = Tensor::<f32>::from_storage(TensorStorage::cpu(vec![1.0]), vec![1], true).unwrap();

        assert!(t.mul_scalar_(2.0).is_err());
    }

    // -----------------------------------------------------------------------
    // fill_
    // -----------------------------------------------------------------------

    #[test]
    fn test_fill_basic() {
        let t = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0f32, 2.0, 3.0, 4.0]),
            vec![2, 2],
            false,
        )
        .unwrap();

        t.fill_(99.0).unwrap();

        let data = t.data().unwrap();
        assert_eq!(data, &[99.0, 99.0, 99.0, 99.0]);
    }

    #[test]
    // reason: round-trip bit-equality — fill_(42.0) writes the exact bit
    // pattern of 42.0 (no arithmetic), so equality is the correct check.
    #[allow(clippy::float_cmp)]
    fn test_fill_scalar_tensor() {
        let t = Tensor::from_storage(TensorStorage::cpu(vec![0.0f32]), vec![], false).unwrap();

        t.fill_(42.0).unwrap();

        assert_eq!(t.item().unwrap(), 42.0);
    }

    #[test]
    fn test_fill_rejects_requires_grad_leaf() {
        let t =
            Tensor::<f64>::from_storage(TensorStorage::cpu(vec![1.0, 2.0]), vec![2], true).unwrap();

        assert!(t.fill_(0.0).is_err());
    }

    // -----------------------------------------------------------------------
    // zero_
    // -----------------------------------------------------------------------

    #[test]
    fn test_zero_basic() {
        let t = Tensor::from_storage(TensorStorage::cpu(vec![1.0f32, 2.0, 3.0]), vec![3], false)
            .unwrap();

        t.zero_().unwrap();

        let data = t.data().unwrap();
        assert_eq!(data, &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_zero_empty_tensor() {
        let t =
            Tensor::from_storage(TensorStorage::cpu(Vec::<f32>::new()), vec![0], false).unwrap();

        t.zero_().unwrap();

        assert_eq!(t.numel(), 0);
    }

    #[test]
    fn test_zero_rejects_requires_grad_leaf() {
        let t = Tensor::<f32>::from_storage(TensorStorage::cpu(vec![1.0]), vec![1], true).unwrap();

        assert!(t.zero_().is_err());
    }

    // -----------------------------------------------------------------------
    // clamp_
    // -----------------------------------------------------------------------

    #[test]
    fn test_clamp_basic() {
        let t = Tensor::from_storage(
            TensorStorage::cpu(vec![-5.0f32, 0.0, 3.0, 10.0, 100.0]),
            vec![5],
            false,
        )
        .unwrap();

        t.clamp_(0.0, 10.0).unwrap();

        let data = t.data().unwrap();
        assert_eq!(data, &[0.0, 0.0, 3.0, 10.0, 10.0]);
    }

    #[test]
    fn test_clamp_all_within_range() {
        let t = Tensor::from_storage(TensorStorage::cpu(vec![1.0f64, 2.0, 3.0]), vec![3], false)
            .unwrap();

        t.clamp_(0.0, 10.0).unwrap();

        let data = t.data().unwrap();
        assert_eq!(data, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_clamp_single_value_range() {
        let t = Tensor::from_storage(
            TensorStorage::cpu(vec![-1.0f32, 0.0, 1.0, 5.0]),
            vec![4],
            false,
        )
        .unwrap();

        t.clamp_(3.0, 3.0).unwrap();

        let data = t.data().unwrap();
        assert_eq!(data, &[3.0, 3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_clamp_min_greater_than_max_matches_torch_scalar_kernel() {
        let t =
            Tensor::from_storage(TensorStorage::cpu(vec![1.0f32, 2.0]), vec![2], false).unwrap();

        t.clamp_(10.0, 0.0).unwrap();

        let data = t.data().unwrap();
        assert_eq!(data, &[0.0, 0.0]);
    }

    #[test]
    fn test_clamp_nan_bound_fills_all_values_like_torch() {
        let t = Tensor::from_storage(TensorStorage::cpu(vec![1.0f32, 2.0, 3.0]), vec![3], false)
            .unwrap();

        t.clamp_(f32::NAN, 10.0).unwrap();

        let data = t.data().unwrap();
        assert!(
            data.iter().all(|x| x.is_nan()),
            "NaN scalar bound must fill the whole tensor with NaN, got {data:?}"
        );
    }

    #[test]
    fn test_clamp_rejects_requires_grad_leaf() {
        let t =
            Tensor::<f32>::from_storage(TensorStorage::cpu(vec![1.0, 2.0]), vec![2], true).unwrap();

        assert!(t.clamp_(0.0, 1.0).is_err());
    }

    // -----------------------------------------------------------------------
    // Integration: detached tensors are mutable
    // -----------------------------------------------------------------------

    #[test]
    fn test_detached_tensor_allows_inplace() {
        let t = Tensor::from_storage(TensorStorage::cpu(vec![1.0f32, 2.0, 3.0]), vec![3], true)
            .unwrap();

        // Detach drops requires_grad and grad_fn.
        let d = t.detach();
        assert!(!d.requires_grad());

        d.add_scalar_(10.0).unwrap();
        let data = d.data().unwrap();
        assert_eq!(data, &[11.0, 12.0, 13.0]);
    }

    // -----------------------------------------------------------------------
    // Chaining multiple different in-place ops
    // -----------------------------------------------------------------------

    #[test]
    fn test_mixed_inplace_chaining() {
        let t = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0f32, 2.0, 3.0, 4.0]),
            vec![4],
            false,
        )
        .unwrap();

        // (x + 10) * 2, then clamp to [20, 25]
        t.add_scalar_(10.0)
            .unwrap()
            .mul_scalar_(2.0)
            .unwrap()
            .clamp_(20.0, 25.0)
            .unwrap();

        let data = t.data().unwrap();
        // [1+10, 2+10, 3+10, 4+10] = [11, 12, 13, 14]
        // * 2 = [22, 24, 26, 28]
        // clamp [20, 25] = [22, 24, 25, 25]
        assert_eq!(data, &[22.0, 24.0, 25.0, 25.0]);
    }

    // -----------------------------------------------------------------------
    // f64 coverage
    // -----------------------------------------------------------------------

    #[test]
    fn test_inplace_ops_f64() {
        let t = Tensor::from_storage(TensorStorage::cpu(vec![1.0f64, 2.0, 3.0]), vec![3], false)
            .unwrap();

        t.add_scalar_(100.0).unwrap();
        t.mul_scalar_(0.1).unwrap();

        let data = t.data().unwrap();
        assert!((data[0] - 10.1).abs() < 1e-10);
        assert!((data[1] - 10.2).abs() < 1e-10);
        assert!((data[2] - 10.3).abs() < 1e-10);
    }
}
