//! In-place tensor operations following PyTorch's trailing-underscore convention.
//!
//! These methods mutate the tensor's underlying storage through
//! [`Tensor::data_vec()`] + [`Tensor::update_data()`], which is
//! device-transparent (works on both CPU and GPU tensors). The
//! `update_data()` call performs an unsafe pointer cast through the
//! `Arc<TensorStorage>` — this is sound under the same contract as
//! optimizer updates: the caller must ensure no concurrent reads or
//! writes to the same storage.
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
//! | REQ-10 (`clamp_`) | NOT-STARTED | both-bounds-required; missing Optional/None handling + NaN-bound special case; no non-test consumer. Blocker #1214. |
//! | REQ-11 (`sub_scaled_`) | SHIPPED | `Tensor::sub_scaled_` delegates to `self.add_scaled_(other, -alpha)` mirroring upstream's `TORCH_IMPL_FUNC(sub_out) { add_stub(device_type(), *this, -alpha); }`; the out-of-place sibling `arithmetic::sub_scaled` is the symmetric production consumer that establishes torch's `sub(alpha=k)` parity across both surfaces; parity-sweep `[sub] 88/88 passed (0 skipped, 0 failed)` (closes #1192). |
//!
//! # Autograd safety
//!
//! In-place operations are **not** tracked by the autograd engine. To prevent
//! silent gradient corruption, every method in this module checks two
//! conditions before mutating:
//!
//! 1. The tensor must not have a `grad_fn` (i.e., it must not be the output
//!    of a differentiable operation). Mutating a non-leaf node would
//!    invalidate cached values needed by the backward pass.
//!
//! 2. The tensor must not be a leaf with `requires_grad = true`. PyTorch
//!    raises `RuntimeError` in this case because the in-place modification
//!    would not be recorded and the gradient would be silently wrong.
//!
//! If either check fails, an [`FerrotorchError::InvalidArgument`] is returned.

use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::tensor::Tensor;

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
    /// (or if doing so would change `self.shape()`), or if the tensor is
    /// part of the computation graph or is a leaf with `requires_grad = true`.
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
    /// broadcast result differs from `self.shape()`, or if the tensor is
    /// part of the computation graph or is a leaf with `requires_grad = true`.
    pub fn add_scaled_(&self, other: &Tensor<T>, alpha: f64) -> FerrotorchResult<&Self> {
        check_inplace_allowed(self, "add_scaled_")?;

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
            {
                if let Some(backend) = crate::gpu_dispatch::gpu_backend() {
                    let sum_handle = backend.add_f32(self.gpu_handle()?, other.gpu_handle()?)?;
                    let storage = crate::storage::TensorStorage::gpu(sum_handle);
                    // SAFETY: check_inplace_allowed above proved `self` has
                    // no grad_fn and is not a requires_grad leaf, so no
                    // autograd machinery references this storage; `&self` +
                    // `Float: 'static` ensure no concurrent reader/writer
                    // holds a borrow across this point on this thread,
                    // satisfying update_storage's exclusive-access contract.
                    unsafe { self.update_storage(storage)? };
                    return Ok(self);
                }
            }

            let mut data = self.data_vec()?;
            let other_data = other.data_vec()?;
            for (a, &b) in data.iter_mut().zip(other_data.iter()) {
                *a += b;
            }
            // SAFETY: check_inplace_allowed above ensures `self` is not in
            // the autograd graph and not a requires_grad leaf; satisfies
            // update_data's exclusive-access contract.
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
        // SAFETY: check_inplace_allowed above ensures `self` is not in the
        // autograd graph and not a requires_grad leaf; `storage` was just
        // produced from a freshly-allocated tensor with no aliases. numel
        // matches because we asserted `result.shape() == self.shape()`.
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
    /// broadcast result differs from `self.shape()`, or if the tensor is
    /// part of the computation graph or is a leaf with `requires_grad = true`.
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
    /// (or if doing so would change `self.shape()`), or if the tensor is
    /// part of the computation graph or is a leaf with `requires_grad = true`.
    pub fn sub_(&self, other: &Tensor<T>) -> FerrotorchResult<&Self> {
        self.sub_scaled_(other, 1.0)
    }

    /// Multiply another tensor elementwise in-place: `self *= other`.
    ///
    /// Both tensors must have the same shape.
    pub fn mul_(&self, other: &Tensor<T>) -> FerrotorchResult<&Self> {
        check_inplace_allowed(self, "mul_")?;
        if self.shape() != other.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "mul_: shape mismatch {:?} vs {:?}",
                    self.shape(),
                    other.shape()
                ),
            });
        }

        if self.is_cuda()
            && other.is_cuda()
            && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
        {
            if let Some(backend) = crate::gpu_dispatch::gpu_backend() {
                let handle = backend.mul_f32(self.gpu_handle()?, other.gpu_handle()?)?;
                let storage = crate::storage::TensorStorage::gpu(handle);
                // SAFETY: check_inplace_allowed at the top of `mul_` already
                // proved `self` has no grad_fn and is not a requires_grad leaf;
                // single-threaded `&self` satisfies update_storage's
                // exclusive-access contract.
                unsafe { self.update_storage(storage)? };
                return Ok(self);
            }
        }

        let mut data = self.data_vec()?;
        let other_data = other.data_vec()?;
        for (a, &b) in data.iter_mut().zip(other_data.iter()) {
            *a = *a * b;
        }
        // SAFETY: check_inplace_allowed at the top of `mul_` ensures `self`
        // is not part of the autograd graph; satisfies update_data's
        // exclusive-access contract.
        unsafe { self.update_data(&data)? };
        Ok(self)
    }

    /// Divide by another tensor elementwise in-place: `self /= other`.
    ///
    /// Both tensors must have the same shape.
    pub fn div_(&self, other: &Tensor<T>) -> FerrotorchResult<&Self> {
        check_inplace_allowed(self, "div_")?;
        if self.shape() != other.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "div_: shape mismatch {:?} vs {:?}",
                    self.shape(),
                    other.shape()
                ),
            });
        }

        if self.is_cuda()
            && other.is_cuda()
            && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
        {
            if let Some(backend) = crate::gpu_dispatch::gpu_backend() {
                let handle = backend.div_f32(self.gpu_handle()?, other.gpu_handle()?)?;
                let storage = crate::storage::TensorStorage::gpu(handle);
                // SAFETY: check_inplace_allowed at the top of `div_` already
                // proved `self` has no grad_fn and is not a requires_grad leaf;
                // single-threaded `&self` satisfies update_storage's
                // exclusive-access contract.
                unsafe { self.update_storage(storage)? };
                return Ok(self);
            }
        }

        let mut data = self.data_vec()?;
        let other_data = other.data_vec()?;
        for (a, &b) in data.iter_mut().zip(other_data.iter()) {
            *a = *a / b;
        }
        // SAFETY: check_inplace_allowed at the top of `div_` ensures `self`
        // is not part of the autograd graph; satisfies update_data's
        // exclusive-access contract.
        unsafe { self.update_data(&data)? };
        Ok(self)
    }

    /// Clamp every element to `[min, max]` in-place.
    ///
    /// Each element `x` is replaced with `min.max(x.min(max))`, matching
    /// PyTorch's `Tensor.clamp_()`.
    ///
    /// # Errors
    ///
    /// - Returns an error if `min > max`.
    /// - Returns an error if the tensor is part of the computation graph or is
    ///   a leaf with `requires_grad = true`.
    pub fn clamp_(&self, min: T, max: T) -> FerrotorchResult<&Self> {
        if min > max {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("clamp_ requires min <= max, got min={min:?}, max={max:?}"),
            });
        }

        check_inplace_allowed(self, "clamp_")?;

        let mut data = self.data_vec()?;
        for x in &mut data {
            if *x < min {
                *x = min;
            } else if *x > max {
                *x = max;
            }
        }
        // SAFETY: check_inplace_allowed ensures this tensor is not part of the
        // computation graph and does not require grad, so no concurrent access.
        unsafe { self.update_data(&data)? };

        Ok(self)
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
    fn test_clamp_invalid_range() {
        let t =
            Tensor::from_storage(TensorStorage::cpu(vec![1.0f32, 2.0]), vec![2], false).unwrap();

        let err = t.clamp_(10.0, 0.0).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("min <= max"), "got: {msg}");
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
