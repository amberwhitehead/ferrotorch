//! Saved-tensors hooks for memory offloading in autograd.
//!
//! When a `GradFn` saves tensors for the backward pass (e.g., inputs needed
//! to compute gradients), those tensors consume GPU memory for the entire
//! training iteration. **Saved-tensors hooks** let users intercept the
//! save/restore cycle to offload tensors to CPU, compress them, or apply
//! any custom transformation.
//!
//! # Usage
//!
//! ```ignore
//! use ferrotorch_core::autograd::saved_tensors::saved_tensors_hooks;
//!
//! // Offload saved tensors to CPU during forward, reload during backward:
//! saved_tensors_hooks(
//!     |t| t.cpu(),                  // pack: move to CPU
//!     |t| t.to(Device::Cuda(0)),    // unpack: move back to GPU
//!     || {
//!         let y = model.forward(&x)?;
//!         y.backward()
//!     },
//! )?;
//! ```
//!
//! Hooks are thread-local and nestable (inner scopes override outer ones).
//! ## REQ status (per `.design/ferrotorch-core/autograd/saved_tensors.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | `pub type `PackHook`<T>` at `saved_tensors.rs:35`; consumer: stored in `HOOKS_F32` / `HOOKS_F64` at `:42-50`. Existing pub API — boundary-API grandfathering. |
//! | REQ-2 | SHIPPED | `pub type `UnpackHook`<T>` at `saved_tensors.rs:38`; consumer: stored in `HOOKS_F32` / `HOOKS_F64` at `:42-50`. Existing pub API — boundary-API grandfathering. |
//! | REQ-3 | SHIPPED | `pub fn `saved_tensors_hooks`<T, F, R>` at `saved_tensors.rs:59-104`. Existing pub API — boundary-API grandfathering. |
//! | REQ-4 | SHIPPED | `pub fn `pack_saved_tensor`<T: Float>` at `saved_tensors.rs:110-148`. Existing pub API — boundary-API grandfathering. |
//! | REQ-5 | SHIPPED | `pub fn `unpack_saved_tensor`<T: Float>` at `saved_tensors.rs:154-190`. Existing pub API — boundary-API grandfathering. |
//! | REQ-6 | SHIPPED | `pub fn `has_saved_tensor_hooks`` at `saved_tensors.rs:193-195`; consumer: short-circuits at `:114, :129, :155, :172`. |
//! | REQ-7 | SHIPPED | No-hooks identity passthrough at `saved_tensors.rs:125, :142, :167, :184`; consumer: every GradFn save/load cycle without registered hooks. |
//! | REQ-8 | SHIPPED | Restore-prior-on-exit at `saved_tensors.rs:84, :98`; consumer: every nested `saved_tensors_hooks(...)` call. |
//!

use std::cell::RefCell;
use std::sync::Arc;

use crate::dtype::Float;
use crate::error::FerrotorchResult;
use crate::tensor::Tensor;

/// A pack hook transforms a tensor when it is saved for backward.
pub type PackHook<T> = Arc<dyn Fn(Tensor<T>) -> FerrotorchResult<Tensor<T>> + Send + Sync>;

/// An unpack hook transforms a tensor when it is retrieved during backward.
pub type UnpackHook<T> = Arc<dyn Fn(Tensor<T>) -> FerrotorchResult<Tensor<T>> + Send + Sync>;

// Thread-local saved-tensors hook state for f32.
thread_local! {
    static HOOKS_F32: RefCell<Option<(PackHook<f32>, UnpackHook<f32>)>> =
        const { RefCell::new(None) };
}

// Thread-local saved-tensors hook state for f64.
thread_local! {
    static HOOKS_F64: RefCell<Option<(PackHook<f64>, UnpackHook<f64>)>> =
        const { RefCell::new(None) };
}

// Thread-local saved-tensors hook state for bfloat16.
thread_local! {
    static HOOKS_BF16: RefCell<Option<(PackHook<half::bf16>, UnpackHook<half::bf16>)>> =
        const { RefCell::new(None) };
}

// Thread-local saved-tensors hook state for IEEE float16.
thread_local! {
    static HOOKS_F16: RefCell<Option<(PackHook<half::f16>, UnpackHook<half::f16>)>> =
        const { RefCell::new(None) };
}

/// Run a closure with saved-tensors hooks active on the current thread.
///
/// The `pack` hook is called on every tensor saved for backward during `f()`.
/// The `unpack` hook is called when those tensors are accessed in the backward
/// pass. Hooks are restored (or cleared) when this function returns.
///
/// Hooks are nestable — inner calls override outer hooks for their scope.
pub fn saved_tensors_hooks<T, F, R>(
    pack: impl Fn(Tensor<T>) -> FerrotorchResult<Tensor<T>> + Send + Sync + 'static,
    unpack: impl Fn(Tensor<T>) -> FerrotorchResult<Tensor<T>> + Send + Sync + 'static,
    f: F,
) -> FerrotorchResult<R>
where
    T: Float,
    F: FnOnce() -> FerrotorchResult<R>,
{
    let pack = Arc::new(pack) as PackHook<T>;
    let unpack = Arc::new(unpack) as UnpackHook<T>;

    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        // SAFETY: TypeId equality above proves T == f32 at runtime, so
        // PackHook<T> and PackHook<f32> are the same concrete type
        // (Arc<dyn Fn(Tensor<T>) -> ...>). The transmute is a no-op
        // reinterpretation of the Arc's vtable+data pointers; the underlying
        // Fn was authored as Fn(Tensor<T>) and is identical in layout to
        // Fn(Tensor<f32>) under T==f32. No new aliasing — `pack` is moved.
        let pack_f32: PackHook<f32> = unsafe { std::mem::transmute(pack) };
        // SAFETY: same as pack_f32 above (T == f32 by TypeId guard).
        let unpack_f32: UnpackHook<f32> = unsafe { std::mem::transmute(unpack) };

        let prev = HOOKS_F32.with(|h| h.borrow_mut().replace((pack_f32, unpack_f32)));
        let result = f();
        HOOKS_F32.with(|h| *h.borrow_mut() = prev);
        result
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        // SAFETY: TypeId equality above proves T == f64 at runtime, so
        // PackHook<T> and PackHook<f64> are the same concrete type. The
        // transmute is a no-op reinterpretation of the Arc's vtable+data
        // pointers; the underlying Fn(Tensor<T>) is identical in layout to
        // Fn(Tensor<f64>) under T==f64.
        let pack_f64: PackHook<f64> = unsafe { std::mem::transmute(pack) };
        // SAFETY: same as pack_f64 above (T == f64 by TypeId guard).
        let unpack_f64: UnpackHook<f64> = unsafe { std::mem::transmute(unpack) };

        let prev = HOOKS_F64.with(|h| h.borrow_mut().replace((pack_f64, unpack_f64)));
        let result = f();
        HOOKS_F64.with(|h| *h.borrow_mut() = prev);
        result
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<half::bf16>() {
        // SAFETY: TypeId equality above proves T == half::bf16; see the f32
        // arm for the ownership/layout argument.
        let pack_bf16: PackHook<half::bf16> = unsafe { std::mem::transmute(pack) };
        // SAFETY: same as pack_bf16 above (T == half::bf16 by TypeId guard).
        let unpack_bf16: UnpackHook<half::bf16> = unsafe { std::mem::transmute(unpack) };

        let prev = HOOKS_BF16.with(|h| h.borrow_mut().replace((pack_bf16, unpack_bf16)));
        let result = f();
        HOOKS_BF16.with(|h| *h.borrow_mut() = prev);
        result
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<half::f16>() {
        // SAFETY: TypeId equality above proves T == half::f16; see the f32
        // arm for the ownership/layout argument.
        let pack_f16: PackHook<half::f16> = unsafe { std::mem::transmute(pack) };
        // SAFETY: same as pack_f16 above (T == half::f16 by TypeId guard).
        let unpack_f16: UnpackHook<half::f16> = unsafe { std::mem::transmute(unpack) };

        let prev = HOOKS_F16.with(|h| h.borrow_mut().replace((pack_f16, unpack_f16)));
        let result = f();
        HOOKS_F16.with(|h| *h.borrow_mut() = prev);
        result
    } else {
        // No hooks for other types — just run the closure.
        f()
    }
}

/// Capture the currently active saved-tensor hooks for a tensor dtype.
///
/// Production saved tensors must store the unpack hook with the packed value:
/// PyTorch calls the pack hook while constructing the forward node, then calls
/// the matching unpack hook later during backward even after the context manager
/// has exited. This helper is intentionally crate-private; public callers keep
/// using [`saved_tensors_hooks`], [`pack_saved_tensor`], and
/// [`unpack_saved_tensor`].
pub(crate) fn current_saved_tensor_hooks<T: Float>() -> Option<(PackHook<T>, UnpackHook<T>)> {
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        HOOKS_F32.with(|h| {
            let guard = h.borrow();
            guard.as_ref().map(|(pack, unpack)| {
                // SAFETY: TypeId equality above proves T == f32, so the hook
                // trait objects have identical concrete Tensor parameter types.
                // The Arcs are cloned before transmute, preserving ownership.
                let pack_t: PackHook<T> =
                    unsafe { std::mem::transmute::<PackHook<f32>, PackHook<T>>(pack.clone()) };
                // SAFETY: same TypeId guard as above.
                let unpack_t: UnpackHook<T> = unsafe {
                    std::mem::transmute::<UnpackHook<f32>, UnpackHook<T>>(unpack.clone())
                };
                (pack_t, unpack_t)
            })
        })
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        HOOKS_F64.with(|h| {
            let guard = h.borrow();
            guard.as_ref().map(|(pack, unpack)| {
                // SAFETY: TypeId equality above proves T == f64; see f32 arm.
                let pack_t: PackHook<T> =
                    unsafe { std::mem::transmute::<PackHook<f64>, PackHook<T>>(pack.clone()) };
                // SAFETY: same TypeId guard as above.
                let unpack_t: UnpackHook<T> = unsafe {
                    std::mem::transmute::<UnpackHook<f64>, UnpackHook<T>>(unpack.clone())
                };
                (pack_t, unpack_t)
            })
        })
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<half::bf16>() {
        HOOKS_BF16.with(|h| {
            let guard = h.borrow();
            guard.as_ref().map(|(pack, unpack)| {
                // SAFETY: TypeId equality above proves T == half::bf16; see f32 arm.
                let pack_t: PackHook<T> = unsafe {
                    std::mem::transmute::<PackHook<half::bf16>, PackHook<T>>(pack.clone())
                };
                // SAFETY: same TypeId guard as above.
                let unpack_t: UnpackHook<T> = unsafe {
                    std::mem::transmute::<UnpackHook<half::bf16>, UnpackHook<T>>(unpack.clone())
                };
                (pack_t, unpack_t)
            })
        })
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<half::f16>() {
        HOOKS_F16.with(|h| {
            let guard = h.borrow();
            guard.as_ref().map(|(pack, unpack)| {
                // SAFETY: TypeId equality above proves T == half::f16; see f32 arm.
                let pack_t: PackHook<T> = unsafe {
                    std::mem::transmute::<PackHook<half::f16>, PackHook<T>>(pack.clone())
                };
                // SAFETY: same TypeId guard as above.
                let unpack_t: UnpackHook<T> = unsafe {
                    std::mem::transmute::<UnpackHook<half::f16>, UnpackHook<T>>(unpack.clone())
                };
                (pack_t, unpack_t)
            })
        })
    } else {
        None
    }
}

/// Apply the current pack hook to a tensor (if one is active).
///
/// Called by `GradFn` constructors when saving tensors for backward.
/// Returns the tensor unchanged if no hooks are active.
pub fn pack_saved_tensor<T: Float>(tensor: Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        HOOKS_F32.with(|h| {
            let guard = h.borrow();
            if let Some((ref pack, _)) = *guard {
                // SAFETY: TypeId equality above proves T == f32, so Tensor<T>
                // and Tensor<f32> are the same concrete type and the
                // transmute is a no-op layout reinterpret. `tensor` is moved
                // (not aliased) into the new binding.
                let t_f32: Tensor<f32> =
                    unsafe { std::mem::transmute::<Tensor<T>, Tensor<f32>>(tensor) };
                let result = pack(t_f32)?;
                // SAFETY: T == f32 by the same TypeId guard, so Tensor<f32>
                // and Tensor<T> are identical types; transmute is a no-op.
                Ok(unsafe { std::mem::transmute::<Tensor<f32>, Tensor<T>>(result) })
            } else {
                Ok(tensor)
            }
        })
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        HOOKS_F64.with(|h| {
            let guard = h.borrow();
            if let Some((ref pack, _)) = *guard {
                // SAFETY: TypeId equality above proves T == f64, so Tensor<T>
                // and Tensor<f64> are the same concrete type and the
                // transmute is a no-op layout reinterpret. `tensor` is moved.
                let t_f64: Tensor<f64> =
                    unsafe { std::mem::transmute::<Tensor<T>, Tensor<f64>>(tensor) };
                let result = pack(t_f64)?;
                // SAFETY: T == f64 by the same TypeId guard; transmute is a no-op.
                Ok(unsafe { std::mem::transmute::<Tensor<f64>, Tensor<T>>(result) })
            } else {
                Ok(tensor)
            }
        })
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<half::bf16>() {
        HOOKS_BF16.with(|h| {
            let guard = h.borrow();
            if let Some((ref pack, _)) = *guard {
                // SAFETY: TypeId equality above proves T == half::bf16.
                let t_bf16: Tensor<half::bf16> =
                    unsafe { std::mem::transmute::<Tensor<T>, Tensor<half::bf16>>(tensor) };
                let result = pack(t_bf16)?;
                // SAFETY: T == half::bf16 by the same TypeId guard.
                Ok(unsafe { std::mem::transmute::<Tensor<half::bf16>, Tensor<T>>(result) })
            } else {
                Ok(tensor)
            }
        })
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<half::f16>() {
        HOOKS_F16.with(|h| {
            let guard = h.borrow();
            if let Some((ref pack, _)) = *guard {
                // SAFETY: TypeId equality above proves T == half::f16.
                let t_f16: Tensor<half::f16> =
                    unsafe { std::mem::transmute::<Tensor<T>, Tensor<half::f16>>(tensor) };
                let result = pack(t_f16)?;
                // SAFETY: T == half::f16 by the same TypeId guard.
                Ok(unsafe { std::mem::transmute::<Tensor<half::f16>, Tensor<T>>(result) })
            } else {
                Ok(tensor)
            }
        })
    } else {
        Ok(tensor)
    }
}

/// Apply the current unpack hook to a tensor (if one is active).
///
/// Called during backward when a saved tensor is accessed.
/// Returns the tensor unchanged if no hooks are active.
pub fn unpack_saved_tensor<T: Float>(tensor: Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        HOOKS_F32.with(|h| {
            let guard = h.borrow();
            if let Some((_, ref unpack)) = *guard {
                // SAFETY: TypeId equality above proves T == f32, so Tensor<T>
                // and Tensor<f32> are the same concrete type; transmute is a
                // no-op layout reinterpret. `tensor` is moved into t_f32.
                let t_f32: Tensor<f32> =
                    unsafe { std::mem::transmute::<Tensor<T>, Tensor<f32>>(tensor) };
                let result = unpack(t_f32)?;
                // SAFETY: T == f32 by the same TypeId guard; transmute is a no-op.
                Ok(unsafe { std::mem::transmute::<Tensor<f32>, Tensor<T>>(result) })
            } else {
                Ok(tensor)
            }
        })
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        HOOKS_F64.with(|h| {
            let guard = h.borrow();
            if let Some((_, ref unpack)) = *guard {
                // SAFETY: TypeId equality above proves T == f64, so Tensor<T>
                // and Tensor<f64> are the same concrete type; transmute is a
                // no-op layout reinterpret. `tensor` is moved into t_f64.
                let t_f64: Tensor<f64> =
                    unsafe { std::mem::transmute::<Tensor<T>, Tensor<f64>>(tensor) };
                let result = unpack(t_f64)?;
                // SAFETY: T == f64 by the same TypeId guard; transmute is a no-op.
                Ok(unsafe { std::mem::transmute::<Tensor<f64>, Tensor<T>>(result) })
            } else {
                Ok(tensor)
            }
        })
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<half::bf16>() {
        HOOKS_BF16.with(|h| {
            let guard = h.borrow();
            if let Some((_, ref unpack)) = *guard {
                // SAFETY: TypeId equality above proves T == half::bf16.
                let t_bf16: Tensor<half::bf16> =
                    unsafe { std::mem::transmute::<Tensor<T>, Tensor<half::bf16>>(tensor) };
                let result = unpack(t_bf16)?;
                // SAFETY: T == half::bf16 by the same TypeId guard.
                Ok(unsafe { std::mem::transmute::<Tensor<half::bf16>, Tensor<T>>(result) })
            } else {
                Ok(tensor)
            }
        })
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<half::f16>() {
        HOOKS_F16.with(|h| {
            let guard = h.borrow();
            if let Some((_, ref unpack)) = *guard {
                // SAFETY: TypeId equality above proves T == half::f16.
                let t_f16: Tensor<half::f16> =
                    unsafe { std::mem::transmute::<Tensor<T>, Tensor<half::f16>>(tensor) };
                let result = unpack(t_f16)?;
                // SAFETY: T == half::f16 by the same TypeId guard.
                Ok(unsafe { std::mem::transmute::<Tensor<half::f16>, Tensor<T>>(result) })
            } else {
                Ok(tensor)
            }
        })
    } else {
        Ok(tensor)
    }
}

/// Returns `true` if saved-tensors hooks are currently active on this thread.
pub fn has_saved_tensor_hooks() -> bool {
    HOOKS_F32.with(|h| h.borrow().is_some())
        || HOOKS_F64.with(|h| h.borrow().is_some())
        || HOOKS_BF16.with(|h| h.borrow().is_some())
        || HOOKS_F16.with(|h| h.borrow().is_some())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::TensorStorage;

    #[test]
    fn test_pack_unpack_identity() {
        let t = Tensor::from_storage(TensorStorage::cpu(vec![1.0f32, 2.0, 3.0]), vec![3], false)
            .unwrap();

        // No hooks active — pack/unpack are identity.
        let packed = pack_saved_tensor(t.clone()).unwrap();
        assert_eq!(packed.data_vec().unwrap(), vec![1.0, 2.0, 3.0]);

        let unpacked = unpack_saved_tensor(packed).unwrap();
        assert_eq!(unpacked.data_vec().unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_saved_tensors_hooks_transform() {
        let result = saved_tensors_hooks(
            |t: Tensor<f32>| {
                // Pack: multiply by 2
                let data: Vec<f32> = t.data_vec()?.iter().map(|&x| x * 2.0).collect();
                Tensor::from_storage(TensorStorage::cpu(data), t.shape().to_vec(), false)
            },
            |t: Tensor<f32>| {
                // Unpack: divide by 2
                let data: Vec<f32> = t.data_vec()?.iter().map(|&x| x / 2.0).collect();
                Tensor::from_storage(TensorStorage::cpu(data), t.shape().to_vec(), false)
            },
            || {
                let t = Tensor::from_storage(
                    TensorStorage::cpu(vec![1.0f32, 2.0, 3.0]),
                    vec![3],
                    false,
                )?;
                let packed = pack_saved_tensor(t)?;
                // Packed values should be doubled.
                assert_eq!(packed.data_vec()?, vec![2.0, 4.0, 6.0]);

                let unpacked = unpack_saved_tensor(packed)?;
                // Unpacked values should be back to original.
                assert_eq!(unpacked.data_vec()?, vec![1.0, 2.0, 3.0]);

                Ok(())
            },
        );
        result.unwrap();
    }

    #[test]
    fn test_hooks_cleared_after_scope() {
        saved_tensors_hooks(
            |t: Tensor<f32>| Ok(t),
            |t: Tensor<f32>| Ok(t),
            || {
                assert!(has_saved_tensor_hooks());
                Ok(())
            },
        )
        .unwrap();

        // Hooks should be cleared after scope.
        assert!(!has_saved_tensor_hooks());
    }
}
