//! Dtype-generic GPU dispatch for `Float` tensors.
//!
//! PyTorch's `AT_DISPATCH_FLOATING_TYPES_AND_HALF` analog for ferrotorch:
//! a single macro [`dispatch_floating_dtype`] that switches on the
//! tensor's static `T` (`TypeId`) and routes to one of three closures
//! — one per supported floating dtype — returning
//! [`FerrotorchError::NotImplementedOnCuda`] for any other dtype.
//!
//! # Why a macro
//!
//! Before this module, the typical bf16+CUDA dispatch site in
//! `ferrotorch-core/src/grad_fns/*` was a chain of
//! `if is_f32::<T>() { ... } else if is_f64::<T>() { ... } else { ... }`
//! where the trailing `else` either:
//!
//! 1. Defaulted to the `f32` arm (and died at `unwrap_buffer::<f32>` for
//!    bf16 — issue #23 pattern A).
//! 2. Returned `NotImplementedOnCuda` even when a bf16 kernel exists
//!    (pattern B).
//! 3. Fell through to a CPU helper that errored on the GPU storage
//!    (pattern C).
//!
//! All three patterns share a structural defect: the dispatch list was
//! incomplete because each site enumerated dtypes by hand. Replacing
//! the chain with `dispatch_floating_dtype!` collapses every site to a
//! complete enumeration (`f32 / f64 / bf16`) and a single uniform
//! `NotImplementedOnCuda` for unsupported dtypes, eliminating
//! pattern-A silent fallthrough by construction.
//!
//! # PyTorch parity
//!
//! Matches PyTorch's
//! `AT_DISPATCH_FLOATING_TYPES_AND_HALF(scalar_t, "op_name", [&] { ... })`
//! shape: each arm gets `scalar_t` bound to the concrete dtype and runs
//! the body. The error for unsupported dtypes maps to
//! `RuntimeError: <op> not implemented for '<dtype>'` in PyTorch.
//!
//! # Example
//!
//! ```ignore
//! use ferrotorch_core::dispatch_floating_dtype;
//!
//! fn add_inner<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
//!     // Inputs already on CUDA, contig-materialised; this is the dtype arm only.
//!     let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
//!     dispatch_floating_dtype!(
//!         T,
//!         "add",
//!         f32 => {
//!             let h = backend.add_f32(a.gpu_handle()?, b.gpu_handle()?)?;
//!             Ok(Tensor::from_storage(TensorStorage::gpu(h), a.shape().to_vec(), false)?)
//!         },
//!         f64 => {
//!             let h = backend.add_f64(a.gpu_handle()?, b.gpu_handle()?)?;
//!             Ok(Tensor::from_storage(TensorStorage::gpu(h), a.shape().to_vec(), false)?)
//!         },
//!         bf16 => {
//!             let h = backend.add_bf16_bf16(a.gpu_handle()?, b.gpu_handle()?)?;
//!             Ok(Tensor::from_storage(TensorStorage::gpu(h), a.shape().to_vec(), false)?)
//!         },
//!     )
//! }
//! ```
//!
//! All three arms return `FerrotorchResult<U>`; the macro evaluates to
//! the result of the matching arm or an `Err(NotImplementedOnCuda)` for
//! unsupported dtypes. Each arm is a block, so it can contain prologue
//! statements; per the rust-gpu-discipline rules, callers extract any
//! op-prologue (broadcast inference, `ensure_contig_for_gpu`,
//! `needs_grad` fork) **out** of the arms into a generic helper so the
//! macro doesn't hide complexity inside per-dtype blocks.

/// Dispatch to one of three closures based on the static type `T`,
/// returning `Err(FerrotorchError::NotImplementedOnCuda { op })` for
/// any dtype other than `f32`, `f64`, or `half::bf16`.
///
/// Each arm is an expression that evaluates to the same
/// `FerrotorchResult<U>` type; the macro evaluates to that result.
///
/// # Arguments
///
/// - `$scalar_t`: the static type parameter to dispatch on (typically
///   `T` where `T: Float`).
/// - `$op`: a `&'static str` op name surfaced in the error message
///   when dispatch fails.
/// - `f32 => $f32_arm`: expression run when `T == f32`.
/// - `f64 => $f64_arm`: expression run when `T == f64`.
/// - `bf16 => $bf16_arm`: expression run when `T == half::bf16`.
///
/// # See also
///
/// - PyTorch's `AT_DISPATCH_FLOATING_TYPES_AND_HALF` macro.
/// - `ferrotorch-core/src/dispatch.rs` for the unrelated keyset-based
///   multi-backend dispatcher.
#[macro_export]
macro_rules! dispatch_floating_dtype {
    (
        $scalar_t:ty,
        $op:literal,
        f32 => $f32_arm:expr,
        f64 => $f64_arm:expr,
        bf16 => $bf16_arm:expr $(,)?
    ) => {{
        if ::std::any::TypeId::of::<$scalar_t>() == ::std::any::TypeId::of::<f32>() {
            $f32_arm
        } else if ::std::any::TypeId::of::<$scalar_t>() == ::std::any::TypeId::of::<f64>() {
            $f64_arm
        } else if ::std::any::TypeId::of::<$scalar_t>() == ::std::any::TypeId::of::<half::bf16>() {
            $bf16_arm
        } else {
            ::std::result::Result::Err($crate::error::FerrotorchError::NotImplementedOnCuda { op: $op })
        }
    }};
}

/// Returns `true` if `T` is `f32`. Small helper used by call sites
/// that need to branch on dtype for non-dispatch reasons (e.g.
/// upstream `ensure_contig_for_gpu`).
#[inline]
#[must_use]
pub fn is_f32<T: 'static>() -> bool {
    ::std::any::TypeId::of::<T>() == ::std::any::TypeId::of::<f32>()
}

/// Returns `true` if `T` is `f64`.
#[inline]
#[must_use]
pub fn is_f64<T: 'static>() -> bool {
    ::std::any::TypeId::of::<T>() == ::std::any::TypeId::of::<f64>()
}

/// Returns `true` if `T` is `half::bf16`.
#[inline]
#[must_use]
pub fn is_bf16<T: 'static>() -> bool {
    ::std::any::TypeId::of::<T>() == ::std::any::TypeId::of::<half::bf16>()
}

#[cfg(test)]
mod tests {
    use crate::error::{FerrotorchError, FerrotorchResult};

    fn run_test<T: 'static>() -> FerrotorchResult<&'static str> {
        dispatch_floating_dtype!(
            T,
            "test_op",
            f32 => Ok("f32"),
            f64 => Ok("f64"),
            bf16 => Ok("bf16"),
        )
    }

    #[test]
    fn dispatch_f32() {
        assert_eq!(run_test::<f32>().unwrap(), "f32");
    }

    #[test]
    fn dispatch_f64() {
        assert_eq!(run_test::<f64>().unwrap(), "f64");
    }

    #[test]
    fn dispatch_bf16() {
        assert_eq!(run_test::<half::bf16>().unwrap(), "bf16");
    }

    #[test]
    fn dispatch_unsupported_dtype_returns_not_implemented() {
        // i32 isn't a Float, but the macro is dtype-agnostic at the
        // TypeId level — it returns the structured error for anything
        // outside the three supported floats.
        let result = run_test::<i32>();
        match result {
            Err(FerrotorchError::NotImplementedOnCuda { op }) => assert_eq!(op, "test_op"),
            other => panic!("expected NotImplementedOnCuda, got {other:?}"),
        }
    }
}
