//! CORE-104 (#1798, CLASS-U High) regression battery:
//! `BoolTensor::from_gpu_handle` must be a FALLIBLE constructor that
//! validates, in release builds too, every invariant later kernels and
//! readback trust: the `DType::Bool` handle tag, a non-overflowing shape
//! element count, and `handle.len() == prod(shape)`.
//!
//! Pre-fix observed behavior (R-AHON-1 probe at HEAD, red run pasted in
//! #1798): the constructor was safe/public/INFALLIBLE — the dtype tag was a
//! `debug_assert` (compiled out in release, the default lane) and the length
//! was never checked at all, so both malformed cases below constructed
//! tensors whose `numel()` disagreed with the declared shape or whose
//! storage was tagged `F32`:
//!
//! ```text
//! assertion `left == right` failed: malformed BoolTensor accepted:
//!   storage len 3 disagrees with declared shape [4]   (left: 3, right: 4)
//! assertion `left == right` failed: malformed BoolTensor accepted:
//!   handle dtype tag is F32, not Bool                  (left: F32, right: Bool)
//! ```
//!
//! `GpuBufferHandle::new` is public, so the malformed-handle reds are
//! constructible without hardware; the kernel-produced happy path is pinned
//! under the `gpu` feature with device asserts (R-ORACLE-3).

use ferrotorch_core::BoolTensor;
use ferrotorch_core::dtype::DType;
use ferrotorch_core::error::FerrotorchError;
use ferrotorch_core::gpu_dispatch::GpuBufferHandle;

fn fake_handle(len: usize, dtype: DType) -> GpuBufferHandle {
    // SAFETY: these tests intentionally forge metadata-only handles to probe
    // BoolTensor::from_gpu_handle validation. The handles are never submitted
    // to a CUDA backend or dereferenced as device memory.
    unsafe { GpuBufferHandle::new(Box::new(()), 0, len, dtype) }
}

/// (a) `handle.len()` must equal the shape's element count.
#[test]
fn from_gpu_handle_rejects_wrong_length() {
    let handle = fake_handle(3, DType::Bool);
    match BoolTensor::from_gpu_handle(handle, vec![4]) {
        Err(FerrotorchError::ShapeMismatch { message }) => {
            assert!(
                message.contains("handle.len()=3") && message.contains("4"),
                "error must name both lengths; got: {message}"
            );
        }
        other => panic!("len-3 handle under shape [4] must be ShapeMismatch; got {other:?}"),
    }
}

/// (b) The handle must carry the `DType::Bool` tag (was a debug_assert —
/// release builds accepted any tag).
#[test]
fn from_gpu_handle_rejects_wrong_dtype() {
    let handle = fake_handle(4, DType::F32);
    match BoolTensor::from_gpu_handle(handle, vec![4]) {
        Err(FerrotorchError::DtypeMismatch { expected, got }) => {
            assert_eq!(expected, "Bool");
            assert_eq!(got, "F32");
        }
        other => panic!("F32-tagged handle must be DtypeMismatch; got {other:?}"),
    }
}

/// (c) The shape element count is computed with checked multiplication —
/// an overflowing shape is a structured error, never a wrapped product.
#[test]
fn from_gpu_handle_rejects_shape_product_overflow() {
    let handle = fake_handle(4, DType::Bool);
    match BoolTensor::from_gpu_handle(handle, vec![usize::MAX, 2]) {
        Err(FerrotorchError::ShapeMismatch { message }) => {
            assert!(
                message.contains("overflows"),
                "error must name the overflow; got: {message}"
            );
        }
        other => panic!("overflowing shape product must be ShapeMismatch; got {other:?}"),
    }
}

/// (d) `shape == []` is the 0-d scalar (numel 1) per the #805 convention
/// shared with `from_vec` / `zeros` / `ones`: a 1-element handle is Ok, a
/// 0-element handle is rejected.
#[test]
fn from_gpu_handle_zero_dim_uses_numel_one() {
    let ok = fake_handle(1, DType::Bool);
    let t = BoolTensor::from_gpu_handle(ok, vec![]).expect("0-d scalar handle (len 1)");
    assert_eq!(t.shape(), &[] as &[usize]);
    assert_eq!(t.numel(), 1);

    let bad = fake_handle(0, DType::Bool);
    assert!(
        matches!(
            BoolTensor::from_gpu_handle(bad, vec![]),
            Err(FerrotorchError::ShapeMismatch { .. })
        ),
        "0-len handle under 0-d shape (numel 1) must be ShapeMismatch"
    );
}

// ---------------------------------------------------------------------------
// CUDA happy path (gpu feature + hardware): kernel-produced handles still
// construct Ok and the result stays resident with correct values.
// ---------------------------------------------------------------------------
#[cfg(feature = "gpu")]
mod gpu {
    use ferrotorch_core::{BoolTensor, Device};
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();
    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for the CORE-104 GPU pins");
        });
    }

    /// `not()` on a CUDA-resident mask routes its kernel output through the
    /// now-fallible `from_gpu_handle`: still Ok, still CUDA-resident
    /// (R-ORACLE-3), values bit-exact.
    ///
    /// Live torch 2.11.0+cu130 (RTX 3090):
    /// ```text
    /// >>> ~torch.tensor([True, False, True], device='cuda')
    /// tensor([False,  True, False], device='cuda:0')
    /// ```
    #[test]
    fn kernel_produced_handles_still_construct() {
        ensure_cuda_backend();
        let m = BoolTensor::from_vec(vec![true, false, true], vec![3])
            .unwrap()
            .to(Device::Cuda(0))
            .unwrap();
        let n = m.not().expect("bool not");
        assert!(
            n.is_cuda(),
            "bool_not kernel output must stay CUDA-resident (got {:?})",
            n.device()
        );
        assert_eq!(
            n.to(Device::Cpu).unwrap().data().unwrap(),
            &[false, true, false]
        );
    }
}
