//! Regression tests for CORE-154 (#1848): `Tensor::narrow` must reject
//! out-of-range `(start, length)` metadata with a structured error before
//! any `usize` arithmetic can wrap.
//!
//! PyTorch's nonnegative subset accepts `start == size && length == 0` but
//! rejects starts past that inclusive empty-end boundary:
//!
//! ```python
//! >>> x = torch.arange(6.)
//! >>> x.narrow(0, 6, 0).shape
//! torch.Size([0])
//! >>> x.narrow(0, 7, 0)
//! IndexError: start out of range (expected to be in range of [-6, 6], but got 7)
//! >>> x.narrow(0, 0, -1)
//! RuntimeError: narrow(): length must be non-negative.
//! ```
//!
//! Ferrotorch's API uses `usize` for `start` and `length`, so negative inputs
//! are unrepresentable; the important parity boundary here is that every
//! nonnegative out-of-range request returns `InvalidArgument`, never a debug
//! panic or a release-mode wrapped view. The same validation family also
//! covers `split_t`'s split-size sum, which must not wrap and accidentally
//! pass the dimension-size check in release builds.

use ferrotorch_core::{FerrotorchError, Tensor, TensorStorage, split_t};

fn cpu_tensor() -> Tensor<f32> {
    Tensor::from_storage(
        TensorStorage::cpu(vec![1.0_f32, 2.0, 3.0, 4.0]),
        vec![4],
        false,
    )
    .expect("cpu tensor")
}

fn assert_invalid_argument(result: ferrotorch_core::FerrotorchResult<Tensor<f32>>, needle: &str) {
    match result {
        Err(FerrotorchError::InvalidArgument { message }) => {
            assert!(
                message.contains(needle),
                "error message {message:?} did not contain {needle:?}"
            );
        }
        Err(other) => panic!("expected InvalidArgument, got {other:?}"),
        Ok(tensor) => panic!("expected InvalidArgument, got Ok({tensor:?})"),
    }
}

fn assert_invalid_argument_split(
    result: ferrotorch_core::FerrotorchResult<Vec<Tensor<f32>>>,
    needle: &str,
) {
    match result {
        Err(FerrotorchError::InvalidArgument { message }) => {
            assert!(
                message.contains(needle),
                "error message {message:?} did not contain {needle:?}"
            );
        }
        Err(other) => panic!("expected InvalidArgument, got {other:?}"),
        Ok(tensors) => panic!("expected InvalidArgument, got Ok({tensors:?})"),
    }
}

#[test]
fn cpu_narrow_rejects_wrapping_start_plus_length() {
    let x = cpu_tensor();
    assert_invalid_argument(x.narrow(0, usize::MAX, 2), "overflow");
}

#[test]
fn cpu_narrow_rejects_length_larger_than_dimension_without_underflow() {
    let x = cpu_tensor();
    assert_invalid_argument(x.narrow(0, 0, usize::MAX), "length");
}

#[test]
fn cpu_narrow_accepts_zero_length_at_end_like_torch() {
    let x = cpu_tensor();
    let y = x.narrow(0, 4, 0).expect("zero-length end narrow");
    assert_eq!(y.shape(), &[0]);
    assert_eq!(y.storage_offset(), 4);
    assert!(y.data().expect("empty data").is_empty());
}

#[test]
fn cpu_narrow_rejects_zero_length_past_end_like_torch() {
    let x = cpu_tensor();
    assert_invalid_argument(x.narrow(0, 5, 0), "exceeds dim size");
}

#[test]
fn cpu_split_rejects_wrapping_split_size_sum() {
    let x = cpu_tensor();
    assert_invalid_argument_split(split_t(&x, &[usize::MAX, 5], 0), "overflows");
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use ferrotorch_core::Device;
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for narrow overflow tests");
        });
    }

    #[test]
    fn cuda_narrow_rejects_wrapping_start_plus_length_before_view_metadata() {
        ensure_cuda_backend();
        let x = cpu_tensor().to(Device::Cuda(0)).expect("cpu to cuda");
        assert_eq!(x.device(), Device::Cuda(0));
        assert_invalid_argument(x.narrow(0, usize::MAX, 2), "overflow");
    }
}
