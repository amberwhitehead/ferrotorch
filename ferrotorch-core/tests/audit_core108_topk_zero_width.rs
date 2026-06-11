//! Red-then-green regression tests for audit finding CORE-108 (crosslink
//! #1802): `topk` with a zero-width last dimension and `k == 0` passes the
//! `k > last_dim` validation, then both the CPU and GPU paths compute
//! `numel / last_dim` — an integer divide-by-zero panic inside a fallible
//! (`FerrotorchResult`) API (CLASS-U).
//!
//! torch returns correctly shaped EMPTY values + indices, preserving every
//! outer dimension. Oracle quoted from a LIVE torch session
//! (torch 2.11.0+cu130, R-ORACLE-1 path (b)):
//!
//! ```python
//! >>> v, i = torch.topk(torch.empty(2,3,0), 0)
//! >>> v.shape, i.shape, i.dtype
//! (torch.Size([2, 3, 0]), torch.Size([2, 3, 0]), torch.int64)
//! >>> torch.topk(torch.empty(0), 0)[0].shape
//! torch.Size([0])
//! >>> torch.topk(torch.empty(0,5,0), 0)[0].shape
//! torch.Size([0, 5, 0])
//! >>> v, i = torch.topk(torch.empty(2,3,0, device='cuda'), 0)
//! >>> v.device, i.device
//! (device(type='cuda', index=0), device(type='cuda', index=0))
//! ```

use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;
use ferrotorch_core::topk;

#[cfg(feature = "gpu")]
use ferrotorch_core::Device;

fn empty_f32(shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(vec![]), shape.to_vec(), false).unwrap()
}

/// Nonzero outer dims: topk(empty[2,3,0], k=0) -> values [2,3,0], 0 indices.
/// Pre-fix: panics "attempt to divide by zero" at ops/search.rs:700.
#[test]
fn core108_topk_zero_width_nonzero_outer_cpu() {
    let x = empty_f32(&[2, 3, 0]);
    for largest in [true, false] {
        let (values, indices) = topk(&x, 0, largest).unwrap_or_else(|e| {
            panic!("topk k=0 on [2,3,0] (largest={largest}) must succeed: {e:?}")
        });
        assert_eq!(
            values.shape(),
            &[2, 3, 0],
            "values shape (largest={largest})"
        );
        assert_eq!(values.numel(), 0, "values numel (largest={largest})");
        assert!(indices.is_empty(), "indices empty (largest={largest})");
    }
}

/// 1-D zero-width input: topk(empty[0], k=0) -> values [0], 0 indices.
#[test]
fn core108_topk_zero_width_1d_cpu() {
    let x = empty_f32(&[0]);
    let (values, indices) = topk(&x, 0, true).expect("topk k=0 on [0] must succeed");
    assert_eq!(values.shape(), &[0]);
    assert!(indices.is_empty());
}

/// Zero outer dims as well: topk(empty[0,5,0], k=0) -> values [0,5,0].
#[test]
fn core108_topk_zero_width_zero_outer_cpu() {
    let x = empty_f32(&[0, 5, 0]);
    let (values, indices) = topk(&x, 0, true).expect("topk k=0 on [0,5,0] must succeed");
    assert_eq!(values.shape(), &[0, 5, 0]);
    assert!(indices.is_empty());
}

/// `k > 0` on a zero-width last dim must still be the validation error
/// (torch: "selected index k out of range"), NOT a panic.
#[test]
fn core108_topk_k_positive_zero_width_still_errors() {
    let x = empty_f32(&[2, 0]);
    let r = topk(&x, 1, true);
    assert!(r.is_err(), "k=1 > last_dim=0 must be an Err, got {r:?}");
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for the GPU lane of this suite");
        });
    }

    /// CUDA lane: topk(empty[2,3,0] cuda, k=0) -> CUDA-resident empty values
    /// (torch: v.device == i.device == cuda:0). Pre-fix: the same
    /// `numel / last_dim` divide-by-zero panic in the GPU branch.
    #[test]
    fn core108_gpu_topk_zero_width() {
        ensure_cuda_backend();
        let dev = Device::Cuda(0);
        let x = empty_f32(&[2, 3, 0]).to(dev).expect("upload");
        let (values, indices) = topk(&x, 0, true).expect("topk k=0 on cuda [2,3,0] must succeed");
        assert_eq!(values.shape(), &[2, 3, 0], "values shape");
        // R-ORACLE-3 / post-#1890: assert the result device, no CPU demotion.
        assert_eq!(values.device(), dev, "values must be CUDA-resident");
        assert!(indices.is_empty(), "indices empty");
    }

    /// CUDA lane, zero outer dims: topk(empty[0,5,0] cuda, k=0).
    #[test]
    fn core108_gpu_topk_zero_width_zero_outer() {
        ensure_cuda_backend();
        let dev = Device::Cuda(0);
        let x = empty_f32(&[0, 5, 0]).to(dev).expect("upload");
        let (values, indices) = topk(&x, 0, true).expect("topk k=0 on cuda [0,5,0] must succeed");
        assert_eq!(values.shape(), &[0, 5, 0], "values shape");
        assert_eq!(values.device(), dev, "values must be CUDA-resident");
        assert!(indices.is_empty(), "indices empty");
    }
}
