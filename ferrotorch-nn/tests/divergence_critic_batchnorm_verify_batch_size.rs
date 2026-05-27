//! Critic divergence (#1449 re-audit): torch's `_verify_batch_size` guard is
//! missing from ferrotorch's BatchNorm forward (both CPU and the new GPU fast
//! path). torch raises `ValueError` when training with exactly one value per
//! channel; ferrotorch silently normalizes to zero.
//!
//! This is a pre-existing CPU-path gap that the #1449 GPU path inherits — NOT
//! a GPU-kernel numeric bug — so it is `#[ignore]`d behind tracking issue
//! #1558 rather than blocking. The four parity probes in
//! `divergence_critic_batchnorm_gpu.rs` (which DO run live and pass) cover the
//! GPU kernel's actual math; this pins the validation gap.

#![cfg(feature = "cuda")]

use ferrotorch_core::{Device, Tensor, TensorStorage};
use ferrotorch_nn::module::Module as _;
use ferrotorch_nn::norm::BatchNorm2d;

fn cuda_ready() -> bool {
    ferrotorch_gpu::init_cuda_backend().is_ok()
}

/// Divergence: ferrotorch's `BatchNorm2d::forward` (GPU path, train mode)
/// diverges from `torch/nn/functional.py:2813` (`_verify_batch_size`) for input
/// `[1, 3, 1, 1]` (count == 1 value per channel).
///
/// Upstream torch raises:
///   `ValueError: Expected more than 1 value per channel when training, got
///    input size torch.Size([1, 3, 1, 1])`
/// (verified live, torch 2.11.0+cu130).
/// ferrotorch returns `Ok([0.0, 0.0, 0.0])` (verified live on RTX 3090).
/// Tracking: #1558
#[test]
#[ignore = "divergence: BatchNorm train missing _verify_batch_size guard; tracking #1558"]
fn divergence_batchnorm2d_train_count1_must_reject() {
    if !cuda_ready() {
        return;
    }
    let mut bn = BatchNorm2d::<f32>::new(3, 1e-5, 0.1, true).unwrap();
    bn.to_device(Device::Cuda(0)).unwrap();
    // [1, 3, 1, 1]: batch=1, spatial=1 -> count == 1 per channel.
    let x = Tensor::from_storage(
        TensorStorage::cpu(vec![1.0f32, 2.0, 3.0]),
        vec![1, 3, 1, 1],
        false,
    )
    .unwrap()
    .to(Device::Cuda(0))
    .unwrap();

    let result = bn.forward(&x);
    assert!(
        result.is_err(),
        "torch raises ValueError(\"Expected more than 1 value per channel when \
         training\") for [1,3,1,1]; ferrotorch must also reject, but returned \
         Ok({:?})",
        result.ok().and_then(|t| t.data_vec().ok())
    );
}
