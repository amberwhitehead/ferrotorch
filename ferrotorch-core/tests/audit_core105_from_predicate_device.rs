//! CORE-105 (#1799, CLASS-S High) regression battery:
//! `BoolTensor::from_predicate` must return the mask on the INPUT tensor's
//! device. The predicate is an arbitrary host closure, so for CUDA inputs
//! the values cross to the host (a DOCUMENTED round trip per R-LOUD-2 — see
//! the `from_predicate` doc-comment) and the mask is re-uploaded.
//!
//! Pre-fix observed behavior (R-AHON-1 probe at HEAD, red run pasted in
//! #1799): a CUDA input incurred the full-device readback silently
//! (undocumented) and the mask was returned on `Device::Cpu` — the wrong
//! device, with no transition anywhere in the API or docs.
//!
//! torch contract (live torch 2.11.0+cu130, RTX 3090, R-ORACLE-1(b)):
//! predicate-style mask ops keep the input device —
//! ```text
//! >>> t = torch.tensor([-1.0, 0.0, 2.0, 3.5], device='cuda')
//! >>> m = torch.gt(t, 0)
//! >>> m, m.device
//! (tensor([False, False,  True,  True], device='cuda:0'), device(type='cuda', index=0))
//! ```

#![cfg(feature = "gpu")]

use ferrotorch_core::{BoolTensor, Device, Tensor, TensorStorage};
use std::sync::Once;

static GPU_INIT: Once = Once::new();
fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for the CORE-105 GPU pins");
    });
}

/// A CUDA input must produce a CUDA mask (R-ORACLE-3 device assert) with
/// the values the host predicate computed. Oracle quoted in the module doc:
/// `torch.gt(t_cuda, 0)` → `[False, False, True, True]` on `cuda:0`.
#[test]
fn from_predicate_returns_mask_on_input_device() {
    ensure_cuda_backend();
    let t = Tensor::<f32>::from_storage(
        TensorStorage::cpu(vec![-1.0, 0.0, 2.0, 3.5]),
        vec![4],
        false,
    )
    .unwrap()
    .to(Device::Cuda(0))
    .unwrap();

    let mask = BoolTensor::from_predicate(&t, |v| v > 0.0).unwrap();
    assert_eq!(
        mask.device(),
        Device::Cuda(0),
        "from_predicate on a cuda:0 input must return a cuda:0 mask \
         (torch: torch.gt(t_cuda, 0).device == cuda:0); got {:?}",
        mask.device()
    );
    assert!(mask.is_cuda());
    assert_eq!(mask.shape(), &[4]);
    assert_eq!(
        mask.to(Device::Cpu).unwrap().data().unwrap(),
        &[false, false, true, true]
    );
}

/// A CPU input stays CPU — no gratuitous upload.
#[test]
fn from_predicate_cpu_input_stays_cpu() {
    ensure_cuda_backend();
    let t = Tensor::<f32>::from_storage(
        TensorStorage::cpu(vec![-1.0, 0.0, 2.0, 3.5]),
        vec![4],
        false,
    )
    .unwrap();
    let mask = BoolTensor::from_predicate(&t, |v| v > 0.0).unwrap();
    assert_eq!(mask.device(), Device::Cpu);
    assert_eq!(mask.data().unwrap(), &[false, false, true, true]);
}
