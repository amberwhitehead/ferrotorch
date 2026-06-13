//! Red-then-green regression tests for audit finding CORE-081 (crosslink
//! #1775): sparse SGD replaces tensor identity and leaves aliases stale
//! (CLASS-S — every non-empty `SparseGrad::apply_sgd` path assigned a
//! freshly constructed tensor into `*param`, minting a new `TensorId`,
//! discarding the original tensor's grad/hooks, and leaving clones that
//! shared the old storage observing stale weights).
//!
//! Observed at HEAD (probe, 2026-06-12, rev 74099dd19):
//! - `param.id()` changed across `apply_sgd` ("id stable: false").
//! - a pre-step clone still read `[0.0, 0.0, 0.0, 0.0]` while the param
//!   read `[-1.0, -1.0, 0.0, 0.0]` — alias stale.
//!
//! torch oracle (live session, torch 2.11.0+cu130):
//!
//! ```python
//! >>> p = torch.zeros(4, 3, requires_grad=True)
//! >>> alias = p.data                       # shares storage
//! >>> pid, sptr = id(p), p.data_ptr()
//! >>> p.grad = torch.sparse_coo_tensor(torch.tensor([[1]]),
//! ...                                  torch.ones(1, 3), (4, 3))
//! >>> torch.optim.SGD([p], lr=1.0).step()
//! >>> id(p) == pid, p.data_ptr() == sptr
//! (True, True)                             # identity AND storage stable
//! >>> alias[1].tolist()
//! [-1.0, -1.0, -1.0]                       # alias observes the update
//! >>> p.grad is not None, p.requires_grad
//! (True, True)
//! ```
//!
//! Post-fix contract: `apply_sgd` updates through the CORE-001/#1938
//! in-place primitive (`Tensor::update_storage`): `TensorId`, gradient
//! state, hook registrations, and aliasing clones all survive the step.

use ferrotorch_core::{SparseGrad, Tensor, TensorStorage};

fn grad_fixture() -> SparseGrad<f32> {
    // Row 1 gets [1, 2, 3]; row 3 gets [4, 5, 6].
    SparseGrad::new(vec![1, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3]).unwrap()
}

/// `TensorId` must be stable across the step and pre-step clones must
/// observe the updated weights (torch: `id(p)`/`data_ptr` stable, the
/// `p.data` alias reads the new row).
#[test]
fn core081_cpu_id_stable_and_alias_observes_update() {
    let mut param =
        Tensor::<f32>::from_storage(TensorStorage::cpu(vec![0.0; 12]), vec![4, 3], false).unwrap();
    let alias = param.clone();
    let id_before = param.id();

    grad_fixture().apply_sgd(&mut param, 1.0).expect("cpu sgd");

    assert_eq!(
        param.id(),
        id_before,
        "apply_sgd must not mint a new TensorId (optimizer in-place contract)"
    );

    let p = param.data().expect("param data");
    assert_eq!(&p[3..6], &[-1.0, -2.0, -3.0], "row 1 updated");
    assert_eq!(&p[9..12], &[-4.0, -5.0, -6.0], "row 3 updated");

    let a = alias.data().expect("alias data");
    assert_eq!(
        &a[3..6],
        &[-1.0, -2.0, -3.0],
        "pre-step clone must observe the update (torch: alias reads [-1,-1,-1])"
    );
    assert_eq!(&a[9..12], &[-4.0, -5.0, -6.0]);
}

/// Gradient state set before the step survives it (torch: `p.grad`
/// remains set after `optim.SGD.step()`).
#[test]
fn core081_cpu_grad_and_requires_grad_survive() {
    let mut param =
        Tensor::<f32>::from_storage(TensorStorage::cpu(vec![0.0; 12]), vec![4, 3], true).unwrap();
    let g =
        Tensor::<f32>::from_storage(TensorStorage::cpu(vec![1.0; 12]), vec![4, 3], false).unwrap();
    param.set_grad(Some(g)).expect("set grad");

    grad_fixture().apply_sgd(&mut param, 1.0).expect("cpu sgd");

    assert!(param.requires_grad(), "requires_grad must survive the step");
    assert!(
        param.grad().expect("grad access").is_some(),
        "the dense grad attached before the step must survive (pre-fix: \
         dropped with the replaced tensor)"
    );
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
                .expect("CUDA backend must initialize for the GPU lane");
        });
    }

    /// CUDA lane: id stability + alias observation, same contract.
    #[test]
    fn core081_gpu_id_stable_and_alias_observes_update() {
        ensure_cuda_backend();
        let cpu = Tensor::<f32>::from_storage(TensorStorage::cpu(vec![0.0; 12]), vec![4, 3], false)
            .unwrap();
        let mut param = cpu.to(Device::Cuda(0)).expect("param->cuda");
        let alias = param.clone();
        let id_before = param.id();

        grad_fixture().apply_sgd(&mut param, 1.0).expect("gpu sgd");

        assert!(param.is_cuda(), "param must stay on CUDA");
        assert_eq!(param.id(), id_before, "TensorId must be stable on CUDA too");

        let p = param.cpu().unwrap().data().unwrap().to_vec();
        assert_eq!(&p[3..6], &[-1.0, -2.0, -3.0]);

        assert!(alias.is_cuda(), "alias must stay on CUDA");
        let a = alias.cpu().unwrap().data().unwrap().to_vec();
        assert_eq!(
            &a[3..6],
            &[-1.0, -2.0, -3.0],
            "pre-step CUDA clone must observe the update"
        );
    }
}
