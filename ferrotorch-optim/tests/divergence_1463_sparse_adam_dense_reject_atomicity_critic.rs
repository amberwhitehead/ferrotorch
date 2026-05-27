//! R-BUILD-4 critic audit of commit `9f5218b16` — SparseAdam (#1463),
//! dense-grad rejection ATOMICITY. Expected values are LIVE
//! `torch.optim.SparseAdam` 2.11.0+cu130 outputs (R-CHAR-3).
//!
//! Upstream `torch.optim.SparseAdam.step` (`torch/optim/sparse_adam.py:76-127`)
//! collects ALL params-with-grad into lists in one loop, raising the dense
//! `RuntimeError` the moment it sees a NON-sparse `p.grad`
//! (`sparse_adam.py:88-92`) — BEFORE any call to `F.sparse_adam`
//! (`sparse_adam.py:116`). Therefore, if any param in the group carries a
//! dense grad, the raise happens with NO param DATA mutated, even for params
//! earlier in the group that did have a valid sparse grad.
//!
//! ferrotorch's `SparseAdam::step`
//! (`ferrotorch-optim/src/sparse_adam.rs:331-387`) iterates params one at a
//! time and APPLIES `sparse_step` (writing the param via
//! `tensor.update_data`, `sparse_adam.rs:324-326`) for an earlier param
//! BEFORE it reaches a later param's dense grad and returns the Err. So an
//! earlier param is left half-updated on the error path — diverging from
//! torch's collect-then-raise atomicity. Tracking: #1593.

#![allow(clippy::float_cmp)]

use ferrotorch_core::{FerrotorchError, SparseGrad, Tensor, TensorStorage};
use ferrotorch_nn::Parameter;
use ferrotorch_optim::{Optimizer, ParamGroup, SparseAdam, SparseAdamConfig};

fn make_param_2d(data: &[f64], leading: usize, slab: usize) -> Parameter<f64> {
    assert_eq!(data.len(), leading * slab);
    let t =
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![leading, slab], true).unwrap();
    Parameter::new(t)
}

fn read(param: &Parameter<f64>) -> Vec<f64> {
    param.tensor().data_vec().unwrap()
}

/// Two params in one group: p0 has a valid sparse grad, p1 has a DENSE grad.
/// torch raises the dense-grad RuntimeError in the collection loop and leaves
/// BOTH params' DATA unchanged (p0 stays 1.0, p1 stays 2.0).
///
/// Live torch.optim.SparseAdam 2.11.0 (lr=0.1, betas=(0.9,0.999), eps=1e-8):
/// ```text
/// p0 = nn.Parameter(torch.tensor([[1.0]], dtype=torch.float64))
/// p1 = nn.Parameter(torch.tensor([[2.0]], dtype=torch.float64))
/// opt = torch.optim.SparseAdam([p0, p1], lr=0.1, betas=(0.9,0.999), eps=1e-8)
/// p0.grad = torch.sparse_coo_tensor(torch.tensor([[0]]),
///              torch.tensor([[1.0]], dtype=torch.float64), (1,1))
/// p1.grad = torch.tensor([[0.5]], dtype=torch.float64)   # DENSE
/// opt.step()                       # raises; p0 data == 1.0, p1 data == 2.0
/// ```
///
/// FAILS against `9f5218b16`: ferrotorch leaves p0 == 0.9000000316227666.
#[test]
fn divergence_1463_dense_reject_leaves_earlier_param_unmutated() {
    let p0 = make_param_2d(&[1.0], 1, 1);
    let p1 = make_param_2d(&[2.0], 1, 1);

    let mut opt = SparseAdam::new(
        vec![p0.clone(), p1.clone()],
        SparseAdamConfig::default()
            .with_lr(0.1)
            .with_betas((0.9, 0.999))
            .with_eps(1e-8),
    );

    // p0: valid sparse grad. p1: dense grad (no registered sparse grad).
    let g0 = SparseGrad::<f64>::new(vec![0], vec![1.0], vec![1]).unwrap();
    opt.set_sparse_grad(0, 0, g0);
    let dense = Tensor::from_storage(TensorStorage::cpu(vec![0.5f64]), vec![1, 1], false).unwrap();
    p1.tensor().set_grad(Some(dense)).unwrap();

    let err = opt.step().unwrap_err();
    match err {
        FerrotorchError::InvalidArgument { ref message } => assert_eq!(
            message, "SparseAdam does not support dense gradients, please consider Adam instead",
            "must mirror torch RuntimeError verbatim"
        ),
        other => panic!("expected dense-grad rejection, got {other:?}"),
    }

    // torch leaves p0's DATA at 1.0 (the raise pre-empts F.sparse_adam).
    // ferrotorch applies sparse_step to p0 before reaching p1's dense grad,
    // mutating p0 to 0.9 — the divergence this test pins.
    assert_eq!(
        read(&p0)[0],
        1.0,
        "torch raises on the dense grad BEFORE updating any param data; p0 \
         must be unchanged on the error path, but ferrotorch already applied \
         the masked step to p0 (got {:.17})",
        read(&p0)[0]
    );
    assert_eq!(read(&p1)[0], 2.0, "p1 (dense) must be unchanged");
}

/// Same divergence, surfaced through an explicit second param group: the
/// dense-grad param is in group 1, the valid sparse-grad param in group 0.
/// torch raises before any data write; ferrotorch processes group 0 first.
///
/// Live torch.optim.SparseAdam 2.11.0 (lr=0.1, betas=(0.9,0.999), eps=1e-8):
/// group-0 param `[[10.0]]` with row-0 grad 1.0, group-1 param `[[20.0]]`
/// with a dense grad — torch raises and both stay 10.0 / 20.0.
///
/// FAILS against `9f5218b16`: ferrotorch leaves group-0 == 9.900000031622767.
#[test]
fn divergence_1463_dense_reject_atomicity_across_groups() {
    let p0 = make_param_2d(&[10.0], 1, 1);
    let p1 = make_param_2d(&[20.0], 1, 1);

    let mut opt = SparseAdam::new(
        vec![p0.clone()],
        SparseAdamConfig::default()
            .with_lr(0.1)
            .with_betas((0.9, 0.999))
            .with_eps(1e-8),
    );
    opt.add_param_group(ParamGroup::new(vec![p1.clone()], 0.1));

    let g0 = SparseGrad::<f64>::new(vec![0], vec![1.0], vec![1]).unwrap();
    opt.set_sparse_grad(0, 0, g0);
    let dense = Tensor::from_storage(TensorStorage::cpu(vec![0.5f64]), vec![1, 1], false).unwrap();
    p1.tensor().set_grad(Some(dense)).unwrap();

    let _ = opt.step().unwrap_err();

    // torch: no param data written on the error path.
    assert_eq!(
        read(&p0)[0],
        10.0,
        "group-0 param must be unchanged when a later group has a dense grad; \
         ferrotorch already applied the step (got {:.17})",
        read(&p0)[0]
    );
    assert_eq!(read(&p1)[0], 20.0, "group-1 (dense) param unchanged");
}
