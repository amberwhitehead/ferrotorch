//! R-BUILD-4 critic audit of commit `9f5218b16` — SparseAdam (#1463),
//! materialized-zero edge cases. All expected values are LIVE
//! `torch.optim.SparseAdam` 2.11.0+cu130 outputs (R-CHAR-3).
//!
//! torch's sparse-COO coalesce does NOT drop materialized-zero values: a grad
//! entry whose VALUE is 0.0 (or whose duplicate slabs SUM to 0.0) is still a
//! present index after `grad.coalesce()` (`torch/optim/_functional.py:44`),
//! so the moment buffers still decay (`exp_avg <- beta1*exp_avg + 0`) and the
//! step counter still advances. The "skip" branch fires ONLY for a grad with
//! `numel() == 0`, i.e. genuinely no entries (`_functional.py:47-49`).
//!
//! ferrotorch's `SparseGrad::coalesce` (`ferrotorch-core/src/sparse.rs`) and
//! `SparseAdam::sparse_step` (`ferrotorch-optim/src/sparse_adam.rs:230-234`)
//! treat `nnz == 0` as the skip condition. The question these tests pin: does
//! ferrotorch's coalesce DROP a zero-sum index (turning a should-be-decaying
//! step into a no-op), diverging from torch?

#![allow(clippy::float_cmp)]

use ferrotorch_core::{SparseGrad, Tensor, TensorStorage};
use ferrotorch_nn::Parameter;
use ferrotorch_optim::{Optimizer, SparseAdam, SparseAdamConfig};

fn make_param_2d(data: &[f64], leading: usize, slab: usize) -> Parameter<f64> {
    assert_eq!(data.len(), leading * slab);
    let t = Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![leading, slab], true)
        .unwrap();
    Parameter::new(t)
}

fn read(param: &Parameter<f64>) -> Vec<f64> {
    param.tensor().data_vec().unwrap()
}

/// A materialized-zero grad VALUE (single entry, value 0.0) at a row whose
/// moments are already nonzero from a prior step still moves the param,
/// because the EMA decays the moment and `step_size*m/(sqrt(v)+eps)` is
/// nonzero. torch does NOT short-circuit a 0.0-valued entry.
///
/// Live torch.optim.SparseAdam 2.11.0 (lr=0.1, betas=(0.9,0.999), eps=1e-8),
/// param `[[5.0]]`, grad sequence row0 = [1.0, 0.0, 1.0]:
/// ```text
/// p = nn.Parameter(torch.tensor([[5.0]], dtype=torch.float64))
/// opt = torch.optim.SparseAdam([p], lr=0.1, betas=(0.9,0.999), eps=1e-8)
/// for gv in (1.0, 0.0, 1.0):
///     p.grad = torch.sparse_coo_tensor(torch.tensor([[0]]),
///                  torch.tensor([[gv]], dtype=torch.float64), (1,1))
///     opt.step()
/// # -> [4.900000031622767, 4.832994227408811, 4.751193949314165]
/// ```
#[test]
fn divergence_1463_materialized_zero_value_still_decays_moments() {
    let param = make_param_2d(&[5.0], 1, 1);
    let mut opt = SparseAdam::new(
        vec![param.clone()],
        SparseAdamConfig::default()
            .with_lr(0.1)
            .with_betas((0.9, 0.999))
            .with_eps(1e-8),
    );
    let grad_vals = [1.0, 0.0, 1.0];
    let expected = [
        4.900000031622767,
        4.832994227408811, // torch DECAYS the moment on a 0.0-valued entry
        4.751193949314165,
    ];
    for (i, (&gv, &exp)) in grad_vals.iter().zip(expected.iter()).enumerate() {
        let g = SparseGrad::<f64>::new(vec![0], vec![gv], vec![1]).unwrap();
        opt.set_sparse_grad(0, 0, g);
        opt.step().unwrap();
        let got = read(&param)[0];
        assert!(
            (got - exp).abs() <= 1e-12,
            "materialized-zero step {} (grad={gv}): expected {exp:.17}, got {got:.17} (diff {:.2e})",
            i + 1,
            (got - exp).abs()
        );
    }
}

/// Duplicate slabs that SUM to exactly 0.0 (row0 = +0.5 and -0.5). torch
/// coalesces row0 to a materialized 0.0 and STILL advances the step counter
/// (`state["step"]` becomes 1) — the index is present after `coalesce()`.
/// The param happens to stay 5.0 after step 1 only because both moments end
/// at 0, but the STEP COUNT advanced. This pins whether ferrotorch's coalesce
/// DROPS the zero-sum index (which would leave `step_count == 0`, so the NEXT
/// real grad would compute bias correction at step=1 instead of step=2).
///
/// Live torch.optim.SparseAdam 2.11.0 (lr=0.1, betas=(0.9,0.999), eps=1e-8):
/// step 1 = duplicates summing to 0, step 2 = real grad 1.0:
/// ```text
/// p = nn.Parameter(torch.tensor([[5.0]], dtype=torch.float64))
/// opt = torch.optim.SparseAdam([p], lr=0.1, betas=(0.9,0.999), eps=1e-8)
/// p.grad = torch.sparse_coo_tensor(torch.tensor([[0,0]]),
///              torch.tensor([[0.5],[-0.5]], dtype=torch.float64), (1,1))
/// opt.step()    # step -> 1, param stays 5.0
/// p.grad = torch.sparse_coo_tensor(torch.tensor([[0]]),
///              torch.tensor([[1.0]], dtype=torch.float64), (1,1))
/// opt.step()    # step -> 2, param -> 4.9255863411749665
/// ```
#[test]
fn divergence_1463_zero_sum_coalesce_still_advances_step() {
    let param = make_param_2d(&[5.0], 1, 1);
    let mut opt = SparseAdam::new(
        vec![param.clone()],
        SparseAdamConfig::default()
            .with_lr(0.1)
            .with_betas((0.9, 0.999))
            .with_eps(1e-8),
    );
    // Step 1: row0 with +0.5 and -0.5 -> coalesced sum 0.0 (a present index).
    let g1 = SparseGrad::<f64>::new(vec![0, 0], vec![0.5, -0.5], vec![1]).unwrap();
    opt.set_sparse_grad(0, 0, g1);
    opt.step().unwrap();
    // Param unchanged after step 1 (moments still 0), per torch.
    assert!(
        (read(&param)[0] - 5.0).abs() <= 1e-12,
        "after zero-sum step1 param should be 5.0, got {:.17}",
        read(&param)[0]
    );

    // Step 2: a real grad 1.0. If ferrotorch DROPPED the zero-sum index on
    // step 1 (leaving step_count==0), this update runs at step=1 and yields
    // 4.900000031622767 (the step-1 value); torch runs it at global step=2.
    let g2 = SparseGrad::<f64>::new(vec![0], vec![1.0], vec![1]).unwrap();
    opt.set_sparse_grad(0, 0, g2);
    opt.step().unwrap();

    // Live torch: after step 2 (global step=2).
    let expected_step2 = 4.9255863411749665;
    let got = read(&param)[0];
    assert!(
        (got - expected_step2).abs() <= 1e-12,
        "zero-sum step did not advance the step counter: torch step2 expects \
         {expected_step2:.17} (bias correction at step=2), got {got:.17} (diff {:.2e}). \
         A step2 value of 4.900000031622767 would mean ferrotorch ran this at step=1.",
        (got - expected_step2).abs()
    );
}
