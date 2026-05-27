//! R-BUILD-4 critic audit of commit `9f5218b16` — SparseAdam sparse-COO
//! consumer (#1463). Every expected value below is a LIVE
//! `torch.optim.SparseAdam` 2.11.0+cu130 output (R-CHAR-3), reproduced by the
//! Python snippet quoted in each test's doc comment, NOT copied from the
//! ferrotorch side.
//!
//! These tests target the axes the in-crate lib tests do NOT exercise:
//!   - Axis #2 (HIGHEST RISK): different rows touched on different steps, so
//!     the bias-correction step count for a row first-touched on step 2 must
//!     be the GLOBAL per-parameter step (=2), NOT a per-row count (=1). The
//!     existing `sparse_adam_matches_torch_oracle_multi_step` always touches
//!     the SAME single row, so it cannot distinguish the two.
//!   - Axis #1: a longer (4-step) single-row trajectory triple-confirming the
//!     `step_size = lr*sqrt(bc2)/bc1`, `param -= step_size*m/(sqrt(v)+eps)`
//!     eps placement over many steps (`torch/optim/_functional.py:80-84`).
//!   - Axis #3: duplicate-index coalescing summed (`_functional.py:44`).
//!   - Axis #5: untouched rows fully invariant in a multi-row scenario.

#![allow(clippy::float_cmp)]

use ferrotorch_core::{SparseGrad, Tensor, TensorStorage};
use ferrotorch_nn::Parameter;
use ferrotorch_optim::{Optimizer, SparseAdam, SparseAdamConfig};

/// `[leading, slab]`-shaped f64 parameter from a flat row-major buffer.
fn make_param_2d(data: &[f64], leading: usize, slab: usize) -> Parameter<f64> {
    assert_eq!(data.len(), leading * slab);
    let t =
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![leading, slab], true).unwrap();
    Parameter::new(t)
}

fn read(param: &Parameter<f64>) -> Vec<f64> {
    param.tensor().data_vec().unwrap()
}

fn assert_close(got: &[f64], expected: &[f64], ctx: &str) {
    assert_eq!(got.len(), expected.len(), "{ctx}: length mismatch");
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() <= 1e-12,
            "{ctx}[{i}]: expected {e:.17}, got {g:.17} (diff {:.2e})",
            (g - e).abs()
        );
    }
}

/// AXIS #2 (highest risk): step 1 touches rows {0,2}; step 2 touches rows
/// {1,2}. torch's `state["step"]` is GLOBAL per-parameter, incremented once
/// per `step()` call (`torch/optim/sparse_adam.py:112-114`). So at step 2,
/// row 1 (first touched on step 2) uses bias correction with `step=2`, and
/// row 2 (touched both steps) carries its step-1 moments forward.
///
/// A per-row step count would put row 1's bias correction at `step=1`, giving
/// a DIFFERENT param. This test discriminates the two.
///
/// Live torch.optim.SparseAdam 2.11.0 (lr=0.1, betas=(0.9,0.999), eps=1e-8):
/// ```text
/// p = nn.Parameter(torch.tensor([[1.,2.],[3.,4.],[5.,6.]], dtype=torch.float64))
/// opt = torch.optim.SparseAdam([p], lr=0.1, betas=(0.9,0.999), eps=1e-8)
/// p.grad = torch.sparse_coo_tensor(torch.tensor([[0,2]]),
///              torch.tensor([[0.1,0.2],[0.3,0.4]], dtype=torch.float64), (3,2))
/// opt.step()                                  # -> param after step1 below
/// p.grad = torch.sparse_coo_tensor(torch.tensor([[1,2]]),
///              torch.tensor([[0.5,0.6],[0.7,0.8]], dtype=torch.float64), (3,2))
/// opt.step()                                  # -> param after step2 below
/// ```
/// (`torch/optim/_functional.py:24-84`).
#[test]
fn divergence_1463_axis2_different_rows_per_step_uses_global_step() {
    let param = make_param_2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
    let mut opt = SparseAdam::new(
        vec![param.clone()],
        SparseAdamConfig::default()
            .with_lr(0.1)
            .with_betas((0.9, 0.999))
            .with_eps(1e-8),
    );

    // Step 1: rows 0 and 2.
    let g1 = SparseGrad::<f64>::new(vec![0, 2], vec![0.1, 0.2, 0.3, 0.4], vec![2]).unwrap();
    opt.set_sparse_grad(0, 0, g1);
    opt.step().unwrap();
    // Live torch after step 1.
    let after_step1 = [
        0.900000316226766,
        1.900000158113633,
        3.0,
        4.0,
        4.900000105409144,
        5.900000079056879,
    ];
    assert_close(&read(&param), &after_step1, "axis2 after step1");

    // Step 2: rows 1 and 2.
    let g2 = SparseGrad::<f64>::new(vec![1, 2], vec![0.5, 0.6, 0.7, 0.8], vec![2]).unwrap();
    opt.set_sparse_grad(0, 0, g2);
    opt.step().unwrap();
    // Live torch after step 2. Row 1 first touched here: its bias correction
    // MUST use the global step=2, not a per-row step=1.
    let after_step2 = [
        0.900000316226766,  // row0: untouched on step2 -> identical to after_step1
        1.900000158113633,  // row0
        2.925586364706617,  // row1: first touched, bias-correction at step=2
        3.9255863568627354, // row1
        4.805214137091536,  // row2: touched both steps, moments carried forward
        5.80348191060244,   // row2
    ];
    assert_close(&read(&param), &after_step2, "axis2 after step2");
}

/// AXIS #1: 4-step single-row trajectory. Confirms the
/// `step_size = lr*sqrt(bc2)/bc1` + `param -= step_size*m/(sqrt(v)+eps)` eps
/// placement holds across more steps than the in-crate 3-step test.
///
/// Live torch.optim.SparseAdam 2.11.0 (lr=0.1, betas=(0.9,0.999), eps=1e-8),
/// param `[[5.0]]`, constant grad row 0 = 1.0:
/// ```text
/// p = nn.Parameter(torch.tensor([[5.0]], dtype=torch.float64))
/// opt = torch.optim.SparseAdam([p], lr=0.1, betas=(0.9,0.999), eps=1e-8)
/// for _ in range(4):
///     p.grad = torch.sparse_coo_tensor(torch.tensor([[0]]),
///                  torch.tensor([[1.0]], dtype=torch.float64), (1,1))
///     opt.step()
/// ```
#[test]
fn divergence_1463_axis1_four_step_trajectory_eps_placement() {
    let param = make_param_2d(&[5.0], 1, 1);
    let mut opt = SparseAdam::new(
        vec![param.clone()],
        SparseAdamConfig::default()
            .with_lr(0.1)
            .with_betas((0.9, 0.999))
            .with_eps(1e-8),
    );
    let expected = [
        4.900000031622767,
        4.800000053989034,
        4.700000072255582,
        4.600000088078833,
    ];
    for (step, &exp) in expected.iter().enumerate() {
        let g = SparseGrad::<f64>::new(vec![0], vec![1.0], vec![1]).unwrap();
        opt.set_sparse_grad(0, 0, g);
        opt.step().unwrap();
        let got = read(&param)[0];
        assert!(
            (got - exp).abs() <= 1e-12,
            "axis1 step {}: expected {exp:.17}, got {got:.17}",
            step + 1
        );
    }
}

/// AXIS #3: duplicate-index coalescing. Row 2 appears twice (0.3, 0.7) in the
/// same step; torch coalesces to a single row-2 gradient of 1.0 BEFORE the
/// moment update (`torch/optim/_functional.py:44`), NOT last-wins (0.7) and
/// NOT double-applied. Proven by matching the explicitly-coalesced control.
///
/// Live torch.optim.SparseAdam 2.11.0 (lr=0.5, betas=(0.9,0.999), eps=1e-8):
/// ```text
/// p = nn.Parameter(torch.tensor([[10.],[20.],[30.],[40.]], dtype=torch.float64))
/// opt = torch.optim.SparseAdam([p], lr=0.5, betas=(0.9,0.999), eps=1e-8)
/// p.grad = torch.sparse_coo_tensor(torch.tensor([[2,0,2]]),
///              torch.tensor([[0.3],[5.0],[0.7]], dtype=torch.float64), (4,1))
/// opt.step()   # -> [9.500000031622776, 20.0, 29.50000015811383, 40.0]
/// ```
#[test]
fn divergence_1463_axis3_duplicate_index_summed_not_lastwins() {
    let param = make_param_2d(&[10.0, 20.0, 30.0, 40.0], 4, 1);
    let mut opt = SparseAdam::new(
        vec![param.clone()],
        SparseAdamConfig::default()
            .with_lr(0.5)
            .with_betas((0.9, 0.999))
            .with_eps(1e-8),
    );
    // row 2 listed twice (0.3 then 0.7), row 0 once (5.0).
    let g = SparseGrad::<f64>::new(vec![2, 0, 2], vec![0.3, 5.0, 0.7], vec![1]).unwrap();
    opt.set_sparse_grad(0, 0, g);
    opt.step().unwrap();
    let expected = [9.500000031622776, 20.0, 29.50000015811383, 40.0];
    assert_close(&read(&param), &expected, "axis3 dup-index coalesced sum");
}

/// AXIS #3 control: the explicitly-coalesced equivalent grad (row0=5.0,
/// row2=1.0) must produce the IDENTICAL param to the duplicate case above.
/// If the duplicate case were last-wins (row2=0.7) or double-applied, this
/// control would diverge from it.
///
/// Live torch.optim.SparseAdam 2.11.0 (same config) yields the same param:
/// `[9.500000031622776, 20.0, 29.50000015811383, 40.0]`.
#[test]
fn divergence_1463_axis3_coalesced_equiv_control() {
    let param = make_param_2d(&[10.0, 20.0, 30.0, 40.0], 4, 1);
    let mut opt = SparseAdam::new(
        vec![param.clone()],
        SparseAdamConfig::default()
            .with_lr(0.5)
            .with_betas((0.9, 0.999))
            .with_eps(1e-8),
    );
    let g = SparseGrad::<f64>::new(vec![0, 2], vec![5.0, 1.0], vec![1]).unwrap();
    opt.set_sparse_grad(0, 0, g);
    opt.step().unwrap();
    let expected = [9.500000031622776, 20.0, 29.50000015811383, 40.0];
    assert_close(&read(&param), &expected, "axis3 coalesced-equiv control");
}
