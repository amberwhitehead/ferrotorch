//! R-FIX-4 re-audit of the K-FAC factor-EMA fix (commit e0aa11306, #1588 +
//! #1589 under umbrella #1467) in `ferrotorch-optim/src/natural_gradient.rs`.
//!
//! K-FAC has no `torch.optim` counterpart (R-DEV-7 ecosystem add). The
//! correctness contract is the closed-form K-FAC running-stat math from the
//! KFAC-PyTorch reference impl (github.com/alecwangcq/KFAC-Pytorch,
//! `KFACOptimizer.update_running_stat`):
//!
//! ```python
//! def update_running_stat(aa, m_aa, stat_decay):
//!     m_aa *= stat_decay / (1 - stat_decay)
//!     m_aa += aa
//!     m_aa *= (1 - stat_decay)        # => m_aa = stat_decay*m_aa + (1-stat_decay)*aa
//! # seeded with `self.m_aa[module] = aa` when steps == 0
//! ```
//!
//! These tests independently re-derive the expected factor values from that
//! recursion (R-CHAR-3 — never copied from the ferrotorch side) and cover the
//! cases the builder's in-file tests did NOT exercise:
//!   * seed exactness with a NON-DIAGONAL covariance (catches an `a @ a^T` vs
//!     `a^T @ a` transpose bug that symmetric-diagonal batches hide),
//!   * the EXACT blend direction over 4 batches on BOTH the A and G factors,
//!   * `stat_decay = 0.0` => pure overwrite (no factor memory) while momentum
//!     is independent,
//!   * `stat_decay`/`momentum` non-cross-contamination at the observable level.
//!
//! The only public observable for the stored factor is `state_dict`, so each
//! test reads the factor matrices back through `Kfac::state_dict`.

use std::collections::HashMap;

use ferrotorch_core::{Tensor, TensorStorage};
use ferrotorch_nn::Parameter;
use ferrotorch_optim::optimizer::{Optimizer, OptimizerState};
use ferrotorch_optim::{Kfac, KfacConfig};

/// Reference seed-then-EMA recursion, computed independently on the host.
/// `seeds[0]` seeds the factor; each subsequent entry blends with `decay`.
fn ref_ema(decay: f64, batches: &[f64]) -> f64 {
    let mut acc = batches[0];
    for &b in &batches[1..] {
        acc = decay * acc + (1.0 - decay) * b;
    }
    acc
}

fn factor_of(opt: &Kfac<f64>, key: &str, which: &str) -> Vec<f64> {
    let sd: OptimizerState = opt.state_dict().unwrap();
    sd.get(key)
        .and_then(|e: &HashMap<String, Vec<f64>>| e.get(which))
        .cloned()
        .unwrap_or_default()
}

/// Build a `[batch, features]` activation tensor from explicit row data.
fn act(rows: Vec<f64>, batch: usize, features: usize) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(rows), vec![batch, features], false).unwrap()
}

/// SEED EXACTNESS WITH A NON-DIAGONAL COVARIANCE (re-audit of #1588).
///
/// The builder's `test_update_factors_stores_running_averages` only uses
/// orthonormal rows whose `a^T a` is diagonal — a transposed accumulation
/// (`a @ a^T`, shape [batch,batch]) would either error on shape or coincide on
/// the diagonal. Here `a` has correlated columns so the true factor `A_batch =
/// (a^T a)/N` is a FULL symmetric matrix with non-zero off-diagonals; we pin
/// every entry.
///
/// a = [[1, 2],
///      [3, 4]]   (batch=2, in=2)
/// a^T a = [[1*1+3*3, 1*2+3*4],
///          [2*1+4*3, 2*2+4*4]] = [[10, 14],[14, 20]]
/// A_batch = (a^T a)/2 = [[5, 7],[7, 10]]
///
/// The first `update_factors` call must SEED A = A_batch exactly (no
/// down-scaling by stat_decay), per KFAC-PyTorch `m_aa = aa` at steps == 0.
#[test]
fn kfac_seed_first_update_nondiagonal_exact() {
    let p = Parameter::from_slice(&[0.0; 4], &[2, 2]).unwrap();
    // stat_decay deliberately != 1: if the seed were (1-decay)*A_batch we'd see
    // it. Any decay value must NOT scale the seed.
    let config = KfacConfig::default().with_stat_decay(0.95);
    let mut kfac = Kfac::new(vec![p], config);

    let a = act(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    // g chosen with a non-diagonal g^T g too: g = [[1,1],[1,-1]] (batch=2,out=2)
    // g^T g = [[1*1+1*1, 1*1+1*(-1)],[ ... , 1*1+(-1)*(-1)]] = [[2,0],[0,2]]
    // (that one IS diagonal; pick a correlated g instead)
    // g = [[1,2],[3,4]] -> g^T g = [[10,14],[14,20]] -> /2 = [[5,7],[7,10]]
    let g = act(vec![1.0, 2.0, 3.0, 4.0], 2, 2);

    kfac.update_factors("layer", &a, &g).unwrap();

    let a_fac = factor_of(&kfac, "layer", "a_factor");
    let g_fac = factor_of(&kfac, "layer", "g_factor");
    // Independent closed form: A_batch = (a^T a)/2 = [[5,7],[7,10]].
    let expected = [5.0_f64, 7.0, 7.0, 10.0];
    assert_eq!(a_fac.len(), 4, "A must be [2,2]");
    assert_eq!(g_fac.len(), 4, "G must be [2,2]");
    for (i, &e) in expected.iter().enumerate() {
        assert!(
            (a_fac[i] - e).abs() < 1e-9,
            "seed A[{i}] must equal unbiased batch covariance (a^T a)/N = {e}, got {}",
            a_fac[i]
        );
        assert!(
            (g_fac[i] - e).abs() < 1e-9,
            "seed G[{i}] must equal unbiased batch covariance (g^T g)/N = {e}, got {}",
            g_fac[i]
        );
    }
}

/// EXACT 4-BATCH BLEND DIRECTION ON BOTH A AND G (re-audit of #1589 / point 2).
///
/// The builder's `test_update_factors_multistep_ema_recursion` checks 3 batches
/// on A[0,0] only. This drives 4 distinct batches with a non-round
/// `stat_decay = 0.7` and verifies BOTH factors track the seed-then-EMA
/// recursion `A_k = decay*A_{k-1} + (1-decay)*A_batch_k`. A common bug —
/// swapping the blend so the NEW term gets `decay` (`A = decay*new +
/// (1-decay)*old`) — would diverge sharply after 4 steps with these values.
#[test]
fn kfac_multistep_ema_exact_blend_both_factors() {
    let decay = 0.7_f64;
    let p = Parameter::from_slice(&[0.0; 4], &[2, 2]).unwrap();
    let config = KfacConfig::default().with_stat_decay(decay);
    let mut kfac = Kfac::new(vec![p], config);

    // Per-batch [0,0] covariances for A: a[0,0] = sqrt(2*c) so (a^T a)[0,0]/2 = c.
    let a_cs = [1.0_f64, 4.0, 0.25, 9.0];
    // Per-batch [0,0] covariances for G: g[0,0] = sqrt(2*d) so (g^T g)[0,0]/2 = d.
    let g_ds = [2.0_f64, 0.5, 5.0, 1.5];

    for k in 0..4 {
        let av = (2.0 * a_cs[k]).sqrt();
        let gv = (2.0 * g_ds[k]).sqrt();
        // a = [[av,0],[0,0]] -> (a^T a)/2 = diag(c_k, 0)
        let a = act(vec![av, 0.0, 0.0, 0.0], 2, 2);
        // g = [[gv,0],[0,0]] -> (g^T g)/2 = diag(d_k, 0)
        let g = act(vec![gv, 0.0, 0.0, 0.0], 2, 2);
        kfac.update_factors("layer", &a, &g).unwrap();
    }

    let a_fac = factor_of(&kfac, "layer", "a_factor");
    let g_fac = factor_of(&kfac, "layer", "g_factor");
    let expect_a = ref_ema(decay, &a_cs);
    let expect_g = ref_ema(decay, &g_ds);
    assert!(
        (a_fac[0] - expect_a).abs() < 1e-9,
        "A[0,0] seed-then-EMA(decay={decay}) over {a_cs:?} must be {expect_a}, got {}",
        a_fac[0]
    );
    assert!(
        (g_fac[0] - expect_g).abs() < 1e-9,
        "G[0,0] seed-then-EMA(decay={decay}) over {g_ds:?} must be {expect_g}, got {}",
        g_fac[0]
    );
}

/// `stat_decay = 0.0` => PURE OVERWRITE (no factor memory), independent of
/// `momentum` (point 3).
///
/// With decay=0 the recursion `A = 0*A_old + 1*A_batch` discards all history:
/// after batch 2 the factor equals A_batch_2 exactly, with NO contribution from
/// batch 1. We set `momentum = 0.9` simultaneously to prove the factor EMA does
/// NOT read momentum (if it did, decay would be 0.9 and batch 1 would survive).
#[test]
fn kfac_stat_decay_zero_overwrites_factor_momentum_independent() {
    let p = Parameter::from_slice(&[0.0; 4], &[2, 2]).unwrap();
    // decay 0 (overwrite) but momentum 0.9 (gradient buffer) — they must not
    // cross-contaminate.
    let config = KfacConfig::default().with_stat_decay(0.0).with_momentum(0.9);
    let mut kfac = Kfac::new(vec![p], config);

    // Batch 1 excites channel 0: a1 = [[2,0],[0,0]] -> (a^T a)/2 = diag(2,0).
    let a1 = act(vec![2.0, 0.0, 0.0, 0.0], 2, 2);
    // Batch 2 excites channel 1: a2 = [[0,0],[0,2]] -> (a^T a)/2 = diag(0,2).
    let a2 = act(vec![0.0, 0.0, 0.0, 2.0], 2, 2);
    let g = act(vec![1.0, 0.0, 0.0, 1.0], 2, 2);

    kfac.update_factors("layer", &a1, &g).unwrap();
    kfac.update_factors("layer", &a2, &g).unwrap();

    let a_fac = factor_of(&kfac, "layer", "a_factor");
    // decay=0 overwrite: A == A_batch_2 = diag(0, 2). Channel 0 forgotten.
    assert!(
        a_fac[0].abs() < 1e-9,
        "stat_decay=0 must overwrite: A[0,0] (batch-1 channel) must be 0, got {} \
         (if it read momentum=0.9 instead it'd be 0.9*2=1.8)",
        a_fac[0]
    );
    assert!(
        (a_fac[3] - 2.0).abs() < 1e-9,
        "stat_decay=0 overwrite: A[1,1] must equal A_batch_2[1,1] = 2.0, got {}",
        a_fac[3]
    );
}

/// `stat_decay > 0` retains curvature even when `momentum = 0` (point 3,
/// complementary direction).
///
/// With `momentum = 0` (no gradient buffer) but `stat_decay = 0.9`, the factor
/// must STILL be a running average: after seeding with batch 1 then blending
/// batch 2, channel 0 (excited only by batch 1) must retain `0.9 * 2 = 1.8`.
/// This proves the factor EMA reads `stat_decay`, not `momentum`.
#[test]
fn kfac_stat_decay_retains_curvature_with_zero_momentum() {
    let p = Parameter::from_slice(&[0.0; 4], &[2, 2]).unwrap();
    let decay = 0.9_f64;
    let config = KfacConfig::default()
        .with_stat_decay(decay)
        .with_momentum(0.0);
    let mut kfac = Kfac::new(vec![p], config);

    // Batch 1: channel 0 covariance = 2. Batch 2: channel 0 covariance = 0.
    let a1 = act(vec![2.0, 0.0, 0.0, 0.0], 2, 2);
    let a2 = act(vec![0.0, 0.0, 0.0, 2.0], 2, 2);
    let g = act(vec![1.0, 0.0, 0.0, 1.0], 2, 2);

    kfac.update_factors("layer", &a1, &g).unwrap();
    kfac.update_factors("layer", &a2, &g).unwrap();

    let a_fac = factor_of(&kfac, "layer", "a_factor");
    // Independent: seed=2, then blend with 0: 0.9*2 + 0.1*0 = 1.8.
    let expected = ref_ema(decay, &[2.0, 0.0]);
    assert!(
        (a_fac[0] - expected).abs() < 1e-9,
        "stat_decay={decay} EMA must retain batch-1 curvature A[0,0] = {expected} \
         even with momentum=0, got {}",
        a_fac[0]
    );
}

/// DEFAULTS (point 4): `KfacConfig::default()` yields `stat_decay = 0.95`,
/// `momentum = 0.9`, two distinct values. These are the KFAC-PyTorch defaults
/// (`stat_decay` 0.95 / `momentum` 0.9). Pinned against literal reference
/// constants traceable to the KFAC-PyTorch `__init__` signature (R-CHAR-3 (b)).
#[test]
fn kfac_config_defaults_stat_decay_and_momentum_distinct() {
    // KFAC-PyTorch KFACOptimizer.__init__ defaults.
    const KFAC_PYTORCH_STAT_DECAY: f64 = 0.95;
    const KFAC_PYTORCH_MOMENTUM: f64 = 0.9;
    let c = KfacConfig::default();
    assert!(
        (c.stat_decay - KFAC_PYTORCH_STAT_DECAY).abs() < 1e-12,
        "default stat_decay must be {KFAC_PYTORCH_STAT_DECAY}, got {}",
        c.stat_decay
    );
    assert!(
        (c.momentum - KFAC_PYTORCH_MOMENTUM).abs() < 1e-12,
        "default momentum must be {KFAC_PYTORCH_MOMENTUM}, got {}",
        c.momentum
    );
    assert!(
        (c.stat_decay - c.momentum).abs() > 1e-6,
        "stat_decay and momentum must be DISTINCT defaults (0.95 vs 0.9); \
         got stat_decay={} momentum={}",
        c.stat_decay,
        c.momentum
    );
}
