//! Adversarial divergence tests for the K-FAC natural-gradient optimizer
//! (`ferrotorch-optim/src/natural_gradient.rs`, #1467).
//!
//! K-FAC has no `torch.optim` counterpart (R-DEV-7 ecosystem add), so the
//! correctness contract is the closed-form K-FAC math (Martens & Grosse 2015,
//! arXiv:1503.05671) plus the KFAC-PyTorch reference impl
//! (github.com/alecwangcq/KFAC-Pytorch, `KFACOptimizer`). These tests pin
//! divergences from that contract that the builder's friendly tests miss.
//!
//! All expected values are derived from the closed-form K-FAC definitions
//! (R-CHAR-3 — never copied from the ferrotorch side):
//!   * The Kronecker factor `A` is the curvature estimate `E[a aᵀ]`. For a
//!     single mini-batch it is the *unbiased* batch covariance
//!     `A_batch = (1/N) Σ aᵢ aᵢᵀ`. KFAC-PyTorch seeds the running stat with
//!     this first estimate (`self.m_aa[...] = aa` on the first step) precisely
//!     so the factor is never biased low.
//!   * `stat_decay` (KFAC-PyTorch default 0.95) controls the factor EMA and is
//!     a SEPARATE hyperparameter from the gradient `momentum` (default 0.9).

use std::collections::HashMap;

use ferrotorch_core::{Tensor, TensorStorage};
use ferrotorch_nn::Parameter;
use ferrotorch_optim::optimizer::{Optimizer, OptimizerState};
use ferrotorch_optim::{Kfac, KfacConfig};

fn act_2x3() -> Tensor<f64> {
    // a = [[1,0,0],[0,1,0]]  ->  a^T a = diag(1,1,0)  ->  A_batch = diag(0.5,0.5,0)
    Tensor::from_storage(
        TensorStorage::cpu(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        vec![2, 3],
        false,
    )
    .unwrap()
}

fn grad_2x2() -> Tensor<f64> {
    // g = [[1,0],[0,1]]  ->  g^T g = I_2  ->  G_batch = diag(0.5,0.5)
    Tensor::from_storage(
        TensorStorage::cpu(vec![1.0, 0.0, 0.0, 1.0]),
        vec![2, 2],
        false,
    )
    .unwrap()
}

/// Read the stored `a_factor` from the optimizer's `state_dict` (the only
/// public observable surface for the factor values).
fn a_factor_of(opt: &Kfac<f64>, key: &str) -> Vec<f64> {
    let sd: OptimizerState = opt.state_dict().unwrap();
    sd.get(key)
        .and_then(|e: &HashMap<String, Vec<f64>>| e.get("a_factor"))
        .cloned()
        .unwrap_or_default()
}

/// Divergence: `ferrotorch-optim/src/natural_gradient.rs:330-368`
/// (`Kfac::update_factors`) zero-initializes the running Kronecker factor and
/// then applies the EMA `A = momentum*A_old + (1-momentum)*A_batch` with
/// `A_old = 0`. With the default `momentum = 0.9` the factor after the FIRST
/// mini-batch is `0.1 * A_batch` — biased 10x low — and there is no
/// bias-correction term.
///
/// K-FAC requires `A` to estimate the curvature `E[a aᵀ]`; a single mini-batch
/// is the unbiased estimate `A_batch`. KFAC-PyTorch seeds the running stat with
/// the first batch's estimate so the factor is never down-scaled. ferrotorch
/// returns `0.1 * A_batch` instead, so the preconditioner on early steps is
/// off by ~10x in `A` (and another ~10x in `G`).
///
/// Closed-form expected: A_batch = (a^T a)/N with a = [[1,0,0],[0,1,0]], N=2
///   A_batch = diag(0.5, 0.5, 0.0). Element (0,0) = 0.5.
/// ferrotorch returns 0.1 * 0.5 = 0.05.
/// Tracking: #1588
#[test]
#[ignore = "divergence: Kfac zero-init EMA biases factors ~10x low on early steps; tracking #1588"]
fn divergence_kfac_ema_zero_init_biases_factor_low() {
    let p = Parameter::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let config = KfacConfig::default(); // momentum (== EMA decay here) = 0.9
    let mut kfac = Kfac::new(vec![p], config);

    kfac.update_factors("layer", &act_2x3(), &grad_2x2()).unwrap();

    let a = a_factor_of(&kfac, "layer");
    assert_eq!(a.len(), 9, "A must be [3,3]");
    // Closed-form unbiased single-batch curvature estimate: A_batch[0,0] = 0.5.
    let expected_a00 = 0.5_f64;
    assert!(
        (a[0] - expected_a00).abs() < 1e-9,
        "K-FAC factor A after one mini-batch must equal the unbiased batch \
         covariance A_batch[0,0] = {expected_a00} (E[a aᵀ]); ferrotorch's \
         zero-init EMA returns {} (= 0.1 * A_batch, biased 10x low)",
        a[0]
    );
}

/// Divergence: `Kfac::update_factors` uses the single `momentum`
/// (`KfacConfig::momentum`) field as the factor-EMA decay
/// (`natural_gradient.rs:322,361-368`) AND `Kfac::step` uses the SAME field as
/// the gradient-momentum buffer coefficient (`natural_gradient.rs:547-582`).
///
/// In the K-FAC reference (KFAC-PyTorch `KFACOptimizer`) these are two
/// independent hyperparameters: `stat_decay` (default 0.95) for the factor EMA
/// and `momentum` (default 0.9) for the parameter update. ferrotorch cannot
/// represent `stat_decay != momentum`, so any configuration matching the
/// reference defaults is unreachable.
///
/// We pin the observable consequence: setting `momentum` to a value chosen for
/// the gradient buffer (say 0.0, i.e. no gradient momentum) forces the factor
/// EMA decay to 0.0 as well, meaning the factor is OVERWRITTEN each step
/// (`A = 0*A_old + 1*A_batch`) instead of being a running average. A true
/// running-average factor (stat_decay > 0) blends across batches.
///
/// Closed-form: with a real EMA decay d>0 and two DIFFERENT batches B1, B2,
/// A_after_two retains a contribution from B1. With ferrotorch forced to
/// momentum=0 (because the user wanted NO gradient momentum),
/// A_after_two = B2 exactly — B1 is forgotten. The factor is not a running
/// average at all.
/// Tracking: #1589
#[test]
#[ignore = "divergence: Kfac momentum doubles as stat_decay; cannot match K-FAC reference; tracking #1589"]
fn divergence_kfac_momentum_doubles_as_stat_decay() {
    // User wants NO gradient momentum -> sets momentum = 0.0.
    let p = Parameter::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let config = KfacConfig::default().with_momentum(0.0);
    let mut kfac = Kfac::new(vec![p], config);

    // Batch 1: a1 = [[2,0,0],[0,0,0]] -> a1^T a1 = diag(4,0,0) -> /2 = diag(2,0,0)
    let a1 = Tensor::from_storage(
        TensorStorage::cpu(vec![2.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        vec![2, 3],
        false,
    )
    .unwrap();
    // Batch 2: a2 = [[0,0,0],[0,2,0]] -> a2^T a2 = diag(0,4,0) -> /2 = diag(0,2,0)
    let a2 = Tensor::from_storage(
        TensorStorage::cpu(vec![0.0, 0.0, 0.0, 0.0, 2.0, 0.0]),
        vec![2, 3],
        false,
    )
    .unwrap();
    let g = grad_2x2();

    kfac.update_factors("layer", &a1, &g).unwrap();
    kfac.update_factors("layer", &a2, &g).unwrap();

    let a = a_factor_of(&kfac, "layer");
    // A RUNNING AVERAGE must retain some contribution from batch 1, so A[0,0]
    // (the channel batch 1 excited) must be strictly > 0. The reference with
    // stat_decay = 0.95 (independent of the zero gradient momentum) keeps
    // A[0,0] > 0 regardless. ferrotorch, forced to decay = momentum = 0,
    // overwrites: A[0,0] = 0 (batch 2 has nothing in channel 0).
    assert!(
        a[0] > 1e-9,
        "factor must be a running average retaining batch-1 curvature in \
         channel 0 (A[0,0] > 0); ferrotorch forgets batch 1 because the EMA \
         decay is tied to the gradient momentum (=0 here), giving A[0,0] = {}",
        a[0]
    );
}

/// Independent re-verification of the Kronecker ORDERING for a NON-SQUARE
/// weight (out != in), with DIFFERENT factors than the builder's
/// `test_kronecker_identity_matches_dense_fisher`.
///
/// The identity is `(A ⊗ G)^{-1} vec(W) = vec(G^{-1} W A^{-1})`. A transposed
/// ⊗ ordering (G ⊗ A) passes a symmetric (out==in) test but fails when
/// out != in. We drive a FULL `step()` with hand-set factors and a known
/// gradient, and assert the parameter update equals the closed-form
/// `lr * G^{-1} @ grad @ A^{-1}` computed independently via dense linear
/// algebra on the host. Uses out=2, in=4 (clearly non-square).
///
/// AUDIT RESULT: this test PASSES — the ⊗ ordering is correct for non-square
/// weights. Retained as a non-tautological no-divergence audit artifact.
#[test]
fn divergence_kfac_step_nonsquare_kronecker_ordering() {
    use ferrotorch_core::creation::eye;
    use ferrotorch_core::creation::scalar;
    use ferrotorch_core::grad_fns::arithmetic::{add, mul};
    use ferrotorch_core::linalg::solve;
    use ferrotorch_core::ops::linalg::matmul;

    let out = 2usize;
    let inn = 4usize;
    let lr = 1.0_f64; // unit lr so update == preconditioned grad
    let damping = 0.3_f64;

    // W: [out,in] = [2,4], initialized to zero.
    let w0: Vec<f64> = vec![0.0; out * inn];
    let p = Parameter::from_slice(&w0, &[out, inn]).unwrap();

    // grad_W: [out,in], asymmetric values.
    let grad_w: Vec<f64> = (1..=(out * inn)).map(|v| v as f64).collect();
    p.tensor()
        .set_grad(Some(
            Tensor::from_storage(TensorStorage::cpu(grad_w.clone()), vec![out, inn], false).unwrap(),
        ))
        .unwrap();

    // Activation a (batch=4, in=4) chosen so a^T a = diag(2,3,4,5),
    // A_batch = diag(2,3,4,5)/4 = diag(0.5,0.75,1.0,1.25).
    let mut a_rows = vec![0.0; 4 * inn];
    let diag_a = [2.0_f64, 3.0, 4.0, 5.0];
    for k in 0..inn {
        a_rows[k * inn + k] = diag_a[k].sqrt();
    }
    let act = Tensor::from_storage(TensorStorage::cpu(a_rows), vec![4, inn], false).unwrap();

    // Output gradient g (batch=4, out=2): g^T g = [[6,4],[4,6]] -> /4 = [[1.5,1],[1,1.5]].
    let g_rows = vec![1.0, 0.0, 2.0, 1.0, 1.0, 2.0, 0.0, 1.0];
    let outgrad = Tensor::from_storage(TensorStorage::cpu(g_rows), vec![4, out], false).unwrap();

    // momentum=0 so the EMA overwrites: A==A_batch, G==G_batch exactly. The
    // zero-init / stat_decay divergence is orthogonal and pinned separately.
    let config = KfacConfig::default()
        .with_lr(lr)
        .with_damping(damping)
        .with_momentum(0.0)
        .with_update_freq(1)
        .with_weight_decay(0.0);
    let mut kfac = Kfac::new(vec![p], config);
    // step() keys factors as "g{group}_p{param}"; param 0 in group 0 -> "g0_p0".
    kfac.update_factors("g0_p0", &act, &outgrad).unwrap();
    kfac.step().unwrap();

    let updated: Vec<f64> = kfac.param_groups()[0].params()[0]
        .tensor()
        .data_vec()
        .unwrap();

    // Closed-form expected: with W0 = 0, lr = 1:
    //   W_new = - (G_d^{-1} @ grad @ A_d^{-1}),  A_d = A_batch + λI, G_d = G_batch + λI.
    let a_batch = Tensor::from_storage(
        TensorStorage::cpu({
            let mut m = vec![0.0; inn * inn];
            let d = [0.5, 0.75, 1.0, 1.25];
            for k in 0..inn {
                m[k * inn + k] = d[k];
            }
            m
        }),
        vec![inn, inn],
        false,
    )
    .unwrap();
    let g_batch = Tensor::from_storage(
        TensorStorage::cpu(vec![1.5, 1.0, 1.0, 1.5]),
        vec![out, out],
        false,
    )
    .unwrap();
    let damp_a = mul(&eye::<f64>(inn).unwrap(), &scalar(damping).unwrap()).unwrap();
    let a_d = add(&a_batch, &damp_a).unwrap();
    let damp_g = mul(&eye::<f64>(out).unwrap(), &scalar(damping).unwrap()).unwrap();
    let g_d = add(&g_batch, &damp_g).unwrap();
    let a_inv = solve(&a_d, &eye::<f64>(inn).unwrap()).unwrap();
    let g_inv = solve(&g_d, &eye::<f64>(out).unwrap()).unwrap();
    let grad_t =
        Tensor::from_storage(TensorStorage::cpu(grad_w.clone()), vec![out, inn], false).unwrap();
    let temp = matmul(&g_inv, &grad_t).unwrap(); // [out,in]
    let precond = matmul(&temp, &a_inv).unwrap(); // [out,in]
    let precond_d = precond.data_vec().unwrap();
    let expected: Vec<f64> = precond_d.iter().map(|&v| -lr * v).collect();

    assert_eq!(updated.len(), expected.len());
    for (i, (u, e)) in updated.iter().zip(expected.iter()).enumerate() {
        assert!(
            (u - e).abs() < 1e-7,
            "non-square (out={out},in={inn}) Kronecker ordering: element {i} \
             expected {e} (= -lr * G_d^-1 grad A_d^-1), got {u}"
        );
    }
}
