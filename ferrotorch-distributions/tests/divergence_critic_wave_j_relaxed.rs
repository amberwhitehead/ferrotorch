//! Critic audit (wave-J, #1542) for commit ef271cc1d:
//! RelaxedBernoulli rsample (#1420), ExpRelaxedCategorical (#1424),
//! RelaxedOneHot rsample / Gumbel-softmax (#1425), batched probs (#1426),
//! Mixture multi-event-dim (#1390).
//!
//! The builder's own probes only assert that `probs.grad()` is `Some` + finite.
//! That is necessary but NOT sufficient: a plausible-but-wrong reparameterization
//! gradient is also `Some` + finite (the #1555 GammaRsampleBackward failure mode).
//! These tests FINITE-DIFFERENCE the actual gradient VALUES against the forward
//! function, with the random noise pinned by `manual_seed`, so a wrong custom
//! GradFn is caught. To guard against a gradient that is only *coincidentally*
//! correct at a single operating point, every grad check sweeps several
//! (probs, temperature) points — including skewed probs where
//! `z(1-z)/(p(1-p))` and `z_i(δ_im - z_m)/p_m` vary strongly.
//!
//! Noise-path equivalence (load-bearing for FD validity):
//!   - RelaxedBernoulli `sample` and `rsample` both call `relaxed_bernoulli_sample`
//!     with the SAME `rand(&[n])` draw → identical noise under a fixed seed.
//!   - RelaxedOneHot `rsample` = `softmax(scores)`; `sample` = `exp(scores - lse)`
//!     via the ExpRelaxedCategorical composition. Both draw `num_rows*k` uniforms
//!     in the SAME `[i*k+j]` order and `exp(s-lse) == softmax(s)`, so the FD
//!     forward (via `sample`) sees the SAME noise as the analytic `rsample`.
//!
//! Upstream contracts:
//!   torch/distributions/relaxed_bernoulli.py:104-112   (LogitRelaxedBernoulli.rsample)
//!   torch/distributions/relaxed_categorical.py:87-94   (ExpRelaxedCategorical.rsample)
//!   torch/distributions/relaxed_categorical.py:142-147 (RelaxedOneHotCategorical = ExpTransform∘…)
//!   torch/distributions/mixture_same_family.py:100-217 (multi-event-dim _pad)
//!
//! All expected values are constructed either from the hand-derived analytic
//! forward map (named symbolic constants traceable to the upstream formula) or
//! by finite-difference of the SAME forward — never literal-copied from the
//! ferrotorch backward (R-CHAR-3).

use ferrotorch_core::autograd::graph::backward_with_grad;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

use ferrotorch_distributions::{
    Categorical, Distribution, ExpRelaxedCategorical, Independent, MixtureSameFamily, Normal,
    RelaxedBernoulli, RelaxedOneHotCategorical,
};

const SEED: u64 = 0xA11CE_u64;

fn leaf(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .unwrap()
        .requires_grad_(true)
}

fn plain(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

/// One-hot seed selecting output element `idx` of a tensor with `n` elements.
fn one_hot_seed(n: usize, idx: usize, shape: &[usize]) -> Tensor<f32> {
    let mut s = vec![0.0_f32; n];
    s[idx] = 1.0;
    plain(&s, shape)
}

// ===========================================================================
// #1420 RelaxedBernoulli rsample — forward + gradient finite-difference
// ===========================================================================

/// Forward correctness with a FIXED u: the Concrete draw must equal
/// `sigmoid((logit(u) + logit(p)) / temp)` for the first drawn uniform.
///
/// Upstream `relaxed_bernoulli.py:104-112`: the base LogitRelaxedBernoulli draws
/// `y = (logit(u) + logit(p)) / temp`; the RelaxedBernoulli SigmoidTransform maps
/// it to `z = sigmoid(y)`. We pin `u` via manual_seed and compare to the
/// hand-evaluated map. The expected value is computed here from the upstream
/// formula (NOT copied from ferrotorch).
#[test]
fn relaxed_bernoulli_forward_fixed_u_matches_sigmoid_logit_formula() {
    let p_val = 0.3_f32;
    let temp = 2.0_f32;

    // Recover the exact uniform stream the rsample will consume.
    ferrotorch_core::manual_seed(SEED).unwrap();
    let u = ferrotorch_core::creation::rand::<f32>(&[1]).unwrap();
    let u0 = u.data_vec().unwrap()[0];

    // Upstream forward map, hand-evaluated.
    let logit = |x: f32| (x / (1.0 - x)).ln();
    let expected = {
        let y = (logit(u0) + logit(p_val)) / temp;
        1.0 / (1.0 + (-y).exp())
    };

    ferrotorch_core::manual_seed(SEED).unwrap();
    let probs = leaf(&[p_val], &[1]);
    let d = RelaxedBernoulli::new(temp, probs).unwrap();
    let z = d.rsample(&[1]).unwrap();
    let got = z.data_vec().unwrap()[0];

    assert!(
        (got - expected).abs() < 1e-5,
        "RelaxedBernoulli rsample forward: expected sigmoid((logit(u)+logit(p))/temp)={expected}, got {got} (u0={u0})"
    );
}

/// Gradient correctness: d sample / d p, finite-differenced with the SAME u,
/// swept across several (p, temp) operating points.
///
/// The reparameterization gradient is the WHOLE POINT of #1420. We pin u via
/// manual_seed, take the analytic grad through `RelaxedBernoulliRsampleBackward`,
/// then re-seed and re-run the forward at `p ± eps` to form a central
/// finite-difference. A gradient that is only coincidentally right at one point
/// (the #1555 failure mode) is caught by the skewed points (p=0.05, p=0.92).
#[test]
fn relaxed_bernoulli_rsample_grad_matches_finite_difference() {
    let eps = 1e-3_f32;
    let points = [(0.30_f32, 2.0_f32), (0.05, 0.5), (0.92, 1.0), (0.50, 0.3)];

    for &(p_val, temp) in &points {
        // Analytic grad d z0 / d p.
        ferrotorch_core::manual_seed(SEED).unwrap();
        let probs = leaf(&[p_val], &[1]);
        let d = RelaxedBernoulli::new(temp, probs.clone()).unwrap();
        let z = d.rsample(&[1]).unwrap();
        let seed_grad = one_hot_seed(1, 0, &[1]);
        backward_with_grad(&z, Some(&seed_grad)).unwrap();
        let analytic = probs
            .grad()
            .unwrap()
            .expect("probs.grad must be Some")
            .data_vec()
            .unwrap()[0];

        // Central finite-difference of the SAME forward map with the SAME noise.
        let forward = |p: f32| -> f32 {
            ferrotorch_core::manual_seed(SEED).unwrap();
            let probs = plain(&[p], &[1]);
            let d = RelaxedBernoulli::new(temp, probs).unwrap();
            // sample() (not rsample) avoids building a graph; identical math/noise.
            d.sample(&[1]).unwrap().data_vec().unwrap()[0]
        };
        let numeric = (forward(p_val + eps) - forward(p_val - eps)) / (2.0 * eps);

        // Relative tolerance: large grads (skewed p / small temp) need rel scale.
        let tol = 2e-3_f32.max(numeric.abs() * 5e-3);
        assert!(
            (analytic - numeric).abs() < tol,
            "RelaxedBernoulli grad @ (p={p_val}, temp={temp}): analytic={analytic}, finite-diff={numeric} (Δ={}, tol={tol})",
            (analytic - numeric).abs()
        );
    }
}

// ===========================================================================
// #1425 RelaxedOneHot (Gumbel-softmax) rsample — simplex + gradient FD
// ===========================================================================

/// Forward: the Gumbel-softmax draw lies on the simplex (sums to 1).
#[test]
fn relaxed_one_hot_rsample_on_simplex() {
    ferrotorch_core::manual_seed(SEED).unwrap();
    let probs = leaf(&[0.2, 0.3, 0.5], &[3]);
    let d = RelaxedOneHotCategorical::new(0.5_f32, probs).unwrap();
    let z = d.rsample(&[1]).unwrap();
    let data = z.data_vec().unwrap();
    let sum: f32 = data.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-4,
        "Gumbel-softmax rsample must sum to 1, got {sum} ({data:?})"
    );
}

/// Full Jacobian d z_i / d p_m finite-differenced, swept across temperatures
/// and probability skew. Checks both the unnormalized-probs gradient AND that
/// the simplex constraint Σ_i dz_i/dp_m == 0 holds (a softmax output's columns
/// must sum to zero — a quick structural check the FD also exercises).
fn check_relaxed_one_hot_jacobian(p: &[f32], temp: f32) {
    let k = p.len();
    let eps = 1e-3_f32;

    // Analytic Jacobian J[i][m] = d z_i / d p_m.
    let mut analytic = vec![vec![0.0_f32; k]; k];
    for (i, row) in analytic.iter_mut().enumerate() {
        ferrotorch_core::manual_seed(SEED).unwrap();
        let probs = leaf(p, &[k]);
        let d = RelaxedOneHotCategorical::new(temp, probs.clone()).unwrap();
        let z = d.rsample(&[1]).unwrap();
        let seed = one_hot_seed(k, i, &[1, k]);
        backward_with_grad(&z, Some(&seed)).unwrap();
        let g = probs
            .grad()
            .unwrap()
            .expect("grad Some")
            .data_vec()
            .unwrap();
        row.copy_from_slice(&g[..k]);
    }

    let forward = |pp: &[f32]| -> Vec<f32> {
        ferrotorch_core::manual_seed(SEED).unwrap();
        let probs = plain(pp, &[k]);
        let d = RelaxedOneHotCategorical::new(temp, probs).unwrap();
        d.sample(&[1]).unwrap().data_vec().unwrap()
    };
    for m in 0..k {
        let mut pp_p = p.to_vec();
        let mut pp_m = p.to_vec();
        pp_p[m] += eps;
        pp_m[m] -= eps;
        let zp = forward(&pp_p);
        let zm = forward(&pp_m);
        // Column sum check: simplex output ⇒ Σ_i dz_i/dp_m = 0.
        let col_sum: f32 = (0..k).map(|i| analytic[i][m]).sum();
        assert!(
            col_sum.abs() < 5e-3,
            "RelaxedOneHot Jacobian column m={m} @ temp={temp}: Σ_i dz_i/dp_m = {col_sum}, must be ~0 (simplex)"
        );
        for i in 0..k {
            let numeric = (zp[i] - zm[i]) / (2.0 * eps);
            let tol = 3e-3_f32.max(numeric.abs() * 5e-3);
            assert!(
                (analytic[i][m] - numeric).abs() < tol,
                "RelaxedOneHot dz_{i}/dp_{m} @ (p={p:?}, temp={temp}): analytic={}, finite-diff={numeric} (Δ={}, tol={tol})",
                analytic[i][m],
                (analytic[i][m] - numeric).abs()
            );
        }
    }
}

#[test]
fn relaxed_one_hot_rsample_jacobian_matches_finite_difference() {
    check_relaxed_one_hot_jacobian(&[0.2, 0.3, 0.5], 0.7);
    check_relaxed_one_hot_jacobian(&[0.1, 0.1, 0.8], 1.0);
    check_relaxed_one_hot_jacobian(&[0.05, 0.9, 0.05], 0.5);
    check_relaxed_one_hot_jacobian(&[0.25, 0.25, 0.25, 0.25], 0.8);
}

// ===========================================================================
// #1424 ExpRelaxedCategorical — log-simplex forward + gradient FD
// ===========================================================================

/// Forward: rsample returns log-probabilities (≤ 0 after recentering; exp sums
/// to 1). Upstream `relaxed_categorical.py:87-94`:
///   scores = (logits + gumbels)/temp; return scores - scores.logsumexp(-1).
#[test]
fn exp_relaxed_rsample_is_log_simplex() {
    ferrotorch_core::manual_seed(SEED).unwrap();
    let probs = leaf(&[0.2, 0.3, 0.5], &[3]);
    let d = ExpRelaxedCategorical::new(0.5_f32, probs).unwrap();
    let log_z = d.rsample(&[1]).unwrap();
    let data = log_z.data_vec().unwrap();
    let exp_sum: f32 = data.iter().map(|v| v.exp()).sum();
    assert!(
        (exp_sum - 1.0).abs() < 1e-4,
        "ExpRelaxedCategorical rsample: exp(log_z) must sum to 1, got {exp_sum} ({data:?})"
    );
    for &v in &data {
        assert!(
            v <= 1e-5,
            "log-simplex entries must be ≤ 0 (after recentering), got {v}"
        );
    }
}

/// Gradient correctness for the log-space draw: d (log_z_i) / d p_m via FD,
/// swept across temperatures + skew. The output is `scores - logsumexp(scores)`,
/// NOT a softmax — its Jacobian columns do NOT sum to zero (unlike the simplex
/// case), so a GradFn that mistakenly copied the softmax/simplex column-cancel
/// would diverge here.
fn check_exp_relaxed_jacobian(p: &[f32], temp: f32) {
    let k = p.len();
    let eps = 1e-3_f32;

    let mut analytic = vec![vec![0.0_f32; k]; k];
    for (i, row) in analytic.iter_mut().enumerate() {
        ferrotorch_core::manual_seed(SEED).unwrap();
        let probs = leaf(p, &[k]);
        let d = ExpRelaxedCategorical::new(temp, probs.clone()).unwrap();
        let log_z = d.rsample(&[1]).unwrap();
        let seed = one_hot_seed(k, i, &[1, k]);
        backward_with_grad(&log_z, Some(&seed)).unwrap();
        let g = probs
            .grad()
            .unwrap()
            .expect("grad Some")
            .data_vec()
            .unwrap();
        row.copy_from_slice(&g[..k]);
    }

    let forward = |pp: &[f32]| -> Vec<f32> {
        ferrotorch_core::manual_seed(SEED).unwrap();
        let probs = plain(pp, &[k]);
        let d = ExpRelaxedCategorical::new(temp, probs).unwrap();
        d.sample(&[1]).unwrap().data_vec().unwrap()
    };
    for m in 0..k {
        let mut pp_p = p.to_vec();
        let mut pp_m = p.to_vec();
        pp_p[m] += eps;
        pp_m[m] -= eps;
        let zp = forward(&pp_p);
        let zm = forward(&pp_m);
        for i in 0..k {
            let numeric = (zp[i] - zm[i]) / (2.0 * eps);
            let tol = 3e-3_f32.max(numeric.abs() * 5e-3);
            assert!(
                (analytic[i][m] - numeric).abs() < tol,
                "ExpRelaxed d(log_z_{i})/dp_{m} @ (p={p:?}, temp={temp}): analytic={}, finite-diff={numeric} (Δ={}, tol={tol})",
                analytic[i][m],
                (analytic[i][m] - numeric).abs()
            );
        }
    }
}

#[test]
fn exp_relaxed_rsample_jacobian_matches_finite_difference() {
    check_exp_relaxed_jacobian(&[0.2, 0.3, 0.5], 0.7);
    check_exp_relaxed_jacobian(&[0.1, 0.1, 0.8], 1.0);
    check_exp_relaxed_jacobian(&[0.05, 0.9, 0.05], 0.5);
}

// ===========================================================================
// #1426 batched probs — shapes, per-row independence, per-row gradient FD
// ===========================================================================

/// probs shape [B, K] → rsample [1, B, K]; log_prob over a [B, K] value → [B].
/// Each batch row must be independently normalized/correct, AND the per-row
/// reparameterization gradient must finite-difference-match (a shared/aliased
/// row computation would show up as a wrong off-diagonal-row gradient).
#[test]
fn relaxed_one_hot_batched_shapes_independence_and_grad() {
    // Row 0: uniform; row 1: skewed. Unnormalized.
    let p = [1.0_f32, 1.0, 1.0, 1.0, 1.0, 8.0];
    let temp = 0.5_f32;

    ferrotorch_core::manual_seed(SEED).unwrap();
    let probs = leaf(&p, &[2, 3]);
    let d = RelaxedOneHotCategorical::new(temp, probs).unwrap();

    let z = d.rsample(&[1]).unwrap();
    assert_eq!(z.shape(), &[1, 2, 3], "batched rsample shape");
    let zd = z.data_vec().unwrap();
    for row in 0..2 {
        let s: f32 = (0..3).map(|c| zd[row * 3 + c]).sum();
        assert!(
            (s - 1.0).abs() < 1e-4,
            "batch row {row} sum={s} not on simplex"
        );
    }

    // log_prob over a [2,3] value collapses K → [2], rows differ (independence).
    let value = plain(&[0.3, 0.3, 0.4, 0.1, 0.1, 0.8], &[2, 3]);
    let lp = d.log_prob(&value).unwrap();
    assert_eq!(lp.shape(), &[2], "batched log_prob shape");
    let lpd = lp.data_vec().unwrap();
    assert!(
        lpd.iter().all(|v| v.is_finite()),
        "log_prob rows finite: {lpd:?}"
    );
    assert!(
        (lpd[0] - lpd[1]).abs() > 1e-3,
        "batch rows must be computed independently; got identical log_prob {lpd:?}"
    );

    // Per-row gradient FD: perturbing a parameter in row 1 must NOT move row 0's
    // sample (cross-row gradient must be 0). Pick output element z[0,0]; its grad
    // w.r.t. p[1,*] (indices 3,4,5) must be ~0.
    ferrotorch_core::manual_seed(SEED).unwrap();
    let probs2 = leaf(&p, &[2, 3]);
    let d2 = RelaxedOneHotCategorical::new(temp, probs2.clone()).unwrap();
    let z2 = d2.rsample(&[1]).unwrap();
    let seed = one_hot_seed(6, 0, &[1, 2, 3]); // z[0,0,0] (sample, row0, cat0)
    backward_with_grad(&z2, Some(&seed)).unwrap();
    let g = probs2
        .grad()
        .unwrap()
        .expect("grad Some")
        .data_vec()
        .unwrap();
    for (col, &gv) in g[3..6].iter().enumerate() {
        assert!(
            gv.abs() < 1e-6,
            "cross-row gradient leak: d z[row0,c0] / d p[row1,c{col}] = {gv} must be 0"
        );
    }
    // And the within-row gradient (indices 0,1,2) must finite-diff-match.
    let eps = 1e-3_f32;
    let forward_z00 = |pp: &[f32]| -> f32 {
        ferrotorch_core::manual_seed(SEED).unwrap();
        let probs = plain(pp, &[2, 3]);
        let d = RelaxedOneHotCategorical::new(temp, probs).unwrap();
        d.sample(&[1]).unwrap().data_vec().unwrap()[0]
    };
    for m in 0..3 {
        let mut pp_p = p.to_vec();
        let mut pp_m = p.to_vec();
        pp_p[m] += eps;
        pp_m[m] -= eps;
        let numeric = (forward_z00(&pp_p) - forward_z00(&pp_m)) / (2.0 * eps);
        let tol = 3e-3_f32.max(numeric.abs() * 5e-3);
        assert!(
            (g[m] - numeric).abs() < tol,
            "batched within-row grad d z[0,0]/d p[0,{m}]: analytic={}, finite-diff={numeric} (Δ={})",
            g[m],
            (g[m] - numeric).abs()
        );
    }
}

// ===========================================================================
// #1390 Mixture multi-event-dim — event_shape, log_prob shape, mean
// ===========================================================================

/// Mixture of Independent<Normal> with event_shape [D=2]:
///   - event_shape() == [2]
///   - log_prob over [...batch, 2] reduces to [...batch]
///   - mean == sum_k w_k * component_mean_k (per event element)
///
/// Upstream `mixture_same_family.py:100-217`: K is inserted before the event
/// dims, component.log_prob reduces the event dims, then logsumexp over K.
/// mean (155-162) = sum_k mix_probs[k] * component_mean[k].
/// The log_prob value is also cross-checked against a hand-computed mixture
/// density so a wrong _pad axis is caught (not just the shape).
#[test]
fn mixture_multi_event_dim_shapes_mean_and_log_prob() {
    // 2 components, each a 2-dim diagonal Gaussian (Independent<Normal>, ndims=1).
    // Component means: comp0 = (0, 0), comp1 = (10, -10); stds all 1.
    // Layout: components batch shape [K=2, D=2]; flat means row-major.
    let loc = plain(&[0.0, 0.0, 10.0, -10.0], &[2, 2]);
    let scale = plain(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let base = Normal::new(loc, scale).unwrap();
    let comps = Independent::new(base, 1).unwrap();
    assert_eq!(comps.event_shape(), vec![2], "component event_shape [2]");

    let mix_probs = plain(&[0.25, 0.75], &[2]);
    let mixing = Categorical::new(mix_probs).unwrap();
    let mix = MixtureSameFamily::new(mixing, comps).unwrap();

    assert_eq!(mix.event_shape(), vec![2], "mixture event_shape == [D]");

    // log_prob over a single event vector [2] → scalar (batch_shape []).
    let value = plain(&[5.0, -5.0], &[2]);
    let lp = mix.log_prob(&value).unwrap();
    assert_eq!(
        lp.shape().len(),
        0,
        "log_prob of one event-vector must reduce the event dims to batch_shape [], got {:?}",
        lp.shape()
    );

    // Hand-computed expected: mixture density at x=(5,-5).
    //   log N(x; mu_k, I) over D=2 = -0.5*||x-mu_k||^2 - D/2 * log(2π)
    //   comp0 mu=(0,0):   -0.5*(25+25) = -25      ; + (-1.83788) = const
    //   comp1 mu=(10,-10):-0.5*(25+25) = -25      ; symmetric here
    //   log mixture = logsumexp_k( log w_k + log N_k )
    let two_pi = std::f32::consts::TAU;
    let log_norm_const = -(2.0_f32 / 2.0) * two_pi.ln(); // D/2 * log(2π), D=2
    let logn = |x: [f32; 2], mu: [f32; 2]| -> f32 {
        let sq = (x[0] - mu[0]).powi(2) + (x[1] - mu[1]).powi(2);
        -0.5 * sq + log_norm_const
    };
    let l0 = 0.25_f32.ln() + logn([5.0, -5.0], [0.0, 0.0]);
    let l1 = 0.75_f32.ln() + logn([5.0, -5.0], [10.0, -10.0]);
    let m = l0.max(l1);
    let expected_lp = m + ((l0 - m).exp() + (l1 - m).exp()).ln();
    let got_lp = lp.item().unwrap();
    assert!(
        (got_lp - expected_lp).abs() < 1e-3,
        "mixture multi-event log_prob: expected {expected_lp} (hand-computed), got {got_lp}"
    );

    // mean == weighted sum of component means, per event element.
    let mean = mix.mean().unwrap();
    assert_eq!(
        mean.shape(),
        &[2],
        "mixture mean has event_shape [2], got {:?}",
        mean.shape()
    );
    let md = mean.data_vec().unwrap();
    let expect_0 = 0.25_f32 * 0.0 + 0.75 * 10.0;
    let expect_1 = 0.25_f32 * 0.0 + 0.75 * -10.0;
    assert!(
        (md[0] - expect_0).abs() < 1e-4 && (md[1] - expect_1).abs() < 1e-4,
        "mixture mean: expected [{expect_0}, {expect_1}], got {md:?}"
    );
}
