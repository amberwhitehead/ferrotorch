//! Critic re-audit of #1573 (commit e579271f6): systemic p-vs-q broadcast across
//! ~37 KL pair functions in `ferrotorch-distributions/src/kl.rs`.
//!
//! The builder converted every finite KL pair from the non-broadcasting
//! `zip().cycle()` pattern to a shared `kl_broadcast_index_pairs` /
//! `KlBroadcastPlan` that mirrors torch's `broadcast_all`
//! (`torch/distributions/utils.py:27-59`) — broadcasting p against q jointly
//! across all parameters and emitting the broadcast output shape. The builder
//! claims NO formula changed (equal-shape p/q byte-identical) and that
//! multi-param distributions broadcast all params jointly correctly.
//!
//! This file pins, with the EXACT live-torch float64 oracle (R-CHAR-3, every
//! expected constant produced by calling `torch.distributions.kl_divergence`
//! in `torch==2.11.0`, NOT copied from the ferrotorch side):
//!
//!  (a) Formula preservation: equal-shape KL values for 11 sampled converted
//!      pairs still match torch (a refactor that subtly broke a closed form
//!      would surface here).
//!  (b) Joint multi-param disjoint broadcast: `Normal([2,1]) || Normal([1,3])`
//!      and `Beta([2,1]) || Beta([1,3])` broadcast to `[2,3]` with the correct
//!      OFF-DIAGONAL values (a wrong stride/index mapping in the (pi,qi) plan
//!      would corrupt off-diagonal positions while leaving the diagonal right).
//!  (c) Scalar-p / batched-q normal broadcasts to q's shape.
//!  (d) The Bernoulli disjoint-broadcast R-DEV-6 claim: torch's
//!      `_kl_bernoulli_bernoulli` (kl.py:204-216) raises IndexError on disjoint
//!      p/q because its masked-assign uses a non-broadcast mask; the builder
//!      claims ferrotorch is "more correct" by returning the element-wise
//!      closed form on `broadcast_tensors(p,q)`. We assert ferrotorch returns
//!      exactly that element-wise form so the R-DEV-6 claim is not masking a
//!      formula error.
//!
//! Oracle script (run under torch 2.11, float64) reproduced inline per block.

use ferrotorch_core::creation::{from_slice, tensor};
use ferrotorch_distributions::kl::kl_divergence;
use ferrotorch_distributions::{
    Bernoulli, Beta, Cauchy, Exponential, Gamma, Gumbel, Laplace, Normal, Pareto, Poisson, Uniform,
};

/// Compare two flat f64 slices elementwise to `tol`.
fn assert_close(got: &[f64], exp: &[f64], tol: f64, name: &str) {
    assert_eq!(
        got.len(),
        exp.len(),
        "{name}: length mismatch got {} exp {}",
        got.len(),
        exp.len()
    );
    for (i, (&g, &e)) in got.iter().zip(exp.iter()).enumerate() {
        assert!(
            (g - e).abs() < tol,
            "{name}[{i}]: ferrotorch={g} torch={e} delta={}",
            (g - e).abs()
        );
    }
}

// =========================================================================
// (a) Formula preservation — equal-shape [1] pairs vs live torch float64.
//
// import torch; torch.set_default_dtype(torch.float64)
// from torch.distributions import *
// kl_divergence(Normal([0.5],[1.5]), Normal([-0.3],[2.0])) etc.
// =========================================================================

#[test]
fn divergence_kl_1573_formula_normal_normal() {
    let p = Normal::new(tensor(&[0.5]).unwrap(), tensor(&[1.5]).unwrap()).unwrap();
    let q = Normal::new(tensor(&[-0.3]).unwrap(), tensor(&[2.0]).unwrap()).unwrap();
    let got = kl_divergence(&p, &q).unwrap().data_vec().unwrap();
    assert_close(&got, &[0.14893207245178092], 1e-9, "normal_normal");
}

#[test]
fn divergence_kl_1573_formula_beta_beta() {
    let p = Beta::new(tensor(&[2.0]).unwrap(), tensor(&[3.0]).unwrap()).unwrap();
    let q = Beta::new(tensor(&[1.5]).unwrap(), tensor(&[2.5]).unwrap()).unwrap();
    let got = kl_divergence(&p, &q).unwrap().data_vec().unwrap();
    assert_close(&got, &[0.02371448006428656], 1e-9, "beta_beta");
}

#[test]
fn divergence_kl_1573_formula_gamma_gamma() {
    let p = Gamma::new(tensor(&[2.0]).unwrap(), tensor(&[1.5]).unwrap()).unwrap();
    let q = Gamma::new(tensor(&[3.0]).unwrap(), tensor(&[2.0]).unwrap()).unwrap();
    let got = kl_divergence(&p, &q).unwrap().data_vec().unwrap();
    assert_close(&got, &[0.07398329477280219], 1e-9, "gamma_gamma");
}

#[test]
fn divergence_kl_1573_formula_laplace_laplace() {
    let p = Laplace::new(tensor(&[0.5]).unwrap(), tensor(&[1.2]).unwrap()).unwrap();
    let q = Laplace::new(tensor(&[-0.4]).unwrap(), tensor(&[0.9]).unwrap()).unwrap();
    let got = kl_divergence(&p, &q).unwrap().data_vec().unwrap();
    assert_close(&got, &[0.34213999786957205], 1e-9, "laplace_laplace");
}

#[test]
fn divergence_kl_1573_formula_exponential_exponential() {
    let p = Exponential::new(tensor(&[1.3]).unwrap()).unwrap();
    let q = Exponential::new(tensor(&[0.7]).unwrap()).unwrap();
    let got = kl_divergence(&p, &q).unwrap().data_vec().unwrap();
    assert_close(&got, &[0.15750074686776205], 1e-9, "exponential_exponential");
}

#[test]
fn divergence_kl_1573_formula_gumbel_gumbel() {
    let p = Gumbel::new(tensor(&[0.4]).unwrap(), tensor(&[1.1]).unwrap()).unwrap();
    let q = Gumbel::new(tensor(&[-0.2]).unwrap(), tensor(&[1.6]).unwrap()).unwrap();
    let got = kl_divergence(&p, &q).unwrap().data_vec().unwrap();
    assert_close(&got, &[0.1922240873303238], 1e-9, "gumbel_gumbel");
}

#[test]
fn divergence_kl_1573_formula_pareto_pareto() {
    let p = Pareto::new(tensor(&[1.2]).unwrap(), tensor(&[3.0]).unwrap()).unwrap();
    let q = Pareto::new(tensor(&[1.0]).unwrap(), tensor(&[2.0]).unwrap()).unwrap();
    let got = kl_divergence(&p, &q).unwrap().data_vec().unwrap();
    assert_close(&got, &[0.43677488836274025], 1e-9, "pareto_pareto");
}

#[test]
fn divergence_kl_1573_formula_cauchy_cauchy() {
    let p = Cauchy::new(tensor(&[0.3]).unwrap(), tensor(&[1.4]).unwrap()).unwrap();
    let q = Cauchy::new(tensor(&[-0.5]).unwrap(), tensor(&[0.8]).unwrap()).unwrap();
    let got = kl_divergence(&p, &q).unwrap().data_vec().unwrap();
    assert_close(&got, &[0.2014820545330307], 1e-9, "cauchy_cauchy");
}

#[test]
fn divergence_kl_1573_formula_uniform_uniform() {
    let p = Uniform::new(tensor(&[0.2]).unwrap(), tensor(&[0.8]).unwrap()).unwrap();
    let q = Uniform::new(tensor(&[0.0]).unwrap(), tensor(&[1.0]).unwrap()).unwrap();
    let got = kl_divergence(&p, &q).unwrap().data_vec().unwrap();
    assert_close(&got, &[0.5108256237659906], 1e-9, "uniform_uniform");
}

#[test]
fn divergence_kl_1573_formula_bernoulli_bernoulli() {
    let p = Bernoulli::new(tensor(&[0.3]).unwrap()).unwrap();
    let q = Bernoulli::new(tensor(&[0.6]).unwrap()).unwrap();
    let got = kl_divergence(&p, &q).unwrap().data_vec().unwrap();
    // torch clamps nothing here (0.3/0.6 are well inside (0,1)); ferrotorch
    // clamps to [1e-7, 1-1e-7] but that does not move these values at 1e-7.
    assert_close(&got, &[0.1837868973868122], 1e-6, "bernoulli_bernoulli");
}

#[test]
fn divergence_kl_1573_formula_poisson_poisson() {
    let p = Poisson::new(tensor(&[2.5]).unwrap()).unwrap();
    let q = Poisson::new(tensor(&[1.8]).unwrap()).unwrap();
    let got = kl_divergence(&p, &q).unwrap().data_vec().unwrap();
    assert_close(&got, &[0.12126016743009016], 1e-9, "poisson_poisson");
}

// =========================================================================
// (b) Joint multi-param disjoint broadcast: p:[2,1] vs q:[1,3] -> [2,3].
//
// p = Normal([[0.],[1.]], [[1.],[2.]])           # shape [2,1]
// q = Normal([[0.5,1.0,1.5]], [[1.0,1.5,2.0]])   # shape [1,3]
// kl_divergence(p,q) -> shape (2,3), values below (row-major flat).
// =========================================================================

#[test]
fn divergence_kl_1573_normal_disjoint_2d_broadcast() {
    let p = Normal::new(
        from_slice(&[0.0, 1.0], &[2, 1]).unwrap(),
        from_slice(&[1.0, 2.0], &[2, 1]).unwrap(),
    )
    .unwrap();
    let q = Normal::new(
        from_slice(&[0.5, 1.0, 1.5], &[1, 3]).unwrap(),
        from_slice(&[1.0, 1.5, 2.0], &[1, 3]).unwrap(),
    )
    .unwrap();
    let out = kl_divergence(&p, &q).unwrap();
    assert_eq!(out.shape(), &[2, 3], "normal disjoint broadcast shape");
    let got = out.data_vec().unwrap();
    // Off-diagonal values are the load-bearing check: a wrong (pi,qi) mapping
    // corrupts these while the diagonal might still look right.
    let exp = [
        0.125,
        0.3499095525526088,
        0.5993971805599453,
        0.9318528194400547,
        0.10120681643710794,
        0.03125,
    ];
    assert_close(&got, &exp, 1e-9, "normal_disjoint_2d");
}

#[test]
fn divergence_kl_1573_beta_disjoint_2d_broadcast() {
    // p = Beta([[1.5],[2.5]], [[2.0],[1.0]])           # [2,1]
    // q = Beta([[1.0,2.0,3.0]], [[1.0,1.5,0.5]])       # [1,3]
    let p = Beta::new(
        from_slice(&[1.5, 2.5], &[2, 1]).unwrap(),
        from_slice(&[2.0, 1.0], &[2, 1]).unwrap(),
    )
    .unwrap();
    let q = Beta::new(
        from_slice(&[1.0, 2.0, 3.0], &[1, 3]).unwrap(),
        from_slice(&[1.0, 1.5, 0.5], &[1, 3]).unwrap(),
    )
    .unwrap();
    let out = kl_divergence(&p, &q).unwrap();
    assert_eq!(out.shape(), &[2, 3], "beta disjoint broadcast shape");
    let got = out.data_vec().unwrap();
    let exp = [
        0.10805020110220975,
        0.19314718055994543,
        1.965735902799727,
        0.316290731874155,
        0.23472104466522353,
        0.3406431002383383,
    ];
    assert_close(&got, &exp, 1e-9, "beta_disjoint_2d");
}

// =========================================================================
// (c) Scalar-p / batched-q: p:[1] vs q:[3] -> [3].
//
// p = Normal([0.0],[1.0]); q = Normal([0.5,1.0,1.5],[1.0,1.5,2.0])
// =========================================================================

#[test]
fn divergence_kl_1573_normal_scalarp_batchedq() {
    let p = Normal::new(tensor(&[0.0]).unwrap(), tensor(&[1.0]).unwrap()).unwrap();
    let q = Normal::new(
        tensor(&[0.5, 1.0, 1.5]).unwrap(),
        tensor(&[1.0, 1.5, 2.0]).unwrap(),
    )
    .unwrap();
    let out = kl_divergence(&p, &q).unwrap();
    assert_eq!(out.shape(), &[3], "scalar-p batched-q shape");
    let got = out.data_vec().unwrap();
    assert_close(
        &got,
        &[0.125, 0.3499095525526088, 0.5993971805599453],
        1e-9,
        "normal_scalarp_batchedq",
    );
}

// =========================================================================
// (d) Bernoulli disjoint-broadcast R-DEV-6 claim.
//
// torch's _kl_bernoulli_bernoulli (kl.py:204-216) RAISES IndexError on a
// disjoint p:[2,1]/q:[1,3] broadcast (verified live: "The shape of the mask
// [1, 3] ... does not match the shape of the indexed tensor [2, 3]").
// The builder claims ferrotorch is "more correct" by returning the
// element-wise closed form on broadcast_tensors(p,q). We pin ferrotorch to
// exactly that element-wise oracle so the R-DEV-6 claim cannot be hiding a
// formula error.
//
// pv=[[0.3],[0.6]]; qv=[[0.4,0.5,0.7]]
// pb,qb = broadcast_tensors(pv,qv)
// kl = pb*log(pb/qb) + (1-pb)*log((1-pb)/(1-qb))   # [2,3]
// =========================================================================

#[test]
fn divergence_kl_1573_bernoulli_disjoint_2d_broadcast() {
    let p = Bernoulli::new(from_slice(&[0.3, 0.6], &[2, 1]).unwrap()).unwrap();
    let q = Bernoulli::new(from_slice(&[0.4, 0.5, 0.7], &[1, 3]).unwrap()).unwrap();
    let out = kl_divergence(&p, &q).unwrap();
    assert_eq!(out.shape(), &[2, 3], "bernoulli disjoint broadcast shape");
    let got = out.data_vec().unwrap();
    let exp = [
        0.02160085414354654,
        0.08228287850505178,
        0.33891914415488134,
        0.08109302162163282,
        0.020135513550688863,
        0.022582421084357415,
    ];
    assert_close(&got, &exp, 1e-6, "bernoulli_disjoint_2d");
}
