//! Critic audit for wave-G uncommitted builder work.
//!
//! Probes each closed-blocker claim in the wave-G commit-ready files
//! (LowRankMultivariateNormal, MixtureSameFamily, RelaxedBernoulli,
//! RelaxedOneHotCategorical, VonMises) against the upstream PyTorch contract.
//! Each test pins one observable behaviour cited from
//! `/home/doll/pytorch/torch/distributions/*.py`.
//!
//! Tests that FAIL pin a real divergence; tests that PASS confirm the
//! claim is genuinely wired (not vocab-only).

use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

use ferrotorch_distributions::{
    Categorical, Distribution, LowRankMultivariateNormal, MixtureSameFamily, Normal,
    RelaxedBernoulli, RelaxedOneHotCategorical, VonMises,
};

fn cpu_tensor(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

fn cpu_tensor_f64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

// ============================================================================
// #1385/#1386/#1387 LowRank
// ============================================================================

/// REQ-7 / #1386: variance = diag(W W^T) + cov_diag.
/// Upstream `torch/distributions/lowrank_multivariate_normal.py:189-196`:
///   variance = self.cov_factor.pow(2).sum(-1) + self.cov_diag
///
/// Probe: W = [[3,4],[0,0]], D=[1,1]. Σ_00 = 9+16+1 = 26, Σ_11 = 0+0+1 = 1.
#[test]
fn audit_1386_lowrank_variance_matches_diag_formula() {
    let loc = cpu_tensor(&[0.0, 0.0], &[2]);
    let factor = cpu_tensor(&[3.0, 4.0, 0.0, 0.0], &[2, 2]);
    let diag = cpu_tensor(&[1.0, 1.0], &[2]);
    let d = LowRankMultivariateNormal::new(loc, factor, diag).unwrap();
    let v = d.variance().unwrap();
    let data = v.data_vec().unwrap();
    assert!((data[0] - 26.0).abs() < 1e-5, "variance[0] = {}", data[0]);
    assert!((data[1] - 1.0).abs() < 1e-5, "variance[1] = {}", data[1]);
}

/// REQ-8 / #1387: covariance_matrix() returns dense [d,d] Σ = W W^T + diag(D).
/// Probe: W=[[1],[2]] (rank 1), D=[0.5, 0.5]. Σ should be [[1.5, 2], [2, 4.5]].
#[test]
fn audit_1387_lowrank_covariance_matrix_value() {
    let loc = cpu_tensor(&[0.0, 0.0], &[2]);
    let factor = cpu_tensor(&[1.0, 2.0], &[2, 1]);
    let diag = cpu_tensor(&[0.5, 0.5], &[2]);
    let d = LowRankMultivariateNormal::new(loc, factor, diag).unwrap();
    let cov = d.covariance_matrix().unwrap();
    assert_eq!(cov.shape(), &[2, 2]);
    let data = cov.data_vec().unwrap();
    assert!((data[0] - 1.5).abs() < 1e-4, "Σ[0,0] = {}", data[0]);
    assert!((data[1] - 2.0).abs() < 1e-4, "Σ[0,1] = {}", data[1]);
    assert!((data[2] - 2.0).abs() < 1e-4, "Σ[1,0] = {}", data[2]);
    assert!((data[3] - 4.5).abs() < 1e-4, "Σ[1,1] = {}", data[3]);
}

/// REQ-5: mean() returns loc.
#[test]
fn audit_lowrank_mean_equals_loc() {
    let loc = cpu_tensor(&[1.0, -2.0, 3.5], &[3]);
    let factor = cpu_tensor(&[0.1, 0.1, 0.1], &[3, 1]);
    let diag = cpu_tensor(&[1.0, 1.0, 1.0], &[3]);
    let d = LowRankMultivariateNormal::new(loc.clone(), factor, diag).unwrap();
    let m = d.mean().unwrap();
    let data = m.data_vec().unwrap();
    let loc_data = loc.data_vec().unwrap();
    for i in 0..3 {
        assert!((data[i] - loc_data[i]).abs() < 1e-6);
    }
}

// ============================================================================
// #1388 / #1389 MixtureSameFamily
// ============================================================================

/// REQ-9 / #1388: mean = Σ_k π_k * μ_k.
/// Upstream `torch/distributions/mixture_same_family.py:144-153`.
/// Probe: π=[0.25, 0.75], μ=[-1, 3]. Expected: 0.25*-1 + 0.75*3 = 2.0.
#[test]
fn audit_1388_mixture_mean_matches_weighted_sum() {
    let probs = cpu_tensor(&[0.25, 0.75], &[2]);
    let mixing = Categorical::new(probs).unwrap();
    let loc = cpu_tensor(&[-1.0, 3.0], &[2]);
    let scale = cpu_tensor(&[1.0, 1.0], &[2]);
    let components = Normal::new(loc, scale).unwrap();
    let m = MixtureSameFamily::new(mixing, components).unwrap();
    let mean = m.mean().unwrap();
    let val = mean.data_vec().unwrap()[0];
    assert!(
        (val - 2.0).abs() < 1e-5,
        "mixture mean = {val}, expected 2.0"
    );
}

/// REQ-9 / #1388: variance via law of total variance.
/// For N(0,1) + N(2,1), 50/50:
///   E[Var(X|K)] = 1; Var(E[X|K]) = 1; total = 2.
#[test]
fn audit_1388_mixture_variance_law_of_total_variance() {
    let probs = cpu_tensor(&[0.5, 0.5], &[2]);
    let mixing = Categorical::new(probs).unwrap();
    let loc = cpu_tensor(&[0.0, 2.0], &[2]);
    let scale = cpu_tensor(&[1.0, 1.0], &[2]);
    let components = Normal::new(loc, scale).unwrap();
    let m = MixtureSameFamily::new(mixing, components).unwrap();
    let var = m.variance().unwrap();
    let val = var.data_vec().unwrap()[0];
    assert!((val - 2.0).abs() < 1e-5, "variance = {val}, expected 2.0");
}

/// REQ-10 / #1389: cdf = Σ_k π_k * cdf_k(x).
/// Two identical N(0,1) components, 50/50; cdf(0) = 0.5.
#[test]
fn audit_1389_mixture_cdf_weighted_sum() {
    let probs = cpu_tensor(&[0.5, 0.5], &[2]);
    let mixing = Categorical::new(probs).unwrap();
    let loc = cpu_tensor(&[0.0, 0.0], &[2]);
    let scale = cpu_tensor(&[1.0, 1.0], &[2]);
    let components = Normal::new(loc, scale).unwrap();
    let m = MixtureSameFamily::new(mixing, components).unwrap();
    let value = cpu_tensor(&[0.0], &[1]);
    let c = m.cdf(&value).unwrap();
    let v = c.data_vec().unwrap()[0];
    assert!((v - 0.5).abs() < 1e-4, "cdf(0) = {v}, expected 0.5");
}

/// REQ-10 / #1389: cdf asymmetric weights probe.
/// 0.9 * N(0,1).cdf(0) + 0.1 * N(10,1).cdf(0) ≈ 0.45.
#[test]
fn audit_1389_mixture_cdf_asymmetric_weights() {
    let probs = cpu_tensor(&[0.9, 0.1], &[2]);
    let mixing = Categorical::new(probs).unwrap();
    let loc = cpu_tensor(&[0.0, 10.0], &[2]);
    let scale = cpu_tensor(&[1.0, 1.0], &[2]);
    let components = Normal::new(loc, scale).unwrap();
    let m = MixtureSameFamily::new(mixing, components).unwrap();
    let value = cpu_tensor(&[0.0], &[1]);
    let c = m.cdf(&value).unwrap();
    let v = c.data_vec().unwrap()[0];
    assert!(
        (v - 0.45).abs() < 1e-3,
        "asymmetric cdf(0) = {v}, expected ≈ 0.45"
    );
}

// ============================================================================
// #1411 RelaxedBernoulli mean/variance/expand sanity
// ============================================================================

#[test]
fn audit_1411_relaxed_bernoulli_has_rsample() {
    let probs = cpu_tensor(&[0.3], &[1]);
    let d = RelaxedBernoulli::new(0.5_f32, probs).unwrap();
    assert!(d.has_rsample(), "RelaxedBernoulli should have_rsample=true");
}

/// Upstream `relaxed_bernoulli.py:145`: support = unit_interval.
#[test]
fn audit_1411_relaxed_bernoulli_support_unit_interval() {
    let probs = cpu_tensor(&[0.3], &[1]);
    let d = RelaxedBernoulli::new(0.5_f32, probs).unwrap();
    let s = d.support().expect("must have a support");
    assert_eq!(s.name(), "UnitInterval");
}

#[test]
fn audit_1411_relaxed_bernoulli_expand_broadcasts() {
    let probs = cpu_tensor(&[0.3], &[1]);
    let d = RelaxedBernoulli::new(0.5_f32, probs).unwrap();
    let expanded = d.expand(&[4]).unwrap();
    assert_eq!(expanded.batch_shape(), vec![4]);
}

// ============================================================================
// #1422 RelaxedOneHotCategorical
// ============================================================================

#[test]
fn audit_1422_relaxed_one_hot_support_simplex() {
    let probs = cpu_tensor(&[0.3, 0.7], &[2]);
    let d = RelaxedOneHotCategorical::new(0.5_f32, probs).unwrap();
    let s = d.support().expect("must have a support");
    assert_eq!(s.name(), "Simplex");
    assert!(d.has_rsample());
}

/// Upstream `torch/distributions/relaxed_categorical.py:158-160`:
///   logits = log(normalised probs).
#[test]
fn audit_1422_relaxed_one_hot_logits_match_log_normalized() {
    let probs = cpu_tensor(&[1.0, 1.0, 2.0], &[3]); // normalized -> [0.25, 0.25, 0.5]
    let d = RelaxedOneHotCategorical::new(0.5_f32, probs).unwrap();
    let l = d.logits().unwrap();
    let data = l.data_vec().unwrap();
    assert!((data[0] - 0.25_f32.ln()).abs() < 1e-5);
    assert!((data[1] - 0.25_f32.ln()).abs() < 1e-5);
    assert!((data[2] - 0.5_f32.ln()).abs() < 1e-5);
}

// ============================================================================
// #1432 / #1433 VonMises sampler workspace RNG + Taylor fallback
// ============================================================================

/// REQ-11 / #1432: VonMises::sample must honour `ferrotorch_core::manual_seed`.
#[test]
fn audit_1432_von_mises_sample_honours_manual_seed() {
    ferrotorch_core::manual_seed(0xC0FFEE);
    let d1 = VonMises::new(cpu_tensor_f64(&[0.0], &[1]), cpu_tensor_f64(&[2.0], &[1])).unwrap();
    let s1 = d1.sample(&[50]).unwrap();
    let v1 = s1.data().unwrap().to_vec();

    ferrotorch_core::manual_seed(0xC0FFEE);
    let d2 = VonMises::new(cpu_tensor_f64(&[0.0], &[1]), cpu_tensor_f64(&[2.0], &[1])).unwrap();
    let s2 = d2.sample(&[50]).unwrap();
    let v2 = s2.data().unwrap().to_vec();

    assert_eq!(
        v1, v2,
        "VonMises::sample must be deterministic under manual_seed"
    );
}

/// REQ-12 / #1433: small-kappa Taylor fallback must terminate.
/// Upstream `torch/distributions/von_mises.py:170-171`.
#[test]
fn audit_1433_von_mises_small_kappa_terminates() {
    let d = VonMises::new(cpu_tensor_f64(&[0.0], &[1]), cpu_tensor_f64(&[1e-8], &[1])).unwrap();
    let s = d.sample(&[20]).unwrap();
    let pi = std::f64::consts::PI;
    for &v in s.data().unwrap() {
        assert!(
            (-pi..=pi).contains(&v),
            "small-kappa sample out of [-pi, pi]: {v}"
        );
    }
}

/// REQ-13 / #1434: entropy = log(2π) + log I_0(κ) − κ * I_1(κ)/I_0(κ).
///
/// NOTE: PyTorch's `torch.distributions.VonMises.entropy` raises
/// `NotImplementedError` (verified by `python -c "import torch; ...
/// d.entropy()"`). ferrotorch's entropy override is an R-DEV-7
/// enhancement, NOT a parity claim, so the reference is derived from
/// the named closed-form formula using authoritative Bessel constants
/// from `scipy.special.i0/i1`:
///   I_0(1) = 1.26606587775..., I_1(1) = 0.56515910399...
///   I_1/I_0 = 0.4463899658...
///   log(2π) = 1.8378770664..., log I_0 = 0.2359143585...
///   H exact = 1.6274014590...
#[test]
fn audit_1434_von_mises_entropy_exact_ratio() {
    let d = VonMises::new(cpu_tensor_f64(&[0.0], &[1]), cpu_tensor_f64(&[1.0], &[1])).unwrap();
    let h = d.entropy().unwrap();
    let val = h.data().unwrap()[0];
    assert!(
        (val - 1.6274014590).abs() < 1e-2,
        "VonMises entropy κ=1: expected ~1.6274 from closed-form, got {val}"
    );
}
