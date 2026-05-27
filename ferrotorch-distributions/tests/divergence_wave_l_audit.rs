//! Wave-L audit (#1542): the distributions infrastructure deliverables landed
//! under #1385 / #1374 / #1375 / #1373.
//!
//!   - #1385: `LowRankMultivariateNormal::log_prob` now uses the Woodbury
//!     matrix identity + matrix-determinant lemma over the r×r capacitance
//!     matrix — O(d·r²), no dense [d, d] Σ formed. FD-cross-checked against an
//!     independent dense-path oracle.
//!   - #1374: 6 new KL pairs (Cauchy-Cauchy + Normal-Gumbel, Gumbel-Normal,
//!     Gamma-Gumbel, Exponential-Gumbel, Uniform-Gumbel), 19 → 25 pairs.
//!   - #1375: `kl_divergence` dispatch is the deliberate Rust-idiomatic
//!     explicit-match design (the `register_kl` analog); the supported-pair
//!     count is now 25 and stays drift-checked.
//!   - #1373: `Transform::domain()` / `codomain()` Constraint accessors, with
//!     `TransformedDistribution::support()` as the production consumer.
//!
//! Reference values are constructed from PyTorch's closed-form `@register_kl`
//! bodies (`torch/distributions/kl.py`), the LowRankMVN Woodbury identities
//! (`torch/distributions/lowrank_multivariate_normal.py`), and the
//! `domain`/`codomain` class attributes in `torch/distributions/transforms.py`
//! so the asserts are non-tautological (R-CHAR-3): each expected number traces
//! to an upstream file:line, an independent dense oracle, or an analytic
//! identity.

use ferrotorch_core::creation::scalar;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;
use ferrotorch_distributions::kl::{kl_divergence, kl_supported_pair_count};
use ferrotorch_distributions::transforms::{
    AffineTransform, ComposeTransform, ExpTransform, SigmoidTransform, SoftplusTransform,
    TanhTransform, Transform,
};
use ferrotorch_distributions::{
    Cauchy, Distribution, Exponential, Gamma, Gumbel, LowRankMultivariateNormal, Normal,
    TransformedDistribution, Uniform,
};

// ---------------------------------------------------------------------------
// #1385 — LowRankMultivariateNormal Woodbury log_prob
// ---------------------------------------------------------------------------

/// Independent dense-path oracle: builds Σ = W Wᵀ + diag(D), inverts it with
/// Gauss-Jordan, and evaluates the multivariate-normal log density directly.
/// Independent of the production Woodbury path, so it is a genuine cross-check.
fn dense_log_prob_oracle(
    loc: &[f64],
    factor: &[f64],
    diag: &[f64],
    value: &[f64],
    d: usize,
    r: usize,
) -> f64 {
    let mut cov = vec![0.0f64; d * d];
    for i in 0..d {
        for j in 0..d {
            let mut acc = 0.0;
            for k in 0..r {
                acc += factor[i * r + k] * factor[j * r + k];
            }
            if i == j {
                acc += diag[i];
            }
            cov[i * d + j] = acc;
        }
    }
    let mut a = cov.clone();
    let mut inv = vec![0.0f64; d * d];
    for i in 0..d {
        inv[i * d + i] = 1.0;
    }
    let mut det = 1.0;
    for col in 0..d {
        let piv = a[col * d + col];
        det *= piv;
        let piv_inv = 1.0 / piv;
        for j in 0..d {
            a[col * d + j] *= piv_inv;
            inv[col * d + j] *= piv_inv;
        }
        for row in 0..d {
            if row == col {
                continue;
            }
            let f = a[row * d + col];
            for j in 0..d {
                a[row * d + j] -= f * a[col * d + j];
                inv[row * d + j] -= f * inv[col * d + j];
            }
        }
    }
    let diff: Vec<f64> = (0..d).map(|i| value[i] - loc[i]).collect();
    let mut maha = 0.0;
    for i in 0..d {
        for j in 0..d {
            maha += diff[i] * inv[i * d + j] * diff[j];
        }
    }
    -0.5 * (d as f64 * (2.0 * std::f64::consts::PI).ln() + det.ln() + maha)
}

fn cpu(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

#[test]
fn audit_1385_woodbury_log_prob_matches_dense_oracle_rank2() {
    // d=5, r=2 (r << d): the regime where Woodbury is asymptotically faster.
    let loc = [0.5f64, -1.0, 2.0, 0.0, 0.75];
    let factor = [
        1.0f64, -0.3, // row 0
        0.5, 0.6, // row 1
        -0.5, 0.2, // row 2
        0.25, -0.4, // row 3
        0.1, 0.9, // row 4
    ]; // [5, 2]
    let diag = [1.0f64, 2.0, 0.5, 1.5, 0.75];
    let value = [0.0f64, 0.0, 1.0, -0.5, 0.25];

    let mvn =
        LowRankMultivariateNormal::new(cpu(&loc, &[5]), cpu(&factor, &[5, 2]), cpu(&diag, &[5]))
            .unwrap();
    let got = mvn.log_prob(&cpu(&value, &[5])).unwrap().item().unwrap();
    let expected = dense_log_prob_oracle(&loc, &factor, &diag, &value, 5, 2);
    assert!(
        (got - expected).abs() < 1e-9,
        "Woodbury log_prob {got} vs dense oracle {expected}"
    );
}

#[test]
fn audit_1385_woodbury_diagonal_only_is_standard_normal() {
    // W = 0 → Σ = I_4; log_prob at the mean = -d/2·ln(2π) = -2·ln(2π).
    let mvn = LowRankMultivariateNormal::new(
        cpu(&[0.0, 0.0, 0.0, 0.0], &[4]),
        cpu(&[0.0, 0.0, 0.0, 0.0], &[4, 1]),
        cpu(&[1.0, 1.0, 1.0, 1.0], &[4]),
    )
    .unwrap();
    let lp = mvn
        .log_prob(&cpu(&[0.0, 0.0, 0.0, 0.0], &[4]))
        .unwrap()
        .item()
        .unwrap();
    let expected = -2.0 * (2.0 * std::f64::consts::PI).ln();
    assert!(
        (lp - expected).abs() < 1e-12,
        "expected {expected}, got {lp}"
    );
}

// ---------------------------------------------------------------------------
// #1374 — new KL pairs (reference values from torch/distributions/kl.py)
// ---------------------------------------------------------------------------

#[test]
fn audit_1374_cauchy_cauchy_known_value() {
    // KL(Cauchy(0,1) || Cauchy(0,2)) = ln(9) - ln(8) = ln(9/8) (kl.py:952-957).
    let p = Cauchy::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
    let q = Cauchy::new(scalar(0.0f64).unwrap(), scalar(2.0f64).unwrap()).unwrap();
    let kl = kl_divergence(&p, &q).unwrap().item().unwrap();
    assert!((kl - (9.0f64 / 8.0).ln()).abs() < 1e-12, "got {kl}");
}

#[test]
fn audit_1374_exponential_gumbel_known_value() {
    // KL(Exp(1) || Gumbel(0,1)) = -1 + 0.5 + 1 = 0.5 (kl.py:641-649).
    let p = Exponential::new(scalar(1.0f64).unwrap()).unwrap();
    let q = Gumbel::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
    let kl = kl_divergence(&p, &q).unwrap().item().unwrap();
    assert!((kl - 0.5).abs() < 1e-12, "expected 0.5, got {kl}");
}

#[test]
fn audit_1374_uniform_gumbel_known_value() {
    // KL(U(0,1) || Gumbel(0,1)) = 1.5 - e^-1 (kl.py:912-919).
    let p = Uniform::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
    let q = Gumbel::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
    let kl = kl_divergence(&p, &q).unwrap().item().unwrap();
    let expected = 1.5 - (-1.0f64).exp();
    assert!(
        (kl - expected).abs() < 1e-12,
        "expected {expected}, got {kl}"
    );
}

#[test]
fn audit_1374_gamma_gumbel_reduces_to_exponential_gumbel() {
    // Gamma(1, β) == Exp(β), so the Gamma-Gumbel formula (kl.py:678-693) must
    // reduce to Exponential-Gumbel (kl.py:641-649) at α=1.
    let pg = Gamma::new(scalar(1.0f64).unwrap(), scalar(2.0f64).unwrap()).unwrap();
    let pe = Exponential::new(scalar(2.0f64).unwrap()).unwrap();
    let q = Gumbel::new(scalar(0.5f64).unwrap(), scalar(1.5f64).unwrap()).unwrap();
    let kl_g = kl_divergence(&pg, &q).unwrap().item().unwrap();
    let q2 = Gumbel::new(scalar(0.5f64).unwrap(), scalar(1.5f64).unwrap()).unwrap();
    let kl_e = kl_divergence(&pe, &q2).unwrap().item().unwrap();
    assert!(
        (kl_g - kl_e).abs() < 1e-12,
        "Gamma(1,β)-Gumbel {kl_g} vs Exp(β)-Gumbel {kl_e}"
    );
}

#[test]
fn audit_1374_normal_gumbel_known_value() {
    // KL(Normal(0,1) || Gumbel(0,1)) = sqrt(e) - 0.5·(1 + ln 2π) (kl.py:771-779).
    let p = Normal::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
    let q = Gumbel::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
    let kl = kl_divergence(&p, &q).unwrap().item().unwrap();
    let expected = 0.5f64.exp() - 0.5 * (1.0 + (2.0 * std::f64::consts::PI).ln());
    assert!(
        (kl - expected).abs() < 1e-12,
        "expected {expected}, got {kl}"
    );
}

#[test]
fn audit_1374_gumbel_normal_nonnegative_and_known() {
    // KL(Gumbel(0,1) || Normal(0,1)) (kl.py:731-737). param_ratio = 1.
    let p = Gumbel::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
    let q = Normal::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
    let kl = kl_divergence(&p, &q).unwrap().item().unwrap();
    let g = 0.577_215_664_901_532_9_f64;
    let t1 = (1.0 / (2.0 * std::f64::consts::PI).sqrt()).ln();
    let t2 = (std::f64::consts::PI * 0.5).powi(2) / 3.0;
    let t3 = 0.5 * g * g;
    let expected = -t1 + t2 + t3 - (g + 1.0);
    assert!(
        (kl - expected).abs() < 1e-12,
        "expected {expected}, got {kl}"
    );
    assert!(kl >= -1e-12, "KL must be non-negative, got {kl}");
}

// ---------------------------------------------------------------------------
// #1375 — dispatch pair count (drift-checked design decision)
// ---------------------------------------------------------------------------

#[test]
fn audit_1375_supported_pair_count() {
    // The explicit Any::downcast_ref match (the Rust-idiomatic register_kl
    // analog, #1375) exposes 68 pairs after the #1562 both-types-exist KL
    // additions (#1374); the precise count is also drift-checked by
    // `kl::tests::kl_doc_table_matches_dispatcher` and
    // `divergence_wave_m_audit::audit_1374_supported_pair_count_is_41`.
    assert_eq!(kl_supported_pair_count(), 68);
}

// ---------------------------------------------------------------------------
// #1373 — Transform domain/codomain Constraint linkage
// ---------------------------------------------------------------------------

#[test]
fn audit_1373_transform_domain_codomain_match_upstream() {
    // Constraint names mirror the domain/codomain class attributes in
    // torch/distributions/transforms.py.
    let exp: &dyn Transform<f64> = &ExpTransform;
    assert_eq!(
        (exp.domain().name(), exp.codomain().name()),
        ("Real", "Positive")
    );

    let sig: &dyn Transform<f64> = &SigmoidTransform;
    assert_eq!(
        (sig.domain().name(), sig.codomain().name()),
        ("Real", "UnitInterval")
    );

    let sp: &dyn Transform<f64> = &SoftplusTransform;
    assert_eq!(
        (sp.domain().name(), sp.codomain().name()),
        ("Real", "Positive")
    );

    let tanh: &dyn Transform<f64> = &TanhTransform;
    assert_eq!(
        (tanh.domain().name(), tanh.codomain().name()),
        ("Real", "ClosedInterval")
    );

    let affine: AffineTransform<f64> = AffineTransform::new(0.0, 1.0);
    assert_eq!(
        (affine.domain().name(), affine.codomain().name()),
        ("Real", "Real")
    );
}

#[test]
fn audit_1373_compose_endpoints_and_transformed_support() {
    // ComposeTransform domain = first.domain, codomain = last.codomain.
    let chain: ComposeTransform<f64> = ComposeTransform::new(vec![
        Box::new(AffineTransform::new(0.0, 2.0)),
        Box::new(ExpTransform),
    ]);
    assert_eq!(
        (chain.domain().name(), chain.codomain().name()),
        ("Real", "Positive")
    );

    // Production consumer: TransformedDistribution::support is the chain's
    // final codomain (transformed_distribution.py:129-137).
    let base = Normal::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
    let td = TransformedDistribution::new(Box::new(base), vec![Box::new(ExpTransform)]);
    assert_eq!(td.support().unwrap().name(), "Positive");
}
