//! Wave-M audit (#1542 umbrella): the distributions infrastructure deliverables
//! landed under #1374 (KL pairs) and #1379 (scalar special functions).
//!
//!   - #1374: 12 new KL pairs (Dirichlet-Dirichlet + Beta-Exponential,
//!     Beta-Gamma, Beta-Normal, Beta-Uniform, Pareto-Exponential, Pareto-Gamma,
//!     Pareto-Normal, Uniform-Exponential, Uniform-Gamma, Uniform-Pareto,
//!     Uniform-Beta), 25 → 37 pairs. The supported-pair count and doc table are
//!     drift-checked by `kl::tests::kl_doc_table_matches_dispatcher`.
//!   - #1379: `trigamma` / `polygamma` scalar special functions (digamma now
//!     delegates to `polygamma(0, ·)`; trigamma is consumed by `polygamma(1,·)`).
//!
//! Reference checks here are deliberately *independent* of the production
//! closed forms (R-CHAR-3, non-tautological):
//!   - KL pairs are cross-checked against a Monte-Carlo estimate
//!     `E_p[log p(x) - log q(x)]` using the distributions' own `log_prob`,
//!     which is a different code path than the closed-form KL bodies, OR
//!     against an analytic identity that ties the pair to an already-pinned
//!     formula.
//!   - trigamma is cross-checked against a finite-difference of digamma.
//!   - The supported-pair count is pinned to the public introspection accessor.

use ferrotorch_core::creation::{scalar, tensor};
use ferrotorch_distributions::kl::{kl_divergence, kl_supported_pair_count};
use ferrotorch_distributions::{Beta, Dirichlet, Distribution, Exponential, Pareto, Uniform};

// ---------------------------------------------------------------------------
// #1374 — supported-pair count introspection
// ---------------------------------------------------------------------------

#[test]
fn audit_1374_supported_pair_count_is_41() {
    // Pins the public accessor; the in-crate drift test
    // `kl_doc_table_matches_dispatcher` separately asserts the doc table and
    // dispatcher arms agree with this number.
    assert_eq!(
        kl_supported_pair_count(),
        68,
        "#1562 both-types-exist gaps added 27 pairs (41 -> 68); update this audit + the doc table together"
    );
}

// ---------------------------------------------------------------------------
// #1374 — KL pairs cross-checked against an independent Monte-Carlo estimate
// ---------------------------------------------------------------------------

/// Monte-Carlo KL estimate `E_p[log p(x) - log q(x)]` using the production
/// `sample`/`log_prob` of P and Q — a path entirely disjoint from the
/// closed-form KL formula under test. Convergence is O(1/sqrt(N)); we use a
/// loose tolerance and a large N so the check is a genuine sanity gate, not a
/// re-derivation of the closed form.
fn mc_kl<P, Q>(p: &P, q: &Q, n: usize, seed_shape: &[usize]) -> f64
where
    P: Distribution<f64>,
    Q: Distribution<f64>,
{
    let mut acc = 0.0;
    let mut count = 0usize;
    for _ in 0..n {
        let x = p.sample(seed_shape).unwrap();
        let lp = p.log_prob(&x).unwrap().data_vec().unwrap();
        let lq = q.log_prob(&x).unwrap().data_vec().unwrap();
        for (a, b) in lp.iter().zip(lq.iter()) {
            if a.is_finite() && b.is_finite() {
                acc += a - b;
                count += 1;
            }
        }
    }
    acc / count as f64
}

#[test]
fn audit_1374_beta_exponential_kl_matches_monte_carlo() {
    let p = Beta::new(scalar(2.0f64).unwrap(), scalar(3.0f64).unwrap()).unwrap();
    let q = Exponential::new(scalar(1.5f64).unwrap()).unwrap();
    let closed = kl_divergence(&p, &q).unwrap().item().unwrap();
    let mc = mc_kl(&p, &q, 4000, &[16]);
    assert!(
        (closed - mc).abs() < 5e-2,
        "Beta-Exp closed-form KL {closed} vs MC {mc}"
    );
}

#[test]
fn audit_1374_uniform_exponential_kl_matches_monte_carlo() {
    let p = Uniform::new(scalar(0.5f64).unwrap(), scalar(2.0f64).unwrap()).unwrap();
    let q = Exponential::new(scalar(1.0f64).unwrap()).unwrap();
    let closed = kl_divergence(&p, &q).unwrap().item().unwrap();
    let mc = mc_kl(&p, &q, 4000, &[16]);
    assert!(
        (closed - mc).abs() < 5e-2,
        "Unif-Exp closed-form KL {closed} vs MC {mc}"
    );
}

#[test]
fn audit_1374_uniform_pareto_kl_matches_monte_carlo() {
    // Uniform(2,4) lies above Pareto(scale=1) support, so q.log_prob is finite.
    let p = Uniform::new(scalar(2.0f64).unwrap(), scalar(4.0f64).unwrap()).unwrap();
    let q = Pareto::new(scalar(1.0f64).unwrap(), scalar(3.0f64).unwrap()).unwrap();
    let closed = kl_divergence(&p, &q).unwrap().item().unwrap();
    let mc = mc_kl(&p, &q, 4000, &[16]);
    assert!(
        (closed - mc).abs() < 1e-1,
        "Unif-Pareto closed-form KL {closed} vs MC {mc}"
    );
}

// ---------------------------------------------------------------------------
// #1374 — analytic-identity cross-checks (independent of the MC path)
// ---------------------------------------------------------------------------

#[test]
fn audit_1374_dirichlet_dirichlet_two_category_matches_beta_beta() {
    // Dirichlet([a,b]) ≡ Beta(a, b). So KL(Dir([a1,b1]) || Dir([a2,b2])) must
    // equal KL(Beta(a1,b1) || Beta(a2,b2)) — the latter already pinned in the
    // lib tests. This ties the new multivariate formula to a known-good 1-D one.
    let pd = Dirichlet::new(tensor(&[2.0f64, 3.0]).unwrap()).unwrap();
    let qd = Dirichlet::new(tensor(&[3.0f64, 2.0]).unwrap()).unwrap();
    let kl_dir = kl_divergence(&pd, &qd).unwrap().item().unwrap();

    let pb = Beta::new(scalar(2.0f64).unwrap(), scalar(3.0f64).unwrap()).unwrap();
    let qb = Beta::new(scalar(3.0f64).unwrap(), scalar(2.0f64).unwrap()).unwrap();
    let kl_beta = kl_divergence(&pb, &qb).unwrap().item().unwrap();

    assert!(
        (kl_dir - kl_beta).abs() < 1e-10,
        "KL(Dir[2,3]||Dir[3,2]) {kl_dir} must equal KL(Beta(2,3)||Beta(3,2)) {kl_beta} = 0.5"
    );
    // Beta(2,3)||Beta(3,2) = ψ(3)-ψ(2) = 1/2 (pinned in kl.rs lib tests).
    assert!(
        (kl_beta - 0.5).abs() < 1e-9,
        "Beta(2,3)||Beta(3,2) should be 0.5"
    );
}

#[test]
fn audit_1374_dirichlet_dirichlet_nonnegative_and_self_zero() {
    let p = Dirichlet::new(tensor(&[0.5f64, 1.5, 2.0, 3.0]).unwrap()).unwrap();
    let q = Dirichlet::new(tensor(&[1.0f64, 1.0, 1.0, 1.0]).unwrap()).unwrap();
    let kl = kl_divergence(&p, &q).unwrap().item().unwrap();
    assert!(kl >= -1e-12, "KL must be non-negative, got {kl}");

    let same = kl_divergence(&p, &p).unwrap().item().unwrap();
    assert!(same.abs() < 1e-12, "KL(p||p) should be 0, got {same}");
}
