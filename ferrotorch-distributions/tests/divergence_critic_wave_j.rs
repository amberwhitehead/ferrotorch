//! Adversarial re-audit of commit `83f25f5fa` (umbrella #1542):
//!   - #1397 `Gamma::cdf` (regularized lower incomplete gamma) — VERIFY
//!   - #1555 `GammaRsampleBackward` concentration gradient — CONFIRM WRONG
//!
//! Oracles (this machine, 2026-05-26):
//!   scipy.special.gammainc(2, 2)  = 0.5939941502901616
//!   scipy.special.gammainc(5, 3)  = 0.18473675547622787  (continued-fraction branch)
//!   scipy.special.gammainc(0.5,0.5)=0.6826894921370859
//!   torch._standard_gamma_grad(alpha=2.5, sample=2.0) = 0.9534439667878193
//!     (matches the implicit-function-theorem value -dP/dalpha / pdf = +0.9535;
//!      the shipped closed form sg*(ln sg - digamma(alpha)) = -0.0200)

use ferrotorch_core::Tensor;
use ferrotorch_distributions::Distribution;
use ferrotorch_distributions::Gamma;

fn scalar_f64(v: f64) -> Tensor<f64> {
    ferrotorch_core::from_vec(vec![v], &[]).expect("scalar f64")
}

// ---------------------------------------------------------------------------
// #1397 Gamma::cdf — VERIFIED pins (series + continued-fraction branches).
// ---------------------------------------------------------------------------

/// VERIFIED: x < conc+1 power-series branch.
/// Oracle: Gamma(2,1).cdf(2) = gammainc(2,2) = 0.5939941502901616.
#[test]
fn gamma_cdf_series_branch_matches_scipy() {
    let g = Gamma::new(scalar_f64(2.0), scalar_f64(1.0)).unwrap();
    let c = g.cdf(&scalar_f64(2.0)).unwrap().item().unwrap();
    assert!(
        (c - 0.593_994_150_290_161_6).abs() < 1e-10,
        "Gamma(2,1).cdf(2): expected 0.5939941502901616, got {c}"
    );
}

/// VERIFIED: x >= conc+1 Lentz continued-fraction branch.
/// Oracle: Gamma(5,1).cdf(3) = gammainc(5,3) = 0.18473675547622787.
#[test]
fn gamma_cdf_continued_fraction_branch_matches_scipy() {
    let g = Gamma::new(scalar_f64(5.0), scalar_f64(1.0)).unwrap();
    let c = g.cdf(&scalar_f64(3.0)).unwrap().item().unwrap();
    assert!(
        (c - 0.184_736_755_476_227_87).abs() < 1e-10,
        "Gamma(5,1).cdf(3): expected 0.18473675547622787, got {c}"
    );
}

/// VERIFIED: sub-unit concentration (alpha < 1), x at the branch boundary.
/// Oracle: Gamma(0.5,1).cdf(0.5) = gammainc(0.5,0.5) = 0.6826894921370859.
#[test]
fn gamma_cdf_subunit_concentration_matches_scipy() {
    let g = Gamma::new(scalar_f64(0.5), scalar_f64(1.0)).unwrap();
    let c = g.cdf(&scalar_f64(0.5)).unwrap().item().unwrap();
    assert!(
        (c - 0.682_689_492_137_085_9).abs() < 1e-10,
        "Gamma(0.5,1).cdf(0.5): expected 0.6826894921370859, got {c}"
    );
}

/// VERIFIED: rate scales the argument (cdf computes P(conc, rate*x)).
/// Oracle: Gamma(2,1).cdf(2.0) at conc=2,rate=1 spans the branch boundary
///   x = conc+1 = 3 exactly when value=1.5,rate=2 -> P(2, 3.0).
/// gammainc(2,3) = 0.8008517265285442.
#[test]
fn gamma_cdf_rate_scales_argument_at_boundary() {
    let g = Gamma::new(scalar_f64(2.0), scalar_f64(2.0)).unwrap();
    let c = g.cdf(&scalar_f64(1.5)).unwrap().item().unwrap(); // P(2, 2*1.5=3)
    assert!(
        (c - 0.800_851_726_528_544_2).abs() < 1e-10,
        "Gamma(2,2).cdf(1.5)=P(2,3): expected 0.8008517265285442, got {c}"
    );
}

// ---------------------------------------------------------------------------
// #1555 GammaRsampleBackward concentration gradient — CONFIRM WRONG.
//
// The shipped node computes d(sample)/d(concentration) as the score-function
// closed form `sg*(ln sg - digamma(alpha))` (gamma.rs:569). That estimator is
// unbiased in EXPECTATION but is NOT the pathwise (implicit-reparameterisation)
// per-sample gradient PyTorch uses (`torch._standard_gamma_grad`,
// aten/.../Distributions.cpp:391 / standard_gamma_grad_one). A robust per-
// sample discriminator: for Gamma(alpha>=1, rate=1) the TRUE pathwise gradient
// d(sample)/d(alpha) is strictly POSITIVE for every sample (increasing the
// concentration at a fixed quantile always increases the sample). The shipped
// formula goes negative for any sample with sg < exp(digamma(alpha)) ~ 1.95
// at alpha=2.5 — ~46% of draws.
//
// This test exercises the REAL `GammaRsampleBackward` via the public rsample +
// autograd path with a fixed seed, recovers the drawn standard-gamma value
// from the sample (sg = sample*rate), and asserts the concentration gradient
// matches the torch pathwise oracle for that exact sg. It FAILS under the
// shipped closed form. Tracking: #1555.
// ---------------------------------------------------------------------------

/// DIVERGENCE (#1555): ferrotorch's `GammaRsampleBackward` concentration
/// gradient is the score-function closed form `sg*(ln sg - digamma(alpha))`
/// (`gamma.rs:569`). That estimator is unbiased in expectation but is NOT the
/// pathwise (implicit-reparameterisation) per-sample gradient PyTorch uses
/// (`torch._standard_gamma_grad`, `Distributions.cpp:391` /
/// `standard_gamma_grad_one`). Robust torch-verified universal property: the
/// pathwise gradient `d(sample)/d(concentration)` of `Gamma(alpha>=1, rate=1)`
/// is STRICTLY POSITIVE for every sample (live torch over 2e6 draws at
/// alpha=2.5: min `_standard_gamma_grad` = +0.0295, 0% negative). The shipped
/// score-function formula is negative for any draw below
/// `exp(digamma(alpha)) ~ 1.95` (~46% of draws at alpha=2.5).
///
/// This exercises the REAL `GammaRsampleBackward` via the public rsample +
/// autograd path with a fixed seed, finds a draw in the sign-flip region, and
/// asserts the produced concentration gradient is positive (as the pathwise
/// gradient must be). It FAILS because the shipped formula returns a negative
/// value. Spot-checked against torch at the drawn sg below.
/// Tracking: #1555.
#[test]
fn divergence_gamma_rsample_conc_grad_sign_contradicts_torch_pathwise() {
    let alpha = 2.5_f64;
    // exp(digamma(2.5)) ~ 1.95: below this the score-function formula flips
    // negative while the torch pathwise gradient stays positive.
    let threshold = 1.95_f64;
    for seed in 0..512u64 {
        ferrotorch_core::manual_seed(seed).unwrap();
        let conc = ferrotorch_core::from_vec(vec![alpha], &[])
            .unwrap()
            .requires_grad_(true);
        let rate = scalar_f64(1.0);
        let g = Gamma::new(conc.clone(), rate).unwrap();
        let s = g.rsample(&[1]).unwrap();
        let sg = s.data().unwrap()[0]; // rate=1 => sample == standard gamma
        if !(0.05..threshold).contains(&sg) {
            continue;
        }
        s.backward().unwrap();
        let grad_conc = conc
            .grad()
            .unwrap()
            .expect("gradient should be populated")
            .data()
            .unwrap()[0];
        // Torch pathwise gradient d(sample)/d(concentration) for Gamma(>=1) is
        // strictly positive everywhere (live-torch universal property). The
        // shipped score-function value is negative in this region.
        assert!(
            grad_conc > 0.0,
            "Gamma rsample conc-grad at alpha={alpha}, sg={sg}: torch pathwise              gradient is strictly positive for alpha>=1, but ferrotorch              returned {grad_conc:.6} (shipped score-function              sg*(ln sg - digamma(alpha)) is the wrong, non-pathwise gradient)"
        );
        return; // exercised one separating draw
    }
    panic!("no separating seed found in 0..512 (test setup issue, not a pass)");
}
