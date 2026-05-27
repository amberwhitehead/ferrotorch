//! Divergence audit of commit `630273caf` (KL pairs 41->68, #1374 / #1562).
//!
//! The #1562 work registered 24 support-mismatch `+inf` arms via the new
//! `kl_infinite_like` helper (mirroring `_infinite_like` at
//! `pytorch torch/distributions/kl.py:141-145`). PyTorch registers
//! `(Normal, Uniform)` in EXACTLY that support-mismatch family:
//!
//!   pytorch torch/distributions/kl.py:761-768
//!     @register_kl(Normal, Beta)
//!     @register_kl(Normal, ContinuousBernoulli)
//!     @register_kl(Normal, Exponential)
//!     @register_kl(Normal, Gamma)
//!     @register_kl(Normal, Pareto)
//!     @register_kl(Normal, Uniform)          <-- line 766
//!     def _kl_normal_infinity(p, q):
//!         return _infinite_like(p.loc)        <-- line 768  => +inf everywhere
//!
//! A Normal's support is all of R, which is NOT contained in a Uniform's
//! bounded support [low, high]; KL(Normal || Uniform) is therefore `+inf`
//! at every point (the Normal places mass outside [low, high] where the
//! Uniform density is 0). Live torch 2.11.0 confirms:
//!     >>> kl_divergence(Normal(0., 1.), Uniform(-2., 2.)).item()
//!     inf
//!
//! But ferrotorch ships a PRE-EXISTING finite arm `kl_normal_uniform`
//! (ferrotorch-distributions/src/kl.rs:725) that computes
//!     -entropy(Normal) + ln(high - low)
//! which for Normal(0,1)/Uniform(-2,2) evaluates to ~ -0.03264 (note: even
//! NEGATIVE, which a true KL can never be). The #1562 builder explicitly
//! routed Normal-{Beta,Exponential,Gamma,Pareto} through `kl_infinite_like`
//! but left `Normal-Uniform` pointing at the wrong finite formula, with the
//! doc note "Normal-Uniform is registered separately above as a finite
//! cross-family formula" -- that registration contradicts upstream.
//!
//! The OPPOSITE direction `(Uniform, Normal)` IS legitimately finite in torch
//! (`kl.py:925-932`, Uniform's bounded support is contained in Normal's R), so
//! `kl_uniform_normal` is correct and is NOT touched here.
//!
//! Reference value from live `torch.distributions.kl_divergence` at float64
//! (torch 2.11.0, 2026-05-27); non-tautological per R-CHAR-3 (the expected
//! `+inf` traces to `_infinite_like` at kl.py:145 = `torch.full_like(., inf)`).
//!
//! Tracking: #1563 (release-blocker: wrong-value bug in a pre-existing arm).
//! The two divergence tests are `#[ignore]`d because #1563 now tracks the fix;
//! the control test stays green to pin that the bug is direction-specific.

use ferrotorch_core::creation::scalar;
use ferrotorch_distributions::kl::kl_divergence;
use ferrotorch_distributions::{Normal, Uniform};

fn item(t: ferrotorch_core::tensor::Tensor<f64>) -> f64 {
    t.item().unwrap()
}

/// Divergence: ferrotorch's `kl_normal_uniform`
/// (`ferrotorch-distributions/src/kl.rs:725` -- `-entropy + log_range`)
/// diverges from `pytorch torch/distributions/kl.py:766,768`
/// (`@register_kl(Normal, Uniform)` -> `_kl_normal_infinity` ->
/// `return _infinite_like(p.loc)`) for `Normal(0,1) || Uniform(-2,2)`.
/// Upstream returns `+inf`; ferrotorch returns ~ -0.0326441720847821.
/// Tracking: #1563
#[test]
#[ignore = "divergence: kl_normal_uniform returns finite (even negative) where torch kl.py:766,768 returns +inf; tracking #1563"]
fn divergence_kl_normal_uniform_must_be_positive_infinity() {
    let p = Normal::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
    let q = Uniform::new(scalar(-2.0f64).unwrap(), scalar(2.0f64).unwrap()).unwrap();
    let kl = item(kl_divergence(&p, &q).expect("Normal-Uniform KL must be registered"));
    assert!(
        kl.is_infinite() && kl > 0.0,
        "KL(Normal(0,1) || Uniform(-2,2)) must be +inf (Normal's support R is \
         not contained in Uniform's [low,high]; torch kl.py:766,768 returns \
         _infinite_like(p.loc)), got {kl}"
    );
}

/// Second point, wider Uniform that still cannot cover R. torch returns `+inf`
/// here too (the formula is unconditional `_infinite_like(p.loc)`); ferrotorch's
/// finite formula instead returns a finite number that grows with the range.
/// Tracking: #1563
#[test]
#[ignore = "divergence: kl_normal_uniform returns finite where torch kl.py:766,768 returns +inf; tracking #1563"]
fn divergence_kl_normal_uniform_wide_range_still_inf() {
    let p = Normal::new(scalar(1.0f64).unwrap(), scalar(0.5f64).unwrap()).unwrap();
    let q = Uniform::new(scalar(-100.0f64).unwrap(), scalar(100.0f64).unwrap()).unwrap();
    let kl = item(kl_divergence(&p, &q).expect("Normal-Uniform KL must be registered"));
    assert!(
        kl.is_infinite() && kl > 0.0,
        "KL(Normal(1,0.5) || Uniform(-100,100)) must be +inf per torch \
         kl.py:766,768 (_infinite_like is unconditional), got {kl}"
    );
}

/// Guard the legitimately-finite OPPOSITE direction stays finite and correct,
/// to document that the bug is direction-specific (Uniform support [low,high]
/// IS contained in Normal's R). torch `kl.py:925-932` `_kl_uniform_normal`:
///   KL(Uniform(-2,2) || Normal(0,1)) = 0.19931083875144867 (live torch 2.11.0).
/// This test should PASS today and pins that kl_uniform_normal is NOT the bug.
/// Tracking: #1563 (control — not part of the divergence; must stay green).
#[test]
fn control_kl_uniform_normal_is_finite_and_correct() {
    let p = Uniform::new(scalar(-2.0f64).unwrap(), scalar(2.0f64).unwrap()).unwrap();
    let q = Normal::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
    let kl = item(kl_divergence(&p, &q).expect("Uniform-Normal KL must exist"));
    assert!(
        (kl - 0.199_310_838_751_448_67).abs() < 1e-12,
        "KL(Uniform(-2,2) || Normal(0,1)) must be torch's 0.19931083875144867, got {kl}"
    );
}
