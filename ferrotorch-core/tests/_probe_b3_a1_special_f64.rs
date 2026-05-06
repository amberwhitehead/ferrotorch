//! Probe for #792: special::{erf, erfc, digamma} f64 polynomial residual.
//!
//! Pre-fix, ferrotorch-core uses Abramowitz & Stegun 7.1.26 for `erf`, which
//! has documented worst-case |epsilon| <= 1.5e-7 — three orders of magnitude
//! looser than F64_TRANSCENDENTAL = 1e-10. This probe samples each function
//! across its standard domain at >=1024 grid points, compares against the
//! C-runtime libm reference (linked via the system `libm.so` so no Rust dep
//! is added), and reports the maximum absolute deviation.
//!
//! The probe is permanent — once #792 is closed, it stays in the suite as a
//! regression sentinel. The assertion uses F64_TRANSCENDENTAL = 1e-10
//! verbatim from the conformance gate, so any future regression that
//! reintroduces low-order polynomial residual fails this probe before it
//! reaches the conformance suite.
//!
//! Reference: the system C math library is linked into every binary on this
//! platform; we declare the four entry points with `extern "C"` so we don't
//! pull a new Rust dep into the workspace just for the probe. `erf`, `erfc`,
//! and `tgamma` are POSIX-required; we use `tgamma` to derive `digamma` via
//! the central-difference of `lgamma` (more stable than calling `tgamma`
//! directly, and matches the reference identity psi(x) = d/dx ln Gamma(x)).
//! For the digamma reference we use a high-order Stirling series shifted to
//! z >= 30 — that gives ~1e-15 absolute precision on the sampling domain,
//! well below the 1e-10 gate we are asserting against.

use ferrotorch_core::special::{digamma, erf, erfc};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

const F64_TRANSCENDENTAL: f64 = 1e-10;

// libm reference: links via the C runtime — no new Rust dep is added.
// Local link names avoid collision with the imported `special::{erf, erfc}`
// symbols brought in at module scope.
unsafe extern "C" {
    #[link_name = "erf"]
    fn c_erf(x: f64) -> f64;
    #[link_name = "erfc"]
    fn c_erfc(x: f64) -> f64;
}

fn libm_erf(x: f64) -> f64 {
    // SAFETY: `erf` is a POSIX-required `double erf(double)` with no global
    // state; passing any finite f64 (or +/-inf, or NaN) is safe.
    unsafe { c_erf(x) }
}

fn libm_erfc(x: f64) -> f64 {
    // SAFETY: `erfc` is a POSIX-required `double erfc(double)` with no global
    // state; passing any finite f64 (or +/-inf, or NaN) is safe.
    unsafe { c_erfc(x) }
}

/// High-precision digamma reference using a Stirling series shifted to
/// z >= 30. The asymptotic series for psi(z) is
///   psi(z) ~ ln(z) - 1/(2z) - sum_{k>=1} B_{2k} / (2k z^{2k})
/// with B_{2k}/2k = 1/12, 1/120, 1/252, 1/240, 1/132, 691/32760, ...
/// Truncating after the z^-12 term at z=30 gives error well below 1e-15.
fn ref_digamma(x: f64) -> f64 {
    if x < 0.5 {
        // Reflection: psi(1 - x) = psi(x) + pi * cot(pi * x).
        let pi = std::f64::consts::PI;
        let cot = (pi * x).cos() / (pi * x).sin();
        return ref_digamma(1.0 - x) - pi * cot;
    }

    // Recurrence: psi(x) = psi(x + 1) - 1/x. Shift up to z >= 30.
    let mut acc = 0.0;
    let mut z = x;
    while z < 30.0 {
        acc -= 1.0 / z;
        z += 1.0;
    }

    let z2 = z * z;
    let z4 = z2 * z2;
    let z6 = z4 * z2;
    let z8 = z4 * z4;
    let z10 = z8 * z2;
    let z12 = z8 * z4;

    // Bernoulli-derived coefficients B_{2k} / (2k):
    //   k=1: 1/12, k=2: 1/120, k=3: 1/252, k=4: 1/240,
    //   k=5: 1/132, k=6: 691/32760.
    acc + z.ln() - 1.0 / (2.0 * z) - 1.0 / (12.0 * z2) + 1.0 / (120.0 * z4) - 1.0 / (252.0 * z6)
        + 1.0 / (240.0 * z8)
        - 1.0 / (132.0 * z10)
        + 691.0 / (32_760.0 * z12)
}

fn linspace(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    assert!(n >= 2);
    let step = (hi - lo) / ((n - 1) as f64);
    (0..n).map(|i| lo + step * (i as f64)).collect()
}

fn make_tensor(data: &[f64]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false)
        .expect("from_storage")
}

fn max_abs_err(actual: &[f64], expected: &[f64], xs: &[f64]) -> (f64, f64) {
    let mut max_err = 0.0_f64;
    let mut at = f64::NAN;
    for ((&a, &e), &x) in actual.iter().zip(expected.iter()).zip(xs.iter()) {
        if !a.is_finite() || !e.is_finite() {
            continue;
        }
        let err = (a - e).abs();
        if err > max_err {
            max_err = err;
            at = x;
        }
    }
    (max_err, at)
}

#[test]
fn erf_within_f64_transcendental_gate() {
    // Domain: [-5, 5] covers the full meaningful range of erf (saturates to
    // +/-1 outside |x|=5). 1024 points across the symmetric range.
    let xs = linspace(-5.0, 5.0, 1024);
    let input = make_tensor(&xs);
    let out = erf(&input).expect("erf");
    let actual = out.data().expect("data").to_vec();
    let expected: Vec<f64> = xs.iter().map(|&x| libm_erf(x)).collect();
    let (max_err, at) = max_abs_err(&actual, &expected, &xs);
    println!(
        "erf: max_abs_err = {:.3e} at x = {:+.4} (gate = {:.0e})",
        max_err, at, F64_TRANSCENDENTAL
    );
    assert!(
        max_err < F64_TRANSCENDENTAL,
        "erf max abs err {:.3e} at x={:+.4} exceeds F64_TRANSCENDENTAL = {:.0e}",
        max_err,
        at,
        F64_TRANSCENDENTAL,
    );
}

#[test]
fn erfc_within_f64_transcendental_gate() {
    // Same domain as erf — erfc(x) = 1 - erf(x), so the residual envelope
    // is identical to erf's.
    let xs = linspace(-5.0, 5.0, 1024);
    let input = make_tensor(&xs);
    let out = erfc(&input).expect("erfc");
    let actual = out.data().expect("data").to_vec();
    let expected: Vec<f64> = xs.iter().map(|&x| libm_erfc(x)).collect();
    let (max_err, at) = max_abs_err(&actual, &expected, &xs);
    println!(
        "erfc: max_abs_err = {:.3e} at x = {:+.4} (gate = {:.0e})",
        max_err, at, F64_TRANSCENDENTAL
    );
    assert!(
        max_err < F64_TRANSCENDENTAL,
        "erfc max abs err {:.3e} at x={:+.4} exceeds F64_TRANSCENDENTAL = {:.0e}",
        max_err,
        at,
        F64_TRANSCENDENTAL,
    );
}

#[test]
fn digamma_within_f64_transcendental_gate() {
    // Domain: 0.1 to 100. Avoid x close to nonpositive integers where psi
    // has poles. 1024 points (logarithmically spread to densely sample the
    // small-x region where the recurrence shift does the most work).
    let n = 1024;
    let lo: f64 = 0.1_f64.ln();
    let hi: f64 = 100.0_f64.ln();
    let step = (hi - lo) / ((n - 1) as f64);
    let xs: Vec<f64> = (0..n).map(|i| (lo + step * i as f64).exp()).collect();

    let input = make_tensor(&xs);
    let out = digamma(&input).expect("digamma");
    let actual = out.data().expect("data").to_vec();
    let expected: Vec<f64> = xs.iter().map(|&x| ref_digamma(x)).collect();
    let (max_err, at) = max_abs_err(&actual, &expected, &xs);
    println!(
        "digamma: max_abs_err = {:.3e} at x = {:+.4} (gate = {:.0e})",
        max_err, at, F64_TRANSCENDENTAL
    );
    assert!(
        max_err < F64_TRANSCENDENTAL,
        "digamma max abs err {:.3e} at x={:+.4} exceeds F64_TRANSCENDENTAL = {:.0e}",
        max_err,
        at,
        F64_TRANSCENDENTAL,
    );
}
