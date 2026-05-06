//! Probe for #793: special::erfinv f32 Winitzki (2008) residual.
//!
//! Pre-fix, ferrotorch-core uses the Winitzki (2008) closed-form rational
//! approximation for `erfinv`:
//!   erfinv(x) = sign(x) * sqrt( -b + sqrt(b^2 - c) )
//!     b = 2/(pi*a) + ln(1-x^2)/2
//!     c = ln(1-x^2)/a
//!     a = 0.147
//! Documented worst-case |epsilon| <= 1.3e-3 over (-1, 1); empirically the
//! peak f32 deviation against PyTorch is ~1.3e-4 — three orders of magnitude
//! looser than the F32_TRANSCENDENTAL_CPU = 1e-5 conformance gate.
//!
//! This probe samples 1024 points across [-0.99, +0.99] (with a few
//! near-boundary stress points around ±0.999) and validates ferrotorch's
//! erfinv two ways:
//!
//! 1. **Direct comparison** against a high-precision reference computed in
//!    f64 by Newton-refining the same Winitzki initial guess until
//!    convergence using A1's ~1-ulp `special::erf` (#792). The reference is
//!    independent of any external libm `erfinv` (which POSIX does not
//!    provide).
//! 2. **Round-trip via libm `erf`**: for each sampled `y`, compute
//!    `r = erf(erfinv(y))` using libm's POSIX-required `erf` and assert
//!    `|r - y|` is within the gate. This anchors the test to libm — an
//!    external reference completely independent of ferrotorch — without
//!    needing libm to ship `erfinv`.
//!
//! The probe is permanent — once #793 is closed, it stays in the suite as a
//! regression sentinel. The assertion uses F32_TRANSCENDENTAL_CPU = 1e-5
//! verbatim from the conformance gate; any future regression that
//! reintroduces low-precision erfinv fails this probe before it reaches the
//! conformance suite.
//!
//! f64 lane: A1 did not touch erfinv, but we additionally probe the f64
//! path against the same Newton-refined reference to confirm no regression
//! and that the f64 lane already meets F64_TRANSCENDENTAL.

use ferrotorch_core::special::{erf, erfinv};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

const F32_TRANSCENDENTAL_CPU: f32 = 1e-5;
const F64_TRANSCENDENTAL: f64 = 1e-10;

// libm reference for the round-trip check. POSIX guarantees `double erf(double)`.
unsafe extern "C" {
    #[link_name = "erf"]
    fn c_erf(x: f64) -> f64;
    #[link_name = "erff"]
    fn c_erff(x: f32) -> f32;
}

fn libm_erf_f64(x: f64) -> f64 {
    // SAFETY: POSIX-required `double erf(double)` with no global state.
    unsafe { c_erf(x) }
}

fn libm_erf_f32(x: f32) -> f32 {
    // SAFETY: POSIX-required `float erff(float)` with no global state.
    unsafe { c_erff(x) }
}

fn make_tensor_f32(data: &[f32]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false)
        .expect("from_storage f32")
}

fn make_tensor_f64(data: &[f64]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false)
        .expect("from_storage f64")
}

/// High-precision erfinv reference.
///
/// Strategy: build a coarse initial guess via Winitzki's formula, then
/// Newton-iterate using ferrotorch-core's f64 `erf` (now SunPro fdlibm
/// piecewise rational, ~1 ulp after #792) until the residual drops below
/// 4 * f64::EPSILON or 16 iterations elapse — whichever comes first. The
/// Newton update is
///     x_{n+1} = x_n - (erf(x_n) - y) * (sqrt(pi) / 2) * exp(x_n^2)
/// which is the exact Newton step for f(x) = erf(x) - y, with f'(x) =
/// 2/sqrt(pi) * exp(-x^2).
fn ref_erfinv(y: f64) -> f64 {
    if y == 0.0 {
        return 0.0;
    }
    if y >= 1.0 {
        return f64::INFINITY;
    }
    if y <= -1.0 {
        return f64::NEG_INFINITY;
    }

    // Winitzki initial.
    let sign = if y < 0.0 { -1.0 } else { 1.0 };
    let ay = y.abs();
    let a = 0.147_f64;
    let pi = std::f64::consts::PI;
    let ln_term = (1.0 - ay * ay).ln();
    let b = 2.0 / (pi * a) + ln_term / 2.0;
    let c = ln_term / a;
    let mut x = sign * (-b + (b * b - c).sqrt()).sqrt();

    // Newton refine using ferrotorch's f64 erf (SunPro fdlibm via #792).
    let half_sqrt_pi = 0.5 * std::f64::consts::PI.sqrt();
    let one_elem = make_tensor_f64(&[x]);
    let mut e = erf(&one_elem).expect("erf").data().expect("data")[0];
    for _ in 0..16 {
        let resid = e - y;
        if resid.abs() < 4.0 * f64::EPSILON {
            break;
        }
        // Newton step: dx = (erf(x) - y) * sqrt(pi)/2 * exp(x^2)
        let step = resid * half_sqrt_pi * (x * x).exp();
        x -= step;
        let t = make_tensor_f64(&[x]);
        e = erf(&t).expect("erf").data().expect("data")[0];
    }
    x
}

fn linspace_f32(lo: f32, hi: f32, n: usize) -> Vec<f32> {
    assert!(n >= 2);
    let step = (hi - lo) / ((n - 1) as f32);
    (0..n).map(|i| lo + step * (i as f32)).collect()
}

fn linspace_f64(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    assert!(n >= 2);
    let step = (hi - lo) / ((n - 1) as f64);
    (0..n).map(|i| lo + step * (i as f64)).collect()
}

#[test]
fn erfinv_f32_within_f32_transcendental_cpu_gate() {
    // 1024 points across [-0.99, +0.99] plus a handful of near-boundary
    // stressors at |x| in {0.99, 0.995, 0.999} — Winitzki's residual peaks
    // toward |x|->1.
    let mut xs: Vec<f32> = linspace_f32(-0.99, 0.99, 1024);
    for &b in &[-0.999_f32, -0.995, -0.99, 0.99, 0.995, 0.999] {
        xs.push(b);
    }

    let input = make_tensor_f32(&xs);
    let actual = erfinv(&input).expect("erfinv f32");
    let actual = actual.data().expect("data").to_vec();
    let expected: Vec<f32> = xs.iter().map(|&x| ref_erfinv(x as f64) as f32).collect();

    let mut max_err = 0.0_f32;
    let mut at = f32::NAN;
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
    println!(
        "erfinv f32: max_abs_err = {:.3e} at x = {:+.5} (gate = {:.0e})",
        max_err, at, F32_TRANSCENDENTAL_CPU
    );
    assert!(
        max_err < F32_TRANSCENDENTAL_CPU,
        "erfinv f32 max abs err {:.3e} at x={:+.5} exceeds F32_TRANSCENDENTAL_CPU = {:.0e}",
        max_err,
        at,
        F32_TRANSCENDENTAL_CPU,
    );
}

#[test]
fn erfinv_f32_round_trip_via_libm_erf() {
    // External-reference anchor: erf(erfinv(y)) == y, where erf is libm's
    // POSIX-required entry point, so the only ferrotorch component under
    // test is `erfinv`. Use 1024 points avoiding the immediate neighborhood
    // of ±1 where the round-trip is dominated by erf saturating.
    let xs: Vec<f32> = linspace_f32(-0.95, 0.95, 1024);
    let input = make_tensor_f32(&xs);
    let out = erfinv(&input).expect("erfinv f32");
    let inv = out.data().expect("data").to_vec();

    let mut max_err = 0.0_f32;
    let mut at = f32::NAN;
    for (&y, &z) in xs.iter().zip(inv.iter()) {
        let r = libm_erf_f32(z);
        let err = (r - y).abs();
        if err > max_err {
            max_err = err;
            at = y;
        }
    }
    println!(
        "erfinv f32 round-trip libm: max_abs_err = {:.3e} at y = {:+.5} (gate = {:.0e})",
        max_err, at, F32_TRANSCENDENTAL_CPU
    );
    assert!(
        max_err < F32_TRANSCENDENTAL_CPU,
        "erfinv f32 round-trip libm max abs err {:.3e} at y={:+.5} exceeds F32_TRANSCENDENTAL_CPU = {:.0e}",
        max_err,
        at,
        F32_TRANSCENDENTAL_CPU,
    );
}

#[test]
fn erfinv_f64_round_trip_via_libm_erf() {
    let xs: Vec<f64> = linspace_f64(-0.95, 0.95, 1024);
    let input = make_tensor_f64(&xs);
    let out = erfinv(&input).expect("erfinv f64");
    let inv = out.data().expect("data").to_vec();

    let mut max_err = 0.0_f64;
    let mut at = f64::NAN;
    for (&y, &z) in xs.iter().zip(inv.iter()) {
        let r = libm_erf_f64(z);
        let err = (r - y).abs();
        if err > max_err {
            max_err = err;
            at = y;
        }
    }
    println!(
        "erfinv f64 round-trip libm: max_abs_err = {:.3e} at y = {:+.5} (gate = {:.0e})",
        max_err, at, F64_TRANSCENDENTAL
    );
    assert!(
        max_err < F64_TRANSCENDENTAL,
        "erfinv f64 round-trip libm max abs err {:.3e} at y={:+.5} exceeds F64_TRANSCENDENTAL = {:.0e}",
        max_err,
        at,
        F64_TRANSCENDENTAL,
    );
}

#[test]
fn erfinv_f64_within_f64_transcendental_gate() {
    // f64 lane: A1 did not touch erfinv, but the f64 erf path is now SunPro
    // fdlibm. Our reference is built on top of the same f64 erf, so this
    // test pins that the production erfinv f64 path tracks the reference
    // (i.e., does its own Newton refinement to f64 precision).
    let xs = linspace_f64(-0.99, 0.99, 1024);
    let input = make_tensor_f64(&xs);
    let actual = erfinv(&input).expect("erfinv f64");
    let actual = actual.data().expect("data").to_vec();
    let expected: Vec<f64> = xs.iter().map(|&x| ref_erfinv(x)).collect();

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
    println!(
        "erfinv f64: max_abs_err = {:.3e} at x = {:+.5} (gate = {:.0e})",
        max_err, at, F64_TRANSCENDENTAL
    );
    assert!(
        max_err < F64_TRANSCENDENTAL,
        "erfinv f64 max abs err {:.3e} at x={:+.5} exceeds F64_TRANSCENDENTAL = {:.0e}",
        max_err,
        at,
        F64_TRANSCENDENTAL,
    );
}
