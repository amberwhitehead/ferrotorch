//! Discriminator probes — the f64-narrow-vs-native-f32-overflow divergence
//! across the ported orthogonal-polynomial families (#1642 blast radius).
//!
//! ROOT CAUSE (shared, ONE fix): every CPU polynomial path in
//! `ferrotorch-core/src/special.rs` runs its three-term recurrence through the
//! single helper `elementwise_f64` (`special.rs:1218-1233`):
//!
//! ```ignore
//! .map(|v| T::from(f(v.to_f64().unwrap())).unwrap())
//! ```
//!
//! i.e. it ALWAYS evaluates the recurrence in f64 and narrows to `T`. PyTorch
//! evaluates the recurrence in the tensor's NATIVE scalar type
//! (`aten/src/ATen/native/Math.h` — every `*_forward<T>` declares `T p, q, r;`,
//! so for an f32 tensor the recurrence runs in f32). When the f32 recurrence
//! overflows, torch produces `±inf`, and because the recurrences subtract two
//! terms (`(x+x)*q - k*p`, `((k+k+1)*x*q - k*p)/(k+1)`, `(x+x)*q - p`) the
//! `inf - inf` yields `NaN`; the legendre / laguerre / chebyshev loops then
//! latch that NaN via their `&& !std::isnan(q)` guard and return NaN.
//! ferrotorch's f64 path stays finite far longer (f64 max ~1.8e308 vs f32 max
//! ~3.4e38), so it narrows an f64-finite value to `+inf` (or computes a finite
//! f64 value) where torch returns `NaN`.
//!
//! This is therefore a SINGLE shared `elementwise_f64` f64-narrow pattern, NOT
//! a per-family bug. The fix (for the fixer — the critic only pins) is to run
//! the CPU recurrence in native `T` for f32 tensors so the overflow matches
//! torch. Pinned here, one failing test per affected family, on #1642.
//!
//! Reference values produced by LIVE `torch 2.11.0+cu130` CPU
//! (native-dtype recurrence — see each test's cite) on 2026-05-28. NOT copied
//! from the ferrotorch side (R-CHAR-3): each expected value is `NaN`, the
//! torch.special.*_polynomial_* output for the documented input.

use ferrotorch_core::{Tensor, TensorStorage, special};

fn f32t(x: f32) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(vec![x]), vec![1], false).unwrap()
}

// =====================================================================
// HERMITE — the #1641 "below the limit unchanged" claim is FALSE.
//
// The getHermitianLimit<float>() guard fires only at n>128, but the f32
// recurrence H_n(0.05) already overflows to NaN by n=64. torch returns NaN
// for n in [~62, 128] (guard NOT fired, native-f32 overflow); ferrotorch CPU
// computes in f64 and narrows to ±inf. The guard added in c8e4dc49a does NOT
// cover this window, so #1641's regression claim ("n=64 f32 CPU == torch")
// does not hold against the f32 dtype.
// =====================================================================

/// Divergence: ferrotorch `hermite_polynomial_h` f32 diverges from
/// `pytorch aten/src/ATen/native/Math.h:3072-3081`
/// (`for (k…) { r = (x+x)*q - k*p; … }` in native `T=float`).
/// Input: `hermite_polynomial_h(tensor([0.05], f32), 64)` — BELOW the
/// getHermitianLimit<float>()==128 guard (`Math.h:3068`), so the guard does
/// not fire and torch runs the native-f32 recurrence, which overflows to NaN.
/// Upstream (torch 2.11 CPU) returns NaN; ferrotorch returns +inf
/// (f64-then-narrow via `special.rs:1218 elementwise_f64`).
/// Tracking: #1642.
#[test]
fn divergence_hermite_h_f32_n64_below_limit() {
    let got = special::hermite_polynomial_h(&f32t(0.05), 64)
        .unwrap()
        .data()
        .unwrap()[0];
    // torch.special.hermite_polynomial_h(torch.tensor([0.05]), 64) -> nan
    assert!(
        got.is_nan(),
        "hermite_polynomial_h(0.05, 64) f32: torch=NaN, ferrotorch={got:?}"
    );
}

/// Divergence: ferrotorch `hermite_polynomial_h` f32 at n=128 == the
/// getHermitianLimit<float>() limit, where `n > limit` is FALSE so the guard
/// (`Math.h:3068`) does NOT fire and torch runs the native-f32 recurrence.
/// Input: `hermite_polynomial_h(tensor([0.05], f32), 128)`.
/// Upstream (torch 2.11 CPU) returns NaN (f32 overflow); ferrotorch returns
/// +inf (f64 recurrence = 2.12e126, narrowed to f32 = +inf). This is the
/// exact case #1642 was opened on.
/// Tracking: #1642.
#[test]
fn divergence_hermite_h_f32_at_limit_128() {
    let got = special::hermite_polynomial_h(&f32t(0.05), 128)
        .unwrap()
        .data()
        .unwrap()[0];
    // torch.special.hermite_polynomial_h(torch.tensor([0.05]), 128) -> nan
    assert!(
        got.is_nan(),
        "hermite_polynomial_h(0.05, 128) f32: torch=NaN, ferrotorch={got:?}"
    );
}

/// Divergence: ferrotorch `hermite_polynomial_he` f32 at n=128 (== limit,
/// guard inactive). `pytorch Math.h:3113-3122` runs `r = x*q - k*p` in native
/// `T=float`; the f32 recurrence overflows to NaN.
/// Input: `hermite_polynomial_he(tensor([0.05], f32), 128)`.
/// Upstream returns NaN; ferrotorch returns +inf (f64-narrow).
/// Tracking: #1642.
#[test]
fn divergence_hermite_he_f32_at_limit_128() {
    let got = special::hermite_polynomial_he(&f32t(0.05), 128)
        .unwrap()
        .data()
        .unwrap()[0];
    // torch.special.hermite_polynomial_he(torch.tensor([0.05]), 128) -> nan
    assert!(
        got.is_nan(),
        "hermite_polynomial_he(0.05, 128) f32: torch=NaN, ferrotorch={got:?}"
    );
}

// =====================================================================
// LEGENDRE f32 — same shared elementwise_f64 root cause, different family.
// `pytorch Math.h:3190-3198`:
//   for (k…; (k<n) && !isnan(q); k++) { r = ((k+k+1)*x*q - k*p)/(k+1); … }
// in native T=float. For x=2.0 the values grow ~exponentially; the f32
// recurrence overflows to inf then NaN by n=80. torch=NaN; ferrotorch=+inf.
// =====================================================================

/// Divergence: ferrotorch `legendre_polynomial_p` f32 diverges from
/// `pytorch aten/src/ATen/native/Math.h:3190-3198` (native-f32 recurrence
/// with `&& !std::isnan(q)` latch).
/// Input: `legendre_polynomial_p(tensor([2.0], f32), 80)`.
/// Upstream (torch 2.11 CPU) returns NaN (f32 overflow → inf-inf → NaN,
/// latched by the isnan guard); ferrotorch returns +inf (f64 recurrence is
/// finite-ish then narrows). Confirms the divergence is NOT hermite-only.
/// Tracking: #1642.
#[test]
fn divergence_legendre_p_f32_overflow() {
    let got = special::legendre_polynomial_p(&f32t(2.0), 80)
        .unwrap()
        .data()
        .unwrap()[0];
    // torch.special.legendre_polynomial_p(torch.tensor([2.0]), 80) -> nan
    assert!(
        got.is_nan(),
        "legendre_polynomial_p(2.0, 80) f32: torch=NaN, ferrotorch={got:?}"
    );
}

// =====================================================================
// CHEBYSHEV T f32 — same root cause. `pytorch Math.h:2865-2871`:
//   for (k…; (k<=n) && !isnan(q); k++) { r = (x+x)*q - p; … }
// in native T=float. For x=1.5 (|x|>1, so the cos(n*acos) shortcut at
// Math.h:2850 does NOT apply — it requires |x|<1) the recurrence runs and
// overflows to NaN by n=100. torch=NaN; ferrotorch=+inf.
// =====================================================================

/// Divergence: ferrotorch `chebyshev_polynomial_t` f32 diverges from
/// `pytorch aten/src/ATen/native/Math.h:2865-2871` (native-f32 recurrence).
/// Input: `chebyshev_polynomial_t(tensor([1.5], f32), 100)` — `|x|>1` so the
/// `cos(n*acos(x))` closed form (`Math.h:2850`, gated on `|x|<1`) is NOT
/// taken; the native-f32 recurrence runs and overflows.
/// Upstream (torch 2.11 CPU) returns NaN; ferrotorch returns +inf.
/// Tracking: #1642.
#[test]
fn divergence_chebyshev_t_f32_overflow() {
    let got = special::chebyshev_polynomial_t(&f32t(1.5), 100)
        .unwrap()
        .data()
        .unwrap()[0];
    // torch.special.chebyshev_polynomial_t(torch.tensor([1.5]), 100) -> nan
    assert!(
        got.is_nan(),
        "chebyshev_polynomial_t(1.5, 100) f32: torch=NaN, ferrotorch={got:?}"
    );
}

/// Divergence: ferrotorch `chebyshev_polynomial_u` f32 diverges from
/// `pytorch aten/src/ATen/native/Math.h` `chebyshev_polynomial_u_forward`
/// (native-f32 recurrence `r=(x+x)*q-p`, seed `U_1=2x`).
/// Input: `chebyshev_polynomial_u(tensor([1.5], f32), 100)`.
/// Upstream (torch 2.11 CPU) returns NaN; ferrotorch returns +inf.
/// Tracking: #1642.
#[test]
fn divergence_chebyshev_u_f32_overflow() {
    let got = special::chebyshev_polynomial_u(&f32t(1.5), 100)
        .unwrap()
        .data()
        .unwrap()[0];
    // torch.special.chebyshev_polynomial_u(torch.tensor([1.5]), 100) -> nan
    assert!(
        got.is_nan(),
        "chebyshev_polynomial_u(1.5, 100) f32: torch=NaN, ferrotorch={got:?}"
    );
}
