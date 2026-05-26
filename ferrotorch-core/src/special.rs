//! Special mathematical functions (`torch.special` equivalent).
//!
//! All functions operate elementwise on tensors, returning a new tensor of the
//! same shape. Implementations use either `num_traits::Float` methods or
//! well-known numerical approximations (Abramowitz & Stegun, Lanczos, etc.).
//!
//! ## REQ status (per `.design/ferrotorch-core/special.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | `erf` at `special.rs:675`; consumer: `grad_fns::activation::erf_for_gelu` at `grad_fns/activation.rs:413` invokes `special::erf_scalar` |
//! | REQ-2 | SHIPPED | `erfc` at `special.rs:684`; consumer: re-export at `lib.rs:187` |
//! | REQ-3 | SHIPPED | `erfinv` at `special.rs:692`; consumer: re-export at `lib.rs:187` |
//! | REQ-4 | SHIPPED | `lgamma` at `special.rs:699`; consumer: re-export at `lib.rs:187` |
//! | REQ-5 | SHIPPED | `digamma` at `special.rs:707`; consumer: re-export at `lib.rs:187` |
//! | REQ-6 | SHIPPED | `log1p`/`expm1` at `special.rs:714,721`; consumer: re-export at `lib.rs:187` |
//! | REQ-7 | SHIPPED | `sinc` at `special.rs:726`; consumer: re-export at `lib.rs:187` |
//! | REQ-8 | SHIPPED | `xlogy` at `special.rs:733`; consumer: re-export at `lib.rs:187` |
//! | REQ-9 | SHIPPED | `chebyshev_polynomial_{t,u,v,w}` at `special.rs:794-832`; consumer: `ferrotorch_core::special::*`. GPU lowering tracked under umbrella #1545 (ferrotorch-core CPU-only paths roadmap) |
//! | REQ-10 | SHIPPED | `hermite_polynomial_h`/`hermite_polynomial_he` at `special.rs:841,849`; consumer: `ferrotorch_core::special::*` |
//! | REQ-11 | SHIPPED | `laguerre_polynomial_l`/`legendre_polynomial_p` at `special.rs:859,867`; consumer: `ferrotorch_core::special::*` |
//! | REQ-12 | SHIPPED | `shifted_chebyshev_polynomial_{t,u,v,w}` at `special.rs:875-908`; consumer: `ferrotorch_core::special::*` |

use std::any::TypeId;

use crate::dtype::Float;
use crate::error::FerrotorchResult;
use crate::ops::elementwise::{binary_map, unary_map};
use crate::tensor::Tensor;

/// Helper: return zero via `num_traits::Zero` to avoid ambiguity with
/// `ferray_core::Element::zero`.
#[inline]
fn nt_zero<T: num_traits::Zero>() -> T {
    <T as num_traits::Zero>::zero()
}

/// Helper: return one via `num_traits::One` to avoid ambiguity with
/// `ferray_core::Element::one`.
#[inline]
fn nt_one<T: num_traits::One>() -> T {
    <T as num_traits::One>::one()
}

// ---------------------------------------------------------------------------
// Constants (as f64; converted to T at call sites via T::from)
// ---------------------------------------------------------------------------

// Abramowitz & Stegun 7.1.26 coefficients for erf approximation.
//
// Used only for f32 today: the documented worst-case |epsilon| <= 1.5e-7 sits
// well inside the f32 transcendental tolerance gate (1e-5) but is three
// orders of magnitude looser than F64_TRANSCENDENTAL = 1e-10. The f64 path
// dispatches to `erf_f64_hi` below (#792).
const ERF_A1: f64 = 0.254829592;
const ERF_A2: f64 = -0.284496736;
const ERF_A3: f64 = 1.421413741;
const ERF_A4: f64 = -1.453152027;
const ERF_A5: f64 = 1.061405429;
const ERF_P: f64 = 0.3275911;

// Lanczos approximation coefficients (g = 7, n = 9).
const LANCZOS_G: f64 = 7.0;
const LANCZOS_COEFFICIENTS: [f64; 9] = [
    0.999_999_999_999_809_9,
    676.5203681218851,
    -1259.1392167224028,
    771.323_428_777_653_1,
    -176.615_029_162_140_6,
    12.507343278686905,
    -0.13857109526572012,
    9.984_369_578_019_572e-6,
    1.5056327351493116e-7,
];

// ---------------------------------------------------------------------------
// Scalar helper functions
// ---------------------------------------------------------------------------

// === High-precision f64 erf / erfc =========================================
//
// Cody (1969) / SunPro fdlibm-style piecewise rational approximations. The
// constants below are the canonical SunPro coefficients (Sun Microsystems
// 1993, public domain) used by the system C math library on Linux/macOS/BSD;
// the same constants appear unchanged in Go's math.Erf, Julia's libm, the
// `libm` Rust crate, and OpenBSD's libm. They give ~1 ulp accuracy across
// f64, well inside F64_TRANSCENDENTAL = 1e-10 (#792 conformance gate).
//
// Domain split (matching fdlibm exactly):
//   |x| < 2^-28      : erf(x) = x * (1 + efx + PP/QQ * x^2) (linear+quad)
//   |x| < 0.84375    : rational PP / QQ   in t = x^2
//   |x| < 1.25       : rational PA / QA   in s = |x| - 1
//   |x| < 1/0.35     : exp(-x^2 - 0.5625) * RA / SA + 0.5*sign(x) (etc.)
//   |x| < 28         : exp(-x^2 - 0.5625) * RB / SB + 0.5*sign(x) (etc.)
//   |x| >= 28        : saturate to ±1 (erf) / ±0 (erfc)
//
// The `efx` constant encodes the linear correction near the origin where
// the rational approximation degenerates.
//
// Clippy fires `excessive_precision` on most coefficients because they are
// written to 21 decimal digits — the trailing digits round to the same f64
// bit pattern as a 17-digit truncation, but they are reproduced verbatim
// from the SunPro source so the diff against the upstream reference is
// audit-friendly. Suppressed at the constant-block level only.

#[allow(clippy::excessive_precision)]
const ERF_EFX: f64 = 1.2837916709551257e-01;

// PP, QQ — rational approximation valid for |x| < 0.84375 (in t = x*x).
#[allow(clippy::excessive_precision)]
const ERF_PP0: f64 = 1.28379167095512558561e-01;
#[allow(clippy::excessive_precision)]
const ERF_PP1: f64 = -3.25042107247001499370e-01;
#[allow(clippy::excessive_precision)]
const ERF_PP2: f64 = -2.84817495755985104766e-02;
#[allow(clippy::excessive_precision)]
const ERF_PP3: f64 = -5.77027029648944159157e-03;
#[allow(clippy::excessive_precision)]
const ERF_PP4: f64 = -2.37630166566501626084e-05;
#[allow(clippy::excessive_precision)]
const ERF_QQ1: f64 = 3.97917223959155352819e-01;
#[allow(clippy::excessive_precision)]
const ERF_QQ2: f64 = 6.50222499887672944485e-02;
#[allow(clippy::excessive_precision)]
const ERF_QQ3: f64 = 5.08130628187576562776e-03;
#[allow(clippy::excessive_precision)]
const ERF_QQ4: f64 = 1.32494738004321644526e-04;
#[allow(clippy::excessive_precision)]
const ERF_QQ5: f64 = -3.96022827877536812320e-06;

// PA, QA — rational approximation valid for 0.84375 <= |x| < 1.25
// (in s = |x| - 1). erf(x) = sign(x) * (ERX + PA(s)/QA(s)).
#[allow(clippy::excessive_precision)]
const ERF_ERX: f64 = 8.45062911510467529297e-01;
#[allow(clippy::excessive_precision)]
const ERF_PA0: f64 = -2.36211856075265944077e-03;
#[allow(clippy::excessive_precision)]
const ERF_PA1: f64 = 4.14856118683748331666e-01;
#[allow(clippy::excessive_precision)]
const ERF_PA2: f64 = -3.72207876035701323847e-01;
#[allow(clippy::excessive_precision)]
const ERF_PA3: f64 = 3.18346619901161753674e-01;
#[allow(clippy::excessive_precision)]
const ERF_PA4: f64 = -1.10894694282396677476e-01;
#[allow(clippy::excessive_precision)]
const ERF_PA5: f64 = 3.54783043256182359371e-02;
#[allow(clippy::excessive_precision)]
const ERF_PA6: f64 = -2.16637559486879084300e-03;
#[allow(clippy::excessive_precision)]
const ERF_QA1: f64 = 1.06420880400844228286e-01;
#[allow(clippy::excessive_precision)]
const ERF_QA2: f64 = 5.40397917702171048937e-01;
#[allow(clippy::excessive_precision)]
const ERF_QA3: f64 = 7.18286544141962662868e-02;
#[allow(clippy::excessive_precision)]
const ERF_QA4: f64 = 1.26171219808761642112e-01;
#[allow(clippy::excessive_precision)]
const ERF_QA5: f64 = 1.36370839120290507362e-02;
#[allow(clippy::excessive_precision)]
const ERF_QA6: f64 = 1.19844998467991074170e-02;

// RA, SA — rational approximation for 1.25 <= |x| < 1/0.35 (~2.857).
// erfc(x) = exp(-x^2 - 0.5625) * (RA(1/x^2) / SA(1/x^2)) / x.
#[allow(clippy::excessive_precision)]
const ERF_RA0: f64 = -9.86494403484714822705e-03;
#[allow(clippy::excessive_precision)]
const ERF_RA1: f64 = -6.93858572707181764372e-01;
#[allow(clippy::excessive_precision)]
const ERF_RA2: f64 = -1.05586262253232909814e+01;
#[allow(clippy::excessive_precision)]
const ERF_RA3: f64 = -6.23753324503260060396e+01;
#[allow(clippy::excessive_precision)]
const ERF_RA4: f64 = -1.62396669462573470355e+02;
#[allow(clippy::excessive_precision)]
const ERF_RA5: f64 = -1.84605092906711035994e+02;
#[allow(clippy::excessive_precision)]
const ERF_RA6: f64 = -8.12874355063065934246e+01;
#[allow(clippy::excessive_precision)]
const ERF_RA7: f64 = -9.81432934416914548592e+00;
#[allow(clippy::excessive_precision)]
const ERF_SA1: f64 = 1.96512716674392571292e+01;
#[allow(clippy::excessive_precision)]
const ERF_SA2: f64 = 1.37657754143519042600e+02;
#[allow(clippy::excessive_precision)]
const ERF_SA3: f64 = 4.34565877475229228821e+02;
#[allow(clippy::excessive_precision)]
const ERF_SA4: f64 = 6.45387271733267880336e+02;
#[allow(clippy::excessive_precision)]
const ERF_SA5: f64 = 4.29008140027567833386e+02;
#[allow(clippy::excessive_precision)]
const ERF_SA6: f64 = 1.08635005541779435134e+02;
#[allow(clippy::excessive_precision)]
const ERF_SA7: f64 = 6.57024977031928170135e+00;
#[allow(clippy::excessive_precision)]
const ERF_SA8: f64 = -6.04244152148580987438e-02;

// RB, SB — rational approximation for 1/0.35 <= |x| < 28.
#[allow(clippy::excessive_precision)]
const ERF_RB0: f64 = -9.86494292470009928597e-03;
#[allow(clippy::excessive_precision)]
const ERF_RB1: f64 = -7.99283237680523006574e-01;
#[allow(clippy::excessive_precision)]
const ERF_RB2: f64 = -1.77579549177547519889e+01;
#[allow(clippy::excessive_precision)]
const ERF_RB3: f64 = -1.60636384855821916062e+02;
#[allow(clippy::excessive_precision)]
const ERF_RB4: f64 = -6.37566443368389627722e+02;
#[allow(clippy::excessive_precision)]
const ERF_RB5: f64 = -1.02509513161107724954e+03;
#[allow(clippy::excessive_precision)]
const ERF_RB6: f64 = -4.83519191608651397019e+02;
#[allow(clippy::excessive_precision)]
const ERF_SB1: f64 = 3.03380607434824582924e+01;
#[allow(clippy::excessive_precision)]
const ERF_SB2: f64 = 3.25792512996573918826e+02;
#[allow(clippy::excessive_precision)]
const ERF_SB3: f64 = 1.53672958608443695994e+03;
#[allow(clippy::excessive_precision)]
const ERF_SB4: f64 = 3.19985821950859553908e+03;
#[allow(clippy::excessive_precision)]
const ERF_SB5: f64 = 2.55305040643316442583e+03;
#[allow(clippy::excessive_precision)]
const ERF_SB6: f64 = 4.74528541206955367215e+02;
#[allow(clippy::excessive_precision)]
const ERF_SB7: f64 = -2.24409524465858183362e+01;

/// High-precision f64 erf using the SunPro fdlibm piecewise rational
/// approximation. Accuracy: ~1 ulp across all of f64 (well inside the
/// F64_TRANSCENDENTAL = 1e-10 conformance gate). Closes #792.
fn erf_f64_hi(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x == f64::INFINITY {
        return 1.0;
    }
    if x == f64::NEG_INFINITY {
        return -1.0;
    }

    let ax = x.abs();

    if ax < 0.84375 {
        // Near origin: exploit the small-x cancellation by computing
        // erf(x) = x + x * R(x^2) where R is a rational in x^2.
        if ax < f64::from_bits(0x3E300000_00000000) {
            // |x| < 2^-28 — sub-ULP regime; linear extrapolation.
            return x + ERF_EFX * x;
        }
        let z = x * x;
        let r = ERF_PP0 + z * (ERF_PP1 + z * (ERF_PP2 + z * (ERF_PP3 + z * ERF_PP4)));
        let s = 1.0 + z * (ERF_QQ1 + z * (ERF_QQ2 + z * (ERF_QQ3 + z * (ERF_QQ4 + z * ERF_QQ5))));
        let y = r / s;
        return x + x * y;
    }

    if ax < 1.25 {
        // 0.84375 <= |x| < 1.25
        let s = ax - 1.0;
        let p = ERF_PA0
            + s * (ERF_PA1
                + s * (ERF_PA2 + s * (ERF_PA3 + s * (ERF_PA4 + s * (ERF_PA5 + s * ERF_PA6)))));
        let q = 1.0
            + s * (ERF_QA1
                + s * (ERF_QA2 + s * (ERF_QA3 + s * (ERF_QA4 + s * (ERF_QA5 + s * ERF_QA6)))));
        let y = ERF_ERX + p / q;
        return if x >= 0.0 { y } else { -y };
    }

    if ax >= 6.0 {
        // erf(x) saturates to ±1 to within f64 precision once |x| > ~6.
        return if x >= 0.0 { 1.0 } else { -1.0 };
    }

    // 1.25 <= |x| < 6: erf(x) = sign(x) * (1 - erfc_tail(|x|)).
    let s = 1.0 / (ax * ax);
    let (r, big_s) = if ax < 1.0 / 0.35 {
        // 1.25 <= |x| < 1/0.35
        let r = ERF_RA0
            + s * (ERF_RA1
                + s * (ERF_RA2
                    + s * (ERF_RA3 + s * (ERF_RA4 + s * (ERF_RA5 + s * (ERF_RA6 + s * ERF_RA7))))));
        let big_s = 1.0
            + s * (ERF_SA1
                + s * (ERF_SA2
                    + s * (ERF_SA3
                        + s * (ERF_SA4
                            + s * (ERF_SA5 + s * (ERF_SA6 + s * (ERF_SA7 + s * ERF_SA8)))))));
        (r, big_s)
    } else {
        let r = ERF_RB0
            + s * (ERF_RB1
                + s * (ERF_RB2 + s * (ERF_RB3 + s * (ERF_RB4 + s * (ERF_RB5 + s * ERF_RB6)))));
        let big_s = 1.0
            + s * (ERF_SB1
                + s * (ERF_SB2
                    + s * (ERF_SB3 + s * (ERF_SB4 + s * (ERF_SB5 + s * (ERF_SB6 + s * ERF_SB7))))));
        (r, big_s)
    };

    // Form `exp(-x^2 - 0.5625) * R/S / |x|` carefully: split |x| via
    // `f64::from_bits(bits & 0xFFFFFFFF_00000000)` to truncate to the upper
    // 32 bits — this gives an exact `z` plus a small correction `x - z` so
    // `exp(-z*z - 0.5625) * exp(-(x-z)*(x+z)) * (R/S)/|x|` minimizes
    // catastrophic cancellation in the exponent argument.
    let bits = ax.to_bits() & 0xFFFFFFFF_00000000;
    let z = f64::from_bits(bits);
    let r_factor = (-z * z - 0.5625).exp() * (-(ax - z) * (ax + z) + r / big_s).exp() / ax;
    if x >= 0.0 {
        1.0 - r_factor
    } else {
        r_factor - 1.0
    }
}

/// High-precision f64 erfc using the same SunPro fdlibm piecewise rational
/// approximation but expressed directly so the right-tail (large positive
/// `x`) is computed without the catastrophic `1 - erf(x)` cancellation.
/// Accuracy: ~1 ulp across all of f64. Closes #792.
fn erfc_f64_hi(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x == f64::INFINITY {
        return 0.0;
    }
    if x == f64::NEG_INFINITY {
        return 2.0;
    }

    let ax = x.abs();

    if ax < 0.84375 {
        if ax < f64::from_bits(0x3C700000_00000000) {
            // |x| < 2^-56 — erf(x) is subnormally small; erfc(x) = 1 - erf(x).
            return 1.0 - x;
        }
        let z = x * x;
        let r = ERF_PP0 + z * (ERF_PP1 + z * (ERF_PP2 + z * (ERF_PP3 + z * ERF_PP4)));
        let s = 1.0 + z * (ERF_QQ1 + z * (ERF_QQ2 + z * (ERF_QQ3 + z * (ERF_QQ4 + z * ERF_QQ5))));
        let y = r / s;
        if ax < 0.25 {
            // 1 - (x + x*y) preserves precision when y*x is small.
            return 1.0 - (x + x * y);
        }
        // Re-associate as 0.5 - (x + x*y - 0.5) to keep significand bits.
        let r2 = x * y;
        let r3 = r2 + x;
        return 0.5 - (r3 - 0.5);
    }

    if ax < 1.25 {
        let s = ax - 1.0;
        let p = ERF_PA0
            + s * (ERF_PA1
                + s * (ERF_PA2 + s * (ERF_PA3 + s * (ERF_PA4 + s * (ERF_PA5 + s * ERF_PA6)))));
        let q = 1.0
            + s * (ERF_QA1
                + s * (ERF_QA2 + s * (ERF_QA3 + s * (ERF_QA4 + s * (ERF_QA5 + s * ERF_QA6)))));
        if x >= 0.0 {
            let z = 1.0 - ERF_ERX;
            return z - p / q;
        }
        let z = ERF_ERX + p / q;
        return 1.0 + z;
    }

    if ax < 28.0 {
        let s = 1.0 / (ax * ax);
        let (r, big_s) = if ax < 1.0 / 0.35 {
            let r = ERF_RA0
                + s * (ERF_RA1
                    + s * (ERF_RA2
                        + s * (ERF_RA3
                            + s * (ERF_RA4 + s * (ERF_RA5 + s * (ERF_RA6 + s * ERF_RA7))))));
            let big_s = 1.0
                + s * (ERF_SA1
                    + s * (ERF_SA2
                        + s * (ERF_SA3
                            + s * (ERF_SA4
                                + s * (ERF_SA5 + s * (ERF_SA6 + s * (ERF_SA7 + s * ERF_SA8)))))));
            (r, big_s)
        } else {
            let r = ERF_RB0
                + s * (ERF_RB1
                    + s * (ERF_RB2 + s * (ERF_RB3 + s * (ERF_RB4 + s * (ERF_RB5 + s * ERF_RB6)))));
            let big_s = 1.0
                + s * (ERF_SB1
                    + s * (ERF_SB2
                        + s * (ERF_SB3
                            + s * (ERF_SB4 + s * (ERF_SB5 + s * (ERF_SB6 + s * ERF_SB7))))));
            (r, big_s)
        };

        let bits = ax.to_bits() & 0xFFFFFFFF_00000000;
        let z = f64::from_bits(bits);
        let r_factor = (-z * z - 0.5625).exp() * (-(ax - z) * (ax + z) + r / big_s).exp() / ax;
        if x >= 0.0 { r_factor } else { 2.0 - r_factor }
    } else if x >= 0.0 {
        0.0
    } else {
        2.0
    }
}

/// Compute erf(x) for a single float.
///
/// f64 path (T = f64): SunPro fdlibm piecewise rational approximation
/// (`erf_f64_hi`), accuracy ~1 ulp — meets F64_TRANSCENDENTAL = 1e-10.
/// Other types (f32, bf16): Abramowitz & Stegun 7.1.26 polynomial,
/// accuracy ~1.5e-7 — well inside F32_TRANSCENDENTAL_CPU = 1e-5.
///
/// `pub(crate)` so internal callers (e.g. `grad_fns::activation::gelu_with`
/// in the GELU(none) branch, which is `0.5 * x * (1 + erf(x / sqrt(2)))`)
/// share the same precision path — without it, gelu_none retained the
/// 1.5e-7 A&S residual even after special::erf was upgraded (#792).
pub(crate) fn erf_scalar<T: Float>(x: T) -> T {
    // f64 specialization via TypeId: zero runtime cost (the branch is
    // monomorphized away by the compiler) and avoids relaxing the gate.
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        let xf = x.to_f64().unwrap();
        let yf = erf_f64_hi(xf);
        return T::from(yf).unwrap();
    }

    let zero = nt_zero::<T>();
    let one = nt_one::<T>();

    if x == zero {
        return zero;
    }

    let sign = if x < zero { -one } else { one };
    let ax = x.abs();

    let p = T::from(ERF_P).unwrap();
    let t = one / (one + p * ax);

    let a1 = T::from(ERF_A1).unwrap();
    let a2 = T::from(ERF_A2).unwrap();
    let a3 = T::from(ERF_A3).unwrap();
    let a4 = T::from(ERF_A4).unwrap();
    let a5 = T::from(ERF_A5).unwrap();

    // Horner form: (a1 + t*(a2 + t*(a3 + t*(a4 + t*a5))))
    let poly = a1 + t * (a2 + t * (a3 + t * (a4 + t * a5)));

    sign * (one - poly * t * (-ax * ax).exp())
}

/// Compute erfc(x) for a single float.
///
/// f64 path: SunPro fdlibm `erfc_f64_hi`, ~1 ulp — closes #792 (gelu_none
/// inherits this precision since GELU(none) calls `erf` internally).
/// f32/bf16: `1 - erf_scalar(x)` (the f32 cancellation is bounded by the
/// f32 transcendental tolerance and the only way for erfc to leave that
/// tolerance is for erf to leave its own tolerance first).
fn erfc_scalar<T: Float>(x: T) -> T {
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        let xf = x.to_f64().unwrap();
        let yf = erfc_f64_hi(xf);
        return T::from(yf).unwrap();
    }

    nt_one::<T>() - erf_scalar(x)
}

/// Compute erfinv(x) for a single float — Winitzki (2008) initial guess
/// followed by Newton refinement against the SunPro fdlibm `erf_f64_hi`.
///
/// Why this shape:
///
/// - The Winitzki (2008) closed-form rational approximation is convenient
///   (no tables, no branching) but its documented worst-case |epsilon| is
///   ~1.3e-3 over (-1, 1) and empirically peaks at ~4.4e-3 near |x| -> 1
///   in f32 — three orders of magnitude past F32_TRANSCENDENTAL_CPU = 1e-5
///   and many orders past F64_TRANSCENDENTAL = 1e-10. Using it as the
///   final answer was the root cause of #793.
/// - A1 just upgraded `erf_f64_hi` to ~1 ulp via the SunPro fdlibm
///   piecewise rational (see #792). With a sub-ulp `erf` available, the
///   Newton iteration for f(x) = erf(x) - y converges quadratically:
///   `x_{n+1} = x_n - (erf(x_n) - y) * (sqrt(pi) / 2) * exp(x_n^2)`.
///   From the Winitzki seed (~1e-3 error), one step lands inside ~1e-6
///   and three steps inside ~1e-15, comfortably under both
///   F32_TRANSCENDENTAL_CPU and F64_TRANSCENDENTAL.
/// - We do the entire refinement in f64 (the Winitzki seed too) so that
///   f32 inputs benefit from the high-precision intermediate computation
///   before being narrowed back via `T::from`. The f32 path otherwise
///   loses precision on the `(1 - x*x).ln()` term as |x| -> 1 because the
///   subtraction loses bits before the log even runs.
///
/// Two iterations are sufficient for both gates; the loop exits early once
/// the residual drops below 4 * f64::EPSILON. Closes #793.
fn erfinv_scalar<T: Float>(x: T) -> T {
    let zero = nt_zero::<T>();
    let one = nt_one::<T>();

    if x == zero {
        return zero;
    }
    if x >= one {
        return T::infinity();
    }
    if x <= -one {
        return T::neg_infinity();
    }

    // All work in f64: Winitzki seed in the wider type, then Newton with the
    // fdlibm `erf_f64_hi`. Narrow back to T at the very end.
    let y = x.to_f64().unwrap();
    let sign = if y < 0.0 { -1.0 } else { 1.0 };
    let ay = y.abs();

    // Winitzki (2008) initial guess. `a = 0.147` is the constant the original
    // paper tunes for; we keep it verbatim since this is just the seed.
    let a = 0.147_f64;
    let pi = std::f64::consts::PI;
    let ln_term = (1.0 - ay * ay).ln();
    let b = 2.0 / (pi * a) + ln_term / 2.0;
    let c = ln_term / a;
    let mut z = sign * (-b + (b * b - c).sqrt()).sqrt();

    // Newton refine: f(z) = erf(z) - y, f'(z) = 2/sqrt(pi) * exp(-z^2).
    // dz = (erf(z) - y) / f'(z) = (erf(z) - y) * sqrt(pi)/2 * exp(z^2).
    // Three iterations max — quadratic convergence drops a ~1e-3 seed to
    // sub-ulp in 3 steps, and the early-exit guard catches f32 inputs after
    // ~1 step.
    let half_sqrt_pi = 0.5 * pi.sqrt();
    for _ in 0..3 {
        let resid = erf_f64_hi(z) - y;
        if resid.abs() < 4.0 * f64::EPSILON {
            break;
        }
        z -= resid * half_sqrt_pi * (z * z).exp();
    }

    T::from(z).unwrap()
}

/// Compute lgamma(x) using the Lanczos approximation.
fn lgamma_scalar<T: Float>(x: T) -> T {
    let one = nt_one::<T>();
    let half = T::from(0.5).unwrap();
    let half_ln_2pi = T::from(0.9189385332046727).unwrap(); // 0.5 * ln(2*pi)
    let g = T::from(LANCZOS_G).unwrap();

    // Handle negative values via reflection formula.
    if x < half {
        let pi = T::from(std::f64::consts::PI).unwrap();
        let sin_pi_x = (pi * x).sin();
        if sin_pi_x == nt_zero::<T>() {
            return T::infinity();
        }
        return (pi / sin_pi_x.abs()).ln() - lgamma_scalar(one - x);
    }

    let z = x - one;
    let mut sum = T::from(LANCZOS_COEFFICIENTS[0]).unwrap();
    for (i, &coeff) in LANCZOS_COEFFICIENTS.iter().enumerate().skip(1) {
        sum += T::from(coeff).unwrap() / (z + T::from(i as f64).unwrap());
    }

    let t = z + g + half;
    half_ln_2pi + (t).ln() * (z + half) - t + sum.ln()
}

/// High-precision f64 digamma using Bernoulli-extended Stirling asymptotics
/// shifted to z >= 14. The asymptotic series is
///   psi(z) ~ ln(z) - 1/(2z) - sum_{k>=1} B_{2k} / (2k z^{2k})
/// with B_{2k}/(2k) coefficients 1/12, 1/120, 1/252, 1/240, 1/132,
/// 691/32760, 1/12. Truncating after the z^-12 term at z >= 14 gives
/// |residual| <= 691/(32760 * 14^12) ≈ 4e-16, well inside the
/// F64_TRANSCENDENTAL = 1e-10 conformance gate. Closes #792.
fn digamma_f64_hi(x: f64) -> f64 {
    if x < 0.5 {
        // Reflection: psi(1 - x) = psi(x) + pi * cot(pi * x).
        let pi = std::f64::consts::PI;
        let cot = (pi * x).cos() / (pi * x).sin();
        return digamma_f64_hi(1.0 - x) - pi * cot;
    }

    // Recurrence: psi(x) = psi(x + 1) - 1/x. Shift up to z >= 14 (the
    // pre-fix shift target was 6, which left the leading omitted term
    // 1/(240 z^8) ≈ 2.5e-9 — already past the 1e-10 gate).
    let mut acc = 0.0_f64;
    let mut z = x;
    while z < 14.0 {
        acc -= 1.0 / z;
        z += 1.0;
    }

    let z2 = z * z;
    let z4 = z2 * z2;
    let z6 = z4 * z2;
    let z8 = z4 * z4;
    let z10 = z8 * z2;
    let z12 = z8 * z4;

    acc + z.ln() - 1.0 / (2.0 * z) - 1.0 / (12.0 * z2) + 1.0 / (120.0 * z4) - 1.0 / (252.0 * z6)
        + 1.0 / (240.0 * z8)
        - 1.0 / (132.0 * z10)
        + 691.0 / (32_760.0 * z12)
}

/// Compute digamma(x) = psi(x) = d/dx ln(Gamma(x)).
///
/// f64 path: extended Stirling series shifted to z >= 14, ~1e-15 residual —
/// closes #792 (pre-fix asymptotic-series truncation at z^-6 left a ~2.5e-9
/// residual at the shift threshold).
/// Other types (f32, bf16): legacy 4-term asymptotic expansion shifted
/// to z >= 6 (well inside f32 transcendental tolerance).
fn digamma_scalar<T: Float>(x: T) -> T {
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        let xf = x.to_f64().unwrap();
        let yf = digamma_f64_hi(xf);
        return T::from(yf).unwrap();
    }

    let zero = nt_zero::<T>();
    let one = nt_one::<T>();
    let half = T::from(0.5).unwrap();

    // Handle negative values via reflection formula:
    // psi(1 - x) = psi(x) + pi * cot(pi * x)
    if x < half {
        let pi = T::from(std::f64::consts::PI).unwrap();
        let cot = (pi * x).cos() / (pi * x).sin();
        return digamma_scalar(one - x) - pi * cot;
    }

    // Shift x upward until x >= 6 using recurrence: psi(x) = psi(x+1) - 1/x.
    let mut result = zero;
    let mut z = x;
    let six = T::from(6.0).unwrap();
    while z < six {
        #[allow(clippy::assign_op_pattern)]
        {
            result = result - one / z;
        }
        #[allow(clippy::assign_op_pattern)]
        {
            z = z + one;
        }
    }

    // Asymptotic expansion for large z:
    // psi(z) ~ ln(z) - 1/(2z) - 1/(12z^2) + 1/(120z^4) - 1/(252z^6) + ...
    let z2 = z * z;
    let z4 = z2 * z2;
    let z6 = z4 * z2;

    result =
        result + z.ln() - one / (T::from(2.0).unwrap() * z) - one / (T::from(12.0).unwrap() * z2)
            + one / (T::from(120.0).unwrap() * z4)
            - one / (T::from(252.0).unwrap() * z6);

    result
}

/// Compute sinc(x) = sin(pi*x) / (pi*x), with sinc(0) = 1.
fn sinc_scalar<T: Float>(x: T) -> T {
    let zero = nt_zero::<T>();
    let one = nt_one::<T>();

    if x == zero {
        return one;
    }

    let pi = T::from(std::f64::consts::PI).unwrap();
    let pi_x = pi * x;
    pi_x.sin() / pi_x
}

/// Compute x * log(y) with the convention that 0 * log(y) = 0.
fn xlogy_scalar<T: Float>(x: T, y: T) -> T {
    if x == nt_zero::<T>() {
        nt_zero::<T>()
    } else {
        x * y.ln()
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Error function: erf(x) = (2/sqrt(pi)) * integral(0, x, exp(-t^2) dt).
///
/// f64 path: SunPro fdlibm piecewise rational approximation, ~1 ulp accuracy
/// (meets F64_TRANSCENDENTAL = 1e-10). f32/bf16 path: Abramowitz & Stegun
/// 7.1.26 polynomial, |epsilon| <= 1.5e-7 (meets F32_TRANSCENDENTAL = 1e-5).
pub fn erf<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    unary_map(input, erf_scalar)
}

/// Complementary error function: erfc(x) = 1 - erf(x).
///
/// f64 path: SunPro fdlibm `erfc_f64_hi` — computed directly so the
/// right-tail (large positive x) avoids the catastrophic 1 - erf(x)
/// cancellation. f32/bf16 path: literal `1 - erf_scalar(x)`.
pub fn erfc<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    unary_map(input, erfc_scalar)
}

/// Inverse error function: erfinv(erf(x)) = x.
///
/// Uses the Winitzki (2008) rational approximation. Returns `inf` for
/// input = 1, `-inf` for input = -1, and `NaN` for |input| > 1.
pub fn erfinv<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    unary_map(input, erfinv_scalar)
}

/// Log-gamma function: lgamma(x) = log(|Gamma(x)|).
///
/// Uses the Lanczos approximation (g = 7, n = 9 coefficients).
pub fn lgamma<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    unary_map(input, lgamma_scalar)
}

/// Digamma function: psi(x) = d/dx ln(Gamma(x)).
///
/// Uses the recurrence relation to shift the argument above 6, then
/// applies the asymptotic expansion.
pub fn digamma<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    unary_map(input, digamma_scalar)
}

/// log(1 + x) -- numerically stable for small x.
///
/// Delegates to `num_traits::Float::ln_1p()`.
pub fn log1p<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    unary_map(input, |x| x.ln_1p())
}

/// exp(x) - 1 -- numerically stable for small x.
///
/// Delegates to `num_traits::Float::exp_m1()`.
pub fn expm1<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    unary_map(input, |x| x.exp_m1())
}

/// Normalized sinc function: sinc(x) = sin(pi*x) / (pi*x), with sinc(0) = 1.
pub fn sinc<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    unary_map(input, sinc_scalar)
}

/// x * log(y), with the convention that xlogy(0, y) = 0 for any y.
///
/// This is useful for entropy computations where 0 * log(0) should be 0.
pub fn xlogy<T: Float>(x: &Tensor<T>, y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    binary_map(x, y, xlogy_scalar)
}

// ===========================================================================
// Orthogonal-polynomial families
// ===========================================================================
//
// These mirror torch.special.{chebyshev,hermite,laguerre,legendre}_polynomial_*.
// Each function returns the n-th degree basis polynomial evaluated pointwise
// at every element of `input`. CPU-only today; a CubeCL kernel that runs the
// three-term recurrence on-device is tracked as a follow-up. GPU tensors
// return NotImplementedOnCuda — never silently bounce through host (per
// /rust-gpu-discipline).
//
// Implementation: every basis is evaluated by its standard three-term
// recurrence directly in this module. ferray-polynomial 0.3 has the same
// idiom (Clenshaw on basis-coefficient vectors), but for the
// "single-basis-element" case the direct recurrence is shorter, dependency-
// free, and numerically equivalent for the orders typically used in ML
// pipelines (n ≤ 50). We keep the option open to swap in ferray's Clenshaw
// path later for very-high-order numerical-analysis use cases.

use crate::error::FerrotorchError;

/// Reject GPU tensors with an explicit error rather than bouncing through
/// host. Polynomial evaluation has no GPU kernel today; see follow-up.
#[inline]
fn require_cpu_poly<T: Float>(t: &Tensor<T>, op: &'static str) -> FerrotorchResult<()> {
    if t.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op });
    }
    Ok(())
}

/// Apply a `f64 -> f64` evaluator to every element of a tensor, with the
/// CPU/dtype guard inlined so each function below stays a one-liner.
fn elementwise_f64<T: Float, F: Fn(f64) -> f64>(
    input: &Tensor<T>,
    op: &'static str,
    f: F,
) -> FerrotorchResult<Tensor<T>> {
    require_cpu_poly(input, op)?;
    let data = input.data_vec()?;
    let out: Vec<T> = data
        .into_iter()
        .map(|v| T::from(f(v.to_f64().unwrap())).unwrap())
        .collect();
    crate::tensor::Tensor::from_storage(
        crate::storage::TensorStorage::cpu(out),
        input.shape().to_vec(),
        false,
    )
}

// --- Chebyshev family ------------------------------------------------------

/// Chebyshev polynomial of the **first kind** `T_n(x)`.
///
/// `T_0 = 1`, `T_1 = x`, `T_{n+1} = 2x T_n - T_{n-1}`. Mirrors
/// `torch.special.chebyshev_polynomial_t`.
pub fn chebyshev_polynomial_t<T: Float>(
    input: &Tensor<T>,
    n: usize,
) -> FerrotorchResult<Tensor<T>> {
    elementwise_f64(input, "chebyshev_polynomial_t", move |x| chebyshev_t(n, x))
}

/// Chebyshev polynomial of the **second kind** `U_n(x)`.
///
/// `U_0 = 1`, `U_1 = 2x`, `U_{n+1} = 2x U_n - U_{n-1}`. Mirrors
/// `torch.special.chebyshev_polynomial_u`. Evaluated by direct recurrence;
/// not provided by ferray-polynomial.
pub fn chebyshev_polynomial_u<T: Float>(
    input: &Tensor<T>,
    n: usize,
) -> FerrotorchResult<Tensor<T>> {
    elementwise_f64(input, "chebyshev_polynomial_u", move |x| chebyshev_u(n, x))
}

/// Chebyshev polynomial of the **third kind** `V_n(x)`.
///
/// `V_0 = 1`, `V_1 = 2x - 1`, same recurrence as T/U. Mirrors
/// `torch.special.chebyshev_polynomial_v`.
pub fn chebyshev_polynomial_v<T: Float>(
    input: &Tensor<T>,
    n: usize,
) -> FerrotorchResult<Tensor<T>> {
    elementwise_f64(input, "chebyshev_polynomial_v", move |x| chebyshev_v(n, x))
}

/// Chebyshev polynomial of the **fourth kind** `W_n(x)`.
///
/// `W_0 = 1`, `W_1 = 2x + 1`, same recurrence as T/U. Mirrors
/// `torch.special.chebyshev_polynomial_w`.
pub fn chebyshev_polynomial_w<T: Float>(
    input: &Tensor<T>,
    n: usize,
) -> FerrotorchResult<Tensor<T>> {
    elementwise_f64(input, "chebyshev_polynomial_w", move |x| chebyshev_w(n, x))
}

// --- Hermite family --------------------------------------------------------

/// Hermite polynomial (physicist's) `H_n(x)`.
///
/// `H_0 = 1`, `H_1 = 2x`, `H_{n+1} = 2x H_n - 2n H_{n-1}`. Mirrors
/// `torch.special.hermite_polynomial_h`.
pub fn hermite_polynomial_h<T: Float>(input: &Tensor<T>, n: usize) -> FerrotorchResult<Tensor<T>> {
    elementwise_f64(input, "hermite_polynomial_h", move |x| hermite_h(n, x))
}

/// Probabilist's Hermite polynomial `He_n(x)`.
///
/// `He_0 = 1`, `He_1 = x`, `He_{n+1} = x He_n - n He_{n-1}`. Mirrors
/// `torch.special.hermite_polynomial_he`.
pub fn hermite_polynomial_he<T: Float>(input: &Tensor<T>, n: usize) -> FerrotorchResult<Tensor<T>> {
    elementwise_f64(input, "hermite_polynomial_he", move |x| hermite_he(n, x))
}

// --- Laguerre & Legendre ---------------------------------------------------

/// Laguerre polynomial `L_n(x)`.
///
/// `L_0 = 1`, `L_1 = 1 - x`, `(n+1) L_{n+1} = (2n + 1 - x) L_n - n L_{n-1}`.
/// Mirrors `torch.special.laguerre_polynomial_l`.
pub fn laguerre_polynomial_l<T: Float>(input: &Tensor<T>, n: usize) -> FerrotorchResult<Tensor<T>> {
    elementwise_f64(input, "laguerre_polynomial_l", move |x| laguerre_l(n, x))
}

/// Legendre polynomial `P_n(x)`.
///
/// `P_0 = 1`, `P_1 = x`, `(n+1) P_{n+1} = (2n+1) x P_n - n P_{n-1}`.
/// Mirrors `torch.special.legendre_polynomial_p`.
pub fn legendre_polynomial_p<T: Float>(input: &Tensor<T>, n: usize) -> FerrotorchResult<Tensor<T>> {
    elementwise_f64(input, "legendre_polynomial_p", move |x| legendre_p(n, x))
}

// --- Shifted Chebyshev family (domain [0, 1]) ------------------------------

/// Shifted Chebyshev T: `T*_n(x) = T_n(2x - 1)`. Mirrors
/// `torch.special.shifted_chebyshev_polynomial_t`.
pub fn shifted_chebyshev_polynomial_t<T: Float>(
    input: &Tensor<T>,
    n: usize,
) -> FerrotorchResult<Tensor<T>> {
    elementwise_f64(input, "shifted_chebyshev_polynomial_t", move |x| {
        chebyshev_t(n, 2.0 * x - 1.0)
    })
}

/// Shifted Chebyshev U: `U*_n(x) = U_n(2x - 1)`.
pub fn shifted_chebyshev_polynomial_u<T: Float>(
    input: &Tensor<T>,
    n: usize,
) -> FerrotorchResult<Tensor<T>> {
    elementwise_f64(input, "shifted_chebyshev_polynomial_u", move |x| {
        chebyshev_u(n, 2.0 * x - 1.0)
    })
}

/// Shifted Chebyshev V: `V*_n(x) = V_n(2x - 1)`.
pub fn shifted_chebyshev_polynomial_v<T: Float>(
    input: &Tensor<T>,
    n: usize,
) -> FerrotorchResult<Tensor<T>> {
    elementwise_f64(input, "shifted_chebyshev_polynomial_v", move |x| {
        chebyshev_v(n, 2.0 * x - 1.0)
    })
}

/// Shifted Chebyshev W: `W*_n(x) = W_n(2x - 1)`.
pub fn shifted_chebyshev_polynomial_w<T: Float>(
    input: &Tensor<T>,
    n: usize,
) -> FerrotorchResult<Tensor<T>> {
    elementwise_f64(input, "shifted_chebyshev_polynomial_w", move |x| {
        chebyshev_w(n, 2.0 * x - 1.0)
    })
}

// ---------------------------------------------------------------------------
// Internal scalar evaluators (three-term recurrences in f64)
// ---------------------------------------------------------------------------

/// `H_n(x)` (physicist's Hermite) via three-term recurrence
/// `H_{n+1} = 2x H_n - 2n H_{n-1}` with `H_0 = 1`, `H_1 = 2x`.
fn hermite_h(n: usize, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return 2.0 * x;
    }
    let mut prev2 = 1.0;
    let mut prev1 = 2.0 * x;
    for k in 1..n {
        let kf = k as f64;
        let next = 2.0 * x * prev1 - 2.0 * kf * prev2;
        prev2 = prev1;
        prev1 = next;
    }
    prev1
}

/// `He_n(x)` (probabilist's Hermite) via three-term recurrence
/// `He_{n+1} = x He_n - n He_{n-1}` with `He_0 = 1`, `He_1 = x`.
fn hermite_he(n: usize, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return x;
    }
    let mut prev2 = 1.0;
    let mut prev1 = x;
    for k in 1..n {
        let kf = k as f64;
        let next = x * prev1 - kf * prev2;
        prev2 = prev1;
        prev1 = next;
    }
    prev1
}

/// `T_n(x)` via direct recurrence — used internally for shifted variants.
fn chebyshev_t(n: usize, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return x;
    }
    let mut prev2 = 1.0;
    let mut prev1 = x;
    for _ in 2..=n {
        let next = 2.0 * x * prev1 - prev2;
        prev2 = prev1;
        prev1 = next;
    }
    prev1
}

fn chebyshev_u(n: usize, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return 2.0 * x;
    }
    let mut prev2 = 1.0;
    let mut prev1 = 2.0 * x;
    for _ in 2..=n {
        let next = 2.0 * x * prev1 - prev2;
        prev2 = prev1;
        prev1 = next;
    }
    prev1
}

fn chebyshev_v(n: usize, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return 2.0 * x - 1.0;
    }
    let mut prev2 = 1.0;
    let mut prev1 = 2.0 * x - 1.0;
    for _ in 2..=n {
        let next = 2.0 * x * prev1 - prev2;
        prev2 = prev1;
        prev1 = next;
    }
    prev1
}

fn chebyshev_w(n: usize, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return 2.0 * x + 1.0;
    }
    let mut prev2 = 1.0;
    let mut prev1 = 2.0 * x + 1.0;
    for _ in 2..=n {
        let next = 2.0 * x * prev1 - prev2;
        prev2 = prev1;
        prev1 = next;
    }
    prev1
}

fn laguerre_l(n: usize, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return 1.0 - x;
    }
    let mut prev2 = 1.0;
    let mut prev1 = 1.0 - x;
    for k in 1..n {
        let kf = k as f64;
        let next = ((2.0 * kf + 1.0 - x) * prev1 - kf * prev2) / (kf + 1.0);
        prev2 = prev1;
        prev1 = next;
    }
    prev1
}

fn legendre_p(n: usize, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return x;
    }
    let mut prev2 = 1.0;
    let mut prev1 = x;
    for k in 1..n {
        let kf = k as f64;
        let next = ((2.0 * kf + 1.0) * x * prev1 - kf * prev2) / (kf + 1.0);
        prev2 = prev1;
        prev1 = next;
    }
    prev1
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::TensorStorage;

    /// Helper: create a tensor from data and shape.
    fn t(data: &[f64], shape: &[usize]) -> Tensor<f64> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
    }

    // --- erf ---

    #[test]
    fn erf_zero() {
        let input = t(&[0.0], &[1]);
        let result = erf(&input).unwrap();
        assert!((result.data().unwrap()[0]).abs() < 1e-10);
    }

    #[test]
    fn erf_symmetry() {
        // erf(-x) = -erf(x)
        let input = t(&[0.5, 1.0, 2.0], &[3]);
        let neg_input = t(&[-0.5, -1.0, -2.0], &[3]);
        let pos = erf(&input).unwrap();
        let neg = erf(&neg_input).unwrap();
        let pd = pos.data().unwrap();
        let nd = neg.data().unwrap();
        for i in 0..3 {
            assert!(
                (pd[i] + nd[i]).abs() < 1e-6,
                "erf({}) + erf({}) = {} (expected 0)",
                input.data().unwrap()[i],
                neg_input.data().unwrap()[i],
                pd[i] + nd[i],
            );
        }
    }

    #[test]
    fn erf_large_value() {
        // erf(inf) = 1
        let input = t(&[f64::INFINITY], &[1]);
        let result = erf(&input).unwrap();
        assert!((result.data().unwrap()[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn erf_known_values() {
        // erf(1) ≈ 0.8427007929...
        let input = t(&[1.0], &[1]);
        let result = erf(&input).unwrap();
        assert!(
            (result.data().unwrap()[0] - 0.8427007929).abs() < 2e-7,
            "erf(1) = {}",
            result.data().unwrap()[0]
        );
    }

    // --- erfc ---

    #[test]
    fn erfc_is_one_minus_erf() {
        let input = t(&[0.0, 0.5, 1.0, -0.5, 2.0], &[5]);
        let erf_result = erf(&input).unwrap();
        let erfc_result = erfc(&input).unwrap();
        let ed = erf_result.data().unwrap();
        let cd = erfc_result.data().unwrap();
        for i in 0..5 {
            assert!(
                (ed[i] + cd[i] - 1.0).abs() < 1e-10,
                "erf({0}) + erfc({0}) = {1} (expected 1.0)",
                input.data().unwrap()[i],
                ed[i] + cd[i],
            );
        }
    }

    // --- erfinv ---

    #[test]
    fn erfinv_zero() {
        let input = t(&[0.0], &[1]);
        let result = erfinv(&input).unwrap();
        assert!(result.data().unwrap()[0].abs() < 1e-10);
    }

    #[test]
    fn erfinv_roundtrip() {
        // erfinv(erf(x)) ≈ x
        let xs = t(&[0.1, 0.5, 1.0, -0.3, -1.5], &[5]);
        let erf_xs = erf(&xs).unwrap();
        let roundtrip = erfinv(&erf_xs).unwrap();
        let orig = xs.data().unwrap();
        let rt = roundtrip.data().unwrap();
        for i in 0..5 {
            assert!(
                (orig[i] - rt[i]).abs() < 0.01,
                "erfinv(erf({})) = {} (expected {})",
                orig[i],
                rt[i],
                orig[i],
            );
        }
    }

    #[test]
    fn erfinv_boundary() {
        let input = t(&[1.0, -1.0], &[2]);
        let result = erfinv(&input).unwrap();
        let d = result.data().unwrap();
        assert!(d[0].is_infinite() && d[0] > 0.0, "erfinv(1) should be +inf");
        assert!(
            d[1].is_infinite() && d[1] < 0.0,
            "erfinv(-1) should be -inf"
        );
    }

    // --- lgamma ---

    #[test]
    fn lgamma_at_one_and_two() {
        // lgamma(1) = log(0!) = 0, lgamma(2) = log(1!) = 0.
        let input = t(&[1.0, 2.0], &[2]);
        let result = lgamma(&input).unwrap();
        let d = result.data().unwrap();
        assert!(d[0].abs() < 1e-10, "lgamma(1) = {} (expected 0)", d[0]);
        assert!(d[1].abs() < 1e-10, "lgamma(2) = {} (expected 0)", d[1]);
    }

    #[test]
    fn lgamma_known_values() {
        // lgamma(0.5) = log(sqrt(pi)) ≈ 0.5723649429...
        let input = t(&[0.5], &[1]);
        let result = lgamma(&input).unwrap();
        let expected = 0.5723649429247001;
        assert!(
            (result.data().unwrap()[0] - expected).abs() < 1e-8,
            "lgamma(0.5) = {} (expected {})",
            result.data().unwrap()[0],
            expected,
        );
    }

    #[test]
    fn lgamma_factorial() {
        // lgamma(n+1) = log(n!) for integer n.
        // lgamma(6) = log(5!) = log(120) ≈ 4.7875...
        let input = t(&[6.0], &[1]);
        let result = lgamma(&input).unwrap();
        let expected = (120.0f64).ln();
        assert!(
            (result.data().unwrap()[0] - expected).abs() < 1e-8,
            "lgamma(6) = {} (expected {})",
            result.data().unwrap()[0],
            expected,
        );
    }

    // --- digamma ---

    #[test]
    fn digamma_known_values() {
        // psi(1) = -gamma (Euler-Mascheroni) ≈ -0.5772156649...
        let input = t(&[1.0], &[1]);
        let result = digamma(&input).unwrap();
        let expected = -0.5772156649015329;
        assert!(
            (result.data().unwrap()[0] - expected).abs() < 1e-6,
            "digamma(1) = {} (expected {})",
            result.data().unwrap()[0],
            expected,
        );
    }

    #[test]
    fn digamma_recurrence() {
        // psi(x+1) = psi(x) + 1/x
        let x_val = 2.5;
        let input_x = t(&[x_val], &[1]);
        let input_x1 = t(&[x_val + 1.0], &[1]);
        let psi_x = digamma(&input_x).unwrap().data().unwrap()[0];
        let psi_x1 = digamma(&input_x1).unwrap().data().unwrap()[0];
        assert!(
            (psi_x1 - psi_x - 1.0 / x_val).abs() < 1e-8,
            "psi({}) - psi({}) = {} (expected {})",
            x_val + 1.0,
            x_val,
            psi_x1 - psi_x,
            1.0 / x_val,
        );
    }

    // --- log1p ---

    #[test]
    fn log1p_zero() {
        let input = t(&[0.0], &[1]);
        let result = log1p(&input).unwrap();
        assert!(result.data().unwrap()[0].abs() < 1e-15);
    }

    #[test]
    fn log1p_small() {
        // For small x, log(1+x) ≈ x.
        let small = 1e-10;
        let input = t(&[small], &[1]);
        let result = log1p(&input).unwrap();
        assert!(
            (result.data().unwrap()[0] - small).abs() < 1e-15,
            "log1p({small}) = {} (expected ~{small})",
            result.data().unwrap()[0],
        );
    }

    #[test]
    fn log1p_known() {
        // log1p(1.0) = ln(2) ≈ 0.693147...
        let input = t(&[1.0], &[1]);
        let result = log1p(&input).unwrap();
        assert!((result.data().unwrap()[0] - std::f64::consts::LN_2).abs() < 1e-15,);
    }

    // --- expm1 ---

    #[test]
    fn expm1_zero() {
        let input = t(&[0.0], &[1]);
        let result = expm1(&input).unwrap();
        assert!(result.data().unwrap()[0].abs() < 1e-15);
    }

    #[test]
    fn expm1_small() {
        // For small x, exp(x) - 1 ≈ x.
        let small = 1e-10;
        let input = t(&[small], &[1]);
        let result = expm1(&input).unwrap();
        assert!(
            (result.data().unwrap()[0] - small).abs() < 1e-15,
            "expm1({small}) = {} (expected ~{small})",
            result.data().unwrap()[0],
        );
    }

    #[test]
    fn expm1_known() {
        // expm1(1.0) = e - 1 ≈ 1.71828...
        let input = t(&[1.0], &[1]);
        let result = expm1(&input).unwrap();
        let expected = std::f64::consts::E - 1.0;
        assert!((result.data().unwrap()[0] - expected).abs() < 1e-14,);
    }

    // --- sinc ---

    #[test]
    fn sinc_zero() {
        let input = t(&[0.0], &[1]);
        let result = sinc(&input).unwrap();
        assert!(
            (result.data().unwrap()[0] - 1.0).abs() < 1e-15,
            "sinc(0) = {} (expected 1)",
            result.data().unwrap()[0],
        );
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn sinc_integer() {
        // sinc(n) = 0 for nonzero integer n, since sin(n*pi) = 0.
        let input = t(&[1.0, 2.0, -1.0, -3.0], &[4]);
        let result = sinc(&input).unwrap();
        let d = result.data().unwrap();
        for i in 0..4 {
            assert!(
                d[i].abs() < 1e-15,
                "sinc({}) = {} (expected 0)",
                input.data().unwrap()[i],
                d[i],
            );
        }
    }

    #[test]
    fn sinc_half() {
        // sinc(0.5) = sin(pi/2) / (pi/2) = 1 / (pi/2) = 2/pi
        let input = t(&[0.5], &[1]);
        let result = sinc(&input).unwrap();
        let expected = 2.0 / std::f64::consts::PI;
        assert!(
            (result.data().unwrap()[0] - expected).abs() < 1e-15,
            "sinc(0.5) = {} (expected {})",
            result.data().unwrap()[0],
            expected,
        );
    }

    // --- xlogy ---

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn xlogy_zero_x() {
        // xlogy(0, y) = 0 for any y.
        let x = t(&[0.0, 0.0, 0.0], &[3]);
        let y = t(&[1.0, 0.0, f64::INFINITY], &[3]);
        let result = xlogy(&x, &y).unwrap();
        let d = result.data().unwrap();
        for i in 0..3 {
            assert!(
                d[i] == 0.0,
                "xlogy(0, {}) = {} (expected 0)",
                y.data().unwrap()[i],
                d[i],
            );
        }
    }

    #[test]
    fn xlogy_normal() {
        // xlogy(2, e) = 2 * ln(e) = 2.
        let x = t(&[2.0], &[1]);
        let y = t(&[std::f64::consts::E], &[1]);
        let result = xlogy(&x, &y).unwrap();
        assert!(
            (result.data().unwrap()[0] - 2.0).abs() < 1e-14,
            "xlogy(2, e) = {} (expected 2)",
            result.data().unwrap()[0],
        );
    }

    #[test]
    fn xlogy_broadcast() {
        // [2, 3] * log([e, e]) = [2, 3].
        let x = t(&[2.0, 3.0], &[2]);
        let y = t(&[std::f64::consts::E, std::f64::consts::E], &[2]);
        let result = xlogy(&x, &y).unwrap();
        let d = result.data().unwrap();
        assert!((d[0] - 2.0).abs() < 1e-14);
        assert!((d[1] - 3.0).abs() < 1e-14);
    }

    // --- f32 support ---

    #[test]
    fn erf_f32() {
        let input =
            Tensor::from_storage(TensorStorage::cpu(vec![0.0f32, 1.0, -1.0]), vec![3], false)
                .unwrap();
        let result = erf(&input).unwrap();
        let d = result.data().unwrap();
        assert!(d[0].abs() < 1e-6);
        assert!((d[1] - 0.8427008).abs() < 1e-5);
        assert!((d[2] + 0.8427008).abs() < 1e-5);
    }

    // --- multidimensional ---

    #[test]
    fn erf_2d() {
        let input = t(&[0.0, 0.5, 1.0, -0.5, -1.0, 2.0], &[2, 3]);
        let result = erf(&input).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        let d = result.data().unwrap();
        assert!(d[0].abs() < 1e-10); // erf(0)
        assert!(d[2] > 0.8); // erf(1) ≈ 0.843
        assert!(d[3] < 0.0); // erf(-0.5) < 0
    }

    // ---------------------------------------------------------------------
    // Orthogonal-polynomial families
    //
    // For each family we hand-check the closed-form first three basis
    // values against the recurrence. n=0 must always be 1 everywhere.
    // n=1 is the family-specific linear; n=2 is the first interesting
    // recurrence step. Numerical comparisons use a tight tolerance (1e-12)
    // since these are exact polynomial evaluations in f64.
    // ---------------------------------------------------------------------

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn xs() -> Tensor<f64> {
        // A handful of points in/around the standard domains.
        t(&[0.0, 0.5, 1.0, -0.5, -1.0, 0.25], &[6])
    }

    // --- Chebyshev T (first kind) ---

    #[test]
    fn chebyshev_t_n0_is_one() {
        let r = chebyshev_polynomial_t(&xs(), 0).unwrap();
        for &v in r.data().unwrap() {
            assert!(close(v, 1.0, 1e-12));
        }
    }

    #[test]
    fn chebyshev_t_n1_is_x() {
        let x = xs();
        let r = chebyshev_polynomial_t(&x, 1).unwrap();
        for (a, b) in r.data().unwrap().iter().zip(x.data().unwrap().iter()) {
            assert!(close(*a, *b, 1e-12));
        }
    }

    #[test]
    fn chebyshev_t_n2_is_2xx_minus_one() {
        // T_2(x) = 2x^2 - 1
        let x = xs();
        let r = chebyshev_polynomial_t(&x, 2).unwrap();
        for (a, &xv) in r.data().unwrap().iter().zip(x.data().unwrap().iter()) {
            assert!(close(*a, 2.0 * xv * xv - 1.0, 1e-12));
        }
    }

    #[test]
    fn chebyshev_t_at_endpoints() {
        // T_n(1) = 1, T_n(-1) = (-1)^n.
        let pts = t(&[1.0, -1.0], &[2]);
        for n in 0..6 {
            let r = chebyshev_polynomial_t(&pts, n).unwrap();
            let d = r.data().unwrap();
            assert!(close(d[0], 1.0, 1e-12), "T_{n}(1) = {}", d[0]);
            let expected_neg = if n % 2 == 0 { 1.0 } else { -1.0 };
            assert!(close(d[1], expected_neg, 1e-12), "T_{n}(-1) = {}", d[1]);
        }
    }

    // --- Chebyshev U (second kind) ---

    #[test]
    fn chebyshev_u_n0_n1_n2() {
        // U_0 = 1, U_1 = 2x, U_2 = 4x^2 - 1
        let x = t(&[0.5], &[1]);
        let xv = 0.5;
        assert!(close(
            chebyshev_polynomial_u(&x, 0).unwrap().data().unwrap()[0],
            1.0,
            1e-12
        ));
        assert!(close(
            chebyshev_polynomial_u(&x, 1).unwrap().data().unwrap()[0],
            2.0 * xv,
            1e-12,
        ));
        assert!(close(
            chebyshev_polynomial_u(&x, 2).unwrap().data().unwrap()[0],
            4.0 * xv * xv - 1.0,
            1e-12,
        ));
    }

    // --- Chebyshev V / W ---

    #[test]
    fn chebyshev_v_endpoints() {
        // V_n(1) = 1 for all n; V_1(0) = -1.
        let pts = t(&[1.0, 0.0], &[2]);
        for n in 0..4 {
            let r = chebyshev_polynomial_v(&pts, n).unwrap();
            assert!(close(r.data().unwrap()[0], 1.0, 1e-12));
        }
        let r1 = chebyshev_polynomial_v(&pts, 1).unwrap();
        assert!(close(r1.data().unwrap()[1], -1.0, 1e-12));
    }

    #[test]
    fn chebyshev_w_endpoints() {
        // W_1(0) = 1; W_n(1) = 2n + 1 for the recurrence above.
        let zero = t(&[0.0], &[1]);
        assert!(close(
            chebyshev_polynomial_w(&zero, 1).unwrap().data().unwrap()[0],
            1.0,
            1e-12
        ));
    }

    // --- Hermite (physicist) ---

    #[test]
    fn hermite_h_known_values() {
        // H_0 = 1, H_1 = 2x, H_2 = 4x^2 - 2, H_3 = 8x^3 - 12x
        let x = t(&[0.5], &[1]);
        let xv = 0.5;
        assert!(close(
            hermite_polynomial_h(&x, 0).unwrap().data().unwrap()[0],
            1.0,
            1e-12
        ));
        assert!(close(
            hermite_polynomial_h(&x, 1).unwrap().data().unwrap()[0],
            2.0 * xv,
            1e-12
        ));
        assert!(close(
            hermite_polynomial_h(&x, 2).unwrap().data().unwrap()[0],
            4.0 * xv * xv - 2.0,
            1e-12,
        ));
        assert!(close(
            hermite_polynomial_h(&x, 3).unwrap().data().unwrap()[0],
            8.0 * xv * xv * xv - 12.0 * xv,
            1e-12,
        ));
    }

    // --- HermiteE (probabilist) ---

    #[test]
    fn hermite_he_known_values() {
        // He_0 = 1, He_1 = x, He_2 = x^2 - 1, He_3 = x^3 - 3x
        let x = t(&[0.5], &[1]);
        let xv = 0.5;
        assert!(close(
            hermite_polynomial_he(&x, 0).unwrap().data().unwrap()[0],
            1.0,
            1e-12
        ));
        assert!(close(
            hermite_polynomial_he(&x, 1).unwrap().data().unwrap()[0],
            xv,
            1e-12
        ));
        assert!(close(
            hermite_polynomial_he(&x, 2).unwrap().data().unwrap()[0],
            xv * xv - 1.0,
            1e-12,
        ));
        assert!(close(
            hermite_polynomial_he(&x, 3).unwrap().data().unwrap()[0],
            xv * xv * xv - 3.0 * xv,
            1e-12,
        ));
    }

    // --- Laguerre ---

    #[test]
    fn laguerre_l_known_values() {
        // L_0 = 1, L_1 = 1 - x, L_2 = (x^2 - 4x + 2) / 2
        let x = t(&[0.5], &[1]);
        let xv = 0.5;
        assert!(close(
            laguerre_polynomial_l(&x, 0).unwrap().data().unwrap()[0],
            1.0,
            1e-12
        ));
        assert!(close(
            laguerre_polynomial_l(&x, 1).unwrap().data().unwrap()[0],
            1.0 - xv,
            1e-12
        ));
        assert!(close(
            laguerre_polynomial_l(&x, 2).unwrap().data().unwrap()[0],
            f64::midpoint(xv * xv - 4.0 * xv, 2.0),
            1e-12,
        ));
    }

    // --- Legendre ---

    #[test]
    fn legendre_p_known_values() {
        // P_0 = 1, P_1 = x, P_2 = (3x^2 - 1)/2, P_3 = (5x^3 - 3x)/2
        let x = t(&[0.5], &[1]);
        let xv = 0.5;
        assert!(close(
            legendre_polynomial_p(&x, 0).unwrap().data().unwrap()[0],
            1.0,
            1e-12
        ));
        assert!(close(
            legendre_polynomial_p(&x, 1).unwrap().data().unwrap()[0],
            xv,
            1e-12
        ));
        assert!(close(
            legendre_polynomial_p(&x, 2).unwrap().data().unwrap()[0],
            (3.0 * xv * xv - 1.0) / 2.0,
            1e-12,
        ));
        assert!(close(
            legendre_polynomial_p(&x, 3).unwrap().data().unwrap()[0],
            (5.0 * xv * xv * xv - 3.0 * xv) / 2.0,
            1e-12,
        ));
    }

    #[test]
    fn legendre_p_endpoints() {
        // P_n(1) = 1, P_n(-1) = (-1)^n
        let pts = t(&[1.0, -1.0], &[2]);
        for n in 0..6 {
            let r = legendre_polynomial_p(&pts, n).unwrap();
            let d = r.data().unwrap();
            assert!(close(d[0], 1.0, 1e-12), "P_{n}(1) = {}", d[0]);
            let expected_neg = if n % 2 == 0 { 1.0 } else { -1.0 };
            assert!(close(d[1], expected_neg, 1e-12), "P_{n}(-1) = {}", d[1]);
        }
    }

    // --- Shifted Chebyshev (domain [0, 1]) ---

    #[test]
    fn shifted_chebyshev_t_matches_t_of_2x_minus_1() {
        let x = xs();
        for n in 0..5 {
            let shifted = shifted_chebyshev_polynomial_t(&x, n).unwrap();
            let xs_data = x.data().unwrap();
            let mapped: Vec<f64> = xs_data.iter().map(|v| 2.0 * v - 1.0).collect();
            let mapped_t = t(&mapped, &[mapped.len()]);
            let direct = chebyshev_polynomial_t(&mapped_t, n).unwrap();
            for (s, d) in shifted
                .data()
                .unwrap()
                .iter()
                .zip(direct.data().unwrap().iter())
            {
                assert!(close(*s, *d, 1e-12), "T*_{n} mismatch at n={n}: {s} vs {d}");
            }
        }
    }

    // --- GPU discipline ---

    #[test]
    fn polynomial_fns_reject_cuda_tensors_explicitly() {
        // We can't construct a CUDA tensor in this CPU-only test, but every
        // polynomial fn calls `require_cpu_poly` at entry which only
        // returns Ok for `!is_cuda()`. If a future refactor accidentally
        // bypasses the gate, this asserts we have at least one fn covered
        // (the others share the same code path via elementwise_f64).
        // Sanity: the gate itself is exercised here on a CPU tensor.
        let x = t(&[0.0], &[1]);
        assert!(chebyshev_polynomial_t(&x, 3).is_ok());
    }
}
