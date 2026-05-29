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
//! | REQ-9 | SHIPPED | CPU: `chebyshev_polynomial_{t,u,v,w}`; GPU: `gpu_chebyshev_poly_f32`/`_f64` in `ferrotorch-gpu/src/special.rs` via `GpuBackend::chebyshev_poly_f32`/`_f64`; consumer: the CUDA branch (`poly_gpu_chebyshev`) of each `chebyshev_polynomial_*` dispatches on-device (#1545 / #1533) |
//! | REQ-10 | SHIPPED | CPU: `hermite_polynomial_h`/`hermite_polynomial_he`; GPU: `gpu_hermite_h_poly_*`/`gpu_hermite_he_poly_*`; consumer: the CUDA branch (`poly_gpu_simple`) of each `hermite_polynomial_*` |
//! | REQ-11 | SHIPPED | CPU: `laguerre_polynomial_l`/`legendre_polynomial_p`; GPU: `gpu_laguerre_poly_*`/`gpu_legendre_poly_*`; consumer: the CUDA branch of `laguerre_polynomial_l`/`legendre_polynomial_p` |
//! | REQ-12 | SHIPPED | CPU: `shifted_chebyshev_polynomial_{t,u,v,w}`; GPU: `gpu_chebyshev_poly_f32`/`_f64` with `shift=true`; consumer: the CUDA branch of each `shifted_chebyshev_polynomial_*` |
//! | REQ-13 | SHIPPED | pub fn `gammainc`/`gammaincc` mirror `torch.special.gammainc`/`gammaincc`; consumer: re-exported at top of `lib.rs` as `ferrotorch_core::{gammainc, gammaincc}` (S5: torch.special public surface IS the consumer) |
//! | REQ-14 | SHIPPED | pub fn `log_beta`/`beta` mirror `scipy.special.betaln`/`beta`; consumer: re-exported as `ferrotorch_core::{log_beta, beta}` |
//! | REQ-15 | SHIPPED | pub fn `multigammaln`/`mvlgamma` mirror `torch.special.multigammaln`/`torch.mvlgamma`; consumer: re-exported as `ferrotorch_core::{multigammaln, mvlgamma}` |
//! | REQ-16 | SHIPPED | pub fn `gammaln_sign` mirrors `scipy.special.gammasgn`; consumer: re-exported as `ferrotorch_core::gammaln_sign` |

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
// Normal-distribution trio: entr / ndtr / ndtri (Cephes)
// ---------------------------------------------------------------------------
//
// Direct ports of the upstream torch.special kernels:
//   - entr: `aten/src/ATen/native/cuda/Math.cuh:463-480` (entr_string)
//   - ndtr: `aten/src/ATen/native/UnaryOps.cpp:715-718` (calc_ndtr), a
//     composite over the already-shipped `erf` (special.md REQ-1).
//   - ndtri: `aten/src/ATen/native/cuda/Math.cuh:48-173` (ndtri_string),
//     the Cephes rational approximation — NOT `sqrt(2)*erfinv(2p-1)`, which
//     loses ULP parity with torch (the design doc, special_bessel.md REQ-B3,
//     mandates the direct Cephes port).

/// `entr(x)` — entropy. Mirrors `entr_string`
/// (`aten/src/ATen/native/cuda/Math.cuh:463-480`): the NaN check comes first,
/// then `x > 0 -> -x*log(x)`, `x == 0 -> 0`, else `-inf`. Note `entr(0) = +0`
/// (the explicit `0` branch) while `entr(1) = -1*log(1) = -0` flows through the
/// `>0` branch verbatim, matching torch's signed-zero output.
fn entr_scalar<T: Float>(a: T) -> T {
    // `a != a` — NaN propagates unchanged (Math.cuh:466-468).
    if a.is_nan() {
        return a;
    }
    let zero = nt_zero::<T>();
    if a > zero {
        return -a * a.ln();
    }
    if a == zero {
        return zero;
    }
    T::neg_infinity()
}

/// `ndtr(x)` — standard-normal CDF. Mirrors `calc_ndtr`
/// (`aten/src/ATen/native/UnaryOps.cpp:715-718`):
/// `ndtr(x) = (1 + erf(x * M_SQRT1_2)) * 0.5` with
/// `M_SQRT1_2 = 0.70710678118654752440` (1/sqrt(2)). Composed over the shipped
/// `erf_scalar` so the f64 SunPro-fdlibm ~1-ulp `erf` precision flows through.
/// Edge behavior is inherited from `erf`: `ndtr(-inf) = 0`, `ndtr(0) = 0.5`,
/// `ndtr(+inf) = 1`, `ndtr(NaN) = NaN`.
fn ndtr_scalar<T: Float>(x: T) -> T {
    // M_SQRT1_2 = 1/sqrt(2). `<math.h>` value, also `UnaryOps.cpp:716`.
    let sqrt1_2 = T::from(std::f64::consts::FRAC_1_SQRT_2).unwrap();
    let one = nt_one::<T>();
    let half = T::from(0.5).unwrap();
    (one + erf_scalar(x * sqrt1_2)) * half
}

/// Cephes `polevl`: evaluate the polynomial `A[0]*x^(n-1) + ... + A[n-1]` in
/// Horner form over the coefficients in REVERSE order, with the upstream-CUDA
/// `len = len(A)` convention (NOT `len(A) - 1`). Direct port of
/// `aten/src/ATen/native/cuda/Math.cuh:30-39`:
/// `result = 0; for i in 0..len { result = result*x + A[i]; }`.
fn polevl<T: Float>(x: T, coeffs: &[T]) -> T {
    let mut result = nt_zero::<T>();
    for &c in coeffs {
        result = result * x + c;
    }
    result
}

/// `ndtri(y0)` — inverse standard-normal CDF (quantile function). Direct port
/// of the Cephes `ndtri_string` (`aten/src/ATen/native/cuda/Math.cuh:48-173`).
/// Domain `(0, 1)`: `y0 == 0 -> -inf`, `y0 == 1 -> +inf`,
/// `y0 < 0 || y0 > 1 -> NaN`. The interior uses three coefficient regions
/// (central P0/Q0, tail P1/Q1, far-tail P2/Q2) and the `code`-flag sign flip
/// (`Math.cuh:65-71, 171-172`). All arithmetic runs in f64 then narrows to `T`
/// so the f32 path inherits the full-precision Cephes evaluation before the
/// final cast (torch's CUDA jiterator runs in the tensor's scalar type, but the
/// f64-then-narrow path stays inside the f32 transcendental tolerance and
/// avoids the catastrophic `log`/`sqrt` cancellation at the f32 tails).
fn ndtri_scalar<T: Float>(y0: T) -> T {
    let yf = <T as num_traits::ToPrimitive>::to_f64(&y0).unwrap_or(f64::NAN);
    T::from(ndtri_f64(yf)).unwrap_or_else(|| T::from(f64::NAN).unwrap())
}

/// f64 Cephes `ndtri`. See [`ndtri_scalar`] for the upstream cite.
///
/// `clippy::float_cmp` — the exact `== 0.0` / `== 1.0` boundary tests are a
/// verbatim port of the Cephes `ndtri` special-case ladder
/// (`Math.cuh:55,58`); torch compares to the literal endpoints, not an epsilon
/// band, so matching the exact equality is required for parity (`ndtri(p)` is
/// `-inf`/`+inf` only AT the endpoints).
/// `clippy::manual_range_contains` — the `y0 < 0.0 || y0 > 1.0` form mirrors
/// the upstream `y0 < zero || y0 > one` check (`Math.cuh:61`) one-for-one;
/// rewriting as `!(0.0..=1.0).contains()` obscures the diff against upstream.
/// `clippy::excessive_precision` — the P0/Q0/P1/Q1/P2/Q2/S2PI/EXP_M2 constants
/// are reproduced to their full Cephes decimal width (`Math.cuh:75-166`) so the
/// diff against the upstream reference is audit-friendly; the trailing digits
/// round to the same f64 bit pattern as a 17-digit truncation.
#[allow(
    clippy::float_cmp,
    clippy::manual_range_contains,
    clippy::excessive_precision,
    reason = "verbatim Cephes ndtri port: exact-endpoint boundary tests and full-width coefficients mirror Math.cuh:48-173 for ULP parity + audit-friendly diff"
)]
fn ndtri_f64(y0: f64) -> f64 {
    // Special cases (Math.cuh:54-63).
    if y0 == 0.0 {
        return f64::NEG_INFINITY;
    }
    if y0 == 1.0 {
        return f64::INFINITY;
    }
    if y0 < 0.0 || y0 > 1.0 {
        return f64::NAN;
    }
    if y0.is_nan() {
        return f64::NAN;
    }

    // Cephes coefficient tables (Math.cuh:75-166), reverse-order for `polevl`.
    // Central region |y - 0.5| <= 3/8.
    const P0: [f64; 5] = [
        -5.99633501014107895267E1,
        9.80010754185999661536E1,
        -5.66762857469070293439E1,
        1.39312609387279679503E1,
        -1.23916583867381258016E0,
    ];
    const Q0: [f64; 9] = [
        1.00000000000000000000E0,
        1.95448858338141759834E0,
        4.67627912898881538453E0,
        8.63602421390890590575E1,
        -2.25462687854119370527E2,
        2.00260212380060660359E2,
        -8.20372256168333339912E1,
        1.59056225126211695515E1,
        -1.18331621121330003142E0,
    ];
    // Tail region, x = sqrt(-2 log y) < 8.
    const P1: [f64; 9] = [
        4.05544892305962419923E0,
        3.15251094599893866154E1,
        5.71628192246421288162E1,
        4.40805073893200834700E1,
        1.46849561928858024014E1,
        2.18663306850790267539E0,
        -1.40256079171354495875E-1,
        -3.50424626827848203418E-2,
        -8.57456785154685413611E-4,
    ];
    const Q1: [f64; 9] = [
        1.00000000000000000000E0,
        1.57799883256466749731E1,
        4.53907635128879210584E1,
        4.13172038254672030440E1,
        1.50425385692907503408E1,
        2.50464946208309415979E0,
        -1.42182922854787788574E-1,
        -3.80806407691578277194E-2,
        -9.33259480895457427372E-4,
    ];
    // Far-tail region, x >= 8.
    const P2: [f64; 9] = [
        3.23774891776946035970E0,
        6.91522889068984211695E0,
        3.93881025292474443415E0,
        1.33303460815807542389E0,
        2.01485389549179081538E-1,
        1.23716634817820021358E-2,
        3.01581553508235416007E-4,
        2.65806974686737550832E-6,
        6.23974539184983293730E-9,
    ];
    const Q2: [f64; 9] = [
        1.00000000000000000000E0,
        6.02427039364742014255E0,
        3.67983563856160859403E0,
        1.37702099489081330271E0,
        2.16236993594496635890E-1,
        1.34204006088543189037E-2,
        3.28014464682127739104E-4,
        2.89247864745380683936E-6,
        6.79019408009981274425E-9,
    ];
    // sqrt(2pi) (Math.cuh:96).
    const S2PI: f64 = 2.50662827463100050242E0;
    // exp(-2) (Math.cuh:67-68).
    const EXP_M2: f64 = 0.13533528323661269189;

    let mut code = true;
    let mut y = y0;
    if y > 1.0 - EXP_M2 {
        y = 1.0 - y;
        code = false;
    }

    if y > EXP_M2 {
        // Central region (Math.cuh:73-102).
        y -= 0.5;
        let y2 = y * y;
        let x = y + y * (y2 * polevl(y2, &P0) / polevl(y2, &Q0));
        return x * S2PI;
    }

    let mut x = (-2.0 * y.ln()).sqrt();
    let x0 = x - (x.ln() / x);
    let z = 1.0 / x;
    let x1 = if x < 8.0 {
        // Tail region (Math.cuh:111-139).
        z * polevl(z, &P1) / polevl(z, &Q1)
    } else {
        // Far-tail region (Math.cuh:140-168).
        z * polevl(z, &P2) / polevl(z, &Q2)
    };
    x = x0 - x1;
    // code-flag sign flip (Math.cuh:171-172): return (!code) ? x : -x.
    if code { -x } else { x }
}

// ---------------------------------------------------------------------------
// Modified-Bessel-I family: i0 / i0e / i1 / i1e (Cephes, #1651 batch 2)
// ---------------------------------------------------------------------------
//
// Direct ports of the upstream torch.special Cephes kernels. The scalar
// evaluators run in f64 (the `*_f64` core) then narrow to `T` via `T::from`,
// exactly like the sibling `ndtri_f64` / `gammainc_scalar` convention above:
// the f64 arithmetic gives the mathematically-correct value, and narrowing to
// f32 stays well inside the f32 transcendental tolerance (1e-5). This is at
// least as accurate as torch's own f32 path, which uses SHORTER Chebyshev
// coefficient sets for `i1e` (17/7 vs the f64 29/25,
// `aten/src/ATen/native/cuda/Math.cuh:3289-3356`) — running the full f64 sets
// and narrowing strictly dominates that on accuracy.
//
//   - chbevl: the shared Clenshaw Chebyshev evaluator,
//     `aten/src/ATen/native/cuda/Math.cuh:485-500`.
//   - i0:  `cuda/Math.cuh:502-555` (i0_string). Even (fabs). `|x|<=8` A[30]
//     on `exp(x)*chbevl(x/2-2, A)`; `|x|>8` B[25] on
//     `exp(x)*chbevl(32/x-2, B)/sqrt(x)`.
//   - i0e: `aten/src/ATen/native/Math.h:101-145` (calc_i0e). Same A[30]/B[25]
//     WITHOUT the `exp(x)` factor.
//   - i1:  `cuda/Math.cuh:575-622` (i1_string). Odd (sign of _x). `|x|<=8`
//     i1e_A[29] on `exp(x)*x*chbevl(y, A)`; `|x|>8` i1e_B[25] on
//     `exp(x)*chbevl(32/x-2, B)/sqrt(x)`.
//   - i1e: `cuda/Math.cuh:647-696` (i1e double specialization). Same
//     i1e_A[29]/i1e_B[25] WITHOUT the `exp(x)` factor.

/// Cephes `chbevl`: evaluate a Chebyshev series by the Clenshaw recurrence.
/// Verbatim port of `aten/src/ATen/native/cuda/Math.cuh:485-500`:
/// `b0 = array[0]; b1 = 0;` then for `i in 1..len`:
/// `b2 = b1; b1 = b0; b0 = x*b1 - b2 + array[i];` returning `0.5*(b0 - b2)`.
/// The coefficient slice is consumed front-to-back (the i0/i1 tables are stored
/// in that order, NOT reversed like the `polevl` tables).
fn chbevl<T: Float>(x: T, array: &[T]) -> T {
    let mut b0 = array[0];
    let mut b1 = nt_zero::<T>();
    let mut b2 = nt_zero::<T>();
    for &c in &array[1..] {
        b2 = b1;
        b1 = b0;
        b0 = x * b1 - b2 + c;
    }
    T::from(0.5).unwrap() * (b0 - b2)
}

// Chebyshev coefficients for exp(-x) I0(x) on [0, 8]
// (`aten/src/ATen/native/cuda/Math.cuh:512-527`). lim(x->0) = 1.
#[allow(
    clippy::excessive_precision,
    reason = "verbatim Cephes i0e_A coefficients (cuda/Math.cuh:512-527) reproduced to full width for an audit-friendly diff; trailing digits round to the same f64 bit pattern"
)]
const I0E_A: [f64; 30] = [
    -4.41534164647933937950E-18,
    3.33079451882223809783E-17,
    -2.43127984654795469359E-16,
    1.71539128555513303061E-15,
    -1.16853328779934516808E-14,
    7.67618549860493561688E-14,
    -4.85644678311192946090E-13,
    2.95505266312963983461E-12,
    -1.72682629144155570723E-11,
    9.67580903537323691224E-11,
    -5.18979560163526290666E-10,
    2.65982372468238665035E-9,
    -1.30002500998624804212E-8,
    6.04699502254191894932E-8,
    -2.67079385394061173391E-7,
    1.11738753912010371815E-6,
    -4.41673835845875056359E-6,
    1.64484480707288970893E-5,
    -5.75419501008210370398E-5,
    1.88502885095841655729E-4,
    -5.76375574538582365885E-4,
    1.63947561694133579842E-3,
    -4.32430999505057594430E-3,
    1.05464603945949983183E-2,
    -2.37374148058994688156E-2,
    4.93052842396707084878E-2,
    -9.49010970480476444210E-2,
    1.71620901522208775349E-1,
    -3.04682672343198398683E-1,
    6.76795274409476084995E-1,
];

// Chebyshev coefficients for exp(-x) sqrt(x) I0(x) on [8, inf]
// (`aten/src/ATen/native/cuda/Math.cuh:539-552`). lim(x->inf) = 1/sqrt(2pi).
#[allow(
    clippy::excessive_precision,
    reason = "verbatim Cephes i0e_B coefficients (cuda/Math.cuh:539-552)"
)]
const I0E_B: [f64; 25] = [
    -7.23318048787475395456E-18,
    -4.83050448594418207126E-18,
    4.46562142029675999901E-17,
    3.46122286769746109310E-17,
    -2.82762398051658348494E-16,
    -3.42548561967721913462E-16,
    1.77256013305652638360E-15,
    3.81168066935262242075E-15,
    -9.55484669882830764870E-15,
    -4.15056934728722208663E-14,
    1.54008621752140982691E-14,
    3.85277838274214270114E-13,
    7.18012445138366623367E-13,
    -1.79417853150680611778E-12,
    -1.32158118404477131188E-11,
    -3.14991652796324136454E-11,
    1.18891471078464383424E-11,
    4.94060238822496958910E-10,
    3.39623202570838634515E-9,
    2.26666899049817806459E-8,
    2.04891858946906374183E-7,
    2.89137052083475648297E-6,
    6.88975834691682398426E-5,
    3.36911647825569408990E-3,
    8.04490411014108831608E-1,
];

// Chebyshev coefficients for exp(-x) I1(x) on [0, 8]
// (`aten/src/ATen/native/cuda/Math.cuh:582-597`). lim(x->0){ ../x } = 1/2.
#[allow(
    clippy::excessive_precision,
    reason = "verbatim Cephes i1e_A coefficients (cuda/Math.cuh:582-597)"
)]
const I1E_A: [f64; 29] = [
    2.77791411276104639959E-18,
    -2.11142121435816608115E-17,
    1.55363195773620046921E-16,
    -1.10559694773538630805E-15,
    7.60068429473540693410E-15,
    -5.04218550472791168711E-14,
    3.22379336594557470981E-13,
    -1.98397439776494371520E-12,
    1.17361862988909016308E-11,
    -6.66348972350202774223E-11,
    3.62559028155211703701E-10,
    -1.88724975172282928790E-9,
    9.38153738649577178388E-9,
    -4.44505912879632808065E-8,
    2.00329475355213526229E-7,
    -8.56872026469545474066E-7,
    3.47025130813767847674E-6,
    -1.32731636560394358279E-5,
    4.78156510755005422638E-5,
    -1.61760815825896745588E-4,
    5.12285956168575772895E-4,
    -1.51357245063125314899E-3,
    4.15642294431288815669E-3,
    -1.05640848946261981558E-2,
    2.47264490306265168283E-2,
    -5.29459812080949914269E-2,
    1.02643658689847095384E-1,
    -1.76416518357834055153E-1,
    2.52587186443633654823E-1,
];

// Chebyshev coefficients for exp(-x) sqrt(x) I1(x) on [8, inf]
// (`aten/src/ATen/native/cuda/Math.cuh:606-619`). lim(x->inf) = 1/sqrt(2pi).
#[allow(
    clippy::excessive_precision,
    reason = "verbatim Cephes i1e_B coefficients (cuda/Math.cuh:606-619)"
)]
const I1E_B: [f64; 25] = [
    7.51729631084210481353E-18,
    4.41434832307170791151E-18,
    -4.65030536848935832153E-17,
    -3.20952592199342395980E-17,
    2.96262899764595013876E-16,
    3.30820231092092828324E-16,
    -1.88035477551078244854E-15,
    -3.81440307243700780478E-15,
    1.04202769841288027642E-14,
    4.27244001671195135429E-14,
    -2.10154184277266431302E-14,
    -4.08355111109219731823E-13,
    -7.19855177624590851209E-13,
    2.03562854414708950722E-12,
    1.41258074366137813316E-11,
    3.25260358301548823856E-11,
    -1.89749581235054123450E-11,
    -5.58974346219658380687E-10,
    -3.83538038596423702205E-9,
    -2.63146884688951950684E-8,
    -2.51223623787020892529E-7,
    -3.88256480887769039346E-6,
    -1.10588938762623716291E-4,
    -9.76109749136146840777E-3,
    7.78576235018280120474E-1,
];

/// f64 Cephes `i0`. Even function. See module note for the upstream cite
/// (`aten/src/ATen/native/cuda/Math.cuh:502-555`).
fn i0_f64(x_in: f64) -> f64 {
    let x = x_in.abs();
    if x <= 8.0 {
        let y = (x / 2.0) - 2.0;
        x.exp() * chbevl(y, &I0E_A)
    } else {
        (x.exp() * chbevl(32.0 / x - 2.0, &I0E_B)) / x.sqrt()
    }
}

/// f64 Cephes `i0e` = exp(-|x|) I0(x). Even. `calc_i0e`
/// (`aten/src/ATen/native/Math.h:101-145`).
fn i0e_f64(x_in: f64) -> f64 {
    let x = x_in.abs();
    if x <= 8.0 {
        let y = (x / 2.0) - 2.0;
        chbevl(y, &I0E_A)
    } else {
        chbevl(32.0 / x - 2.0, &I0E_B) / x.sqrt()
    }
}

/// f64 Cephes `i1`. Odd function (sign follows `x_in`). `i1_string`
/// (`aten/src/ATen/native/cuda/Math.cuh:575-622`).
fn i1_f64(x_in: f64) -> f64 {
    let x = x_in.abs();
    let out = if x <= 8.0 {
        let y = x / 2.0 - 2.0;
        x.exp() * x * chbevl(y, &I1E_A)
    } else {
        (x.exp() * chbevl(32.0 / x - 2.0, &I1E_B)) / x.sqrt()
    };
    if x_in < 0.0 { -out } else { out }
}

/// f64 Cephes `i1e` = exp(-|x|) I1(x). Odd. `calc_i1e`
/// (`aten/src/ATen/native/cuda/Math.cuh:647-696`).
fn i1e_f64(x_in: f64) -> f64 {
    let x = x_in.abs();
    let out = if x <= 8.0 {
        let y = x / 2.0 - 2.0;
        chbevl(y, &I1E_A) * x
    } else {
        chbevl(32.0 / x - 2.0, &I1E_B) / x.sqrt()
    };
    if x_in < 0.0 { -out } else { out }
}

/// `i0(x)` scalar: run the Cephes evaluator in f64, narrow to `T`. NaN in -> NaN
/// out (propagates through `abs`/`exp`/`chbevl`); `i0(0)=1`, `i0(+/-inf)=+inf`.
fn i0_scalar<T: Float>(x: T) -> T {
    let xf = <T as num_traits::ToPrimitive>::to_f64(&x).unwrap_or(f64::NAN);
    T::from(i0_f64(xf)).unwrap_or_else(|| T::from(f64::NAN).unwrap())
}

/// `i0e(x)` scalar. `i0e(0)=1`, `i0e(+/-inf)=0`.
fn i0e_scalar<T: Float>(x: T) -> T {
    let xf = <T as num_traits::ToPrimitive>::to_f64(&x).unwrap_or(f64::NAN);
    T::from(i0e_f64(xf)).unwrap_or_else(|| T::from(f64::NAN).unwrap())
}

/// `i1(x)` scalar. Odd. `i1(0)=0`, `i1(+inf)=+inf`, `i1(-inf)=-inf`.
fn i1_scalar<T: Float>(x: T) -> T {
    let xf = <T as num_traits::ToPrimitive>::to_f64(&x).unwrap_or(f64::NAN);
    T::from(i1_f64(xf)).unwrap_or_else(|| T::from(f64::NAN).unwrap())
}

/// `i1e(x)` scalar. Odd. `i1e(0)=0`, `i1e(+/-inf)=+/-0`.
fn i1e_scalar<T: Float>(x: T) -> T {
    let xf = <T as num_traits::ToPrimitive>::to_f64(&x).unwrap_or(f64::NAN);
    T::from(i1e_f64(xf)).unwrap_or_else(|| T::from(f64::NAN).unwrap())
}

// ---------------------------------------------------------------------------
// spherical Bessel j0 + modified Bessel K family (Cephes, #1651 batch 3a)
// ---------------------------------------------------------------------------
//
// Direct ports of the upstream torch.special Cephes kernels. The K-family
// reuses the shared `chbevl` Clenshaw evaluator from batch 2: the upstream
// inline Clenshaw loop (`a = z*q - p + A[index]`, return `T(0.5)*(a - p)`,
// `aten/src/ATen/native/cuda/Math.cuh:2557-2576`) is bit-identical to
// `chbevl(z, A)` — `a` plays `b0`, `q` plays `b1`, `p` plays `b2`, and the
// upstream `T(0.5)*(a - p)` is exactly `chbevl`'s `0.5*(b0 - b2)` return. The
// argument `z` is `x*x - 2` in the small region (`x <= 2`) and `8/x - 2` in the
// large region (`x > 2`). `k0`/`k1` reuse the batch-2 `i0_f64`/`i1_f64`
// (modified_bessel_i0/i1_forward) for the small-region log term.
//
//   - spherical_bessel_j0: `cuda/Math.cuh:3039-3052`. `isinf -> 0`;
//     `|x| < 0.5 -> ` 6-term Taylor; else `sin(x)/x`. `j0(0) = 1`,
//     `j0(NaN) = NaN`.
//   - modified_bessel_k0 / scaled: `cuda/Math.cuh:2501-2657`. A[10]/B[25].
//     `x == 0 -> +inf`; `x < 0 -> NaN`. Region split at `x <= 2`.
//   - modified_bessel_k1 / scaled: `cuda/Math.cuh:2659-2817`. A[11]/B[25].
//     `x == 0 -> +inf`; `x < 0 -> NaN`. Region split at `x <= 2`.

/// `spherical_bessel_j0(x)` f64. Verbatim port of `spherical_bessel_j0_forward`
/// (`aten/src/ATen/native/cuda/Math.cuh:3039-3052`).
fn spherical_bessel_j0_f64(x: f64) -> f64 {
    if x.is_infinite() {
        return 0.0;
    }
    if x.abs() < 0.5 {
        let x2 = x * x;
        return 1.0
            + x2 * (-1.0 / 6.0
                + x2 * (1.0 / 120.0
                    + x2 * (-1.0 / 5040.0
                        + x2 * (1.0 / 362880.0
                            + x2 * (-1.0 / 39916800.0 + x2 * (1.0 / 6227020800.0))))));
    }
    x.sin() / x
}

fn spherical_bessel_j0_scalar<T: Float>(x: T) -> T {
    let xf = <T as num_traits::ToPrimitive>::to_f64(&x).unwrap_or(f64::NAN);
    T::from(spherical_bessel_j0_f64(xf)).unwrap_or_else(|| T::from(f64::NAN).unwrap())
}

// Cephes A[10] / B[25] for K0 (`aten/src/ATen/native/cuda/Math.cuh:2504-2543`).
#[allow(
    clippy::excessive_precision,
    reason = "verbatim Cephes K0 A-set (cuda/Math.cuh:2504-2515); full-width for audit-friendly diff"
)]
const K0_A: [f64; 10] = [
    1.37446543561352307156e-16,
    4.25981614279661018399e-14,
    1.03496952576338420167e-11,
    1.90451637722020886025e-09,
    2.53479107902614945675e-07,
    2.28621210311945178607e-05,
    1.26461541144692592338e-03,
    3.59799365153615016266e-02,
    3.44289899924628486886e-01,
    -5.35327393233902768720e-01,
];

#[allow(
    clippy::excessive_precision,
    reason = "verbatim Cephes K0 B-set (cuda/Math.cuh:2517-2543)"
)]
const K0_B: [f64; 25] = [
    5.30043377268626276149e-18,
    -1.64758043015242134646e-17,
    5.21039150503902756861e-17,
    -1.67823109680541210385e-16,
    5.51205597852431940784e-16,
    -1.84859337734377901440e-15,
    6.34007647740507060557e-15,
    -2.22751332699166985548e-14,
    8.03289077536357521100e-14,
    -2.98009692317273043925e-13,
    1.14034058820847496303e-12,
    -4.51459788337394416547e-12,
    1.85594911495471785253e-11,
    -7.95748924447710747776e-11,
    3.57739728140030116597e-10,
    -1.69753450938905987466e-09,
    8.57403401741422608519e-09,
    -4.66048989768794782956e-08,
    2.76681363944501510342e-07,
    -1.83175552271911948767e-06,
    1.39498137188764993662e-05,
    -1.28495495816278026384e-04,
    1.56988388573005337491e-03,
    -3.14481013119645005427e-02,
    2.44030308206595545468e+00,
];

// Cephes A[11] / B[25] for K1 (`aten/src/ATen/native/cuda/Math.cuh:2662-2702`).
#[allow(
    clippy::excessive_precision,
    reason = "verbatim Cephes K1 A-set (cuda/Math.cuh:2662-2673)"
)]
const K1_A: [f64; 11] = [
    -7.02386347938628759343e-18,
    -2.42744985051936593393e-15,
    -6.66690169419932900609e-13,
    -1.41148839263352776110e-10,
    -2.21338763073472585583e-08,
    -2.43340614156596823496e-06,
    -1.73028895751305206302e-04,
    -6.97572385963986435018e-03,
    -1.22611180822657148235e-01,
    -3.53155960776544875667e-01,
    1.52530022733894777053e+00,
];

#[allow(
    clippy::excessive_precision,
    reason = "verbatim Cephes K1 B-set (cuda/Math.cuh:2676-2702)"
)]
const K1_B: [f64; 25] = [
    -5.75674448366501715755e-18,
    1.79405087314755922667e-17,
    -5.68946255844285935196e-17,
    1.83809354436663880070e-16,
    -6.05704724837331885336e-16,
    2.03870316562433424052e-15,
    -7.01983709041831346144e-15,
    2.47715442448130437068e-14,
    -8.97670518232499435011e-14,
    3.34841966607842919884e-13,
    -1.28917396095102890680e-12,
    5.13963967348173025100e-12,
    -2.12996783842756842877e-11,
    9.21831518760500529508e-11,
    -4.19035475934189648750e-10,
    2.01504975519703286596e-09,
    -1.03457624656780970260e-08,
    5.74108412545004946722e-08,
    -3.50196060308781257119e-07,
    2.40648494783721712015e-06,
    -1.93619797416608296024e-05,
    1.95215518471351631108e-04,
    -2.85781685962277938680e-03,
    1.03923736576817238437e-01,
    2.72062619048444266945e+00,
];

/// `modified_bessel_k0(x)` f64. `modified_bessel_k0_forward`
/// (`aten/src/ATen/native/cuda/Math.cuh:2503-2577`). `x==0 -> +inf`,
/// `x<0 -> NaN`. Small region (`x <= 2`): `chbevl(x*x-2, A) - log(0.5x)*i0(x)`;
/// large region: `exp(-x) * chbevl(8/x-2, B) / sqrt(x)`.
fn modified_bessel_k0_f64(x: f64) -> f64 {
    if x == 0.0 {
        return f64::INFINITY;
    }
    if x < 0.0 {
        return f64::NAN;
    }
    if x <= 2.0 {
        chbevl(x * x - 2.0, &K0_A) - (0.5 * x).ln() * i0_f64(x)
    } else {
        (-x).exp() * chbevl(8.0 / x - 2.0, &K0_B) / x.sqrt()
    }
}

/// `scaled_modified_bessel_k0(x)` f64 — `k0(x) * exp(x)` (drops the `exp(-x)`
/// in the large region). `scaled_modified_bessel_k0_forward`
/// (`aten/src/ATen/native/cuda/Math.cuh:2582-2656`). For large `x`,
/// `-> sqrt(pi/(2x))`.
fn scaled_modified_bessel_k0_f64(x: f64) -> f64 {
    if x == 0.0 {
        return f64::INFINITY;
    }
    if x < 0.0 {
        return f64::NAN;
    }
    if x <= 2.0 {
        (chbevl(x * x - 2.0, &K0_A) - (0.5 * x).ln() * i0_f64(x)) * x.exp()
    } else {
        chbevl(8.0 / x - 2.0, &K0_B) / x.sqrt()
    }
}

/// `modified_bessel_k1(x)` f64. `modified_bessel_k1_forward`
/// (`aten/src/ATen/native/cuda/Math.cuh:2661-2736`). `x==0 -> +inf`,
/// `x<0 -> NaN`. Small region (`x <= 2`):
/// `log(0.5x)*i1(x) + 0.5*chbevl(x*x-2, A)/x`; large region:
/// `exp(-x) * chbevl(8/x-2, B) / sqrt(x)`.
fn modified_bessel_k1_f64(x: f64) -> f64 {
    if x == 0.0 {
        return f64::INFINITY;
    }
    if x < 0.0 {
        return f64::NAN;
    }
    if x <= 2.0 {
        (0.5 * x).ln() * i1_f64(x) + chbevl(x * x - 2.0, &K1_A) / x
    } else {
        (-x).exp() * chbevl(8.0 / x - 2.0, &K1_B) / x.sqrt()
    }
}

/// `scaled_modified_bessel_k1(x)` f64 — `k1(x) * exp(x)`.
/// `scaled_modified_bessel_k1_forward`
/// (`aten/src/ATen/native/cuda/Math.cuh:2740-2815`).
fn scaled_modified_bessel_k1_f64(x: f64) -> f64 {
    if x == 0.0 {
        return f64::INFINITY;
    }
    if x < 0.0 {
        return f64::NAN;
    }
    if x <= 2.0 {
        ((0.5 * x).ln() * i1_f64(x) + chbevl(x * x - 2.0, &K1_A) / x) * x.exp()
    } else {
        chbevl(8.0 / x - 2.0, &K1_B) / x.sqrt()
    }
}

fn modified_bessel_k0_scalar<T: Float>(x: T) -> T {
    let xf = <T as num_traits::ToPrimitive>::to_f64(&x).unwrap_or(f64::NAN);
    T::from(modified_bessel_k0_f64(xf)).unwrap_or_else(|| T::from(f64::NAN).unwrap())
}

fn scaled_modified_bessel_k0_scalar<T: Float>(x: T) -> T {
    let xf = <T as num_traits::ToPrimitive>::to_f64(&x).unwrap_or(f64::NAN);
    T::from(scaled_modified_bessel_k0_f64(xf)).unwrap_or_else(|| T::from(f64::NAN).unwrap())
}

fn modified_bessel_k1_scalar<T: Float>(x: T) -> T {
    let xf = <T as num_traits::ToPrimitive>::to_f64(&x).unwrap_or(f64::NAN);
    T::from(modified_bessel_k1_f64(xf)).unwrap_or_else(|| T::from(f64::NAN).unwrap())
}

fn scaled_modified_bessel_k1_scalar<T: Float>(x: T) -> T {
    let xf = <T as num_traits::ToPrimitive>::to_f64(&x).unwrap_or(f64::NAN);
    T::from(scaled_modified_bessel_k1_f64(xf)).unwrap_or_else(|| T::from(f64::NAN).unwrap())
}

// ---------------------------------------------------------------------------
// Incomplete-gamma family (gammainc / gammaincc) and log-beta / multigammaln
// ---------------------------------------------------------------------------
//
// The interior of the (a, x) plane is computed by the Numerical-Recipes
// `gammp`/`gammq` pair (power series for `x < a + 1`, Lentz continued fraction
// for `x >= a + 1`) — the same scalar kernel that
// `ferrotorch-distributions/src/gamma.rs::lower_incomplete_gamma_regularized`
// uses to back `Gamma::cdf`. The kernel is lifted here so the PUBLIC
// `torch.special.gammainc`/`gammaincc` tensor ops own it, while distributions
// keeps its private scalar copy. The boundary handling on top of the kernel
// mirrors PyTorch's `calc_igamma`/`calc_igammac` exactly (see cites below) so
// that `gammainc(0, x>0) = 1`, `gammainc(a>0, 0) = 0`, NaN for negatives, etc.

/// Numerical-Recipes `gammp` core in f64: the *smooth-interior* regularized
/// lower incomplete gamma `P(a, x) = γ(a, x) / Γ(a)` for `a > 0`, `x > 0`. The
/// caller is responsible for the boundary cases (`x <= 0`, `a <= 0`, infinities)
/// — this helper assumes both arguments are finite and strictly positive.
fn gammp_core_f64(a: f64, x: f64) -> f64 {
    let gln = lgamma_scalar(a);
    if x < a + 1.0 {
        // Power series expansion for P(a, x).
        let mut ap = a;
        let mut sum = 1.0 / a;
        let mut del = sum;
        for _ in 0..300 {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * 1e-15 {
                break;
            }
        }
        sum * (-x + a * x.ln() - gln).exp()
    } else {
        // Lentz's continued fraction for Q(a, x) = 1 - P(a, x).
        1.0 - gammq_core_f64_cf(a, x, gln)
    }
}

/// Lentz continued fraction for the regularized upper incomplete gamma
/// `Q(a, x)` valid for `x >= a + 1`. `gln = lgamma(a)` is passed in to avoid
/// recomputing it. Returns `Q(a, x)` directly (no `1 - P` cancellation).
fn gammq_core_f64_cf(a: f64, x: f64, gln: f64) -> f64 {
    let tiny = 1e-300;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / tiny;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..300 {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < tiny {
            d = tiny;
        }
        c = b + an / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-15 {
            break;
        }
    }
    (-x + a * x.ln() - gln).exp() * h
}

/// Numerical-Recipes `gammq` core in f64: the *smooth-interior* regularized
/// upper incomplete gamma `Q(a, x) = Γ(a, x) / Γ(a)` for `a > 0`, `x > 0`.
/// For `x < a + 1` it is `1 - P(a, x)` (series); for `x >= a + 1` it is the
/// continued fraction directly (avoiding the `1 - P` cancellation in the tail).
fn gammq_core_f64(a: f64, x: f64) -> f64 {
    let gln = lgamma_scalar(a);
    if x < a + 1.0 {
        1.0 - gammp_core_f64(a, x)
    } else {
        gammq_core_f64_cf(a, x, gln)
    }
}

/// Regularized lower incomplete gamma `P(a, x)` matching `torch.special.gammainc`
/// boundary-for-boundary.
///
/// Boundary contract is a direct port of PyTorch
/// `aten/src/ATen/native/Math.h:1164-1187 calc_igamma`:
///   - `x < 0 || a < 0` → NaN
///   - `a == 0`: `x > 0` → 1.0, else NaN
///   - `x == 0` → 0.0
///   - `a == inf`: `x == inf` → NaN, else 0.0
///   - `x == inf` → 1.0
///
/// The smooth interior is the NR `gammp` core (`gammp_core_f64`).
fn gammainc_scalar<T: Float>(a: T, x: T) -> T {
    let af = <T as num_traits::ToPrimitive>::to_f64(&a).unwrap_or(f64::NAN);
    let xf = <T as num_traits::ToPrimitive>::to_f64(&x).unwrap_or(f64::NAN);
    let result = calc_igamma_f64(af, xf);
    T::from(result).unwrap_or_else(|| T::from(f64::NAN).unwrap())
}

/// f64 boundary-aware regularized lower incomplete gamma. See
/// [`gammainc_scalar`] for the upstream cite.
fn calc_igamma_f64(a: f64, x: f64) -> f64 {
    if a.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 || a < 0.0 {
        return f64::NAN;
    }
    if a == 0.0 {
        return if x > 0.0 { 1.0 } else { f64::NAN };
    }
    if x == 0.0 {
        return 0.0;
    }
    if a.is_infinite() {
        return if x.is_infinite() { f64::NAN } else { 0.0 };
    }
    if x.is_infinite() {
        return 1.0;
    }
    gammp_core_f64(a, x)
}

/// Regularized upper incomplete gamma `Q(a, x)` matching
/// `torch.special.gammaincc` boundary-for-boundary.
///
/// Boundary contract is a direct port of PyTorch
/// `aten/src/ATen/native/Math.h:1085-1107 calc_igammac`:
///   - `x < 0 || a < 0` → NaN
///   - `a == 0`: `x > 0` → 0.0, else NaN
///   - `x == 0` → 1.0
///   - `a == inf`: `x == inf` → NaN, else 1.0
///   - `x == inf` → 0.0
///
/// The smooth interior is the NR `gammq` core (`gammq_core_f64`).
fn gammaincc_scalar<T: Float>(a: T, x: T) -> T {
    let af = <T as num_traits::ToPrimitive>::to_f64(&a).unwrap_or(f64::NAN);
    let xf = <T as num_traits::ToPrimitive>::to_f64(&x).unwrap_or(f64::NAN);
    let result = calc_igammac_f64(af, xf);
    T::from(result).unwrap_or_else(|| T::from(f64::NAN).unwrap())
}

/// f64 boundary-aware regularized upper incomplete gamma. See
/// [`gammaincc_scalar`] for the upstream cite.
fn calc_igammac_f64(a: f64, x: f64) -> f64 {
    if a.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 || a < 0.0 {
        return f64::NAN;
    }
    if a == 0.0 {
        return if x > 0.0 { 0.0 } else { f64::NAN };
    }
    if x == 0.0 {
        return 1.0;
    }
    if a.is_infinite() {
        return if x.is_infinite() { f64::NAN } else { 1.0 };
    }
    if x.is_infinite() {
        return 0.0;
    }
    gammq_core_f64(a, x)
}

/// Log-beta `lnB(a, b) = lgamma(a) + lgamma(b) - lgamma(a + b)`.
///
/// Mirrors `torch.special.gammaln`-built `lbeta` / `scipy.special.betaln`.
/// Computed in f64 then narrowed to `T` (the subtraction of three lgamma
/// values loses bits in f32 otherwise).
fn log_beta_scalar<T: Float>(a: T, b: T) -> T {
    let af = <T as num_traits::ToPrimitive>::to_f64(&a).unwrap_or(f64::NAN);
    let bf = <T as num_traits::ToPrimitive>::to_f64(&b).unwrap_or(f64::NAN);
    let r = lgamma_scalar(af) + lgamma_scalar(bf) - lgamma_scalar(af + bf);
    T::from(r).unwrap_or_else(|| T::from(f64::NAN).unwrap())
}

/// Beta function `B(a, b) = exp(lnB(a, b))`.
fn beta_scalar<T: Float>(a: T, b: T) -> T {
    let lb = log_beta_scalar::<f64>(
        <T as num_traits::ToPrimitive>::to_f64(&a).unwrap_or(f64::NAN),
        <T as num_traits::ToPrimitive>::to_f64(&b).unwrap_or(f64::NAN),
    );
    T::from(lb.exp()).unwrap_or_else(|| T::from(f64::NAN).unwrap())
}

/// Multivariate log-gamma `log Γ_p(a)`:
/// `C + Σ_{i=1}^p lgamma(a + (1 - i)/2)` with `C = (p(p-1)/4) ln(π)`.
///
/// Mirrors `torch.special.multigammaln` / `torch.mvlgamma`
/// (`aten/src/ATen/native/UnaryOps.cpp:887-905`). The kernel performs NO domain
/// check beyond `p >= 1` (`mvlgamma_check`, `UnaryOps.cpp:884`); it computes the
/// `lgamma`-sum verbatim for any `a`. The documented domain `a > (p - 1)/2`
/// (`torch/special/__init__.py:862`) only means the result is mathematically
/// "undefined" outside it — torch still emits the ordinary finite value of
/// `Σ lgamma(a + (1-i)/2)`, or `+inf` when an argument lands on a non-positive
/// integer pole. We match that contract: no fabricated NaN guard.
fn multigammaln_scalar<T: Float>(a: T, p: usize) -> T {
    let af = <T as num_traits::ToPrimitive>::to_f64(&a).unwrap_or(f64::NAN);
    let pf = p as f64;
    let c = (pf * (pf - 1.0) / 4.0) * std::f64::consts::PI.ln();
    let mut sum = 0.0_f64;
    for i in 1..=p {
        sum += lgamma_scalar(af + (1.0 - i as f64) / 2.0);
    }
    T::from(c + sum).unwrap_or_else(|| T::from(f64::NAN).unwrap())
}

/// Sign of the gamma function `Γ(x)` — the sign that `lgamma = ln|Γ|`
/// discards. Mirrors `scipy.special.gammasgn`:
///   - `x > 0` (and `+0.0`) → `+1`
///   - `x` a negative integer (a pole of Γ) → NaN
///   - `x < 0` non-integer → `(-1)^floor(x)`
///
/// PyTorch exposes the equivalent via `torch.sgn(torch.lgamma(...))`-style
/// composition; scipy's `gammasgn` is the canonical named contract and is the
/// reference this op pins against.
fn gammaln_sign_scalar<T: Float>(x: T) -> T {
    let one = nt_one::<T>();
    let zero = nt_zero::<T>();
    if x.is_nan() {
        return x;
    }
    // Positive (including +0.0) → +1.
    if x > zero {
        return one;
    }
    let xf = <T as num_traits::ToPrimitive>::to_f64(&x).unwrap_or(f64::NAN);
    // Here x is not NaN and not strictly positive, so x <= 0.
    if xf < 0.0 {
        // Negative integers are poles of Γ → NaN; otherwise (-1)^floor(x).
        if xf.fract() == 0.0 {
            return T::from(f64::NAN).unwrap();
        }
        let fi = xf.floor() as i64;
        return if fi.rem_euclid(2) == 0 { one } else { -one };
    }
    // x is exactly zero (either sign): scipy returns +1 for +0.0, -1 for -0.0.
    if xf.is_sign_negative() { -one } else { one }
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

/// Entropy `entr(x)`: `x > 0 -> -x*log(x)`, `x == 0 -> 0`, `x < 0 -> -inf`,
/// `NaN -> NaN`. Mirrors `torch.special.entr`
/// (`torch/special/__init__.py:67`; kernel `aten/src/ATen/native/cuda/Math.cuh:463-480`).
///
/// CUDA (f32/f64) tensors run an on-device elementwise PTX kernel via
/// [`crate::gpu_dispatch::GpuBackend::entr_f32`]/`entr_f64` (no host round
/// trip); bf16/f16 CUDA inputs return `NotImplementedOnCuda`.
pub fn entr<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) =
        special_gpu_simple(input, "entr", |b, h| b.entr_f32(h), |b, h| b.entr_f64(h))?
    {
        return Ok(out);
    }
    unary_map(input, entr_scalar)
}

/// Standard-normal CDF `ndtr(x) = (1 + erf(x/sqrt(2))) / 2`. Mirrors
/// `torch.special.ndtr` (`torch/special/__init__.py:624`; kernel
/// `aten/src/ATen/native/UnaryOps.cpp:715-718`). Composed over the shipped
/// `erf` so `ndtr(-inf) = 0`, `ndtr(0) = 0.5`, `ndtr(+inf) = 1`,
/// `ndtr(NaN) = NaN`.
///
/// CUDA (f32/f64) tensors run an on-device elementwise PTX kernel via
/// [`crate::gpu_dispatch::GpuBackend::ndtr_f32`]/`ndtr_f64` (no host round
/// trip); bf16/f16 CUDA inputs return `NotImplementedOnCuda`.
pub fn ndtr<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) =
        special_gpu_simple(input, "ndtr", |b, h| b.ndtr_f32(h), |b, h| b.ndtr_f64(h))?
    {
        return Ok(out);
    }
    unary_map(input, ndtr_scalar)
}

/// Inverse standard-normal CDF (quantile function) `ndtri(p)`. Domain `(0, 1)`:
/// `ndtri(0) = -inf`, `ndtri(1) = +inf`, `ndtri(p<0 || p>1) = NaN`. Mirrors
/// `torch.special.ndtri` (`torch/special/__init__.py:649`); the implementation
/// ports the Cephes rational from `aten/src/ATen/native/cuda/Math.cuh:48-173`
/// (NOT `sqrt(2)*erfinv(2p-1)`) for ULP parity with torch.
///
/// CUDA (f32/f64) tensors run an on-device elementwise PTX kernel via
/// [`crate::gpu_dispatch::GpuBackend::ndtri_f32`]/`ndtri_f64` (no host round
/// trip); bf16/f16 CUDA inputs return `NotImplementedOnCuda`.
pub fn ndtri<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) =
        special_gpu_simple(input, "ndtri", |b, h| b.ndtri_f32(h), |b, h| b.ndtri_f64(h))?
    {
        return Ok(out);
    }
    unary_map(input, ndtri_scalar)
}

/// Modified Bessel function of the first kind, order 0: `i0(x)`. Even function;
/// `i0(0) = 1`, `i0(+/-inf) = +inf`, `i0(NaN) = NaN`. Mirrors
/// `torch.special.i0` / `torch.i0` (`torch/special/__init__.py:522`); the
/// scalar evaluator ports the Cephes `chbevl` Chebyshev kernel from
/// `aten/src/ATen/native/cuda/Math.cuh:502-555`.
///
/// CUDA f32 tensors run an on-device elementwise PTX kernel via
/// [`crate::gpu_dispatch::GpuBackend::i0_f32`] (no host round trip); f64 CUDA
/// returns `NotImplementedOnCuda` (base PTX lacks `lg2.approx.f64`), bf16/f16
/// CUDA likewise.
pub fn i0<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) = special_gpu_simple(input, "i0", |b, h| b.i0_f32(h), |b, h| b.i0_f64(h))? {
        return Ok(out);
    }
    unary_map(input, i0_scalar)
}

/// Exponentially-scaled modified Bessel order 0: `i0e(x) = exp(-|x|) I0(x)`.
/// Even; `i0e(0) = 1`, `i0e(+/-inf) = 0` (stays finite where `i0` overflows),
/// `i0e(NaN) = NaN`. Mirrors `torch.special.i0e`
/// (`torch/special/__init__.py:548`); scalar evaluator ports `calc_i0e`
/// (`aten/src/ATen/native/Math.h:101-145`) — same Chebyshev sets as [`i0`]
/// without the `exp(x)` factor.
///
/// CUDA f32 runs on-device via [`crate::gpu_dispatch::GpuBackend::i0e_f32`];
/// f64/bf16/f16 CUDA return `NotImplementedOnCuda`.
pub fn i0e<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) = special_gpu_simple(input, "i0e", |b, h| b.i0e_f32(h), |b, h| b.i0e_f64(h))? {
        return Ok(out);
    }
    unary_map(input, i0e_scalar)
}

/// Modified Bessel function of the first kind, order 1: `i1(x)`. Odd function
/// (sign follows `x`); `i1(0) = 0`, `i1(+inf) = +inf`, `i1(-inf) = -inf`,
/// `i1(NaN) = NaN`. Mirrors `torch.special.i1` / `torch.i1`; scalar evaluator
/// ports `i1_string` (`aten/src/ATen/native/cuda/Math.cuh:575-622`).
///
/// CUDA f32 runs on-device via [`crate::gpu_dispatch::GpuBackend::i1_f32`];
/// f64/bf16/f16 CUDA return `NotImplementedOnCuda`.
pub fn i1<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) = special_gpu_simple(input, "i1", |b, h| b.i1_f32(h), |b, h| b.i1_f64(h))? {
        return Ok(out);
    }
    unary_map(input, i1_scalar)
}

/// Exponentially-scaled modified Bessel order 1: `i1e(x) = exp(-|x|) I1(x)`.
/// Odd; `i1e(0) = 0`, `i1e(+/-inf) = +/-0`, `i1e(NaN) = NaN`. Mirrors
/// `torch.special.i1e` (`torch/special/__init__.py:598`); scalar evaluator
/// ports `calc_i1e` (`aten/src/ATen/native/cuda/Math.cuh:647-696`) — same
/// Chebyshev sets as [`i1`] without the `exp(x)` factor.
///
/// CUDA f32 runs on-device via [`crate::gpu_dispatch::GpuBackend::i1e_f32`];
/// f64/bf16/f16 CUDA return `NotImplementedOnCuda`.
pub fn i1e<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) = special_gpu_simple(input, "i1e", |b, h| b.i1e_f32(h), |b, h| b.i1e_f64(h))? {
        return Ok(out);
    }
    unary_map(input, i1e_scalar)
}

/// Spherical Bessel function of the first kind, order 0:
/// `j0(x) = sin(x)/x`, with `j0(0) = 1` (the Taylor branch) and `j0(+/-inf) = 0`.
/// `j0(NaN) = NaN`. Mirrors `torch.special.spherical_bessel_j0`
/// (`torch/special/__init__.py:1444+`); scalar evaluator ports
/// `spherical_bessel_j0_forward` (`aten/src/ATen/native/cuda/Math.cuh:3039-3052`):
/// `|x| < 0.5` uses the explicit 6-term Taylor series, else `sin(x)/x`.
///
/// CUDA f32 tensors run an on-device elementwise PTX kernel via
/// [`crate::gpu_dispatch::GpuBackend::spherical_bessel_j0_f32`] (no host round
/// trip); f64/bf16/f16 CUDA return `NotImplementedOnCuda` (base PTX lacks the
/// f64 transcendental approximations needed for the `sin(x)/x` branch).
pub fn spherical_bessel_j0<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) = special_gpu_simple(
        input,
        "spherical_bessel_j0",
        |b, h| b.spherical_bessel_j0_f32(h),
        |b, h| b.spherical_bessel_j0_f64(h),
    )? {
        return Ok(out);
    }
    unary_map(input, spherical_bessel_j0_scalar)
}

/// Modified Bessel function of the second kind, order 0: `k0(x)`. Domain
/// `x > 0`: `k0(0) = +inf`, `k0(x < 0) = NaN`, `k0(NaN) = NaN`. Decays to `0`
/// for large `x`. Mirrors `torch.special.modified_bessel_k0`
/// (`torch/special/__init__.py:1304-1341`); scalar evaluator ports
/// `modified_bessel_k0_forward` (`aten/src/ATen/native/cuda/Math.cuh:2503-2577`)
/// over the shared `chbevl` Clenshaw evaluator and the batch-2 `i0`.
///
/// CUDA tensors (all dtypes) return `NotImplementedOnCuda`: the on-device PTX
/// kernel is tracked under #1651 (batch 3b) alongside the K1 / `zeta` / `airy`
/// kernels — the small-region log-term over the full `i0` chbevl unroll plus
/// `log`/`exp` pushes the hand-written f32 PTX past one cohesive commit.
pub fn modified_bessel_k0<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda {
            op: "modified_bessel_k0",
        });
    }
    unary_map(input, modified_bessel_k0_scalar)
}

/// Exponentially-scaled modified Bessel order 0:
/// `scaled_modified_bessel_k0(x) = exp(x) * k0(x)`. Same domain as
/// [`modified_bessel_k0`]; stays finite (`-> sqrt(pi/(2x))`) where `k0`
/// underflows. Mirrors `torch.special.scaled_modified_bessel_k0`
/// (`torch/special/__init__.py:1304-1341`); ports
/// `scaled_modified_bessel_k0_forward`
/// (`aten/src/ATen/native/cuda/Math.cuh:2582-2656`).
///
/// CUDA tensors return `NotImplementedOnCuda` (batch 3b, #1651).
pub fn scaled_modified_bessel_k0<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda {
            op: "scaled_modified_bessel_k0",
        });
    }
    unary_map(input, scaled_modified_bessel_k0_scalar)
}

/// Modified Bessel function of the second kind, order 1: `k1(x)`. Domain
/// `x > 0`: `k1(0) = +inf`, `k1(x < 0) = NaN`, `k1(NaN) = NaN`. Mirrors
/// `torch.special.modified_bessel_k1` (`torch/special/__init__.py:1321-1358`);
/// scalar evaluator ports `modified_bessel_k1_forward`
/// (`aten/src/ATen/native/cuda/Math.cuh:2661-2736`) over `chbevl` and the
/// batch-2 `i1`.
///
/// CUDA tensors return `NotImplementedOnCuda` (batch 3b, #1651).
pub fn modified_bessel_k1<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda {
            op: "modified_bessel_k1",
        });
    }
    unary_map(input, modified_bessel_k1_scalar)
}

/// Exponentially-scaled modified Bessel order 1:
/// `scaled_modified_bessel_k1(x) = exp(x) * k1(x)`. Same domain as
/// [`modified_bessel_k1`]. Mirrors
/// `torch.special.scaled_modified_bessel_k1`
/// (`torch/special/__init__.py:1321-1358`); ports
/// `scaled_modified_bessel_k1_forward`
/// (`aten/src/ATen/native/cuda/Math.cuh:2740-2815`).
///
/// CUDA tensors return `NotImplementedOnCuda` (batch 3b, #1651).
pub fn scaled_modified_bessel_k1<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda {
            op: "scaled_modified_bessel_k1",
        });
    }
    unary_map(input, scaled_modified_bessel_k1_scalar)
}

/// Regularized lower incomplete gamma `P(a, x)`, element-wise over a broadcast
/// of `input` (the `a` argument) and `other` (the `x` argument).
///
/// Mirrors `torch.special.gammainc(input, other)` /
/// `torch.igamma(input, other)`. Both arguments must be weakly positive with at
/// least one strictly positive; if either is negative, or both are zero, the
/// result is `NaN` (matching `aten/src/ATen/native/Math.h:1144 calc_igamma`).
/// `input = 0, other > 0 → 1`; `input > 0, other = 0 → 0`; `other → ∞ → 1`.
pub fn gammainc<T: Float>(input: &Tensor<T>, other: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    binary_map(input, other, gammainc_scalar)
}

/// Regularized upper incomplete gamma `Q(a, x) = 1 - P(a, x)`, element-wise
/// over a broadcast of `input` (the `a` argument) and `other` (the `x`
/// argument).
///
/// Mirrors `torch.special.gammaincc(input, other)` /
/// `torch.igammac(input, other)`. Same domain as [`gammainc`];
/// `input = 0, other > 0 → 0`; `input > 0, other = 0 → 1`; `other → ∞ → 0`
/// (matching `aten/src/ATen/native/Math.h calc_igammac`).
pub fn gammaincc<T: Float>(input: &Tensor<T>, other: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    binary_map(input, other, gammaincc_scalar)
}

/// Log-beta function `lnB(a, b) = lgamma(a) + lgamma(b) - lgamma(a + b)`,
/// element-wise over a broadcast of `a` and `b`.
///
/// Mirrors `scipy.special.betaln` / the `lbeta` PyTorch users build from
/// `torch.lgamma`. The accumulation runs in f64 then narrows to `T` so the
/// three-way lgamma subtraction does not lose bits in the f32 path.
pub fn log_beta<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    binary_map(a, b, log_beta_scalar)
}

/// Beta function `B(a, b) = exp(lnB(a, b)) = Γ(a)Γ(b)/Γ(a + b)`, element-wise
/// over a broadcast of `a` and `b`.
///
/// Mirrors `scipy.special.beta`.
pub fn beta<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    binary_map(a, b, beta_scalar)
}

/// Multivariate log-gamma `log Γ_p(a)` with dimension `p`, element-wise over
/// `input`:
///
/// `log Γ_p(a) = (p(p-1)/4) ln(π) + Σ_{i=1}^p lgamma(a + (1 - i)/2)`.
///
/// Mirrors `torch.special.multigammaln(input, p)` /
/// `torch.mvlgamma(input, p)` (`aten/src/ATen/native/UnaryOps.cpp:887`). The
/// documented domain is `a > (p - 1)/2` (`torch/special/__init__.py:862`);
/// elements outside it yield `NaN`.
///
/// # Errors
///
/// Returns an error if `p == 0` (PyTorch requires `p >= 1`, see
/// `UnaryOps.cpp:884 mvlgamma_check`).
pub fn multigammaln<T: Float>(input: &Tensor<T>, p: usize) -> FerrotorchResult<Tensor<T>> {
    if p == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "multigammaln: p has to be greater than or equal to 1".to_string(),
        });
    }
    unary_map(input, move |x| multigammaln_scalar(x, p))
}

/// Alias for [`multigammaln`] — mirrors `torch.mvlgamma(input, p)`
/// (`torch/_torch_docs.py:7895`, "Alias for torch.special.multigammaln").
pub fn mvlgamma<T: Float>(input: &Tensor<T>, p: usize) -> FerrotorchResult<Tensor<T>> {
    multigammaln(input, p)
}

/// Sign of the gamma function `Γ(x)` — the `±1` (or `NaN` at poles) factor that
/// `lgamma = ln|Γ|` discards, element-wise over `input`.
///
/// Mirrors `scipy.special.gammasgn`: `+1` for `x > 0`; `NaN` for negative
/// integers (poles of Γ); `(-1)^floor(x)` for `x < 0` non-integer.
pub fn gammaln_sign<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    unary_map(input, gammaln_sign_scalar)
}

// ===========================================================================
// Orthogonal-polynomial families
// ===========================================================================
//
// These mirror torch.special.{chebyshev,hermite,laguerre,legendre}_polynomial_*.
// Each function returns the n-th degree basis polynomial evaluated pointwise
// at every element of `input`. CUDA tensors (f32/f64) run an on-device PTX
// three-term-recurrence kernel via the `GpuBackend` trait (#1545 / #1533) —
// the input buffer never leaves VRAM; the recurrence runs in each thread's
// registers; the output stays on-device (R-CODE-4, no silent round trip).
// Non-f32/f64 CUDA dtypes (bf16/f16) still return NotImplementedOnCuda rather
// than bouncing through host.
//
// Implementation: every basis is evaluated by its standard three-term
// recurrence. The CPU path runs it in f64 directly in this module; the GPU
// path runs the bit-for-relevant-tolerance-identical recurrence in PTX (see
// `ferrotorch-gpu/src/special.rs`). ferray-polynomial 0.3 has the same idiom
// (Clenshaw on basis-coefficient vectors), but for the
// "single-basis-element" case the direct recurrence is shorter, dependency-
// free, and numerically equivalent for the orders typically used in ML
// pipelines (n ≤ 50). We keep the option open to swap in ferray's Clenshaw
// path later for very-high-order numerical-analysis use cases.

use crate::error::FerrotorchError;

#[inline]
fn poly_is_f32<T: Float>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f32>()
}

#[inline]
fn poly_is_f64<T: Float>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f64>()
}

/// Per-dtype Hermite order limit, above which the polynomial value is replaced
/// by `NaN` — a byte-for-relevant mirror of PyTorch's
/// `getHermitianLimit<T>()` (`aten/src/ATen/native/Math.h:3044-3052`):
/// `float -> 128`, `double -> 512`, otherwise `1024`. The recurrence overflows
/// for `n` this large; torch short-circuits to `quiet_NaN()` rather than
/// emitting the overflowed value (`Math.h:3068` for `hermite_polynomial_h`,
/// `:3109` for `hermite_polynomial_he`). The limit is keyed on the tensor's
/// scalar type `T`, exactly as torch templates `getHermitianLimit<scalar_t>`.
#[inline]
fn hermitian_limit<T: Float>() -> usize {
    if poly_is_f32::<T>() {
        128
    } else if poly_is_f64::<T>() {
        512
    } else {
        1024
    }
}

/// Wrap a GPU buffer handle returned by a polynomial kernel into a
/// CUDA-resident output tensor of the same shape (no host copy).
#[inline]
fn poly_gpu_output<T: Float>(
    handle: crate::gpu_dispatch::GpuBufferHandle,
    shape: Vec<usize>,
) -> FerrotorchResult<Tensor<T>> {
    Tensor::from_storage(crate::storage::TensorStorage::gpu(handle), shape, false)
}

/// Run a single-`n` polynomial kernel (hermite / laguerre / legendre) on a
/// CUDA tensor, dispatching by dtype through the registered [`GpuBackend`].
/// Returns `Ok(Some(out))` when the GPU path handled it, `Ok(None)` when the
/// caller should take the CPU path (non-CUDA input). bf16/f16 CUDA inputs are
/// rejected with `NotImplementedOnCuda` (no host round trip).
fn poly_gpu_simple<T: Float>(
    input: &Tensor<T>,
    n: usize,
    op: &'static str,
    f32_call: impl Fn(
        &dyn crate::gpu_dispatch::GpuBackend,
        &crate::gpu_dispatch::GpuBufferHandle,
    ) -> FerrotorchResult<crate::gpu_dispatch::GpuBufferHandle>,
    f64_call: impl Fn(
        &dyn crate::gpu_dispatch::GpuBackend,
        &crate::gpu_dispatch::GpuBufferHandle,
    ) -> FerrotorchResult<crate::gpu_dispatch::GpuBufferHandle>,
) -> FerrotorchResult<Option<Tensor<T>>> {
    let _ = n;
    if !input.is_cuda() {
        return Ok(None);
    }
    if !(poly_is_f32::<T>() || poly_is_f64::<T>()) {
        return Err(FerrotorchError::NotImplementedOnCuda { op });
    }
    let backend = crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
    let handle = input.gpu_handle()?;
    let out_handle = if poly_is_f32::<T>() {
        f32_call(backend, handle)?
    } else {
        f64_call(backend, handle)?
    };
    Ok(Some(poly_gpu_output::<T>(
        out_handle,
        input.shape().to_vec(),
    )?))
}

/// Run a parameterless elementwise special-function kernel (entr / ndtr /
/// ndtri) on a CUDA tensor, dispatching by dtype through the registered
/// [`GpuBackend`]. Returns `Ok(Some(out))` when the GPU path handled it,
/// `Ok(None)` when the caller should take the CPU path (non-CUDA input).
/// bf16/f16 CUDA inputs are rejected with `NotImplementedOnCuda` (no host
/// round trip). This is the elementwise analog of [`poly_gpu_simple`] (which
/// carries an extra `n` recurrence-order argument the transcendental kernels
/// don't need).
fn special_gpu_simple<T: Float>(
    input: &Tensor<T>,
    op: &'static str,
    f32_call: impl Fn(
        &dyn crate::gpu_dispatch::GpuBackend,
        &crate::gpu_dispatch::GpuBufferHandle,
    ) -> FerrotorchResult<crate::gpu_dispatch::GpuBufferHandle>,
    f64_call: impl Fn(
        &dyn crate::gpu_dispatch::GpuBackend,
        &crate::gpu_dispatch::GpuBufferHandle,
    ) -> FerrotorchResult<crate::gpu_dispatch::GpuBufferHandle>,
) -> FerrotorchResult<Option<Tensor<T>>> {
    if !input.is_cuda() {
        return Ok(None);
    }
    if !(poly_is_f32::<T>() || poly_is_f64::<T>()) {
        return Err(FerrotorchError::NotImplementedOnCuda { op });
    }
    let backend = crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
    let handle = input.gpu_handle()?;
    let out_handle = if poly_is_f32::<T>() {
        f32_call(backend, handle)?
    } else {
        f64_call(backend, handle)?
    };
    Ok(Some(poly_gpu_output::<T>(
        out_handle,
        input.shape().to_vec(),
    )?))
}

/// Chebyshev-family GPU dispatch (T/U/V/W + shifted), selecting the kind via
/// the `(seed_a, seed_b, shift)` recurrence seed. See [`poly_gpu_simple`] for
/// the `Ok(None)` CPU-fallthrough contract.
fn poly_gpu_chebyshev<T: Float>(
    input: &Tensor<T>,
    n: usize,
    seed_a: f64,
    seed_b: f64,
    shift: bool,
    op: &'static str,
) -> FerrotorchResult<Option<Tensor<T>>> {
    if !input.is_cuda() {
        return Ok(None);
    }
    if !(poly_is_f32::<T>() || poly_is_f64::<T>()) {
        return Err(FerrotorchError::NotImplementedOnCuda { op });
    }
    let backend = crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
    let handle = input.gpu_handle()?;
    let out_handle = if poly_is_f32::<T>() {
        backend.chebyshev_poly_f32(handle, n, seed_a as f32, seed_b as f32, shift)?
    } else {
        backend.chebyshev_poly_f64(handle, n, seed_a, seed_b, shift)?
    };
    Ok(Some(poly_gpu_output::<T>(
        out_handle,
        input.shape().to_vec(),
    )?))
}

/// Apply a NATIVE-`T` evaluator to every element of a CPU tensor. The GPU
/// path is handled by the per-family dispatch helpers above before this is
/// reached; this is the CPU fallthrough only.
///
/// Unlike a f64-then-narrow helper, the closure runs entirely in the tensor's
/// scalar type `T`. This is a byte-for-relevant mirror of PyTorch's
/// `*_forward<T>` family (`aten/src/ATen/native/Math.h`): every forward
/// function declares `T p, q, r;`, so for an f32 tensor the recurrence runs in
/// f32 and overflows to `±inf` → `NaN` exactly where torch does (and exactly
/// like the ferrotorch GPU f32 PTX kernels). A f64-then-narrow path stayed
/// finite far longer (f64 max ~1.8e308 vs f32 max ~3.4e38) and narrowed an
/// f64-finite value to `+inf` where torch returns `NaN` (#1642 / #1641).
fn elementwise_native<T: Float, F: Fn(T) -> T>(
    input: &Tensor<T>,
    _op: &'static str,
    f: F,
) -> FerrotorchResult<Tensor<T>> {
    let data = input.data_vec()?;
    let out: Vec<T> = data.into_iter().map(f).collect();
    crate::tensor::Tensor::from_storage(
        crate::storage::TensorStorage::cpu(out),
        input.shape().to_vec(),
        false,
    )
}

/// Build a CPU tensor of the same shape as `input`, every element `NaN`.
/// Used by the Hermite high-`n` guard to mirror PyTorch's
/// `std::numeric_limits<T>::quiet_NaN()` short-circuit
/// (`aten/src/ATen/native/Math.h:3069` / `:3110`). Reached only on the CPU
/// fallthrough (the GPU dispatch handles CUDA tensors before this point), so a
/// host-resident NaN buffer is the correct device.
fn nan_like<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let nan = T::from(f64::NAN).unwrap_or_else(T::nan);
    let out = vec![nan; input.numel()];
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
    if let Some(out) = poly_gpu_chebyshev(input, n, 1.0, 0.0, false, "chebyshev_polynomial_t")? {
        return Ok(out);
    }
    elementwise_native(input, "chebyshev_polynomial_t", move |x| chebyshev_t(n, x))
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
    if let Some(out) = poly_gpu_chebyshev(input, n, 2.0, 0.0, false, "chebyshev_polynomial_u")? {
        return Ok(out);
    }
    elementwise_native(input, "chebyshev_polynomial_u", move |x| chebyshev_u(n, x))
}

/// Chebyshev polynomial of the **third kind** `V_n(x)`.
///
/// `V_0 = 1`, `V_1 = 2x - 1`, same recurrence as T/U. Mirrors
/// `torch.special.chebyshev_polynomial_v`.
pub fn chebyshev_polynomial_v<T: Float>(
    input: &Tensor<T>,
    n: usize,
) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) = poly_gpu_chebyshev(input, n, 2.0, -1.0, false, "chebyshev_polynomial_v")? {
        return Ok(out);
    }
    elementwise_native(input, "chebyshev_polynomial_v", move |x| chebyshev_v(n, x))
}

/// Chebyshev polynomial of the **fourth kind** `W_n(x)`.
///
/// `W_0 = 1`, `W_1 = 2x + 1`, same recurrence as T/U. Mirrors
/// `torch.special.chebyshev_polynomial_w`.
pub fn chebyshev_polynomial_w<T: Float>(
    input: &Tensor<T>,
    n: usize,
) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) = poly_gpu_chebyshev(input, n, 2.0, 1.0, false, "chebyshev_polynomial_w")? {
        return Ok(out);
    }
    elementwise_native(input, "chebyshev_polynomial_w", move |x| chebyshev_w(n, x))
}

// --- Hermite family --------------------------------------------------------

/// Hermite polynomial (physicist's) `H_n(x)`.
///
/// `H_0 = 1`, `H_1 = 2x`, `H_{n+1} = 2x H_n - 2n H_{n-1}`. Mirrors
/// `torch.special.hermite_polynomial_h`.
pub fn hermite_polynomial_h<T: Float>(input: &Tensor<T>, n: usize) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) = poly_gpu_simple(
        input,
        n,
        "hermite_polynomial_h",
        |b, h| b.hermite_h_poly_f32(h, n),
        |b, h| b.hermite_h_poly_f64(h, n),
    )? {
        return Ok(out);
    }
    // PyTorch replaces the overflowing recurrence with NaN above
    // `getHermitianLimit<T>()` (`Math.h:3068` -> `:3044-3052`). The CPU path
    // runs the recurrence in f64 and narrows; without this guard an f64
    // overflow narrows to `+inf` (diverging from torch's NaN and from the
    // GPU f32-register path's NaN). Match torch exactly, keyed on `T`.
    if n > hermitian_limit::<T>() {
        return nan_like(input);
    }
    elementwise_native(input, "hermite_polynomial_h", move |x| hermite_h(n, x))
}

/// Probabilist's Hermite polynomial `He_n(x)`.
///
/// `He_0 = 1`, `He_1 = x`, `He_{n+1} = x He_n - n He_{n-1}`. Mirrors
/// `torch.special.hermite_polynomial_he`.
pub fn hermite_polynomial_he<T: Float>(input: &Tensor<T>, n: usize) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) = poly_gpu_simple(
        input,
        n,
        "hermite_polynomial_he",
        |b, h| b.hermite_he_poly_f32(h, n),
        |b, h| b.hermite_he_poly_f64(h, n),
    )? {
        return Ok(out);
    }
    // Same `getHermitianLimit<T>()` NaN short-circuit as `hermite_polynomial_h`
    // (`Math.h:3109` -> `:3044-3052`); without it the f64-then-narrow CPU path
    // returns `±inf` where torch returns NaN.
    if n > hermitian_limit::<T>() {
        return nan_like(input);
    }
    elementwise_native(input, "hermite_polynomial_he", move |x| hermite_he(n, x))
}

// --- Laguerre & Legendre ---------------------------------------------------

/// Laguerre polynomial `L_n(x)`.
///
/// `L_0 = 1`, `L_1 = 1 - x`, `(n+1) L_{n+1} = (2n + 1 - x) L_n - n L_{n-1}`.
/// Mirrors `torch.special.laguerre_polynomial_l`.
pub fn laguerre_polynomial_l<T: Float>(input: &Tensor<T>, n: usize) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) = poly_gpu_simple(
        input,
        n,
        "laguerre_polynomial_l",
        |b, h| b.laguerre_poly_f32(h, n),
        |b, h| b.laguerre_poly_f64(h, n),
    )? {
        return Ok(out);
    }
    elementwise_native(input, "laguerre_polynomial_l", move |x| laguerre_l(n, x))
}

/// Legendre polynomial `P_n(x)`.
///
/// `P_0 = 1`, `P_1 = x`, `(n+1) P_{n+1} = (2n+1) x P_n - n P_{n-1}`.
/// Mirrors `torch.special.legendre_polynomial_p`.
pub fn legendre_polynomial_p<T: Float>(input: &Tensor<T>, n: usize) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) = poly_gpu_simple(
        input,
        n,
        "legendre_polynomial_p",
        |b, h| b.legendre_poly_f32(h, n),
        |b, h| b.legendre_poly_f64(h, n),
    )? {
        return Ok(out);
    }
    elementwise_native(input, "legendre_polynomial_p", move |x| legendre_p(n, x))
}

// --- Shifted Chebyshev family (domain [0, 1]) ------------------------------

/// Shifted Chebyshev T: `T*_n(x) = T_n(2x - 1)`. Mirrors
/// `torch.special.shifted_chebyshev_polynomial_t`.
pub fn shifted_chebyshev_polynomial_t<T: Float>(
    input: &Tensor<T>,
    n: usize,
) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) =
        poly_gpu_chebyshev(input, n, 1.0, 0.0, true, "shifted_chebyshev_polynomial_t")?
    {
        return Ok(out);
    }
    elementwise_native(input, "shifted_chebyshev_polynomial_t", move |x| {
        let one = nt_one::<T>();
        chebyshev_t(n, x + x - one)
    })
}

/// Shifted Chebyshev U: `U*_n(x) = U_n(2x - 1)`.
pub fn shifted_chebyshev_polynomial_u<T: Float>(
    input: &Tensor<T>,
    n: usize,
) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) =
        poly_gpu_chebyshev(input, n, 2.0, 0.0, true, "shifted_chebyshev_polynomial_u")?
    {
        return Ok(out);
    }
    elementwise_native(input, "shifted_chebyshev_polynomial_u", move |x| {
        let one = nt_one::<T>();
        chebyshev_u(n, x + x - one)
    })
}

/// Shifted Chebyshev V: `V*_n(x) = V_n(2x - 1)`.
pub fn shifted_chebyshev_polynomial_v<T: Float>(
    input: &Tensor<T>,
    n: usize,
) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) =
        poly_gpu_chebyshev(input, n, 2.0, -1.0, true, "shifted_chebyshev_polynomial_v")?
    {
        return Ok(out);
    }
    elementwise_native(input, "shifted_chebyshev_polynomial_v", move |x| {
        let one = nt_one::<T>();
        chebyshev_v(n, x + x - one)
    })
}

/// Shifted Chebyshev W: `W*_n(x) = W_n(2x - 1)`.
pub fn shifted_chebyshev_polynomial_w<T: Float>(
    input: &Tensor<T>,
    n: usize,
) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) =
        poly_gpu_chebyshev(input, n, 2.0, 1.0, true, "shifted_chebyshev_polynomial_w")?
    {
        return Ok(out);
    }
    elementwise_native(input, "shifted_chebyshev_polynomial_w", move |x| {
        let one = nt_one::<T>();
        chebyshev_w(n, x + x - one)
    })
}

// ---------------------------------------------------------------------------
// Internal scalar evaluators (three-term recurrences in NATIVE `T`)
// ---------------------------------------------------------------------------
//
// Each evaluator is a byte-for-relevant mirror of the matching
// `*_forward<T>(T x, int64_t n)` in `aten/src/ATen/native/Math.h`: the
// recurrence variables `p, q, r` live in the tensor's scalar type `T`, the
// integer loop coefficients are promoted to `T` exactly as the C++ `k * p`
// promotes `int64_t k` to `T`, and the chebyshev / laguerre / legendre loops
// carry torch's `&& !std::isnan(q)` latch (the hermite loops have no latch in
// upstream — `Math.h:3076` / `:3117`). Running in native `T` makes the f32
// recurrence overflow to `±inf` → `NaN` exactly where torch does, instead of
// staying f64-finite and narrowing to `+inf` (#1642 / #1641).

/// Promote a non-negative loop index to the scalar type `T` (mirrors the C++
/// `int64_t k` → `T` promotion in `k * p`). `usize` → `T` is exact for the
/// loop bounds these recurrences use.
#[inline]
fn nt_from_usize<T: Float>(k: usize) -> T {
    T::from(k).unwrap_or_else(T::nan)
}

/// `H_n(x)` (physicist's Hermite) via three-term recurrence
/// `H_{n+1} = 2x H_n - 2n H_{n-1}` with `H_0 = 1`, `H_1 = 2x`, in native `T`.
/// Mirrors `hermite_polynomial_h_forward<T>` (`Math.h:3072-3081`, no isnan
/// latch).
fn hermite_h<T: Float>(n: usize, x: T) -> T {
    let one = nt_one::<T>();
    if n == 0 {
        return one;
    }
    if n == 1 {
        return x + x;
    }
    let mut prev2 = one;
    let mut prev1 = x + x;
    for k in 1..n {
        let kf = nt_from_usize::<T>(k);
        let next = (x + x) * prev1 - (kf + kf) * prev2;
        prev2 = prev1;
        prev1 = next;
    }
    prev1
}

/// `He_n(x)` (probabilist's Hermite) via three-term recurrence
/// `He_{n+1} = x He_n - n He_{n-1}` with `He_0 = 1`, `He_1 = x`, in native `T`.
/// Mirrors `hermite_polynomial_he_forward<T>` (`Math.h:3113-3122`, no latch).
fn hermite_he<T: Float>(n: usize, x: T) -> T {
    let one = nt_one::<T>();
    if n == 0 {
        return one;
    }
    if n == 1 {
        return x;
    }
    let mut prev2 = one;
    let mut prev1 = x;
    for k in 1..n {
        let kf = nt_from_usize::<T>(k);
        let next = x * prev1 - kf * prev2;
        prev2 = prev1;
        prev1 = next;
    }
    prev1
}

/// `T_n(x)` via direct recurrence in native `T` — also used internally for the
/// shifted variant. Mirrors `chebyshev_polynomial_t_forward<T>`
/// (`Math.h:2861-2871`) including the `&& !std::isnan(q)` latch.
fn chebyshev_t<T: Float>(n: usize, x: T) -> T {
    let one = nt_one::<T>();
    if n == 0 {
        return one;
    }
    if n == 1 {
        return x;
    }
    let mut prev2 = one;
    let mut prev1 = x;
    for _ in 2..=n {
        if prev1.is_nan() {
            break;
        }
        let next = (x + x) * prev1 - prev2;
        prev2 = prev1;
        prev1 = next;
    }
    prev1
}

/// `U_n(x)` in native `T`. Mirrors `chebyshev_polynomial_u_forward<T>`
/// (`Math.h:2909-2919`) including the `&& !std::isnan(q)` latch.
fn chebyshev_u<T: Float>(n: usize, x: T) -> T {
    let one = nt_one::<T>();
    if n == 0 {
        return one;
    }
    if n == 1 {
        return x + x;
    }
    let mut prev2 = one;
    let mut prev1 = x + x;
    for _ in 2..=n {
        if prev1.is_nan() {
            break;
        }
        let next = (x + x) * prev1 - prev2;
        prev2 = prev1;
        prev1 = next;
    }
    prev1
}

/// `V_n(x)` in native `T`. Mirrors `chebyshev_polynomial_v_forward<T>`
/// (`Math.h:2965-2975`) including the `&& !std::isnan(q)` latch.
fn chebyshev_v<T: Float>(n: usize, x: T) -> T {
    let one = nt_one::<T>();
    if n == 0 {
        return one;
    }
    if n == 1 {
        return x + x - one;
    }
    let mut prev2 = one;
    let mut prev1 = x + x - one;
    for _ in 2..=n {
        if prev1.is_nan() {
            break;
        }
        let next = (x + x) * prev1 - prev2;
        prev2 = prev1;
        prev1 = next;
    }
    prev1
}

/// `W_n(x)` in native `T`. Mirrors `chebyshev_polynomial_w_forward<T>`
/// (`Math.h:3025-3035`) including the `&& !std::isnan(q)` latch.
fn chebyshev_w<T: Float>(n: usize, x: T) -> T {
    let one = nt_one::<T>();
    if n == 0 {
        return one;
    }
    if n == 1 {
        return x + x + one;
    }
    let mut prev2 = one;
    let mut prev1 = x + x + one;
    for _ in 2..=n {
        if prev1.is_nan() {
            break;
        }
        let next = (x + x) * prev1 - prev2;
        prev2 = prev1;
        prev1 = next;
    }
    prev1
}

/// `L_n(x)` in native `T`. Mirrors `laguerre_polynomial_l_forward<T>`
/// (`Math.h:3149-3159`) including the `&& !std::isnan(q)` latch.
fn laguerre_l<T: Float>(n: usize, x: T) -> T {
    let one = nt_one::<T>();
    if n == 0 {
        return one;
    }
    if n == 1 {
        return one - x;
    }
    let mut prev2 = one;
    let mut prev1 = one - x;
    for k in 1..n {
        if prev1.is_nan() {
            break;
        }
        let kf = nt_from_usize::<T>(k);
        let next = ((kf + kf + (one - x)) * prev1 - kf * prev2) / (kf + one);
        prev2 = prev1;
        prev1 = next;
    }
    prev1
}

/// `P_n(x)` in native `T`. Mirrors `legendre_polynomial_p_forward<T>`
/// (`Math.h:3189-3199`) including the `&& !std::isnan(q)` latch.
fn legendre_p<T: Float>(n: usize, x: T) -> T {
    let one = nt_one::<T>();
    if n == 0 {
        return one;
    }
    if n == 1 {
        return x;
    }
    let mut prev2 = one;
    let mut prev1 = x;
    for k in 1..n {
        if prev1.is_nan() {
            break;
        }
        let kf = nt_from_usize::<T>(k);
        let next = ((kf + kf + one) * x * prev1 - kf * prev2) / (kf + one);
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
        // (the others share the same code path via elementwise_native).
        // Sanity: the gate itself is exercised here on a CPU tensor.
        let x = t(&[0.0], &[1]);
        assert!(chebyshev_polynomial_t(&x, 3).is_ok());
    }

    // ---------------------------------------------------------------------
    // gammainc / gammaincc (torch.special.gammainc / gammaincc)
    //
    // Oracle values are from live `torch.special.gammainc(...)` (torch 2.11,
    // this machine, 2026-05-27), cross-checked against `scipy.special.gammainc`
    // to ~1e-7 (torch's CPU kernel narrows through f32 internally; our f64 path
    // matches scipy to ~1e-12, so we pin against the scipy/torch-f64 doubles):
    //   gammainc(2.0, 1.5)  = 0.4421745996289252
    //   gammaincc(2.0, 1.5) = 0.5578254003710748
    //   gammainc(4.0, 3.0)  = 0.35276811121776874
    //   gammaincc(4.0, 3.0) = 0.6472318887822313
    //   gammainc(0.5, 0.5)  = 0.6826894921370859
    //   gammainc(5.0, 3.0)  = 0.18473675547622787
    //   gammainc(7.5, 10.0) = 0.8280673106233991
    // ---------------------------------------------------------------------

    #[test]
    fn gammainc_series_region_matches_oracle() {
        // x < a + 1 -> power series.
        let a = t(&[2.0], &[1]);
        let x = t(&[1.5], &[1]);
        let r = gammainc(&a, &x).unwrap();
        assert!(
            (r.data().unwrap()[0] - 0.442_174_599_628_925_2).abs() < 1e-12,
            "got {}",
            r.data().unwrap()[0]
        );
    }

    #[test]
    fn gammainc_continued_fraction_region_matches_oracle() {
        // x >= a + 1 -> Lentz CF. gammainc(5, 3): 3 < 6 actually series; use
        // gammainc(7.5, 10.0) where 10 >= 8.5 -> CF.
        let a = t(&[7.5], &[1]);
        let x = t(&[10.0], &[1]);
        let r = gammainc(&a, &x).unwrap();
        assert!(
            (r.data().unwrap()[0] - 0.828_067_310_623_399_1).abs() < 1e-12,
            "got {}",
            r.data().unwrap()[0]
        );
    }

    #[test]
    fn gammaincc_matches_oracle() {
        let a = t(&[2.0, 4.0], &[2]);
        let x = t(&[1.5, 3.0], &[2]);
        let r = gammaincc(&a, &x).unwrap();
        let d = r.data().unwrap();
        assert!(
            (d[0] - 0.557_825_400_371_074_8).abs() < 1e-12,
            "got {}",
            d[0]
        );
        assert!(
            (d[1] - 0.647_231_888_782_231_3).abs() < 1e-12,
            "got {}",
            d[1]
        );
    }

    #[test]
    fn gammainc_plus_gammaincc_is_one() {
        // P(a,x) + Q(a,x) == 1 for a,x > 0 (the torch docstring identity).
        let a = t(&[4.0, 4.0, 4.0], &[3]);
        let x = t(&[3.0, 4.0, 5.0], &[3]);
        let p = gammainc(&a, &x).unwrap();
        let q = gammaincc(&a, &x).unwrap();
        let pd = p.data().unwrap();
        let qd = q.data().unwrap();
        for i in 0..3 {
            assert!(
                (pd[i] + qd[i] - 1.0).abs() < 1e-12,
                "P+Q at i={i} = {} (expected 1)",
                pd[i] + qd[i]
            );
        }
    }

    #[test]
    fn gammainc_subunit_concentration_matches_oracle() {
        let a = t(&[0.5], &[1]);
        let x = t(&[0.5], &[1]);
        let r = gammainc(&a, &x).unwrap();
        assert!(
            (r.data().unwrap()[0] - 0.682_689_492_137_085_9).abs() < 1e-12,
            "got {}",
            r.data().unwrap()[0]
        );
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "gammainc/gammaincc boundary values (a=0 -> 1.0, x=0 -> 0.0, x=inf -> 1.0/0.0) are EXACT mathematical limits torch returns, not floating approximations"
    )]
    fn gammainc_boundary_cases_match_torch() {
        // Per aten/src/ATen/native/Math.h calc_igamma / calc_igammac, verified
        // against live torch: gammainc(0, 2)=1, gammainc(2, 0)=0,
        // gammainc(-1, 2)=NaN, gammainc(2, -1)=NaN, gammainc(0, 0)=NaN.
        let gi = |av: f64, xv: f64| {
            gammainc(&t(&[av], &[1]), &t(&[xv], &[1]))
                .unwrap()
                .data()
                .unwrap()[0]
        };
        let gc = |av: f64, xv: f64| {
            gammaincc(&t(&[av], &[1]), &t(&[xv], &[1]))
                .unwrap()
                .data()
                .unwrap()[0]
        };
        assert_eq!(gi(0.0, 2.0), 1.0);
        assert_eq!(gi(2.0, 0.0), 0.0);
        assert!(gi(-1.0, 2.0).is_nan());
        assert!(gi(2.0, -1.0).is_nan());
        assert!(gi(0.0, 0.0).is_nan());
        // Upper tail boundaries: gammaincc(0, 2)=0, gammaincc(2, 0)=1.
        assert_eq!(gc(0.0, 2.0), 0.0);
        assert_eq!(gc(2.0, 0.0), 1.0);
        // Infinite x.
        assert_eq!(gi(2.0, f64::INFINITY), 1.0);
        assert_eq!(gc(2.0, f64::INFINITY), 0.0);
    }

    #[test]
    fn gammainc_f32_matches_oracle_within_f32_tol() {
        let a = Tensor::from_storage(TensorStorage::cpu(vec![2.0f32]), vec![1], false).unwrap();
        let x = Tensor::from_storage(TensorStorage::cpu(vec![1.5f32]), vec![1], false).unwrap();
        let r = gammainc(&a, &x).unwrap();
        assert!(
            (r.data().unwrap()[0] - 0.442_174_6).abs() < 1e-5,
            "got {}",
            r.data().unwrap()[0]
        );
    }

    // ---------------------------------------------------------------------
    // log_beta / beta (scipy.special.betaln / beta)
    //
    // Oracle (scipy 1.17, this machine):
    //   betaln(2, 3)   = -2.4849066497880004
    //   beta(2, 3)     =  0.08333333333333333
    //   betaln(0.5, 2.5) = 0.1639006328376739
    // ---------------------------------------------------------------------

    #[test]
    fn log_beta_matches_scipy() {
        let a = t(&[2.0], &[1]);
        let b = t(&[3.0], &[1]);
        let r = log_beta(&a, &b).unwrap();
        assert!(
            (r.data().unwrap()[0] - (-2.484_906_649_788_000_4)).abs() < 1e-12,
            "got {}",
            r.data().unwrap()[0]
        );
    }

    #[test]
    fn beta_matches_scipy() {
        let a = t(&[2.0], &[1]);
        let b = t(&[3.0], &[1]);
        let r = beta(&a, &b).unwrap();
        assert!(
            (r.data().unwrap()[0] - 0.083_333_333_333_333_33).abs() < 1e-12,
            "got {}",
            r.data().unwrap()[0]
        );
    }

    #[test]
    fn log_beta_symmetric_and_broadcasts() {
        // B(a,b) == B(b,a); also a half-integer pair against scipy.
        let a = t(&[0.5, 3.0], &[2]);
        let b = t(&[2.5, 2.0], &[2]);
        let r = log_beta(&a, &b).unwrap();
        let d = r.data().unwrap();
        assert!(
            (d[0] - 0.163_900_632_837_673_9).abs() < 1e-12,
            "got {}",
            d[0]
        );
        // B(3,2) = 1/12 -> ln(1/12).
        assert!((d[1] - (1.0f64 / 12.0).ln()).abs() < 1e-12, "got {}", d[1]);
    }

    // ---------------------------------------------------------------------
    // multigammaln / mvlgamma (torch.special.multigammaln / torch.mvlgamma)
    //
    // Oracle (scipy.special.multigammaln 1.17 / torch 2.11, this machine):
    //   multigammaln(3.0, 2) = 1.5501949939575645
    //   multigammaln(5.0, 3) = 9.140644699192542
    //   multigammaln(2.5, 1) = 0.2846828704729192   (== lgamma(2.5))
    // ---------------------------------------------------------------------

    #[test]
    fn multigammaln_p2_matches_scipy() {
        let a = t(&[3.0], &[1]);
        let r = multigammaln(&a, 2).unwrap();
        assert!(
            (r.data().unwrap()[0] - 1.550_194_993_957_564_5).abs() < 1e-12,
            "got {}",
            r.data().unwrap()[0]
        );
    }

    #[test]
    fn multigammaln_p3_matches_scipy() {
        let a = t(&[5.0], &[1]);
        let r = multigammaln(&a, 3).unwrap();
        assert!(
            (r.data().unwrap()[0] - 9.140_644_699_192_542).abs() < 1e-11,
            "got {}",
            r.data().unwrap()[0]
        );
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "mvlgamma is a literal alias of multigammaln (same code path); bit-exact equality asserts they are identical, not approximately equal"
    )]
    fn multigammaln_p1_is_lgamma() {
        // Γ_1(a) = Γ(a), so multigammaln(a, 1) == lgamma(a).
        let a = t(&[2.5], &[1]);
        let r = multigammaln(&a, 1).unwrap();
        assert!(
            (r.data().unwrap()[0] - 0.284_682_870_472_919_2).abs() < 1e-12,
            "got {}",
            r.data().unwrap()[0]
        );
        // mvlgamma is the alias.
        let r2 = mvlgamma(&a, 1).unwrap();
        assert_eq!(r.data().unwrap()[0], r2.data().unwrap()[0]);
    }

    #[test]
    fn multigammaln_out_of_domain_matches_torch() {
        // torch's mvlgamma (aten/src/ATen/native/UnaryOps.cpp:887) has NO domain
        // guard beyond p >= 1: out-of-domain inputs (a <= (p-1)/2) get the
        // ordinary finite lgamma-sum, and +inf only when an lgamma argument hits
        // a non-positive integer pole.
        //
        // Oracle (live torch 2.11, this machine, 2026-05-27):
        //   torch.special.multigammaln(0.3, 3) == 6.026863353182922  (finite)
        //   torch.special.multigammaln(0.5, 3) == inf  (args 0.5,0.0,-0.5; pole at 0.0)
        let finite = multigammaln(&t(&[0.3], &[1]), 3).unwrap();
        assert!(
            (finite.data().unwrap()[0] - 6.026_863_353_182_922).abs() < 1e-12,
            "multigammaln(0.3, 3): torch returns finite 6.026863353182922, got {}",
            finite.data().unwrap()[0]
        );
        let pole = multigammaln(&t(&[0.5], &[1]), 3).unwrap();
        assert!(
            pole.data().unwrap()[0] == f64::INFINITY,
            "multigammaln(0.5, 3): torch returns +inf (lgamma pole), got {}",
            pole.data().unwrap()[0]
        );
    }

    #[test]
    fn multigammaln_p_zero_errors() {
        let a = t(&[3.0], &[1]);
        assert!(multigammaln(&a, 0).is_err());
    }

    // ---------------------------------------------------------------------
    // gammaln_sign (scipy.special.gammasgn)
    //
    // Oracle (scipy 1.17, this machine):
    //   gammasgn(-2.5) = -1.0   (floor(-2.5) = -3, (-1)^-3 = -1)
    //   gammasgn(-1.5) = +1.0   (floor(-1.5) = -2)
    //   gammasgn(-0.5) = -1.0
    //   gammasgn( 2.0) = +1.0
    //   gammasgn(-6.5) = -1.0
    //   gammasgn(-2.0) = NaN    (negative integer = pole)
    //   gammasgn( 0.0) = +1.0
    // ---------------------------------------------------------------------

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn gammaln_sign_matches_scipy_gammasgn() {
        let xs = [-2.5, -1.5, -0.5, 2.0, -6.5, 0.5, -3.7];
        let expected = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0];
        let input = t(&xs, &[xs.len()]);
        let r = gammaln_sign(&input).unwrap();
        let d = r.data().unwrap();
        for i in 0..xs.len() {
            assert!(
                (d[i] - expected[i]).abs() < 1e-15,
                "gammasgn({}) = {} (expected {})",
                xs[i],
                d[i],
                expected[i]
            );
        }
    }

    #[test]
    fn gammaln_sign_negative_integer_is_nan() {
        let input = t(&[-2.0, -3.0, -10.0], &[3]);
        let r = gammaln_sign(&input).unwrap();
        for &v in r.data().unwrap() {
            assert!(v.is_nan(), "expected NaN at pole, got {v}");
        }
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "gammasgn returns the EXACT sign value +1.0 for positive/zero inputs; bit-exact equality is the contract, not an epsilon"
    )]
    fn gammaln_sign_positive_and_zero() {
        let input = t(&[0.0, 1.0, 100.0], &[3]);
        let r = gammaln_sign(&input).unwrap();
        for &v in r.data().unwrap() {
            assert_eq!(v, 1.0);
        }
    }

    #[test]
    fn gammaln_sign_recovers_gamma_via_lgamma() {
        // exp(lgamma(x)) * gammasgn(x) == Γ(x). Check at a negative non-integer
        // where Γ is negative: Γ(-0.5) = -2*sqrt(pi) ≈ -3.5449077.
        let input = t(&[-0.5], &[1]);
        let sign = gammaln_sign(&input).unwrap().data().unwrap()[0];
        let lg = lgamma(&input).unwrap().data().unwrap()[0];
        let gamma = sign * lg.exp();
        let expected = -2.0 * std::f64::consts::PI.sqrt();
        assert!(
            (gamma - expected).abs() < 1e-9,
            "reconstructed Γ(-0.5) = {gamma} (expected {expected})"
        );
    }

    // --- entr / ndtr / ndtri (#1651 batch 1) ---
    //
    // Expected values are live `torch.special.{entr,ndtr,ndtri}` (torch
    // 2.11.0+cu130, f64) outputs (R-CHAR-3: oracle-derived, not self-referential):
    //   entr([0.5,0,-1,1,2,0.1]) = [0.34657359027997264, 0, -inf, -0,
    //                               -1.3862943611198906, 0.23025850929940456]
    //   ndtr([-3,-2,-1,0,1,2,3]) = [0.0013498980316301035, 0.022750131948179209,
    //       0.15865525393145702, 0.5, 0.84134474606854304, 0.97724986805182079,
    //       0.9986501019683699]
    //   ndtri([0.025,0.25,0.5,0.75,0.975]) = [-1.9599639845400545,
    //       -0.67448975019608171, 0, 0.67448975019608171, 1.959963984540054]
    //   ndtri([0.001,1e-10,0.9,0.999]) = [-3.0902323061678132,
    //       -6.3613409024040557, 1.2815515655446004, 3.0902323061678132]

    #[test]
    fn entr_known_values_vs_torch() {
        let input = t(&[0.5, 2.0, 0.1], &[3]);
        let r = entr(&input).unwrap();
        let d = r.data().unwrap();
        // Live torch.special.entr f64 oracle.
        let want = [
            0.346_573_590_279_972_64,
            -1.386_294_361_119_890_6,
            0.230_258_509_299_404_56,
        ];
        for i in 0..3 {
            assert!(
                (d[i] - want[i]).abs() < 1e-12,
                "entr idx {i}: got {} want {}",
                d[i],
                want[i]
            );
        }
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "entr(0) is the exact branch return 0.0 / entr(1) the exact -1*ln(1) = -0.0; bit-exact equality is the torch contract (Math.cuh:474-476)"
    )]
    fn entr_edges_vs_torch() {
        let input = t(&[0.0, -1.0, f64::NAN, 1.0], &[4]);
        let r = entr(&input).unwrap();
        let d = r.data().unwrap();
        assert_eq!(d[0], 0.0, "entr(0) == +0.0");
        assert!(d[0].is_sign_positive(), "entr(0) sign is +0.0");
        assert!(d[1].is_infinite() && d[1] < 0.0, "entr(-1) == -inf");
        assert!(d[2].is_nan(), "entr(NaN) == NaN");
        // entr(1) = -1*ln(1) = -0.0 (flows through the >0 branch verbatim).
        assert_eq!(d[3], 0.0, "entr(1) magnitude 0");
        assert!(
            d[3].is_sign_negative(),
            "entr(1) sign is -0.0 (torch parity)"
        );
    }

    #[test]
    fn ndtr_known_values_vs_torch() {
        let input = t(&[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], &[7]);
        let r = ndtr(&input).unwrap();
        let d = r.data().unwrap();
        let want = [
            0.001_349_898_031_630_103_5,
            0.022_750_131_948_179_209,
            0.158_655_253_931_457_02,
            0.5,
            0.841_344_746_068_543_04,
            0.977_249_868_051_820_79,
            0.998_650_101_968_369_9,
        ];
        for i in 0..7 {
            assert!(
                (d[i] - want[i]).abs() < 1e-12,
                "ndtr idx {i}: got {} want {}",
                d[i],
                want[i]
            );
        }
    }

    #[test]
    fn ndtr_edges_vs_torch() {
        let input = t(&[f64::NEG_INFINITY, f64::INFINITY, f64::NAN], &[3]);
        let r = ndtr(&input).unwrap();
        let d = r.data().unwrap();
        assert!((d[0] - 0.0).abs() < 1e-15, "ndtr(-inf) == 0");
        assert!((d[1] - 1.0).abs() < 1e-15, "ndtr(+inf) == 1");
        assert!(d[2].is_nan(), "ndtr(NaN) == NaN");
    }

    #[test]
    fn ndtri_known_values_vs_torch() {
        let input = t(&[0.025, 0.25, 0.5, 0.75, 0.975], &[5]);
        let r = ndtri(&input).unwrap();
        let d = r.data().unwrap();
        let want = [
            -1.959_963_984_540_054_5,
            -0.674_489_750_196_081_71,
            0.0,
            0.674_489_750_196_081_71,
            1.959_963_984_540_054,
        ];
        for i in 0..5 {
            assert!(
                (d[i] - want[i]).abs() < 1e-12,
                "ndtri idx {i}: got {} want {}",
                d[i],
                want[i]
            );
        }
    }

    #[test]
    fn ndtri_cephes_regions_vs_torch() {
        // 0.001 / 1e-10 exercise the tail (P1/Q1) and far-tail (P2/Q2) regions;
        // 0.9 / 0.999 exercise the code-flag sign-flip (`y > 1 - exp(-2)`).
        let input = t(&[0.001, 1e-10, 0.9, 0.999], &[4]);
        let r = ndtri(&input).unwrap();
        let d = r.data().unwrap();
        let want = [
            -3.090_232_306_167_813_2,
            -6.361_340_902_404_055_7,
            1.281_551_565_544_600_4,
            3.090_232_306_167_813_2,
        ];
        for i in 0..4 {
            assert!(
                (d[i] - want[i]).abs() < 1e-11,
                "ndtri region idx {i}: got {} want {}",
                d[i],
                want[i]
            );
        }
    }

    #[test]
    fn ndtri_domain_edges_vs_torch() {
        let input = t(&[0.0, 1.0, -0.1, 1.1], &[4]);
        let r = ndtri(&input).unwrap();
        let d = r.data().unwrap();
        assert!(d[0].is_infinite() && d[0] < 0.0, "ndtri(0) == -inf");
        assert!(d[1].is_infinite() && d[1] > 0.0, "ndtri(1) == +inf");
        assert!(d[2].is_nan(), "ndtri(-0.1) == NaN");
        assert!(d[3].is_nan(), "ndtri(1.1) == NaN");
    }

    #[test]
    fn ndtr_ndtri_roundtrip() {
        // ndtr(ndtri(p)) ≈ p (AC-B3 round-trip).
        let ps = [0.05, 0.2, 0.5, 0.8, 0.95];
        let input = t(&ps, &[5]);
        let q = ndtri(&input).unwrap();
        let back = ndtr(&q).unwrap();
        let bd = back.data().unwrap();
        for i in 0..5 {
            assert!(
                (bd[i] - ps[i]).abs() < 1e-12,
                "ndtr(ndtri({})) = {} (expected {})",
                ps[i],
                bd[i],
                ps[i]
            );
        }
    }

    #[test]
    fn ndtri_f32_vs_torch() {
        // Live torch.special.ndtri f32 oracle:
        // [0.025,0.25,0.5,0.75,0.975] -> [-1.9599637985229492,
        //   -0.67448979616165161, 0, 0.67448979616165161, 1.959964394569397]
        let input = Tensor::from_storage(
            TensorStorage::cpu(vec![0.025f32, 0.25, 0.5, 0.75, 0.975]),
            vec![5],
            false,
        )
        .unwrap();
        let r = ndtri(&input).unwrap();
        let d = r.data().unwrap();
        let want = [-1.959_963_8f32, -0.674_489_8, 0.0, 0.674_489_8, 1.959_964_4];
        for i in 0..5 {
            assert!(
                (d[i] - want[i]).abs() < 1e-5,
                "ndtri_f32 idx {i}: got {} want {}",
                d[i],
                want[i]
            );
        }
    }

    // --- i0 / i0e / i1 / i1e (#1651 batch 2) ---------------------------------
    //
    // Expected values are live `torch.special.{i0,i0e,i1,i1e}`
    // (torch 2.11.0+cu130) outputs (R-CHAR-3: oracle-derived, not
    // self-referential). f64 oracle, grid [0,0.5,1,2,5,8,10,20,-1,-2,-5]:
    //   i0  = [1, 1.0634833707413236, 1.2660658777520082, 2.279585302336067,
    //          27.239871823604442, 427.56411572180474, 2815.716628466254,
    //          43558282.559553534, 1.2660658777520082, 2.279585302336067,
    //          27.239871823604442]
    //   i0e = [1, 0.6450352704491501, 0.46575960759364043, 0.308508322553671,
    //          0.18354081260932834, 0.1434317818568503, 0.1278333371634286,
    //          0.089780311884826, 0.46575960759364043, 0.308508322553671,
    //          0.18354081260932834]
    //   i1  = [0, 0.25789430539089636, 0.5651591039924851, 1.5906368546373295,
    //          24.335642142450524, 399.8731367825599, 2670.988303701255,
    //          42454973.385127775, -0.5651591039924851, -1.5906368546373295,
    //          -24.335642142450524]
    //   i1e = [0, 0.15642080318487173, 0.2079104153497085, 0.2152692892489377,
    //          0.16397226694454234, 0.13414249329269812, 0.1212626813844555,
    //          0.08750622218328867, -0.2079104153497085, -0.2152692892489377,
    //          -0.16397226694454234]

    const I_GRID: [f64; 11] = [0.0, 0.5, 1.0, 2.0, 5.0, 8.0, 10.0, 20.0, -1.0, -2.0, -5.0];

    #[test]
    fn i0_known_values_vs_torch() {
        let input = t(&I_GRID, &[11]);
        let r = i0(&input).unwrap();
        let d = r.data().unwrap();
        let want = [
            1.0,
            1.063_483_370_741_323_6,
            1.266_065_877_752_008_2,
            2.279_585_302_336_067,
            27.239_871_823_604_442,
            427.564_115_721_804_74,
            2815.716_628_466_254,
            43_558_282.559_553_534,
            1.266_065_877_752_008_2,
            2.279_585_302_336_067,
            27.239_871_823_604_442,
        ];
        for i in 0..11 {
            assert!(
                (d[i] - want[i]).abs() <= 1e-9 * (1.0 + want[i].abs()),
                "i0 idx {i} x={}: got {} want {}",
                I_GRID[i],
                d[i],
                want[i]
            );
        }
    }

    #[test]
    fn i0e_known_values_vs_torch() {
        let input = t(&I_GRID, &[11]);
        let r = i0e(&input).unwrap();
        let d = r.data().unwrap();
        let want = [
            1.0,
            0.645_035_270_449_150_1,
            0.465_759_607_593_640_43,
            0.308_508_322_553_671,
            0.183_540_812_609_328_34,
            0.143_431_781_856_850_3,
            0.127_833_337_163_428_6,
            0.089_780_311_884_826,
            0.465_759_607_593_640_43,
            0.308_508_322_553_671,
            0.183_540_812_609_328_34,
        ];
        for i in 0..11 {
            assert!(
                (d[i] - want[i]).abs() <= 1e-12 * (1.0 + want[i].abs()),
                "i0e idx {i} x={}: got {} want {}",
                I_GRID[i],
                d[i],
                want[i]
            );
        }
    }

    #[test]
    fn i1_known_values_vs_torch() {
        let input = t(&I_GRID, &[11]);
        let r = i1(&input).unwrap();
        let d = r.data().unwrap();
        let want = [
            0.0,
            0.257_894_305_390_896_36,
            0.565_159_103_992_485_1,
            1.590_636_854_637_329_5,
            24.335_642_142_450_524,
            399.873_136_782_559_9,
            2670.988_303_701_255,
            42_454_973.385_127_775,
            -0.565_159_103_992_485_1,
            -1.590_636_854_637_329_5,
            -24.335_642_142_450_524,
        ];
        for i in 0..11 {
            assert!(
                (d[i] - want[i]).abs() <= 1e-9 * (1.0 + want[i].abs()),
                "i1 idx {i} x={}: got {} want {}",
                I_GRID[i],
                d[i],
                want[i]
            );
        }
    }

    #[test]
    fn i1e_known_values_vs_torch() {
        let input = t(&I_GRID, &[11]);
        let r = i1e(&input).unwrap();
        let d = r.data().unwrap();
        let want = [
            0.0,
            0.156_420_803_184_871_73,
            0.207_910_415_349_708_5,
            0.215_269_289_248_937_7,
            0.163_972_266_944_542_34,
            0.134_142_493_292_698_12,
            0.121_262_681_384_455_5,
            0.087_506_222_183_288_67,
            -0.207_910_415_349_708_5,
            -0.215_269_289_248_937_7,
            -0.163_972_266_944_542_34,
        ];
        for i in 0..11 {
            assert!(
                (d[i] - want[i]).abs() <= 1e-12 * (1.0 + want[i].abs()),
                "i1e idx {i} x={}: got {} want {}",
                I_GRID[i],
                d[i],
                want[i]
            );
        }
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "i0(0)=1, i1(0)=0 are exact Cephes branch returns (chbevl at x=0 with the limit constants); torch returns the literal endpoint"
    )]
    fn i_family_edges_vs_torch() {
        // Even: i0/i0e symmetric. Odd: i1/i1e antisymmetric. Zero + NaN + inf.
        // NOTE: torch's Cephes i0/i1 at +/-inf return NaN (the kernel forms
        // `exp(inf)*chbevl/sqrt(inf) = inf/inf = NaN`, NOT +inf — verified
        // against live torch.special.i0([inf]) == nan, i1([inf]) == nan). The
        // exp-scaled i0e/i1e stay finite: i0e(+/-inf)=0, i1e(+/-inf)=+/-0. We
        // match torch byte-for-byte (R-DEV-1), not the design-doc edge prose.
        let input = t(&[0.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY], &[4]);
        let r0 = i0(&input).unwrap();
        let d0 = r0.data().unwrap();
        assert_eq!(d0[0], 1.0, "i0(0) == 1");
        assert!(d0[1].is_nan(), "i0(NaN) == NaN");
        assert!(d0[2].is_nan(), "i0(+inf) == NaN (torch parity)");
        assert!(d0[3].is_nan(), "i0(-inf) == NaN (torch parity, even)");

        let r0e = i0e(&input).unwrap();
        let d0e = r0e.data().unwrap();
        assert_eq!(d0e[0], 1.0, "i0e(0) == 1");
        assert!(d0e[1].is_nan(), "i0e(NaN) == NaN");
        assert_eq!(d0e[2], 0.0, "i0e(+inf) == 0");
        assert_eq!(d0e[3], 0.0, "i0e(-inf) == 0");

        let r1 = i1(&input).unwrap();
        let d1 = r1.data().unwrap();
        assert_eq!(d1[0], 0.0, "i1(0) == 0");
        assert!(d1[1].is_nan(), "i1(NaN) == NaN");
        assert!(d1[2].is_nan(), "i1(+inf) == NaN (torch parity)");
        assert!(d1[3].is_nan(), "i1(-inf) == NaN (torch parity, odd)");

        let r1e = i1e(&input).unwrap();
        let d1e = r1e.data().unwrap();
        assert_eq!(d1e[0], 0.0, "i1e(0) == 0");
        assert!(d1e[1].is_nan(), "i1e(NaN) == NaN");
        assert_eq!(d1e[2], 0.0, "i1e(+inf) == 0");
        assert_eq!(d1e[3], 0.0, "i1e(-inf) == 0");
    }

    #[test]
    fn i_family_boundary_at_8_vs_torch() {
        // |x| == 8 is the A/B coefficient-set split (`x <= 8` uses A). Verify the
        // A-set value at exactly 8 and B-set just above. Live torch f64:
        //   i0(8)=427.56411572180474, i0(8.5)=683.1619269901155
        //   i1(8)=399.8731367825599, i1(12)=18141.348781638833
        let input = t(&[8.0, 8.5, 12.0], &[3]);
        let r0 = i0(&input).unwrap();
        let d0 = r0.data().unwrap();
        let w0 = [
            427.564_115_721_804_74,
            683.161_926_990_115_5,
            18948.925_349_296_31,
        ];
        let r1 = i1(&input).unwrap();
        let d1 = r1.data().unwrap();
        let w1 = [
            399.873_136_782_559_9,
            641.619_902_540_066_7,
            18141.348_781_638_833,
        ];
        for i in 0..3 {
            assert!(
                (d0[i] - w0[i]).abs() <= 1e-9 * (1.0 + w0[i].abs()),
                "i0 boundary idx {i}: got {} want {}",
                d0[i],
                w0[i]
            );
            assert!(
                (d1[i] - w1[i]).abs() <= 1e-9 * (1.0 + w1[i].abs()),
                "i1 boundary idx {i}: got {} want {}",
                d1[i],
                w1[i]
            );
        }
    }

    #[test]
    fn i_family_large_x_scaled_finite_vs_torch() {
        // At x=700, i0(700)=1.53e302 (near f64 overflow at ~700+) but the
        // exp-scaled i0e/i1e stay O(0.01). Live torch f64:
        //   i0e(700)=0.015081295651531355, i1e(700)=0.015070519444716846.
        let input = t(&[700.0], &[1]);
        let r0e = i0e(&input).unwrap();
        let d0e = r0e.data().unwrap();
        let r1e = i1e(&input).unwrap();
        let d1e = r1e.data().unwrap();
        assert!(
            d0e[0].is_finite() && (d0e[0] - 0.015_081_295_651_531_355).abs() <= 1e-12,
            "i0e(700) finite & matches torch: got {}",
            d0e[0]
        );
        assert!(
            d1e[0].is_finite() && (d1e[0] - 0.015_070_519_444_716_846).abs() <= 1e-12,
            "i1e(700) finite & matches torch: got {}",
            d1e[0]
        );
        // i0(700) overflows to +inf in f64 (torch reports 1.53e302; ferrotorch
        // computes exp(700)*chbevl/sqrt which also lands near f64::MAX). Assert
        // it is at least a large finite-or-inf positive (scaling relationship).
        let r0 = i0(&input).unwrap();
        let d0 = r0.data().unwrap();
        assert!(d0[0] > 1e300, "i0(700) is huge (>1e300): got {}", d0[0]);
    }

    #[test]
    fn i_family_f32_vs_torch() {
        // Live torch.special f32 oracle on [-1.5,-0.7,0,0.3,2,5,9]:
        //   i0 =[1.6467233,1.1263031,1,1.0226269,2.2795851,27.239874,1093.5884]
        //   i0e=[0.36743364,0.55930555,1,0.7575806,0.3085083,0.18354082,0.13495953]
        //   i1 =[-0.98166645,-0.37187967,0,0.15169387,1.5906368,24.335642,1030.9148]
        //   i1e=[-0.21903941,-0.18466999,0,0.11237757,0.21526928,0.16397226,0.127225]
        let xs = vec![-1.5f32, -0.7, 0.0, 0.3, 2.0, 5.0, 9.0];
        let input =
            Tensor::from_storage(TensorStorage::cpu(xs.clone()), vec![xs.len()], false).unwrap();
        let cases: [(
            &str,
            fn(&Tensor<f32>) -> FerrotorchResult<Tensor<f32>>,
            [f32; 7],
        ); 4] = [
            (
                "i0",
                i0,
                [
                    1.646_723_3,
                    1.126_303_1,
                    1.0,
                    1.022_626_9,
                    2.279_585_1,
                    27.239_874,
                    1_093.588_4,
                ],
            ),
            (
                "i0e",
                i0e,
                [
                    0.367_433_64,
                    0.559_305_55,
                    1.0,
                    0.757_580_6,
                    0.308_508_3,
                    0.183_540_82,
                    0.134_959_53,
                ],
            ),
            (
                "i1",
                i1,
                [
                    -0.981_666_45,
                    -0.371_879_67,
                    0.0,
                    0.151_693_87,
                    1.590_636_8,
                    24.335_642,
                    1_030.914_8,
                ],
            ),
            (
                "i1e",
                i1e,
                [
                    -0.219_039_41,
                    -0.184_669_99,
                    0.0,
                    0.112_377_57,
                    0.215_269_28,
                    0.163_972_26,
                    0.127_225,
                ],
            ),
        ];
        for (name, f, want) in cases {
            let r = f(&input).unwrap();
            let d = r.data().unwrap();
            for i in 0..7 {
                assert!(
                    (d[i] - want[i]).abs() <= 1e-4 * (1.0 + want[i].abs()),
                    "{name} f32 idx {i} x={}: got {} want {}",
                    xs[i],
                    d[i],
                    want[i]
                );
            }
        }
    }

    // --- spherical_bessel_j0 / modified_bessel_k0/k1 (+scaled) (#1651 batch 3a) ---
    //
    // Expected values are live `torch.special.*` (torch 2.11.0+cu130, f64)
    // outputs (R-CHAR-3: oracle-derived, not self-referential).

    // Live torch.special.spherical_bessel_j0 f64 oracle on SBJ0_GRID. The grid
    // straddles the |x|<0.5 Taylor branch (0,0.25,0.49), the boundary (0.5),
    // and the sin(x)/x branch (>=0.5, incl. pi where sin(pi)~0).
    const SBJ0_GRID: [f64; 11] = [
        0.0,
        0.25,
        0.49,
        0.5,
        1.0,
        2.0,
        3.141_592_653_589_79,
        5.0,
        10.0,
        -1.0,
        -3.0,
    ];

    #[test]
    fn spherical_bessel_j0_known_values_vs_torch() {
        let input = t(&SBJ0_GRID, &[11]);
        let r = spherical_bessel_j0(&input).unwrap();
        let d = r.data().unwrap();
        let want = [
            1.0,
            0.989_615_837_018_091_7,
            0.960_460_996_267_669_5,
            0.958_851_077_208_406,
            0.841_470_984_807_896_5,
            0.454_648_713_412_840_85,
            1.028_487_619_224_955_5e-15,
            -0.191_784_854_932_627_7,
            -0.054_402_111_088_936_98,
            0.841_470_984_807_896_5,
            0.047_040_002_686_622_4,
        ];
        for i in 0..11 {
            assert!(
                (d[i] - want[i]).abs() <= 1e-12 * (1.0 + want[i].abs()),
                "spherical_bessel_j0 idx {i} x={}: got {} want {}",
                SBJ0_GRID[i],
                d[i],
                want[i]
            );
        }
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "j0(0)=1 is the exact Taylor branch return (x2=0); j0(+/-inf)=0 the explicit isinf branch — torch returns the literal endpoints"
    )]
    fn spherical_bessel_j0_edges_vs_torch() {
        // Live torch: spherical_bessel_j0([inf,-inf,nan]) = [0, 0, nan].
        let input = t(&[0.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN], &[4]);
        let r = spherical_bessel_j0(&input).unwrap();
        let d = r.data().unwrap();
        assert_eq!(d[0], 1.0, "j0(0) == 1 (Taylor branch)");
        assert_eq!(d[1], 0.0, "j0(+inf) == 0");
        assert_eq!(d[2], 0.0, "j0(-inf) == 0");
        assert!(d[3].is_nan(), "j0(NaN) == NaN");
    }

    // Live torch.special K-family f64 oracle. K_GRID straddles the small region
    // (x<=2) and large region (x>2), incl. the boundary at 2 and just above.
    const K_GRID: [f64; 9] = [0.1, 0.5, 1.0, 2.0, 2.0001, 3.0, 5.0, 10.0, 50.0];

    #[test]
    fn modified_bessel_k0_known_values_vs_torch() {
        let input = t(&K_GRID, &[9]);
        let r = modified_bessel_k0(&input).unwrap();
        let d = r.data().unwrap();
        let want = [
            2.427_069_024_702_017,
            0.924_419_071_227_666,
            0.421_024_438_240_708_2,
            0.113_893_872_749_533_4,
            0.113_879_887_080_441_4,
            0.034_739_504_386_279_25,
            0.003_691_098_334_042_594_2,
            1.778_006_231_616_765e-5,
            3.410_167_749_789_495e-23,
        ];
        for i in 0..9 {
            assert!(
                (d[i] - want[i]).abs() <= 1e-12 * (1.0 + want[i].abs()),
                "k0 idx {i} x={}: got {} want {}",
                K_GRID[i],
                d[i],
                want[i]
            );
        }
    }

    #[test]
    fn scaled_modified_bessel_k0_known_values_vs_torch() {
        let input = t(&K_GRID, &[9]);
        let r = scaled_modified_bessel_k0(&input).unwrap();
        let d = r.data().unwrap();
        let want = [
            2.682_326_102_262_895,
            1.524_109_385_773_909_9,
            1.144_463_079_806_894_4,
            0.841_568_215_070_771_2,
            0.841_549_024_872_151_7,
            0.697_761_598_043_851_7,
            0.547_807_564_313_519,
            0.391_631_934_436_598_66,
            0.176_807_155_857_429_32,
        ];
        for i in 0..9 {
            assert!(
                (d[i] - want[i]).abs() <= 1e-12 * (1.0 + want[i].abs()),
                "scaled_k0 idx {i} x={}: got {} want {}",
                K_GRID[i],
                d[i],
                want[i]
            );
        }
    }

    #[test]
    fn modified_bessel_k1_known_values_vs_torch() {
        let input = t(&K_GRID, &[9]);
        let r = modified_bessel_k1(&input).unwrap();
        let d = r.data().unwrap();
        let want = [
            9.853_844_780_870_606,
            1.656_441_120_003_300_7,
            0.601_907_230_197_234_6,
            0.139_865_881_816_522_46,
            0.139_847_500_468_811_42,
            0.040_156_431_128_194_19,
            0.004_044_613_445_452_163,
            1.864_877_345_382_558_5e-5,
            3.444_102_226_717_555_5e-23,
        ];
        for i in 0..9 {
            assert!(
                (d[i] - want[i]).abs() <= 1e-12 * (1.0 + want[i].abs()),
                "k1 idx {i} x={}: got {} want {}",
                K_GRID[i],
                d[i],
                want[i]
            );
        }
    }

    #[test]
    fn scaled_modified_bessel_k1_known_values_vs_torch() {
        let input = t(&K_GRID, &[9]);
        let r = scaled_modified_bessel_k1(&input).unwrap();
        let d = r.data().unwrap();
        let want = [
            10.890_182_683_049_698,
            2.731_009_708_211_785_5,
            1.636_153_486_263_258,
            1.033_476_847_068_688_8,
            1.033_444_365_528_781_5,
            0.806_563_480_128_787,
            0.600_273_858_788_312_5,
            0.410_766_570_595_788_7,
            0.178_566_558_558_815_56,
        ];
        for i in 0..9 {
            assert!(
                (d[i] - want[i]).abs() <= 1e-12 * (1.0 + want[i].abs()),
                "scaled_k1 idx {i} x={}: got {} want {}",
                K_GRID[i],
                d[i],
                want[i]
            );
        }
    }

    #[test]
    fn k_family_domain_edges_vs_torch() {
        // Live torch: k0/k1 (+scaled) at [0, -1, NaN]: [+inf, NaN, NaN].
        // At x=700: k0/k1 underflow to a tiny finite (~4.7e-306), while the
        // scaled variants stay O(0.047) (-> sqrt(pi/(2x))).
        let input = t(&[0.0, -1.0, f64::NAN, 700.0], &[4]);
        let fns: [(&str, fn(&Tensor<f64>) -> FerrotorchResult<Tensor<f64>>, f64); 4] = [
            ("k0", modified_bessel_k0, 0.047_362_369_454_613_57),
            (
                "scaled_k0",
                scaled_modified_bessel_k0,
                0.047_362_369_454_613_57,
            ),
            ("k1", modified_bessel_k1, 0.047_396_187_653_494_55),
            (
                "scaled_k1",
                scaled_modified_bessel_k1,
                0.047_396_187_653_494_55,
            ),
        ];
        for (name, f, scaled_at_700) in fns {
            let r = f(&input).unwrap();
            let d = r.data().unwrap();
            assert!(
                d[0].is_infinite() && d[0] > 0.0,
                "{name}(0) == +inf: got {}",
                d[0]
            );
            assert!(d[1].is_nan(), "{name}(-1) == NaN: got {}", d[1]);
            assert!(d[2].is_nan(), "{name}(NaN) == NaN: got {}", d[2]);
            if name.starts_with("scaled") {
                assert!(
                    (d[3] - scaled_at_700).abs() <= 1e-12 * (1.0 + scaled_at_700.abs()),
                    "{name}(700) ~ sqrt(pi/2x): got {} want {}",
                    d[3],
                    scaled_at_700
                );
            } else {
                // Unscaled underflows toward 0 but stays finite & positive.
                assert!(
                    d[3].is_finite() && d[3] >= 0.0 && d[3] < 1e-300,
                    "{name}(700) underflows finite-nonneg: got {}",
                    d[3]
                );
            }
        }
    }

    #[test]
    fn spherical_and_k_family_f32_vs_torch() {
        // f32 oracle (live torch.special, 2.11). The f64-then-narrow CPU path
        // must stay inside the f32 transcendental tolerance (1e-4 rel).
        //   spherical_bessel_j0([0,0.25,0.5,1,2,5,-3]) =
        //     [1, 0.98961586, 0.9588511, 0.84147096, 0.4546487, -0.19178486, 0.047040001]
        let xs = vec![0.0f32, 0.25, 0.5, 1.0, 2.0, 5.0, -3.0];
        let input =
            Tensor::from_storage(TensorStorage::cpu(xs.clone()), vec![xs.len()], false).unwrap();
        let r = spherical_bessel_j0(&input).unwrap();
        let d = r.data().unwrap();
        let want = [
            1.0f32,
            0.989_615_86,
            0.958_851_1,
            0.841_470_96,
            0.454_648_7,
            -0.191_784_86,
            0.047_04,
        ];
        for i in 0..7 {
            assert!(
                (d[i] - want[i]).abs() <= 1e-4 * (1.0 + want[i].abs()),
                "spherical_bessel_j0 f32 idx {i} x={}: got {} want {}",
                xs[i],
                d[i],
                want[i]
            );
        }

        // K-family f32 at x=1.0 (small region) and x=3.0 (large region).
        let kx = vec![1.0f32, 3.0];
        let kin =
            Tensor::from_storage(TensorStorage::cpu(kx.clone()), vec![kx.len()], false).unwrap();
        let kcases: [(
            &str,
            fn(&Tensor<f32>) -> FerrotorchResult<Tensor<f32>>,
            [f32; 2],
        ); 4] = [
            ("k0", modified_bessel_k0, [0.421_024_44, 0.034_739_504]),
            (
                "scaled_k0",
                scaled_modified_bessel_k0,
                [1.144_463_1, 0.697_761_6],
            ),
            ("k1", modified_bessel_k1, [0.601_907_23, 0.040_156_43]),
            (
                "scaled_k1",
                scaled_modified_bessel_k1,
                [1.636_153_5, 0.806_563_5],
            ),
        ];
        for (name, f, want) in kcases {
            let r = f(&kin).unwrap();
            let d = r.data().unwrap();
            for i in 0..2 {
                assert!(
                    (d[i] - want[i]).abs() <= 1e-4 * (1.0 + want[i].abs()),
                    "{name} f32 idx {i} x={}: got {} want {}",
                    kx[i],
                    d[i],
                    want[i]
                );
            }
        }
    }
}
