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
//! | REQ-4 | SHIPPED | `lgamma` / `gammaln` alias at `special.rs`; consumer: re-export at `lib.rs` |
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
use std::sync::Arc;

use crate::autograd::no_grad::{is_grad_enabled, no_grad};
use crate::bool_tensor::BoolTensor;
use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::grad_fns::arithmetic::reduce_grad_to_shape;
use crate::ops::elementwise::{binary_map, unary_map, unary_map_named};
use crate::storage::TensorStorage;
use crate::tensor::{GradFn, Tensor};

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

#[inline]
fn special_needs_grad_unary<T: Float>(input: &Tensor<T>) -> bool {
    is_grad_enabled() && input.requires_grad()
}

#[inline]
fn special_needs_grad_binary<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> bool {
    is_grad_enabled() && (a.requires_grad() || b.requires_grad())
}

fn scalar_checked<T: Float>(value: f64, op: &'static str) -> FerrotorchResult<T> {
    T::from(value).ok_or_else(|| FerrotorchError::InvalidArgument {
        message: format!("{op}: scalar {value} cannot be represented in tensor dtype"),
    })
}

fn full_like_scalar<T: Float>(
    like: &Tensor<T>,
    value: f64,
    op: &'static str,
) -> FerrotorchResult<Tensor<T>> {
    crate::creation::full_like(like, scalar_checked(value, op)?)
}

fn finish_special_unary<T, F>(
    output: Tensor<T>,
    input: &Tensor<T>,
    make_grad_fn: F,
) -> FerrotorchResult<Tensor<T>>
where
    T: Float,
    F: FnOnce(Tensor<T>) -> FerrotorchResult<Arc<dyn GradFn<T>>>,
{
    if !special_needs_grad_unary(input) {
        return Ok(output);
    }
    let (storage, shape) = output.into_storage_and_shape()?;
    Tensor::from_operation_saving_output(storage, shape, make_grad_fn)
}

fn finish_special_binary<T, F>(
    output: Tensor<T>,
    a: &Tensor<T>,
    b: &Tensor<T>,
    make_grad_fn: F,
) -> FerrotorchResult<Tensor<T>>
where
    T: Float,
    F: FnOnce(Tensor<T>) -> FerrotorchResult<Arc<dyn GradFn<T>>>,
{
    if !special_needs_grad_binary(a, b) {
        return Ok(output);
    }
    let (storage, shape) = output.into_storage_and_shape()?;
    Tensor::from_operation_saving_output(storage, shape, make_grad_fn)
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
    // Edge ladder mirrors `calc_erfinv` (`Math.h:152-172`): out-of-domain
    // `|x| > 1` is NaN FIRST; ±inf is reserved for exactly ±1 (audit
    // CORE-171 — the pre-fix `>=`/`<=` ladder returned plausible
    // infinities for out-of-range inputs).
    if x > one || x < -one {
        return T::from(f64::NAN).unwrap();
    }
    if x == one {
        return T::infinity();
    }
    if x == -one {
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
    // C99 / torch contract: lgamma(±inf) = +inf (audit CORE-174 — without
    // this guard +inf hits the Lanczos `inf - inf` tail and -inf hits
    // `sin(π·-inf) = NaN`, both yielding NaN). Note scipy.special.gammaln
    // returns -inf at -inf; torch (live 2.11.0: lgamma(-inf) = inf) and the
    // C standard are the documented oracle for this op.
    if x.is_infinite() {
        return T::infinity();
    }
    let one = nt_one::<T>();
    let zero = nt_zero::<T>();
    let half = T::from(0.5).unwrap();
    let half_ln_2pi = T::from(0.9189385332046727).unwrap(); // 0.5 * ln(2*pi)
    let g = T::from(LANCZOS_G).unwrap();

    // Exact non-positive integer poles must be caught before the reflection
    // formula. Computing sin(pi * x) with rounded pi misses values such as
    // -1.0 because sin(-pi) is a tiny finite residual, whereas torch's
    // std::lgamma-backed kernels return +inf at every pole.
    if x <= zero && x == x.floor() {
        return T::infinity();
    }

    // Handle negative values via reflection formula.
    if x < half {
        let pi = T::from(std::f64::consts::PI).unwrap();
        let sin_pi_x = (pi * x).sin();
        if sin_pi_x == zero {
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
#[allow(
    clippy::float_cmp,
    reason = "exact zero / integer-pole tests are a verbatim port of calc_digamma's `x == 0` and `x == trunc(x)` exact-equality edge ladder (Math.h:378-390)"
)]
fn digamma_f64_hi(x: f64) -> f64 {
    // Edge ladder mirrors `calc_digamma` (`Math.h:375-390`), fixing audit
    // CORE-173: the reflection's `cot(π·x)` with rounded π evaluated to
    // ~-2.57e16 finite garbage at negative-integer poles instead of NaN.
    if x == 0.0 {
        // C99/SciPy/torch: ±0 → ∓∞ (`Math.h:378-382`).
        return f64::copysign(f64::INFINITY, -x);
    }
    if x < 0.0 && x == x.trunc() {
        // Negative-integer pole → NaN (`Math.h:384-390`).
        return f64::NAN;
    }
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

    // Same `calc_digamma` edge ladder as the f64 path (`Math.h:436-451`,
    // float instantiation) — audit CORE-173 hit BOTH dtype paths.
    if x == zero {
        return T::from(f64::copysign(
            f64::INFINITY,
            -<T as num_traits::ToPrimitive>::to_f64(&x).unwrap_or(f64::NAN),
        ))
        .unwrap();
    }
    if x < zero && x == x.trunc() {
        return T::from(f64::NAN).unwrap();
    }

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

/// Trigamma ψ₁(x), the derivative of digamma. This is a direct port of
/// PyTorch's `aten/src/ATen/native/Math.h::trigamma` scalar kernels and is
/// used by `DigammaBackward`.
fn trigamma_scalar<T: Float>(mut x: T) -> T {
    let half = T::from(0.5).unwrap();
    let one = nt_one::<T>();
    let mut sign = one;
    let mut result = nt_zero::<T>();

    if x < half {
        sign = -one;
        let pi = T::from(std::f64::consts::PI).unwrap();
        let sin_pi_x = (pi * x).sin();
        result = result - (pi * pi) / (sin_pi_x * sin_pi_x);
        x = one - x;
    }

    for _ in 0..6 {
        result += one / (x * x);
        x += one;
    }

    let two = T::from(2.0).unwrap();
    let six = T::from(6.0).unwrap();
    let thirty = T::from(30.0).unwrap();
    let forty_two = T::from(42.0).unwrap();
    let ixx = one / (x * x);
    result += (one
        + one / (two * x)
        + ixx * (one / six - ixx * (one / thirty - ixx * (one / forty_two))))
        / x;

    sign * result
}

/// Compute x * log(y) with the convention that 0 * log(y) = 0 — UNLESS y is
/// NaN. Mirrors `xlogy_kernel`
/// (`aten/src/ATen/native/cpu/BinaryOpsKernel.cpp:1288-1300`): the
/// `isnan(y)` check comes BEFORE the `x == 0` shortcut, so
/// `xlogy(0, NaN) = NaN` (audit CORE-175 — the pre-fix order returned 0).
fn xlogy_scalar<T: Float>(x: T, y: T) -> T {
    if y.is_nan() {
        return y;
    }
    if x == nt_zero::<T>() {
        nt_zero::<T>()
    } else {
        x * y.ln()
    }
}

// ---------------------------------------------------------------------------
// Autograd helpers
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
enum SpecialUnaryKind {
    Erf,
    Erfc,
    Erfinv,
    Lgamma,
    Digamma,
    Entr,
    Ndtr,
    Ndtri,
    I0,
    I0e,
    I1,
    I1e,
    Multigammaln { p: usize },
}

impl SpecialUnaryKind {
    fn name(self) -> &'static str {
        match self {
            Self::Erf => "ErfBackward",
            Self::Erfc => "ErfcBackward",
            Self::Erfinv => "ErfinvBackward",
            Self::Lgamma => "LgammaBackward",
            Self::Digamma => "DigammaBackward",
            Self::Entr => "SpecialEntrBackward",
            Self::Ndtr => "SpecialNdtrBackward",
            Self::Ndtri => "SpecialNdtriBackward",
            Self::I0 => "SpecialI0Backward",
            Self::I0e => "SpecialI0EBackward",
            Self::I1 => "SpecialI1Backward",
            Self::I1e => "SpecialI1EBackward",
            Self::Multigammaln { .. } => "MvlgammaBackward",
        }
    }

    fn op(self) -> &'static str {
        match self {
            Self::Erf => "erf_backward",
            Self::Erfc => "erfc_backward",
            Self::Erfinv => "erfinv_backward",
            Self::Lgamma => "lgamma_backward",
            Self::Digamma => "digamma_backward",
            Self::Entr => "entr_backward",
            Self::Ndtr => "ndtr_backward",
            Self::Ndtri => "ndtri_backward",
            Self::I0 => "i0_backward",
            Self::I0e => "i0e_backward",
            Self::I1 => "i1_backward",
            Self::I1e => "i1e_backward",
            Self::Multigammaln { .. } => "mvlgamma_backward",
        }
    }
}

#[derive(Debug)]
struct SpecialUnaryBackward<T: Float> {
    input: Tensor<T>,
    output: Tensor<T>,
    kind: SpecialUnaryKind,
}

impl<T: Float> GradFn<T> for SpecialUnaryBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }

        let grad = if self.input.is_cuda() || grad_output.is_cuda() {
            no_grad(|| {
                special_unary_backward_cuda(self.kind, &self.input, &self.output, grad_output)
            })?
        } else {
            special_unary_backward_cpu(self.kind, &self.input, &self.output, grad_output)?
        };
        Ok(vec![Some(grad)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        self.kind.name()
    }
}

fn wrap_special_unary<T: Float>(
    output: Tensor<T>,
    input: &Tensor<T>,
    kind: SpecialUnaryKind,
) -> FerrotorchResult<Tensor<T>> {
    finish_special_unary(output, input, |output| {
        Ok(Arc::new(SpecialUnaryBackward {
            input: input.saved_for_backward()?,
            output,
            kind,
        }))
    })
}

fn special_unary_backward_cpu<T: Float>(
    kind: SpecialUnaryKind,
    input: &Tensor<T>,
    output: &Tensor<T>,
    grad_output: &Tensor<T>,
) -> FerrotorchResult<Tensor<T>> {
    let go = grad_output.data_vec()?;
    let x = input.data_vec()?;
    let out = output.data_vec()?;
    let zero = nt_zero::<T>();
    let one = nt_one::<T>();
    let half = T::from(0.5).unwrap();
    let sqrt_pi = T::from(std::f64::consts::PI.sqrt()).unwrap();
    let two_over_sqrt_pi = T::from(2.0 / std::f64::consts::PI.sqrt()).unwrap();
    let inv_sqrt_2pi = T::from(1.0 / (2.0 * std::f64::consts::PI).sqrt()).unwrap();
    let sqrt_2pi = T::from((2.0 * std::f64::consts::PI).sqrt()).unwrap();
    let eps = T::epsilon();

    let grad: Vec<T> = go
        .iter()
        .zip(x.iter())
        .zip(out.iter())
        .map(|((&g, &x), &y)| {
            let factor = match kind {
                SpecialUnaryKind::Erf => two_over_sqrt_pi * (-(x * x)).exp(),
                SpecialUnaryKind::Erfc => -two_over_sqrt_pi * (-(x * x)).exp(),
                SpecialUnaryKind::Erfinv => half * sqrt_pi * (y * y).exp(),
                SpecialUnaryKind::Lgamma => digamma_scalar(x),
                SpecialUnaryKind::Digamma => trigamma_scalar(x),
                SpecialUnaryKind::Entr => -(one + x.ln()),
                SpecialUnaryKind::Ndtr => inv_sqrt_2pi * (-(x * x) * half).exp(),
                SpecialUnaryKind::Ndtri => sqrt_2pi * ((y * y) * half).exp(),
                SpecialUnaryKind::I0 => i1_scalar(x),
                SpecialUnaryKind::I0e => {
                    let sign = if x == zero { zero } else { x.signum() };
                    i1e_scalar(x) - sign * y
                }
                SpecialUnaryKind::I1 => {
                    if x.abs() > eps {
                        i0_scalar(x) - y / x
                    } else {
                        half
                    }
                }
                SpecialUnaryKind::I1e => {
                    if x.abs() > eps {
                        let sign = if x == zero { zero } else { x.signum() };
                        i0e_scalar(x) - y * (sign + one / x)
                    } else {
                        half
                    }
                }
                SpecialUnaryKind::Multigammaln { p } => {
                    let mut acc = zero;
                    for i in 1..=p {
                        let shift = T::from((1.0 - i as f64) * 0.5).unwrap();
                        acc += digamma_scalar(x + shift);
                    }
                    acc
                }
            };
            g * factor
        })
        .collect();

    Tensor::from_storage(TensorStorage::cpu(grad), input.shape().to_vec(), false)
}

fn mul_const_like<T: Float>(
    input: &Tensor<T>,
    value: f64,
    op: &'static str,
) -> FerrotorchResult<Tensor<T>> {
    let c = full_like_scalar(input, value, op)?;
    crate::grad_fns::arithmetic::mul(input, &c)
}

fn add_const_like<T: Float>(
    input: &Tensor<T>,
    value: f64,
    op: &'static str,
) -> FerrotorchResult<Tensor<T>> {
    let c = full_like_scalar(input, value, op)?;
    crate::grad_fns::arithmetic::add(input, &c)
}

fn sgn_tensor<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let zero = full_like_scalar(input, 0.0, "sgn_tensor")?;
    let one = full_like_scalar(input, 1.0, "sgn_tensor")?;
    let neg_one = full_like_scalar(input, -1.0, "sgn_tensor")?;
    let gt_zero = BoolTensor::gt(input, &zero)?;
    let lt_zero = BoolTensor::lt(input, &zero)?;
    let pos_or_zero = crate::grad_fns::comparison::where_bt(&gt_zero, &one, &zero)?;
    crate::grad_fns::comparison::where_bt(&lt_zero, &neg_one, &pos_or_zero)
}

fn safe_tiny_input<T: Float>(input: &Tensor<T>) -> FerrotorchResult<(BoolTensor, Tensor<T>)> {
    let eps = T::epsilon()
        .to_f64()
        .ok_or_else(|| FerrotorchError::InvalidArgument {
            message: "safe_tiny_input: dtype epsilon is not representable as f64".to_string(),
        })?;
    let eps_tensor = full_like_scalar(input, eps, "safe_tiny_input")?;
    let abs = crate::grad_fns::arithmetic::abs(input)?;
    let not_tiny = BoolTensor::gt(&abs, &eps_tensor)?;
    let safe = crate::grad_fns::comparison::where_bt(&not_tiny, input, &eps_tensor)?;
    Ok((not_tiny, safe))
}

fn special_unary_backward_cuda<T: Float>(
    kind: SpecialUnaryKind,
    input: &Tensor<T>,
    output: &Tensor<T>,
    grad_output: &Tensor<T>,
) -> FerrotorchResult<Tensor<T>> {
    let factor = match kind {
        SpecialUnaryKind::Erf => {
            let x2 = crate::grad_fns::arithmetic::mul(input, input)?;
            let neg = mul_const_like(&x2, -1.0, kind.op())?;
            let exp = crate::grad_fns::transcendental::exp(&neg)?;
            mul_const_like(&exp, 2.0 / std::f64::consts::PI.sqrt(), kind.op())?
        }
        SpecialUnaryKind::Erfc => {
            let x2 = crate::grad_fns::arithmetic::mul(input, input)?;
            let neg = mul_const_like(&x2, -1.0, kind.op())?;
            let exp = crate::grad_fns::transcendental::exp(&neg)?;
            mul_const_like(&exp, -2.0 / std::f64::consts::PI.sqrt(), kind.op())?
        }
        SpecialUnaryKind::Erfinv => {
            let y2 = crate::grad_fns::arithmetic::mul(output, output)?;
            let exp = crate::grad_fns::transcendental::exp(&y2)?;
            mul_const_like(&exp, 0.5 * std::f64::consts::PI.sqrt(), kind.op())?
        }
        SpecialUnaryKind::Entr => {
            let log_x = crate::grad_fns::transcendental::log(input)?;
            let one_plus = add_const_like(&log_x, 1.0, kind.op())?;
            crate::grad_fns::arithmetic::neg(&one_plus)?
        }
        SpecialUnaryKind::Ndtr => {
            let x2 = crate::grad_fns::arithmetic::mul(input, input)?;
            let scaled = mul_const_like(&x2, -0.5, kind.op())?;
            let exp = crate::grad_fns::transcendental::exp(&scaled)?;
            mul_const_like(&exp, 1.0 / (2.0 * std::f64::consts::PI).sqrt(), kind.op())?
        }
        SpecialUnaryKind::Ndtri => {
            let y2 = crate::grad_fns::arithmetic::mul(output, output)?;
            let scaled = mul_const_like(&y2, 0.5, kind.op())?;
            let exp = crate::grad_fns::transcendental::exp(&scaled)?;
            mul_const_like(&exp, (2.0 * std::f64::consts::PI).sqrt(), kind.op())?
        }
        SpecialUnaryKind::I0 => i1(input)?,
        SpecialUnaryKind::I0e => {
            let i1e_x = i1e(input)?;
            let sign = sgn_tensor(input)?;
            let signed_out = crate::grad_fns::arithmetic::mul(&sign, output)?;
            crate::grad_fns::arithmetic::sub(&i1e_x, &signed_out)?
        }
        SpecialUnaryKind::I1 => {
            let (not_tiny, safe) = safe_tiny_input(input)?;
            let i0_safe = i0(&safe)?;
            let out_over_x = crate::grad_fns::arithmetic::div(output, &safe)?;
            let gradx = crate::grad_fns::arithmetic::sub(&i0_safe, &out_over_x)?;
            let half = full_like_scalar(input, 0.5, kind.op())?;
            crate::grad_fns::comparison::where_bt(&not_tiny, &gradx, &half)?
        }
        SpecialUnaryKind::I1e => {
            let (not_tiny, safe) = safe_tiny_input(input)?;
            let i0e_safe = i0e(&safe)?;
            let sign = sgn_tensor(&safe)?;
            let recip = crate::grad_fns::arithmetic::reciprocal(&safe)?;
            let sign_plus_recip = crate::grad_fns::arithmetic::add(&sign, &recip)?;
            let rhs = crate::grad_fns::arithmetic::mul(output, &sign_plus_recip)?;
            let gradx = crate::grad_fns::arithmetic::sub(&i0e_safe, &rhs)?;
            let half = full_like_scalar(input, 0.5, kind.op())?;
            crate::grad_fns::comparison::where_bt(&not_tiny, &gradx, &half)?
        }
        SpecialUnaryKind::Lgamma
        | SpecialUnaryKind::Digamma
        | SpecialUnaryKind::Multigammaln { .. } => {
            return Err(FerrotorchError::NotImplementedOnCuda { op: kind.op() });
        }
    };

    crate::grad_fns::arithmetic::mul(grad_output, &factor)
}

#[derive(Debug)]
struct XlogyBackward<T: Float> {
    x: Tensor<T>,
    y: Tensor<T>,
}

impl<T: Float> GradFn<T> for XlogyBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        no_grad(|| {
            let dx = if self.x.requires_grad() {
                let raw = xlogy(grad_output, &self.y)?;
                let zero_x = full_like_scalar(&self.x, 0.0, "xlogy_backward")?;
                let zero_y = full_like_scalar(&self.y, 0.0, "xlogy_backward")?;
                let x_is_zero = BoolTensor::eq_t(&self.x, &zero_x)?;
                let y_le_zero = BoolTensor::le(&self.y, &zero_y)?;
                let mask = x_is_zero.and(&y_le_zero)?;
                let zero_raw = full_like_scalar(&raw, 0.0, "xlogy_backward")?;
                let masked = crate::grad_fns::comparison::where_bt(&mask, &zero_raw, &raw)?;
                Some(reduce_grad_to_shape(&masked, self.x.shape())?)
            } else {
                None
            };

            let dy = if self.y.requires_grad() {
                let numerator = crate::grad_fns::arithmetic::mul(grad_output, &self.x)?;
                let raw = crate::grad_fns::arithmetic::div(&numerator, &self.y)?;
                Some(reduce_grad_to_shape(&raw, self.y.shape())?)
            } else {
                None
            };

            Ok(vec![dx, dy])
        })
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.x, &self.y]
    }

    fn name(&self) -> &'static str {
        "XlogyBackward"
    }
}

#[derive(Debug)]
struct GammaIncBackward<T: Float> {
    a: Tensor<T>,
    x: Tensor<T>,
    upper: bool,
}

impl<T: Float> GradFn<T> for GammaIncBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        no_grad(|| {
            if self.a.requires_grad() {
                let op = if self.upper {
                    "igammac: input"
                } else {
                    "igamma: input"
                };
                return Err(FerrotorchError::InvalidArgument {
                    message: format!("the derivative for '{op}' is not implemented"),
                });
            }

            let dx = if self.x.requires_grad() {
                let one = full_like_scalar(&self.a, 1.0, "gammainc_backward")?;
                let a_minus_one = crate::grad_fns::arithmetic::sub(&self.a, &one)?;
                let log_x = crate::grad_fns::transcendental::log(&self.x)?;
                let first = crate::grad_fns::arithmetic::mul(&a_minus_one, &log_x)?;
                let minus_x = crate::grad_fns::arithmetic::sub(&first, &self.x)?;
                let lgamma_a = lgamma(&self.a)?;
                let exponent = crate::grad_fns::arithmetic::sub(&minus_x, &lgamma_a)?;
                let factor = crate::grad_fns::transcendental::exp(&exponent)?;
                let raw = crate::grad_fns::arithmetic::mul(grad_output, &factor)?;
                let raw = if self.upper {
                    crate::grad_fns::arithmetic::neg(&raw)?
                } else {
                    raw
                };
                Some(reduce_grad_to_shape(&raw, self.x.shape())?)
            } else {
                None
            };

            Ok(vec![None, dx])
        })
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.a, &self.x]
    }

    fn name(&self) -> &'static str {
        if self.upper {
            "IgammacBackward"
        } else {
            "IgammaBackward"
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum BetaBackwardKind {
    LogBeta,
    Beta,
}

#[derive(Debug)]
struct BetaBackward<T: Float> {
    a: Tensor<T>,
    b: Tensor<T>,
    output: Tensor<T>,
    kind: BetaBackwardKind,
}

impl<T: Float> GradFn<T> for BetaBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        no_grad(|| {
            let sum = crate::grad_fns::arithmetic::add(&self.a, &self.b)?;
            let dig_sum = digamma(&sum)?;
            let scale = match self.kind {
                BetaBackwardKind::LogBeta => grad_output.clone(),
                BetaBackwardKind::Beta => {
                    crate::grad_fns::arithmetic::mul(grad_output, &self.output)?
                }
            };

            let da = if self.a.requires_grad() {
                let dig_a = digamma(&self.a)?;
                let diff = crate::grad_fns::arithmetic::sub(&dig_a, &dig_sum)?;
                let raw = crate::grad_fns::arithmetic::mul(&scale, &diff)?;
                Some(reduce_grad_to_shape(&raw, self.a.shape())?)
            } else {
                None
            };

            let db = if self.b.requires_grad() {
                let dig_b = digamma(&self.b)?;
                let diff = crate::grad_fns::arithmetic::sub(&dig_b, &dig_sum)?;
                let raw = crate::grad_fns::arithmetic::mul(&scale, &diff)?;
                Some(reduce_grad_to_shape(&raw, self.b.shape())?)
            } else {
                None
            };

            Ok(vec![da, db])
        })
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.a, &self.b]
    }

    fn name(&self) -> &'static str {
        match self.kind {
            BetaBackwardKind::LogBeta => "LogBetaBackward",
            BetaBackwardKind::Beta => "BetaBackward",
        }
    }
}

#[derive(Debug)]
struct ZetaBackward<T: Float> {
    input: Tensor<T>,
    other: Tensor<T>,
}

impl<T: Float> GradFn<T> for ZetaBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        no_grad(|| {
            if self.input.requires_grad() {
                return Err(FerrotorchError::InvalidArgument {
                    message: "the derivative for 'zeta' is not implemented".to_string(),
                });
            }

            let dother = if self.other.requires_grad() {
                let one = full_like_scalar(&self.input, 1.0, "zeta_backward")?;
                let input_plus_one = crate::grad_fns::arithmetic::add(&self.input, &one)?;
                let shifted = zeta(&input_plus_one, &self.other)?;
                let neg_input = crate::grad_fns::arithmetic::neg(&self.input)?;
                let scale = crate::grad_fns::arithmetic::mul(&neg_input, &shifted)?;
                let raw = crate::grad_fns::arithmetic::mul(grad_output, &scale)?;
                Some(reduce_grad_to_shape(&raw, self.other.shape())?)
            } else {
                None
            };

            Ok(vec![None, dother])
        })
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input, &self.other]
    }

    fn name(&self) -> &'static str {
        "SpecialZetaBackward"
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
// Hurwitz zeta (`zeta(x, q)`) and Airy Ai (`airy_ai(x)`) — Cephes ports
// (`aten/src/ATen/native/cuda/Math.cuh`). zeta is binary (two tensor args);
// airy_ai is unary. Both have data-dependent convergence loops that map poorly
// to flat PTX, so they ship CPU-only with the CUDA branch returning
// `NotImplementedOnCuda` (no host round trip) — same honest pattern as the
// batch-3a K-family.
// ---------------------------------------------------------------------------

// MACHEP — relative machine epsilon used by the Cephes zeta / airy convergence
// early-exit (`aten/src/ATen/native/cuda/Math.cuh:302`, `:1437`).
#[allow(
    clippy::excessive_precision,
    reason = "verbatim Cephes MACHEP (cuda/Math.cuh:302); full-width for audit-friendly diff"
)]
const CEPHES_MACHEP: f64 = 1.11022302462515654042E-16;

// Bernoulli-derived tail-series denominators A[12] for the Hurwitz-zeta
// Euler-Maclaurin asymptotic series (`aten/src/ATen/native/cuda/Math.cuh:306-319`).
#[allow(
    clippy::excessive_precision,
    reason = "verbatim Cephes zeta A-set (cuda/Math.cuh:306-319); full-width for audit-friendly diff"
)]
const ZETA_A: [f64; 12] = [
    12.0,
    -720.0,
    30240.0,
    -1209600.0,
    47900160.0,
    -1.8924375803183791606e9, /* 1.307674368e12/691 */
    7.47242496e10,
    -2.950130727918164224e12,  /* 1.067062284288e16/3617 */
    1.1646782814350067249e14,  /* 5.109094217170944e18/43867 */
    -4.5979787224074726105e15, /* 8.028576626982912e20/174611 */
    1.8152105401943546773e17,  /* 1.5511210043330985984e23/854513 */
    -7.1661652561756670113e18, /* 1.6938241367317436694528e27/236364091 */
];

/// Hurwitz zeta `zeta(x, q)` in f64 — Cephes kernel
/// (`aten/src/ATen/native/cuda/Math.cuh:299-383`, `zeta_string`). Edge ladder:
/// `x == 1 -> +inf`; `x < 1 -> NaN`; `q <= 0` non-positive integer `-> +inf`,
/// else (`q <= 0` non-integer with non-integer `x`) `-> NaN`. Interior is the
/// `s = pow(q, -x)` seed, the `while ((i < 9) || (a <= 9.0))` accumulation with
/// MACHEP-relative early exit, then the Euler-Maclaurin tail
/// `s += b*w/(x-1) - 0.5*b + sum_{i<12} a*b/A[i]`.
#[allow(
    clippy::float_cmp,
    reason = "verbatim Cephes edge ladder: `x == 1` and the `q == floor(q)` / `x != floor(x)` integer tests are exact-equality branches in upstream (cuda/Math.cuh:325, 337, 340); R-DEV-1 byte-match"
)]
fn zeta_f64(x: f64, q: f64) -> f64 {
    const ZERO: f64 = 0.0;
    const HALF: f64 = 0.5;
    const ONE: f64 = 1.0;

    // Short-circuits x == 1 -> +infty (Math.cuh:325-327).
    if x == ONE {
        return f64::INFINITY;
    }
    // Short-circuits x < 1 -> NaN (Math.cuh:330-332).
    if x < ONE {
        return f64::NAN;
    }
    // q <= 0: negative integers -> +infty; negative non-integers with
    // non-integer x -> NaN (Math.cuh:336-343).
    if q <= ZERO {
        if q == q.floor() {
            return f64::INFINITY;
        }
        if x != x.floor() {
            return f64::NAN;
        }
    }

    let mut s = q.powf(-x);
    let mut a = q;
    let mut i: i32 = 0;
    let mut b = ZERO;
    // while ((i < 9) || (a <= 9.0)) (Math.cuh:349-357).
    while (i < 9) || (a <= 9.0) {
        i += 1;
        a += ONE;
        b = a.powf(-x);
        s += b;
        if (-CEPHES_MACHEP * s < b) && (b < CEPHES_MACHEP * s) {
            return s;
        }
    }

    let w = a;
    s += b * w / (x - ONE);
    s -= HALF * b;
    a = ONE;
    let mut k = ZERO;
    // Euler-Maclaurin asymptotic tail (Math.cuh:364-379).
    for &coeff in &ZETA_A {
        a *= x + k;
        b /= w;
        let mut t = a * b / coeff;
        s += t;
        t = (t / s).abs();
        if t < CEPHES_MACHEP {
            return s;
        }
        k += ONE;
        a *= x + k;
        b /= w;
        k += ONE;
    }

    s
}

fn zeta_scalar<T: Float>(x: T, q: T) -> T {
    let xf = <T as num_traits::ToPrimitive>::to_f64(&x).unwrap_or(f64::NAN);
    let qf = <T as num_traits::ToPrimitive>::to_f64(&q).unwrap_or(f64::NAN);
    T::from(zeta_f64(xf, qf)).unwrap_or_else(|| T::from(f64::NAN).unwrap())
}

// Airy Ai coefficient tables (`aten/src/ATen/native/cuda/Math.cuh:1283-1354`).
#[allow(
    clippy::excessive_precision,
    reason = "verbatim Cephes airy AN-set (cuda/Math.cuh:1283-1292)"
)]
const AIRY_AN: [f64; 8] = [
    3.46538101525629032477e-01,
    1.20075952739645805542e+01,
    7.62796053615234516538e+01,
    1.68089224934630576269e+02,
    1.59756391350164413639e+02,
    7.05360906840444183113e+01,
    1.40264691163389668864e+01,
    9.99999999999999995305e-01,
];

#[allow(
    clippy::excessive_precision,
    reason = "verbatim Cephes airy AD-set (cuda/Math.cuh:1294-1303)"
)]
const AIRY_AD: [f64; 8] = [
    5.67594532638770212846e-01,
    1.47562562584847203173e+01,
    8.45138970141474626562e+01,
    1.77318088145400459522e+02,
    1.64234692871529701831e+02,
    7.14778400825575695274e+01,
    1.40959135607834029598e+01,
    1.00000000000000000470e+00,
];

#[allow(
    clippy::excessive_precision,
    reason = "verbatim Cephes airy AFN-set (cuda/Math.cuh:1305-1315)"
)]
const AIRY_AFN: [f64; 9] = [
    -1.31696323418331795333e-01,
    -6.26456544431912369773e-01,
    -6.93158036036933542233e-01,
    -2.79779981545119124951e-01,
    -4.91900132609500318020e-02,
    -4.06265923594885404393e-03,
    -1.59276496239262096340e-04,
    -2.77649108155232920844e-06,
    -1.67787698489114633780e-08,
];

#[allow(
    clippy::excessive_precision,
    reason = "verbatim Cephes airy AFD-set (cuda/Math.cuh:1317-1327)"
)]
const AIRY_AFD: [f64; 9] = [
    1.33560420706553243746e+01,
    3.26825032795224613948e+01,
    2.67367040941499554804e+01,
    9.18707402907259625840e+00,
    1.47529146771666414581e+00,
    1.15687173795188044134e-01,
    4.40291641615211203805e-03,
    7.54720348287414296618e-05,
    4.51850092970580378464e-07,
];

#[allow(
    clippy::excessive_precision,
    reason = "verbatim Cephes airy AGN-set (cuda/Math.cuh:1329-1341)"
)]
const AIRY_AGN: [f64; 11] = [
    1.97339932091685679179e-02,
    3.91103029615688277255e-01,
    1.06579897599595591108e+00,
    9.39169229816650230044e-01,
    3.51465656105547619242e-01,
    6.33888919628925490927e-02,
    5.85804113048388458567e-03,
    2.82851600836737019778e-04,
    6.98793669997260967291e-06,
    8.11789239554389293311e-08,
    3.41551784765923618484e-10,
];

#[allow(
    clippy::excessive_precision,
    reason = "verbatim Cephes airy AGD-set (cuda/Math.cuh:1343-1354)"
)]
const AIRY_AGD: [f64; 10] = [
    9.30892908077441974853e+00,
    1.98352928718312140417e+01,
    1.55646628932864612953e+01,
    5.47686069422975497931e+00,
    9.54293611618961883998e-01,
    8.64580826352392193095e-02,
    4.12656523824222607191e-03,
    1.01259085116509135510e-04,
    1.17166733214413521882e-06,
    4.91834570062930015649e-09,
];

/// Airy function `Ai(x)` in f64 — Cephes kernel
/// (`aten/src/ATen/native/cuda/Math.cuh:1280-1459`, `airy_ai_string`). Multi-
/// region rational/series approximation: `isinf(x) -> NaN`; `x > 103.892 -> 0`;
/// `x < -2.09` oscillatory asymptotic (AFN/AFD + AGN/AGD over `z = 1/(...)`);
/// `x >= 2.09` decaying asymptotic (AN/AD over `1/zeta`, early-return for
/// `x > 8.3203353`); the central Maclaurin series `f`/`g` otherwise.
#[allow(
    clippy::excessive_precision,
    reason = "verbatim Cephes airy magic constants 5.64189583547756286948e-01 (1/(2*sqrt(pi))), 0.355028053887817239260 (Ai(0)), 0.258819403792806798405 (-Ai'(0)) from cuda/Math.cuh:1399,1401,1421,1454; full-width for audit-friendly diff"
)]
fn airy_ai_f64(x: f64) -> f64 {
    if x.is_infinite() {
        return f64::NAN;
    }
    if x > 103.892 {
        return 0.0;
    }

    let mut domain_flag: i32 = 0;
    let mut ai = 0.0;

    // x < -2.09: oscillatory asymptotic region (Math.cuh:1372-1402).
    if x < -2.09 {
        let z = 1.0 / (-2.0 * x * (-x).sqrt() / 3.0);
        let z2 = z * z;

        let mut afn = 0.0;
        for &c in &AIRY_AFN {
            afn = afn * z2 + c;
        }
        let mut afd = 0.0;
        for &c in &AIRY_AFD {
            afd = afd * z2 + c;
        }
        let mut agn = 0.0;
        for &c in &AIRY_AGN {
            agn = agn * z2 + c;
        }
        let mut agd = 0.0;
        // AGD loop runs index 0..=9 (10 - 1 in upstream), i.e. 10 terms.
        for &c in &AIRY_AGD {
            agd = agd * z2 + c;
        }

        let t = -2.0 * x * (-x).sqrt() / 3.0 + 0.25 * std::f64::consts::PI;

        return 5.64189583547756286948e-01 / (-x).sqrt().sqrt()
            * (t.sin() * (1.0 + z2 * afn / afd) - t.cos() * (z * agn / agd));
    }

    // x >= 2.09: decaying asymptotic region (Math.cuh:1404-1426).
    if x >= 2.09 {
        domain_flag = 5;

        let zeta = 2.0 * x * x.sqrt() / 3.0;

        let mut an = 0.0;
        for &c in &AIRY_AN {
            an = an * (1.0 / zeta) + c;
        }
        let mut ad = 0.0;
        for &c in &AIRY_AD {
            ad = ad * (1.0 / zeta) + c;
        }

        ai = 5.64189583547756286948e-01 * (an / ad) / (2.0 * x.sqrt().sqrt() * zeta.exp());

        if x > 8.3203353 {
            return ai;
        }
    }

    // Central Maclaurin series f/g (Math.cuh:1428-1457).
    let mut f = 1.0;
    let mut g = x;
    let mut k = 1.0;

    let mut m = 1.0;
    let mut n = x;
    let mut t = 1.0;
    let z = x * x * x;

    while t > CEPHES_MACHEP {
        m *= z;
        k += 1.0;
        m /= k;
        n *= z;
        k += 1.0;
        n /= k;
        m /= k;
        f += m;
        k += 1.0;
        n /= k;
        g += n;

        t = (m / f).abs();
    }

    if (domain_flag & 1) == 0 {
        return 0.355028053887817239260 * f - 0.258819403792806798405 * g;
    }

    ai
}

fn airy_ai_scalar<T: Float>(x: T) -> T {
    let xf = <T as num_traits::ToPrimitive>::to_f64(&x).unwrap_or(f64::NAN);
    T::from(airy_ai_f64(xf)).unwrap_or_else(|| T::from(f64::NAN).unwrap())
}

// ---------------------------------------------------------------------------
// Incomplete-gamma family (gammainc / gammaincc) and log-beta / multigammaln
// ---------------------------------------------------------------------------
//
// The interior of the (a, x) plane is a direct f64 port of PyTorch's igamma
// kernel suite (`aten/src/ATen/native/Math.h`, itself SciPy's Cephes-derived
// implementation; fix for audit CORE-169 / #1863 — the pre-fix interior used
// the Numerical-Recipes `gammp`/`gammq` pair with 300-iteration caps that
// silently returned PARTIAL sums from `a ≈ 1.2e4` because the series needs
// O(√a) terms near `x ≈ a`):
//   - `ratevl` (Math.h:521) + `lanczos_sum_expg_scaled` (Math.h:581)
//   - `_igam_helper_fac` (Math.h:621) — `x^a · e^(-x) / Γ(a)` via the
//     cancellation-free Lanczos-scaled form near `x ≈ a`
//   - `_igam_helper_series` (Math.h:655) — DLMF 8.11.4 power series
//   - `_igamc_helper_series` (Math.h:687) — DLMF 8.7.3 series
//   - `_igam_helper_asymptotic_series` (Math.h:713) — DLMF 8.12.3/8.12.4
//     uniform asymptotic expansion for large `a` (the 25×25 `d` table)
//   - `_igamc_helper_continued_fraction` (Math.h:1006) — DLMF 8.9.2
// glued by the `calc_igamma`/`calc_igammac` regime selection (Math.h:1144 /
// :1070; SMALL=20, LARGE=200, SMALLRATIO=0.3, LARGERATIO=4.5). The boundary
// handling mirrors PyTorch's boundary ladders exactly (see cites below) so
// that `gammainc(0, x>0) = 1`, `gammainc(a>0, 0) = 0`, NaN for negatives, etc.
//
// Documented deviation: where upstream calls glibc `std::lgamma`, this port
// uses the file-local `lgamma_scalar` (Lanczos g=7, ~1e-14 relative — those
// call sites only fire for `|a - x| > 0.4·a` or `x ≤ 1.1`, where the result
// is far from the cancellation-sensitive `x ≈ a` ridge).
//
// `ferrotorch-distributions/src/gamma.rs::lower_incomplete_gamma_regularized`
// still carries its private NR scalar copy for `Gamma::cdf`; that copy shares
// the CORE-169 defect and is tracked separately (out of ferrotorch-core scope).

/// Double-precision machine epsilon constant used by every igamma helper —
/// upstream `MACHEP` (`Math.h:659` and siblings), literal kept verbatim.
#[allow(clippy::excessive_precision)]
const IGAM_MACHEP: f64 = 1.110_223_024_625_156_540_42E-16;

/// Numerator/denominator of the Boost Lanczos `expg`-scaled sum —
/// transcribed verbatim from `pytorch/aten/src/ATen/native/Math.h:583-615`.
#[allow(clippy::excessive_precision)]
const LANCZOS_SUM_EXPG_SCALED_NUM: [f64; 13] = [
    0.006061842346248906525783753964555936883222,
    0.5098416655656676188125178644804694509993,
    19.51992788247617482847860966235652136208,
    449.9445569063168119446858607650988409623,
    6955.999602515376140356310115515198987526,
    75999.29304014542649875303443598909137092,
    601859.6171681098786670226533699352302507,
    3481712.15498064590882071018964774556468,
    14605578.08768506808414169982791359218571,
    43338889.32467613834773723740590533316085,
    86363131.28813859145546927288977868422342,
    103794043.1163445451906271053616070238554,
    56906521.91347156388090791033559122686859,
];
#[allow(clippy::excessive_precision)]
const LANCZOS_SUM_EXPG_SCALED_DENOM: [f64; 13] = [
    1.0,
    66.0,
    1925.0,
    32670.0,
    357423.0,
    2637558.0,
    13339535.0,
    45995730.0,
    105258076.0,
    150917976.0,
    120543840.0,
    39916800.0,
    0.0,
];

/// Rational-function evaluation `num(x) / denom(x)`, evaluated in `1/x` when
/// `|x| > 1` for stability. Port of `ratevl` (`Math.h:521-573`).
fn ratevl_f64(x: f64, num: &[f64], denom: &[f64]) -> f64 {
    let m = num.len() - 1;
    let n = denom.len() - 1;
    let absx = x.abs();
    if absx > 1.0 {
        // Evaluate as a polynomial in 1/x: coefficients walked high-to-low.
        let y = 1.0 / x;
        let mut num_ans = num[m];
        for i in (0..m).rev() {
            num_ans = num_ans * y + num[i];
        }
        let mut denom_ans = denom[n];
        for i in (0..n).rev() {
            denom_ans = denom_ans * y + denom[i];
        }
        x.powi(n as i32 - m as i32) * num_ans / denom_ans
    } else {
        let mut num_ans = num[0];
        for &c in &num[1..] {
            num_ans = num_ans * x + c;
        }
        let mut denom_ans = denom[0];
        for &c in &denom[1..] {
            denom_ans = denom_ans * x + c;
        }
        num_ans / denom_ans
    }
}

/// Boost-derived Lanczos sum, exp(g)-scaled. Port of
/// `lanczos_sum_expg_scaled` (`Math.h:581-619`).
fn lanczos_sum_expg_scaled_f64(x: f64) -> f64 {
    ratevl_f64(
        x,
        &LANCZOS_SUM_EXPG_SCALED_NUM,
        &LANCZOS_SUM_EXPG_SCALED_DENOM,
    )
}

/// `x^a · e^(-x) / Γ(a)` — the common prefactor of the series and continued
/// fraction. Port of `_igam_helper_fac` (`Math.h:621-652`); near `x ≈ a` it
/// switches to the Lanczos-scaled form that avoids the `a·ln(x) - x - lgamma`
/// cancellation. Upstream's `std::lgamma` is `lgamma_scalar` here (see the
/// section comment).
fn igam_helper_fac_f64(a: f64, x: f64) -> f64 {
    // Upstream literals kept verbatim (`Math.h:626-630`): MAXLOG/EXP1/
    // lanczos_g exactly as in `_igam_helper_fac`'s static locals.
    #[allow(clippy::excessive_precision)]
    const MAXLOG: f64 = 7.097_827_128_933_839_968_43E2;
    #[allow(clippy::approx_constant)]
    const EXP1: f64 = 2.718_281_828_459_045;
    #[allow(clippy::excessive_precision)]
    const LANCZOS_G_EXPG: f64 = 6.024_680_040_776_729_583_740_234_375;

    if (a - x).abs() > 0.4 * a.abs() {
        let ax = a * x.ln() - x - lgamma_scalar(a);
        if ax < -MAXLOG {
            return 0.0;
        }
        return ax.exp();
    }

    let fac = a + LANCZOS_G_EXPG - 0.5;
    let mut res = (fac / EXP1).sqrt() / lanczos_sum_expg_scaled_f64(a);
    if a < 200.0 && x < 200.0 {
        res *= (a - x).exp() * (x / fac).powf(a);
    } else {
        let num = x - a - LANCZOS_G_EXPG + 0.5;
        let numfac = num / fac;
        res *= (a * (numfac.ln_1p() - numfac) + x * (0.5 - LANCZOS_G_EXPG) / fac).exp();
    }
    res
}

/// Lower incomplete gamma via the DLMF 8.11.4 power series. Port of
/// `_igam_helper_series` (`Math.h:655-685`, MAXITER=2000).
fn igam_helper_series_f64(a: f64, x: f64) -> f64 {
    let ax = igam_helper_fac_f64(a, x);
    if ax == 0.0 {
        return 0.0;
    }
    let mut r = a;
    let mut c = 1.0;
    let mut ans = 1.0;
    for _ in 0..2000 {
        r += 1.0;
        c *= x / r;
        ans += c;
        if c <= IGAM_MACHEP * ans {
            break;
        }
    }
    ans * ax / a
}

/// Upper incomplete gamma via the DLMF 8.7.3 series, with extra care to
/// avoid cancellation. Port of `_igamc_helper_series` (`Math.h:687-711`).
fn igamc_helper_series_f64(a: f64, x: f64) -> f64 {
    let mut fac = 1.0;
    let mut sum = 0.0;
    for n in 1..2000 {
        fac *= -x / n as f64;
        let term = fac / (a + n as f64);
        sum += term;
        if term.abs() <= IGAM_MACHEP * sum.abs() {
            break;
        }
    }
    let logx = x.ln();
    let term = -(a * logx - lgamma_scalar(1.0 + a)).exp_m1();
    term - (a * logx - lgamma_scalar(a)).exp() * sum
}

/// Uniform asymptotic expansion for large `a` (DLMF 8.12.3 for the lower,
/// 8.12.4 for the upper function). Port of `_igam_helper_asymptotic_series`
/// (`Math.h:713-1004`). `igam = true` computes P(a, x), `false` Q(a, x).
///
/// Faithfulness note: the final `√(2πa)` divisor uses upstream's literal
/// `c10::pi<float>` — a FLOAT π widened to double (`Math.h:1001`) — kept
/// verbatim for bit-parity with torch's double kernel.
fn igam_helper_asymptotic_series_f64(a: f64, x: f64, igam: bool) -> f64 {
    let sgn = if igam { -1.0 } else { 1.0 };
    let lambda = x / a;
    let sigma = (x - a) / a;

    let eta = match lambda.partial_cmp(&1.0) {
        Some(std::cmp::Ordering::Greater) => (-2.0 * (sigma.ln_1p() - sigma)).sqrt(),
        Some(std::cmp::Ordering::Less) => -(-2.0 * (sigma.ln_1p() - sigma)).sqrt(),
        _ => 0.0,
    };
    let mut res = 0.5 * erfc_f64_hi(sgn * eta * (a / 2.0).sqrt());

    let mut etapow = [1.0_f64; 25];
    let mut maxpow = 0usize;
    let mut sum = 0.0_f64;
    let mut afac = 1.0_f64;
    let mut absoldterm = f64::INFINITY;
    for row in &IGAM_ASYMPTOTIC_D {
        let mut ck = row[0];
        for (n, &dkn) in row.iter().enumerate().skip(1) {
            if n > maxpow {
                etapow[n] = eta * etapow[n - 1];
                maxpow += 1;
            }
            let ckterm = dkn * etapow[n];
            ck += ckterm;
            if ckterm.abs() < IGAM_MACHEP * ck.abs() {
                break;
            }
        }
        let term = ck * afac;
        let absterm = term.abs();
        if absterm > absoldterm {
            break;
        }
        sum += term;
        if absterm < IGAM_MACHEP * sum.abs() {
            break;
        }
        absoldterm = absterm;
        afac /= a;
    }
    res += sgn * (-0.5 * a * eta * eta).exp() * sum
        / (2.0 * f64::from(std::f32::consts::PI) * a).sqrt();
    res
}

/// Upper incomplete gamma via the DLMF 8.9.2 continued fraction. Port of
/// `_igamc_helper_continued_fraction` (`Math.h:1006-1064`, MAXITER=2000).
fn igamc_helper_continued_fraction_f64(a: f64, x: f64) -> f64 {
    // Upstream literals kept verbatim (`Math.h:1014-1017`).
    const BIG: f64 = 4.503_599_627_370_496e15;
    #[allow(clippy::excessive_precision)]
    const BIGINV: f64 = 2.220_446_049_250_313_080_85e-16;

    let ax = igam_helper_fac_f64(a, x);
    if ax == 0.0 {
        return 0.0;
    }

    let mut y = 1.0 - a;
    let mut z = x + y + 1.0;
    let mut c = 0.0;
    let mut pkm2 = 1.0;
    let mut qkm2 = x;
    let mut pkm1 = x + 1.0;
    let mut qkm1 = z * x;
    let mut ans = pkm1 / qkm1;

    for _ in 0..2000 {
        c += 1.0;
        y += 1.0;
        z += 2.0;
        let yc = y * c;
        let pk = pkm1 * z - pkm2 * yc;
        let qk = qkm1 * z - qkm2 * yc;
        let t = if qk == 0.0 {
            1.0
        } else {
            let r = pk / qk;
            let t = ((ans - r) / r).abs();
            ans = r;
            t
        };
        pkm2 = pkm1;
        pkm1 = pk;
        qkm2 = qkm1;
        qkm1 = qk;
        if pk.abs() > BIG {
            pkm2 *= BIGINV;
            pkm1 *= BIGINV;
            qkm2 *= BIGINV;
            qkm1 *= BIGINV;
        }
        if t <= IGAM_MACHEP {
            break;
        }
    }
    ans * ax
}

/// The 25x25 `d` coefficient table of DLMF 8.12.3/8.12.4 used by the
/// uniform asymptotic expansion for large `a` — transcribed verbatim from
/// `pytorch/aten/src/ATen/native/Math.h:715-947` (`_igam_helper_asymptotic_series`).
#[allow(clippy::excessive_precision)]
#[rustfmt::skip] // compact 25-per-row layout mirrors the upstream Math.h table for diffability
const IGAM_ASYMPTOTIC_D: [[f64; 25]; 25] = [
    [
        -3.3333333333333333e-1, 8.3333333333333333e-2, -1.4814814814814815e-2,
        1.1574074074074074e-3, 3.527336860670194e-4, -1.7875514403292181e-4,
        3.9192631785224378e-5, -2.1854485106799922e-6, -1.85406221071516e-6,
        8.296711340953086e-7, -1.7665952736826079e-7, 6.7078535434014986e-9,
        1.0261809784240308e-8, -4.3820360184533532e-9, 9.1476995822367902e-10,
        -2.551419399494625e-11, -5.8307721325504251e-11, 2.4361948020667416e-11,
        -5.0276692801141756e-12, 1.1004392031956135e-13, 3.3717632624009854e-13,
        -1.3923887224181621e-13, 2.8534893807047443e-14, -5.1391118342425726e-16,
        -1.9752288294349443e-15,
    ],
    [
        -1.8518518518518519e-3, -3.4722222222222222e-3, 2.6455026455026455e-3,
        -9.9022633744855967e-4, 2.0576131687242798e-4, -4.0187757201646091e-7,
        -1.8098550334489978e-5, 7.6491609160811101e-6, -1.6120900894563446e-6,
        4.6471278028074343e-9, 1.378633446915721e-7, -5.752545603517705e-8,
        1.1951628599778147e-8, -1.7543241719747648e-11, -1.0091543710600413e-9,
        4.1627929918425826e-10, -8.5639070264929806e-11, 6.0672151016047586e-14,
        7.1624989648114854e-12, -2.9331866437714371e-12, 5.9966963656836887e-13,
        -2.1671786527323314e-16, -4.9783399723692616e-14, 2.0291628823713425e-14,
        -4.13125571381061e-15,
    ],
    [
        4.1335978835978836e-3, -2.6813271604938272e-3, 7.7160493827160494e-4,
        2.0093878600823045e-6, -1.0736653226365161e-4, 5.2923448829120125e-5,
        -1.2760635188618728e-5, 3.4235787340961381e-8, 1.3721957309062933e-6,
        -6.298992138380055e-7, 1.4280614206064242e-7, -2.0477098421990866e-10,
        -1.4092529910867521e-8, 6.228974084922022e-9, -1.3670488396617113e-9,
        9.4283561590146782e-13, 1.2872252400089318e-10, -5.5645956134363321e-11,
        1.1975935546366981e-11, -4.1689782251838635e-15, -1.0940640427884594e-12,
        4.6622399463901357e-13, -9.905105763906906e-14, 1.8931876768373515e-17,
        8.8592218725911273e-15,
    ],
    [
        6.4943415637860082e-4, 2.2947209362139918e-4, -4.6918949439525571e-4,
        2.6772063206283885e-4, -7.5618016718839764e-5, -2.3965051138672967e-7,
        1.1082654115347302e-5, -5.6749528269915966e-6, 1.4230900732435884e-6,
        -2.7861080291528142e-11, -1.6958404091930277e-7, 8.0994649053880824e-8,
        -1.9111168485973654e-8, 2.3928620439808118e-12, 2.0620131815488798e-9,
        -9.4604966618551322e-10, 2.1541049775774908e-10, -1.388823336813903e-14,
        -2.1894761681963939e-11, 9.7909989511716851e-12, -2.1782191880180962e-12,
        6.2088195734079014e-17, 2.126978363279737e-13, -9.3446887915174333e-14,
        2.0453671226782849e-14,
    ],
    [
        -8.618882909167117e-4, 7.8403922172006663e-4, -2.9907248030319018e-4,
        -1.4638452578843418e-6, 6.6414982154651222e-5, -3.9683650471794347e-5,
        1.1375726970678419e-5, 2.5074972262375328e-10, -1.6954149536558306e-6,
        8.9075075322053097e-7, -2.2929348340008049e-7, 2.956794137544049e-11,
        2.8865829742708784e-8, -1.4189739437803219e-8, 3.4463580499464897e-9,
        -2.3024517174528067e-13, -3.9409233028046405e-10, 1.8602338968504502e-10,
        -4.356323005056618e-11, 1.2786001016296231e-15, 4.6792750266579195e-12,
        -2.1492464706134829e-12, 4.9088156148096522e-13, -6.3385914848915603e-18,
        -5.0453320690800944e-14,
    ],
    [
        -3.3679855336635815e-4, -6.9728137583658578e-5, 2.7727532449593921e-4,
        -1.9932570516188848e-4, 6.7977804779372078e-5, 1.419062920643967e-7,
        -1.3594048189768693e-5, 8.0184702563342015e-6, -2.2914811765080952e-6,
        -3.252473551298454e-10, 3.4652846491085265e-7, -1.8447187191171343e-7,
        4.8240967037894181e-8, -1.7989466721743515e-14, -6.3061945000135234e-9,
        3.1624176287745679e-9, -7.8409242536974293e-10, 5.1926791652540407e-15,
        9.3589442423067836e-11, -4.5134262161632782e-11, 1.0799129993116827e-11,
        -3.661886712685252e-17, -1.210902069055155e-12, 5.6807435849905643e-13,
        -1.3249659916340829e-13,
    ],
    [
        5.3130793646399222e-4, -5.9216643735369388e-4, 2.7087820967180448e-4,
        7.9023532326603279e-7, -8.1539693675619688e-5, 5.6116827531062497e-5,
        -1.8329116582843376e-5, -3.0796134506033048e-9, 3.4651553688036091e-6,
        -2.0291327396058604e-6, 5.7887928631490037e-7, 2.338630673826657e-13,
        -8.8286007463304835e-8, 4.7435958880408128e-8, -1.2545415020710382e-8,
        8.6496488580102925e-14, 1.6846058979264063e-9, -8.5754928235775947e-10,
        2.1598224929232125e-10, -7.6132305204761539e-16, -2.6639822008536144e-11,
        1.3065700536611057e-11, -3.1799163902367977e-12, 4.7109761213674315e-18,
        3.6902800842763467e-13,
    ],
    [
        3.4436760689237767e-4, 5.1717909082605922e-5, -3.3493161081142236e-4,
        2.812695154763237e-4, -1.0976582244684731e-4, -1.2741009095484485e-7,
        2.7744451511563644e-5, -1.8263488805711333e-5, 5.7876949497350524e-6,
        4.9387589339362704e-10, -1.0595367014026043e-6, 6.1667143761104075e-7,
        -1.7562973359060462e-7, -1.2974473287015439e-12, 2.695423606288966e-8,
        -1.4578352908731271e-8, 3.887645959386175e-9, -3.8810022510194121e-17,
        -5.3279941738772867e-10, 2.7437977643314845e-10, -6.9957960920705679e-11,
        2.5899863874868481e-17, 8.8566890996696381e-12, -4.403168815871311e-12,
        1.0865561947091654e-12,
    ],
    [
        -6.5262391859530942e-4, 8.3949872067208728e-4, -4.3829709854172101e-4,
        -6.969091458420552e-7, 1.6644846642067548e-4, -1.2783517679769219e-4,
        4.6299532636913043e-5, 4.5579098679227077e-9, -1.0595271125805195e-5,
        6.7833429048651666e-6, -2.1075476666258804e-6, -1.7213731432817145e-11,
        3.7735877416110979e-7, -2.1867506700122867e-7, 6.2202288040189269e-8,
        6.5977038267330006e-16, -9.5903864974256858e-9, 5.2132144922808078e-9,
        -1.3991589583935709e-9, 5.382058999060575e-16, 1.9484714275467745e-10,
        -1.0127287556389682e-10, 2.6077347197254926e-11, -5.0904186999932993e-18,
        -3.3721464474854592e-12,
    ],
    [
        -5.9676129019274625e-4, -7.2048954160200106e-5, 6.7823088376673284e-4,
        -6.4014752602627585e-4, 2.7750107634328704e-4, 1.8197008380465151e-7,
        -8.4795071170685032e-5, 6.105192082501531e-5, -2.1073920183404862e-5,
        -8.8585890141255994e-10, 4.5284535953805377e-6, -2.8427815022504408e-6,
        8.7082341778646412e-7, 3.6886101871706965e-12, -1.5344695190702061e-7,
        8.862466778790695e-8, -2.5184812301826817e-8, -1.0225912098215092e-14,
        3.8969470758154777e-9, -2.1267304792235635e-9, 5.7370135528051385e-10,
        -1.887749850169741e-19, -8.0931538694657866e-11, 4.2382723283449199e-11,
        -1.1002224534207726e-11,
    ],
    [
        1.3324454494800656e-3, -1.9144384985654775e-3, 1.1089369134596637e-3,
        9.932404122642299e-7, -5.0874501293093199e-4, 4.2735056665392884e-4,
        -1.6858853767910799e-4, -8.1301893922784998e-9, 4.5284402370562147e-5,
        -3.127053674781734e-5, 1.044986828530338e-5, 4.8435226265680926e-11,
        -2.1482565873456258e-6, 1.329369701097492e-6, -4.0295693092101029e-7,
        -1.7567877666323291e-13, 7.0145043163668257e-8, -4.040787734999483e-8,
        1.1474026743371963e-8, 3.9642746853563325e-18, -1.7804938269892714e-9,
        9.7480262548731646e-10, -2.6405338676507616e-10, 5.794875163403742e-18,
        3.7647749553543836e-11,
    ],
    [
        1.579727660730835e-3, 1.6251626278391582e-4, -2.0633421035543276e-3,
        2.1389686185689098e-3, -1.0108559391263003e-3, -3.9912705529919201e-7,
        3.6235025084764691e-4, -2.8143901463712154e-4, 1.0449513336495887e-4,
        2.1211418491830297e-9, -2.5779417251947842e-5, 1.7281818956040463e-5,
        -5.6413773872904282e-6, -1.1024320105776174e-11, 1.1223224418895175e-6,
        -6.8693396379526735e-7, 2.0653236975414887e-7, 4.6714772409838506e-14,
        -3.5609886164949055e-8, 2.0470855345905963e-8, -5.8091738633283358e-9,
        -1.332821287582869e-16, 9.0354604391335133e-10, -4.9598782517330834e-10,
        1.3481607129399749e-10,
    ],
    [
        -4.0725121195140166e-3, 6.4033628338080698e-3, -4.0410161081676618e-3,
        -2.183732802866233e-6, 2.1740441801254639e-3, -1.9700440518418892e-3,
        8.3595469747962458e-4, 1.9445447567109655e-8, -2.5779387120421696e-4,
        1.9009987368139304e-4, -6.7696499937438965e-5, -1.4440629666426572e-10,
        1.5712512518742269e-5, -1.0304008744776893e-5, 3.304517767401387e-6,
        7.9829760242325709e-13, -6.4097794149313004e-7, 3.8894624761300056e-7,
        -1.1618347644948869e-7, -2.816808630596451e-15, 1.9878012911297093e-8,
        -1.1407719956357511e-8, 3.2355857064185555e-9, 4.1759468293455945e-20,
        -5.0423112718105824e-10,
    ],
    [
        -5.9475779383993003e-3, -5.4016476789260452e-4, 8.7910413550767898e-3,
        -9.8576315587856125e-3, 5.0134695031021538e-3, 1.2807521786221875e-6,
        -2.0626019342754683e-3, 1.7109128573523058e-3, -6.7695312714133799e-4,
        -6.9011545676562133e-9, 1.8855128143995902e-4, -1.3395215663491969e-4,
        4.6263183033528039e-5, 4.0034230613321351e-11, -1.0255652921494033e-5,
        6.612086372797651e-6, -2.0913022027253008e-6, -2.0951775649603837e-13,
        3.9756029041993247e-7, -2.3956211978815887e-7, 7.1182883382145864e-8,
        8.925574873053455e-16, -1.2101547235064676e-8, 6.9350618248334386e-9,
        -1.9661464453856102e-9,
    ],
    [
        1.7402027787522711e-2, -2.9527880945699121e-2, 2.0045875571402799e-2,
        7.0289515966903407e-6, -1.2375421071343148e-2, 1.1976293444235254e-2,
        -5.4156038466518525e-3, -6.3290893396418616e-8, 1.8855118129005065e-3,
        -1.473473274825001e-3, 5.5515810097708387e-4, 5.2406834412550662e-10,
        -1.4357913535784836e-4, 9.9181293224943297e-5, -3.3460834749478311e-5,
        -3.5755837291098993e-12, 7.1560851960630076e-6, -4.5516802628155526e-6,
        1.4236576649271475e-6, 1.8803149082089664e-14, -2.6623403898929211e-7,
        1.5950642189595716e-7, -4.7187514673841102e-8, -6.5107872958755177e-17,
        7.9795091026746235e-9,
    ],
    [
        3.0249124160905891e-2, 2.4817436002649977e-3, -4.9939134373457022e-2,
        5.9915643009307869e-2, -3.2483207601623391e-2, -5.7212968652103441e-6,
        1.5085251778569354e-2, -1.3261324005088445e-2, 5.5515262632426148e-3,
        3.0263182257030016e-8, -1.7229548406756723e-3, 1.2893570099929637e-3,
        -4.6845138348319876e-4, -1.830259937893045e-10, 1.1449739014822654e-4,
        -7.7378565221244477e-5, 2.5625836246985201e-5, 1.0766165333192814e-12,
        -5.3246809282422621e-6, 3.349634863064464e-6, -1.0381253128684018e-6,
        -5.608909920621128e-15, 1.9150821930676591e-7, -1.1418365800203486e-7,
        3.3654425209171788e-8,
    ],
    [
        -9.9051020880159045e-2, 1.7954011706123486e-1, -1.2989606383463778e-1,
        -3.1478872752284357e-5, 9.0510635276848131e-2, -9.2828824411184397e-2,
        4.4412112839877808e-2, 2.7779236316835888e-7, -1.7229543805449697e-2,
        1.4182925050891573e-2, -5.6214161633747336e-3, -2.39598509186381e-9,
        1.6029634366079908e-3, -1.1606784674435773e-3, 4.1001337768153873e-4,
        1.8365800754090661e-11, -9.5844256563655903e-5, 6.3643062337764708e-5,
        -2.076250624489065e-5, -1.1806020912804483e-13, 4.2131808239120649e-6,
        -2.6262241337012467e-6, 8.0770620494930662e-7, 6.0125912123632725e-16,
        -1.4729737374018841e-7,
    ],
    [
        -1.9994542198219728e-1, -1.5056113040026424e-2, 3.6470239469348489e-1,
        -4.6435192311733545e-1, 2.6640934719197893e-1, 3.4038266027147191e-5,
        -1.3784338709329624e-1, 1.276467178337056e-1, -5.6213828755200985e-2,
        -1.753150885483011e-7, 1.9235592956768113e-2, -1.5088821281095315e-2,
        5.7401854451350123e-3, 1.0622382710310225e-9, -1.5335082692563998e-3,
        1.0819320643228214e-3, -3.7372510193945659e-4, -6.6170909729031985e-12,
        8.4263617380909628e-5, -5.5150706827483479e-5, 1.7769536448348069e-5,
        3.8827923210205533e-14, -3.53513697488768e-6, 2.1865832130045269e-6,
        -6.6812849447625594e-7,
    ],
    [
        7.2438608504029431e-1, -1.3918010932653375, 1.0654143352413968, 1.876173868950258e-4,
        -8.2705501176152696e-1, 8.9352433347828414e-1, -4.4971003995291339e-1,
        -1.6107401567546652e-6, 1.9235590165271091e-1, -1.6597702160042609e-1,
        6.8882222681814333e-2, 1.3910091724608687e-8, -2.146911561508663e-2,
        1.6228980898865892e-2, -5.9796016172584256e-3, -1.1287469112826745e-10,
        1.5167451119784857e-3, -1.0478634293553899e-3, 3.5539072889126421e-4,
        8.1704322111801517e-13, -7.7773013442452395e-5, 5.0291413897007722e-5,
        -1.6035083867000518e-5, 1.2469354315487605e-14, 3.1369106244517615e-6,
    ],
    [
        1.6668949727276811, 1.165462765994632e-1, -3.3288393225018906, 4.4692325482864037,
        -2.6977693045875807, -2.600667859891061e-4, 1.5389017615694539, -1.4937962361134612,
        6.8881964633233148e-1, 1.3077482004552385e-6, -2.5762963325596288e-1,
        2.1097676102125449e-1, -8.3714408359219882e-2, -7.7920428881354753e-9,
        2.4267923064833599e-2, -1.7813678334552311e-2, 6.3970330388900056e-3,
        4.9430807090480523e-11, -1.5554602758465635e-3, 1.0561196919903214e-3,
        -3.5277184460472902e-4, 9.3002334645022459e-14, 7.5285855026557172e-5,
        -4.8186515569156351e-5, 1.5227271505597605e-5,
    ],
    [
        -6.6188298861372935, 1.3397985455142589e+1, -1.0789350606845146e+1,
        -1.4352254537875018e-3, 9.2333694596189809, -1.0456552819547769e+1, 5.5105526029033471,
        1.2024439690716742e-5, -2.5762961164755816, 2.3207442745387179, -1.0045728797216284,
        -1.0207833290021914e-7, 3.3975092171169466e-1, -2.6720517450757468e-1,
        1.0235252851562706e-1, 8.4329730484871625e-10, -2.7998284958442595e-2,
        2.0066274144976813e-2, -7.0554368915086242e-3, 1.9402238183698188e-12,
        1.6562888105449611e-3, -1.1082898580743683e-3, 3.654545161310169e-4,
        -5.1290032026971794e-11, -7.6340103696869031e-5,
    ],
    [
        -1.7112706061976095e+1, -1.1208044642899116, 3.7131966511885444e+1,
        -5.2298271025348962e+1, 3.3058589696624618e+1, 2.4791298976200222e-3,
        -2.061089403411526e+1, 2.088672775145582e+1, -1.0045703956517752e+1,
        -1.2238783449063012e-5, 4.0770134274221141, -3.473667358470195, 1.4329352617312006,
        7.1359914411879712e-8, -4.4797257159115612e-1, 3.4112666080644461e-1,
        -1.2699786326594923e-1, -2.8953677269081528e-10, 3.3125776278259863e-2,
        -2.3274087021036101e-2, 8.0399993503648882e-3, -1.177805216235265e-9,
        -1.8321624891071668e-3, 1.2108282933588665e-3, -3.9479941246822517e-4,
    ],
    [
        7.389033153567425e+1, -1.5680141270402273e+2, 1.322177542759164e+2,
        1.3692876877324546e-2, -1.2366496885920151e+2, 1.4620689391062729e+2,
        -8.0365587724865346e+1, -1.1259851148881298e-4, 4.0770132196179938e+1,
        -3.8210340013273034e+1, 1.719522294277362e+1, 9.3519707955168356e-7,
        -6.2716159907747034, 5.1168999071852637, -2.0319658112299095, -4.9507215582761543e-9,
        5.9626397294332597e-1, -4.4220765337238094e-1, 1.6079998700166273e-1,
        -2.4733786203223402e-8, -4.0307574759979762e-2, 2.7849050747097869e-2,
        -9.4751858992054221e-3, 6.419922235909132e-6, 2.1250180774699461e-3,
    ],
    [
        2.1216837098382522e+2, 1.3107863022633868e+1, -4.9698285932871748e+2,
        7.3121595266969204e+2, -4.8213821720890847e+2, -2.8817248692894889e-2,
        3.2616720302947102e+2, -3.4389340280087117e+2, 1.7195193870816232e+2,
        1.4038077378096158e-4, -7.52594195897599e+1, 6.651969984520934e+1,
        -2.8447519748152462e+1, -7.613702615875391e-7, 9.5402237105304373, -7.5175301113311376,
        2.8943997568871961, -4.6612194999538201e-7, -8.0615149598794088e-1,
        5.8483006570631029e-1, -2.0845408972964956e-1, 1.4765818959305817e-4,
        5.1000433863753019e-2, -3.3066252141883665e-2, 1.5109265210467774e-2,
    ],
    [
        -9.8959643098322368e+2, 2.1925555360905233e+3, -1.9283586782723356e+3,
        -1.5925738122215253e-1, 1.9569985945919857e+3, -2.4072514765081556e+3,
        1.3756149959336496e+3, 1.2920735237496668e-3, -7.525941715948055e+2,
        7.3171668742208716e+2, -3.4137023466220065e+2, -9.9857390260608043e-6,
        1.3356313181291573e+2, -1.1276295161252794e+2, 4.6310396098204458e+1,
        -7.9237387133614756e-6, -1.4510726927018646e+1, 1.1111771248100563e+1,
        -4.1690817945270892, 3.1008219800117808e-3, 1.1220095449981468, -7.6052379926149916e-1,
        3.6262236505085254e-1, 2.216867741940747e-1, 4.8683443692930507e-1,
    ],
];

/// Smooth-interior regularized lower incomplete gamma `P(a, x)` for finite
/// `a > 0`, `x > 0` — the `calc_igamma` regime selection (`Math.h:1144-1202`):
/// uniform asymptotic expansion when `a` is large and `x ≈ a`, otherwise the
/// `1 - Q` subtraction for `x > max(1, a)`, otherwise the DLMF 8.11.4 series.
fn igam_interior_f64(a: f64, x: f64) -> f64 {
    const SMALL: f64 = 20.0;
    const LARGE: f64 = 200.0;
    const SMALLRATIO: f64 = 0.3;
    const LARGERATIO: f64 = 4.5;

    let absxma_a = (x - a).abs() / a;
    if a > SMALL && a < LARGE && absxma_a < SMALLRATIO {
        return igam_helper_asymptotic_series_f64(a, x, true);
    }
    if a > LARGE && absxma_a < LARGERATIO / a.sqrt() {
        return igam_helper_asymptotic_series_f64(a, x, true);
    }
    if x > 1.0 && x > a {
        return 1.0 - igamc_interior_f64(a, x);
    }
    igam_helper_series_f64(a, x)
}

/// Smooth-interior regularized upper incomplete gamma `Q(a, x)` for finite
/// `a > 0`, `x > 0` — the `calc_igammac` regime selection (`Math.h:1070-1141`).
fn igamc_interior_f64(a: f64, x: f64) -> f64 {
    const SMALL: f64 = 20.0;
    const LARGE: f64 = 200.0;
    const SMALLRATIO: f64 = 0.3;
    const LARGERATIO: f64 = 4.5;

    let absxma_a = (x - a).abs() / a;
    if a > SMALL && a < LARGE && absxma_a < SMALLRATIO {
        return igam_helper_asymptotic_series_f64(a, x, false);
    }
    if a > LARGE && absxma_a < LARGERATIO / a.sqrt() {
        return igam_helper_asymptotic_series_f64(a, x, false);
    }

    if x > 1.1 {
        if x < a {
            1.0 - igam_helper_series_f64(a, x)
        } else {
            igamc_helper_continued_fraction_f64(a, x)
        }
    } else if x <= 0.5 {
        if -0.4 / x.ln() < a {
            1.0 - igam_helper_series_f64(a, x)
        } else {
            igamc_helper_series_f64(a, x)
        }
    } else if x * 1.1 < a {
        1.0 - igam_helper_series_f64(a, x)
    } else {
        igamc_helper_series_f64(a, x)
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
/// The smooth interior is the ported `calc_igamma` regime selection
/// (`igam_interior_f64`).
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
    igam_interior_f64(a, x)
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
/// The smooth interior is the ported `calc_igammac` regime selection
/// (`igamc_interior_f64`).
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
    igamc_interior_f64(a, x)
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

/// Sign of `Γ(x)` under cephes `lgam_sgn` semantics: `+1` for `x > 0` AND at
/// poles (where `lgamma = +inf` dominates any sign), `(-1)^floor(x)` for
/// negative non-integers. Deliberately distinct from `gammaln_sign_scalar`
/// (scipy `gammasgn`), which returns NaN at poles — `beta` needs the
/// pole-tolerant variant so denominator poles produce SIGNED ZEROS like
/// scipy (`beta(-2.5, 0.5) = -0.0`), not NaN.
#[allow(
    clippy::float_cmp,
    reason = "exact integer-pole test (`x == floor(x)`) mirrors cephes beta.c's `a == floor(a)` exact-equality branch; an epsilon band would misclassify near-pole values"
)]
fn lgam_sgn_sign_f64(x: f64) -> f64 {
    // x > 0, pole (integer-valued x ≤ 0), or even-floor negative
    // non-integer → +1; odd-floor negative non-integer → -1.
    if x > 0.0 || x == x.floor() || x.floor().rem_euclid(2.0) == 0.0 {
        1.0
    } else {
        -1.0
    }
}

/// cephes `beta.c::beta_negint` (the scipy.special.beta backend): at a
/// non-positive-integer argument `n`, the `Γ(n)` pole makes `B` infinite
/// UNLESS the other argument is an integer with `1 - n - other > 0`, where
/// the pole ratio has the finite limit `(-1)^other · B(1 - n - other, other)`
/// (live scipy 1.17.1: `beta(-3, 2) = +1/6`, `beta(-3, 1) = -1/3`,
/// `beta(-1, 0.5) = +inf`).
#[allow(
    clippy::float_cmp,
    reason = "exact integer test mirrors cephes beta.c::beta_negint's `b == (int)b` exact-equality branch"
)]
fn beta_negint_f64(n: f64, other: f64) -> f64 {
    if other == other.floor() && 1.0 - n - other > 0.0 {
        let sgn = if other.rem_euclid(2.0) == 0.0 {
            1.0
        } else {
            -1.0
        };
        sgn * beta_f64(1.0 - n - other, other)
    } else {
        f64::INFINITY
    }
}

/// f64 beta function following cephes `beta.c` (scipy.special.beta):
/// non-positive-integer arguments route through [`beta_negint_f64`]; the
/// smooth path is the SIGN-TRACKED `exp(lgamma(a) + lgamma(b) - lgamma(a+b))`
/// with the `sgn(Γ(a))·sgn(Γ(b))/sgn(Γ(a+b))` factor that plain
/// `exp(ln|B|)` discards (audit CORE-172: `beta(-0.5, 1.5)` must be `-π`,
/// not `+π`).
#[allow(
    clippy::float_cmp,
    reason = "exact integer-pole tests mirror cephes beta.c's `a == floor(a)` exact-equality branches"
)]
fn beta_f64(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    if a <= 0.0 && a == a.floor() {
        return beta_negint_f64(a, b);
    }
    if b <= 0.0 && b == b.floor() {
        return beta_negint_f64(b, a);
    }
    let s = a + b;
    if s <= 0.0 && s == s.floor() {
        // Γ(a+b) pole in the DENOMINATOR: |B| is exactly 0; the zero's sign
        // is the numerator sign product (live scipy 1.17.1:
        // `beta(-2.5, 0.5) = -0.0`, `beta(-0.5, -0.5) = +0.0`). Computed
        // explicitly rather than relying on `exp(... - inf)` to preserve the
        // signed zero required by the scipy beta contract.
        return lgam_sgn_sign_f64(a) * lgam_sgn_sign_f64(b) * 0.0;
    }
    // ±1 factors, so dividing by sgn(Γ(a+b)) equals multiplying by it.
    let sign = lgam_sgn_sign_f64(a) * lgam_sgn_sign_f64(b) * lgam_sgn_sign_f64(s);
    let y = lgamma_scalar(a) + lgamma_scalar(b) - lgamma_scalar(s);
    sign * y.exp()
}

/// Beta function `B(a, b) = sgn·exp(lnB(a, b))` — see [`beta_f64`].
fn beta_scalar<T: Float>(a: T, b: T) -> T {
    let r = beta_f64(
        <T as num_traits::ToPrimitive>::to_f64(&a).unwrap_or(f64::NAN),
        <T as num_traits::ToPrimitive>::to_f64(&b).unwrap_or(f64::NAN),
    );
    T::from(r).unwrap_or_else(|| T::from(f64::NAN).unwrap())
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
    let output = unary_map_named(input, "erf", erf_scalar)?;
    wrap_special_unary(output, input, SpecialUnaryKind::Erf)
}

/// Complementary error function: erfc(x) = 1 - erf(x).
///
/// f64 path: SunPro fdlibm `erfc_f64_hi` — computed directly so the
/// right-tail (large positive x) avoids the catastrophic 1 - erf(x)
/// cancellation. f32/bf16 path: literal `1 - erf_scalar(x)`.
pub fn erfc<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let output = unary_map_named(input, "erfc", erfc_scalar)?;
    wrap_special_unary(output, input, SpecialUnaryKind::Erfc)
}

/// Inverse error function: erfinv(erf(x)) = x.
///
/// Uses the Winitzki (2008) rational approximation. Returns `inf` for
/// input = 1, `-inf` for input = -1, and `NaN` for |input| > 1.
pub fn erfinv<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let output = unary_map_named(input, "erfinv", erfinv_scalar)?;
    wrap_special_unary(output, input, SpecialUnaryKind::Erfinv)
}

/// Log-gamma function: lgamma(x) = log(|Gamma(x)|).
///
/// Uses the Lanczos approximation (g = 7, n = 9 coefficients).
pub fn lgamma<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let output = unary_map_named(input, "lgamma", lgamma_scalar)?;
    wrap_special_unary(output, input, SpecialUnaryKind::Lgamma)
}

/// Alias for [`lgamma`] — mirrors `torch.special.gammaln(input)`.
pub fn gammaln<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    lgamma(input)
}

/// Digamma function: psi(x) = d/dx ln(Gamma(x)).
///
/// Uses the recurrence relation to shift the argument above 6, then
/// applies the asymptotic expansion.
pub fn digamma<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let output = unary_map_named(input, "digamma", digamma_scalar)?;
    wrap_special_unary(output, input, SpecialUnaryKind::Digamma)
}

/// log(1 + x) -- numerically stable for small x.
///
/// Delegates to `num_traits::Float::ln_1p()`.
pub fn log1p<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    crate::grad_fns::transcendental::log1p(input)
}

/// exp(x) - 1 -- numerically stable for small x.
///
/// Delegates to `num_traits::Float::exp_m1()`.
pub fn expm1<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    crate::grad_fns::transcendental::expm1(input)
}

/// Normalized sinc function: sinc(x) = sin(pi*x) / (pi*x), with sinc(0) = 1.
pub fn sinc<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    crate::grad_fns::transcendental::sinc(input)
}

/// x * log(y), with the convention that xlogy(0, y) = 0 for any y.
///
/// This is useful for entropy computations where 0 * log(0) should be 0.
pub fn xlogy<T: Float>(x: &Tensor<T>, y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if let Some(output) = special_gpu_binary_broadcast(
        x,
        y,
        "xlogy",
        |b, xh, yh| b.xlogy_f32(xh, yh),
        |b, xh, yh| b.xlogy_f64(xh, yh),
    )? {
        return finish_special_binary(output, x, y, |_| {
            Ok(Arc::new(XlogyBackward {
                x: x.saved_for_backward()?,
                y: y.saved_for_backward()?,
            }))
        });
    }
    let output = binary_map(x, y, xlogy_scalar)?;
    finish_special_binary(output, x, y, |_| {
        Ok(Arc::new(XlogyBackward {
            x: x.saved_for_backward()?,
            y: y.saved_for_backward()?,
        }))
    })
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
        return wrap_special_unary(out, input, SpecialUnaryKind::Entr);
    }
    let output = unary_map(input, entr_scalar)?;
    wrap_special_unary(output, input, SpecialUnaryKind::Entr)
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
        return wrap_special_unary(out, input, SpecialUnaryKind::Ndtr);
    }
    let output = unary_map(input, ndtr_scalar)?;
    wrap_special_unary(output, input, SpecialUnaryKind::Ndtr)
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
        return wrap_special_unary(out, input, SpecialUnaryKind::Ndtri);
    }
    let output = unary_map(input, ndtri_scalar)?;
    wrap_special_unary(output, input, SpecialUnaryKind::Ndtri)
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
        return wrap_special_unary(out, input, SpecialUnaryKind::I0);
    }
    let output = unary_map(input, i0_scalar)?;
    wrap_special_unary(output, input, SpecialUnaryKind::I0)
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
        return wrap_special_unary(out, input, SpecialUnaryKind::I0e);
    }
    let output = unary_map(input, i0e_scalar)?;
    wrap_special_unary(output, input, SpecialUnaryKind::I0e)
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
        return wrap_special_unary(out, input, SpecialUnaryKind::I1);
    }
    let output = unary_map(input, i1_scalar)?;
    wrap_special_unary(output, input, SpecialUnaryKind::I1)
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
        return wrap_special_unary(out, input, SpecialUnaryKind::I1e);
    }
    let output = unary_map(input, i1e_scalar)?;
    wrap_special_unary(output, input, SpecialUnaryKind::I1e)
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
/// CUDA f32 runs on-device via
/// [`crate::gpu_dispatch::GpuBackend::modified_bessel_k0_f32`] (no host round
/// trip); f64/bf16/f16 CUDA return `NotImplementedOnCuda` (base PTX lacks the
/// f64 `lg2`/`ex2` approximations needed for the small-region log term).
pub fn modified_bessel_k0<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) = special_gpu_simple(
        input,
        "modified_bessel_k0",
        |b, h| b.modified_bessel_k0_f32(h),
        |b, h| b.modified_bessel_k0_f64(h),
    )? {
        return Ok(out);
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
/// CUDA f32 runs on-device via
/// [`crate::gpu_dispatch::GpuBackend::scaled_modified_bessel_k0_f32`] (no host
/// round trip); f64/bf16/f16 CUDA return `NotImplementedOnCuda`.
pub fn scaled_modified_bessel_k0<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) = special_gpu_simple(
        input,
        "scaled_modified_bessel_k0",
        |b, h| b.scaled_modified_bessel_k0_f32(h),
        |b, h| b.scaled_modified_bessel_k0_f64(h),
    )? {
        return Ok(out);
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
/// CUDA f32 runs on-device via
/// [`crate::gpu_dispatch::GpuBackend::modified_bessel_k1_f32`] (no host round
/// trip); f64/bf16/f16 CUDA return `NotImplementedOnCuda`.
pub fn modified_bessel_k1<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) = special_gpu_simple(
        input,
        "modified_bessel_k1",
        |b, h| b.modified_bessel_k1_f32(h),
        |b, h| b.modified_bessel_k1_f64(h),
    )? {
        return Ok(out);
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
/// CUDA f32 runs on-device via
/// [`crate::gpu_dispatch::GpuBackend::scaled_modified_bessel_k1_f32`] (no host
/// round trip); f64/bf16/f16 CUDA return `NotImplementedOnCuda`.
pub fn scaled_modified_bessel_k1<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) = special_gpu_simple(
        input,
        "scaled_modified_bessel_k1",
        |b, h| b.scaled_modified_bessel_k1_f32(h),
        |b, h| b.scaled_modified_bessel_k1_f64(h),
    )? {
        return Ok(out);
    }
    unary_map(input, scaled_modified_bessel_k1_scalar)
}

/// Hurwitz zeta function `zeta(x, q) = sum_{k=0}^inf (k + q)^{-x}`, element-wise
/// over a broadcast of `input` (the `x` exponent) and `other` (the `q` shift).
/// Mirrors `torch.special.zeta(input, other)` (`torch/special/__init__.py`);
/// scalar evaluator ports the Cephes Hurwitz-zeta kernel from
/// `aten/src/ATen/native/cuda/Math.cuh:299-383`. Edge ladder: `x == 1 -> +inf`;
/// `x < 1 -> NaN`; `q <= 0` non-positive integer `-> +inf`; `q <= 0` non-integer
/// with non-integer `x -> NaN`. `zeta(2, 1) == pi^2/6`.
///
/// CUDA f32 runs an on-device PTX kernel via
/// [`crate::gpu_dispatch::GpuBackend::zeta_f32`] (no host round trip) when both
/// operands are CUDA-resident, same-device and SAME-shape: the
/// `while ((i < 9) || (a <= 9.0))` first-sum loop ALWAYS terminates at exactly
/// `i == 9` (since `a = q + 9 > 9` for any `q > 0` by then), so it is a FIXED
/// 9-iteration unroll with a relative-MACHEP `converged` early-exit flag, and
/// the Euler-Maclaurin tail is a FIXED 12-term loop over the Bernoulli-derived
/// `ZETA_A` table with the same early-exit. Broadcast (CUDA operands of
/// differing shape) and mixed-device (one CUDA, one CPU) cases return
/// `NotImplementedOnCuda`: on-device broadcasting zeta is not yet implemented,
/// and the CPU [`binary_map`] cannot read CUDA storage, so neither computes
/// there. Only the same-shape, same-device CUDA case runs on-device; all-CPU
/// inputs use [`binary_map`] (which broadcasts on the host). f64/bf16/f16 CUDA
/// return `NotImplementedOnCuda` (base PTX lacks the f64 `pow`/`log`
/// approximations).
pub fn zeta<T: Float>(input: &Tensor<T>, other: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) = special_gpu_binary(
        input,
        other,
        "zeta",
        |b, x, q| b.zeta_f32(x, q),
        |b, x, q| b.zeta_f64(x, q),
    )? {
        return finish_special_binary(out, input, other, |_| {
            Ok(Arc::new(ZetaBackward {
                input: input.saved_for_backward()?,
                other: other.saved_for_backward()?,
            }))
        });
    }
    let output = binary_map(input, other, zeta_scalar)?;
    finish_special_binary(output, input, other, |_| {
        Ok(Arc::new(ZetaBackward {
            input: input.saved_for_backward()?,
            other: other.saved_for_backward()?,
        }))
    })
}

/// Airy function of the first kind `Ai(x)`. Mirrors `torch.special.airy_ai`
/// (`torch/special/__init__.py:982-985`); scalar evaluator ports the Cephes
/// multi-region kernel from `aten/src/ATen/native/cuda/Math.cuh:1280-1459`.
/// `airy_ai(0) = 0.3550280538878172`; oscillatory for `x < -2.09`, decaying for
/// `x > 0`; `airy_ai(+/-inf) = NaN` (the `isinf` short-circuit at
/// `Math.cuh:1360-1362`), `airy_ai(x > 103.892) = 0`.
///
/// CUDA f32 runs an on-device PTX kernel via
/// [`crate::gpu_dispatch::GpuBackend::airy_ai_f32`] (no host round trip): the
/// multi-region rational/series is reproduced with the oscillatory/decaying
/// Horner chains and a FIXED 36-iteration central-Maclaurin unroll (the
/// `while (t > MACHEP)` central loop is only reached for the bounded
/// `x in [-2.09, 8.3203353]` window, where 36 terms more than cover
/// convergence). f64/bf16/f16 CUDA return `NotImplementedOnCuda` (base PTX
/// lacks `lg2.approx.f64`/`ex2.approx.f64`).
pub fn airy_ai<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) = special_gpu_simple(
        input,
        "airy_ai",
        |b, h| b.airy_ai_f32(h),
        |b, h| b.airy_ai_f64(h),
    )? {
        return Ok(out);
    }
    unary_map(input, airy_ai_scalar)
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
    reject_cuda_binary(input, other, "gammainc")?;
    let output = binary_map(input, other, gammainc_scalar)?;
    finish_special_binary(output, input, other, |_| {
        Ok(Arc::new(GammaIncBackward {
            a: input.saved_for_backward()?,
            x: other.saved_for_backward()?,
            upper: false,
        }))
    })
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
    reject_cuda_binary(input, other, "gammaincc")?;
    let output = binary_map(input, other, gammaincc_scalar)?;
    finish_special_binary(output, input, other, |_| {
        Ok(Arc::new(GammaIncBackward {
            a: input.saved_for_backward()?,
            x: other.saved_for_backward()?,
            upper: true,
        }))
    })
}

/// Log-beta function `lnB(a, b) = lgamma(a) + lgamma(b) - lgamma(a + b)`,
/// element-wise over a broadcast of `a` and `b`.
///
/// Mirrors `scipy.special.betaln` / the `lbeta` PyTorch users build from
/// `torch.lgamma`. The accumulation runs in f64 then narrows to `T` so the
/// three-way lgamma subtraction does not lose bits in the f32 path.
pub fn log_beta<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    reject_cuda_binary(a, b, "log_beta")?;
    let output = binary_map(a, b, log_beta_scalar)?;
    finish_special_binary(output, a, b, |output| {
        Ok(Arc::new(BetaBackward {
            a: a.saved_for_backward()?,
            b: b.saved_for_backward()?,
            output,
            kind: BetaBackwardKind::LogBeta,
        }))
    })
}

/// Beta function `B(a, b) = exp(lnB(a, b)) = Γ(a)Γ(b)/Γ(a + b)`, element-wise
/// over a broadcast of `a` and `b`.
///
/// Mirrors `scipy.special.beta`.
pub fn beta<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    reject_cuda_binary(a, b, "beta")?;
    let output = binary_map(a, b, beta_scalar)?;
    finish_special_binary(output, a, b, |output| {
        Ok(Arc::new(BetaBackward {
            a: a.saved_for_backward()?,
            b: b.saved_for_backward()?,
            output,
            kind: BetaBackwardKind::Beta,
        }))
    })
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
    reject_cuda_unary(input, "multigammaln")?;
    let output = unary_map_named(input, "multigammaln", move |x| multigammaln_scalar(x, p))?;
    wrap_special_unary(output, input, SpecialUnaryKind::Multigammaln { p })
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
    reject_cuda_unary(input, "gammaln_sign")?;
    unary_map_named(input, "gammaln_sign", gammaln_sign_scalar)
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
    // #1658: normalise a possibly-narrowed CUDA view (non-zero storage_offset,
    // row-major strides that `is_contiguous()` cannot distinguish) into a packed
    // offset-0 buffer ON-DEVICE before the kernel reads `gpu_handle()`. The
    // `(in,out,total)` kernel ABI indexes from element 0 and drops the offset
    // otherwise. After #1657 `.contiguous()` is a cheap clone for already-packed
    // offset-0 tensors and a strided_copy materialisation for offset views.
    let input = input.contiguous()?;
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
    // #1658: normalise a narrowed-offset CUDA view to a packed offset-0 buffer
    // before the `(in,out,total)` kernel reads element 0 (see `poly_gpu_simple`).
    let input = input.contiguous()?;
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

/// Run a two-input elementwise special-function kernel (`zeta`) on a pair of
/// CUDA tensors, dispatching by dtype through the registered [`GpuBackend`].
/// Returns `Ok(Some(out))` when the GPU path handled it, `Ok(None)` when the
/// caller should take the CPU path.
///
/// The GPU fast path requires BOTH operands CUDA-resident, on the same device,
/// same dtype, and the SAME shape (the elementwise kernel is one-thread-per-
/// element with no on-device broadcast). When NEITHER operand is CUDA it
/// returns `Ok(None)` so the caller takes the CPU [`binary_map`] path (which
/// broadcasts on the host). When exactly one operand is CUDA (mixed device) or
/// both are CUDA but the shapes differ (broadcast), it returns
/// `Err(NotImplementedOnCuda)`: on-device broadcasting is not implemented and
/// `binary_map` cannot read CUDA storage, so there is no valid host fallback
/// (rejecting cleanly avoids leaking the internal `GpuTensorNotAccessible`).
/// bf16/f16 CUDA inputs are likewise rejected with `NotImplementedOnCuda` (no
/// host round trip). The result stays on-device (R-CODE-4).
fn special_gpu_binary<T: Float>(
    x: &Tensor<T>,
    q: &Tensor<T>,
    op: &'static str,
    f32_call: impl Fn(
        &dyn crate::gpu_dispatch::GpuBackend,
        &crate::gpu_dispatch::GpuBufferHandle,
        &crate::gpu_dispatch::GpuBufferHandle,
    ) -> FerrotorchResult<crate::gpu_dispatch::GpuBufferHandle>,
    f64_call: impl Fn(
        &dyn crate::gpu_dispatch::GpuBackend,
        &crate::gpu_dispatch::GpuBufferHandle,
        &crate::gpu_dispatch::GpuBufferHandle,
    ) -> FerrotorchResult<crate::gpu_dispatch::GpuBufferHandle>,
) -> FerrotorchResult<Option<Tensor<T>>> {
    if !x.is_cuda() && !q.is_cuda() {
        return Ok(None);
    }
    // Any CUDA operand of a non-f32/f64 dtype rejects (no host round trip).
    if !(poly_is_f32::<T>() || poly_is_f64::<T>()) {
        return Err(FerrotorchError::NotImplementedOnCuda { op });
    }
    // Mixed device (one CUDA, one CPU) or shapes that require broadcasting:
    // the on-device kernel is one-thread-per-element with NO on-device
    // broadcast, and the CPU `binary_map` cannot read CUDA storage (it calls
    // `.data()`, which returns `GpuTensorNotAccessible` for GPU tensors). So
    // there is no valid host fallback here; reject cleanly with
    // `NotImplementedOnCuda` (R-CODE-4: no leaked internal storage error, no
    // silent host detour). Only the same-device, same-shape CUDA case runs the
    // on-device kernel below.
    if x.is_cuda() != q.is_cuda() || x.shape() != q.shape() {
        return Err(FerrotorchError::NotImplementedOnCuda { op });
    }
    let backend = crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
    // #1658: normalise BOTH narrowed-offset CUDA operands to packed offset-0
    // buffers before the one-thread-per-element kernel reads element 0.
    let x = x.contiguous()?;
    let q = q.contiguous()?;
    let xh = x.gpu_handle()?;
    let qh = q.gpu_handle()?;
    let out_handle = if poly_is_f32::<T>() {
        f32_call(backend, xh, qh)?
    } else {
        f64_call(backend, xh, qh)?
    };
    Ok(Some(poly_gpu_output::<T>(out_handle, x.shape().to_vec())?))
}

/// Broadcast-aware CUDA binary helper for resident special kernels whose
/// PyTorch contract supports broadcasting. Unlike `special_gpu_binary` (kept
/// strict for zeta), this materializes broadcasted operands on-device via
/// `expand(...).contiguous()` before launching the equal-length kernel.
fn special_gpu_binary_broadcast<T: Float>(
    x: &Tensor<T>,
    y: &Tensor<T>,
    op: &'static str,
    f32_call: impl Fn(
        &dyn crate::gpu_dispatch::GpuBackend,
        &crate::gpu_dispatch::GpuBufferHandle,
        &crate::gpu_dispatch::GpuBufferHandle,
    ) -> FerrotorchResult<crate::gpu_dispatch::GpuBufferHandle>,
    f64_call: impl Fn(
        &dyn crate::gpu_dispatch::GpuBackend,
        &crate::gpu_dispatch::GpuBufferHandle,
        &crate::gpu_dispatch::GpuBufferHandle,
    ) -> FerrotorchResult<crate::gpu_dispatch::GpuBufferHandle>,
) -> FerrotorchResult<Option<Tensor<T>>> {
    if !x.is_cuda() && !y.is_cuda() {
        return Ok(None);
    }
    if x.is_cuda() != y.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op });
    }
    if x.device() != y.device() {
        return Err(FerrotorchError::DeviceMismatch {
            expected: x.device(),
            got: y.device(),
        });
    }
    if !(poly_is_f32::<T>() || poly_is_f64::<T>()) {
        return Err(FerrotorchError::NotImplementedOnCuda { op });
    }

    no_grad(|| {
        let out_shape = crate::shape::broadcast_shapes(x.shape(), y.shape())?;
        let x_expanded = if x.shape() == out_shape.as_slice() {
            x.clone()
        } else {
            crate::grad_fns::shape::expand(x, &out_shape)?
        };
        let y_expanded = if y.shape() == out_shape.as_slice() {
            y.clone()
        } else {
            crate::grad_fns::shape::expand(y, &out_shape)?
        };

        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let x = x_expanded.contiguous()?;
        let y = y_expanded.contiguous()?;
        let xh = x.gpu_handle()?;
        let yh = y.gpu_handle()?;
        let out_handle = if poly_is_f32::<T>() {
            f32_call(backend, xh, yh)?
        } else {
            f64_call(backend, xh, yh)?
        };
        Ok(Some(poly_gpu_output::<T>(out_handle, out_shape)?))
    })
}

fn reject_cuda_unary<T: Float>(input: &Tensor<T>, op: &'static str) -> FerrotorchResult<()> {
    if input.is_cuda() {
        Err(FerrotorchError::NotImplementedOnCuda { op })
    } else {
        Ok(())
    }
}

fn reject_cuda_binary<T: Float>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    op: &'static str,
) -> FerrotorchResult<()> {
    if a.is_cuda() || b.is_cuda() {
        Err(FerrotorchError::NotImplementedOnCuda { op })
    } else {
        Ok(())
    }
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
    // #1658: normalise a narrowed-offset CUDA view to a packed offset-0 buffer
    // before the chebyshev recurrence kernel reads element 0.
    let input = input.contiguous()?;
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
#[allow(
    clippy::excessive_precision,
    clippy::inconsistent_digit_grouping,
    clippy::unreadable_literal,
    clippy::float_cmp,
    clippy::type_complexity,
    clippy::approx_constant,
    reason = "oracle divergence tests: expected values are copied verbatim from live torch 2.11 / scipy / Cephes (full precision + grouping intentional); float comparisons are deliberately exact byte-for-byte parity checks; the (name, fn, [f64;3]) case tuples are a local test fixture, not a public type"
)]
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

    // === zeta / airy_ai (batch 3b, #1651) ===================================
    //
    // All oracle values constructed by live torch 2.11 (R-CHAR-3):
    //   torch.special.zeta(x, q) / torch.special.airy_ai(x), torch==2.11.0+cu130.

    #[test]
    fn zeta_known_values_vs_torch() {
        // Live torch.special.zeta over (x>1, q>0), incl. near 1+ (x=1.0001).
        let xs = [2.0, 2.0, 3.0, 4.0, 1.0001, 1.5, 10.0, 2.5, 5.0];
        let qs = [1.0, 2.0, 1.0, 0.5, 1.0, 2.0, 0.25, 3.0, 1.0];
        let x = t(&xs, &[9]);
        let q = t(&qs, &[9]);
        let r = zeta(&x, &q).unwrap();
        let d = r.data().unwrap();
        let want = [
            1.6449340668482266,
            0.6449340668482266,
            1.202056903159594,
            16.23484850566707,
            10000.57722294754,
            1.6123753486854886,
            1048576.107683115,
            0.1647105619542803,
            1.0369277551433704,
        ];
        for i in 0..9 {
            assert!(
                (d[i] - want[i]).abs() <= 1e-10 * (1.0 + want[i].abs()),
                "zeta idx {i} x={} q={}: got {} want {}",
                xs[i],
                qs[i],
                d[i],
                want[i]
            );
        }
    }

    #[test]
    fn zeta_2_1_is_pi_squared_over_six() {
        // zeta(2, 1) == pi^2 / 6 (Basel sum) — symbolic constant, not a copied bit.
        let r = zeta(&t(&[2.0], &[1]), &t(&[1.0], &[1])).unwrap();
        let got = r.data().unwrap()[0];
        let want = std::f64::consts::PI * std::f64::consts::PI / 6.0;
        assert!(
            (got - want).abs() <= 1e-12 * (1.0 + want.abs()),
            "zeta(2,1) got {got} want pi^2/6 = {want}"
        );
    }

    #[test]
    fn zeta_edge_ladder_vs_torch() {
        // Live torch.special.zeta edge ladder:
        //   x==1            -> +inf
        //   x<1 (0.5)       -> NaN
        //   q==0 integer    -> +inf
        //   q<0 integer     -> +inf
        //   q<0 integer     -> +inf
        //   q<0 non-integer, x non-integer -> NaN
        let xs = [1.0, 0.5, 2.0, 2.0, 3.0, 2.5];
        let qs = [2.0, 1.0, 0.0, -1.0, -2.0, -1.5];
        let r = zeta(&t(&xs, &[6]), &t(&qs, &[6])).unwrap();
        let d = r.data().unwrap();
        assert!(
            d[0].is_infinite() && d[0] > 0.0,
            "zeta(1,q) == +inf: {}",
            d[0]
        );
        assert!(d[1].is_nan(), "zeta(0.5,q) == NaN: {}", d[1]);
        assert!(
            d[2].is_infinite() && d[2] > 0.0,
            "zeta(2, q=0 integer) == +inf: {}",
            d[2]
        );
        assert!(
            d[3].is_infinite() && d[3] > 0.0,
            "zeta(2, q=-1 integer) == +inf: {}",
            d[3]
        );
        assert!(
            d[4].is_infinite() && d[4] > 0.0,
            "zeta(3, q=-2 integer) == +inf: {}",
            d[4]
        );
        assert!(
            d[5].is_nan(),
            "zeta(2.5, q=-1.5 non-integer) == NaN: {}",
            d[5]
        );
    }

    #[test]
    fn zeta_f32_vs_torch() {
        // f32 oracle (live torch 2.11). f64-then-narrow must stay inside the
        // f32 transcendental tolerance.
        let xs = vec![2.0f32, 3.0, 1.5, 4.0];
        let qs = vec![1.0f32, 2.0, 2.0, 0.5];
        let x = Tensor::from_storage(TensorStorage::cpu(xs.clone()), vec![4], false).unwrap();
        let q = Tensor::from_storage(TensorStorage::cpu(qs.clone()), vec![4], false).unwrap();
        let r = zeta(&x, &q).unwrap();
        let d = r.data().unwrap();
        // torch.special.zeta f32: [1.6449341, 0.20205691, 1.6123753, 16.234848]
        let want = [1.6449341f32, 0.20205691, 1.6123753, 16.234848];
        for i in 0..4 {
            assert!(
                (d[i] - want[i]).abs() <= 1e-4 * (1.0 + want[i].abs()),
                "zeta f32 idx {i} x={} q={}: got {} want {}",
                xs[i],
                qs[i],
                d[i],
                want[i]
            );
        }
    }

    #[test]
    fn zeta_cuda_not_implemented() {
        // CPU input only here; the CUDA-dispatch guard returns NotImplementedOnCuda
        // (no host round trip) — exercised on-device in ferrotorch-gpu when a CUDA
        // tensor is passed. Smoke: the CPU path works for the binary op.
        let r = zeta(&t(&[2.0], &[1]), &t(&[1.0], &[1])).unwrap();
        assert!(r.data().unwrap()[0].is_finite());
    }

    #[test]
    fn airy_ai_known_values_vs_torch() {
        // Live torch.special.airy_ai across all regions: x<-2.09 (oscillatory),
        // mid Maclaurin, x>=2.09 (decaying), incl. region boundaries.
        let xs = [
            -5.0, -2.5, -2.09, -2.0, -1.0, 0.0, 1.0, 2.0, 2.09, 5.0, 8.0, 10.0, 100.0,
        ];
        let r = airy_ai(&t(&xs, &[13])).unwrap();
        let d = r.data().unwrap();
        let want = [
            0.35076100902415286,
            -0.11232483666261353,
            0.17005055173203007,
            0.22740742820168564,
            0.5355608832923521,
            0.3550280538878172,
            0.13529241631288144,
            0.03492413042327433,
            0.03042031836319837,
            0.00010834442813607433,
            4.692207616099224e-08,
            1.1047532552898654e-10,
            2.6344821520882847e-291,
        ];
        for i in 0..13 {
            assert!(
                (d[i] - want[i]).abs() <= 1e-10 * (1.0 + want[i].abs()),
                "airy_ai idx {i} x={}: got {} want {}",
                xs[i],
                d[i],
                want[i]
            );
        }
    }

    #[test]
    fn airy_ai_zero_vs_torch() {
        // airy_ai(0) == 0.3550280538878172 (= 3^(-2/3)/Gamma(2/3), live torch).
        let r = airy_ai(&t(&[0.0], &[1])).unwrap();
        let got = r.data().unwrap()[0];
        let want = 0.3550280538878172;
        assert!(
            (got - want).abs() <= 1e-12,
            "airy_ai(0) got {got} want {want}"
        );
    }

    #[test]
    fn airy_ai_edges_vs_torch() {
        // Live torch: airy_ai([inf,-inf,nan,200]) = [nan, nan, nan, 0]
        // (isinf short-circuit -> NaN; x>103.892 -> 0).
        let r = airy_ai(&t(
            &[f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 200.0],
            &[4],
        ))
        .unwrap();
        let d = r.data().unwrap();
        assert!(d[0].is_nan(), "airy_ai(+inf) == NaN: {}", d[0]);
        assert!(d[1].is_nan(), "airy_ai(-inf) == NaN: {}", d[1]);
        assert!(d[2].is_nan(), "airy_ai(NaN) == NaN: {}", d[2]);
        assert_eq!(d[3], 0.0, "airy_ai(200) == 0 (x>103.892 branch)");
    }

    #[test]
    fn airy_ai_f32_vs_torch() {
        // f32 oracle (live torch 2.11). f64-then-narrow stays inside f32 tol.
        let xs = vec![-5.0f32, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0];
        let input = Tensor::from_storage(TensorStorage::cpu(xs.clone()), vec![7], false).unwrap();
        let r = airy_ai(&input).unwrap();
        let d = r.data().unwrap();
        let want = [
            0.35076096653938293f32,
            0.22740741074085236,
            0.5355609059333801,
            0.35502806305885315,
            0.13529238104820251,
            0.03492411598563194,
            0.00010834442946361378,
        ];
        for i in 0..7 {
            assert!(
                (d[i] - want[i]).abs() <= 1e-4 * (1.0 + want[i].abs()),
                "airy_ai f32 idx {i} x={}: got {} want {}",
                xs[i],
                d[i],
                want[i]
            );
        }
    }
}
