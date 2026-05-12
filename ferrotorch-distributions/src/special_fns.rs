//! Scalar special functions used internally by distribution implementations.
//!
//! These mirror the implementations in `ferrotorch_core::special` but operate
//! on scalars rather than tensors, which is what the distribution code needs
//! for per-element map operations.
//!
//! [CL-329]

use ferrotorch_core::dtype::Float;

// ---------------------------------------------------------------------------
// Lanczos approximation for lgamma
// ---------------------------------------------------------------------------

const LANCZOS_G: f64 = 7.0;

#[rustfmt::skip]
const LANCZOS_COEFFICIENTS: [f64; 9] = [
    0.999_999_999_999_809_9,
    676.520_368_121_885_1,
   -1_259.139_216_722_402_8,
    771.323_428_777_653_1,
   -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_572e-6,
    1.505_632_735_149_311_6e-7,
];

/// Compute lgamma(x) = log(|Gamma(x)|) using the Lanczos approximation.
pub(crate) fn lgamma_scalar<T: Float>(x: T) -> T {
    let one = <T as num_traits::One>::one();
    let half = T::from(0.5).unwrap();
    let half_ln_2pi = T::from(0.918_938_533_204_672_7).unwrap();
    let g = T::from(LANCZOS_G).unwrap();

    // Handle negative values via reflection formula.
    if x < half {
        let pi = T::from(std::f64::consts::PI).unwrap();
        let sin_pi_x = (pi * x).sin();
        if sin_pi_x == <T as num_traits::Zero>::zero() {
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
    half_ln_2pi + t.ln() * (z + half) - t + sum.ln()
}

/// Compute digamma(x) = psi(x) = d/dx ln(Gamma(x)).
///
/// Uses the recurrence relation psi(x+1) = psi(x) + 1/x to shift x into the
/// range [6, inf), then applies the asymptotic expansion.
pub(crate) fn digamma_scalar<T: Float>(x: T) -> T {
    let zero = <T as num_traits::Zero>::zero();
    let one = <T as num_traits::One>::one();
    let half = T::from(0.5).unwrap();

    if x.is_nan() {
        return x; // NaN
    }

    // For negative x, use the reflection formula:
    // psi(1 - x) - pi * cot(pi * x) = psi(x)
    // => psi(x) = psi(1 - x) - pi * cos(pi*x) / sin(pi*x)
    if x < zero {
        let pi = T::from(std::f64::consts::PI).unwrap();
        let pi_x = pi * x;
        let cot = pi_x.cos() / pi_x.sin();
        return digamma_scalar(one - x) - pi * cot;
    }

    // Shift x upward until x >= 6 using psi(x) = psi(x+1) - 1/x.
    let mut result = zero;
    let mut y = x;
    let six = T::from(6.0).unwrap();
    while y < six {
        result = result - one / y;
        y += one;
    }

    // Asymptotic expansion (Abramowitz & Stegun 6.3.18).
    let y2 = one / (y * y);
    result = result + y.ln()
        - half / y
        - y2 * (T::from(1.0 / 12.0).unwrap()
            - y2 * (T::from(1.0 / 120.0).unwrap()
                - y2 * (T::from(1.0 / 252.0).unwrap()
                    - y2 * (T::from(1.0 / 240.0).unwrap() - y2 * T::from(1.0 / 132.0).unwrap()))));

    result
}

#[cfg(test)]
mod tests {
    //! Reference values produced from `scipy.special.gammaln` and
    //! `scipy.special.digamma` at the same arguments. The Lanczos approximation
    //! in this module is accurate to ~1e-12 across x in [0.1, 100]; the digamma
    //! asymptotic-with-shift expansion is accurate to ~1e-10 in the same range.
    //!
    //! See ferrotorch issue #1119 for the consolidation context.

    use super::{digamma_scalar, lgamma_scalar};

    /// scipy reference pairs — generated via:
    /// ```text
    /// python -c "import scipy.special as s; print(repr(s.gammaln(x)))"
    /// ```
    /// f64 literals use Rust's underscore-grouping form to stay within the
    /// clippy `excessive_precision` budget; values are still shortest-unique
    /// round-trip representations of the scipy doubles.
    #[allow(clippy::approx_constant)] // intentional: scipy reference values
    fn lgamma_reference_cases() -> [(f64, f64); 15] {
        [
            (0.1, 2.252_712_651_734_206),
            (0.25, 1.288_022_524_698_077_4),
            (0.5, 0.572_364_942_924_7),
            (0.75, 0.203_280_951_431_295_38),
            (1.0, 0.0),
            (1.5, -0.120_782_237_635_245_26),
            (2.0, 0.0),
            (3.0, 0.693_147_180_559_945_3),
            (4.0, 1.791_759_469_228_055),
            (5.0, 3.178_053_830_347_945_8),
            (6.0, 4.787_491_742_782_046),
            (7.5, 7.534_364_236_758_734),
            (10.0, 12.801_827_480_081_469),
            (25.0, 54.784_729_398_112_32),
            (100.0, 359.134_205_369_575_4),
        ]
    }

    #[allow(clippy::approx_constant)] // intentional: scipy reference values
    fn digamma_reference_cases() -> [(f64, f64); 15] {
        [
            (0.1, -10.423_754_940_411_076),
            (0.25, -4.227_453_533_376_266),
            (0.5, -1.963_510_026_021_423_5),
            (0.75, -1.085_860_879_786_472_2),
            (1.0, -0.577_215_664_901_532_9),
            (1.5, 0.036_489_973_978_576_52),
            (2.0, 0.422_784_335_098_467_13),
            (3.0, 0.922_784_335_098_467_1),
            (4.0, 1.256_117_668_431_800_3),
            (5.0, 1.506_117_668_431_800_3),
            (6.0, 1.706_117_668_431_800_5),
            (7.5, 1.946_757_484_246_086_6),
            (10.0, 2.251_752_589_066_721),
            (25.0, 3.198_742_512_851_974),
            (100.0, 4.600_161_852_738_088),
        ]
    }

    #[test]
    fn lgamma_matches_scipy_reference() {
        for (x, expected) in lgamma_reference_cases() {
            let got = lgamma_scalar(x);
            let err = (got - expected).abs();
            // Use relative tol where |expected| > 1, absolute otherwise.
            let tol = 1e-11 * expected.abs().max(1.0);
            assert!(
                err < tol,
                "lgamma({x}): got {got}, expected {expected}, |err| = {err}, tol = {tol}"
            );
        }
    }

    #[test]
    fn digamma_matches_scipy_reference() {
        for (x, expected) in digamma_reference_cases() {
            let got = digamma_scalar(x);
            let err = (got - expected).abs();
            let tol = 1e-9 * expected.abs().max(1.0);
            assert!(
                err < tol,
                "digamma({x}): got {got}, expected {expected}, |err| = {err}, tol = {tol}"
            );
        }
    }

    #[test]
    fn lgamma_negative_reflection_matches_scipy() {
        // gammaln(-0.5) = ln(2*sqrt(pi)); gammaln(-1.5) = ln(4*sqrt(pi)/3).
        for (x, expected) in [
            (-0.5_f64, 1.265_512_123_484_645_4_f64),
            (-1.5_f64, 0.860_047_015_376_481_f64),
        ] {
            let got = lgamma_scalar(x);
            let err = (got - expected).abs();
            assert!(
                err < 1e-11,
                "lgamma({x}): got {got}, expected {expected}, |err| = {err}"
            );
        }
    }

    #[test]
    fn lgamma_f32_round_trip() {
        // f32 carries only ~7 decimal digits of precision; relax to 1e-5.
        for (x, expected) in lgamma_reference_cases() {
            let got_f32 = lgamma_scalar(x as f32);
            let err = (f64::from(got_f32) - expected).abs();
            // f32 lgamma has relative error ~1e-6 (Lanczos coeffs round-tripped).
            let tol = 5e-6 * expected.abs().max(1.0);
            assert!(
                err < tol,
                "lgamma_f32({x}): got {got_f32}, expected {expected}, err = {err}"
            );
        }
    }
}
