//! Scalar special functions used internally by distribution implementations.
//!
//! These mirror the implementations in `ferrotorch_core::special` but operate
//! on scalars rather than tensors, which is what the distribution code needs
//! for per-element map operations.
//!
//! [CL-329]
//!
//! ## REQ status (per `.design/ferrotorch-distributions/special_fns.md`)
//!
//! Full evidence rows (impl + non-test production consumer + upstream cites)
//! live in the design doc; this synopsis is a one-line summary per REQ.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`lgamma_scalar<T>` Lanczos + reflection) | SHIPPED | `pub(crate) fn lgamma_scalar<T: Float>(x: T) -> T` with 9-coefficient Lanczos + reflection branch in `special_fns.rs` matching scipy reference to 1e-11 f64; consumers: `fn kl_gamma_scalar` in `kl.rs`; `fn Gamma::log_prob` in `gamma.rs`; `fn Beta::log_prob` in `beta.rs` (3 calls for `lnB(α,β)`); 9 production callers total |
//! | REQ-2 (`digamma_scalar<T>` recurrence + asymptotic) | SHIPPED | `pub(crate) fn digamma_scalar<T: Float>(x: T) -> T` with recurrence + Abramowitz-Stegun asymptotic + reflection branch in `special_fns.rs` matching scipy reference to 1e-9 f64; consumers: `fn kl_gamma_scalar` in `kl.rs`; `fn Beta::entropy` in `beta.rs`; `fn Dirichlet::entropy` in `dirichlet.rs`; 5 production callers |
//! | REQ-3 (`<T: Float>` dual-precision pattern) | SHIPPED | both fns are `<T: Float>` generic with f64 `LANCZOS_COEFFICIENTS: [f64; 9]` const promoted via `T::from(<f64>).unwrap()` in `special_fns.rs`; consumer: `fn Beta::log_prob` operates on `T = f32` AND `T = f64` (both tested by `_f64` variants) routing through the same generic body |
//! | REQ-4 (`pub(crate)` visibility) | SHIPPED | `pub(crate) fn lgamma_scalar`, `pub(crate) fn digamma_scalar` + `pub(crate) mod special_fns;` in `lib.rs` together make the module crate-internal; consumers: 9 production sites within `ferrotorch-distributions/src/` (`gamma.rs`, `beta.rs`, `dirichlet.rs`, `kumaraswamy.rs`, `multinomial.rs`, `poisson.rs`, `student_t.rs`, `weibull.rs`, `kl.rs`); `cargo doc` omits these from public docs |

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

// ---------------------------------------------------------------------------
// Pathwise (implicit-reparameterization) gradient of a standard-Gamma sample
// ---------------------------------------------------------------------------

/// Compute the reparameterized gradient `d sample / d alpha = -(d/dalpha
/// cdf(x; alpha)) / pdf(x; alpha)` for a sample `x` drawn from a standard
/// Gamma distribution `Gamma(alpha, 1)`.
///
/// This is a direct port of PyTorch's `standard_gamma_grad_one` from
/// `aten/src/ATen/native/Distributions.h:302-368` (the kernel behind
/// `torch._standard_gamma_grad`). It is the PATHWISE per-sample gradient — for
/// `alpha >= 1` it is strictly positive for every `x > 0` — and it replaces the
/// high-variance score-function closed form `x * (ln x - psi(alpha))`, which is
/// only unbiased in expectation and routinely flips sign per-sample.
///
/// Three branches mirror upstream exactly:
/// 1. small `x` (`x < 0.8`): Taylor series for the incomplete gamma + its
///    derivative, divided by the pdf;
/// 2. large `alpha` (`alpha > 8`): Rice saddle-point expansion (with a near-mean
///    polynomial sub-branch for `0.9 alpha <= x <= 1.1 alpha`);
/// 3. otherwise: a bivariate rational approximation in `(ln(x/alpha), ln alpha)`.
pub(crate) fn standard_gamma_grad_one<T: Float>(alpha: T, x: T) -> T {
    let zero = <T as num_traits::Zero>::zero();
    let one = <T as num_traits::One>::one();

    // 1. Taylor series expansion for small x.
    if x < T::from(0.8).unwrap() {
        let mut numer = one;
        let mut denom = alpha;
        let mut series1 = numer / denom;
        let mut series2 = numer / (denom * denom);
        for i in 1..=5 {
            numer = numer * (-x / T::from(i as f64).unwrap());
            denom += one;
            series1 += numer / denom;
            series2 += numer / (denom * denom);
        }
        let pow_x_alpha = x.powf(alpha);
        let gamma_pdf = x.powf(alpha - one) * (-x).exp();
        let gamma_cdf = pow_x_alpha * series1;
        let gamma_cdf_alpha = (x.ln() - digamma_scalar(alpha)) * gamma_cdf - pow_x_alpha * series2;
        let result = -gamma_cdf_alpha / gamma_pdf;
        return if result.is_nan() { zero } else { result };
    }

    // 2. Rice saddle point expansion for large alpha.
    if alpha > T::from(8.0).unwrap() {
        let p9 = T::from(0.9).unwrap();
        let p11 = T::from(1.1).unwrap();
        if p9 * alpha <= x && x <= p11 * alpha {
            let c24 = T::from(24.0).unwrap();
            let c12 = T::from(12.0).unwrap();
            let numer_1 = one + c24 * alpha * (one + c12 * alpha);
            let c1440 = T::from(1440.0).unwrap();
            let c6 = T::from(6.0).unwrap();
            let c53 = T::from(53.0).unwrap();
            let c120 = T::from(120.0).unwrap();
            let c65 = T::from(65.0).unwrap();
            let c107 = T::from(107.0).unwrap();
            let c3600 = T::from(3600.0).unwrap();
            let numer_2 = c1440 * (alpha * alpha) + c6 * x * (c53 - c120 * x) - c65 * x * x / alpha
                + alpha * (c107 + c3600 * x);
            let denom = T::from(1_244_160.0).unwrap() * (alpha * alpha) * (alpha * alpha);
            return numer_1 * numer_2 / denom;
        }
        let denom = (T::from(8.0).unwrap() * alpha).sqrt();
        let term2 = denom / (alpha - x);
        let term3 = (x - alpha - alpha * (x / alpha).ln()).powf(T::from(-1.5).unwrap());
        let term23 = if x < alpha {
            term2 - term3
        } else {
            term2 + term3
        };
        let two = T::from(2.0).unwrap();
        let term1 = (x / alpha).ln() * term23
            - (two / alpha).sqrt() * (alpha + x) / ((alpha - x) * (alpha - x));
        let c12 = T::from(12.0).unwrap();
        let c24 = T::from(24.0).unwrap();
        let stirling = one + one / (c12 * alpha) * (one + one / (c24 * alpha));
        let numer = x * term1;
        return -stirling * numer / denom;
    }

    // 3. Bivariate rational approximation to the reparameterized gradient.
    let u = (x / alpha).ln();
    let v = alpha.ln();
    #[rustfmt::skip]
    const COEF_UV: [[f64; 8]; 3] = [
        [0.160_093_98, -0.094_634_809, 0.025_146_376, -0.003_064_834_3,
         1.0, 0.326_681_15, 0.104_060_89, 0.001_417_908_4],
        [0.534_878_93, 0.129_807_1, 0.065_735_949, -0.001_564_975_8,
         0.166_394_65, 0.020_070_113, -0.003_593_891_5, -0.000_583_926_23],
        [0.040_121_004, -0.006_591_402_2, -0.002_628_604_7, -0.001_344_177_7,
         0.017_050_642, -0.002_130_932_6, 0.000_850_923_67, -1.524_787_7e-7],
    ];
    let mut coef_v = [zero; 8];
    for (i, cv) in coef_v.iter_mut().enumerate() {
        let c0 = T::from(COEF_UV[0][i]).unwrap();
        let c1 = T::from(COEF_UV[1][i]).unwrap();
        let c2 = T::from(COEF_UV[2][i]).unwrap();
        *cv = c0 + u * (c1 + u * c2);
    }
    let p = coef_v[0] + v * (coef_v[1] + v * (coef_v[2] + v * coef_v[3]));
    let q = coef_v[4] + v * (coef_v[5] + v * (coef_v[6] + v * coef_v[7]));
    (p / q).exp()
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

    /// Reference values from live `torch._standard_gamma_grad(alpha, x)`
    /// (this machine, 2026-05-26). The (alpha, x) pairs deliberately span all
    /// three branches of `standard_gamma_grad_one`:
    ///   - small x (x < 0.8): rows with x in {0.5, 0.3, 0.2, 0.7, 0.4, 0.1}
    ///   - large alpha (alpha > 8): rows with alpha in {10, 12, 8.5, 20}
    ///   - bivariate rational (else): rows like (2.5, 2.0), (3.0, 1.5), (5.0, 4.0)
    use super::standard_gamma_grad_one;

    #[test]
    fn standard_gamma_grad_one_matches_torch() {
        let cases: [(f64, f64, f64); 14] = [
            (2.5, 0.5, 0.426_839_502_982),
            (2.5, 2.0, 0.953_443_966_788),
            (2.5, 0.3, 0.305_572_785_677),
            (0.5, 0.2, 0.794_050_566_982),
            (0.5, 0.7, 1.586_772_237_284),
            (1.0, 0.4, 0.708_752_718_551),
            (10.0, 9.5, 0.990_918_561_469),
            (12.0, 12.0, 1.013_961_394_301),
            (3.0, 1.5, 0.731_518_719_374),
            (5.0, 4.0, 0.922_753_048_730),
            (8.5, 8.5, 1.019_752_489_616),
            (2.5, 0.5586, 0.457_775_125_088),
            (0.9, 0.1, 0.314_776_859_958),
            (20.0, 18.0, 0.956_250_962_552),
        ];
        for (alpha, x, expected) in cases {
            let got = standard_gamma_grad_one(alpha, x);
            // torch uses the same approximation; the rational/saddle branches
            // carry the approximation's intrinsic error (~1e-3 rel near edges).
            let tol = 1e-4 * expected.abs().max(1.0);
            assert!(
                (got - expected).abs() < tol,
                "standard_gamma_grad_one({alpha}, {x}): got {got}, torch {expected}, |err|={}",
                (got - expected).abs()
            );
        }
    }

    /// The pathwise gradient must equal the implicit-function value
    /// `-(d/dalpha CDF(x;alpha)) / pdf(x;alpha)`, which we recover by a central
    /// finite difference of the regularized lower-incomplete-gamma CDF
    /// `P(alpha, x)` w.r.t. alpha, divided by the Gamma pdf. This is an
    /// independent oracle (no torch dependency) confirming the port is the
    /// genuine pathwise gradient and not a different-but-still-wrong formula.
    #[test]
    fn standard_gamma_grad_one_matches_finite_difference_implicit() {
        // Lanczos lgamma is already in this module; build P(a,x) via a simple
        // series / continued-fraction regularized incomplete gamma.
        fn gammp(a: f64, x: f64) -> f64 {
            if x <= 0.0 {
                return 0.0;
            }
            let gln = lgamma_scalar(a);
            if x < a + 1.0 {
                // power series
                let mut ap = a;
                let mut sum = 1.0 / a;
                let mut del = sum;
                for _ in 0..500 {
                    ap += 1.0;
                    del *= x / ap;
                    sum += del;
                    if del.abs() < sum.abs() * 1e-15 {
                        break;
                    }
                }
                sum * (-x + a * x.ln() - gln).exp()
            } else {
                // Lentz continued fraction for Q, return 1 - Q
                let tiny = 1e-300;
                let mut b = x + 1.0 - a;
                let mut c = 1.0 / tiny;
                let mut d = 1.0 / b;
                let mut h = d;
                for i in 1..500 {
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
                let q = (-x + a * x.ln() - gln).exp() * h;
                1.0 - q
            }
        }
        // pdf of standard Gamma(alpha) at x = x^(a-1) e^-x / Gamma(a).
        fn gamma_pdf(a: f64, x: f64) -> f64 {
            ((a - 1.0) * x.ln() - x - lgamma_scalar(a)).exp()
        }
        // Pairs in the small-x and rational branches (the FD oracle for the
        // saddle branch is dominated by approximation error in both, so we pin
        // the saddle branch against torch above instead).
        for (alpha, x) in [(2.5, 0.5), (2.5, 2.0), (3.0, 1.5), (5.0, 4.0), (0.9, 0.3)] {
            let h = 1e-6;
            let dp_dalpha = (gammp(alpha + h, x) - gammp(alpha - h, x)) / (2.0 * h);
            let implicit = -dp_dalpha / gamma_pdf(alpha, x);
            let got = standard_gamma_grad_one(alpha, x);
            let tol = 2e-3 * implicit.abs().max(1.0);
            assert!(
                (got - implicit).abs() < tol,
                "FD implicit grad at alpha={alpha}, x={x}: port={got}, FD-implicit={implicit}, |err|={}",
                (got - implicit).abs()
            );
        }
    }
}
