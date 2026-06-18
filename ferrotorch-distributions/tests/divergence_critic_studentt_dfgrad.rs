//! Critic re-audit of commit `e9f446255` (#1427): StudentT `df` gradient via
//! the Chi2 implicit reparameterization (the #1555-class trap — a plausible
//! single-channel or wrong-arg-order gradient that passes a shallow probe).
//!
//! ## Upstream contract
//!
//! `torch/distributions/studentT.py:rsample` (this clone):
//! ```text
//!   X = standard_normal(shape)
//!   Z = self._chi2.rsample(sample_shape)   # Chi2(df), reparameterized -> grad flows to df
//!   Y = X * torch.rsqrt(Z / self.df)
//!   return self.loc + self.scale * Y
//! ```
//! So `sample = loc + scale * z * sqrt(df / Z)` with `Z = Chi2(df) = 2*sg`,
//! `sg ~ Gamma(df/2, 1)`. The total derivative w.r.t. `df` has TWO channels:
//!
//!   d sample/d df = scale*z*[ 0.5/sqrt(df*Z)
//!                             - 0.5*sqrt(df)*Z^(-1.5) * dZ/d df ]
//!   dZ/d df = 2 * d sg/d alpha * d alpha/d df  (alpha=df/2, d alpha/d df = 0.5)
//!           = d sg/d alpha    (the 2 and the 0.5 cancel)
//!   d sg/d alpha = the PATHWISE standard-Gamma reparam gradient.
//!
//! ## The trap (verified)
//!
//! The pathwise grad is `-(d_alpha P(alpha, x)) / pdf(x; alpha)` — the
//! upstream kernel `standard_gamma_grad_one(alpha_, x_)`
//! (`aten/.../Distributions.h:302`, FIRST arg = alpha, SECOND = the sample x;
//! `native_functions.yaml:6851` `_standard_gamma_grad(self=alpha, output=x)`).
//! ferrotorch calls `standard_gamma_grad_one(df*0.5, chi2*0.5)` =
//! `standard_gamma_grad_one(alpha=df/2, x=sg)` — CORRECT order. A swapped-arg
//! bug would still be finite + plausible (the #1555 failure mode), so the
//! oracle below is anchored to torch with the arguments in the CORRECT order
//! AND cross-checked against a pure reparameterization finite difference
//! (`P(alpha, sg) = const`, invert for sg at perturbed alpha) which is the
//! ground truth independent of any kernel.
//!
//! ## Independent oracle (does NOT reuse ferrotorch's `standard_gamma_grad_one`
//! nor the builder's in-crate `gammp_ref` FD)
//!
//! `oracle_dsg_dalpha` is a fresh Lanczos-series regularized lower incomplete
//! gamma `P(a,x)` plus a central FD of `-(d_a P)/pdf`. It is ANCHORED to live
//! `torch._standard_gamma_grad(alpha, x)` constants (`TORCH_DSG_*`) at four
//! `(df, sg)` cases spanning the small-x / rational / large-alpha branches.

use ferrotorch_core::creation::scalar;
use ferrotorch_core::manual_seed;
use ferrotorch_distributions::{Distribution, StudentT};

// ---------------------------------------------------------------------------
// torch golden constants: live `torch._standard_gamma_grad(alpha, x)` with the
// CORRECT argument order alpha=df/2 (self), x=sg (output) — torch 2.11.0+cu130,
// this machine, 2026-05-26. Cross-checked against a pure reparameterization FD
// (see `studentt_df_grad_pure_finite_difference`).
// ---------------------------------------------------------------------------
const TORCH_DSG_DF5_SG2: f64 = 0.953_443_966_787_819_3; // alpha=2.5 rational branch
const TORCH_DSG_DF3_SG15: f64 = 1.115_491_930_557_962_7; // alpha=1.5 rational branch
const TORCH_DSG_DF16_SG03: f64 = 0.703_735_803_842_609_4; // alpha=0.8 small-x branch
const TORCH_DSG_DF20_SG9: f64 = 0.964_218_107_437_950_1; // alpha=10 large-alpha branch

/// lgamma via Lanczos (g=7, n=9) — fresh implementation, independent of the
/// production `lgamma_scalar` and the builder's in-crate copy.
fn lgamma_lanczos(x: f64) -> f64 {
    const G: [f64; 9] = [
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
    if x < 0.5 {
        // reflection
        (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - lgamma_lanczos(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = G[0];
        let t = x + 7.5;
        for (i, &g) in G.iter().enumerate().skip(1) {
            a += g / (x + i as f64);
        }
        0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}

/// Regularized lower incomplete gamma `P(a, x)` via series (x < a+1) or
/// continued fraction (else). Fresh impl (distinct from builder's `gammp_ref`).
fn regularized_lower_gamma(a: f64, x: f64) -> f64 {
    let gln = lgamma_lanczos(a);
    if x < a + 1.0 {
        let mut ap = a;
        let mut del = 1.0 / a;
        let mut sum = del;
        for _ in 0..1000 {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * 1e-16 {
                break;
            }
        }
        sum * (-x + a * x.ln() - gln).exp()
    } else {
        let fpmin = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / fpmin;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..1000 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < fpmin {
                d = fpmin;
            }
            c = b + an / c;
            if c.abs() < fpmin {
                c = fpmin;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-16 {
                break;
            }
        }
        1.0 - (-x + a * x.ln() - gln).exp() * h
    }
}

/// Independent oracle for `d sg / d alpha` of a reparameterized standard-Gamma
/// sample, via the implicit-function identity `-(d_a P(a, sg)) / pdf(sg)`.
fn oracle_dsg_dalpha(alpha: f64, sg: f64) -> f64 {
    let h = 1e-6;
    let dp = (regularized_lower_gamma(alpha + h, sg) - regularized_lower_gamma(alpha - h, sg))
        / (2.0 * h);
    let pdf = ((alpha - 1.0) * sg.ln() - sg - lgamma_lanczos(alpha)).exp();
    -dp / pdf
}

/// The full two-channel total derivative `d sample / d df`, computed by hand
/// from fixed `(z, sg)` using the independent oracle for `dZ/d df`.
fn oracle_dsample_ddf(df: f64, z: f64, sg: f64, scale: f64) -> f64 {
    let chi2 = 2.0 * sg;
    let dz_ddf = oracle_dsg_dalpha(df / 2.0, sg); // = 2 * dsg * 0.5
    let explicit = 0.5 / (df * chi2).sqrt();
    let implicit = 0.5 * df.sqrt() * chi2.powf(-1.5) * dz_ddf;
    scale * z * (explicit - implicit)
}

/// Step 0: anchor the independent oracle to torch's own kernel (CORRECT arg
/// order alpha, x). If this fails, the oracle is wrong and every downstream
/// assertion is suspect.
#[test]
fn oracle_dsg_dalpha_matches_torch_standard_gamma_grad() {
    let cases = [
        (5.0_f64, 2.0_f64, TORCH_DSG_DF5_SG2),
        (3.0, 1.5, TORCH_DSG_DF3_SG15),
        (1.6, 0.3, TORCH_DSG_DF16_SG03),
        (20.0, 9.0, TORCH_DSG_DF20_SG9),
    ];
    for (df, sg, torch_val) in cases {
        let got = oracle_dsg_dalpha(df / 2.0, sg);
        let err = (got - torch_val).abs();
        // FD on a 1e-12-accurate incomplete gamma: ~1e-3 relative is realistic.
        let tol = 2e-3 * torch_val.abs().max(1.0);
        assert!(
            err < tol,
            "oracle dsg/dalpha(df={df}, sg={sg}) = {got}, torch._standard_gamma_grad(alpha,x) = {torch_val}, |err|={err}"
        );
    }
}

/// Drive the public `StudentT::rsample` end-to-end with a fixed seed, recover
/// the exact internal `z` and `chi2` by replaying the same RNG draw sequence,
/// then assert the production `df.grad` matches the INDEPENDENT torch-anchored
/// oracle. df spans {1.6, 3, 5, 20} (all `standard_gamma_grad_one` branches).
#[test]
fn studentt_df_grad_endtoend_matches_independent_oracle() {
    let scale_v = 1.4_f64;
    for &df_v in &[1.6_f64, 3.0, 5.0, 20.0] {
        // Production end-to-end.
        manual_seed(20260526).unwrap();
        let df = scalar(df_v).unwrap().requires_grad_(true);
        let loc = scalar(0.0_f64).unwrap();
        let scale = scalar(scale_v).unwrap();
        let dist = StudentT::new(df.clone(), loc, scale).unwrap();
        let s = dist.rsample(&[1]).unwrap();
        let sample_val = s.item().unwrap();
        s.sum_all().unwrap().backward().unwrap();
        let prod_df_grad = df
            .grad()
            .unwrap()
            .expect("gradient should be populated")
            .item()
            .unwrap();

        // Recover z and chi2 by replaying the exact RNG sequence rsample used.
        manual_seed(20260526).unwrap();
        let (z, chi2) = replay_z_and_chi2(df_v);

        // Sanity: sample = loc + scale*z*sqrt(df/chi2) must equal the production
        // sample, proving the replay reconstructed the SAME internal draws.
        let reconstructed = scale_v * z * (df_v / chi2).sqrt();
        assert!(
            (reconstructed - sample_val).abs() < 1e-9,
            "RNG replay mismatch at df={df_v}: reconstructed sample {reconstructed} vs production {sample_val}"
        );

        let sg = chi2 / 2.0;
        let oracle = oracle_dsample_ddf(df_v, z, sg, scale_v);

        let tol = 5e-3 * oracle.abs().max(1.0);
        assert!(
            (prod_df_grad - oracle).abs() < tol,
            "df-grad DIVERGENCE at df={df_v} (z={z}, sg={sg}): production={prod_df_grad}, \
             torch-anchored oracle={oracle}, |err|={}",
            (prod_df_grad - oracle).abs()
        );
    }
}

/// Replay `randn([1])` then the production `sample_chi2` Marsaglia-Tsang loop
/// to recover the exact `(z, chi2)` the seeded rsample consumed. Mirrors
/// `student_t.rs:sample_chi2` (batch=256, Gamma(df/2,1)*2) exactly.
fn replay_z_and_chi2(df: f64) -> (f64, f64) {
    use ferrotorch_core::creation::{rand, randn};

    // 1. z: rsample draws randn(shape) first.
    let z = randn::<f64>(&[1]).unwrap().data_vec().unwrap()[0];

    // 2. sample_chi2(df_values=[df], n=1): batch = 1.max(256) = 256.
    let batch = 256usize;
    let mut norm_buf: Vec<f64> = randn::<f64>(&[batch]).unwrap().data_vec().unwrap();
    let mut unif_buf: Vec<f64> = rand::<f64>(&[batch]).unwrap().data_vec().unwrap();
    let mut ni = 0usize;
    let mut ui = 0usize;
    let next_normal = |ni: &mut usize, nb: &mut Vec<f64>| {
        if *ni >= nb.len() {
            *nb = randn::<f64>(&[batch]).unwrap().data_vec().unwrap();
            *ni = 0;
        }
        let v = nb[*ni];
        *ni += 1;
        v
    };
    let next_uniform = |ui: &mut usize, ub: &mut Vec<f64>| {
        if *ui >= ub.len() {
            *ub = rand::<f64>(&[batch]).unwrap().data_vec().unwrap();
            *ui = 0;
        }
        let v = ub[*ui];
        *ui += 1;
        v
    };

    let alpha = df * 0.5;
    let (effective_alpha, needs_boost) = if alpha < 1.0 {
        (alpha + 1.0, true)
    } else {
        (alpha, false)
    };
    let d = effective_alpha - 1.0 / 3.0;
    let c = (1.0 / 3.0) / d.sqrt();

    let gamma_sample = loop {
        let x = next_normal(&mut ni, &mut norm_buf);
        let v_base = 1.0 + c * x;
        if v_base <= 0.0 {
            continue;
        }
        let v = v_base * v_base * v_base;
        let u = next_uniform(&mut ui, &mut unif_buf);
        let x2 = x * x;
        if u < 1.0 - 0.0331 * x2 * x2 {
            break d * v;
        }
        if u.ln() < 0.5 * x2 + d * (1.0 - v + v.ln()) {
            break d * v;
        }
    };
    let gamma_final = if needs_boost {
        let u = next_uniform(&mut ui, &mut unif_buf);
        let u_safe = u.max(1e-30);
        gamma_sample * u_safe.powf(1.0 / alpha)
    } else {
        gamma_sample
    };
    let chi2 = gamma_final * 2.0;
    (z, chi2)
}

/// Pure reparameterization finite difference (ground truth, no kernel): hold
/// the underlying uniform `p = P(df/2, sg)` fixed, perturb `df`, invert
/// `P(df'/2, sg') = p` for `sg'`, and central-difference `sample(df)`. Confirms
/// the analytic oracle at the four df values fully independently of production.
#[test]
fn studentt_df_grad_pure_finite_difference() {
    let scale_v = 1.4_f64;
    let z = 1.0_f64;
    for &df_v in &[1.6_f64, 3.0, 5.0, 20.0] {
        let sg = df_v / 2.0; // sg = alpha (near the mode), valid in every branch
        let h = 1e-5;

        let p0 = regularized_lower_gamma(df_v / 2.0, sg);
        let sample_at = |dfp: f64| -> f64 {
            // invert P(dfp/2, sg') = p0 for sg' by Newton on the pdf.
            let a = dfp / 2.0;
            let mut x = sg;
            for _ in 0..60 {
                let f = regularized_lower_gamma(a, x) - p0;
                let pdf = ((a - 1.0) * x.ln() - x - lgamma_lanczos(a)).exp();
                if pdf.abs() < 1e-300 {
                    break;
                }
                let step = f / pdf;
                x -= step;
                if step.abs() < 1e-14 {
                    break;
                }
            }
            let chi2 = 2.0 * x;
            scale_v * z * (dfp / chi2).sqrt()
        };

        let fd = (sample_at(df_v + h) - sample_at(df_v - h)) / (2.0 * h);
        let oracle = oracle_dsample_ddf(df_v, z, sg, scale_v);
        let tol = 5e-3 * oracle.abs().max(1.0);
        assert!(
            (fd - oracle).abs() < tol,
            "pure-FD vs analytic oracle mismatch at df={df_v}: fd={fd}, oracle={oracle}, |err|={}",
            (fd - oracle).abs()
        );
    }
}
