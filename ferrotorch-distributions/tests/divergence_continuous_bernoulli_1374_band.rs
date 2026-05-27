//! Critic re-audit of commit c5cb6529d (#1374 ContinuousBernoulli sub-part).
//!
//! Every expected value below is from live `torch.distributions.ContinuousBernoulli`
//! / `torch.distributions.kl_divergence` (torch 2.11.0+cu130, 2026-05-27),
//! traced to a `torch/distributions/continuous_bernoulli.py` or
//! `torch/distributions/kl.py` line (R-CHAR-3, non-tautological — no value was
//! copied from the ferrotorch side).
//!
//! HIGH-RISK focus: the `_lims = (0.499, 0.501)` numerical-stability band where
//! torch swaps the exact closed form for a Taylor expansion about 0.5. A
//! mismatched cutoff predicate or Taylor coefficient shows up as a
//! discontinuity across the 0.499/0.501 boundary that diverges from torch.

use ferrotorch_distributions::kl::kl_divergence;
use ferrotorch_distributions::{
    Beta, ContinuousBernoulli, Distribution, Exponential, Normal, Uniform,
};
use ferrotorch_core::creation::{from_slice, scalar};

fn cb(p: f64) -> ContinuousBernoulli<f64> {
    ContinuousBernoulli::new(scalar(p).unwrap()).unwrap()
}

fn close(a: f64, b: f64, tol: f64, ctx: &str) {
    assert!(
        (a - b).abs() <= tol || (a.is_infinite() && b.is_infinite() && a.signum() == b.signum()),
        "{ctx}: ferrotorch={a:?} torch={b:?} (|Δ|={:?} > tol={tol:?})",
        (a - b).abs()
    );
}

// ---------------------------------------------------------------------------
// log_prob(x=0.3) across the WHOLE 0.5 band (the most important check).
// torch continuous_bernoulli.py:187-194 + :120-138 (_cont_bern_log_norm).
// ---------------------------------------------------------------------------
#[test]
fn divergence_cb_log_prob_band_x03() {
    // (probs, torch log_prob(0.3))
    let cases = [
        (0.1, 0.245_810_670_632_161_9),
        (0.3, 0.139_723_448_825_565_3),
        (0.7, -0.199_195_695_329_316_02),
        (0.9, -0.633_079_160_302_325_8),
        // band edges + interior — these straddle the exact/Taylor boundary:
        (0.498, 0.001_597_341_839_731_592),
        (0.4989, 0.000_879_194_750_604_050_9),
        (0.499, 0.000_799_334_398_304_152_3),       // exact branch (p<=0.499)
        (0.499_000_1, 0.000_799_254_531_321_014_2),  // Taylor branch
        (0.4995, 0.000_399_833_466_561_139_16),
        (0.5, 0.0),
        (0.5005, -0.000_400_166_800_105_572_5),
        (0.501, -0.000_800_667_735_024_740_4),       // Taylor branch (p<=0.501)
        (0.501_000_1, -0.000_800_747_868_694_218_2), // exact branch (p>0.501)
        (0.502, -0.001_602_675_227_098_893),
    ];
    for (p, want) in cases {
        let got = cb(p)
            .log_prob(&scalar(0.3f64).unwrap())
            .unwrap()
            .item()
            .unwrap();
        close(got, want, 1e-12, &format!("log_prob(p={p}, x=0.3)"));
    }
}

// ---------------------------------------------------------------------------
// mean across {0.1,0.5,0.9} + band edges. continuous_bernoulli.py:140-148.
// ---------------------------------------------------------------------------
#[test]
fn divergence_cb_mean_band() {
    let cases = [
        (0.1, 0.330_119_613_313_418_77),
        (0.5, 0.5),
        (0.9, 0.669_880_386_686_581_3),
        (0.499, 0.499_666_666_313_316_9),       // exact
        (0.499_000_1, 0.499_666_699_644_551_1),  // Taylor
        (0.501, 0.500_333_333_688_888_9),        // Taylor
        (0.501_000_1, 0.500_333_367_020_005),    // exact
    ];
    for (p, want) in cases {
        let got = cb(p).mean().unwrap().item().unwrap();
        close(got, want, 1e-12, &format!("mean(p={p})"));
    }
}

// ---------------------------------------------------------------------------
// variance + entropy + cdf + icdf across the band. py:154-162,224-231,196-222.
// ---------------------------------------------------------------------------
#[test]
fn divergence_cb_variance_band() {
    let cases = [
        (0.3, 0.080_425_152_206_194_28),
        (0.499, 0.083_333_267_764_828_63),       // exact
        (0.499_000_1, 0.083_333_266_680_134_73),  // Taylor
        (0.5, 0.083_333_333_333_333_33),
        (0.501, 0.083_333_266_666_802_12),        // Taylor
        (0.501_000_1, 0.083_333_267_823_036_3),   // exact
    ];
    for (p, want) in cases {
        let got = cb(p).variance().unwrap().item().unwrap();
        close(got, want, 1e-12, &format!("variance(p={p})"));
    }
}

#[test]
fn divergence_cb_entropy_band() {
    let cases = [
        (0.3, -0.029_386_202_232_129_12),
        (0.499, -6.666_681_593_436_863e-7),      // exact
        (0.499_000_1, -6.665_348_505_352_497e-7), // Taylor
        (0.5, 0.0),
        (0.501, -6.666_681_777_733_885e-7),       // Taylor
        (0.501_000_1, -6.668_015_003_485_905e-7), // exact
    ];
    for (p, want) in cases {
        let got = cb(p).entropy().unwrap().item().unwrap();
        close(got, want, 1e-12, &format!("entropy(p={p})"));
    }
}

#[test]
fn divergence_cb_cdf_band() {
    // cdf(0.3) across the band; Taylor branch returns the value verbatim.
    let cases = [
        (0.3, 0.392_796_368_275_436_5),
        (0.5, 0.3),
        (0.7, 0.217_061_956_894_589_8),
        (0.499, 0.300_420_112_442_633_95),    // exact
        (0.499_000_1, 0.3),                    // Taylor -> value
        (0.501, 0.3),                          // Taylor -> value
        (0.501_000_1, 0.299_580_069_580_106_97), // exact
    ];
    for (p, want) in cases {
        let got = cb(p)
            .cdf(&scalar(0.3f64).unwrap())
            .unwrap()
            .item()
            .unwrap();
        close(got, want, 1e-12, &format!("cdf(p={p}, 0.3)"));
    }
}

#[test]
fn divergence_cb_icdf_band() {
    let cases = [
        (0.3, 0.221_943_475_010_077_75),
        (0.5, 0.3),
        (0.7, 0.397_112_104_670_546_2),
        (0.499, 0.299_580_223_585_924_73),    // exact
        (0.499_000_1, 0.3),                    // Taylor -> value
        (0.501, 0.3),                          // Taylor -> value
        (0.501_000_1, 0.300_420_266_459_690_6), // exact
    ];
    for (p, want) in cases {
        let got = cb(p)
            .icdf(&scalar(0.3f64).unwrap())
            .unwrap()
            .item()
            .unwrap();
        close(got, want, 1e-12, &format!("icdf(p={p}, 0.3)"));
    }
}

// ---------------------------------------------------------------------------
// Batch broadcast: scalar value vs batched probs (the #1569 contract).
// ---------------------------------------------------------------------------
#[test]
fn divergence_cb_log_prob_batch_broadcast() {
    // torch: CB([0.3,0.7]).log_prob(0.5) == [-0.029736123251875357, -0.029736123251874913]
    let d = ContinuousBernoulli::new(from_slice(&[0.3f64, 0.7], &[2]).unwrap()).unwrap();
    let lp = d.log_prob(&scalar(0.5f64).unwrap()).unwrap();
    assert_eq!(lp.shape(), &[2], "shape must be [2] not the #1569 bug");
    let d = lp.data().unwrap();
    close(
        d[0],
        -0.029_736_123_251_875_357,
        1e-12,
        "CB[.3,.7].log_prob(.5)[0]",
    );
    close(
        d[1],
        -0.029_736_123_251_874_913,
        1e-12,
        "CB[.3,.7].log_prob(.5)[1]",
    );
}

// ---------------------------------------------------------------------------
// KL pairs — 6 finite + the where-mask +inf branches.
// ---------------------------------------------------------------------------
#[test]
fn divergence_kl_cb_cb_band() {
    // torch kl.py:255-260.
    close(
        kl_divergence(&cb(0.3), &cb(0.6)).unwrap().item().unwrap(),
        0.064_519_264_453_216_65,
        1e-12,
        "KL(CB(0.3),CB(0.6))",
    );
    // p=0.5 engages Taylor in p.mean / p._cont_bern_log_norm:
    close(
        kl_divergence(&cb(0.5), &cb(0.7)).unwrap().item().unwrap(),
        0.029_736_123_251_875_357,
        1e-12,
        "KL(CB(0.5),CB(0.7))",
    );
}

#[test]
fn divergence_kl_cb_cb_batched_broadcast() {
    // torch: KL(CB([0.3,0.5]),CB([0.6,0.4])) == [0.06451926445321665, 0.006840721103852643]
    let p = ContinuousBernoulli::new(from_slice(&[0.3f64, 0.5], &[2]).unwrap()).unwrap();
    let q = ContinuousBernoulli::new(from_slice(&[0.6f64, 0.4], &[2]).unwrap()).unwrap();
    let kl = kl_divergence(&p, &q).unwrap();
    assert_eq!(kl.shape(), &[2]);
    let d = kl.data().unwrap();
    close(d[0], 0.064_519_264_453_216_65, 1e-12, "KL batched 0");
    close(d[1], 0.006_840_721_103_852_643, 1e-12, "KL batched 1");
    // scalar-p broadcasts against batched-q:
    let p2 = cb(0.3);
    let kl2 = kl_divergence(&p2, &q).unwrap();
    assert_eq!(kl2.shape(), &[2]);
    let d2 = kl2.data().unwrap();
    close(d2[0], 0.064_519_264_453_216_65, 1e-12, "KL bcast 0");
    close(d2[1], 0.007_934_582_218_746_96, 1e-12, "KL bcast 1");
}

#[test]
fn divergence_kl_cb_cross_family_finite() {
    // Beta(2,3)||CB(0.4): kl.py:518-525.
    close(
        kl_divergence(
            &Beta::new(scalar(2.0f64).unwrap(), scalar(3.0f64).unwrap()).unwrap(),
            &cb(0.4),
        )
        .unwrap()
        .item()
        .unwrap(),
        0.201_200_860_081_036_55,
        1e-10,
        "KL(Beta(2,3),CB(0.4))",
    );
    // CB(0.4)||Exp(1.5): kl.py:586-588.
    close(
        kl_divergence(&cb(0.4), &Exponential::new(scalar(1.5f64).unwrap()).unwrap())
            .unwrap()
            .item()
            .unwrap(),
        0.300_812_134_623_041_46,
        1e-12,
        "KL(CB(0.4),Exp(1.5))",
    );
    // CB(0.4)||Normal(0.5,2.0): kl.py:595-604.
    close(
        kl_divergence(
            &cb(0.4),
            &Normal::new(scalar(0.5f64).unwrap(), scalar(2.0f64).unwrap()).unwrap(),
        )
        .unwrap()
        .item()
        .unwrap(),
        1.629_381_291_078_400_5,
        1e-12,
        "KL(CB(0.4),Normal(0.5,2))",
    );
}

#[test]
fn divergence_kl_cb_uniform_where_mask() {
    // CB-Uniform: finite ONLY when low<0 AND high>1 (kl.py:607-617).
    // torch: KL(CB(0.4),Uniform(-1,2)) == 1.105434337834668
    close(
        kl_divergence(
            &cb(0.4),
            &Uniform::new(scalar(-1.0f64).unwrap(), scalar(2.0f64).unwrap()).unwrap(),
        )
        .unwrap()
        .item()
        .unwrap(),
        1.105_434_337_834_668,
        1e-12,
        "KL(CB(0.4),Uniform(-1,2)) finite",
    );
    // low>=0 -> +inf
    let inf_lo = kl_divergence(
        &cb(0.4),
        &Uniform::new(scalar(0.0f64).unwrap(), scalar(2.0f64).unwrap()).unwrap(),
    )
    .unwrap()
    .item()
    .unwrap();
    assert!(
        inf_lo.is_infinite() && inf_lo > 0.0,
        "KL(CB,U(0,2)) must be +inf, got {inf_lo}"
    );
    // high<=1 -> +inf
    let inf_hi = kl_divergence(
        &cb(0.4),
        &Uniform::new(scalar(-1.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap(),
    )
    .unwrap()
    .item()
    .unwrap();
    assert!(
        inf_hi.is_infinite() && inf_hi > 0.0,
        "KL(CB,U(-1,1)) must be +inf, got {inf_hi}"
    );
}

#[test]
fn divergence_kl_uniform_cb_where_mask() {
    // Uniform-CB: finite ONLY when low>0 AND high<1 (kl.py:871-886).
    // torch: KL(Uniform(0.2,0.8),CB(0.4)) == 0.5176663448698431
    close(
        kl_divergence(
            &Uniform::new(scalar(0.2f64).unwrap(), scalar(0.8f64).unwrap()).unwrap(),
            &cb(0.4),
        )
        .unwrap()
        .item()
        .unwrap(),
        0.517_666_344_869_843_1,
        1e-12,
        "KL(Uniform(0.2,0.8),CB(0.4)) finite",
    );
    // low<=0 -> +inf
    let inf_lo = kl_divergence(
        &Uniform::new(scalar(0.0f64).unwrap(), scalar(0.8f64).unwrap()).unwrap(),
        &cb(0.4),
    )
    .unwrap()
    .item()
    .unwrap();
    assert!(
        inf_lo.is_infinite() && inf_lo > 0.0,
        "KL(U(0,0.8),CB) must be +inf, got {inf_lo}"
    );
    // high>=1 -> +inf
    let inf_hi = kl_divergence(
        &Uniform::new(scalar(0.2f64).unwrap(), scalar(1.0f64).unwrap()).unwrap(),
        &cb(0.4),
    )
    .unwrap()
    .item()
    .unwrap();
    assert!(
        inf_hi.is_infinite() && inf_hi > 0.0,
        "KL(U(0.2,1),CB) must be +inf, got {inf_hi}"
    );
}
