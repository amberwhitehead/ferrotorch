//! Re-audit supplement for commit `8b876ad34` (#1635/#1636).
//!
//! `divergence_dropout_multidim_feature_alpha_1635_1636.rs` already pins the
//! per-channel flat-`[N,C]` draw order + spatial broadcast on multi-dim inputs
//! and the alpha affine across p; all 9 of its assertions PASS against
//! live-torch-2.11-confirmed values (independently re-derived in this audit).
//!
//! This file closes the residual gaps named in the re-audit charter that the
//! existing suite did not cover:
//!   1. Dropout3d on `[1,2,2,2,2]` (charter shape — distinct from the existing
//!      `[2,3,2,2,2]` case).
//!   2. FeatureAlphaDropout on `[1,4,2,2]` (charter multi-spatial shape).
//!   3. Determinism: two `manual_seed(42)` + Dropout2d runs byte-identical
//!      (the pre-commit impl used a system-time xorshift; this regression-guards
//!      the MT19937 rewire).
//!   4. Plain `F.dropout` (#1634) still byte-matches under seed (no regression
//!      from the feature/alpha rewire).
//!
//! ALL reference values produced by LIVE torch 2.11.0+cu130 via
//! `/tmp/dropout_oracle.py` in this audit session (R-CHAR-3):
//!   torch.manual_seed(s); F.dropout3d / nn.FeatureAlphaDropout(p).train() /
//!   F.dropout on the exact arange/ones inputs reconstructed below.

use ferrotorch_core::{Tensor, TensorStorage};
use ferrotorch_nn::{Dropout, Dropout2d, Dropout3d, FeatureAlphaDropout, Module};

fn arange(shape: Vec<usize>) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|i| (i as f32) + 1.0).collect();
    Tensor::<f32>::from_storage(TensorStorage::cpu(data), shape, false).unwrap()
}

fn ones(n: usize) -> Tensor<f32> {
    Tensor::<f32>::from_storage(TensorStorage::cpu(vec![1.0f32; n]), vec![n], false).unwrap()
}

fn approx(got: &[f32], want: &[f32], tol: f32, ctx: &str) {
    assert_eq!(got.len(), want.len(), "{ctx}: length mismatch");
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() <= tol,
            "{ctx}: element {i} got {g} want {w}\n got={got:?}\nwant={want:?}"
        );
    }
}

/// LIVE torch: torch.manual_seed(1); F.dropout3d(arange(1,2,2,2,2)+1,0.5,True).
/// N=1,C=2,spatial=8. Both channels survive at this seed -> scale 2.0.
#[test]
fn dropout3d_1x2x2x2x2_arange_seed1_p05_matches_torch() {
    let want: [f32; 16] = [
        2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0,
    ];
    ferrotorch_core::rng::manual_seed(1);
    let layer = Dropout3d::<f32>::new(0.5).unwrap();
    let y = layer.forward(&arange(vec![1, 2, 2, 2, 2])).unwrap();
    approx(
        &y.data_vec().unwrap(),
        &want,
        1e-3,
        "Dropout3d [1,2,2,2,2] arange seed1 p0.5",
    );
}

/// LIVE torch: torch.manual_seed(7); nn.FeatureAlphaDropout(0.5).train()(arange(1,4,2,2)+1).
/// N=1,C=4,spatial=4. Keep pattern over flat [N,C]=4: [keep,keep,DROP,DROP].
/// Kept -> a*x+alpha*a*p; dropped -> constant -0.7791939 over the whole slice.
#[test]
fn feature_alpha_dropout_1x4x2x2_arange_seed7_p05_matches_torch() {
    let want: [f32; 16] = [
        1.6655989, 2.5520036, 3.4384084, 4.3248134, // ch0 keep
        5.2112184, 6.0976229, 6.9840279, 7.8704329, // ch1 keep
        -0.7791939, -0.7791939, -0.7791939, -0.7791939, // ch2 DROP
        -0.7791939, -0.7791939, -0.7791939, -0.7791939, // ch3 DROP
    ];
    ferrotorch_core::rng::manual_seed(7);
    let layer = FeatureAlphaDropout::<f32>::new(0.5).unwrap();
    let y = layer.forward(&arange(vec![1, 4, 2, 2])).unwrap();
    approx(
        &y.data_vec().unwrap(),
        &want,
        2e-4,
        "FeatureAlphaDropout [1,4,2,2] arange seed7 p0.5",
    );
}

/// Determinism: two manual_seed(42)+Dropout2d forwards must be byte-identical
/// (the pre-8b876ad34 impl drew from a system-time xorshift, so reseeding did
/// NOT reproduce). Regression-guards the MT19937 rewire.
#[test]
fn dropout2d_manual_seed_is_deterministic() {
    let layer = Dropout2d::<f32>::new(0.5).unwrap();

    ferrotorch_core::rng::manual_seed(42);
    let y1 = layer
        .forward(&arange(vec![2, 4, 3, 3]))
        .unwrap()
        .data_vec()
        .unwrap();

    ferrotorch_core::rng::manual_seed(42);
    let y2 = layer
        .forward(&arange(vec![2, 4, 3, 3]))
        .unwrap()
        .data_vec()
        .unwrap();

    approx(&y1, &y2, 0.0, "Dropout2d seed42 determinism (run1 vs run2)");
}

/// Regression #1634: plain F.dropout under seed still byte-matches torch.
/// LIVE torch: torch.manual_seed(42); F.dropout(ones(10),0.5,True).
/// Per-ELEMENT (not per-channel) Bernoulli stream, keep iff u<0.5, scale 2.0.
#[test]
fn plain_dropout_seed42_p05_no_regression_matches_torch() {
    let want: [f32; 10] = [2.0, 2.0, 2.0, 2.0, 0.0, 2.0, 0.0, 0.0, 2.0, 2.0];
    ferrotorch_core::rng::manual_seed(42);
    let layer = Dropout::<f32>::new(0.5).unwrap();
    let y = layer.forward(&ones(10)).unwrap();
    approx(
        &y.data_vec().unwrap(),
        &want,
        1e-3,
        "plain Dropout [10] ones seed42 p0.5 (#1634 regression)",
    );
}
