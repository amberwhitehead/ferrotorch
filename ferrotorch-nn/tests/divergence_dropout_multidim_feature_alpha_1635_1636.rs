//! Re-audit of commit `8b876ad34` (#1635/#1636 — per-channel + alpha dropout
//! via byte-exact MT19937).
//!
//! The builder's existing seed-42 tests in
//! `divergence_dropout_seed_extended_and_feature_1634.rs` only exercise
//! TRIVIAL feature shapes: `Dropout2d` on `[1,8,1,1]`, `Dropout1d` on
//! `[1,6,3]` (uniform `ones`), `Dropout3d` on `[1,6,1,1,1]`,
//! `FeatureAlphaDropout` on `[1,6,1,1]`. Every one has N=1 and either a
//! single-element spatial extent or a uniform `ones` input, so a WRONG draw
//! order (per-element instead of per-channel, C-major instead of N-major, or a
//! broadcast that does not tile the spatial volume contiguously) would still
//! pass them.
//!
//! This file pins the per-channel draw order + spatial broadcast on REAL
//! multi-dim inputs (N>1 AND C>1 AND spatial>1, with an `arange` input so each
//! element is distinct), and the alpha affine across MULTIPLE p (the builder
//! only verified p=0.5).
//!
//! Upstream contract (`aten/src/ATen/native/Dropout.cpp:30-41,73-79`):
//!   - `make_feature_noise(input)` -> `[N, C, 1, 1...]`; `noise.bernoulli_(1-p)`
//!     draws exactly N*C values in flat `[N, C]` (N-major) order; broadcast
//!     over the trailing spatial volume; survivors scaled `1/(1-p)`.
//!   - alpha: `alpha=1.7580993408473766`, `a=1/sqrt((alpha^2 p+1)(1-p))`,
//!     kept -> `a*x+alpha*a*p`, dropped -> `-alpha*a+alpha*a*p`.
//!
//! ALL reference values produced by LIVE torch 2.11.0+cu130 (R-CHAR-3):
//!   torch.manual_seed(s); F.dropout2d/1d/3d / nn.AlphaDropout(p).train() /
//!   nn.FeatureAlphaDropout(p).train(), on the exact `arange`/`ones` inputs
//!   reconstructed below.

#![allow(clippy::excessive_precision)]

use ferrotorch_core::{Tensor, TensorStorage};
use ferrotorch_nn::Module;
use std::sync::{Mutex, MutexGuard};

fn dropout_rng_lock() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn arange(shape: Vec<usize>) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|i| (i as f32) + 1.0).collect();
    Tensor::<f32>::from_storage(TensorStorage::cpu(data), shape, false).unwrap()
}

fn ones_shape(shape: Vec<usize>) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    Tensor::<f32>::from_storage(TensorStorage::cpu(vec![1.0f32; n]), shape, false).unwrap()
}

fn approx(got: &[f32], want: &[f32], tol: f32, ctx: &str) {
    assert_eq!(got.len(), want.len(), "{ctx}: length mismatch");
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() <= tol,
            "{ctx}: element {i} got {g} want {w} (tol {tol})\n got={got:?}\nwant={want:?}"
        );
    }
}

// ===========================================================================
// Dropout2d — N>1, C>1, spatial>1, distinct (arange) input.
//
// This is the case the builder's [1,8,1,1] test cannot distinguish from a
// per-element or transposed draw order.
// ===========================================================================

/// torch.manual_seed(42); F.dropout2d(arange(2,4,3,3)+1, 0.5, True).
/// N=2, C=4, spatial=9. Surviving channels scaled by 2.0; the per-channel
/// keep pattern over the flat [N,C]=8-element stream (seed 42, p=0.5) is
/// [keep,keep,keep,keep, DROP,keep,DROP,DROP] broadcast over each 3x3 slice.
#[test]
fn dropout2d_multidim_arange_seed42_p05_matches_torch() {
    let _rng = dropout_rng_lock();
    let want: [f32; 72] = [
        2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, // ch0 keep
        20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0, 36.0, // ch1 keep
        38.0, 40.0, 42.0, 44.0, 46.0, 48.0, 50.0, 52.0, 54.0, // ch2 keep
        56.0, 58.0, 60.0, 62.0, 64.0, 66.0, 68.0, 70.0, 72.0, // ch3 keep
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // ch4 DROP
        92.0, 94.0, 96.0, 98.0, 100.0, 102.0, 104.0, 106.0, 108.0, // ch5 keep
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // ch6 DROP
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // ch7 DROP
    ];
    ferrotorch_core::rng::manual_seed(42);
    let layer = ferrotorch_nn::Dropout2d::<f32>::new(0.5).unwrap();
    let y = layer.forward(&arange(vec![2, 4, 3, 3])).unwrap();
    approx(
        &y.data_vec().unwrap(),
        &want,
        1e-3,
        "Dropout2d [2,4,3,3] arange seed42 p0.5",
    );
}

/// torch.manual_seed(7); F.dropout2d(arange(2,4,3,3)+1, 0.3, True).
/// Scale 1/(1-0.3)=1.4285714. Keep pattern over flat [N,C]=8 stream:
/// [keep,keep,DROP,keep,DROP,DROP,keep,keep].
#[test]
fn dropout2d_multidim_arange_seed7_p03_matches_torch() {
    let _rng = dropout_rng_lock();
    let want: [f32; 72] = [
        1.428571, 2.857143, 4.285714, 5.714286, 7.142858, 8.571428, 10.0, 11.428572, 12.857143,
        14.285715, 15.714286, 17.142857, 18.571428, 20.0, 21.428572, 22.857143, 24.285715,
        25.714287, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 40.0, 41.428574, 42.857143,
        44.285717, 45.714287, 47.142857, 48.57143, 50.0, 51.428574, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 78.571434, 80.0, 81.428574,
        82.857147, 84.285713, 85.714287, 87.14286, 88.571434, 90.0, 91.428574, 92.857147,
        94.285713, 95.714287, 97.14286, 98.571434, 100.0, 101.428574, 102.857147,
    ];
    ferrotorch_core::rng::manual_seed(7);
    let layer = ferrotorch_nn::Dropout2d::<f32>::new(0.3).unwrap();
    let y = layer.forward(&arange(vec![2, 4, 3, 3])).unwrap();
    approx(
        &y.data_vec().unwrap(),
        &want,
        1e-3,
        "Dropout2d [2,4,3,3] arange seed7 p0.3",
    );
}

/// Per-channel first-element pattern for [3,5,2,2] ones, seed=123, p=0.5.
/// Pins the N-major draw order over 15 channels (N=3,C=5). Live torch
/// per-channel scaled value (first spatial element of each channel):
#[test]
fn dropout2d_3x5_ones_seed123_p05_per_channel_matches_torch() {
    let _rng = dropout_rng_lock();
    // torch.manual_seed(123); F.dropout2d(ones(3,5,2,2),0.5,True)[:,:,0,0]
    let want_per_chan: [f32; 15] = [
        2.0, 2.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 2.0, 2.0,
    ];
    ferrotorch_core::rng::manual_seed(123);
    let layer = ferrotorch_nn::Dropout2d::<f32>::new(0.5).unwrap();
    let y = layer.forward(&ones_shape(vec![3, 5, 2, 2])).unwrap();
    let data = y.data_vec().unwrap();
    // first spatial element of each 2x2 channel slice
    let per_chan: Vec<f32> = (0..15).map(|c| data[c * 4]).collect();
    approx(
        &per_chan,
        &want_per_chan,
        1e-3,
        "Dropout2d [3,5,2,2] ones seed123 p0.5 per-channel",
    );
}

// ===========================================================================
// Dropout1d / Dropout3d — N>1, C>1, spatial>1, arange.
// ===========================================================================

/// torch.manual_seed(42); F.dropout1d(arange(2,3,4)+1, 0.5, True).
/// N=2,C=3,L=4. Keep pattern over flat [N,C]=6 stream: [k,k,k,k,DROP,k].
#[test]
fn dropout1d_multidim_arange_seed42_p05_matches_torch() {
    let _rng = dropout_rng_lock();
    let want: [f32; 24] = [
        2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0,
        0.0, 0.0, 0.0, 0.0, 42.0, 44.0, 46.0, 48.0,
    ];
    ferrotorch_core::rng::manual_seed(42);
    let layer = ferrotorch_nn::Dropout1d::<f32>::new(0.5).unwrap();
    let y = layer.forward(&arange(vec![2, 3, 4])).unwrap();
    approx(
        &y.data_vec().unwrap(),
        &want,
        1e-3,
        "Dropout1d [2,3,4] arange seed42 p0.5",
    );
}

/// torch.manual_seed(42); F.dropout3d(arange(2,3,2,2,2)+1, 0.5, True).
/// N=2,C=3,spatial=8. Keep pattern over flat [N,C]=6: [k,k,k,k,DROP,k].
#[test]
fn dropout3d_multidim_arange_seed42_p05_matches_torch() {
    let _rng = dropout_rng_lock();
    let want: [f32; 48] = [
        2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0,
        34.0, 36.0, 38.0, 40.0, 42.0, 44.0, 46.0, 48.0, 50.0, 52.0, 54.0, 56.0, 58.0, 60.0, 62.0,
        64.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 82.0, 84.0, 86.0, 88.0, 90.0, 92.0, 94.0,
        96.0,
    ];
    ferrotorch_core::rng::manual_seed(42);
    let layer = ferrotorch_nn::Dropout3d::<f32>::new(0.5).unwrap();
    let y = layer.forward(&arange(vec![2, 3, 2, 2, 2])).unwrap();
    approx(
        &y.data_vec().unwrap(),
        &want,
        1e-3,
        "Dropout3d [2,3,2,2,2] arange seed42 p0.5",
    );
}

// ===========================================================================
// AlphaDropout — affine across MULTIPLE p (builder only verified p=0.5).
// kept = a*x + alpha*a*p, dropped = -alpha*a + alpha*a*p, with
// alpha=1.7580993408473766, a=1/sqrt((alpha^2 p+1)(1-p)).
// ===========================================================================

/// torch.manual_seed(42); nn.AlphaDropout(p).train()(ones(10)) for p in
/// {0.1, 0.3, 0.75}. Both the keep/drop SET and the kept/dropped magnitudes
/// must match torch for each p.
#[test]
fn alpha_dropout_affine_across_p_matches_torch() {
    let _rng = dropout_rng_lock();
    let cases: &[(f64, [f32; 10])] = &[
        (
            0.1,
            [
                1.0832555, 1.0832555, 1.0832555, 1.0832555, 1.0832555, 1.0832555, -1.4577388,
                -1.4577388, 1.0832555, 1.0832555,
            ],
        ),
        (
            0.3,
            [
                1.3150446, 1.3150446, 1.3150446, 1.3150446, 1.3150446, 1.3150446, -1.0595483,
                -1.0595483, 1.3150446, 1.3150446,
            ],
        ),
        (
            0.75,
            [
                2.5456619, 2.5456619, 2.5456619, 2.5456619, -0.4825732, -0.4825732, -0.4825732,
                -0.4825732, 2.5456619, 2.5456619,
            ],
        ),
    ];
    for (p, want) in cases {
        ferrotorch_core::rng::manual_seed(42);
        let layer = ferrotorch_nn::AlphaDropout::<f32>::new(*p).unwrap();
        let x = Tensor::<f32>::from_storage(TensorStorage::cpu(vec![1.0f32; 10]), vec![10], false)
            .unwrap();
        let y = layer.forward(&x).unwrap();
        approx(
            &y.data_vec().unwrap(),
            want,
            2e-4,
            &format!("AlphaDropout p={p} seed42 ones(10)"),
        );
    }
}

// ===========================================================================
// FeatureAlphaDropout — multi-dim per-channel alpha (N>1,C>1,spatial>1).
// Builder only verified [1,6,1,1]. Here channels actually DROP and the
// affine + per-channel broadcast both matter.
// ===========================================================================

/// torch.manual_seed(7); nn.FeatureAlphaDropout(0.5).train()(arange(2,3,2,2)+1).
/// N=2,C=3,spatial=4. Keep pattern over flat [N,C]=6: [keep,keep,DROP,DROP,DROP,DROP].
/// Kept channels -> a*x+alpha*a*p (a varies with x within the channel);
/// dropped channels -> constant -0.7791939 across the whole 2x2 slice.
#[test]
fn feature_alpha_dropout_multidim_arange_seed7_p05_matches_torch() {
    let _rng = dropout_rng_lock();
    let want: [f32; 24] = [
        1.6655989, 2.5520036, 3.4384084, 4.3248134, // ch0 keep (x=1..4)
        5.2112184, 6.0976229, 6.9840279, 7.8704329, // ch1 keep (x=5..8)
        -0.7791939, -0.7791939, -0.7791939, -0.7791939, // ch2 DROP
        -0.7791939, -0.7791939, -0.7791939, -0.7791939, // ch3 DROP
        -0.7791939, -0.7791939, -0.7791939, -0.7791939, // ch4 DROP
        -0.7791939, -0.7791939, -0.7791939, -0.7791939, // ch5 DROP
    ];
    ferrotorch_core::rng::manual_seed(7);
    let layer = ferrotorch_nn::FeatureAlphaDropout::<f32>::new(0.5).unwrap();
    let y = layer.forward(&arange(vec![2, 3, 2, 2])).unwrap();
    approx(
        &y.data_vec().unwrap(),
        &want,
        2e-4,
        "FeatureAlphaDropout [2,3,2,2] arange seed7 p0.5",
    );
}

/// torch.manual_seed(42); nn.FeatureAlphaDropout(0.3).train()(arange(2,3,2,2)+1).
/// At this seed/p NO channel drops: all kept -> a*x + alpha*a*p with the p=0.3
/// affine. Pins the kept-affine on multi-dim (distinct from the dropped case).
#[test]
fn feature_alpha_dropout_multidim_arange_seed42_p03_matches_torch() {
    let _rng = dropout_rng_lock();
    let want: [f32; 24] = [
        1.3150446, 2.1759973, 3.0369499, 3.8979025, 4.7588549, 5.6198077, 6.4807606, 7.341713,
        8.2026653, 9.0636177, 9.924571, 10.7855234, 11.6464758, 12.5074291, 13.3683815, 14.2293339,
        15.0902863, 15.9512386, 16.812191, 17.6731434, 18.5340977, 19.39505, 20.2560024,
        21.1169548,
    ];
    ferrotorch_core::rng::manual_seed(42);
    let layer = ferrotorch_nn::FeatureAlphaDropout::<f32>::new(0.3).unwrap();
    let y = layer.forward(&arange(vec![2, 3, 2, 2])).unwrap();
    approx(
        &y.data_vec().unwrap(),
        &want,
        2e-4,
        "FeatureAlphaDropout [2,3,2,2] arange seed42 p0.3",
    );
}

// ===========================================================================
// Eval / p=0 edges for the per-channel + alpha variants.
// ===========================================================================

#[test]
fn feature_and_alpha_eval_and_p0_are_identity() {
    let x = arange(vec![2, 3, 2, 2]);
    let want = x.data_vec().unwrap();

    // eval mode identity
    let mut d2 = ferrotorch_nn::Dropout2d::<f32>::new(0.5).unwrap();
    d2.eval();
    approx(
        &d2.forward(&x).unwrap().data_vec().unwrap(),
        &want,
        0.0,
        "Dropout2d eval",
    );

    let mut fad = ferrotorch_nn::FeatureAlphaDropout::<f32>::new(0.5).unwrap();
    fad.eval();
    approx(
        &fad.forward(&x).unwrap().data_vec().unwrap(),
        &want,
        0.0,
        "FAD eval",
    );

    let mut ad = ferrotorch_nn::AlphaDropout::<f32>::new(0.5).unwrap();
    ad.eval();
    let x1 = ones_shape(vec![10]);
    approx(
        &ad.forward(&x1).unwrap().data_vec().unwrap(),
        &[1.0f32; 10],
        0.0,
        "AlphaDropout eval",
    );

    // p=0 identity in training mode
    let d2_p0 = ferrotorch_nn::Dropout2d::<f32>::new(0.0).unwrap();
    approx(
        &d2_p0.forward(&x).unwrap().data_vec().unwrap(),
        &want,
        0.0,
        "Dropout2d p0",
    );
    let ad_p0 = ferrotorch_nn::AlphaDropout::<f32>::new(0.0).unwrap();
    approx(
        &ad_p0.forward(&x1).unwrap().data_vec().unwrap(),
        &[1.0f32; 10],
        0.0,
        "AlphaDropout p0",
    );
}
