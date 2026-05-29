//! Re-audit of commit `5b30b427d` (#1634 — dropout CPU mask via byte-exact
//! MT19937).
//!
//! The commit rewired the PLAIN `functional::dropout` / `Dropout` CPU mask
//! through `ferrotorch_core::rng::Generator` (byte-exact torch-CPU MT19937)
//! with torch's exact consumption: one `next_uniform_f64()` per element in
//! flat order, keep iff `u < (1 - p)`, survivors scaled by `1/(1-p)`
//! (`aten/src/ATen/native/Dropout.cpp:74` `noise.bernoulli_(1 - p)`,
//! `:81` `noise.div_(1 - p)`).
//!
//! This file does two things:
//!
//!   (A) PUSHES the plain-dropout reproducibility claim past the builder's
//!       seed-0/1, p=0.5, N=10 spot-check: more seeds (42, 123), more p
//!       (0.1, 0.75, 0.9), a non-round size (N=7), and a 2-D input to pin the
//!       flat draw order. These are expected to PASS (regression coverage for
//!       #1634's core claim).
//!
//!   (B) PINS the spillover the builder flagged: the per-channel feature
//!       variants `Dropout1d` / `Dropout2d` / `Dropout3d` and the alpha
//!       variants `AlphaDropout` / `FeatureAlphaDropout` STILL draw their mask
//!       from a system-time-seeded `xorshift64` (`dropout.rs::xorshift_seed` /
//!       `xorshift_next`) with the wrong-direction comparison
//!       `xorshift_next(..) >= self.p` — the SAME wrong-comparison + non-MT19937
//!       class that #1634/#1452 fixed for plain dropout. These tests are
//!       expected to FAIL and are `#[ignore]`'d against a fresh tracking issue.
//!
//! Upstream feature/alpha dropout share the plain-dropout MT19937 stream:
//! `_feature_dropout` calls `_dropout_impl<feature=true>` which draws
//! `make_feature_noise(input).bernoulli_(1 - p)` over the reduced `[N, C, 1..]`
//! tensor in flat `[N, C]` order (`aten/src/ATen/native/Dropout.cpp:30-41,73-74`).
//! `_alpha_dropout` uses `alpha = 1.7580993408473766` and the affine
//! correction `a = 1/sqrt((alpha^2 p + 1)(1-p))`,
//! `out_kept = a + alpha*a*p`, `out_dropped = -alpha*a + alpha*a*p`
//! (`Dropout.cpp:76-79`).
//!
//! ALL reference values below produced by LIVE torch 2.11.0+cu130 (R-CHAR-3 —
//! NOT copied from the ferrotorch side):
//!   torch.manual_seed(s); F.dropout / F.dropout2d / F.dropout1d /
//!   F.dropout3d / nn.AlphaDropout(p).train() / nn.FeatureAlphaDropout(p).train()

use ferrotorch_core::{Tensor, TensorStorage};
use ferrotorch_nn::Module;

fn ones(n: usize) -> Tensor<f32> {
    Tensor::<f32>::from_storage(TensorStorage::cpu(vec![1.0f32; n]), vec![n], false).unwrap()
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
// (A) Plain dropout — extended seeds / p / sizes / draw-order. EXPECTED PASS.
// ===========================================================================

/// KEY (requested): seed=42, p=0.75, N=10. Live torch:
///   torch.manual_seed(42); F.dropout(torch.ones(10), p=0.75, training=True)
///     -> [4, 4, 4, 4, 0, 0, 0, 0, 4, 4]   (survivors scaled by 1/(1-0.75)=4)
#[test]
fn plain_dropout_seed42_p075_matches_torch() {
    let torch_seed42_p075: [f32; 10] =
        [4.0, 4.0, 4.0, 4.0, 0.0, 0.0, 0.0, 0.0, 4.0, 4.0];
    ferrotorch_core::rng::manual_seed(42);
    let y = ferrotorch_nn::functional::dropout(&ones(10), 0.75, true).unwrap();
    approx(&y.data_vec().unwrap(), &torch_seed42_p075, 1e-5, "seed42 p0.75 N10");
}

/// Multiple seeds × multiple p, plain dropout. Live torch references.
#[test]
fn plain_dropout_seed_p_grid_matches_torch() {
    // (seed, p, expected) from live torch 2.11.
    let cases: &[(u64, f64, &[f32])] = &[
        (0, 0.1, &[0.0, 1.1111112, 1.1111112, 0.0, 1.1111112, 1.1111112, 1.1111112, 1.1111112, 1.1111112, 1.1111112]),
        (0, 0.75, &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0]),
        (1, 0.75, &[4.0, 4.0, 4.0, 4.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0]),
        (1, 0.9, &[10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        (42, 0.5, &[2.0, 2.0, 2.0, 2.0, 0.0, 2.0, 0.0, 0.0, 2.0, 2.0]),
        (42, 0.9, &[10.0, 10.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0]),
        (123, 0.5, &[2.0, 2.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0]),
        (123, 0.75, &[0.0, 4.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0]),
    ];
    for &(s, p, want) in cases {
        ferrotorch_core::rng::manual_seed(s);
        let y = ferrotorch_nn::functional::dropout(&ones(10), p, true).unwrap();
        approx(&y.data_vec().unwrap(), want, 1e-5, &format!("seed={s} p={p}"));
    }
}

/// Non-round size N=7 (no SIMD-friendly length). Live torch:
///   torch.manual_seed(42); F.dropout(ones(7), 0.5, True) -> [2,2,2,2,0,2,0]
#[test]
fn plain_dropout_nonround_size_matches_torch() {
    let want: [f32; 7] = [2.0, 2.0, 2.0, 2.0, 0.0, 2.0, 0.0];
    ferrotorch_core::rng::manual_seed(42);
    let y = ferrotorch_nn::functional::dropout(&ones(7), 0.5, true).unwrap();
    approx(&y.data_vec().unwrap(), &want, 1e-5, "seed42 p0.5 N7");
}

/// 2-D input draws in flat row-major order, identical to the flat ones(10):
///   torch.manual_seed(42); F.dropout(ones(2,5),0.5,True)
///     -> [[2,2,2,2,0],[2,0,0,2,2]]  == flat [2,2,2,2,0,2,0,0,2,2]
#[test]
fn plain_dropout_2d_draw_order_matches_torch() {
    let want: [f32; 10] = [2.0, 2.0, 2.0, 2.0, 0.0, 2.0, 0.0, 0.0, 2.0, 2.0];
    ferrotorch_core::rng::manual_seed(42);
    let y = ferrotorch_nn::functional::dropout(&ones_shape(vec![2, 5]), 0.5, true).unwrap();
    approx(&y.data_vec().unwrap(), &want, 1e-5, "seed42 p0.5 shape[2,5]");
}

/// Plain `Dropout` MODULE, seed=42 p=0.75 — guards the `<Dropout as
/// Module>::forward` CPU branch (distinct from `functional::dropout`).
#[test]
fn plain_dropout_module_seed42_p075_matches_torch() {
    let want: [f32; 10] = [4.0, 4.0, 4.0, 4.0, 0.0, 0.0, 0.0, 0.0, 4.0, 4.0];
    ferrotorch_core::rng::manual_seed(42);
    let layer = ferrotorch_nn::Dropout::<f32>::new(0.75).unwrap();
    let y = layer.forward(&ones(10)).unwrap();
    approx(&y.data_vec().unwrap(), &want, 1e-5, "module seed42 p0.75");
}

// ===========================================================================
// (A2) Deterministic edges — EXPECTED PASS (unregressed by #1634).
// ===========================================================================

/// training=False is identity.
#[test]
fn plain_dropout_eval_is_identity() {
    let x = ones(10);
    let y = ferrotorch_nn::functional::dropout(&x, 0.5, false).unwrap();
    assert_eq!(y.data_vec().unwrap(), vec![1.0f32; 10]);
}

/// p=0 is identity even in training mode.
#[test]
fn plain_dropout_p0_is_identity() {
    let x = ones(10);
    let y = ferrotorch_nn::functional::dropout(&x, 0.0, true).unwrap();
    assert_eq!(y.data_vec().unwrap(), vec![1.0f32; 10]);
}

// ===========================================================================
// (B) Per-channel feature dropout — EXPECTED FAIL (xorshift + wrong RNG).
//
//     ferrotorch `Dropout2d`/`1d`/`3d`/`AlphaDropout`/`FeatureAlphaDropout`
//     draw the channel mask from `dropout.rs::xorshift_seed()`
//     (`SystemTime::now()` + thread id) with `xorshift_next(..) >= self.p`,
//     IGNORING `ferrotorch_core::manual_seed` and NOT matching torch's MT19937
//     stream. Upstream feature dropout shares the plain-dropout MT19937 stream
//     (`Dropout.cpp:73-74`), so a seeded run is reproducible AND equals the
//     first `N*C` plain-dropout draws.
// ===========================================================================

/// Divergence: ferrotorch's `Dropout2d::forward` diverges from
/// `pytorch aten/src/ATen/native/Dropout.cpp:73-74`
/// (`make_feature_noise(input).bernoulli_(1 - p)`) for a seeded input.
/// Upstream (seed=42, p=0.5, 8 channels) keeps channels per the MT19937 flat
/// stream -> per-channel scaled vals [2,2,2,2,0,2,0,0]; ferrotorch draws from a
/// system-time xorshift64 (`dropout.rs::xorshift_seed`) with `>= p`, so the
/// result is non-reproducible and does not equal torch's seeded mask.
/// Tracking: #1635
#[test]
#[ignore = "divergence: Dropout2d uses system-time xorshift + keep-on(u>=p), not seeded MT19937; tracking #1635"]
fn dropout2d_seed42_per_channel_matches_torch() {
    // torch.manual_seed(42); F.dropout2d(ones(1,8,1,1),0.5,True), per-channel val.
    let want: [f32; 8] = [2.0, 2.0, 2.0, 2.0, 0.0, 2.0, 0.0, 0.0];
    ferrotorch_core::rng::manual_seed(42);
    let layer = ferrotorch_nn::Dropout2d::<f32>::new(0.5).unwrap();
    let y = layer.forward(&ones_shape(vec![1, 8, 1, 1])).unwrap();
    // collapse [1,8,1,1] -> per-channel (each channel is a single element here)
    approx(&y.data_vec().unwrap(), &want, 1e-5, "Dropout2d seed42 p0.5 8chan");
}

/// Divergence: ferrotorch's `Dropout1d::forward` diverges from upstream feature
/// dropout for a seeded input. Upstream (seed=42, p=0.5, 6 channels) ->
/// per-channel scaled vals [2,2,2,2,0,2]; ferrotorch uses system-time xorshift.
/// Tracking: #1635
#[test]
#[ignore = "divergence: Dropout1d uses system-time xorshift + keep-on(u>=p), not seeded MT19937; tracking #1635"]
fn dropout1d_seed42_per_channel_matches_torch() {
    // torch.manual_seed(42); F.dropout1d(ones(1,6,3),0.5,True), per-channel val.
    let want: [f32; 6] = [2.0, 2.0, 2.0, 2.0, 0.0, 2.0];
    ferrotorch_core::rng::manual_seed(42);
    let layer = ferrotorch_nn::Dropout1d::<f32>::new(0.5).unwrap();
    let y = layer.forward(&ones_shape(vec![1, 6, 3])).unwrap();
    // first element of each length-3 channel
    let data = y.data_vec().unwrap();
    let per_chan: Vec<f32> = (0..6).map(|c| data[c * 3]).collect();
    approx(&per_chan, &want, 1e-5, "Dropout1d seed42 p0.5 6chan");
}

/// Divergence: ferrotorch's `Dropout3d::forward` diverges from upstream feature
/// dropout for a seeded input. Upstream (seed=42, p=0.5, 6 channels) ->
/// per-channel scaled vals [2,2,2,2,0,2]; ferrotorch uses system-time xorshift.
/// Tracking: #1635
#[test]
#[ignore = "divergence: Dropout3d uses system-time xorshift + keep-on(u>=p), not seeded MT19937; tracking #1635"]
fn dropout3d_seed42_per_channel_matches_torch() {
    // torch.manual_seed(42); F.dropout3d(ones(1,6,1,1,1),0.5,True), per-channel.
    let want: [f32; 6] = [2.0, 2.0, 2.0, 2.0, 0.0, 2.0];
    ferrotorch_core::rng::manual_seed(42);
    let layer = ferrotorch_nn::Dropout3d::<f32>::new(0.5).unwrap();
    let y = layer.forward(&ones_shape(vec![1, 6, 1, 1, 1])).unwrap();
    approx(&y.data_vec().unwrap(), &want, 1e-5, "Dropout3d seed42 p0.5 6chan");
}

/// Divergence (reproducibility, RNG-agnostic): two consecutive seeded
/// `Dropout2d` forwards under the SAME `manual_seed(42)` must produce the SAME
/// mask (torch's MT19937 is reset by manual_seed). ferrotorch's
/// `xorshift_seed()` reseeds from `SystemTime::now()` each call, so two seeded
/// runs differ. This pins the non-determinism WITHOUT depending on the exact
/// torch byte pattern — it should hold for ANY correct seeded RNG.
/// Tracking: #1635
#[test]
#[ignore = "divergence: Dropout2d not reproducible under manual_seed (system-time xorshift); tracking #1635"]
fn dropout2d_reproducible_under_manual_seed() {
    let layer = ferrotorch_nn::Dropout2d::<f32>::new(0.5).unwrap();

    ferrotorch_core::rng::manual_seed(42);
    let y1 = layer.forward(&ones_shape(vec![1, 64, 1, 1])).unwrap();
    ferrotorch_core::rng::manual_seed(42);
    let y2 = layer.forward(&ones_shape(vec![1, 64, 1, 1])).unwrap();

    assert_eq!(
        y1.data_vec().unwrap(),
        y2.data_vec().unwrap(),
        "Dropout2d under the same manual_seed must reproduce the same channel \
         mask (torch resets MT19937 on manual_seed); ferrotorch reseeds from \
         SystemTime each call"
    );
}

// ===========================================================================
// (B2) Alpha dropout — EXPECTED FAIL (wrong RNG AND wrong SELU constant/math).
// ===========================================================================

/// Divergence: ferrotorch's `AlphaDropout::forward` diverges from
/// `pytorch aten/src/ATen/native/Dropout.cpp:76-79` on TWO axes:
///  (1) RNG — system-time xorshift, not seeded MT19937 (so non-reproducible).
///  (2) MATH — torch uses `alpha = 1.7580993408473766` with
///      `a = 1/sqrt((alpha^2 p + 1)(1-p))`, kept = a + alpha*a*p,
///      dropped = -alpha*a + alpha*a*p. ferrotorch uses
///      `alpha' = -SELU_LAMBDA*SELU_ALPHA` (= -1.6732632*1.0507009) and a
///      different affine formula, so even the kept/dropped *magnitudes* differ.
/// Upstream (seed=42, p=0.5, ones(10)) -> kept=1.66559887, dropped=-0.77919394
/// in the MT19937 keep pattern [keep,keep,keep,keep,DROP,keep,DROP,DROP,keep,keep].
/// Tracking: #1636
#[test]
#[ignore = "divergence: AlphaDropout wrong SELU alpha + affine math AND system-time xorshift; tracking #1636"]
fn alpha_dropout_seed42_matches_torch() {
    // torch.manual_seed(42); nn.AlphaDropout(0.5).train()(ones(10))
    let want: [f32; 10] = [
        1.6655989, 1.6655989, 1.6655989, 1.6655989, -0.7791939, 1.6655989,
        -0.7791939, -0.7791939, 1.6655989, 1.6655989,
    ];
    ferrotorch_core::rng::manual_seed(42);
    let layer = ferrotorch_nn::AlphaDropout::<f32>::new(0.5).unwrap();
    let y = layer.forward(&ones(10)).unwrap();
    approx(&y.data_vec().unwrap(), &want, 1e-4, "AlphaDropout seed42 p0.5");
}

/// Divergence: ferrotorch's `FeatureAlphaDropout::forward` diverges from
/// `pytorch aten/src/ATen/native/Dropout.cpp:73-79` (feature+alpha). Upstream
/// (seed=42, p=0.5, [1,6,1,1]) -> per-channel
/// [1.66559887, 1.66559887, 1.66559887, 1.66559887, -0.77919394, 1.66559887]
/// (same alpha-dropout values, dropped per-channel via MT19937 stream).
/// ferrotorch uses system-time xorshift and a different SELU affine.
/// Tracking: #1636
#[test]
#[ignore = "divergence: FeatureAlphaDropout wrong SELU math AND system-time xorshift; tracking #1636"]
fn feature_alpha_dropout_seed42_matches_torch() {
    let want: [f32; 6] = [
        1.6655989, 1.6655989, 1.6655989, 1.6655989, -0.7791939, 1.6655989,
    ];
    ferrotorch_core::rng::manual_seed(42);
    let layer = ferrotorch_nn::FeatureAlphaDropout::<f32>::new(0.5).unwrap();
    let y = layer.forward(&ones_shape(vec![1, 6, 1, 1])).unwrap();
    approx(&y.data_vec().unwrap(), &want, 1e-4, "FeatureAlphaDropout seed42 p0.5");
}
