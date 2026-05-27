//! Adversarial re-audit of commit `83f25f5fa` (umbrella #1542):
//!   - #1335 `nextafter` (transcendental.rs)
//!   - #1306 `median_with_dim` / `nanmedian_with_dim` (reduction.rs)
//!
//! The builder's own unit tests in `transcendental.rs` exercise `nextafter`
//! only on `f64` leaves (`leaf_vec_f64`), where the implementation's
//! "step one ULP in f64, then `T::from`" round-trip is exact. This file pins
//! the divergence that surfaces for the `f32` dtype: stepping a single ULP in
//! `f64` produces an `f64` strictly *between* two adjacent `f32` values, so
//! the `T::from::<f32>(stepped)` cast rounds straight back to the ORIGINAL
//! `f32`. The op is therefore a no-op for every `f32` input, where
//! `torch.nextafter` advances exactly one `f32` ULP.
//!
//! Oracle (live torch 2026-05-26, dtype=torch.float32, bit patterns via
//! struct.pack):
//!   nextafter(1.0, 2.0) -> 1.0000001  (0x3f800001)   ferrotorch: 1.0 (0x3f800000)
//!   nextafter(1.0, 0.0) -> 0.99999994 (0x3f7fffff)   ferrotorch: 1.0 (0x3f800000)
//!   nextafter(0.0, 1.0) -> 1.4e-45    (0x00000001)   ferrotorch: 0.0 (0x00000000)
//!
//! Upstream forward: `aten/src/ATen/native/BinaryOps.cpp:551 nextafter_out`
//! routes the CPU kernel to `std::nextafter` at the TENSOR dtype, i.e. for an
//! `f32` tensor it steps an `f32` ULP — not an `f64` ULP.
//!
//! The `f64` cases and the median cases below are VERIFIED (oracle match) and
//! left un-`#[ignore]`d as regression pins.

use ferrotorch_core::{Tensor, grad_fns, nextafter};

fn leaf_vec_f32(data: &[f32]) -> Tensor<f32> {
    ferrotorch_core::from_vec(data.to_vec(), &[data.len()]).expect("from_vec f32")
}
fn leaf_vec_f64(data: &[f64]) -> Tensor<f64> {
    ferrotorch_core::from_vec(data.to_vec(), &[data.len()]).expect("from_vec f64")
}

// ---------------------------------------------------------------------------
// #1335 nextafter — f32 dtype divergence (RELEASE-BLOCKING, left un-ignored).
// ---------------------------------------------------------------------------

/// Divergence: `ferrotorch_core::nextafter` for `f32` diverges from
/// `pytorch aten/src/ATen/native/BinaryOps.cpp:551` (`nextafter_out` ->
/// `std::nextafter` at tensor dtype). Upstream `nextafter(1.0f32, 2.0f32)`
/// returns `0x3f800001` (one f32 ULP up); ferrotorch's
/// `transcendental.rs::nextafter_scalar` steps one *f64* ULP then casts back
/// to f32 (`T::from(stepped)`), which rounds to the original `0x3f800000`.
/// Tracking: #1556.
#[test]
fn divergence_nextafter_f32_up_is_a_noop() {
    let a = leaf_vec_f32(&[1.0]);
    let b = leaf_vec_f32(&[2.0]);
    let c = nextafter(&a, &b).expect("nextafter fwd");
    let got = c.data().expect("data")[0];
    // Oracle: torch.nextafter(1.0f32, 2.0f32) bit pattern.
    let expected = f32::from_bits(0x3f80_0001);
    assert_eq!(
        got.to_bits(),
        expected.to_bits(),
        "nextafter(1.0f32, 2.0f32): expected 0x{:08x} ({expected}), got 0x{:08x} ({got})",
        expected.to_bits(),
        got.to_bits(),
    );
}

/// Divergence: same root cause, stepping toward `-inf`.
/// Oracle: torch.nextafter(1.0f32, 0.0f32) = 0x3f7fffff (0.99999994).
#[test]
fn divergence_nextafter_f32_down_is_a_noop() {
    let a = leaf_vec_f32(&[1.0]);
    let b = leaf_vec_f32(&[0.0]);
    let c = nextafter(&a, &b).expect("nextafter fwd");
    let got = c.data().expect("data")[0];
    let expected = f32::from_bits(0x3f7f_ffff);
    assert_eq!(
        got.to_bits(),
        expected.to_bits(),
        "nextafter(1.0f32, 0.0f32): expected 0x{:08x}, got 0x{:08x}",
        expected.to_bits(),
        got.to_bits(),
    );
}

/// Divergence: from +0.0 the next f32 toward +inf is the smallest positive
/// subnormal 0x00000001; ferrotorch returns +0.0 because the f64 subnormal
/// `f64::from_bits(1)` underflows to 0 on the cast back to f32.
/// Oracle: torch.nextafter(0.0f32, 1.0f32) = 0x00000001 (1.4e-45).
#[test]
fn divergence_nextafter_f32_from_zero_subnormal() {
    let a = leaf_vec_f32(&[0.0]);
    let b = leaf_vec_f32(&[1.0]);
    let c = nextafter(&a, &b).expect("nextafter fwd");
    let got = c.data().expect("data")[0];
    assert_eq!(
        got.to_bits(),
        0x0000_0001u32,
        "nextafter(0.0f32, 1.0f32): expected 0x00000001, got 0x{:08x}",
        got.to_bits(),
    );
}

/// VERIFIED pin: the f64 path IS correct (the round-trip is exact for f64).
/// Oracle: torch.nextafter(1.0f64, 2.0f64) = 0x1.0000000000001p+0.
#[test]
fn nextafter_f64_up_matches_torch() {
    let a = leaf_vec_f64(&[1.0]);
    let b = leaf_vec_f64(&[2.0]);
    let c = nextafter(&a, &b).expect("nextafter fwd");
    let got = c.data().expect("data")[0];
    let expected = f64::from_bits(0x3ff0_0000_0000_0001); // 1.0000000000000002
    assert_eq!(got.to_bits(), expected.to_bits());
}

/// VERIFIED pin: nextafter(x, x) == x (signed-zero tie returns `b`).
/// Oracle: torch.nextafter(5.0, 5.0) = 5.0.
#[test]
fn nextafter_f32_tie_returns_self() {
    let a = leaf_vec_f32(&[5.0]);
    let b = leaf_vec_f32(&[5.0]);
    let c = nextafter(&a, &b).expect("nextafter fwd");
    assert_eq!(c.data().expect("data")[0], 5.0f32);
}

// ---------------------------------------------------------------------------
// #1306 median_with_dim / nanmedian_with_dim — VERIFIED pins.
// ---------------------------------------------------------------------------

/// VERIFIED: lower-median + original-position index.
/// Oracle: torch.median(tensor([[1,5,3],[4,2,6]]), dim=1) ->
///   values=[3,4], indices=[2,0].
#[test]
fn median_with_dim_matches_torch() {
    let x = ferrotorch_core::from_vec(vec![1.0f64, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3]).unwrap();
    let (vals, inds) = grad_fns::reduction::median_with_dim(&x, 1, false).unwrap();
    assert_eq!(vals.data().unwrap(), &[3.0, 4.0]);
    assert_eq!(inds.data().unwrap(), &[2, 0]);
}

/// VERIFIED: median NaN-poisons a slice (returns NaN + first-NaN index),
/// nanmedian skips the NaN.
/// Oracle:
///   torch.median(tensor([[1,nan,3],[4,2,6]]),dim=1).values  -> [nan, 4]
///   torch.median(...).indices                               -> [1, 0]
///   torch.nanmedian(...).values                             -> [1, 4]
///   torch.nanmedian(...).indices                            -> [0, 0]
#[test]
fn median_vs_nanmedian_nan_handling_matches_torch() {
    let x = ferrotorch_core::from_vec(vec![1.0f64, f64::NAN, 3.0, 4.0, 2.0, 6.0], &[2, 3]).unwrap();

    let (mv, mi) = grad_fns::reduction::median_with_dim(&x, 1, false).unwrap();
    let mvd = mv.data().unwrap();
    assert!(mvd[0].is_nan(), "median row0 expected NaN, got {}", mvd[0]);
    assert_eq!(mvd[1], 4.0);
    assert_eq!(mi.data().unwrap(), &[1, 0]);

    let (nv, ni) = grad_fns::reduction::nanmedian_with_dim(&x, 1, false).unwrap();
    assert_eq!(nv.data().unwrap(), &[1.0, 4.0]);
    assert_eq!(ni.data().unwrap(), &[0, 0]);
}

/// VERIFIED: median backward scatters grad 1.0 to each slice's selected
/// median ORIGINAL index, 0 elsewhere.
/// Oracle: torch median([[3,1,2],[4,2,6]],dim=1).values.sum().backward()
///   -> x.grad = [[0,0,1],[1,0,0]]  (medians at idx 2 and 0).
#[test]
fn median_backward_scatters_to_original_index() {
    let x = ferrotorch_core::from_vec(vec![3.0f64, 1.0, 2.0, 4.0, 2.0, 6.0], &[2, 3])
        .unwrap()
        .requires_grad_(true);
    let (vals, _inds) = grad_fns::reduction::median_with_dim(&x, 1, false).unwrap();
    // values is shape [2]; sum to a scalar to seed backward with grad 1.
    let s = grad_fns::reduction::sum_dim(&vals, 0, false).unwrap();
    s.backward().unwrap();
    let g = x.grad().unwrap().unwrap();
    assert_eq!(g.data().unwrap(), &[0.0, 0.0, 1.0, 1.0, 0.0, 0.0]);
}
