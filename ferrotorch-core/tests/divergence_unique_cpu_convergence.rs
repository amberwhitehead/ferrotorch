//! Convergence regression guard for the #1664/#1665/#1666 CPU `unique` chain
//! (final fix commit 9d8809b9c, `ferrotorch-core/src/ops/search.rs`).
//!
//! #1666 changed the dedup SEED from `counts.push(0)` + iterate ALL `indices`
//! (which left a spurious leading count-0 entry whenever the SORTED-FIRST
//! element was NaN, because `NaN != NaN` opened a second entry on iteration 0)
//! to `counts.push(1)` + `inverse[indices[0]] = 0` + iterate `indices[1..]`.
//! This mirrors upstream `IsUnique<false>::operator()` which returns `true`
//! unconditionally for `i == 0` and `data[i] != data[i-1]` for `i > 0`
//! (`aten/src/ATen/native/Unique.cpp:124-130`), with the NaN-to-rear sort at
//! `Unique.cpp:185` (`auto [sorted, idx] = input_flattened.sort();`).
//!
//! REGRESSION RISK of that seed/loop change: the EMPTY and SINGLE-element edges.
//!   - EMPTY []: `indices[0]` would PANIC; the `n == 0` early-return at
//!     `search.rs:191-197` must fire BEFORE the seed. This test pins that []
//!     returns ([], [], []) and does not panic.
//!   - SINGLE: `indices[1..]` is empty, so the result is just the seed.
//!
//! Every expected value below is the LIVE output of torch 2.11.0+cu130
//! `torch.unique(t, sorted=True, return_inverse=True, return_counts=True)`
//! (captured via the live oracle, NOT copied from the ferrotorch side —
//! R-CHAR-3). Captured matrix (identical for float32 and float64):
//!
//!   []                                 -> vals []                       inv []              counts []
//!   [5]                                -> vals [5]                      inv [0]             counts [1]
//!   [nan]                              -> vals [nan]                    inv [0]             counts [1]
//!   [4,4]                              -> vals [4]                      inv [0,0]           counts [2]
//!   [4,2]                              -> vals [2,4]                    inv [1,0]           counts [1,1]
//!   [nan,nan]                          -> vals [nan,nan]                inv [0,1]           counts [1,1]
//!   [3,nan,-0,3,inf,0,nan,-inf]        -> vals [-inf,-0,3,inf,nan,nan]  inv [2,4,1,2,3,1,5,0] counts [1,2,2,1,1,1]
//!
//! The MIXED case is the comprehensive convergence check: it exercises NaN tail
//! ordering (two distinct NaNs sorted last), `-0.0`/`+0.0` collapse to ONE
//! entry whose representative is `-0.0` (the sorted-first occurrence) with
//! count 2, `-inf` first and `+inf` just before the NaN tail, and a finite
//! duplicate (3.0, count 2). If any of #1664/#1665/#1666 regresses, the
//! assertions here fail.

use ferrotorch_core::ops::search::unique;
use ferrotorch_core::{Tensor, TensorStorage};

fn cpu_f32(data: &[f32]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false)
        .expect("cpu f32 tensor")
}

fn cpu_f64(data: &[f64]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false)
        .expect("cpu f64 tensor")
}

// --- EMPTY: must NOT panic (the n==0 early-return guards the indices[0] seed) -

#[test]
fn convergence_f32_empty_no_panic() {
    let (vals, inv, counts) = unique(&cpu_f32(&[])).unwrap();
    assert_eq!(vals.numel(), 0, "empty vals");
    assert!(inv.is_empty(), "empty inverse");
    assert!(counts.is_empty(), "empty counts");
}

#[test]
fn convergence_f64_empty_no_panic() {
    let (vals, inv, counts) = unique(&cpu_f64(&[])).unwrap();
    assert_eq!(vals.numel(), 0, "empty vals");
    assert!(inv.is_empty(), "empty inverse");
    assert!(counts.is_empty(), "empty counts");
}

// --- SINGLE finite ----------------------------------------------------------

#[test]
fn convergence_f32_single_finite() {
    let (vals, inv, counts) = unique(&cpu_f32(&[5.0])).unwrap();
    assert_eq!(vals.data_vec().unwrap(), vec![5.0]);
    assert_eq!(inv, vec![0]);
    assert_eq!(counts, vec![1]);
}

#[test]
fn convergence_f64_single_finite() {
    let (vals, inv, counts) = unique(&cpu_f64(&[5.0])).unwrap();
    assert_eq!(vals.data_vec().unwrap(), vec![5.0]);
    assert_eq!(inv, vec![0]);
    assert_eq!(counts, vec![1]);
}

// --- SINGLE NaN (the #1666 regression: was 2 entries with leading count-0) ---

#[test]
fn convergence_f32_single_nan() {
    let (vals, inv, counts) = unique(&cpu_f32(&[f32::NAN])).unwrap();
    let v = vals.data_vec().unwrap();
    assert_eq!(
        v.len(),
        1,
        "exactly one entry, not a leading count-0 ghost: {v:?}"
    );
    assert!(v[0].is_nan(), "the lone value is NaN: {v:?}");
    assert_eq!(inv, vec![0]);
    assert_eq!(counts, vec![1]);
}

#[test]
fn convergence_f64_single_nan() {
    let (vals, inv, counts) = unique(&cpu_f64(&[f64::NAN])).unwrap();
    let v = vals.data_vec().unwrap();
    assert_eq!(v.len(), 1, "exactly one entry: {v:?}");
    assert!(v[0].is_nan(), "the lone value is NaN: {v:?}");
    assert_eq!(inv, vec![0]);
    assert_eq!(counts, vec![1]);
}

// --- TWO equal --------------------------------------------------------------

#[test]
fn convergence_f32_two_equal() {
    let (vals, inv, counts) = unique(&cpu_f32(&[4.0, 4.0])).unwrap();
    assert_eq!(vals.data_vec().unwrap(), vec![4.0]);
    assert_eq!(inv, vec![0, 0]);
    assert_eq!(counts, vec![2]);
}

#[test]
fn convergence_f64_two_equal() {
    let (vals, inv, counts) = unique(&cpu_f64(&[4.0, 4.0])).unwrap();
    assert_eq!(vals.data_vec().unwrap(), vec![4.0]);
    assert_eq!(inv, vec![0, 0]);
    assert_eq!(counts, vec![2]);
}

// --- TWO distinct (unsorted input -> sorted vals, inverse maps back) --------

#[test]
fn convergence_f32_two_distinct() {
    let (vals, inv, counts) = unique(&cpu_f32(&[4.0, 2.0])).unwrap();
    assert_eq!(vals.data_vec().unwrap(), vec![2.0, 4.0]);
    assert_eq!(inv, vec![1, 0]);
    assert_eq!(counts, vec![1, 1]);
}

#[test]
fn convergence_f64_two_distinct() {
    let (vals, inv, counts) = unique(&cpu_f64(&[4.0, 2.0])).unwrap();
    assert_eq!(vals.data_vec().unwrap(), vec![2.0, 4.0]);
    assert_eq!(inv, vec![1, 0]);
    assert_eq!(counts, vec![1, 1]);
}

// --- TWO NaN (each distinct, both at the end) -------------------------------

#[test]
fn convergence_f32_two_nan() {
    let (vals, inv, counts) = unique(&cpu_f32(&[f32::NAN, f32::NAN])).unwrap();
    let v = vals.data_vec().unwrap();
    assert_eq!(v.len(), 2, "two distinct NaN entries: {v:?}");
    assert!(v[0].is_nan() && v[1].is_nan(), "both NaN: {v:?}");
    assert_eq!(inv, vec![0, 1]);
    assert_eq!(counts, vec![1, 1]);
}

#[test]
fn convergence_f64_two_nan() {
    let (vals, inv, counts) = unique(&cpu_f64(&[f64::NAN, f64::NAN])).unwrap();
    let v = vals.data_vec().unwrap();
    assert_eq!(v.len(), 2, "two distinct NaN entries: {v:?}");
    assert!(v[0].is_nan() && v[1].is_nan(), "both NaN: {v:?}");
    assert_eq!(inv, vec![0, 1]);
    assert_eq!(counts, vec![1, 1]);
}

// --- MIXED: comprehensive convergence (NaN tail, -0/+0 collapse, inf order) -
// Input:  [3, nan, -0.0, 3, inf, 0.0, nan, -inf]
// Torch:  vals [-inf, -0.0, 3, inf, nan, nan]  inv [2,4,1,2,3,1,5,0]  counts [1,2,2,1,1,1]

#[test]
fn convergence_f32_mixed_full() {
    let n = f32::NAN;
    let inf = f32::INFINITY;
    let input = cpu_f32(&[3.0, n, -0.0, 3.0, inf, 0.0, n, -inf]);
    let (vals, inv, counts) = unique(&input).unwrap();
    let v = vals.data_vec().unwrap();
    assert_eq!(v.len(), 6, "six unique entries: {v:?}");
    assert_eq!(v[0], f32::NEG_INFINITY, "v[0]=-inf: {v:?}");
    // -0.0/+0.0 collapse to ONE entry; representative is the sorted-first
    // occurrence (-0.0). Check sign bit, since 0.0 == -0.0 numerically.
    assert_eq!(v[1], 0.0, "v[1] numerically zero: {v:?}");
    assert!(
        v[1].is_sign_negative(),
        "v[1] is -0.0 (negative sign): {v:?}"
    );
    assert_eq!(v[2], 3.0, "v[2]=3: {v:?}");
    assert_eq!(v[3], f32::INFINITY, "v[3]=+inf: {v:?}");
    assert!(v[4].is_nan() && v[5].is_nan(), "v[4..]=nan tail: {v:?}");
    assert_eq!(inv, vec![2, 4, 1, 2, 3, 1, 5, 0], "inverse");
    assert_eq!(counts, vec![1, 2, 2, 1, 1, 1], "counts");
}

#[test]
fn convergence_f64_mixed_full() {
    let n = f64::NAN;
    let inf = f64::INFINITY;
    let input = cpu_f64(&[3.0, n, -0.0, 3.0, inf, 0.0, n, -inf]);
    let (vals, inv, counts) = unique(&input).unwrap();
    let v = vals.data_vec().unwrap();
    assert_eq!(v.len(), 6, "six unique entries: {v:?}");
    assert_eq!(v[0], f64::NEG_INFINITY, "v[0]=-inf: {v:?}");
    assert_eq!(v[1], 0.0, "v[1] numerically zero: {v:?}");
    assert!(
        v[1].is_sign_negative(),
        "v[1] is -0.0 (negative sign): {v:?}"
    );
    assert_eq!(v[2], 3.0, "v[2]=3: {v:?}");
    assert_eq!(v[3], f64::INFINITY, "v[3]=+inf: {v:?}");
    assert!(v[4].is_nan() && v[5].is_nan(), "v[4..]=nan tail: {v:?}");
    assert_eq!(inv, vec![2, 4, 1, 2, 3, 1, 5, 0], "inverse");
    assert_eq!(counts, vec![1, 2, 2, 1, 1, 1], "counts");
}
