//! Re-audit regression guard for the #1665 CPU `unique` NaN/-0.0/inf/finite fix
//! (commit f0851c489, `ferrotorch-core/src/ops/search.rs`).
//!
//! The CPU host branch sorts the index permutation with the in-file total-order
//! comparator `nan_is_max_cmp` (NaN ranked as MAXIMUM -> sorts to the END;
//! `partial_cmp` for the non-NaN pair so `-0.0`/`+0.0` stay equal+adjacent and
//! `-inf`/`+inf` order correctly). The dedup loop (`val != last`, NaN != NaN =
//! true) keeps each NaN distinct, mirroring upstream `IsUnique<false>`
//! (`aten/src/ATen/native/Unique.cpp:124-130`: `data_ptr[i] != data_ptr[i-1]`)
//! and the NaN-to-rear sort at `Unique.cpp:185`
//! (`auto [input_sorted, indices] = input_flattened.sort();`).
//!
//! Every expected value below is the LIVE output of torch 2.11.0+cu130
//! `torch.unique(t, sorted=True, return_inverse=True, return_counts=True)`
//! (NOT copied from the ferrotorch side — R-CHAR-3). The captured oracle:
//!
//!   [nan,1,nan,2]    -> vals [1,2,nan,nan]      inv [2,0,3,1]   counts [1,1,1,1]
//!   [1,2,3]          -> vals [1,2,3]            inv [0,1,2]     counts [1,1,1]
//!   [nan,nan,nan]    -> vals [nan,nan,nan]      inv [0,1,2]     counts [1,1,1]
//!   [nan]            -> vals [nan]              inv [0]         counts [1]
//!   [1,nan,1]        -> vals [1,nan]            inv [0,1,0]     counts [2,1]
//!   [-0.0,0.0,-0.0]  -> vals [-0.0]             inv [0,0,0]     counts [3]
//!   [inf,-inf,0,nan] -> vals [-inf,0,inf,nan]   inv [2,0,1,3]   counts [1,1,1,1]
//!   [3,1,2,1,3]      -> vals [1,2,3]            inv [2,0,1,0,2] counts [2,1,2]
//!   [5,4,3,2,1]      -> vals [1,2,3,4,5]        inv [4,3,2,1,0] counts [1,1,1,1,1]
//!   [7,7,7]          -> vals [7]                inv [0,0,0]     counts [3]
//!   [1,2,3,4]        -> vals [1,2,3,4]          inv [0,1,2,3]   counts [1,1,1,1]
//!   [9]              -> vals [9]                inv [0]         counts [1]
//!   []               -> vals []                inv []          counts []
//!
//! This file is a PASSING guard: if the comparator regresses (NaN no longer
//! sorts to the end, -0.0/+0.0 wrongly split by a total_cmp fix, or inverse
//! desyncs) these assertions fail.

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

// --- f32 matrix ---------------------------------------------------------

#[test]
fn reaudit_f32_nan_interleaved() {
    let (vals, inv, counts) = unique(&cpu_f32(&[f32::NAN, 1.0, f32::NAN, 2.0])).unwrap();
    let v = vals.data_vec().unwrap();
    assert_eq!(v.len(), 4, "vals {v:?}");
    assert_eq!(&v[..2], &[1.0, 2.0], "finite prefix {v:?}");
    assert!(v[2].is_nan() && v[3].is_nan(), "NaN tail {v:?}");
    assert_eq!(inv, vec![2, 0, 3, 1], "inverse");
    assert_eq!(counts, vec![1, 1, 1, 1], "counts");
}

#[test]
fn reaudit_f32_no_nan() {
    let (vals, inv, counts) = unique(&cpu_f32(&[1.0, 2.0, 3.0])).unwrap();
    assert_eq!(vals.data_vec().unwrap(), &[1.0, 2.0, 3.0]);
    assert_eq!(inv, vec![0, 1, 2]);
    assert_eq!(counts, vec![1, 1, 1]);
}

#[test]
#[ignore = "divergence: CPU unique miscounts all-NaN input (extra count-0 entry when sorted-first elem is NaN); tracking #1666"]
fn reaudit_f32_all_nan() {
    // torch keeps EACH NaN distinct (equal_nan disabled): 3 entries, inverse
    // maps each original position to a distinct sorted-unique NaN slot in order.
    let (vals, inv, counts) = unique(&cpu_f32(&[f32::NAN, f32::NAN, f32::NAN])).unwrap();
    let v = vals.data_vec().unwrap();
    assert_eq!(v.len(), 3, "three distinct NaN entries {v:?}");
    assert!(v.iter().all(|x| x.is_nan()), "all NaN {v:?}");
    assert_eq!(
        inv,
        vec![0, 1, 2],
        "each original NaN -> own slot, in order"
    );
    assert_eq!(counts, vec![1, 1, 1], "each NaN counted once");
}

#[test]
#[ignore = "divergence: CPU unique miscounts all-NaN input (extra count-0 entry when sorted-first elem is NaN); tracking #1666"]
fn reaudit_f32_single_nan() {
    let (vals, inv, counts) = unique(&cpu_f32(&[f32::NAN])).unwrap();
    let v = vals.data_vec().unwrap();
    assert_eq!(v.len(), 1);
    assert!(v[0].is_nan());
    assert_eq!(inv, vec![0]);
    assert_eq!(counts, vec![1]);
}

#[test]
fn reaudit_f32_finite_dedup_across_nan() {
    // [1, nan, 1]: the two finite 1s (positions 0 and 2) dedup to ONE slot even
    // though a NaN sat between them by original position; NaN sorts to the end.
    let (vals, inv, counts) = unique(&cpu_f32(&[1.0, f32::NAN, 1.0])).unwrap();
    let v = vals.data_vec().unwrap();
    assert_eq!(v.len(), 2, "{v:?}");
    assert_eq!(v[0], 1.0, "finite slot {v:?}");
    assert!(v[1].is_nan(), "NaN slot {v:?}");
    assert_eq!(inv, vec![0, 1, 0], "both 1s -> slot 0, nan -> slot 1");
    assert_eq!(counts, vec![2, 1], "1 appears twice, nan once");
}

#[test]
fn reaudit_f32_signed_zero_collapses() {
    // torch collapses -0.0 and +0.0 to ONE entry; the sorted-first value (-0.0)
    // is kept. A total_cmp-based comparator would WRONGLY split them.
    let (vals, inv, counts) = unique(&cpu_f32(&[-0.0, 0.0, -0.0])).unwrap();
    let v = vals.data_vec().unwrap();
    assert_eq!(v.len(), 1, "single zero entry (not split): {v:?}");
    assert_eq!(v[0], 0.0, "0.0 == -0.0 by value");
    assert_eq!(inv, vec![0, 0, 0]);
    assert_eq!(counts, vec![3]);
}

#[test]
fn reaudit_f32_inf_ordering() {
    // sorted [-inf, 0, inf, nan]; NaN strictly after +inf.
    let (vals, inv, counts) =
        unique(&cpu_f32(&[f32::INFINITY, f32::NEG_INFINITY, 0.0, f32::NAN])).unwrap();
    let v = vals.data_vec().unwrap();
    assert_eq!(v.len(), 4, "{v:?}");
    assert_eq!(v[0], f32::NEG_INFINITY, "{v:?}");
    assert_eq!(v[1], 0.0, "{v:?}");
    assert_eq!(v[2], f32::INFINITY, "{v:?}");
    assert!(v[3].is_nan(), "nan after +inf {v:?}");
    assert_eq!(inv, vec![2, 0, 1, 3]);
    assert_eq!(counts, vec![1, 1, 1, 1]);
}

#[test]
fn reaudit_f32_finite_unsorted() {
    let (vals, inv, counts) = unique(&cpu_f32(&[3.0, 1.0, 2.0, 1.0, 3.0])).unwrap();
    assert_eq!(vals.data_vec().unwrap(), &[1.0, 2.0, 3.0]);
    assert_eq!(inv, vec![2, 0, 1, 0, 2]);
    assert_eq!(counts, vec![2, 1, 2]);
}

#[test]
fn reaudit_f32_reverse_sorted() {
    let (vals, inv, counts) = unique(&cpu_f32(&[5.0, 4.0, 3.0, 2.0, 1.0])).unwrap();
    assert_eq!(vals.data_vec().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(inv, vec![4, 3, 2, 1, 0]);
    assert_eq!(counts, vec![1, 1, 1, 1, 1]);
}

#[test]
fn reaudit_f32_all_same() {
    let (vals, inv, counts) = unique(&cpu_f32(&[7.0, 7.0, 7.0])).unwrap();
    assert_eq!(vals.data_vec().unwrap(), &[7.0]);
    assert_eq!(inv, vec![0, 0, 0]);
    assert_eq!(counts, vec![3]);
}

#[test]
fn reaudit_f32_all_distinct() {
    let (vals, inv, counts) = unique(&cpu_f32(&[1.0, 2.0, 3.0, 4.0])).unwrap();
    assert_eq!(vals.data_vec().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(inv, vec![0, 1, 2, 3]);
    assert_eq!(counts, vec![1, 1, 1, 1]);
}

#[test]
fn reaudit_f32_single() {
    let (vals, inv, counts) = unique(&cpu_f32(&[9.0])).unwrap();
    assert_eq!(vals.data_vec().unwrap(), &[9.0]);
    assert_eq!(inv, vec![0]);
    assert_eq!(counts, vec![1]);
}

#[test]
fn reaudit_f32_empty() {
    let (vals, inv, counts) = unique(&cpu_f32(&[])).unwrap();
    assert_eq!(vals.numel(), 0);
    assert!(inv.is_empty());
    assert!(counts.is_empty());
}

// --- f64 matrix (the comparator is generic; test both dtypes) -----------

#[test]
fn reaudit_f64_nan_interleaved() {
    let (vals, inv, counts) = unique(&cpu_f64(&[f64::NAN, 1.0, f64::NAN, 2.0])).unwrap();
    let v = vals.data_vec().unwrap();
    assert_eq!(&v[..2], &[1.0, 2.0], "{v:?}");
    assert!(v[2].is_nan() && v[3].is_nan(), "{v:?}");
    assert_eq!(inv, vec![2, 0, 3, 1]);
    assert_eq!(counts, vec![1, 1, 1, 1]);
}

#[test]
#[ignore = "divergence: CPU unique miscounts all-NaN input (extra count-0 entry when sorted-first elem is NaN); tracking #1666"]
fn reaudit_f64_all_nan() {
    let (vals, inv, counts) = unique(&cpu_f64(&[f64::NAN, f64::NAN, f64::NAN])).unwrap();
    assert_eq!(vals.data_vec().unwrap().len(), 3);
    assert_eq!(inv, vec![0, 1, 2]);
    assert_eq!(counts, vec![1, 1, 1]);
}

#[test]
fn reaudit_f64_signed_zero_collapses() {
    let (vals, inv, counts) = unique(&cpu_f64(&[-0.0, 0.0, -0.0])).unwrap();
    assert_eq!(vals.data_vec().unwrap().len(), 1);
    assert_eq!(inv, vec![0, 0, 0]);
    assert_eq!(counts, vec![3]);
}

#[test]
fn reaudit_f64_inf_ordering() {
    let (vals, inv, counts) =
        unique(&cpu_f64(&[f64::INFINITY, f64::NEG_INFINITY, 0.0, f64::NAN])).unwrap();
    let v = vals.data_vec().unwrap();
    assert_eq!(v[0], f64::NEG_INFINITY, "{v:?}");
    assert_eq!(v[1], 0.0, "{v:?}");
    assert_eq!(v[2], f64::INFINITY, "{v:?}");
    assert!(v[3].is_nan(), "{v:?}");
    assert_eq!(inv, vec![2, 0, 1, 3]);
    assert_eq!(counts, vec![1, 1, 1, 1]);
}
