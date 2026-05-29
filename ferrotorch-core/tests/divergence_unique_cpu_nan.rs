//! Divergence: the CPU (non-CUDA) `unique` path mishandles NaN.
//!
//! `ferrotorch_core::ops::search::unique` (the host branch, search.rs:188-231)
//! sorts the index permutation with
//!   `data[a].partial_cmp(&data[b]).unwrap_or(Ordering::Equal)`
//! (search.rs:201-205). For any pair where either operand is NaN, `partial_cmp`
//! returns `None`, which is coerced to `Ordering::Equal`. NaN therefore does NOT
//! sort to the end — it stays wherever the stable sort leaves it relative to the
//! finite values. The subsequent dedup loop (search.rs:216-225) compares each
//! sorted value to the *previous unique* with `val != *unique_vals.last()`;
//! because `NaN != NaN` and `NaN != <finite>` are BOTH true, a NaN sitting
//! mid-array opens a fresh unique entry whose neighbours are finite, corrupting
//! the sorted-unique ordering, the inverse mapping, and (when the very first
//! sorted slot is a NaN that is later re-counted) the `counts` vector.
//!
//! Upstream contract (`torch.unique(sorted=True, return_inverse=True,
//! return_counts=True)` — verified LIVE on torch 2.11.0+cu130):
//!   torch.unique([nan, 1, nan, 2]) ->
//!       vals    = [1, 2, nan, nan]   (NaNs sorted to the END, each distinct)
//!       inverse = [2, 0, 3, 1]
//!       counts  = [1, 1, 1, 1]
//! This matches `aten/src/ATen/native/cpu/UniqueOps`/`Unique.cpp` `unique_cpu`'s
//! sort-then-adjacent-difference, where the sort comparator (`std::sort` with
//! the upstream `LessOrNanFunctor`, NaN ranked as the maximum) drives every NaN
//! past every finite value before the run-length dedup.
//!
//! The GPU path (`gpu_unique_f32`, bitonic `setp.neu` comparator) already
//! produces the correct `[1,2,nan,nan]` / `[2,0,3,1]` / `[1,1,1,1]` (covered by
//! `ferrotorch-gpu/tests/divergence_unique_gpu.rs::unique_f32_nan_each_distinct_at_end`);
//! this is a CPU-only divergence.
//!
//! Tracking: see crosslink issue filed alongside this test (CPU NaN spillover).

use ferrotorch_core::ops::search::unique;
use ferrotorch_core::{Tensor, TensorStorage};

fn cpu_f32(data: &[f32]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false)
        .expect("cpu f32 tensor")
}

/// torch.unique([nan,1,nan,2]) sorts both NaNs to the END as distinct entries.
/// The CPU path leaves NaN unsorted and mis-builds inverse/counts.
#[test]
fn divergence_cpu_unique_nan_sorts_to_end() {
    let input = cpu_f32(&[f32::NAN, 1.0, f32::NAN, 2.0]);
    let (vals, inverse, counts) = unique(&input).expect("cpu unique");
    let v = vals.data_vec().expect("vals");

    // torch: vals = [1, 2, nan, nan]
    assert_eq!(v.len(), 4, "four unique entries (two finite + two NaN): {v:?}");
    assert_eq!(v[0], 1.0, "first sorted-unique is the smallest finite: {v:?}");
    assert_eq!(v[1], 2.0, "second sorted-unique: {v:?}");
    assert!(
        v[2].is_nan() && v[3].is_nan(),
        "both NaNs sort to the END: {v:?}"
    );

    // torch: inverse = [2, 0, 3, 1] (input pos 0 -> NaN slot 2, pos 1 -> 1.0
    // slot 0, pos 2 -> NaN slot 3, pos 3 -> 2.0 slot 1).
    assert_eq!(inverse, vec![2, 0, 3, 1], "inverse vs torch");

    // torch: counts = [1, 1, 1, 1]
    assert_eq!(counts, vec![1, 1, 1, 1], "counts vs torch");
}

/// A second NaN arrangement: torch.unique([nan,1,nan,2,nan]) ->
///   vals=[1,2,nan,nan,nan], inverse=[2,0,3,1,4], counts=[1,1,1,1,1].
/// (Verified live torch 2.11.0+cu130.)
#[test]
fn divergence_cpu_unique_nan_three_tail() {
    let input = cpu_f32(&[f32::NAN, 1.0, f32::NAN, 2.0, f32::NAN]);
    let (vals, inverse, counts) = unique(&input).expect("cpu unique");
    let v = vals.data_vec().expect("vals");

    assert_eq!(v.len(), 5, "two finite + three NaN: {v:?}");
    assert_eq!(&v[..2], &[1.0, 2.0], "finite prefix sorted: {v:?}");
    assert!(
        v[2].is_nan() && v[3].is_nan() && v[4].is_nan(),
        "three NaNs at the tail: {v:?}"
    );
    assert_eq!(inverse, vec![2, 0, 3, 1, 4], "inverse vs torch");
    assert_eq!(counts, vec![1, 1, 1, 1, 1], "counts vs torch");
}
