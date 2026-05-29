//! Re-audit (#1648 / #1545): extra NaN/±inf ordering cases for GPU `topk`,
//! beyond the three cases pinned in `divergence_topk_nan_ordering.rs`.
//!
//! Upstream cite:
//!   `aten/src/ATen/native/cuda/SortingCommon.cuh:46-60` — `GTOp`/`LTOp` with
//!   `handleNaN=true`:
//!       GTOp: `(handleNaN && _isnan(lhs) && !_isnan(rhs)) || (lhs > rhs)`
//!       LTOp: `(handleNaN && _isnan(rhs) && !_isnan(lhs)) || (lhs < rhs)`
//!   i.e. NaN compares GREATER than every finite value AND greater than ±inf
//!   (`_isnan` is true only for NaN; +inf is finite-comparable). So for
//!   `largest=True` NaN outranks +inf, and for `largest=False` NaN is ranked
//!   last (only selected once finite/±inf values are exhausted).
//!   Driver: `topk_out_cuda` gathers then `sortKeyValueInplace(.., stable=false)`
//!   (`aten/src/ATen/native/cuda/TensorTopK.cpp:97,101`), so the per-tie INDEX
//!   order among equal-rank elements (incl. NaN==NaN) is torch-UNSPECIFIED.
//!   We therefore assert on the VALUE contract (byte-exact NaN, exact finite /
//!   ±inf values, in rank order) and that any NaN-slot index points to an
//!   original NaN position — NOT a specific torch-CUDA tie permutation.
//!
//! Live torch 2.11.0+cu130 (RTX 3090) oracle for the inputs below:
//!   t=[3,NaN,inf,5,-inf,NaN]
//!     k=4 largest=True  -> [NaN,NaN,inf,5]            (NaN outranks +inf)
//!     k=6 largest=True  -> [NaN,NaN,inf,5,3,-inf]
//!     k=4 largest=False -> [-inf,3,5,inf]             (no NaN; -inf first)
//!     k=6 largest=False -> [-inf,3,5,inf,NaN,NaN]     (NaN last)
//!   t=[NaN,NaN,NaN,NaN] k=4 largest=True/False -> all NaN
//!   t=[3,1,NaN,2]
//!     k=4 largest=True  -> [NaN,3,2,1]
//!     k=4 largest=False -> [1,2,3,NaN]
//!   t=[3,inf,1,-inf,2] (no NaN — control that the NaN change didn't perturb inf)
//!     k=5 largest=True  -> [inf,3,2,1,-inf]
//!     k=5 largest=False -> [-inf,1,2,3,inf]
//!
//! Tracking: #1648 (NaN ordering), #1545 (topk parity).

#![cfg(feature = "cuda")]

use ferrotorch_gpu::{GpuDevice, cpu_to_gpu, gpu_topk_f32, gpu_topk_f64, init_cuda_backend};
use std::sync::Once;

fn ensure_cuda() -> Option<GpuDevice> {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let _ = init_cuda_backend();
    });
    GpuDevice::new(0).ok()
}

const INF: f32 = f32::INFINITY;
const NINF: f32 = f32::NEG_INFINITY;
const NAN: f32 = f32::NAN;

fn topk32(row: &[f32], k: usize, largest: bool) -> (Vec<f32>, Vec<i64>) {
    let device = ensure_cuda().expect("cuda required for this re-audit");
    let dim = row.len();
    let g = cpu_to_gpu(row, &device).expect("upload");
    let (vals, idx) = gpu_topk_f32(g.inner(), 1, dim, k, largest, &device).expect("topk");
    let hv = device.stream().clone_dtoh(&vals).expect("readback vals");
    let hi = device.stream().clone_dtoh(&idx).expect("readback idx");
    (hv, hi)
}

/// largest=True: NaN OUTRANKS +inf. torch: [NaN,NaN,inf,5].
#[test]
fn divergence_topk_f32_nan_outranks_pos_inf() {
    if ensure_cuda().is_none() {
        return;
    }
    let row = [3.0_f32, NAN, INF, 5.0, NINF, NAN];
    let (v, i) = topk32(&row, 4, true);
    assert!(
        v[0].is_nan() && v[1].is_nan(),
        "torch: NaN ranks above +inf for largest=True; got {v:?}"
    );
    // The NaN slots must index original NaN positions (1 or 5), tie order unspec.
    for slot in 0..2 {
        let pos = i[slot] as usize;
        assert!(
            row[pos].is_nan(),
            "NaN-slot {slot} index {pos} must point to an original NaN; got {}",
            row[pos]
        );
    }
    assert!(
        v[2].is_infinite() && v[2] > 0.0,
        "third element is +inf (below NaN, above finite); got {}",
        v[2]
    );
    assert_eq!(i[2] as usize, 2, "+inf came from original index 2");
    assert_eq!(v[3], 5.0_f32, "fourth element is the largest finite 5.0");
    assert_eq!(i[3] as usize, 3);
}

/// largest=True, full k=6: complete rank order [NaN,NaN,inf,5,3,-inf].
#[test]
fn divergence_topk_f32_largest_full_order_with_inf() {
    if ensure_cuda().is_none() {
        return;
    }
    let row = [3.0_f32, NAN, INF, 5.0, NINF, NAN];
    let (v, _i) = topk32(&row, 6, true);
    assert!(v[0].is_nan() && v[1].is_nan(), "NaNs first: {v:?}");
    assert_eq!(v[2], INF, "then +inf");
    assert_eq!(v[3], 5.0);
    assert_eq!(v[4], 3.0);
    assert_eq!(v[5], NINF, "then -inf last (smallest finite)");
}

/// largest=False: -inf first, NaN never selected until finite/inf exhausted.
#[test]
fn divergence_topk_f32_smallest_neg_inf_first_nan_last() {
    if ensure_cuda().is_none() {
        return;
    }
    let row = [3.0_f32, NAN, INF, 5.0, NINF, NAN];

    // k=4: the four non-NaN extrema ascending, NO NaN.
    let (v4, i4) = topk32(&row, 4, false);
    assert!(
        v4.iter().take(4).all(|x| !x.is_nan()),
        "torch: largest=False picks no NaN at k<n_finite; got {v4:?}"
    );
    assert_eq!(v4[0], NINF, "smallest is -inf");
    assert_eq!(v4[1], 3.0);
    assert_eq!(v4[2], 5.0);
    assert_eq!(v4[3], INF, "+inf is the largest non-NaN");
    assert_eq!(i4[0] as usize, 4);

    // k=6: NaN finally lands LAST.
    let (v6, i6) = topk32(&row, 6, false);
    assert_eq!(&v6[..4], &[NINF, 3.0, 5.0, INF]);
    assert!(
        v6[4].is_nan() && v6[5].is_nan(),
        "torch: NaN ranks last for largest=False; got {v6:?}"
    );
    for slot in 4..6 {
        let pos = i6[slot] as usize;
        assert!(
            row[pos].is_nan(),
            "trailing NaN-slot {slot} index {pos} must point to original NaN"
        );
    }
}

/// All-NaN input: every output is NaN, indices form a permutation of [0,n).
#[test]
fn divergence_topk_f32_all_nan() {
    if ensure_cuda().is_none() {
        return;
    }
    let row = [NAN, NAN, NAN, NAN];
    for largest in [true, false] {
        let (v, i) = topk32(&row, 4, largest);
        assert!(
            v.iter().all(|x| x.is_nan()),
            "all-NaN topk(largest={largest}) -> all NaN; got {v:?}"
        );
        let mut idx: Vec<i64> = i.clone();
        idx.sort_unstable();
        assert_eq!(idx, vec![0, 1, 2, 3], "indices are a permutation of [0,4)");
    }
}

/// Single-NaN: largest -> NaN leads [NaN,3,2,1]; smallest -> NaN trails [1,2,3,NaN].
#[test]
fn divergence_topk_f32_single_nan() {
    if ensure_cuda().is_none() {
        return;
    }
    let row = [3.0_f32, 1.0, NAN, 2.0];

    let (vl, il) = topk32(&row, 4, true);
    assert!(vl[0].is_nan(), "largest: NaN first; got {vl:?}");
    assert_eq!(il[0] as usize, 2, "NaN came from index 2");
    assert_eq!(&vl[1..], &[3.0, 2.0, 1.0]);

    let (vs, is_) = topk32(&row, 4, false);
    assert_eq!(&vs[..3], &[1.0, 2.0, 3.0]);
    assert!(vs[3].is_nan(), "smallest: NaN last; got {vs:?}");
    assert_eq!(is_[3] as usize, 2, "trailing NaN came from index 2");
}

/// Control: pure ±inf (no NaN) ordering is UNAFFECTED by the NaN-as-max change.
/// largest -> [inf,3,2,1,-inf]; smallest -> [-inf,1,2,3,inf].
#[test]
fn divergence_topk_f32_inf_no_nan_unregressed() {
    if ensure_cuda().is_none() {
        return;
    }
    let row = [3.0_f32, INF, 1.0, NINF, 2.0];

    let (vl, il) = topk32(&row, 5, true);
    assert_eq!(vl, vec![INF, 3.0, 2.0, 1.0, NINF], "largest inf order");
    assert_eq!(il, vec![1, 0, 4, 2, 3]);

    let (vs, is_) = topk32(&row, 5, false);
    assert_eq!(vs, vec![NINF, 1.0, 2.0, 3.0, INF], "smallest inf order");
    assert_eq!(is_, vec![3, 2, 4, 0, 1]);
}

/// f64 NaN-outranks-+inf parity with the f32 path.
#[test]
fn divergence_topk_f64_nan_outranks_pos_inf() {
    let device = match ensure_cuda() {
        Some(d) => d,
        None => return,
    };
    let row = [
        3.0_f64,
        f64::NAN,
        f64::INFINITY,
        5.0,
        f64::NEG_INFINITY,
        f64::NAN,
    ];
    let g = cpu_to_gpu(&row, &device).expect("upload");
    let (vals, idx) = gpu_topk_f64(g.inner(), 1, 6, 4, true, &device).expect("topk");
    let v = device.stream().clone_dtoh(&vals).expect("readback");
    let i = device.stream().clone_dtoh(&idx).expect("readback idx");
    assert!(v[0].is_nan() && v[1].is_nan(), "f64 NaN outranks +inf: {v:?}");
    for slot in 0..2 {
        assert!(row[i[slot] as usize].is_nan(), "NaN-slot indexes a NaN");
    }
    assert_eq!(v[2], f64::INFINITY);
    assert_eq!(v[3], 5.0);
}
