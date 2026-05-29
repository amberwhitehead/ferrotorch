//! Divergence: GPU `topk` (largest=true) drops NaN from the top-k VALUES,
//! whereas upstream torch CUDA treats NaN as **greater than every finite value**
//! and returns it first.
//!
//! Upstream cite:
//!   `aten/src/ATen/native/cuda/TensorTopK.cpp:97`
//!     `launch_gather_topk_kernel(self, k, dim, largest, values, indices);`
//!   followed by `:101` `sortKeyValueInplace(values, indices, dim, largest)`.
//!   The selection + sort use CUDA's `THCNumerics`-style comparator, under which
//!   `NaN` orders as the maximum for `largest=true`. Verified on live torch
//!   2.11.0+cu130 (RTX 3090):
//!       torch.topk([3, NaN, 1, 5, NaN, 2], k=4, largest=True, sorted=True)
//!         -> values = [NaN, NaN, 5.0, 3.0]   (f32 AND f64)
//!         -> indices = [1, 4, 3, 0]
//!   i.e. the two NaN values are the two LARGEST elements and lead the result.
//!
//! ferrotorch cite:
//!   `ferrotorch-gpu/src/search.rs` `topk_ptx!` kernel + module note
//!     "`setp.gt`/`setp.lt`/`setp.eq` are ORDERED (false for NaN); a NaN value
//!      never outranks and is never eligible after a finite pick, so finite
//!      elements are selected first."
//!   For the same input the kernel therefore selects the four FINITE extrema
//!   `[5.0, 3.0, 2.0, 1.0]` and never returns NaN — the first two output VALUES
//!   diverge from torch (5.0 vs NaN, 3.0 vs NaN).
//!
//! This is NOT the (acceptable, torch-unspecified) tie-INDEX reordering the
//! generator documented: here the returned VALUES themselves are wrong. The
//! kernel returns a finite value in a slot where torch returns NaN.
//!
//! The CPU production path (`ferrotorch_core::ops::search::topk`) has the same
//! class of bug via `partial_cmp(...).unwrap_or(Ordering::Equal)`, but this file
//! pins the GPU kernel that commit 732341941 shipped.
//!
//! Tracking: #1648 (blocker).

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

const NAN: f32 = f32::NAN;
const NAN64: f64 = f64::NAN;

/// Input slice (single row, dim=6). Two NaNs at indices 1 and 4.
const ROW_F32: [f32; 6] = [3.0, NAN, 1.0, 5.0, NAN, 2.0];
const ROW_F64: [f64; 6] = [3.0, NAN64, 1.0, 5.0, NAN64, 2.0];

/// torch.topk(ROW, k=4, largest=True, sorted=True) on CUDA, 2.11.0+cu130:
///   values = [NaN, NaN, 5.0, 3.0]
/// The first two output positions are NaN (NaN is the largest under torch's
/// CUDA comparator). We assert position 0 and 1 are NaN; ferrotorch returns
/// finite 5.0 / 3.0 there.
#[test]
fn divergence_topk_f32_largest_nan_is_top() {
    let device = match ensure_cuda() {
        Some(d) => d,
        None => return,
    };
    let g = cpu_to_gpu(&ROW_F32, &device).expect("upload");
    let (vals, _idx) = gpu_topk_f32(g.inner(), 1, 6, 4, true, &device).expect("topk");
    let host = device.stream().clone_dtoh(&vals).expect("readback");

    // Upstream torch: the two largest elements are the NaNs -> output[0],[1] NaN.
    assert!(
        host[0].is_nan(),
        "torch CUDA topk(largest=True) returns NaN at position 0 (NaN is largest); \
         ferrotorch returned {} (finite value selected instead of NaN)",
        host[0]
    );
    assert!(
        host[1].is_nan(),
        "torch CUDA topk(largest=True) returns NaN at position 1; \
         ferrotorch returned {}",
        host[1]
    );
    // Positions 2,3 are the two largest finite values, descending.
    assert_eq!(
        host[2], 5.0_f32,
        "third element should be 5.0 (largest finite)"
    );
    assert_eq!(host[3], 3.0_f32, "fourth element should be 3.0");
}

/// f64 counterpart — same torch CUDA result `[NaN, NaN, 5.0, 3.0]`.
#[test]
fn divergence_topk_f64_largest_nan_is_top() {
    let device = match ensure_cuda() {
        Some(d) => d,
        None => return,
    };
    let g = cpu_to_gpu(&ROW_F64, &device).expect("upload");
    let (vals, _idx) = gpu_topk_f64(g.inner(), 1, 6, 4, true, &device).expect("topk");
    let host = device.stream().clone_dtoh(&vals).expect("readback");

    assert!(
        host[0].is_nan(),
        "torch CUDA topk(largest=True) returns NaN at position 0; ferrotorch returned {}",
        host[0]
    );
    assert!(
        host[1].is_nan(),
        "torch CUDA topk(largest=True) returns NaN at position 1; ferrotorch returned {}",
        host[1]
    );
    assert_eq!(host[2], 5.0_f64);
    assert_eq!(host[3], 3.0_f64);
}

/// largest=False (smallest) with NaN. torch ranks NaN as the MAXIMUM under the
/// same comparator (`GTOp`/`LTOp` handleNaN, `SortingCommon.cuh:47-60`), so for
/// ascending sort NaN goes to the END and `topk(largest=False)` does NOT pick a
/// NaN until the finite values are exhausted. Verified live on torch
/// 2.11.0+cu130 (RTX 3090):
///   torch.topk([3,NaN,1,5,NaN,2], k=4, largest=False) -> [1,2,3,5] idx [2,5,0,3]
///   torch.topk([3,NaN,1,5,NaN,2], k=6, largest=False) -> [1,2,3,5,NaN,NaN]
///                                                         idx [2,5,0,3,1,4]
#[test]
fn divergence_topk_f32_smallest_nan_is_last() {
    let device = match ensure_cuda() {
        Some(d) => d,
        None => return,
    };
    let g = cpu_to_gpu(&ROW_F32, &device).expect("upload");

    // k=4 smallest: the four finite extrema, ascending; NO NaN selected.
    let (vals, idx) = gpu_topk_f32(g.inner(), 1, 6, 4, false, &device).expect("topk k4");
    let host = device.stream().clone_dtoh(&vals).expect("readback");
    let hidx = device.stream().clone_dtoh(&idx).expect("readback idx");
    assert!(
        host.iter().take(4).all(|v| !v.is_nan()),
        "torch topk(largest=False, k=4) selects no NaN (NaN ranks last); got {host:?}"
    );
    assert_eq!(&host[..4], &[1.0_f32, 2.0, 3.0, 5.0]);
    assert_eq!(&hidx[..4], &[2_i64, 5, 0, 3]);

    // k=6 == numel: now the finite values are exhausted and the two NaNs land
    // in the LAST two positions (NaN ranks as the maximum under ascending sort).
    let (vals6, idx6) = gpu_topk_f32(g.inner(), 1, 6, 6, false, &device).expect("topk k6");
    let host6 = device.stream().clone_dtoh(&vals6).expect("readback6");
    let hidx6 = device.stream().clone_dtoh(&idx6).expect("readback6 idx");
    assert_eq!(&host6[..4], &[1.0_f32, 2.0, 3.0, 5.0]);
    assert!(
        host6[4].is_nan() && host6[5].is_nan(),
        "NaNs last: {host6:?}"
    );
    assert_eq!(&hidx6[..4], &[2_i64, 5, 0, 3]);
    // Two NaNs in ascending original-index order (indices 1 and 4).
    assert_eq!(&hidx6[4..], &[1_i64, 4]);
}
