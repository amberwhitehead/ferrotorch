//! Divergence probe: GPU `searchsorted` NaN-value handling vs LIVE torch 2.11.
//!
//! The PTX kernels in `ferrotorch-gpu/src/search.rs` build the binary-search
//! advance predicate from `setp.lt.f32`/`setp.le.f32`:
//!
//!   left  (lower_bound): advance while `bv <  v`   (`setp.lt`)
//!   right (upper_bound): advance while `bv <= v`   (`setp.le`)
//!
//! Upstream `aten/src/ATen/native/cuda/Bucketization.cu:33` advances on the
//! NEGATION of the opposite comparison instead:
//!
//!   lower_bound: `if (!(mid_val >= val)) start = mid + 1;`
//!   upper_bound: `if (!(mid_val > val))  start = mid + 1;`
//!
//! For finite operands `(bv < v) == !(bv >= v)` and `(bv <= v) == !(bv > v)`,
//! so the two formulations agree. They DIVERGE when `v` is NaN:
//!
//!   - `bv <  NaN`  -> false  (kernel: never advance -> lo stays 0)
//!   - `!(bv >= NaN)` -> `!false` -> true (upstream: always advance -> lo = end)
//!
//! So torch places a NaN value at index `len(boundaries)` (one past the end,
//! BOTH sides), while the ferrotorch GPU kernel places it at 0.
//!
//! LIVE torch 2.11.0+cu130 oracle (b=[1,3,5,7], v=[NaN,2]):
//!   torch.searchsorted(b, v, right=False) -> [4, 1]
//!   torch.searchsorted(b, v, right=True)  -> [4, 1]
//! i.e. the NaN value resolves to index 4 (== len) on BOTH sides.

#![cfg(feature = "cuda")]

use ferrotorch_core::ops::search::searchsorted;
use ferrotorch_core::{Device, Tensor, TensorStorage};
use ferrotorch_gpu::init_cuda_backend;

fn ensure_cuda() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        init_cuda_backend().expect("CUDA backend init");
    });
}

fn cpu_f32(data: &[f32]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false).unwrap()
}

fn cpu_f64(data: &[f64]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false).unwrap()
}

/// Divergence: `ferrotorch_gpu::gpu_searchsorted_f32` (via
/// `ferrotorch_core::ops::search::searchsorted` CUDA branch) diverges from
/// `pytorch aten/src/ATen/native/cuda/Bucketization.cu:33` for a NaN value.
/// Upstream advances on `!(mid_val >= val)` (true for NaN) -> index == len;
/// ferrotorch advances on `setp.lt.f32` (false for NaN) -> index 0.
/// Live torch 2.11: searchsorted([1,3,5,7], [NaN,2], right=False) == [4, 1].
/// Tracking: #1645
#[test]
fn divergence_searchsorted_f32_nan_value_left() {
    ensure_cuda();
    let bounds = cpu_f32(&[1.0, 3.0, 5.0, 7.0]).to(Device::Cuda(0)).unwrap();
    let vals = cpu_f32(&[f32::NAN, 2.0]).to(Device::Cuda(0)).unwrap();
    assert!(bounds.is_cuda() && vals.is_cuda());
    let got = searchsorted(&bounds, &vals, false).unwrap();
    // LIVE torch 2.11.0+cu130: torch.searchsorted(b, v, right=False) -> [4, 1]
    assert_eq!(got, vec![4, 1]);
}

/// Same divergence on the right (upper_bound) side. Live torch 2.11:
/// searchsorted([1,3,5,7], [NaN,2], right=True) == [4, 1] — the NaN still
/// resolves to index 4 (== len). ferrotorch GPU kernel returns 0 for the NaN.
/// Tracking: #1645
#[test]
fn divergence_searchsorted_f32_nan_value_right() {
    ensure_cuda();
    let bounds = cpu_f32(&[1.0, 3.0, 5.0, 7.0]).to(Device::Cuda(0)).unwrap();
    let vals = cpu_f32(&[f32::NAN, 2.0]).to(Device::Cuda(0)).unwrap();
    let got = searchsorted(&bounds, &vals, true).unwrap();
    // LIVE torch 2.11.0+cu130: torch.searchsorted(b, v, right=True) -> [4, 1]
    assert_eq!(got, vec![4, 1]);
}

/// f64 NaN value, both sides. Live torch 2.11: a NaN value resolves to
/// index == len(boundaries) regardless of `right`. Mirrors the f64 PTX kernel
/// (`setp.lt.f64`/`setp.le.f64`) carrying the same NaN miscompute.
/// Tracking: #1645
#[test]
fn divergence_searchsorted_f64_nan_value_both_sides() {
    ensure_cuda();
    let bounds = cpu_f64(&[1.0, 3.0, 5.0, 7.0]).to(Device::Cuda(0)).unwrap();
    let vals = cpu_f64(&[f64::NAN]).to(Device::Cuda(0)).unwrap();
    let left = searchsorted(&bounds, &vals, false).unwrap();
    let right = searchsorted(&bounds, &vals, true).unwrap();
    // LIVE torch 2.11.0+cu130: NaN value -> index 4 (== len) on both sides.
    assert_eq!(left, vec![4]);
    assert_eq!(right, vec![4]);
}
