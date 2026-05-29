//! Full-reduction (sum_all / mean_all / prod_all / amin / amax) pass-2
//! ON-DEVICE correctness on a live RTX 3090 (#1672).
//!
//! Before #1672 the second pass of `gpu_reduce_{sum,min,max,prod}` (combining
//! the per-block partials) did a blocking Device->Host readback + sync for
//! `1 < num_blocks <= 256` partials. For a >= 256K-element input the first
//! pass hits the `num_blocks == 1024` cap, recurses to 4 partials, then took
//! the host-readback branch every call — ~110x slower than the axis path.
//!
//! These tests pin the two invariants the fix must preserve:
//!   1. CORRECTNESS — the CUDA full reduction matches the CPU reference value
//!      computed from the SAME host data (within f32 reduction-order
//!      tolerance; full-reduction order is unspecified, torch included).
//!   2. RESIDENCY — the result tensor stays `is_cuda()` (the input data never
//!      round-trips to the host mid-reduction; only the final scalar is
//!      materialised on-device).
//!
//! Sizes are chosen to exercise BOTH regimes:
//!   - small  (< 256 elements, num_blocks == 1 — single partial IS result)
//!   - medium (1 < num_blocks <= 256 — the old host-readback branch)
//!   - cap    (>= 256K elements, num_blocks == 1024 — recurses twice)

#![cfg(feature = "cuda")]

use ferrotorch_core::{Device, Tensor, TensorStorage};
use ferrotorch_gpu::init_cuda_backend;

fn ensure_cuda() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        init_cuda_backend().expect("CUDA backend init");
    });
}

fn cpu_f32(data: Vec<f32>) -> Tensor<f32> {
    let n = data.len();
    Tensor::from_storage(TensorStorage::cpu(data), vec![n], false).unwrap()
}

fn cpu_f64(data: Vec<f64>) -> Tensor<f64> {
    let n = data.len();
    Tensor::from_storage(TensorStorage::cpu(data), vec![n], false).unwrap()
}

/// Deterministic pseudo-random f32 data in a bounded range (no RNG dep).
/// Mixed signs keep min != max and exercise the sign of the reduction.
fn det_data_f32(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i as f32 * 0.618_034).fract(); // golden-ratio hash in [0,1)
            (x - 0.5) * 4.0 // -> [-2.0, 2.0)
        })
        .collect()
}

/// The three block regimes. BLOCK == 256, num_blocks capped at 1024, so
/// >= 256 * 1024 == 262_144 elements saturate the cap.
const SMALL: usize = 200; // num_blocks == 1
const MEDIUM: usize = 50_000; // 1 < num_blocks <= 256 (was host-readback)
const CAP: usize = 300_000; // num_blocks == 1024 (recurses twice)

// ---------------------------------------------------------------------------
// sum_all
// ---------------------------------------------------------------------------

#[test]
fn sum_all_cuda_matches_cpu_all_regimes_f32() {
    ensure_cuda();
    for &n in &[SMALL, MEDIUM, CAP] {
        let data = det_data_f32(n);
        let cpu_ref = cpu_f32(data.clone()).sum_all().unwrap().item().unwrap();

        let g = cpu_f32(data).to(Device::Cuda(0)).unwrap();
        let out = g.sum_all().unwrap();
        // RESIDENCY: the reduction result stayed on the device.
        assert!(out.is_cuda(), "sum_all result must stay on CUDA (n={n})");
        let got = out.cpu().unwrap().item().unwrap();
        // CORRECTNESS within reduction-order tolerance (relative to magnitude).
        let tol = 1e-2 + cpu_ref.abs() * 1e-4;
        assert!(
            (got - cpu_ref).abs() <= tol,
            "sum_all n={n}: gpu={got} cpu={cpu_ref} (tol={tol})"
        );
    }
}

#[test]
fn sum_all_cuda_exact_ones_cap_f32() {
    // 300_000 ones -> exactly 300_000.0 (representable in f32). The
    // tree-reduction order cannot perturb an all-equal-integer sum below the
    // f32 integer-exactness limit (2^24), so this pins the cap path exactly.
    ensure_cuda();
    let g = cpu_f32(vec![1.0_f32; CAP]).to(Device::Cuda(0)).unwrap();
    let out = g.sum_all().unwrap();
    assert!(out.is_cuda());
    let got = out.cpu().unwrap().item().unwrap();
    assert_eq!(got, CAP as f32, "all-ones cap sum must be exact");
}

// ---------------------------------------------------------------------------
// mean_all (inherits the sum pass-2 fix, then divides)
// ---------------------------------------------------------------------------

#[test]
fn mean_all_cuda_matches_cpu_all_regimes_f32() {
    ensure_cuda();
    for &n in &[SMALL, MEDIUM, CAP] {
        let data = det_data_f32(n);
        let cpu_ref = cpu_f32(data.clone()).mean_all().unwrap().item().unwrap();

        let g = cpu_f32(data).to(Device::Cuda(0)).unwrap();
        let out = g.mean_all().unwrap();
        assert!(out.is_cuda(), "mean_all result must stay on CUDA (n={n})");
        let got = out.cpu().unwrap().item().unwrap();
        let tol = 1e-4 + cpu_ref.abs() * 1e-4;
        assert!(
            (got - cpu_ref).abs() <= tol,
            "mean_all n={n}: gpu={got} cpu={cpu_ref} (tol={tol})"
        );
    }
}

// ---------------------------------------------------------------------------
// amin / amax (min/max have NO reduction-order error — exact regardless)
// ---------------------------------------------------------------------------

#[test]
fn amin_amax_cuda_match_cpu_all_regimes_f32() {
    ensure_cuda();
    for &n in &[SMALL, MEDIUM, CAP] {
        let data = det_data_f32(n);
        let cpu_min = cpu_f32(data.clone()).amin().unwrap().item().unwrap();
        let cpu_max = cpu_f32(data.clone()).amax().unwrap().item().unwrap();

        let g = cpu_f32(data).to(Device::Cuda(0)).unwrap();

        let omin = g.amin().unwrap();
        let omax = g.amax().unwrap();
        assert!(omin.is_cuda(), "amin result must stay on CUDA (n={n})");
        assert!(omax.is_cuda(), "amax result must stay on CUDA (n={n})");

        // min/max are associative+commutative with no rounding — exact match.
        assert_eq!(omin.cpu().unwrap().item().unwrap(), cpu_min, "amin n={n}");
        assert_eq!(omax.cpu().unwrap().item().unwrap(), cpu_max, "amax n={n}");
    }
}

#[test]
fn amin_amax_cuda_unique_extremes_cap_f32() {
    // Plant a unique min and max far from the bulk at cap size so the extreme
    // must propagate through two on-device combine launches intact.
    ensure_cuda();
    let mut data = vec![0.5_f32; CAP];
    data[CAP / 3] = -123.5;
    data[2 * CAP / 3] = 456.25;
    let g = cpu_f32(data).to(Device::Cuda(0)).unwrap();
    assert_eq!(g.amin().unwrap().cpu().unwrap().item().unwrap(), -123.5);
    assert_eq!(g.amax().unwrap().cpu().unwrap().item().unwrap(), 456.25);
}

// ---------------------------------------------------------------------------
// prod_all
// ---------------------------------------------------------------------------

#[test]
fn prod_all_cuda_unit_factors_cap_f32() {
    // All-ones product == 1.0 exactly at cap size (multi-launch combine).
    ensure_cuda();
    let g = cpu_f32(vec![1.0_f32; CAP]).to(Device::Cuda(0)).unwrap();
    let out = g.prod_all().unwrap();
    assert!(out.is_cuda(), "prod_all result must stay on CUDA");
    let got = out.cpu().unwrap().item().unwrap();
    assert!((got - 1.0).abs() < 1e-3, "all-ones cap prod: {got}");
}

#[test]
fn prod_all_cuda_matches_cpu_medium_f32() {
    // Factors near 1.0 so the product neither overflows nor underflows across
    // the medium regime (the former host-readback branch).
    ensure_cuda();
    let n = MEDIUM;
    let data: Vec<f32> = (0..n)
        .map(|i| 1.0 + (((i % 7) as f32) - 3.0) * 1e-5)
        .collect();
    let cpu_ref = cpu_f32(data.clone()).prod_all().unwrap().item().unwrap();
    let g = cpu_f32(data).to(Device::Cuda(0)).unwrap();
    let out = g.prod_all().unwrap();
    assert!(out.is_cuda());
    let got = out.cpu().unwrap().item().unwrap();
    let tol = cpu_ref.abs() * 1e-3;
    assert!(
        (got - cpu_ref).abs() <= tol,
        "prod_all n={n}: gpu={got} cpu={cpu_ref}"
    );
}

// ---------------------------------------------------------------------------
// f64 paths share the identical pass-2 pattern (#1672 fixed them too).
// ---------------------------------------------------------------------------

#[test]
fn sum_amin_amax_cuda_match_cpu_cap_f64() {
    ensure_cuda();
    let n = CAP;
    let data: Vec<f64> = (0..n)
        .map(|i| ((i as f64 * 0.6180339887).fract() - 0.5) * 4.0)
        .collect();
    let cpu_sum = cpu_f64(data.clone()).sum_all().unwrap().item().unwrap();
    let cpu_min = cpu_f64(data.clone()).amin().unwrap().item().unwrap();
    let cpu_max = cpu_f64(data.clone()).amax().unwrap().item().unwrap();

    let g = cpu_f64(data).to(Device::Cuda(0)).unwrap();

    let osum = g.sum_all().unwrap();
    assert!(osum.is_cuda(), "f64 sum_all must stay on CUDA");
    let got_sum = osum.cpu().unwrap().item().unwrap();
    let tol = 1e-6 + cpu_sum.abs() * 1e-9;
    assert!(
        (got_sum - cpu_sum).abs() <= tol,
        "f64 sum_all cap: gpu={got_sum} cpu={cpu_sum}"
    );

    assert_eq!(g.amin().unwrap().cpu().unwrap().item().unwrap(), cpu_min);
    assert_eq!(g.amax().unwrap().cpu().unwrap().item().unwrap(), cpu_max);
}
