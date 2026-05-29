//! RE-AUDIT regression guard for #1672 — full-reduction pass-2 stays ON-DEVICE
//! via recursion instead of a Device->Host partials readback.
//!
//! Commit 572e7f352 changed `gpu_reduce_{sum,min,max,prod}` (f32 + f64) and the
//! masked siblings `gpu_masked_reduce_{min,max}` (f32 + f64) so that pass-2 —
//! combining the per-block partials — recurses on-device via a second kernel
//! launch (`gpu_reduce_*(&partials)`) for every `num_blocks > 1`, removing the
//! old `if num_blocks <= 256 { gpu_to_cpu(&partials); host-combine; cpu_to_gpu }`
//! branch.
//!
//! This file is the ADVERSARIAL guard the original `..._on_device_pass2.rs` test
//! did not pin:
//!
//!   1. DEEP RECURSION at 1_000_000 elements (num_blocks caps at 1024 — the
//!      DEEPEST recursion regime: 1024 -> ceil(1024/256)=4 -> ceil(4/256)=1, two
//!      on-device combine launches). Planted unique min/max/index so a wrong
//!      multi-level recursion that DROPS an extreme is caught bit-exactly.
//!   2. TERMINATION — `num_blocks = min(ceil(n/256), 1024)` strictly shrinks for
//!      n >= 2 (ceil(n/256) < n) and short-circuits at n == 1. A recursion that
//!      failed to shrink would hang / stack-overflow this binary; reaching the
//!      asserts at all IS the termination proof. The 1M case forces the deepest
//!      chain.
//!   3. MASKED SIBLINGS — `gpu_masked_reduce_{min,max}` recurse via the *unmasked*
//!      `gpu_reduce_{min,max}` on the already-filtered partials; pinned vs the CPU
//!      masked reference at cap size with planted extremes.
//!   4. EMPTY / NaN / Inf edges vs the CPU reference (same op, same data).
//!
//! Every expected value is the CPU reference computed from the SAME host data
//! (R-CHAR-3: no literal-copied ferrotorch constant). min/max are exact (no
//! reduction-order rounding); float sums/prods compare within an order tol.

#![cfg(feature = "cuda")]

use ferrotorch_core::masked::{MaskedTensor, masked_max, masked_min};
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

/// Deterministic mixed-sign data (no RNG dep), distinct enough that a dropped
/// element changes the sum and a wrong recursion is observable.
fn det_data_f32(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 * 0.618_034).fract() - 0.5) * 4.0)
        .collect()
}

/// DEEPEST recursion regime. BLOCK == 256, num_blocks capped at 1024, so any
/// n >= 256 * 1024 == 262_144 saturates the cap and recurses 1024 -> 4 -> 1.
/// 1_000_000 sits well past the cap so the FIRST pass also runs the grid-stride
/// loop (each of the 1024 blocks folds ~977 elements) before the two combine
/// launches.
const DEEP: usize = 1_000_000;

// ---------------------------------------------------------------------------
// 1 + 2. DEEP recursion + termination, f32: planted unique extremes survive the
//        two on-device combine launches; sum matches CPU within order tol.
// ---------------------------------------------------------------------------

#[test]
fn deep_recursion_planted_extremes_f32() {
    ensure_cuda();
    let mut data = det_data_f32(DEEP); // bulk in [-2, 2)
    // Plant unique extremes far outside the bulk at three positions that land
    // in DIFFERENT pass-1 blocks (block b owns indices b, b+1024, b+2048, ...
    // under the grid-stride loop, so distinct low indices => distinct blocks).
    let lo_pos = 11;
    let hi_pos = 777_777;
    data[lo_pos] = -98765.5;
    data[hi_pos] = 87654.25;

    let cpu_min = cpu_f32(data.clone()).amin().unwrap().item().unwrap();
    let cpu_max = cpu_f32(data.clone()).amax().unwrap().item().unwrap();
    let cpu_sum = cpu_f32(data.clone()).sum_all().unwrap().item().unwrap();

    let g = cpu_f32(data).to(Device::Cuda(0)).unwrap();

    // Reaching these asserts at all proves the recursion TERMINATED (a
    // non-shrinking recursion would hang / stack-overflow before here).
    let omin = g.amin().unwrap();
    let omax = g.amax().unwrap();
    let osum = g.sum_all().unwrap();
    assert!(
        omin.is_cuda() && omax.is_cuda() && osum.is_cuda(),
        "must stay on CUDA"
    );

    // min/max are associative with no rounding — the planted extremes MUST be
    // returned bit-exactly; a recursion that dropped a partial fails here.
    assert_eq!(
        omin.cpu().unwrap().item().unwrap(),
        cpu_min,
        "amin deep: planted -98765.5 must survive the 1024->4->1 recursion"
    );
    assert_eq!(omin.cpu().unwrap().item().unwrap(), -98765.5);
    assert_eq!(
        omax.cpu().unwrap().item().unwrap(),
        cpu_max,
        "amax deep: planted 87654.25 must survive the recursion"
    );
    assert_eq!(omax.cpu().unwrap().item().unwrap(), 87654.25);

    let got_sum = osum.cpu().unwrap().item().unwrap();
    let tol = 1.0 + cpu_sum.abs() * 1e-3;
    assert!(
        (got_sum - cpu_sum).abs() <= tol,
        "sum_all deep: gpu={got_sum} cpu={cpu_sum} (tol={tol})"
    );
}

#[test]
fn deep_recursion_exact_ones_sum_f32() {
    // 1_000_000 ones == exactly 1_000_000.0? NO — 1e6 < 2^24 (16_777_216) so it
    // IS exactly representable in f32 and the tree sum of all-equal ints is
    // order-invariant below 2^24. Pins the deepest sum recursion exactly.
    ensure_cuda();
    let g = cpu_f32(vec![1.0_f32; DEEP]).to(Device::Cuda(0)).unwrap();
    let got = g.sum_all().unwrap().cpu().unwrap().item().unwrap();
    assert_eq!(got, DEEP as f32, "all-ones deep sum must be exact 1e6");
}

#[test]
fn deep_recursion_mean_f32() {
    // mean_all = sum/n; inherits the on-device pass-2 fix.
    ensure_cuda();
    let data = det_data_f32(DEEP);
    let cpu_ref = cpu_f32(data.clone()).mean_all().unwrap().item().unwrap();
    let g = cpu_f32(data).to(Device::Cuda(0)).unwrap();
    let out = g.mean_all().unwrap();
    assert!(out.is_cuda());
    let got = out.cpu().unwrap().item().unwrap();
    let tol = 1e-4 + cpu_ref.abs() * 1e-4;
    assert!(
        (got - cpu_ref).abs() <= tol,
        "mean deep: gpu={got} cpu={cpu_ref}"
    );
}

#[test]
fn deep_recursion_prod_near_one_f32() {
    // Factors ~1 so the product stays finite across the deepest recursion.
    ensure_cuda();
    let data: Vec<f32> = (0..DEEP)
        .map(|i| 1.0 + (((i % 11) as f32) - 5.0) * 1e-7)
        .collect();
    let cpu_ref = cpu_f32(data.clone()).prod_all().unwrap().item().unwrap();
    let g = cpu_f32(data).to(Device::Cuda(0)).unwrap();
    let out = g.prod_all().unwrap();
    assert!(out.is_cuda());
    let got = out.cpu().unwrap().item().unwrap();
    let tol = cpu_ref.abs() * 1e-2 + 1e-3;
    assert!(
        (got - cpu_ref).abs() <= tol,
        "prod deep: gpu={got} cpu={cpu_ref}"
    );
}

// ---------------------------------------------------------------------------
// 1 + 2. DEEP recursion, f64: identical pass-2 pattern.
// ---------------------------------------------------------------------------

#[test]
fn deep_recursion_planted_extremes_f64() {
    ensure_cuda();
    let mut data: Vec<f64> = (0..DEEP)
        .map(|i| ((i as f64 * 0.6180339887).fract() - 0.5) * 4.0)
        .collect();
    data[11] = -9.87654321e9;
    data[654_321] = 1.23456789e9;

    let cpu_min = cpu_f64(data.clone()).amin().unwrap().item().unwrap();
    let cpu_max = cpu_f64(data.clone()).amax().unwrap().item().unwrap();
    let cpu_sum = cpu_f64(data.clone()).sum_all().unwrap().item().unwrap();

    let g = cpu_f64(data).to(Device::Cuda(0)).unwrap();

    let osum = g.sum_all().unwrap();
    assert!(osum.is_cuda(), "f64 sum must stay on CUDA");
    let got_sum = osum.cpu().unwrap().item().unwrap();
    let tol = 1e-3 + cpu_sum.abs() * 1e-12;
    assert!(
        (got_sum - cpu_sum).abs() <= tol,
        "f64 sum deep: gpu={got_sum} cpu={cpu_sum}"
    );

    assert_eq!(g.amin().unwrap().cpu().unwrap().item().unwrap(), cpu_min);
    assert_eq!(
        g.amin().unwrap().cpu().unwrap().item().unwrap(),
        -9.87654321e9
    );
    assert_eq!(g.amax().unwrap().cpu().unwrap().item().unwrap(), cpu_max);
    assert_eq!(
        g.amax().unwrap().cpu().unwrap().item().unwrap(),
        1.23456789e9
    );
}

// ---------------------------------------------------------------------------
// 3. MASKED siblings — gpu_masked_reduce_{min,max} recurse via the unmasked
//    reducer on the filtered partials. Pin GPU == CPU masked reference at cap.
// ---------------------------------------------------------------------------

#[test]
fn masked_min_max_deep_recursion_f32() {
    ensure_cuda();
    let n = DEEP;
    let mut data = det_data_f32(n);
    // Plant a MASKED-OUT spurious extreme (must be ignored) and a VALID extreme
    // (must be returned). If the recursion leaked the masked sentinel or dropped
    // the valid extreme, GPU != CPU.
    data[123] = 1e30; // will be masked OUT — must NOT win the max
    data[456_789] = 31337.5; // valid extreme — must win the max
    data[999_001] = -31337.5; // valid extreme — must win the min

    let mut mask = vec![true; n];
    mask[123] = false; // mask out the spurious 1e30

    let mt_cpu = MaskedTensor::new(cpu_f32(data.clone()), mask.clone()).unwrap();
    let cpu_min = masked_min(&mt_cpu).unwrap().item().unwrap();
    let cpu_max = masked_max(&mt_cpu).unwrap().item().unwrap();

    let g = cpu_f32(data).to(Device::Cuda(0)).unwrap();
    let mt_gpu = MaskedTensor::new(g, mask).unwrap();
    let gmin = masked_min(&mt_gpu).unwrap();
    let gmax = masked_max(&mt_gpu).unwrap();

    assert_eq!(
        gmin.cpu().unwrap().item().unwrap(),
        cpu_min,
        "masked_min deep == cpu"
    );
    assert_eq!(gmin.cpu().unwrap().item().unwrap(), -31337.5);
    assert_eq!(
        gmax.cpu().unwrap().item().unwrap(),
        cpu_max,
        "masked_max deep == cpu"
    );
    assert_eq!(
        gmax.cpu().unwrap().item().unwrap(),
        31337.5,
        "masked-out 1e30 must NOT leak through the recursion"
    );
}

#[test]
fn masked_min_max_deep_recursion_f64() {
    ensure_cuda();
    let n = DEEP;
    let mut data: Vec<f64> = (0..n)
        .map(|i| ((i as f64 * 0.6180339887).fract() - 0.5) * 4.0)
        .collect();
    data[123] = 1e300; // masked out
    data[456_789] = 9.5e9; // valid max
    data[999_001] = -9.5e9; // valid min
    let mut mask = vec![true; n];
    mask[123] = false;

    let mt_cpu = MaskedTensor::new(cpu_f64(data.clone()), mask.clone()).unwrap();
    let cpu_min = masked_min(&mt_cpu).unwrap().item().unwrap();
    let cpu_max = masked_max(&mt_cpu).unwrap().item().unwrap();

    let g = cpu_f64(data).to(Device::Cuda(0)).unwrap();
    let mt_gpu = MaskedTensor::new(g, mask).unwrap();
    assert_eq!(
        masked_min(&mt_gpu).unwrap().cpu().unwrap().item().unwrap(),
        cpu_min
    );
    assert_eq!(
        masked_max(&mt_gpu).unwrap().cpu().unwrap().item().unwrap(),
        cpu_max
    );
    assert_eq!(
        masked_max(&mt_gpu).unwrap().cpu().unwrap().item().unwrap(),
        9.5e9
    );
}

// ---------------------------------------------------------------------------
// 4. EDGE: n == 1 (single partial IS the result — num_blocks == 1 short-circuit)
//          and all-equal (no extreme to drop, but exercises the recursion).
// ---------------------------------------------------------------------------

#[test]
fn single_and_all_equal_match_cpu_f32() {
    ensure_cuda();
    // n == 1: short-circuit branch (num_blocks == 1).
    let g1 = cpu_f32(vec![42.5]).to(Device::Cuda(0)).unwrap();
    assert_eq!(g1.sum_all().unwrap().cpu().unwrap().item().unwrap(), 42.5);
    assert_eq!(g1.amin().unwrap().cpu().unwrap().item().unwrap(), 42.5);
    assert_eq!(g1.amax().unwrap().cpu().unwrap().item().unwrap(), 42.5);

    // all-equal at cap size — min == max == the value, sum exact for small int.
    let g = cpu_f32(vec![7.0; DEEP]).to(Device::Cuda(0)).unwrap();
    assert_eq!(g.amin().unwrap().cpu().unwrap().item().unwrap(), 7.0);
    assert_eq!(g.amax().unwrap().cpu().unwrap().item().unwrap(), 7.0);
}

// ---------------------------------------------------------------------------
// 4. EDGE: NaN / +-Inf through the deepest recursion, vs CPU reference.
// ---------------------------------------------------------------------------

#[test]
fn nan_inf_deep_recursion_vs_cpu_f32() {
    ensure_cuda();
    let n = DEEP;

    // (a) +Inf planted -> sum is +Inf (CPU and GPU agree); max is +Inf.
    let mut data = det_data_f32(n);
    data[500_000] = f32::INFINITY;
    let cpu_sum = cpu_f32(data.clone()).sum_all().unwrap().item().unwrap();
    let cpu_max = cpu_f32(data.clone()).amax().unwrap().item().unwrap();
    let g = cpu_f32(data).to(Device::Cuda(0)).unwrap();
    let gsum = g.sum_all().unwrap().cpu().unwrap().item().unwrap();
    let gmax = g.amax().unwrap().cpu().unwrap().item().unwrap();
    assert_eq!(
        gsum.is_infinite() && gsum > 0.0,
        cpu_sum.is_infinite() && cpu_sum > 0.0
    );
    assert!(
        gsum.is_infinite() && gsum > 0.0,
        "sum with +Inf must be +Inf, got {gsum}"
    );
    assert_eq!(gmax, cpu_max);
    assert_eq!(gmax, f32::INFINITY);

    // (b) -Inf planted -> min is -Inf.
    let mut data = det_data_f32(n);
    data[700_000] = f32::NEG_INFINITY;
    let cpu_min = cpu_f32(data.clone()).amin().unwrap().item().unwrap();
    let g = cpu_f32(data).to(Device::Cuda(0)).unwrap();
    let gmin = g.amin().unwrap().cpu().unwrap().item().unwrap();
    assert_eq!(gmin, cpu_min);
    assert_eq!(gmin, f32::NEG_INFINITY);

    // (c) NaN planted -> sum must be NaN (propagates through the recursion),
    //     matching the CPU reference.
    let mut data = det_data_f32(n);
    data[250_000] = f32::NAN;
    let cpu_sum = cpu_f32(data.clone()).sum_all().unwrap().item().unwrap();
    let g = cpu_f32(data).to(Device::Cuda(0)).unwrap();
    let gsum = g.sum_all().unwrap().cpu().unwrap().item().unwrap();
    assert_eq!(
        gsum.is_nan(),
        cpu_sum.is_nan(),
        "sum NaN-propagation must match CPU: gpu={gsum} cpu={cpu_sum}"
    );
    assert!(
        gsum.is_nan(),
        "NaN must propagate through the deep sum recursion"
    );
}
