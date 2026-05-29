//! RE-AUDIT (commit f544819ef, #1656 fix) of the GPU `unique_consecutive`
//! run-start predicate change `setp.ne` (ordered) -> `setp.neu` (unordered) at
//! the 4 PTX sites (RUN_FLAG f32/f64 + COMPACT f32/f64,
//! `ferrotorch-gpu/src/search.rs:1545,:1610,:1672,:1736`).
//!
//! The prior re-audit (`divergence_roll_unique_consecutive_gpu_reaudit.rs`)
//! pinned the two consecutive-NaN cases that the fix targeted; those now PASS on
//! hardware. THIS file pins the edges that change did NOT directly touch but
//! that the change could have regressed, per the re-audit brief:
//!
//!   - The CRITICAL mixed case: NaN runs + finite runs + finite duplicates in
//!     one array. `inverse`/`counts` are derived host-side from the RUN_FLAG ->
//!     cumsum `incl` array, while the COMPACT kernel re-derives run-starts
//!     INDEPENDENTLY from its own `setp.neu`. If the two predicates disagreed,
//!     `out_len` (RUN_FLAG prefix-sum) would not match the scatter positions
//!     (COMPACT) and the counts/inverse would mismatch torch even when the
//!     values look plausible. This test pins values AND counts AND inverse.
//!   - Signed zero: IEEE `-0.0 == +0.0`; `setp.neu` returns FALSE for them (same
//!     as `setp.ne`), so `[+0.0, -0.0]` is ONE run keeping the run-start value
//!     `+0.0`. Confirms the `neu` change touched ONLY NaN behavior, not signed
//!     zero.
//!   - Infinity: `[inf, inf]` -> one run; `[inf, -inf]` -> two runs
//!     (`inf != -inf`, `neu` -> true). Confirms ordered-comparable finite/inf
//!     values still compute the right run boundaries.
//!   - Finite no-regression: all-same, no-dup, alternating still correct f32+f64.
//!
//! R-CHAR-3: every expected value below is the LIVE output of
//! `torch.unique_consecutive(t, return_inverse=True, return_counts=True)` on
//! torch 2.11.0+cu130 / RTX 3090, recorded here as a named torch reference (the
//! oracle run is reproduced in the brief), NOT copied from the ferrotorch GPU
//! side. The GPU result is additionally asserted equal to the ferrotorch CPU
//! path (itself torch-parity via the `data[i] == data[i-1]` PartialEq scan) on
//! identical data.
//!
//! Verdict at audit time: FIX COMPLETE — all cases PASS. This is a PASSING
//! regression guard (left un-`#[ignore]`d so any future predicate regression
//! re-breaks CI).

#![cfg(feature = "cuda")]

use ferrotorch_core::ops::search::unique_consecutive;
use ferrotorch_core::{Device, Tensor, TensorStorage};
use std::sync::Once;

static INIT: Once = Once::new();

fn ensure_cuda() {
    INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend().expect("CUDA backend init");
    });
}

fn cuda_f32(data: &[f32]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false)
        .expect("cpu f32 tensor")
        .to(Device::Cuda(0))
        .expect("upload f32 to cuda")
}

fn cuda_f64(data: &[f64]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false)
        .expect("cpu f64 tensor")
        .to(Device::Cuda(0))
        .expect("upload f64 to cuda")
}

fn read_back_f32(t: &Tensor<f32>) -> Vec<f32> {
    t.to(Device::Cpu)
        .expect("download")
        .data_vec()
        .expect("data")
}

fn read_back_f64(t: &Tensor<f64>) -> Vec<f64> {
    t.to(Device::Cpu)
        .expect("download")
        .data_vec()
        .expect("data")
}

// Compare a GPU value vector to torch, treating NaN-vs-NaN as equal (torch
// preserves the NaN run-start; bit identity of the NaN payload is not part of
// the contract).
fn assert_vals_f32(got: &[f32], expect_nan_mask: &[bool], expect_finite: &[f32]) {
    assert_eq!(got.len(), expect_nan_mask.len(), "wrong out_len");
    for (i, &g) in got.iter().enumerate() {
        if expect_nan_mask[i] {
            assert!(g.is_nan(), "out[{i}] expected NaN, got {g}");
        } else {
            assert_eq!(
                g.to_bits(),
                expect_finite[i].to_bits(),
                "out[{i}] bit mismatch"
            );
        }
    }
}

fn assert_vals_f64(got: &[f64], expect_nan_mask: &[bool], expect_finite: &[f64]) {
    assert_eq!(got.len(), expect_nan_mask.len(), "wrong out_len");
    for (i, &g) in got.iter().enumerate() {
        if expect_nan_mask[i] {
            assert!(g.is_nan(), "out[{i}] expected NaN, got {g}");
        } else {
            assert_eq!(
                g.to_bits(),
                expect_finite[i].to_bits(),
                "out[{i}] bit mismatch"
            );
        }
    }
}

// ===========================================================================
// CRITICAL — RUN_FLAG / COMPACT agreement on a mixed NaN + finite + dup array
// ===========================================================================

/// Re-audit of `ferrotorch-gpu/src/search.rs` `gpu_unique_consecutive_f32`
/// against `pytorch aten/src/ATen/native/cuda/Unique.cu:63` (`not_equal`
/// functor, IEEE `lhs != rhs`) for the mixed input
/// `[nan, nan, 5, 5, 5, nan, 3]`.
///
/// This is the RUN_FLAG/COMPACT-agreement probe: `inverse`/`counts` come from
/// the RUN_FLAG -> cumsum `incl` array; the compacted VALUES come from the
/// COMPACT kernel's independent `setp.neu` re-derivation. Both must flag run
/// boundaries identically or counts/inverse desync from the scatter.
///
/// Live torch (2.11.0+cu130, RTX 3090):
///   values=[nan, nan, 5.0, nan, 3.0]  inverse=[0,1,2,2,2,3,4]  counts=[1,1,3,1,1]
/// 5 runs: NaN, NaN, the 5.0-triple, the trailing NaN, the 3.0.
#[test]
fn unique_consecutive_f32_mixed_nan_finite_dup_runflag_compact_agree() {
    ensure_cuda();
    let x: Vec<f32> = vec![f32::NAN, f32::NAN, 5.0, 5.0, 5.0, f32::NAN, 3.0];
    let xg = cuda_f32(&x);
    let (vals, inverse, counts) = unique_consecutive(&xg).expect("gpu unique mixed");
    assert!(vals.is_cuda(), "unique values must stay on device");
    let got = read_back_f32(&vals);

    // torch reference (named).
    let torch_nan_mask = [true, true, false, true, false];
    let torch_finite = [0.0f32, 0.0, 5.0, 0.0, 3.0];
    let torch_inverse = vec![0usize, 1, 2, 2, 2, 3, 4];
    let torch_counts = vec![1usize, 1, 3, 1, 1];

    assert_vals_f32(&got, &torch_nan_mask, &torch_finite);
    assert_eq!(
        inverse, torch_inverse,
        "inverse desync (RUN_FLAG vs COMPACT)"
    );
    assert_eq!(counts, torch_counts, "counts desync (RUN_FLAG vs COMPACT)");
    // counts must sum to n (no run dropped or double-counted).
    assert_eq!(counts.iter().sum::<usize>(), x.len());

    // GPU == ferrotorch CPU path on identical data (CPU is torch-parity).
    let xc = Tensor::from_storage(TensorStorage::cpu(x.clone()), vec![x.len()], false).unwrap();
    let (vc, ic, cc) = unique_consecutive(&xc).unwrap();
    let cpu_vals = read_back_f32(&vc);
    assert_vals_f32(&cpu_vals, &torch_nan_mask, &torch_finite);
    assert_eq!(inverse, ic, "GPU inverse diverges from CPU");
    assert_eq!(counts, cc, "GPU counts diverges from CPU");
}

/// Same mixed case in f64 (8-byte value path, RUN_FLAG_F64 + COMPACT_F64).
/// Live torch: identical structure — values=[nan,nan,5,nan,3],
/// inverse=[0,1,2,2,2,3,4], counts=[1,1,3,1,1].
#[test]
fn unique_consecutive_f64_mixed_nan_finite_dup_runflag_compact_agree() {
    ensure_cuda();
    let x: Vec<f64> = vec![f64::NAN, f64::NAN, 5.0, 5.0, 5.0, f64::NAN, 3.0];
    let xg = cuda_f64(&x);
    let (vals, inverse, counts) = unique_consecutive(&xg).expect("gpu unique mixed f64");
    assert!(vals.is_cuda());
    let got = read_back_f64(&vals);

    let torch_nan_mask = [true, true, false, true, false];
    let torch_finite = [0.0f64, 0.0, 5.0, 0.0, 3.0];
    let torch_inverse = vec![0usize, 1, 2, 2, 2, 3, 4];
    let torch_counts = vec![1usize, 1, 3, 1, 1];

    assert_vals_f64(&got, &torch_nan_mask, &torch_finite);
    assert_eq!(
        inverse, torch_inverse,
        "inverse desync (RUN_FLAG_F64 vs COMPACT_F64)"
    );
    assert_eq!(
        counts, torch_counts,
        "counts desync (RUN_FLAG_F64 vs COMPACT_F64)"
    );
    assert_eq!(counts.iter().sum::<usize>(), x.len());

    let xc = Tensor::from_storage(TensorStorage::cpu(x.clone()), vec![x.len()], false).unwrap();
    let (vc, ic, cc) = unique_consecutive(&xc).unwrap();
    assert_vals_f64(&read_back_f64(&vc), &torch_nan_mask, &torch_finite);
    assert_eq!(inverse, ic);
    assert_eq!(counts, cc);
}

// ===========================================================================
// Signed zero — neu (like ne) returns FALSE for -0.0 vs +0.0 (they are equal)
// ===========================================================================

/// `[+0.0, -0.0]`: IEEE `-0.0 == +0.0`, so `setp.neu` -> FALSE -> ONE run.
/// Live torch keeps the run-start value `+0.0` (bits 0x00000000):
///   values=[0.0]  inverse=[0,0]  counts=[2].
/// Confirms the `ne`->`neu` change did NOT alter signed-zero semantics.
#[test]
fn unique_consecutive_f32_signed_zero_is_one_run() {
    ensure_cuda();
    let xg = cuda_f32(&[0.0f32, -0.0f32]);
    let (vals, inverse, counts) = unique_consecutive(&xg).expect("gpu unique signed-zero");
    assert!(vals.is_cuda());
    let got = read_back_f32(&vals);
    assert_eq!(got.len(), 1, "torch collapses +0.0/-0.0 into one run");
    // torch keeps the FIRST element's bits (+0.0 = 0x00000000).
    assert_eq!(
        got[0].to_bits(),
        0x0000_0000u32,
        "run-start value must be +0.0"
    );
    assert_eq!(inverse, vec![0, 0]);
    assert_eq!(counts, vec![2]);
}

/// f64 signed-zero: `[+0.0, -0.0]` -> one run, value +0.0 (bits 0x0..0).
/// Live torch: values=[0.0], inverse=[0,0], counts=[2].
#[test]
fn unique_consecutive_f64_signed_zero_is_one_run() {
    ensure_cuda();
    let xg = cuda_f64(&[0.0f64, -0.0f64]);
    let (vals, inverse, counts) = unique_consecutive(&xg).expect("gpu unique signed-zero f64");
    assert!(vals.is_cuda());
    let got = read_back_f64(&vals);
    assert_eq!(got.len(), 1);
    assert_eq!(
        got[0].to_bits(),
        0x0000_0000_0000_0000u64,
        "run-start value must be +0.0"
    );
    assert_eq!(inverse, vec![0, 0]);
    assert_eq!(counts, vec![2]);
}

// ===========================================================================
// Infinity — ordered-comparable: [inf,inf]->1 run; [inf,-inf]->2 runs
// ===========================================================================

/// `[inf, inf]` -> one run (`inf == inf`, neu FALSE); `[inf, -inf]` -> two runs
/// (`inf != -inf`, neu TRUE). Live torch:
///   [inf,inf]:  values=[inf]      inverse=[0,0]  counts=[2]
///   [inf,-inf]: values=[inf,-inf] inverse=[0,1]  counts=[1,1]
#[test]
fn unique_consecutive_f32_infinity_runs() {
    ensure_cuda();
    // inf, inf -> one run.
    let g1 = cuda_f32(&[f32::INFINITY, f32::INFINITY]);
    let (v1, i1, c1) = unique_consecutive(&g1).expect("gpu unique inf inf");
    let r1 = read_back_f32(&v1);
    assert_eq!(r1.len(), 1, "[inf,inf] is one run");
    assert_eq!(r1[0], f32::INFINITY);
    assert_eq!(i1, vec![0, 0]);
    assert_eq!(c1, vec![2]);

    // inf, -inf -> two runs.
    let g2 = cuda_f32(&[f32::INFINITY, f32::NEG_INFINITY]);
    let (v2, i2, c2) = unique_consecutive(&g2).expect("gpu unique inf -inf");
    let r2 = read_back_f32(&v2);
    assert_eq!(r2.len(), 2, "[inf,-inf] is two runs");
    assert_eq!(r2[0], f32::INFINITY);
    assert_eq!(r2[1], f32::NEG_INFINITY);
    assert_eq!(i2, vec![0, 1]);
    assert_eq!(c2, vec![1, 1]);
}

/// f64 infinity: identical structure to f32.
#[test]
fn unique_consecutive_f64_infinity_runs() {
    ensure_cuda();
    let g1 = cuda_f64(&[f64::INFINITY, f64::INFINITY]);
    let (v1, i1, c1) = unique_consecutive(&g1).expect("gpu unique inf inf f64");
    let r1 = read_back_f64(&v1);
    assert_eq!(r1, vec![f64::INFINITY]);
    assert_eq!(i1, vec![0, 0]);
    assert_eq!(c1, vec![2]);

    let g2 = cuda_f64(&[f64::INFINITY, f64::NEG_INFINITY]);
    let (v2, i2, c2) = unique_consecutive(&g2).expect("gpu unique inf -inf f64");
    let r2 = read_back_f64(&v2);
    assert_eq!(r2, vec![f64::INFINITY, f64::NEG_INFINITY]);
    assert_eq!(i2, vec![0, 1]);
    assert_eq!(c2, vec![1, 1]);
}

// ===========================================================================
// Finite no-regression — neu must not have broken ordered-finite run detection
// ===========================================================================

/// all-same / no-dup / alternating, f32 AND f64. Live torch:
///   [7,7,7]   -> values=[7]       inverse=[0,0,0] counts=[3]
///   [1,2,3,4] -> values=[1,2,3,4] inverse=[0,1,2,3] counts=[1,1,1,1]
///   [1,2,1,2] -> values=[1,2,1,2] inverse=[0,1,2,3] counts=[1,1,1,1]
#[test]
fn unique_consecutive_finite_no_regression() {
    ensure_cuda();

    // all-same f32
    let (v, i, c) = unique_consecutive(&cuda_f32(&[7.0, 7.0, 7.0])).unwrap();
    assert_eq!(read_back_f32(&v), vec![7.0]);
    assert_eq!(i, vec![0, 0, 0]);
    assert_eq!(c, vec![3]);

    // no-dup f32
    let (v, i, c) = unique_consecutive(&cuda_f32(&[1.0, 2.0, 3.0, 4.0])).unwrap();
    assert_eq!(read_back_f32(&v), vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(i, vec![0, 1, 2, 3]);
    assert_eq!(c, vec![1, 1, 1, 1]);

    // alternating f32
    let (v, i, c) = unique_consecutive(&cuda_f32(&[1.0, 2.0, 1.0, 2.0])).unwrap();
    assert_eq!(read_back_f32(&v), vec![1.0, 2.0, 1.0, 2.0]);
    assert_eq!(i, vec![0, 1, 2, 3]);
    assert_eq!(c, vec![1, 1, 1, 1]);

    // all-same f64
    let (v, i, c) = unique_consecutive(&cuda_f64(&[7.0, 7.0, 7.0])).unwrap();
    assert_eq!(read_back_f64(&v), vec![7.0]);
    assert_eq!(i, vec![0, 0, 0]);
    assert_eq!(c, vec![3]);

    // no-dup f64
    let (v, i, c) = unique_consecutive(&cuda_f64(&[1.0, 2.0, 3.0, 4.0])).unwrap();
    assert_eq!(read_back_f64(&v), vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(i, vec![0, 1, 2, 3]);
    assert_eq!(c, vec![1, 1, 1, 1]);

    // alternating f64
    let (v, i, c) = unique_consecutive(&cuda_f64(&[1.0, 2.0, 1.0, 2.0])).unwrap();
    assert_eq!(read_back_f64(&v), vec![1.0, 2.0, 1.0, 2.0]);
    assert_eq!(i, vec![0, 1, 2, 3]);
    assert_eq!(c, vec![1, 1, 1, 1]);
}
