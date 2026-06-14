//! ADVERSARIAL RE-AUDIT (#1545 sub #1535, commit 87eb6597f) of the two GPU ops:
//!   1. `roll` f64 (`ferrotorch_gpu::gpu_roll_f64` via `ferrotorch_core::roll`)
//!   2. The original `unique_consecutive` f32/f64 GPU path (on-device run-flag
//!      -> prefix-sum -> scatter compaction,
//!      `ferrotorch_gpu::gpu_unique_consecutive_f{32,64}`); f16/bf16 coverage
//!      is added separately in `test_gpu_unique_consecutive_half.rs`.
//!
//! The generator's own test file
//! (`divergence_roll_f64_unique_consecutive_gpu.rs`) covers the basic run
//! pattern, no-dup, all-same, and 2-D-flatten cases. This re-audit hunts the
//! edges that file did NOT pin, exactly as the audit brief enumerates:
//!   - run boundaries / the trailing-equal-as-separate-run case,
//!   - empty / single-element edge lengths,
//!   - alternating runs,
//!   - return_counts / return_inverse correctness on every case,
//!   - consecutive-NaN runs (the run-flag `setp.ne` NaN predicate),
//!   - a 2050-element input whose runs cross the cumsum scan's 256/1024
//!     block/tile boundaries (the "most likely real bug" per the brief),
//!   - roll with k > len, k == 0, and large-negative k (wrap modulo).
//!
//! R-CHAR-3: every expected value below is the LIVE output of
//! `torch.unique_consecutive(..., return_inverse=True, return_counts=True)` /
//! `torch.roll` on torch 2.11.0+cu130 (RTX 3090), recorded as a named torch
//! reference — NOT copied from the ferrotorch GPU side. The CPU path (also
//! `torch`-parity) is asserted equal on identical data where applicable.
//!
//! DIVERGENCE FOUND (#1656, release blocker): the consecutive-NaN tests below
//! FAIL on hardware. The GPU run-flag uses `setp.ne.f32/f64` and a comment
//! claiming "NaN != anything -> true, its own run", but on the RTX 3090 (sm_52
//! PTX via ptxas) `setp.ne` is the ORDERED not-equal — it returns FALSE for NaN
//! operands. So `[nan, nan, 1.0]` flags only idx 0 as a run-start (idx1:
//! nan!=nan ordered-false; idx2: 1.0!=nan ordered-false) -> incl=[1,1,1],
//! out_len=1, collapsing the second NaN AND the following finite 1.0 into the
//! first run. torch and the ferrotorch CPU PartialEq path (`data[i]==data[i-1]`,
//! false for NaN) make EACH NaN its own run. These two tests are left
//! UN-`#[ignore]`d: the failure IS the block.

#![cfg(feature = "cuda")]

use ferrotorch_core::ops::search::unique_consecutive;
use ferrotorch_core::{Device, Tensor, TensorStorage, roll};
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

// ===========================================================================
// unique_consecutive — consecutive-NaN run semantics (run-flag setp.ne)
// ===========================================================================

/// Divergence: ferrotorch's `gpu_unique_consecutive_f32` diverges from
/// `pytorch aten/src/ATen/native/cuda/Unique.cu` (`unique_consecutive_cuda`)
/// AND from the ferrotorch CPU path
/// (`ferrotorch-core/src/ops/search.rs:256` `data[i] == data[i-1]`) for the
/// consecutive-NaN input `[nan, nan, 1.0]`.
///
/// Upstream/CPU: each NaN is its own run (NaN != NaN), so the output is
/// values `[nan, nan, 1.0]`, inverse `[0, 1, 2]`, counts `[1, 1, 1]` (3 runs).
/// ferrotorch GPU returns 1 output: the run-flag kernel's
/// `setp.ne.f32 %p_ne, %cur, %prev` (`ferrotorch-gpu/src/search.rs:1540`,
/// re-derived in COMPACT at :1661) is ORDERED not-equal on the RTX 3090 and
/// returns FALSE for NaN operands, so idx1 (nan vs nan) and idx2 (1.0 vs nan)
/// are NOT flagged as run-starts -> incl=[1,1,1], out_len=1.
///
/// Tracking: #1656 (release blocker — left un-ignored).
#[test]
fn unique_consecutive_f32_consecutive_nan_each_own_run() {
    ensure_cuda();
    let x: Vec<f32> = vec![f32::NAN, f32::NAN, 1.0];
    let xg = cuda_f32(&x);
    let (vals, inverse, counts) = unique_consecutive(&xg).expect("gpu unique nan");
    assert!(vals.is_cuda(), "unique values must stay on device");
    let got = read_back_f32(&vals);
    // torch: 3 outputs — two NaN run-starts then 1.0.
    assert_eq!(
        got.len(),
        3,
        "torch keeps each NaN as its own run -> 3 outputs"
    );
    assert!(got[0].is_nan(), "out[0] is the first NaN run-start");
    assert!(got[1].is_nan(), "out[1] is the second NaN run-start");
    assert_eq!(got[2], 1.0);
    assert_eq!(
        inverse,
        vec![0, 1, 2],
        "inverse maps each elem to its own run"
    );
    assert_eq!(counts, vec![1, 1, 1], "every run length 1");
}

/// Divergence: ferrotorch's `gpu_unique_consecutive_f64` diverges from torch /
/// the CPU path for `[1.0, nan, nan, 1.0]`.
///
/// Upstream/CPU: 4 separate runs — values `[1, nan, nan, 1]`,
/// inverse `[0,1,2,3]`, counts `[1,1,1,1]` (the two middle NaNs do not collapse
/// and the trailing 1 is its own run). ferrotorch GPU returns 1 output for the
/// same `setp.ne.f64`-ordered reason (`ferrotorch-gpu/src/search.rs:1604`,
/// re-derived in COMPACT at :1723).
///
/// Tracking: #1656 (release blocker — left un-ignored).
#[test]
fn unique_consecutive_f64_nan_sandwich_no_collapse() {
    ensure_cuda();
    let x: Vec<f64> = vec![1.0, f64::NAN, f64::NAN, 1.0];
    let xg = cuda_f64(&x);
    let (vals, inverse, counts) = unique_consecutive(&xg).expect("gpu unique nan f64");
    assert!(vals.is_cuda());
    let got = read_back_f64(&vals);
    assert_eq!(got.len(), 4, "torch: 1, nan, nan, 1 -> 4 separate runs");
    assert_eq!(got[0], 1.0);
    assert!(got[1].is_nan());
    assert!(got[2].is_nan());
    assert_eq!(got[3], 1.0);
    assert_eq!(inverse, vec![0, 1, 2, 3]);
    assert_eq!(counts, vec![1, 1, 1, 1]);
}

// ===========================================================================
// unique_consecutive — edge lengths (PASS — regression guards)
// ===========================================================================

/// torch.unique_consecutive(tensor([], dtype=f32)) -> empty values, empty
/// inverse, empty counts. Verified live. PASSES (regression guard).
#[test]
fn unique_consecutive_f32_empty_input() {
    ensure_cuda();
    let xg = cuda_f32(&[]);
    let (vals, inverse, counts) = unique_consecutive(&xg).expect("gpu unique empty");
    assert!(vals.is_cuda());
    assert_eq!(read_back_f32(&vals), Vec::<f32>::new());
    assert_eq!(inverse, Vec::<usize>::new());
    assert_eq!(counts, Vec::<usize>::new());
}

/// torch.unique_consecutive([5.]) -> values [5], inverse [0], counts [1].
/// PASSES (regression guard).
#[test]
fn unique_consecutive_f32_single_element() {
    ensure_cuda();
    let xg = cuda_f32(&[5.0]);
    let (vals, inverse, counts) = unique_consecutive(&xg).expect("gpu unique single");
    assert!(vals.is_cuda());
    assert_eq!(read_back_f32(&vals), vec![5.0]);
    assert_eq!(inverse, vec![0]);
    assert_eq!(counts, vec![1]);
}

/// torch.unique_consecutive([1,2,1,2]) -> values [1,2,1,2], inverse [0,1,2,3],
/// counts [1,1,1,1]. Alternating values are all distinct runs (NOT global
/// unique). Verified live. PASSES (regression guard).
#[test]
fn unique_consecutive_f64_alternating_is_identity() {
    ensure_cuda();
    let x: Vec<f64> = vec![1.0, 2.0, 1.0, 2.0];
    let xg = cuda_f64(&x);
    let (vals, inverse, counts) = unique_consecutive(&xg).expect("gpu unique alt");
    assert!(vals.is_cuda());
    assert_eq!(read_back_f64(&vals), vec![1.0, 2.0, 1.0, 2.0]);
    assert_eq!(inverse, vec![0, 1, 2, 3]);
    assert_eq!(counts, vec![1, 1, 1, 1]);
}

// ===========================================================================
// unique_consecutive — block-boundary-crossing scan (PASS — the scan is clean)
// ===========================================================================

/// 2050-element input whose runs (lengths 1,2,3 cycling, so each value changes)
/// cross the cumsum scan's 256/1024 block/tile boundaries. torch gives
/// out_len=1026, values [0,1,2,...,1025] (every run a distinct sequential
/// float), inverse[last]=1025, sum(counts)=2050. Verified live on cuda.
/// This is where a multi-block prefix-sum carry drop OR f32 accumulation drift
/// would show up; it does NOT — the flat-axis `gpu_cumsum` is a single serial
/// scan and the scan total (1026) is exact in f32. PASSES (regression guard).
#[test]
fn unique_consecutive_f32_block_boundary_scan() {
    ensure_cuda();
    // Reconstruct the exact data the torch oracle was run on.
    let mut data: Vec<f32> = Vec::new();
    let mut v: i64 = 0;
    while data.len() < 2050 {
        let rep = (v % 3) + 1; // run lengths 1,2,3 cycling -> value changes each run
        for _ in 0..rep {
            data.push(v as f32);
        }
        v += 1;
    }
    data.truncate(2050);

    let xg = cuda_f32(&data);
    let (vals, inverse, counts) = unique_consecutive(&xg).expect("gpu unique block-boundary");
    assert!(vals.is_cuda());
    let got = read_back_f32(&vals);

    // torch oracle: out_len == 1026, values are exactly 0..=1025.
    assert_eq!(
        got.len(),
        1026,
        "block-boundary scan produced wrong run count"
    );
    let expected: Vec<f32> = (0..1026).map(|i| i as f32).collect();
    assert_eq!(
        got, expected,
        "compacted values diverge across the scan boundary"
    );

    // counts must sum back to the input length (no run dropped/double-counted).
    assert_eq!(counts.iter().sum::<usize>(), 2050);
    // last input element maps to the last output run.
    assert_eq!(*inverse.last().unwrap(), 1025);
    assert_eq!(inverse.len(), 2050);

    // GPU == CPU on identical data (CPU is also torch-parity).
    let xc = Tensor::from_storage(TensorStorage::cpu(data), vec![2050], false).unwrap();
    let (vc, ic, cc) = unique_consecutive(&xc).unwrap();
    assert_eq!(got, read_back_f32(&vc));
    assert_eq!(inverse, ic);
    assert_eq!(counts, cc);
}

/// Same block-boundary stress in f64 (8-byte value path, f32 scan). torch
/// gives the identical run structure. Verified live. PASSES (regression guard).
#[test]
fn unique_consecutive_f64_block_boundary_scan() {
    ensure_cuda();
    let mut data: Vec<f64> = Vec::new();
    let mut v: i64 = 0;
    while data.len() < 2050 {
        let rep = (v % 3) + 1;
        for _ in 0..rep {
            data.push(v as f64);
        }
        v += 1;
    }
    data.truncate(2050);

    let xg = cuda_f64(&data);
    let (vals, inverse, counts) = unique_consecutive(&xg).expect("gpu unique block-boundary f64");
    assert!(vals.is_cuda());
    let got = read_back_f64(&vals);

    assert_eq!(got.len(), 1026);
    let expected: Vec<f64> = (0..1026).map(|i| i as f64).collect();
    assert_eq!(got, expected);
    assert_eq!(counts.iter().sum::<usize>(), 2050);
    assert_eq!(*inverse.last().unwrap(), 1025);
}

// ===========================================================================
// roll f64 — wrap edges (PASS — k > len, k == 0, large-negative k)
// ===========================================================================

/// torch.roll(arange(5).double(), k) for k in {0, 5, 7, -7}. The core wraps
/// shift modulo len BEFORE the GPU kernel
/// (`ferrotorch-core/src/ops/tensor_ops.rs` `((shifts % dim_size) + dim_size)
/// % dim_size`), so the f64 kernel only ever sees a normalized shift. Pin every
/// wrap edge against torch. Verified live. PASSES (regression guard).
#[test]
fn roll_f64_wrap_edges_match_torch() {
    ensure_cuda();
    let base: Vec<f64> = (0..5).map(|i| i as f64).collect();
    let xg = cuda_f64(&base);

    // k == 0 -> identity.
    let y0 = roll(&xg, 0, 0).expect("roll 0");
    assert_eq!(read_back_f64(&y0), vec![0.0, 1.0, 2.0, 3.0, 4.0]);

    // k == len (5) -> identity (full wrap). torch.roll(x,5) == x.
    let y5 = roll(&xg, 5, 0).expect("roll 5");
    assert!(y5.is_cuda());
    assert_eq!(read_back_f64(&y5), vec![0.0, 1.0, 2.0, 3.0, 4.0]);

    // k == 7 -> same as k == 2. torch.roll(x,7) -> [3,4,0,1,2].
    let y7 = roll(&xg, 7, 0).expect("roll 7");
    assert!(y7.is_cuda());
    assert_eq!(read_back_f64(&y7), vec![3.0, 4.0, 0.0, 1.0, 2.0]);

    // k == -7 -> same as k == -2 == +3. torch.roll(x,-7) -> [2,3,4,0,1].
    let yn7 = roll(&xg, -7, 0).expect("roll -7");
    assert!(yn7.is_cuda());
    assert_eq!(read_back_f64(&yn7), vec![2.0, 3.0, 4.0, 0.0, 1.0]);

    // GPU == CPU for the k==7 case.
    let xc = Tensor::from_storage(TensorStorage::cpu(base), vec![5], false).unwrap();
    let yc = roll(&xc, 7, 0).unwrap();
    assert_eq!(read_back_f64(&y7), yc.data_vec().unwrap());
}
