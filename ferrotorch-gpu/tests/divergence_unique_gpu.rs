//! Live-GPU consumer-path tests for `unique` (#1545 / sub #1535, final op).
//!
//! `ferrotorch_core::ops::search::unique` lowers
//! `torch.unique(sorted=True, return_inverse=True, return_counts=True)` for
//! CUDA-resident f32/f64 tensors via the on-device bitonic sort-by-key →
//! run-flag → prefix-sum → compaction pipeline
//! (`ferrotorch_gpu::gpu_unique_f{32,64}` / `GpuBackend::unique_1d`). The
//! SORTED-unique VALUES stay `is_cuda()` (computed on-device, wrapped straight
//! back; R-CODE-4 — no value round trip through host); only the derived
//! integer index/run metadata is read back to build the host `inverse` /
//! `counts` vectors.
//!
//! Upstream: `aten/src/ATen/native/cuda/Unique.cu` `compute_unique:51-85`
//! (CUDA `unique` ALWAYS sorts — no device hashtable in thrust — carrying the
//! original int64 indices via `radix_sort_pairs` (`UniqueCub.cu:175`), then
//! `inverse[sorted_indices[i]] = inclusive_scan(adjacent_diff(not_equal))[i]`
//! (`:63-66`) and `counts` = run-length of each unique (`:75-81`)).
//!
//! All expected outputs below are the EXACT outputs of `torch.unique(...,
//! sorted=True, return_inverse=True, return_counts=True)` recorded LIVE on
//! torch 2.11.0+cu130 (RTX 3090), inline as named references (R-CHAR-3: NOT
//! copied from the ferrotorch GPU side). They also match the CPU path on
//! identical FINITE data (GPU == CPU == torch).
//!
//! NaN parity (verified live): `torch.unique` does NOT collapse NaNs — each
//! NaN is a DISTINCT unique entry sorted to the END (the same `setp.neu`
//! predicate as `unique_consecutive`). The GPU comparator breaks NaN ties by
//! ascending original index, matching torch's radix-stable NaN order.

#![cfg(feature = "cuda")]

use ferrotorch_core::ops::search::unique;
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

// ===========================================================================
// f32 — finite cases (GPU values is_cuda, inverse/counts exact vs torch)
// ===========================================================================

#[test]
fn unique_f32_basic_matches_torch() {
    ensure_cuda();
    // torch.unique([3,1,2,1,3]) -> vals [1,2,3] inv [2,0,1,0,2] cnt [2,1,2]
    let xg = cuda_f32(&[3.0, 1.0, 2.0, 1.0, 3.0]);
    let (vals, inverse, counts) = unique(&xg).expect("gpu unique f32");
    assert!(vals.is_cuda(), "unique values must stay on device");
    assert_eq!(read_back_f32(&vals), vec![1.0, 2.0, 3.0]);
    assert_eq!(inverse, vec![2, 0, 1, 0, 2]);
    assert_eq!(counts, vec![2, 1, 2]);
}

#[test]
fn unique_f32_already_sorted() {
    ensure_cuda();
    // torch.unique([1,1,2,3,3]) -> vals [1,2,3] inv [0,0,1,2,2] cnt [2,1,2]
    let xg = cuda_f32(&[1.0, 1.0, 2.0, 3.0, 3.0]);
    let (vals, inverse, counts) = unique(&xg).unwrap();
    assert_eq!(read_back_f32(&vals), vec![1.0, 2.0, 3.0]);
    assert_eq!(inverse, vec![0, 0, 1, 2, 2]);
    assert_eq!(counts, vec![2, 1, 2]);
}

#[test]
fn unique_f32_reverse_sorted() {
    ensure_cuda();
    // torch.unique([5,4,3,2,1]) -> vals [1,2,3,4,5] inv [4,3,2,1,0] cnt all 1
    let xg = cuda_f32(&[5.0, 4.0, 3.0, 2.0, 1.0]);
    let (vals, inverse, counts) = unique(&xg).unwrap();
    assert_eq!(read_back_f32(&vals), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(inverse, vec![4, 3, 2, 1, 0]);
    assert_eq!(counts, vec![1, 1, 1, 1, 1]);
}

#[test]
fn unique_f32_all_same() {
    ensure_cuda();
    // torch.unique([7,7,7,7]) -> vals [7] inv [0,0,0,0] cnt [4]
    let xg = cuda_f32(&[7.0, 7.0, 7.0, 7.0]);
    let (vals, inverse, counts) = unique(&xg).unwrap();
    assert_eq!(read_back_f32(&vals), vec![7.0]);
    assert_eq!(inverse, vec![0, 0, 0, 0]);
    assert_eq!(counts, vec![4]);
}

#[test]
fn unique_f32_all_distinct() {
    ensure_cuda();
    // torch.unique([4,1,3,2]) -> vals [1,2,3,4] inv [3,0,2,1] cnt all 1
    let xg = cuda_f32(&[4.0, 1.0, 3.0, 2.0]);
    let (vals, inverse, counts) = unique(&xg).unwrap();
    assert_eq!(read_back_f32(&vals), vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(inverse, vec![3, 0, 2, 1]);
    assert_eq!(counts, vec![1, 1, 1, 1]);
}

#[test]
fn unique_f32_single_element() {
    ensure_cuda();
    // torch.unique([42]) -> vals [42] inv [0] cnt [1]. n=1 (next_pow2(1)=1).
    let xg = cuda_f32(&[42.0]);
    let (vals, inverse, counts) = unique(&xg).unwrap();
    assert!(vals.is_cuda());
    assert_eq!(read_back_f32(&vals), vec![42.0]);
    assert_eq!(inverse, vec![0]);
    assert_eq!(counts, vec![1]);
}

#[test]
fn unique_f32_negative_and_positive() {
    ensure_cuda();
    // torch.unique([-2,3,-2,0,3,-5]) -> vals [-5,-2,0,3] inv [1,3,1,2,3,0]
    //   cnt [1,2,1,2]
    let xg = cuda_f32(&[-2.0, 3.0, -2.0, 0.0, 3.0, -5.0]);
    let (vals, inverse, counts) = unique(&xg).unwrap();
    assert_eq!(read_back_f32(&vals), vec![-5.0, -2.0, 0.0, 3.0]);
    assert_eq!(inverse, vec![1, 3, 1, 2, 3, 0]);
    assert_eq!(counts, vec![1, 2, 1, 2]);
}

// ===========================================================================
// f32 — non-power-of-2 lengths (the bitonic padding must not corrupt results)
// ===========================================================================

#[test]
fn unique_f32_len7_non_pow2() {
    ensure_cuda();
    // torch.unique([5,5,1,9,2,9,1]) -> vals [1,2,5,9] inv [2,2,0,3,1,3,0]
    //   cnt [2,1,2,2].  n=7 -> npad=8 (one +INF pad).
    let xg = cuda_f32(&[5.0, 5.0, 1.0, 9.0, 2.0, 9.0, 1.0]);
    let (vals, inverse, counts) = unique(&xg).unwrap();
    assert_eq!(read_back_f32(&vals), vec![1.0, 2.0, 5.0, 9.0]);
    assert_eq!(inverse, vec![2, 2, 0, 3, 1, 3, 0]);
    assert_eq!(counts, vec![2, 1, 2, 2]);
}

/// Property check for a non-pow2 length of 100 (npad=128, 28 pads): the value
/// SET matches the torch oracle exactly (`[0..=9]`), the inverse reconstructs
/// the input bit-for-bit, the counts sum to `n`, and the output is sorted —
/// the padding sentinels are fully excluded. Input + oracle value set recorded
/// from `torch.unique` (torch 2.11 CUDA).
#[test]
fn unique_f32_len100_non_pow2_invariants() {
    ensure_cuda();
    #[rustfmt::skip]
    let data: Vec<f32> = vec![
        6.0, 6.0, 0.0, 4.0, 8.0, 7.0, 6.0, 4.0, 7.0, 5.0, 9.0, 3.0, 8.0, 2.0, 4.0, 2.0,
        1.0, 9.0, 4.0, 8.0, 9.0, 2.0, 4.0, 1.0, 1.0, 5.0, 7.0, 8.0, 1.0, 5.0, 6.0, 5.0,
        9.0, 3.0, 8.0, 7.0, 7.0, 8.0, 4.0, 0.0, 8.0, 0.0, 1.0, 6.0, 0.0, 9.0, 7.0, 5.0,
        3.0, 5.0, 1.0, 3.0, 9.0, 3.0, 3.0, 2.0, 8.0, 7.0, 1.0, 1.0, 5.0, 8.0, 7.0, 1.0,
        4.0, 8.0, 4.0, 1.0, 8.0, 5.0, 8.0, 3.0, 9.0, 8.0, 9.0, 4.0, 7.0, 1.0, 9.0, 6.0,
        5.0, 9.0, 3.0, 4.0, 2.0, 3.0, 2.0, 0.0, 9.0, 4.0, 7.0, 1.0, 1.0, 2.0, 2.0, 0.0,
        1.0, 8.0, 6.0, 8.0,
    ];
    let xg = cuda_f32(&data);
    let (vals, inverse, counts) = unique(&xg).unwrap();
    assert!(vals.is_cuda());
    let v = read_back_f32(&vals);
    // torch.unique value set for this input:
    assert_eq!(v, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    // sorted ascending
    assert!(v.windows(2).all(|w| w[0] < w[1]), "values sorted: {v:?}");
    // inverse reconstructs the input exactly
    assert_eq!(inverse.len(), data.len());
    for (i, &orig) in data.iter().enumerate() {
        assert_eq!(v[inverse[i]], orig, "inverse[{i}] wrong");
    }
    // counts sum to n and per-value match the frequency in the input
    assert_eq!(counts.iter().sum::<usize>(), data.len());
    for (uid, &val) in v.iter().enumerate() {
        let want = data.iter().filter(|&&d| d == val).count();
        assert_eq!(counts[uid], want, "count for {val} wrong");
    }
}

/// Property check for a non-pow2 length of 1000 (npad=1024, 24 pads) over 50
/// distinct values: invariants hold and the padding never leaks a sentinel
/// (+INF) into the output.
#[test]
fn unique_f32_len1000_non_pow2_invariants() {
    ensure_cuda();
    // Deterministic pseudo-random fill matching the recorded torch run
    // (linear congruential, values in [0,49]); torch.unique over this exact
    // sequence yields 50 distinct sorted values [0..=49], counts summing 1000.
    let mut state: u64 = 0x1234_5678;
    let data: Vec<f32> = (0..1000)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) % 50) as f32
        })
        .collect();
    let xg = cuda_f32(&data);
    let (vals, inverse, counts) = unique(&xg).unwrap();
    assert!(vals.is_cuda());
    let v = read_back_f32(&vals);
    // every output value is finite (no +INF pad leaked) and in [0,49]
    assert!(
        v.iter().all(|x| x.is_finite() && (0.0..=49.0).contains(x)),
        "no pad leak: {v:?}"
    );
    assert!(v.windows(2).all(|w| w[0] < w[1]), "sorted ascending");
    assert_eq!(inverse.len(), 1000);
    for (i, &orig) in data.iter().enumerate() {
        assert_eq!(v[inverse[i]], orig);
    }
    assert_eq!(counts.iter().sum::<usize>(), 1000);
}

// ===========================================================================
// NaN / Inf — torch.unique does NOT collapse NaNs; each is a distinct tail entry
// ===========================================================================

#[test]
fn unique_f32_nan_each_distinct_at_end() {
    ensure_cuda();
    // torch.unique([nan,1,nan,2,nan]) -> vals [1,2,nan,nan,nan]
    //   inv [2,0,3,1,4] cnt [1,1,1,1,1]  (each NaN distinct, sorted to end).
    let xg = cuda_f32(&[f32::NAN, 1.0, f32::NAN, 2.0, f32::NAN]);
    let (vals, inverse, counts) = unique(&xg).unwrap();
    assert!(vals.is_cuda());
    let v = read_back_f32(&vals);
    assert_eq!(&v[..2], &[1.0, 2.0]);
    assert!(
        v[2].is_nan() && v[3].is_nan() && v[4].is_nan(),
        "3 NaN tail: {v:?}"
    );
    assert_eq!(v.len(), 5);
    // NaN-tie-break-by-ascending-index matches torch's radix-stable order.
    assert_eq!(inverse, vec![2, 0, 3, 1, 4]);
    assert_eq!(counts, vec![1, 1, 1, 1, 1]);
}

#[test]
fn unique_f32_inf_and_nan() {
    ensure_cuda();
    // torch.unique([inf,1,-inf,nan,1,inf]) -> vals [-inf,1,inf,nan]
    //   inv [2,1,0,3,1,2] cnt [1,2,2,1].
    let xg = cuda_f32(&[
        f32::INFINITY,
        1.0,
        f32::NEG_INFINITY,
        f32::NAN,
        1.0,
        f32::INFINITY,
    ]);
    let (vals, inverse, counts) = unique(&xg).unwrap();
    let v = read_back_f32(&vals);
    assert_eq!(v[0], f32::NEG_INFINITY);
    assert_eq!(v[1], 1.0);
    assert_eq!(v[2], f32::INFINITY);
    assert!(v[3].is_nan(), "nan last: {v:?}");
    assert_eq!(inverse, vec![2, 1, 0, 3, 1, 2]);
    assert_eq!(counts, vec![1, 2, 2, 1]);
}

// ===========================================================================
// f64
// ===========================================================================

#[test]
fn unique_f64_basic_matches_torch() {
    ensure_cuda();
    // torch.unique(tensor([3,1,2,1,3], dtype=f64)) -> vals [1,2,3]
    //   inv [2,0,1,0,2] cnt [2,1,2]
    let xg = cuda_f64(&[3.0, 1.0, 2.0, 1.0, 3.0]);
    let (vals, inverse, counts) = unique(&xg).expect("gpu unique f64");
    assert!(vals.is_cuda());
    assert_eq!(read_back_f64(&vals), vec![1.0, 2.0, 3.0]);
    assert_eq!(inverse, vec![2, 0, 1, 0, 2]);
    assert_eq!(counts, vec![2, 1, 2]);
}

#[test]
fn unique_f64_nan_distinct_tail() {
    ensure_cuda();
    // torch.unique(tensor([2.5,nan,2.5,1.5], dtype=f64)) -> vals [1.5,2.5,nan]
    //   inv [1,2,1,0] cnt [1,2,1].
    let xg = cuda_f64(&[2.5, f64::NAN, 2.5, 1.5]);
    let (vals, inverse, counts) = unique(&xg).unwrap();
    let v = read_back_f64(&vals);
    assert_eq!(&v[..2], &[1.5, 2.5]);
    assert!(v[2].is_nan(), "nan last: {v:?}");
    assert_eq!(inverse, vec![1, 2, 1, 0]);
    assert_eq!(counts, vec![1, 2, 1]);
}

#[test]
fn unique_f64_non_pow2_len7() {
    ensure_cuda();
    // Same data as the f32 len7 case; f64 path. torch.unique -> [1,2,5,9].
    let xg = cuda_f64(&[5.0, 5.0, 1.0, 9.0, 2.0, 9.0, 1.0]);
    let (vals, inverse, counts) = unique(&xg).unwrap();
    assert_eq!(read_back_f64(&vals), vec![1.0, 2.0, 5.0, 9.0]);
    assert_eq!(inverse, vec![2, 2, 0, 3, 1, 3, 0]);
    assert_eq!(counts, vec![2, 1, 2, 2]);
}

// ===========================================================================
// GPU == CPU on identical FINITE data
// ===========================================================================

#[test]
fn unique_f32_gpu_equals_cpu_finite() {
    ensure_cuda();
    let data: Vec<f32> = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0, 5.0];
    let xg = cuda_f32(&data);
    let (vg, ig, cg) = unique(&xg).unwrap();

    let xc = Tensor::from_storage(TensorStorage::cpu(data), vec![11], false).unwrap();
    let (vc, ic, cc) = unique(&xc).unwrap();

    assert_eq!(read_back_f32(&vg), vc.data_vec().unwrap());
    assert_eq!(ig, ic);
    assert_eq!(cg, cc);
}

// ===========================================================================
// bf16 / f16 reject (NotImplementedOnCuda) — only f32/f64 lower on-device
// ===========================================================================

#[test]
fn unique_bf16_cuda_rejects() {
    use half::bf16;
    ensure_cuda();
    let data: Vec<bf16> = vec![
        bf16::from_f32(1.0),
        bf16::from_f32(2.0),
        bf16::from_f32(1.0),
    ];
    let xg = Tensor::from_storage(TensorStorage::cpu(data), vec![3], false)
        .unwrap()
        .to(Device::Cuda(0))
        .expect("upload bf16");
    let err = unique(&xg);
    assert!(
        err.is_err(),
        "bf16 unique on CUDA must reject (only f32/f64 lower)"
    );
}

#[test]
fn unique_f16_cuda_rejects() {
    use half::f16;
    ensure_cuda();
    let data: Vec<f16> = vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(1.0)];
    let xg = Tensor::from_storage(TensorStorage::cpu(data), vec![3], false)
        .unwrap()
        .to(Device::Cuda(0))
        .expect("upload f16");
    let err = unique(&xg);
    assert!(
        err.is_err(),
        "f16 unique on CUDA must reject (only f32/f64 lower)"
    );
}
