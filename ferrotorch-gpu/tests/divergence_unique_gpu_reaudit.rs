//! Adversarial re-audit of GPU `unique` (commit 2cd03a52b, #1664 / #1545).
//!
//! `ferrotorch_core::ops::search::unique` lowers
//! `torch.unique(sorted=True, return_inverse=True, return_counts=True)` for
//! CUDA-resident f32/f64/f16/bf16 via the on-device bitonic sort-by-key (pad to next
//! pow2 with +INF/idx-MAX sentinels) -> run-flag -> gpu_cumsum -> compaction
//! pipeline (`gpu_unique_f{32,64,16}` / `gpu_unique_bf16` in
//! `ferrotorch-gpu/src/search.rs`).
//!
//! The shipped suite `divergence_unique_gpu.rs` covers small finite cases, a
//! len-7 non-pow2, len-100/1000 invariant checks, NaN/Inf, f64, GPU==CPU, and
//! f16/bf16 support. This file pins the cases that suite leaves open and that the
//! bitonic-padding network is most likely to break:
//!   - EXACT powers of two (256, 1024): no padding at all -> the +INF sentinel
//!     branch never runs; a stage-boundary bug shows here.
//!   - One MORE than a pow2 (257, 1025): npad jumps to the next power (512/2048)
//!     so the array is MOSTLY pads -> sentinel ranking is stressed.
//!   - One LESS than a pow2 (255, 1023): a single pad.
//!   - LARGE sizes (5000, 10000): cross many bitonic (k,j) stages.
//!   - Signed zero: -0.0 and +0.0 collapse to ONE entry.
//!   - storage_offset: unique on a narrowed-offset CUDA view must operate on the
//!     LOGICAL values (the consumer `.contiguous()`-normalises first).
//!
//! Reproducible-in-test input: `x[i] = ((i*7 + 3) % m) as f32`. Because
//! `gcd(7, m) == 1` and the size spans all residues, the unique values are
//! exactly `0..m-1` sorted ascending, so `inverse[i] == (i*7+3) % m` (verified
//! `True` against `torch.unique(...).indices` live on torch 2.11.0+cu130, RTX
//! 3090). The per-value `counts` are recorded inline from that same live run
//! (R-CHAR-3: NOT copied from the ferrotorch side). The inverse is reconstructed
//! from the input formula, not from the implementation under test.

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

/// Build `x[i] = ((i*7+3) % m) as f32`.
fn formula(size: usize, m: usize) -> Vec<f32> {
    (0..size).map(|i| ((i * 7 + 3) % m) as f32).collect()
}

/// Drive `unique` over the formula input of length `size` (modulus `m`) and
/// assert values == sorted `0..m`, inverse == `(i*7+3)%m`, and counts ==
/// `expected_counts` (the live-torch record).
fn check_formula(size: usize, m: usize, expected_counts: &[usize]) {
    let data = formula(size, m);
    let xg = cuda_f32(&data);
    let (vals, inverse, counts) = unique(&xg).expect("gpu unique");
    assert!(vals.is_cuda(), "values stay on device (size={size})");
    let v = read_back_f32(&vals);

    // values: sorted ascending 0..m, no +INF sentinel leaked.
    let want_vals: Vec<f32> = (0..m).map(|k| k as f32).collect();
    assert_eq!(v, want_vals, "values vs torch (size={size}, m={m})");

    // inverse: each input maps to its own value's slot (== the value itself).
    let want_inv: Vec<usize> = (0..size).map(|i| (i * 7 + 3) % m).collect();
    assert_eq!(inverse, want_inv, "inverse vs torch (size={size}, m={m})");

    // counts: live-torch record; also must sum to size.
    assert_eq!(
        counts, expected_counts,
        "counts vs torch (size={size}, m={m})"
    );
    assert_eq!(counts.iter().sum::<usize>(), size, "counts sum to n");
}

// --- EXACT powers of two (no padding) ---

#[test]
fn unique_f32_exact_pow2_256() {
    ensure_cuda();
    check_formula(256, 9, &[28, 29, 28, 29, 28, 28, 29, 28, 29]);
}

#[test]
fn unique_f32_exact_pow2_1024() {
    ensure_cuda();
    check_formula(
        1024,
        13,
        &[79, 79, 78, 79, 79, 79, 79, 79, 78, 78, 79, 79, 79],
    );
}

// --- one MORE than a pow2 (mostly-pad arrays) ---

#[test]
fn unique_f32_pow2_plus_one_257() {
    ensure_cuda();
    // 257 -> npad 512 (255 pads). torch counts for m=11.
    check_formula(257, 11, &[23, 23, 24, 24, 23, 23, 24, 23, 23, 23, 24]);
}

#[test]
fn unique_f32_pow2_plus_one_1025() {
    ensure_cuda();
    // 1025 -> npad 2048 (1023 pads). torch counts for m=17.
    check_formula(
        1025,
        17,
        &[
            61, 60, 60, 61, 60, 60, 60, 61, 60, 60, 61, 60, 60, 60, 61, 60, 60,
        ],
    );
}

// --- one LESS than a pow2 (single pad) ---

#[test]
fn unique_f32_pow2_minus_one_255() {
    ensure_cuda();
    check_formula(
        255,
        13,
        &[20, 19, 19, 20, 20, 20, 20, 19, 19, 19, 20, 20, 20],
    );
}

// --- LARGE sizes crossing many bitonic stages ---

#[test]
fn unique_f32_large_5000() {
    ensure_cuda();
    check_formula(
        5000,
        19,
        &[
            263, 263, 263, 264, 263, 263, 263, 263, 263, 263, 264, 263, 263, 263, 263, 263, 263,
            264, 263,
        ],
    );
}

#[test]
fn unique_f32_large_10000() {
    ensure_cuda();
    check_formula(
        10000,
        23,
        &[
            435, 435, 435, 435, 435, 434, 435, 435, 435, 435, 435, 435, 434, 435, 434, 435, 435,
            435, 435, 434, 435, 434, 435,
        ],
    );
}

// --- inverse-permutation correctness with scattered duplicates ---

#[test]
fn unique_f32_scattered_inverse() {
    ensure_cuda();
    // torch.unique([5,1,3,1,5,3,1]) -> vals [1,3,5] inv [2,0,1,0,2,1,0]
    //   counts [3,2,2]. A wrong sort-permutation scatter corrupts inverse here.
    let xg = cuda_f32(&[5.0, 1.0, 3.0, 1.0, 5.0, 3.0, 1.0]);
    let (vals, inverse, counts) = unique(&xg).unwrap();
    assert_eq!(read_back_f32(&vals), vec![1.0, 3.0, 5.0]);
    assert_eq!(
        inverse,
        vec![2, 0, 1, 0, 2, 1, 0],
        "scatter inverse vs torch"
    );
    assert_eq!(counts, vec![3, 2, 2]);
}

// --- signed zero: -0.0 and +0.0 collapse to ONE entry ---

#[test]
fn unique_f32_signed_zero_collapses() {
    ensure_cuda();
    // torch.unique([-0.0,0.0,-0.0,0.0]) -> vals [+0.0] inv [0,0,0,0] counts [4].
    // The sign bit matters: sorted unique keeps the last original finite-equal
    // representative, so this input's compacted zero is the trailing +0.0.
    let xg = cuda_f32(&[-0.0, 0.0, -0.0, 0.0]);
    let (vals, inverse, counts) = unique(&xg).unwrap();
    let v = read_back_f32(&vals);
    assert_eq!(v.len(), 1, "+/-0 collapse to one entry: {v:?}");
    assert_eq!(v[0], 0.0);
    assert_eq!(v[0].to_bits(), 0x0000_0000);
    assert_eq!(inverse, vec![0, 0, 0, 0]);
    assert_eq!(counts, vec![4]);
}

#[test]
fn unique_f32_signed_zero_keeps_last_original_representative() {
    ensure_cuda();
    let (vals_a, inverse_a, counts_a) = unique(&cuda_f32(&[0.0, -0.0])).unwrap();
    let a = read_back_f32(&vals_a);
    assert_eq!(a.len(), 1);
    assert_eq!(a[0].to_bits(), 0x8000_0000, "[+0,-0] keeps trailing -0");
    assert_eq!(inverse_a, vec![0, 0]);
    assert_eq!(counts_a, vec![2]);

    let (vals_b, inverse_b, counts_b) = unique(&cuda_f32(&[-0.0, 0.0])).unwrap();
    let b = read_back_f32(&vals_b);
    assert_eq!(b.len(), 1);
    assert_eq!(b[0].to_bits(), 0x0000_0000, "[-0,+0] keeps trailing +0");
    assert_eq!(inverse_b, vec![0, 0]);
    assert_eq!(counts_b, vec![2]);
}

#[test]
fn unique_f64_signed_zero_keeps_last_original_representative() {
    ensure_cuda();
    let (vals_a, inverse_a, counts_a) = unique(&cuda_f64(&[0.0, -0.0])).unwrap();
    let a = read_back_f64(&vals_a);
    assert_eq!(a.len(), 1);
    assert_eq!(a[0].to_bits(), 0x8000_0000_0000_0000);
    assert_eq!(inverse_a, vec![0, 0]);
    assert_eq!(counts_a, vec![2]);

    let (vals_b, inverse_b, counts_b) = unique(&cuda_f64(&[-0.0, 0.0])).unwrap();
    let b = read_back_f64(&vals_b);
    assert_eq!(b.len(), 1);
    assert_eq!(b[0].to_bits(), 0x0000_0000_0000_0000);
    assert_eq!(inverse_b, vec![0, 0]);
    assert_eq!(counts_b, vec![2]);
}

// --- storage_offset: unique on a narrowed CUDA view uses LOGICAL values ---

#[test]
fn unique_f32_storage_offset_narrow() {
    ensure_cuda();
    // big = [100,200,5,1,3,1,5,300]; big.narrow(0,2,5) = [5,1,3,1,5].
    // torch.unique(narrow) -> vals [1,3,5] inv [2,0,1,0,2] counts [2,1,2].
    // If the GPU path read the base buffer at offset 0 it would see 100/200/...
    let base = cuda_f32(&[100.0, 200.0, 5.0, 1.0, 3.0, 1.0, 5.0, 300.0]);
    let narrowed = base.narrow(0, 2, 5).expect("narrow");
    let (vals, inverse, counts) = unique(&narrowed).expect("unique on narrow view");
    let v = read_back_f32(&vals);
    assert_eq!(v, vec![1.0, 3.0, 5.0], "narrow view logical values: {v:?}");
    assert_eq!(inverse, vec![2, 0, 1, 0, 2], "narrow inverse vs torch");
    assert_eq!(counts, vec![2, 1, 2]);
}

// --- f64 coverage of the pow2-boundary + large cases ---

#[test]
fn unique_f64_exact_pow2_256() {
    ensure_cuda();
    let data: Vec<f64> = (0..256).map(|i| ((i * 7 + 3) % 9) as f64).collect();
    let xg = cuda_f64(&data);
    let (vals, inverse, counts) = unique(&xg).unwrap();
    assert_eq!(
        read_back_f64(&vals),
        (0..9).map(|k| k as f64).collect::<Vec<_>>()
    );
    let want_inv: Vec<usize> = (0..256).map(|i| (i * 7 + 3) % 9).collect();
    assert_eq!(inverse, want_inv, "f64 pow2 inverse vs torch");
    assert_eq!(counts, vec![28, 29, 28, 29, 28, 28, 29, 28, 29]);
}

#[test]
fn unique_f64_large_5000() {
    ensure_cuda();
    let data: Vec<f64> = (0..5000).map(|i| ((i * 7 + 3) % 19) as f64).collect();
    let xg = cuda_f64(&data);
    let (vals, inverse, counts) = unique(&xg).unwrap();
    assert_eq!(
        read_back_f64(&vals),
        (0..19).map(|k| k as f64).collect::<Vec<_>>()
    );
    let want_inv: Vec<usize> = (0..5000).map(|i| (i * 7 + 3) % 19).collect();
    assert_eq!(inverse, want_inv, "f64 large inverse vs torch");
    assert_eq!(counts.iter().sum::<usize>(), 5000);
}
