//! Live-GPU consumer-path tests for the two ops shipped in #1545 / sub #1535:
//!
//!   1. `roll` f64 — the f64 sibling of the existing GPU `roll_f32` kernel
//!      (`ferrotorch_gpu::gpu_roll_f64` / `GpuBackend::roll_f64`), wired through
//!      `ferrotorch_core::roll`'s f64 CUDA branch. `roll` is pure index
//!      movement, so the GPU f64 result is bit-exact with both the CPU path and
//!      `torch.roll`.
//!   2. The original `unique_consecutive` f32/f64 path — on-device run compaction
//!      (`ferrotorch_gpu::gpu_unique_consecutive_f{32,64}` /
//!      `GpuBackend::unique_consecutive_1d`) wired through
//!      `ferrotorch_core::ops::search::unique_consecutive`'s CUDA branch. f16
//!      and bf16 coverage is added separately in
//!      `test_gpu_unique_consecutive_half.rs`.
//!
//! Each test confirms the result tensor stays `is_cuda()` (the deduplicated /
//! rolled VALUES are computed on-device and wrapped straight back; R-CODE-4 —
//! no value round trip through host) and that the GPU output matches the LIVE
//! torch oracle byte-for-byte. The torch expectations below are the exact
//! outputs of `torch.roll` / `torch.unique_consecutive` on torch 2.11
//! (RTX 3090), recorded inline as named references (R-CHAR-3: NOT copied from
//! the ferrotorch GPU side). They also match the CPU path on identical data
//! (GPU==CPU).
//!
//! Upstream contract:
//!   - roll: `aten/src/ATen/native/cuda/TensorTransformations.cu:84`
//!     (`roll_cuda_kernel`, single-axis cyclic shift).
//!   - unique_consecutive: collapses maximal runs of equal ADJACENT elements;
//!     `torch.unique_consecutive(x, return_inverse=True, return_counts=True)`.
//!     NaN starts its own run (NaN != NaN), matching the CPU PartialEq path.

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

fn cuda_f32_shaped(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .expect("cpu f32 tensor")
        .to(Device::Cuda(0))
        .expect("upload f32 to cuda")
}

fn cuda_f64_shaped(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
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
// roll f64
// ===========================================================================

#[test]
fn roll_f64_1d_positive_and_negative_match_torch() {
    ensure_cuda();
    // torch.roll(arange(8).double(), 3) -> [5,6,7,0,1,2,3,4]
    let x: Vec<f64> = (0..8).map(|i| i as f64).collect();
    let xg = cuda_f64(&x);
    let yg = roll(&xg, 3, 0).expect("gpu roll +3");
    assert!(yg.is_cuda(), "roll f64 result must stay on device");
    let torch_plus3 = vec![5.0, 6.0, 7.0, 0.0, 1.0, 2.0, 3.0, 4.0];
    assert_eq!(read_back_f64(&yg), torch_plus3);

    // torch.roll(arange(8).double(), -2) -> [2,3,4,5,6,7,0,1]
    let yg_neg = roll(&xg, -2, 0).expect("gpu roll -2");
    assert!(yg_neg.is_cuda());
    let torch_minus2 = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.0, 1.0];
    assert_eq!(read_back_f64(&yg_neg), torch_minus2);

    // GPU == CPU on identical data.
    let xc = Tensor::from_storage(TensorStorage::cpu(x), vec![8], false).unwrap();
    let yc = roll(&xc, 3, 0).unwrap();
    assert_eq!(read_back_f64(&yg), yc.data_vec().unwrap());
}

#[test]
fn roll_f64_2d_per_dim_matches_torch() {
    ensure_cuda();
    // x = arange(12).double().reshape(3,4)
    // torch.roll(x, 1, dim=0) shifts ROWS down by 1:
    //   [[8,9,10,11],[0,1,2,3],[4,5,6,7]]
    let x: Vec<f64> = (0..12).map(|i| i as f64).collect();
    let xg = cuda_f64_shaped(&x, &[3, 4]);
    let yg0 = roll(&xg, 1, 0).expect("gpu roll dim0");
    assert!(yg0.is_cuda());
    let torch_dim0 = vec![8.0, 9.0, 10.0, 11.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    assert_eq!(read_back_f64(&yg0), torch_dim0);

    // torch.roll(x, -1, dim=1) shifts COLS left by 1 within each row:
    //   [[1,2,3,0],[5,6,7,4],[9,10,11,8]]
    let yg1 = roll(&xg, -1, 1).expect("gpu roll dim1");
    assert!(yg1.is_cuda());
    let torch_dim1 = vec![1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 7.0, 4.0, 9.0, 10.0, 11.0, 8.0];
    assert_eq!(read_back_f64(&yg1), torch_dim1);

    // GPU == CPU.
    let xc = Tensor::from_storage(TensorStorage::cpu(x), vec![3, 4], false).unwrap();
    let yc = roll(&xc, 1, 0).unwrap();
    assert_eq!(read_back_f64(&yg0), yc.data_vec().unwrap());
}

// ===========================================================================
// unique_consecutive
// ===========================================================================

#[test]
fn unique_consecutive_f32_runs_match_torch() {
    ensure_cuda();
    // torch.unique_consecutive(tensor([1,1,2,3,3,3,1]),
    //   return_inverse=True, return_counts=True) ->
    //   values  = [1,2,3,1]
    //   inverse = [0,0,1,2,2,2,3]
    //   counts  = [2,1,3,1]
    let x: Vec<f32> = vec![1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 1.0];
    let xg = cuda_f32(&x);
    let (vals, inverse, counts) = unique_consecutive(&xg).expect("gpu unique_consecutive f32");
    assert!(vals.is_cuda(), "unique values must stay on device");
    assert_eq!(read_back_f32(&vals), vec![1.0, 2.0, 3.0, 1.0]);
    assert_eq!(inverse, vec![0, 0, 1, 2, 2, 2, 3]);
    assert_eq!(counts, vec![2, 1, 3, 1]);

    // GPU == CPU on identical data.
    let xc = Tensor::from_storage(TensorStorage::cpu(x), vec![7], false).unwrap();
    let (vc, ic, cc) = unique_consecutive(&xc).unwrap();
    assert_eq!(read_back_f32(&vals), vc.data_vec().unwrap());
    assert_eq!(inverse, ic);
    assert_eq!(counts, cc);
}

#[test]
fn unique_consecutive_f32_no_duplicates_is_identity() {
    ensure_cuda();
    // torch.unique_consecutive([1,2,3,4,5]) -> values [1,2,3,4,5],
    //   inverse [0,1,2,3,4], counts [1,1,1,1,1]
    let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let xg = cuda_f32(&x);
    let (vals, inverse, counts) = unique_consecutive(&xg).expect("gpu unique f32 no-dup");
    assert!(vals.is_cuda());
    assert_eq!(read_back_f32(&vals), x);
    assert_eq!(inverse, vec![0, 1, 2, 3, 4]);
    assert_eq!(counts, vec![1, 1, 1, 1, 1]);
}

#[test]
fn unique_consecutive_f32_all_same_collapses_to_one() {
    ensure_cuda();
    // torch.unique_consecutive([7,7,7,7]) -> values [7], inverse [0,0,0,0],
    //   counts [4]
    let x: Vec<f32> = vec![7.0, 7.0, 7.0, 7.0];
    let xg = cuda_f32(&x);
    let (vals, inverse, counts) = unique_consecutive(&xg).expect("gpu unique f32 all-same");
    assert!(vals.is_cuda());
    assert_eq!(read_back_f32(&vals), vec![7.0]);
    assert_eq!(inverse, vec![0, 0, 0, 0]);
    assert_eq!(counts, vec![4]);
}

#[test]
fn unique_consecutive_f64_runs_match_torch() {
    ensure_cuda();
    // torch.unique_consecutive(tensor([1.,1.,2.,3.,3.,3.,1.]).double(), ...) ->
    //   values [1,2,3,1], inverse [0,0,1,2,2,2,3], counts [2,1,3,1]
    let x: Vec<f64> = vec![1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 1.0];
    let xg = cuda_f64(&x);
    let (vals, inverse, counts) = unique_consecutive(&xg).expect("gpu unique_consecutive f64");
    assert!(vals.is_cuda());
    assert_eq!(read_back_f64(&vals), vec![1.0, 2.0, 3.0, 1.0]);
    assert_eq!(inverse, vec![0, 0, 1, 2, 2, 2, 3]);
    assert_eq!(counts, vec![2, 1, 3, 1]);

    // GPU == CPU.
    let xc = Tensor::from_storage(TensorStorage::cpu(x), vec![7], false).unwrap();
    let (vc, ic, cc) = unique_consecutive(&xc).unwrap();
    assert_eq!(read_back_f64(&vals), vc.data_vec().unwrap());
    assert_eq!(inverse, ic);
    assert_eq!(counts, cc);
}

#[test]
fn unique_consecutive_f64_2d_input_flattens_like_torch() {
    ensure_cuda();
    // torch.unique_consecutive on a 2-D tensor operates over the FLATTENED
    // C-order data (no dim given). [[2,2],[2,5]] flattens to [2,2,2,5] ->
    //   values [2,5], inverse [0,0,0,1], counts [3,1]
    let x: Vec<f64> = vec![2.0, 2.0, 2.0, 5.0];
    let xg = cuda_f64_shaped(&x, &[2, 2]);
    let (vals, inverse, counts) = unique_consecutive(&xg).expect("gpu unique f64 2d");
    assert!(vals.is_cuda());
    assert_eq!(read_back_f64(&vals), vec![2.0, 5.0]);
    assert_eq!(inverse, vec![0, 0, 0, 1]);
    assert_eq!(counts, vec![3, 1]);
}

#[test]
fn roll_f32_still_matches_torch_after_f64_addition() {
    // Guard: the f64 dispatch widening did not regress the f32 GPU roll path.
    ensure_cuda();
    let x: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let xg = cuda_f32_shaped(&x, &[6]);
    let yg = roll(&xg, 2, 0).expect("gpu roll f32 +2");
    assert!(yg.is_cuda());
    // torch.roll(arange(6).float(), 2) -> [4,5,0,1,2,3]
    assert_eq!(read_back_f32(&yg), vec![4.0, 5.0, 0.0, 1.0, 2.0, 3.0]);
}
