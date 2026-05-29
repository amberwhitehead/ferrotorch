//! Live-GPU verification of the segmented row scatter-add
//! (`ferrotorch_core::scatter_add_segments`) on CUDA (crosslink #1545 / sub
//! #1535). These exercise the real PTX kernel
//! `scatter_add_segments_f{32,64}_kernel` in
//! `ferrotorch-gpu/src/scatter_gather_kernels.rs` through the
//! `ferrotorch-core::ops::scatter::scatter_add_segments` CUDA dispatch branch
//! and verify byte-exact parity with the torch oracle
//! `torch.zeros(dim_size, D).index_add_(0, index, src)` (the canonical
//! segmented row sum — same result `torch_scatter.scatter_add(src, index,
//! dim=0, dim_size=N)` produces).
//!
//! # R-CHAR-3 provenance (live torch 2.x, CUDA)
//!
//! Every expected value below is the output of the named torch python on the
//! identical fixture; these are symbolic constants traceable to the exact
//! torch call (not self-derived from ferrotorch).
//!
//! ```python
//! import torch
//! d = "cuda"
//!
//! # basic: 3 rows of D=2 -> 2 output rows, index = [0, 1, 0]
//! src = torch.tensor([[1.,2.],[3.,4.],[5.,6.]], device=d)        # [3,2]
//! idx = torch.tensor([0, 1, 0], device=d)
//! torch.zeros(2, 2, device=d).index_add_(0, idx, src)
//! #   out[0] = src[0] + src[2] = [6., 8.]; out[1] = src[1] = [3., 4.]
//! #   tensor([[6., 8.], [3., 4.]])
//!
//! # DUPLICATE segments (atomic accumulation — the key case):
//! # 100 rows of D=3, every value 1.0, all index 0, dim_size = 2.
//! src = torch.ones(100, 3, device=d)
//! idx = torch.zeros(100, dtype=torch.long, device=d)
//! torch.zeros(2, 3, device=d).index_add_(0, idx, src)
//! #   out[0] = column sums = [100., 100., 100.]; out[1] = [0., 0., 0.]
//! #   (1.0 * 100 is exact in f32 — chosen so the sum has no rounding)
//!
//! # empty output row stays exactly 0:
//! # src = [[7., 0.5],[8., 0.25]], index = [0, 0], dim_size = 3
//! src = torch.tensor([[7.,0.5],[8.,0.25]], device=d)
//! idx = torch.tensor([0, 0], device=d)
//! torch.zeros(3, 2, device=d).index_add_(0, idx, src)
//! #   out[0] = [15., 0.75]; out[1] = [0., 0.]; out[2] = [0., 0.]
//! ```
//!
//! The f64 variants use the identical fixtures with `dtype=torch.float64`;
//! torch's f64 outputs equal the f32 outputs exactly for these data (no
//! rounding for the integer / exactly-representable values chosen).

#![cfg(feature = "cuda")]

use ferrotorch_core::{Device, Tensor, TensorStorage, scatter_add_segments};
use ferrotorch_gpu::init_cuda_backend;
use half::{bf16, f16};

fn ensure_cuda() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        init_cuda_backend().expect("CUDA backend init");
    });
}

fn cpu_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .expect("cpu f32 tensor")
}

fn cpu_f64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .expect("cpu f64 tensor")
}

fn host_f32(t: &Tensor<f32>) -> Vec<f32> {
    t.cpu().expect("cpu()").data().unwrap().to_vec()
}

fn host_f64(t: &Tensor<f64>) -> Vec<f64> {
    t.cpu().expect("cpu()").data().unwrap().to_vec()
}

// ===========================================================================
// basic aggregation — torch index_add_(0, [0,1,0], [[1,2],[3,4],[5,6]])
// ===========================================================================

#[test]
fn scatter_add_segments_gpu_f32_basic_matches_torch() {
    ensure_cuda();
    let cpu = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let gpu = cpu.to(Device::Cuda(0)).expect("to cuda");
    let index = [0i64, 1, 0];
    let out = scatter_add_segments(&gpu, &index, 2).expect("gpu scatter_add_segments");
    assert!(out.is_cuda(), "result must stay GPU-resident");
    assert_eq!(out.shape(), &[2, 2]);
    // torch.zeros(2,2).index_add_(0, [0,1,0], src) == [[6,8],[3,4]]
    assert_eq!(host_f32(&out), vec![6.0, 8.0, 3.0, 4.0]);
    // GPU == ferrotorch CPU reference.
    let cpu_out = scatter_add_segments(&cpu, &index, 2).expect("cpu scatter_add_segments");
    assert_eq!(host_f32(&out), cpu_out.data().unwrap().to_vec());
}

#[test]
fn scatter_add_segments_gpu_f64_basic_matches_torch() {
    ensure_cuda();
    let cpu = cpu_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let gpu = cpu.to(Device::Cuda(0)).expect("to cuda");
    let index = [0i64, 1, 0];
    let out = scatter_add_segments(&gpu, &index, 2).expect("gpu scatter_add_segments f64");
    assert!(out.is_cuda());
    assert_eq!(host_f64(&out), vec![6.0, 8.0, 3.0, 4.0]);
    let cpu_out = scatter_add_segments(&cpu, &index, 2).expect("cpu scatter_add_segments f64");
    assert_eq!(host_f64(&out), cpu_out.data().unwrap().to_vec());
}

// ===========================================================================
// DUPLICATE segments — 100 rows all into segment 0 (the atomic case).
// 1.0 * 100 is exact in f32, so the accumulated sum is checked exactly.
// ===========================================================================

#[test]
fn scatter_add_segments_gpu_f32_duplicate_atomic_matches_torch() {
    ensure_cuda();
    let e = 100usize;
    let dd = 3usize;
    let cpu = cpu_f32(&vec![1.0f32; e * dd], &[e, dd]);
    let gpu = cpu.to(Device::Cuda(0)).expect("to cuda");
    let index = vec![0i64; e]; // all rows -> segment 0
    let out = scatter_add_segments(&gpu, &index, 2).expect("gpu dup atomic");
    assert!(out.is_cuda());
    assert_eq!(out.shape(), &[2, 3]);
    // torch.zeros(2,3).index_add_(0, zeros(100), ones(100,3))
    //   == [[100,100,100],[0,0,0]]
    assert_eq!(
        host_f32(&out),
        vec![100.0, 100.0, 100.0, 0.0, 0.0, 0.0],
        "atomic accumulation of 100 duplicate-segment rows must sum, not last-write-wins"
    );
    let cpu_out = scatter_add_segments(&cpu, &index, 2).expect("cpu dup atomic");
    assert_eq!(host_f32(&out), cpu_out.data().unwrap().to_vec());
}

#[test]
fn scatter_add_segments_gpu_f64_duplicate_atomic_matches_torch() {
    ensure_cuda();
    let e = 100usize;
    let dd = 3usize;
    let cpu = cpu_f64(&vec![1.0f64; e * dd], &[e, dd]);
    let gpu = cpu.to(Device::Cuda(0)).expect("to cuda");
    let index = vec![0i64; e];
    let out = scatter_add_segments(&gpu, &index, 2).expect("gpu dup atomic f64");
    assert!(out.is_cuda());
    assert_eq!(
        host_f64(&out),
        vec![100.0, 100.0, 100.0, 0.0, 0.0, 0.0],
        "f64 atomic accumulation of 100 duplicate-segment rows must sum"
    );
}

// ===========================================================================
// empty output row stays exactly 0
// ===========================================================================

#[test]
fn scatter_add_segments_gpu_f32_empty_row_stays_zero() {
    ensure_cuda();
    // src=[[7,0.5],[8,0.25]], index=[0,0], dim_size=3.
    let cpu = cpu_f32(&[7.0, 0.5, 8.0, 0.25], &[2, 2]);
    let gpu = cpu.to(Device::Cuda(0)).expect("to cuda");
    let index = [0i64, 0];
    let out = scatter_add_segments(&gpu, &index, 3).expect("gpu empty-row");
    assert!(out.is_cuda());
    assert_eq!(out.shape(), &[3, 2]);
    // torch.zeros(3,2).index_add_(0, [0,0], src) == [[15,0.75],[0,0],[0,0]]
    let got = host_f32(&out);
    assert_eq!(got[..2], [15.0, 0.75]);
    // Rows 1 and 2 are zero-initialised on device and never touched by any
    // atomic add — exact bitwise zero is the right tightness.
    for &v in &got[2..] {
        assert_eq!(v, 0.0, "empty output row must stay exactly 0, got {v}");
    }
}

// ===========================================================================
// bf16 / f16 on CUDA reject with NotImplementedOnCuda
// ===========================================================================

#[test]
fn scatter_add_segments_gpu_bf16_rejects() {
    ensure_cuda();
    let data: Vec<bf16> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .map(|&v| bf16::from_f32(v))
        .collect();
    let cpu = Tensor::from_storage(TensorStorage::cpu(data), vec![2, 2], false).expect("bf16 cpu");
    let gpu = cpu.to(Device::Cuda(0)).expect("to cuda");
    let err = scatter_add_segments(&gpu, &[0i64, 1], 2);
    assert!(
        err.is_err(),
        "bf16 CUDA scatter_add_segments must reject NotImplementedOnCuda"
    );
}

#[test]
fn scatter_add_segments_gpu_f16_rejects() {
    ensure_cuda();
    let data: Vec<f16> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .map(|&v| f16::from_f32(v))
        .collect();
    let cpu = Tensor::from_storage(TensorStorage::cpu(data), vec![2, 2], false).expect("f16 cpu");
    let gpu = cpu.to(Device::Cuda(0)).expect("to cuda");
    let err = scatter_add_segments(&gpu, &[0i64, 1], 2);
    assert!(
        err.is_err(),
        "f16 CUDA scatter_add_segments must reject NotImplementedOnCuda"
    );
}
