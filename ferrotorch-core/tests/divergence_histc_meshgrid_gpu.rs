//! GPU consumer-path audit (#1545 / sub #1535): the on-device `histc` /
//! `meshgrid` kernels added in `ferrotorch-gpu/src/search.rs`
//! (`gpu_histc_f{32,64}` / `gpu_meshgrid_f{32,64}`) and wired through
//! `ferrotorch-core/src/ops/search.rs` (`histc` / `meshgrid` CUDA branches via
//! `GpuBackend::histc_1d` / `meshgrid_grid`).
//!
//! These assert the PRODUCTION CONSUMER path (`ferrotorch_core::{histc,
//! meshgrid}`) on a genuinely CUDA-resident input:
//!   1. The result tensor stays `is_cuda()` (the histogram counts / grids are
//!      computed on-device and wrapped straight back; R-CODE-4 — no value
//!      round trip through host).
//!   2. The values match the LIVE torch oracle, byte-for-byte. The expected
//!      vectors below are the exact outputs of `torch.histc` / `torch.meshgrid`
//!      on torch 2.11.0+cu130 (RTX 3090), recorded inline as named references
//!      (R-CHAR-3: not copied from the ferrotorch GPU side):
//!        torch.histc(arange(0,11.), bins=5, min=0, max=10) -> [2,2,2,2,3]
//!        torch.histc([-1,.5,1.5,2.5,4,5,nan], 4, 0, 4)     -> [1,1,1,1]
//!        torch.histc(linspace(0,1,5).f64, 4, 0, 1)         -> [1,1,1,2]
//!        torch.meshgrid([1,2,3],[4,5], indexing='ij')[0]   -> [1,1,2,2,3,3]
//!        torch.meshgrid([1,2,3],[4,5], indexing='ij')[1]   -> [4,5,4,5,4,5]
//!
//! Upstream contract:
//!   - histc bins: `aten/src/ATen/native/cuda/SummaryOps.cu:41,47,92`
//!     (`getBin` + last-bin clamp + `[min,max]` guard; NaN/oob skipped).
//!   - meshgrid 'ij': `aten/src/ATen/native/TensorShape.cpp:4462-4467`
//!     (`view(view_shape).expand(shape)`).

#![cfg(feature = "gpu")]
#![allow(
    clippy::excessive_precision,
    reason = "all numeric literals in this file are live-torch 2.11 oracle values recorded verbatim (R-CHAR-3); full precision is intentional and rounds to the tensor dtype at compile time. Test-only oracle file."
)]
#![allow(
    clippy::doc_overindented_list_items,
    reason = "the oracle-call lines in the module doc are deliberately aligned as a fixed-width transcript block, not a markdown list"
)]

use ferrotorch_core::{Device, Tensor, TensorStorage, histc, meshgrid};
use std::sync::Once;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for the GPU histc/meshgrid audit");
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

/// Read a CUDA-resident tensor's values back to host for comparison (this is
/// the ONLY host crossing — the op itself ran entirely on device).
fn read_back_f32(t: &Tensor<f32>) -> Vec<f32> {
    t.to(Device::Cpu)
        .expect("download to cpu")
        .data_vec()
        .expect("data")
}

fn read_back_f64(t: &Tensor<f64>) -> Vec<f64> {
    t.to(Device::Cpu)
        .expect("download to cpu")
        .data_vec()
        .expect("data")
}

#[test]
fn gpu_histc_f32_matches_torch_and_stays_on_device() {
    ensure_cuda_backend();
    let input = cuda_f32(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    assert!(input.is_cuda(), "input must be CUDA-resident");
    let out = histc(&input, 5, 0.0, 10.0).expect("gpu histc f32");
    assert!(
        out.is_cuda(),
        "histc result must stay on device (R-CODE-4: no value round trip)"
    );
    // torch.histc(arange(0,11.), bins=5, min=0, max=10) -> [2,2,2,2,3]
    assert_eq!(read_back_f32(&out), vec![2.0, 2.0, 2.0, 2.0, 3.0]);
}

#[test]
fn gpu_histc_f32_skips_oob_and_nan_like_torch() {
    ensure_cuda_backend();
    let input = cuda_f32(&[-1.0, 0.5, 1.5, 2.5, 4.0, 5.0, f32::NAN]);
    let out = histc(&input, 4, 0.0, 4.0).expect("gpu histc f32 oob");
    assert!(out.is_cuda(), "histc result must stay on device");
    // torch drops -1 (below min), 5 (above max), nan -> [1,1,1,1]
    assert_eq!(read_back_f32(&out), vec![1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn gpu_histc_f64_matches_torch() {
    ensure_cuda_backend();
    let input = cuda_f64(&[0.0, 0.25, 0.5, 0.75, 1.0]);
    let out = histc(&input, 4, 0.0, 1.0).expect("gpu histc f64");
    assert!(out.is_cuda(), "histc f64 result must stay on device");
    // torch.histc(linspace(0,1,5).f64, 4, 0, 1) -> [1,1,1,2] (1.0 in last bin)
    assert_eq!(read_back_f64(&out), vec![1.0, 1.0, 1.0, 2.0]);
}

#[test]
fn gpu_meshgrid_f32_ij_matches_torch_and_stays_on_device() {
    ensure_cuda_backend();
    let a = cuda_f32(&[1.0, 2.0, 3.0]);
    let b = cuda_f32(&[4.0, 5.0]);
    let grids = meshgrid(&[a, b]).expect("gpu meshgrid f32");
    assert_eq!(grids.len(), 2);
    assert_eq!(grids[0].shape(), &[3, 2]);
    assert_eq!(grids[1].shape(), &[3, 2]);
    assert!(
        grids[0].is_cuda() && grids[1].is_cuda(),
        "grids must stay on device"
    );
    // torch.meshgrid([1,2,3],[4,5], indexing='ij')
    assert_eq!(read_back_f32(&grids[0]), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
    assert_eq!(read_back_f32(&grids[1]), vec![4.0, 5.0, 4.0, 5.0, 4.0, 5.0]);
}

#[test]
fn gpu_meshgrid_f64_ij_matches_torch() {
    ensure_cuda_backend();
    let a = cuda_f64(&[10.0, 20.0]);
    let b = cuda_f64(&[0.0, 1.0, 2.0]);
    let grids = meshgrid(&[a, b]).expect("gpu meshgrid f64");
    assert_eq!(grids.len(), 2);
    assert_eq!(grids[0].shape(), &[2, 3]);
    assert!(
        grids[0].is_cuda() && grids[1].is_cuda(),
        "grids must stay on device"
    );
    // torch.meshgrid([10,20],[0,1,2], indexing='ij')[0] -> 3x10 then 3x20
    assert_eq!(
        read_back_f64(&grids[0]),
        vec![10.0, 10.0, 10.0, 20.0, 20.0, 20.0]
    );
    // [1] -> [0,1,2] repeated per row
    assert_eq!(read_back_f64(&grids[1]), vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0]);
}
