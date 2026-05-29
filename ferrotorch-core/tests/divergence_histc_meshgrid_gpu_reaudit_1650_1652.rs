//! Acto-critic RE-AUDIT (GPU paths) of commit `7cbb0a27d`. The commit's
//! `divergence_histc_meshgrid_gpu.rs` covers GPU histc skip-oob/NaN and GPU
//! meshgrid 'ij', but NOT (a) the GPU default-range `min==max` inference path
//! or (b) GPU meshgrid 'xy'. These pin those two on a genuinely CUDA-resident
//! input, asserting the result equals LIVE torch 2.11 on the RTX 3090 AND
//! stays on device (R-CODE-4).
//!
//! LIVE torch oracle (torch 2.11.0+cu130, cuda device; named refs per R-CHAR-3,
//! NOT copied from the ferrotorch GPU side):
//!   torch.histc(tensor([1,2,3,4,5],device='cuda'), bins=4)    -> [1,1,1,2]
//!   torch.histc(tensor([3,3,3],device='cuda'),     bins=4)    -> [0,0,3,0]
//!   torch.meshgrid([1,2,3],[4,5] (cuda), indexing='xy')[0]    -> [1,2,3,1,2,3] (shape [2,3], device cuda)
//!   torch.meshgrid([1,2,3],[4,5] (cuda), indexing='xy')[1]    -> [4,4,4,5,5,5]
//!
//! These confirm CPU==GPU==torch for the new default-range + 'xy' logic (the
//! same scalar inference runs before the device branch, so the GPU histogram
//! must agree with the CPU one and with torch).

#![cfg(feature = "gpu")]

use ferrotorch_core::{Device, MeshIndexing, Tensor, TensorStorage, histc, meshgrid_indexing};
use std::sync::Once;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for the GPU histc/meshgrid re-audit");
    });
}

fn cuda_f32(data: &[f32]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false)
        .expect("cpu f32 tensor")
        .to(Device::Cuda(0))
        .expect("upload f32 to cuda")
}

fn read_back_f32(t: &Tensor<f32>) -> Vec<f32> {
    t.to(Device::Cpu)
        .expect("download to cpu")
        .data_vec()
        .expect("data")
}

/// RE-AUDIT GPU histc default-range inference: a CUDA-resident input with the
/// default `min==max==0` must infer the range from the data and produce the
/// same counts as CPU torch (`SummaryOps.cu:328-331`). Verifies CPU==GPU==torch.
#[test]
fn reaudit_gpu_histc_default_minmax_infers_range() {
    ensure_cuda_backend();
    let input = cuda_f32(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    assert!(input.is_cuda(), "input must be CUDA-resident");
    let out = histc(&input, 4, 0.0, 0.0).expect("gpu histc default min==max==0");
    // torch.histc(tensor([1,2,3,4,5],device='cuda'), bins=4) -> [1,1,1,2]
    assert_eq!(read_back_f32(&out), vec![1.0, 1.0, 1.0, 2.0]);
}

/// RE-AUDIT GPU histc all-equal widen: CUDA input [3,3,3] default range -> torch
/// widens to [2,4] (`SummaryOps.cu:333-335`) -> [0,0,3,0]. CPU==GPU==torch.
#[test]
fn reaudit_gpu_histc_all_equal_widens() {
    ensure_cuda_backend();
    let input = cuda_f32(&[3.0, 3.0, 3.0]);
    let out = histc(&input, 4, 0.0, 0.0).expect("gpu histc all-equal default range");
    // torch.histc(tensor([3,3,3],device='cuda'), bins=4) -> [0,0,3,0]
    assert_eq!(read_back_f32(&out), vec![0.0, 0.0, 3.0, 0.0]);
}

/// RE-AUDIT GPU meshgrid 'xy': CUDA-resident inputs, the 'xy' swap, grids stay
/// on device (`TensorShape.cpp:4433-4438,4470-4472`).
/// torch.meshgrid([1,2,3],[4,5] cuda, indexing='xy') -> shape [2,3],
/// grid0=[1,2,3,1,2,3], grid1=[4,4,4,5,5,5], both device cuda.
#[test]
fn reaudit_gpu_meshgrid_xy_stays_on_device() {
    ensure_cuda_backend();
    let a = cuda_f32(&[1.0, 2.0, 3.0]);
    let b = cuda_f32(&[4.0, 5.0]);
    let grids = meshgrid_indexing(&[a, b], MeshIndexing::Xy).expect("gpu meshgrid xy");
    assert_eq!(grids.len(), 2);
    assert_eq!(grids[0].shape(), &[2, 3]);
    assert_eq!(grids[1].shape(), &[2, 3]);
    assert!(
        grids[0].is_cuda() && grids[1].is_cuda(),
        "xy grids must stay on device (R-CODE-4)"
    );
    assert_eq!(read_back_f32(&grids[0]), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    assert_eq!(read_back_f32(&grids[1]), vec![4.0, 4.0, 4.0, 5.0, 5.0, 5.0]);
}
