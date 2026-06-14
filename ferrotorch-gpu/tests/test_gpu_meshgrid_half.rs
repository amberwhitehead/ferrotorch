//! End-to-end CUDA `meshgrid` coverage for f16 / bf16.
//!
//! These tests exercise the production `ferrotorch_core::meshgrid` path:
//! CUDA tensors dispatch through `GpuBackend::meshgrid_grid`, each grid remains
//! GPU-resident with the original half dtype tag, and the only host crossing is
//! the test readback.

#![cfg(feature = "cuda")]

use ferrotorch_core::{Device, MeshIndexing, Tensor, TensorStorage, meshgrid, meshgrid_indexing};
use ferrotorch_gpu::init_cuda_backend;
use half::{bf16, f16};

fn ensure_cuda() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        init_cuda_backend().expect("CUDA backend init");
    });
}

fn cpu_f16(data: &[f16], shape: &[usize]) -> Tensor<f16> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .expect("cpu f16 tensor")
}

fn cpu_bf16(data: &[bf16], shape: &[usize]) -> Tensor<bf16> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .expect("cpu bf16 tensor")
}

fn read_f16(t: &Tensor<f16>) -> Vec<f32> {
    t.to(Device::Cpu)
        .expect("download f16 grid")
        .data()
        .expect("host f16 data")
        .iter()
        .map(|v| v.to_f32())
        .collect()
}

fn read_bf16(t: &Tensor<bf16>) -> Vec<f32> {
    t.to(Device::Cpu)
        .expect("download bf16 grid")
        .data()
        .expect("host bf16 data")
        .iter()
        .map(|v| v.to_f32())
        .collect()
}

#[test]
fn meshgrid_f16_cuda_ij_matches_torch_and_stays_resident() {
    ensure_cuda();
    let a = cpu_f16(
        &[f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)],
        &[3],
    )
    .to(Device::Cuda(0))
    .expect("upload f16 a")
    .detach()
    .requires_grad_(true);
    let b = cpu_f16(&[f16::from_f32(4.0), f16::from_f32(5.0)], &[2])
        .to(Device::Cuda(0))
        .expect("upload f16 b")
        .detach()
        .requires_grad_(true);

    let grids = meshgrid(&[a, b]).expect("meshgrid f16 cuda");
    assert_eq!(grids.len(), 2);
    assert_eq!(grids[0].shape(), &[3, 2]);
    assert_eq!(grids[1].shape(), &[3, 2]);
    assert!(grids[0].is_cuda() && grids[1].is_cuda());
    assert!(grids[0].requires_grad() && grids[1].requires_grad());
    assert_eq!(read_f16(&grids[0]), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
    assert_eq!(read_f16(&grids[1]), vec![4.0, 5.0, 4.0, 5.0, 4.0, 5.0]);
}

#[test]
fn meshgrid_bf16_cuda_xy_matches_torch_and_stays_resident() {
    ensure_cuda();
    let a = cpu_bf16(
        &[
            bf16::from_f32(1.0),
            bf16::from_f32(2.0),
            bf16::from_f32(3.0),
        ],
        &[3],
    )
    .to(Device::Cuda(0))
    .expect("upload bf16 a")
    .detach()
    .requires_grad_(true);
    let b = cpu_bf16(&[bf16::from_f32(4.0), bf16::from_f32(5.0)], &[2])
        .to(Device::Cuda(0))
        .expect("upload bf16 b")
        .detach()
        .requires_grad_(true);

    let grids = meshgrid_indexing(&[a, b], MeshIndexing::Xy).expect("meshgrid bf16 cuda xy");
    assert_eq!(grids.len(), 2);
    assert_eq!(grids[0].shape(), &[2, 3]);
    assert_eq!(grids[1].shape(), &[2, 3]);
    assert!(grids[0].is_cuda() && grids[1].is_cuda());
    assert!(grids[0].requires_grad() && grids[1].requires_grad());
    assert_eq!(read_bf16(&grids[0]), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    assert_eq!(read_bf16(&grids[1]), vec![4.0, 4.0, 4.0, 5.0, 5.0, 5.0]);
}

#[test]
fn meshgrid_f16_cuda_empty_axis_returns_empty_cuda_grids() {
    ensure_cuda();
    let a = cpu_f16(&[], &[0])
        .to(Device::Cuda(0))
        .expect("upload empty f16 a");
    let b = cpu_f16(&[f16::from_f32(4.0), f16::from_f32(5.0)], &[2])
        .to(Device::Cuda(0))
        .expect("upload f16 b");

    let grids = meshgrid(&[a, b]).expect("meshgrid f16 cuda empty");
    assert_eq!(grids.len(), 2);
    assert_eq!(grids[0].shape(), &[0, 2]);
    assert_eq!(grids[1].shape(), &[0, 2]);
    assert!(grids[0].is_cuda() && grids[1].is_cuda());
    assert!(read_f16(&grids[0]).is_empty());
    assert!(read_f16(&grids[1]).is_empty());
}
