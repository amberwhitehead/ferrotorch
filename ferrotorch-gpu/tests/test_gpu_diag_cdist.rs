//! GPU `diag` / `diagflat` / `cdist` integration tests — crosslink
//! #1545 / sub #1535.
//!
//! Exercises the full `ferrotorch_core::{diag, diagflat, cdist}` dispatch on a
//! real CUDA device. Asserts the result tensor is GPU-resident (`is_cuda()`,
//! NO `.cpu()` round trip) AND that its on-device values match the CPU
//! reference (byte-identical for the pure-gather/scatter diag/diagflat; within
//! fp tolerance for the cdist reductions). Parity is with `torch.diag`
//! (`aten/src/ATen/native/TensorShape.cpp:4610`) and `torch.cdist`
//! (`aten/src/ATen/native/cuda/DistanceKernel.cu:195`).

#![cfg(feature = "cuda")]

use ferrotorch_core::{Device, Tensor, TensorStorage, cdist, diag, diagflat};
use ferrotorch_gpu::init_cuda_backend;

fn ensure_cuda() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        init_cuda_backend().expect("CUDA backend init");
    });
}

fn t1d(data: &[f32]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false)
        .expect("cpu tensor")
}

fn t2d(data: &[f32], r: usize, c: usize) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![r, c], false).expect("cpu tensor")
}

fn t1d_f64(data: &[f64]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false)
        .expect("cpu tensor")
}

#[test]
fn diag_embed_f32_on_cuda_matches_cpu_and_stays_resident() {
    ensure_cuda();
    let cpu = t1d(&[1.0, 2.0, 3.0, 4.0]);
    let gpu = cpu.clone().to(Device::Cuda(0)).expect("to cuda");
    assert!(gpu.is_cuda());

    for k in [-2i64, -1, 0, 1, 2] {
        let gpu_out = diag(&gpu, k).expect("gpu diag embed");
        assert!(gpu_out.is_cuda(), "diag(1d, k={k}) result not GPU-resident");
        let cpu_out = diag(&cpu, k).expect("cpu diag embed");
        assert_eq!(gpu_out.shape(), cpu_out.shape());
        let gpu_host = gpu_out.cpu().expect("cpu()").data().unwrap().to_vec();
        let cpu_data = cpu_out.data().unwrap().to_vec();
        assert_eq!(gpu_host, cpu_data, "GPU diag_embed(k={k}) != CPU reference");
    }
}

#[test]
fn diag_extract_f32_on_cuda_matches_cpu_and_stays_resident() {
    ensure_cuda();
    // 3x4 rectangular so positive/negative offsets exercise both clamps.
    let data: Vec<f32> = (1..=12).map(|i| i as f32).collect();
    let cpu = t2d(&data, 3, 4);
    let gpu = cpu.clone().to(Device::Cuda(0)).expect("to cuda");

    for k in [-2i64, -1, 0, 1, 2, 3] {
        let gpu_out = diag(&gpu, k).expect("gpu diag extract");
        assert!(gpu_out.is_cuda(), "diag(2d, k={k}) result not GPU-resident");
        let cpu_out = diag(&cpu, k).expect("cpu diag extract");
        assert_eq!(gpu_out.shape(), cpu_out.shape(), "shape mismatch k={k}");
        let gpu_host = gpu_out.cpu().expect("cpu()").data().unwrap().to_vec();
        let cpu_data = cpu_out.data().unwrap().to_vec();
        assert_eq!(
            gpu_host, cpu_data,
            "GPU diag_extract(k={k}) != CPU reference"
        );
    }
}

#[test]
fn diagflat_f32_on_cuda_matches_cpu_and_stays_resident() {
    ensure_cuda();
    // 2x2 input flattened to 4 then embedded -> 4x4.
    let cpu = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let gpu = cpu.clone().to(Device::Cuda(0)).expect("to cuda");

    let gpu_out = diagflat(&gpu, 0).expect("gpu diagflat");
    assert!(gpu_out.is_cuda(), "diagflat result not GPU-resident");
    assert_eq!(gpu_out.shape(), &[4, 4]);
    let cpu_out = diagflat(&cpu, 0).expect("cpu diagflat");
    let gpu_host = gpu_out.cpu().expect("cpu()").data().unwrap().to_vec();
    let cpu_data = cpu_out.data().unwrap().to_vec();
    assert_eq!(gpu_host, cpu_data, "GPU diagflat != CPU reference");
}

#[test]
fn diag_embed_f64_on_cuda_matches_cpu() {
    ensure_cuda();
    let cpu = t1d_f64(&[1.0, 2.0, 3.0]);
    let gpu = cpu.clone().to(Device::Cuda(0)).expect("to cuda");
    let gpu_out = diag(&gpu, 1).expect("gpu diag f64");
    assert!(gpu_out.is_cuda());
    let cpu_out = diag(&cpu, 1).expect("cpu diag f64");
    let gpu_host = gpu_out.cpu().expect("cpu()").data().unwrap().to_vec();
    assert_eq!(gpu_host, cpu_out.data().unwrap().to_vec());
}

fn assert_close(got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(want) {
        assert!((g - w).abs() < tol, "got {g} want {w}");
    }
}

#[test]
fn cdist_f32_l2_on_cuda_matches_cpu_and_stays_resident() {
    ensure_cuda();
    let x1 = t2d(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0], 3, 2);
    let x2 = t2d(&[1.0, 1.0], 1, 2);
    let g1 = x1.clone().to(Device::Cuda(0)).expect("to cuda");
    let g2 = x2.clone().to(Device::Cuda(0)).expect("to cuda");

    let gpu_out = cdist(&g1, &g2, 2.0).expect("gpu cdist l2");
    assert!(gpu_out.is_cuda(), "cdist result not GPU-resident");
    assert_eq!(gpu_out.shape(), &[3, 1]);
    let cpu_out = cdist(&x1, &x2, 2.0).expect("cpu cdist l2");
    let gpu_host = gpu_out.cpu().expect("cpu()").data().unwrap().to_vec();
    assert_close(&gpu_host, cpu_out.data().unwrap(), 1e-4);
    // torch: [sqrt(2), 1, 1]
    assert_close(&gpu_host, &[2.0f32.sqrt(), 1.0, 1.0], 1e-4);
}

#[test]
fn cdist_f32_l1_linf_on_cuda_match_cpu() {
    ensure_cuda();
    let x1 = t2d(&[0.0, 0.0, 3.0, 1.0], 2, 2);
    let x2 = t2d(&[1.0, 5.0], 1, 2);
    let g1 = x1.clone().to(Device::Cuda(0)).expect("to cuda");
    let g2 = x2.clone().to(Device::Cuda(0)).expect("to cuda");

    let l1 = cdist(&g1, &g2, 1.0).expect("gpu cdist l1");
    assert!(l1.is_cuda());
    let l1_host = l1.cpu().expect("cpu()");
    let l1_cpu = cdist(&x1, &x2, 1.0).unwrap();
    assert_close(l1_host.data().unwrap(), l1_cpu.data().unwrap(), 1e-4);

    let linf = cdist(&g1, &g2, f64::INFINITY).expect("gpu cdist linf");
    assert!(linf.is_cuda());
    let host = linf.cpu().expect("cpu()").data().unwrap().to_vec();
    // max(|0-1|,|0-5|)=5 ; max(|3-1|,|1-5|)=4
    assert_close(&host, &[5.0, 4.0], 1e-4);
}

#[test]
fn cdist_f32_batched_on_cuda_matches_cpu() {
    ensure_cuda();
    let x1d: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let x2d: Vec<f32> = (8..16).map(|i| i as f32).collect();
    let x1 = Tensor::from_storage(TensorStorage::cpu(x1d), vec![2, 2, 2], false).unwrap();
    let x2 = Tensor::from_storage(TensorStorage::cpu(x2d), vec![2, 2, 2], false).unwrap();
    let g1 = x1.clone().to(Device::Cuda(0)).expect("to cuda");
    let g2 = x2.clone().to(Device::Cuda(0)).expect("to cuda");

    let gpu_out = cdist(&g1, &g2, 2.0).expect("gpu cdist batched");
    assert!(gpu_out.is_cuda());
    assert_eq!(gpu_out.shape(), &[2, 2, 2]);
    let cpu_out = cdist(&x1, &x2, 2.0).expect("cpu cdist batched");
    let gpu_host = gpu_out.cpu().expect("cpu()");
    assert_close(gpu_host.data().unwrap(), cpu_out.data().unwrap(), 1e-4);
}
