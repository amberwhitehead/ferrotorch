//! GPU triangular-mask (`triu` / `tril`) integration tests — crosslink
//! #1545 / sub #1535.
//!
//! Exercises the full `ferrotorch_core::{triu, tril}` dispatch on a real CUDA
//! device. Asserts the result tensor is GPU-resident (`is_cuda()`, NO
//! `.cpu()` round trip) AND that its on-device values are byte-identical to
//! the CPU `triu`/`tril` reference. Predicate parity is with
//! `torch.{triu,tril}` (`aten/src/ATen/native/cuda/TriangularOps.cu:100`):
//! triu keeps `col - row >= k`, tril keeps `col - row <= k`, else 0.

#![cfg(feature = "cuda")]

use ferrotorch_core::{Device, Tensor, TensorStorage, tril, triu};
use ferrotorch_gpu::init_cuda_backend;

fn ensure_cuda() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        init_cuda_backend().expect("CUDA backend init");
    });
}

fn cpu_mat(rows: usize, cols: usize) -> Tensor<f32> {
    let data: Vec<f32> = (0..rows * cols).map(|i| (i + 1) as f32).collect();
    Tensor::from_storage(TensorStorage::cpu(data), vec![rows, cols], false).expect("cpu tensor")
}

fn cpu_mat_f64(rows: usize, cols: usize) -> Tensor<f64> {
    let data: Vec<f64> = (0..rows * cols).map(|i| (i + 1) as f64).collect();
    Tensor::from_storage(TensorStorage::cpu(data), vec![rows, cols], false).expect("cpu tensor")
}

#[test]
fn triu_f32_on_cuda_matches_cpu_and_stays_resident() {
    ensure_cuda();
    let cpu = cpu_mat(4, 4);
    let gpu = cpu.clone().to(Device::Cuda(0)).expect("to cuda");
    assert!(gpu.is_cuda());

    for k in [-2i64, -1, 0, 1, 2] {
        let gpu_out = triu(&gpu, k).expect("gpu triu");
        // R-CODE-4: result must stay GPU-resident — no host round trip.
        assert!(gpu_out.is_cuda(), "triu(k={k}) result not GPU-resident");
        assert_eq!(gpu_out.shape(), &[4, 4]);

        let cpu_out = triu(&cpu, k).expect("cpu triu");
        let gpu_host = gpu_out.cpu().expect("cpu()").data().unwrap().to_vec();
        let cpu_data = cpu_out.data().unwrap().to_vec();
        assert_eq!(gpu_host, cpu_data, "GPU triu(k={k}) != CPU reference");
    }
}

#[test]
fn tril_f32_on_cuda_matches_cpu_and_stays_resident() {
    ensure_cuda();
    let cpu = cpu_mat(4, 4);
    let gpu = cpu.clone().to(Device::Cuda(0)).expect("to cuda");

    for k in [-2i64, -1, 0, 1, 2] {
        let gpu_out = tril(&gpu, k).expect("gpu tril");
        assert!(gpu_out.is_cuda(), "tril(k={k}) result not GPU-resident");
        let cpu_out = tril(&cpu, k).expect("cpu tril");
        let gpu_host = gpu_out.cpu().expect("cpu()").data().unwrap().to_vec();
        let cpu_data = cpu_out.data().unwrap().to_vec();
        assert_eq!(gpu_host, cpu_data, "GPU tril(k={k}) != CPU reference");
    }
}

#[test]
fn triu_tril_f32_nonsquare_on_cuda() {
    ensure_cuda();
    // Rectangular 3x5 and 5x3 — verifies row/col factorisation in the kernel.
    for (r, c) in [(3usize, 5usize), (5, 3)] {
        let cpu = cpu_mat(r, c);
        let gpu = cpu.clone().to(Device::Cuda(0)).expect("to cuda");
        let up = triu(&gpu, 0).expect("gpu triu");
        let lo = tril(&gpu, 0).expect("gpu tril");
        assert!(up.is_cuda() && lo.is_cuda());
        assert_eq!(
            up.cpu().unwrap().data().unwrap().to_vec(),
            triu(&cpu, 0).unwrap().data().unwrap().to_vec()
        );
        assert_eq!(
            lo.cpu().unwrap().data().unwrap().to_vec(),
            tril(&cpu, 0).unwrap().data().unwrap().to_vec()
        );
    }
}

#[test]
fn triu_tril_f64_on_cuda_matches_cpu() {
    ensure_cuda();
    let cpu = cpu_mat_f64(4, 4);
    let gpu = cpu.clone().to(Device::Cuda(0)).expect("to cuda");
    let up = triu(&gpu, 0).expect("gpu triu f64");
    let lo = tril(&gpu, -1).expect("gpu tril f64");
    assert!(up.is_cuda() && lo.is_cuda());
    assert_eq!(
        up.cpu().unwrap().data().unwrap().to_vec(),
        triu(&cpu, 0).unwrap().data().unwrap().to_vec()
    );
    assert_eq!(
        lo.cpu().unwrap().data().unwrap().to_vec(),
        tril(&cpu, -1).unwrap().data().unwrap().to_vec()
    );
}
