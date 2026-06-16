//! Red-then-green regression tests for audit finding CORE-076 (crosslink
//! #1770): the CUDA `sparse_matmul_24` fallback silently returns a CPU
//! tensor (CLASS-S — when the cuSPARSELt backend declines (feature off /
//! `libcusparseLt` missing / shape), or when `T != f32`, the reference
//! path downloaded `a` via `data_vec` and constructed CPU storage, so a
//! SUCCESSFUL op changes device).
//!
//! Observed at HEAD (red run, 2026-06-12, `--features gpu` [cusparselt
//! off], RTX 3090):
//! - f32 CUDA `a`: backend declined -> `Ok` with `out.is_cuda() == false`.
//! - f64 CUDA `a`: fast path skipped -> `Ok` with `out.is_cuda() == false`.
//! - bf16 CUDA `a`: same silent host detour / later explicit gap.
//!
//! Contract (rust-gpu-discipline §3 / goal-audit-fix.md R-LOUD-1; torch
//! parity: `torch.sparse.SparseSemiStructuredTensor @ dense` on CUDA
//! returns a CUDA tensor — the dispatcher never silently migrates the
//! output to CPU). Post-fix: a CUDA `a` never reaches the host reference
//! path — f32/f16/bf16 fall back to an ON-DEVICE composite against the
//! uploaded decompressed weight when cuSPARSELt declines, and f64 runs
//! the on-device composite directly.

#![cfg(feature = "gpu")]

use ferrotorch_core::{
    Device, SemiStructuredSparseTensor, Tensor, TensorStorage, sparse_matmul_24,
};
use std::sync::Once;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend().expect("CUDA backend must initialize for the GPU lane");
    });
}

fn mk_f32(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data), shape, false).unwrap()
}

/// f32 with a declined cuSPARSELt backend (this build has `gpu` but not
/// `cusparselt`): the composite fallback must stay on CUDA and match the
/// CPU reference.
#[test]
fn core076_gpu_f32_declined_backend_stays_on_cuda() {
    ensure_cuda_backend();

    let a_data: Vec<f32> = (1..=8).map(|x| x as f32).collect(); // [2, 4]
    let b_data: Vec<f32> = vec![
        1.0, 4.0, 2.0, 3.0, //
        -5.0, 2.0, 0.0, 1.0, //
        0.5, -0.25, 8.0, 7.0, //
        3.0, 3.0, -3.0, 0.125,
    ]; // [4, 4]
    let a_cpu = mk_f32(a_data, vec![2, 4]);
    let b = SemiStructuredSparseTensor::compress(&mk_f32(b_data, vec![4, 4])).unwrap();

    let cpu_out = sparse_matmul_24(&a_cpu, &b).expect("cpu reference");
    let cpu_ref = cpu_out.data().expect("cpu data").to_vec();

    let a_gpu = a_cpu.to(Device::Cuda(0)).expect("a->cuda");
    let out = sparse_matmul_24(&a_gpu, &b).expect("cuda sparse_matmul_24");
    assert!(
        out.is_cuda(),
        "sparse_matmul_24 output must stay on CUDA when `a` is CUDA \
         (pre-fix: declined backend silently returned CPU storage)"
    );
    assert_eq!(out.shape(), &[2, 4]);
    let back = out.cpu().expect("gpu->cpu");
    let got = back.data().expect("data");
    for (i, (g, e)) in got.iter().zip(cpu_ref.iter()).enumerate() {
        // f32 GEMM accumulation over k=4 — 1e-5 relative covers kernel
        // vs scalar-loop association differences at this tiny k.
        assert!(
            (g - e).abs() < 1e-5 * (1.0 + e.abs()),
            "elem {i}: gpu={g} cpu={e}"
        );
    }
}

/// f64 CUDA `a`: must run the on-device composite (pre-fix it skipped
/// the fast path entirely and silently returned CPU storage).
#[test]
fn core076_gpu_f64_stays_on_cuda() {
    ensure_cuda_backend();

    let a_data: Vec<f64> = (1..=8).map(|x| x as f64 * 0.5).collect();
    let b_data: Vec<f64> = (1..=16).map(|x| (x as f64) - 8.0).collect();
    let a_cpu = Tensor::<f64>::from_storage(TensorStorage::cpu(a_data), vec![2, 4], false).unwrap();
    let b_dense =
        Tensor::<f64>::from_storage(TensorStorage::cpu(b_data), vec![4, 4], false).unwrap();
    let b = SemiStructuredSparseTensor::compress(&b_dense).unwrap();

    let cpu_out = sparse_matmul_24(&a_cpu, &b).expect("cpu reference");
    let cpu_ref = cpu_out.data().expect("cpu data").to_vec();

    let a_gpu = a_cpu.to(Device::Cuda(0)).expect("a->cuda");
    let out = sparse_matmul_24(&a_gpu, &b).expect("cuda f64 sparse_matmul_24");
    assert!(
        out.is_cuda(),
        "f64 sparse_matmul_24 output must stay on CUDA (pre-fix: silent CPU return)"
    );
    let back = out.cpu().expect("gpu->cpu");
    let got = back.data().expect("data");
    for (i, (g, e)) in got.iter().zip(cpu_ref.iter()).enumerate() {
        // f64 epsilon-scale tolerance for a k=4 accumulation.
        assert!(
            (g - e).abs() < 1e-12 * (1.0 + e.abs()),
            "elem {i}: gpu={g} cpu={e}"
        );
    }
}

/// bf16 CUDA `a`: must stay on CUDA and compute through the GPU half/bfloat
/// matmul fallback when cuSPARSELt is unavailable or declines the shape.
#[test]
fn core076_gpu_bf16_stays_on_cuda() {
    ensure_cuda_backend();
    use half::bf16;

    let a_data: Vec<bf16> = (1..=8).map(|x| bf16::from_f32(x as f32 * 0.25)).collect();
    let b_data: Vec<bf16> = vec![
        1.0, 4.0, 2.0, 3.0, //
        -5.0, 2.0, 0.0, 1.0, //
        0.5, -0.25, 8.0, 7.0, //
        3.0, 3.0, -3.0, 0.125,
    ]
    .into_iter()
    .map(bf16::from_f32)
    .collect();
    let a_cpu = Tensor::<bf16>::from_storage(TensorStorage::cpu(a_data.clone()), vec![2, 4], false)
        .unwrap();
    let b_dense =
        Tensor::<bf16>::from_storage(TensorStorage::cpu(b_data), vec![4, 4], false).unwrap();
    let b = SemiStructuredSparseTensor::compress(&b_dense).unwrap();

    let b_decompressed = b.decompress().expect("decompress");
    let b_ref: Vec<f32> = b_decompressed
        .data()
        .expect("b data")
        .iter()
        .map(|v| v.to_f32())
        .collect();
    let a_ref: Vec<f32> = a_data.iter().map(|v| v.to_f32()).collect();
    let mut expected = [0.0f32; 2 * 4];
    for i in 0..2 {
        for j in 0..4 {
            let mut acc = 0.0f32;
            for kk in 0..4 {
                acc += a_ref[i * 4 + kk] * b_ref[kk * 4 + j];
            }
            expected[i * 4 + j] = bf16::from_f32(acc).to_f32();
        }
    }

    let a_gpu = a_cpu.to(Device::Cuda(0)).expect("a->cuda");
    let out = sparse_matmul_24(&a_gpu, &b).expect("cuda bf16 sparse_matmul_24");
    assert!(
        out.is_cuda(),
        "bf16 sparse_matmul_24 output must stay on CUDA"
    );
    let back = out.cpu().expect("gpu->cpu");
    let got: Vec<f32> = back
        .data()
        .expect("data")
        .iter()
        .map(|v| v.to_f32())
        .collect();
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!((g - e).abs() <= 0.02, "bf16 elem {i}: gpu={g} expected={e}");
    }
}

/// f16 CUDA `a`: same contract as bf16, with IEEE half storage tagged
/// separately from bfloat despite the shared u16 buffer representation.
#[test]
fn core076_gpu_f16_stays_on_cuda() {
    ensure_cuda_backend();
    use half::f16;

    let a_data: Vec<f16> = (1..=8).map(|x| f16::from_f32(x as f32 * 0.25)).collect();
    let b_data: Vec<f16> = vec![
        1.0, 4.0, 2.0, 3.0, //
        -5.0, 2.0, 0.0, 1.0, //
        0.5, -0.25, 8.0, 7.0, //
        3.0, 3.0, -3.0, 0.125,
    ]
    .into_iter()
    .map(f16::from_f32)
    .collect();
    let a_cpu =
        Tensor::<f16>::from_storage(TensorStorage::cpu(a_data.clone()), vec![2, 4], false).unwrap();
    let b_dense =
        Tensor::<f16>::from_storage(TensorStorage::cpu(b_data), vec![4, 4], false).unwrap();
    let b = SemiStructuredSparseTensor::compress(&b_dense).unwrap();

    let b_decompressed = b.decompress().expect("decompress");
    let b_ref: Vec<f32> = b_decompressed
        .data()
        .expect("b data")
        .iter()
        .map(|v| v.to_f32())
        .collect();
    let a_ref: Vec<f32> = a_data.iter().map(|v| v.to_f32()).collect();
    let mut expected = [0.0f32; 2 * 4];
    for i in 0..2 {
        for j in 0..4 {
            let mut acc = 0.0f32;
            for kk in 0..4 {
                acc += a_ref[i * 4 + kk] * b_ref[kk * 4 + j];
            }
            expected[i * 4 + j] = f16::from_f32(acc).to_f32();
        }
    }

    let a_gpu = a_cpu.to(Device::Cuda(0)).expect("a->cuda");
    let out = sparse_matmul_24(&a_gpu, &b).expect("cuda f16 sparse_matmul_24");
    assert!(
        out.is_cuda(),
        "f16 sparse_matmul_24 output must stay on CUDA"
    );
    let back = out.cpu().expect("gpu->cpu");
    let got: Vec<f32> = back
        .data()
        .expect("data")
        .iter()
        .map(|v| v.to_f32())
        .collect();
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!((g - e).abs() <= 0.002, "f16 elem {i}: gpu={g} expected={e}");
    }
}
