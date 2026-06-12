//! Critic re-audit GPU probes for the CORE-055 (#1749) cat fix: device
//! validation ordering (CPU-first vs CUDA-first) and CUDA repeat-zero forward.
//! Requires `--features gpu` and a CUDA device (cuda:0).
//!
//! Oracle (R-ORACLE-1b): live torch 2.11.0+cu130 on cuda:0, 2026-06-11.
//!
//! ```python
//! import torch
//! a = torch.arange(6.).reshape(2,3)
//! g = torch.arange(6., device='cuda:0').reshape(2,3)
//! torch.cat([a, g], 0)   # RuntimeError: Expected all tensors on same device
//! torch.cat([g, a], 0)   # RuntimeError (CUDA-first order also rejected)
//! # CUDA zero-count repeat collapses the axis on-device:
//! torch.arange(6., device='cuda:0').reshape(2,3).repeat(0,3).shape  # (0, 9)
//! ```
#![cfg(feature = "gpu")]

use ferrotorch_core::cat;
use ferrotorch_core::error::FerrotorchError;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;
use ferrotorch_core::{Device, creation::from_vec};

fn ensure_cuda_backend() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        ferrotorch_gpu::init_cuda_backend().expect("CUDA backend must initialise");
    });
}

fn arange(n: usize) -> Vec<f32> {
    (0..n).map(|v| v as f32).collect()
}

fn cpu(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

fn cuda(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    from_vec(data.to_vec(), shape)
        .expect("construct cpu tensor")
        .to(Device::Cuda(0))
        .expect("H2D")
}

/// P-GPU-1 — CPU tensor first, CUDA tensor second: torch raises; ferrotorch
/// must reject with DeviceMismatch BEFORE touching any GPU handle.
#[test]
fn pgpu1_cat_cpu_then_cuda_rejects() {
    ensure_cuda_backend();
    let a = cpu(&arange(6), &[2, 3]);
    let g = cuda(&arange(6), &[2, 3]);
    let err = cat(&[a, g], 0).expect_err("cat([cpu, cuda]) must error");
    assert!(
        matches!(err, FerrotorchError::DeviceMismatch { .. }),
        "expected DeviceMismatch, got {err:?}"
    );
}

/// P-GPU-2 — CUDA tensor first, CPU tensor second: torch raises in this order
/// too. The validation walks tensors[1..] against tensors[0], so a foreign
/// CPU tensor in slot 1 must be caught.
#[test]
fn pgpu2_cat_cuda_then_cpu_rejects() {
    ensure_cuda_backend();
    let g = cuda(&arange(6), &[2, 3]);
    let a = cpu(&arange(6), &[2, 3]);
    let err = cat(&[g, a], 0).expect_err("cat([cuda, cpu]) must error");
    assert!(
        matches!(err, FerrotorchError::DeviceMismatch { .. }),
        "expected DeviceMismatch, got {err:?}"
    );
}

/// P-GPU-3 — CUDA zero-count repeat (zero axis followed by a >=2 axis): the
/// narrow+cat composition must stay on-device and yield (0, 9).
/// torch: (2,3 cuda).repeat(0,3).shape == (0, 9).
#[test]
fn pgpu3_cuda_repeat_zero_then_positive() {
    ensure_cuda_backend();
    let x = cuda(&arange(6), &[2, 3]);
    let out = x
        .repeat_t(&[0, 3])
        .expect("cuda repeat([0,3]) must succeed");
    assert_eq!(out.shape(), &[0, 9], "cuda (2,3).repeat(0,3)");
    assert_eq!(out.numel(), 0);
    assert_eq!(out.device(), Device::Cuda(0), "result stays on cuda:0");
}

/// P-GPU-4 — CUDA cat mixing zero-numel and non-empty inputs. The
/// `strided_cat` loop must handle a `t_numel == 0` chunk (skip, offset
/// unchanged) and place the non-empty chunk correctly.
/// torch: cat([empty(0,3), (2,3), empty(0,3)],0) == (2,3) [0..5].
#[test]
fn pgpu4_cuda_cat_zero_numel_mixed() {
    ensure_cuda_backend();
    let z = cuda(&[], &[0, 3]);
    let m = cuda(&arange(6), &[2, 3]);
    let out = cat(&[z.clone(), m, z], 0).expect("cuda cat with empty inputs must succeed");
    assert_eq!(out.shape(), &[2, 3], "shape");
    assert_eq!(out.device(), Device::Cuda(0));
    let host = out.data_vec().expect("readback");
    assert_eq!(host, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], "data");
}
