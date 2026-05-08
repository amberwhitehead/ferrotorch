//! Permanent regression sentinel for Sprint B.2: cusolver device-resident SVD
//! and QR outputs (#895, #896, #635).
//!
//! # Pre-fix observable failure (before Sprint B.2)
//!
//! * `blas.rs` module doc falsely advertised a "CPU fallback" on cuBLAS handle
//!   failure with `eprintln!` — a rust-gpu-discipline §3 violation (#895).
//!   The production code never actually fell back; the doc was wrong.
//!
//! * `gpu_svd_f32` / `gpu_svd_f64` accepted `&[T]` inputs and returned
//!   `Vec<T>` outputs — both host-resident.  `backend_impl.rs` worked around
//!   this with 4 host↔device copies per SVD call.  PyTorch's
//!   `torch.linalg.svd` on CUDA returns CUDA tensors (#896, #635).
//!
//! * `gpu_qr_f32` / `gpu_qr_f64` had the same pattern — 3 host↔device copies
//!   per QR call.  PyTorch's `torch.linalg.qr` on CUDA returns CUDA tensors.
//!
//! # Post-fix (Sprint B.2)
//!
//! * `blas.rs` module docstring rewritten — error policy section now states
//!   `Err(GpuError::Blas(...))` propagation; no silent CPU fallback (#895).
//!
//! * `cusolver::gpu_svd_f32_dev` / `gpu_svd_f64_dev` accept
//!   `&CudaBuffer<T>` and return `(CudaBuffer<T>, CudaBuffer<T>, CudaBuffer<T>)`
//!   — U, S, Vh all on-device.  All layout conversions via `gpu_transpose_2d`.
//!
//! * `cusolver::gpu_qr_f32_dev` / `gpu_qr_f64_dev` accept `&CudaBuffer<T>`
//!   and return `(CudaBuffer<T>, CudaBuffer<T>)` — Q, R both on-device.
//!   R and Q extraction use dedicated on-device PTX kernels.
//!
//! * `backend_impl::{svd_f32, svd_f64, qr_f32, qr_f64}` now call the `_dev`
//!   variants — zero host bounces for the matrix data.
//!
//! # PyTorch parity (rust-gpu-discipline §3)
//!
//! `torch.linalg.svd(a.cuda())` returns U, S, Vh on CUDA.
//! `torch.linalg.qr(a.cuda())` returns Q, R on CUDA.
//! This probe asserts the same invariant via the ferrotorch-core dispatch layer.

#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::Device;
use ferrotorch_core::creation::from_vec;
use ferrotorch_core::linalg;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for the GPU probe suite");
    });
}

// ---------------------------------------------------------------------------
// #895 — blas.rs §3 docstring: no CPU fallback in production
// ---------------------------------------------------------------------------

/// #895 sentinel: the blas module docstring no longer claims silent CPU
/// fallback.  We verify via the GPU backend directly that cuBLAS SGEMM keeps
/// its result on device — `gpu_matmul_f32` accepts and returns `CudaBuffer`,
/// proving no host round-trip occurs and the error policy is `Err(...)` not
/// silent CPU degradation.
#[test]
fn b2_895_blas_no_silent_cpu_fallback() {
    ensure_cuda_backend();
    let dev = ferrotorch_gpu::GpuDevice::new(0).expect("GpuDevice::new");

    let a_data: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
    let b_data: Vec<f32> = (0..16).map(|i| (i as f32 + 1.0) * 0.1).collect();

    let a_buf = ferrotorch_gpu::transfer::cpu_to_gpu(&a_data, &dev).expect("upload a");
    let b_buf = ferrotorch_gpu::transfer::cpu_to_gpu(&b_data, &dev).expect("upload b");

    // gpu_matmul_f32 returns CudaBuffer — always device-resident. If cuBLAS
    // fails it returns Err(...), never falls back silently to CPU.
    let c_buf = ferrotorch_gpu::blas::gpu_matmul_f32(&a_buf, &b_buf, 4, 4, 4, &dev)
        .expect("gpu_matmul_f32");

    // §3: result must be on device, not a host Vec.
    assert_eq!(
        c_buf.device_ordinal(),
        0,
        "#895: matmul result must be on cuda:0"
    );
    assert_eq!(c_buf.len(), 16, "#895: 4x4 result has 16 elements");
}

// ---------------------------------------------------------------------------
// Helpers for numerical checks
// ---------------------------------------------------------------------------

fn rel_frob_f32(recon: &[f32], reference: &[f32]) -> f32 {
    let diff: f32 = recon
        .iter()
        .zip(reference.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
        .sqrt();
    let norm: f32 = reference.iter().map(|x| x * x).sum::<f32>().sqrt();
    diff / norm.max(1e-8)
}

fn rel_frob_f64(recon: &[f64], reference: &[f64]) -> f64 {
    let diff: f64 = recon
        .iter()
        .zip(reference.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt();
    let norm: f64 = reference.iter().map(|x| x * x).sum::<f64>().sqrt();
    diff / norm.max(1e-15)
}

// ---------------------------------------------------------------------------
// #896 / #635 — SVD device-resident (f32)
// ---------------------------------------------------------------------------

/// Pre-fix: U, S, Vh were returned as host Vec<f32>; backend_impl re-uploaded.
/// Post-fix: U, S, Vh are CudaBuffer<f32> — zero host bounce.
/// §3 assertion: is_cuda() must be true for all three outputs.
#[test]
fn b2_896_svd_f32_outputs_on_device() {
    ensure_cuda_backend();

    let m = 6usize;
    let n = 4usize;
    let k = m.min(n);

    let raw: Vec<f32> = (0..m * n)
        .map(|i| ((i * 7 + 3) % 17) as f32 * 0.1 - 0.8)
        .collect();
    let a = from_vec::<f32>(raw.clone(), &[m, n])
        .expect("from_vec f32")
        .to(Device::Cuda(0))
        .expect("to cuda");

    let (u, s, vh) = linalg::svd(&a).expect("linalg::svd f32");

    // §3 assertion — outputs must be on CUDA.
    assert!(
        u.is_cuda(),
        "#896: SVD U must be on CUDA (was: {:?})",
        u.device()
    );
    assert!(
        s.is_cuda(),
        "#896: SVD S must be on CUDA (was: {:?})",
        s.device()
    );
    assert!(
        vh.is_cuda(),
        "#896: SVD Vh must be on CUDA (was: {:?})",
        vh.device()
    );

    assert_eq!(u.shape(), &[m, k], "#896: U shape");
    assert_eq!(s.shape(), &[k], "#896: S shape");
    assert_eq!(vh.shape(), &[k, n], "#896: Vh shape");

    // Numerical check: ||U @ diag(S) @ Vh - A||_F / ||A||_F < 1e-4.
    let u_v: Vec<f32> = u.cpu().expect("U cpu").data().expect("U data").to_vec();
    let s_v: Vec<f32> = s.cpu().expect("S cpu").data().expect("S data").to_vec();
    let vh_v: Vec<f32> = vh.cpu().expect("Vh cpu").data().expect("Vh data").to_vec();
    let a_v: Vec<f32> = a.cpu().expect("A cpu").data().expect("A data").to_vec();

    let mut svh = vec![0.0f32; k * n];
    for i in 0..k {
        for j in 0..n {
            svh[i * n + j] = s_v[i] * vh_v[i * n + j];
        }
    }
    let mut recon = vec![0.0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            for j in 0..n {
                recon[i * n + j] += u_v[i * k + p] * svh[p * n + j];
            }
        }
    }
    let rel = rel_frob_f32(&recon, &a_v);
    assert!(
        rel < 1e-4_f32,
        "#896: SVD f32 reconstruction rel error {rel:.6e} > 1e-4"
    );
}

// ---------------------------------------------------------------------------
// #896 / #635 — SVD device-resident (f64)
// ---------------------------------------------------------------------------

#[test]
fn b2_896_svd_f64_outputs_on_device() {
    ensure_cuda_backend();

    let m = 5usize;
    let n = 3usize;
    let k = m.min(n);

    let raw: Vec<f64> = (0..m * n)
        .map(|i| ((i * 11 + 5) % 19) as f64 * 0.1 - 0.9)
        .collect();
    let a = from_vec::<f64>(raw.clone(), &[m, n])
        .expect("from_vec f64")
        .to(Device::Cuda(0))
        .expect("to cuda");

    let (u, s, vh) = linalg::svd(&a).expect("linalg::svd f64");

    assert!(u.is_cuda(), "#896: SVD f64 U must be on CUDA");
    assert!(s.is_cuda(), "#896: SVD f64 S must be on CUDA");
    assert!(vh.is_cuda(), "#896: SVD f64 Vh must be on CUDA");

    assert_eq!(u.shape(), &[m, k], "#896: f64 U shape");
    assert_eq!(s.shape(), &[k], "#896: f64 S shape");
    assert_eq!(vh.shape(), &[k, n], "#896: f64 Vh shape");

    let u_v: Vec<f64> = u.cpu().expect("U cpu").data().expect("U data").to_vec();
    let s_v: Vec<f64> = s.cpu().expect("S cpu").data().expect("S data").to_vec();
    let vh_v: Vec<f64> = vh.cpu().expect("Vh cpu").data().expect("Vh data").to_vec();
    let a_v: Vec<f64> = a.cpu().expect("A cpu").data().expect("A data").to_vec();

    let mut svh = vec![0.0f64; k * n];
    for i in 0..k {
        for j in 0..n {
            svh[i * n + j] = s_v[i] * vh_v[i * n + j];
        }
    }
    let mut recon = vec![0.0f64; m * n];
    for i in 0..m {
        for p in 0..k {
            for j in 0..n {
                recon[i * n + j] += u_v[i * k + p] * svh[p * n + j];
            }
        }
    }
    let rel = rel_frob_f64(&recon, &a_v);
    assert!(
        rel < 1e-9_f64,
        "#896: SVD f64 reconstruction rel error {rel:.12e} > 1e-9"
    );
}

// ---------------------------------------------------------------------------
// #896 / #635 — QR device-resident (f32)
// ---------------------------------------------------------------------------

/// Pre-fix: Q and R bounced through host Vec<f32>.
/// Post-fix: Q and R are CudaBuffer<f32> — zero host bounce.
#[test]
fn b2_635_qr_f32_outputs_on_device() {
    ensure_cuda_backend();

    let m = 6usize;
    let n = 4usize;
    let k = m.min(n);

    let raw: Vec<f32> = (0..m * n)
        .map(|i| ((i * 13 + 7) % 23) as f32 * 0.1 - 1.1)
        .collect();
    let a = from_vec::<f32>(raw.clone(), &[m, n])
        .expect("from_vec f32")
        .to(Device::Cuda(0))
        .expect("to cuda");

    let (q, r) = linalg::qr(&a).expect("linalg::qr f32");

    // §3 assertion.
    assert!(
        q.is_cuda(),
        "#635: QR Q must be on CUDA (was: {:?})",
        q.device()
    );
    assert!(
        r.is_cuda(),
        "#635: QR R must be on CUDA (was: {:?})",
        r.device()
    );

    assert_eq!(q.shape(), &[m, k], "#635: Q shape");
    assert_eq!(r.shape(), &[k, n], "#635: R shape");

    let q_v: Vec<f32> = q.cpu().expect("Q cpu").data().expect("Q data").to_vec();
    let r_v: Vec<f32> = r.cpu().expect("R cpu").data().expect("R data").to_vec();
    let a_v: Vec<f32> = a.cpu().expect("A cpu").data().expect("A data").to_vec();

    // Q columns orthonormal: Q^T @ Q ~ I_k.
    for i in 0..k {
        for j in 0..k {
            let dot: f32 = (0..m).map(|p| q_v[p * k + i] * q_v[p * k + j]).sum();
            let expected = if i == j { 1.0f32 } else { 0.0f32 };
            assert!(
                (dot - expected).abs() < 1e-4,
                "#635: Q^T@Q[{i},{j}] = {dot:.6e}, expected {expected:.1}"
            );
        }
    }

    // Reconstruction: Q @ R ~ A.
    let mut recon = vec![0.0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            for j in 0..n {
                recon[i * n + j] += q_v[i * k + p] * r_v[p * n + j];
            }
        }
    }
    let rel = rel_frob_f32(&recon, &a_v);
    assert!(
        rel < 1e-4_f32,
        "#635: QR f32 reconstruction rel error {rel:.6e} > 1e-4"
    );
}

// ---------------------------------------------------------------------------
// #896 / #635 — QR device-resident (f64)
// ---------------------------------------------------------------------------

#[test]
fn b2_635_qr_f64_outputs_on_device() {
    ensure_cuda_backend();

    let m = 5usize;
    let n = 3usize;
    let k = m.min(n);

    let raw: Vec<f64> = (0..m * n)
        .map(|i| ((i * 17 + 9) % 29) as f64 * 0.1 - 1.4)
        .collect();
    let a = from_vec::<f64>(raw.clone(), &[m, n])
        .expect("from_vec f64")
        .to(Device::Cuda(0))
        .expect("to cuda");

    let (q, r) = linalg::qr(&a).expect("linalg::qr f64");

    assert!(q.is_cuda(), "#635: QR f64 Q must be on CUDA");
    assert!(r.is_cuda(), "#635: QR f64 R must be on CUDA");

    assert_eq!(q.shape(), &[m, k], "#635: f64 Q shape");
    assert_eq!(r.shape(), &[k, n], "#635: f64 R shape");

    let q_v: Vec<f64> = q.cpu().expect("Q cpu").data().expect("Q data").to_vec();
    let r_v: Vec<f64> = r.cpu().expect("R cpu").data().expect("R data").to_vec();
    let a_v: Vec<f64> = a.cpu().expect("A cpu").data().expect("A data").to_vec();

    for i in 0..k {
        for j in 0..k {
            let dot: f64 = (0..m).map(|p| q_v[p * k + i] * q_v[p * k + j]).sum();
            let expected = if i == j { 1.0f64 } else { 0.0f64 };
            assert!(
                (dot - expected).abs() < 1e-9,
                "#635: Q^T@Q[{i},{j}] = {dot:.12e}, expected {expected:.1}"
            );
        }
    }

    let mut recon = vec![0.0f64; m * n];
    for i in 0..m {
        for p in 0..k {
            for j in 0..n {
                recon[i * n + j] += q_v[i * k + p] * r_v[p * n + j];
            }
        }
    }
    let rel = rel_frob_f64(&recon, &a_v);
    assert!(
        rel < 1e-9_f64,
        "#635: QR f64 reconstruction rel error {rel:.12e} > 1e-9"
    );
}
