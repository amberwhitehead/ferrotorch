//! Permanent regression sentinel for #816 / #817 / #818: GPU vector-matmul
//! kernels — `dot` (1D x 1D), `mv` (2D x 1D), `vm` (1D x 2D).
//!
//! Pre-fix: each of these shapes routes to `dot_differentiable` /
//! `mv_differentiable` (which call `.data()?` on a CUDA tensor) or to the
//! generic `linalg::matmul` (CPU-only) — surfacing as
//! `Err(GpuTensorNotAccessible)` for CUDA inputs. PyTorch supports all three
//! on CUDA for both f32 and f64, so the parity-violation gate is the absence
//! of a GPU dispatch.
//!
//! Post-fix (this probe):
//! - All 3 shapes × 2 dtypes route through the new `dot_f{32,64}` /
//!   `mv_f{32,64}` / `vm_f{32,64}` GPU backend methods.
//! - Result `is_cuda()` is true (the data stays on device — no host detour).
//! - Values match the CPU reference within the workspace
//!   `F32_MATMUL_GPU = 1e-3` / `F64_MATMUL_GPU = 1e-9` gate.
//! - Edge cases: empty (length 0), small (length 4), non-aligned (length 7),
//!   larger (length 1024).
//!
//! The tolerance constants are inlined here because the workspace-wide
//! constants live as private items inside `tests/conformance_linalg.rs`.

#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::Device;
use ferrotorch_core::Tensor;
use ferrotorch_core::creation::from_vec;
use ferrotorch_core::grad_fns::linalg::{
    dot_differentiable, matmul_differentiable, mv_differentiable,
};

/// Mirrors `tests/conformance_linalg.rs::tolerance::F32_MATMUL_GPU`.
const F32_MATMUL_GPU: f32 = 1e-3;
/// Mirrors `tests/conformance_linalg.rs::tolerance::F64_MATMUL_GPU`.
const F64_MATMUL_GPU: f64 = 1e-9;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for the GPU probe suite");
    });
}

fn read_back_f32(t: &Tensor<f32>) -> Vec<f32> {
    let cpu = if t.is_cuda() {
        t.cpu().expect("gpu->cpu copy")
    } else {
        t.clone()
    };
    cpu.data().expect("read_back").to_vec()
}

fn read_back_f64(t: &Tensor<f64>) -> Vec<f64> {
    let cpu = if t.is_cuda() {
        t.cpu().expect("gpu->cpu copy")
    } else {
        t.clone()
    };
    cpu.data().expect("read_back").to_vec()
}

// Deterministic value generators so failures are reproducible.
fn vec_f32(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i as u32).wrapping_mul(2654435761).wrapping_add(seed);
            // Map to [-1.0, 1.0]
            ((x as f32) / (u32::MAX as f32)) * 2.0 - 1.0
        })
        .collect()
}

fn vec_f64(n: usize, seed: u32) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let x = (i as u32).wrapping_mul(2654435761).wrapping_add(seed);
            ((x as f64) / (u32::MAX as f64)) * 2.0 - 1.0
        })
        .collect()
}

// CPU reference dot/mv/vm using f64 accumulation for both f32 and f64
// inputs to prevent the reference from being itself a noisy approximation.
fn cpu_dot_ref_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| x as f64 * y as f64)
        .sum::<f64>() as f32
}
fn cpu_dot_ref_f64(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum::<f64>()
}
fn cpu_mv_ref_f32(a: &[f32], x: &[f32], m: usize, k: usize) -> Vec<f32> {
    (0..m)
        .map(|i| {
            (0..k)
                .map(|j| a[i * k + j] as f64 * x[j] as f64)
                .sum::<f64>() as f32
        })
        .collect()
}
fn cpu_mv_ref_f64(a: &[f64], x: &[f64], m: usize, k: usize) -> Vec<f64> {
    (0..m)
        .map(|i| (0..k).map(|j| a[i * k + j] * x[j]).sum::<f64>())
        .collect()
}
fn cpu_vm_ref_f32(x: &[f32], b: &[f32], k: usize, n: usize) -> Vec<f32> {
    (0..n)
        .map(|j| {
            (0..k)
                .map(|i| x[i] as f64 * b[i * n + j] as f64)
                .sum::<f64>() as f32
        })
        .collect()
}
fn cpu_vm_ref_f64(x: &[f64], b: &[f64], k: usize, n: usize) -> Vec<f64> {
    (0..n)
        .map(|j| (0..k).map(|i| x[i] * b[i * n + j]).sum::<f64>())
        .collect()
}

const SIZES: &[usize] = &[0, 4, 7, 1024];

// ---------------------------------------------------------------------------
// dot — 1D x 1D
// ---------------------------------------------------------------------------

#[test]
fn gpu_dot_f32_matches_cpu() {
    ensure_cuda_backend();
    for &n in SIZES {
        let a_vals = vec_f32(n, 0x100);
        let b_vals = vec_f32(n, 0x200);
        let a_cpu = from_vec::<f32>(a_vals.clone(), &[n]).expect("cpu a");
        let b_cpu = from_vec::<f32>(b_vals.clone(), &[n]).expect("cpu b");
        let a_gpu = a_cpu.to(Device::Cuda(0)).expect("a gpu");
        let b_gpu = b_cpu.to(Device::Cuda(0)).expect("b gpu");

        // Direct dot path
        let y_dot = dot_differentiable(&a_gpu, &b_gpu).expect("gpu dot");
        // matmul-dispatch path (1D x 1D)
        let y_mm = matmul_differentiable(&a_gpu, &b_gpu).expect("matmul 1Dx1D");

        assert!(y_dot.is_cuda(), "n={n}: dot result must stay on GPU");
        assert!(y_mm.is_cuda(), "n={n}: matmul 1Dx1D result must stay on GPU");
        assert_eq!(y_dot.shape(), &[] as &[usize], "n={n}: dot returns scalar");

        let want = cpu_dot_ref_f32(&a_vals, &b_vals);
        let got_dot = read_back_f32(&y_dot);
        let got_mm = read_back_f32(&y_mm);
        assert_eq!(got_dot.len(), 1);
        assert_eq!(got_mm.len(), 1);
        let d_dot = (got_dot[0] - want).abs();
        let d_mm = (got_mm[0] - want).abs();
        assert!(
            d_dot <= F32_MATMUL_GPU,
            "dot_f32 n={n}: got={} want={want} diff={d_dot} gate={F32_MATMUL_GPU}",
            got_dot[0]
        );
        assert!(
            d_mm <= F32_MATMUL_GPU,
            "matmul 1Dx1D f32 n={n}: got={} want={want} diff={d_mm} gate={F32_MATMUL_GPU}",
            got_mm[0]
        );
    }
}

#[test]
fn gpu_dot_f64_matches_cpu() {
    ensure_cuda_backend();
    for &n in SIZES {
        let a_vals = vec_f64(n, 0x300);
        let b_vals = vec_f64(n, 0x400);
        let a_cpu = from_vec::<f64>(a_vals.clone(), &[n]).expect("cpu a");
        let b_cpu = from_vec::<f64>(b_vals.clone(), &[n]).expect("cpu b");
        let a_gpu = a_cpu.to(Device::Cuda(0)).expect("a gpu");
        let b_gpu = b_cpu.to(Device::Cuda(0)).expect("b gpu");

        let y_dot = dot_differentiable(&a_gpu, &b_gpu).expect("gpu dot f64");
        let y_mm = matmul_differentiable(&a_gpu, &b_gpu).expect("matmul 1Dx1D f64");

        assert!(y_dot.is_cuda(), "n={n}: dot f64 result must stay on GPU");
        assert!(y_mm.is_cuda(), "n={n}: matmul 1Dx1D f64 result must stay on GPU");

        let want = cpu_dot_ref_f64(&a_vals, &b_vals);
        let got_dot = read_back_f64(&y_dot);
        let got_mm = read_back_f64(&y_mm);
        let d_dot = (got_dot[0] - want).abs();
        let d_mm = (got_mm[0] - want).abs();
        // f64 dot accumulates n terms; allow a small per-term tolerance for
        // non-deterministic reduction order on the GPU.
        let gate = F64_MATMUL_GPU.max(1e-12 * (n as f64).max(1.0));
        assert!(
            d_dot <= gate,
            "dot_f64 n={n}: got={} want={want} diff={d_dot} gate={gate}",
            got_dot[0]
        );
        assert!(
            d_mm <= gate,
            "matmul 1Dx1D f64 n={n}: got={} want={want} diff={d_mm} gate={gate}",
            got_mm[0]
        );
    }
}

// ---------------------------------------------------------------------------
// mv — 2D x 1D
// ---------------------------------------------------------------------------

fn run_mv_f32_case(m: usize, k: usize) {
    let a_vals = vec_f32(m * k, 0x500);
    let x_vals = vec_f32(k, 0x600);
    let a_cpu = from_vec::<f32>(a_vals.clone(), &[m, k]).expect("cpu A");
    let x_cpu = from_vec::<f32>(x_vals.clone(), &[k]).expect("cpu x");
    let a_gpu = a_cpu.to(Device::Cuda(0)).expect("A gpu");
    let x_gpu = x_cpu.to(Device::Cuda(0)).expect("x gpu");

    let y_mv = mv_differentiable(&a_gpu, &x_gpu).expect("gpu mv");
    let y_mm = matmul_differentiable(&a_gpu, &x_gpu).expect("matmul 2Dx1D");

    assert!(y_mv.is_cuda(), "m={m} k={k}: mv result must stay on GPU");
    assert!(y_mm.is_cuda(), "m={m} k={k}: matmul 2Dx1D result must stay on GPU");
    assert_eq!(y_mv.shape(), &[m]);
    assert_eq!(y_mm.shape(), &[m]);

    let want = cpu_mv_ref_f32(&a_vals, &x_vals, m, k);
    let got_mv = read_back_f32(&y_mv);
    let got_mm = read_back_f32(&y_mm);
    for i in 0..m {
        let d_mv = (got_mv[i] - want[i]).abs();
        let d_mm = (got_mm[i] - want[i]).abs();
        assert!(
            d_mv <= F32_MATMUL_GPU,
            "mv_f32 m={m} k={k} i={i}: got={} want={} diff={d_mv} gate={F32_MATMUL_GPU}",
            got_mv[i], want[i]
        );
        assert!(
            d_mm <= F32_MATMUL_GPU,
            "matmul 2Dx1D f32 m={m} k={k} i={i}: got={} want={} diff={d_mm} gate={F32_MATMUL_GPU}",
            got_mm[i], want[i]
        );
    }
}

fn run_mv_f64_case(m: usize, k: usize) {
    let a_vals = vec_f64(m * k, 0x700);
    let x_vals = vec_f64(k, 0x800);
    let a_cpu = from_vec::<f64>(a_vals.clone(), &[m, k]).expect("cpu A");
    let x_cpu = from_vec::<f64>(x_vals.clone(), &[k]).expect("cpu x");
    let a_gpu = a_cpu.to(Device::Cuda(0)).expect("A gpu");
    let x_gpu = x_cpu.to(Device::Cuda(0)).expect("x gpu");

    let y_mv = mv_differentiable(&a_gpu, &x_gpu).expect("gpu mv f64");
    let y_mm = matmul_differentiable(&a_gpu, &x_gpu).expect("matmul 2Dx1D f64");

    assert!(y_mv.is_cuda());
    assert!(y_mm.is_cuda());
    let want = cpu_mv_ref_f64(&a_vals, &x_vals, m, k);
    let got_mv = read_back_f64(&y_mv);
    let got_mm = read_back_f64(&y_mm);
    let gate = F64_MATMUL_GPU.max(1e-12 * (k as f64).max(1.0));
    for i in 0..m {
        let d_mv = (got_mv[i] - want[i]).abs();
        let d_mm = (got_mm[i] - want[i]).abs();
        assert!(
            d_mv <= gate,
            "mv_f64 m={m} k={k} i={i}: diff={d_mv} gate={gate} got={} want={}",
            got_mv[i], want[i]
        );
        assert!(
            d_mm <= gate,
            "matmul 2Dx1D f64 m={m} k={k} i={i}: diff={d_mm} gate={gate} got={} want={}",
            got_mm[i], want[i]
        );
    }
}

#[test]
fn gpu_mv_f32_matches_cpu() {
    ensure_cuda_backend();
    // m and k vary independently; mix sizes to stress alignment/loop.
    let cases = [(0, 0), (3, 0), (0, 5), (4, 4), (3, 7), (7, 3), (1024, 64), (64, 1024)];
    for (m, k) in cases {
        run_mv_f32_case(m, k);
    }
}

#[test]
fn gpu_mv_f64_matches_cpu() {
    ensure_cuda_backend();
    let cases = [(0, 0), (3, 0), (0, 5), (4, 4), (3, 7), (7, 3), (1024, 64), (64, 1024)];
    for (m, k) in cases {
        run_mv_f64_case(m, k);
    }
}

// ---------------------------------------------------------------------------
// vm — 1D x 2D
// ---------------------------------------------------------------------------

fn run_vm_f32_case(k: usize, n: usize) {
    let x_vals = vec_f32(k, 0x900);
    let b_vals = vec_f32(k * n, 0xA00);
    let x_cpu = from_vec::<f32>(x_vals.clone(), &[k]).expect("cpu x");
    let b_cpu = from_vec::<f32>(b_vals.clone(), &[k, n]).expect("cpu B");
    let x_gpu = x_cpu.to(Device::Cuda(0)).expect("x gpu");
    let b_gpu = b_cpu.to(Device::Cuda(0)).expect("B gpu");

    let y_mm = matmul_differentiable(&x_gpu, &b_gpu).expect("matmul 1Dx2D");
    assert!(y_mm.is_cuda(), "k={k} n={n}: vm result must stay on GPU");
    assert_eq!(y_mm.shape(), &[n]);

    let want = cpu_vm_ref_f32(&x_vals, &b_vals, k, n);
    let got = read_back_f32(&y_mm);
    for j in 0..n {
        let d = (got[j] - want[j]).abs();
        assert!(
            d <= F32_MATMUL_GPU,
            "vm f32 k={k} n={n} j={j}: got={} want={} diff={d} gate={F32_MATMUL_GPU}",
            got[j], want[j]
        );
    }
}

fn run_vm_f64_case(k: usize, n: usize) {
    let x_vals = vec_f64(k, 0xB00);
    let b_vals = vec_f64(k * n, 0xC00);
    let x_cpu = from_vec::<f64>(x_vals.clone(), &[k]).expect("cpu x");
    let b_cpu = from_vec::<f64>(b_vals.clone(), &[k, n]).expect("cpu B");
    let x_gpu = x_cpu.to(Device::Cuda(0)).expect("x gpu");
    let b_gpu = b_cpu.to(Device::Cuda(0)).expect("B gpu");

    let y_mm = matmul_differentiable(&x_gpu, &b_gpu).expect("matmul 1Dx2D f64");
    assert!(y_mm.is_cuda());
    let want = cpu_vm_ref_f64(&x_vals, &b_vals, k, n);
    let got = read_back_f64(&y_mm);
    let gate = F64_MATMUL_GPU.max(1e-12 * (k as f64).max(1.0));
    for j in 0..n {
        let d = (got[j] - want[j]).abs();
        assert!(
            d <= gate,
            "vm f64 k={k} n={n} j={j}: diff={d} gate={gate}"
        );
    }
}

#[test]
fn gpu_vm_f32_matches_cpu() {
    ensure_cuda_backend();
    let cases = [(0, 0), (0, 5), (5, 0), (4, 4), (7, 3), (3, 7), (64, 1024), (1024, 64)];
    for (k, n) in cases {
        run_vm_f32_case(k, n);
    }
}

#[test]
fn gpu_vm_f64_matches_cpu() {
    ensure_cuda_backend();
    let cases = [(0, 0), (0, 5), (5, 0), (4, 4), (7, 3), (3, 7), (64, 1024), (1024, 64)];
    for (k, n) in cases {
        run_vm_f64_case(k, n);
    }
}
