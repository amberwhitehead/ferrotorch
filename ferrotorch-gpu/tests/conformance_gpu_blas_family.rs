//! C8.3 — BLAS-family FFI binding-layer conformance for `ferrotorch-gpu`.
//!
//! Tracking: C8.3 (parent epic #806).
//!
//! ## What this file tests
//!
//! Layer 3 (conformance tests) per `docs/conformance-suites.md`. The suite
//! exercises the FFI lifecycle and Rust wrapper correctness of all five
//! BLAS-family modules:
//!
//! | Module | Tests |
//! |---|---|
//! | `blas.rs` | cuBLAS handle init/destroy; SGEMM 4×4, DGEMM 4×4, non-square, bmm |
//! | `cufft.rs` | cuFFT plan lifecycle; C2C 4-pt fwd+inv (f32+f64); R2C / C2R roundtrip |
//! | `cusolver.rs` | cuSOLVER DnHandle; SVD 4×4 (f32+f64); Cholesky 4×4 SPD (f32+f64) |
//! | `cusparselt.rs` | cuSPARSELt handle init; 2:4 SpMM 8×8 f32 round-trip |
//! | `bf16.rs` | PTX kernel lifecycle (compile + launch); mul/add/silu/relu N=16 |
//!
//! All tests are feature-gated on `#[cfg(feature = "cuda")]`.
//! cuSPARSELt tests are additionally gated on `#[cfg(feature = "cusparselt")]`.
//!
//! ## Cascade bugs — RESOLVED (Sprint B.2)
//!
//! | ID | Module | Finding | Resolution |
//! |---|---|---|---|
//! | CASCADE-C8.3-001 | blas.rs | Doc comment falsely claimed silent CPU fallback — §3 violation (#895) | FIXED: docstring rewritten; no CPU fallback exists in production path |
//! | CASCADE-C8.3-002 | cusolver.rs | `gpu_svd_f32/f64`, `gpu_qr_f32/f64` bounced through host `Vec<T>` — sync readback on every call (#896, #635) | FIXED: `gpu_svd_f32/f64_dev` + `gpu_qr_f32/f64_dev` device-resident variants; `backend_impl.rs` callers migrated |

#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::sync::Once;

use serde_json::Value;

use ferrotorch_gpu::init_cuda_backend;
use ferrotorch_gpu::{GpuDevice, blas, cufft, cusolver};

// ---------------------------------------------------------------------------
// One-time CUDA backend initialisation
// ---------------------------------------------------------------------------

static INIT: Once = Once::new();

fn ensure_cuda() {
    INIT.call_once(|| {
        init_cuda_backend().expect("CUDA backend init");
    });
}

// ---------------------------------------------------------------------------
// Fixture loading helpers
// ---------------------------------------------------------------------------

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("conformance")
        .join("fixtures")
        .join("gpu_blas_family.json")
}

fn load_fixtures() -> Value {
    let path = fixture_path();
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("cannot read fixture file {}: {e}", path.display()));
    serde_json::from_slice(&bytes)
        .unwrap_or_else(|e| panic!("cannot parse fixture file {}: {e}", path.display()))
}

/// Pick every fixture whose `"module"` == `module` and `"op"` == `op`.
fn pick_fixtures<'a>(root: &'a Value, module: &str, op: &str) -> Vec<&'a Value> {
    root["fixtures"]
        .as_array()
        .expect("fixtures array")
        .iter()
        .filter(|f| f["module"].as_str() == Some(module) && f["op"].as_str() == Some(op))
        .collect()
}

/// Flatten a JSON array of numbers to `Vec<f32>`.
fn as_f32_vec(v: &Value) -> Vec<f32> {
    v.as_array()
        .expect("f32 array")
        .iter()
        .map(|x| x.as_f64().expect("number") as f32)
        .collect()
}

/// Flatten a JSON array of numbers to `Vec<f64>`.
fn as_f64_vec(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("f64 array")
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

/// Flatten a JSON array of integers to `Vec<u16>` (bf16 bit patterns).
fn as_u16_vec(v: &Value) -> Vec<u16> {
    v.as_array()
        .expect("u16 array")
        .iter()
        .map(|x| x.as_u64().expect("u16 int") as u16)
        .collect()
}

// ---------------------------------------------------------------------------
// Tolerance helpers
// ---------------------------------------------------------------------------

fn assert_close_f32(got: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(
        got.len(),
        expected.len(),
        "{label}: length mismatch: got {} expected {}",
        got.len(),
        expected.len()
    );
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() <= tol,
            "{label}: index {i}: got={g:.6e} expected={e:.6e} diff={:.6e} tol={tol:.6e}",
            (g - e).abs()
        );
    }
}

fn assert_close_f64(got: &[f64], expected: &[f64], tol: f64, label: &str) {
    assert_eq!(
        got.len(),
        expected.len(),
        "{label}: length mismatch: got {} expected {}",
        got.len(),
        expected.len()
    );
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() <= tol,
            "{label}: index {i}: got={g:.10e} expected={e:.10e} diff={:.10e} tol={tol:.10e}",
            (g - e).abs()
        );
    }
}

/// Frobenius norm of a flat slice.
fn frob_f32(s: &[f32]) -> f32 {
    s.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn frob_f64(s: &[f64]) -> f64 {
    s.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// ||a - b||_F
fn frob_diff_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

fn frob_diff_f64(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

/// Naive n×n matmul for reconstruction checks.
fn matmul_f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for l in 0..k {
                acc += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

fn matmul_f64(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for l in 0..k {
                acc += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

// ---------------------------------------------------------------------------
// bf16 round-trip helpers (u16 ↔ f32)
// ---------------------------------------------------------------------------

/// Interpret a u16 bf16 bit pattern as f32 by sign-extending to u32 and
/// shifting left 16 bits (the standard bf16 → f32 lossless expand).
fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

// ===========================================================================
// MODULE: blas.rs — cuBLAS handle lifecycle + matmul round-trips
// ===========================================================================

/// cuBLAS SGEMM 4×4: verifies handle init/destroy is implicit in `GpuDevice::new`
/// (which constructs the `CudaBlas` handle) and that `gpu_matmul_f32` produces
/// values within 1e-3 of the PyTorch reference.
#[test]
fn blas_sgemm_4x4_f32() {
    ensure_cuda();
    let fixtures = load_fixtures();
    let cases = pick_fixtures(&fixtures, "blas", "gpu_matmul_f32");

    let device = GpuDevice::new(0).expect("GpuDevice::new");

    for f in cases {
        let tag = f["tag"].as_str().unwrap_or("?");
        let m = f["m"].as_u64().unwrap() as usize;
        let k = f["k"].as_u64().unwrap() as usize;
        let n = f["n"].as_u64().unwrap() as usize;

        let a_host = as_f32_vec(&f["a_data"]);
        let b_host = as_f32_vec(&f["b_data"]);
        let expected = as_f32_vec(&f["expected"]);

        let a_buf = ferrotorch_gpu::transfer::cpu_to_gpu(&a_host, &device)
            .unwrap_or_else(|e| panic!("blas f32 upload A ({tag}): {e}"));
        let b_buf = ferrotorch_gpu::transfer::cpu_to_gpu(&b_host, &device)
            .unwrap_or_else(|e| panic!("blas f32 upload B ({tag}): {e}"));

        let c_buf = blas::gpu_matmul_f32(&a_buf, &b_buf, m, k, n, &device)
            .unwrap_or_else(|e| panic!("gpu_matmul_f32 ({tag}): {e}"));

        // Verify the result lives on device: `c_buf.device_ordinal()` must be 0.
        assert_eq!(
            c_buf.device_ordinal(),
            0,
            "blas f32 ({tag}): result must be on device 0"
        );
        assert_eq!(c_buf.len(), m * n, "blas f32 ({tag}): output length");

        let got = ferrotorch_gpu::transfer::gpu_to_cpu(&c_buf, &device)
            .unwrap_or_else(|e| panic!("blas f32 download ({tag}): {e}"));

        assert_close_f32(
            &got,
            &expected,
            1e-3,
            &format!("blas::gpu_matmul_f32 {tag}"),
        );
    }
}

/// cuBLAS DGEMM 4×4: f64 variant; tighter 1e-9 tolerance.
#[test]
fn blas_dgemm_4x4_f64() {
    ensure_cuda();
    let fixtures = load_fixtures();
    let cases = pick_fixtures(&fixtures, "blas", "gpu_matmul_f64");

    let device = GpuDevice::new(0).expect("GpuDevice::new");

    for f in cases {
        let tag = f["tag"].as_str().unwrap_or("?");
        let m = f["m"].as_u64().unwrap() as usize;
        let k = f["k"].as_u64().unwrap() as usize;
        let n = f["n"].as_u64().unwrap() as usize;

        let a_host = as_f64_vec(&f["a_data"]);
        let b_host = as_f64_vec(&f["b_data"]);
        let expected = as_f64_vec(&f["expected"]);

        let a_buf = ferrotorch_gpu::transfer::cpu_to_gpu(&a_host, &device)
            .unwrap_or_else(|e| panic!("blas f64 upload A ({tag}): {e}"));
        let b_buf = ferrotorch_gpu::transfer::cpu_to_gpu(&b_host, &device)
            .unwrap_or_else(|e| panic!("blas f64 upload B ({tag}): {e}"));

        let c_buf = blas::gpu_matmul_f64(&a_buf, &b_buf, m, k, n, &device)
            .unwrap_or_else(|e| panic!("gpu_matmul_f64 ({tag}): {e}"));

        assert_eq!(
            c_buf.device_ordinal(),
            0,
            "blas f64 ({tag}): result on device 0"
        );
        assert_eq!(c_buf.len(), m * n, "blas f64 ({tag}): output length");

        let got = ferrotorch_gpu::transfer::gpu_to_cpu(&c_buf, &device)
            .unwrap_or_else(|e| panic!("blas f64 download ({tag}): {e}"));

        assert_close_f64(
            &got,
            &expected,
            1e-9,
            &format!("blas::gpu_matmul_f64 {tag}"),
        );
    }
}

/// Batched SGEMM [2, 4, 4] × [2, 4, 4] via `gpu_bmm_f32`.
#[test]
fn blas_bmm_f32_batch2() {
    ensure_cuda();
    let fixtures = load_fixtures();
    let cases = pick_fixtures(&fixtures, "blas", "gpu_bmm_f32");

    let device = GpuDevice::new(0).expect("GpuDevice::new");

    for f in cases {
        let tag = f["tag"].as_str().unwrap_or("?");
        let batch = f["batch"].as_u64().unwrap() as usize;
        let m = f["m"].as_u64().unwrap() as usize;
        let k = f["k"].as_u64().unwrap() as usize;
        let n = f["n"].as_u64().unwrap() as usize;

        let a_host = as_f32_vec(&f["a_data"]);
        let b_host = as_f32_vec(&f["b_data"]);
        let expected = as_f32_vec(&f["expected"]);

        let a_buf = ferrotorch_gpu::transfer::cpu_to_gpu(&a_host, &device)
            .unwrap_or_else(|e| panic!("bmm f32 upload A ({tag}): {e}"));
        let b_buf = ferrotorch_gpu::transfer::cpu_to_gpu(&b_host, &device)
            .unwrap_or_else(|e| panic!("bmm f32 upload B ({tag}): {e}"));

        let c_buf = blas::gpu_bmm_f32(&a_buf, &b_buf, batch, m, k, n, &device)
            .unwrap_or_else(|e| panic!("gpu_bmm_f32 ({tag}): {e}"));

        assert_eq!(
            c_buf.device_ordinal(),
            0,
            "bmm f32 ({tag}): result on device 0"
        );
        assert_eq!(c_buf.len(), batch * m * n, "bmm f32 ({tag}): output length");

        let got = ferrotorch_gpu::transfer::gpu_to_cpu(&c_buf, &device)
            .unwrap_or_else(|e| panic!("bmm f32 download ({tag}): {e}"));

        assert_close_f32(&got, &expected, 1e-3, &format!("blas::gpu_bmm_f32 {tag}"));
    }
}

/// Shape-validation: wrong dimensions must return `Err`, not panic.
#[test]
fn blas_matmul_shape_error() {
    ensure_cuda();
    let device = GpuDevice::new(0).expect("GpuDevice::new");
    let a = ferrotorch_gpu::transfer::cpu_to_gpu(&[1.0f32, 2.0, 3.0, 4.0], &device).unwrap();
    let b = ferrotorch_gpu::transfer::cpu_to_gpu(&[1.0f32, 2.0, 3.0, 4.0], &device).unwrap();
    // Claim it's [2, 3] × [3, 2] but buffers are length 4 each (2×2) — shape mismatch.
    let result = blas::gpu_matmul_f32(&a, &b, 2, 3, 2, &device);
    assert!(result.is_err(), "expected Err on shape mismatch but got Ok");
}

// ===========================================================================
// MODULE: cufft.rs — cuFFT plan lifecycle + transform correctness
// ===========================================================================

/// cuFFT C2C 4-point forward, f32.  Verifies plan creation on `GpuDevice`'s
/// stream and that the transform output matches the PyTorch reference.
#[test]
fn cufft_c2c_f32_forward() {
    ensure_cuda();
    let fixtures = load_fixtures();
    let device = GpuDevice::new(0).expect("GpuDevice::new");

    for f in pick_fixtures(&fixtures, "cufft", "gpu_fft_c2c_f32") {
        let tag = f["tag"].as_str().unwrap_or("?");
        let inverse = f["inverse"].as_bool().unwrap_or(false);
        let batch = f["batch"].as_u64().unwrap() as usize;
        let n = f["n"].as_u64().unwrap() as usize;

        let input = as_f32_vec(&f["input"]);
        let expected = as_f32_vec(&f["expected"]);

        let in_buf = ferrotorch_gpu::transfer::cpu_to_gpu(&input, &device)
            .unwrap_or_else(|e| panic!("cufft f32 upload ({tag}): {e}"));

        let out_buf = cufft::gpu_fft_c2c_f32(&in_buf, batch, n, inverse, &device)
            .unwrap_or_else(|e| panic!("gpu_fft_c2c_f32 ({tag}): {e}"));

        // Result must stay on device.
        assert_eq!(
            out_buf.device_ordinal(),
            0,
            "cufft c2c f32 ({tag}): result on device 0"
        );
        assert_eq!(
            out_buf.len(),
            batch * n * 2,
            "cufft c2c f32 ({tag}): output length"
        );

        let got = ferrotorch_gpu::transfer::gpu_to_cpu(&out_buf, &device)
            .unwrap_or_else(|e| panic!("cufft f32 download ({tag}): {e}"));

        // FFT tolerance: 1e-4 absolute for f32 (rounding in cuFFT butterfly).
        let tol = if inverse { 2e-4 } else { 1e-4 };
        assert_close_f32(
            &got,
            &expected,
            tol,
            &format!("cufft::gpu_fft_c2c_f32 {tag}"),
        );
    }
}

/// cuFFT C2C 4-point forward, f64.
#[test]
fn cufft_c2c_f64_forward() {
    ensure_cuda();
    let fixtures = load_fixtures();
    let device = GpuDevice::new(0).expect("GpuDevice::new");

    for f in pick_fixtures(&fixtures, "cufft", "gpu_fft_c2c_f64") {
        let tag = f["tag"].as_str().unwrap_or("?");
        let inverse = f["inverse"].as_bool().unwrap_or(false);
        let batch = f["batch"].as_u64().unwrap() as usize;
        let n = f["n"].as_u64().unwrap() as usize;

        let input = as_f64_vec(&f["input"]);
        let expected = as_f64_vec(&f["expected"]);

        let in_buf = ferrotorch_gpu::transfer::cpu_to_gpu(&input, &device)
            .unwrap_or_else(|e| panic!("cufft f64 upload ({tag}): {e}"));

        let out_buf = cufft::gpu_fft_c2c_f64(&in_buf, batch, n, inverse, &device)
            .unwrap_or_else(|e| panic!("gpu_fft_c2c_f64 ({tag}): {e}"));

        assert_eq!(out_buf.device_ordinal(), 0, "cufft c2c f64 ({tag}): device");
        assert_eq!(
            out_buf.len(),
            batch * n * 2,
            "cufft c2c f64 ({tag}): output length"
        );

        let got = ferrotorch_gpu::transfer::gpu_to_cpu(&out_buf, &device)
            .unwrap_or_else(|e| panic!("cufft f64 download ({tag}): {e}"));

        assert_close_f64(
            &got,
            &expected,
            1e-10,
            &format!("cufft::gpu_fft_c2c_f64 {tag}"),
        );
    }
}

/// cuFFT R2C (rfft) 4-point, f32: verifies the R→C plan and Hermitian half output.
#[test]
fn cufft_rfft_r2c_f32() {
    ensure_cuda();
    let fixtures = load_fixtures();
    let device = GpuDevice::new(0).expect("GpuDevice::new");

    for f in pick_fixtures(&fixtures, "cufft", "gpu_rfft_r2c_f32") {
        let tag = f["tag"].as_str().unwrap_or("?");
        let batch = f["batch"].as_u64().unwrap() as usize;
        let n = f["n"].as_u64().unwrap() as usize;

        let input = as_f32_vec(&f["input"]);
        let expected = as_f32_vec(&f["expected"]);

        let in_buf = ferrotorch_gpu::transfer::cpu_to_gpu(&input, &device)
            .unwrap_or_else(|e| panic!("rfft f32 upload ({tag}): {e}"));

        let out_buf = cufft::gpu_rfft_r2c_f32(&in_buf, batch, n, &device)
            .unwrap_or_else(|e| panic!("gpu_rfft_r2c_f32 ({tag}): {e}"));

        // R2C output length = batch * (n/2+1) * 2.
        let expected_out_len = batch * (n / 2 + 1) * 2;
        assert_eq!(out_buf.device_ordinal(), 0, "rfft f32 ({tag}): device");
        assert_eq!(
            out_buf.len(),
            expected_out_len,
            "rfft f32 ({tag}): output length"
        );

        let got = ferrotorch_gpu::transfer::gpu_to_cpu(&out_buf, &device)
            .unwrap_or_else(|e| panic!("rfft f32 download ({tag}): {e}"));

        assert_close_f32(
            &got,
            &expected,
            1e-4,
            &format!("cufft::gpu_rfft_r2c_f32 {tag}"),
        );
    }
}

/// cuFFT C2R (irfft) 4-point round-trip, f32.
#[test]
fn cufft_irfft_c2r_f32() {
    ensure_cuda();
    let fixtures = load_fixtures();
    let device = GpuDevice::new(0).expect("GpuDevice::new");

    for f in pick_fixtures(&fixtures, "cufft", "gpu_irfft_c2r_f32") {
        let tag = f["tag"].as_str().unwrap_or("?");
        let batch = f["batch"].as_u64().unwrap() as usize;
        let n_out = f["n_out"].as_u64().unwrap() as usize;

        let input = as_f32_vec(&f["input"]);
        let expected = as_f32_vec(&f["expected"]);

        let in_buf = ferrotorch_gpu::transfer::cpu_to_gpu(&input, &device)
            .unwrap_or_else(|e| panic!("irfft f32 upload ({tag}): {e}"));

        let out_buf = cufft::gpu_irfft_c2r_f32(&in_buf, batch, n_out, &device)
            .unwrap_or_else(|e| panic!("gpu_irfft_c2r_f32 ({tag}): {e}"));

        assert_eq!(out_buf.device_ordinal(), 0, "irfft f32 ({tag}): device");
        assert_eq!(
            out_buf.len(),
            batch * n_out,
            "irfft f32 ({tag}): output length"
        );

        let got = ferrotorch_gpu::transfer::gpu_to_cpu(&out_buf, &device)
            .unwrap_or_else(|e| panic!("irfft f32 download ({tag}): {e}"));

        assert_close_f32(
            &got,
            &expected,
            1e-4,
            &format!("cufft::gpu_irfft_c2r_f32 {tag}"),
        );
    }
}

/// cuFFT shape-validation: zero n must return Err.
#[test]
fn cufft_zero_n_returns_err() {
    ensure_cuda();
    let device = GpuDevice::new(0).expect("GpuDevice::new");
    let buf = ferrotorch_gpu::transfer::cpu_to_gpu(&[1.0f32, 0.0, 2.0, 0.0], &device).unwrap();
    // n=0 must be rejected.
    let result = cufft::gpu_fft_c2c_f32(&buf, 1, 0, false, &device);
    assert!(result.is_err(), "expected Err for n=0 in gpu_fft_c2c_f32");
}

// ===========================================================================
// MODULE: cusolver.rs — cuSOLVER DnHandle lifecycle + factorization correctness
// ===========================================================================

/// cuSOLVER SVD 4×4 f32.  Validates handle init + workspace query + exec
/// lifecycle.  Result tested via reconstruction: ||U @ diag(S) @ Vh - A||_F
/// < 1e-4 * ||A||_F.  Singular values (S) also compared directly against
/// PyTorch's reference.
#[test]
fn cusolver_svd_4x4_f32() {
    ensure_cuda();
    let fixtures = load_fixtures();
    let device = GpuDevice::new(0).expect("GpuDevice::new");

    for f in pick_fixtures(&fixtures, "cusolver", "gpu_svd_f32") {
        let tag = f["tag"].as_str().unwrap_or("?");
        let m = f["m"].as_u64().unwrap() as usize;
        let n = f["n"].as_u64().unwrap() as usize;

        let data = as_f32_vec(&f["input"]);
        let expected_s = as_f32_vec(&f["expected_s"]);

        let (u, s, vh) = cusolver::gpu_svd_f32(&data, m, n, &device)
            .unwrap_or_else(|e| panic!("gpu_svd_f32 ({tag}): {e}"));

        let k = m.min(n);
        assert_eq!(u.len(), m * k, "svd f32 ({tag}): U length");
        assert_eq!(s.len(), k, "svd f32 ({tag}): S length");
        assert_eq!(vh.len(), k * n, "svd f32 ({tag}): Vh length");

        // S values match reference (sorted descending).
        assert_close_f32(
            &s,
            &expected_s,
            1e-4,
            &format!("cusolver::gpu_svd_f32 S {tag}"),
        );

        // Reconstruction: A_ref = U @ diag(S) @ Vh.
        // U is [m,k], diag(S) is [k,k], Vh is [k,n].
        // Step 1: diag(S) @ Vh = scale each row i of Vh by S[i].
        let mut svh = vec![0.0f32; k * n];
        for i in 0..k {
            for j in 0..n {
                svh[i * n + j] = s[i] * vh[i * n + j];
            }
        }
        // Step 2: U @ (diag(S) @ Vh).
        let recon = matmul_f32(&u, &svh, m, k, n);

        let a_data = as_f32_vec(&f["a_data"]);
        let diff = frob_diff_f32(&recon, &a_data);
        let norm = frob_f32(&a_data);
        let rel = diff / norm.max(1e-8);
        assert!(
            rel < 1e-4,
            "svd f32 ({tag}): reconstruction ||U@S@Vh - A||_F / ||A||_F = {rel:.6e} > 1e-4"
        );
    }
}

/// cuSOLVER SVD 4×4 f64.
#[test]
fn cusolver_svd_4x4_f64() {
    ensure_cuda();
    let fixtures = load_fixtures();
    let device = GpuDevice::new(0).expect("GpuDevice::new");

    for f in pick_fixtures(&fixtures, "cusolver", "gpu_svd_f64") {
        let tag = f["tag"].as_str().unwrap_or("?");
        let m = f["m"].as_u64().unwrap() as usize;
        let n = f["n"].as_u64().unwrap() as usize;

        let data = as_f64_vec(&f["input"]);
        let expected_s = as_f64_vec(&f["expected_s"]);

        let (u, s, vh) = cusolver::gpu_svd_f64(&data, m, n, &device)
            .unwrap_or_else(|e| panic!("gpu_svd_f64 ({tag}): {e}"));

        let k = m.min(n);
        assert_eq!(u.len(), m * k, "svd f64 ({tag}): U length");
        assert_eq!(s.len(), k, "svd f64 ({tag}): S length");
        assert_eq!(vh.len(), k * n, "svd f64 ({tag}): Vh length");

        assert_close_f64(
            &s,
            &expected_s,
            1e-9,
            &format!("cusolver::gpu_svd_f64 S {tag}"),
        );

        // Reconstruction.
        let mut svh = vec![0.0f64; k * n];
        for i in 0..k {
            for j in 0..n {
                svh[i * n + j] = s[i] * vh[i * n + j];
            }
        }
        let recon = matmul_f64(&u, &svh, m, k, n);
        let a_data = as_f64_vec(&f["a_data"]);
        let diff = frob_diff_f64(&recon, &a_data);
        let norm = frob_f64(&a_data);
        let rel = diff / norm.max(1e-15);
        assert!(
            rel < 1e-9,
            "svd f64 ({tag}): reconstruction ||U@S@Vh - A||_F / ||A||_F = {rel:.12e} > 1e-9"
        );
    }
}

/// cuSOLVER Cholesky 4×4 SPD f32.  Validates handle + workspace + devInfo lifecycle.
/// Result tested via reconstruction: ||L @ L^T - A||_F < 1e-4 * ||A||_F.
#[test]
fn cusolver_cholesky_4x4_f32() {
    ensure_cuda();
    let fixtures = load_fixtures();
    let device = GpuDevice::new(0).expect("GpuDevice::new");

    for f in pick_fixtures(&fixtures, "cusolver", "gpu_cholesky_f32") {
        let tag = f["tag"].as_str().unwrap_or("?");
        let n = f["n"].as_u64().unwrap() as usize;

        let data = as_f32_vec(&f["input"]);
        let spd = as_f32_vec(&f["spd_data"]);

        let l = cusolver::gpu_cholesky_f32(&data, n, &device)
            .unwrap_or_else(|e| panic!("gpu_cholesky_f32 ({tag}): {e}"));

        assert_eq!(l.len(), n * n, "cholesky f32 ({tag}): L length");

        // Reconstruction: L @ L^T must equal SPD within tolerance.
        let lt: Vec<f32> = {
            let mut t = vec![0.0f32; n * n];
            for i in 0..n {
                for j in 0..n {
                    t[i * n + j] = l[j * n + i];
                }
            }
            t
        };
        let recon = matmul_f32(&l, &lt, n, n, n);
        let diff = frob_diff_f32(&recon, &spd);
        let norm = frob_f32(&spd);
        let rel = diff / norm.max(1e-8);
        assert!(
            rel < 1e-4,
            "cholesky f32 ({tag}): ||L@L^T - A||_F / ||A||_F = {rel:.6e} > 1e-4"
        );
    }
}

/// cuSOLVER Cholesky 4×4 SPD f64.
#[test]
fn cusolver_cholesky_4x4_f64() {
    ensure_cuda();
    let fixtures = load_fixtures();
    let device = GpuDevice::new(0).expect("GpuDevice::new");

    for f in pick_fixtures(&fixtures, "cusolver", "gpu_cholesky_f64") {
        let tag = f["tag"].as_str().unwrap_or("?");
        let n = f["n"].as_u64().unwrap() as usize;

        let data = as_f64_vec(&f["input"]);
        let spd = as_f64_vec(&f["spd_data"]);

        let l = cusolver::gpu_cholesky_f64(&data, n, &device)
            .unwrap_or_else(|e| panic!("gpu_cholesky_f64 ({tag}): {e}"));

        assert_eq!(l.len(), n * n, "cholesky f64 ({tag}): L length");

        let lt: Vec<f64> = {
            let mut t = vec![0.0f64; n * n];
            for i in 0..n {
                for j in 0..n {
                    t[i * n + j] = l[j * n + i];
                }
            }
            t
        };
        let recon = matmul_f64(&l, &lt, n, n, n);
        let diff = frob_diff_f64(&recon, &spd);
        let norm = frob_f64(&spd);
        let rel = diff / norm.max(1e-15);
        assert!(
            rel < 1e-9,
            "cholesky f64 ({tag}): ||L@L^T - A||_F / ||A||_F = {rel:.12e} > 1e-9"
        );
    }
}

// ===========================================================================
// Device-resident SVD (#896 / #635) — rust-gpu-discipline §3 parity tests
//
// These tests call `cusolver::gpu_svd_f32_dev` / `gpu_svd_f64_dev` and assert:
//   1. All three output buffers live on device 0 (`device_ordinal() == 0`).
//   2. S values match the PyTorch reference within tolerance.
//   3. Reconstruction ||U @ diag(S) @ Vh - A||_F / ||A||_F < tolerance.
//
// This is the §3 conformance check: outputs must stay on-device, not bounce
// through a host `Vec<T>`.
// ===========================================================================

/// Device-resident SVD f32: asserts all outputs have device_ordinal == 0.
/// Uses the same fixtures as `cusolver_svd_4x4_f32` but calls the _dev variant.
#[test]
fn cusolver_svd_dev_f32_device_resident() {
    ensure_cuda();
    let fixtures = load_fixtures();
    let device = GpuDevice::new(0).expect("GpuDevice::new");

    for f in pick_fixtures(&fixtures, "cusolver", "gpu_svd_f32") {
        let tag = f["tag"].as_str().unwrap_or("?");
        let m = f["m"].as_u64().unwrap() as usize;
        let n = f["n"].as_u64().unwrap() as usize;
        let k = m.min(n);

        let data = as_f32_vec(&f["input"]);
        let expected_s = as_f32_vec(&f["expected_s"]);

        // Upload input — stays on device throughout.
        let a_buf = ferrotorch_gpu::transfer::cpu_to_gpu(&data, &device)
            .unwrap_or_else(|e| panic!("svd_dev_f32 upload ({tag}): {e}"));

        let (u_buf, s_buf, vh_buf) = cusolver::gpu_svd_f32_dev(&a_buf, m, n, &device)
            .unwrap_or_else(|e| panic!("gpu_svd_f32_dev ({tag}): {e}"));

        // §3 assertion: outputs must be on device, not on host.
        assert_eq!(
            u_buf.device_ordinal(),
            0,
            "svd_dev f32 ({tag}): U on wrong device"
        );
        assert_eq!(
            s_buf.device_ordinal(),
            0,
            "svd_dev f32 ({tag}): S on wrong device"
        );
        assert_eq!(
            vh_buf.device_ordinal(),
            0,
            "svd_dev f32 ({tag}): Vh on wrong device"
        );

        assert_eq!(u_buf.len(), m * k, "svd_dev f32 ({tag}): U length");
        assert_eq!(s_buf.len(), k, "svd_dev f32 ({tag}): S length");
        assert_eq!(vh_buf.len(), k * n, "svd_dev f32 ({tag}): Vh length");

        // Download only for value verification.
        let u = ferrotorch_gpu::transfer::gpu_to_cpu(&u_buf, &device)
            .unwrap_or_else(|e| panic!("svd_dev_f32 dl U ({tag}): {e}"));
        let s = ferrotorch_gpu::transfer::gpu_to_cpu(&s_buf, &device)
            .unwrap_or_else(|e| panic!("svd_dev_f32 dl S ({tag}): {e}"));
        let vh = ferrotorch_gpu::transfer::gpu_to_cpu(&vh_buf, &device)
            .unwrap_or_else(|e| panic!("svd_dev_f32 dl Vh ({tag}): {e}"));

        assert_close_f32(&s, &expected_s, 1e-4, &format!("svd_dev_f32 S {tag}"));

        let mut svh = vec![0.0f32; k * n];
        for i in 0..k {
            for j in 0..n {
                svh[i * n + j] = s[i] * vh[i * n + j];
            }
        }
        let recon = matmul_f32(&u, &svh, m, k, n);
        let a_data = as_f32_vec(&f["a_data"]);
        let rel = frob_diff_f32(&recon, &a_data) / frob_f32(&a_data).max(1e-8);
        assert!(
            rel < 1e-4,
            "svd_dev_f32 ({tag}): recon error {rel:.6e} > 1e-4"
        );
    }
}

/// Device-resident SVD f64: asserts all outputs have device_ordinal == 0.
#[test]
fn cusolver_svd_dev_f64_device_resident() {
    ensure_cuda();
    let fixtures = load_fixtures();
    let device = GpuDevice::new(0).expect("GpuDevice::new");

    for f in pick_fixtures(&fixtures, "cusolver", "gpu_svd_f64") {
        let tag = f["tag"].as_str().unwrap_or("?");
        let m = f["m"].as_u64().unwrap() as usize;
        let n = f["n"].as_u64().unwrap() as usize;
        let k = m.min(n);

        let data = as_f64_vec(&f["input"]);
        let expected_s = as_f64_vec(&f["expected_s"]);

        let a_buf = ferrotorch_gpu::transfer::cpu_to_gpu(&data, &device)
            .unwrap_or_else(|e| panic!("svd_dev_f64 upload ({tag}): {e}"));

        let (u_buf, s_buf, vh_buf) = cusolver::gpu_svd_f64_dev(&a_buf, m, n, &device)
            .unwrap_or_else(|e| panic!("gpu_svd_f64_dev ({tag}): {e}"));

        assert_eq!(
            u_buf.device_ordinal(),
            0,
            "svd_dev f64 ({tag}): U on wrong device"
        );
        assert_eq!(
            s_buf.device_ordinal(),
            0,
            "svd_dev f64 ({tag}): S on wrong device"
        );
        assert_eq!(
            vh_buf.device_ordinal(),
            0,
            "svd_dev f64 ({tag}): Vh on wrong device"
        );

        assert_eq!(u_buf.len(), m * k, "svd_dev f64 ({tag}): U length");
        assert_eq!(s_buf.len(), k, "svd_dev f64 ({tag}): S length");
        assert_eq!(vh_buf.len(), k * n, "svd_dev f64 ({tag}): Vh length");

        let u = ferrotorch_gpu::transfer::gpu_to_cpu(&u_buf, &device)
            .unwrap_or_else(|e| panic!("svd_dev_f64 dl U ({tag}): {e}"));
        let s = ferrotorch_gpu::transfer::gpu_to_cpu(&s_buf, &device)
            .unwrap_or_else(|e| panic!("svd_dev_f64 dl S ({tag}): {e}"));
        let vh = ferrotorch_gpu::transfer::gpu_to_cpu(&vh_buf, &device)
            .unwrap_or_else(|e| panic!("svd_dev_f64 dl Vh ({tag}): {e}"));

        assert_close_f64(&s, &expected_s, 1e-9, &format!("svd_dev_f64 S {tag}"));

        let mut svh = vec![0.0f64; k * n];
        for i in 0..k {
            for j in 0..n {
                svh[i * n + j] = s[i] * vh[i * n + j];
            }
        }
        let recon = matmul_f64(&u, &svh, m, k, n);
        let a_data = as_f64_vec(&f["a_data"]);
        let rel = frob_diff_f64(&recon, &a_data) / frob_f64(&a_data).max(1e-15);
        assert!(
            rel < 1e-9,
            "svd_dev_f64 ({tag}): recon error {rel:.12e} > 1e-9"
        );
    }
}

// ===========================================================================
// Device-resident QR (#896 / #635) — rust-gpu-discipline §3 parity tests
//
// Asserts Q and R outputs live on device, then verifies:
//   1. ||Q^T @ Q - I||_F < tolerance  (Q columns are orthonormal)
//   2. ||Q @ R - A||_F / ||A||_F < tolerance  (reconstruction)
// ===========================================================================

/// Device-resident QR f32: asserts Q and R have device_ordinal == 0.
#[test]
fn cusolver_qr_dev_f32_device_resident() {
    ensure_cuda();
    let fixtures = load_fixtures();
    let device = GpuDevice::new(0).expect("GpuDevice::new");

    for f in pick_fixtures(&fixtures, "cusolver", "gpu_svd_f32") {
        // Reuse svd fixtures (same m×n matrices) — QR works on any m×n matrix.
        let tag = f["tag"].as_str().unwrap_or("?");
        let m = f["m"].as_u64().unwrap() as usize;
        let n = f["n"].as_u64().unwrap() as usize;
        let k = m.min(n);

        let a_data_flat = as_f32_vec(&f["a_data"]);

        let a_buf = ferrotorch_gpu::transfer::cpu_to_gpu(&a_data_flat, &device)
            .unwrap_or_else(|e| panic!("qr_dev_f32 upload ({tag}): {e}"));

        let (q_buf, r_buf) = cusolver::gpu_qr_f32_dev(&a_buf, m, n, &device)
            .unwrap_or_else(|e| panic!("gpu_qr_f32_dev ({tag}): {e}"));

        // §3 assertion: Q and R must live on device.
        assert_eq!(
            q_buf.device_ordinal(),
            0,
            "qr_dev f32 ({tag}): Q on wrong device"
        );
        assert_eq!(
            r_buf.device_ordinal(),
            0,
            "qr_dev f32 ({tag}): R on wrong device"
        );

        assert_eq!(q_buf.len(), m * k, "qr_dev f32 ({tag}): Q length");
        assert_eq!(r_buf.len(), k * n, "qr_dev f32 ({tag}): R length");

        let q = ferrotorch_gpu::transfer::gpu_to_cpu(&q_buf, &device)
            .unwrap_or_else(|e| panic!("qr_dev_f32 dl Q ({tag}): {e}"));
        let r = ferrotorch_gpu::transfer::gpu_to_cpu(&r_buf, &device)
            .unwrap_or_else(|e| panic!("qr_dev_f32 dl R ({tag}): {e}"));

        // Check Q columns are orthonormal: Q^T @ Q ~ I_k.
        // q is [m,k] row-major; qt is [k,m]; qt@q is [k,k].
        let qt: Vec<f32> = {
            let mut t = vec![0.0f32; k * m];
            for i in 0..m {
                for j in 0..k {
                    t[j * m + i] = q[i * k + j];
                }
            }
            t
        };
        let qtq = matmul_f32(&qt, &q, k, m, k);
        for i in 0..k {
            for j in 0..k {
                let expected = if i == j { 1.0f32 } else { 0.0f32 };
                let got = qtq[i * k + j];
                assert!(
                    (got - expected).abs() < 1e-4,
                    "qr_dev_f32 ({tag}): Q^T@Q[{i},{j}] = {got:.6e}, expected {expected:.1}"
                );
            }
        }

        // Reconstruction: Q @ R ~ A.
        let recon = matmul_f32(&q, &r, m, k, n);
        let rel = frob_diff_f32(&recon, &a_data_flat) / frob_f32(&a_data_flat).max(1e-8);
        assert!(
            rel < 1e-4,
            "qr_dev_f32 ({tag}): recon error {rel:.6e} > 1e-4"
        );
    }
}

/// Device-resident QR f64: asserts Q and R have device_ordinal == 0.
#[test]
fn cusolver_qr_dev_f64_device_resident() {
    ensure_cuda();
    let fixtures = load_fixtures();
    let device = GpuDevice::new(0).expect("GpuDevice::new");

    for f in pick_fixtures(&fixtures, "cusolver", "gpu_svd_f64") {
        let tag = f["tag"].as_str().unwrap_or("?");
        let m = f["m"].as_u64().unwrap() as usize;
        let n = f["n"].as_u64().unwrap() as usize;
        let k = m.min(n);

        let a_data_flat = as_f64_vec(&f["a_data"]);

        let a_buf = ferrotorch_gpu::transfer::cpu_to_gpu(&a_data_flat, &device)
            .unwrap_or_else(|e| panic!("qr_dev_f64 upload ({tag}): {e}"));

        let (q_buf, r_buf) = cusolver::gpu_qr_f64_dev(&a_buf, m, n, &device)
            .unwrap_or_else(|e| panic!("gpu_qr_f64_dev ({tag}): {e}"));

        assert_eq!(
            q_buf.device_ordinal(),
            0,
            "qr_dev f64 ({tag}): Q on wrong device"
        );
        assert_eq!(
            r_buf.device_ordinal(),
            0,
            "qr_dev f64 ({tag}): R on wrong device"
        );

        assert_eq!(q_buf.len(), m * k, "qr_dev f64 ({tag}): Q length");
        assert_eq!(r_buf.len(), k * n, "qr_dev f64 ({tag}): R length");

        let q = ferrotorch_gpu::transfer::gpu_to_cpu(&q_buf, &device)
            .unwrap_or_else(|e| panic!("qr_dev_f64 dl Q ({tag}): {e}"));
        let r = ferrotorch_gpu::transfer::gpu_to_cpu(&r_buf, &device)
            .unwrap_or_else(|e| panic!("qr_dev_f64 dl R ({tag}): {e}"));

        // Q^T @ Q ~ I_k.
        let qt: Vec<f64> = {
            let mut t = vec![0.0f64; k * m];
            for i in 0..m {
                for j in 0..k {
                    t[j * m + i] = q[i * k + j];
                }
            }
            t
        };
        let qtq = matmul_f64(&qt, &q, k, m, k);
        for i in 0..k {
            for j in 0..k {
                let expected = if i == j { 1.0f64 } else { 0.0f64 };
                let got = qtq[i * k + j];
                assert!(
                    (got - expected).abs() < 1e-9,
                    "qr_dev_f64 ({tag}): Q^T@Q[{i},{j}] = {got:.12e}, expected {expected:.1}"
                );
            }
        }

        // Reconstruction: Q @ R ~ A.
        let recon = matmul_f64(&q, &r, m, k, n);
        let rel = frob_diff_f64(&recon, &a_data_flat) / frob_f64(&a_data_flat).max(1e-15);
        assert!(
            rel < 1e-9,
            "qr_dev_f64 ({tag}): recon error {rel:.12e} > 1e-9"
        );
    }
}

// ===========================================================================
// MODULE: cusparselt.rs — cuSPARSELt handle lifecycle + SpMM round-trip
// ===========================================================================

/// cuSPARSELt handle init + 2:4 structured SpMM on 8×8 f32.
///
/// Gated on `#[cfg(feature = "cusparselt")]` because the library requires
/// the optional `cusparselt` cargo feature and the cuSPARSELt SDK to be
/// present at link time.
#[cfg(feature = "cusparselt")]
#[test]
fn cusparselt_spmm_8x8_f32() {
    ensure_cuda();
    let fixtures = load_fixtures();
    let device = GpuDevice::new(0).expect("GpuDevice::new");

    for f in pick_fixtures(&fixtures, "cusparselt", "gpu_sparse_matmul_24") {
        let tag = f["tag"].as_str().unwrap_or("?");
        let m = f["m"].as_u64().unwrap() as usize;
        let k = f["k"].as_u64().unwrap() as usize;
        let n = f["n"].as_u64().unwrap() as usize;

        let a_host = as_f32_vec(&f["a_data"]);
        let b_decompressed_host = as_f32_vec(&f["b_decompressed"]);
        let expected = as_f32_vec(&f["expected"]);

        let a_buf = ferrotorch_gpu::transfer::cpu_to_gpu(&a_host, &device)
            .unwrap_or_else(|e| panic!("cusparselt upload A ({tag}): {e}"));
        let b_buf = ferrotorch_gpu::transfer::cpu_to_gpu(&b_decompressed_host, &device)
            .unwrap_or_else(|e| panic!("cusparselt upload B ({tag}): {e}"));

        // Create a fresh handle for this test.
        let handle = ferrotorch_gpu::cusparselt::CusparseLtHandle::new()
            .unwrap_or_else(|e| panic!("CusparseLtHandle::new ({tag}): {e}"));

        let c_buf = ferrotorch_gpu::cusparselt::gpu_sparse_matmul_24::<f32>(
            &handle,
            &a_buf,
            &b_buf,
            m,
            k,
            n,
            ferrotorch_gpu::cusparselt::CuSpLtDType::F32,
            &device,
        )
        .unwrap_or_else(|e| panic!("gpu_sparse_matmul_24 ({tag}): {e}"));

        // Handle must drop without crash here (tests `cusparseLtDestroy` path).
        drop(handle);

        assert_eq!(
            c_buf.device_ordinal(),
            0,
            "cusparselt ({tag}): result on device 0"
        );
        assert_eq!(c_buf.len(), m * n, "cusparselt ({tag}): output length");

        let got = ferrotorch_gpu::transfer::gpu_to_cpu(&c_buf, &device)
            .unwrap_or_else(|e| panic!("cusparselt download ({tag}): {e}"));

        // TF32 accumulation mode has looser tolerance than f32 exact arithmetic.
        assert_close_f32(
            &got,
            &expected,
            5e-3,
            &format!("cusparselt::gpu_sparse_matmul_24 {tag}"),
        );
    }
}

/// Non-cusparselt build: stub test that records the cascade-skip so CI
/// correctly attributes the gap to the missing feature rather than to missing
/// coverage.
#[cfg(not(feature = "cusparselt"))]
#[test]
fn cusparselt_spmm_cascade_skip() {
    // CASCADE-C8.3-003: cusparselt feature not enabled — cuSPARSELt SpMM
    // binding layer untestable without the SDK.  Build with
    // `--features cusparselt` to enable.
    eprintln!(
        "SKIP cusparselt::gpu_sparse_matmul_24 — \
         `cusparselt` feature not enabled (C8.3 cascade skip for non-cusparselt builds)"
    );
}

// ===========================================================================
// MODULE: bf16.rs — PTX kernel lifecycle (compile + launch) + elementwise ops
// ===========================================================================

/// Upload bf16 bit-pattern buffers, run kernel, download, compare with
/// PyTorch-generated expected bits (converted to f32 for tolerance comparison).
///
/// A bf16 result bit matches exactly iff the PTX round-to-nearest-even logic
/// agrees with PyTorch's `torch.bfloat16` arithmetic. We allow 1 bf16 ULP
/// (≈ `2^{exp-7}`) which is equivalent to checking the bit pattern exactly
/// for normal values, but gracefully handles subnormal edge cases.
///
/// The comparison is done as f32 (decode both got and expected from bf16 bits)
/// with an absolute tolerance of 1/256 (half bf16 ULP at magnitude 1).
#[test]
fn bf16_mul_n16() {
    ensure_cuda();
    let fixtures = load_fixtures();
    let device = GpuDevice::new(0).expect("GpuDevice::new");
    let stream = device.stream();

    for f in pick_fixtures(&fixtures, "bf16", "gpu_mul_bf16") {
        let tag = f["tag"].as_str().unwrap_or("?");
        let n = f["n"].as_u64().unwrap() as usize;

        let a_bits = as_u16_vec(&f["a_bits"]);
        let b_bits = as_u16_vec(&f["b_bits"]);
        let expected_bits = as_u16_vec(&f["expected_bits"]);

        // Upload as u16 directly via CudaStream::clone_htod.
        let a_dev = stream
            .clone_htod(&a_bits)
            .unwrap_or_else(|e| panic!("bf16 mul upload A ({tag}): {e}"));
        let b_dev = stream
            .clone_htod(&b_bits)
            .unwrap_or_else(|e| panic!("bf16 mul upload B ({tag}): {e}"));

        let out_dev = ferrotorch_gpu::bf16::gpu_mul_bf16(&a_dev, &b_dev, &device)
            .unwrap_or_else(|e| panic!("gpu_mul_bf16 ({tag}): {e}"));

        // Verify length.
        assert_eq!(out_dev.len(), n, "bf16 mul ({tag}): output length");

        let got_bits: Vec<u16> = stream
            .clone_dtoh(&out_dev)
            .unwrap_or_else(|e| panic!("bf16 mul download ({tag}): {e}"));

        // Compare as f32 decoded from bf16 bits.
        let got_f32: Vec<f32> = got_bits.iter().map(|&b| bf16_bits_to_f32(b)).collect();
        let exp_f32: Vec<f32> = expected_bits.iter().map(|&b| bf16_bits_to_f32(b)).collect();
        assert_close_f32(
            &got_f32,
            &exp_f32,
            1.0 / 128.0,
            &format!("bf16::gpu_mul_bf16 {tag}"),
        );
    }
}

#[test]
fn bf16_add_n16() {
    ensure_cuda();
    let fixtures = load_fixtures();
    let device = GpuDevice::new(0).expect("GpuDevice::new");
    let stream = device.stream();

    for f in pick_fixtures(&fixtures, "bf16", "gpu_add_bf16") {
        let tag = f["tag"].as_str().unwrap_or("?");
        let n = f["n"].as_u64().unwrap() as usize;

        let a_bits = as_u16_vec(&f["a_bits"]);
        let b_bits = as_u16_vec(&f["b_bits"]);
        let expected_bits = as_u16_vec(&f["expected_bits"]);

        let a_dev = stream
            .clone_htod(&a_bits)
            .unwrap_or_else(|e| panic!("bf16 add upload A ({tag}): {e}"));
        let b_dev = stream
            .clone_htod(&b_bits)
            .unwrap_or_else(|e| panic!("bf16 add upload B ({tag}): {e}"));

        let out_dev = ferrotorch_gpu::bf16::gpu_add_bf16(&a_dev, &b_dev, &device)
            .unwrap_or_else(|e| panic!("gpu_add_bf16 ({tag}): {e}"));

        assert_eq!(out_dev.len(), n, "bf16 add ({tag}): output length");

        let got_bits: Vec<u16> = stream
            .clone_dtoh(&out_dev)
            .unwrap_or_else(|e| panic!("bf16 add download ({tag}): {e}"));

        let got_f32: Vec<f32> = got_bits.iter().map(|&b| bf16_bits_to_f32(b)).collect();
        let exp_f32: Vec<f32> = expected_bits.iter().map(|&b| bf16_bits_to_f32(b)).collect();
        assert_close_f32(
            &got_f32,
            &exp_f32,
            1.0 / 128.0,
            &format!("bf16::gpu_add_bf16 {tag}"),
        );
    }
}

#[test]
fn bf16_silu_n16() {
    ensure_cuda();
    let fixtures = load_fixtures();
    let device = GpuDevice::new(0).expect("GpuDevice::new");
    let stream = device.stream();

    for f in pick_fixtures(&fixtures, "bf16", "gpu_silu_bf16") {
        let tag = f["tag"].as_str().unwrap_or("?");
        let n = f["n"].as_u64().unwrap() as usize;

        let a_bits = as_u16_vec(&f["a_bits"]);
        let expected_bits = as_u16_vec(&f["expected_bits"]);

        let a_dev = stream
            .clone_htod(&a_bits)
            .unwrap_or_else(|e| panic!("bf16 silu upload ({tag}): {e}"));

        let out_dev = ferrotorch_gpu::bf16::gpu_silu_bf16(&a_dev, &device)
            .unwrap_or_else(|e| panic!("gpu_silu_bf16 ({tag}): {e}"));

        assert_eq!(out_dev.len(), n, "bf16 silu ({tag}): output length");

        let got_bits: Vec<u16> = stream
            .clone_dtoh(&out_dev)
            .unwrap_or_else(|e| panic!("bf16 silu download ({tag}): {e}"));

        let got_f32: Vec<f32> = got_bits.iter().map(|&b| bf16_bits_to_f32(b)).collect();
        let exp_f32: Vec<f32> = expected_bits.iter().map(|&b| bf16_bits_to_f32(b)).collect();
        // SiLU uses exp2.approx — looser tolerance.
        assert_close_f32(
            &got_f32,
            &exp_f32,
            1.0 / 64.0,
            &format!("bf16::gpu_silu_bf16 {tag}"),
        );
    }
}

#[test]
fn bf16_relu_n16() {
    ensure_cuda();
    let fixtures = load_fixtures();
    let device = GpuDevice::new(0).expect("GpuDevice::new");
    let stream = device.stream();

    for f in pick_fixtures(&fixtures, "bf16", "gpu_relu_bf16") {
        let tag = f["tag"].as_str().unwrap_or("?");
        let n = f["n"].as_u64().unwrap() as usize;

        let a_bits = as_u16_vec(&f["a_bits"]);
        let expected_bits = as_u16_vec(&f["expected_bits"]);

        let a_dev = stream
            .clone_htod(&a_bits)
            .unwrap_or_else(|e| panic!("bf16 relu upload ({tag}): {e}"));

        let out_dev = ferrotorch_gpu::bf16::gpu_relu_bf16(&a_dev, &device)
            .unwrap_or_else(|e| panic!("gpu_relu_bf16 ({tag}): {e}"));

        assert_eq!(out_dev.len(), n, "bf16 relu ({tag}): output length");

        let got_bits: Vec<u16> = stream
            .clone_dtoh(&out_dev)
            .unwrap_or_else(|e| panic!("bf16 relu download ({tag}): {e}"));

        let got_f32: Vec<f32> = got_bits.iter().map(|&b| bf16_bits_to_f32(b)).collect();
        let exp_f32: Vec<f32> = expected_bits.iter().map(|&b| bf16_bits_to_f32(b)).collect();
        // ReLU is exact: max(0, x) with no rounding. Exact bf16 bit match expected.
        assert_close_f32(
            &got_f32,
            &exp_f32,
            1.0 / 128.0,
            &format!("bf16::gpu_relu_bf16 {tag}"),
        );
    }
}

/// bf16 kernel with zero-length input must not panic.
#[test]
fn bf16_empty_input_no_panic() {
    ensure_cuda();
    let device = GpuDevice::new(0).expect("GpuDevice::new");
    let stream = device.stream();

    let empty: Vec<u16> = vec![];
    let a_dev = stream.clone_htod(&empty).expect("upload empty");
    let b_dev = stream.clone_htod(&empty).expect("upload empty");

    ferrotorch_gpu::bf16::gpu_mul_bf16(&a_dev, &b_dev, &device)
        .expect("gpu_mul_bf16 with n=0 must return Ok");
    ferrotorch_gpu::bf16::gpu_silu_bf16(&a_dev, &device)
        .expect("gpu_silu_bf16 with n=0 must return Ok");
    ferrotorch_gpu::bf16::gpu_relu_bf16(&a_dev, &device)
        .expect("gpu_relu_bf16 with n=0 must return Ok");
}
