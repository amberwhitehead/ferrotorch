//! bf16 elementwise dispatcher conformance (ferrotorch#23).
//!
//! Verifies that every `*_f32` `GpuBackend` trait method that we extended in
//! `backend_impl.rs` accepts a bf16 (`CudaSlice<u16>`) buffer and routes to
//! the matching `gpu_*_bf16` PTX kernel, instead of erroring with
//! `"GPU handle does not contain a CudaBuffer<f32>"`. The pre-fix behaviour
//! (an immediate downcast to `&CudaBuffer<f32>`) is exactly what
//! forecast-bio/ferrotorch#23 reports.
//!
//! Each test:
//!   1. Builds an input as `Vec<bf16>` on the host.
//!   2. Uploads via `cpu_to_gpu` with `elem_size = 2` → bf16 handle.
//!   3. Calls the `*_f32` trait method with the bf16 handle.
//!   4. Downloads the output and decodes it as `Vec<bf16>`.
//!   5. Compares to a CPU bf16 reference within bf16 tolerance.
//!
//! The tolerance is per-element (`max_abs <= rel * max(|ref|, 1.0)`) with
//! `rel = 5e-2` — well above bf16 ULP for the activations we exercise but
//! well below "off by an order of magnitude" — so a regression that wires
//! the wrong kernel will fail loudly.

#![cfg(feature = "cuda")]

use ferrotorch_core::gpu_dispatch::{self, GpuBackend, GpuBufferHandle};
use ferrotorch_gpu::init_cuda_backend;
use half::bf16;

fn ensure_init() {
    if !gpu_dispatch::has_gpu_backend() {
        init_cuda_backend().expect("init_cuda_backend");
    }
}

fn bf16_slice_bytes(data: &[bf16]) -> &[u8] {
    // SAFETY: bf16 is repr(transparent) over u16, 2 bytes per element.
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data)) }
}

fn bytes_to_bf16(bytes: &[u8]) -> Vec<bf16> {
    bytes
        .chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

fn upload_bf16(data: &[bf16], backend: &dyn GpuBackend) -> GpuBufferHandle {
    backend
        .cpu_to_gpu(bf16_slice_bytes(data), 2, 0)
        .expect("cpu_to_gpu bf16")
}

fn download_bf16(h: &GpuBufferHandle, backend: &dyn GpuBackend) -> Vec<bf16> {
    bytes_to_bf16(&backend.gpu_to_cpu(h).expect("gpu_to_cpu bf16"))
}

fn assert_close_bf16(got: &[bf16], expected: &[bf16], rel: f32, ctx: &str) {
    assert_eq!(got.len(), expected.len(), "{ctx}: length mismatch");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let gf = g.to_f32();
        let ef = e.to_f32();
        let tol = rel * ef.abs().max(1.0);
        assert!(
            (gf - ef).abs() <= tol,
            "{ctx}[{i}]: got {gf}, want {ef} (tol {tol})"
        );
    }
}

// ---------------------------------------------------------------------------
// add_f32 / mul_f32 (binary elementwise)
// ---------------------------------------------------------------------------

#[test]
fn add_f32_routes_bf16_inputs_to_gpu_add_bf16() {
    ensure_init();
    let backend = gpu_dispatch::gpu_backend().expect("backend");

    let a: Vec<bf16> = (0..256).map(|i| bf16::from_f32(i as f32 * 0.125)).collect();
    let b: Vec<bf16> = (0..256)
        .map(|i| bf16::from_f32(-(i as f32) * 0.25))
        .collect();
    let a_h = upload_bf16(&a, backend);
    let b_h = upload_bf16(&b, backend);

    let out_h = backend
        .add_f32(&a_h, &b_h)
        .expect("add_f32 on bf16 handles");
    let out = download_bf16(&out_h, backend);

    let expected: Vec<bf16> = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| bf16::from_f32(x.to_f32() + y.to_f32()))
        .collect();
    assert_close_bf16(&out, &expected, 5e-2, "add_f32(bf16, bf16)");
}

#[test]
fn mul_f32_routes_bf16_inputs_to_gpu_mul_bf16() {
    ensure_init();
    let backend = gpu_dispatch::gpu_backend().expect("backend");

    let a: Vec<bf16> = (0..256)
        .map(|i| bf16::from_f32(i as f32 * 0.0625))
        .collect();
    let b: Vec<bf16> = (0..256)
        .map(|i| bf16::from_f32(0.5 - i as f32 * 0.01))
        .collect();
    let a_h = upload_bf16(&a, backend);
    let b_h = upload_bf16(&b, backend);

    let out_h = backend
        .mul_f32(&a_h, &b_h)
        .expect("mul_f32 on bf16 handles");
    let out = download_bf16(&out_h, backend);

    let expected: Vec<bf16> = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| bf16::from_f32(x.to_f32() * y.to_f32()))
        .collect();
    assert_close_bf16(&out, &expected, 5e-2, "mul_f32(bf16, bf16)");
}

// ---------------------------------------------------------------------------
// scale_f32 (scalar multiply)
// ---------------------------------------------------------------------------

#[test]
fn scale_f32_routes_bf16_input_to_gpu_scale_bf16() {
    ensure_init();
    let backend = gpu_dispatch::gpu_backend().expect("backend");

    let a: Vec<bf16> = (0..128).map(|i| bf16::from_f32(i as f32 - 64.0)).collect();
    let a_h = upload_bf16(&a, backend);

    let out_h = backend
        .scale_f32(&a_h, 0.125)
        .expect("scale_f32 on bf16 handle");
    let out = download_bf16(&out_h, backend);

    let expected: Vec<bf16> = a
        .iter()
        .map(|x| bf16::from_f32(x.to_f32() * 0.125))
        .collect();
    assert_close_bf16(&out, &expected, 5e-2, "scale_f32(bf16, 0.125)");
}

// ---------------------------------------------------------------------------
// relu_f32 / silu_f32 / gelu_f32 (unary activations)
// ---------------------------------------------------------------------------

#[test]
fn relu_f32_routes_bf16_input_to_gpu_relu_bf16() {
    ensure_init();
    let backend = gpu_dispatch::gpu_backend().expect("backend");

    let a: Vec<bf16> = (0..128).map(|i| bf16::from_f32(i as f32 - 64.0)).collect();
    let a_h = upload_bf16(&a, backend);

    let out_h = backend.relu_f32(&a_h).expect("relu_f32 on bf16 handle");
    let out = download_bf16(&out_h, backend);

    let expected: Vec<bf16> = a
        .iter()
        .map(|x| bf16::from_f32(x.to_f32().max(0.0)))
        .collect();
    assert_close_bf16(&out, &expected, 5e-2, "relu_f32(bf16)");
}

#[test]
fn silu_f32_routes_bf16_input_to_gpu_silu_bf16() {
    ensure_init();
    let backend = gpu_dispatch::gpu_backend().expect("backend");

    let a: Vec<bf16> = (-32..32).map(|i| bf16::from_f32(i as f32 * 0.25)).collect();
    let a_h = upload_bf16(&a, backend);

    let out_h = backend.silu_f32(&a_h).expect("silu_f32 on bf16 handle");
    let out = download_bf16(&out_h, backend);

    let expected: Vec<bf16> = a
        .iter()
        .map(|x| {
            let f = x.to_f32();
            bf16::from_f32(f / (1.0 + (-f).exp()))
        })
        .collect();
    assert_close_bf16(&out, &expected, 5e-2, "silu_f32(bf16)");
}

#[test]
fn gelu_f32_routes_bf16_input_to_gpu_gelu_bf16() {
    ensure_init();
    let backend = gpu_dispatch::gpu_backend().expect("backend");

    let a: Vec<bf16> = (-32..32).map(|i| bf16::from_f32(i as f32 * 0.25)).collect();
    let a_h = upload_bf16(&a, backend);

    let out_h = backend.gelu_f32(&a_h).expect("gelu_f32 on bf16 handle");
    let out = download_bf16(&out_h, backend);

    // Hastings erf approx (matches gpu_gelu_bf16's PTX impl exactly).
    #[allow(clippy::excessive_precision)]
    fn erf(x: f32) -> f32 {
        let sign = x.signum();
        let ax = x.abs();
        let t = 1.0_f32 / (1.0_f32 + 0.3275911_f32 * ax);
        let poly = ((((1.061405429_f32 * t - 1.453152027_f32) * t + 1.421413741_f32) * t
            - 0.284496736_f32)
            * t
            + 0.254829592_f32)
            * t;
        sign * (1.0 - poly * (-(ax * ax)).exp())
    }
    let expected: Vec<bf16> = a
        .iter()
        .map(|x| {
            let f = x.to_f32();
            bf16::from_f32(0.5 * f * (1.0 + erf(f / std::f32::consts::SQRT_2)))
        })
        .collect();
    assert_close_bf16(&out, &expected, 5e-2, "gelu_f32(bf16)");
}

// ---------------------------------------------------------------------------
// softmax_f32 (row-wise normalize)
// ---------------------------------------------------------------------------

#[test]
fn softmax_f32_routes_bf16_input_to_gpu_softmax_bf16() {
    ensure_init();
    let backend = gpu_dispatch::gpu_backend().expect("backend");

    let rows = 4_usize;
    let cols = 16_usize;
    let a: Vec<bf16> = (0..rows * cols)
        .map(|i| bf16::from_f32((i as f32 * 0.1).sin()))
        .collect();
    let a_h = upload_bf16(&a, backend);

    let out_h = backend
        .softmax_f32(&a_h, rows, cols)
        .expect("softmax_f32 on bf16 handle");
    let out = download_bf16(&out_h, backend);

    let mut expected = vec![bf16::from_f32(0.0); rows * cols];
    for r in 0..rows {
        let row: Vec<f32> = a[r * cols..(r + 1) * cols]
            .iter()
            .map(|v| v.to_f32())
            .collect();
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|&v| (v - m).exp()).collect();
        let s: f32 = exps.iter().sum();
        for (c, e) in exps.iter().enumerate() {
            expected[r * cols + c] = bf16::from_f32(e / s);
        }
    }
    // Softmax outputs are bounded in [0, 1]; bf16's per-elem precision near
    // small values is loose, so we allow `abs_tol = 5e-2` directly on the
    // value (not relative).
    for (i, (g, e)) in out.iter().zip(expected.iter()).enumerate() {
        let gf = g.to_f32();
        let ef = e.to_f32();
        assert!(
            (gf - ef).abs() <= 5e-2,
            "softmax_f32(bf16)[{i}]: got {gf}, want {ef}"
        );
    }
}

// ---------------------------------------------------------------------------
// layernorm_f32 / rmsnorm_f32
// ---------------------------------------------------------------------------

#[test]
fn layernorm_f32_routes_bf16_input_to_gpu_layernorm_bf16() {
    ensure_init();
    let backend = gpu_dispatch::gpu_backend().expect("backend");

    let rows = 2_usize;
    let cols = 32_usize;
    let x: Vec<bf16> = (0..rows * cols)
        .map(|i| bf16::from_f32((i as f32 * 0.13).cos() * 2.0))
        .collect();
    let gamma: Vec<bf16> = (0..cols)
        .map(|i| bf16::from_f32(1.0 + i as f32 * 0.01))
        .collect();
    let beta: Vec<bf16> = (0..cols)
        .map(|i| bf16::from_f32(i as f32 * 0.005))
        .collect();
    let eps = 1e-5_f32;

    let x_h = upload_bf16(&x, backend);
    let g_h = upload_bf16(&gamma, backend);
    let b_h = upload_bf16(&beta, backend);

    let out_h = backend
        .layernorm_f32(&x_h, &g_h, &b_h, rows, cols, eps)
        .expect("layernorm_f32 on bf16 handles");
    let out = download_bf16(&out_h, backend);

    let mut expected = vec![bf16::from_f32(0.0); rows * cols];
    for r in 0..rows {
        let row: Vec<f32> = x[r * cols..(r + 1) * cols]
            .iter()
            .map(|v| v.to_f32())
            .collect();
        let mean = row.iter().sum::<f32>() / (cols as f32);
        let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / (cols as f32);
        let inv_std = 1.0 / (var + eps).sqrt();
        for c in 0..cols {
            let n = (row[c] - mean) * inv_std * gamma[c].to_f32() + beta[c].to_f32();
            expected[r * cols + c] = bf16::from_f32(n);
        }
    }
    assert_close_bf16(&out, &expected, 1e-1, "layernorm_f32(bf16)");
}

#[test]
fn rmsnorm_f32_routes_bf16_input_to_gpu_rmsnorm_bf16() {
    ensure_init();
    let backend = gpu_dispatch::gpu_backend().expect("backend");

    let rows = 2_usize;
    let cols = 32_usize;
    let x: Vec<bf16> = (0..rows * cols)
        .map(|i| bf16::from_f32((i as f32 * 0.07).sin() * 3.0))
        .collect();
    let w: Vec<bf16> = (0..cols)
        .map(|i| bf16::from_f32(0.5 + i as f32 * 0.02))
        .collect();
    let eps = 1e-5_f32;

    let x_h = upload_bf16(&x, backend);
    let w_h = upload_bf16(&w, backend);

    let out_h = backend
        .rmsnorm_f32(&x_h, &w_h, rows, cols, eps)
        .expect("rmsnorm_f32 on bf16 handles");
    let out = download_bf16(&out_h, backend);

    let mut expected = vec![bf16::from_f32(0.0); rows * cols];
    for r in 0..rows {
        let row: Vec<f32> = x[r * cols..(r + 1) * cols]
            .iter()
            .map(|v| v.to_f32())
            .collect();
        let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / (cols as f32);
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();
        for c in 0..cols {
            expected[r * cols + c] = bf16::from_f32(row[c] * inv_rms * w[c].to_f32());
        }
    }
    assert_close_bf16(&out, &expected, 1e-1, "rmsnorm_f32(bf16)");
}

// ---------------------------------------------------------------------------
// Mixed-dtype guard: passing an f32 and a bf16 handle to add_f32 must error,
// not silently coerce. The dispatcher's tail-error path is the spec.
// ---------------------------------------------------------------------------

#[test]
fn add_f32_mixed_dtypes_returns_error() {
    ensure_init();
    let backend = gpu_dispatch::gpu_backend().expect("backend");

    let a_bf16: Vec<bf16> = vec![bf16::from_f32(1.0); 16];
    let a_h = upload_bf16(&a_bf16, backend);

    let b_f32: Vec<f32> = vec![1.0_f32; 16];
    // SAFETY: f32 -> u8 view, 4 bytes per element.
    let b_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            b_f32.as_ptr().cast::<u8>(),
            std::mem::size_of_val(b_f32.as_slice()),
        )
    };
    let b_h = backend.cpu_to_gpu(b_bytes, 4, 0).expect("cpu_to_gpu f32");

    let err = backend
        .add_f32(&a_h, &b_h)
        .expect_err("mixed dtypes must err");
    let msg = format!("{err}");
    assert!(
        msg.contains("f32") || msg.contains("bf16") || msg.contains("CudaSlice<u16>"),
        "expected dtype-mismatch error, got: {msg}"
    );
}
