//! Phase 1 empirical proof (GPU dtype-parity epic, crosslink #1185):
//! `Tensor<half::f16>` executes the full bf16 op set ON THE GPU with NO CPU
//! round trips.
//!
//! IEEE float16 (`half::f16`) is distinct from bf16 — both are 2 bytes and
//! both store on-device as `CudaSlice<u16>`, disambiguated only by the
//! `GpuBufferHandle` `DType::F16` tag (Phase 0 payoff). This sweep enumerates
//! every op the dispatch macro routes for f16 and, for each:
//!
//!   (a) constructs inputs on CUDA,
//!   (b) runs the op,
//!   (c) asserts the result STAYED RESIDENT (`is_cuda()` and
//!       `device() == Cuda(0)` — no silent CPU detour),
//!   (d) compares values against an f32 CPU reference within f16 tolerance.
//!
//! The empirical bar (rust-gpu-discipline §4): f16 ops must run on the GPU,
//! verified — not assumed. The only host reads in this file are the
//! end-of-op value checks in step (d). The op outputs themselves never leave
//! VRAM until we explicitly read them back for comparison.
//!
//! Prints a PASS/FAIL table ending in `PASS: N, FAIL: 0`.

#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::Tensor;
use ferrotorch_core::creation::from_vec;
use ferrotorch_core::device::Device;
use ferrotorch_core::error::FerrotorchError;
use ferrotorch_core::grad_fns::activation::GeluApproximate;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for the f16 op sweep");
    });
}

/// Build an f16 tensor resident on CUDA(0) from f32 source data.
fn f16_cuda(data: &[f32], shape: &[usize]) -> Tensor<half::f16> {
    let h: Vec<half::f16> = data.iter().copied().map(half::f16::from_f32).collect();
    let cpu = from_vec::<half::f16>(h, shape).expect("f16 cpu tensor");
    cpu.to(Device::Cuda(0))
        .expect("Tensor<f16>::to(Cuda) must succeed")
}

/// Read an f16 CUDA tensor back to host f32 values (the ONLY allowed host
/// read — used by the value-check in step (d)).
fn to_host_f32(t: &Tensor<half::f16>) -> Result<Vec<f32>, String> {
    let cpu = t.clone().to(Device::Cpu).map_err(|e| format!("{e}"))?;
    let data = cpu.data().map_err(|e| format!("{e}"))?;
    Ok(data.iter().map(|x| x.to_f32()).collect())
}

struct OpResult {
    name: &'static str,
    pass: bool,
    err: Option<String>,
}

/// Run one op: produce the f16 CUDA result + an f32 reference vector, then
/// assert (c) residency and (d) value parity within f16 tolerance.
fn run(
    name: &'static str,
    f: impl FnOnce() -> Result<(Tensor<half::f16>, Vec<f32>), FerrotorchError>,
) -> OpResult {
    let outcome = (|| -> Result<(), String> {
        let (out, reference) = f().map_err(|e| format!("op errored: {e}"))?;

        // (c) Residency: the op output must have stayed on the GPU. No silent
        // CPU detour — assert both `is_cuda()` and the exact device.
        if !out.is_cuda() {
            return Err("result is NOT cuda-resident (silent CPU detour?)".into());
        }
        match out.device() {
            Device::Cuda(0) => {}
            other => return Err(format!("result device is {other:?}, expected Cuda(0)")),
        }

        // (d) Value parity vs. f32 CPU reference within f16 tolerance.
        let got = to_host_f32(&out)?;
        if got.len() != reference.len() {
            return Err(format!(
                "length mismatch: got {} values, reference {}",
                got.len(),
                reference.len()
            ));
        }
        for (i, (&g, &r)) in got.iter().zip(reference.iter()).enumerate() {
            if r.is_nan() {
                if !g.is_nan() {
                    return Err(format!("value[{i}] = {g}, reference NaN"));
                }
                continue;
            }
            if g.is_nan() {
                return Err(format!("value[{i}] = NaN, reference {r}"));
            }
            // ~1e-2 relative (f16 has a 10-bit mantissa ≈ 3 decimal digits);
            // add a small absolute floor for values near zero.
            let tol = 1e-2_f32 * r.abs().max(1.0) + 1e-2;
            if (g - r).abs() > tol {
                return Err(format!(
                    "value[{i}] = {g}, reference {r}, |diff| {} > tol {tol}",
                    (g - r).abs()
                ));
            }
        }
        Ok(())
    })();

    match outcome {
        Ok(()) => OpResult {
            name,
            pass: true,
            err: None,
        },
        Err(e) => OpResult {
            name,
            pass: false,
            err: Some(e),
        },
    }
}

#[test]
fn f16_op_sweep_gpu() {
    ensure_cuda_backend();

    let n = 8usize;
    let a_src: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32) * 0.1).collect();
    let b_src: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.05).collect();
    let signed_src: Vec<f32> = vec![
        -2.0,
        -1.25,
        -0.0,
        0.0,
        0.5,
        1.0,
        -3.5,
        4.0,
        f32::NAN,
        f32::from_bits(0xffc0_0000),
    ];

    let mut results = Vec::new();

    // ── Elementwise binary ──────────────────────────────────────────────
    results.push(run("add", || {
        let a = f16_cuda(&a_src, &[n]);
        let b = f16_cuda(&b_src, &[n]);
        let out = (a + b)?;
        let r = a_src.iter().zip(&b_src).map(|(x, y)| x + y).collect();
        Ok((out, r))
    }));
    results.push(run("sub", || {
        let a = f16_cuda(&a_src, &[n]);
        let b = f16_cuda(&b_src, &[n]);
        let out = (a - b)?;
        let r = a_src.iter().zip(&b_src).map(|(x, y)| x - y).collect();
        Ok((out, r))
    }));
    results.push(run("mul", || {
        let a = f16_cuda(&a_src, &[n]);
        let b = f16_cuda(&b_src, &[n]);
        let out = (a * b)?;
        let r = a_src.iter().zip(&b_src).map(|(x, y)| x * y).collect();
        Ok((out, r))
    }));
    results.push(run("div", || {
        let a = f16_cuda(&a_src, &[n]);
        let b = f16_cuda(&b_src, &[n]);
        let out = (a / b)?;
        let r = a_src.iter().zip(&b_src).map(|(x, y)| x / y).collect();
        Ok((out, r))
    }));
    results.push(run("neg", || {
        let a = f16_cuda(&a_src, &[n]);
        let out = (-a)?;
        let r = a_src.iter().map(|x| -x).collect();
        Ok((out, r))
    }));
    results.push(run("abs", || {
        let a = f16_cuda(&signed_src, &[signed_src.len()]);
        let out = ferrotorch_core::grad_fns::arithmetic::abs(&a)?;
        let r = signed_src.iter().map(|x| x.abs()).collect();
        Ok((out, r))
    }));
    results.push(run("abs_backward", || {
        let a = f16_cuda(&signed_src, &[signed_src.len()]).requires_grad_(true);
        let out = ferrotorch_core::grad_fns::arithmetic::abs(&a)?;
        let loss = ferrotorch_core::grad_fns::reduction::sum(&out)?;
        ferrotorch_core::autograd::graph::backward(&loss)?;
        let grad = a.grad()?.ok_or_else(|| FerrotorchError::Internal {
            message: "abs_backward did not populate f16 CUDA grad".into(),
        })?;
        let r = signed_src
            .iter()
            .map(|&x| {
                if x.is_nan() {
                    0.0
                } else if x > 0.0 {
                    1.0
                } else if x < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            })
            .collect();
        Ok((grad, r))
    }));

    // ── Broadcast: [1, n] (op) [n, 1] -> [n, n] ──────────────────────────
    results.push(run("broadcast_add", || {
        let a_row = f16_cuda(&a_src, &[1, n]);
        let b_col = f16_cuda(&b_src, &[n, 1]);
        let out = (a_row + b_col)?;
        let mut r = Vec::with_capacity(n * n);
        for &bi in &b_src {
            for &aj in &a_src {
                r.push(aj + bi);
            }
        }
        Ok((out, r))
    }));

    // ── Reductions ──────────────────────────────────────────────────────
    results.push(run("sum", || {
        let a = f16_cuda(&a_src, &[n]);
        let out = ferrotorch_core::grad_fns::reduction::sum(&a)?;
        let r = vec![a_src.iter().sum()];
        Ok((out, r))
    }));
    results.push(run("mean", || {
        let a = f16_cuda(&a_src, &[n]);
        let out = ferrotorch_core::grad_fns::reduction::mean(&a)?;
        let r = vec![a_src.iter().sum::<f32>() / n as f32];
        Ok((out, r))
    }));
    results.push(run("sum_dim", || {
        // [4, 2] -> sum over dim 0 -> [2]
        let mat_src: Vec<f32> = (0..8).map(|i| 0.2 + (i as f32) * 0.1).collect();
        let mat = f16_cuda(&mat_src, &[4, 2]);
        let out = mat.sum_dim(0, false)?;
        let mut r = vec![0.0f32; 2];
        for i in 0..4 {
            for j in 0..2 {
                r[j] += mat_src[i * 2 + j];
            }
        }
        Ok((out, r))
    }));

    // ── Transcendental / activation unary ───────────────────────────────
    results.push(run("exp", || {
        let a = f16_cuda(&a_src, &[n]);
        let out = ferrotorch_core::grad_fns::transcendental::exp(&a)?;
        let r = a_src.iter().map(|x| x.exp()).collect();
        Ok((out, r))
    }));
    results.push(run("log", || {
        let a = f16_cuda(&a_src, &[n]);
        let out = ferrotorch_core::grad_fns::transcendental::log(&a)?;
        let r = a_src.iter().map(|x| x.ln()).collect();
        Ok((out, r))
    }));
    results.push(run("sqrt", || {
        // sqrt has no direct f16 dispatch arm; exercise it via the public
        // Tensor API which decomposes to on-device primitives.
        let a = f16_cuda(&a_src, &[n]);
        let out = a.sqrt_t()?;
        let r = a_src.iter().map(|x| x.sqrt()).collect();
        Ok((out, r))
    }));
    results.push(run("atan", || {
        let a = f16_cuda(&signed_src, &[signed_src.len()]);
        let out = ferrotorch_core::grad_fns::transcendental::atan(&a)?;
        let r = signed_src.iter().map(|x| x.atan()).collect();
        Ok((out, r))
    }));
    results.push(run("ceil", || {
        let a = f16_cuda(&signed_src, &[signed_src.len()]);
        let out = ferrotorch_core::grad_fns::transcendental::ceil(&a)?;
        let r = signed_src.iter().map(|x| x.ceil()).collect();
        Ok((out, r))
    }));
    results.push(run("floor", || {
        let a = f16_cuda(&signed_src, &[signed_src.len()]);
        let out = ferrotorch_core::grad_fns::transcendental::floor(&a)?;
        let r = signed_src.iter().map(|x| x.floor()).collect();
        Ok((out, r))
    }));
    results.push(run("round", || {
        let src = [-2.5, -1.5, -0.5, -0.0, 0.0, 0.5, 1.5, 2.5, f32::NAN];
        let a = f16_cuda(&src, &[src.len()]);
        let out = ferrotorch_core::grad_fns::transcendental::round(&a)?;
        let r = src.iter().copied().map(round_ties_even).collect();
        Ok((out, r))
    }));
    results.push(run("trunc", || {
        let a = f16_cuda(&signed_src, &[signed_src.len()]);
        let out = ferrotorch_core::grad_fns::transcendental::trunc(&a)?;
        let r = signed_src.iter().map(|x| x.trunc()).collect();
        Ok((out, r))
    }));
    results.push(run("frac", || {
        let a = f16_cuda(&signed_src, &[signed_src.len()]);
        let out = ferrotorch_core::grad_fns::transcendental::frac(&a)?;
        let r = signed_src.iter().map(|x| x - x.trunc()).collect();
        Ok((out, r))
    }));
    results.push(run("sign", || {
        let a = f16_cuda(&signed_src, &[signed_src.len()]);
        let out = ferrotorch_core::grad_fns::transcendental::sign(&a)?;
        let r = signed_src
            .iter()
            .map(|&x| {
                if x.is_nan() || x == 0.0 {
                    0.0
                } else {
                    x.signum()
                }
            })
            .collect();
        Ok((out, r))
    }));
    results.push(run("tanh", || {
        let a = f16_cuda(&a_src, &[n]);
        let out = ferrotorch_core::grad_fns::activation::tanh(&a)?;
        let r = a_src.iter().map(|x| x.tanh()).collect();
        Ok((out, r))
    }));
    results.push(run("sigmoid", || {
        let a = f16_cuda(&a_src, &[n]);
        let out = ferrotorch_core::grad_fns::activation::sigmoid(&a)?;
        let r = a_src.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
        Ok((out, r))
    }));
    results.push(run("relu", || {
        // Mix of negative and positive inputs to exercise the clamp.
        let src: Vec<f32> = (0..n).map(|i| (i as f32) - 4.0).collect();
        let a = f16_cuda(&src, &[n]);
        let out = ferrotorch_core::grad_fns::activation::relu(&a)?;
        let r = src.iter().map(|x| x.max(0.0)).collect();
        Ok((out, r))
    }));
    results.push(run("gelu", || {
        let a = f16_cuda(&a_src, &[n]);
        let out = ferrotorch_core::grad_fns::activation::gelu(&a)?;
        let r = a_src
            .iter()
            .map(|x| 0.5 * x * (1.0 + libm_erf(x / 2.0_f32.sqrt())))
            .collect();
        Ok((out, r))
    }));
    results.push(run("gelu_tanh", || {
        let a = f16_cuda(&a_src, &[n]);
        let out = ferrotorch_core::grad_fns::activation::gelu_with(&a, GeluApproximate::Tanh)?;
        let sqrt_2_over_pi = (2.0_f32 / std::f32::consts::PI).sqrt();
        let r = a_src
            .iter()
            .map(|x| 0.5 * x * (1.0 + (sqrt_2_over_pi * (x + 0.044715 * x * x * x)).tanh()))
            .collect();
        Ok((out, r))
    }));
    results.push(run("gelu_sigmoid", || {
        let a = f16_cuda(&a_src, &[n]);
        let out = ferrotorch_core::grad_fns::activation::gelu_with(&a, GeluApproximate::Sigmoid)?;
        let r = a_src
            .iter()
            .map(|x| x / (1.0 + (-1.702 * x).exp()))
            .collect();
        Ok((out, r))
    }));
    results.push(run("silu", || {
        let a = f16_cuda(&a_src, &[n]);
        let out = ferrotorch_core::grad_fns::activation::silu(&a)?;
        let r = a_src.iter().map(|x| x / (1.0 + (-x).exp())).collect();
        Ok((out, r))
    }));

    // ── Softmax (row-wise over last dim) ─────────────────────────────────
    results.push(run("softmax", || {
        let x = f16_cuda(&a_src, &[1, n]);
        let out = ferrotorch_core::grad_fns::activation::softmax(&x)?;
        let mx = a_src.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = a_src.iter().map(|v| (v - mx).exp()).collect();
        let s: f32 = exps.iter().sum();
        let r = exps.iter().map(|e| e / s).collect();
        Ok((out, r))
    }));

    // ── Matmul (resident f16, cuBLAS GemmEx CUDA_R_16F, f32 compute) ─────
    results.push(run("matmul", || {
        // A: [2,3], B: [3,2].
        let a_m: Vec<f32> = (0..6).map(|i| 0.1 * (i as f32 + 1.0)).collect();
        let b_m: Vec<f32> = (0..6).map(|i| 0.2 * (i as f32 + 1.0)).collect();
        let a = f16_cuda(&a_m, &[2, 3]);
        let b = f16_cuda(&b_m, &[3, 2]);
        let out = a.matmul(&b)?;
        // f32 reference C = A @ B, row-major.
        let mut r = vec![0.0f32; 4];
        for i in 0..2 {
            for j in 0..2 {
                let mut acc = 0.0f32;
                for kk in 0..3 {
                    acc += a_m[i * 3 + kk] * b_m[kk * 2 + j];
                }
                r[i * 2 + j] = acc;
            }
        }
        Ok((out, r))
    }));

    // ── Report ──────────────────────────────────────────────────────────
    let pass = results.iter().filter(|r| r.pass).count();
    let fail = results.iter().filter(|r| !r.pass).count();
    let total = results.len();

    println!("\n========== f16 op sweep (Phase 1, #1185) ==========");
    println!("(each PASS proves: GPU-resident output + value parity vs f32)");
    for r in &results {
        if r.pass {
            println!("  PASS: {} [is_cuda + Cuda(0) + values ok]", r.name);
        } else {
            println!("  FAIL: {} — {}", r.name, r.err.as_deref().unwrap_or("<?>"));
        }
    }
    println!("===================================================");
    println!("PASS: {pass}, FAIL: {fail}, TOTAL: {total}");
    println!("===================================================\n");

    assert_eq!(
        fail, 0,
        "f16 op sweep had {fail} failures (see table above)"
    );
}

/// Minimal erf for the GELU reference (Abramowitz & Stegun 7.1.26, the same
/// Hastings polynomial the kernel uses, so the reference matches the kernel
/// to well within f16 tolerance). Computed in f64 to carry the published
/// coefficients without truncation, then narrowed for the f32 caller.
fn libm_erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = f64::from(x.abs());
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    (sign * y) as f32
}

fn round_ties_even(x: f32) -> f32 {
    if !x.is_finite() {
        return x;
    }
    let lo = x.floor();
    let frac = x - lo;
    if frac < 0.5 {
        lo
    } else if frac > 0.5 {
        lo + 1.0
    } else if (lo as i64) % 2 == 0 {
        lo
    } else {
        lo + 1.0
    }
}
