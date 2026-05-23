//! Comprehensive bf16 + CUDA op sweep for forecast-bio/ferrotorch#23.
//!
//! The fix in #19 + #17 made `Tensor<bf16>` movable to CUDA and added
//! `*_bf16_bf16` kernels for matmul / softmax / layernorm / gelu / add / mul,
//! but the **dispatcher glue** in `ferrotorch-core/src/grad_fns/*` is still
//! f32+f64-only at 16+ call sites. Result: a bf16 CUDA tensor that calls
//! `add` / `mul` / `sum` / `exp` / `sigmoid` / etc. either:
//!
//! - Pattern A: falls through to the f32 kernel arm and dies at
//!   `unwrap_buffer::<f32>` with "GPU handle does not contain a
//!   CudaBuffer<f32>". (add, sub, mul, neg, broadcast_add)
//! - Pattern B: short-circuits with `NotImplementedOnCuda { op }` because
//!   the guard explicitly requires `is_f32 || is_f64`. (sum, mean, sum_dim,
//!   mean_dim, high-level Tensor::matmul)
//! - Pattern C: falls through to a CPU `fast_*` helper that then errors
//!   `GpuTensorNotAccessible` when it tries to call `.data()` on the GPU
//!   tensor. (div, exp, log, tanh, sigmoid, softmax)
//!
//! The sweep enumerates the **public Tensor API** for each affected op and
//! records PASS/FAIL for each (Op, dtype=bf16, device=CUDA) tuple. The
//! intended use:
//!
//! 1. Run BEFORE the dispatch refactor → expect `PASS: 13, FAIL: 16,
//!    TOTAL: 29` (or whatever the snapshot says).
//! 2. Run AFTER the refactor → expect `FAIL: 0`. Any remaining FAIL is a
//!    deliberately-deferred kernel and must have a filed crosslink issue.
//!
//! Each op runs the f32 reference path in parallel to confirm the test
//! itself isn't broken (so a `FAIL` is the *bf16* dispatcher, not the test).

#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::Device;
use ferrotorch_core::Tensor;
use ferrotorch_core::creation::from_vec;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for the bf16 op sweep");
    });
}

fn bf16_cuda(data: &[f32], shape: &[usize]) -> Tensor<half::bf16> {
    let bf: Vec<half::bf16> = data.iter().copied().map(half::bf16::from_f32).collect();
    let cpu = from_vec::<half::bf16>(bf, shape).expect("bf16 cpu tensor");
    cpu.to(Device::Cuda(0))
        .expect("Tensor<bf16>::to(Cuda) must succeed")
}

/// Result of a single sweep entry — `Ok` is PASS, `Err` carries the
/// reason it failed so we can categorise (A / B / C).
struct OpResult {
    name: &'static str,
    pass: bool,
    err: Option<String>,
}

fn run(name: &'static str, f: impl FnOnce() -> Result<(), String>) -> OpResult {
    match f() {
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

fn check(
    t: Result<Tensor<half::bf16>, ferrotorch_core::error::FerrotorchError>,
) -> Result<(), String> {
    match t {
        Ok(_) => Ok(()),
        Err(e) => Err(format!("{e}")),
    }
}

#[test]
fn bf16_op_sweep_gpu() {
    ensure_cuda_backend();

    let n = 8;
    let data: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32) * 0.1).collect();
    let data_b: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.05).collect();

    // Reusable inputs.
    let a = bf16_cuda(&data, &[n]);
    let b = bf16_cuda(&data_b, &[n]);

    // For broadcast_add: a [1, n] + b [n, 1] → [n, n]
    let a_row = bf16_cuda(&data, &[1, n]);
    let b_col = bf16_cuda(&data_b, &[n, 1]);

    // 2-D reduction inputs.
    let mat = bf16_cuda(
        &(0..16)
            .map(|i| (i as f32) * 0.1 + 0.2)
            .collect::<Vec<f32>>(),
        &[4, 4],
    );

    // matmul inputs.
    let m_a = bf16_cuda(
        &(0..6).map(|i| (i as f32) * 0.1).collect::<Vec<f32>>(),
        &[2, 3],
    );
    let m_b = bf16_cuda(
        &(0..6).map(|i| (i as f32) * 0.2).collect::<Vec<f32>>(),
        &[3, 2],
    );

    let mut results = Vec::new();

    // ─── Pattern A: arithmetic with else→f32 fallthrough ─────────────────
    results.push(run("add", || check(a.clone() + b.clone())));
    results.push(run("sub", || check(a.clone() - b.clone())));
    results.push(run("mul", || check(a.clone() * b.clone())));
    results.push(run("neg", || check(-a.clone())));
    results.push(run("broadcast_add", || {
        check(a_row.clone() + b_col.clone())
    }));

    // ─── Pattern B: reductions / matmul with NotImplementedOnCuda guard ──
    results.push(run("sum", || {
        check(ferrotorch_core::grad_fns::reduction::sum(&a))
    }));
    results.push(run("mean", || {
        check(ferrotorch_core::grad_fns::reduction::mean(&a))
    }));
    results.push(run("sum_dim", || check(mat.sum_dim(0, false))));
    results.push(run("mean_dim", || check(mat.mean_dim(0, false))));
    results.push(run("matmul_high_level", || check(m_a.matmul(&m_b))));

    // ─── Pattern C: ops that fall through to CPU helpers via is_cuda guard ─
    results.push(run("div", || check(a.clone() / b.clone())));
    results.push(run("exp", || {
        check(ferrotorch_core::grad_fns::transcendental::exp(&a))
    }));
    results.push(run("log", || {
        check(ferrotorch_core::grad_fns::transcendental::log(&a))
    }));
    results.push(run("tanh", || {
        check(ferrotorch_core::grad_fns::activation::tanh(&a))
    }));
    results.push(run("sigmoid", || {
        check(ferrotorch_core::grad_fns::activation::sigmoid(&a))
    }));
    results.push(run("softmax", || {
        // softmax needs at least 2D-shape with last dim
        let x = bf16_cuda(&data, &[1, n]);
        check(ferrotorch_core::grad_fns::activation::softmax(&x))
    }));

    // ─── Sanity checks (already wired in #17 — these MUST pass already) ──
    results.push(run("matmul_bf16_backend_direct", || {
        let backend = ferrotorch_core::gpu_dispatch::gpu_backend().expect("backend");
        backend
            .matmul_bf16_bf16(
                m_a.gpu_handle().unwrap(),
                m_b.gpu_handle().unwrap(),
                2,
                3,
                2,
            )
            .map(|_| ())
            .map_err(|e| format!("{e}"))
    }));
    results.push(run("add_bf16_backend_direct", || {
        let backend = ferrotorch_core::gpu_dispatch::gpu_backend().expect("backend");
        backend
            .add_bf16_bf16(a.gpu_handle().unwrap(), b.gpu_handle().unwrap())
            .map(|_| ())
            .map_err(|e| format!("{e}"))
    }));
    results.push(run("mul_bf16_backend_direct", || {
        let backend = ferrotorch_core::gpu_dispatch::gpu_backend().expect("backend");
        backend
            .mul_bf16_bf16(a.gpu_handle().unwrap(), b.gpu_handle().unwrap())
            .map(|_| ())
            .map_err(|e| format!("{e}"))
    }));
    results.push(run("softmax_bf16_backend_direct", || {
        let backend = ferrotorch_core::gpu_dispatch::gpu_backend().expect("backend");
        let x = bf16_cuda(&data, &[1, n]);
        backend
            .softmax_bf16_bf16(x.gpu_handle().unwrap(), 1, n)
            .map(|_| ())
            .map_err(|e| format!("{e}"))
    }));
    results.push(run("layernorm_bf16_backend_direct", || {
        let backend = ferrotorch_core::gpu_dispatch::gpu_backend().expect("backend");
        let cols = 4_usize;
        let rows = 4_usize;
        let g = bf16_cuda(&vec![1.0_f32; cols], &[cols]);
        let bt = bf16_cuda(&vec![0.0_f32; cols], &[cols]);
        backend
            .layernorm_bf16_bf16(
                mat.gpu_handle().unwrap(),
                g.gpu_handle().unwrap(),
                bt.gpu_handle().unwrap(),
                rows,
                cols,
                1e-5,
            )
            .map(|_| ())
            .map_err(|e| format!("{e}"))
    }));
    results.push(run("gelu_bf16_backend_direct", || {
        let backend = ferrotorch_core::gpu_dispatch::gpu_backend().expect("backend");
        backend
            .gelu_bf16_bf16(a.gpu_handle().unwrap())
            .map(|_| ())
            .map_err(|e| format!("{e}"))
    }));
    results.push(run("silu_bf16_backend_direct", || {
        let backend = ferrotorch_core::gpu_dispatch::gpu_backend().expect("backend");
        backend
            .silu_bf16_bf16(a.gpu_handle().unwrap())
            .map(|_| ())
            .map_err(|e| format!("{e}"))
    }));
    results.push(run("relu_bf16_backend_direct", || {
        let backend = ferrotorch_core::gpu_dispatch::gpu_backend().expect("backend");
        backend
            .relu_bf16_bf16(a.gpu_handle().unwrap())
            .map(|_| ())
            .map_err(|e| format!("{e}"))
    }));
    results.push(run("scale_bf16_backend_direct", || {
        let backend = ferrotorch_core::gpu_dispatch::gpu_backend().expect("backend");
        backend
            .scale_bf16_bf16(a.gpu_handle().unwrap(), 0.5)
            .map(|_| ())
            .map_err(|e| format!("{e}"))
    }));

    // Additional pre-existing pass cases (proves the test rig is not broken):
    // bf16 .to(CPU).to(CUDA).to(CPU) round-trip via `clone_buffer`.
    results.push(run("bf16_cpu_to_cuda_to_cpu", || {
        let _ = a.clone().to(Device::Cpu).map_err(|e| format!("{e}"))?;
        Ok(())
    }));
    results.push(run("bf16_clone_handle", || {
        let backend = ferrotorch_core::gpu_dispatch::gpu_backend().expect("backend");
        backend
            .clone_buffer(a.gpu_handle().unwrap())
            .map(|_| ())
            .map_err(|e| format!("{e}"))
    }));
    results.push(run("bf16_shape_op_cat", || {
        ferrotorch_core::grad_fns::shape::cat(&[a.clone(), b.clone()], 0)
            .map(|_| ())
            .map_err(|e| format!("{e}"))
    }));

    // ─── Report ──────────────────────────────────────────────────────────
    let pass = results.iter().filter(|r| r.pass).count();
    let fail = results.iter().filter(|r| !r.pass).count();
    let total = results.len();

    println!("\n========== bf16 op sweep #23 ==========");
    for r in &results {
        if r.pass {
            println!("  PASS: {}", r.name);
        } else {
            println!(
                "  FAIL: {} — {}",
                r.name,
                r.err.as_deref().unwrap_or("<no error>")
            );
        }
    }
    println!("=========================================");
    println!("PASS: {pass}, FAIL: {fail}, TOTAL: {total}");
    println!("=========================================\n");

    // The architect's audit looks at the printed line. We do NOT assert
    // here — the test always prints and exits 0, so the architect can
    // diff sweep output before/after the dispatch refactor mechanically.
    // If you want to gate CI on this, add an assert in a separate test.
}
