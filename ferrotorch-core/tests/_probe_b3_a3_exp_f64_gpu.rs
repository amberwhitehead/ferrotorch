//! Permanent regression sentinel for #797:
//! `exp_f64_kernel` PTX JIT compilation failure.
//!
//! Pre-fix: `gpu_exp_f64` returns `Err(GpuError::PtxCompileFailed)` because
//! `EXP_F64_PTX` writes to `%ln2_hi` / `%ln2_lo` registers that are never
//! declared in the kernel's `.reg` block — `ptxas` rejects the module at
//! load time with `CUDA_ERROR_INVALID_PTX`. (Bonus observation noted in
//! commit 1656a7e9 / #781 that this dispatch closes.)
//!
//! Post-fix: every test in this file passes; the PTX JIT compiles and the
//! kernel produces values within `F64_TRANSCENDENTAL = 1e-10` tolerance of
//! `f64::exp` reference across multiple shapes (1, 4, 256, 4096) and edge
//! values (large positive, large negative, near-overflow, near-zero).
//!
//! This file is committed permanently — it is the regression sentinel
//! that prevents the same class of bug (undeclared register reference in a
//! PTX kernel template) from re-emerging silently in `EXP_F64_PTX`.

#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::Device;
use ferrotorch_core::creation::from_vec;
use ferrotorch_core::grad_fns::transcendental::exp;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for the GPU probe suite");
    });
}

/// Workspace tolerance for f64 transcendentals on GPU (matches the
/// `tolerance::F64_TRANSCENDENTAL` constant in conformance_activation).
const F64_TRANSCENDENTAL: f64 = 1e-10;

/// Helper: upload `data` to CUDA, run f64 `exp`, return host-side readback.
/// Pretty-prints the cudarc error chain on failure so the captured stdout
/// names the offending PTX line if JIT fails.
fn exp_f64_or_dump(data: &[f64]) -> Vec<f64> {
    let cpu = from_vec::<f64>(data.to_vec(), &[data.len()]).expect("cpu tensor");
    let gpu = cpu.to(Device::Cuda(0)).expect("cpu->gpu");
    let y = match exp(&gpu) {
        Ok(t) => t,
        Err(e) => panic!("gpu exp(f64) failed: {e:#?}"),
    };
    let y_cpu = y.cpu().expect("gpu->cpu");
    y_cpu.data().expect("readback").to_vec()
}

fn assert_within_tol(label: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    let mut max_abs = 0.0_f64;
    let mut max_rel = 0.0_f64;
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let abs_err = (g - w).abs();
        let rel_err = abs_err / w.abs().max(1e-300);
        max_abs = max_abs.max(abs_err);
        max_rel = max_rel.max(rel_err);
        let ok = abs_err < F64_TRANSCENDENTAL || rel_err < F64_TRANSCENDENTAL;
        assert!(
            ok,
            "{label}[{i}]: got {g} want {w} abs_err={abs_err:.3e} rel_err={rel_err:.3e}",
        );
    }
    println!(
        "  {label}: max_abs={max_abs:.3e} max_rel={max_rel:.3e} (tol={F64_TRANSCENDENTAL:.0e})"
    );
}

/// Smoke: shape=1, simple value. Pre-fix this fails at PTX module load.
#[test]
fn exp_f64_jit_compiles_n1() {
    ensure_cuda_backend();
    let got = exp_f64_or_dump(&[1.0]);
    let want = [1.0_f64.exp()];
    assert_within_tol("exp_f64 n=1", &got, &want);
}

/// Small shape — exercises the basic launch geometry.
#[test]
fn exp_f64_jit_compiles_n4() {
    ensure_cuda_backend();
    let input = [-1.0_f64, 0.0, 1.0, 2.0];
    let got = exp_f64_or_dump(&input);
    let want: Vec<f64> = input.iter().map(|x| x.exp()).collect();
    assert_within_tol("exp_f64 n=4", &got, &want);
}

/// Mid shape — covers a single-block-with-tail launch.
#[test]
fn exp_f64_n256() {
    ensure_cuda_backend();
    // Linear sweep across [-5, 5] — covers normal NN range.
    let input: Vec<f64> = (0..256).map(|i| -5.0 + 10.0 * (i as f64) / 255.0).collect();
    let got = exp_f64_or_dump(&input);
    let want: Vec<f64> = input.iter().map(|x| x.exp()).collect();
    assert_within_tol("exp_f64 n=256 [-5,5]", &got, &want);
}

/// Larger shape — multi-block launch.
#[test]
fn exp_f64_n4096() {
    ensure_cuda_backend();
    // Sweep across [-20, 20] — exercises a wider input range (still well
    // away from f64 exp overflow at ~709).
    let input: Vec<f64> = (0..4096)
        .map(|i| -20.0 + 40.0 * (i as f64) / 4095.0)
        .collect();
    let got = exp_f64_or_dump(&input);
    let want: Vec<f64> = input.iter().map(|x| x.exp()).collect();
    assert_within_tol("exp_f64 n=4096 [-20,20]", &got, &want);
}

/// Edge values — large positive (just below overflow), large negative
/// (deeply underflow), near zero (smallest perturbations), exact integers.
#[test]
fn exp_f64_edge_values() {
    ensure_cuda_backend();
    let input = [
        700.0,  // near overflow ceiling (exp(709.78) overflows f64)
        -700.0, // deeply underflow but still representable
        0.0,    // exp(0) = 1 exactly
        -0.0,   // exp(-0) = 1 exactly
        1e-15,  // near zero — Taylor regime
        -1e-15,
        1.0,  // exp(1) = e
        -1.0, // exp(-1) = 1/e
        10.0,
        -10.0,
        50.0,
        -50.0,
        std::f64::consts::LN_2, // exp(ln 2) = 2 exactly
        709.0,                  // very close to overflow ceiling
    ];
    let got = exp_f64_or_dump(&input);
    let want: Vec<f64> = input.iter().map(|x| x.exp()).collect();
    assert_within_tol("exp_f64 edge values", &got, &want);
}

/// Identities at exact inputs: exp(0) == 1.0 (bit-exact), exp(ln 2) == 2.0
/// (within tol), confirming the inline polynomial respects the canonical
/// identity inputs.
#[test]
fn exp_f64_identities() {
    ensure_cuda_backend();
    let got = exp_f64_or_dump(&[0.0, std::f64::consts::LN_2]);
    assert_eq!(got[0], 1.0, "exp(0.0) must be exactly 1.0");
    assert!(
        (got[1] - 2.0).abs() < F64_TRANSCENDENTAL,
        "exp(ln 2) = {} (want 2.0)",
        got[1],
    );
}
