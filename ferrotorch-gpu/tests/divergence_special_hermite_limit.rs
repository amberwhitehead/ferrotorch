//! Discriminator probes for the orthogonal-polynomial GPU kernels landed in
//! commit b854b7398 (`ferrotorch-gpu/src/special.rs`, #1545 / #1533).
//!
//! Two kinds of test live here:
//!
//!   * `divergence_*` — FAILING tests that pin a real divergence between the
//!     ferrotorch GPU/CPU recurrence and live PyTorch (and between the two
//!     ferrotorch backends). `#[ignore]`d with a tracking issue.
//!
//!   * `verify_*` — PASSING positive controls that pin the GPU f64 path to
//!     LIVE PyTorch reference values (not just the ferrotorch CPU path), so
//!     the f64 kernel is proven genuinely on-device AND torch-correct on the
//!     normal domain to 1e-12 (R-CHAR-3: expected values are torch outputs).
//!
//! Reference values were produced by `torch 2.11.0+cu130`:
//!   torch.special.legendre_polynomial_p / hermite_polynomial_he / _h on
//!   f64 and f32.

#![cfg(feature = "cuda")]

use std::sync::Once;

use ferrotorch_core::{Device, Tensor, TensorStorage, special};
use ferrotorch_gpu::init_cuda_backend;

fn ensure_cuda() -> bool {
    static INIT: Once = Once::new();
    static mut OK: bool = false;
    if ferrotorch_gpu::device::GpuDevice::new(0).is_err() {
        return false;
    }
    INIT.call_once(|| {
        init_cuda_backend().expect("CUDA backend init");
        unsafe { OK = true }
    });
    unsafe { OK }
}

fn cpu_f32(data: &[f32]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false).unwrap()
}

fn cpu_f64(data: &[f64]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false).unwrap()
}

// =====================================================================
// DIVERGENCE 1 (backend-internal): GPU path != CPU path for the same op.
//
// The commit message asserts the GPU path is "bit-for-relevant-tolerance
// identical to the ferrotorch CPU path". At n above the f32 hermitian limit
// (128) this is FALSE: the CPU path runs the recurrence in f64
// (`elementwise_f64` in `ferrotorch-core/src/special.rs:1199`) then narrows
// to f32 — an f64 overflow narrows to ±inf; the GPU f32 kernel
// (`HERMITE_H_F32_PTX` in `ferrotorch-gpu/src/special.rs:260`) runs the
// recurrence in f32 registers and the f32 overflow yields NaN (inf - inf).
// Observed live on RTX 3090: GPU = NaN, CPU = inf, for the SAME op / dtype /
// input. This is a silent CPU/GPU divergence the commit explicitly denies.
// =====================================================================

/// Divergence: `hermite_polynomial_h` GPU(f32-register) result disagrees
/// with CPU(f64-then-narrow) result for n above the f32 hermitian limit.
/// GPU returns NaN; CPU returns +inf for `hermite_polynomial_h(0.05, 200)`
/// on f32. The commit b854b7398 claims the two backends are bit-identical.
///
/// Tracking: #1641.
#[test]
fn divergence_hermite_h_gpu_cpu_disagree_above_limit() {
    if !ensure_cuda() {
        return;
    }
    let n = 200usize;
    let cpu_in = cpu_f32(&[0.05]);
    let cpu_val = special::hermite_polynomial_h(&cpu_in, n)
        .unwrap()
        .data()
        .unwrap()[0];
    let gpu_in = cpu_in.to(Device::Cuda(0)).expect("to GPU");
    let gpu_val = special::hermite_polynomial_h(&gpu_in, n)
        .unwrap()
        .to(Device::Cpu)
        .unwrap()
        .data()
        .unwrap()[0];
    // The commit claims GPU == CPU. Pin that claim; it is false here.
    assert!(
        (gpu_val.is_nan() && cpu_val.is_nan()) || gpu_val == cpu_val,
        "hermite_polynomial_h(0.05, n=200) f32: GPU={gpu_val} CPU={cpu_val} \
         disagree (commit b854b7398 claims bit-identical GPU/CPU)"
    );
}

// =====================================================================
// DIVERGENCE 2 (vs torch): the ferrotorch CPU path lacks torch's
// getHermitianLimit NaN guard (`Math.h:3044-3052`, used at :3068 / :3109).
// torch.special.hermite_polynomial_h(0.05, 200) f32 == NaN; the ferrotorch
// CPU path returns +inf (f64 overflow narrowed to f32). inf != NaN — an
// observable divergence (isnan masks differ).
// =====================================================================

/// Divergence: ferrotorch CPU `hermite_polynomial_h` returns +inf where
/// `pytorch aten/src/ATen/native/Math.h:3068` returns NaN for n above
/// `getHermitianLimit<float>() == 128` (`Math.h:3044-3052`). Verified live:
/// `torch.special.hermite_polynomial_h(tensor([0.05], f32), 200) == NaN`.
///
/// Tracking: #1641.
#[test]
fn divergence_hermite_h_cpu_vs_torch_above_limit() {
    if !ensure_cuda() {
        return;
    }
    let cpu_val = special::hermite_polynomial_h(&cpu_f32(&[0.05]), 200)
        .unwrap()
        .data()
        .unwrap()[0];
    // torch returns NaN; ferrotorch CPU returns +inf.
    assert!(
        cpu_val.is_nan(),
        "hermite_polynomial_h(0.05, 200) f32 CPU: torch returns NaN \
         (Math.h:3068 getHermitianLimit); ferrotorch returned {cpu_val}"
    );
}

/// Divergence: same `getHermitianLimit` gap for probabilist's Hermite
/// `hermite_polynomial_he` — `pytorch aten/src/ATen/native/Math.h:3109`.
/// Verified live: `torch.special.hermite_polynomial_he(tensor([0.05], f32),
/// 200) == NaN`; ferrotorch CPU returns ±inf.
///
/// Tracking: #1641.
#[test]
fn divergence_hermite_he_cpu_vs_torch_above_limit() {
    if !ensure_cuda() {
        return;
    }
    let cpu_val = special::hermite_polynomial_he(&cpu_f32(&[0.05]), 200)
        .unwrap()
        .data()
        .unwrap()[0];
    assert!(
        cpu_val.is_nan(),
        "hermite_polynomial_he(0.05, 200) f32 CPU: torch returns NaN \
         (Math.h:3109 getHermitianLimit); ferrotorch returned {cpu_val}"
    );
}

// =====================================================================
// POSITIVE CONTROL: GPU f64 path matches LIVE TORCH (not just CPU) to 1e-12.
// Proves the f64 kernel is genuinely on-device AND torch-correct, and is
// NOT silently downcast to f32 (an f32 path could not reach 1e-12 at n=20+).
// =====================================================================

/// Verify `legendre_polynomial_p` GPU f64 matches torch f64 to 1e-12 on the
/// normal domain. Expected values are LIVE torch 2.11.0 outputs:
///   P_5(0.3)   = 0.34538625
///   P_8(0.65)  = 0.30323855787200926
///   P_12(-0.4) = 0.09948820927600006
///   P_20(0.9)  = -0.14930823530984821
/// (`torch.special.legendre_polynomial_p`, dtype=float64).
#[test]
fn verify_legendre_p_gpu_f64_matches_torch_1e12() {
    if !ensure_cuda() {
        return;
    }
    let cases: [(usize, f64, f64); 4] = [
        (5, 0.3, 0.34538625),
        (8, 0.65, 0.30323855787200926),
        (12, -0.4, 0.09948820927600006),
        (20, 0.9, -0.14930823530984821),
    ];
    for (n, x, torch_val) in cases {
        let gpu_in = cpu_f64(&[x]).to(Device::Cuda(0)).expect("to GPU");
        let gpu_out = special::legendre_polynomial_p(&gpu_in, n).unwrap();
        assert!(
            gpu_out.is_cuda(),
            "legendre f64 GPU out must stay on device"
        );
        let got = gpu_out.to(Device::Cpu).unwrap().data().unwrap()[0];
        assert!(
            (got - torch_val).abs() <= 1e-12,
            "legendre_p GPU f64 n={n} x={x}: got {got}, torch {torch_val}, \
             delta {}",
            (got - torch_val).abs()
        );
    }
}

/// Verify `hermite_polynomial_he` GPU f64 matches torch f64 on the normal
/// domain (below the hermitian limit). Expected from LIVE torch 2.11.0:
///   He_6(0.5)  = -4.671875
///   He_10(1.2) = 1021.0298932224002
/// (`torch.special.hermite_polynomial_he`, dtype=float64).
#[test]
fn verify_hermite_he_gpu_f64_matches_torch() {
    if !ensure_cuda() {
        return;
    }
    let cases: [(usize, f64, f64); 2] = [(6, 0.5, -4.671875), (10, 1.2, 1021.0298932224002)];
    for (n, x, torch_val) in cases {
        let gpu_in = cpu_f64(&[x]).to(Device::Cuda(0)).expect("to GPU");
        let gpu_out = special::hermite_polynomial_he(&gpu_in, n).unwrap();
        assert!(gpu_out.is_cuda());
        let got = gpu_out.to(Device::Cpu).unwrap().data().unwrap()[0];
        assert!(
            (got - torch_val).abs() <= 1e-9 * (1.0 + torch_val.abs()),
            "hermite_he GPU f64 n={n} x={x}: got {got}, torch {torch_val}"
        );
    }
}

/// Verify the f64 GPU kernel is genuinely f64 and NOT silently downcast to
/// f32: a high-order Legendre value where the f32 recurrence would lose far
/// more than 1e-12. Expected from LIVE torch 2.11.0:
///   torch.special.legendre_polynomial_p(
///       torch.tensor([0.123], dtype=torch.float64), 25).item()
///   == -0.0005514865833485101
#[test]
fn verify_f64_not_downcast_high_order() {
    if !ensure_cuda() {
        return;
    }
    let torch_val = -0.0005514865833485101_f64;
    let gpu_in = cpu_f64(&[0.123]).to(Device::Cuda(0)).expect("to GPU");
    let got = special::legendre_polynomial_p(&gpu_in, 25)
        .unwrap()
        .to(Device::Cpu)
        .unwrap()
        .data()
        .unwrap()[0];
    assert!(
        (got - torch_val).abs() <= 1e-12,
        "legendre_p GPU f64 n=25 x=0.123: got {got}, torch {torch_val} \
         (if this fails by ~1e-7 the f64 path is silently f32)"
    );
}
