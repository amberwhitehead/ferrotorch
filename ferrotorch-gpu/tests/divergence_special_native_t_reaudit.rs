//! Re-audit of commit 7c8b87118 (`ferrotorch-core/src/special.rs`): the CPU
//! orthogonal-polynomial recurrence now runs in NATIVE `T` (was f64-then-narrow
//! via the old `elementwise_f64`). The commit claims "CPU == GPU == torch for
//! all families and both dtypes" (#1642 / #1641, sub-family of #1545).
//!
//! This file pins that claim with LIVE-torch reference values for the families
//! and inputs NOT previously covered by
//! `divergence_special_hermite_limit.rs` (which only exercised hermite_h/he +
//! legendre). Here we add `chebyshev_polynomial_v`, `chebyshev_polynomial_w`,
//! and `laguerre_polynomial_l` at:
//!
//!   * f32 inputs where torch's native-f32 recurrence overflows to NaN
//!     (CPU == GPU == torch == NaN — the bug the commit fixed),
//!   * a finite large-n laguerre case where torch f32 does NOT overflow
//!     (CPU == GPU == torch finite — guards against an over-eager NaN),
//!   * normal-domain f64 to 1e-12 (the native-T change must not hurt f64
//!     precision — f64 IS native for f64 tensors), and
//!   * normal-domain f32 vs torch f32 (the new native-f32 path).
//!
//! R-CHAR-3: every expected value below is a LIVE `torch 2.11.0+cu130` output
//! (`torch.special.{chebyshev_polynomial_v,_w,laguerre_polynomial_l}` and the
//! non-polynomial `erf/erfinv/gammaln/digamma`), NOT copied from ferrotorch.
//!
//! These are PASSING positive controls. If any FAILS, the named family still
//! diverges and the commit's verdict is an overclaim — pin it as a tracked
//! blocker.

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

fn gpu_f32_eval(
    f: impl Fn(&Tensor<f32>, usize) -> ferrotorch_core::FerrotorchResult<Tensor<f32>>,
    x: f32,
    n: usize,
) -> f32 {
    let g = cpu_f32(&[x]).to(Device::Cuda(0)).expect("to GPU");
    let out = f(&g, n).unwrap();
    out.to(Device::Cpu).unwrap().data().unwrap()[0]
}

fn gpu_f64_eval(
    f: impl Fn(&Tensor<f64>, usize) -> ferrotorch_core::FerrotorchResult<Tensor<f64>>,
    x: f64,
    n: usize,
) -> f64 {
    let g = cpu_f64(&[x]).to(Device::Cuda(0)).expect("to GPU");
    let out = f(&g, n).unwrap();
    out.to(Device::Cpu).unwrap().data().unwrap()[0]
}

// =====================================================================
// f32 OVERFLOW -> NaN : CPU == GPU == torch (the families #1642 missed pinning)
//
// LIVE torch 2.11.0+cu130:
//   torch.special.chebyshev_polynomial_v(tensor([1.7], f32), 120) -> nan
//   torch.special.chebyshev_polynomial_v(tensor([2.0], f32),  90) -> nan
//   torch.special.chebyshev_polynomial_w(tensor([1.7], f32), 120) -> nan
//   torch.special.chebyshev_polynomial_w(tensor([2.0], f32),  90) -> nan
//   torch.special.laguerre_polynomial_l(tensor([-30.0], f32), 120) -> nan
// (each torch f64 stays finite ~1e51..1e58 / 6.3e44, so an f64-then-narrow
//  path would return +inf, not NaN — this distinguishes the fix.)
// =====================================================================

/// chebyshev_polynomial_v f32 overflow: CPU == GPU == torch == NaN.
/// Pins commit 7c8b87118 for the V family (the original #1642 pins only
/// covered T/U/hermite/legendre).
#[test]
fn reaudit_cheb_v_f32_overflow_cpu_gpu_torch_nan() {
    if !ensure_cuda() {
        return;
    }
    for (x, n) in [(1.7f32, 120usize), (2.0, 90)] {
        let cpu = special::chebyshev_polynomial_v(&cpu_f32(&[x]), n)
            .unwrap()
            .data()
            .unwrap()[0];
        let gpu = gpu_f32_eval(special::chebyshev_polynomial_v, x, n);
        // torch returns NaN for both cases.
        assert!(
            cpu.is_nan() && gpu.is_nan(),
            "chebyshev_polynomial_v(x={x}, n={n}) f32: torch=NaN, CPU={cpu}, GPU={gpu}"
        );
    }
}

/// chebyshev_polynomial_w f32 overflow: CPU == GPU == torch == NaN.
#[test]
fn reaudit_cheb_w_f32_overflow_cpu_gpu_torch_nan() {
    if !ensure_cuda() {
        return;
    }
    for (x, n) in [(1.7f32, 120usize), (2.0, 90)] {
        let cpu = special::chebyshev_polynomial_w(&cpu_f32(&[x]), n)
            .unwrap()
            .data()
            .unwrap()[0];
        let gpu = gpu_f32_eval(special::chebyshev_polynomial_w, x, n);
        assert!(
            cpu.is_nan() && gpu.is_nan(),
            "chebyshev_polynomial_w(x={x}, n={n}) f32: torch=NaN, CPU={cpu}, GPU={gpu}"
        );
    }
}

/// laguerre_polynomial_l f32 overflow: CPU == GPU == torch == NaN at
/// x=-30.0, n=120 (torch f64 = 6.32e44, so an f64-narrow path returns +inf).
#[test]
fn reaudit_laguerre_f32_overflow_cpu_gpu_torch_nan() {
    if !ensure_cuda() {
        return;
    }
    let (x, n) = (-30.0f32, 120usize);
    let cpu = special::laguerre_polynomial_l(&cpu_f32(&[x]), n)
        .unwrap()
        .data()
        .unwrap()[0];
    let gpu = gpu_f32_eval(special::laguerre_polynomial_l, x, n);
    assert!(
        cpu.is_nan() && gpu.is_nan(),
        "laguerre_polynomial_l(x={x}, n={n}) f32: torch=NaN, CPU={cpu}, GPU={gpu}"
    );
}

/// NOT every large-n laguerre overflows: at x=50.0, n=200 torch f32 stays
/// FINITE (== torch f64 == -3.502024704e9). Native-T must not over-eagerly
/// NaN here — CPU == GPU == torch finite value.
/// LIVE torch: laguerre_polynomial_l(tensor([50.0], f32), 200) == -3502024704.0
#[test]
fn reaudit_laguerre_f32_large_n_finite_cpu_gpu_torch() {
    if !ensure_cuda() {
        return;
    }
    let torch_f32 = -3_502_024_704.0f32;
    let (x, n) = (50.0f32, 200usize);
    let cpu = special::laguerre_polynomial_l(&cpu_f32(&[x]), n)
        .unwrap()
        .data()
        .unwrap()[0];
    let gpu = gpu_f32_eval(special::laguerre_polynomial_l, x, n);
    let tol = 1e-3 * (1.0 + torch_f32.abs());
    assert!(
        (cpu - torch_f32).abs() <= tol && (gpu - torch_f32).abs() <= tol,
        "laguerre_polynomial_l(50.0, 200) f32 should be finite ~{torch_f32}; \
         CPU={cpu}, GPU={gpu}"
    );
}

// =====================================================================
// NORMAL-DOMAIN f64 to 1e-12 : the native-T change must not hurt f64
// precision (f64 IS native for f64 tensors). CPU == GPU == torch f64.
//
// LIVE torch 2.11.0+cu130, dtype=float64:
//   chebyshev_polynomial_v(0.3, 12)  = -1.231673421824
//   chebyshev_polynomial_v(-0.7, 10) =  2.2687420415999946
//   chebyshev_polynomial_w(0.3, 12)  = -0.1995522007040017
//   chebyshev_polynomial_w(0.55, 9)  =  0.07308270100000186
//   laguerre_polynomial_l(0.5, 8)    = -0.498362998356895
//   laguerre_polynomial_l(2.0, 10)   = -0.3090652557319224
// =====================================================================

#[test]
fn reaudit_normal_domain_f64_1e12_cpu_gpu_torch() {
    if !ensure_cuda() {
        return;
    }
    type Fn64 = fn(&Tensor<f64>, usize) -> ferrotorch_core::FerrotorchResult<Tensor<f64>>;
    let cases: [(&str, Fn64, f64, usize, f64); 6] = [
        (
            "cheb_v",
            special::chebyshev_polynomial_v,
            0.3,
            12,
            -1.231673421824,
        ),
        (
            "cheb_v",
            special::chebyshev_polynomial_v,
            -0.7,
            10,
            2.2687420415999946,
        ),
        (
            "cheb_w",
            special::chebyshev_polynomial_w,
            0.3,
            12,
            -0.1995522007040017,
        ),
        (
            "cheb_w",
            special::chebyshev_polynomial_w,
            0.55,
            9,
            0.07308270100000186,
        ),
        (
            "laguerre",
            special::laguerre_polynomial_l,
            0.5,
            8,
            -0.498362998356895,
        ),
        (
            "laguerre",
            special::laguerre_polynomial_l,
            2.0,
            10,
            -0.3090652557319224,
        ),
    ];
    for (name, f, x, n, torch_val) in cases {
        let cpu = f(&cpu_f64(&[x]), n).unwrap().data().unwrap()[0];
        let gpu = gpu_f64_eval(f, x, n);
        assert!(
            (cpu - torch_val).abs() <= 1e-12 && (gpu - torch_val).abs() <= 1e-12,
            "{name}(x={x}, n={n}) f64: torch={torch_val}, CPU={cpu} (d={:.2e}), \
             GPU={gpu} (d={:.2e}) — 1e-12 precision lost",
            (cpu - torch_val).abs(),
            (gpu - torch_val).abs()
        );
    }
}

// =====================================================================
// NORMAL-DOMAIN f32 : the NEW native-f32 path must match torch f32.
//
// LIVE torch 2.11.0+cu130, dtype=float32:
//   chebyshev_polynomial_v(0.3, 6) = -0.45510390400886536
//   laguerre_polynomial_l(0.5, 8)  = -0.4983629882335663
// =====================================================================

#[test]
#[allow(
    clippy::excessive_precision,
    reason = "oracle-derived f32 expected values copied verbatim from live torch 2.11 — \
              full precision is intentional (rounded to f32 at compile time)"
)]
fn reaudit_normal_domain_f32_cpu_gpu_torch() {
    if !ensure_cuda() {
        return;
    }
    type Fn32 = fn(&Tensor<f32>, usize) -> ferrotorch_core::FerrotorchResult<Tensor<f32>>;
    let cases: [(&str, Fn32, f32, usize, f32); 2] = [
        (
            "cheb_v",
            special::chebyshev_polynomial_v,
            0.3,
            6,
            -0.45510390400886536,
        ),
        (
            "laguerre",
            special::laguerre_polynomial_l,
            0.5,
            8,
            -0.4983629882335663,
        ),
    ];
    for (name, f, x, n, torch_val) in cases {
        let cpu = f(&cpu_f32(&[x]), n).unwrap().data().unwrap()[0];
        let gpu = gpu_f32_eval(f, x, n);
        let tol = 1e-5 * (1.0 + torch_val.abs());
        assert!(
            (cpu - torch_val).abs() <= tol && (gpu - torch_val).abs() <= tol,
            "{name}(x={x}, n={n}) f32: torch={torch_val}, CPU={cpu}, GPU={gpu}"
        );
    }
}

// =====================================================================
// NON-POLYNOMIAL fns UNREGRESSED : the commit claims erf/erfinv/lgamma/
// digamma do NOT route through the changed helper (`elementwise_native`).
// Spot-check they still match torch f64. If any FAILS, the change leaked
// into the non-polynomial path.
//
// LIVE torch 2.11.0+cu130, dtype=float64:
//   erf(0.5)     = 0.5204998778130465
//   erfinv(0.7)  = 0.7328690779592167
//   gammaln(3.5) = 1.2009736023470743
//   digamma(2.5) = 0.7031566406452431
// =====================================================================

#[test]
fn reaudit_non_polynomial_unregressed_f64() {
    type FnU = fn(&Tensor<f64>) -> ferrotorch_core::FerrotorchResult<Tensor<f64>>;
    let cases: [(&str, FnU, f64, f64); 4] = [
        ("erf", special::erf, 0.5, 0.5204998778130465),
        ("erfinv", special::erfinv, 0.7, 0.7328690779592167),
        ("lgamma", special::lgamma, 3.5, 1.2009736023470743),
        ("digamma", special::digamma, 2.5, 0.7031566406452431),
    ];
    for (name, f, x, torch_val) in cases {
        let got = f(&cpu_f64(&[x])).unwrap().data().unwrap()[0];
        assert!(
            (got - torch_val).abs() <= 1e-12,
            "{name}({x}) f64: torch={torch_val}, ferro={got} (d={:.2e}) — \
             non-polynomial path regressed",
            (got - torch_val).abs()
        );
    }
}
