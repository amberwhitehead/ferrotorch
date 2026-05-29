//! Adversarial re-audit of commit f8c7024b3 (#1673).
//!
//! Background: `GELU_BACKWARD_TANH_PTX` (ferrotorch-gpu/src/kernels.rs:2116)
//! had non-ASCII glyphs (pi / superscript-2 / superscript-3) in its `//`
//! comments. On the WSL CUDA JIT that produced `CUDA_ERROR_INVALID_PTX` for
//! the WHOLE module, and `gpu_gelu_backward_tanh` (kernels.rs:17214) uses
//! `try_launch_binary` with NO scalar fallback — so the tanh-approx GELU
//! backward returned `PtxCompileFailed` on EVERY call. #1673 replaced the
//! glyphs with ASCII; the kernel now compiles and EXECUTES FOR THE FIRST
//! TIME on the production path
//! `GeluBackward::backward` -> `gelu_backward_tanh_f32`
//! (ferrotorch-core/src/grad_fns/activation.rs:336).
//!
//! THE KEY RISK this file pins: the kernel never ran in production before, so
//! its arithmetic was never validated against torch. We drive the real
//! production kernel `gpu_gelu_backward_tanh` and compare element-wise to
//! LIVE torch 2.11.0+cu130 `F.gelu(x, approximate='tanh')` autograd backward.
//!
//! DIVERGENCE FOUND (kernels.rs:2162): the `c3` constant is wrong.
//!   `// 3 * 0.044715 = 0.134145 = 0x3E096B8C`
//!   `mov.f32 %c3, 0f3E096B8C;`
//! But `0x3E096B8C` decodes to 0.134199321, NOT 0.134145. The nearest f32 to
//! the documented value 3*0.044715 = 0.134145 is `0x3E095D4F` = 0.13414501.
//! The literal is 3645 ULPs too large. PyTorch computes
//! `inner_derivative = kBeta*(1 + 3*kKappa*x^2)` with `kKappa = 0.044715`
//! (aten/src/ATen/native/cpu/Activation.cpp:358,371), i.e. the multiplier on
//! x^2 is exactly `3*0.044715 = 0.134145`. The wrong `c3` makes ferrotorch's
//! `right_derivative` term too large, so the gradient diverges by up to
//! ~1.5e-5 absolute / ~1.8e-4 relative — well outside f32 rtol 1e-5.
//!
//! The f64 kernel `GELU_BACKWARD_TANH_F64_PTX` (kernels.rs:2259) carries the
//! SAME wrong constant: `0d3FC12D7180000000` = 0.134199321 (comment claims
//! 0.134145). Both kernels are pinned below.
//!
//! Reference oracle (R-CHAR-3): every expected value below is the live torch
//! 2.11.0+cu130 autograd gradient, recorded by:
//!   x.requires_grad_(); y = F.gelu(x, approximate='tanh'); y.backward(go)
//! and read from `x.grad`. NONE are copied from the ferrotorch side.
//!
//! All tests require a live CUDA device (RTX 3090 in the audit env).

#![cfg(feature = "cuda")]

use std::sync::Once;

use ferrotorch_gpu::device::GpuDevice;
use ferrotorch_gpu::transfer::{cpu_to_gpu, gpu_to_cpu};

fn ensure_cuda() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend().expect("CUDA backend init");
    });
}

fn device() -> GpuDevice {
    GpuDevice::new(0).expect("GpuDevice::new(0)")
}

// ---------------------------------------------------------------------------
// LIVE torch 2.11.0+cu130 oracle — F.gelu(x, approximate='tanh') backward.
// Generated with grad_output varied (not all 1.0) so a wrong dy-multiply is
// also caught. grad[i] = dgelu_tanh/dx(x[i]) * go[i].
// ---------------------------------------------------------------------------

const XS: [f32; 15] = [
    -8.0, -5.0, -3.0, -2.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0,
];
const GOS: [f32; 15] = [
    1.0, 0.5, -1.0, 2.0, 1.5, -0.5, 3.0, 1.0, -2.0, 0.25, 1.0, -1.5, 2.5, 1.0, -1.0,
];
/// torch x.grad after `F.gelu(XS, approximate='tanh').backward(GOS)`.
const TORCH_TANH_GRAD: [f32; 15] = [
    0.0,
    -1.005_437_9e-6,
    0.011_584_297,
    -0.172_198_44,
    -0.124_446_14,
    -0.066_315_055,
    1.261_434_6,
    0.5,
    -1.159_043_5,
    0.216_842_47,
    1.082_964_1,
    -1.629_149,
    2.528_960_7,
    1.000_002,
    -1.0,
];

/// Element-wise relative-or-absolute compare against the torch oracle.
/// rtol = 1e-5, atol = 1e-6 (f32 single-precision parity). The wrong `c3`
/// constant produces ~1.8e-4 relative error at |x| ~ 1..2, so this FAILS.
fn assert_close_to_torch(got: &[f32], want: &[f32], label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    let rtol = 1e-5_f32;
    let atol = 1e-6_f32;
    let mut worst = String::new();
    let mut worst_rel = 0.0_f32;
    let mut any_bad = false;
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let diff = (g - w).abs();
        let tol = atol + rtol * w.abs();
        let rel = if w != 0.0 { diff / w.abs() } else { diff };
        if diff > tol {
            any_bad = true;
            if rel > worst_rel {
                worst_rel = rel;
                worst = format!(
                    "[{i}] x={} go={}: ferrotorch-GPU={g} torch={w} \
                     abs_err={diff:.3e} rel_err={rel:.3e} (tol={tol:.3e})",
                    XS[i], GOS[i]
                );
            }
        }
    }
    assert!(
        !any_bad,
        "{label}: GPU tanh-GELU backward diverges from live torch 2.11.0. \
         Worst: {worst}. Root cause: c3 constant 0x3E096B8C=0.134199 in \
         kernels.rs:2162 should be 3*0.044715=0.134145 (0x3E095D4F)."
    );
}

fn run_tanh_backward_f32(xs: &[f32], gos: &[f32], dev: &GpuDevice) -> Vec<f32> {
    let gx = cpu_to_gpu(gos, dev).expect("upload grad_output");
    let ix = cpu_to_gpu(xs, dev).expect("upload input");
    let out = ferrotorch_gpu::kernels::gpu_gelu_backward_tanh(&gx, &ix, dev)
        .expect("gpu_gelu_backward_tanh");
    gpu_to_cpu(&out, dev).expect("download")
}

// ---------------------------------------------------------------------------
// 1 + 2. CORRECTNESS vs LIVE torch across the value range, AND compiles-now.
//   The `.expect(...)` inside run_* asserts Ok (not PtxCompileFailed) — the
//   #1673 subject. The close-compare is the numerical discriminator that
//   FAILS on the wrong c3 constant.
// ---------------------------------------------------------------------------

#[test]
fn gpu_tanh_gelu_backward_f32_matches_torch() {
    ensure_cuda();
    let dev = device();
    let got = run_tanh_backward_f32(&XS, &GOS, &dev);
    assert_close_to_torch(&got, &TORCH_TANH_GRAD, "f32 tanh-GELU backward");
}

// ---------------------------------------------------------------------------
// 2 (isolated). Kernel COMPILES + returns Ok on the 3090 (the #1673 fix).
//   This passes post-fix regardless of the c3 bug; kept as the compiles-now
//   guard so a future non-ASCII regression (PtxCompileFailed) is caught
//   independently of the arithmetic.
// ---------------------------------------------------------------------------

#[test]
fn gpu_tanh_gelu_backward_f32_compiles_and_runs() {
    ensure_cuda();
    let dev = device();
    let xs = [0.0_f32, 1.0, -1.0, 2.5];
    let gos = [1.0_f32, 1.0, 1.0, 1.0];
    let gx = cpu_to_gpu(&gos, &dev).expect("upload");
    let ix = cpu_to_gpu(&xs, &dev).expect("upload");
    // The pre-#1673 bug made this Err(PtxCompileFailed) on every call.
    let res = ferrotorch_gpu::kernels::gpu_gelu_backward_tanh(&gx, &ix, &dev);
    assert!(
        res.is_ok(),
        "gpu_gelu_backward_tanh must compile + run post-#1673 (got {:?})",
        res.err()
    );
    let h = gpu_to_cpu(&res.unwrap(), &dev).expect("download");
    assert_eq!(h.len(), 4);
    // gelu'(0) = 0.5 exactly (torch oracle: 0.5).
    assert!(
        (h[0] - 0.5).abs() < 1e-6,
        "gelu_tanh'(0) must be 0.5, got {}",
        h[0]
    );
}

// ---------------------------------------------------------------------------
// 3. EDGE behavior vs torch: x=0 -> 0.5, large +x -> ~1, large -x -> ~0,
//    saturated tanh. All from the same XS oracle; broken out as a focused
//    discriminator on the asymptotes (which the c3 bug perturbs least at the
//    extremes but the close-compare still pins the saturation region x~+/-2).
// ---------------------------------------------------------------------------

#[test]
fn gpu_tanh_gelu_backward_f32_edges_match_torch() {
    ensure_cuda();
    let dev = device();
    // x=0 (->0.5), large positive (->1), large negative (->0), tanh-saturating.
    let xs = [0.0_f32, 8.0, -8.0, 2.0, -2.0];
    let gos = [1.0_f32; 5];
    let got = run_tanh_backward_f32(&xs, &gos, &dev);
    // Live torch oracle for these exact inputs, grad_output=1:
    //   x=0:0.5  x=8:1.0  x=-8:0.0  x=2:1.0860993  x=-2:-0.086099222
    let want = [0.5_f32, 1.0, 0.0, 1.086_099_3, -0.086_099_22];
    assert_close_to_torch(&got, &want, "f32 tanh-GELU backward edges");
}

// ---------------------------------------------------------------------------
// 4a. f64 tanh-GELU backward — SAME c3 bug (kernels.rs:2259,
//     0d3FC12D7180000000=0.134199, comment claims 0.134145). Pinned vs torch
//     f64 oracle at f64 rtol (1e-12) — the wrong constant exceeds it grossly.
// ---------------------------------------------------------------------------

#[test]
fn gpu_tanh_gelu_backward_f64_matches_torch() {
    ensure_cuda();
    let dev = device();
    let xs = [-3.0_f64, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0];
    let gos = [1.0_f64, 2.0, -1.0, 1.5, 1.0, 0.5, -2.0, 1.0, 3.0];
    // Live torch 2.11.0 f64 oracle: F.gelu(xs, approximate='tanh').backward(gos)
    let want = [
        -0.011_584_166_630_969_648_f64,
        -0.172_198_513_247_236_6,
        0.082_964_083_845_782_58,
        0.198_945_144_698_036_48,
        0.5,
        0.433_684_951_767_321_2,
        -2.165_928_167_691_565,
        1.086_099_256_623_618_3,
        3.034_752_499_892_908_5,
    ];
    let gx = cpu_to_gpu(&gos, &dev).expect("upload");
    let ix = cpu_to_gpu(&xs, &dev).expect("upload");
    let out = ferrotorch_gpu::kernels::gpu_gelu_backward_tanh_f64(&gx, &ix, &dev)
        .expect("gpu_gelu_backward_tanh_f64");
    let got = gpu_to_cpu(&out, &dev).expect("download");
    let rtol = 1e-9_f64; // generous for the f64 Horner-exp tanh; the c3 error is ~4e-4 rel.
    let atol = 1e-11_f64;
    let mut worst = String::new();
    let mut worst_rel = 0.0_f64;
    let mut any_bad = false;
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let diff = (g - w).abs();
        let tol = atol + rtol * w.abs();
        if diff > tol {
            any_bad = true;
            let rel = if w != 0.0 { diff / w.abs() } else { diff };
            if rel > worst_rel {
                worst_rel = rel;
                worst = format!("[{i}] x={} ferrotorch={g} torch={w} rel={rel:.3e}", xs[i]);
            }
        }
    }
    assert!(
        !any_bad,
        "f64 tanh-GELU backward diverges from torch. Worst: {worst}. \
         Root cause: c3 = 0d3FC12D7180000000 = 0.134199 (kernels.rs:2259) \
         should be 3*0.044715 = 0.134145."
    );
}

// ---------------------------------------------------------------------------
// 4b. SANITY: the EXACT (erf) GELU backward GPU kernel is a DIFFERENT kernel
//     (gpu_gelu_backward_erf, GELU_BACKWARD_ERF_PTX). The #1673 ASCII fix did
//     not touch its constants. Confirm it matches torch approximate='none'.
//     If THIS fails too, the c3-class bug is broader than the tanh kernel.
// ---------------------------------------------------------------------------

#[test]
fn gpu_erf_gelu_backward_f32_matches_torch() {
    ensure_cuda();
    let dev = device();
    let xs = [-3.0_f32, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0];
    let gos = [1.0_f32, 2.0, -1.0, 1.5, 1.0, 0.5, -2.0, 1.0, 3.0];
    // Live torch 2.11.0 oracle: F.gelu(xs, approximate='none').backward(gos)
    let want = [
        -0.011_945_605_f32,
        -0.170_463_74,
        0.083_315_43,
        0.198_757_37,
        0.5,
        0.433_747_53,
        -2.166_630_7,
        1.085_231_9,
        3.035_836_7,
    ];
    let gx = cpu_to_gpu(&gos, &dev).expect("upload");
    let ix = cpu_to_gpu(&xs, &dev).expect("upload");
    let out = ferrotorch_gpu::kernels::gpu_gelu_backward_erf(&gx, &ix, &dev)
        .expect("gpu_gelu_backward_erf");
    let got = gpu_to_cpu(&out, &dev).expect("download");
    assert_close_to_torch(&got, &want, "f32 erf-GELU backward (sanity)");
}
