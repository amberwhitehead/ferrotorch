//! Discriminator audit of the GPU f32 i0/i0e/i1/i1e PTX kernels at the
//! |x| == 8 A/B Chebyshev-set boundary and large-x finiteness.
//!
//! The shipped on-device tests (`{i0,i0e,i1,i1e}_on_device_matches_torch` in
//! ferrotorch-gpu/src/special.rs) sample x in [-1.5 .. 9.0] — only one point
//! (9.0) above 8, so a wrong A/B split in the unrolled PTX boundary compare
//! would be nearly invisible. This file pins the transition tightly (7.99 /
//! 8.0 / 8.01) and the large-x scaled finiteness on the actual PTX path,
//! mirroring the CPU audit in
//! ferrotorch-core/tests/divergence_bessel_i_boundary_audit.rs.
//!
//! Expected values are LIVE torch 2.11.0+cu130 f32 outputs (R-CHAR-3):
//!   python3 -c "import torch; print(torch.special.i0(
//!     torch.tensor([7.99,8.0,8.01], dtype=torch.float32)).tolist())"
//! Upstream: pytorch 2ec0222669f1bcd37b5670ce384f8608c033b158,
//! aten/src/ATen/native/cuda/Math.cuh:502-696 (i0/i1/i1e jiterator strings).

#![cfg(feature = "cuda")]

use ferrotorch_gpu::special::{gpu_i0_f32, gpu_i0e_f32, gpu_i1_f32, gpu_i1e_f32};
use ferrotorch_gpu::{cpu_to_gpu, gpu_to_cpu, init_cuda_backend, GpuDevice};
use std::sync::Once;

fn ensure_cuda() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        init_cuda_backend().expect("CUDA backend init");
    });
}

fn assert_rel(got: f32, want: f32, tol: f32, ctx: &str) {
    assert!(
        (got - want).abs() <= tol * (1.0 + want.abs()),
        "{ctx}: got {got} want {want} (tol {tol})"
    );
}

/// GPU f32 i0/i0e/i1/i1e across the |x|==8 A/B Chebyshev-set boundary, on-device.
/// Live torch f32: i0([7.99,8.0,8.01]) = [423.58417, 427.5642, 431.58182].
#[test]
fn divergence_bessel_gpu_boundary_at_8() {
    ensure_cuda();
    let device = GpuDevice::new(0).expect("CUDA device 0");
    let xs: [f32; 3] = [7.99, 8.0, 8.01];
    let xg = cpu_to_gpu(&xs, &device).unwrap();

    let yg = gpu_i0_f32(&xg, &device).unwrap();
    assert_eq!(yg.device_ordinal(), device.ordinal(), "i0 stays on device");
    let got = gpu_to_cpu(&yg, &device).unwrap();
    let want_i0: [f32; 3] = [423.58417, 427.5642, 431.58182];
    for k in 0..3 {
        assert_rel(got[k], want_i0[k], 2e-4, &format!("gpu i0(x={})", xs[k]));
    }

    let yg = gpu_i0e_f32(&xg, &device).unwrap();
    let got = gpu_to_cpu(&yg, &device).unwrap();
    let want_i0e: [f32; 3] = [0.14352478, 0.14343181, 0.14333896];
    for k in 0..3 {
        assert_rel(got[k], want_i0e[k], 2e-4, &format!("gpu i0e(x={})", xs[k]));
    }

    let yg = gpu_i1_f32(&xg, &device).unwrap();
    let got = gpu_to_cpu(&yg, &device).unwrap();
    let want_i1: [f32; 3] = [396.1153, 399.87354, 403.66702];
    for k in 0..3 {
        assert_rel(got[k], want_i1[k], 2e-4, &format!("gpu i1(x={})", xs[k]));
    }

    let yg = gpu_i1e_f32(&xg, &device).unwrap();
    let got = gpu_to_cpu(&yg, &device).unwrap();
    let want_i1e: [f32; 3] = [0.1342174, 0.13414262, 0.13406777];
    for k in 0..3 {
        assert_rel(got[k], want_i1e[k], 2e-4, &format!("gpu i1e(x={})", xs[k]));
    }
}

/// GPU f32 odd/even on negative x across the boundary, on-device.
/// i0/i0e EVEN, i1/i1e ODD (sign follows _x via PTX `neg` on x<0).
#[test]
fn divergence_bessel_gpu_boundary_negative() {
    ensure_cuda();
    let device = GpuDevice::new(0).expect("CUDA device 0");
    let xs: [f32; 3] = [-7.99, -8.0, -8.01];
    let xg = cpu_to_gpu(&xs, &device).unwrap();

    let got = gpu_to_cpu(&gpu_i0_f32(&xg, &device).unwrap(), &device).unwrap();
    let want_i0: [f32; 3] = [423.58417, 427.5642, 431.58182];
    for k in 0..3 {
        assert_rel(got[k], want_i0[k], 2e-4, "gpu i0 even");
    }

    let got = gpu_to_cpu(&gpu_i1_f32(&xg, &device).unwrap(), &device).unwrap();
    let want_i1: [f32; 3] = [-396.1153, -399.87354, -403.66702];
    for k in 0..3 {
        assert_rel(got[k], want_i1[k], 2e-4, "gpu i1 odd");
    }

    let got = gpu_to_cpu(&gpu_i1e_f32(&xg, &device).unwrap(), &device).unwrap();
    let want_i1e: [f32; 3] = [-0.1342174, -0.13414262, -0.13406777];
    for k in 0..3 {
        assert_rel(got[k], want_i1e[k], 2e-4, "gpu i1e odd");
    }
}

/// GPU f32 i0e/i1e stay finite for large x on the PTX `ex2.approx`/`sqrt.rn`
/// B-set path. Live torch f32:
///   i0e([50,100]) = [0.05656162276864052, 0.039944376796483994]
///   i1e([50,100]) = [0.05599312484264374, 0.03974415361881256]
#[test]
fn divergence_bessel_gpu_large_x_finite() {
    ensure_cuda();
    let device = GpuDevice::new(0).expect("CUDA device 0");
    let xs: [f32; 2] = [50.0, 100.0];
    let xg = cpu_to_gpu(&xs, &device).unwrap();

    let got = gpu_to_cpu(&gpu_i0e_f32(&xg, &device).unwrap(), &device).unwrap();
    assert!(got[0].is_finite(), "gpu i0e(50) finite, got {}", got[0]);
    assert!(got[1].is_finite(), "gpu i0e(100) finite, got {}", got[1]);
    assert_rel(got[0], 0.05656162, 3e-4, "gpu i0e(50)");
    assert_rel(got[1], 0.039944377, 3e-4, "gpu i0e(100)");

    let got = gpu_to_cpu(&gpu_i1e_f32(&xg, &device).unwrap(), &device).unwrap();
    assert!(got[0].is_finite(), "gpu i1e(50) finite, got {}", got[0]);
    assert!(got[1].is_finite(), "gpu i1e(100) finite, got {}", got[1]);
    assert_rel(got[0], 0.05599312, 3e-4, "gpu i1e(50)");
    assert_rel(got[1], 0.039744154, 3e-4, "gpu i1e(100)");
}
