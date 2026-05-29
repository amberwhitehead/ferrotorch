//! Discriminator probe for #1651 batch 3a (commit c9b9ffe6c):
//! GPU spherical_bessel_j0 f32 uses `sin.approx.f32` (a low-precision hardware
//! approximation with a bounded argument-reduction range) where upstream
//! torch's CUDA kernel uses full-precision `::sin()` (libdevice `__nv_sinf`,
//! which performs full Payne-Hanek range reduction; `return sin(x) / x;`,
//! `aten/src/ATen/native/cuda/Math.cuh:3051`).
//!
//! `sin.approx.f32` (PTX label `SINX:` in `SPHERICAL_BESSEL_J0_F32_PTX`,
//! `ferrotorch-gpu/src/special.rs`) has NO argument range reduction beyond the
//! hardware's bounded interval; for |x| past that interval its result is
//! meaningless, whereas torch's `__nv_sinf` stays accurate. The generator's own
//! on-device test tops out at |x|=5, so it never exercises this regime.
//!
//! Each `want` is a LIVE torch.special.spherical_bessel_j0 f32 value (torch
//! 2.11.0+cu130), NOT copied from ferrotorch (R-CHAR-3).

#![cfg(feature = "cuda")]

use ferrotorch_gpu::device::GpuDevice;
use ferrotorch_gpu::special::gpu_spherical_bessel_j0_f32;
use ferrotorch_gpu::transfer::{cpu_to_gpu, gpu_to_cpu};

/// Moderate regime (10..314): `sin.approx.f32` is still within its accurate
/// range here; this part is expected to PASS and is kept as a positive control.
#[test]
fn spherical_bessel_j0_on_device_moderate_x_matches_torch() {
    ferrotorch_gpu::init_cuda_backend().expect("CUDA init");
    let Ok(device) = GpuDevice::new(0) else {
        eprintln!("no CUDA device; skipping");
        return;
    };
    let xs: [f32; 5] = [10.0, 30.0, 50.0, 100.0, 314.0];
    let want: [f32; 5] = [
        -0.054_402_113,
        -0.032_934_386,
        -0.005_247_497,
        -0.005_063_656_7,
        -0.000_505_072_94,
    ];
    let xg = cpu_to_gpu(&xs, &device).expect("cpu_to_gpu");
    let yg = gpu_spherical_bessel_j0_f32(&xg, &device).expect("kernel launch");
    assert_eq!(yg.device_ordinal(), device.ordinal());
    let got = gpu_to_cpu(&yg, &device).expect("gpu_to_cpu");
    for i in 0..xs.len() {
        assert!(
            (got[i] - want[i]).abs() <= 2e-4 * (1.0 + want[i].abs()),
            "spherical_bessel_j0 on-device x={}: got {} want {} (torch f32)",
            xs[i],
            got[i],
            want[i]
        );
    }
}

/// Large regime (1e4..1e6): past `sin.approx.f32`'s accurate argument range.
/// torch's `__nv_sinf` range-reduces and stays accurate; if ferrotorch's
/// `sin.approx.f32` does not, the on-device result diverges and this FAILS,
/// pinning the divergence.
///
/// Live torch.special.spherical_bessel_j0(f32):
///   1e4 -> -3.05614375974983e-05
///   1e5 ->  3.574879769985273e-07
///   1e6 -> -3.499934848605335e-07
#[test]
fn spherical_bessel_j0_on_device_large_x_matches_torch() {
    ferrotorch_gpu::init_cuda_backend().expect("CUDA init");
    let Ok(device) = GpuDevice::new(0) else {
        eprintln!("no CUDA device; skipping");
        return;
    };
    let xs: [f32; 3] = [10_000.0, 100_000.0, 1_000_000.0];
    let want: [f32; 3] = [-3.056_143_8e-5, 3.574_879_8e-7, -3.499_934_8e-7];
    let xg = cpu_to_gpu(&xs, &device).expect("cpu_to_gpu");
    let yg = gpu_spherical_bessel_j0_f32(&xg, &device).expect("kernel launch");
    assert_eq!(yg.device_ordinal(), device.ordinal());
    let got = gpu_to_cpu(&yg, &device).expect("gpu_to_cpu");
    for i in 0..xs.len() {
        assert!(
            (got[i] - want[i]).abs() <= 2e-4 * (1.0 + want[i].abs()),
            "spherical_bessel_j0 on-device LARGE x={}: got {} want {} (torch f32); \
             sin.approx.f32 lacks the range reduction torch's __nv_sinf does",
            xs[i],
            got[i],
            want[i]
        );
    }
}
