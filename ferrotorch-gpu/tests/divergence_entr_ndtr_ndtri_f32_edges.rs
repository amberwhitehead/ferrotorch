//! Discriminator GPU probe for #1651 batch 1 (entr/ndtr/ndtri f32 on-device).
//!
//! The in-crate lib tests cover a comfortable interior grid; this probe targets
//! the edges they skip and where the `lg2.approx.f32` / A&S-7.1.26 device paths
//! are most strained:
//!   - entr near x = 1 (relative-error-sensitive: -x*ln(x) -> 0),
//!   - ndtr at large |x| (A&S 7.1.26 erf saturation),
//!   - ndtri deep tail (P2/Q2 far-tail + sqrt(-2 log y) with lg2.approx).
//!
//! Expected values are live `torch.special.*` (torch 2.11.0+cu130, f32) oracle
//! outputs (R-CHAR-3, not self-referential). Compared at f32 tolerance.

#![cfg(feature = "cuda")]

use ferrotorch_gpu::{
    GpuDevice, cpu_to_gpu, gpu_entr_f32, gpu_ndtr_f32, gpu_ndtri_f32, gpu_to_cpu,
};

fn dev() -> Option<GpuDevice> {
    GpuDevice::new(0).ok()
}

/// entr near x = 1 (and just above): torch.special.entr f32 oracle.
#[test]
fn entr_near_one_on_device_matches_torch() {
    let Some(device) = dev() else { return };
    let xs: [f32; 6] = [0.95, 0.99, 0.999, 1.001, 1.01, 1.05];
    // live torch.special.entr(f32):
    let want: [f32; 6] = [
        0.048_728_64,
        0.009_949_824,
        0.000_999_487,
        -0.001_000_546_5,
        -0.010_049_825,
        -0.051_229_62,
    ];
    let xg = cpu_to_gpu(&xs, &device).unwrap();
    let yg = gpu_entr_f32(&xg, &device).unwrap();
    assert_eq!(yg.device_ordinal(), device.ordinal());
    let got = gpu_to_cpu(&yg, &device).unwrap();
    for i in 0..6 {
        assert!(
            (got[i] - want[i]).abs() <= 1e-5 * (1.0 + want[i].abs()),
            "entr near-1 idx {i} x={}: got {} want {}",
            xs[i],
            got[i],
            want[i]
        );
    }
}

/// ndtr at large |x|: torch.special.ndtr f32 oracle.
#[test]
#[allow(
    clippy::excessive_precision,
    reason = "live torch.special f32 oracle literals; rounds to f32 at compile time"
)]
fn ndtr_large_x_on_device_matches_torch() {
    let Some(device) = dev() else { return };
    let xs: [f32; 6] = [-4.0, -3.0, 3.0, 4.0, 6.0, 10.0];
    // live torch.special.ndtr(f32):
    let want: [f32; 6] = [
        3.167_987e-5,
        0.001_349_896_2,
        0.998_650_07,
        0.999_968_3,
        1.0,
        1.0,
    ];
    let xg = cpu_to_gpu(&xs, &device).unwrap();
    let yg = gpu_ndtr_f32(&xg, &device).unwrap();
    assert_eq!(yg.device_ordinal(), device.ordinal());
    let got = gpu_to_cpu(&yg, &device).unwrap();
    for i in 0..6 {
        assert!(
            (got[i] - want[i]).abs() <= 1e-5 * (1.0 + want[i].abs()),
            "ndtr large-x idx {i} x={}: got {} want {}",
            xs[i],
            got[i],
            want[i]
        );
    }
}

/// ndtri deep tail + central symmetry: torch.special.ndtri f32 oracle.
/// 1e-6/1e-5 -> far-tail (P2/Q2, x >= 8); 1e-3 -> tail (P1/Q1); 0.4/0.6
/// exercise central-region symmetry; 0.999 -> code-flag flip region.
#[test]
fn ndtri_deep_tail_on_device_matches_torch() {
    let Some(device) = dev() else { return };
    let ps: [f32; 6] = [1e-6, 1e-5, 1e-3, 0.4, 0.6, 0.999];
    // live torch.special.ndtri(f32):
    let want: [f32; 6] = [
        -4.753_424,
        -4.264_890_7,
        -3.090_232_4,
        -0.253_347_1,
        0.253_347_16,
        3.090_236,
    ];
    let xg = cpu_to_gpu(&ps, &device).unwrap();
    let yg = gpu_ndtri_f32(&xg, &device).unwrap();
    assert_eq!(yg.device_ordinal(), device.ordinal());
    let got = gpu_to_cpu(&yg, &device).unwrap();
    for i in 0..6 {
        assert!(
            (got[i] - want[i]).abs() <= 1e-4 * (1.0 + want[i].abs()),
            "ndtri deep-tail idx {i} p={}: got {} want {}",
            ps[i],
            got[i],
            want[i]
        );
    }
}
