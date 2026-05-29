//! Live-GPU verification of the modified-Bessel-K family f32 PTX kernels
//! landed for #1651 batch 3b (`ferrotorch-gpu/src/special.rs`):
//! `modified_bessel_k0` / `scaled_modified_bessel_k0` / `modified_bessel_k1` /
//! `scaled_modified_bessel_k1`.
//!
//! Each `verify_*` test runs the FULL `ferrotorch_core::special::*` Tensor
//! dispatch path on a genuinely CUDA-resident f32 input and asserts:
//!   (a) the result stays on device (`Tensor::is_cuda()` — the GPU `_f32`
//!       backend op returned a CUDA handle, no host round trip, R-CODE-4), AND
//!   (b) the values match LIVE `torch.special.*` (torch 2.11.0+cu130, f32),
//!       to ~1e-5 rtol (f32 transcendental).
//!
//! Reference values are LIVE torch 2.11.0+cu130 f32 outputs (R-CHAR-3), NOT
//! copied from ferrotorch. Produced under
//! `LD_LIBRARY_PATH="$HOME/.local/lib:$LD_LIBRARY_PATH"`:
//!   x = [0.5, 1.0, 1.5, 2.5, 5.0]   (domain-valid: x > 0; K diverges at x->0)
//!   torch.special.modified_bessel_k0(x.float())        =
//!     [9.24419165e-01, 4.21024472e-01, 2.13805586e-01, 6.23475537e-02,
//!      3.69109819e-03]
//!   torch.special.scaled_modified_bessel_k0(x.float())  =
//!     [1.52410948e+00, 1.14446318e+00, 9.58210230e-01, 7.59548664e-01,
//!      5.47807574e-01]
//!   torch.special.modified_bessel_k1(x.float())        =
//!     [1.65644097e+00, 6.01907313e-01, 2.77387798e-01, 7.38908127e-02,
//!      4.04461287e-03]
//!   torch.special.scaled_modified_bessel_k1(x.float())  =
//!     [2.73100948e+00, 1.63615370e+00, 1.24316585e+00, 9.00174439e-01,
//!      6.00273848e-01]
//!
//! The kernels span the `x <= 2` (small, chbevl(x*x-2,A) over the log-term
//! composition with the inner i0/i1) and `x > 2` (big, chbevl(8/x-2,B)/sqrt(x))
//! Cephes regions — the inputs straddle the split (0.5/1.0/1.5/2.5/5.0).
//! Upstream: `aten/src/ATen/native/cuda/Math.cuh:2503-2577, 2582-2656,
//! 2661-2736, 2740-2815` (modified_bessel_k{0,1}_forward + scaled variants).

#![cfg(feature = "cuda")]

use std::sync::Once;

use ferrotorch_core::{Device, FerrotorchError, Tensor, TensorStorage, special};
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

const XS: [f32; 5] = [0.5, 1.0, 1.5, 2.5, 5.0];
const RTOL: f32 = 1e-5;

fn cuda_f32(data: &[f32]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false)
        .unwrap()
        .to(Device::Cuda(0))
        .expect("Tensor<f32>::to(Cuda)")
}

/// Run a K-family pub fn on a CUDA f32 tensor, assert on-device, return the
/// host-copied results for value comparison.
fn run_on_device(
    op: fn(&Tensor<f32>) -> Result<Tensor<f32>, FerrotorchError>,
    name: &str,
) -> Vec<f32> {
    let gpu_in = cuda_f32(&XS);
    assert!(gpu_in.is_cuda(), "{name}: input must be CUDA-resident");
    let gpu_out = op(&gpu_in).unwrap_or_else(|e| panic!("{name} GPU dispatch: {e:?}"));
    assert!(
        gpu_out.is_cuda(),
        "{name}: result must stay on device (no host round trip)"
    );
    gpu_out.to(Device::Cpu).unwrap().data().unwrap().to_vec()
}

fn assert_matches(got: &[f32], want: &[f32], name: &str) {
    for i in 0..XS.len() {
        let tol = RTOL * (1.0 + want[i].abs());
        assert!(
            (got[i] - want[i]).abs() <= tol,
            "{name} on-device x={}: got {} want {} (torch f32; tol {tol})",
            XS[i],
            got[i],
            want[i]
        );
    }
}

#[test]
#[allow(
    clippy::excessive_precision,
    reason = "literals are the verbatim torch 2.11.0 f32 printout (R-CHAR-3 provenance in the module doc); the extra printed digit documents the oracle, the f32 parse rounds it"
)]
fn verify_modified_bessel_k0_gpu_f32_on_device_matches_torch() {
    if !ensure_cuda() {
        eprintln!("no CUDA device; skipping");
        return;
    }
    let got = run_on_device(special::modified_bessel_k0, "modified_bessel_k0");
    let want: [f32; 5] = [
        9.244_191_65e-1,
        4.210_244_72e-1,
        2.138_055_86e-1,
        6.234_755_37e-2,
        3.691_098_19e-3,
    ];
    assert_matches(&got, &want, "modified_bessel_k0");
}

#[test]
#[allow(
    clippy::excessive_precision,
    reason = "verbatim torch 2.11.0 f32 oracle printout (R-CHAR-3)"
)]
fn verify_scaled_modified_bessel_k0_gpu_f32_on_device_matches_torch() {
    if !ensure_cuda() {
        eprintln!("no CUDA device; skipping");
        return;
    }
    let got = run_on_device(
        special::scaled_modified_bessel_k0,
        "scaled_modified_bessel_k0",
    );
    let want: [f32; 5] = [
        1.524_109_48e0,
        1.144_463_18e0,
        9.582_102_30e-1,
        7.595_486_64e-1,
        5.478_075_74e-1,
    ];
    assert_matches(&got, &want, "scaled_modified_bessel_k0");
}

#[test]
#[allow(
    clippy::excessive_precision,
    reason = "verbatim torch 2.11.0 f32 oracle printout (R-CHAR-3)"
)]
fn verify_modified_bessel_k1_gpu_f32_on_device_matches_torch() {
    if !ensure_cuda() {
        eprintln!("no CUDA device; skipping");
        return;
    }
    let got = run_on_device(special::modified_bessel_k1, "modified_bessel_k1");
    let want: [f32; 5] = [
        1.656_440_97e0,
        6.019_073_13e-1,
        2.773_877_98e-1,
        7.389_081_27e-2,
        4.044_612_87e-3,
    ];
    assert_matches(&got, &want, "modified_bessel_k1");
}

#[test]
#[allow(
    clippy::excessive_precision,
    reason = "verbatim torch 2.11.0 f32 oracle printout (R-CHAR-3)"
)]
fn verify_scaled_modified_bessel_k1_gpu_f32_on_device_matches_torch() {
    if !ensure_cuda() {
        eprintln!("no CUDA device; skipping");
        return;
    }
    let got = run_on_device(
        special::scaled_modified_bessel_k1,
        "scaled_modified_bessel_k1",
    );
    let want: [f32; 5] = [
        2.731_009_48e0,
        1.636_153_70e0,
        1.243_165_85e0,
        9.001_744_39e-1,
        6.002_738_48e-1,
    ];
    assert_matches(&got, &want, "scaled_modified_bessel_k1");
}

/// The CPU path is unchanged: an identical CPU f32 tensor must produce the same
/// torch-matching values (the GPU dispatch is purely additive — `is_cuda()`
/// false short-circuits `special_gpu_simple` to the scalar `unary_map`).
#[test]
#[allow(
    clippy::excessive_precision,
    reason = "verbatim torch 2.11.0 f32 oracle printout (R-CHAR-3)"
)]
fn cpu_path_unchanged_matches_torch_f32() {
    let cpu_in =
        Tensor::from_storage(TensorStorage::cpu(XS.to_vec()), vec![XS.len()], false).unwrap();
    assert!(!cpu_in.is_cuda());
    let out = special::modified_bessel_k0(&cpu_in).unwrap();
    let got = out.data().unwrap();
    let want: [f32; 5] = [
        9.244_191_65e-1,
        4.210_244_72e-1,
        2.138_055_86e-1,
        6.234_755_37e-2,
        3.691_098_19e-3,
    ];
    assert_matches(got, &want, "modified_bessel_k0 CPU");
}

/// f64 CUDA tensors must cleanly reject with `NotImplementedOnCuda` (base PTX
/// has no `lg2.approx.f64`/`ex2.approx.f64`), for all four ops — never a host
/// round trip, never a wrong-precision answer.
#[test]
fn f64_cuda_rejects_not_implemented() {
    if !ensure_cuda() {
        eprintln!("no CUDA device; skipping");
        return;
    }
    let gpu_in = Tensor::from_storage(TensorStorage::cpu(vec![1.0_f64, 2.5]), vec![2], false)
        .unwrap()
        .to(Device::Cuda(0))
        .expect("Tensor<f64>::to(Cuda)");
    assert!(gpu_in.is_cuda());
    for (name, res) in [
        ("k0", special::modified_bessel_k0(&gpu_in)),
        ("sk0", special::scaled_modified_bessel_k0(&gpu_in)),
        ("k1", special::modified_bessel_k1(&gpu_in)),
        ("sk1", special::scaled_modified_bessel_k1(&gpu_in)),
    ] {
        assert!(
            matches!(res, Err(FerrotorchError::NotImplementedOnCuda { .. })),
            "f64 CUDA {name} must be NotImplementedOnCuda, got {res:?}"
        );
    }
}

/// bf16 CUDA tensors must cleanly reject with `NotImplementedOnCuda` (the
/// dispatch rejects any dtype that is not f32/f64 before touching the device),
/// for all four ops.
#[test]
fn bf16_cuda_rejects_not_implemented() {
    if !ensure_cuda() {
        eprintln!("no CUDA device; skipping");
        return;
    }
    let bf: Vec<half::bf16> = [1.0_f32, 2.5]
        .iter()
        .copied()
        .map(half::bf16::from_f32)
        .collect();
    let gpu_in = Tensor::from_storage(TensorStorage::cpu(bf), vec![2], false)
        .unwrap()
        .to(Device::Cuda(0))
        .expect("Tensor<bf16>::to(Cuda)");
    assert!(gpu_in.is_cuda());
    for (name, res) in [
        ("k0", special::modified_bessel_k0(&gpu_in)),
        ("sk0", special::scaled_modified_bessel_k0(&gpu_in)),
        ("k1", special::modified_bessel_k1(&gpu_in)),
        ("sk1", special::scaled_modified_bessel_k1(&gpu_in)),
    ] {
        assert!(
            matches!(res, Err(FerrotorchError::NotImplementedOnCuda { .. })),
            "bf16 CUDA {name} must be NotImplementedOnCuda, got {res:?}"
        );
    }
}

/// f16 CUDA tensors must cleanly reject with `NotImplementedOnCuda` for all
/// four ops (same non-f32/f64 dtype guard as bf16).
#[test]
fn f16_cuda_rejects_not_implemented() {
    if !ensure_cuda() {
        eprintln!("no CUDA device; skipping");
        return;
    }
    let hf: Vec<half::f16> = [1.0_f32, 2.5]
        .iter()
        .copied()
        .map(half::f16::from_f32)
        .collect();
    let gpu_in = Tensor::from_storage(TensorStorage::cpu(hf), vec![2], false)
        .unwrap()
        .to(Device::Cuda(0))
        .expect("Tensor<f16>::to(Cuda)");
    assert!(gpu_in.is_cuda());
    for (name, res) in [
        ("k0", special::modified_bessel_k0(&gpu_in)),
        ("sk0", special::scaled_modified_bessel_k0(&gpu_in)),
        ("k1", special::modified_bessel_k1(&gpu_in)),
        ("sk1", special::scaled_modified_bessel_k1(&gpu_in)),
    ] {
        assert!(
            matches!(res, Err(FerrotorchError::NotImplementedOnCuda { .. })),
            "f16 CUDA {name} must be NotImplementedOnCuda, got {res:?}"
        );
    }
}
