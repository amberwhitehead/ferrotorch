//! ADVERSARIAL RE-AUDIT of commit 886db4be1 (#1651 GPU tail) — the `airy_ai`
//! and `zeta` GPU f32 PTX kernels in `ferrotorch-gpu/src/special.rs`, wired
//! through the `special_gpu_simple` / `special_gpu_binary` CUDA-f32 dispatch
//! branches of `ferrotorch_core::special::{airy_ai,zeta}`.
//!
//! # DIVERGENCE PINNED — broadcast / mixed-device CUDA `zeta` (priority probe)
//!
//! `ferrotorch_core::special::zeta` (special.rs:2290) takes the GPU fast path
//! only when BOTH operands are CUDA-resident with the SAME shape
//! (`special_gpu_binary`, special.rs:2588: `if x.is_cuda() != q.is_cuda() ||
//! x.shape() != q.shape() { return Ok(None); }`). For broadcasting CUDA inputs
//! (e.g. `zeta(CUDA[3], CUDA[1])`, `zeta(CUDA[2,3], CUDA[3])`) and for mixed
//! device (`zeta(CUDA, CPU)`) it returns `Ok(None)` and falls to
//! `binary_map(input, other, zeta_scalar)` (special.rs:2300).
//!
//! The `special_gpu_binary` doc comment (special.rs:2559-2560) AND the `zeta`
//! doc comment (special.rs:2287-2288) BOTH claim those cases "fall through to
//! the CPU `binary_map` (which broadcasts)". That claim is FALSE: `binary_map`
//! (`ferrotorch-core/src/ops/elementwise.rs:889`) first `.contiguous()`-clones
//! the operands (a CUDA-contiguous tensor stays CUDA — methods.rs:1572-1573)
//! and then calls `.data()` (elementwise.rs:908), which returns
//! `Err(GpuTensorNotAccessible)` for any GPU tensor (tensor.rs:662). So the op
//! does NOT compute on CPU, does NOT stay on-device, and does NOT cleanly
//! reject with `NotImplementedOnCuda`: it LEAKS the internal
//! `GpuTensorNotAccessible` storage error to the caller.
//!
//! Upstream `torch.special.zeta` broadcasts on the input device and KEEPS the
//! result there (verified live, torch 2.11.0):
//!   torch.special.zeta(tensor([2.,3.,4.]), tensor([1.]))      -> shape [3],
//!     [1.644934058, 1.202056885, 1.082323194]   (stays on input device)
//!   torch.special.zeta(tensor([[2,3,4],[1.5,2.5,5]]), tensor([1.,2.,1.]))
//!     -> shape [2,3]
//! The documented contract — "either stay is_cuda() OR cleanly reject
//! NotImplementedOnCuda, never a silent host detour, never a leaked internal
//! error" — is violated.
//!
//! Tracking: #1654.
//!
//! The remaining tests in this file are PASSING regression guards that the
//! airy/zeta on-device math, region boundaries, edge ladder and dtype
//! rejection are clean (no divergence there). Oracle values are LIVE torch
//! 2.11.0 f32 (R-CHAR-3), printed under
//! `LD_LIBRARY_PATH="$HOME/.local/lib:$LD_LIBRARY_PATH" python3`:
//!
//!   import torch
//!   bx = torch.tensor([-2.09,-2.10,-2.08,8.3203353,8.32,8.33,-7.95,103.892,
//!                      103.9,0.0], dtype=torch.float32)
//!   torch.special.airy_ai(bx) =
//!     [0.17005064, 0.16348736, 0.17658111, 1.8613099e-08, 1.8631324e-08,
//!      1.8096109e-08, -0.0055570463, 0.0, 0.0, 0.35502806]
//!   zx=torch.tensor([2.,3.,4.]); zq=torch.tensor([1.])
//!   torch.special.zeta(zx,zq) -> [1.644934058, 1.202056885, 1.082323194]

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

fn cuda_f32(data: &[f32], shape: Vec<usize>) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape, false)
        .unwrap()
        .to(Device::Cuda(0))
        .expect("Tensor<f32>::to(Cuda)")
}

fn cpu_f32(data: &[f32], shape: Vec<usize>) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape, false).unwrap()
}

fn assert_close(got: &[f32], want: &[f32], rtol: f32, atol: f32, name: &str) {
    assert_eq!(got.len(), want.len(), "{name}: length mismatch");
    for i in 0..want.len() {
        if want[i].is_nan() {
            assert!(got[i].is_nan(), "{name} idx {i}: want NaN, got {}", got[i]);
            continue;
        }
        let tol = atol + rtol * want[i].abs();
        assert!(
            (got[i] - want[i]).abs() <= tol,
            "{name} idx {i}: got {} want {} (torch f32; tol {tol})",
            got[i],
            want[i]
        );
    }
}

// ===========================================================================
// DIVERGENCE (priority probe) — FAILING under commit 886db4be1.
// ===========================================================================

/// Divergence: `ferrotorch_core::special::zeta` for a BROADCASTING pair of
/// CUDA tensors (`special.rs:2300` -> `binary_map`) leaks
/// `GpuTensorNotAccessible` from `ops/elementwise.rs:908` instead of staying
/// on-device (as `torch.special.zeta` does) or cleanly rejecting with
/// `NotImplementedOnCuda`. The `special_gpu_binary` doc (special.rs:2560)
/// claims it "falls through to the CPU binary_map (which broadcasts)", but
/// `binary_map` cannot read CUDA storage, so it errors out.
///
/// Upstream `torch.special.zeta([2.,3.,4.], [1.])` -> shape [3]
/// [1.644934058, 1.202056885, 1.082323194], staying on the input device.
///
/// Tracking: #1654.
#[test]
fn divergence_zeta_broadcast_cuda_3_1() {
    if !ensure_cuda() {
        eprintln!("no CUDA device; skipping");
        return;
    }
    let x = cuda_f32(&[2.0, 3.0, 4.0], vec![3]);
    let q = cuda_f32(&[1.0], vec![1]);
    let result = special::zeta(&x, &q);

    // The documented contract: EITHER on-device result OR clean
    // NotImplementedOnCuda. Never a leaked internal storage error.
    match result {
        Ok(out) => {
            // torch broadcasts to [3] and keeps the result on device.
            assert_eq!(out.shape(), &[3], "zeta broadcast shape mismatch vs torch");
            assert!(
                out.is_cuda(),
                "zeta broadcast on two CUDA inputs must stay on device (torch does)"
            );
            let v = out.to(Device::Cpu).unwrap().data().unwrap().to_vec();
            assert_close(
                &v,
                &[1.644_934_1, 1.202_056_9, 1.082_323_2],
                1e-4,
                5e-7,
                "zeta broadcast [3]+[1]",
            );
        }
        Err(FerrotorchError::NotImplementedOnCuda { .. }) => {
            // Acceptable per the documented R-CODE-4 alternative.
        }
        Err(other) => panic!(
            "zeta(CUDA[3], CUDA[1]) leaked {other:?}; contract requires on-device \
             result or NotImplementedOnCuda, never a raw storage error"
        ),
    }
}

/// Divergence: rank-2 broadcasting CUDA `zeta` (`zeta(CUDA[2,3], CUDA[3])`)
/// — same mechanism as `divergence_zeta_broadcast_cuda_3_1`. torch broadcasts
/// to [2,3]; ferrotorch leaks `GpuTensorNotAccessible`. Tracking: #1654.
#[test]
fn divergence_zeta_broadcast_cuda_2x3_3() {
    if !ensure_cuda() {
        eprintln!("no CUDA device; skipping");
        return;
    }
    let x = cuda_f32(&[2.0, 3.0, 4.0, 1.5, 2.5, 5.0], vec![2, 3]);
    let q = cuda_f32(&[1.0, 2.0, 1.0], vec![3]);
    match special::zeta(&x, &q) {
        Ok(out) => {
            assert_eq!(out.shape(), &[2, 3], "zeta broadcast shape vs torch");
            assert!(out.is_cuda(), "zeta broadcast must stay on device");
        }
        Err(FerrotorchError::NotImplementedOnCuda { .. }) => {}
        Err(other) => panic!(
            "zeta(CUDA[2,3], CUDA[3]) leaked {other:?}; want on-device result or \
             NotImplementedOnCuda"
        ),
    }
}

/// Divergence: MIXED-device `zeta(CUDA, CPU)` leaks `GpuTensorNotAccessible`
/// rather than cleanly rejecting `NotImplementedOnCuda` (or matching torch,
/// which errors with an explicit device-mismatch message). A raw storage
/// error is not a documented outcome. Tracking: #1654.
#[test]
fn divergence_zeta_mixed_device_cuda_cpu() {
    if !ensure_cuda() {
        eprintln!("no CUDA device; skipping");
        return;
    }
    let x = cuda_f32(&[2.0, 3.0], vec![2]);
    let q = cpu_f32(&[1.0, 1.0], vec![2]);
    match special::zeta(&x, &q) {
        Ok(out) => {
            // If it produces a value, it must at least be defined; torch keeps
            // it on the CUDA operand's device.
            assert!(
                out.is_cuda(),
                "mixed-device zeta result on CUDA operand should stay CUDA"
            );
        }
        Err(FerrotorchError::NotImplementedOnCuda { .. }) => {}
        Err(other) => panic!(
            "zeta(CUDA, CPU) leaked {other:?}; contract requires NotImplementedOnCuda \
             (or an explicit device-mismatch), never a raw storage error"
        ),
    }
}

// ===========================================================================
// PASSING regression guards — on-device math is CLEAN (no divergence).
// ===========================================================================

/// Region-boundary guard: airy_ai f32 on-device matches torch f32 at and just
/// inside/outside the -2.09 (oscillatory|central) and 8.3203353
/// (central|decaying) splits and the 103.892 zero cutoff — no step
/// discontinuity. Stays on device.
#[test]
#[allow(
    clippy::excessive_precision,
    reason = "verbatim torch 2.11.0 f32 oracle (R-CHAR-3 provenance in module doc)"
)]
fn guard_airy_ai_region_boundaries_on_device() {
    if !ensure_cuda() {
        eprintln!("no CUDA device; skipping");
        return;
    }
    let xs: [f32; 10] = [
        -2.09, -2.10, -2.08, 8.3203353, 8.32, 8.33, -7.95, 103.892, 103.9, 0.0,
    ];
    let out = special::airy_ai(&cuda_f32(&xs, vec![10])).expect("airy_ai GPU dispatch");
    assert!(out.is_cuda(), "airy_ai must stay on device (no host round trip)");
    let got = out.to(Device::Cpu).unwrap().data().unwrap().to_vec();
    let want: [f32; 10] = [
        0.17005064,
        0.16348736,
        0.17658111,
        1.8613099e-08,
        1.8631324e-08,
        1.8096109e-08,
        -0.0055570463,
        0.0,
        0.0,
        0.35502806,
    ];
    // Region boundaries: torch's own f32 is the oracle. -7.95 oscillatory lands
    // ~4.4e-7 from torch f32 (each path's f32 rounding differs); inside atol.
    assert_close(&got, &want, 1e-5, 6e-7, "airy_ai region boundaries");
}

/// Zeta convergence guard: x>1 cases + the q!=1 case match torch f32, on
/// device. Mirrors the same-shape same-device fast path.
#[test]
#[allow(
    clippy::excessive_precision,
    reason = "verbatim torch 2.11.0 f32 oracle (R-CHAR-3)"
)]
fn guard_zeta_convergent_on_device() {
    if !ensure_cuda() {
        eprintln!("no CUDA device; skipping");
        return;
    }
    let xs: [f32; 6] = [2.0, 3.0, 2.0, 4.0, 1.5, 2.5];
    let qs: [f32; 6] = [1.0, 1.0, 2.0, 0.5, 2.0, 3.0];
    let out =
        special::zeta(&cuda_f32(&xs, vec![6]), &cuda_f32(&qs, vec![6])).expect("zeta GPU dispatch");
    assert!(out.is_cuda(), "zeta same-shape CUDA must stay on device");
    let got = out.to(Device::Cpu).unwrap().data().unwrap().to_vec();
    let want: [f32; 6] = [
        1.64493406e+00,
        1.20205688e+00,
        6.44934058e-01,
        1.62348480e+01,
        1.61237538e+00,
        1.64710566e-01,
    ];
    assert_close(&got, &want, 1e-4, 5e-7, "zeta convergent");
}

/// Zeta edge ladder guard: x==1 -> +inf; x<1 -> NaN; q<=0 poles. Matches
/// torch f32, on device.
#[test]
fn guard_zeta_edge_ladder_on_device() {
    if !ensure_cuda() {
        eprintln!("no CUDA device; skipping");
        return;
    }
    let xs: [f32; 5] = [1.0, 0.5, 2.0, 3.0, 2.5];
    let qs: [f32; 5] = [2.0, 1.0, 0.0, -2.0, -1.5];
    let out =
        special::zeta(&cuda_f32(&xs, vec![5]), &cuda_f32(&qs, vec![5])).expect("zeta GPU dispatch");
    assert!(out.is_cuda());
    let d = out.to(Device::Cpu).unwrap().data().unwrap().to_vec();
    assert!(d[0].is_infinite() && d[0] > 0.0, "zeta(1,2)==+inf: {}", d[0]);
    assert!(d[1].is_nan(), "zeta(0.5,1)==NaN: {}", d[1]);
    assert!(d[2].is_infinite() && d[2] > 0.0, "zeta(2,0)==+inf: {}", d[2]);
    assert!(d[3].is_infinite() && d[3] > 0.0, "zeta(3,-2)==+inf: {}", d[3]);
    assert!(d[4].is_nan(), "zeta(2.5,-1.5)==NaN: {}", d[4]);
}

/// f64 CUDA rejects `NotImplementedOnCuda` for BOTH ops (no host fallback, no
/// panic). Guards R-CODE-4.
#[test]
fn guard_f64_cuda_rejects() {
    if !ensure_cuda() {
        eprintln!("no CUDA device; skipping");
        return;
    }
    let airy_in = Tensor::from_storage(TensorStorage::cpu(vec![1.0_f64, 2.5]), vec![2], false)
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();
    assert!(matches!(
        special::airy_ai(&airy_in),
        Err(FerrotorchError::NotImplementedOnCuda { .. })
    ));
    let zx = Tensor::from_storage(TensorStorage::cpu(vec![2.0_f64, 3.0]), vec![2], false)
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();
    let zq = Tensor::from_storage(TensorStorage::cpu(vec![1.0_f64, 1.0]), vec![2], false)
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();
    assert!(matches!(
        special::zeta(&zx, &zq),
        Err(FerrotorchError::NotImplementedOnCuda { .. })
    ));
}
