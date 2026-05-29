//! Live-GPU verification of the Airy-Ai (unary) and Hurwitz-zeta (binary) f32
//! PTX kernels landed for the #1651 GPU tail (`ferrotorch-gpu/src/special.rs`):
//! `airy_ai` / `zeta`.
//!
//! Each `verify_*` test runs the FULL `ferrotorch_core::special::*` Tensor
//! dispatch path on genuinely CUDA-resident f32 input(s) and asserts:
//!   (a) the result stays on device (`Tensor::is_cuda()` — the GPU `_f32`
//!       backend op returned a CUDA handle, no host round trip, R-CODE-4), AND
//!   (b) the values match LIVE `torch.special.{airy_ai,zeta}` (torch 2.11.0,
//!       f32), to ~1e-5 rtol / ~5e-7 atol (f32 transcendental). The GPU f32
//!       kernel evaluates in f32 throughout, so it lands near torch's f32
//!       precision (NOT the ferrotorch CPU f64-then-narrow precision); that is
//!       expected (commit 4d516a3bd).
//!
//! Reference values are LIVE torch 2.11.0 f32 outputs (R-CHAR-3), NOT copied
//! from ferrotorch. Produced under
//! `LD_LIBRARY_PATH="$HOME/.local/lib:$LD_LIBRARY_PATH"`:
//!
//!   import torch
//!   ax = torch.tensor([-5.,-3.,-2.5,-1.,0.,0.5,1.,2.5,5.], dtype=torch.float32)
//!   torch.special.airy_ai(ax) =
//!     [ 3.50760967e-01, -3.78814310e-01, -1.12324826e-01,  5.35560906e-01,
//!       3.55028063e-01,  2.31693625e-01,  1.35292381e-01,  1.57259256e-02,
//!       1.08344429e-04]
//!   torch.special.airy_ai([inf,-inf,nan,200.].float()) = [nan, nan, nan, 0.]
//!
//!   zx = torch.tensor([2.,3.,2.,4.,1.5,2.5], dtype=torch.float32)
//!   zq = torch.tensor([1.,1.,2.,0.5,2.,3.], dtype=torch.float32)
//!   torch.special.zeta(zx, zq) =
//!     [1.64493406e+00, 1.20205688e+00, 6.44934058e-01, 1.62348480e+01,
//!      1.61237538e+00, 1.64710566e-01]
//!   zex = torch.tensor([1.,0.5,2.,3.,2.5], dtype=torch.float32)
//!   zeq = torch.tensor([2.,1.,0.,-2.,-1.5], dtype=torch.float32)
//!   torch.special.zeta(zex, zeq) = [inf, nan, inf, inf, nan]
//!
//! airy spans all three regions: oscillatory (x=-5,-3,-2.5 < -2.09), central
//! ([-1,1]), decaying (x=2.5,5 >= 2.09). zeta spans x>1 convergent cases and
//! the full `x<=1` / `q<=0` edge ladder. Upstream:
//! `aten/src/ATen/native/cuda/Math.cuh:1280-1459` (airy_ai_forward) and
//! `:299-383` (zeta).

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

fn cuda_f32(data: &[f32]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false)
        .unwrap()
        .to(Device::Cuda(0))
        .expect("Tensor<f32>::to(Cuda)")
}

fn assert_close(got: &[f32], want: &[f32], rtol: f32, atol: f32, name: &str) {
    assert_eq!(got.len(), want.len(), "{name}: length mismatch");
    for i in 0..want.len() {
        let tol = atol + rtol * want[i].abs();
        assert!(
            (got[i] - want[i]).abs() <= tol,
            "{name} idx {i}: got {} want {} (torch f32; tol {tol})",
            got[i],
            want[i]
        );
    }
}

#[test]
#[allow(
    clippy::excessive_precision,
    reason = "literals are the verbatim torch 2.11.0 f32 oracle printout (R-CHAR-3 provenance in the module doc)"
)]
fn verify_airy_ai_gpu_f32_on_device_matches_torch() {
    if !ensure_cuda() {
        eprintln!("no CUDA device; skipping");
        return;
    }
    let xs: [f32; 9] = [-5.0, -3.0, -2.5, -1.0, 0.0, 0.5, 1.0, 2.5, 5.0];
    let gpu_in = cuda_f32(&xs);
    assert!(gpu_in.is_cuda(), "airy_ai: input must be CUDA-resident");
    let gpu_out = special::airy_ai(&gpu_in).expect("airy_ai GPU dispatch");
    assert!(
        gpu_out.is_cuda(),
        "airy_ai: result must stay on device (no host round trip)"
    );
    let got = gpu_out.to(Device::Cpu).unwrap().data().unwrap().to_vec();
    let want: [f32; 9] = [
        3.50760967e-01,
        -3.78814310e-01,
        -1.12324826e-01,
        5.35560906e-01,
        3.55028063e-01,
        2.31693625e-01,
        1.35292381e-01,
        1.57259256e-02,
        1.08344429e-04,
    ];
    assert_close(&got, &want, 1e-5, 5e-7, "airy_ai");
}

#[test]
fn verify_airy_ai_gpu_f32_edges_matches_torch() {
    if !ensure_cuda() {
        eprintln!("no CUDA device; skipping");
        return;
    }
    // [inf, -inf, nan, 200] -> [nan, nan, nan, 0] (isinf -> NaN; x>103.892 -> 0).
    let xs: [f32; 4] = [f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 200.0];
    let gpu_in = cuda_f32(&xs);
    let gpu_out = special::airy_ai(&gpu_in).expect("airy_ai GPU dispatch");
    assert!(
        gpu_out.is_cuda(),
        "airy_ai edges: result must stay on device"
    );
    let got = gpu_out.to(Device::Cpu).unwrap().data().unwrap().to_vec();
    assert!(got[0].is_nan(), "airy_ai(+inf) == NaN: {}", got[0]);
    assert!(got[1].is_nan(), "airy_ai(-inf) == NaN: {}", got[1]);
    assert!(got[2].is_nan(), "airy_ai(NaN) == NaN: {}", got[2]);
    assert_eq!(got[3], 0.0, "airy_ai(200) == 0 (x>103.892 branch)");
}

#[test]
#[allow(
    clippy::excessive_precision,
    reason = "verbatim torch 2.11.0 f32 oracle printout (R-CHAR-3)"
)]
fn verify_zeta_gpu_f32_on_device_matches_torch() {
    if !ensure_cuda() {
        eprintln!("no CUDA device; skipping");
        return;
    }
    let xs: [f32; 6] = [2.0, 3.0, 2.0, 4.0, 1.5, 2.5];
    let qs: [f32; 6] = [1.0, 1.0, 2.0, 0.5, 2.0, 3.0];
    let gx = cuda_f32(&xs);
    let gq = cuda_f32(&qs);
    assert!(
        gx.is_cuda() && gq.is_cuda(),
        "zeta: inputs must be CUDA-resident"
    );
    let gpu_out = special::zeta(&gx, &gq).expect("zeta GPU dispatch");
    assert!(
        gpu_out.is_cuda(),
        "zeta: result must stay on device (no host round trip)"
    );
    let got = gpu_out.to(Device::Cpu).unwrap().data().unwrap().to_vec();
    let want: [f32; 6] = [
        1.64493406e+00,
        1.20205688e+00,
        6.44934058e-01,
        1.62348480e+01,
        1.61237538e+00,
        1.64710566e-01,
    ];
    assert_close(&got, &want, 1e-4, 5e-7, "zeta");
}

#[test]
fn verify_zeta_gpu_f32_edge_ladder_matches_torch() {
    if !ensure_cuda() {
        eprintln!("no CUDA device; skipping");
        return;
    }
    // x==1 -> +inf; x<1 -> NaN; q<=0 integer -> +inf (x2); q<=0 non-int,
    // x non-int -> NaN.
    let xs: [f32; 5] = [1.0, 0.5, 2.0, 3.0, 2.5];
    let qs: [f32; 5] = [2.0, 1.0, 0.0, -2.0, -1.5];
    let gpu_out = special::zeta(&cuda_f32(&xs), &cuda_f32(&qs)).expect("zeta GPU dispatch");
    assert!(gpu_out.is_cuda(), "zeta edges: result must stay on device");
    let d = gpu_out.to(Device::Cpu).unwrap().data().unwrap().to_vec();
    assert!(
        d[0].is_infinite() && d[0] > 0.0,
        "zeta(1,2) == +inf: {}",
        d[0]
    );
    assert!(d[1].is_nan(), "zeta(0.5,1) == NaN: {}", d[1]);
    assert!(
        d[2].is_infinite() && d[2] > 0.0,
        "zeta(2,0) == +inf: {}",
        d[2]
    );
    assert!(
        d[3].is_infinite() && d[3] > 0.0,
        "zeta(3,-2) == +inf: {}",
        d[3]
    );
    assert!(d[4].is_nan(), "zeta(2.5,-1.5) == NaN: {}", d[4]);
}

/// f64 CUDA tensors must cleanly reject with `NotImplementedOnCuda` (base PTX
/// has no `lg2.approx.f64`/`ex2.approx.f64`), for both ops — never a host round
/// trip, never a wrong-precision answer.
#[test]
fn f64_cuda_rejects_not_implemented() {
    if !ensure_cuda() {
        eprintln!("no CUDA device; skipping");
        return;
    }
    let airy_in = Tensor::from_storage(TensorStorage::cpu(vec![1.0_f64, 2.5]), vec![2], false)
        .unwrap()
        .to(Device::Cuda(0))
        .expect("Tensor<f64>::to(Cuda)");
    assert!(airy_in.is_cuda());
    assert!(
        matches!(
            special::airy_ai(&airy_in),
            Err(FerrotorchError::NotImplementedOnCuda { .. })
        ),
        "f64 CUDA airy_ai must be NotImplementedOnCuda"
    );
    let zx = Tensor::from_storage(TensorStorage::cpu(vec![2.0_f64, 3.0]), vec![2], false)
        .unwrap()
        .to(Device::Cuda(0))
        .expect("Tensor<f64>::to(Cuda)");
    let zq = Tensor::from_storage(TensorStorage::cpu(vec![1.0_f64, 1.0]), vec![2], false)
        .unwrap()
        .to(Device::Cuda(0))
        .expect("Tensor<f64>::to(Cuda)");
    assert!(
        matches!(
            special::zeta(&zx, &zq),
            Err(FerrotorchError::NotImplementedOnCuda { .. })
        ),
        "f64 CUDA zeta must be NotImplementedOnCuda"
    );
}

/// bf16 / f16 CUDA tensors must cleanly reject with `NotImplementedOnCuda` (the
/// dispatch rejects any non-f32/f64 dtype before touching the device).
#[test]
fn bf16_f16_cuda_rejects_not_implemented() {
    if !ensure_cuda() {
        eprintln!("no CUDA device; skipping");
        return;
    }
    let bf: Vec<half::bf16> = [1.0_f32, 2.5]
        .iter()
        .copied()
        .map(half::bf16::from_f32)
        .collect();
    let airy_bf = Tensor::from_storage(TensorStorage::cpu(bf), vec![2], false)
        .unwrap()
        .to(Device::Cuda(0))
        .expect("Tensor<bf16>::to(Cuda)");
    assert!(matches!(
        special::airy_ai(&airy_bf),
        Err(FerrotorchError::NotImplementedOnCuda { .. })
    ));

    let hf: Vec<half::f16> = [1.0_f32, 2.5]
        .iter()
        .copied()
        .map(half::f16::from_f32)
        .collect();
    let airy_hf = Tensor::from_storage(TensorStorage::cpu(hf), vec![2], false)
        .unwrap()
        .to(Device::Cuda(0))
        .expect("Tensor<f16>::to(Cuda)");
    assert!(matches!(
        special::airy_ai(&airy_hf),
        Err(FerrotorchError::NotImplementedOnCuda { .. })
    ));

    let zbf_x: Vec<half::bf16> = [2.0_f32, 3.0]
        .iter()
        .copied()
        .map(half::bf16::from_f32)
        .collect();
    let zbf_q: Vec<half::bf16> = [1.0_f32, 1.0]
        .iter()
        .copied()
        .map(half::bf16::from_f32)
        .collect();
    let zx = Tensor::from_storage(TensorStorage::cpu(zbf_x), vec![2], false)
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();
    let zq = Tensor::from_storage(TensorStorage::cpu(zbf_q), vec![2], false)
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();
    assert!(matches!(
        special::zeta(&zx, &zq),
        Err(FerrotorchError::NotImplementedOnCuda { .. })
    ));
}

/// The CPU path is unchanged: identical CPU f32 tensors produce the same
/// torch-matching values (the GPU dispatch is purely additive — `is_cuda()`
/// false short-circuits to the scalar `unary_map` / `binary_map`).
#[test]
#[allow(
    clippy::excessive_precision,
    reason = "verbatim torch 2.11.0 f32 oracle printout (R-CHAR-3)"
)]
fn cpu_path_unchanged_matches_torch_f32() {
    let xs: [f32; 9] = [-5.0, -3.0, -2.5, -1.0, 0.0, 0.5, 1.0, 2.5, 5.0];
    let cpu_in = Tensor::from_storage(TensorStorage::cpu(xs.to_vec()), vec![9], false).unwrap();
    assert!(!cpu_in.is_cuda());
    let got = special::airy_ai(&cpu_in).unwrap().data().unwrap().to_vec();
    let want: [f32; 9] = [
        3.50760967e-01,
        -3.78814310e-01,
        -1.12324826e-01,
        5.35560906e-01,
        3.55028063e-01,
        2.31693625e-01,
        1.35292381e-01,
        1.57259256e-02,
        1.08344429e-04,
    ];
    // CPU is f64-then-narrow (more precise than torch f32); still well inside tol.
    assert_close(&got, &want, 1e-4, 5e-6, "airy_ai CPU");
}
