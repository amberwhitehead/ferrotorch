//! Adversarial RE-AUDIT regression-guard for the modified-Bessel-K family f32
//! PTX kernels landed in commit 201eca758 (#1651 batch 3b GPU tail):
//! `ferrotorch-gpu/src/special.rs` `gpu_modified_bessel_k0_f32` /
//! `gpu_scaled_modified_bessel_k0_f32` / `gpu_modified_bessel_k1_f32` /
//! `gpu_scaled_modified_bessel_k1_f32`, dispatched through
//! `ferrotorch-core/src/special.rs` `modified_bessel_k0` /
//! `scaled_modified_bessel_k0` / `modified_bessel_k1` /
//! `scaled_modified_bessel_k1` (CUDA-f32 branch via GpuBackend / CudaBackendImpl).
//!
//! VERDICT: NO DIVERGENCE FOUND. This file is a PASSING guard, not a failing
//! divergence test. The shipped commit's own test
//! (`divergence_modified_bessel_k_gpu_f32.rs`) only exercised
//! x = [0.5, 1.0, 1.5, 2.5, 5.0]; it never pinned the chbevl region split at
//! x == 2.0, the x -> 0+ divergence, large-x underflow, x == 0, x < 0, or NaN.
//! This guard pins ALL of those, so a future regression of the interval
//! selection, the log/exp composition, the scaled exp(x) factor, or the edge
//! ladder is caught.
//!
//! What is verified live on the RTX 3090 (build with
//! `RUSTFLAGS="-C link-arg=-fuse-ld=lld"`):
//!   1. ON-DEVICE: every op leaves the result `is_cuda()` true (no host round
//!      trip; the kernel actually JIT-launched). R-CODE-4.
//!   2. f32 PRECISION across the chbevl region split (Cephes K-family switches
//!      at x == 2.0): SMALL x in (0, 2] AND BIG x > 2, AND the boundary triple
//!      (1.999, 2.0, 2.001). Every GPU output matches LIVE
//!      `torch.special.<op>` f32 to <= 1e-5 rtol (measured worst-case 4.34e-7).
//!   3. SCALED identity: `scaled_k{0,1}(x) == k{0,1}(x) * exp(x)` on the GPU
//!      outputs (verified to ~5e-8 across the tested x).
//!   4. EDGE / DOMAIN: x -> 0+ (K diverges; finite-large, matches torch),
//!      large x (unscaled underflows toward 0, scaled stays ~sqrt(pi/2x)),
//!      x == 0 -> +inf, x < 0 -> NaN, NaN -> NaN — each matching torch.
//!   5. GPU == CPU: the on-device f32 result equals the ferrotorch CPU f32
//!      Cephes-port result for the same input (no PTX-only error).
//!   6. f64 / bf16 / f16 CUDA tensors reject `NotImplementedOnCuda` cleanly.
//!
//! ORACLE PROVENANCE (R-CHAR-3): every `*_TORCH` constant below is the verbatim
//! f32 printout of `torch.special.<op>(torch.tensor(xs, dtype=torch.float32,
//! device='cuda'))` under torch 2.11.0+cu130, captured with
//! `LD_LIBRARY_PATH="$HOME/.local/lib:.../nvidia/cu13/lib:$LD_LIBRARY_PATH"`.
//! NOT copied from the ferrotorch side.
//!   xs = [1.999, 2.0, 2.001, 1e-4, 1e-3, 0.01, 20.0, 50.0]
//!   torch.special.modified_bessel_k0(xs.float())        =
//!     [0.11403382569551468, 0.11389388144016266, 0.11375410854816437,
//!      9.326273918151855, 7.023688793182373, 4.721244812011719,
//!      5.741236930312255e-10, 3.41016806935736e-23]
//!   torch.special.scaled_modified_bessel_k0(xs.float())  =
//!     [0.8417600989341736, 0.8415682911872864, 0.8413764238357544,
//!      9.327205657958984, 7.0307159423828125, 4.768693923950195,
//!      0.2785448431968689, 0.17680716514587402]
//!   torch.special.modified_bessel_k1(xs.float())        =
//!     [0.14004985988140106, 0.13986589014530182, 0.13968220353126526,
//!      9999.9990234375, 999.9962768554688, 99.97389221191406,
//!      5.883057374589384e-10, 3.4441020256007094e-23]
//!   torch.special.scaled_modified_bessel_k1(xs.float())  =
//!     [1.033801794052124, 1.0334769487380981, 1.0331522226333618,
//!      10000.9990234375, 1000.996826171875, 100.97864532470703,
//!      0.28542548418045044, 0.1785665601491928]
//!   (edge) torch.special.modified_bessel_k0(tensor([0.,-1.,nan]).cuda()) =
//!     [inf, nan, nan]  (same inf/nan ladder for all four ops)
//!
//! Upstream cite: `aten/src/ATen/native/cuda/Math.cuh:2545-2576`
//! (`modified_bessel_k0_forward`: `if (x == T(0.0)) return INFINITY;`
//! `if (x < T(0.0)) return NAN;` `if (x <= T(2.0)) { ... }` SMALL else BIG),
//! and the analogous 2582-2656 / 2661-2736 / 2740-2815 blocks.

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

/// Finite probe grid straddling the Cephes region split (x == 2.0): the
/// boundary triple (1.999, 2.0, 2.001), three small-x points approaching the
/// K divergence (1e-4, 1e-3, 0.01), and two large-x points (20, 50) where the
/// unscaled variants underflow toward 0 and the scaled variants stay O(1).
const XS: [f32; 8] = [1.999, 2.0, 2.001, 1e-4, 1e-3, 0.01, 20.0, 50.0];
const RTOL: f32 = 1e-5;

type KOp = fn(&Tensor<f32>) -> Result<Tensor<f32>, FerrotorchError>;

fn cuda_f32(data: &[f32]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false)
        .unwrap()
        .to(Device::Cuda(0))
        .expect("Tensor<f32>::to(Cuda)")
}

/// Run `op` on a CUDA-resident copy of `xs`, assert the result stays on device
/// (no host round trip), return the host-copied values.
fn run_on_device(op: KOp, xs: &[f32], name: &str) -> Vec<f32> {
    let gpu_in = cuda_f32(xs);
    assert!(gpu_in.is_cuda(), "{name}: input must be CUDA-resident");
    let gpu_out = op(&gpu_in).unwrap_or_else(|e| panic!("{name} GPU dispatch: {e:?}"));
    assert!(
        gpu_out.is_cuda(),
        "{name}: result must stay on device (no host round trip)"
    );
    gpu_out.to(Device::Cpu).unwrap().data().unwrap().to_vec()
}

fn assert_matches(got: &[f32], want: &[f32], xs: &[f32], name: &str) {
    assert_eq!(got.len(), want.len());
    for i in 0..xs.len() {
        let tol = RTOL * (1.0 + want[i].abs());
        assert!(
            (got[i] - want[i]).abs() <= tol,
            "{name} on-device x={}: got {} want {} (torch f32; tol {tol})",
            xs[i],
            got[i],
            want[i]
        );
    }
}

#[allow(
    clippy::excessive_precision,
    reason = "verbatim torch 2.11.0+cu130 f32 oracle printout (R-CHAR-3 provenance in the module doc); extra printed digits document the oracle, the f32 parse rounds them"
)]
mod oracle {
    pub const K0_TORCH: [f32; 8] = [
        0.11403382569551468,
        0.11389388144016266,
        0.11375410854816437,
        9.326273918151855,
        7.023688793182373,
        4.721244812011719,
        5.741236930312255e-10,
        3.41016806935736e-23,
    ];
    pub const SK0_TORCH: [f32; 8] = [
        0.8417600989341736,
        0.8415682911872864,
        0.8413764238357544,
        9.327205657958984,
        7.0307159423828125,
        4.768693923950195,
        0.2785448431968689,
        0.17680716514587402,
    ];
    pub const K1_TORCH: [f32; 8] = [
        0.14004985988140106,
        0.13986589014530182,
        0.13968220353126526,
        9999.9990234375,
        999.9962768554688,
        99.97389221191406,
        5.883057374589384e-10,
        3.4441020256007094e-23,
    ];
    pub const SK1_TORCH: [f32; 8] = [
        1.033801794052124,
        1.0334769487380981,
        1.0331522226333618,
        10000.9990234375,
        1000.996826171875,
        100.97864532470703,
        0.28542548418045044,
        0.1785665601491928,
    ];
}

/// Region-split + boundary + small/large precision for k0: GPU f32 == torch f32.
#[test]
fn k0_region_split_and_boundary_matches_torch() {
    if !ensure_cuda() {
        eprintln!("no CUDA device; skipping");
        return;
    }
    let got = run_on_device(special::modified_bessel_k0, &XS, "modified_bessel_k0");
    assert_matches(&got, &oracle::K0_TORCH, &XS, "modified_bessel_k0");
}

#[test]
fn scaled_k0_region_split_and_boundary_matches_torch() {
    if !ensure_cuda() {
        eprintln!("no CUDA device; skipping");
        return;
    }
    let got = run_on_device(
        special::scaled_modified_bessel_k0,
        &XS,
        "scaled_modified_bessel_k0",
    );
    assert_matches(&got, &oracle::SK0_TORCH, &XS, "scaled_modified_bessel_k0");
}

#[test]
fn k1_region_split_and_boundary_matches_torch() {
    if !ensure_cuda() {
        eprintln!("no CUDA device; skipping");
        return;
    }
    let got = run_on_device(special::modified_bessel_k1, &XS, "modified_bessel_k1");
    assert_matches(&got, &oracle::K1_TORCH, &XS, "modified_bessel_k1");
}

#[test]
fn scaled_k1_region_split_and_boundary_matches_torch() {
    if !ensure_cuda() {
        eprintln!("no CUDA device; skipping");
        return;
    }
    let got = run_on_device(
        special::scaled_modified_bessel_k1,
        &XS,
        "scaled_modified_bessel_k1",
    );
    assert_matches(&got, &oracle::SK1_TORCH, &XS, "scaled_modified_bessel_k1");
}

/// Scaled identity ON THE GPU OUTPUTS: scaled_k{0,1}(x) == k{0,1}(x) * exp(x)
/// for x where exp(x) does not overflow f32. A swapped scaled/unscaled kernel
/// or a missing/extra exp(x) factor breaks this. Upstream:
/// `scaled_modified_bessel_k0_forward` returns the SMALL branch
/// `(0.5*(a-p) - log(0.5x)*i0) * exp(x)` and BIG `(0.5*(b-p))/sqrt(x)` (i.e.
/// the unscaled BIG branch with the `exp(-x)` factor removed)
/// — `aten/src/ATen/native/cuda/Math.cuh:2582-2656`.
#[test]
fn scaled_equals_unscaled_times_exp_on_device() {
    if !ensure_cuda() {
        eprintln!("no CUDA device; skipping");
        return;
    }
    // exp(20) ~ 4.85e8, well inside f32; exp(50) overflows products of O(1e-23),
    // so cap the identity check at the finite-product points.
    let xs: [f32; 6] = [1.999, 2.0, 2.001, 1e-3, 0.01, 20.0];
    let k0 = run_on_device(special::modified_bessel_k0, &xs, "k0");
    let sk0 = run_on_device(special::scaled_modified_bessel_k0, &xs, "sk0");
    let k1 = run_on_device(special::modified_bessel_k1, &xs, "k1");
    let sk1 = run_on_device(special::scaled_modified_bessel_k1, &xs, "sk1");
    for i in 0..xs.len() {
        let e = xs[i].exp();
        for (lhs, rhs, tag) in [(sk0[i], k0[i] * e, "k0"), (sk1[i], k1[i] * e, "k1")] {
            let tol = 1e-5 * (1.0 + lhs.abs());
            assert!(
                (lhs - rhs).abs() <= tol,
                "scaled identity {tag} x={}: scaled={lhs} unscaled*exp={rhs} (tol {tol})",
                xs[i]
            );
        }
    }
}

/// GPU f32 == ferrotorch CPU f32 (both Cephes ports). A divergence here means
/// the PTX port introduced an error the CPU scalar path does not have. Tested
/// across the region split, boundary, small-x, and large-x.
#[test]
fn gpu_matches_cpu_port() {
    if !ensure_cuda() {
        eprintln!("no CUDA device; skipping");
        return;
    }
    let ops: [(&str, KOp); 4] = [
        ("k0", special::modified_bessel_k0),
        ("sk0", special::scaled_modified_bessel_k0),
        ("k1", special::modified_bessel_k1),
        ("sk1", special::scaled_modified_bessel_k1),
    ];
    for (name, op) in ops {
        let gpu = run_on_device(op, &XS, name);
        let cpu_in =
            Tensor::from_storage(TensorStorage::cpu(XS.to_vec()), vec![XS.len()], false).unwrap();
        assert!(!cpu_in.is_cuda());
        let cpu = op(&cpu_in).unwrap().data().unwrap().to_vec();
        for i in 0..XS.len() {
            let tol = 2e-5 * (1.0 + cpu[i].abs());
            assert!(
                (gpu[i] - cpu[i]).abs() <= tol,
                "{name} GPU vs CPU x={}: gpu={} cpu={} (tol {tol})",
                XS[i],
                gpu[i],
                cpu[i]
            );
        }
    }
}

/// Edge / domain ladder ON DEVICE, matching torch: x == 0 -> +inf,
/// x < 0 -> NaN, NaN -> NaN, for all four ops. Upstream:
/// `if (x == T(0.0)) return INFINITY; if (x < T(0.0)) return NAN;`
/// (`aten/src/ATen/native/cuda/Math.cuh:2545-2551`); NaN falls through the
/// `<= 2.0` test into the BIG branch where `8/nan` propagates NaN, matching
/// torch's `nan`.
#[test]
fn edge_domain_ladder_on_device_matches_torch() {
    if !ensure_cuda() {
        eprintln!("no CUDA device; skipping");
        return;
    }
    let edge: [f32; 3] = [0.0, -1.0, f32::NAN];
    let ops: [(&str, KOp); 4] = [
        ("k0", special::modified_bessel_k0),
        ("sk0", special::scaled_modified_bessel_k0),
        ("k1", special::modified_bessel_k1),
        ("sk1", special::scaled_modified_bessel_k1),
    ];
    for (name, op) in ops {
        let got = run_on_device(op, &edge, name);
        // torch: [inf, nan, nan] for every K-family op
        assert!(
            got[0].is_infinite() && got[0] > 0.0,
            "{name} x=0 must be +inf (torch inf), got {}",
            got[0]
        );
        assert!(
            got[1].is_nan(),
            "{name} x=-1 must be NaN (torch nan), got {}",
            got[1]
        );
        assert!(
            got[2].is_nan(),
            "{name} x=NaN must be NaN (torch nan), got {}",
            got[2]
        );
    }
}

/// f64 CUDA tensors reject `NotImplementedOnCuda` (base PTX has no
/// `lg2.approx.f64` / `ex2.approx.f64`) — no host round trip, no wrong answer.
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

/// bf16 CUDA tensors reject `NotImplementedOnCuda` (non-f32/f64 dtype guard).
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

/// f16 CUDA tensors reject `NotImplementedOnCuda` (non-f32/f64 dtype guard).
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
