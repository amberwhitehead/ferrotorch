//! CORE-122 (#1816, CLASS-V High) regression battery: CPU `cdist`
//! (`ferrotorch-core/src/ops/tensor_ops.rs`) must compute every p-norm branch
//! like torch — in particular `p = 0` (zero-"norm": COUNT of unequal
//! coordinates) and `p = inf` (max |diff|) — instead of pushing every `p`
//! through the generic `sum(|d|^p)^(1/p)` formula.
//!
//! Pre-fix CPU behavior: `p = 0` evaluates `|d|^0 = 1` per coordinate
//! (counting EQUAL coordinates too) and then `sum^(1/0) = sum^inf`
//! (`inf`/garbage); `p = inf` degenerates through `|d|^inf` and a zero final
//! exponent. The GPU dispatch has dedicated `{one, two, inf, general}`
//! finishers (`ferrotorch-gpu/src/distance.rs`, mirroring
//! `DistanceKernel.cu` `dists<scalar_t>::{p,one,two,inf}`), so the two
//! devices disagreed.
//!
//! LIVE torch 2.11.0+cu130 oracle (quoted; identical on cpu and cuda):
//! ```text
//! >>> x1 = torch.tensor([[0.0, 0.5, 2.0], [1.0, 1.0, 1.0]])
//! >>> x2 = torch.tensor([[0.0, 1.5, 0.5], [1.0, 1.0, 1.0]])
//! >>> torch.cdist(x1, x2, p=0.).tolist()
//! [[2.0, 3.0], [3.0, 0.0]]
//! >>> torch.cdist(x1, x2, p=1.).tolist()
//! [[2.5, 2.5], [2.0, 0.0]]
//! >>> torch.cdist(x1, x2, p=2.).tolist()
//! [[1.8027756214141846, 1.5], [1.2247449159622192, 0.0]]
//! >>> torch.cdist(x1, x2, p=float('inf')).tolist()
//! [[1.5, 1.0], [1.0, 0.0]]
//! >>> torch.cdist(x1, x2, p=3.).tolist()
//! [[1.6355332136154175, 1.285640835762024], [1.0772173404693604, 0.0]]
//! >>> torch.cdist(x1, x2, p=0.5).tolist()
//! [[4.949489593505859, 7.328427314758301], [5.828427791595459, 0.0]]
//! >>> torch.cdist(x1, x2, p=-1.)
//! RuntimeError: cdist only supports non-negative p values
//! ```
//! The vectors deliberately include an all-equal row pair (`[1,1,1]` vs
//! `[1,1,1]`), an equal coordinate inside an unequal pair (`0.0` vs `0.0`),
//! and |diff| both below one (`0.5`) and above one (`1.5`, `2.0`).

use ferrotorch_core::ops::tensor_ops::cdist;
use ferrotorch_core::{FerrotorchError, Tensor, TensorStorage};

const X1: [f32; 6] = [0.0, 0.5, 2.0, 1.0, 1.0, 1.0];
const X2: [f32; 6] = [0.0, 1.5, 0.5, 1.0, 1.0, 1.0];

fn cpu_pair() -> (Tensor<f32>, Tensor<f32>) {
    let x1 =
        Tensor::from_storage(TensorStorage::cpu(X1.to_vec()), vec![2, 3], false).expect("cpu x1");
    let x2 =
        Tensor::from_storage(TensorStorage::cpu(X2.to_vec()), vec![2, 3], false).expect("cpu x2");
    (x1, x2)
}

/// Tolerance: oracle values are O(1)-O(10); f32 eps is ~1.2e-7, the m=3
/// accumulation plus one `powf` round-trip keeps the error within a few ULP
/// of 1e-6 — 1e-5 absolute bounds that with margin and stays far below the
/// pre-fix divergence (inf / wrong branch entirely).
const TOL: f32 = 1e-5;

#[track_caller]
fn assert_close(name: &str, got: &[f32], expected: &[f32]) {
    assert_eq!(got.len(), expected.len(), "{name} length");
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() <= TOL,
            "{name}[{i}]: got {g}, expected {e} (tol {TOL})"
        );
    }
}

fn oracle(p: f64) -> &'static [f32] {
    // Values quoted from the LIVE torch session in the module header.
    if p == 0.0 {
        &[2.0, 3.0, 3.0, 0.0]
    } else if p == 1.0 {
        &[2.5, 2.5, 2.0, 0.0]
    } else if p == 2.0 {
        &[1.802_775_6, 1.5, 1.224_744_9, 0.0]
    } else if p == f64::INFINITY {
        &[1.5, 1.0, 1.0, 0.0]
    } else if p == 3.0 {
        &[1.635_533_2, 1.285_640_8, 1.077_217_3, 0.0]
    } else if p == 0.5 {
        &[4.949_489_6, 7.328_427_3, 5.828_428, 0.0]
    } else {
        panic!("no oracle for p={p}")
    }
}

// ─────────────────────────────────────────────────────────────────────────
// CPU lane. Pre-fix observed (R-AHON-1 probe): p=0 returns inf everywhere
// (counts equal coordinates then sum^inf), p=inf returns the wrong finisher.
// ─────────────────────────────────────────────────────────────────────────

/// p = 0: zero-"norm" — COUNT of unequal coordinates (torch oracle above).
#[test]
fn cdist_cpu_p_zero_counts_unequal_coordinates() {
    let (x1, x2) = cpu_pair();
    let r = cdist(&x1, &x2, 0.0).expect("cdist p=0");
    assert_eq!(r.shape(), &[2, 2]);
    assert_close("cdist p=0", r.data().expect("data"), oracle(0.0));
}

/// p = inf: max |diff| (torch oracle above).
#[test]
fn cdist_cpu_p_inf_is_max_abs_diff() {
    let (x1, x2) = cpu_pair();
    let r = cdist(&x1, &x2, f64::INFINITY).expect("cdist p=inf");
    assert_close(
        "cdist p=inf",
        r.data().expect("data"),
        oracle(f64::INFINITY),
    );
}

/// The explicit p = 1 / 2 and general (3, 0.5) branches keep matching torch.
#[test]
fn cdist_cpu_finite_p_branches_match_torch() {
    let (x1, x2) = cpu_pair();
    for p in [1.0, 2.0, 3.0, 0.5] {
        let r = cdist(&x1, &x2, p).unwrap_or_else(|e| panic!("cdist p={p}: {e:?}"));
        assert_close(&format!("cdist p={p}"), r.data().expect("data"), oracle(p));
    }
}

/// Negative p is rejected with a structured error (torch:
/// `RuntimeError: cdist only supports non-negative p values`).
#[test]
fn cdist_cpu_negative_p_is_invalid_argument() {
    let (x1, x2) = cpu_pair();
    match cdist(&x1, &x2, -1.0) {
        Err(FerrotorchError::InvalidArgument { message }) => {
            assert!(
                message.contains("non-negative"),
                "message should state the non-negative contract, got: {message}"
            );
        }
        other => panic!("cdist p=-1: expected InvalidArgument, got {other:?}"),
    }
}

// ─────────────────────────────────────────────────────────────────────────
// CUDA lane: CPU and GPU must agree with each other and with torch for the
// kernel-covered norms (f32: {1, 2, inf, general}); the uncovered p=0
// count-norm keeps its loud NotImplementedOnCuda contract (REQ-6); negative
// p is rejected BEFORE the kernel dispatch on both devices.
// ─────────────────────────────────────────────────────────────────────────

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use ferrotorch_core::Device;
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for the CORE-122 GPU pins");
        });
    }

    fn cuda_pair() -> (Tensor<f32>, Tensor<f32>) {
        let (x1, x2) = cpu_pair();
        (
            x1.to(Device::Cuda(0)).expect("upload x1"),
            x2.to(Device::Cuda(0)).expect("upload x2"),
        )
    }

    /// CPU == GPU == torch for every kernel-covered norm (R-ORACLE-3 device
    /// assertion on the GPU result).
    #[test]
    fn cdist_cuda_matches_cpu_and_torch() {
        ensure_cuda_backend();
        let (cx1, cx2) = cpu_pair();
        let (gx1, gx2) = cuda_pair();
        for p in [1.0, 2.0, f64::INFINITY, 3.0] {
            let g = cdist(&gx1, &gx2, p).unwrap_or_else(|e| panic!("cuda cdist p={p}: {e:?}"));
            assert_eq!(g.device(), Device::Cuda(0), "cdist p={p} output device");
            let g_host = g.cpu().expect("D2H readback");
            let g_data = g_host.data().expect("data").to_vec();
            assert_close(&format!("cuda cdist p={p}"), &g_data, oracle(p));

            let c = cdist(&cx1, &cx2, p).unwrap_or_else(|e| panic!("cpu cdist p={p}: {e:?}"));
            let c_data = c.data().expect("data").to_vec();
            assert_close(&format!("cpu-vs-gpu cdist p={p}"), &g_data, &c_data);
        }
    }

    /// p = 0 count-norm has no f32 kernel: loud `NotImplementedOnCuda`
    /// (pinned single contract per R-ORACLE-4; torch computes this on cuda —
    /// the gap is the documented REQ-6 kernel-coverage boundary, not a
    /// silent fallback).
    #[test]
    fn cdist_cuda_p_zero_is_loud_not_implemented() {
        ensure_cuda_backend();
        let (gx1, gx2) = cuda_pair();
        match cdist(&gx1, &gx2, 0.0) {
            Err(FerrotorchError::NotImplementedOnCuda { op }) => assert_eq!(op, "cdist"),
            other => panic!("cuda cdist p=0: expected NotImplementedOnCuda, got {other:?}"),
        }
    }

    /// Negative p must be rejected before any kernel launch (torch raises on
    /// cuda too).
    #[test]
    fn cdist_cuda_negative_p_is_invalid_argument() {
        ensure_cuda_backend();
        let (gx1, gx2) = cuda_pair();
        match cdist(&gx1, &gx2, -1.0) {
            Err(FerrotorchError::InvalidArgument { .. }) => {}
            other => panic!("cuda cdist p=-1: expected InvalidArgument, got {other:?}"),
        }
    }
}
