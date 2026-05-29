//! Regression guard: `ferrotorch_core::cdist` (p=2) is MORE PRECISE than
//! `torch.cdist`'s DEFAULT `compute_mode` for large inputs — and that is
//! ACCEPTED per the goal.md precision contract (ferrotorch may be more precise
//! than PyTorch; we only forbid being LESS precise / diverging in a way that
//! loses correctness).
//!
//! ## Why torch's default is worse here (a documented torch numerical wart)
//!
//! Upstream `aten/src/ATen/native/Distance.cpp:105`:
//!   `if (!(p == 2 && (mode == 1 || (mode == 0 && (r1 > 25 || r2 > 25)))))`
//! → `Distance.cpp:138` `at::_euclidean_dist(x1, x2)`.
//! For the USER-FACING default `torch.cdist(x1, x2, p=2)` with more than 25
//! rows, PyTorch switches to the matrix-multiply Euclidean expansion
//! `sqrt(x^2 + y^2 - 2xy)` — chosen for BLAS SPEED, not accuracy. That
//! expansion suffers CATASTROPHIC CANCELLATION at large magnitudes: for
//! `x[i,j] = 10000 + i*0.1` (shape `[26,4]`, r1=26>25) the true self-distance
//! `cdist(x, x)` diagonal is 0, but torch's default returns up to ~8.0 (live
//! torch 2.11.0+cu130).
//!
//! ferrotorch's `cdist` (ops/tensor_ops.rs) uses the direct `sqrt(sum (a-b)^2)`
//! reduction (the same algorithm as torch's `compute_mode='donot_use_mm_for_
//! euclid_dist'`, mode 2), which returns the mathematically-correct 0.0. This
//! is STRICTLY MORE ACCURATE than torch's default.
//!
//! ## Decision (user ruling, 2026-05-29)
//!
//! We do NOT replicate torch's less-precise mm-expansion to "match" the wart —
//! that would make ferrotorch worse to imitate a torch defect (the inverse of a
//! real divergence). Per the precision contract, ferrotorch being more precise
//! than torch is acceptable. #1647 is resolved as "ferrotorch more precise,
//! accepted" (NOT a fix-to-match). This test PINS the correct, more-precise
//! behavior so a future change can't silently regress it to torch's cancellation
//! result. Expected value (0.0) is the mathematically-exact self-distance, which
//! torch's own mode-2 (`donot_use_mm_for_euclid_dist`) also produces.
//!
//! Tracking: #1647 (resolved — ferrotorch more precise).

#![cfg(feature = "cuda")]

use ferrotorch_core::{Device, Tensor, TensorStorage, cdist};
use ferrotorch_gpu::init_cuda_backend;

fn ensure_cuda() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        init_cuda_backend().expect("CUDA backend init");
    });
}

/// Build the deterministic fixture `x[i,j] = 10000.0 + i*0.1`, shape `[rows, 4]`.
fn fixture(rows: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(rows * 4);
    for i in 0..rows {
        for _ in 0..4 {
            v.push(10000.0f32 + (i as f32) * 0.1);
        }
    }
    v
}

/// ferrotorch GPU `cdist(x, x, 2.0)` on a `[26,4]` input (r1 = 26 > 25) returns
/// the mathematically-correct self-distance 0.0 on the diagonal — MORE PRECISE
/// than torch's DEFAULT compute_mode (mm-expansion → up to ~8.0 via catastrophic
/// cancellation at magnitude 1e4). This is accepted per the precision contract;
/// torch's own mode-2 (`donot_use_mm_for_euclid_dist`) also gives 0.0.
///
/// Upstream: aten/src/ATen/native/Distance.cpp:105,138 (`_euclidean_dist`, the
/// less-accurate default for r1/r2>25). ferrotorch: direct sqrt-sum reduction.
#[test]
fn cdist_p2_large_rows_gpu_is_more_precise_than_torch_default() {
    ensure_cuda();
    let rows = 26;
    let data = fixture(rows);
    let cpu = Tensor::from_storage(TensorStorage::cpu(data.clone()), vec![rows, 4], false)
        .expect("cpu tensor");
    let gpu = cpu.to(Device::Cuda(0)).expect("to cuda");

    let out = cdist(&gpu, &gpu, 2.0).expect("gpu cdist p=2");
    let host = out.cpu().expect("cpu()").data().unwrap().to_vec();

    // Every diagonal self-distance must be the exact 0.0 (more precise than
    // torch's default mm-expansion, which reaches ~8.0 on this fixture).
    for i in 0..rows {
        let d = host[i * rows + i];
        assert!(
            d.abs() < 1e-3,
            "cdist p=2 self-distance diag[{i},{i}]={d} must be ~0 (direct, precise); \
             torch DEFAULT (mm-expansion, r1>25) loses this to ~8.0 cancellation — \
             ferrotorch is intentionally more precise (#1647, accepted)."
        );
    }
}

/// Companion CPU path: the public `cdist` CPU reduction is the same direct,
/// precise algorithm — also more accurate than torch's default for r1>25.
///
/// Upstream: aten/src/ATen/native/Distance.cpp:105,138.
#[test]
fn cdist_p2_large_rows_cpu_is_more_precise_than_torch_default() {
    let rows = 26;
    let data = fixture(rows);
    let cpu =
        Tensor::from_storage(TensorStorage::cpu(data), vec![rows, 4], false).expect("cpu tensor");

    let out = cdist(&cpu, &cpu, 2.0).expect("cpu cdist p=2");
    let host = out.data().unwrap().to_vec();
    for i in 0..rows {
        let d = host[i * rows + i];
        assert!(
            d.abs() < 1e-3,
            "cdist p=2 CPU self-distance diag[{i},{i}]={d} must be ~0 (direct, precise); \
             torch DEFAULT mm-expansion loses this — ferrotorch more precise (#1647, accepted)."
        );
    }
}
