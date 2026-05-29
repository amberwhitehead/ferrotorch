//! Divergence pin: `ferrotorch_core::cdist` (p=2) does NOT implement
//! `torch.cdist`'s DEFAULT `compute_mode` (mode 0), which switches to the
//! matrix-multiply Euclidean expansion (`x^2 + y^2 - 2xy`) when
//! `r1 > 25 || r2 > 25`.
//!
//! Upstream `aten/src/ATen/native/Distance.cpp:105`:
//!   `if (!(p == 2 && (mode == 1 || (mode == 0 && (r1 > 25 || r2 > 25)))))`
//! and `Distance.cpp:138`:
//!   `Tensor dist = ... at::_euclidean_dist(x1, x2) ...`
//! i.e. for the USER-FACING default `torch.cdist(x1, x2, p=2)` with more than
//! 25 rows, PyTorch computes the distance via the mm-expansion, which is
//! numerically different from the direct `sqrt(sum (a-b)^2)`.
//!
//! ferrotorch's `cdist` (ops/tensor_ops.rs:472) has no `compute_mode` arg and
//! ALWAYS uses the direct reduction (both CPU loop at tensor_ops.rs:583 and the
//! GPU kernel `gpu_cdist_f32` in distance.rs), which is equivalent to torch's
//! `compute_mode='donot_use_mm_for_euclid_dist'` (mode 2) only — NOT the
//! default mode 0. The docstring "Matches PyTorch's `torch.cdist`" overclaims.
//!
//! Observable, deterministic divergence (LIVE torch 2.11.0+cu130):
//! fixture x[i,j] = 10000.0 + i*0.1, shape [26,4] (r1 = 26 > 25). The
//! self-distance diagonal of `torch.cdist(x, x, p=2)` (default mode) reaches
//! 8.0 (worst cell, row 3) due to catastrophic cancellation in the mm
//! expansion at magnitude 1e4; ferrotorch (direct) returns exactly 0.0. The
//! gap (8.0) is far beyond any fp tolerance.
//!
//! Expected values are from live torch (the parity oracle), NOT copied from
//! ferrotorch (R-CHAR-3): direct algorithm gives 0.0, torch DEFAULT gives 8.0.
//!
//! Tracking: #1647

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

/// Divergence: `cdist(x, x, 2.0)` for a `[26,4]` input (r1 = 26 > 25) must match
/// `torch.cdist(x, x, p=2)` with its DEFAULT compute_mode, whose worst
/// self-distance diagonal cell is 8.0 (live torch). ferrotorch returns 0.0
/// (direct algorithm == torch mode-2 only), diverging by 8.0.
///
/// Upstream: aten/src/ATen/native/Distance.cpp:105,138 (`_euclidean_dist`).
/// ferrotorch: ferrotorch-core/src/ops/tensor_ops.rs:472 (no compute_mode).
#[test]
#[ignore = "divergence: cdist p=2 ignores torch default compute_mode mm-expansion for r1/r2>25; tracking #1647"]
fn divergence_cdist_p2_default_compute_mode_large_rows_gpu() {
    ensure_cuda();
    let rows = 26;
    let data = fixture(rows);
    let cpu = Tensor::from_storage(TensorStorage::cpu(data.clone()), vec![rows, 4], false)
        .expect("cpu tensor");
    let gpu = cpu.to(Device::Cuda(0)).expect("to cuda");

    let out = cdist(&gpu, &gpu, 2.0).expect("gpu cdist p=2");
    let host = out.cpu().expect("cpu()").data().unwrap().to_vec();

    // Worst self-distance diagonal cell per live torch DEFAULT mode is row 3.
    let worst = host[3 * rows + 3];
    // torch.cdist default (mm-expansion, r1>25) -> 8.0 on this fixture.
    const TORCH_DEFAULT_WORST_DIAG: f32 = 8.0;
    assert!(
        (worst - TORCH_DEFAULT_WORST_DIAG).abs() < 1e-2,
        "cdist p=2 default compute_mode divergence: ferrotorch diag[3,3]={worst} \
         (direct sqrt-sum), torch.cdist(x,x,p=2) default mode={TORCH_DEFAULT_WORST_DIAG} \
         (mm-expansion, r1=26>25)"
    );
}

/// Companion (CPU path, same divergence): the public `cdist` CPU loop is the
/// same direct algorithm, so it too cannot reproduce torch's default mode for
/// r1 > 25. Pins that the divergence is not GPU-specific.
///
/// Upstream: aten/src/ATen/native/Distance.cpp:105,138.
/// ferrotorch: ferrotorch-core/src/ops/tensor_ops.rs:583 (direct CPU reduction).
#[test]
#[ignore = "divergence: cdist p=2 ignores torch default compute_mode mm-expansion for r1/r2>25; tracking #1647"]
fn divergence_cdist_p2_default_compute_mode_large_rows_cpu() {
    let rows = 26;
    let data = fixture(rows);
    let cpu = Tensor::from_storage(TensorStorage::cpu(data), vec![rows, 4], false)
        .expect("cpu tensor");

    let out = cdist(&cpu, &cpu, 2.0).expect("cpu cdist p=2");
    let host = out.data().unwrap().to_vec();
    let worst = host[3 * rows + 3];
    const TORCH_DEFAULT_WORST_DIAG: f32 = 8.0;
    assert!(
        (worst - TORCH_DEFAULT_WORST_DIAG).abs() < 1e-2,
        "cdist p=2 default compute_mode divergence (CPU): ferrotorch diag[3,3]={worst}, \
         torch default={TORCH_DEFAULT_WORST_DIAG}"
    );
}
