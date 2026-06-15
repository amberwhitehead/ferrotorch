//! EXTENDED re-audit of the #1658 storage-offset fix: one representative op
//! from EACH of the 8 manifest files exercised on a row-narrowed CUDA view
//! (non-zero `storage_offset`, row-major strides — the case `is_contiguous()`
//! reports `true`, see `c10::TensorImpl::is_contiguous` which inspects strides
//! ONLY). The narrowed view's GPU result must honour `storage_offset` after the
//! fix inserts `.contiguous()` (a packed offset-0 on-device materialisation via
//! #1657's `strided_copy`) before each GPU dispatch reads `gpu_handle()`.
//!
//! The 5 pinned probes in
//! `divergence_storage_offset_gpu_ops_reaudit.rs` cover special (entr/i0/ndtr),
//! transcendental (exp), and tensor_ops (triu). This file widens coverage to
//! the remaining families so the whole-class fix is regression-pinned:
//!
//!   - special.rs            -> `i0e`        (`special_gpu_simple` / `poly_gpu_simple`)
//!   - transcendental.rs     -> `log`        (`log_inner` elementwise)
//!   - activation.rs         -> `sigmoid`    (`sigmoid_inner` elementwise)
//!   - tensor_ops.rs         -> `diag`       (`diag_extract` gather)
//!   - grad_fns/reduction.rs -> `sum_dim`    (`sum_axis` strided reduce)
//!   - ops/cumulative.rs     -> `cumsum`     (`cumsum` strided scan)
//!   - ops/search.rs         -> `topk`       (`topk_nd` k-selection)
//!   - masked.rs             -> `masked_min` (the `masked_min_gpu` fused reduce)
//!
//! # R-CHAR-3 provenance (live torch 2.11.0+cu130, this env)
//!
//! All expected values were captured from a live torch call on the IDENTICAL
//! narrowed view and are pasted below as named symbolic constants. Each is
//! independently cross-checked against the ferrotorch CPU path on a fresh
//! contiguous CPU tensor carrying the logical-view values (which torch and
//! ferrotorch-CPU agree on), never against the ferrotorch GPU output.
//!
//! ```python
//! import torch
//! full = torch.arange(1,9,dtype=torch.float32).reshape(4,2)
//! view = full[1:4]                       # [[3,4],[5,6],[7,8]] storage_offset 2
//! torch.special.i0e(view).flatten()      # I0E below
//! torch.log(view).flatten()              # LOG below
//! torch.sigmoid(view).flatten()          # SIGMOID below
//! torch.diag(view).flatten()             # DIAG  -> [3, 6]
//! torch.sum(view, dim=0).flatten()       # SUM_DIM0 -> [15, 18]
//! torch.cumsum(view, dim=0).flatten()    # CUMSUM_DIM0 -> [3,4,8,10,15,18]
//! v,i = torch.topk(view, 2, dim=-1)      # TOPK_VALS  -> [4,3, 6,5, 8,7]
//! torch.min(view)                        # MASKED_MIN (all finite) -> 3
//! ```
//!
//! Tracking: #1658 (companion to #1657).

#![cfg(feature = "cuda")]

use ferrotorch_core::{
    Device, MaskedTensor, Tensor, TensorStorage, cumsum, diag, i0e, log, masked_min, sigmoid,
    sum_dim, topk,
};
use ferrotorch_gpu::init_cuda_backend;

fn ensure_cuda() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        init_cuda_backend().expect("CUDA backend init");
    });
}

fn cpu_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .expect("cpu f32 tensor")
}

fn host_f32(t: &Tensor<f32>) -> Vec<f32> {
    t.cpu().expect("cpu()").data().unwrap().to_vec()
}

fn assert_close(got: &[f32], want: &[f32], tol: f32, ctx: &str) {
    assert_eq!(got.len(), want.len(), "{ctx}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let d = (g - w).abs();
        let rel = d / (w.abs().max(1.0));
        assert!(
            d <= tol || rel <= tol,
            "{ctx}: element {i}: got {g}, want {w} (abs {d}, rel {rel})"
        );
    }
}

// The shared fixture: CUDA [4,2] = [[1,2],[3,4],[5,6],[7,8]], narrow rows 1..4
// -> logical [3,2] = [[3,4],[5,6],[7,8]] with storage_offset 2, is_contiguous().
fn narrowed_cuda_view() -> Tensor<f32> {
    let full = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2])
        .to(Device::Cuda(0))
        .expect("to cuda");
    let view = full.narrow(0, 1, 3).expect("narrow rows 1..4");
    assert_eq!(view.shape(), &[3, 2]);
    assert!(
        view.is_contiguous(),
        "row-narrowed view keeps row-major strides -> is_contiguous() must be true"
    );
    assert_ne!(
        view.storage_offset(),
        0,
        "row-narrowed view must carry a non-zero storage_offset to exercise the gap"
    );
    view
}

// The same logical values on a FRESH contiguous offset-0 CPU tensor — the torch
// reference cross-check (torch and ferrotorch-CPU agree on this).
fn logical_cpu() -> Tensor<f32> {
    cpu_f32(&[3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[3, 2])
}

// ---- live-torch symbolic constants (see module docs) ----------------------
const TORCH_I0E: [f32; 6] = [
    0.243_000_36,
    0.207_001_91,
    0.183_540_82,
    0.166_657_45,
    0.153_737_75,
    0.143_431_81,
];
const TORCH_LOG: [f32; 6] = [
    1.098_612_3,
    1.386_294_4,
    1.609_438,
    1.791_759_5,
    1.945_910_2,
    2.079_441_5,
];
const TORCH_SIGMOID: [f32; 6] = [
    0.952_574_13,
    0.982_013_76,
    0.993_307_2,
    0.997_527_4,
    0.999_089,
    0.999_664_66,
];
const TORCH_DIAG: [f32; 2] = [3.0, 6.0];
const TORCH_SUM_DIM0: [f32; 2] = [15.0, 18.0];
const TORCH_CUMSUM_DIM0: [f32; 6] = [3.0, 4.0, 8.0, 10.0, 15.0, 18.0];
const TORCH_TOPK_VALS: [f32; 6] = [4.0, 3.0, 6.0, 5.0, 8.0, 7.0];
const TORCH_MIN: [f32; 1] = [3.0];

// ===========================================================================
// special.rs — `i0e` on a narrowed-offset CUDA view (`special_gpu_simple`).
// ===========================================================================
#[test]
fn i0e_narrowed_offset_view_gpu_matches_torch() {
    ensure_cuda();
    let cpu_out = i0e(&logical_cpu()).expect("cpu i0e");
    assert_close(
        cpu_out.data().unwrap(),
        &TORCH_I0E,
        1e-4,
        "ferrotorch CPU i0e must match torch on the logical view",
    );
    let gpu_out = i0e(&narrowed_cuda_view()).expect("gpu i0e");
    assert!(gpu_out.is_cuda());
    assert_close(
        &host_f32(&gpu_out),
        &TORCH_I0E,
        1e-4,
        "i0e narrowed-offset GPU",
    );
}

// ===========================================================================
// transcendental.rs — `log` on a narrowed-offset CUDA view (`log_inner`).
// ===========================================================================
#[test]
fn log_narrowed_offset_view_gpu_matches_torch() {
    ensure_cuda();
    let cpu_out = log(&logical_cpu()).expect("cpu log");
    assert_close(
        cpu_out.data().unwrap(),
        &TORCH_LOG,
        1e-4,
        "ferrotorch CPU log must match torch on the logical view",
    );
    let gpu_out = log(&narrowed_cuda_view()).expect("gpu log");
    assert!(gpu_out.is_cuda());
    assert_close(
        &host_f32(&gpu_out),
        &TORCH_LOG,
        1e-4,
        "log narrowed-offset GPU",
    );
}

// ===========================================================================
// activation.rs — `sigmoid` on a narrowed-offset CUDA view (`sigmoid_inner`).
// ===========================================================================
#[test]
fn sigmoid_narrowed_offset_view_gpu_matches_torch() {
    ensure_cuda();
    let cpu_out = sigmoid(&logical_cpu()).expect("cpu sigmoid");
    assert_close(
        cpu_out.data().unwrap(),
        &TORCH_SIGMOID,
        1e-5,
        "ferrotorch CPU sigmoid must match torch on the logical view",
    );
    let gpu_out = sigmoid(&narrowed_cuda_view()).expect("gpu sigmoid");
    assert!(gpu_out.is_cuda());
    assert_close(
        &host_f32(&gpu_out),
        &TORCH_SIGMOID,
        1e-5,
        "sigmoid narrowed-offset GPU",
    );
}

// ===========================================================================
// tensor_ops.rs — `diag` (2-D extract) on a narrowed-offset CUDA view.
// torch.diag([[3,4],[5,6],[7,8]]) extracts the main diagonal -> [3, 6].
// ===========================================================================
#[test]
fn diag_narrowed_offset_view_gpu_matches_torch() {
    ensure_cuda();
    let cpu_out = diag(&logical_cpu(), 0).expect("cpu diag");
    assert_close(
        cpu_out.data().unwrap(),
        &TORCH_DIAG,
        1e-6,
        "ferrotorch CPU diag must match torch on the logical view",
    );
    let gpu_out = diag(&narrowed_cuda_view(), 0).expect("gpu diag");
    assert!(gpu_out.is_cuda());
    assert_eq!(gpu_out.shape(), &[2]);
    assert_close(
        &host_f32(&gpu_out),
        &TORCH_DIAG,
        1e-6,
        "diag narrowed-offset GPU",
    );
}

// ===========================================================================
// grad_fns/reduction.rs — `sum_dim(0)` on a narrowed-offset CUDA view
// (`sum_axis` strided reduction). torch.sum(view, dim=0) -> [15, 18].
// ===========================================================================
#[test]
fn sum_dim_narrowed_offset_view_gpu_matches_torch() {
    ensure_cuda();
    let cpu_out = sum_dim(&logical_cpu(), 0, false).expect("cpu sum_dim");
    assert_close(
        cpu_out.data().unwrap(),
        &TORCH_SUM_DIM0,
        1e-5,
        "ferrotorch CPU sum_dim must match torch on the logical view",
    );
    let gpu_out = sum_dim(&narrowed_cuda_view(), 0, false).expect("gpu sum_dim");
    assert!(gpu_out.is_cuda());
    assert_eq!(gpu_out.shape(), &[2]);
    assert_close(
        &host_f32(&gpu_out),
        &TORCH_SUM_DIM0,
        1e-5,
        "sum_dim narrowed-offset GPU",
    );
}

// ===========================================================================
// ops/cumulative.rs — `cumsum(0)` on a narrowed-offset CUDA view
// (strided scan). torch.cumsum(view, dim=0) -> [3,4, 8,10, 15,18].
// ===========================================================================
#[test]
fn cumsum_narrowed_offset_view_gpu_matches_torch() {
    ensure_cuda();
    let cpu_out = cumsum(&logical_cpu(), 0).expect("cpu cumsum");
    assert_close(
        cpu_out.data().unwrap(),
        &TORCH_CUMSUM_DIM0,
        1e-5,
        "ferrotorch CPU cumsum must match torch on the logical view",
    );
    let gpu_out = cumsum(&narrowed_cuda_view(), 0).expect("gpu cumsum");
    assert!(gpu_out.is_cuda());
    assert_eq!(gpu_out.shape(), &[3, 2]);
    assert_close(
        &host_f32(&gpu_out),
        &TORCH_CUMSUM_DIM0,
        1e-5,
        "cumsum narrowed-offset GPU",
    );
}

// ===========================================================================
// ops/search.rs — `topk(2)` over the last dim on a narrowed-offset CUDA view
// (`topk_nd` k-selection). Rows [3,4],[5,6],[7,8] -> largest-2 [4,3],[6,5],[8,7].
// ===========================================================================
#[test]
fn topk_narrowed_offset_view_gpu_matches_torch() {
    ensure_cuda();
    let (cpu_vals, _cpu_idx) = topk(&logical_cpu(), 2, true).expect("cpu topk");
    assert_close(
        cpu_vals.data().unwrap(),
        &TORCH_TOPK_VALS,
        1e-6,
        "ferrotorch CPU topk values must match torch on the logical view",
    );
    let (gpu_vals, _gpu_idx) = topk(&narrowed_cuda_view(), 2, true).expect("gpu topk");
    assert!(gpu_vals.is_cuda());
    assert_close(
        &host_f32(&gpu_vals),
        &TORCH_TOPK_VALS,
        1e-6,
        "topk narrowed-offset GPU",
    );
}

// ===========================================================================
// masked.rs — `masked_min` (the fused `masked_min_gpu` reduce branch) on a
// narrowed-offset CUDA view. The host-built mask is all-true, so the result is
// the minimum over the six logical values = torch.min(view) = 3. The data
// tensor is the narrowed CUDA view; `masked_min_gpu` reads `mt.data.gpu_handle`
// and must normalise the offset first (the fix shadows `mt.data.contiguous()`).
// ===========================================================================
#[test]
fn masked_min_narrowed_offset_view_gpu_matches_torch() {
    ensure_cuda();
    let mask = vec![true; 6];

    // CPU cross-check on the logical view.
    let cpu_mt = MaskedTensor::new(logical_cpu(), mask.clone()).expect("cpu MaskedTensor");
    let cpu_min = masked_min(&cpu_mt).expect("cpu masked_min");
    assert_close(
        cpu_min.data().unwrap(),
        &TORCH_MIN,
        1e-6,
        "ferrotorch CPU masked_min must match torch.min on the logical view",
    );

    let gpu_mt = MaskedTensor::new(narrowed_cuda_view(), mask).expect("gpu MaskedTensor");
    let gpu_min = masked_min(&gpu_mt).expect("gpu masked_min");
    assert!(gpu_min.is_cuda());
    assert_close(
        &host_f32(&gpu_min),
        &TORCH_MIN,
        1e-6,
        "masked_min on a narrowed-offset CUDA view must honour storage_offset \
         (masked_min_gpu reads mt.data.gpu_handle)",
    );
}
