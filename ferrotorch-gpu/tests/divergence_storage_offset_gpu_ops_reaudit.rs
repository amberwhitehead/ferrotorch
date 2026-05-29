//! HIGH-BLAST-RADIUS RE-AUDIT of the #1657 storage-offset fix
//! (commit c72fd36e2, `ferrotorch-core/src/methods.rs` `contiguous_t`).
//!
//! The #1657 fix is localised to `contiguous_t`: its `clone()` fast-path now
//! also requires `storage_offset() == 0`, so any GPU op that calls
//! `.contiguous()` BEFORE reading `gpu_handle()` is now correct on a
//! row-narrowed CUDA view (row-major strides, non-zero `storage_offset` — the
//! exact case `is_contiguous()` cannot detect, see `tensor.rs::is_contiguous`
//! / `c10::TensorImpl::is_contiguous` which inspect strides ONLY).
//!
//! THE KEY GAP this re-audit pins: GPU ops that read `gpu_handle()` DIRECTLY,
//! WITHOUT a preceding `.contiguous()`, are NOT helped by the fix. They still
//! hand the kernel the BASE buffer pointer (element 0) and drop the offset.
//!
//! Surveyed call sites that pass RAW `input.gpu_handle()` (no `.contiguous()`):
//!   - special.rs `special_gpu_simple`/`poly_gpu_simple`/`special_gpu_binary`
//!     (entr / i0 / ndtr / i0e / i1 / zeta / chebyshev / ...) — special.rs:2505,
//!     2544, 2606, 2634; kernel ABI `(in,out,total)` indexes from element 0
//!     (`ferrotorch-gpu/src/special.rs::launch_elementwise_f32:5253-5285`).
//!   - grad_fns/transcendental.rs `exp`/`log`/`clamp` — :161, :263, :558.
//!   - grad_fns/reduction.rs `sum`/`mean`/`prod`/`min`/`max`/`sum_axis` — :144+.
//!   - grad_fns/activation.rs `relu`/`sigmoid`/`tanh`/`silu` — :736+.
//!   - ops/tensor_ops.rs `triu`/`tril`/`diag`/`cdist` — :89, :172, :251+, :558.
//!   - ops/cumulative.rs `cumsum`/`cumprod`/`cummax`/`cummin`/`logcumsumexp` — :88+.
//!   - ops/search.rs `searchsorted`/`unique_consecutive`/`histc`/`meshgrid`/
//!     `topk` — :57, :227, :366, :505, :604.
//!   - masked.rs masked mul / masked_min / masked_max / isfinite / ne_scalar — :242+.
//!
//! NONE of the public wrappers for these call `.contiguous()` before the GPU
//! dispatch. The #1657 contiguous-layer fix DOES NOT reach this WHOLE class.
//! This re-audit pins a representative sample — `entr`, `i0`, `ndtr` (special),
//! `exp` (elementwise transcendental), `triu` (structured) — proving the gap is
//! systemic, not a single-op artifact. This is a NEW, separately-tracked
//! release-blocker, exposed by the same narrowed-offset probe that motivated
//! #1657 (which is correctly fixed at the `contiguous()` layer for the
//! scatter/gather family that DOES call `.contiguous()`).
//!
//! # R-CHAR-3 provenance (live torch 2.11.0+cu130)
//!
//! All expected values are from a live torch call on the identical narrowed
//! view. torch's CUDA and CPU elementwise/structured paths produce identical
//! results for these ops by design (the CUDA TensorIterator uses an
//! `OffsetCalculator` that honours `storage_offset`; `triu` likewise). The
//! reference below was captured from the torch CPU path (this env's nvrtc
//! JIT-fusion is broken — unrelated to op semantics), which is the
//! documented-correct value:
//!
//! ```python
//! import torch
//! full = torch.arange(1,9,dtype=torch.float32).reshape(4,2)
//! view = full[1:4]            # [[3,4],[5,6],[7,8]] storage_offset 2, is_contiguous() True
//! torch.special.entr(view).flatten()
//! #   [-3.2958369, -5.5451775, -8.0471897, -10.750557, -13.62137, -16.635532]
//! torch.special.i0(view).flatten()
//! #   [4.8807926, 11.301921, 27.239874, 67.234413, 168.59392, 427.56421]
//! torch.special.ndtr(view).flatten()
//! #   [0.99865007, 0.99996829, 0.9999997, 1.0, 1.0, 1.0]
//! torch.exp(view).flatten()
//! #   [20.085537, 54.598148, 148.41316, 403.4288, 1096.6332, 2980.958]
//!
//! fullsq = torch.arange(1,13,dtype=torch.float32).reshape(4,3)
//! vsq = fullsq[1:4]           # [[4,5,6],[7,8,9],[10,11,12]] storage_offset 3
//! torch.triu(vsq).flatten()
//! #   [4,5,6, 0,8,9, 0,0,12]
//!
//! # The offset-dropped GPU reads base rows from element 0 instead:
//! torch.special.entr(full[0:3]).flatten()
//! #   [-0.0, -1.3862944, -3.2958369, -5.5451775, -8.0471897, -10.750557]  (WRONG)
//! torch.triu(fullsq[0:3]).flatten()
//! #   [1,2,3, 0,5,6, 0,0,9]  (WRONG)
//! ```
//!
//! These are symbolic constants from the torch call, NOT self-derived from
//! ferrotorch. Each is cross-checked against the ferrotorch CPU path on a FRESH
//! contiguous CPU tensor carrying the logical view values (which torch and
//! ferrotorch-CPU agree on), never against the ferrotorch GPU output.
//!
//! Tracking: #1658 (NEW blocker, separate from #1657).

#![cfg(feature = "cuda")]

use ferrotorch_core::{Device, Tensor, TensorStorage, entr, exp, i0, ndtr, triu};
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
// -> logical [3,2] = [[3,4],[5,6],[7,8]] with storage_offset 2, is_contiguous()==True.
fn narrowed_cuda_view() -> Tensor<f32> {
    let full = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2])
        .to(Device::Cuda(0))
        .expect("to cuda");
    let view = full.narrow(0, 1, 3).expect("narrow rows 1..4");
    assert_eq!(view.shape(), &[3, 2]);
    assert!(
        view.is_contiguous(),
        "row-narrowed view keeps row-major strides -> is_contiguous() must be true \
         (this is the #1657 trap: contiguity is stride-only)"
    );
    assert_ne!(
        view.storage_offset(),
        0,
        "row-narrowed view must carry a non-zero storage_offset to exercise the gap"
    );
    view
}

// torch.special.entr(full[1:4]) — live torch CPU reference.
const TORCH_ENTR: [f32; 6] = [
    -3.295_837,
    -5.545_177_5,
    -8.047_19,
    -10.750_557,
    -13.621_37,
    -16.635_532,
];
// torch.special.i0(full[1:4]) — live torch CPU reference.
const TORCH_I0: [f32; 6] = [
    4.880_792_6,
    11.301_921,
    27.239_874,
    67.234_41,
    168.593_92,
    427.564_2,
];
// torch.special.ndtr(full[1:4]) — live torch CPU reference.
const TORCH_NDTR: [f32; 6] = [0.998_650_07, 0.999_968_3, 0.999_999_7, 1.0, 1.0, 1.0];
// torch.exp(full[1:4]) — live torch CPU reference.
const TORCH_EXP: [f32; 6] = [
    20.085_537, 54.598_15, 148.413_16, 403.428_8, 1096.633_2, 2980.958,
];
// torch.triu(fullsq[1:4]) — live torch CPU reference (3x3, offset 3).
const TORCH_TRIU: [f32; 9] = [4.0, 5.0, 6.0, 0.0, 8.0, 9.0, 0.0, 0.0, 12.0];

// ===========================================================================
// *** DIVERGENCE *** — torch.special.entr on a narrowed-offset CUDA view.
//
// `special_gpu_simple` reads `input.gpu_handle()?` with NO `.contiguous()`, and
// the `(in,out,total)` kernel indexes from element 0, dropping the +2 offset.
// torch / ferrotorch-CPU agree on the logical-view result; ferrotorch-GPU
// instead returns entr of base rows 0..2.  Left UN-#[ignore]d — release-blocker.
// ===========================================================================
#[test]
fn entr_narrowed_offset_view_gpu_matches_torch() {
    ensure_cuda();
    let view = narrowed_cuda_view();

    // ferrotorch-CPU agrees with torch (offset honoured via data_vec).
    let cpu_view = cpu_f32(&[3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[3, 2]);
    let cpu_out = entr(&cpu_view).expect("cpu entr");
    assert_close(
        &cpu_out.data().unwrap().to_vec(),
        &TORCH_ENTR,
        1e-4,
        "ferrotorch CPU entr must match torch on the logical view",
    );

    let gpu_out = entr(&view).expect("gpu entr");
    assert!(gpu_out.is_cuda(), "result must stay GPU-resident");
    assert_eq!(gpu_out.shape(), &[3, 2]);
    assert_close(
        &host_f32(&gpu_out),
        &TORCH_ENTR,
        1e-4,
        "entr on a narrowed-offset CUDA view must honour storage_offset; \
         special_gpu_simple drops it (gap NOT covered by the #1657 contiguous fix)",
    );
}

// ===========================================================================
// *** DIVERGENCE *** — torch.special.i0 on a narrowed-offset CUDA view.
// ===========================================================================
#[test]
fn i0_narrowed_offset_view_gpu_matches_torch() {
    ensure_cuda();
    let view = narrowed_cuda_view();

    let cpu_view = cpu_f32(&[3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[3, 2]);
    let cpu_out = i0(&cpu_view).expect("cpu i0");
    assert_close(
        &cpu_out.data().unwrap().to_vec(),
        &TORCH_I0,
        1e-3,
        "ferrotorch CPU i0 must match torch on the logical view",
    );

    let gpu_out = i0(&view).expect("gpu i0");
    assert!(gpu_out.is_cuda());
    assert_close(
        &host_f32(&gpu_out),
        &TORCH_I0,
        1e-3,
        "i0 on a narrowed-offset CUDA view must honour storage_offset",
    );
}

// ===========================================================================
// *** DIVERGENCE *** — torch.special.ndtr on a narrowed-offset CUDA view.
// ===========================================================================
#[test]
fn ndtr_narrowed_offset_view_gpu_matches_torch() {
    ensure_cuda();
    let view = narrowed_cuda_view();

    let cpu_view = cpu_f32(&[3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[3, 2]);
    let cpu_out = ndtr(&cpu_view).expect("cpu ndtr");
    assert_close(
        &cpu_out.data().unwrap().to_vec(),
        &TORCH_NDTR,
        1e-5,
        "ferrotorch CPU ndtr must match torch on the logical view",
    );

    let gpu_out = ndtr(&view).expect("gpu ndtr");
    assert!(gpu_out.is_cuda());
    assert_close(
        &host_f32(&gpu_out),
        &TORCH_NDTR,
        1e-5,
        "ndtr on a narrowed-offset CUDA view must honour storage_offset",
    );
}

// ===========================================================================
// *** DIVERGENCE *** — torch.exp on a narrowed-offset CUDA view (transcendental
// elementwise; reduction.rs/transcendental.rs share the raw-gpu_handle pattern).
// ===========================================================================
#[test]
fn exp_narrowed_offset_view_gpu_matches_torch() {
    ensure_cuda();
    let view = narrowed_cuda_view();

    let cpu_view = cpu_f32(&[3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[3, 2]);
    let cpu_out = exp(&cpu_view).expect("cpu exp");
    assert_close(
        &cpu_out.data().unwrap().to_vec(),
        &TORCH_EXP,
        1e-2,
        "ferrotorch CPU exp must match torch on the logical view",
    );

    let gpu_out = exp(&view).expect("gpu exp");
    assert!(gpu_out.is_cuda());
    assert_close(
        &host_f32(&gpu_out),
        &TORCH_EXP,
        1e-2,
        "exp on a narrowed-offset CUDA view must honour storage_offset \
         (transcendental.rs::exp_inner passes raw gpu_handle)",
    );
}

// ===========================================================================
// *** DIVERGENCE *** — torch.triu on a narrowed-offset SQUARE CUDA view.
// tensor_ops.rs::triu passes raw `input.gpu_handle()`; the kernel reads the
// [batch, rows, cols] block from element 0, masking the WRONG matrix.
// ===========================================================================
#[test]
fn triu_narrowed_offset_view_gpu_matches_torch() {
    ensure_cuda();
    // fullsq [4,3] = 1..12; narrow rows 1..4 -> logical 3x3 [[4,5,6],[7,8,9],[10,11,12]]
    let fullsq = cpu_f32(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[4, 3],
    )
    .to(Device::Cuda(0))
    .expect("to cuda");
    let vsq = fullsq.narrow(0, 1, 3).expect("narrow rows 1..4");
    assert_eq!(vsq.shape(), &[3, 3]);
    assert!(vsq.is_contiguous());
    assert_ne!(vsq.storage_offset(), 0);

    let cpu_view = cpu_f32(&[4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 3]);
    let cpu_out = triu(&cpu_view, 0).expect("cpu triu");
    assert_close(
        &cpu_out.data().unwrap().to_vec(),
        &TORCH_TRIU,
        1e-6,
        "ferrotorch CPU triu must match torch on the logical view",
    );

    let gpu_out = triu(&vsq, 0).expect("gpu triu");
    assert!(gpu_out.is_cuda());
    assert_close(
        &host_f32(&gpu_out),
        &TORCH_TRIU,
        1e-6,
        "triu on a narrowed-offset CUDA view must honour storage_offset \
         (tensor_ops.rs::triu passes raw gpu_handle)",
    );
}

// ===========================================================================
// REGRESSION GUARD (must PASS): offset-0 contiguous CUDA tensor still correct.
// Confirms the special-fn GPU path is right for the common case and the bug is
// specifically the dropped offset.
// ===========================================================================
#[test]
fn entr_offset0_contiguous_cuda_still_correct() {
    ensure_cuda();
    // Plain offset-0 contiguous CUDA [3,2] with the SAME logical values.
    let t = cpu_f32(&[3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[3, 2])
        .to(Device::Cuda(0))
        .expect("to cuda");
    assert_eq!(t.storage_offset(), 0);
    let out = entr(&t).expect("gpu entr offset0");
    assert!(out.is_cuda());
    assert_close(
        &host_f32(&out),
        &TORCH_ENTR,
        1e-4,
        "offset-0 contiguous CUDA entr must match torch (common-case guard)",
    );
}
