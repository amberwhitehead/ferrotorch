//! COMPLETENESS re-audit of the #1658 storage-offset GPU-dispatch class
//! (commit b91c8192c). The #1658 fix inserted `.contiguous()` normalisation at
//! an *enumerated* set of GPU dispatch sites (special/transcendental/activation/
//! reduction/cumulative/search/tensor_ops/masked). This file pins THREE class
//! members the enumeration MISSED — every one reads a raw `gpu_handle()` on a
//! float input that can carry a non-zero `storage_offset` (a row-narrowed CUDA
//! view, `is_contiguous()==true`, base buffer longer than `numel`) WITHOUT
//! first normalising layout:
//!
//!   1. `BoolTensor::compare_float`  (ferrotorch-core/src/bool_tensor.rs:521)
//!      backs gt/lt/ge/le/eq_t/ne — reads `a.gpu_handle()` / `b.gpu_handle()`.
//!   2. `where_cond_bt`              (ferrotorch-core/src/ops/indexing.rs:862)
//!      reads `x.gpu_handle()` / `y.gpu_handle()`.
//!   3. `masked_select`             (ferrotorch-core/src/ops/indexing.rs:926)
//!      reads `input.gpu_handle()`.
//!
//! These belong to the SAME class as #1658 (element-0-indexed GPU kernels that
//! drop `storage_offset`). The observed failure mode is an *error* — the kernels
//! validate `buffer.len() == numel`, and a narrowed view's BASE buffer length
//! (8) exceeds `numel` (6), so the call returns `Err(InvalidArgument ...)`
//! instead of computing the logical-view result. PyTorch returns a valid tensor.
//! Either way (silent-wrong, as the other class members were, or error, as here)
//! the divergence from upstream is real: torch CUDA's TensorIterator
//! OffsetCalculator honours `storage_offset`; ferrotorch does not at these sites.
//!
//! Upstream contract: `c10/core/TensorImpl.h` — `is_contiguous()` /
//! `compute_contiguous()` inspect sizes+strides ONLY (a row-narrowed view reports
//! contiguous despite `storage_offset != 0`), while the CUDA kernels honour the
//! offset.
//!
//! # R-CHAR-3 provenance (live torch 2.11.0+cu130, this env, RTX 3090)
//!
//! All expected values are live-torch constants captured on the IDENTICAL
//! narrowed CUDA view and pasted below, never copied from a ferrotorch GPU run:
//!
//! ```python
//! import torch
//! full = torch.arange(1,9,dtype=torch.float32).reshape(4,2).cuda()
//! view = full[1:4]                       # [[3,4],[5,6],[7,8]] storage_offset 2
//! b    = torch.full_like(view, 4.5)
//! (view > b).flatten().cpu().tolist()    # GT_GT45 -> [F,F,T,T,T,T]
//! cond = view > 4.5
//! z    = torch.zeros_like(view)
//! torch.where(cond, view, z).flatten().cpu().tolist()   # WHERE -> [0,0,5,6,7,8]
//! torch.masked_select(view, cond).cpu().tolist()        # MASKED -> [5,6,7,8]
//! ```
//!
//! VERDICT: GENERATOR MUST FIX. Tracking: crosslink #1660 (member of #1658
//! class). These tests are left UN-`#[ignore]`d: storage_offset correctness for
//! shipped public ops (`>`, `torch.where`, `torch.masked_select`) on CUDA views
//! is a release-blocker, and the failing tests ARE the block.

#![cfg(feature = "cuda")]

use ferrotorch_core::{BoolTensor, Device, Tensor, TensorStorage, masked_select, where_cond_bt};
use ferrotorch_gpu::init_cuda_backend;

fn ensure_cuda() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let _ = init_cuda_backend();
    });
}

fn cpu_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .expect("cpu f32 tensor")
}

fn host_f32(t: &Tensor<f32>) -> Vec<f32> {
    t.cpu().expect("cpu()").data().unwrap().to_vec()
}

// A CUDA-resident BoolTensor returns `GpuTensorNotAccessible` from `.data()` by
// device-error policy (a host count needs an explicit D2H copy). Move to CPU
// first, then read — this compares the GPU-computed result against live torch.
fn host_bool(b: &BoolTensor) -> Vec<bool> {
    b.to(Device::Cpu)
        .expect("bool to cpu")
        .data()
        .expect("bool data")
        .to_vec()
}

/// CUDA [4,2] = [[1,2],[3,4],[5,6],[7,8]], narrow rows 1..4 -> logical [3,2] =
/// [[3,4],[5,6],[7,8]] with storage_offset 2 and `is_contiguous() == true`.
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

// Fresh offset-0 GPU bool cond (cond = view > 4.5), so the ONLY offset-carrying
// operand under test in where/masked_select is the float data tensor.
fn cond_gt45_gpu() -> BoolTensor {
    BoolTensor::from_vec(vec![false, false, true, true, true, true], vec![3, 2])
        .expect("bool cond")
        .to(Device::Cuda(0))
        .expect("cond to cuda")
}

// ── live-torch symbolic constants (see module docs) ────────────────────────
const TORCH_GT_GT45: [bool; 6] = [false, false, true, true, true, true];
const TORCH_WHERE: [f32; 6] = [0.0, 0.0, 5.0, 6.0, 7.0, 8.0];
const TORCH_MASKED: [f32; 4] = [5.0, 6.0, 7.0, 8.0];

// ===========================================================================
// MISSED #1: BoolTensor::gt (compare_float) on a narrowed-offset CUDA view.
// ===========================================================================
#[test]
fn compare_gt_narrowed_offset_view_gpu_matches_torch() {
    ensure_cuda();
    let view = narrowed_cuda_view();
    let b = cpu_f32(&[4.5; 6], &[3, 2])
        .to(Device::Cuda(0))
        .expect("b cuda");
    let got = BoolTensor::gt(&view, &b)
        .expect("BoolTensor::gt on narrowed CUDA view must honour storage_offset (not error)");
    let got: Vec<bool> = host_bool(&got);
    assert_eq!(
        got,
        TORCH_GT_GT45.to_vec(),
        "gt on narrowed-offset CUDA view: ferrotorch dropped storage_offset"
    );
}

// ===========================================================================
// MISSED #2: where_cond_bt (torch.where) — narrowed-offset x operand.
// ===========================================================================
#[test]
fn where_cond_bt_narrowed_offset_view_gpu_matches_torch() {
    ensure_cuda();
    let view = narrowed_cuda_view();
    let cond = cond_gt45_gpu();
    let zeros = cpu_f32(&[0.0; 6], &[3, 2])
        .to(Device::Cuda(0))
        .expect("zeros cuda");
    let got = where_cond_bt(&cond, &view, &zeros)
        .expect("where_cond_bt on narrowed CUDA x must honour storage_offset (not error)");
    assert_eq!(
        host_f32(&got),
        TORCH_WHERE.to_vec(),
        "where_cond_bt on narrowed-offset CUDA x: ferrotorch dropped storage_offset"
    );
}

// ===========================================================================
// #1660 regression: BOTH compare operands are narrowed views, so BOTH are
// materialised by `.contiguous()` into POOLED buffers whose raw `CudaSlice`
// len is rounded up (>= 256) while the logical numel is 6. The kernel-level
// validation must compare LOGICAL lens (6 == 6) and launch exactly 6 threads,
// reading only the first 6 elements of each over-allocated backing slice. A
// raw-len equality check would PASS here (both 256) but a raw-len LAUNCH would
// read 256 elements of garbage; this pins logical-len LAUNCH dimensions too.
//
// Live torch: full[1:4] > full2[1:4] where full2 = arange(1,9)+1 reshaped,
// view2 = [[4,5],[6,7],[8,9]]; (view > view2) = [F,F,F,F,F,F] (each lhs < rhs).
const TORCH_GT_BOTH_VIEWS: [bool; 6] = [false, false, false, false, false, false];
#[test]
fn compare_gt_both_narrowed_views_pooled_logical_len_gpu_matches_torch() {
    ensure_cuda();
    let view = narrowed_cuda_view(); // [3,4],[5,6],[7,8]
    let full2 = cpu_f32(&[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[4, 2])
        .to(Device::Cuda(0))
        .expect("full2 cuda");
    let view2 = full2.narrow(0, 1, 3).expect("narrow full2"); // [4,5],[6,7],[8,9]
    assert_ne!(view2.storage_offset(), 0);
    let got = BoolTensor::gt(&view, &view2).expect(
        "gt with both narrowed (pooled, over-allocated) CUDA views must launch on logical n",
    );
    let got: Vec<bool> = host_bool(&got);
    assert_eq!(
        got,
        TORCH_GT_BOTH_VIEWS.to_vec(),
        "gt(view, view2): pooled over-allocated operands must be read only over logical numel"
    );
}

// #1660 guard: the common EXACT-length case (offset-0 contiguous CUDA inputs)
// must keep working after the logical-len change — both operands have raw len
// == logical len, so the `>= n` backing-store check is trivially satisfied.
const TORCH_GT_OFFSET0: [bool; 6] = [false, false, true, true, true, true];
#[test]
fn compare_gt_offset0_exact_len_gpu_unaffected() {
    ensure_cuda();
    let a = cpu_f32(&[3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[3, 2])
        .to(Device::Cuda(0))
        .expect("a cuda");
    let b = cpu_f32(&[4.5; 6], &[3, 2])
        .to(Device::Cuda(0))
        .expect("b cuda");
    let got: Vec<bool> =
        host_bool(&BoolTensor::gt(&a, &b).expect("gt on exact-len offset-0 CUDA inputs"));
    assert_eq!(got, TORCH_GT_OFFSET0.to_vec());
}

// ===========================================================================
// MISSED #3: masked_select (torch.masked_select) — narrowed-offset input.
// ===========================================================================
#[test]
fn masked_select_narrowed_offset_view_gpu_matches_torch() {
    ensure_cuda();
    let view = narrowed_cuda_view();
    let cond = cond_gt45_gpu();
    let got = masked_select(&view, &cond)
        .expect("masked_select on narrowed CUDA input must honour storage_offset (not error)");
    assert_eq!(
        host_f32(&got),
        TORCH_MASKED.to_vec(),
        "masked_select on narrowed-offset CUDA input: ferrotorch dropped storage_offset"
    );
}
