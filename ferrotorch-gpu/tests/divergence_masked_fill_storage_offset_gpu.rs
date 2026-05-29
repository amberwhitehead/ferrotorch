//! SPILLOVER DIVERGENCE (distinct from #1660; tracking #1661): `masked_fill`
//! drops `storage_offset` on a row-narrowed CUDA input.
//!
//! #1660 fixed `BoolTensor::compare_float`, `where_cond_bt`, and `masked_select`
//! by inserting `.contiguous()` normalisation at their GPU dispatch sites and
//! threading the logical numel into the compare/where kernels. The SAME
//! storage-offset class member exists, UNFIXED, in `masked_fill`:
//!
//!   - `ferrotorch_core::grad_fns::indexing::masked_fill_bt` (indexing.rs:1047)
//!     reads `input.gpu_handle()?` directly (NO `.contiguous()`).
//!   - `ferrotorch_core::grad_fns::indexing::masked_fill` (indexing.rs:418)
//!     reads `input.gpu_handle()?` directly (NO `.contiguous()`).
//!   - `Tensor::masked_fill` (tensor.rs:1151) -> `masked_fill_bt` is the public
//!     surface (mirrors `torch.Tensor.masked_fill`).
//!
//! The GPU dispatch `CudaBackendImpl::masked_fill_dt` (backend_impl.rs:8044)
//! validates `input.len() != mask.len()`, and the kernel launcher
//! `launch_masked_fill` (masked_kernels.rs:483) repeats `input.len() != mask.len()`.
//! A row-narrowed view's BASE buffer length (8) exceeds its logical numel (6),
//! so the handle's reported len mismatches the mask's len (6) -> the call
//! returns `Err(InvalidArgument "masked_fill: input numel 8 != mask numel 6")`
//! INSTEAD of the masked-filled logical-view result. PyTorch returns a valid
//! tensor (its CUDA TensorIterator OffsetCalculator honours `storage_offset`).
//!
//! # R-CHAR-3 provenance (live torch 2.11.0+cu130, this env, RTX 3090)
//!
//! ```python
//! import torch
//! full = torch.arange(1,9,dtype=torch.float32).cuda().reshape(4,2)
//! view = full[1:4]                       # [[3,4],[5,6],[7,8]] storage_offset 2
//! mask = torch.tensor([[False,False],[True,True],[True,True]],device='cuda')
//! view.masked_fill(mask, -1.0).flatten().cpu().tolist()  # -> [3,4,-1,-1,-1,-1]
//! ```
//!
//! VERDICT: GENERATOR MUST FIX (separate dispatch from #1660). Tracking #1661
//! (filed via crosslink, --kind blocker). `#[ignore]`d so it does not block the
//! #1660 gauntlet — #1661 is a distinct follow-up dispatch; the issue is now
//! tracked and the failing test is the reproduction.

#![cfg(feature = "cuda")]

use ferrotorch_core::{BoolTensor, Device, Tensor, TensorStorage};
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

/// CUDA [4,2] = [[1,2],[3,4],[5,6],[7,8]], narrow rows 1..4 -> logical [3,2] =
/// [[3,4],[5,6],[7,8]] with storage_offset 2 and `is_contiguous() == true`.
fn narrowed_cuda_view() -> Tensor<f32> {
    let full = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2])
        .to(Device::Cuda(0))
        .expect("to cuda");
    let view = full.narrow(0, 1, 3).expect("narrow rows 1..4");
    assert_eq!(view.shape(), &[3, 2]);
    assert!(view.is_contiguous());
    assert_ne!(view.storage_offset(), 0);
    view
}

// live-torch symbolic constant (see module docs)
const TORCH_MASKED_FILL: [f32; 6] = [3.0, 4.0, -1.0, -1.0, -1.0, -1.0];

// #1661 FIXED: `masked_fill` now `.contiguous()`-normalises the narrowed CUDA
// view before the kernel read (ferrotorch-core grad_fns/indexing.rs), and
// `launch_masked_fill` validates + launches on the LOGICAL numel (the pooled
// `.contiguous()` buffer is a backing store `>= n`), so the storage_offset is
// honoured. This test is now a permanent regression guard (NOT `#[ignore]`d).
#[test]
fn masked_fill_narrowed_offset_view_gpu_matches_torch() {
    ensure_cuda();
    let view = narrowed_cuda_view();
    // mask = [[F,F],[T,T],[T,T]] resident on CUDA -> masked_fill_bt GPU path.
    let mask = BoolTensor::from_vec(vec![false, false, true, true, true, true], vec![3, 2])
        .expect("mask")
        .to(Device::Cuda(0))
        .expect("mask cuda");
    let got = view.masked_fill(&mask, -1.0).expect(
        "masked_fill on narrowed CUDA view must honour storage_offset (currently errors: \
         input numel 8 != mask numel 6)",
    );
    assert_eq!(
        host_f32(&got),
        TORCH_MASKED_FILL.to_vec(),
        "masked_fill on narrowed-offset CUDA view: ferrotorch dropped storage_offset"
    );
    assert!(got.is_cuda(), "masked_fill result must stay CUDA-resident");
}

// ---------------------------------------------------------------------------
// NO-REGRESSION guards (#1661): masked_fill on NORMAL offset-0 contiguous CUDA
// inputs must stay correct vs torch, including non-256-multiple sizes and sizes
// exceeding ROUND_ELEMENTS (the logical-len launch must launch exactly `n`
// threads over a possibly over-allocated backing store, never the raw len).
// ---------------------------------------------------------------------------

/// live-torch (2.11.0+cu130, RTX 3090): n=5, mask [T,F,T,F,T], fill -7.0.
///   `torch.tensor([1,2,3,4,5]).cuda().masked_fill(m, -7.0) -> [-7,2,-7,4,-7]`
const TORCH_FILL_N5: [f32; 5] = [-7.0, 2.0, -7.0, 4.0, -7.0];

#[test]
fn masked_fill_normal_offset0_small_nonmultiple_gpu_matches_torch() {
    ensure_cuda();
    let x = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5])
        .to(Device::Cuda(0))
        .expect("to cuda");
    assert_eq!(x.storage_offset(), 0);
    let mask = BoolTensor::from_vec(vec![true, false, true, false, true], vec![5])
        .expect("mask")
        .to(Device::Cuda(0))
        .expect("mask cuda");
    let got = x.masked_fill(&mask, -7.0).expect("masked_fill n=5");
    assert!(got.is_cuda());
    assert_eq!(host_f32(&got), TORCH_FILL_N5.to_vec());
}

#[test]
fn masked_fill_normal_offset0_above_round_elements_gpu_matches_torch() {
    ensure_cuda();
    // n=300 > ROUND_ELEMENTS=256, non-multiple. mask[i] = (i % 3 == 0) -> fill -1.
    //   torch sum of result = 29900.0 (live torch 2.11.0+cu130), first6
    //   [-1,1,2,-1,4,5], last3 [-1,298,299].
    let data: Vec<f32> = (0..300).map(|i| i as f32).collect();
    let mask_vec: Vec<bool> = (0..300).map(|i| i % 3 == 0).collect();
    let x = cpu_f32(&data, &[300]).to(Device::Cuda(0)).expect("to cuda");
    assert_eq!(x.storage_offset(), 0);
    let mask = BoolTensor::from_vec(mask_vec, vec![300])
        .expect("mask")
        .to(Device::Cuda(0))
        .expect("mask cuda");
    let got = x.masked_fill(&mask, -1.0).expect("masked_fill n=300");
    assert!(got.is_cuda());
    let r = host_f32(&got);
    assert_eq!(r.len(), 300);
    assert_eq!(&r[..6], &[-1.0, 1.0, 2.0, -1.0, 4.0, 5.0]);
    assert_eq!(&r[297..], &[-1.0, 298.0, 299.0]);
    let sum: f32 = r.iter().sum();
    assert_eq!(
        sum, 29900.0,
        "masked_fill n=300 sum diverged from live torch"
    );
}
