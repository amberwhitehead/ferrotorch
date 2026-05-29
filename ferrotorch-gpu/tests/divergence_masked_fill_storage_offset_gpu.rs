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

#[test]
#[ignore = "divergence: masked_fill drops storage_offset on narrowed CUDA view; tracking #1661"]
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
}
