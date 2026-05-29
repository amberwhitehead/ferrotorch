//! ADVERSARIAL RE-AUDIT of the dim-aware GPU gather/scatter family
//! (commit b2793d6a9, #1545 / sub #1535). The builder verified only "nice"
//! contiguous cases. This file hunts the cases it skipped.
//!
//! # DIVERGENCE FOUND — non-contiguous (transposed) CUDA input ignores strides
//!
//! `ferrotorch_core::ops::indexing::{gather,scatter,scatter_value,scatter_add}`
//! take a CUDA fast path (guarded only by `input.is_cuda()`, NOT by
//! `is_contiguous()`) that passes `input.gpu_handle()` (the RAW physical
//! buffer, in original memory order, ignoring `strides()` and
//! `storage_offset()`) into the PTX launchers together with `factor(
//! input.shape(), dim)` computed from the LOGICAL (permuted) shape. See
//! `ferrotorch-core/src/ops/indexing.rs:175-231` (gather), `:333-387`
//! (scatter), `:480-541` (scatter_value), `:620-674` (scatter_add). None of
//! the four branches materialise `.contiguous()` first, unlike
//! `ferrotorch-core/src/ops/elementwise.rs:549` and `:592` which DO guard with
//! `if input.is_contiguous() { .. } else { .. }`.
//!
//! The kernels themselves (`ferrotorch-gpu/src/scatter_gather_kernels.rs:14-21`
//! module doc) explicitly assume a "C-contiguous" `[outer, axis, inner]`
//! buffer: `o*axis*inner + a*inner + k`. For a transposed view the logical
//! shape no longer matches the physical buffer layout, so every address is
//! computed against the wrong stride.
//!
//! The CPU path (the within-framework reference) does NOT diverge: its
//! `input.data_vec()?` (`ferrotorch-core/src/tensor.rs:723-752`) walks the
//! strides for a non-contiguous tensor, materialising the logical order. So
//! CPU == torch, GPU != torch — a real GPU-vs-CPU AND GPU-vs-torch divergence.
//!
//! ## Mechanism, gather example
//!
//! `base = [[1,2,3],[4,5,6]]` shape `[2,3]`, physical buffer
//! `[1,2,3,4,5,6]`. `base.transpose(0,1)` is the `[3,2]` view
//! `[[1,4],[2,5],[3,6]]` (strides `[1,3]`, SAME buffer). `torch.gather(view,
//! 1, idx)` with `idx=[[0,1],[1,0],[0,1]]` returns `[[1,4],[5,2],[3,6]]`
//! (verified live, torch 2.11.0+cu130, CUDA). ferrotorch's GPU kernel instead
//! reads the buffer as a fresh C-contiguous `[3,2]` = `[[1,2],[3,4],[5,6]]`
//! and returns `[[1,2],[4,3],[5,6]]` — wrong (observed left
//! `[1,2,4,3,5,6]`).
//!
//! # R-CHAR-3 provenance (live torch 2.11.0+cu130, CUDA — RTX 3090)
//!
//! ```python
//! import torch; d="cuda"
//! base = torch.tensor([[1.,2.,3.],[4.,5.,6.]], device=d)   # [2,3]
//! tt = base.t()                                            # [3,2] view, NOT contiguous
//! idx = torch.tensor([[0,1],[1,0],[0,1]], device=d)
//! torch.gather(tt, 1, idx).cpu().tolist()
//! #   [[1.,4.],[5.,2.],[3.,6.]]
//!
//! z = torch.zeros(2,3, device=d).t()                       # [3,2] view
//! src  = torch.tensor([[10.,20.],[30.,40.],[50.,60.]], device=d)
//! sidx = torch.tensor([[0,1],[1,0],[0,1]], device=d)
//! torch.scatter(z, 0, sidx, src).cpu().tolist()
//! #   [[10.,40.],[30.,20.],[0.,0.]]
//!
//! za   = torch.tensor([[1.,2.,3.],[4.,5.,6.]], device=d).t()   # [3,2] view [[1,4],[2,5],[3,6]]
//! aidx = torch.tensor([[0,0],[1,1],[0,1]], device=d)
//! asrc = torch.tensor([[1.,2.],[3.,4.],[5.,6.]], device=d)
//! torch.scatter_add(za, 1, aidx, asrc).cpu().tolist()
//! #   [[4.,4.],[2.,12.],[8.,12.]]
//!
//! # --- clean (regression-guard) fixtures, contiguous inputs ---
//! inp = torch.arange(1.,13.,device=d).reshape(3,4)
//! idx = torch.tensor([[0,3],[1,2],[3,0]], device=d)        # smaller than input
//! torch.gather(inp,1,idx).cpu().tolist()
//! #   [[1.,4.],[6.,7.],[12.,9.]]
//! z = torch.zeros(3,4,device=d)
//! src = torch.tensor([[1.,2.],[3.,4.],[5.,6.]],device=d)
//! torch.scatter(z,1,torch.tensor([[0,3],[1,2],[3,0]],device=d),src).cpu().tolist()
//! #   [[1.,0.,0.,2.],[0.,3.,4.,0.],[6.,0.,0.,5.]]
//! za = torch.zeros(3,device=d); aidx=torch.zeros(1000,dtype=torch.long,device=d)
//! torch.scatter_add(za,0,aidx,torch.ones(1000,device=d)).cpu().tolist()[0]
//! #   1000.0  (atomic, no lost updates)
//! ```

#![cfg(feature = "cuda")]

use ferrotorch_core::ops::indexing::scatter_value;
use ferrotorch_core::{Device, Tensor, TensorStorage, gather, scatter, scatter_add};
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

fn cpu_f64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .expect("cpu f64 tensor")
}

fn host_f32(t: &Tensor<f32>) -> Vec<f32> {
    t.cpu().expect("cpu()").data().unwrap().to_vec()
}

fn host_f64(t: &Tensor<f64>) -> Vec<f64> {
    t.cpu().expect("cpu()").data().unwrap().to_vec()
}

// ===========================================================================
// DIVERGENCE: non-contiguous (transposed) CUDA input — strides ignored.
// These are HARD FAILURES (release-blocker): the GPU kernel silently
// corrupts the result for any transposed/permuted CUDA tensor, a class of
// inputs PyTorch handles transparently.
//
// TRACKING: ferrotorch-core/src/ops/indexing.rs CUDA fast paths must
// `.contiguous()`-materialise the input (and src) before dispatch, OR the
// kernels must honor strides.
// ===========================================================================

/// Divergence: `gather` on a transposed (non-contiguous) CUDA tensor diverges
/// from `torch.gather` because `ferrotorch-core/src/ops/indexing.rs:175`
/// passes the raw physical buffer (ignoring `strides()`) with the LOGICAL
/// permuted shape into `gather_dim_f32`.
/// Upstream torch returns `[[1,4],[5,2],[3,6]]`; ferrotorch GPU returns the
/// contiguous-misread `[[1,2],[4,3],[5,6]]`.
#[test]
fn divergence_gather_transposed_cuda_input_f32() {
    ensure_cuda();
    // base [2,3] contiguous -> transpose to [3,2] non-contiguous view on GPU.
    let base = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .to(Device::Cuda(0))
        .expect("to cuda");
    let tt = base.transpose(0, 1).expect("transpose"); // [3,2] view, non-contig
    assert!(tt.is_cuda());
    assert!(!tt.is_contiguous(), "transposed view must be non-contiguous");

    // gather along dim=1 of the [3,2] view, index shape [3,2].
    let index = [0usize, 1, 1, 0, 0, 1];
    let out = gather(&tt, 1, &index, &[3, 2]).expect("gpu gather transposed");
    assert!(out.is_cuda(), "result must stay GPU-resident");

    // CPU reference honors strides (tensor.rs:723) and equals torch.
    let cpu_view = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .transpose(0, 1)
        .expect("cpu transpose");
    let cpu_out = gather(&cpu_view, 1, &index, &[3, 2]).expect("cpu gather transposed");
    assert_eq!(
        cpu_out.data_vec().unwrap(),
        vec![1.0, 4.0, 5.0, 2.0, 3.0, 6.0],
        "CPU reference must equal torch (sanity)"
    );

    // torch.gather(base.t(), 1, idx) == [[1,4],[5,2],[3,6]].
    assert_eq!(
        host_f32(&out),
        vec![1.0, 4.0, 5.0, 2.0, 3.0, 6.0],
        "GPU gather on transposed input must match torch (and CPU)"
    );
}

/// Divergence: `scatter` on a transposed (non-contiguous) CUDA `self` diverges
/// from `torch.scatter` (`ferrotorch-core/src/ops/indexing.rs:333` clones the
/// raw buffer and writes against the logical shape, ignoring strides).
/// Upstream torch returns `[[10,40],[30,20],[0,0]]`; ferrotorch GPU observed
/// `[50,40,30,60,0,0]`.
#[test]
fn divergence_scatter_transposed_cuda_self_f32() {
    ensure_cuda();
    // self: zeros [2,3] -> transpose -> [3,2] non-contig view.
    let z = cpu_f32(&[0.0; 6], &[2, 3])
        .to(Device::Cuda(0))
        .expect("to cuda");
    let zt = z.transpose(0, 1).expect("transpose"); // [3,2] view
    assert!(!zt.is_contiguous());
    let src = cpu_f32(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[3, 2])
        .to(Device::Cuda(0))
        .expect("src cuda");
    let index = [0usize, 1, 1, 0, 0, 1];
    let out = scatter(&zt, 0, &index, &[3, 2], &src).expect("gpu scatter transposed");
    assert!(out.is_cuda());

    let zt_cpu = cpu_f32(&[0.0; 6], &[2, 3]).transpose(0, 1).expect("cpu t");
    let src_cpu = cpu_f32(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[3, 2]);
    let cpu_out = scatter(&zt_cpu, 0, &index, &[3, 2], &src_cpu).expect("cpu scatter t");
    assert_eq!(
        cpu_out.data_vec().unwrap(),
        vec![10.0, 40.0, 30.0, 20.0, 0.0, 0.0]
    );

    // torch.scatter(z.t(), 0, idx, src) == [[10,40],[30,20],[0,0]].
    assert_eq!(
        host_f32(&out),
        vec![10.0, 40.0, 30.0, 20.0, 0.0, 0.0],
        "GPU scatter into transposed self must match torch (and CPU)"
    );
}

/// Divergence: `scatter_add` on a transposed (non-contiguous) CUDA `self`
/// with NON-ZERO `self` data diverges from `torch.scatter_add`
/// (`ferrotorch-core/src/ops/indexing.rs:620`). The `self` clone is read in
/// physical (un-permuted) order, so the preserved-and-accumulated values land
/// in the wrong slots. Upstream torch returns `[[4,4],[2,12],[8,12]]`.
///
/// NB: a zeros-`self` fixture would MASK this divergence (an all-zeros buffer
/// reads identically contiguous or strided) — the builder's "nice" cases used
/// zeros. The non-zero `self` here is what exposes it.
#[test]
fn divergence_scatter_add_transposed_cuda_nonzero_self_f32() {
    ensure_cuda();
    // self = [[1,2,3],[4,5,6]] -> transpose -> [[1,4],[2,5],[3,6]] [3,2] view.
    let z = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .to(Device::Cuda(0))
        .expect("to cuda");
    let zt = z.transpose(0, 1).expect("transpose"); // [3,2] view
    assert!(!zt.is_contiguous());
    let src = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2])
        .to(Device::Cuda(0))
        .expect("src cuda");
    let index = [0usize, 0, 1, 1, 0, 1];
    let out = scatter_add(&zt, 1, &index, &[3, 2], &src).expect("gpu scatter_add t");
    assert!(out.is_cuda());

    let zt_cpu = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .transpose(0, 1)
        .expect("cpu t");
    let src_cpu = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let cpu_out = scatter_add(&zt_cpu, 1, &index, &[3, 2], &src_cpu).expect("cpu sa t");
    assert_eq!(
        cpu_out.data_vec().unwrap(),
        vec![4.0, 4.0, 2.0, 12.0, 8.0, 12.0]
    );

    // torch.scatter_add(self.t(), 1, idx, src) == [[4,4],[2,12],[8,12]].
    assert_eq!(
        host_f32(&out),
        vec![4.0, 4.0, 2.0, 12.0, 8.0, 12.0],
        "GPU scatter_add into transposed nonzero self must match torch (and CPU)"
    );
}

// ===========================================================================
// REGRESSION GUARDS — hard cases that ARE clean on contiguous inputs.
// These PASS; they pin the harder coverage the builder skipped so a future
// regression is caught.
// ===========================================================================

/// Index smaller than input along the non-gathered axis: input [3,4], gather
/// dim=1 with index [3,2]. Kernel must iterate index-extent, not input-extent.
/// torch: [[1,4],[6,7],[12,9]].
#[test]
fn gather_smaller_index_than_input_f32() {
    ensure_cuda();
    let data: Vec<f32> = (1..=12).map(|v| v as f32).collect();
    let gpu = cpu_f32(&data, &[3, 4]).to(Device::Cuda(0)).unwrap();
    let index = [0usize, 3, 1, 2, 3, 0];
    let out = gather(&gpu, 1, &index, &[3, 2]).expect("gpu gather smaller idx");
    assert!(out.is_cuda());
    assert_eq!(out.shape(), &[3, 2]);
    assert_eq!(host_f32(&out), vec![1.0, 4.0, 6.0, 7.0, 12.0, 9.0]);
    let cpu_out = gather(&cpu_f32(&data, &[3, 4]), 1, &index, &[3, 2]).unwrap();
    assert_eq!(host_f32(&out), cpu_out.data().unwrap().to_vec());
}

/// Scatter with index smaller than self along non-scattered axis: self [3,4]
/// zeros, dim=1, index [3,2], src [3,2]. Untouched positions must stay 0
/// (in-place clone semantics). torch:
/// [[1,0,0,2],[0,3,4,0],[6,0,0,5]].
#[test]
fn scatter_smaller_index_preserves_untouched_f32() {
    ensure_cuda();
    let gpu = cpu_f32(&[0.0; 12], &[3, 4]).to(Device::Cuda(0)).unwrap();
    let src = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2])
        .to(Device::Cuda(0))
        .unwrap();
    let index = [0usize, 3, 1, 2, 3, 0];
    let out = scatter(&gpu, 1, &index, &[3, 2], &src).expect("gpu scatter smaller idx");
    assert!(out.is_cuda());
    assert_eq!(
        host_f32(&out),
        vec![1.0, 0.0, 0.0, 2.0, 0.0, 3.0, 4.0, 0.0, 6.0, 0.0, 0.0, 5.0]
    );
}

/// Large-scale atomic accumulation: 1000 src=1.0 all targeting slot 0 of a
/// [3] zeros along dim 0. Atomic add must lose no updates -> slot0 == 1000.0
/// (exactly f32-representable). A non-atomic / last-write-wins kernel fails.
#[test]
fn scatter_add_large_atomic_no_lost_updates_f32() {
    ensure_cuda();
    let gpu = cpu_f32(&[0.0, 0.0, 0.0], &[3]).to(Device::Cuda(0)).unwrap();
    let src = cpu_f32(&vec![1.0f32; 1000], &[1000])
        .to(Device::Cuda(0))
        .unwrap();
    let index = vec![0usize; 1000];
    let out = scatter_add(&gpu, 0, &index, &[1000], &src).expect("gpu large atomic");
    assert!(out.is_cuda());
    let h = host_f32(&out);
    assert_eq!(h[0], 1000.0, "atomic add must accumulate all 1000 ones");
    assert_eq!(h[1], 0.0);
    assert_eq!(h[2], 0.0);
}

/// f64 large-scale atomic accumulation companion.
#[test]
fn scatter_add_large_atomic_no_lost_updates_f64() {
    ensure_cuda();
    let gpu = cpu_f64(&[0.0, 0.0, 0.0], &[3]).to(Device::Cuda(0)).unwrap();
    let src = cpu_f64(&vec![1.0f64; 1000], &[1000])
        .to(Device::Cuda(0))
        .unwrap();
    let index = vec![0usize; 1000];
    let out = scatter_add(&gpu, 0, &index, &[1000], &src).expect("gpu large atomic f64");
    assert!(out.is_cuda());
    assert_eq!(host_f64(&out)[0], 1000.0);
}

/// scatter vs scatter_value distinction: scatter writes a SRC tensor,
/// scatter_value writes a SCALAR. Confirm the two are not swapped and each
/// matches torch on the SAME index/self.
/// self zeros [3,2] dim=0 idx=[[0,1],[2,0]] (shape [2,2]):
///   scatter src=[[7,8],[9,10]] -> [[7,10],[0,8],[9,0]]
///   scatter_value 5.0          -> [[5,5],[0,5],[5,0]]
#[test]
fn scatter_vs_scatter_value_not_swapped_f32() {
    ensure_cuda();
    let base = cpu_f32(&[0.0; 6], &[3, 2]);
    let gpu = base.clone().to(Device::Cuda(0)).unwrap();
    let index = [0usize, 1, 2, 0];

    // scatter (tensor src)
    let src = cpu_f32(&[7.0, 8.0, 9.0, 10.0], &[2, 2])
        .to(Device::Cuda(0))
        .unwrap();
    let s_out = scatter(&gpu, 0, &index, &[2, 2], &src).expect("gpu scatter");
    let s_ref = scatter(
        &base,
        0,
        &index,
        &[2, 2],
        &cpu_f32(&[7.0, 8.0, 9.0, 10.0], &[2, 2]),
    )
    .unwrap();
    assert_eq!(host_f32(&s_out), s_ref.data().unwrap().to_vec());
    assert_eq!(host_f32(&s_out), vec![7.0, 10.0, 0.0, 8.0, 9.0, 0.0]);

    // scatter_value (scalar)
    let gpu2 = base.clone().to(Device::Cuda(0)).unwrap();
    let v_out = scatter_value(&gpu2, 0, &index, &[2, 2], 5.0f32).expect("gpu scatter_value");
    assert_eq!(host_f32(&v_out), vec![5.0, 5.0, 0.0, 5.0, 5.0, 0.0]);
}

/// scatter must PRESERVE the untouched positions of `self` (in-place clone
/// semantics, not zero-fill). self = [1..6] reshaped [2,3], dim=1, write
/// src=[[10],[20]] at idx [[2],[0]]. torch: [[1,2,10],[20,5,6]].
#[test]
fn scatter_preserves_self_untouched_positions_f32() {
    ensure_cuda();
    let base = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let gpu = base.clone().to(Device::Cuda(0)).unwrap();
    let src = cpu_f32(&[10.0, 20.0], &[2, 1]).to(Device::Cuda(0)).unwrap();
    let index = [2usize, 0];
    let out = scatter(&gpu, 1, &index, &[2, 1], &src).expect("gpu scatter preserve");
    assert!(out.is_cuda());
    assert_eq!(host_f32(&out), vec![1.0, 2.0, 10.0, 20.0, 5.0, 6.0]);
}

/// 3D non-square [2,5,3], gather at each axis. Verifies the
/// outer/axis/inner factorisation does not swap a row/col stride.
#[test]
fn gather_3d_nonsquare_each_axis_f32() {
    ensure_cuda();
    let data: Vec<f32> = (0..30).map(|v| v as f32).collect(); // [2,5,3]
    let gpu = cpu_f32(&data, &[2, 5, 3]).to(Device::Cuda(0)).unwrap();
    let cpu = cpu_f32(&data, &[2, 5, 3]);

    // dim=0, index [1,5,3] all zeros -> picks slab 0 == arange(0..15).
    let i0 = vec![0usize; 15];
    let o0 = gather(&gpu, 0, &i0, &[1, 5, 3]).unwrap();
    let r0 = gather(&cpu, 0, &i0, &[1, 5, 3]).unwrap();
    assert_eq!(host_f32(&o0), r0.data().unwrap().to_vec());
    assert_eq!(o0.shape(), &[1, 5, 3]);

    // dim=1, index [2,1,3] all zeros -> first row of each slab.
    let i1 = vec![0usize; 6];
    let o1 = gather(&gpu, 1, &i1, &[2, 1, 3]).unwrap();
    let r1 = gather(&cpu, 1, &i1, &[2, 1, 3]).unwrap();
    assert_eq!(host_f32(&o1), r1.data().unwrap().to_vec());

    // dim=2, index [2,5,1] all index 2 -> last col of each row.
    let i2 = vec![2usize; 10];
    let o2 = gather(&gpu, 2, &i2, &[2, 5, 1]).unwrap();
    let r2 = gather(&cpu, 2, &i2, &[2, 5, 1]).unwrap();
    assert_eq!(host_f32(&o2), r2.data().unwrap().to_vec());
}

/// bf16/f16 CUDA must reject with NotImplementedOnCuda (no dim-aware kernel).
#[test]
fn gather_bf16_f16_cuda_rejects() {
    ensure_cuda();
    use half::{bf16, f16};
    let bf = Tensor::<bf16>::from_storage(
        TensorStorage::cpu(vec![bf16::from_f32(1.0), bf16::from_f32(2.0)]),
        vec![2],
        false,
    )
    .unwrap()
    .to(Device::Cuda(0))
    .unwrap();
    assert!(gather(&bf, 0, &[0usize], &[1]).is_err());

    let hf = Tensor::<f16>::from_storage(
        TensorStorage::cpu(vec![f16::from_f32(1.0), f16::from_f32(2.0)]),
        vec![2],
        false,
    )
    .unwrap()
    .to(Device::Cuda(0))
    .unwrap();
    assert!(gather(&hf, 0, &[0usize], &[1]).is_err());
}
