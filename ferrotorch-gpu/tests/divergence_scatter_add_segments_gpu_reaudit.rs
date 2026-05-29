//! ADVERSARIAL RE-AUDIT of the GPU `scatter_add_segments` kernel
//! (`SCATTER_ADD_SEGMENTS_F{32,64}_PTX` / `gpu_scatter_add_segments_f{32,64}`
//! in `ferrotorch-gpu/src/scatter_gather_kernels.rs`, wired through
//! `ferrotorch_core::ops::scatter::scatter_add_segments`) for commit
//! 7f6fbef88 (#1545 sub #1535).
//!
//! The original landing test (`divergence_scatter_add_segments_gpu.rs`) only
//! covered basic + duplicate-atomic + empty-row + bf16/f16-reject. This re-audit
//! hits what the builder did NOT:
//!
//!   1. NON-CONTIGUOUS src (the #1655 stride-bug class) — a transposed CUDA
//!      `[D,E]`->`[E,D]` view and a narrowed/offset view. The CUDA dispatch
//!      must `.contiguous()`-materialise src before the kernel reads it as
//!      C-contiguous `[E,D]`; if it passed the raw strided buffer it would
//!      misread elements.
//!   2. LARGE-SCALE atomic correctness — 5000 rows, D=4, deterministic segment
//!      ids covering all `dim_size` rows, all values 1.0 (exactly representable
//!      sum), so every output cell must equal its exact edge count (no lost
//!      atomic updates / races). f32 AND f64.
//!   3. OUT-OF-RANGE / NEGATIVE segment id — GPU path must reject IDENTICALLY
//!      to the CPU path (shared host validation hoisted ahead of the device
//!      split), never launch and corrupt OOB device memory.
//!   4. dim_size > max segment id (trailing all-zero rows), exact zero.
//!   5. D=1 single feature column and single-row src [1,D]. Degenerate shapes.
//!   6. EMPTY src [0,D] (E=0) — zero-element launch, shape [dim_size, D] zeros.
//!   7. is_cuda() preserved; GPU == ferrotorch CPU for every valid case.
//!
//! # DIVERGENCE FOUND (#1657, release-blocker, left un-#[ignore]d)
//!
//! `scatter_add_segments_gpu_f32_narrowed_offset_view_matches_torch` FAILS:
//! the CUDA path ignores `storage_offset` for a row-narrowed src view.
//! `Tensor::is_contiguous()` (`ferrotorch-core/src/tensor.rs:1482`) checks only
//! strides, NOT `storage_offset`; a row-narrowed view has row-major strides so
//! `contiguous()` (`methods.rs:1572-1573`) returns `input.clone()` unchanged and
//! the kernel reads `gpu_handle()` from the base buffer (element 0), dropping the
//! offset. torch / ferrotorch-CPU agree on `[10,12,5,6]`; GPU returns `[6,8,3,4]`.
//!
//! # R-CHAR-3 provenance (live torch 2.11.0+cu130, CUDA, RTX 3090)
//!
//! Every expected value is the output of `torch.zeros(dim_size, D).index_add_(
//! 0, index, src)` on the identical fixture (the canonical segmented row sum,
//! same as `torch_scatter.scatter_add(src, index, dim=0, dim_size=N)`). These
//! are symbolic constants traceable to the torch call, NOT self-derived from
//! ferrotorch.
//!
//! ```python
//! import torch
//! d = "cuda"
//!
//! # (1a) NON-CONTIGUOUS transposed view:
//! base = torch.tensor([[1.,2.,3.],[4.,5.,6.]], device=d)   # [D=2, E=3]
//! src_view = base.t()                                       # [E=3, D=2] non-contig
//! #   logical src_view == [[1,4],[2,5],[3,6]]
//! idx = torch.tensor([0,1,0], device=d)
//! torch.zeros(2,2,device=d).index_add_(0, idx, src_view)
//! #   out[0]=row0+row2=[1+3,4+6]=[4,10]; out[1]=row1=[2,5]
//! #   tensor([[4., 10.], [2., 5.]])   ->  flat [4, 10, 2, 5]
//!
//! # (1b) NARROWED / OFFSET view (the failing case):
//! full = torch.arange(1,9,dtype=torch.float32,device=d).reshape(4,2)
//! view = full[1:4]                 # [[3,4],[5,6],[7,8]], storage_offset 2
//! torch.zeros(2,2,device=d).index_add_(0, torch.tensor([0,1,0],device=d), view)
//! #   out[0]=row0+row2=[3+7,4+8]=[10,12]; out[1]=row1=[5,6]  -> flat [10,12,5,6]
//!
//! # (2) LARGE-SCALE atomic: E=5000, D=4, all ones, ids = e % N, N=37.
//! #   out[i, :] == count of e with (e % 37 == i) for each col.
//! #   total sum == E*D == 20000.0 (exact in f32).
//!
//! # (5) D=1 single col: src=[[2.],[3.],[5.]], idx=[1,1,0], dim_size=2
//! torch.zeros(2,1,device=d).index_add_(0, torch.tensor([1,1,0],device=d),
//!     torch.tensor([[2.],[3.],[5.]],device=d))
//! #   out[0]=5; out[1]=2+3=5  -> flat [5, 5]
//!
//! # (6) EMPTY src E=0, D=3, dim_size=4 -> zeros(4,3)
//! ```

#![cfg(feature = "cuda")]

use ferrotorch_core::{Device, Tensor, TensorStorage, scatter_add_segments};
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
// (1a) NON-CONTIGUOUS transposed src view.
//     Build a CUDA [D=2, E=3] tensor and .transpose -> [E=3, D=2] non-contig
//     view. The kernel reads src as C-contiguous [E,D]; the transposed view IS
//     non-contiguous-by-strides so `contiguous()` triggers strided_copy and
//     this case PASSES (the trap is the offset-only case below).
// ===========================================================================

#[test]
fn scatter_add_segments_gpu_f32_transposed_view_matches_torch() {
    ensure_cuda();
    // base [D=2, E=3] = [[1,2,3],[4,5,6]] on CUDA.
    let base = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .to(Device::Cuda(0))
        .expect("to cuda");
    // transpose(0,1) -> logical [E=3, D=2] non-contiguous view == [[1,4],[2,5],[3,6]]
    let src_view = base.transpose(0, 1).expect("transpose");
    assert_eq!(src_view.shape(), &[3, 2]);
    assert!(
        !src_view.is_contiguous(),
        "transposed view must be non-contiguous to exercise the stride path"
    );

    let index = [0i64, 1, 0];
    let out = scatter_add_segments(&src_view, &index, 2).expect("gpu transposed scatter");
    assert!(out.is_cuda(), "result must stay GPU-resident");
    assert_eq!(out.shape(), &[2, 2]);
    // torch.zeros(2,2).index_add_(0, [0,1,0], [[1,4],[2,5],[3,6]]) == [[4,10],[2,5]]
    assert_eq!(
        host_f32(&out),
        vec![4.0, 10.0, 2.0, 5.0],
        "transposed-view src must be read by LOGICAL [E,D] layout, not the raw \
         row-major transposed buffer (#1655 stride-bug class)"
    );

    // GPU must equal the ferrotorch CPU reference on the same logical view.
    let cpu_view = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .transpose(0, 1)
        .expect("cpu transpose");
    let cpu_out = scatter_add_segments(&cpu_view, &index, 2).expect("cpu transposed scatter");
    assert_eq!(host_f32(&out), cpu_out.data().unwrap().to_vec());
}

#[test]
fn scatter_add_segments_gpu_f64_transposed_view_matches_torch() {
    ensure_cuda();
    let base = cpu_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .to(Device::Cuda(0))
        .expect("to cuda");
    let src_view = base.transpose(0, 1).expect("transpose");
    assert!(!src_view.is_contiguous());
    let index = [0i64, 1, 0];
    let out = scatter_add_segments(&src_view, &index, 2).expect("gpu transposed scatter f64");
    assert!(out.is_cuda());
    assert_eq!(
        host_f64(&out),
        vec![4.0, 10.0, 2.0, 5.0],
        "f64 transposed-view must read logical [E,D] layout"
    );
}

// ===========================================================================
// (1b) *** DIVERGENCE #1657 *** — NON-CONTIGUOUS by STORAGE_OFFSET only.
//
// Build [E=4, D=2] then narrow rows [1..4) -> logical [E=3, D=2] == rows
// [[3,4],[5,6],[7,8]] with storage_offset == 2. The narrowed view has row-major
// strides [2,1] so `Tensor::is_contiguous()` returns TRUE (it inspects strides
// only, never storage_offset, ferrotorch-core/src/tensor.rs:1482) and
// `contiguous()` returns `input.clone()` UNCHANGED. The CUDA dispatch
// (`scatter_add_segments_cuda`, ops/scatter.rs:157) then hands the kernel
// `src.gpu_handle()` — the BASE buffer pointer — so the kernel reads rows
// 0..3 (== [[1,2],[3,4],[5,6]]), dropping the +2-element offset.
//
// Upstream: torch.zeros(2,2).index_add_(0,[0,1,0], full[1:4]) == [[10,12],[5,6]].
// ferrotorch CPU (data_vec honours offset)              == [10,12,5,6].
// ferrotorch GPU (offset dropped)                       == [ 6, 8,3,4]  (WRONG).
//
// This is a release-blocker: silent wrong numerical results from a GNN
// message-passing primitive whenever the src is a row-sliced CUDA view (a
// completely routine input). Left UN-#[ignore]d — the failing test IS the block.
// Tracking: #1657.
// ===========================================================================

#[test]
fn scatter_add_segments_gpu_f32_narrowed_offset_view_matches_torch() {
    ensure_cuda();
    let full = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2])
        .to(Device::Cuda(0))
        .expect("to cuda");
    let view = full.narrow(0, 1, 3).expect("narrow rows 1..4");
    assert_eq!(view.shape(), &[3, 2]);
    let index = [0i64, 1, 0];
    let out = scatter_add_segments(&view, &index, 2).expect("gpu narrowed scatter");
    assert!(out.is_cuda());
    // src_view == [[3,4],[5,6],[7,8]]; idx=[0,1,0]:
    //   out[0]=row0+row2=[3+7,4+8]=[10,12]; out[1]=row1=[5,6]
    // torch: torch.zeros(2,2).index_add_(0,[0,1,0], full[1:4]) == [[10,12],[5,6]]
    assert_eq!(
        host_f32(&out),
        vec![10.0, 12.0, 5.0, 6.0],
        "narrowed view with non-zero storage_offset must honour the offset (#1657)"
    );
    let cpu_view = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2])
        .narrow(0, 1, 3)
        .expect("cpu narrow");
    let cpu_out = scatter_add_segments(&cpu_view, &index, 2).expect("cpu narrowed scatter");
    assert_eq!(host_f32(&out), cpu_out.data().unwrap().to_vec());
}

// ===========================================================================
// (2) LARGE-SCALE atomic correctness — 5000 rows, D=4, ids = e % 37,
//     all values 1.0. out[i, col] must equal the exact count of e with
//     (e % 37 == i). No lost atomic updates. f32 AND f64.
// ===========================================================================

const BIG_E: usize = 5000;
const BIG_D: usize = 4;
const BIG_N: usize = 37;

fn big_index() -> Vec<i64> {
    (0..BIG_E).map(|e| (e % BIG_N) as i64).collect()
}

fn big_expected_counts() -> Vec<usize> {
    let mut counts = vec![0usize; BIG_N];
    for e in 0..BIG_E {
        counts[e % BIG_N] += 1;
    }
    counts
}

#[test]
fn scatter_add_segments_gpu_f32_large_scale_atomic_matches_torch() {
    ensure_cuda();
    let cpu = cpu_f32(&vec![1.0f32; BIG_E * BIG_D], &[BIG_E, BIG_D]);
    let gpu = cpu.to(Device::Cuda(0)).expect("to cuda");
    let index = big_index();
    let out = scatter_add_segments(&gpu, &index, BIG_N).expect("gpu large-scale");
    assert!(out.is_cuda());
    assert_eq!(out.shape(), &[BIG_N, BIG_D]);
    let got = host_f32(&out);
    let counts = big_expected_counts();
    // torch.index_add_ with all-ones src yields, in each output cell, the exact
    // edge count for that segment (1.0 * count is exact in f32 for count<2^24).
    for i in 0..BIG_N {
        for col in 0..BIG_D {
            let v = got[i * BIG_D + col];
            assert_eq!(
                v, counts[i] as f32,
                "row {i} col {col}: lost atomic update — got {v}, expected exact \
                 edge count {}",
                counts[i]
            );
        }
    }
    // Total must equal E*D with no races.
    let total: f32 = got.iter().sum();
    assert_eq!(total, (BIG_E * BIG_D) as f32);
}

#[test]
fn scatter_add_segments_gpu_f64_large_scale_atomic_matches_torch() {
    ensure_cuda();
    let cpu = cpu_f64(&vec![1.0f64; BIG_E * BIG_D], &[BIG_E, BIG_D]);
    let gpu = cpu.to(Device::Cuda(0)).expect("to cuda");
    let index = big_index();
    let out = scatter_add_segments(&gpu, &index, BIG_N).expect("gpu large-scale f64");
    assert!(out.is_cuda());
    let got = host_f64(&out);
    let counts = big_expected_counts();
    for i in 0..BIG_N {
        for col in 0..BIG_D {
            let v = got[i * BIG_D + col];
            assert_eq!(
                v, counts[i] as f64,
                "f64 row {i} col {col}: lost atomic update — got {v}, expected {}",
                counts[i]
            );
        }
    }
    let total: f64 = got.iter().sum();
    assert_eq!(total, (BIG_E * BIG_D) as f64);
}

// ===========================================================================
// (3) OUT-OF-RANGE / NEGATIVE segment id — GPU path must reject IDENTICALLY to
//     the CPU path. torch.index_add_ errors on an index >= dim_size or < 0; the
//     ferrotorch host validator (hoisted ahead of the device split) must reject
//     BEFORE any device launch. A GPU path that launched and corrupted OOB
//     device memory would be a safety divergence.
// ===========================================================================

#[test]
fn scatter_add_segments_gpu_f32_oob_index_rejects_like_cpu() {
    ensure_cuda();
    let cpu = cpu_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let gpu = cpu.to(Device::Cuda(0)).expect("to cuda");
    // dim_size=2 so valid ids are {0,1}; 2 is OOB. torch.index_add_ errors.
    let index = [0i64, 2];
    let gpu_err = scatter_add_segments(&gpu, &index, 2);
    let cpu_err = scatter_add_segments(&cpu, &index, 2);
    assert!(
        gpu_err.is_err(),
        "GPU path must reject OOB segment id BEFORE launch (no device OOB write)"
    );
    assert!(cpu_err.is_err(), "CPU path must reject OOB segment id");
}

#[test]
fn scatter_add_segments_gpu_f32_negative_index_rejects_like_cpu() {
    ensure_cuda();
    let cpu = cpu_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let gpu = cpu.to(Device::Cuda(0)).expect("to cuda");
    let index = [0i64, -1];
    let gpu_err = scatter_add_segments(&gpu, &index, 2);
    let cpu_err = scatter_add_segments(&cpu, &index, 2);
    assert!(
        gpu_err.is_err(),
        "GPU path must reject negative segment id BEFORE launch"
    );
    assert!(cpu_err.is_err(), "CPU path must reject negative segment id");
}

// ===========================================================================
// (4) dim_size > max segment id — trailing all-zero rows, shape [dim_size, D].
// ===========================================================================

#[test]
fn scatter_add_segments_gpu_f32_trailing_zero_rows() {
    ensure_cuda();
    // ids only ever hit row 1; dim_size=5 -> rows 0,2,3,4 stay exactly 0.
    let cpu = cpu_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let gpu = cpu.to(Device::Cuda(0)).expect("to cuda");
    let index = [1i64, 1];
    let out = scatter_add_segments(&gpu, &index, 5).expect("gpu trailing zeros");
    assert!(out.is_cuda());
    assert_eq!(out.shape(), &[5, 2]);
    let got = host_f32(&out);
    // torch.zeros(5,2).index_add_(0,[1,1],[[1,2],[3,4]]):
    //   row1 = [1+3, 2+4] = [4,6]; all other rows exactly 0.
    let mut expected = vec![0.0f32; 10];
    expected[2] = 4.0;
    expected[3] = 6.0;
    assert_eq!(got, expected, "trailing rows past max segment id must be 0");
}

// ===========================================================================
// (5) D=1 single feature column AND single-row src [1, D]. Degenerate shapes.
// ===========================================================================

#[test]
fn scatter_add_segments_gpu_f32_d1_single_column() {
    ensure_cuda();
    // src=[[2],[3],[5]] D=1, idx=[1,1,0], dim_size=2.
    let cpu = cpu_f32(&[2.0, 3.0, 5.0], &[3, 1]);
    let gpu = cpu.to(Device::Cuda(0)).expect("to cuda");
    let index = [1i64, 1, 0];
    let out = scatter_add_segments(&gpu, &index, 2).expect("gpu D=1");
    assert!(out.is_cuda());
    assert_eq!(out.shape(), &[2, 1]);
    // torch: out[0]=5; out[1]=2+3=5 -> [5, 5]
    assert_eq!(host_f32(&out), vec![5.0, 5.0]);
}

#[test]
fn scatter_add_segments_gpu_f32_single_row_src() {
    ensure_cuda();
    // src=[[7,8,9]] E=1, D=3, idx=[2], dim_size=4.
    let cpu = cpu_f32(&[7.0, 8.0, 9.0], &[1, 3]);
    let gpu = cpu.to(Device::Cuda(0)).expect("to cuda");
    let index = [2i64];
    let out = scatter_add_segments(&gpu, &index, 4).expect("gpu single-row");
    assert!(out.is_cuda());
    assert_eq!(out.shape(), &[4, 3]);
    let got = host_f32(&out);
    // torch.zeros(4,3).index_add_(0,[2],[[7,8,9]]): row2=[7,8,9], rest 0.
    let mut expected = vec![0.0f32; 12];
    expected[6] = 7.0;
    expected[7] = 8.0;
    expected[8] = 9.0;
    assert_eq!(got, expected);
}

// ===========================================================================
// (6) EMPTY src [0, D] (E=0) — zero-element launch, returns zeros [dim_size, D]
//     without panic. torch.zeros(dim_size,D).index_add_(0, empty, empty) ==
//     zeros(dim_size,D).
// ===========================================================================

#[test]
fn scatter_add_segments_gpu_f32_empty_src() {
    ensure_cuda();
    // E=0, D=3, dim_size=4. src is an empty [0,3] CUDA tensor.
    let cpu = cpu_f32(&[], &[0, 3]);
    let gpu = cpu.to(Device::Cuda(0)).expect("to cuda");
    let index: [i64; 0] = [];
    let out = scatter_add_segments(&gpu, &index, 4).expect("gpu empty src");
    assert!(out.is_cuda());
    assert_eq!(out.shape(), &[4, 3]);
    assert_eq!(
        host_f32(&out),
        vec![0.0f32; 12],
        "empty src -> all-zero output"
    );
}
