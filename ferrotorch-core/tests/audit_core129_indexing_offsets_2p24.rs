//! CORE-129 (#1823, CLASS-V High) regression battery: CUDA indexing
//! backwards must route gradients to EXACT integer destinations for flat
//! offsets above 2^24 — the f32-encoded index upload
//! (`upload_f32_to_gpu(gather_dst_flat_indices(..))`) rounds
//! 16_777_217 (2^24 + 1) to 16_777_216, silently scattering the gradient to
//! the neighboring element.
//!
//! Pre-fix observed behavior (R-AHON-1 probe, pasted in #1823, RTX 3090):
//! `gather([2^24+8]-input, 0, [2^24+1]).sum().backward()` put
//! `grad[16777216]=1, grad[16777217]=0`.
//!
//! Oracle (LIVE `torch==2.11.0+cu130`, cuda, R-ORACLE-1(b)):
//!   - `x.gather(0, tensor([16777217])).sum().backward()` ->
//!     `x.grad.nonzero() == [16777217]`, value 1.0;
//!   - `x.index_select(0, tensor([16777217]))` backward -> same;
//!   - scatter grad_src with upstream grad `g[16777217]=7, g[16777216]=3` ->
//!     `src.grad == 7.0` (the f32-rounded offset reads the 3.0 neighbor);
//!   - 2-D `x[2, 2^23+4].index_select(1, [2^23+1])`, backward with
//!     `[[5],[9]]` -> nonzero exactly at `[0, 8388609]` (=5.0) and
//!     `[1, 8388609]` (=9.0); the o=1 flat destination (2^24 + 5) exceeds
//!     2^24 while the index VALUE itself (8_388_609) does not — pinning the
//!     dst-offset encoding, not just the index payload.
//!
//! All comparisons are exact: gradients here are pure routing of exact
//! small integers; the divergence is a WRONG DESTINATION, not rounding.
//! Memory: each buffer is ~67 MB f32 — fine on the 24 GB RTX 3090.

#![cfg(feature = "gpu")]

use ferrotorch_core::autograd::graph::{backward, backward_with_grad};
use ferrotorch_core::grad_fns::indexing::{index_select_1d, index_select_dim};
use ferrotorch_core::int_tensor::IntTensor;
use ferrotorch_core::{Device, Tensor, TensorStorage, gather, scatter};
use std::sync::Once;

static GPU_INIT: Once = Once::new();
fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for the CORE-129 GPU pins");
    });
}

const N: usize = (1 << 24) + 8; // 16_777_224
const TARGET: usize = (1 << 24) + 1; // 16_777_217 — rounds to 2^24 in f32

fn cuda_zeros(n: usize, rg: bool) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(vec![0.0f32; n]), vec![n], false)
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap()
        .requires_grad_(rg)
}

#[track_caller]
fn assert_exact_landing(g: &Tensor<f32>, target: usize, want: f32, what: &str) {
    assert!(g.is_cuda(), "{what}: gradient must stay CUDA-resident");
    let gd = g.cpu().unwrap().data_vec().unwrap();
    assert_eq!(
        gd[target], want,
        "{what}: gradient must land at the exact index {target} (torch oracle)"
    );
    assert_eq!(
        gd[target - 1],
        0.0,
        "{what}: the f32-rounded neighbor {} must stay zero",
        target - 1
    );
    assert_eq!(
        gd[target + 1],
        0.0,
        "{what}: the right neighbor must stay zero"
    );
}

/// GatherBackward dst offsets above 2^24.
#[test]
fn core129_cuda_gather_backward_lands_at_exact_offset_above_2p24() {
    ensure_cuda_backend();
    let x = cuda_zeros(N, true);
    let out = gather(&x, 0, &[TARGET], &[1]).unwrap();
    backward(&out.sum_all().unwrap()).unwrap();
    let g = x.grad().unwrap().expect("grad must exist");
    assert_exact_landing(&g, TARGET, 1.0, "gather backward");
}

/// IndexSelectBackward (1-D) index values above 2^24.
#[test]
fn core129_cuda_index_select_1d_backward_lands_at_exact_offset_above_2p24() {
    ensure_cuda_backend();
    let x = cuda_zeros(N, true);
    let out = index_select_1d(&x, &[TARGET]).unwrap();
    backward(&out.sum_all().unwrap()).unwrap();
    let g = x.grad().unwrap().expect("grad must exist");
    assert_exact_landing(&g, TARGET, 1.0, "index_select_1d backward");
}

/// ScatterBackward grad_src READS grad_output at a flat offset above 2^24:
/// upstream grad has 7.0 at the target and 3.0 at the f32-rounded neighbor,
/// so a rounded offset is observable as grad_src == 3.0.
#[test]
fn core129_cuda_scatter_backward_grad_src_reads_exact_offset_above_2p24() {
    ensure_cuda_backend();
    let inp = cuda_zeros(N, true);
    let src = cuda_zeros(1, true);
    let out = scatter(&inp, 0, &[TARGET], &[1], &src).unwrap();
    let mut g_host = vec![1.0f32; N];
    g_host[TARGET] = 7.0;
    g_host[TARGET - 1] = 3.0;
    let g = Tensor::from_storage(TensorStorage::cpu(g_host), vec![N], false)
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();
    backward_with_grad(&out, Some(&g)).unwrap();
    let gs = src.grad().unwrap().expect("grad_src must exist");
    assert!(gs.is_cuda(), "grad_src must stay CUDA-resident");
    assert_eq!(
        gs.cpu().unwrap().data_vec().unwrap(),
        vec![7.0],
        "grad_src must be grad_output[{TARGET}] = 7.0 (torch oracle); 3.0 \
         means the offset was rounded to {}",
        TARGET - 1
    );
    // grad_input: zeroed exactly at TARGET, untouched at the neighbor.
    let gi = inp.grad().unwrap().expect("grad_input must exist");
    let gid = gi.cpu().unwrap().data_vec().unwrap();
    assert_eq!(
        gid[TARGET], 0.0,
        "grad_input zeroed at the exact written slot"
    );
    assert_eq!(gid[TARGET - 1], 3.0, "neighbor keeps its upstream grad");
}

/// IndexSelectDimBackward dst offsets above 2^24 while the index VALUE
/// itself stays below 2^24 (outer=2 pushes the o=1 destinations past it).
#[test]
fn core129_cuda_index_select_dim_backward_dst_offsets_above_2p24() {
    ensure_cuda_backend();
    const M: usize = (1 << 23) + 4; // 8_388_612 per row, x2 rows
    const K: usize = (1 << 23) + 1; // 8_388_609 — fits in f32 exactly...
    // ...but the o=1 flat destination is M + K = 2^24 + 5, which does not.
    let x = Tensor::from_storage(TensorStorage::cpu(vec![0.0f32; 2 * M]), vec![2, M], false)
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap()
        .requires_grad_(true);
    let idx = IntTensor::<i64>::from_slice(&[K as i64], &[1]).unwrap();
    let out = index_select_dim(&x, 1, &idx).unwrap();
    let g = Tensor::from_storage(TensorStorage::cpu(vec![5.0f32, 9.0]), vec![2, 1], false)
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();
    backward_with_grad(&out, Some(&g)).unwrap();
    let gx = x.grad().unwrap().expect("grad must exist");
    assert!(gx.is_cuda(), "grad must stay CUDA-resident");
    let gd = gx.cpu().unwrap().data_vec().unwrap();
    // Torch oracle: nonzero exactly at [0, 8388609] = 5.0, [1, 8388609] = 9.0.
    assert_eq!(gd[K], 5.0, "row-0 grad lands at exact column {K}");
    let o1 = M + K; // flat 2^24 + 5
    assert_eq!(gd[o1], 9.0, "row-1 grad lands at exact flat offset {o1}");
    assert_eq!(gd[o1 - 1], 0.0, "f32-rounded neighbor must stay zero");
    assert_eq!(gd[o1 + 1], 0.0, "right neighbor must stay zero");
}
