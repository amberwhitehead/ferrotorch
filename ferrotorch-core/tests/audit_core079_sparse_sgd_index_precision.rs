//! Red-then-green regression tests for audit finding CORE-079 (crosslink
//! #1773): CUDA sparse SGD loses row-index precision above 2^24
//! (CLASS-V — both CUDA lanes encoded `usize` row indices as f32 for the
//! `scatter_add_rows_*` ABI; integers above 2^24 are not exactly
//! representable in f32, so a validated row index rounds to a
//! neighboring row before the scatter consumes it).
//!
//! Observed at HEAD (red run, 2026-06-12, `--features gpu`, RTX 3090):
//! - f32 lane: the update for row 2^24+1 landed on row 2^24
//!   (`p[2^24+1] == 0.0`, `p[2^24] == -1.0` — exactly the f32 rounding).
//! - f64 lane: every CUDA f64 sparse-SGD step failed with
//!   "scatter_add_rows_f64 GPU op not yet implemented".
//!
//! torch oracle (live session, torch 2.11.0+cu130, RTX 3090):
//!
//! ```python
//! >>> V = 2**24 + 2
//! >>> p = torch.zeros(V, device="cuda", requires_grad=True)
//! >>> p.grad = torch.sparse_coo_tensor(torch.tensor([[2**24 + 1]]),
//! ...                                  torch.tensor([1.0]), (V,)).cuda()
//! >>> torch.optim.SGD([p], lr=1.0).step()
//! >>> p[2**24+1].item(), p[2**24].item(), p[2**24-1].item()
//! (-1.0, 0.0, 0.0)        # int64 indices: the EXACT row updates
//! ```
//!
//! Post-fix contract: both CUDA lanes use the integer-index
//! `scatter_add_segments_{f32,f64}` kernels (i64 per-row segment ids,
//! the #1822/#1823 integer-ABI pattern); no float round-trip of indices.

#![cfg(feature = "gpu")]

use ferrotorch_core::{Device, SparseGrad, Tensor, TensorStorage};
use std::sync::Once;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend().expect("CUDA backend must initialize for the GPU lane");
    });
}

/// Row 2^24 + 1 must receive the update — not its f32-rounded neighbor
/// 2^24 (torch oracle: p[2^24+1] == -1.0, p[2^24] == 0.0).
#[test]
fn core079_gpu_f32_row_index_above_2pow24_is_exact() {
    ensure_cuda_backend();
    const V: usize = (1usize << 24) + 2;
    const TARGET: usize = (1usize << 24) + 1;

    let cpu_param =
        Tensor::<f32>::from_storage(TensorStorage::cpu(vec![0.0f32; V]), vec![V], false).unwrap();
    let mut param = cpu_param.to(Device::Cuda(0)).expect("param->cuda");

    // Scalar slabs (1-D param): one gradient value of 1.0 at TARGET.
    let grad = SparseGrad::<f32>::new(vec![TARGET], vec![1.0], vec![]).expect("grad");
    grad.apply_sgd(&mut param, 1.0).expect("gpu sparse sgd");

    assert!(param.is_cuda(), "param must stay on CUDA");
    let back = param.cpu().expect("gpu->cpu");
    let d = back.data().expect("data");
    assert!(
        (d[TARGET] - (-1.0)).abs() < 1e-6,
        "row 2^24+1 must be updated to -1.0 (torch oracle), got {}",
        d[TARGET]
    );
    assert!(
        d[TARGET - 1].abs() < 1e-6,
        "row 2^24 must stay 0.0 (f32 index encoding corrupts it to -1.0), got {}",
        d[TARGET - 1]
    );
    assert!(
        d[TARGET - 2].abs() < 1e-6,
        "row 2^24-1 must stay 0.0, got {}",
        d[TARGET - 2]
    );
}

/// The f64 CUDA lane performs an on-device update matching the CPU lane
/// (pre-fix: "scatter_add_rows_f64 GPU op not yet implemented" error on
/// every f64 CUDA sparse-SGD step). Duplicate indices accumulate.
#[test]
fn core079_gpu_f64_lane_updates_on_device() {
    ensure_cuda_backend();
    let data: Vec<f64> = (0..12).map(|i| i as f64 * 0.25).collect();
    let cpu_param =
        Tensor::<f64>::from_storage(TensorStorage::cpu(data), vec![4, 3], false).unwrap();
    let mut gpu_param = cpu_param.to(Device::Cuda(0)).expect("param->cuda");
    let mut cpu_clone = cpu_param.clone();

    // Duplicate index 2 — accumulation semantics must match CPU.
    let grad = SparseGrad::<f64>::new(
        vec![2, 0, 2],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        vec![3],
    )
    .expect("grad");

    grad.apply_sgd(&mut cpu_clone, 0.5).expect("cpu sgd");
    grad.apply_sgd(&mut gpu_param, 0.5)
        .expect("f64 CUDA sparse sgd must run on device (integer-index segments kernel)");

    assert!(gpu_param.is_cuda(), "param must stay on CUDA");
    let back = gpu_param.cpu().expect("gpu->cpu");
    let g = back.data().expect("gpu data");
    let c = cpu_clone.data().expect("cpu data");
    for (i, (a, b)) in g.iter().zip(c.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-12,
            "f64 CUDA vs CPU mismatch at {i}: gpu={a} cpu={b}"
        );
    }
}
