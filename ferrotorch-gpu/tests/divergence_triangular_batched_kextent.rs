//! Discriminator re-audit of commit `a439c4f53` — GPU triu/tril (#1545/#1535).
//!
//! Two concerns pinned here against LIVE torch 2.11 reference values:
//!
//!  1. BATCHED N-D INPUT (DIVERGENCE — ignored, tracking issue).
//!     `torch.triu` / `torch.tril` operate on the LAST TWO DIMS of any tensor
//!     with `dim() >= 2`, batching over all leading dims. Upstream:
//!     `pytorch/aten/src/ATen/native/TriangularOps.cpp:31`
//!       `TORCH_CHECK(self.dim() >= 2, "triu: input tensor must have at least 2 dimensions")`
//!     and the CUDA template batches via
//!     `pytorch/aten/src/ATen/native/cuda/TriangularOps.cu:120`
//!       `int64_t N_padded = c10::multiply_integers(sizes.begin(), sizes.end()-1) * last_dim_padded;`
//!     ferrotorch `triu`/`tril` HARD-REJECT anything that is not exactly 2-D at
//!     `ferrotorch-core/src/ops/tensor_ops.rs:52` / `:120`
//!       `if input.ndim() != 2 { return Err(InvalidArgument ...) }`
//!     so batched CUDA input never reaches the resident kernel — it errors out.
//!
//!  2. k-BEYOND-EXTENT + both rectangular orientations (POSITIVE coverage —
//!     should PASS; pins the resident-kernel math against torch for |k| larger
//!     than the matrix extent, which the builder only tested for k in -2..2).
//!
//! All expected values were produced by LIVE `torch.triu`/`torch.tril`
//! (torch 2.11.0+cu130), not copied from the ferrotorch side (R-CHAR-3).

#![cfg(feature = "cuda")]

use ferrotorch_core::{Device, Tensor, TensorStorage, tril, triu};
use ferrotorch_gpu::init_cuda_backend;

fn ensure_cuda() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        init_cuda_backend().expect("CUDA backend init");
    });
}

/// Row-major f32 tensor of `0,1,2,...` with the given shape.
fn arange(shape: Vec<usize>) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    Tensor::from_storage(TensorStorage::cpu(data), shape, false).expect("cpu tensor")
}

fn arange_f64(shape: Vec<usize>) -> Tensor<f64> {
    let n: usize = shape.iter().product();
    let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
    Tensor::from_storage(TensorStorage::cpu(data), shape, false).expect("cpu tensor")
}

// ===========================================================================
// 1. BATCHED N-D DIVERGENCE
// ===========================================================================

/// Divergence: ferrotorch's `triu` diverges from
/// `pytorch aten/src/ATen/native/TriangularOps.cpp:31` (`dim() >= 2`) and the
/// batching CUDA template `cuda/TriangularOps.cu:120` for a 4-D `[2,2,3,3]`
/// CUDA input.
/// Upstream applies the upper-triangular mask to each `[3,3]` matrix over the
/// LAST TWO DIMS, returning shape `[2,2,3,3]`.
/// ferrotorch returns `Err(InvalidArgument "triu: expected 2-D tensor")`
/// (`ferrotorch-core/src/ops/tensor_ops.rs:52`) — batched input never reaches
/// the GPU kernel.
/// Tracking: #1644
#[test]
#[ignore = "divergence: triu/tril reject N-D batched input torch handles over last-2-dims; tracking #1644"]
fn divergence_triu_batched_4d_on_cuda() {
    ensure_cuda();
    let cpu = arange(vec![2, 2, 3, 3]);
    let gpu = cpu.to(Device::Cuda(0)).expect("to cuda");

    // LIVE torch 2.11: torch.triu(arange(2*2*3*3).reshape(2,2,3,3), 0).flatten()
    let expected: Vec<f32> = vec![
        0.0, 1.0, 2.0, 0.0, 4.0, 5.0, 0.0, 0.0, 8.0, // batch [0,0]
        9.0, 10.0, 11.0, 0.0, 13.0, 14.0, 0.0, 0.0, 17.0, // batch [0,1]
        18.0, 19.0, 20.0, 0.0, 22.0, 23.0, 0.0, 0.0, 26.0, // batch [1,0]
        27.0, 28.0, 29.0, 0.0, 31.0, 32.0, 0.0, 0.0, 35.0, // batch [1,1]
    ];

    let out = triu(&gpu, 0).expect("torch accepts 4-D triu; ferrotorch must too");
    assert!(out.is_cuda(), "batched triu must stay GPU-resident");
    assert_eq!(out.shape(), &[2, 2, 3, 3]);
    assert_eq!(out.cpu().unwrap().data().unwrap().to_vec(), expected);
}

/// Divergence: ferrotorch's `tril` diverges from
/// `pytorch aten/src/ATen/native/TriangularOps.cpp:25` (`dim() >= 2`) for a
/// batched 3-D `[2,3,3]` CUDA input.
/// Upstream applies the lower-triangular mask per `[3,3]` matrix; ferrotorch
/// returns `Err(InvalidArgument)` at `ops/tensor_ops.rs:120`.
/// Tracking: #1644
#[test]
#[ignore = "divergence: tril rejects N-D batched input torch handles; tracking #1644"]
fn divergence_tril_batched_3d_on_cuda() {
    ensure_cuda();
    let cpu = arange(vec![2, 3, 3]);
    let gpu = cpu.to(Device::Cuda(0)).expect("to cuda");

    // LIVE torch 2.11: torch.tril(arange(18).reshape(2,3,3), 0).flatten()
    let expected: Vec<f32> = vec![
        0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 6.0, 7.0, 8.0, // batch 0
        9.0, 0.0, 0.0, 12.0, 13.0, 0.0, 15.0, 16.0, 17.0, // batch 1
    ];

    let out = tril(&gpu, 0).expect("torch accepts 3-D tril; ferrotorch must too");
    assert!(out.is_cuda());
    assert_eq!(out.shape(), &[2, 3, 3]);
    assert_eq!(out.cpu().unwrap().data().unwrap().to_vec(), expected);
}

// ===========================================================================
// 2. k-BEYOND-EXTENT + both rectangular orientations (POSITIVE — expect PASS)
//    Expected vectors from LIVE torch 2.11.
// ===========================================================================

#[test]
fn triu_tril_k_beyond_extent_3x5_on_cuda_f32() {
    ensure_cuda();
    let cpu = arange(vec![3, 5]);
    let gpu = cpu.to(Device::Cuda(0)).expect("to cuda");

    // torch.triu(arange(15).reshape(3,5), 10) -> all zero
    let triu_k10 = triu(&gpu, 10).expect("triu k=10");
    assert!(triu_k10.is_cuda());
    assert_eq!(
        triu_k10.cpu().unwrap().data().unwrap().to_vec(),
        vec![0.0f32; 15]
    );

    // torch.triu(arange(15).reshape(3,5), -10) -> full copy
    let triu_km10 = triu(&gpu, -10).expect("triu k=-10");
    assert_eq!(
        triu_km10.cpu().unwrap().data().unwrap().to_vec(),
        (0..15).map(|i| i as f32).collect::<Vec<_>>()
    );

    // torch.tril(arange(15).reshape(3,5), 10) -> full copy
    let tril_k10 = tril(&gpu, 10).expect("tril k=10");
    assert_eq!(
        tril_k10.cpu().unwrap().data().unwrap().to_vec(),
        (0..15).map(|i| i as f32).collect::<Vec<_>>()
    );

    // torch.tril(arange(15).reshape(3,5), -10) -> all zero
    let tril_km10 = tril(&gpu, -10).expect("tril k=-10");
    assert_eq!(
        tril_km10.cpu().unwrap().data().unwrap().to_vec(),
        vec![0.0f32; 15]
    );
}

#[test]
fn triu_tril_5x3_on_cuda_f32() {
    ensure_cuda();
    let cpu = arange(vec![5, 3]);
    let gpu = cpu.to(Device::Cuda(0)).expect("to cuda");

    // torch.triu(arange(15).reshape(5,3), 0)
    let triu0 = triu(&gpu, 0).expect("triu");
    assert!(triu0.is_cuda());
    assert_eq!(
        triu0.cpu().unwrap().data().unwrap().to_vec(),
        vec![0.0f32, 1.0, 2.0, 0.0, 4.0, 5.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    );

    // torch.tril(arange(15).reshape(5,3), 0)
    let tril0 = tril(&gpu, 0).expect("tril");
    assert_eq!(
        tril0.cpu().unwrap().data().unwrap().to_vec(),
        vec![0.0f32, 0.0, 0.0, 3.0, 4.0, 0.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
    );
}

#[test]
fn triu_k_beyond_extent_4x4_on_cuda_f64() {
    ensure_cuda();
    let cpu = arange_f64(vec![4, 4]);
    let gpu = cpu.to(Device::Cuda(0)).expect("to cuda");

    // torch.triu(arange(16,dtype=f64).reshape(4,4), 5) -> all zero (extent 3)
    let out = triu(&gpu, 5).expect("triu k=5 f64");
    assert!(out.is_cuda());
    assert_eq!(out.cpu().unwrap().data().unwrap().to_vec(), vec![0.0f64; 16]);

    // torch.triu(..., -5) -> full copy (extent -3)
    let out2 = triu(&gpu, -5).expect("triu k=-5 f64");
    assert_eq!(
        out2.cpu().unwrap().data().unwrap().to_vec(),
        (0..16).map(|i| i as f64).collect::<Vec<_>>()
    );
}
