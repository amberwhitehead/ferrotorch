//! Discriminator re-audit of commit `67057b8f6` — batched N-D triu/tril (#1644).
//!
//! The #1644 fix made triu/tril apply the triangular mask to the LAST TWO DIMS
//! of every trailing `[rows, cols]` matrix, batching over all leading dims, on
//! BOTH the CPU path (`ferrotorch-core/src/ops/tensor_ops.rs`) and the GPU PTX
//! kernel (`ferrotorch-gpu/src/triangular.rs`, `col = t % cols`,
//! `row = (t / cols) % rows`).
//!
//! The builder's own tests only exercise SQUARE trailing dims (`[2,2,3,3]`,
//! `[2,3,3]`). A batch-indexing bug — row not resetting per matrix, or a wrong
//! batch stride — would NOT show on square trailing dims but WOULD show on
//! NON-SQUARE trailing dims (`[2,3,5]`, `[2,5,3]`) where `rows != cols` means
//! `b*rows*cols + r*cols + c` and the `(t/cols)%rows` arithmetic must use the
//! correct extents. This file pins the non-square + f64 + k-extent batched
//! cases and cross-checks CPU == GPU == torch.
//!
//! Upstream: `pytorch/aten/src/ATen/native/TriangularOps.cpp:30`
//!   `TORCH_CHECK(self.dim() >= 2, "triu: input tensor must have at least 2 dimensions")`
//! and the CUDA batching template
//! `pytorch/aten/src/ATen/native/cuda/TriangularOps.cu:120`
//!   `int64_t N_padded = c10::multiply_integers(sizes.begin(), sizes.end()-1) * last_dim_padded;`
//! predicate `cuda/TriangularOps.cu:100`
//!   `bool mask = upper ? (col + i - row >= k) : (col + i - row <= k);`
//!
//! All expected vectors were produced by LIVE `torch.triu`/`torch.tril`
//! (torch 2.11.0+cu130) — NOT copied from the ferrotorch side (R-CHAR-3).

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

fn arange_f32(shape: Vec<usize>) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    Tensor::from_storage(TensorStorage::cpu(data), shape, false).expect("cpu tensor")
}

fn arange_f64(shape: Vec<usize>) -> Tensor<f64> {
    let n: usize = shape.iter().product();
    let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
    Tensor::from_storage(TensorStorage::cpu(data), shape, false).expect("cpu tensor")
}

/// Assert CPU == GPU == torch-gold for one (op, shape, k).
/// `op_triu` selects triu (true) or tril (false).
fn check_f32(shape: Vec<usize>, k: i64, op_triu: bool, gold: &[f32]) {
    ensure_cuda();
    let cpu_in = arange_f32(shape.clone());
    let gpu_in = cpu_in.to(Device::Cuda(0)).expect("to cuda");

    let cpu_out = if op_triu {
        triu(&cpu_in, k)
    } else {
        tril(&cpu_in, k)
    }
    .expect("cpu triu/tril");
    let gpu_out = if op_triu {
        triu(&gpu_in, k)
    } else {
        tril(&gpu_in, k)
    }
    .expect("gpu triu/tril");

    assert!(gpu_out.is_cuda(), "batched GPU result must stay resident");
    assert_eq!(cpu_out.shape(), shape.as_slice(), "CPU output shape");
    assert_eq!(gpu_out.shape(), shape.as_slice(), "GPU output shape");

    let cpu_vec = cpu_out.data().unwrap().to_vec();
    let gpu_vec = gpu_out.cpu().unwrap().data().unwrap().to_vec();

    assert_eq!(cpu_vec, gold, "CPU != torch for shape {shape:?} k={k}");
    assert_eq!(gpu_vec, gold, "GPU != torch for shape {shape:?} k={k}");
    assert_eq!(cpu_vec, gpu_vec, "CPU != GPU for shape {shape:?} k={k}");
}

// ===========================================================================
// Non-square trailing [2,3,5]: rows=3, cols=5 (cols > rows)
// ===========================================================================

#[test]
fn batched_nonsquare_2x3x5_triu_f32() {
    // torch.triu(arange(30).reshape(2,3,5), k)
    check_f32(
        vec![2, 3, 5],
        -1,
        true,
        &[
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 0.0, 26.0, 27.0, 28.0, 29.0,
        ],
    );
    check_f32(
        vec![2, 3, 5],
        0,
        true,
        &[
            0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 6.0, 7.0, 8.0, 9.0, 0.0, 0.0, 12.0, 13.0, 14.0, 15.0,
            16.0, 17.0, 18.0, 19.0, 0.0, 21.0, 22.0, 23.0, 24.0, 0.0, 0.0, 27.0, 28.0, 29.0,
        ],
    );
    check_f32(
        vec![2, 3, 5],
        1,
        true,
        &[
            0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 7.0, 8.0, 9.0, 0.0, 0.0, 0.0, 13.0, 14.0, 0.0, 16.0,
            17.0, 18.0, 19.0, 0.0, 0.0, 22.0, 23.0, 24.0, 0.0, 0.0, 0.0, 28.0, 29.0,
        ],
    );
}

#[test]
fn batched_nonsquare_2x3x5_tril_f32() {
    // torch.tril(arange(30).reshape(2,3,5), k)
    check_f32(
        vec![2, 3, 5],
        -1,
        false,
        &[
            0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 10.0, 11.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 25.0, 26.0, 0.0, 0.0, 0.0,
        ],
    );
    check_f32(
        vec![2, 3, 5],
        0,
        false,
        &[
            0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 6.0, 0.0, 0.0, 0.0, 10.0, 11.0, 12.0, 0.0, 0.0, 15.0,
            0.0, 0.0, 0.0, 0.0, 20.0, 21.0, 0.0, 0.0, 0.0, 25.0, 26.0, 27.0, 0.0, 0.0,
        ],
    );
    check_f32(
        vec![2, 3, 5],
        1,
        false,
        &[
            0.0, 1.0, 0.0, 0.0, 0.0, 5.0, 6.0, 7.0, 0.0, 0.0, 10.0, 11.0, 12.0, 13.0, 0.0, 15.0,
            16.0, 0.0, 0.0, 0.0, 20.0, 21.0, 22.0, 0.0, 0.0, 25.0, 26.0, 27.0, 28.0, 0.0,
        ],
    );
}

// ===========================================================================
// Non-square trailing [2,5,3]: rows=5, cols=3 (rows > cols) — the orientation
// most likely to expose a row-reset / batch-stride bug, since
// (t/cols)%rows must wrap at rows=5 while cols=3.
// ===========================================================================

#[test]
fn batched_nonsquare_2x5x3_triu_f32() {
    // torch.triu(arange(30).reshape(2,5,3), k)
    check_f32(
        vec![2, 5, 3],
        -1,
        true,
        &[
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 7.0, 8.0, 0.0, 0.0, 11.0, 0.0, 0.0, 0.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 0.0, 22.0, 23.0, 0.0, 0.0, 26.0, 0.0, 0.0, 0.0,
        ],
    );
    check_f32(
        vec![2, 5, 3],
        0,
        true,
        &[
            0.0, 1.0, 2.0, 0.0, 4.0, 5.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.0, 16.0,
            17.0, 0.0, 19.0, 20.0, 0.0, 0.0, 23.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
    );
    check_f32(
        vec![2, 5, 3],
        1,
        true,
        &[
            0.0, 1.0, 2.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 16.0,
            17.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
    );
}

#[test]
fn batched_nonsquare_2x5x3_tril_f32() {
    // torch.tril(arange(30).reshape(2,5,3), k)
    check_f32(
        vec![2, 5, 3],
        -1,
        false,
        &[
            0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 6.0, 7.0, 0.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 0.0,
            0.0, 0.0, 18.0, 0.0, 0.0, 21.0, 22.0, 0.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0,
        ],
    );
    check_f32(
        vec![2, 5, 3],
        0,
        false,
        &[
            0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            0.0, 0.0, 18.0, 19.0, 0.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0,
        ],
    );
    check_f32(
        vec![2, 5, 3],
        1,
        false,
        &[
            0.0, 1.0, 0.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0, 0.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0,
        ],
    );
}

// ===========================================================================
// f64 batched non-square [2,3,5] k=0 — exercises the f64 PTX kernel batching.
// ===========================================================================

#[test]
fn batched_nonsquare_2x3x5_f64() {
    ensure_cuda();
    let cpu = arange_f64(vec![2, 3, 5]);
    let gpu = cpu.to(Device::Cuda(0)).expect("to cuda");

    // torch.triu(arange(30,dtype=f64).reshape(2,3,5), 0)
    let triu_gold: Vec<f64> = vec![
        0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 6.0, 7.0, 8.0, 9.0, 0.0, 0.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        17.0, 18.0, 19.0, 0.0, 21.0, 22.0, 23.0, 24.0, 0.0, 0.0, 27.0, 28.0, 29.0,
    ];
    // torch.tril(arange(30,dtype=f64).reshape(2,3,5), 0)
    let tril_gold: Vec<f64> = vec![
        0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 6.0, 0.0, 0.0, 0.0, 10.0, 11.0, 12.0, 0.0, 0.0, 15.0, 0.0,
        0.0, 0.0, 0.0, 20.0, 21.0, 0.0, 0.0, 0.0, 25.0, 26.0, 27.0, 0.0, 0.0,
    ];

    let cpu_u = triu(&cpu, 0).unwrap();
    let gpu_u = triu(&gpu, 0).unwrap();
    assert!(gpu_u.is_cuda());
    assert_eq!(cpu_u.data().unwrap().to_vec(), triu_gold);
    assert_eq!(gpu_u.cpu().unwrap().data().unwrap().to_vec(), triu_gold);

    let cpu_l = tril(&cpu, 0).unwrap();
    let gpu_l = tril(&gpu, 0).unwrap();
    assert!(gpu_l.is_cuda());
    assert_eq!(cpu_l.data().unwrap().to_vec(), tril_gold);
    assert_eq!(gpu_l.cpu().unwrap().data().unwrap().to_vec(), tril_gold);
}

// ===========================================================================
// k beyond extent on a batched non-square tensor [2,3,5].
// ===========================================================================

#[test]
fn batched_nonsquare_k_beyond_extent() {
    ensure_cuda();
    let cpu = arange_f32(vec![2, 3, 5]);
    let gpu = cpu.to(Device::Cuda(0)).expect("to cuda");

    // torch.triu(arange(30).reshape(2,3,5), 10) -> all zero
    let z = triu(&gpu, 10).unwrap();
    assert_eq!(z.cpu().unwrap().data().unwrap().to_vec(), vec![0.0f32; 30]);
    assert_eq!(
        triu(&cpu, 10).unwrap().data().unwrap().to_vec(),
        vec![0.0f32; 30]
    );

    // torch.triu(arange(30).reshape(2,3,5), -10) -> full copy
    let full: Vec<f32> = (0..30).map(|i| i as f32).collect();
    let f = triu(&gpu, -10).unwrap();
    assert_eq!(f.cpu().unwrap().data().unwrap().to_vec(), full);
    assert_eq!(triu(&cpu, -10).unwrap().data().unwrap().to_vec(), full);
}

// ===========================================================================
// ndim < 2 rejection on the GPU path (matches torch RuntimeError
// "triu: input tensor must have at least 2 dimensions").
// ===========================================================================

#[test]
fn batched_reject_1d_on_cuda() {
    ensure_cuda();
    let cpu = arange_f32(vec![5]);
    let gpu = cpu.to(Device::Cuda(0)).expect("to cuda");
    assert!(triu(&gpu, 0).is_err(), "1-D triu on cuda must be Err");
    assert!(tril(&gpu, 0).is_err(), "1-D tril on cuda must be Err");
}
