//! Red regression tests for CORE-152 (#1846) — CLASS-V, Medium.
//!
//! `chunk_t` builds its split-size list with a `while remaining > 0`
//! loop, so when the chunked dimension has size zero the loop never
//! executes and `chunk` returns an EMPTY vector. ATen special-cases
//! this and returns `chunks` zero-sized tensors — PyTorch guarantees
//! `chunk` never returns an empty tuple
//! (`aten/src/ATen/native/TensorShape.cpp` `chunk`: `split_size == 0 &&
//! dim_size == 0` → `split_with_sizes(vec![0; chunks])`). Translated
//! code that destructures (`q, k, v = x.chunk(3, -1)`) breaks on empty
//! batches.
//!
//! Oracle (R-ORACLE-1(b)): live torch 2.11.0+cu130, 2026-06-11:
//!
//! ```python
//! [t.shape for t in torch.empty(0, 3).chunk(3, 0)]
//! # [torch.Size([0, 3]), torch.Size([0, 3]), torch.Size([0, 3])]
//! [t.shape for t in torch.empty(2, 0).chunk(3, 1)]
//! # [torch.Size([2, 0]), torch.Size([2, 0]), torch.Size([2, 0])]
//! [t.shape for t in torch.empty(0, 3).split([0, 0, 0], dim=0)]
//! # [torch.Size([0, 3]), torch.Size([0, 3]), torch.Size([0, 3])]
//! ```

use ferrotorch_core::Tensor;
use ferrotorch_core::creation::from_vec;

fn empty_2d(rows: usize, cols: usize) -> Tensor<f32> {
    from_vec(Vec::<f32>::new(), &[rows, cols]).expect("construct zero-sized tensor")
}

/// `torch.empty(0, 3).chunk(3, 0)` -> three `(0, 3)` tensors.
/// Pre-fix ferrotorch returns ZERO chunks.
#[test]
fn chunk_over_zero_sized_dim_returns_chunks_empty_tensors() {
    let x = empty_2d(0, 3);
    let chunks = x.chunk(3, 0).expect("chunk over empty dim");
    assert_eq!(
        chunks.len(),
        3,
        "torch.empty(0, 3).chunk(3, 0) returns 3 tensors — chunk never \
         returns an empty tuple (CORE-152)"
    );
    for (i, c) in chunks.iter().enumerate() {
        assert_eq!(c.shape(), &[0, 3], "chunk {i} shape");
        assert_eq!(c.numel(), 0, "chunk {i} numel");
    }
}

/// Same special case along a non-leading dimension:
/// `torch.empty(2, 0).chunk(3, 1)` -> three `(2, 0)` tensors.
#[test]
fn chunk_over_zero_sized_inner_dim_returns_chunks_empty_tensors() {
    let x = empty_2d(2, 0);
    let chunks = x.chunk(3, 1).expect("chunk over empty inner dim");
    assert_eq!(
        chunks.len(),
        3,
        "torch.empty(2, 0).chunk(3, 1) returns 3 tensors"
    );
    for (i, c) in chunks.iter().enumerate() {
        assert_eq!(c.shape(), &[2, 0], "chunk {i} shape");
        assert_eq!(c.numel(), 0, "chunk {i} numel");
    }
}

/// The `split` primitive itself already accepts explicit zero sizes —
/// `torch.empty(0, 3).split([0, 0, 0], 0)` -> three `(0, 3)` tensors.
/// Pinned here so the chunk fix can lower onto `split_t` unchanged.
#[test]
fn split_with_explicit_zero_sizes_matches_torch() {
    let x = empty_2d(0, 3);
    let parts = x.split(&[0, 0, 0], 0).expect("split with zero sizes");
    assert_eq!(parts.len(), 3);
    for (i, p) in parts.iter().enumerate() {
        assert_eq!(p.shape(), &[0, 3], "part {i} shape");
    }
}

/// GPU lane (device-asserting per the post-#1890 pattern): the same
/// ATen special case on a CUDA f32 tensor — three `(0, 3)` chunks that
/// stay on CUDA.
#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use ferrotorch_core::Device;
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for the CORE-152 GPU lane");
        });
    }

    #[test]
    fn gpu_chunk_over_zero_sized_dim_returns_chunks_empty_cuda_tensors() {
        ensure_cuda_backend();
        let x = empty_2d(0, 3)
            .to(Device::Cuda(0))
            .expect("cpu->gpu upload of zero-sized tensor");
        let chunks = x.chunk(3, 0).expect("chunk over empty dim on CUDA");
        assert_eq!(
            chunks.len(),
            3,
            "CUDA chunk over empty dim returns 3 tensors"
        );
        for (i, c) in chunks.iter().enumerate() {
            assert!(c.is_cuda(), "chunk {i} must stay on CUDA");
            assert_eq!(c.shape(), &[0, 3], "chunk {i} shape");
            assert_eq!(c.numel(), 0, "chunk {i} numel");
        }
    }
}
