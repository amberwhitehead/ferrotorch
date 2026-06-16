//! Red-then-green regression tests for audit finding CORE-078 (crosslink
//! #1772): zero-sized sparse-gradient slabs pass validation and panic
//! during SGD (CLASS-U — `product(slab_shape).max(1)` conflates a
//! NON-EMPTY shape containing zero (`[0]`, zero elements per slab) with
//! the EMPTY scalar shape (`[]`, one element per slab)).
//!
//! Observed at HEAD (probe, 2026-06-12, rev 74099dd19):
//! - `SparseGrad::new([1], [9.0], [0])` ACCEPTED (slab_size computed as
//!   `max(0, 1) = 1`).
//! - `apply_sgd` against a `[2, 0]` param then PANICKED:
//!   "index out of bounds: the len is 0 but the index is 1"
//!   (sparse.rs:2361).
//!
//! torch oracle (live session, torch 2.11.0+cu130):
//!
//! ```python
//! >>> torch.sparse_coo_tensor(torch.tensor([[1]]), torch.tensor([[9.0]]),
//! ...                         (2, 0))   # 1 phantom value for a [_, 0] grad
//! RuntimeError: values has incorrect size, expected [1, 0], got [1, 1]
//! >>> p = torch.zeros(2, 0, requires_grad=True)
//! >>> p.grad = torch.sparse_coo_tensor(torch.tensor([[1]]),
//! ...                                  torch.zeros(1, 0), (2, 0))
//! >>> torch.optim.SGD([p], lr=0.1).step()   # no-op, no panic
//! ```
//!
//! Post-fix contract: slab size is the actual product of `slab_shape`
//! (the empty product, 1, is already the correct scalar-slab factor —
//! no `.max(1)`), so `[0]`-slab gradients require zero values; the
//! correct zero-valued form constructs, coalesces, and applies as a
//! no-op against a zero-width parameter.

use ferrotorch_core::{FerrotorchError, SparseGrad, Tensor, TensorStorage};

/// One phantom value for a `[0]` slab must be rejected at construction
/// (torch: "values has incorrect size, expected [1, 0], got [1, 1]").
#[test]
fn core078_rejects_nonempty_values_for_zero_slab() {
    let r = SparseGrad::<f32>::new(vec![1], vec![9.0], vec![0]);
    assert!(
        matches!(r, Err(FerrotorchError::ShapeMismatch { .. })),
        "1 value for slab_shape [0] (0 elements/slab) must be rejected, got {:?}",
        r.map(|g| (g.nnz(), g.values().len()))
    );
}

/// Higher-rank zero-containing slab `[3, 0]` likewise holds zero
/// elements per slab.
#[test]
fn core078_rejects_nonempty_values_for_rank2_zero_slab() {
    let r = SparseGrad::<f64>::new(vec![0, 1], vec![1.0, 2.0, 3.0], vec![3, 0]);
    assert!(
        matches!(r, Err(FerrotorchError::ShapeMismatch { .. })),
        "3 values for slab_shape [3, 0] must be rejected"
    );
}

/// The CORRECT zero-sized gradient (indices present, zero values)
/// constructs, coalesces, and applies as a no-op (torch: SGD step on a
/// [2, 0] param succeeds).
#[test]
fn core078_zero_slab_grad_applies_as_noop() {
    let g = SparseGrad::<f32>::new(vec![1, 1], vec![], vec![0])
        .expect("zero values is the CORRECT input for a [0] slab");
    assert_eq!(g.nnz(), 2);
    assert_eq!(g.slab_size(), 0, "slab [0] holds zero elements");

    let c = g.coalesce();
    assert_eq!(c.indices(), &[1], "duplicate index 1 coalesces");
    assert!(c.values().is_empty());

    let mut param =
        Tensor::<f32>::from_storage(TensorStorage::cpu(vec![]), vec![2, 0], false).unwrap();
    g.apply_sgd(&mut param, 0.1)
        .expect("zero-width update is a no-op, not a panic");
    assert_eq!(param.shape(), &[2, 0]);
    assert_eq!(param.numel(), 0);
}

/// Out-of-range indices are still validated even when the slab is
/// zero-sized (the index itself is meaningful metadata).
#[test]
fn core078_zero_slab_still_validates_indices() {
    let g = SparseGrad::<f32>::new(vec![5], vec![], vec![0]).expect("valid zero-sized grad");
    let mut param =
        Tensor::<f32>::from_storage(TensorStorage::cpu(vec![]), vec![2, 0], false).unwrap();
    let err = g.apply_sgd(&mut param, 0.1).unwrap_err();
    assert!(
        matches!(err, FerrotorchError::InvalidArgument { .. }),
        "index 5 >= leading 2 must still be rejected, got {err:?}"
    );
}

/// Scalar slabs (`slab_shape = []`, one element per index) keep their
/// pre-fix behavior: the empty product is 1, NOT 0.
#[test]
fn core078_scalar_slab_unaffected() {
    let g = SparseGrad::<f32>::new(vec![0, 2], vec![10.0, 20.0], vec![]).expect("scalar slabs");
    assert_eq!(g.slab_size(), 1);
    let mut param =
        Tensor::<f32>::from_storage(TensorStorage::cpu(vec![1.0, 2.0, 3.0]), vec![3], false)
            .unwrap();
    g.apply_sgd(&mut param, 1.0).expect("scalar-slab sgd");
    let d = param.data().unwrap();
    assert!((d[0] - (1.0 - 10.0)).abs() < 1e-6);
    assert!((d[1] - 2.0).abs() < 1e-6);
    assert!((d[2] - (3.0 - 20.0)).abs() < 1e-6);
}

/// Oversized dense slab metadata is still user input. Construction must
/// return a structured error instead of panicking through `numel`.
#[test]
fn core078_constructor_rejects_slab_numel_overflow() {
    let err = SparseGrad::<f32>::new(vec![0], vec![], vec![usize::MAX, 2]).unwrap_err();
    assert!(
        matches!(err, FerrotorchError::InvalidArgument { .. }),
        "overflowing slab_shape product must be InvalidArgument, got {err:?}"
    );
}

/// Even when the slab product itself fits, `nnz * slab_size` can overflow.
/// That must be caught before length validation so release builds never wrap.
#[test]
fn core078_constructor_rejects_expected_value_len_overflow() {
    let err = SparseGrad::<f32>::new(vec![0, 1], vec![], vec![usize::MAX]).unwrap_err();
    assert!(
        matches!(err, FerrotorchError::InvalidArgument { .. }),
        "overflowing nnz * slab_size must be InvalidArgument, got {err:?}"
    );
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for CORE-078 GPU audit")
        });
    }

    /// Zero-width sparse gradients are no-ops on CUDA too, and must not route
    /// through CPU storage or demote the parameter.
    #[test]
    fn core078_cuda_zero_slab_grad_applies_as_noop_and_stays_cuda() {
        ensure_cuda_backend();

        let mut param = Tensor::<f32>::from_storage(TensorStorage::cpu(vec![]), vec![2, 0], false)
            .expect("zero-width cpu param")
            .to(ferrotorch_core::Device::Cuda(0))
            .expect("param->cuda");
        assert!(param.is_cuda(), "setup must create a CUDA parameter");

        let g = SparseGrad::<f32>::new(vec![1, 1], vec![], vec![0])
            .expect("valid zero-width sparse grad");
        g.apply_sgd(&mut param, 0.1)
            .expect("zero-width CUDA update is a no-op");

        assert!(
            param.is_cuda(),
            "zero-width sparse SGD must keep CUDA parameters on CUDA"
        );
        assert_eq!(param.shape(), &[2, 0]);
        let host = param.cpu().expect("cuda zero-width param -> cpu");
        assert_eq!(host.numel(), 0);
    }
}
