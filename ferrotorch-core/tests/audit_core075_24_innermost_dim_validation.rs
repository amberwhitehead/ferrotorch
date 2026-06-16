//! Red-then-green regression tests for audit finding CORE-075 (crosslink
//! #1769): 2:4 compression validates total size instead of the innermost
//! dimension (CLASS-S — `compress` checked only `numel % 4 == 0` and then
//! grouped the FLAT buffer, so a shape like `[2, 2]` passed with its only
//! group spanning two rows; `sparse_matmul_24` also omitted its documented
//! `n % 4 == 0` check). CORE-084 / #1778 later tightened the public 2:4
//! pruning contract to match PyTorch's `WeightNormSparsifier`: non-empty
//! 2-D weights only, final dimension divisible by 4.
//!
//! Observed at HEAD (probe, 2026-06-12, rev 74099dd19):
//! - `compress([2,2])` ACCEPTED (numel 4 divisible by 4, last dim 2 not).
//! - `sparse_matmul_24(a, b)` with `b.shape() = [2, 2]` (n = 2) ACCEPTED.
//!
//! torch oracle (live session, torch 2.11.0+cu130, RTX 3090): upstream
//! 2:4 NEVER groups across rows — invalid shapes are rejected at
//! conversion:
//!
//! ```python
//! >>> torch.sparse.to_sparse_semi_structured(
//! ...     torch.tensor([[1.,2.],[3.,4.]], dtype=torch.float16, device="cuda"))
//! RuntimeError: Error original_tensor.shape torch.Size([2, 2]) is not
//! supported! Both dimensions must be larger or equal than and a multiple
//! of (32, 64)
//! ```
//!
//! ferrotorch's CPU/reference pruning-layout contract is intentionally the
//! `WeightNormSparsifier` contract, not a fake 1-D shortcut: non-empty 2-D
//! tensors with final dimension divisible by 4; `sparse_matmul_24` requires
//! `n % 4 == 0`. Post-fix both are enforced.

use ferrotorch_core::{FerrotorchError, SemiStructuredSparseTensor, Tensor, TensorStorage};

fn mk(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data), shape, false).unwrap()
}

/// `[2, 2]`: numel divisible by 4 but the only group would span two
/// rows. Must be rejected (the HEAD probe's accepted shape).
#[test]
fn core075_compress_rejects_2x2() {
    let r = SemiStructuredSparseTensor::compress(&mk(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]));
    assert!(
        matches!(r, Err(FerrotorchError::InvalidArgument { .. })),
        "[2,2] (last dim 2) must be rejected, got {r:?}"
    );
}

/// More shapes with `numel % 4 == 0` but `last_dim % 4 != 0`.
#[test]
fn core075_compress_rejects_total_divisible_but_last_dim_not() {
    for shape in [vec![4usize, 3], vec![2, 6], vec![4, 1], vec![2, 2, 2]] {
        let numel: usize = shape.iter().product();
        assert_eq!(numel % 4, 0, "test precondition");
        let r = SemiStructuredSparseTensor::compress(&mk(vec![1.0; numel], shape.clone()));
        assert!(
            matches!(r, Err(FerrotorchError::InvalidArgument { .. })),
            "shape {shape:?} (last dim not %4) must be rejected, got {r:?}"
        );
    }
}

/// Scalar (0-d) input must be rejected: there is no innermost dimension
/// to group along.
#[test]
fn core075_compress_rejects_scalar() {
    let r = SemiStructuredSparseTensor::compress(&mk(vec![1.0], vec![]));
    assert!(
        matches!(r, Err(FerrotorchError::InvalidArgument { .. })),
        "scalar must be rejected, got {r:?}"
    );
}

/// Valid 2-D shapes (last dim a multiple of 4) keep working, and groups
/// stay within rows. Hand-derived: for `[3, 4]` each row is one group
/// keeping its 2 largest magnitudes — row 1's large values (8, 7) must
/// not "win" slots in row 0's group, which a flat grouping with rows of
/// width 2 would have allowed.
#[test]
// reason: compress/decompress copy kept values verbatim and zero dropped
// positions — no arithmetic, so bitwise equality is the right assertion.
#[allow(clippy::float_cmp)]
fn core075_compress_valid_last_dim_groups_per_row() {
    let t = mk(
        vec![
            1.0, 4.0, 2.0, 3.0, // row 0: keep 4, 3
            -8.0, 7.0, 0.5, 0.1, // row 1: keep -8, 7
            0.0, 0.0, 1.0, -1.0, // row 2: keep 1, -1
        ],
        vec![3, 4],
    );
    let sp = SemiStructuredSparseTensor::compress(&t).expect("[3,4] is valid (last dim 4)");
    assert_eq!(sp.num_groups(), 3);
    let d = sp.decompress().unwrap();
    assert_eq!(
        d.data().unwrap(),
        &[0.0, 4.0, 0.0, 3.0, -8.0, 7.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0]
    );
}

/// 1-D shapes used to be accepted by a fixture generator that reshaped them
/// into `[1, n]` before calling torch. PyTorch's public sparsifier path does
/// not accept that input shape, so the Rust helper must reject it too.
#[test]
fn core075_compress_rejects_1d_even_when_divisible_by_four() {
    let r = SemiStructuredSparseTensor::compress(&mk(vec![1.0; 8], vec![8]));
    assert!(
        matches!(r, Err(FerrotorchError::InvalidArgument { .. })),
        "1-D [8] must be rejected, got {r:?}"
    );
}

#[test]
fn core075_compress_rejects_empty_2d_dimensions() {
    for shape in [vec![0usize, 4], vec![1, 0]] {
        let numel: usize = shape.iter().product();
        let r = SemiStructuredSparseTensor::compress(&mk(vec![1.0; numel], shape.clone()));
        assert!(
            matches!(r, Err(FerrotorchError::InvalidArgument { .. })),
            "empty shape {shape:?} must be rejected, got {r:?}"
        );
    }
}
