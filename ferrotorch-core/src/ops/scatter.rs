//! Segmented scatter-add — the message-passing primitive used by graph
//! neural networks.
//!
//! `scatter_add_segments(src, index, dim_size)` produces an output tensor
//! `out` of shape `[dim_size, D]` where
//!
//! ```text
//! out[i, :] = sum over { e : index[e] == i } of src[e, :]
//! ```
//!
//! This is the same operation `torch_scatter.scatter_add(src, index,
//! dim=0, dim_size=N)` performs, and is the primitive that
//! `torch_geometric.nn.MessagePassing.aggregate(...)` calls into for the
//! default `aggr="add"` aggregation.
//!
//! The existing `ops::indexing::scatter_add(input, dim, index, ..., src)`
//! in this crate is a different operator: it does per-element scatter
//! along an arbitrary axis with the same shape on `input` and `src` and
//! returns a tensor the shape of `input`. The graph-side aggregation has
//! a different signature (a 1-D `index` over `E` edges that maps into a
//! pre-decided segment count `dim_size`) and is significantly simpler to
//! reason about, so we keep it as a separate, narrower primitive.
//!
//! # Autograd
//!
//! Forward only — the GCN inference harness in `ferrotorch-graph` runs
//! under `no_grad`. The grad of a segmented scatter-add is a simple
//! `gather` (`grad_src[e, :] = grad_out[index[e], :]`), which can be
//! added in a follow-up if/when an autograd-based GCN training path is
//! needed.
//!
//! ## REQ status (per `.design/ferrotorch-core/ops/scatter.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | `scatter_add_segments` at `ops/scatter.rs:74`; consumer: re-export `ferrotorch_core::scatter_add_segments` at `lib.rs:175`; downstream `ferrotorch-graph::MessagePassing` per `ferrotorch-graph/README.md:26` |
//! | REQ-2 | SHIPPED | shape validation at `ops/scatter.rs:84-99`; consumer: `scatter_add_segments` entry |
//! | REQ-3 | SHIPPED | per-edge validation at `ops/scatter.rs:107-119`; consumer: `scatter_add_segments` entry |
//! | REQ-4 | SHIPPED | zero-init `out` at `ops/scatter.rs:101-102`; consumer: `scatter_add_segments` |
//! | REQ-5 | SHIPPED | `NotImplementedOnCuda` at `ops/scatter.rs:79-83`; consumer: `scatter_add_segments`. GPU blocker #1535 |
//! | REQ-6 | SHIPPED | module `//!` at `ops/scatter.rs:24-30`; consumer: `ferrotorch-graph` inference harness under `no_grad` |

use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::storage::TensorStorage;
use crate::tensor::Tensor;

/// Segmented scatter-add of a `[E, D]` source into an `[dim_size, D]`
/// output, indexed along dim 0 by `index[e]`.
///
/// # Shape
///
/// * `src` — `[E, D]`. The values to scatter.
/// * `index` — flat `&[i64]` of length `E`. Each entry names a row of
///   the output to accumulate into.
/// * `dim_size` — number of output rows (`>= max(index) + 1`).
///
/// # Output
///
/// Tensor of shape `[dim_size, D]`. Rows with no incoming edges are
/// zero.
///
/// # Errors
///
/// * `ShapeMismatch` if `src` is not 2-D, or if `index.len() != src.shape()[0]`.
/// * `InvalidArgument` if any `index[e]` is negative or `>= dim_size`.
/// * `NotImplementedOnCuda` if `src` is on CUDA.
///
/// # Example
///
/// ```ignore
/// use ferrotorch_core::{Tensor, TensorStorage};
/// use ferrotorch_core::ops::scatter::scatter_add_segments;
///
/// // 3 edges, feature dim 2, output rows = 2.
/// // edge 0: 1.0,2.0 -> out[0]; edge 1: 3.0,4.0 -> out[1]; edge 2: 5.0,6.0 -> out[0]
/// let src = Tensor::<f32>::from_storage(
///     TensorStorage::cpu(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
///     vec![3, 2],
///     false,
/// ).unwrap();
/// let out = scatter_add_segments(&src, &[0, 1, 0], 2).unwrap();
/// // out == [[6.0, 8.0], [3.0, 4.0]]
/// ```
pub fn scatter_add_segments<T: Float>(
    src: &Tensor<T>,
    index: &[i64],
    dim_size: usize,
) -> FerrotorchResult<Tensor<T>> {
    if src.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda {
            op: "scatter_add_segments",
        });
    }
    let shape = src.shape();
    if shape.len() != 2 {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!("scatter_add_segments: src must be 2-D [E, D], got shape {shape:?}"),
        });
    }
    let e = shape[0];
    let d = shape[1];
    if index.len() != e {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "scatter_add_segments: index length {} != src.shape()[0] {e}",
                index.len()
            ),
        });
    }

    let zero = <T as num_traits::Zero>::zero();
    let mut out = vec![zero; dim_size * d];

    let src_data = src.data_vec()?;

    for (e_idx, &dst_i64) in index.iter().enumerate() {
        if dst_i64 < 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("scatter_add_segments: index[{e_idx}] = {dst_i64} is negative"),
            });
        }
        let dst = dst_i64 as usize;
        if dst >= dim_size {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "scatter_add_segments: index[{e_idx}] = {dst} >= dim_size {dim_size}"
                ),
            });
        }
        let src_row = &src_data[e_idx * d..(e_idx + 1) * d];
        let out_row = &mut out[dst * d..(dst + 1) * d];
        for (o, &v) in out_row.iter_mut().zip(src_row.iter()) {
            *o += v;
        }
    }

    Tensor::from_storage(TensorStorage::cpu(out), vec![dim_size, d], false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
    }

    #[test]
    fn segments_basic_aggregation() {
        // 3 rows of D=2 features mapped onto 2 segments.
        // index = [0, 1, 0] -> out[0] = src[0] + src[2], out[1] = src[1].
        let src = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let out = scatter_add_segments(&src, &[0, 1, 0], 2).unwrap();
        assert_eq!(out.shape(), &[2, 2]);
        let data = out.data().unwrap();
        assert!((data[0] - 6.0).abs() < 1e-6);
        assert!((data[1] - 8.0).abs() < 1e-6);
        assert!((data[2] - 3.0).abs() < 1e-6);
        assert!((data[3] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn segments_empty_rows_are_zero() {
        // No edge targets row 1; it should stay zero.
        let src = t(&[7.0, 0.5, 8.0, 0.25], &[2, 2]);
        let out = scatter_add_segments(&src, &[0, 0], 3).unwrap();
        assert_eq!(out.shape(), &[3, 2]);
        let data = out.data().unwrap();
        // Row 0: 7+8, 0.5+0.25 = 15.0, 0.75
        assert!((data[0] - 15.0).abs() < 1e-6);
        assert!((data[1] - 0.75).abs() < 1e-6);
        // Row 1 and 2: zero. The unwritten output rows come straight
        // from `vec![T::zero(); ...]` with no arithmetic applied, so a
        // bitwise-magnitude compare is the right tightness here.
        for &v in &data[2..] {
            assert!(v.abs() < 1e-12, "expected exact zero, got {v}");
        }
    }

    #[test]
    fn segments_single_edge_per_segment() {
        // Identity-like permutation.
        let src = t(&[1.0, 1.5, 2.0, 2.5, 3.0, 3.5], &[3, 2]);
        let out = scatter_add_segments(&src, &[2, 0, 1], 3).unwrap();
        let data = out.data().unwrap();
        // out[0] = src[1], out[1] = src[2], out[2] = src[0]
        assert!((data[0] - 2.0).abs() < 1e-6);
        assert!((data[1] - 2.5).abs() < 1e-6);
        assert!((data[2] - 3.0).abs() < 1e-6);
        assert!((data[3] - 3.5).abs() < 1e-6);
        assert!((data[4] - 1.0).abs() < 1e-6);
        assert!((data[5] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn segments_rejects_non_2d_src() {
        let src = t(&[1.0, 2.0, 3.0], &[3]);
        assert!(scatter_add_segments(&src, &[0, 1, 0], 2).is_err());
    }

    #[test]
    fn segments_rejects_index_length_mismatch() {
        let src = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        // E=2 but index has 3 entries.
        assert!(scatter_add_segments(&src, &[0, 1, 0], 2).is_err());
    }

    #[test]
    fn segments_rejects_negative_index() {
        let src = t(&[1.0, 2.0], &[1, 2]);
        assert!(scatter_add_segments(&src, &[-1], 2).is_err());
    }

    #[test]
    fn segments_rejects_oob_index() {
        let src = t(&[1.0, 2.0], &[1, 2]);
        // dim_size = 2 so index must be in [0, 1].
        assert!(scatter_add_segments(&src, &[2], 2).is_err());
    }
}
