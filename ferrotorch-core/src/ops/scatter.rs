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
//! Autograd-aware for `src`: the VJP is the same gather rule PyTorch uses for
//! `scatter_add` (`tools/autograd/derivatives.yaml:1519-1523`):
//! `grad_src[e, :] = grad_out[index[e], :]`. Segment ids are integer routing
//! metadata and are non-differentiable. CUDA backward uploads the saved host
//! segment index once and gathers rows from the resident `grad_out` buffer with
//! `GpuBackend::index_select_intidx`; tensor values do not round-trip through
//! the CPU.
//!
//! ## REQ status (per `.design/ferrotorch-core/ops/scatter.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | `scatter_add_segments` at `ops/scatter.rs:74`; consumer: re-export `ferrotorch_core::scatter_add_segments` at `lib.rs:175`; downstream `ferrotorch-graph::MessagePassing` per `ferrotorch-graph/README.md:26` |
//! | REQ-2 | SHIPPED | shape validation at `ops/scatter.rs:84-99`; consumer: `scatter_add_segments` entry |
//! | REQ-3 | SHIPPED | per-edge validation at `ops/scatter.rs:107-119`; consumer: `scatter_add_segments` entry |
//! | REQ-4 | SHIPPED | zero-init `out` at `ops/scatter.rs:101-102`; consumer: `scatter_add_segments` |
//! | REQ-5 | SHIPPED | CPU row-loop + CUDA `scatter_add_segments_cuda` (host-`&[i64]` index upload, atomic GPU kernel via `GpuBackend::scatter_add_segments_f{32,64,16,bf16}`, GPU-resident result). Consumer: the `is_cuda()` branch of `scatter_add_segments`. GPU lowering landed #1545 / sub #1535 |
//! | REQ-6 | SHIPPED | module `//!` at `ops/scatter.rs:24-30`; consumer: `ferrotorch-graph` inference harness under `no_grad` |
//! | REQ-7 | SHIPPED | `ScatterAddSegmentsBackward` attaches when `src` tracks grads; CPU and CUDA VJP gather rows with saved segment ids (`grad_src[e, :] = grad_out[index[e], :]`), matching PyTorch `scatter_add` autograd. |

use std::sync::Arc;

use crate::autograd::no_grad::{is_grad_enabled, no_grad};
use crate::dtype::{DType, Float};
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::gpu_dispatch::GpuBufferHandle;
use crate::storage::TensorStorage;
use crate::tensor::{GradFn, Tensor};

#[derive(Debug)]
struct ScatterAddSegmentsBackward<T: Float> {
    src: Tensor<T>,
    index: Vec<usize>,
    dim_size: usize,
    d: usize,
}

impl<T: Float> GradFn<T> for ScatterAddSegmentsBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !is_grad_enabled() {
            return Ok(vec![None]);
        }

        let expected_shape = [self.dim_size, self.d];
        if grad_output.shape() != expected_shape {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "ScatterAddSegmentsBackward: grad_output shape {:?} != expected {:?}",
                    grad_output.shape(),
                    expected_shape
                ),
            });
        }
        if grad_output.device() != self.src.device() {
            return Err(FerrotorchError::DeviceMismatch {
                expected: self.src.device(),
                got: grad_output.device(),
            });
        }

        let e = self.index.len();
        if grad_output.is_cuda() {
            let ordinal = match grad_output.device() {
                crate::device::Device::Cuda(ordinal) => ordinal,
                got => {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!(
                            "ScatterAddSegmentsBackward: expected CUDA grad_output, got {got:?}"
                        ),
                    });
                }
            };
            let backend =
                crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            if e == 0 || self.d == 0 {
                let h = backend.alloc_zeros(0, T::dtype(), ordinal)?;
                let grad_src =
                    Tensor::from_storage(TensorStorage::gpu(h), self.src.shape().to_vec(), false)?;
                return Ok(vec![Some(grad_src)]);
            }
            let grad_output = no_grad(|| grad_output.contiguous())?;
            let idx_handle = crate::ops::indexing::upload_index_i64(&self.index, ordinal)?;
            let h = backend.index_select_intidx(
                grad_output.gpu_handle()?,
                &idx_handle,
                1,
                self.dim_size,
                e,
                self.d,
            )?;
            let grad_src =
                Tensor::from_storage(TensorStorage::gpu(h), self.src.shape().to_vec(), false)?;
            return Ok(vec![Some(grad_src)]);
        }

        let go = grad_output.data_vec()?;
        let mut grad_src = vec![<T as num_traits::Zero>::zero(); e * self.d];
        for (edge, &segment) in self.index.iter().enumerate() {
            let src_row = edge * self.d;
            let out_row = segment * self.d;
            grad_src[src_row..src_row + self.d].copy_from_slice(&go[out_row..out_row + self.d]);
        }
        let grad_src = Tensor::from_storage(
            TensorStorage::cpu(grad_src),
            self.src.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_src)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.src]
    }

    fn name(&self) -> &'static str {
        "ScatterAddSegmentsBackward"
    }
}

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
/// * `NotImplementedOnCuda` if `src` is an unsupported CUDA dtype. f32, f64,
///   f16, and bf16 CUDA `src` run on the GPU.
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

    // Per-edge segment-id validation (shared by the CPU and CUDA paths). The
    // CUDA kernel does NO device-side bounds check (a host round trip to
    // validate would defeat the no-CPU contract for the data buffers), so the
    // host must reject negative / out-of-range segment ids before the index is
    // uploaded — exactly as the CPU loop does below.
    let mut index_usize = Vec::with_capacity(index.len());
    for (e_idx, &dst_i64) in index.iter().enumerate() {
        if dst_i64 < 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("scatter_add_segments: index[{e_idx}] = {dst_i64} is negative"),
            });
        }
        if dst_i64 as usize >= dim_size {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "scatter_add_segments: index[{e_idx}] = {dst_i64} >= dim_size {dim_size}"
                ),
            });
        }
        index_usize.push(dst_i64 as usize);
    }

    let result = if src.is_cuda() {
        scatter_add_segments_cuda(src, &index_usize, e, d, dim_size)?
    } else {
        let zero = <T as num_traits::Zero>::zero();
        let mut out = vec![zero; dim_size * d];

        let src_data = src.data_vec()?;

        for (e_idx, &dst) in index_usize.iter().enumerate() {
            let src_row = &src_data[e_idx * d..(e_idx + 1) * d];
            let out_row = &mut out[dst * d..(dst + 1) * d];
            for (o, &v) in out_row.iter_mut().zip(src_row.iter()) {
                *o += v;
            }
        }

        Tensor::from_storage(TensorStorage::cpu(out), vec![dim_size, d], false)?
    };

    if is_grad_enabled() && src.requires_grad() {
        let grad_fn = Arc::new(ScatterAddSegmentsBackward {
            src: src.clone(),
            index: index_usize,
            dim_size,
            d,
        });
        let (storage, shape) = result.into_storage_and_shape()?;
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Ok(result)
    }
}

/// CUDA lowering of [`scatter_add_segments`] (crosslink #1545 / sub #1535).
///
/// `src` is a CUDA `[e, d]` tensor; `index` is the host `&[i64]` per-row
/// segment id (already validated in-range by the caller). The result stays
/// GPU-resident: `src` is materialised contiguous ON-DEVICE, the host index is
/// uploaded once to a resident `i64` buffer, and the segmented atomic
/// row-scatter-add runs on the device. f16/bf16 use a CAS-based half-word
/// atomic add with f32 accumulation and round back to the storage dtype.
fn scatter_add_segments_cuda<T: Float>(
    src: &Tensor<T>,
    index: &[usize],
    e: usize,
    d: usize,
    dim_size: usize,
) -> FerrotorchResult<Tensor<T>> {
    // The kernel reads `src` as C-contiguous `[e, d]`; materialise on-device
    // (strided_copy — no host round trip) so a transposed/permuted view's
    // physical buffer matches the logical `[e, d]` shape.
    let src = src.contiguous()?;
    let src_handle = src.gpu_handle()?;
    let ordinal = src_handle.device_ordinal();

    let backend = crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;

    // Upload the host segment index once to a resident `i64` buffer.
    // This is uploading a freshly-provided host INPUT (the index is not a CUDA
    // tensor), not a forbidden host round trip of device data.
    let idx_handle = upload_segment_index_i64(index, ordinal)?;

    let h = match T::dtype() {
        DType::F32 => backend.scatter_add_segments_f32(src_handle, &idx_handle, e, d, dim_size)?,
        DType::F64 => backend.scatter_add_segments_f64(src_handle, &idx_handle, e, d, dim_size)?,
        DType::F16 => backend.scatter_add_segments_f16(src_handle, &idx_handle, e, d, dim_size)?,
        DType::BF16 => {
            backend.scatter_add_segments_bf16(src_handle, &idx_handle, e, d, dim_size)?
        }
        _ => {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "scatter_add_segments",
            });
        }
    };

    Tensor::from_storage(TensorStorage::gpu(h), vec![dim_size, d], false)
}

fn upload_segment_index_i64(index: &[usize], ordinal: usize) -> FerrotorchResult<GpuBufferHandle> {
    crate::ops::indexing::upload_index_i64(index, ordinal)
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
