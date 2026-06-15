//! Cross-world integer ops — crosslink #1185 Phase 2c.
//!
//! `argmax` / `argmin` (→ `IntTensor<i64>`), `index_select` / `gather` driven
//! by a GPU-resident `IntTensor` index, and dtype casts (`Tensor::to_int`,
//! `IntTensor::to_float`, `IntTensor::cast` GPU path). Each op runs on CUDA when
//! the input is `is_cuda()` (real PTX kernel; the result stays GPU-resident —
//! no `.to(Cpu)`, no host readback of the DATA) and on CPU otherwise via a
//! reference loop matching the same PyTorch semantics the GPU kernels
//! implement. `index_select` / `gather` require the input and index to be on
//! the same device, matching PyTorch's operator wrappers; ferrotorch never
//! downloads a CUDA index to make a CPU input work. CORE-111 / #1805: CUDA
//! `index_select` / `gather` validate index values before the unchecked copy
//! kernels can compute addresses. Non-grad float forwards and integer tensor
//! forwards validate on the GPU and read back only a tiny status payload;
//! tracked float forwards save a CUDA-resident i64 index for backward.
//!
//! These unblock the Llama generation loop, which today round-trips to CPU for
//! argmax sampling and uses raw cudarc slices for token-id embedding gather.
//!
//! ## REQ status (per `.design/ferrotorch-core/ops/phase2c.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | `Tensor::argmax` at `ops/phase2c.rs:218`; consumer: `methods::Tensor::argmax_t` at `methods.rs:670`, `grad_fns::reduction::argmax` at `grad_fns/reduction.rs:1541` |
//! | REQ-2 | SHIPPED | `Tensor::argmin` at `ops/phase2c.rs:223`; consumer: `Tensor::argmin_t` at `methods.rs:684` |
//! | REQ-3 | SHIPPED | `Tensor::index_select` at `ops/phase2c.rs:232`; consumer: `grad_fns::indexing::index_select_differentiable` at `grad_fns/indexing.rs:1217` |
//! | REQ-4 | SHIPPED | `Tensor::gather` at `ops/phase2c.rs:283`; consumer: `grad_fns::indexing::GatherBackward::backward` recurses through it |
//! | REQ-5 | SHIPPED | `Tensor::to_int` at `ops/phase2c.rs:326`; consumer: `int_tensor::Tensor::to_int` re-export path |
//! | REQ-6 | SHIPPED | `IntTensor::argmax`/`argmin` at `ops/phase2c.rs:369,374`; consumer: downstream logit-arg callers |
//! | REQ-7 | SHIPPED | `IntTensor::index_select`/`gather` at `ops/phase2c.rs:380,423`; consumer: `IntTensor` method surface re-export |
//! | REQ-8 | SHIPPED | `IntTensor::to_float` at `ops/phase2c.rs:458`; consumer: `IntTensor` method surface |
//! | REQ-9 | SHIPPED | `IntTensor::cast_gpu` at `ops/phase2c.rs:481`; consumer: `IntTensor::cast` in `int_tensor.rs` |

use crate::device::Device;
use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::int_tensor::{IntElement, IntTensor};
use crate::shape::normalize_axis;
use crate::storage::TensorStorage;
use crate::tensor::Tensor;
use crate::{autograd::no_grad::is_grad_enabled, grad_fns::indexing};
use std::sync::Arc;

/// Factorise `shape` around `dim` into `(outer, dim_size, inner)` for the
/// `[outer, dim_size, inner]` kernel layout. `outer = prod(shape[..dim])`,
/// `inner = prod(shape[dim+1..])`.
fn factor(shape: &[usize], dim: usize) -> (usize, usize, usize) {
    let outer: usize = crate::shape::numel(&shape[..dim]);
    let dim_size = shape[dim];
    let inner: usize = crate::shape::numel(&shape[dim + 1..]);
    (outer, dim_size, inner)
}

/// Shape after removing axis `dim` (the argmax/argmin along-dim output shape).
fn shape_without(shape: &[usize], dim: usize) -> Vec<usize> {
    let mut s = shape.to_vec();
    s.remove(dim);
    s
}

// ── argmax / argmin reference (CPU), generic over a comparator ──────────────

/// First-occurrence argmax/argmin over `data` laid out `[outer, dim_size,
/// inner]`. `better(candidate, current)` returns true iff `candidate` strictly
/// beats `current` (so ties keep the earliest index — PyTorch parity).
fn arg_reduce_ref<V: Copy>(
    data: &[V],
    outer: usize,
    dim_size: usize,
    inner: usize,
    better: impl Fn(V, V) -> bool,
) -> Vec<i64> {
    let mut out = vec![0i64; outer * inner];
    for o in 0..outer {
        for k in 0..inner {
            let base = o * dim_size * inner + k;
            let mut best_j = 0usize;
            let mut best = data[base];
            for j in 1..dim_size {
                let v = data[base + j * inner];
                if better(v, best) {
                    best = v;
                    best_j = j;
                }
            }
            out[o * inner + k] = best_j as i64;
        }
    }
    out
}

/// Run argmax/argmin on a float `Tensor<T>`, returning `IntTensor<i64>`.
/// `dim = None` reduces the flattened tensor to a 0-d scalar index.
fn tensor_arg<T: Float>(
    input: &Tensor<T>,
    dim: Option<isize>,
    is_max: bool,
) -> FerrotorchResult<IntTensor<i64>> {
    let op = if is_max { "argmax" } else { "argmin" };
    if input.numel() == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("{op}: cannot reduce an empty tensor"),
        });
    }
    let input = input.contiguous()?;
    let (outer, dim_size, inner, out_shape) = match dim {
        None => (1usize, input.numel(), 1usize, Vec::new()),
        Some(d) => {
            let d = normalize_axis(d, input.ndim())?;
            let (o, ds, inn) = factor(input.shape(), d);
            (o, ds, inn, shape_without(input.shape(), d))
        }
    };

    if input.is_cuda() {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let h = input.gpu_handle()?;
        let out_h = if is_max {
            backend.argmax(h, outer, dim_size, inner)?
        } else {
            backend.argmin(h, outer, dim_size, inner)?
        };
        Ok(IntTensor::from_gpu_handle(out_h, out_shape))
    } else {
        let data = input.data_vec()?;
        let out = if is_max {
            arg_reduce_ref(&data, outer, dim_size, inner, |c, b| c > b)
        } else {
            arg_reduce_ref(&data, outer, dim_size, inner, |c, b| c < b)
        };
        IntTensor::<i64>::from_vec(out, out_shape)
    }
}

/// Run argmax/argmin on an `IntTensor<I>`, returning `IntTensor<i64>`.
fn inttensor_arg<I: IntElement>(
    input: &IntTensor<I>,
    dim: Option<isize>,
    is_max: bool,
) -> FerrotorchResult<IntTensor<i64>> {
    let op = if is_max { "argmax" } else { "argmin" };
    if input.numel() == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("{op}: cannot reduce an empty tensor"),
        });
    }
    let (outer, dim_size, inner, out_shape) = match dim {
        None => (1usize, input.numel(), 1usize, Vec::new()),
        Some(d) => {
            let d = normalize_axis(d, input.ndim())?;
            let (o, ds, inn) = factor(input.shape(), d);
            (o, ds, inn, shape_without(input.shape(), d))
        }
    };

    if input.is_cuda() {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let h = input.gpu_handle()?;
        let out_h = if is_max {
            backend.argmax(h, outer, dim_size, inner)?
        } else {
            backend.argmin(h, outer, dim_size, inner)?
        };
        Ok(IntTensor::from_gpu_handle(out_h, out_shape))
    } else {
        let data: Vec<i64> = input.data()?.iter().map(|v| v.to_i64()).collect();
        let out = if is_max {
            arg_reduce_ref(&data, outer, dim_size, inner, |c, b| c > b)
        } else {
            arg_reduce_ref(&data, outer, dim_size, inner, |c, b| c < b)
        };
        IntTensor::<i64>::from_vec(out, out_shape)
    }
}

// ── index_select / gather host references (raw element copy) ────────────────

fn index_select_ref<V: Copy>(
    data: &[V],
    indices: &[i64],
    outer: usize,
    in_dim: usize,
    inner: usize,
    zero: V,
) -> Vec<V> {
    let out_dim = indices.len();
    let mut out = vec![zero; outer * out_dim * inner];
    for o in 0..outer {
        for (i, &sel) in indices.iter().enumerate() {
            let sel = sel as usize;
            for k in 0..inner {
                let src = o * in_dim * inner + sel * inner + k;
                out[(o * out_dim + i) * inner + k] = data[src];
            }
        }
    }
    out
}

#[inline]
fn flat_index(coords: &[usize], shape: &[usize]) -> usize {
    let mut idx = 0usize;
    let mut stride = 1usize;
    for d in (0..shape.len()).rev() {
        idx += coords[d] * stride;
        stride *= shape[d];
    }
    idx
}

#[inline]
fn increment_coords(coords: &mut [usize], shape: &[usize]) {
    for d in (0..shape.len()).rev() {
        coords[d] += 1;
        if coords[d] < shape[d] {
            return;
        }
        coords[d] = 0;
    }
}

fn gather_ref<V: Copy>(
    data: &[V],
    indices: &[i64],
    input_shape: &[usize],
    index_shape: &[usize],
    dim: usize,
    zero: V,
) -> Vec<V> {
    let out_numel = indices.len();
    let mut out = vec![zero; out_numel];
    if out_numel == 0 {
        return out;
    }

    let mut coords = vec![0usize; index_shape.len()];
    for t in 0..out_numel {
        let mut src_coords = coords.clone();
        src_coords[dim] = indices[t] as usize;
        out[t] = data[flat_index(&src_coords, input_shape)];

        if t + 1 < out_numel {
            increment_coords(&mut coords, index_shape);
        }
    }
    out
}

fn gather_matches_dim_kernel_layout(
    input_shape: &[usize],
    index_shape: &[usize],
    dim: usize,
) -> bool {
    input_shape
        .iter()
        .zip(index_shape.iter())
        .enumerate()
        .all(|(axis, (&input, &index))| axis == dim || input == index)
}

/// Read an `IntTensor<I>` index as host `Vec<i64>`. CPU references consume this
/// directly. CUDA no-grad and integer paths validate resident indices on the
/// GPU; tracked float CUDA forwards still use this to build the legacy
/// `Vec<usize>` autograd payload after validating the same values.
fn index_as_i64<I: IntElement>(index: &IntTensor<I>) -> FerrotorchResult<Vec<i64>> {
    Ok(index.data()?.iter().map(|v| v.to_i64()).collect())
}

/// Validate every index value against the selected axis (CORE-111 / #1805).
///
/// PyTorch rejects BOTH negative and `>= dim_size` indices — negatives are
/// NOT wrapped (verified live on torch 2.11.0: `torch.gather(x, 1,
/// tensor([[-1],[0],[0]]))` → `RuntimeError: index -1 is out of bounds for
/// dimension 1 with size 4`). Upstream contract:
/// - `index_select` CPU: `TORCH_CHECK_INDEX((self_i >= 0) && (self_i <
///   self_dim_size), "index out of range in self")` —
///   `aten/src/ATen/native/TensorAdvancedIndexing.cpp:1704-1706`.
/// - `gather` CPU: `TORCH_CHECK(idx_dim >= 0 && idx_dim < index_upper_bound,
///   "index ", ..., " is out of bounds for dimension ", dim, " with size ",
///   index_upper_bound)` — `aten/src/ATen/native/cpu/ScatterGatherKernel.cpp:116-120`.
///
/// `usize::try_from` is the checked signed→unsigned conversion (CORE-007
/// class); it makes the unchecked `as usize` casts in the hot reference loops
/// (and the unchecked address math in the `ferrotorch-gpu/src/gather_int.rs`
/// PTX kernels) unreachable with invalid indices. The error carries the
/// offending value, its flat position in the index tensor, the dimension, and
/// the valid range, mirroring torch's gather message shape.
fn check_index_bounds(idx: &[i64], dim: usize, dim_size: usize, op: &str) -> FerrotorchResult<()> {
    for (pos, &v) in idx.iter().enumerate() {
        let in_range = usize::try_from(v).is_ok_and(|u| u < dim_size);
        if !in_range {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "{op}: index {v} is out of bounds for dimension {dim} with size {dim_size} \
                     (flat index position {pos}, valid range 0..{dim_size})"
                ),
            });
        }
    }
    Ok(())
}

// ===========================================================================
// Tensor<T> high-level API
// ===========================================================================

impl<T: Float> Tensor<T> {
    /// Index of the maximum value (PyTorch `torch.argmax`), as `IntTensor<i64>`.
    ///
    /// `dim = None` flattens and returns a 0-d index. `dim = Some(d)` reduces
    /// along `d` (negative indices allowed). Ties resolve to the FIRST (lowest)
    /// index. GPU-resident result when `self` is on CUDA.
    pub fn argmax(&self, dim: Option<isize>) -> FerrotorchResult<IntTensor<i64>> {
        tensor_arg(self, dim, true)
    }

    /// Index of the minimum value (PyTorch `torch.argmin`). See [`Self::argmax`].
    pub fn argmin(&self, dim: Option<isize>) -> FerrotorchResult<IntTensor<i64>> {
        tensor_arg(self, dim, false)
    }

    /// `index_select(dim, indices)` (PyTorch `torch.index_select`) using a
    /// GPU-resident-or-CPU `IntTensor` index. The `indices` tensor must be 1-D.
    /// Output keeps `self`'s dtype; shape is `self.shape` with `shape[dim]`
    /// replaced by `indices.numel()`. `self` and `indices` must be on the same
    /// device; CUDA results stay GPU-resident.
    ///
    /// Every index value must satisfy `0 <= idx < self.shape()[dim]` (PyTorch
    /// parity — negative indices are NOT wrapped); otherwise returns
    /// [`FerrotorchError::InvalidArgument`] carrying the offending value, its
    /// position, and the valid range. On CUDA, forwards validate on device and
    /// read back only a small status payload before the unchecked copy kernel
    /// launches; tracked forwards save an i64 CUDA index for backward without
    /// downloading the full index tensor.
    pub fn index_select<I: IntElement>(
        &self,
        dim: isize,
        indices: &IntTensor<I>,
    ) -> FerrotorchResult<Tensor<T>> {
        if indices.ndim() > 1 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "index_select: indices must be 1-D, got shape {:?}",
                    indices.shape()
                ),
            });
        }
        let d = normalize_axis(dim, self.ndim())?;
        check_same_device(self.device(), indices.device(), "index_select")?;
        let input = self.contiguous()?;
        let (outer, in_dim, inner) = factor(input.shape(), d);
        let out_dim = indices.numel();
        let mut out_shape = input.shape().to_vec();
        out_shape[d] = out_dim;

        if input.is_cuda() {
            let backend =
                crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            backend.check_int_indices_in_bounds(
                indices.gpu_handle()?,
                d,
                in_dim,
                "index_select",
            )?;
            let saved_index_cuda = if input.requires_grad() && is_grad_enabled() {
                Some(indices.cast::<i64>()?)
            } else {
                None
            };
            let h = backend.index_select_intidx(
                input.gpu_handle()?,
                indices.gpu_handle()?,
                outer,
                in_dim,
                out_dim,
                inner,
            )?;
            let storage = TensorStorage::gpu(h);
            if let Some(indices_cuda) = saved_index_cuda {
                let grad_fn = Arc::new(indexing::IndexSelectDimBackward {
                    input: input.clone(),
                    dim: d,
                    indices: Vec::new(),
                    indices_cuda: Some(indices_cuda),
                });
                Tensor::from_operation(storage, out_shape, grad_fn)
            } else {
                Tensor::from_storage(storage, out_shape, false)
            }
        } else {
            let data = input.data_vec()?;
            let idx = index_as_i64(indices)?;
            check_index_bounds(&idx, d, in_dim, "index_select")?;
            let out = index_select_ref(
                &data,
                &idx,
                outer,
                in_dim,
                inner,
                <T as num_traits::Zero>::zero(),
            );
            let storage = TensorStorage::cpu(out);
            if input.requires_grad() && is_grad_enabled() {
                let idx_usize: Vec<usize> = idx
                    .iter()
                    .map(|&v| usize::try_from(v).expect("validated non-negative index"))
                    .collect();
                let grad_fn = Arc::new(indexing::IndexSelectDimBackward {
                    input: input.clone(),
                    dim: d,
                    indices: idx_usize,
                    indices_cuda: None,
                });
                Tensor::from_operation(storage, out_shape, grad_fn)
            } else {
                Tensor::from_storage(storage, out_shape, false)
            }
        }
    }

    /// `gather(dim, index)` (PyTorch `torch.gather`) using a GPU-resident-or-CPU
    /// `IntTensor` index. `index` must have the same ndim as `self`; output has
    /// `index`'s shape and `self`'s dtype. `self` and `index` must be on the
    /// same device; CUDA results stay resident.
    ///
    /// Every index value must satisfy `0 <= idx < self.shape()[dim]` (PyTorch
    /// parity — negative indices are NOT wrapped); otherwise returns
    /// [`FerrotorchError::InvalidArgument`] carrying the offending value, its
    /// position, and the valid range. On CUDA, forwards validate on device and
    /// read back only a small status payload before the unchecked copy kernel
    /// launches; tracked forwards save an i64 CUDA index for backward without
    /// downloading the full index tensor.
    pub fn gather<I: IntElement>(
        &self,
        dim: isize,
        index: &IntTensor<I>,
    ) -> FerrotorchResult<Tensor<T>> {
        let d = normalize_axis(dim, self.ndim())?;
        gather_check_shapes(self.shape(), index.shape(), d, "gather")?;
        check_same_device(self.device(), index.device(), "gather")?;
        let input = self.contiguous()?;
        let (outer, in_dim, inner) = factor(input.shape(), d);
        let out_dim = index.shape()[d];
        let out_shape = index.shape().to_vec();

        if input.is_cuda() {
            let backend =
                crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            backend.check_int_indices_in_bounds(index.gpu_handle()?, d, in_dim, "gather")?;
            let saved_index_cuda = if input.requires_grad() && is_grad_enabled() {
                Some(index.cast::<i64>()?)
            } else {
                None
            };
            let h = if gather_matches_dim_kernel_layout(input.shape(), index.shape(), d) {
                backend.gather_intidx(
                    input.gpu_handle()?,
                    index.gpu_handle()?,
                    outer,
                    in_dim,
                    out_dim,
                    inner,
                )?
            } else {
                backend.gather_intidx_nd(
                    input.gpu_handle()?,
                    index.gpu_handle()?,
                    input.shape(),
                    index.shape(),
                    d,
                )?
            };
            let storage = TensorStorage::gpu(h);
            if let Some(index_cuda) = saved_index_cuda {
                let grad_fn = Arc::new(indexing::GatherBackward {
                    input: input.clone(),
                    dim: d,
                    index: Vec::new(),
                    index_cuda: Some(index_cuda),
                    index_shape: index.shape().to_vec(),
                });
                Tensor::from_operation(storage, out_shape, grad_fn)
            } else {
                Tensor::from_storage(storage, out_shape, false)
            }
        } else {
            let data = input.data_vec()?;
            let idx = index_as_i64(index)?;
            check_index_bounds(&idx, d, in_dim, "gather")?;
            let out = gather_ref(
                &data,
                &idx,
                input.shape(),
                index.shape(),
                d,
                <T as num_traits::Zero>::zero(),
            );
            let storage = TensorStorage::cpu(out);
            if input.requires_grad() && is_grad_enabled() {
                let idx_usize: Vec<usize> = idx
                    .iter()
                    .map(|&v| usize::try_from(v).expect("validated non-negative index"))
                    .collect();
                let grad_fn = Arc::new(indexing::GatherBackward {
                    input: input.clone(),
                    dim: d,
                    index: idx_usize,
                    index_cuda: None,
                    index_shape: index.shape().to_vec(),
                });
                Tensor::from_operation(storage, out_shape, grad_fn)
            } else {
                Tensor::from_storage(storage, out_shape, false)
            }
        }
    }

    /// Cast this float tensor to `IntTensor<I>` (PyTorch `.to(int)`):
    /// **truncate toward zero**. GPU-resident result when `self` is on CUDA.
    pub fn to_int<I: IntElement>(&self) -> FerrotorchResult<IntTensor<I>> {
        let input = self.contiguous()?;
        let shape = input.shape().to_vec();
        if input.is_cuda() {
            let backend =
                crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let h = backend.cast_f_to_i(input.gpu_handle()?, I::dtype())?;
            Ok(IntTensor::from_gpu_handle(h, shape))
        } else {
            let data = input.data_vec()?;
            let mut out: Vec<I> = Vec::with_capacity(data.len());
            for &v in &data {
                // Truncate toward zero (PyTorch `.to(int)`): drop the fraction.
                let truncated = num_traits::Float::trunc(v);
                let as_i64 = float_to_i64_trunc(truncated);
                out.push(
                    I::try_from_i64(as_i64).ok_or(FerrotorchError::InvalidArgument {
                        message: format!("to_int: value out of range for {}", I::dtype_name()),
                    })?,
                );
            }
            IntTensor::<I>::from_vec(out, shape)
        }
    }
}

/// Convert an already-truncated float to i64 (saturating at the i64 range,
/// matching the CPU `as` cast which saturates rather than wraps).
fn float_to_i64_trunc<T: Float>(v: T) -> i64 {
    // `T: Float` -> f64 is lossless for f32/bf16/f16 and exact enough for the
    // integer range here; `as i64` on f64 saturates (Rust 1.45+ semantics),
    // matching PyTorch's clamp-on-overflow for `.to(int64)`.
    let f: f64 = num_traits::ToPrimitive::to_f64(&v).unwrap_or(0.0);
    f as i64
}

// ===========================================================================
// IntTensor<I> high-level API
// ===========================================================================

impl<I: IntElement> IntTensor<I> {
    /// Index of the maximum value, as `IntTensor<i64>`. See
    /// [`Tensor::argmax`](crate::tensor::Tensor::argmax).
    pub fn argmax(&self, dim: Option<isize>) -> FerrotorchResult<IntTensor<i64>> {
        inttensor_arg(self, dim, true)
    }

    /// Index of the minimum value, as `IntTensor<i64>`.
    pub fn argmin(&self, dim: Option<isize>) -> FerrotorchResult<IntTensor<i64>> {
        inttensor_arg(self, dim, false)
    }

    /// `index_select(dim, indices)` on integer data (1-D `indices`). Output
    /// keeps this tensor's int dtype; GPU-resident when on CUDA.
    ///
    /// Index values are validated against `self.shape()[dim]` exactly like
    /// [`Tensor::index_select`](crate::tensor::Tensor::index_select) (PyTorch
    /// parity: no negative wrapping; `InvalidArgument` on out-of-bounds).
    /// `self` and `indices` must be on the same device. On CUDA, validation
    /// runs on the device and reads back only a small status payload before
    /// the unchecked copy kernel launches.
    pub fn index_select<J: IntElement>(
        &self,
        dim: isize,
        indices: &IntTensor<J>,
    ) -> FerrotorchResult<IntTensor<I>> {
        if indices.ndim() > 1 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "index_select: indices must be 1-D, got shape {:?}",
                    indices.shape()
                ),
            });
        }
        let d = normalize_axis(dim, self.ndim())?;
        check_same_device(self.device(), indices.device(), "index_select")?;
        let (outer, in_dim, inner) = factor(self.shape(), d);
        let out_dim = indices.numel();
        let mut out_shape = self.shape().to_vec();
        out_shape[d] = out_dim;

        if self.is_cuda() {
            let backend =
                crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            backend.check_int_indices_in_bounds(
                indices.gpu_handle()?,
                d,
                in_dim,
                "index_select",
            )?;
            let h = backend.index_select_intidx(
                self.gpu_handle()?,
                indices.gpu_handle()?,
                outer,
                in_dim,
                out_dim,
                inner,
            )?;
            Ok(IntTensor::from_gpu_handle(h, out_shape))
        } else {
            let data = self.data()?;
            let idx = index_as_i64(indices)?;
            check_index_bounds(&idx, d, in_dim, "index_select")?;
            let zero = I::try_from_i64(0).expect("0 is in range for i32/i64");
            let out = index_select_ref(data, &idx, outer, in_dim, inner, zero);
            IntTensor::<I>::from_vec(out, out_shape)
        }
    }

    /// `gather(dim, index)` on integer data. `index` must match this tensor's
    /// ndim; output has `index`'s shape and this tensor's int dtype.
    ///
    /// Index values are validated against `self.shape()[dim]` exactly like
    /// [`Tensor::gather`](crate::tensor::Tensor::gather) (PyTorch parity: no
    /// negative wrapping; `InvalidArgument` on out-of-bounds). `self` and
    /// `index` must be on the same device. On CUDA, validation runs on the
    /// device and reads back only a small status payload before the unchecked
    /// copy kernel launches.
    pub fn gather<J: IntElement>(
        &self,
        dim: isize,
        index: &IntTensor<J>,
    ) -> FerrotorchResult<IntTensor<I>> {
        let d = normalize_axis(dim, self.ndim())?;
        gather_check_shapes(self.shape(), index.shape(), d, "gather")?;
        check_same_device(self.device(), index.device(), "gather")?;
        let (outer, in_dim, inner) = factor(self.shape(), d);
        let out_dim = index.shape()[d];
        let out_shape = index.shape().to_vec();

        if self.is_cuda() {
            let backend =
                crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            backend.check_int_indices_in_bounds(index.gpu_handle()?, d, in_dim, "gather")?;
            let h = if gather_matches_dim_kernel_layout(self.shape(), index.shape(), d) {
                backend.gather_intidx(
                    self.gpu_handle()?,
                    index.gpu_handle()?,
                    outer,
                    in_dim,
                    out_dim,
                    inner,
                )?
            } else {
                backend.gather_intidx_nd(
                    self.gpu_handle()?,
                    index.gpu_handle()?,
                    self.shape(),
                    index.shape(),
                    d,
                )?
            };
            Ok(IntTensor::from_gpu_handle(h, out_shape))
        } else {
            let data = self.data()?;
            let idx = index_as_i64(index)?;
            check_index_bounds(&idx, d, in_dim, "gather")?;
            let zero = I::try_from_i64(0).expect("0 is in range for i32/i64");
            let out = gather_ref(data, &idx, self.shape(), index.shape(), d, zero);
            IntTensor::<I>::from_vec(out, out_shape)
        }
    }

    /// Cast this integer tensor to a float `Tensor<T>` (PyTorch `.to(float)`),
    /// round-to-nearest-even. GPU-resident result when `self` is on CUDA.
    pub fn to_float<T: Float>(&self) -> FerrotorchResult<Tensor<T>> {
        let shape = self.shape().to_vec();
        if self.is_cuda() {
            let backend =
                crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let h = backend.cast_i_to_f(self.gpu_handle()?, T::dtype())?;
            Tensor::from_storage(TensorStorage::gpu(h), shape, false)
        } else {
            let data = self.data()?;
            let mut out: Vec<T> = Vec::with_capacity(data.len());
            for &v in data {
                out.push(num_traits::NumCast::from(v.to_i64()).ok_or(
                    FerrotorchError::InvalidArgument {
                        message: "to_float: integer not representable in target float".into(),
                    },
                )?);
            }
            Tensor::from_storage(TensorStorage::cpu(out), shape, false)
        }
    }

    /// GPU path for [`IntTensor::cast`] (i32 ↔ i64). Returns `None` so the
    /// caller's CPU path handles non-CUDA tensors; `Some(Ok/Err)` on CUDA.
    pub(crate) fn cast_gpu<J: IntElement>(&self) -> Option<FerrotorchResult<IntTensor<J>>> {
        if !self.is_cuda() {
            return None;
        }
        let shape = self.shape().to_vec();
        let backend = match crate::gpu_dispatch::gpu_backend() {
            Some(b) => b,
            None => return Some(Err(FerrotorchError::DeviceUnavailable)),
        };
        let h = match self.gpu_handle() {
            Ok(h) => h,
            Err(e) => return Some(Err(e)),
        };
        Some(
            backend
                .cast_i_to_i(h, J::dtype())
                .map(|out_h| IntTensor::from_gpu_handle(out_h, shape)),
        )
    }
}

// ── shared validation ───────────────────────────────────────────────────────

fn check_same_device(a: Device, b: Device, op: &str) -> FerrotorchResult<()> {
    if a != b {
        return Err(FerrotorchError::DeviceMismatch {
            expected: a,
            got: b,
        });
    }
    let _ = op;
    Ok(())
}

/// `gather` shape rule: `index.ndim() == input.ndim()`, and for every axis
/// `ax != dim`, `index.shape[ax] <= input.shape[ax]` (PyTorch allows the index
/// to be smaller off the gather axis). The gather axis itself is unconstrained
/// in size (each element selects independently).
fn gather_check_shapes(
    input_shape: &[usize],
    index_shape: &[usize],
    dim: usize,
    op: &str,
) -> FerrotorchResult<()> {
    if index_shape.len() != input_shape.len() {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "{op}: index ndim {} != input ndim {}",
                index_shape.len(),
                input_shape.len()
            ),
        });
    }
    for (ax, (&isz, &xsz)) in index_shape.iter().zip(input_shape.iter()).enumerate() {
        if ax != dim && isz > xsz {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!("{op}: index dim {ax} size {isz} exceeds input size {xsz}"),
            });
        }
    }
    Ok(())
}
