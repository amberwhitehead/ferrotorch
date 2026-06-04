//! Common tensor manipulation operations.
//!
//! - [`triu`] / [`tril`] — upper/lower triangular masks
//! - [`diag`] / [`diagflat`] — diagonal extraction/construction
//! - [`roll`] — circular shift along a dimension
//! - [`cdist`] — pairwise distance matrix
//!
//! ## REQ status (per `.design/ferrotorch-core/ops/tensor_ops.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | `triu` in `ops/tensor_ops.rs` (CPU + GPU f32/f64 `is_cuda()` branch); GPU kernel `gpu_triu_f32`/`gpu_triu_f64` in `ferrotorch-gpu/src/triangular.rs`; consumer: re-export `ferrotorch_core::triu` in `lib.rs`; GPU consumer: `CudaBackendImpl::triu_f32`/`triu_f64` in `ferrotorch-gpu/src/backend_impl.rs` (crosslink #1545 / sub #1535) |
//! | REQ-2 | SHIPPED | `tril` in `ops/tensor_ops.rs` (CPU + GPU f32/f64 `is_cuda()` branch); GPU kernel `gpu_tril_f32`/`gpu_tril_f64` in `ferrotorch-gpu/src/triangular.rs`; consumer: re-export in `lib.rs`; GPU consumer: `CudaBackendImpl::tril_f32`/`tril_f64` in `ferrotorch-gpu/src/backend_impl.rs` |
//! | REQ-3 | SHIPPED | `diag` in `ops/tensor_ops.rs` (CPU + GPU f32/f64 `is_cuda()` branch); GPU kernels `gpu_diag_embed_f32`/`gpu_diag_extract_f32` (+f64) in `ferrotorch-gpu/src/diag.rs`; consumer: re-export `ferrotorch_core::diag` in `lib.rs`; GPU consumer: `CudaBackendImpl::diag_embed_f32`/`diag_extract_f32` (+f64) in `ferrotorch-gpu/src/backend_impl.rs` (crosslink #1545 / sub #1535) |
//! | REQ-4 | SHIPPED | `diagflat` in `ops/tensor_ops.rs` flattens via device-aware `Tensor::view_reshape` then delegates to `diag` (so CUDA inputs ride the `diag` GPU fast path, GPU-resident); consumer: re-export `ferrotorch_core::diagflat` in `lib.rs` |
//! | REQ-5 | SHIPPED | `roll` in `ops/tensor_ops.rs` (CPU + GPU f32/f64 `is_cuda()` branch); consumer: re-export `ferrotorch_core::roll` in `lib.rs`; `RollBackward` autograd; GPU f64 kernel `gpu_roll_f64` in `ferrotorch-gpu/src/roll.rs`, GPU consumer `CudaBackendImpl::roll_f64` in `ferrotorch-gpu/src/backend_impl.rs` (crosslink #1545 / sub #1535) |
//! | REQ-6 | SHIPPED | `cdist` in `ops/tensor_ops.rs` (CPU + GPU f32 {1,2,inf,general}/f64 {1,2,inf} `is_cuda()` branch); GPU kernel `gpu_cdist_f32`/`gpu_cdist_f64` in `ferrotorch-gpu/src/distance.rs`; consumer: re-export `ferrotorch_core::cdist` in `lib.rs`; GPU consumer: `CudaBackendImpl::cdist_f32`/`cdist_f64` in `ferrotorch-gpu/src/backend_impl.rs` |
//! | REQ-7 | SHIPPED | `roll_cpu_inner` in `ops/tensor_ops.rs`; consumer: `grad_fns::shape::RollBackward::backward` in `grad_fns/shape.rs` invokes `ops::tensor_ops::roll_cpu_inner` |

use std::any::TypeId;
use std::sync::Arc;

use crate::autograd::no_grad::is_grad_enabled;
use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::storage::TensorStorage;
use crate::tensor::Tensor;

#[inline]
fn is_f32<T: Float>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f32>()
}

#[inline]
fn is_f64<T: Float>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f64>()
}

/// Upper triangular part of a tensor with at least 2 dimensions.
///
/// Elements below the `diagonal`-th diagonal are set to zero.
/// `diagonal=0` is the main diagonal, `diagonal>0` is above, `diagonal<0` is below.
///
/// For an N-D tensor (`ndim >= 2`) the mask is applied to the LAST TWO DIMS of
/// every trailing `[rows, cols]` matrix, batching over all leading dims, so the
/// output shape equals the input shape. Mirrors PyTorch's `torch.triu`
/// (`aten/src/ATen/native/TriangularOps.cpp:31` requires `dim() >= 2`; the CUDA
/// template batches via `cuda/TriangularOps.cu:120`).
///
/// # Backward
/// Autograd-aware (CPU): when grad tracking is active for `input`, this routes
/// through `crate::grad_fns::linalg::triu_differentiable` (the VJP masks the
/// upstream gradient by the kept upper triangle, per `triu -> grad.triu_symint`
/// at upstream `tools/autograd/derivatives.yaml:1809`).
pub fn triu<T: Float>(input: &Tensor<T>, diagonal: i64) -> FerrotorchResult<Tensor<T>> {
    if input.ndim() < 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "triu: input tensor must have at least 2 dimensions, got shape {:?}",
                input.shape()
            ),
        });
    }
    // Autograd path: delegate to the differentiable wrapper, which computes
    // the forward inside `no_grad` (preventing re-entry here) and attaches
    // `TriangularBackward`. The `no_grad` re-entry lands in the GPU/CPU
    // forward branches below with grad disabled, so a CUDA input still runs
    // the resident kernel for the forward value.
    if is_grad_enabled() && input.requires_grad() {
        return crate::grad_fns::linalg::triu_differentiable(input, diagonal);
    }

    // Trailing two dims are the matrix; all leading dims are batched (mirrors
    // `aten/src/ATen/native/cuda/TriangularOps.cu:120`).
    let shape = input.shape();
    let ndim = shape.len();
    let rows = shape[ndim - 2];
    let cols = shape[ndim - 1];
    let batch: usize = shape[..ndim - 2].iter().product();

    // GPU fast path: f32/f64 resident kernel (crosslink #1545 / sub #1535).
    // One thread per element; predicate `col - row >= diagonal` mirrors
    // `aten/src/ATen/native/cuda/TriangularOps.cu:100`, batched per trailing
    // `[rows, cols]` matrix per `:120`. Result stays GPU-resident — NO host
    // round-trip. Other GPU dtypes keep the `NotImplementedOnCuda` contract.
    if input.is_cuda() {
        if let Some(backend) = crate::gpu_dispatch::gpu_backend() {
            // #1658: normalise a narrowed-offset CUDA view to a packed offset-0
            // buffer before the per-element kernel reads element 0.
            let input = input.contiguous()?;
            let handle = if is_f32::<T>() {
                Some(backend.triu_f32(input.gpu_handle()?, batch, rows, cols, diagonal)?)
            } else if is_f64::<T>() {
                Some(backend.triu_f64(input.gpu_handle()?, batch, rows, cols, diagonal)?)
            } else {
                None
            };
            if let Some(handle) = handle {
                let storage = TensorStorage::gpu(handle);
                return Tensor::from_storage(storage, shape.to_vec(), false);
            }
        }
        return Err(FerrotorchError::NotImplementedOnCuda { op: "triu" });
    }

    let data = input.data()?;
    let zero = <T as num_traits::Zero>::zero();

    let matrix = rows * cols;
    let mut out = Vec::with_capacity(batch * matrix);
    for b in 0..batch {
        let base = b * matrix;
        for r in 0..rows {
            for c in 0..cols {
                if (c as i64) >= (r as i64) + diagonal {
                    out.push(data[base + r * cols + c]);
                } else {
                    out.push(zero);
                }
            }
        }
    }

    Tensor::from_storage(TensorStorage::cpu(out), shape.to_vec(), false)
}

/// Lower triangular part of a tensor with at least 2 dimensions.
///
/// Elements above the `diagonal`-th diagonal are set to zero.
///
/// For an N-D tensor (`ndim >= 2`) the mask is applied to the LAST TWO DIMS of
/// every trailing `[rows, cols]` matrix, batching over all leading dims.
/// Mirrors PyTorch's `torch.tril` (`aten/src/ATen/native/TriangularOps.cpp:25`
/// requires `dim() >= 2`; the CUDA template batches via
/// `cuda/TriangularOps.cu:120`).
///
/// # Backward
/// Autograd-aware (CPU): when grad tracking is active for `input`, this routes
/// through `crate::grad_fns::linalg::tril_differentiable` (the VJP masks the
/// upstream gradient by the kept lower triangle, per `tril -> grad.tril_symint`
/// at upstream `tools/autograd/derivatives.yaml:1805`).
pub fn tril<T: Float>(input: &Tensor<T>, diagonal: i64) -> FerrotorchResult<Tensor<T>> {
    if input.ndim() < 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "tril: input tensor must have at least 2 dimensions, got shape {:?}",
                input.shape()
            ),
        });
    }
    // Autograd path: delegate to the differentiable wrapper, which computes
    // the forward inside `no_grad` (preventing re-entry here) and attaches
    // `TriangularBackward`. The `no_grad` re-entry lands in the GPU/CPU
    // forward branches below with grad disabled.
    if is_grad_enabled() && input.requires_grad() {
        return crate::grad_fns::linalg::tril_differentiable(input, diagonal);
    }

    // Trailing two dims are the matrix; all leading dims are batched (mirrors
    // `aten/src/ATen/native/cuda/TriangularOps.cu:120`).
    let shape = input.shape();
    let ndim = shape.len();
    let rows = shape[ndim - 2];
    let cols = shape[ndim - 1];
    let batch: usize = shape[..ndim - 2].iter().product();

    // GPU fast path: f32/f64 resident kernel (crosslink #1545 / sub #1535).
    // Predicate `col - row <= diagonal` mirrors
    // `aten/src/ATen/native/cuda/TriangularOps.cu:100`, batched per trailing
    // `[rows, cols]` matrix per `:120`. Result stays GPU-resident — NO host
    // round-trip.
    if input.is_cuda() {
        if let Some(backend) = crate::gpu_dispatch::gpu_backend() {
            // #1658: normalise a narrowed-offset CUDA view to a packed offset-0
            // buffer before the per-element kernel reads element 0.
            let input = input.contiguous()?;
            let handle = if is_f32::<T>() {
                Some(backend.tril_f32(input.gpu_handle()?, batch, rows, cols, diagonal)?)
            } else if is_f64::<T>() {
                Some(backend.tril_f64(input.gpu_handle()?, batch, rows, cols, diagonal)?)
            } else {
                None
            };
            if let Some(handle) = handle {
                let storage = TensorStorage::gpu(handle);
                return Tensor::from_storage(storage, shape.to_vec(), false);
            }
        }
        return Err(FerrotorchError::NotImplementedOnCuda { op: "tril" });
    }

    let data = input.data()?;
    let zero = <T as num_traits::Zero>::zero();

    let matrix = rows * cols;
    let mut out = Vec::with_capacity(batch * matrix);
    for b in 0..batch {
        let base = b * matrix;
        for r in 0..rows {
            for c in 0..cols {
                if (c as i64) <= (r as i64) + diagonal {
                    out.push(data[base + r * cols + c]);
                } else {
                    out.push(zero);
                }
            }
        }
    }

    Tensor::from_storage(TensorStorage::cpu(out), shape.to_vec(), false)
}

/// Extract the diagonal of a 2-D tensor, or construct a 2-D diagonal matrix
/// from a 1-D tensor.
///
/// - If `input` is 2-D: returns the `diagonal`-th diagonal as a 1-D tensor.
/// - If `input` is 1-D: returns a 2-D tensor with `input` on the `diagonal`-th diagonal.
///
/// Matches PyTorch's `torch.diag`.
///
/// # Backward
/// Autograd-aware (CPU): when grad tracking is active for `input`, this routes
/// through `crate::grad_fns::linalg::diag_differentiable` (the adjoint of the
/// 0/1 selection — for a 1-D input the VJP gathers grad's diagonal, for a 2-D
/// input it scatters grad onto the `diagonal`-th diagonal of a zero matrix).
pub fn diag<T: Float>(input: &Tensor<T>, diagonal: i64) -> FerrotorchResult<Tensor<T>> {
    // Autograd path: delegate to the differentiable wrapper, which computes
    // the forward inside `no_grad` (preventing re-entry here, so the `no_grad`
    // re-entry lands in the GPU/CPU forward branches below with grad disabled)
    // and attaches `DiagBackward`.
    if is_grad_enabled() && input.requires_grad() {
        return crate::grad_fns::linalg::diag_differentiable(input, diagonal);
    }

    // Validate rank first so the CUDA and CPU paths share the 1-D/2-D contract
    // (mirrors `aten/src/ATen/native/TensorShape.cpp:4612` `ndim == 1 || 2`).
    let ndim = input.ndim();
    if ndim != 1 && ndim != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("diag: expected 1-D or 2-D tensor, got {:?}", input.shape()),
        });
    }

    // GPU fast path: f32/f64 resident kernel (crosslink #1545 / sub #1535).
    // 1-D → `diag_embed` scatter onto the k-th diagonal of a `[size, size]`
    // matrix (`size = n + |k|`); 2-D → `diag_extract` gather of the k-th
    // diagonal. Both mirror `torch.diag` (`TensorShape.cpp:4610`); pure
    // gather/scatter, so the GPU result is bit-identical to the CPU path and
    // stays GPU-resident — NO host round-trip. Other GPU dtypes keep the
    // `NotImplementedOnCuda` contract.
    if input.is_cuda() {
        if let Some(backend) = crate::gpu_dispatch::gpu_backend() {
            // #1658: normalise a narrowed-offset CUDA view to a packed offset-0
            // buffer before the gather/scatter kernel reads element 0.
            let input = input.contiguous()?;
            let result = if ndim == 1 {
                let n = input.shape()[0];
                let size = n + diagonal.unsigned_abs() as usize;
                let handle = if is_f32::<T>() {
                    Some(backend.diag_embed_f32(input.gpu_handle()?, n, diagonal)?)
                } else if is_f64::<T>() {
                    Some(backend.diag_embed_f64(input.gpu_handle()?, n, diagonal)?)
                } else {
                    None
                };
                handle.map(|h| (h, vec![size, size]))
            } else {
                let rows = input.shape()[0];
                let cols = input.shape()[1];
                let (start_r, start_c) = if diagonal >= 0 {
                    (0usize, diagonal as usize)
                } else {
                    ((-diagonal) as usize, 0usize)
                };
                let diag_len = rows
                    .saturating_sub(start_r)
                    .min(cols.saturating_sub(start_c));
                let handle = if is_f32::<T>() {
                    Some(backend.diag_extract_f32(input.gpu_handle()?, rows, cols, diagonal)?)
                } else if is_f64::<T>() {
                    Some(backend.diag_extract_f64(input.gpu_handle()?, rows, cols, diagonal)?)
                } else {
                    None
                };
                handle.map(|h| (h, vec![diag_len]))
            };
            if let Some((handle, shape)) = result {
                let storage = TensorStorage::gpu(handle);
                return Tensor::from_storage(storage, shape, false);
            }
        }
        return Err(FerrotorchError::NotImplementedOnCuda { op: "diag" });
    }

    match ndim {
        1 => {
            // 1-D → 2-D diagonal matrix
            let data = input.data()?;
            let n = data.len();
            let offset = diagonal.unsigned_abs() as usize;
            let size = n + offset;
            let zero = <T as num_traits::Zero>::zero();
            let mut out = vec![zero; size * size];

            for (i, &val) in data[..n].iter().enumerate() {
                let (r, c) = if diagonal >= 0 {
                    (i, i + offset)
                } else {
                    (i + offset, i)
                };
                out[r * size + c] = val;
            }

            Tensor::from_storage(TensorStorage::cpu(out), vec![size, size], false)
        }
        _ => {
            // 2-D → extract diagonal
            let rows = input.shape()[0];
            let cols = input.shape()[1];
            let data = input.data()?;

            let (start_r, start_c) = if diagonal >= 0 {
                (0, diagonal as usize)
            } else {
                ((-diagonal) as usize, 0)
            };

            let diag_len = (rows - start_r).min(cols - start_c);
            let mut out = Vec::with_capacity(diag_len);
            for i in 0..diag_len {
                out.push(data[(start_r + i) * cols + (start_c + i)]);
            }

            Tensor::from_storage(TensorStorage::cpu(out), vec![diag_len], false)
        }
    }
}

/// Construct a diagonal matrix from a 1-D tensor (flattened if needed).
///
/// Like `diag` with a 1-D input, but first flattens multi-dimensional input.
///
/// Matches PyTorch's `torch.diagflat`.
pub fn diagflat<T: Float>(input: &Tensor<T>, diagonal: i64) -> FerrotorchResult<Tensor<T>> {
    // Flatten to 1-D, then delegate to `diag` (which carries the GPU f32/f64
    // resident `diag_embed` fast path). `view_reshape` is device-aware: a
    // contiguous CUDA tensor stays GPU-resident (shares storage, NO host
    // round-trip), a non-contiguous one is gathered on-device first. This
    // mirrors `torch.diagflat` flattening before `diag_embed`
    // (`aten/src/ATen/native/TensorShape.cpp:1230`).
    let flat = if input.ndim() == 1 {
        input.clone()
    } else {
        input.view_reshape(vec![input.numel()])?
    };

    diag(&flat, diagonal)
}

/// Roll (circular shift) a tensor along a dimension.
///
/// Elements shifted past the last position wrap to the beginning.
///
/// Matches PyTorch's `torch.roll`.
///
/// Autograd: when `input.requires_grad()` and grad is enabled, the result
/// carries a [`RollBackward`](crate::grad_fns::shape::RollBackward) grad_fn
/// that pushes gradients back through the inverse shift
/// (`grad_input = roll(grad_output, -shifts, dim)`).
pub fn roll<T: Float>(input: &Tensor<T>, shifts: i64, dim: usize) -> FerrotorchResult<Tensor<T>> {
    let shape = input.shape();
    if dim >= shape.len() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("roll: dim {dim} out of range for shape {shape:?}"),
        });
    }

    let dim_size = shape[dim] as i64;
    // Empty axis: roll is a no-op, but we still need a grad_fn for graph continuity.
    let shift_norm = if dim_size == 0 {
        0
    } else {
        ((shifts % dim_size) + dim_size) % dim_size
    };

    if shift_norm == 0 {
        // Early-return preserves the existing eval-mode behaviour. There is
        // no shape change and the data is identical, so identity-grad is
        // correct: forwarding `input.clone()` keeps the upstream grad_fn
        // intact when it exists.
        return Ok(input.clone());
    }

    // GPU fast path: f32 and f64. `roll` is pure index movement (a circular
    // shift), so the f64 kernel is bit-exact (no transcendentals). Other
    // dtypes (bf16/f16/...) fall through to the existing NotImplementedOnCuda
    // error so the contract matches the rest of `tensor_ops`.
    if input.is_cuda() {
        if (is_f32::<T>() || is_f64::<T>())
            && let Some(backend) = crate::gpu_dispatch::gpu_backend()
        {
            let outer: usize = shape[..dim].iter().product();
            let inner: usize = shape[dim + 1..].iter().product();
            let handle = if is_f32::<T>() {
                backend.roll_f32(
                    input.gpu_handle()?,
                    outer,
                    shape[dim],
                    inner,
                    shift_norm as usize,
                )?
            } else {
                backend.roll_f64(
                    input.gpu_handle()?,
                    outer,
                    shape[dim],
                    inner,
                    shift_norm as usize,
                )?
            };
            let storage = TensorStorage::gpu(handle);
            return if input.requires_grad() && is_grad_enabled() {
                let grad_fn = Arc::new(crate::grad_fns::shape::RollBackward::new(
                    input.clone(),
                    shifts,
                    dim,
                ));
                Tensor::from_operation(storage, shape.to_vec(), grad_fn)
            } else {
                Tensor::from_storage(storage, shape.to_vec(), false)
            };
        }
        return Err(FerrotorchError::NotImplementedOnCuda { op: "roll" });
    }

    let data = input.data_vec()?;
    let out = roll_cpu_inner(&data, shape, shift_norm as usize, dim);

    if input.requires_grad() && is_grad_enabled() {
        let grad_fn = Arc::new(crate::grad_fns::shape::RollBackward::new(
            input.clone(),
            shifts,
            dim,
        ));
        Tensor::from_operation(TensorStorage::cpu(out), shape.to_vec(), grad_fn)
    } else {
        Tensor::from_storage(TensorStorage::cpu(out), shape.to_vec(), false)
    }
}

/// CPU shift kernel shared by `roll` (forward) and `RollBackward` (backward).
///
/// Performs `out[..., new_d, ...] = data[..., d, ...]` where
/// `new_d = (d + shift_norm) % dim_size`. `shift_norm` is the
/// already-normalized non-negative shift (`shift_norm < dim_size`).
///
/// `shape[dim]` is assumed > 0 (callers handle the empty-axis early return).
pub(crate) fn roll_cpu_inner<T: Float>(
    data: &[T],
    shape: &[usize],
    shift_norm: usize,
    dim: usize,
) -> Vec<T> {
    let numel = data.len();
    let dim_size = shape[dim];
    let inner: usize = shape[dim + 1..].iter().product();
    let outer: usize = numel / (dim_size * inner);
    let mut out = vec![<T as num_traits::Zero>::zero(); numel];

    for o in 0..outer {
        for d in 0..dim_size {
            let new_d = (d + shift_norm) % dim_size;
            for i in 0..inner {
                let src = o * dim_size * inner + d * inner + i;
                let dst = o * dim_size * inner + new_d * inner + i;
                out[dst] = data[src];
            }
        }
    }
    out
}

/// Pairwise distance matrix between two sets of vectors.
///
/// `x1` has shape `[B, P, M]`, `x2` has shape `[B, R, M]`.
/// Returns shape `[B, P, R]` with Lp distances.
///
/// If `x1` is 2-D `[P, M]` and `x2` is 2-D `[R, M]`, returns `[P, R]`.
///
/// Matches PyTorch's `torch.cdist`.
pub fn cdist<T: Float>(x1: &Tensor<T>, x2: &Tensor<T>, p: f64) -> FerrotorchResult<Tensor<T>> {
    let (batched, b, p_dim, r_dim, m) = match (x1.ndim(), x2.ndim()) {
        (2, 2) => {
            let p_dim = x1.shape()[0];
            let m1 = x1.shape()[1];
            let r_dim = x2.shape()[0];
            let m2 = x2.shape()[1];
            if m1 != m2 {
                return Err(FerrotorchError::ShapeMismatch {
                    message: format!("cdist: feature dims mismatch: {m1} vs {m2}"),
                });
            }
            (false, 1, p_dim, r_dim, m1)
        }
        (3, 3) => {
            if x1.shape()[0] != x2.shape()[0] {
                return Err(FerrotorchError::ShapeMismatch {
                    message: format!(
                        "cdist: batch dims mismatch: {} vs {}",
                        x1.shape()[0],
                        x2.shape()[0]
                    ),
                });
            }
            if x1.shape()[2] != x2.shape()[2] {
                return Err(FerrotorchError::ShapeMismatch {
                    message: format!(
                        "cdist: feature dims mismatch: {} vs {}",
                        x1.shape()[2],
                        x2.shape()[2]
                    ),
                });
            }
            (
                true,
                x1.shape()[0],
                x1.shape()[1],
                x2.shape()[1],
                x1.shape()[2],
            )
        }
        _ => {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "cdist: expected 2-D or 3-D inputs, got {:?} and {:?}",
                    x1.shape(),
                    x2.shape()
                ),
            });
        }
    };

    let out_shape = if batched {
        vec![b, p_dim, r_dim]
    } else {
        vec![p_dim, r_dim]
    };

    // GPU fast path: f32/f64 resident kernel (crosslink #1545 / sub #1535).
    // Both inputs must be CUDA-resident on the same device. Mirrors
    // `aten/src/ATen/native/cuda/DistanceKernel.cu:195` (`cdist_kernel_cuda_impl`)
    // + the per-norm accumulate/finish in `dists<scalar_t>::{p,one,two,inf}`
    // (`:50-86`). Result stays GPU-resident — NO host round-trip. The f32
    // kernel covers `p in {1, 2, inf}` and general `p`; the f64 kernel covers
    // `p in {1, 2, inf}`. Unsupported (op, dtype, p) combinations (e.g. the
    // `p == 0` count-norm) surface as `NotImplementedOnCuda` rather than a
    // silent host fallback, matching the rest of `tensor_ops`'s GPU contract.
    if x1.is_cuda() || x2.is_cuda() {
        if !x1.is_cuda() || !x2.is_cuda() {
            return Err(FerrotorchError::InvalidArgument {
                message: "cdist: x1 and x2 must be on the same device".into(),
            });
        }
        if let Some(backend) = crate::gpu_dispatch::gpu_backend() {
            // #1658: normalise BOTH narrowed-offset CUDA operands to packed
            // offset-0 buffers before the distance kernel reads element 0.
            let x1 = x1.contiguous()?;
            let x2 = x2.contiguous()?;
            let handle = if is_f32::<T>() && crate::gpu_dispatch::cdist_supported_f32(p) {
                Some(backend.cdist_f32(
                    x1.gpu_handle()?,
                    x2.gpu_handle()?,
                    b,
                    p_dim,
                    r_dim,
                    m,
                    p,
                )?)
            } else if is_f64::<T>() && crate::gpu_dispatch::cdist_supported_f64(p) {
                Some(backend.cdist_f64(
                    x1.gpu_handle()?,
                    x2.gpu_handle()?,
                    b,
                    p_dim,
                    r_dim,
                    m,
                    p,
                )?)
            } else {
                None
            };
            if let Some(handle) = handle {
                let storage = TensorStorage::gpu(handle);
                return Tensor::from_storage(storage, out_shape, false);
            }
        }
        return Err(FerrotorchError::NotImplementedOnCuda { op: "cdist" });
    }

    let d1 = x1.data()?;
    let d2 = x2.data()?;
    let p_val = T::from(p).unwrap();
    let inv_p = T::from(1.0 / p).unwrap();
    let mut out = Vec::with_capacity(b * p_dim * r_dim);

    for batch in 0..b {
        let off1 = batch * p_dim * m;
        let off2 = batch * r_dim * m;
        for i in 0..p_dim {
            for j in 0..r_dim {
                let mut dist = <T as num_traits::Zero>::zero();
                for k in 0..m {
                    let diff = d1[off1 + i * m + k] - d2[off2 + j * m + k];
                    let abs_diff = if diff < <T as num_traits::Zero>::zero() {
                        <T as num_traits::Zero>::zero() - diff
                    } else {
                        diff
                    };
                    dist += abs_diff.powf(p_val);
                }
                out.push(dist.powf(inv_p));
            }
        }
    }

    Tensor::from_storage(TensorStorage::cpu(out), out_shape, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t2d(data: &[f32], rows: usize, cols: usize) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![rows, cols], false).unwrap()
    }

    fn t1d(data: &[f32]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false).unwrap()
    }

    #[test]
    fn test_triu_main_diagonal() {
        let input = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3);
        let result = triu(&input, 0).unwrap();
        assert_eq!(
            result.data().unwrap(),
            &[1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0]
        );
    }

    #[test]
    fn test_tril_main_diagonal() {
        let input = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3);
        let result = tril(&input, 0).unwrap();
        assert_eq!(
            result.data().unwrap(),
            &[1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0]
        );
    }

    #[test]
    fn test_triu_positive_diagonal() {
        let input = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3);
        let result = triu(&input, 1).unwrap();
        assert_eq!(
            result.data().unwrap(),
            &[0.0, 2.0, 3.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0]
        );
    }

    /// Batched N-D triu (CPU): the upper-triangular mask is applied per
    /// trailing `[3,3]` matrix over a `[2,2,3,3]` tensor, batching over the two
    /// leading dims. Mirrors `torch.triu` (`TriangularOps.cpp:31` `dim() >= 2`,
    /// batched per `cuda/TriangularOps.cu:120`). Expected vector from LIVE
    /// torch: `torch.triu(arange(36).reshape(2,2,3,3), 0).flatten()`.
    #[test]
    fn test_triu_batched_4d_cpu() {
        let data: Vec<f32> = (0..36).map(|i| i as f32).collect();
        let input =
            Tensor::from_storage(TensorStorage::cpu(data), vec![2, 2, 3, 3], false).unwrap();
        let result = triu(&input, 0).unwrap();
        assert_eq!(result.shape(), &[2, 2, 3, 3]);
        assert_eq!(
            result.data().unwrap(),
            &[
                0.0, 1.0, 2.0, 0.0, 4.0, 5.0, 0.0, 0.0, 8.0, // batch [0,0]
                9.0, 10.0, 11.0, 0.0, 13.0, 14.0, 0.0, 0.0, 17.0, // batch [0,1]
                18.0, 19.0, 20.0, 0.0, 22.0, 23.0, 0.0, 0.0, 26.0, // batch [1,0]
                27.0, 28.0, 29.0, 0.0, 31.0, 32.0, 0.0, 0.0, 35.0, // batch [1,1]
            ]
        );
    }

    /// Batched N-D tril (CPU): lower-triangular mask per trailing `[3,3]`
    /// matrix over a `[2,3,3]` tensor. Expected from LIVE torch:
    /// `torch.tril(arange(18).reshape(2,3,3), 0).flatten()`.
    #[test]
    fn test_tril_batched_3d_cpu() {
        let data: Vec<f32> = (0..18).map(|i| i as f32).collect();
        let input = Tensor::from_storage(TensorStorage::cpu(data), vec![2, 3, 3], false).unwrap();
        let result = tril(&input, 0).unwrap();
        assert_eq!(result.shape(), &[2, 3, 3]);
        assert_eq!(
            result.data().unwrap(),
            &[
                0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 6.0, 7.0, 8.0, // batch 0
                9.0, 0.0, 0.0, 12.0, 13.0, 0.0, 15.0, 16.0, 17.0, // batch 1
            ]
        );
    }

    /// `triu`/`tril` reject sub-2-D input (mirrors `TriangularOps.cpp:25,31`
    /// `TORCH_CHECK(self.dim() >= 2, ...)`).
    #[test]
    fn test_triu_tril_reject_1d() {
        let input = t1d(&[1.0, 2.0, 3.0]);
        assert!(triu(&input, 0).is_err());
        assert!(tril(&input, 0).is_err());
    }

    #[test]
    fn test_diag_extract() {
        let input = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3);
        let result = diag(&input, 0).unwrap();
        assert_eq!(result.data().unwrap(), &[1.0, 5.0, 9.0]);
    }

    #[test]
    fn test_diag_construct() {
        let input = t1d(&[1.0, 2.0, 3.0]);
        let result = diag(&input, 0).unwrap();
        assert_eq!(result.shape(), &[3, 3]);
        assert_eq!(
            result.data().unwrap(),
            &[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]
        );
    }

    #[test]
    fn test_diag_off_diagonal() {
        let input = t1d(&[1.0, 2.0]);
        let result = diag(&input, 1).unwrap();
        assert_eq!(result.shape(), &[3, 3]);
        assert_eq!(
            result.data().unwrap(),
            &[0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn test_roll_basic() {
        let input = t1d(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = roll(&input, 2, 0).unwrap();
        assert_eq!(result.data().unwrap(), &[4.0, 5.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_roll_negative() {
        let input = t1d(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = roll(&input, -1, 0).unwrap();
        assert_eq!(result.data().unwrap(), &[2.0, 3.0, 4.0, 5.0, 1.0]);
    }

    #[test]
    fn test_cdist_l2() {
        let x1 = t2d(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0], 3, 2);
        let x2 = t2d(&[1.0, 1.0], 1, 2);
        let result = cdist(&x1, &x2, 2.0).unwrap();
        assert_eq!(result.shape(), &[3, 1]);
        let d = result.data().unwrap();
        assert!((d[0] - 2.0f32.sqrt()).abs() < 1e-5); // dist([0,0],[1,1]) = sqrt(2)
        assert!((d[1] - 1.0).abs() < 1e-5); // dist([1,0],[1,1]) = 1
        assert!((d[2] - 1.0).abs() < 1e-5); // dist([0,1],[1,1]) = 1
    }

    #[test]
    // reason: diagflat is pure indexing — each input element is copied to
    // a diagonal slot without arithmetic, so the bit pattern is preserved
    // and equality is the right check.
    #[allow(clippy::float_cmp)]
    fn test_diagflat() {
        let input = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let result = diagflat(&input, 0).unwrap();
        assert_eq!(result.shape(), &[4, 4]);
        let d = result.data().unwrap();
        assert_eq!(d[0], 1.0);
        assert_eq!(d[5], 2.0);
        assert_eq!(d[10], 3.0);
        assert_eq!(d[15], 4.0);
    }
}
