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
use crate::dtype::{DType, Element, Float};
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::storage::TensorStorage;
use crate::tensor::{GradFn, Tensor};

#[inline]
fn is_f32<T: Float>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f32>()
}

#[inline]
fn is_f64<T: Float>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f64>()
}

fn ensure_packed_cuda_f32_f64<T: Float>(
    input: &Tensor<T>,
    backend: &dyn crate::gpu_dispatch::GpuBackend,
    op: &'static str,
) -> FerrotorchResult<Tensor<T>> {
    if !(is_f32::<T>() || is_f64::<T>()) {
        return Err(FerrotorchError::NotImplementedOnCuda { op });
    }
    if input.is_contiguous() && input.storage_offset() == 0 && input.storage_len() == input.numel()
    {
        return Ok(input.clone());
    }
    if input.ndim() > 8 {
        return Err(FerrotorchError::NotImplementedOnCuda { op });
    }

    let shape = input.shape().to_vec();
    let strides = input.strides().to_vec();
    let offset = input.storage_offset();
    let handle = if is_f32::<T>() {
        backend.strided_copy_f32(input.gpu_handle()?, &shape, &strides, offset)?
    } else {
        backend.strided_copy_f64(input.gpu_handle()?, &shape, &strides, offset)?
    };
    Tensor::from_storage(TensorStorage::gpu(handle), shape, false)
}

fn ensure_packed_cuda_float<T: Float>(
    input: &Tensor<T>,
    backend: &dyn crate::gpu_dispatch::GpuBackend,
    op: &'static str,
) -> FerrotorchResult<Tensor<T>> {
    if !matches!(
        <T as Element>::dtype(),
        DType::F32 | DType::F64 | DType::F16 | DType::BF16
    ) {
        return Err(FerrotorchError::NotImplementedOnCuda { op });
    }
    if input.is_contiguous() && input.storage_offset() == 0 && input.storage_len() == input.numel()
    {
        return Ok(input.clone());
    }
    if input.ndim() > 8 {
        return Err(FerrotorchError::NotImplementedOnCuda { op });
    }

    let shape = input.shape().to_vec();
    let strides = input.strides().to_vec();
    let offset = input.storage_offset();
    let handle = match <T as Element>::dtype() {
        DType::F32 => backend.strided_copy_f32(input.gpu_handle()?, &shape, &strides, offset)?,
        DType::F64 => backend.strided_copy_f64(input.gpu_handle()?, &shape, &strides, offset)?,
        DType::F16 | DType::BF16 => {
            backend.strided_copy_u16(input.gpu_handle()?, &shape, &strides, offset)?
        }
        _ => unreachable!("dtype checked above"),
    };
    Tensor::from_storage(TensorStorage::gpu(handle), shape, false)
}

fn reshape_for_cdist<T: Float>(input: &Tensor<T>, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
    let shape_isize: Vec<isize> = shape
        .iter()
        .map(|&d| {
            isize::try_from(d).map_err(|_| FerrotorchError::InvalidArgument {
                message: format!("cdist: shape {shape:?} dimension overflows isize"),
            })
        })
        .collect::<FerrotorchResult<_>>()?;
    crate::grad_fns::shape::reshape(input, &shape_isize)
}

struct CdistPrepared<T: Float> {
    x1: Tensor<T>,
    x2: Tensor<T>,
    out_shape: Vec<usize>,
    batch: usize,
    p_dim: usize,
    r_dim: usize,
    m: usize,
}

fn normalize_cdist_inputs<T: Float>(
    x1: &Tensor<T>,
    x2: &Tensor<T>,
) -> FerrotorchResult<CdistPrepared<T>> {
    if x1.ndim() < 2 || x2.ndim() < 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "cdist: expected inputs with at least 2 dimensions, got {:?} and {:?}",
                x1.shape(),
                x2.shape()
            ),
        });
    }

    let s1 = x1.shape();
    let s2 = x2.shape();
    let m1 = s1[s1.len() - 1];
    let m2 = s2[s2.len() - 1];
    if m1 != m2 {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!("cdist: feature dims mismatch: {m1} vs {m2}"),
        });
    }

    let p_dim = s1[s1.len() - 2];
    let r_dim = s2[s2.len() - 2];
    let batch = crate::shape::broadcast_shapes(&s1[..s1.len() - 2], &s2[..s2.len() - 2])?;
    let b = batch.iter().product();
    let out_shape = if s1.len() == 2 && s2.len() == 2 {
        vec![p_dim, r_dim]
    } else {
        let mut shape = batch.clone();
        shape.push(p_dim);
        shape.push(r_dim);
        shape
    };

    let mut x1_view_shape = batch.clone();
    x1_view_shape.push(p_dim);
    x1_view_shape.push(m1);
    let mut x2_view_shape = batch.clone();
    x2_view_shape.push(r_dim);
    x2_view_shape.push(m1);

    let x1_base = if s1.len() == 2 {
        let mut shape = vec![1; batch.len()];
        shape.push(p_dim);
        shape.push(m1);
        reshape_for_cdist(x1, &shape)?
    } else {
        x1.clone()
    };
    let x2_base = if s2.len() == 2 {
        let mut shape = vec![1; batch.len()];
        shape.push(r_dim);
        shape.push(m1);
        reshape_for_cdist(x2, &shape)?
    } else {
        x2.clone()
    };

    let x1_exp = crate::grad_fns::shape::expand(&x1_base, &x1_view_shape)?;
    let x2_exp = crate::grad_fns::shape::expand(&x2_base, &x2_view_shape)?;
    let x1_3d = reshape_for_cdist(&x1_exp, &[b, p_dim, m1])?;
    let x2_3d = reshape_for_cdist(&x2_exp, &[b, r_dim, m1])?;

    Ok(CdistPrepared {
        x1: x1_3d,
        x2: x2_3d,
        out_shape,
        batch: b,
        p_dim,
        r_dim,
        m: m1,
    })
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

    // GPU fast path: resident kernel for every floating dtype supported by
    // Tensor<T> (f32/f64/f16/bf16). f16 and bf16 use raw u16 payload kernels and
    // keep the dtype tag in `GpuBufferHandle`.
    // One thread per element; predicate `col - row >= diagonal` mirrors
    // `aten/src/ATen/native/cuda/TriangularOps.cu:100`, batched per trailing
    // `[rows, cols]` matrix per `:120`. Result stays GPU-resident — NO host
    // round-trip. Other GPU dtypes keep the `NotImplementedOnCuda` contract.
    if input.is_cuda() {
        if let Some(backend) = crate::gpu_dispatch::gpu_backend() {
            let input = ensure_packed_cuda_float(input, backend, "triu")?;
            let handle = match <T as Element>::dtype() {
                DType::F32 => backend.triu_f32(input.gpu_handle()?, batch, rows, cols, diagonal)?,
                DType::F64 => backend.triu_f64(input.gpu_handle()?, batch, rows, cols, diagonal)?,
                DType::F16 | DType::BF16 => {
                    backend.triu_u16(input.gpu_handle()?, batch, rows, cols, diagonal)?
                }
                _ => return Err(FerrotorchError::NotImplementedOnCuda { op: "triu" }),
            };
            let storage = TensorStorage::gpu(handle);
            return Tensor::from_storage(storage, shape.to_vec(), false);
        }
        return Err(FerrotorchError::NotImplementedOnCuda { op: "triu" });
    }

    let data = input.data_vec()?;
    let zero = <T as num_traits::Zero>::zero();

    let matrix = rows * cols;
    let mut out = Vec::with_capacity(batch * matrix);
    for b in 0..batch {
        let base = b * matrix;
        for r in 0..rows {
            for c in 0..cols {
                // `c - r >= diagonal` evaluated as a DIFFERENCE: both indices
                // are in-bounds (< isize::MAX), so the difference always fits
                // i64, whereas `r + diagonal` overflows for offsets within
                // `rows` of i64::MAX/MIN (CORE-121 / #1815). Mirrors the PTX
                // kernel's `diff = col - row` compare (`triangular.rs`).
                if (c as i64) - (r as i64) >= diagonal {
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

    // GPU fast path: resident kernel for f32/f64/f16/bf16. f16 and bf16 use raw
    // u16 payload kernels and keep the dtype tag in `GpuBufferHandle`.
    // Predicate `col - row <= diagonal` mirrors
    // `aten/src/ATen/native/cuda/TriangularOps.cu:100`, batched per trailing
    // `[rows, cols]` matrix per `:120`. Result stays GPU-resident — NO host
    // round-trip.
    if input.is_cuda() {
        if let Some(backend) = crate::gpu_dispatch::gpu_backend() {
            let input = ensure_packed_cuda_float(input, backend, "tril")?;
            let handle = match <T as Element>::dtype() {
                DType::F32 => backend.tril_f32(input.gpu_handle()?, batch, rows, cols, diagonal)?,
                DType::F64 => backend.tril_f64(input.gpu_handle()?, batch, rows, cols, diagonal)?,
                DType::F16 | DType::BF16 => {
                    backend.tril_u16(input.gpu_handle()?, batch, rows, cols, diagonal)?
                }
                _ => return Err(FerrotorchError::NotImplementedOnCuda { op: "tril" }),
            };
            let storage = TensorStorage::gpu(handle);
            return Tensor::from_storage(storage, shape.to_vec(), false);
        }
        return Err(FerrotorchError::NotImplementedOnCuda { op: "tril" });
    }

    let data = input.data_vec()?;
    let zero = <T as num_traits::Zero>::zero();

    let matrix = rows * cols;
    let mut out = Vec::with_capacity(batch * matrix);
    for b in 0..batch {
        let base = b * matrix;
        for r in 0..rows {
            for c in 0..cols {
                // Overflow-free difference form — see the `triu` mask note
                // (CORE-121 / #1815).
                if (c as i64) - (r as i64) <= diagonal {
                    out.push(data[base + r * cols + c]);
                } else {
                    out.push(zero);
                }
            }
        }
    }

    Tensor::from_storage(TensorStorage::cpu(out), shape.to_vec(), false)
}

/// Checked `[size, size]` sizing for the 1-D `diag` embed
/// (`size = n + |diagonal|`).
///
/// Returns a structured `InvalidArgument` when the side, the element count
/// `size * size`, or the byte size is unrepresentable — the analog of torch's
/// `RuntimeError: Storage size calculation overflowed with sizes=[...]`
/// (live torch 2.11.0 raises for `torch.diag(torch.ones(2), 2**62)`).
/// Pre-fix the unchecked `n + |k|` / `size * size` wrapped and then scattered
/// out of bounds on CPU or dispatched a CUDA kernel sized from the wrapped
/// count (CORE-121 / #1815).
fn diag_embed_size<T: Float>(n: usize, diagonal: i64) -> FerrotorchResult<usize> {
    let overflow = || FerrotorchError::InvalidArgument {
        message: format!(
            "diag: storage size calculation overflowed with sizes=[n={n} + |diagonal={diagonal}|]^2"
        ),
    };
    let offset = usize::try_from(diagonal.unsigned_abs()).map_err(|_| overflow())?;
    let size = n.checked_add(offset).ok_or_else(overflow)?;
    let total = size.checked_mul(size).ok_or_else(overflow)?;
    // Byte-size representability: Rust allocations are capped at isize::MAX
    // bytes (a `Vec` past that panics with "capacity overflow" — still a
    // panic inside a fallible API), so reject it here like torch rejects its
    // int64 storage-size overflow.
    match total.checked_mul(std::mem::size_of::<T>()) {
        Some(bytes) if isize::try_from(bytes).is_ok() => Ok(size),
        _ => Err(overflow()),
    }
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

    // GPU fast path: resident kernel for f32/f64/f16/bf16. f16 and bf16 copy raw
    // 16-bit payloads and preserve the dtype tag.
    // 1-D → `diag_embed` scatter onto the k-th diagonal of a `[size, size]`
    // matrix (`size = n + |k|`); 2-D → `diag_extract` gather of the k-th
    // diagonal. Both mirror `torch.diag` (`TensorShape.cpp:4610`); pure
    // gather/scatter, so the GPU result is bit-identical to the CPU path and
    // stays GPU-resident — NO host round-trip. Other GPU dtypes keep the
    // `NotImplementedOnCuda` contract.
    if input.is_cuda() {
        if let Some(backend) = crate::gpu_dispatch::gpu_backend() {
            let input = ensure_packed_cuda_float(input, backend, "diag")?;
            let (handle, out_shape) = if ndim == 1 {
                let n = input.shape()[0];
                // Checked BEFORE kernel dispatch: pre-fix a wrapped size
                // reached the backend and sized the scatter kernel from the
                // wrapped count (CORE-121 / #1815).
                let size = diag_embed_size::<T>(n, diagonal)?;
                let handle = match <T as Element>::dtype() {
                    DType::F32 => backend.diag_embed_f32(input.gpu_handle()?, n, diagonal)?,
                    DType::F64 => backend.diag_embed_f64(input.gpu_handle()?, n, diagonal)?,
                    DType::F16 | DType::BF16 => {
                        backend.diag_embed_u16(input.gpu_handle()?, n, diagonal)?
                    }
                    _ => return Err(FerrotorchError::NotImplementedOnCuda { op: "diag" }),
                };
                (handle, vec![size, size])
            } else {
                let rows = input.shape()[0];
                let cols = input.shape()[1];
                let (start_r, start_c) = if diagonal >= 0 {
                    (0usize, diagonal as usize)
                } else {
                    // `unsigned_abs`, NOT negation: i64::MIN is unnegatable
                    // (CORE-121 / #1815).
                    (diagonal.unsigned_abs() as usize, 0usize)
                };
                let diag_len = rows
                    .saturating_sub(start_r)
                    .min(cols.saturating_sub(start_c));
                let handle = match <T as Element>::dtype() {
                    DType::F32 => {
                        backend.diag_extract_f32(input.gpu_handle()?, rows, cols, diagonal)?
                    }
                    DType::F64 => {
                        backend.diag_extract_f64(input.gpu_handle()?, rows, cols, diagonal)?
                    }
                    DType::F16 | DType::BF16 => {
                        backend.diag_extract_u16(input.gpu_handle()?, rows, cols, diagonal)?
                    }
                    _ => return Err(FerrotorchError::NotImplementedOnCuda { op: "diag" }),
                };
                (handle, vec![diag_len])
            };
            let storage = TensorStorage::gpu(handle);
            return Tensor::from_storage(storage, out_shape, false);
        }
        return Err(FerrotorchError::NotImplementedOnCuda { op: "diag" });
    }

    match ndim {
        1 => {
            // 1-D → 2-D diagonal matrix
            let data = input.data_vec()?;
            let n = data.len();
            // Checked sizing — unrepresentable `[n + |k|]^2` is a structured
            // error, never a wrapped allocation (CORE-121 / #1815).
            let size = diag_embed_size::<T>(n, diagonal)?;
            let offset = size - n;
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
            let data = input.data_vec()?;

            let (start_r, start_c) = if diagonal >= 0 {
                (0, diagonal as usize)
            } else {
                // `unsigned_abs`, NOT negation: i64::MIN is unnegatable
                // (CORE-121 / #1815).
                (diagonal.unsigned_abs() as usize, 0)
            };

            // Saturating bounds: an offset at/beyond the matrix edge selects
            // an EMPTY diagonal (mirrors `torch.diag` length clamping,
            // `aten/src/ATen/native/TensorShape.cpp` `apply_diag`
            // `sz = std::max<int64_t>(0, ...)`), matching the CUDA branch
            // above. Pre-fix the unchecked subtraction underflowed and drove
            // out-of-bounds indexing (CORE-120 / #1814).
            let diag_len = rows
                .saturating_sub(start_r)
                .min(cols.saturating_sub(start_c));
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
            let input_packed = ensure_packed_cuda_f32_f64(input, backend, "roll")?;
            let outer: usize = shape[..dim].iter().product();
            let inner: usize = shape[dim + 1..].iter().product();
            let handle = if is_f32::<T>() {
                backend.roll_f32(
                    input_packed.gpu_handle()?,
                    outer,
                    shape[dim],
                    inner,
                    shift_norm as usize,
                )?
            } else {
                backend.roll_f64(
                    input_packed.gpu_handle()?,
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
/// `x1` has shape `[..., P, M]`, `x2` has shape `[..., R, M]`.
/// Leading batch dimensions are broadcast exactly like PyTorch, and the result
/// has shape `[..., P, R]`.
///
/// If `x1` is 2-D `[P, M]` and `x2` is 2-D `[R, M]`, returns `[P, R]`.
///
/// Matches PyTorch's `torch.cdist`.
pub fn cdist<T: Float>(x1: &Tensor<T>, x2: &Tensor<T>, p: f64) -> FerrotorchResult<Tensor<T>> {
    // Input contract BEFORE any device dispatch (mirrors
    // `aten/src/ATen/native/Distance.cpp` `TORCH_CHECK(p >= 0, "cdist only
    // supports non-negative p values")`); pre-fix both devices accepted a
    // negative `p` into the norm formula/kernel (CORE-122 / #1816).
    if p < 0.0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("cdist only supports non-negative p values, got {p}"),
        });
    }

    let prepared = normalize_cdist_inputs(x1, x2)?;

    // GPU fast path: f32/f64 resident kernel (crosslink #1545 / sub #1535).
    // Both inputs must be CUDA-resident on the same device. Mirrors
    // `aten/src/ATen/native/cuda/DistanceKernel.cu:195` (`cdist_kernel_cuda_impl`)
    // + the per-norm accumulate/finish in `dists<scalar_t>::{p,one,two,inf}`
    // (`:50-86`). Result stays GPU-resident — NO host round-trip. The f32
    // kernel covers `p in {1, 2, inf}` and general `p`; the f64 kernel covers
    // `p in {1, 2, inf}`. Unsupported (op, dtype, p) combinations (e.g. the
    // `p == 0` count-norm) surface as `NotImplementedOnCuda` rather than a
    // silent host fallback, matching the rest of `tensor_ops`'s GPU contract.
    if prepared.x1.is_cuda() || prepared.x2.is_cuda() {
        // EXACT device equality — ordinal included — before ANY backend
        // access (CORE-124 / #1818). Mirrors `aten/src/ATen/native/
        // Distance.cpp` `cdist_impl`'s operand-device TORCH_CHECK: a
        // CPU×CUDA mix and a cross-ordinal CUDA pair both refuse here;
        // pre-fix the `is_cuda()`-only pair check let operands on different
        // GPU ordinals reach one kernel with a pointer owned by another
        // device.
        if prepared.x1.device() != prepared.x2.device() {
            return Err(FerrotorchError::DeviceMismatch {
                expected: prepared.x1.device(),
                got: prepared.x2.device(),
            });
        }
        if let Some(backend) = crate::gpu_dispatch::gpu_backend() {
            let x1_packed = ensure_packed_cuda_f32_f64(&prepared.x1, backend, "cdist")?;
            let x2_packed = ensure_packed_cuda_f32_f64(&prepared.x2, backend, "cdist")?;
            let handle = if is_f32::<T>() && crate::gpu_dispatch::cdist_supported_f32(p) {
                Some(backend.cdist_f32(
                    x1_packed.gpu_handle()?,
                    x2_packed.gpu_handle()?,
                    prepared.batch,
                    prepared.p_dim,
                    prepared.r_dim,
                    prepared.m,
                    p,
                )?)
            } else if is_f64::<T>() && crate::gpu_dispatch::cdist_supported_f64(p) {
                Some(backend.cdist_f64(
                    x1_packed.gpu_handle()?,
                    x2_packed.gpu_handle()?,
                    prepared.batch,
                    prepared.p_dim,
                    prepared.r_dim,
                    prepared.m,
                    p,
                )?)
            } else {
                None
            };
            if let Some(handle) = handle {
                let storage = TensorStorage::gpu(handle);
                // Autograd edge for CUDA results too (the forward stays
                // GPU-resident; `CdistBackward` documents its host round
                // trip). Pre-fix the GPU result was silently detached
                // (CORE-123 / #1817).
                return if is_grad_enabled()
                    && (prepared.x1.requires_grad() || prepared.x2.requires_grad())
                {
                    let grad_fn = Arc::new(CdistBackward {
                        x1: prepared.x1,
                        x2: prepared.x2,
                        p,
                    });
                    Tensor::from_operation(storage, prepared.out_shape, grad_fn)
                } else {
                    Tensor::from_storage(storage, prepared.out_shape, false)
                };
            }
        }
        return Err(FerrotorchError::NotImplementedOnCuda { op: "cdist" });
    }

    let d1 = prepared.x1.data_vec()?;
    let d2 = prepared.x2.data_vec()?;
    let out = cdist_cpu_distances(
        &d1,
        &d2,
        prepared.batch,
        prepared.p_dim,
        prepared.r_dim,
        prepared.m,
        norm_for_p::<T>(p),
    );

    let storage = TensorStorage::cpu(out);
    if is_grad_enabled() && (prepared.x1.requires_grad() || prepared.x2.requires_grad()) {
        let grad_fn = Arc::new(CdistBackward {
            x1: prepared.x1,
            x2: prepared.x2,
            p,
        });
        Tensor::from_operation(storage, prepared.out_shape, grad_fn)
    } else {
        Tensor::from_storage(storage, prepared.out_shape, false)
    }
}

/// Norm selector for the explicit `cdist` branch dispatch, mirroring the
/// upstream per-norm accumulate/finish structs
/// (`aten/src/ATen/native/cuda/DistanceKernel.cu`
/// `dists<scalar_t>::{zero, one, two, inf, p}`, selected at `:232-238` with
/// exact `p ==` compares; the CPU path `aten/src/ATen/native/Distance.cpp`
/// dispatches identically). Pre-CORE-122 every `p` ran the generic
/// `sum(|d|^p)^(1/p)` formula, which is wrong for the `p == 0` count-"norm"
/// (counted EQUAL coordinates too, then `sum^inf`) and for `p == inf`
/// (max |d|) — and disagreed with the GPU kernels that already branch
/// (CORE-122 / #1816).
#[derive(Clone, Copy)]
enum Norm<T> {
    Zero,
    One,
    Two,
    Inf,
    /// General finite p: carries (p, p - 1, 1/p) as T.
    P(T, T, T),
}

// reason: `p` is a discrete norm selector, not a measured value — the exact
// compares mirror the upstream norm dispatch (see
// `gpu_dispatch::cdist_supported_f32`).
#[allow(
    clippy::float_cmp,
    reason = "discrete norm selector, mirrors upstream exact p compares"
)]
fn norm_for_p<T: Float>(p: f64) -> Norm<T> {
    if p == 0.0 {
        Norm::Zero
    } else if p == 1.0 {
        Norm::One
    } else if p == 2.0 {
        Norm::Two
    } else if p.is_infinite() {
        Norm::Inf
    } else {
        Norm::P(
            T::from(p).unwrap(),
            T::from(p - 1.0).unwrap(),
            T::from(1.0 / p).unwrap(),
        )
    }
}

/// CPU pairwise-distance kernel shared by the `cdist` forward and the
/// distance recomputation inside [`CdistBackward`]. Layouts are the packed
/// `[b, p_dim, m]` / `[b, r_dim, m]` row-major buffers; output is the packed
/// `[b, p_dim, r_dim]` distance buffer.
fn cdist_cpu_distances<T: Float>(
    d1: &[T],
    d2: &[T],
    b: usize,
    p_dim: usize,
    r_dim: usize,
    m: usize,
    norm: Norm<T>,
) -> Vec<T> {
    let zero = <T as num_traits::Zero>::zero();
    let one = <T as num_traits::One>::one();
    let mut out = Vec::with_capacity(b * p_dim * r_dim);
    for batch in 0..b {
        let off1 = batch * p_dim * m;
        let off2 = batch * r_dim * m;
        for i in 0..p_dim {
            for j in 0..r_dim {
                let mut acc = zero;
                for k in 0..m {
                    let diff = d1[off1 + i * m + k] - d2[off2 + j * m + k];
                    let abs_diff = diff.abs();
                    match norm {
                        // Zero-"norm": `min(ceil(|d|), 1)` summed — NOT a
                        // `!= 0` count (identical for finite diffs, but NaN
                        // must PROPAGATE: `ceil(NaN) = NaN` and C++ `min`
                        // returns its NaN first operand) — upstream
                        // `DistanceOpsKernel.cpp:94` / `DistanceKernel.cu`
                        // `dists::zero`. Rust's `min` would return the
                        // non-NaN operand, so clamp via comparison: `NaN > 1`
                        // is false, letting the NaN flow into the sum
                        // (discriminator pin
                        // `divergence_tensor_ops_cdist_p0_nan.rs`).
                        Norm::Zero => {
                            let c = abs_diff.ceil();
                            acc += if c > one { one } else { c };
                        }
                        Norm::One => acc += abs_diff,
                        Norm::Two => acc += abs_diff * abs_diff,
                        // `dists::inf`: running max of |diff|.
                        Norm::Inf => acc = acc.max(abs_diff),
                        Norm::P(p_val, _, _) => acc += abs_diff.powf(p_val),
                    }
                }
                let dist = match norm {
                    Norm::Zero | Norm::One | Norm::Inf => acc,
                    Norm::Two => acc.sqrt(),
                    Norm::P(_, _, inv_p) => acc.powf(inv_p),
                };
                out.push(dist);
            }
        }
    }
    out
}

/// Backward for `cdist(x1, x2, p)` — differentiates BOTH point sets
/// (pre-CORE-123 every result was silently detached; CORE-123 / #1817).
///
/// VJP (upstream `tools/autograd/derivatives.yaml`:
/// `_cdist_forward -> _cdist_backward`; per-norm weights per
/// `aten/src/ATen/native/cuda/DistanceKernel.cu`
/// `dists::{zero, one, two, inf, p}::backward`, all probed against live
/// torch 2.11.0 — probe transcripts quoted in
/// `tests/audit_core123_cdist_backward.rs`):
///
/// ```text
/// w(d, dist) = 0                                  (p = 0: count-norm is
///                                                  piecewise constant)
///            | sign(d), sign(0) = 0               (p = 1)
///            | d / dist, 0 at dist == 0           (p = 2)
///            | sign(d) * (|d| == dist)            (p = inf: EVERY tied max)
///            | sign(d) * (|d|/dist)^(p-1),
///              0 at d == 0 or dist == 0           (general p, incl. p < 1)
/// grad_x1[b,i,k] = Σ_j g[b,i,j] · w;   grad_x2[b,j,k] = −Σ_i g[b,i,j] · w
/// ```
///
/// The pairwise distances are RECOMPUTED with the same [`Norm`] dispatch the
/// forward uses (bit-identical to the forward values on CPU), instead of
/// retaining the output tensor (which would create an `Arc` cycle through
/// its own grad_fn).
///
/// # Device
/// The gradient math runs on the host. For CUDA inputs this is an explicit
/// host round trip (D2H copies of `x1`/`x2`/`grad_output`, H2D upload of
/// both gradients back to the input device) — the VALUES are identical to a
/// resident implementation; only WHERE the backward computes differs
/// (documented per the module's R-LOUD-2 contract). The forward stays
/// GPU-resident.
#[derive(Debug)]
struct CdistBackward<T: Float> {
    x1: Tensor<T>,
    x2: Tensor<T>,
    p: f64,
}

impl<T: Float> GradFn<T> for CdistBackward<T> {
    // reason: `dist == 0` / `|d| == dist` are the upstream guard semantics —
    // `dist` is recomputed bit-identically from the same inputs, so the exact
    // compares select exactly the coordinates torch's backward selects
    // (`DistanceKernel.cu` `dists::inf::backward` compares `device(diff) ==
    // dist` exactly).
    #[allow(
        clippy::float_cmp,
        reason = "upstream-exact zero/max guards on bit-identical recomputed distances"
    )]
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !is_grad_enabled() || (!self.x1.requires_grad() && !self.x2.requires_grad()) {
            return Ok(vec![None, None]);
        }

        // Host copies (documented round trip for CUDA inputs — see the
        // struct-level `# Device` note).
        let x1_host = if self.x1.is_cpu() {
            self.x1.clone()
        } else {
            self.x1.cpu()?
        };
        let x2_host = if self.x2.is_cpu() {
            self.x2.clone()
        } else {
            self.x2.cpu()?
        };
        let g_host = if grad_output.is_cpu() {
            grad_output.clone()
        } else {
            grad_output.cpu()?
        };

        let s1 = self.x1.shape();
        let s2 = self.x2.shape();
        let (b, p_dim, m) = (s1[0], s1[1], s1[2]);
        let r_dim = s2[1];

        let d1 = x1_host.data_vec()?;
        let d2 = x2_host.data_vec()?;
        let g = g_host.data_vec()?;
        let norm = norm_for_p::<T>(self.p);
        // Recompute the forward distances (same kernel ⇒ bit-identical).
        let dists = cdist_cpu_distances(&d1, &d2, b, p_dim, r_dim, m, norm);

        let zero = <T as num_traits::Zero>::zero();
        let one = <T as num_traits::One>::one();
        let sign = |d: T| {
            if d > zero {
                one
            } else if d < zero {
                -one
            } else {
                zero
            }
        };

        let mut g1 = vec![zero; b * p_dim * m];
        let mut g2 = vec![zero; b * r_dim * m];
        for batch in 0..b {
            let off1 = batch * p_dim * m;
            let off2 = batch * r_dim * m;
            let offd = batch * p_dim * r_dim;
            for i in 0..p_dim {
                for j in 0..r_dim {
                    let go = g[offd + i * r_dim + j];
                    let dist = dists[offd + i * r_dim + j];
                    for k in 0..m {
                        let diff = d1[off1 + i * m + k] - d2[off2 + j * m + k];
                        let w = match norm {
                            Norm::Zero => zero,
                            Norm::One => sign(diff),
                            Norm::Two => {
                                if dist == zero {
                                    zero
                                } else {
                                    diff / dist
                                }
                            }
                            // sign(0) = 0 also covers the all-zero pair
                            // (dist == 0 ⇒ every |d| == dist but sign is 0).
                            Norm::Inf => {
                                if diff.abs() == dist {
                                    sign(diff)
                                } else {
                                    zero
                                }
                            }
                            Norm::P(_, p_m1, _) => {
                                if dist == zero || diff == zero {
                                    zero
                                } else {
                                    sign(diff) * (diff.abs() / dist).powf(p_m1)
                                }
                            }
                        };
                        let contrib = go * w;
                        g1[off1 + i * m + k] += contrib;
                        g2[off2 + j * m + k] += -contrib;
                    }
                }
            }
        }

        // Re-materialize each gradient on its input's device (H2D upload for
        // CUDA leaves) so accumulation happens device-local (R-ORACLE-3).
        let to_input_device = |data: Vec<T>, input: &Tensor<T>| -> FerrotorchResult<Tensor<T>> {
            let t = Tensor::from_storage(TensorStorage::cpu(data), input.shape().to_vec(), false)?;
            if input.is_cpu() {
                Ok(t)
            } else {
                t.to(input.device())
            }
        };

        let grad1 = if self.x1.requires_grad() {
            Some(to_input_device(g1, &self.x1)?)
        } else {
            None
        };
        let grad2 = if self.x2.requires_grad() {
            Some(to_input_device(g2, &self.x2)?)
        } else {
            None
        };
        Ok(vec![grad1, grad2])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.x1, &self.x2]
    }

    fn name(&self) -> &'static str {
        "CdistBackward"
    }

    fn scalar_args(&self) -> Vec<f64> {
        vec![self.p]
    }
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
