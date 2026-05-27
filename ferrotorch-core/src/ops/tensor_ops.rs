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
//! | REQ-1 | SHIPPED | `triu` at `ops/tensor_ops.rs:28`; consumer: re-export `ferrotorch_core::triu` at `lib.rs:177` |
//! | REQ-2 | SHIPPED | `tril` at `ops/tensor_ops.rs:62`; consumer: re-export at `lib.rs:177` |
//! | REQ-3 | SHIPPED | `diag` at `ops/tensor_ops.rs:98`; consumer: re-export at `lib.rs:177` |
//! | REQ-4 | SHIPPED | `diagflat` at `ops/tensor_ops.rs:155`; consumer: re-export at `lib.rs:177` |
//! | REQ-5 | SHIPPED | `roll` at `ops/tensor_ops.rs:181`; consumer: re-export at `lib.rs:177`; `RollBackward` autograd |
//! | REQ-6 | SHIPPED | `cdist` at `ops/tensor_ops.rs:292`; consumer: re-export at `lib.rs:177` |
//! | REQ-7 | SHIPPED | `roll_cpu_inner` at `ops/tensor_ops.rs:259`; consumer: `grad_fns::shape::RollBackward::backward` at `grad_fns/shape.rs:1006` invokes `ops::tensor_ops::roll_cpu_inner` |

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

/// Upper triangular part of a 2-D tensor.
///
/// Elements below the `diagonal`-th diagonal are set to zero.
/// `diagonal=0` is the main diagonal, `diagonal>0` is above, `diagonal<0` is below.
///
/// Matches PyTorch's `torch.triu`.
///
/// # Backward
/// Autograd-aware (CPU): when grad tracking is active for `input`, this routes
/// through `crate::grad_fns::linalg::triu_differentiable` (the VJP masks the
/// upstream gradient by the kept upper triangle, per `triu -> grad.triu_symint`
/// at upstream `tools/autograd/derivatives.yaml:1809`).
pub fn triu<T: Float>(input: &Tensor<T>, diagonal: i64) -> FerrotorchResult<Tensor<T>> {
    if input.ndim() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("triu: expected 2-D tensor, got shape {:?}", input.shape()),
        });
    }
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "triu" });
    }

    // Autograd path: delegate to the differentiable wrapper, which computes
    // the forward inside `no_grad` (preventing re-entry here) and attaches
    // `TriangularBackward`.
    if is_grad_enabled() && input.requires_grad() {
        return crate::grad_fns::linalg::triu_differentiable(input, diagonal);
    }

    let rows = input.shape()[0];
    let cols = input.shape()[1];
    let data = input.data()?;
    let zero = <T as num_traits::Zero>::zero();

    let mut out = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            if (c as i64) >= (r as i64) + diagonal {
                out.push(data[r * cols + c]);
            } else {
                out.push(zero);
            }
        }
    }

    Tensor::from_storage(TensorStorage::cpu(out), vec![rows, cols], false)
}

/// Lower triangular part of a 2-D tensor.
///
/// Elements above the `diagonal`-th diagonal are set to zero.
///
/// Matches PyTorch's `torch.tril`.
///
/// # Backward
/// Autograd-aware (CPU): when grad tracking is active for `input`, this routes
/// through `crate::grad_fns::linalg::tril_differentiable` (the VJP masks the
/// upstream gradient by the kept lower triangle, per `tril -> grad.tril_symint`
/// at upstream `tools/autograd/derivatives.yaml:1805`).
pub fn tril<T: Float>(input: &Tensor<T>, diagonal: i64) -> FerrotorchResult<Tensor<T>> {
    if input.ndim() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("tril: expected 2-D tensor, got shape {:?}", input.shape()),
        });
    }
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "tril" });
    }

    // Autograd path: delegate to the differentiable wrapper, which computes
    // the forward inside `no_grad` (preventing re-entry here) and attaches
    // `TriangularBackward`.
    if is_grad_enabled() && input.requires_grad() {
        return crate::grad_fns::linalg::tril_differentiable(input, diagonal);
    }

    let rows = input.shape()[0];
    let cols = input.shape()[1];
    let data = input.data()?;
    let zero = <T as num_traits::Zero>::zero();

    let mut out = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            if (c as i64) <= (r as i64) + diagonal {
                out.push(data[r * cols + c]);
            } else {
                out.push(zero);
            }
        }
    }

    Tensor::from_storage(TensorStorage::cpu(out), vec![rows, cols], false)
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
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "diag" });
    }

    // Autograd path: delegate to the differentiable wrapper, which computes
    // the forward inside `no_grad` (preventing re-entry here) and attaches
    // `DiagBackward`.
    if is_grad_enabled() && input.requires_grad() {
        return crate::grad_fns::linalg::diag_differentiable(input, diagonal);
    }

    match input.ndim() {
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
        2 => {
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
        _ => Err(FerrotorchError::InvalidArgument {
            message: format!("diag: expected 1-D or 2-D tensor, got {:?}", input.shape()),
        }),
    }
}

/// Construct a diagonal matrix from a 1-D tensor (flattened if needed).
///
/// Like `diag` with a 1-D input, but first flattens multi-dimensional input.
///
/// Matches PyTorch's `torch.diagflat`.
pub fn diagflat<T: Float>(input: &Tensor<T>, diagonal: i64) -> FerrotorchResult<Tensor<T>> {
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "diagflat" });
    }

    let flat = if input.ndim() == 1 {
        input.clone()
    } else {
        let data = input.data_vec()?;
        let n = data.len();
        Tensor::from_storage(TensorStorage::cpu(data), vec![n], false)?
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

    // GPU fast path: f32 only — matches the f32-first dispatch shape used
    // by the cumulative scans (see `cumsum_forward`). Other dtypes fall
    // through to the existing NotImplementedOnCuda error so the contract
    // matches the rest of `tensor_ops`.
    if input.is_cuda() {
        if is_f32::<T>() {
            if let Some(backend) = crate::gpu_dispatch::gpu_backend() {
                let outer: usize = shape[..dim].iter().product();
                let inner: usize = shape[dim + 1..].iter().product();
                let handle = backend.roll_f32(
                    input.gpu_handle()?,
                    outer,
                    shape[dim],
                    inner,
                    shift_norm as usize,
                )?;
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
    if x1.is_cuda() || x2.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "cdist" });
    }

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

    let out_shape = if batched {
        vec![b, p_dim, r_dim]
    } else {
        vec![p_dim, r_dim]
    };

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
