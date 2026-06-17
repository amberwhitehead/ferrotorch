//! Helpers for propagating the meta device through tensor operations.
//!
//! When all inputs to an operation are on `Device::Meta`, the op produces
//! a meta tensor with the correct output shape and skips the data
//! computation entirely. When inputs are mixed (some meta, some real),
//! the op returns an error since meta tensors carry no data.
//!
//! Each helper returns:
//! - `Ok(Some(t))` — the inputs were all meta, here is the meta result
//! - `Ok(None)` — no inputs were meta, the caller should run the normal
//!   computation path
//! - `Err(e)` — inputs were mixed, or the requested op is invalid for the
//!   given shapes
//!
//! Op authors call these at the top of their implementations:
//!
//! ```ignore
//! pub fn add<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
//!     if let Some(out) = meta_propagate::binary_broadcast(a, b)? {
//!         return Ok(out);
//!     }
//!     // ... normal path ...
//! }
//! ```
//!
//! CL-500 builds on the meta device foundation in CL-395.
//!
//! ## REQ status (per `.design/ferrotorch-core/meta_propagate.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | impl `unary_same_shape`; non-test consumer `grad_fns::activation::{relu, sigmoid, tanh, gelu, silu, softmax}`. |
//! | REQ-2 | SHIPPED | impl `binary_broadcast`; non-test consumer `grad_fns::arithmetic::add` + every binary broadcast op. |
//! | REQ-3 | SHIPPED | impl `ternary_broadcast_shape` / `ternary_broadcast`; non-test consumer `grad_fns::arithmetic::{addcmul, addcdiv}`. |
//! | REQ-4 | SHIPPED | impl `reduce_dim`; non-test consumer `grad_fns::reduction::sum_dim` + `mean_dim`. |
//! | REQ-5 | SHIPPED | impl `reduce_all`; non-test consumer `grad_fns::reduction::sum_all` + `mean_all`, `prod_all`. |
//! | REQ-6 | SHIPPED | impl `matmul`; non-test consumer `ops::linalg::matmul`. |

use std::sync::Arc;

use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::shape::{broadcast_shapes, checked_numel};
use crate::storage::TensorStorage;
use crate::tensor::{GradFn, Tensor};

pub(crate) fn meta_tensor<T: Float>(shape: Vec<usize>) -> FerrotorchResult<Tensor<T>> {
    let numel = checked_numel(&shape, "meta_propagate::meta_tensor")?;
    Tensor::from_storage(TensorStorage::meta(numel), shape, false)
}

pub(crate) fn meta_operation<T: Float>(
    shape: Vec<usize>,
    grad_fn: Arc<dyn GradFn<T>>,
) -> FerrotorchResult<Tensor<T>> {
    let numel = checked_numel(&shape, "meta_propagate::meta_operation")?;
    Tensor::from_operation(TensorStorage::meta(numel), shape, grad_fn)
}

pub(crate) fn meta_operation_saving_output<T: Float, F>(
    shape: Vec<usize>,
    make_grad_fn: F,
) -> FerrotorchResult<Tensor<T>>
where
    F: FnOnce(Tensor<T>) -> FerrotorchResult<Arc<dyn GradFn<T>>>,
{
    let numel = checked_numel(&shape, "meta_propagate::meta_operation_saving_output")?;
    Tensor::from_operation_saving_output(TensorStorage::meta(numel), shape, make_grad_fn)
}

/// Meta-device fast path for unary ops that produce an output of the same
/// shape as the input (most elementwise activations, neg, abs, sqrt, etc.).
pub fn unary_same_shape<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Option<Tensor<T>>> {
    if input.is_meta() {
        Ok(Some(meta_tensor(input.shape().to_vec())?))
    } else {
        Ok(None)
    }
}

/// Meta-device fast path for binary broadcast ops (add, sub, mul, div, etc.).
///
/// Returns the broadcast meta output when both inputs are meta. Errors
/// when only one input is meta — there is no defined behavior for mixing
/// real and meta tensors in an op, since the real side has data the
/// meta side does not.
pub fn binary_broadcast<T: Float>(
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> FerrotorchResult<Option<Tensor<T>>> {
    match (a.is_meta(), b.is_meta()) {
        (true, true) => {
            let out_shape = broadcast_shapes(a.shape(), b.shape())?;
            Ok(Some(meta_tensor(out_shape)?))
        }
        (false, false) => Ok(None),
        _ => Err(FerrotorchError::DeviceMismatch {
            expected: a.device(),
            got: b.device(),
        }),
    }
}

pub(crate) fn ternary_broadcast_shape<T: Float>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    c: &Tensor<T>,
) -> FerrotorchResult<Vec<usize>> {
    let bc_shape = broadcast_shapes(b.shape(), c.shape())?;
    broadcast_shapes(a.shape(), &bc_shape)
}

/// Meta-device fast path for ternary broadcast ops (`addcmul`, `addcdiv`,
/// `where`, etc.).
///
/// Returns the broadcast meta output when all inputs are meta. Errors when
/// exactly some inputs are meta, matching PyTorch's device checks: a real
/// tensor cannot provide data for a meta-only computation and a meta tensor
/// cannot be read by a real-data kernel.
pub fn ternary_broadcast<T: Float>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    c: &Tensor<T>,
) -> FerrotorchResult<Option<Tensor<T>>> {
    match (a.is_meta(), b.is_meta(), c.is_meta()) {
        (true, true, true) => {
            let out_shape = ternary_broadcast_shape(a, b, c)?;
            Ok(Some(meta_tensor(out_shape)?))
        }
        (false, false, false) => Ok(None),
        _ => {
            let expected = crate::device::Device::Meta;
            let got = [a.device(), b.device(), c.device()]
                .into_iter()
                .find(|device| *device != expected)
                .unwrap_or(expected);
            Err(FerrotorchError::DeviceMismatch { expected, got })
        }
    }
}

/// Meta-device fast path for reductions over a single dimension.
///
/// Returns a meta tensor whose shape is the input shape with the given
/// dim removed (when `keepdim == false`) or reduced to size 1 (when
/// `keepdim == true`). Mirrors `sum_dim`/`mean_dim`/`max_dim` shape rules.
pub fn reduce_dim<T: Float>(
    input: &Tensor<T>,
    dim: i64,
    keepdim: bool,
) -> FerrotorchResult<Option<Tensor<T>>> {
    if !input.is_meta() {
        return Ok(None);
    }
    let ndim = input.ndim();
    if ndim == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "meta_propagate::reduce_dim: cannot reduce a scalar tensor".into(),
        });
    }
    let norm_dim = if dim < 0 {
        (ndim as i64 + dim) as usize
    } else {
        dim as usize
    };
    if norm_dim >= ndim {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "meta_propagate::reduce_dim: dim {dim} out of bounds for tensor with {ndim} dimensions"
            ),
        });
    }
    let mut out_shape: Vec<usize> = input.shape().to_vec();
    if keepdim {
        out_shape[norm_dim] = 1;
    } else {
        out_shape.remove(norm_dim);
    }
    Ok(Some(meta_tensor(out_shape)?))
}

/// Meta-device fast path for full reductions (sum, mean, prod) that
/// collapse the entire tensor to a scalar.
pub fn reduce_all<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Option<Tensor<T>>> {
    if input.is_meta() {
        // Scalar (0-D) shape.
        Ok(Some(meta_tensor(vec![])?))
    } else {
        Ok(None)
    }
}

/// Meta-device fast path for matmul-style ops following PyTorch's
/// shape rules:
///
/// - 1-D × 1-D → scalar (dot product)
/// - 2-D × 1-D → 1-D vector
/// - 1-D × 2-D → 1-D vector
/// - 2-D × 2-D → 2-D matrix
/// - otherwise, promote 1-D operands to synthetic row/column matrices,
///   broadcast the real batch dims, then squeeze those synthetic axes.
pub fn matmul<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Option<Tensor<T>>> {
    match (a.is_meta(), b.is_meta()) {
        (false, false) => return Ok(None),
        (true, true) => {}
        _ => {
            return Err(FerrotorchError::DeviceMismatch {
                expected: a.device(),
                got: b.device(),
            });
        }
    }

    let a_shape = a.shape();
    let b_shape = b.shape();
    let a_ndim = a_shape.len();
    let b_ndim = b_shape.len();

    if a_ndim == 0 || b_ndim == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "meta_propagate::matmul: scalar operands not supported, got {a_shape:?} and {b_shape:?}"
            ),
        });
    }

    // Mirrors PyTorch's `_matmul_impl` general shape rule: 1-D inputs
    // contribute no batch dims and synthesize a row/column matrix axis only
    // for the contraction. That synthetic axis is not emitted in `out_shape`.
    let lhs_rows = if a_ndim > 1 { a_shape[a_ndim - 2] } else { 1 };
    let lhs_contract = a_shape[a_ndim - 1];
    let rhs_contract = if b_ndim > 1 {
        b_shape[b_ndim - 2]
    } else {
        b_shape[0]
    };
    let rhs_cols = if b_ndim > 1 { b_shape[b_ndim - 1] } else { 1 };

    if lhs_contract != rhs_contract {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "meta_propagate::matmul: inner dimensions mismatch for shapes {a_shape:?} and {b_shape:?}: {lhs_contract} vs {rhs_contract}"
            ),
        });
    }

    let lhs_batch: &[usize] = if a_ndim > 1 {
        &a_shape[..a_ndim - 2]
    } else {
        &[]
    };
    let rhs_batch: &[usize] = if b_ndim > 1 {
        &b_shape[..b_ndim - 2]
    } else {
        &[]
    };

    let mut out_shape = broadcast_shapes(lhs_batch, rhs_batch)?;
    if a_ndim > 1 {
        out_shape.push(lhs_rows);
    }
    if b_ndim > 1 {
        out_shape.push(rhs_cols);
    }

    Ok(Some(meta_tensor(out_shape)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;
    use crate::creation;

    fn meta<T: Float>(shape: &[usize]) -> Tensor<T> {
        creation::zeros_meta(shape).unwrap()
    }

    fn cpu(shape: &[usize]) -> Tensor<f32> {
        creation::zeros(shape).unwrap()
    }

    // -----------------------------------------------------------------------
    // unary_same_shape
    // -----------------------------------------------------------------------

    #[test]
    fn test_unary_same_shape_meta_returns_meta() {
        let m: Tensor<f32> = meta(&[3, 4]);
        let out = unary_same_shape(&m).unwrap().unwrap();
        assert!(out.is_meta());
        assert_eq!(out.shape(), &[3, 4]);
    }

    #[test]
    fn test_unary_same_shape_cpu_returns_none() {
        let t: Tensor<f32> = cpu(&[3, 4]);
        let out = unary_same_shape(&t).unwrap();
        assert!(out.is_none());
    }

    // -----------------------------------------------------------------------
    // binary_broadcast
    // -----------------------------------------------------------------------

    #[test]
    fn test_binary_broadcast_both_meta_returns_broadcasted() {
        let a: Tensor<f32> = meta(&[3, 1]);
        let b: Tensor<f32> = meta(&[1, 4]);
        let out = binary_broadcast(&a, &b).unwrap().unwrap();
        assert!(out.is_meta());
        assert_eq!(out.shape(), &[3, 4]);
    }

    #[test]
    fn test_binary_broadcast_neither_meta_returns_none() {
        let a: Tensor<f32> = cpu(&[2, 3]);
        let b: Tensor<f32> = cpu(&[2, 3]);
        let out = binary_broadcast(&a, &b).unwrap();
        assert!(out.is_none());
    }

    #[test]
    fn test_binary_broadcast_mixed_errors() {
        let a: Tensor<f32> = meta(&[2, 3]);
        let b: Tensor<f32> = cpu(&[2, 3]);
        let result = binary_broadcast(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_binary_broadcast_meta_shape_mismatch_errors() {
        let a: Tensor<f32> = meta(&[3, 4]);
        let b: Tensor<f32> = meta(&[5, 6]);
        let result = binary_broadcast(&a, &b);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // ternary_broadcast
    // -----------------------------------------------------------------------

    #[test]
    fn test_ternary_broadcast_all_meta_returns_broadcasted() {
        let a: Tensor<f32> = meta(&[5, 1, 7]);
        let b: Tensor<f32> = meta(&[3, 1]);
        let c: Tensor<f32> = meta(&[1, 3, 1]);
        let out = ternary_broadcast(&a, &b, &c).unwrap().unwrap();
        assert!(out.is_meta());
        assert_eq!(out.shape(), &[5, 3, 7]);
    }

    #[test]
    fn test_ternary_broadcast_all_cpu_returns_none() {
        let a = cpu(&[2, 3]);
        let b = cpu(&[2, 3]);
        let c = cpu(&[2, 3]);
        let out = ternary_broadcast(&a, &b, &c).unwrap();
        assert!(out.is_none());
    }

    #[test]
    fn test_ternary_broadcast_mixed_meta_errors() {
        let a: Tensor<f32> = meta(&[2, 3]);
        let b = cpu(&[2, 3]);
        let c: Tensor<f32> = meta(&[2, 3]);
        let result = ternary_broadcast(&a, &b, &c);
        assert!(result.is_err());
    }

    #[test]
    fn test_ternary_broadcast_zero_size_shapes_match_torch() {
        let a: Tensor<f32> = meta(&[]);
        let b: Tensor<f32> = meta(&[2, 0, 3]);
        let c: Tensor<f32> = meta(&[1, 3]);
        let out = ternary_broadcast(&a, &b, &c).unwrap().unwrap();
        assert!(out.is_meta());
        assert_eq!(out.shape(), &[2, 0, 3]);
        assert_eq!(out.numel(), 0);
    }

    #[test]
    fn test_ternary_broadcast_shape_mismatch_errors() {
        let a: Tensor<f32> = meta(&[2, 3]);
        let b: Tensor<f32> = meta(&[4, 3]);
        let c: Tensor<f32> = meta(&[2, 3]);
        let result = ternary_broadcast(&a, &b, &c);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // reduce_dim
    // -----------------------------------------------------------------------

    #[test]
    fn test_reduce_dim_meta_removes_axis() {
        let m: Tensor<f32> = meta(&[2, 3, 4]);
        let out = reduce_dim(&m, 1, false).unwrap().unwrap();
        assert!(out.is_meta());
        assert_eq!(out.shape(), &[2, 4]);
    }

    #[test]
    fn test_reduce_dim_meta_keepdim_keeps_size_one() {
        let m: Tensor<f32> = meta(&[2, 3, 4]);
        let out = reduce_dim(&m, 1, true).unwrap().unwrap();
        assert!(out.is_meta());
        assert_eq!(out.shape(), &[2, 1, 4]);
    }

    #[test]
    fn test_reduce_dim_negative_axis() {
        let m: Tensor<f32> = meta(&[2, 3, 4]);
        let out = reduce_dim(&m, -1, false).unwrap().unwrap();
        assert_eq!(out.shape(), &[2, 3]);
    }

    #[test]
    fn test_reduce_dim_cpu_returns_none() {
        let t: Tensor<f32> = cpu(&[2, 3]);
        let out = reduce_dim(&t, 0, false).unwrap();
        assert!(out.is_none());
    }

    #[test]
    fn test_reduce_dim_out_of_bounds_errors() {
        let m: Tensor<f32> = meta(&[2, 3]);
        assert!(reduce_dim(&m, 5, false).is_err());
    }

    // -----------------------------------------------------------------------
    // reduce_all
    // -----------------------------------------------------------------------

    #[test]
    fn test_reduce_all_meta_returns_scalar() {
        let m: Tensor<f32> = meta(&[2, 3, 4]);
        let out = reduce_all(&m).unwrap().unwrap();
        assert!(out.is_meta());
        assert_eq!(out.shape(), [] as [usize; 0]);
    }

    #[test]
    fn test_reduce_all_cpu_returns_none() {
        let t: Tensor<f32> = cpu(&[2, 3]);
        let out = reduce_all(&t).unwrap();
        assert!(out.is_none());
    }

    // -----------------------------------------------------------------------
    // matmul shape rules
    // -----------------------------------------------------------------------

    #[test]
    fn test_matmul_1d_1d_dot() {
        let a: Tensor<f32> = meta(&[5]);
        let b: Tensor<f32> = meta(&[5]);
        let out = matmul(&a, &b).unwrap().unwrap();
        assert_eq!(out.shape(), [] as [usize; 0]);
    }

    #[test]
    fn test_matmul_2d_1d_mv() {
        let a: Tensor<f32> = meta(&[3, 5]);
        let b: Tensor<f32> = meta(&[5]);
        let out = matmul(&a, &b).unwrap().unwrap();
        assert_eq!(out.shape(), &[3]);
    }

    #[test]
    fn test_matmul_1d_2d_vm() {
        let a: Tensor<f32> = meta(&[5]);
        let b: Tensor<f32> = meta(&[5, 4]);
        let out = matmul(&a, &b).unwrap().unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_matmul_2d_2d_mm() {
        let a: Tensor<f32> = meta(&[3, 5]);
        let b: Tensor<f32> = meta(&[5, 4]);
        let out = matmul(&a, &b).unwrap().unwrap();
        assert_eq!(out.shape(), &[3, 4]);
    }

    #[test]
    fn test_matmul_batched_3d() {
        let a: Tensor<f32> = meta(&[2, 3, 5]);
        let b: Tensor<f32> = meta(&[2, 5, 4]);
        let out = matmul(&a, &b).unwrap().unwrap();
        assert_eq!(out.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_matmul_batched_with_broadcast() {
        let a: Tensor<f32> = meta(&[1, 3, 5]);
        let b: Tensor<f32> = meta(&[4, 5, 7]);
        let out = matmul(&a, &b).unwrap().unwrap();
        // Batch broadcast (1, 4) → 4, then 3×5 @ 5×7 → 3×7
        assert_eq!(out.shape(), &[4, 3, 7]);
    }

    #[test]
    fn test_matmul_vector_batched_shapes_match_torch() {
        // Shape oracles from torch.matmul on torch 2.11.0+cu130:
        // (3,)@(2,3,4) -> (2,4)
        // (2,3,4)@(4,) -> (2,3)
        // (3,)@(5,2,3,4) -> (5,2,4)
        // (5,2,3,4)@(4,) -> (5,2,3)
        // (2,1,3,4)@(3,4,5) -> (2,3,3,5)
        let cases: &[(&[usize], &[usize], &[usize])] = &[
            (&[3], &[2, 3, 4], &[2, 4]),
            (&[2, 3, 4], &[4], &[2, 3]),
            (&[3], &[5, 2, 3, 4], &[5, 2, 4]),
            (&[5, 2, 3, 4], &[4], &[5, 2, 3]),
            (&[2, 1, 3, 4], &[3, 4, 5], &[2, 3, 3, 5]),
        ];

        for &(a_shape, b_shape, expected) in cases {
            let a: Tensor<f32> = meta(a_shape);
            let b: Tensor<f32> = meta(b_shape);
            let out = matmul(&a, &b)
                .unwrap_or_else(|err| panic!("matmul({a_shape:?}, {b_shape:?}) errored: {err}"))
                .unwrap();
            assert_eq!(
                out.shape(),
                expected,
                "torch.matmul shape oracle for {a_shape:?} @ {b_shape:?}"
            );
        }
    }

    #[test]
    fn test_matmul_vector_batched_zero_dim_shapes_match_torch() {
        // Shape oracles from torch.matmul on torch 2.11.0+cu130:
        // (0,)@(2,0,4) -> (2,4)
        // (2,3,0)@(0,) -> (2,3)
        // (2,0,3,4)@(1,4,5) -> (2,0,3,5)
        let cases: &[(&[usize], &[usize], &[usize])] = &[
            (&[0], &[2, 0, 4], &[2, 4]),
            (&[2, 3, 0], &[0], &[2, 3]),
            (&[2, 0, 3, 4], &[1, 4, 5], &[2, 0, 3, 5]),
        ];

        for &(a_shape, b_shape, expected) in cases {
            let a: Tensor<f32> = meta(a_shape);
            let b: Tensor<f32> = meta(b_shape);
            let out = matmul(&a, &b)
                .unwrap_or_else(|err| panic!("matmul({a_shape:?}, {b_shape:?}) errored: {err}"))
                .unwrap();
            assert_eq!(
                out.shape(),
                expected,
                "torch.matmul zero-dim shape oracle for {a_shape:?} @ {b_shape:?}"
            );
            assert_eq!(out.numel(), expected.iter().product::<usize>());
        }
    }

    #[test]
    fn test_matmul_vector_batched_invalid_shapes_error_not_panic() {
        let cases: &[(&[usize], &[usize])] = &[
            (&[3], &[2, 4, 5]),
            (&[2, 3, 4], &[5]),
            (&[2, 0, 3, 4], &[7, 4, 5]),
        ];

        for &(a_shape, b_shape) in cases {
            let a: Tensor<f32> = meta(a_shape);
            let b: Tensor<f32> = meta(b_shape);
            assert!(
                matmul(&a, &b).is_err(),
                "invalid torch.matmul shape {a_shape:?} @ {b_shape:?} must error"
            );
        }
    }

    #[test]
    fn test_matmul_inner_dim_mismatch_errors() {
        let a: Tensor<f32> = meta(&[3, 5]);
        let b: Tensor<f32> = meta(&[6, 4]);
        assert!(matmul(&a, &b).is_err());
    }

    #[test]
    fn test_matmul_cpu_returns_none() {
        let a: Tensor<f32> = cpu(&[3, 5]);
        let b: Tensor<f32> = cpu(&[5, 4]);
        let out = matmul(&a, &b).unwrap();
        assert!(out.is_none());
    }

    #[test]
    fn test_matmul_mixed_meta_errors() {
        let a: Tensor<f32> = meta(&[3, 5]);
        let b: Tensor<f32> = cpu(&[5, 4]);
        assert!(matmul(&a, &b).is_err());
    }

    // -----------------------------------------------------------------------
    // End-to-end pipeline tests through the actual instrumented ops.
    //
    // These verify that meta tensors propagate correctly through the public
    // arithmetic, reduction, linalg, and activation entry points without
    // ever touching data.
    // -----------------------------------------------------------------------

    #[test]
    fn test_e2e_meta_arithmetic_chain() {
        use crate::grad_fns::arithmetic::{add, mul, neg, sqrt};

        let a: Tensor<f32> = meta(&[3, 4]);
        let b: Tensor<f32> = meta(&[3, 4]);
        let c = add(&a, &b).unwrap();
        let d = mul(&c, &a).unwrap();
        let e = neg(&d).unwrap();
        let f = sqrt(&e).unwrap();
        assert!(f.is_meta());
        assert_eq!(f.shape(), &[3, 4]);
    }

    #[test]
    fn test_e2e_meta_arithmetic_with_broadcast() {
        use crate::grad_fns::arithmetic::add;

        let a: Tensor<f32> = meta(&[5, 1, 7]);
        let b: Tensor<f32> = meta(&[3, 1]);
        let out = add(&a, &b).unwrap();
        assert!(out.is_meta());
        // Broadcast: [5, 1, 7] x [3, 1] -> [5, 3, 7]
        assert_eq!(out.shape(), &[5, 3, 7]);
    }

    #[test]
    fn test_e2e_meta_reductions() {
        use crate::grad_fns::reduction::{mean_dim, sum, sum_dim};

        let x: Tensor<f32> = meta(&[2, 3, 4]);
        let s = sum(&x).unwrap();
        assert!(s.is_meta());
        assert_eq!(s.shape(), [] as [usize; 0]);

        let s2 = sum_dim(&x, 1, false).unwrap();
        assert!(s2.is_meta());
        assert_eq!(s2.shape(), &[2, 4]);

        let m = mean_dim(&x, 2, true).unwrap();
        assert!(m.is_meta());
        assert_eq!(m.shape(), &[2, 3, 1]);
    }

    #[test]
    fn test_e2e_meta_matmul() {
        use crate::ops::linalg::matmul as op_matmul;

        let a: Tensor<f32> = meta(&[8, 16]);
        let b: Tensor<f32> = meta(&[16, 32]);
        let out = op_matmul(&a, &b).unwrap();
        assert!(out.is_meta());
        assert_eq!(out.shape(), &[8, 32]);
    }

    #[test]
    fn test_e2e_meta_matmul_vector_batched() {
        use crate::ops::linalg::matmul as op_matmul;

        let vector_lhs: Tensor<f32> = meta(&[3]);
        let batched_rhs: Tensor<f32> = meta(&[2, 3, 4]);
        let out = op_matmul(&vector_lhs, &batched_rhs).unwrap();
        assert!(out.is_meta());
        assert_eq!(out.shape(), &[2, 4]);

        let batched_lhs: Tensor<f32> = meta(&[2, 3, 4]);
        let vector_rhs: Tensor<f32> = meta(&[4]);
        let out = op_matmul(&batched_lhs, &vector_rhs).unwrap();
        assert!(out.is_meta());
        assert_eq!(out.shape(), &[2, 3]);
    }

    #[test]
    fn test_e2e_meta_activations() {
        use crate::grad_fns::activation::{gelu, relu, sigmoid, silu, softmax, tanh};

        let x: Tensor<f32> = meta(&[2, 5]);
        for op_out in [
            relu(&x).unwrap(),
            sigmoid(&x).unwrap(),
            tanh(&x).unwrap(),
            gelu(&x).unwrap(),
            silu(&x).unwrap(),
            softmax(&x).unwrap(),
        ] {
            assert!(op_out.is_meta());
            assert_eq!(op_out.shape(), &[2, 5]);
        }
    }

    #[test]
    fn test_e2e_meta_mlp_dry_run() {
        // Simulate building a tiny MLP and running its forward on meta
        // inputs to determine output shape. No real allocation happens
        // for the activations, only for the parameter tensors (which we
        // also keep on meta). This is the canonical use case for the
        // meta device.
        use crate::grad_fns::activation::relu;
        use crate::grad_fns::arithmetic::add;
        use crate::ops::linalg::matmul as op_matmul;

        // Layer 1: 64 -> 32
        let x: Tensor<f32> = meta(&[16, 64]); // batch=16
        let w1: Tensor<f32> = meta(&[64, 32]);
        let b1: Tensor<f32> = meta(&[32]);
        let h1 = add(&op_matmul(&x, &w1).unwrap(), &b1).unwrap();
        let h1_relu = relu(&h1).unwrap();
        assert!(h1_relu.is_meta());
        assert_eq!(h1_relu.shape(), &[16, 32]);

        // Layer 2: 32 -> 10
        let w2: Tensor<f32> = meta(&[32, 10]);
        let b2: Tensor<f32> = meta(&[10]);
        let logits = add(&op_matmul(&h1_relu, &w2).unwrap(), &b2).unwrap();
        assert!(logits.is_meta());
        assert_eq!(logits.shape(), &[16, 10]);
    }
}
