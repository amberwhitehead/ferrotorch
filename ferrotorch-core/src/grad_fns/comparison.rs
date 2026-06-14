//! Backward function for the differentiable conditional `where_` operation.
//!
//! `where_(condition, x, y)` selects from `x` where `condition` is true, and
//! from `y` where `condition` is false. The public wrappers route CUDA and
//! broadcasted cases through the resident `ops::indexing::where_cond_bt` /
//! `grad_fns::indexing::where_cond_bcast` path.
//!
//! ## REQ status (per `.design/ferrotorch-core/grad_fns/comparison.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`where_` forward + backward) | SHIPPED | `where_` at `comparison.rs` handles the host-mask public method. CPU same-shape keeps the legacy `WhereBackward`; CUDA and broadcasted x/y route through `where_cond_bcast`, with a full-output flat condition mask. Non-test production consumer: `Tensor::where_t` at `methods.rs` delegates to `crate::grad_fns::comparison::where_`. |
//! | REQ-2 (`where_bt` `BoolTensor` variant) | SHIPPED | `where_bt` at `comparison.rs` delegates to `grad_fns::indexing::where_cond_bcast`, so condition/x/y broadcast by PyTorch rules and CUDA operands stay resident. Non-test production consumer: `Tensor::where_bt_t` at `methods.rs`. |
//! | REQ-3 (device handling + NaN/Inf passthrough) | SHIPPED | First-class BoolTensor CUDA where uses `ops::indexing::where_cond_bt` via `where_cond_bcast`; host-mask CUDA where uploads only the boolean condition and keeps x/y/result resident. |
//! | REQ-4 (17 comparison parity ops the route declares) | NOT-STARTED | the 17 ops (`eq`, `ne`, `lt`, `le`, `gt`, `ge`, `logical_and`, `logical_or`, `logical_xor`, `logical_not`, `max`, `min`, `maximum`, `minimum`, `isnan`, `isinf`, `isfinite`) are not implemented in this file; they live in `bool_tensor.rs` or elsewhere. Route retarget tracked under #1293. |

use std::sync::Arc;

use crate::autograd::no_grad::is_grad_enabled;
use crate::dtype::Float;
use crate::error::FerrotorchResult;
use crate::storage::TensorStorage;
use crate::tensor::{GradFn, Tensor};

use crate::bool_tensor::BoolTensor;
use crate::device::Device;
use crate::error::FerrotorchError;

/// Backward node for `where_(condition, x, y)`.
///
/// Stores the boolean condition mask and references to both input tensors
/// so the autograd engine can traverse the graph.
#[derive(Debug)]
pub struct WhereBackward<T: Float> {
    condition: Vec<bool>,
    x: Tensor<T>,
    y: Tensor<T>,
}

impl<T: Float> GradFn<T> for WhereBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let device = grad_output.device();
        let go = grad_output.data_vec()?;
        let zero = <T as num_traits::Zero>::zero();

        // grad_x = grad_output where condition is true, 0 otherwise
        let grad_x: Vec<T> = go
            .iter()
            .zip(self.condition.iter())
            .map(|(&g, &c)| if c { g } else { zero })
            .collect();

        // grad_y = grad_output where condition is false, 0 otherwise
        let grad_y: Vec<T> = go
            .iter()
            .zip(self.condition.iter())
            .map(|(&g, &c)| if c { zero } else { g })
            .collect();

        let grad_x_tensor =
            Tensor::from_storage(TensorStorage::cpu(grad_x), self.x.shape().to_vec(), false)?;
        let grad_y_tensor =
            Tensor::from_storage(TensorStorage::cpu(grad_y), self.y.shape().to_vec(), false)?;

        if device.is_cuda() {
            Ok(vec![
                Some(grad_x_tensor.to(device)?),
                Some(grad_y_tensor.to(device)?),
            ])
        } else {
            Ok(vec![Some(grad_x_tensor), Some(grad_y_tensor)])
        }
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.x, &self.y]
    }

    fn name(&self) -> &'static str {
        "WhereBackward"
    }
}

fn checked_numel(shape: &[usize], op: &'static str) -> FerrotorchResult<usize> {
    if shape.is_empty() {
        return Ok(1);
    }
    shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| FerrotorchError::InvalidArgument {
            message: format!("{op}: output shape {shape:?} element count overflows usize"),
        })
}

/// Differentiable conditional selection.
///
/// For each element `i`, the output is `x[i]` if `condition[i]` is true,
/// otherwise `y[i]`.
///
/// # Validation (CORE-043 / #1737)
///
/// This host-mask entry treats `condition` as a flat mask over the broadcasted
/// output shape of `x` and `y`. A raw `&[bool]` has no shape of its own, so it
/// cannot express a lower-rank broadcasted condition tensor; callers needing
/// condition broadcasting should use [`where_bt`] with a [`BoolTensor`].
/// Structured errors at the boundary (R-LOUD-1), each a case live torch
/// 2.11.0+cu130 also rejects with a `RuntimeError`:
///
/// - `x.device() != y.device()` → [`FerrotorchError::DeviceMismatch`]
///   (torch: "Expected all tensors to be on the same device, but found at
///   least two devices, cuda:0 and cpu!").
/// - `x.shape()` and `y.shape()` not broadcast-compatible →
///   [`FerrotorchError::ShapeMismatch`] (torch: "The size of tensor a ...").
/// - `condition.len() != prod(broadcast_shape(x, y))` →
///   [`FerrotorchError::ShapeMismatch`] (formerly a `debug_assert_eq!`, i.e.
///   silent zip truncation in release).
///
/// # Device behavior (R-LOUD-2)
///
/// The condition is an inherently host-side `&[bool]`. For CUDA `x`/`y`
/// (same device, enforced above) this uploads only the boolean condition to
/// CUDA and delegates to the resident `where_cond_bcast` path. The value
/// tensors and result stay on-device.
///
/// When gradient tracking is enabled and either input requires grad, the
/// returned tensor carries a backward node that routes gradients to the
/// appropriate input during the backward pass. The same-shape CPU fast path
/// uses [`WhereBackward`]; CUDA and broadcasted paths use
/// [`crate::grad_fns::indexing::WhereCondBackward`] through
/// `where_cond_bcast`.
pub fn where_<T: Float>(
    condition: &[bool],
    x: &Tensor<T>,
    y: &Tensor<T>,
) -> FerrotorchResult<Tensor<T>> {
    let device = x.device();
    if y.device() != device {
        return Err(FerrotorchError::DeviceMismatch {
            expected: device,
            got: y.device(),
        });
    }
    let common = crate::shape::broadcast_shapes(x.shape(), y.shape()).map_err(|_| {
        FerrotorchError::ShapeMismatch {
            message: format!(
                "where_: x shape {:?} and y shape {:?} are not broadcast-compatible",
                x.shape(),
                y.shape()
            ),
        }
    })?;
    let expected = checked_numel(&common, "where_")?;
    if condition.len() != expected {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "where_: condition length {} != broadcast output numel {} \
                 (broadcast shape {:?})",
                condition.len(),
                expected,
                common
            ),
        });
    }

    if device.is_cuda() || x.shape() != common || y.shape() != common {
        let cond = BoolTensor::from_slice(condition, &common)?;
        let cond = if device == Device::Cpu {
            cond
        } else {
            cond.to(device)?
        };
        return crate::grad_fns::indexing::where_cond_bcast(&cond, x, y);
    }

    let x_data = x.data_vec()?;
    let y_data = y.data_vec()?;

    let result: Vec<T> = condition
        .iter()
        .zip(x_data.iter().zip(y_data.iter()))
        .map(|(&c, (&xv, &yv))| if c { xv } else { yv })
        .collect();

    let needs_grad = is_grad_enabled() && (x.requires_grad() || y.requires_grad());

    let storage = TensorStorage::on_device(result, device)?;
    if needs_grad {
        let grad_fn = Arc::new(WhereBackward {
            condition: condition.to_vec(),
            x: x.clone(),
            y: y.clone(),
        });
        Tensor::from_operation(storage, x.shape().to_vec(), grad_fn)
    } else {
        Tensor::from_storage(storage, x.shape().to_vec(), false)
    }
}

// ---------------------------------------------------------------------------
// First-class BoolTensor wrapper (#615)
// ---------------------------------------------------------------------------

/// Pointwise ternary `where(cond, x, y)` taking a [`BoolTensor`] for
/// the condition. Mirrors `torch.where(cond, x, y)`: condition, `x`, and `y`
/// broadcast to a common shape by PyTorch/NumPy rules.
///
/// # Validation (CORE-043 / #1737)
///
/// Device validation and resident CUDA execution are handled by
/// [`crate::grad_fns::indexing::where_cond_bcast`].
pub fn where_bt<T: Float>(
    cond: &BoolTensor,
    x: &Tensor<T>,
    y: &Tensor<T>,
) -> FerrotorchResult<Tensor<T>> {
    crate::grad_fns::indexing::where_cond_bcast(cond, x, y)
}

#[cfg(test)]
mod first_class_tests {
    use super::*;

    #[test]
    fn where_bt_picks_correctly() {
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0_f32, 2.0, 3.0, 4.0]),
            vec![4],
            false,
        )
        .unwrap();
        let y = Tensor::from_storage(
            TensorStorage::cpu(vec![10.0_f32, 20.0, 30.0, 40.0]),
            vec![4],
            false,
        )
        .unwrap();
        let cond = BoolTensor::from_vec(vec![true, false, true, false], vec![4]).unwrap();
        let out = where_bt(&cond, &x, &y).unwrap();
        assert_eq!(out.data().unwrap(), &[1.0, 20.0, 3.0, 40.0]);
    }

    #[test]
    fn where_bt_rejects_shape_mismatch() {
        let x = Tensor::from_storage(TensorStorage::cpu(vec![1.0_f32; 4]), vec![4], false).unwrap();
        let y = Tensor::from_storage(TensorStorage::cpu(vec![0.0_f32; 4]), vec![4], false).unwrap();
        let cond = BoolTensor::from_vec(vec![true; 3], vec![3]).unwrap();
        let err = where_bt(&cond, &x, &y).unwrap_err();
        assert!(matches!(err, FerrotorchError::ShapeMismatch { .. }));
    }

    #[test]
    fn where_bt_broadcasts_three_inputs_and_reduces_grads() {
        let cond = BoolTensor::from_vec(vec![true, false], vec![2, 1]).unwrap();
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0_f32, 2.0, 3.0]),
            vec![1, 3],
            true,
        )
        .unwrap();
        let y = Tensor::from_storage(TensorStorage::cpu(vec![10.0_f32]), vec![], true).unwrap();

        let out = where_bt(&cond, &x, &y).unwrap();
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out.data().unwrap(), &[1.0, 2.0, 3.0, 10.0, 10.0, 10.0]);

        crate::grad_fns::reduction::sum(&out)
            .unwrap()
            .backward()
            .unwrap();
        let gx = x.grad().unwrap().unwrap();
        let gy = y.grad().unwrap().unwrap();
        assert_eq!(gx.shape(), &[1, 3]);
        assert_eq!(gx.data().unwrap(), &[1.0, 1.0, 1.0]);
        assert!(gy.shape().is_empty());
        assert_eq!(gy.data().unwrap(), &[3.0]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::graph::backward;
    use crate::storage::TensorStorage;

    /// Helper to make a leaf tensor from a slice.
    fn leaf(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
        Tensor::from_storage(
            TensorStorage::cpu(data.to_vec()),
            shape.to_vec(),
            requires_grad,
        )
        .unwrap()
    }

    #[test]
    fn test_where_forward() {
        let cond = vec![true, false, true, false];
        let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[4], false);
        let y = leaf(&[10.0, 20.0, 30.0, 40.0], &[4], false);

        let out = where_(&cond, &x, &y).unwrap();
        assert_eq!(out.data().unwrap(), &[1.0, 20.0, 3.0, 40.0]);
    }

    #[test]
    fn test_where_host_mask_broadcasts_operands() {
        let cond = vec![true, false, true];
        let x = leaf(&[1.0, 2.0, 3.0], &[1, 3], false);
        let y = leaf(&[10.0], &[], false);

        let out = where_(&cond, &x, &y).unwrap();
        assert_eq!(out.shape(), &[1, 3]);
        assert_eq!(out.data().unwrap(), &[1.0, 10.0, 3.0]);
    }

    #[test]
    fn test_where_backward() {
        // condition = [true, false, true, false]
        // out = where_(cond, x, y) = [x0, y1, x2, y3]
        //
        // To get a scalar for backward, compute sum(out).
        // grad_output for where_ is all 1s (from sum backward).
        //
        // Expected gradients:
        //   grad_x = [1.0, 0.0, 1.0, 0.0]  (gradient flows where condition is true)
        //   grad_y = [0.0, 1.0, 0.0, 1.0]  (gradient flows where condition is false)
        let cond = vec![true, false, true, false];
        let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[4], true);
        let y = leaf(&[10.0, 20.0, 30.0, 40.0], &[4], true);

        let out = where_(&cond, &x, &y).unwrap();

        // sum(out) to get a scalar for backward
        let out_data = out.data().unwrap();
        let total: f32 = out_data.iter().sum();

        // Build sum node: backward of sum passes ones as grad to its input.
        #[derive(Debug)]
        struct SumBackward<T: Float> {
            input: Tensor<T>,
            numel: usize,
        }

        impl<T: Float> GradFn<T> for SumBackward<T> {
            fn backward(
                &self,
                _grad_output: &Tensor<T>,
            ) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
                let ones = vec![<T as num_traits::One>::one(); self.numel];
                let t = Tensor::from_storage(
                    TensorStorage::cpu(ones),
                    self.input.shape().to_vec(),
                    false,
                )?;
                Ok(vec![Some(t)])
            }
            fn inputs(&self) -> Vec<&Tensor<T>> {
                vec![&self.input]
            }
            fn name(&self) -> &'static str {
                "SumBackward"
            }
        }

        let scalar = Tensor::from_operation(
            TensorStorage::cpu(vec![total]),
            vec![],
            Arc::new(SumBackward {
                input: out.clone(),
                numel: 4,
            }),
        )
        .unwrap();

        backward(&scalar).unwrap();

        let x_grad = x.grad().unwrap().unwrap();
        let y_grad = y.grad().unwrap().unwrap();

        assert_eq!(x_grad.data().unwrap(), &[1.0, 0.0, 1.0, 0.0]);
        assert_eq!(y_grad.data().unwrap(), &[0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_where_no_grad() {
        crate::autograd::no_grad::no_grad(|| {
            let cond = vec![true, false];
            let x = leaf(&[1.0, 2.0], &[2], true);
            let y = leaf(&[10.0, 20.0], &[2], true);

            let out = where_(&cond, &x, &y).unwrap();
            assert!(!out.requires_grad());
            assert!(out.grad_fn().is_none());
            assert_eq!(out.data().unwrap(), &[1.0, 20.0]);
        });
    }
}
