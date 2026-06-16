//! Fixed-point implicit differentiation for ferrotorch autograd.
//!
//! Given a fixed-point equation `x* = f(x*, p)` where `x*` is the fixed point
//! and `p` are parameters, the implicit function theorem gives:
//!
//! ```text
//! dx*/dp = (I - df/dx|_{x*})^{-1} @ (df/dp|_{x*})
//! ```
//!
//! This avoids unrolling through the entire iterative process (which can be
//! thousands of steps), making it memory-efficient for:
//!
//! - Deep Equilibrium Models (DEQ)
//! - Long-context RNNs (fixed point of the recurrence)
//! - Neural ODEs (fixed point of the flow)
//! - Neural Cellular Automata (fixed point of the update rule)
//!
//! The backward pass uses the Neumann series approximation:
//!
//! ```text
//! v = (I - J_x^T)^{-1} @ grad_output = sum_{k=0}^{K} (J_x^T)^k @ grad_output
//! ```
//!
//! which converges when the spectral radius of `J_x` is less than 1 (guaranteed
//! when `f` is contractive).
//! ## REQ status (per `.design/ferrotorch-core/autograd/fixed_point.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | `pub fn `fixed_point`<T, F>` at `fixed_point.rs:79-135`; consumer: re-exported at `ferrotorch-core/src/autograd/mod.rs:27 pub use fixed_point::fixed_point` and `lib.rs:127`. Existing pub API — boundary-API grandfathering. |
//! | REQ-2 | SHIPPED | Forward-iteration loop at `fixed_point.rs:91-109`; consumer: inside REQ-1. |
//! | REQ-3 | SHIPPED | `if params.iter().any(\|p\| p.requires_grad())` at `fixed_point.rs:113`; consumer: inside REQ-1. |
//! | REQ-4 | SHIPPED | `struct `FixedPointBackward`<T: Float>` at `fixed_point.rs:147-158` + `impl GradFn` at `:172-322`; consumer: instantiated inside REQ-1 at `:124-130`. |
//! | REQ-5 | SHIPPED | Neumann series solve at `fixed_point.rs:177-245`; consumer: inside REQ-4's `backward` impl. |
//! | REQ-6 | SHIPPED | Per-parameter gradient distribution at `fixed_point.rs:247-313`; consumer: inside REQ-4's `backward` impl. |
//! | REQ-7 | SHIPPED | `fn `elementwise_mul_sum`<T: Float>` at `fixed_point.rs:328-331`; consumer: called twice inside REQ-5 at `:222` and REQ-6 at `:282`. |
//!

use std::fmt;
use std::sync::Arc;

use crate::autograd::higher_order::grad;
use crate::autograd::no_grad::no_grad;
use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::methods::contiguous_t;
use crate::tensor::{GradFn, Tensor};

/// Type alias for the fixed-point function f(x, params) -> x.
type FixedPointFn<T> =
    Arc<dyn Fn(&Tensor<T>, &[&Tensor<T>]) -> FerrotorchResult<Tensor<T>> + Send + Sync>;

/// Find a fixed point of `f` starting from `x0`, then compute its derivative
/// w.r.t. `params` using the implicit function theorem.
///
/// This is used for:
/// - Long-context RNNs (fixed point of the recurrence)
/// - Neural ODEs (fixed point of the flow)
/// - Neural Cellular Automata (fixed point of the update rule)
/// - Equilibrium models (DEQ)
///
/// # Arguments
/// - `f`: The function x_{n+1} = f(x_n, params). Must be contractive.
/// - `x0`: Initial guess.
/// - `params`: Parameters to differentiate w.r.t.
/// - `max_iter`: Maximum iterations to find the fixed point.
/// - `tol`: Convergence tolerance (stop when ||x_{n+1} - x_n|| < tol).
///
/// # Returns
/// The fixed point x* as a Tensor with grad_fn attached so that
/// backward() computes dx*/dp via the implicit function theorem.
///
/// # Examples
///
/// ```ignore
/// // f(x, a) = a * x, a = 0.5, x0 = 10
/// // Fixed point: x* = 0
/// let a = Tensor::from_storage(TensorStorage::cpu(vec![0.5f32]), vec![], true)?;
/// let x0 = Tensor::from_storage(TensorStorage::cpu(vec![10.0f32]), vec![], false)?;
/// let x_star = fixed_point(
///     |x, p| {
///         // f(x, a) = a * x
///         crate::grad_fns::arithmetic::mul(x, p[0])
///     },
///     &x0,
///     &[&a],
///     1000,
///     1e-8,
/// )?;
/// ```
pub fn fixed_point<T, F>(
    f: F,
    x0: &Tensor<T>,
    params: &[&Tensor<T>],
    max_iter: usize,
    tol: f64,
) -> FerrotorchResult<Tensor<T>>
where
    T: Float,
    F: Fn(&Tensor<T>, &[&Tensor<T>]) -> FerrotorchResult<Tensor<T>> + Send + Sync + 'static,
{
    validate_fixed_point_config(max_iter, tol, "fixed_point")?;

    // 1. Find the fixed point by iteration (forward pass, no grad needed).
    let x_star = no_grad(|| -> FerrotorchResult<Tensor<T>> {
        let mut x = x0.clone();
        let mut last_residual = f64::INFINITY;
        for iter in 0..max_iter {
            let x_next = f(&x, params)?;
            validate_fixed_point_iterate("fixed_point forward", &x, &x_next)?;

            let residual = fixed_point_residual_l1(&x, &x_next, "fixed_point forward")?;
            validate_residual_is_finite(residual, iter, "fixed_point forward")?;
            if residual <= tol {
                return Ok::<Tensor<T>, FerrotorchError>(x_next);
            }
            last_residual = residual;
            x = x_next;
        }
        Err(non_convergence_error(
            "fixed_point forward",
            max_iter,
            last_residual,
            tol,
        ))
    })?;

    // 2. If any parameter requires grad, attach a FixedPointBackward node
    //    that uses implicit differentiation via the Neumann series.
    if params.iter().any(|p| p.requires_grad()) {
        let x_star = contiguous_t(&x_star)?;
        let backward_x_star = x_star.clone();
        let (storage, x_star_shape) = x_star.into_storage_and_shape()?;

        // Clone params for storage in the backward node.
        let params_owned: Vec<Tensor<T>> = params.iter().map(|p| (*p).clone()).collect();

        Tensor::from_operation(
            storage,
            x_star_shape,
            Arc::new(FixedPointBackward {
                f_closure: Arc::new(f),
                x_star: backward_x_star,
                params: params_owned,
                backward_max_iter: max_iter.min(50), // Cap backward iterations.
                backward_tol: tol,
            }),
        )
    } else {
        Ok(x_star)
    }
}

fn validate_fixed_point_config(max_iter: usize, tol: f64, op: &str) -> FerrotorchResult<()> {
    if max_iter == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("{op}: max_iter must be greater than zero"),
        });
    }
    if !tol.is_finite() || tol < 0.0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("{op}: tol must be finite and non-negative, got {tol}"),
        });
    }
    Ok(())
}

fn validate_fixed_point_iterate<T: Float>(
    op: &str,
    expected: &Tensor<T>,
    actual: &Tensor<T>,
) -> FerrotorchResult<()> {
    if expected.device() != actual.device() {
        return Err(FerrotorchError::DeviceMismatch {
            expected: expected.device(),
            got: actual.device(),
        });
    }
    if expected.shape() != actual.shape() {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "{op}: iterate shape mismatch: expected {:?}, got {:?}",
                expected.shape(),
                actual.shape()
            ),
        });
    }
    Ok(())
}

fn validate_residual_is_finite(residual: f64, iter: usize, op: &str) -> FerrotorchResult<()> {
    if residual.is_finite() {
        Ok(())
    } else {
        Err(FerrotorchError::InvalidArgument {
            message: format!("{op}: iteration {iter} produced non-finite residual {residual}"),
        })
    }
}

fn non_convergence_error(
    op: &str,
    max_iter: usize,
    last_residual: f64,
    tol: f64,
) -> FerrotorchError {
    FerrotorchError::InvalidArgument {
        message: format!(
            "{op}: failed to converge after {max_iter} iterations \
             (last L1 residual {last_residual}, tolerance {tol})"
        ),
    }
}

fn tensor_scalar_to_f64<T: Float>(value: &Tensor<T>, op: &str) -> FerrotorchResult<f64> {
    let host_value = if value.device().is_cpu() {
        value.clone()
    } else {
        value.cpu()?
    };
    host_value
        .item()?
        .to_f64()
        .ok_or_else(|| FerrotorchError::InvalidArgument {
            message: format!("{op}: scalar residual is not representable as f64"),
        })
}

fn fixed_point_residual_l1<T: Float>(
    prev: &Tensor<T>,
    next: &Tensor<T>,
    op: &str,
) -> FerrotorchResult<f64> {
    validate_fixed_point_iterate(op, prev, next)?;
    let residual = no_grad(|| {
        let diff = crate::grad_fns::arithmetic::sub(next, prev)?;
        let abs = crate::grad_fns::arithmetic::abs(&diff)?;
        crate::grad_fns::reduction::sum(&abs)
    })?;
    tensor_scalar_to_f64(&residual, op)
}

/// Backward node for fixed-point implicit differentiation.
///
/// Uses the Neumann series to solve the implicit derivative system:
///
/// ```text
/// (I - J_x^T) @ v = grad_output
/// v = sum_{k=0}^{K} (J_x^T)^k @ grad_output
/// ```
///
/// Then distributes `v` through `df/dp` to produce gradients for each parameter.
struct FixedPointBackward<T: Float> {
    /// The function f(x, params) whose fixed point was found.
    f_closure: FixedPointFn<T>,
    /// The fixed point x*.
    x_star: Tensor<T>,
    /// The parameters to differentiate w.r.t.
    params: Vec<Tensor<T>>,
    /// Maximum iterations for the Neumann series in the backward pass.
    backward_max_iter: usize,
    /// Convergence tolerance for the Neumann series.
    backward_tol: f64,
}

impl<T: Float> fmt::Debug for FixedPointBackward<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FixedPointBackward")
            .field("f_closure", &"<closure>")
            .field("x_star_shape", &self.x_star.shape())
            .field("num_params", &self.params.len())
            .field("backward_max_iter", &self.backward_max_iter)
            .field("backward_tol", &self.backward_tol)
            .finish()
    }
}

impl<T: Float> GradFn<T> for FixedPointBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let num_params = self.params.len();
        validate_fixed_point_config(
            self.backward_max_iter,
            self.backward_tol,
            "FixedPointBackward",
        )?;
        validate_fixed_point_iterate("FixedPointBackward grad_output", &self.x_star, grad_output)?;

        // Step 1: Solve (I - J_x^T) v = grad_output via the Neumann series.
        //
        // v_0 = grad_output
        // v_{k+1} = grad_output + J_x^T @ v_k
        //
        // This converges because f is contractive => spectral_radius(J_x) < 1.
        //
        // We compute J_x^T @ v via a VJP: if y = f(x, p), then
        // VJP(y, x, v) = J_x^T @ v = grad(y, x, grad_output=v).
        let mut v = grad_output.detach();
        let mut converged = false;
        let mut last_residual = f64::INFINITY;

        for iter in 0..self.backward_max_iter {
            // Create a fresh x* that requires grad so we can compute J_x^T @ v.
            let x_fresh = self.x_star.detach().requires_grad_(true);

            // Detached params (we only want the Jacobian w.r.t. x here).
            let params_detached: Vec<Tensor<T>> = self.params.iter().map(Tensor::detach).collect();
            let params_ref: Vec<&Tensor<T>> = params_detached.iter().collect();

            // Evaluate f(x*, params) with grad tracking on x.
            let y = (self.f_closure)(&x_fresh, &params_ref)?;
            validate_fixed_point_iterate("FixedPointBackward Jx evaluation", &x_fresh, &y)?;

            // Compute VJP: J_x^T @ v via grad(y, x, grad_output=v).
            // We need to make y scalar to use grad(), so we use a dot product:
            // L = sum(y * v), then grad(L, x) = J_x^T @ v.
            let yv = elementwise_mul_sum(&y, &v)?;

            let grads = grad(&yv, &[&x_fresh], false, false)?;

            let jt_v = match grads[0].as_ref() {
                Some(g) => {
                    validate_fixed_point_iterate("FixedPointBackward Jx VJP", &self.x_star, g)?;
                    g.clone()
                }
                None => crate::creation::zeros_like(&self.x_star)?,
            };

            // v_new = grad_output + J_x^T @ v
            let v_new = no_grad(|| crate::grad_fns::arithmetic::add(grad_output, &jt_v))?;
            validate_fixed_point_iterate("FixedPointBackward Neumann update", &v, &v_new)?;
            let residual = fixed_point_residual_l1(&v, &v_new, "FixedPointBackward")?;
            validate_residual_is_finite(residual, iter, "FixedPointBackward")?;
            last_residual = residual;
            v = v_new.detach();

            if residual <= self.backward_tol {
                converged = true;
                break;
            }
        }
        if !converged {
            return Err(non_convergence_error(
                "FixedPointBackward",
                self.backward_max_iter,
                last_residual,
                self.backward_tol,
            ));
        }

        // Step 2: Compute gradients for each parameter.
        //
        // For each param p_i, the gradient is:
        //   grad_p_i = J_{p_i}^T @ v = grad(f(x*, p), p_i, grad_output=v)
        //
        // We evaluate f(x*, params) with grad tracking on params, then
        // compute grad(L, params) where L = sum(y * v).

        // Create x* without grad (we don't need x gradients here).
        let x_detached = self.x_star.detach();

        // Create params with grad enabled.
        let params_with_grad: Vec<Tensor<T>> = self
            .params
            .iter()
            .map(|p| p.detach().requires_grad_(p.requires_grad()))
            .collect();
        let params_ref: Vec<&Tensor<T>> = params_with_grad.iter().collect();

        // Evaluate f(x*, params).
        let y = (self.f_closure)(&x_detached, &params_ref)?;
        validate_fixed_point_iterate("FixedPointBackward parameter evaluation", &x_detached, &y)?;

        // L = sum(y * v)
        let loss = elementwise_mul_sum(&y, &v)?;

        // Compute grad(L, params).
        let grad_inputs: Vec<&Tensor<T>> = params_with_grad
            .iter()
            .filter(|p| p.requires_grad())
            .collect();

        let mut result: Vec<Option<Tensor<T>>> = Vec::with_capacity(num_params);

        if grad_inputs.is_empty() {
            for _ in 0..num_params {
                result.push(None);
            }
        } else {
            let param_grads = grad(&loss, &grad_inputs[..], false, false)?;

            // Map back: grad_inputs is a filtered subset; we need to map each
            // param in self.params to its gradient (or None if !requires_grad).
            let mut grad_idx = 0;
            for p in &params_with_grad {
                if p.requires_grad() {
                    result.push(param_grads[grad_idx].clone());
                    grad_idx += 1;
                } else {
                    result.push(None);
                }
            }
        }

        Ok(result)
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        self.params.iter().collect()
    }

    fn name(&self) -> &'static str {
        "FixedPointBackward"
    }
}

/// Compute `sum(a * b)` as a scalar tensor, preserving the autograd graph.
///
/// This is equivalent to a dot product when both tensors are 1-D, and a
/// Frobenius inner product for higher-dimensional tensors.
fn elementwise_mul_sum<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let prod = crate::grad_fns::arithmetic::mul(a, b)?;
    crate::grad_fns::reduction::sum(&prod)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::graph::backward;
    use crate::device::Device;
    use crate::storage::TensorStorage;

    /// Create a leaf scalar tensor.
    fn leaf_scalar(val: f32, requires_grad: bool) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(vec![val]), vec![], requires_grad).unwrap()
    }

    /// Assert a scalar tensor is approximately equal to `expected`.
    fn assert_approx(actual: f32, expected: f32, tol: f32, msg: &str) {
        assert!(
            (actual - expected).abs() < tol,
            "{msg}: expected {expected}, got {actual}"
        );
    }

    // -----------------------------------------------------------------------
    // Basic fixed-point convergence tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_fixed_point_affine() {
        // f(x) = 0.5 * x + 1
        // Fixed point: x* = 0.5 * x* + 1 => 0.5 * x* = 1 => x* = 2
        let x0 = leaf_scalar(0.0, false);
        let dummy_param = leaf_scalar(1.0, false);

        let x_star = fixed_point(
            |x, _params| {
                // f(x) = 0.5 * x + 1
                let half = Tensor::from_storage(TensorStorage::cpu(vec![0.5f32]), vec![], false)?;
                let one = Tensor::from_storage(TensorStorage::cpu(vec![1.0f32]), vec![], false)?;
                let half_x = crate::grad_fns::arithmetic::mul(x, &half)?;
                crate::grad_fns::arithmetic::add(&half_x, &one)
            },
            &x0,
            &[&dummy_param],
            1000,
            1e-8,
        )
        .unwrap();

        assert_approx(x_star.item().unwrap(), 2.0, 1e-4, "fixed point of 0.5x + 1");
    }

    #[test]
    fn test_fixed_point_contractive_to_zero() {
        // f(x, a) = a * x with a = 0.5, starting from x = 10
        // Fixed point: x* = 0.5 * x* => x* = 0
        let x0 = leaf_scalar(10.0, false);
        let a = leaf_scalar(0.5, false);

        let x_star = fixed_point(
            |x, params| crate::grad_fns::arithmetic::mul(x, params[0]),
            &x0,
            &[&a],
            1000,
            1e-8,
        )
        .unwrap();

        assert_approx(x_star.item().unwrap(), 0.0, 1e-4, "fixed point of 0.5*x");
    }

    #[test]
    fn test_fixed_point_tolerance() {
        // f(x) = 0.5 * x + 1, fixed point x* = 2
        // With a loose tolerance, it should converge in fewer iterations.
        let x0 = leaf_scalar(0.0, false);
        let dummy_param = leaf_scalar(1.0, false);

        let x_star = fixed_point(
            |x, _params| {
                let half = Tensor::from_storage(TensorStorage::cpu(vec![0.5f32]), vec![], false)?;
                let one = Tensor::from_storage(TensorStorage::cpu(vec![1.0f32]), vec![], false)?;
                let half_x = crate::grad_fns::arithmetic::mul(x, &half)?;
                crate::grad_fns::arithmetic::add(&half_x, &one)
            },
            &x0,
            &[&dummy_param],
            1000,
            0.1, // Loose tolerance.
        )
        .unwrap();

        // Should be close to 2.0 but not exact.
        let val = x_star.item().unwrap();
        assert!(
            (val - 2.0).abs() < 0.2,
            "loose tolerance: expected near 2.0, got {val}"
        );
    }

    #[test]
    fn test_fixed_point_max_iter_reached() {
        // f(x) = 0.99 * x + 0.01, fixed point x* = 1
        // With very few iterations, it won't converge fully.
        let x0 = leaf_scalar(0.0, false);
        let dummy_param = leaf_scalar(1.0, false);

        let err = fixed_point(
            |x, _params| {
                let scale = Tensor::from_storage(TensorStorage::cpu(vec![0.99f32]), vec![], false)?;
                let bias = Tensor::from_storage(TensorStorage::cpu(vec![0.01f32]), vec![], false)?;
                let sx = crate::grad_fns::arithmetic::mul(x, &scale)?;
                crate::grad_fns::arithmetic::add(&sx, &bias)
            },
            &x0,
            &[&dummy_param],
            5, // Very few iterations.
            1e-10,
        )
        .unwrap_err();

        let msg = format!("{err}");
        assert!(msg.contains("failed to converge after 5 iterations"));
        assert!(msg.contains("last L1 residual"));
    }

    #[test]
    fn test_fixed_point_rejects_zero_max_iter() {
        let x0 = leaf_scalar(0.0, false);
        let dummy_param = leaf_scalar(1.0, false);

        let err =
            fixed_point(|x, _params| Ok(x.clone()), &x0, &[&dummy_param], 0, 1e-8).unwrap_err();

        assert!(format!("{err}").contains("max_iter must be greater than zero"));
    }

    #[test]
    fn test_fixed_point_rejects_invalid_tolerance() {
        let x0 = leaf_scalar(0.0, false);
        let dummy_param = leaf_scalar(1.0, false);

        for tol in [-1.0, f64::NAN, f64::INFINITY] {
            let err =
                fixed_point(|x, _params| Ok(x.clone()), &x0, &[&dummy_param], 1, tol).unwrap_err();
            assert!(
                format!("{err}").contains("tol must be finite and non-negative"),
                "unexpected error for tol={tol}: {err}"
            );
        }
    }

    #[test]
    fn test_fixed_point_rejects_iterate_shape_mismatch() {
        let x0 =
            Tensor::from_storage(TensorStorage::cpu(vec![1.0f32, 2.0]), vec![2], false).unwrap();
        let dummy_param = leaf_scalar(1.0, false);

        let err = fixed_point(
            |_x, _params| Tensor::from_storage(TensorStorage::cpu(vec![1.0f32]), vec![], false),
            &x0,
            &[&dummy_param],
            4,
            1e-8,
        )
        .unwrap_err();

        let msg = format!("{err}");
        assert!(msg.contains("shape mismatch"));
        assert!(msg.contains("expected [2], got []"));
    }

    #[test]
    fn test_fixed_point_rejects_iterate_device_mismatch() {
        let x0 = leaf_scalar(0.0, false);
        let dummy_param = leaf_scalar(1.0, false);

        let err = fixed_point(
            |x, _params| x.to(Device::Meta),
            &x0,
            &[&dummy_param],
            4,
            1e-8,
        )
        .unwrap_err();

        assert!(matches!(err, FerrotorchError::DeviceMismatch { .. }));
    }

    #[test]
    fn test_fixed_point_allows_exact_zero_tolerance_convergence() {
        let x0 = leaf_scalar(3.0, false);
        let dummy_param = leaf_scalar(1.0, false);

        let x_star = fixed_point(|x, _params| Ok(x.clone()), &x0, &[&dummy_param], 1, 0.0).unwrap();

        assert!((x_star.item().unwrap() - 3.0).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // Gradient tests via implicit differentiation
    // -----------------------------------------------------------------------

    #[test]
    fn test_fixed_point_gradient_affine() {
        // f(x, a) = a * x + (1 - a)
        // Fixed point: x* = a * x* + 1 - a => x*(1 - a) = 1 - a => x* = 1
        // dx*/da = 0 (the fixed point is always 1 regardless of a)
        //
        // Actually for a < 1, let's use a different formulation:
        // f(x, b) = 0.5 * x + b
        // Fixed point: x* = 0.5 * x* + b => x* = 2b
        // dx*/db = 2
        let x0 = leaf_scalar(0.0, false);
        let b = leaf_scalar(3.0, true);

        let x_star = fixed_point(
            |x, params| {
                let half = Tensor::from_storage(TensorStorage::cpu(vec![0.5f32]), vec![], false)?;
                let half_x = crate::grad_fns::arithmetic::mul(x, &half)?;
                crate::grad_fns::arithmetic::add(&half_x, params[0])
            },
            &x0,
            &[&b],
            1000,
            1e-8,
        )
        .unwrap();

        // x* should be 2b = 6
        assert_approx(x_star.item().unwrap(), 6.0, 1e-3, "x* = 2b = 6");

        // Compute gradient: dx*/db should be 2.
        backward(&x_star).unwrap();
        let grad_b = b.grad().unwrap().unwrap();
        assert_approx(grad_b.item().unwrap(), 2.0, 0.2, "dx*/db = 2");
    }

    #[test]
    fn test_fixed_point_gradient_scaling() {
        // f(x, a) = a * x, starting from x0 = 10
        // Fixed point: x* = 0 for any |a| < 1
        // dx*/da = 0 (the fixed point is always 0)
        let x0 = leaf_scalar(10.0, false);
        let a = leaf_scalar(0.5, true);

        let x_star = fixed_point(
            |x, params| crate::grad_fns::arithmetic::mul(x, params[0]),
            &x0,
            &[&a],
            1000,
            1e-8,
        )
        .unwrap();

        // x* should be 0
        assert_approx(x_star.item().unwrap(), 0.0, 1e-3, "x* = 0");

        // Compute gradient: dx*/da = 0 since x* = 0 regardless of a.
        backward(&x_star).unwrap();
        let grad_a = a.grad().unwrap().unwrap();
        assert_approx(grad_a.item().unwrap(), 0.0, 0.1, "dx*/da = 0");
    }

    #[test]
    fn test_fixed_point_no_grad_params() {
        // If no parameter requires grad, no backward node is attached.
        let x0 = leaf_scalar(0.0, false);
        let b = leaf_scalar(3.0, false); // No grad.

        let x_star = fixed_point(
            |x, params| {
                let half = Tensor::from_storage(TensorStorage::cpu(vec![0.5f32]), vec![], false)?;
                let half_x = crate::grad_fns::arithmetic::mul(x, &half)?;
                crate::grad_fns::arithmetic::add(&half_x, params[0])
            },
            &x0,
            &[&b],
            1000,
            1e-8,
        )
        .unwrap();

        assert_approx(x_star.item().unwrap(), 6.0, 1e-3, "x* = 2b = 6");
        // No grad_fn should be attached.
        assert!(
            x_star.grad_fn().is_none(),
            "no grad_fn when params don't require grad"
        );
    }
}
