//! Numerical gradient checking utilities.
//!
//! [`gradcheck`] verifies that the analytical gradients computed by autograd
//! match finite-difference numerical gradients. This is essential for testing
//! custom backward implementations.
//!
//! Matches PyTorch's `torch.autograd.gradcheck` API.
//! ## REQ status (per `.design/ferrotorch-core/autograd/gradcheck.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | `pub fn `gradcheck`<T, F>` at `gradcheck.rs:43-184`; consumer: re-exported at `ferrotorch-core/src/autograd/mod.rs:33 pub use gradcheck::gradcheck`. Existing pub API — boundary-API grandfathering. |
//! | REQ-2 | SHIPPED | Adaptive default selection at `gradcheck.rs:54-69`; consumer: inside REQ-1. |
//! | REQ-3 | SHIPPED | Scalar-output validation at `gradcheck.rs:78-85`; consumer: inside REQ-1; covered by `test_gradcheck_non_scalar_fails` at `:247-252`. |
//! | REQ-4 | SHIPPED | Central finite difference at `gradcheck.rs:88-181`; consumer: inside REQ-1. |
//! | REQ-5 | SHIPPED | Per-element mismatch error at `gradcheck.rs:159-180`; consumer: inside REQ-1. |
//! | REQ-6 | SHIPPED | Multi-input outer-loop at `gradcheck.rs:89-181` with per-input substitution at `:128-145`; consumer: `test_gradcheck_linear_combination` and `test_gradcheck_add`. |
//!

use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::storage::TensorStorage;
use crate::tensor::Tensor;

fn finite_dtype_param<T: Float>(
    name: &'static str,
    value: f64,
    require_positive: bool,
) -> FerrotorchResult<T> {
    if !value.is_finite() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("gradcheck: {name} must be finite, got {value}"),
        });
    }
    if require_positive {
        if value <= 0.0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("gradcheck: {name} must be > 0, got {value}"),
            });
        }
    } else if value < 0.0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("gradcheck: {name} must be >= 0, got {value}"),
        });
    }
    let converted = T::from(value).ok_or_else(|| FerrotorchError::InvalidArgument {
        message: format!("gradcheck: {name}={value} cannot be represented in input dtype"),
    })?;
    if !converted.is_finite() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "gradcheck: {name}={value} is not finite after conversion to input dtype"
            ),
        });
    }
    if require_positive && converted <= <T as num_traits::Zero>::zero() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "gradcheck: {name}={value} becomes non-positive after conversion to input dtype"
            ),
        });
    }
    Ok(converted)
}

fn ensure_finite<T: Float>(
    name: &'static str,
    value: T,
    input_idx: usize,
    elem_idx: usize,
) -> FerrotorchResult<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(FerrotorchError::InvalidArgument {
            message: format!(
                "gradcheck failed at input {input_idx}, element {elem_idx}: {name} is non-finite ({value:?})"
            ),
        })
    }
}

fn scalar_output_value<T: Float>(
    output: &Tensor<T>,
    label: &'static str,
    input_idx: usize,
    elem_idx: usize,
) -> FerrotorchResult<T> {
    if output.numel() != 1 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "gradcheck: {label} at input {input_idx}, element {elem_idx} must return a scalar, got shape {:?}",
                output.shape()
            ),
        });
    }
    Ok(output.data_vec()?[0])
}

/// Check analytical gradients against numerical (finite-difference) gradients.
///
/// `func` takes a slice of input tensors and returns a scalar output.
/// `inputs` are the tensors to check gradients for (must require grad).
/// `eps` is the finite-difference step size (default: 1e-6).
/// `atol` is the absolute tolerance for comparison (default: 1e-5).
/// `rtol` is the relative tolerance for comparison (default: 1e-3).
///
/// Returns `Ok(true)` if all gradients match, `Err` with a descriptive
/// message if any gradient mismatches are found.
///
/// # Example
///
/// ```ignore
/// use ferrotorch_core::autograd::gradcheck::gradcheck;
///
/// let x = Tensor::from_storage(TensorStorage::cpu(vec![1.0f32, 2.0, 3.0]), vec![3], true)?;
/// let result = gradcheck(
///     |inputs| {
///         let x = &inputs[0];
///         // sum(x^2)
///         let x2 = ferrotorch_core::ops::elementwise::unary_map(x, |v| v * v)?;
///         ferrotorch_core::grad_fns::reduction::sum(&x2)
///     },
///     &[x],
///     None, None, None,
/// )?;
/// assert!(result);
/// ```
pub fn gradcheck<T, F>(
    func: F,
    inputs: &[Tensor<T>],
    eps: Option<f64>,
    atol: Option<f64>,
    rtol: Option<f64>,
) -> FerrotorchResult<bool>
where
    T: Float,
    F: Fn(&[Tensor<T>]) -> FerrotorchResult<Tensor<T>>,
{
    // Default eps is larger for f32 to avoid cancellation in finite differences.
    let default_eps = if std::mem::size_of::<T>() <= 4 {
        1e-3
    } else {
        1e-6
    };
    let default_atol = if std::mem::size_of::<T>() <= 4 {
        1e-3
    } else {
        1e-5
    };
    let default_rtol = if std::mem::size_of::<T>() <= 4 {
        1e-2
    } else {
        1e-3
    };
    let eps = eps.unwrap_or(default_eps);
    let atol = atol.unwrap_or(default_atol);
    let rtol = rtol.unwrap_or(default_rtol);

    let eps_t = finite_dtype_param::<T>("eps", eps, true)?;
    let atol_t = finite_dtype_param::<T>("atol", atol, false)?;
    let rtol_t = finite_dtype_param::<T>("rtol", rtol, false)?;
    let two_eps = eps_t + eps_t;

    // Step 1: Compute analytical gradients via autograd.
    let output = func(inputs)?;
    if output.numel() != 1 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "gradcheck: function must return a scalar, got shape {:?}",
                output.shape()
            ),
        });
    }
    let input_refs: Vec<&Tensor<T>> = inputs.iter().collect();
    let analytical_grads = crate::autograd::higher_order::grad(&output, &input_refs, false, false)?;

    // Step 2: For each input, compare analytical grad with numerical.
    for (input_idx, input) in inputs.iter().enumerate() {
        let analytical_grad = match analytical_grads.get(input_idx).and_then(|g| g.as_ref()) {
            Some(g) => g.clone(),
            None => {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "gradcheck: input {input_idx} has no gradient (requires_grad=false?)"
                    ),
                });
            }
        };
        if analytical_grad.shape() != input.shape() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "gradcheck: analytical gradient shape {:?} for input {input_idx} does not match input shape {:?}",
                    analytical_grad.shape(),
                    input.shape()
                ),
            });
        }
        let analytical_data = analytical_grad.data_vec()?;

        let input_data = input.data_vec()?;
        let n = input_data.len();

        // Compute numerical gradient via central difference.
        for elem_idx in 0..n {
            // f(x + eps)
            let mut perturbed_plus = input_data.clone();
            perturbed_plus[elem_idx] += eps_t;

            // f(x - eps)
            let mut perturbed_minus = input_data.clone();
            perturbed_minus[elem_idx] = perturbed_minus[elem_idx] - eps_t;

            // Create perturbed tensors (no grad needed).
            let plus_tensor = Tensor::from_storage(
                TensorStorage::on_device(perturbed_plus, input.device())?,
                input.shape().to_vec(),
                false,
            )?;
            let minus_tensor = Tensor::from_storage(
                TensorStorage::on_device(perturbed_minus, input.device())?,
                input.shape().to_vec(),
                false,
            )?;

            // Build input slices with the perturbed tensor replacing this input.
            let mut plus_inputs: Vec<Tensor<T>> = Vec::with_capacity(inputs.len());
            let mut minus_inputs: Vec<Tensor<T>> = Vec::with_capacity(inputs.len());
            for (i, inp) in inputs.iter().enumerate() {
                if i == input_idx {
                    plus_inputs.push(plus_tensor.clone());
                    minus_inputs.push(minus_tensor.clone());
                } else {
                    // Use a detached copy.
                    let data = inp.data_vec()?;
                    let t = Tensor::from_storage(
                        TensorStorage::on_device(data, inp.device())?,
                        inp.shape().to_vec(),
                        false,
                    )?;
                    plus_inputs.push(t.clone());
                    minus_inputs.push(t);
                }
            }

            let f_plus = func(&plus_inputs)?;
            let f_minus = func(&minus_inputs)?;
            let f_plus_val = scalar_output_value(&f_plus, "f(x + eps)", input_idx, elem_idx)?;
            let f_minus_val = scalar_output_value(&f_minus, "f(x - eps)", input_idx, elem_idx)?;

            // Numerical gradient: (f(x+eps) - f(x-eps)) / (2*eps)
            let numerical = (f_plus_val - f_minus_val) / two_eps;
            let analytical = analytical_data[elem_idx];
            ensure_finite("analytical gradient", analytical, input_idx, elem_idx)?;
            ensure_finite("numerical gradient", numerical, input_idx, elem_idx)?;

            // Check closeness: |a - n| <= atol + rtol * |n|
            let diff = (analytical - numerical).abs();
            let tolerance = atol_t + rtol_t * numerical.abs();
            ensure_finite("gradient difference", diff, input_idx, elem_idx)?;
            ensure_finite("comparison tolerance", tolerance, input_idx, elem_idx)?;

            if diff > tolerance {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "gradcheck failed at input {input_idx}, element {elem_idx}: \
                         analytical={analytical:?}, numerical={numerical:?}, diff={diff:?}, tol={tolerance:?}"
                    ),
                });
            }
        }
    }

    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grad_fns::{arithmetic, reduction};

    fn leaf(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
    }

    #[test]
    fn test_gradcheck_sum_of_squares() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3]);
        let result = gradcheck(
            |inputs| {
                // sum(x * x) — uses autograd-aware mul
                let x2 = arithmetic::mul(&inputs[0], &inputs[0])?;
                reduction::sum(&x2)
            },
            &[x],
            None,
            None,
            None,
        );
        assert!(result.is_ok(), "gradcheck failed: {:?}", result.err());
        assert!(result.unwrap());
    }

    #[test]
    fn test_gradcheck_linear_combination() {
        let a = leaf(&[2.0, 3.0], &[2]);
        let b = leaf(&[4.0, 5.0], &[2]);
        let result = gradcheck(
            |inputs| {
                let prod = arithmetic::mul(&inputs[0], &inputs[1])?;
                reduction::sum(&prod)
            },
            &[a, b],
            None,
            None,
            None,
        );
        assert!(result.is_ok(), "gradcheck failed: {:?}", result.err());
    }

    #[test]
    fn test_gradcheck_add() {
        let a = leaf(&[1.0, 2.0, 3.0], &[3]);
        let b = leaf(&[4.0, 5.0, 6.0], &[3]);
        let result = gradcheck(
            |inputs| {
                let s = arithmetic::add(&inputs[0], &inputs[1])?;
                reduction::sum(&s)
            },
            &[a, b],
            None,
            None,
            None,
        );
        assert!(result.is_ok(), "gradcheck failed: {:?}", result.err());
    }

    #[test]
    fn test_gradcheck_non_scalar_fails() {
        let x = leaf(&[1.0, 2.0], &[2]);
        let result = gradcheck(|inputs| Ok(inputs[0].clone()), &[x], None, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_gradcheck_rejects_invalid_difference_parameters() {
        let cases = [
            (Some(0.0), None, None, "eps must be > 0"),
            (Some(f64::NAN), None, None, "eps must be finite"),
            (Some(f64::INFINITY), None, None, "eps must be finite"),
            (Some(1.0e-50), None, None, "becomes non-positive"),
            (None, Some(f64::NAN), None, "atol must be finite"),
            (None, Some(f64::INFINITY), None, "atol must be finite"),
            (None, Some(-1.0e-5), None, "atol must be >= 0"),
            (None, None, Some(f64::NAN), "rtol must be finite"),
            (None, None, Some(f64::INFINITY), "rtol must be finite"),
            (None, None, Some(-1.0e-3), "rtol must be >= 0"),
        ];

        for (eps, atol, rtol, expected) in cases {
            let x = leaf(&[1.0, 2.0], &[2]);
            let err = gradcheck(
                |inputs| {
                    let x2 = arithmetic::mul(&inputs[0], &inputs[0])?;
                    reduction::sum(&x2)
                },
                &[x],
                eps,
                atol,
                rtol,
            )
            .expect_err("invalid gradcheck parameter must be rejected");
            assert!(
                format!("{err:?}").contains(expected),
                "expected error containing {expected:?}, got {err:?}"
            );
        }
    }

    #[test]
    fn test_gradcheck_fails_on_non_finite_analytical_gradient() {
        let x = leaf(&[0.0], &[1]);
        let err = gradcheck(
            |inputs| {
                let quotient = arithmetic::div(&inputs[0], &inputs[0])?;
                reduction::sum(&quotient)
            },
            &[x],
            None,
            None,
            None,
        )
        .expect_err("NaN analytical gradients must not compare as success");
        assert!(
            format!("{err:?}").contains("analytical gradient is non-finite"),
            "unexpected non-finite analytical error: {err:?}"
        );
    }

    #[test]
    fn test_gradcheck_fails_on_non_finite_numerical_gradient() {
        let x = leaf(&[0.0], &[1]);
        let err = gradcheck(
            |inputs| {
                let logged = crate::grad_fns::transcendental::log(&inputs[0])?;
                reduction::sum(&logged)
            },
            &[x],
            None,
            None,
            None,
        )
        .expect_err("NaN or infinite numerical gradients must not compare as success");
        let err_text = format!("{err:?}");
        assert!(
            err_text.contains("analytical gradient is non-finite")
                || err_text.contains("numerical gradient is non-finite"),
            "unexpected non-finite numerical-path error: {err:?}"
        );
    }

    #[test]
    fn test_gradcheck_validates_each_perturbed_output_is_scalar() {
        let x = leaf(&[1.0, 2.0], &[2]);
        let err = gradcheck(
            |inputs| {
                if inputs[0].data_vec()?[0] > 1.0 {
                    Ok(inputs[0].clone())
                } else {
                    reduction::sum(&inputs[0])
                }
            },
            &[x],
            None,
            None,
            None,
        )
        .expect_err("finite-difference outputs must remain scalar");
        assert!(
            format!("{err:?}").contains("f(x + eps)"),
            "unexpected perturbed-output shape error: {err:?}"
        );
    }
}
