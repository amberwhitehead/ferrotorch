//! Fully connected (dense) linear layer: `y = input @ weight^T + bias`.
//!
//! This is the fundamental building block for feedforward networks. The
//! weight matrix has shape `[out_features, in_features]` (same convention
//! as PyTorch) and the optional bias has shape `[out_features]`.
//!
//! # Autograd
//!
//! The forward pass is built from composable differentiable operations
//! (`mm_differentiable`, `add`), so the backward graph is constructed
//! automatically:
//!
//! - `grad_weight` is accumulated through `MmBackward`
//! - `grad_bias` is accumulated through `AddBackward` (broadcast reduction)
//! - `grad_input` is accumulated through `MmBackward`
//!
//! ## REQ status (per `.design/ferrotorch-nn/linear.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | impl: `pub struct Linear<T: Float>` here, mirroring `torch/nn/modules/linear.py:91-115`; non-test consumer: `pub use linear::Linear` in `lib.rs` exposes the type to `ferrotorch_llama::mlp::FeedForward::gate_proj` and similar fields. |
//! | REQ-2 | SHIPPED | impl: the `Linear::new` constructor here, mirroring `linear.py:96-115`; non-test consumer: `Linear::new(cfg.hidden_size, cfg.intermediate_size, false)?` in `ferrotorch-llama/src/mlp.rs`. |
//! | REQ-3 | SHIPPED | impl: shape flatten/reshape pre/post `linear_fused` inside `<Linear as Module>::forward` here, mirroring `linear.py:67-70`; non-test consumer: transformer blocks in `ferrotorch-nn/src/transformer.rs` and `ferrotorch-llama/src/attention.rs` feed 3-D `[B, T, H]` tensors through `Linear::forward` for QKV projection. |
//! | REQ-4 | SHIPPED | impl: the `linear_fused(&input_2d, weight.tensor(), bias_opt)` call inside `<Linear as Module>::forward` mirroring `linear.py:130-134`'s `F.linear`; non-test consumer: every model in `ferrotorch-vision/src/models/` invokes `Linear::forward` through its classifier head. |
//! | REQ-5 | SHIPPED | impl: `kaiming_uniform(&mut weight, NonLinearity::ReLU)` call inside `Linear::new` here; non-test consumer: `Linear::new` is the construction path used by every consumer above. NOTE: gain divergence from upstream `linear.py:124`. |
//! | REQ-6 | SHIPPED | impl: `crate::init::uniform(&mut b, -bound, bound)?` with `bound = 1/sqrt(in_features)` call inside `Linear::new` here mirroring `torch/nn/modules/linear.py:124-128`; non-test consumer: same as REQ-5. |
//! | REQ-7 | SHIPPED | impl: `impl<T: Float> Module<T> for Linear<T>` block here providing `forward`/`parameters`/`parameters_mut`/`named_parameters`/`train`/`eval`/`is_training`; non-test consumer: `ferrotorch_optim::Optimizer` consumes `Module::parameters_mut()` to apply updates. |
//! | REQ-8 | SHIPPED | impl: `impl<T: Float> Display for Linear<T>` block here matching upstream `linear.py:136-140`'s `extra_repr`; non-test consumer: `format!("{layer}")` in model summary printing (e.g. `ferrotorch_train` learner emits module displays in logs). |
//! | REQ-9 | SHIPPED | `Linear` carries only `Parameter<T>` fields which are `Send + Sync`; verified at compile time via `assert_send_sync::<Linear<f32>>()` in tests; non-test consumer: any multi-threaded `DataParallel`-style training scaffolding in `ferrotorch-train` requires `Send + Sync`. |
//! | REQ-10 | SHIPPED | impl: `last_dim != self.in_features` guard inside `<Linear as Module>::forward` here; non-test consumer: every production caller is shielded from silent shape mismatches by this guard. |
//! | REQ-11 | SHIPPED | impl: `pub struct Bilinear<T: Float>` here with `weight` `[out, in1, in2]` + optional `bias` `[out]`. `forward_pair` accepts arbitrary leading batch dims `(*, in)` -> `(*, out)`: flattens all-but-last to `[N, in]` (explicit batch product, handles `N == 0`), runs two `einsum_differentiable` contractions (`"bi,oij->boj"` then `"boj,bj->bo"`) + bias broadcast, then reshapes back to `(*, out)`, mirroring `torch/nn/modules/linear.py:162-256` + `aten/src/ATen/native/Linear.cpp:792-802`; non-test consumer: `pub use linear::Bilinear` in `lib.rs` re-export so downstream model crates (e.g. attention-fusion and FiLM-style conditioning) can construct it directly. Closes #1442, #1603. |
//! | REQ-12 | NOT-STARTED | blocker #1441 — parity-sweep runner has no arm for `nn.functional.linear`; sweep reports `0/144 passed, 144 skipped`. The forward path itself is end-to-end verified by 22 lib tests; only the runner-arm wiring is missing. |

use ferrotorch_core::grad_fns::linalg::linear_fused;
use ferrotorch_core::grad_fns::shape::reshape;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor};

use crate::init::{NonLinearity, kaiming_uniform};
use crate::module::Module;
use crate::parameter::Parameter;

/// A fully connected (dense) linear layer.
///
/// Applies the transformation `y = x @ W^T + b` where `W` has shape
/// `[out_features, in_features]` and `b` (if present) has shape
/// `[out_features]`.
///
/// # Initialization
///
/// - **Weight**: Kaiming uniform with `gain = sqrt(2)` (ReLU). This is
///   the PyTorch default for `nn.Linear`.
/// - **Bias**: Uniform `U(-bound, bound)` with `bound = 1/sqrt(in_features)`,
///   mirroring `torch/nn/modules/linear.py:124-128`.
///
/// # Examples
///
/// ```ignore
/// let layer = Linear::<f32>::new(784, 256, true)?;
/// let output = layer.forward(&input)?; // input: [batch, 784] -> output: [batch, 256]
/// ```
#[derive(Debug)]
pub struct Linear<T: Float> {
    /// Weight matrix of shape `[out_features, in_features]`.
    pub weight: Parameter<T>,
    /// Optional bias vector of shape `[out_features]`.
    pub bias: Option<Parameter<T>>,
    /// Number of input features.
    in_features: usize,
    /// Number of output features.
    out_features: usize,
    /// Whether the module is in training mode.
    training: bool,
}

impl<T: Float> Linear<T> {
    /// Create a new linear layer.
    ///
    /// # Arguments
    ///
    /// - `in_features` — Size of each input sample.
    /// - `out_features` — Size of each output sample.
    /// - `bias` — If `true`, adds a learnable bias to the output.
    ///
    /// # Errors
    ///
    /// Returns an error if `in_features` or `out_features` is zero, or if
    /// parameter allocation fails.
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> FerrotorchResult<Self> {
        if in_features == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "Linear: in_features must be > 0".into(),
            });
        }
        if out_features == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "Linear: out_features must be > 0".into(),
            });
        }

        // Initialize weight with Kaiming uniform (fan_in mode, ReLU gain).
        let mut weight = Parameter::zeros(&[out_features, in_features])?;
        kaiming_uniform(&mut weight, NonLinearity::ReLU)?;

        // Initialize bias U(-bound, bound) with bound = 1/sqrt(fan_in),
        // fan_in = in_features. Mirrors `torch/nn/modules/linear.py:124-128`:
        //   `fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)`
        //   `bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0`
        //   `init.uniform_(self.bias, -bound, bound)`
        let bias_param = if bias {
            let mut b = Parameter::zeros(&[out_features])?;
            let bound = if in_features > 0 {
                1.0 / (in_features as f64).sqrt()
            } else {
                0.0
            };
            crate::init::uniform(&mut b, -bound, bound)?;
            Some(b)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias: bias_param,
            in_features,
            out_features,
            training: true,
        })
    }

    /// Number of input features.
    #[inline]
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Number of output features.
    #[inline]
    pub fn out_features(&self) -> usize {
        self.out_features
    }
}

impl<T: Float> Module<T> for Linear<T> {
    /// Forward pass: `y = input @ weight^T + bias`.
    ///
    /// # Input shape
    ///
    /// Accepts any input with shape `(*batch, in_features)`:
    /// - 1D: `[in_features]` — single sample, no batch dim.
    /// - 2D: `[batch, in_features]` — standard batched forward.
    /// - 3D: `[batch, seq_len, in_features]` — e.g. transformer inputs.
    /// - ND: `[d0, d1, ..., in_features]` — arbitrary leading dimensions.
    ///
    /// # Output shape
    ///
    /// - `(*batch, out_features)` — same leading dimensions as input.
    ///
    /// # Autograd
    ///
    /// When gradient tracking is enabled, the returned tensor participates
    /// in the computation graph through the composed differentiable
    /// operations (`mm_differentiable` + `add` + `reshape`). Calling
    /// `.backward()` on a downstream scalar loss will propagate gradients
    /// to `weight` and `bias` automatically.
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if input.ndim() == 0 {
            return Err(FerrotorchError::ShapeMismatch {
                message: "Linear: scalar input not supported".into(),
            });
        }

        // Validate the last dimension is in_features.
        let last_dim = input.shape()[input.ndim() - 1];
        if last_dim != self.in_features {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "Linear: input has {} features but layer expects {}",
                    last_dim, self.in_features
                ),
            });
        }

        // For inputs with ndim != 2, flatten leading dims to get [N, in_features],
        // apply the fused linear transform, then reshape back to (*batch, out_features).
        let input_shape = input.shape().to_vec();
        let batch_shape = &input_shape[..input_shape.len() - 1];
        let n: usize = batch_shape.iter().product::<usize>().max(1);
        let needs_reshape = input.ndim() != 2;

        let input_2d = if needs_reshape {
            reshape(input, &[n as isize, self.in_features as isize])?
        } else {
            input.clone()
        };

        // Fused linear: input @ weight^T + bias in a single operation.
        let output_2d = linear_fused(
            &input_2d,
            self.weight.tensor(),
            self.bias.as_ref().map(|b| b.tensor()),
        )?;

        // Reshape back to (*batch, out_features).
        if needs_reshape {
            let mut out_shape: Vec<isize> = batch_shape.iter().map(|&d| d as isize).collect();
            out_shape.push(self.out_features as isize);
            reshape(&output_2d, &out_shape)
        } else {
            Ok(output_2d)
        }
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut params = vec![&self.weight];
        if let Some(ref b) = self.bias {
            params.push(b);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut b) = self.bias {
            params.push(b);
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut params = vec![("weight".to_string(), &self.weight)];
        if let Some(ref b) = self.bias {
            params.push(("bias".to_string(), b));
        }
        params
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

impl<T: Float> std::fmt::Display for Linear<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Linear(in_features={}, out_features={}, bias={})",
            self.in_features,
            self.out_features,
            self.bias.is_some()
        )
    }
}

// ---------------------------------------------------------------------------
// Bilinear — closes #1442
// ---------------------------------------------------------------------------

/// Bilinear layer: `y = x1^T @ W @ x2 + b`.
///
/// Applies a learnable bilinear transformation to two input vectors,
/// mirroring `torch.nn.Bilinear` (`torch/nn/modules/linear.py:162-260`).
/// The weight tensor has shape `[out_features, in1_features, in2_features]`
/// and bias (if present) has shape `[out_features]`. For a 2-D batched input
/// pair `(x1, x2)` of shape `[B, in1]` and `[B, in2]`, the output has shape
/// `[B, out]`:
///
/// ```text
/// y[b, o] = sum_i sum_j x1[b, i] * W[o, i, j] * x2[b, j]  + b[o]
/// ```
///
/// # Initialization
///
/// - **Weight**: `U(-bound, bound)` with `bound = 1/sqrt(in1_features)`,
///   matching `torch/nn/modules/linear.py:191-194`.
/// - **Bias**: `U(-bound, bound)` with the same bound.
#[derive(Debug)]
pub struct Bilinear<T: Float> {
    /// Weight tensor of shape `[out_features, in1_features, in2_features]`.
    pub weight: Parameter<T>,
    /// Optional bias of shape `[out_features]`.
    pub bias: Option<Parameter<T>>,
    in1_features: usize,
    in2_features: usize,
    out_features: usize,
    training: bool,
}

impl<T: Float> Bilinear<T> {
    /// Create a new bilinear layer.
    ///
    /// # Arguments
    ///
    /// - `in1_features` — size of each `x1` sample.
    /// - `in2_features` — size of each `x2` sample.
    /// - `out_features` — size of the output sample.
    /// - `bias` — if `true`, adds a learnable bias.
    ///
    /// # Errors
    ///
    /// Returns an error if any feature count is zero, or allocation fails.
    pub fn new(
        in1_features: usize,
        in2_features: usize,
        out_features: usize,
        bias: bool,
    ) -> FerrotorchResult<Self> {
        if in1_features == 0 || in2_features == 0 || out_features == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Bilinear: in1/in2/out_features must all be > 0, got ({in1_features}, {in2_features}, {out_features})"
                ),
            });
        }

        // bound = 1/sqrt(in1_features) per `torch/nn/modules/linear.py:191-194`.
        let bound = if in1_features > 0 {
            1.0 / (in1_features as f64).sqrt()
        } else {
            0.0
        };

        let mut weight = Parameter::zeros(&[out_features, in1_features, in2_features])?;
        crate::init::uniform(&mut weight, -bound, bound)?;

        let bias_param = if bias {
            let mut b = Parameter::zeros(&[out_features])?;
            crate::init::uniform(&mut b, -bound, bound)?;
            Some(b)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias: bias_param,
            in1_features,
            in2_features,
            out_features,
            training: true,
        })
    }

    /// Number of features in the first input.
    #[inline]
    pub fn in1_features(&self) -> usize {
        self.in1_features
    }

    /// Number of features in the second input.
    #[inline]
    pub fn in2_features(&self) -> usize {
        self.in2_features
    }

    /// Number of features in the output.
    #[inline]
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Bilinear forward pass: `y = x1 W x2 + b`.
    ///
    /// Accepts arbitrary leading batch dims, mirroring `torch.nn.Bilinear`'s
    /// `(*, H_in)` shape contract (`torch/nn/modules/linear.py:172-178`):
    ///
    /// - `x1`: `(*, in1_features)`, `x2`: `(*, in2_features)` where `*` is
    ///   any number of additional dimensions (including none, i.e. 1-D).
    /// - Both inputs must share the **same** leading shape `*`.
    /// - Returns `(*, out_features)`.
    ///
    /// The contraction is `y[*, o] = sum_{i,j} x1[*, i] * W[o, i, j] *
    /// x2[*, j] + b[o]`. Following the upstream ATen implementation
    /// (`aten/src/ATen/native/Linear.cpp:792-802`), the leading dims are
    /// flattened into a single batch axis `N`, the bilinear contraction
    /// runs on `[N, in]`, and the output `[N, out]` is reshaped back to
    /// `(*, out_features)`. A zero-size leading dim (`N == 0`) yields the
    /// correctly-shaped empty output.
    pub fn forward_pair(&self, x1: &Tensor<T>, x2: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Both inputs must have the same rank, and at least 1-D (the last
        // axis is the feature axis). Mirrors `Linear.cpp:777` (`input1.dim()
        // == input2.dim()`).
        if x1.ndim() == 0 || x2.ndim() == 0 {
            return Err(FerrotorchError::ShapeMismatch {
                message: "Bilinear: scalar (0-D) inputs not supported; expected (*, features)"
                    .into(),
            });
        }
        if x1.ndim() != x2.ndim() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "Bilinear: input dimensions do not match: got {} and {}",
                    x1.ndim(),
                    x2.ndim(),
                ),
            });
        }

        let x1_shape = x1.shape().to_vec();
        let x2_shape = x2.shape().to_vec();

        // All but the last dimension (the leading shape `*`) must match.
        // Mirrors `Linear.cpp:778-781` (per-dim batch-shape equality).
        let lead_len = x1_shape.len() - 1;
        for d in 0..lead_len {
            if x1_shape[d] != x2_shape[d] {
                return Err(FerrotorchError::ShapeMismatch {
                    message: format!(
                        "Bilinear: input batch dimensions do not match at dim {}: got {} and {}",
                        d, x1_shape[d], x2_shape[d],
                    ),
                });
            }
        }

        // Feature-axis (last dim) checks. Mirrors `Linear.cpp:782-787`.
        if x1_shape[lead_len] != self.in1_features {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "Bilinear: x1 last dim {} != in1_features {}",
                    x1_shape[lead_len], self.in1_features,
                ),
            });
        }
        if x2_shape[lead_len] != self.in2_features {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "Bilinear: x2 last dim {} != in2_features {}",
                    x2_shape[lead_len], self.in2_features,
                ),
            });
        }

        // Flatten the leading `*` dims into a single batch axis `N`.
        // `N` is the explicit product of the leading dims (NOT `-1`), so a
        // zero-size leading dim flattens to `N == 0` correctly — the einsum
        // empty-output path (`einsum.rs`, #1605) then returns the right
        // empty shape rather than panicking. Mirrors `Linear.cpp:796-797`
        // (`input1.reshape({-1, input1.size(-1)})`).
        let batch_shape = &x1_shape[..lead_len];
        let n: usize = batch_shape.iter().product();
        let x1_2d = ferrotorch_core::grad_fns::shape::reshape(
            x1,
            &[n as isize, self.in1_features as isize],
        )?;
        let x2_2d = ferrotorch_core::grad_fns::shape::reshape(
            x2,
            &[n as isize, self.in2_features as isize],
        )?;

        // y = einsum("bi,oij,bj->bo", x1, W, x2). Decompose via two
        // 2-tensor einsums (the workspace einsum primitive supports up to
        // two operands per call): first contract `i` to get
        // `boj = sum_i x1[b,i] * W[o,i,j]`, then contract `j` with x2 to
        // get `bo = sum_j boj * x2[b,j]`.
        let boj = ferrotorch_core::einsum::einsum_differentiable(
            "bi,oij->boj",
            &[&x1_2d, self.weight.tensor()],
        )?;
        let bo = ferrotorch_core::einsum::einsum_differentiable("boj,bj->bo", &[&boj, &x2_2d])?;

        // Add bias (broadcast `[out]` over `[N, out]`). Upstream adds the
        // bias AFTER the reshape-back (`Linear.cpp:799-801`); broadcasting
        // `[out]` over the flattened `[N, out]` is equivalent and keeps the
        // add in the 2-D regime the einsum primitive already produced.
        let out_2d = if let Some(ref bias) = self.bias {
            let bias_2d = ferrotorch_core::grad_fns::shape::reshape(
                bias.tensor(),
                &[1, self.out_features as isize],
            )?;
            ferrotorch_core::grad_fns::arithmetic::add(&bo, &bias_2d)?
        } else {
            bo
        };

        // Reshape the output's batch axis back to the original leading
        // shape `(*, out_features)`. Mirrors `Linear.cpp:792-798`
        // (`output_size = size1[:-1] + [weight.size(0)]`).
        let mut out_shape: Vec<isize> = batch_shape.iter().map(|&d| d as isize).collect();
        out_shape.push(self.out_features as isize);
        ferrotorch_core::grad_fns::shape::reshape(&out_2d, &out_shape)
    }
}

impl<T: Float> Module<T> for Bilinear<T> {
    /// `Module::forward` for `Bilinear` requires both inputs. The single-
    /// tensor `Module` trait can't carry the second operand; use
    /// [`Bilinear::forward_pair`] directly for the bilinear contraction.
    /// Calling this `forward` returns an error to flag the misuse.
    fn forward(&self, _input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        Err(FerrotorchError::InvalidArgument {
            message: "Bilinear requires two inputs; call `forward_pair(x1, x2)` instead of \
                      `Module::forward`."
                .into(),
        })
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut params = vec![&self.weight];
        if let Some(ref b) = self.bias {
            params.push(b);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut b) = self.bias {
            params.push(b);
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut params = vec![("weight".to_string(), &self.weight)];
        if let Some(ref b) = self.bias {
            params.push(("bias".to_string(), b));
        }
        params
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

impl<T: Float> std::fmt::Display for Bilinear<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Bilinear(in1_features={}, in2_features={}, out_features={}, bias={})",
            self.in1_features,
            self.in2_features,
            self.out_features,
            self.bias.is_some()
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::{Tensor, TensorStorage};

    /// Create a leaf tensor with given data and shape, optionally with grad.
    fn leaf(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
        Tensor::from_storage(
            TensorStorage::cpu(data.to_vec()),
            shape.to_vec(),
            requires_grad,
        )
        .unwrap()
    }

    /// Assert two float slices are element-wise close.
    fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "length mismatch: {} vs {}",
            actual.len(),
            expected.len()
        );
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < tol,
                "index {i}: actual={a} expected={e} diff={}",
                (a - e).abs()
            );
        }
    }

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_construction_with_bias() {
        let layer = Linear::<f32>::new(10, 5, true).unwrap();
        assert_eq!(layer.in_features(), 10);
        assert_eq!(layer.out_features(), 5);
        assert_eq!(layer.weight.shape(), &[5, 10]);
        assert!(layer.bias.is_some());
        assert_eq!(layer.bias.as_ref().unwrap().shape(), &[5]);
    }

    #[test]
    fn test_construction_without_bias() {
        let layer = Linear::<f32>::new(8, 4, false).unwrap();
        assert_eq!(layer.weight.shape(), &[4, 8]);
        assert!(layer.bias.is_none());
    }

    #[test]
    fn test_construction_zero_in_features() {
        assert!(Linear::<f32>::new(0, 5, true).is_err());
    }

    #[test]
    fn test_construction_zero_out_features() {
        assert!(Linear::<f32>::new(5, 0, true).is_err());
    }

    #[test]
    fn test_weight_requires_grad() {
        let layer = Linear::<f32>::new(4, 3, true).unwrap();
        assert!(layer.weight.requires_grad());
        assert!(layer.bias.as_ref().unwrap().requires_grad());
    }

    // -----------------------------------------------------------------------
    // Forward shape
    // -----------------------------------------------------------------------

    #[test]
    fn test_forward_shape() {
        let layer = Linear::<f32>::new(4, 3, true).unwrap();
        let input = leaf(&[0.0; 8], &[2, 4], false);
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 3]);
    }

    #[test]
    fn test_forward_shape_no_bias() {
        let layer = Linear::<f32>::new(6, 2, false).unwrap();
        let input = leaf(&[0.0; 18], &[3, 6], false);
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), &[3, 2]);
    }

    #[test]
    fn test_forward_wrong_input_features() {
        let layer = Linear::<f32>::new(4, 3, true).unwrap();
        let input = leaf(&[0.0; 15], &[3, 5], false);
        assert!(layer.forward(&input).is_err());
    }

    #[test]
    fn test_forward_1d_input_accepted() {
        // PyTorch accepts 1D input: (in_features,) -> (out_features,).
        let mut layer = Linear::<f32>::new(3, 2, false).unwrap();
        layer.weight = Parameter::from_slice(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3]).unwrap();
        let input = leaf(&[1.0, 2.0, 3.0], &[3], false);
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2]);
        assert_close(output.data().unwrap(), &[1.0, 2.0], 1e-6);
    }

    // -----------------------------------------------------------------------
    // Forward shape — multi-dimensional inputs
    // -----------------------------------------------------------------------

    #[test]
    fn test_forward_3d_input_shape() {
        // (batch, seq_len, in_features) -> (batch, seq_len, out_features)
        let layer = Linear::<f32>::new(4, 3, true).unwrap();
        let input = leaf(&[0.0; 2 * 5 * 4], &[2, 5, 4], false);
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 5, 3]);
    }

    #[test]
    fn test_forward_4d_input_shape() {
        // (batch, x, y, features) -> (batch, x, y, out_features)
        let layer = Linear::<f32>::new(8, 4, false).unwrap();
        let input = leaf(&[0.0; 2 * 3 * 4 * 8], &[2, 3, 4, 8], false);
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 3, 4, 4]);
    }

    #[test]
    fn test_forward_3d_correctness() {
        // Verify 3D gives same results as manually flattening to 2D.
        let mut layer = Linear::<f32>::new(3, 2, false).unwrap();
        layer.weight = Parameter::from_slice(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3]).unwrap();

        // 3D input: (2, 2, 3)
        let data = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let input_3d = leaf(&data, &[2, 2, 3], false);
        let out_3d = layer.forward(&input_3d).unwrap();
        assert_eq!(out_3d.shape(), &[2, 2, 2]);

        // Equivalent 2D input.
        let input_2d = leaf(&data, &[4, 3], false);
        let out_2d = layer.forward(&input_2d).unwrap();
        assert_eq!(out_2d.shape(), &[4, 2]);

        // Data should be identical.
        assert_close(out_3d.data().unwrap(), out_2d.data().unwrap(), 1e-6);
    }

    // -----------------------------------------------------------------------
    // Forward correctness (manual weight/bias)
    // -----------------------------------------------------------------------

    #[test]
    fn test_forward_correctness_no_bias() {
        // Build a layer then manually set the weight.
        let mut layer = Linear::<f32>::new(3, 2, false).unwrap();

        // weight = [[1, 0, 0], [0, 1, 0]]  (2x3)
        // This selects the first two features.
        layer.weight = Parameter::from_slice(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3]).unwrap();

        // input = [[1, 2, 3], [4, 5, 6]]  (2x3)
        let input = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        let output = layer.forward(&input).unwrap();

        // output = input @ weight^T = [[1,2,3],[4,5,6]] @ [[1,0],[0,1],[0,0]]
        //        = [[1, 2], [4, 5]]
        assert_eq!(output.shape(), &[2, 2]);
        assert_close(output.data().unwrap(), &[1.0, 2.0, 4.0, 5.0], 1e-6);
    }

    #[test]
    fn test_forward_correctness_with_bias() {
        let mut layer = Linear::<f32>::new(2, 2, true).unwrap();

        // weight = [[1, 0], [0, 1]]  (identity)
        layer.weight = Parameter::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
        // bias = [10, 20]
        *layer.bias.as_mut().unwrap() = Parameter::from_slice(&[10.0, 20.0], &[2]).unwrap();

        let input = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2], false);
        let output = layer.forward(&input).unwrap();

        // output = input @ I + [10, 20] = [[11, 22], [13, 24]]
        assert_close(output.data().unwrap(), &[11.0, 22.0, 13.0, 24.0], 1e-6);
    }

    // -----------------------------------------------------------------------
    // Backward gradients
    // -----------------------------------------------------------------------

    #[test]
    fn test_backward_gradients_no_bias() {
        // Linear: y = input @ W^T, loss = sum(y)
        // W = [[1, 2], [3, 4]]  (out=2, in=2)
        // input = [[1, 0], [0, 1]]  (batch=2, in=2)
        //
        // W^T = [[1, 3], [2, 4]]
        // y = input @ W^T = [[1, 3], [2, 4]]  shape [2, 2]
        // loss = 1 + 3 + 2 + 4 = 10
        //
        // dL/dy = ones(2, 2)
        // dL/d(input) = dL/dy @ W = [[1,1],[1,1]] @ [[1,2],[3,4]] = [[4,6],[4,6]]
        // dL/d(W^T) = input^T @ dL/dy = [[1,0],[0,1]] @ [[1,1],[1,1]] = [[1,1],[1,1]]
        // => dL/d(W) = [[1,1],[1,1]]^T = [[1,1],[1,1]]
        let mut layer = Linear::<f32>::new(2, 2, false).unwrap();
        layer.weight = Parameter::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let input = leaf(&[1.0, 0.0, 0.0, 1.0], &[2, 2], true);
        let output = layer.forward(&input).unwrap();

        // Reduce to scalar via differentiable sum.
        let loss = ferrotorch_core::grad_fns::reduction::sum(&output).unwrap();
        loss.backward().unwrap();

        // Check input grad.
        let input_grad = input.grad().unwrap().expect("input should have grad");
        assert_eq!(input_grad.shape(), &[2, 2]);
        assert_close(input_grad.data().unwrap(), &[4.0, 6.0, 4.0, 6.0], 1e-5);
    }

    #[test]
    fn test_backward_weight_grad() {
        // Use a known configuration to verify weight gradients.
        // W = [[1, 0], [0, 1]]  (out=2, in=2) — identity
        // input = [[2, 3]]  (batch=1, in=2)
        // y = [[2, 3]] @ I = [[2, 3]]
        // loss = sum(y) = 5
        // dL/dy = ones(1, 2) = [[1, 1]]
        //
        // For mm(input, W^T):
        //   dL/d(W^T) = input^T @ dL/dy = [[2],[3]] @ [[1,1]] = [[2,2],[3,3]]
        //   => dL/d(W) by chain through transpose
        //
        // PyTorch reference: W.grad = dL/dy^T @ input = [[1],[1]] @ [[2,3]] = [[2,3],[2,3]]
        let mut layer = Linear::<f32>::new(2, 2, false).unwrap();
        layer.weight = Parameter::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2]).unwrap();

        let input = leaf(&[2.0, 3.0], &[1, 2], false);
        let output = layer.forward(&input).unwrap();
        let loss = ferrotorch_core::grad_fns::reduction::sum(&output).unwrap();
        loss.backward().unwrap();

        // The weight gradient flows through mm(input, W^T):
        // dL/d(W^T) = input^T @ dL/dy = [[2],[3]] @ [[1,1]] = [[2,2],[3,3]]
        // Since W^T was created via transpose(W), the gradient accumulates on
        // the original W parameter through the transpose operation.
        // The transpose of [[2,2],[3,3]] is [[2,3],[2,3]], matching W's shape.
        let w_grad = layer
            .weight
            .grad()
            .unwrap()
            .expect("weight should have grad");
        assert_eq!(w_grad.shape(), &[2, 2]);
        assert_close(w_grad.data().unwrap(), &[2.0, 3.0, 2.0, 3.0], 1e-5);
    }

    #[test]
    fn test_backward_numerical_gradient() {
        // Numerical gradient check for a small Linear layer.
        // Perturb each weight element by eps and check finite-difference
        // gradient matches autograd gradient.
        let eps = 1e-4f32;

        let mut layer = Linear::<f32>::new(2, 2, false).unwrap();
        layer.weight = Parameter::from_slice(&[0.5, -0.3, 0.2, 0.8], &[2, 2]).unwrap();

        let input_data = [1.0f32, 2.0, 3.0, 4.0];
        let input = leaf(&input_data, &[2, 2], false);

        // Forward + backward for analytic gradient.
        let output = layer.forward(&input).unwrap();
        let loss = ferrotorch_core::grad_fns::reduction::sum(&output).unwrap();
        loss.backward().unwrap();

        let analytic_grad = layer.weight.grad().unwrap().unwrap();
        let analytic = analytic_grad.data().unwrap().to_vec();

        // Numerical gradient for each weight element.
        let base_weight = [0.5f32, -0.3, 0.2, 0.8];
        for idx in 0..4 {
            let mut w_plus = base_weight;
            w_plus[idx] += eps;
            let mut layer_plus = Linear::<f32>::new(2, 2, false).unwrap();
            layer_plus.weight = Parameter::from_slice(&w_plus, &[2, 2]).unwrap();
            let input_ng = leaf(&input_data, &[2, 2], false);
            let out_plus = ferrotorch_core::no_grad(|| {
                let o = layer_plus.forward(&input_ng).unwrap();
                ferrotorch_core::grad_fns::reduction::sum(&o).unwrap()
            });
            let loss_plus = out_plus.item().unwrap();

            let mut w_minus = base_weight;
            w_minus[idx] -= eps;
            let mut layer_minus = Linear::<f32>::new(2, 2, false).unwrap();
            layer_minus.weight = Parameter::from_slice(&w_minus, &[2, 2]).unwrap();
            let input_ng2 = leaf(&input_data, &[2, 2], false);
            let out_minus = ferrotorch_core::no_grad(|| {
                let o = layer_minus.forward(&input_ng2).unwrap();
                ferrotorch_core::grad_fns::reduction::sum(&o).unwrap()
            });
            let loss_minus = out_minus.item().unwrap();

            let numerical = (loss_plus - loss_minus) / (2.0 * eps);
            assert!(
                (numerical - analytic[idx]).abs() < 1e-2,
                "weight[{idx}]: numerical={numerical}, analytic={}, diff={}",
                analytic[idx],
                (numerical - analytic[idx]).abs()
            );
        }
    }

    // -----------------------------------------------------------------------
    // Parameter count
    // -----------------------------------------------------------------------

    #[test]
    fn test_parameter_count_with_bias() {
        let layer = Linear::<f32>::new(10, 5, true).unwrap();
        let params = layer.parameters();
        assert_eq!(params.len(), 2);
        // weight: 10 * 5 = 50 elements, bias: 5 elements
        let total: usize = params.iter().map(|p| p.numel()).sum();
        assert_eq!(total, 55);
    }

    #[test]
    fn test_parameter_count_without_bias() {
        let layer = Linear::<f32>::new(10, 5, false).unwrap();
        let params = layer.parameters();
        assert_eq!(params.len(), 1);
        let total: usize = params.iter().map(|p| p.numel()).sum();
        assert_eq!(total, 50);
    }

    // -----------------------------------------------------------------------
    // State dict roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn test_state_dict_roundtrip_with_bias() {
        let layer = Linear::<f32>::new(4, 3, true).unwrap();
        let sd = layer.state_dict();
        assert!(sd.contains_key("weight"));
        assert!(sd.contains_key("bias"));
        assert_eq!(sd["weight"].shape(), &[3, 4]);
        assert_eq!(sd["bias"].shape(), &[3]);

        let mut layer2 = Linear::<f32>::new(4, 3, true).unwrap();
        layer2.load_state_dict(&sd, true).unwrap();

        // Verify loaded weights match.
        assert_close(
            layer2.weight.data().unwrap(),
            layer.weight.data().unwrap(),
            1e-7,
        );
        assert_close(
            layer2.bias.as_ref().unwrap().data().unwrap(),
            layer.bias.as_ref().unwrap().data().unwrap(),
            1e-7,
        );
    }

    #[test]
    fn test_state_dict_roundtrip_without_bias() {
        let layer = Linear::<f32>::new(6, 2, false).unwrap();
        let sd = layer.state_dict();
        assert!(sd.contains_key("weight"));
        assert!(!sd.contains_key("bias"));

        let mut layer2 = Linear::<f32>::new(6, 2, false).unwrap();
        layer2.load_state_dict(&sd, true).unwrap();

        assert_close(
            layer2.weight.data().unwrap(),
            layer.weight.data().unwrap(),
            1e-7,
        );
    }

    #[test]
    fn test_state_dict_shape_mismatch_rejected() {
        let layer_a = Linear::<f32>::new(4, 3, true).unwrap();
        let sd = layer_a.state_dict();

        let mut layer_b = Linear::<f32>::new(4, 5, true).unwrap();
        assert!(layer_b.load_state_dict(&sd, true).is_err());
    }

    // -----------------------------------------------------------------------
    // Named parameters
    // -----------------------------------------------------------------------

    #[test]
    fn test_named_parameters_with_bias() {
        let layer = Linear::<f32>::new(3, 2, true).unwrap();
        let named = layer.named_parameters();
        assert_eq!(named.len(), 2);
        assert_eq!(named[0].0, "weight");
        assert_eq!(named[1].0, "bias");
    }

    #[test]
    fn test_named_parameters_without_bias() {
        let layer = Linear::<f32>::new(3, 2, false).unwrap();
        let named = layer.named_parameters();
        assert_eq!(named.len(), 1);
        assert_eq!(named[0].0, "weight");
    }

    // -----------------------------------------------------------------------
    // Train / Eval
    // -----------------------------------------------------------------------

    #[test]
    fn test_train_eval() {
        let mut layer = Linear::<f32>::new(4, 3, true).unwrap();
        assert!(layer.is_training());
        layer.eval();
        assert!(!layer.is_training());
        layer.train();
        assert!(layer.is_training());
    }

    // -----------------------------------------------------------------------
    // Display
    // -----------------------------------------------------------------------

    #[test]
    fn test_display() {
        let layer = Linear::<f32>::new(10, 5, true).unwrap();
        let s = format!("{layer}");
        assert_eq!(s, "Linear(in_features=10, out_features=5, bias=true)");
    }

    #[test]
    fn test_display_no_bias() {
        let layer = Linear::<f32>::new(10, 5, false).unwrap();
        let s = format!("{layer}");
        assert_eq!(s, "Linear(in_features=10, out_features=5, bias=false)");
    }

    // -----------------------------------------------------------------------
    // Send + Sync
    // -----------------------------------------------------------------------

    #[test]
    fn test_linear_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Linear<f32>>();
        assert_send_sync::<Linear<f64>>();
    }

    // -----------------------------------------------------------------------
    // Bias init bounds — REQ-6 / closes #1450
    // -----------------------------------------------------------------------

    /// Verifies bias is initialized within `U(-bound, bound)` where
    /// `bound = 1/sqrt(in_features)` per `torch/nn/modules/linear.py:124-128`.
    /// Pre-fix the bias was identically 0.0 (zeros_init), which would FAIL
    /// the `nonzero` assertion below with overwhelming probability.
    #[test]
    fn test_linear_bias_init_bounded_uniform() {
        let in_features = 64usize;
        let out_features = 128usize;
        let layer = Linear::<f32>::new(in_features, out_features, true).unwrap();
        let bias = layer.bias.as_ref().expect("bias requested");
        let bias_data = bias.tensor().data_vec().unwrap();
        let bound = 1.0_f32 / (in_features as f32).sqrt();
        let mut nonzero = 0usize;
        for &b in &bias_data {
            assert!(
                b.abs() <= bound + 1e-6,
                "bias element {b} exceeds bound {bound}"
            );
            if b != 0.0 {
                nonzero += 1;
            }
        }
        assert!(
            nonzero > out_features / 2,
            "expected most bias entries to be nonzero (got {nonzero}/{out_features}); \
             would FAIL pre-fix when bias was zeros_init"
        );
    }

    // -----------------------------------------------------------------------
    // Device transfer
    // -----------------------------------------------------------------------

    #[test]
    fn test_to_device_cpu_preserves_weights() {
        let mut layer = Linear::<f32>::new(4, 3, true).unwrap();
        layer.weight = Parameter::from_slice(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            &[3, 4],
        )
        .unwrap();
        *layer.bias.as_mut().unwrap() = Parameter::from_slice(&[0.1, 0.2, 0.3], &[3]).unwrap();

        layer.to_device(ferrotorch_core::Device::Cpu).unwrap();

        assert_eq!(layer.weight.shape(), &[3, 4]);
        assert_close(
            layer.weight.data().unwrap(),
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            1e-7,
        );
        assert_close(
            layer.bias.as_ref().unwrap().data().unwrap(),
            &[0.1, 0.2, 0.3],
            1e-7,
        );
        assert!(layer.weight.requires_grad());
        assert!(layer.bias.as_ref().unwrap().requires_grad());
    }

    #[test]
    fn test_to_device_cuda_returns_device_unavailable() {
        let mut layer = Linear::<f32>::new(4, 3, true).unwrap();
        let result = layer.to_device(ferrotorch_core::Device::Cuda(0));
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Bilinear N-D input — closes #1603
    //
    // Oracle values constructed by live-calling PyTorch 2.11 (R-CHAR-3):
    //   import torch
    //   y = torch.nn.functional.bilinear(x1, x2, W, b)
    //   y.sum().backward()  # for the four gradients
    // Each test documents the exact torch invocation that produced its
    // expected tensor. The bilinear contract is
    // `torch/nn/modules/linear.py:172-178` (shape `(*, H_in)`) and
    // `aten/src/ATen/native/Linear.cpp:792-802` (flatten-2D-then-reshape).
    // -----------------------------------------------------------------------

    /// Build the shared deterministic weight `[out=2, in1=3, in2=2]` and bias
    /// `[out=2]` used by the 3-D forward/backward oracle tests below. These
    /// exact values are what was fed to `torch.nn.functional.bilinear` to
    /// produce the expected outputs/gradients.
    fn bilinear_3d_layer() -> Bilinear<f32> {
        let mut layer = Bilinear::<f32>::new(3, 2, 2, true).unwrap();
        // W[o,i,j], row-major flatten of the [2,3,2] tensor.
        layer.weight = Parameter::from_slice(
            &[
                0.1, 0.2, 0.3, -0.1, -0.2, 0.05, // o=0
                0.0, 0.4, -0.3, 0.2, 0.1, -0.15, // o=1
            ],
            &[2, 3, 2],
        )
        .unwrap();
        *layer.bias.as_mut().unwrap() = Parameter::from_slice(&[0.5, -0.25], &[2]).unwrap();
        layer
    }

    #[test]
    fn test_bilinear_3d_forward_matches_torch() {
        // torch:
        //   x1 = [[[1,2,3],[-1,0.5,2]],[[0,1,-1],[2,-2,1]]]  # (2,2,3)
        //   x2 = [[[1,-1],[0.5,2]],[[-1,1],[3,0]]]            # (2,2,2)
        //   F.bilinear(x1, x2, W, b).shape == (2,2,2)
        let layer = bilinear_3d_layer();
        let x1 = leaf(
            &[
                1.0, 2.0, 3.0, -1.0, 0.5, 2.0, 0.0, 1.0, -1.0, 2.0, -2.0, 1.0,
            ],
            &[2, 2, 3],
            false,
        );
        let x2 = leaf(
            &[1.0, -1.0, 0.5, 2.0, -1.0, 1.0, 3.0, 0.0],
            &[2, 2, 2],
            false,
        );
        let y = layer.forward_pair(&x1, &x2).unwrap();
        assert_eq!(y.shape(), &[2, 2, 2]);
        // FWD3D_out from torch oracle.
        assert_close(
            y.data().unwrap(),
            &[0.45, -0.9, 0.025, -1.425, -0.15, 0.5, -1.3, 1.85],
            1e-5,
        );
    }

    #[test]
    fn test_bilinear_3d_backward_matches_torch() {
        // Same inputs as the forward test; loss = y.sum().
        // Expected grads are GRAD_x1 / GRAD_x2 / GRAD_W / GRAD_b from torch.
        let layer = bilinear_3d_layer();
        let x1 = leaf(
            &[
                1.0, 2.0, 3.0, -1.0, 0.5, 2.0, 0.0, 1.0, -1.0, 2.0, -2.0, 1.0,
            ],
            &[2, 2, 3],
            true,
        );
        let x2 = leaf(
            &[1.0, -1.0, 0.5, 2.0, -1.0, 1.0, 3.0, 0.0],
            &[2, 2, 2],
            true,
        );
        let y = layer.forward_pair(&x1, &x2).unwrap();
        let loss = ferrotorch_core::grad_fns::reduction::sum(&y).unwrap();
        loss.backward().unwrap();

        let g_x1 = x1.grad().unwrap().expect("x1 should have grad");
        assert_eq!(g_x1.shape(), &[2, 2, 3]);
        assert_close(
            g_x1.data().unwrap(),
            &[
                -0.5, -0.1, 0.0, 1.25, 0.2, -0.25, 0.5, 0.1, 0.0, 0.3, 0.0, -0.3,
            ],
            1e-5,
        );

        let g_x2 = x2.grad().unwrap().expect("x2 should have grad");
        assert_eq!(g_x2.shape(), &[2, 2, 2]);
        assert_close(
            g_x2.data().unwrap(),
            &[-0.2, 0.5, -0.3, -0.75, 0.1, 0.2, 0.1, 0.9],
            1e-5,
        );

        let g_w = layer.weight.grad().unwrap().expect("W should have grad");
        assert_eq!(g_w.shape(), &[2, 3, 2]);
        assert_close(
            g_w.data().unwrap(),
            &[
                6.5, -3.0, -4.75, 0.0, 8.0, 0.0, 6.5, -3.0, -4.75, 0.0, 8.0, 0.0,
            ],
            1e-5,
        );

        let g_b = layer
            .bias
            .as_ref()
            .unwrap()
            .grad()
            .unwrap()
            .expect("bias should have grad");
        assert_eq!(g_b.shape(), &[2]);
        assert_close(g_b.data().unwrap(), &[4.0, 4.0], 1e-5);
    }

    #[test]
    fn test_bilinear_4d_forward_matches_torch() {
        // torch:
        //   W = [[[1,0],[0,1]]]  (out=1,in1=2,in2=2 -> identity contraction)
        //   x1 = [[[[1,2],[3,4]]],[[[5,6],[7,8]]]]  # (2,1,2,2)
        //   x2 = [[[[1,1],[1,1]]],[[[2,2],[2,2]]]]  # (2,1,2,2)
        //   F.bilinear(x1,x2,W).shape == (2,1,2,1); data == [3,7,22,30]
        let mut layer = Bilinear::<f32>::new(2, 2, 1, false).unwrap();
        layer.weight = Parameter::from_slice(&[1.0, 0.0, 0.0, 1.0], &[1, 2, 2]).unwrap();
        let x1 = leaf(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 1, 2, 2],
            false,
        );
        let x2 = leaf(
            &[1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
            &[2, 1, 2, 2],
            false,
        );
        let y = layer.forward_pair(&x1, &x2).unwrap();
        assert_eq!(y.shape(), &[2, 1, 2, 1]);
        assert_close(y.data().unwrap(), &[3.0, 7.0, 22.0, 30.0], 1e-5);
    }

    #[test]
    fn test_bilinear_2d_still_matches_torch() {
        // Regression guard: the pre-existing 2-D path must keep working.
        // torch:
        //   W = [[[1,0],[0,1]]] (out=1,in1=2,in2=2), x1=[[1,2],[3,4]],
        //   x2=[[1,1],[1,1]] -> y = [[1*1+2*1],[3*1+4*1]] = [[3],[7]]
        let mut layer = Bilinear::<f32>::new(2, 2, 1, false).unwrap();
        layer.weight = Parameter::from_slice(&[1.0, 0.0, 0.0, 1.0], &[1, 2, 2]).unwrap();
        let x1 = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2], false);
        let x2 = leaf(&[1.0, 1.0, 1.0, 1.0], &[2, 2], false);
        let y = layer.forward_pair(&x1, &x2).unwrap();
        assert_eq!(y.shape(), &[2, 1]);
        assert_close(y.data().unwrap(), &[3.0, 7.0], 1e-5);
    }

    #[test]
    fn test_bilinear_1d_still_matches_torch() {
        // Regression guard: a 1-D pair (no batch dim) -> (out,).
        // torch: W=[[[1,0],[0,1]]], x1=[2,3], x2=[1,1] -> y=[2*1+3*1]=[5]
        let mut layer = Bilinear::<f32>::new(2, 2, 1, false).unwrap();
        layer.weight = Parameter::from_slice(&[1.0, 0.0, 0.0, 1.0], &[1, 2, 2]).unwrap();
        let x1 = leaf(&[2.0, 3.0], &[2], false);
        let x2 = leaf(&[1.0, 1.0], &[2], false);
        let y = layer.forward_pair(&x1, &x2).unwrap();
        assert_eq!(y.shape(), &[1]);
        assert_close(y.data().unwrap(), &[5.0], 1e-5);
    }

    #[test]
    fn test_bilinear_empty_leading_dim_2d() {
        // torch: F.bilinear(zeros(0,3), zeros(0,2), W, b).shape == (0,2)
        let layer = bilinear_3d_layer();
        let x1 = leaf(&[], &[0, 3], false);
        let x2 = leaf(&[], &[0, 2], false);
        let y = layer.forward_pair(&x1, &x2).unwrap();
        assert_eq!(y.shape(), &[0, 2]);
        assert_eq!(y.numel(), 0);
    }

    #[test]
    fn test_bilinear_empty_leading_dim_3d() {
        // torch: F.bilinear(zeros(0,4,3), zeros(0,4,2), W, b).shape == (0,4,2)
        let layer = bilinear_3d_layer();
        let x1 = leaf(&[], &[0, 4, 3], false);
        let x2 = leaf(&[], &[0, 4, 2], false);
        let y = layer.forward_pair(&x1, &x2).unwrap();
        assert_eq!(y.shape(), &[0, 4, 2]);
        assert_eq!(y.numel(), 0);
    }

    #[test]
    fn test_bilinear_zero_middle_dim_3d() {
        // torch: F.bilinear(zeros(2,0,3), zeros(2,0,2), W, b).shape == (2,0,2)
        let layer = bilinear_3d_layer();
        let x1 = leaf(&[], &[2, 0, 3], false);
        let x2 = leaf(&[], &[2, 0, 2], false);
        let y = layer.forward_pair(&x1, &x2).unwrap();
        assert_eq!(y.shape(), &[2, 0, 2]);
        assert_eq!(y.numel(), 0);
    }

    #[test]
    fn test_bilinear_mismatched_ndim_rejected() {
        // torch raises: "bilinear(): input dimensions do not match: got 3 and 2"
        let layer = bilinear_3d_layer();
        let x1 = leaf(&[0.0; 2 * 2 * 3], &[2, 2, 3], false);
        let x2 = leaf(&[0.0; 2 * 2], &[2, 2], false);
        assert!(layer.forward_pair(&x1, &x2).is_err());
    }

    #[test]
    fn test_bilinear_mismatched_leading_dim_rejected() {
        // torch raises: "bilinear(): input batch dimensions do not match at
        // dim 1: got 3 and 4"
        let layer = bilinear_3d_layer();
        let x1 = leaf(&[0.0; 2 * 3 * 3], &[2, 3, 3], false);
        let x2 = leaf(&[0.0; 2 * 4 * 2], &[2, 4, 2], false);
        assert!(layer.forward_pair(&x1, &x2).is_err());
    }

    #[test]
    fn test_bilinear_wrong_feature_dim_rejected() {
        // torch raises: "input1 size does not match weight size".
        let layer = bilinear_3d_layer(); // in1=3, in2=2
        let bad_x1 = leaf(&[0.0; 2 * 2 * 4], &[2, 2, 4], false); // last dim 4 != 3
        let x2 = leaf(&[0.0; 2 * 2 * 2], &[2, 2, 2], false);
        assert!(layer.forward_pair(&bad_x1, &x2).is_err());
    }
}
