//! Backward functions for activation operations.
//!
//! Each struct stores the tensors needed for the VJP (vector-Jacobian product)
//! and implements [`GradFn`] to participate in reverse-mode autodiff.
//!
//! ## REQ status (per `.design/ferrotorch-core/grad_fns/activation.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`relu`) | SHIPPED | `relu` + `ReluBackward` consumed by `Tensor::relu` in `methods.rs` and by the forward-AD primal in `autograd/forward_ad.rs`. |
//! | REQ-2 (`sigmoid`) | SHIPPED | `sigmoid` + `SigmoidBackward` consumed by `Tensor::sigmoid`, `ferrotorch-nn/src/rnn.rs` (gate computation), and `ferrotorch-nn/src/loss.rs` (BCE-with-logits). |
//! | REQ-3 (`tanh`) | SHIPPED | `tanh` + `TanhBackward` consumed by `Tensor::tanh_t` and `ferrotorch-nn/src/rnn.rs`. |
//! | REQ-4 (`gelu`) | SHIPPED | `gelu` / `gelu_with` + `GeluBackward` (None / Tanh / Sigmoid modes) consumed by `Tensor::gelu` / `Tensor::gelu_with`. |
//! | REQ-5 (`silu`) | SHIPPED | `silu` + `SiluBackward` consumed by `Tensor::silu` and by `ferrotorch-nn/src/transformer.rs` (SwiGLU). |
//! | REQ-6 (`softmax` last-axis) | SHIPPED | `softmax` + `SoftmaxBackward` (bf16 promotes accumulator to f32) consumed by `Tensor::softmax`, by `ferrotorch-nn/src/attention.rs`, and by `flex_attention.rs`. |
//! | REQ-7 (`log_softmax` last-axis) | SHIPPED | `log_softmax` + `LogSoftmaxBackward` consumed by `Tensor::log_softmax`. |
//! | REQ-8 (`softplus`) | SHIPPED | `softplus` + `SoftplusBackward` (threshold-branch identity, GPU backward built from primitives per #796) consumed by `ferrotorch-nn/src/functional.rs`. |
//! | REQ-9 (`elu`) | SHIPPED | `elu` + `EluBackward` consumed by `ferrotorch-nn/src/functional.rs` and the `ELU` Module. |
//! | REQ-10 (`mish`) | SHIPPED | `mish` + `MishBackward` consumed by `ferrotorch-nn/src/functional.rs`. |
//! | REQ-11 (`leaky_relu`) | SHIPPED | `leaky_relu` + `LeakyReluBackward` with resident CUDA f32/f64 paths consumed by `ferrotorch-nn/src/functional.rs`. |
//! | REQ-12 (`hardtanh` / `relu6`) | SHIPPED | `hardtanh` + `HardtanhBackward` consumed by `ferrotorch-nn/src/functional.rs` (`hardtanh` + `relu6`). |
//! | REQ-13 (`hardsigmoid`) | SHIPPED | `hardsigmoid` + `HardsigmoidBackward` consumed by `ferrotorch-nn/src/functional.rs`. |
//! | REQ-14 (`hardswish`) | SHIPPED | `hardswish` + `HardswishBackward` consumed by `ferrotorch-nn/src/functional.rs`. |
//! | REQ-15 (`selu`) | SHIPPED | `selu` + `SeluBackward` (Klambauer constants) consumed by `ferrotorch-nn/src/functional.rs`. |
//! | REQ-16 (`softsign`) | SHIPPED | `softsign` + `SoftsignBackward` consumed by `ferrotorch-nn/src/functional.rs`. |
//! | REQ-17 (`prelu`, scalar or per-channel alpha) | SHIPPED | `prelu` + `PReluBackward` (fused dual VJP routing to input + alpha) consumed by `ferrotorch-nn/src/functional.rs` and the `PReLU` Module. |
//! | REQ-18 (`glu`) | SHIPPED | `glu` + `GluBackward` consumed by `ferrotorch-nn/src/functional.rs`. |
//! | REQ-19 (`threshold`) | SHIPPED | `threshold` + `ThresholdBackward` consumed by `Tensor::threshold_t` in `methods.rs` (closes #1341 REQ-19). |
//! | REQ-20 (`rrelu`) | SHIPPED | `rrelu` + `RReluBackward` (deterministic mean-slope inference path) and `RReluTrainBackward` (training mode: per-element `Uniform[lower, upper]` slopes drawn from the process-default MT19937, bit-exact vs torch CPU under `manual_seed`, #1738) consumed by `Tensor::rrelu_t` in `methods.rs`. |
//! | REQ-21 (`celu`) | SHIPPED | `celu` + `CeluBackward` consumed by `Tensor::celu_t` in `methods.rs`. |
//! | REQ-22 (`softmin` fused) | SHIPPED | `softmin` + `SoftminBackward` (fused single-`GradFn`) consumed by `Tensor::softmin_t` in `methods.rs`; the explicit-composition `neg -> softmax` route still ships in `ferrotorch-nn`. |
//! | REQ-23 (autograd gating) | SHIPPED | every public forward checks `is_grad_enabled() && input.requires_grad()` before attaching a `*Backward` node; verified by the `test_*_no_grad` family. |

use std::any::TypeId;
use std::sync::Arc;

use crate::autograd::no_grad::is_grad_enabled;
use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::gpu_dispatch::{CompareOp, gpu_backend};
use crate::ops::elementwise::{fast_sigmoid, fast_tanh, unary_map};
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

#[inline]
fn is_bf16<T: Float>() -> bool {
    TypeId::of::<T>() == TypeId::of::<half::bf16>()
}

/// Returns `true` if `T` is `half::f16` (IEEE float16, crosslink #1185 Phase 1).
#[inline]
fn is_f16<T: Float>() -> bool {
    TypeId::of::<T>() == TypeId::of::<half::f16>()
}

#[inline]
fn is_higher_order_backward() -> bool {
    crate::autograd::higher_order::is_create_graph_enabled()
}

fn meta_unary_operation<T, F>(
    input: &Tensor<T>,
    make_grad_fn: F,
) -> FerrotorchResult<Option<Tensor<T>>>
where
    T: Float,
    F: FnOnce() -> Arc<dyn GradFn<T>>,
{
    if !input.is_meta() {
        return Ok(None);
    }
    let shape = input.shape().to_vec();
    if is_grad_enabled() && input.requires_grad() {
        Ok(Some(crate::meta_propagate::meta_operation(
            shape,
            make_grad_fn(),
        )?))
    } else {
        Ok(Some(crate::meta_propagate::meta_tensor(shape)?))
    }
}

fn meta_unary_operation_saving_output<T, F>(
    input: &Tensor<T>,
    make_grad_fn: F,
) -> FerrotorchResult<Option<Tensor<T>>>
where
    T: Float,
    F: FnOnce(Tensor<T>) -> FerrotorchResult<Arc<dyn GradFn<T>>>,
{
    if !input.is_meta() {
        return Ok(None);
    }
    let shape = input.shape().to_vec();
    if is_grad_enabled() && input.requires_grad() {
        Ok(Some(crate::meta_propagate::meta_operation_saving_output(
            shape,
            make_grad_fn,
        )?))
    } else {
        Ok(Some(crate::meta_propagate::meta_tensor(shape)?))
    }
}

fn meta_input_grad<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
    Ok(vec![Some(crate::meta_propagate::meta_tensor(
        input.shape().to_vec(),
    )?)])
}

fn cuda_scalar_to_f32<T: Float>(value: T, op: &str) -> FerrotorchResult<f32> {
    <T as num_traits::ToPrimitive>::to_f32(&value).ok_or_else(|| FerrotorchError::InvalidArgument {
        message: format!("{op}: scalar is not representable as f32"),
    })
}

fn cuda_scalar_to_f64<T: Float>(value: T, op: &str) -> FerrotorchResult<f64> {
    <T as num_traits::ToPrimitive>::to_f64(&value).ok_or_else(|| FerrotorchError::InvalidArgument {
        message: format!("{op}: scalar is not representable as f64"),
    })
}

fn full_like_detached<T: Float>(
    input: &Tensor<T>,
    value: T,
    op: &'static str,
) -> FerrotorchResult<Tensor<T>> {
    if let crate::device::Device::Cuda(ordinal) = input.device() {
        let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let handle: crate::gpu_dispatch::GpuBufferHandle = crate::dispatch_floating_dtype!(
            T,
            "activation_cuda_fill",
            f32 => backend.fill_f32(input.numel(), cuda_scalar_to_f32(value, op)?, ordinal),
            f64 => backend.fill_f64(input.numel(), cuda_scalar_to_f64(value, op)?, ordinal),
            bf16 => backend.fill_bf16_bf16(input.numel(), cuda_scalar_to_f32(value, op)?, ordinal),
            f16 => backend.fill_f16(input.numel(), cuda_scalar_to_f32(value, op)?, ordinal),
        )?;
        return Tensor::from_storage(TensorStorage::gpu(handle), input.shape().to_vec(), false);
    }

    Tensor::from_storage(
        TensorStorage::on_device(vec![value; input.numel()], input.device())?,
        input.shape().to_vec(),
        false,
    )
}

fn relu_mask<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let one = <T as num_traits::One>::one();
    let zero = <T as num_traits::Zero>::zero();

    if input.is_cuda() {
        if !(is_f32::<T>() || is_f64::<T>()) {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "relu backward",
            });
        }
        let ones = full_like_detached(input, one, "relu_backward_mask_ones")?;
        let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let result_h = if is_f32::<T>() {
            backend.relu_backward_f32(ones.gpu_handle()?, input.gpu_handle()?)?
        } else {
            backend.relu_backward_f64(ones.gpu_handle()?, input.gpu_handle()?)?
        };
        return Tensor::from_storage(TensorStorage::gpu(result_h), input.shape().to_vec(), false);
    }

    let input_data = input.data()?;
    let mask: Vec<T> = input_data
        .iter()
        .map(|&x| if x > zero { one } else { zero })
        .collect();
    Tensor::from_storage(
        TensorStorage::on_device(mask, input.device())?,
        input.shape().to_vec(),
        false,
    )
}

// ---------------------------------------------------------------------------
// ReLU
// ---------------------------------------------------------------------------

/// Backward for `relu(x)`.
///
/// VJP: `grad * (x > 0)` — the step-function mask.
#[derive(Debug)]
pub struct ReluBackward<T: Float> {
    input: Tensor<T>,
}

impl<T: Float> ReluBackward<T> {
    pub fn new(input: Tensor<T>) -> Self {
        Self { input }
    }
}

impl<T: Float> GradFn<T> for ReluBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if self.input.is_meta() {
            return meta_input_grad(&self.input);
        }
        if is_higher_order_backward() {
            if grad_output.device() != self.input.device() {
                return Err(FerrotorchError::DeviceMismatch {
                    expected: self.input.device(),
                    got: grad_output.device(),
                });
            }
            let mask = relu_mask(&self.input)?;
            let zero = full_like_detached(
                &self.input,
                <T as num_traits::Zero>::zero(),
                "relu_backward_connected_zero",
            )?;
            let zero_edge = crate::grad_fns::arithmetic::mul(&self.input, &zero)?;
            let connected_mask = crate::grad_fns::arithmetic::add(&mask, &zero_edge)?;
            let grad_input = crate::grad_fns::arithmetic::mul(grad_output, &connected_mask)?;
            return Ok(vec![Some(grad_input)]);
        }

        // GPU-native path for f32/f64
        if grad_output.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let result_h = if is_f32::<T>() {
                backend.relu_backward_f32(grad_output.gpu_handle()?, self.input.gpu_handle()?)?
            } else {
                backend.relu_backward_f64(grad_output.gpu_handle()?, self.input.gpu_handle()?)?
            };
            let grad_input = Tensor::from_storage(
                TensorStorage::gpu(result_h),
                self.input.shape().to_vec(),
                false,
            )?;
            return Ok(vec![Some(grad_input)]);
        }

        if grad_output.is_cuda() || self.input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "relu backward",
            });
        }

        let input_data = self.input.data()?;
        let grad_data = grad_output.data()?;
        let zero = <T as num_traits::Zero>::zero();

        let result: Vec<T> = input_data
            .iter()
            .zip(grad_data.iter())
            .map(|(&x, &g)| if x > zero { g } else { zero })
            .collect();

        let grad_input = Tensor::from_storage(
            TensorStorage::cpu(result),
            self.input.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "ReluBackward"
    }
}

// ---------------------------------------------------------------------------
// Sigmoid
// ---------------------------------------------------------------------------

/// Backward for `sigmoid(x)`.
///
/// VJP: `grad * s * (1 - s)` where `s = sigmoid(x)` (the output).
#[derive(Debug)]
pub struct SigmoidBackward<T: Float> {
    input: Tensor<T>,
    output: Tensor<T>,
}

impl<T: Float> SigmoidBackward<T> {
    pub fn new(input: Tensor<T>, output: Tensor<T>) -> Self {
        Self { input, output }
    }
}

impl<T: Float> GradFn<T> for SigmoidBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if self.input.is_meta() {
            return meta_input_grad(&self.input);
        }
        // GPU-native path for f32/f64
        if grad_output.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let result_h = if is_f32::<T>() {
                backend
                    .sigmoid_backward_f32(grad_output.gpu_handle()?, self.output.gpu_handle()?)?
            } else {
                backend
                    .sigmoid_backward_f64(grad_output.gpu_handle()?, self.output.gpu_handle()?)?
            };
            let grad_input = Tensor::from_storage(
                TensorStorage::gpu(result_h),
                self.input.shape().to_vec(),
                false,
            )?;
            return Ok(vec![Some(grad_input)]);
        }

        if grad_output.is_cuda() || self.output.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "sigmoid backward",
            });
        }

        let s_data = self.output.data()?;
        let grad_data = grad_output.data()?;
        let one = <T as num_traits::One>::one();

        let result: Vec<T> = s_data
            .iter()
            .zip(grad_data.iter())
            .map(|(&s, &g)| g * s * (one - s))
            .collect();

        let grad_input = Tensor::from_storage(
            TensorStorage::cpu(result),
            self.input.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "SigmoidBackward"
    }
}

// ---------------------------------------------------------------------------
// Tanh
// ---------------------------------------------------------------------------

/// Backward for `tanh(x)`.
///
/// VJP: `grad * (1 - t^2)` where `t = tanh(x)` (the output).
#[derive(Debug)]
pub struct TanhBackward<T: Float> {
    input: Tensor<T>,
    output: Tensor<T>,
}

impl<T: Float> TanhBackward<T> {
    pub fn new(input: Tensor<T>, output: Tensor<T>) -> Self {
        Self { input, output }
    }
}

impl<T: Float> GradFn<T> for TanhBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if self.input.is_meta() {
            return meta_input_grad(&self.input);
        }
        // GPU-native path for f32/f64
        if grad_output.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let result_h = if is_f32::<T>() {
                backend.tanh_backward_f32(grad_output.gpu_handle()?, self.output.gpu_handle()?)?
            } else {
                backend.tanh_backward_f64(grad_output.gpu_handle()?, self.output.gpu_handle()?)?
            };
            let grad_input = Tensor::from_storage(
                TensorStorage::gpu(result_h),
                self.input.shape().to_vec(),
                false,
            )?;
            return Ok(vec![Some(grad_input)]);
        }

        if grad_output.is_cuda() || self.output.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "tanh backward",
            });
        }

        let t_data = self.output.data()?;
        let grad_data = grad_output.data()?;
        let one = <T as num_traits::One>::one();

        let result: Vec<T> = t_data
            .iter()
            .zip(grad_data.iter())
            .map(|(&t, &g)| g * (one - t * t))
            .collect();

        let grad_input = Tensor::from_storage(
            TensorStorage::cpu(result),
            self.input.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "TanhBackward"
    }
}

// ---------------------------------------------------------------------------
// GELU — configurable approximation
// ---------------------------------------------------------------------------

/// Selects the GELU approximation method.
///
/// Matches PyTorch's `approximate` parameter on `nn.GELU`, plus the existing
/// fast sigmoid approximation as a third option.
///
/// - **`None`** (default) — exact: `x * 0.5 * (1 + erf(x / sqrt(2)))`.
///   Matches PyTorch `nn.GELU(approximate="none")`.
/// - **`Tanh`** — `x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`.
///   Matches PyTorch `nn.GELU(approximate="tanh")`.
/// - **`Sigmoid`** — `x * sigmoid(1.702 * x)`.
///   Fast approximation from the original ferrotorch implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum GeluApproximate {
    /// Exact GELU using the error function (PyTorch default).
    #[default]
    None,
    /// Tanh approximation (PyTorch `approximate="tanh"`).
    Tanh,
    /// Fast sigmoid approximation: `x * sigmoid(1.702 * x)`.
    Sigmoid,
}

impl std::fmt::Display for GeluApproximate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GeluApproximate::None => write!(f, "none"),
            GeluApproximate::Tanh => write!(f, "tanh"),
            GeluApproximate::Sigmoid => write!(f, "sigmoid"),
        }
    }
}

/// Backward for `gelu(x)`, dispatching based on approximation mode.
#[derive(Debug)]
pub struct GeluBackward<T: Float> {
    input: Tensor<T>,
    approximate: GeluApproximate,
}

impl<T: Float> GeluBackward<T> {
    pub fn new(input: Tensor<T>, approximate: GeluApproximate) -> Self {
        Self { input, approximate }
    }
}

impl<T: Float> GradFn<T> for GeluBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if self.input.is_meta() {
            return meta_input_grad(&self.input);
        }
        // GPU-native path — all approximation modes have PTX kernels.
        if grad_output.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let go_h = grad_output.gpu_handle()?;
            let in_h = self.input.gpu_handle()?;
            let result_h = if is_f32::<T>() {
                match self.approximate {
                    GeluApproximate::Sigmoid => backend.gelu_backward_f32(go_h, in_h)?,
                    GeluApproximate::Tanh => backend.gelu_backward_tanh_f32(go_h, in_h)?,
                    GeluApproximate::None => backend.gelu_backward_erf_f32(go_h, in_h)?,
                }
            } else {
                match self.approximate {
                    GeluApproximate::Sigmoid => backend.gelu_backward_f64(go_h, in_h)?,
                    GeluApproximate::Tanh => backend.gelu_backward_tanh_f64(go_h, in_h)?,
                    GeluApproximate::None => backend.gelu_backward_erf_f64(go_h, in_h)?,
                }
            };
            let grad_input = Tensor::from_storage(
                TensorStorage::gpu(result_h),
                self.input.shape().to_vec(),
                false,
            )?;
            return Ok(vec![Some(grad_input)]);
        }

        // Non-f32/f64 CUDA tensors are not supported.
        if grad_output.is_cuda() || self.input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "gelu backward",
            });
        }

        let input_data = self.input.data()?;
        let grad_data = grad_output.data()?;
        let one = <T as num_traits::One>::one();

        let result: Vec<T> = match self.approximate {
            GeluApproximate::None => {
                let sqrt_2 = T::from(std::f64::consts::SQRT_2).unwrap();
                let inv_sqrt_2pi = T::from(1.0 / (2.0 * std::f64::consts::PI).sqrt()).unwrap();
                let half = T::from(0.5).unwrap();
                input_data
                    .iter()
                    .zip(grad_data.iter())
                    .map(|(&x, &g)| {
                        let cdf = half * (one + erf_approx(x / sqrt_2));
                        let pdf = inv_sqrt_2pi * (-(x * x) / (one + one)).exp();
                        g * (cdf + x * pdf)
                    })
                    .collect()
            }
            GeluApproximate::Tanh => {
                let half = T::from(0.5).unwrap();
                let sqrt_2_over_pi = T::from((2.0 / std::f64::consts::PI).sqrt()).unwrap();
                let c = T::from(0.044715).unwrap();
                let c3 = T::from(3.0 * 0.044715).unwrap();
                input_data
                    .iter()
                    .zip(grad_data.iter())
                    .map(|(&x, &g)| {
                        let x3 = x * x * x;
                        let inner = sqrt_2_over_pi * (x + c * x3);
                        let tanh_inner = inner.tanh();
                        let dtanh = one - tanh_inner * tanh_inner;
                        let d_inner = sqrt_2_over_pi * (one + c3 * x * x);
                        g * (half * (one + tanh_inner) + half * x * dtanh * d_inner)
                    })
                    .collect()
            }
            GeluApproximate::Sigmoid => {
                let k = T::from(1.702).unwrap();
                input_data
                    .iter()
                    .zip(grad_data.iter())
                    .map(|(&x, &g)| {
                        let s = one / (one + (-k * x).exp());
                        g * (s + k * x * s * (one - s))
                    })
                    .collect()
            }
        };

        let grad_input = Tensor::from_storage(
            TensorStorage::cpu(result),
            self.input.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "GeluBackward"
    }
}

/// Approximate erf(x) for the GELU(none) forward / backward path.
///
/// Delegates to `crate::special::erf_scalar`, which dispatches per-T:
///   - f64 → SunPro fdlibm piecewise rational approximation (~1 ulp;
///     meets F64_TRANSCENDENTAL = 1e-10);
///   - f32 / bf16 → Abramowitz & Stegun 7.1.26 polynomial (~1.5e-7;
///     inside F32_TRANSCENDENTAL_CPU = 1e-5).
///
/// Pre-#792 this function held a private A&S 7.1.26 copy that left
/// gelu_none f64 at ~5e-8 residual, ~500x past the conformance gate.
/// Routing through `special::erf_scalar` keeps the f64 / f32 dispatch
/// in one place and lets gelu_none inherit the precision upgrade.
fn erf_approx<T: Float>(x: T) -> T {
    crate::special::erf_scalar(x)
}

// ---------------------------------------------------------------------------
// SiLU (Swish)
// ---------------------------------------------------------------------------

/// Backward for `silu(x) = x * sigmoid(x)`.
///
/// VJP: `grad * (s + x * s * (1 - s))` where `s = sigmoid(x)`.
#[derive(Debug)]
pub struct SiluBackward<T: Float> {
    input: Tensor<T>,
}

impl<T: Float> SiluBackward<T> {
    pub fn new(input: Tensor<T>) -> Self {
        Self { input }
    }
}

impl<T: Float> GradFn<T> for SiluBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if self.input.is_meta() {
            return meta_input_grad(&self.input);
        }
        // GPU-native path for f32/f64
        if grad_output.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let result_h = if is_f32::<T>() {
                backend.silu_backward_f32(grad_output.gpu_handle()?, self.input.gpu_handle()?)?
            } else {
                backend.silu_backward_f64(grad_output.gpu_handle()?, self.input.gpu_handle()?)?
            };
            let grad_input = Tensor::from_storage(
                TensorStorage::gpu(result_h),
                self.input.shape().to_vec(),
                false,
            )?;
            return Ok(vec![Some(grad_input)]);
        }

        // Non-f32/f64 CUDA tensors are not supported.
        if grad_output.is_cuda() || self.input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "silu backward",
            });
        }

        let input_data = self.input.data()?;
        let grad_data = grad_output.data()?;
        let one = <T as num_traits::One>::one();

        let result: Vec<T> = input_data
            .iter()
            .zip(grad_data.iter())
            .map(|(&x, &g)| {
                let s = one / (one + (-x).exp());
                g * (s + x * s * (one - s))
            })
            .collect();

        let grad_input = Tensor::from_storage(
            TensorStorage::cpu(result),
            self.input.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "SiluBackward"
    }
}

// ---------------------------------------------------------------------------
// Softmax
// ---------------------------------------------------------------------------

/// Backward for `softmax(x)` along the last axis.
///
/// VJP: `softmax * (grad - sum(grad * softmax, axis=-1, keepdim))`.
///
/// Stores the softmax **output** (not input) for efficiency.
#[derive(Debug)]
pub struct SoftmaxBackward<T: Float> {
    input: Tensor<T>,
    output: Tensor<T>,
}

impl<T: Float> SoftmaxBackward<T> {
    pub fn new(input: Tensor<T>, output: Tensor<T>) -> Self {
        Self { input, output }
    }
}

impl<T: Float> GradFn<T> for SoftmaxBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if self.input.is_meta() {
            return meta_input_grad(&self.input);
        }
        // GPU-native path for f32/f64
        if grad_output.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let cols = *self.output.shape().last().unwrap_or(&1);
            let result_h = if is_f32::<T>() {
                backend.softmax_backward_f32(
                    grad_output.gpu_handle()?,
                    self.output.gpu_handle()?,
                    cols,
                )?
            } else {
                backend.softmax_backward_f64(
                    grad_output.gpu_handle()?,
                    self.output.gpu_handle()?,
                    cols,
                )?
            };
            let grad_input = Tensor::from_storage(
                TensorStorage::gpu(result_h),
                self.input.shape().to_vec(),
                false,
            )?;
            return Ok(vec![Some(grad_input)]);
        }

        // Non-f32/f64 CUDA tensors are not supported.
        if grad_output.is_cuda() || self.output.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "softmax backward",
            });
        }

        let s_data = self.output.data()?;
        let grad_data = grad_output.data()?;
        let shape = self.output.shape();

        if shape.is_empty() {
            let zero = <T as num_traits::Zero>::zero();
            let grad_input = Tensor::from_storage(TensorStorage::cpu(vec![zero]), vec![], false)?;
            return Ok(vec![Some(grad_input)]);
        }

        let last_dim = *shape.last().unwrap();
        let outer = s_data.len() / last_dim.max(1);
        let mut result = vec![<T as num_traits::Zero>::zero(); s_data.len()];

        for i in 0..outer {
            let base = i * last_dim;
            let mut dot = <T as num_traits::Zero>::zero();
            for j in 0..last_dim {
                dot += grad_data[base + j] * s_data[base + j];
            }
            for j in 0..last_dim {
                result[base + j] = s_data[base + j] * (grad_data[base + j] - dot);
            }
        }

        let grad_input = Tensor::from_storage(
            TensorStorage::cpu(result),
            self.input.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "SoftmaxBackward"
    }
}

// ---------------------------------------------------------------------------
// LogSoftmax
// ---------------------------------------------------------------------------

/// Backward for `log_softmax(x)` along the last axis.
///
/// VJP: `grad - softmax * sum(grad, axis=-1, keepdim)`.
///
/// Stores the **softmax output** (= exp(log_softmax)) for efficiency.
#[derive(Debug)]
pub struct LogSoftmaxBackward<T: Float> {
    input: Tensor<T>,
    /// The softmax output, i.e. `exp(log_softmax(x))`.
    softmax_output: Tensor<T>,
}

impl<T: Float> LogSoftmaxBackward<T> {
    pub fn new(input: Tensor<T>, softmax_output: Tensor<T>) -> Self {
        Self {
            input,
            softmax_output,
        }
    }
}

impl<T: Float> GradFn<T> for LogSoftmaxBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // GPU-native path for f32/f64
        // log_softmax_backward: out[j] = grad[j] - softmax[j] * sum(grad) per row
        if grad_output.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let shape = self.input.shape();
            let cols = *shape.last().unwrap_or(&1);
            let result_h = if is_f32::<T>() {
                backend.log_softmax_backward_f32(
                    grad_output.gpu_handle()?,
                    self.softmax_output.gpu_handle()?,
                    cols,
                )?
            } else {
                backend.log_softmax_backward_f64(
                    grad_output.gpu_handle()?,
                    self.softmax_output.gpu_handle()?,
                    cols,
                )?
            };
            let grad_input = Tensor::from_storage(
                TensorStorage::gpu(result_h),
                self.input.shape().to_vec(),
                false,
            )?;
            return Ok(vec![Some(grad_input)]);
        }

        // Non-f32/f64 CUDA tensors are not supported.
        if grad_output.is_cuda() || self.softmax_output.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "log_softmax backward",
            });
        }

        let sm_data = self.softmax_output.data()?;
        let grad_data = grad_output.data()?;
        let shape = self.input.shape();

        if shape.is_empty() {
            let zero = <T as num_traits::Zero>::zero();
            let grad_input = Tensor::from_storage(TensorStorage::cpu(vec![zero]), vec![], false)?;
            return Ok(vec![Some(grad_input)]);
        }

        let last_dim = *shape.last().unwrap();
        let outer = sm_data.len() / last_dim.max(1);
        let mut result = vec![<T as num_traits::Zero>::zero(); sm_data.len()];

        for i in 0..outer {
            let base = i * last_dim;
            let mut sum_grad = <T as num_traits::Zero>::zero();
            for j in 0..last_dim {
                sum_grad += grad_data[base + j];
            }
            for j in 0..last_dim {
                result[base + j] = grad_data[base + j] - sm_data[base + j] * sum_grad;
            }
        }

        let grad_input = Tensor::from_storage(
            TensorStorage::cpu(result),
            self.input.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "LogSoftmaxBackward"
    }
}

// ---------------------------------------------------------------------------
// Forward activation helpers (attach grad_fn when grad is enabled)
// ---------------------------------------------------------------------------

/// Compute `relu(x)`, attaching a backward node when gradients are enabled.
pub fn relu<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) = meta_unary_operation(input, || Arc::new(ReluBackward::new(input.clone())))? {
        return Ok(out);
    }
    crate::profiler_hook::profile_op_scope("relu", "activation", &[input.shape()], || {
        relu_inner(input)
    })
}

fn relu_inner<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if input.is_cuda() && (is_f32::<T>() || is_f64::<T>() || is_bf16::<T>() || is_f16::<T>()) {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        // #1658: normalise a narrowed-offset CUDA view to a packed offset-0
        // buffer before the elementwise kernel reads element 0.
        let input = input.contiguous()?;
        let handle: crate::gpu_dispatch::GpuBufferHandle = crate::dispatch_floating_dtype!(
            T,
            "relu",
            f32 => backend.relu_f32(input.gpu_handle()?),
            f64 => backend.relu_f64(input.gpu_handle()?),
            bf16 => backend.relu_bf16_bf16(input.gpu_handle()?),
            f16 => backend.relu_f16(input.gpu_handle()?),
        )?;
        let storage = TensorStorage::gpu(handle);
        let shape = input.shape().to_vec();

        if is_grad_enabled() && input.requires_grad() {
            let grad_fn = Arc::new(ReluBackward::new(input.clone()));
            Tensor::from_operation(storage, shape, grad_fn)
        } else {
            Tensor::from_storage(storage, shape, false)
        }
    } else {
        let zero = <T as num_traits::Zero>::zero();
        let output = unary_map(input, |x| if x > zero { x } else { zero })?;

        if is_grad_enabled() && input.requires_grad() {
            let grad_fn = Arc::new(ReluBackward::new(input.clone()));
            let (storage, shape) = output.into_storage_and_shape()?;
            Tensor::from_operation(storage, shape, grad_fn)
        } else {
            Ok(output)
        }
    }
}

/// Compute `sigmoid(x)`, attaching a backward node when gradients are enabled.
pub fn sigmoid<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) = meta_unary_operation_saving_output(input, |output| {
        Ok(Arc::new(SigmoidBackward::new(input.clone(), output)))
    })? {
        return Ok(out);
    }
    crate::profiler_hook::profile_op_scope("sigmoid", "activation", &[input.shape()], || {
        sigmoid_inner(input)
    })
}

fn sigmoid_inner<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if input.is_cuda() && (is_f32::<T>() || is_f64::<T>() || is_bf16::<T>() || is_f16::<T>()) {
        let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        // #1658: normalise a narrowed-offset CUDA view to a packed offset-0
        // buffer before the elementwise kernel reads element 0.
        let input = input.contiguous()?;
        // #23: bf16 dispatch via `dispatch_floating_dtype!` — no silent CPU
        // fallback for GPU bf16 inputs.
        let handle: crate::gpu_dispatch::GpuBufferHandle = crate::dispatch_floating_dtype!(
            T,
            "sigmoid",
            f32 => backend.sigmoid_f32(input.gpu_handle()?),
            f64 => backend.sigmoid_f64(input.gpu_handle()?),
            bf16 => backend.sigmoid_bf16_bf16(input.gpu_handle()?),
            f16 => backend.sigmoid_f16(input.gpu_handle()?),
        )?;
        let storage = TensorStorage::gpu(handle);
        let shape = input.shape().to_vec();

        if is_grad_enabled() && input.requires_grad() {
            // Build the output tensor first so backward can reference sigmoid(x).
            let output = Tensor::from_storage(storage, shape.clone(), false)?;
            let grad_fn = Arc::new(SigmoidBackward::new(input.clone(), output.clone()));
            let (s, sh) = output.into_storage_and_shape()?;
            Tensor::from_operation(s, sh, grad_fn)
        } else {
            Tensor::from_storage(storage, shape, false)
        }
    } else {
        // SIMD-accelerated sigmoid with rayon parallelism for large tensors.
        let output = fast_sigmoid(input)?;

        let device = input.device();
        if is_grad_enabled() && input.requires_grad() {
            let storage = TensorStorage::on_device(output.data()?.to_vec(), device)?;
            Tensor::from_operation(
                storage,
                output.shape().to_vec(),
                Arc::new(SigmoidBackward::new(input.clone(), output.clone())),
            )
        } else if device.is_cuda() {
            output.to(device)
        } else {
            Ok(output)
        }
    }
}

/// Compute `tanh(x)`, attaching a backward node when gradients are enabled.
pub fn tanh<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) = meta_unary_operation_saving_output(input, |output| {
        Ok(Arc::new(TanhBackward::new(input.clone(), output)))
    })? {
        return Ok(out);
    }
    crate::profiler_hook::profile_op_scope("tanh", "activation", &[input.shape()], || {
        tanh_inner(input)
    })
}

fn tanh_inner<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if input.is_cuda() && (is_f32::<T>() || is_f64::<T>() || is_bf16::<T>() || is_f16::<T>()) {
        let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        // #1658: normalise a narrowed-offset CUDA view to a packed offset-0
        // buffer before the elementwise kernel reads element 0.
        let input = input.contiguous()?;
        // #23: bf16 routes through `tanh_bf16_bf16` (f32 internal via the
        // `(e^(2x)-1)/(e^(2x)+1)` PTX kernel).
        let handle: crate::gpu_dispatch::GpuBufferHandle = crate::dispatch_floating_dtype!(
            T,
            "tanh",
            f32 => backend.tanh_f32(input.gpu_handle()?),
            f64 => backend.tanh_f64(input.gpu_handle()?),
            bf16 => backend.tanh_bf16_bf16(input.gpu_handle()?),
            f16 => backend.tanh_f16(input.gpu_handle()?),
        )?;
        let storage = TensorStorage::gpu(handle);
        let shape = input.shape().to_vec();

        if is_grad_enabled() && input.requires_grad() {
            // Build output tensor so backward can reference tanh(x).
            let output = Tensor::from_storage(storage, shape.clone(), false)?;
            let grad_fn = Arc::new(TanhBackward::new(input.clone(), output.clone()));
            let (s, sh) = output.into_storage_and_shape()?;
            Tensor::from_operation(s, sh, grad_fn)
        } else {
            Tensor::from_storage(storage, shape, false)
        }
    } else {
        // SIMD-accelerated tanh with rayon parallelism for large tensors.
        let output = fast_tanh(input)?;

        let device = input.device();
        if is_grad_enabled() && input.requires_grad() {
            let storage = TensorStorage::on_device(output.data()?.to_vec(), device)?;
            Tensor::from_operation(
                storage,
                output.shape().to_vec(),
                Arc::new(TanhBackward::new(input.clone(), output.clone())),
            )
        } else if device.is_cuda() {
            output.to(device)
        } else {
            Ok(output)
        }
    }
}

/// Compute `gelu(x)` with configurable approximation, attaching a backward
/// node when gradients are enabled.
///
/// See [`GeluApproximate`] for the available modes.
pub fn gelu_with<T: Float>(
    input: &Tensor<T>,
    approximate: GeluApproximate,
) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) = meta_unary_operation(input, || {
        Arc::new(GeluBackward::new(input.clone(), approximate))
    })? {
        return Ok(out);
    }
    // GPU fast path for all approximation modes and floating dtypes.
    if input.is_cuda()
        && (is_f32::<T>() || is_f64::<T>() || is_bf16::<T>() || is_f16::<T>())
        && let Some(backend) = crate::gpu_dispatch::gpu_backend()
    {
        // #1658: normalise a narrowed-offset CUDA view to a packed offset-0
        // buffer before the elementwise kernel reads element 0.
        let input = input.contiguous()?;
        let handle: crate::gpu_dispatch::GpuBufferHandle = crate::dispatch_floating_dtype!(
            T,
            "gelu",
            f32 => match approximate {
                GeluApproximate::Sigmoid => backend.gelu_f32(input.gpu_handle()?),
                GeluApproximate::Tanh => backend.gelu_tanh_f32(input.gpu_handle()?),
                GeluApproximate::None => backend.gelu_erf_f32(input.gpu_handle()?),
            },
            f64 => match approximate {
                GeluApproximate::Sigmoid => backend.gelu_f64(input.gpu_handle()?),
                GeluApproximate::Tanh => backend.gelu_tanh_f64(input.gpu_handle()?),
                GeluApproximate::None => backend.gelu_erf_f64(input.gpu_handle()?),
            },
            bf16 => match approximate {
                GeluApproximate::Sigmoid => backend.gelu_sigmoid_bf16_bf16(input.gpu_handle()?),
                GeluApproximate::Tanh => backend.gelu_tanh_bf16_bf16(input.gpu_handle()?),
                GeluApproximate::None => backend.gelu_bf16_bf16(input.gpu_handle()?),
            },
            f16 => match approximate {
                GeluApproximate::Sigmoid => backend.gelu_sigmoid_f16(input.gpu_handle()?),
                GeluApproximate::Tanh => backend.gelu_tanh_f16(input.gpu_handle()?),
                GeluApproximate::None => backend.gelu_f16(input.gpu_handle()?),
            },
        )?;
        return if is_grad_enabled() && input.requires_grad() {
            Tensor::from_operation(
                TensorStorage::gpu(handle),
                input.shape().to_vec(),
                Arc::new(GeluBackward::new(input.clone(), approximate)),
            )
        } else {
            Tensor::from_storage(TensorStorage::gpu(handle), input.shape().to_vec(), false)
        };
    }

    // CPU path.
    let one = <T as num_traits::One>::one();
    let output = match approximate {
        GeluApproximate::None => {
            // Exact: gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
            let sqrt_2 = T::from(std::f64::consts::SQRT_2).unwrap();
            let half = T::from(0.5).unwrap();
            unary_map(input, |x| x * half * (one + erf_approx(x / sqrt_2)))?
        }
        GeluApproximate::Tanh => {
            // Tanh approx: gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
            let half = T::from(0.5).unwrap();
            let sqrt_2_over_pi = T::from((2.0 / std::f64::consts::PI).sqrt()).unwrap();
            let c = T::from(0.044715).unwrap();
            unary_map(input, |x| {
                let inner = sqrt_2_over_pi * (x + c * x * x * x);
                half * x * (one + inner.tanh())
            })?
        }
        GeluApproximate::Sigmoid => {
            let k = T::from(1.702).unwrap();
            unary_map(input, |x| {
                let s = one / (one + (-k * x).exp());
                x * s
            })?
        }
    };

    if is_grad_enabled() && input.requires_grad() {
        let (storage, shape) = output.into_storage_and_shape()?;
        Tensor::from_operation(
            storage,
            shape,
            Arc::new(GeluBackward::new(input.clone(), approximate)),
        )
    } else {
        Ok(output)
    }
}

/// Compute `gelu(x)` with the default exact (erf-based) approximation.
///
/// This matches PyTorch's `nn.GELU(approximate="none")` default. For other
/// modes, use [`gelu_with`].
pub fn gelu<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    crate::profiler_hook::profile_op_scope("gelu", "activation", &[input.shape()], || {
        gelu_with(input, GeluApproximate::default())
    })
}

/// Compute `silu(x) = x * sigmoid(x)`, attaching a backward node when
/// gradients are enabled.
pub fn silu<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) = meta_unary_operation(input, || Arc::new(SiluBackward::new(input.clone())))? {
        return Ok(out);
    }
    crate::profiler_hook::profile_op_scope("silu", "activation", &[input.shape()], || {
        silu_inner(input)
    })
}

fn silu_inner<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    // GPU fast path for all floating dtypes with resident CUDA kernels.
    if input.is_cuda()
        && (is_f32::<T>() || is_f64::<T>() || is_bf16::<T>() || is_f16::<T>())
        && let Some(backend) = crate::gpu_dispatch::gpu_backend()
    {
        // #1658: normalise a narrowed-offset CUDA view to a packed offset-0
        // buffer before the elementwise kernel reads element 0.
        let input = input.contiguous()?;
        let handle: crate::gpu_dispatch::GpuBufferHandle = crate::dispatch_floating_dtype!(
            T,
            "silu",
            f32 => backend.silu_f32(input.gpu_handle()?),
            f64 => backend.silu_f64(input.gpu_handle()?),
            bf16 => backend.silu_bf16_bf16(input.gpu_handle()?),
            f16 => backend.silu_f16(input.gpu_handle()?),
        )?;
        return if is_grad_enabled() && input.requires_grad() {
            Tensor::from_operation(
                TensorStorage::gpu(handle),
                input.shape().to_vec(),
                Arc::new(SiluBackward::new(input.clone())),
            )
        } else {
            Tensor::from_storage(TensorStorage::gpu(handle), input.shape().to_vec(), false)
        };
    }

    // CPU path
    let one = <T as num_traits::One>::one();
    let output = unary_map(input, |x| {
        let s = one / (one + (-x).exp());
        x * s
    })?;

    let device = input.device();
    if is_grad_enabled() && input.requires_grad() {
        let storage = TensorStorage::on_device(output.data()?.to_vec(), device)?;
        Tensor::from_operation(
            storage,
            output.shape().to_vec(),
            Arc::new(SiluBackward::new(input.clone())),
        )
    } else if device.is_cuda() {
        output.to(device)
    } else {
        Ok(output)
    }
}

/// Compute `softmax(x)` along the last axis, attaching a backward node when
/// gradients are enabled.
pub fn softmax<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) = meta_unary_operation_saving_output(input, |output| {
        Ok(Arc::new(SoftmaxBackward::new(input.clone(), output)))
    })? {
        return Ok(out);
    }
    crate::profiler_hook::profile_op_scope("softmax", "activation", &[input.shape()], || {
        softmax_inner(input)
    })
}

fn softmax_inner<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let shape = input.shape().to_vec();

    // GPU fast path: dispatch to native softmax kernel.
    if input.is_cuda()
        && (is_f32::<T>() || is_f64::<T>() || is_bf16::<T>() || is_f16::<T>())
        && let Some(backend) = crate::gpu_dispatch::gpu_backend()
    {
        // #1658: normalise a narrowed-offset CUDA view to a packed offset-0
        // buffer before the row-major softmax kernel reads element 0.
        let input = input.contiguous()?;
        let last_dim = *shape.last().unwrap_or(&1);
        let rows = input.numel() / last_dim.max(1);
        // #23: bf16 routes through `softmax_bf16_bf16` (existing kernel
        // from #17; this site was the missing dispatch arm).
        let handle: crate::gpu_dispatch::GpuBufferHandle = crate::dispatch_floating_dtype!(
            T,
            "softmax",
            f32 => backend.softmax_f32(input.gpu_handle()?, rows, last_dim),
            f64 => backend.softmax_f64(input.gpu_handle()?, rows, last_dim),
            bf16 => backend.softmax_bf16_bf16(input.gpu_handle()?, rows, last_dim),
            f16 => backend.softmax_f16(input.gpu_handle()?, rows, last_dim),
        )?;

        return if is_grad_enabled() && input.requires_grad() {
            // Clone the result buffer so backward can reference the output.
            let cache_handle = backend.clone_buffer(&handle)?;
            let output_cache =
                Tensor::from_storage(TensorStorage::gpu(cache_handle), shape.clone(), false)?;
            Tensor::from_operation(
                TensorStorage::gpu(handle),
                shape,
                Arc::new(SoftmaxBackward::new(input.clone(), output_cache)),
            )
        } else {
            Tensor::from_storage(TensorStorage::gpu(handle), shape, false)
        };
    }

    // CPU path.
    let data = input.data()?;

    // bf16 softmax: promote accumulator (sum_exp + division) to f32 so
    // small differences between exp() values don't collapse into the
    // same bf16 quantum.
    let is_bf16 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<half::bf16>();

    let result = if shape.is_empty() {
        vec![<T as num_traits::One>::one()]
    } else {
        let last_dim = *shape.last().unwrap();
        let outer = data.len() / last_dim.max(1);
        let mut out = vec![<T as num_traits::Zero>::zero(); data.len()];

        if is_bf16 {
            let mut scratch = vec![0.0f32; last_dim];
            for i in 0..outer {
                let base = i * last_dim;
                let mut row_max = f32::NEG_INFINITY;
                for j in 0..last_dim {
                    let v = data[base + j].to_f32().unwrap();
                    scratch[j] = v;
                    if v > row_max {
                        row_max = v;
                    }
                }
                let mut sum_exp = 0.0f32;
                for slot in &mut scratch[..last_dim] {
                    let e = (*slot - row_max).exp();
                    *slot = e;
                    sum_exp += e;
                }
                if sum_exp > 0.0 {
                    let inv = 1.0f32 / sum_exp;
                    for j in 0..last_dim {
                        out[base + j] = T::from(scratch[j] * inv).unwrap();
                    }
                } else {
                    for j in 0..last_dim {
                        out[base + j] = <T as num_traits::Zero>::zero();
                    }
                }
            }
        } else {
            for i in 0..outer {
                let base = i * last_dim;
                let mut max_val = data[base];
                for j in 1..last_dim {
                    if data[base + j] > max_val {
                        max_val = data[base + j];
                    }
                }
                let mut sum_exp = <T as num_traits::Zero>::zero();
                for j in 0..last_dim {
                    let e = (data[base + j] - max_val).exp();
                    out[base + j] = e;
                    sum_exp += e;
                }
                #[allow(clippy::assign_op_pattern)]
                for j in 0..last_dim {
                    out[base + j] = out[base + j] / sum_exp;
                }
            }
        }
        out
    };

    let output = Tensor::from_storage(TensorStorage::cpu(result), shape, false)?;

    if is_grad_enabled() && input.requires_grad() {
        Tensor::from_operation(
            TensorStorage::cpu(output.data()?.to_vec()),
            output.shape().to_vec(),
            Arc::new(SoftmaxBackward::new(input.clone(), output.clone())),
        )
    } else {
        Ok(output)
    }
}

/// Compute `log_softmax(x)` along the last axis, attaching a backward node
/// when gradients are enabled.
pub fn log_softmax<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    crate::profiler_hook::profile_op_scope("log_softmax", "activation", &[input.shape()], || {
        log_softmax_inner(input)
    })
}

fn log_softmax_inner<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let shape = input.shape();

    // GPU fast path for f32/f64
    if input.is_cuda()
        && (is_f32::<T>() || is_f64::<T>())
        && !shape.is_empty()
        && let Some(backend) = crate::gpu_dispatch::gpu_backend()
    {
        let cols = *shape.last().unwrap();
        // #1658: normalise a narrowed-offset CUDA view to a packed offset-0
        // buffer before the row-major log_softmax kernel reads element 0.
        let input = input.contiguous()?;
        let handle = if is_f32::<T>() {
            backend.log_softmax_f32(input.gpu_handle()?, cols)?
        } else {
            backend.log_softmax_f64(input.gpu_handle()?, cols)?
        };
        return if is_grad_enabled() && input.requires_grad() {
            // Compute softmax = exp(log_softmax) on GPU for backward storage
            let sm_handle = if is_f32::<T>() {
                backend.exp_f32(&handle)?
            } else {
                backend.exp_f64(&handle)?
            };
            let softmax_tensor =
                Tensor::from_storage(TensorStorage::gpu(sm_handle), shape.to_vec(), false)?;
            Tensor::from_operation(
                TensorStorage::gpu(handle),
                shape.to_vec(),
                Arc::new(LogSoftmaxBackward::new(input.clone(), softmax_tensor)),
            )
        } else {
            Tensor::from_storage(TensorStorage::gpu(handle), shape.to_vec(), false)
        };
    }

    // Non-f32/f64 CUDA tensors are not supported.
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "log_softmax" });
    }
    let data = input.data()?;

    // Compute softmax and log_softmax simultaneously for efficiency.
    let (sm_vec, lsm_vec) = if shape.is_empty() {
        // Scalar: softmax = 1, log_softmax = 0.
        (
            vec![<T as num_traits::One>::one()],
            vec![<T as num_traits::Zero>::zero()],
        )
    } else {
        let last_dim = *shape.last().unwrap();
        let outer = data.len() / last_dim.max(1);
        let mut sm = vec![<T as num_traits::Zero>::zero(); data.len()];
        let mut lsm = vec![<T as num_traits::Zero>::zero(); data.len()];

        for i in 0..outer {
            let base = i * last_dim;
            let mut max_val = data[base];
            for j in 1..last_dim {
                if data[base + j] > max_val {
                    max_val = data[base + j];
                }
            }
            let mut sum_exp = <T as num_traits::Zero>::zero();
            for j in 0..last_dim {
                let e = (data[base + j] - max_val).exp();
                sm[base + j] = e;
                sum_exp += e;
            }
            let log_sum = sum_exp.ln();
            #[allow(clippy::assign_op_pattern)]
            for j in 0..last_dim {
                sm[base + j] = sm[base + j] / sum_exp;
                lsm[base + j] = data[base + j] - max_val - log_sum;
            }
        }
        (sm, lsm)
    };

    let softmax_tensor = Tensor::from_storage(TensorStorage::cpu(sm_vec), shape.to_vec(), false)?;

    let output = Tensor::from_storage(TensorStorage::cpu(lsm_vec), shape.to_vec(), false)?;

    if is_grad_enabled() && input.requires_grad() {
        Tensor::from_operation(
            TensorStorage::cpu(output.data()?.to_vec()),
            output.shape().to_vec(),
            Arc::new(LogSoftmaxBackward::new(input.clone(), softmax_tensor)),
        )
    } else {
        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// Softplus
// ---------------------------------------------------------------------------

/// Backward for `softplus(x)` with configurable beta and threshold.
///
/// VJP: `grad * sigmoid(beta * x)`.
///
/// For the threshold branch (beta * x > threshold), the derivative is 1
/// (identity function), so grad passes through unchanged.
#[derive(Debug)]
pub struct SoftplusBackward<T: Float> {
    input: Tensor<T>,
    beta: f64,
    threshold: f64,
}

impl<T: Float> SoftplusBackward<T> {
    pub fn new(input: Tensor<T>, beta: f64, threshold: f64) -> Self {
        Self {
            input,
            beta,
            threshold,
        }
    }
}

impl<T: Float> GradFn<T> for SoftplusBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if grad_output.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let x_h = self.input.gpu_handle()?;
            let g_h = grad_output.gpu_handle()?;
            let ordinal = match grad_output.device() {
                crate::device::Device::Cuda(ordinal) => ordinal,
                _ => unreachable!("grad_output.is_cuda() checked above"),
            };
            let result_h = if is_f32::<T>() {
                let beta_x = backend.scale_f32(x_h, self.beta as f32)?;
                let threshold =
                    backend.fill_f32(self.input.numel(), self.threshold as f32, ordinal)?;
                let cond = backend.compare(&beta_x, &threshold, CompareOp::Gt)?;
                let exp_beta_x = backend.exp_f32(&beta_x)?;
                let one = backend.fill_f32(self.input.numel(), 1.0, ordinal)?;
                let denom = backend.add_f32(&exp_beta_x, &one)?;
                let nonlinear_factor = backend.div_f32(&exp_beta_x, &denom)?;
                let nonlinear = backend.mul_f32(g_h, &nonlinear_factor)?;
                backend.where_cond(&cond, g_h, &nonlinear)?
            } else {
                let beta_x = backend.scale_f64(x_h, self.beta)?;
                let threshold = backend.fill_f64(self.input.numel(), self.threshold, ordinal)?;
                let cond = backend.compare(&beta_x, &threshold, CompareOp::Gt)?;
                let exp_beta_x = backend.exp_f64(&beta_x)?;
                let one = backend.fill_f64(self.input.numel(), 1.0, ordinal)?;
                let denom = backend.add_f64(&exp_beta_x, &one)?;
                let nonlinear_factor = backend.div_f64(&exp_beta_x, &denom)?;
                let nonlinear = backend.mul_f64(g_h, &nonlinear_factor)?;
                backend.where_cond(&cond, g_h, &nonlinear)?
            };
            let grad_input = Tensor::from_storage(
                TensorStorage::gpu(result_h),
                self.input.shape().to_vec(),
                false,
            )?;
            return Ok(vec![Some(grad_input)]);
        }

        // Non-f32/f64 CUDA tensors are not supported.
        if grad_output.is_cuda() || self.input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "softplus backward",
            });
        }

        let input_data = self.input.data()?;
        let grad_data = grad_output.data()?;
        let one = <T as num_traits::One>::one();
        let beta = T::from(self.beta).unwrap();
        let threshold = T::from(self.threshold).unwrap();

        let result: Vec<T> = input_data
            .iter()
            .zip(grad_data.iter())
            .map(|(&x, &g)| {
                let bx = beta * x;
                if bx > threshold {
                    g
                } else {
                    let sig = one / (one + (-bx).exp());
                    g * sig
                }
            })
            .collect();

        let grad_input = Tensor::from_storage(
            TensorStorage::cpu(result),
            self.input.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "SoftplusBackward"
    }
}

/// Compute `softplus(x)` with configurable `beta` and `threshold`, attaching
/// a backward node when gradients are enabled.
///
/// ```text
/// softplus(x) = log(1 + exp(beta * x)) / beta
/// ```
///
/// For numerical stability, when `beta * x > threshold` the output is `x`.
pub fn softplus<T: Float>(
    input: &Tensor<T>,
    beta: f64,
    threshold: f64,
) -> FerrotorchResult<Tensor<T>> {
    if input.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
        let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let input = input.contiguous()?;
        let input_h = input.gpu_handle()?;
        let ordinal = match input.device() {
            crate::device::Device::Cuda(ordinal) => ordinal,
            _ => unreachable!("input.is_cuda() checked above"),
        };
        let handle = if is_f32::<T>() {
            let beta_x = backend.scale_f32(input_h, beta as f32)?;
            let threshold_h = backend.fill_f32(input.numel(), threshold as f32, ordinal)?;
            let cond = backend.compare(&beta_x, &threshold_h, CompareOp::Gt)?;
            let exp_beta_x = backend.exp_f32(&beta_x)?;
            let one = backend.fill_f32(input.numel(), 1.0, ordinal)?;
            let one_plus = backend.add_f32(&one, &exp_beta_x)?;
            let log = backend.log_f32(&one_plus)?;
            let nonlinear = backend.scale_f32(&log, (1.0 / beta) as f32)?;
            backend.where_cond(&cond, input_h, &nonlinear)?
        } else {
            let beta_x = backend.scale_f64(input_h, beta)?;
            let threshold_h = backend.fill_f64(input.numel(), threshold, ordinal)?;
            let cond = backend.compare(&beta_x, &threshold_h, CompareOp::Gt)?;
            let exp_beta_x = backend.exp_f64(&beta_x)?;
            let one = backend.fill_f64(input.numel(), 1.0, ordinal)?;
            let one_plus = backend.add_f64(&one, &exp_beta_x)?;
            let log = backend.log_f64(&one_plus)?;
            let nonlinear = backend.scale_f64(&log, 1.0 / beta)?;
            backend.where_cond(&cond, input_h, &nonlinear)?
        };
        let storage = TensorStorage::gpu(handle);
        let shape = input.shape().to_vec();
        return if is_grad_enabled() && input.requires_grad() {
            Tensor::from_operation(
                storage,
                shape,
                Arc::new(SoftplusBackward::new(input.clone(), beta, threshold)),
            )
        } else {
            Tensor::from_storage(storage, shape, false)
        };
    }

    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "softplus" });
    }

    let beta_t = T::from(beta).unwrap();
    let threshold_t = T::from(threshold).unwrap();

    let output = unary_map(input, |x| {
        let bx = beta_t * x;
        if bx > threshold_t {
            x
        } else {
            bx.exp().ln_1p() / beta_t
        }
    })?;

    if is_grad_enabled() && input.requires_grad() {
        let (storage, shape) = output.into_storage_and_shape()?;
        Tensor::from_operation(
            storage,
            shape,
            Arc::new(SoftplusBackward::new(input.clone(), beta, threshold)),
        )
    } else {
        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// ELU
// ---------------------------------------------------------------------------

/// Backward for `elu(x)` with configurable alpha.
///
/// VJP:
/// - For `x > 0`: `grad * 1`
/// - For `x <= 0`: `grad * alpha * exp(x)` (equivalently `grad * (output + alpha)`)
#[derive(Debug)]
pub struct EluBackward<T: Float> {
    input: Tensor<T>,
    alpha: f64,
}

impl<T: Float> EluBackward<T> {
    pub fn new(input: Tensor<T>, alpha: f64) -> Self {
        Self { input, alpha }
    }
}

impl<T: Float> GradFn<T> for EluBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // GPU-native path for f32/f64
        if grad_output.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let result_h = if is_f32::<T>() {
                backend.elu_backward_f32(
                    grad_output.gpu_handle()?,
                    self.input.gpu_handle()?,
                    self.alpha as f32,
                )?
            } else {
                backend.elu_backward_f64(
                    grad_output.gpu_handle()?,
                    self.input.gpu_handle()?,
                    self.alpha,
                )?
            };
            let grad_input = Tensor::from_storage(
                TensorStorage::gpu(result_h),
                self.input.shape().to_vec(),
                false,
            )?;
            return Ok(vec![Some(grad_input)]);
        }

        // Non-f32/f64 CUDA tensors are not supported.
        if grad_output.is_cuda() || self.input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda { op: "elu backward" });
        }

        let input_data = self.input.data()?;
        let grad_data = grad_output.data()?;
        let zero = <T as num_traits::Zero>::zero();
        let alpha = T::from(self.alpha).unwrap();

        let result: Vec<T> = input_data
            .iter()
            .zip(grad_data.iter())
            .map(|(&x, &g)| if x > zero { g } else { g * alpha * x.exp() })
            .collect();

        let grad_input = Tensor::from_storage(
            TensorStorage::cpu(result),
            self.input.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "EluBackward"
    }
}

/// Compute `elu(x)` with configurable `alpha`, attaching a backward node
/// when gradients are enabled.
///
/// ```text
/// elu(x) = x                    if x > 0
///        = alpha * (exp(x) - 1)  if x <= 0
/// ```
pub fn elu<T: Float>(input: &Tensor<T>, alpha: f64) -> FerrotorchResult<Tensor<T>> {
    // GPU fast path for f32/f64
    if input.is_cuda()
        && (is_f32::<T>() || is_f64::<T>())
        && let Some(backend) = crate::gpu_dispatch::gpu_backend()
    {
        // #1658: normalise a narrowed-offset CUDA view to a packed offset-0
        // buffer before the elementwise kernel reads element 0.
        let input = input.contiguous()?;
        let handle = if is_f32::<T>() {
            backend.elu_f32(input.gpu_handle()?, alpha as f32)?
        } else {
            backend.elu_f64(input.gpu_handle()?, alpha)?
        };
        return if is_grad_enabled() && input.requires_grad() {
            Tensor::from_operation(
                TensorStorage::gpu(handle),
                input.shape().to_vec(),
                Arc::new(EluBackward::new(input.clone(), alpha)),
            )
        } else {
            Tensor::from_storage(TensorStorage::gpu(handle), input.shape().to_vec(), false)
        };
    }

    // CPU path
    let zero = <T as num_traits::Zero>::zero();
    let one = <T as num_traits::One>::one();
    let alpha_t = T::from(alpha).unwrap();

    let output = unary_map(input, |x| {
        if x > zero {
            x
        } else {
            alpha_t * (x.exp() - one)
        }
    })?;

    if is_grad_enabled() && input.requires_grad() {
        Tensor::from_operation(
            TensorStorage::cpu(output.data()?.to_vec()),
            output.shape().to_vec(),
            Arc::new(EluBackward::new(input.clone(), alpha)),
        )
    } else {
        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// Mish
// ---------------------------------------------------------------------------

/// Backward for `mish(x) = x * tanh(softplus(x))`.
///
/// Let `sp = softplus(x) = ln(1 + exp(x))` and `t = tanh(sp)`.
///
/// The derivative is:
/// ```text
/// d/dx mish(x) = t + x * (1 - t^2) * sigmoid(x)
///              = t + x * sigmoid(x) * sech^2(sp)
/// ```
///
/// which simplifies to: `tanh(sp) + x * sigmoid(x) * (1 - tanh(sp)^2)`.
#[derive(Debug)]
pub struct MishBackward<T: Float> {
    input: Tensor<T>,
}

impl<T: Float> MishBackward<T> {
    pub fn new(input: Tensor<T>) -> Self {
        Self { input }
    }
}

impl<T: Float> GradFn<T> for MishBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // GPU-native path for f32/f64
        if grad_output.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let result_h = if is_f32::<T>() {
                backend.mish_backward_f32(grad_output.gpu_handle()?, self.input.gpu_handle()?)?
            } else {
                backend.mish_backward_f64(grad_output.gpu_handle()?, self.input.gpu_handle()?)?
            };
            let grad_input = Tensor::from_storage(
                TensorStorage::gpu(result_h),
                self.input.shape().to_vec(),
                false,
            )?;
            return Ok(vec![Some(grad_input)]);
        }

        // Non-f32/f64 CUDA tensors are not supported.
        if grad_output.is_cuda() || self.input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "mish backward",
            });
        }

        let input_data = self.input.data()?;
        let grad_data = grad_output.data()?;
        let one = <T as num_traits::One>::one();

        let result: Vec<T> = input_data
            .iter()
            .zip(grad_data.iter())
            .map(|(&x, &g)| {
                let sp = (one + x.exp()).ln();
                let t = sp.tanh();
                let sig = one / (one + (-x).exp());
                let dmish = t + x * sig * (one - t * t);
                g * dmish
            })
            .collect();

        let grad_input = Tensor::from_storage(
            TensorStorage::cpu(result),
            self.input.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "MishBackward"
    }
}

/// Compute `mish(x) = x * tanh(softplus(x))`, attaching a backward node
/// when gradients are enabled.
pub fn mish<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    // GPU fast path for f32/f64
    if input.is_cuda()
        && (is_f32::<T>() || is_f64::<T>())
        && let Some(backend) = crate::gpu_dispatch::gpu_backend()
    {
        // #1658: normalise a narrowed-offset CUDA view to a packed offset-0
        // buffer before the elementwise kernel reads element 0.
        let input = input.contiguous()?;
        let handle = if is_f32::<T>() {
            backend.mish_f32(input.gpu_handle()?)?
        } else {
            backend.mish_f64(input.gpu_handle()?)?
        };
        return if is_grad_enabled() && input.requires_grad() {
            Tensor::from_operation(
                TensorStorage::gpu(handle),
                input.shape().to_vec(),
                Arc::new(MishBackward::new(input.clone())),
            )
        } else {
            Tensor::from_storage(TensorStorage::gpu(handle), input.shape().to_vec(), false)
        };
    }

    // CPU path. Use `ln_1p(exp(x))` (= `log1p(exp(x))`) for the softplus
    // intermediate to match upstream `mish_kernel` at
    // `aten/src/ATen/native/cpu/Activation.cpp:1228-1230`:
    //   `return static_cast<scalar_t>(x * std::tanh(std::log1p(std::exp(x))));`
    // The pre-fix path computed `(1 + exp(x)).ln()` which drifts in f32 for
    // |x| > 5 (loss of significance in `1 + epsilon` when `epsilon = exp(-x)`
    // is much smaller than 1). Switching to `ln_1p` keeps full f32 precision
    // and matches upstream byte-for-byte in the regime op_db's sample_inputs
    // tests (verified by parity-sweep --op mish: pre-fix 1/3 sample diff up
    // to 3.7e-7, post-fix 0 failures).

    let output = unary_map(input, |x| x * x.exp().ln_1p().tanh())?;

    if is_grad_enabled() && input.requires_grad() {
        Tensor::from_operation(
            TensorStorage::cpu(output.data()?.to_vec()),
            output.shape().to_vec(),
            Arc::new(MishBackward::new(input.clone())),
        )
    } else {
        Ok(output)
    }
}

// ===========================================================================
// Activation tail (#594) — native fused GradFns for the activations that
// are otherwise reachable via composition from `ferrotorch-nn::functional`.
// The fused versions skip a layer of intermediate tensor allocation and
// keep the backward pass at O(1) rather than O(k) extra ops.
//
// CPU plus resident CUDA f32/f64 implementations for the supported scalar
// activation family. Unsupported CUDA dtypes return explicit errors rather
// than routing through host closures.
// ===========================================================================

// --- leaky_relu ------------------------------------------------------------

/// Backward for `leaky_relu(x; negative_slope)`.
/// VJP: grad * (1 if x > 0 else negative_slope).
#[derive(Debug)]
pub struct LeakyReluBackward<T: Float> {
    input: Tensor<T>,
    negative_slope: f64,
}

impl<T: Float> LeakyReluBackward<T> {
    pub fn new(input: Tensor<T>, negative_slope: f64) -> Self {
        Self {
            input,
            negative_slope,
        }
    }
}

impl<T: Float> GradFn<T> for LeakyReluBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let zero = <T as num_traits::Zero>::zero();
        let one = <T as num_traits::One>::one();
        let slope = T::from(self.negative_slope).unwrap();

        if self.input.is_cuda() || grad_output.is_cuda() {
            if self.input.device() != grad_output.device() {
                return Err(FerrotorchError::DeviceMismatch {
                    expected: self.input.device(),
                    got: grad_output.device(),
                });
            }
            if !(is_f32::<T>() || is_f64::<T>()) {
                return Err(FerrotorchError::NotImplementedOnCuda {
                    op: "leaky_relu backward",
                });
            }
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let result_h = if is_f32::<T>() {
                backend.leaky_relu_backward_f32(
                    grad_output.gpu_handle()?,
                    self.input.gpu_handle()?,
                    self.negative_slope as f32,
                )?
            } else {
                backend.leaky_relu_backward_f64(
                    grad_output.gpu_handle()?,
                    self.input.gpu_handle()?,
                    self.negative_slope,
                )?
            };
            let grad_input = Tensor::from_storage(
                TensorStorage::gpu(result_h),
                self.input.shape().to_vec(),
                false,
            )?;
            return Ok(vec![Some(grad_input)]);
        }

        let input_data = self.input.data()?;
        let grad_data = grad_output.data()?;
        let result: Vec<T> = input_data
            .iter()
            .zip(grad_data.iter())
            .map(|(&x, &g)| if x > zero { g * one } else { g * slope })
            .collect();
        let grad_input = Tensor::from_storage(
            TensorStorage::cpu(result),
            self.input.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "LeakyReluBackward"
    }
}

/// Native `leaky_relu(x; negative_slope) = max(0, x) + negative_slope * min(0, x)`.
pub fn leaky_relu<T: Float>(input: &Tensor<T>, negative_slope: f64) -> FerrotorchResult<Tensor<T>> {
    if input.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
        let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let input = input.contiguous()?;
        let handle = if is_f32::<T>() {
            backend.leaky_relu_f32(input.gpu_handle()?, negative_slope as f32)?
        } else {
            backend.leaky_relu_f64(input.gpu_handle()?, negative_slope)?
        };
        let storage = TensorStorage::gpu(handle);
        let shape = input.shape().to_vec();
        return if is_grad_enabled() && input.requires_grad() {
            Tensor::from_operation(
                storage,
                shape,
                Arc::new(LeakyReluBackward::new(input.clone(), negative_slope)),
            )
        } else {
            Tensor::from_storage(storage, shape, false)
        };
    }

    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "leaky_relu" });
    }

    let zero = <T as num_traits::Zero>::zero();
    let slope = T::from(negative_slope).unwrap();
    let output = unary_map(input, |x| if x > zero { x } else { slope * x })?;
    if is_grad_enabled() && input.requires_grad() {
        let (storage, shape) = output.into_storage_and_shape()?;
        Tensor::from_operation(
            storage,
            shape,
            Arc::new(LeakyReluBackward::new(input.clone(), negative_slope)),
        )
    } else {
        Ok(output)
    }
}

// --- hardtanh --------------------------------------------------------------

/// Backward for `hardtanh(x; min, max) = clamp(x, min, max)`.
/// VJP: grad if min < x < max else 0.
#[derive(Debug)]
pub struct HardtanhBackward<T: Float> {
    input: Tensor<T>,
    min_val: f64,
    max_val: f64,
}

impl<T: Float> HardtanhBackward<T> {
    pub fn new(input: Tensor<T>, min_val: f64, max_val: f64) -> Self {
        Self {
            input,
            min_val,
            max_val,
        }
    }
}

impl<T: Float> GradFn<T> for HardtanhBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let zero = <T as num_traits::Zero>::zero();
        let lo = T::from(self.min_val).unwrap();
        let hi = T::from(self.max_val).unwrap();

        if self.input.is_cuda() || grad_output.is_cuda() {
            if self.input.device() != grad_output.device() {
                return Err(FerrotorchError::DeviceMismatch {
                    expected: self.input.device(),
                    got: grad_output.device(),
                });
            }
            if !(is_f32::<T>() || is_f64::<T>()) {
                return Err(FerrotorchError::NotImplementedOnCuda {
                    op: "hardtanh backward",
                });
            }
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let result_h = if is_f32::<T>() {
                backend.hardtanh_backward_f32(
                    grad_output.gpu_handle()?,
                    self.input.gpu_handle()?,
                    self.min_val as f32,
                    self.max_val as f32,
                )?
            } else {
                backend.hardtanh_backward_f64(
                    grad_output.gpu_handle()?,
                    self.input.gpu_handle()?,
                    self.min_val,
                    self.max_val,
                )?
            };
            let grad_input = Tensor::from_storage(
                TensorStorage::gpu(result_h),
                self.input.shape().to_vec(),
                false,
            )?;
            return Ok(vec![Some(grad_input)]);
        }

        let input_data = self.input.data()?;
        let grad_data = grad_output.data()?;
        let result: Vec<T> = input_data
            .iter()
            .zip(grad_data.iter())
            .map(|(&x, &g)| if x > lo && x < hi { g } else { zero })
            .collect();
        let grad_input = Tensor::from_storage(
            TensorStorage::cpu(result),
            self.input.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "HardtanhBackward"
    }
}

/// Native `hardtanh(x; min, max) = clamp(x, min, max)`. Default torch range
/// is `[-1, 1]`; pass other bounds via [`hardtanh_with`].
pub fn hardtanh<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    hardtanh_with(input, -1.0, 1.0)
}

/// Hardtanh with explicit bounds.
///
pub fn hardtanh_with<T: Float>(
    input: &Tensor<T>,
    min_val: f64,
    max_val: f64,
) -> FerrotorchResult<Tensor<T>> {
    if input.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
        let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let input = input.contiguous()?;
        let handle = if is_f32::<T>() {
            backend.clamp_f32(input.gpu_handle()?, min_val as f32, max_val as f32)?
        } else {
            backend.clamp_f64(input.gpu_handle()?, min_val, max_val)?
        };
        let storage = TensorStorage::gpu(handle);
        let shape = input.shape().to_vec();
        return if is_grad_enabled() && input.requires_grad() {
            Tensor::from_operation(
                storage,
                shape,
                Arc::new(HardtanhBackward::new(input.clone(), min_val, max_val)),
            )
        } else {
            Tensor::from_storage(storage, shape, false)
        };
    }

    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "hardtanh" });
    }

    let lo = T::from(min_val).unwrap();
    let hi = T::from(max_val).unwrap();
    let output = unary_map(input, |x| {
        if x < lo {
            lo
        } else if x > hi {
            hi
        } else {
            x
        }
    })?;
    if is_grad_enabled() && input.requires_grad() {
        let (storage, shape) = output.into_storage_and_shape()?;
        Tensor::from_operation(
            storage,
            shape,
            Arc::new(HardtanhBackward::new(input.clone(), min_val, max_val)),
        )
    } else {
        Ok(output)
    }
}

/// `relu6(x) = clamp(x, 0, 6)`. Differentiable via the same backward as
/// hardtanh restricted to `[0, 6]`.
pub fn relu6<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    hardtanh_with(input, 0.0, 6.0)
}

// --- hardsigmoid -----------------------------------------------------------

/// Backward for `hardsigmoid(x) = clamp((x + 3) / 6, 0, 1)`.
/// VJP: grad * (1/6) when -3 < x < 3 else 0.
#[derive(Debug)]
pub struct HardsigmoidBackward<T: Float> {
    input: Tensor<T>,
}

impl<T: Float> HardsigmoidBackward<T> {
    pub fn new(input: Tensor<T>) -> Self {
        Self { input }
    }
}

impl<T: Float> GradFn<T> for HardsigmoidBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let zero = <T as num_traits::Zero>::zero();
        let inv_six = T::from(1.0 / 6.0).unwrap();
        let lo = T::from(-3.0).unwrap();
        let hi = T::from(3.0).unwrap();

        if self.input.is_cuda() || grad_output.is_cuda() {
            if self.input.device() != grad_output.device() {
                return Err(FerrotorchError::DeviceMismatch {
                    expected: self.input.device(),
                    got: grad_output.device(),
                });
            }
            if !(is_f32::<T>() || is_f64::<T>()) {
                return Err(FerrotorchError::NotImplementedOnCuda {
                    op: "hardsigmoid backward",
                });
            }
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let result_h = if is_f32::<T>() {
                backend
                    .hardsigmoid_backward_f32(grad_output.gpu_handle()?, self.input.gpu_handle()?)?
            } else {
                backend
                    .hardsigmoid_backward_f64(grad_output.gpu_handle()?, self.input.gpu_handle()?)?
            };
            let grad_input = Tensor::from_storage(
                TensorStorage::gpu(result_h),
                self.input.shape().to_vec(),
                false,
            )?;
            return Ok(vec![Some(grad_input)]);
        }

        let input_data = self.input.data()?;
        let grad_data = grad_output.data()?;
        let result: Vec<T> = input_data
            .iter()
            .zip(grad_data.iter())
            .map(|(&x, &g)| if x > lo && x < hi { g * inv_six } else { zero })
            .collect();
        let grad_input = Tensor::from_storage(
            TensorStorage::cpu(result),
            self.input.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "HardsigmoidBackward"
    }
}

/// Native `hardsigmoid(x) = clamp((x + 3) / 6, 0, 1)` (MobileNetV3).
pub fn hardsigmoid<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if input.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
        let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let input = input.contiguous()?;
        let handle = if is_f32::<T>() {
            backend.hardsigmoid_f32(input.gpu_handle()?)?
        } else {
            backend.hardsigmoid_f64(input.gpu_handle()?)?
        };
        let storage = TensorStorage::gpu(handle);
        let shape = input.shape().to_vec();
        return if is_grad_enabled() && input.requires_grad() {
            Tensor::from_operation(
                storage,
                shape,
                Arc::new(HardsigmoidBackward::new(input.clone())),
            )
        } else {
            Tensor::from_storage(storage, shape, false)
        };
    }

    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "hardsigmoid" });
    }

    let zero = <T as num_traits::Zero>::zero();
    let one = <T as num_traits::One>::one();
    let three = T::from(3.0).unwrap();
    let inv_six = T::from(1.0 / 6.0).unwrap();
    let output = unary_map(input, |x| {
        let v = (x + three) * inv_six;
        if v < zero {
            zero
        } else if v > one {
            one
        } else {
            v
        }
    })?;
    if is_grad_enabled() && input.requires_grad() {
        let (storage, shape) = output.into_storage_and_shape()?;
        Tensor::from_operation(
            storage,
            shape,
            Arc::new(HardsigmoidBackward::new(input.clone())),
        )
    } else {
        Ok(output)
    }
}

// --- hardswish -------------------------------------------------------------

/// Backward for `hardswish(x) = x * hardsigmoid(x)`.
/// VJP: grad * (1 if x ≥ 3, 0 if x ≤ -3, else (2x + 3)/6).
#[derive(Debug)]
pub struct HardswishBackward<T: Float> {
    input: Tensor<T>,
}

impl<T: Float> HardswishBackward<T> {
    pub fn new(input: Tensor<T>) -> Self {
        Self { input }
    }
}

impl<T: Float> GradFn<T> for HardswishBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let zero = <T as num_traits::Zero>::zero();
        let two = T::from(2.0).unwrap();
        let three = T::from(3.0).unwrap();
        let neg_three = T::from(-3.0).unwrap();
        let inv_six = T::from(1.0 / 6.0).unwrap();

        if self.input.is_cuda() || grad_output.is_cuda() {
            if self.input.device() != grad_output.device() {
                return Err(FerrotorchError::DeviceMismatch {
                    expected: self.input.device(),
                    got: grad_output.device(),
                });
            }
            if !(is_f32::<T>() || is_f64::<T>()) {
                return Err(FerrotorchError::NotImplementedOnCuda {
                    op: "hardswish backward",
                });
            }
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let result_h = if is_f32::<T>() {
                backend
                    .hardswish_backward_f32(grad_output.gpu_handle()?, self.input.gpu_handle()?)?
            } else {
                backend
                    .hardswish_backward_f64(grad_output.gpu_handle()?, self.input.gpu_handle()?)?
            };
            let grad_input = Tensor::from_storage(
                TensorStorage::gpu(result_h),
                self.input.shape().to_vec(),
                false,
            )?;
            return Ok(vec![Some(grad_input)]);
        }

        let input_data = self.input.data()?;
        let grad_data = grad_output.data()?;
        let result: Vec<T> = input_data
            .iter()
            .zip(grad_data.iter())
            .map(|(&x, &g)| {
                if x <= neg_three {
                    zero
                } else if x >= three {
                    g
                } else {
                    g * (two * x + three) * inv_six
                }
            })
            .collect();
        let grad_input = Tensor::from_storage(
            TensorStorage::cpu(result),
            self.input.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "HardswishBackward"
    }
}

/// Native `hardswish(x) = x * hardsigmoid(x)`.
pub fn hardswish<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if input.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
        let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let input = input.contiguous()?;
        let handle = if is_f32::<T>() {
            backend.hardswish_f32(input.gpu_handle()?)?
        } else {
            backend.hardswish_f64(input.gpu_handle()?)?
        };
        let storage = TensorStorage::gpu(handle);
        let shape = input.shape().to_vec();
        return if is_grad_enabled() && input.requires_grad() {
            Tensor::from_operation(
                storage,
                shape,
                Arc::new(HardswishBackward::new(input.clone())),
            )
        } else {
            Tensor::from_storage(storage, shape, false)
        };
    }

    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "hardswish" });
    }

    let zero = <T as num_traits::Zero>::zero();
    let three = T::from(3.0).unwrap();
    let neg_three = T::from(-3.0).unwrap();
    let inv_six = T::from(1.0 / 6.0).unwrap();
    let output = unary_map(input, |x| {
        if x <= neg_three {
            zero
        } else if x >= three {
            x
        } else {
            x * (x + three) * inv_six
        }
    })?;
    if is_grad_enabled() && input.requires_grad() {
        let (storage, shape) = output.into_storage_and_shape()?;
        Tensor::from_operation(
            storage,
            shape,
            Arc::new(HardswishBackward::new(input.clone())),
        )
    } else {
        Ok(output)
    }
}

// --- selu ------------------------------------------------------------------

const SELU_ALPHA: f64 = 1.6732632423543772;
const SELU_SCALE: f64 = 1.0507009873554805;

/// Backward for `selu(x) = scale * elu(x, alpha)`.
/// VJP: grad * scale * (1 if x > 0 else alpha * exp(x)).
#[derive(Debug)]
pub struct SeluBackward<T: Float> {
    input: Tensor<T>,
}

impl<T: Float> SeluBackward<T> {
    pub fn new(input: Tensor<T>) -> Self {
        Self { input }
    }
}

impl<T: Float> GradFn<T> for SeluBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let zero = <T as num_traits::Zero>::zero();
        let alpha = T::from(SELU_ALPHA).unwrap();
        let scale = T::from(SELU_SCALE).unwrap();

        if self.input.is_cuda() || grad_output.is_cuda() {
            if self.input.device() != grad_output.device() {
                return Err(FerrotorchError::DeviceMismatch {
                    expected: self.input.device(),
                    got: grad_output.device(),
                });
            }
            if !(is_f32::<T>() || is_f64::<T>()) {
                return Err(FerrotorchError::NotImplementedOnCuda {
                    op: "selu backward",
                });
            }
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let result_h = if is_f32::<T>() {
                let scaled_grad =
                    backend.scale_f32(grad_output.gpu_handle()?, SELU_SCALE as f32)?;
                backend.elu_backward_f32(
                    &scaled_grad,
                    self.input.gpu_handle()?,
                    SELU_ALPHA as f32,
                )?
            } else {
                let scaled_grad = backend.scale_f64(grad_output.gpu_handle()?, SELU_SCALE)?;
                backend.elu_backward_f64(&scaled_grad, self.input.gpu_handle()?, SELU_ALPHA)?
            };
            let grad_input = Tensor::from_storage(
                TensorStorage::gpu(result_h),
                self.input.shape().to_vec(),
                false,
            )?;
            return Ok(vec![Some(grad_input)]);
        }

        let input_data = self.input.data()?;
        let grad_data = grad_output.data()?;
        let result: Vec<T> = input_data
            .iter()
            .zip(grad_data.iter())
            .map(|(&x, &g)| {
                if x > zero {
                    g * scale
                } else {
                    g * scale * alpha * x.exp()
                }
            })
            .collect();
        let grad_input = Tensor::from_storage(
            TensorStorage::cpu(result),
            self.input.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "SeluBackward"
    }
}

/// Native `selu(x) = scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))`
/// with the canonical SELU constants from Klambauer et al. 2017.
pub fn selu<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if input.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
        let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let input = input.contiguous()?;
        let handle = if is_f32::<T>() {
            let elu = backend.elu_f32(input.gpu_handle()?, SELU_ALPHA as f32)?;
            backend.scale_f32(&elu, SELU_SCALE as f32)?
        } else {
            let elu = backend.elu_f64(input.gpu_handle()?, SELU_ALPHA)?;
            backend.scale_f64(&elu, SELU_SCALE)?
        };
        let storage = TensorStorage::gpu(handle);
        let shape = input.shape().to_vec();
        return if is_grad_enabled() && input.requires_grad() {
            Tensor::from_operation(storage, shape, Arc::new(SeluBackward::new(input.clone())))
        } else {
            Tensor::from_storage(storage, shape, false)
        };
    }

    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "selu" });
    }

    let zero = <T as num_traits::Zero>::zero();
    let one = <T as num_traits::One>::one();
    let alpha = T::from(SELU_ALPHA).unwrap();
    let scale = T::from(SELU_SCALE).unwrap();
    let output = unary_map(input, |x| {
        if x > zero {
            scale * x
        } else {
            scale * alpha * (x.exp() - one)
        }
    })?;
    if is_grad_enabled() && input.requires_grad() {
        let (storage, shape) = output.into_storage_and_shape()?;
        Tensor::from_operation(storage, shape, Arc::new(SeluBackward::new(input.clone())))
    } else {
        Ok(output)
    }
}

// --- softsign --------------------------------------------------------------

/// Backward for `softsign(x) = x / (1 + |x|)`.
/// VJP: grad / (1 + |x|)^2.
#[derive(Debug)]
pub struct SoftsignBackward<T: Float> {
    input: Tensor<T>,
}

impl<T: Float> SoftsignBackward<T> {
    pub fn new(input: Tensor<T>) -> Self {
        Self { input }
    }
}

impl<T: Float> GradFn<T> for SoftsignBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let one = <T as num_traits::One>::one();

        if self.input.is_cuda() || grad_output.is_cuda() {
            if self.input.device() != grad_output.device() {
                return Err(FerrotorchError::DeviceMismatch {
                    expected: self.input.device(),
                    got: grad_output.device(),
                });
            }
            if !(is_f32::<T>() || is_f64::<T>()) {
                return Err(FerrotorchError::NotImplementedOnCuda {
                    op: "softsign backward",
                });
            }
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let result_h = if is_f32::<T>() {
                backend
                    .softsign_backward_f32(grad_output.gpu_handle()?, self.input.gpu_handle()?)?
            } else {
                backend
                    .softsign_backward_f64(grad_output.gpu_handle()?, self.input.gpu_handle()?)?
            };
            let grad_input = Tensor::from_storage(
                TensorStorage::gpu(result_h),
                self.input.shape().to_vec(),
                false,
            )?;
            return Ok(vec![Some(grad_input)]);
        }

        let input_data = self.input.data()?;
        let grad_data = grad_output.data()?;
        let result: Vec<T> = input_data
            .iter()
            .zip(grad_data.iter())
            .map(|(&x, &g)| {
                let denom = one + x.abs();
                g / (denom * denom)
            })
            .collect();
        let grad_input = Tensor::from_storage(
            TensorStorage::cpu(result),
            self.input.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "SoftsignBackward"
    }
}

/// Native `softsign(x) = x / (1 + |x|)`.
pub fn softsign<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if input.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
        let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let input = input.contiguous()?;
        let handle = if is_f32::<T>() {
            backend.softsign_f32(input.gpu_handle()?)?
        } else {
            backend.softsign_f64(input.gpu_handle()?)?
        };
        let storage = TensorStorage::gpu(handle);
        let shape = input.shape().to_vec();
        return if is_grad_enabled() && input.requires_grad() {
            Tensor::from_operation(
                storage,
                shape,
                Arc::new(SoftsignBackward::new(input.clone())),
            )
        } else {
            Tensor::from_storage(storage, shape, false)
        };
    }

    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "softsign" });
    }

    let one = <T as num_traits::One>::one();
    let output = unary_map(input, |x| x / (one + x.abs()))?;
    if is_grad_enabled() && input.requires_grad() {
        let (storage, shape) = output.into_storage_and_shape()?;
        Tensor::from_operation(
            storage,
            shape,
            Arc::new(SoftsignBackward::new(input.clone())),
        )
    } else {
        Ok(output)
    }
}

// --- prelu -----------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PReluWeightMode {
    Scalar,
    Channel { channels: usize, inner: usize },
}

impl PReluWeightMode {
    fn alpha_index(self, flat_index: usize) -> usize {
        match self {
            Self::Scalar => 0,
            Self::Channel { channels, inner } => (flat_index / inner) % channels,
        }
    }
}

fn prelu_weight_mode<T: Float>(
    input: &Tensor<T>,
    alpha: &Tensor<T>,
) -> FerrotorchResult<PReluWeightMode> {
    if alpha.ndim() > 1 {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "prelu: Expected `weight` to be a scalar or 1D tensor, but got ndim = {}",
                alpha.ndim()
            ),
        });
    }

    let alpha_numel = alpha.numel();
    if alpha_numel == 1 {
        return Ok(PReluWeightMode::Scalar);
    }

    if input.ndim() == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "prelu: Not allow zero-dim input tensor.".into(),
        });
    }

    let channel_size = if input.ndim() > 1 {
        input.shape()[1]
    } else {
        1
    };
    if channel_size != alpha_numel {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "prelu: Mismatch of parameter numbers and input channel size. \
                 Found parameter numbers = {alpha_numel} and channel size = {channel_size}."
            ),
        });
    }

    let inner = input.shape()[2..].iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: "prelu: input inner dimension product overflows usize".into(),
            })
    })?;

    Ok(PReluWeightMode::Channel {
        channels: channel_size,
        inner,
    })
}

/// Backward for `prelu(x, alpha) = max(0, x) + alpha * min(0, x)`.
///
/// Gradients (strict comparison per torch's `_prelu_kernel_backward`,
/// `dx = x > 0 ? grad : weight * grad` — x == 0 takes the WEIGHT branch,
/// #1951):
///   `dL/dx[i]    = grad[i] * (x[i] > 0 ? 1 : alpha)`
///   `dL/dalpha   = sum_i grad[i] * (!(x[i] > 0) ? x[i] : 0)`
///   (x == 0 contributes zero; NaN follows PyTorch's false `x > 0` branch)
///
/// Single-pass fused VJP — replaces the previous
/// `(1 - alpha) * relu(x) + alpha * x` decomposition that walked through three
/// separate GradFn nodes per call.
#[derive(Debug)]
pub struct PReluBackward<T: Float> {
    input: Tensor<T>,
    alpha: Tensor<T>,
}

impl<T: Float> PReluBackward<T> {
    pub fn new(input: Tensor<T>, alpha: Tensor<T>) -> Self {
        Self { input, alpha }
    }
}

impl<T: Float> GradFn<T> for PReluBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let zero = <T as num_traits::Zero>::zero();
        let one = <T as num_traits::One>::one();
        let mode = prelu_weight_mode(&self.input, &self.alpha)?;

        if self.input.is_cuda() || grad_output.is_cuda() || self.alpha.is_cuda() {
            if self.input.device() != grad_output.device() {
                return Err(FerrotorchError::DeviceMismatch {
                    expected: self.input.device(),
                    got: grad_output.device(),
                });
            }
            if self.input.device() != self.alpha.device() {
                return Err(FerrotorchError::DeviceMismatch {
                    expected: self.input.device(),
                    got: self.alpha.device(),
                });
            }
            if !(is_f32::<T>() || is_f64::<T>()) {
                return Err(FerrotorchError::NotImplementedOnCuda {
                    op: "prelu backward",
                });
            }
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let grad_output = grad_output.contiguous()?;
            let input = self.input.contiguous()?;
            let alpha = self.alpha.contiguous()?;
            let (grad_input_h, grad_alpha_h) = match mode {
                PReluWeightMode::Scalar => {
                    if is_f32::<T>() {
                        (
                            backend.prelu_scalar_backward_input_f32(
                                grad_output.gpu_handle()?,
                                input.gpu_handle()?,
                                alpha.gpu_handle()?,
                            )?,
                            backend.prelu_scalar_backward_alpha_f32(
                                grad_output.gpu_handle()?,
                                input.gpu_handle()?,
                            )?,
                        )
                    } else {
                        (
                            backend.prelu_scalar_backward_input_f64(
                                grad_output.gpu_handle()?,
                                input.gpu_handle()?,
                                alpha.gpu_handle()?,
                            )?,
                            backend.prelu_scalar_backward_alpha_f64(
                                grad_output.gpu_handle()?,
                                input.gpu_handle()?,
                            )?,
                        )
                    }
                }
                PReluWeightMode::Channel { channels, inner } => {
                    if is_f32::<T>() {
                        (
                            backend.prelu_channel_backward_input_f32(
                                grad_output.gpu_handle()?,
                                input.gpu_handle()?,
                                alpha.gpu_handle()?,
                                channels,
                                inner,
                            )?,
                            backend.prelu_channel_backward_alpha_f32(
                                grad_output.gpu_handle()?,
                                input.gpu_handle()?,
                                channels,
                                inner,
                            )?,
                        )
                    } else {
                        (
                            backend.prelu_channel_backward_input_f64(
                                grad_output.gpu_handle()?,
                                input.gpu_handle()?,
                                alpha.gpu_handle()?,
                                channels,
                                inner,
                            )?,
                            backend.prelu_channel_backward_alpha_f64(
                                grad_output.gpu_handle()?,
                                input.gpu_handle()?,
                                channels,
                                inner,
                            )?,
                        )
                    }
                }
            };
            let grad_input_t = Tensor::from_storage(
                TensorStorage::gpu(grad_input_h),
                self.input.shape().to_vec(),
                false,
            )?;
            let grad_alpha_t = Tensor::from_storage(
                TensorStorage::gpu(grad_alpha_h),
                self.alpha.shape().to_vec(),
                false,
            )?;
            return Ok(vec![Some(grad_input_t), Some(grad_alpha_t)]);
        }

        let x = self.input.data_vec()?;
        let alpha_data = self.alpha.data_vec()?;
        let g = grad_output.data_vec()?;

        // grad wrt input — strict `x > 0` per torch's `_prelu_kernel_backward`
        // (`dx = x > 0 ? grad : weight * grad`): x == 0 takes the weight
        // branch (#1951).
        let grad_input: Vec<T> = x
            .iter()
            .zip(g.iter())
            .enumerate()
            .map(|(idx, (&xv, &gv))| {
                let alpha_v = alpha_data[mode.alpha_index(idx)];
                if xv > zero { gv * one } else { gv * alpha_v }
            })
            .collect();
        let grad_input_t = Tensor::from_storage(
            TensorStorage::cpu(grad_input),
            self.input.shape().to_vec(),
            false,
        )?;

        let mut grad_alpha = vec![zero; alpha_data.len()];
        for (idx, (&xv, &gv)) in x.iter().zip(g.iter()).enumerate() {
            if xv.partial_cmp(&zero) != Some(std::cmp::Ordering::Greater) {
                grad_alpha[mode.alpha_index(idx)] += gv * xv;
            }
        }
        let grad_alpha_t = Tensor::from_storage(
            TensorStorage::cpu(grad_alpha),
            self.alpha.shape().to_vec(),
            false,
        )?;

        Ok(vec![Some(grad_input_t), Some(grad_alpha_t)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input, &self.alpha]
    }

    fn name(&self) -> &'static str {
        "PReluBackward"
    }
}

/// Native `prelu(x, alpha) = max(0, x) + alpha * min(0, x)`.
///
/// `alpha` is a scalar/length-1 slope tensor or a 1D channel-weight tensor
/// whose length matches dim 1 when `input.ndim() > 1`, matching PyTorch PReLU.
pub fn prelu<T: Float>(input: &Tensor<T>, alpha: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if input.device() != alpha.device() {
        return Err(FerrotorchError::DeviceMismatch {
            expected: input.device(),
            got: alpha.device(),
        });
    }

    let mode = prelu_weight_mode(input, alpha)?;

    if input.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
        let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let input = input.contiguous()?;
        let alpha = alpha.contiguous()?;
        let handle = match mode {
            PReluWeightMode::Scalar => {
                if is_f32::<T>() {
                    backend.prelu_scalar_f32(input.gpu_handle()?, alpha.gpu_handle()?)?
                } else {
                    backend.prelu_scalar_f64(input.gpu_handle()?, alpha.gpu_handle()?)?
                }
            }
            PReluWeightMode::Channel { channels, inner } => {
                if is_f32::<T>() {
                    backend.prelu_channel_f32(
                        input.gpu_handle()?,
                        alpha.gpu_handle()?,
                        channels,
                        inner,
                    )?
                } else {
                    backend.prelu_channel_f64(
                        input.gpu_handle()?,
                        alpha.gpu_handle()?,
                        channels,
                        inner,
                    )?
                }
            }
        };
        let storage = TensorStorage::gpu(handle);
        let shape = input.shape().to_vec();
        return if is_grad_enabled() && (input.requires_grad() || alpha.requires_grad()) {
            Tensor::from_operation(
                storage,
                shape,
                Arc::new(PReluBackward::new(input.clone(), alpha.clone())),
            )
        } else {
            Tensor::from_storage(storage, shape, false)
        };
    }

    if input.is_cuda() || alpha.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "prelu" });
    }

    let alpha_data = alpha.data_vec()?;
    let zero = <T as num_traits::Zero>::zero();

    let output = match mode {
        PReluWeightMode::Scalar => {
            let alpha_v = alpha_data[0];
            unary_map(input, |x| if x > zero { x } else { alpha_v * x })?
        }
        PReluWeightMode::Channel { .. } => {
            let input_data = input.data_vec()?;
            let out = input_data
                .iter()
                .enumerate()
                .map(|(idx, &x)| {
                    if x > zero {
                        x
                    } else {
                        alpha_data[mode.alpha_index(idx)] * x
                    }
                })
                .collect();
            Tensor::from_storage(TensorStorage::cpu(out), input.shape().to_vec(), false)?
        }
    };
    if is_grad_enabled() && (input.requires_grad() || alpha.requires_grad()) {
        let (storage, shape) = output.into_storage_and_shape()?;
        Tensor::from_operation(
            storage,
            shape,
            Arc::new(PReluBackward::new(input.clone(), alpha.clone())),
        )
    } else {
        Ok(output)
    }
}

// --- glu -------------------------------------------------------------------

/// Backward for `glu(x; dim) = a * sigmoid(b)` where `(a, b) = split(x, dim/2)`.
///
/// Let `s = sigmoid(b)` and `y = a * s`. Then:
///   `dL/da = grad * s`
///   `dL/db = grad * a * s * (1 - s)`
/// and the full input gradient is the concatenation of `[dL/da, dL/db]` along
/// the splitting dim.
///
/// We cache the split (`a`, `b`) plus `s` so we don't recompute the sigmoid in
/// the backward pass.
#[derive(Debug)]
pub struct GluBackward<T: Float> {
    input: Tensor<T>,
    a: Vec<T>,
    sigmoid_b: Vec<T>,
    dim: usize,
}

impl<T: Float> GluBackward<T> {
    pub fn new(input: Tensor<T>, a: Vec<T>, sigmoid_b: Vec<T>, dim: usize) -> Self {
        Self {
            input,
            a,
            sigmoid_b,
            dim,
        }
    }
}

impl<T: Float> GradFn<T> for GluBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let g = grad_output.data()?;
        let one = <T as num_traits::One>::one();
        let n_half = self.a.len();

        // Per-element grads for the two halves.
        let grad_a: Vec<T> = g
            .iter()
            .zip(self.sigmoid_b.iter())
            .map(|(&gv, &sv)| gv * sv)
            .collect();
        let grad_b: Vec<T> = g
            .iter()
            .zip(self.a.iter())
            .zip(self.sigmoid_b.iter())
            .map(|((&gv, &av), &sv)| gv * av * sv * (one - sv))
            .collect();

        debug_assert_eq!(grad_a.len(), n_half);
        debug_assert_eq!(grad_b.len(), n_half);

        // Concatenate [grad_a, grad_b] along self.dim back into input shape.
        let in_shape = self.input.shape();
        let mut full = vec![<T as num_traits::Zero>::zero(); self.input.numel()];

        // Compute strides (row-major).
        let mut strides = vec![1usize; in_shape.len()];
        for i in (0..in_shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * in_shape[i + 1];
        }

        let len_dim = in_shape[self.dim];
        let half = len_dim / 2;
        let stride_dim = strides[self.dim];

        // Outer dims contribution (product of dims < self.dim).
        let outer: usize = crate::shape::numel(&in_shape[..self.dim]);
        // Inner dims contribution (product of dims > self.dim).
        let inner: usize = crate::shape::numel(&in_shape[(self.dim + 1)..]);

        // Walk every (outer, k_in_dim, inner) cell.
        for o in 0..outer {
            for k in 0..len_dim {
                for i in 0..inner {
                    let flat_full = o * (len_dim * inner) + k * inner + i;
                    if k < half {
                        // First half: grad_a
                        let flat_half = o * (half * inner) + k * inner + i;
                        full[flat_full] = grad_a[flat_half];
                    } else {
                        // Second half: grad_b
                        let flat_half = o * (half * inner) + (k - half) * inner + i;
                        full[flat_full] = grad_b[flat_half];
                    }
                }
            }
        }
        // stride_dim isn't used directly (we computed via outer/inner) but we
        // keep it bound to surface a future bug if shape walking changes.
        let _ = stride_dim;

        let grad_input =
            Tensor::from_storage(TensorStorage::cpu(full), self.input.shape().to_vec(), false)?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "GluBackward"
    }
}

/// Native `glu(x; dim) = a * sigmoid(b)` where `x` is split into two equal
/// halves `(a, b)` along `dim`.
///
/// Single-pass fused VJP — replaces the previous functional implementation
/// that chained `split` + `sigmoid` + `mul` into three separate backward nodes.
pub fn glu<T: Float>(input: &Tensor<T>, dim: i64) -> FerrotorchResult<Tensor<T>> {
    let shape = input.shape();
    let ndim = shape.len();
    if ndim == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "glu: input must have at least 1 dimension".into(),
        });
    }
    let resolved = if dim < 0 {
        (ndim as i64 + dim) as usize
    } else {
        dim as usize
    };
    if resolved >= ndim {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("glu: dim {dim} out of range for {ndim}-D tensor"),
        });
    }
    let len_dim = shape[resolved];
    if !len_dim.is_multiple_of(2) {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("glu: split dim must be even, got {len_dim} along dim {resolved}"),
        });
    }
    let half = len_dim / 2;

    // Output shape = input shape with dim halved.
    let mut out_shape = shape.to_vec();
    out_shape[resolved] = half;
    let n_out: usize = crate::shape::numel(&out_shape);

    let in_data = input.data()?;
    let outer: usize = crate::shape::numel(&shape[..resolved]);
    let inner: usize = crate::shape::numel(&shape[(resolved + 1)..]);

    let mut a_vals = Vec::with_capacity(n_out);
    let mut b_vals = Vec::with_capacity(n_out);
    for o in 0..outer {
        for k in 0..half {
            for i in 0..inner {
                let flat_a = o * (len_dim * inner) + k * inner + i;
                let flat_b = o * (len_dim * inner) + (k + half) * inner + i;
                a_vals.push(in_data[flat_a]);
                b_vals.push(in_data[flat_b]);
            }
        }
    }

    // sigmoid(b) — element-wise; we only need the scalar form here so don't
    // round-trip through the full Tensor sigmoid op.
    let one = <T as num_traits::One>::one();
    let sigmoid_scalar = |v: T| -> T {
        // Numerically-stable scalar sigmoid: 1/(1+exp(-v)) for v >= 0,
        // exp(v)/(1+exp(v)) for v < 0.
        let zero = <T as num_traits::Zero>::zero();
        if v >= zero {
            one / (one + (-v).exp())
        } else {
            let ev = v.exp();
            ev / (one + ev)
        }
    };
    let sigmoid_b: Vec<T> = b_vals.iter().map(|&v| sigmoid_scalar(v)).collect();
    let out: Vec<T> = a_vals
        .iter()
        .zip(sigmoid_b.iter())
        .map(|(&av, &sv)| av * sv)
        .collect();

    if is_grad_enabled() && input.requires_grad() {
        Tensor::from_operation(
            TensorStorage::cpu(out),
            out_shape,
            Arc::new(GluBackward::new(input.clone(), a_vals, sigmoid_b, resolved)),
        )
    } else {
        Tensor::from_storage(TensorStorage::cpu(out), out_shape, false)
    }
}

// ===========================================================================
// Shared helper: T::from(f64) returning a FerrotorchError on representability
// failure (used by the #1341 additions below to honour R-CODE-2). The
// pre-existing forward functions in this file predate the anti-pattern gate
// and use the older `T::from(_).unwrap()` form; we don't retroactively
// rewrite those — only new surface flows through here.
// ===========================================================================

#[inline]
fn t_from<T: Float>(value: f64, op: &'static str) -> FerrotorchResult<T> {
    T::from(value).ok_or_else(|| FerrotorchError::InvalidArgument {
        message: format!("{op}: value={value} cannot be represented in the tensor dtype"),
    })
}

// ===========================================================================
// Threshold (#1341 REQ-19)
// ===========================================================================
//
// `threshold(x; threshold, value) = x` if `x > threshold` else `value`.
// Upstream:
//   - `TORCH_IMPL_FUNC(threshold_out)` at
//     `aten/src/ATen/native/Activation.cpp:688-690`: dispatches
//     `threshold_stub(device_type(), *this, threshold, value)`.
//   - `TORCH_IMPL_FUNC(threshold_backward_out)` at
//     `aten/src/ATen/native/Activation.cpp:692-694`: dispatches
//     `threshold_stub(device_type(), *this, threshold, 0)` (with `value=0`,
//     reusing the same kernel — the backward IS a thresholded copy of
//     `grad_output` where the cells with `self > threshold` pass through
//     and the rest are zeroed).
//   - User-facing surface: `torch.nn.functional.threshold(input, threshold,
//     value)` at `torch/nn/functional.py:1682-1700`.
//   - Backward formula per `tools/autograd/derivatives.yaml:2243-2244`:
//     `self: threshold_backward(grad, self, threshold)` — i.e.
//     `grad if x > threshold else 0`.

/// Backward for `threshold(x; threshold, value)`.
///
/// VJP: `grad if x > threshold else 0` per
/// `aten/src/ATen/native/Activation.cpp:692-694
/// TORCH_IMPL_FUNC(threshold_backward_out)` and
/// `tools/autograd/derivatives.yaml:2243-2244
/// self: threshold_backward(grad, self, threshold)`.
#[derive(Debug)]
pub struct ThresholdBackward<T: Float> {
    input: Tensor<T>,
    threshold: f64,
}

impl<T: Float> ThresholdBackward<T> {
    pub fn new(input: Tensor<T>, threshold: f64) -> Self {
        Self { input, threshold }
    }
}

impl<T: Float> GradFn<T> for ThresholdBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if grad_output.is_cuda() || self.input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "threshold backward",
            });
        }
        let input_data = self.input.data()?;
        let grad_data = grad_output.data()?;
        let zero = <T as num_traits::Zero>::zero();
        let thr = t_from::<T>(self.threshold, "threshold backward")?;
        let result: Vec<T> = input_data
            .iter()
            .zip(grad_data.iter())
            .map(|(&x, &g)| if x > thr { g } else { zero })
            .collect();
        let grad_input = Tensor::from_storage(
            TensorStorage::cpu(result),
            self.input.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "ThresholdBackward"
    }
}

/// Compute `threshold(x; threshold, value) = x if x > threshold else value`,
/// attaching a backward node when gradients are enabled.
///
/// Mirrors `TORCH_IMPL_FUNC(threshold_out)` at
/// `aten/src/ATen/native/Activation.cpp:688-690` and
/// `torch.nn.functional.threshold` at `torch/nn/functional.py:1682-1700`.
pub fn threshold<T: Float>(
    input: &Tensor<T>,
    threshold_val: f64,
    value: f64,
) -> FerrotorchResult<Tensor<T>> {
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "threshold" });
    }
    let thr = t_from::<T>(threshold_val, "threshold")?;
    let val = t_from::<T>(value, "threshold")?;
    let output = unary_map(input, |x| if x > thr { x } else { val })?;
    if is_grad_enabled() && input.requires_grad() {
        let (storage, shape) = output.into_storage_and_shape()?;
        Tensor::from_operation(
            storage,
            shape,
            Arc::new(ThresholdBackward::new(input.clone(), threshold_val)),
        )
    } else {
        Ok(output)
    }
}

// ===========================================================================
// RReLU (#1341 REQ-20)
// ===========================================================================
//
// `rrelu(x; lower, upper, training)` — randomized leaky ReLU.
// Upstream:
//   - `Tensor& rrelu_with_noise_out_cpu(...)` at
//     `aten/src/ATen/native/Activation.cpp:611-654`. In TRAINING mode each
//     negative element gets a slope drawn uniformly from `[lower, upper]`
//     (the noise tensor saves the per-element slope). In INFERENCE mode the
//     op delegates to `leaky_relu` with the deterministic mean slope
//     `(lower + upper) / 2` — see the `else` branch at lines 624-630:
//     `auto negative = (lower_tensor + upper_tensor) / 2; ...
//      return at::leaky_relu_out(output, self, negative_slope);`.
//   - User-facing: `torch.nn.functional.rrelu(input, lower=1/8, upper=1/3,
//     training=False, inplace=False)` at `torch/nn/functional.py:1962-1989`.
//
// ferrotorch ships BOTH paths (#1738 / CORE-044):
//   - inference (`training=false`): fused `RReluBackward` with the
//     deterministic mean slope, mirroring the upstream `else` branch;
//   - training (`training=true`): per-element slope drawn from
//     `Uniform[lower, upper]` via the process-default MT19937 generator
//     (`crate::rng::with_thread_rng`), saved as the `noise` buffer for the
//     backward — mirroring `_rrelu_with_noise_train` exactly. The draw is
//     `at::uniform_real_distribution<double>(lower, upper)` per
//     `aten/src/ATen/core/DistributionsHelper.h:60-70`, i.e.
//     `next_uniform_f64() * (upper - lower) + lower` (one u64 = two MT19937
//     u32 calls) for every element with `x <= 0` (zero included), in
//     sequential element order, double precision regardless of dtype. Since
//     `rng::Generator` is byte-identical to torch's CPU MT19937, the
//     training forward is bit-exact vs torch CPU under `manual_seed`
//     (pinned by `tests/audit_core044_rrelu_training.rs`).

/// Backward for `rrelu(x; lower, upper, training=False)` — inference mode.
///
/// VJP: `grad * 1` if `x > 0` else `grad * negative_slope` where
/// `negative_slope = (lower + upper) / 2`. Mirrors the inference-mode
/// delegation `return at::leaky_relu_out(output, self, negative_slope)` at
/// `aten/src/ATen/native/Activation.cpp:629`.
#[derive(Debug)]
pub struct RReluBackward<T: Float> {
    input: Tensor<T>,
    negative_slope: f64,
}

impl<T: Float> RReluBackward<T> {
    pub fn new(input: Tensor<T>, negative_slope: f64) -> Self {
        Self {
            input,
            negative_slope,
        }
    }
}

impl<T: Float> GradFn<T> for RReluBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if grad_output.is_cuda() || self.input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "rrelu backward",
            });
        }
        let input_data = self.input.data()?;
        let grad_data = grad_output.data()?;
        let zero = <T as num_traits::Zero>::zero();
        let slope = t_from::<T>(self.negative_slope, "rrelu backward")?;
        let result: Vec<T> = input_data
            .iter()
            .zip(grad_data.iter())
            .map(|(&x, &g)| if x > zero { g } else { g * slope })
            .collect();
        let grad_input = Tensor::from_storage(
            TensorStorage::cpu(result),
            self.input.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "RReluBackward"
    }
}

/// Backward for `rrelu(x; lower, upper, training=true)` — training mode.
///
/// VJP: `grad * noise` where `noise[i]` is the per-element slope drawn in
/// the forward pass (`noise[i] = 1` for `x[i] > 0`, `noise[i] ~
/// Uniform[lower, upper]` for `x[i] <= 0`). Mirrors
/// `rrelu_with_noise_backward` at `aten/src/ATen/native/Activation.cpp`
/// (training branch: `grad_output * noise`) wired via
/// `tools/autograd/derivatives.yaml` — `rrelu_with_noise` saves `noise` and
/// the backward multiplies by it.
///
/// Private: not part of the conformance surface; constructed only by
/// [`rrelu`]. (#1738 / CORE-044)
#[derive(Debug)]
struct RReluTrainBackward<T: Float> {
    input: Tensor<T>,
    /// Per-element slopes drawn in forward (`1` on the positive branch).
    noise: Vec<T>,
}

impl<T: Float> RReluTrainBackward<T> {
    fn new(input: Tensor<T>, noise: Vec<T>) -> Self {
        Self { input, noise }
    }
}

impl<T: Float> GradFn<T> for RReluTrainBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // The training forward rejects CUDA inputs, so the saved state is
        // CPU-resident; a CUDA grad_output would mean a mixed-device graph.
        if grad_output.is_cuda() || self.input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "rrelu training backward",
            });
        }
        let grad_data = grad_output.data()?;
        let result: Vec<T> = self
            .noise
            .iter()
            .zip(grad_data.iter())
            .map(|(&r, &g)| g * r)
            .collect();
        let grad_input = Tensor::from_storage(
            TensorStorage::cpu(result),
            self.input.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "RReluTrainBackward"
    }
}

/// Compute `rrelu(x; lower, upper, training)`.
///
/// - `training=false` (inference) mirrors the upstream delegation
///   `return at::leaky_relu_out(output, self, (lower + upper) / 2)` at
///   `aten/src/ATen/native/Activation.cpp:624-630` — deterministic mean
///   slope, no RNG consumption.
/// - `training=true` draws an independent per-element slope from
///   `Uniform[lower, upper]` for every element with `x <= 0` (zero
///   included), mirroring `_rrelu_with_noise_train` at
///   `aten/src/ATen/native/Activation.cpp:578-608`: the slope is drawn in
///   DOUBLE precision via `at::uniform_real_distribution<double>` from the
///   process-default MT19937 generator (`crate::manual_seed` controls it),
///   saved as the `noise` buffer, and the backward applies `grad * noise`.
///   Bit-exact vs torch CPU under `manual_seed` (#1738 / CORE-044).
pub fn rrelu<T: Float>(
    input: &Tensor<T>,
    lower: f64,
    upper: f64,
    training: bool,
) -> FerrotorchResult<Tensor<T>> {
    if lower > upper {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("rrelu: lower ({lower}) must be <= upper ({upper})"),
        });
    }
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "rrelu" });
    }
    if training {
        return rrelu_train(input, lower, upper);
    }
    let negative_slope = f64::midpoint(lower, upper);
    let slope_t = t_from::<T>(negative_slope, "rrelu")?;
    let zero = <T as num_traits::Zero>::zero();
    let output = unary_map(input, |x| if x > zero { x } else { slope_t * x })?;
    if is_grad_enabled() && input.requires_grad() {
        let (storage, shape) = output.into_storage_and_shape()?;
        Tensor::from_operation(
            storage,
            shape,
            Arc::new(RReluBackward::new(input.clone(), negative_slope)),
        )
    } else {
        Ok(output)
    }
}

/// Training-mode rrelu forward — see [`rrelu`]. CPU-only (the caller has
/// already rejected CUDA inputs). Mirrors `_rrelu_with_noise_train` at
/// `aten/src/ATen/native/Activation.cpp:578-608` element for element:
///
/// ```cpp
/// using opmath_t = at::opmath_type<scalar_t>;
/// if (input_data[i] <= 0) {
///   at::uniform_real_distribution<double> uniform(lower, upper);
///   const opmath_t r = (opmath_t)uniform(gen);
///   output_data[i] = input_data[i] * r;   // multiply in opmath_t
///   noise_data[i] = r;                    // store rounds to scalar_t
/// } else {
///   noise_data[i] = 1;
///   output_data[i] = input_data[i];
/// }
/// ```
///
/// `opmath_type<bf16/f16> = float`: the f64 draw is cast to f32 (NOT to the
/// storage dtype), the product is computed in f32, and only the store rounds
/// to bf16/f16 — rounding `r` to the storage dtype first double-rounds and
/// drifts by 1 ULP vs torch (#1953). For f32/f64, `opmath_t == scalar_t`,
/// so the plain path is already exact.
fn rrelu_train<T: Float>(input: &Tensor<T>, lower: f64, upper: f64) -> FerrotorchResult<Tensor<T>> {
    let data = input.data()?;
    let zero = <T as num_traits::Zero>::zero();
    let one = <T as num_traits::One>::one();
    let n = data.len();
    let mut out = Vec::with_capacity(n);
    let mut noise = Vec::with_capacity(n);
    let reduced_fp = is_bf16::<T>() || is_f16::<T>();
    crate::rng::with_thread_rng(|g| -> FerrotorchResult<()> {
        for &x in data {
            if x <= zero {
                // `at::uniform_real_distribution<double>(lower, upper)(gen)`
                // = next_uniform_f64() * (upper - lower) + lower, per
                // `aten/src/ATen/core/DistributionsHelper.h:60-70`. Drawn in
                // f64 regardless of T, then cast to opmath_t.
                let r64 = g.next_uniform_f64() * (upper - lower) + lower;
                if reduced_fp {
                    // opmath_t = f32 for bf16/f16 (#1953): r → f32, promote
                    // x exactly to f32, multiply in f32, round ONCE on store.
                    let r_op = r64 as f32;
                    let x_op = x.to_f32().ok_or_else(|| FerrotorchError::InvalidArgument {
                        message: "rrelu training: input not representable as f32".into(),
                    })?;
                    out.push(t_from::<T>(f64::from(x_op * r_op), "rrelu training")?);
                    noise.push(t_from::<T>(f64::from(r_op), "rrelu training")?);
                } else {
                    // f32/f64: opmath_t == scalar_t — single rounding either way.
                    let r = t_from::<T>(r64, "rrelu training")?;
                    out.push(x * r);
                    noise.push(r);
                }
            } else {
                out.push(x);
                noise.push(one);
            }
        }
        Ok(())
    })?;
    let shape = input.shape().to_vec();
    if is_grad_enabled() && input.requires_grad() {
        Tensor::from_operation(
            TensorStorage::cpu(out),
            shape,
            Arc::new(RReluTrainBackward::new(input.clone(), noise)),
        )
    } else {
        Tensor::from_storage(TensorStorage::cpu(out), shape, false)
    }
}

// ===========================================================================
// CELU (#1341 REQ-21)
// ===========================================================================
//
// `celu(x; alpha) = max(0, x) + min(0, alpha * (exp(x / alpha) - 1))`.
// Upstream `Tensor celu(const Tensor& self, const Scalar& alpha)` at
// `aten/src/ATen/native/Activation.cpp:540-545`:
//
//   TORCH_CHECK(alpha.to<double>() != 0, "ZeroDivisionError: alpha cannot be 0 for CELU");
//   double inv_alpha = 1. / alpha.to<double>();
//   return at::elu(self, alpha, Scalar(1.0), Scalar(inv_alpha));
//
// Forward expansion: with `(alpha, scale=1.0, input_scale=1/alpha)`, ELU's
//   `f(x) = x` for x > 0
//   `f(x) = scale * alpha * (exp(input_scale * x) - 1) = alpha * (exp(x/alpha) - 1)`
// for x <= 0.
//
// User-facing: `torch.nn.functional.celu(input, alpha=1.0, inplace=False)` at
// `torch/nn/functional.py:1874-1894`. Default `alpha=1.0` collapses CELU to
// the standard ELU (since `exp(x/1) - 1 == exp(x) - 1` and `alpha=1`).
//
// Backward per the chain rule applied to the ELU-with-input-scale formulation:
//   `dceu/dx = 1` for x > 0
//   `dceu/dx = exp(x / alpha)` for x <= 0
// (Since `d/dx[alpha * (exp(x/alpha) - 1)] = alpha * (1/alpha) * exp(x/alpha)
//  = exp(x/alpha)`.)

/// Backward for `celu(x; alpha) = max(0, x) + min(0, alpha * (exp(x/alpha) - 1))`.
///
/// VJP: `grad * 1` if `x > 0` else `grad * exp(x / alpha)`. Mirrors upstream
/// delegation `at::elu(self, alpha, 1.0, 1/alpha)` at
/// `aten/src/ATen/native/Activation.cpp:543-544` — the closed-form CELU
/// derivative simplifies to `exp(x / alpha)` on the negative branch (the
/// `alpha` factor and `1/alpha` chain-rule factor cancel).
#[derive(Debug)]
pub struct CeluBackward<T: Float> {
    input: Tensor<T>,
    alpha: f64,
}

impl<T: Float> CeluBackward<T> {
    pub fn new(input: Tensor<T>, alpha: f64) -> Self {
        Self { input, alpha }
    }
}

impl<T: Float> GradFn<T> for CeluBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if grad_output.is_cuda() || self.input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "celu backward",
            });
        }
        let input_data = self.input.data()?;
        let grad_data = grad_output.data()?;
        let zero = <T as num_traits::Zero>::zero();
        let inv_alpha = t_from::<T>(1.0 / self.alpha, "celu backward")?;
        let result: Vec<T> = input_data
            .iter()
            .zip(grad_data.iter())
            .map(|(&x, &g)| {
                if x > zero {
                    g
                } else {
                    g * (x * inv_alpha).exp()
                }
            })
            .collect();
        let grad_input = Tensor::from_storage(
            TensorStorage::cpu(result),
            self.input.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "CeluBackward"
    }
}

/// Compute `celu(x; alpha) = max(0, x) + min(0, alpha * (exp(x / alpha) - 1))`,
/// attaching a backward node when gradients are enabled.
///
/// Mirrors `Tensor celu(const Tensor& self, const Scalar& alpha)` at
/// `aten/src/ATen/native/Activation.cpp:540-545` (which delegates to
/// `at::elu(self, alpha, 1.0, 1/alpha)`) and
/// `torch.nn.functional.celu(input, alpha=1.0)` at
/// `torch/nn/functional.py:1874-1894`.
///
/// `alpha` must be non-zero — upstream raises
/// `"ZeroDivisionError: alpha cannot be 0 for CELU"` at line 541-542.
pub fn celu<T: Float>(input: &Tensor<T>, alpha: f64) -> FerrotorchResult<Tensor<T>> {
    if alpha == 0.0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "celu: alpha cannot be 0 (ZeroDivisionError)".into(),
        });
    }
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "celu" });
    }
    let zero = <T as num_traits::Zero>::zero();
    let one = <T as num_traits::One>::one();
    let alpha_t = t_from::<T>(alpha, "celu")?;
    let inv_alpha = t_from::<T>(1.0 / alpha, "celu")?;
    let output = unary_map(input, |x| {
        if x > zero {
            x
        } else {
            alpha_t * ((x * inv_alpha).exp() - one)
        }
    })?;
    if is_grad_enabled() && input.requires_grad() {
        let (storage, shape) = output.into_storage_and_shape()?;
        Tensor::from_operation(
            storage,
            shape,
            Arc::new(CeluBackward::new(input.clone(), alpha)),
        )
    } else {
        Ok(output)
    }
}

// ===========================================================================
// Softmin (#1341 REQ-22)
// ===========================================================================
//
// `softmin(x; dim) = softmax(-x; dim)`. Upstream
// `torch.nn.functional.softmin(input, dim=None, dtype=None)` at
// `torch/nn/functional.py:2095-2125`:
//
//   if dtype is None:
//       ret = (-input).softmax(dim)
//   else:
//       ret = (-input).softmax(dim, dtype=dtype)
//
// The composition route already works in `ferrotorch-nn::functional::softmin`
// (neg -> softmax, two GradFn nodes). The fused SoftminBackward here
// matches activation.rs's one-node-per-op convention.
//
// Forward: `out = softmax(-x)` along the last axis (matching this file's
// `softmax` convention).
// Backward: Let `s = softmax(-x) = softmin(x)`. By the chain rule
// `dout/dx = -dsoftmax/du * du/dx` where `u = -x` (so `du/dx = -1`). Hence
// the softmax-VJP applied to `-grad_output` gives the softmin-VJP applied to
// `grad_output`:
//   `grad_input = -(softmax_VJP(grad, s))`
//          `= -(s * (grad - sum_k(grad_k * s_k)))`
//          `= s * (sum_k(grad_k * s_k) - grad)`
// per the chain rule on `softmax(-x)`.

/// Backward for `softmin(x) = softmax(-x)` along the last axis.
///
/// VJP: `s * (sum_k(grad_k * s_k) - grad)` where `s = softmin(x)` (cached
/// output). Derived from `softmax(-x)` via chain rule (`du/dx = -1`).
#[derive(Debug)]
pub struct SoftminBackward<T: Float> {
    input: Tensor<T>,
    output: Tensor<T>,
}

impl<T: Float> SoftminBackward<T> {
    pub fn new(input: Tensor<T>, output: Tensor<T>) -> Self {
        Self { input, output }
    }
}

impl<T: Float> GradFn<T> for SoftminBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if grad_output.is_cuda() || self.output.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "softmin backward",
            });
        }

        let s_data = self.output.data()?;
        let grad_data = grad_output.data()?;
        let shape = self.output.shape();

        if shape.is_empty() {
            let zero = <T as num_traits::Zero>::zero();
            let grad_input = Tensor::from_storage(TensorStorage::cpu(vec![zero]), vec![], false)?;
            return Ok(vec![Some(grad_input)]);
        }

        let last_dim = match shape.last() {
            Some(&d) => d,
            None => 1,
        };
        let outer = s_data.len() / last_dim.max(1);
        let mut result = vec![<T as num_traits::Zero>::zero(); s_data.len()];

        for i in 0..outer {
            let base = i * last_dim;
            let mut dot = <T as num_traits::Zero>::zero();
            for j in 0..last_dim {
                dot += grad_data[base + j] * s_data[base + j];
            }
            // softmin_VJP = s * (dot - grad)   [negation of softmax_VJP]
            for j in 0..last_dim {
                result[base + j] = s_data[base + j] * (dot - grad_data[base + j]);
            }
        }

        let grad_input = Tensor::from_storage(
            TensorStorage::cpu(result),
            self.input.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "SoftminBackward"
    }
}

/// Compute `softmin(x) = softmax(-x)` along the last axis, attaching a fused
/// backward node when gradients are enabled.
///
/// Mirrors `torch.nn.functional.softmin` at
/// `torch/nn/functional.py:2095-2125` (`ret = (-input).softmax(dim)`).
/// Stores the output (= softmin(x) = softmax(-x)) for backward efficiency —
/// one VJP node, vs. the two-node `neg -> softmax` composition still
/// available via `ferrotorch_nn::functional::softmin`.
pub fn softmin<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "softmin" });
    }
    let shape = input.shape().to_vec();
    let data = input.data()?;

    let is_bf16_t = std::any::TypeId::of::<T>() == std::any::TypeId::of::<half::bf16>();

    let result = if shape.is_empty() {
        vec![<T as num_traits::One>::one()]
    } else {
        let last_dim = match shape.last() {
            Some(&d) => d,
            None => 1,
        };
        let outer = data.len() / last_dim.max(1);
        let mut out = vec![<T as num_traits::Zero>::zero(); data.len()];

        if is_bf16_t {
            // bf16 path: promote accumulator to f32 (mirrors softmax_inner).
            let mut scratch = vec![0.0f32; last_dim];
            for i in 0..outer {
                let base = i * last_dim;
                // softmin = softmax(-x): negate inputs into scratch first.
                let mut row_max = f32::NEG_INFINITY;
                for j in 0..last_dim {
                    let v_t = data[base + j];
                    let v32 = match v_t.to_f32() {
                        Some(v) => -v,
                        None => {
                            return Err(FerrotorchError::InvalidArgument {
                                message: "softmin: bf16 input not representable as f32".into(),
                            });
                        }
                    };
                    scratch[j] = v32;
                    if v32 > row_max {
                        row_max = v32;
                    }
                }
                let mut sum_exp = 0.0f32;
                for slot in &mut scratch[..last_dim] {
                    let e = (*slot - row_max).exp();
                    *slot = e;
                    sum_exp += e;
                }
                if sum_exp > 0.0 {
                    let inv = 1.0f32 / sum_exp;
                    for j in 0..last_dim {
                        out[base + j] = T::from(scratch[j] * inv).ok_or_else(|| {
                            FerrotorchError::InvalidArgument {
                                message: "softmin: bf16 output not representable".into(),
                            }
                        })?;
                    }
                } else {
                    for j in 0..last_dim {
                        out[base + j] = <T as num_traits::Zero>::zero();
                    }
                }
            }
        } else {
            for i in 0..outer {
                let base = i * last_dim;
                // softmin = softmax(-x): work with -x[j].
                let mut max_val = -data[base];
                for j in 1..last_dim {
                    let v = -data[base + j];
                    if v > max_val {
                        max_val = v;
                    }
                }
                let mut sum_exp = <T as num_traits::Zero>::zero();
                for j in 0..last_dim {
                    let e = (-data[base + j] - max_val).exp();
                    out[base + j] = e;
                    sum_exp += e;
                }
                #[allow(clippy::assign_op_pattern)]
                for j in 0..last_dim {
                    out[base + j] = out[base + j] / sum_exp;
                }
            }
        }
        out
    };

    let output = Tensor::from_storage(TensorStorage::cpu(result), shape, false)?;

    if is_grad_enabled() && input.requires_grad() {
        let saved_input = input.clone();
        let saved_output = output.clone();
        Tensor::from_operation(
            TensorStorage::cpu(output.data()?.to_vec()),
            output.shape().to_vec(),
            Arc::new(SoftminBackward::new(saved_input, saved_output)),
        )
    } else {
        Ok(output)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::graph::backward;
    use crate::storage::TensorStorage;

    /// Helper: create a scalar leaf tensor with `requires_grad = true`.
    fn leaf_scalar(val: f64) -> Tensor<f64> {
        Tensor::from_storage(TensorStorage::cpu(vec![val]), vec![], true).unwrap()
    }

    /// Helper: create a 1-D leaf tensor with `requires_grad = true`.
    fn leaf_vec(vals: &[f64]) -> Tensor<f64> {
        Tensor::from_storage(TensorStorage::cpu(vals.to_vec()), vec![vals.len()], true).unwrap()
    }

    /// Numerical gradient via central difference: (f(x+h) - f(x-h)) / (2h).
    fn numerical_grad_scalar(f: impl Fn(f64) -> f64, x: f64) -> f64 {
        let h = 1e-5;
        (f(x + h) - f(x - h)) / (2.0 * h)
    }

    // -----------------------------------------------------------------------
    // ReLU
    // -----------------------------------------------------------------------

    #[test]
    fn test_relu_forward_positive() {
        let x = leaf_scalar(2.0);
        let y = relu(&x).unwrap();
        assert!((y.item().unwrap() - 2.0).abs() < 1e-7);
    }

    #[test]
    fn test_relu_forward_negative() {
        let x = leaf_scalar(-3.0);
        let y = relu(&x).unwrap();
        assert!((y.item().unwrap()).abs() < 1e-7);
    }

    #[test]
    fn test_relu_backward_positive() {
        let x = leaf_scalar(2.0);
        let y = relu(&x).unwrap();
        backward(&y).unwrap();

        let grad = x.grad().unwrap().unwrap();
        // d(relu)/dx at x=2 is 1.
        assert!(
            (grad.item().unwrap() - 1.0).abs() < 1e-6,
            "relu grad at x=2: expected 1.0, got {}",
            grad.item().unwrap()
        );
    }

    #[test]
    fn test_relu_backward_negative() {
        let x = leaf_scalar(-1.5);
        let y = relu(&x).unwrap();
        backward(&y).unwrap();

        let grad = x.grad().unwrap().unwrap();
        // d(relu)/dx at x=-1.5 is 0.
        assert!(
            grad.item().unwrap().abs() < 1e-6,
            "relu grad at x=-1.5: expected 0.0, got {}",
            grad.item().unwrap()
        );
    }

    #[test]
    fn test_relu_forward_vector() {
        let x = leaf_vec(&[-1.0, 0.5, 2.0, -0.3]);
        let y = relu(&x).unwrap();
        let y_data = y.data().unwrap();
        assert!((y_data[0] - 0.0).abs() < 1e-7);
        assert!((y_data[1] - 0.5).abs() < 1e-7);
        assert!((y_data[2] - 2.0).abs() < 1e-7);
        assert!((y_data[3] - 0.0).abs() < 1e-7);
    }

    // -----------------------------------------------------------------------
    // Sigmoid
    // -----------------------------------------------------------------------

    #[test]
    fn test_sigmoid_forward() {
        let x = leaf_scalar(0.0);
        let y = sigmoid(&x).unwrap();
        assert!((y.item().unwrap() - 0.5).abs() < 1e-7);
    }

    #[test]
    fn test_sigmoid_backward() {
        let x = leaf_scalar(0.0);
        let y = sigmoid(&x).unwrap();
        backward(&y).unwrap();

        let grad = x.grad().unwrap().unwrap();
        // sigmoid'(0) = 0.5 * (1 - 0.5) = 0.25.
        assert!(
            (grad.item().unwrap() - 0.25).abs() < 1e-6,
            "sigmoid grad at x=0: expected 0.25, got {}",
            grad.item().unwrap()
        );
    }

    #[test]
    fn test_sigmoid_backward_nonzero() {
        let val = 1.0_f64;
        let x = leaf_scalar(val);
        let y = sigmoid(&x).unwrap();
        backward(&y).unwrap();

        let grad = x.grad().unwrap().unwrap();
        // Compare with numerical gradient.
        let expected = numerical_grad_scalar(|v| 1.0 / (1.0 + (-v).exp()), val);
        assert!(
            (grad.item().unwrap() - expected).abs() < 1e-5,
            "sigmoid grad at x={}: expected {}, got {}",
            val,
            expected,
            grad.item().unwrap()
        );
    }

    // -----------------------------------------------------------------------
    // Tanh
    // -----------------------------------------------------------------------

    #[test]
    fn test_tanh_forward() {
        let x = leaf_scalar(0.0);
        let y = tanh(&x).unwrap();
        assert!(y.item().unwrap().abs() < 1e-7);
    }

    #[test]
    fn test_tanh_backward_at_zero() {
        let x = leaf_scalar(0.0);
        let y = tanh(&x).unwrap();
        backward(&y).unwrap();

        let grad = x.grad().unwrap().unwrap();
        // tanh'(0) = 1 - tanh(0)^2 = 1.
        assert!(
            (grad.item().unwrap() - 1.0).abs() < 1e-6,
            "tanh grad at x=0: expected 1.0, got {}",
            grad.item().unwrap()
        );
    }

    #[test]
    fn test_tanh_backward_nonzero() {
        let val = 0.8_f64;
        let x = leaf_scalar(val);
        let y = tanh(&x).unwrap();
        backward(&y).unwrap();

        let grad = x.grad().unwrap().unwrap();
        let expected = numerical_grad_scalar(|v| v.tanh(), val);
        assert!(
            (grad.item().unwrap() - expected).abs() < 1e-5,
            "tanh grad at x={}: expected {}, got {}",
            val,
            expected,
            grad.item().unwrap()
        );
    }

    // -----------------------------------------------------------------------
    // GELU
    // -----------------------------------------------------------------------

    #[test]
    fn test_gelu_forward_zero() {
        // gelu(0) = 0 for all modes.
        for mode in [
            GeluApproximate::None,
            GeluApproximate::Tanh,
            GeluApproximate::Sigmoid,
        ] {
            let x = leaf_scalar(0.0);
            let y = gelu_with(&x, mode).unwrap();
            assert!(
                y.item().unwrap().abs() < 1e-7,
                "gelu({mode}) at 0 should be 0"
            );
        }
    }

    #[test]
    fn test_gelu_exact_forward_values() {
        // Test exact (erf) mode against known values.
        // gelu(1.0) = 1.0 * 0.5 * (1 + erf(1/sqrt(2))) ≈ 0.8413
        let x = leaf_scalar(1.0);
        let y = gelu_with(&x, GeluApproximate::None).unwrap();
        let val = y.item().unwrap();
        assert!(
            (val - 0.8413).abs() < 1e-3,
            "exact gelu(1.0) ≈ 0.8413, got {val}"
        );

        // gelu(-1.0) ≈ -0.1587
        let x = leaf_scalar(-1.0);
        let y = gelu_with(&x, GeluApproximate::None).unwrap();
        let val = y.item().unwrap();
        assert!(
            (val - (-0.1587)).abs() < 1e-3,
            "exact gelu(-1.0) ≈ -0.1587, got {val}"
        );
    }

    #[test]
    fn test_gelu_tanh_forward_values() {
        // Tanh approx should be close to exact.
        let x = leaf_scalar(1.0);
        let y = gelu_with(&x, GeluApproximate::Tanh).unwrap();
        let val = y.item().unwrap();
        assert!(
            (val - 0.8412).abs() < 2e-3,
            "tanh gelu(1.0) ≈ 0.8412, got {val}"
        );
    }

    #[test]
    fn test_gelu_sigmoid_forward_values() {
        // Sigmoid approx: gelu(x) = x * sigmoid(1.702 * x)
        let x = leaf_scalar(1.0);
        let y = gelu_with(&x, GeluApproximate::Sigmoid).unwrap();
        let val = y.item().unwrap();
        let expected = 1.0 / (1.0 + (-1.702_f64).exp());
        assert!(
            (val - expected).abs() < 1e-5,
            "sigmoid gelu(1.0) ≈ {expected}, got {val}"
        );
    }

    #[test]
    fn test_gelu_backward_exact() {
        let val = 1.0_f64;
        let x = leaf_scalar(val);
        let y = gelu_with(&x, GeluApproximate::None).unwrap();
        backward(&y).unwrap();

        let grad = x.grad().unwrap().unwrap();
        let expected = numerical_grad_scalar(
            |v| {
                let sqrt_2 = std::f64::consts::SQRT_2;
                let cdf = 0.5 * (1.0 + erf_approx(v / sqrt_2));
                v * cdf
            },
            val,
        );
        assert!(
            (grad.item().unwrap() - expected).abs() < 1e-4,
            "exact gelu grad at x={val}: expected {expected}, got {}",
            grad.item().unwrap()
        );
    }

    #[test]
    fn test_gelu_backward_tanh() {
        let val = 1.0_f64;
        let x = leaf_scalar(val);
        let y = gelu_with(&x, GeluApproximate::Tanh).unwrap();
        backward(&y).unwrap();

        let grad = x.grad().unwrap().unwrap();
        let expected = numerical_grad_scalar(
            |v| {
                let sqrt_2_over_pi = (2.0 / std::f64::consts::PI).sqrt();
                let inner = sqrt_2_over_pi * (v + 0.044715 * v * v * v);
                0.5 * v * (1.0 + inner.tanh())
            },
            val,
        );
        assert!(
            (grad.item().unwrap() - expected).abs() < 1e-4,
            "tanh gelu grad at x={val}: expected {expected}, got {}",
            grad.item().unwrap()
        );
    }

    #[test]
    fn test_gelu_backward_sigmoid() {
        let val = 1.0_f64;
        let x = leaf_scalar(val);
        let y = gelu_with(&x, GeluApproximate::Sigmoid).unwrap();
        backward(&y).unwrap();

        let grad = x.grad().unwrap().unwrap();
        let k = 1.702_f64;
        let expected = numerical_grad_scalar(
            |v| {
                let s = 1.0 / (1.0 + (-k * v).exp());
                v * s
            },
            val,
        );
        assert!(
            (grad.item().unwrap() - expected).abs() < 1e-4,
            "sigmoid gelu grad at x={val}: expected {expected}, got {}",
            grad.item().unwrap()
        );
    }

    #[test]
    fn test_gelu_default_is_exact() {
        // gelu() without mode should use exact (erf).
        let x = leaf_scalar(1.0);
        let y_default = gelu(&x).unwrap();
        let x2 = leaf_scalar(1.0);
        let y_exact = gelu_with(&x2, GeluApproximate::None).unwrap();
        assert!(
            (y_default.item().unwrap() - y_exact.item().unwrap()).abs() < 1e-10,
            "default gelu should match exact mode"
        );
    }

    // -----------------------------------------------------------------------
    // SiLU
    // -----------------------------------------------------------------------

    #[test]
    fn test_silu_forward_zero() {
        let x = leaf_scalar(0.0);
        let y = silu(&x).unwrap();
        // silu(0) = 0 * sigmoid(0) = 0.
        assert!(y.item().unwrap().abs() < 1e-7);
    }

    #[test]
    fn test_silu_backward() {
        let val = 1.5_f64;
        let x = leaf_scalar(val);
        let y = silu(&x).unwrap();
        backward(&y).unwrap();

        let grad = x.grad().unwrap().unwrap();
        let expected = numerical_grad_scalar(
            |v| {
                let s = 1.0 / (1.0 + (-v).exp());
                v * s
            },
            val,
        );
        assert!(
            (grad.item().unwrap() - expected).abs() < 1e-4,
            "silu grad at x={}: expected {}, got {}",
            val,
            expected,
            grad.item().unwrap()
        );
    }

    // -----------------------------------------------------------------------
    // Softmax
    // -----------------------------------------------------------------------

    #[test]
    fn test_softmax_forward_1d() {
        let x = leaf_vec(&[1.0, 2.0, 3.0]);
        let y = softmax(&x).unwrap();
        let d = y.data().unwrap();
        // Softmax values should sum to 1.
        let total: f64 = d.iter().copied().sum();
        assert!(
            (total - 1.0).abs() < 1e-7,
            "softmax sum: expected 1.0, got {total}"
        );
        // Monotonicity: s(1) < s(2) < s(3).
        assert!(d[0] < d[1]);
        assert!(d[1] < d[2]);
    }

    #[test]
    fn test_softmax_backward_1d() {
        // For a 1D softmax, verify the backward struct directly.
        let vals = [1.0_f64, 2.0, 3.0];
        let x = leaf_vec(&vals);
        let y = softmax(&x).unwrap();
        let y_data = y.data().unwrap().to_vec();

        // Use grad_output = [1, 0, 0] to probe d(softmax_0)/d(x_j).
        let grad_output =
            Tensor::from_storage(TensorStorage::cpu(vec![1.0, 0.0, 0.0]), vec![3], false).unwrap();

        let bwd = SoftmaxBackward::new(x.clone(), y.clone());
        let grads = bwd.backward(&grad_output).unwrap();
        let gx = grads[0].as_ref().unwrap().data().unwrap().to_vec();

        // Expected: s_0 * (delta_{0j} - s_j)
        let s0 = y_data[0];
        let s1 = y_data[1];
        let s2 = y_data[2];
        let expected = [s0 * (1.0 - s0), s0 * (0.0 - s1), s0 * (0.0 - s2)];

        for (i, (&got, &exp)) in gx.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-7,
                "softmax grad[{i}]: expected {exp}, got {got}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // LogSoftmax
    // -----------------------------------------------------------------------

    #[test]
    fn test_log_softmax_forward_1d() {
        let x = leaf_vec(&[1.0, 2.0, 3.0]);
        let y = log_softmax(&x).unwrap();
        let d = y.data().unwrap();
        // exp(log_softmax) should sum to 1.
        let total: f64 = d.iter().map(|&v| v.exp()).sum();
        assert!(
            (total - 1.0).abs() < 1e-7,
            "exp(log_softmax) sum: expected 1.0, got {total}"
        );
    }

    #[test]
    fn test_log_softmax_backward_1d() {
        let vals = [1.0_f64, 2.0, 3.0];
        let x = leaf_vec(&vals);

        // Compute softmax for reference (on a non-grad tensor to avoid
        // entangling computation graphs).
        let x_nograd =
            Tensor::from_storage(TensorStorage::cpu(vals.to_vec()), vec![3], false).unwrap();
        let sm = softmax(&x_nograd).unwrap();
        let sm_data = sm.data().unwrap().to_vec();

        let grad_output =
            Tensor::from_storage(TensorStorage::cpu(vec![1.0, 0.0, 0.0]), vec![3], false).unwrap();

        let bwd = LogSoftmaxBackward::new(x.clone(), sm);
        let grads = bwd.backward(&grad_output).unwrap();
        let gx = grads[0].as_ref().unwrap().data().unwrap().to_vec();

        // Expected: grad - softmax * sum(grad)
        // sum(grad) = 1.0
        // grad_input = [1 - s0, 0 - s1, 0 - s2]
        let expected = [1.0 - sm_data[0], 0.0 - sm_data[1], 0.0 - sm_data[2]];

        for (i, (&got, &exp)) in gx.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-7,
                "log_softmax grad[{i}]: expected {exp}, got {got}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // no_grad disables backward nodes
    // -----------------------------------------------------------------------

    #[test]
    fn test_relu_no_grad() {
        crate::autograd::no_grad::no_grad(|| {
            let x = leaf_scalar(2.0);
            let y = relu(&x).unwrap();
            assert!(
                y.grad_fn().is_none(),
                "relu inside no_grad should not attach grad_fn"
            );
        });
    }

    #[test]
    fn test_sigmoid_no_grad() {
        crate::autograd::no_grad::no_grad(|| {
            let x = leaf_scalar(1.0);
            let y = sigmoid(&x).unwrap();
            assert!(
                y.grad_fn().is_none(),
                "sigmoid inside no_grad should not attach grad_fn"
            );
        });
    }

    // -----------------------------------------------------------------------
    // Softplus
    // -----------------------------------------------------------------------

    #[test]
    fn test_softplus_forward_zero() {
        let x = leaf_scalar(0.0);
        let y = softplus(&x, 1.0, 20.0).unwrap();
        // softplus(0) = ln(1 + 1) = ln(2)
        assert!(
            (y.item().unwrap() - 2.0_f64.ln()).abs() < 1e-7,
            "softplus(0) = {}, expected {}",
            y.item().unwrap(),
            2.0_f64.ln()
        );
    }

    #[test]
    fn test_softplus_forward_large() {
        let x = leaf_scalar(25.0);
        let y = softplus(&x, 1.0, 20.0).unwrap();
        // For beta*x > threshold, softplus(x) = x.
        assert!(
            (y.item().unwrap() - 25.0).abs() < 1e-5,
            "softplus(25) = {}, expected 25.0",
            y.item().unwrap()
        );
    }

    #[test]
    fn test_softplus_backward_at_zero() {
        let x = leaf_scalar(0.0);
        let y = softplus(&x, 1.0, 20.0).unwrap();
        backward(&y).unwrap();

        let grad = x.grad().unwrap().unwrap();
        // d/dx softplus(0) = sigmoid(0) = 0.5
        assert!(
            (grad.item().unwrap() - 0.5).abs() < 1e-6,
            "softplus grad at x=0: expected 0.5, got {}",
            grad.item().unwrap()
        );
    }

    #[test]
    fn test_softplus_backward_positive() {
        let val = 2.0_f64;
        let x = leaf_scalar(val);
        let y = softplus(&x, 1.0, 20.0).unwrap();
        backward(&y).unwrap();

        let grad = x.grad().unwrap().unwrap();
        let expected = numerical_grad_scalar(|v| (1.0 + v.exp()).ln(), val);
        assert!(
            (grad.item().unwrap() - expected).abs() < 1e-4,
            "softplus grad at x={}: expected {}, got {}",
            val,
            expected,
            grad.item().unwrap()
        );
    }

    #[test]
    fn test_softplus_backward_negative() {
        let val = -1.5_f64;
        let x = leaf_scalar(val);
        let y = softplus(&x, 1.0, 20.0).unwrap();
        backward(&y).unwrap();

        let grad = x.grad().unwrap().unwrap();
        let expected = numerical_grad_scalar(|v| (1.0 + v.exp()).ln(), val);
        assert!(
            (grad.item().unwrap() - expected).abs() < 1e-4,
            "softplus grad at x={}: expected {}, got {}",
            val,
            expected,
            grad.item().unwrap()
        );
    }

    #[test]
    fn test_softplus_backward_custom_beta() {
        let val = 1.0_f64;
        let beta = 2.0_f64;
        let x = leaf_scalar(val);
        let y = softplus(&x, beta, 20.0).unwrap();
        backward(&y).unwrap();

        let grad = x.grad().unwrap().unwrap();
        let expected = numerical_grad_scalar(|v| (1.0 + (beta * v).exp()).ln() / beta, val);
        assert!(
            (grad.item().unwrap() - expected).abs() < 1e-4,
            "softplus grad at x={}, beta={}: expected {}, got {}",
            val,
            beta,
            expected,
            grad.item().unwrap()
        );
    }

    #[test]
    fn test_softplus_backward_vector() {
        let x = leaf_vec(&[-2.0, -0.5, 0.0, 1.0, 3.0]);
        let y = softplus(&x, 1.0, 20.0).unwrap();

        // Sum to get a scalar for backward.
        let sum = crate::grad_fns::reduction::sum(&y).unwrap();
        backward(&sum).unwrap();

        let grad = x.grad().unwrap().unwrap();
        let grad_data = grad.data().unwrap();

        for (i, &val) in [-2.0_f64, -0.5, 0.0, 1.0, 3.0].iter().enumerate() {
            let expected = numerical_grad_scalar(|v| (1.0 + v.exp()).ln(), val);
            assert!(
                (grad_data[i] - expected).abs() < 1e-4,
                "softplus grad[{}] at x={}: expected {}, got {}",
                i,
                val,
                expected,
                grad_data[i]
            );
        }
    }

    #[test]
    fn test_softplus_no_grad() {
        crate::autograd::no_grad::no_grad(|| {
            let x = leaf_scalar(1.0);
            let y = softplus(&x, 1.0, 20.0).unwrap();
            assert!(
                y.grad_fn().is_none(),
                "softplus inside no_grad should not attach grad_fn"
            );
        });
    }

    // -----------------------------------------------------------------------
    // ELU
    // -----------------------------------------------------------------------

    #[test]
    fn test_elu_forward_positive() {
        let x = leaf_scalar(2.0);
        let y = elu(&x, 1.0).unwrap();
        assert!((y.item().unwrap() - 2.0).abs() < 1e-7);
    }

    #[test]
    fn test_elu_forward_negative() {
        let x = leaf_scalar(-1.0);
        let y = elu(&x, 1.0).unwrap();
        let expected = (-1.0_f64).exp() - 1.0;
        assert!(
            (y.item().unwrap() - expected).abs() < 1e-7,
            "elu(-1) = {}, expected {}",
            y.item().unwrap(),
            expected
        );
    }

    #[test]
    fn test_elu_backward_positive() {
        let x = leaf_scalar(2.0);
        let y = elu(&x, 1.0).unwrap();
        backward(&y).unwrap();

        let grad = x.grad().unwrap().unwrap();
        // d/dx elu(x) at x=2 > 0 is 1.
        assert!(
            (grad.item().unwrap() - 1.0).abs() < 1e-6,
            "elu grad at x=2: expected 1.0, got {}",
            grad.item().unwrap()
        );
    }

    #[test]
    fn test_elu_backward_negative() {
        let val = -1.0_f64;
        let alpha = 1.0_f64;
        let x = leaf_scalar(val);
        let y = elu(&x, alpha).unwrap();
        backward(&y).unwrap();

        let grad = x.grad().unwrap().unwrap();
        let expected =
            numerical_grad_scalar(|v| if v > 0.0 { v } else { alpha * (v.exp() - 1.0) }, val);
        assert!(
            (grad.item().unwrap() - expected).abs() < 1e-4,
            "elu grad at x={}: expected {}, got {}",
            val,
            expected,
            grad.item().unwrap()
        );
    }

    #[test]
    fn test_elu_backward_custom_alpha() {
        let val = -0.5_f64;
        let alpha = 2.0_f64;
        let x = leaf_scalar(val);
        let y = elu(&x, alpha).unwrap();
        backward(&y).unwrap();

        let grad = x.grad().unwrap().unwrap();
        // d/dx [alpha * (exp(x) - 1)] = alpha * exp(x) at x = -0.5
        let expected = alpha * val.exp();
        assert!(
            (grad.item().unwrap() - expected).abs() < 1e-5,
            "elu grad at x={}, alpha={}: expected {}, got {}",
            val,
            alpha,
            expected,
            grad.item().unwrap()
        );
    }

    #[test]
    fn test_elu_no_grad() {
        crate::autograd::no_grad::no_grad(|| {
            let x = leaf_scalar(1.0);
            let y = elu(&x, 1.0).unwrap();
            assert!(
                y.grad_fn().is_none(),
                "elu inside no_grad should not attach grad_fn"
            );
        });
    }

    // -----------------------------------------------------------------------
    // Mish
    // -----------------------------------------------------------------------

    #[test]
    fn test_mish_forward_zero() {
        let x = leaf_scalar(0.0);
        let y = mish(&x).unwrap();
        // mish(0) = 0 * tanh(ln(2)) = 0
        assert!(y.item().unwrap().abs() < 1e-7);
    }

    #[test]
    fn test_mish_forward_positive() {
        let x = leaf_scalar(20.0);
        let y = mish(&x).unwrap();
        // For large x, mish(x) -> x.
        assert!(
            (y.item().unwrap() - 20.0).abs() < 0.01,
            "mish(20) = {}, expected ~20",
            y.item().unwrap()
        );
    }

    #[test]
    fn test_mish_backward_at_zero() {
        let x = leaf_scalar(0.0);
        let y = mish(&x).unwrap();
        backward(&y).unwrap();

        let grad = x.grad().unwrap().unwrap();
        let expected = numerical_grad_scalar(
            |v| {
                let sp = (1.0 + v.exp()).ln();
                v * sp.tanh()
            },
            0.0,
        );
        assert!(
            (grad.item().unwrap() - expected).abs() < 1e-4,
            "mish grad at x=0: expected {}, got {}",
            expected,
            grad.item().unwrap()
        );
    }

    #[test]
    fn test_mish_backward_positive() {
        let val = 1.5_f64;
        let x = leaf_scalar(val);
        let y = mish(&x).unwrap();
        backward(&y).unwrap();

        let grad = x.grad().unwrap().unwrap();
        let expected = numerical_grad_scalar(
            |v| {
                let sp = (1.0 + v.exp()).ln();
                v * sp.tanh()
            },
            val,
        );
        assert!(
            (grad.item().unwrap() - expected).abs() < 1e-4,
            "mish grad at x={}: expected {}, got {}",
            val,
            expected,
            grad.item().unwrap()
        );
    }

    #[test]
    fn test_mish_backward_negative() {
        let val = -1.0_f64;
        let x = leaf_scalar(val);
        let y = mish(&x).unwrap();
        backward(&y).unwrap();

        let grad = x.grad().unwrap().unwrap();
        let expected = numerical_grad_scalar(
            |v| {
                let sp = (1.0 + v.exp()).ln();
                v * sp.tanh()
            },
            val,
        );
        assert!(
            (grad.item().unwrap() - expected).abs() < 1e-4,
            "mish grad at x={}: expected {}, got {}",
            val,
            expected,
            grad.item().unwrap()
        );
    }

    #[test]
    fn test_mish_no_grad() {
        crate::autograd::no_grad::no_grad(|| {
            let x = leaf_scalar(1.0);
            let y = mish(&x).unwrap();
            assert!(
                y.grad_fn().is_none(),
                "mish inside no_grad should not attach grad_fn"
            );
        });
    }

    // -----------------------------------------------------------------------
    // Activation tail (#594)
    // -----------------------------------------------------------------------

    #[test]
    fn test_leaky_relu_forward_positive_unchanged() {
        let x = leaf_scalar(2.0);
        let y = leaky_relu(&x, 0.1).unwrap();
        assert!((y.item().unwrap() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_leaky_relu_forward_negative_scaled() {
        let x = leaf_scalar(-3.0);
        let y = leaky_relu(&x, 0.1).unwrap();
        assert!((y.item().unwrap() - (-0.3)).abs() < 1e-9);
    }

    #[test]
    fn test_leaky_relu_backward() {
        let x = leaf_vec(&[2.0, -1.0]);
        let y = leaky_relu(&x, 0.25).unwrap();
        let s = y.sum_all().unwrap();
        backward(&s).unwrap();
        let g = x.grad().unwrap().unwrap();
        let gd = g.data().unwrap();
        // d/dx where x=2: 1; where x=-1: 0.25
        assert!((gd[0] - 1.0).abs() < 1e-9);
        assert!((gd[1] - 0.25).abs() < 1e-9);
    }

    #[test]
    fn test_hardtanh_clamps_and_grad() {
        let x = leaf_vec(&[-2.0, -0.5, 0.5, 2.0]);
        let y = hardtanh(&x).unwrap();
        let yd = y.data().unwrap();
        assert_eq!(yd, &[-1.0, -0.5, 0.5, 1.0]);
        let s = y.sum_all().unwrap();
        backward(&s).unwrap();
        let g = x.grad().unwrap().unwrap();
        let gd = g.data().unwrap();
        // d/dx is 0 at clamped boundaries, 1 inside.
        assert_eq!(gd, &[0.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn test_relu6_clamps_top_at_6() {
        let x = leaf_vec(&[-1.0, 0.0, 3.0, 7.0]);
        let y = relu6(&x).unwrap();
        let d = y.data().unwrap();
        assert_eq!(d, &[0.0, 0.0, 3.0, 6.0]);
    }

    #[test]
    fn test_hardsigmoid_endpoints_and_slope() {
        // Forward.
        let x = leaf_vec(&[-5.0, -3.0, 0.0, 3.0, 5.0]);
        let y = hardsigmoid(&x).unwrap();
        let d = y.data().unwrap();
        assert!((d[0] - 0.0).abs() < 1e-9);
        assert!((d[1] - 0.0).abs() < 1e-9);
        assert!((d[2] - 0.5).abs() < 1e-9);
        assert!((d[3] - 1.0).abs() < 1e-9);
        assert!((d[4] - 1.0).abs() < 1e-9);

        // Backward: slope = 1/6 inside (-3, 3); 0 outside.
        let s = y.sum_all().unwrap();
        backward(&s).unwrap();
        let g = x.grad().unwrap().unwrap();
        let gd = g.data().unwrap();
        assert!((gd[0]).abs() < 1e-9);
        assert!((gd[2] - 1.0 / 6.0).abs() < 1e-9);
        assert!((gd[4]).abs() < 1e-9);
    }

    #[test]
    fn test_hardswish_zero_below_minus_three() {
        let x = leaf_vec(&[-5.0, -3.0, 0.0, 1.0, 5.0]);
        let y = hardswish(&x).unwrap();
        let d = y.data().unwrap();
        assert!((d[0]).abs() < 1e-9);
        assert!((d[1]).abs() < 1e-9);
        assert!((d[2]).abs() < 1e-9);
        // x=1 → 1 * (1+3)/6 = 4/6 ≈ 0.6667
        assert!((d[3] - 4.0 / 6.0).abs() < 1e-9);
        assert!((d[4] - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_hardswish_backward_matches_numerical() {
        // Pick x=1 (in the linear region). Closed-form: (2x + 3)/6 = 5/6.
        let x = leaf_scalar(1.0);
        let y = hardswish(&x).unwrap();
        backward(&y).unwrap();
        let g = x.grad().unwrap().unwrap();
        assert!((g.item().unwrap() - 5.0 / 6.0).abs() < 1e-9);
    }

    #[test]
    fn test_selu_zero_at_origin() {
        let x = leaf_vec(&[-1.0, 0.0, 1.0]);
        let y = selu(&x).unwrap();
        let d = y.data().unwrap();
        // selu(0) = 0; selu(1) = scale * 1 = 1.0507; selu(-1) = scale * alpha * (e^-1 - 1)
        assert!((d[1]).abs() < 1e-9);
        assert!((d[2] - 1.0507009873554805).abs() < 1e-9);
        let neg_expected = 1.0507009873554805 * 1.6732632423543772 * ((-1.0_f64).exp() - 1.0);
        assert!((d[0] - neg_expected).abs() < 1e-9);
    }

    #[test]
    fn test_selu_backward_at_one_is_scale() {
        // d/dx selu(x) at x=1 = scale * 1 = 1.0507.
        let x = leaf_scalar(1.0);
        let y = selu(&x).unwrap();
        backward(&y).unwrap();
        let g = x.grad().unwrap().unwrap();
        assert!((g.item().unwrap() - 1.0507009873554805).abs() < 1e-9);
    }

    #[test]
    fn test_softsign_bounded_and_zero_origin() {
        let x = leaf_vec(&[-1000.0, -1.0, 0.0, 1.0, 1000.0]);
        let y = softsign(&x).unwrap();
        let d = y.data().unwrap();
        assert!((d[0] + 1.0).abs() < 1e-2); // approaches -1
        assert!((d[1] + 0.5).abs() < 1e-9);
        assert!((d[2]).abs() < 1e-9);
        assert!((d[3] - 0.5).abs() < 1e-9);
        assert!((d[4] - 1.0).abs() < 1e-2);
    }

    #[test]
    fn test_softsign_backward_closed_form() {
        // d/dx softsign(x) = 1 / (1 + |x|)^2; at x=1, = 1/4.
        let x = leaf_scalar(1.0);
        let y = softsign(&x).unwrap();
        backward(&y).unwrap();
        let g = x.grad().unwrap().unwrap();
        assert!((g.item().unwrap() - 0.25).abs() < 1e-9);
    }

    #[test]
    fn test_softsign_backward_at_zero_is_one() {
        let x = leaf_scalar(0.0);
        let y = softsign(&x).unwrap();
        backward(&y).unwrap();
        let g = x.grad().unwrap().unwrap();
        assert!((g.item().unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_softsign_backward_matches_numerical() {
        let x_val = 0.7_f64;
        let analytic = 1.0 / (1.0 + x_val.abs()).powi(2);
        let numerical = numerical_grad_scalar(|t| t / (1.0 + t.abs()), x_val);
        assert!((analytic - numerical).abs() < 1e-5);
    }

    #[test]
    fn test_activation_tail_no_grad_does_not_attach_grad_fn() {
        crate::autograd::no_grad::no_grad(|| {
            let x = leaf_scalar(0.5);
            for y in [
                leaky_relu(&x, 0.1).unwrap(),
                hardtanh(&x).unwrap(),
                relu6(&x).unwrap(),
                hardsigmoid(&x).unwrap(),
                hardswish(&x).unwrap(),
                selu(&x).unwrap(),
                softsign(&x).unwrap(),
            ] {
                assert!(y.grad_fn().is_none());
            }
        });
    }

    // -------------------------------------------------------------------
    // PReLU / GLU fused (#614)
    // -------------------------------------------------------------------

    #[test]
    fn prelu_forward_matches_definition() {
        let x = leaf_vec(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        let alpha =
            Tensor::from_storage(TensorStorage::cpu(vec![0.25_f64]), vec![1], false).unwrap();
        let y = prelu(&x, &alpha).unwrap();
        // alpha=0.25 -> negatives scaled by 0.25; non-negatives unchanged.
        assert_eq!(y.data().unwrap(), &[-0.5, -0.25, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn prelu_backward_routes_to_input_and_alpha() {
        // Build x and alpha as leaves with grad. Compute prelu, then sum.
        let x = leaf_vec(&[-2.0, 1.0]);
        let alpha = Tensor::from_storage(TensorStorage::cpu(vec![0.5_f64]), vec![1], true).unwrap();
        let y = prelu(&x, &alpha).unwrap();

        // Build a sum scalar so backward sees grad_output = ones.
        #[derive(Debug)]
        struct SumBack<T: Float> {
            input: Tensor<T>,
            numel: usize,
        }
        impl<T: Float> GradFn<T> for SumBack<T> {
            fn backward(&self, _g: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
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
                "Sum"
            }
        }

        let total: f64 = y.data().unwrap().iter().sum();
        let scalar = Tensor::from_operation(
            TensorStorage::cpu(vec![total]),
            vec![],
            Arc::new(SumBack {
                input: y.clone(),
                numel: 2,
            }),
        )
        .unwrap();
        backward(&scalar).unwrap();

        // dL/dx[0]: x[0]=-2 -> alpha=0.5
        // dL/dx[1]: x[1]= 1 -> 1.0
        let gx = x.grad().unwrap().unwrap();
        assert_eq!(gx.data().unwrap(), &[0.5, 1.0]);
        // dL/dalpha = sum(grad * x) over negatives only = 1.0 * -2.0 = -2.0
        let ga = alpha.grad().unwrap().unwrap();
        assert_eq!(ga.data().unwrap(), &[-2.0]);
    }

    #[test]
    fn prelu_per_channel_forward_backward_matches_torch_contract() {
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![
                -2.0_f64, 1.0, -3.0, 4.0, 0.0, -5.0, 6.0, -7.0, -8.0, 9.0, 10.0, -11.0,
            ]),
            vec![2, 3, 2],
            true,
        )
        .unwrap();
        let alpha =
            Tensor::from_storage(TensorStorage::cpu(vec![0.1_f64, 0.2, 0.3]), vec![3], true)
                .unwrap();

        let y = prelu(&x, &alpha).unwrap();
        let y_expected = [
            -0.2, 1.0, -0.6, 4.0, 0.0, -1.5, 6.0, -0.7, -1.6, 9.0, 10.0, -3.3,
        ];
        for (&actual, &expected) in y.data().unwrap().iter().zip(y_expected.iter()) {
            assert!((actual - expected).abs() < 1e-12);
        }

        let s = y.sum_all().unwrap();
        backward(&s).unwrap();

        let gx = x.grad().unwrap().unwrap();
        let gx_expected = [0.1, 1.0, 0.2, 1.0, 0.3, 0.3, 1.0, 0.1, 0.2, 1.0, 1.0, 0.3];
        for (&actual, &expected) in gx.data().unwrap().iter().zip(gx_expected.iter()) {
            assert!((actual - expected).abs() < 1e-12);
        }
        let ga = alpha.grad().unwrap().unwrap();
        assert_eq!(ga.data().unwrap(), &[-9.0, -11.0, -16.0]);
    }

    #[test]
    fn prelu_scalar_alpha_grad_preserves_nan_like_torch() {
        let x = leaf_vec(&[f64::NAN, -2.0, 1.0, 0.0]);
        let alpha = Tensor::from_storage(TensorStorage::cpu(vec![0.5_f64]), vec![1], true).unwrap();
        let y = prelu(&x, &alpha).unwrap();
        backward(&y.sum_all().unwrap()).unwrap();

        let ga = alpha.grad().unwrap().unwrap();
        assert!(
            ga.data().unwrap()[0].is_nan(),
            "torch prelu backward treats NaN input as the false x>0 branch"
        );
    }

    #[test]
    fn prelu_rejects_channel_mismatch_for_1d_input() {
        let x = leaf_vec(&[-1.0, 1.0]);
        let alpha =
            Tensor::from_storage(TensorStorage::cpu(vec![0.1, 0.2]), vec![2], false).unwrap();
        let err = prelu(&x, &alpha).unwrap_err();
        assert!(matches!(err, FerrotorchError::ShapeMismatch { .. }));
    }

    #[test]
    fn prelu_rejects_rank_two_alpha_even_when_numel_one() {
        let x = leaf_vec(&[-1.0, 1.0]);
        let alpha =
            Tensor::from_storage(TensorStorage::cpu(vec![0.25_f64]), vec![1, 1], false).unwrap();
        let err = prelu(&x, &alpha).unwrap_err();
        assert!(matches!(err, FerrotorchError::ShapeMismatch { .. }));
    }

    #[test]
    fn glu_forward_matches_split_sigmoid_mul() {
        // Input: [a0, a1, b0, b1] split on dim 0 -> a*[sigma(b0), sigma(b1)]
        let x = leaf_vec(&[1.0, 2.0, 0.0, 0.0]);
        let y = glu(&x, 0).unwrap();
        // sigmoid(0) = 0.5; output = [1.0*0.5, 2.0*0.5] = [0.5, 1.0]
        assert_eq!(y.shape(), &[2]);
        let out = y.data().unwrap();
        assert!((out[0] - 0.5).abs() < 1e-9);
        assert!((out[1] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn glu_backward_matches_decomposition() {
        // For x = [a, b] (1-D length 2), y = a * sigmoid(b).
        // sum(y) backward: grad_a = sigmoid(b), grad_b = a*sigmoid(b)*(1-sigmoid(b))
        let x = leaf_vec(&[3.0, 0.5]);
        let y = glu(&x, 0).unwrap();

        #[derive(Debug)]
        struct SumBack<T: Float> {
            input: Tensor<T>,
            numel: usize,
        }
        impl<T: Float> GradFn<T> for SumBack<T> {
            fn backward(&self, _g: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
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
                "Sum"
            }
        }

        let total: f64 = y.data().unwrap().iter().sum();
        let scalar = Tensor::from_operation(
            TensorStorage::cpu(vec![total]),
            vec![],
            Arc::new(SumBack {
                input: y.clone(),
                numel: 1,
            }),
        )
        .unwrap();
        backward(&scalar).unwrap();

        let g = x.grad().unwrap().unwrap();
        let g_data = g.data().unwrap();
        // sigmoid(0.5) ~ 0.6224593312
        let s = 1.0 / (1.0 + (-0.5_f64).exp());
        assert!((g_data[0] - s).abs() < 1e-9);
        assert!((g_data[1] - 3.0 * s * (1.0 - s)).abs() < 1e-9);
    }

    #[test]
    fn glu_rejects_odd_dim() {
        let x = leaf_vec(&[1.0, 2.0, 3.0]);
        let err = glu(&x, 0).unwrap_err();
        assert!(matches!(err, FerrotorchError::InvalidArgument { .. }));
    }

    #[test]
    fn glu_2d_dim1() {
        // Shape [2, 4] split on dim 1 -> [2, 2].
        // Row 0: [1, 2 | 0, 0] -> [1*0.5, 2*0.5] = [0.5, 1.0]
        // Row 1: [3, 4 | 0, 0] -> [1.5, 2.0]
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0_f64, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0]),
            vec![2, 4],
            true,
        )
        .unwrap();
        let y = glu(&x, 1).unwrap();
        assert_eq!(y.shape(), &[2, 2]);
        let out = y.data().unwrap();
        assert!((out[0] - 0.5).abs() < 1e-9);
        assert!((out[1] - 1.0).abs() < 1e-9);
        assert!((out[2] - 1.5).abs() < 1e-9);
        assert!((out[3] - 2.0).abs() < 1e-9);
    }
}
