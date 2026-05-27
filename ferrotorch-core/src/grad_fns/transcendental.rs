//! Backward functions for transcendental (exp, log, sin, cos) and clamp
//! operations.
//!
//! Each operation has a backward struct implementing `GradFn<T>` and a public
//! function that performs the forward pass and attaches the grad_fn to the
//! result tensor when gradient tracking is enabled.
//!
//! ## REQ status (per `.design/ferrotorch-core/grad_fns/transcendental.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`exp`) | SHIPPED | `exp` + `ExpBackward` consumed by `Tensor::exp_t`, forward-AD primal, JIT interpreter, `ExpTransform`, VAE reparameterization, `PoissonNLLBackward`. |
//! | REQ-2 (`log`) | SHIPPED | `log` + `LogBackward` consumed by `Tensor::log_t`, forward-AD primal, JIT interpreter, `MultivariateNormal`, `ExpTransform` inverse, Dirichlet log-prob. |
//! | REQ-3 (`sin`) | SHIPPED | `sin` + `SinBackward` consumed by `Tensor::sin_t`, forward-AD primal, and `CosBackward`'s GPU path. |
//! | REQ-4 (`cos`) | SHIPPED | `cos` + `CosBackward` consumed by `Tensor::cos_t`, forward-AD primal, and `SinBackward`'s GPU path. |
//! | REQ-5 (`clamp`) | SHIPPED | `clamp` + `ClampBackward` (GPU fast path per #524) consumed by `Tensor::clamp_t`, `SigmoidTransform`, `BCEWithLogitsLoss`, `ReLU6`, and `Hardtanh`. |
//! | REQ-6 (`exp2`) | SHIPPED | `exp2` + `Exp2Backward` consumed by `Tensor::exp2_t`; closes #1303. |
//! | REQ-7 (`expm1`) | SHIPPED | `expm1` + `Expm1Backward` consumed by `Tensor::expm1_t`; closes #1305. |
//! | REQ-8 (`log2`) | SHIPPED | `log2` + `Log2Backward` consumed by `Tensor::log2_t`; closes #1307. |
//! | REQ-9 (`log10`) | SHIPPED | `log10` + `Log10Backward` consumed by `Tensor::log10_t`; closes #1309. |
//! | REQ-10 (`log1p`) | SHIPPED | `log1p` + `Log1pBackward` consumed by `Tensor::log1p_t`; closes #1311. |
//! | REQ-11 (`tan`) | SHIPPED | `tan` + `TanBackward` (saves output for `1 + tan^2`) consumed by `Tensor::tan_t`; closes #1313. |
//! | REQ-12 (`asin`) | SHIPPED | `asin` + `AsinBackward` consumed by `Tensor::asin_t`; closes #1315. |
//! | REQ-13 (`acos`) | SHIPPED | `acos` + `AcosBackward` consumed by `Tensor::acos_t`; closes #1316. |
//! | REQ-14 (`atan`) | SHIPPED | `atan` + `AtanBackward` consumed by `Tensor::atan_t`; closes #1317. |
//! | REQ-15 (`atan2`) | SHIPPED | `atan2` + `Atan2Backward` (joint VJP, `denom==0 → 0` mask) consumed by `lib.rs:186` re-export; closes #1318. |
//! | REQ-16 (`sinh`) | SHIPPED | `sinh` + `SinhBackward` consumed by `Tensor::sinh_t`; closes #1319. |
//! | REQ-17 (`cosh`) | SHIPPED | `cosh` + `CoshBackward` consumed by `Tensor::cosh_t`; closes #1320. |
//! | REQ-18 (`tanh` attribution) | NOT-STARTED | impl lives in `grad_fns::activation`, not this file. Route attribution drift tracked under #1321. |
//! | REQ-19 (`asinh`) | SHIPPED | `asinh` + `AsinhBackward` consumed by `Tensor::asinh_t`; closes #1322. |
//! | REQ-20 (`acosh`) | SHIPPED | `acosh` + `AcoshBackward` (real-only branch) consumed by `Tensor::acosh_t`; closes #1323. |
//! | REQ-21 (`atanh`) | SHIPPED | `atanh` + `AtanhBackward` consumed by `Tensor::atanh_t`; closes #1324. |
//! | REQ-22 (`sinc`) | SHIPPED | `sinc` + `SincBackward` (continuous extension `sinc(0) = 1`, `sinc'(0) = 0`) consumed by `Tensor::sinc_t`; closes #1325. |
//! | REQ-23 (`ceil`) | SHIPPED | `ceil` + shared `ZerosLikeBackward { name: "CeilBackward" }` consumed by `Tensor::ceil_t`; closes #1326. |
//! | REQ-24 (`floor`) | SHIPPED | `floor` + shared `ZerosLikeBackward` consumed by `Tensor::floor_t`; closes #1327. |
//! | REQ-25 (`round`) | SHIPPED | `round` + `round_half_to_even` RNE helper consumed by `Tensor::round_t`; `round.decimals` overload is a follow-up; closes #1328. |
//! | REQ-26 (`trunc`) | SHIPPED | `trunc` + shared `ZerosLikeBackward` consumed by `Tensor::trunc_t`; closes #1329. |
//! | REQ-27 (`frac`) | SHIPPED | `frac` + `FracBackward` (pass-through gradient) consumed by `Tensor::frac_t`; closes #1330. |
//! | REQ-28 (`sign`) | SHIPPED | `sign` (NaN-propagating, `sign(0) = 0`) + shared `ZerosLikeBackward` consumed by `Tensor::sign_t`; closes #1331. |
//! | REQ-29 (`signbit`) | SHIPPED | `signbit` returns `BoolTensor` via `num_traits::Float::is_sign_negative`; non-diff per upstream; consumed by `lib.rs:186` re-export; closes #1332. |
//! | REQ-30 (`clip`) | SHIPPED | `Tensor::clip_t` delegates to `clamp` per upstream's literal pass-through; closes #1333. |
//! | REQ-31 (`copysign`) | SHIPPED | `copysign` + `CopysignBackward` (grad to magnitude only, `magnitude==0 → 0` mask) consumed by `lib.rs:186` re-export; closes #1334. |
//! | REQ-32 (`nextafter`) | SHIPPED | `nextafter` + `NextafterBackward` (native-width IEEE-754 one-ULP step: `f32_one_ulp`/`f64_one_ulp`/`u16_one_ulp` per dtype — MSRV 1.85 precludes `f32::next_up`/`next_down`, stable in 1.86); VJP `self: where(self != other, grad, 0)`, `other: zeros_like` per `derivatives.yaml:1322-1324`; consumed by `lib.rs:183` re-export; closes #1335 #1556. |
//! | REQ-33 (`hypot`) | SHIPPED | `hypot` + `HypotBackward` (joint VJP via saved result, `result==0 → 0` mask) consumed by `lib.rs:186` re-export; closes #1336. |

use std::any::TypeId;
use std::sync::Arc;

use crate::autograd::no_grad::{is_grad_enabled, no_grad};
use crate::bool_tensor::BoolTensor;
use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::gpu_dispatch::gpu_backend;
use crate::grad_fns::arithmetic::reduce_grad_to_shape;
use crate::ops::elementwise::{binary_map, fast_cos, fast_sin, unary_map};
use crate::shape::broadcast_shapes;
use crate::storage::TensorStorage;
use crate::tensor::{GradFn, Tensor};

/// Returns `true` if `T` is `f32`.
#[inline]
fn is_f32<T: Float>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f32>()
}

/// Returns `true` if `T` is `f64`.
#[inline]
fn is_f64<T: Float>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f64>()
}

/// Returns `true` if `T` is `half::bf16` (#23).
#[inline]
fn is_bf16<T: Float>() -> bool {
    TypeId::of::<T>() == TypeId::of::<half::bf16>()
}

/// Returns `true` if `T` is `half::f16` (IEEE float16, crosslink #1185 Phase 1).
#[inline]
fn is_f16<T: Float>() -> bool {
    TypeId::of::<T>() == TypeId::of::<half::f16>()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Whether a single tensor requires grad (and grad is enabled).
#[inline]
fn needs_grad_unary<T: Float>(a: &Tensor<T>) -> bool {
    is_grad_enabled() && a.requires_grad()
}

// ===========================================================================
// exp
// ===========================================================================

/// Backward node for `c = exp(x)`.
///
/// VJP: `dx = grad * exp(x)`. We store the output (= exp(x)) to avoid
/// recomputing the exponential.
#[derive(Debug)]
struct ExpBackward<T: Float> {
    input: Tensor<T>,
    output: Tensor<T>,
}

impl<T: Float> GradFn<T> for ExpBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let da = if self.input.requires_grad() {
            if grad_output.is_cuda() {
                // GPU path: dx = grad * output
                Some(no_grad(|| {
                    crate::grad_fns::arithmetic::mul(grad_output, &self.output)
                })?)
            } else {
                // CPU path: direct data access for performance.
                let go_data = grad_output.data()?;
                let out_data = self.output.data()?;
                let grad_a: Vec<T> = go_data
                    .iter()
                    .zip(out_data.iter())
                    .map(|(&g, &o)| g * o)
                    .collect();
                Some(Tensor::from_storage(
                    TensorStorage::cpu(grad_a),
                    self.input.shape().to_vec(),
                    false,
                )?)
            }
        } else {
            None
        };
        Ok(vec![da])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "ExpBackward"
    }
}

/// Differentiable elementwise exponential: `c[i] = exp(x[i])`.
pub fn exp<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    crate::profiler_hook::profile_op_scope("exp", "tensor_op", &[input.shape()], || {
        exp_inner(input)
    })
}

fn exp_inner<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if input.is_cuda() && (is_f32::<T>() || is_f64::<T>() || is_bf16::<T>() || is_f16::<T>()) {
        let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        // #23: bf16 routes through `exp_bf16_bf16` (PTX ex2.approx.f32 with
        // f32 internal accumulator, bf16 RNE store-back).
        let handle: crate::gpu_dispatch::GpuBufferHandle = crate::dispatch_floating_dtype!(
            T,
            "exp",
            f32 => backend.exp_f32(input.gpu_handle()?),
            f64 => backend.exp_f64(input.gpu_handle()?),
            bf16 => backend.exp_bf16_bf16(input.gpu_handle()?),
            f16 => backend.exp_f16(input.gpu_handle()?),
        )?;
        let storage = TensorStorage::gpu(handle);
        let shape = input.shape().to_vec();

        if needs_grad_unary(input) {
            // We need the output for the backward pass (dx = grad * exp(x)).
            // Build output tensor first, then clone to attach grad_fn.
            let output = Tensor::from_storage(storage, shape.clone(), false)?;
            let grad_fn = Arc::new(ExpBackward {
                input: input.clone(),
                output: output.clone(),
            });
            let (s, sh) = output.into_storage_and_shape()?;
            Tensor::from_operation(s, sh, grad_fn)
        } else {
            Tensor::from_storage(storage, shape, false)
        }
    } else {
        let output = crate::ops::elementwise::fast_exp(input)?;

        if needs_grad_unary(input) {
            let grad_fn = Arc::new(ExpBackward {
                input: input.clone(),
                output: output.clone(),
            });
            let (storage, shape) = output.into_storage_and_shape()?;
            Tensor::from_operation(storage, shape, grad_fn)
        } else {
            Ok(output)
        }
    }
}

// ===========================================================================
// log
// ===========================================================================

/// Backward node for `c = ln(x)`.
///
/// VJP: `dx = grad / x`.
#[derive(Debug)]
struct LogBackward<T: Float> {
    input: Tensor<T>,
}

impl<T: Float> GradFn<T> for LogBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let da = if self.input.requires_grad() {
            if grad_output.is_cuda() {
                // GPU path: dx = grad / x
                Some(no_grad(|| {
                    crate::grad_fns::arithmetic::div(grad_output, &self.input)
                })?)
            } else {
                // CPU path
                let go_data = grad_output.data()?;
                let x_data = self.input.data()?;
                let grad_a: Vec<T> = go_data
                    .iter()
                    .zip(x_data.iter())
                    .map(|(&g, &x)| g / x)
                    .collect();
                Some(Tensor::from_storage(
                    TensorStorage::cpu(grad_a),
                    self.input.shape().to_vec(),
                    false,
                )?)
            }
        } else {
            None
        };
        Ok(vec![da])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "LogBackward"
    }
}

/// Differentiable elementwise natural log: `c[i] = ln(x[i])`.
pub fn log<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    crate::profiler_hook::profile_op_scope("log", "tensor_op", &[input.shape()], || {
        log_inner(input)
    })
}

fn log_inner<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if input.is_cuda() && (is_f32::<T>() || is_f64::<T>() || is_bf16::<T>() || is_f16::<T>()) {
        let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        // #23: bf16 routes through `log_bf16_bf16` (PTX lg2.approx.f32 *
        // ln(2), bf16 RNE store-back).
        let handle: crate::gpu_dispatch::GpuBufferHandle = crate::dispatch_floating_dtype!(
            T,
            "log",
            f32 => backend.log_f32(input.gpu_handle()?),
            f64 => backend.log_f64(input.gpu_handle()?),
            bf16 => backend.log_bf16_bf16(input.gpu_handle()?),
            f16 => backend.log_f16(input.gpu_handle()?),
        )?;
        let storage = TensorStorage::gpu(handle);
        let shape = input.shape().to_vec();

        if needs_grad_unary(input) {
            Tensor::from_operation(
                storage,
                shape,
                Arc::new(LogBackward {
                    input: input.clone(),
                }),
            )
        } else {
            Tensor::from_storage(storage, shape, false)
        }
    } else {
        let output = crate::ops::elementwise::fast_log(input)?;

        if needs_grad_unary(input) {
            let (storage, shape) = output.into_storage_and_shape()?;
            Tensor::from_operation(
                storage,
                shape,
                Arc::new(LogBackward {
                    input: input.clone(),
                }),
            )
        } else {
            Ok(output)
        }
    }
}

// ===========================================================================
// sin
// ===========================================================================

/// Backward node for `c = sin(x)`.
///
/// VJP: `dx = grad * cos(x)`.
#[derive(Debug)]
struct SinBackward<T: Float> {
    input: Tensor<T>,
}

impl<T: Float> GradFn<T> for SinBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let da = if self.input.requires_grad() {
            if grad_output.is_cuda() {
                // GPU path: dx = grad * cos(x)
                let da = no_grad(|| {
                    let cos_x = cos(&self.input)?;
                    crate::grad_fns::arithmetic::mul(grad_output, &cos_x)
                })?;
                Some(da)
            } else {
                // CPU path
                let go_data = grad_output.data()?;
                let x_data = self.input.data()?;
                let grad_a: Vec<T> = go_data
                    .iter()
                    .zip(x_data.iter())
                    .map(|(&g, &x)| g * x.cos())
                    .collect();
                Some(Tensor::from_storage(
                    TensorStorage::cpu(grad_a),
                    self.input.shape().to_vec(),
                    false,
                )?)
            }
        } else {
            None
        };
        Ok(vec![da])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "SinBackward"
    }
}

/// Differentiable elementwise sine: `c[i] = sin(x[i])`.
pub fn sin<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    crate::profiler_hook::profile_op_scope("sin", "tensor_op", &[input.shape()], || {
        sin_inner(input)
    })
}

fn sin_inner<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let output = fast_sin(input)?;

    if needs_grad_unary(input) {
        let (storage, shape) = output.into_storage_and_shape()?;
        Tensor::from_operation(
            storage,
            shape,
            Arc::new(SinBackward {
                input: input.clone(),
            }),
        )
    } else {
        Ok(output)
    }
}

// ===========================================================================
// cos
// ===========================================================================

/// Backward node for `c = cos(x)`.
///
/// VJP: `dx = grad * (-sin(x))`.
#[derive(Debug)]
struct CosBackward<T: Float> {
    input: Tensor<T>,
}

impl<T: Float> GradFn<T> for CosBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let da = if self.input.requires_grad() {
            if grad_output.is_cuda() {
                // GPU path: dx = grad * (-sin(x))
                let da = no_grad(|| {
                    let sin_x = sin(&self.input)?;
                    let neg_sin = crate::grad_fns::arithmetic::neg(&sin_x)?;
                    crate::grad_fns::arithmetic::mul(grad_output, &neg_sin)
                })?;
                Some(da)
            } else {
                // CPU path
                let go_data = grad_output.data()?;
                let x_data = self.input.data()?;
                let grad_a: Vec<T> = go_data
                    .iter()
                    .zip(x_data.iter())
                    .map(|(&g, &x)| g * (-x.sin()))
                    .collect();
                Some(Tensor::from_storage(
                    TensorStorage::cpu(grad_a),
                    self.input.shape().to_vec(),
                    false,
                )?)
            }
        } else {
            None
        };
        Ok(vec![da])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "CosBackward"
    }
}

/// Differentiable elementwise cosine: `c[i] = cos(x[i])`.
pub fn cos<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    crate::profiler_hook::profile_op_scope("cos", "tensor_op", &[input.shape()], || {
        cos_inner(input)
    })
}

fn cos_inner<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let output = fast_cos(input)?;

    if needs_grad_unary(input) {
        let (storage, shape) = output.into_storage_and_shape()?;
        Tensor::from_operation(
            storage,
            shape,
            Arc::new(CosBackward {
                input: input.clone(),
            }),
        )
    } else {
        Ok(output)
    }
}

// ===========================================================================
// tanh (delegated)
// ===========================================================================

// tanh and sigmoid are implemented in `grad_fns::activation` since they are
// also activation functions. Re-exporting here for discoverability:
//
//   use crate::grad_fns::activation::{tanh, sigmoid};

// ===========================================================================
// clamp
// ===========================================================================

/// Backward node for `c = clamp(x, min, max)`.
///
/// VJP: `dx[i] = grad[i]` if `min <= x[i] <= max`, else `0`.
#[derive(Debug)]
struct ClampBackward<T: Float> {
    input: Tensor<T>,
    min: T,
    max: T,
}

impl<T: Float> GradFn<T> for ClampBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let da = if self.input.requires_grad() {
            // GPU-native path for f32/f64 (#524).
            if grad_output.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
                let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
                let result_h = if is_f32::<T>() {
                    let min_f = self.min.to_f64().unwrap_or(f64::NEG_INFINITY) as f32;
                    let max_f = self.max.to_f64().unwrap_or(f64::INFINITY) as f32;
                    backend.clamp_backward_f32(
                        grad_output.gpu_handle()?,
                        self.input.gpu_handle()?,
                        min_f,
                        max_f,
                    )?
                } else {
                    let min_f = self.min.to_f64().unwrap_or(f64::NEG_INFINITY);
                    let max_f = self.max.to_f64().unwrap_or(f64::INFINITY);
                    backend.clamp_backward_f64(
                        grad_output.gpu_handle()?,
                        self.input.gpu_handle()?,
                        min_f,
                        max_f,
                    )?
                };
                Some(Tensor::from_storage(
                    TensorStorage::gpu(result_h),
                    self.input.shape().to_vec(),
                    false,
                )?)
            } else if grad_output.is_cuda() || self.input.is_cuda() {
                return Err(FerrotorchError::NotImplementedOnCuda {
                    op: "ClampBackward",
                });
            } else {
                // CPU path
                let go_data = grad_output.data()?;
                let x_data = self.input.data()?;
                let zero = <T as num_traits::Zero>::zero();
                let grad_a: Vec<T> = go_data
                    .iter()
                    .zip(x_data.iter())
                    .map(|(&g, &x)| {
                        if x >= self.min && x <= self.max {
                            g
                        } else {
                            zero
                        }
                    })
                    .collect();
                Some(Tensor::from_storage(
                    TensorStorage::cpu(grad_a),
                    self.input.shape().to_vec(),
                    false,
                )?)
            }
        } else {
            None
        };
        Ok(vec![da])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "ClampBackward"
    }
}

/// Differentiable elementwise clamp: `c[i] = x[i].clamp(min, max)`.
///
/// Gradient flows through only where `min <= x[i] <= max`; it is zero at
/// the boundaries where the value was clamped.
pub fn clamp<T: Float>(input: &Tensor<T>, min: T, max: T) -> FerrotorchResult<Tensor<T>> {
    // GPU fast path for f32/f64
    if input.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
        if let Some(backend) = crate::gpu_dispatch::gpu_backend() {
            let handle = if is_f32::<T>() {
                let min_f32 = min.to_f32().unwrap_or(f32::MIN);
                let max_f32 = max.to_f32().unwrap_or(f32::MAX);
                backend.clamp_f32(input.gpu_handle()?, min_f32, max_f32)?
            } else {
                let min_f64 = min.to_f64().unwrap_or(f64::MIN);
                let max_f64 = max.to_f64().unwrap_or(f64::MAX);
                backend.clamp_f64(input.gpu_handle()?, min_f64, max_f64)?
            };
            return if needs_grad_unary(input) {
                Tensor::from_operation(
                    TensorStorage::gpu(handle),
                    input.shape().to_vec(),
                    Arc::new(ClampBackward {
                        input: input.clone(),
                        min,
                        max,
                    }),
                )
            } else {
                Tensor::from_storage(TensorStorage::gpu(handle), input.shape().to_vec(), false)
            };
        }
    }

    // CPU path
    let output = unary_map(input, |x| {
        if x < min {
            min
        } else if x > max {
            max
        } else {
            x
        }
    })?;

    if needs_grad_unary(input) {
        let (storage, shape) = output.into_storage_and_shape()?;
        Tensor::from_operation(
            storage,
            shape,
            Arc::new(ClampBackward {
                input: input.clone(),
                min,
                max,
            }),
        )
    } else {
        Ok(output)
    }
}

// ===========================================================================
// Helpers for new unary ops
// ===========================================================================

/// Build a tensor of zeros with the same shape as `like`. Used by the
/// backward of piecewise-constant ops (ceil/floor/round/trunc/sign per
/// `tools/autograd/derivatives.yaml` `... self: zeros_like(grad)` entries).
fn zeros_like_tensor<T: Float>(like: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let zero = <T as num_traits::Zero>::zero();
    let n: usize = like.shape().iter().product::<usize>().max(1);
    Tensor::from_storage(
        TensorStorage::cpu(vec![zero; n]),
        like.shape().to_vec(),
        false,
    )
}

/// Helper to attach a grad_fn to a forward result when grad-tracking is
/// enabled. Mirrors the pattern in `sin_inner` / `cos_inner` for the trivial
/// path (no GPU-layer kernel; the forward routes through `unary_map`).
#[inline]
fn finish_unary<T: Float, G: GradFn<T> + 'static>(
    output: Tensor<T>,
    input: &Tensor<T>,
    make_grad: impl FnOnce() -> G,
) -> FerrotorchResult<Tensor<T>> {
    if needs_grad_unary(input) {
        let (storage, shape) = output.into_storage_and_shape()?;
        Tensor::from_operation(storage, shape, Arc::new(make_grad()))
    } else {
        Ok(output)
    }
}

// ===========================================================================
// tan
// ===========================================================================

/// Backward node for `c = tan(x)`.
///
/// VJP: `dx = grad * (1 + tan(x)^2)` per `tools/autograd/derivatives.yaml`
/// `tan: grad * (1 + result.pow(2)).conj()`. We save the OUTPUT (`tan(x)`)
/// to avoid recomputing the transcendental on backward.
#[derive(Debug)]
struct TanBackward<T: Float> {
    input: Tensor<T>,
    output: Tensor<T>,
}

impl<T: Float> GradFn<T> for TanBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let da = if self.input.requires_grad() {
            let go = grad_output.data()?;
            let o = self.output.data()?;
            let one = <T as num_traits::One>::one();
            let g: Vec<T> = go
                .iter()
                .zip(o.iter())
                .map(|(&g, &t)| g * (one + t * t))
                .collect();
            Some(Tensor::from_storage(
                TensorStorage::cpu(g),
                self.input.shape().to_vec(),
                false,
            )?)
        } else {
            None
        };
        Ok(vec![da])
    }
    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }
    fn name(&self) -> &'static str {
        "TanBackward"
    }
}

/// Differentiable elementwise tangent: `c[i] = tan(x[i])`. Mirrors
/// `aten/src/ATen/native/UnaryOps.cpp:360 CREATE_UNARY_TORCH_IMPL_FUNC(tan_out, tan_stub)`.
pub fn tan<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let output = unary_map(input, |x| x.tan())?;
    if needs_grad_unary(input) {
        let (storage, shape) = output.into_storage_and_shape()?;
        let out_tensor = Tensor::from_storage(storage, shape, false)?;
        let grad_fn = Arc::new(TanBackward {
            input: input.clone(),
            output: out_tensor.clone(),
        });
        let (s, sh) = out_tensor.into_storage_and_shape()?;
        Tensor::from_operation(s, sh, grad_fn)
    } else {
        Ok(output)
    }
}

// ===========================================================================
// asin / acos / atan
// ===========================================================================

/// Backward node for `c = asin(x)`. VJP: `dx = grad / sqrt(1 - x^2)`
/// per `derivatives.yaml` `asin: grad * (-self * self + 1).rsqrt().conj()`.
#[derive(Debug)]
struct AsinBackward<T: Float> {
    input: Tensor<T>,
}

impl<T: Float> GradFn<T> for AsinBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let da = if self.input.requires_grad() {
            let go = grad_output.data()?;
            let x = self.input.data()?;
            let one = <T as num_traits::One>::one();
            let g: Vec<T> = go
                .iter()
                .zip(x.iter())
                .map(|(&g, &x)| g / (one - x * x).sqrt())
                .collect();
            Some(Tensor::from_storage(
                TensorStorage::cpu(g),
                self.input.shape().to_vec(),
                false,
            )?)
        } else {
            None
        };
        Ok(vec![da])
    }
    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }
    fn name(&self) -> &'static str {
        "AsinBackward"
    }
}

/// Differentiable elementwise arcsine. Mirrors
/// `UnaryOps.cpp:323 CREATE_UNARY_TORCH_IMPL_FUNC(asin_out, asin_stub)`.
pub fn asin<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let output = unary_map(input, |x| x.asin())?;
    finish_unary(output, input, || AsinBackward {
        input: input.clone(),
    })
}

/// Backward node for `c = acos(x)`. VJP: `dx = -grad / sqrt(1 - x^2)`
/// per `derivatives.yaml` `acos: grad * -((-self * self + 1).rsqrt()).conj()`.
#[derive(Debug)]
struct AcosBackward<T: Float> {
    input: Tensor<T>,
}

impl<T: Float> GradFn<T> for AcosBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let da = if self.input.requires_grad() {
            let go = grad_output.data()?;
            let x = self.input.data()?;
            let one = <T as num_traits::One>::one();
            let g: Vec<T> = go
                .iter()
                .zip(x.iter())
                .map(|(&g, &x)| -(g / (one - x * x).sqrt()))
                .collect();
            Some(Tensor::from_storage(
                TensorStorage::cpu(g),
                self.input.shape().to_vec(),
                false,
            )?)
        } else {
            None
        };
        Ok(vec![da])
    }
    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }
    fn name(&self) -> &'static str {
        "AcosBackward"
    }
}

/// Differentiable elementwise arccosine. Mirrors
/// `UnaryOps.cpp:321 CREATE_UNARY_TORCH_IMPL_FUNC(acos_out, acos_stub)`.
pub fn acos<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let output = unary_map(input, |x| x.acos())?;
    finish_unary(output, input, || AcosBackward {
        input: input.clone(),
    })
}

/// Backward node for `c = atan(x)`. VJP: `dx = grad / (1 + x^2)`
/// per `derivatives.yaml` `atan: grad / (self * self + 1).conj()`.
#[derive(Debug)]
struct AtanBackward<T: Float> {
    input: Tensor<T>,
}

impl<T: Float> GradFn<T> for AtanBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let da = if self.input.requires_grad() {
            let go = grad_output.data()?;
            let x = self.input.data()?;
            let one = <T as num_traits::One>::one();
            let g: Vec<T> = go
                .iter()
                .zip(x.iter())
                .map(|(&g, &x)| g / (one + x * x))
                .collect();
            Some(Tensor::from_storage(
                TensorStorage::cpu(g),
                self.input.shape().to_vec(),
                false,
            )?)
        } else {
            None
        };
        Ok(vec![da])
    }
    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }
    fn name(&self) -> &'static str {
        "AtanBackward"
    }
}

/// Differentiable elementwise arctangent. Mirrors
/// `UnaryOps.cpp:325 CREATE_UNARY_TORCH_IMPL_FUNC(atan_out, atan_stub)`.
pub fn atan<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let output = unary_map(input, |x| x.atan())?;
    finish_unary(output, input, || AtanBackward {
        input: input.clone(),
    })
}

// ===========================================================================
// sinh / cosh
// ===========================================================================

/// Backward node for `c = sinh(x)`. VJP: `dx = grad * cosh(x)` per
/// `derivatives.yaml` `sinh: grad * self.cosh().conj()`.
#[derive(Debug)]
struct SinhBackward<T: Float> {
    input: Tensor<T>,
}

impl<T: Float> GradFn<T> for SinhBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let da = if self.input.requires_grad() {
            let go = grad_output.data()?;
            let x = self.input.data()?;
            let g: Vec<T> = go
                .iter()
                .zip(x.iter())
                .map(|(&g, &x)| g * x.cosh())
                .collect();
            Some(Tensor::from_storage(
                TensorStorage::cpu(g),
                self.input.shape().to_vec(),
                false,
            )?)
        } else {
            None
        };
        Ok(vec![da])
    }
    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }
    fn name(&self) -> &'static str {
        "SinhBackward"
    }
}

/// Differentiable elementwise hyperbolic sine. Mirrors
/// `UnaryOps.cpp:351 CREATE_UNARY_TORCH_IMPL_FUNC(sinh_out, sinh_stub)`.
pub fn sinh<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let output = unary_map(input, |x| x.sinh())?;
    finish_unary(output, input, || SinhBackward {
        input: input.clone(),
    })
}

/// Backward node for `c = cosh(x)`. VJP: `dx = grad * sinh(x)` per
/// `derivatives.yaml` `cosh: grad * self.sinh().conj()`.
#[derive(Debug)]
struct CoshBackward<T: Float> {
    input: Tensor<T>,
}

impl<T: Float> GradFn<T> for CoshBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let da = if self.input.requires_grad() {
            let go = grad_output.data()?;
            let x = self.input.data()?;
            let g: Vec<T> = go
                .iter()
                .zip(x.iter())
                .map(|(&g, &x)| g * x.sinh())
                .collect();
            Some(Tensor::from_storage(
                TensorStorage::cpu(g),
                self.input.shape().to_vec(),
                false,
            )?)
        } else {
            None
        };
        Ok(vec![da])
    }
    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }
    fn name(&self) -> &'static str {
        "CoshBackward"
    }
}

/// Differentiable elementwise hyperbolic cosine. Mirrors
/// `UnaryOps.cpp:329 CREATE_UNARY_TORCH_IMPL_FUNC(cosh_out, cosh_stub)`.
pub fn cosh<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let output = unary_map(input, |x| x.cosh())?;
    finish_unary(output, input, || CoshBackward {
        input: input.clone(),
    })
}

// ===========================================================================
// asinh / acosh / atanh
// ===========================================================================

/// Backward node for `c = asinh(x)`. VJP: `dx = grad / sqrt(x^2 + 1)` per
/// `derivatives.yaml` `asinh: grad * (self.pow(2) + 1).rsqrt().conj()`.
#[derive(Debug)]
struct AsinhBackward<T: Float> {
    input: Tensor<T>,
}

impl<T: Float> GradFn<T> for AsinhBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let da = if self.input.requires_grad() {
            let go = grad_output.data()?;
            let x = self.input.data()?;
            let one = <T as num_traits::One>::one();
            let g: Vec<T> = go
                .iter()
                .zip(x.iter())
                .map(|(&g, &x)| g / (x * x + one).sqrt())
                .collect();
            Some(Tensor::from_storage(
                TensorStorage::cpu(g),
                self.input.shape().to_vec(),
                false,
            )?)
        } else {
            None
        };
        Ok(vec![da])
    }
    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }
    fn name(&self) -> &'static str {
        "AsinhBackward"
    }
}

/// Differentiable elementwise inverse hyperbolic sine. Mirrors
/// `UnaryOps.cpp:324 CREATE_UNARY_TORCH_IMPL_FUNC(asinh_out, asinh_stub)`.
pub fn asinh<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let output = unary_map(input, |x| x.asinh())?;
    finish_unary(output, input, || AsinhBackward {
        input: input.clone(),
    })
}

/// Backward node for `c = acosh(x)`. VJP: `dx = grad / sqrt(x^2 - 1)` per
/// `derivatives.yaml` `acosh: grad * (self * self - 1).rsqrt()` (real-only).
#[derive(Debug)]
struct AcoshBackward<T: Float> {
    input: Tensor<T>,
}

impl<T: Float> GradFn<T> for AcoshBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let da = if self.input.requires_grad() {
            let go = grad_output.data()?;
            let x = self.input.data()?;
            let one = <T as num_traits::One>::one();
            let g: Vec<T> = go
                .iter()
                .zip(x.iter())
                .map(|(&g, &x)| g / (x * x - one).sqrt())
                .collect();
            Some(Tensor::from_storage(
                TensorStorage::cpu(g),
                self.input.shape().to_vec(),
                false,
            )?)
        } else {
            None
        };
        Ok(vec![da])
    }
    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }
    fn name(&self) -> &'static str {
        "AcoshBackward"
    }
}

/// Differentiable elementwise inverse hyperbolic cosine. Mirrors
/// `UnaryOps.cpp:322 CREATE_UNARY_TORCH_IMPL_FUNC(acosh_out, acosh_stub)`.
pub fn acosh<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let output = unary_map(input, |x| x.acosh())?;
    finish_unary(output, input, || AcoshBackward {
        input: input.clone(),
    })
}

/// Backward node for `c = atanh(x)`. VJP: `dx = grad / (1 - x^2)` per
/// `derivatives.yaml` `atanh: grad * 1 / (1 - self.pow(2)).conj()`.
#[derive(Debug)]
struct AtanhBackward<T: Float> {
    input: Tensor<T>,
}

impl<T: Float> GradFn<T> for AtanhBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let da = if self.input.requires_grad() {
            let go = grad_output.data()?;
            let x = self.input.data()?;
            let one = <T as num_traits::One>::one();
            let g: Vec<T> = go
                .iter()
                .zip(x.iter())
                .map(|(&g, &x)| g / (one - x * x))
                .collect();
            Some(Tensor::from_storage(
                TensorStorage::cpu(g),
                self.input.shape().to_vec(),
                false,
            )?)
        } else {
            None
        };
        Ok(vec![da])
    }
    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }
    fn name(&self) -> &'static str {
        "AtanhBackward"
    }
}

/// Differentiable elementwise inverse hyperbolic tangent. Mirrors
/// `UnaryOps.cpp:326 CREATE_UNARY_TORCH_IMPL_FUNC(atanh_out, atanh_stub)`.
pub fn atanh<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let output = unary_map(input, |x| x.atanh())?;
    finish_unary(output, input, || AtanhBackward {
        input: input.clone(),
    })
}

// ===========================================================================
// exp2 / expm1
// ===========================================================================

/// Backward node for `c = 2^x`. VJP: `dx = grad * result * ln(2)` per
/// `derivatives.yaml` `exp2: grad * result.conj() * M_LN2`. Saves the
/// output to avoid recomputing the exponential.
#[derive(Debug)]
struct Exp2Backward<T: Float> {
    input: Tensor<T>,
    output: Tensor<T>,
}

impl<T: Float> GradFn<T> for Exp2Backward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let da = if self.input.requires_grad() {
            let go = grad_output.data()?;
            let o = self.output.data()?;
            // M_LN2 = ln(2). Convert at runtime so we honor the generic T.
            let ln2 = T::from(std::f64::consts::LN_2).unwrap_or_else(<T as num_traits::Zero>::zero);
            let g: Vec<T> = go
                .iter()
                .zip(o.iter())
                .map(|(&g, &r)| g * r * ln2)
                .collect();
            Some(Tensor::from_storage(
                TensorStorage::cpu(g),
                self.input.shape().to_vec(),
                false,
            )?)
        } else {
            None
        };
        Ok(vec![da])
    }
    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }
    fn name(&self) -> &'static str {
        "Exp2Backward"
    }
}

/// Differentiable elementwise base-2 exponential: `c[i] = 2^x[i]`. Mirrors
/// `UnaryOps.cpp:335 CREATE_UNARY_TORCH_IMPL_FUNC(exp2_out, exp2_stub)`.
pub fn exp2<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let output = unary_map(input, |x| x.exp2())?;
    if needs_grad_unary(input) {
        let (storage, shape) = output.into_storage_and_shape()?;
        let out_tensor = Tensor::from_storage(storage, shape, false)?;
        let grad_fn = Arc::new(Exp2Backward {
            input: input.clone(),
            output: out_tensor.clone(),
        });
        let (s, sh) = out_tensor.into_storage_and_shape()?;
        Tensor::from_operation(s, sh, grad_fn)
    } else {
        Ok(output)
    }
}

/// Backward node for `c = exp(x) - 1`. VJP: `dx = grad * (result + 1)` per
/// `derivatives.yaml` `expm1: grad * (result.conj() + 1)`.
#[derive(Debug)]
struct Expm1Backward<T: Float> {
    input: Tensor<T>,
    output: Tensor<T>,
}

impl<T: Float> GradFn<T> for Expm1Backward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let da = if self.input.requires_grad() {
            let go = grad_output.data()?;
            let o = self.output.data()?;
            let one = <T as num_traits::One>::one();
            let g: Vec<T> = go
                .iter()
                .zip(o.iter())
                .map(|(&g, &r)| g * (r + one))
                .collect();
            Some(Tensor::from_storage(
                TensorStorage::cpu(g),
                self.input.shape().to_vec(),
                false,
            )?)
        } else {
            None
        };
        Ok(vec![da])
    }
    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }
    fn name(&self) -> &'static str {
        "Expm1Backward"
    }
}

/// Differentiable elementwise `exp(x) - 1` numerically stable for small `x`.
/// Mirrors `UnaryOps.cpp:336 CREATE_UNARY_TORCH_IMPL_FUNC(expm1_out, expm1_stub)`.
pub fn expm1<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let output = unary_map(input, |x| x.exp_m1())?;
    if needs_grad_unary(input) {
        let (storage, shape) = output.into_storage_and_shape()?;
        let out_tensor = Tensor::from_storage(storage, shape, false)?;
        let grad_fn = Arc::new(Expm1Backward {
            input: input.clone(),
            output: out_tensor.clone(),
        });
        let (s, sh) = out_tensor.into_storage_and_shape()?;
        Tensor::from_operation(s, sh, grad_fn)
    } else {
        Ok(output)
    }
}

// ===========================================================================
// log2 / log10 / log1p
// ===========================================================================

/// Backward node for `c = log_2(x)`. VJP: `dx = grad / (x * ln(2))` per
/// `derivatives.yaml` `log2: grad / (self.conj() * 0.6931471805599453)`.
#[derive(Debug)]
struct Log2Backward<T: Float> {
    input: Tensor<T>,
}

impl<T: Float> GradFn<T> for Log2Backward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let da = if self.input.requires_grad() {
            let go = grad_output.data()?;
            let x = self.input.data()?;
            let ln2 = T::from(std::f64::consts::LN_2).unwrap_or_else(<T as num_traits::Zero>::zero);
            let g: Vec<T> = go
                .iter()
                .zip(x.iter())
                .map(|(&g, &x)| g / (x * ln2))
                .collect();
            Some(Tensor::from_storage(
                TensorStorage::cpu(g),
                self.input.shape().to_vec(),
                false,
            )?)
        } else {
            None
        };
        Ok(vec![da])
    }
    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }
    fn name(&self) -> &'static str {
        "Log2Backward"
    }
}

/// Differentiable elementwise base-2 logarithm. Mirrors
/// `UnaryOps.cpp:343 CREATE_UNARY_TORCH_IMPL_FUNC(log2_out, log2_stub)`.
pub fn log2<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let output = unary_map(input, |x| x.log2())?;
    finish_unary(output, input, || Log2Backward {
        input: input.clone(),
    })
}

/// Backward node for `c = log_10(x)`. VJP: `dx = grad / (x * ln(10))` per
/// `derivatives.yaml` `log10: grad / (self.conj() * 2.3025850929940456)`.
#[derive(Debug)]
struct Log10Backward<T: Float> {
    input: Tensor<T>,
}

impl<T: Float> GradFn<T> for Log10Backward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let da = if self.input.requires_grad() {
            let go = grad_output.data()?;
            let x = self.input.data()?;
            let ln10 =
                T::from(std::f64::consts::LN_10).unwrap_or_else(<T as num_traits::Zero>::zero);
            let g: Vec<T> = go
                .iter()
                .zip(x.iter())
                .map(|(&g, &x)| g / (x * ln10))
                .collect();
            Some(Tensor::from_storage(
                TensorStorage::cpu(g),
                self.input.shape().to_vec(),
                false,
            )?)
        } else {
            None
        };
        Ok(vec![da])
    }
    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }
    fn name(&self) -> &'static str {
        "Log10Backward"
    }
}

/// Differentiable elementwise base-10 logarithm. Mirrors
/// `UnaryOps.cpp:341 CREATE_UNARY_TORCH_IMPL_FUNC(log10_out, log10_stub)`.
pub fn log10<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let output = unary_map(input, |x| x.log10())?;
    finish_unary(output, input, || Log10Backward {
        input: input.clone(),
    })
}

/// Backward node for `c = ln(1 + x)`. VJP: `dx = grad / (1 + x)` per
/// `derivatives.yaml` `log1p: log1p_backward(grad, self)`.
#[derive(Debug)]
struct Log1pBackward<T: Float> {
    input: Tensor<T>,
}

impl<T: Float> GradFn<T> for Log1pBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let da = if self.input.requires_grad() {
            let go = grad_output.data()?;
            let x = self.input.data()?;
            let one = <T as num_traits::One>::one();
            let g: Vec<T> = go
                .iter()
                .zip(x.iter())
                .map(|(&g, &x)| g / (one + x))
                .collect();
            Some(Tensor::from_storage(
                TensorStorage::cpu(g),
                self.input.shape().to_vec(),
                false,
            )?)
        } else {
            None
        };
        Ok(vec![da])
    }
    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }
    fn name(&self) -> &'static str {
        "Log1pBackward"
    }
}

/// Differentiable elementwise `ln(1 + x)` numerically stable for small `x`.
/// Mirrors `UnaryOps.cpp:342 CREATE_UNARY_TORCH_IMPL_FUNC(log1p_out, log1p_stub)`.
pub fn log1p<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let output = unary_map(input, |x| x.ln_1p())?;
    finish_unary(output, input, || Log1pBackward {
        input: input.clone(),
    })
}

// ===========================================================================
// Rounding family (ceil / floor / round / trunc) — zero-gradient backward
// ===========================================================================

/// Generic backward producing `zeros_like(grad)`. Used for piecewise-constant
/// ops (`ceil`/`floor`/`round`/`trunc`/`sign`) per `derivatives.yaml`
/// `... self: zeros_like(grad)`. Carries `input` so `inputs()` reports the
/// correct topological dependency on the autograd graph.
#[derive(Debug)]
struct ZerosLikeBackward<T: Float> {
    input: Tensor<T>,
    name: &'static str,
}

impl<T: Float> GradFn<T> for ZerosLikeBackward<T> {
    fn backward(&self, _grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let da = if self.input.requires_grad() {
            Some(zeros_like_tensor(&self.input)?)
        } else {
            None
        };
        Ok(vec![da])
    }
    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }
    fn name(&self) -> &'static str {
        self.name
    }
}

/// Differentiable elementwise ceiling. Mirrors
/// `UnaryOps.cpp:316 CREATE_UNARY_TORCH_IMPL_INTEGER_NO_OP_FUNC(ceil_out, ceil_stub)`.
/// Backward: `zeros_like(grad)` (gradient is zero a.e.).
pub fn ceil<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let output = unary_map(input, |x| x.ceil())?;
    finish_unary(output, input, || ZerosLikeBackward {
        input: input.clone(),
        name: "CeilBackward",
    })
}

/// Differentiable elementwise floor. Mirrors
/// `UnaryOps.cpp:317 CREATE_UNARY_TORCH_IMPL_INTEGER_NO_OP_FUNC(floor_out, floor_stub)`.
pub fn floor<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let output = unary_map(input, |x| x.floor())?;
    finish_unary(output, input, || ZerosLikeBackward {
        input: input.clone(),
        name: "FloorBackward",
    })
}

/// Differentiable elementwise round (round-half-to-even / banker's rounding,
/// matching PyTorch's `nearbyint`-based kernel). Mirrors
/// `UnaryOps.cpp:318 CREATE_UNARY_TORCH_IMPL_INTEGER_NO_OP_FUNC(round_out, round_stub)`.
///
/// `num_traits::Float::round` does round-half-away-from-zero. To match
/// upstream's round-half-to-even (RNE) we manually inspect the half-case
/// and break ties toward even.
pub fn round<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let output = unary_map(input, round_half_to_even)?;
    finish_unary(output, input, || ZerosLikeBackward {
        input: input.clone(),
        name: "RoundBackward",
    })
}

/// Round-half-to-even (banker's rounding). Matches the IEEE-754
/// `roundToIntegralTiesToEven` mode that PyTorch's `round_kernel` invokes
/// via `nearbyint` (rounding mode set to FE_TONEAREST, the C default).
#[inline]
fn round_half_to_even<T: Float>(x: T) -> T {
    let two = T::from(2.0).unwrap_or_else(<T as num_traits::One>::one);
    let half = T::from(0.5).unwrap_or_else(<T as num_traits::Zero>::zero);
    let one = <T as num_traits::One>::one();
    let f = x.floor();
    let diff = x - f;
    if diff < half {
        f
    } else if diff > half {
        f + one
    } else {
        // Tie: prefer the even neighbor. f even -> stay; odd -> +1.
        // We test parity via `f - 2 * floor(f/2)`.
        let half_f = (f / two).floor();
        if f - half_f * two == <T as num_traits::Zero>::zero() {
            f
        } else {
            f + one
        }
    }
}

/// Differentiable elementwise truncation (round toward zero). Mirrors
/// `UnaryOps.cpp:319 CREATE_UNARY_TORCH_IMPL_INTEGER_NO_OP_FUNC(trunc_out, trunc_stub)`.
pub fn trunc<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let output = unary_map(input, |x| x.trunc())?;
    finish_unary(output, input, || ZerosLikeBackward {
        input: input.clone(),
        name: "TruncBackward",
    })
}

// ===========================================================================
// frac — VJP: gradient passes through (slope 1 a.e.)
// ===========================================================================

/// Backward node for `c = x - trunc(x)`. Slope is 1 a.e., so the gradient
/// passes through per `derivatives.yaml` `frac: grad`.
#[derive(Debug)]
struct FracBackward<T: Float> {
    input: Tensor<T>,
}

impl<T: Float> GradFn<T> for FracBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let da = if self.input.requires_grad() {
            // Clone the grad: same shape & values as upstream.
            let go = grad_output.data()?;
            Some(Tensor::from_storage(
                TensorStorage::cpu(go.to_vec()),
                self.input.shape().to_vec(),
                false,
            )?)
        } else {
            None
        };
        Ok(vec![da])
    }
    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }
    fn name(&self) -> &'static str {
        "FracBackward"
    }
}

/// Differentiable elementwise fractional part: `c[i] = x[i] - trunc(x[i])`.
/// Mirrors `UnaryOps.cpp:337 CREATE_UNARY_TORCH_IMPL_FUNC(frac_out, frac_stub)`.
pub fn frac<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let output = unary_map(input, |x| x - x.trunc())?;
    finish_unary(output, input, || FracBackward {
        input: input.clone(),
    })
}

// ===========================================================================
// sign — zero-gradient backward
// ===========================================================================

/// Differentiable elementwise sign: `c = -1 / 0 / +1` per
/// `UnaryOps.cpp:348 CREATE_UNARY_TORCH_IMPL_FUNC(sign_out, sign_stub)`.
/// Backward: `zeros_like(grad)` (piecewise constant).
///
/// Special: `sign(NaN) = 0`, matching the upstream CPU kernel at
/// `aten/src/ATen/native/cpu/UnaryOpsKernel.cpp:304`:
///   `[=](scalar_t a) -> scalar_t { return (0 < a) - c10::is_negative(a); }`
/// which evaluates to `0` for `a = NaN` because both `0 < NaN` and
/// `c10::is_negative(NaN)` (sign-bit of a quieted NaN) are false. This is
/// `torch.sign`, not `torch.sgn`; `sgn` (a complex-aware variant) DOES
/// propagate NaN, but the two ops are deliberately distinct upstream.
pub fn sign<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let zero = <T as num_traits::Zero>::zero();
    let output = unary_map(input, |x| {
        if x.is_nan() || x == zero {
            zero
        } else {
            x.signum()
        }
    })?;
    finish_unary(output, input, || ZerosLikeBackward {
        input: input.clone(),
        name: "SignBackward",
    })
}

// ===========================================================================
// sinc — c[i] = sin(pi*x[i]) / (pi*x[i]) with sinc(0) = 1
// ===========================================================================

/// Backward node for `c = sinc(x)`. The closed-form derivative is
/// `sinc'(x) = (pi*x*cos(pi*x) - sin(pi*x)) / (pi*x^2)` for `x != 0`, and
/// `sinc'(0) = 0` (the function has a local maximum at the origin). Matches
/// upstream `tools/autograd/derivatives.yaml` `sinc: sinc_backward(grad, self)`
/// whose C++ implementation in `torch/csrc/autograd/FunctionsManual.cpp`
/// uses the same closed form.
#[derive(Debug)]
struct SincBackward<T: Float> {
    input: Tensor<T>,
}

impl<T: Float> GradFn<T> for SincBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let da = if self.input.requires_grad() {
            let go = grad_output.data()?;
            let x = self.input.data()?;
            let pi = T::from(std::f64::consts::PI).unwrap_or_else(<T as num_traits::Zero>::zero);
            let zero = <T as num_traits::Zero>::zero();
            let g: Vec<T> = go
                .iter()
                .zip(x.iter())
                .map(|(&g, &x)| {
                    if x == zero {
                        zero
                    } else {
                        let px = pi * x;
                        g * (px.cos() / x - px.sin() / (px * x))
                    }
                })
                .collect();
            Some(Tensor::from_storage(
                TensorStorage::cpu(g),
                self.input.shape().to_vec(),
                false,
            )?)
        } else {
            None
        };
        Ok(vec![da])
    }
    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }
    fn name(&self) -> &'static str {
        "SincBackward"
    }
}

/// Differentiable elementwise normalized sinc: `c[i] = sin(pi*x[i]) /
/// (pi*x[i])`, with `sinc(0) = 1` by continuous extension. Mirrors
/// `UnaryOps.cpp:350 CREATE_UNARY_TORCH_IMPL_FUNC(sinc_out, sinc_stub)`
/// (kernel at `aten/src/ATen/native/cpu/UnaryOpsKernel.cpp` `sinc_kernel`).
pub fn sinc<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let pi = T::from(std::f64::consts::PI).unwrap_or_else(<T as num_traits::Zero>::zero);
    let zero = <T as num_traits::Zero>::zero();
    let one = <T as num_traits::One>::one();
    let output = unary_map(input, |x| {
        if x == zero {
            one
        } else {
            let px = pi * x;
            px.sin() / px
        }
    })?;
    finish_unary(output, input, || SincBackward {
        input: input.clone(),
    })
}

// ===========================================================================
// Binary transcendentals: atan2, copysign, hypot, nextafter; plus signbit
// ===========================================================================

/// Whether at least one of two tensors requires grad (and grad is enabled).
#[inline]
fn needs_grad_binary<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> bool {
    is_grad_enabled() && (a.requires_grad() || b.requires_grad())
}

// ---------------------------------------------------------------------------
// atan2(y, x) — element-wise four-quadrant arctangent.
// Forward mirrors `aten/src/ATen/native/BinaryOps.cpp:795 TORCH_IMPL_FUNC(atan2_out)`.
// Backward per `tools/autograd/derivatives.yaml:355-356`:
//   - name: atan2(Tensor self, Tensor other) -> Tensor
//     self, other: atan2_backward(grad, self, other, grad_input_mask)
// Implementation in `torch/csrc/autograd/FunctionsManual.cpp:3391-3410`:
//   denom = self*self + other*other; recip = denom.reciprocal();
//   recip.masked_fill_(denom == 0, 0);
//   { grad * other * recip, grad * -self * recip }
// where `self` is `y` and `other` is `x` per `torch/_torch_docs.py atan2`.
// ---------------------------------------------------------------------------

/// Backward node for `c = atan2(y, x)`. Saves both broadcast operands so the
/// VJP can reconstruct `denom = y^2 + x^2` and route gradients with the
/// `denom == 0 -> 0` masked guard.
#[derive(Debug)]
struct Atan2Backward<T: Float> {
    y: Tensor<T>,
    x: Tensor<T>,
}

impl<T: Float> GradFn<T> for Atan2Backward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if grad_output.is_cuda() || self.y.is_cuda() || self.x.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "atan2 backward",
            });
        }
        // Build broadcast-shape recip mask = 1/(y^2+x^2), with 0 where denom==0.
        let out_shape = broadcast_shapes(self.y.shape(), self.x.shape())?;
        // Reconstruct broadcast y, x (no_grad views, materialized to a flat
        // broadcast buffer via binary_map identity-on-component).
        let zero = <T as num_traits::Zero>::zero();
        let y_b = binary_map(&self.y, &self.x, |y, _x| y)?;
        let x_b = binary_map(&self.y, &self.x, |_y, x| x)?;
        let denom = binary_map(&y_b, &x_b, |y, x| y * y + x * x)?;
        let denom_data = denom.data()?;
        let y_data = y_b.data()?;
        let x_data = x_b.data()?;
        let go_data = grad_output.data()?;
        let recip: Vec<T> = denom_data
            .iter()
            .map(|&d| {
                if d == zero {
                    zero
                } else {
                    <T as num_traits::One>::one() / d
                }
            })
            .collect();
        let grad_y_raw: Vec<T> = go_data
            .iter()
            .zip(x_data.iter())
            .zip(recip.iter())
            .map(|((&g, &x), &r)| g * x * r)
            .collect();
        let grad_x_raw: Vec<T> = go_data
            .iter()
            .zip(y_data.iter())
            .zip(recip.iter())
            .map(|((&g, &y), &r)| -(g * y * r))
            .collect();
        let grad_y_tensor =
            Tensor::from_storage(TensorStorage::cpu(grad_y_raw), out_shape.clone(), false)?;
        let grad_x_tensor = Tensor::from_storage(TensorStorage::cpu(grad_x_raw), out_shape, false)?;
        let da = if self.y.requires_grad() {
            Some(reduce_grad_to_shape(&grad_y_tensor, self.y.shape())?)
        } else {
            None
        };
        let db = if self.x.requires_grad() {
            Some(reduce_grad_to_shape(&grad_x_tensor, self.x.shape())?)
        } else {
            None
        };
        Ok(vec![da, db])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.y, &self.x]
    }

    fn name(&self) -> &'static str {
        "Atan2Backward"
    }
}

/// Differentiable element-wise `atan2(y, x)`. Forward mirrors
/// `aten/src/ATen/native/BinaryOps.cpp:795 TORCH_IMPL_FUNC(atan2_out)`. The
/// argument order matches `torch.atan2(input, other)` per `torch/_torch_docs.py`,
/// where `input == y` and `other == x` (the result is the angle whose tangent
/// equals `y/x`, with quadrant-aware sign per IEEE-754 `atan2`).
///
/// Backward routes through the joint formula:
///   `grad_y = grad * x / (y^2 + x^2)`; `grad_x = -grad * y / (y^2 + x^2)`,
/// with the `denom == 0 -> 0` masked guard for the trivial (0,0) input.
pub fn atan2<T: Float>(y: &Tensor<T>, x: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if y.is_cuda() || x.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "atan2" });
    }
    let output = binary_map(y, x, |yy, xx| yy.atan2(xx))?;
    if needs_grad_binary(y, x) {
        let (storage, shape) = output.into_storage_and_shape()?;
        Tensor::from_operation(
            storage,
            shape,
            Arc::new(Atan2Backward {
                y: y.clone(),
                x: x.clone(),
            }),
        )
    } else {
        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// signbit(x) — Bool output, non-differentiable.
// Mirrors `aten/src/ATen/native/UnaryOps.cpp:389 TORCH_IMPL_FUNC(signbit_out)`
// and the meta-func at `:279-282` which asserts non-complex input + Bool dtype
// output. Returns `true` for negative-zero (`-0.0`) and `false` for NaN that
// has a clear sign bit; matches `x.is_sign_negative()` semantics in IEEE-754.
// ---------------------------------------------------------------------------

/// Non-differentiable element-wise `signbit(x)`. Returns a [`BoolTensor`]
/// where each element is `true` iff the corresponding input is negative
/// (sign bit set), matching `f32::is_sign_negative` / `f64::is_sign_negative`.
/// Bool output is not differentiable — there is no `derivatives.yaml` entry.
pub fn signbit<T: Float>(input: &Tensor<T>) -> FerrotorchResult<BoolTensor> {
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "signbit" });
    }
    let data = input.data()?;
    // `num_traits::Float::is_sign_negative` is the canonical IEEE-754 sign-bit
    // check: returns `true` for `-0.0` (sign bit set, value compares equal to
    // +0.0) and honors the sign bit of NaN. Matches upstream
    // `aten/src/ATen/native/UnaryOps.cpp:389 TORCH_IMPL_FUNC(signbit_out)`
    // which routes to `std::signbit` on CPU.
    let bits: Vec<bool> = data
        .iter()
        .map(|&v| <T as num_traits::Float>::is_sign_negative(v))
        .collect();
    BoolTensor::from_vec(bits, input.shape().to_vec())
}

// ---------------------------------------------------------------------------
// copysign(magnitude, sign) — element-wise; differentiable through magnitude.
// Forward mirrors `aten/src/ATen/native/BinaryOps.cpp:865 copysign_out`.
// Backward per `tools/autograd/derivatives.yaml:493-496`:
//   - name: copysign.Tensor(Tensor self, Tensor other) -> Tensor
//     self: copysign_tensor_self_backward(grad, self, result)
//     other: zeros_like(other)
// Implementation in `torch/csrc/autograd/FunctionsManual.cpp:106-114`:
//   ratio = result / self; ratio.masked_fill_(self == 0, 0); return grad * ratio
// — the ratio is +/-1 (the effective sign factor); the `self == 0` mask
// avoids the 0/0 NaN. Gradient to `other` (the sign source) is zero.
// ---------------------------------------------------------------------------

/// Backward node for `c = copysign(magnitude, sign)`. Saves both broadcast
/// operands plus the forward result. The VJP is
/// `grad_magnitude = grad * (result / magnitude)` with the `magnitude == 0 -> 0`
/// masked guard; gradient to `sign` is identically zero.
#[derive(Debug)]
struct CopysignBackward<T: Float> {
    magnitude: Tensor<T>,
    sign: Tensor<T>,
    result: Tensor<T>,
}

impl<T: Float> GradFn<T> for CopysignBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if grad_output.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "copysign backward",
            });
        }
        let zero = <T as num_traits::Zero>::zero();
        let out_shape = broadcast_shapes(self.magnitude.shape(), self.sign.shape())?;
        let mag_b = binary_map(&self.magnitude, &self.sign, |m, _s| m)?;
        let res_data = self.result.data()?;
        let mag_data = mag_b.data()?;
        let go_data = grad_output.data()?;
        let raw: Vec<T> = go_data
            .iter()
            .zip(res_data.iter())
            .zip(mag_data.iter())
            .map(|((&g, &r), &m)| if m == zero { zero } else { g * (r / m) })
            .collect();
        let grad_b = Tensor::from_storage(TensorStorage::cpu(raw), out_shape, false)?;
        let da = if self.magnitude.requires_grad() {
            Some(reduce_grad_to_shape(&grad_b, self.magnitude.shape())?)
        } else {
            None
        };
        // Gradient to `sign` is zero (sign is non-diff per derivatives.yaml).
        let db = if self.sign.requires_grad() {
            let n: usize = self.sign.shape().iter().product::<usize>().max(1);
            Some(Tensor::from_storage(
                TensorStorage::cpu(vec![zero; n]),
                self.sign.shape().to_vec(),
                false,
            )?)
        } else {
            None
        };
        Ok(vec![da, db])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.magnitude, &self.sign]
    }

    fn name(&self) -> &'static str {
        "CopysignBackward"
    }
}

/// Differentiable element-wise `copysign(magnitude, sign)`. Returns a tensor
/// with the magnitude of `magnitude` and the sign of `sign`. Mirrors
/// `aten/src/ATen/native/BinaryOps.cpp:865 copysign_out`. Backward: gradient
/// flows to `magnitude` scaled by `sign_factor = result / magnitude` (zeroed
/// where `magnitude == 0`); gradient to `sign` is identically zero.
pub fn copysign<T: Float>(magnitude: &Tensor<T>, sign: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if magnitude.is_cuda() || sign.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "copysign" });
    }
    let output = binary_map(magnitude, sign, |m, s| {
        // f64::copysign / f32::copysign do the right IEEE-754 thing. We
        // route through num_traits::Float::copysign which dispatches to those.
        <T as num_traits::Float>::copysign(m, s)
    })?;
    if needs_grad_binary(magnitude, sign) {
        let (storage, shape) = output.into_storage_and_shape()?;
        let out_tensor = Tensor::from_storage(storage, shape, false)?;
        let grad_fn = Arc::new(CopysignBackward {
            magnitude: magnitude.clone(),
            sign: sign.clone(),
            result: out_tensor.clone(),
        });
        let (s, sh) = out_tensor.into_storage_and_shape()?;
        Tensor::from_operation(s, sh, grad_fn)
    } else {
        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// hypot(x, y) — element-wise sqrt(x^2 + y^2) with overflow-safe accumulation.
// Forward mirrors `aten/src/ATen/native/BinaryOps.cpp:548
//   CREATE_BINARY_TORCH_IMPL_FUNC(hypot_out, hypot_stub)`.
// Backward per `tools/autograd/derivatives.yaml:814-817`:
//   - name: hypot(Tensor self, Tensor other) -> Tensor
//     self: grad * self / result
//     other: grad * other / result
// — when result == 0 (both inputs zero), the gradient is 0/0 = NaN per IEEE;
// upstream does not mask this (matches `torch.hypot(0,0).backward()` live).
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct HypotBackward<T: Float> {
    a: Tensor<T>,
    b: Tensor<T>,
    result: Tensor<T>,
}

impl<T: Float> GradFn<T> for HypotBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if grad_output.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "hypot backward",
            });
        }
        let out_shape = broadcast_shapes(self.a.shape(), self.b.shape())?;
        let a_b = binary_map(&self.a, &self.b, |x, _y| x)?;
        let b_b = binary_map(&self.a, &self.b, |_x, y| y)?;
        let go_data = grad_output.data()?;
        let res_data = self.result.data()?;
        let a_data = a_b.data()?;
        let b_data = b_b.data()?;
        let zero = <T as num_traits::Zero>::zero();
        let raw_a: Vec<T> = go_data
            .iter()
            .zip(a_data.iter())
            .zip(res_data.iter())
            .map(|((&g, &x), &r)| if r == zero { zero } else { g * x / r })
            .collect();
        let raw_b: Vec<T> = go_data
            .iter()
            .zip(b_data.iter())
            .zip(res_data.iter())
            .map(|((&g, &y), &r)| if r == zero { zero } else { g * y / r })
            .collect();
        let ga = Tensor::from_storage(TensorStorage::cpu(raw_a), out_shape.clone(), false)?;
        let gb = Tensor::from_storage(TensorStorage::cpu(raw_b), out_shape, false)?;
        let da = if self.a.requires_grad() {
            Some(reduce_grad_to_shape(&ga, self.a.shape())?)
        } else {
            None
        };
        let db = if self.b.requires_grad() {
            Some(reduce_grad_to_shape(&gb, self.b.shape())?)
        } else {
            None
        };
        Ok(vec![da, db])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.a, &self.b]
    }

    fn name(&self) -> &'static str {
        "HypotBackward"
    }
}

/// Differentiable element-wise `hypot(x, y) = sqrt(x^2 + y^2)` with the
/// overflow-safe accumulation provided by `num_traits::Float::hypot`
/// (delegates to `f32::hypot` / `f64::hypot`). Mirrors
/// `aten/src/ATen/native/BinaryOps.cpp:548 hypot_out`. Backward:
///   `grad_x = grad * x / result; grad_y = grad * y / result`,
/// with `result == 0 -> 0` masking (matching the upstream behavior in
/// `derivatives.yaml:814-817` whose `grad * self / result` is implicitly
/// degenerate at the origin — we mask to a safe zero rather than producing
/// NaN, which differs from torch's literal IEEE 0/0 output at the (0,0) tie
/// only; the divergence is filed as documentation, not a parity blocker).
pub fn hypot<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if a.is_cuda() || b.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "hypot" });
    }
    let output = binary_map(a, b, |x, y| <T as num_traits::Float>::hypot(x, y))?;
    if needs_grad_binary(a, b) {
        let (storage, shape) = output.into_storage_and_shape()?;
        let out_tensor = Tensor::from_storage(storage, shape, false)?;
        let grad_fn = Arc::new(HypotBackward {
            a: a.clone(),
            b: b.clone(),
            result: out_tensor.clone(),
        });
        let (s, sh) = out_tensor.into_storage_and_shape()?;
        Tensor::from_operation(s, sh, grad_fn)
    } else {
        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// nextafter(a, b) — element-wise next representable float from `a` toward `b`.
// Forward mirrors `aten/src/ATen/native/BinaryOps.cpp:551
//   CREATE_BINARY_TORCH_IMPL_FUNC(nextafter_out, nextafter_stub)` whose CPU
// kernel routes to `std::nextafter`. Backward per
// `tools/autograd/derivatives.yaml:1322-1324`:
//   - name: nextafter(Tensor self, Tensor other) -> Tensor
//     self: at::where(self != other, grad, 0)
//     other: zeros_like(other)
// — the gradient passes through to `self` (the step is one ULP, so locally
// the map is the identity for autograd purposes) but is masked to zero on the
// `self == other` tie (where `nextafter(a, a) == a` is exactly flat); gradient
// to `other` (the direction operand) is identically zero.
// ---------------------------------------------------------------------------

/// One-ULP step of an `f64` toward `+inf` (`up = true`) or `-inf`
/// (`up = false`) via bit-pattern increment/decrement, mirroring the classic
/// `std::nextafter` integer-monotonicity trick: the IEEE-754 bit pattern of a
/// positive float is monotonically increasing in value, so adding 1 to the
/// bits advances exactly one ULP. Crossing zero is handled by the sign-flip
/// branch. Used by [`nextafter_scalar`]; bit-manipulation is chosen over
/// `f64::next_up`/`next_down` because those stabilised in Rust 1.86 and the
/// workspace MSRV is 1.85.
#[inline]
#[allow(
    clippy::float_cmp,
    reason = "exact IEEE-754 zero test gates the cross-zero ULP branch; \
              an epsilon tolerance would corrupt nextafter's bit-exact step."
)]
fn f64_one_ulp(x: f64, up: bool) -> f64 {
    debug_assert!(!x.is_nan());
    if x == 0.0 {
        // From +/-0, the next value toward +inf is +MIN_POSITIVE_SUBNORMAL,
        // toward -inf is its negation. `f64::from_bits(1)` is the smallest
        // positive subnormal.
        return if up {
            f64::from_bits(1)
        } else {
            -f64::from_bits(1)
        };
    }
    let bits = x.to_bits();
    // For x > 0, value increases with bits; for x < 0, value increases as bits
    // decrease (sign bit set). `up == (x > 0)` means step in the +bits dir.
    let step_up_in_bits = up == (x > 0.0);
    let new_bits = if step_up_in_bits { bits + 1 } else { bits - 1 };
    f64::from_bits(new_bits)
}

/// One-ULP step of an `f32` toward `+inf` (`up = true`) or `-inf`
/// (`up = false`). The f64 routing in the original [`nextafter_scalar`] stepped
/// one ULP in `f64`, producing a value strictly *between* two adjacent `f32`s
/// that `T::from::<f32>` then rounded straight back to the original `f32` — so
/// the op was a no-op for every `f32` input (#1335/#1556). Stepping at the
/// native `u32` width matches `std::nextafter` at the tensor dtype
/// (`BinaryOpsKernel.cpp:1257` `std::nextafter(a, b)` under
/// `AT_DISPATCH_FLOATING_TYPES`).
#[inline]
#[allow(
    clippy::float_cmp,
    reason = "exact IEEE-754 zero test gates the cross-zero ULP branch; \
              an epsilon tolerance would corrupt nextafter's bit-exact step."
)]
fn f32_one_ulp(x: f32, up: bool) -> f32 {
    debug_assert!(!x.is_nan());
    if x == 0.0 {
        return if up {
            f32::from_bits(1)
        } else {
            -f32::from_bits(1)
        };
    }
    let bits = x.to_bits();
    let step_up_in_bits = up == (x > 0.0);
    let new_bits = if step_up_in_bits { bits + 1 } else { bits - 1 };
    f32::from_bits(new_bits)
}

/// One-ULP step of a `half` 16-bit float (`f16` or `bf16`) toward `+inf`
/// (`up = true`) or `-inf` (`up = false`), parameterised by the value's sign
/// and IEEE-754 bit pattern. Both `half::f16` and `half::bf16` share the same
/// `[sign:1][exp][mantissa]` monotone layout, so the integer-increment trick
/// is identical to the `f32`/`f64` cases; only the dtype reconstruction differs
/// (handled by the caller via `to_bits`/`from_bits`). Returns the stepped
/// 16-bit pattern.
#[inline]
fn u16_one_ulp(bits: u16, is_zero: bool, is_positive: bool, up: bool) -> u16 {
    if is_zero {
        // Smallest positive subnormal is 0x0001; its negation sets the sign
        // bit (0x8001).
        return if up { 0x0001 } else { 0x8001 };
    }
    let step_up_in_bits = up == is_positive;
    if step_up_in_bits { bits + 1 } else { bits - 1 }
}

/// IEEE-754 next-representable value from `a` toward `b`, generic over the
/// workspace `Float` dtypes. The single ULP step is taken at the value's
/// NATIVE width — `u32` for `f32`, `u64` for `f64`, `u16` for `f16`/`bf16` —
/// matching `std::nextafter(a, b)` at the tensor dtype
/// (`aten/src/ATen/native/cpu/BinaryOpsKernel.cpp:1257`). Stepping in `f64` and
/// casting back (the prior behaviour) rounded straight back to the original
/// value for every narrower dtype, making the op a no-op (#1335/#1556).
/// Matches `std::nextafter` semantics:
///   - `a == b` (incl. signed-zero compare-equal) -> return `b`,
///   - either operand NaN -> NaN,
///   - otherwise step one ULP from `a` in the direction of `b`.
#[inline]
#[allow(
    clippy::float_cmp,
    reason = "exact IEEE-754 equality is the std::nextafter tie semantics: \
              on a == b (incl. signed-zero) the result is exactly b."
)]
fn nextafter_scalar<T: Float>(a: T, b: T) -> T {
    if a.is_nan() || b.is_nan() {
        return T::nan();
    }
    // `==` treats `-0.0 == 0.0`; `std::nextafter` returns `b` (the toward
    // operand) on equality so the result carries `b`'s sign for the zero tie.
    if a == b {
        return b;
    }
    // Direction: step toward `+inf` iff `b > a`.
    let up = b > a;

    if is_f32::<T>() {
        let af = <T as num_traits::ToPrimitive>::to_f32(&a).unwrap_or(f32::NAN);
        let stepped = f32_one_ulp(af, up);
        return T::from(stepped).unwrap_or(b);
    }
    if is_f64::<T>() {
        let af = <T as num_traits::ToPrimitive>::to_f64(&a).unwrap_or(f64::NAN);
        let stepped = f64_one_ulp(af, up);
        return T::from(stepped).unwrap_or(b);
    }
    if is_f16::<T>() {
        // `NumCast::from` is bit-exact here: TypeId has confirmed `T == f16`,
        // so the cast is the identity that recovers the native `half::f16`.
        let ah: half::f16 = match <half::f16 as num_traits::NumCast>::from(a) {
            Some(v) => v,
            None => return b,
        };
        let bits = u16_one_ulp(
            ah.to_bits(),
            ah == half::f16::ZERO,
            ah > half::f16::ZERO,
            up,
        );
        return T::from(half::f16::from_bits(bits)).unwrap_or(b);
    }
    if is_bf16::<T>() {
        let ah: half::bf16 = match <half::bf16 as num_traits::NumCast>::from(a) {
            Some(v) => v,
            None => return b,
        };
        let bits = u16_one_ulp(
            ah.to_bits(),
            ah == half::bf16::ZERO,
            ah > half::bf16::ZERO,
            up,
        );
        return T::from(half::bf16::from_bits(bits)).unwrap_or(b);
    }
    // Unreachable for the four `Float` dtypes; conservative fallback steps in
    // f64 (still strictly toward `b`).
    let af = <T as num_traits::ToPrimitive>::to_f64(&a).unwrap_or(f64::NAN);
    T::from(f64_one_ulp(af, up)).unwrap_or(b)
}

/// Backward node for `c = nextafter(a, b)`. The VJP routes `grad` straight
/// through to `a` everywhere `a != b`, masking the gradient to zero on the
/// `a == b` tie (the flat self-step); gradient to `b` is identically zero.
/// Saves both broadcast operands to rebuild the `a != b` mask.
#[derive(Debug)]
struct NextafterBackward<T: Float> {
    a: Tensor<T>,
    b: Tensor<T>,
}

impl<T: Float> GradFn<T> for NextafterBackward<T> {
    #[allow(
        clippy::float_cmp,
        reason = "the `a != b` mask is the exact upstream gradient gate per \
                  derivatives.yaml:1323 `at::where(self != other, grad, 0)`; \
                  an epsilon tolerance would misroute the tie's zero gradient."
    )]
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if grad_output.is_cuda() || self.a.is_cuda() || self.b.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "nextafter backward",
            });
        }
        let zero = <T as num_traits::Zero>::zero();
        let out_shape = broadcast_shapes(self.a.shape(), self.b.shape())?;
        let a_b = binary_map(&self.a, &self.b, |a, _b| a)?;
        let b_b = binary_map(&self.a, &self.b, |_a, b| b)?;
        let go_data = grad_output.data()?;
        let a_data = a_b.data()?;
        let b_data = b_b.data()?;
        // self: at::where(self != other, grad, 0).
        let raw_a: Vec<T> = go_data
            .iter()
            .zip(a_data.iter())
            .zip(b_data.iter())
            .map(|((&g, &av), &bv)| if av == bv { zero } else { g })
            .collect();
        let grad_a = Tensor::from_storage(TensorStorage::cpu(raw_a), out_shape, false)?;
        let da = if self.a.requires_grad() {
            Some(reduce_grad_to_shape(&grad_a, self.a.shape())?)
        } else {
            None
        };
        // other: zeros_like(other).
        let db = if self.b.requires_grad() {
            let n: usize = self.b.shape().iter().product::<usize>().max(1);
            Some(Tensor::from_storage(
                TensorStorage::cpu(vec![zero; n]),
                self.b.shape().to_vec(),
                false,
            )?)
        } else {
            None
        };
        Ok(vec![da, db])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.a, &self.b]
    }

    fn name(&self) -> &'static str {
        "NextafterBackward"
    }
}

/// Differentiable element-wise `nextafter(a, b)`: the next representable
/// floating-point value after `a` in the direction of `b`. Forward mirrors
/// `aten/src/ATen/native/BinaryOps.cpp:551 nextafter_out` (CPU kernel
/// `std::nextafter`). Backward per `derivatives.yaml:1322-1324` routes `grad`
/// to `a` where `a != b` (zero on the `a == b` tie); gradient to `b` is zero.
pub fn nextafter<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if a.is_cuda() || b.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "nextafter" });
    }
    let output = binary_map(a, b, nextafter_scalar)?;
    if needs_grad_binary(a, b) {
        let (storage, shape) = output.into_storage_and_shape()?;
        let out_tensor = Tensor::from_storage(storage, shape, false)?;
        let grad_fn = Arc::new(NextafterBackward {
            a: a.clone(),
            b: b.clone(),
        });
        let (s, sh) = out_tensor.into_storage_and_shape()?;
        Tensor::from_operation(s, sh, grad_fn)
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

    /// Create a leaf scalar tensor.
    fn leaf_scalar(val: f32, requires_grad: bool) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(vec![val]), vec![], requires_grad).unwrap()
    }

    /// Create a leaf 1-D tensor.
    fn leaf_vec(data: &[f32], requires_grad: bool) -> Tensor<f32> {
        Tensor::from_storage(
            TensorStorage::cpu(data.to_vec()),
            vec![data.len()],
            requires_grad,
        )
        .unwrap()
    }

    /// Assert a scalar tensor is approximately equal to `expected`.
    fn assert_scalar_approx(t: &Tensor<f32>, expected: f32, tol: f32) {
        let val = t.item().unwrap();
        assert!(
            (val - expected).abs() < tol,
            "expected {expected}, got {val}"
        );
    }

    // -----------------------------------------------------------------------
    // Forward tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_exp_forward() {
        let a = leaf_vec(&[0.0, 1.0, 2.0], false);
        let c = exp(&a).unwrap();
        let d = c.data().unwrap();
        assert!((d[0] - 1.0).abs() < 1e-5);
        assert!((d[1] - std::f32::consts::E).abs() < 1e-5);
        assert!((d[2] - std::f32::consts::E * std::f32::consts::E).abs() < 1e-4);
    }

    #[test]
    fn test_log_forward() {
        let a = leaf_vec(
            &[
                1.0,
                std::f32::consts::E,
                std::f32::consts::E * std::f32::consts::E,
            ],
            false,
        );
        let c = log(&a).unwrap();
        let d = c.data().unwrap();
        assert!((d[0] - 0.0).abs() < 1e-5);
        assert!((d[1] - 1.0).abs() < 1e-5);
        assert!((d[2] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_sin_forward() {
        let a = leaf_vec(
            &[0.0, std::f32::consts::FRAC_PI_2, std::f32::consts::PI],
            false,
        );
        let c = sin(&a).unwrap();
        let d = c.data().unwrap();
        assert!((d[0] - 0.0).abs() < 1e-6);
        assert!((d[1] - 1.0).abs() < 1e-6);
        assert!(d[2].abs() < 1e-6);
    }

    #[test]
    fn test_cos_forward() {
        let a = leaf_vec(
            &[0.0, std::f32::consts::FRAC_PI_2, std::f32::consts::PI],
            false,
        );
        let c = cos(&a).unwrap();
        let d = c.data().unwrap();
        assert!((d[0] - 1.0).abs() < 1e-6);
        assert!(d[1].abs() < 1e-6);
        assert!((d[2] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_clamp_forward() {
        let a = leaf_vec(&[-2.0, 0.5, 1.5, 3.0], false);
        let c = clamp(&a, 0.0, 2.0).unwrap();
        assert_eq!(c.data().unwrap(), &[0.0, 0.5, 1.5, 2.0]);
    }

    // -----------------------------------------------------------------------
    // Backward tests (scalar tensors for simplicity)
    // -----------------------------------------------------------------------

    #[test]
    fn test_exp_backward() {
        // c = exp(a); dc/da = exp(a).
        // a = 1.0 => dc/da = e ~= 2.7183.
        let a = leaf_scalar(1.0, true);
        let c = exp(&a).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), std::f32::consts::E, 1e-5);
    }

    #[test]
    fn test_log_backward() {
        // c = ln(a); dc/da = 1/a.
        // a = 2.0 => dc/da = 0.5.
        let a = leaf_scalar(2.0, true);
        let c = log(&a).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), 0.5, 1e-6);
    }

    #[test]
    fn test_sin_backward() {
        // c = sin(a); dc/da = cos(a).
        // a = 0.0 => dc/da = cos(0) = 1.0.
        let a = leaf_scalar(0.0, true);
        let c = sin(&a).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), 1.0, 1e-6);
    }

    #[test]
    fn test_sin_backward_pi_over_3() {
        // a = pi/3 => dc/da = cos(pi/3) = 0.5.
        let a = leaf_scalar(std::f32::consts::FRAC_PI_3, true);
        let c = sin(&a).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), 0.5, 1e-5);
    }

    #[test]
    fn test_cos_backward() {
        // c = cos(a); dc/da = -sin(a).
        // a = 0.0 => dc/da = -sin(0) = 0.
        let a = leaf_scalar(0.0, true);
        let c = cos(&a).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), 0.0, 1e-6);
    }

    #[test]
    fn test_cos_backward_pi_over_2() {
        // a = pi/2 => dc/da = -sin(pi/2) = -1.0.
        let a = leaf_scalar(std::f32::consts::FRAC_PI_2, true);
        let c = cos(&a).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), -1.0, 1e-5);
    }

    #[test]
    fn test_clamp_backward_interior() {
        // a = 1.5, clamp(0, 2) => interior, so dc/da = 1.
        let a = leaf_scalar(1.5, true);
        let c = clamp(&a, 0.0, 2.0).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), 1.0, 1e-6);
    }

    #[test]
    fn test_clamp_backward_clamped_low() {
        // a = -1.0, clamp(0, 2) => clamped to min, so dc/da = 0.
        let a = leaf_scalar(-1.0, true);
        let c = clamp(&a, 0.0, 2.0).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), 0.0, 1e-6);
    }

    #[test]
    fn test_clamp_backward_clamped_high() {
        // a = 5.0, clamp(0, 2) => clamped to max, so dc/da = 0.
        let a = leaf_scalar(5.0, true);
        let c = clamp(&a, 0.0, 2.0).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), 0.0, 1e-6);
    }

    // -----------------------------------------------------------------------
    // Chain rule tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_chain_exp_log() {
        // c = log(exp(a)) = a. dc/da = 1.
        let a = leaf_scalar(3.0, true);
        let b = exp(&a).unwrap();
        let c = log(&b).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), 1.0, 1e-4);
    }

    #[test]
    fn test_chain_sin_cos() {
        // c = sin(a)^2 + cos(a)^2 = 1 for all a.
        // But that requires add/mul/pow which are separate ops.
        // Instead: c = cos(sin(a)).
        // dc/da = -sin(sin(a)) * cos(a).
        // a = 0.5 => dc/da = -sin(sin(0.5)) * cos(0.5)
        //          = -sin(0.4794) * 0.8776
        //          = -0.4611 * 0.8776 ~= -0.4047
        let a = leaf_scalar(0.5, true);
        let b = sin(&a).unwrap();
        let c = cos(&b).unwrap();
        c.backward().unwrap();

        let expected = -(0.5_f32.sin().sin()) * 0.5_f32.cos();
        assert_scalar_approx(&a.grad().unwrap().unwrap(), expected, 1e-4);
    }

    // -----------------------------------------------------------------------
    // No-grad test
    // -----------------------------------------------------------------------

    #[test]
    fn test_exp_no_grad_fn_when_not_tracking() {
        let a = leaf_scalar(1.0, false);
        let c = exp(&a).unwrap();
        assert!(c.grad_fn().is_none());
    }

    #[test]
    fn test_log_no_grad_fn_when_not_tracking() {
        let a = leaf_scalar(1.0, false);
        let c = log(&a).unwrap();
        assert!(c.grad_fn().is_none());
    }

    #[test]
    fn test_clamp_no_grad_fn_when_not_tracking() {
        let a = leaf_scalar(1.0, false);
        let c = clamp(&a, 0.0, 2.0).unwrap();
        assert!(c.grad_fn().is_none());
    }

    // -----------------------------------------------------------------------
    // Numerical gradient check (finite difference)
    // -----------------------------------------------------------------------

    /// Check gradient using central finite differences:
    ///   grad ~= (f(x+h) - f(x-h)) / (2*h)
    fn numerical_grad_check(f: impl Fn(f32) -> f32, x: f32, analytic_grad: f32, tol: f32) {
        let h = 1e-4_f32;
        let numerical = (f(x + h) - f(x - h)) / (2.0 * h);
        assert!(
            (analytic_grad - numerical).abs() < tol,
            "analytic={analytic_grad}, numerical={numerical}",
        );
    }

    #[test]
    fn test_exp_numerical_grad() {
        let x = 1.5_f32;
        let a = leaf_scalar(x, true);
        let c = exp(&a).unwrap();
        c.backward().unwrap();
        let g = a.grad().unwrap().unwrap().item().unwrap();
        numerical_grad_check(|v| v.exp(), x, g, 1e-3);
    }

    #[test]
    fn test_log_numerical_grad() {
        let x = 2.0_f32;
        let a = leaf_scalar(x, true);
        let c = log(&a).unwrap();
        c.backward().unwrap();
        let g = a.grad().unwrap().unwrap().item().unwrap();
        numerical_grad_check(|v| v.ln(), x, g, 1e-3);
    }

    #[test]
    fn test_sin_numerical_grad() {
        let x = 1.0_f32;
        let a = leaf_scalar(x, true);
        let c = sin(&a).unwrap();
        c.backward().unwrap();
        let g = a.grad().unwrap().unwrap().item().unwrap();
        numerical_grad_check(|v| v.sin(), x, g, 1e-3);
    }

    #[test]
    fn test_cos_numerical_grad() {
        let x = 1.0_f32;
        let a = leaf_scalar(x, true);
        let c = cos(&a).unwrap();
        c.backward().unwrap();
        let g = a.grad().unwrap().unwrap().item().unwrap();
        numerical_grad_check(|v| v.cos(), x, g, 1e-3);
    }

    #[test]
    fn test_clamp_numerical_grad_interior() {
        let x = 0.5_f32;
        let a = leaf_scalar(x, true);
        let c = clamp(&a, 0.0, 1.0).unwrap();
        c.backward().unwrap();
        let g = a.grad().unwrap().unwrap().item().unwrap();
        numerical_grad_check(|v| v.clamp(0.0, 1.0), x, g, 1e-3);
    }

    // -----------------------------------------------------------------------
    // New unary ops (tan, asin, acos, atan, sinh, cosh, asinh, acosh, atanh,
    // exp2, expm1, log2, log10, log1p, ceil, floor, round, trunc, frac, sign,
    // sinc) — forward + backward + numerical-gradient parity.
    // -----------------------------------------------------------------------

    // tan
    #[test]
    fn test_tan_forward_and_backward() {
        let a = leaf_scalar(0.5_f32, true);
        let c = tan(&a).unwrap();
        let v = c.item().unwrap();
        assert!((v - 0.5_f32.tan()).abs() < 1e-6, "tan(0.5) = {v}");
        c.backward().unwrap();
        let g = a.grad().unwrap().unwrap().item().unwrap();
        numerical_grad_check(|v| v.tan(), 0.5, g, 1e-3);
    }

    // asin
    #[test]
    fn test_asin_forward_and_backward() {
        let a = leaf_scalar(0.5_f32, true);
        let c = asin(&a).unwrap();
        let v = c.item().unwrap();
        assert!((v - 0.5_f32.asin()).abs() < 1e-6);
        c.backward().unwrap();
        let g = a.grad().unwrap().unwrap().item().unwrap();
        numerical_grad_check(|v| v.asin(), 0.5, g, 1e-3);
    }

    // acos
    #[test]
    fn test_acos_forward_and_backward() {
        let a = leaf_scalar(0.5_f32, true);
        let c = acos(&a).unwrap();
        let v = c.item().unwrap();
        assert!((v - 0.5_f32.acos()).abs() < 1e-6);
        c.backward().unwrap();
        let g = a.grad().unwrap().unwrap().item().unwrap();
        numerical_grad_check(|v| v.acos(), 0.5, g, 1e-3);
    }

    // atan
    #[test]
    fn test_atan_forward_and_backward() {
        let a = leaf_scalar(1.0_f32, true);
        let c = atan(&a).unwrap();
        let v = c.item().unwrap();
        assert!((v - 1.0_f32.atan()).abs() < 1e-6);
        c.backward().unwrap();
        let g = a.grad().unwrap().unwrap().item().unwrap();
        numerical_grad_check(|v| v.atan(), 1.0, g, 1e-3);
    }

    // sinh
    #[test]
    fn test_sinh_forward_and_backward() {
        let a = leaf_scalar(0.5_f32, true);
        let c = sinh(&a).unwrap();
        let v = c.item().unwrap();
        assert!((v - 0.5_f32.sinh()).abs() < 1e-6);
        c.backward().unwrap();
        let g = a.grad().unwrap().unwrap().item().unwrap();
        numerical_grad_check(|v| v.sinh(), 0.5, g, 1e-3);
    }

    // cosh
    #[test]
    fn test_cosh_forward_and_backward() {
        let a = leaf_scalar(0.5_f32, true);
        let c = cosh(&a).unwrap();
        let v = c.item().unwrap();
        assert!((v - 0.5_f32.cosh()).abs() < 1e-6);
        c.backward().unwrap();
        let g = a.grad().unwrap().unwrap().item().unwrap();
        numerical_grad_check(|v| v.cosh(), 0.5, g, 1e-3);
    }

    // asinh
    #[test]
    fn test_asinh_forward_and_backward() {
        let a = leaf_scalar(0.7_f32, true);
        let c = asinh(&a).unwrap();
        let v = c.item().unwrap();
        assert!((v - 0.7_f32.asinh()).abs() < 1e-6);
        c.backward().unwrap();
        let g = a.grad().unwrap().unwrap().item().unwrap();
        numerical_grad_check(|v| v.asinh(), 0.7, g, 1e-3);
    }

    // acosh
    #[test]
    fn test_acosh_forward_and_backward() {
        let a = leaf_scalar(1.5_f32, true);
        let c = acosh(&a).unwrap();
        let v = c.item().unwrap();
        assert!((v - 1.5_f32.acosh()).abs() < 1e-6);
        c.backward().unwrap();
        let g = a.grad().unwrap().unwrap().item().unwrap();
        numerical_grad_check(|v| v.acosh(), 1.5, g, 1e-3);
    }

    // atanh
    #[test]
    fn test_atanh_forward_and_backward() {
        let a = leaf_scalar(0.5_f32, true);
        let c = atanh(&a).unwrap();
        let v = c.item().unwrap();
        assert!((v - 0.5_f32.atanh()).abs() < 1e-6);
        c.backward().unwrap();
        let g = a.grad().unwrap().unwrap().item().unwrap();
        numerical_grad_check(|v| v.atanh(), 0.5, g, 1e-3);
    }

    // exp2
    #[test]
    fn test_exp2_forward_and_backward() {
        let a = leaf_scalar(2.0_f32, true);
        let c = exp2(&a).unwrap();
        let v = c.item().unwrap();
        assert!((v - 4.0).abs() < 1e-5, "exp2(2) = {v}");
        c.backward().unwrap();
        let g = a.grad().unwrap().unwrap().item().unwrap();
        numerical_grad_check(|v| v.exp2(), 2.0, g, 1e-3);
    }

    // expm1
    #[test]
    fn test_expm1_forward_and_backward() {
        let a = leaf_scalar(0.5_f32, true);
        let c = expm1(&a).unwrap();
        let v = c.item().unwrap();
        assert!((v - 0.5_f32.exp_m1()).abs() < 1e-6);
        c.backward().unwrap();
        let g = a.grad().unwrap().unwrap().item().unwrap();
        numerical_grad_check(|v| v.exp_m1(), 0.5, g, 1e-3);
    }

    // log2
    #[test]
    fn test_log2_forward_and_backward() {
        let a = leaf_scalar(8.0_f32, true);
        let c = log2(&a).unwrap();
        let v = c.item().unwrap();
        assert!((v - 3.0).abs() < 1e-5);
        c.backward().unwrap();
        let g = a.grad().unwrap().unwrap().item().unwrap();
        numerical_grad_check(|v| v.log2(), 8.0, g, 1e-3);
    }

    // log10
    #[test]
    fn test_log10_forward_and_backward() {
        let a = leaf_scalar(100.0_f32, true);
        let c = log10(&a).unwrap();
        let v = c.item().unwrap();
        assert!((v - 2.0).abs() < 1e-5);
        c.backward().unwrap();
        let g = a.grad().unwrap().unwrap().item().unwrap();
        numerical_grad_check(|v| v.log10(), 100.0, g, 1e-1);
    }

    // log1p
    #[test]
    fn test_log1p_forward_and_backward() {
        let a = leaf_scalar(0.5_f32, true);
        let c = log1p(&a).unwrap();
        let v = c.item().unwrap();
        assert!((v - 0.5_f32.ln_1p()).abs() < 1e-6);
        c.backward().unwrap();
        let g = a.grad().unwrap().unwrap().item().unwrap();
        numerical_grad_check(|v| v.ln_1p(), 0.5, g, 1e-3);
    }

    // ceil
    #[test]
    fn test_ceil_forward_and_zero_backward() {
        let a = leaf_vec(&[-1.4, 0.5, 2.0], false);
        let c = ceil(&a).unwrap();
        assert_eq!(c.data().unwrap(), &[-1.0, 1.0, 2.0]);
        // backward returns zeros
        let a2 = leaf_scalar(0.3, true);
        let c2 = ceil(&a2).unwrap();
        c2.backward().unwrap();
        assert_scalar_approx(&a2.grad().unwrap().unwrap(), 0.0, 1e-6);
    }

    // floor
    #[test]
    fn test_floor_forward_and_zero_backward() {
        let a = leaf_vec(&[-1.4, 0.5, 2.9], false);
        let c = floor(&a).unwrap();
        assert_eq!(c.data().unwrap(), &[-2.0, 0.0, 2.0]);
        let a2 = leaf_scalar(0.3, true);
        let c2 = floor(&a2).unwrap();
        c2.backward().unwrap();
        assert_scalar_approx(&a2.grad().unwrap().unwrap(), 0.0, 1e-6);
    }

    // round (banker's)
    #[test]
    fn test_round_banker_rounding() {
        // Half-cases go to nearest even, matching torch's nearbyint default.
        let a = leaf_vec(&[0.5, 1.5, 2.5, 3.5, -0.5, -1.5], false);
        let c = round(&a).unwrap();
        assert_eq!(c.data().unwrap(), &[0.0, 2.0, 2.0, 4.0, 0.0, -2.0]);
    }

    // trunc
    #[test]
    fn test_trunc_forward_and_zero_backward() {
        let a = leaf_vec(&[-1.7, 0.4, 2.9], false);
        let c = trunc(&a).unwrap();
        assert_eq!(c.data().unwrap(), &[-1.0, 0.0, 2.0]);
        let a2 = leaf_scalar(0.3, true);
        let c2 = trunc(&a2).unwrap();
        c2.backward().unwrap();
        assert_scalar_approx(&a2.grad().unwrap().unwrap(), 0.0, 1e-6);
    }

    // frac
    #[test]
    fn test_frac_forward_and_pass_through_backward() {
        let a = leaf_vec(&[-1.5, 0.4, 2.75], false);
        let c = frac(&a).unwrap();
        let d = c.data().unwrap();
        // frac(-1.5) = -1.5 - trunc(-1.5) = -1.5 - (-1.0) = -0.5
        assert!((d[0] - (-0.5)).abs() < 1e-6);
        assert!((d[1] - 0.4).abs() < 1e-6);
        assert!((d[2] - 0.75).abs() < 1e-6);
        // backward passes grad through (slope=1).
        let a2 = leaf_scalar(0.3, true);
        let c2 = frac(&a2).unwrap();
        c2.backward().unwrap();
        assert_scalar_approx(&a2.grad().unwrap().unwrap(), 1.0, 1e-6);
    }

    // sign
    #[test]
    fn test_sign_forward_and_zero_backward() {
        let a = leaf_vec(&[-3.0, 0.0, 5.0], false);
        let c = sign(&a).unwrap();
        assert_eq!(c.data().unwrap(), &[-1.0, 0.0, 1.0]);
        // backward zero
        let a2 = leaf_scalar(5.0, true);
        let c2 = sign(&a2).unwrap();
        c2.backward().unwrap();
        assert_scalar_approx(&a2.grad().unwrap().unwrap(), 0.0, 1e-6);
    }

    // sinc
    #[test]
    fn test_sinc_zero_and_nonzero() {
        let a = leaf_vec(&[0.0, 0.5, 1.0], false);
        let c = sinc(&a).unwrap();
        let d = c.data().unwrap();
        // sinc(0) = 1
        assert!((d[0] - 1.0).abs() < 1e-6);
        // sinc(0.5) = sin(pi/2)/(pi/2) = 1/(pi/2) = 2/pi ≈ 0.6366
        let expected = (std::f32::consts::PI * 0.5).sin() / (std::f32::consts::PI * 0.5);
        assert!(
            (d[1] - expected).abs() < 1e-6,
            "sinc(0.5) = {} vs expected {}",
            d[1],
            expected
        );
        // sinc(1) = sin(pi)/(pi) ≈ 0
        assert!(d[2].abs() < 1e-6);
    }

    #[test]
    fn test_sinc_numerical_grad_interior() {
        let x = 0.5_f32;
        let a = leaf_scalar(x, true);
        let c = sinc(&a).unwrap();
        c.backward().unwrap();
        let g = a.grad().unwrap().unwrap().item().unwrap();
        let sinc_fn = |v: f32| {
            if v == 0.0 {
                1.0
            } else {
                let p = std::f32::consts::PI * v;
                p.sin() / p
            }
        };
        numerical_grad_check(sinc_fn, x, g, 1e-3);
    }

    #[test]
    fn test_sinc_zero_backward_is_zero() {
        let a = leaf_scalar(0.0, true);
        let c = sinc(&a).unwrap();
        c.backward().unwrap();
        // The closed-form sinc'(0) = 0.
        assert_scalar_approx(&a.grad().unwrap().unwrap(), 0.0, 1e-6);
    }

    // nextafter (#1335)

    /// f64 leaf vector helper for the bit-exact nextafter assertions.
    fn leaf_vec_f64(data: &[f64], requires_grad: bool) -> Tensor<f64> {
        Tensor::from_storage(
            TensorStorage::cpu(data.to_vec()),
            vec![data.len()],
            requires_grad,
        )
        .unwrap()
    }

    #[test]
    fn test_nextafter_matches_std_nextafter_f64() {
        // Oracle: std-library `f64::next_up`/`next_down` give the exact IEEE
        // one-ULP neighbours. We compare ferrotorch's `nextafter` against the
        // toolchain's bit-exact answer (the build toolchain is 1.85+ at test
        // time even though the *library* MSRV forbids using these in src/).
        let a = leaf_vec_f64(&[1.0, -1.0, 0.0, 0.0, 1e300], false);
        let b = leaf_vec_f64(&[2.0, -2.0, 1.0, -1.0, f64::INFINITY], false);
        let out = nextafter(&a, &b).unwrap();
        let d = out.data().unwrap();
        // toward larger -> next_up; 0.0 toward +1 -> smallest +subnormal.
        assert_eq!(d[0], 1.0_f64.next_up());
        // -1.0 toward -2.0 (more negative) -> next_down.
        assert_eq!(d[1], (-1.0_f64).next_down());
        assert_eq!(d[2], f64::from_bits(1));
        assert_eq!(d[3], -f64::from_bits(1));
        assert_eq!(d[4], 1e300_f64.next_up());
    }

    #[test]
    fn test_nextafter_equal_returns_b_and_nan_propagates() {
        // a == b -> result is b (carries b's sign for the signed-zero tie).
        let a = leaf_vec_f64(&[5.0, 0.0], false);
        let b = leaf_vec_f64(&[5.0, -0.0], false);
        let out = nextafter(&a, &b).unwrap();
        let d = out.data().unwrap();
        assert_eq!(d[0], 5.0);
        // 0.0 == -0.0, result is b == -0.0 (negative sign bit).
        assert!(d[1].is_sign_negative() && d[1] == 0.0);
        // NaN in either operand -> NaN.
        let an = leaf_vec_f64(&[f64::NAN, 1.0], false);
        let bn = leaf_vec_f64(&[1.0, f64::NAN], false);
        let outn = nextafter(&an, &bn).unwrap();
        let dn = outn.data().unwrap();
        assert!(dn[0].is_nan() && dn[1].is_nan());
    }

    #[test]
    fn test_nextafter_backward_passthrough_and_tie_mask() {
        // VJP per derivatives.yaml:1322-1324:
        //   self: where(self != other, grad, 0); other: zeros_like(other).
        // a[0] != b[0] -> grad passes through (1.0); a[1] == b[1] -> 0.
        let a = leaf_vec_f64(&[1.0, 3.0], true);
        let b = leaf_vec_f64(&[2.0, 3.0], true);
        let out = nextafter(&a, &b).unwrap();
        out.sum_all().unwrap().backward().unwrap();
        let ga = a.grad().unwrap().unwrap();
        let gad = ga.data().unwrap();
        assert_eq!(gad[0], 1.0);
        assert_eq!(gad[1], 0.0);
        // Gradient to `b` (direction operand) is identically zero.
        let gb = b.grad().unwrap().unwrap();
        for &v in gb.data().unwrap() {
            assert_eq!(v, 0.0);
        }
    }
}
