//! Backward functions for reduction operations: sum, mean, prod, amin,
//! amax, sum_dim, mean_dim, logsumexp, argmax, argmin, std, var, any,
//! all, count_nonzero.
//!
//! Each reduction collapses an input tensor along zero, one, or more
//! dimensions. The VJP (vector-Jacobian product) routes the upstream
//! gradient back to the input layout — broadcast for sum/mean,
//! prefix-suffix product for prod, softmax-weighted for logsumexp,
//! count-scaled for amin/amax, 2*(x-mean)/(n-c) for var/std, etc.
//! argmax/argmin/any/all/count_nonzero are integer- or bool-output and
//! NON-differentiable (no `derivatives.yaml` entry); they carry no
//! `*Backward` node.
//!
//! ## REQ status (per `.design/ferrotorch-core/grad_fns/reduction.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`sum`) | NOT-STARTED | `sum` + `SumBackward` ship with non-test production consumer `Tensor::sum_all` plus invocations across `autograd/grad_penalty.rs`, `einsum.rs`, `vmap.rs`, `flex_attention.rs`, and `ferrotorch-nn`; parity-sweep runner arm gated by #1314 (currently `0/80 passed (80 skipped)`). |
//! | REQ-2 (`mean`) | NOT-STARTED | `mean` + `MeanBackward` ship with non-test consumer `Tensor::mean_all`; parity runner arm gated by #1314. |
//! | REQ-3 (`prod`) | NOT-STARTED | `prod` + `ProdBackward` (prefix-suffix product, zero-aware) ship with consumer `Tensor::prod_all`; parity runner arm gated by #1314. |
//! | REQ-4 (`amin` / `amax`) | SHIPPED | `amin` / `amax` + `AminBackward` / `AmaxBackward` implement PyTorch NaN propagation, tie-split gradients, CUDA f32/f64 forward/backward, and empty-global errors; consumers `Tensor::amin` / `Tensor::amax`; regression coverage in `divergence_global_extrema_reduction_parity.rs`. |
//! | REQ-5 (`sum_dim`) | NOT-STARTED | `sum_dim` + `SumDimBackward` ship; consumed by `einsum.rs`, `einops.rs`, `grad_fns/linalg.rs`, `meta_propagate.rs`, `ferrotorch-distributions`; parity runner arm gated by #1314. |
//! | REQ-6 (`mean_dim`) | NOT-STARTED | `mean_dim` + `MeanDimBackward` ship; consumed by `meta_propagate.rs`; parity runner arm gated by #1314. |
//! | REQ-7 (backward VJP wiring) | NOT-STARTED | every `*Backward` struct implements the `GradFn` trait with `backward` / `inputs` / `name`; the no-grad / `requires_grad=false` short-circuit is exercised by the `test_*_no_grad*` family; gated by #1314 closing the parity runner arms. |
//! | REQ-8 (`std` / `var`) | NOT-STARTED | no `WelfordBackward` in this file. Blocker #1301. |
//! | REQ-9 (`max(dim)` / `min(dim)` with `(values, indices)`) | SHIPPED | `max_with_dim` / `min_with_dim` return `(Tensor<T>, IntTensor<i64>)`; shared `MaxMinDimBackward` scatters grad at saved per-slice argmax/argmin; NaN-poisoning per upstream `SharedReduceOps.h:26-34`; consumed by `lib.rs:182` re-export; closes #1302. |
//! | REQ-10 (`argmax` / `argmin`) | NOT-STARTED | integer-output, non-differentiable; no integer-output reduction scaffold. Blocker #1304. |
//! | REQ-11 (`median` / `nanmedian`) | SHIPPED | `median_with_dim` / `nanmedian_with_dim` return `(Tensor<T>, IntTensor<i64>)`; lower-median rank `(effective-1)/2` over a stable index sort with the upstream `ip[i]<ip[j] || (==&&i<j)` tie-break per `Sorting.cpp:503-607`; median NaN-poisons, nanmedian skips NaNs; shared `MaxMinDimBackward` scatters grad at the saved per-slice median index per `derivatives.yaml:1179-1185`; consumed by `lib.rs:181` re-export; closes #1306. |
//! | REQ-12 (`norm`) | SHIPPED | `norm_with_dim(input, p, dim, keepdim)` + `NormDimBackward` for `p > 0 finite` per `derivatives.yaml` `norm.ScalarOpt_dim`; `result==0 → 0` mask; consumed by `lib.rs:182` re-export; closes #1308. F32 L2 forward (`p==2.0`, `T==f32`, `inner==1`) reduces via `crate::simd_reduce::l2_norm_f32_torch` to match torch's vectorized last-dim L2 kernel byte-for-byte (#1614); generic f64 path retained for other p / dtype / strided cases. |
//! | REQ-13 (`logsumexp`) | NOT-STARTED | kernel-layer forward exists in `ops::elementwise`; no autograd wrapper here. Blocker #1310. |
//! | REQ-14 (`any` / `all` / `count_nonzero`) | NOT-STARTED | bool/integer-output, non-differentiable; no scaffold. Blocker #1312. |
//! | REQ-15 (parity-sweep runner arms) | NOT-STARTED | the runner has arms only for the five cumulative ops (owned by `grad_fns/cumulative.rs`); no arms for `sum`, `mean`, `prod`, `amin`, `amax` or any NOT-STARTED op above. Umbrella blocker #1314. |

use std::sync::Arc;

use crate::autograd::no_grad::is_grad_enabled;
use crate::bool_tensor::BoolTensor;
use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::int_tensor::IntTensor;
use crate::ops::elementwise;
use crate::storage::TensorStorage;
use crate::tensor::{GradFn, Tensor};

// ---------------------------------------------------------------------------
// SumBackward
// ---------------------------------------------------------------------------

/// Backward node for `sum(input) -> scalar`.
///
/// VJP: `grad_input[i] = grad_output` for all i (broadcast scalar to input shape).
#[derive(Debug)]
pub struct SumBackward<T: Float> {
    input: Tensor<T>,
}

impl<T: Float> GradFn<T> for SumBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // Extract the scalar value — works for both CPU and GPU by
        // transferring to CPU if needed (it's just one number).
        let go = if grad_output.is_cuda() {
            let cpu = grad_output.cpu()?;
            cpu.data()?[0]
        } else {
            grad_output.data()?[0]
        };
        let numel = self.input.numel();

        // GPU-native path: skip the `vec![go; numel]` CPU allocation +
        // upload by calling the on-device `fill` primitive. Falls back
        // to the CPU build + `.to(device)` for non-f32/f64 types or if
        // the backend hasn't been initialised.
        if self.input.is_cuda() {
            use crate::device::Device;
            use crate::gpu_dispatch::gpu_backend;
            use std::any::TypeId;
            let ordinal = match self.input.device() {
                Device::Cuda(o) => o,
                _ => 0,
            };
            let is_t_f32 = TypeId::of::<T>() == TypeId::of::<f32>();
            let is_t_f64 = TypeId::of::<T>() == TypeId::of::<f64>();
            if let Some(backend) = gpu_backend() {
                if is_t_f32 {
                    let scalar_f32: f32 = <T as num_traits::ToPrimitive>::to_f32(&go).unwrap();
                    let handle = backend.fill_f32(numel, scalar_f32, ordinal)?;
                    let grad_input = Tensor::from_storage(
                        TensorStorage::gpu(handle),
                        self.input.shape().to_vec(),
                        false,
                    )?;
                    return Ok(vec![Some(grad_input)]);
                } else if is_t_f64 {
                    let scalar_f64: f64 = <T as num_traits::ToPrimitive>::to_f64(&go).unwrap();
                    let handle = backend.fill_f64(numel, scalar_f64, ordinal)?;
                    let grad_input = Tensor::from_storage(
                        TensorStorage::gpu(handle),
                        self.input.shape().to_vec(),
                        false,
                    )?;
                    return Ok(vec![Some(grad_input)]);
                }
            }
        }
        // CPU / fallback path — legacy behaviour.
        let data = vec![go; numel];
        let grad_cpu =
            Tensor::from_storage(TensorStorage::cpu(data), self.input.shape().to_vec(), false)?;
        let grad_input = grad_cpu.to(self.input.device())?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "SumBackward"
    }
}

/// Differentiable sum reduction: returns a scalar that is the sum of all elements.
///
/// When gradient tracking is enabled and the input requires grad, the returned
/// tensor carries a [`SumBackward`] node.
pub fn sum<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) = crate::meta_propagate::reduce_all(input)? {
        return Ok(out);
    }
    crate::profiler_hook::profile_op_scope("sum", "reduction", &[input.shape()], || {
        sum_inner(input)
    })
}

fn sum_inner<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if input.is_cuda() {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        // #1658: normalise a narrowed-offset CUDA view to a packed offset-0
        // buffer before the full reduction reads element 0.
        let input = input.contiguous()?;
        // #23: bf16 now has a real GPU sum kernel (f32 accumulator,
        // bf16 RNE store back).
        let handle: crate::gpu_dispatch::GpuBufferHandle = crate::dispatch_floating_dtype!(
            T,
            "sum",
            f32 => backend.sum_f32(input.gpu_handle()?, input.numel()),
            f64 => backend.sum_f64(input.gpu_handle()?, input.numel()),
            bf16 => backend.sum_bf16_bf16(input.gpu_handle()?),
            f16 => backend.sum_f16(input.gpu_handle()?),
        )?;
        let storage = TensorStorage::gpu(handle);
        let shape = vec![];

        if is_grad_enabled() && input.requires_grad() {
            let grad_fn = Arc::new(SumBackward {
                input: input.clone(),
            });
            Tensor::from_operation(storage, shape, grad_fn)
        } else {
            Tensor::from_storage(storage, shape, false)
        }
    } else {
        let result = elementwise::sum(input)?;

        if is_grad_enabled() && input.requires_grad() {
            let grad_fn = Arc::new(SumBackward {
                input: input.clone(),
            });
            let (storage, shape) = result.into_storage_and_shape()?;
            Tensor::from_operation(storage, shape, grad_fn)
        } else {
            Ok(result)
        }
    }
}

// ---------------------------------------------------------------------------
// MeanBackward
// ---------------------------------------------------------------------------

/// Backward node for `mean(input) -> scalar`.
///
/// VJP: `grad_input[i] = grad_output / numel` for all i.
#[derive(Debug)]
pub struct MeanBackward<T: Float> {
    input: Tensor<T>,
}

impl<T: Float> GradFn<T> for MeanBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let go = if grad_output.is_cuda() {
            let cpu = grad_output.cpu()?;
            cpu.data()?[0]
        } else {
            grad_output.data()?[0]
        };
        let numel = self.input.numel();
        let n = T::from(numel).unwrap();
        let val = go / n;

        // GPU-native path mirrors SumBackward: use on-device fill
        // instead of allocating `vec![val; numel]` on CPU and uploading.
        if self.input.is_cuda() {
            use crate::device::Device;
            use crate::gpu_dispatch::gpu_backend;
            use std::any::TypeId;
            let ordinal = match self.input.device() {
                Device::Cuda(o) => o,
                _ => 0,
            };
            let is_t_f32 = TypeId::of::<T>() == TypeId::of::<f32>();
            let is_t_f64 = TypeId::of::<T>() == TypeId::of::<f64>();
            if let Some(backend) = gpu_backend() {
                if is_t_f32 {
                    let scalar_f32: f32 = <T as num_traits::ToPrimitive>::to_f32(&val).unwrap();
                    let handle = backend.fill_f32(numel, scalar_f32, ordinal)?;
                    let grad_input = Tensor::from_storage(
                        TensorStorage::gpu(handle),
                        self.input.shape().to_vec(),
                        false,
                    )?;
                    return Ok(vec![Some(grad_input)]);
                } else if is_t_f64 {
                    let scalar_f64: f64 = <T as num_traits::ToPrimitive>::to_f64(&val).unwrap();
                    let handle = backend.fill_f64(numel, scalar_f64, ordinal)?;
                    let grad_input = Tensor::from_storage(
                        TensorStorage::gpu(handle),
                        self.input.shape().to_vec(),
                        false,
                    )?;
                    return Ok(vec![Some(grad_input)]);
                }
            }
        }
        let data = vec![val; numel];
        let grad_cpu =
            Tensor::from_storage(TensorStorage::cpu(data), self.input.shape().to_vec(), false)?;
        let grad_input = grad_cpu.to(self.input.device())?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "MeanBackward"
    }
}

/// Differentiable mean reduction: returns a scalar that is the mean of all elements.
///
/// When gradient tracking is enabled and the input requires grad, the returned
/// tensor carries a [`MeanBackward`] node.
pub fn mean<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) = crate::meta_propagate::reduce_all(input)? {
        return Ok(out);
    }
    crate::profiler_hook::profile_op_scope("mean", "reduction", &[input.shape()], || {
        mean_inner(input)
    })
}

fn mean_inner<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    // GPU path: use GPU sum kernel + scalar divide (avoids CPU round-trip).
    let result = if input.is_cuda() {
        if let Some(backend) = crate::gpu_dispatch::gpu_backend() {
            // #1658: normalise a narrowed-offset CUDA view to a packed offset-0
            // buffer before the full reduction reads element 0.
            let input = input.contiguous()?;
            // #23: dispatch by dtype. bf16 routes to `mean_bf16_bf16`
            // (sum_bf16 + on-device scale by 1/n, both in f32 accumulator).
            let mean_handle: crate::gpu_dispatch::GpuBufferHandle = crate::dispatch_floating_dtype!(
                T,
                "mean",
                f32 => {
                    // mean = sum / n. The sum is a 1-element on-device scalar;
                    // scale it by 1/n with the host-scalar `scale_f32` kernel
                    // (one launch, no H2D upload) rather than uploading a
                    // 1-element `1/n` buffer and running a full `mul` kernel.
                    let sum_handle = backend.sum_f32(input.gpu_handle()?, input.numel())?;
                    let inv_n = 1.0f32 / input.numel() as f32;
                    Ok::<_, crate::error::FerrotorchError>(backend.scale_f32(&sum_handle, inv_n)?)
                },
                f64 => {
                    let sum_handle = backend.sum_f64(input.gpu_handle()?, input.numel())?;
                    let inv_n = 1.0f64 / input.numel() as f64;
                    Ok::<_, crate::error::FerrotorchError>(backend.scale_f64(&sum_handle, inv_n)?)
                },
                bf16 => Ok::<_, crate::error::FerrotorchError>(
                    backend.mean_bf16_bf16(input.gpu_handle()?)?
                ),
                f16 => Ok::<_, crate::error::FerrotorchError>(
                    backend.mean_f16(input.gpu_handle()?)?
                ),
            )?;
            Tensor::from_storage(TensorStorage::gpu(mean_handle), vec![], false)?
        } else {
            elementwise::mean(input)?
        }
    } else {
        elementwise::mean(input)?
    };

    if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(MeanBackward {
            input: input.clone(),
        });
        let (storage, shape) = result.into_storage_and_shape()?;
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// ProdBackward
// ---------------------------------------------------------------------------

/// Backward node for `prod(input) -> scalar`.
///
/// VJP: `grad_input[i] = grad_output * prod(input) / input[i]`.
///
/// When any `input[i]` is zero, we recompute the partial product excluding
/// that element to avoid division by zero. This is done via prefix/suffix
/// products.
#[derive(Debug)]
pub struct ProdBackward<T: Float> {
    input: Tensor<T>,
}

impl<T: Float> GradFn<T> for ProdBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // GPU fast path (#785): on-device prefix-suffix kernel handles
        // all zero cases (no zero, single zero, multi-zero) with a
        // single launch — no host detour for the gradient values.
        let t_is_f32 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>();
        let t_is_f64 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>();
        if self.input.is_cuda()
            && (t_is_f32 || t_is_f64)
            && let Some(backend) = crate::gpu_dispatch::gpu_backend()
        {
            // grad_output may live on CPU when constructed by
            // the autograd seed; upload it to the same device as
            // the input so the kernel can read it directly.
            let go_on_device = if grad_output.is_cuda() {
                grad_output.clone()
            } else {
                grad_output.to(self.input.device())?
            };
            let grad_handle = if t_is_f32 {
                backend.prod_backward_f32(self.input.gpu_handle()?, go_on_device.gpu_handle()?)?
            } else {
                backend.prod_backward_f64(self.input.gpu_handle()?, go_on_device.gpu_handle()?)?
            };
            let storage = TensorStorage::gpu(grad_handle);
            let grad_input = Tensor::from_storage(storage, self.input.shape().to_vec(), false)?;
            return Ok(vec![Some(grad_input)]);
        }

        if self.input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "prod backward",
            });
        }

        let go = grad_output.data()?[0];
        let input_data = self.input.data_vec()?;
        let n = input_data.len();

        // Use prefix/suffix products to avoid division by zero.
        // prefix[i] = product of input[0..i]
        // suffix[i] = product of input[i+1..n]
        // grad[i] = go * prefix[i] * suffix[i]
        let mut prefix = vec![<T as num_traits::One>::one(); n];
        for i in 1..n {
            prefix[i] = prefix[i - 1] * input_data[i - 1];
        }

        let mut suffix = vec![<T as num_traits::One>::one(); n];
        if n > 1 {
            for i in (0..n - 1).rev() {
                suffix[i] = suffix[i + 1] * input_data[i + 1];
            }
        }

        let grad_data: Vec<T> = (0..n).map(|i| go * prefix[i] * suffix[i]).collect();

        let grad_cpu = Tensor::from_storage(
            TensorStorage::cpu(grad_data),
            self.input.shape().to_vec(),
            false,
        )?;
        // Place gradient on the same device as the input.
        let grad_input = grad_cpu.to(self.input.device())?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "ProdBackward"
    }
}

/// Differentiable product reduction: returns a scalar that is the product
/// of all elements.
///
/// When gradient tracking is enabled and the input requires grad, the returned
/// tensor carries a [`ProdBackward`] node.
pub fn prod<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) = crate::meta_propagate::reduce_all(input)? {
        return Ok(out);
    }
    crate::profiler_hook::profile_op_scope("prod", "reduction", &[input.shape()], || {
        prod_inner(input)
    })
}

fn prod_inner<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let t_is_f32 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>();
    let t_is_f64 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>();

    // GPU path: native reduce_prod kernel (#524).
    if input.is_cuda() && (t_is_f32 || t_is_f64) {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        // #1658: normalise a narrowed-offset CUDA view to a packed offset-0
        // buffer before the reduction reads element 0. Shadowing here also makes
        // the `ProdBackward` capture the packed buffer (the VJP reads input
        // values, so the stored input must honour the offset too).
        let input = input.contiguous()?;
        let handle = if t_is_f32 {
            backend.prod_f32(input.gpu_handle()?, input.numel())?
        } else {
            backend.prod_f64(input.gpu_handle()?, input.numel())?
        };
        let storage = TensorStorage::gpu(handle);
        if is_grad_enabled() && input.requires_grad() {
            let grad_fn = Arc::new(ProdBackward {
                input: input.clone(),
            });
            return Tensor::from_operation(storage, vec![], grad_fn);
        }
        return Tensor::from_storage(storage, vec![], false);
    }
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "prod" });
    }
    let data = input.data_vec()?;
    let total = data
        .iter()
        .copied()
        .fold(<T as num_traits::One>::one(), |a, b| a * b);
    let result = Tensor::from_storage(TensorStorage::cpu(vec![total]), vec![], false)?;

    if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(ProdBackward {
            input: input.clone(),
        });
        let (storage, shape) = result.into_storage_and_shape()?;
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// AminBackward / AmaxBackward (#627)
// ---------------------------------------------------------------------------
//
// `amin` / `amax` reduce a tensor to the global min/max scalar. These are
// the closed-form `torch.amin` / `torch.amax` ops. The backward routes
// the gradient to every input position equal to the extremum (subgradient
// at ties), matching torch's behavior.

fn torch_global_extreme<T: Float>(data: &[T], find_max: bool) -> FerrotorchResult<T> {
    if data.is_empty() {
        return Err(FerrotorchError::InvalidArgument {
            message: "amin/amax: reduction over an empty tensor requires an explicit dim".into(),
        });
    }

    let mut acc = data[0];
    for &v in &data[1..] {
        let take = v.is_nan() || (!acc.is_nan() && if find_max { v > acc } else { v < acc });
        if take {
            acc = v;
        }
    }
    Ok(acc)
}

fn torch_global_extreme_backward_cpu<T: Float>(
    input: &Tensor<T>,
    extreme: T,
    grad_output: T,
) -> FerrotorchResult<Tensor<T>> {
    let input_data = input.data_vec()?;
    let result = if extreme.is_nan() {
        vec![T::nan(); input_data.len()]
    } else {
        let zero = <T as num_traits::Zero>::zero();
        let count = input_data.iter().filter(|&&v| v == extreme).count();
        let scale = if count == 0 {
            T::nan()
        } else {
            grad_output
                / T::from(count).ok_or_else(|| FerrotorchError::InvalidArgument {
                    message: "amin/amax backward: match count is not representable".into(),
                })?
        };
        input_data
            .iter()
            .map(|&v| if v == extreme { scale } else { zero })
            .collect()
    };
    Tensor::from_storage(TensorStorage::cpu(result), input.shape().to_vec(), false)
}

#[derive(Debug)]
pub struct AminBackward<T: Float> {
    input: Tensor<T>,
    extreme: Tensor<T>,
}

impl<T: Float> GradFn<T> for AminBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let is_f32 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>();
        let is_f64 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>();
        if self.input.is_cuda() && (is_f32 || is_f64) {
            let backend =
                crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let grad_output = if grad_output.is_cuda() {
                grad_output.clone()
            } else {
                grad_output.to(self.input.device())?
            };
            let handle = if is_f32 {
                backend.extreme_backward_f32(
                    self.input.gpu_handle()?,
                    self.extreme.gpu_handle()?,
                    grad_output.gpu_handle()?,
                )?
            } else {
                backend.extreme_backward_f64(
                    self.input.gpu_handle()?,
                    self.extreme.gpu_handle()?,
                    grad_output.gpu_handle()?,
                )?
            };
            let grad_input = Tensor::from_storage(
                TensorStorage::gpu(handle),
                self.input.shape().to_vec(),
                false,
            )?;
            return Ok(vec![Some(grad_input)]);
        }

        let go = grad_output.data()?[0];
        let mn = self.extreme.data()?[0];
        let grad_input = torch_global_extreme_backward_cpu(&self.input, mn, go)?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "AminBackward"
    }
}

#[derive(Debug)]
pub struct AmaxBackward<T: Float> {
    input: Tensor<T>,
    extreme: Tensor<T>,
}

impl<T: Float> GradFn<T> for AmaxBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let is_f32 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>();
        let is_f64 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>();
        if self.input.is_cuda() && (is_f32 || is_f64) {
            let backend =
                crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let grad_output = if grad_output.is_cuda() {
                grad_output.clone()
            } else {
                grad_output.to(self.input.device())?
            };
            let handle = if is_f32 {
                backend.extreme_backward_f32(
                    self.input.gpu_handle()?,
                    self.extreme.gpu_handle()?,
                    grad_output.gpu_handle()?,
                )?
            } else {
                backend.extreme_backward_f64(
                    self.input.gpu_handle()?,
                    self.extreme.gpu_handle()?,
                    grad_output.gpu_handle()?,
                )?
            };
            let grad_input = Tensor::from_storage(
                TensorStorage::gpu(handle),
                self.input.shape().to_vec(),
                false,
            )?;
            return Ok(vec![Some(grad_input)]);
        }

        let go = grad_output.data()?[0];
        let mx = self.extreme.data()?[0];
        let grad_input = torch_global_extreme_backward_cpu(&self.input, mx, go)?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "AmaxBackward"
    }
}

/// Differentiable global minimum reduction. Mirrors `torch.amin(input)`
/// with no `dim` argument: returns a 0-d tensor holding the smallest
/// element. On CUDA f32/f64, dispatches to the native PTX
/// `gpu_reduce_min` kernel; on CPU and other dtypes, walks the buffer.
/// Backward routes the upstream grad to every input element equal to
/// the min (subgradient at ties). (#627)
pub fn amin<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let is_f32 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>();
    let is_f64 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>();

    if input.numel() == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "amin: reduction over an empty tensor requires an explicit dim".into(),
        });
    }

    if input.is_cuda() && (is_f32 || is_f64) {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        // #1658: normalise a narrowed-offset CUDA view to a packed offset-0
        // buffer before the reduction reads element 0 (and so the AminBackward
        // subgradient compares against the correct logical input values).
        let input = input.contiguous()?;
        let handle = if is_f32 {
            backend.min_f32(input.gpu_handle()?, input.numel())?
        } else {
            backend.min_f64(input.gpu_handle()?, input.numel())?
        };
        let result = Tensor::from_storage(TensorStorage::gpu(handle), vec![], false)?;
        if is_grad_enabled() && input.requires_grad() {
            let grad_fn = Arc::new(AminBackward {
                input: input.clone(),
                extreme: result.clone(),
            });
            let (storage, shape) = result.into_storage_and_shape()?;
            return Tensor::from_operation(storage, shape, grad_fn);
        }
        return Ok(result);
    }
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "amin" });
    }
    // CPU walk.
    let data = input.data_vec()?;
    let mn = torch_global_extreme(&data, false)?;
    let result = Tensor::from_storage(TensorStorage::cpu(vec![mn]), vec![], false)?;
    if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(AminBackward {
            input: input.clone(),
            extreme: result.clone(),
        });
        let (storage, shape) = result.into_storage_and_shape()?;
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Ok(result)
    }
}

/// Differentiable global maximum reduction. Counterpart of [`amin`].
/// Mirrors `torch.amax(input)`. (#627)
pub fn amax<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let is_f32 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>();
    let is_f64 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>();

    if input.numel() == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "amax: reduction over an empty tensor requires an explicit dim".into(),
        });
    }

    if input.is_cuda() && (is_f32 || is_f64) {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        // #1658: normalise a narrowed-offset CUDA view to a packed offset-0
        // buffer before the reduction reads element 0 (and so the AmaxBackward
        // subgradient compares against the correct logical input values).
        let input = input.contiguous()?;
        let handle = if is_f32 {
            backend.max_f32(input.gpu_handle()?, input.numel())?
        } else {
            backend.max_f64(input.gpu_handle()?, input.numel())?
        };
        let result = Tensor::from_storage(TensorStorage::gpu(handle), vec![], false)?;
        if is_grad_enabled() && input.requires_grad() {
            let grad_fn = Arc::new(AmaxBackward {
                input: input.clone(),
                extreme: result.clone(),
            });
            let (storage, shape) = result.into_storage_and_shape()?;
            return Tensor::from_operation(storage, shape, grad_fn);
        }
        return Ok(result);
    }
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "amax" });
    }
    let data = input.data_vec()?;
    let mx = torch_global_extreme(&data, true)?;
    let result = Tensor::from_storage(TensorStorage::cpu(vec![mx]), vec![], false)?;
    if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(AmaxBackward {
            input: input.clone(),
            extreme: result.clone(),
        });
        let (storage, shape) = result.into_storage_and_shape()?;
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// SumDimBackward
// ---------------------------------------------------------------------------

/// Backward node for `sum_dim(input, dim, keepdim) -> reduced tensor`.
///
/// VJP: expand the gradient back along the reduced dimension to match the
/// input shape. If `keepdim` was false, we first unsqueeze the reduced dim
/// before expanding.
#[derive(Debug)]
pub struct SumDimBackward<T: Float> {
    input: Tensor<T>,
    dim: usize,
    keepdim: bool,
}

impl<T: Float> GradFn<T> for SumDimBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let input_shape = self.input.shape();
        let outer: usize = input_shape[..self.dim].iter().product::<usize>().max(1);
        let inner: usize = input_shape[(self.dim + 1)..]
            .iter()
            .product::<usize>()
            .max(1);
        let repeat_count = input_shape[self.dim];

        // GPU-native path: expand-along-dim via the dedicated kernel (#524).
        let t_is_f32 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>();
        let t_is_f64 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>();
        if grad_output.is_cuda() && (t_is_f32 || t_is_f64) {
            let backend =
                crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let result_h = if t_is_f32 {
                backend.repeat_along_dim_f32(
                    grad_output.gpu_handle()?,
                    outer,
                    repeat_count,
                    inner,
                )?
            } else {
                backend.repeat_along_dim_f64(
                    grad_output.gpu_handle()?,
                    outer,
                    repeat_count,
                    inner,
                )?
            };
            let grad_input =
                Tensor::from_storage(TensorStorage::gpu(result_h), input_shape.to_vec(), false)?;
            return Ok(vec![Some(grad_input)]);
        }
        if grad_output.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "sum_dim backward",
            });
        }

        // If keepdim was false, reinsert the reduced dimension as size 1.
        let grad = if self.keepdim {
            grad_output.clone()
        } else {
            let mut unsqueezed_shape = grad_output.shape().to_vec();
            unsqueezed_shape.insert(self.dim, 1);
            let data = grad_output.data()?.to_vec();
            Tensor::from_storage(TensorStorage::cpu(data), unsqueezed_shape, false)?
        };

        // Now expand (repeat) along the reduced dim to match input shape.
        let grad_data = grad.data()?;
        let grad_shape = grad.shape();

        let out_numel: usize = input_shape.iter().product();
        let mut result = Vec::with_capacity(out_numel);

        for flat in 0..out_numel {
            // Decompose flat index into input coords.
            let mut rem = flat;
            let mut coords = vec![0usize; input_shape.len()];
            for d in (0..input_shape.len()).rev() {
                coords[d] = rem % input_shape[d];
                rem /= input_shape[d];
            }
            // Map to grad index: the reduced dim coordinate becomes 0 (size 1 in grad).
            let mut grad_flat = 0usize;
            let mut stride = 1usize;
            for d in (0..grad_shape.len()).rev() {
                let c = if d == self.dim { 0 } else { coords[d] };
                grad_flat += c * stride;
                stride *= grad_shape[d];
            }
            result.push(grad_data[grad_flat]);
        }

        let grad_input =
            Tensor::from_storage(TensorStorage::cpu(result), input_shape.to_vec(), false)?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "SumDimBackward"
    }
}

/// Sum along a specific dimension.
///
/// If `keepdim` is true, the output tensor has the reduced dimension with size 1.
/// If `keepdim` is false, the reduced dimension is removed.
///
/// `dim` supports negative indexing: `-1` means the last dimension.
pub fn sum_dim<T: Float>(
    input: &Tensor<T>,
    dim: i64,
    keepdim: bool,
) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) = crate::meta_propagate::reduce_dim(input, dim, keepdim)? {
        return Ok(out);
    }
    crate::profiler_hook::profile_op_scope("sum_dim", "reduction", &[input.shape()], || {
        sum_dim_inner(input, dim, keepdim)
    })
}

/// Fast single-dim sum of a CONTIGUOUS row-major buffer, decomposed as
/// `[outer, axis, inner]`:
///   `outer = prod(in_shape[..norm_dim])`,
///   `axis  = in_shape[norm_dim]`,
///   `inner = prod(in_shape[norm_dim+1..])`.
/// Element `(o, a, i)` of the input lives at flat `(o*axis + a)*inner + i` and
/// accumulates into `accum[o*inner + i]` (the keepdim-shape accumulator, axis
/// extent collapsed to 1, `numel == outer*inner`).
///
/// Two regimes mirror torch's two contiguous-reduction kernels
/// (`aten/src/ATen/native/cpu/Reduce.h`):
///
/// 1. `inner > 1` (reduce a non-last dim, e.g. dim 0 of `[1000,1000]`): the hot
///    inner loop `accum[ab+i] += in[ib+i]` runs over CONTIGUOUS `i` writing to
///    DISTINCT accumulator slots, so the lanes are independent and the
///    autovectorizer (this crate sets `target-cpu=native`) lowers it to AVX
///    adds. This is torch's `vectorized_outer_reduction` (`Reduce.h:94-113`),
///    which accumulates lane-wise down each column into the output row.
///
/// 2. `inner == 1` (reduce the contiguous last dim, e.g. dim 1 of
///    `[1000,1000]`): each output element is the horizontal sum of one
///    contiguous slice `in[o*axis .. o*axis+axis]`. A scalar `iter().sum()` does
///    NOT autovectorize (sequential FP-add dependency), so f32/f64 rows go
///    through `simd_reduce::sum_f32` / `sum_f64` (the lane-grouped
///    multi-accumulator), torch's `vectorized_inner_reduction`
///    (`Reduce.h:80-91`). Other dtypes use a correct scalar fold (not the perf
///    target).
///
/// Returns the accumulator (length `outer*inner`).
fn reduce_axis_sum_contiguous<T: Float>(
    in_data: &[T],
    outer: usize,
    axis: usize,
    inner: usize,
) -> Vec<T> {
    let accum_numel = outer * inner;
    let mut accum = vec![<T as num_traits::Zero>::zero(); accum_numel];

    if inner == 1 {
        // Regime 2: each output row is a horizontal sum of `axis` contiguous
        // input elements. f32/f64 route through the SIMD multi-accumulator.
        let t_is_f32 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>();
        let t_is_f64 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>();
        if t_is_f32 {
            // T is f32 here, so `to_f32` is the identity conversion — no
            // precision loss, just a typed view we can hand to the typed SIMD
            // kernel without an `unsafe` transmute. The scratch buffer is
            // allocated ONCE and reused across rows (per-row alloc would dominate
            // a tall-thin reduction like [1000,1000] dim=1).
            let mut buf: Vec<f32> = vec![0.0; axis];
            for (o, slot) in accum.iter_mut().enumerate() {
                let row = &in_data[o * axis..o * axis + axis];
                for (b, &v) in buf.iter_mut().zip(row.iter()) {
                    *b = num_traits::ToPrimitive::to_f32(&v).unwrap_or(0.0);
                }
                let s = crate::simd_reduce::sum_f32(&buf);
                *slot = <T as num_traits::NumCast>::from(s).unwrap_or(*slot);
            }
        } else if t_is_f64 {
            let mut buf: Vec<f64> = vec![0.0; axis];
            for (o, slot) in accum.iter_mut().enumerate() {
                let row = &in_data[o * axis..o * axis + axis];
                for (b, &v) in buf.iter_mut().zip(row.iter()) {
                    *b = num_traits::ToPrimitive::to_f64(&v).unwrap_or(0.0);
                }
                let s = crate::simd_reduce::sum_f64(&buf);
                *slot = <T as num_traits::NumCast>::from(s).unwrap_or(*slot);
            }
        } else {
            // Generic correct scalar fold (f16/bf16/other T — not the perf
            // target; correctness only).
            for (o, slot) in accum.iter_mut().enumerate() {
                let row = &in_data[o * axis..o * axis + axis];
                let mut acc = <T as num_traits::Zero>::zero();
                for &v in row {
                    acc += v;
                }
                *slot = acc;
            }
        }
    } else {
        // Regime 1: `accum[o*inner + i] += in[(o*axis + a)*inner + i]`. The
        // innermost `i` loop over contiguous, distinct accumulator lanes
        // autovectorizes to AVX adds.
        for o in 0..outer {
            let ab = o * inner;
            for a in 0..axis {
                let ib = (o * axis + a) * inner;
                let src = &in_data[ib..ib + inner];
                let dst = &mut accum[ab..ab + inner];
                for (acc_i, &v) in dst.iter_mut().zip(src.iter()) {
                    *acc_i += v;
                }
            }
        }
    }

    accum
}

/// Decompose a single normalised reduce dimension into `[outer, axis, inner]`
/// over a row-major shape. `outer = prod(shape[..norm_dim])`,
/// `axis = shape[norm_dim]`, `inner = prod(shape[norm_dim+1..])`.
fn outer_axis_inner(in_shape: &[usize], norm_dim: usize) -> (usize, usize, usize) {
    let outer: usize = in_shape[..norm_dim].iter().product();
    let axis: usize = in_shape[norm_dim];
    let inner: usize = in_shape[norm_dim + 1..].iter().product();
    (outer, axis, inner)
}

fn sum_dim_inner<T: Float>(
    input: &Tensor<T>,
    dim: i64,
    keepdim: bool,
) -> FerrotorchResult<Tensor<T>> {
    let ndim = input.ndim();
    if ndim == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "sum_dim: cannot reduce a scalar (0-D) tensor along a dimension".into(),
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
                "sum_dim: dim {dim} is out of bounds for tensor with {ndim} dimensions"
            ),
        });
    }

    let in_shape = input.shape();

    // Compute output shape.
    let mut out_shape: Vec<usize> = in_shape.to_vec();
    if keepdim {
        out_shape[norm_dim] = 1;
    } else {
        out_shape.remove(norm_dim);
    }

    // GPU path: use sum_axis kernel (no CPU round-trip).
    if input.is_cuda() {
        if let Some(backend) = crate::gpu_dispatch::gpu_backend() {
            // #1658: normalise a narrowed-offset CUDA view to a packed offset-0
            // buffer before the strided sum_axis kernel reads element 0.
            let input = input.contiguous()?;
            // #23: bf16 routes through `sum_axis_bf16_bf16` (f32 accumulator,
            // bf16 RNE store-back). The shape+axis signature is identical
            // across all three dtypes.
            let handle: crate::gpu_dispatch::GpuBufferHandle = crate::dispatch_floating_dtype!(
                T,
                "sum_dim",
                f32 => backend.sum_axis_f32(input.gpu_handle()?, in_shape, norm_dim),
                f64 => backend.sum_axis_f64(input.gpu_handle()?, in_shape, norm_dim),
                bf16 => backend.sum_axis_bf16_bf16(input.gpu_handle()?, in_shape, norm_dim),
                f16 => backend.sum_axis_f16(input.gpu_handle()?, in_shape, norm_dim),
            )?;

            let storage = TensorStorage::gpu(handle);
            return if is_grad_enabled() && input.requires_grad() {
                let grad_fn = Arc::new(SumDimBackward {
                    input: input.clone(),
                    dim: norm_dim,
                    keepdim,
                });
                Tensor::from_operation(storage, out_shape, grad_fn)
            } else {
                Tensor::from_storage(storage, out_shape, false)
            };
        }
        return Err(FerrotorchError::DeviceUnavailable);
    }
    let input_ref = if input.is_contiguous() {
        input.clone()
    } else {
        input.contiguous()?
    };
    let in_data = input_ref.data()?;

    // `input_ref` is contiguous (cloned or materialised above), so decompose
    // the single reduced dim as `[outer, axis, inner]` and run the AVX-friendly
    // fast accumulate (regime 1 lane-add for inner>1, SIMD horizontal sum for
    // inner==1). This replaces the prior per-element odometer scan (~10ms on a
    // [1000,1000] reduction) — see `reduce_axis_sum_contiguous`. The result is
    // the keepdim-shape accumulator with the reduced axis collapsed to 1.
    let (outer, axis, inner) = outer_axis_inner(in_shape, norm_dim);
    let accum = reduce_axis_sum_contiguous(&in_data[..input.numel()], outer, axis, inner);

    let device = input.device();
    if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(SumDimBackward {
            input: input.clone(),
            dim: norm_dim,
            keepdim,
        });
        let storage = TensorStorage::on_device(accum, device)?;
        Tensor::from_operation(storage, out_shape, grad_fn)
    } else {
        let storage = TensorStorage::on_device(accum, device)?;
        Tensor::from_storage(storage, out_shape, false)
    }
}

// ---------------------------------------------------------------------------
// MeanDimBackward
// ---------------------------------------------------------------------------

/// Backward node for `mean_dim(input, dim, keepdim) -> reduced tensor`.
///
/// VJP: expand the gradient back along the reduced dimension and divide by
/// the size of that dimension.
#[derive(Debug)]
pub struct MeanDimBackward<T: Float> {
    input: Tensor<T>,
    dim: usize,
    keepdim: bool,
}

impl<T: Float> GradFn<T> for MeanDimBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let input_shape = self.input.shape();
        let dim_size = input_shape[self.dim];

        // GPU path: native expand-and-scale via fill + broadcast_mul.
        // Conceptually grad_input[..., j, ...] = grad_output[..., 0, ...] / N
        // for every j in the reduced dim. Implement that as:
        //   ones = fill(input_numel, 1/N)         shape: input_shape
        //   grad_input = broadcast_mul(ones, grad_output_keepdim)
        // grad_output_keepdim is grad_output with a size-1 dim re-inserted
        // when keepdim=false (free metadata change, no copy).
        let is_f32 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>();
        if grad_output.is_cuda()
            && is_f32
            && let Some(backend) = crate::gpu_dispatch::gpu_backend()
        {
            let grad_shape_keepdim: Vec<usize> = if self.keepdim {
                grad_output.shape().to_vec()
            } else {
                let mut s = grad_output.shape().to_vec();
                s.insert(self.dim, 1);
                s
            };
            let input_numel: usize = input_shape.iter().product();
            let inv_n = 1.0f32 / (dim_size as f32);
            // Use device 0 — current backend doesn't expose handle's
            // ordinal at this layer; the upstream GPU pipeline is
            // single-device for now. (Multi-device support lives in
            // a wider refactor; not blocking this.)
            let ones_handle = backend.fill_f32(input_numel, inv_n, 0)?;
            let grad_handle = grad_output.gpu_handle()?;
            let grad_input_handle = backend.broadcast_mul_f32(
                &ones_handle,
                grad_handle,
                input_shape,
                &grad_shape_keepdim,
                input_shape,
            )?;
            let storage = TensorStorage::gpu(grad_input_handle);
            let grad_input = Tensor::from_storage(storage, input_shape.to_vec(), false)?;
            return Ok(vec![Some(grad_input)]);
        }

        // f64 GPU path via the new repeat_along_dim kernel + scale (#524).
        let is_f64 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>();
        if grad_output.is_cuda()
            && is_f64
            && let Some(backend) = crate::gpu_dispatch::gpu_backend()
        {
            let outer: usize = input_shape[..self.dim].iter().product::<usize>().max(1);
            let inner: usize = input_shape[(self.dim + 1)..]
                .iter()
                .product::<usize>()
                .max(1);
            let repeat_count = dim_size;
            let expanded = backend.repeat_along_dim_f64(
                grad_output.gpu_handle()?,
                outer,
                repeat_count,
                inner,
            )?;
            // Scale by 1/repeat_count to get the mean's gradient.
            let scaled = backend.scale_f64(&expanded, 1.0 / repeat_count as f64)?;
            let grad_input =
                Tensor::from_storage(TensorStorage::gpu(scaled), input_shape.to_vec(), false)?;
            return Ok(vec![Some(grad_input)]);
        }

        if grad_output.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "mean_dim backward",
            });
        }

        let n = T::from(dim_size).unwrap();

        // If keepdim was false, reinsert the reduced dimension as size 1.
        let grad = if self.keepdim {
            grad_output.clone()
        } else {
            let mut unsqueezed_shape = grad_output.shape().to_vec();
            unsqueezed_shape.insert(self.dim, 1);
            let data = grad_output.data()?.to_vec();
            Tensor::from_storage(TensorStorage::cpu(data), unsqueezed_shape, false)?
        };

        // Expand along the reduced dim, dividing by dim_size.
        let grad_data = grad.data()?;
        let grad_shape = grad.shape();

        let out_numel: usize = input_shape.iter().product();
        let mut result = Vec::with_capacity(out_numel);

        for flat in 0..out_numel {
            let mut rem = flat;
            let mut coords = vec![0usize; input_shape.len()];
            for d in (0..input_shape.len()).rev() {
                coords[d] = rem % input_shape[d];
                rem /= input_shape[d];
            }
            let mut grad_flat = 0usize;
            let mut stride = 1usize;
            for d in (0..grad_shape.len()).rev() {
                let c = if d == self.dim { 0 } else { coords[d] };
                grad_flat += c * stride;
                stride *= grad_shape[d];
            }
            result.push(grad_data[grad_flat] / n);
        }

        let grad_input =
            Tensor::from_storage(TensorStorage::cpu(result), input_shape.to_vec(), false)?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "MeanDimBackward"
    }
}

/// Mean along a specific dimension.
///
/// If `keepdim` is true, the output tensor has the reduced dimension with size 1.
/// If `keepdim` is false, the reduced dimension is removed.
///
/// `dim` supports negative indexing: `-1` means the last dimension.
pub fn mean_dim<T: Float>(
    input: &Tensor<T>,
    dim: i64,
    keepdim: bool,
) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) = crate::meta_propagate::reduce_dim(input, dim, keepdim)? {
        return Ok(out);
    }
    crate::profiler_hook::profile_op_scope("mean_dim", "reduction", &[input.shape()], || {
        mean_dim_inner(input, dim, keepdim)
    })
}

fn mean_dim_inner<T: Float>(
    input: &Tensor<T>,
    dim: i64,
    keepdim: bool,
) -> FerrotorchResult<Tensor<T>> {
    let ndim = input.ndim();
    if ndim == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "mean_dim: cannot reduce a scalar (0-D) tensor along a dimension".into(),
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
                "mean_dim: dim {dim} is out of bounds for tensor with {ndim} dimensions"
            ),
        });
    }

    let in_shape = input.shape();
    let dim_size = in_shape[norm_dim];
    let n = T::from(dim_size).unwrap();

    // Compute output shape.
    let mut out_shape: Vec<usize> = in_shape.to_vec();
    if keepdim {
        out_shape[norm_dim] = 1;
    } else {
        out_shape.remove(norm_dim);
    }

    // GPU path: native sum-then-scale on the existing backend kernels.
    // Mirrors `sum_dim_inner`'s dispatch but composes a second op
    // (`scale_f32(1/dim_size)`) so the result stays on-device end-to-end.
    if input.is_cuda() {
        if let Some(backend) = crate::gpu_dispatch::gpu_backend() {
            // #1658: normalise a narrowed-offset CUDA view to a packed offset-0
            // buffer before the strided sum_axis kernel reads element 0.
            let input = input.contiguous()?;
            // #23: bf16 routes through `mean_axis_bf16_bf16` (f32 accumulator,
            // divide-by-axis-size + bf16 RNE store-back in the same kernel —
            // no separate `scale` op needed). f32/f64 keep the
            // sum + scale composition because their reductions return the
            // same dtype (no internal divide).
            let mean_handle: crate::gpu_dispatch::GpuBufferHandle = crate::dispatch_floating_dtype!(
                T,
                "mean_dim",
                f32 => {
                    let s = backend.sum_axis_f32(input.gpu_handle()?, in_shape, norm_dim)?;
                    Ok::<_, crate::error::FerrotorchError>(backend.scale_f32(&s, 1.0 / dim_size as f32)?)
                },
                f64 => {
                    let s = backend.sum_axis_f64(input.gpu_handle()?, in_shape, norm_dim)?;
                    Ok::<_, crate::error::FerrotorchError>(backend.scale_f64(&s, 1.0 / dim_size as f64)?)
                },
                bf16 => Ok::<_, crate::error::FerrotorchError>(
                    backend.mean_axis_bf16_bf16(input.gpu_handle()?, in_shape, norm_dim)?
                ),
                f16 => Ok::<_, crate::error::FerrotorchError>(
                    backend.mean_axis_f16(input.gpu_handle()?, in_shape, norm_dim)?
                ),
            )?;
            let storage = TensorStorage::gpu(mean_handle);
            return if is_grad_enabled() && input.requires_grad() {
                let grad_fn = Arc::new(MeanDimBackward {
                    input: input.clone(),
                    dim: norm_dim,
                    keepdim,
                });
                Tensor::from_operation(storage, out_shape, grad_fn)
            } else {
                Tensor::from_storage(storage, out_shape, false)
            };
        }
        return Err(FerrotorchError::DeviceUnavailable);
    }

    // The fast `[outer, axis, inner]` accumulate (shared with `sum_dim`)
    // requires a CONTIGUOUS row-major buffer, so normalise a strided / narrowed
    // view first (free clone when already contiguous).
    let input_ref = if input.is_contiguous() {
        input.clone()
    } else {
        input.contiguous()?
    };
    let in_data = input_ref.data()?;

    // mean_dim == sum_dim then scale by 1/axis. Reuse the same AVX-friendly
    // accumulate (regime 1 lane-add for inner>1, SIMD horizontal sum for
    // inner==1), then divide — replacing the prior per-element odometer scan.
    let (outer, axis, inner) = outer_axis_inner(in_shape, norm_dim);
    let mut accum = reduce_axis_sum_contiguous(&in_data[..input.numel()], outer, axis, inner);

    // Divide by dim size to get mean.
    for v in &mut accum {
        *v = *v / n;
    }

    if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(MeanDimBackward {
            input: input.clone(),
            dim: norm_dim,
            keepdim,
        });
        let result = Tensor::from_operation(TensorStorage::cpu(accum), out_shape, grad_fn)?;
        result.to(input.device())
    } else {
        let result = Tensor::from_storage(TensorStorage::cpu(accum), out_shape, false)?;
        result.to(input.device())
    }
}

// ---------------------------------------------------------------------------
// LogsumexpBackward (full reduction)
// ---------------------------------------------------------------------------
//
// `logsumexp(input)` reduces to a 0-D scalar `log(sum(exp(input)))`. Mirrors
// `Tensor logsumexp(const Tensor& self, IntArrayRef dims, bool keepdim)` at
// `aten/src/ATen/native/ReduceOps.cpp:1548-1559` (the full-reduction form
// passes `dims = []` and produces a scalar result). The VJP per
// `tools/autograd/derivatives.yaml:1052-1054`:
//
//   - name: logsumexp(Tensor self, int[1] dim, bool keepdim=False) -> Tensor
//     self: logsumexp_backward(grad, self, result, dim, keepdim)
//
// expands to `grad_input = grad * exp(input - result)` (softmax-weighted
// routing). The forward kernel lives at `ferrotorch-core/src/ops/elementwise.
// rs:1233 pub fn logsumexp` (numerically stable max-subtraction form).

/// Small helper to convert a `f64` to `T: Float` for backward formulas.
/// Returns a typed error instead of panicking when the value is outside
/// `T`'s representable range. Stays on the production-code side of the
/// anti-pattern gate (no `.unwrap()`). Used by the new logsumexp / std /
/// var / argmax / argmin / any / all / count_nonzero arms below.
#[inline]
fn float_from_f64<T: Float>(v: f64) -> FerrotorchResult<T> {
    <T as num_traits::NumCast>::from(v).ok_or(FerrotorchError::InvalidArgument {
        message: format!("reduction: value {v} not representable in target Float dtype"),
    })
}

/// Convert a `T: Float` to `f64` for accumulator math. Mirrors
/// `<T as num_traits::ToPrimitive>::to_f64`; the only failure mode is
/// when `T` is a custom Float whose `ToPrimitive` doesn't implement
/// `to_f64`, which the workspace's `Float` blanket impls never trigger.
#[inline]
fn to_f64<T: Float>(v: T) -> FerrotorchResult<f64> {
    <T as num_traits::ToPrimitive>::to_f64(&v).ok_or(FerrotorchError::InvalidArgument {
        message: "reduction: cannot convert Float to f64".into(),
    })
}

/// Backward node for `logsumexp(input) -> scalar`.
///
/// VJP: `grad_input[i] = grad_output * exp(input[i] - result_scalar)`.
#[derive(Debug)]
pub struct LogsumexpBackward<T: Float> {
    input: Tensor<T>,
    /// Saved forward output (a 0-D scalar). Storing the scalar lets the
    /// backward compute `exp(input - result)` without re-running the
    /// max-subtraction forward.
    result: Tensor<T>,
}

impl<T: Float> GradFn<T> for LogsumexpBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if self.input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "logsumexp backward",
            });
        }
        let go = grad_output.data()?[0];
        let r = self.result.data()?[0];
        let input_data = self.input.data()?;
        let grad_data: Vec<T> = input_data.iter().map(|&v| go * (v - r).exp()).collect();
        let grad_input = Tensor::from_storage(
            TensorStorage::cpu(grad_data),
            self.input.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "LogsumexpBackward"
    }
}

/// Differentiable full-reduction logsumexp: numerically stable
/// `log(sum(exp(input)))` collapsing to a 0-D scalar.
///
/// Mirrors `torch.logsumexp(input, dim=[], keepdim=False)`. The forward
/// kernel is in `ops::elementwise::logsumexp` (max-subtraction for
/// stability); the autograd VJP is `grad * exp(input - result)`.
pub fn logsumexp<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) = crate::meta_propagate::reduce_all(input)? {
        return Ok(out);
    }
    let result = elementwise::logsumexp(input)?;

    if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(LogsumexpBackward {
            input: input.clone(),
            result: result.clone(),
        });
        let (storage, shape) = result.into_storage_and_shape()?;
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// LogsumexpDimBackward
// ---------------------------------------------------------------------------

/// Backward node for `logsumexp(input, dim, keepdim) -> reduced tensor`.
///
/// VJP per `derivatives.yaml:1052-1054`:
///   `grad_input[..., j, ...] = grad_output[..., 0, ...] * exp(input[..., j, ...] - result[..., 0, ...])`
/// expanded from the keepdim-shaped grad / result along the reduced dim.
#[derive(Debug)]
pub struct LogsumexpDimBackward<T: Float> {
    input: Tensor<T>,
    /// Forward output (keepdim shape — even when forward squeezed it,
    /// the backward re-inserts the size-1 dim for the broadcast).
    result_keepdim: Tensor<T>,
    dim: usize,
}

impl<T: Float> GradFn<T> for LogsumexpDimBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if self.input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "logsumexp_dim backward",
            });
        }
        let input_shape = self.input.shape();
        let input_data = self.input.data()?;
        let result_data = self.result_keepdim.data()?;
        let result_shape = self.result_keepdim.shape();

        // If forward squeezed the reduced dim (keepdim=false), the
        // upstream grad has the same squeezed shape; we re-insert the
        // size-1 dim at position `self.dim` for the broadcast walk.
        let grad_keepdim_data = grad_output.data()?.to_vec();

        let in_numel: usize = input_shape.iter().product();
        let mut out = Vec::with_capacity(in_numel);
        for flat in 0..in_numel {
            // Decompose flat -> per-axis coords of the input.
            let mut rem = flat;
            let mut coords = vec![0usize; input_shape.len()];
            for d in (0..input_shape.len()).rev() {
                coords[d] = rem % input_shape[d];
                rem /= input_shape[d];
            }
            // Map to result_keepdim/grad_keepdim index (reduced-dim coord -> 0).
            let mut ki = 0usize;
            let mut ks = 1usize;
            for d in (0..result_shape.len()).rev() {
                let c = if d == self.dim { 0 } else { coords[d] };
                ki += c * ks;
                ks *= result_shape[d];
            }
            let r = result_data[ki];
            let g = grad_keepdim_data[ki];
            out.push(g * (input_data[flat] - r).exp());
        }
        let grad_input =
            Tensor::from_storage(TensorStorage::cpu(out), input_shape.to_vec(), false)?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "LogsumexpDimBackward"
    }
}

/// Differentiable dim-keyed logsumexp. Mirrors
/// `torch.logsumexp(input, dim, keepdim=False)`. Single-dim variant
/// (consumer chains for multi-dim; the route's parity-sweep `logsumexp`
/// op emits both single-int and list-of-int dim arguments — the runner
/// arm handles the multi-dim fan-out by repeated calls).
///
/// Negative dim is normalized (`(ndim + dim) as usize`); out-of-range
/// errors via the standard `InvalidArgument`.
pub fn logsumexp_dim<T: Float>(
    input: &Tensor<T>,
    dim: i64,
    keepdim: bool,
) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) = crate::meta_propagate::reduce_dim(input, dim, keepdim)? {
        return Ok(out);
    }
    let ndim = input.ndim();
    if ndim == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "logsumexp_dim: cannot reduce a 0-D tensor along a dimension".into(),
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
                "logsumexp_dim: dim {dim} is out of bounds for tensor with {ndim} dimensions"
            ),
        });
    }

    // Always produce the keepdim form first; squeeze later if requested.
    // This matches upstream's `at::sum_out(result, (self - maxes).exp_(), dims,
    // keepdim).log_().add_(maxes_squeezed)` pipeline at `ReduceOps.cpp:1520`,
    // where `maxes` is computed via `at::amax(..., dims, /*keepdim=*/true)`.
    let result_keepdim = elementwise::logsumexp_dim(input, norm_dim, true)?;
    let in_shape = input.shape();
    let mut keepdim_shape: Vec<usize> = in_shape.to_vec();
    keepdim_shape[norm_dim] = 1;

    let final_result = if keepdim {
        result_keepdim.clone()
    } else {
        // Squeeze the reduced dim. The result was built fresh; rebuild a
        // tensor with the squeezed shape over the same CPU buffer.
        let data = result_keepdim.data()?.to_vec();
        let mut s = keepdim_shape.clone();
        s.remove(norm_dim);
        Tensor::from_storage(TensorStorage::cpu(data), s, false)?
    };

    if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(LogsumexpDimBackward {
            input: input.clone(),
            result_keepdim,
            dim: norm_dim,
        });
        let (storage, shape) = final_result.into_storage_and_shape()?;
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Ok(final_result)
    }
}

// ---------------------------------------------------------------------------
// argmax / argmin (non-differentiable; integer-indexed)
// ---------------------------------------------------------------------------
//
// Mirrors `TORCH_IMPL_FUNC(argmax_out)` at `aten/src/ATen/native/ReduceOps.
// cpp:1809-1815` and `TORCH_IMPL_FUNC(argmin_out)` at `:1817-1823`. Both
// dispatch through `argmax_argmin_impl` at `:1775-1807` which (a) when
// `dim` is None, flattens via `self.reshape({-1})` and reduces all (b)
// when `sizes[dim] == 1`, fills the result with 0 (`:1789-1792`).
// Integer-output, NO `derivatives.yaml` entry → non-differentiable.

fn argmax_argmin_full<T: Float>(
    input: &Tensor<T>,
    find_max: bool,
) -> FerrotorchResult<IntTensor<i64>> {
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda {
            op: if find_max { "argmax" } else { "argmin" },
        });
    }
    let data = input.data()?;
    if data.is_empty() {
        return Err(FerrotorchError::InvalidArgument {
            message: "argmax/argmin: cannot reduce an empty tensor".into(),
        });
    }
    let mut best_idx = 0i64;
    let mut best_val = data[0];
    for (i, &v) in data.iter().enumerate().skip(1) {
        // NaN-poisoning per upstream `aten/src/ATen/native/SharedReduceOps.h:26-34
        // `max_propagate_nan` / `min_propagate_nan`: when `v` is NaN, take it
        // unconditionally; once `best_val` is NaN, no later finite value can
        // displace it (every `v > NaN` / `v < NaN` is false, AND we skip them
        // here via the `!best_val.is_nan()` guard so we don't accidentally
        // overwrite NaN with a finite). Matches torch live oracle:
        //   torch.argmax(torch.tensor([1, nan, 3])).item() == 1
        let take = if find_max {
            v.is_nan() || (!best_val.is_nan() && v > best_val)
        } else {
            v.is_nan() || (!best_val.is_nan() && v < best_val)
        };
        if take {
            best_idx = i as i64;
            best_val = v;
        }
    }
    Ok(IntTensor::<i64>::scalar(best_idx))
}

fn argmax_argmin_dim<T: Float>(
    input: &Tensor<T>,
    dim: i64,
    keepdim: bool,
    find_max: bool,
) -> FerrotorchResult<IntTensor<i64>> {
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda {
            op: if find_max { "argmax_dim" } else { "argmin_dim" },
        });
    }
    let ndim = input.ndim();
    if ndim == 0 {
        // Upstream `argmax(scalar, dim=0)` returns a 0-D zero tensor (the
        // single element is trivially the argmax). `:1789-1792 fill_(0)`
        // path for `sizes[dim] == 1`.
        return Ok(IntTensor::<i64>::scalar(0));
    }
    let norm_dim = if dim < 0 {
        (ndim as i64 + dim) as usize
    } else {
        dim as usize
    };
    if norm_dim >= ndim {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "argmax/argmin: dim {dim} is out of bounds for tensor with {ndim} dimensions"
            ),
        });
    }
    let input_ref = if input.is_contiguous() {
        input.clone()
    } else {
        input.contiguous()?
    };
    let in_data = input_ref.data()?;
    let in_shape = input_ref.shape();
    let dim_size = in_shape[norm_dim];
    let outer: usize = in_shape[..norm_dim].iter().product();
    let inner: usize = in_shape[norm_dim + 1..].iter().product();

    let mut out = Vec::with_capacity(outer * inner);
    for o in 0..outer {
        for i in 0..inner {
            let base = o * dim_size * inner + i;
            let mut best_idx = 0i64;
            let mut best_val = in_data[base];
            for d in 1..dim_size {
                let v = in_data[base + d * inner];
                // NaN-poison per-slice — same predicate as `argmax_argmin_full`
                // mirroring upstream `SharedReduceOps.h:26-34`.
                let take = if find_max {
                    v.is_nan() || (!best_val.is_nan() && v > best_val)
                } else {
                    v.is_nan() || (!best_val.is_nan() && v < best_val)
                };
                if take {
                    best_idx = d as i64;
                    best_val = v;
                }
            }
            out.push(best_idx);
        }
    }
    let mut out_shape: Vec<usize> = in_shape.to_vec();
    if keepdim {
        out_shape[norm_dim] = 1;
    } else {
        out_shape.remove(norm_dim);
    }
    IntTensor::<i64>::from_vec(out, out_shape)
}

/// Non-differentiable global argmax: returns a 0-D IntTensor with the
/// flat index of the largest element. Mirrors
/// `torch.argmax(input)` (`dim=None` upstream path at `ReduceOps.cpp:1796-
/// 1798 in = MaybeOwned::owned(self.reshape({-1}))`). Integer output →
/// no autograd. Closes blocker #1304.
pub fn argmax<T: Float>(input: &Tensor<T>) -> FerrotorchResult<IntTensor<i64>> {
    argmax_argmin_full(input, true)
}

/// Non-differentiable dim-keyed argmax. Mirrors
/// `torch.argmax(input, dim, keepdim)` at `ReduceOps.cpp:1809`.
pub fn argmax_dim<T: Float>(
    input: &Tensor<T>,
    dim: i64,
    keepdim: bool,
) -> FerrotorchResult<IntTensor<i64>> {
    argmax_argmin_dim(input, dim, keepdim, true)
}

/// Non-differentiable global argmin. Mirrors `torch.argmin(input)`.
pub fn argmin<T: Float>(input: &Tensor<T>) -> FerrotorchResult<IntTensor<i64>> {
    argmax_argmin_full(input, false)
}

/// Non-differentiable dim-keyed argmin. Mirrors
/// `torch.argmin(input, dim, keepdim)` at `ReduceOps.cpp:1817`.
pub fn argmin_dim<T: Float>(
    input: &Tensor<T>,
    dim: i64,
    keepdim: bool,
) -> FerrotorchResult<IntTensor<i64>> {
    argmax_argmin_dim(input, dim, keepdim, false)
}

// ---------------------------------------------------------------------------
// std / var (Welford-derived; Bessel correction via `correction` scalar)
// ---------------------------------------------------------------------------
//
// Mirrors `Tensor var(const Tensor& self, bool unbiased)` at
// `aten/src/ATen/native/ReduceOps.cpp:2085-2089` and
// `Tensor std(const Tensor& self, bool unbiased)` at `:2105-2108`. Upstream
// uses Welford's online algorithm via `SharedReduceOps.h:86-148 WelfordOps`
// for numerical stability; ferrotorch uses the simpler two-pass form
// (mean, then sum-of-squared-deviations) which matches upstream byte-for-
// byte on the f32 input domain op_db sweeps (the Welford win is only
// material on >1e7-element inputs that op_db does not emit).
//
// VJP per `derivatives.yaml:1924-1925`:
//   - name: var.correction(Tensor self, int[1]? dim=None, *, Scalar?
//     correction=None, bool keepdim=False) -> Tensor
//     self: var_backward(grad, self, dim, correction, keepdim)
//
// expanding to `grad_input = grad * 2 * (input - mean) / (n - correction)`
// (full reduction; broadcast `grad` and `mean` to input shape). For std,
// chain rule adds a `/ (2 * result)` factor with the `result == 0 -> 0`
// degeneracy guard per `derivatives.yaml:1676`.

#[derive(Debug)]
pub struct VarBackward<T: Float> {
    input: Tensor<T>,
    mean: T,
    /// `n - correction` denominator (`unbiased=true` → `n-1`; else `n`).
    denom: f64,
}

impl<T: Float> GradFn<T> for VarBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if self.input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda { op: "var backward" });
        }
        let go = grad_output.data()?[0];
        let scale = float_from_f64::<T>(2.0 / self.denom)?;
        let data = self.input.data()?;
        let grad_data: Vec<T> = data.iter().map(|&v| go * scale * (v - self.mean)).collect();
        let grad_input = Tensor::from_storage(
            TensorStorage::cpu(grad_data),
            self.input.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "VarBackward"
    }
}

#[derive(Debug)]
pub struct StdBackward<T: Float> {
    input: Tensor<T>,
    mean: T,
    /// `n - correction` denominator.
    denom: f64,
    /// `result = sqrt(var)` saved at forward time. Chain-rule factor in
    /// the backward is `/ (2 * result)`, masked to zero when `result == 0`
    /// per `derivatives.yaml:1676 .masked_fill_(result == 0, 0)`.
    result: T,
}

impl<T: Float> GradFn<T> for StdBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if self.input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda { op: "std backward" });
        }
        let go = grad_output.data()?[0];
        let data = self.input.data()?;
        let zero = <T as num_traits::Zero>::zero();
        if self.result == zero {
            // Degenerate: variance is zero → all input values equal mean →
            // gradient is zero (upstream's masked_fill).
            let grad_data = vec![zero; data.len()];
            let grad_input = Tensor::from_storage(
                TensorStorage::cpu(grad_data),
                self.input.shape().to_vec(),
                false,
            )?;
            return Ok(vec![Some(grad_input)]);
        }
        // d(std)/d(x_i) = (x_i - mean) / (denom * std)
        let scale = float_from_f64::<T>(1.0 / self.denom)? / self.result;
        let grad_data: Vec<T> = data.iter().map(|&v| go * scale * (v - self.mean)).collect();
        let grad_input = Tensor::from_storage(
            TensorStorage::cpu(grad_data),
            self.input.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "StdBackward"
    }
}

/// Full-reduction variance with arbitrary Bessel correction. Mirrors
/// `Tensor& std_var_out(... correction_opt ...)` and `std_var_all_cpu`
/// at `aten/src/ATen/native/ReduceOps.cpp:1858-1864` /
/// `:1938-1941`. The denominator is `max(0, n - correction)`; on empty
/// input the upstream `iter.numel() == 0` branch at `:1938-1941` returns
/// `result.fill_(NaN)` (matched here by skipping the sum loop and
/// emitting NaN via IEEE-754 `0.0 / 0.0`).
///
/// `correction` is a real-valued scalar — fractional / negative
/// corrections are valid per torch (`torch.var(x, correction=0.5)`,
/// `torch.var(x, correction=-1)` are documented and live-oracle-tested).
fn var_inner<T: Float>(
    input: &Tensor<T>,
    correction: f64,
    take_sqrt: bool,
) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) = crate::meta_propagate::reduce_all(input)? {
        return Ok(out);
    }
    let op_name = if take_sqrt { "std" } else { "var" };
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: op_name });
    }
    let data = input.data()?;
    let n = data.len();
    if n == 0 {
        // Upstream `ReduceOps.cpp:1938-1941`:
        //   if (iter.numel() == 0) {
        //     result.fill_(std::numeric_limits<double>::quiet_NaN());
        //     return result;
        //   }
        // Live torch oracle: `torch.var(torch.tensor([])).item() is nan`.
        let nan_val = <T as num_traits::Float>::nan();
        return Tensor::from_storage(TensorStorage::cpu(vec![nan_val]), vec![], false);
    }
    // Upstream div-by-zero protection at `ReduceOps.cpp:1862-1864
    // `std::max(0.0, self.numel() - correction)``. We mirror the
    // `max(0, ...)` clamp so `var([single_element], unbiased=true)`
    // returns +Inf via Rust f64 IEEE-754 div-by-zero semantics (matches
    // upstream's __ubsan_ignore_float_divide_by_zero__ contract).
    let denom_f = (n as f64 - correction).max(0.0);
    let mut sum_f: f64 = 0.0;
    for &v in data {
        sum_f += to_f64::<T>(v)?;
    }
    let mean_f = sum_f / n as f64;
    let mut sum_sq: f64 = 0.0;
    for &v in data {
        let d = to_f64::<T>(v)? - mean_f;
        sum_sq += d * d;
    }
    let var_f = sum_sq / denom_f;
    let final_f = if take_sqrt { var_f.sqrt() } else { var_f };
    let result_val = float_from_f64::<T>(final_f)?;
    let result = Tensor::from_storage(TensorStorage::cpu(vec![result_val]), vec![], false)?;
    if is_grad_enabled() && input.requires_grad() {
        let mean_t = float_from_f64::<T>(mean_f)?;
        if take_sqrt {
            let grad_fn = Arc::new(StdBackward {
                input: input.clone(),
                mean: mean_t,
                denom: denom_f,
                result: result_val,
            });
            let (storage, shape) = result.into_storage_and_shape()?;
            Tensor::from_operation(storage, shape, grad_fn)
        } else {
            let grad_fn = Arc::new(VarBackward {
                input: input.clone(),
                mean: mean_t,
                denom: denom_f,
            });
            let (storage, shape) = result.into_storage_and_shape()?;
            Tensor::from_operation(storage, shape, grad_fn)
        }
    } else {
        Ok(result)
    }
}

/// Full-reduction variance. Mirrors `Tensor var(const Tensor& self, bool
/// unbiased)` at `aten/src/ATen/native/ReduceOps.cpp:2085`. When
/// `unbiased=true`, divides by `n-1` (Bessel-corrected sample variance);
/// when false, divides by `n` (population variance). Empty input returns
/// a NaN scalar to match upstream's `:1938-1941` trivial-reduction path.
///
/// Closes blocker #1301 (var).
pub fn var<T: Float>(input: &Tensor<T>, unbiased: bool) -> FerrotorchResult<Tensor<T>> {
    var_inner(input, if unbiased { 1.0 } else { 0.0 }, false)
}

/// Full-reduction variance with arbitrary `correction`. Mirrors the
/// `*.correction` overload at `aten/src/ATen/native/ReduceOps.cpp:1858-1864`
/// — `denom = max(0, n - correction)`, so `correction=2` over 5 elements
/// yields `sum_sq / 3` (live oracle:
/// `torch.var(torch.tensor([1,2,3,4,5]), correction=2).item() == 10/3`).
/// Empty input returns NaN per `:1938-1941`.
pub fn var_with_correction<T: Float>(
    input: &Tensor<T>,
    correction: f64,
) -> FerrotorchResult<Tensor<T>> {
    var_inner(input, correction, false)
}

/// Full-reduction standard deviation. Mirrors `Tensor std(const Tensor&
/// self, bool unbiased)` at `aten/src/ATen/native/ReduceOps.cpp:2105`.
/// Computes `sqrt(var(input, unbiased))`. Empty input returns NaN.
///
/// Closes blocker #1301 (std).
pub fn std<T: Float>(input: &Tensor<T>, unbiased: bool) -> FerrotorchResult<Tensor<T>> {
    var_inner(input, if unbiased { 1.0 } else { 0.0 }, true)
}

/// Full-reduction standard deviation with arbitrary `correction`. Mirrors
/// the `*.correction` overload at `aten/src/ATen/native/ReduceOps.cpp:1858-1864`
/// — `sqrt(sum_sq / max(0, n - correction))` (live oracle:
/// `torch.std(torch.tensor([1,2,3,4,5]), correction=2).item() == sqrt(10/3)`).
pub fn std_with_correction<T: Float>(
    input: &Tensor<T>,
    correction: f64,
) -> FerrotorchResult<Tensor<T>> {
    var_inner(input, correction, true)
}

// ---------------------------------------------------------------------------
// std_dim / var_dim (per-slice two-pass: mean → sum-of-sq-dev → div)
// ---------------------------------------------------------------------------
//
// Mirrors the dim-keyed `var.correction` / `std.correction` overloads at
// `aten/src/ATen/native/ReduceOps.cpp` (the multi-dim path in
// `std_var_out` recursively factors single-dim reductions on the
// keepdim-shaped slice; ferrotorch handles multi-dim by chaining
// `std_dim` over each dim in descending-order at the runner layer).
// Single-dim correctness suffices for the sweep — the chain stays
// numerically equivalent for `var` (sum-of-squared deviations is
// associative across disjoint axes) but NOT for `std` (sqrt breaks
// associativity); the runner therefore uses `var_dim` chain then
// `sqrt` for multi-dim std.

/// Forward result of the shared dim-keyed std/var kernel, carrying the
/// per-slice statistics the backward nodes need (means, denominator).
struct StdVarDimForward<T: Float> {
    /// Reduced values, row-major over `outer * inner` slices.
    data: Vec<T>,
    /// Output shape (keepdim-aware).
    out_shape: Vec<usize>,
    /// Normalized (non-negative) reduced dim.
    norm_dim: usize,
    /// Per-slice means (row-major over `outer * inner`), f64 as computed.
    means: Vec<f64>,
    /// `max(0, n - correction)` — shared by every slice.
    denom: f64,
}

fn std_var_dim_forward<T: Float>(
    input: &Tensor<T>,
    dim: i64,
    keepdim: bool,
    correction: f64,
    take_sqrt: bool,
) -> FerrotorchResult<StdVarDimForward<T>> {
    let ndim = input.ndim();
    if ndim == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "std_dim/var_dim: cannot reduce a 0-D tensor".into(),
        });
    }
    let norm_dim = if dim < 0 {
        (ndim as i64 + dim) as usize
    } else {
        dim as usize
    };
    if norm_dim >= ndim {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("std_dim/var_dim: dim {dim} out of bounds"),
        });
    }
    let input_ref = if input.is_contiguous() {
        input.clone()
    } else {
        input.contiguous()?
    };
    let in_data = input_ref.data()?;
    let in_shape = input_ref.shape().to_vec();
    let dim_size = in_shape[norm_dim];
    let outer: usize = in_shape[..norm_dim].iter().product();
    let inner: usize = in_shape[norm_dim + 1..].iter().product();

    let denom = (dim_size as f64 - correction).max(0.0);
    let mut result = Vec::with_capacity(outer * inner);
    let mut means = Vec::with_capacity(outer * inner);
    for o in 0..outer {
        for i in 0..inner {
            // Pass 1: mean.
            let mut s = 0.0_f64;
            for d in 0..dim_size {
                s += to_f64::<T>(in_data[o * dim_size * inner + d * inner + i])?;
            }
            let mean_f = s / dim_size as f64;
            means.push(mean_f);
            // Pass 2: sum-of-squared deviations.
            let mut ss = 0.0_f64;
            for d in 0..dim_size {
                let v = to_f64::<T>(in_data[o * dim_size * inner + d * inner + i])?;
                let dv = v - mean_f;
                ss += dv * dv;
            }
            let var_f = ss / denom;
            let final_f = if take_sqrt { var_f.sqrt() } else { var_f };
            result.push(float_from_f64::<T>(final_f)?);
        }
    }
    let mut out_shape: Vec<usize> = in_shape.clone();
    if keepdim {
        out_shape[norm_dim] = 1;
    } else {
        out_shape.remove(norm_dim);
    }
    Ok(StdVarDimForward {
        data: result,
        out_shape,
        norm_dim,
        means,
        denom,
    })
}

/// Backward node for `var_dim(input, dim, correction, keepdim)`.
///
/// VJP per `derivatives.yaml` `var.correction` →
/// `var_backward(grad, self, dim, correction, keepdim)` in
/// `torch/csrc/autograd/FunctionsManual.cpp`:
/// `dx = grad * 2 * (x - mean(dim, keepdim=True)) / max(0, n - correction)`
/// — the keepdim-shaped `grad`/`mean` broadcast over the reduced dim.
///
/// `denom == 0` (correction >= slice length) yields NaN gradients via IEEE
/// `inf * 0` to match torch (live oracle on 2.11.0:
/// `torch.var(torch.tensor([[3.],[4.]]), dim=1, correction=1)` →
/// fwd `[nan, nan]`, `.backward` grad `[[nan],[nan]]`; `correction=0` →
/// fwd `[0., 0.]`, grad `[[0.],[0.]]`). Closes CORE-046 (#1740).
#[derive(Debug)]
pub struct VarDimBackward<T: Float> {
    input: Tensor<T>,
    /// Normalized (non-negative) reduced dim.
    norm_dim: usize,
    /// Per-slice means (row-major over `outer * inner`), saved at forward.
    means: Vec<f64>,
    /// `max(0, n - correction)` denominator shared by every slice.
    denom: f64,
}

impl<T: Float> GradFn<T> for VarDimBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if grad_output.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "var_dim backward",
            });
        }
        // Whether keepdim was true (out slice dim retained as size 1) or
        // false (removed), a contiguous grad_output flattens to the same
        // `o * inner + i` slice order the forward used.
        let go_ref = if grad_output.is_contiguous() {
            grad_output.clone()
        } else {
            grad_output.contiguous()?
        };
        let go = go_ref.data()?;
        let in_ref = if self.input.is_contiguous() {
            self.input.clone()
        } else {
            self.input.contiguous()?
        };
        let in_data = in_ref.data()?;
        let in_shape = in_ref.shape().to_vec();
        let dim_size = in_shape[self.norm_dim];
        let outer: usize = in_shape[..self.norm_dim].iter().product();
        let inner: usize = in_shape[self.norm_dim + 1..].iter().product();
        // May be +inf when denom == 0 — the NaN-propagation path above.
        let scale = 2.0 / self.denom;
        let mut dx = vec![<T as num_traits::Zero>::zero(); in_data.len()];
        for o in 0..outer {
            for i in 0..inner {
                let g = to_f64::<T>(go[o * inner + i])?;
                let mean_f = self.means[o * inner + i];
                for d in 0..dim_size {
                    let idx = o * dim_size * inner + d * inner + i;
                    let v = to_f64::<T>(in_data[idx])?;
                    dx[idx] = float_from_f64::<T>(g * scale * (v - mean_f))?;
                }
            }
        }
        let grad_input = Tensor::from_storage(TensorStorage::cpu(dx), in_shape, false)?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "VarDimBackward"
    }
}

/// Backward node for `std_dim(input, dim, correction, keepdim)`.
///
/// VJP per `derivatives.yaml` `std.correction` →
/// `std_backward(result, grad, self, dim, correction, keepdim)` in
/// `torch/csrc/autograd/FunctionsManual.cpp` — chain rule through
/// `sqrt`: `dx = grad * (x - mean) / (max(0, n - correction) * result)`,
/// with the `result == 0 → 0` degeneracy guard per `derivatives.yaml`'s
/// `masked_fill_(result == 0, 0)` (live oracle:
/// `torch.std([[5.,5.,5.],[1.,2.,3.]], dim=1, correction=1)` → grad
/// `[[0,0,0],[-0.5,0,0.5]]`). A NaN `result` (correction >= slice length)
/// fails the `== 0` test and propagates NaN, matching torch.
#[derive(Debug)]
pub struct StdDimBackward<T: Float> {
    input: Tensor<T>,
    /// Normalized (non-negative) reduced dim.
    norm_dim: usize,
    /// Per-slice means (row-major over `outer * inner`), saved at forward.
    means: Vec<f64>,
    /// `max(0, n - correction)` denominator shared by every slice.
    denom: f64,
    /// Per-slice `result = sqrt(var)` saved at forward time (zero guard +
    /// chain-rule factor).
    results: Vec<T>,
}

impl<T: Float> GradFn<T> for StdDimBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if grad_output.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "std_dim backward",
            });
        }
        let go_ref = if grad_output.is_contiguous() {
            grad_output.clone()
        } else {
            grad_output.contiguous()?
        };
        let go = go_ref.data()?;
        let in_ref = if self.input.is_contiguous() {
            self.input.clone()
        } else {
            self.input.contiguous()?
        };
        let in_data = in_ref.data()?;
        let in_shape = in_ref.shape().to_vec();
        let dim_size = in_shape[self.norm_dim];
        let outer: usize = in_shape[..self.norm_dim].iter().product();
        let inner: usize = in_shape[self.norm_dim + 1..].iter().product();
        let zero = <T as num_traits::Zero>::zero();
        let mut dx = vec![zero; in_data.len()];
        for o in 0..outer {
            for i in 0..inner {
                let slice = o * inner + i;
                let result = self.results[slice];
                if result == zero {
                    // masked_fill_(result == 0, 0): every element of the
                    // slice equals the mean; subgradient 0 (dx pre-zeroed).
                    continue;
                }
                let g = to_f64::<T>(go[slice])?;
                let mean_f = self.means[slice];
                // NaN result (denom 0): `denom * result_f` is NaN and the
                // `0 / NaN` division propagates NaN per torch.
                let result_f = to_f64::<T>(result)?;
                let denom_result = self.denom * result_f;
                for d in 0..dim_size {
                    let idx = o * dim_size * inner + d * inner + i;
                    let v = to_f64::<T>(in_data[idx])?;
                    dx[idx] = float_from_f64::<T>(g * (v - mean_f) / denom_result)?;
                }
            }
        }
        let grad_input = Tensor::from_storage(TensorStorage::cpu(dx), in_shape, false)?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "StdDimBackward"
    }
}

/// Dim-keyed variance. Attaches [`VarDimBackward`] when grad is needed
/// (CORE-046 / #1740 — was forward-only). Closes the `var` runner-arm
/// half of #1301 + #1314.
pub fn var_dim<T: Float>(
    input: &Tensor<T>,
    dim: i64,
    correction: f64,
    keepdim: bool,
) -> FerrotorchResult<Tensor<T>> {
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "var_dim" });
    }
    let fwd = std_var_dim_forward(input, dim, keepdim, correction, false)?;
    let result = Tensor::from_storage(TensorStorage::cpu(fwd.data), fwd.out_shape, false)?;
    if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(VarDimBackward {
            input: input.clone(),
            norm_dim: fwd.norm_dim,
            means: fwd.means,
            denom: fwd.denom,
        });
        let (storage, shape) = result.into_storage_and_shape()?;
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Ok(result)
    }
}

/// Dim-keyed standard deviation. Attaches [`StdDimBackward`] when grad is
/// needed (CORE-046 / #1740 — was forward-only). Closes the `std`
/// runner-arm half of #1301 + #1314.
pub fn std_dim<T: Float>(
    input: &Tensor<T>,
    dim: i64,
    correction: f64,
    keepdim: bool,
) -> FerrotorchResult<Tensor<T>> {
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "std_dim" });
    }
    let fwd = std_var_dim_forward(input, dim, keepdim, correction, true)?;
    let results = fwd.data.clone();
    let result = Tensor::from_storage(TensorStorage::cpu(fwd.data), fwd.out_shape, false)?;
    if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(StdDimBackward {
            input: input.clone(),
            norm_dim: fwd.norm_dim,
            means: fwd.means,
            denom: fwd.denom,
            results,
        });
        let (storage, shape) = result.into_storage_and_shape()?;
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// any / all / count_nonzero (non-differentiable; bool/int output)
// ---------------------------------------------------------------------------
//
// Mirrors `TORCH_IMPL_FUNC(all_out)` at `aten/src/ATen/native/ReduceOps.cpp:
// 1667-1670` and `TORCH_IMPL_FUNC(any_out)` at `:1681-1684`; `count_nonzero`
// lives in `aten/src/ATen/native/SummaryOps.cpp`. All three return integer-
// or bool-typed tensors and have NO `derivatives.yaml` entry → non-
// differentiable, no `*Backward` node needed.
//
// Treatment of "non-zero" for floats: matches upstream (`v != 0.0`); NaN
// is non-zero (NaN != 0.0 is true in IEEE-754), matching
// `at::native::nonzero_count` semantics.

fn is_nonzero_float<T: Float>(v: T) -> bool {
    v != <T as num_traits::Zero>::zero()
}

/// Non-differentiable full-reduction `any`. Returns a 0-D BoolTensor
/// holding `true` iff any input element is non-zero. Mirrors
/// `Tensor any(const Tensor& self)` (the `_out` variant at
/// `aten/src/ATen/native/ReduceOps.cpp:1681`).
///
/// Closes blocker #1312 (any).
pub fn any<T: Float>(input: &Tensor<T>) -> FerrotorchResult<BoolTensor> {
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "any" });
    }
    let data = input.data()?;
    // Upstream: `any(empty) == false` (zero of the disjunction monoid).
    let result = data.iter().copied().any(is_nonzero_float::<T>);
    BoolTensor::from_vec(vec![result], vec![])
}

/// Non-differentiable full-reduction `all`. Returns a 0-D BoolTensor
/// holding `true` iff every input element is non-zero. Mirrors
/// `Tensor all(const Tensor& self)` (the `_out` variant at
/// `aten/src/ATen/native/ReduceOps.cpp:1667`).
///
/// Closes blocker #1312 (all).
pub fn all<T: Float>(input: &Tensor<T>) -> FerrotorchResult<BoolTensor> {
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "all" });
    }
    let data = input.data()?;
    // Upstream: `all(empty) == true` (identity of the conjunction monoid).
    let result = data.iter().copied().all(is_nonzero_float::<T>);
    BoolTensor::from_vec(vec![result], vec![])
}

/// Non-differentiable full-reduction `count_nonzero`. Returns a 0-D
/// IntTensor<i64> with the count of non-zero elements. Mirrors
/// `Tensor count_nonzero(const Tensor& self)` in
/// `aten/src/ATen/native/SummaryOps.cpp`.
///
/// Closes blocker #1312 (count_nonzero).
pub fn count_nonzero<T: Float>(input: &Tensor<T>) -> FerrotorchResult<IntTensor<i64>> {
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda {
            op: "count_nonzero",
        });
    }
    let data = input.data()?;
    let n = data
        .iter()
        .copied()
        .filter(|&v| is_nonzero_float(v))
        .count();
    Ok(IntTensor::<i64>::scalar(n as i64))
}

// ---------------------------------------------------------------------------
// Dim-keyed any / all / count_nonzero (non-differentiable)
// ---------------------------------------------------------------------------
//
// Mirrors `at::native::any.dim(self, dim, keepdim)` and `all.dim(...)`
// from `aten/src/ATen/native/ReduceOps.cpp:1690-1706` (the `_dims_out`
// overloads). Non-differentiable; integer/bool outputs.

fn reduce_dim_loop_bool<T, F>(
    input: &Tensor<T>,
    dim: i64,
    keepdim: bool,
    init: bool,
    op_name: &'static str,
    fold: F,
) -> FerrotorchResult<BoolTensor>
where
    T: Float,
    F: Fn(bool, T) -> bool,
{
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: op_name });
    }
    let ndim = input.ndim();
    if ndim == 0 {
        // 0-D: degenerate single-element reduction; return the element's
        // truthiness wrapped as 0-D BoolTensor.
        let v = input.data()?[0];
        return BoolTensor::from_vec(vec![fold(init, v)], vec![]);
    }
    let norm_dim = if dim < 0 {
        (ndim as i64 + dim) as usize
    } else {
        dim as usize
    };
    if norm_dim >= ndim {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("{op_name}_dim: dim {dim} out of bounds for {ndim}-D tensor"),
        });
    }
    let input_ref = if input.is_contiguous() {
        input.clone()
    } else {
        input.contiguous()?
    };
    let in_data = input_ref.data()?;
    let in_shape = input_ref.shape();
    let dim_size = in_shape[norm_dim];
    let outer: usize = in_shape[..norm_dim].iter().product();
    let inner: usize = in_shape[norm_dim + 1..].iter().product();

    let mut out = Vec::with_capacity(outer * inner);
    for o in 0..outer {
        for i in 0..inner {
            let mut acc = init;
            for d in 0..dim_size {
                acc = fold(acc, in_data[o * dim_size * inner + d * inner + i]);
            }
            out.push(acc);
        }
    }
    let mut out_shape: Vec<usize> = in_shape.to_vec();
    if keepdim {
        out_shape[norm_dim] = 1;
    } else {
        out_shape.remove(norm_dim);
    }
    BoolTensor::from_vec(out, out_shape)
}

/// Non-differentiable dim-keyed `any`. Mirrors
/// `torch.any(input, dim, keepdim=False)`.
pub fn any_dim<T: Float>(
    input: &Tensor<T>,
    dim: i64,
    keepdim: bool,
) -> FerrotorchResult<BoolTensor> {
    reduce_dim_loop_bool(input, dim, keepdim, false, "any", |acc, v| {
        acc || is_nonzero_float(v)
    })
}

/// Non-differentiable dim-keyed `all`. Mirrors
/// `torch.all(input, dim, keepdim=False)`.
pub fn all_dim<T: Float>(
    input: &Tensor<T>,
    dim: i64,
    keepdim: bool,
) -> FerrotorchResult<BoolTensor> {
    reduce_dim_loop_bool(input, dim, keepdim, true, "all", |acc, v| {
        acc && is_nonzero_float(v)
    })
}

/// Non-differentiable dim-keyed `count_nonzero`. Mirrors
/// `torch.count_nonzero(input, dim)`. Note: upstream's `count_nonzero.dim`
/// has NO `keepdim` parameter — the dim is always squeezed.
pub fn count_nonzero_dim<T: Float>(
    input: &Tensor<T>,
    dim: i64,
) -> FerrotorchResult<IntTensor<i64>> {
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda {
            op: "count_nonzero_dim",
        });
    }
    let ndim = input.ndim();
    if ndim == 0 {
        let v = input.data()?[0];
        return Ok(IntTensor::<i64>::scalar(i64::from(is_nonzero_float(v))));
    }
    let norm_dim = if dim < 0 {
        (ndim as i64 + dim) as usize
    } else {
        dim as usize
    };
    if norm_dim >= ndim {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("count_nonzero_dim: dim {dim} out of bounds for {ndim}-D tensor"),
        });
    }
    let input_ref = if input.is_contiguous() {
        input.clone()
    } else {
        input.contiguous()?
    };
    let in_data = input_ref.data()?;
    let in_shape = input_ref.shape();
    let dim_size = in_shape[norm_dim];
    let outer: usize = in_shape[..norm_dim].iter().product();
    let inner: usize = in_shape[norm_dim + 1..].iter().product();

    let mut out = Vec::with_capacity(outer * inner);
    for o in 0..outer {
        for i in 0..inner {
            let mut count: i64 = 0;
            for d in 0..dim_size {
                if is_nonzero_float(in_data[o * dim_size * inner + d * inner + i]) {
                    count += 1;
                }
            }
            out.push(count);
        }
    }
    let mut out_shape: Vec<usize> = in_shape.to_vec();
    out_shape.remove(norm_dim);
    IntTensor::<i64>::from_vec(out, out_shape)
}

// ---------------------------------------------------------------------------
// Dim-keyed amin / amax (differentiable values — gradient routes to
// every position equal to the per-slice extremum, scaled by 1/count)
// ---------------------------------------------------------------------------
//
// Mirrors `aten/src/ATen/native/ReduceOps.cpp:1758 TORCH_IMPL_FUNC(amin_out)`
// / `:1766 TORCH_IMPL_FUNC(amax_out)` for the dim-keyed form. VJP per
// `tools/autograd/derivatives.yaml:1205-1211`:
//   self: scale_grad_by_count(restore_reduced_dims(grad, dim, keepdim),
//                             restore_reduced_dims(result, dim, keepdim) ==
//                             self, dim)
// — every input position equal to the per-slice extremum gets
// `grad / count_of_extremums_in_that_slice`.

fn normalize_reduction_dim(dim: i64, ndim: usize, op: &str) -> FerrotorchResult<usize> {
    let norm_dim = if dim < 0 { ndim as i64 + dim } else { dim };
    if norm_dim < 0 || norm_dim as usize >= ndim {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("{op}: dim {dim} out of bounds for {ndim}-D tensor"),
        });
    }
    Ok(norm_dim as usize)
}

#[derive(Debug)]
pub struct AminDimBackward<T: Float> {
    input: Tensor<T>,
    /// Forward output flattened as `[outer, inner]` regardless of keepdim.
    result: Tensor<T>,
    dim: usize,
    keepdim: bool,
}

#[derive(Debug)]
pub struct AmaxDimBackward<T: Float> {
    input: Tensor<T>,
    result: Tensor<T>,
    dim: usize,
    keepdim: bool,
}

fn amin_amax_dim_backward<T: Float>(
    input: &Tensor<T>,
    result: &Tensor<T>,
    grad_output: &Tensor<T>,
    dim: usize,
    keepdim: bool,
) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
    let is_f32 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>();
    let is_f64 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>();
    if input.is_cuda() && (is_f32 || is_f64) {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let grad_output = if grad_output.is_cuda() {
            grad_output.contiguous()?
        } else {
            grad_output.to(input.device())?.contiguous()?
        };
        let handle = if is_f32 {
            backend.extreme_axis_backward_f32(
                input.gpu_handle()?,
                result.gpu_handle()?,
                grad_output.gpu_handle()?,
                input.shape(),
                dim,
            )?
        } else {
            backend.extreme_axis_backward_f64(
                input.gpu_handle()?,
                result.gpu_handle()?,
                grad_output.gpu_handle()?,
                input.shape(),
                dim,
            )?
        };
        let grad_input =
            Tensor::from_storage(TensorStorage::gpu(handle), input.shape().to_vec(), false)?;
        return Ok(vec![Some(grad_input)]);
    }
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda {
            op: "amin/amax_dim backward",
        });
    }
    let input_data = input.data_vec()?;
    let result_data = result.data()?;
    let in_shape = input.shape();
    let dim_size = in_shape[dim];
    let outer: usize = in_shape[..dim].iter().product();
    let inner: usize = in_shape[dim + 1..].iter().product();

    // For each (o, i) slice, count how many positions match the extremum.
    let mut counts = vec![0i64; outer * inner];
    for o in 0..outer {
        for i in 0..inner {
            let target = result_data[o * inner + i];
            let mut c = 0i64;
            if !target.is_nan() {
                for d in 0..dim_size {
                    if input_data[o * dim_size * inner + d * inner + i] == target {
                        c += 1;
                    }
                }
            }
            counts[o * inner + i] = c;
        }
    }

    // Re-broadcast grad to input shape: if keepdim=false, grad is shape
    // [outer..., inner...]; if keepdim=true, the reduced dim is size 1 and
    // present in grad. Either way, the per-(o,i) slot value is at
    // grad[o * inner + i] in flat layout.
    let grad_data = grad_output.data()?;
    let _ = keepdim; // shape info absorbed via the o*inner+i indexing

    let in_numel: usize = in_shape.iter().product();
    let mut out = Vec::with_capacity(in_numel);
    for o in 0..outer {
        for d in 0..dim_size {
            for i in 0..inner {
                let target = result_data[o * inner + i];
                let val = input_data[o * dim_size * inner + d * inner + i];
                let c = counts[o * inner + i];
                if c == 0 {
                    out.push(T::nan());
                } else if val == target {
                    let g = grad_data[o * inner + i];
                    let scale = float_from_f64::<T>(1.0 / c as f64)?;
                    out.push(g * scale);
                } else {
                    out.push(<T as num_traits::Zero>::zero());
                }
            }
        }
    }
    let grad_input = Tensor::from_storage(TensorStorage::cpu(out), in_shape.to_vec(), false)?;
    Ok(vec![Some(grad_input)])
}

impl<T: Float> GradFn<T> for AminDimBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        amin_amax_dim_backward(
            &self.input,
            &self.result,
            grad_output,
            self.dim,
            self.keepdim,
        )
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "AminDimBackward"
    }
}

impl<T: Float> GradFn<T> for AmaxDimBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        amin_amax_dim_backward(
            &self.input,
            &self.result,
            grad_output,
            self.dim,
            self.keepdim,
        )
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "AmaxDimBackward"
    }
}

#[allow(
    clippy::type_complexity,
    reason = "Single-use helper returning (result, norm_dim, out_shape); \
              a named tuple struct adds boilerplate without clarifying the local \
              flow at the two callers (amin_dim/amax_dim)."
)]
fn amin_amax_dim_forward<T: Float>(
    input: &Tensor<T>,
    dim: i64,
    keepdim: bool,
    find_max: bool,
) -> FerrotorchResult<(Vec<T>, usize, Vec<usize>)> {
    let ndim = input.ndim();
    let norm_dim = if dim < 0 {
        (ndim as i64 + dim) as usize
    } else {
        dim as usize
    };
    if norm_dim >= ndim {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("amin/amax_dim: dim {dim} out of bounds for {ndim}-D tensor"),
        });
    }
    let input_ref = if input.is_contiguous() {
        input.clone()
    } else {
        input.contiguous()?
    };
    let in_data = input_ref.data()?;
    let in_shape = input_ref.shape();
    let dim_size = in_shape[norm_dim];
    if dim_size == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "amin/amax_dim: cannot reduce over an empty dimension".into(),
        });
    }
    let outer: usize = in_shape[..norm_dim].iter().product();
    let inner: usize = in_shape[norm_dim + 1..].iter().product();

    let mut result = Vec::with_capacity(outer * inner);
    // First pass: compute per-slice extremum, NaN-poisoning per upstream
    // `aten/src/ATen/native/SharedReduceOps.h:26-34
    // `max_propagate_nan` / `min_propagate_nan`. Live torch:
    //   torch.amin(torch.tensor([[1, nan, 3], [4, 5, 6]]), dim=1) == [nan, 4]
    //   torch.amax(...) == [nan, 6]
    for o in 0..outer {
        for i in 0..inner {
            let mut best = in_data[o * dim_size * inner + i];
            for d in 1..dim_size {
                let v = in_data[o * dim_size * inner + d * inner + i];
                let take = if find_max {
                    v.is_nan() || (!best.is_nan() && v > best)
                } else {
                    v.is_nan() || (!best.is_nan() && v < best)
                };
                if take {
                    best = v;
                }
            }
            result.push(best);
        }
    }
    let mut out_shape: Vec<usize> = in_shape.to_vec();
    if keepdim {
        out_shape[norm_dim] = 1;
    } else {
        out_shape.remove(norm_dim);
    }
    Ok((result, norm_dim, out_shape))
}

/// Differentiable dim-keyed amin. Mirrors `torch.amin(input, dim, keepdim)`.
/// VJP routes the grad to every position equal to the per-slice min,
/// scaled by `1/count`.
pub fn amin_dim<T: Float>(
    input: &Tensor<T>,
    dim: i64,
    keepdim: bool,
) -> FerrotorchResult<Tensor<T>> {
    let ndim = input.ndim();
    if ndim == 0 {
        return grad_clone(input);
    }
    let norm_dim = normalize_reduction_dim(dim, ndim, "amin_dim")?;
    if input.shape()[norm_dim] == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "amin_dim: cannot reduce over an empty dimension".into(),
        });
    }
    let is_f32 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>();
    let is_f64 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>();

    if input.is_cuda() && (is_f32 || is_f64) {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let input = input.contiguous()?;
        let mut out_shape: Vec<usize> = input.shape().to_vec();
        if keepdim {
            out_shape[norm_dim] = 1;
        } else {
            out_shape.remove(norm_dim);
        }
        let handle = if is_f32 {
            backend.min_axis_f32(input.gpu_handle()?, input.shape(), norm_dim)?
        } else {
            backend.min_axis_f64(input.gpu_handle()?, input.shape(), norm_dim)?
        };
        let compact_result =
            Tensor::from_storage(TensorStorage::gpu(handle), out_shape.clone(), false)?;
        if is_grad_enabled() && input.requires_grad() {
            let grad_fn = Arc::new(AminDimBackward {
                input: input.clone(),
                result: compact_result.clone(),
                dim: norm_dim,
                keepdim,
            });
            let (s, sh) = compact_result.into_storage_and_shape()?;
            return Tensor::from_operation(s, sh, grad_fn);
        }
        return Ok(compact_result);
    }
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "amin_dim" });
    }

    let (result, norm_dim, out_shape) = amin_amax_dim_forward(input, dim, keepdim, false)?;
    let result_t = Tensor::from_storage(TensorStorage::cpu(result), out_shape, false)?;
    if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(AminDimBackward {
            input: input.clone(),
            result: result_t.clone(),
            dim: norm_dim,
            keepdim,
        });
        let (s, sh) = result_t.into_storage_and_shape()?;
        Tensor::from_operation(s, sh, grad_fn)
    } else {
        Ok(result_t)
    }
}

/// Differentiable dim-keyed amax. Symmetric to [`amin_dim`].
pub fn amax_dim<T: Float>(
    input: &Tensor<T>,
    dim: i64,
    keepdim: bool,
) -> FerrotorchResult<Tensor<T>> {
    let ndim = input.ndim();
    if ndim == 0 {
        return grad_clone(input);
    }
    let norm_dim = normalize_reduction_dim(dim, ndim, "amax_dim")?;
    if input.shape()[norm_dim] == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "amax_dim: cannot reduce over an empty dimension".into(),
        });
    }
    let is_f32 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>();
    let is_f64 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>();

    if input.is_cuda() && (is_f32 || is_f64) {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let input = input.contiguous()?;
        let mut out_shape: Vec<usize> = input.shape().to_vec();
        if keepdim {
            out_shape[norm_dim] = 1;
        } else {
            out_shape.remove(norm_dim);
        }
        let handle = if is_f32 {
            backend.max_axis_f32(input.gpu_handle()?, input.shape(), norm_dim)?
        } else {
            backend.max_axis_f64(input.gpu_handle()?, input.shape(), norm_dim)?
        };
        let compact_result =
            Tensor::from_storage(TensorStorage::gpu(handle), out_shape.clone(), false)?;
        if is_grad_enabled() && input.requires_grad() {
            let grad_fn = Arc::new(AmaxDimBackward {
                input: input.clone(),
                result: compact_result.clone(),
                dim: norm_dim,
                keepdim,
            });
            let (s, sh) = compact_result.into_storage_and_shape()?;
            return Tensor::from_operation(s, sh, grad_fn);
        }
        return Ok(compact_result);
    }
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "amax_dim" });
    }

    let (result, norm_dim, out_shape) = amin_amax_dim_forward(input, dim, keepdim, true)?;
    let result_t = Tensor::from_storage(TensorStorage::cpu(result), out_shape, false)?;
    if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(AmaxDimBackward {
            input: input.clone(),
            result: result_t.clone(),
            dim: norm_dim,
            keepdim,
        });
        let (s, sh) = result_t.into_storage_and_shape()?;
        Tensor::from_operation(s, sh, grad_fn)
    } else {
        Ok(result_t)
    }
}

/// Helper: 0-D scalar passes through `amin`/`amax_dim` unchanged
/// (matches torch's `amin(scalar, dim=0)` returning the scalar).
fn grad_clone<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let data = input.data()?.to_vec();
    Tensor::from_storage(TensorStorage::cpu(data), input.shape().to_vec(), false)
}

// ---------------------------------------------------------------------------
// prod_dim (differentiable — prefix-suffix product VJP per upstream's
// `prod_backward(grad, self.to(grad.scalar_type()), result)` family at
// `tools/autograd/derivatives.yaml:1413-1415`, adapted to the per-slice
// dim-keyed form via the same zero-safe prefix-suffix scan)
// ---------------------------------------------------------------------------
//
// Upstream: `Tensor prod(const Tensor& self, int64_t dim, bool keepdim,
// std::optional<ScalarType> dtype)` at `aten/src/ATen/native/ReduceOps.cpp`
// (the `prod.dim_int` overload).

#[derive(Debug)]
pub struct ProdDimBackward<T: Float> {
    input: Tensor<T>,
    dim: usize,
    keepdim: bool,
}

impl<T: Float> GradFn<T> for ProdDimBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if self.input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "prod_dim backward",
            });
        }
        let input_data = self.input.data()?;
        let in_shape = self.input.shape();
        let dim_size = in_shape[self.dim];
        let outer: usize = in_shape[..self.dim].iter().product();
        let inner: usize = in_shape[self.dim + 1..].iter().product();
        let _ = self.keepdim; // grad_output's shape carries the keepdim info

        let go_data = grad_output.data()?;
        // Per-slice prefix-suffix scan: for slice (o, i), gradient w.r.t.
        // position d is grad_output[o, i] * prefix[d] * suffix[d].
        let one = <T as num_traits::One>::one();
        let mut out = vec![<T as num_traits::Zero>::zero(); in_shape.iter().product()];
        for o in 0..outer {
            for i in 0..inner {
                let mut prefix = vec![one; dim_size];
                let mut suffix = vec![one; dim_size];
                for d in 1..dim_size {
                    prefix[d] =
                        prefix[d - 1] * input_data[o * dim_size * inner + (d - 1) * inner + i];
                }
                if dim_size > 1 {
                    for d in (0..dim_size - 1).rev() {
                        suffix[d] =
                            suffix[d + 1] * input_data[o * dim_size * inner + (d + 1) * inner + i];
                    }
                }
                let g = go_data[o * inner + i];
                for d in 0..dim_size {
                    out[o * dim_size * inner + d * inner + i] = g * prefix[d] * suffix[d];
                }
            }
        }
        let grad_input = Tensor::from_storage(TensorStorage::cpu(out), in_shape.to_vec(), false)?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "ProdDimBackward"
    }
}

/// Differentiable dim-keyed product. Mirrors
/// `torch.prod(input, dim, keepdim=False)`.
pub fn prod_dim<T: Float>(
    input: &Tensor<T>,
    dim: i64,
    keepdim: bool,
) -> FerrotorchResult<Tensor<T>> {
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "prod_dim" });
    }
    let ndim = input.ndim();
    if ndim == 0 {
        return grad_clone(input);
    }
    let norm_dim = if dim < 0 {
        (ndim as i64 + dim) as usize
    } else {
        dim as usize
    };
    if norm_dim >= ndim {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("prod_dim: dim {dim} out of bounds for {ndim}-D tensor"),
        });
    }
    let input_ref = if input.is_contiguous() {
        input.clone()
    } else {
        input.contiguous()?
    };
    let in_data = input_ref.data()?;
    let in_shape = input_ref.shape();
    let dim_size = in_shape[norm_dim];
    let outer: usize = in_shape[..norm_dim].iter().product();
    let inner: usize = in_shape[norm_dim + 1..].iter().product();
    let one = <T as num_traits::One>::one();
    let mut result = Vec::with_capacity(outer * inner);
    for o in 0..outer {
        for i in 0..inner {
            let mut acc = one;
            for d in 0..dim_size {
                acc = acc * in_data[o * dim_size * inner + d * inner + i];
            }
            result.push(acc);
        }
    }
    let mut out_shape: Vec<usize> = in_shape.to_vec();
    if keepdim {
        out_shape[norm_dim] = 1;
    } else {
        out_shape.remove(norm_dim);
    }
    let result_t = Tensor::from_storage(TensorStorage::cpu(result), out_shape, false)?;
    if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(ProdDimBackward {
            input: input.clone(),
            dim: norm_dim,
            keepdim,
        });
        let (s, sh) = result_t.into_storage_and_shape()?;
        Tensor::from_operation(s, sh, grad_fn)
    } else {
        Ok(result_t)
    }
}

// ---------------------------------------------------------------------------
// max(dim) / min(dim) — tuple return (values, indices). Differentiable.
// ---------------------------------------------------------------------------
//
// Mirrors `aten/src/ATen/native/ReduceOps.cpp` `max.dim` / `min.dim` overloads
// — both return a `(values, indices)` named tuple. VJP per
// `tools/autograd/derivatives.yaml`:
//   - name: max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
//     self: value_selecting_reduction_backward(grad, dim, indices, self.sizes(), keepdim)
// — the gradient is scatter-routed to the `indices` positions only.
//
// NaN propagation matches `aten/src/ATen/native/SharedReduceOps.h:26-34
// `max_propagate_nan` / `min_propagate_nan`: when a slice contains NaN, the
// extremum reported is NaN, and the recorded index is the FIRST NaN
// position. The CPU walk below mirrors that semantics. Closes #1302.

#[derive(Debug)]
pub struct MaxMinDimBackward<T: Float> {
    input: Tensor<T>,
    /// Flat index per output slot (outer * inner-many). Drives the
    /// scatter-routing in backward — gradient lands at exactly these
    /// positions in the input's flat buffer.
    indices_flat: Vec<i64>,
    dim: usize,
    keepdim: bool,
    /// Human-readable kind for the `name()` reporter ("MaxDimBackward" /
    /// "MinDimBackward").
    name: &'static str,
}

impl<T: Float> GradFn<T> for MaxMinDimBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if self.input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "max_with_dim/min_with_dim backward",
            });
        }
        let in_shape = self.input.shape();
        let dim_size = in_shape[self.dim];
        let outer: usize = in_shape[..self.dim].iter().product();
        let inner: usize = in_shape[self.dim + 1..].iter().product();
        let go = grad_output.data()?;
        let _ = self.keepdim; // shape info absorbed via the (o*inner + i) flat layout

        let zero = <T as num_traits::Zero>::zero();
        let in_numel: usize = in_shape.iter().product();
        let mut out = vec![zero; in_numel];
        // For each output slot (o, i), drop grad_output[o*inner+i] at
        // input position (o, indices_flat[o*inner+i], i).
        for o in 0..outer {
            for i in 0..inner {
                let slot = o * inner + i;
                let d = self.indices_flat[slot] as usize;
                debug_assert!(d < dim_size);
                let flat_in = o * dim_size * inner + d * inner + i;
                out[flat_in] = go[slot];
            }
        }
        let grad_input = Tensor::from_storage(TensorStorage::cpu(out), in_shape.to_vec(), false)?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        self.name
    }
}

#[allow(
    clippy::type_complexity,
    reason = "single-use helper returning (values, indices_flat, indices_int, out_shape, dim); \
              wrapping in a struct adds boilerplate without aiding the two callers."
)]
fn max_min_with_dim_forward<T: Float>(
    input: &Tensor<T>,
    dim: i64,
    keepdim: bool,
    find_max: bool,
) -> FerrotorchResult<(Vec<T>, Vec<i64>, IntTensor<i64>, Vec<usize>, usize)> {
    let ndim = input.ndim();
    if ndim == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "max/min_with_dim: cannot reduce a 0-D tensor along a dimension".into(),
        });
    }
    let norm_dim = if dim < 0 {
        (ndim as i64 + dim) as usize
    } else {
        dim as usize
    };
    if norm_dim >= ndim {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("max/min_with_dim: dim {dim} out of bounds for {ndim}-D tensor"),
        });
    }
    let input_ref = if input.is_contiguous() {
        input.clone()
    } else {
        input.contiguous()?
    };
    let in_data = input_ref.data()?;
    let in_shape = input_ref.shape();
    let dim_size = in_shape[norm_dim];
    let outer: usize = in_shape[..norm_dim].iter().product();
    let inner: usize = in_shape[norm_dim + 1..].iter().product();

    let mut values = Vec::with_capacity(outer * inner);
    let mut indices = Vec::with_capacity(outer * inner);
    for o in 0..outer {
        for i in 0..inner {
            let base = o * dim_size * inner + i;
            let mut best = in_data[base];
            let mut best_idx: i64 = 0;
            for d in 1..dim_size {
                let v = in_data[base + d * inner];
                // NaN-poison per upstream `SharedReduceOps.h:26-34`.
                let take = if find_max {
                    v.is_nan() || (!best.is_nan() && v > best)
                } else {
                    v.is_nan() || (!best.is_nan() && v < best)
                };
                if take {
                    best = v;
                    best_idx = d as i64;
                }
            }
            values.push(best);
            indices.push(best_idx);
        }
    }
    let mut out_shape: Vec<usize> = in_shape.to_vec();
    if keepdim {
        out_shape[norm_dim] = 1;
    } else {
        out_shape.remove(norm_dim);
    }
    let indices_int = IntTensor::<i64>::from_vec(indices.clone(), out_shape.clone())?;
    Ok((values, indices, indices_int, out_shape, norm_dim))
}

/// Differentiable `(values, indices) = max(input, dim, keepdim)` with the
/// PyTorch named-tuple return. Mirrors `torch.max(input, dim, keepdim)` at
/// `aten/src/ATen/native/ReduceOps.cpp` `max.dim` overload. NaN propagation
/// per `SharedReduceOps.h:26-34`. Backward scatters `grad` to the input
/// positions identified by `indices`. Closes #1302 (max).
pub fn max_with_dim<T: Float>(
    input: &Tensor<T>,
    dim: i64,
    keepdim: bool,
) -> FerrotorchResult<(Tensor<T>, IntTensor<i64>)> {
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "max_with_dim" });
    }
    let (values, indices_flat, indices_int, out_shape, norm_dim) =
        max_min_with_dim_forward(input, dim, keepdim, true)?;
    let storage = TensorStorage::cpu(values);
    let values_t = Tensor::from_storage(storage, out_shape, false)?;
    let values_t = if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(MaxMinDimBackward {
            input: input.clone(),
            indices_flat,
            dim: norm_dim,
            keepdim,
            name: "MaxDimBackward",
        });
        let (s, sh) = values_t.into_storage_and_shape()?;
        Tensor::from_operation(s, sh, grad_fn)?
    } else {
        values_t
    };
    Ok((values_t, indices_int))
}

/// Differentiable `(values, indices) = min(input, dim, keepdim)` —
/// symmetric to [`max_with_dim`]. Closes #1302 (min).
pub fn min_with_dim<T: Float>(
    input: &Tensor<T>,
    dim: i64,
    keepdim: bool,
) -> FerrotorchResult<(Tensor<T>, IntTensor<i64>)> {
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "min_with_dim" });
    }
    let (values, indices_flat, indices_int, out_shape, norm_dim) =
        max_min_with_dim_forward(input, dim, keepdim, false)?;
    let storage = TensorStorage::cpu(values);
    let values_t = Tensor::from_storage(storage, out_shape, false)?;
    let values_t = if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(MaxMinDimBackward {
            input: input.clone(),
            indices_flat,
            dim: norm_dim,
            keepdim,
            name: "MinDimBackward",
        });
        let (s, sh) = values_t.into_storage_and_shape()?;
        Tensor::from_operation(s, sh, grad_fn)?
    } else {
        values_t
    };
    Ok((values_t, indices_int))
}

// ---------------------------------------------------------------------------
// median(dim) / nanmedian(dim) — tuple return (values, indices). Differentiable.
// ---------------------------------------------------------------------------
//
// Mirrors `aten/src/ATen/native/Sorting.cpp:503-607 median_with_indices_impl`
// — both `median.dim` and `nanmedian.dim` return a `(values, indices)` named
// tuple where `indices` is the position OF the median element IN the original
// (unsorted) slice. Semantics per the upstream loop body:
//   * median (ignore_nan=false): if the slice contains any NaN, the result is
//     that NaN and its index (first NaN). Otherwise the lower median at sorted
//     rank `(size - 1) / 2` with tie-break `ip[i] < ip[j] || (ip[i]==ip[j] && i<j)`.
//   * nanmedian (ignore_nan=true): NaNs are excluded; the median is the
//     non-NaN element at rank `(size - num_nan - 1) / 2`. If the entire slice
//     is NaN (num_nan == size), the rank-`(size-1)/2` element (a NaN) is taken.
// The reported index is into the original slice (not the sorted order),
// matching `*indp = *nth` where `nth` indexes the iota-initialised index vec.
//
// VJP per `tools/autograd/derivatives.yaml:1179-1185`:
//   self: value_selecting_reduction_backward(grad, dim, indices, self.sizes(), keepdim)
// — identical scatter-routing to `max.dim`/`min.dim`, so the existing
// `MaxMinDimBackward` (which scatters grad at the saved per-slice flat index)
// is reused verbatim. Closes #1306.

/// Forward kernel shared by `median_with_dim` / `nanmedian_with_dim`. Sorts a
/// stable copy of each slice's indices and selects the median rank. Returns
/// `(values, indices_flat, indices_int, out_shape, norm_dim)` mirroring
/// `max_min_with_dim_forward`. `ignore_nan` toggles the median vs. nanmedian
/// branch per `Sorting.cpp:562-596`.
#[allow(
    clippy::type_complexity,
    reason = "single-use helper returning (values, indices_flat, indices_int, out_shape, dim); \
              matches max_min_with_dim_forward's tuple shape for the two callers."
)]
fn median_with_dim_forward<T: Float>(
    input: &Tensor<T>,
    dim: i64,
    keepdim: bool,
    ignore_nan: bool,
) -> FerrotorchResult<(Vec<T>, Vec<i64>, IntTensor<i64>, Vec<usize>, usize)> {
    let ndim = input.ndim();
    if ndim == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "median/nanmedian_with_dim: cannot reduce a 0-D tensor along a dimension"
                .into(),
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
                "median/nanmedian_with_dim: dim {dim} out of bounds for {ndim}-D tensor"
            ),
        });
    }
    let input_ref = if input.is_contiguous() {
        input.clone()
    } else {
        input.contiguous()?
    };
    let in_data = input_ref.data()?;
    let in_shape = input_ref.shape();
    let dim_size = in_shape[norm_dim];
    let outer: usize = in_shape[..norm_dim].iter().product();
    let inner: usize = in_shape[norm_dim + 1..].iter().product();

    let mut values = Vec::with_capacity(outer * inner);
    let mut indices = Vec::with_capacity(outer * inner);
    for o in 0..outer {
        for i in 0..inner {
            let base = o * dim_size * inner + i;
            // Gather the slice's (local-index, value) pairs.
            let slice: Vec<(usize, T)> = (0..dim_size)
                .map(|d| (d, in_data[base + d * inner]))
                .collect();

            // median (not nanmedian): a NaN anywhere wins, at its first index.
            if !ignore_nan && let Some(&(nan_idx, nan_val)) = slice.iter().find(|(_, v)| v.is_nan())
            {
                values.push(nan_val);
                indices.push(nan_idx as i64);
                continue;
            }

            // Stable sort by value with the upstream tie-break: equal values
            // keep ascending original index (`ip[i]==ip[j] && i<j`). NaNs are
            // ordered to the tail for the nanmedian branch.
            let mut order: Vec<usize> = (0..dim_size).collect();
            order.sort_by(|&i, &j| {
                let vi = slice[i].1;
                let vj = slice[j].1;
                let i_nan = vi.is_nan();
                let j_nan = vj.is_nan();
                match (i_nan, j_nan) {
                    (true, true) => i.cmp(&j),
                    (true, false) => std::cmp::Ordering::Greater, // NaN sinks to tail
                    (false, true) => std::cmp::Ordering::Less,
                    (false, false) => match vi.partial_cmp(&vj) {
                        Some(std::cmp::Ordering::Equal) | None => i.cmp(&j),
                        Some(ord) => ord,
                    },
                }
            });

            // Median rank: lower median over the non-NaN prefix (nanmedian) or
            // the full slice (median, which has no NaN at this point).
            let num_nan = if ignore_nan {
                slice.iter().filter(|(_, v)| v.is_nan()).count()
            } else {
                0
            };
            let effective = dim_size - num_nan;
            let rank = if effective == 0 {
                // Whole slice is NaN under nanmedian: upstream nth still points
                // into the (NaN-only) sorted order at (size-1)/2.
                (dim_size - 1) / 2
            } else {
                (effective - 1) / 2
            };
            let median_local = order[rank];
            values.push(slice[median_local].1);
            indices.push(median_local as i64);
        }
    }

    let mut out_shape: Vec<usize> = in_shape.to_vec();
    if keepdim {
        out_shape[norm_dim] = 1;
    } else {
        out_shape.remove(norm_dim);
    }
    let indices_int = IntTensor::<i64>::from_vec(indices.clone(), out_shape.clone())?;
    Ok((values, indices, indices_int, out_shape, norm_dim))
}

/// Differentiable `(values, indices) = median(input, dim, keepdim)` with the
/// PyTorch named-tuple return. Mirrors `torch.median(input, dim, keepdim)` at
/// `aten/src/ATen/native/Sorting.cpp:503 median_with_indices_impl` (ignore_nan
/// = false: a NaN in the slice poisons the result). Backward scatters `grad`
/// to the input positions identified by `indices` via the shared
/// `MaxMinDimBackward`. Closes #1306 (median).
pub fn median_with_dim<T: Float>(
    input: &Tensor<T>,
    dim: i64,
    keepdim: bool,
) -> FerrotorchResult<(Tensor<T>, IntTensor<i64>)> {
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda {
            op: "median_with_dim",
        });
    }
    let (values, indices_flat, indices_int, out_shape, norm_dim) =
        median_with_dim_forward(input, dim, keepdim, false)?;
    let storage = TensorStorage::cpu(values);
    let values_t = Tensor::from_storage(storage, out_shape, false)?;
    let values_t = if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(MaxMinDimBackward {
            input: input.clone(),
            indices_flat,
            dim: norm_dim,
            keepdim,
            name: "MedianDimBackward",
        });
        let (s, sh) = values_t.into_storage_and_shape()?;
        Tensor::from_operation(s, sh, grad_fn)?
    } else {
        values_t
    };
    Ok((values_t, indices_int))
}

/// Differentiable `(values, indices) = nanmedian(input, dim, keepdim)` —
/// NaN-skipping counterpart of [`median_with_dim`]. Mirrors
/// `torch.nanmedian(input, dim, keepdim)` (`ignore_nan = true`): NaNs are
/// excluded from the median rank computation. Closes #1306 (nanmedian).
pub fn nanmedian_with_dim<T: Float>(
    input: &Tensor<T>,
    dim: i64,
    keepdim: bool,
) -> FerrotorchResult<(Tensor<T>, IntTensor<i64>)> {
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda {
            op: "nanmedian_with_dim",
        });
    }
    let (values, indices_flat, indices_int, out_shape, norm_dim) =
        median_with_dim_forward(input, dim, keepdim, true)?;
    let storage = TensorStorage::cpu(values);
    let values_t = Tensor::from_storage(storage, out_shape, false)?;
    let values_t = if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(MaxMinDimBackward {
            input: input.clone(),
            indices_flat,
            dim: norm_dim,
            keepdim,
            name: "NanmedianDimBackward",
        });
        let (s, sh) = values_t.into_storage_and_shape()?;
        Tensor::from_operation(s, sh, grad_fn)?
    } else {
        values_t
    };
    Ok((values_t, indices_int))
}

// ---------------------------------------------------------------------------
// norm(input, p, dim, keepdim) — p-norm along a dimension.
// ---------------------------------------------------------------------------
//
// Mirrors `aten/src/ATen/native/ReduceOps.cpp` `linalg_vector_norm.out` and
// the `Tensor::norm(p, dim, keepdim)` overload at `aten/src/ATen/native/
// LinearAlgebra.cpp`. Forward: `result = (sum(|x|^p, dim))^(1/p)`. Backward
// per `tools/autograd/derivatives.yaml`:
//   - name: norm.ScalarOpt_dim(Tensor self, Scalar? p, int[1] dim, bool keepdim=False) -> Tensor
//     self: norm_backward(grad, self, p, result, dim, keepdim)
// reducing (for non-degenerate `result`) to:
//   d/dx_j = |x_j|^(p-1) * sign(x_j) * (result_broadcast)^(1-p) * grad_broadcast
// with `result == 0 -> 0` mask (every input was zero, no gradient signal).
//
// Closes #1308.

#[derive(Debug)]
pub struct NormDimBackward<T: Float> {
    input: Tensor<T>,
    p: f64,
    /// Saved forward output in *keepdim shape* so broadcast indexing
    /// stays simple in backward (re-insert size-1 dim if forward squeezed).
    result_keepdim: Tensor<T>,
    dim: usize,
}

impl<T: Float> GradFn<T> for NormDimBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if self.input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "norm_with_dim backward",
            });
        }
        let in_shape = self.input.shape();
        let dim_size = in_shape[self.dim];
        let outer: usize = in_shape[..self.dim].iter().product();
        let inner: usize = in_shape[self.dim + 1..].iter().product();

        let in_data = self.input.data()?;
        let go = grad_output.data()?;
        let res = self.result_keepdim.data()?;
        let p = self.p;
        let zero = <T as num_traits::Zero>::zero();
        let one_f64 = 1.0_f64;
        let in_numel: usize = in_shape.iter().product();
        let mut out = vec![zero; in_numel];

        for o in 0..outer {
            for i in 0..inner {
                let slot = o * inner + i;
                let r_f = to_f64::<T>(res[slot])?;
                if r_f == 0.0 {
                    // result==0 means every input in this slice is 0 → grad is 0.
                    continue;
                }
                let g_f = to_f64::<T>(go[slot])?;
                let scale_pow = r_f.powf(one_f64 - p);
                for d in 0..dim_size {
                    let xf = to_f64::<T>(in_data[o * dim_size * inner + d * inner + i])?;
                    let abs_x = xf.abs();
                    let s = if xf > 0.0 {
                        1.0
                    } else if xf < 0.0 {
                        -1.0
                    } else {
                        0.0
                    };
                    let grad_xf = g_f * abs_x.powf(p - one_f64) * s * scale_pow;
                    out[o * dim_size * inner + d * inner + i] = float_from_f64::<T>(grad_xf)?;
                }
            }
        }
        let grad_input = Tensor::from_storage(TensorStorage::cpu(out), in_shape.to_vec(), false)?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "NormDimBackward"
    }
}

/// Differentiable p-norm along a dimension: `result = (sum(|x|^p, dim))^(1/p)`.
/// Mirrors `aten/src/ATen/native/ReduceOps.cpp` `linalg_vector_norm` / the
/// `Tensor::norm(p, dim, keepdim)` overload. Backward per
/// `tools/autograd/derivatives.yaml` `norm.ScalarOpt_dim`. Closes #1308.
pub fn norm_with_dim<T: Float>(
    input: &Tensor<T>,
    p: f64,
    dim: i64,
    keepdim: bool,
) -> FerrotorchResult<Tensor<T>> {
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda {
            op: "norm_with_dim",
        });
    }
    if !(p.is_finite() && p > 0.0) {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "norm_with_dim: p must be finite and > 0; got {p}. Other norms (inf, 0) \
                 are non-differentiable / piecewise and tracked separately."
            ),
        });
    }
    let ndim = input.ndim();
    if ndim == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "norm_with_dim: cannot reduce a 0-D tensor along a dimension".into(),
        });
    }
    let norm_dim = if dim < 0 {
        (ndim as i64 + dim) as usize
    } else {
        dim as usize
    };
    if norm_dim >= ndim {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("norm_with_dim: dim {dim} out of bounds for {ndim}-D tensor"),
        });
    }
    let input_ref = if input.is_contiguous() {
        input.clone()
    } else {
        input.contiguous()?
    };
    let in_data = input_ref.data()?;
    let in_shape = input_ref.shape();
    let dim_size = in_shape[norm_dim];
    let outer: usize = in_shape[..norm_dim].iter().product();
    let inner: usize = in_shape[norm_dim + 1..].iter().product();
    let mut result_keepdim_data = Vec::with_capacity(outer * inner);

    // L2 fast path (#1614): for `p == 2.0` over a CONTIGUOUS last-dim slice
    // (`inner == 1`) with an f32 dtype, torch's `at::norm(2.0)` goes through the
    // vectorized last-dim L2 kernel (`ReduceOpsKernel.cpp:222-255`) — a width-8
    // lane accumulate + left-fold + scalar FMA tail + `sqrt`, with an f32 (NOT
    // f64) accumulator (`opmath_type<float> == float`, `OpMathType.h:16`). The
    // generic `Σ |v|^p` then `^(1/p)` f64 path below differs from that by up to
    // one ULP, which flips boundary decisions. Route the f32 last-dim L2 case
    // through the shared `simd_reduce::l2_norm_f32_torch` primitive so it
    // matches torch byte-for-byte (modulo the documented ~3% residual).
    let t_is_f32 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>();
    #[allow(
        clippy::float_cmp,
        reason = "exact `p == 2.0` mirrors torch's norm-kernel dispatch `if (val == 2.0)` at ReduceOpsKernel.cpp:195 — the L2 vectorized path is selected by exact equality, not an epsilon band; a margin compare would mis-route p values near 2.0 that torch routes to the generic NormOps path"
    )]
    let is_l2_lastdim_f32 = p == 2.0 && t_is_f32 && inner == 1;
    if is_l2_lastdim_f32 {
        // `inner == 1` means each reduced slice is `dim_size` CONTIGUOUS
        // elements at `o * dim_size .. (o + 1) * dim_size`. Collect them as
        // f32 (T is f32 here) and reduce with the torch-matching primitive.
        for o in 0..outer {
            let slice_start = o * dim_size;
            let mut row: Vec<f32> = Vec::with_capacity(dim_size);
            for d in 0..dim_size {
                let v = in_data[slice_start + d];
                // T == f32 here, so this conversion is exact (identity).
                row.push(num_traits::ToPrimitive::to_f32(&v).ok_or(
                    FerrotorchError::InvalidArgument {
                        message: "norm_with_dim: f32 element not representable".into(),
                    },
                )?);
            }
            let norm_f32 = crate::simd_reduce::l2_norm_f32_torch(&row);
            result_keepdim_data.push(float_from_f64::<T>(f64::from(norm_f32))?);
        }
    } else {
        for o in 0..outer {
            for i in 0..inner {
                let mut acc = 0.0_f64;
                for d in 0..dim_size {
                    let v = to_f64::<T>(in_data[o * dim_size * inner + d * inner + i])?;
                    acc += v.abs().powf(p);
                }
                let r = acc.powf(1.0 / p);
                result_keepdim_data.push(float_from_f64::<T>(r)?);
            }
        }
    }
    let mut keepdim_shape: Vec<usize> = in_shape.to_vec();
    keepdim_shape[norm_dim] = 1;
    let result_keepdim = Tensor::from_storage(
        TensorStorage::cpu(result_keepdim_data.clone()),
        keepdim_shape.clone(),
        false,
    )?;
    let mut out_shape = keepdim_shape.clone();
    if !keepdim {
        out_shape.remove(norm_dim);
    }
    let final_result =
        Tensor::from_storage(TensorStorage::cpu(result_keepdim_data), out_shape, false)?;

    if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(NormDimBackward {
            input: input.clone(),
            p,
            result_keepdim,
            dim: norm_dim,
        });
        let (s, sh) = final_result.into_storage_and_shape()?;
        Tensor::from_operation(s, sh, grad_fn)
    } else {
        Ok(final_result)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::no_grad::no_grad;
    use crate::storage::TensorStorage;

    /// Helper: create a leaf tensor with given data, shape, and requires_grad.
    fn leaf(data: &[f64], shape: &[usize], requires_grad: bool) -> Tensor<f64> {
        Tensor::from_storage(
            TensorStorage::cpu(data.to_vec()),
            shape.to_vec(),
            requires_grad,
        )
        .unwrap()
    }

    /// Helper: create a leaf scalar.
    fn leaf_scalar(val: f64, requires_grad: bool) -> Tensor<f64> {
        Tensor::from_storage(TensorStorage::cpu(vec![val]), vec![], requires_grad).unwrap()
    }

    // --- Forward tests ---

    #[test]
    fn test_sum_forward_1d() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[4], false);
        let s = sum(&x).unwrap();
        assert!(s.is_scalar());
        assert!((s.item().unwrap() - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_sum_forward_2d() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        let s = sum(&x).unwrap();
        assert!((s.item().unwrap() - 21.0).abs() < 1e-12);
    }

    #[test]
    fn test_mean_forward() {
        let x = leaf(&[2.0, 4.0, 6.0, 8.0], &[4], false);
        let m = mean(&x).unwrap();
        assert!((m.item().unwrap() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_prod_forward() {
        let x = leaf(&[2.0, 3.0, 4.0], &[3], false);
        let p = prod(&x).unwrap();
        assert!((p.item().unwrap() - 24.0).abs() < 1e-12);
    }

    #[test]
    fn test_prod_forward_scalar() {
        let x = leaf_scalar(7.0, false);
        let p = prod(&x).unwrap();
        assert!((p.item().unwrap() - 7.0).abs() < 1e-12);
    }

    #[test]
    fn test_prod_forward_with_zero() {
        let x = leaf(&[3.0, 0.0, 5.0], &[3], false);
        let p = prod(&x).unwrap();
        assert!((p.item().unwrap()).abs() < 1e-12);
    }

    // --- Backward tests ---

    #[test]
    fn test_sum_backward_scalar_input() {
        // sum(x) where x is a scalar = x. Gradient should be 1.
        let x = leaf_scalar(5.0, true);
        let s = sum(&x).unwrap();
        s.backward().unwrap();

        let g = x.grad().unwrap().unwrap();
        assert!((g.item().unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_sum_backward_1d() {
        // sum([a, b, c]) = a + b + c. d/d(each) = 1.
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let s = sum(&x).unwrap();
        s.backward().unwrap();

        let g = x.grad().unwrap().unwrap();
        let gd = g.data().unwrap();
        assert_eq!(gd.len(), 3);
        for &v in gd {
            assert!((v - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_sum_backward_2d() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
        let s = sum(&x).unwrap();
        s.backward().unwrap();

        let g = x.grad().unwrap().unwrap();
        assert_eq!(g.shape(), &[2, 3]);
        for &v in g.data().unwrap() {
            assert!((v - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_mean_backward_scalar_input() {
        // mean(x) where x is a scalar = x. Gradient should be 1.
        let x = leaf_scalar(5.0, true);
        let m = mean(&x).unwrap();
        m.backward().unwrap();

        let g = x.grad().unwrap().unwrap();
        assert!((g.item().unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_mean_backward_1d() {
        // mean([a, b, c]) = (a + b + c) / 3. d/d(each) = 1/3.
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let m = mean(&x).unwrap();
        m.backward().unwrap();

        let g = x.grad().unwrap().unwrap();
        let gd = g.data().unwrap();
        let expected = 1.0 / 3.0;
        for &v in gd {
            assert!((v - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn test_mean_backward_2d() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
        let m = mean(&x).unwrap();
        m.backward().unwrap();

        let g = x.grad().unwrap().unwrap();
        assert_eq!(g.shape(), &[2, 3]);
        let expected = 1.0 / 6.0;
        for &v in g.data().unwrap() {
            assert!((v - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn test_prod_backward_scalar_input() {
        // prod(x) where x is scalar = x. Gradient should be 1.
        let x = leaf_scalar(5.0, true);
        let p = prod(&x).unwrap();
        p.backward().unwrap();

        let g = x.grad().unwrap().unwrap();
        assert!((g.item().unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_prod_backward_1d() {
        // prod([a, b, c]) = a*b*c.
        // d/da = b*c, d/db = a*c, d/dc = a*b.
        let x = leaf(&[2.0, 3.0, 4.0], &[3], true);
        let p = prod(&x).unwrap();
        p.backward().unwrap();

        let g = x.grad().unwrap().unwrap();
        let gd = g.data().unwrap();
        assert!(
            (gd[0] - 12.0).abs() < 1e-12,
            "d/da = 3*4 = 12, got {}",
            gd[0]
        );
        assert!((gd[1] - 8.0).abs() < 1e-12, "d/db = 2*4 = 8, got {}", gd[1]);
        assert!((gd[2] - 6.0).abs() < 1e-12, "d/dc = 2*3 = 6, got {}", gd[2]);
    }

    #[test]
    fn test_prod_backward_with_zero() {
        // prod([3, 0, 5]) = 0.
        // d/d(x0) = 0*5 = 0, d/d(x1) = 3*5 = 15, d/d(x2) = 3*0 = 0.
        let x = leaf(&[3.0, 0.0, 5.0], &[3], true);
        let p = prod(&x).unwrap();
        p.backward().unwrap();

        let g = x.grad().unwrap().unwrap();
        let gd = g.data().unwrap();
        assert!((gd[0] - 0.0).abs() < 1e-12, "got {}", gd[0]);
        assert!((gd[1] - 15.0).abs() < 1e-12, "got {}", gd[1]);
        assert!((gd[2] - 0.0).abs() < 1e-12, "got {}", gd[2]);
    }

    #[test]
    fn test_prod_backward_two_zeros() {
        // prod([0, 0, 5]) = 0.
        // All gradients should be 0 (each product-excluding-one still contains a zero).
        let x = leaf(&[0.0, 0.0, 5.0], &[3], true);
        let p = prod(&x).unwrap();
        p.backward().unwrap();

        let g = x.grad().unwrap().unwrap();
        let gd = g.data().unwrap();
        for &v in gd {
            assert!((v).abs() < 1e-12, "expected 0, got {v}");
        }
    }

    // --- Gradient tracking / no_grad tests ---

    #[test]
    fn test_sum_no_grad_fn_when_input_not_requires_grad() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], false);
        let s = sum(&x).unwrap();
        assert!(s.grad_fn().is_none());
        assert!(!s.requires_grad());
    }

    #[test]
    fn test_sum_has_grad_fn_when_input_requires_grad() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let s = sum(&x).unwrap();
        assert!(s.grad_fn().is_some());
        assert_eq!(s.grad_fn().unwrap().name(), "SumBackward");
        assert!(s.requires_grad());
    }

    #[test]
    fn test_mean_has_grad_fn_when_input_requires_grad() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let m = mean(&x).unwrap();
        assert!(m.grad_fn().is_some());
        assert_eq!(m.grad_fn().unwrap().name(), "MeanBackward");
    }

    #[test]
    fn test_prod_has_grad_fn_when_input_requires_grad() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let p = prod(&x).unwrap();
        assert!(p.grad_fn().is_some());
        assert_eq!(p.grad_fn().unwrap().name(), "ProdBackward");
    }

    #[test]
    fn test_sum_no_grad_fn_in_no_grad_context() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let s = no_grad(|| sum(&x)).unwrap();
        assert!(s.grad_fn().is_none());
        assert!(!s.requires_grad());
    }

    #[test]
    fn test_mean_no_grad_fn_in_no_grad_context() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let m = no_grad(|| mean(&x)).unwrap();
        assert!(m.grad_fn().is_none());
    }

    #[test]
    fn test_prod_no_grad_fn_in_no_grad_context() {
        let x = leaf(&[2.0, 3.0], &[2], true);
        let p = no_grad(|| prod(&x)).unwrap();
        assert!(p.grad_fn().is_none());
    }

    // --- Numerical gradient checking ---

    /// Finite-difference gradient check for a scalar -> scalar function.
    fn numerical_grad_check(
        f: impl Fn(&Tensor<f64>) -> FerrotorchResult<Tensor<f64>>,
        x_val: f64,
        expected_analytic: f64,
        tol: f64,
    ) {
        let eps = 1e-7;

        let x_plus = leaf_scalar(x_val + eps, false);
        let x_minus = leaf_scalar(x_val - eps, false);

        let f_plus = f(&x_plus).unwrap().item().unwrap();
        let f_minus = f(&x_minus).unwrap().item().unwrap();
        let numerical = (f_plus - f_minus) / (2.0 * eps);

        assert!(
            (numerical - expected_analytic).abs() < tol,
            "numerical gradient {numerical} differs from analytic {expected_analytic} by more than {tol}"
        );
    }

    #[test]
    fn test_sum_numerical_gradient() {
        // sum(x) for scalar x: d/dx = 1.
        let x = leaf_scalar(3.0, true);
        let s = sum(&x).unwrap();
        s.backward().unwrap();
        let analytic = x.grad().unwrap().unwrap().item().unwrap();

        numerical_grad_check(sum, 3.0, analytic, 1e-5);
    }

    #[test]
    fn test_mean_numerical_gradient() {
        // mean(x) for scalar x: d/dx = 1.
        let x = leaf_scalar(3.0, true);
        let m = mean(&x).unwrap();
        m.backward().unwrap();
        let analytic = x.grad().unwrap().unwrap().item().unwrap();

        numerical_grad_check(mean, 3.0, analytic, 1e-5);
    }

    #[test]
    fn test_prod_numerical_gradient() {
        // prod(x) for scalar x: d/dx = 1.
        let x = leaf_scalar(3.0, true);
        let p = prod(&x).unwrap();
        p.backward().unwrap();
        let analytic = x.grad().unwrap().unwrap().item().unwrap();

        numerical_grad_check(prod, 3.0, analytic, 1e-5);
    }

    // --- sum_dim forward tests ---

    #[test]
    fn test_sum_dim_axis0_2d() {
        // [[1, 2, 3], [4, 5, 6]] sum along axis 0 -> [5, 7, 9]
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        let s = sum_dim(&x, 0, false).unwrap();
        assert_eq!(s.shape(), &[3]);
        let d = s.data().unwrap();
        assert!((d[0] - 5.0).abs() < 1e-12);
        assert!((d[1] - 7.0).abs() < 1e-12);
        assert!((d[2] - 9.0).abs() < 1e-12);
    }

    #[test]
    fn test_sum_dim_axis1_2d() {
        // [[1, 2, 3], [4, 5, 6]] sum along axis 1 -> [6, 15]
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        let s = sum_dim(&x, 1, false).unwrap();
        assert_eq!(s.shape(), &[2]);
        let d = s.data().unwrap();
        assert!((d[0] - 6.0).abs() < 1e-12);
        assert!((d[1] - 15.0).abs() < 1e-12);
    }

    #[test]
    fn test_sum_dim_keepdim_true() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        let s = sum_dim(&x, 0, true).unwrap();
        assert_eq!(s.shape(), &[1, 3]);
        let d = s.data().unwrap();
        assert!((d[0] - 5.0).abs() < 1e-12);
        assert!((d[1] - 7.0).abs() < 1e-12);
        assert!((d[2] - 9.0).abs() < 1e-12);
    }

    #[test]
    fn test_sum_dim_negative_dim() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        // dim=-1 means axis 1
        let s = sum_dim(&x, -1, false).unwrap();
        assert_eq!(s.shape(), &[2]);
        let d = s.data().unwrap();
        assert!((d[0] - 6.0).abs() < 1e-12);
        assert!((d[1] - 15.0).abs() < 1e-12);
    }

    #[test]
    fn test_sum_dim_1d() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[4], false);
        let s = sum_dim(&x, 0, false).unwrap();
        assert!(s.is_scalar());
        assert!((s.item().unwrap() - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_sum_dim_1d_keepdim() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[4], false);
        let s = sum_dim(&x, 0, true).unwrap();
        assert_eq!(s.shape(), &[1]);
        assert!((s.data().unwrap()[0] - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_sum_dim_3d() {
        // shape [2, 2, 3], sum along dim=1
        let data: Vec<f64> = (1..=12).map(|x| x as f64).collect();
        let x = leaf(&data, &[2, 2, 3], false);
        let s = sum_dim(&x, 1, false).unwrap();
        assert_eq!(s.shape(), &[2, 3]);
        let d = s.data().unwrap();
        // [1,2,3] + [4,5,6] = [5,7,9]
        assert!((d[0] - 5.0).abs() < 1e-12);
        assert!((d[1] - 7.0).abs() < 1e-12);
        assert!((d[2] - 9.0).abs() < 1e-12);
        // [7,8,9] + [10,11,12] = [17,19,21]
        assert!((d[3] - 17.0).abs() < 1e-12);
        assert!((d[4] - 19.0).abs() < 1e-12);
        assert!((d[5] - 21.0).abs() < 1e-12);
    }

    // --- sum_dim backward tests ---

    #[test]
    fn test_sum_dim_backward_axis0_no_keepdim() {
        // x: [2, 3], sum(dim=0) -> [3]
        // grad of sum along axis 0: each row gets the same gradient
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
        let s = sum_dim(&x, 0, false).unwrap();
        // sum the result to get a scalar for backward
        let loss = sum(&s).unwrap();
        loss.backward().unwrap();

        let g = x.grad().unwrap().unwrap();
        assert_eq!(g.shape(), &[2, 3]);
        for &v in g.data().unwrap() {
            assert!((v - 1.0).abs() < 1e-12, "expected 1.0, got {v}");
        }
    }

    #[test]
    fn test_sum_dim_backward_axis1_keepdim() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
        let s = sum_dim(&x, 1, true).unwrap();
        assert_eq!(s.shape(), &[2, 1]);
        let loss = sum(&s).unwrap();
        loss.backward().unwrap();

        let g = x.grad().unwrap().unwrap();
        assert_eq!(g.shape(), &[2, 3]);
        for &v in g.data().unwrap() {
            assert!((v - 1.0).abs() < 1e-12, "expected 1.0, got {v}");
        }
    }

    #[test]
    fn test_sum_dim_has_grad_fn() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let s = sum_dim(&x, 0, false).unwrap();
        assert!(s.grad_fn().is_some());
        assert_eq!(s.grad_fn().unwrap().name(), "SumDimBackward");
    }

    #[test]
    fn test_sum_dim_no_grad_fn_when_not_requires_grad() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], false);
        let s = sum_dim(&x, 0, false).unwrap();
        assert!(s.grad_fn().is_none());
    }

    #[test]
    fn test_sum_dim_no_grad_fn_in_no_grad_context() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let s = no_grad(|| sum_dim(&x, 0, false)).unwrap();
        assert!(s.grad_fn().is_none());
    }

    // --- mean_dim forward tests ---

    #[test]
    fn test_mean_dim_axis0_2d() {
        // [[1, 2, 3], [4, 5, 6]] mean along axis 0 -> [2.5, 3.5, 4.5]
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        let m = mean_dim(&x, 0, false).unwrap();
        assert_eq!(m.shape(), &[3]);
        let d = m.data().unwrap();
        assert!((d[0] - 2.5).abs() < 1e-12);
        assert!((d[1] - 3.5).abs() < 1e-12);
        assert!((d[2] - 4.5).abs() < 1e-12);
    }

    #[test]
    fn test_mean_dim_axis1_2d() {
        // [[1, 2, 3], [4, 5, 6]] mean along axis 1 -> [2, 5]
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        let m = mean_dim(&x, 1, false).unwrap();
        assert_eq!(m.shape(), &[2]);
        let d = m.data().unwrap();
        assert!((d[0] - 2.0).abs() < 1e-12);
        assert!((d[1] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_mean_dim_keepdim() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        let m = mean_dim(&x, 0, true).unwrap();
        assert_eq!(m.shape(), &[1, 3]);
        let d = m.data().unwrap();
        assert!((d[0] - 2.5).abs() < 1e-12);
        assert!((d[1] - 3.5).abs() < 1e-12);
        assert!((d[2] - 4.5).abs() < 1e-12);
    }

    #[test]
    fn test_mean_dim_negative_dim() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        let m = mean_dim(&x, -1, false).unwrap();
        assert_eq!(m.shape(), &[2]);
        let d = m.data().unwrap();
        assert!((d[0] - 2.0).abs() < 1e-12);
        assert!((d[1] - 5.0).abs() < 1e-12);
    }

    // --- mean_dim backward tests ---

    #[test]
    fn test_mean_dim_backward_axis0() {
        // x: [2, 3], mean(dim=0) -> [3]
        // grad: each element gets 1/2 (since dim_size=2)
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
        let m = mean_dim(&x, 0, false).unwrap();
        let loss = sum(&m).unwrap();
        loss.backward().unwrap();

        let g = x.grad().unwrap().unwrap();
        assert_eq!(g.shape(), &[2, 3]);
        let expected = 1.0 / 2.0;
        for &v in g.data().unwrap() {
            assert!((v - expected).abs() < 1e-12, "expected {expected}, got {v}");
        }
    }

    #[test]
    fn test_mean_dim_backward_axis1_keepdim() {
        // x: [2, 3], mean(dim=1, keepdim=true) -> [2, 1]
        // grad: each element gets 1/3 (since dim_size=3)
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
        let m = mean_dim(&x, 1, true).unwrap();
        assert_eq!(m.shape(), &[2, 1]);
        let loss = sum(&m).unwrap();
        loss.backward().unwrap();

        let g = x.grad().unwrap().unwrap();
        assert_eq!(g.shape(), &[2, 3]);
        let expected = 1.0 / 3.0;
        for &v in g.data().unwrap() {
            assert!((v - expected).abs() < 1e-12, "expected {expected}, got {v}");
        }
    }

    #[test]
    fn test_mean_dim_has_grad_fn() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let m = mean_dim(&x, 0, false).unwrap();
        assert!(m.grad_fn().is_some());
        assert_eq!(m.grad_fn().unwrap().name(), "MeanDimBackward");
    }

    // --- median / nanmedian with dim (#1306) ---

    #[test]
    fn test_median_with_dim_odd_lower_median() {
        // Per Sorting.cpp:584, median selects sorted rank (size-1)/2.
        // Slice [3, 1, 2] sorted = [1(idx1), 2(idx2), 3(idx0)]; rank 1 -> value 2
        // at original index 2.
        let x = leaf(&[3.0, 1.0, 2.0], &[3], false);
        let (vals, inds) = median_with_dim(&x, 0, false).unwrap();
        assert_eq!(vals.data().unwrap(), &[2.0]);
        assert_eq!(inds.data().unwrap(), &[2]);
    }

    #[test]
    fn test_median_with_dim_even_takes_lower() {
        // size=4 even -> rank (4-1)/2 = 1 (the lower of the two middles).
        // [4, 2, 1, 3] sorted = [1(2), 2(1), 3(3), 4(0)]; rank 1 -> value 2 at idx 1.
        let x = leaf(&[4.0, 2.0, 1.0, 3.0], &[4], false);
        let (vals, inds) = median_with_dim(&x, 0, false).unwrap();
        assert_eq!(vals.data().unwrap(), &[2.0]);
        assert_eq!(inds.data().unwrap(), &[1]);
    }

    #[test]
    fn test_median_with_dim_2d_axis1() {
        // rows: [5,3,4] -> median 4 @ idx2 ; [1,9,2] -> median 2 @ idx2
        let x = leaf(&[5.0, 3.0, 4.0, 1.0, 9.0, 2.0], &[2, 3], false);
        let (vals, inds) = median_with_dim(&x, 1, false).unwrap();
        assert_eq!(vals.shape(), &[2]);
        assert_eq!(vals.data().unwrap(), &[4.0, 2.0]);
        assert_eq!(inds.data().unwrap(), &[2, 2]);
    }

    #[test]
    fn test_median_nan_poisons_slice() {
        // torch.median: any NaN in the slice -> result is the NaN at its index
        // (Sorting.cpp:562-569).
        let x = leaf(&[1.0, f64::NAN, 3.0], &[3], false);
        let (vals, inds) = median_with_dim(&x, 0, false).unwrap();
        assert!(vals.data().unwrap()[0].is_nan());
        assert_eq!(inds.data().unwrap(), &[1]);
    }

    #[test]
    fn test_nanmedian_skips_nan() {
        // torch.nanmedian: NaNs excluded. [1, NaN, 3, 2] -> non-NaN {1,3,2},
        // effective=3, rank (3-1)/2=1; sorted non-NaN [1(0),2(3),3(2)] -> value 2
        // at original idx 3.
        let x = leaf(&[1.0, f64::NAN, 3.0, 2.0], &[4], false);
        let (vals, inds) = nanmedian_with_dim(&x, 0, false).unwrap();
        assert_eq!(vals.data().unwrap(), &[2.0]);
        assert_eq!(inds.data().unwrap(), &[3]);
    }

    #[test]
    fn test_median_with_dim_keepdim_shape() {
        let x = leaf(&[5.0, 3.0, 4.0, 1.0, 9.0, 2.0], &[2, 3], false);
        let (vals, inds) = median_with_dim(&x, 1, true).unwrap();
        assert_eq!(vals.shape(), &[2, 1]);
        assert_eq!(inds.shape(), &[2, 1]);
    }

    #[test]
    fn test_median_backward_scatters_to_selected_index() {
        // Backward routes grad only to the median position per
        // derivatives.yaml:1179-1181 (value_selecting_reduction_backward).
        // Slice [3,1,2]: median value 2 at idx 2 -> grad lands only at idx 2.
        let x = leaf(&[3.0, 1.0, 2.0], &[3], true);
        let (vals, _inds) = median_with_dim(&x, 0, false).unwrap();
        assert_eq!(vals.grad_fn().unwrap().name(), "MedianDimBackward");
        vals.backward().unwrap();
        let g = x.grad().unwrap().unwrap();
        assert_eq!(g.data().unwrap(), &[0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_median_backward_2d_finite_difference() {
        // Finite-difference check: median is piecewise-linear (selects one
        // element), so the analytic grad must match a one-hot at the selected
        // index. rows [5,3,4]->idx2, [1,9,2]->idx2.
        let x = leaf(&[5.0, 3.0, 4.0, 1.0, 9.0, 2.0], &[2, 3], true);
        let (vals, _inds) = median_with_dim(&x, 1, false).unwrap();
        let loss = sum(&vals).unwrap();
        loss.backward().unwrap();
        let g = x.grad().unwrap().unwrap();
        // one-hot at idx 2 of each row.
        assert_eq!(g.data().unwrap(), &[0.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_nanmedian_backward_scatters_to_nonnan_median() {
        let x = leaf(&[1.0, f64::NAN, 3.0, 2.0], &[4], true);
        let (vals, _inds) = nanmedian_with_dim(&x, 0, false).unwrap();
        assert_eq!(vals.grad_fn().unwrap().name(), "NanmedianDimBackward");
        vals.backward().unwrap();
        let g = x.grad().unwrap().unwrap();
        // median value 2 is at original idx 3.
        assert_eq!(g.data().unwrap(), &[0.0, 0.0, 0.0, 1.0]);
    }
}
