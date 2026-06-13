//! Cumulative (scan) tensor operations along a specified dimension.
//!
//! Provides `cumsum`, `cumprod`, `cummax`, `cummin`, and `logcumsumexp` as
//! pure forward-pass kernels. The differentiable wrappers that attach autograd
//! nodes live in `grad_fns::cumulative`.
//!
//! [CL-306]
//!
//! ## REQ status (per `.design/ferrotorch-core/ops/cumulative.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`cumsum_forward`) | SHIPPED | mirrors `cumsum_cpu_kernel` (registered as `cumsum_stub`, dispatched by `TORCH_IMPL_FUNC(cumsum_out)`); non-test consumer is the `cumsum_forward` invocation inside `grad_fns::cumulative::cumsum`; indirect parity `[cumsum] 32/32 passed`. |
//! | REQ-2 (`cumprod_forward`) | SHIPPED | mirrors `cumprod_cpu_kernel` (registered as `cumprod_stub`); non-test consumer is the `cumprod_forward` invocation inside `grad_fns::cumulative::cumprod`; indirect parity `[cumprod] 80/80 passed`. |
//! | REQ-3 (`cummax_forward`) | SHIPPED | mirrors `cummax_helper_cpu` dispatching `cummax_cummin_helper<..., std::greater_equal>`; `>=` tie-break + NaN-poison predicate match upstream; non-test consumer is the `cummax_forward` invocation inside `grad_fns::cumulative::cummax` (reached by the `EinopsReduction::Max` arm in `einops.rs`); indirect parity `[cummax] 24/24 passed`; closes #1231. |
//! | REQ-4 (`cummin_forward`) | SHIPPED | mirrors `cummin_helper_cpu` dispatching `cummax_cummin_helper<..., std::less_equal>`; `<=` tie-break + same NaN-poison; non-test consumer is the `cummin_forward` invocation inside `grad_fns::cumulative::cummin` (reached by `EinopsReduction::Min` in `einops.rs`); indirect parity `[cummin] 24/24 passed`; closes #1231. |
//! | REQ-5 (`logcumsumexp_forward`) | SHIPPED | mirrors `logcumsumexp_cpu_kernel` (uses `_log_add_exp_helper` for stability); non-test consumer is the `logcumsumexp_forward` invocation inside `grad_fns::cumulative::logcumsumexp`; indirect parity `[logcumsumexp] 48/48 passed`. |
//! | REQ-6 (`reverse_cumsum` helper) | SHIPPED | mirrors upstream's `w.flip(dim).cumsum(dim).flip(dim)` pattern unrolled into a single reverse triple-loop; non-test consumers are `CumsumBackward::backward` and `LogcumsumexpBackward::backward` in `grad_fns/cumulative.rs`. |
//! | REQ-7 (`validate_dim`) | SHIPPED | wraps `crate::shape::normalize_axis` mirroring `maybe_wrap_dim` (with deliberate 0-D rejection as defense-in-depth — the autograd-layer fast paths short-circuit 0-D before reaching the kernel); consumed by all five `*_forward` entry points in this file. |
//! | REQ-8 (`CumExtremeResult` struct) | SHIPPED | mirrors upstream's `std::tuple<Tensor, Tensor>` return; `Vec<usize>` indices (R-DEV-7 deviation since ferrotorch lacks an i64 tensor); re-exported as `pub use ops::cumulative::CumExtremeResult` at the crate root; non-test consumers `grad_fns::cumulative::cummax` / `cummin` / `cumextreme_scalar_identity`. |

use std::any::TypeId;

use crate::Device;
use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::int_tensor::IntTensor;
use crate::shape::normalize_axis;
use crate::storage::TensorStorage;
use crate::tensor::Tensor;

#[inline]
fn is_f32<T: Float>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f32>()
}

#[inline]
fn is_f64<T: Float>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f64>()
}

// ---------------------------------------------------------------------------
// Stride helpers
// ---------------------------------------------------------------------------

/// Compute (outer_size, dim_size, inner_size) for iterating along `dim`.
///
/// For a shape `[d0, d1, ..., dn]` and a given `dim`, this factorises the
/// flat index space into:
///   - `outer_size` = product of dims before `dim`
///   - `dim_size`   = shape[dim]
///   - `inner_size` = product of dims after `dim`
///
/// The flat index of element `(outer, i, inner)` is:
///   `outer * dim_size * inner_size + i * inner_size + inner`
fn dim_strides(shape: &[usize], dim: usize) -> (usize, usize, usize) {
    let outer: usize = shape[..dim].iter().product();
    let dim_size = shape[dim];
    let inner: usize = shape[dim + 1..].iter().product();
    (outer, dim_size, inner)
}

/// Normalise and validate `dim` for cumulative ops.
fn validate_dim(ndim: usize, dim: i64, op_name: &str) -> FerrotorchResult<usize> {
    if ndim == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("{op_name}: cannot operate on a scalar (0-D) tensor"),
        });
    }
    normalize_axis(dim as isize, ndim)
}

fn read_gpu_i64_indices(
    handle: crate::gpu_dispatch::GpuBufferHandle,
    shape: &[usize],
) -> FerrotorchResult<Vec<usize>> {
    let idxs_tensor = IntTensor::<i64>::from_gpu_handle(handle, shape.to_vec());
    let idxs_cpu = idxs_tensor.to(Device::Cpu)?;
    idxs_cpu
        .data()?
        .iter()
        .map(|&v| {
            usize::try_from(v).map_err(|_| FerrotorchError::InvalidArgument {
                message: format!("cummax/cummin: CUDA kernel returned invalid negative index {v}"),
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// cumsum
// ---------------------------------------------------------------------------

/// Cumulative sum along `dim`.
///
/// Output shape is identical to input shape. Element `[..., i, ...]` along
/// `dim` equals `sum(input[..., 0..=i, ...])`.
pub fn cumsum_forward<T: Float>(input: &Tensor<T>, dim: i64) -> FerrotorchResult<Tensor<T>> {
    let norm_dim = validate_dim(input.ndim(), dim, "cumsum")?;
    let shape = input.shape();
    let (outer, dim_size, inner) = dim_strides(shape, norm_dim);

    // GPU fast path for f32/f64
    if input.is_cuda()
        && (is_f32::<T>() || is_f64::<T>())
        && let Some(backend) = crate::gpu_dispatch::gpu_backend()
    {
        // #1658: normalise a narrowed-offset CUDA view to a packed offset-0
        // buffer before the strided cumulative kernel reads element 0.
        let input = input.contiguous()?;
        let handle = if is_f32::<T>() {
            backend.cumsum_f32(input.gpu_handle()?, outer, dim_size, inner)?
        } else {
            backend.cumsum_f64(input.gpu_handle()?, outer, dim_size, inner)?
        };
        return Tensor::from_storage(TensorStorage::gpu(handle), shape.to_vec(), false);
    }

    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "cumsum" });
    }

    let in_data = input.data_vec()?;

    let mut out = vec![<T as num_traits::Zero>::zero(); in_data.len()];

    for o in 0..outer {
        for k in 0..inner {
            let base = o * dim_size * inner + k;
            let mut acc = <T as num_traits::Zero>::zero();
            for i in 0..dim_size {
                let idx = base + i * inner;
                acc += in_data[idx];
                out[idx] = acc;
            }
        }
    }

    Tensor::from_storage(TensorStorage::cpu(out), shape.to_vec(), false)
}

/// Reverse cumulative sum along `dim` (used for cumsum backward).
///
/// `reverse_cumsum[..., i, ...] = sum(input[..., i..dim_size, ...])`
pub fn reverse_cumsum<T: Float>(data: &[T], shape: &[usize], dim: usize) -> Vec<T> {
    let (outer, dim_size, inner) = dim_strides(shape, dim);
    let mut out = vec![<T as num_traits::Zero>::zero(); data.len()];

    for o in 0..outer {
        for k in 0..inner {
            let base = o * dim_size * inner + k;
            let mut acc = <T as num_traits::Zero>::zero();
            for i in (0..dim_size).rev() {
                let idx = base + i * inner;
                acc += data[idx];
                out[idx] = acc;
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// cumprod
// ---------------------------------------------------------------------------

/// Cumulative product along `dim`.
///
/// Output shape is identical to input shape. Element `[..., i, ...]` along
/// `dim` equals `prod(input[..., 0..=i, ...])`.
pub fn cumprod_forward<T: Float>(input: &Tensor<T>, dim: i64) -> FerrotorchResult<Tensor<T>> {
    let norm_dim = validate_dim(input.ndim(), dim, "cumprod")?;
    let shape = input.shape();
    let (outer, dim_size, inner) = dim_strides(shape, norm_dim);

    // GPU fast path for f32/f64
    if input.is_cuda()
        && (is_f32::<T>() || is_f64::<T>())
        && let Some(backend) = crate::gpu_dispatch::gpu_backend()
    {
        // #1658: normalise a narrowed-offset CUDA view to a packed offset-0
        // buffer before the strided cumulative kernel reads element 0.
        let input = input.contiguous()?;
        let handle = if is_f32::<T>() {
            backend.cumprod_f32(input.gpu_handle()?, outer, dim_size, inner)?
        } else {
            backend.cumprod_f64(input.gpu_handle()?, outer, dim_size, inner)?
        };
        return Tensor::from_storage(TensorStorage::gpu(handle), shape.to_vec(), false);
    }

    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "cumprod" });
    }

    let in_data = input.data_vec()?;

    let mut out = vec![<T as num_traits::Zero>::zero(); in_data.len()];

    for o in 0..outer {
        for k in 0..inner {
            let base = o * dim_size * inner + k;
            let mut acc = <T as num_traits::One>::one();
            for i in 0..dim_size {
                let idx = base + i * inner;
                acc = acc * in_data[idx];
                out[idx] = acc;
            }
        }
    }

    Tensor::from_storage(TensorStorage::cpu(out), shape.to_vec(), false)
}

// ---------------------------------------------------------------------------
// cummax
// ---------------------------------------------------------------------------

/// Result of `cummax` / `cummin`: values tensor and indices tensor.
#[derive(Debug)]
pub struct CumExtremeResult<T: Float> {
    pub values: Tensor<T>,
    pub indices: Vec<usize>,
}

/// Cumulative maximum along `dim`.
///
/// Returns `(values, indices)` where `values[..., i, ...]` is the running
/// maximum of `input[..., 0..=i, ...]` and `indices` holds the flat-along-dim
/// index at which each running maximum was attained.
///
/// Tie-breaking matches upstream `std::greater_equal<scalar_t>` at
/// `aten/src/ATen/native/ReduceOps.cpp:832` — on equal values, the LATER
/// index wins. NaN propagation also matches upstream `cummax_cummin_helper`
/// at `:811-826`: once a NaN appears in the prefix, every subsequent
/// `values[..., j, ...]` is NaN and the running `indices[..., j, ...]`
/// pin to the first-NaN position. The update predicate is
/// `isnan(curr) || (!isnan(out) && curr >= out)` (mirrors `:819`).
pub fn cummax_forward<T: Float>(
    input: &Tensor<T>,
    dim: i64,
) -> FerrotorchResult<CumExtremeResult<T>> {
    let norm_dim = validate_dim(input.ndim(), dim, "cummax")?;
    let shape = input.shape();
    let (outer, dim_size, inner) = dim_strides(shape, norm_dim);

    // GPU fast path for f32/f64 — kernel returns both values and indices
    if input.is_cuda()
        && (is_f32::<T>() || is_f64::<T>())
        && let Some(backend) = crate::gpu_dispatch::gpu_backend()
    {
        // #1658: normalise a narrowed-offset CUDA view to a packed offset-0
        // buffer before the strided cumulative kernel reads element 0.
        let input = input.contiguous()?;
        let values;
        let indices: Vec<usize>;
        if is_f32::<T>() {
            let (vals_h, idxs_h) =
                backend.cummax_f32(input.gpu_handle()?, outer, dim_size, inner)?;
            values = Tensor::from_storage(TensorStorage::gpu(vals_h), shape.to_vec(), false)?;
            indices = read_gpu_i64_indices(idxs_h, shape)?;
        } else {
            let (vals_h, idxs_h) =
                backend.cummax_f64(input.gpu_handle()?, outer, dim_size, inner)?;
            values = Tensor::from_storage(TensorStorage::gpu(vals_h), shape.to_vec(), false)?;
            indices = read_gpu_i64_indices(idxs_h, shape)?;
        }
        return Ok(CumExtremeResult { values, indices });
    }

    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "cummax" });
    }

    let in_data = input.data_vec()?;
    let numel = in_data.len();
    let mut out_vals = vec![<T as num_traits::Zero>::zero(); numel];
    let mut out_idxs = vec![0usize; numel];

    // Seed and update rule mirror upstream `cummax_cummin_helper` at
    // `aten/src/ATen/native/ReduceOps.cpp:811-826`:
    //   T1 out = c10::load(self_data);          // seed = first element
    //   int idx = 0;
    //   for i in irange(dim_size):
    //     curr = self_data[i*stride];
    //     if (isnan_(curr) || (!isnan_(out) && op(curr, out))) {  // op = std::greater_equal
    //       out = curr; idx = i;
    //     }
    //     values_data[i*stride] = out;
    //     indices_data[i*stride] = idx;
    if dim_size > 0 {
        for o in 0..outer {
            for k in 0..inner {
                let base = o * dim_size * inner + k;
                // Seed with the first element along this scan line (i == 0)
                // exactly as upstream's `T1 out = c10::load(self_data)`.
                let mut cur = in_data[base];
                let mut cur_idx = 0usize;
                for i in 0..dim_size {
                    let idx = base + i * inner;
                    let curr = in_data[idx];
                    // Tie-break: `>=` so later index wins on ties (mirrors
                    // `std::greater_equal<scalar_t>` at `ReduceOps.cpp:832`).
                    // NaN: once `cur` is NaN it poisons every subsequent
                    // position because `!isnan(cur)` short-circuits both
                    // branches of the OR; a fresh `curr == NaN` still wins
                    // via `isnan_(curr_elem)`.
                    let curr_is_nan = <T as num_traits::Float>::is_nan(curr);
                    let cur_is_nan = <T as num_traits::Float>::is_nan(cur);
                    if curr_is_nan || (!cur_is_nan && curr >= cur) {
                        cur = curr;
                        cur_idx = i;
                    }
                    out_vals[idx] = cur;
                    out_idxs[idx] = cur_idx;
                }
            }
        }
    }

    let values = Tensor::from_storage(TensorStorage::cpu(out_vals), shape.to_vec(), false)?;
    Ok(CumExtremeResult {
        values,
        indices: out_idxs,
    })
}

// ---------------------------------------------------------------------------
// cummin
// ---------------------------------------------------------------------------

/// Cumulative minimum along `dim`.
///
/// Returns `(values, indices)` analogous to [`cummax_forward`] but tracking
/// the running minimum.
///
/// Tie-breaking matches upstream `std::less_equal<scalar_t>` at
/// `aten/src/ATen/native/ReduceOps.cpp:871` — on equal values, the LATER
/// index wins. NaN propagation matches the same templated helper at
/// `:811-826`: any NaN in the prefix poisons all subsequent positions.
pub fn cummin_forward<T: Float>(
    input: &Tensor<T>,
    dim: i64,
) -> FerrotorchResult<CumExtremeResult<T>> {
    let norm_dim = validate_dim(input.ndim(), dim, "cummin")?;
    let shape = input.shape();
    let (outer, dim_size, inner) = dim_strides(shape, norm_dim);

    // GPU fast path for f32/f64 — kernel returns both values and indices
    if input.is_cuda()
        && (is_f32::<T>() || is_f64::<T>())
        && let Some(backend) = crate::gpu_dispatch::gpu_backend()
    {
        // #1658: normalise a narrowed-offset CUDA view to a packed offset-0
        // buffer before the strided cumulative kernel reads element 0.
        let input = input.contiguous()?;
        let values;
        let indices: Vec<usize>;
        if is_f32::<T>() {
            let (vals_h, idxs_h) =
                backend.cummin_f32(input.gpu_handle()?, outer, dim_size, inner)?;
            values = Tensor::from_storage(TensorStorage::gpu(vals_h), shape.to_vec(), false)?;
            indices = read_gpu_i64_indices(idxs_h, shape)?;
        } else {
            let (vals_h, idxs_h) =
                backend.cummin_f64(input.gpu_handle()?, outer, dim_size, inner)?;
            values = Tensor::from_storage(TensorStorage::gpu(vals_h), shape.to_vec(), false)?;
            indices = read_gpu_i64_indices(idxs_h, shape)?;
        }
        return Ok(CumExtremeResult { values, indices });
    }

    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "cummin" });
    }

    let in_data = input.data_vec()?;
    let numel = in_data.len();
    let mut out_vals = vec![<T as num_traits::Zero>::zero(); numel];
    let mut out_idxs = vec![0usize; numel];

    // Mirrors upstream `cummax_cummin_helper` at
    // `aten/src/ATen/native/ReduceOps.cpp:811-826` with
    // `op = std::less_equal<scalar_t>`.
    if dim_size > 0 {
        for o in 0..outer {
            for k in 0..inner {
                let base = o * dim_size * inner + k;
                let mut cur = in_data[base];
                let mut cur_idx = 0usize;
                for i in 0..dim_size {
                    let idx = base + i * inner;
                    let curr = in_data[idx];
                    let curr_is_nan = <T as num_traits::Float>::is_nan(curr);
                    let cur_is_nan = <T as num_traits::Float>::is_nan(cur);
                    // Tie-break: `<=` so later index wins on ties (mirrors
                    // `std::less_equal<scalar_t>` at `ReduceOps.cpp:871`).
                    if curr_is_nan || (!cur_is_nan && curr <= cur) {
                        cur = curr;
                        cur_idx = i;
                    }
                    out_vals[idx] = cur;
                    out_idxs[idx] = cur_idx;
                }
            }
        }
    }

    let values = Tensor::from_storage(TensorStorage::cpu(out_vals), shape.to_vec(), false)?;
    Ok(CumExtremeResult {
        values,
        indices: out_idxs,
    })
}

// ---------------------------------------------------------------------------
// logcumsumexp
// ---------------------------------------------------------------------------

/// Pairwise log-add-exp with PyTorch's equal-infinity guards.
///
/// Mirrors `_log_add_exp_helper` at
/// `pytorch/aten/src/ATen/native/cpu/LogAddExp.h:22-33`:
///
/// ```cpp
/// scalar_t min = at::_isnan(y) ? y : std::min(x, y);
/// scalar_t max = at::_isnan(y) ? y : std::max(x, y);
/// if (min != max || std::isfinite(min)) {
///   return std::log1p(std::exp(min - max)) + max;  // nan propagates here
/// } else {
///   return x;  // special case to correctly handle infinite cases
/// }
/// ```
///
/// When both arguments are the same infinity, `min - max` would be
/// `inf - inf = NaN`, so the equal-infinity branch returns `x` directly
/// (CORE-133 / #1827). NaN in either argument propagates through the
/// `log1p(exp(min - max)) + max` arithmetic.
#[inline]
fn log_add_exp<T: Float>(x: T, y: T) -> T {
    let (min, max) = if y.is_nan() {
        (y, y)
    } else if x.is_nan() {
        // `std::min(x, y)` / `std::max(x, y)` both return `x` when `x` is NaN.
        (x, x)
    } else if x < y {
        (x, y)
    } else {
        (y, x)
    };
    if min != max || min.is_finite() {
        // NaN propagates here.
        (min - max).exp().ln_1p() + max
    } else {
        // Equal infinities pass through instead of entering `inf - inf`.
        x
    }
}

/// Log-cumulative-sum-exp along `dim`.
///
/// `output[..., i, ...] = log(sum(exp(input[..., 0..=i, ...])))`
///
/// Numerically stable: sequential [`log_add_exp`] scan with a `-inf` initial
/// accumulator, mirroring `logcumsumexp_cpu_kernel` at
/// `pytorch/aten/src/ATen/native/cpu/ReduceOpsKernel.cpp:117-136`
/// (`cum_number = _log_add_exp_helper(x, cum_number)` with
/// `init_val = -std::numeric_limits<scalar_t>::infinity()`).
pub fn logcumsumexp_forward<T: Float>(input: &Tensor<T>, dim: i64) -> FerrotorchResult<Tensor<T>> {
    let norm_dim = validate_dim(input.ndim(), dim, "logcumsumexp")?;
    let shape = input.shape();
    let (outer, dim_size, inner) = dim_strides(shape, norm_dim);

    // GPU fast path for f32/f64
    if input.is_cuda()
        && (is_f32::<T>() || is_f64::<T>())
        && let Some(backend) = crate::gpu_dispatch::gpu_backend()
    {
        // #1658: normalise a narrowed-offset CUDA view to a packed offset-0
        // buffer before the strided cumulative kernel reads element 0.
        let input = input.contiguous()?;
        let handle = if is_f32::<T>() {
            backend.logcumsumexp_f32(input.gpu_handle()?, outer, dim_size, inner)?
        } else {
            backend.logcumsumexp_f64(input.gpu_handle()?, outer, dim_size, inner)?
        };
        return Tensor::from_storage(TensorStorage::gpu(handle), shape.to_vec(), false);
    }

    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "logcumsumexp" });
    }

    let in_data = input.data_vec()?;

    let mut out = vec![<T as num_traits::Zero>::zero(); in_data.len()];
    let neg_inf = <T as num_traits::Float>::neg_infinity();

    // Sequential scan: acc starts at -inf and folds each element through
    // `log_add_exp` (the `_log_add_exp_helper` port above), matching
    // `logcumsumexp_cpu_kernel`'s accumulation loop at
    // `pytorch/aten/src/ATen/native/cpu/ReduceOpsKernel.cpp:126-132`.
    for o in 0..outer {
        for k in 0..inner {
            let base = o * dim_size * inner + k;
            let mut acc = neg_inf;
            for i in 0..dim_size {
                let idx = base + i * inner;
                acc = log_add_exp(in_data[idx], acc);
                out[idx] = acc;
            }
        }
    }

    Tensor::from_storage(TensorStorage::cpu(out), shape.to_vec(), false)
}
