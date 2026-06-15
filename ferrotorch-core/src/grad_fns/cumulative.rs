//! Backward functions and differentiable wrappers for cumulative (scan) ops.
//!
//! - `cumsum`       — backward is reverse cumsum of the gradient
//! - `cumprod`      — backward uses the forward output and prefix/suffix products
//! - `cummax`       — backward routes through saved indices tensor via `scatter_add`
//! - `cummin`       — backward routes through saved indices tensor via `scatter_add`
//! - `logcumsumexp` — backward via softmax-weighted reverse cumsum
//!
//! [CL-306]
//!
//! ## REQ status (per `.design/ferrotorch-core/grad_fns/cumulative.md`)
//!
//! Full evidence rows (impl + non-test production consumer + parity smoke
//! counts + upstream `file:line` cites) live in the design doc; this
//! synopsis is a one-line summary per REQ.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (cumsum) | SHIPPED | `cumsum` at `cumulative.rs:105` + `CumsumBackward` at `:53` (0-D fast path mirrors `ReduceOps.cpp:501-504`); consumer `Tensor::cumsum_t` in `methods.rs`; parity `[cumsum] 32/32` (grep=1) |
//! | REQ-2 (cumprod) | SHIPPED | `cumprod` at `cumulative.rs:400` + `CumprodBackward` at `:283` (O(n^3) zeros-path); consumer `Tensor::cumprod_t` in `methods.rs`; parity `[cumprod] 80/80` (grep=1) |
//! | REQ-3 (cummax) | SHIPPED | `cummax` at `cumulative.rs:671` + `CummaxBackward` at `:460` via private helper `cummaxmin_backward_impl` (scatter_add VJP per `derivatives.yaml:533-535`); consumer `einops.rs:796` (`EinopsReduction::Max`); parity `[cummax] 24/24` (grep=1) |
//! | REQ-4 (cummin) | SHIPPED | `cummin` at `cumulative.rs:712` + `CumminBackward` at `:501` shares the private `cummaxmin_backward_impl` helper; consumer `einops.rs:802` (`EinopsReduction::Min`); parity `[cummin] 24/24` (grep=1) |
//! | REQ-5 (logcumsumexp) | SHIPPED | `logcumsumexp` + `LogcumsumexpBackward` use PyTorch's signed log-space VJP from `FunctionsManual.cpp::logcumsumexp_backward`, with resident CUDA kernels for f32/f64/f16/bf16 and CPU logical-view handling; consumer `Tensor::logcumsumexp_t`; parity `[logcumsumexp] 48/48` (grep=1) |
//! | REQ-6 (dim normalization) | SHIPPED | `normalize_axis(dim as isize, ndim)` calls at `cumulative.rs:109, :404, :675, :716, :1004` mirroring `maybe_wrap_dim` at `ReduceOps.cpp:506, :622, :851, :890`; consumers are the five `pub fn` bodies themselves (reached through `einops.rs:796 / :802` and the `methods.rs` `*_t` surfaces) |
//! | REQ-7 (reverse_cumsum helper) | SHIPPED | impl `ops/cumulative.rs:163` mirroring `static Tensor reversed_cumsum(...)` at `ReduceOps.cpp:527-529`; non-test consumer `CumsumBackward::backward` at `cumulative.rs:77` |

use std::sync::Arc;

use crate::autograd::no_grad::{is_grad_enabled, no_grad};
use crate::dtype::DType;
use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::int_tensor::IntTensor;
use crate::ops::cumulative::{
    CumExtremeResult, cummax_forward, cummin_forward, cumprod_forward, cumsum_forward,
    logcumsumexp_forward, reverse_cumsum,
};
use crate::shape::normalize_axis;
use crate::storage::TensorStorage;
use crate::tensor::{GradFn, Tensor};

// ---------------------------------------------------------------------------
// CumsumBackward
// ---------------------------------------------------------------------------

/// Backward node for `cumsum(input, dim)`.
///
/// VJP: `grad_input = reverse_cumsum(grad_output, dim)`.
///
/// Intuition: cumsum is a lower-triangular matrix multiply along dim.
/// Its transpose is the upper-triangular (reverse cumsum).
#[derive(Debug)]
pub struct CumsumBackward<T: Float> {
    input: Tensor<T>,
    dim: usize,
}

impl<T: Float> GradFn<T> for CumsumBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // 0-D fast path: cumsum on a scalar is the identity, so the VJP is
        // also identity (`grad_input = grad_output`). Mirrors PyTorch's
        // `impl_func_cum_ops` at `aten/src/ATen/native/ReduceOps.cpp:501-504`
        // where the 0-D branch `result.fill_(self)` bypasses the cumsum
        // stub entirely. `reverse_cumsum` is not 0-D-safe (its
        // `dim_strides` helper indexes `shape[dim]`), so we must short
        // circuit before reaching the kernel.
        if self.input.ndim() == 0 {
            return Ok(vec![Some(grad_output.clone())]);
        }
        if grad_output.is_cuda() {
            let grad_input = reverse_cumsum_cuda(grad_output, self.dim)?;
            return Ok(vec![Some(grad_input)]);
        }
        let go_data = grad_output.data()?;
        let shape = grad_output.shape();

        let grad_data = reverse_cumsum(go_data, shape, self.dim);

        let grad_input =
            Tensor::from_storage(TensorStorage::cpu(grad_data), shape.to_vec(), false)?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "CumsumBackward"
    }
}

/// Differentiable cumulative sum along `dim`.
///
/// When gradient tracking is enabled, the returned tensor carries a
/// [`CumsumBackward`] node.
///
/// `dim` supports negative indexing.
///
/// 0-D (scalar) inputs return the input copied through, matching PyTorch's
/// `impl_func_cum_ops` 0-D branch at
/// `aten/src/ATen/native/ReduceOps.cpp:501-504`. The only valid `dim`
/// values for a scalar are `0` and `-1`; anything else mirrors upstream's
/// `IndexError: Dimension out of range`.
pub fn cumsum<T: Float>(input: &Tensor<T>, dim: i64) -> FerrotorchResult<Tensor<T>> {
    if input.ndim() == 0 {
        return cumulative_scalar_identity(input, dim, "cumsum", ScalarBackwardKind::Cumsum);
    }
    let norm_dim = normalize_axis(dim as isize, input.ndim())?;
    let result = cumsum_forward(input, dim)?;

    if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(CumsumBackward {
            input: input.clone(),
            dim: norm_dim,
        });
        let (storage, shape) = result.into_storage_and_shape()?;
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// 0-D scalar identity helper (shared by cumsum / cumprod / logcumsumexp)
// ---------------------------------------------------------------------------

/// Discriminant for [`cumulative_scalar_identity`] — picks which
/// `*Backward` node to attach on the 0-D fast path.
#[derive(Clone, Copy)]
enum ScalarBackwardKind {
    Cumsum,
    Cumprod,
    Logcumsumexp,
}

/// 0-D (scalar) fast path shared by `cumsum`, `cumprod`, and `logcumsumexp`.
///
/// PyTorch's `impl_func_cum_ops` (`aten/src/ATen/native/ReduceOps.cpp:501-504`)
/// special-cases `self.dim() == 0` to `result.fill_(self)` — the scalar is
/// copied through unchanged. Live-verified 2026-05-25 with torch 2.11.0:
///
/// ```text
/// torch.cumsum(torch.tensor(5.0), 0)       == tensor(5.)
/// torch.cumprod(torch.tensor(-3.5), 0)     == tensor(-3.5000)
/// torch.logcumsumexp(torch.tensor(5.0), 0) == tensor(5.)
/// torch.cumsum(torch.tensor(5.0), -1)      == tensor(5.)
/// torch.cumsum(torch.tensor(5.0), 1)       -> IndexError: Dimension out of range
/// ```
///
/// All three ops are the identity on a scalar, so their VJP is also the
/// identity (`grad_input = grad_output`). The returned tensor carries the
/// appropriate `*Backward` node when `input.requires_grad()` is true so
/// the autograd graph stays connected; the saved `dim` is `0` because the
/// only normalized scalar dim is `0`.
///
/// Device contract (CORE-042 / #1736, R-LOUD-2): the output lands on the
/// INPUT's device. CUDA scalars are materialised with the existing on-device
/// strided-copy path under `no_grad`, so offset scalar views are gathered
/// correctly into fresh storage without a host readback. The identity backward
/// passes `grad_output` through unchanged, which is device-consistent with the
/// device-preserving forward. Live torch 2.11.0+cu130 succeeds on 0-D CUDA
/// tensors for all three ops, returning 0-D outputs on `cuda:0`.
fn cumulative_scalar_identity<T: Float>(
    input: &Tensor<T>,
    dim: i64,
    op_name: &str,
    kind: ScalarBackwardKind,
) -> FerrotorchResult<Tensor<T>> {
    // PyTorch accepts dim ∈ {-1, 0} on a 0-D tensor; any other value
    // raises `IndexError`. cumsum/cumprod/logcumsumexp do NOT route
    // through `zero_numel_check_dims`; they hit the `result.fill_(self)`
    // branch in `impl_func_cum_ops` (aten/src/ATen/native/ReduceOps.cpp:501-504)
    // and then call `c10::maybe_wrap_dim` in the non-0-D path. The error
    // emitted on a bad dim is therefore from `maybe_wrap_dim_slow` at
    // `c10/core/WrapDimMinimal.cpp:23-31`, which formats:
    //   "Dimension out of range (expected to be in range of [<min>, <max>], but got <dim>)"
    // For a 0-D tensor `dim_post_expr == 0` recurses with
    // `dim_post_expr=1`, giving min=-1 and max=0.
    if dim != 0 && dim != -1 {
        let _ = op_name; // upstream's wrap_dim message does not include the op name.
        return Err(crate::error::FerrotorchError::InvalidArgument {
            message: format!(
                "Dimension out of range (expected to be in range of [-1, 0], but got {dim})"
            ),
        });
    }

    // Materialize a fresh scalar storage with the input's logical element so
    // the returned tensor has a distinct identity from `input` (autograd graph
    // invariant: Tensor::from_operation needs a new TensorId). CUDA uses the
    // on-device strided gather, which is critical for 0-D views with a
    // non-zero storage offset.
    let result = scalar_identity_value(input)?;

    if !(is_grad_enabled() && input.requires_grad()) {
        return Ok(result);
    }

    // Attach the op-specific *Backward. Each backward implements the
    // identity VJP on 0-D (handled by its own ndim()==0 fast path).
    let (storage, shape) = result.into_storage_and_shape()?;
    match kind {
        ScalarBackwardKind::Cumsum => {
            let grad_fn = Arc::new(CumsumBackward {
                input: input.clone(),
                dim: 0,
            });
            Tensor::from_operation(storage, shape, grad_fn)
        }
        ScalarBackwardKind::Cumprod => {
            // CumprodBackward saves `output` too; on 0-D the output
            // equals the input. Materialize a second device-resident scalar;
            // the 0-D backward never reads it, but keeping it on the same
            // device matches the non-scalar saved-output contract.
            let saved_output = result_for_saved_output(input)?;
            let grad_fn = Arc::new(CumprodBackward {
                input: input.clone(),
                output: saved_output,
                dim: 0,
            });
            Tensor::from_operation(storage, shape, grad_fn)
        }
        ScalarBackwardKind::Logcumsumexp => {
            let saved_output = result_for_saved_output(input)?;
            let grad_fn = Arc::new(LogcumsumexpBackward {
                input: input.clone(),
                output: saved_output,
                dim: 0,
            });
            Tensor::from_operation(storage, shape, grad_fn)
        }
    }
}

fn scalar_identity_value<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if input.is_cuda() {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let handle = input.gpu_handle()?;
        let shape: &[usize] = &[];
        let strides: &[isize] = &[];
        let offset = input.storage_offset();
        let out_handle = match T::dtype() {
            DType::F32 => backend.strided_copy_f32(handle, shape, strides, offset)?,
            DType::F64 => backend.strided_copy_f64(handle, shape, strides, offset)?,
            DType::F16 | DType::BF16 => backend.strided_copy_u16(handle, shape, strides, offset)?,
            _ => {
                return Err(FerrotorchError::NotImplementedOnCuda {
                    op: "cumulative scalar identity",
                });
            }
        };
        return Tensor::from_storage(TensorStorage::gpu(out_handle), Vec::new(), false);
    }
    let scalar_val = input.data_vec()?[0];
    Tensor::from_storage(
        TensorStorage::on_device(vec![scalar_val], input.device())?,
        Vec::new(),
        false,
    )
}

fn result_for_saved_output<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    scalar_identity_value(input)
}

// ---------------------------------------------------------------------------
// CumprodBackward
// ---------------------------------------------------------------------------

/// Backward node for `cumprod(input, dim)`.
///
/// VJP for cumprod is:
///   `grad_input[i] = sum_{j >= i} grad_output[j] * cumprod_output[j] / input[i]`
///
/// To handle zeros safely, we use the identity:
///   `grad_input[i] = (1/input[i]) * reverse_cumsum(grad_output * cumprod_output, dim)[i]`
///
/// When `input[i] == 0`, we recompute using prefix/suffix products along
/// the scan direction to avoid division by zero.
#[derive(Debug)]
pub struct CumprodBackward<T: Float> {
    input: Tensor<T>,
    output: Tensor<T>,
    dim: usize,
}

impl<T: Float> GradFn<T> for CumprodBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // 0-D fast path: cumprod on a scalar is the identity (output ==
        // input), so the VJP is also the identity. Mirrors PyTorch's
        // `impl_func_cum_ops` 0-D branch at
        // `aten/src/ATen/native/ReduceOps.cpp:501-504`. The triple-loop
        // below uses `dim_strides` which would index `shape[0]` on an
        // empty shape — we must short-circuit here.
        if self.input.ndim() == 0 {
            return Ok(vec![Some(grad_output.clone())]);
        }
        if grad_output.is_cuda() || self.input.is_cuda() || self.output.is_cuda() {
            if !(grad_output.is_cuda() && self.input.is_cuda() && self.output.is_cuda()) {
                return Err(FerrotorchError::DeviceMismatch {
                    expected: self.input.device(),
                    got: grad_output.device(),
                });
            }
            let grad_input = cumprod_backward_cuda(&self.input, grad_output, self.dim)?;
            return Ok(vec![Some(grad_input)]);
        }

        let go_data = grad_output.data()?;
        let in_data = self.input.data()?;
        let out_data = self.output.data()?;
        let shape = self.input.shape();

        let (outer, dim_size, inner) = dim_strides(shape, self.dim);
        let numel = in_data.len();
        let mut grad_input = vec![<T as num_traits::Zero>::zero(); numel];

        for o in 0..outer {
            for k in 0..inner {
                let base = o * dim_size * inner + k;

                // Check if any element along this scan line is zero.
                let has_zero = (0..dim_size)
                    .any(|i| in_data[base + i * inner] == <T as num_traits::Zero>::zero());

                if has_zero {
                    // Slow path: zeros present. Use prefix/suffix product
                    // approach to avoid division by zero.
                    //
                    // For each position i:
                    //   grad_input[i] = sum_{j >= i} go[j] * prod_{k in [i..=j], k != i} input[k]
                    //                 = sum_{j >= i} go[j] * (cumprod[j] / cumprod[i-1]) / input[i]
                    //
                    // But with zeros this is fragile, so we just brute-force
                    // the partial products for correctness.
                    for i in 0..dim_size {
                        let mut acc = <T as num_traits::Zero>::zero();
                        for j in i..dim_size {
                            // Compute product of input[k] for k in [0..=j], excluding k=i.
                            let mut partial = <T as num_traits::One>::one();
                            for kk in 0..=j {
                                if kk != i {
                                    #[allow(clippy::assign_op_pattern)]
                                    {
                                        partial = partial * in_data[base + kk * inner];
                                    }
                                }
                            }
                            acc += go_data[base + j * inner] * partial;
                        }
                        grad_input[base + i * inner] = acc;
                    }
                } else {
                    // Fast path: no zeros, safe to use output / input.
                    // grad_input[i] = reverse_cumsum(grad_output * output)[i] / input[i]
                    //
                    // We compute `product = go * out` then reverse-cumsum it,
                    // then divide each element by input[i].
                    let mut product = vec![<T as num_traits::Zero>::zero(); dim_size];
                    for (i, prod_elem) in product.iter_mut().enumerate().take(dim_size) {
                        let idx = base + i * inner;
                        *prod_elem = go_data[idx] * out_data[idx];
                    }
                    // Reverse cumsum of product.
                    let mut rev_acc = <T as num_traits::Zero>::zero();
                    for i in (0..dim_size).rev() {
                        let idx = base + i * inner;
                        rev_acc += product[i];
                        grad_input[idx] = rev_acc / in_data[idx];
                    }
                }
            }
        }

        let result = Tensor::from_storage(TensorStorage::cpu(grad_input), shape.to_vec(), false)?;
        Ok(vec![Some(result)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "CumprodBackward"
    }
}

/// Differentiable cumulative product along `dim`.
///
/// When gradient tracking is enabled, the returned tensor carries a
/// [`CumprodBackward`] node.
///
/// `dim` supports negative indexing.
///
/// 0-D (scalar) inputs return the input copied through (identity), matching
/// PyTorch's `impl_func_cum_ops` at
/// `aten/src/ATen/native/ReduceOps.cpp:501-504`.
pub fn cumprod<T: Float>(input: &Tensor<T>, dim: i64) -> FerrotorchResult<Tensor<T>> {
    if input.ndim() == 0 {
        return cumulative_scalar_identity(input, dim, "cumprod", ScalarBackwardKind::Cumprod);
    }
    let norm_dim = normalize_axis(dim as isize, input.ndim())?;
    let result = cumprod_forward(input, dim)?;

    if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(CumprodBackward {
            input: input.clone(),
            output: result.clone(),
            dim: norm_dim,
        });
        let (storage, shape) = result.into_storage_and_shape()?;
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// CummaxBackward / CumminBackward
// ---------------------------------------------------------------------------

/// Backward node for `cummax(input, dim)` (and symmetrically `cummin`).
///
/// VJP: `grad_input = zeros_like(input).scatter_add_(dim, indices, grad_output)`.
///
/// Mirrors upstream `cummaxmin_backward` at
/// `aten/src/ATen/native/ReduceOps.cpp:906-918`:
///
/// ```text
/// Tensor cummaxmin_backward(const Tensor& grad, const Tensor& input,
///                            const Tensor& indices, int64_t dim) {
///   if (input.sym_numel() == 0) { return input; }
///   auto result = at::zeros_symint(input.sym_sizes(), input.options());
///   return result.scatter_add_(dim, indices, grad);
/// }
/// ```
///
/// Per `tools/autograd/derivatives.yaml:533-535`:
///
/// ```text
/// - name: cummax(Tensor self, int dim) -> (Tensor values, Tensor indices)
///   self: cummaxmin_backward(grad, self, indices, dim)
/// ```
///
/// The saved `indices_tensor` carries the position-along-dim at which each
/// running max was attained; CUDA keeps it resident and CPU keeps a host cache
/// for the reference scatter path. `scatter_add` accumulates grad_output at
/// those positions so each input position receives gradient proportional to
/// the number of output positions whose running max it "won".
///
/// Tie-break correctness: when two positions in the prefix carry equal
/// values, upstream's `std::greater_equal` picks the LATER index. The
/// `cummax_forward` kernel at `ops/cumulative.rs` mirrors this. Without the
/// matching tie-break, `scatter_add` would deposit gradient at the wrong
/// input position on ties — that's why the kernel fix and the autograd
/// backward MUST land together (this commit, closing #1231).
#[derive(Debug)]
pub struct CummaxBackward<T: Float> {
    input: Tensor<T>,
    indices: Vec<usize>,
    indices_tensor: IntTensor<i64>,
    input_shape: Vec<usize>,
    dim: usize,
}

impl<T: Float> GradFn<T> for CummaxBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // 0-D fast path: cummax on a scalar is the identity in both values
        // and indices, so the VJP is also identity. Mirrors PyTorch's
        // `impl_func_cum_ops` 0-D branch at `ReduceOps.cpp:501-504`.
        if self.input.ndim() == 0 {
            return Ok(vec![Some(grad_output.clone())]);
        }
        cummaxmin_backward_impl(
            grad_output,
            &self.input_shape,
            &self.indices,
            &self.indices_tensor,
            self.dim,
        )
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "CummaxBackward"
    }
}

/// Backward node for `cummin(input, dim)` — symmetric to [`CummaxBackward`].
///
/// Upstream uses the SAME `cummaxmin_backward` C++ function for both ops
/// (`ReduceOps.cpp:906-918`); the only difference is which `indices`
/// tensor the forward saved. We mirror this by sharing
/// `cummaxmin_backward_impl` and only differing in the grad-fn name.
#[derive(Debug)]
pub struct CumminBackward<T: Float> {
    input: Tensor<T>,
    indices: Vec<usize>,
    indices_tensor: IntTensor<i64>,
    input_shape: Vec<usize>,
    dim: usize,
}

impl<T: Float> GradFn<T> for CumminBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if self.input.ndim() == 0 {
            return Ok(vec![Some(grad_output.clone())]);
        }
        cummaxmin_backward_impl(
            grad_output,
            &self.input_shape,
            &self.indices,
            &self.indices_tensor,
            self.dim,
        )
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "CumminBackward"
    }
}

/// Shared backward kernel for cummax / cummin.
///
/// Computes `grad_input = zeros_like(input).scatter_add_(dim, indices, grad_output)`
/// per upstream `cummaxmin_backward` at
/// `aten/src/ATen/native/ReduceOps.cpp:906-918`. We materialise it as a
/// fresh tensor (out-of-place — ferrotorch lacks in-place scatter_add as a
/// production primitive yet) which matches the
/// `areAnyTensorSubclassLike(...)` composite-compliance branch at `:914-916`.
fn cummaxmin_backward_impl<T: Float>(
    grad_output: &Tensor<T>,
    input_shape: &[usize],
    indices: &[usize],
    indices_tensor: &IntTensor<i64>,
    dim: usize,
) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
    if grad_output.is_cuda() {
        return cummaxmin_backward_cuda(grad_output, input_shape, indices_tensor, dim);
    }
    // Empty-input fast path mirrors upstream
    // `ReduceOps.cpp:907-908 if (input.sym_numel() == 0) { return input; }`
    let numel: usize = crate::shape::numel(input_shape);
    if numel == 0 {
        let empty = Tensor::from_storage(
            TensorStorage::cpu(Vec::<T>::new()),
            input_shape.to_vec(),
            false,
        )?;
        return Ok(vec![Some(empty)]);
    }
    if indices.len() != numel {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "CummaxBackward/CumminBackward: CPU backward requires {numel} host saved indices, got {}",
                indices.len()
            ),
        });
    }
    let zeros = crate::creation::zeros::<T>(input_shape)?;
    let grad_input =
        crate::ops::indexing::scatter_add(&zeros, dim as isize, indices, input_shape, grad_output)?;
    Ok(vec![Some(grad_input)])
}

fn cummaxmin_backward_cuda<T: Float>(
    grad_output: &Tensor<T>,
    input_shape: &[usize],
    indices_tensor: &IntTensor<i64>,
    dim: usize,
) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
    if !indices_tensor.is_cuda() {
        return Err(FerrotorchError::DeviceMismatch {
            expected: grad_output.device(),
            got: indices_tensor.device(),
        });
    }
    if indices_tensor.device() != grad_output.device() {
        return Err(FerrotorchError::DeviceMismatch {
            expected: grad_output.device(),
            got: indices_tensor.device(),
        });
    }
    if indices_tensor.shape() != input_shape {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "CummaxBackward/CumminBackward: indices shape {:?} != input shape {:?}",
                indices_tensor.shape(),
                input_shape
            ),
        });
    }

    let grad_output = grad_output.contiguous()?;
    let grad_handle = grad_output.gpu_handle()?;
    let ordinal = grad_handle.device_ordinal();
    let backend = crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
    let numel: usize = crate::shape::numel(input_shape);
    let zeros_h = backend.alloc_zeros(numel, T::dtype(), ordinal)?;
    let out_h = match T::dtype() {
        DType::F32 => backend.scatter_add_nd_f32(
            &zeros_h,
            indices_tensor.gpu_handle()?,
            grad_handle,
            input_shape,
            input_shape,
            dim,
        )?,
        DType::F64 => backend.scatter_add_nd_f64(
            &zeros_h,
            indices_tensor.gpu_handle()?,
            grad_handle,
            input_shape,
            input_shape,
            dim,
        )?,
        DType::F16 => backend.scatter_add_nd_f16(
            &zeros_h,
            indices_tensor.gpu_handle()?,
            grad_handle,
            input_shape,
            input_shape,
            dim,
        )?,
        DType::BF16 => backend.scatter_add_nd_bf16(
            &zeros_h,
            indices_tensor.gpu_handle()?,
            grad_handle,
            input_shape,
            input_shape,
            dim,
        )?,
        _ => {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "CummaxBackward",
            });
        }
    };
    let grad_input = Tensor::from_storage(TensorStorage::gpu(out_h), input_shape.to_vec(), false)?;
    Ok(vec![Some(grad_input)])
}

/// Cumulative maximum along `dim`.
///
/// Returns `(values, indices)` where `values` has the running maximum and
/// `indices` has the position (along `dim`) where each maximum was attained.
///
/// When gradient tracking is enabled, the returned `values` tensor carries
/// a [`CummaxBackward`] node implementing the
/// `scatter_add(grad, dim, indices)` VJP per
/// `tools/autograd/derivatives.yaml:533-535
/// - name: cummax(Tensor self, int dim) -> (Tensor values, Tensor indices)
///   self: cummaxmin_backward(grad, self, indices, dim)`.
///
/// `indices_tensor` is the authoritative PyTorch-style int64 result. The
/// legacy `indices` Vec is populated for CPU/scalar results only; non-scalar
/// CUDA results keep indices device-resident and leave that host cache empty.
///
/// 0-D (scalar) inputs return `CumExtremeResult { values: scalar copy,
/// indices: vec![0] }`, matching PyTorch's `impl_func_cum_ops` 0-D branch
/// at `aten/src/ATen/native/ReduceOps.cpp:501-504`.
pub fn cummax<T: Float>(input: &Tensor<T>, dim: i64) -> FerrotorchResult<CumExtremeResult<T>> {
    if input.ndim() == 0 {
        return cumextreme_scalar_identity(input, dim, "cummax", ScalarExtremeKind::Cummax);
    }
    let norm_dim = normalize_axis(dim as isize, input.ndim())?;
    let result = cummax_forward(input, dim)?;

    if !(is_grad_enabled() && input.requires_grad()) {
        return Ok(result);
    }

    let CumExtremeResult {
        values,
        indices_tensor,
        indices,
    } = result;
    let input_shape = values.shape().to_vec();
    let grad_fn = Arc::new(CummaxBackward {
        input: input.clone(),
        indices: indices.clone(),
        indices_tensor: indices_tensor.clone(),
        input_shape: input_shape.clone(),
        dim: norm_dim,
    });
    let (storage, shape) = values.into_storage_and_shape()?;
    let values = Tensor::from_operation(storage, shape, grad_fn)?;
    Ok(CumExtremeResult {
        values,
        indices_tensor,
        indices,
    })
}

/// Cumulative minimum along `dim`.
///
/// Returns `(values, indices)` analogous to [`cummax`] but tracking the
/// running minimum. Differentiable through the same scatter-add VJP per
/// `tools/autograd/derivatives.yaml:537-539`.
///
/// 0-D (scalar) inputs return `CumExtremeResult { values: scalar copy,
/// indices: vec![0] }`.
pub fn cummin<T: Float>(input: &Tensor<T>, dim: i64) -> FerrotorchResult<CumExtremeResult<T>> {
    if input.ndim() == 0 {
        return cumextreme_scalar_identity(input, dim, "cummin", ScalarExtremeKind::Cummin);
    }
    let norm_dim = normalize_axis(dim as isize, input.ndim())?;
    let result = cummin_forward(input, dim)?;

    if !(is_grad_enabled() && input.requires_grad()) {
        return Ok(result);
    }

    let CumExtremeResult {
        values,
        indices_tensor,
        indices,
    } = result;
    let input_shape = values.shape().to_vec();
    let grad_fn = Arc::new(CumminBackward {
        input: input.clone(),
        indices: indices.clone(),
        indices_tensor: indices_tensor.clone(),
        input_shape: input_shape.clone(),
        dim: norm_dim,
    });
    let (storage, shape) = values.into_storage_and_shape()?;
    let values = Tensor::from_operation(storage, shape, grad_fn)?;
    Ok(CumExtremeResult {
        values,
        indices_tensor,
        indices,
    })
}

/// Discriminant for [`cumextreme_scalar_identity`] — picks which
/// `*Backward` node to attach on the 0-D fast path (mirrors
/// [`ScalarBackwardKind`] for the single-output identity helper).
#[derive(Clone, Copy)]
enum ScalarExtremeKind {
    Cummax,
    Cummin,
}

/// 0-D scalar fast path for `cummax` / `cummin`.
///
/// PyTorch's `impl_func_cum_ops` 0-D branch (`ReduceOps.cpp:501-504`)
/// copies the scalar through. For the (values, indices) tuple ops the
/// indices tensor is the scalar `0` (the only valid position on a
/// 0-element-axis scalar). Live-verified 2026-05-25 with torch 2.11.0:
///
/// ```text
/// torch.cummax(torch.tensor(5.0), 0)  -> (tensor(5.), tensor(0))
/// torch.cummin(torch.tensor(-3.5), 0) -> (tensor(-3.5), tensor(0))
/// torch.cummax(torch.tensor(5.0), 1)  -> IndexError: cummax(): Expected
///                                       reduction dim -1 or 0 for scalar
///                                       but got 1
/// ```
///
/// Device contract (CORE-042 / #1736, R-LOUD-2): `values` lands on the
/// INPUT's device. CUDA scalar values are gathered with on-device strided
/// copy, and CUDA scalar indices are allocated directly as an i64
/// zero buffer. Live torch 2.11.0+cu130 returns `(tensor(5.,
/// device='cuda:0'), tensor(0, device='cuda:0'))` for a 0-D CUDA input;
/// ferrotorch's `indices` carrier stays the documented `Vec<usize>` deviation
/// (no device dimension).
///
/// Autograd (CORE-042 / #1736 companion): when gradient tracking is
/// enabled, `values` carries the op's `*Backward` node (whose existing
/// 0-D fast path is the identity VJP) so the graph stays connected —
/// pre-fix this helper returned a detached tensor and the gradient never
/// reached the leaf, while live torch 2.11.0 differentiates through 0-D
/// `cummax`/`cummin` (`x.grad == tensor(2.5)` for `vals.backward(
/// torch.tensor(2.5))`).
fn cumextreme_scalar_identity<T: Float>(
    input: &Tensor<T>,
    dim: i64,
    op_name: &str,
    kind: ScalarExtremeKind,
) -> FerrotorchResult<CumExtremeResult<T>> {
    // PyTorch's `cummax_helper`/`cummin_helper` route 0-D through
    // `zero_numel_check_dims` at
    // `aten/src/ATen/native/ReduceOpsUtils.h:277-281`, which formats:
    //   "<fn_name>: Expected reduction dim -1 or 0 for scalar but got <dim>"
    // The `fn_name` arg is `"cummax()"` / `"cummin()"` per
    // `aten/src/ATen/native/ReduceOps.cpp:840,879`. Capital 'E' on
    // "Expected".
    if dim != 0 && dim != -1 {
        return Err(crate::error::FerrotorchError::InvalidArgument {
            message: format!(
                "{op_name}(): Expected reduction dim -1 or 0 for scalar but got {dim}"
            ),
        });
    }
    // CORE-042 (#1736): device-transparent scalar identity. CUDA offset scalar
    // views are gathered directly on device and the i64 index scalar is
    // allocated on device.
    let values = scalar_identity_value(input)?;
    let indices_tensor = scalar_zero_i64(input.device())?;

    if !(is_grad_enabled() && input.requires_grad()) {
        return Ok(CumExtremeResult {
            values,
            indices_tensor,
            indices: vec![0],
        });
    }

    // CORE-042 (#1736) companion: attach the op's backward node so the
    // 0-D fast path stays differentiable (identity VJP via the node's own
    // `ndim() == 0` short-circuit). The saved `indices`/`input_shape`
    // mirror the 0-D forward result (`indices == [0]`, empty shape); the
    // backward never reads them on the 0-D path.
    let (storage, shape) = values.into_storage_and_shape()?;
    let values = match kind {
        ScalarExtremeKind::Cummax => {
            let grad_fn = Arc::new(CummaxBackward {
                input: input.clone(),
                indices: vec![0],
                indices_tensor: indices_tensor.clone(),
                input_shape: Vec::new(),
                dim: 0,
            });
            Tensor::from_operation(storage, shape, grad_fn)?
        }
        ScalarExtremeKind::Cummin => {
            let grad_fn = Arc::new(CumminBackward {
                input: input.clone(),
                indices: vec![0],
                indices_tensor: indices_tensor.clone(),
                input_shape: Vec::new(),
                dim: 0,
            });
            Tensor::from_operation(storage, shape, grad_fn)?
        }
    };
    Ok(CumExtremeResult {
        values,
        indices_tensor,
        indices: vec![0],
    })
}

fn scalar_zero_i64(device: crate::device::Device) -> FerrotorchResult<IntTensor<i64>> {
    if let crate::device::Device::Cuda(ordinal) = device {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let handle = backend.alloc_zeros(1, DType::I64, ordinal)?;
        return Ok(IntTensor::<i64>::from_gpu_handle(handle, Vec::new()));
    }
    IntTensor::<i64>::from_vec(vec![0], Vec::new())?.to(device)
}

// ---------------------------------------------------------------------------
// LogcumsumexpBackward
// ---------------------------------------------------------------------------

/// Backward node for `logcumsumexp(input, dim)`.
///
/// VJP: `grad_input[i] = sum_{j >= i} grad_output[j] * softmax_weight(i, j)`
/// where `softmax_weight(i, j) = exp(input[i] - logcumsumexp_output[j])`.
///
/// PyTorch does not implement this as the visually simple
/// `exp(input) * reverse_cumsum(grad * exp(-output))` for real tensors. That
/// expression is algebraically correct but overflows/underflows and propagates
/// NaN upstream gradients differently. `FunctionsManual.cpp:
/// logcumsumexp_backward` splits positive and negative upstream gradients in
/// log-space:
///
/// `exp(input + reverse_logcumsumexp(log(abs(grad_pos)) - output))
///  - exp(input + reverse_logcumsumexp(log(abs(grad_neg)) - output))`.
#[derive(Debug)]
pub struct LogcumsumexpBackward<T: Float> {
    input: Tensor<T>,
    output: Tensor<T>,
    dim: usize,
}

impl<T: Float> GradFn<T> for LogcumsumexpBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // 0-D fast path: logcumsumexp on a scalar is `log(exp(x)) == x`
        // (the identity). VJP is identity. Mirrors PyTorch's 0-D
        // short-circuit at `aten/src/ATen/native/ReduceOps.cpp:501-504`.
        // The non-0-D body below calls `reverse_cumsum` which would
        // index `shape[0]` on an empty shape — we must short-circuit.
        if self.input.ndim() == 0 {
            return Ok(vec![Some(grad_output.clone())]);
        }
        if grad_output.is_cuda() || self.input.is_cuda() || self.output.is_cuda() {
            if !(grad_output.is_cuda() && self.input.is_cuda() && self.output.is_cuda()) {
                return Err(FerrotorchError::DeviceMismatch {
                    expected: self.input.device(),
                    got: grad_output.device(),
                });
            }
            let grad_input = no_grad(|| {
                logcumsumexp_backward_cuda(&self.input, &self.output, grad_output, self.dim)
            })?;
            return Ok(vec![Some(grad_input)]);
        }

        let go_data = grad_output.data_vec()?;
        let in_data = self.input.data_vec()?;
        let out_data = self.output.data_vec()?;
        let shape = self.input.shape();
        let grad_data = logcumsumexp_backward_cpu(&in_data, &out_data, &go_data, shape, self.dim);

        let grad_input =
            Tensor::from_storage(TensorStorage::cpu(grad_data), shape.to_vec(), false)?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "LogcumsumexpBackward"
    }
}

#[inline]
fn log_add_exp_stable<T: Float>(x: T, y: T) -> T {
    let (min, max) = if y.is_nan() {
        (y, y)
    } else if x.is_nan() {
        (x, x)
    } else if x < y {
        (x, y)
    } else {
        (y, x)
    };
    if min != max || min.is_finite() {
        (min - max).exp().ln_1p() + max
    } else {
        x
    }
}

fn logcumsumexp_backward_cpu<T: Float>(
    input: &[T],
    output: &[T],
    grad_output: &[T],
    shape: &[usize],
    dim: usize,
) -> Vec<T> {
    let (outer, dim_size, inner) = dim_strides(shape, dim);
    let mut grad_input = vec![<T as num_traits::Zero>::zero(); input.len()];
    let zero = <T as num_traits::Zero>::zero();
    let neg_inf = <T as num_traits::Float>::neg_infinity();

    for o in 0..outer {
        for i in 0..inner {
            let base = o * dim_size * inner + i;
            let mut acc_pos = neg_inf;
            let mut acc_neg = neg_inf;
            for k in (0..dim_size).rev() {
                let idx = base + k * inner;
                let g = grad_output[idx];
                if g > zero {
                    acc_pos = log_add_exp_stable(acc_pos, g.abs().ln() - output[idx]);
                } else if g < zero {
                    acc_neg = log_add_exp_stable(acc_neg, g.abs().ln() - output[idx]);
                }
                grad_input[idx] = (input[idx] + acc_pos).exp() - (input[idx] + acc_neg).exp();
            }
        }
    }

    grad_input
}

/// Differentiable log-cumulative-sum-exp along `dim`.
///
/// `output[..., i, ...] = log(sum(exp(input[..., 0..=i, ...])))`
///
/// When gradient tracking is enabled, the returned tensor carries a
/// [`LogcumsumexpBackward`] node.
///
/// `dim` supports negative indexing.
///
/// 0-D (scalar) inputs return the input copied through (identity), matching
/// PyTorch's `impl_func_cum_ops` at
/// `aten/src/ATen/native/ReduceOps.cpp:501-504`. The numerical identity is
/// `logcumsumexp(x) = log(exp(x)) = x` for a single-element scan.
pub fn logcumsumexp<T: Float>(input: &Tensor<T>, dim: i64) -> FerrotorchResult<Tensor<T>> {
    if input.ndim() == 0 {
        return cumulative_scalar_identity(
            input,
            dim,
            "logcumsumexp",
            ScalarBackwardKind::Logcumsumexp,
        );
    }
    let norm_dim = normalize_axis(dim as isize, input.ndim())?;
    let result = logcumsumexp_forward(input, dim)?;

    if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(LogcumsumexpBackward {
            input: input.clone(),
            output: result.clone(),
            dim: norm_dim,
        });
        let (storage, shape) = result.into_storage_and_shape()?;
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Helpers (re-used from ops::cumulative internally)
// ---------------------------------------------------------------------------

/// Compute (outer_size, dim_size, inner_size) — mirrors the one in `ops::cumulative`.
fn dim_strides(shape: &[usize], dim: usize) -> (usize, usize, usize) {
    let outer: usize = crate::shape::numel(&shape[..dim]);
    let dim_size = shape[dim];
    let inner: usize = crate::shape::numel(&shape[dim + 1..]);
    (outer, dim_size, inner)
}

fn reverse_cumsum_cuda<T: Float>(input: &Tensor<T>, dim: usize) -> FerrotorchResult<Tensor<T>> {
    let shape = input.shape().to_vec();
    let (outer, dim_size, inner) = dim_strides(&shape, dim);
    let input = input.contiguous()?;
    let handle = input.gpu_handle()?;
    let backend = crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
    let out_h = match T::dtype() {
        DType::F32 => backend.reverse_cumsum_f32(handle, outer, dim_size, inner)?,
        DType::F64 => backend.reverse_cumsum_f64(handle, outer, dim_size, inner)?,
        DType::F16 => backend.reverse_cumsum_f16(handle, outer, dim_size, inner)?,
        DType::BF16 => backend.reverse_cumsum_bf16(handle, outer, dim_size, inner)?,
        _ => {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "reverse_cumsum",
            });
        }
    };
    Tensor::from_storage(TensorStorage::gpu(out_h), shape, false)
}

fn logcumsumexp_backward_cuda<T: Float>(
    input: &Tensor<T>,
    output: &Tensor<T>,
    grad_output: &Tensor<T>,
    dim: usize,
) -> FerrotorchResult<Tensor<T>> {
    if input.device() != output.device() {
        return Err(FerrotorchError::DeviceMismatch {
            expected: input.device(),
            got: output.device(),
        });
    }
    if input.device() != grad_output.device() {
        return Err(FerrotorchError::DeviceMismatch {
            expected: input.device(),
            got: grad_output.device(),
        });
    }
    if input.shape() != output.shape() || input.shape() != grad_output.shape() {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "LogcumsumexpBackward: input shape {:?}, output shape {:?}, grad_output shape {:?}",
                input.shape(),
                output.shape(),
                grad_output.shape()
            ),
        });
    }

    let shape = input.shape().to_vec();
    let (outer, dim_size, inner) = dim_strides(&shape, dim);
    let input = input.contiguous()?;
    let output = output.contiguous()?;
    let grad_output = grad_output.contiguous()?;
    let backend = crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
    let out_h = match T::dtype() {
        DType::F32 => backend.logcumsumexp_backward_f32(
            input.gpu_handle()?,
            output.gpu_handle()?,
            grad_output.gpu_handle()?,
            outer,
            dim_size,
            inner,
        )?,
        DType::F64 => backend.logcumsumexp_backward_f64(
            input.gpu_handle()?,
            output.gpu_handle()?,
            grad_output.gpu_handle()?,
            outer,
            dim_size,
            inner,
        )?,
        DType::F16 => backend.logcumsumexp_backward_f16(
            input.gpu_handle()?,
            output.gpu_handle()?,
            grad_output.gpu_handle()?,
            outer,
            dim_size,
            inner,
        )?,
        DType::BF16 => backend.logcumsumexp_backward_bf16(
            input.gpu_handle()?,
            output.gpu_handle()?,
            grad_output.gpu_handle()?,
            outer,
            dim_size,
            inner,
        )?,
        _ => {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "logcumsumexp_backward",
            });
        }
    };
    Tensor::from_storage(TensorStorage::gpu(out_h), shape, false)
}

fn cumprod_backward_cuda<T: Float>(
    input: &Tensor<T>,
    grad_output: &Tensor<T>,
    dim: usize,
) -> FerrotorchResult<Tensor<T>> {
    if input.device() != grad_output.device() {
        return Err(FerrotorchError::DeviceMismatch {
            expected: input.device(),
            got: grad_output.device(),
        });
    }
    if input.shape() != grad_output.shape() {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "CumprodBackward: grad_output shape {:?} != input shape {:?}",
                grad_output.shape(),
                input.shape()
            ),
        });
    }
    let shape = input.shape().to_vec();
    let (outer, dim_size, inner) = dim_strides(&shape, dim);
    let input = input.contiguous()?;
    let grad_output = grad_output.contiguous()?;
    let backend = crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
    let out_h = match T::dtype() {
        DType::F32 => backend.cumprod_backward_f32(
            input.gpu_handle()?,
            grad_output.gpu_handle()?,
            outer,
            dim_size,
            inner,
        )?,
        DType::F64 => backend.cumprod_backward_f64(
            input.gpu_handle()?,
            grad_output.gpu_handle()?,
            outer,
            dim_size,
            inner,
        )?,
        DType::F16 => backend.cumprod_backward_f16(
            input.gpu_handle()?,
            grad_output.gpu_handle()?,
            outer,
            dim_size,
            inner,
        )?,
        DType::BF16 => backend.cumprod_backward_bf16(
            input.gpu_handle()?,
            grad_output.gpu_handle()?,
            outer,
            dim_size,
            inner,
        )?,
        _ => {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "cumprod_backward",
            });
        }
    };
    Tensor::from_storage(TensorStorage::gpu(out_h), shape, false)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::no_grad::no_grad;
    use crate::grad_fns::reduction::sum;
    use crate::storage::TensorStorage;

    /// Helper: create a leaf tensor.
    fn leaf(data: &[f64], shape: &[usize], requires_grad: bool) -> Tensor<f64> {
        Tensor::from_storage(
            TensorStorage::cpu(data.to_vec()),
            shape.to_vec(),
            requires_grad,
        )
        .unwrap()
    }

    // =======================================================================
    // cumsum forward
    // =======================================================================

    #[test]
    fn test_cumsum_1d() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[4], false);
        let cs = cumsum(&x, 0).unwrap();
        assert_eq!(cs.shape(), &[4]);
        let d = cs.data().unwrap();
        assert!((d[0] - 1.0).abs() < 1e-12);
        assert!((d[1] - 3.0).abs() < 1e-12);
        assert!((d[2] - 6.0).abs() < 1e-12);
        assert!((d[3] - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_cumsum_2d_dim0() {
        // [[1, 2, 3], [4, 5, 6]] cumsum along dim 0
        // -> [[1, 2, 3], [5, 7, 9]]
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        let cs = cumsum(&x, 0).unwrap();
        assert_eq!(cs.shape(), &[2, 3]);
        let d = cs.data().unwrap();
        assert!((d[0] - 1.0).abs() < 1e-12);
        assert!((d[1] - 2.0).abs() < 1e-12);
        assert!((d[2] - 3.0).abs() < 1e-12);
        assert!((d[3] - 5.0).abs() < 1e-12);
        assert!((d[4] - 7.0).abs() < 1e-12);
        assert!((d[5] - 9.0).abs() < 1e-12);
    }

    #[test]
    fn test_cumsum_2d_dim1() {
        // [[1, 2, 3], [4, 5, 6]] cumsum along dim 1
        // -> [[1, 3, 6], [4, 9, 15]]
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        let cs = cumsum(&x, 1).unwrap();
        let d = cs.data().unwrap();
        assert!((d[0] - 1.0).abs() < 1e-12);
        assert!((d[1] - 3.0).abs() < 1e-12);
        assert!((d[2] - 6.0).abs() < 1e-12);
        assert!((d[3] - 4.0).abs() < 1e-12);
        assert!((d[4] - 9.0).abs() < 1e-12);
        assert!((d[5] - 15.0).abs() < 1e-12);
    }

    #[test]
    fn test_cumsum_negative_dim() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        let cs = cumsum(&x, -1).unwrap();
        let d = cs.data().unwrap();
        // Same as dim=1.
        assert!((d[0] - 1.0).abs() < 1e-12);
        assert!((d[1] - 3.0).abs() < 1e-12);
        assert!((d[2] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_cumsum_3d() {
        // shape [2, 2, 3], cumsum along dim=1
        let data: Vec<f64> = (1..=12).map(|x| x as f64).collect();
        let x = leaf(&data, &[2, 2, 3], false);
        let cs = cumsum(&x, 1).unwrap();
        assert_eq!(cs.shape(), &[2, 2, 3]);
        let d = cs.data().unwrap();
        // First slice: [[1,2,3],[4,5,6]] -> [[1,2,3],[5,7,9]]
        assert!((d[0] - 1.0).abs() < 1e-12);
        assert!((d[3] - 5.0).abs() < 1e-12);
        assert!((d[4] - 7.0).abs() < 1e-12);
        assert!((d[5] - 9.0).abs() < 1e-12);
    }

    // =======================================================================
    // cumsum backward
    // =======================================================================

    #[test]
    fn test_cumsum_backward_1d() {
        // cumsum([a, b, c]) = [a, a+b, a+b+c]
        // d(sum of cumsum)/da = 3, /db = 2, /dc = 1
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let cs = cumsum(&x, 0).unwrap();
        let loss = sum(&cs).unwrap();
        loss.backward().unwrap();

        let g = x.grad().unwrap().unwrap();
        let gd = g.data().unwrap();
        assert!((gd[0] - 3.0).abs() < 1e-12, "got {}", gd[0]);
        assert!((gd[1] - 2.0).abs() < 1e-12, "got {}", gd[1]);
        assert!((gd[2] - 1.0).abs() < 1e-12, "got {}", gd[2]);
    }

    #[test]
    fn test_cumsum_backward_2d_dim0() {
        // x: [[1, 2], [3, 4]], cumsum(dim=0) -> [[1, 2], [4, 6]]
        // loss = sum = 1+2+4+6 = 13
        // d/dx[0,0] = d(1)/dx[0,0] + d(4)/dx[0,0] = 1 + 1 = 2
        // d/dx[0,1] = d(2)/dx[0,1] + d(6)/dx[0,1] = 1 + 1 = 2
        // d/dx[1,0] = d(4)/dx[1,0] = 1
        // d/dx[1,1] = d(6)/dx[1,1] = 1
        let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2], true);
        let cs = cumsum(&x, 0).unwrap();
        let loss = sum(&cs).unwrap();
        loss.backward().unwrap();

        let g = x.grad().unwrap().unwrap();
        let gd = g.data().unwrap();
        assert!((gd[0] - 2.0).abs() < 1e-12, "got {}", gd[0]);
        assert!((gd[1] - 2.0).abs() < 1e-12, "got {}", gd[1]);
        assert!((gd[2] - 1.0).abs() < 1e-12, "got {}", gd[2]);
        assert!((gd[3] - 1.0).abs() < 1e-12, "got {}", gd[3]);
    }

    #[test]
    fn test_cumsum_has_grad_fn() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let cs = cumsum(&x, 0).unwrap();
        assert!(cs.grad_fn().is_some());
        assert_eq!(cs.grad_fn().unwrap().name(), "CumsumBackward");
    }

    #[test]
    fn test_cumsum_no_grad_fn_when_not_requires_grad() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], false);
        let cs = cumsum(&x, 0).unwrap();
        assert!(cs.grad_fn().is_none());
    }

    #[test]
    fn test_cumsum_no_grad_fn_in_no_grad_context() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let cs = no_grad(|| cumsum(&x, 0)).unwrap();
        assert!(cs.grad_fn().is_none());
    }

    // =======================================================================
    // cumprod forward
    // =======================================================================

    #[test]
    fn test_cumprod_1d() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[4], false);
        let cp = cumprod(&x, 0).unwrap();
        let d = cp.data().unwrap();
        assert!((d[0] - 1.0).abs() < 1e-12);
        assert!((d[1] - 2.0).abs() < 1e-12);
        assert!((d[2] - 6.0).abs() < 1e-12);
        assert!((d[3] - 24.0).abs() < 1e-12);
    }

    #[test]
    fn test_cumprod_2d_dim0() {
        // [[1, 2, 3], [4, 5, 6]] cumprod dim 0
        // -> [[1, 2, 3], [4, 10, 18]]
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        let cp = cumprod(&x, 0).unwrap();
        let d = cp.data().unwrap();
        assert!((d[0] - 1.0).abs() < 1e-12);
        assert!((d[1] - 2.0).abs() < 1e-12);
        assert!((d[2] - 3.0).abs() < 1e-12);
        assert!((d[3] - 4.0).abs() < 1e-12);
        assert!((d[4] - 10.0).abs() < 1e-12);
        assert!((d[5] - 18.0).abs() < 1e-12);
    }

    #[test]
    fn test_cumprod_2d_dim1() {
        // [[1, 2, 3], [4, 5, 6]] cumprod dim 1
        // -> [[1, 2, 6], [4, 20, 120]]
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        let cp = cumprod(&x, 1).unwrap();
        let d = cp.data().unwrap();
        assert!((d[0] - 1.0).abs() < 1e-12);
        assert!((d[1] - 2.0).abs() < 1e-12);
        assert!((d[2] - 6.0).abs() < 1e-12);
        assert!((d[3] - 4.0).abs() < 1e-12);
        assert!((d[4] - 20.0).abs() < 1e-12);
        assert!((d[5] - 120.0).abs() < 1e-12);
    }

    // =======================================================================
    // cumprod backward
    // =======================================================================

    #[test]
    fn test_cumprod_backward_1d() {
        // cumprod([a, b, c]) = [a, ab, abc]
        // loss = sum = a + ab + abc
        // d/da = 1 + b + bc = 1 + 2 + 6 = 9
        // d/db = 0 + a + ac = 0 + 1 + 3 = 4
        // d/dc = 0 + 0 + ab = 0 + 0 + 2 = 2
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let cp = cumprod(&x, 0).unwrap();
        let loss = sum(&cp).unwrap();
        loss.backward().unwrap();

        let g = x.grad().unwrap().unwrap();
        let gd = g.data().unwrap();
        assert!((gd[0] - 9.0).abs() < 1e-10, "got {}", gd[0]);
        assert!((gd[1] - 4.0).abs() < 1e-10, "got {}", gd[1]);
        assert!((gd[2] - 2.0).abs() < 1e-10, "got {}", gd[2]);
    }

    #[test]
    fn test_cumprod_backward_with_zero() {
        // cumprod([2, 0, 3]) = [2, 0, 0]
        // loss = sum = 2 + 0 + 0 = 2
        // d/dx[0] = 1 + 0 + 0 = 1 (only first element of cumprod depends on x[0] non-zero)
        // Actually: cumprod = [x0, x0*x1, x0*x1*x2]
        // cumprod([2, 0, 3]) = [2, 0, 0]
        // d(cumprod[j])/d(input[i]) = prod_{k in 0..=j, k!=i} input[k]
        // d(loss)/dx0 = prod(empty) + prod(x1) + prod(x1,x2) = 1 + 0 + 0 = 1
        // d(loss)/dx1 = prod(x0) + prod(x0,x2) = 2 + 6 = 8
        // d(loss)/dx2 = prod(x0,x1) = 0
        let x = leaf(&[2.0, 0.0, 3.0], &[3], true);
        let cp = cumprod(&x, 0).unwrap();
        let loss = sum(&cp).unwrap();
        loss.backward().unwrap();

        let g = x.grad().unwrap().unwrap();
        let gd = g.data().unwrap();
        assert!((gd[0] - 1.0).abs() < 1e-10, "d/dx[0]: got {}", gd[0]);
        assert!((gd[1] - 8.0).abs() < 1e-10, "d/dx[1]: got {}", gd[1]);
        assert!((gd[2] - 0.0).abs() < 1e-10, "d/dx[2]: got {}", gd[2]);
    }

    #[test]
    fn test_cumprod_has_grad_fn() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let cp = cumprod(&x, 0).unwrap();
        assert!(cp.grad_fn().is_some());
        assert_eq!(cp.grad_fn().unwrap().name(), "CumprodBackward");
    }

    #[test]
    fn test_cumprod_no_grad_fn_in_no_grad_context() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let cp = no_grad(|| cumprod(&x, 0)).unwrap();
        assert!(cp.grad_fn().is_none());
    }

    // =======================================================================
    // cummax forward
    // =======================================================================

    #[test]
    fn test_cummax_1d() {
        let x = leaf(&[3.0, 1.0, 4.0, 1.0, 5.0], &[5], false);
        let r = cummax(&x, 0).unwrap();
        let d = r.values.data().unwrap();
        assert!((d[0] - 3.0).abs() < 1e-12);
        assert!((d[1] - 3.0).abs() < 1e-12);
        assert!((d[2] - 4.0).abs() < 1e-12);
        assert!((d[3] - 4.0).abs() < 1e-12);
        assert!((d[4] - 5.0).abs() < 1e-12);
        assert_eq!(r.indices, vec![0, 0, 2, 2, 4]);
    }

    #[test]
    fn test_cummax_2d_dim1() {
        // [[1, 3, 2], [5, 4, 6]] cummax along dim 1
        // -> values: [[1, 3, 3], [5, 5, 6]]
        // -> indices: [[0, 1, 1], [0, 0, 2]]
        let x = leaf(&[1.0, 3.0, 2.0, 5.0, 4.0, 6.0], &[2, 3], false);
        let r = cummax(&x, 1).unwrap();
        let d = r.values.data().unwrap();
        assert!((d[0] - 1.0).abs() < 1e-12);
        assert!((d[1] - 3.0).abs() < 1e-12);
        assert!((d[2] - 3.0).abs() < 1e-12);
        assert!((d[3] - 5.0).abs() < 1e-12);
        assert!((d[4] - 5.0).abs() < 1e-12);
        assert!((d[5] - 6.0).abs() < 1e-12);
        assert_eq!(r.indices, vec![0, 1, 1, 0, 0, 2]);
    }

    // =======================================================================
    // cummin forward
    // =======================================================================

    #[test]
    fn test_cummin_1d() {
        let x = leaf(&[3.0, 1.0, 4.0, 1.0, 5.0], &[5], false);
        let r = cummin(&x, 0).unwrap();
        let d = r.values.data().unwrap();
        assert!((d[0] - 3.0).abs() < 1e-12);
        assert!((d[1] - 1.0).abs() < 1e-12);
        assert!((d[2] - 1.0).abs() < 1e-12);
        assert!((d[3] - 1.0).abs() < 1e-12);
        assert!((d[4] - 1.0).abs() < 1e-12);
        // Upstream tie-break: later index wins on ties (`std::less_equal`
        // at `ReduceOps.cpp:871`). Ties between input[1]=1.0 and
        // input[3]=1.0 resolve at index 3. Live-verified 2026-05-25 torch
        // 2.11.0: torch.cummin(torch.tensor([3.,1.,4.,1.,5.]), 0).indices
        //   == tensor([0, 1, 1, 3, 3])
        assert_eq!(r.indices, vec![0, 1, 1, 3, 3]);
    }

    #[test]
    fn test_cummin_2d_dim0() {
        // [[5, 2], [3, 4]] cummin along dim 0
        // -> values: [[5, 2], [3, 2]]
        // -> indices: [[0, 0], [1, 0]]
        let x = leaf(&[5.0, 2.0, 3.0, 4.0], &[2, 2], false);
        let r = cummin(&x, 0).unwrap();
        let d = r.values.data().unwrap();
        assert!((d[0] - 5.0).abs() < 1e-12);
        assert!((d[1] - 2.0).abs() < 1e-12);
        assert!((d[2] - 3.0).abs() < 1e-12);
        assert!((d[3] - 2.0).abs() < 1e-12);
        assert_eq!(r.indices, vec![0, 0, 1, 0]);
    }

    // =======================================================================
    // logcumsumexp forward
    // =======================================================================

    #[test]
    fn test_logcumsumexp_1d() {
        // logcumsumexp([a, b, c]) = [a, log(exp(a)+exp(b)), log(exp(a)+exp(b)+exp(c))]
        let x = leaf(&[1.0, 2.0, 3.0], &[3], false);
        let lcs = logcumsumexp(&x, 0).unwrap();
        let d = lcs.data().unwrap();

        let expected_0 = 1.0_f64;
        let expected_1 = (1.0_f64.exp() + 2.0_f64.exp()).ln();
        let expected_2 = (1.0_f64.exp() + 2.0_f64.exp() + 3.0_f64.exp()).ln();

        assert!((d[0] - expected_0).abs() < 1e-10, "got {}", d[0]);
        assert!((d[1] - expected_1).abs() < 1e-10, "got {}", d[1]);
        assert!((d[2] - expected_2).abs() < 1e-10, "got {}", d[2]);
    }

    #[test]
    fn test_logcumsumexp_2d_dim1() {
        // [[0, 1], [2, 3]] logcumsumexp along dim 1
        let x = leaf(&[0.0, 1.0, 2.0, 3.0], &[2, 2], false);
        let lcs = logcumsumexp(&x, 1).unwrap();
        let d = lcs.data().unwrap();

        let e0 = 0.0_f64;
        let e1 = (0.0_f64.exp() + 1.0_f64.exp()).ln();
        let e2 = 2.0_f64;
        let e3 = (2.0_f64.exp() + 3.0_f64.exp()).ln();

        assert!((d[0] - e0).abs() < 1e-10, "got {}", d[0]);
        assert!((d[1] - e1).abs() < 1e-10, "got {}", d[1]);
        assert!((d[2] - e2).abs() < 1e-10, "got {}", d[2]);
        assert!((d[3] - e3).abs() < 1e-10, "got {}", d[3]);
    }

    #[test]
    fn test_logcumsumexp_numerical_stability() {
        // Large values that would overflow naive exp.
        let x = leaf(&[1000.0, 1001.0, 1002.0], &[3], false);
        let lcs = logcumsumexp(&x, 0).unwrap();
        let d = lcs.data().unwrap();

        // All results should be finite.
        for &v in d {
            assert!(v.is_finite(), "got non-finite: {v}");
        }

        // First element should be 1000.
        assert!((d[0] - 1000.0).abs() < 1e-10);

        // Second: log(exp(1000) + exp(1001)) = 1001 + log(exp(-1) + 1)
        let expected_1 = 1001.0 + ((-1.0_f64).exp() + 1.0).ln();
        assert!((d[1] - expected_1).abs() < 1e-8, "got {}", d[1]);
    }

    // =======================================================================
    // logcumsumexp backward
    // =======================================================================

    #[test]
    fn test_logcumsumexp_backward_1d() {
        // Numerical gradient check.
        let x_vals = [1.0_f64, 2.0, 3.0];
        let eps = 1e-6;

        let x = leaf(&x_vals, &[3], true);
        let lcs = logcumsumexp(&x, 0).unwrap();
        let loss = sum(&lcs).unwrap();
        loss.backward().unwrap();

        let g = x.grad().unwrap().unwrap();
        let gd = g.data().unwrap();

        // Check against finite differences.
        for idx in 0..3 {
            let mut x_plus = x_vals.to_vec();
            let mut x_minus = x_vals.to_vec();
            x_plus[idx] += eps;
            x_minus[idx] -= eps;

            let tp = leaf(&x_plus, &[3], false);
            let lp = logcumsumexp(&tp, 0).unwrap();
            let sp = sum(&lp).unwrap().item().unwrap();

            let tm = leaf(&x_minus, &[3], false);
            let lm = logcumsumexp(&tm, 0).unwrap();
            let sm = sum(&lm).unwrap().item().unwrap();

            let numerical = (sp - sm) / (2.0 * eps);
            assert!(
                (gd[idx] - numerical).abs() < 1e-4,
                "index {idx}: analytic={}, numerical={}",
                gd[idx],
                numerical,
            );
        }
    }

    #[test]
    fn test_logcumsumexp_has_grad_fn() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let lcs = logcumsumexp(&x, 0).unwrap();
        assert!(lcs.grad_fn().is_some());
        assert_eq!(lcs.grad_fn().unwrap().name(), "LogcumsumexpBackward");
    }

    #[test]
    fn test_logcumsumexp_no_grad_fn_in_no_grad_context() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let lcs = no_grad(|| logcumsumexp(&x, 0)).unwrap();
        assert!(lcs.grad_fn().is_none());
    }

    // =======================================================================
    // 0-D (scalar) passthrough — closes blocker #1233.
    //
    // PyTorch accepts 0-D inputs and copies the scalar through unchanged.
    // Mirrors `impl_func_cum_ops` at
    // `aten/src/ATen/native/ReduceOps.cpp:501-504`:
    //
    //   if (self.dim() == 0) {
    //     result.fill_(self);
    //   }
    //
    // The expected values are NAMED CONSTANTS traced to live PyTorch
    // (torch 2.11.0, verified 2026-05-25):
    //   - PT_SCALAR_5_0       = 5.0  per `torch.cumsum(tensor(5.0), 0).item()`
    //   - PT_SCALAR_NEG_3_5   = -3.5 per `torch.cumprod(tensor(-3.5), 0).item()`
    //   - PT_SCALAR_INDEX_0   = 0    per `torch.cummax(tensor(5.0), 0)[1].item()`
    // These satisfy R-CHAR-3 (named typed bits traceable to upstream
    // file:line) — not the tautological pattern that compares ferrotorch
    // to itself.
    // =======================================================================

    const PT_SCALAR_5_0: f64 = 5.0;
    const PT_SCALAR_NEG_3_5: f64 = -3.5;
    const PT_SCALAR_INDEX_0: usize = 0;

    #[test]
    fn test_cumsum_scalar_passthrough() {
        let x =
            Tensor::from_storage(TensorStorage::cpu(vec![PT_SCALAR_5_0]), vec![], false).unwrap();
        let r = cumsum(&x, 0).unwrap();
        assert_eq!(r.shape(), &[] as &[usize]);
        assert!((r.item().unwrap() - PT_SCALAR_5_0).abs() < 1e-12);

        // Negative dim wraps to 0 (matches PyTorch's maybe_wrap_dim).
        let r_neg = cumsum(&x, -1).unwrap();
        assert!((r_neg.item().unwrap() - PT_SCALAR_5_0).abs() < 1e-12);
    }

    #[test]
    fn test_cumprod_scalar_passthrough() {
        // Use a negative value so we are not accidentally testing 1.0
        // (the multiplicative identity), which would mask a buggy
        // multiplier.
        let x = Tensor::from_storage(TensorStorage::cpu(vec![PT_SCALAR_NEG_3_5]), vec![], false)
            .unwrap();
        let r = cumprod(&x, 0).unwrap();
        assert_eq!(r.shape(), &[] as &[usize]);
        assert!((r.item().unwrap() - PT_SCALAR_NEG_3_5).abs() < 1e-12);

        let r_neg = cumprod(&x, -1).unwrap();
        assert!((r_neg.item().unwrap() - PT_SCALAR_NEG_3_5).abs() < 1e-12);
    }

    #[test]
    fn test_cummax_scalar_passthrough() {
        let x =
            Tensor::from_storage(TensorStorage::cpu(vec![PT_SCALAR_5_0]), vec![], false).unwrap();
        let r = cummax(&x, 0).unwrap();
        assert_eq!(r.values.shape(), &[] as &[usize]);
        assert!((r.values.item().unwrap() - PT_SCALAR_5_0).abs() < 1e-12);
        assert_eq!(r.indices, vec![PT_SCALAR_INDEX_0]);
    }

    #[test]
    fn test_cummin_scalar_passthrough() {
        let x = Tensor::from_storage(TensorStorage::cpu(vec![PT_SCALAR_NEG_3_5]), vec![], false)
            .unwrap();
        let r = cummin(&x, 0).unwrap();
        assert!((r.values.item().unwrap() - PT_SCALAR_NEG_3_5).abs() < 1e-12);
        assert_eq!(r.indices, vec![PT_SCALAR_INDEX_0]);
    }

    #[test]
    fn test_logcumsumexp_scalar_passthrough() {
        // logcumsumexp(scalar) = log(exp(scalar)) = scalar.
        let x =
            Tensor::from_storage(TensorStorage::cpu(vec![PT_SCALAR_5_0]), vec![], false).unwrap();
        let r = logcumsumexp(&x, 0).unwrap();
        assert!((r.item().unwrap() - PT_SCALAR_5_0).abs() < 1e-12);
    }

    // dim-out-of-range cases for 0-D inputs still error, mirroring
    // upstream's `IndexError`. cummax/cummin's error message uses the
    // exact upstream phrasing "Expected reduction dim -1 or 0 for scalar".
    #[test]
    fn test_cumsum_scalar_dim_out_of_range() {
        let x = Tensor::from_storage(TensorStorage::cpu(vec![1.0_f64]), vec![], false).unwrap();
        assert!(cumsum(&x, 1).is_err());
        assert!(cumsum(&x, -2).is_err());
    }

    #[test]
    fn test_cummax_scalar_dim_out_of_range() {
        let x = Tensor::from_storage(TensorStorage::cpu(vec![1.0_f64]), vec![], false).unwrap();
        assert!(cummax(&x, 1).is_err());
        assert!(cummin(&x, -2).is_err());
    }

    // Differentiable 0-D passthrough: the VJP is the identity, so a
    // gradient of 1.0 flowing back from `loss = sum(scalar_cumsum_output)`
    // arrives at the input as 1.0 unchanged.
    #[test]
    fn test_cumsum_scalar_backward_is_identity() {
        let x = leaf(&[PT_SCALAR_5_0], &[], true);
        let cs = cumsum(&x, 0).unwrap();
        let loss = sum(&cs).unwrap();
        loss.backward().unwrap();
        let g = x.grad().unwrap().unwrap();
        assert!((g.item().unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_cumprod_scalar_backward_is_identity() {
        let x = leaf(&[PT_SCALAR_NEG_3_5], &[], true);
        let cp = cumprod(&x, 0).unwrap();
        let loss = sum(&cp).unwrap();
        loss.backward().unwrap();
        let g = x.grad().unwrap().unwrap();
        assert!((g.item().unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_logcumsumexp_scalar_backward_is_identity() {
        let x = leaf(&[PT_SCALAR_5_0], &[], true);
        let lcs = logcumsumexp(&x, 0).unwrap();
        let loss = sum(&lcs).unwrap();
        loss.backward().unwrap();
        let g = x.grad().unwrap().unwrap();
        assert!((g.item().unwrap() - 1.0).abs() < 1e-12);
    }

    /// CORE-042 (#1736) companion: the 0-D fast path of `cummax` must stay
    /// differentiable — pre-fix `cumextreme_scalar_identity` returned a
    /// fresh detached tensor and the gradient never reached the leaf
    /// (`CummaxBackward` already carried the 0-D identity fast path but was
    /// never attached). Live torch 2.11.0 (verified 2026-06-11):
    /// ```text
    /// x = torch.tensor(5.0, requires_grad=True)
    /// vals, idx = torch.cummax(x, 0)
    /// vals.backward(torch.tensor(2.5))
    /// x.grad == tensor(2.5000)
    /// ```
    #[test]
    fn test_cummax_scalar_backward_is_identity() {
        let x = leaf(&[PT_SCALAR_5_0], &[], true);
        let r = cummax(&x, 0).unwrap();
        assert_eq!(r.values.grad_fn().unwrap().name(), "CummaxBackward");
        let loss = sum(&r.values).unwrap();
        loss.backward().unwrap();
        let g = x
            .grad()
            .unwrap()
            .expect("grad must reach the leaf through the 0-D cummax fast path");
        assert!((g.item().unwrap() - 1.0).abs() < 1e-12);
    }

    /// Symmetric `cummin` companion (same upstream identity VJP).
    #[test]
    fn test_cummin_scalar_backward_is_identity() {
        let x = leaf(&[PT_SCALAR_NEG_3_5], &[], true);
        let r = cummin(&x, 0).unwrap();
        assert_eq!(r.values.grad_fn().unwrap().name(), "CumminBackward");
        let loss = sum(&r.values).unwrap();
        loss.backward().unwrap();
        let g = x
            .grad()
            .unwrap()
            .expect("grad must reach the leaf through the 0-D cummin fast path");
        assert!((g.item().unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_cumsum_dim_out_of_bounds() {
        let x = leaf(&[1.0, 2.0], &[2], false);
        assert!(cumsum(&x, 1).is_err());
        assert!(cumsum(&x, -2).is_err());
    }

    // =======================================================================
    // cumprod backward numerical gradient check
    // =======================================================================

    #[test]
    fn test_cumprod_backward_numerical() {
        let x_vals = [2.0_f64, 3.0, 0.5];
        let eps = 1e-6;

        let x = leaf(&x_vals, &[3], true);
        let cp = cumprod(&x, 0).unwrap();
        let loss = sum(&cp).unwrap();
        loss.backward().unwrap();

        let g = x.grad().unwrap().unwrap();
        let gd = g.data().unwrap();

        for idx in 0..3 {
            let mut x_plus = x_vals.to_vec();
            let mut x_minus = x_vals.to_vec();
            x_plus[idx] += eps;
            x_minus[idx] -= eps;

            let tp = leaf(&x_plus, &[3], false);
            let fp = sum(&cumprod(&tp, 0).unwrap()).unwrap().item().unwrap();

            let tm = leaf(&x_minus, &[3], false);
            let fm = sum(&cumprod(&tm, 0).unwrap()).unwrap().item().unwrap();

            let numerical = (fp - fm) / (2.0 * eps);
            assert!(
                (gd[idx] - numerical).abs() < 1e-4,
                "index {idx}: analytic={}, numerical={}",
                gd[idx],
                numerical,
            );
        }
    }

    // =======================================================================
    // cumsum backward numerical gradient check
    // =======================================================================

    #[test]
    fn test_cumsum_backward_numerical() {
        let x_vals = [1.0_f64, -2.0, 3.5, 0.7];
        let eps = 1e-6;

        let x = leaf(&x_vals, &[4], true);
        let cs = cumsum(&x, 0).unwrap();
        let loss = sum(&cs).unwrap();
        loss.backward().unwrap();

        let g = x.grad().unwrap().unwrap();
        let gd = g.data().unwrap();

        for idx in 0..4 {
            let mut x_plus = x_vals.to_vec();
            let mut x_minus = x_vals.to_vec();
            x_plus[idx] += eps;
            x_minus[idx] -= eps;

            let tp = leaf(&x_plus, &[4], false);
            let fp = sum(&cumsum(&tp, 0).unwrap()).unwrap().item().unwrap();

            let tm = leaf(&x_minus, &[4], false);
            let fm = sum(&cumsum(&tm, 0).unwrap()).unwrap().item().unwrap();

            let numerical = (fp - fm) / (2.0 * eps);
            assert!(
                (gd[idx] - numerical).abs() < 1e-4,
                "index {idx}: analytic={}, numerical={}",
                gd[idx],
                numerical,
            );
        }
    }

    // =======================================================================
    // cummax / cummin backward — closes blocker #1231.
    //
    // The expected gradients are constructed from SYMBOLIC CONSTANTS
    // traceable to upstream's `cummaxmin_backward` formula at
    // `aten/src/ATen/native/ReduceOps.cpp:906-918`:
    //
    //   grad_input = zeros_like(input).scatter_add_(dim, indices, grad_output)
    //
    // For a 1D scan with `loss = sum(values)`, grad_output = ones, so
    // grad_input[k] counts the number of output positions whose running
    // max/min pointed at input index k. That count equals the number of
    // entries == k in the indices vector.
    //
    // Tie-break per upstream `std::greater_equal` / `std::less_equal`
    // (`ReduceOps.cpp:832, :871`): on equal values the LATER index wins.
    //
    // Live-verified 2026-05-25 with torch 2.11.0 — the named CONST blocks
    // below cite the verbatim live-torch reproductions, satisfying
    // R-CHAR-3 (no tautological tests; expected values traced to
    // upstream).
    // =======================================================================

    /// `cummax([1,2,3,4]).sum().backward()` — monotonic input, no ties.
    ///
    /// Live torch (verified 2026-05-25):
    /// ```text
    /// x = torch.tensor([1.,2.,3.,4.], requires_grad=True)
    /// torch.cummax(x, 0).values.sum().backward()
    /// x.grad == tensor([1., 1., 1., 1.])
    /// ```
    /// Because the running max is strictly increasing, each input
    /// position is the argmax exactly once — indices = [0,1,2,3], so
    /// scatter_add of ones at those positions gives ones.
    #[test]
    fn test_cummax_backward_monotonic() {
        const EXPECTED: [f64; 4] = [1.0, 1.0, 1.0, 1.0];
        let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[4], true);
        let r = cummax(&x, 0).unwrap();
        // Verify saved indices match upstream's tie-break for a monotonic
        // input (no ties — every position is a fresh argmax).
        assert_eq!(r.indices, vec![0, 1, 2, 3]);
        let loss = sum(&r.values).unwrap();
        loss.backward().unwrap();
        let g = x.grad().unwrap().unwrap();
        let gd = g.data().unwrap();
        for (i, expected) in EXPECTED.iter().enumerate() {
            assert!(
                (gd[i] - expected).abs() < 1e-12,
                "cummax_backward_monotonic: idx={i} got={} expected={}",
                gd[i],
                expected,
            );
        }
        // grad-fn name introspection — proves the autograd node attached.
        assert_eq!(r.values.grad_fn().unwrap().name(), "CummaxBackward");
    }

    /// `cummax([1,2,2,3]).sum().backward()` — tied input at indices 1,2.
    ///
    /// Live torch (verified 2026-05-25):
    /// ```text
    /// x = torch.tensor([1.,2.,2.,3.], requires_grad=True)
    /// vals, idx = torch.cummax(x, 0)
    /// idx  == tensor([0, 1, 2, 3])    # later wins on ties: idx 2 chosen
    /// vals.sum().backward()
    /// x.grad == tensor([1., 1., 1., 1.])
    /// ```
    /// Without the upstream-aligned `>=` tie-break, the saved indices
    /// would be `[0, 1, 1, 3]` (earlier wins) and the resulting gradient
    /// would be `[1, 2, 0, 1]` — the wrong gradient on a tied input.
    /// That mismatch is the precise reason the kernel fix and the
    /// backward landed together.
    #[test]
    fn test_cummax_backward_tie() {
        const EXPECTED_INDICES: [usize; 4] = [0, 1, 2, 3];
        const EXPECTED_GRAD: [f64; 4] = [1.0, 1.0, 1.0, 1.0];
        let x = leaf(&[1.0, 2.0, 2.0, 3.0], &[4], true);
        let r = cummax(&x, 0).unwrap();
        assert_eq!(r.indices, EXPECTED_INDICES.to_vec());
        let loss = sum(&r.values).unwrap();
        loss.backward().unwrap();
        let g = x.grad().unwrap().unwrap();
        let gd = g.data().unwrap();
        for (i, expected) in EXPECTED_GRAD.iter().enumerate() {
            assert!(
                (gd[i] - expected).abs() < 1e-12,
                "cummax_backward_tie: idx={i} got={} expected={}",
                gd[i],
                expected,
            );
        }
    }

    /// `cummin([5,2,2,1]).sum().backward()` — tied input at indices 1,2
    /// for the running minimum.
    ///
    /// Live torch (verified 2026-05-25):
    /// ```text
    /// x = torch.tensor([5.,2.,2.,1.], requires_grad=True)
    /// vals, idx = torch.cummin(x, 0)
    /// idx  == tensor([0, 1, 2, 3])    # later wins on ties: idx 2 chosen
    /// vals.sum().backward()
    /// x.grad == tensor([1., 1., 1., 1.])
    /// ```
    #[test]
    fn test_cummin_backward_tie() {
        const EXPECTED_INDICES: [usize; 4] = [0, 1, 2, 3];
        const EXPECTED_GRAD: [f64; 4] = [1.0, 1.0, 1.0, 1.0];
        let x = leaf(&[5.0, 2.0, 2.0, 1.0], &[4], true);
        let r = cummin(&x, 0).unwrap();
        assert_eq!(r.indices, EXPECTED_INDICES.to_vec());
        let loss = sum(&r.values).unwrap();
        loss.backward().unwrap();
        let g = x.grad().unwrap().unwrap();
        let gd = g.data().unwrap();
        for (i, expected) in EXPECTED_GRAD.iter().enumerate() {
            assert!(
                (gd[i] - expected).abs() < 1e-12,
                "cummin_backward_tie: idx={i} got={} expected={}",
                gd[i],
                expected,
            );
        }
        assert_eq!(r.values.grad_fn().unwrap().name(), "CumminBackward");
    }

    /// `cummax([1.0, NaN, 3.0, 4.0])` — NaN poisons subsequent positions.
    ///
    /// Live torch (verified 2026-05-25):
    /// ```text
    /// x = torch.tensor([1.0, float('nan'), 3.0, 4.0])
    /// torch.cummax(x, 0)
    ///   values  == tensor([1., nan, nan, nan])
    ///   indices == tensor([0, 1, 1, 1])
    /// ```
    /// Once `cur` becomes NaN at position 1, the update predicate
    /// `isnan(curr) || (!isnan(cur) && curr >= cur)` becomes
    /// `isnan(curr) || false` — only a fresh NaN curr can trigger an
    /// update, and 3.0/4.0 are not NaN. So `cur` and `cur_idx` stay
    /// frozen at (NaN, 1) for all subsequent positions, exactly matching
    /// upstream `cummax_cummin_helper` at `ReduceOps.cpp:819`.
    #[test]
    fn test_cummax_forward_nan_propagates() {
        const EXPECTED_INDICES: [usize; 4] = [0, 1, 1, 1];
        let x = leaf(&[1.0, f64::NAN, 3.0, 4.0], &[4], false);
        let r = cummax(&x, 0).unwrap();
        let d = r.values.data().unwrap();
        assert!((d[0] - 1.0).abs() < 1e-12, "values[0] should be 1.0");
        assert!(d[1].is_nan(), "values[1] should be NaN (input is NaN)");
        assert!(d[2].is_nan(), "values[2] should propagate NaN");
        assert!(d[3].is_nan(), "values[3] should propagate NaN");
        assert_eq!(r.indices, EXPECTED_INDICES.to_vec());
    }
}
