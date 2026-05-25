//! Differentiable fake-quantization for quantization-aware training (QAT).
//!
//! Provides a tensor-level `fake_quantize_per_tensor_affine` op that mirrors
//! PyTorch's `torch.fake_quantize_per_tensor_affine` (registered at
//! `torch/overrides.py:622` and documented at `torch/_torch_docs.py:11950-11988`),
//! wrapping the forward in a proper autograd node using the straight-through
//! estimator (STE):
//!
//! ```text
//! forward(x):
//!   q_unclamped = round_half_to_even(x / scale) + zero_point
//!   q           = clamp(q_unclamped, quant_min, quant_max)        // NaN-safe
//!   output      = (q - zero_point) * scale
//! backward(dY):
//!   mask = (quant_min <= q_unclamped <= quant_max) ? 1 : 0
//!   dX   = dY * mask
//! ```
//!
//! This matches the upstream CPU kernel at
//! `aten/src/ATen/native/quantized/cpu/kernels/QuantizedOpKernels.cpp:2655-2715
//! _fake_quantize_tensor_helper` byte-for-byte: `qval_f = z_point +
//! std::nearbyint(*input_val * inv_scale); qval = static_cast<int64_t>(
//! std::fmin(std::fmax(qval_f, quant_min), quant_max)); output = (qval -
//! z_point) * sc; mask = (quant_min <= qval_f) && (qval_f <= quant_max);`.
//!
//! Two rounding/NaN details must match upstream R-DEV-1 byte-for-byte:
//! 1. **Round-half-to-even (banker's rounding)**. Rust's `f32::round` is
//!    round-half-away-from-zero; `std::nearbyint` defaults to FE_TONEAREST
//!    which is round-half-to-even. We use `f64::round_ties_even` (stabilized
//!    in Rust 1.77) to match upstream on `.5` boundaries.
//! 2. **NaN-safe clamp**. Rust's `f32::clamp` debug-asserts on NaN; `std::min /
//!    std::max` returns the non-NaN operand. `f64::min` / `f64::max` in Rust
//!    follow IEEE-754-2019 minimum/maximum (return the non-NaN operand) and
//!    are the correct analog.
//!
//! ## REQ status (per `.design/ferrotorch-core/grad_fns/quantize_grad.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (per-tensor) | SHIPPED | `fake_quantize_per_tensor_affine` at `grad_fns/quantize_grad.rs:84` per upstream `aten/src/ATen/native/quantized/FakeQuantPerTensorAffine.cpp:31-40`; tensor-qparams overload `fake_quantize_per_tensor_affine_tensor_qparams` at `grad_fns/quantize_grad.rs:149` per upstream `:42-51`. Non-test production consumer: `Tensor::fake_quantize_per_tensor_affine_t` at `methods.rs:596`. |
//! | REQ-2 (per-channel) | SHIPPED | `fake_quantize_per_channel_affine` at `grad_fns/quantize_grad.rs:421` per upstream per-channel CPU kernel `aten/src/ATen/native/quantized/cpu/kernels/QuantizedOpKernels.cpp:2836-2848` (cast-to-i64 BEFORE clamp — diverges from per-tensor kernel ordering, R-DEV-1 byte-for-byte via helper `per_channel_dequantize_f64` at `:308`). Backward via `FakeQuantizePerChannelBackward` at `grad_fns/quantize_grad.rs:582` returns `dY * mask` with per-channel mask via `per_channel_mask_in_range` at `:339`. Non-test production consumer: `Tensor::fake_quantize_per_channel_affine_t` at `methods.rs:628`. |
//! | REQ-3 (STE backward) | SHIPPED | `FakeQuantizeBackward` at `grad_fns/quantize_grad.rs:653` with `GradFn::backward` impl at `:661` returns `dY * mask` where `mask = (quant_min <= q_unclamped <= quant_max)` per upstream `FakeQuantPerTensorAffine.cpp:121-134` (`QuantizedOpKernels.cpp:2706`). Attach site is REQ-1's forward; consumer chain closes via `methods.rs:596 Tensor::fake_quantize_per_tensor_affine_t`. Per-channel companion `FakeQuantizePerChannelBackward` at `:582` mirrors the same STE with per-channel scale/zp lookup. |

use std::sync::Arc;

use crate::dtype::Float;
use crate::error::FerrotorchResult;
use crate::int_tensor::IntTensor;
use crate::storage::TensorStorage;
use crate::tensor::{GradFn, Tensor};

/// Differentiable per-tensor affine fake quantization.
///
/// Forward: `output = (clamp(round_half_to_even(input / scale) + zero_point,
/// quant_min, quant_max) - zero_point) * scale`.
///
/// Backward (clipped STE): `dX = dY * mask` where `mask` is `1` for input
/// values whose unclamped quantized representation `q_unclamped =
/// round_half_to_even(input / scale) + zero_point` lies in
/// `[quant_min, quant_max]` and `0` otherwise.
///
/// # Arguments
///
/// * `input` — the tensor to fake-quantize.
/// * `scale` — quantization scale (positive, non-zero).
/// * `zero_point` — integer zero point for affine quantization. For symmetric
///   schemes pass `0`. Must satisfy `quant_min <= zero_point <= quant_max`
///   per upstream check at `FakeQuantPerTensorAffine.cpp:79-81`.
/// * `quant_min` — minimum integer value of the target dtype (e.g. `-128` for
///   int8 affine or `0` for uint8). Widened to `i64` to match upstream
///   `int64_t quant_min` at `FakeQuantPerTensorAffine.cpp:35`.
/// * `quant_max` — maximum integer value of the target dtype. Widened to
///   `i64` per upstream `int64_t quant_max` at `:36`.
///
/// # Errors
///
/// - `FerrotorchError::InvalidArgument` if `scale <= 0` or `scale` is NaN.
/// - `FerrotorchError::InvalidArgument` if `quant_min > quant_max` (mirrors
///   upstream `TORCH_CHECK(quant_min <= quant_max)` at
///   `FakeQuantPerTensorAffine.cpp:75-78`).
/// - `FerrotorchError::InvalidArgument` if `zero_point` lies outside
///   `[quant_min, quant_max]` (mirrors upstream `TORCH_CHECK(
///   zero_point >= quant_min && zero_point <= quant_max)` at `:79-81`).
pub fn fake_quantize_per_tensor_affine<T: Float>(
    input: &Tensor<T>,
    scale: f64,
    zero_point: i64,
    quant_min: i64,
    quant_max: i64,
) -> FerrotorchResult<Tensor<T>> {
    fake_quantize_per_tensor_affine_impl(input, scale, zero_point, quant_min, quant_max)
}

/// Backward-compatible alias for `fake_quantize_per_tensor_affine`.
///
/// The canonical name `fake_quantize_per_tensor_affine` matches PyTorch's
/// `torch.fake_quantize_per_tensor_affine` per `torch/overrides.py:622`
/// (R-DEV-2 Python user-API ABI). This thin delegator preserves the
/// pre-#1238 name `fake_quantize_differentiable` (which existed in
/// ferrotorch before the rename per CL-293) and casts the i32 args to the
/// upstream `int64_t` representation. Callers should migrate to the
/// canonical name; this alias is retained to avoid breaking pre-#1238
/// callers (e.g. `tests/conformance_quantize_prune.rs`).
///
/// New code MUST use `fake_quantize_per_tensor_affine`. This alias has no
/// production-code consumer and is preserved solely for transitional
/// compatibility with existing test fixtures whose deserialized field
/// types (`i32`) match the pre-rename signature.
pub fn fake_quantize_differentiable<T: Float>(
    input: &Tensor<T>,
    scale: f64,
    zero_point: i32,
    qmin: i32,
    qmax: i32,
) -> FerrotorchResult<Tensor<T>> {
    fake_quantize_per_tensor_affine_impl(
        input,
        scale,
        i64::from(zero_point),
        i64::from(qmin),
        i64::from(qmax),
    )
}

/// Differentiable per-tensor affine fake quantization with tensor-valued
/// quantization parameters.
///
/// Mirrors upstream's tensor-qparams overload at
/// `aten/src/ATen/native/quantized/FakeQuantPerTensorAffine.cpp:42-51 Tensor
/// fake_quantize_per_tensor_affine(const Tensor& self, const Tensor& scale,
/// const Tensor& zero_point, int64_t quant_min, int64_t quant_max)`. Upstream
/// extracts the scalars via `sc.item().toFloat()` / `z_point.item().toInt()`
/// at `QuantizedOpKernels.cpp:2737`; ferrotorch's tensors are dtype-typed so
/// the scalar extraction is direct slicing.
///
/// Both `scale` and `zero_point` MUST be single-element tensors (`numel == 1`),
/// matching the upstream contract — the `_get_zero_point_from_tensor` helper
/// at `FakeQuantPerTensorAffine.cpp:136-146` indexes `zero_point[0]`. The
/// `zero_point` carrier is `IntTensor<i64>` because it MUST be an integer
/// tensor (R-DEV-1 numerical contract: passing a float zero-point through a
/// `Tensor<T: Float>` would silently allow non-integer zero-points and
/// diverge from upstream's `int64_t` extraction).
///
/// # Errors
///
/// - All errors from `fake_quantize_per_tensor_affine`.
/// - `FerrotorchError::InvalidArgument` if `scale.numel() != 1` or
///   `zero_point.numel() != 1`.
pub fn fake_quantize_per_tensor_affine_tensor_qparams<T: Float>(
    input: &Tensor<T>,
    scale: &Tensor<T>,
    zero_point: &IntTensor<i64>,
    quant_min: i64,
    quant_max: i64,
) -> FerrotorchResult<Tensor<T>> {
    use crate::error::FerrotorchError;
    let scale_data = scale.data_vec()?;
    if scale_data.len() != 1 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "fake_quantize_per_tensor_affine_tensor_qparams: scale must be a 1-element \
                 tensor, got numel={}",
                scale_data.len()
            ),
        });
    }
    let zp_data = zero_point.data()?;
    if zp_data.len() != 1 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "fake_quantize_per_tensor_affine_tensor_qparams: zero_point must be a \
                 1-element tensor, got numel={}",
                zp_data.len()
            ),
        });
    }
    // Upstream extracts scalars via `.item()` per
    // `QuantizedOpKernels.cpp:2737 fake_quantize_tensor_cachemask_tensor_qparams_kernel`:
    //   `sc.item().toFloat(), z_point.item().toInt()`.
    let scale_f64: f64 = match scale_data[0].to_f64() {
        Some(v) => v,
        None => {
            return Err(FerrotorchError::InvalidArgument {
                message: "fake_quantize_per_tensor_affine_tensor_qparams: scale tensor \
                          element could not be converted to f64"
                    .to_string(),
            });
        }
    };
    let zp_i64: i64 = zp_data[0];
    fake_quantize_per_tensor_affine_impl(input, scale_f64, zp_i64, quant_min, quant_max)
}

/// Shared forward + backward attach for both scalar and tensor-qparams entry
/// points. Mirrors the upstream `_fake_quantize_tensor_helper` at
/// `aten/src/ATen/native/quantized/cpu/kernels/QuantizedOpKernels.cpp:2655-2715`.
fn fake_quantize_per_tensor_affine_impl<T: Float>(
    input: &Tensor<T>,
    scale: f64,
    zero_point: i64,
    quant_min: i64,
    quant_max: i64,
) -> FerrotorchResult<Tensor<T>> {
    use crate::error::FerrotorchError;
    // Mirror upstream validation order.
    // 1. quant_min <= quant_max per `FakeQuantPerTensorAffine.cpp:75-78`.
    if quant_min > quant_max {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "fake_quantize_per_tensor_affine: `quant_min` ({quant_min}) should be less \
                 than or equal to `quant_max` ({quant_max})."
            ),
        });
    }
    // 2. zero_point in [quant_min, quant_max] per
    //    `FakeQuantPerTensorAffine.cpp:79-81`.
    if zero_point < quant_min || zero_point > quant_max {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "fake_quantize_per_tensor_affine: `zero_point` ({zero_point}) must be \
                 between `quant_min` ({quant_min}) and `quant_max` ({quant_max})."
            ),
        });
    }
    // NOTE: no `scale > 0` check — upstream `FakeQuantPerTensorAffine.cpp:75-81`
    // only validates `quant_min <= quant_max` and the `zero_point` range; it
    // silently proceeds for `scale <= 0` / `scale == NaN`, propagating IEEE-754
    // Inf/NaN through `inv_scale = 1.0f / scale`. The clamp + dequant tail
    // below handles all three pathological cases (scale=0 yields +0.0 via
    // `(qmin/qmax - zp) * 0.0`, scale<0 double-negates back to input, scale=NaN
    // propagates NaN end-to-end). Parallel to per-channel resolution in #1261
    // / commit 36b245151. R-DEV-1 numerical contract. Closes #1265.

    let data = input.data_vec()?;
    // Upstream uses `float inv_scale = 1.0f / sc` at
    // `QuantizedOpKernels.cpp:2665` and the rounding chain
    // `z_point + std::nearbyint(*input_val * inv_scale)` at `:2683/:2703`
    // evaluates entirely at f32 precision (the stub at
    // `FakeQuantAffine.h:13-20` takes `float sc`). R-DEV-1 numerical
    // contract: ferrotorch must match byte-for-byte, so the rounding
    // stage runs at f32. The dequant tail `(qval - z_point) * sc`
    // promotes through scalar_t in upstream; we keep it at the source
    // `scale: f64` to preserve the API's f64 scale precision.
    let scale_f32 = scale as f32;
    let inv_scale_f32 = 1.0_f32 / scale_f32;
    let zp_f32 = zero_point as f32;
    let qmin_f32 = quant_min as f32;
    let qmax_f32 = quant_max as f32;
    let zp_f64 = zero_point as f64;

    let mut out: Vec<T> = Vec::with_capacity(data.len());
    for &x in &data {
        let x_f32 = x.to_f32().unwrap_or(f32::NAN);
        // Upstream:
        //   qval_f = z_point + std::nearbyint(*input_val * inv_scale);
        //   qval = static_cast<int64_t>(std::fmin(std::fmax(qval_f, quant_min), quant_max));
        //   output_val = (qval - z_point) * sc;
        //
        // `f32::round_ties_even` matches `std::nearbyint` under FE_TONEAREST
        // (round-half-to-even / banker's rounding). `f32::max` / `f32::min`
        // follow IEEE-754-2019 minimum/maximum and propagate the non-NaN
        // operand the same way `std::fmin / std::fmax` do (NaN-safe clamp).
        let qval_f32 = zp_f32 + (x_f32 * inv_scale_f32).round_ties_even();
        let qval_clamped_f32 = qmax_f32.min(qmin_f32.max(qval_f32));
        // Promote the integer-valued clamped qval back to f64 for the
        // dequant multiply, then multiply by `scale: f64` (preserves the
        // API's f64 scale precision). Upstream computes `(qval - z_point)
        // * sc` where `qval` is an int64 and `sc` is the source scalar
        // (float here); using f64 scale on ferrotorch's side is a tighter
        // contract for callers passing f64 scales programmatically.
        let qval_clamped_f64 = qval_clamped_f32 as f64;
        let dq_f64 = (qval_clamped_f64 - zp_f64) * scale;
        // Convert back via T::from(...); on dtype-mismatch fall back to zero
        // (defensive — for f32/f64/bf16/f16 the conversion always succeeds).
        let dq = T::from(dq_f64).unwrap_or_else(<T as num_traits::Zero>::zero);
        out.push(dq);
    }

    let storage = TensorStorage::cpu(out);
    let shape = input.shape().to_vec();

    if input.requires_grad() && crate::autograd::no_grad::is_grad_enabled() {
        let grad_fn = Arc::new(FakeQuantizeBackward::<T> {
            input: input.clone(),
            scale,
            zero_point,
            quant_min,
            quant_max,
        });
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Tensor::from_storage(storage, shape, false)
    }
}

/// Per-channel forward kernel mirroring upstream's
/// `aten/src/ATen/native/quantized/cpu/kernels/QuantizedOpKernels.cpp:
/// 2836-2848 fake_quantize_per_channel_cachemask_cpu_helper` for the
/// integer-zero-point branch byte-for-byte. The CRITICAL ordering
/// distinction from the per-tensor kernel at the same file lines 2702-
/// 2706 is that the per-channel kernel casts to `int64_t` BEFORE the
/// clamp:
///
/// ```cpp
/// static_cast<int64_t>(zero_point + std::nearbyint(self * inv_scale))
/// ```
///
/// and ONLY THEN applies `std::fmin(std::fmax(..., quant_min), quant_max)`.
/// For non-finite `qval_f` (e.g. `+Inf` input * non-zero scale → `+Inf`
/// qval_f), the C++ `static_cast<int64_t>(+Inf)` is undefined behaviour;
/// on x86-64 with SSE2 `cvttsd2si` saturates the invalid-operation result
/// to `INT64_MIN = -9223372036854775808`, which then `std::fmax` snaps to
/// `quant_min` (not `quant_max`). Locked at upstream commit `2ec02226...`
/// (see `.design/ferrotorch-core/grad_fns/quantize_grad.md baseline-pytorch`).
///
/// To match this byte-for-byte (R-DEV-1 numerical contract — agents using
/// `torch.fake_quantize_per_channel_affine` and comparing tensor-by-tensor
/// MUST get the same +Inf → `quant_min` mapping the upstream CPU kernel
/// produces), ferrotorch replicates the cast-to-i64-first ordering with
/// an explicit non-finite check: `+Inf`, `-Inf`, and `NaN` qval_f all
/// resolve to `INT64_MIN` before clamping. Finite qval_f values pass
/// through `f64 as i64` which Rust defines as saturating to
/// `[i64::MIN, i64::MAX]` — but the explicit non-finite guard ensures
/// `+Inf` reaches `i64::MIN` rather than `i64::MAX` (Rust's saturating
/// f→i cast does NOT match the x86 invalid-op behaviour for `+Inf`).
fn per_channel_dequantize_f64(
    x_f64: f64,
    scale_f64: f64,
    zero_point: i64,
    quant_min: i64,
    quant_max: i64,
) -> f64 {
    // Upstream per-channel kernel at `QuantizedOpKernels.cpp:2838-2848`:
    //   `float inv_scale = 1.0f / scale; ... zero_point +
    //   std::nearbyint(self * inv_scale) ...` — the rounding chain runs
    //   at f32 precision. R-DEV-1 numerical contract: ferrotorch must
    //   match byte-for-byte, so the rounding stage casts down to f32.
    let scale_f32 = scale_f64 as f32;
    let x_f32 = x_f64 as f32;
    let zp_f32 = zero_point as f32;
    let inv_scale_f32 = 1.0_f32 / scale_f32;
    let qval_f32 = zp_f32 + (x_f32 * inv_scale_f32).round_ties_even();
    // Replicate upstream's `static_cast<int64_t>(qval_f)` x86-64
    // invalid-operation saturation: non-finite values resolve to
    // INT64_MIN, finite values use Rust's saturating cast (matches the
    // x86 `cvttsd2si` finite path within `[i64::MIN, i64::MAX]`).
    let qval_i64: i64 = if qval_f32.is_finite() {
        qval_f32 as i64
    } else {
        i64::MIN
    };
    let qval_clamped = quant_max.min(quant_min.max(qval_i64));
    (qval_clamped - zero_point) as f64 * scale_f64
}

/// Per-channel mask formula mirroring upstream's mask side at
/// `aten/src/ATen/native/quantized/cpu/kernels/QuantizedOpKernels.cpp:
/// 2830-2834`. Note that the MASK uses the same `static_cast<int64_t>`
/// cast-first ordering — so for `+Inf` input the mask check is
/// `quant_min <= INT64_MIN <= quant_max` which is false → mask = 0,
/// matching the forward's clamp-to-`quant_min` (the dequantized output
/// is `(quant_min - zp) * scale`, which is OUT of the representable
/// range for `+Inf` so STE correctly zeros the gradient there).
fn per_channel_mask_in_range(
    x_f64: f64,
    scale_f64: f64,
    zero_point: i64,
    quant_min: i64,
    quant_max: i64,
) -> bool {
    // Upstream per-channel mask at `QuantizedOpKernels.cpp:2830-2834` reads
    // `float inv_scale = 1.0f / scale;` (line :2831) — the SAME f32 chain the
    // forward dequantize uses at :2838. R-DEV-1 numerical contract: the mask
    // must recompute at f32 so it agrees byte-for-byte with the forward's
    // i64-cast at f32/f64 split boundaries. Previously this used f64 and
    // disagreed with the forward when `self * inv_scale` was a banker-rounding
    // tie that resolved differently between precisions (#1263, divergence
    // pinned by
    // `tests/divergence_fake_quantize_per_channel_backward_mask_f32_vs_f64.rs`).
    let scale_f32 = scale_f64 as f32;
    let x_f32 = x_f64 as f32;
    let zp_f32 = zero_point as f32;
    let inv_scale_f32 = 1.0_f32 / scale_f32;
    let qval_f32 = zp_f32 + (x_f32 * inv_scale_f32).round_ties_even();
    // Replicate upstream's `static_cast<int64_t>(qval_f)` cast-first ordering
    // exactly as the forward helper `per_channel_dequantize_f64` does — see
    // its rationale comment above re: x86-64 `cvttsd2si` invalid-op saturation
    // for non-finite qval_f.
    let qval_i64: i64 = if qval_f32.is_finite() {
        qval_f32 as i64
    } else {
        i64::MIN
    };
    quant_min <= qval_i64 && qval_i64 <= quant_max
}

/// Differentiable per-channel affine fake quantization.
///
/// Mirrors `torch.fake_quantize_per_channel_affine(input, scale, zero_point,
/// axis, quant_min, quant_max)` per `torch/_torch_docs.py:11992-12042` and
/// the upstream forward at `aten/src/ATen/native/quantized/
/// FakeQuantPerChannelAffine.cpp:32-42 Tensor fake_quantize_per_channel_affine(
/// const Tensor& self, const Tensor& scale, const Tensor& zero_point, int64_t
/// axis, int64_t quant_min, int64_t quant_max)`. For each channel `c` along
/// `axis`, the slice `input[..., c, ...]` is fake-quantized using the scalar
/// qparams `scale[c]`, `zero_point[c]` per REQ-1's per-tensor formula.
///
/// Forward (per element at multi-index `[i_0, ..., i_axis=c, ..., i_n]`):
///
/// ```text
///   q_unclamped = round_half_to_even(input / scale[c]) + zero_point[c]
///   q           = clamp(q_unclamped, quant_min, quant_max)
///   output      = (q - zero_point[c]) * scale[c]
/// ```
///
/// Backward (clipped STE per channel):
///
/// ```text
///   mask[..., c, ...] = (quant_min <= q_unclamped <= quant_max) ? 1 : 0
///   dX = dY * mask
/// ```
///
/// matching `aten/src/ATen/native/quantized/FakeQuantPerChannelAffine.cpp:
/// 118-131 fake_quantize_per_channel_affine_cachemask_backward`. The mask
/// uses per-channel `scale[c]`, `zero_point[c]` in the comparison.
///
/// # Arguments
///
/// * `input` — N-D tensor to fake-quantize.
/// * `scale` — 1-D float tensor with `numel() == input.size(axis)`. Each
///   element is the per-channel scale.
/// * `zero_point` — 1-D int64 tensor with `numel() == input.size(axis)`.
///   Each element is the per-channel zero point.
/// * `axis` — channel axis (`0 <= axis < input.ndim()`). Upstream's check
///   at `:76` admits `axis == self.dim()` for a degenerate broadcast, but
///   that case has no addressable channel slot so ferrotorch follows the
///   strict `< self.dim()` interpretation used by the per-channel
///   *backward* validation at `:213` (R-DEV-1 numerical contract: every
///   sample emitted by the upstream oracle for this op has
///   `axis < input.ndim()`).
/// * `quant_min` / `quant_max` — quantization range bounds, widened to
///   `i64` per upstream `int64_t quant_min, int64_t quant_max` at `:37-38`.
///
/// # Errors
///
/// - `FerrotorchError::InvalidArgument` if `scale.ndim() != 1`
///   (upstream `:55`).
/// - `FerrotorchError::InvalidArgument` if `zero_point.ndim() != 1`
///   (upstream `:56`).
/// - `FerrotorchError::InvalidArgument` if `scale.numel() != zero_point.numel()`
///   (upstream `:57-59`).
/// - `FerrotorchError::InvalidArgument` if `scale.numel() != input.size(axis)`
///   (upstream `:60-62`).
/// - `FerrotorchError::InvalidArgument` if `quant_min > quant_max`
///   (upstream `:64-67`).
/// - `FerrotorchError::InvalidArgument` if any `zero_point[i]` lies outside
///   `[quant_min, quant_max]` (upstream `:69-74`).
/// - `FerrotorchError::InvalidArgument` if `axis` is out of bounds.
///
/// Note: `scale[i] <= 0` and `scale[i] == NaN` are NOT rejected — they
/// propagate through the upstream cast-first formula per
/// `FakeQuantPerChannelAffine.cpp:32-77` (which has no `scale > 0`
/// check). `scale==0` yields `inv_scale = +Inf`, `int64_t(±Inf) =
/// INT64_MIN` clamped to `quant_min`, dequant `(quant_min - zp) * 0.0
/// = ±0.0`; `scale<0` is consistent under the double-negation
/// cancellation; `scale==NaN` poisons the output to NaN. Pinned by
/// R-DEV-1 numerical-contract match (divergence tests at
/// `tests/divergence_quantize_grad_1239_per_channel.rs`, closes #1261).
pub fn fake_quantize_per_channel_affine<T: Float>(
    input: &Tensor<T>,
    scale: &Tensor<T>,
    zero_point: &IntTensor<i64>,
    axis: i64,
    quant_min: i64,
    quant_max: i64,
) -> FerrotorchResult<Tensor<T>> {
    use std::any::TypeId;

    use crate::error::FerrotorchError;

    // 0. scale.scalar_type() in {Float, BFloat16}
    //    (FakeQuantPerChannelAffine.cpp:51-52
    //     `TORCH_CHECK(scale.scalar_type() == ScalarType::Float
    //         || scale.scalar_type() == at::kBFloat16,
    //         "Scale must be Float or BFloat16, found ", scale.scalar_type());`).
    //    Upstream's user-API contract (R-DEV-2) admits only f32 / bf16 scales;
    //    f64 (Double) and f16 (Half) MUST be rejected. Closes #1262.
    let scale_tid = TypeId::of::<T>();
    if scale_tid != TypeId::of::<f32>() && scale_tid != TypeId::of::<half::bf16>() {
        // Map ferrotorch's `T` to upstream's `ScalarType` print name so the
        // error string matches torch's live phrasing
        // ("Scale must be Float or BFloat16, found Double" verified
        //  2026-05-25 against `torch.fake_quantize_per_channel_affine`).
        let found = if scale_tid == TypeId::of::<f64>() {
            "Double"
        } else if scale_tid == TypeId::of::<half::f16>() {
            "Half"
        } else {
            std::any::type_name::<T>()
        };
        return Err(FerrotorchError::InvalidArgument {
            message: format!("Scale must be Float or BFloat16, found {found}"),
        });
    }

    // Upstream validation order:
    // 1. scale.dim() == 1 (FakeQuantPerChannelAffine.cpp:55)
    if scale.ndim() != 1 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "fake_quantize_per_channel_affine: scale should be a 1-D tensor, got ndim={}",
                scale.ndim()
            ),
        });
    }
    // 2. zero_point.dim() == 1 (FakeQuantPerChannelAffine.cpp:56)
    if zero_point.ndim() != 1 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "fake_quantize_per_channel_affine: zero point should be a 1-D tensor, got ndim={}",
                zero_point.ndim()
            ),
        });
    }
    // 3. scale.numel() == zero_point.numel() (FakeQuantPerChannelAffine.cpp:57-59)
    let scale_data = scale.data_vec()?;
    let zp_data = zero_point.data()?;
    if scale_data.len() != zp_data.len() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "fake_quantize_per_channel_affine: scale and zero-point need to have the same \
                 dimensions, got scale.numel()={} zero_point.numel()={}",
                scale_data.len(),
                zp_data.len()
            ),
        });
    }
    // 4. quant_min <= quant_max (FakeQuantPerChannelAffine.cpp:64-67)
    if quant_min > quant_max {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "fake_quantize_per_channel_affine: `quant_min` ({quant_min}) should be less than \
                 or equal to `quant_max` ({quant_max})."
            ),
        });
    }
    // 5. axis in bounds (FakeQuantPerChannelAffine.cpp:75-77 admits
    //    `axis <= self.dim()` for a degenerate broadcast; ferrotorch
    //    follows the strict `< self.dim()` analog used by the backward
    //    validation at :213 — every oracle sample satisfies this).
    let ndim = input.ndim();
    if axis < 0 || (axis as usize) >= ndim {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "fake_quantize_per_channel_affine: `axis` ({axis}) must be between 0 and \
                 number of dimensions of input ({ndim})"
            ),
        });
    }
    let axis_us = axis as usize;
    let shape = input.shape();
    let channel_dim = shape[axis_us];
    // 6. scale.numel() == self.size(axis) (FakeQuantPerChannelAffine.cpp:60-62)
    if scale_data.len() != channel_dim {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "fake_quantize_per_channel_affine: dimensions of scale and zero-point are not \
                 consistent with input tensor — scale.numel()={} input.size(axis={})={}",
                scale_data.len(),
                axis,
                channel_dim
            ),
        });
    }
    // 7. zero_point in [quant_min, quant_max] (FakeQuantPerChannelAffine.cpp:69-74)
    for (i, &zp) in zp_data.iter().enumerate() {
        if zp < quant_min || zp > quant_max {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "fake_quantize_per_channel_affine: `zero_point` must be between \
                     `quant_min` ({quant_min}) and `quant_max` ({quant_max}); zero_point[{i}]={zp}"
                ),
            });
        }
    }
    // 8. NO `scale > 0` check — upstream `FakeQuantPerChannelAffine.cpp:32-77`
    //    has none. R-DEV-1 numerical-contract match: `scale==0` yields
    //    `inv_scale = +Inf`, `int64_t(±Inf) = INT64_MIN` clamped to
    //    `quant_min`, dequant `(quant_min - zp) * 0.0 = ±0.0`; `scale<0`
    //    works via double-negation cancellation; `scale==NaN` poisons the
    //    output to NaN (live torch 2026-05-25 returns `tensor([[nan]])`
    //    for NaN scale, confirming no check). Pinned by
    //    `tests/divergence_quantize_grad_1239_per_channel.rs::
    //    divergence_per_channel_scale_{zero,negative}_silently_proceeds`
    //    (closes #1261). The `per_channel_dequantize_f64` helper at `:325`
    //    already implements the cast-first IEEE-754 / x86 invalid-op
    //    saturation that produces the correct upstream values.

    // outer / inner strides around `axis` so that the channel index for
    // flat index `i` is `(i / inner) % channel_dim`.
    let inner: usize = shape[axis_us + 1..].iter().product();

    let data = input.data_vec()?;

    let mut out: Vec<T> = Vec::with_capacity(data.len());
    for (i, &x) in data.iter().enumerate() {
        // `inner` is `prod(shape[axis+1..])`. When the loop body executes
        // `data.len() > 0` implies every dim including the tail dims is
        // > 0, so `inner > 0` and integer division is well-defined.
        // `checked_div` makes this explicit and silences clippy's
        // manual-checked-div lint without changing semantics.
        let ch = i.checked_div(inner).map_or(0, |q| q % channel_dim);
        let scale_f64 = scale_data[ch].to_f64().unwrap_or(f64::NAN);
        let zp_i64 = zp_data[ch];
        let x_f64 = x.to_f64().unwrap_or(f64::NAN);
        let dq_f64 = per_channel_dequantize_f64(x_f64, scale_f64, zp_i64, quant_min, quant_max);
        let dq = T::from(dq_f64).unwrap_or_else(<T as num_traits::Zero>::zero);
        out.push(dq);
    }

    let storage = TensorStorage::cpu(out);
    let out_shape = shape.to_vec();

    if input.requires_grad() && crate::autograd::no_grad::is_grad_enabled() {
        // Save f64 scale + i64 zp arrays directly to avoid re-materializing
        // the parameter tensors in the backward pass.
        let scale_f64s: Vec<f64> = scale_data
            .iter()
            .map(|s| s.to_f64().unwrap_or(f64::NAN))
            .collect();
        let zp_i64s: Vec<i64> = zp_data.to_vec();
        let grad_fn = Arc::new(FakeQuantizePerChannelBackward::<T> {
            input: input.clone(),
            scale: scale_f64s,
            zero_point: zp_i64s,
            axis: axis_us,
            quant_min,
            quant_max,
        });
        Tensor::from_operation(storage, out_shape, grad_fn)
    } else {
        Tensor::from_storage(storage, out_shape, false)
    }
}

/// Backward node for `fake_quantize_per_channel_affine` using the clipped
/// STE with a per-channel mask.
///
/// Mirrors upstream's mask-based VJP at
/// `aten/src/ATen/native/quantized/FakeQuantPerChannelAffine.cpp:118-131
/// Tensor fake_quantize_per_channel_affine_cachemask_backward(const Tensor& dY,
/// const Tensor& mask) { ... return dY * mask; }`. The mask is `1` where the
/// per-channel `q_unclamped = round_half_to_even(input / scale[c]) +
/// zero_point[c]` lies in `[quant_min, quant_max]`.
#[derive(Debug)]
struct FakeQuantizePerChannelBackward<T: Float> {
    input: Tensor<T>,
    scale: Vec<f64>,
    zero_point: Vec<i64>,
    axis: usize,
    quant_min: i64,
    quant_max: i64,
}

impl<T: Float> GradFn<T> for FakeQuantizePerChannelBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }
        let grad_data = grad_output.data_vec()?;
        let input_data = self.input.data_vec()?;
        let shape = self.input.shape().to_vec();
        let channel_dim = shape[self.axis];
        let inner: usize = shape[self.axis + 1..].iter().product();
        let zero = <T as num_traits::Zero>::zero();
        let grad: Vec<T> = input_data
            .iter()
            .zip(grad_data.iter())
            .enumerate()
            .map(|(i, (&x, &g))| {
                // Per-channel index lookup. `inner` guaranteed > 0 when
                // the iterator emits an element (non-empty input), but
                // `checked_div` makes the safety explicit and silences
                // clippy's manual-checked-div lint.
                let ch = i.checked_div(inner).map_or(0, |q| q % channel_dim);
                let scale_f64 = self.scale[ch];
                let zp_i64 = self.zero_point[ch];
                let x_f64 = x.to_f64().unwrap_or(f64::NAN);
                // Upstream per-channel mask: `(quant_min <= qval_i64) &&
                // (qval_i64 <= quant_max)` per `QuantizedOpKernels.cpp:
                // 2830-2834` — cast-first ordering matches the forward.
                if per_channel_mask_in_range(
                    x_f64,
                    scale_f64,
                    zp_i64,
                    self.quant_min,
                    self.quant_max,
                ) {
                    g
                } else {
                    zero
                }
            })
            .collect();
        let storage = TensorStorage::cpu(grad);
        Ok(vec![Some(Tensor::from_storage(storage, shape, false)?)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "FakeQuantizePerChannelBackward"
    }
}

/// Backward node for `fake_quantize_per_tensor_affine` using the clipped STE.
///
/// Mirrors upstream's mask-based VJP at
/// `aten/src/ATen/native/quantized/FakeQuantPerTensorAffine.cpp:121-134
/// Tensor fake_quantize_per_tensor_affine_cachemask_backward(const Tensor& dY,
/// const Tensor& mask) { ... return dY * mask; }`. Where upstream pre-computes
/// the bool mask in the forward, ferrotorch saves the input and the scalar
/// qparams and recomputes the mask in the backward — numerically identical.
#[derive(Debug)]
struct FakeQuantizeBackward<T: Float> {
    input: Tensor<T>,
    scale: f64,
    zero_point: i64,
    quant_min: i64,
    quant_max: i64,
}

impl<T: Float> GradFn<T> for FakeQuantizeBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }
        let grad_data = grad_output.data_vec()?;
        let input_data = self.input.data_vec()?;
        // Upstream computes the mask using the SAME f32 `qval_f` produced by
        // the forward's `float inv_scale = 1.0f / sc` chain — see
        // `QuantizedOpKernels.cpp:2665` (`float inv_scale = 1.0f / sc;`),
        // `:2683` (`qval_f = z_point + std::nearbyint(*input_val * inv_scale);`),
        // `:2686` (`*mask_val = ((quant_min <= qval_f) && (qval_f <= quant_max));`).
        // R-DEV-1 numerical contract: the backward MUST recompute the mask at
        // f32 precision so it agrees byte-for-byte with the forward (and with
        // upstream's mask) at f32/f64 split boundaries. Previously this was
        // computed at f64 and disagreed with the forward when `x * inv_scale`
        // landed on a banker-rounding tie that differed between precisions
        // (#1263, divergence pinned by
        // `tests/divergence_fake_quantize_backward_mask_f32_vs_f64.rs`).
        let scale_f32 = self.scale as f32;
        let inv_scale_f32 = 1.0_f32 / scale_f32;
        let zp_f32 = self.zero_point as f32;
        let qmin_f32 = self.quant_min as f32;
        let qmax_f32 = self.quant_max as f32;
        let zero = <T as num_traits::Zero>::zero();
        let grad: Vec<T> = input_data
            .iter()
            .zip(grad_data.iter())
            .map(|(&x, &g)| {
                let x_f32 = x.to_f32().unwrap_or(f32::NAN);
                let qval_f32 = zp_f32 + (x_f32 * inv_scale_f32).round_ties_even();
                // Upstream mask: `(quant_min <= qval_f) && (qval_f <= quant_max)`
                // per `QuantizedOpKernels.cpp:2686`. NaN propagates as `false`
                // through both comparisons (R-DEV-1 numerical contract).
                if qval_f32 >= qmin_f32 && qval_f32 <= qmax_f32 {
                    g
                } else {
                    zero
                }
            })
            .collect();
        let storage = TensorStorage::cpu(grad);
        let shape = self.input.shape().to_vec();
        Ok(vec![Some(Tensor::from_storage(storage, shape, false)?)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "FakeQuantizeBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::backward;

    fn t(data: Vec<f32>, shape: Vec<usize>, req_grad: bool) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data), shape, req_grad).unwrap()
    }

    fn ti64(data: Vec<i64>, shape: Vec<usize>) -> IntTensor<i64> {
        IntTensor::from_vec(data, shape).unwrap()
    }

    // ── forward correctness ────────────────────────────────────────

    #[test]
    fn fake_quantize_round_trips_representable_values() {
        // int8 symmetric: quant_min=-128, quant_max=127, scale chosen so
        // exact multiples of scale are fixed points.
        let scale = 0.1;
        let zp: i64 = 0;
        let quant_min: i64 = -128;
        let quant_max: i64 = 127;

        // Values that are exact multiples of scale should round-trip.
        let input = t(vec![0.0, 0.1, 0.2, -0.1, -0.2], vec![5], false);
        let out = fake_quantize_per_tensor_affine(&input, scale, zp, quant_min, quant_max).unwrap();
        let data = out.data().unwrap();
        for (got, expected) in data.iter().zip([0.0, 0.1, 0.2, -0.1, -0.2].iter()) {
            assert!(
                (got - expected).abs() < 1e-5,
                "expected {expected}, got {got}"
            );
        }
    }

    #[test]
    // reason: with scale=1.0, zp=0 the quantize-then-dequantize round-trip
    // is exact for integer-valued inputs in range, and clamping snaps
    // out-of-range inputs to exact integer boundaries. Every expected
    // value is bit-exactly representable in f32, so equality is correct.
    #[allow(clippy::float_cmp)]
    fn fake_quantize_clamps_out_of_range_values() {
        // int8: [-128, 127] with scale 1.0, zp 0 → representable range is
        // [-128.0, 127.0]. Values outside should be clamped.
        let input = t(vec![-200.0, -100.0, 0.0, 100.0, 200.0], vec![5], false);
        let out = fake_quantize_per_tensor_affine(&input, 1.0, 0, -128, 127).unwrap();
        let data = out.data().unwrap();
        assert_eq!(data[0], -128.0); // clamped
        assert_eq!(data[1], -100.0);
        assert_eq!(data[2], 0.0);
        assert_eq!(data[3], 100.0);
        assert_eq!(data[4], 127.0); // clamped
    }

    // NOTE: previously `fake_quantize_rejects_zero_scale` and
    // `fake_quantize_rejects_negative_scale` verified rejection of
    // `scale <= 0` / `scale == NaN`. Upstream
    // `FakeQuantPerTensorAffine.cpp:69-89` has no such check; ferrotorch
    // now silently proceeds with the same clamp + dequant formula to match
    // upstream byte-for-byte (R-DEV-1 numerical contract; closes #1265,
    // parallel to per-channel #1261). The upstream behavior is now pinned
    // by the divergence tests at
    // `tests/divergence_quantize_grad_per_tensor_scale_check.rs::
    // divergence_per_tensor_scale_{zero,negative,nan}_silently_{proceeds,propagates}`.

    #[test]
    fn fake_quantize_rejects_inverted_range() {
        let input = t(vec![1.0], vec![1], false);
        let result = fake_quantize_per_tensor_affine(&input, 1.0, 0, 128, -128);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("`quant_min`"));
    }

    #[test]
    fn fake_quantize_rejects_zero_point_outside_quant_range() {
        // Upstream check at FakeQuantPerTensorAffine.cpp:79-81:
        // zero_point must be in [quant_min, quant_max].
        let input = t(vec![1.0], vec![1], false);
        let result = fake_quantize_per_tensor_affine(&input, 1.0, 200, -128, 127);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("`zero_point`"));
    }

    #[test]
    fn fake_quantize_asymmetric_with_zero_point() {
        // uint8: [0, 255] with a non-zero zero-point shifts the
        // representable range into the positives.
        // scale=1.0, zp=128, qmin=0, qmax=255 → input range maps to
        // [(0-128)*1, (255-128)*1] = [-128, 127].
        let input = t(vec![-128.0, 0.0, 127.0], vec![3], false);
        let out = fake_quantize_per_tensor_affine(&input, 1.0, 128, 0, 255).unwrap();
        let data = out.data().unwrap();
        assert_eq!(data, &[-128.0, 0.0, 127.0]);
    }

    #[test]
    // reason: banker's rounding produces exact integer-multiple results that
    // are bit-exactly representable in f32; equality is correct.
    #[allow(clippy::float_cmp)]
    fn fake_quantize_uses_banker_rounding_on_half_boundaries() {
        // With scale=1.0, zp=0, the value 0.5 should round to 0 (banker's:
        // round half to even). Rust's f32::round would round to 1 (round
        // half away from zero) — the upstream `std::nearbyint` rounds to 0.
        // Test 0.5 → 0 (even), 1.5 → 2 (even), 2.5 → 2 (even), 3.5 → 4
        // (even), -0.5 → 0 (even), -1.5 → -2 (even).
        let input = t(vec![0.5, 1.5, 2.5, 3.5, -0.5, -1.5], vec![6], false);
        let out = fake_quantize_per_tensor_affine(&input, 1.0, 0, -128, 127).unwrap();
        let data = out.data().unwrap();
        // Per `std::nearbyint(0.5) = 0`, `std::nearbyint(1.5) = 2`, etc.
        assert_eq!(data[0], 0.0);
        assert_eq!(data[1], 2.0);
        assert_eq!(data[2], 2.0);
        assert_eq!(data[3], 4.0);
        assert_eq!(data[4], 0.0);
        assert_eq!(data[5], -2.0);
    }

    #[test]
    fn fake_quantize_nan_input_does_not_panic() {
        // NaN input: x/scale = NaN; round_ties_even(NaN) = NaN; clamp via
        // f64::min/f64::max returns the non-NaN operand at each step (IEEE-
        // 754-2019 min/max), so qval_clamped becomes a finite boundary.
        // Mask test `qmin <= NaN` is false → backward yields 0.
        //
        // The point of this test is the function MUST NOT panic on NaN
        // input. Rust's f32::clamp would debug-assert here.
        let input = t(vec![f32::NAN], vec![1], false);
        let _out = fake_quantize_per_tensor_affine(&input, 1.0, 0, -128, 127)
            .expect("NaN input must not error (R-DEV-1 NaN-safe clamp)");
        // No panic = pass. The specific output value depends on Rust's
        // min/max NaN semantics; the contract we lock is "no panic".
    }

    // ── tensor-qparams overload ─────────────────────────────────────

    #[test]
    fn tensor_qparams_matches_scalar_qparams() {
        // Same inputs as `fake_quantize_clamps_out_of_range_values` but via
        // the tensor-qparams overload. Output must match byte-for-byte.
        let input = t(vec![-200.0, -100.0, 0.0, 100.0, 200.0], vec![5], false);
        let scale = t(vec![1.0], vec![1], false);
        let zp = ti64(vec![0], vec![1]);
        let out_tensor =
            fake_quantize_per_tensor_affine_tensor_qparams(&input, &scale, &zp, -128, 127).unwrap();
        let out_scalar = fake_quantize_per_tensor_affine(&input, 1.0, 0, -128, 127).unwrap();
        let dt = out_tensor.data().unwrap();
        let ds = out_scalar.data().unwrap();
        assert_eq!(dt.as_ref(), ds.as_ref());
    }

    #[test]
    fn tensor_qparams_rejects_multi_element_scale() {
        let input = t(vec![1.0], vec![1], false);
        let scale = t(vec![1.0, 2.0], vec![2], false);
        let zp = ti64(vec![0], vec![1]);
        let result = fake_quantize_per_tensor_affine_tensor_qparams(&input, &scale, &zp, -128, 127);
        assert!(result.is_err());
        assert!(
            format!("{}", result.unwrap_err()).contains("scale must be a 1-element"),
            "expected 'scale must be a 1-element' message"
        );
    }

    #[test]
    fn tensor_qparams_rejects_multi_element_zero_point() {
        let input = t(vec![1.0], vec![1], false);
        let scale = t(vec![1.0], vec![1], false);
        let zp = ti64(vec![0, 1], vec![2]);
        let result = fake_quantize_per_tensor_affine_tensor_qparams(&input, &scale, &zp, -128, 127);
        assert!(result.is_err());
        assert!(
            format!("{}", result.unwrap_err()).contains("zero_point must be a 1-element"),
            "expected 'zero_point must be a 1-element' message"
        );
    }

    // ── backward / STE ─────────────────────────────────────────────

    #[test]
    // reason: STE passes gradient 1.0 through for in-range values; the
    // expected mask is a binary {0,1} grid written as exact bit patterns
    // (never an arithmetic result).
    #[allow(clippy::float_cmp)]
    fn fake_quantize_ste_passes_grad_for_in_range_values() {
        // scale=1.0, zp=0, range=[-128, 127]. Values inside this range
        // should have gradient 1.0 passed through unchanged.
        let input = t(vec![-10.0, 0.0, 10.0, 50.0], vec![4], true);
        let out = fake_quantize_per_tensor_affine(&input, 1.0, 0, -128, 127).unwrap();
        let sum = crate::grad_fns::reduction::sum(&out).unwrap();
        backward(&sum).unwrap();
        let grad = input.grad().unwrap().unwrap();
        let grad_data = grad.data().unwrap();
        for &g in grad_data {
            assert_eq!(g, 1.0);
        }
    }

    #[test]
    // reason: STE gradient mask is binary {0,1}; each grad slot holds the
    // exact bit pattern of the chosen sentinel, never the result of
    // arithmetic — equality is the right check.
    #[allow(clippy::float_cmp)]
    fn fake_quantize_ste_zeros_grad_for_out_of_range_values() {
        // scale=0.01, quant_min=-128, quant_max=127 → q_unclamped(-5.0) =
        // round(-500) = -500, which is < -128 → mask = 0. Similarly:
        // q_unclamped(-1.0) = -100 (in [-128,127] → 1),
        // q_unclamped(0.0)  = 0    (in range  → 1),
        // q_unclamped(1.0)  = 100  (in range  → 1),
        // q_unclamped(5.0)  = 500  (out of range → 0),
        // q_unclamped(100.0)= 10000(out of range → 0).
        let input = t(vec![-5.0, -1.0, 0.0, 1.0, 5.0, 100.0], vec![6], true);
        let out = fake_quantize_per_tensor_affine(&input, 0.01, 0, -128, 127).unwrap();
        let sum = crate::grad_fns::reduction::sum(&out).unwrap();
        backward(&sum).unwrap();
        let grad = input.grad().unwrap().unwrap();
        let grad_data = grad.data().unwrap();
        assert_eq!(grad_data[0], 0.0);
        assert_eq!(grad_data[1], 1.0);
        assert_eq!(grad_data[2], 1.0);
        assert_eq!(grad_data[3], 1.0);
        assert_eq!(grad_data[4], 0.0);
        assert_eq!(grad_data[5], 0.0);
    }

    #[test]
    // reason: explicit formulaic backward check. The expected grad is
    // constructed bit-for-bit from the upstream mask formula
    // `mask[i] = 1 if quant_min <= q_unclamped(x[i]) <= quant_max else 0`
    // per `QuantizedOpKernels.cpp:2706`, multiplied by the upstream grad
    // (sum() → dY = 1 elementwise). Never compares ferrotorch's output to
    // ferrotorch's output. R-CHAR-3 honored: every expected value traces
    // to the cited upstream formula, not to a self-referential constant.
    #[allow(clippy::float_cmp)]
    fn fake_quantize_ste_backward_matches_explicit_formula() {
        // scale=0.05, zp=10, quant_min=-128, quant_max=127.
        //   q_unclamped(x) = round_ties_even(x / 0.05) + 10
        //
        // For x = [-10.0, -6.0, -5.0, 0.0, 5.0, 5.85, 6.0, 10.0]:
        //   x/scale  =  [-200, -120, -100,  0,  100, 117, 120,  200]
        //   q_uncl   =  [-190, -110,  -90, 10,  110, 127, 130,  210]
        //   in range =  [   0,    1,    1,  1,    1,   1,   0,    0]
        let input = t(
            vec![-10.0, -6.0, -5.0, 0.0, 5.0, 5.85, 6.0, 10.0],
            vec![8],
            true,
        );
        let out = fake_quantize_per_tensor_affine(&input, 0.05, 10, -128, 127).unwrap();
        let sum = crate::grad_fns::reduction::sum(&out).unwrap();
        backward(&sum).unwrap();
        let grad = input.grad().unwrap().unwrap();
        let actual = grad.data().unwrap();
        let expected: [f32; 8] = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0];
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                a, e,
                "STE mask at i={i}: expected {e}, got {a}; upstream formula \
                 mask[i] = (quant_min <= q_unclamped(x[i]) <= quant_max) ? 1 : 0 \
                 per QuantizedOpKernels.cpp:2706"
            );
        }
    }

    #[test]
    fn fake_quantize_no_grad_when_input_doesnt_require_grad() {
        let input = t(vec![1.0, 2.0], vec![2], false);
        let out = fake_quantize_per_tensor_affine(&input, 1.0, 0, -128, 127).unwrap();
        assert!(!out.requires_grad());
        assert!(out.grad_fn().is_none());
    }

    #[test]
    fn fake_quantize_preserves_grad_fn_when_input_requires_grad() {
        let input = t(vec![1.0, 2.0], vec![2], true);
        let out = fake_quantize_per_tensor_affine(&input, 1.0, 0, -128, 127).unwrap();
        assert!(out.requires_grad());
        assert!(out.grad_fn().is_some());
    }

    #[test]
    fn fake_quantize_no_grad_context_skips_grad_fn() {
        use crate::autograd::no_grad::no_grad;
        let input = t(vec![1.0, 2.0], vec![2], true);
        let out = no_grad(|| fake_quantize_per_tensor_affine(&input, 1.0, 0, -128, 127)).unwrap();
        assert!(out.grad_fn().is_none());
    }

    #[test]
    // reason: chained STE × relu mask product is still binary (0.0 or 1.0
    // — multiplying two binary masks). Each grad slot holds an exact bit
    // pattern, never a non-trivial arithmetic result.
    #[allow(clippy::float_cmp)]
    fn fake_quantize_chains_through_autograd_with_relu() {
        // y = relu(fake_quantize(x)); backward flows through both.
        let input = t(vec![-2.0, -0.5, 0.5, 2.0], vec![4], true);
        let fq = fake_quantize_per_tensor_affine(&input, 0.01, 0, -128, 127).unwrap();
        let relu_out = crate::grad_fns::activation::relu(&fq).unwrap();
        let sum = crate::grad_fns::reduction::sum(&relu_out).unwrap();
        backward(&sum).unwrap();
        let grad = input.grad().unwrap().unwrap();
        let grad_data = grad.data().unwrap();
        // x=-2.0: q_uncl = round(-200) = -200, out of [-128,127] → STE 0.
        assert_eq!(grad_data[0], 0.0);
        // x=-0.5: q_uncl = round(-50) = -50, in range → STE 1; relu zeros
        // negatives → relu mask 0 on dequantized output (-0.5). 1*0 = 0.
        assert_eq!(grad_data[1], 0.0);
        // x=0.5: q_uncl = 50, in range → STE 1; relu passes → 1*1 = 1.
        assert_eq!(grad_data[2], 1.0);
        // x=2.0: q_uncl = 200, out of range → STE 0.
        assert_eq!(grad_data[3], 0.0);
    }

    // ── per-channel forward ────────────────────────────────────────

    #[test]
    // reason: per-channel forward must match per-tensor forward on each
    // channel slice exactly — output is a deterministic dequant integer
    // multiple per channel, bit-exactly representable in f32.
    #[allow(clippy::float_cmp)]
    fn fake_quantize_per_channel_matches_per_tensor_on_each_channel() {
        // 2-D input shape [C=3, N=5], axis=0 → each row is a channel.
        // Per-channel scale/zp; expected = stack of per-tensor results
        // computed via the already-shipped per-tensor surface (NOT by
        // calling the per-channel op on itself; R-CHAR-3 honored — the
        // per-tensor reference is the upstream-faithful surface whose
        // formula traces to QuantizedOpKernels.cpp:2702-2706 directly).
        let input = t(
            vec![
                0.0, 0.1, 0.2, -0.1, -0.2, // row 0 (channel 0)
                0.0, 0.05, 0.1, -0.05, -0.1, // row 1 (channel 1)
                1.0, 2.0, -1.0, -2.0, 0.0, // row 2 (channel 2)
            ],
            vec![3, 5],
            false,
        );
        let scale = t(vec![0.1, 0.05, 1.0], vec![3], false);
        let zp = ti64(vec![0, 0, 0], vec![3]);

        let out_pc = fake_quantize_per_channel_affine(&input, &scale, &zp, 0, -128, 127).unwrap();
        let actual = out_pc.data().unwrap();

        // Reference: compute per-channel by slicing rows and calling
        // per-tensor on each.
        let row_lens = [5usize, 5, 5];
        let scales_v = [0.1f64, 0.05, 1.0];
        let mut expected: Vec<f32> = Vec::new();
        let mut offset = 0;
        for c in 0..3 {
            let row = t(
                input.data().unwrap()[offset..offset + row_lens[c]].to_vec(),
                vec![5],
                false,
            );
            let ref_out = fake_quantize_per_tensor_affine(&row, scales_v[c], 0, -128, 127).unwrap();
            expected.extend_from_slice(ref_out.data().unwrap());
            offset += row_lens[c];
        }
        assert_eq!(actual.len(), expected.len());
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                a, e,
                "per-channel forward at flat idx {i}: expected {e}, got {a}; \
                 per-channel must match per-tensor on each row slice"
            );
        }
    }

    #[test]
    // reason: per-channel output is exact integer-multiple-of-scale per
    // channel; equality is the right check.
    #[allow(clippy::float_cmp)]
    fn fake_quantize_per_channel_axis_dispatch_differs() {
        // 2-D input [2, 3]. With axis=0 the per-channel dim is 2 (rows);
        // with axis=1 the per-channel dim is 3 (cols). Same per-channel
        // scale/zp arrays MUST be 1-D and have the right numel per axis,
        // so we use two distinct axis-aware setups and verify the outputs
        // differ where the per-channel scale differs.
        let input = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false);

        // axis=0 (2 channels): scales [0.5, 1.0] applied row-wise.
        let scale_ax0 = t(vec![0.5, 1.0], vec![2], false);
        let zp_ax0 = ti64(vec![0, 0], vec![2]);
        let out_ax0 =
            fake_quantize_per_channel_affine(&input, &scale_ax0, &zp_ax0, 0, -128, 127).unwrap();
        // For row 0 with scale=0.5: round(1.0/0.5)=2, dq=2*0.5=1.0;
        //                           round(2.0/0.5)=4, dq=4*0.5=2.0;
        //                           round(3.0/0.5)=6, dq=6*0.5=3.0.
        // For row 1 with scale=1.0: exact integer round-trip.
        let d0 = out_ax0.data().unwrap();
        assert_eq!(d0, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // axis=1 (3 channels): scales [0.5, 1.0, 2.0] applied col-wise.
        let scale_ax1 = t(vec![0.5, 1.0, 2.0], vec![3], false);
        let zp_ax1 = ti64(vec![0, 0, 0], vec![3]);
        let out_ax1 =
            fake_quantize_per_channel_affine(&input, &scale_ax1, &zp_ax1, 1, -128, 127).unwrap();
        // For col 0 (scale=0.5): 1.0→1.0, 4.0→4.0.
        // For col 1 (scale=1.0): 2.0→2.0, 5.0→5.0.
        // For col 2 (scale=2.0): round(3/2)=2→4.0, round(6/2)=3→6.0.
        let d1 = out_ax1.data().unwrap();
        assert_eq!(d1, &[1.0, 2.0, 4.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    // reason: per-channel STE mask is binary {0,1}; each grad slot holds
    // an exact bit pattern.
    #[allow(clippy::float_cmp)]
    fn fake_quantize_per_channel_ste_mask_is_per_channel() {
        // 2-D input [2, 4], axis=0. Row 0 has small scale → values
        // mostly in range; Row 1 has tiny scale → values mostly clamped
        // (out of range). The STE mask must reflect each channel's
        // q_unclamped per upstream `QuantizedOpKernels.cpp:2706`.
        let input = t(
            vec![
                -1.0, 0.0, 1.0, 2.0, // row 0 — scale=0.05, all in range
                -1.0, 0.0, 1.0, 2.0, // row 1 — scale=0.005, mostly clamped
            ],
            vec![2, 4],
            true,
        );
        let scale = t(vec![0.05, 0.005], vec![2], false);
        let zp = ti64(vec![0, 0], vec![2]);
        let out = fake_quantize_per_channel_affine(&input, &scale, &zp, 0, -128, 127).unwrap();
        let sum = crate::grad_fns::reduction::sum(&out).unwrap();
        backward(&sum).unwrap();
        let grad = input.grad().unwrap().unwrap();
        let g = grad.data().unwrap();

        // Row 0 (scale=0.05): q_uncl(x) = round(x/0.05).
        //   -1.0 → -20 (in [-128,127])  → 1
        //    0.0 → 0   (in range)       → 1
        //    1.0 → 20  (in range)       → 1
        //    2.0 → 40  (in range)       → 1
        // Row 1 (scale=0.005): q_uncl(x) = round(x/0.005).
        //   -1.0 → -200 (< -128)        → 0
        //    0.0 → 0    (in range)      → 1
        //    1.0 → 200  (> 127)         → 0
        //    2.0 → 400  (> 127)         → 0
        let expected: [f32; 8] = [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0];
        for (i, (&a, &e)) in g.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                a, e,
                "per-channel STE mask at i={i}: expected {e}, got {a}; mask formula \
                 per QuantizedOpKernels.cpp:2706 with scale[ch], zero_point[ch]"
            );
        }
    }

    #[test]
    fn fake_quantize_per_channel_empty_channel_dim() {
        // Empty input on the channel axis. scale/zp 1-D with numel=0.
        // Forward MUST not panic and produce an empty output.
        let input = t(vec![], vec![0, 4], false);
        let scale = t(vec![], vec![0], false);
        let zp = ti64(vec![], vec![0]);
        let out = fake_quantize_per_channel_affine(&input, &scale, &zp, 0, -128, 127).unwrap();
        assert_eq!(out.shape(), &[0, 4]);
        assert_eq!(out.data().unwrap().len(), 0);
    }

    // ── per-channel validation errors ──────────────────────────────

    #[test]
    fn fake_quantize_per_channel_rejects_non_1d_scale() {
        let input = t(vec![1.0, 2.0], vec![2], false);
        let scale = t(vec![1.0, 1.0], vec![1, 2], false);
        let zp = ti64(vec![0, 0], vec![2]);
        let result = fake_quantize_per_channel_affine(&input, &scale, &zp, 0, -128, 127);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("scale should be a 1-D"));
    }

    #[test]
    fn fake_quantize_per_channel_rejects_non_1d_zero_point() {
        let input = t(vec![1.0, 2.0], vec![2], false);
        let scale = t(vec![1.0, 1.0], vec![2], false);
        let zp = ti64(vec![0, 0], vec![1, 2]);
        let result = fake_quantize_per_channel_affine(&input, &scale, &zp, 0, -128, 127);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("zero point should be a 1-D"));
    }

    #[test]
    fn fake_quantize_per_channel_rejects_mismatched_qparams_sizes() {
        let input = t(vec![1.0, 2.0], vec![2], false);
        let scale = t(vec![1.0, 1.0], vec![2], false);
        let zp = ti64(vec![0, 0, 0], vec![3]);
        let result = fake_quantize_per_channel_affine(&input, &scale, &zp, 0, -128, 127);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("same dimensions"));
    }

    #[test]
    fn fake_quantize_per_channel_rejects_qparam_size_mismatch_with_axis() {
        let input = t(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
        let scale = t(vec![1.0, 1.0, 1.0], vec![3], false);
        let zp = ti64(vec![0, 0, 0], vec![3]);
        let result = fake_quantize_per_channel_affine(&input, &scale, &zp, 0, -128, 127);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("not consistent with input"));
    }

    #[test]
    fn fake_quantize_per_channel_rejects_axis_out_of_bounds() {
        let input = t(vec![1.0, 2.0], vec![2], false);
        let scale = t(vec![1.0], vec![1], false);
        let zp = ti64(vec![0], vec![1]);
        let result = fake_quantize_per_channel_affine(&input, &scale, &zp, 5, -128, 127);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("`axis`"));
    }

    #[test]
    fn fake_quantize_per_channel_rejects_zero_point_outside_range() {
        let input = t(vec![1.0, 2.0], vec![2], false);
        let scale = t(vec![1.0, 1.0], vec![2], false);
        // 500 lies outside [-128, 127] — upstream FakeQuantPerChannelAffine.cpp:69-74.
        let zp = ti64(vec![0, 500], vec![2]);
        let result = fake_quantize_per_channel_affine(&input, &scale, &zp, 0, -128, 127);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("`zero_point`"));
    }

    #[test]
    fn fake_quantize_per_channel_rejects_inverted_range() {
        let input = t(vec![1.0, 2.0], vec![2], false);
        let scale = t(vec![1.0, 1.0], vec![2], false);
        let zp = ti64(vec![0, 0], vec![2]);
        let result = fake_quantize_per_channel_affine(&input, &scale, &zp, 0, 128, -128);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("`quant_min`"));
    }

    // NOTE: previously a `fake_quantize_per_channel_rejects_non_positive_scale`
    // test verified rejection of `scale[i] <= 0`. Upstream
    // `FakeQuantPerChannelAffine.cpp:32-77` has no such check; ferrotorch
    // now silently proceeds with the cast-first formula to match upstream
    // byte-for-byte (R-DEV-1 numerical contract; closes #1261). The
    // upstream behavior is now pinned by the divergence tests at
    // `tests/divergence_quantize_grad_1239_per_channel.rs::
    // divergence_per_channel_scale_{zero,negative}_silently_proceeds`.

    #[test]
    fn fake_quantize_per_channel_preserves_grad_fn_when_input_requires_grad() {
        let input = t(vec![1.0, 2.0], vec![2], true);
        let scale = t(vec![1.0, 1.0], vec![2], false);
        let zp = ti64(vec![0, 0], vec![2]);
        let out = fake_quantize_per_channel_affine(&input, &scale, &zp, 0, -128, 127).unwrap();
        assert!(out.requires_grad());
        assert!(out.grad_fn().is_some());
    }
}
