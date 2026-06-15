# Quantize grad_fns

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158 (Revert "[dynamo] Implement nb_or/nb_inplace_or slot dispatch for | and |= operators (#181326)")
upstream-paths:
  - aten/src/ATen/native/quantized/FakeQuantPerTensorAffine.cpp
  - aten/src/ATen/native/quantized/FakeQuantPerChannelAffine.cpp
  - aten/src/ATen/native/quantized/FakeQuantAffine.h
  - torch/ao/quantization/fake_quantize.py
  - torch/_torch_docs.py
  - torch/overrides.py
  - tools/autograd/derivatives.yaml
-->

## Summary

`ferrotorch-core/src/grad_fns/quantize_grad.rs` is the autograd-tracking layer
for differentiable fake quantization (quantization-aware training / QAT). It
mirrors the per-tensor side of `aten/src/ATen/native/quantized/
FakeQuantPerTensorAffine.cpp` (the user-facing `torch.fake_quantize_per_tensor_affine`
documented at `torch/_torch_docs.py:11950-11988` and registered in
`torch/overrides.py:622 torch.fake_quantize_per_tensor_affine: lambda input,
scale, zero_point, quant_min, quant_max: -1`). The forward computes
`dequantize(round(input/scale + zero_point).clamp(quant_min, quant_max))` and
the backward applies the clipped straight-through estimator (STE): gradient
flows through 1:1 for values whose pre-quantized representation lies inside
`[quant_min, quant_max]` and zero otherwise. The per-channel sibling
`fake_quantize_per_channel_affine`
(`aten/src/ATen/native/quantized/FakeQuantPerChannelAffine.cpp:32-42`) is in
scope for this route but is not yet implemented in the file under design.

The file is 349 LOC (149 production + 200 `#[cfg(test)]`), exporting a single
`pub fn fake_quantize_differentiable<T: Float>(input, scale, zero_point, qmin,
qmax)` + a private `FakeQuantizeBackward<T>` grad-fn struct.

## Requirements

- REQ-1: `fake_quantize_per_tensor_affine(input, scale, zero_point,
  quant_min, quant_max)` — forward computes
  `output = (clamp(round_ties_even(input/scale) + zero_point, quant_min,
  quant_max) - zero_point) * scale` per the formula at
  `torch/_torch_docs.py:11958-11967` and `aten/src/ATen/native/quantized/
  FakeQuantPerTensorAffine.cpp:31-40` (delegating to
  `fake_quantize_per_tensor_affine_cachemask`). Backward applies the clipped
  STE per `tools/autograd/derivatives.yaml:673-674
  - name: fake_quantize_per_tensor_affine_cachemask(Tensor self, float scale,
    int zero_point, int quant_min, int quant_max) -> (Tensor output, Tensor mask)
    self: fake_quantize_per_tensor_affine_cachemask_backward(grad, mask)`
  where `fake_quantize_per_tensor_affine_cachemask_backward` at
  `FakeQuantPerTensorAffine.cpp:121-134` is `dY * mask` and the mask is `1`
  for in-range values, `0` otherwise. Public signature MUST match upstream's
  `(input, scale: f64, zero_point: i64, quant_min: i64, quant_max: i64)`
  per `torch/overrides.py:622`.

  SHIPPED (#1238 closed). The canonical `fake_quantize_per_tensor_affine`
  at `ferrotorch-core/src/grad_fns/quantize_grad.rs:84` matches upstream
  byte-for-byte:
  * Function name: `fake_quantize_per_tensor_affine` matches
    `torch/overrides.py:622 torch.fake_quantize_per_tensor_affine: lambda
    input, scale, zero_point, quant_min, quant_max: -1` (R-DEV-2).
  * Argument names: `scale: f64, zero_point: i64, quant_min: i64,
    quant_max: i64` match upstream `double scale, int64_t zero_point,
    int64_t quant_min, int64_t quant_max` at
    `FakeQuantPerTensorAffine.cpp:33-36`.
  * Tensor-qparam overload: `fake_quantize_per_tensor_affine_tensor_qparams`
    at `quantize_grad.rs:149` mirrors `FakeQuantPerTensorAffine.cpp:42-51
    Tensor fake_quantize_per_tensor_affine(const Tensor& self, const Tensor&
    scale, const Tensor& zero_point, int64_t quant_min, int64_t quant_max)`,
    extracting the scalars via `scale_data[0]` / `zp_data[0]` analogously to
    upstream's `.item()` extraction at `QuantizedOpKernels.cpp:2737`.
  * Rounding: `f64::round_ties_even` (Rust 1.77+) matches upstream's
    `std::nearbyint` at `QuantizedOpKernels.cpp:2703` under FE_TONEAREST
    (round-half-to-even / banker's rounding). Locked by the unit test
    `fake_quantize_uses_banker_rounding_on_half_boundaries` at
    `quantize_grad.rs:419-435`.
  * NaN-safe clamp: `f64::min` / `f64::max` follow IEEE-754-2019
    minimum/maximum (return the non-NaN operand) matching upstream's
    `std::fmin` / `std::fmax` at `QuantizedOpKernels.cpp:2704`. Locked by
    `fake_quantize_nan_input_does_not_panic` at `quantize_grad.rs:437-449`.
  * `zero_point` in `[quant_min, quant_max]` check matches upstream
    `TORCH_CHECK` at `FakeQuantPerTensorAffine.cpp:79-81`; locked by
    `fake_quantize_rejects_zero_point_outside_quant_range` at
    `quantize_grad.rs:392-400`.

  Non-test production consumer: `Tensor::fake_quantize_per_tensor_affine_t`
  at `fake_quantize_per_tensor_affine_t in ferrotorch-core/src/methods.rs` is the chainable method-style
  surface, delegating to
  `grad_fns::quantize_grad::fake_quantize_per_tensor_affine`. This closes
  the R-DEFER-1 requirement (a method on `Tensor<T>` is a production-
  callable consumer, analogous to `Tensor::cumsum_t` at `cumsum_t in methods.rs`).

  The pre-#1238 name `fake_quantize_differentiable` is preserved at
  `quantize_grad.rs:104` as a back-compat thin delegator (i32 args cast
  to i64) to keep the existing conformance test
  `ferrotorch-core/tests/conformance_quantize_prune.rs:735-784` working
  without an i32/i64 widening churn on the deserialization side. New code
  must use the canonical name.

  Parity-sweep: `[fake_quantize_per_tensor_affine] 72/72 passed (0
  skipped, 0 failed)` at `--seeds 8`. Runner dispatch arm at
  `tools/parity-sweep/runner/src/main.rs:540-572`.

- REQ-2: `fake_quantize_per_channel_affine(input, scale, zero_point, axis,
  quant_min, quant_max)` — per-channel fake quantization that broadcasts a
  1-D `scale` / `zero_point` along `axis` per
  `aten/src/ATen/native/quantized/FakeQuantPerChannelAffine.cpp:32-42`
  (delegating to `fake_quantize_per_channel_affine_cachemask`). Backward
  per `tools/autograd/derivatives.yaml:666-683
  - name: fake_quantize_per_channel_affine_cachemask(Tensor self, Tensor scale,
    Tensor zero_point, int axis, int quant_min, int quant_max) -> (Tensor output,
    Tensor mask)
    self: fake_quantize_per_channel_affine_cachemask_backward(grad, mask)`
  where the mask is again the in-range indicator (same STE structure as
  REQ-1, just broadcasted along `axis`). Forward formula matches REQ-1
  formula at `torch/_torch_docs.py:11999-12008` byte-for-byte, with the
  scalar `scale` and `zero_point` replaced by a per-channel broadcast.

  SHIPPED (#1239 closed). The canonical
  `fake_quantize_per_channel_affine<T: Float>(input, scale: &Tensor<T>,
  zero_point: &IntTensor<i64>, axis: i64, quant_min: i64, quant_max: i64)
  -> FerrotorchResult<Tensor<T>>` at
  `ferrotorch-core/src/grad_fns/quantize_grad.rs:342` matches the upstream
  forward at `aten/src/ATen/native/quantized/FakeQuantPerChannelAffine.cpp:
  32-42 Tensor fake_quantize_per_channel_affine(const Tensor& self, const
  Tensor& scale, const Tensor& zero_point, int64_t axis, int64_t quant_min,
  int64_t quant_max)` byte-for-byte:
  * Per-element formula mirrors the per-channel CPU kernel at
    `QuantizedOpKernels.cpp:2836-2848` with `scale[ch]` / `zero_point[ch]`
    lookup for each element's channel index `ch = (i / inner) % channel_dim`
    where `inner = prod(shape[axis+1..])`. The per-channel kernel
    DIVERGES from the per-tensor kernel at `:2702-2706` in cast ordering:
    per-channel casts to `int64_t` BEFORE clamping
    (`static_cast<int64_t>(zp + std::nearbyint(self * inv_scale))` then
    `std::fmin(std::fmax(qval_i64, quant_min), quant_max)`), while
    per-tensor casts AFTER the clamp. This matters for non-finite inputs:
    for `+Inf` upstream's `static_cast<int64_t>(+Inf)` is undefined
    behaviour but in practice on x86-64 SSE2 saturates to `INT64_MIN`,
    which then `std::fmax(INT64_MIN, quant_min) = quant_min`, snapping
    the output to `(quant_min - zp) * scale`. ferrotorch replicates this
    R-DEV-1 byte-for-byte via the `per_channel_dequantize_f64` helper at
    `grad_fns/quantize_grad.rs` which explicitly maps non-finite
    qval_f to `i64::MIN` before the clamp. Locked by parity-sweep sample
    6 (`shape=[1, 4]`, input `[+Inf, -Inf, 0.0, 1.0]`): per-tensor would
    clamp +Inf to `quant_max=127`, but per-channel correctly produces
    `-128` matching upstream.
  * Validation matches upstream `:55-77` order: `scale.dim()==1`,
    `zero_point.dim()==1`, `scale.numel()==zero_point.numel()`,
    `quant_min<=quant_max`, `axis` in bounds, `scale.numel()==self.size(axis)`,
    `zero_point[i] in [quant_min, quant_max]`. Each rule has a dedicated
    rejection unit test.
  * Backward node `FakeQuantizePerChannelBackward<T>` at
    `FakeQuantizePerChannelBackward in ferrotorch-core/src/grad_fns/quantize_grad.rs` with
    `GradFn::backward` impl at `:665` returns `dY * mask` where the mask
    is per-channel: `mask[i] = (quant_min <= round_ties_even(input[i]/
    scale[ch(i)]) + zero_point[ch(i)] <= quant_max)`, matching upstream
    `FakeQuantPerChannelAffine.cpp:118-131 fake_quantize_per_channel_affine_
    cachemask_backward = dY * mask`. The struct saves `scale: Vec<f64>`,
    `zero_point: Vec<i64>`, `axis: usize` directly (no need to
    re-materialize the parameter tensors in backward; mirror of
    `FakeQuantizeBackward<T>`'s saved-scalars pattern).

  Non-test production consumer:
  `Tensor::fake_quantize_per_channel_affine_t(&self, scale, zero_point,
  axis, quant_min, quant_max)` at `ferrotorch-core/src/methods.rs:628` is
  the chainable method-style surface (analogous to
  `Tensor::fake_quantize_per_tensor_affine_t` at `fake_quantize_per_tensor_affine_t in methods.rs`)
  delegating to `grad_fns::quantize_grad::fake_quantize_per_channel_affine`.
  Closes the R-DEFER-1 vocabulary-only loophole.

  Parity-sweep: `[fake_quantize_per_channel_affine] 72/72 passed (0
  skipped, 0 failed)` at `--seeds 8`. Runner dispatch arm at
  `tools/parity-sweep/runner/src/main.rs::dispatch_f32` arm
  `"fake_quantize_per_channel_affine"` widens the int32 oracle-emitted
  zero_point tensor to `IntTensor<i64>` via the
  `WireTensor::to_int_tensor_i64` helper (matching upstream's
  `int32 | float | half` zero_point accepted at
  `FakeQuantPerChannelAffine.cpp:53` — ferrotorch's typed carrier
  always widens to i64).

  Unit-test coverage (12 new tests added under #1239, all R-CHAR-3
  compliant):
  * `fake_quantize_per_channel_matches_per_tensor_on_each_channel` —
    forward parity vs per-tensor on each row slice (the per-tensor
    surface itself traces back to upstream `QuantizedOpKernels.cpp:
    2702-2706`).
  * `fake_quantize_per_channel_axis_dispatch_differs` — axis=0 vs axis=1
    on the same input must produce per-axis-correct results.
  * `fake_quantize_per_channel_ste_mask_is_per_channel` — backward
    STE mask reflects each channel's `q_unclamped`, with per-channel
    quantum range producing different in/out-of-range patterns per row.
  * `fake_quantize_per_channel_empty_channel_dim` — degenerate
    `shape=[0, 4]` produces empty output without panic.
  * 7 validation-rejection tests + 1 grad-fn-attach test.

- REQ-3: Clipped STE backward — gradient is `dY * mask` where the mask is
  `1` for input values whose unclamped quantized representation `q =
  round(input/scale) + zero_point` satisfies `quant_min <= q <= quant_max`
  and `0` otherwise. This mirrors
  `aten/src/ATen/native/quantized/FakeQuantPerTensorAffine.cpp:121-134
  Tensor fake_quantize_per_tensor_affine_cachemask_backward(const Tensor& dY,
  const Tensor& mask) { ...; return dY * mask; }` and is the consumer that
  `derivatives.yaml:673-674` wires into autograd. The current
  `FakeQuantizeBackward` at `ferrotorch-core/src/grad_fns/quantize_grad.rs:801-863`
  implements an equivalent STE — it computes the range boundary
  `[dequantize(qmin), dequantize(qmax)] = [(qmin - zp) * scale, (qmax - zp) *
  scale]` and zeros gradient outside it — but the consumer chain ends at
  REQ-1's missing surface (the only entry point that attaches
  `FakeQuantizeBackward` is `fake_quantize_differentiable`, which itself
  has no non-test production consumer per REQ-1's analysis).

  SHIPPED (#1238 closed). The `FakeQuantizeBackward<T>` grad-function struct at
  `ferrotorch-core/src/grad_fns/quantize_grad.rs:727` with the
  `GradFn::backward` impl at `:735-788` mechanically implements the
  upstream `dY * mask` STE per `FakeQuantPerTensorAffine.cpp:121-134`,
  computing the mask via
  `mask = (quant_min <= round_ties_even(input/scale) + zero_point <= quant_max)`
  matching `QuantizedOpKernels.cpp:2706` byte-for-byte (and switching from
  the prior "dequantized boundary" formulation, which was approximately
  correct but diverged on `.5` boundaries because of Rust's
  round-half-away-from-zero default). Three independent unit tests lock
  the contract:
  `fake_quantize_ste_passes_grad_for_in_range_values` at
  `quantize_grad.rs:464-481` (all in-range inputs receive grad 1.0),
  `fake_quantize_ste_zeros_grad_for_out_of_range_values` at `:483-510`
  (out-of-range inputs receive grad 0.0), and
  `fake_quantize_ste_backward_matches_explicit_formula` at `:512-549`
  (each grad slot constructed explicitly from the upstream mask formula,
  not from calling the op on itself — R-CHAR-3 honored). The consumer
  chain now closes through REQ-1's
  `Tensor::fake_quantize_per_tensor_affine_t` at
  `fake_quantize_per_tensor_affine_t in ferrotorch-core/src/methods.rs`, which is the production-callable
  forward surface that attaches `FakeQuantizeBackward` and drives the
  STE.

## Acceptance Criteria

- [x] AC-1: `fake_quantize_per_tensor_affine` parity-sweep at `--seeds 8`
  returns `[fake_quantize_per_tensor_affine] 72/72 passed (0 skipped, 0
  failed)` with `grep -c "passed (0 skipped, 0 failed)" == 1`.
  Post-#1238 close, the runner-side dispatch arm at
  `tools/parity-sweep/runner/src/main.rs:540-572` routes the oracle
  samples through `grad_fns::quantize_grad::fake_quantize_per_tensor_affine`
  and the 9 hand-crafted samples × 8 seeds = 72 samples all pass.
- [x] AC-2: `fake_quantize_per_channel_affine` parity-sweep at `--seeds 8`
  returns `[fake_quantize_per_channel_affine] 72/72 passed (0 skipped, 0
  failed)` with `grep -c "passed (0 skipped, 0 failed)" == 1`.
  Post-#1239 close, the runner-side dispatch arm at
  `tools/parity-sweep/runner/src/main.rs::dispatch_f32` routes the
  oracle samples through
  `grad_fns::quantize_grad::fake_quantize_per_channel_affine` and the 9
  hand-crafted samples × 8 seeds = 72 samples all pass.
- [x] AC-3: `cargo test -p ferrotorch-core --lib grad_fns::quantize_grad`
  passes all 19 tests in the `#[cfg(test)] mod tests` block at
  `quantize_grad.rs:340-595`, including the new upstream-faithful coverage:
  `fake_quantize_round_trips_representable_values`,
  `fake_quantize_clamps_out_of_range_values`,
  `fake_quantize_rejects_zero_scale`,
  `fake_quantize_rejects_negative_scale`,
  `fake_quantize_rejects_inverted_range`,
  `fake_quantize_rejects_zero_point_outside_quant_range` (NEW —
  upstream `FakeQuantPerTensorAffine.cpp:79-81` check),
  `fake_quantize_asymmetric_with_zero_point`,
  `fake_quantize_uses_banker_rounding_on_half_boundaries` (NEW —
  upstream `std::nearbyint` parity on `.5` boundaries),
  `fake_quantize_nan_input_does_not_panic` (NEW — IEEE-754-2019
  min/max NaN propagation parity),
  `tensor_qparams_matches_scalar_qparams` (NEW — tensor-qparams
  overload byte-for-byte parity with scalar variant),
  `tensor_qparams_rejects_multi_element_scale` (NEW),
  `tensor_qparams_rejects_multi_element_zero_point` (NEW),
  `fake_quantize_ste_passes_grad_for_in_range_values`,
  `fake_quantize_ste_zeros_grad_for_out_of_range_values`,
  `fake_quantize_ste_backward_matches_explicit_formula` (NEW —
  R-CHAR-3-compliant formulaic STE backward verification),
  `fake_quantize_no_grad_when_input_doesnt_require_grad`,
  `fake_quantize_preserves_grad_fn_when_input_requires_grad`,
  `fake_quantize_no_grad_context_skips_grad_fn`,
  `fake_quantize_chains_through_autograd_with_relu`. These tests
  exercise the upstream-faithful canonical
  `fake_quantize_per_tensor_affine` signature; the pre-#1238
  `fake_quantize_differentiable` back-compat alias is exercised by the
  pre-existing `conformance_quantize_prune.rs` fixture test which
  continues to pass unchanged.
- [x] AC-4: The forward formula at `quantize_grad.rs:159-237` matches the
  documented PyTorch identity at `torch/_torch_docs.py:11958-11967` and
  the upstream CPU kernel at `aten/src/ATen/native/quantized/cpu/kernels/
  QuantizedOpKernels.cpp:2702-2706` byte-for-byte:
  `qval_f = z_point + std::nearbyint(input * inv_scale); qval =
  static_cast<int64_t>(std::fmin(std::fmax(qval_f, quant_min), quant_max));
  output = (qval - z_point) * sc; mask = (quant_min <= qval_f) && (qval_f
  <= quant_max)`. ferrotorch's `f64::round_ties_even` matches
  `std::nearbyint` (round-half-to-even / banker's rounding under
  FE_TONEAREST); `f64::min` / `f64::max` follow IEEE-754-2019
  minimum/maximum (return the non-NaN operand on NaN input) matching
  `std::fmin` / `std::fmax`. Locked by
  `fake_quantize_uses_banker_rounding_on_half_boundaries` at
  `quantize_grad.rs:419-435` and `fake_quantize_nan_input_does_not_panic`
  at `:437-449`.
- [x] AC-5: Non-test production consumer for the autograd-tracking fake
  quantize surface — `Tensor::fake_quantize_per_tensor_affine_t` at
  `fake_quantize_per_tensor_affine_t in ferrotorch-core/src/methods.rs` is the chainable method-style
  surface (analogous to `Tensor::cumsum_t` at `cumsum_t in methods.rs`) that
  invokes `grad_fns::quantize_grad::fake_quantize_per_tensor_affine`
  outside any `#[cfg(test)]` block. Closes the R-DEFER-1 vocabulary-only
  loophole that REQ-1 was previously stuck behind.
- [x] AC-6: STE backward node correctness — `FakeQuantizeBackward` at
  `FakeQuantizeBackward in quantize_grad.rs` with `backward` impl at `backward in quantize_grad.rs` returns
  `grad_output * 1` for in-range values and `grad_output * 0` otherwise,
  matching upstream's `dY * mask` at `FakeQuantPerTensorAffine.cpp:133`.
  Verified by `fake_quantize_ste_passes_grad_for_in_range_values` at
  `quantize_grad.rs:464-481`,
  `fake_quantize_ste_zeros_grad_for_out_of_range_values` at `:483-510`,
  and the new R-CHAR-3-compliant formulaic check
  `fake_quantize_ste_backward_matches_explicit_formula` at `:512-549`
  whose expected gradient is constructed bit-for-bit from the upstream
  mask formula at `QuantizedOpKernels.cpp:2706`, never from calling the
  op on itself.

## Architecture

### Layer split (`quantize_grad` vs `quantize`)

There are two distinct quantization layers in `ferrotorch-core`:

1. `ferrotorch-core/src/quantize.rs` (1700+ LOC): non-autograd, slice-based
   primitives (`FakeQuantize::forward(weights: &[f32]) -> (Vec<f32>,
   Vec<bool>)`), with the higher-level `QatModel`, `prepare_qat`, and
   per-tensor / per-channel utilities that operate on raw f32 slices. This
   is consumed by `ferrotorch-nn` (`QatModel.fake_quantize_weights` is the
   ferrotorch-nn-facing call surface; tested in
   `ferrotorch-nn/tests/conformance_nn_structural.rs:1651-1677`).
2. `ferrotorch-core/src/grad_fns/quantize_grad.rs` (349 LOC, the file
   under design): the autograd-tracking `Tensor<T>` surface that wraps a
   `FakeQuantizeBackward` grad-fn. **This file's surface has no in-tree
   non-test consumer.** The QAT path in (1) does its own thing on slices
   and does not flow gradients through autograd; this file's autograd
   path was added per CL-293 (`CHANGELOG.md:650`) but never wired into a
   downstream caller.

The split is structurally legitimate (slice-API vs tensor-API), but means
the autograd half stalls at vocabulary until a Tensor-API consumer
materializes. PyTorch upstream's analog is unified:
`torch.fake_quantize_per_tensor_affine(Tensor input, ...) -> Tensor` is
both autograd-aware and the only public surface, with the `QuantStub` /
`FakeQuantize` Python module
(`torch/ao/quantization/fake_quantize.py:244-259`) calling it on every
`forward()`. The ferrotorch analog of that call chain — a `Tensor`-API
`fake_quantize_per_tensor_affine` invoked from a `ferrotorch-nn` QAT
module's tensor-shaped forward pass — does not yet exist.

### Current forward (lines 51-110)

`pub fn fake_quantize_differentiable<T: Float>(input, scale, zero_point,
qmin, qmax)` at `quantize_grad.rs:51-57` validates `scale > 0` and `qmin
< qmax`, then iterates over `input.data_vec()` computing
`dequantize(clamp(round(input/scale + zp), qmin, qmax))` elementwise.
When `input.requires_grad() && is_grad_enabled()`, it attaches a
`FakeQuantizeBackward` saving `input` (for the STE mask check) and the
pre-computed dequantized range boundaries
`range_min = (qmin - zp) * scale`, `range_max = (qmax - zp) * scale`
(`quantize_grad.rs:77-78`). The `Tensor::from_operation` /
`Tensor::from_storage` branching at `:106-109` follows the standard
grad-fn-attach pattern.

Divergences from the upstream contract this file claims to mirror:
* `scale: f64` matches upstream `double scale` at
  `FakeQuantPerTensorAffine.cpp:33`, but `zero_point: i32` widens upstream
  `int64_t zero_point` at `:34` only partially — the i32 cast loses range
  on large int64 zero-points, which in practice does not arise (zero_point
  is always in `[quant_min, quant_max] = [-128, 127]` or `[0, 255]`).
* `qmin / qmax: i32` similarly truncates upstream `int64_t quant_min /
  quant_max`. Same practical irrelevance, same vocabulary-level divergence
  (R-DEV-2 API-shape match).
* No tensor-qparam overload — upstream
  `FakeQuantPerTensorAffine.cpp:42-51 Tensor fake_quantize_per_tensor_affine(
  const Tensor& self, const Tensor& scale, const Tensor& zero_point, ...)`
  is missing.

### Current backward (lines 113-151)

`FakeQuantizeBackward<T>` saves `input: Tensor<T>` (a clone, refcounted),
`range_min: T`, `range_max: T`. The `backward(&self, grad_output)` impl
at `:121-142` materializes `input` and `grad_output` data, then computes
`grad[i] = if range_min <= input[i] <= range_max { grad_output[i] } else
{ 0 }`. This is the clipped STE.

Equivalence with upstream's mask-based VJP: upstream stores the BoolTensor
mask in the forward pass
(`fake_quantize_per_tensor_affine_cachemask` at
`FakeQuantPerTensorAffine.cpp:69-90` returns `(output, mask)`) and the
backward at `:121-134` is literally `dY * mask`. ferrotorch instead saves
the input and recomputes the mask in the backward via the
`range_min/range_max` boundary check. The numerical result is identical
(both produce `grad * 1` for in-range, `grad * 0` for out-of-range), but
the memory profile differs: upstream allocates a bool mask once in
forward (1 byte/element; the `TODO(future, optional): packing the mask
further` at `FakeQuantPerTensorAffine.cpp:87` notes this could be 1
bit/element), while ferrotorch re-reads the input tensor in backward
(4-or-8 bytes/element, but the input is already refcounted in the
graph, so no extra allocation). Both are valid implementation choices
for the same VJP; ferrotorch's is the input-saved variant analogous to
what `_fake_quantize_learnable_per_tensor_affine_backward` at
`FakeQuantPerTensorAffine.cpp:161-235` does (it also re-reads `X` in
backward rather than threading a mask).

The `GradFn::name()` returns `"FakeQuantizeBackward"` (`:148-150`),
matching the upstream `grad_fn=<FakeQuantizePerTensorAffineCachemaskBackward>`
print only by abbreviation. No JIT tracer / `scalar_args()` exposure of
the saved `range_min / range_max`.

### Per-channel (REQ-2)

No code in the file. Upstream's per-channel variant at
`FakeQuantPerChannelAffine.cpp:32-107` differs structurally from
per-tensor:
* `scale: Tensor` (1-D, `numel() == self.size(axis)`),
* `zero_point: Tensor` (1-D, same numel),
* `axis: int64_t` indicating the broadcast axis,
* The TensorIterator reshapes `scale` and `zero_point` into the per-axis
  shape (`expected_shape[axis] = self.size(axis)`, all others 1) via
  `_unsafe_view(scale, expected_shape)` at `:89-90`.

Reusing the per-tensor scalar path with per-channel scalar broadcasts is
not a sufficient implementation — the autograd VJP also needs to be
broadcast-aware so the gradient flows back to the per-channel scale /
zero_point if learnable; the non-learnable variant
(`fake_quantize_per_channel_affine_cachemask`,
`derivatives.yaml:666-683`) gradients only the input via `dY * mask`,
with mask broadcast along `axis`.

### Validation + error paths (lines 58-68)

`fake_quantize_differentiable` returns `FerrotorchError::InvalidArgument`
when `scale.is_nan() || scale <= 0.0` (`:59-63`) and when `qmin >= qmax`
(`:64-68`). Upstream's equivalents:
* `quant_min <= quant_max` check at `FakeQuantPerTensorAffine.cpp:75-78`
  (raises `RuntimeError` — ferrotorch's `Result::Err` is the R-DEV-4
  Result-vs-raise vocabulary substitution).
* `zero_point >= quant_min && zero_point <= quant_max` check at
  `FakeQuantPerTensorAffine.cpp:79-81` — **DIVERGES**: ferrotorch does
  not check that `zero_point` lies in the quantization range. With a
  bad zero_point, the round-then-clamp formula still produces SOME
  result (the clamp covers it), but upstream rejects it explicitly.
  Flagged here for the REQ-1 re-implementation.
* `scale > 0` check at `FakeQuantPerTensorAffine.cpp` is NOT explicit
  upstream; ferrotorch's explicit `scale > 0` check is a strict
  superset of upstream behavior (R-DEV-1 numerical contract: dividing
  by zero would produce inf/NaN that propagates, which is what
  upstream does; ferrotorch's pre-check is friendlier but not
  upstream-byte-faithful).

## Parity contract

| Op | Upstream entry | Backward formula source | Expected behavior on edge cases |
|---|---|---|---|
| `fake_quantize_per_tensor_affine` | `aten/src/ATen/native/quantized/FakeQuantPerTensorAffine.cpp:31-40 Tensor fake_quantize_per_tensor_affine(const Tensor& self, double scale, int64_t zero_point, int64_t quant_min, int64_t quant_max)` (scalar-qparams overload) and `:42-51` (tensor-qparams overload) | `tools/autograd/derivatives.yaml:673-674` (`fake_quantize_per_tensor_affine_cachemask_backward = dY * mask`) | NaN input: `(NaN / scale).round() = NaN`, then `clamp(NaN, qmin, qmax)` is implementation-defined under IEEE-754 — Rust's `f32::clamp` panics on NaN (`debug_assert!(!self.is_nan())`) whereas C `std::min/std::max` on NaN returns the non-NaN operand → output is NaN-poisoned in upstream but undefined in ferrotorch. Inf input: `(inf / scale).round() = inf` → clamp to qmax. Denormals: round-to-nearest may flush; both languages match here. Empty input: `numel() == 0` → upstream returns empty (`FakeQuantPerTensorAffine.cpp:128-130 if (dY.sym_numel() <= 0) { return dY; }`); ferrotorch's `data_vec()` iteration on an empty vec produces an empty output naturally. Non-contiguous: ferrotorch's `input.data_vec()` materializes contiguously, then writes contiguously — same numerical result as upstream's TensorIterator-based dispatch but lossy on the storage-layout side. Dtype promotion: upstream requires `self.scalar_type() == ScalarType::Float` (f32 only, no f64/bf16); ferrotorch generic `T: Float` admits f32 / f64 / bf16 / f16 — a strict super-set, which is a deliberate R-DEV-7 deviation but should be documented when REQ-1 lands. **Status: NOT-STARTED (oracle missing per #1240; signature mismatch per #1238).** |
| `fake_quantize_per_channel_affine` | `aten/src/ATen/native/quantized/FakeQuantPerChannelAffine.cpp:32-42 Tensor fake_quantize_per_channel_affine(const Tensor& self, const Tensor& scale, const Tensor& zero_point, int64_t axis, int64_t quant_min, int64_t quant_max)` | `tools/autograd/derivatives.yaml:666-683` (`fake_quantize_per_channel_affine_cachemask_backward = dY * mask`, mask broadcast along `axis`) | Same elementwise NaN / Inf / denormal / empty cases as per-tensor, plus: `scale.dim() == 1` enforced at `FakeQuantPerChannelAffine.cpp:55`, `zero_point.dim() == 1` at `:56`, `scale.numel() == self.size(axis)` at `:61`. Axis out-of-bounds: `axis >= 0 && axis <= self.dim()` at `:76` — note the `<=` is upstream's actual contract (axis-on-the-trailing-dim is permitted for a degenerate broadcast); ferrotorch should match. Zero-point dtype: upstream accepts `kInt`, `kFloat`, `kHalf` for zero_point with the float types triggering a `_get_rounded_zero_point` round-then-clamp at `:133-139`; ferrotorch's int-only zero_point sidesteps this. **Status: NOT-STARTED (impl missing per #1239; oracle missing per #1240).** |

Parity-sweep audit reference: BOTH ops are **MISSING** from
`tools/parity-sweep/parity_audit.json`. The PyTorch oracle previously
did not expose `fake_quantize_per_tensor_affine` or
`fake_quantize_per_channel_affine` via
`torch.testing._internal.opinfo.op_db` — these are not standard op_db
entries (op_db is the unit-test op set; quantization ops live under a
separate `torch.testing._internal.quantization` harness).

**As of #1240's close, the oracle gap is closed.** A custom-op registry
in `tools/parity-sweep/oracle.py` (the `_CUSTOM_OPS` dict and
`_fake_quantize_per_tensor_affine_samples` /
`_fake_quantize_per_channel_affine_samples` generators) hand-crafts 9
samples per op covering 1-D / 2-D / 3-D / 4-D shapes, various scales
(0.001, 0.01, 0.05, 0.1, 1.0, 10.0, 100.0), zero_points across the
representable int8/uint8 range, both int8 (-128/127) and uint8 (0/255)
quant ranges, and edge inputs (zeros, exact-multiple-of-scale fixed
points, out-of-range clamps, +Inf/-Inf). The oracle's `sample`,
`execute`, and `list_ops` commands now route through `_CUSTOM_OPS` for
these op names; the runner-side `dispatch_f32` still returns
`Ok(None)` for these ops (handled as skips), so until #1238 (per-tensor
REQ-1 impl) and #1239 (per-channel REQ-2 impl) land, the sweeps report
`0/72 passed (72 skipped, 0 failed)` — no more `unknown op` errors.
Closing AC-1 / AC-2 requires the REQ-1 / REQ-2 implementations to add
runner-side dispatch arms.

## Verification

### Existing unit tests (all passing)

Located at `ferrotorch-core/src/grad_fns/quantize_grad.rs:153-348` (the
`#[cfg(test)] mod tests` block, 12 tests). Coverage:

Forward correctness (5 tests):
- `fake_quantize_round_trips_representable_values` (`:164-183`) — int8
  symmetric, scale 0.1, exact-multiple inputs round-trip to themselves.
- `fake_quantize_clamps_out_of_range_values` (`:185-203`) — int8
  symmetric, inputs `[-200, -100, 0, 100, 200]` clamp to `[-128, -100,
  0, 100, 127]`.
- `fake_quantize_rejects_zero_scale` (`:205-211`) — `scale=0.0` returns
  `Err` containing `"scale must be > 0"`.
- `fake_quantize_rejects_negative_scale` (`:213-218`) — `scale=-0.1`
  returns `Err`.
- `fake_quantize_rejects_inverted_range` (`:220-226`) — `qmin > qmax`
  returns `Err` containing `"qmin"`.
- `fake_quantize_asymmetric_with_zero_point` (`:228-239`) — uint8 with
  `zp=128` shifts the representable range to `[-128, 127]`.

Backward / STE (3 tests):
- `fake_quantize_ste_passes_grad_for_in_range_values` (`:243-269`) — all
  in-range inputs receive grad 1.0 through the backward.
- `fake_quantize_ste_zeros_grad_for_out_of_range_values` (`:271-297`) —
  out-of-range inputs receive grad 0.0, in-range receive 1.0.
- `fake_quantize_chains_through_autograd_with_relu` (`:325-348`) — the
  STE mask and the ReLU mask compose multiplicatively through autograd.

Graph integration (3 tests):
- `fake_quantize_no_grad_when_input_doesnt_require_grad` (`:299-305`)
- `fake_quantize_preserves_grad_fn_when_input_requires_grad` (`:307-313`)
- `fake_quantize_no_grad_context_skips_grad_fn` (`:315-323`)

Post-#1238 close (this iter) these tests now exercise the canonical
`fake_quantize_per_tensor_affine` signature (i64 args, banker's rounding,
NaN-safe clamp) plus the tensor-qparams overload, with 7 NEW unit tests
covering banker-rounding parity on `.5` boundaries, NaN-safe clamping,
the `zero_point` in-range check, the tensor-qparams equivalence,
multi-element-qparams rejection, and an explicit-formula STE backward
check (R-CHAR-3 compliant). The pre-#1238 private surface
`fake_quantize_differentiable` is retained as a thin back-compat
delegator and exercised by the unchanged
`fn fake_quantize_differentiable_forward_and_ste_backward` fixture
test in `ferrotorch-core/tests/conformance_quantize_prune.rs`.

### Parity-sweep status

Post-#1238 close, the runner-side dispatch arm at
`tools/parity-sweep/runner/src/main.rs:540-572` routes the oracle
samples through `grad_fns::quantize_grad::fake_quantize_per_tensor_affine`
and the sweep returns full parity:

```
$ ./target/release/parity-sweep sweep --op fake_quantize_per_tensor_affine --seeds 8
  [fake_quantize_per_tensor_affine] 72/72 passed (0 skipped, 0 failed)

$ ./target/release/parity-sweep sweep --op fake_quantize_per_channel_affine --seeds 8
  [fake_quantize_per_channel_affine] 0/72 passed (72 skipped, 0 failed)
```

Smoke grep count (`grep -c "passed (0 skipped, 0 failed)"`) is `1` for
`fake_quantize_per_tensor_affine` (post-#1238 close) and `0` for
`fake_quantize_per_channel_affine` (still pending #1239 because skips
remain). Closing AC-2 now requires only the per-channel impl (#1239) —
the oracle dependency #1240 is resolved and AC-1 / REQ-1 is SHIPPED.

Note on the kernel layer: there is no `ops/quantize_grad.rs` analogous
to `ops/cumulative.rs`. The forward and backward both live entirely
inside `grad_fns/quantize_grad.rs` because the forward is a simple
elementwise loop and the backward is a binary mask multiply — neither
warrants a separate kernel-layer split.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (per-tensor) | SHIPPED | Canonical `fake_quantize_per_tensor_affine<T: Float>(input, scale: f64, zero_point: i64, quant_min: i64, quant_max: i64)` at `fake_quantize_per_tensor_affine in ferrotorch-core/src/grad_fns/quantize_grad.rs` matches `torch/overrides.py:622 torch.fake_quantize_per_tensor_affine: lambda input, scale, zero_point, quant_min, quant_max: -1` and the upstream forward at `aten/src/ATen/native/quantized/FakeQuantPerTensorAffine.cpp:31-40` byte-for-byte (banker's rounding via `f64::round_ties_even` matching `std::nearbyint` at `QuantizedOpKernels.cpp:2703`; NaN-safe clamp via `f64::min` / `f64::max` matching `std::fmin` / `std::fmax` at `:2704`; `zero_point` in `[quant_min, quant_max]` check matching `FakeQuantPerTensorAffine.cpp:79-81`). Tensor-qparams overload `fake_quantize_per_tensor_affine_tensor_qparams<T: Float>(input, scale: &Tensor<T>, zero_point: &IntTensor<i64>, quant_min: i64, quant_max: i64)` at `fake_quantize_per_tensor_affine_tensor_qparams in ferrotorch-core/src/grad_fns/quantize_grad.rs` mirrors `FakeQuantPerTensorAffine.cpp:42-51`. Non-test production consumer: `Tensor::fake_quantize_per_tensor_affine_t(&self, scale, zero_point, quant_min, quant_max)` at `fake_quantize_per_tensor_affine_t in ferrotorch-core/src/methods.rs` — chainable method-style surface, analogous to `Tensor::cumsum_t` at `cumsum_t in methods.rs`. The pre-#1238 private surface `fake_quantize_differentiable in ferrotorch-core/src/grad_fns/quantize_grad.rs` is preserved as a thin back-compat delegator (casts i32 args to i64 and forwards to the canonical impl); it exists solely so the pre-existing `fn fake_quantize_differentiable_forward_and_ste_backward` fixture test in `ferrotorch-core/tests/conformance_quantize_prune.rs` continues to pass without a downstream churn cascade. Parity-sweep: `[fake_quantize_per_tensor_affine] 72/72 passed (0 skipped, 0 failed)` at `--seeds 8`. Runner dispatch arm: `ferrotorch in tools/parity-sweep/runner/src/main.rs`. |
| REQ-2 (per-channel) | SHIPPED | `fake_quantize_per_channel_affine<T: Float>(input, scale: &Tensor<T>, zero_point: &IntTensor<i64>, axis: i64, quant_min: i64, quant_max: i64)` at `fake_quantize_per_channel_affine in ferrotorch-core/src/grad_fns/quantize_grad.rs` matches the upstream forward at `aten/src/ATen/native/quantized/FakeQuantPerChannelAffine.cpp:32-42`. Per-element kernel mirrors upstream's per-channel CPU kernel at `QuantizedOpKernels.cpp:2836-2848` — note this DIVERGES from the per-tensor kernel at `:2702-2706` in cast ordering: per-channel casts to `int64_t` BEFORE the clamp (`static_cast<int64_t>(zp + std::nearbyint(self * inv_scale))` then `std::fmin(std::fmax(qval_i64, quant_min), quant_max)`). For non-finite qval_f the C `static_cast<int64_t>(+Inf)` is undefined behaviour but on x86-64 SSE2 saturates to `INT64_MIN`, snapping the output to `(quant_min - zp) * scale`. ferrotorch replicates this R-DEV-1 byte-for-byte via the `per_channel_dequantize_f64` helper at `per_channel_dequantize_f64 in grad_fns/quantize_grad.rs` which explicitly maps non-finite qval_f to `i64::MIN` before clamping. Validation order mirrors upstream `per_channel_dequantize_f64 in grad_fns/quantize_grad.rs`. Backward node `FakeQuantizePerChannelBackward<T>` at `FakeQuantizePerChannelBackward in grad_fns/quantize_grad.rs` returns `dY * mask` per `FakeQuantPerChannelAffine.cpp:118-131` with the per-channel mask sharing the cast-first ordering via `per_channel_mask_in_range in grad_fns/quantize_grad.rs`. Non-test production consumer: `Tensor::fake_quantize_per_channel_affine_t(&self, scale, zero_point, axis, quant_min, quant_max)` at `fake_quantize_per_channel_affine_t in ferrotorch-core/src/methods.rs` — chainable method-style surface analogous to `Tensor::fake_quantize_per_tensor_affine_t` at `fake_quantize_per_tensor_affine_t in methods.rs`. Parity-sweep: `[fake_quantize_per_channel_affine] 72/72 passed (0 skipped, 0 failed)` at `--seeds 8` (sample 6 `[+Inf, -Inf, 0.0, 1.0]` locks the cast-first ordering: per-channel maps +Inf to `quant_min`, not `quant_max`). Runner dispatch arm at `tools/parity-sweep/runner/src/main.rs::dispatch_f32` widens the oracle-emitted int32 zero_point to `IntTensor<i64>` via a new `WireTensor::to_int_tensor_i64` helper (matching upstream's `int32 | float | half` zero_point at `FakeQuantPerChannelAffine.cpp:53`; ferrotorch's typed carrier widens to i64). Twelve new R-CHAR-3-compliant unit tests cover forward parity-vs-per-tensor on each channel, axis dispatch, per-channel STE mask, empty channel dim, and 8 validation rejections. |
| REQ-3 (STE backward) | SHIPPED | The `FakeQuantizeBackward<T>` grad-function struct at `FakeQuantizeBackward in ferrotorch-core/src/grad_fns/quantize_grad.rs` with `GradFn::backward` impl at `backward in ferrotorch-core/src/grad_fns/quantize_grad.rs` returns `dY * mask` where `mask = (quant_min <= round_ties_even(input/scale) + zero_point <= quant_max)` matching upstream `QuantizedOpKernels.cpp:2706` and `FakeQuantPerTensorAffine.cpp:121-134` byte-for-byte. Verified by `fake_quantize_ste_passes_grad_for_in_range_values in quantize_grad.rs`, `fake_quantize_ste_zeros_grad_for_out_of_range_values in quantize_grad.rs`, and `fake_quantize_ste_backward_matches_explicit_formula in quantize_grad.rs` (R-CHAR-3-compliant: expected grad constructed bit-for-bit from the upstream mask formula at `QuantizedOpKernels.cpp:2706`, never from calling the op on itself). Non-test production consumer: REQ-1's consumer chain — `Tensor::fake_quantize_per_tensor_affine_t` at `fake_quantize_per_tensor_affine_t in ferrotorch-core/src/methods.rs` attaches `FakeQuantizeBackward` whenever its input requires grad, driving the STE through real autograd backward. |
