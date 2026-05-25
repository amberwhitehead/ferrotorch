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
  `output = (clamp(round(input/scale) + zero_point, quant_min, quant_max) -
  zero_point) * scale` per the formula at
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

  NOT-STARTED. The current `fake_quantize_differentiable` at
  `ferrotorch-core/src/grad_fns/quantize_grad.rs:51-110` is the closest
  in-tree analog but DIVERGES on:
  * Function name: `fake_quantize_differentiable` vs upstream
    `fake_quantize_per_tensor_affine` (R-DEV-2 Python user-API ABI mismatch).
  * Argument names: `qmin: i32 / qmax: i32` vs upstream `quant_min: i64 /
    quant_max: i64`.
  * Argument types: `zero_point: i32` vs upstream `zero_point: i64`.
  * Tensor-qparam overload: upstream additionally accepts
    `scale: Tensor / zero_point: Tensor` (`FakeQuantPerTensorAffine.cpp:42-51`
    delegating to `_fake_quantize_per_tensor_affine_cachemask_tensor_qparams`);
    ferrotorch admits scalar qparams only.
  * No non-test production consumer in any `ferrotorch-*/src/` file. The
    sole reference outside `quantize_grad.rs` is the `pub use
    grad_fns::quantize_grad::fake_quantize_differentiable;` re-export at
    `ferrotorch-core/src/lib.rs:163`, which is vocabulary-only and does
    NOT satisfy R-DEFER-1 (a `pub use` is API exposure, not a call site).
    `QatModel::fake_quantize_weights` at
    `ferrotorch-core/src/quantize.rs:1012-1031` is a separate, non-autograd
    primitive on `&[f32]` slices that delegates to a `FakeQuantize::forward`
    struct in `quantize.rs` and does NOT call into this file's
    `fake_quantize_differentiable`.
  Open prereq blocker: #1238.

- REQ-2: `fake_quantize_per_channel_affine(input, scale, zero_point, axis,
  quant_min, quant_max)` — per-channel fake quantization that broadcasts a
  1-D `scale` / `zero_point` along `axis` per
  `aten/src/ATen/native/quantized/FakeQuantPerChannelAffine.cpp:32-42`
  (delegating to `fake_quantize_per_channel_affine_cachemask`). Backward
  per `tools/autograd/derivatives.yaml:682-683
  - name: fake_quantize_per_channel_affine_cachemask(Tensor self, Tensor scale,
    Tensor zero_point, int axis, int quant_min, int quant_max) -> (Tensor output,
    Tensor mask)
    self: fake_quantize_per_channel_affine_cachemask_backward(grad, mask)`
  where the mask is again the in-range indicator (same STE structure as
  REQ-1, just broadcasted along `axis`). Forward formula matches REQ-1
  formula at `torch/_torch_docs.py:11999-12008` byte-for-byte, with the
  scalar `scale` and `zero_point` replaced by a per-channel broadcast.

  NOT-STARTED. No `fake_quantize_per_channel_affine` function exists in
  `ferrotorch-core/src/grad_fns/quantize_grad.rs` (verified by
  `grep -n 'fake_quantize_per_channel\|per_channel_affine' quantize_grad.rs`
  returning empty). The file currently ships only the per-tensor variant.
  Open prereq blocker: #1239.

- REQ-3: Clipped STE backward — gradient is `dY * mask` where the mask is
  `1` for input values whose unclamped quantized representation `q =
  round(input/scale) + zero_point` satisfies `quant_min <= q <= quant_max`
  and `0` otherwise. This mirrors
  `aten/src/ATen/native/quantized/FakeQuantPerTensorAffine.cpp:121-134
  Tensor fake_quantize_per_tensor_affine_cachemask_backward(const Tensor& dY,
  const Tensor& mask) { ...; return dY * mask; }` and is the consumer that
  `derivatives.yaml:673-674` wires into autograd. The current
  `FakeQuantizeBackward` at `ferrotorch-core/src/grad_fns/quantize_grad.rs:113-151`
  implements an equivalent STE — it computes the range boundary
  `[dequantize(qmin), dequantize(qmax)] = [(qmin - zp) * scale, (qmax - zp) *
  scale]` and zeros gradient outside it — but the consumer chain ends at
  REQ-1's missing surface (the only entry point that attaches
  `FakeQuantizeBackward` is `fake_quantize_differentiable`, which itself
  has no non-test production consumer per REQ-1's analysis).

  NOT-STARTED. The STE backward node exists in code
  (`FakeQuantizeBackward` at `quantize_grad.rs:113-151`) but the consumer
  chain that closes the SHIPPED bar is the REQ-1 forward acquiring a
  non-test consumer. Until REQ-1 lands, REQ-3 is structurally NOT-STARTED:
  a backward node with no production caller path is vocabulary, not
  shipped autograd. Tracked under blocker #1238 (closes when REQ-1
  consumer wiring lands).

## Acceptance Criteria

- [ ] AC-1: `fake_quantize_per_tensor_affine` parity-sweep at `--seeds 8`
  returns `[fake_quantize_per_tensor_affine] N/N passed (0 skipped, 0
  failed)` with `N >= 1` and `grep -c "passed (0 skipped, 0 failed)" == 1`.
  Currently fails with `oracle: unknown op:
  fake_quantize_per_tensor_affine` — the parity-sweep PyTorch oracle does
  not expose this op via `torch.testing._internal.opinfo.op_db`. Tracked
  by blocker #1240.
- [ ] AC-2: `fake_quantize_per_channel_affine` parity-sweep at `--seeds 8`
  returns `[fake_quantize_per_channel_affine] N/N passed (0 skipped, 0
  failed)` with `N >= 1`. Same oracle-availability blocker #1240 plus
  REQ-2 impl blocker #1239.
- [x] AC-3: `cargo test -p ferrotorch-core --lib grad_fns::quantize_grad`
  passes all 10 tests at `quantize_grad.rs:153-348`:
  `fake_quantize_round_trips_representable_values`,
  `fake_quantize_clamps_out_of_range_values`,
  `fake_quantize_rejects_zero_scale`,
  `fake_quantize_rejects_negative_scale`,
  `fake_quantize_rejects_inverted_range`,
  `fake_quantize_asymmetric_with_zero_point`,
  `fake_quantize_ste_passes_grad_for_in_range_values`,
  `fake_quantize_ste_zeros_grad_for_out_of_range_values`,
  `fake_quantize_no_grad_when_input_doesnt_require_grad`,
  `fake_quantize_preserves_grad_fn_when_input_requires_grad`,
  `fake_quantize_no_grad_context_skips_grad_fn`,
  `fake_quantize_chains_through_autograd_with_relu`. These tests exercise
  the CURRENT `fake_quantize_differentiable` signature, NOT the upstream
  `fake_quantize_per_tensor_affine` signature, so AC-3 passing does not
  imply REQ-1 is SHIPPED — only that the existing private surface is
  internally consistent.
- [ ] AC-4: The forward formula at `quantize_grad.rs:81-95` matches the
  documented PyTorch identity at `torch/_torch_docs.py:11958-11967`. The
  current implementation uses Rust `f32::round` which is round-half-away-
  from-zero, while upstream uses `std::nearbyint` which is round-half-
  to-even (banker's rounding) per FE_TONEAREST default. This is a real
  numerical divergence (off-by-1 ULP on `.5` boundaries) that lives on
  the REQ-1 SHIPPED critical path; flagged here so the fix lands when
  REQ-1 is re-implemented.
- [ ] AC-5: Non-test production consumer exists for the autograd-tracking
  fake quantize surface — i.e., a call site outside `#[cfg(test)]` blocks
  and outside the `pub use` re-export in `ferrotorch-core/src/lib.rs:163`
  invokes the function in a real consumer (e.g., `Tensor::fake_quantize_t`
  method on `Tensor<T>` analogous to `Tensor::cumsum_t` in `methods.rs:282`,
  or a QAT layer in `ferrotorch-nn` that wraps a `Tensor` rather than a
  raw `&[f32]` slice). Blocked by #1238.
- [x] AC-6: STE backward node correctness — `FakeQuantizeBackward` at
  `quantize_grad.rs:120-151` returns `grad_output * 1` for in-range values
  and `grad_output * 0` otherwise, matching upstream's `dY * mask` at
  `FakeQuantPerTensorAffine.cpp:133`. Verified by
  `fake_quantize_ste_passes_grad_for_in_range_values` and
  `fake_quantize_ste_zeros_grad_for_out_of_range_values` at
  `quantize_grad.rs:248-297`. (Mechanical correctness, not full SHIPPED —
  REQ-3's consumer chain still depends on REQ-1.)

## Architecture

### Layer split (`quantize_grad` vs `quantize`)

There are two distinct quantization layers in `ferrotorch-core`:

1. `ferrotorch-core/src/quantize.rs` (1700+ LOC): non-autograd, slice-based
   primitives (`FakeQuantize::forward(weights: &[f32]) -> (Vec<f32>,
   Vec<bool>)`), with the higher-level `QatModel`, `prepare_qat`, and
   per-tensor / per-channel utilities that operate on raw f32 slices. This
   is consumed by `ferrotorch-nn` (`QatModel.fake_quantize_weights` is the
   ferrotorch-nn-facing call surface; tested in
   `ferrotorch-nn/tests/conformance_nn_structural.rs:1651-1690`).
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
`derivatives.yaml:682-683`) gradients only the input via `dY * mask`,
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
| `fake_quantize_per_channel_affine` | `aten/src/ATen/native/quantized/FakeQuantPerChannelAffine.cpp:32-42 Tensor fake_quantize_per_channel_affine(const Tensor& self, const Tensor& scale, const Tensor& zero_point, int64_t axis, int64_t quant_min, int64_t quant_max)` | `tools/autograd/derivatives.yaml:682-683` (`fake_quantize_per_channel_affine_cachemask_backward = dY * mask`, mask broadcast along `axis`) | Same elementwise NaN / Inf / denormal / empty cases as per-tensor, plus: `scale.dim() == 1` enforced at `FakeQuantPerChannelAffine.cpp:55`, `zero_point.dim() == 1` at `:56`, `scale.numel() == self.size(axis)` at `:61`. Axis out-of-bounds: `axis >= 0 && axis <= self.dim()` at `:76` — note the `<=` is upstream's actual contract (axis-on-the-trailing-dim is permitted for a degenerate broadcast); ferrotorch should match. Zero-point dtype: upstream accepts `kInt`, `kFloat`, `kHalf` for zero_point with the float types triggering a `_get_rounded_zero_point` round-then-clamp at `:133-139`; ferrotorch's int-only zero_point sidesteps this. **Status: NOT-STARTED (impl missing per #1239; oracle missing per #1240).** |

Parity-sweep audit reference: BOTH ops are **MISSING** from
`tools/parity-sweep/parity_audit.json`. The PyTorch oracle
(`tools/parity-sweep/oracle/`) does not currently expose
`fake_quantize_per_tensor_affine` or `fake_quantize_per_channel_affine`
via `torch.testing._internal.opinfo.op_db` — these are not standard
op_db entries (op_db is the unit-test op set; quantization ops live
under a separate `torch.testing._internal.quantization` harness).
Adding parity-sweep oracle coverage is a strict prerequisite to AC-1
and AC-2 and is tracked by #1240 (analogous to the cumulative
#1230 oracle gap).

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

These tests exercise the CURRENT private surface
`fake_quantize_differentiable`. They do not exercise an upstream-byte-
faithful `fake_quantize_per_tensor_affine`. Therefore AC-3 passing is
necessary but not sufficient for REQ-1 SHIPPED — REQ-1 SHIPPED requires
additionally the upstream-matched public signature + a non-test
production consumer + parity-sweep coverage.

### Parity-sweep status

Both ops return `oracle: unknown op` at the current build:

```
$ ./target/release/parity-sweep sweep --op fake_quantize_per_tensor_affine --seeds 8
  FAIL: seed=0 i=0 oracle: oracle: unknown op: fake_quantize_per_tensor_affine
  ... (all 8 seeds fail with same oracle error)

$ ./target/release/parity-sweep sweep --op fake_quantize_per_channel_affine --seeds 8
  FAIL: seed=0 i=0 oracle: oracle: unknown op: fake_quantize_per_channel_affine
  ... (all 8 seeds fail with same oracle error)
```

Smoke grep count (`grep -c "passed (0 skipped, 0 failed)"`) is `0` for
both ops. Closing AC-1 / AC-2 requires landing oracle coverage in the
PyTorch oracle layer of the parity-sweep (issue #1240) plus the
upstream-byte-faithful Rust implementations of REQ-1 (#1238) and REQ-2
(#1239).

Note on the kernel layer: there is no `ops/quantize_grad.rs` analogous
to `ops/cumulative.rs`. The forward and backward both live entirely
inside `grad_fns/quantize_grad.rs` because the forward is a simple
elementwise loop and the backward is a binary mask multiply — neither
warrants a separate kernel-layer split.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (per-tensor) | NOT-STARTED | The current `fake_quantize_differentiable` at `ferrotorch-core/src/grad_fns/quantize_grad.rs:51-110` is a near-analog with divergent name (`fake_quantize_differentiable` vs upstream `fake_quantize_per_tensor_affine` per `torch/overrides.py:622`), divergent arg widths (i32 vs upstream i64 for `zero_point` / `quant_min` / `quant_max`), and no tensor-qparams overload (missing `fake_quantize_per_tensor_affine(input, scale: Tensor, zero_point: Tensor, ...)` from `FakeQuantPerTensorAffine.cpp:42-51`). Non-test production consumer is absent: only `pub use grad_fns::quantize_grad::fake_quantize_differentiable;` at `ferrotorch-core/src/lib.rs:163` references it outside `#[cfg(test)]` blocks, and a `pub use` is vocabulary-only per R-DEFER-1. `QatModel::fake_quantize_weights` at `ferrotorch-core/src/quantize.rs:1012` is a separate `&[f32]`-slice surface using a different `FakeQuantize::forward` struct in `quantize.rs`, not this file's `fake_quantize_differentiable`. Parity-sweep oracle does not expose the op (`oracle: unknown op`). Open prereq blocker: #1238 (rename + signature match + non-test consumer). Oracle dependency: #1240. |
| REQ-2 (per-channel) | NOT-STARTED | No `fake_quantize_per_channel_affine` function exists in `ferrotorch-core/src/grad_fns/quantize_grad.rs` (`grep -n 'per_channel' quantize_grad.rs` returns empty). The upstream contract at `aten/src/ATen/native/quantized/FakeQuantPerChannelAffine.cpp:32-42` requires a per-axis broadcast of 1-D `scale` / `zero_point` tensors, which has no analog in the current file. Open prereq blocker: #1239 (implement the per-channel forward + backward). Oracle dependency: #1240. |
| REQ-3 (STE backward) | NOT-STARTED | The `FakeQuantizeBackward<T>` grad-fn struct at `ferrotorch-core/src/grad_fns/quantize_grad.rs:113-151` mechanically implements the upstream `dY * mask` STE per `FakeQuantPerTensorAffine.cpp:121-134` (verified by `fake_quantize_ste_passes_grad_for_in_range_values` at `:248-269` and `fake_quantize_ste_zeros_grad_for_out_of_range_values` at `:271-297`), but the only attach site is `fake_quantize_differentiable` (REQ-1), which itself has no non-test production consumer. Per R-DEFER-1, a grad-fn struct without a production-callable forward surface is vocabulary-only. Tracked under the REQ-1 consumer-wiring blocker #1238 — when REQ-1's consumer lands, REQ-3 moves to SHIPPED simultaneously because the backward struct's correctness is already covered by AC-6. |
