# Quantization

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - torch/ao/quantization/__init__.py
  - torch/ao/quantization/observer.py
  - torch/ao/quantization/qconfig.py
  - aten/src/ATen/native/quantized/cpu/qmatmul.cpp
  - aten/src/ATen/native/quantized/cpu/kernels/QuantizedOpKernels.cpp
  - torch/_torch_docs.py
  - torch/overrides.py
-->

## Summary

`ferrotorch-core/src/quantize.rs` ships post-training quantization (PTQ)
and quantization-aware training (QAT) primitives for ferrotorch
tensors. Mirrors `torch.ao.quantization` and the `at::quantized` op
family. The module supports symmetric and asymmetric quantization to
INT8 / INT4 / UINT8, per-tensor and per-channel granularity, with the
classical observers (MinMax, PerChannelMinMax, Histogram) and a
`FakeQuantize` op for QAT.

## Requirements

- REQ-1: `enum QuantScheme { PerTensor, PerChannel(axis) }` and
  `enum QuantDtype { Int8, Int4, Uint8 }` — granularity + target
  integer dtype. Mirrors `torch.per_tensor_affine` /
  `torch.per_channel_affine` `Quant` types.
- REQ-2: `struct QuantizedTensor` — i8-storage with `scale: Vec<f32>`,
  `zero_point: Vec<i32>`, `shape`, `scheme`, `dtype`. Recovery is
  `x = (q - zp) * scale`. Mirrors `torch.qint8` / `torch.qint4` /
  `torch.quint8` storage.
- REQ-3: `quantize(input, scheme, dtype)` — compute scale + zero-point
  from min/max using PyTorch observer formulas, clip, and round to int
  with round-half-to-even. Mirrors
  `torch.quantize_per_tensor` / `torch.quantize_per_channel`.
- REQ-4: `dequantize(qtensor) -> Tensor<T>` — inverse of REQ-3.
  Mirrors `torch.dequantize`.
- REQ-5: `quantized_matmul(a, b)` — INT8 matmul over quantized
  tensors, dequantized output. Mirrors the
  `aten/src/ATen/native/quantized/cpu/qmatmul.cpp` kernel.
- REQ-6: `struct QParams` — observed (scale, zero_point) bundle used
  to thread quantization parameters through QAT.
- REQ-7: `trait Observer` — calibration interface with
  `observe(tensor)` and `calculate_qparams()`. Implementations:
  `MinMaxObserver`, `PerChannelMinMaxObserver`,
  `HistogramObserver`. Mirrors
  `torch.ao.quantization.observer.*`.
- REQ-8: `struct FakeQuantize` — straight-through quantization op for
  QAT. Forward: quantize-then-dequantize. Backward: identity (STE).
  Mirrors `torch.fake_quantize_per_tensor_affine` (the backward is
  implemented in `grad_fns/quantize_grad.rs`, not here).
- REQ-9: `struct QatLayer` / `struct QatModel` — high-level wrappers
  for managing per-parameter observers + fake-quantize during a QAT
  training run. `prepare_qat(param_names, dtype)` builds a fresh
  `QatModel`. Mirrors `torch.ao.quantization.prepare_qat`.
- REQ-10: `quantize_named_tensors` — bulk quantize a `HashMap<String,
  Tensor<T>>` into a `HashMap<String, QuantizedTensor>` for saving
  a quantized state-dict.

## Acceptance Criteria

- [x] AC-1: `dequantize(quantize(t)) ≈ t` within scale-determined
  tolerance for INT8 PerTensor.
- [x] AC-2: `quantize` with `PerChannel(axis)` produces one
  scale/zp per slice along `axis`.
- [x] AC-3: `MinMaxObserver` after observing `t` produces
  `(scale, zp)` such that `quantize` clips to the observed range.
- [x] AC-4: `FakeQuantize.forward(t)` is quantize-then-dequantize and
  rounds via the same path as `quantize`.
- [x] AC-5: `quantized_matmul` matches the reference dequantize-matmul-
  requantize chain within INT8 rounding error.
- [x] AC-6: `prepare_qat(["w"], Int8)` returns a `QatModel` with the
  named parameter wired through `FakeQuantize`.
- [x] AC-7: `cargo test -p ferrotorch-core --lib quantize` passes.

## Architecture

The 1700+ LOC file is organised as:

- **Enums** (`quantize.rs:36-65`): `QuantScheme`, `QuantDtype`. The
  dtype carries `qmin()` / `qmax()` accessors for the per-dtype int
  range.
- **`QuantizedTensor`** (`quantize.rs:88`): owns the i8 data plus
  the per-channel scale/zp vectors. The `Int4` dtype packs two
  4-bit values per `i8` storage byte (low 4 bits significant).
- **Quantize / dequantize** (`quantize.rs:288`, `quantize.rs:399`):
  - `quantize(input, scheme, dtype)` — computes per-tensor or
    per-channel min/max, derives `(scale, zp)` with
    `MinMaxObserver._calculate_qparams` parity, then rounds via
    inverse-scale multiply plus nearest-even rounding and clips.
  - `dequantize(q)` — inverse using `(q - zp) * scale` per element.
- **`quantized_matmul`** (`quantize.rs:439`) — validates 2-D
  per-tensor quantized inputs, accumulates centered integer products
  in `i64`, computes the real output range in `f64`, derives INT8
  output qparams, and requantizes back to a `QuantizedTensor`.
  This mirrors PyTorch's quantized matmul contract at
  `aten/src/ATen/native/quantized/cpu/qmatmul.cpp`: supported kernels
  use integer accumulation plus requantization, while the non-RUY
  fallback dequantizes, calls float `matmul`, and quantizes the result.
  ferrotorch widens the internal accumulator beyond PyTorch's backend
  `int32_t` implementation detail because this API derives output
  qparams from the full result and must not debug-panic or release-wrap
  on ordinary long INT8 inner dimensions (CORE-086 / #1780).
- **`quantize_named_tensors`** (`quantize.rs:615`) — bulk
  state-dict path; one entry per `(name, scheme, dtype)`.
- **`QParams`** (`quantize.rs:634`) — `(scale, zp)` bundle plus
  the qmin/qmax for downstream consumers.
- **`trait Observer`** (`quantize.rs:675`) — `observe(tensor)`
  updates internal state; `calculate_qparams()` returns the
  `QParams`.
- **`MinMaxObserver`** (`quantize.rs:692`): scalar running min /
  max. Per-tensor symmetric / asymmetric supported via `QuantScheme`
  passed at construction.
- **`PerChannelMinMaxObserver`** (`quantize.rs:747`): per-slice
  running min/max along `axis`.
- **`HistogramObserver`** (`quantize.rs:866`, constructor
  `quantize.rs:882`): histogram observer with strictly positive bin
  construction, NaN/Inf filtering, range expansion, and redistributed
  counts. The zero-bin boundary differs from PyTorch's delayed
  `torch.histogram` runtime error because ferrotorch's `Observer::observe`
  is infallible; invalid bin counts are rejected by the constructor instead.
- **`FakeQuantize`** (`quantize.rs:1031`): forward quantize-then-
  dequantize, backward STE (the gradient impl lives in
  `grad_fns/quantize_grad.rs`).
- **`QatLayer`** (`quantize.rs:1119`) / **`QatModel`** (`quantize.rs:1133`) —
  per-param observer + fake-quantize bundles. `prepare_qat` is a factory.

Non-test production consumers:

- `grad_fns/quantize_grad.rs:334, 632` instantiate
  `FakeQuantizeBackward` and `FakeQuantizePerChannelBackward` which
  reference the forward output produced by the surfaces in this file.
- `methods.rs:596, 628` route `Tensor::fake_quantize_per_tensor_affine_t`
  and `Tensor::fake_quantize_per_channel_affine_t` through the
  `FakeQuantize` forward + backward chain.
- `lib.rs:219-227` re-exports the `Observer` trait + concrete
  observers + `QuantizedTensor` + `FakeQuantize` + the QAT helpers
  for downstream `ferrotorch-nn` callers.

## Parity contract

`parity_ops = []`. Quantization parity is enforced at the op level
(`fake_quantize_per_tensor_affine`, `fake_quantize_per_channel_affine`
in the `grad_fns/quantize_grad.rs` parity-sweep entries documented in
that module's REQ table). The PTQ flow (this file) is exercised by
`tests/conformance_quantize_prune.rs` fixtures generated from live
PyTorch observer and quantized tensor APIs. The conformance suite pins
bit-exact integer codes, all-zero scale floors, affine zero-point
rounding/clamping, unobserved observer defaults, symmetric scale
denominators, dequantized round-trips, quantized-matmul accumulator
behavior at and just beyond the `i32` boundary, and histogram-observer
zero/one-bin construction boundaries.

## Verification

```bash
cargo test -p ferrotorch-core --test conformance_quantize_prune
cargo test -p ferrotorch-core --test conformance_quantize_prune -- quantized_matmul
cargo test -p ferrotorch-core --lib quantize
```

Expected: PyTorch fixture conformance, round-trip, and observer tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `QuantScheme in ferrotorch-core/src/quantize.rs`, `QuantDtype in ferrotorch-core/src/quantize.rs`; non-test consumer: re-exported at `ferrotorch-core/src/lib.rs`, reachable by downstream callers via `ferrotorch_core::QuantDtype::Int8`. |
| REQ-2 | SHIPPED | impl: `QuantizedTensor in ferrotorch-core/src/quantize.rs`; non-test consumer: re-exported at `lib.rs`; threaded through `quantize_named_tensors`'s return type and bulk state-dict callers. |
| REQ-3 | SHIPPED | impl: `quantize in ferrotorch-core/src/quantize.rs`; non-test consumer: re-exported at `quantize in lib.rs`; called by `quantize_named_tensors` at `quantize in lib.rs` and by `FakeQuantize::forward` (transitively, via `forward in grad_fns/quantize_grad.rs`). |
| REQ-4 | SHIPPED | impl: `dequantize in ferrotorch-core/src/quantize.rs`; non-test consumer: re-exported at `lib.rs`; called by `quantized_matmul` at `lib.rs` and by `FakeQuantize::forward`. |
| REQ-5 | SHIPPED | impl: `quantized_matmul` at `ferrotorch-core/src/quantize.rs:439`; non-test consumer: re-exported at `lib.rs:219-227`. Accumulator parity/safety: public conformance tests `quantized_matmul_accumulator_crosses_i32_boundary_without_wrapping` and `quantized_matmul_negative_accumulator_crosses_i32_boundary_without_wrapping` prove long-inner-dimension centered INT8 products do not debug-panic or release-wrap at the i32 boundary. |
| REQ-6 | SHIPPED | impl: `QParams in ferrotorch-core/src/quantize.rs`; non-test consumer: re-exported at `lib.rs`; produced by every observer and consumed by `QatModel.step()`. |
| REQ-7 | SHIPPED | impl: `trait Observer` at `MinMaxObserver in ferrotorch-core/src/quantize.rs`, `MinMaxObserver in ferrotorch-core/src/quantize.rs`, `PerChannelMinMaxObserver in ferrotorch-core/src/quantize.rs`, `HistogramObserver in ferrotorch-core/src/quantize.rs`; non-test consumer: re-exported at `lib.rs`; threaded through `QatLayer` at `lib.rs`. |
| REQ-8 | SHIPPED | impl: `FakeQuantize in ferrotorch-core/src/quantize.rs`; non-test consumer: `FakeQuantize in grad_fns/quantize_grad.rs` (`FakeQuantizeBackward` attaches via the forward op) — production autograd graph. Consumer chain `Tensor::fake_quantize_per_tensor_affine_t` at `fake_quantize_per_tensor_affine_t in methods.rs`. |
| REQ-9 | SHIPPED | impl: `QatLayer in ferrotorch-core/src/quantize.rs`, `QatModel in ferrotorch-core/src/quantize.rs`, `prepare_qat in ferrotorch-core/src/quantize.rs`; non-test consumer: re-exported at `lib.rs`; entrypoint for the QAT training-loop API. |
| REQ-10 | SHIPPED | impl: `quantize_named_tensors` at `ferrotorch-core/src/quantize.rs:615`; non-test consumer: re-exported at `lib.rs:219-227`; called by state-dict-save flows that emit a quantized checkpoint. |
