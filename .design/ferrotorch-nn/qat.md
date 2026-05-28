# ferrotorch-nn — `qat` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/nn/
  - aten/src/ATen/native/
-->

## Summary

`ferrotorch-nn/src/qat.rs` provides nn-module-level integration for
quantization-aware training (QAT). Re-exports the core types from
`ferrotorch_core::quantize` (`QatLayer`, `QatModel`,
`prepare_qat` as `core_prepare_qat`) and adds an
nn-aware `prepare_qat` that scans a `Module<T>`'s named parameters
to construct a `QatModel`.

The closest upstream counterpart is
`torch.ao.quantization.prepare_qat` and friends; upstream does not
ship the entire QAT pipeline as a single `torch.nn` module, so
ferrotorch's `qat.rs` is a translation of the user-API contract
rather than a strict line-for-line mirror.

## Requirements

- REQ-1: `pub enum ObserverType` — selects the calibration observer:
  `MinMax` (global min/max), `MovingAverageMinMax` (EMA-smoothed
  min/max), `Histogram` (percentile clipping), `PerChannelMinMax`
  (weight-only per-output-channel). Mirrors upstream's
  `torch.ao.quantization.observer.{MinMaxObserver,
  MovingAverageMinMaxObserver, HistogramObserver,
  PerChannelMinMaxObserver}`.

- REQ-2: `pub struct QatConfig` — quantization configuration with
  `weight_dtype`, `activation_dtype`, `weight_symmetric`,
  `activation_symmetric`, `weight_observer`,
  `activation_observer` fields. Mirrors upstream's `QConfig`.

- REQ-3: Pre-built configs — `QatConfig::default_symmetric_int8`,
  `::per_channel_int8`, `::int4_weight_int8_activation`. Mirror the
  common `prepare_qat_qat_qconfig` presets from upstream.

- REQ-4: `pub struct QuantizedModel` — deployment-ready quantized
  model. Stores `weights: HashMap<String, QuantizedTensor>` and
  `weight_qparams: HashMap<String, QParams>` plus the producing
  `QatConfig`. Mirrors upstream's `torch.ao.quantization.convert`
  output structure.

- REQ-5: `QuantizedModel` accessors — `weight(name)`,
  `weight_qparams(name)`, `weight_names()`, `num_weights()`,
  `dequantize_weight(name)`, `quantized_size_bytes`,
  `compression_ratio`, `config`. Mirror upstream's
  `state_dict()`-style accessors plus the convenience metrics.

- REQ-6: `pub fn prepare_qat(module: &dyn Module<f32>, config:
  QatConfig) -> QatModel` — scans `module.named_parameters()`,
  drops bias names (the convention: only `*.weight` parameters get
  FakeQuantize), and forwards the weight name list to
  `core_prepare_qat(&param_names, config.weight_dtype)`. Mirrors
  upstream's `torch.ao.quantization.prepare_qat(model)`.

- REQ-7: `pub use ferrotorch_core::quantize::{QatLayer, QatModel,
  prepare_qat as core_prepare_qat}` — re-export of the core
  quantize types so consumers can write
  `use ferrotorch_nn::{QatModel, prepare_qat}` instead of pulling
  from two crates. Matches upstream's flat
  `torch.ao.quantization` namespace.

## Acceptance Criteria

- [x] AC-1: `QatConfig::default_symmetric_int8()` returns
  `weight_dtype = Int8`, `activation_dtype = Int8`,
  `weight_symmetric = true`.
- [x] AC-2: `QatConfig::per_channel_int8()` returns
  `weight_observer = PerChannelMinMax`.
- [x] AC-3: `QatConfig::int4_weight_int8_activation()` returns
  `weight_dtype = Int4`, `activation_dtype = Int8`.
- [x] AC-4: `QuantizedModel` with empty weights returns
  `num_weights() = 0` and `compression_ratio() = 1.0` (the
  no-compression sentinel).
- [x] AC-5: `prepare_qat(module, config)` accepts a
  `&dyn Module<f32>` and produces a `QatModel`.

## Architecture

### ObserverType (REQ-1)

`pub enum ObserverType` at
`pub enum ObserverType in qat.rs` carries four `Copy, Eq` variants
selecting the calibration observer. Used inside `QatConfig` to
pick the per-weight and per-activation observer at config time.

### QatConfig (REQ-2, REQ-3)

`pub struct QatConfig` at
`pub struct QatConfig in qat.rs` holds the six configuration
fields. The three associated constructors
(`default_symmetric_int8`, `per_channel_int8`,
`int4_weight_int8_activation`) package the common presets.

### QuantizedModel (REQ-4, REQ-5)

`pub struct QuantizedModel` at
`pub struct QuantizedModel in qat.rs` carries two `HashMap<String,
_>` collections (one for the quantized tensors, one for the
qparams) plus the originating `QatConfig`.

`quantized_size_bytes` sums `qt.numel()` (1 byte per element for
INT8; INT4 packing is handled at the core layer). `compression_ratio`
returns `float_bytes / quantized_bytes` (defaults to `1.0` for the
empty model). `dequantize_weight` calls
`ferrotorch_core::dequantize(qt)` to recover an f32 tensor for
inspection.

### prepare_qat (REQ-6)

`pub fn prepare_qat` at
`pub fn prepare_qat in qat.rs` takes a `&dyn Module<f32>`, calls
`module.named_parameters()`, collects the parameter names into a
`Vec<&str>`, and forwards them to
`core_prepare_qat(&param_names, config.weight_dtype)`. The bias
filtering convention (only `*.weight` keys get FakeQuantize) is
applied at the core layer.

### Re-exports (REQ-7)

`pub use ferrotorch_core::quantize::{QatLayer, QatModel,
prepare_qat as core_prepare_qat}` at the top of `qat.rs` flattens
the core API into ferrotorch-nn's surface so callers can pull
everything from a single `ferrotorch_nn::*` import.

### Non-test production consumers

- `pub use qat::{ObserverType, QatConfig, QatModel,
  QuantizedModel, prepare_qat}` at `ferrotorch-nn/src/lib.rs:244`
  — grandfathered public API surface. No downstream model crate in
  the main tree currently consumes QAT directly; the API is
  exposed for external user code that wants to QAT-prepare an
  existing model and then `convert` it to a `QuantizedModel`.

## Parity contract

`parity_ops = []`. QAT is a deployment pipeline rather than a
numerical op, so it has no parity-sweep oracle. The numerical
contract is verified by the `ferrotorch-core::quantize` tests
which check that `quantize → dequantize` round-trips preserve
values up to the per-dtype tolerance (Int8: `1/127.5` per output
unit; Int4: `1/7.5`).

Edge cases preserved:

- **Empty model** — `QuantizedModel::compression_ratio` returns
  `1.0` (no compression). Matches upstream's defensive default.
- **`dequantize_weight` on missing key** — returns
  `FerrotorchError::InvalidArgument` rather than panicking.
- **Per-channel INT8** — `PerChannelMinMax` observer produces
  one qparam per output channel; matches upstream's
  `torch.ao.quantization.PerChannelMinMaxObserver`.

## Verification

Tests in `mod tests in qat.rs`:

- `test_qat_config_presets` — checks the three pre-built configs
  contain the expected dtype / observer combinations.
- `test_quantized_model_empty` — checks the empty-model sentinel
  values.

No parity-sweep ops. Smoke command:

```bash
cargo test -p ferrotorch-nn --lib qat:: 2>&1 | tail -3
```

Expected: 2 tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub enum ObserverType` in `qat.rs`; non-test consumer: re-export at `ferrotorch-nn/src/lib.rs:244`. |
| REQ-2 | SHIPPED | impl: `pub struct QatConfig` with six fields in `qat.rs`; non-test consumer: re-export at `lib.rs`. |
| REQ-3 | SHIPPED | impl: `pub fn default_symmetric_int8`, `pub fn per_channel_int8`, `pub fn int4_weight_int8_activation` on `QatConfig` in `qat.rs`; non-test consumer: re-export at `lib.rs`. |
| REQ-4 | SHIPPED | impl: `pub struct QuantizedModel` in `qat.rs`; non-test consumer: re-export at `lib.rs`. |
| REQ-5 | SHIPPED | impl: `pub fn weight`, `pub fn weight_qparams`, `pub fn weight_names`, `pub fn num_weights`, `pub fn dequantize_weight`, `pub fn quantized_size_bytes`, `pub fn compression_ratio`, `pub fn config` on `QuantizedModel` in `qat.rs`; non-test consumer: re-export at `lib.rs`. |
| REQ-6 | SHIPPED | impl: `pub fn prepare_qat` in `qat.rs` (calls `module.named_parameters()` then `core_prepare_qat`); non-test consumer: re-export at `lib.rs`. |
| REQ-7 | SHIPPED | impl: `pub use ferrotorch_core::quantize::{QatLayer, QatModel, prepare_qat as core_prepare_qat}` near the top of `qat.rs`; non-test consumer: re-export at `lib.rs` exposes the flattened surface. |
