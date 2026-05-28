# ferrotorch-vision — `models::segmentation::lraspp` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669 (working tree at /home/doll/pytorch)
baseline-torchvision: /home/doll/.local/lib/python3.13/site-packages/torchvision/models/segmentation/lraspp.py
upstream-paths:
  - /home/doll/.local/lib/python3.13/site-packages/torchvision/models/segmentation/lraspp.py
  - /home/doll/.local/lib/python3.13/site-packages/torchvision/models/mobilenetv3.py
-->

## Summary

`ferrotorch-vision/src/models/segmentation/lraspp.rs` ships the Lite R-ASPP
semantic-segmentation model with MobileNetV3-Large dilated backbone,
mirroring `torchvision.models.segmentation.lraspp_mobilenet_v3_large`
(torchvision 0.21.x). The file declares the `LrasppHead` (cbr + scale + low
+ high classifiers) and the composite `Lraspp` model with the
`lraspp_mobilenet_v3_large` constructor.

## Requirements

- REQ-1: `pub struct LrasppHead<T: Float>` mirrors torchvision's `LRASPPHead`
  at `torchvision/models/segmentation/lraspp.py` — it owns `cbr_conv` (1x1
  conv 960→128 bias=False), `cbr_bn` (BatchNorm2d 128), `scale_pool`
  (AdaptiveAvgPool2d(1)), `scale_conv` (1x1 conv 960→128 bias=False),
  `low_classifier` (1x1 conv 40→num_classes bias=True), `high_classifier`
  (1x1 conv 128→num_classes bias=True), and a `training: bool`. The ReLU
  and Sigmoid in torchvision's `cbr.2` / `scale.2` slots are
  parameter-free, applied inline.
- REQ-2: `LrasppHead::forward_features(low, high)` consumes the two-scale
  backbone features and returns `[B, num_classes, low.H, low.W]`. The
  algorithm: `x_cbr = relu(cbr_bn(cbr_conv(high)))`, `s =
  sigmoid(scale_conv(scale_pool(high)))`, `x = x_cbr * s`,
  `x_up = interpolate(x, low.shape[-2:], bilinear, align_corners=false)`,
  return `low_classifier(low) + high_classifier(x_up)`. Matches
  `LRASPPHead.forward` in `torchvision/models/segmentation/lraspp.py`.
- REQ-3: `Module::forward(input)` for `LrasppHead` returns
  `FerrotorchError::InvalidArgument`. The head consumes a `(low, high)`
  pair, not a single tensor; the single-input forward contract makes no
  sense for this module, so the type forces callers to use
  `forward_features(low, high)`.
- REQ-4: `pub struct Lraspp<T: Float>` owns a `MobileNetV3Large<T>` backbone
  built via `MobileNetV3Large::new_dilated(num_classes)` and an
  `LrasppHead<T>`. `Module::forward` runs the backbone's
  `forward_low_high` to extract `low: [B, 40, H/8, W/8]` and `high: [B,
  960, H/16, W/16]`, then `head.forward_features(low, high)`, then a final
  bilinear upsample to `[B, num_classes, H, W]`.
- REQ-5: The backbone's classification head (`classifier.<0,3>.*` Linear
  layers inside `MobileNetV3Large`) is unused; `Lraspp::parameters`,
  `parameters_mut`, `named_parameters` filter the `classifier.*` keys so
  the loader's view matches torchvision's `IntermediateLayerGetter`-stripped
  backbone schema.
- REQ-6: `pub fn lraspp_mobilenet_v3_large<T: Float>(num_classes: usize)`
  is the user-facing constructor matching
  `torchvision.models.segmentation.lraspp_mobilenet_v3_large(weights=None,
  num_classes=21)`. Channel widths low=40, high=960, inter=128 are fixed
  to match torchvision.
- REQ-7: `named_parameters` emits torchvision's exact state-dict-shaped
  keys: `backbone.features.<0..16>.<...>` (MobileNetV3-Large features) and
  `classifier.cbr.{0,1}.<...>`, `classifier.scale.1.<...>`,
  `classifier.low_classifier.<weight,bias>`,
  `classifier.high_classifier.<weight,bias>` (verified against the actual
  `lraspp_mobilenet_v3_large(weights='DEFAULT').state_dict()` dump).
- REQ-8: `children` / `named_children` overrides expose every BN-bearing
  sub-module so the BN-buffer loader's `named_descendants_dyn()` walk can
  resolve the running-statistic keys for every BN in the
  MobileNetV3-Large backbone AND in the LRASPP head's `cbr.1` slot.
- REQ-9: `Lraspp::backbone_forward_low_high` and
  `backbone_forward_with_block_dumps` expose per-stage / per-block
  diagnostics used by `examples/probe_lraspp_stages.rs` to localize parity
  failures in pretrained-weight workflows (#1146).

## Acceptance Criteria

- [x] AC-1: `lraspp_mobilenet_v3_large::<f32>(21).forward(&[1, 3, 32, 32])`
  returns `[1, 21, 32, 32]` (`test_lraspp_output_shape_small`).
- [x] AC-2: `lraspp_mobilenet_v3_large::<f32>(21).forward(&[1, 3, 64, 64])`
  returns `[1, 21, 64, 64]` (`test_lraspp_output_shape_64x64`).
- [x] AC-3: `lraspp_mobilenet_v3_large::<f32>(21).forward(&[2, 3, 32, 32])`
  returns `[2, 21, 32, 32]` (`test_lraspp_batch_size_2`).
- [x] AC-4: `lraspp_mobilenet_v3_large::<f32>(5).forward(&[1, 3, 32, 32])`
  returns `[1, 5, 32, 32]` (`test_lraspp_custom_num_classes`).
- [x] AC-5: `named_parameters` contains every critical torchvision key:
  `backbone.features.0.0.weight`, `backbone.features.1.block.0.0.weight`,
  `backbone.features.16.0.weight`, `classifier.cbr.0.weight`,
  `classifier.cbr.1.weight`, `classifier.cbr.1.bias`,
  `classifier.scale.1.weight`, `classifier.low_classifier.weight`,
  `classifier.low_classifier.bias`, `classifier.high_classifier.weight`,
  `classifier.high_classifier.bias` (`test_lraspp_named_parameter_prefixes`).
- [x] AC-6: `train` / `eval` toggle works (`test_lraspp_train_eval_toggle`).

## Architecture

`pub struct LrasppHead<T: Float>` (lines 85-101) carries six parameter-bearing
modules (cbr_conv, cbr_bn, scale_pool — the pool is param-free —,
scale_conv, low_classifier, high_classifier) and a `training` flag.
`LrasppHead::new(low_channels=40, high_channels=960, num_classes,
inter_channels=128)` (lines 113-140) constructs all six.

`LrasppHead::forward_features(low, high)` (lines 146-184):

```text
x_cbr = ReLU::new().forward( cbr_bn(cbr_conv(high)) )            # [B, 128, H/16, W/16]
s     = sigmoid( scale_conv( scale_pool(high) ) )                # [B, 128, 1, 1]
x     = x_cbr * s                                                # broadcast multiply
x_up  = interpolate(x, low.shape[-2:], bilinear, false)          # [B, 128, H/8, W/8]
return low_classifier(low) + high_classifier(x_up)               # [B, num_classes, H/8, W/8]
```

The broadcast multiply between `x_cbr [B, 128, H/16, W/16]` and `s [B, 128,
1, 1]` uses ferrotorch's `mul` which already supports NumPy-style
broadcasting (validated by the SE-block pattern in `ferrotorch-nn::se`).

`Module<T>::forward` for `LrasppHead` (lines 188-200) intentionally returns
`FerrotorchError::InvalidArgument` because the head's contract is a
two-input pair, not a single tensor.

`pub struct Lraspp<T: Float>` (lines 314-318) owns `backbone:
MobileNetV3Large<T>` (built via `MobileNetV3Large::new_dilated(num_classes)`)
and `head: LrasppHead<T>`. `Module::forward` (lines 362-380) runs
`backbone.forward_low_high(input)` → `head.forward_features(&low, &high)` →
bilinear interpolate to `[B, num_classes, H, W]`.

`parameters` / `parameters_mut` (lines 382-441) collect backbone names
first, filter out `classifier.*` (the unused inner backbone classifier),
zip with the params, then extend with the head's params — preserving the
order torchvision's safetensors emits (backbone-then-head).

`named_parameters` (lines 443-461) emits `backbone.<features.*>` (skipping
the inner `classifier.*` Linear heads) and `classifier.<cbr|scale|...>` for
the head — verified against the actual state-dict dump.

`named_children` (lines 467-479) exposes the backbone at path `"backbone"`
and the LRASPP head at path `"classifier"`.

### Non-test production consumers

- `pub use lraspp::{Lraspp, LrasppHead, lraspp_mobilenet_v3_large}` at
  `ferrotorch-vision/src/models/segmentation/mod.rs:24` and
  `ferrotorch-vision/src/models/mod.rs:40-43`.
- `default_registry()` in `ferrotorch-vision/src/models/registry.rs:338-340`
  binds the `"lraspp_mobilenet_v3_large"` entry, constructing the model
  via `super::segmentation::lraspp_mobilenet_v3_large::<f32>(num_classes)`
  inside `maybe_load_pretrained`.
- `scripts/probe_lraspp_descendants.py` (Python harness with embedded Rust
  used for #1146 parity diagnostics) imports
  `ferrotorch_vision::models::segmentation::lraspp_mobilenet_v3_large`.

## Parity contract

`parity_ops = []`. LRASPP composes `Conv2d`, `BatchNorm2d`,
`AdaptiveAvgPool2d`, `relu`, `sigmoid`, the broadcasting `mul`, `add`,
`interpolate`, and the MobileNetV3-Large dilated backbone. Every primitive
is covered under its own parity entry.

Edge cases preserved versus torchvision:

- **`MobileNetV3Large::new_dilated`**: `replace_stride_with_dilation=[True,
  True, True]` on the MobileNetV3-Large feature stack so the high-res
  output is `[B, 960, H/16, W/16]` (matches `lraspp.py`'s
  `mobilenet_v3_large(dilated=True)`).
- **Low / high tap points**: `features.4` → low (40 channels, H/8, W/8) and
  `features.16` → high (960 channels, H/16, W/16). Matches torchvision's
  `IntermediateLayerGetter(return_layers={"4": "low", "16": "high"})`
  pattern in `lraspp_mobilenet_v3_large`.
- **Inter channels**: `128`, matching torchvision's hard-coded constant in
  `_LRASPPHead.__init__`.
- **No bias on `cbr` / `scale` convs**: the BN that follows absorbs the
  constant.
- **Bias on classifiers**: `low_classifier` and `high_classifier` both
  carry `bias=True` (torchvision Conv2d default).
- **`align_corners=False`** on every bilinear upsample.

## Verification

Tests in `mod tests` in `lraspp.rs`:

- `test_lraspp_output_shape_{small,64x64}` — 32x32 and 64x64 inputs.
- `test_lraspp_batch_size_2` — batch dimension propagates.
- `test_lraspp_custom_num_classes` — `num_classes=5`.
- `test_lraspp_named_parameter_prefixes` — every critical torchvision key
  exists (11 explicit assertions, verified against
  `lraspp_mobilenet_v3_large(weights='DEFAULT').state_dict()`).
- `test_lraspp_train_eval_toggle`.

Smoke command:

```bash
cargo test -p ferrotorch-vision --lib segmentation::lraspp:: 2>&1 | tail -3
```

Expected: all tests pass; no `parity-sweep` ops to run.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct LrasppHead<T: Float>` at `LrasppHead in lraspp.rs` + `LrasppHead::new` at `LrasppHead in lraspp.rs`; non-test consumer: `Lraspp::new` constructs `LrasppHead::new(40, 960, num_classes, 128)` at `lraspp in lraspp.rs`. |
| REQ-2 | SHIPPED | impl: `LrasppHead::forward_features(low, high)` at `lraspp in lraspp.rs`; non-test consumer: `Lraspp::forward` invokes `self.head.forward_features(&low, &high)` at `lraspp in lraspp.rs`. |
| REQ-3 | SHIPPED | impl: `Module<T>::forward` for `LrasppHead in lraspp.rs` returns `FerrotorchError::InvalidArgument`; non-test consumer: enforced by `Lraspp::forward` at `lraspp in lraspp.rs` which uses the typed two-input `forward_features` directly, never the trait's single-input `forward`. |
| REQ-4 | SHIPPED | impl: `pub struct Lraspp<T: Float>` at `Lraspp in lraspp.rs` + `Module::forward` at `Lraspp in lraspp.rs` (backbone `forward_low_high` → head → upsample); non-test consumer: `lraspp_mobilenet_v3_large` constructor at `lraspp in lraspp.rs` returns a `Lraspp<T>` to the registry caller `registry.rs`. |
| REQ-5 | SHIPPED | impl: `parameters`, `parameters_mut`, `named_parameters` filter `classifier.*` from the backbone's keys at `lraspp in lraspp.rs, 443-461`; non-test consumer: `maybe_load_pretrained` at `named_parameters in registry.rs` consumes the filtered view via `model.load_state_dict(&state_dict, false)`. |
| REQ-6 | SHIPPED | impl: `pub fn lraspp_mobilenet_v3_large<T: Float>(num_classes: usize) -> FerrotorchResult<Lraspp<T>>` at `lraspp in lraspp.rs`; non-test consumer: `registry.rs` calls `super::segmentation::lraspp_mobilenet_v3_large::<f32>(num_classes)` inside the `default_registry()` `maybe_load_pretrained` closure. |
| REQ-7 | SHIPPED | impl: `Module::named_parameters` for `Lraspp in lraspp.rs` emits `backbone.features.<...>` (filtered) and `classifier.<...>`; non-test consumer: `maybe_load_pretrained` at `named_parameters in registry.rs` consumes the keys when loading the pretrained safetensors. |
| REQ-8 | SHIPPED | impl: `children` / `named_children` for `LrasppHead in lraspp.rs` and for `Lraspp in lraspp.rs`; non-test consumer: `apply_bn_buffers_from_state_dict` at `registry.rs` walks `named_descendants_dyn()` reaching every BN under `backbone.features.<...>` and `classifier.cbr.1`. |
| REQ-9 | SHIPPED | impl: `Lraspp::backbone_forward_low_high` at `lraspp in lraspp.rs` and `backbone_forward_with_block_dumps` at `lraspp in lraspp.rs`; non-test consumer: invoked by the Rust source embedded in `scripts/probe_lraspp_descendants.py` (per the file's own `pub`-visible docs at `lraspp in lraspp.rs`), and exposed transitively via the `pub use lraspp_mobilenet_v3_large` re-export. |
