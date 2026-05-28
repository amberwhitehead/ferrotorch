# ferrotorch-vision — `models::segmentation::deeplabv3` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669 (working tree at /home/doll/pytorch)
baseline-torchvision: /home/doll/.local/lib/python3.13/site-packages/torchvision/models/segmentation/deeplabv3.py
upstream-paths:
  - /home/doll/.local/lib/python3.13/site-packages/torchvision/models/segmentation/deeplabv3.py
  - /home/doll/.local/lib/python3.13/site-packages/torchvision/models/segmentation/_utils.py
-->

## Summary

`ferrotorch-vision/src/models/segmentation/deeplabv3.rs` ships the DeepLabV3
semantic-segmentation model with ResNet-50 dilated backbone, mirroring
`torchvision.models.segmentation.deeplabv3_resnet50` (torchvision 0.21.x). The
file declares a `ResNet50Dilated` backbone wrapper, a `DeepLabV3Head` (ASPP +
3x3 conv + BN + ReLU + 1x1 classifier), the composite `DeepLabV3` model, and
the `deeplabv3_resnet50` constructor.

## Requirements

- REQ-1: `pub struct ResNet50Dilated<T: Float>` is a thin wrapper around
  `resnet50_dilated::<T>(1000, [false, true, true])` exposing only the
  layer-4 feature map for the head. The classifier head (`fc`) is unused;
  `parameters()` / `named_parameters()` / `parameters_mut()` filter the
  `fc.*` keys to mirror torchvision's `IntermediateLayerGetter`-wrapped
  backbone schema (no `backbone.fc.*` keys in the state dict). Mirrors
  `_deeplabv3_resnet` at `torchvision/models/segmentation/deeplabv3.py:117`.
- REQ-2: `ResNet50Dilated::forward_layer4` runs the inner ResNet via
  `IntermediateFeatures::forward_features`, retrieves the `"layer4"` entry,
  and returns `[B, 2048, H/16, W/16]`. The dilated-stride threading
  ([false, true, true]) replaces layer3+layer4 stride-2 with dilation 2x,
  4x so the spatial resolution stays at H/16 throughout (output stride =
  16). Mirrors torchvision's `resnet50(replace_stride_with_dilation=[False,
  True, True])` at `deeplabv3.py:275`.
- REQ-3: `pub struct DeepLabV3Head<T: Float>` is a 5-element pipeline
  matching torchvision's `DeepLabHead(nn.Sequential)` at `deeplabv3.py:49`:
  index 0 = ASPP, index 1 = `Conv2d(256, 256, 3x3, pad=1, bias=False)`,
  index 2 = `BatchNorm2d(256)`, index 3 = ReLU (parameter-free, applied
  inline), index 4 = `Conv2d(256, num_classes, 1x1, bias=True)`. State-dict
  layout uses the torchvision-shaped keys `aspp.<...>`,
  `conv_intermediate.<...>`, `bn_intermediate.<...>`, `classifier.<...>`.
- REQ-4: `DeepLabV3Head::new(in_channels, num_classes, atrous_rates)` accepts
  the three atrous rates as a tuple, threading them through `Aspp::new`.
  Mirrors `DeepLabHead.__init__(in_channels, num_classes, atrous_rates=(12,
  24, 36))` at `deeplabv3.py:50`.
- REQ-5: `pub struct DeepLabV3<T: Float>` composes the backbone + head,
  performs the forward pass (backbone layer4 → head → bilinear upsample to
  input spatial size), and returns `[B, num_classes, H, W]`. Mirrors
  `_SimpleSegmentationModel.forward` at
  `torchvision/models/segmentation/_utils.py` (which calls
  `F.interpolate(x, size=input_shape, mode="bilinear",
  align_corners=False)`).
- REQ-6: `pub fn deeplabv3_resnet50<T: Float>(num_classes: usize)` is the
  user-facing constructor: defaults to atrous rates `(12, 24, 36)` and
  starts in `eval` mode (matching torchvision's
  `deeplabv3_resnet50(weights=None, num_classes=21)` default at
  `deeplabv3.py:233`).
- REQ-7: `named_parameters` over the composite model produces the
  torchvision-shaped keys `backbone.<...>` (prefixed via ResNet50Dilated
  which strips `fc.*`) and `head.<...>`. This layout supports the strict
  value-parity loader's `model.load_state_dict(&state_dict, false)`
  workflow.
- REQ-8: `children` / `named_children` overrides at every level expose the
  full descendant tree without a leading `.` (regression test
  `deeplabv3_named_descendants_no_leading_dot` locks #1142 — previously the
  transparent backbone wrapper emitted `.backbone` instead of
  `backbone`, breaking the BN-buffer loader silently).

## Acceptance Criteria

- [x] AC-1: `deeplabv3_resnet50::<f32>(21).forward(&[1, 3, 32, 32])` returns
  `[1, 21, 32, 32]` (`test_deeplabv3_output_shape_small`).
- [x] AC-2: `deeplabv3_resnet50::<f32>(21).forward(&[1, 3, 64, 64])` returns
  `[1, 21, 64, 64]` (`test_deeplabv3_output_shape_64x64`).
- [x] AC-3: `deeplabv3_resnet50::<f32>(21).forward(&[2, 3, 32, 32])` returns
  `[2, 21, 32, 32]` (`test_deeplabv3_batch_size_2`).
- [x] AC-4: `deeplabv3_resnet50::<f32>(5).forward(&[1, 3, 32, 32])` returns
  `[1, 5, 32, 32]` (`test_deeplabv3_custom_num_classes`).
- [x] AC-5: `named_parameters` contains both `backbone.` and `head.`
  prefixed names (`test_deeplabv3_named_parameter_prefixes`).
- [x] AC-6: Total parameter count > 30M, matching torchvision's ~39.6M
  (`test_deeplabv3_param_count_sanity`).
- [x] AC-7: `train` / `eval` toggle works (`test_deeplabv3_train_eval_toggle`).
- [x] AC-8: `named_descendants_dyn` paths have no leading `.` and the
  canonical `backbone.layer1.0.bn1` path is reachable
  (`deeplabv3_named_descendants_no_leading_dot`).

## Architecture

`pub struct ResNet50Dilated<T: Float>` at lines 84-113 owns
`inner: ResNet<T>` (constructed via
`resnet50_dilated::<T>(1000, [false, true, true])`) and a `training: bool`.
`forward_layer4` (line 105-112) calls `self.inner.forward_features(input)`,
retrieves the `"layer4"` HashMap entry, and returns a clone. The threading
of `replace_stride_with_dilation` is inherited from `resnet50_dilated`
verbatim; layer3[0] uses dilation=1 (`previous_dilation`) while
layer3[1..] use dilation=2, and layer4[0] uses dilation=2 with layer4[1..]
at dilation=4. The `_make_layer` `previous_dilation` snapshot semantics are
the same as torchvision's `ResNet._make_layer` (see also
`.design/ferrotorch-vision/models/resnet.md` REQ-5).

`Module<T> for ResNet50Dilated<T>` (lines 115-197) filters `fc.*` from
`parameters`, `parameters_mut`, and `named_parameters`; the
`named_parameters` impl prepends `backbone.` to every surviving key. The
`children` / `named_children` overrides expose the inner ResNet at path
`"backbone"` so the BN-buffer loader's descendant walk produces
`backbone.layer1.0.bn1` etc.

`pub struct DeepLabV3Head<T: Float>` (lines 224-233) owns `aspp` (the
`Aspp` utility from the sibling `aspp.rs` file), `conv_intermediate`
(`Conv2d(256, 256, 3x3, pad=1, bias=False)`), `bn_intermediate`
(`BatchNorm2d(256)`), and `classifier` (`Conv2d(256, num_classes, 1x1,
bias=True)`). `Module::forward` (lines 267-275) runs `aspp →
conv_intermediate → bn_intermediate → relu → classifier`. The 5-element
torchvision layout (with the ReLU at index 3 contributing no params) is
preserved in `named_parameters` via `aspp.<...>` / `conv_intermediate.<...>`
/ `bn_intermediate.<...>` / `classifier.<...>` keys.

`pub struct DeepLabV3<T: Float>` (lines 363-367) owns `backbone:
ResNet50Dilated<T>`, `head: DeepLabV3Head<T>`, and a `training: bool`. The
`Module::forward` impl (lines 401-419) captures `h_in, w_in`, runs the
backbone to get layer4 features (`[B, 2048, H/16, W/16]`), runs the head
(`[B, num_classes, H/16, W/16]`), and bilinear-upsamples to `[B,
num_classes, H, W]` via
`interpolate(.., InterpolateMode::Bilinear, false)`.

`pub fn deeplabv3_resnet50<T: Float>(num_classes: usize)` at line 487
returns `DeepLabV3::new(num_classes)` which calls
`DeepLabV3::with_atrous_rates(num_classes, (12, 24, 36))`.

`named_children` at the composite-model level returns the backbone at the
empty top-level path `""` (transparent — backbone's own
`named_parameters` already prefixes with `backbone.`) and the head at
`"head"`. The regression test `deeplabv3_named_descendants_no_leading_dot`
(lines 580-616) verifies that the descendant walk never yields a path
starting with `.`, locking the #1142 fix in the transparent-wrapper branch
of `Module::named_descendants_dyn`.

### Non-test production consumers

- `pub use deeplabv3::{DeepLabV3, DeepLabV3Head, ResNet50Dilated,
  deeplabv3_resnet50}` at
  `ferrotorch-vision/src/models/segmentation/mod.rs:22` and
  `ferrotorch-vision/src/models/mod.rs:40-43`.
- `default_registry()` in `ferrotorch-vision/src/models/registry.rs:315-318`
  binds the `"deeplabv3_resnet50"` registry entry, constructing the model
  via `super::segmentation::deeplabv3_resnet50::<f32>(num_classes)` inside
  the `maybe_load_pretrained` closure.

## Parity contract

`parity_ops = []`. DeepLabV3 composes `Conv2d`, `BatchNorm2d`, `relu`, the
`Aspp` utility (which composes the same primitives), `interpolate`
(bilinear), and the dilated ResNet-50 backbone — every primitive is covered
under its own parity entry.

Edge cases preserved versus torchvision:

- **`replace_stride_with_dilation=[False, True, True]`**: layer1/layer2
  unchanged, layer3 dilated by 2x with the `_make_layer` previous_dilation
  threading, layer4 dilated by 4x with the same threading. Output stride =
  16, not 32. Matches `deeplabv3.py:275`.
- **ASPP atrous rates default**: `(12, 24, 36)` for `deeplabv3_resnet50`
  (matches `deeplabv3.py:50`, the `DeepLabHead.__init__` default). The
  smaller `(6, 12, 18)` is reachable via `DeepLabV3::with_atrous_rates`.
- **Final upsample**: `align_corners=False` bilinear from `H/16, W/16` →
  `H, W`. Matches the implicit `_SimpleSegmentationModel` upsample at
  `torchvision/models/segmentation/_utils.py` (`F.interpolate(x,
  size=input_shape, mode="bilinear", align_corners=False)`).
- **Classifier bias**: the final 1x1 conv carries `bias=True`, matching
  torchvision's `nn.Conv2d(256, num_classes, 1)` at `deeplabv3.py:56`
  (default bias=True).

## Verification

Tests in `mod tests` in `deeplabv3.rs`:

- `test_deeplabv3_output_shape_{small,64x64}` — 32x32 and 64x64 inputs.
- `test_deeplabv3_batch_size_2` — batch dimension propagates correctly.
- `test_deeplabv3_custom_num_classes` — `num_classes=5`.
- `test_deeplabv3_named_parameter_prefixes` — `backbone.` and `head.`
  exist.
- `test_deeplabv3_param_count_sanity` — `np > 30_000_000`.
- `test_deeplabv3_train_eval_toggle`.
- `deeplabv3_named_descendants_no_leading_dot` — #1142 regression lock.

Smoke command:

```bash
cargo test -p ferrotorch-vision --lib segmentation::deeplabv3:: 2>&1 | tail -3
```

Expected: all tests pass; no `parity-sweep` ops to run.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct ResNet50Dilated<T: Float>` + `Module<T>` impl at `ResNet50Dilated in deeplabv3.rs`, calls `resnet50_dilated::<T>(1000, [false, true, true])` at `deeplabv3 in deeplabv3.rs` and filters `fc.*` at `deeplabv3 in deeplabv3.rs, 150, 158`; non-test consumer: `DeepLabV3::with_atrous_rates` constructs `ResNet50Dilated::new()` at `deeplabv3 in deeplabv3.rs`, invoked by `deeplabv3_resnet50` at `deeplabv3 in deeplabv3.rs` and the registry closure at `registry.rs`. |
| REQ-2 | SHIPPED | impl: `ResNet50Dilated::forward_layer4` at `deeplabv3 in deeplabv3.rs` calls `self.inner.forward_features(input)` and pulls the `"layer4"` entry; non-test consumer: `DeepLabV3::forward` invokes `self.backbone.forward_layer4(input)` at `deeplabv3 in deeplabv3.rs`. |
| REQ-3 | SHIPPED | impl: `pub struct DeepLabV3Head<T: Float>` at `DeepLabV3Head in deeplabv3.rs` + `Module<T>` impl at `deeplabv3 in deeplabv3.rs` (aspp → conv_intermediate → bn_intermediate → relu → classifier); non-test consumer: `DeepLabV3::with_atrous_rates` constructs `DeepLabV3Head::new(2048, num_classes, atrous_rates)` at `deeplabv3 in deeplabv3.rs`. |
| REQ-4 | SHIPPED | impl: `DeepLabV3Head::new(in_channels, num_classes, atrous_rates: (usize, usize, usize))` at `deeplabv3 in deeplabv3.rs`; non-test consumer: `DeepLabV3::with_atrous_rates(num_classes, atrous_rates)` at `deeplabv3 in deeplabv3.rs` forwards both args; the default `(12, 24, 36)` is bound at `DeepLabV3::new` (`deeplabv3 in deeplabv3.rs`) and ultimately invoked by `deeplabv3_resnet50` (`deeplabv3 in deeplabv3.rs`) reached via `registry.rs`. |
| REQ-5 | SHIPPED | impl: `pub struct DeepLabV3<T: Float>` at `DeepLabV3 in deeplabv3.rs` + `Module::forward` at `DeepLabV3 in deeplabv3.rs` runs backbone → head → bilinear upsample; non-test consumer: `deeplabv3_resnet50` constructor at `deeplabv3 in deeplabv3.rs` returns a `DeepLabV3<T>` directly to the registry caller `registry.rs`. |
| REQ-6 | SHIPPED | impl: `pub fn deeplabv3_resnet50<T: Float>(num_classes: usize) -> FerrotorchResult<DeepLabV3<T>>` at `deeplabv3 in deeplabv3.rs`; non-test consumer: `registry.rs` calls `super::segmentation::deeplabv3_resnet50::<f32>(num_classes)` inside the `default_registry()` `maybe_load_pretrained` closure. |
| REQ-7 | SHIPPED | impl: `Module::named_parameters` for `DeepLabV3 in deeplabv3.rs` emits `backbone.<...>` (via ResNet50Dilated's own prefix at `deeplabv3 in deeplabv3.rs`) and `head.<...>`; non-test consumer: `maybe_load_pretrained` at `named_parameters in registry.rs` calls `model.load_state_dict(&state_dict, false)` which walks `named_parameters` to copy tensors into the model. |
| REQ-8 | SHIPPED | impl: `children` / `named_children` overrides at every level (`Aspp::named_children` `named_children in aspp.rs`, `DeepLabV3Head::named_children` `deeplabv3 in deeplabv3.rs`, `ResNet50Dilated::named_children` `deeplabv3 in deeplabv3.rs`, `DeepLabV3::named_children` `deeplabv3 in deeplabv3.rs`); non-test consumer: `apply_bn_buffers_from_state_dict` at `registry.rs` walks `named_descendants_dyn()` on the live model and resolves `backbone.bn1.<...>`, `backbone.layer1.0.bn1.<...>`, etc. through these overrides. |
