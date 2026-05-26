# ferrotorch-vision — `models::mobilenet` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669 (working tree at /home/doll/pytorch)
baseline-torchvision: /home/doll/.local/lib/python3.13/site-packages/torchvision/models/mobilenetv3.py
upstream-paths:
  - /home/doll/.local/lib/python3.13/site-packages/torchvision/models/mobilenetv3.py
-->

## Summary

`ferrotorch-vision/src/models/mobilenet.rs` ships three architectures
end-to-end with full torchvision parameter-naming parity: MobileNetV2
(Sandler et al. 2018), MobileNetV3-Small, and MobileNetV3-Large (Howard
et al. 2019, with a dilated-backbone variant for LRASPP segmentation).
Phase 7 (#1007) rebuilt the V2 path from ReLU-only standard-Conv2d
placeholders into faithful reproductions of torchvision's reference;
Phase 7 also lands V3 with `HardSwish` activations, depthwise convs via
`Conv2d::new_full`, and `SqueezeExcitation` with `HardSigmoid` scale
activation. #1146 adds the dilated MobileNetV3-Large backbone for the
LRASPP segmentation head.

## Requirements

- REQ-1: A private `ConvBnAct<T: Float>` ships torchvision's
  `Conv2dNormActivation` — `Conv2d(bias=false) → BatchNorm2d → activation`
  with `ActivationKind ∈ {Relu, Relu6, HardSwish}` and optional `None`
  (linear bottleneck projection). `BatchNorm2d` eps/momentum are
  configurable per call site (V2 uses 1e-5 / 0.1; V3 uses different).
  `new_with_dilation` exposes the dilation kwarg needed for the LRASPP
  dilated backbone.
- REQ-2: A private `V2InvertedResidual<T: Float>` ships torchvision's V2
  inverted residual: optional 1×1 expand (None when `expand_ratio == 1`),
  3×3 depthwise (`groups=hidden`), 1×1 project (linear, no activation),
  BN after project, residual add when `stride == 1 && in_ch == out_ch`.
  Inner-`conv` Sequential indexing matches torchvision:
  `expand_ratio == 1` → `conv.{0,1,2}`;
  `expand_ratio > 1` → `conv.{0,1,2,3}`.
- REQ-3: `pub struct MobileNetV2<T: Float>` carries `features:
  Vec<Box<dyn Module<T>>>` (`features.0` = stem ConvBnAct,
  `features.<1..18>` = `V2InvertedResidual`s, `features.18` = head
  ConvBnAct), `avgpool: AdaptiveAvgPool2d`, `classifier: Linear(1280,
  num_classes)`. Param naming uses `features.<i>.<n>` and
  `classifier.1.<n>` to match torchvision.
- REQ-4: A private `V3InvertedResidual<T: Float>` ships torchvision's V3
  inverted residual: ConvBnAct expand (optional), ConvBnAct depthwise,
  optional SqueezeExcitation with `HardSigmoid` scale activation, no-act
  ConvBnAct project. Residual add when stride 1 + in == out.
- REQ-5: `pub struct MobileNetV3Small<T: Float>` and `pub struct
  MobileNetV3Large<T: Float>` ship the canonical V3 stacks. Both carry
  `features: Vec<Box<dyn Module<T>>>` (`features.0` = stem, ... =
  `V3InvertedResidual`s, `features.{12 V3-Small, 16 V3-Large}` = head
  ConvBnAct) and the two-Linear `classifier` head with HardSwish in
  between (params at `classifier.{0,3}.{weight,bias}`).
- REQ-6: `pub fn mobilenet_v2` / `pub fn mobilenet_v3_small` / `pub fn
  mobilenet_v3_large` are the canonical constructors.
- REQ-7: `pub fn mobilenet_v3_large_dilated` returns a
  `MobileNetV3LargeStaged<T>` exposing per-stage outputs and dilated
  `features.{12,13,14}` blocks (dilation=2 on the 3×3 depthwise). Used
  by the LRASPP segmentation backbone (#1146).
- REQ-8: `named_parameters` and `named_children` return torchvision-flat
  paths so a strict torchvision state dict loads without remap.
- REQ-9: Each top-level model implements `IntermediateFeatures<T>`
  exposing per-stage activations.

## Acceptance Criteria

- [x] AC-1: `mobilenet_v2::<f32>(1000)` constructs and forwards on
  `[1, 3, 224, 224]` returning `[1, 1000]`.
- [x] AC-2: `mobilenet_v3_small::<f32>(1000)` constructs and forwards.
- [x] AC-3: `mobilenet_v3_large::<f32>(1000)` constructs and forwards.
- [x] AC-4: `mobilenet_v3_large_dilated::<f32>(num_classes)` returns
  `MobileNetV3LargeStaged<T>` whose `features.{12,13,14}.conv.0` carry
  dilation=2 (LRASPP backbone path).
- [x] AC-5: `named_parameters` includes torchvision-shaped prefixes
  `features.0.{0,1}.weight`, `features.<i>.conv.<j>.<n>` (V2) /
  `features.<i>.block.<j>.<n>` (V3), `classifier.{0,1,3}.<n>`.
- [x] AC-6: Each top-level model satisfies
  `Module: train/eval` propagation through BN children.

## Architecture

`ConvBnAct<T: Float>` is the base unit (torchvision's
`Conv2dNormActivation`). Children index `0=conv`, `1=bn`; the activation
slot is parameter-free so it doesn't appear in `named_parameters`. The
`new_with_dilation` overload exposes the dilation kwarg the LRASPP
dilated backbone needs.

`V2InvertedResidual<T: Float>` flattens to a single `conv` Sequential
inside each block per torchvision:

```text
expand_ratio == 1: conv = [depthwise+BN+ReLU6, project_1x1, project_BN]   (3)
expand_ratio  > 1: conv = [expand+BN+ReLU6, depthwise+BN+ReLU6,
                           project_1x1, project_BN]                       (4)
```

`pub struct MobileNetV2<T>` stores `features` as a flat
`Vec<Box<dyn Module<T>>>` so the per-block `named_parameters` indexing
matches torchvision's outer `nn.Sequential` (`features.<i>.conv.<j>.<n>`).
The head is at `features.18`, the classifier head is at `classifier.1`.

`V3InvertedResidual<T>` wraps the inner `block` Sequential whose
sub-indices vary by config (some blocks omit the expand step or the SE
block). The SE block uses `HardSigmoid` for the scale activation per
torchvision's `partial(SElayer, scale_activation=nn.Hardsigmoid)`.

`pub struct MobileNetV3Large<T>` ships the full V3-Large stack and is
also exposed as `MobileNetV3LargeStaged<T>` for segmentation backbones.
The `_dilated` constructor sets `dilation=2` on the depthwise convs in
features 12, 13, 14 (per torchvision's
`mobilenet_v3_large(..., dilated=True)` path used by
`lraspp_mobilenet_v3_large`).

### Non-test production consumers

- `pub use mobilenet::{MobileNetV2, MobileNetV3Large, MobileNetV3Small,
  mobilenet_v2, mobilenet_v3_large, mobilenet_v3_large_dilated,
  mobilenet_v3_small}` at `ferrotorch-vision/src/models/mod.rs`.
- `default_registry()` registers `"mobilenet_v2"` and
  `"mobilenet_v3_small"` via `maybe_load_pretrained` at `registry.rs:231`
  and `:240`.
- `ferrotorch-vision/src/models/segmentation/lraspp.rs:52` imports
  `MobileNetV3Large, MobileNetV3LargeStaged` for the dilated LRASPP
  backbone (`lraspp_mobilenet_v3_large` registered at
  `registry.rs:322`).

## Parity contract

`parity_ops = []`. MobileNet composes `Conv2d`, `BatchNorm2d`, `Linear`,
`AdaptiveAvgPool2d`, `SqueezeExcitation`, the differentiable
`add`/`relu`/`relu6`/`hardswish`/`hardsigmoid`, plus the existing
`ferrotorch-nn` primitives. No new op surface.

Edge cases preserved versus torchvision:

- **Depthwise via `Conv2d::new_full(..., groups=hidden, ...)`**: the
  grouped path falls back to CPU im2col on CUDA (groups>1 fast path not
  enabled); preserves param count + element-wise output.
- **V2 BN `eps=1e-5, momentum=0.1`**: matches torchvision default.
- **V3 SE uses `HardSigmoid` scale activation** (not `Sigmoid`):
  `SqueezeExcitation::new_with_activations(..., HardSigmoid)` mirrors
  torchvision's `partial(SElayer, scale_activation=nn.Hardsigmoid)`.
- **V3 HardSwish vs ReLU per block**: the V3 config table drives which
  activation each block uses (`use_hs` flag).
- **`mobilenet_v3_large_dilated` `dilation=2` on blocks 12-14**:
  required for the LRASPP segmentation head's 1/16 output stride. Per
  torchvision `mobilenet_v3_large(dilated=True)` (#1146).
- **Linear bottleneck projection** (no activation after final 1×1):
  V2's `ConvBnAct(... act=None)` and V3's no-act project both preserve
  this paper-mandated property.

## Verification

Tests in `mod tests` in `mobilenet.rs` cover construction, forward
shape, parameter-count bands matching torchvision, named-parameter
layout, train/eval propagation, and Send+Sync.

Smoke command:

```bash
cargo test -p ferrotorch-vision --lib mobilenet:: 2>&1 | tail -3
```

Expected: all tests pass; no parity-sweep ops.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `struct ConvBnAct<T: Float>` + `Module<T>` impl + `new_with_dilation` in `mobilenet.rs`; non-test consumer: every `V2InvertedResidual::new` / `V3InvertedResidual::new` / `MobileNetV{2,3Small,3Large}::new` builds them in `mobilenet.rs`. |
| REQ-2 | SHIPPED | impl: `struct V2InvertedResidual<T: Float>` + `Module<T>` impl in `mobilenet.rs`; non-test consumer: `MobileNetV2::new` builds the 17 inverted-residual blocks in `mobilenet.rs`. |
| REQ-3 | SHIPPED | impl: `pub struct MobileNetV2<T: Float>` + `Module<T>` impl in `mobilenet.rs`; non-test consumer: `default_registry()` constructs it via `maybe_load_pretrained` at `registry.rs:231`. |
| REQ-4 | SHIPPED | impl: `struct V3InvertedResidual<T: Float>` + `Module<T>` impl in `mobilenet.rs`; non-test consumer: `MobileNetV3Small::new` / `MobileNetV3Large::new` build them in `mobilenet.rs`. |
| REQ-5 | SHIPPED | impl: `pub struct MobileNetV3Small<T: Float>` and `pub struct MobileNetV3Large<T: Float>` in `mobilenet.rs`; non-test consumer: `default_registry()` constructs `mobilenet_v3_small` at `registry.rs:240`; `MobileNetV3Large` is constructed by `MobileNetV3LargeStaged` for the LRASPP backbone in `segmentation/lraspp.rs`. |
| REQ-6 | SHIPPED | impl: `pub fn mobilenet_v2`, `pub fn mobilenet_v3_small`, `pub fn mobilenet_v3_large` in `mobilenet.rs`; non-test consumer: `default_registry()` invokes V2/V3-small (`registry.rs:233`, `:243`); V3-large flows through `lraspp_mobilenet_v3_large` (`registry.rs:325`). |
| REQ-7 | SHIPPED | impl: `pub fn mobilenet_v3_large_dilated` + `pub struct MobileNetV3LargeStaged<T: Float>` in `mobilenet.rs`; non-test consumer: `segmentation/lraspp.rs:52` imports `MobileNetV3LargeStaged` and `super::lraspp_mobilenet_v3_large` flows through `default_registry()` at `registry.rs:322`. |
| REQ-8 | SHIPPED | impl: `named_parameters` / `named_children` for `ConvBnAct`, `V2InvertedResidual`, `V3InvertedResidual`, and each top-level model in `mobilenet.rs`; non-test consumer: `load_state_dict(&state_dict, false)` + `apply_bn_buffers_from_state_dict(&model, &state_dict)` at `registry.rs:53` and `:62`. |
| REQ-9 | SHIPPED | impl: `impl IntermediateFeatures<T> for MobileNetV2<T>` and `for MobileNetV3Small<T>` in `mobilenet.rs`; non-test consumer: `feature_extractor.rs:232` imports `mobilenet_v2` for the production helper composition. |
