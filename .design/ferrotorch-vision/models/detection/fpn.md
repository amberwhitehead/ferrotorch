# ferrotorch-vision — `detection::fpn` module

<!--
tier: 3-component
status: draft
baseline-pytorch: torchvision 0.26.0+cu130 (git 336d36e8db990a905498c73933e35231876e28bc)
upstream-paths:
  - /home/doll/.local/lib/python3.13/site-packages/torchvision/ops/feature_pyramid_network.py
  - /home/doll/.local/lib/python3.13/site-packages/torchvision/models/detection/backbone_utils.py
-->

## Summary

`ferrotorch-vision/src/models/detection/fpn.rs` ships the Feature
Pyramid Network used by Faster R-CNN, Mask R-CNN, and Keypoint R-CNN.
Mirrors `torchvision.ops.FeaturePyramidNetwork` composed with the
`LastLevelMaxPool` extra block (the FPN configuration that
`BackboneWithFPN(..., extra_blocks=LastLevelMaxPool())` produces for
the `_resnet_fpn_extractor` family). Output keys are `"p2"..."p6"`.

## Requirements

- REQ-1: `pub struct FeaturePyramidNetwork<T>` owns four lateral
  `1×1` convs (`lateral2..lateral5`), four `3×3` output convs
  (`output2..output5`), and a `MaxPool2d` for the P6 extra
  block.
- REQ-2: All lateral and output convolutions include a bias.
  Matches torchvision's `Conv2dNormActivation(..., norm_layer=None)`
  which leaves `nn.Conv2d` at its default `bias=True`. The
  `bias=false` variant was the root cause of #1141; this REQ is
  the regression contract.
- REQ-3: `pub const FPN_OUT_CHANNELS: usize = 256;` — every FPN
  output level has 256 channels.
- REQ-4: `pub fn FeaturePyramidNetwork::new` builds the default
  ResNet-50 FPN with input channels `[256, 512, 1024, 2048]` for
  layers 1..4.
- REQ-5: `pub fn forward` consumes a `HashMap<String, Tensor<T>>`
  keyed by `"layer1".."layer4"` (ResNet's `forward_features` output
  keys) and returns a `HashMap<String, Tensor<T>>` keyed by
  `"p2".."p6"`.
- REQ-6: Top-down path: lateral5 (`P5`) is upsampled to `C4`'s
  spatial size via nearest-neighbor and added to lateral4 (forming
  `P4_inner`); same pattern produces `P3_inner` and `P2_inner`. The
  final P2..P5 are produced by the `3×3` output convs applied to
  the `_inner` tensors.
- REQ-7: P6 is `LastLevelMaxPool` — a `kernel=1, stride=2,
  padding=0` sub-sample on P5, **NOT** a `3×3` maxpool. With
  `kernel=1` it equals P5 sub-sampled at `(2i, 2j)` exactly (no max
  reduction). #1141 regression guard.
- REQ-8: `pub fn named_parameters` exposes the eight conv subtrees
  with keys `lateral2..lateral5` and `output2..output5` (plus
  `.weight` and `.bias` per conv), matching the torchvision
  state-dict layout.

## Acceptance Criteria

- [x] AC-1: `FeaturePyramidNetwork::<f32>::new()` constructs.
- [x] AC-2: `forward({"layer1", "layer2", "layer3", "layer4"})`
  returns the five `"p2".."p6"` keys.
- [x] AC-3: Every FPN output level has 256 channels
  (`FPN_OUT_CHANNELS`).
- [x] AC-4: For a 64-pixel input (ResNet strides /4..32), the
  resulting spatial sizes are P2=16×16, P3=8×8, P4=4×4, P5=2×2,
  P6=1×1.
- [x] AC-5: Every lateral/output conv exposes a `*.bias` named
  parameter.
- [x] AC-6: P6 equals P5 sub-sampled at stride 2 (kernel=1,
  padding=0) — `p6[c, h, w] == p5[c, 2h, 2w]` exactly.
- [x] AC-7: Total parameter count is in `(3M, 4M)` (4 lateral +
  4 output convs at 256 channels).

## Architecture

`pub struct FeaturePyramidNetwork<T: Float>` in `fpn.rs` owns eight
`Conv2d` modules plus one `MaxPool2d`. The constructor builds the
lateral convs with `kernel=(1,1), padding=0, bias=true` and the
output convs with `kernel=(3,3), padding=1, bias=true`. The P6
sub-sampler is `MaxPool2d::new([1, 1], [2, 2], [0, 0])` — the
`kernel=1` is what makes the operation an integer-index
sub-sample rather than a windowed max.

`Self::forward` runs the laterals first, then the top-down path
(`P5_inner = lat5; P4_inner = upsample(P5_inner) + lat4; ...`), then
the output convs. P6 is finally produced as
`Module::<T>::forward(&self.pool_p6, &p5)`. The upsample uses
`InterpolateMode::Nearest` (matching torchvision's
`F.interpolate(..., mode='nearest')`).

`Self::named_parameters` emits keys as `lateral{i}.weight`,
`lateral{i}.bias`, `output{i}.weight`, `output{i}.bias` for
`i ∈ {2, 3, 4, 5}`. Exact match for the
`fpn.inner_blocks.{i}.0.{weight,bias}` /
`fpn.layer_blocks.{i}.0.{weight,bias}` keys upstream produces is
handled by the pin script's key remapping in
`scripts/pin_pretrained_weights.py`.

### Non-test production consumers

- `pub use FeaturePyramidNetwork` at
  `ferrotorch-vision/src/models/detection/mod.rs:34` and
  `ferrotorch-vision/src/lib.rs:21`.
- `pub use FPN_OUT_CHANNELS` at the same paths; consumed by
  `ferrotorch-vision/src/models/detection/retinanet.rs` (sizing
  the RetinaNet head trunk) and
  `ferrotorch-vision/src/models/detection/fcos.rs` (sizing the
  FCOS head trunk).
- `use crate::models::detection::fpn::FeaturePyramidNetwork` at
  `ferrotorch-vision/src/models/detection/faster_rcnn.rs`:
  `FasterRcnn::new` calls `FeaturePyramidNetwork::new()?` and
  stores it as the `fpn` field; `FasterRcnn::forward_fpn` (a public
  helper consumed by Mask R-CNN and Keypoint R-CNN) delegates to
  `self.fpn.forward(...)`.

## Parity contract

`parity_ops = []`. FPN is composition over `Conv2d` + nearest-neighbor
interpolate + `MaxPool2d`, all of which are covered by their own
parity contracts. The whole-pipeline parity is exercised by the
pretrained-loading harness (`ferrotorch-hub/tests/pretrained_loading.rs`
`test_pretrained_fasterrcnn_resnet50_fpn`).

Numerical / structural edge cases preserved:

- **Lateral + output conv biases.** `Conv2d::new(..., bias=true)` —
  the #1141 root cause. Tested by
  `test_fpn_named_params_include_biases`.
- **P6 is sub-sample, not max-pool.** `kernel=1, stride=2`
  produces `p6[h, w] == p5[2h, 2w]`. Tested by
  `test_fpn_p6_uses_stride_2_subsample_not_3x3_pool`.
- **Top-down add order: upsample-then-add.** The lateral feeds the
  current level; the higher level is upsampled first and added on
  top. Mirrors torchvision exactly.
- **Nearest-neighbor upsample.** Lossless integer-stride upsample
  (no bilinear smoothing). Matches
  `F.interpolate(..., mode='nearest')`.

## Verification

Tests in `mod tests in fpn.rs`:

- `test_fpn_output_keys`
- `test_fpn_output_channels`
- `test_fpn_spatial_sizes_batch1`
- `test_fpn_named_params_include_biases` (#1141 regression)
- `test_fpn_p6_uses_stride_2_subsample_not_3x3_pool` (#1141
  regression)
- `test_fpn_parameter_count`

No parity-sweep ops. Smoke command:

```bash
cargo test -p ferrotorch-vision --lib detection::fpn:: 2>&1 | tail -3
```

Expected: 6 tests passed.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct FeaturePyramidNetwork<T>` + `Self::new` in `fpn.rs`; non-test consumer: `FasterRcnn::new` at `ferrotorch-vision/src/models/detection/faster_rcnn.rs:232` calls `FeaturePyramidNetwork::new()?` and stores the result. |
| REQ-2 | SHIPPED | impl: `Conv2d::new(..., true)` calls (bias=true) in `Self::new` in `fpn.rs`; non-test consumer: same `FasterRcnn::new` path; the bias parameters are loaded by `scripts/pin_pretrained_weights.py` and consumed during the pretrained-weight-loading test at `ferrotorch-hub/tests/pretrained_loading.rs::test_pretrained_fasterrcnn_resnet50_fpn`. |
| REQ-3 | SHIPPED | impl: `pub const FPN_OUT_CHANNELS: usize = 256;` in `fpn.rs`; non-test consumer: `RetinaNet::new` at `new in ferrotorch-vision/src/models/detection/retinanet.rs` passes it as the head trunk input channels. |
| REQ-4 | SHIPPED | impl: `Self::new` in `fpn.rs` (`in_channels = [256, 512, 1024, 2048]`); non-test consumer: `FasterRcnn::new` at `ferrotorch-vision/src/models/detection/faster_rcnn.rs:232`. |
| REQ-5 | SHIPPED | impl: `pub fn forward` in `fpn.rs` (reads `"layer1".."layer4"`, writes `"p2".."p6"`); non-test consumer: `FasterRcnn::forward` at `ferrotorch-vision/src/models/detection/faster_rcnn.rs:305` calls `self.fpn.forward(&backbone_features)`. |
| REQ-6 | SHIPPED | impl: top-down body (`lat5 → upsample → add lat4 → output4`) in `Self::forward` in `fpn.rs`; non-test consumer: `FasterRcnn::forward_fpn` at `ferrotorch-vision/src/models/detection/faster_rcnn.rs:279` delegates to `self.fpn.forward(backbone_features)`. |
| REQ-7 | SHIPPED | impl: `MaxPool2d::new([1, 1], [2, 2], [0, 0])` field `pool_p6` in `fpn.rs`; non-test consumer: `Self::forward` invokes `Module::<T>::forward(&self.pool_p6, &p5)` to materialise P6, which is then consumed by every consumer of `forward` (FasterRcnn body — at `ferrotorch-vision/src/models/detection/faster_rcnn.rs:305`). |
| REQ-8 | SHIPPED | impl: `pub fn named_parameters` in `fpn.rs` emits `lateral{i}` and `output{i}` keyed entries; non-test consumer: `FasterRcnn::named_parameters` at `ferrotorch-vision/src/models/detection/faster_rcnn.rs:506` prefixes the FPN keys with `fpn.` for state-dict ingest. |
