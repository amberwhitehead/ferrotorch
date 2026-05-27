# ferrotorch-vision — `detection::keypoint_rcnn` module

<!--
tier: 3-component
status: draft
baseline-pytorch: torchvision 0.26.0+cu130 (git 336d36e8db990a905498c73933e35231876e28bc)
upstream-paths:
  - /home/doll/.local/lib/python3.13/site-packages/torchvision/models/detection/keypoint_rcnn.py
  - /home/doll/.local/lib/python3.13/site-packages/torchvision/models/detection/roi_heads.py
-->

## Summary

`ferrotorch-vision/src/models/detection/keypoint_rcnn.rs` ships
Keypoint R-CNN with ResNet-50 + FPN backbone. Mirrors
`torchvision.models.detection.keypointrcnn_resnet50_fpn`
(`KeypointRCNN_ResNet50_FPN_Weights.COCO_V1`). Extends `FasterRcnn`
with a parallel keypoint branch (14×14 ROI Align on p2..p5 →
8-conv FCN `KeypointHead` → `ConvTranspose2d` + bilinear 2× upsample
`KeypointPredictor` → `heatmaps_to_keypoints` decoder).

## Requirements

- REQ-1: `pub const KEYPOINT_RCNN_NUM_KEYPOINTS: usize = 17;` —
  COCO person keypoints.
- REQ-2: `pub const KEYPOINT_RCNN_NUM_CLASSES: usize = 2;` —
  background + person (the COCO pretrained box predictor).
- REQ-3: `pub struct KeypointHead<T>` mirrors
  `KeypointRCNNHeads((512, 512, 512, 512, 512, 512, 512, 512))`:
  eight `Conv2d(3×3, p=1, bias=true)` blocks (first projects
  `in→512`, the rest are `512→512`). `named_parameters` emits
  `conv{0,2,4,6,8,10,12,14}` matching torchvision's
  `nn.Sequential` interleaved-ReLU index layout.
- REQ-4: `pub struct KeypointPredictor<T>` mirrors
  `KeypointRCNNPredictor`: a single
  `ConvTranspose2d(in, num_keypoints, k=4, s=2, p=1)` (14 → 28)
  followed by a parameter-free `F.interpolate(scale_factor=2,
  mode='bilinear', align_corners=false)` (28 → 56).
- REQ-5: `pub struct KeypointDetections<T>` extends `Detections`
  with `keypoints: [N_det, 17, 3]` (xyv image-space pixel coords,
  visibility flag always 1.0) and `keypoint_scores: [N_det, 17]`
  (raw heatmap logit at argmax).
- REQ-6: `pub struct KeypointRcnn<T>` composes a `FasterRcnn<T>`
  (configured with `num_classes=2`) plus the keypoint-specific
  `KeypointHead<T>` + `KeypointPredictor<T>`. No body code
  duplication.
- REQ-7: `pub fn KeypointRcnn::forward` runs Faster R-CNN's
  detection forward, then re-uses backbone+FPN features
  (`forward_backbone`/`forward_fpn`) to feed ROI Align at
  output_size=14 on p2..p5 (NOT p6 — keypoint head uses
  `MultiScaleRoIAlign(featmap_names=["0","1","2","3"])`), then
  mask-head → mask-predictor → `heatmaps_to_keypoints` decoder.
- REQ-8: `assign_fpn_levels_keypoint` clamps to `[k_min=2,
  k_max=5]` (p6 excluded for the keypoint head).
- REQ-9: `pub fn heatmaps_to_keypoints` decodes per-detection
  heatmap argmaxes to image-space using the Heckbert (1990)
  `c = d + 0.5` continuous-coordinate convention and the
  `width / ceil(width)` correction factor — matches torchvision
  exactly.
- REQ-10: ROI Align uses `aligned=false` (legacy mode) per #1145.
- REQ-11: `Module::forward` returns first-image scores (1-D
  `[N_det]`); inherent `KeypointRcnn::forward` returns
  `Vec<KeypointDetections<T>>`.
- REQ-12: `pub fn keypointrcnn_resnet50_fpn` returns
  `KeypointRcnn::new(2, 17)` — the COCO defaults.

## Acceptance Criteria

- [x] AC-1: `keypointrcnn_resnet50_fpn::<f32>()` constructs.
- [x] AC-2: `num_parameters()` lies in `(55M, 75M)` — torchvision
  pretrained reports 59M with the keypoint branch.
- [x] AC-3: `named_parameters()` exposes `faster_rcnn.*`,
  `keypoint_head.*`, `keypoint_predictor.*` prefixes, including
  the `keypoint_head.conv{0..14}.{weight,bias}` Sequential indices
  and `keypoint_predictor.kps_score_lowres.{weight,bias}`.
- [x] AC-4: `KeypointHead::<f32>::new(256).forward(&randn([2,
  256, 14, 14]))` returns `[2, 512, 14, 14]`.
- [x] AC-5: `KeypointPredictor::<f32>::new(512, 17).forward(&randn([2,
  512, 14, 14]))` returns `[2, 17, 56, 56]`.
- [x] AC-6: `heatmaps_to_keypoints` on a single-peak synthetic
  heatmap returns the argmax location with Heckbert correction.
- [x] AC-7: `KeypointRcnn::forward([1, 3, 64, 64])` returns one
  `KeypointDetections` with `keypoints: [N, 17, 3]` and
  `keypoint_scores: [N, 17]`.
- [x] AC-8: `<KeypointRcnn<f32> as Module<f32>>::forward` returns a
  1-D tensor.

## Architecture

`pub struct KeypointHead<T: Float>` is an 8-layer FCN over 256/512
features. The first conv projects `in_channels → 512`; the rest are
`512 → 512`. All convs are `3×3, pad=1, bias=true`. Forward is
`conv → relu` eight times; the trailing relu matches torchvision
(`KeypointRCNNHeads.forward` ends with a relu).

`pub struct KeypointPredictor<T: Float>` has a single
`ConvTranspose2d` (the `kps_score_lowres` learned upsample) and uses
a parameter-free `F.interpolate(scale_factor=2, mode='bilinear',
align_corners=false)` to go from 28×28 to 56×56. The post-deconv
upsample has no state_dict keys (matches torchvision exactly).

`pub struct KeypointRcnn<T: Float>` composes:

- `faster_rcnn: FasterRcnn<T>` (with `num_classes=2`).
- `keypoint_head: KeypointHead<T>` (output 512ch, 14×14).
- `keypoint_predictor: KeypointPredictor<T>` (output `num_keypoints`
  channels, 56×56).

`Self::forward` runs the detection pipeline first, then for each
image with at least one detection re-derives backbone + FPN features
via `forward_backbone` / `forward_fpn`, performs ROI Align on
p2..p5 keyed by the detection boxes (assignment via
`assign_fpn_levels_keypoint(..., k_min=2, k_max=5)`), feeds the
result through the keypoint head + predictor (producing
`[N_det, 17, 56, 56]` heatmaps), and finally decodes via
`heatmaps_to_keypoints` (per-ROI bicubic upsample to box size,
per-keypoint argmax + Heckbert continuous coords + raw heatmap
logit as score).

`pub fn heatmaps_to_keypoints` is the public decoder. It is
written generically over `T: Float` and matches torchvision's
non-tracing path (the tracing path uses a different formulation but
the same Heckbert correction).

### Non-test production consumers

- `pub use KEYPOINT_RCNN_NUM_CLASSES, KEYPOINT_RCNN_NUM_KEYPOINTS,
  KeypointDetections, KeypointHead, KeypointPredictor,
  KeypointRcnn, keypointrcnn_resnet50_fpn` at
  `ferrotorch-vision/src/models/detection/mod.rs:35`.
- `register_model("keypointrcnn_resnet50_fpn", ...)` at
  `ferrotorch-vision/src/models/registry.rs:378`.
- The `keypointrcnn_resnet50_fpn` hub entry at
  `ferrotorch-hub/src/registry.rs:385` (`#1145`) is the production
  pretrained-weight binding.

## Parity contract

`parity_ops = []`. End-to-end keypoint parity is exercised against
torchvision's `KeypointRCNN_ResNet50_FPN_Weights.COCO_V1` checkpoint
loaded via the pin script. The hub registry pin (SHA
`73e282340493d58731dc08314df5f4f483fd537f55b3bb2fc188c17cfd922dfb`)
is the contract; #1145 pinned it after the FPN-bias regression
landed.

Numerical / structural edge cases preserved:

- **`MultiScaleRoIAlign(featmap_names=["0","1","2","3"])`** — only
  four FPN levels (p2..p5). The `LevelMapper` clamps to
  `[k_min=2, k_max=5]`; a box that would map to p6 by the area
  heuristic reuses p5. (#1145)
- **`aligned=false`** — legacy ROI Align mode for pretrained
  weights. (#1145)
- **Heckbert `c = d + 0.5` continuous-coord convention** plus
  `width_correction = width / ceil(width)`. Matches torchvision's
  `heatmaps_to_keypoints` non-tracing path exactly.
- **Visibility flag is always 1.0** — torchvision's
  `keypoint_rcnn.py` doesn't gate per-keypoint visibility at
  inference time. Per-keypoint score (raw logit at argmax) is the
  visibility proxy.
- **`KeypointRCNNPredictor` deconv kernel=4, stride=2, padding=1**
  doubles 14 → 28; the bilinear 2× post-step doubles to 56 → 56.

## Verification

Tests in `mod tests in keypoint_rcnn.rs`:

- `test_keypoint_rcnn_constructs`
- `test_keypoint_rcnn_param_count_ballpark`
- `test_keypoint_rcnn_named_params_prefixes`
- `test_keypoint_head_shapes`
- `test_keypoint_predictor_shapes`
- `test_heatmaps_to_keypoints_argmax_location`
- `test_keypoint_rcnn_forward_output_structure`
- `test_keypoint_rcnn_module_forward_returns_1d_scores`
- `test_keypoint_rcnn_train_eval`

No parity-sweep ops. Smoke command:

```bash
cargo test -p ferrotorch-vision --lib detection::keypoint_rcnn:: 2>&1 | tail -3
```

Expected: 9 tests passed.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub const KEYPOINT_RCNN_NUM_KEYPOINTS: usize = 17;` in `keypoint_rcnn.rs`; non-test consumer: `pub fn keypointrcnn_resnet50_fpn` in same file passes it to `KeypointRcnn::new`. |
| REQ-2 | SHIPPED | impl: `pub const KEYPOINT_RCNN_NUM_CLASSES: usize = 2;` in `keypoint_rcnn.rs`; non-test consumer: same `keypointrcnn_resnet50_fpn` factory. |
| REQ-3 | SHIPPED | impl: `pub struct KeypointHead<T>` + `Self::new` + `Self::named_parameters` in `keypoint_rcnn.rs`; non-test consumer: `KeypointRcnn::new` in same file calls `KeypointHead::new(256)?` and stores it as the `keypoint_head` field. |
| REQ-4 | SHIPPED | impl: `pub struct KeypointPredictor<T>` + `Self::new` + `Self::forward` in `keypoint_rcnn.rs`; non-test consumer: `KeypointRcnn::new` in same file calls `KeypointPredictor::new(512, num_keypoints)?`. |
| REQ-5 | SHIPPED | impl: `pub struct KeypointDetections<T>` in `keypoint_rcnn.rs`; non-test consumer: `KeypointRcnn::forward` returns `Vec<KeypointDetections<T>>` — `register_model("keypointrcnn_resnet50_fpn", ...)` at `ferrotorch-vision/src/models/registry.rs:378` calls into it via `Module::forward`. |
| REQ-6 | SHIPPED | impl: `pub struct KeypointRcnn<T>` (fields `faster_rcnn`, `keypoint_head`, `keypoint_predictor`) + `Self::new` in `keypoint_rcnn.rs`; non-test consumer: `register_model("keypointrcnn_resnet50_fpn", ...)` at `ferrotorch-vision/src/models/registry.rs:378`. |
| REQ-7 | SHIPPED | impl: `pub fn KeypointRcnn::forward` in `keypoint_rcnn.rs`; non-test consumer: `impl<T> Module<T> for KeypointRcnn<T>::forward` invokes it; the registry closure at `ferrotorch-vision/src/models/registry.rs:378` calls into it via `Module::forward`. |
| REQ-8 | SHIPPED | impl: `fn assign_fpn_levels_keypoint(..., k_max=5)` in `keypoint_rcnn.rs`; non-test consumer: `KeypointRcnn::forward` in same file at the call site `let roi_levels = assign_fpn_levels_keypoint(&det.boxes, 4.0, 224.0, 2, 5)?`. |
| REQ-9 | SHIPPED | impl: `pub fn heatmaps_to_keypoints` in `keypoint_rcnn.rs`; non-test consumer: `KeypointRcnn::forward` in same file at the call site `let (keypoints, keypoint_scores) = heatmaps_to_keypoints(&kp_heatmaps, &det.boxes, self.num_keypoints)?`. |
| REQ-10 | SHIPPED | impl: `roi_align_with_aligned(..., false)` call in `KeypointRcnn::forward` in `keypoint_rcnn.rs`; non-test consumer: same `Self::forward` reachable via `Module::forward` from `ferrotorch-vision/src/models/registry.rs:378`. |
| REQ-11 | SHIPPED | impl: `impl<T> Module<T> for KeypointRcnn<T>::forward` in `keypoint_rcnn.rs` returns 1-D `[N_det]` scores; non-test consumer: same registry closure. |
| REQ-12 | SHIPPED | impl: `pub fn keypointrcnn_resnet50_fpn` in `keypoint_rcnn.rs`; non-test consumer: `register_model("keypointrcnn_resnet50_fpn", ...)` at `ferrotorch-vision/src/models/registry.rs:378` calls it inside the closure. |
