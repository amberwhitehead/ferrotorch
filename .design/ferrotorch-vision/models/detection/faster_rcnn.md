# ferrotorch-vision — `detection::faster_rcnn` module

<!--
tier: 3-component
status: draft
baseline-pytorch: torchvision 0.26.0+cu130 (git 336d36e8db990a905498c73933e35231876e28bc)
upstream-paths:
  - /home/doll/.local/lib/python3.13/site-packages/torchvision/models/detection/faster_rcnn.py
  - /home/doll/.local/lib/python3.13/site-packages/torchvision/models/detection/roi_heads.py
  - /home/doll/.local/lib/python3.13/site-packages/torchvision/models/detection/generalized_rcnn.py
-->

## Summary

`ferrotorch-vision/src/models/detection/faster_rcnn.rs` ships Faster
R-CNN with ResNet-50 + FPN backbone. Mirrors
`torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)`.
End-to-end pipeline: backbone → FPN → RPN → ROI-Align → TwoMlpHead
detection head → `RoIHeads.postprocess_detections`.

## Requirements

- REQ-1: `pub struct FasterRcnn<T>` owns `ResNet<T>`,
  `FeaturePyramidNetwork<T>`, `Rpn<T>`, `TwoMlpHead<T>`, plus
  `num_classes`, `roi_output_size`, `roi_spatial_scales`, and a
  training flag.
- REQ-2: `pub struct TwoMlpHead<T>` mirrors
  `TwoMLPHead` + `FastRCNNPredictor`: `fc6: in*P*P → 1024`,
  `fc7: 1024 → 1024`, `cls_score: 1024 → num_classes`,
  `bbox_pred: 1024 → num_classes * 4`.
- REQ-3: `pub struct Detections<T>` holds `boxes: [N_det, 4]`,
  `scores: [N_det]`, `labels: Vec<usize>` (background dropped).
- REQ-4: `pub fn FasterRcnn::new` constructs ResNet-50 + FPN + RPN
  + 7×7 ROI head with default torchvision sizes
  (representation=1024, roi=7, spatial_scales `[1/4..1/64]`).
- REQ-5: `pub fn FasterRcnn::forward(images: &Tensor)` runs the full
  pipeline and returns `Vec<Detections<T>>` of length B.
- REQ-6: `assign_fpn_levels` (private helper) mirrors torchvision's
  `LevelMapper`: `level = floor(k0 + log2(sqrt(area) /
  canonical_size) + 1e-6)`, clamped to `[min_level, max_level]`.
  The `LEVEL_MAPPER_EPS = 1e-6` matches `LevelMapper.eps`.
- REQ-7: `pub fn forward_backbone` + `pub fn forward_fpn` expose
  intermediate backbone / FPN outputs so Mask R-CNN and Keypoint
  R-CNN can reuse them without running the full pipeline twice.
- REQ-8: `Module::forward` returns `dets[0].scores` (1-D `[N_det]`)
  — matches the `model(img)[0]["scores"]` convention used by the
  #1139 verification harness.
- REQ-9: ROI Align uses `aligned=false` (legacy mode, matching
  pretrained-weight semantics) — see #1145.
- REQ-10: `pub fn fasterrcnn_resnet50_fpn` returns `FasterRcnn::new`
  with the user-supplied `num_classes` (default COCO: 91).
- REQ-11: `Module::children` exposes the ResNet backbone (so the
  Phase 2 BN-buffer loader walks into it).

## Acceptance Criteria

- [x] AC-1: `fasterrcnn_resnet50_fpn::<f32>(91)` constructs.
- [x] AC-2: `num_parameters()` lies in `(40M, 80M)` — torchvision
  pretrained reports 41,755,286; ferrotorch with BN-affine params
  reports `41_810_455` (matches the hub registry pin).
- [x] AC-3: `named_parameters()` emits keys with prefixes
  `backbone.`, `fpn.`, `rpn.`, `head.`.
- [x] AC-4: `forward([1, 3, 64, 64])` returns `Vec<Detections>`
  with one entry; the per-entry shapes obey
  `boxes: [N, 4]`, `scores: [N]`, `labels.len() == N`, no
  background labels.
- [x] AC-5: `forward([2, 3, 64, 64])` returns a length-2
  `Vec<Detections>`.
- [x] AC-6: `train()` / `eval()` flip `is_training()`.
- [x] AC-7: `TwoMlpHead::<f32>::new(7, 256, 1024, 91)
  .forward(&randn([4, 256, 7, 7]))` returns
  `(cls: [4, 91], bbox: [4, 91*4])`.

## Architecture

`pub struct FasterRcnn<T: Float>` is the top-level container.
`Self::forward` runs:

1. Validate `[B, 3, H, W]` input shape.
2. `self.backbone.forward_features` produces
   `layer1..layer4`.
3. `self.fpn.forward` produces `p2..p6`.
4. Per-image loop:
   - Slice each FPN level to the current image.
   - Call `self.rpn.forward(&single_refs, &RpnConfig::default_eval([img_h, img_w]))`.
   - If no proposals, push empty detections and continue.
   - `assign_fpn_levels(proposals, k0=4, canonical=224, [2,6])`.
   - For each FPN level, gather indices that landed on this
     level, run `roi_align_with_aligned(feat, boxes, (7,7),
     scale, sampling_ratio=2, aligned=false)`, store per-ROI
     slices.
   - Assemble `[N, 256, 7, 7]`, run `self.head.forward(...)`,
     hand the result to `postprocess_detections`.

`pub struct TwoMlpHead<T: Float>` mirrors
`TwoMLPHead` (`fc6 → relu → fc7 → relu`) followed by
`FastRCNNPredictor` (`cls_score`, `bbox_pred` linears). The
flatten step at the top of `Self::forward` reshapes
`[N, C, P, P] → [N, C*P*P]`.

`assign_fpn_levels` is the private helper used by both
`FasterRcnn::forward` and (via `assign_fpn_levels_keypoint` in the
keypoint module) `KeypointRcnn::forward`. The `LEVEL_MAPPER_EPS =
1e-6` nudge mirrors torchvision exactly.

`Module<T>::forward` exposes the post-NMS scores of the first
image as a 1-D `[N_det]` tensor. This matches the
#1139 verification harness, which calls
`model(img)[0]["scores"]` upstream.

### Non-test production consumers

- `pub use FasterRcnn, TwoMlpHead, fasterrcnn_resnet50_fpn` at
  `ferrotorch-vision/src/models/detection/mod.rs` and
  `ferrotorch-vision/src/lib.rs:21`.
- `register_model("fasterrcnn_resnet50_fpn", ...)` at
  `ferrotorch-vision/src/models/registry.rs:270`.
- `MaskRcnn::new` at
  `ferrotorch-vision/src/models/detection/mask_rcnn.rs:247` calls
  `fasterrcnn_resnet50_fpn::<T>(num_classes)?` and stores the
  result as its `faster_rcnn` field; `MaskRcnn::forward` consumes
  `self.faster_rcnn.forward(images)?`.
- `KeypointRcnn::new` at
  `ferrotorch-vision/src/models/detection/keypoint_rcnn.rs:305`
  same call path.
- `pub use ... FasterRcnn ...` re-export at the hub level (the
  pretrained-loading test
  `ferrotorch-hub/tests/pretrained_loading.rs::test_pretrained_fasterrcnn_resnet50_fpn`
  is the production end-to-end consumer of the pinned weights).

## Parity contract

`parity_ops = []`. End-to-end parity is exercised by the
pretrained-weight harness in
`ferrotorch-hub/tests/pretrained_loading.rs::test_pretrained_fasterrcnn_resnet50_fpn`,
which loads
`huggingface.co/ferrotorch/fasterrcnn_resnet50_fpn/resolve/main/model.safetensors`
(SHA-256 `1d8a19e81e91f5ce86ce5a65127dda566d6ae1fb7e2e64596d1ecf373ed06494`,
41,810,455 params) and verifies forward output matches torchvision's
pretrained output on COCO image 87038 (#1141 reference image).

Numerical / structural edge cases preserved:

- **`roi_align(aligned=false)`** — legacy mode matches pretrained
  weights (#1145).
- **`LevelMapper.eps = 1e-6`** — numerical nudge for ROI level
  assignment on integer area boundaries.
- **Per-dim anchor strides from padded image size** — see
  `anchor_utils::generate_anchors_for_image` (#1141).
- **`Module::forward` returns first-image scores** — the
  detector's `Vec<Detections>` API is the canonical path;
  `Module::forward` is the registry / harness shim.

## Verification

Tests in `mod tests in faster_rcnn.rs`:

- `test_faster_rcnn_constructs`
- `test_faster_rcnn_param_count_ballpark`
- `test_faster_rcnn_named_params_prefixes`
- `test_faster_rcnn_forward_output_structure`
- `test_faster_rcnn_two_images_batch`
- `test_faster_rcnn_train_eval`
- `test_two_mlp_head_shapes`

No parity-sweep ops. Smoke command:

```bash
cargo test -p ferrotorch-vision --lib detection::faster_rcnn:: 2>&1 | tail -3
```

Expected: 7 tests passed.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct FasterRcnn<T>` in `faster_rcnn.rs`; non-test consumer: `register_model("fasterrcnn_resnet50_fpn", ...)` at `ferrotorch-vision/src/models/registry.rs:270` invokes the factory which returns it. |
| REQ-2 | SHIPPED | impl: `pub struct TwoMlpHead<T>` + `Self::new` + `Self::forward` in `faster_rcnn.rs`; non-test consumer: `FasterRcnn::new` at `ferrotorch-vision/src/models/detection/faster_rcnn.rs:234` calls `TwoMlpHead::new(7, 256, 1024, num_classes)?` and stores it as the `head` field. |
| REQ-3 | SHIPPED | impl: `pub struct Detections<T>` in `faster_rcnn.rs`; non-test consumer: `MaskRcnn::forward` at `ferrotorch-vision/src/models/detection/mask_rcnn.rs` consumes the `Vec<Detections<T>>` returned by `self.faster_rcnn.forward(images)?`. |
| REQ-4 | SHIPPED | impl: `pub fn FasterRcnn::new` in `faster_rcnn.rs`; non-test consumer: `pub fn fasterrcnn_resnet50_fpn` in same file delegates to it; that factory is registered at `ferrotorch-vision/src/models/registry.rs:270`. |
| REQ-5 | SHIPPED | impl: `pub fn FasterRcnn::forward` in `faster_rcnn.rs`; non-test consumer: `MaskRcnn::forward` at `ferrotorch-vision/src/models/detection/mask_rcnn.rs` calls `self.faster_rcnn.forward(images)?`. |
| REQ-6 | SHIPPED | impl: `fn assign_fpn_levels` in `faster_rcnn.rs`; non-test consumer: `FasterRcnn::forward` at `ferrotorch-vision/src/models/detection/faster_rcnn.rs:341` calls `assign_fpn_levels(&proposals, 4.0, 224.0, 2, 6)?`. |
| REQ-7 | SHIPPED | impl: `pub fn FasterRcnn::forward_backbone` + `pub fn FasterRcnn::forward_fpn` in `faster_rcnn.rs`; non-test consumer: `KeypointRcnn::forward` at `ferrotorch-vision/src/models/detection/keypoint_rcnn.rs:339` calls both methods to reuse backbone/FPN features for the keypoint branch. |
| REQ-8 | SHIPPED | impl: `impl<T> Module<T> for FasterRcnn<T>::forward` in `faster_rcnn.rs` (returns `dets[0].scores.clone()`); non-test consumer: `register_model("fasterrcnn_resnet50_fpn", ...)` at `ferrotorch-vision/src/models/registry.rs:270` registers it under the `ModelConstructor<f32>` trait, which is invoked via `Module::forward` by harness callers. |
| REQ-9 | SHIPPED | impl: `roi_align_with_aligned(... false)` call in `FasterRcnn::forward` in `faster_rcnn.rs` (`aligned=false` argument); non-test consumer: same `FasterRcnn::forward` path consumed by `MaskRcnn::forward` and the registry factory. |
| REQ-10 | SHIPPED | impl: `pub fn fasterrcnn_resnet50_fpn` in `faster_rcnn.rs`; non-test consumer: `register_model("fasterrcnn_resnet50_fpn", ...)` at `ferrotorch-vision/src/models/registry.rs:270` calls it inside the closure. |
| REQ-11 | SHIPPED | impl: `fn children`/`fn named_children` in `impl<T> Module<T> for FasterRcnn<T>` in `faster_rcnn.rs` returns `vec![&self.backbone]`; non-test consumer: the BN-buffer loader at `ferrotorch-vision/src/bn_buffer_loader.rs` walks `Module::children()` to find ResNet BN buffers, and is called via `ferrotorch-hub` pretrained-loading paths. |
