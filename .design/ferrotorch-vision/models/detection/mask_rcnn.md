# ferrotorch-vision — `detection::mask_rcnn` module

<!--
tier: 3-component
status: draft
baseline-pytorch: torchvision 0.26.0+cu130 (git 336d36e8db990a905498c73933e35231876e28bc)
upstream-paths:
  - /home/doll/.local/lib/python3.13/site-packages/torchvision/models/detection/mask_rcnn.py
  - /home/doll/.local/lib/python3.13/site-packages/torchvision/models/detection/roi_heads.py
-->

## Summary

`ferrotorch-vision/src/models/detection/mask_rcnn.rs` ships Mask R-CNN
with ResNet-50 + FPN backbone. Mirrors
`torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)`.
Extends `FasterRcnn` with a parallel mask branch (14×14 ROI Align →
4-conv FCN `MaskHead` → 2× deconv `MaskPredictor` →
`paste_masks_in_image`).

## Requirements

- REQ-1: `pub struct MaskHead<T>` mirrors
  `MaskRCNNHeads(layers=(256, 256, 256, 256), dilation=1)`: four
  `Conv2d(C, 256, k=3, p=1, bias=true)` blocks with `relu` between
  them.
- REQ-2: `pub struct MaskPredictor<T>` mirrors
  `MaskRCNNPredictor`: a `ConvTranspose2d(C, 256, k=2, s=2, p=0)`
  upsample + `relu`, then `Conv2d(256, num_classes, k=1)` to project
  to per-class mask logits.
- REQ-3: `pub struct MaskDetections<T>` extends `Detections` with
  `masks: Tensor<T>` (`[N_det, 1, H_img, W_img]` after paste).
- REQ-4: `pub struct MaskRcnn<T>` composes a `FasterRcnn<T>` plus
  the mask-specific `MaskHead<T>` + `MaskPredictor<T>` — no
  backbone/FPN/RPN/box-head duplication.
- REQ-5: `pub fn MaskRcnn::forward` runs Faster R-CNN's detection
  forward, then re-runs ROI Align at output_size=14 on the FPN
  features (sharing the backbone+FPN call from
  `forward_backbone`/`forward_fpn`), then mask-head → mask-predictor
  → `postprocess_masks(..., paste=true)`.
- REQ-6: `Module::forward` returns first-image post-NMS scores
  (1-D `[N_det]`), matching the harness convention.
- REQ-7: `pub fn maskrcnn_resnet50_fpn(num_classes)` returns
  `MaskRcnn::new(num_classes)`.
- REQ-8: `pub fn MaskHead::named_parameters` / `pub fn
  MaskPredictor::named_parameters` emit torchvision-aligned keys
  (`conv{1..4}.weight`, `conv{1..4}.bias`, `deconv.{weight,bias}`,
  `conv_logits.{weight,bias}`).

## Acceptance Criteria

- [x] AC-1: `maskrcnn_resnet50_fpn::<f32>(91)` constructs.
- [x] AC-2: `num_parameters()` matches the hub registry pin
  (44,456,562 params with FPN biases).
- [x] AC-3: `named_parameters()` exposes `faster_rcnn.*`,
  `mask_head.*`, `mask_predictor.*` prefixes.
- [x] AC-4: `MaskHead::<f32>::new(256).forward(&randn([N, 256, 14, 14]))`
  returns `[N, 256, 14, 14]`.
- [x] AC-5: `MaskPredictor::<f32>::new(256, 91)
  .forward(&randn([N, 256, 14, 14]))` returns `[N, 91, 28, 28]`.
- [x] AC-6: `MaskRcnn::forward([1, 3, 64, 64])` returns one
  `MaskDetections` with `boxes.shape() == [N, 4]`,
  `scores.shape() == [N]`, `labels.len() == N`,
  `masks.shape() == [N, 1, 64, 64]`.

## Architecture

`pub struct MaskHead<T: Float>` is a four-layer FCN over 256-channel
features. All four convs are `3×3, pad=1, bias=true`. Forward is
`conv → relu` four times in sequence.

`pub struct MaskPredictor<T: Float>` has a `ConvTranspose2d(in, 256,
k=2, s=2, p=0)` (doubles spatial resolution from 14 to 28) and a
`Conv2d(256, num_classes, k=1)` projection. Forward is
`deconv → relu → conv_logits` (no relu after the final conv —
the logits feed into the postprocess sigmoid).

`pub struct MaskRcnn<T: Float>` composes:

- `faster_rcnn: FasterRcnn<T>` (delegated for backbone + FPN + RPN
  + box head + box postprocess).
- `mask_head: MaskHead<T>` (14×14 spatial).
- `mask_predictor: MaskPredictor<T>` (14 → 28 upsample +
  per-class projection).

`Self::forward` runs `self.faster_rcnn.forward(images)?` once to get
detection boxes, then re-derives backbone + FPN features via
`forward_backbone` / `forward_fpn` (so the mask branch can reuse
them — no double-pass), then performs ROI Align at output_size=14 on
the FPN features keyed by the detection boxes, runs the mask
head + predictor, and finally invokes
`postprocess_masks(..., paste=true)` to paste per-detection masks
back into the image-space canvas.

`Module<T>::forward` returns `dets[0].scores` (the first-image
post-NMS scores) — the registry / harness shim.

### Non-test production consumers

- `pub use MaskRcnn, MaskHead, MaskPredictor, MaskDetections,
  maskrcnn_resnet50_fpn` at
  `ferrotorch-vision/src/models/detection/mod.rs:39` and
  `ferrotorch-vision/src/lib.rs:21`.
- `register_model("maskrcnn_resnet50_fpn", ...)` at
  `ferrotorch-vision/src/models/registry.rs:285`.
- Pretrained-loading test at
  `ferrotorch-hub/tests/pretrained_loading.rs::test_pretrained_maskrcnn_resnet50_fpn`
  (production end-to-end consumer of the pinned weights).

## Parity contract

`parity_ops = []`. End-to-end parity is exercised by the
pretrained-weight harness which loads
`huggingface.co/ferrotorch/maskrcnn_resnet50_fpn/resolve/main/model.safetensors`
(SHA-256 `dc472afa1ba8bb321c142b05c7f4a6ca20ee0ae191087d4e8f1030af7cfb3d2e`,
44,456,562 params).

Numerical / structural edge cases preserved:

- **Mask paste-back via `expand_masks` + `paste_masks_in_image`** —
  matches torchvision's geometry exactly (see `postprocess_masks`
  contract in `roi_heads_postprocess.md`).
- **Box head + box postprocess from FasterRcnn** — Mask R-CNN
  composes, does not duplicate.
- **`MaskRCNNPredictor` deconv kernel=2, stride=2, padding=0** —
  doubles 14 → 28 exactly.

## Verification

Tests in `mod tests in mask_rcnn.rs`:

- `test_mask_rcnn_constructs`
- `test_mask_rcnn_param_count_ballpark`
- `test_mask_rcnn_named_params_prefixes`
- `test_mask_head_shapes`
- `test_mask_predictor_shapes`
- `test_mask_rcnn_forward_output_structure`
- `test_mask_rcnn_train_eval`

No parity-sweep ops. Smoke command:

```bash
cargo test -p ferrotorch-vision --lib detection::mask_rcnn:: 2>&1 | tail -3
```

Expected: all `detection::mask_rcnn::tests` pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct MaskHead<T>` + `Self::new` + `Self::forward` in `mask_rcnn.rs`; non-test consumer: `MaskRcnn::new` in `mask_rcnn.rs` calls `MaskHead::new(256)?` and stores it as the `mask_head` field. |
| REQ-2 | SHIPPED | impl: `pub struct MaskPredictor<T>` + `Self::new` + `Self::forward` in `mask_rcnn.rs`; non-test consumer: `MaskRcnn::new` in `mask_rcnn.rs` calls `MaskPredictor::new(256, num_classes)?`. |
| REQ-3 | SHIPPED | impl: `pub struct MaskDetections<T>` in `mask_rcnn.rs`; non-test consumer: `MaskRcnn::forward` returns `Vec<MaskDetections<T>>`, consumed by the registry-registered closure at `ferrotorch-vision/src/models/registry.rs:285` via `Module::forward`. |
| REQ-4 | SHIPPED | impl: `pub struct MaskRcnn<T>` (fields `faster_rcnn`, `mask_head`, `mask_predictor`) + `Self::new` in `mask_rcnn.rs`; non-test consumer: `register_model("maskrcnn_resnet50_fpn", ...)` at `ferrotorch-vision/src/models/registry.rs:285`. |
| REQ-5 | SHIPPED | impl: `pub fn MaskRcnn::forward` in `mask_rcnn.rs`; non-test consumer: `Module::forward` impl in `mask_rcnn.rs` invokes it; the registry closure at `ferrotorch-vision/src/models/registry.rs:285` calls into it. |
| REQ-6 | SHIPPED | impl: `impl<T> Module<T> for MaskRcnn<T>::forward` in `mask_rcnn.rs` returns first-image scores; non-test consumer: registered as `ModelConstructor<f32>` at `ModelConstructor in ferrotorch-vision/src/models/registry.rs` — the harness/shell consumes it via `Module::forward`. |
| REQ-7 | SHIPPED | impl: `pub fn maskrcnn_resnet50_fpn` in `mask_rcnn.rs`; non-test consumer: `register_model("maskrcnn_resnet50_fpn", ...)` at `ferrotorch-vision/src/models/registry.rs:285` calls it inside the closure. |
| REQ-8 | SHIPPED | impl: `pub fn MaskHead::named_parameters` + `pub fn MaskPredictor::named_parameters` in `mask_rcnn.rs`; non-test consumer: `MaskRcnn::named_parameters` (the Module impl in `mask_rcnn.rs`) prefixes them with `mask_head.` / `mask_predictor.` and exposes them to state-dict ingest. The pretrained loader at `ferrotorch-hub/tests/pretrained_loading.rs::test_pretrained_maskrcnn_resnet50_fpn` is the production consumer. |
