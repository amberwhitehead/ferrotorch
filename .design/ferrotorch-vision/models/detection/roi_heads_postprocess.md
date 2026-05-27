# ferrotorch-vision — `detection::roi_heads_postprocess` module

<!--
tier: 3-component
status: draft
baseline-pytorch: torchvision 0.26.0+cu130 (git 336d36e8db990a905498c73933e35231876e28bc)
upstream-paths:
  - /home/doll/.local/lib/python3.13/site-packages/torchvision/models/detection/roi_heads.py
  - /home/doll/.local/lib/python3.13/site-packages/torchvision/models/detection/_utils.py
-->

## Summary

`ferrotorch-vision/src/models/detection/roi_heads_postprocess.rs`
ships the per-image postprocess pipelines shared by Faster R-CNN and
Mask R-CNN: per-class box decode + score-threshold + per-class NMS
+ cross-class top-K (`postprocess_detections`), and mask
sigmoid + class-select + optional paste-back
(`postprocess_masks`). Mirrors
`RoIHeads.postprocess_detections` + `maskrcnn_inference` +
`paste_masks_in_image` from upstream.

## Requirements

- REQ-1: `pub const ROI_SCORE_THRESH: f64 = 0.05;` — matches
  `FasterRCNN(score_thresh=0.05)`.
- REQ-2: `pub const ROI_NMS_THRESH: f64 = 0.5;` — matches
  `FasterRCNN(nms_thresh=0.5)`.
- REQ-3: `pub const ROI_DETECTIONS_PER_IMG: usize = 100;` — matches
  `FasterRCNN(detections_per_img=100)`.
- REQ-4: `pub const ROI_MIN_BOX_SIDE: f64 = 1e-2;` — matches
  `box_ops.remove_small_boxes(boxes, min_size=1e-2)`.
- REQ-5: `pub const ROI_BOX_CODER_WEIGHTS: (f64, f64, f64, f64) =
  (10.0, 10.0, 5.0, 5.0);` — the FasterRCNN detection-head box-coder
  weights.
- REQ-6: `pub const ROI_BBOX_XFORM_CLIP: f64 = log(1000/16) ≈
  4.135;` — one-sided `max=` clamp on `dw`/`dh`.
- REQ-7: `pub fn decode_per_class` decodes `[N, num_classes * 4]`
  deltas against `[N, 4]` proposals → `[N, num_classes, 4]` xyxy
  boxes, applying the one-sided `dw`/`dh` clamp.
- REQ-8: `pub fn postprocess_detections` runs the full FasterRCNN
  postprocess: softmax → decode → clip → drop background (class 0)
  → score threshold → small-box filter → per-class batched_nms →
  cross-class top-K. Returns a `PostprocessedDetections<T>`
  (boxes / scores / labels).
- REQ-9: `pub fn postprocess_masks` runs the Mask R-CNN mask
  postprocess: sigmoid → class-select (per-detection channel)
  → optional `expand_masks` (pad-by-1) + `paste_masks_in_image`
  with `expand_boxes` truncation and bilinear resize.
- REQ-10: `paste=false` short-circuits after sigmoid + class-select,
  returning `[N_det, 1, mask_h, mask_w]` — used by the #1139
  verification harness which patches the outer
  `GeneralizedRCNNTransform.postprocess` to identity.

## Acceptance Criteria

- [x] AC-1: `decode_per_class` on zero deltas reproduces the
  proposals per class.
- [x] AC-2: `decode_per_class` with `dx == wx` shifts the box centre
  by exactly one width.
- [x] AC-3: Large positive `dw` is one-side clamped to `log(1000/16)`;
  large negative `dw` passes through.
- [x] AC-4: `postprocess_detections` with two heavily-overlapping
  foreground-class proposals keeps the higher-scoring one and drops
  the other.
- [x] AC-5: `postprocess_detections` returns exactly N detections
  for N well-separated foreground proposals (no spurious drops).
- [x] AC-6: `postprocess_detections` drops detections below
  `score_thresh = 0.05`.
- [x] AC-7: `postprocess_masks(paste=true)` returns `[N_det, 1,
  H_img, W_img]` and inside-box values are ~1 while outside-box
  values are exactly 0.
- [x] AC-8: `postprocess_masks(paste=false)` returns `[N_det, 1,
  mask_h, mask_w]` containing the picked class channel.

## Architecture

`pub fn decode_per_class<T: Float>` in `roi_heads_postprocess.rs`
is the per-class box decoder. For each (proposal, class) pair, it
recovers the proposal's `(cx, cy, w, h)`, applies
`(dx, dy, dw, dh) / weights`, one-side clamps `dw/dh` to
`bbox_xform_clip`, and decodes to xyxy via the exponentiated
predicted width/height.

`pub fn postprocess_detections` is the FasterRCNN postprocess. It
softmaxes over class logits, decodes per-class boxes, clips to image
bounds, drops the background class (index 0), gathers candidates
above `score_thresh` and the `min_side` filter, then calls
`batched_nms` keyed on the (1-indexed) class label. The final slice
is `kept[..ROI_DETECTIONS_PER_IMG]` — `batched_nms` already returns
indices in descending-score order, so the slice is exactly
torchvision's `keep[: self.detections_per_img]`.

`pub fn postprocess_masks` is the Mask R-CNN mask postprocess. It
sigmoids the per-class mask logits, picks the channel matching each
detection's predicted label, and either:

- Returns the `[N_det, 1, mh, mw]` tensor directly (`paste=false`),
- Or runs the torchvision `expand_masks` (pad by 1) and
  `paste_masks_in_image` (`expand_boxes` scaling, integer-truncated
  box, bilinear resize, image-space crop) (`paste=true`).

The `paste=true` path mirrors `paste_mask_in_image` exactly,
including the `expand_boxes` `scale = (M + 2*padding) / M` rescale
around the box centre, the `.to(int64)` truncation toward zero
(`f64::trunc as i64`), and the `int(box[2] - box[0] + 1)` size
formula.

### Non-test production consumers

- `use crate::models::detection::roi_heads_postprocess::{
  PostprocessedDetections, postprocess_detections}` at
  `ferrotorch-vision/src/models/detection/faster_rcnn.rs:34`.
  `FasterRcnn::forward` invokes `postprocess_detections(...)` at
  `ferrotorch-vision/src/models/detection/faster_rcnn.rs:434`.
- `use crate::models::detection::roi_heads_postprocess::postprocess_masks`
  at `ferrotorch-vision/src/models/detection/mask_rcnn.rs:31`.

## Parity contract

`parity_ops = []`. Postprocess is geometry + sigmoid + softmax +
batched_nms — each underlying primitive has its own parity
contract. End-to-end FasterRCNN/MaskRCNN parity is exercised by
`ferrotorch-hub/tests/pretrained_loading.rs`.

Numerical / structural edge cases preserved:

- **NaN scores are dropped.** `s.partial_cmp(&score_thresh) !=
  Some(Greater)` matches `torch.where(scores > score_thresh)` —
  NaN fails the comparison and is dropped.
- **`>=` for the side filter.** `box_ops.remove_small_boxes` uses
  `(ws >= min_size) & (hs >= min_size)`.
- **Background class (index 0) is never emitted.** The flatten loop
  starts at `c = 1`.
- **`detections_per_img = 100` after cross-class NMS.** Strict cap
  on detections per image.
- **One-sided `bbox_xform_clip = log(1000/16)`.** `dw/dh` get a
  `max=` clamp; negatives pass through unchanged.
- **`expand_masks` pad = 1.** Adds a 1-pixel border before paste so
  bilinear resize samples slightly outside the predicted boundary.

## Verification

Tests in `mod tests in roi_heads_postprocess.rs`:

- `decode_per_class_identity_zero_deltas`
- `decode_per_class_weights_match_torchvision`
- `decode_per_class_dw_one_sided_clamp`
- `postprocess_nms_drops_overlap`
- `postprocess_strong_signal_survives_full_pipeline`
- `postprocess_score_thresh_drops_low_confidence`
- `mask_paste_shape_and_extent`
- `mask_no_paste_returns_pre_paste_tensor`

No parity-sweep ops. Smoke command:

```bash
cargo test -p ferrotorch-vision --lib detection::roi_heads_postprocess:: 2>&1 | tail -3
```

Expected: 8 tests passed.

### torchvision-oracle parity (R-CHAR-3)

Hand-constructed unit tests above guard structure; the oracle-derived
parity test asserts byte-level agreement with a *live* torchvision
0.26.0+cu130 call. In `ferrotorch-vision/tests/roi_heads_postprocess_torchvision_oracle.rs`:

- `decode_per_class_matches_torchvision_box_coder` — `decode_per_class`
  vs `BoxCoder.decode` (`_utils.py:162`).
- `postprocess_detections_matches_torchvision_roiheads` — full pipeline
  vs `RoIHeads.postprocess_detections` (`roi_heads.py:680`), including
  the exact `batched_nms` descending-score ordering + `[: detections_per_img]`
  slice (`det_labels == [2, 1, 1, 2]`).
- `postprocess_masks_no_paste_matches_maskrcnn_inference` —
  `postprocess_masks(paste=false)` vs `maskrcnn_inference`
  (`roi_heads.py:56`).

The expected constants are emitted by the frozen reproduction script in
that file's module doc-comment; they are NOT re-derived from the
ferrotorch formula (so the test cannot tautologically pass).

```bash
cargo test -p ferrotorch-vision --test roi_heads_postprocess_torchvision_oracle 2>&1 | tail -3
```

Expected: 3 tests passed.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub const ROI_SCORE_THRESH: f64 = 0.05;` in `roi_heads_postprocess.rs`; non-test consumer: `postprocess_detections` body uses it as the score gate, invoked by `FasterRcnn::forward` at `ferrotorch-vision/src/models/detection/faster_rcnn.rs:434`. |
| REQ-2 | SHIPPED | impl: `pub const ROI_NMS_THRESH: f64 = 0.5;` in `roi_heads_postprocess.rs`; non-test consumer: same `postprocess_detections` body, invoked by `FasterRcnn::forward`. |
| REQ-3 | SHIPPED | impl: `pub const ROI_DETECTIONS_PER_IMG: usize = 100;` in `roi_heads_postprocess.rs`; non-test consumer: `postprocess_detections` truncation at the end of the pipeline. |
| REQ-4 | SHIPPED | impl: `pub const ROI_MIN_BOX_SIDE: f64 = 1e-2;` in `roi_heads_postprocess.rs`; non-test consumer: `postprocess_detections` small-box filter, invoked by `FasterRcnn::forward`. |
| REQ-5 | SHIPPED | impl: `pub const ROI_BOX_CODER_WEIGHTS: (f64, f64, f64, f64) = (10.0, 10.0, 5.0, 5.0);` in `roi_heads_postprocess.rs`; non-test consumer: `postprocess_detections` passes it to `decode_per_class`, invoked by `FasterRcnn::forward`. |
| REQ-6 | SHIPPED | impl: `pub const ROI_BBOX_XFORM_CLIP: f64 = 4.135_166_556_742_356;` in `roi_heads_postprocess.rs`; non-test consumer: passed by `postprocess_detections` to `decode_per_class`, invoked by `FasterRcnn::forward`. |
| REQ-7 | SHIPPED | impl: `pub fn decode_per_class` in `roi_heads_postprocess.rs`; non-test consumer: `postprocess_detections` calls it; that function is called by `FasterRcnn::forward` at `ferrotorch-vision/src/models/detection/faster_rcnn.rs:434`. |
| REQ-8 | SHIPPED | impl: `pub fn postprocess_detections` in `roi_heads_postprocess.rs`; non-test consumer: `FasterRcnn::forward` at `ferrotorch-vision/src/models/detection/faster_rcnn.rs:434` calls it once per image. |
| REQ-9 | SHIPPED | impl: `pub fn postprocess_masks` in `roi_heads_postprocess.rs`; non-test consumer: `MaskRcnn::forward` at `ferrotorch-vision/src/models/detection/mask_rcnn.rs:31` imports it; the call site is `postprocess_masks::<T>(...)` inside `MaskRcnn::forward`. |
| REQ-10 | SHIPPED | impl: `paste=false` short-circuit branch in `postprocess_masks` in `roi_heads_postprocess.rs`; non-test consumer: same import path — `MaskRcnn::forward` (and the #1139 verification harness which calls `postprocess_masks(..., paste=false)`). |
