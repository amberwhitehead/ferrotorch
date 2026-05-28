# ferrotorch-vision — `detection::rpn` module

<!--
tier: 3-component
status: draft
baseline-pytorch: torchvision 0.26.0+cu130 (git 336d36e8db990a905498c73933e35231876e28bc)
upstream-paths:
  - /home/doll/.local/lib/python3.13/site-packages/torchvision/models/detection/rpn.py
  - /home/doll/.local/lib/python3.13/site-packages/torchvision/models/detection/_utils.py
-->

## Summary

`ferrotorch-vision/src/models/detection/rpn.rs` ships the Region
Proposal Network (RPN). Mirrors
`torchvision.models.detection.rpn.RegionProposalNetwork` plus
`torchvision.models.detection.rpn.RPNHead`. Produces a flat
`[N_proposals, 4]` proposal tensor (xyxy pixel coords) from a list of
FPN feature maps for each image in the batch.

## Requirements

- REQ-1: `pub struct RpnHead<T>` owns a `3×3` same-pad conv, a `1×1`
  objectness conv (`num_anchors` channels), and a `1×1` bbox-delta
  conv (`num_anchors * 4` channels). All biases enabled.
- REQ-2: `pub fn RpnHead::forward_level` runs `conv → relu →
  (cls_logits, bbox_pred)` and returns `([B, A, H, W],
  [B, A*4, H, W])`.
- REQ-3: `pub struct RpnConfig` carries `pre_nms_top_n`,
  `post_nms_top_n`, `nms_thresh`, `min_size`, `score_thresh`,
  `image_size`. `pub fn RpnConfig::default_eval` returns the
  torchvision eval defaults: `(1000, 1000, 0.7, 1e-3, 0.0,
  image_size)`.
- REQ-4: `pub struct Rpn<T>` owns one `RpnHead` and one
  `AnchorGenerator` (the default FasterRCNN generator, 3 anchors per
  cell at 5 levels).
- REQ-5: `pub fn Rpn::forward` runs the full proposal pipeline:
  per-level head → sigmoid scores → **per-level** pre-NMS top-K (NOT
  global) → decode → clip → small-box filter → score threshold →
  per-level batched NMS → post-NMS top-K.
- REQ-6: Anchor generation uses `generate_anchors_for_image` (per-dim
  strides derived from `cfg.image_size`, not canonical
  `[4,8,16,32,64]`). Critical for non-64-aligned padded image sizes
  (#1141).
- REQ-7: Per-level batched NMS via
  `crate::ops::batched_nms` keyed on the FPN-level id — proposals
  on different levels never suppress each other.
- REQ-8: `pub fn named_parameters` exposes the head subtree with
  keys `conv.{weight,bias}`, `cls_logits.{weight,bias}`,
  `bbox_pred.{weight,bias}`, matching torchvision's
  `rpn.head.{conv,cls_logits,bbox_pred}.{weight,bias}`.

## Acceptance Criteria

- [x] AC-1: `RpnHead::<f32>::new(256, 3).forward_level(&x)` on
  `x: [1, 256, 8, 8]` returns logits `[1, 3, 8, 8]` and deltas
  `[1, 12, 8, 8]`.
- [x] AC-2: `Rpn::<f32>::new(256)` constructs with 5 FPN levels,
  3 anchors per cell.
- [x] AC-3: `Rpn::forward(&[p2..p6], &RpnConfig::default_eval([H, W]))`
  returns a `[N, 4]` tensor.
- [x] AC-4: All returned proposals satisfy
  `0 ≤ x1, x2 ≤ image_w` and `0 ≤ y1, y2 ≤ image_h`.

## Architecture

`pub struct RpnHead<T: Float>` in `rpn.rs` owns three `Conv2d`
modules. `Self::forward_level` is the canonical per-level forward
(used by `Rpn::forward`'s outer loop). `Self::parameters` /
`Self::named_parameters` flatten the conv subtrees into the
torchvision state-dict layout.

`pub struct Rpn<T: Float>` owns a `RpnHead` and an
`AnchorGenerator`. The constructor pins the anchor generator to the
FasterRCNN default (5 levels, 3 anchors per cell).
`Self::forward` runs:

1. Generate per-dim anchors from `cfg.image_size`.
2. For each FPN level: head → sigmoid scores → collect deltas;
   record `level_offsets` for per-level top-K bookkeeping.
3. Per-level pre-NMS top-K: sort by descending score, take
   `pre_nms_top_n` per level, concat. Critical: global top-K would
   bias toward large-anchor levels and miss small objects.
4. Decode selected deltas against selected anchors (intermediate
   `f64` tensor for numerical stability).
5. Clip to image, drop small boxes, apply score threshold.
6. Per-level batched NMS via `batched_nms` keyed on the FPN-level
   id captured at top-K time.
7. Cross-level post-NMS top-K (`post_nms_top_n`).

The `B==1` guard is documented at the top of `Self::forward` —
proposal generation is per-image (this is also how torchvision
runs the RPN). Multi-batch images are processed by the outer
detector's per-batch loop.

### Non-test production consumers

- `pub use Rpn, RpnConfig, RpnHead` at
  `ferrotorch-vision/src/models/detection/mod.rs:48` and
  `ferrotorch-vision/src/lib.rs:22`.
- `use crate::models::detection::rpn::{Rpn, RpnConfig}` at
  `ferrotorch-vision/src/models/detection/faster_rcnn.rs`.
  `FasterRcnn::new` calls `Rpn::new(256)?` (storing it as the `rpn`
  field) and `FasterRcnn::forward` invokes
  `self.rpn.forward(&single_refs, &rpn_cfg)` in the per-image loop.

## Parity contract

`parity_ops = []`. The RPN composes `Conv2d`, `relu`, sigmoid,
`AnchorGenerator`, `decode_boxes`, `clip_boxes_to_image`,
`remove_small_boxes`, and `batched_nms` — all covered by their own
parity tests where applicable. End-to-end RPN parity is exercised by
the pretrained-loading harness
(`ferrotorch-hub/tests/pretrained_loading.rs::test_pretrained_fasterrcnn_resnet50_fpn`).

Numerical / structural edge cases preserved:

- **Per-level top-K, not global.** Torchvision's
  `_get_top_n_idx` picks `pre_nms_top_n` per level. Global top-K
  would bias against small objects (which only have proposals at
  the fine FPN levels).
- **`>= score_thresh` (not `>`).** Backwards-compat comment in
  torchvision's `filter_proposals`.
- **Per-level batched NMS.** Proposals on different FPN levels
  never cross-suppress (different scale characteristics).
- **`min_size = 1e-3`.** Drops degenerate sub-pixel boxes.
- **Sigmoid on `cls_logits`.** RPN uses sigmoid (one-vs-rest),
  not softmax.

## Verification

Tests in `mod tests in rpn.rs`:

- `test_rpn_head_output_shapes`
- `test_rpn_forward_returns_proposals`
- `test_rpn_proposals_within_image_bounds`

No parity-sweep ops. Smoke command:

```bash
cargo test -p ferrotorch-vision --lib detection::rpn:: 2>&1 | tail -3
```

Expected: 3 tests passed.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct RpnHead` + `Self::new` in `rpn.rs`; non-test consumer: `Rpn::new` at `RpnHead in ferrotorch-vision/src/models/detection/rpn.rs` calls `RpnHead::new(in_channels, 3)?` and stores it as the `head` field, which is then exercised by `FasterRcnn::forward` via `self.rpn.forward(...)` at `forward in ferrotorch-vision/src/models/detection/faster_rcnn.rs`. |
| REQ-2 | SHIPPED | impl: `pub fn RpnHead::forward_level` in `rpn.rs`; non-test consumer: `Rpn::forward` at `RpnHead in ferrotorch-vision/src/models/detection/rpn.rs` invokes `self.head.forward_level(feat)` per FPN level. |
| REQ-3 | SHIPPED | impl: `pub struct RpnConfig` + `RpnConfig::default_eval` in `rpn.rs`; non-test consumer: `FasterRcnn::forward` at `ferrotorch-vision/src/models/detection/faster_rcnn.rs:324` calls `RpnConfig::default_eval([img_h, img_w])`. |
| REQ-4 | SHIPPED | impl: `pub struct Rpn<T>` + `Self::new` in `rpn.rs`; non-test consumer: `FasterRcnn::new` at `ferrotorch-vision/src/models/detection/faster_rcnn.rs:233` calls `Rpn::new(256)?` and stores the result. |
| REQ-5 | SHIPPED | impl: `pub fn Rpn::forward` body in `rpn.rs` (top-K → decode → clip → filter → NMS → post-NMS); non-test consumer: `FasterRcnn::forward` at `ferrotorch-vision/src/models/detection/faster_rcnn.rs:325` calls `self.rpn.forward(&single_refs, &rpn_cfg)`. |
| REQ-6 | SHIPPED | impl: `self.anchor_gen.generate_anchors_for_image(&fm_sizes, (img_h, img_w))` at `ferrotorch-vision/src/models/detection/rpn.rs:193`; non-test consumer: same `FasterRcnn::forward` path. |
| REQ-7 | SHIPPED | impl: `batched_nms::<f64>(&nms_boxes_t, &nms_scores_t, &nms_levels, cfg.nms_thresh)?` at `ferrotorch-vision/src/models/detection/rpn.rs:334`; non-test consumer: same `FasterRcnn::forward` path. |
| REQ-8 | SHIPPED | impl: `pub fn RpnHead::named_parameters` in `rpn.rs`; non-test consumer: `Rpn::named_parameters` in `rpn.rs` prefixes the keys with `head.`, which `FasterRcnn::named_parameters` at `ferrotorch-vision/src/models/detection/faster_rcnn.rs:509` further prefixes with `rpn.` for state-dict ingest. |
