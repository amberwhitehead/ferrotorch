# ferrotorch-vision — `detection::retinanet` module

<!--
tier: 3-component
status: draft
baseline-pytorch: torchvision 0.26.0+cu130 (git 336d36e8db990a905498c73933e35231876e28bc)
upstream-paths:
  - /home/doll/.local/lib/python3.13/site-packages/torchvision/models/detection/retinanet.py
  - /home/doll/.local/lib/python3.13/site-packages/torchvision/ops/feature_pyramid_network.py
-->

## Summary

`ferrotorch-vision/src/models/detection/retinanet.rs` ships the
RetinaNet single-stage detector with ResNet-50 + FPN(P3-P7) backbone.
Mirrors `torchvision.models.detection.retinanet_resnet50_fpn`
(`RetinaNet_ResNet50_FPN_Weights.COCO_V1`). Distinct from
FasterRCNN: anchor-based, no RPN, no ROI Align, per-class sigmoid
scoring (focal-loss-trained); LastLevelP6P7 (P6 = `3x3 stride-2
conv` on P5, P7 = same on ReLU(P6) — NOT max-pool).

## Requirements

- REQ-1: `pub const RETINANET_NUM_ANCHORS_PER_LOC: usize = 9;` —
  3 sizes × 3 aspect ratios per spatial location.
- REQ-2: `pub const RETINANET_ASPECT_RATIOS: [f64; 3] = [0.5, 1.0,
  2.0];` and `RETINANET_BASE_SIZES: [f64; 5] = [32, 64, 128, 256,
  512];` mirror `_default_anchorgen`.
- REQ-3: `pub const RETINANET_SCORE_THRESH = 0.05`,
  `RETINANET_NMS_THRESH = 0.5`,
  `RETINANET_TOPK_CANDIDATES = 1000`,
  `RETINANET_DETECTIONS_PER_IMG = 300` — matches torchvision
  defaults.
- REQ-4: `pub struct RetinaFpn<T>` is the P3..P7 FPN: 3 laterals
  (C3..C5), 3 output convs (P3..P5), plus `p6` and `p7` extras
  (each a `Conv2d(256, 256, k=3, s=2, p=1, bias=true)`). P6 input
  is P5 (use_P5=True because in_ch == out_ch == 256). P7 input is
  ReLU(P6).
- REQ-5: `pub struct RetinaNetClassificationHead<T>` mirrors
  `RetinaNetClassificationHead`: 4-conv trunk (no GroupNorm) + final
  `cls_logits` Conv producing `num_anchors * num_classes` channels.
- REQ-6: `pub struct RetinaNetRegressionHead<T>` mirrors
  `RetinaNetRegressionHead`: 4-conv trunk + final `bbox_reg` Conv
  producing `num_anchors * 4` channels. Output is **not** ReLU-gated
  (deltas can be signed; the activation differs from FCOS).
- REQ-7: `retinanet_cell_anchors` uses
  `(x, int(x * 2^(1/3)), int(x * 2^(2/3)))` size triplets at each
  base (Python `int()` truncates) and rounds half-extents before
  emission. `retinanet_anchors_per_level` uses per-dim strides from
  the padded image size.
- REQ-8: `pub struct RetinaNet<T>` composes ResNet-50 backbone +
  `RetinaFpn` + classification head + regression head.
- REQ-9: `pub fn RetinaNet::forward` runs per-level head outputs;
  for each image: sigmoid scores → score-thresh → per-level top-K
  (1000) → decode (via `decode_boxes`) → clip → cross-class batched
  NMS → top-K (300).
- REQ-10: Box decode uses `decode_boxes` from `anchor_utils.rs`
  with default `(1.0, 1.0, 1.0, 1.0)` weights — RetinaNet's
  `BoxCoder` has unit weights, unlike FasterRCNN's
  `(10, 10, 5, 5)`.
- REQ-11: `Module::forward` returns first-image scores as 1-D
  `[N_det]` tensor.
- REQ-12: `pub fn retinanet_resnet50_fpn(num_classes)` returns
  `RetinaNet::new(num_classes)`.

## Acceptance Criteria

- [x] AC-1: `retinanet_resnet50_fpn::<f32>(91)` constructs.
- [x] AC-2: `num_parameters()` matches the hub registry pin
  (`34_014_999` with BN + fc body).
- [x] AC-3: `named_parameters()` exposes `backbone.*`, `fpn.*`,
  `classification_head.*`, `regression_head.*` prefixes including
  `fpn.lateral{3..5}`, `fpn.output{3..5}`, `fpn.p6`, `fpn.p7`,
  and the head `conv.{0..3}` + `cls_logits` / `bbox_reg`.
- [x] AC-4: `RetinaFpn::<f32>::new()` constructs and `forward`
  returns the five `p3..p7` keys.
- [x] AC-5: `RetinaNet::forward` returns one `Detections` per
  image with `boxes: [N, 4]`, `scores: [N]`, `labels.len() == N`.
- [x] AC-6: Default constants match torchvision (
  `RETINANET_SCORE_THRESH=0.05`, `_NMS_THRESH=0.5`,
  `_TOPK_CANDIDATES=1000`, `_DETECTIONS_PER_IMG=300`).

## Architecture

`pub struct RetinaFpn<T: Float>` is the RetinaNet-specific FPN.
Layout differs from `FeaturePyramidNetwork` (FasterRCNN's FPN):

- 3 lateral inputs (`layer2..layer4` = C3..C5), not 4.
- Output levels P3..P5 from the lateral+top-down pathway.
- `LastLevelP6P7` extras: `p6 = Conv2d(256, 256, k=3, s=2, p=1)`
  applied to P5, then `p7 = Conv2d(256, 256, k=3, s=2, p=1)`
  applied to `relu(p6)`.

`pub struct RetinaNetClassificationHead<T: Float>` is the
shared-across-levels classifier head. The 4-conv trunk has biases
but no normalization (unlike FCOS which adds GroupNorm). The final
`cls_logits` conv produces `num_anchors * num_classes` channels;
the `forward_level` method permutes `[B, A*K, H, W] → [B, H*W*A, K]`
in an explicit nested loop (avoiding intermediate permuted views).

`pub struct RetinaNetRegressionHead<T: Float>` is the analogous
regression head. The final `bbox_reg` conv produces
`num_anchors * 4` channels and the `forward_level` method
permutes to `[B, H*W*A, 4]`. Output is **not** ReLU-gated.

`pub struct RetinaNet<T: Float>` is the top-level container.
`Self::forward` runs:

1. Backbone → FPN.
2. Per-level: classification head + regression head → cache the
   `[B, HWA, K]` and `[B, HWA, 4]` tensors.
3. Per-image, per-level: sigmoid scores → score-thresh → per-level
   top-K (1000); decode surviving anchors (caching per-anchor
   decoded boxes since multiple classes can share an anchor); clip
   to image; stash boxes/scores/labels.
4. Cross-class batched NMS keyed by class label.
5. Top-K (300) post-NMS detections per image.

### Non-test production consumers

- `pub use RETINANET_NUM_ANCHORS_PER_LOC, RetinaFpn, RetinaNet,
  RetinaNetClassificationHead, RetinaNetRegressionHead,
  retinanet_resnet50_fpn` at
  `ferrotorch-vision/src/models/detection/mod.rs:44`.
- `register_model("retinanet_resnet50_fpn", ...)` at
  `ferrotorch-vision/src/models/registry.rs:348`.
- `use crate::models::detection::retinanet::RetinaFpn` at
  `ferrotorch-vision/src/models/detection/fcos.rs` (FCOS reuses
  the P3..P7 FPN structure).

## Parity contract

`parity_ops = []`. End-to-end parity is exercised against the
`RetinaNet_ResNet50_FPN_Weights.COCO_V1` (legacy / canonical, NOT
`_v2`) pretrained weights via the
`huggingface.co/ferrotorch/retinanet_resnet50_fpn` pin (SHA
`2f3593e7a2a1c15c5f2f7e6327e3c3d9de3cb4839922956ffec14b22f362b448`,
34,014,999 params). (#1143)

Numerical / structural edge cases preserved:

- **Size triplets via Python `int()` truncation.**
  `int(x * 2^(1/3))` truncates toward zero — matches torchvision's
  Python integer cast.
- **Half-extent rounding** matches `anchor_utils` convention.
- **Per-dim image-derived strides** — see anchor_utils (#1141).
- **`LastLevelP6P7` uses convs, not max-pool.** P6 + P7 each are
  `Conv2d(256, 256, k=3, s=2, p=1, bias=true)`. P7 input is
  `ReLU(P6)`.
- **Sigmoid + score_thresh + per-level top-K + cross-class NMS** —
  the postprocess order matters; cross-class NMS keyed by label
  prevents same-class overlapping detections from suppressing each
  other.

## Verification

Tests in `mod tests in retinanet.rs` cover construction, parameter
counts, FPN keys, anchor convention, head output shapes, and the
forward output structure.

No parity-sweep ops. Smoke command:

```bash
cargo test -p ferrotorch-vision --lib detection::retinanet:: 2>&1 | tail -3
```

Expected: all `detection::retinanet::tests` pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub const RETINANET_NUM_ANCHORS_PER_LOC: usize = 9;` in `retinanet.rs`; non-test consumer: `RetinaNet::new` in same file passes it to both head constructors. |
| REQ-2 | SHIPPED | impl: `pub const RETINANET_ASPECT_RATIOS` + `pub const RETINANET_BASE_SIZES` in `retinanet.rs`; non-test consumer: `retinanet_cell_anchors` in same file reads them inside `RetinaNet::forward` via `retinanet_anchors_per_level`. |
| REQ-3 | SHIPPED | impl: `pub const RETINANET_SCORE_THRESH` / `_NMS_THRESH` / `_TOPK_CANDIDATES` / `_DETECTIONS_PER_IMG` in `retinanet.rs`; non-test consumer: `RetinaNet::forward` body uses each — invoked by `register_model("retinanet_resnet50_fpn", ...)` at `ferrotorch-vision/src/models/registry.rs:348`. |
| REQ-4 | SHIPPED | impl: `pub struct RetinaFpn<T>` + `Self::new` + `Self::forward` in `retinanet.rs`; non-test consumer: `RetinaNet::new` calls `RetinaFpn::new()?`; `Fcos::new` at `new in ferrotorch-vision/src/models/detection/fcos.rs` also instantiates `RetinaFpn::new()?` and reuses it. |
| REQ-5 | SHIPPED | impl: `pub struct RetinaNetClassificationHead<T>` + `Self::new` + `Self::forward_level` in `retinanet.rs`; non-test consumer: `RetinaNet::new` calls `RetinaNetClassificationHead::new(FPN_OUT_CHANNELS, RETINANET_NUM_ANCHORS_PER_LOC, num_classes)?`. |
| REQ-6 | SHIPPED | impl: `pub struct RetinaNetRegressionHead<T>` + `Self::new` + `Self::forward_level` in `retinanet.rs`; non-test consumer: `RetinaNet::new` calls `RetinaNetRegressionHead::new(FPN_OUT_CHANNELS, RETINANET_NUM_ANCHORS_PER_LOC)?`. |
| REQ-7 | SHIPPED | impl: `fn retinanet_cell_anchors` + `fn retinanet_anchors_per_level` in `retinanet.rs`; non-test consumer: `RetinaNet::forward` calls `retinanet_anchors_per_level::<T>(&fm_sizes, (img_h, img_w))?`. |
| REQ-8 | SHIPPED | impl: `pub struct RetinaNet<T>` + `Self::new` in `retinanet.rs`; non-test consumer: `register_model("retinanet_resnet50_fpn", ...)` at `ferrotorch-vision/src/models/registry.rs:348`. |
| REQ-9 | SHIPPED | impl: `pub fn RetinaNet::forward` in `retinanet.rs`; non-test consumer: `impl<T> Module<T> for RetinaNet<T>::forward` calls it; the registry closure at `ferrotorch-vision/src/models/registry.rs:348` reaches it via `Module::forward`. |
| REQ-10 | SHIPPED | impl: anchor-decode block inside `RetinaNet::forward` in `retinanet.rs` (effectively `decode_boxes` semantics inline; the import `use crate::models::detection::anchor_utils::decode_boxes` at `retinanet in ferrotorch-vision/src/models/detection/retinanet.rs` makes the helper available); non-test consumer: same `Self::forward` path consumed by the registry closure. |
| REQ-11 | SHIPPED | impl: `impl<T> Module<T> for RetinaNet<T>::forward` in `retinanet.rs` returns first-image scores; non-test consumer: registered as `ModelConstructor<f32>` at `ModelConstructor in ferrotorch-vision/src/models/registry.rs`. |
| REQ-12 | SHIPPED | impl: `pub fn retinanet_resnet50_fpn` in `retinanet.rs`; non-test consumer: `register_model("retinanet_resnet50_fpn", ...)` at `ferrotorch-vision/src/models/registry.rs:348` calls it inside the closure. |
