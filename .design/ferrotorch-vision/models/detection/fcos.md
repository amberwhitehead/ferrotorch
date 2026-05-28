# ferrotorch-vision — `detection::fcos` module

<!--
tier: 3-component
status: draft
baseline-pytorch: torchvision 0.26.0+cu130 (git 336d36e8db990a905498c73933e35231876e28bc)
upstream-paths:
  - /home/doll/.local/lib/python3.13/site-packages/torchvision/models/detection/fcos.py
-->

## Summary

`ferrotorch-vision/src/models/detection/fcos.rs` ships the FCOS
anchor-free single-stage detector with ResNet-50 + FPN(P3-P7)
backbone. Mirrors `torchvision.models.detection.fcos_resnet50_fpn`
(`FCOS_ResNet50_FPN_Weights.COCO_V1`). Distinct from RetinaNet (the
other ResNet-50 + FPN single-stage detector): one anchor per cell,
regression predicts `(l, t, r, b)` distances from the cell center
to box edges (ReLU-gated), GroupNorm in the head trunk, and a
centerness branch that gates the classification score.

## Requirements

- REQ-1: `pub const FCOS_NUM_ANCHORS_PER_LOC: usize = 1;` — one
  anchor per spatial location (single aspect ratio 1.0).
- REQ-2: `pub const FCOS_NUM_CONVS: usize = 4;` — four-conv head
  trunks.
- REQ-3: `pub const FCOS_GN_GROUPS: usize = 32;` — GroupNorm group
  count.
- REQ-4: `pub const FCOS_BASE_SIZES: [f64; 5] = [8, 16, 32, 64,
  128];` per-cell anchor base sizes equal to per-level stride.
- REQ-5: `pub const FCOS_SCORE_THRESH = 0.2`,
  `FCOS_NMS_THRESH = 0.6`,
  `FCOS_TOPK_CANDIDATES = 1000`,
  `FCOS_DETECTIONS_PER_IMG = 100` — matches torchvision defaults.
- REQ-6: Shared `FcosConvGnTrunk<T>` (module-private) is 4 × (Conv
  3×3 + GroupNorm(32) + ReLU). `named_parameters` emits
  `conv.{0,3,6,9}` for the convs and `conv.{1,4,7,10}` for the
  GroupNorms — matching torchvision's `Sequential` interleaved-ReLU
  indexing.
- REQ-7: `pub struct FcosClassificationHead<T>` mirrors
  `FCOSClassificationHead`: trunk + final
  `Conv2d(256, num_anchors * num_classes)` (no ReLU on logits).
- REQ-8: `pub struct FcosRegressionHead<T>` mirrors
  `FCOSRegressionHead`: trunk + parallel `bbox_reg` (4 channels,
  **ReLU-gated**) and `bbox_ctrness` (1 channel, raw logits).
- REQ-9: `fcos_anchors_per_level` emits one anchor per cell at
  `[-S/2, -S/2, +S/2, +S/2]` (rounded), shifted by `(col*stride_w,
  row*stride_h)` derived from `image_size // grid_size`.
- REQ-10: `pub struct Fcos<T>` composes ResNet-50 + `RetinaFpn`
  (shared P3..P7 structure) + classification head + regression head.
- REQ-11: `pub fn Fcos::forward` per-level postprocess:
  `score = sqrt(sigmoid(cls) * sigmoid(centerness))`, gate by
  `FCOS_SCORE_THRESH=0.2`, per-level top-K (1000), decode boxes via
  `BoxLinearCoder(normalize_by_size=True)`
  (`pred = [cx - l*w, cy - t*h, cx + r*w, cy + b*h]`), clip,
  cross-class batched NMS (IoU 0.6), top-K (100).
- REQ-12: `Module::forward` returns first-image scores as a 1-D
  `[N_det]` tensor.
- REQ-13: `pub fn fcos_resnet50_fpn(num_classes)` returns
  `Fcos::new(num_classes)`.

## Acceptance Criteria

- [x] AC-1: `fcos_resnet50_fpn::<f32>(91)` constructs.
- [x] AC-2: `num_parameters() == 32_324_769` — equals torchvision's
  `32_269_600` plus 53,120 BN-affine deltas plus 2,049 for the
  ResNet `.fc` head (kept in ferrotorch as a Parameter while
  torchvision strips it).
- [x] AC-3: `named_parameters()` exposes `backbone.*`,
  `fpn.lateral{3..5}.*`, `fpn.output{3..5}.*`, `fpn.p6.*`,
  `fpn.p7.*`, `classification_head.conv.{0,1,3,4,6,7,9,10}.*`,
  `classification_head.cls_logits.*`,
  `regression_head.conv.{0,1,3,4,6,7,9,10}.*`,
  `regression_head.bbox_reg.*`, `regression_head.bbox_ctrness.*`.
- [x] AC-4: `FcosClassificationHead::<f32>::new(256, 1, 91)
  .forward_level(&randn([1, 256, 4, 4]))` returns `[1, 16, 91]`.
- [x] AC-5: `FcosRegressionHead::<f32>::new(256, 1)
  .forward_level(&randn([1, 256, 4, 4]))` returns
  `(reg: [1, 16, 4], ctr: [1, 16, 1])`.
- [x] AC-6: `bbox_reg` outputs are non-negative after the ReLU
  gate.
- [x] AC-7: `fcos_anchors_per_level` on a 1×1 grid at stride 8 →
  one anchor at `[-4, -4, 4, 4]`.
- [x] AC-8: `fcos_anchors_per_level` on a 2×2 grid at stride 8 →
  anchors shifted by `(col*8, row*8)`.
- [x] AC-9: `Fcos::forward([1, 3, 128, 128])` returns one
  per-image Detections.

## Architecture

`FcosConvGnTrunk<T: Float>` (module-private) is the shared
trunk used by both heads. Two arrays of length `FCOS_NUM_CONVS`
(four convs + four GroupNorms) interleaved with ReLU. The trunk
mirrors torchvision's `nn.Sequential[*conv]` layout
exactly — `named_parameters` emits Sequential-style indices
(`conv.{0,3,6,9}` for Conv2d and `conv.{1,4,7,10}` for
GroupNorm, with ReLU at the missing `{2,5,8,11}` indices).

`pub struct FcosClassificationHead<T: Float>` composes the trunk
with a final `cls_logits` conv producing `num_anchors *
num_classes` channels. Logits are not ReLU-gated.

`pub struct FcosRegressionHead<T: Float>` composes the trunk with
parallel `bbox_reg` and `bbox_ctrness` convs. The
`bbox_reg` output is gated through `relu` on every forward
(matching torchvision's `nn.functional.relu(self.bbox_reg(bbox_feature))`)
so the live `(l, t, r, b)` predictions are non-negative even before
decode. Centerness is raw logits.

The `permute_a_k_hw_to_hwa_k` helper (module-private) implements
the `(B, A*K, H, W) → (B, H*W*A, K)` permute as an explicit nested
loop, avoiding intermediate permuted-view allocations. Same pattern
as RetinaNet's head permute.

`pub struct Fcos<T: Float>` composes:

- `backbone: ResNet<T>` (configured for 1-class classification head;
  only `forward_features` is used).
- `fpn: RetinaFpn<T>` (shared P3..P7 structure — REQ-4 in the
  retinanet design doc).
- `classification_head: FcosClassificationHead<T>`.
- `regression_head: FcosRegressionHead<T>`.

`Self::forward` runs per-level postprocess in a single pass:

1. Per level: gather per-(anchor, class) combined score
   `sqrt(sigmoid(cls) * sigmoid(centerness))`; gate by
   `FCOS_SCORE_THRESH`.
2. Per-level top-K (1000): partial-sort by descending score.
3. Decode per anchor using `BoxLinearCoder(normalize_by_size=True)`:
   `cx, cy = box_center; w, h = box_size;
   pred = [cx - l*w, cy - t*h, cx + r*w, cy + b*h]`. A small cache
   avoids re-decoding the same anchor for different classes.
4. Clip per-level (the cross-level concat clips again — no-op for
   valid boxes).
5. Cross-class batched NMS (IoU 0.6) keyed by class.
6. Top-K (100) post-NMS detections per image.

### Non-test production consumers

- `pub use Fcos, FcosClassificationHead, FcosRegressionHead,
  fcos_resnet50_fpn` at
  `ferrotorch-vision/src/models/detection/mod.rs:33`.
- `register_model("fcos_resnet50_fpn", ...)` at
  `ferrotorch-vision/src/models/registry.rs:362`.

## Parity contract

`parity_ops = []`. End-to-end parity is exercised against the
`FCOS_ResNet50_FPN_Weights.COCO_V1` pretrained weights via the
`huggingface.co/ferrotorch/fcos_resnet50_fpn` pin (SHA
`f6446fb9456ed6845f142eff160eae6b67313e6690079b4512a15e274d06e325`,
32,269,600 params upstream). (#1144)

Numerical / structural edge cases preserved:

- **Anchor-free, one-anchor-per-cell.** No (size, ratio) grid — one
  anchor at `[-S/2, -S/2, +S/2, +S/2]` per spatial location.
- **`BoxLinearCoder` normalize_by_size=True.** Predictions are
  normalized by anchor (w, h), so the regression scale is invariant
  across levels.
- **Combined score `sqrt(cls * ctr)`.** Centerness gates the
  classification score multiplicatively (in probability space).
- **Sigmoid logits** for both cls and centerness. No softmax.
- **ReLU on `bbox_reg`.** The `(l, t, r, b)` predictions are
  non-negative — applied on every forward, not just at decode time.
- **No focal-loss prior offset at construction.** Both ferrotorch
  and torchvision initialize the `cls_logits` bias to `-log((1-π)/π)`
  internally, but the COCO_V1 pretrained checkpoint overwrites these
  on load — so the ferrotorch from-scratch path uses the default
  `Conv2d` init (which the pin script overwrites verbatim from the
  upstream checkpoint).

## Verification

Tests in `mod tests in fcos.rs`:

- `test_fcos_constructs`
- `test_fcos_param_count_matches_torchvision_plus_bn_affine` (locks
  in the exact `32_324_769` count)
- `test_fcos_named_params_prefixes`
- `test_cls_logits_output_dim`
- `test_bbox_reg_and_ctrness_output_dims`
- `test_cls_head_forward_layout`
- `test_reg_head_forward_layout`
- `test_reg_head_outputs_non_negative_after_relu`
- `test_fcos_anchor_box_at_origin_level_0`
- `test_fcos_anchor_shifts_match_torchvision_convention`
- `test_fcos_forward_small_image_returns_per_image_detections`
- `test_fcos_train_eval_toggle`

No parity-sweep ops. Smoke command:

```bash
cargo test -p ferrotorch-vision --lib detection::fcos:: 2>&1 | tail -3
```

Expected: 12 tests passed.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub const FCOS_NUM_ANCHORS_PER_LOC: usize = 1;` in `fcos.rs`; non-test consumer: `Fcos::new` in same file passes it to both head constructors. |
| REQ-2 | SHIPPED | impl: `pub const FCOS_NUM_CONVS: usize = 4;` in `fcos.rs`; non-test consumer: `FcosConvGnTrunk::new` in same file uses the constant to size the conv and GN arrays — invoked by `Fcos::new` via the head constructors. |
| REQ-3 | SHIPPED | impl: `pub const FCOS_GN_GROUPS: usize = 32;` in `fcos.rs`; non-test consumer: `FcosConvGnTrunk::new` reads it inside `Fcos::new`. |
| REQ-4 | SHIPPED | impl: `pub const FCOS_BASE_SIZES: [f64; 5]` in `fcos.rs`; non-test consumer: `fcos_anchors_per_level` reads it inside `Fcos::forward`. |
| REQ-5 | SHIPPED | impl: `pub const FCOS_SCORE_THRESH / _NMS_THRESH / _TOPK_CANDIDATES / _DETECTIONS_PER_IMG` in `fcos.rs`; non-test consumer: `Fcos::forward` reads each — invoked by `register_model("fcos_resnet50_fpn", ...)` at `ferrotorch-vision/src/models/registry.rs:362`. |
| REQ-6 | SHIPPED | impl: `FcosConvGnTrunk<T>` (private) + `Self::named_parameters` mapping in `fcos.rs`; non-test consumer: both head structs in same file own one, named-parameters surface via `FcosClassificationHead::named_parameters` (a public method) consumed by `Fcos::named_parameters` reachable from the registry. |
| REQ-7 | SHIPPED | impl: `pub struct FcosClassificationHead<T>` + `Self::new` + `Self::forward_level` in `fcos.rs`; non-test consumer: `Fcos::new` calls `FcosClassificationHead::new(FPN_OUT_CHANNELS, FCOS_NUM_ANCHORS_PER_LOC, num_classes)?`. |
| REQ-8 | SHIPPED | impl: `pub struct FcosRegressionHead<T>` + `Self::forward_level` (with `relu(&raw_reg)` ReLU gate) in `fcos.rs`; non-test consumer: `Fcos::new` calls `FcosRegressionHead::new(FPN_OUT_CHANNELS, FCOS_NUM_ANCHORS_PER_LOC)?`. |
| REQ-9 | SHIPPED | impl: `fn fcos_anchors_per_level` in `fcos.rs`; non-test consumer: `Fcos::forward` calls `fcos_anchors_per_level::<T>(&fm_sizes, (img_h, img_w))?`. |
| REQ-10 | SHIPPED | impl: `pub struct Fcos<T>` + `Self::new` in `fcos.rs`; non-test consumer: `register_model("fcos_resnet50_fpn", ...)` at `ferrotorch-vision/src/models/registry.rs:362`. |
| REQ-11 | SHIPPED | impl: `pub fn Fcos::forward` in `fcos.rs`; non-test consumer: `impl<T> Module<T> for Fcos<T>::forward` invokes it; the registry closure at `ferrotorch-vision/src/models/registry.rs:362` reaches it via `Module::forward`. |
| REQ-12 | SHIPPED | impl: `impl<T> Module<T> for Fcos<T>::forward` in `fcos.rs` returns first-image scores; non-test consumer: registered as `ModelConstructor<f32>` at `ModelConstructor in ferrotorch-vision/src/models/registry.rs`. |
| REQ-13 | SHIPPED | impl: `pub fn fcos_resnet50_fpn` in `fcos.rs`; non-test consumer: `register_model("fcos_resnet50_fpn", ...)` at `ferrotorch-vision/src/models/registry.rs:362` calls it inside the closure. |
