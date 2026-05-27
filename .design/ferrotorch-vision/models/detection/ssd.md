# ferrotorch-vision — `detection::ssd` module

<!--
tier: 3-component
status: draft
baseline-pytorch: torchvision 0.26.0+cu130 (git 336d36e8db990a905498c73933e35231876e28bc)
upstream-paths:
  - /home/doll/.local/lib/python3.13/site-packages/torchvision/models/detection/ssd.py
  - /home/doll/.local/lib/python3.13/site-packages/torchvision/models/detection/anchor_utils.py
-->

## Summary

`ferrotorch-vision/src/models/detection/ssd.rs` ships SSD300 with
VGG-16 backbone. Mirrors
`torchvision.models.detection.ssd300_vgg16(weights=None)`. Distinct
from FasterRCNN/RetinaNet: VGG-16 backbone (not ResNet-50 + FPN),
L2-normalisation on `conv4_3`, six feature-map scales with
torchvision's `DefaultBoxGenerator` priors (8732 boxes for
300×300), softmax classifier (not sigmoid), per-class postprocess
with cross-class top-K.

## Requirements

- REQ-1: `pub const SSD_ANCHORS_PER_SCALE: [usize; 6] = [4, 6, 6,
  6, 4, 4];` — torchvision's
  `DefaultBoxGenerator(aspect_ratios=[[2], [2,3], [2,3], [2,3], [2],
  [2]])` produces this distribution.
- REQ-2: `pub const SSD_FM_SIZES: [usize; 6] = [38, 19, 10, 5, 3,
  1];` — feature-map spatial sizes for SSD300.
- REQ-3: `pub const SSD_TOTAL_ANCHORS: usize = 8732;` —
  `38*38*4 + 19*19*6 + 10*10*6 + 5*5*6 + 3*3*4 + 1*1*4 = 8732`.
- REQ-4: `pub const SSD_SCORE_THRESH = 0.01`,
  `SSD_NMS_THRESH = 0.45`,
  `SSD_TOPK_CANDIDATES = 400`,
  `SSD_DETECTIONS_PER_IMG = 200` — matches torchvision defaults.
- REQ-5: `L2Norm` (module-private) per-spatial-location channel L2
  normalisation with a learnable per-channel scale initialised to
  20.0. Matches torchvision's `L2Norm` used on the conv4_3 feature
  map.
- REQ-6: `ConvBnRelu` (module-private) is the VGG / extra-layer
  building block. Routes dilation through `Conv2d::new_full` so
  atrous conv6 (k=3, dilation=6, padding=6) produces a 19×19
  feature map (the pre-#1142 dropped-dilation bug produced 29×29).
- REQ-7: `pub struct Ssd300<T>` owns the VGG features, L2Norm,
  extra layers, six classification heads, six regression heads.
- REQ-8: SSD anchor generator emits `(cx, cy, w, h)` normalised
  to `[0, 1]`. Centre stride mirrors torchvision's
  `DefaultBoxGenerator(steps=[8, 16, 32, 64, 100, 300])` — uses the
  explicit `step` value, NOT `300 / fm_size`. The `300 / fm_size`
  approximation is wrong for `fm=10` (step=32, not 30) and `fm=5`
  (step=64, not 60); the #1142 fix is the regression contract.
- REQ-9: `pub struct SsdDetections<T>` per-image output: boxes,
  scores, labels (background dropped).
- REQ-10: `Ssd300::forward` runs VGG features → L2Norm(conv4_3)
  + extras → per-scale (cls, reg) heads → decode against priors
  → softmax → per-class score gate (`>= 0.01`) → per-class NMS
  (`crate::ops::nms`) → cross-class top-K (200).
- REQ-11: `pub fn ssd300_vgg16(num_classes)` returns `Ssd300::new`
  with the user-supplied `num_classes`.

## Acceptance Criteria

- [x] AC-1: `ssd300_vgg16::<f32>(91)` constructs.
- [x] AC-2: `num_parameters()` matches the hub registry pin
  (`35_641_826`).
- [x] AC-3: `named_parameters()` exposes the
  `features`/`extra`/`head` prefixes for state-dict ingest.
- [x] AC-4: `SsdDetections::boxes` shape is `[N, 4]`, scores
  `[N]`, labels `len() == N`, with no background labels.
- [x] AC-5: The anchor generator produces exactly 8732 boxes for
  300×300.
- [x] AC-6: For an `fm=10` scale, the centre stride is 32 (not 30
  — the #1142 regression).

## Architecture

`L2Norm` is a `Module<T>` implementing per-spatial channel-wise L2
normalisation. The `weight: Parameter<T>` is initialised to 20.0
(matching torchvision's `L2Norm`). Forward uses an `f64`
accumulator for `norm_sq` to match torchvision's numerical path.

`ConvBnRelu` is the VGG / extras building block. Each block is a
`Conv2d` (built via `new_full` so dilation flows through), optional
`BatchNorm2d` (the VGG modifications for SSD300 don't actually
enable BN by default but the block carries the slot), then a relu.
Routing dilation through `new_full` is the #1142 fix — pre-fix the
block silently dropped `dilation`, so atrous conv6 (k=3, d=6, p=6)
produced 29×29 instead of 19×19, breaking the entire downstream
score path.

`pub struct Ssd300<T: Float>` is the top-level container. It owns
the VGG features, the L2Norm, the four extra-layer blocks, and the
six (cls, reg) head pairs. `Self::forward` produces 8732 anchor
predictions, applies softmax over classes, per-class score gating
and NMS, and finally cross-class top-K.

`generate_ssd_anchors` is the prior-box generator. The
`SsdScaleConfig.step` field is the verbatim
`DefaultBoxGenerator(steps=[8, 16, 32, 64, 100, 300])` from
torchvision; the centre divisor is `300 / step` (not
`300 / fm_size`). The two agree for `fm ∈ {38, 19, 3, 1}` but
diverge for `fm=10` (step=32 vs 30) and `fm=5` (step=64 vs 60).
Pre-#1142 ferrotorch divided by `fm_size`, baking the wrong
centre stride into every scale-2 / scale-3 anchor.

### Non-test production consumers

- `pub use SSD_ANCHORS_PER_SCALE, SSD_FM_SIZES, SSD_TOTAL_ANCHORS,
  Ssd300, SsdDetections, ssd300_vgg16` at
  `ferrotorch-vision/src/models/detection/mod.rs:49` and
  `ferrotorch-vision/src/lib.rs:23`.
- `register_model("ssd300_vgg16", ...)` at
  `ferrotorch-vision/src/models/registry.rs:335`.
- Pretrained-loading test
  `ferrotorch-hub/tests/pretrained_loading.rs::test_pretrained_ssd300_vgg16`
  (production end-to-end consumer of the pinned weights).

## Parity contract

`parity_ops = []`. End-to-end parity is exercised against
`SSD300_VGG16_Weights.COCO_V1` via the
`huggingface.co/ferrotorch/ssd300_vgg16` pin (SHA
`2db78702af742ec5882bc62e068e5337f366bc1dc00f069c34bbce91c5109dfe`,
35,641,826 params). (#1099)

Numerical / structural edge cases preserved:

- **Atrous conv6 dilation=6, padding=6.** The #1142 fix routes
  dilation through `Conv2d::new_full` so the kernel's receptive
  field is correct. Pre-fix the conv6 output was 29×29 instead of
  19×19.
- **Explicit `DefaultBoxGenerator(steps=...)`.** The #1142 fix
  uses the verbatim torchvision step values; `300 / fm_size`
  approximation gave drifted centres at fm=10/5.
- **Per-channel L2Norm scale init = 20.0.** Matches torchvision's
  `L2Norm` exactly. The weight is a learnable Parameter loaded
  from the pretrained checkpoint.
- **Softmax classifier + per-class score-thresh + per-class NMS**
  → cross-class top-K. The order is non-commutative (softmax
  before threshold; NMS per class; top-K across classes).
- **Boxes in `(cx, cy, w, h)` normalised to [0, 1]** at the prior
  stage; converted to xyxy pixel coords post-decode (clip + NMS).

## Verification

Tests in `mod tests in ssd.rs` cover construction, the 8732 anchor
count, anchor centre strides, head output shapes, and forward
output structure.

No parity-sweep ops. Smoke command:

```bash
cargo test -p ferrotorch-vision --lib detection::ssd:: 2>&1 | tail -3
```

Expected: all `detection::ssd::tests` pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub const SSD_ANCHORS_PER_SCALE: [usize; 6] = [4, 6, 6, 6, 4, 4];` in `ssd.rs`; non-test consumer: `Ssd300::new` in same file uses it to size each per-scale (cls, reg) head. |
| REQ-2 | SHIPPED | impl: `pub const SSD_FM_SIZES: [usize; 6] = [38, 19, 10, 5, 3, 1];` in `ssd.rs`; non-test consumer: `Ssd300::forward` reads it to slice the per-scale head outputs into the anchor-aligned layout. |
| REQ-3 | SHIPPED | impl: `pub const SSD_TOTAL_ANCHORS: usize = 8732;` in `ssd.rs`; non-test consumer: `Ssd300::forward` uses it to validate the per-scale concat shape. |
| REQ-4 | SHIPPED | impl: `pub const SSD_SCORE_THRESH / _NMS_THRESH / _TOPK_CANDIDATES / _DETECTIONS_PER_IMG` in `ssd.rs`; non-test consumer: `Ssd300::forward` reads each — invoked by `register_model("ssd300_vgg16", ...)` at `ferrotorch-vision/src/models/registry.rs:335`. |
| REQ-5 | SHIPPED | impl: `struct L2Norm<T>` + its `Module<T>::forward` in `ssd.rs`; non-test consumer: `Ssd300::new` in same file owns one (applied to `conv4_3` features), and `Ssd300::forward` calls into its `Module::forward`. |
| REQ-6 | SHIPPED | impl: `struct ConvBnRelu<T>` + `Self::new` routing through `Conv2d::new_full` in `ssd.rs`; non-test consumer: `Ssd300::new` builds the VGG features + extras out of `ConvBnRelu` blocks; the forward path runs through them. |
| REQ-7 | SHIPPED | impl: `pub struct Ssd300<T>` in `ssd.rs`; non-test consumer: `register_model("ssd300_vgg16", ...)` at `ferrotorch-vision/src/models/registry.rs:335`. |
| REQ-8 | SHIPPED | impl: `fn generate_ssd_anchors` in `ssd.rs` (with `SsdScaleConfig.step` field carrying the verbatim torchvision steps); non-test consumer: `Ssd300::forward` calls it to build the prior box tensor. The #1142 step-vs-fm_size regression test exercises the same code path. |
| REQ-9 | SHIPPED | impl: `pub struct SsdDetections<T>` in `ssd.rs`; non-test consumer: `Ssd300::forward` returns `Vec<SsdDetections<T>>`, consumed by the registry closure at `ferrotorch-vision/src/models/registry.rs:335` via `Module::forward`. |
| REQ-10 | SHIPPED | impl: `pub fn Ssd300::forward` body in `ssd.rs` (softmax → score-thresh → per-class NMS → top-K); non-test consumer: `impl<T> Module<T> for Ssd300<T>::forward` invokes it; the registry closure reaches it via `Module::forward`. |
| REQ-11 | SHIPPED | impl: `pub fn ssd300_vgg16<T>` in `ssd.rs`; non-test consumer: `register_model("ssd300_vgg16", ...)` at `ferrotorch-vision/src/models/registry.rs:335` calls it inside the closure. |
