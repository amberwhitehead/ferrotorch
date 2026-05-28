# ferrotorch-vision — `detection::anchor_utils` module

<!--
tier: 3-component
status: draft
baseline-pytorch: torchvision 0.26.0+cu130 (git 336d36e8db990a905498c73933e35231876e28bc)
upstream-paths:
  - /home/doll/.local/lib/python3.13/site-packages/torchvision/models/detection/anchor_utils.py
  - /home/doll/.local/lib/python3.13/site-packages/torchvision/models/detection/_utils.py
-->

## Summary

`ferrotorch-vision/src/models/detection/anchor_utils.rs` ships the
multi-scale anchor generator used by Faster R-CNN and Mask R-CNN
(`AnchorGenerator`) plus the `BoxCoder.decode_single` shape-decoder
(`decode_boxes`). Mirrors
`torchvision.models.detection.anchor_utils.AnchorGenerator` and the
`BoxCoder.decode_single` body in
`torchvision.models.detection._utils`.

## Requirements

- REQ-1: `pub struct AnchorGenerator` (over `Vec<LevelConfig>`) tiles
  base anchors `[-w/2, -h/2, w/2, h/2]` (rounded) across every cell of
  every FPN level. Aspect ratio is the outer loop, sizes the inner —
  matching torchvision's `(w_ratios[:, None] * scales[None, :]).view(-1)`.
- REQ-2: `pub fn AnchorGenerator::default_fasterrcnn` returns the
  torchvision FasterRCNN defaults: sizes `(32, 64, 128, 256, 512)`
  one per level, aspect ratios `(0.5, 1.0, 2.0)` every level, strides
  `(4, 8, 16, 32, 64)`.
- REQ-3: `pub fn generate_anchors_for_image` derives per-dim strides
  from the padded image size (`stride_h = image_h / fh`,
  `stride_w = image_w / fw`) — matches torchvision's
  `AnchorGenerator.forward` which recomputes strides from the actual
  padded input shape each call (rather than reading from a per-level
  config). Critical for non-64-aligned padded image sizes (#1141
  diagnosis).
- REQ-4: `pub fn generate_anchors` uses the canonical per-level square
  stride from the `LevelConfig` (legacy / test-friendly path; never
  used by production callers — see REQ-3).
- REQ-5: Anchor tiling visits `(fy, fx)` in row-major order, with the
  cell **corner** as the centre (`shifts_x = arange(grid_w) * stride_w`
  — no `+ 0.5` offset). Matches torchvision `grid_anchors`.
- REQ-6: `pub fn num_anchors_per_location` returns
  `sizes.len() * aspect_ratios.len()` for a given level.
- REQ-7: `pub fn decode_boxes` applies `(dx, dy, dw, dh)` deltas
  against anchors with one-sided `bbox_xform_clip = log(1000/16) ≈
  4.135` clamp on `dw`/`dh` (positive side only — large negatives
  pass through). Default RPN weights are `(1.0, 1.0, 1.0, 1.0)`.

## Acceptance Criteria

- [x] AC-1: `AnchorGenerator::default_fasterrcnn().generate_anchors::<f32>(&[(2, 2); 5])`
  returns shape `[60, 4]` (5 levels × 4 cells × 3 anchors).
- [x] AC-2: Every emitted box satisfies `x1 < x2` and `y1 < y2`.
- [x] AC-3: For an 800×1088 padded image with FPN sizes
  `[(200,272),(100,136),(50,68),(25,34),(13,17)]`,
  `generate_anchors_for_image` differs from `generate_anchors`
  on p6 anchors (per-dim stride `(61, 64)` vs canonical
  `(64, 64)`).
- [x] AC-4: `decode_boxes` with zero deltas + unit weights is the
  identity (`pred == anchors`).
- [x] AC-5: Large positive `dw` is one-side clamped to
  `log(1000/16)` before `exp`.

## Architecture

`pub struct AnchorGenerator` in `anchor_utils.rs` owns
`Vec<LevelConfig>`. `Self::cell_anchors` materialises the 9-anchor
template (or fewer, if `sizes × aspect_ratios` is smaller) by
iterating `aspect_ratios` then `sizes`. The half-extents are
**rounded** (`.round()`) before negation so the resulting
xyxy values are integer-aligned, matching torchvision exactly.
The same routine is used by `RetinaNet` and `FCOS` anchor builders
(which mirror the same convention without going through
`AnchorGenerator` directly).

`pub fn generate_anchors_for_image` is the production path: it
derives `(stride_h, stride_w)` per level from the padded
image size and the per-level grid size. The 800×1088 → p6
divergence test (`test_anchor_generator_per_dim_stride_image_800x1088`)
is the regression guard for the #1141 round-4 anchor-stride bug.

`pub fn generate_anchors` accepts a `&[(usize, usize)]` of
feature-map sizes and uses the per-level square `cfg.stride` from
`LevelConfig`. It is reserved for tests and inline construction
where the caller can guarantee the canonical stride.

`pub fn decode_boxes` is the shape decoder. For each anchor it:

1. Recovers the xy-center + width/height.
2. Applies `(dx, dy) / (wx, wy)` to shift the center.
3. Applies one-sided `max=` clamp on `dw`/`dh` before `exp`.
4. Recovers x1/y1/x2/y2 from the predicted (cx, cy, w, h).

### Non-test production consumers

- `pub use AnchorGenerator` at
  `ferrotorch-vision/src/models/detection/mod.rs` and
  `ferrotorch-vision/src/lib.rs:21`.
- `use crate::models::detection::anchor_utils::{AnchorGenerator,
  decode_boxes}` at `ferrotorch-vision/src/models/detection/rpn.rs:28`:
  RPN's `Self::new` instantiates the default generator and
  `Self::forward` calls `decode_boxes` after the per-level top-K.
- `use crate::models::detection::anchor_utils::decode_boxes` at
  `ferrotorch-vision/src/models/detection/retinanet.rs`.

## Parity contract

`parity_ops = []`. Anchor generation is geometry, not a parity-sweep
op. Geometric correctness is locked in by characterization tests
(see Verification).

Numerical / structural edge cases preserved:

- **Half-extent rounding.** `(w * 0.5).round()` matches
  torchvision's `(w / 2).round_()` so the xyxy values are integer.
- **Aspect-ratio outer / sizes inner ordering.** Torchvision's
  `(w_ratios[:, None] * scales[None, :]).view(-1)` materialises
  one (size, ratio) pair per cell anchor in this order.
- **Per-dim image-derived strides.** `image_h / fh` truncates as
  integer division — the resulting `(61, 64)` p6 stride for an
  800×1088 image is what torchvision's `AnchorGenerator.forward`
  computes from the padded image shape.
- **One-sided `bbox_xform_clip` on `dw`/`dh`.** Negative deltas
  pass through (consistent with `torch.clamp(dw, max=clip)`).

## Verification

Tests in `mod tests in anchor_utils.rs`:

- `test_anchor_generator_count`
- `test_anchor_generator_box_format`
- `test_anchor_generator_per_dim_stride_image_800x1088` (the #1141
  regression test)
- `test_decode_boxes_identity_delta_zero`

No parity-sweep ops. Smoke command:

```bash
cargo test -p ferrotorch-vision --lib detection::anchor_utils:: 2>&1 | tail -3
```

Expected: 4 tests passed.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct AnchorGenerator` + `Self::cell_anchors` in `anchor_utils.rs`; non-test consumer: `Rpn::new` at `new in ferrotorch-vision/src/models/detection/rpn.rs` calls `AnchorGenerator::default_fasterrcnn()` and stores the generator as the `anchor_gen` field. |
| REQ-2 | SHIPPED | impl: `pub fn AnchorGenerator::default_fasterrcnn` in `anchor_utils.rs`; non-test consumer: `Rpn::new` at `ferrotorch-vision/src/models/detection/rpn.rs:162`. |
| REQ-3 | SHIPPED | impl: `pub fn AnchorGenerator::generate_anchors_for_image` in `anchor_utils.rs`; non-test consumer: `Rpn::forward` at `ferrotorch-vision/src/models/detection/rpn.rs:193` calls `self.anchor_gen.generate_anchors_for_image(&fm_sizes, (img_h, img_w))`. |
| REQ-4 | SHIPPED | impl: `pub fn AnchorGenerator::generate_anchors` in `anchor_utils.rs`; non-test consumer: `pub use AnchorGenerator` at `anchor_utils in ferrotorch-vision/src/models/detection/mod.rs` exposes the method via the crate's public API. The same `AnchorGenerator::generate_anchors_with_strides` core is invoked by `generate_anchors_for_image` (REQ-3). |
| REQ-5 | SHIPPED | impl: nested `for fy / for fx` loops in `Self::generate_anchors_with_strides` in `anchor_utils.rs` (matches `arange(grid_w) * stride_w`, no `+0.5`); non-test consumer: `Rpn::forward` at `ferrotorch-vision/src/models/detection/rpn.rs:193` consumes the resulting `[N_total, 4]` tensor. |
| REQ-6 | SHIPPED | impl: `pub fn AnchorGenerator::num_anchors_per_location` in `anchor_utils.rs`; non-test consumer: `pub use AnchorGenerator` at `anchor_utils in ferrotorch-vision/src/models/detection/mod.rs` exposes the method via the crate's public API for downstream RPN/per-level head configuration. |
| REQ-7 | SHIPPED | impl: `pub fn decode_boxes` in `anchor_utils.rs`; non-test consumer: `Rpn::forward` at `forward in ferrotorch-vision/src/models/detection/rpn.rs` calls `decode_boxes::<f64>(&anc_t, &del_t, (1.0, 1.0, 1.0, 1.0))`. RetinaNet uses the same module-private decoder via `use crate::models::detection::anchor_utils::decode_boxes` at `forward in ferrotorch-vision/src/models/detection/retinanet.rs`. |
