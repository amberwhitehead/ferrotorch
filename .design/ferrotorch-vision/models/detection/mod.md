# ferrotorch-vision — `detection` module index

<!--
tier: 3-component
status: draft
baseline-pytorch: torchvision 0.26.0+cu130 (git 336d36e8db990a905498c73933e35231876e28bc, installed at /home/doll/.local/lib/python3.13/site-packages/torchvision)
upstream-paths:
  - /home/doll/.local/lib/python3.13/site-packages/torchvision/models/detection/__init__.py
-->

## Summary

`ferrotorch-vision/src/models/detection/mod.rs` is a thin module-index
file. It declares the seven detection sub-modules and re-exports the
public surface (model structs, factory functions, anchor / FPN / RPN
helpers, per-model detection-result types). Mirrors
`torchvision.models.detection.__init__` — names live in submodules and
the package surface is the union of the re-exports.

## Requirements

- REQ-1: Declare submodules `anchor_utils`, `faster_rcnn`, `fcos`,
  `fpn`, `keypoint_rcnn`, `mask_rcnn`, `retinanet`,
  `roi_heads_postprocess`, `rpn`, `ssd` as `pub mod`.
- REQ-2: Re-export `AnchorGenerator` from `anchor_utils`.
- REQ-3: Re-export the Faster R-CNN surface (`Detections`,
  `FasterRcnn`, `TwoMlpHead`, `fasterrcnn_resnet50_fpn`).
- REQ-4: Re-export the FCOS surface (`Fcos`,
  `FcosClassificationHead`, `FcosRegressionHead`, `fcos_resnet50_fpn`)
  while **not** re-exporting `fcos::Detections` (name collision with
  `faster_rcnn::Detections` — qualified path required).
- REQ-5: Re-export `FPN_OUT_CHANNELS` and `FeaturePyramidNetwork`
  from `fpn`.
- REQ-6: Re-export the Keypoint R-CNN surface
  (`KEYPOINT_RCNN_NUM_CLASSES`, `KEYPOINT_RCNN_NUM_KEYPOINTS`,
  `KeypointDetections`, `KeypointHead`, `KeypointPredictor`,
  `KeypointRcnn`, `keypointrcnn_resnet50_fpn`).
- REQ-7: Re-export the Mask R-CNN surface (`MaskDetections`,
  `MaskHead`, `MaskPredictor`, `MaskRcnn`, `maskrcnn_resnet50_fpn`).
- REQ-8: Re-export the RetinaNet surface with the same
  `Detections`-collision pattern as FCOS — qualified path required
  for `retinanet::Detections`.
- REQ-9: Re-export `Rpn`, `RpnConfig`, `RpnHead` from `rpn`.
- REQ-10: Re-export the SSD surface (`SSD_ANCHORS_PER_SCALE`,
  `SSD_FM_SIZES`, `SSD_TOTAL_ANCHORS`, `Ssd300`, `SsdDetections`,
  `ssd300_vgg16`).

## Acceptance Criteria

- [x] AC-1: `cargo check -p ferrotorch-vision` compiles with the
  module declarations above.
- [x] AC-2: `pub use ferrotorch_vision::detection::FasterRcnn` is
  resolvable from `ferrotorch-vision/src/lib.rs`.
- [x] AC-3: `pub use ferrotorch_vision::detection::fcos::Detections`
  resolves via the qualified path (the bare `Detections` symbol
  points at the `faster_rcnn` variant).
- [x] AC-4: All ten submodule declarations are present (`anchor_utils`,
  `faster_rcnn`, `fcos`, `fpn`, `keypoint_rcnn`, `mask_rcnn`,
  `retinanet`, `roi_heads_postprocess`, `rpn`, `ssd`).

## Architecture

`pub mod` declarations in `mod.rs` expose each of the ten submodules.
The two `Detections` type-name collisions (FCOS vs Faster R-CNN, and
RetinaNet vs Faster R-CNN) are resolved by leaving the colliding
type names un-re-exported and documenting the qualified path in the
module-level doc-comment. Faster R-CNN's `Detections` wins the
bare-name re-export because it is the most-used detection-result
type (consumed by Mask R-CNN and Keypoint R-CNN's wrappers).

### Non-test production consumers

- `pub use detection::{...}` at `ferrotorch-vision/src/models/mod.rs`
  fan-out re-exports the entire detection public surface to
  `ferrotorch_vision::models::*`.
- `pub use models::{...}` at `ferrotorch-vision/src/lib.rs` (lines
  105–110) propagates the surface to `ferrotorch_vision::*` —
  this is the crate's external API.

## Parity contract

`parity_ops = []`. The mod file ships no ops directly — it composes
the public API of its submodules. Each submodule has its own design
doc + parity contract.

## Verification

The file itself has no `mod tests` block — its correctness is exercised
transitively by every consumer of the detection public API
(e.g. `mod tests in ferrotorch-vision/src/models/registry.rs`, which
registers the six detection factories under their torchvision names).

```bash
cargo check -p ferrotorch-vision 2>&1 | tail -3
```

Expected: clean check (no compilation errors).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub mod` declarations in `mod.rs`; non-test consumer: `pub use detection::{...}` in `ferrotorch-vision/src/models/mod.rs:20`. |
| REQ-2 | SHIPPED | impl: `pub use anchor_utils::AnchorGenerator;` in `mod.rs`; non-test consumer: `ferrotorch-vision/src/lib.rs:21` re-exports `AnchorGenerator` to the crate root. |
| REQ-3 | SHIPPED | impl: `pub use faster_rcnn::{Detections, FasterRcnn, TwoMlpHead, fasterrcnn_resnet50_fpn};` in `mod.rs`; non-test consumer: `register_model("fasterrcnn_resnet50_fpn", ...)` at `ferrotorch-vision/src/models/registry.rs:270` invokes the factory. |
| REQ-4 | SHIPPED | impl: `pub use fcos::{Fcos, FcosClassificationHead, FcosRegressionHead, fcos_resnet50_fpn};` in `mod.rs`; non-test consumer: `register_model("fcos_resnet50_fpn", ...)` at `ferrotorch-vision/src/models/registry.rs:362`. |
| REQ-5 | SHIPPED | impl: `pub use fpn::{FPN_OUT_CHANNELS, FeaturePyramidNetwork};` in `mod.rs`; non-test consumer: `use crate::models::detection::fpn::FPN_OUT_CHANNELS` at `ferrotorch-vision/src/models/detection/retinanet.rs:41` and `ferrotorch-vision/src/models/detection/fcos.rs:54`. |
| REQ-6 | SHIPPED | impl: `pub use keypoint_rcnn::{KEYPOINT_RCNN_NUM_CLASSES, KEYPOINT_RCNN_NUM_KEYPOINTS, KeypointDetections, KeypointHead, KeypointPredictor, KeypointRcnn, keypointrcnn_resnet50_fpn};` in `mod.rs`; non-test consumer: `register_model("keypointrcnn_resnet50_fpn", ...)` at `ferrotorch-vision/src/models/registry.rs:378`. |
| REQ-7 | SHIPPED | impl: `pub use mask_rcnn::{MaskDetections, MaskHead, MaskPredictor, MaskRcnn, maskrcnn_resnet50_fpn};` in `mod.rs`; non-test consumer: `register_model("maskrcnn_resnet50_fpn", ...)` at `ferrotorch-vision/src/models/registry.rs:285`. |
| REQ-8 | SHIPPED | impl: `pub use retinanet::{RETINANET_NUM_ANCHORS_PER_LOC, RetinaFpn, RetinaNet, RetinaNetClassificationHead, RetinaNetRegressionHead, retinanet_resnet50_fpn};` in `mod.rs`; non-test consumer: `register_model("retinanet_resnet50_fpn", ...)` at `ferrotorch-vision/src/models/registry.rs:348`. |
| REQ-9 | SHIPPED | impl: `pub use rpn::{Rpn, RpnConfig, RpnHead};` in `mod.rs`; non-test consumer: `use crate::models::detection::rpn::{Rpn, RpnConfig}` at `ferrotorch-vision/src/models/detection/faster_rcnn.rs:37`. |
| REQ-10 | SHIPPED | impl: `pub use ssd::{SSD_ANCHORS_PER_SCALE, SSD_FM_SIZES, SSD_TOTAL_ANCHORS, Ssd300, SsdDetections, ssd300_vgg16};` in `mod.rs`; non-test consumer: `register_model("ssd300_vgg16", ...)` at `ferrotorch-vision/src/models/registry.rs:335`. |
