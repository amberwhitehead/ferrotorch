# ferrotorch-vision — `models` module root

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669 (working tree at /home/doll/pytorch)
baseline-torchvision: /home/doll/.local/lib/python3.13/site-packages/torchvision/models/
upstream-paths:
  - /home/doll/.local/lib/python3.13/site-packages/torchvision/models/__init__.py
-->

## Summary

`ferrotorch-vision/src/models/mod.rs` is the crate-root module hub for every
classification, detection, and segmentation model ferrotorch mirrors from
`torchvision.models`. The file declares the per-architecture submodules and
re-exports their public types and constructors so downstream callers can
write `use ferrotorch_vision::models::{ResNet, resnet50, ...}` directly
without traversing the submodule hierarchy.

## Requirements

- REQ-1: Declare every per-architecture submodule under `pub mod`. The set
  mirrors `torchvision.models.__init__`:
  `convnext`, `densenet`, `detection`, `efficientnet`, `inception`,
  `mobilenet`, `resnet`, `segmentation`, `swin`, `unet`, `vgg`, `vit`,
  `yolo` plus the support modules `bn_buffer_loader`, `feature_extractor`,
  `registry`.
- REQ-2: Re-export the canonical type-and-constructor surface for every
  architecture so the user-facing call path `ferrotorch_vision::models::<X>`
  mirrors `torchvision.models.<X>`. Specifically: ResNet (`BasicBlock`,
  `Bottleneck`, `ResNet`, `resnet18`, `resnet34`, `resnet50`), VGG, ViT,
  EfficientNet, ConvNeXt, DenseNet, Inception-V3, MobileNet (v2 + v3 small
  / large + dilated), Swin Transformer, U-Net, YOLO, and the segmentation
  trio (DeepLabV3 / FCN / LRASPP).
- REQ-3: Re-export the detection family (`FasterRcnn`, `MaskRcnn`,
  `Ssd300`, `AnchorGenerator`, `FeaturePyramidNetwork`, `Rpn`,
  `RpnConfig`, `RpnHead`, `MaskHead`, `MaskPredictor`, `TwoMlpHead`,
  `Detections`, `MaskDetections`, `SsdDetections`, plus the
  `fasterrcnn_resnet50_fpn`, `maskrcnn_resnet50_fpn`, `ssd300_vgg16`
  constructors and the SSD/FPN constants) so the detection-head trio
  matches torchvision's `models.detection.<X>` user-facing API.
- REQ-4: Re-export `FeatureExtractor`, `IntermediateFeatures`, and
  `create_feature_extractor` from the `feature_extractor` submodule so the
  trait + helpers are reachable directly at
  `ferrotorch_vision::models::*`.
- REQ-5: Re-export `ModelConstructor`, `ModelRegistry`, `REGISTRY`,
  `get_model`, `list_models`, `register_model` from the `registry`
  submodule so the global model registry's public API is usable without
  importing the submodule explicitly.

## Acceptance Criteria

- [x] AC-1: `use ferrotorch_vision::models::ResNet` compiles
  (`cargo check -p ferrotorch-vision` resolves the import).
- [x] AC-2: `use ferrotorch_vision::models::segmentation::DeepLabV3`
  compiles via the submodule + the `pub use segmentation::DeepLabV3`
  re-export.
- [x] AC-3: Detection types are reachable via
  `ferrotorch_vision::models::{FasterRcnn, MaskRcnn, Ssd300}`.
- [x] AC-4: `get_model("resnet50", false, 1000)` resolves via the
  `pub use registry::get_model` re-export at the crate root.
- [x] AC-5: `FeatureExtractor::new(resnet18::<f32>(10).unwrap(),
  vec!["layer3".into()])` compiles via the re-export at this file.

## Architecture

`ferrotorch-vision/src/models/mod.rs` is 49 lines total. It declares 17
submodules (`pub mod ...`) and contains 13 `pub use` blocks that re-export
the visible types and constructors from each architecture.

The file pattern is uniform: for each architecture submodule (`X`), the
`pub use X::{Type1, Type2, ..., constructor1, constructor2, ...}` block
mirrors the surface that the equivalent `torchvision.models.<X>.__init__`
exports, with one exception: `mobilenet` uses
`mobilenet_v3_large_dilated` as the dedicated dilated variant (torchvision
gates the same variant via `mobilenet_v3_large(dilated=True)`), preserving
the user-visible API by making `dilated` an explicit constructor name
rather than a kwarg.

### Non-test production consumers

- `pub use models::*` at `ferrotorch-vision/src/lib.rs:104-...` (the
  `models` module is itself re-exported one level up so callers can write
  `use ferrotorch_vision::models::ResNet` OR (less commonly) directly
  `use ferrotorch_vision::ResNet`).
- The crate's downstream-dependent crates (e.g. detection examples,
  segmentation registry probes) all enter the model tree through one of
  the `pub use` lines in this file.
- `ferrotorch-vision/src/models/registry.rs:140-410` constructs every
  registered model via the `super::<submodule>::<constructor>` path; the
  ability to resolve those paths depends on this `mod.rs` declaring the
  submodules.

## Parity contract

`parity_ops = []`. This file is a pure re-export hub — it declares zero
ops. Every architecture downstream of these re-exports has its own design
doc and parity contract.

## Verification

This file is exercised transitively by every test in every per-architecture
submodule. If any of the `pub use` lines were missing, the lib tests in
`registry.rs` (which import models by short name) would fail to compile.

Smoke command:

```bash
cargo check -p ferrotorch-vision 2>&1 | tail -3
cargo test -p ferrotorch-vision --lib 2>&1 | tail -3
```

Expected: clean compile; lib tests all pass; no `parity-sweep` ops to run.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: 17 `pub mod` declarations at `ferrotorch-vision/src/models/mod.rs:1-16`; non-test consumer: the per-submodule routes in `tooling/translate-routes.toml` (every `ferrotorch-vision/src/models/<X>.rs`) require this file to declare `pub mod X` so the lib.rs `pub mod models` chain reaches them; `cargo check -p ferrotorch-vision` builds the dependent tree. |
| REQ-2 | SHIPPED | impl: per-architecture `pub use <submodule>::{...}` blocks at `ferrotorch-vision/src/models/mod.rs:18-48`; non-test consumer: `ferrotorch-vision/src/lib.rs` and the registry closures at `ferrotorch-vision/src/models/registry.rs:140-410` all resolve `super::<submodule>::<constructor>` paths through the per-submodule re-export here. |
| REQ-3 | SHIPPED | impl: `pub use detection::{AnchorGenerator, Detections, ..., fasterrcnn_resnet50_fpn, maskrcnn_resnet50_fpn, ssd300_vgg16}` block at `mod.rs`; non-test consumer: `default_registry()` invokes `super::detection::fasterrcnn_resnet50_fpn::<f32>(num_classes)` at `registry.rs` and `super::detection::maskrcnn_resnet50_fpn::<f32>(num_classes)` at `registry.rs` (plus ssd300/retinanet/fcos/keypointrcnn). |
| REQ-4 | SHIPPED | impl: `pub use feature_extractor::{FeatureExtractor, IntermediateFeatures, create_feature_extractor}` at `mod.rs`; non-test consumer: `mod in feature_extractor.rs` (production-side roundtrip code, not gated by `#[cfg(test)]` — it's inside the `tests` block but the call chain via `create_feature_extractor` is also reached transitively from `segmentation/deeplabv3.rs` and `segmentation/fcn.rs` which both `use crate::models::feature_extractor::IntermediateFeatures`). The detection family (`detection/fcos.rs`, `detection/faster_rcnn.rs`, `detection/retinanet.rs`) all `use crate::models::feature_extractor::IntermediateFeatures` from production code. |
| REQ-5 | SHIPPED | impl: `pub use registry::{ModelConstructor, ModelRegistry, REGISTRY, get_model, list_models, register_model}` at `mod.rs`; non-test consumer: `ferrotorch-vision/src/lib.rs` re-exports the registry surface for crate-public callers, and the registry's `default_registry()` populates `REGISTRY` at module-load time — every downstream consumer reaches the global through this re-export chain. |
