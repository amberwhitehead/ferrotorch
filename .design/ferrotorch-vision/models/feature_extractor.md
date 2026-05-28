# ferrotorch-vision — `models::feature_extractor` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669 (working tree at /home/doll/pytorch)
baseline-torchvision: /home/doll/.local/lib/python3.13/site-packages/torchvision/models/feature_extraction.py
upstream-paths:
  - /home/doll/.local/lib/python3.13/site-packages/torchvision/models/feature_extraction.py
-->

## Summary

`ferrotorch-vision/src/models/feature_extractor.rs` ships the
`IntermediateFeatures` trait and `FeatureExtractor` wrapper that mirror
`torchvision.models.feature_extraction.create_feature_extractor`. The trait
exposes a model's per-stage activations as a `HashMap<String, Tensor<T>>`,
and `FeatureExtractor` filters that map down to a caller-requested subset
of stage names — the building block for transfer learning, multi-scale
detection / segmentation heads, and feature visualization.

## Requirements

- REQ-1: `pub trait IntermediateFeatures<T: Float>: Module<T>` declares two
  methods: `forward_features(&self, input: &Tensor<T>) ->
  FerrotorchResult<HashMap<String, Tensor<T>>>` returning every stage and
  its activation; `feature_node_names(&self) -> Vec<String>` listing the
  available stage names in execution order. The trait is `: Module<T>`-bound
  so any implementor is also a full Module. Mirrors the node-graph contract
  of torchvision's `create_feature_extractor(model, return_nodes=...)`.
- REQ-2: `pub struct FeatureExtractor<T: Float, M: IntermediateFeatures<T>>`
  wraps a model that implements `IntermediateFeatures` along with a
  filtered `return_nodes: Vec<String>` list. The struct stores the model
  by value (not by reference) so the extractor can be constructed once and
  passed around.
- REQ-3: `FeatureExtractor::new(model, return_nodes)` validates every
  requested node name against `model.feature_node_names()` at construction
  time. Unknown nodes surface as `FerrotorchError::InvalidArgument`
  listing the available names — fail-fast rather than fail-on-forward.
- REQ-4: `FeatureExtractor::forward(input)` runs the wrapped model's
  `forward_features`, filters the resulting HashMap to only the
  `return_nodes` set, and returns the filtered map. A missing requested
  node post-construction is an internal invariant violation
  (`FerrotorchError::InvalidArgument`).
- REQ-5: `FeatureExtractor::return_nodes(&self) -> &[String]` exposes the
  requested-node list; `model(&self)` and `model_mut(&mut self)` expose
  the wrapped model for further configuration (train/eval toggles, custom
  param ops).
- REQ-6: `pub fn create_feature_extractor<T: Float, M:
  IntermediateFeatures<T>>(model, return_nodes)` is the
  torchvision-named convenience constructor equivalent to
  `FeatureExtractor::new` (matches
  `torchvision.models.feature_extraction.create_feature_extractor`).
- REQ-7: At least one vision-model implementor of `IntermediateFeatures<T>`
  exists in production (a trait with no implementations is vacuous);
  current implementors include `ResNet<T>`, `MobileNetV2<T>`,
  `DenseNet<T>`, `VGG<T>`, `Yolo<T>`, `UNet<T>`.

## Acceptance Criteria

- [x] AC-1: `resnet18::<f32>(1000).feature_node_names()` includes `"stem"`,
  `"layer1"`, ..., `"layer4"`, `"avgpool"`, `"fc"`
  (`test_resnet_intermediate_features_keys`).
- [x] AC-2: `FeatureExtractor::new(resnet18(10), vec!["layer3", "layer4"])`
  produces a 2-key map with the layer3 channel count = 256, layer4 = 512
  (`test_feature_extractor_filters_to_requested_nodes`).
- [x] AC-3: `FeatureExtractor::new(model, vec!["bogus_node"])` errors with
  a message containing both `bogus_node` and `Available nodes`
  (`test_feature_extractor_rejects_unknown_node`).
- [x] AC-4: For ResNet, `forward_features` includes every node in
  `feature_node_names`
  (`test_feature_extractor_full_forward_features_includes_all_nodes`).
- [x] AC-5: The `"fc"` entry in ResNet's `forward_features` map equals
  `Module::forward`'s return value (within 1e-5)
  (`test_feature_extractor_final_output_matches_module_forward`).
- [x] AC-6: Empty `return_nodes` returns an empty filtered map
  (`test_feature_extractor_empty_return_nodes_returns_empty_map`).
- [x] AC-7: `MobileNetV2`, `DenseNet`, `VGG`, and `Yolo` round-trip through
  `FeatureExtractor` (CL-499:
  `test_{mobilenet_v2,densenet,vgg11,yolo}_feature_extractor_roundtrip`).

## Architecture

`pub trait IntermediateFeatures<T: Float>: Module<T>` (lines 29-42)
declares the two methods. The `: Module<T>` bound means every implementor
already has `forward`, `parameters`, `named_parameters`, `children`, and
`named_children`; the trait extends with the named-stage map.

`pub struct FeatureExtractor<T: Float, M: IntermediateFeatures<T>>` (lines
63-67) carries `model: M`, `return_nodes: Vec<String>`, and a
`_phantom: PhantomData<T>` so the type parameter `T` is bound even when
not used directly in a field.

`FeatureExtractor::new` (lines 77-94) iterates `return_nodes`, checks each
against `model.feature_node_names()`, and errors at the first unknown
name. `FeatureExtractor::forward` (lines 97-114) calls
`self.model.forward_features(input)` then filters the resulting
HashMap to the `return_nodes` set, cloning the kept tensors.

`pub fn create_feature_extractor` (lines 136-141) is a thin shim over
`FeatureExtractor::new`.

The trait is implemented for:

- `ResNet<T>` in `ferrotorch-vision/src/models/resnet.rs` (canonical
  template referenced in this file's module doc).
- `MobileNetV2<T>` in `mobilenet.rs`.
- `DenseNet<T>` in `densenet.rs`.
- `VGG<T>` in `vgg.rs`.
- `Yolo<T>` in `yolo.rs` (lines 298-331 of `yolo.rs`).
- `UNet<T>` in `unet.rs` (lines 578-635 of `unet.rs`).

### Non-test production consumers

- `pub use feature_extractor::{FeatureExtractor, IntermediateFeatures,
  create_feature_extractor}` at
  `ferrotorch-vision/src/models/mod.rs:27`.
- `crate::models::feature_extractor::IntermediateFeatures` import in:
  - `ferrotorch-vision/src/models/segmentation/deeplabv3.rs:55` —
    `ResNet50Dilated::forward_layer4` calls
    `self.inner.forward_features(input)` to extract layer4 features.
  - `ferrotorch-vision/src/models/segmentation/fcn.rs:39` —
    `Fcn::forward` calls `self.backbone.forward_features(input)` to
    extract layer4 features.
  - `ferrotorch-vision/src/models/detection/faster_rcnn.rs:38` — backbone
    feature extraction for the FPN path.
  - `ferrotorch-vision/src/models/detection/retinanet.rs` — backbone
    feature extraction for the RetinaNet head.
  - `ferrotorch-vision/src/models/detection/fcos.rs` — backbone
    feature extraction for the FCOS head.

## Parity contract

`parity_ops = []`. The trait + wrapper carry no op surface of their own;
they expose existing model forward computations through a name-keyed view.
Every implementor's `forward_features` is a deterministic restructuring of
the model's existing `Module::forward` (the per-stage tensors are the
intermediate values the forward already computes).

Edge cases preserved versus torchvision:

- **Return-node validation at construction time**: torchvision's
  `create_feature_extractor(model, return_nodes=["bogus"])` raises a
  `ValueError` at construction; ferrotorch raises `InvalidArgument` at
  construction. Both fail fast rather than at first forward.
- **Final-output node**: the model's final output is always present in
  the `forward_features` map under a conventional name (`"fc"` for
  ResNet, `"head"` for YOLO/UNet, `"classifier"` for MobileNetV2/DenseNet
  /VGG). Asking for that name via `return_nodes` returns the same tensor
  `Module::forward` returns.

## Verification

Tests in `mod tests` in `feature_extractor.rs`:

- `test_resnet_intermediate_features_keys`,
  `test_feature_extractor_filters_to_requested_nodes`,
  `test_feature_extractor_rejects_unknown_node`,
  `test_feature_extractor_full_forward_features_includes_all_nodes`,
  `test_feature_extractor_final_output_matches_module_forward`,
  `test_feature_extractor_empty_return_nodes_returns_empty_map`.
- CL-499 roundtrip smokes:
  `test_mobilenet_v2_feature_extractor_roundtrip`,
  `test_densenet_feature_extractor_roundtrip`,
  `test_vgg11_feature_extractor_roundtrip`,
  `test_yolo_feature_extractor_roundtrip`.

Smoke command:

```bash
cargo test -p ferrotorch-vision --lib feature_extractor:: 2>&1 | tail -3
```

Expected: 10 tests pass; no `parity-sweep` ops to run.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub trait IntermediateFeatures<T: Float>: Module<T>` at `IntermediateFeatures in feature_extractor.rs`; non-test consumer: `crate::models::feature_extractor::IntermediateFeatures` imported and called (`backbone.forward_features(input)`) at `segmentation/deeplabv3.rs, 106` and `segmentation/fcn.rs, 212`. |
| REQ-2 | SHIPPED | impl: `pub struct FeatureExtractor<T: Float, M: IntermediateFeatures<T>>` at `FeatureExtractor in feature_extractor.rs`; non-test consumer: `pub use feature_extractor::{FeatureExtractor, IntermediateFeatures, create_feature_extractor}` at `feature_extractor in ferrotorch-vision/src/models/mod.rs`, transitively re-exported via `ferrotorch-vision/src/lib.rs` for downstream-crate callers (referenced in the rustdoc example at `feature_extractor.rs`). |
| REQ-3 | SHIPPED | impl: `FeatureExtractor::new` validation block at `new in feature_extractor.rs`; non-test consumer: `pub use create_feature_extractor` at `mod.rs` is the torchvision-aliased entry point; the validation error path is exercised by the construction-time test `test_feature_extractor_rejects_unknown_node`. |
| REQ-4 | SHIPPED | impl: `FeatureExtractor::forward` at `forward in feature_extractor.rs`; non-test consumer: this method is reached transitively wherever any consumer constructs a `FeatureExtractor` and calls `.forward(input)`. Direct consumer-site for `IntermediateFeatures::forward_features` (which `FeatureExtractor::forward` calls into) is `forward in segmentation/deeplabv3.rs` and `forward in segmentation/fcn.rs`. |
| REQ-5 | SHIPPED | impl: `return_nodes`, `model`, `model_mut` accessors at `model_mut in feature_extractor.rs`; non-test consumer: the `pub use` re-export at `mod.rs` exposes the full impl block including these accessors to crate-public callers. |
| REQ-6 | SHIPPED | impl: `pub fn create_feature_extractor<T: Float, M: IntermediateFeatures<T>>` at `create_feature_extractor in feature_extractor.rs`; non-test consumer: `pub use feature_extractor::{..., create_feature_extractor}` at `mod.rs` makes the torchvision-named function reachable at the crate-public path. |
| REQ-7 | SHIPPED | impl: `impl IntermediateFeatures<T> for <X>` declarations exist in `resnet.rs` (consumed by `deeplabv3 in deeplabv3.rs` / `fcn in fcn.rs`), `mobilenet.rs`, `densenet.rs`, `vgg.rs`, `yolo in yolo.rs`, `unet in unet.rs`; non-test consumer: the `ResNet<T>` impl is called from production code at `ResNet in segmentation/deeplabv3.rs` (`self.inner.forward_features(input)`) and `fcn in segmentation/fcn.rs` (`self.backbone.forward_features(input)`), plus the detection family `detection/faster_rcnn.rs`, `detection/retinanet.rs`, `detection/fcos.rs`. |
