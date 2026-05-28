# ferrotorch-vision — `models::yolo` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669 (working tree at /home/doll/pytorch)
baseline-torchvision: N/A (torchvision does not ship YOLO; ferrotorch carries a simplified
DarkNet-style YOLO as a vision-domain reference object detector)
upstream-paths:
  - /home/doll/pytorch/torch/nn/modules/conv.py
  - /home/doll/pytorch/torch/nn/modules/pooling.py
-->

## Summary

`ferrotorch-vision/src/models/yolo.rs` ships a simplified DarkNet-style
single-shot object-detection model with five convolutional backbone
stages and a 1x1 detection head. The model takes `[B, 3, 416, 416]`
images and produces a dense prediction grid of shape `[B, num_anchors *
(5 + num_classes), 13, 13]` where each cell predicts `(x, y, w, h,
objectness)` + per-class scores per anchor. Torchvision does not ship
YOLO; this file is a reference detector implementation that exercises the
core conv-relu-pool primitives without depending on detection-specific
plumbing (no anchors module, no NMS post-processing — those live in
`models::detection::*`).

## Requirements

- REQ-1: A private `BackboneStage<T: Float>` is a single backbone stage:
  one 3x3 conv (`stride` configurable, default 1; pad=1; bias=False) +
  ReLU + 2x2 max-pool (stride 2, no pad). Spatial dimensions are halved
  by the pool; channels go from `in_ch` to `out_ch`.
- REQ-2: `pub struct Yolo<T: Float>` carries five `BackboneStage`s
  (3→32→64→128→256→512), a 1x1 detection head Conv2d (`512 →
  num_anchors * (5 + num_classes)`, bias=True), `num_classes` and
  `num_anchors` config integers, and a `training: bool`. The head's bias
  is `True` (in contrast to the bias=False backbone convs) so the output
  logits can carry per-anchor mean offsets directly.
- REQ-3: `Yolo::new(num_classes, num_anchors)` is the configurable
  constructor; `Yolo::num_classes`, `Yolo::num_anchors`,
  `Yolo::num_parameters` are getter helpers.
- REQ-4: `Module::forward(input)` runs the five backbone stages
  sequentially then the 1x1 head, producing the dense prediction grid.
- REQ-5: `named_parameters` emits `backbone.stage{1..5}.<weight>` and
  `head.<weight,bias>` — a self-consistent layout (no upstream
  torchvision counterpart to match against).
- REQ-6: `children` / `named_children` (Phase 4 #995) expose the five
  backbone stages under `backbone.stage{1..5}` and the head under `head`,
  satisfying the #995 sweep invariant. YOLO has no BatchNorm in the
  current implementation (BN was omitted per the module-level comment)
  so the named_children override does not change loader behaviour for
  any existing fixture.
- REQ-7: `impl IntermediateFeatures<T> for Yolo<T>` (CL-499) exposes
  per-stage features keyed by `"stage1"..."stage5"` and `"head"` for
  feature-extraction / probe workflows.
- REQ-8: `pub fn yolo<T: Float>(num_classes: usize)` is the canonical
  3-anchor convenience constructor, matching the per-architecture naming
  used elsewhere in `ferrotorch-vision`.

## Acceptance Criteria

- [x] AC-1: `Yolo::<f32>::new(20, 3).forward(&[1, 3, 416, 416])` returns
  `[1, 75, 13, 13]` (`test_yolo_forward_shape_default_anchors`).
- [x] AC-2: `Yolo::<f32>::new(80, 3).forward(&[2, 3, 416, 416])` returns
  `[2, 255, 13, 13]` (`test_yolo_forward_shape_batch`).
- [x] AC-3: `Yolo::<f32>::new(10, 5).forward(&[1, 3, 416, 416])` returns
  `[1, 75, 13, 13]` (`test_yolo_forward_shape_custom_anchors`).
- [x] AC-4: `yolo::<f32>(20)` defaults to 3 anchors and 20 classes
  (`test_yolo_convenience_constructor`).
- [x] AC-5: Parameter count exactly equals the closed-form expression for
  20-class / 3-anchor and 80-class / 3-anchor configurations
  (`test_yolo_parameter_count` and `test_yolo_parameter_count_coco`).
- [x] AC-6: `named_parameters` contains `backbone.stage{1..5}.` and
  `head.` prefixed names (`test_yolo_named_parameters_prefixes`).
- [x] AC-7: `train` / `eval` toggle works (`test_yolo_train_eval`).
- [x] AC-8: YOLO is `Send + Sync` (`test_yolo_is_send_sync`).
- [x] AC-9: Gradient flows through the full forward pass to the input
  tensor (`test_gradient_flow_through_yolo`).

## Architecture

`BackboneStage<T: Float>` (lines 46-114) owns one `Conv2d(in, out, 3x3,
stride=stride, pad=1, bias=False)` and one `MaxPool2d([2,2], [2,2],
[0,0])`. `Module::forward` is `conv → relu → pool`. The stage exposes
its conv and pool as `children` / `named_children` (`conv` /
`pool` paths) so the parent `Yolo` can compose them into the
`backbone.stage1.conv`, `backbone.stage1.pool` dotted-path tree.

`Yolo<T: Float>` (lines 130-148) owns five named stages
(`stage1..stage5`), a 1x1 conv head (`Conv2d(512, num_anchors * (5 +
num_classes), 1x1, bias=True)`), `num_classes`, `num_anchors`, and
`training: bool`. `Module::forward` (lines 201-212) is a five-stage
sequential pass then `self.head.forward(&x)`.

For a 416×416 input the five stride-1 conv + stride-2 pool stages drop
the spatial size by a factor of 32 (`416 / 2^5 = 13`); the head's 1x1
conv preserves that grid. The output channels are
`num_anchors * (5 + num_classes)` — the standard YOLO encoding of `(x, y,
w, h, objectness) + num_classes` per anchor.

`children` / `named_children` (lines 260-279) expose the five stages and
the head. `impl IntermediateFeatures<T> for Yolo<T>` (lines 298-331)
replays the forward pass storing per-stage activations under
`"stage1"..."stage5"` and the final head output under `"head"`.

### Non-test production consumers

- `pub use yolo::{Yolo, yolo}` at `ferrotorch-vision/src/models/mod.rs:48`.
- `default_registry()` in `ferrotorch-vision/src/models/registry.rs:238`
  binds the `"yolo"` entry via `super::yolo::yolo::<f32>(num_classes)`
  inside `maybe_load_pretrained`.
- `feature_extractor.rs` is the round-trip smoke test for the
  YOLO IntermediateFeatures impl (the test confirms
  `feature_node_names()` returns the canonical 6-name list — it is
  `#[cfg(test)]`-gated and not a production consumer; the production
  consumer is the registry closure cited above).

## Parity contract

`parity_ops = []`. YOLO composes `Conv2d`, `MaxPool2d`, `relu` — each
primitive carries its own parity coverage. The output decoding (box
coordinates / objectness / class probabilities from the dense grid) is
not part of this file; consumers post-process the grid externally
(typically inside detection-domain pipelines or training-loop code).

Edge cases preserved:

- **`bias=False` on backbone convs, `bias=True` on head**: matches the
  classic DarkNet-Tiny pattern where the head's per-anchor predictions
  rely on a learned bias.
- **3-anchor default**: the convenience constructor `yolo(num_classes)`
  uses `Yolo::new(num_classes, 3)` matching the common YOLOv2 / YOLOv3
  Tiny anchors-per-cell count.
- **Output channel formula**: `num_anchors * (5 + num_classes)` is the
  standard YOLO encoding; the file's parameter-count tests verify it
  exactly for 20-class (output 75) and 80-class (output 255)
  configurations.

## Verification

Tests in `mod tests` in `yolo.rs`:

- `test_yolo_forward_shape_{default_anchors,batch,custom_anchors}`.
- `test_yolo_convenience_constructor`.
- `test_yolo_parameter_count` (closed-form for 20-class), `test_yolo_parameter_count_coco` (closed-form for 80-class COCO).
- `test_yolo_named_parameters_prefixes`.
- `test_yolo_train_eval`, `test_yolo_is_send_sync`.
- `test_gradient_flow_through_yolo`.

Smoke command:

```bash
cargo test -p ferrotorch-vision --lib yolo:: 2>&1 | tail -3
```

Expected: 9 tests pass; no `parity-sweep` ops to run.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `struct BackboneStage<T: Float>` + `BackboneStage::new` + `Module<T>` impl at `BackboneStage in yolo.rs`; non-test consumer: `Yolo::new` constructs five `BackboneStage::new` at `yolo in yolo.rs`. |
| REQ-2 | SHIPPED | impl: `pub struct Yolo<T: Float>` at `Yolo in yolo.rs`; non-test consumer: `yolo` constructor at `yolo in yolo.rs` returns `Yolo::new(num_classes, 3)` to the registry caller `registry.rs`. |
| REQ-3 | SHIPPED | impl: `Yolo::new`, `num_classes`, `num_anchors`, `num_parameters` at `yolo in yolo.rs`; non-test consumer: the `yolo` convenience constructor at `yolo in yolo.rs` calls `Yolo::new`, reached from `registry.rs`. |
| REQ-4 | SHIPPED | impl: `Module::forward` for `Yolo in yolo.rs`; non-test consumer: invoked from `default_registry()` via `yolo` at `registry.rs` whenever the registry's model is `.forward`'d. |
| REQ-5 | SHIPPED | impl: `Module::named_parameters` for `Yolo in yolo.rs` emits `backbone.stage{1..5}.<weight>` and `head.<weight,bias>`; non-test consumer: `maybe_load_pretrained` at `named_parameters in registry.rs` calls `model.load_state_dict(&state_dict, false)` which walks `named_parameters`. |
| REQ-6 | SHIPPED | impl: `children` / `named_children` for `Yolo in yolo.rs` and for `BackboneStage in yolo.rs`; non-test consumer: `apply_bn_buffers_from_state_dict` at `registry.rs` walks `named_descendants_dyn()` — for YOLO no BN modules exist and the loader returns Ok without effect; the override is in place for the sweep invariant. |
| REQ-7 | SHIPPED | impl: `impl IntermediateFeatures<T> for Yolo<T>` at `yolo in yolo.rs`; non-test consumer: `pub use feature_extractor::{FeatureExtractor, IntermediateFeatures, create_feature_extractor}` at `mod.rs` exposes the trait so callers can construct `FeatureExtractor::new(yolo_model, vec!["stage5".into()])`. |
| REQ-8 | SHIPPED | impl: `pub fn yolo<T: Float>(num_classes: usize) -> FerrotorchResult<Yolo<T>>` at `yolo in yolo.rs`; non-test consumer: `registry.rs` calls `super::yolo::yolo::<f32>(num_classes)` inside the `default_registry()` `maybe_load_pretrained` closure. |
