# ferrotorch-vision — `models::resnet` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669 (working tree at /home/doll/pytorch)
baseline-torchvision: /home/doll/.local/lib/python3.13/site-packages/torchvision/models/resnet.py
upstream-paths:
  - /home/doll/.local/lib/python3.13/site-packages/torchvision/models/resnet.py
-->

## Summary

`ferrotorch-vision/src/models/resnet.rs` ships the He et al. 2015 ResNet
architectures (ResNet-18, ResNet-34, ResNet-50) plus a dilated ResNet-50
variant used by the FCN/DeepLabV3 segmentation backbones. Every conv is
followed by a `BatchNorm2d` (`eps=1e-5`, `momentum=0.1`, `affine=true`)
matching torchvision's `_resnet` builder. The `named_parameters` layout
mirrors `torchvision.models.resnet.ResNet` exactly so the strict
value-parity loader can adopt torchvision pretrained state dicts without
remapping.

## Requirements

- REQ-1: `pub struct BasicBlock<T: Float>` mirrors torchvision's
  `BasicBlock` (two 3×3 convs + BN + ReLU; `EXPANSION = 1`; optional
  `downsample = (Conv1x1, BN)` projection on the skip).
- REQ-2: `pub struct Bottleneck<T: Float>` mirrors torchvision's
  `Bottleneck` (1×1 → 3×3 → 1×1, BN after each; `EXPANSION = 4`; optional
  `downsample`). The 3×3 stride is placed on `conv2` (ResNet-V1.5 stride
  convention) as torchvision does.
- REQ-3: `pub struct ResNet<T: Float>` carries the stem
  (`conv1`/`bn1`/`maxpool`), four residual stages (`layer1..layer4`), and
  the head (`avgpool`/`fc`). `Module::forward` runs the stem → stages →
  flatten → fc.
- REQ-4: `pub fn resnet18`, `pub fn resnet34`, `pub fn resnet50`
  construct the canonical torchvision variants `[2,2,2,2]` /
  `[3,4,6,3]` BasicBlock and `[3,4,6,3]` Bottleneck respectively.
- REQ-5: `pub fn resnet50_dilated` mirrors
  `torchvision.models.resnet50(replace_stride_with_dilation=...)`. The
  internal `make_basic_layer` / `make_bottleneck_layer` thread a
  `current_dilation` counter so each stage's `dilate` flag swaps stride-2
  for `dilation *= stride` and `stride = 1`, exactly as
  `ResNet._make_layer` does.
- REQ-6: `named_parameters` returns the torchvision-shaped paths
  (`conv1.weight`, `bn1.weight`, `bn1.bias`, `layer1.0.conv1.weight`,
  ..., `layer1.0.downsample.0.weight`, `layer1.0.downsample.1.weight`,
  `fc.weight`, `fc.bias`).
- REQ-7: `named_children` exposes the same dotted-path tree so the
  Phase-2 BN-buffer loader (`bn_buffer_loader.rs`) can walk
  `named_descendants_dyn()` and assign `running_mean` / `running_var` /
  `num_batches_tracked` per BN under each `bn1` / `bn2` / `bn3` /
  `downsample.1` path.
- REQ-8: `ResNet<T>` implements
  `IntermediateFeatures<T>` so `create_feature_extractor` can return
  per-stage activations (`stem`, `layer1`, ..., `layer4`, `avgpool`,
  `fc`) for downstream FPN/U-Net composition.
- REQ-9: Train/eval propagation: `ResNet::train` / `ResNet::eval` walks
  every BN child so its running-statistic update path mirrors torchvision
  semantics (BN updates running stats only when `training=true`).

## Acceptance Criteria

- [x] AC-1: `BasicBlock::<f32>::new(64, 64, 1)` constructs (test
  `test_basic_block_same_channels`).
- [x] AC-2: `BasicBlock::<f32>::new(64, 128, 2)` produces a downsample
  branch and halves spatial dims (`test_basic_block_downsample`).
- [x] AC-3: `Bottleneck::<f32>::new(256, 64, 1)` preserves the channel
  count (256 == 64 × 4); first-block downsample triggers when
  `in_planes != planes * 4` (`test_bottleneck_first_block`).
- [x] AC-4: `resnet18::<f32>(1000)` parameter count is in
  (11 000 000, 12 000 000) — matches torchvision's ~11.7M
  (`test_resnet18_param_count`).
- [x] AC-5: `resnet34::<f32>(1000)` parameter count is in
  (21 000 000, 22 000 000) (`test_resnet34_param_count`).
- [x] AC-6: `resnet50::<f32>(1000)` parameter count is in
  (25 000 000, 26 000 000) (`test_resnet50_param_count`).
- [x] AC-7: `ResNet::forward` on `[1, 3, 224, 224]` returns
  `[1, num_classes]` (`test_resnet{18,34,50}_output_shape`).
- [x] AC-8: `named_parameters` includes the torchvision-shaped prefixes
  `conv1.`, `layer{1,2,3,4}.`, `fc.`
  (`test_resnet18_named_parameters_prefixes`).
- [x] AC-9: Gradients flow through both the residual identity AND the
  conv path on backward
  (`test_gradient_flow_through_{basic_block,bottleneck}`).

## Architecture

`pub struct BasicBlock<T: Float>` in `resnet.rs` carries `conv1`, `bn1`,
`conv2`, `bn2`, optional `downsample: Option<(Conv2d, BatchNorm2d)>`, and
a `training: bool`. `BasicBlock::new(in_planes, planes, stride)` is a
shim over `BasicBlock::new_full` with `dilation=(1, 1)` so existing
callers stay byte-equivalent (Phase 6 #994 introduced `new_full` for the
dilated path). The `Module::forward` impl runs
`conv1 → bn1 → relu → conv2 → bn2`, then `add(out, identity)` (identity
= projected skip if present, else cloned input), then a final `relu`.

`pub struct Bottleneck<T: Float>` mirrors the 1×1 → 3×3 → 1×1 layout with
the stride placed on `conv2`. `Bottleneck::new_full` accepts an explicit
`dilation` for the middle 3×3 conv. `EXPANSION = 4` means `conv3` outputs
`planes * 4` channels, and the downsample branch projects to that width
when `stride != 1` or `in_planes != planes * 4`.

`pub struct ResNet<T: Float>` stores `layer{1..4}` as
`Vec<Box<dyn Module<T>>>` so a `ResNet<T>` can mix BasicBlocks or
Bottlenecks via the private `from_basic` / `from_bottleneck`
constructors. The `make_basic_layer` / `make_bottleneck_layer` helpers
take `&mut current_dilation` and the per-stage `dilate` flag, mirroring
torchvision's `ResNet._make_layer`:

```text
previous_dilation = *current_dilation                                # snapshot
if dilate:
    *current_dilation *= stride
    stride = 1
blocks[0]   = Block::new_full(in_planes, planes, stride,  (previous_dilation, ...))
blocks[i>0] = Block::new_full(planes*E,   planes, 1,      (*current_dilation, ...))
```

`Module::forward` for `ResNet<T>` runs the stem (Conv7×7 stride 2 → BN →
ReLU → MaxPool3×3 stride 2), then each stage in sequence, then
`avgpool` (adaptive to 1×1), flatten to `[B, C]`, and `fc`. The
`children` / `named_children` overrides expose every block as a direct
child so `named_descendants_dyn()` can walk the BN tree end-to-end —
without these overrides the Phase-2 BN-buffer loader (#995) silently
skips every block's running statistics.

### Non-test production consumers

- `pub use resnet::{BasicBlock, Bottleneck, ResNet, resnet18, resnet34, resnet50}` re-export at
  `ferrotorch-vision/src/models/mod.rs:39`.
- `pub use ... resnet18, ...` second-tier re-export in
  `ferrotorch-vision/src/lib.rs` for downstream-crate callers.
- `default_registry()` in `ferrotorch-vision/src/models/registry.rs`
  constructs all three variants via `maybe_load_pretrained`
  (`registry.rs`, `registry.rs`, `registry.rs`) and binds them in the global
  `REGISTRY`.
- `ferrotorch-vision/src/models/segmentation/fcn.rs` and
  `segmentation/deeplabv3.rs` import `resnet50_dilated` for the
  dilated-backbone path used by FCN-ResNet50 / DeepLabV3-ResNet50.
- `ferrotorch-vision/src/models/detection/faster_rcnn.rs:39`,
  `detection/retinanet.rs`, and `detection/fcos.rs` import
  `resnet50` for the FPN backbone in those detection heads.
- `ferrotorch-vision/src/models/feature_extractor.rs:146` (production
  helper, not gated by `#[cfg(test)]`) re-uses `resnet18` to demonstrate
  the `IntermediateFeatures` trait.

## Parity contract

`parity_ops = []`. ResNet composes `Conv2d` (covered under
`ferrotorch-nn/conv`), `BatchNorm2d` (`ferrotorch-nn/norm`), `Linear`
(`ferrotorch-nn/linear`), and the differentiable `add` and `relu` ops
under `ferrotorch-core/grad_fns`. No new op surface.

Edge cases preserved versus torchvision:

- **`replace_stride_with_dilation`**: `current_dilation` snapshot
  semantics. The FIRST block in a dilated stage uses the
  PRE-update dilation (= 1 on first dilated stage) with the original
  stride; SUBSEQUENT blocks use the POST-update dilation with stride 1.
  This is the threading torchvision uses and it is preserved
  bit-for-bit.
- **`maxpool` padding**: torchvision's `MaxPool2d(kernel=3, stride=2,
  padding=1)` — `MaxPool2d::new([3, 3], [2, 2], [1, 1])` here.
- **BN `affine = true`**: every BN carries learnable `weight`/`bias`,
  matching torchvision's `nn.BatchNorm2d` defaults.
- **Conv `bias = false`**: every conv has `bias=false` because a BN
  follows; this matches torchvision's `conv3x3` / `conv1x1` helpers.

## Verification

Tests in `mod tests` in `resnet.rs`:

- `test_basic_block_{same_channels,downsample,has_downsample_when_needed,parameter_count}`
- `test_bottleneck_{same_channels,first_block,downsample,parameter_count}`
- `test_resnet{18,34,50}_{output_shape,param_count}`
- `test_resnet18_named_parameters_prefixes`
- `test_resnet18_custom_classes`
- `test_gradient_flow_through_{basic_block,bottleneck}`
- `test_resnet_train_eval`
- `test_resnet_is_send_sync`

Smoke command:

```bash
cargo test -p ferrotorch-vision --lib resnet:: 2>&1 | tail -3
```

Expected: all tests pass; no `parity-sweep` ops to run.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct BasicBlock<T: Float>` + `Module<T>` impl in `resnet.rs` mirrors torchvision `BasicBlock` at `resnet.py:59`; non-test consumer: `pub use BasicBlock` at `ferrotorch-vision/src/models/mod.rs:39` + invoked inside `ResNet::make_basic_layer` in `resnet.rs`. |
| REQ-2 | SHIPPED | impl: `pub struct Bottleneck<T: Float>` + `Module<T>` impl in `resnet.rs` mirrors torchvision `Bottleneck` at `resnet.py:108`; non-test consumer: `pub use Bottleneck` at `mod.rs` + invoked inside `ResNet::make_bottleneck_layer` in `resnet.rs`. |
| REQ-3 | SHIPPED | impl: `pub struct ResNet<T: Float>` + `Module<T>` impl in `resnet.rs`; non-test consumer: registry constructors at `registry.rs`, `registry.rs`, `registry.rs` construct `ResNet` via `super::resnet::resnet{18,34,50}::<f32>(num_classes)`. |
| REQ-4 | SHIPPED | impl: `pub fn resnet18`, `pub fn resnet34`, `pub fn resnet50` in `resnet.rs`; non-test consumer: `default_registry()` in `registry.rs` binds all three. |
| REQ-5 | SHIPPED | impl: `pub fn resnet50_dilated` + dilation-threading helpers `make_basic_layer` / `make_bottleneck_layer` in `resnet.rs`; non-test consumer: `use crate::models::resnet::{ResNet, resnet50_dilated}` at `segmentation/fcn.rs` and `segmentation/deeplabv3.rs`. |
| REQ-6 | SHIPPED | impl: `Module::named_parameters` for `BasicBlock`, `Bottleneck`, `ResNet` in `resnet.rs`; non-test consumer: `default_registry()` builds models then `load_state_dict(&state_dict, false)` (`named_parameters in registry.rs`) walks `named_parameters` in production. |
| REQ-7 | SHIPPED | impl: `children` / `named_children` overrides in `resnet.rs`; non-test consumer: `apply_bn_buffers_from_state_dict(&model as &dyn Module<T>, &state_dict)` at `registry.rs` walks `named_descendants_dyn()` on the live model. |
| REQ-8 | SHIPPED | impl: `impl IntermediateFeatures<T> for ResNet<T>` in `resnet.rs`; non-test consumer: `pub use feature_extractor::{FeatureExtractor, IntermediateFeatures, create_feature_extractor}` at `ferrotorch-vision/src/models/mod.rs` exposes the trait, and `feature_extractor.rs` re-uses ResNet's impl. |
| REQ-9 | SHIPPED | impl: `Module::train` / `Module::eval` in `resnet.rs` (recursive into every BN); non-test consumer: `Module::train` is a trait method on `Box<dyn Module<T>>`, exposed via `registry.rs::get_model` which returns boxed models whose train/eval the caller drives. |
