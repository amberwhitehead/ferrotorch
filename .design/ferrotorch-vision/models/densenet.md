# ferrotorch-vision — `models::densenet` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669 (working tree at /home/doll/pytorch)
baseline-torchvision: /home/doll/.local/lib/python3.13/site-packages/torchvision/models/densenet.py
upstream-paths:
  - /home/doll/.local/lib/python3.13/site-packages/torchvision/models/densenet.py
-->

## Summary

`ferrotorch-vision/src/models/densenet.rs` ships DenseNet-121 (Huang et
al. 2017) end-to-end with BatchNorm. Phase 6 (#989) lands the
BN-bearing version with torchvision-exact parameter naming so the
strict value-parity loader can adopt `densenet121(weights=...)` state
dicts without remap.

## Requirements

- REQ-1: `pub struct DenseLayer<T: Float>` carries `norm1` /`conv1` /
  `norm2` / `conv2` mirroring torchvision's `_DenseLayer`. Forward is
  `BN → ReLU → Conv1×1 (bn_size·growth) → BN → ReLU → Conv3×3
  (growth)`, then `cat([input, new_features], dim=1)` along the channel
  axis.
- REQ-2: `pub struct DenseBlock<T: Float>` carries `Vec<DenseLayer<T>>`.
  Forward runs each layer in sequence (each layer concatenates onto the
  growing channel count). `named_parameters` indexes 1-based
  (`denselayer1`, `denselayer2`, ...) matching torchvision.
- REQ-3: `pub struct TransitionLayer<T: Float>` carries `norm`, `conv`,
  `pool` — `BN → ReLU → Conv1×1 → AvgPool2×2` matching torchvision's
  `_Transition`. Halves channels and spatial dims.
- REQ-4: `pub struct DenseNet<T: Float>` carries the DenseNet-121 stack:
  stem (`conv0` / `norm0` / `pool0`), 4 dense blocks `[6, 12, 24, 16]`
  with growth rate 32 and bn_size 4, 3 transitions, final `norm5`, and
  classifier (`avgpool` / `classifier`).
- REQ-5: `Module::forward` for `DenseNet` follows torchvision exactly:
  stem → blocks/transitions → `norm5` → ReLU (functional) →
  `AdaptiveAvgPool2d(1,1)` → flatten → `classifier`.
- REQ-6: `named_parameters` for `DenseNet` produces torchvision-flat
  keys with the `features.` prefix
  (`features.conv0.weight`, `features.norm0.{weight,bias}`,
  `features.denseblock<i>.denselayer<j>.{norm,conv}<k>.<n>`,
  `features.transition<i>.{norm,conv}.<n>`, `features.norm5.<n>`,
  `classifier.<n>`). BN running statistics live under these paths and
  are reachable via `named_descendants_dyn()`.
- REQ-7: `named_children` overrides on `DenseLayer`, `DenseBlock`,
  `TransitionLayer`, `DenseNet` use the torchvision-shaped paths so the
  Phase-2 BN-buffer loader (`bn_buffer_loader.rs`) walks the tree
  end-to-end.
- REQ-8: `DenseNet<T>` implements `IntermediateFeatures<T>` exposing
  per-stage activations under their torchvision-flat paths so
  `feature_extractor::create_feature_extractor` works against this
  module.
- REQ-9: `pub fn densenet121` is the canonical constructor.

## Acceptance Criteria

- [x] AC-1: `DenseLayer::<f32>::new(4, 2, 4)` produces
  `[1, 6, 8, 8]` from `[1, 4, 8, 8]` input (growth=2 added to in_ch=4)
  (`test_dense_layer_concatenates_output`).
- [x] AC-2: `DenseBlock::<f32>::new(3, 8, 4, 4)` reports
  `output_channels(8, 4) == 20` and forwards to `[1, 20, 8, 8]`
  (`test_dense_block_output_channels_calculation`).
- [x] AC-3: `TransitionLayer::<f32>::new(8, 4)` halves spatial dims and
  produces `[1, 4, 8, 8]` from `[1, 8, 16, 16]`
  (`test_transition_layer_halves_spatial`).
- [x] AC-4: `densenet121::<f32>(10)` forward on `[1, 3, 32, 32]`
  returns `[1, 10]` (`test_densenet121_output_shape`).
- [x] AC-5: `densenet121::<f32>(1000)` parameter count is in
  (7 000 000, 9 000 000) — torchvision's reference is ~7.98M
  (`test_densenet121_param_count`).
- [x] AC-6: `named_parameters` exposes the torchvision-shaped prefixes
  `features.conv0.`, `features.norm0.`,
  `features.denseblock1.denselayer1.`, `features.transition1.`,
  `features.norm5.`, `classifier.`
  (`test_densenet121_named_parameters_prefixes`).

## Architecture

`pub struct DenseLayer<T: Float>` stores `norm1: BatchNorm2d`,
`conv1: Conv2d` (1×1, expand to `bn_size * growth_rate`), `norm2:
BatchNorm2d`, `conv2: Conv2d` (3×3, project to `growth_rate`). Forward:
`norm1(input) → relu → conv1 → norm2 → relu → conv2 → cat([input,
new_features], 1)`. The `cat` along the channel axis is the densely-connected
behavior — each layer's output is the concatenation of every prior
layer's output plus the new `growth_rate` feature maps.

`pub struct DenseBlock<T: Float>` is a `Vec<DenseLayer<T>>`. Forward
threads the input through each layer, with the channel count growing by
`growth_rate` each step. `output_channels(in_ch, growth_rate)` returns
the post-block channel count for the transition layer to size against.
Parameter naming uses 1-based indexing (`denselayer1` ... `denselayerN`)
to match torchvision.

`pub struct TransitionLayer<T: Float>` stores `norm: BatchNorm2d`,
`conv: Conv2d` (1×1, halving channels), `pool: AvgPool2d(2, 2)`. The
ReLU between BN and Conv is functional (no parameters) — its
`named_parameters` skips it cleanly, matching torchvision's
`_Transition` which has `norm`/`conv`/`pool` only as state-dict-visible
children.

`pub struct DenseNet<T: Float>` is the full DenseNet-121: stem
(`conv0`/`norm0`/`pool0`), four dense blocks `[6, 12, 24, 16]` with
growth rate 32 and bn_size 4, three transitions (halving channels and
spatial), a final `norm5`, an `AdaptiveAvgPool2d(1, 1)`, and a `Linear`
classifier. `Module::forward` runs the full pipeline.

The `named_parameters` impl for `DenseNet` prefixes every internal name
with `features.` (matching torchvision's `densenet.features` Sequential
wrapper). The internal naming is already torchvision-exact (`conv0`,
`norm0`, `denseblock<i>.denselayer<j>.{norm,conv}<k>`, ...), so the
final `named_parameters` walks are byte-identical to torchvision's.

### Non-test production consumers

- `pub use densenet::{DenseBlock, DenseLayer, DenseNet, TransitionLayer, densenet121}` at
  `ferrotorch-vision/src/models/mod.rs`.
- `default_registry()` in `registry.rs:248` binds `"densenet121"` to
  `super::densenet::densenet121::<f32>(num_classes)` via
  `maybe_load_pretrained`.

## Parity contract

`parity_ops = []`. DenseNet composes `BatchNorm2d`, `Conv2d`, `Linear`,
`AvgPool2d`, `AdaptiveAvgPool2d`, `MaxPool2d`, the differentiable `relu`
activation, and `cat` (channel-axis concatenation). All covered by
`ferrotorch-nn` / `ferrotorch-core` parity sweeps. No new op surface.

Edge cases preserved versus torchvision:

- **`cat(dim=1)`** is the channel-axis concatenation that distinguishes
  DenseNet from ResNet (where it would be an `add`). Each layer's
  output grows the channel count by `growth_rate`.
- **`bn_size=4`**: the 1×1 conv1 expands to `4 * growth_rate` channels
  before the 3×3 conv2 projects back to `growth_rate`. Matches
  torchvision's default.
- **Transition halves both**: channels via `Conv1×1(in, in/2)` and
  spatial via `AvgPool2×2`.
- **Final ReLU before classifier**: applied functionally in
  `Module::forward` after `norm5` — torchvision does the same in
  `DenseNet.forward`.

## Verification

Tests in `mod tests` in `densenet.rs`:

- `test_dense_layer_concatenates_output`
- `test_dense_block_output_channels_calculation`
- `test_transition_layer_halves_spatial`
- `test_densenet121_{output_shape,custom_classes,param_count,named_parameters_prefixes,train_eval}`

Smoke command:

```bash
cargo test -p ferrotorch-vision --lib densenet:: 2>&1 | tail -3
```

Expected: all tests pass; no parity-sweep ops.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct DenseLayer<T: Float>` + `Module<T>` impl in `densenet.rs` mirrors torchvision `_DenseLayer` at `densenet.py:31`; non-test consumer: `pub use DenseLayer` at `ferrotorch-vision/src/models/mod.rs` + `DenseBlock::new` constructs them in `densenet.rs`. |
| REQ-2 | SHIPPED | impl: `pub struct DenseBlock<T: Float>` + `Module<T>` impl in `densenet.rs`; non-test consumer: `DenseNet::new` constructs four DenseBlocks in `densenet.rs` (`denseblock1..denseblock4`). |
| REQ-3 | SHIPPED | impl: `pub struct TransitionLayer<T: Float>` + `Module<T>` impl in `densenet.rs` mirrors torchvision `_Transition`; non-test consumer: `DenseNet::new` constructs three transitions in `densenet.rs`. |
| REQ-4 | SHIPPED | impl: `pub struct DenseNet<T: Float>` + `DenseNet::new` in `densenet.rs`; non-test consumer: `default_registry()` constructs `densenet121` via `maybe_load_pretrained` at `registry.rs:248`. |
| REQ-5 | SHIPPED | impl: `Module::forward` for `DenseNet<T>` in `densenet.rs` mirrors torchvision `DenseNet.forward`; non-test consumer: `Module::forward` is a trait method called by `Box<dyn Module<T>>` returned from `registry.rs::get_model`. |
| REQ-6 | SHIPPED | impl: `Module::named_parameters` for `DenseNet<T>` in `densenet.rs` (prefixes with `features.`); non-test consumer: `load_state_dict(&state_dict, false)` at `registry.rs:53` walks the result. |
| REQ-7 | SHIPPED | impl: `children` / `named_children` overrides on `DenseLayer`, `DenseBlock`, `TransitionLayer`, `DenseNet` in `densenet.rs`; non-test consumer: `apply_bn_buffers_from_state_dict` at `registry.rs:62` walks `named_descendants_dyn()` to apply BN running stats. |
| REQ-8 | SHIPPED | impl: `impl IntermediateFeatures<T> for DenseNet<T>` in `densenet.rs`; non-test consumer: `pub use feature_extractor::IntermediateFeatures` exposes the trait at `mod.rs`; `feature_extractor.rs` is the production composition site. |
| REQ-9 | SHIPPED | impl: `pub fn densenet121` in `densenet.rs`; non-test consumer: `default_registry()` invokes it at `registry.rs:251`. |
