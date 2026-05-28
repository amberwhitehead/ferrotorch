# ferrotorch-vision — `models::vgg` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669 (working tree at /home/doll/pytorch)
baseline-torchvision: /home/doll/.local/lib/python3.13/site-packages/torchvision/models/vgg.py
upstream-paths:
  - /home/doll/.local/lib/python3.13/site-packages/torchvision/models/vgg.py
-->

## Summary

`ferrotorch-vision/src/models/vgg.rs` ships the Simonyan & Zisserman
2014 VGG-11 and VGG-16 BN-free variants. Phase 6 (#1004) restored
torchvision-flat indexing: `features` is a flat `Vec<Box<dyn Module>>`
of `Conv2d`/`ReLU`/`MaxPool2d` and `classifier` is a flat
`Vec<Box<dyn Module>>` of `Linear`/`ReLU`/`Dropout` so
`named_parameters` produces the indices torchvision's
`vgg{11,16}` state dict uses (`features.0.weight`, `features.3.weight`,
..., `classifier.0.weight`, `classifier.3.weight`,
`classifier.6.weight`).

## Requirements

- REQ-1: `pub struct VGG<T: Float>` carries `features:
  Vec<Box<dyn Module<T>>>`, `avgpool: AdaptiveAvgPool2d`, and
  `classifier: Vec<Box<dyn Module<T>>>`. `Module::forward` runs
  `features → avgpool → flatten → classifier`.
- REQ-2: `vgg11_cfg()` returns the per-block channel config
  `[64, M, 128, M, 256, 256, M, 512, 512, M, 512, 512, M]` and
  `vgg16_cfg()` returns the longer `[64, 64, M, ...]` config — both
  matching torchvision's `cfgs["A"]` and `cfgs["D"]` exactly.
- REQ-3: `make_features` builds the feature extractor from the config:
  each `Conv(out_ch)` becomes `Conv2d(in_ch, out_ch, 3, pad=1,
  bias=true)` + `ReLU`; each `Pool` becomes `MaxPool2d(2, 2)`. Conv and
  ReLU are pushed as SEPARATE flat entries (this is what produces
  `features.0.weight`, `features.3.weight`, ... — the indices torchvision
  ships).
- REQ-4: `make_classifier` builds the 3-FC head as flat entries:
  `Linear(512*7*7, 4096)`, `ReLU`, `Dropout(0.5)`, `Linear(4096, 4096)`,
  `ReLU`, `Dropout(0.5)`, `Linear(4096, num_classes)` so
  `named_parameters` yields `classifier.{0,3,6}.{weight,bias}`.
- REQ-5: `pub fn vgg11` / `pub fn vgg16` are the canonical
  constructors. VGG-11 has ~132.9M parameters; VGG-16 has ~138.4M.
- REQ-6: `named_parameters` returns torchvision-shaped paths so the
  pretrained-weight loader can adopt a torchvision state dict without
  remap.
- REQ-7: `named_children` exposes every `features.<i>` and
  `classifier.<i>` entry plus `avgpool` so `named_descendants_dyn()`
  can walk the tree (#995 sweep contract: every vision model overrides
  `named_children`).
- REQ-8: `VGG<T>` implements `IntermediateFeatures<T>` exposing every
  layer-index activation in `forward_features` for downstream feature
  composition (used by `ssd300_vgg16` detection head).

## Acceptance Criteria

- [x] AC-1: `vgg11::<f32>(1000)` constructs and `Module::forward` on
  `[1, 3, 224, 224]` returns `[1, 1000]` (`test_vgg11_output_shape`).
- [x] AC-2: `vgg11::<f32>(1000)` has between 132 000 000 and
  134 000 000 parameters (`test_vgg11_param_count`).
- [x] AC-3: `vgg16::<f32>(1000)` constructs and forwards
  (`test_vgg16_output_shape`).
- [x] AC-4: `vgg16::<f32>(1000)` has between 138 000 000 and
  139 000 000 parameters (`test_vgg16_param_count`).
- [x] AC-5: `named_parameters` includes `features.` and `classifier.`
  prefixes (`test_vgg{11,16}_named_parameters_prefixes`).
- [x] AC-6: `vgg{11,16}::<f32>(10)` works with a custom class count
  (`test_vgg{11,16}_custom_classes`).

## Architecture

`pub struct VGG<T: Float>` in `vgg.rs` stores `features`, `avgpool`, and
`classifier` as flat lists. The private `enum VggCfg { Conv(usize), Pool
}` and the `vgg11_cfg()` / `vgg16_cfg()` constants encode the per-block
channel counts.

`make_features` walks the config: each `Conv(out)` pushes a
`Conv2d::new(in_ch, out, (3,3), (1,1), (1,1), true)` (`bias=true` matches
torchvision's default `nn.Conv2d(... bias=True)` for the BN-free VGG
variant; Phase 4 #1001 closed the prior `bias=false` divergence) and
then pushes a `ReLU::new()`. Pool entries push a `MaxPool2d::new([2, 2],
[2, 2], [0, 0])`. The flat layout means `features.0` is the first conv,
`features.1` is the ReLU after it, `features.2` is the next conv (or a
MaxPool if a pool came next), etc. — exactly what torchvision's
state-dict key indexing uses.

`make_classifier` returns a `Vec<Box<dyn Module<T>>>` of 7 entries:
`Linear(512*7*7, 4096)`, `ReLU`, `Dropout(0.5)`, `Linear(4096, 4096)`,
`ReLU`, `Dropout(0.5)`, `Linear(4096, num_classes)`. The Linears land at
indices 0, 3, 6 — the dropouts at 2 and 5 are parameter-free so they
don't appear in `named_parameters`.

`Module::forward` runs each `features.<i>` in order on the input, then
`avgpool` to `[B, 512, 7, 7]`, then `reshape` to `[B, 512*7*7]`, then
each `classifier.<i>` in order. The forward exists in two near-identical
forms — the main `Module::forward` and the `IntermediateFeatures`
version that captures per-layer activations.

### Non-test production consumers

- `pub use vgg::{VGG, vgg11, vgg16}` re-export at
  `ferrotorch-vision/src/models/mod.rs`.
- `default_registry()` registers both `"vgg11"` and `"vgg16"` via
  `maybe_load_pretrained` in `registry.rs` (around lines 160 and 169).
- `ferrotorch-vision/src/models/detection/ssd.rs` uses `VGG` (via the
  factory `super::vgg::vgg16`) as the SSD300 backbone (registered as
  `"ssd300_vgg16"` at `registry.rs`).

## Parity contract

`parity_ops = []`. VGG composes `Conv2d`, `MaxPool2d`,
`AdaptiveAvgPool2d`, `Linear`, `ReLU`, `Dropout` — all covered by
`ferrotorch-nn` parity sweeps. No new op surface.

Edge cases preserved:

- **`bias=true` on convs**: Phase 4 (#1001) ensured `Conv2d::new(...,
  true)` matches torchvision's `nn.Conv2d(...).bias` default
  (`vgg.py:81`).
- **Dropout in eval is identity**: `Dropout::forward` in eval-mode is a
  pass-through, matching `nn.Dropout(...).eval()`.
- **Flat `features` / `classifier`**: `features.0` is the FIRST CONV;
  `features.1` is the FIRST RELU (not a Conv+ReLU compound). Without
  this, the state-dict keys would be `features.0.weight` (compound) vs
  torchvision's `features.0.weight` (conv-only) — they print the same
  but reference different tensors. Phase 6 (#1004) closed this.
- **`AdaptiveAvgPool2d((7, 7))`**: matches torchvision's
  `vgg.py:42`.
- **Classifier head dim `512*7*7`**: matches torchvision's
  `vgg.py:44`.

## Verification

Tests in `mod tests` in `vgg.rs`:

- `test_vgg{11,16}_output_shape`
- `test_vgg{11,16}_param_count`
- `test_vgg{11,16}_custom_classes`
- `test_vgg{11,16}_named_parameters_prefixes`
- `test_vgg_train_eval`
- `test_vgg_is_send_sync`

Smoke command:

```bash
cargo test -p ferrotorch-vision --lib vgg:: 2>&1 | tail -3
```

Expected: all tests pass; no parity-sweep ops.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct VGG<T: Float>` + `Module<T>` impl in `vgg.rs` mirrors torchvision `VGG` at `vgg.py:35`; non-test consumer: `pub use VGG` at `ferrotorch-vision/src/models/mod.rs` + `default_registry()` constructs both variants in `registry.rs`. |
| REQ-2 | SHIPPED | impl: `vgg11_cfg` / `vgg16_cfg` in `vgg.rs` mirror torchvision `cfgs["A"]` / `cfgs["D"]` at `vgg.py:90`; non-test consumer: `VGG::from_cfg` invoked by `pub fn vgg{11,16}` in `vgg.rs`. |
| REQ-3 | SHIPPED | impl: `make_features` in `vgg.rs` (flat-entries layout) mirrors torchvision `make_layers` at `vgg.py:73`; non-test consumer: `VGG::from_cfg` calls it in `vgg.rs`. |
| REQ-4 | SHIPPED | impl: `make_classifier` in `vgg.rs` (flat 7-entry classifier) mirrors torchvision `nn.Sequential(Linear, ReLU, Dropout, ...)` at `vgg.py:43`; non-test consumer: `VGG::from_cfg` in `vgg.rs` calls it. |
| REQ-5 | SHIPPED | impl: `pub fn vgg11` / `pub fn vgg16` in `vgg.rs`; non-test consumer: `default_registry()` in `registry.rs` binds both via `maybe_load_pretrained`. |
| REQ-6 | SHIPPED | impl: `Module::named_parameters` for `VGG<T>` in `vgg.rs`; non-test consumer: `load_state_dict(&state_dict, false)` at `named_parameters in registry.rs` walks the result in production. |
| REQ-7 | SHIPPED | impl: `children` / `named_children` for `VGG<T>` in `vgg.rs`; non-test consumer: `apply_bn_buffers_from_state_dict(&model as &dyn Module<T>, &state_dict)` at `registry.rs` walks the tree (the call is a no-op for VGG since there are no BNs, but the consumer site is real). |
| REQ-8 | SHIPPED | impl: `impl IntermediateFeatures<T> for VGG<T>` in `vgg.rs`; non-test consumer: `pub use feature_extractor::{IntermediateFeatures, create_feature_extractor}` exposes the trait via `mod.rs`; `feature_extractor.rs` is the production helper. |
