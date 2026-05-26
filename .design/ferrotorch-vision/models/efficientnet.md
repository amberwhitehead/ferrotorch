# ferrotorch-vision — `models::efficientnet` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669 (working tree at /home/doll/pytorch)
baseline-torchvision: /home/doll/.local/lib/python3.13/site-packages/torchvision/models/efficientnet.py
upstream-paths:
  - /home/doll/.local/lib/python3.13/site-packages/torchvision/models/efficientnet.py
-->

## Summary

`ferrotorch-vision/src/models/efficientnet.rs` ships EfficientNet-B0
(Tan & Le 2019) with full torchvision parity for the MBConv layout
(mobile inverted bottleneck + depthwise + Squeeze-and-Excite). Phase 7
(#1007) replaced the pre-Phase-7 standard-Conv2d placeholder with the
real MBConv stack. The `named_parameters` layout matches
`torchvision.models.efficientnet_b0` exactly so the strict
value-parity loader can adopt the torchvision pretrained state dict
without remap.

## Requirements

- REQ-1: A private `ConvBnSiLU<T: Float>` ships
  torchvision's `Conv2dNormActivation` with SiLU activation. Three
  child indices (`0=Conv2d`, `1=BatchNorm2d`) at the named-child level;
  `BatchNorm2d(eps=1e-3, momentum=1e-2)` per torchvision's
  `_efficientnet` builder. The project step's `apply_silu = false`
  switches the activation off (linear bottleneck projection).
- REQ-2: A private `MBConv<T: Float>` ships torchvision's MBConv with
  optional expand (`None` when `expand_ratio == 1`), a depthwise
  k×k stage, a `SqueezeExcitation` with SiLU/Sigmoid activations and
  `sq = max(1, in_ch / 4)` mirroring
  `partial(SElayer, activation=partial(nn.SiLU))`, and a no-activation
  project. Residual add applied when `stride == 1 && in_ch == out_ch`.
- REQ-3: A const stage table `EFFICIENTNET_B0_STAGES` encodes the 7
  stages with `num_blocks`/`out_ch`/`kernel`/`stride`/`expand_ratio`
  matching torchvision's `_MBConvConfig`-derived stages.
- REQ-4: `pub struct EfficientNet<T: Float>` carries `stem:
  ConvBnSiLU` (`features.0`, 3→32 stride 2), `stages:
  Vec<Vec<MBConv<T>>>` (`features.<i>.<j>` for i ∈ 1..=7),
  `head: ConvBnSiLU` (`features.8`, 320→1280 1×1), `avgpool`, and
  `classifier: Linear(1280, num_classes)` (`classifier.1`).
- REQ-5: `Module::forward` for `EfficientNet<T>` runs `stem → all stages
  (in order, all blocks in each stage) → head → avgpool → flatten →
  classifier`.
- REQ-6: `named_parameters` returns torchvision-shaped paths:
  `features.0.{0,1}.<n>` (stem Conv/BN), `features.<i+1>.<j>.block.<k>.<n>`
  (MBConv inner block — 3 sub-indices when `expand_ratio == 1`, else 4),
  `features.8.{0,1}.<n>` (head), `classifier.1.<n>` (Linear).
- REQ-7: `named_children` exposes the same dotted-path layout so
  `named_descendants_dyn()` can apply BN running statistics through
  every stage.
- REQ-8: `EfficientNet<T>` implements `IntermediateFeatures<T>`
  exposing per-stage activations (`stem_conv`, `stage<j>`,
  `head_conv`, `avgpool`, `fc`).
- REQ-9: `pub fn efficientnet_b0` is the canonical constructor.
- REQ-10: Eval-mode parity is the contract. Stochastic depth wraps the
  MBConv residual in training; in eval it is identity, so the eval-mode
  forward matches torchvision element-wise. Training-mode parity is out
  of scope (separate blocker tracked under Phase 7 finding §15).

## Acceptance Criteria

- [x] AC-1: `efficientnet_b0::<f32>(1000)` constructs and
  `Module::forward` on `[1, 3, 224, 224]` returns `[1, 1000]`
  (`test_efficientnet_b0_output_shape`).
- [x] AC-2: `efficientnet_b0::<f32>(1000)` parameter count is in
  (4 900 000, 5 700 000) — torchvision's reference is 5 288 548
  (`test_efficientnet_b0_param_count_in_range`).
- [x] AC-3: `named_parameters` includes the torchvision-shaped keys
  `features.0.{0,1}.weight`, `features.1.0.block.{0..2}.<n>` (expand
  ratio 1 stage),
  `features.2.0.block.{0..3}.<n>` (expand ratio 6 stage),
  `features.8.{0,1}.weight`, `classifier.1.{weight,bias}`
  (`test_efficientnet_b0_named_parameters_match_torchvision_layout`).
- [x] AC-4: `efficientnet_b0::<f32>(10)` forward on `[2, 3, 224, 224]`
  returns `[2, 10]` (`test_efficientnet_b0_custom_classes`).
- [x] AC-5: `train()` / `eval()` toggle works
  (`test_efficientnet_train_eval`).

## Architecture

`ConvBnSiLU<T: Float>` is the building block. It wraps `Conv2d::new_full
(in_ch, out_ch, ..., dilation=(1, 1), groups, bias=false)` + `BatchNorm2d`
with `apply_silu: bool` toggling the post-BN activation. The named-child
layout (`0=conv`, `1=bn`) mirrors torchvision's `Conv2dNormActivation`
which is a 3-entry Sequential (conv, BN, activation) — the activation
slot is parameter-free so it doesn't appear in `named_parameters`.

`MBConv<T: Float>` wraps the four sub-modules. The optional `expand`
field is `None` when `expand_ratio == 1` (the only such case is stage 1
in B0); otherwise it expands `in_ch → expanded`. `depthwise` is a
`ConvBnSiLU` with `groups = expanded` (depthwise conv). `se` is a
`SqueezeExcitation::new_with_activations(expanded, sq, SiLU,
Sigmoid)` where `sq = max(1, in_ch / 4)` matches torchvision's
`partial(SElayer, ...)`. `project` is a `ConvBnSiLU` with
`apply_silu = false` (linear bottleneck projection).

The `MBConv::named_parameters` impl increments `idx` per child so the
4-sub-index layout shrinks to 3 when `expand` is `None`:

```text
expand_ratio == 1: block.0=depthwise, block.1=SE, block.2=project    (3 entries)
expand_ratio  > 1: block.0=expand, block.1=depthwise, block.2=SE, block.3=project  (4 entries)
```

This matches torchvision's inner Sequential indexing exactly.

`pub struct EfficientNet<T: Float>` ties them together. `head_index()`
returns `1 + stages.len() == 8` so the head's `named_parameters` prefix
is `features.8.<n>` regardless of how the stages count is configured.

### Non-test production consumers

- `pub use efficientnet::{EfficientNet, efficientnet_b0}` re-export at
  `ferrotorch-vision/src/models/mod.rs`.
- `default_registry()` registers `"efficientnet_b0"` via
  `maybe_load_pretrained` at `registry.rs:188`.

## Parity contract

`parity_ops = []`. EfficientNet composes `Conv2d`, `BatchNorm2d`
(`eps=1e-3, momentum=1e-2`), `SqueezeExcitation` (with SiLU + Sigmoid
activations), `Linear`, `AdaptiveAvgPool2d`, and the differentiable
`silu` / `add` ops. All covered upstream. No new op surface.

Edge cases preserved versus torchvision:

- **BN `eps=1e-3, momentum=1e-2`** (not the 1e-5/0.1 defaults). The two
  constants are pinned at the top of `efficientnet.rs` as `EN_BN_EPS` /
  `EN_BN_MOM`.
- **`sq = max(1, in_ch / 4)`** — the SE squeeze size uses the BLOCK
  input channels (not the expanded width), per torchvision.
- **SE activation = SiLU + Sigmoid** — torchvision's `_efficientnet`
  passes `partial(nn.SiLU, inplace=True)` for the squeeze stage and the
  default `nn.Sigmoid` for the scale stage. We construct the SE
  explicitly via `SqueezeExcitation::new_with_activations(...)` to lock
  this.
- **Stochastic depth = eval identity**: training-mode Bernoulli scaling
  is deliberately not implemented for Phase 7's eval-parity push.
- **`features.8` (head) is at the dynamic index `1 + stages.len()`** so
  even if the stage count changes, the head index stays consistent.

## Verification

Tests in `mod tests` in `efficientnet.rs`:

- `test_efficientnet_b0_output_shape`
- `test_efficientnet_b0_param_count_in_range`
- `test_efficientnet_b0_custom_classes`
- `test_efficientnet_b0_named_parameters_match_torchvision_layout`
- `test_efficientnet_train_eval`
- `test_efficientnet_is_send_sync`

Smoke command:

```bash
cargo test -p ferrotorch-vision --lib efficientnet:: 2>&1 | tail -3
```

Expected: all tests pass; no parity-sweep ops.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `struct ConvBnSiLU<T: Float>` + `Module<T>` impl in `efficientnet.rs` mirrors torchvision's `Conv2dNormActivation` (SiLU variant) used by `_efficientnet`; non-test consumer: every `MBConv::new` constructs `ConvBnSiLU::new(...)` in `efficientnet.rs`; `EfficientNet::new` uses it for stem and head. |
| REQ-2 | SHIPPED | impl: `struct MBConv<T: Float>` + `Module<T>` impl in `efficientnet.rs` mirrors torchvision `MBConv` at `efficientnet.py:82`; non-test consumer: `EfficientNet::new` builds the seven stages of MBConvs in `efficientnet.rs`. |
| REQ-3 | SHIPPED | impl: `const EFFICIENTNET_B0_STAGES: [Stage; 7]` in `efficientnet.rs` mirrors torchvision's B0 stage table at `efficientnet.py:_efficientnet_b0_setup`; non-test consumer: `EfficientNet::new` iterates `&EFFICIENTNET_B0_STAGES` to build the model. |
| REQ-4 | SHIPPED | impl: `pub struct EfficientNet<T: Float>` + `EfficientNet::new` in `efficientnet.rs`; non-test consumer: `default_registry()` constructs it via `maybe_load_pretrained` at `registry.rs:188`. |
| REQ-5 | SHIPPED | impl: `Module::forward` for `EfficientNet<T>` in `efficientnet.rs`; non-test consumer: trait method invoked through `Box<dyn Module<T>>` returned by `registry.rs::get_model`. |
| REQ-6 | SHIPPED | impl: `Module::named_parameters` for `EfficientNet<T>` in `efficientnet.rs` (dynamic `head_index`); non-test consumer: `load_state_dict(&state_dict, false)` at `registry.rs:53` walks the result. |
| REQ-7 | SHIPPED | impl: `children` / `named_children` overrides on `ConvBnSiLU`, `MBConv`, `EfficientNet` in `efficientnet.rs`; non-test consumer: `apply_bn_buffers_from_state_dict` at `registry.rs:62` walks `named_descendants_dyn()` for BN running stats. |
| REQ-8 | SHIPPED | impl: `impl IntermediateFeatures<T> for EfficientNet<T>` in `efficientnet.rs`; non-test consumer: `pub use feature_extractor::IntermediateFeatures` at `mod.rs`; `feature_extractor.rs` re-uses it. |
| REQ-9 | SHIPPED | impl: `pub fn efficientnet_b0` in `efficientnet.rs`; non-test consumer: `default_registry()` invokes it at `registry.rs:191`. |
| REQ-10 | SHIPPED | impl: `MBConv::forward` skips `StochasticDepth` and applies a plain `add(&x, input)` when `use_residual`; non-test consumer: same model returned by `default_registry()` runs through `Box<dyn Module<T>>::forward` with `.eval()` in the registry pretrained-weight verify path (`registry.rs:62` via the loader). |
