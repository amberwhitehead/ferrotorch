# ferrotorch-vision — `models::convnext` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669 (working tree at /home/doll/pytorch)
baseline-torchvision: /home/doll/.local/lib/python3.13/site-packages/torchvision/models/convnext.py
upstream-paths:
  - /home/doll/.local/lib/python3.13/site-packages/torchvision/models/convnext.py
-->

## Summary

`ferrotorch-vision/src/models/convnext.rs` ships ConvNeXt-Tiny (Liu et
al. 2022). Phase 6 (#997) migrated the spatial 7×7 conv to a depthwise
`Conv2d::new_full(.., groups=C, ..)`, restoring torchvision parity. Phase
8 (#1005) added the per-channel `layer_scale_gamma` parameter and set
`bias=true` on the depthwise / pointwise / downsample / stem convs to
match torchvision exactly.

## Requirements

- REQ-1: `pub struct ConvNeXtBlock<T: Float>` carries `dwconv:
  Conv2d` (7×7, `groups=C`, `bias=true`), `norm: LayerNorm`,
  `pwconv1: Conv2d` (1×1, C→4C, `bias=true`), `pwconv2: Conv2d` (1×1,
  4C→C, `bias=true`), `layer_scale_gamma: Parameter` (shape `[C, 1, 1]`,
  init `1e-6`), and `gelu: GELU`. Forward:
  `dwconv → channel_layer_norm → pwconv1 → gelu → pwconv2 →
  mul(layer_scale_gamma) → add(input)`.
- REQ-2: `channel_layer_norm` applies LayerNorm on the channel dim of
  `[B, C, H, W]` by permuting to `[B, H, W, C]`, normalizing, then
  permuting back. Both permutes use the autograd-correct
  `Tensor::permute(...).contiguous()` chain (#996, closes #986/#987).
- REQ-3: `Downsample<T: Float>` ships the inter-stage downsample
  `LayerNorm(C) → Conv2d(C, 2C, kernel=2, stride=2, bias=true)`.
- REQ-4: `pub struct ConvNeXt<T: Float>` carries the stem
  (`stem_conv: Conv2d(3, dims[0], 4, 4, bias=true)` and `stem_norm:
  LayerNorm(dims[0])`), 4 stages (`stages: Vec<Vec<ConvNeXtBlock<T>>>`),
  3 downsamples (`downsamples: Vec<Downsample<T>>`), and the head
  (`avgpool: AdaptiveAvgPool2d`, `head_norm: LayerNorm(dims[3])`,
  `head_fc: Linear(dims[3], num_classes)`).
- REQ-5: `Module::forward` runs stem → stage 0 → (downsample, stage)
  for stages 1..=3 → avgpool → flatten → head_norm → head_fc.
- REQ-6: `ConvNeXt::new` validates `depths.len() == 4` and
  `dims.len() == 4`, surfacing `FerrotorchError::InvalidArgument` on
  mismatch.
- REQ-7: `named_parameters` returns the dotted paths
  `stem.conv.`, `stem.norm.`, `stages.<s>.<i>.{dwconv,norm,pwconv1,pwconv2,layer_scale_gamma}.<n>`,
  `downsample.<i>.{norm,conv}.<n>`, `head.norm.<n>`, `head.fc.<n>`. The
  test-side remap (`remap_torchvision_to_ferrotorch_convnext_keys`)
  translates these to/from torchvision's `features.<i>.<j>.block.<k>.<n>` layout for the strict loader.
- REQ-8: `named_children` exposes the same dotted-path tree so
  `named_descendants_dyn()` can walk every sub-module.
- REQ-9: `ConvNeXt<T>` implements `IntermediateFeatures<T>` exposing
  `stem`, `stage<i>`, `avgpool`, `head_fc` activations.
- REQ-10: `pub fn convnext_tiny` is the canonical constructor (depths
  `[3, 3, 9, 3]`, dims `[96, 192, 384, 768]`).

## Acceptance Criteria

- [x] AC-1: `ConvNeXtBlock::<f32>::new(96)` forward on
  `[1, 96, 8, 8]` returns `[1, 96, 8, 8]`
  (`test_convnext_block_output_shape`).
- [x] AC-2: ConvNeXt block parameter count matches the Phase 8 formula
  `C*49 + C + 2*C + (4*C^2 + 4*C) + (4*C^2 + C) + C`
  (`test_convnext_block_parameter_count`).
- [x] AC-3: `Downsample::<f32>::new(96, 192)` halves spatial dims and
  doubles channels (`test_downsample_output_shape`).
- [x] AC-4: `nhwc_from_nchw` and `nchw_from_nhwc` are inverses
  element-wise (`test_nhwc_nchw_roundtrip`).
- [x] AC-5: `convnext_tiny::<f32>(1000)` forward on `[1, 3, 224, 224]`
  returns `[1, 1000]` (`test_convnext_tiny_output_shape`).
- [x] AC-6: `convnext_tiny::<f32>(1000)` parameter count is in
  (27 000 000, 32 000 000) — torchvision reference is 28 589 128
  (`test_convnext_tiny_param_count`).
- [x] AC-7: `named_parameters` includes the dotted prefixes `stem.conv.`,
  `stem.norm.`, `stages.0.`, `stages.3.`, `downsample.0.`,
  `downsample.2.`, `head.norm.`, `head.fc.`
  (`test_convnext_named_parameters_prefixes`).
- [x] AC-8: Backward through the residual / layer_scale_gamma flows
  gradients to the input (`test_gradient_flow_through_convnext_block`).

## Architecture

`pub struct ConvNeXtBlock<T: Float>` carries the depthwise 7×7, the
channel LayerNorm, two pointwise 1×1 convs, the per-channel
`layer_scale_gamma`, and `GELU`. Forward:

```text
out = dwconv(input)                            # [B, C, H, W], depthwise groups=C
out = channel_layer_norm(norm, out)            # permute, LN, permute back
out = pwconv1(out)                             # C → 4C
out = gelu(out)
out = pwconv2(out)                             # 4C → C
scaled = mul(out, layer_scale_gamma)           # γ broadcasts [C,1,1] over [B,C,H,W]
return add(scaled, input)                      # residual
```

`channel_layer_norm` is `nhwc_from_nchw → ln.forward → nchw_from_nhwc`
where both NHWC↔NCHW conversions are
`tensor.permute(...).contiguous()` (no `data_vec()` round-trip).

`Downsample<T: Float>` is the inter-stage `LayerNorm(C) → Conv2d(C, 2C,
2, stride=2, bias=true)`. The Phase 8 (#1005) `bias=true` matches
torchvision's `features.{2,4,6}.1` Conv2d default.

`pub struct ConvNeXt<T: Float>` ties stem, 4 stages, 3 downsamples, and
the head together. Forward:

```text
x = stem_conv(input)
x = channel_layer_norm(stem_norm, x)
for block in stages[0]: x = block(x)
for s in 1..4:
    x = downsamples[s-1](x)
    for block in stages[s]: x = block(x)
x = avgpool(x) → flatten → head_norm(x) → head_fc(x)
```

The `named_parameters` impl prefixes each child with the dotted path.
The test-side remap (`remap_torchvision_to_ferrotorch_convnext_keys` in
`tests/conformance_vision_models.rs`) handles the bridge to
torchvision's `features.<i>.<j>.block.<k>.<n>` layout — since the
ferrotorch tree uses Rust-native `stages.<s>.<i>...` names, this
test-side remap is the only place the two schemas connect.

### Non-test production consumers

- `pub use convnext::{ConvNeXt, ConvNeXtBlock, convnext_tiny}` re-export
  at `ferrotorch-vision/src/models/mod.rs`.
- `default_registry()` registers `"convnext_tiny"` via
  `maybe_load_pretrained` at `registry.rs:213`.

## Parity contract

`parity_ops = []`. ConvNeXt composes `Conv2d` (with depthwise/groups
support), `LayerNorm`, `Linear`, `AdaptiveAvgPool2d`, `GELU`, and the
differentiable `add` / `mul` / `permute` / `contiguous` /
`reshape` ops. No new op surface.

Edge cases preserved versus torchvision:

- **Depthwise 7×7 conv** (`groups=channels`): the grouped CUDA fast path
  is not enabled for `groups>1`, so depthwise blocks transparently fall
  back to the CPU im2col path while preserving the parameter count +
  element-wise output. Phase 6 (#997) closed the prior
  not-actually-depthwise divergence.
- **`bias=true` on dwconv, pwconv1, pwconv2, stem_conv, downsample.conv**:
  Phase 8 (#1005) mirrors torchvision's `nn.Conv2d(..., bias=True)`
  default. Without this the per-block bias parameters would be missing
  from `named_parameters` and the fixture loader would surface
  UnmappedFixtureKey on `features.<i>.<j>.block.{0,3,5}.bias` and
  `features.{0,2,4,6}.0.bias`.
- **`layer_scale_gamma` init `1e-6`**: matches torchvision's
  `layer_scale = 1e-6` CNBlock default.
- **LayerNorm `eps=1e-6`**: matches torchvision's
  `partial(nn.LayerNorm, eps=1e-6)`.

## Verification

Tests in `mod tests` in `convnext.rs`:

- `test_convnext_block_{output_shape,parameter_count,batch_2}`
- `test_downsample_{output_shape,parameter_count}`
- `test_nhwc_nchw_roundtrip`
- `test_convnext_tiny_{output_shape,param_count,custom_classes}`
- `test_small_convnext_forward`
- `test_convnext_named_parameters_prefixes`
- `test_convnext_train_eval`
- `test_gradient_flow_through_convnext_block`
- `test_convnext_is_send_sync`

Smoke command:

```bash
cargo test -p ferrotorch-vision --lib convnext:: 2>&1 | tail -3
```

Expected: all tests pass; no parity-sweep ops.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct ConvNeXtBlock<T: Float>` + `Module<T>` impl in `convnext.rs` mirrors torchvision `CNBlock` at `convnext.py:39`; non-test consumer: `ConvNeXt::new` builds `Vec<Vec<ConvNeXtBlock<T>>>` for the 4 stages in `convnext.rs`. |
| REQ-2 | SHIPPED | impl: `fn channel_layer_norm` + `nhwc_from_nchw` / `nchw_from_nhwc` in `convnext.rs`; non-test consumer: `ConvNeXtBlock::forward` and `Downsample::forward` and `ConvNeXt::forward` all call it in `convnext.rs`. |
| REQ-3 | SHIPPED | impl: `struct Downsample<T: Float>` + `Module<T>` impl in `convnext.rs`; non-test consumer: `ConvNeXt::new` constructs the three inter-stage downsamples in `convnext.rs`. |
| REQ-4 | SHIPPED | impl: `pub struct ConvNeXt<T: Float>` + `ConvNeXt::new` in `convnext.rs`; non-test consumer: `default_registry()` constructs it via `maybe_load_pretrained` at `registry.rs:213`. |
| REQ-5 | SHIPPED | impl: `Module::forward` for `ConvNeXt<T>` in `convnext.rs`; non-test consumer: trait method invoked through `Box<dyn Module<T>>` returned from `registry.rs::get_model`. |
| REQ-6 | SHIPPED | impl: argument validation in `ConvNeXt::new` returns `FerrotorchError::InvalidArgument` on bad depths/dims (`convnext.rs`); non-test consumer: `convnext_tiny` passes the validated `&[3,3,9,3]` / `&[96,192,384,768]`. |
| REQ-7 | SHIPPED | impl: `Module::named_parameters` for `ConvNeXt<T>` in `convnext.rs`; non-test consumer: `load_state_dict(&state_dict, false)` at `registry.rs:53` walks the result. |
| REQ-8 | SHIPPED | impl: `children` / `named_children` overrides on `ConvNeXtBlock`, `Downsample`, `ConvNeXt` in `convnext.rs`; non-test consumer: `apply_bn_buffers_from_state_dict` at `registry.rs:62` walks `named_descendants_dyn()` (ConvNeXt is BN-free, but the consumer site is real). |
| REQ-9 | SHIPPED | impl: `impl IntermediateFeatures<T> for ConvNeXt<T>` in `convnext.rs`; non-test consumer: `pub use feature_extractor::IntermediateFeatures` at `mod.rs`. |
| REQ-10 | SHIPPED | impl: `pub fn convnext_tiny` in `convnext.rs`; non-test consumer: `default_registry()` invokes it at `registry.rs:216`. |
