# ferrotorch-vision — `models::unet` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669 (working tree at /home/doll/pytorch)
baseline-torchvision: N/A (torchvision does not ship U-Net; ferrotorch carries it as a vision-domain
classic alongside the torchvision-mirrored model trio)
upstream-paths:
  - /home/doll/pytorch/torch/nn/modules/conv.py
  - /home/doll/pytorch/torch/nn/modules/pooling.py
  - /home/doll/pytorch/torch/nn/functional.py
-->

## Summary

`ferrotorch-vision/src/models/unet.rs` ships the U-Net architecture for
semantic segmentation as described in Ronneberger et al. 2015 ("U-Net:
Convolutional Networks for Biomedical Image Segmentation"). Torchvision
itself does not ship U-Net (it predates the segmentation-models lineage in
torchvision and lives in `segmentation-models-pytorch` / community
repositories); ferrotorch carries this file as a vision-domain classic
alongside the torchvision-mirrored segmentation trio
(DeepLabV3 / FCN / LRASPP). The primitives used (`Conv2d`, `MaxPool2d`,
`relu`, autograd-aware `cat` and a custom nearest-2x upsample) are all
direct PyTorch counterparts.

## Requirements

- REQ-1: A private `upsample_nearest_2x<T: Float>(input)` op replicates a
  `[B, C, H, W]` tensor into `[B, C, 2H, 2W]` by 2x2 block replication.
  The op participates in autograd via the custom `UpsampleNearest2xBackward`
  GradFn that sums each 2x2 block back into the corresponding input
  element (forward = replicate; VJP = sum-pool with stride 2). Mirrors
  `nn.functional.interpolate(..., mode="nearest", scale_factor=2)` with
  its trivial backward pass.
- REQ-2: A private `EncoderBlock<T>` carries two 3x3 convs (bias=false)
  with intervening ReLU. `forward` = `Conv → ReLU → Conv → ReLU`. The
  block deliberately does NOT pool — the parent applies the pool
  externally so the pre-pool feature map is stored as a skip connection.
- REQ-3: A private `DecoderBlock<T>` upsamples 2x via
  `upsample_nearest_2x`, applies a 1x1 reduce conv (`in_ch → out_ch`),
  concatenates along axis 1 with the matching encoder skip, then runs two
  3x3 convs with ReLU. Mirrors the classic U-Net up-conv-cat-conv-conv
  pattern; `cat` is the autograd-aware
  `ferrotorch_core::grad_fns::shape::cat`.
- REQ-4: `pub struct UNet<T: Float>` carries four encoder stages
  (3→64→128→256→512), a 2x2 max-pool shared across stages, a bottleneck
  (512→1024 with two 3x3 convs + ReLU), four decoder stages
  (1024→512→256→128→64), and a 1x1 classifier head (64→num_classes).
  Spatial dims must be divisible by 16 (four 2x downsamples).
- REQ-5: `Module::forward` runs encoder → bottleneck → decoder → head,
  threading skip connections `s1..s4` from each encoder stage to the
  matching decoder stage. Returns `[B, num_classes, H, W]`.
- REQ-6: `named_parameters` emits the U-Net-shaped paths `enc{1..4}.conv{1,2}.<weight>`,
  `bottleneck.conv{1,2}.<weight>`, `dec{1..4}.{reduce,conv1,conv2}.<weight>`,
  and `head.<weight>` — a self-consistent layout (no upstream
  torchvision counterpart to match against).
- REQ-7: `children` / `named_children` (Phase 4 #995) expose every
  internal Conv2d sub-module under the same dotted-path layout as
  `named_parameters`, satisfying the #995 named_children-coverage sweep
  for vision models. U-Net has no BatchNorm in the current implementation
  (BN was omitted per the module-level comment) so the named_children
  override does not change loader behaviour for any existing fixture,
  but it satisfies the sweep invariant.
- REQ-8: `impl IntermediateFeatures<T> for UNet<T>` (CL-499) exposes
  per-stage features keyed by `"enc1"..."enc4"`, `"bottleneck"`,
  `"dec1"..."dec4"`, `"head"` for feature-extraction / probe workflows.
- REQ-9: `pub fn unet<T: Float>(num_classes: usize)` is the canonical
  constructor, matching the per-architecture naming convention used by
  every other model in `ferrotorch-vision`.

## Acceptance Criteria

- [x] AC-1: `upsample_nearest_2x` on `[2, 4, 3, 3]` returns `[2, 4, 6, 6]`
  (`test_upsample_nearest_2x_shape`).
- [x] AC-2: `upsample_nearest_2x` on `[1, 1, 2, 2]` produces the exact 2x2
  block-replicated pattern (`test_upsample_nearest_2x_values`).
- [x] AC-3: `EncoderBlock::<f32>::new(3, 64).forward(&[1, 3, 16, 16])`
  returns `[1, 64, 16, 16]` (`test_encoder_block_shape`).
- [x] AC-4: `DecoderBlock::<f32>::new(128, 64).forward(&[1, 128, 4, 4],
  &[1, 64, 8, 8])` returns `[1, 64, 8, 8]` (`test_decoder_block_shape`).
- [x] AC-5: `unet::<f32>(21).forward(&[1, 3, 256, 256])` returns `[1, 21,
  256, 256]` (`test_unet_forward_shape`).
- [x] AC-6: Smallest valid spatial size `[1, 3, 16, 16]` → `[1, 2, 16,
  16]` (`test_unet_forward_small`).
- [x] AC-7: Batch dimension propagates (`test_unet_batch_size`).
- [x] AC-8: Exact parameter count `28,937,216` (manual closed-form check
  in `test_unet_parameter_count`).
- [x] AC-9: `named_parameters` contains every expected prefix
  (`test_unet_named_parameters_prefixes`).
- [x] AC-10: `train` / `eval` toggle works (`test_unet_train_eval`).
- [x] AC-11: U-Net is `Send + Sync` (`test_unet_is_send_sync`).
- [x] AC-12: Gradient flows through `upsample_nearest_2x` (each input
  element sees gradient 4.0 when forward output is summed —
  `test_gradient_flow_through_upsample`).

## Architecture

`UpsampleNearest2xBackward<T>` (lines 43-109) is the GradFn for the
custom upsample op. The backward pass iterates the 4-D shape (B, C, H, W),
sums the four corresponding output gradients per input element (`g00`,
`g01`, `g10`, `g11`), and writes to `grad_input`. The output tensor is
constructed via `Tensor::from_storage` then `.to(device)` to preserve the
input device.

`upsample_nearest_2x<T: Float>` (lines 115-164) iterates B, C, H, W and
pushes each input element twice per row across two rows, building the
output buffer directly. The `if is_grad_enabled() && input.requires_grad()
{ Tensor::from_operation(..., grad_fn) } else { Tensor::from_storage(...)
}` branch keeps the autograd hook live only when needed.

`EncoderBlock<T>` and `DecoderBlock<T>` (lines 174-301) are private
helpers — they implement `forward` directly without going through
`Module<T>` (they predate the trait in this file's history). Their
`named_parameters` method takes an explicit `prefix: &str` so the parent
`UNet` can construct the dotted-path layout (`enc1.conv1.<...>` etc.).

`UNet<T>` (lines 333-355) owns four EncoderBlocks, one MaxPool2d (shared),
two bottleneck Conv2ds, four DecoderBlocks, and one head Conv2d. The
`Module::forward` impl (lines 411-439) is a linear sequence threading
`s1..s4` through to the decoder.

`children` / `named_children` (lines 503-559) expose 24 child modules
(every encoder/decoder Conv2d, the pool, the bottleneck convs, the head).

`impl IntermediateFeatures<T> for UNet<T>` (lines 578-635) replays the
forward pass and stashes each major activation under `enc{N}`,
`bottleneck`, `dec{N}`, `head`.

### Non-test production consumers

- `pub use unet::{UNet, unet}` at `ferrotorch-vision/src/models/mod.rs:45`.
- `default_registry()` in `ferrotorch-vision/src/models/registry.rs:222`
  binds the `"unet"` entry via `super::unet::unet::<f32>(num_classes)`
  inside `maybe_load_pretrained`.

## Parity contract

`parity_ops = []`. U-Net composes existing primitives (`Conv2d`,
`MaxPool2d`, `relu`, autograd-aware `cat`, and the custom
`upsample_nearest_2x`). Each primitive carries its own parity coverage.

Edge cases preserved:

- **Skip-connection alignment**: each `dec{i}` receives its matching
  `s{i}` from the encoder. The spatial sizes match exactly because the
  encoder block does NOT pool (the parent applies pool after capturing
  the skip), and the decoder block upsamples by 2x to reach the skip's
  resolution.
- **Channel doubling on cat**: decoder `conv1` takes `2 * out_ch` input
  channels because the skip carries `out_ch` channels and the reduced-
  upsample carries `out_ch` channels post-`reduce`.
- **No BatchNorm**: the module comment at line 20 records that BN was
  omitted; the loss-of-BN affects pretrained-weight inference quality
  but does not affect structural parity with the U-Net paper (which
  predates BatchNorm widespread use).
- **Spatial divisibility**: input H, W must be divisible by 16. The
  smallest valid spatial size in the test suite is 16x16 (4 halvings →
  1x1 bottleneck).

## Verification

Tests in `mod tests` in `unet.rs`:

- `test_upsample_nearest_2x_{shape,values}`,
  `test_encoder_block_shape`,
  `test_decoder_block_shape`.
- `test_unet_forward_{shape,small}`,
  `test_unet_batch_size`,
  `test_unet_parameter_count`,
  `test_unet_named_parameters_prefixes`,
  `test_unet_train_eval`,
  `test_unet_is_send_sync`,
  `test_gradient_flow_through_upsample`.

Smoke command:

```bash
cargo test -p ferrotorch-vision --lib unet:: 2>&1 | tail -3
```

Expected: 11 tests pass; no `parity-sweep` ops to run.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `fn upsample_nearest_2x<T: Float>` at `unet in unet.rs` + `UpsampleNearest2xBackward<T>` at `unet in unet.rs`; non-test consumer: `DecoderBlock::forward` invokes `upsample_nearest_2x(input)` at `unet in unet.rs` which is reached on every `UNet::forward` call via decoder stages at `unet in unet.rs`. |
| REQ-2 | SHIPPED | impl: `struct EncoderBlock<T: Float>` + `forward` at `unet in unet.rs`; non-test consumer: `UNet::forward` calls `self.enc1.forward`, `self.enc2.forward`, `self.enc3.forward`, `self.enc4.forward` at `unet in unet.rs, 416, 419, 422`. |
| REQ-3 | SHIPPED | impl: `struct DecoderBlock<T: Float>` + `forward` at `unet in unet.rs`; non-test consumer: `UNet::forward` calls `self.dec4.forward`, `self.dec3.forward`, `self.dec2.forward`, `self.dec1.forward` at `unet in unet.rs`. |
| REQ-4 | SHIPPED | impl: `pub struct UNet<T: Float>` at `UNet in unet.rs` + `UNet::new` at `UNet in unet.rs`; non-test consumer: `unet` constructor at `unet in unet.rs` returns `UNet::new(num_classes)` to the registry caller `registry.rs`. |
| REQ-5 | SHIPPED | impl: `Module::forward` for `UNet in unet.rs`; non-test consumer: invoked from `default_registry()` via `unet` at `registry.rs` whenever the registry's model is `.forward`'d. |
| REQ-6 | SHIPPED | impl: `Module::named_parameters` for `UNet in unet.rs`; non-test consumer: `maybe_load_pretrained` at `named_parameters in registry.rs` calls `model.load_state_dict(&state_dict, false)` which walks `named_parameters`. |
| REQ-7 | SHIPPED | impl: `children` / `named_children` at `unet in unet.rs` exposing 24 child modules; non-test consumer: `apply_bn_buffers_from_state_dict` at `registry.rs` walks `named_descendants_dyn()` — for UNet the walk finds no BN modules (BN is omitted) and the loader returns Ok without effect, but the override is in place for the sweep invariant. |
| REQ-8 | SHIPPED | impl: `impl IntermediateFeatures<T> for UNet<T>` at `unet in unet.rs`; non-test consumer: `pub use feature_extractor::{FeatureExtractor, IntermediateFeatures, create_feature_extractor}` at `mod.rs` exposes the trait so callers can construct `FeatureExtractor::new(unet_model, vec!["enc4".into()])`. The trait's `feature_node_names()` is exercised at `feature_extractor.rs` test `test_yolo_feature_extractor_roundtrip` for the parallel YOLO impl. |
| REQ-9 | SHIPPED | impl: `pub fn unet<T: Float>(num_classes: usize) -> FerrotorchResult<UNet<T>>` at `unet in unet.rs`; non-test consumer: `registry.rs` calls `super::unet::unet::<f32>(num_classes)` inside the `default_registry()` `maybe_load_pretrained` closure. |
