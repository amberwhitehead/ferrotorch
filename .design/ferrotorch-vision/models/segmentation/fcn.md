# ferrotorch-vision — `models::segmentation::fcn` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669 (working tree at /home/doll/pytorch)
baseline-torchvision: /home/doll/.local/lib/python3.13/site-packages/torchvision/models/segmentation/fcn.py
upstream-paths:
  - /home/doll/.local/lib/python3.13/site-packages/torchvision/models/segmentation/fcn.py
  - /home/doll/.local/lib/python3.13/site-packages/torchvision/models/segmentation/_utils.py
-->

## Summary

`ferrotorch-vision/src/models/segmentation/fcn.rs` ships the Fully
Convolutional Network with ResNet-50 dilated backbone, mirroring
`torchvision.models.segmentation.fcn_resnet50` (torchvision 0.21.x). The
file declares an `FcnHead` (3x3 conv → BN → ReLU → Dropout(0.1) → 1x1
classifier), the composite `Fcn` model, and the `fcn_resnet50` constructor.

## Requirements

- REQ-1: `pub struct FcnHead<T: Float>` mirrors torchvision's `FCNHead` at
  `torchvision/models/segmentation/fcn.py:35`: a 5-slot Sequential with
  index 0 = `Conv2d(in_channels, in_channels/4, 3x3, pad=1, bias=False)`,
  index 1 = `BatchNorm2d(in_channels/4)`, index 2 = ReLU (parameter-free,
  applied inline), index 3 = `Dropout(p=0.1)`, index 4 = `Conv2d(in_channels/4,
  num_classes, 1x1, bias=True)`. The intermediate width = `in_channels // 4`
  follows the upstream constant.
- REQ-2: `FcnHead::forward` applies conv → BN → ReLU → (dropout in training
  mode) → classifier. Dropout is skipped when `training=false` (mirrors
  torchvision's BN/Dropout eval semantics).
- REQ-3: `FcnHead::named_parameters` exposes the torchvision-shaped indexed
  keys `0.weight`, `1.{weight,bias}`, `4.{weight,bias}` (slot 3 = Dropout
  is parameter-free and skipped). Used by the strict-value-parity loader.
- REQ-4: `pub struct Fcn<T: Float>` owns a dilated ResNet-50 backbone
  (`resnet50_dilated::<T>(1000, [false, true, true])`) and an `FcnHead`.
  The backbone's classifier head (`fc.*`) is unused; `parameters`,
  `parameters_mut`, `named_parameters` filter `fc.*` so the loader's view
  matches torchvision's `IntermediateLayerGetter`-stripped backbone schema.
  Mirrors `_fcn_resnet` at `torchvision/models/segmentation/fcn.py` (which
  uses `IntermediateLayerGetter(backbone, return_layers={"layer4": "out"})`).
- REQ-5: `Fcn::forward` runs `backbone.forward_features` to extract layer4
  (`[B, 2048, H/16, W/16]` thanks to the dilated stride), runs the FCN
  head, then bilinear-upsamples to `[B, num_classes, H, W]` via
  `interpolate(.., InterpolateMode::Bilinear, false)`.
- REQ-6: `pub fn fcn_resnet50<T: Float>(num_classes: usize)` is the
  user-facing constructor, defaulting to eval mode (matching
  `torchvision.models.segmentation.fcn_resnet50(weights=None,
  num_classes=21)`).
- REQ-7: `named_parameters` for the composite model emits `backbone.<...>`
  and `classifier.<...>` (the latter being the torchvision top-level head
  prefix — `classifier`, not `head`).
- REQ-8: `children` / `named_children` overrides expose `backbone:
  ResNet<T>` at path `"backbone"` and `head: FcnHead<T>` at path
  `"classifier"`, so the BN-buffer loader's descendant walk reaches every
  BN in the backbone AND the head's `1.<...>` BN slot.

## Acceptance Criteria

- [x] AC-1: `fcn_resnet50::<f32>(21).forward(&[1, 3, 32, 32])` returns
  `[1, 21, 32, 32]` (`test_fcn_output_shape_small`).
- [x] AC-2: `fcn_resnet50::<f32>(21).forward(&[1, 3, 64, 64])` returns
  `[1, 21, 64, 64]` (`test_fcn_output_shape_64x64`).
- [x] AC-3: `fcn_resnet50::<f32>(21).forward(&[2, 3, 32, 32])` returns
  `[2, 21, 32, 32]` (`test_fcn_batch_size_2`).
- [x] AC-4: `fcn_resnet50::<f32>(5).forward(&[1, 3, 32, 32])` returns
  `[1, 5, 32, 32]` (`test_fcn_custom_num_classes`).
- [x] AC-5: `named_parameters` contains `backbone.` AND `classifier.`
  prefixed names (`test_fcn_named_parameter_prefixes`).
- [x] AC-6: Total parameter count > 25M, matching torchvision's ~32.9M
  (`test_fcn_param_count_sanity`).
- [x] AC-7: `train` / `eval` toggle works (`test_fcn_train_eval_toggle`).

## Architecture

`pub struct FcnHead<T: Float>` (lines 53-59) owns `conv` (Conv2d 2048→512
3x3 pad=1 bias=False), `bn` (BatchNorm2d 512), `dropout` (Dropout p=0.1),
`classifier` (Conv2d 512→num_classes 1x1 bias=True), and `training: bool`.
The bias=True on the classifier is the Phase 6 (#994) structural fix that
closed the parity-blocker against torchvision's state-dict
`classifier.4.bias` entry.

`Module::forward` for `FcnHead` (lines 87-98):

```text
x = conv(input)
x = bn(x)
x = relu(x)
x = if training { dropout(x) } else { x }
classifier(x)
```

`pub struct Fcn<T: Float>` (lines 177-181) owns `backbone: ResNet<T>` and
`head: FcnHead<T>`. `Fcn::new` constructs `resnet50_dilated::<T>(1000,
[false, true, true])` directly (Phase 6 #994 closed the prior
non-dilated-backbone divergence). `Module::forward` (lines 207-230) calls
`backbone.forward_features`, pulls the `"layer4"` HashMap entry, runs the
head, and bilinear-upsamples to the input spatial size.

The composite model's `parameters`, `parameters_mut`, and `named_parameters`
filter `fc.*` from the inner ResNet's keys (lines 232-296) — the
torchvision FCN state-dict has no `backbone.fc.*` keys because
`IntermediateLayerGetter` strips them.

`named_children` (lines 304-309) exposes `backbone` at path `"backbone"`
and the FCN head at path `"classifier"`. This matches the torchvision
top-level `classifier.<...>` prefix used in the safetensors state dict.

### Non-test production consumers

- `pub use fcn::{Fcn, FcnHead, fcn_resnet50}` at
  `ferrotorch-vision/src/models/segmentation/mod.rs:23` and
  `ferrotorch-vision/src/models/mod.rs:40-43`.
- `default_registry()` in `ferrotorch-vision/src/models/registry.rs:324-326`
  binds the `"fcn_resnet50"` entry, constructing the model via
  `super::segmentation::fcn_resnet50::<f32>(num_classes)` inside
  `maybe_load_pretrained`.

## Parity contract

`parity_ops = []`. FCN composes `Conv2d`, `BatchNorm2d`, `Dropout`,
`relu`, the dilated ResNet-50 backbone, and `interpolate`. Every primitive
is covered under its own parity entry.

Edge cases preserved versus torchvision:

- **Intermediate width**: `inter = in_channels / 4 = 2048 / 4 = 512`.
  Matches `FCNHead.__init__` in `torchvision/models/segmentation/fcn.py:37`
  (`inter_channels = in_channels // 4`).
- **Dropout p**: `0.1`, matching `fcn.py:42`.
- **Final classifier bias**: `bias=True` on the 1x1 conv, the loader's
  `classifier.4.bias` entry depends on this (Phase 6 #994).
- **Backbone dilation**: `replace_stride_with_dilation=[False, True, True]`
  matching torchvision's `fcn_resnet50` reference (the previous module-comment
  diagram of `H/8, W/8` predates Phase 6 #994 and is incorrect; the head
  receives `H/16, W/16` features under the dilated backbone — this is a
  documentation-only error in the source file, the implementation matches
  torchvision).
- **`align_corners=False`** on the bilinear upsample.

## Verification

Tests in `mod tests` in `fcn.rs`:

- `test_fcn_output_shape_{small,64x64}` — 32x32 and 64x64 inputs.
- `test_fcn_batch_size_2` — batch dimension propagates.
- `test_fcn_custom_num_classes` — `num_classes=5`.
- `test_fcn_named_parameter_prefixes` — `backbone.` AND `classifier.`
  exist (note: Phase 6 #994 renamed the top-level head prefix `head.` →
  `classifier.`).
- `test_fcn_param_count_sanity` — `np > 25_000_000`.
- `test_fcn_train_eval_toggle`.

Smoke command:

```bash
cargo test -p ferrotorch-vision --lib segmentation::fcn:: 2>&1 | tail -3
```

Expected: all tests pass; no `parity-sweep` ops to run.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct FcnHead<T: Float>` at `FcnHead in fcn.rs` + `FcnHead::new` at `FcnHead in fcn.rs` (conv `Conv2d(in, in/4, 3x3, pad=1, bias=false)`, bn `BatchNorm2d(in/4)`, dropout `Dropout(0.1)`, classifier `Conv2d(in/4, num_classes, 1x1, bias=true)`); non-test consumer: `Fcn::new` constructs `FcnHead::new(2048, num_classes)` at `fcn in fcn.rs`. |
| REQ-2 | SHIPPED | impl: `Module::forward` for `FcnHead in fcn.rs` applies conv → bn → relu → optional-dropout → classifier; non-test consumer: `Fcn::forward` invokes `self.head.forward(layer4)` at `fcn in fcn.rs`. |
| REQ-3 | SHIPPED | impl: `Module::named_parameters` for `FcnHead in fcn.rs` emits `0.<...>`, `1.<...>`, `4.<...>`; non-test consumer: `maybe_load_pretrained` at `0 in registry.rs` calls `model.load_state_dict(&state_dict, false)` which walks `named_parameters`. |
| REQ-4 | SHIPPED | impl: `pub struct Fcn<T: Float>` at `Fcn in fcn.rs` + `Fcn::new` at `Fcn in fcn.rs` calls `resnet50_dilated::<T>(1000, [false, true, true])`; impl filters `fc.*` at `fcn in fcn.rs, 263-269, 281-283`; non-test consumer: `fcn_resnet50` constructor at `fcn in fcn.rs` returns `Fcn::new(num_classes)` to the registry caller `registry.rs`. |
| REQ-5 | SHIPPED | impl: `Module::forward` for `Fcn in fcn.rs` runs `backbone.forward_features`, pulls `"layer4"`, runs head, bilinear-upsamples to `[B, num_classes, H, W]`; non-test consumer: invoked from `default_registry()` via `fcn_resnet50` at `registry.rs` whenever the registry's model is `.forward`'d. |
| REQ-6 | SHIPPED | impl: `pub fn fcn_resnet50<T: Float>(num_classes: usize) -> FerrotorchResult<Fcn<T>>` at `fcn_resnet50 in fcn.rs`; non-test consumer: `registry.rs` calls `super::segmentation::fcn_resnet50::<f32>(num_classes)` inside the `default_registry()` `maybe_load_pretrained` closure. |
| REQ-7 | SHIPPED | impl: `Module::named_parameters` for `Fcn in fcn.rs` emits `backbone.<...>` for surviving (non-`fc.*`) ResNet keys and `classifier.<...>` for the head; non-test consumer: `maybe_load_pretrained` at `named_parameters in registry.rs` consumes both prefix families when loading the pretrained state dict. |
| REQ-8 | SHIPPED | impl: `Module::named_children` for `Fcn in fcn.rs` exposes `backbone` and `classifier`; non-test consumer: `apply_bn_buffers_from_state_dict` at `registry.rs` walks `named_descendants_dyn()` to reach every BN under `backbone.<...>` and `classifier.1.<...>`. |
