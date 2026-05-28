# ferrotorch-vision — `models::segmentation::aspp` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669 (working tree at /home/doll/pytorch)
baseline-torchvision: /home/doll/.local/lib/python3.13/site-packages/torchvision/models/segmentation/deeplabv3.py
upstream-paths:
  - /home/doll/.local/lib/python3.13/site-packages/torchvision/models/segmentation/deeplabv3.py
-->

## Summary

`ferrotorch-vision/src/models/segmentation/aspp.rs` ships the Atrous Spatial
Pyramid Pooling module used by DeepLabV3's head. It mirrors torchvision's
`ASPP` (defined inline in `torchvision/models/segmentation/deeplabv3.py:86`):
five parallel branches (1x1 conv, three 3x3 dilated convs at configurable
atrous rates, global avg-pool branch) followed by a 1x1 projection from
`5 * 256 = 1280` channels back to `256`.

## Requirements

- REQ-1: `pub struct DilatedConv2d<T: Float>` wraps a 3x3 `Conv2d` with explicit
  `dilation` plus a `BatchNorm2d` and inline ReLU, mirroring torchvision's
  `ASPPConv(nn.Sequential)` building block at
  `torchvision/models/segmentation/deeplabv3.py:60`. Same-size padding rule
  (`pad = dilation` for `k=3`) is enforced internally.
- REQ-2: A private `ASPPConv1x1<T: Float>` branch carries the rate-1 1x1 conv
  + BN + ReLU; a private `ASPPPooling<T: Float>` branch carries
  `AdaptiveAvgPool2d(1)` + 1x1 conv + BN + ReLU + bilinear upsample back to
  the input spatial size, matching `ASPPPooling` at
  `deeplabv3.py:70`.
- REQ-3: `pub struct Aspp<T: Float>` composes the five branches plus a
  projection `Conv2d(1280 -> 256, 1x1, bias=false)` + `BatchNorm2d` + ReLU,
  with `Dropout(p=0.5)` applied to the concatenated tensor before the
  projection (training-only). Mirrors `ASPP(nn.Module)` at
  `deeplabv3.py:86`.
- REQ-4: `Aspp::new(in_channels, out_channels, atrous_rates: (usize, usize, usize))`
  takes the three dilated-branch rates as a tuple; torchvision's
  `deeplabv3_resnet50` default is `(12, 24, 36)` (`deeplabv3.py:50`).
- REQ-5: `Aspp::forward` runs five branches, concatenates along axis 1 via
  the autograd-aware `ferrotorch_core::grad_fns::shape::cat`, applies
  Dropout (training only) → project conv → project BN → ReLU. Channel
  concatenation participates in autograd; gradients flow back through every
  branch.
- REQ-6: `named_parameters` exposes the torchvision-shaped path layout
  (`0.conv.weight`, `0.bn.{weight,bias}`, `1.{weight,bn.weight,bn.bias}`,
  `2.{...}`, `3.{...}`, `4.{conv.weight,bn.{weight,bias}}`,
  `project.weight`, `project_bn.{weight,bias}`) so the strict value-parity
  loader can adopt torchvision state dicts.
- REQ-7: `children` / `named_children` overrides expose the full per-branch
  + project + project_bn + dropout child tree so `named_descendants_dyn()`
  can walk every BN under the right dotted path for the BN-buffer loader
  (Phase 4 #995).
- REQ-8: `Module::train` / `Module::eval` propagate to every BN-bearing
  child (`ASPPConv1x1`, `DilatedConv2d` x3, `ASPPPooling`, `project_bn`) so
  BN's eval-mode forward path engages everywhere when the parent toggles
  mode.

## Acceptance Criteria

- [x] AC-1: `Aspp::<f32>::new(2048, 256, (12, 24, 36))` constructs and the
  forward over `[1, 2048, 4, 4]` returns `[1, 256, 4, 4]`
  (`test_aspp_output_shape`).
- [x] AC-2: `Aspp::<f32>::new(256, 256, (6, 12, 18))` preserves spatial dims
  on `[2, 256, 8, 8]` (`test_aspp_preserves_spatial_dims`).
- [x] AC-3: `DilatedConv2d::<f32>::new(64, 64, 6)` preserves spatial dims on
  `[1, 64, 8, 8]` (`test_dilated_conv2d_same_size_output`).
- [x] AC-4: `Aspp::<f32>::new(2048, 256, (12, 24, 36))` total parameter count
  exceeds 2M (`test_aspp_parameter_count`).

## Architecture

`pub struct DilatedConv2d<T: Float>` at `aspp.rs` lines 59-96 carries a
`Conv2d` constructed via `Conv2d::new_full(in, out, (3, 3), (1, 1),
(dilation, dilation), (dilation, dilation), 1, false)` plus a `BatchNorm2d`.
`forward_inner` runs `conv → bn → relu`. The dilated 3x3 branches were
migrated off a host-side `data_vec` + 7-deep CPU loop to `Conv2d`'s native
dilation path in Phase 6 (#988); the prior `requires_grad=false` workaround
is gone and gradients now flow through.

`ASPPConv1x1<T>` (private) and `ASPPPooling<T>` (private) are the
parameter-bearing branches without spatial dilation. `ASPPPooling::forward`
captures the input's `h_in, w_in`, runs the adaptive pool to `1x1`, the 1x1
conv, BN, ReLU, then bilinear-upsamples back to the input spatial size via
`ferrotorch_nn::upsample::interpolate(.., InterpolateMode::Bilinear,
align_corners=false)`.

`pub struct Aspp<T: Float>` at `aspp.rs` lines 361-419 owns `conv1`
(ASPPConv1x1), `conv_r1`, `conv_r2`, `conv_r3` (three DilatedConv2d at the
configured atrous rates), `pool` (ASPPPooling), `project` (Conv2d 1280->256
1x1 bias=false), `project_bn` (BatchNorm2d), `dropout` (Dropout p=0.5), and
a `training: bool`.

`Module::forward` for `Aspp` (lines 423-444):

```text
b1 = conv1(input)
b2 = conv_r1(input)
b3 = conv_r2(input)
b4 = conv_r3(input)
b5 = pool(input)
x  = cat([b1, b2, b3, b4, b5], dim=1)       # [B, 1280, H, W]
x  = if training { dropout(x) } else { x }
x  = project(x)
x  = project_bn(x)
relu(x)
```

The `named_parameters` impl (lines 470-494) emits torchvision's index-keyed
ModuleList layout (`0.<...>` for ASPPConv1x1, `1.<...>` / `2.<...>` /
`3.<...>` for the dilated convs, `4.<...>` for ASPPPooling) plus the
`project.<...>` / `project_bn.<...>` keys. The `DilatedConv2d`
sub-`named_parameters` (lines 115-132) intentionally flattens the inner
Conv2d's `weight` to no-prefix so `Aspp`'s `1.weight` matches torchvision's
state-dict key.

`children` / `named_children` (lines 497-520) list every parameter-bearing
child plus the parameter-free `dropout` so the descendant walk used by the
BN-buffer loader resolves `0.bn`, `1.bn`, `2.bn`, `3.bn`, `4.bn`,
`project_bn` to the underlying `BatchNorm2d` modules.

### Non-test production consumers

- `pub use aspp::Aspp` at
  `ferrotorch-vision/src/models/segmentation/mod.rs` and again at
  `ferrotorch-vision/src/models/mod.rs:40-43` (the segmentation-module
  re-export).
- `DeepLabV3Head::new` in
  `ferrotorch-vision/src/models/segmentation/deeplabv3.rs:243-263`
  constructs an `Aspp::<T>::new(in_channels, 256, atrous_rates)` as the
  first stage of the head, and `DeepLabV3Head::forward` (line 267) drives
  `self.aspp.forward(input)` on every DeepLabV3 inference call.
- `DeepLabV3Head` is itself owned by `DeepLabV3<T>` (`DeepLabV3 in deeplabv3.rs`),
  which is instantiated by the registry closure at
  `ferrotorch-vision/src/models/registry.rs:316`.

## Parity contract

`parity_ops = []`. ASPP composes `Conv2d`, `BatchNorm2d`,
`AdaptiveAvgPool2d`, `Dropout`, `interpolate`, the differentiable `cat`
along axis 1, and `relu` — every primitive is covered by its own
parity-sweep entry under `ferrotorch-core` / `ferrotorch-nn`.

Edge cases preserved versus torchvision:

- **`ASPPPooling` upsample target**: spatial size is captured from the
  *input* (not from a sibling branch), matching
  `deeplabv3.py:79-83`: `size = x.shape[-2:]` then
  `F.interpolate(..., size=size, mode="bilinear", align_corners=False)`.
- **Dropout p=0.5**: `deeplabv3.py:106` — applied to the concatenated
  tensor *before* the projection conv (training only).
- **`align_corners=False`** on the bilinear upsample: matches
  torchvision's call site at `deeplabv3.py:83`.
- **`bias=False`** on every conv: matches torchvision's `nn.Conv2d(..,
  bias=False)` at `deeplabv3.py:63, 74, 91, 103`. The BN that follows
  absorbs the constant.

## Verification

Tests in `mod tests` in `aspp.rs`:

- `test_aspp_output_shape` — `(12, 24, 36)` rates, `[1, 2048, 4, 4]` →
  `[1, 256, 4, 4]`.
- `test_aspp_preserves_spatial_dims` — `(6, 12, 18)` rates,
  `[2, 256, 8, 8]` → `[2, 256, 8, 8]`.
- `test_dilated_conv2d_same_size_output` — `dilation=6` 3x3 kernel keeps
  spatial size.
- `test_aspp_parameter_count` — `np > 2_000_000`.

Smoke command:

```bash
cargo test -p ferrotorch-vision --lib segmentation::aspp:: 2>&1 | tail -3
```

Expected: all four tests pass; no `parity-sweep` ops.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct DilatedConv2d<T: Float>` + `Module<T>` impl at `aspp.rs` lines 59-163, calls `Conv2d::new_full(.., dilation, .., bias=false)` mirroring torchvision `ASPPConv` at `deeplabv3.py:60`; non-test consumer: `Aspp::new` constructs three `DilatedConv2d` instances at `DilatedConv2d in aspp.rs` which is itself invoked by `DeepLabV3Head::new` in `deeplabv3 in deeplabv3.rs`. |
| REQ-2 | SHIPPED | impl: private `ASPPConv1x1<T>` at `ASPPConv1x1 in aspp.rs` and `ASPPPooling<T>` at `ASPPPooling in aspp.rs` mirroring `deeplabv3.py:70-83`; non-test consumer: `Aspp::new` constructs both at `new in aspp.rs, 394`, driven by `DeepLabV3Head::new` in `deeplabv3 in deeplabv3.rs`. |
| REQ-3 | SHIPPED | impl: `pub struct Aspp<T: Float>` at `Aspp in aspp.rs` + `Module<T>` impl at `Aspp in aspp.rs`; non-test consumer: `DeepLabV3Head::new` constructs `Aspp::new(in_channels, 256, atrous_rates)` at `deeplabv3 in deeplabv3.rs`; the `Aspp` instance's `forward` is driven on every DeepLabV3 inference call at `deeplabv3 in deeplabv3.rs`. |
| REQ-4 | SHIPPED | impl: `Aspp::new(in_channels, out_channels, atrous_rates: (usize, usize, usize))` at `new in aspp.rs` plumbs the three rates as a tuple; non-test consumer: `DeepLabV3::with_atrous_rates(num_classes, atrous_rates)` at `deeplabv3 in deeplabv3.rs` forwards `atrous_rates` to `DeepLabV3Head::new` to `Aspp::new`, default `(12, 24, 36)` chosen at `deeplabv3 in deeplabv3.rs`. |
| REQ-5 | SHIPPED | impl: `Module::forward` for `Aspp in aspp.rs` runs the five branches, calls `ferrotorch_core::grad_fns::shape::cat(&[b1..b5], 1)` at `cat in aspp.rs`, applies dropout/project/project_bn/relu; non-test consumer: `DeepLabV3Head::forward` invokes `self.aspp.forward(input)` at `deeplabv3 in deeplabv3.rs`. |
| REQ-6 | SHIPPED | impl: `Module::named_parameters` for `Aspp in aspp.rs` emits index-keyed `0.<...>` through `4.<...>` + `project.<...>` + `project_bn.<...>`; non-test consumer: the strict-value-parity loader uses `model.named_parameters()` via `model.load_state_dict(&state_dict, false)` invoked by `maybe_load_pretrained` in `0 in registry.rs` for the `deeplabv3_resnet50` closure at `0 in registry.rs`. |
| REQ-7 | SHIPPED | impl: `Module::children` and `Module::named_children` for `Aspp in aspp.rs` expose every parameter-bearing branch + project + project_bn + dropout; non-test consumer: `apply_bn_buffers_from_state_dict` at `0 in registry.rs` walks `named_descendants_dyn()` to resolve `0.bn`, `1.bn`, ..., `project_bn` paths on the live DeepLabV3 model. |
| REQ-8 | SHIPPED | impl: `Module::train` / `Module::eval` for `Aspp in aspp.rs` recursively call train/eval on `conv1`, `conv_r1..3`, `pool`, `project_bn`; non-test consumer: `DeepLabV3::eval` and `DeepLabV3::train` (`deeplabv3 in deeplabv3.rs`) drive the parent toggle which propagates here. |
