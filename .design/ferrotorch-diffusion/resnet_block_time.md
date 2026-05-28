# ResnetBlock2DTime

<!--
tier: 3-component
status: draft
baseline-pytorch: /home/doll/pytorch (HEAD)
upstream-paths:
  - diffusers/src/diffusers/models/resnet.py
-->

## Summary

The time-conditioned variant of `ResnetBlock2D` used by the SD UNet.
Mirrors `diffusers.models.resnet.ResnetBlock2D` with
`temb_channels = 1280` (the SD-1.5 setting) and the default
`time_embedding_norm = "default"`: SiLU on the temb, project to
`out_channels`, broadcast into the spatial activation as a `(B, C,
1, 1)` bias between the two conv stacks.

## Requirements

- REQ-1: `ResnetBlock2DTime::new(in_channels, out_channels,
  temb_channels, norm_num_groups, eps)` builds the six-module
  layout: `norm1` (GroupNorm), `conv1` (Conv2d k=3), `time_emb_proj`
  (Linear), `norm2` (GroupNorm), `conv2` (Conv2d k=3), optional
  `conv_shortcut` (Conv2d k=1) iff `in_channels != out_channels`.
- REQ-2: `forward_t(x, temb)` implements the diffusers recipe:
  `h = conv1(silu(norm1(x)))`; `t = time_emb_proj(silu(temb))`;
  `h = h + t.view(B, out, 1, 1)`; `h = conv2(silu(norm2(h)))`;
  `out = h + (x if in==out else conv_shortcut(x))`.
- REQ-3: State-dict layout matches diffusers exactly:
  `norm1.*`, `conv1.*`, `time_emb_proj.*`, `norm2.*`, `conv2.*`,
  `conv_shortcut.*` (optional).
- REQ-4: `Module::forward` (the trait method) errors out because
  the time-conditioned block requires a temb — callers must use the
  explicit `forward_t`.

## Acceptance Criteria

- [x] AC-1: `resnet_time_shape_same_channels` and
  `resnet_time_shape_change_channels` verify shape preservation +
  channel-projection (`resnet_block_time.rs`).
- [x] AC-2: `resnet_time_named_parameters` lists the seven
  diffusers prefixes (`resnet_block_time.rs`).

## Architecture

- `ResnetBlock2DTime<T>` (`ResnetBlock2DTime in resnet_block_time.rs`): six
  module fields + `SiLU` activation + `in_channels`/`out_channels`/
  `training`.
- `new` (`new in resnet_block_time.rs`): allocates each sub-module
  with bias = true (matches diffusers); `conv_shortcut` exists iff
  the channel counts differ.
- `forward_t` (`forward_t in resnet_block_time.rs`): the exact
  diffusers `ResnetBlock2D.forward` with `temb` mode `default`.
  Validates `x.ndim() == 4`, `x.shape()[1] == in_channels`, and
  `temb.ndim() == 2`. Computes the time bias as
  `time_emb_proj(silu(temb)).reshape([B, out, 1, 1])` and adds it
  to the post-conv1 activation. The residual takes the optional 1x1
  shortcut.
- `Module<T>::forward` (`forward in resnet_block_time.rs`) is a
  typestate guard: returns `InvalidArgument` because callers must
  pass the temb via `forward_t`.

Non-test production consumers:

- `ferrotorch-diffusion/src/unet.rs` imports
  `ResnetBlock2DTime`. The UNet's `CrossAttnDownBlock2D::new`,
  `DownBlock2D::new`, `UNetMidBlock2DCrossAttn::new`,
  `CrossAttnUpBlock2D::new`, and `UpBlock2D::new` all call
  `ResnetBlock2DTime::<T>::new(...)` for every resnet they own
  (`unet in unet.rs, 302, 468, 478, 657, 865`).

## Parity contract

`parity_ops = []`. The contract is byte-equivalence with
`diffusers.models.resnet.ResnetBlock2D` when configured with
`temb_channels = 1280` and `time_embedding_norm = "default"`. Edge
cases:

- `x.shape()[1] != in_channels` → `ShapeMismatch`.
- `temb.ndim() != 2` → `ShapeMismatch`.
- `Module::forward` (without temb) → `InvalidArgument`.

## Verification

Three lib tests in `resnet_block_time.rs`:

- `resnet_time_shape_same_channels` — `in==out` path,
  `conv_shortcut is None`, shape preserved.
- `resnet_time_shape_change_channels` — `in!=out` path,
  `conv_shortcut is Some`, channel projection works.
- `resnet_time_named_parameters` — confirms all seven diffusers
  state-dict prefixes (`norm1.weight`, `conv1.weight`,
  `time_emb_proj.weight`/`bias`, `norm2.weight`, `conv2.weight`,
  `conv_shortcut.weight`).

No parity-sweep ops apply.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `ResnetBlock2DTime::new` at `new in ferrotorch-diffusion/src/resnet_block_time.rs`; non-test consumer: `new in ferrotorch-diffusion/src/unet.rs`, `unet in unet.rs`, `unet in unet.rs`, `unet in unet.rs`, `unet in unet.rs`, and `unet in unet.rs` all call `ResnetBlock2DTime::<T>::new(...)` in production UNet blocks |
| REQ-2 | SHIPPED | impl: `forward_t in ferrotorch-diffusion/src/resnet_block_time.rs`; non-test consumer: the UNet block `forward_t` paths in `ferrotorch-diffusion/src/unet.rs` call `resnet.forward_t(&h, temb)?` to apply the time bias |
| REQ-3 | SHIPPED | impl: `named_parameters` at `ferrotorch-diffusion/src/resnet_block_time.rs:182..205` and `load_state_dict` at `ferrotorch-diffusion/src/resnet_block_time.rs:215..248`; non-test consumer: `ferrotorch-diffusion/src/safetensors_loader.rs:151..175` `load_unet` routes the HF UNet checkpoint through this state-dict layout |
| REQ-4 | SHIPPED | impl: `Module::forward` error at `ferrotorch-diffusion/src/resnet_block_time.rs:150..156`; non-test consumer: every UNet caller in `ferrotorch-diffusion/src/unet.rs` uses `forward_t` explicitly, so the error guard surfaces immediately for any future caller that forgets the temb |
