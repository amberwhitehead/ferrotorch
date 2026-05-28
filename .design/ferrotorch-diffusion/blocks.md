# UNet/VAE building blocks

<!--
tier: 3-component
status: draft
baseline-pytorch: /home/doll/pytorch (HEAD)
upstream-paths:
  - diffusers/src/diffusers/models/resnet.py
  - diffusers/src/diffusers/models/upsampling.py
  - diffusers/src/diffusers/models/downsampling.py
  - diffusers/src/diffusers/models/attention_processor.py
  - diffusers/src/diffusers/models/unets/unet_2d_blocks.py
-->

## Summary

The leaf and composite blocks of the Stable-Diffusion VAE decoder /
encoder. Mirrors diffusers `models/resnet.py`,
`models/upsampling.py`, `models/downsampling.py`,
`models/attention_processor.py`, and `models/unets/unet_2d_blocks.py`
1:1 in parameter naming and forward semantics so the upstream
state dict (`runwayml/stable-diffusion-v1-5/vae/diffusion_pytorch_model.safetensors`)
loads byte-for-byte.

## Requirements

- REQ-1: `ResnetBlock2D` implements the VAE-flavour residual block
  (no time embedding): `h = conv1(silu(norm1(x)))`; `h =
  conv2(silu(norm2(h)))`; `out = h + (x if in==out else
  conv_shortcut(x))`. State-dict layout: `norm1.*`, `conv1.*`,
  `norm2.*`, `conv2.*`, `conv_shortcut.*` (optional).
- REQ-2: `AttnBlock2D` implements the VAE mid-block single-head
  spatial self-attention with residual + GroupNorm. State-dict
  layout matches diffusers: `group_norm.*`, `to_q.*`, `to_k.*`,
  `to_v.*`, `to_out.0.*`.
- REQ-3: `Upsample2D` performs nearest-neighbor 2x upsample
  followed by `Conv2d(C, C, k=3, pad=1, bias)`. State-dict key:
  `conv.*`.
- REQ-4: `Downsample2D` is a `Conv2d(C, C, k=3, stride=2, pad=1,
  bias)`. State-dict key: `conv.*`.
- REQ-5: `UpDecoderBlock2D` chains `num_resnets` `ResnetBlock2D`
  layers followed by an optional `Upsample2D`. State-dict layout:
  `resnets.{i}.*`, `upsamplers.0.*`.
- REQ-6: `DownEncoderBlock2D` chains `num_resnets` `ResnetBlock2D`
  layers followed by an optional `Downsample2D`. State-dict layout:
  `resnets.{i}.*`, `downsamplers.0.*`.
- REQ-7: `UNetMidBlock2D` (VAE flavour) chains
  `resnets[0] → attentions[0] → resnets[1]` (two resnets and one
  attention). State-dict listing order matches the HF safetensors:
  `attentions.*` first, then `resnets.*`.

## Acceptance Criteria

- [x] AC-1: `resnet_same_channels_no_shortcut` &
  `resnet_different_channels_has_shortcut` validate the residual
  branch + shape (`blocks.rs`).
- [x] AC-2: `resnet_named_parameters_layout` lists the diffusers
  state-dict keys (`blocks.rs`).
- [x] AC-3: `attn_shape_and_residual` confirms shape preservation
  and finiteness (`blocks.rs`).
- [x] AC-4: `attn_named_parameters_layout` lists the diffusers
  state-dict keys (`blocks.rs`).
- [x] AC-5: `upsample2d_doubles_spatial`,
  `up_decoder_block_shape_with_upsample`,
  `up_decoder_block_shape_no_upsample`,
  `down_encoder_block_shape_with_downsample`,
  `down_encoder_block_shape_no_downsample` pass
  (`blocks.rs`).
- [x] AC-6: `mid_block_shape` & `mid_block_named_parameters_layout`
  pass (`blocks.rs`).

## Architecture

Each block is a `Module<T: Float>` implementor with public fields
and a `new` constructor.

- `ResnetBlock2D` (`ResnetBlock2D in blocks.rs`): five fields
  (`norm1`, `conv1`, `norm2`, `conv2`, optional `conv_shortcut`).
  Forward pass (`blocks.rs`) follows diffusers: norm + SiLU
  + conv, twice, then residual add.
- `AttnBlock2D` (`AttnBlock2D in blocks.rs`): single-head spatial
  self-attention configured as
  `heads = in_channels / attention_head_dim = 1` for the SD VAE.
  Forward (`blocks.rs`) reshapes `[B,C,H,W]` to `[B,HW,C]`,
  group-norms via two transposes, runs q/k/v projections, computes
  `softmax(q @ k^T / sqrt(C)) @ v` via `bmm`, projects back, then
  reshapes to `[B,C,H,W]` and adds the residual.
- `Upsample2D` (`Upsample2D in blocks.rs`): one Conv2d field. Forward
  (`Upsample in blocks.rs`) calls `ferrotorch_nn::Upsample` with
  `InterpolateMode::Nearest` at 2x then applies the conv.
- `Downsample2D` (`Downsample2D in blocks.rs`): one Conv2d field; the
  forward is just `conv(input)` because SD-1.5 uses
  `use_conv=True, padding=1` (no separate pre-pad).
- `UpDecoderBlock2D` (`UpDecoderBlock2D in blocks.rs`): chains resnets +
  optional upsample. Forward at `blocks.rs`.
- `DownEncoderBlock2D` (`DownEncoderBlock2D in blocks.rs`): mirror image of
  `UpDecoderBlock2D` with downsample instead. Forward at
  `blocks.rs`.
- `UNetMidBlock2D` (`UNetMidBlock2D in blocks.rs`): two resnets + one
  attention, as
  `resnets[0] → attentions[0] → resnets[1]`. Forward at
  `blocks.rs`.

State-dict load/store helpers (`load_state_dict`,
`named_parameters`) at every block faithfully reproduce the
diffusers HF safetensors prefix layout, including the
`to_out.0.*` (not `to_out.weight`) convention which exists because
diffusers wraps the output projection in a `Sequential` whose
second element is a Dropout.

Non-test production consumers:

- `ferrotorch-diffusion/src/vae.rs` imports `UNetMidBlock2D` and
  `UpDecoderBlock2D`; the VAE decoder is built from these.
- `ferrotorch-diffusion/src/vae_encoder.rs` imports
  `DownEncoderBlock2D` and `UNetMidBlock2D`; the VAE encoder is
  built from these.
- `ferrotorch-diffusion/src/unet.rs` imports `Downsample2D` and
  `Upsample2D`; the UNet's `DownBlock2D`/`UpBlock2D` use them.
- The SD safetensors loader at `safetensors_loader.rs` exercises the
  state-dict layout end-to-end against the HF VAE checkpoint.

## Parity contract

`parity_ops = []`. Each block's contract is byte-equivalence with
the diffusers reference forward pass when loaded with the same HF
state dict. Edge cases:

- `input.shape()[1] != channels` → `ShapeMismatch`.
- `norm_num_groups` not dividing `channels` → propagated
  `InvalidArgument` from `GroupNorm::new`.
- Single-head attention scale = `1/sqrt(channels)` (not
  `1/sqrt(head_dim)` — single-head means they're the same).

## Verification

Twelve lib tests in `blocks.rs` covering shape
preservation, residual presence, state-dict key layout for each
block, and finiteness of attention output.

No parity-sweep ops apply.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `ResnetBlock2D in ferrotorch-diffusion/src/blocks.rs`; non-test consumer: `ferrotorch-diffusion/src/vae.rs` and `vae_encoder.rs` consume `ResnetBlock2D` transitively through `UpDecoderBlock2D`/`DownEncoderBlock2D`/`UNetMidBlock2D` |
| REQ-2 | SHIPPED | impl: `AttnBlock2D in ferrotorch-diffusion/src/blocks.rs`; non-test consumer: `UNetMidBlock2D::new` at `new in ferrotorch-diffusion/src/blocks.rs` invokes `AttnBlock2D::<T>::new`, and `vae in vae.rs` constructs the mid block in production |
| REQ-3 | SHIPPED | impl: `Upsample2D in ferrotorch-diffusion/src/blocks.rs`; non-test consumer: `UpBlock2D in ferrotorch-diffusion/src/unet.rs` imports `Upsample2D` for the UNet's `UpBlock2D`/`CrossAttnUpBlock2D` |
| REQ-4 | SHIPPED | impl: `Downsample2D in ferrotorch-diffusion/src/blocks.rs`; non-test consumer: `DownBlock2D in ferrotorch-diffusion/src/unet.rs` imports `Downsample2D` for the UNet's `DownBlock2D`/`CrossAttnDownBlock2D` |
| REQ-5 | SHIPPED | impl: `UpDecoderBlock2D in ferrotorch-diffusion/src/blocks.rs`; non-test consumer: `new in ferrotorch-diffusion/src/vae.rs` calls `UpDecoderBlock2D::<T>::new(...)` for every decoder up-block |
| REQ-6 | SHIPPED | impl: `DownEncoderBlock2D in ferrotorch-diffusion/src/blocks.rs`; non-test consumer: `new in ferrotorch-diffusion/src/vae_encoder.rs` calls `DownEncoderBlock2D::<T>::new(...)` for every encoder down-block |
| REQ-7 | SHIPPED | impl: `UNetMidBlock2D in ferrotorch-diffusion/src/blocks.rs`; non-test consumer: `new in ferrotorch-diffusion/src/vae.rs` and `vae_encoder in vae_encoder.rs` both call `UNetMidBlock2D::<T>::new(top_channels, groups, resnet_eps)?` |
