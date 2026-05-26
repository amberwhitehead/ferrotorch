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
  branch + shape (`blocks.rs:1134..1163`).
- [x] AC-2: `resnet_named_parameters_layout` lists the diffusers
  state-dict keys (`blocks.rs:1166..1183`).
- [x] AC-3: `attn_shape_and_residual` confirms shape preservation
  and finiteness (`blocks.rs:1186..1200`).
- [x] AC-4: `attn_named_parameters_layout` lists the diffusers
  state-dict keys (`blocks.rs:1203..1220`).
- [x] AC-5: `upsample2d_doubles_spatial`,
  `up_decoder_block_shape_with_upsample`,
  `up_decoder_block_shape_no_upsample`,
  `down_encoder_block_shape_with_downsample`,
  `down_encoder_block_shape_no_downsample` pass
  (`blocks.rs:1223..1287`).
- [x] AC-6: `mid_block_shape` & `mid_block_named_parameters_layout`
  pass (`blocks.rs:1305..1330`).

## Architecture

Each block is a `Module<T: Float>` implementor with public fields
and a `new` constructor.

- `ResnetBlock2D` (`blocks.rs:35..96`): five fields
  (`norm1`, `conv1`, `norm2`, `conv2`, optional `conv_shortcut`).
  Forward pass (`blocks.rs:98..127`) follows diffusers: norm + SiLU
  + conv, twice, then residual add.
- `AttnBlock2D` (`blocks.rs:242..282`): single-head spatial
  self-attention configured as
  `heads = in_channels / attention_head_dim = 1` for the SD VAE.
  Forward (`blocks.rs:284..351`) reshapes `[B,C,H,W]` to `[B,HW,C]`,
  group-norms via two transposes, runs q/k/v projections, computes
  `softmax(q @ k^T / sqrt(C)) @ v` via `bmm`, projects back, then
  reshapes to `[B,C,H,W]` and adds the residual.
- `Upsample2D` (`blocks.rs:446..467`): one Conv2d field. Forward
  (`blocks.rs:469..485`) calls `ferrotorch_nn::Upsample` with
  `InterpolateMode::Nearest` at 2x then applies the conv.
- `Downsample2D` (`blocks.rs:536..558`): one Conv2d field; the
  forward is just `conv(input)` because SD-1.5 uses
  `use_conv=True, padding=1` (no separate pre-pad).
- `UpDecoderBlock2D` (`blocks.rs:626..676`): chains resnets +
  optional upsample. Forward at `blocks.rs:678..688`.
- `DownEncoderBlock2D` (`blocks.rs:806..861`): mirror image of
  `UpDecoderBlock2D` with downsample instead. Forward at
  `blocks.rs:863..873`.
- `UNetMidBlock2D` (`blocks.rs:988..1018`): two resnets + one
  attention, as
  `resnets[0] → attentions[0] → resnets[1]`. Forward at
  `blocks.rs:1020..1032`.

State-dict load/store helpers (`load_state_dict`,
`named_parameters`) at every block faithfully reproduce the
diffusers HF safetensors prefix layout, including the
`to_out.0.*` (not `to_out.weight`) convention which exists because
diffusers wraps the output projection in a `Sequential` whose
second element is a Dropout.

Non-test production consumers:

- `ferrotorch-diffusion/src/vae.rs:26` imports `UNetMidBlock2D` and
  `UpDecoderBlock2D`; the VAE decoder is built from these.
- `ferrotorch-diffusion/src/vae_encoder.rs:32` imports
  `DownEncoderBlock2D` and `UNetMidBlock2D`; the VAE encoder is
  built from these.
- `ferrotorch-diffusion/src/unet.rs:48` imports `Downsample2D` and
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

Twelve lib tests in `blocks.rs:1129..1332` covering shape
preservation, residual presence, state-dict key layout for each
block, and finiteness of attention output.

No parity-sweep ops apply.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `ResnetBlock2D` at `ferrotorch-diffusion/src/blocks.rs:35..215`; non-test consumer: `ferrotorch-diffusion/src/vae.rs` and `vae_encoder.rs` consume `ResnetBlock2D` transitively through `UpDecoderBlock2D`/`DownEncoderBlock2D`/`UNetMidBlock2D` |
| REQ-2 | SHIPPED | impl: `AttnBlock2D` at `ferrotorch-diffusion/src/blocks.rs:242..434`; non-test consumer: `UNetMidBlock2D::new` at `ferrotorch-diffusion/src/blocks.rs:1010` invokes `AttnBlock2D::<T>::new`, and `vae.rs:83` constructs the mid block in production |
| REQ-3 | SHIPPED | impl: `Upsample2D` at `ferrotorch-diffusion/src/blocks.rs:446..525`; non-test consumer: `ferrotorch-diffusion/src/unet.rs:48` imports `Upsample2D` for the UNet's `UpBlock2D`/`CrossAttnUpBlock2D` |
| REQ-4 | SHIPPED | impl: `Downsample2D` at `ferrotorch-diffusion/src/blocks.rs:536..611`; non-test consumer: `ferrotorch-diffusion/src/unet.rs:48` imports `Downsample2D` for the UNet's `DownBlock2D`/`CrossAttnDownBlock2D` |
| REQ-5 | SHIPPED | impl: `UpDecoderBlock2D` at `ferrotorch-diffusion/src/blocks.rs:626..786`; non-test consumer: `ferrotorch-diffusion/src/vae.rs:92` calls `UpDecoderBlock2D::<T>::new(...)` for every decoder up-block |
| REQ-6 | SHIPPED | impl: `DownEncoderBlock2D` at `ferrotorch-diffusion/src/blocks.rs:806..972`; non-test consumer: `ferrotorch-diffusion/src/vae_encoder.rs:123` calls `DownEncoderBlock2D::<T>::new(...)` for every encoder down-block |
| REQ-7 | SHIPPED | impl: `UNetMidBlock2D` at `ferrotorch-diffusion/src/blocks.rs:988..1127`; non-test consumer: `ferrotorch-diffusion/src/vae.rs:83` and `vae_encoder.rs:130` both call `UNetMidBlock2D::<T>::new(top_channels, groups, resnet_eps)?` |
