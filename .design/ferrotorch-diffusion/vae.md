# VaeDecoder (CPU)

<!--
tier: 3-component
status: draft
baseline-pytorch: /home/doll/pytorch (HEAD)
upstream-paths:
  - diffusers/src/diffusers/models/autoencoders/autoencoder_kl.py
  - diffusers/src/diffusers/models/autoencoders/vae.py
-->

## Summary

CPU SD-1.5 VAE decoder. Mirrors
`diffusers.AutoencoderKL.decode(z).sample` for
`runwayml/stable-diffusion-v1-5`. Composes `post_quant_conv` (1×1)
with a `Decoder` stack (`conv_in` + `UNetMidBlock2D` + 4×
`UpDecoderBlock2D` + `conv_norm_out` + SiLU + `conv_out`). The
`decode_with_scaling` helper applies the canonical
`z / scaling_factor` divide first.

## Requirements

- REQ-1: `Decoder<T>` carries a typed `Conv2d` pipeline matching the
  diffusers `Decoder` layout: `conv_in (latent_channels →
  block_out_channels[-1])` → `mid_block (UNetMidBlock2D)` → reversed
  `up_blocks` (each `UpDecoderBlock2D` with N resnets + optional
  `Upsample2D`) → `conv_norm_out` (`GroupNorm`) → `SiLU` →
  `conv_out (block_out_channels[0] → out_channels)`.
- REQ-2: `VaeDecoder<T>` wraps `Decoder<T>` with a 1×1
  `post_quant_conv` over `latent_channels`, mirroring
  `AutoencoderKL.post_quant_conv`.
- REQ-3: `Module<T>::forward` accepts a `[B, latent_channels, H, W]`
  tensor (post-scaling), rejects rank or channel mismatches with
  `FerrotorchError::ShapeMismatch`, and returns
  `[B, out_channels, H * 2^(N-1), W * 2^(N-1)]`.
- REQ-4: `decode_with_scaling(latent)` divides by
  `config.scaling_factor` before invoking `forward`, matching
  `AutoencoderKL.decode(z).sample` (which receives `z /
  scaling_factor` from the pipeline-side scaling).
- REQ-5: `Module<T>::load_state_dict` accepts the canonical diffusers
  layout (`post_quant_conv.*` / `decoder.conv_in.*` /
  `decoder.mid_block.*` / `decoder.up_blocks.{i}.*` /
  `decoder.conv_norm_out.*` / `decoder.conv_out.*`); strict mode
  rejects unknown prefixes.

## Acceptance Criteria

- [x] AC-1: `decoder_forward_shape` test exercises the full forward
  on a 4-block tiny config and asserts the output shape +
  finiteness (`vae in vae.rs`).
- [x] AC-2: `vae_decoder_named_parameters_include_post_quant_conv`
  enumerates the canonical key list (`vae in vae.rs`).
- [x] AC-3: `vae_decoder_decode_with_scaling_matches_manual_div`
  verifies `decode_with_scaling(z) == forward(z / scaling_factor)`
  to 1e-4 (`vae in vae.rs`).
- [x] AC-4: `round_trip_state_dict` proves the named-parameters and
  `load_state_dict` agree (`vae in vae.rs`).

## Architecture

- `Decoder<T>` at `Decoder in vae.rs` holds the conv pipeline. `new` at
  `vae in vae.rs` builds: `conv_in` (k=3, pad=1), `UNetMidBlock2D`
  at top channels with `resnet_eps = 1e-6`, then reversed up-blocks
  (each `UpDecoderBlock2D` with `cfg.resnets_per_up_block()` resnets
  and an upsample on all but the last). `conv_norm_out` is a
  `GroupNorm(groups, block_out_channels[0], eps=1e-6, affine=true)`.
- `VaeDecoder<T>` at `VaeDecoder in vae.rs` wraps a `Decoder<T>` with a
  1×1 `post_quant_conv` over `latent_channels`. `new` at
  `vae in vae.rs` builds both pieces.
- `decode_with_scaling` at `vae in vae.rs` computes
  `inv = (1.0 / scaling_factor)` (via `T::from(f64) → Float`) and
  multiplies the latent before calling `forward`. The cast uses the
  shared `ferrotorch_core::scalar::<T>` helper.
- `Module<T>::forward` at `vae in vae.rs` rejects rank-≠-4 or
  wrong-channel inputs, calls `post_quant_conv` then `decoder`.
- `load_state_dict` at `vae in vae.rs` splits keys into
  `post_quant_conv.*` and `decoder.*` prefixes; strict mode rejects
  anything else.

Non-test production consumers:

- `ferrotorch-diffusion/src/safetensors_loader.rs:318..333`
  `load_vae_decoder` calls `VaeDecoder::<T>::new(cfg)` then
  `load_hf_state_dict`. This is the standard loader path.
- `ferrotorch-diffusion/src/pipeline.rs:75` `StableDiffusionPipeline`
  holds a `VaeDecoder<T>` field; `pipeline in pipeline.rs` calls
  `vae.decode_with_scaling(...)` in `generate`.
- `ferrotorch-diffusion/src/gpu/vae.rs:344..351`
  `GpuVaeDecoder::from_module(cpu: &VaeDecoder<f32>, ...)` consumes
  the CPU module's `state_dict()` to build the GPU twin.
- `ferrotorch-diffusion/examples/vae_decode_dump.rs:218` loads via
  `load_vae_decoder` and runs `decode_with_scaling` in the
  inference-dump binary.

## Parity contract

`parity_ops = []`. The contract is structural — every Conv2d,
GroupNorm, ResnetBlock2D, and Upsample2D underneath is an existing
ferrotorch op whose own parity is exercised at the op layer
(`tools/parity-sweep/parity_audit.json`). The VAE decoder's
job is correct composition. End-to-end numerical parity is checked
by `conformance_vae_decode.rs` against the pinned `vae_decode/<seed>/`
reference dump.

The `1 / scaling_factor` divide in `decode_with_scaling` mirrors
`vae.config.scaling_factor = 0.18215` (the SD-1.5 published value)
and the diffusers convention
`image = self.vae.decode(latents / self.vae.config.scaling_factor).sample`.

## Verification

- `decoder_forward_shape` (`vae in vae.rs`) — `[1, 4, 1, 1]` →
  `[1, 3, 8, 8]` on the tiny 4-block config.
- `vae_decoder_forward_shape` (`vae in vae.rs`) — same with the
  full `VaeDecoder` (incl. `post_quant_conv`).
- `vae_decoder_named_parameters_include_post_quant_conv`
  (`vae in vae.rs`) — checks canonical keys
  (`post_quant_conv.weight`, `decoder.mid_block.attentions.0.to_q.weight`,
  …) are exposed.
- `vae_decoder_decode_with_scaling_matches_manual_div`
  (`vae in vae.rs`) — proves the helper agrees with manual scaling.
- `round_trip_state_dict` (`vae in vae.rs`) — state_dict / load
  round-trip yields identical forward outputs to 1e-5.

End-to-end parity:
`ferrotorch-diffusion/tests/conformance_vae_decode.rs` runs the
loaded VAE against the pinned mirror.

No parity-sweep ops apply (composition module).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `Decoder<T>` at `Decoder in ferrotorch-diffusion/src/vae.rs` and `Decoder::new` at `new in ferrotorch-diffusion/src/vae.rs`; non-test consumer: `new in ferrotorch-diffusion/src/vae.rs` `VaeDecoder::new` builds it; itself consumed by `new in ferrotorch-diffusion/src/safetensors_loader.rs` `load_vae_decoder` |
| REQ-2 | SHIPPED | impl: `VaeDecoder<T>` at `VaeDecoder in ferrotorch-diffusion/src/vae.rs` and `VaeDecoder::new` at `new in ferrotorch-diffusion/src/vae.rs`; non-test consumer: `new in ferrotorch-diffusion/src/safetensors_loader.rs` `load_vae_decoder` instantiates and loads it; `new in ferrotorch-diffusion/src/pipeline.rs` carries it as a pipeline field |
| REQ-3 | SHIPPED | impl: `Module<T>::forward` at `forward in ferrotorch-diffusion/src/vae.rs` (shape check at `vae in vae.rs`); non-test consumer: `ferrotorch-diffusion/src/pipeline.rs` `vae.decode_with_scaling(...)` (which calls `forward` internally) |
| REQ-4 | SHIPPED | impl: `decode_with_scaling` at `ferrotorch-diffusion/src/vae.rs:297..308`; non-test consumer: `ferrotorch-diffusion/src/pipeline.rs:227` and `ferrotorch-diffusion/examples/vae_decode_dump.rs` invoke it for the SD-1.5 decoding step |
| REQ-5 | SHIPPED | impl: `Module<T>::load_state_dict` at `ferrotorch-diffusion/src/vae.rs:366..389`; non-test consumer: `ferrotorch-diffusion/src/safetensors_loader.rs:89` `VaeDecoder::load_hf_state_dict` calls `self.load_state_dict(&remapped, strict)` after stripping the `vae.` prefix |
