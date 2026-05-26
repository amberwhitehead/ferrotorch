# GpuVaeDecoder

<!--
tier: 3-component
status: draft
baseline-pytorch: /home/doll/pytorch (HEAD)
upstream-paths:
  - diffusers/src/diffusers/models/autoencoders/autoencoder_kl.py
  - diffusers/src/diffusers/models/autoencoders/vae.py
-->

## Summary

GPU twin of [`VaeDecoder`] for SD-1.5. VRAM-resident forward path
mirroring the CPU decoder op-for-op via `ferrotorch-gpu` kernels
(`gpu_conv2d_f32`, `gpu_group_norm_f32`, `gpu_matmul_f32`,
`gpu_bmm_f32`, `gpu_softmax`, `gpu_silu`,
`gpu_nearest_upsample2x_f32`, `gpu_scale`, `gpu_add`,
`gpu_broadcast_add`). One-shot weight upload at construction;
downloads only the final image `[B, 3, 512, 512]` at `decode`.
`decode` applies the `1 / scaling_factor` divide on the input
latent (matching `AutoencoderKL.decode(z).sample`).

## Requirements

- REQ-1: `GpuVaeDecoder::new(config, state, device)` validates the
  config, uploads every parameter tensor once into VRAM. Returns
  `(Self, DropReport)` for state-dict keys not consumed.
- REQ-2: `GpuVaeDecoder::from_module(cpu, device)` is the
  convenience constructor: extracts `cpu.state_dict()` and calls
  `Self::new`.
- REQ-3: `decode(latent)` runs the full forward:
  - validate `[B, latent_channels, H, W]`;
  - upload + `gpu_scale(1 / scaling_factor)`;
  - `post_quant_conv` (1×1);
  - `conv_in` (3×3, pad=1, latent → top_c);
  - mid-block `resnet0 → attn → resnet1`;
  - four up-blocks (each: stack of resnets + optional upsample);
  - `conv_norm_out → SiLU → conv_out`;
  - download → host `Tensor<f32>` `[B, 3, 512, 512]`.
- REQ-4: Shape validation rejects rank-≠-4 inputs or wrong
  channel count with `FerrotorchError::ShapeMismatch`.
- REQ-5: Mid-block attention is the VAE single-head flavour
  (GroupNorm + Linear q/k/v + Linear to_out.0; one head), distinct
  from the multi-head transformer used by the UNet.

## Acceptance Criteria

- [x] AC-1: `gpu_vae_matches_cpu` (lib test, `feature = "cuda"`)
  loads a tiny CPU `VaeDecoder`, builds the GPU twin via
  `from_module`, runs both on the same latent, asserts cosine
  similarity ≥ 0.999 on the decoded image.
- [x] AC-2: `decode` produces finite outputs on the tiny config.
- [x] AC-3: `decode(latent)` ≈ `cpu.decode_with_scaling(latent)`
  (the `1 / scaling_factor` divide is applied internally, mirroring
  the CPU helper).

## Architecture

- Per-component bundles at `gpu/vae.rs:50..127`:
  `GpuConv2d`, `GpuGroupNorm`, `GpuLinear` (kept in `[out, in]`
  PyTorch layout — VAE's Linear footprint is small and the
  transpose-on-host trick used by UNet/CLIP wasn't applied here),
  `GpuResnet` (VAE flavour, no temb), `GpuAttn` (single-head
  GroupNorm + Linear q/k/v + Linear to_out.0), `GpuUpsample`,
  `GpuUpDecoderBlock`, `GpuMidBlock`.
- `GpuVaeDecoder` at `gpu/vae.rs:150..159` holds
  `post_quant_conv`, `conv_in`, `mid_block`, four
  `up_blocks`, `conv_norm_out`, `conv_out`, frozen config, device
  handle.
- `new` at `gpu/vae.rs:182..` validates the config, pops every
  state-dict key by name (matching the CPU `VaeDecoder.state_dict()`
  layout: `post_quant_conv.*`, `decoder.conv_in.*`,
  `decoder.mid_block.*`, `decoder.up_blocks.{i}.*`,
  `decoder.conv_norm_out.*`, `decoder.conv_out.*`), enforces
  per-key length, uploads.
- `from_module` at `gpu/vae.rs:344..351`: extracts CPU
  `state_dict()` and delegates to `new`.
- `decode` at `gpu/vae.rs:367..423`:
  - shape check;
  - `inv = 1.0 / config.scaling_factor` (f64 → f32);
  - upload latent and `gpu_scale(latent, inv)`;
  - `post_quant_conv` then `conv_in`;
  - mid-block: `resnet0 → attn → resnet1`;
  - four up-blocks: stack of resnets (+ shortcut if in_c ≠ out_c),
    then optional upsample;
  - `conv_norm_out → gpu_silu → conv_out`;
  - download → `Tensor<f32>`.
- Helpers in the same file: `pop_tensor`, `pop_conv`,
  `pop_group_norm`, `pop_resnet`, `pop_attn`, `pop_mid_block`,
  `resnet_forward`, `attn_forward`, `upsample_forward`,
  `conv_forward`, `group_norm_forward`.

Non-test production consumers:

- `ferrotorch-diffusion/src/gpu/mod.rs:37` re-exports
  `GpuVaeDecoder`.
- `ferrotorch-diffusion/src/gpu/pipeline.rs:48,68,89` holds
  `vae: GpuVaeDecoder` as a pipeline field; `generate` at
  `gpu/pipeline.rs:230` calls `self.vae.decode(&latent)` as the
  final step.
- `ferrotorch-diffusion/examples/vae_decode_dump.rs:307,315`
  imports and constructs `GpuVaeDecoder` via `from_module` for the
  SD-1.5 VAE inference-dump binary.
- `ferrotorch-diffusion/examples/sd_pipeline_dump.rs:482,501`
  imports `GpuVaeDecoder` and calls `from_module` for the
  end-to-end SD-1.5 GPU dump.

## Parity contract

`parity_ops = []`. The decoder is a composition of
ferrotorch-gpu kernels; per-op parity is checked at the kernel
layer. End-to-end vs diffusers is checked by
`tests/conformance_pretrained_diffusion.rs` (cosine similarity
≥ 0.99 against the reference decoded image).

Critical invariants:

- **`scaling_factor = 0.18215`** (SD-1.5 published value), applied
  as `1 / scaling_factor` inside `decode` so callers pass the raw
  latent.
- **GroupNorm eps = 1e-6** throughout the VAE decoder (vs `1e-5` in
  the UNet).
- **Single-head mid-block attention** — the VAE attention is
  spatial-flatten + one Linear-projection of q/k/v + Linear-out,
  with channels playing the role of the head dim. The
  `GpuAttn::channels` field at `gpu/vae.rs:99..106` records the
  channel count for the per-call reshape.
- **Resnet shortcut**: present iff `in_channels != out_channels`.
  `pop_resnet` builds the shortcut conv lazily.

## Verification

Lib tests are in `gpu/vae.rs` (`#[cfg(test)] mod tests` under
`feature = "cuda"`). Cargo invocation:

```text
cargo test -p ferrotorch-diffusion --lib --features cuda gpu::vae
```

End-to-end: `tests/conformance_pretrained_diffusion.rs` runs the
GPU VAE decoder against the pinned SD-1.5 mirror.

No parity-sweep ops apply.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `GpuVaeDecoder::new` at `ferrotorch-diffusion/src/gpu/vae.rs:182..`; non-test consumer: `ferrotorch-diffusion/src/gpu/vae.rs:350` `from_module` calls `Self::new(cpu.config.clone(), state, device_clone)`; production binary `ferrotorch-diffusion/examples/vae_decode_dump.rs:315` constructs the decoder via `from_module` |
| REQ-2 | SHIPPED | impl: `from_module` at `ferrotorch-diffusion/src/gpu/vae.rs:344..351`; non-test consumer: `ferrotorch-diffusion/examples/vae_decode_dump.rs:315` `GpuVaeDecoder::from_module(decoder, &device)?`; `ferrotorch-diffusion/examples/sd_pipeline_dump.rs:501` `GpuVaeDecoder::from_module(vae, &device)?` |
| REQ-3 | SHIPPED | impl: `decode` at `ferrotorch-diffusion/src/gpu/vae.rs:367..423`; non-test consumer: `ferrotorch-diffusion/src/gpu/pipeline.rs:230` `self.vae.decode(&latent)?` is the canonical final decode call in the SD-1.5 GPU pipeline |
| REQ-4 | SHIPPED | impl: shape check at `ferrotorch-diffusion/src/gpu/vae.rs:369..376`; non-test consumer: the pipeline's `generate` exercises the shape contract on every dump call (`gpu/pipeline.rs:230`) |
| REQ-5 | SHIPPED | impl: `GpuAttn` struct at `ferrotorch-diffusion/src/gpu/vae.rs:99..106` with single-head `GroupNorm` + four `GpuLinear` (q/k/v/out), and `attn_forward` helper invoked from `decode` at `gpu/vae.rs:402`; non-test consumer: `decode` (and thus the SD-1.5 GPU pipeline) runs the mid-block attention exactly once per inference |
