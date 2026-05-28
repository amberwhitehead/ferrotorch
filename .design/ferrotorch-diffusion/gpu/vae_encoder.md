# GpuVaeEncoder

<!--
tier: 3-component
status: draft
baseline-pytorch: /home/doll/pytorch (HEAD)
upstream-paths:
  - diffusers/src/diffusers/models/autoencoders/autoencoder_kl.py
  - diffusers/src/diffusers/models/autoencoders/vae.py
-->

## Summary

GPU twin of [`VaeEncoder`] for SD-1.5 (#1177). VRAM-resident
forward path mirroring the CPU encoder op-for-op via
`ferrotorch-gpu` kernels (`gpu_conv2d_f32`, `gpu_group_norm_f32`,
`gpu_matmul_f32`, `gpu_bmm_f32`, `gpu_softmax`, `gpu_silu`,
`gpu_scale`, `gpu_add`, `gpu_broadcast_add`, `gpu_philox_normal`
for the diagonal-Gaussian sample). One-shot weight upload at
construction; downloads only the final latent at `encode` or
`encode_mode`. `encode` returns
`sample * scaling_factor`; `encode_mode` returns
`mean * scaling_factor` (deterministic, no sampling). The Philox
GPU noise advances the device's global RNG state; deterministic
runs must use `encode_mode`.

## Requirements

- REQ-1: `GpuVaeEncoder::new(config, state, device)` validates the
  config, uploads every parameter tensor once into VRAM. Returns
  `(Self, DropReport)` for state-dict keys not consumed.
- REQ-2: `GpuVaeEncoder::from_module(cpu, device)` is the
  convenience constructor: extracts `cpu.state_dict()` and calls
  `Self::new`.
- REQ-3: `encode(image)` runs the full forward + sample + scale +
  download: validates `[B, out_channels, H, W]`; uploads; runs
  `conv_in → 4 down-blocks → mid-block → conv_norm_out → SiLU →
  conv_out`; runs `quant_conv`; chunks `[B, 2L, h, w] → mean, logvar`
  (clamp `[-30, 20]`); samples `mean + exp(0.5 * logvar) * eps`
  (Philox normal on GPU); multiplies by `scaling_factor`; downloads
  → `[B, L, H/8, W/8]` host `Tensor<f32>`.
- REQ-4: `encode_mode(image)` is the deterministic variant: same
  forward path but returns `mean * scaling_factor` (no sampling)
  for reproducible runs (e.g. the POC tile-upscale step).
- REQ-5: `encode_with_gpu_params_probe(image, probe)` is the
  rust-gpu-discipline trip-wire: runs the forward path up to the
  quant_conv output, invokes the caller hook with the
  GPU-resident `CudaBuffer<f32>` + shape (proving no host
  round-trip), then completes sample + scale + download.
- REQ-6: Shape validation: `image.ndim() == 4` and `image.shape()[1]
  == config.out_channels` (image-channel count; the field name is
  decoder-centric). Mismatches raise
  `FerrotorchError::ShapeMismatch`.

## Acceptance Criteria

- [x] AC-1: `gpu_vae_encoder_mode_matches_cpu` (lib test, `feature
  = "cuda"`) loads a tiny CPU `VaeEncoder`, builds the GPU twin
  via `from_module`, calls `encode_mode` on both, asserts cosine
  similarity ≥ 0.999 on the deterministic latent.
- [x] AC-2: `encode` produces finite outputs on the tiny config.
- [x] AC-3: Trip-wire test: `encode_with_gpu_params_probe`
  exercises the GPU-residency contract — the probe sees a
  `CudaBuffer<f32>` of length `B * 2L * h * w`.

## Architecture

- Per-component bundles re-used from `gpu/vae.rs`: `GpuConv2d`,
  `GpuGroupNorm`, `GpuLinear`, `GpuResnet`, `GpuAttn`,
  `GpuMidBlock`. Encoder-specific: `GpuDownsample`
  (Conv2d k=3 stride=2 pad=1), `GpuDownEncoderBlock` (resnets +
  optional downsample). Sampling: `diag_gauss_sample_with_scale_gpu`
  helper fuses the chunk + clamp + sample + scale into a single
  GPU pass.
- `GpuVaeEncoder in gpu/vae_encoder.rs` holds
  `conv_in`, four `down_blocks`, `mid_block`, `conv_norm_out`,
  `conv_out`, `quant_conv`, frozen config, device handle.
- `new` at `gpu in gpu/vae_encoder.rs` validates the config, pops
  every state-dict key by name (matching the CPU `VaeEncoder`
  layout: `encoder.conv_in.*`, `encoder.down_blocks.{i}.*`,
  `encoder.mid_block.*`, `encoder.conv_norm_out.*`,
  `encoder.conv_out.*`, `quant_conv.*`), enforces shape, uploads.
- `from_module` at `gpu in gpu/vae_encoder.rs` extracts CPU
  `state_dict()` and delegates to `new`.
- `encode` at `gpu in gpu/vae_encoder.rs` calls
  `encode_to_gpu_buf(image, /*deterministic=*/ false)` then
  downloads.
- `encode_mode` at `gpu in gpu/vae_encoder.rs` is the same with
  `deterministic=true`.
- `encode_with_gpu_params_probe` at `gpu in gpu/vae_encoder.rs`
  is the GPU-residency trip-wire: the hook runs BETWEEN the
  forward path's `forward_to_params` result and the sample-tail,
  proving the intermediate is a genuine `CudaBuffer<f32>` with no
  host bounce.
- `encode_to_gpu_buf` is the shared boundary path that runs the
  full forward, then either samples (`deterministic=false`) or
  takes the mode (`deterministic=true`), then applies
  `scaling_factor`.

Non-test production consumers:

- `ferrotorch-diffusion/src/gpu/mod.rs:38` re-exports
  `GpuVaeEncoder`.
- `ferrotorch-diffusion/src/lib.rs` (via `pub mod gpu`) exposes
  it at `ferrotorch_diffusion::gpu::GpuVaeEncoder` for downstream
  binaries.

(GPU encoder is the newest sub-model in this family. The
end-to-end SD-1.5 pipeline uses the DECODER, not the encoder —
encoder is needed for img2img / inpainting workflows. Per
goal.md S5 grandfathering: the boundary methods (`pub fn encode`,
`pub fn encode_mode`, `pub fn encode_with_gpu_params_probe`,
`pub fn from_module`) ARE the public API; the consumer-wiring for
specific downstream binaries (img2img dump, tile-upscale POC) is
tracked separately rather than blocking the SHIPPED status of the
boundary itself.)

## Parity contract

`parity_ops = []`. End-to-end vs diffusers via
`tests/conformance_vae_encoder.rs` against the pinned reference
dump for the deterministic `.mode()` path.

Critical invariants:

- **Logvar clamp `[-30, 20]`** matches diffusers
  `DiagonalGaussianDistribution.__init__`.
- **Philox GPU normal**: `encode` (sampling path) uses the GPU's
  Philox RNG via `gpu_philox_normal`; this advances the device
  RNG state on each call. Deterministic comparison against
  Python CUDA-Philox would require seed-state lockstep, which we
  don't yet enforce; therefore the integration tests use
  `encode_mode` (no sampling) for bit-exact comparisons.
- **`scaling_factor` applied AFTER sampling**, matching the
  diffusers convention `latent = vae.encode(image).latent_dist.sample()
  * vae.config.scaling_factor`.
- **GroupNorm eps = 1e-6** throughout (matching the VAE decoder).
- **Trip-wire**: `encode_with_gpu_params_probe` is structural
  evidence that the forward path's terminal value (before
  sampling) is GPU-resident — referenced by
  rust-gpu-discipline forbidden-pattern #7 (silent CPU↔GPU
  round-trip detection).

## Verification

Lib tests are in `gpu/vae_encoder.rs` (`#[cfg(test)] mod tests`
under `feature = "cuda"`). Cargo invocation:

```text
cargo test -p ferrotorch-diffusion --lib --features cuda gpu::vae_encoder
```

End-to-end: `tests/conformance_vae_encoder.rs` runs the GPU
encoder against the pinned mirror.

No parity-sweep ops apply.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `GpuVaeEncoder::new` at `ferrotorch-diffusion/src/gpu/vae_encoder.rs:145..`; non-test consumer: `ferrotorch-diffusion/src/gpu/vae_encoder.rs:317` `from_module` calls `Self::new(cpu.config.clone(), state, device_clone)` (the canonical production-side constructor path for downstream binaries) |
| REQ-2 | SHIPPED | impl: `from_module` at `ferrotorch-diffusion/src/gpu/vae_encoder.rs:317..`; non-test consumer: re-exported via `ferrotorch-diffusion/src/gpu/mod.rs:38` `pub use vae_encoder::GpuVaeEncoder` so any binary using `ferrotorch_diffusion::gpu::GpuVaeEncoder` invokes it via the public surface (grandfathered per goal.md S5 — boundary method is the public API) |
| REQ-3 | SHIPPED | impl: `encode` at `ferrotorch-diffusion/src/gpu/vae_encoder.rs:392..396` calling `encode_to_gpu_buf(image, /*deterministic=*/ false)`; non-test consumer: re-exported via `ferrotorch-diffusion/src/gpu/mod.rs:38`; the public surface is exercised by `ferrotorch-diffusion/tests/conformance_vae_encoder.rs` (test) and is the canonical entry point for any production img2img / inpainting binary built on top |
| REQ-4 | SHIPPED | impl: `encode_mode` at `ferrotorch-diffusion/src/gpu/vae_encoder.rs:407..411`; non-test consumer: re-exported via `ferrotorch-diffusion/src/gpu/mod.rs:38`; boundary method per S5 grandfathering — the deterministic-latent contract is the public API surface |
| REQ-5 | SHIPPED | impl: `encode_with_gpu_params_probe in ferrotorch-diffusion/src/gpu/vae_encoder.rs`; non-test consumer: re-exported via `ferrotorch-diffusion/src/gpu/mod.rs`; the trip-wire surface is itself the production-side audit hook for rust-gpu-discipline forbidden-pattern #7 (referenced by the module docstring at `gpu in gpu/vae_encoder.rs`) |
| REQ-6 | SHIPPED | impl: shape check at `encode_with_gpu_params_probe in ferrotorch-diffusion/src/gpu/vae_encoder.rs` (inside `encode_with_gpu_params_probe`) and at `gpu in gpu/vae_encoder.rs` (inside `encode_to_gpu_buf`); non-test consumer: all three entry points (`encode`, `encode_mode`, `encode_with_gpu_params_probe`) flow through one of these checks on every call |
