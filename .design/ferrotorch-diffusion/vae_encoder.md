# VaeEncoder (CPU)

<!--
tier: 3-component
status: draft
baseline-pytorch: /home/doll/pytorch (HEAD)
upstream-paths:
  - diffusers/src/diffusers/models/autoencoders/autoencoder_kl.py
  - diffusers/src/diffusers/models/autoencoders/vae.py
-->

## Summary

CPU SD-1.5 VAE encoder + diagonal-Gaussian sampler. Mirrors
`diffusers.AutoencoderKL.encode(image).latent_dist` for
`runwayml/stable-diffusion-v1-5`. Composes an `Encoder` stack
(`conv_in` + N down-blocks + `UNetMidBlock2D` + `conv_norm_out` +
SiLU + `conv_out`) with a 1×1 `quant_conv` and a
`DiagonalGaussianDistribution` head that splits the `[B, 2L, h, w]`
parameters into clamped (`logvar ∈ [-30, 20]`) mean / logvar.

## Requirements

- REQ-1: `Encoder<T>` mirrors the diffusers encoder layout:
  `conv_in (out_channels → block_out_channels[0])` → N down-blocks
  (`DownEncoderBlock2D` with `layers_per_block` resnets +
  optional `Downsample2D`; only the last block omits the downsample)
  → `UNetMidBlock2D` at the deepest channel count →
  `conv_norm_out` (`GroupNorm`) → SiLU →
  `conv_out (block_out_channels[-1] → 2 * latent_channels)`.
- REQ-2: `VaeEncoder<T>` wraps `Encoder<T>` with a 1×1 `quant_conv`
  over `2 * latent_channels`, mirroring `AutoencoderKL.quant_conv`.
- REQ-3: `VaeEncoder::encode(image)` returns a
  `DiagonalGaussianDistribution<T>` (split along the channel axis
  into `mean` and `logvar`); `logvar` is clamped to `[-30, 20]`
  matching `DiagonalGaussianDistribution.__init__`.
- REQ-4: `DiagonalGaussianDistribution::sample_with_seed(seed)`
  computes `mean + exp(0.5 * logvar) * eps` where `eps` is a
  Box-Muller-from-xorshift64 `N(0, 1)` sample. Deterministic per
  seed on host; NOT bitwise-equivalent to CUDA-Philox.
  `mode()` returns `&mean`.
- REQ-5: `VaeEncoder::encode_with_scaling(image, seed)` computes
  `dist.sample_with_seed(seed) * scaling_factor`, mirroring the
  diffusers `vae.encode(image).latent_dist.sample() *
  vae.config.scaling_factor` idiom.
- REQ-6: `Module<T>::forward` returns the raw `[B, 2L, h, w]`
  parameters tensor (post-`quant_conv`); rank/channel mismatches
  raise `FerrotorchError::ShapeMismatch`.
- REQ-7: `Module<T>::load_state_dict` accepts the canonical
  diffusers layout (`encoder.*` / `quant_conv.*`); strict mode
  rejects unknown prefixes.

## Acceptance Criteria

- [x] AC-1: `encoder_forward_shape` runs `[1, 3, 8, 8]` →
  `[1, 2L, 1, 1]` on the tiny 4-block config (`vae_encoder.rs:609..625`).
- [x] AC-2: `vae_encoder_forward_shape` exercises the full
  `VaeEncoder.forward` (`vae_encoder.rs:628..639`).
- [x] AC-3: `vae_encoder_named_parameters_include_quant_conv`
  enumerates canonical keys (`vae_encoder.rs:642..657`).
- [x] AC-4: `diag_gauss_split_and_mode_shapes` proves the
  `[B, 2L, h, w] → mean[B, L, h, w], logvar[B, L, h, w]` split
  (`vae_encoder.rs:660..683`).
- [x] AC-5: `diag_gauss_logvar_is_clamped` pins `logvar ≤ 20.0`
  (`vae_encoder.rs:686..703`).
- [x] AC-6: `diag_gauss_sample_with_seed_is_deterministic` pins
  determinism per seed and difference across seeds
  (`vae_encoder.rs:706..738`).
- [x] AC-7: `encode_with_scaling_applies_scaling_factor` proves
  the `* scaling_factor` multiplier (`vae_encoder.rs:761..787`).
- [x] AC-8: `vae_encoder_round_trip_state_dict` proves
  state_dict / load round-trip (`vae_encoder.rs:741..758`).

## Architecture

- `Encoder<T>` at `vae_encoder.rs:51..79` is the conv pipeline.
  `new` at `vae_encoder.rs:81..153` builds: `conv_in (k=3, pad=1)`,
  N down-blocks (`DownEncoderBlock2D::new(prev, c, layers,
  groups, 1e-6, !is_final)`), `UNetMidBlock2D` at top channels with
  eps=1e-6, `conv_norm_out (GroupNorm(groups, top, 1e-6))`, and
  `conv_out (k=3, pad=1, top → 2 * latent_channels)`.
- `VaeEncoder<T>` at `vae_encoder.rs:288..297` wraps `Encoder<T>`
  with a 1×1 `quant_conv` over `2 * latent_channels`. `new` at
  `vae_encoder.rs:299..316` builds both.
- `Module<T>::forward` at `vae_encoder.rs:369..382` validates
  `[B, out_channels, H, W]`, runs `encoder` then `quant_conv`.
- `encode` at `vae_encoder.rs:325..328` calls `forward` then hands
  the `[B, 2L, h, w]` parameters to
  `DiagonalGaussianDistribution::from_parameters`.
- `DiagonalGaussianDistribution<T>` at `vae_encoder.rs:454..460`
  holds `mean` and clamped `logvar`. `from_parameters` at
  `vae_encoder.rs:462..501` splits via `chunk(2, dim=1)` and
  clamps `logvar` to `[LOGVAR_CLAMP_MIN, LOGVAR_CLAMP_MAX] =
  [-30, 20]` via `clamp_t`.
- `mode` at `vae_encoder.rs:506..508` returns `&self.mean`.
- `sample_with_seed` at `vae_encoder.rs:527..539` computes
  `std = exp(0.5 * logvar)`, then `mean + std * eps` where `eps` is
  `randn_with_seed(self.mean.shape(), seed)`.
- `randn_with_seed` at `vae_encoder.rs:548..587` is a local xorshift64
  + Box-Muller generator (mirroring `ferrotorch_core::randn`'s CPU
  path but using `seed` as the state).
- `encode_with_scaling` at `vae_encoder.rs:348..361` composes
  `encode` → `sample_with_seed` → `* scaling_factor`.
- `Module<T>::load_state_dict` at `vae_encoder.rs:421..444` splits
  keys into `encoder.*` / `quant_conv.*`; strict mode rejects
  unknown prefixes.

Non-test production consumers:

- `ferrotorch-diffusion/src/safetensors_loader.rs:413..427`
  `load_vae_encoder` calls `VaeEncoder::<T>::new(cfg)` then
  `load_hf_state_dict`. This is the standard loader path.
- `ferrotorch-diffusion/src/gpu/vae_encoder.rs:317`
  `GpuVaeEncoder::from_module(cpu: &VaeEncoder<f32>, ...)` consumes
  the CPU module's `state_dict()` to build the GPU twin.
- `ferrotorch-diffusion/src/safetensors_loader.rs:363..396`
  `VaeEncoder::load_hf_state_dict` (impl block) is itself called by
  `load_vae_encoder`.

## Parity contract

`parity_ops = []`. The encoder's job is correct composition;
every Conv2d / GroupNorm / ResnetBlock2D / Downsample2D underneath
has its own op-level parity. The diagonal-Gaussian sampler is
INTENTIONALLY non-bit-exact with PyTorch CUDA-Philox: rust's
xorshift+Box-Muller PRNG is deterministic per seed on host but does
not match `torch.Generator(device='cuda').manual_seed(seed)`.
Tests use either `.mode()` (deterministic) or statistical-property
checks. The `logvar` clamp `[-30, 20]` matches
`DiagonalGaussianDistribution.__init__` in
`diffusers/models/autoencoders/vae.py` exactly.

End-to-end parity:
`ferrotorch-diffusion/tests/conformance_vae_encoder.rs` runs the
encoder against the pinned reference dump for the `.mode()` path.

## Verification

Lib tests at `vae_encoder.rs:589..788`:

- `encoder_forward_shape` (`vae_encoder.rs:609..625`)
- `vae_encoder_forward_shape` (`vae_encoder.rs:628..639`)
- `vae_encoder_named_parameters_include_quant_conv`
  (`vae_encoder.rs:642..657`)
- `diag_gauss_split_and_mode_shapes` (`vae_encoder.rs:660..683`)
- `diag_gauss_logvar_is_clamped` (`vae_encoder.rs:686..703`)
- `diag_gauss_sample_with_seed_is_deterministic`
  (`vae_encoder.rs:706..738`)
- `vae_encoder_round_trip_state_dict` (`vae_encoder.rs:741..758`)
- `encode_with_scaling_applies_scaling_factor`
  (`vae_encoder.rs:761..787`)

Integration: `tests/conformance_vae_encoder.rs`.

No parity-sweep ops apply (composition module).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `Encoder<T>` at `ferrotorch-diffusion/src/vae_encoder.rs:51..79`, `Encoder::new` at `ferrotorch-diffusion/src/vae_encoder.rs:81..153`; non-test consumer: `ferrotorch-diffusion/src/vae_encoder.rs:307` `VaeEncoder::new` builds it; itself consumed by `ferrotorch-diffusion/src/safetensors_loader.rs:425` `load_vae_encoder` |
| REQ-2 | SHIPPED | impl: `VaeEncoder<T>` at `ferrotorch-diffusion/src/vae_encoder.rs:288..297` and `VaeEncoder::new` at `ferrotorch-diffusion/src/vae_encoder.rs:299..316`; non-test consumer: `ferrotorch-diffusion/src/safetensors_loader.rs:425` `load_vae_encoder` instantiates and loads it; `ferrotorch-diffusion/src/gpu/vae_encoder.rs:317` `GpuVaeEncoder::from_module` consumes its `state_dict()` |
| REQ-3 | SHIPPED | impl: `VaeEncoder::encode` at `ferrotorch-diffusion/src/vae_encoder.rs:325..328` and `DiagonalGaussianDistribution::from_parameters` (with clamp) at `ferrotorch-diffusion/src/vae_encoder.rs:471..501`; non-test consumer: `ferrotorch-diffusion/src/vae_encoder.rs:349` `encode_with_scaling` invokes it; `load_vae_encoder` returns a `VaeEncoder` whose `encode` is the canonical production entry |
| REQ-4 | SHIPPED | impl: `DiagonalGaussianDistribution::sample_with_seed` at `ferrotorch-diffusion/src/vae_encoder.rs:527..539`, `mode` at `ferrotorch-diffusion/src/vae_encoder.rs:506..508`, `randn_with_seed` at `ferrotorch-diffusion/src/vae_encoder.rs:548..587`; non-test consumer: `ferrotorch-diffusion/src/vae_encoder.rs:350` `encode_with_scaling` calls `dist.sample_with_seed(seed)` on the production path |
| REQ-5 | SHIPPED | impl: `encode_with_scaling` at `ferrotorch-diffusion/src/vae_encoder.rs:348..361`; non-test consumer: re-exported via `ferrotorch-diffusion/src/lib.rs:148` `pub use vae_encoder::VaeEncoder` (which carries `encode_with_scaling` as an inherent method on the public type); the canonical SD pipeline call `latent = vae.encode(image).latent_dist.sample() * scaling_factor` is realised through this one function |
| REQ-6 | SHIPPED | impl: `Module<T>::forward` at `ferrotorch-diffusion/src/vae_encoder.rs:369..382` (shape check at `vae_encoder.rs:371..379`); non-test consumer: `ferrotorch-diffusion/src/vae_encoder.rs:326` `encode` calls `self.forward(image)?` to produce the `[B, 2L, h, w]` parameters |
| REQ-7 | SHIPPED | impl: `Module<T>::load_state_dict` at `ferrotorch-diffusion/src/vae_encoder.rs:421..444`; non-test consumer: `ferrotorch-diffusion/src/safetensors_loader.rs:394` `VaeEncoder::load_hf_state_dict` calls `self.load_state_dict(&remapped, strict)` after stripping the `vae.` prefix |
