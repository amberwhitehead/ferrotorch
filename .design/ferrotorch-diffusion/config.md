# VaeDecoderConfig

<!--
tier: 3-component
status: draft
baseline-pytorch: /home/doll/pytorch (HEAD)
upstream-paths:
  - diffusers/src/diffusers/models/autoencoders/autoencoder_kl.py
  - diffusers/src/diffusers/configuration_utils.py
-->

## Summary

Frozen configuration struct for the Stable-Diffusion v1.5 VAE decoder.
Mirrors the decoder-relevant subset of `diffusers.AutoencoderKL.config`
(`out_channels`, `latent_channels`, `block_out_channels`,
`layers_per_block`, `norm_num_groups`, `sample_size`,
`scaling_factor`). Encoder-only fields like `down_block_types` are
intentionally omitted — the decoder mirror is decoder-only by design.

## Requirements

- REQ-1: `VaeDecoderConfig` carries the exact SD-1.5 default values as
  `Default::default()` and as `VaeDecoderConfig::sd_v1_5()`:
  `out_channels=3`, `latent_channels=4`,
  `block_out_channels=[128,256,512,512]`, `layers_per_block=2`,
  `norm_num_groups=32`, `sample_size=512`, `scaling_factor=0.18215`.
- REQ-2: `validate()` rejects any out-of-bounds field with a
  `FerrotorchError::InvalidArgument` carrying the offending field
  name and value.
- REQ-3: `from_json_str()` parses a `vae/config.json` document into a
  `VaeDecoderConfig`, applying SD-1.5 defaults for missing keys.
- REQ-4: `resnets_per_up_block()` returns `layers_per_block + 1`
  (matching the diffusers convention: encoder has
  `layers_per_block`, decoder has one more).

## Acceptance Criteria

- [x] AC-1: `default_is_sd_v1_5` test passes
  (`config in config.rs`).
- [x] AC-2: `validate_catches_bad_groups` rejects `channels %
  norm_num_groups != 0` (`config.rs:232..239`).
- [x] AC-3: `from_json_str_round_trip` parses the upstream
  `vae/config.json` shape (`config in config.rs`).

## Architecture

Plain `struct` with public fields plus `validate` + `from_json_str` +
`from_file` constructors. All operations are pure Rust; no tensor
math, no GPU code.

- The SD-1.5 defaults live in `Default::default` at
  `config in config.rs`; `sd_v1_5()` at `config in config.rs` is an alias.
- `validate()` at `config in config.rs` enforces (a) `block_out_channels`
  non-empty, (b) `norm_num_groups > 0`, (c) each
  `block_out_channels` entry divisible by `norm_num_groups`,
  (d) positive `latent_channels` / `out_channels` /
  `layers_per_block` / `sample_size`, (e) `scaling_factor` finite and
  non-zero.
- `from_json_str()` at `config in config.rs` is permissive: any subset
  of the published keys overrides the defaults; the rest fall back to
  SD-1.5. Unknown extra keys (`in_channels`, `down_block_types`,
  `up_block_types`, `act_fn`) are silently ignored — they belong to
  the encoder side which we don't model here.
- `resnets_per_up_block()` at `config in config.rs` and `num_up_blocks`
  at `config in config.rs` are derived getters.

Non-test production consumers:

- `ferrotorch-diffusion/src/vae.rs` imports and uses
  `VaeDecoderConfig` to build the `VaeDecoder`.
- `ferrotorch-diffusion/src/vae_encoder.rs` imports it and aliases
  it as `VaeEncoderConfig` (encoder and decoder share the same
  config shape).
- `ferrotorch-diffusion/src/safetensors_loader.rs` imports it for
  `load_vae_decoder`.
- `ferrotorch-diffusion/src/gpu/vae_encoder.rs` imports it.
- `ferrotorch-diffusion/src/pipeline.rs:236` (test) and
  `examples/vae_decode_dump.rs` consume it through the crate
  re-export.

## Parity contract

`parity_ops = []`. This is a configuration carrier, not an op. The
contract is structural: the SD-1.5 default values must agree
byte-for-byte with the published `vae/config.json` shipped under
`runwayml/stable-diffusion-v1-5`.

## Verification

Tests in `config in config.rs`:

- `default_is_sd_v1_5` — asserts every default field matches the
  SD-1.5 published values, including `scaling_factor` within `1e-9`.
- `validate_catches_bad_groups` — `norm_num_groups = 33` is rejected
  because `128 % 33 != 0`.
- `from_json_str_round_trip` — a JSON document including encoder-side
  noise (`in_channels`, `down_block_types`, `act_fn`) parses cleanly,
  the decoder-relevant fields land in the struct, and the validator
  passes.

No parity-sweep ops (config-only).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `Default::default` at `ferrotorch-diffusion/src/config.rs:42..55` and `sd_v1_5` at `ferrotorch-diffusion/src/config.rs:58..61`; non-test consumer: `ferrotorch-diffusion/src/vae.rs:271` calls `VaeDecoder::<T>::new(cfg)` with this config |
| REQ-2 | SHIPPED | impl: `validate` at `ferrotorch-diffusion/src/config.rs:70..121`; non-test consumer: invoked from `from_json_str` at `ferrotorch-diffusion/src/config.rs:191` and from `VaeDecoder::new` at `ferrotorch-diffusion/src/vae.rs:62` |
| REQ-3 | SHIPPED | impl: `from_json_str` at `ferrotorch-diffusion/src/config.rs:148..193`; non-test consumer: `from_file` at `ferrotorch-diffusion/src/config.rs:201..209` and the dump examples that read `vae/config.json` from disk |
| REQ-4 | SHIPPED | impl: `resnets_per_up_block` at `ferrotorch-diffusion/src/config.rs:125..127`; non-test consumer: `ferrotorch-diffusion/src/vae.rs:88` uses `cfg.resnets_per_up_block()` to size each `UpDecoderBlock2D` |
