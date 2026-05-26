# UNet2DConditionConfig

<!--
tier: 3-component
status: draft
baseline-pytorch: /home/doll/pytorch (HEAD)
upstream-paths:
  - diffusers/src/diffusers/models/unets/unet_2d_condition.py
  - diffusers/src/diffusers/configuration_utils.py
-->

## Summary

Frozen configuration for the SD-1.5 `UNet2DConditionModel`. Mirrors
the public surface of `diffusers.models.unets.UNet2DConditionModel.config`
for the fields the UNet's forward pass consumes. Encoder-side and
training-only fields are not stored; the SD-1.5 defaults are
hard-wired in `Default::default` and exposed via `sd_v1_5()`.

## Requirements

- REQ-1: `UNet2DConditionConfig` carries the SD-1.5 published
  defaults: `in_channels=4`, `out_channels=4`,
  `block_out_channels=[320,640,1280,1280]`, `layers_per_block=2`,
  `attention_head_dim=8`, `cross_attention_dim=768`,
  `norm_num_groups=32`, `sample_size=64`, `flip_sin_to_cos=true`,
  `freq_shift=0`, `transformer_layers_per_block=1`,
  `down_block_has_attn=[true,true,true,false]`,
  `up_block_has_attn=[false,true,true,true]`.
- REQ-2: `validate()` rejects out-of-bounds fields and enforces:
  every `block_out_channels` entry divisible by both
  `norm_num_groups` and `attention_head_dim`;
  `down_block_has_attn.len() == up_block_has_attn.len() ==
  block_out_channels.len()`.
- REQ-3: `from_json_str()` parses `unet/config.json`. Translates
  diffusers's `down_block_types` / `up_block_types` arrays
  (`CrossAttnDownBlock2D`, `DownBlock2D`, `CrossAttnUpBlock2D`,
  `UpBlock2D`) into the boolean `*_has_attn` vectors. Unknown
  block types are rejected with `InvalidArgument`.
- REQ-4: `time_embed_dim()` returns `block_out_channels[0] * 4`
  (1280 for SD-1.5) — the canonical diffusers convention.

## Acceptance Criteria

- [x] AC-1: `default_is_sd_v1_5` matches every SD-1.5 default
  including `time_embed_dim() == 1280` and the boolean attn vectors
  (`unet_config.rs:305..318`).
- [x] AC-2: `from_json_parses_block_types` translates the
  diffusers block-type strings into the boolean vectors
  (`unet_config.rs:320..341`).

## Architecture

Pure config struct with public fields plus `validate` + JSON
parsers. No tensor math.

- `Default::default` (`unet_config.rs:51..69`) carries the
  SD-1.5 published values from `runwayml/stable-diffusion-v1-5/unet/config.json`.
- `validate` (`unet_config.rs:83..139`) enforces:
  - `block_out_channels` non-empty;
  - `norm_num_groups > 0`;
  - every channel divisible by both `norm_num_groups` and
    `attention_head_dim`;
  - all positive-only fields non-zero;
  - `down_block_has_attn.len() == up_block_has_attn.len() ==
    block_out_channels.len()`.
- `time_embed_dim` (`unet_config.rs:143..145`) — derived getter.
- `from_json_str` (`unet_config.rs:158..282`) — permissive parser
  with SD-1.5 fallback for missing keys. Translates
  `down_block_types`/`up_block_types` arrays into boolean
  `*_has_attn` vectors via per-string match (`CrossAttnDownBlock2D
  → true`, `DownBlock2D → false`, etc.); unknown block types raise
  `InvalidArgument`.

Non-test production consumers:

- `ferrotorch-diffusion/src/unet.rs:51` imports
  `UNet2DConditionConfig`; `UNet2DConditionModel::new` consumes it
  to size every sub-module.
- `ferrotorch-diffusion/src/safetensors_loader.rs:20` imports it
  for `load_unet`; `safetensors_loader.rs:165` is the function
  signature taking `cfg: UNet2DConditionConfig`.
- `ferrotorch-diffusion/src/pipeline.rs:239` uses
  `UNet2DConditionConfig::sd_v1_5()` (test-only) and
  `examples/unet_predict_dump.rs` / `examples/unet_probe_dump.rs`
  consume it in production dump binaries.

## Parity contract

`parity_ops = []`. The contract is structural: the SD-1.5 defaults
must agree byte-for-byte with the published `unet/config.json`
shipped under `runwayml/stable-diffusion-v1-5`. The
`down_block_types`/`up_block_types` translation must keep the
diffusers ordering invariant
(`[CrossAttn, CrossAttn, CrossAttn, DownBlock2D]` on the down side
and the mirror image on the up side).

## Verification

Two lib tests in `unet_config.rs:301..341`:

- `default_is_sd_v1_5` — every default field equals the published
  SD-1.5 value, including `time_embed_dim()`,
  `down_block_has_attn`, and `up_block_has_attn`. `validate` passes.
- `from_json_parses_block_types` — a JSON document with the
  diffusers `down_block_types`/`up_block_types` arrays parses into
  the canonical boolean vectors.

No parity-sweep ops apply.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `Default::default` at `ferrotorch-diffusion/src/unet_config.rs:51..69` and `sd_v1_5()` at `ferrotorch-diffusion/src/unet_config.rs:72..75`; non-test consumer: `ferrotorch-diffusion/src/safetensors_loader.rs:165` `load_unet` takes `cfg: UNet2DConditionConfig` and `UNet2DConditionModel::<T>::new(cfg)` at `safetensors_loader.rs:175` consumes it |
| REQ-2 | SHIPPED | impl: `validate` at `ferrotorch-diffusion/src/unet_config.rs:83..139`; non-test consumer: `from_json_str` at `ferrotorch-diffusion/src/unet_config.rs:280` calls `cfg.validate()?` before returning, so every production `from_file`/`from_json_str` path passes through validation |
| REQ-3 | SHIPPED | impl: `from_json_str` at `ferrotorch-diffusion/src/unet_config.rs:158..282` (block-type translation at `unet_config.rs:226..279`); non-test consumer: `from_file` at `ferrotorch-diffusion/src/unet_config.rs:290..298` and `examples/unet_predict_dump.rs` read `unet/config.json` from disk |
| REQ-4 | SHIPPED | impl: `time_embed_dim` at `ferrotorch-diffusion/src/unet_config.rs:143..145`; non-test consumer: `ferrotorch-diffusion/src/unet.rs` calls `cfg.time_embed_dim()` to size the `TimestepEmbedding` MLP and every `ResnetBlock2DTime` |
