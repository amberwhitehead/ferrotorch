# safetensors loader (SD-1.5 model checkpoint loaders)

<!--
tier: 3-component
status: draft
baseline-pytorch: /home/doll/pytorch (HEAD)
upstream-paths:
  - diffusers/src/diffusers/models/autoencoders/autoencoder_kl.py
  - diffusers/src/diffusers/models/unets/unet_2d_condition.py
  - transformers/src/transformers/models/clip/modeling_clip.py
-->

## Summary

Production loaders that turn a path-to-`*.safetensors` plus a parsed
config into a fully-populated SD-1.5 sub-model. Provides:
`load_vae_decoder`, `load_vae_encoder`, `load_unet`,
`load_clip_text_encoder`, plus inherent `load_hf_state_dict` methods
on the model types that handle the diffusers HF prefix layouts
(bare and `vae.` / `unet.` / `text_model.` prefixed) and return
a `DropReport` listing keys that were intentionally dropped (e.g.
loading a full `AutoencoderKL` checkpoint into a decoder-only model).

## Requirements

- REQ-1: `DropReport` records HF keys not consumed by the target
  model (sorted, deterministic equality), so a pin-script audit can
  assert the drop set is exactly the documented set.
- REQ-2: `VaeDecoder::load_hf_state_dict` accepts both bare
  (`post_quant_conv.*` / `decoder.*`) and `vae.<rest>`-prefixed key
  layouts. Strict mode rejects anything outside that prefix set;
  non-strict mode silently drops `encoder.*` / `quant_conv.*` keys
  into `DropReport`.
- REQ-3: `VaeEncoder::load_hf_state_dict` mirrors REQ-2 for the
  encoder half: accepts bare (`encoder.*` / `quant_conv.*`) and
  `vae.<rest>` prefixes; drops `decoder.*` / `post_quant_conv.*` in
  non-strict mode.
- REQ-4: `UNet2DConditionModel::load_hf_state_dict` accepts both
  bare (`time_embedding.*` / `conv_in.*` / `down_blocks.*` /
  `mid_block.*` / `up_blocks.*` / `conv_norm_out.*` / `conv_out.*`)
  and `unet.<rest>`-prefixed layouts.
- REQ-5: `load_clip_text_encoder` drops the int64
  `embeddings.position_ids` (or `text_model.embeddings.position_ids`)
  buffer BEFORE the generic-`T` decode, so a `load_safetensors::<f32>`
  pass does not fail on the i64 tensor. The drop set is surfaced via
  `DropReport`. The same routine accepts both bare and
  `text_model.<rest>`-prefixed layouts via the encoder's own
  `load_hf_state_dict`.
- REQ-6: `load_unet`, `load_vae_decoder`, `load_vae_encoder`,
  `load_clip_text_encoder` are the top-level entry points: read +
  parse the safetensors file, construct the model from the config,
  invoke `load_hf_state_dict`, return `(model, DropReport)`.

## Acceptance Criteria

- [x] AC-1: `round_trip_safetensors_into_decoder` saves a fresh
  `VaeDecoder.state_dict()` to a temp file, loads it back, and
  proves the forward outputs match to 1e-5
  (`safetensors_loader.rs:460..482`).
- [x] AC-2: `load_hf_drops_encoder_keys_nonstrict` proves
  `encoder.*` and `quant_conv.*` are dropped (not loaded) when
  loading a full checkpoint into a decoder-only model
  (`safetensors_loader.rs:484..507`).
- [x] AC-3: `load_hf_strict_rejects_encoder_keys` proves strict
  mode rejects them (`safetensors_loader.rs:509..519`).
- [x] AC-4: `load_hf_strips_vae_prefix` proves the `vae.<rest>`
  layout is accepted (`safetensors_loader.rs:521..545`).
- [x] AC-5: `round_trip_safetensors_into_encoder` mirrors AC-1 for
  the encoder side (`safetensors_loader.rs:555..577`).
- [x] AC-6: `encoder_load_hf_drops_decoder_keys_nonstrict`,
  `encoder_load_hf_strict_rejects_decoder_keys`,
  `encoder_load_hf_strips_vae_prefix`
  (`safetensors_loader.rs:579..638`).
- [x] AC-7: `full_vae_checkpoint_loadable_by_both_halves` proves a
  single combined state-dict is loadable by both encoder and
  decoder with the right drop set on each side
  (`safetensors_loader.rs:640..685`).

## Architecture

- `DropReport` at `safetensors_loader.rs:31..36` is a small struct
  carrying a sorted `Vec<String>` of dropped keys.
- `VaeDecoder::load_hf_state_dict` at
  `safetensors_loader.rs:58..91`: per-key, strip optional `vae.`
  prefix, keep iff `post_quant_conv.` / `decoder.` prefix; strict
  mode raises `InvalidArgument` on unmapped keys; otherwise records
  in `dropped`. After the filter, delegates to
  `self.load_state_dict(&remapped, strict)`.
- `UNet2DConditionModel::load_hf_state_dict` at
  `safetensors_loader.rs:113..148`: same pattern with the seven UNet
  prefixes (`time_embedding.`, `conv_in.`, `down_blocks.`,
  `mid_block.`, `up_blocks.`, `conv_norm_out.`, `conv_out.`).
- `VaeEncoder::load_hf_state_dict` at
  `safetensors_loader.rs:363..396`: same pattern with `encoder.` /
  `quant_conv.` prefixes.
- `load_unet` at `safetensors_loader.rs:163..178`:
  `load_safetensors::<T>` → `UNet2DConditionModel::new(cfg)` →
  `load_hf_state_dict`.
- `load_safetensors_clip_filtered` at
  `safetensors_loader.rs:191..246` is the CLIP-specific helper:
  parses the safetensors file, drops the int64 `position_ids`
  buffer, re-serializes the remaining tensors into a temp file, then
  delegates to the generic `load_safetensors::<T>`. The
  `had_position_ids` flag is propagated for the `DropReport`.
- `load_clip_text_encoder` at `safetensors_loader.rs:270..302` calls
  the filtered loader, optionally re-inserts a placeholder
  `position_ids` entry (so `DropReport` captures the upstream key),
  constructs the encoder, and runs `load_hf_state_dict`.
- `load_vae_decoder` at `safetensors_loader.rs:318..333` and
  `load_vae_encoder` at `safetensors_loader.rs:413..428` are the
  obvious mirror loaders.

Non-test production consumers:

- `ferrotorch-diffusion/src/lib.rs:137..139` re-exports `DropReport`,
  `load_clip_text_encoder`, `load_unet`, `load_vae_decoder`,
  `load_vae_encoder`.
- `ferrotorch-diffusion/examples/unet_predict_dump.rs:45,292`
  imports `load_unet` and calls it on the SD-1.5 mirror.
- `ferrotorch-diffusion/examples/unet_probe_dump.rs:15,121`
  imports `load_unet`.
- `ferrotorch-diffusion/examples/vae_decode_dump.rs:42,218`
  imports `load_vae_decoder`.
- `ferrotorch-diffusion/examples/clip_text_encode_dump.rs:44,265`
  imports `load_clip_text_encoder`.
- `ferrotorch-diffusion/examples/sd_pipeline_dump.rs:48` imports
  all three top-level loaders for the end-to-end dump pipeline.

## Parity contract

`parity_ops = []`. The loader is correctness-via-shape only:
every key that lands in a module's `load_state_dict` must match the
declared shape exactly; mismatches surface as
`ShapeMismatch`. End-to-end byte-for-byte parity is checked by
`tests/conformance_pretrained_diffusion.rs` against the pinned
`runwayml/stable-diffusion-v1-5` mirror.

The HF prefix translation rules match diffusers' own dual layout:
single-file `vae/diffusion_pytorch_model.safetensors` ships bare,
while `model.safetensors` from a full-pipeline checkpoint ships
prefixed. This loader handles both via prefix-strip.

## Verification

Lib tests at `safetensors_loader.rs:430..686`:

- VAE decoder: `round_trip_safetensors_into_decoder`,
  `load_hf_drops_encoder_keys_nonstrict`,
  `load_hf_strict_rejects_encoder_keys`,
  `load_hf_strips_vae_prefix`.
- VAE encoder: `round_trip_safetensors_into_encoder`,
  `encoder_load_hf_drops_decoder_keys_nonstrict`,
  `encoder_load_hf_strict_rejects_decoder_keys`,
  `encoder_load_hf_strips_vae_prefix`.
- Cross: `full_vae_checkpoint_loadable_by_both_halves`.

Integration: `tests/conformance_pretrained_diffusion.rs` exercises
all four loaders end-to-end against the pinned mirror.

No parity-sweep ops apply.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `DropReport` at `ferrotorch-diffusion/src/safetensors_loader.rs:31..36`; non-test consumer: `ferrotorch-diffusion/src/safetensors_loader.rs:88..90` sorts and returns the populated `dropped` set; `ferrotorch-diffusion/src/lib.rs:138` re-exports it for external pin-script auditing |
| REQ-2 | SHIPPED | impl: `VaeDecoder::load_hf_state_dict` at `ferrotorch-diffusion/src/safetensors_loader.rs:58..91`; non-test consumer: `load_vae_decoder` at `ferrotorch-diffusion/src/safetensors_loader.rs:331` invokes it on every checkpoint load |
| REQ-3 | SHIPPED | impl: `VaeEncoder::load_hf_state_dict` at `ferrotorch-diffusion/src/safetensors_loader.rs:363..396`; non-test consumer: `load_vae_encoder` at `ferrotorch-diffusion/src/safetensors_loader.rs:426` invokes it |
| REQ-4 | SHIPPED | impl: `UNet2DConditionModel::load_hf_state_dict` at `ferrotorch-diffusion/src/safetensors_loader.rs:113..148`; non-test consumer: `load_unet` at `ferrotorch-diffusion/src/safetensors_loader.rs:176` invokes it |
| REQ-5 | SHIPPED | impl: `load_safetensors_clip_filtered` at `ferrotorch-diffusion/src/safetensors_loader.rs:191..246` (position_ids filter) and `load_clip_text_encoder` at `ferrotorch-diffusion/src/safetensors_loader.rs:270..302`; non-test consumer: `ferrotorch-diffusion/examples/clip_text_encode_dump.rs:265` invokes `load_clip_text_encoder` on the SD-1.5 mirror |
| REQ-6 | SHIPPED | impl: `load_unet` at `ferrotorch-diffusion/src/safetensors_loader.rs:163..178`, `load_vae_decoder` at `safetensors_loader.rs:318..333`, `load_vae_encoder` at `safetensors_loader.rs:413..428`, `load_clip_text_encoder` at `safetensors_loader.rs:270..302`; non-test consumer: all four examples (`unet_predict_dump.rs`, `vae_decode_dump.rs`, `clip_text_encode_dump.rs`, `sd_pipeline_dump.rs`) import and call them |
