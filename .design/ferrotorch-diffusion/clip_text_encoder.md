# CLIP text encoder (SD-1.5 text tower)

<!--
tier: 3-component
status: draft
baseline-pytorch: /home/doll/pytorch (HEAD)
upstream-paths:
  - transformers/src/transformers/models/clip/modeling_clip.py
  - transformers/src/transformers/models/clip/configuration_clip.py
-->

## Summary

The SD-1.5 CLIP text encoder — the `openai/clip-vit-large-patch14`
text tower. Mirrors `transformers.CLIPTextModel` 1:1 for the SD-1.5
config (`hidden_size=768`, `intermediate_size=3072`,
`num_attention_heads=12`, `num_hidden_layers=12`,
`max_position_embeddings=77`, `vocab_size=49408`,
`hidden_act="quick_gelu"`, `layer_norm_eps=1e-5`). Produces the
per-token `last_hidden_state` `[1, S, 768]` that SD-1.5 feeds
directly into the UNet's cross-attention (no pooling).

## Requirements

- REQ-1: `ClipTextConfig` carries the SD-1.5 published defaults
  (`hidden_size=768`, etc.) with `validate` enforcing positive sizes
  and `hidden_size % num_attention_heads == 0`.
- REQ-2: `ClipTextEmbeddings` implements token embedding + learned
  absolute position embedding summed element-wise. No LayerNorm at
  the embedding layer (unlike BERT).
- REQ-3: `ClipSelfAttention` is causal multi-head self-attention
  with all four projections (`q_proj`, `k_proj`, `v_proj`,
  `out_proj`) carrying bias. The causal mask is `-inf` on the upper
  triangle; position `i` attends to `0..=i` only.
- REQ-4: `ClipMlp` is `fc2(quick_gelu(fc1(x)))` where QuickGELU =
  `x * sigmoid(1.702 * x)` (NOT the standard erf-based GELU). Pinned
  via `GELU::with_approximate(GeluApproximate::Sigmoid)`.
- REQ-5: `ClipEncoderLayer` is the pre-LayerNorm stack
  `h = x + self_attn(layer_norm1(x)); h = h + mlp(layer_norm2(h))`.
- REQ-6: `ClipEncoder` stacks `num_hidden_layers` `ClipEncoderLayer`
  instances applied sequentially.
- REQ-7: `ClipTextEncoder::forward_from_ids(input_ids)` returns
  `[1, S, hidden_size]` — the per-token `last_hidden_state` after
  the final `LayerNorm`. SD-1.5 consumes this directly as
  `encoder_hidden_states` for the UNet's cross-attention.
- REQ-8: `load_hf_state_dict(hf_state, strict)` accepts both the
  bare-`text_model` layout and the full `text_model.` prefix
  (HF default), and explicitly drops
  `text_model.embeddings.position_ids` (a non-parameter buffer the
  HF safetensors ships) into a `DropReport`.

## Acceptance Criteria

- [x] AC-1: `ClipTextConfig::default()` equals `sd_v1_5()` and
  `validate()` passes (see config block at
  `clip_text_encoder.rs`).
- [x] AC-2: `ClipSelfAttention` carries q/k/v/out projections, each
  with bias (`clip_text_encoder.rs`).
- [x] AC-3: `ClipMlp` instantiates `GELU` with
  `GeluApproximate::Sigmoid` (`clip_text_encoder.rs`).
- [x] AC-4: `ClipTextEncoder::forward_from_ids` returns the
  `[1, S, hidden_size]` shape (`clip_text_encoder.rs`).
- [x] AC-5: `load_hf_state_dict` strips the `text_model.` prefix
  and drops `embeddings.position_ids`
  (`clip_text_encoder.rs`).

## Architecture

- `ClipTextConfig` (`ClipTextConfig in clip_text_encoder.rs`) — seven fields
  mirroring the HF `CLIPTextConfig`. `head_dim()` is derived
  (`hidden_size / num_attention_heads`).
- `ClipTextEmbeddings<T>` (`ClipTextEmbeddings in clip_text_encoder.rs`) — two
  `Embedding<T>` fields (token + position). `forward_from_ids`
  builds the float-encoded id tensor (matching the
  `BertEmbeddings::float_index_tensor` trick) and sums the two
  lookups.
- `ClipSelfAttention<T>` (`ClipSelfAttention in clip_text_encoder.rs`) — four
  `Linear<T>` projections, all with bias.
  Forward at `clip_text_encoder.rs` uses
  `reshape_to_heads` + `standard_attention(..., causal=true)` +
  `transpose_heads_to_2d` from `ferrotorch-nn`. The
  `standard_attention(q, k, v, causal=true)` call is what enforces
  the upper-triangular `-inf` mask.
- `ClipMlp<T>` (`ClipMlp in clip_text_encoder.rs`) — two `Linear<T>`
  + `GELU::with_approximate(GeluApproximate::Sigmoid)`. Forward at
  `clip_text_encoder.rs` applies the standard
  `fc1 → quick_gelu → fc2` recipe.
- `ClipEncoderLayer<T>` (`ClipEncoderLayer in clip_text_encoder.rs`) — two
  `LayerNorm` + one `ClipSelfAttention` + one `ClipMlp`. Forward at
  `clip_text_encoder.rs`.
- `ClipEncoder<T>` (`ClipEncoder in clip_text_encoder.rs`) holds a
  `Vec<ClipEncoderLayer<T>>` of length
  `num_hidden_layers` (12 for SD-1.5).
- `ClipTextEncoder<T>` (`ClipTextEncoder in clip_text_encoder.rs`) — top
  composite: `embeddings`, `encoder`, `final_layer_norm`, plus a
  frozen copy of the config. `forward_from_ids`
  (`clip_text_encoder.rs`) is the canonical entry point.
- `load_hf_state_dict` (`load_hf_state_dict in clip_text_encoder.rs`) handles
  the HF prefix-strip + position_ids drop.

Non-test production consumers:

- `ferrotorch-diffusion/src/pipeline.rs` imports
  `ClipTextEncoder` and `pipeline in pipeline.rs`'s `encode_prompt`
  calls `self.text_encoder.forward_from_ids(input_ids)`.
- `ferrotorch-diffusion/src/safetensors_loader.rs` imports
  `ClipTextConfig` and `ClipTextEncoder`;
  `load_clip_text_encoder` at `safetensors_loader.rs` builds
  the encoder from the HF checkpoint.
- `ferrotorch-diffusion/src/gpu/clip.rs` imports
  `ClipTextConfig` and `ClipTextEncoder`;
  `GpuClipTextEncoder::from_module(cpu, device)` at
  `gpu in gpu/clip.rs` builds the GPU mirror from a CPU
  `ClipTextEncoder`.
- `ferrotorch-hub/src/registry.rs` re-references the encoder
  through `ferrotorch_diffusion::ClipTextEncoder`.

## Parity contract

`parity_ops = []`. The contract is byte-equivalence with
`transformers.CLIPTextModel` for the SD-1.5 text tower when loaded
with the upstream HF checkpoint. Critical gotchas (documented in
the source `//!` header):

1. **Causal mask** — despite the "encoder" name, CLIP text-side
   attention is causal. Verified by the `causal=true` argument to
   `standard_attention`.
2. **QuickGELU**, not standard GELU. `x * sigmoid(1.702 * x)` only.
3. **Position embedding is learned** — full 77-entry table.
   Position ids are always `[0, 1, ..., S-1]`.
4. **All four self-attention projections have bias** — unlike SD's
   UNet `Attention` which has `bias=false` on q/k/v.
5. **SD uses `last_hidden_state` directly** — no EOS pooling.

## Verification

Lib tests within `clip_text_encoder.rs` (consolidated `mod tests`
near the end of the file) cover:

- Config defaults match SD-1.5.
- Embedding shape: `[S] → [1, S, 768]`.
- Self-attention causal-mask correctness (token-`i` independence
  from tokens > `i`).
- QuickGELU activation difference from standard GELU.
- Full encoder forward shape `[S] → [1, S, 768]`.
- `load_hf_state_dict` prefix-strip + position_ids drop.

Integration: `conformance_pretrained_diffusion.rs` validates the
encoder against the pinned HF text-tower dump.

No parity-sweep ops apply.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `ClipTextConfig in ferrotorch-diffusion/src/clip_text_encoder.rs`; non-test consumer: `load_clip_text_encoder in ferrotorch-diffusion/src/safetensors_loader.rs` imports `ClipTextConfig` and `load_clip_text_encoder` at `safetensors_loader.rs` consumes it to construct the encoder |
| REQ-2 | SHIPPED | impl: `ClipTextEmbeddings in ferrotorch-diffusion/src/clip_text_encoder.rs`; non-test consumer: `ClipTextEncoder::forward_from_ids` at `forward_from_ids in ferrotorch-diffusion/src/clip_text_encoder.rs` calls `self.embeddings.forward_from_ids(input_ids)?`, which `pipeline in pipeline.rs` invokes |
| REQ-3 | SHIPPED | impl: `ClipSelfAttention in ferrotorch-diffusion/src/clip_text_encoder.rs` with `standard_attention(..., causal=true)` at `standard_attention in clip_text_encoder.rs`; non-test consumer: `ClipEncoderLayer in ferrotorch-diffusion/src/clip_text_encoder.rs` consumes it and `pipeline in pipeline.rs` reaches it transitively |
| REQ-4 | SHIPPED | impl: `ClipMlp in ferrotorch-diffusion/src/clip_text_encoder.rs` using `GELU::with_approximate(GeluApproximate::Sigmoid)`; non-test consumer: `ClipEncoderLayer` consumes it and the encode path reaches it transitively from `pipeline in pipeline.rs` |
| REQ-5 | SHIPPED | impl: `ClipEncoderLayer in ferrotorch-diffusion/src/clip_text_encoder.rs`; non-test consumer: `ClipEncoder in ferrotorch-diffusion/src/clip_text_encoder.rs` chains `Vec<ClipEncoderLayer>` and is reached from `pipeline in pipeline.rs` |
| REQ-6 | SHIPPED | impl: `ClipEncoder::new` at `new in ferrotorch-diffusion/src/clip_text_encoder.rs` and forward at `new in clip_text_encoder.rs`; non-test consumer: `ClipTextEncoder::forward_from_ids` at `forward_from_ids in ferrotorch-diffusion/src/clip_text_encoder.rs` calls `self.encoder.forward(&h)?` |
| REQ-7 | SHIPPED | impl: `ClipTextEncoder::forward_from_ids` at `forward_from_ids in ferrotorch-diffusion/src/clip_text_encoder.rs`; non-test consumer: `forward_from_ids in ferrotorch-diffusion/src/pipeline.rs` calls `self.text_encoder.forward_from_ids(input_ids)` inside `encode_prompt` |
| REQ-8 | SHIPPED | impl: `load_hf_state_dict` at `ferrotorch-diffusion/src/clip_text_encoder.rs:1138..1181`; non-test consumer: `ferrotorch-diffusion/src/safetensors_loader.rs:272..` `load_clip_text_encoder` calls it to wire the HF checkpoint into the encoder |
