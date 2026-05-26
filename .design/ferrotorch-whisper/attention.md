# ferrotorch-whisper — `attention` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - /home/doll/pytorch/torch/ (scaled_dot_product_attention at
    aten/src/ATen/native/transformers/attention.cpp; HF
    WhisperAttention at
    huggingface/transformers/src/transformers/models/whisper/modeling_whisper.py;
    openai/whisper/model.py is the original reference)
-->

## Summary

`ferrotorch-whisper/src/attention.rs` is the Whisper encoder
self-attention sub-block: Q / K / V / out_proj linear projections
(with K having NO bias — the HF Whisper convention), multi-head
scaled dot-product attention (non-causal, bidirectional). Unlike
BERT, the residual is NOT added here — that is the layer module's
responsibility (Whisper uses PRE-norm).

## Requirements

- REQ-1: `pub struct WhisperEncoderSelfAttention<T: Float>` holds
  four `ferrotorch_nn::Linear<T>` projections:
  - `q_proj` — `[d_model -> d_model]`, bias=true
  - `k_proj` — `[d_model -> d_model]`, **bias=false**
  - `v_proj` — `[d_model -> d_model]`, bias=true
  - `out_proj` — `[d_model -> d_model]`, bias=true
  plus cached `num_heads`, `head_dim`, `hidden` for the forward path.
- REQ-2: `Module::forward` projects QKV, reshapes to `[H, S, d]`,
  calls `ferrotorch_nn::standard_attention(q, k, v, causal=false)`,
  transposes back to `[S, H*d]`, reshapes to `[1, S, d_model]`, and
  applies the output projection. Output is the raw attention output
  (no residual add).
- REQ-3: HF state-dict key layout: `q_proj.{weight,bias}`,
  `k_proj.weight` (no bias key), `v_proj.{weight,bias}`,
  `out_proj.{weight,bias}`. Loadable in strict mode without
  rewriting keys.
- REQ-4: Input rank/shape validation: forward returns
  `FerrotorchError::ShapeMismatch` when the input is not
  `[1, S, d_model]`.
- REQ-5: `Module<T>` impl exposes the standard surface
  (`parameters` / `parameters_mut` / `named_parameters` /
  `state_dict` / `load_state_dict` / `train` / `eval`) for the four
  projections.

## Acceptance Criteria

- [x] AC-1: `WhisperEncoderSelfAttention::<f32>::new(&tiny_cfg)`
  constructs.
- [x] AC-2: `forward([1, 4, 16])` returns `[1, 4, 16]` with finite
  values.
- [x] AC-3: `named_parameters()` exposes the HF-layout keys with
  `q_proj.bias`, `v_proj.bias`, `out_proj.bias` present but
  `k_proj.bias` absent (HF Whisper convention — k_proj has NO bias).

## Architecture

`pub struct WhisperEncoderSelfAttention<T: Float>` in
`attention.rs` carries the four `Linear<T>` plus dim caches. The
key bias-asymmetry is in the constructor: `k_proj` is built with
`Linear::new(.., .., bias=false)`, the other three with
`bias=true`. The `named_parameters` block therefore emits
`k_proj.weight` but not `k_proj.bias`, and the round-trip
`load_state_dict` works in strict mode against a real HF Whisper
checkpoint.

The forward path uses the same shared shape utilities as
`ferrotorch-bert` / `ferrotorch-llama`:
`ferrotorch_nn::{reshape_to_heads, standard_attention,
transpose_heads_to_2d}`. The forward concludes by applying
`self.out_proj.forward(&ctx3)` so the output is the post-projection
attention tensor; the residual + LayerNorm wrap is done by
`WhisperEncoderLayer`.

Private helpers `reshape_2d` / `reshape_3d` at the bottom of
`attention.rs` own the data; needed because `reshape_to_heads`
consumes a 2-D tensor while the boundary is 3-D
`[1, S, d_model]`.

### Non-test production consumers

- `pub use attention::WhisperEncoderSelfAttention` at
  `ferrotorch-whisper/src/lib.rs:105`.
- `pub self_attn: WhisperEncoderSelfAttention<T>` field of
  `pub struct WhisperEncoderLayer` at
  `ferrotorch-whisper/src/layer.rs:23`; `WhisperEncoderLayer::new`
  at `ferrotorch-whisper/src/layer.rs:48` constructs it;
  `WhisperEncoderLayer::Module::forward` at
  `ferrotorch-whisper/src/layer.rs:62` invokes
  `self.self_attn.forward(&normed)`.

## Parity contract

`parity_ops = []`. The attention sub-block composes
`scaled_dot_product_attention` (covered by `ferrotorch-nn`'s
`standard_attention` parity) and `linear` (covered by
`ferrotorch-nn` parity).

Numerical / structural edge cases preserved:

- **Non-causal** (`standard_attention(..., causal=false)`). The
  Whisper encoder operates on the full mel spectrogram window;
  attention is bidirectional.
- **`k_proj.bias` is absent.** Matches HF
  `WhisperAttention.k_proj = nn.Linear(..., bias=False)` —
  distinguishes Whisper from BERT (which has all-three biases) and
  from Llama-style attention (which has no biases at all).
- **Output is post-projection, pre-residual.** The layer module is
  responsible for `input + self_attn(self_attn_layer_norm(input))`
  (PRE-norm residual).
- **Same scale as standard SDPA.** `softmax(QK^T / sqrt(d)) V` —
  the head_dim is `d_model / encoder_attention_heads` (=64 for
  Whisper-tiny).

## Verification

Tests in `mod tests in attention.rs`:

- `attention_shape`
- `named_parameters_match_hf_layout_no_k_bias`

No parity-sweep ops. Smoke command:

```bash
cargo test -p ferrotorch-whisper --lib attention:: 2>&1 | tail -3
```

Expected: all tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct WhisperEncoderSelfAttention<T: Float>` + `WhisperEncoderSelfAttention::new` (with `k_proj: Linear::new(.., .., false)`) at `ferrotorch-whisper/src/attention.rs:58`; non-test consumer: field `self_attn` of `pub struct WhisperEncoderLayer` at `ferrotorch-whisper/src/layer.rs:23`. |
| REQ-2 | SHIPPED | impl: `Module::forward` for `WhisperEncoderSelfAttention` at `ferrotorch-whisper/src/attention.rs:78`; non-test consumer: `WhisperEncoderLayer::Module::forward` at `ferrotorch-whisper/src/layer.rs:62` calls `self.self_attn.forward(&normed)`. |
| REQ-3 | SHIPPED | impl: `named_parameters` / `load_state_dict` for `WhisperEncoderSelfAttention` in `attention.rs`; non-test consumer: `WhisperEncoderLayer::load_state_dict` at `ferrotorch-whisper/src/layer.rs:163` recurses through `self_attn.*`. |
| REQ-4 | SHIPPED | impl: rank/shape check at the top of `Module::forward` for `WhisperEncoderSelfAttention` at `ferrotorch-whisper/src/attention.rs:79`; non-test consumer: propagated up through `WhisperEncoderLayer::Module::forward` at `ferrotorch-whisper/src/layer.rs:62`. |
| REQ-5 | SHIPPED | impl: `impl<T: Float> Module<T> for WhisperEncoderSelfAttention<T>` in `attention.rs`; non-test consumer: `Module` blanket calls from `WhisperEncoderLayer`'s `Module` impl at `ferrotorch-whisper/src/layer.rs:58` (parameters / state_dict / load_state_dict). |
