# ferrotorch-whisper — `encoder` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - /home/doll/pytorch/torch/ (Conv1d / LayerNorm / Module
    foundations; HF WhisperEncoder at
    huggingface/transformers/src/transformers/models/whisper/modeling_whisper.py;
    openai/whisper/model.py is the original architecture reference)
-->

## Summary

`ferrotorch-whisper/src/encoder.rs` is the full Whisper audio encoder:
a 2× Conv1d + GELU stem, a sinusoidal positional embedding added to
the post-stem hidden state, `N × WhisperEncoderLayer`, and a final
LayerNorm. Mirrors HF's `WhisperEncoder`. Out of scope: the decoder
(cross-attention, kv-cache, beam search) and `proj_out` — Phase B.2
is encoder-only.

## Requirements

- REQ-1: `pub struct WhisperConvStem<T: Float>` carries:
  - `conv1: Conv1d<T>` (`num_mel_bins -> d_model`, k=3, stride=1,
    pad=1, bias=true)
  - `conv2: Conv1d<T>` (`d_model -> d_model`, k=3, stride=2, pad=1,
    bias=true)
  - `activation: GELU`
  `Module::forward` runs `GELU(conv1(input))` → `GELU(conv2(.))`.
- REQ-2: `pub struct WhisperEncoder<T: Float>` carries:
  - `conv_stem: WhisperConvStem<T>`
  - `embed_positions: Parameter<T>` (shape
    `[max_source_positions, d_model]`)
  - `layers: Vec<WhisperEncoderLayer<T>>` (length =
    `cfg.encoder_layers`)
  - `layer_norm: LayerNorm<T>` (final post-stack LayerNorm)
  - `config: WhisperConfig` (frozen copy)
- REQ-3: `WhisperEncoder::forward_from_mel(mel)` consumes a
  `[1, num_mel_bins, max_source_positions * 2]` log-mel spectrogram
  and returns `[1, max_source_positions, d_model]`. Pipeline:
  conv stem → transpose `[1, C, T] → [1, T, C]` → add positional
  embedding → N encoder layers → final LayerNorm.
- REQ-4: Input shape validation: `forward_from_mel` returns
  `FerrotorchError::ShapeMismatch` when the input is not
  `[1, num_mel_bins, max_source_positions * 2]`. After the conv
  stem, a second shape assertion checks that the conv halved the
  time axis to `max_source_positions`.
- REQ-5: `embed_positions.weight` is loaded from the HF state dict
  as a parameter (HF ships it as a non-trainable buffer initialised
  to the sinusoidal pattern; ferrotorch stores it as a parameter so
  the state-dict key layout maps directly).
- REQ-6: `WhisperEncoder::load_hf_state_dict` accepts a `StateDict`
  whose keys are prefixed `encoder.` (raw `WhisperEncoder`
  checkpoint) or `model.encoder.` (full
  `WhisperForConditionalGeneration` checkpoint). Strips the
  prefix and forwards to `load_state_dict`. Keys without either
  prefix are recorded in `DropReport` (non-strict) or rejected
  (strict).
- REQ-7: `pub struct DropReport` records every key the
  HF-aware loader did not consume, sorted for deterministic
  equality, so the pin script can cross-check that the drop set is
  exactly the decoder/proj_out surface (no silent encoder-key drop).
- REQ-8: HF state-dict key layout for `WhisperEncoder`:
  `conv1.{weight,bias}`, `conv2.{weight,bias}`,
  `embed_positions.weight`, `layers.{i}.{...}`,
  `layer_norm.{weight,bias}`. Loadable in strict mode (after
  HF-prefix strip) without rewriting keys.
- REQ-9: Round-trip — saving the encoder's state dict and loading
  it back reproduces `forward_from_mel` output (tolerance `1e-6`
  on f32).

## Acceptance Criteria

- [x] AC-1: `WhisperConvStem::<f32>::new(&tiny_cfg)` constructs.
- [x] AC-2: `WhisperConvStem::forward([1, 80, 8])` returns
  `[1, 8, 4]` (conv2's stride-2 halves the time axis).
- [x] AC-3: `WhisperEncoder::<f32>::new(tiny_cfg)` constructs.
- [x] AC-4: `WhisperEncoder::forward_from_mel([1, 80, 8])` returns
  `[1, 4, 8]` with finite values.
- [x] AC-5: `named_parameters()` exposes the HF-layout keys
  (`conv1.weight`, `conv1.bias`, `conv2.weight`, `conv2.bias`,
  `embed_positions.weight`, `layers.0.self_attn.q_proj.weight`,
  `layers.0.self_attn.k_proj.weight`,
  `layers.0.self_attn_layer_norm.weight`,
  `layers.0.final_layer_norm.weight`, `layers.0.fc1.weight`,
  `layers.0.fc2.weight`, `layer_norm.weight`, `layer_norm.bias`).
- [x] AC-6: Round-trip state-dict load reproduces the original
  `forward_from_mel` output (tolerance `1e-6`).
- [x] AC-7: `load_hf_state_dict(strict=false)` drops a
  `decoder.embed_tokens.weight` key and records it in the
  `DropReport`.
- [x] AC-8: `load_hf_state_dict(strict=true)` rejects a
  `decoder.embed_tokens.weight` key.

## Architecture

`pub struct WhisperConvStem<T: Float>` in `encoder.rs` packages the
two `Conv1d` layers plus the GELU. Its `Module::forward` applies
`conv1 → GELU → conv2 → GELU`; the second GELU matches HF (the
upstream `WhisperEncoder` runs GELU after both conv layers).

`pub struct WhisperEncoder<T: Float>` in `encoder.rs` owns the stem,
the positional-embedding parameter, the layer vector, the final
LayerNorm, and a frozen `WhisperConfig`. The `forward_from_mel`
path:

1. Validate input shape.
2. Run the conv stem → assert post-stem shape.
3. `transpose_b_c_t_to_b_t_c` (`[1, C, T] → [1, T, C]`) via the
   private helper at the bottom of `encoder.rs` (owns the data; no
   view tricks).
4. `reshape_pos` materialises `embed_positions.weight` as a
   `[1, max_pos, d_model]` tensor so the `add` broadcasts.
5. Iterate the layers.
6. Apply the final `layer_norm`.

`WhisperEncoder::load_hf_state_dict` strips one of two known prefixes
(`encoder.` or `model.encoder.`) and dispatches to the recursive
`load_state_dict`. Keys without either prefix are recorded in
`DropReport.dropped` (non-strict) or returned as `InvalidArgument`
(strict). The recursive `load_state_dict` then handles per-
sub-module dispatch including the scalar `embed_positions.weight`
key (which is `Parameter::new(pos.clone())`-replaced rather than
loaded through a sub-module's `load_state_dict`).

`pub struct DropReport` in `encoder.rs` is the audit-trail return
type — Phase B.2's analog of `ferrotorch-bert::DropReport`. It
records every dropped upstream key so the pin script can confirm
the drop set is exactly the documented decoder/proj_out surface.

### Non-test production consumers

- `pub use encoder::{DropReport, WhisperConvStem, WhisperEncoder}`
  at `ferrotorch-whisper/src/lib.rs:108`.
- `load_whisper_encoder` at
  `ferrotorch-whisper/src/safetensors_loader.rs:31` constructs
  `WhisperEncoder::<T>::new(cfg)?` then calls
  `encoder.load_hf_state_dict(&state, strict)` and returns
  `(WhisperEncoder<T>, DropReport)`.

## Parity contract

`parity_ops = []`. The encoder composes
`Conv1d` (covered by `ferrotorch-nn` parity), GELU (covered),
LayerNorm (covered), `Linear` + `standard_attention` (covered), the
sinusoidal positional embedding (parameter — not a parity op).

Numerical / structural edge cases preserved:

- **Conv2 has stride=2.** Halves the time axis (3000 → 1500 for
  whisper-tiny). A post-stem shape assertion catches misconfig.
- **GELU runs after both convs.** Matches HF
  `WhisperEncoder.forward`.
- **Positional embedding stored as parameter, not buffer.** HF
  marks `embed_positions` as a non-trainable buffer initialised
  with the sinusoidal pattern; ferrotorch stores it as a
  `Parameter<T>` so the state-dict layout maps directly. There is
  no training-loop concern (encoder-only, no autograd through this
  module).
- **Final LayerNorm.** `encoder.layer_norm` is applied after the
  stack — matches HF.
- **Prefix flexibility.** `load_hf_state_dict` accepts either
  `encoder.<rest>` (raw encoder checkpoint) or
  `model.encoder.<rest>` (full
  `WhisperForConditionalGeneration` checkpoint). Both arise in real
  Hub artifacts.

## Verification

Tests in `mod tests in encoder.rs`:

- `conv_stem_shape`
- `encoder_forward_shape`
- `named_parameters_match_hf_layout`
- `round_trip_state_dict`
- `load_hf_drops_decoder_keys_nonstrict`
- `load_hf_strict_rejects_decoder_keys`

No parity-sweep ops. Smoke command:

```bash
cargo test -p ferrotorch-whisper --lib encoder:: 2>&1 | tail -3
```

Expected: all tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct WhisperConvStem<T: Float>` + its `Module<T>` impl in `encoder.rs`; non-test consumer: field `conv_stem` of `pub struct WhisperEncoder` in `encoder.rs`; `WhisperEncoder::forward_from_mel` in `encoder.rs` calls `self.conv_stem.forward(mel)`. |
| REQ-2 | SHIPPED | impl: `pub struct WhisperEncoder<T: Float>` + `WhisperEncoder::new` in `encoder.rs`; non-test consumer: `load_whisper_encoder` at `ferrotorch-whisper/src/safetensors_loader.rs:31` constructs and returns it. |
| REQ-3 | SHIPPED | impl: `WhisperEncoder::forward_from_mel` in `encoder.rs`; non-test consumer: `Module::forward` for `WhisperEncoder` in `encoder.rs` delegates to `self.forward_from_mel(input)`, and `load_whisper_encoder` at `ferrotorch-whisper/src/safetensors_loader.rs:31` returns the encoder so callers can invoke `forward_from_mel`. |
| REQ-4 | SHIPPED | impl: shape checks at the top of `forward_from_mel` and after the conv stem in `encoder.rs`; non-test consumer: same call path as REQ-3 (errors propagate through `Module::forward`). |
| REQ-5 | SHIPPED | impl: `embed_positions: Parameter<T>` field initialised to zeros in `WhisperEncoder::new` and replaced via `self.embed_positions = Parameter::new(pos.clone())` in `WhisperEncoder::load_state_dict` in `encoder.rs`; non-test consumer: `load_whisper_encoder` at `ferrotorch-whisper/src/safetensors_loader.rs:31` invokes the load path. |
| REQ-6 | SHIPPED | impl: `WhisperEncoder::load_hf_state_dict` in `encoder.rs`; non-test consumer: `load_whisper_encoder` at `ferrotorch-whisper/src/safetensors_loader.rs:31` calls it. |
| REQ-7 | SHIPPED | impl: `pub struct DropReport` in `encoder.rs`; non-test consumer: returned by `load_whisper_encoder` at `ferrotorch-whisper/src/safetensors_loader.rs:35` and propagated up through the `Result` to the caller. |
| REQ-8 | SHIPPED | impl: `named_parameters` + recursive `load_state_dict` for `WhisperEncoder` in `encoder.rs`; non-test consumer: `WhisperEncoder::load_hf_state_dict` in `encoder.rs` calls `self.load_state_dict(&remapped, strict)` after stripping the HF prefix. |
| REQ-9 | SHIPPED | impl: round-trip-tested via `round_trip_state_dict` in `mod tests in encoder.rs`; non-test consumer: `load_whisper_encoder` at `ferrotorch-whisper/src/safetensors_loader.rs:31` is the production path exercising the same load logic against a real `model.safetensors` file (round-tripped through `save_safetensors` in the loader's `mod tests` block). |
