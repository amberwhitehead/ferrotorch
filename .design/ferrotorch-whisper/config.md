# ferrotorch-whisper — `config` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - /home/doll/pytorch/torch/ (foundational ops; HF WhisperConfig at
    huggingface/transformers/src/transformers/models/whisper/configuration_whisper.py
    is the architecture-shape upstream;
    openai/whisper/model.py is the original architecture reference)
-->

## Summary

`ferrotorch-whisper/src/config.rs` carries the typed configuration the
Whisper audio encoder consumes. Mirrors HuggingFace's `WhisperConfig`
JSON schema and exposes both a typed `WhisperConfig` value and a
serde-fed `HfWhisperConfig` for parsing real `config.json` files from
the Hub.

## Requirements

- REQ-1: `pub struct WhisperConfig` — flat hyperparameter record
  covering every field the encoder forward path consumes
  (`num_mel_bins`, `d_model`, `encoder_layers`,
  `encoder_attention_heads`, `encoder_ffn_dim`,
  `max_source_positions`) plus the decoder fields preserved for
  round-trip (`decoder_layers`, `decoder_attention_heads`,
  `decoder_ffn_dim`, `max_target_positions`) and `vocab_size`.
  Marked `Copy`, `Clone`, `PartialEq`, `Debug`.
- REQ-2: `pub struct HfWhisperConfig` with `#[derive(Deserialize)]`
  and `#[non_exhaustive]` parses the full HF JSON schema. Unknown
  fields are ignored; `activation_function` defaults to `"gelu"`.
- REQ-3: `WhisperConfig::validate` enforces invariants the encoder
  relies on (all encoder sizes positive,
  `d_model % encoder_attention_heads == 0`).
- REQ-4: `HfWhisperConfig::validate` enforces architecture
  invariants ferrotorch-whisper implements: only
  `activation_function="gelu"`, only `num_mel_bins == 80`
  (the only filter bank shipped in `audio.rs`).
- REQ-5: `WhisperConfig::from_hf` produces a typed `WhisperConfig`
  from a parsed `HfWhisperConfig`; round-trip preserves all fields.
- REQ-6: `WhisperConfig::whisper_tiny()` exposes the published
  `openai/whisper-tiny` shape (vocab 51865, mel 80, d_model 384,
  encoder 4 layers × 6 heads × ffn 1536, max_source_positions 1500).
- REQ-7: `HfWhisperConfig::from_file` / `from_json_str` cleanly map
  IO and serde errors onto `FerrotorchError::InvalidArgument` with a
  contextual message including the file path.
- REQ-8: `WhisperConfig::encoder_head_dim()` returns
  `d_model / encoder_attention_heads` — a derived quantity the
  encoder attention block reads on construction.
- REQ-9: Decoder fields (`decoder_layers`, `decoder_attention_heads`,
  `decoder_ffn_dim`, `max_target_positions`, `vocab_size`) are
  parsed and preserved so the config round-trips a real upstream
  Whisper checkpoint, even though the encoder forward path does not
  consume them (Phase B.2 = encoder-only).

## Acceptance Criteria

- [x] AC-1: `HfWhisperConfig::from_json_str(TINY_CONFIG)` parses the
  published 51865-vocab Whisper-tiny config.
- [x] AC-2: `WhisperConfig::from_hf` produces the same value as
  `WhisperConfig::whisper_tiny()` when fed the Whisper-tiny JSON.
- [x] AC-3: `WhisperConfig::validate` rejects zero-valued
  `d_model`.
- [x] AC-4: `WhisperConfig::validate` rejects
  `encoder_attention_heads` that does not divide `d_model`.
- [x] AC-5: `HfWhisperConfig::validate` rejects
  `activation_function="relu"`.
- [x] AC-6: `HfWhisperConfig::validate` rejects
  `num_mel_bins=128`.
- [x] AC-7: Unknown JSON fields parse without error.

## Architecture

`pub struct WhisperConfig` in `config.rs` holds eleven encoder-relevant
+ decoder-preserved fields. Marked `Copy` so it can be passed by
value through the `WhisperEncoder::new(cfg: WhisperConfig)` signature
without lifetime gymnastics.

`pub struct HfWhisperConfig` in `config.rs` mirrors the HF JSON
schema. `serde::Deserialize` does the parsing; only
`activation_function` has a serde default (`"gelu"`). The struct is
`#[non_exhaustive]`.

`HfWhisperConfig::validate` in `config.rs` is invoked by `from_hf`
before the typed `WhisperConfig` is constructed; it surfaces
unsupported activation / mel-bin count as `InvalidArgument`. The
typed `WhisperConfig::validate` in `config.rs` runs a shape check
called from every `WhisperConvStem::new` / `WhisperEncoderLayer::new`
constructor.

### Non-test production consumers

- `pub use config::{HfWhisperConfig, WhisperConfig}` at
  `ferrotorch-whisper/src/lib.rs:107`.
- `WhisperConvStem::new(cfg: &WhisperConfig)` at
  `ferrotorch-whisper/src/encoder.rs:49` consumes `WhisperConfig`.
- `WhisperEncoderSelfAttention::new` at
  `ferrotorch-whisper/src/attention.rs:58` consumes `WhisperConfig`.
- `WhisperEncoderLayer::new` at
  `ferrotorch-whisper/src/layer.rs:40` consumes `WhisperConfig`.
- `WhisperEncoder::new` at
  `ferrotorch-whisper/src/encoder.rs:165` consumes `WhisperConfig`
  (by value — the encoder owns a frozen copy).
- `load_whisper_encoder` at
  `ferrotorch-whisper/src/safetensors_loader.rs:31` consumes
  `WhisperConfig`.

## Parity contract

`parity_ops = []`. Configuration parsing has no parity-sweep
counterpart — exercised by `parses_whisper_tiny_config` /
`from_hf_round_trips_tiny` / `unknown_fields_ignored` in the file's
`mod tests` block.

Numerical / structural edge cases preserved:

- **`layer_norm_eps` is NOT a config field.** HF Whisper's
  `config.json` does not carry the eps; ferrotorch-whisper hard-codes
  `1e-5` (the PyTorch default) at every LayerNorm construction site
  in `encoder.rs` / `layer.rs`. Documented in `WhisperEncoderLayer::new`.
- **`activation_function` accepted = `"gelu"` only.** Other
  variants return `InvalidArgument`.
- **`num_mel_bins == 80` only.** Only filter bank shipped in
  `audio.rs/assets/mel_filters_80x201.bin`; 128-mel large-model
  variants are rejected.
- **Decoder fields parsed-not-consumed.** The encoder is Phase B.2;
  the decoder is intentionally out of scope. The fields are still
  preserved in `WhisperConfig` so the config round-trips a real HF
  checkpoint without losing data.

## Verification

Tests in `mod tests in config.rs`:

- `parses_whisper_tiny_config`
- `from_hf_round_trips_tiny`
- `whisper_tiny_is_valid`
- `validate_rejects_zero_fields`
- `validate_rejects_non_divisible_heads`
- `validate_rejects_unsupported_activation`
- `validate_rejects_unsupported_mel_bins`
- `unknown_fields_ignored`

No parity-sweep ops. Smoke command:

```bash
cargo test -p ferrotorch-whisper --lib config:: 2>&1 | tail -3
```

Expected: all tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct WhisperConfig` in `config.rs`; non-test consumer: `pub use` at `config in ferrotorch-whisper/src/lib.rs` plus `WhisperEncoder::new` at `new in ferrotorch-whisper/src/encoder.rs`. |
| REQ-2 | SHIPPED | impl: `pub struct HfWhisperConfig` (with `#[derive(Deserialize)]`) in `config.rs`; non-test consumer: `pub use` at `ferrotorch-whisper/src/lib.rs:107`. |
| REQ-3 | SHIPPED | impl: `WhisperConfig::validate` in `config.rs`; non-test consumer: invoked by `WhisperConvStem::new` at `ferrotorch-whisper/src/encoder.rs:50` and `WhisperEncoderSelfAttention::new` at `ferrotorch-whisper/src/attention.rs:59`. |
| REQ-4 | SHIPPED | impl: `HfWhisperConfig::validate` in `config.rs`; non-test consumer: invoked by `WhisperConfig::from_hf` in `config.rs`. |
| REQ-5 | SHIPPED | impl: `WhisperConfig::from_hf` in `config.rs`; non-test consumer: `pub use` at `ferrotorch-whisper/src/lib.rs:107` (called by Hub-load helpers and pin scripts). |
| REQ-6 | SHIPPED | impl: `WhisperConfig::whisper_tiny` in `config.rs`; non-test consumer: `pub use` at `ferrotorch-whisper/src/lib.rs:107`. |
| REQ-7 | SHIPPED | impl: `HfWhisperConfig::from_file` / `from_json_str` in `config.rs`; non-test consumer: `pub use` at `ferrotorch-whisper/src/lib.rs:107`. |
| REQ-8 | SHIPPED | impl: `WhisperConfig::encoder_head_dim` in `config.rs`; non-test consumer: `WhisperEncoderSelfAttention::new` at `ferrotorch-whisper/src/attention.rs:66` reads `cfg.encoder_head_dim()`. |
| REQ-9 | SHIPPED | impl: decoder fields on `WhisperConfig` + `HfWhisperConfig` in `config.rs`; non-test consumer: `pub use` at `ferrotorch-whisper/src/lib.rs:107` (callers round-trip the full config without losing data). |
