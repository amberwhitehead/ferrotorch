# ferrotorch-whisper — `safetensors_loader` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - /home/doll/pytorch/torch/ (the safetensors crate is the on-disk
    spec; HF Whisper conventions provide the key layout the loader
    expects)
-->

## Summary

`ferrotorch-whisper/src/safetensors_loader.rs` turns a path-to-safetensors
on disk into a loaded `WhisperEncoder` plus a `DropReport` recording
which non-encoder keys were dropped. The HF Whisper safetensors
carries both encoder and decoder weights; this loader filters in only
`encoder.*` (or `model.encoder.*`) and rejects everything else with a
`DropReport` entry so the pin script can audit.

## Requirements

- REQ-1: `load_whisper_encoder<T: Float>(weights_path, cfg, strict)`
  loads the safetensors at `weights_path` via
  `ferrotorch_serialize::load_safetensors`, constructs
  `WhisperEncoder::<T>::new(cfg)`, and calls
  `encoder.load_hf_state_dict(&state, strict)` to populate. Returns
  `(WhisperEncoder<T>, DropReport)`.
- REQ-2: `strict=false` is required for full Whisper checkpoints
  (they ship decoder + `proj_out` weights this encoder-only loader
  has no slot for). The `DropReport` captures every dropped key.
- REQ-3: All IO / parse / decode errors map onto
  `FerrotorchError::InvalidArgument` with a contextual message
  including the offending path.

## Acceptance Criteria

- [x] AC-1: Round-trip — `save_safetensors(&encoder_with_encoder_prefix,
  &path)` followed by `load_whisper_encoder::<f32>(&path, tiny_cfg,
  false)` returns an encoder whose `forward_from_mel` matches the
  source within `1e-6`.
- [x] AC-2: The `DropReport` from AC-1 has empty `dropped` (the
  round-tripped state dict has no non-encoder keys).

## Architecture

`load_whisper_encoder` in `safetensors_loader.rs` is a thin wrapper:

1. `load_safetensors::<T>(weights_path)` to decode the typed
   `StateDict<T>` (any IO / parse / decode error is mapped onto
   `FerrotorchError::InvalidArgument` with the file path).
2. `WhisperEncoder::<T>::new(cfg)?` to build the empty encoder.
3. `encoder.load_hf_state_dict(&state, strict)` to populate the
   parameters from the state dict (strips the `encoder.` or
   `model.encoder.` prefix; records anything else in the
   `DropReport`).
4. Return `(encoder, report)`.

This is intentionally simpler than the BERT counterpart because the
HF Whisper safetensors does not carry an int64 buffer that would
break the f32-typed decoder; the entire upstream state dict decodes
through `load_safetensors::<f32>` cleanly.

### Non-test production consumers

- `pub use safetensors_loader::load_whisper_encoder` at
  `ferrotorch-whisper/src/lib.rs:110`.
- `load_whisper_encoder` is the canonical public entry point for
  loading a real Hub checkpoint into a `WhisperEncoder`; it is the
  ferrotorch-whisper API surface consumed by speech-to-text pin
  scripts.

## Parity contract

`parity_ops = []`. The loader composes
`ferrotorch_serialize::load_safetensors` (covered by the
`ferrotorch-serialize` parity surface) and
`WhisperEncoder::load_hf_state_dict` (covered by `encoder.md`).

Numerical / structural edge cases preserved:

- **Full Whisper checkpoints contain decoder + `proj_out`.** Strict
  mode rejects these; non-strict drops them and records in
  `DropReport.dropped`. Real Hub checkpoints (e.g.
  `openai/whisper-tiny`) require `strict=false`.
- **Both `encoder.<rest>` and `model.encoder.<rest>` prefixes are
  accepted.** Raw encoder checkpoints use the former; full
  conditional-generation checkpoints use the latter. The prefix
  strip is in `WhisperEncoder::load_hf_state_dict`.

## Verification

Tests in `mod tests in safetensors_loader.rs`:

- `round_trip_safetensors_into_encoder`

No parity-sweep ops. Smoke command:

```bash
cargo test -p ferrotorch-whisper --lib safetensors_loader:: 2>&1 | tail -3
```

Expected: all tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn load_whisper_encoder` in `safetensors_loader.rs`; non-test consumer: `pub use` at `ferrotorch-whisper/src/lib.rs:110` (the canonical Hub-load entry point used by integration tests + pin scripts). |
| REQ-2 | SHIPPED | impl: `strict: bool` parameter forwarded to `encoder.load_hf_state_dict(&state, strict)` in `safetensors_loader.rs`; non-test consumer: same `pub use` at `ferrotorch-whisper/src/lib.rs:110` — callers pass `strict=false` for full Whisper checkpoints. |
| REQ-3 | SHIPPED | impl: `.map_err(\|e\| FerrotorchError::InvalidArgument { message: format!("load_whisper_encoder: failed to decode safetensors {}: {e}", ...) })` in `safetensors_loader.rs`; non-test consumer: error type surfaces through the `pub use` at `ferrotorch-whisper/src/lib.rs:110`. |
