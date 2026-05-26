# ferrotorch-whisper — crate root (`lib`)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - /home/doll/pytorch/torch/ (foundational Module / Linear / LayerNorm
    / Conv1d contracts; HF WhisperModel at
    huggingface/transformers/src/transformers/models/whisper/modeling_whisper.py
    is the architecture-shape upstream; openai/whisper/model.py is
    the original reference)
-->

## Summary

`ferrotorch-whisper/src/lib.rs` is the crate root. It declares the
crate-level lint posture, enumerates the six modules that compose the
Whisper encoder (`attention`, `audio`, `config`, `encoder`, `layer`,
`safetensors_loader`), and re-exports the public types at the crate
root. The decoder (cross-attention, kv-cache, beam search) is out of
scope — Phase B.2 of real-artifact-driven development is encoder-only.

## Requirements

- REQ-1: Crate-level lint posture matches the workspace ML-numerics
  baseline (deny `unsafe_code`, `rust_2018_idioms`,
  `missing_debug_implementations`, `missing_docs`; warn
  `clippy::all` + `clippy::pedantic`; allow a documented list of
  pedantic lints that are wrong for kernel code, including the
  Whisper-specific `clippy::module_name_repetitions` —
  every type name starts with `Whisper` to mirror HF naming).
- REQ-2: Module declarations enumerate every Whisper composition
  file: `attention`, `audio`, `config`, `encoder`, `layer`,
  `safetensors_loader`. Each is `pub` so downstream callers can
  reach the modules directly.
- REQ-3: Public re-exports at the crate root expose the API surface:
  - `attention::WhisperEncoderSelfAttention`
  - `audio::{N_FRAMES, N_MELS, SAMPLE_RATE, log_mel_spectrogram}`
  - `config::{HfWhisperConfig, WhisperConfig}`
  - `encoder::{DropReport, WhisperConvStem, WhisperEncoder}`
  - `layer::WhisperEncoderLayer`
  - `safetensors_loader::load_whisper_encoder`
- REQ-4: Crate-level doc-comment documents:
  - The encoder composition tree (Conv stem → embed_positions →
    `WhisperEncoderLayer × N` → final LayerNorm).
  - The audio preprocessing entry point
    (`audio::log_mel_spectrogram` turns 16 kHz mono `f32` PCM into
    `[1, 80, 3000]`).
  - The HF state-dict loading contract (drops surfaced via
    `encoder::DropReport`).
  - Out-of-scope: decoder + cross-attention + kv-cache + beam search.

## Acceptance Criteria

- [x] AC-1: `cargo check -p ferrotorch-whisper` succeeds with the
  crate-level lints active.
- [x] AC-2: `cargo clippy -p ferrotorch-whisper --lib -- -D warnings`
  succeeds (no new clippy warnings).
- [x] AC-3: The crate-level `pub use` block exposes the documented
  public API.

## Architecture

`ferrotorch-whisper/src/lib.rs:5-44` carries the crate-level lint
posture. Each `#![allow]` line is followed by a `// <reason>` comment
explaining why the lint is wrong for the kernel-code substrate.
`clippy::module_name_repetitions` is allowed because every type
starts with `Whisper` to match HF naming — the lint would force
renames that lose the upstream-1:1 mapping.

`ferrotorch-whisper/src/lib.rs:98-103` declares the six modules.

`ferrotorch-whisper/src/lib.rs:105-110` re-exports the public types.

### Non-test production consumers

The crate-root re-exports are themselves the production consumers
of every type ferrotorch-whisper ships — they form the public API
surface that downstream binaries (Hub-load helpers,
speech-to-text pin scripts) link against. Each `pub use` is the
non-test production consumer cited in the per-module design docs.

## Parity contract

`parity_ops = []`. The crate root has no parity-sweep surface —
its contract is "module enumeration + re-export shape" and is
exercised mechanically by the gauntlet.

## Verification

No `mod tests` block at the crate root. Smoke command:

```bash
cargo check -p ferrotorch-whisper 2>&1 | tail -3
cargo clippy -p ferrotorch-whisper --lib -- -D warnings 2>&1 | tail -3
```

Expected: both succeed.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `#![deny(...)]` / `#![allow(...)]` block at `lib.rs:5-44`; non-test consumer: enforced by every other file in the crate. |
| REQ-2 | SHIPPED | impl: `pub mod` declarations in `lib.rs`; non-test consumer: every test module + every other `.rs` file in the crate uses `crate::<mod>::...` paths. |
| REQ-3 | SHIPPED | impl: `pub use` block at `lib.rs:105-110`; non-test consumer: downstream binaries (Hub-load helpers, pin scripts in `ferrotorch-whisper/tests/`, speech-to-text integrations) import these names directly. |
| REQ-4 | SHIPPED | impl: `//!` doc-comment block at `lib.rs:46-96` (encoder composition tree, audio preprocessing entry point, HF state-dict loading, out-of-scope decoder note); non-test consumer: published via `cargo doc -p ferrotorch-whisper` and visible at the crate root. |
