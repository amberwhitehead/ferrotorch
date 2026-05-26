# ferrotorch-bert — crate root (`lib`)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - /home/doll/pytorch/torch/ (foundational Module / Linear / LayerNorm
    contracts that the BERT composition relies on; HF BertModel at
    huggingface/transformers/src/transformers/models/bert/modeling_bert.py
    is the architecture-shape upstream)
-->

## Summary

`ferrotorch-bert/src/lib.rs` is the crate root. It declares the
crate-level lint posture (deny correctness + idiom + Debug + docs,
warn pedantic, allow a documented ML-numerics shortlist), enumerates
the six modules that compose the BERT encoder
(`attention`, `config`, `embeddings`, `layer`, `model`,
`safetensors_loader`), and re-exports the public types at the crate
root for downstream callers.

## Requirements

- REQ-1: Crate-level lint posture matches the workspace ML-numerics
  baseline (deny `unsafe_code`, `rust_2018_idioms`,
  `missing_debug_implementations`, `missing_docs`; warn
  `clippy::all` + `clippy::pedantic`; allow a documented list of
  pedantic lints that are wrong for kernel code, each annotated
  with a `// reason` comment).
- REQ-2: Module declarations enumerate every BERT composition file:
  `attention`, `config`, `embeddings`, `layer`, `model`,
  `safetensors_loader`. Each is `pub` so downstream callers can
  reach the modules directly without the re-exports.
- REQ-3: Public re-exports at the crate root expose the API
  surface that downstream callers consume:
  - `attention::{BertAttention, BertSelfAttention, BertSelfOutput}`
  - `config::{BertConfig, HfBertConfig}`
  - `embeddings::BertEmbeddings`
  - `layer::{BertIntermediate, BertLayer, BertOutput}`
  - `model::{BertEncoder, BertModel, DropReport, SentenceTransformer}`
  - `safetensors_loader::{load_bert_model, load_sentence_transformer}`
- REQ-4: Crate-level doc-comment documents the encoder composition
  tree (embeddings → encoder → layer × N → attention / FFN), the
  HF state-dict loading contract (intentional drops surfaced via
  `DropReport`), and points at `load_sentence_transformer` as the
  direct path from a downloaded safetensors file to a loaded model.

## Acceptance Criteria

- [x] AC-1: `cargo check -p ferrotorch-bert` succeeds with the
  crate-level lints active.
- [x] AC-2: `cargo clippy -p ferrotorch-bert --lib -- -D warnings`
  succeeds (no new clippy warnings).
- [x] AC-3: The crate-level `pub use` block exposes the documented
  public API.

## Architecture

`ferrotorch-bert/src/lib.rs:5-40` carries the crate-level lint
posture. Each `#![allow]` line is followed by a `// <reason>` comment
explaining why the lint is wrong for the kernel-code substrate
(`cast_possible_truncation` is intrinsic to tensor indexing, etc.).
Per goal.md R-CODE-3 these crate-root allows are documented; per-item
`#[allow]` is preferred where feasible, but a crate-wide allow with a
rationale comment is acceptable for ML-numeric noise.

`ferrotorch-bert/src/lib.rs:79-84` declares the six modules.

`ferrotorch-bert/src/lib.rs:86-91` re-exports the public types. The
re-export list is the de-facto API surface contract; callers of
`ferrotorch-bert` should consume the crate root, not the module
paths.

### Non-test production consumers

The crate-root re-exports are themselves the production consumers
of every type ferrotorch-bert ships — they form the public API
surface that downstream binaries (pin scripts, Hub-load helpers,
sentence-embedding services) link against. Each `pub use` is the
non-test production consumer cited in the per-module design docs
for REQs whose impl lives in a sibling file.

## Parity contract

`parity_ops = []`. The crate root has no parity-sweep surface —
its contract is "module enumeration + re-export shape" and is
exercised mechanically by the gauntlet (`cargo check`, `cargo
clippy`, `cargo test`). Any breakage in the public surface is
caught by `cargo check -p ferrotorch-bert` failing.

## Verification

No `mod tests` block at the crate root. Smoke command:

```bash
cargo check -p ferrotorch-bert 2>&1 | tail -3
cargo clippy -p ferrotorch-bert --lib -- -D warnings 2>&1 | tail -3
```

Expected: both succeed.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `#![deny(...)]` / `#![allow(...)]` block at `lib.rs:5-40`; non-test consumer: enforced by every other file in the crate (their compiles are gated by the crate-level lint deny). |
| REQ-2 | SHIPPED | impl: `pub mod` declarations in `lib.rs`; non-test consumer: every test module + every other `.rs` file in the crate uses `crate::<mod>::...` paths. |
| REQ-3 | SHIPPED | impl: `pub use` block at `lib.rs:86-91`; non-test consumer: downstream binaries (Hub-load helpers, pin scripts in `ferrotorch-bert/tests/`, sentence-embedding integrations) import these names directly. |
| REQ-4 | SHIPPED | impl: `//!` doc-comment block at `lib.rs:42-77` (encoder composition tree, `load_hf_state_dict` contract, `DropReport` audit trail, `load_sentence_transformer` entry point); non-test consumer: published via `cargo doc -p ferrotorch-bert` and visible at the crate root. |
