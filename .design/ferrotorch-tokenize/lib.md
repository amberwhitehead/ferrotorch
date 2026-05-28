# ferrotorch-tokenize — crate root (lib.rs)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - tokenizers crate v0.22.2 (registry: index.crates.io tokenizers-0.22.2/src/tokenizer/mod.rs)
  - transformers/tokenization_utils_base.py (HuggingFace transformers, Python 3.13 site-packages)
  - minijinja 2.x (Jinja2 in Rust, mirrors transformers' jinja2 environment)
-->

## Summary

`ferrotorch-tokenize/src/lib.rs` is a thin wrapper around the HuggingFace
`tokenizers` Rust crate that mirrors the Python surface exposed by
`transformers.AutoTokenizer` — load a `tokenizer.json`, `encode` /
`decode` strings, query vocab size, render Jinja2 chat templates via
`apply_chat_template`. PyTorch itself ships no tokenizer; the upstream
substrate is the HuggingFace stack (`tokenizers` for the algorithms,
`transformers` for the chat-template envelope) that the Llama 3 / BERT /
Whisper / GPT-2 ferrotorch model crates load weights against.

## Requirements

- REQ-1: Lint baseline. `#![warn(clippy::all, clippy::pedantic)]` plus
  `#![deny(rust_2018_idioms, missing_debug_implementations)]`. A single
  documented `#![allow(missing_docs)]` opt-out is allowed (the workspace
  has not yet completed its rustdoc-completeness pass; per-item docs are
  written but `missing_docs` would block compilation prematurely). No
  module-root `#![allow]` is used to silence real clippy lints.

- REQ-2: I/O-vs-parse error categorization (issue #734). `load_tokenizer`
  and `load_chat_template` MUST emit `FerrotorchError::Internal` for
  `std::io::Error` cases (missing file, permission denied) and
  `FerrotorchError::InvalidArgument` for parse / structural-validation
  failures (malformed JSON, unknown tokenizer model type, wrong
  `chat_template` field shape). The two categories cannot be collapsed —
  callers distinguish "your environment is broken" from "you handed us
  bad data".

- REQ-3: Tokenizer surface. `load_tokenizer(path) -> Tokenizer`,
  `encode(&tok, text, add_special_tokens) -> Vec<u32>`,
  `encode_batch(&tok, &[&str], add_special_tokens) -> Vec<Vec<u32>>`,
  `decode(&tok, &[u32], skip_special_tokens) -> String`,
  `vocab_size(&tok, with_added_tokens) -> usize`,
  `token_to_id(&tok, &str) -> Option<u32>`,
  `id_to_token(&tok, u32) -> Option<String>` — each delegating to the
  identically-named `tokenizers::Tokenizer` method, mapping any error
  to `FerrotorchError::InvalidArgument` with a useful message. The
  re-export `pub use tokenizers::Tokenizer;` keeps the upstream type
  reachable so advanced configuration is one method-call away.

- REQ-4: Chat-template rendering (issue #588). `apply_chat_template(template,
  &[ChatMessage], add_generation_prompt, bos_token, eos_token) -> String`
  evaluates a Jinja2 template against the `messages` array using
  `minijinja`, producing the exact string `transformers.AutoTokenizer
  .apply_chat_template` would. The template environment includes two
  helpers — `raise_exception(msg)` (called by Mistral / Llama 3 system
  prompts to abort rendering on invalid input) and `strftime_now(fmt)`
  (called by Llama 3.1's system prompt to inject the current wall-clock
  date). `apply_chat_template_to_ids(...)` is the one-shot
  template-render + encode convenience.

- REQ-5: `ChatMessage` data shape. `pub struct ChatMessage { role: String,
  content: String, extra: BTreeMap<String, serde_json::Value> }` with
  `#[serde(flatten)] extra` so any additional fields (`name`,
  `tool_calls`, `tool_call_id`) propagate to the Jinja context
  unmodified. `#[non_exhaustive]` keeps the struct forward-compatible
  with new optional fields. `ChatMessage::new(role, content)` is the
  canonical constructor for the common case.

- REQ-6: `load_chat_template(tokenizer_config_path) -> Option<String>`
  reads the `chat_template` field out of a HuggingFace
  `tokenizer_config.json`. Accepts both the legacy string form and the
  modern array-of-`{name, template}` form (returning the first array
  entry to match `transformers.AutoTokenizer`'s default), or `None`
  when the file exists but no template is defined. I/O failures stay
  `Internal`; structural failures stay `InvalidArgument` (REQ-2).

## Acceptance Criteria

- [x] AC-1: `cargo check -p ferrotorch-tokenize` is clean.
- [x] AC-2: `cargo test -p ferrotorch-tokenize` passes (17 lib unit tests
  + 7 `conformance_encode_decode` tests + 1 surface-coverage + 1
  surface-inventory + 1 hf-parity-families harness + 1 doctest = 28
  total, with 1 Llama-3-live `#[ignore]` arm and 5 hf-parity-family
  `#[ignore]` arms reserved for HF-download gated environments).
- [x] AC-3: `cargo clippy -p ferrotorch-tokenize -- -D warnings`
  produces zero warnings.
- [x] AC-4: `load_tokenizer` returns `FerrotorchError::Internal` for
  missing files and `FerrotorchError::InvalidArgument` for malformed
  JSON (proved by `loader_rejects_missing_file` and
  `loader_rejects_malformed_json` in `mod tests in lib.rs`).
- [x] AC-5: `load_chat_template` returns `Internal` for missing files
  and `InvalidArgument` for malformed JSON (proved by
  `load_chat_template_categorizes_missing_file_as_internal` and
  `load_chat_template_categorizes_malformed_json_as_invalid_argument`).
- [x] AC-6: `apply_chat_template` reproduces the per-message header +
  EOT envelope plus optional `add_generation_prompt` tail (proved by
  `chat_template_renders_simple_two_turn` /
  `chat_template_appends_generation_prompt_when_requested`).
- [x] AC-7: `strftime_now('%d %b %Y')` produces a structurally valid
  `DD Mon YYYY` and rejects malformed format strings (proved by
  `chat_template_strftime_now_renders_wall_clock_date` /
  `chat_template_strftime_now_rejects_malformed_format_string`).
- [x] AC-8: `raise_exception(msg)` propagates as a `FerrotorchError::
  InvalidArgument` carrying `msg` (proved by
  `chat_template_raise_exception_function_propagates_error`).
- [x] AC-9: `load_chat_template` handles both string and array forms
  (proved by `load_chat_template_extracts_string_field` /
  `load_chat_template_handles_array_form`).
- [x] AC-10: `ChatMessage` round-trips through serde-json with the
  flattened `extra` map (proved by `chat_message_roundtrip_through_serde`).

## Architecture

### Lint baseline (REQ-1)

The crate-root lint header at the top of `lib.rs` sets
`#![warn(clippy::all, clippy::pedantic)]` and
`#![deny(rust_2018_idioms, missing_debug_implementations)]`. A single
`#![allow(missing_docs)]` with an in-source comment explains the opt-out
("workspace-wide rustdoc completeness pass is tracked separately; this
crate's public items are documented but `missing_docs` is not yet
enforced workspace-wide"). No clippy lint is silenced at module root.

### I/O-vs-parse error categories (REQ-2)

`pub fn load_tokenizer in lib.rs` first reads the file via
`std::fs::read_to_string` so the typed `std::io::Error` is observable
at the boundary; that error maps to `FerrotorchError::Internal { message }`
with the path and the debug-format of the io error. The bytes are then
fed to `tokenizers::Tokenizer::from_str` (via `FromStr`), whose error
maps to `FerrotorchError::InvalidArgument { message }`. The split lets
callers distinguish environment failure ("there is no tokenizer.json at
that path") from parameter failure ("the tokenizer.json you gave us is
corrupt").

`pub fn load_chat_template in lib.rs` follows the identical split:
`std::fs::read` (Internal on io error) → `serde_json::from_slice`
(InvalidArgument on parse error) → structural validation of the
`chat_template` field (InvalidArgument when the field exists but is
neither a string nor an array of `{name, template}` objects).

### Tokenizer surface (REQ-3)

The free functions are 1-to-1 wrappers over `tokenizers::Tokenizer`:

- `pub fn encode in lib.rs` calls `tokenizer.encode(text,
  add_special_tokens)` and `.get_ids().to_vec()`. The upstream returns
  an `Encoding` carrying ids + offsets + special-token mask + word ids;
  we project to `Vec<u32>` because that is what ferrotorch model code
  needs.
- `pub fn encode_batch in lib.rs` calls `tokenizer.encode_batch(
  texts.to_vec(), add_special_tokens)`. We pass `&str` slice entries
  directly through the `Into<EncodeInput>` chain so no
  `Vec<String>` intermediate allocation is required.
- `pub fn decode in lib.rs` calls `tokenizer.decode(ids,
  skip_special_tokens)`.
- `pub fn vocab_size in lib.rs` and `pub fn token_to_id in lib.rs` and
  `pub fn id_to_token in lib.rs` are 1-line forwards
  to the upstream methods of the same name.

The `pub use tokenizers::Tokenizer;` re-export keeps the upstream type
reachable so advanced features (added-token manipulation, post-processor
inspection, normalizer config) are accessible without re-deriving them.

Upstream API mirrored from `tokenizers-0.22.2/src/tokenizer/mod.rs`:
- `from_file` / `from_str` (`tokenizers::Tokenizer::from_file` / `FromStr`)
- `get_vocab_size` (`tokenizers::Tokenizer::get_vocab_size`)
- `token_to_id` (`tokenizers::Tokenizer::token_to_id`)
- `id_to_token` (`tokenizers::Tokenizer::id_to_token`)
- `encode` (`tokenizers::Tokenizer::encode`)
- `decode` (`tokenizers::Tokenizer::decode`)
- `encode_batch` (`tokenizers::Tokenizer::encode_batch`)

### Chat-template rendering (REQ-4, REQ-5)

`pub fn apply_chat_template in lib.rs` constructs a
`minijinja::Environment`, registers two helper functions, adds the
template under the name `"chat"`, builds a context with
`messages`, `add_generation_prompt`, `bos_token`, `eos_token`, and
renders. The two helpers reproduce the Jinja-side functions that
upstream `transformers/tokenization_utils_base.py:1547`
(`apply_chat_template`) calls into:

- `raise_exception(msg) -> Result<String, minijinja::Error>` —
  unconditionally returns `Err(InvalidOperation, msg)`. Used by Mistral
  and Llama 3 system prompts to abort rendering when message validation
  fails.
- `strftime_now(fmt) -> Result<String, minijinja::Error>` — parses
  `fmt` via `chrono::format::StrftimeItems`, detects malformed format
  strings by scanning for `Item::Error` (chrono's tombstone variant),
  then formats `chrono::Local::now()` with the parsed items. Used by
  Llama 3.1's system prompt: `{{ strftime_now('%d %b %Y') }}`.

`pub struct ChatMessage in lib.rs` is `#[non_exhaustive]` (forward
compatibility on new optional fields), `Debug + Clone + Serialize +
Deserialize`, with `#[serde(flatten)] extra: BTreeMap<String,
serde_json::Value>` so arbitrary JSON keys flow into the Jinja context
unmodified. `ChatMessage::new(role, content)` constructs the
common-case `{role, content}` message.

`pub fn apply_chat_template_to_ids in lib.rs` chains
`apply_chat_template` then `encode`, returning the rendered prompt
alongside the token ids so callers can log / inspect the prompt that
went into the model.

### tokenizer_config.json loader (REQ-6)

`pub fn load_chat_template in lib.rs`:

1. Reads the file (Internal on io error per REQ-2).
2. Parses with serde_json (InvalidArgument on parse error).
3. Looks at the `chat_template` field:
   - Absent → `Ok(None)`.
   - String → `Ok(Some(s))`.
   - Array of `{name, template}` objects → returns the first
     `template` string. Matches what
     `transformers.AutoTokenizer.from_pretrained` does when a model
     ships multiple named templates (default = first).
   - Any other shape → `Err(InvalidArgument)`.

### Non-test production consumers

The crate's public API is re-exported by the meta-crate at
the meta-crate `pub use ferrotorch_tokenize::*` re-export in `ferrotorch/src/lib.rs`:
```rust
#[cfg(feature = "tokenize")]
pub mod tokenize { pub use ferrotorch_tokenize::*; }
```
This exposes every public item (`load_tokenizer`, `encode`,
`encode_batch`, `decode`, `vocab_size`, `token_to_id`, `id_to_token`,
`apply_chat_template`, `apply_chat_template_to_ids`,
`load_chat_template`, `ChatMessage`, the re-exported `Tokenizer`) at
`ferrotorch::tokenize::*` when the `tokenize` feature is enabled.

Direct (non-meta-crate) non-test production consumers in the workspace
are the four Llama 3 example binaries, each declared as
`[[example]]` Cargo targets in `ferrotorch-llama/Cargo.toml` (built by
`cargo build --release --examples`, run by `cargo run --example ...`,
not gated by `#[cfg(test)]`):

- `ferrotorch-llama/examples/llama3_8b.rs` (the `use ferrotorch_tokenize::{decode, encode, load_tokenizer};` line) —
  `use ferrotorch_tokenize::{decode, encode, load_tokenizer};` then
  `load_tokenizer(&tok_path)`, `encode(&tok, &prompt, true)`,
  `decode(&tok, &[best_id], false)`, `decode(&tok, &tokens, true)`.
- `ferrotorch-llama/examples/llama3_8b_gpu.rs` — identical surface
  on the CUDA path.
- `ferrotorch-llama/examples/llama3_70b_gpu.rs` — identical surface
  on the 70B CUDA path.
- `ferrotorch-llama/examples/llm_inference_dump.rs` — uses
  `load_tokenizer` + `encode` for cross-checking against frozen Python
  token id fixtures.

`apply_chat_template`, `apply_chat_template_to_ids`, `encode_batch`,
`ChatMessage`, and `load_chat_template` are exercised by tests in
`ferrotorch-tokenize/tests/conformance_encode_decode.rs` against live
HuggingFace fixtures (see Verification). They are NOT consumed by any
in-workspace `*/src/*.rs` file today — application code that wants
chat-formatted inference uses them through the meta-crate
`ferrotorch::tokenize::*` boundary. Per goal.md S5 (R-DEFER-1
grandfathering), boundary methods on existing pub API surface ARE the
public API; the meta-crate re-export is the production consumer.

## Parity contract

`parity_ops = []` for this route. The tokenize crate has no
parity-sweep ops — it is not a tensor-op file. PyTorch itself ships no
tokenizer; parity is verified against the HuggingFace `tokenizers` and
`transformers` libraries (not parity-sweep), via the dedicated
fixture-driven conformance harness in
`ferrotorch-tokenize/tests/conformance_hf_parity.rs` which downloads
the `ferrotorch/tokenizer-parity-v1` HF dataset (filed as Phase G.2
fixtures in `ferrotorch-hub/src/registry.rs`).

Edge cases preserved:

- **Missing tokenizer.json file**: `load_tokenizer` returns
  `FerrotorchError::Internal` (REQ-2). NOT `InvalidArgument` —
  the file system is a different failure mode from a bad parameter.
- **Malformed tokenizer.json**: `load_tokenizer` returns
  `FerrotorchError::InvalidArgument` carrying the parse error message
  and the path.
- **Empty messages list with `apply_chat_template`**: rendered output is
  whatever the template produces for an empty `for` loop (typically
  empty string). If the template uses
  `{% if messages | length == 0 %}{{ raise_exception("...") }}{% endif %}`
  (Mistral pattern) the exception propagates as `InvalidArgument`.
- **`strftime_now` with malformed format**: surfaces as
  `FerrotorchError::InvalidArgument` carrying `"strftime_now: invalid
  format string ..."` rather than silently producing a garbled string.
- **`chat_template` field shape mismatch**: a `chat_template` that is
  neither a string nor an array of `{template: string}` records
  produces `InvalidArgument` rather than a panic or silent
  `Ok(None)`.
- **Multi-template `chat_template` array**: returns the first entry's
  `template`, matching upstream `transformers.AutoTokenizer`'s default
  selection. Named-template selection (the second array element's
  `name` for `"tool_use"`, etc.) is NOT exposed at this layer — callers
  who need it parse the JSON themselves and pass the chosen template
  string to `apply_chat_template` directly.
- **`encode_batch` empty batch**: passes through to upstream
  `tokenizers::Tokenizer::encode_batch` which returns an empty `Vec`.
- **`token_to_id` / `id_to_token` for unknown tokens**: returns `None`,
  matching upstream's `Option<u32>` / `Option<String>` semantics.

## Verification

Unit tests in `mod tests in lib.rs` (17 tests):

- `loader_rejects_missing_file` — REQ-2.
- `loader_rejects_malformed_json` — REQ-2.
- `load_chat_template_categorizes_missing_file_as_internal` — REQ-2 / REQ-6.
- `load_chat_template_categorizes_malformed_json_as_invalid_argument`
  — REQ-2 / REQ-6.
- `llama3_tokenizer_loads_and_round_trips` (`#[ignore]`, REQ-3 live arm)
  — gated on Meta-Llama-3-8B presence in HF cache.
- `chat_template_renders_simple_two_turn` — REQ-4.
- `chat_template_appends_generation_prompt_when_requested` — REQ-4.
- `chat_template_trims_whitespace_in_content` — REQ-4 (Jinja `| trim`).
- `chat_template_passes_bos_token` — REQ-4.
- `chat_template_propagates_extra_fields` — REQ-5.
- `chat_template_rejects_invalid_template` — REQ-4.
- `chat_template_strftime_now_renders_wall_clock_date` — REQ-4.
- `chat_template_strftime_now_rejects_malformed_format_string` — REQ-4.
- `chat_template_raise_exception_function_propagates_error` — REQ-4.
- `load_chat_template_extracts_string_field` — REQ-6.
- `load_chat_template_handles_array_form` — REQ-6.
- `load_chat_template_returns_none_when_missing` — REQ-6.
- `chat_message_roundtrip_through_serde` — REQ-5.

Integration tests in `ferrotorch-tokenize/tests/`:
- `conformance_encode_decode.rs` (7 tests) drives the public surface
  through serialized HuggingFace tokenizer fixtures
  (`tests/conformance/fixtures/`). Includes
  `vocab_size_matches_python_reference`,
  `encode_matches_python_reference_per_case`,
  `decode_matches_python_reference_per_case`,
  `encode_batch_matches_per_input_encode`,
  `token_to_id_resolves_known_special_tokens`,
  `chat_template_round_trip_matches_minijinja`,
  `load_chat_template_round_trips_through_disk`.
- `conformance_hf_parity.rs` — 5 `#[ignore]` per-family arms
  (`hf_parity_llama3`, `_clip`, `_bert`, `_gpt2`, `_smollm`) +
  1 always-on completeness check `families_constant_is_complete`. The
  ignored arms download `ferrotorch/tokenizer-parity-v1` from
  HuggingFace and assert encode / decode / chat-template parity
  byte-for-byte against pre-computed Python references.
- `conformance_surface_coverage.rs` (1 test) — gates the public API
  surface against `tests/conformance/_surface.json` so new pub items
  must be enumerated explicitly.
- `conformance_surface_inventory.rs` (1 test) — regenerates the
  inventory under `BLESS=1`.

Doctest in `lib.rs` (the crate-root `# Quick start` block) — `no_run` example compiles
under `cargo test --doc -p ferrotorch-tokenize`.

Smoke command (no parity ops):

```bash
cargo check -p ferrotorch-tokenize 2>&1 | tail -3
cargo clippy -p ferrotorch-tokenize -- -D warnings 2>&1 | tail -3
cargo test -p ferrotorch-tokenize 2>&1 | grep "test result:"
```

Expected: `Finished dev profile`, no clippy warnings, six
`test result: ok` lines (17 + 7 + 1 + 1 + 1 + 1 passed; 1 + 5 + 0 + 0 +
0 + 0 ignored).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: lint baseline at top of `lib.rs` — `#![warn(clippy::all, clippy::pedantic)]` + `#![deny(rust_2018_idioms, missing_debug_implementations)]` + documented `#![allow(missing_docs)]`; non-test consumer: the meta-crate `pub use ferrotorch_tokenize::*` re-export in `ferrotorch/src/lib.rs` `pub use ferrotorch_tokenize::*;` (the feature-gated re-export compiles under this baseline); verified by `cargo clippy -p ferrotorch-tokenize -- -D warnings` clean. |
| REQ-2 | SHIPPED | impl: `pub fn load_tokenizer in lib.rs` splits `std::fs::read_to_string` (`Internal`) from `Tokenizer::from_str` (`InvalidArgument`); `pub fn load_chat_template in lib.rs` splits `std::fs::read` from `serde_json::from_slice` identically; non-test consumer: `ferrotorch-llama/examples/llama3_8b.rs` (greedy-decode `load_tokenizer` call site) and `:42` calls `load_tokenizer(&tok_path).expect("failed to load tokenizer.json")` through the meta-crate boundary — the error type chosen here flows through to the user's terminal; verified by `loader_rejects_missing_file`, `loader_rejects_malformed_json`, `load_chat_template_categorizes_missing_file_as_internal`, `load_chat_template_categorizes_malformed_json_as_invalid_argument` in `mod tests in lib.rs` (issue #734). |
| REQ-3 | SHIPPED | impl: `pub fn load_tokenizer in lib.rs`, `pub fn encode in lib.rs`, `pub fn encode_batch in lib.rs`, `pub fn decode in lib.rs`, `pub fn vocab_size in lib.rs`, `pub fn token_to_id in lib.rs`, `pub fn id_to_token in lib.rs` plus `pub use tokenizers::Tokenizer in lib.rs` mirror the identically-named methods on `tokenizers::Tokenizer` (upstream crate registry path `tokenizers-0.22.2/src/tokenizer/mod.rs`, methods `from_file` / `get_vocab_size` / `token_to_id` / `id_to_token` / `encode` / `decode` / `encode_batch`); non-test consumers: `ferrotorch-llama/examples/llama3_8b.rs` (the example's `use ferrotorch_tokenize::{decode, encode, load_tokenizer};` plus `load_tokenizer` + `encode` + two `decode` calls in the greedy-decode loop), `ferrotorch-llama/examples/llama3_8b_gpu.rs` (CUDA path: identical surface — `load_tokenizer` + `encode` + decode calls), `ferrotorch-llama/examples/llama3_70b_gpu.rs` (70B CUDA path: identical surface), `ferrotorch-llama/examples/llm_inference_dump.rs` (`load_tokenizer` + `encode` for cross-checking against frozen Python token id fixtures) — all four are `[[example]]` Cargo binary targets (NOT `#[cfg(test)]`). Meta-crate boundary at the meta-crate `pub use ferrotorch_tokenize::*` re-export in `ferrotorch/src/lib.rs`. Verified by `conformance_encode_decode::vocab_size_matches_python_reference / encode_matches_python_reference_per_case / decode_matches_python_reference_per_case / encode_batch_matches_per_input_encode / token_to_id_resolves_known_special_tokens`. |
| REQ-4 | SHIPPED | impl: `pub fn apply_chat_template in lib.rs` constructs a `minijinja::Environment`, registers `raise_exception` and `strftime_now` helpers, and renders against the messages array, mirroring `transformers/tokenization_utils_base.py:1547`; `pub fn apply_chat_template_to_ids in lib.rs` chains it through `encode`; non-test consumer: the meta-crate `pub use ferrotorch_tokenize::*` re-export in `ferrotorch/src/lib.rs` `pub use ferrotorch_tokenize::*;` propagates both functions to `ferrotorch::tokenize::apply_chat_template` (boundary surface per goal.md S5 R-DEFER-1 grandfathering: existing pub APIs across multiple prior commits — landed in #588 + #734 — ARE the public API and don't need additional in-tree callers). Verified by `chat_template_renders_simple_two_turn`, `chat_template_appends_generation_prompt_when_requested`, `chat_template_trims_whitespace_in_content`, `chat_template_passes_bos_token`, `chat_template_rejects_invalid_template`, `chat_template_strftime_now_renders_wall_clock_date`, `chat_template_strftime_now_rejects_malformed_format_string`, `chat_template_raise_exception_function_propagates_error`, plus `conformance_encode_decode::chat_template_round_trip_matches_minijinja`. |
| REQ-5 | SHIPPED | impl: `#[non_exhaustive] pub struct ChatMessage in lib.rs` with `#[serde(flatten)] extra: BTreeMap<String, serde_json::Value>` and `impl ChatMessage::new in lib.rs`; non-test consumer: the meta-crate `pub use ferrotorch_tokenize::*` re-export in `ferrotorch/src/lib.rs` re-exports `ChatMessage` via the glob, making `ferrotorch::tokenize::ChatMessage` the boundary surface (existing pub API across #588 — grandfathered under S5). Verified by `chat_template_propagates_extra_fields` and `chat_message_roundtrip_through_serde`. |
| REQ-6 | SHIPPED | impl: `pub fn load_chat_template in lib.rs` reads bytes (Internal on io error), parses JSON (InvalidArgument on parse error), and dispatches on the `chat_template` field shape (None / String / Array of `{template: string}` / other → InvalidArgument); non-test consumer: the meta-crate `pub use ferrotorch_tokenize::*` re-export in `ferrotorch/src/lib.rs` re-export makes `ferrotorch::tokenize::load_chat_template` the boundary surface (existing pub API across #588 / #734 — grandfathered under S5). Verified by `load_chat_template_extracts_string_field`, `load_chat_template_handles_array_form`, `load_chat_template_returns_none_when_missing`, plus the issue-#734 error-category arms in REQ-2's evidence, plus `conformance_encode_decode::load_chat_template_round_trips_through_disk`. |
