# ferrotorch-hub — `download` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/hub.py
-->

## Summary

`ferrotorch-hub/src/download.rs` is the network-side of the hub. It
mirrors `torch.hub.download_url_to_file` (`torch/hub.py:707`) +
`torch.hub.load_state_dict_from_url` (`torch/hub.py:829`) for the
single-file weights case, and adds a HuggingFace-shaped
`hf_download_model` for the sharded-checkpoint case (multi-shard
safetensors + `config.json` + tokenizer files), which has no exact
counterpart in `torch.hub` (HF Python's `huggingface_hub.snapshot_download`
is the conceptual analog). The verification policy is fail-fast on
the all-zero SHA-256 placeholder (audit #6) — silent skip on
placeholder was the original-shipped behaviour and the audit found it
unsafe. Path-traversal guards (`sanitize_path_component`,
`assert_within_cache`) defend against malicious server responses that
inject `../` into shard filenames returned in
`model.safetensors.index.json`.

## Requirements

- REQ-1: `pub fn download_weights(info: &ModelInfo, cache: &HubCache)
  -> FerrotorchResult<PathBuf>` resolves a model's weights file. If
  the canonical cached path (`cache.path_for_model(info)`) exists,
  return it. If the legacy bare-name path (`cache.path(info.name)`)
  exists, return that. Otherwise, with `http` feature: download +
  verify + cache. Without `http`: return a clear error with manual
  download instructions.
- REQ-2: `pub fn load_pretrained<T: Float>(name: &str)
  -> FerrotorchResult<StateDict<T>>` is the user-facing top-level
  entry point. Looks up `name` in the registry, calls
  `download_weights`, then dispatches by `WeightsFormat` to
  `ferrotorch_serialize::load_safetensors` or
  `ferrotorch_serialize::load_state_dict`. Unknown model name →
  `InvalidArgument` with a `list_models()` suggestion in the message.
- REQ-3: `download_and_verify(info, cache)` (http-only) does the
  bytes-on-the-wire work: GET via `ureq` with auth header injection,
  4 GB cap on the body, SHA-256 over the downloaded bytes,
  byte-for-byte compare against `info.weights_sha256`, write to
  cache via `cache.store(canonical_filename, &body)`. Returns the
  cache path on success.
- REQ-4: Fail-fast on the all-zero SHA-256 placeholder. A registry
  entry whose `weights_sha256` is `"0".repeat(64)` triggers an
  immediate `InvalidArgument` error with a message naming the model
  and pointing at follow-up issue #739. No bytes are fetched.
  Documented as security audit finding #6.
- REQ-5: `pub fn hf_download_model(repo: &str, revision: &str,
  cache: &HubCache) -> FerrotorchResult<PathBuf>` (http-only)
  downloads a full HuggingFace repo into the cache: `config.json`,
  every shard from `model.safetensors.index.json` (or
  `model.safetensors` if no index), and a best-effort attempt at
  `tokenizer.json` / `tokenizer_config.json` / `special_tokens_map.json`.
  Auth header is injected on every request. Returns the cache
  directory containing the repo.
- REQ-6: Caller-input sanitisation. `repo` is split on `/` and every
  segment validated via `sanitize_path_component`; empty `repo` is
  rejected. Empty `revision` defaults to `"main"`; non-empty
  `revision` is validated via `sanitize_path_component`.
- REQ-7: Server-input sanitisation. Every shard filename parsed from
  the server-supplied `model.safetensors.index.json` is validated
  via `sanitize_path_component` BEFORE being inserted into the
  download set. Empty shard list is `InvalidArgument`.
- REQ-8: Defense-in-depth path containment. `assert_within_cache`
  verifies the final resolved cache path is within `cache.cache_dir()`
  before any write. Tries `std::fs::canonicalize` on both ends
  (strongest, resolves symlinks). Falls back to lexical normalisation
  of the file path against a canonicalized cache dir when the file
  doesn't yet exist (the common case at write time). A failed
  containment check is `InvalidArgument` and the write is refused.
- REQ-9: Tokenizer best-effort persistence. Each tokenizer file
  fetch_optional result returning `Some(body)` is `cache.store(...)`'d
  after `assert_within_cache`. The previous regression (#1147)
  discarded the body via `let _ = fetch_optional(...)`; this REQ
  captures the fix.

## Acceptance Criteria

- [x] AC-1: `download_weights` returns the cached path when the
  canonical file exists (`test_download_weights_returns_path_when_cached`).
- [x] AC-2: `download_weights` finds the legacy bare-name location
  (`test_download_weights_finds_bare_name`).
- [x] AC-3: `load_pretrained` on an unknown model errors with
  "Unknown model" + `list_models` suggestion
  (`test_load_pretrained_unknown_model`).
- [x] AC-4: `hex_lower` produces lowercase hex
  (`test_hex_lower_known_vector`).
- [x] AC-5: `canonical_filename` picks the right extension
  (`test_canonical_filename_safetensors`,
  `test_canonical_filename_fts`).
- [x] AC-6: SHA-256 placeholder is detected against the registry
  (`test_sha256_placeholder_detected`).
- [x] AC-7: SHA-256 hashing matches a known empty-string vector
  (`test_sha256_mismatch_against_known_value`).
- [x] AC-8: `sanitize_path_component` rejects every attack vector:
  empty, `..`, `.`, `/x`, `x/y`, `x\\y`, `\0`, leading `.`, `:`,
  overlong (`sanitize_rejects_*` — 13 negative tests).
- [x] AC-9: `sanitize_path_component` accepts normal shard
  filenames, revision strings, and max-length strings
  (`sanitize_accepts_*` — 3 positive tests).
- [x] AC-10: `assert_within_cache` rejects escaped paths
  (`within_cache_rejects_escaped_path`) and accepts legitimate
  sub-paths (`within_cache_accepts_subpath`).

## Architecture

### Single-file download path (REQ-1, REQ-2, REQ-3, REQ-4)

`pub fn download_weights`:

1. `cache.path_for_model(info).exists()` → return.
2. `cache.path(info.name).exists()` (legacy bare-name layout) → return.
3. With `http`: `download_and_verify(info, cache)`.
4. Without `http`: `InvalidArgument` with the URL + canonical path
   embedded.

`pub fn load_pretrained<T: Float>(name)`:

1. `get_model_info(name)` lookup; `None` → `InvalidArgument` with
   "Unknown model '{name}'. Use ferrotorch_hub::list_models() to see
   available models."
2. `HubCache::with_default_dir()`.
3. `download_weights(info, &cache)`.
4. Dispatch by `info.format` → `ferrotorch_serialize::load_safetensors`
   or `ferrotorch_serialize::load_state_dict`.

`download_and_verify(info, cache)` (http-only, private):

1. `is_placeholder_sha256(info.weights_sha256)` → fail-fast with
   "registry SHA-256 ... is the all-zero placeholder; refusing to
   download without integrity verification." (audit #6)
2. `crate::auth::with_auth(ureq::get(info.weights_url)).call()`.
3. `response.into_reader().take(MAX_BODY_BYTES = 4 GB).read_to_end(&mut body)`.
4. `sha2::Sha256::new() + update(&body) + finalize()` → hex string.
5. Compare against `info.weights_sha256.to_lowercase()`; mismatch
   → `InvalidArgument` with both digests in the message.
6. `cache.store(&canonical_filename(info), &body)?`.
7. Return `cache.path(&canonical_filename)`.

`is_placeholder_sha256(s)` checks for 64 ASCII `'0'` bytes.
`canonical_filename(info)` mirrors `HubCache::path_for_model`'s
extension dispatch. `hex_lower(&[u8])` is a hand-written 2-hex-nibble
encoder. `hex_nibble(n)` handles the 0-15 range via the `0-9`/`a-f`
ASCII offset.

### Sharded HF download (REQ-5, REQ-6, REQ-7, REQ-8, REQ-9)

`pub fn hf_download_model(repo, revision, cache)` (http-only):

1. Split `repo` on `/`. Reject empty. Validate each segment via
   `sanitize_path_component`.
2. `revision = if empty { "main" } else { revision }`. Validate.
3. Helper `fetch_one(repo, revision, relative, cache)`:
   - URL: `https://huggingface.co/{repo}/resolve/{revision}/{relative}`
   - Auth-decorate, GET, body cap 16 GB.
   - `cache_name = format!("{repo}/{relative}")`.
   - `assert_within_cache(cache.cache_dir(), &cache.path(&cache_name))`
   - `cache.store(&cache_name, &body)`.
4. Helper `fetch_optional(repo, revision, relative)`:
   - Same URL build + auth decoration.
   - `Ok(response)` → `Some(body)`.
   - `Err(ureq::Error::Status(404, _))` → `None`.
   - Other `Err` → `InvalidArgument`.
5. `fetch_one(repo, revision, "config.json", cache)` — required.
6. `fetch_optional("model.safetensors.index.json")`:
   - `Some(index_bytes)` → parse `serde_json::Value`, persist the
     index file via `cache.store`, enumerate `weight_map` values.
     For each shard filename, run `sanitize_path_component(s,
     "shard filename")` BEFORE inserting into the BTreeSet. Reject
     empty set as `InvalidArgument`. For each shard,
     `fetch_one(repo, revision, shard, cache)`.
   - `None` → single-file model; `fetch_one(repo, revision,
     "model.safetensors", cache)`.
7. Best-effort tokenizer files (`tokenizer.json`,
   `tokenizer_config.json`, `special_tokens_map.json`): each via
   `fetch_optional` → on `Some(body)`, `assert_within_cache` +
   `cache.store`.
8. Return `cache.path(repo)`.

### `sanitize_path_component(s, role)` (REQ-6, REQ-7)

The protocol-boundary defense. Rejects, in order:

1. empty string,
2. > 255 bytes (POSIX `NAME_MAX`),
3. `\0` byte,
4. `/` or `\` separator,
5. `:` (Windows ADS / illegal),
6. literal `..` or `.`,
7. leading `.` (any hidden-file pattern).

Acceptance test: typical shard filenames like
`model-00001-of-00004.safetensors` and revisions like `main` /
`v1.0` / `abc123def456` / `feature-branch` all pass. The hex-digest
branch SHAs are alphanumeric, so the validation is happy.

Note: `sanitize_path_component` is STRICTER than
`crate::cache::validate_cache_relative`. The first runs on individual
filename segments before they are joined with `repo/...`; the second
runs at the cache-write boundary on the joined relative path
(which legitimately contains `/`). Both layers fire on every shard
download.

### `assert_within_cache(base, full_path)` (REQ-8)

Two strategies in order:

1. **Strong**: both `base` and `full_path` exist on disk →
   `fs::canonicalize` both, check `canonical_path.starts_with(
   canonical_base)`. Resolves symlinks so a symlink inside the cache
   pointing outside cannot escape.
2. **Fallback** (file doesn't exist yet — common at write time):
   `fs::canonicalize(base)` (which DOES exist) and lexically
   normalise `full_path` via `lexical_normalize`. Then
   `path_norm.starts_with(base_norm)`.

`lexical_normalize` is a `Component`-walker that collapses `.` (skip)
and `..` (pop). A `..` at the root underflows the stack and is
silently dropped — safe because the subsequent `starts_with` check
catches the escape.

A failed containment check is `InvalidArgument` and the write is
refused — `fetch_one` and the tokenizer loop both call this before
`cache.store`.

### Non-test production consumers

- `pub use download::{download_weights, load_pretrained}` and
  `pub use download::hf_download_model` (http-gated) in `lib.rs`
  flatten the surface.
- `ferrotorch-jit/examples/jit_trace_dump.rs` calls
  `load_pretrained` directly to fetch `jit-trace-parity-v1` weights
  for the trace dump.
- `ferrotorch-llama/examples/llm_inference_dump.rs`,
  `ferrotorch-bert/examples/text_embedding_dump.rs`,
  `ferrotorch-diffusion/examples/{clip_text_encode_dump,
  vae_decode_dump, unet_predict_dump, unet_probe_dump,
  sd_pipeline_dump}.rs`,
  `ferrotorch-graph/examples/gcn_inference_dump.rs`,
  `ferrotorch-rl/examples/ppo_policy_dump.rs` all call
  `hf_download_model`. (Examples are production binaries — they
  compile under `cargo build --examples`, not under `--tests`.)

## Parity contract

`parity_ops = []`. The contract is HTTP/safetensors/SHA-256, not a
PyTorch numerical op.

Upstream-derived behaviour:

- `download_url_to_file` (`torch/hub.py:707`) parallel:
  `download_and_verify`. Both stream the body, compute SHA-256, and
  compare. ferrotorch's version FAILS FAST on the placeholder
  digest (the audit's #6 deviation, R-DEV-6 — upstream behavior was
  unsafe by their own admission once flagged).
- `load_state_dict_from_url` (`torch/hub.py:829`) parallel:
  `load_pretrained`. Both check for cached file presence before
  downloading. ferrotorch dispatches to safetensors / fts loaders;
  upstream falls back to `_legacy_zip_load` for legacy `.pth.zip`
  format (we do not mirror that legacy path; `ferrotorch-serialize`
  has a separate `pytorch_export` module for `.pth` cases).

HuggingFace Hub-specific behaviour with no `torch.hub` counterpart:

- `hf_download_model` mirrors the responsibilities of
  `huggingface_hub.snapshot_download` (the Python HF client), which
  downloads every file in a repo. ferrotorch's version downloads
  the safetensors shards + config + tokenizer files (the LLM
  inference subset) rather than every file in the repo.

Edge cases:

- Cached file present → skip download, return cached path.
- Legacy bare-name layout → also accepted for backwards-compat.
- All-zero SHA-256 placeholder → fail-fast (audit #6).
- Real SHA-256 mismatch → `InvalidArgument`; cache is not written.
- Body exceeds 4 GB (single weights) or 16 GB (sharded): `take(...)`
  truncates and `read_to_end` returns the truncated content; the
  SHA check would then fail.
- Server returns 404 for tokenizer files: treated as expected,
  silently skipped.
- Server returns 404 for `model.safetensors.index.json`: fall back
  to single-file `model.safetensors`.
- Malicious server returns `../../.bashrc` in `weight_map`:
  `sanitize_path_component` rejects at parse time before insertion.
- TOCTOU between sanitisation and write: `assert_within_cache`
  re-validates after path join, using `canonicalize` when possible
  (symlinks resolved).
- Unicode-normalisation attack on shard filename: `assert_within_cache`'s
  canonicalize path catches it; the lexical-normalize fallback
  catches the rest before the write.

## Verification

Tests in `mod tests in download.rs` (28 tests; some `#[cfg(feature = "http")]`-gated):

- Cache-resolution + happy paths: `test_download_weights_returns_path_when_cached`,
  `test_download_weights_finds_bare_name`.
- Top-level entry: `test_load_pretrained_unknown_model`.
- Hex / SHA helpers: `test_hex_lower_known_vector`,
  `test_canonical_filename_safetensors`,
  `test_canonical_filename_fts`,
  `test_sha256_placeholder_detected`,
  `test_sha256_mismatch_against_known_value`.
- `sanitize_path_component` rejections: `sanitize_rejects_empty_string`,
  `sanitize_rejects_dotdot`, `sanitize_rejects_single_dot`,
  `sanitize_rejects_leading_slash`, `sanitize_rejects_embedded_slash`,
  `sanitize_rejects_backslash`, `sanitize_rejects_null_byte`,
  `sanitize_rejects_dotdot_embedded`,
  `sanitize_rejects_dotdot_at_start_with_slash`,
  `sanitize_rejects_leading_dot`, `sanitize_rejects_colon`,
  `sanitize_rejects_windows_ads`,
  `sanitize_rejects_overlong_string`.
- `sanitize_path_component` acceptances: `sanitize_accepts_normal_shard_filename`,
  `sanitize_accepts_normal_revision`,
  `sanitize_accepts_max_length_string`.
- `assert_within_cache`: `within_cache_rejects_escaped_path`,
  `within_cache_accepts_subpath`.

Live network downloads are NOT exercised — they would be flaky and
require external connectivity. Real integration is covered by
`ferrotorch-llama/tests/integration_quant_loaders.rs` (runs
`hf_download_model` against gated GPTQ/AWQ/HQQ repos when
`HF_TOKEN` is set, `#[ignore]`'d otherwise) and by the workspace
example binaries (`*_dump.rs`) which exercise the full path under
manual invocation.

Smoke command:

```bash
cargo test -p ferrotorch-hub --lib download:: 2>&1 | tail -3
```

Expected: 28 passed (default features).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn download_weights` in `download.rs` doing cache-hit → legacy-bare → http-fallback → http-off-error; non-test consumer: `download.rs::load_pretrained` calls `download_weights(info, &cache)` on every `load_pretrained::<T>(name)` invocation; downstream `ferrotorch-jit/examples/jit_trace_dump.rs` `use ferrotorch_hub::load_pretrained;` exercises the chain. |
| REQ-2 | SHIPPED | impl: `pub fn load_pretrained<T: Float>` in `download.rs` doing registry lookup → cache construction → `download_weights` → format-dispatch to `ferrotorch_serialize::load_*`; non-test consumer: `pub use download::load_pretrained` in `lib.rs`; `ferrotorch-jit/examples/jit_trace_dump.rs` imports and calls `load_pretrained`. |
| REQ-3 | SHIPPED | impl: `fn download_and_verify` in `download.rs` (http-gated, private) doing the auth-GET → cap-read → SHA-256 → compare → cache.store flow; non-test consumer: `download.rs::download_weights` calls `download_and_verify(info, cache)` on every cache-miss under `#[cfg(feature = "http")]`. |
| REQ-4 | SHIPPED | impl: `fn is_placeholder_sha256` in `download.rs` + the early-return guard at the top of `download_and_verify`; non-test consumer: the same `download_weights` chain fires this guard before every download; production callers see the fail-fast `InvalidArgument` when they `load_pretrained("unet")` (which still carries the placeholder digest pending public-mirror availability — see `registry.rs` comment on `unet`). |
| REQ-5 | SHIPPED | impl: `pub fn hf_download_model` in `download.rs` (http-gated); non-test consumer: `pub use download::hf_download_model` in `lib.rs`; called from `ferrotorch-llama/examples/llm_inference_dump.rs`, `ferrotorch-bert/examples/text_embedding_dump.rs`, `ferrotorch-diffusion/examples/{clip_text_encode_dump, vae_decode_dump, unet_predict_dump, unet_probe_dump, sd_pipeline_dump}.rs`, `ferrotorch-graph/examples/gcn_inference_dump.rs`, `ferrotorch-rl/examples/ppo_policy_dump.rs` (all production example binaries). |
| REQ-6 | SHIPPED | impl: the `repo.split('/')` + per-segment `sanitize_path_component(part, "repo component")` block at the top of `hf_download_model` in `download.rs`, plus the `revision` default + validation; non-test consumer: every shard download from `ferrotorch-llama/examples/llm_inference_dump.rs` and the other example binaries fires this validation on the user-supplied `repo` and `revision` arguments. |
| REQ-7 | SHIPPED | impl: the `sanitize_path_component(s, "shard filename")` call inside the `weight_map` loop in `hf_download_model` in `download.rs`, plus the empty-set guard; non-test consumer: every parsed `model.safetensors.index.json` from a real HF download flows through this guard; downstream examples are the production callers. |
| REQ-8 | SHIPPED | impl: `pub(crate) fn assert_within_cache` + `fn lexical_normalize` in `download.rs`; non-test consumer: `hf_download_model::fetch_one` calls `assert_within_cache(cache.cache_dir(), &final_path)` before every `cache.store`; the tokenizer best-effort block does the same. |
| REQ-9 | SHIPPED | impl: the `for opt in &["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"] { if let Some(body) = fetch_optional(...)? { assert_within_cache(...); cache.store(&cache_name, &body)?; } }` block in `hf_download_model` in `download.rs` (the #1147 regression fix); non-test consumer: `ferrotorch-llama/examples/llm_inference_dump.rs` and `ferrotorch-bert/examples/text_embedding_dump.rs` load `tokenizer.json` from the cache after `hf_download_model` returns — these downstream consumers depend on the bytes being written rather than silently discarded, which is the bug #1147 surfaced. |
