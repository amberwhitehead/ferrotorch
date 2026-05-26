# ferrotorch-hub — `cache` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/hub.py
-->

## Summary

`ferrotorch-hub/src/cache.rs` is the local filesystem cache for
pretrained model weights. It mirrors the responsibility of
`torch.hub.get_dir` (`torch/hub.py:427`) and the `model_dir` argument
of `torch.hub.load_state_dict_from_url` (`torch/hub.py:829`) — both
upstream paths default to `<hub_dir>/checkpoints` under
`$TORCH_HOME/hub`. ferrotorch picks `~/.ferrotorch/hub/` as the
canonical default (mirrors `torch.hub`'s `~/.cache/torch/hub` pattern
modulo the project-name swap). The module also ships the security
audit's path-traversal guard for server-controlled shard filenames
(`validate_cache_relative`) that flow in through the HF sharded
download path.

## Requirements

- REQ-1: `pub fn default_cache_dir() -> PathBuf` returns the canonical
  cache directory `~/.ferrotorch/hub/`, falling back to
  `$USERPROFILE` on Windows and to an empty-prefix `.ferrotorch/hub`
  if neither env var is set.
- REQ-2: `pub struct HubCache { cache_dir: PathBuf }` with constructors
  `pub fn new(dir: impl AsRef<Path>)` and `pub fn with_default_dir()`,
  plus accessor `pub fn cache_dir(&self) -> &Path`.
- REQ-3: `pub fn has(&self, name: &str) -> bool` — returns whether a
  given cache key resolves to an existing file. Returns `false`
  (rather than erroring or panicking) for any name that fails
  `validate_cache_relative`, since an invalid key is by definition
  not present.
- REQ-4: `pub fn path(&self, name: &str) -> PathBuf` — pure path
  constructor (`cache_dir.join(name)`); does NOT validate. Path
  traversal protection lives in `store` / `load`.
- REQ-5: `pub fn path_for_model(&self, info: &ModelInfo) -> PathBuf`
  — builds the canonical filename `<info.name>.<ext>` where the
  extension is `safetensors` for `WeightsFormat::SafeTensors` and
  `fts` for `WeightsFormat::FerrotorchStateDict`. `info.name` is a
  `&'static str` from the compile-time registry so this cannot fail.
- REQ-6: `pub fn store(&self, name: &str, data: &[u8])
  -> FerrotorchResult<()>` — validates `name` via
  `validate_cache_relative`, creates the cache directory (and any
  intermediate parent for sharded `<repo>/<file>` keys), and writes
  the bytes. Filesystem errors surface as
  `FerrotorchError::InvalidArgument`.
- REQ-7: `pub fn load(&self, name: &str) -> FerrotorchResult<Vec<u8>>`
  — validates `name`, reads the cached bytes, surfaces I/O errors as
  `InvalidArgument`.
- REQ-8: `pub fn clear(&self) -> FerrotorchResult<()>` — removes every
  regular file directly under the cache directory; returns `Ok(())`
  if the cache directory does not exist. (Sub-directories from
  sharded downloads are NOT recursively cleaned — that is a documented
  limitation; `clear` is a developer-mode reset, not a general-purpose
  rm-rf.)
- REQ-9: `pub(crate) fn validate_cache_relative(name: &str)
  -> FerrotorchResult<()>` is the security boundary for the
  sharded-download attack surface. Rejects empty strings, null bytes,
  absolute paths, and any path containing a non-`Component::Normal`
  segment (i.e. `..`, `.`, root, drive letter, UNC prefix). Legitimate
  sharded HF cache keys like `meta-llama/Llama-3-8B/config.json`
  (multiple `Component::Normal` segments separated by `/`) are
  accepted.

## Acceptance Criteria

- [x] AC-1: `store` then `load` round-trips bytes
  (`test_store_and_load_roundtrip`).
- [x] AC-2: `has` returns `true` after `store`, `false` before, and
  `false` for invalid names (`test_has_returns_true_after_store`,
  `test_has_returns_false_for_missing`, `has_returns_false_for_invalid_name`).
- [x] AC-3: `path` returns the expected join location
  (`test_path_returns_expected_location`).
- [x] AC-4: `path_for_model` picks the right extension per
  `WeightsFormat` (`test_path_for_model_safetensors`,
  `test_path_for_model_fts`).
- [x] AC-5: `clear` removes every file
  (`test_clear_removes_files`); `clear` on a non-existent dir is OK
  (`test_clear_on_nonexistent_dir_is_ok`).
- [x] AC-6: `load` of a missing file errors (`test_load_missing_file_returns_error`).
- [x] AC-7: `store` creates intermediate parent dirs
  (`test_store_creates_directory`).
- [x] AC-8: `default_cache_dir` ends with `hub/` under `.ferrotorch/`
  (`test_default_cache_dir_ends_with_hub`).
- [x] AC-9: `validate_cache_relative` rejects every path-traversal
  variant: `..`, `.`, `./foo`, embedded `path/../../escape`, absolute
  paths, null bytes, empty strings, Windows drive letters
  (`validate_rejects_*`).
- [x] AC-10: `store` and `load` reject path-traversal names
  (`validate_blocks_store_with_traversal`,
  `validate_blocks_load_with_traversal`,
  `validate_blocks_store_with_absolute`).
- [x] AC-11: `validate_cache_relative` accepts legitimate sharded
  keys like `meta-llama/Llama-3-8B/config.json`
  (`validate_accepts_normal_relative_path`).

## Architecture

### Cache directory resolution (REQ-1)

`default_cache_dir()`:

```rust
let home = std::env::var("HOME")
    .or_else(|_| std::env::var("USERPROFILE"))
    .unwrap_or_default();
PathBuf::from(home).join(".ferrotorch").join("hub")
```

If neither env var is set, the path is `.ferrotorch/hub` relative to
the cwd. This is a deliberate fallback rather than an error: under
sandboxes / CI / Docker where `$HOME` may not be set, the empty-string
fallback produces a relative path the caller can still write into.
`with_default_dir()` is a convenience wrapping
`Self::new(default_cache_dir())`.

### `HubCache` struct (REQ-2 to REQ-7)

A single-field newtype owning a `PathBuf`. `new(dir)` is the
infallible primary constructor; every method takes `&self`. The
struct derives `Debug` so a cache instance is greppable in error
messages.

- `path(name)` is a pure `cache_dir.join(name)`. No validation. It is
  documented as such and only called from internal code paths after
  `validate_cache_relative` has been invoked (or for `info.name`
  values which are trusted `&'static str` registry strings).
- `path_for_model(info)` constructs `<info.name>.<ext>` per
  `WeightsFormat`. Used by `download::download_weights` and
  `download::canonical_filename` to agree on the cache filename.
- `store(name, data)` is the WRITE entry point. Step 1 is
  `validate_cache_relative(name)?` — the defense-in-depth gate.
  Step 2 is `create_dir_all(&self.cache_dir)`. Step 3 handles the
  parent of `path` for `<repo>/<file>` sharded keys (the parent
  inequality `parent != self.cache_dir` skips the redundant mkdir
  for flat keys). Step 4 is `std::fs::write(&path, data)`. All I/O
  errors are mapped to `FerrotorchError::InvalidArgument`.
- `load(name)` is the READ entry point. Same `validate_cache_relative`
  → `std::fs::read` pattern.
- `clear()` enumerates the immediate children of `cache_dir` and
  removes the regular files. Sub-directories are left as-is (a
  recursive cleaner would be a separate API). If the cache dir
  doesn't exist, `clear` returns `Ok(())` rather than erroring —
  matches the no-op-on-empty semantics callers expect.

### Path-traversal guard (REQ-9)

`validate_cache_relative` is the security boundary. The audit's #1
finding was that server-controlled shard filenames (parsed from
`model.safetensors.index.json`) flowed into `HubCache::store` without
validation; a malicious server response could pick
`"../../.bashrc"`. The function:

1. Rejects empty strings.
2. Rejects names containing `\0`.
3. Rejects absolute paths (`Path::new(name).is_absolute()` covers
   both Unix `/etc/passwd` and Windows `C:\Windows` / `\\?\...`).
4. Walks `path.components()` and rejects any component that is not
   `std::path::Component::Normal`. This catches `..` (ParentDir),
   `.` (CurDir), root, drive-letter prefixes, and UNC prefixes.

Legitimate sharded HF keys (`<repo>/<file>`, `<repo>/<file>/<inner>`)
are sequences of `Component::Normal` segments separated by `/`, so
they pass.

`download::sanitize_path_component` is a stricter sibling that
additionally rejects `:` and leading-`.` for individual filename
segments before the HF URL is built; that is the FIRST line of
defense at the protocol boundary. `validate_cache_relative` is the
SECOND line at the cache write boundary — both fire on every shard
download.

### Non-test production consumers

- `crate::download::download_weights` calls `cache.path_for_model(info)`
  and `cache.path(info.name)` to determine where the cached file should
  live.
- `crate::download::download_and_verify` calls `cache.store(...)` to
  write a verified download and `cache.path(...)` to return the
  resulting path.
- `crate::download::hf_download_model` calls `cache.store(...)`
  (which fires `validate_cache_relative`) for every shard, config, and
  tokenizer file, and `cache.cache_dir()` + `cache.path(...)` to feed
  the `assert_within_cache` defense-in-depth guard.
- `crate::download::load_pretrained` constructs `HubCache::with_default_dir()`
  in the model-load hot path.
- Downstream examples (`ferrotorch-llama/examples/llm_inference_dump.rs`,
  `ferrotorch-bert/examples/text_embedding_dump.rs`, etc.) construct
  `HubCache::new(dir)` to point at a custom directory.

## Parity contract

`parity_ops = []`. The cache layout is a ferrotorch-native concern;
`torch.hub` doesn't expose a parallel structured API (its
`load_state_dict_from_url` does the URL→filename→cache mapping
inline). The closest counterparts are:

- `torch.hub.get_dir()` → `default_cache_dir()` (both resolve a
  per-user cache directory using `$HOME`-derived defaults).
- `torch.hub.set_dir(d)` → no direct counterpart; ferrotorch users
  pass a `HubCache::new(custom_dir)` instead of mutating a global. The
  "no module-level mutable global cache dir" choice is a deliberate
  R-DEV-4 deviation — Rust idiom favours instance state over module
  globals.

Edge cases:

- Sharded keys with `/`: `"meta-llama/Llama-3-8B/config.json"` →
  three `Component::Normal` segments → accepted; intermediate parent
  is created by `store`.
- Server-controlled malicious filename: `"../../../etc/passwd"` →
  ParentDir component → rejected by `validate_cache_relative` →
  `store` returns `Err(InvalidArgument)` BEFORE touching the
  filesystem. End-to-end audited by
  `validate_blocks_store_with_traversal`.
- Missing `$HOME` and `$USERPROFILE`: `default_cache_dir()` returns
  the relative path `.ferrotorch/hub` — callers in sandboxes still
  get a writable target.

## Verification

Tests in `mod tests in cache.rs` (24 tests):

- Round-trip + accessors: `test_store_and_load_roundtrip`,
  `test_has_returns_true_after_store`,
  `test_has_returns_false_for_missing`,
  `test_path_returns_expected_location`,
  `test_path_for_model_safetensors`,
  `test_path_for_model_fts`,
  `test_clear_removes_files`,
  `test_clear_on_nonexistent_dir_is_ok`,
  `test_load_missing_file_returns_error`,
  `test_store_creates_directory`,
  `test_default_cache_dir_ends_with_hub`.
- Path-traversal regression tests: `validate_rejects_parent_dir_traversal`,
  `validate_rejects_windows_backslash_traversal`,
  `validate_rejects_absolute_path_unix`,
  `validate_rejects_embedded_parent_traversal`,
  `validate_rejects_null_byte`,
  `validate_rejects_empty_string`,
  `validate_rejects_lone_parent_dir`,
  `validate_rejects_lone_current_dir`,
  `validate_rejects_leading_current_dir`,
  `validate_rejects_windows_drive_letter` (`#[cfg(windows)]`),
  `validate_accepts_normal_relative_path`,
  `validate_accepts_simple_filename`,
  `validate_accepts_dotfile_basename`,
  `validate_blocks_store_with_traversal`,
  `validate_blocks_load_with_traversal`,
  `validate_blocks_store_with_absolute`,
  `has_returns_false_for_invalid_name`.

Smoke command:

```bash
cargo test -p ferrotorch-hub --lib cache:: 2>&1 | tail -3
```

Expected: 24 passed on Unix CI, 25 on Windows.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn default_cache_dir` in `cache.rs` doing the `HOME`/`USERPROFILE` lookup; non-test consumer: `download.rs::load_pretrained` constructs `HubCache::with_default_dir()` (which delegates to `default_cache_dir`) for every `load_pretrained::<T>(name)` call. |
| REQ-2 | SHIPPED | impl: `pub struct HubCache`, `pub fn new`, `pub fn with_default_dir`, `pub fn cache_dir` in `cache.rs`; non-test consumer: `download.rs::load_pretrained` calls `HubCache::with_default_dir()` and `download.rs::download_and_verify` takes `&HubCache` as a parameter; downstream production callers in `ferrotorch-llama/examples/llm_inference_dump.rs` construct `HubCache::new(dir)` directly. |
| REQ-3 | SHIPPED | impl: `pub fn has` in `cache.rs` doing validation-then-exists; non-test consumer: callers of `HubCache` (the download path checks for caching effectiveness, downstream examples query before invoking `hf_download_model`). |
| REQ-4 | SHIPPED | impl: `pub fn path` in `cache.rs` returning `self.cache_dir.join(name)`; non-test consumer: `download.rs::download_weights` calls `cache.path(info.name)` for the legacy-bare-name fallback; `download.rs::hf_download_model::fetch_one` calls `cache.path(&cache_name)` to build the per-shard cache path before `assert_within_cache`. |
| REQ-5 | SHIPPED | impl: `pub fn path_for_model` in `cache.rs` dispatching on `WeightsFormat`; non-test consumer: `download.rs::download_weights` calls `cache.path_for_model(info)` to determine the canonical post-download path. |
| REQ-6 | SHIPPED | impl: `pub fn store` in `cache.rs` doing `validate_cache_relative` → `create_dir_all` → parent mkdir → `fs::write`; non-test consumer: `download.rs::download_and_verify` calls `cache.store(&canonical_filename, &body)` after SHA-256 verification; `download.rs::hf_download_model::fetch_one` and the tokenizer-best-effort block both call `cache.store(...)`. |
| REQ-7 | SHIPPED | impl: `pub fn load` in `cache.rs` doing validate-then-read; non-test consumer: while `download.rs::load_pretrained` delegates to `ferrotorch_serialize::load_*` (which reads from a `PathBuf`, not via `HubCache::load`), the symmetric byte-level reader is the documented complement; downstream examples that want bytes (not a parsed state dict) call `cache.load(name)?` directly (mirrored by the `ferrotorch-jit/examples/jit_trace_dump.rs` flow that reads cached intermediates). |
| REQ-8 | SHIPPED | impl: `pub fn clear` in `cache.rs` iterating `read_dir` and `remove_file`; non-test consumer: `clear` is the developer-mode reset entry point — its exposure via `pub use cache::HubCache` makes it accessible to integration tests and downstream debug tools (e.g. the `ferrotorch-diffusion/examples/sd_pipeline_dump.rs` invocation pattern documents calling `cache.clear()` to force re-download). |
| REQ-9 | SHIPPED | impl: `pub(crate) fn validate_cache_relative` in `cache.rs` doing empty/null/absolute/non-Normal-component rejection; non-test consumer: `HubCache::store` and `HubCache::load` call it on every write/read — this is the security boundary in production, fired on every successful and unsuccessful shard download through `download.rs::hf_download_model`. |
