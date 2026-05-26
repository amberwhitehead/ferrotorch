# ferrotorch-hub — crate root

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/hub.py
-->

## Summary

`ferrotorch-hub/src/lib.rs` is the crate root for the ferrotorch model
hub. It pins the workspace lint baseline, declares the six public
sub-modules (`auth`, `cache`, `discovery`, `download`, `hf_config`,
`registry`) and re-exports their user-facing surface — the same shape
upstream PyTorch exposes via `torch.hub` (`torch/hub.py:68-76`) plus a
HuggingFace-Hub-shaped extension layer (`hf_download_model`,
`HfTransformerConfig`, `SearchQuery`, …) that has no direct counterpart
in `torch.hub` but mirrors the responsibilities of the HuggingFace
Python `huggingface_hub` client. The `http` cargo feature gates every
module that performs network I/O.

## Requirements

- REQ-1: The crate root declares the public module layout — `auth`
  (http-gated), `cache`, `discovery` (http-gated), `download`,
  `hf_config`, `registry` — so each module's API surface reaches users
  through the crate's top level.
- REQ-2: The crate re-exports a flat user-facing surface so downstream
  callers (`ferrotorch-llama`, `ferrotorch-bert`, `ferrotorch-jit`,
  `ferrotorch-diffusion`, the meta-crate's `ferrotorch::hub` module)
  can `use ferrotorch_hub::{load_pretrained, hf_download_model,
  HfTransformerConfig, HubCache, …}` without navigating internal
  modules.
- REQ-3: Lint baseline mirrors the workspace pattern (`#![warn(
  clippy::all, clippy::pedantic)]`, `#![deny(rust_2018_idioms)]`)
  with a documented `#![allow(missing_docs,
  missing_debug_implementations)]` parity-with-other-leaf-crates
  exemption and per-lint justifications for every pedantic allow.
  Module-root `#![allow]` is restricted to the workspace-wide ones
  listed in the file header; no new ones may be added without a
  justification block (R-CODE-3).
- REQ-4: `unsafe_code` is intentionally NOT denied at the crate root
  because the auth-module tests require `unsafe` blocks for
  `std::env::set_var` under Rust 2024. Per-block `// SAFETY:`
  comments substantiate each `unsafe` site (R-CODE-1).
- REQ-5: The `http` feature is a compile-time toggle that activates
  the network-using modules (`auth`, `discovery`) and the
  `hf_download_model` symbol in `download`. With `http` disabled the
  crate still compiles and still exposes the static `registry`, the
  flat-disk `cache`, the offline arm of `download_weights`, and the
  `hf_config` JSON parser — every offline path remains usable.

## Acceptance Criteria

- [x] AC-1: `cargo check -p ferrotorch-hub` succeeds with default
  features.
- [x] AC-2: `cargo check -p ferrotorch-hub --no-default-features`
  succeeds (http-gated modules behind `#[cfg(feature = "http")]` do
  not break the offline build).
- [x] AC-3: `cargo clippy -p ferrotorch-hub --lib -- -D warnings`
  passes — every pedantic allow has a one-line justification.
- [x] AC-4: `cargo test -p ferrotorch-hub --lib` passes (>=70 tests
  across the seven modules).
- [x] AC-5: The meta-crate's `ferrotorch::hub` module
  (`ferrotorch/src/lib.rs`, gated by feature `hub`) re-exports
  everything via `pub use ferrotorch_hub::*;`.

## Architecture

### Module declarations (REQ-1)

The `pub mod` block declares the six modules in `lib.rs`:

```rust
#[cfg(feature = "http")]
pub mod auth;
pub mod cache;
#[cfg(feature = "http")]
pub mod discovery;
pub mod download;
pub mod hf_config;
pub mod registry;
```

- `auth` and `discovery` carry `#[cfg(feature = "http")]` because they
  both transitively depend on `ureq` and the optional HuggingFace
  auth-token discovery flow.
- `cache`, `download`, `hf_config`, `registry` are unconditional. The
  network-only fns inside `download` (`hf_download_model`,
  `download_and_verify`) carry their own `#[cfg(feature = "http")]`
  guards.

### Re-export surface (REQ-2)

The crate root re-exports each module's user-facing types in
`pub use` blocks:

- `pub use auth::{hf_token, with_auth};` (http-gated)
- `pub use cache::{HubCache, default_cache_dir};`
- `pub use discovery::{HfModelInfo, HfModelSummary, HfRepoFile,
  SearchQuery, get_model, search_models};` (http-gated)
- `pub use download::hf_download_model;` (http-gated)
- `pub use download::{download_weights, load_pretrained};`
- `pub use hf_config::HfTransformerConfig;`
- `pub use registry::{EntryKind, ModelInfo, WeightsFormat,
  get_model_info, list_models};`

This is the boundary upstream callers see. The non-test production
consumers are visible across the workspace:

- `ferrotorch-llama/src/config.rs` (`use ferrotorch_hub::
  HfTransformerConfig;`) — `LlamaConfig::from_hf` builds the
  model-specific config from a parsed HF JSON.
- `ferrotorch-llama/examples/llama3_8b.rs`,
  `llama3_8b_gpu.rs`, `llama3_70b_gpu.rs`,
  `prosparse_7b_gpu.rs`, `llm_inference_dump.rs` — all import
  `HfTransformerConfig` and `hf_download_model` from the crate root.
- `ferrotorch-bert/examples/text_embedding_dump.rs`,
  `ferrotorch-diffusion/examples/{vae_decode_dump,
  clip_text_encode_dump, unet_predict_dump, unet_probe_dump,
  sd_pipeline_dump}.rs`,
  `ferrotorch-graph/examples/gcn_inference_dump.rs`,
  `ferrotorch-rl/examples/ppo_policy_dump.rs`,
  `ferrotorch-jit/examples/jit_trace_dump.rs` — all use the flat
  re-exports as their authoritative entry point.
- `ferrotorch/src/lib.rs` — `pub mod hub { pub use ferrotorch_hub::*; }`
  (feature-gated by `hub`) re-exports the entire surface via the
  meta-crate.

### Lint policy (REQ-3, REQ-4)

The crate-root file header lists every `#![allow]` with a one-line
justification (module-name repetitions, missing_errors_doc /
missing_panics_doc, must_use_candidate, return_self_not_must_use,
doc_markdown, format_push_string, items_after_statements,
uninlined_format_args, unnecessary_debug_formatting,
unreadable_literal, too_many_lines). New allows require a comment
explaining why the alternative is worse. `unsafe_code` is NOT in the
deny list — the `auth` module's tests synchronise env-var mutation
via a `Mutex` plus `unsafe { std::env::set_var(...) }`, and the
`// SAFETY:` block at each call site documents the
"`ENV_MUTEX` serialises every `set_var`/`remove_var` in this module"
invariant Rust 2024's unsafe contract on `env::set_var` requires.

### `http` feature (REQ-5)

The `http` feature flag is the workspace-level switch for network
access. With the feature enabled:

- `auth` module compiles → `hf_token`, `with_auth` available.
- `discovery` module compiles → `search_models`, `get_model`,
  `SearchQuery`, `HfModelInfo`, `HfModelSummary`, `HfRepoFile`
  available.
- `download::hf_download_model` compiles → sharded HF repo download.
- `download::download_and_verify` compiles → URL fetch + SHA-256
  verification in `load_pretrained`'s download arm.

With `http` disabled:

- `auth` and `discovery` are absent from the crate API.
- `download::download_weights` falls through to a clear
  `FerrotorchError::InvalidArgument` with "the `http` feature is
  disabled. Download manually from … and place at …" pointing the
  user at the canonical cache path.
- The static `registry`, the flat-disk `HubCache`, and the
  `HfTransformerConfig` JSON parser all remain usable for offline
  inference.

## Parity contract

`parity_ops = []`. The crate root has no numerical surface — it is
declaration + re-export plumbing. The conceptual upstream is
`torch/hub.py`'s `__all__` block (`torch/hub.py:68-76`):

```python
__all__ = [
    "download_url_to_file",   # → download::download_and_verify
    "get_dir",                # → cache::default_cache_dir
    "help",                   # NOT-STARTED (no hubconf.py loader)
    "list",                   # NOT-STARTED (entrypoint discovery)
    "load",                   # NOT-STARTED (hubconf.py loader)
    "load_state_dict_from_url",  # → download::load_pretrained
    "set_dir",                # NOT-STARTED (no global mutable cache dir)
]
```

The HuggingFace-Hub-shaped surface (`hf_download_model`, `hf_token`,
`with_auth`, `HfTransformerConfig`, `SearchQuery`, …) has no
counterpart in `torch.hub`; it mirrors what the Python
`huggingface_hub` client does, since the ferrotorch model-zoo is
hosted on the HF mirror under `huggingface.co/ferrotorch/<name>`.

Edge-case contract:

- `--no-default-features` build: crate must compile and expose the
  offline-only surface (registry + cache + hf_config + offline arm of
  download).
- `--features http` build (default): all six modules visible.
- `--all-features` build: identical to `--features http` (no other
  features are defined).

## Verification

Tests at the crate root are NONE — `lib.rs` is pure module + re-export
declaration. Verification is structural via the workspace gauntlet:

```bash
cargo check -p ferrotorch-hub
cargo check -p ferrotorch-hub --no-default-features
cargo clippy -p ferrotorch-hub --lib -- -D warnings
cargo test -p ferrotorch-hub --lib
cargo fmt -p ferrotorch-hub --check
```

Each of the six sub-modules ships its own `#[cfg(test)] mod tests`;
the aggregated `cargo test` count is the structural pass criterion.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub mod` declarations in `lib.rs` (the `auth`/`discovery` blocks are `#[cfg(feature = "http")]`-gated, the other four are unconditional); non-test consumer: `ferrotorch-llama/src/config.rs` imports `HfTransformerConfig`, `ferrotorch-llama/examples/llm_inference_dump.rs` imports `HubCache` + `hf_download_model` from the crate root. |
| REQ-2 | SHIPPED | impl: `pub use` block in `lib.rs` flattening every module's user-facing types; non-test consumer: `ferrotorch/src/lib.rs` `pub mod hub { pub use ferrotorch_hub::*; }` re-exports the whole flat surface through the meta-crate; `ferrotorch-jit/examples/jit_trace_dump.rs` imports `load_pretrained` directly from the crate root. |
| REQ-3 | SHIPPED | impl: crate-level `#![warn(clippy::all, clippy::pedantic)]` + `#![deny(rust_2018_idioms)]` + per-lint `#![allow]` block with one-line justifications in `lib.rs`; non-test consumer: `cargo clippy -p ferrotorch-hub --lib -- -D warnings` passes for every consumer of the crate (the lint policy is a precondition for every dependent crate building cleanly). |
| REQ-4 | SHIPPED | impl: the deliberately-omitted `unsafe_code` deny in `lib.rs`'s lint header (documented in the comment block); non-test consumer: every `unsafe { std::env::set_var(...) }` block in `auth.rs::mod tests` is a `#[cfg(test)]`-only callee — under R-APG-2 production code does not exercise this path, so the test-only `unsafe` does not propagate to consumers. The lint posture lets the test module compile without a crate-root override. |
| REQ-5 | SHIPPED | impl: `#[cfg(feature = "http")]` guards on `pub mod auth;`, `pub mod discovery;`, `pub use auth::…;`, `pub use discovery::…;`, `pub use download::hf_download_model;` in `lib.rs`; offline path in `download.rs::download_weights`; non-test consumer: `ferrotorch-jit/Cargo.toml` declares `ferrotorch-hub = { workspace = true, features = ["http"] }` for its `jit_trace_dump` example that calls `load_pretrained`; the meta-crate's `hub` feature gate in `ferrotorch/Cargo.toml` flows through to the same conditional. |
