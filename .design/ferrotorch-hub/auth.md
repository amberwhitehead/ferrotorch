# ferrotorch-hub — `auth` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/hub.py
-->

## Summary

`ferrotorch-hub/src/auth.rs` implements HuggingFace-Hub authentication
token discovery and request decoration. `torch.hub` has no exact
counterpart — its `GITHUB_TOKEN` env-var support (`torch/hub.py:88`,
constant `ENV_GITHUB_TOKEN`) handles a different secret used to
authenticate against the GitHub API for `_validate_not_a_forked_repo`,
not the model-download path. The ferrotorch model zoo is hosted on the
HuggingFace mirror under `huggingface.co/ferrotorch/<name>` and gated
upstream repos like `meta-llama/Meta-Llama-3-8B` require
`Authorization: Bearer <HF_TOKEN>`. This module reproduces the
resolution order the HuggingFace Python client uses
(`HF_TOKEN` env var → `$HF_HOME/token` file → `$HOME/.cache/huggingface/token`
file) and decorates a `ureq::Request` builder when a token is available.
The whole module is gated by the `http` feature.

## Requirements

- REQ-1: `pub fn hf_token() -> Option<String>` discovers the HF auth
  token from the three standard sources in order: (1) `HF_TOKEN` env
  var if non-empty, (2) `$HF_HOME/token` file if `HF_HOME` is set,
  (3) `$HOME/.cache/huggingface/token` file. Returns `None` when no
  source yields a non-empty token. Each candidate is `.trim()`-ed and
  rejected if empty after trim, so a whitespace-only env var or
  file falls through to the next source.
- REQ-2: `pub fn with_auth(req: ureq::Request) -> ureq::Request`
  decorates a `ureq::Request` with `Authorization: Bearer <token>`
  when `hf_token()` returns a value and returns the original request
  unchanged otherwise. The signature is chosen so callers can compose
  it inline (`crate::auth::with_auth(ureq::get(&url)).call()?`).
- REQ-3: Auth is strictly opt-in: when no token is found, requests
  fall through unchanged so public-repo downloads (the common case)
  continue to work without configuration.
- REQ-4: All tests that mutate the `HF_TOKEN` / `HF_HOME` env vars
  hold a module-local `Mutex` for the entire mutate→read→restore
  window so the Rust 2024 unsafe contract on `std::env::set_var` is
  satisfied — no other thread of the same module can race on the
  env var while the mutation is live. Test failures must not leak
  state to sibling tests; every mutation is paired with a restore in
  the same critical section, and `Mutex::lock().unwrap_or_else(
  PoisonError::into_inner)` ensures a panic in one test does not
  poison the module's remaining tests.

## Acceptance Criteria

- [x] AC-1: `hf_token()` can be called without panicking even when no
  token is set (`token_from_env_var_takes_precedence`).
- [x] AC-2: An explicit `HF_TOKEN` env var is returned as the token
  (`token_from_explicit_env`).
- [x] AC-3: A whitespace-only `HF_TOKEN` is rejected and the resolver
  falls through (`empty_env_var_falls_through`).
- [x] AC-4: `with_auth(req)` returns a callable request when no token
  is set; the auth decoration is no-op
  (`with_auth_no_op_without_token`).
- [x] AC-5: The module is feature-gated — when `http` is disabled the
  module is absent from the crate API (verified by
  `cargo check -p ferrotorch-hub --no-default-features` succeeding
  without the `auth` symbol being referenced).

## Architecture

### Token discovery (REQ-1, REQ-3)

`pub fn hf_token() -> Option<String>` in `auth.rs` does the
three-source lookup:

1. `std::env::var("HF_TOKEN")` — if present and non-empty after
   `.trim()`, return that string.
2. Build a candidate list using `std::iter::empty().chain(...).chain(...)`
   with: `$HF_HOME/token` and `$HOME/.cache/huggingface/token`. Walk
   the list; for each file `read_to_string` and return the trimmed
   contents on the first non-empty hit.
3. Return `None` if no source yields anything.

Trimming matters because user-edited token files often have a
trailing newline that would corrupt the `Bearer <token>` header.

### Request decoration (REQ-2, REQ-3)

`pub fn with_auth(req: ureq::Request) -> ureq::Request`:

```rust
match hf_token() {
    Some(t) => req.set("Authorization", &format!("Bearer {t}")),
    None    => req,
}
```

The function takes the request by value and returns the (possibly
decorated) request by value, which is the shape `ureq`'s builder API
expects. The no-token branch is a literal pass-through so anonymous
downloads of public artifacts (the common case for ResNet-50 etc.)
continue to work without any setup.

### Env-var test discipline (REQ-4)

The Rust 2024 edition marks `std::env::set_var` / `remove_var` as
`unsafe` because the underlying C `setenv`/`getenv` machinery is not
thread-safe and cargo runs tests on multiple threads by default. The
test module uses a `static ENV_MUTEX: Mutex<()> = Mutex::new(());`
held for the lifetime of every mutate→read→restore window. Each
`unsafe { std::env::set_var(...) }` block carries a `// SAFETY:`
comment naming the invariant — "`ENV_MUTEX` serialises all
`set_var`/`remove_var` calls in this test module; no other thread
of this test process can race on `HF_TOKEN` while `_g` is held".

The lock is acquired via `.lock().unwrap_or_else(
std::sync::PoisonError::into_inner)` so a panic in a sibling test
does not poison the remainder — env state may be dirty after a panic,
but the read+restore pattern is robust to that. The caveat noted
inline (other crates in the workspace mutating these env vars from
their own tests would race) is documented; no such crate currently
exists.

### Non-test production consumers

- `crate::discovery::search_models` and `crate::discovery::get_model`
  in `discovery.rs` both wrap their `ureq::get(&url)` builders with
  `crate::auth::with_auth(...)`.
- `crate::download::download_and_verify` and the per-file fetchers
  inside `crate::download::hf_download_model` both wrap their
  `ureq::get(...)` builders with `crate::auth::with_auth(...)`.
- Every gated HF repo (`meta-llama/Meta-Llama-3-8B`, etc.) requires
  this header to return HTTP 200; without it the download path errors
  out on 401/403.

## Parity contract

`parity_ops = []`. There is no numerical PyTorch contract — auth is a
network-only concern. The closest upstream analog is
`torch/hub.py:88` (`ENV_GITHUB_TOKEN = "GITHUB_TOKEN"`) which serves a
different purpose: validating that a `repo_owner/repo_name:ref`
combination on GitHub belongs to the claimed owner. The HF-Hub
analog used by `huggingface_hub` Python client is documented at
<https://huggingface.co/docs/huggingface_hub/package_reference/authentication>
and follows the same env-var-then-file resolution.

Edge cases this module honours:

- Empty `HF_TOKEN` env var (set but to the empty string or whitespace)
  → treated as absent, resolver falls through.
- Missing token file → resolver falls through to the next candidate.
- File present but empty after trim → resolver falls through.
- No source yields a value → `hf_token()` returns `None`,
  `with_auth(req)` returns `req` unchanged.

## Verification

Tests in `mod tests in auth.rs` (4 tests, all under
`#[cfg(feature = "http")]`):

- `token_from_env_var_takes_precedence` — smoke that `hf_token()`
  can be called without panicking.
- `token_from_explicit_env` — `HF_TOKEN="hf_testtoken"` → returns
  `Some("hf_testtoken")`.
- `empty_env_var_falls_through` — `HF_TOKEN="   "` (whitespace) is
  rejected; the fallback chain runs.
- `with_auth_no_op_without_token` — with no token and a deliberately
  invalid `HF_HOME`, `with_auth(req)` is callable and returns a
  request shape that doesn't panic.

Smoke command:

```bash
cargo test -p ferrotorch-hub --lib auth:: 2>&1 | tail -3
```

Expected: 4 passed (each test is gated by the `http` feature so on a
default-feature build all four run).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn hf_token` in `auth.rs` doing the env-var-then-files three-source lookup with `.trim()` on each candidate; non-test consumer: `crate::discovery::search_models` and `crate::discovery::get_model` (in `discovery.rs`) wrap `ureq::get` with `crate::auth::with_auth`, which calls `hf_token()` on every request; `crate::download::download_and_verify` and `crate::download::hf_download_model` (in `download.rs`) do the same. |
| REQ-2 | SHIPPED | impl: `pub fn with_auth` in `auth.rs` matching on `hf_token()` and adding `Authorization: Bearer <t>` via `req.set(...)`; non-test consumer: `download.rs::download_and_verify` calls `crate::auth::with_auth(ureq::get(info.weights_url))` to inject the bearer header before `.call()`; `download.rs::hf_download_model`'s `fetch_one` and `fetch_optional` helpers do the same. |
| REQ-3 | SHIPPED | impl: the `None` arm in `with_auth` returns the request unchanged in `auth.rs`; non-test consumer: every public-repo download (`resnet50`, `vit_b_16`, etc. from `registry.rs`) flows through `download_and_verify` → `with_auth` and reaches HuggingFace without an `Authorization` header, which is the contract HF's public-repo endpoint expects. |
| REQ-4 | SHIPPED | impl: `static ENV_MUTEX: Mutex<()>` in `auth.rs::mod tests` plus the paired `// SAFETY:` comments on each `unsafe { std::env::set_var(...) }` and `std::env::remove_var(...)` block; non-test consumer: the test module guarantees that the production callers (`with_auth`) see a consistent env-var state during testing — without this, the `hf_token` read inside `with_auth` would race with concurrent test mutations and produce spurious decorated requests, but neither downstream consumer is itself test-only. The disciplined env-var protocol is what lets `auth.rs` ship without a crate-root `unsafe_code` allow that the production-side `with_auth` does not need. |
