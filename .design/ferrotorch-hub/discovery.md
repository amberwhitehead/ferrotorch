# ferrotorch-hub — `discovery` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/hub.py
-->

## Summary

`ferrotorch-hub/src/discovery.rs` provides dynamic model discovery
against the live HuggingFace Hub API. It is the runtime complement to
the compiled-in `registry.rs` table — the static registry only knows
about models ferrotorch has explicitly curated, while this module
exposes the full Hub catalogue via `GET /api/models` and
`GET /api/models/{repo_id}` so users can search, browse, and look up
arbitrary repos. The module has no exact counterpart in
`torch/hub.py`; upstream's `torch.hub.list(github)` enumerates
`hubconf.py` entrypoints from a GitHub repo
(`torch/hub.py:459-523`) — a fundamentally different discovery model
since `torch.hub` indexes GitHub repos, while HuggingFace Hub is the
ferrotorch ecosystem's model index. The whole module is gated by the
`http` feature.

## Requirements

- REQ-1: `pub struct HfModelSummary` mirrors a Hub model-search-result
  entry. Deserialises from JSON via `serde` with selective fields
  (`model_id` aliased to both `modelId` and `id`, plus
  `author`, `downloads`, `likes`, `tags`, `library_name`,
  `pipeline_tag`). Unknown JSON fields are ignored so a Hub schema
  bump does not break the deserializer.
- REQ-2: `pub struct HfModelInfo` mirrors a Hub
  `GET /api/models/{repo_id}` response — the superset of
  `HfModelSummary` plus per-file siblings (`Vec<HfRepoFile>`,
  each carrying a `rfilename`).
- REQ-3: `pub struct SearchQuery` is a builder for the Hub search
  query string. Methods: `new`, `with_search`, `with_pipeline_tag`,
  `with_library`, `with_limit`, `with_sort`. Empty query → bare
  `/api/models` endpoint; any field set → `?key=value` pairs URL-
  encoded.
- REQ-4: `pub fn search_models(query: &SearchQuery)
  -> FerrotorchResult<Vec<HfModelSummary>>` performs the blocking
  HTTP GET against `https://huggingface.co/api/models?...`, injects
  the optional bearer token via `crate::auth::with_auth`, and
  parses the JSON response. Errors at the HTTP layer or JSON layer
  surface as `FerrotorchError::InvalidArgument` with the full URL
  in the message for diagnosis. Missing `author` fields are populated
  from the leading segment of `model_id` so consumers can rely on
  the field.
- REQ-5: `pub fn get_model(repo_id: &str) -> FerrotorchResult<HfModelInfo>`
  fetches `https://huggingface.co/api/models/{repo_id}` for a single
  repo and returns the full record including the `siblings` file
  list. Empty `repo_id` is rejected with `InvalidArgument` before
  any HTTP call. Missing `author` field is populated from the
  leading segment of `model_id`.
- REQ-6: URL encoding for the `search` / `pipeline_tag` / `library` /
  `sort` query-string values handles the small subset of
  characters Hub search values actually use (alphanumerics, `-`,
  `_`, `.`, `~` pass through; everything else becomes `%XX`).
  Conservative subset of RFC 3986 unreserved characters; no external
  URL-encoding dep.

## Acceptance Criteria

- [x] AC-1: Empty `SearchQuery` builds the bare `/api/models`
  endpoint (`test_search_query_empty_is_bare_endpoint`).
- [x] AC-2: `SearchQuery::with_search` produces `?search=<term>`
  (`test_search_query_search_only`).
- [x] AC-3: All five `SearchQuery` fields encode into the query
  string (`test_search_query_all_fields`).
- [x] AC-4: `url_encode` passes through alphanumerics, hyphen,
  underscore, dot, tilde (`test_url_encode_alphanumeric_passthrough`).
- [x] AC-5: `url_encode` percent-encodes spaces, slashes, ampersands,
  equals (`test_url_encode_special_chars`).
- [x] AC-6: `extract_author` derives a namespace from
  `namespace/model` ids (`test_extract_author_namespaced`).
- [x] AC-7: `extract_author` returns `None` for top-level models
  (`test_extract_author_top_level`).
- [x] AC-8: `HfModelSummary` deserialises minimal `{modelId}`
  (`test_deserialize_model_summary_minimal`).
- [x] AC-9: `HfModelSummary` honours the `id` alias for `modelId`
  (`test_deserialize_model_summary_id_alias`).
- [x] AC-10: `HfModelSummary` fills every populated field
  (`test_deserialize_model_summary_full`).
- [x] AC-11: Unknown JSON fields are ignored
  (`test_deserialize_model_summary_unknown_fields_ignored`).
- [x] AC-12: `HfModelInfo` parses siblings
  (`test_deserialize_model_info_with_siblings`).
- [x] AC-13: `populate_authors` fills missing authors without
  clobbering set ones (`test_populate_authors_fills_missing`).
- [x] AC-14: `get_model("")` rejects empty id before HTTP
  (`test_get_model_empty_repo_id_errors`).

## Architecture

### Data types (REQ-1, REQ-2)

`HfModelSummary` and `HfModelInfo` are flat `serde::Deserialize`
structs. The Hub API has many more fields per model than these — we
deserialize only the ones useful for ferrotorch's discovery flow.
`#[serde(default)]` is set on every optional field so a minimal
`{"modelId": "..."}` response parses with all other fields at their
defaults. `#[serde(rename = "modelId", alias = "id")]` on `model_id`
handles both Hub endpoint variants (some endpoints emit `modelId`,
others `id`). `HfRepoFile` is a single-field wrapper around
`rfilename` for the `siblings: Vec<HfRepoFile>` array.

All three structs derive `Debug`, `Clone`, `Serialize`, `Deserialize`,
`PartialEq`, `Eq` (modulo `HfModelInfo` which omits `PartialEq`/`Eq`
because `Vec<HfRepoFile>` doesn't gain them by default and there's no
consumer that needs equality on the full info struct).

### Query builder (REQ-3, REQ-6)

`SearchQuery` is a builder-pattern struct with `Option<...>` fields
for every parameter. `with_*` methods take `self` by value and
return `Self` so callers can chain:

```rust
SearchQuery::new()
    .with_search("resnet")
    .with_limit(10)
    .with_pipeline_tag("image-classification")
```

`to_query_string(&self) -> String` (the `pub(crate)` builder method)
walks the fields and produces:

- empty query → `"/api/models"` (no `?`).
- any field set → `"/api/models?<k1>=<v1>&<k2>=<v2>&..."` with each
  string-valued field URL-encoded via `url_encode`.

`url_encode` is a hand-written percent-encoder for the conservative
RFC 3986 unreserved subset. The whole module has no external
URL-encoding dep, which keeps the dependency surface honest.

### Search + lookup (REQ-4, REQ-5)

`search_models(query)`:

1. Build URL from `HUB_BASE` + `query.to_query_string()`.
2. `crate::auth::with_auth(ureq::get(&url)).call()?` injects the
   optional bearer token.
3. `response.into_json::<Vec<HfModelSummary>>()?`.
4. `populate_authors(summaries)` walks the list and fills any
   `author == None` from the leading segment of `model_id`.
5. Return the populated vec.

`get_model(repo_id)`:

1. Reject empty `repo_id` early with `InvalidArgument`.
2. URL `https://huggingface.co/api/models/{repo_id}`.
3. Auth-decorate, call, `into_json::<HfModelInfo>()`.
4. Populate `info.author` from `extract_author(&info.model_id)` if
   absent.
5. Return.

Both fns map HTTP / JSON errors to `FerrotorchError::InvalidArgument`
with the URL embedded so a failure is debuggable.

### Helper fns (REQ-4, REQ-5)

- `populate_authors(summaries)` — walks `&mut Vec<HfModelSummary>`,
  fills the `author` field for entries where the Hub didn't set it,
  by splitting `model_id` on the first `/`.
- `extract_author(model_id) -> Option<String>` — `split_once('/')`,
  returning `Some(namespace)` for `ns/name` ids and `None` for
  top-level (un-namespaced) models like `bert-base-uncased`.

### Non-test production consumers

`search_models` and `get_model` are user-facing API entry points; the
ferrotorch crate ecosystem itself does not transitively call them
(the static registry is sufficient for shipped models). The flat
re-exports in `lib.rs::pub use discovery::{...}` make them available
through the meta-crate `ferrotorch::hub::*` glob. The
`SearchQuery::to_query_string()` builder is exercised by every
discovery call via the production fns `search_models`. Downstream
CLI tools and notebooks instantiate `SearchQuery::new()` directly
through the re-export.

The intra-module non-test production consumer chain is:

- `search_models` (pub) → `query.to_query_string()` (pub(crate)) →
  `url_encode` (private) and → `populate_authors` (private) →
  `extract_author` (private).
- `get_model` (pub) → `extract_author` (private).
- Both → `crate::auth::with_auth` (in `auth.rs`, http-gated).

## Parity contract

`parity_ops = []`. The Hub API is an external HTTP contract; we
mirror what `https://huggingface.co/docs/hub/api` documents rather
than a PyTorch source file. The conceptual upstream is
`torch.hub.list(github, ...)` (`torch/hub.py:459`) but that lists
hubconf entrypoints in a GitHub repo, not models on a model index —
the abstraction is different enough that this module is best
understood as a HuggingFace-Hub-shaped extension layer with no direct
PyTorch counterpart (R-DEV-3: this is an on-the-wire HF API contract,
and we mirror it; deviation only if HF documents one).

Edge cases:

- Hub endpoint returns `modelId` vs `id`: handled by serde alias.
- Hub returns minimal `{"modelId": ...}` with no other fields: every
  optional field has `#[serde(default)]` so deserialization succeeds.
- Hub adds new fields: ignored via the default serde behaviour
  (unknown fields are dropped).
- `author` absent in response: populated from the leading segment of
  `model_id` so downstream consumers can rely on the field being set
  for namespaced repos.
- Empty `repo_id`: rejected before HTTP via explicit guard.
- Top-level repos (no `/` in `model_id`): `extract_author` returns
  `None`; downstream code reads it as `Option<String>` and handles
  the absence.

## Verification

Tests in `mod tests in discovery.rs` (all `#[cfg(all(test, feature = "http"))]`,
13 tests):

- Query builder: `test_search_query_empty_is_bare_endpoint`,
  `test_search_query_search_only`,
  `test_search_query_all_fields`.
- URL encoding: `test_url_encode_alphanumeric_passthrough`,
  `test_url_encode_special_chars`.
- Author extraction: `test_extract_author_namespaced`,
  `test_extract_author_top_level`.
- Deserialization: `test_deserialize_model_summary_minimal`,
  `test_deserialize_model_summary_id_alias`,
  `test_deserialize_model_summary_full`,
  `test_deserialize_model_summary_unknown_fields_ignored`,
  `test_deserialize_model_info_with_siblings`,
  `test_populate_authors_fills_missing`.
- API guard: `test_get_model_empty_repo_id_errors`.

Actual live network calls are NOT exercised in the unit-test gauntlet
because they would be flaky and require external connectivity. The
`search_models` and `get_model` happy paths are reachable through
integration tests behind feature gates; the unit suite exercises
every fn except the two `.call()`-issuing wrappers themselves.

Smoke command:

```bash
cargo test -p ferrotorch-hub --lib discovery:: 2>&1 | tail -3
```

Expected: 13 passed.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct HfModelSummary` in `discovery.rs` with the serde `rename`/`alias`/`default` annotations; non-test consumer: `pub use discovery::HfModelSummary` in `lib.rs` re-exports it; it is the return-element type of `pub fn search_models` which is the production entry point for the entire flat re-export surface (used by downstream notebooks / CLI tools via `ferrotorch_hub::HfModelSummary`). |
| REQ-2 | SHIPPED | impl: `pub struct HfModelInfo` and `pub struct HfRepoFile` in `discovery.rs`; non-test consumer: `pub use discovery::{HfModelInfo, HfRepoFile}` in `lib.rs`; returned by `pub fn get_model` which is the production entry point for per-repo lookup. |
| REQ-3 | SHIPPED | impl: `pub struct SearchQuery` + `pub fn new` + builder methods + `pub(crate) fn to_query_string` in `discovery.rs`; non-test consumer: `discovery.rs::search_models` calls `query.to_query_string()` on every search request — that is the in-crate production consumer; the `pub use discovery::SearchQuery` in `lib.rs` exposes the struct so downstream callers build instances. |
| REQ-4 | SHIPPED | impl: `pub fn search_models` in `discovery.rs` doing URL build → auth-decorated GET → into_json → `populate_authors`; non-test consumer: `pub use discovery::search_models` in `lib.rs` — search_models is the user-facing search entry point; the URL/auth/json plumbing is integration-tested through the broader workspace's network examples (e.g. `ferrotorch-llama/examples/llm_inference_dump.rs`'s discovery flow). |
| REQ-5 | SHIPPED | impl: `pub fn get_model` in `discovery.rs` with the empty-id guard + auth-decorated GET + json + author populate; non-test consumer: `pub use discovery::get_model` in `lib.rs` exposes it; it is the lookup entry point for `repo_id → HfModelInfo` lookups during model selection. |
| REQ-6 | SHIPPED | impl: `fn url_encode` (private) in `discovery.rs` doing the RFC 3986 unreserved-subset percent-encoding; non-test consumer: `SearchQuery::to_query_string` (pub(crate)) calls `url_encode` for every string-valued query parameter, and `to_query_string` is in turn called by `search_models` on every search request. |
