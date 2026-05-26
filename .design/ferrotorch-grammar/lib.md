# ferrotorch-grammar — crate root

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - (none — ferrotorch-native; PyTorch has no constrained-decoding /
    JSON-schema grammar module. Conceptually mirrors `xgrammar`,
    `outlines`, and Microsoft's `llguidance`, not anything in
    /home/doll/pytorch/torch/.)
-->

## Summary

`ferrotorch-grammar/src/lib.rs` is the crate root for the constrained-
decoding grammar processors. It declares the three always-on submodules
(`schema`, `state`, `json_schema`), the optional `gpu_dispatch`
submodule gated on the `cuda` feature, and re-exports the public surface
(`Schema`, `JsonGrammar`, `BooleanEmissionStage`, `JsonSchemaProcessor`,
`TokenMask`, `GrammarError`, and — under `cuda` — `PackedVocab`,
`compute_mask_gpu`). The crate has no upstream PyTorch equivalent; it
is a ferrotorch-native facility for producing per-step token-allow
masks that downstream samplers in `ferrotorch-llama` consume.

## Requirements

- REQ-1: Declare the three core submodules (`schema`, `state`,
  `json_schema`) unconditionally so non-CUDA builds expose the full
  CPU constrained-decoding API. `gpu_dispatch` is `#[cfg(feature =
  "cuda")]`-gated and only compiled when the `cuda` feature is on.

- REQ-2: Re-export the public surface so downstream crates
  (`ferrotorch-llama`, future `ferrotorch-mistral`, etc.) can `use
  ferrotorch_grammar::{JsonSchemaProcessor, TokenMask, Schema,
  JsonGrammar, GrammarError, BooleanEmissionStage}` without reaching
  into private submodules. Under `cuda`, additionally re-export
  `PackedVocab` and `compute_mask_gpu`.

- REQ-3: The `pub use ferrotorch_grammar as grammar;` alias in
  `ferrotorch-llama/src/lib.rs` is the active production consumer of
  this crate. Anything `pub` here is reachable to `ferrotorch-llama`
  via that alias (the v0.5.1 split documented in
  `ferrotorch-llama/src/lib.rs:152-156` preserved the
  `ferrotorch_llama::grammar` path for callers that imported the
  grammar from the old monolithic location).

## Acceptance Criteria

- [x] AC-1: `pub mod json_schema; pub mod schema; pub mod state;` are
  always compiled (lib.rs:12-14).
- [x] AC-2: `#[cfg(feature = "cuda")] pub mod gpu_dispatch;` gates the
  GPU module (lib.rs:16-17).
- [x] AC-3: `pub use json_schema::{GrammarError, JsonSchemaProcessor,
  TokenMask};` re-export (lib.rs:19).
- [x] AC-4: `pub use schema::Schema;` re-export (lib.rs:20).
- [x] AC-5: `pub use state::{BooleanEmissionStage, JsonGrammar};` re-export
  (lib.rs:21).
- [x] AC-6: `#[cfg(feature = "cuda")] pub use
  gpu_dispatch::{PackedVocab, compute_mask_gpu};` re-export
  (lib.rs:23-24).
- [x] AC-7: `ferrotorch-llama/src/lib.rs:156` re-exports the entire
  crate as `pub use ferrotorch_grammar as grammar;`, so the public
  surface declared here is reachable by every downstream model crate.

## Architecture

The crate root is intentionally minimal — three `pub mod`
declarations, one `#[cfg]`-gated module declaration, and four
re-export lines. There is no logic to translate from PyTorch because
the entire submodule tree is ferrotorch-native; PyTorch's
`torch/distributions/` ships parameterised distributions, but no
schema-driven token-allow facility (the closest equivalents are the
third-party Python libraries `xgrammar`, `outlines`, and Microsoft
Research's `llguidance`, none of which are vendored into the
ferrotorch tree).

### Module layout

- `schema` — internal `Schema` enum + `from_json_schema` parser for the
  supported JSON-Schema subset (Object, Array, String, StringEnum,
  Number, Integer, Boolean, Null, Nullable).
- `state` — `JsonGrammar` state-machine over a partially-emitted JSON
  value matching a `Schema`, plus the multiple
  `*EmissionStage` enums that the GPU dispatcher reads.
- `json_schema` — public `JsonSchemaProcessor`, which glues a
  `JsonGrammar` to a tokenizer vocabulary and produces per-step
  `TokenMask`s via `compute_mask`.
- `gpu_dispatch` — `cuda`-only DFA compilation + GPU dispatch path
  that returns `Some(TokenMask)` for DFA-compilable grammar states
  and `None` for fall-through-to-CPU cases.

### Non-test production consumer

`ferrotorch-llama/src/lib.rs:150-156`:

```
/// Re-export of [`ferrotorch_grammar`] for backward compatibility.
///
/// The constrained-decoding grammar processors used to live in
/// `ferrotorch_llama::grammar`; in v0.5.1 they were extracted into a
/// standalone [`ferrotorch_grammar`] crate. This alias keeps the old
/// `ferrotorch_llama::grammar::...` import paths working.
pub use ferrotorch_grammar as grammar;
```

This `pub use` is a NON-TEST production-level consumer: it is in
`ferrotorch-llama`'s top-level `lib.rs`, outside any `#[cfg(test)]`
block, and the entire grammar crate is reachable through it from
every downstream model crate that depends on `ferrotorch-llama`.
Per goal.md S5 ("Boundary methods ARE the public API; they don't need
further downstream callers to be SHIPPED"), this satisfies the
non-test-production-consumer requirement for every public re-export in
`lib.rs`.

## Parity contract

`parity_ops = []`. The crate has no parity-sweep op coverage because
PyTorch has no counterpart. Verification is the lib-test suite of
each submodule, which exercises CPU-side and (under `--features
cuda`) GPU-side allow-mask correctness directly.

## Verification

The crate's `lib.rs` itself is tested transitively by every test in
the four submodules: every test imports through
`ferrotorch_grammar::...` and therefore touches the re-export chain
declared here.

```bash
cargo test -p ferrotorch-grammar --lib 2>&1 | tail -3
```

Expected output of the form `test result: ok. N passed; 0 failed; 0
ignored`. No parity-sweep smoke is applicable (`parity_ops = []`).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub mod json_schema; pub mod schema; pub mod state;` + `#[cfg(feature = "cuda")] pub mod gpu_dispatch;` in `lib.rs`; non-test consumer: `pub use ferrotorch_grammar as grammar;` in `ferrotorch-llama/src/lib.rs:156` makes the submodule tree reachable to every downstream model crate (grandfathered public API per goal.md S5). |
| REQ-2 | SHIPPED | impl: `pub use json_schema::{GrammarError, JsonSchemaProcessor, TokenMask}; pub use schema::Schema; pub use state::{BooleanEmissionStage, JsonGrammar};` plus `#[cfg(feature = "cuda")] pub use gpu_dispatch::{PackedVocab, compute_mask_gpu};` in `lib.rs`; non-test consumer: `ferrotorch-llama/src/lib.rs:156` aliases the whole crate (S5 grandfathered). |
| REQ-3 | SHIPPED | impl: the re-export chain in `lib.rs`; non-test consumer: `pub use ferrotorch_grammar as grammar;` in `ferrotorch-llama/src/lib.rs:156` documented at lines 150-155 is the literal alias that makes the entire public surface a member of `ferrotorch_llama::grammar`. |
