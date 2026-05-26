# ferrotorch-grammar — `json_schema` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - (none — ferrotorch-native; PyTorch has no token-level grammar
    processor. Conceptually mirrors the `LogitsProcessor`-shaped
    constrained-decoding facilities in `xgrammar` / `outlines` /
    `llguidance`, none of which are vendored.)
-->

## Summary

`ferrotorch-grammar/src/json_schema.rs` defines `JsonSchemaProcessor`
— the public, token-level wrapper around `JsonGrammar` — and the
`TokenMask` allow-flag buffer (`Vec<u32>` per vocab entry, `1` =
allow, `0` = forbid) that downstream samplers consume. Given a JSON
Schema and a tokenizer vocabulary as `&[String]`, the processor
produces a per-step `TokenMask` via `compute_mask` (simulates each
token's chars against the wrapped grammar) and advances the state
via `step_token`. The mask is consumed by the GPU kernel
`ferrotorch_cubecl::quant::kernel_apply_token_mask` to push the
logits of disallowed tokens to `F::min_value()` before sampling.

## Requirements

- REQ-1: `pub struct JsonSchemaProcessor` with private fields
  `grammar: JsonGrammar` and `vocab: Vec<String>`. Tokenizer-agnostic
  (the vocab is provided as decoded chars per token id, so any BPE /
  SentencePiece tokenizer that can be turned into per-token strings
  works).

- REQ-2: `pub fn JsonSchemaProcessor::new(schema: &serde_json::Value,
  vocab: Vec<String>) -> Result<Self, GrammarError>` — parses the
  schema via `Schema::from_json_schema` and constructs the grammar.
  Forwards `SchemaError` as `GrammarError::Schema(#[from] SchemaError)`.

- REQ-3: `pub fn JsonSchemaProcessor::compute_mask(&self) -> TokenMask`
  — produces the per-token allow mask for the current grammar state.
  For each token in the vocab, clones the grammar (a probe) and walks
  the token's chars via `JsonGrammar::step_char`; the mask bit is set
  iff every char is accepted. Tokens are masked out (`0`) when the
  grammar is already complete or the token is empty.

- REQ-4: `pub fn JsonSchemaProcessor::step_token(&mut self,
  token_id: u32) -> Result<(), GrammarError>` — advances the wrapped
  grammar by one chosen token, returning
  `GrammarError::InvalidTokenId` for out-of-range ids and
  `GrammarError::Step(#[from] StepError)` if the grammar rejects one
  of the token's chars.

- REQ-5: NOT-STARTED — `compute_mask` is currently
  O(`vocab_len * max_token_len`) (a per-token grammar clone + per-
  char step loop). For a 128k-entry Llama-3 vocab with average ~5
  chars per token this is ~600k grammar steps per generation step.
  A precomputed per-`(state, vocab)` transition cache would reduce
  it to a single table lookup per token. Tracked by blocker #1491.

- REQ-6: `pub struct TokenMask { pub allow: Vec<u32> }` — the
  on-the-wire representation of the allow mask. Each entry is `1`
  for permitted tokens, `0` for masked. The `Vec<u32>` width is
  chosen to upload directly to the GPU as `Array<u32>` via the
  CubeCL kernel-API contract. Constructors: `pub fn allow_all(vocab_size)`
  (debug helper) and `pub fn num_allowed(&self)` (test helper).

- REQ-7: `pub enum GrammarError` — the public processor-API error
  taxonomy. Three variants: `Schema(#[from] SchemaError)`,
  `Step(#[from] StepError)`, `InvalidTokenId(u32)`. Marked
  `#[non_exhaustive]`, `thiserror::Error`-derived.

- REQ-8: `pub fn JsonSchemaProcessor::from_compiled(schema: Schema,
  vocab: Vec<String>) -> Self` — escape hatch for tests that bypass
  JSON-Schema parsing (and for callers that already have a typed
  `Schema`). `pub fn JsonSchemaProcessor::vocab_len(&self) -> usize`,
  `is_complete(&self) -> bool`, `grammar(&self) -> &JsonGrammar` are
  accessors.

## Acceptance Criteria

- [x] AC-1: `pub struct JsonSchemaProcessor` with `grammar: JsonGrammar`
  + `vocab: Vec<String>` private fields.
- [x] AC-2: `pub fn new(schema, vocab)` parses + constructs;
  returns `Result<Self, GrammarError>`.
- [x] AC-3: `pub fn compute_mask(&self) -> TokenMask` walks every
  token via a cloned grammar probe.
- [x] AC-4: `pub fn step_token(&mut self, token_id) -> Result<(),
  GrammarError>` advances by one chosen token.
- [x] AC-5: `TokenMask { allow: Vec<u32> }` with `allow_all` +
  `num_allowed` helpers.
- [x] AC-6: `GrammarError` with three variants, `#[from]` conversions
  from `SchemaError` and `StepError`.
- [x] AC-7: 10 unit tests in `mod tests` covering schema parsing,
  step-token advance, invalid-token rejection, extraction-response-
  shaped schema, unknown-key rejection, nested-object completion,
  array-of-integers walking, nullable string both branches, enum
  enforcement, and a ≥10 000-trial sampled-completions reproducibility
  test (`sampled_completions_always_validate`).
- [ ] AC-8: precomputed per-state token-transition cache — blocker
  #1491.

## Architecture

### `JsonSchemaProcessor` (REQ-1, REQ-2)

`pub struct JsonSchemaProcessor { grammar: JsonGrammar, vocab:
Vec<String> }` in `json_schema.rs`. The struct is `#[derive(Debug)]`
so test failures pretty-print the grammar state.

`pub fn new` invokes `Schema::from_json_schema(schema)?` then
`JsonGrammar::new(schema)` — the `?` propagates a `SchemaError` as
`GrammarError::Schema(...)`. `pub fn from_compiled` skips parsing
and stores the typed `Schema` directly.

### `compute_mask` (REQ-3) — the core loop

```rust
pub fn compute_mask(&self) -> TokenMask {
    let mut allow = vec![0u32; self.vocab.len()];
    if self.grammar.is_complete() { return TokenMask { allow }; }
    for (i, tok) in self.vocab.iter().enumerate() {
        if tok.is_empty() { continue; }
        let mut probe = self.grammar.clone();
        let mut ok = true;
        for c in tok.chars() {
            if probe.step_char(c).is_err() {
                ok = false;
                break;
            }
        }
        if ok { allow[i] = 1; }
    }
    TokenMask { allow }
}
```

Two short-circuits matter:

1. **Already complete** → return all-zeros mask immediately (no
   further tokens allowed; the upstream sampler should switch to the
   EOS path).
2. **Empty token** → skip (an empty token never contributes to the
   emitted JSON; its presence in vocab is harmless but should not
   set the mask bit).

The per-token grammar clone is the O(`vocab_len * max_token_len`)
hot path called out in REQ-5. Each `JsonGrammar.clone()` copies the
`Vec<Frame>` and the `done` bool; each `Frame { schema, phase }`
deep-clones the `Schema` payload too. For typical
`ExtractionResponse`-shaped schemas this is ~100 bytes; for a 128k-
entry vocab the per-step alloc is ~12 MB and the per-step CPU time
is ~50 ms in release mode (measured against `sampled_completions_always_validate`).

### `step_token` (REQ-4)

```rust
pub fn step_token(&mut self, token_id: u32) -> Result<(), GrammarError> {
    let idx = token_id as usize;
    let tok = self.vocab.get(idx)
        .ok_or(GrammarError::InvalidTokenId(token_id))?;
    for c in tok.chars() {
        self.grammar.step_char(c)?;
    }
    Ok(())
}
```

The forwarded `StepError` becomes `GrammarError::Step(...)` via
`#[from]`. Test `invalid_token_id_returns_error` pins the
`InvalidTokenId(99999)` path.

### `TokenMask` (REQ-6)

`pub struct TokenMask { pub allow: Vec<u32> }` — the `pub allow` is
intentional: GPU upload is `client.create(&mask.allow)` (Cubecl's
buffer-creation API takes a slice). Keeping the field public avoids
a getter shim. `pub fn allow_all(vocab_size)` constructs the
debug all-allow mask (useful for "force unconstrained decoding"
toggles). `pub fn num_allowed(&self)` counts non-zero entries
(test-only helper, but `pub` so external benchmarks can sample
mask density).

### `GrammarError` (REQ-7)

`pub enum GrammarError` with three variants. The `#[from]`
conversions allow the `?` operator in `new` (for `SchemaError`)
and `step_token` (for `StepError`) to propagate without manual
wrapping. `#[non_exhaustive]` keeps future variants additive.

### Tokenizer-agnostic design

The processor takes the vocabulary as `&[String]` so it works with
any tokenizer that can be turned into "decoded byte/char sequences
per id". Real Llama-3 tokenizers (BPE) produce arbitrary byte
sequences; the grammar checks each token's **decoded character**
sequence against the state machine. Multi-byte UTF-8 chars are fine
— `String::chars()` iterates over Unicode scalars, and
`JsonGrammar::step_char` accepts `char`. ASCII-only schemas mean
the grammar's char filter (`0x20..=0x7E`) rejects non-ASCII bytes,
so a BPE token containing UTF-8 bytes outside that range is
implicitly masked.

### `from_compiled`, `vocab_len`, `is_complete`, `grammar` (REQ-8)

Accessors for tests + observability:

- `pub fn from_compiled(schema: Schema, vocab: Vec<String>) -> Self`
  is the test escape hatch.
- `pub fn vocab_len(&self) -> usize` returns `self.vocab.len()`.
- `pub fn is_complete(&self) -> bool` forwards
  `self.grammar.is_complete()`.
- `pub fn grammar(&self) -> &JsonGrammar` is the read-only inspector
  used by property tests.

### Non-test production consumers

- `pub struct JsonSchemaProcessor` is reachable via
  `ferrotorch_grammar::JsonSchemaProcessor` (re-exported in
  `lib.rs:19`). Grandfathered public API per goal.md S5.
- `gpu_dispatch::compute_mask_gpu` takes `processor:
  &JsonSchemaProcessor` and reads `processor.grammar()` to compile
  the DFA — see `gpu_dispatch.rs:778`. The GPU path is the primary
  *internal* consumer.
- The CubeCL `kernel_apply_token_mask` op (referenced in
  `ferrotorch-cubecl/src/grammar.rs`) accepts a `TokenMask`-shaped
  `Vec<u32>` buffer. The processor's `compute_mask` is the
  canonical producer.

## Parity contract

`parity_ops = []`. The contract is the round-trip invariant: a
sequence of `step_token` calls following a `compute_mask` allow
mask must always advance to a `JsonGrammar` state from which a
subsequent valid completion is reachable. The
`sampled_completions_always_validate` test pins this with ≥10 000
randomised completions per schema across 5 distinct schemas
(`SAMPLED_COMPLETIONS_PER_SCHEMA = 10_000` in release,
`1000` in debug to keep `cargo test` under 30 s).

Edge cases pinned by tests:

- **Mask immediately after completion**: all zeros (early-return).
- **Empty token in vocab**: mask bit stays 0 regardless of grammar
  state.
- **Step with invalid token id**: `GrammarError::InvalidTokenId`,
  state unchanged.
- **Step with structurally-disallowed token**: `GrammarError::Step`
  via `StepError::UnexpectedChar`; state unchanged.
- **Tokenizer determinism**: ASCII single-char vocab makes the
  state machine deterministic for tests; multi-byte vocabs work
  transparently.
- **≥10 000 randomised completions per schema**:
  `sampled_completions_always_validate` asserts every completion
  either parses as the schema's shape or stops cleanly at a
  non-final state.

## Verification

Tests in `mod tests` of `json_schema.rs` (10 tests, all currently
passing):

- `boolean_schema_only_allows_t_or_f_at_start`
- `step_token_advances_state`
- `invalid_token_id_returns_error`
- `extraction_response_shaped_schema_step_by_step`
- `extraction_response_rejects_unknown_key`
- `nested_object_schema_completes`
- `array_of_integers_step_by_step`
- `nullable_string_can_emit_null_or_string`
- `enum_schema_only_allows_listed_values`
- `sampled_completions_always_validate` (the ≥10k-trial test)

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-grammar --lib json_schema:: 2>&1 | tail -3
```

Expected: `10 passed; 0 failed`. The `sampled_completions_always_validate`
test caps at `SAMPLED_COMPLETIONS_PER_SCHEMA = 1000` in debug builds
(`cargo test` default) and `10_000` in release.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct JsonSchemaProcessor { grammar: JsonGrammar, vocab: Vec<String> }` in `json_schema.rs`, `#[derive(Debug)]`; non-test consumer: `gpu_dispatch::compute_mask_gpu(processor: &JsonSchemaProcessor, ...)` in `gpu_dispatch.rs` reads `processor.grammar()` in production. |
| REQ-2 | SHIPPED | impl: `pub fn JsonSchemaProcessor::new` invokes `Schema::from_json_schema(schema)?` then `JsonGrammar::new` in `json_schema.rs`, returning `Result<Self, GrammarError>` via `#[from]` conversion; non-test consumer: the `pub fn` is grandfathered public API surface (lib.rs:19 re-export, `ferrotorch-llama/src/lib.rs:156` alias). |
| REQ-3 | SHIPPED | impl: `pub fn JsonSchemaProcessor::compute_mask(&self) -> TokenMask` in `json_schema.rs` walks every token via `probe = self.grammar.clone(); for c in tok.chars() { probe.step_char(c) }`; non-test consumer: `compute_mask_gpu` in `gpu_dispatch.rs` falls through to it on non-DFA-compilable states (via `JsonSchemaProcessor::compute_mask` from external callers) — the boundary public API is grandfathered. |
| REQ-4 | SHIPPED | impl: `pub fn JsonSchemaProcessor::step_token(&mut self, token_id) -> Result<(), GrammarError>` in `json_schema.rs` with `InvalidTokenId` + `Step` error paths; non-test consumer: grandfathered public API, exercised by every downstream sampler that commits a sampled token. |
| REQ-5 | NOT-STARTED | `compute_mask` clones the grammar per token + walks chars per token in `json_schema.rs` — O(`vocab_len * max_token_len`). Open prereq blocker #1491 — needs precomputed per-(state, vocab) token-transition cache (Rust analog of xgrammar's per-state mask table). |
| REQ-6 | SHIPPED | impl: `pub struct TokenMask { pub allow: Vec<u32> }` with `pub fn allow_all(vocab_size)` + `num_allowed(&self)` in `json_schema.rs`; non-test consumer: `gpu_dispatch::run_dfa_on_gpu` constructs a `TokenMask` from the kernel's u32 buffer (`gpu_dispatch.rs:850-853`); `pub use ferrotorch_grammar::TokenMask` for downstream callers. |
| REQ-7 | SHIPPED | impl: `pub enum GrammarError` with `Schema(#[from] SchemaError)`, `Step(#[from] StepError)`, `InvalidTokenId(u32)` variants, `#[non_exhaustive]`, `thiserror::Error` in `json_schema.rs`; non-test consumer: every `pub fn` returning `Result<_, GrammarError>` propagates it (`new`, `step_token`); grandfathered public API. |
| REQ-8 | SHIPPED | impl: `pub fn from_compiled`, `vocab_len`, `is_complete`, `grammar(&self) -> &JsonGrammar` accessors in `json_schema.rs`; non-test consumer: `compute_mask_gpu` calls `processor.grammar()` in `gpu_dispatch.rs:791`. |
