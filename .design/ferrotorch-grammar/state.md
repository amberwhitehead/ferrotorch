# ferrotorch-grammar — `state` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - (none — ferrotorch-native; PyTorch has no JSON value state-machine.
    The contract is the JSON Draft 7 grammar at https://www.json.org/
    and RFC 8259, restricted to the compact, escape-free, exponent-
    free subset documented below.)
-->

## Summary

`ferrotorch-grammar/src/state.rs` implements `JsonGrammar`, a
character-level state machine over a partially-emitted JSON value
matching a `Schema`. At every point during generation it reports
`valid_next_chars()` (the set of bytes the constrained decoder is
allowed to emit) and `is_complete()`. The module also exposes a family
of `*EmissionStage` enums (`BooleanEmissionStage`,
`IntegerEmissionStage`, `NumberEmissionStage`, `StringEmissionStage`,
`StringEnumEmissionStage`, `NullEmissionStage`, `NullableEmissionStage`,
`ObjectKeyEmissionStage`) that the GPU dispatcher in
`gpu_dispatch.rs` reads to compile a per-state DFA. There is no
PyTorch upstream.

## Requirements

- REQ-1: A `pub struct JsonGrammar` holding a stack of frames
  (`Vec<Frame>` where `Frame { schema: Schema, phase: Phase }`) plus a
  `done: bool` flag. The internal `Phase` enum is private but tracks
  per-frame progress through every supported `Schema` variant
  (string body, number digits, object key, etc.).

- REQ-2: `pub fn JsonGrammar::valid_next_chars(&self) -> Vec<char>`
  returns the set of single-byte characters that may legally be
  emitted next. Empty vector when complete. The implementation
  inspects the top-of-stack frame and (if a parent exists) the
  parent's terminator set via `parent_terminators`.

- REQ-3: `pub fn JsonGrammar::step_char(&mut self, c: char) ->
  Result<(), StepError>` advances the state by one char, returning
  `Err` (state unchanged) if `c` is not in the current valid-next-
  chars set. The error taxonomy is `StepError::UnexpectedChar`,
  `AlreadyComplete`, `Unsupported(&'static str)`.
  `pub fn JsonGrammar::step_str` is a convenience wrapper.

- REQ-4: Per-schema state-machine correctness covering every variant
  in `Schema`:
    - `Boolean` / `Null`: literal walks (`true`, `false`, `null`)
      via `Phase::Literal { remaining }`.
    - `Number` / `Integer`: `Phase::NumberDigits` with five booleans
      (`had_sign`, `had_digits`, `had_decimal`, `had_fractional_digit`,
      `is_zero_only`) enforcing JSON's no-leading-zero rule and the
      "digit required after `.`" rule.
    - `String`: `Phase::StringChars { partial, allowed: None }`
      accepting any printable ASCII except `"` and `\\`.
    - `StringEnum`: `Phase::StringChars` with `allowed: Some(values)`
      enforcing the closed value set via prefix matching.
    - `Object`: 6 phases (`Start`, `ObjectFreshOpen`,
      `ObjectExpectKey`, `ObjectKey { partial, keys_seen,
      candidates }`, `ObjectColon`, `ObjectAfterValue`). `required`
      keys cannot be skipped via `}`; duplicate keys cannot reappear.
    - `Array`: 3 phases (`Start`, `ArrayFreshOpen`, `ArrayAfterValue`)
      with per-element frames pushed onto the stack.
    - `Nullable(inner)`: dispatches to `Schema::Null` literal mode on
      `n`, switches `frame.schema = inner; phase = Start` and re-
      dispatches the char on any other valid leading byte.

- REQ-5: NOT-STARTED — JSON string escape sequences (`\"`, `\n`,
  `\\`, `\uXXXX`) are intentionally rejected by the current grammar
  (`StepError::UnexpectedChar` when `\\` is observed inside
  `StringChars`). Tracked by blocker #1488.

- REQ-6: NOT-STARTED — JSON number exponents (`1e5`, `-3.14E+2`) are
  not accepted; only optional sign, integer part, optional `.`
  fractional part. Tracked by blocker #1489.

- REQ-7: NOT-STARTED — the grammar emits compact JSON only (no
  whitespace allowed between tokens). A whitespace-permissive mode is
  a follow-up. Tracked by blocker #1490.

- REQ-8: A family of `pub enum *EmissionStage` types (`BooleanEmissionStage`,
  `NullEmissionStage`, `IntegerEmissionStage`, `NumberEmissionStage`,
  `StringEmissionStage`, `StringEnumEmissionStage<'a>`,
  `NullableEmissionStage<'a>`, `ObjectKeyEmissionStage<'a>`) plus the
  matching `pub fn JsonGrammar::*_emission_stage` and `*_emission_stage_top`
  accessors that surface the *minimal* state slice the GPU dispatcher
  needs to compile a DFA. The `_top` variants accept multi-frame
  grammars (e.g. scalar inside an Array element); the non-`_top`
  variants are single-frame-only. A separate `pub fn
  top_frame_parent_terminators(&self) -> Vec<char>` accessor surfaces
  the parent's terminator-char set for the nested-scalar dispatch.

## Acceptance Criteria

- [x] AC-1: `pub struct JsonGrammar { frames, done }` (private fields)
  with `pub fn new(schema: Schema)`, `is_complete`, `valid_next_chars`,
  `step_char`, `step_str`.
- [x] AC-2: `pub enum StepError` with three variants
  (`UnexpectedChar { got, expected }`, `AlreadyComplete`,
  `Unsupported(&'static str)`), `thiserror::Error`-derived,
  `#[non_exhaustive]`.
- [x] AC-3: JSON-leading-zero rule pinned (`is_zero_only` flag in
  `Phase::NumberDigits`); JSON-digit-after-decimal pinned
  (`mid_decimal = had_decimal && !had_fractional_digit` branch in
  `valid_next_chars_for`).
- [x] AC-4: 18 unit tests in `mod tests` exercise every `Schema`
  variant — `empty_object_round_trip`, `boolean_true_and_false`,
  `null_literal`, `integer_round_trip`, `negative_number`,
  `string_round_trip`, `string_enum_round_trip`, `string_enum_rejects_invalid_prefix`,
  `object_with_required_field`, `object_rejects_unknown_key`,
  `object_rejects_duplicate_key`, `array_of_numbers`, `empty_array`,
  `nested_object`, `nullable_string`, `rejects_string_escape`,
  `rejects_emitting_after_complete`.
- [x] AC-5: 8 `*_emission_stage` accessors return `Some(_)` only for
  single-frame grammars; 6 `*_emission_stage_top` accessors handle
  multi-frame.
- [x] AC-6: `pub fn top_frame_parent_terminators` surfaces the
  parent's terminator chars (e.g. `[',', ']']` for Array,
  `[',', '}']` for Object after-value) for GPU multi-frame dispatch.
- [ ] AC-7: string escapes accepted — blocker #1488.
- [ ] AC-8: number exponents accepted — blocker #1489.
- [ ] AC-9: whitespace-permissive mode — blocker #1490.

## Architecture

### `JsonGrammar` core (REQ-1)

`pub struct JsonGrammar { frames: Vec<Frame>, done: bool }` in
`state.rs`. The stack-of-frames structure makes nesting natural:
when an Object's value is being emitted, a child `Frame { schema:
prop_schema, phase: Phase::Start }` is pushed onto the stack; when
the value completes, the frame is popped and the parent's
`ObjectAfterValue` phase becomes active. The same scheme handles
Array elements.

`pub fn new(schema)` seeds a single frame; `pub fn is_complete` simply
returns `self.done`; the meat is in `step_char`, `apply_step`,
`valid_next_chars`, and `valid_next_chars_for`.

### `valid_next_chars` + `valid_next_chars_for` (REQ-2)

The top-of-stack frame's `(schema, phase)` pair selects an arm in
`valid_next_chars_for`. Each arm builds the set of legal next chars
from the current state:

- **String body**: every printable ASCII byte except `"` and `\\`,
  plus the closing `"`. The `\\` exclusion is the load-bearing
  decision behind REQ-5's NOT-STARTED — without escape support, the
  grammar must reject `\\` to keep the JSON output valid.
- **StringEnum body**: a `BTreeSet<char>` of chars that extend
  `partial` toward at least one allowed value, plus `"` if `partial`
  itself is a complete value. Stable iteration order.
- **NumberDigits**: digits (with leading-zero suppression via
  `is_zero_only`), optionally `.` (if `had_digits && !had_decimal`),
  optionally `parent_terminators(parent)` (if `had_digits &&
  !mid_decimal`). This is where the multi-frame plumbing matters
  most: a top-level integer (no parent) has no terminator chars in
  its set, but an Array element inherits `[',', ']']`.
- **Object phases**: hand-tuned per-phase (e.g. `ObjectFreshOpen`
  allows `"` iff some property hasn't been seen; `}` iff every
  `required` key has been seen).
- **Array phases**: `ArrayFreshOpen` allows the value-start chars of
  the element schema plus `]`; `ArrayAfterValue` allows `,` and `]`.
- **Nullable(inner) at Start**: the union of the inner schema's
  start-set plus `'n'` for the null branch; deduped and sorted.

### `step_char` + `apply_step` (REQ-3)

`step_char` validates `c` against `valid_next_chars`, returning
`StepError::UnexpectedChar` if `c` isn't in the set, then delegates
to the private `apply_step(c)` which mutates the top-frame and (if
the value is now complete) pops the frame and calls
`bubble_value_done`. The `apply_step` body is a big `match` on
`(&frame.schema.clone(), &frame.phase.clone())`. The clones avoid a
borrow conflict where the match arm needs to mutate `frame.phase`
while still pattern-matching the old phase.

A subtle multi-frame transition: when a `Schema::Number` /
`Schema::Integer` frame is "ended by some non-digit" (typically a
parent's terminator like `,`), the frame is popped and the char is
re-dispatched via `return self.apply_step(c)`. This means a single
`step_char('}')` call may walk the stack down multiple frames
without recursing through the user-facing API.

### `Phase::NumberDigits` invariants (REQ-4)

Five booleans guard JSON's number rules:

- `had_sign`: a leading `-` was emitted.
- `had_digits`: at least one digit has been emitted.
- `had_decimal`: a `.` has been emitted.
- `had_fractional_digit`: at least one digit was emitted *after* `.`.
- `is_zero_only`: the first digit was `0` AND nothing else has been
  emitted yet.

The combinations that `valid_next_chars_for` cares about:
`mid_decimal = had_decimal && !had_fractional_digit` (only digits
allowed, no terminator); `is_zero_only` (no more digits, only `.`
or terminator); etc. The `negative_number` and `integer_round_trip`
tests pin these.

### `Nullable(inner)` dispatch (REQ-4)

Phase::Start on a `Schema::Nullable(inner)` frame has two outgoing
arms in `apply_step`:

- If `c == 'n'`, **rewrite the frame in place**: `frame.schema =
  Schema::Null; phase = Phase::Literal { remaining: "ull" }`. The
  null branch is now committed.
- Otherwise, **inline the inner schema**: `frame.schema = inner;
  phase = Phase::Start`, then `return self.apply_step(c)`. The inner
  branch is now committed and the same char that triggered the
  commit gets re-dispatched to the inner schema's Start arm.

This is the cleanest way to encode "Nullable commits to a branch on
the first emitted char" without spawning a second frame.

### NOT-STARTED gaps (REQ-5, REQ-6, REQ-7)

- **String escapes (REQ-5, blocker #1488)**: `\\` is rejected via
  the body-char filter `(0x20..=0x7E).filter(|b| *b != b'"' && *b
  != b'\\')`. Test `rejects_string_escape` pins the rejection.
- **Number exponents (REQ-6, blocker #1489)**: `valid_next_chars_for`
  for `NumberDigits` does not include `e` or `E`; emitting them
  would pop the number frame and re-dispatch. The `negative_number`
  test only exercises decimal, not exponent.
- **Whitespace (REQ-7, blocker #1490)**: every phase emits compact
  JSON. No `' '` / `'\t'` / `'\n'` accepted at structural
  boundaries. Tests `extraction_response_shaped_schema_step_by_step`
  in `json_schema.rs` walk the explicit compact form
  `{"confidence":"high","value":-3.14}` — adding whitespace would
  break.

### `*EmissionStage` accessors (REQ-8)

There are 8 emission-stage enums plus 14 accessor methods on
`JsonGrammar`. Each accessor inspects the current grammar state
and returns `Some(stage)` only when the top frame is the matching
`Schema` variant in a state the GPU DFA compiler can handle; else
`None`, signalling the caller to fall through to the CPU
`compute_mask` loop.

The single-frame accessors (`boolean_emission_stage`,
`null_emission_stage`, `integer_emission_stage`,
`number_emission_stage`, `string_emission_stage`,
`string_enum_emission_stage`, `nullable_emission_stage`,
`object_key_emission_stage`) are the older path; they return `None`
on multi-frame grammars. The `_top` variants
(`boolean_emission_stage_top`, etc.) accept multi-frame and are
paired with `top_frame_parent_terminators` for nested-scalar
dispatch.

Notable inheritance: `ObjectKeyEmissionStage { partial, candidates }`
borrows from the grammar's own `Phase::ObjectKey` payload (no clone)
so the GPU dispatcher pays zero allocation when probing.

### Non-test production consumers

- `pub struct JsonGrammar` is reachable via
  `ferrotorch_grammar::JsonGrammar` (re-exported in `lib.rs:21`).
  Boundary public API per goal.md S5.
- `JsonSchemaProcessor::compute_mask` in `json_schema.rs` clones the
  `JsonGrammar`, walks each token's chars via `step_char`, and uses
  the `Ok`/`Err` result to set the mask bit. The whole non-CUDA
  decoding path runs through this code.
- `JsonSchemaProcessor::step_token` advances the wrapped grammar
  via `step_char` for each char in the chosen token.
- `gpu_dispatch::compute_mask_gpu` calls every `*_emission_stage_top`
  accessor + `top_frame_parent_terminators` to compile the DFA on
  the host.
- `pub enum BooleanEmissionStage` is reachable via
  `ferrotorch_grammar::BooleanEmissionStage` (re-exported in
  `lib.rs:21`). The other `*EmissionStage` enums are
  `pub` but not re-exported — they are intra-crate consumers of the
  `gpu_dispatch` module, which imports them with `use
  super::state::{BooleanEmissionStage, IntegerEmissionStage,
  NullEmissionStage, NullableEmissionStage, NumberEmissionStage,
  ObjectKeyEmissionStage, StringEmissionStage,
  StringEnumEmissionStage};` (`gpu_dispatch.rs:22-24`).

## Parity contract

`parity_ops = []`. The contract is the JSON Draft 7 spec + RFC 8259,
restricted to the compact, escape-free, exponent-free subset
documented in the module header.

Edge cases pinned by unit tests (all in `mod tests`):

- **No emit after complete**: `step_char` after `is_complete` ⇒
  `StepError::AlreadyComplete` (`rejects_emitting_after_complete`).
- **Boolean literal walks**: both `true` and `false`
  (`boolean_true_and_false`).
- **Null literal walk**: `null` (`null_literal`).
- **Number leading sign**: `-` ⇒ digits required next
  (`negative_number`).
- **JSON-leading-zero forbidden**: after `0`, no further digits valid
  (the `AfterZero` state in `IntegerEmissionStage`; CPU side enforces
  via `is_zero_only` flag).
- **JSON digit-after-decimal required**: `1.` is invalid (mid-decimal
  ⇒ only digits valid until `had_fractional_digit`).
- **String body excludes `"` and `\\`**: `rejects_string_escape`
  pins the `\\` rejection; `string_round_trip` shows `"` ends the
  string.
- **StringEnum prefix matching**: `string_enum_rejects_invalid_prefix`
  pins the closed-set rule; `string_enum_round_trip` exercises the
  positive path.
- **Object required-key enforcement**: `object_with_required_field`
  pins "can't `}` until all required seen".
- **Object unknown-key rejection**: `object_rejects_unknown_key`.
- **Object duplicate-key rejection**: `object_rejects_duplicate_key`.
- **Array empty + multi-element**: `empty_array` + `array_of_numbers`
  (`[1,2.5,3]`).
- **Nested object**: `nested_object` (`{"inner":{"v":true}}`).
- **Nullable both branches**: `nullable_string` exercises both
  `null` and `"hi"`.

## Verification

Tests in `mod tests` of `state.rs` (18 tests, all currently passing):

```bash
cargo test -p ferrotorch-grammar --lib state:: 2>&1 | tail -3
```

Expected: `18 passed; 0 failed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct JsonGrammar` with private `frames: Vec<Frame>` + `done: bool`, `pub fn new(schema)`, `is_complete` in `state.rs`; non-test consumer: `JsonSchemaProcessor::new` in `json_schema.rs` calls `JsonGrammar::new(schema)` and stores it as `grammar: JsonGrammar`. |
| REQ-2 | SHIPPED | impl: `pub fn JsonGrammar::valid_next_chars` walking the top frame via `valid_next_chars_for` + parent terminators in `state.rs`; non-test consumer: `JsonSchemaProcessor::compute_mask` in `json_schema.rs` indirectly drives it through `step_char`'s pre-validation; `gpu_dispatch::compute_mask_gpu` calls `grammar.top_frame_parent_terminators()` in `gpu_dispatch.rs`. |
| REQ-3 | SHIPPED | impl: `pub fn JsonGrammar::step_char` validates against `valid_next_chars` then dispatches to private `apply_step` in `state.rs`; `pub fn step_str` is the convenience wrapper; non-test consumer: `JsonSchemaProcessor::compute_mask` calls `probe.step_char(c)` in a per-token loop (`json_schema.rs`) and `JsonSchemaProcessor::step_token` calls it per token-char to commit. |
| REQ-4 | SHIPPED | impl: `apply_step` covers every `(Schema, Phase)` pair in `state.rs`; `valid_next_chars_for` mirrors with the legal-chars side; non-test consumer: every production `compute_mask` / `step_token` call in `json_schema.rs` walks these arms. |
| REQ-5 | NOT-STARTED | string-body filter excludes `\\` in `valid_next_chars_for` `(Schema::String, Phase::StringChars)` arm in `state.rs`. Open prereq blocker #1488 — needs `\\`-handling sub-phase and `\\uXXXX` codepoint accumulator. |
| REQ-6 | NOT-STARTED | `valid_next_chars_for` `(Schema::Number, Phase::NumberDigits)` arm in `state.rs` does not emit `e` / `E`; the parser would pop the number frame on emit. Open prereq blocker #1489 — needs `had_exponent_marker` + `had_exponent_sign` + `had_exponent_digit` flags. |
| REQ-7 | NOT-STARTED | no whitespace permitted at structural boundaries; tests in `json_schema.rs` walk the explicit compact form. Open prereq blocker #1490 — needs a `permissive_whitespace: bool` flag on `JsonGrammar::new` that injects `[' ', '\t', '\n']` into structural-boundary `valid_next_chars` outputs. |
| REQ-8 | SHIPPED | impl: 8 `pub enum *EmissionStage` types + 14 `pub fn JsonGrammar::*_emission_stage{,_top}` accessors + `pub fn top_frame_parent_terminators` in `state.rs`; non-test consumer: `compute_mask_gpu` in `gpu_dispatch.rs` calls `grammar.boolean_emission_stage_top()`, `null_emission_stage_top()`, `integer_emission_stage_top()`, `number_emission_stage_top()`, `string_emission_stage_top()`, `string_enum_emission_stage_top()`, `nullable_emission_stage()`, `object_key_emission_stage()`, `top_frame_parent_terminators()` in production. |
