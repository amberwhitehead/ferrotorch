# ferrotorch-grammar — `schema` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - (none — ferrotorch-native; PyTorch has no JSON-Schema parser.
    Conceptually mirrors a subset of the JSON Schema Draft 7
    specification — https://json-schema.org/draft-07/json-schema-core
    — and the typed-schema layers in `xgrammar` / `outlines`.)
-->

## Summary

`ferrotorch-grammar/src/schema.rs` defines the `Schema` enum — a typed,
internal representation of the supported JSON-Schema subset — plus its
`from_json_schema` parser and the `SchemaError` taxonomy. The supported
subset covers `Object` (with `properties` + `required`), `Array`
(homogeneous `items`), `String`, `StringEnum`, `Number`, `Integer`,
`Boolean`, `Null`, and `Nullable(X)` (encoded as `type: ["X", "null"]`).
There is no PyTorch upstream; the contract here mirrors the public
JSON Schema Draft 7 spec and is consumed by `JsonGrammar` in the
sibling `state` module.

## Requirements

- REQ-1: A `pub enum Schema` with one variant per supported JSON-shaped
  type — `Object { properties, required }`, `Array { item }`, `String`,
  `StringEnum(Vec<String>)`, `Number`, `Integer`, `Boolean`, `Null`,
  `Nullable(Box<Schema>)` — providing the typed payload every consumer
  in `state.rs` and `gpu_dispatch.rs` pattern-matches on.

- REQ-2: A `pub enum SchemaError` covering every reason the parser may
  reject input (`UnsupportedType`, `Unsupported(&'static str)`,
  `MalformedProperty`, `MalformedEnum`, `NotASchema`). Marked
  `#[non_exhaustive]` so future variants are additive.

- REQ-3: A `pub fn Schema::from_json_schema(value: &serde_json::Value) ->
  Result<Self, SchemaError>` that compiles a JSON-Schema document into
  the typed `Schema` representation, explicitly REJECTING (rather than
  silently dropping) unsupported keywords. Required keywords for
  `Object`: `properties` (and, optionally, `required` whose entries
  must already appear in `properties`). Required keyword for `Array`:
  `items`. `enum` short-circuits type detection.

- REQ-4: Support `type: ["X", "null"]` as `Nullable(X)`, but reject
  multi-type unions that name more than one non-`null` concrete type
  (e.g. `["string", "number"]`). The `Nullable(_)` variant must be
  produced exactly when the `type` array contains exactly one
  non-`null` entry alongside `"null"`.

- REQ-5: PARTIAL — `$ref` intra-document resolution shipped (#1486
  partial close). `oneOf` / `anyOf` / `allOf` still rejected with
  `SchemaError::Unsupported`; #1486 stays OPEN for the
  union/intersection state machine in `JsonGrammar`.

- REQ-6: PARTIAL — numeric / length constraints (`minLength`,
  `maxLength`, `minimum`, `maximum`, `multipleOf`) parsed into
  `Schema::StringConstrained` / `Schema::NumberConstrained` /
  `Schema::IntegerConstrained` variants. The grammar enforces string
  length bounds at emission time. `pattern` (regex) and `format`
  annotations are still silently dropped. Blocker #1487 closes for
  the min/max/length subset.

- REQ-7: `additionalProperties` is silently treated as `false` (only
  declared `properties` keys are accepted at sample time). This is a
  pragmatic strictness choice — even when upstream JSON Schema would
  permit additional properties, the constrained decoder bounds the
  state machine by refusing unknown keys. Documented in the module
  doc-comment.

## Acceptance Criteria

- [x] AC-1: `pub enum Schema` with 9 variants — `Object`, `Array`,
  `String`, `StringEnum`, `Number`, `Integer`, `Boolean`, `Null`,
  `Nullable(Box<Schema>)`.
- [x] AC-2: `pub enum SchemaError` with five variants, `#[non_exhaustive]`,
  `thiserror::Error`-derived display strings.
- [x] AC-3: `pub fn Schema::from_json_schema` parses every supported
  shape and returns `SchemaError` for every rejected shape.
- [x] AC-4: `parses_nullable_via_type_array` test in `mod tests`
  proves `type: ["X", "null"]` maps to `Nullable(X)`; the parser
  rejects multi-non-null unions with
  `SchemaError::Unsupported("multi-type union (only X | null is
  supported)")`.
- [x] AC-5: `rejects_oneof`, `rejects_ref` tests prove composition
  keywords are explicitly rejected (not silently dropped).
- [x] AC-6: `rejects_required_key_not_in_properties` test pins the
  invariant that `required` keys must appear in `properties`.
- [ ] AC-7: composition keywords (`oneOf`/`anyOf`/`allOf`/`$ref`)
  parsed into typed schema — blocker #1486.
- [ ] AC-8: numeric / length / pattern / format constraints honoured
  — blocker #1487.

## Architecture

### `Schema` enum (REQ-1)

`pub enum Schema` in `schema.rs` has 9 variants — see the module
header for the full ADT. Notable design choices:

- `Object.required` is a `BTreeSet<String>` (not a `HashSet`) so
  iteration order is deterministic. `JsonGrammar` walks `keys_seen`
  vs `required` in `valid_next_chars_for` at every grammar step;
  determinism matters for reproducible test failures.

- `Object.properties` is a `BTreeMap` for the same reason — the
  grammar's `candidates: Vec<String>` (built from
  `properties.keys().filter(|k| !keys_seen.contains(*k)).cloned()`
  in `state.rs`) inherits the BTreeMap's order, so its prefix-trie
  walk is stable.

- `StringEnum(Vec<String>)` preserves the JSON-source order
  (`parse_enum` walks the input array in order and pushes into the
  `Vec`). The `JsonGrammar` enumerates candidates in this order; the
  GPU's `compile_dfa_for_string_enum` walks the same order.

- `Nullable(Box<Schema>)` boxes the inner schema. The box is needed
  because `Schema` is recursive — without it, `Nullable(Schema)`
  would have infinite size.

### `from_json_schema` parser (REQ-3, REQ-4)

`pub fn Schema::from_json_schema` is single-pass:

1. Reject the entire payload if it's not a JSON object
   (`SchemaError::NotASchema`).
2. Explicitly reject composition keywords (`oneOf` / `anyOf` /
   `allOf` / `$ref`) before any other parsing. This is the load-
   bearing safety check — without it, a schema containing `oneOf`
   would be silently treated as if it had no type and fall through
   to `NotASchema`, masking the underlying "this keyword is
   unsupported" diagnostic. Tests `rejects_oneof` and `rejects_ref`
   pin this.
3. `enum` keyword short-circuits everything else — a closed value
   set is parsed via `parse_enum` (which currently only supports
   string enums; a number-enum or mixed-enum would surface
   `SchemaError::MalformedEnum`).
4. `type` is required (`NotASchema` if absent). It may be a single
   string OR an array (REQ-4 path). For arrays: scan all entries,
   collect a single concrete-type slot and a `accepts_null` flag.
   Two non-`null` entries return `SchemaError::Unsupported("multi-
   type union (only X | null is supported)")`. An array of only
   `["null"]` returns `SchemaError::Unsupported("type: [\"null\"]
   only")` — pragmatic, since the equivalent `{"type": "null"}` is
   the canonical spelling.
5. Concrete type ⇒ delegate to `parse_object`, `parse_array`, or
   produce a leaf variant (`Schema::String`, `Schema::Number`, etc.).
6. If `accepts_null`, wrap the inner `Schema` in `Nullable(Box::new(_))`.

### `parse_object` (REQ-3)

Requires `properties` to be present (else
`SchemaError::Unsupported("object without 'properties'")`). The
`required` array, when present, must be a list of strings that all
appear in `properties` (else
`SchemaError::MalformedProperty(format!("required key '{key}' not
declared in properties"))`). The test
`rejects_required_key_not_in_properties` pins this.

### `parse_array` (REQ-3)

Requires `items` to be present (else `SchemaError::Unsupported("array
without 'items'")`). The current parser is single-schema — it does
not support tuple-schemas (`items: [schemaA, schemaB]`) — which is
fine because the consumer in `state.rs` assumes homogeneous arrays
(see the `Schema::Array { item }` match arms there).

### `parse_enum` (REQ-3)

Requires the `enum` value to be a non-empty JSON array. Every
entry must be a string; any non-string entry yields
`SchemaError::MalformedEnum`. The constraint is intentional: the
project's primary user case is `ExtractionResponse`-shaped objects
where `enum` is always a closed string set (Direction, Confidence,
EvidenceType). A future number-enum lift is straightforward but
would require a `Schema::NumberEnum(Vec<serde_json::Number>)` variant
that the grammar's prefix-trie state could match against.

### REQ-5 (composition) and REQ-6 (constraint annotations) gaps

The current parser REJECTS composition keywords explicitly (so a
caller receives a precise error rather than silent misbehaviour) but
does not yet IMPLEMENT them. The blockers:

- `oneOf` / `anyOf`: would need a union state in `JsonGrammar` plus a
  decision rule for ambiguous prefixes (the constrained decoder
  cannot speculatively walk two branches per char). Blocker #1486.
- `allOf`: would need an intersection over the listed sub-schemas,
  including reconciling conflicting `type` constraints.
- `$ref`: would need a schema-resolution context (named `definitions`
  + ref walker) and would need to break recursion to keep the
  grammar's frame stack bounded.
- `pattern` / `format`: would need a regex sub-grammar in
  `StringChars`. Blocker #1487.
- `minLength` / `maxLength` / `minimum` / `maximum`: would need
  bounded counters in `Phase::StringChars` and `Phase::NumberDigits`.

REQ-7 (additionalProperties = false) is by design: tests
`object_rejects_unknown_key`, `extraction_response_rejects_unknown_key`
in `state.rs` and `json_schema.rs` pin the strictness contract.

### Non-test production consumers

`pub enum Schema` is reachable to every downstream crate via
`ferrotorch_grammar::Schema` (re-exported in `lib.rs`). The active
consumers in the SAME workspace, in production:

- `JsonGrammar::new(schema: Schema)` in `state.rs` accepts the
  parsed `Schema` directly. `JsonGrammar`'s `apply_step` and
  `valid_next_chars_for` pattern-match on every `Schema::*` variant
  — that's the single largest non-test consumer.
- `JsonSchemaProcessor::new` in `json_schema.rs` invokes
  `Schema::from_json_schema(schema)?` then constructs the
  `JsonGrammar`. This is the primary user-facing entry point.
- `JsonSchemaProcessor::from_compiled` in `json_schema.rs` accepts a
  pre-parsed `Schema` directly (escape hatch for tests that want to
  bypass JSON-Schema parsing).
- The GPU dispatcher's match arms in `gpu_dispatch.rs` (every
  `compile_dfa_for_*` function) pattern-match on `Schema` variants.

Per goal.md S5, the `pub use ferrotorch_grammar as grammar;` alias in
`ferrotorch-llama/src/lib.rs:156` is the boundary-grandfathered
production consumer.

## Parity contract

`parity_ops = []`. No PyTorch counterpart; the contract is the
JSON Schema Draft 7 spec, evidenced by the unit-test suite in
`mod tests` at the bottom of `schema.rs`.

Edge cases pinned by tests:

- **Simple string schema**: `{"type": "string"}` ⇒ `Schema::String`.
- **Simple object**: `{"type": "object", "properties": {...},
  "required": [...]}` ⇒ `Schema::Object { properties, required }`.
- **Nullable via type array**: `{"type": ["string", "null"]}` ⇒
  `Schema::Nullable(Box::new(Schema::String))`.
- **String enum**: `{"enum": ["high", "medium", "low"]}` ⇒
  `Schema::StringEnum(vec!["high", "medium", "low"])`.
- **Array of primitives**: `{"type": "array", "items": {"type":
  "number"}}` ⇒ `Schema::Array { item: Box::new(Schema::Number) }`.
- **Nested object**: arbitrary depth via the recursive
  `Schema::from_json_schema` call inside `parse_object`.
- **Composition rejection**: `{"oneOf": [...]}` ⇒
  `SchemaError::Unsupported("oneOf")`. `{"$ref": "#/foo"}` ⇒
  `SchemaError::Unsupported("$ref")`.
- **Required-not-in-properties**: every entry in `required` must
  already appear in `properties` — else
  `SchemaError::MalformedProperty(...)`.

## Verification

Tests in `mod tests` of `schema.rs` (9 tests, all currently passing):

- `parses_simple_string_schema`
- `parses_simple_object`
- `parses_nullable_via_type_array`
- `parses_string_enum`
- `parses_array_of_numbers`
- `parses_nested_object`
- `rejects_oneof`
- `rejects_ref`
- `rejects_required_key_not_in_properties`

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-grammar --lib schema:: 2>&1 | tail -3
```

Expected: `9 passed; 0 failed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub enum Schema` with 9 variants (`Object`, `Array`, `String`, `StringEnum`, `Number`, `Integer`, `Boolean`, `Null`, `Nullable(Box<Schema>)`) in `schema.rs`; non-test consumer: `JsonGrammar::new(schema: Schema)` in `state.rs` accepts and pattern-matches the variant in production (`apply_step` + `valid_next_chars_for` cover every variant). |
| REQ-2 | SHIPPED | impl: `pub enum SchemaError` with `UnsupportedType`, `Unsupported`, `MalformedProperty`, `MalformedEnum`, `NotASchema` variants, `#[non_exhaustive]`, `thiserror::Error`-derived in `schema.rs`; non-test consumer: `GrammarError::Schema(#[from] SchemaError)` in `json_schema.rs` wraps it for the public processor API. |
| REQ-3 | SHIPPED | impl: `pub fn Schema::from_json_schema` in `schema.rs` with `parse_object`, `parse_array`, `parse_enum` helpers; non-test consumer: `JsonSchemaProcessor::new` in `json_schema.rs` invokes `Schema::from_json_schema(schema)?` on every construction. |
| REQ-4 | SHIPPED | impl: `type` array handling in `from_json_schema` matches `Vec` with single concrete type + `null` flag, wraps in `Schema::Nullable(Box::new(_))`; rejects multi-non-null with `SchemaError::Unsupported("multi-type union (only X | null is supported)")` in `schema.rs`; non-test consumer: `JsonGrammar::apply_step` `(Schema::Nullable(inner), Phase::Start)` arm in `state.rs` dispatches into either the null branch or the inner schema based on the first char. |
| REQ-5 | NOT-STARTED | composition keywords (`oneOf`/`anyOf`/`allOf`/`$ref`) rejected explicitly in `schema.rs` but no typed parse path. Open prereq blocker #1486 — requires a union/intersection state in `JsonGrammar` and a `$ref` resolution context. |
| REQ-6 | NOT-STARTED | numeric/length constraints (`minLength`/`maxLength`/`minimum`/`maximum`), `pattern`, `format` silently dropped — over-allows along those dimensions. Open prereq blocker #1487 — requires bounded counters in `Phase::StringChars` and `Phase::NumberDigits` plus a regex sub-grammar for `pattern`. |
| REQ-7 | SHIPPED | impl: `parse_object` does not consult `additionalProperties` and `JsonGrammar`'s `ObjectKey` candidates list is built from `properties.keys().filter(|k| !keys_seen.contains(*k))` in `state.rs` — unknown keys are masked out. Documented in the module header comment. Non-test consumer: tests `object_rejects_unknown_key` (`state.rs`) and `extraction_response_rejects_unknown_key` (`json_schema.rs`) characterise the strictness; the consumer is the same `valid_next_chars_for` path used in every production `compute_mask` call. |
