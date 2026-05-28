# ferrotorch-grammar — `gpu_dispatch` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - (none — ferrotorch-native; PyTorch has no GPU-side constrained-
    decoding facility. The kernel side lives in
    ferrotorch-cubecl/src/grammar.rs and ferrotorch-cubecl/src/quant.rs;
    the upstream is the CubeCL kernel API surface, not PyTorch.)
-->

## Summary

`ferrotorch-grammar/src/gpu_dispatch.rs` is the host-side bridge
between `JsonGrammar` and the CubeCL kernel
`ferrotorch_cubecl::compute_token_mask_dfa_to_gpu`. The flow:

1. Inspect the current grammar via the `*_emission_stage_top`
   accessors exposed by `state.rs`.
2. If the state is DFA-compilable, build a `CompiledDfa` (transition
   table + char-class table + start/reject states + completion
   states) on the host.
3. Attach parent-frame terminator chars to the completion states
   (multi-frame support for scalars nested inside Object/Array).
4. Pack the processor's vocab as `(offsets, chars)` u32 buffers via
   `PackedVocab::pack`.
5. Dispatch the kernel; read back the allow mask as a `TokenMask`.
6. If the state isn't DFA-compilable, return `None` so callers fall
   through to the CPU `JsonSchemaProcessor::compute_mask` loop.

The module is gated on the `cuda` feature
(`#[cfg(feature = "cuda")] pub mod gpu_dispatch;` in `lib.rs`).

## Requirements

- REQ-1: A private `CompiledDfa { transitions, char_classes,
  num_classes, start_state, reject_state, complete_states }` host
  struct representing one DFA built from a grammar state. All
  buffers are owned `Vec<u32>`s because the kernel launcher takes
  them by reference and they need to outlive the launcher call.

- REQ-2: A family of `fn compile_dfa_for_*` constructors covering
  every `Schema` variant the dispatcher supports:
  `compile_dfa_for_boolean`, `compile_dfa_for_null`,
  `compile_dfa_for_integer`, `compile_dfa_for_number`,
  `compile_dfa_for_string`, `compile_dfa_for_string_enum`,
  `compile_dfa_for_object_key`, `compile_dfa_for_nullable`. Each
  consumes the matching `*EmissionStage` from `state.rs` and emits
  a `CompiledDfa`. `compile_linear_literal` is a helper that
  produces a DFA accepting any prefix of a fixed literal.

- REQ-3: A `fn add_terminators_to_states(dfa, terminators)` helper
  that splits the dfa's char-class table to give each terminator
  char its own class, then routes every `complete_state * terminator
  → popped_sink_state` transition. This is the multi-frame extension
  — when a scalar lives inside an Object property value or Array
  element, the parent contributes `,`, `}`, `]` terminator chars that
  legally end the value mid-token.

- REQ-4: A `fn merge_null_branch(inner)` helper that grafts a 4-state
  walk through `"null"` onto an inner-schema DFA's start state. This
  is how `Schema::Nullable(inner)` at `Phase::Start` is compiled —
  the inner DFA's start state gains a `class_n → walk_u → walk_l →
  walk_l2 → accept_null` chain on top of its existing transitions.

- REQ-5: A `pub struct PackedVocab { pub offsets: Vec<u32>, pub
  chars: Vec<u32>, pub max_token_len: u32 }` plus `pub fn PackedVocab::pack`
  constructor packing `vocab: &[String]` into CSR-like `(offsets,
  chars)` buffers (`offsets[i]..offsets[i+1]` is token `i`'s codepoint
  slice; one `u32` per Unicode scalar). The buffers are computed once
  per `(processor, vocab)` and cached on the call site since
  vocabularies are large (Llama-3 = 128k entries).

- REQ-6: A `pub fn compute_mask_gpu<R: Runtime>(processor: &JsonSchemaProcessor,
  client: &ComputeClient<R>, packed: &PackedVocab) -> Option<TokenMask>`
  — the public entry point. Returns `Some(TokenMask)` when the
  current grammar state is DFA-compilable; `None` otherwise (callers
  fall through to CPU). Iterates the supported emission-stage
  accessors and dispatches to the matching DFA compiler.

- REQ-7: NOT-STARTED — DFA dispatch for Object/Array structural
  phases (`ObjectFreshOpen`, `ObjectExpectKey`, `ObjectAfterValue`,
  `ObjectColon`, `ArrayFreshOpen`, `ArrayAfterValue`) is not
  implemented; they fall through to CPU. The structural phases are
  cheap on CPU (one comparison per char) so the kernel-launch
  overhead would dominate, but a future op-batching scheme could
  amortise that. Tracked by blocker #1492.

- REQ-8: NOT-STARTED — cross-boundary BPE under-allow. The kernel
  walks one token's chars and stops at the scalar's first
  parent-terminator char; it does NOT continue the walk into a new
  parent state. So a BPE token like `,"` (which would close the
  scalar, transition into `ObjectExpectKey`, and consume the next
  `"` legally) is conservatively rejected. CPU's `compute_mask`
  accepts the same token because `step_char` walks the stack
  across the boundary via `bubble_value_done` + re-dispatch. For
  ASCII single-char vocabularies this is byte-equal; for real BPE
  vocabs it's a known under-allow on rare cross-boundary
  structural tokens. Tracked by blocker #1493.

## Acceptance Criteria

- [x] AC-1: `struct CompiledDfa` with the 6-field shape above (impl
  in `gpu_dispatch.rs`).
- [x] AC-2: 7 `compile_dfa_for_*` constructors plus
  `compile_linear_literal` + `compile_boolean_full`.
- [x] AC-3: `add_terminators_to_states` + `split_class_for_char` +
  `merge_null_branch` helpers.
- [x] AC-4: `pub struct PackedVocab` with `pub fn pack(vocab: &[String])`
  + manual `Debug` impl that shows lengths (not the full 128k-entry
  vector).
- [x] AC-5: `pub fn compute_mask_gpu<R: Runtime>` entry point
  iterating every supported `*_emission_stage_top` accessor + the
  ObjectKey accessor + `top_frame_parent_terminators`.
- [x] AC-6: `fn run_dfa_on_gpu` builds `DfaMaskInputs`, dispatches
  the kernel, reads back the `Vec<u32>`, packs it as a `TokenMask`.
- [x] AC-7: 25+ CUDA runtime tests in `mod cuda_tests` (gated by
  `#[cfg(all(test, feature = "cuda"))]`) prove byte-equality vs CPU
  `compute_mask` for every supported stage.
- [ ] AC-8: Object/Array structural phases dispatch — blocker #1492.
- [ ] AC-9: Cross-boundary BPE tokens accepted — blocker #1493.

## Architecture

### `CompiledDfa` host struct (REQ-1)

```rust
struct CompiledDfa {
    transitions: Vec<u32>,        // num_states * num_classes flat row-major
    char_classes: Vec<u32>,       // [u8 -> class id] for ASCII range 0..128
    num_classes: u32,
    start_state: u32,
    reject_state: u32,
    complete_states: Vec<u32>,    // for multi-frame terminator attachment
}
```

The kernel walks each token's chars in lockstep: for each char,
look up the class (`char_classes[c as usize]` if c < 128, else
OTHER), then `state = transitions[state * num_classes + class]`.
If `state == reject_state` at any point, the token is masked out.
A token whose walk ends in `state != reject_state` is allowed.

`complete_states` is non-empty only when the wrapped schema has
states that are syntactically valid completion points (e.g. after
`"true"` for Boolean, after at least one digit for Integer).
Multi-frame dispatch uses this to know which states should accept
the parent's terminator chars (`,`, `}`, `]`). Single-frame
dispatch leaves it empty (no terminators apply).

### DFA compilers (REQ-2)

Each `fn compile_dfa_for_*` constructs the smallest DFA that
exactly matches the corresponding CPU `valid_next_chars_for` arm
in `state.rs`. The table sizes are:

- **Boolean**: 11 states × 9 classes (`t`/`r`/`u`/`e`/`f`/`a`/`l`/`s`/OTHER).
  Two completion states: state 4 (after `"true"`) and state 9 (after
  `"false"`). Linear-walk variants for `PartialTrue` and `PartialFalse`
  use the literal walker.
- **Null**: linear walk via `compile_linear_literal("null")` —
  5 states × 4 classes (n/u/l/OTHER).
- **Integer**: 5 states × 4 classes (`-`/`0`/`1..9`/OTHER).
  Completion at AfterZero and AfterDigits, NOT at AfterSign.
- **Number**: 7 states × 5 classes (adds `.` class).
  Completion at every digit-emitting state EXCEPT AfterDecimalNoFrac
  (which requires a fractional digit before the value can terminate).
- **String** (non-enum): 8 states × 8 classes mirroring the CPU
  `Schema::String` body + escape DFA. `\\` in the body opens a JSON
  string escape (state.md REQ-5 SHIPPED — escapes ARE accepted): the
  body steps into an escape-start state whose valid set is `" \ / b f
  n r t u`; the eight short escapes return to the body, `u` opens a
  four-hex-digit (`\uXXXX`) walk. Single completion state: after
  closing `"`. (The earlier "backslash always REJECT" model diverged
  from the CPU oracle — fixed under #1596.)
- **StringEnum**: prefix trie. Number of states = 1 (Phase::Start) +
  number of distinct prefixes in any value + 2 (closed + REJECT).
  Number of classes = number of distinct ASCII chars + 1 (`"`) + 1
  (OTHER). The trie node for a complete value gets `'"' → closed`.
- **ObjectKey**: structurally identical to StringEnum's InBody, but
  the value set is the still-unseen properties. Reuses the StringEnum
  compiler.
- **Nullable**: `compile_dfa_for_nullable` switches on the inner
  schema; if supported, compiles the inner DFA and grafts the null
  branch via `merge_null_branch`. Object / Array / nested Nullable
  inner schemas return `None` (fall through to CPU).

### `add_terminators_to_states` (REQ-3)

Multi-frame nested-scalar dispatch:

```rust
fn add_terminators_to_states(mut dfa: CompiledDfa, terminators: &[char]) -> CompiledDfa {
    if terminators.is_empty() || dfa.complete_states.is_empty() { return dfa; }
    let mut term_classes = ...;
    for &c in terminators {
        if (c as u32) < 128 {
            term_classes.push(split_class_for_char(&mut dfa, c as u8));
        }
    }
    // Append a single "popped" sink state.
    let popped = old_n as u32;
    let mut new_t = vec![reject_state; new_total * nc];
    new_t[..old_n * nc].copy_from_slice(&dfa.transitions);
    for &complete in &dfa.complete_states {
        for &cls in &term_classes {
            new_t[complete * nc + cls] = popped;
        }
    }
    dfa
}
```

The popped sink state rejects any further char — that's the REQ-8
under-allow: a multi-char token whose second char is a parent
terminator gets one step into the popped state, then rejects on the
next char. CPU's bubble-up re-dispatch handles this case correctly;
GPU under-allows.

### `split_class_for_char` (REQ-3)

The terminator chars must have dedicated char classes so the parent
can specifically route them to `popped`. If a terminator currently
shares a class with at least one other ASCII char (in `0..128`),
`split_class_for_char` introduces a new class, points the terminator
char at it, and copies the original column's transitions into the
new column for every state. Returns the new class id.

### `merge_null_branch` (REQ-4)

Graft `"null"`-walk onto inner's start state. Splits classes for
`n`, `u`, `l` so they're dedicated; adds 4 fresh states
(`walk_u`, `walk_l`, `walk_l2`, `accept_null`); sets up the linear
walk; appends `accept_null` to `inner.complete_states`.

### `PackedVocab` (REQ-5)

`pub struct PackedVocab` is a CSR-like sparse-matrix-of-codepoints
layout. `offsets[i] .. offsets[i+1]` is token `i`'s codepoint slice
(`u32` per Unicode scalar); `max_token_len` is the kernel's
bounded-loop cap. The manual `Debug` impl shows only lengths +
`max_token_len` because printing a 128k-entry `offsets` array makes
output unusable.

### `compute_mask_gpu` (REQ-6) — the public dispatcher

The accessor chain in priority order:

```rust
if let Some(stage) = grammar.object_key_emission_stage() {
    return run_dfa_on_gpu(client, packed,
        &compile_dfa_for_object_key(&stage)?);
}
let dfa = if let Some(stage) = grammar.boolean_emission_stage_top() {
    add_terminators_to_states(compile_dfa_for_boolean(&stage), &terminators)
} else if let Some(stage) = grammar.null_emission_stage_top() {
    add_terminators_to_states(compile_dfa_for_null(&stage), &terminators)
} else if let Some(stage) = grammar.integer_emission_stage_top() {
    add_terminators_to_states(compile_dfa_for_integer(&stage), &terminators)
} else if let Some(stage) = grammar.number_emission_stage_top() {
    add_terminators_to_states(compile_dfa_for_number(&stage), &terminators)
} else if let Some(stage) = grammar.string_emission_stage_top() {
    add_terminators_to_states(compile_dfa_for_string(&stage), &terminators)
} else if let Some((stage, values)) = grammar.string_enum_emission_stage_top() {
    add_terminators_to_states(compile_dfa_for_string_enum(&stage, values)?, &terminators)
} else if let Some(NullableEmissionStage::Start { inner }) = grammar.nullable_emission_stage() {
    add_terminators_to_states(compile_dfa_for_nullable(inner)?, &terminators)
} else {
    return None;
};
run_dfa_on_gpu(client, packed, &dfa)
```

ObjectKey is checked first because its DFA shape is self-contained
(no terminator attachment needed — the closing `"` is intrinsic).
Every other scalar runs through `add_terminators_to_states`; for
single-frame grammars the terminator list is empty and the helper
is a no-op.

### `run_dfa_on_gpu` (REQ-6 internal)

Builds `DfaMaskInputs::new(...)` (the CubeCL kernel-input struct
defined in `ferrotorch-cubecl`), dispatches via
`compute_token_mask_dfa_to_gpu::<R>`, reads back the `u32` allow
buffer with `client.read_one(handle)`, and packs it as a
`TokenMask`. Returns `None` if the device-side read returns the
wrong byte count (corrupt transfer).

### REQ-7 (object/array structural) and REQ-8 (cross-boundary BPE) gaps

The current dispatcher does NOT handle:

- `Phase::ObjectFreshOpen` (just emitted `{`) — needs DFA that
  accepts `"` (start a key) or `}` (close, only if all required
  satisfied).
- `Phase::ObjectExpectKey` (just emitted `,`) — needs DFA that
  only accepts `"`.
- `Phase::ObjectAfterValue` — needs DFA accepting `,` or `}`
  (depending on required-key satisfaction).
- `Phase::ObjectColon` — needs DFA accepting only `:`.
- `Phase::ArrayFreshOpen` — needs DFA accepting `]` or the element
  schema's start-set chars.
- `Phase::ArrayAfterValue` — needs DFA accepting `,` or `]`.

Test `unsupported_schema_returns_none` (gpu_dispatch.rs, in
`cuda_tests`) pins the fall-through-to-`None` for the Object
structural case. Blocker #1492 tracks the work.

Blocker #1493 tracks the cross-boundary BPE issue: a token like
`,"` is conservatively rejected by GPU, accepted by CPU. The fix
would require the kernel to walk into the parent's DFA after
hitting `popped`, OR pre-compose the per-state DFAs across stack
levels.

### Non-test production consumers

- `pub struct PackedVocab` and `pub fn compute_mask_gpu` are reachable
  via `ferrotorch_grammar::{PackedVocab, compute_mask_gpu}`
  (re-exported in `lib.rs` under the `cuda` feature).
- The `ferrotorch_cubecl::compute_token_mask_dfa_to_gpu` kernel-API
  contract is the upstream — see `ferrotorch-cubecl/src/grammar.rs`
  (`//! JsonSchemaProcessor::compute_mask` doc reference).
- The boundary-API consumer per goal.md S5 is the `pub use
  ferrotorch_grammar as grammar;` re-export in
  `ferrotorch-llama/src/lib.rs:156` — under the `cuda` feature,
  `PackedVocab` and `compute_mask_gpu` are reachable as
  `ferrotorch_llama::grammar::{PackedVocab, compute_mask_gpu}`.

## Parity contract

`parity_ops = []`. The contract is the CPU↔GPU byte-equality
invariant: for every DFA-compilable grammar state, the mask
produced by `compute_mask_gpu` MUST equal the mask produced by
`JsonSchemaProcessor::compute_mask` byte-for-byte over the SAME
vocab. The 25+ tests in `mod cuda_tests` (gated
`#[cfg(all(test, feature = "cuda"))]`) prove this for every stage.

Notable edge cases pinned:

- **Boolean@Start**: `compute_mask_gpu` must accept `t` and `f`
  among single-char tokens; reject everything else
  (`boolean_gpu_mask_matches_cpu_at_start`).
- **After `t` (PartialTrue)**: only `r` accepted
  (`boolean_gpu_mask_matches_cpu_after_t`).
- **Integer after `0`**: ZERO chars accepted at top level
  (`integer_gpu_mask_matches_cpu_after_zero`) — JSON forbids `01`.
- **Number mid-decimal**: only digits accepted; `.` specifically
  rejected (`number_gpu_mask_matches_cpu_mid_decimal`).
- **String body**: `\\` opens a JSON string escape and is therefore
  ALLOWED by the CPU oracle (state.md REQ-5); the GPU must match
  (`string_gpu_mask_matches_cpu_in_body` — `bs = vocab.position("\\")`,
  asserts both `cpu_mask.allow[bs] == 1` and `gpu_mask.allow[bs] == 1`;
  #1596 corrected the prior over-rejecting model).
- **StringEnum closing quote**: only allowed when partial matches a
  complete value
  (`string_enum_gpu_mask_matches_cpu_after_complete_value`).
- **Nullable(Boolean)@Start**: `t`, `f`, `n` all allowed
  (`nullable_boolean_gpu_mask_matches_cpu_at_start`).
- **Nested Integer in Array after digit**: `,` and `]` allowed
  (parent terminators)
  (`nested_integer_in_array_after_digit`).
- **Nested String body**: `,` is content (not a terminator —
  terminator-class transitions fire only at complete_states); `\\`
  opens an escape and is ALLOWED (matches the CPU oracle, #1596)
  (`nested_string_in_array_after_open_quote`).
- **ObjectKey at empty partial**: only first-chars of unseen
  property names allowed
  (`object_key_gpu_mask_matches_cpu_at_empty_partial`).
- **ObjectKey at complete name**: only `"` (close) allowed
  (`object_key_gpu_mask_matches_cpu_after_complete_name`).
- **Unsupported schema returns None**: Object schema (structural
  phase) ⇒ `compute_mask_gpu` returns `None`, caller falls through
  to CPU (`unsupported_schema_returns_none`).

## Verification

Tests in `mod cuda_tests` of `gpu_dispatch.rs` (25+ tests,
`#[cfg(all(test, feature = "cuda"))]`-gated):

- Boolean: 3 (`at_start`, `after_t`, `after_f`).
- Null: 2 (`at_start`, `after_n`).
- Integer: 4 (`at_start`, `after_sign`, `after_zero`,
  `after_digits`).
- Number: 4 (`at_start`, `after_zero_is_only_dot`, `mid_decimal`,
  `after_fractional`).
- String: 2 (`at_start`, `in_body`).
- StringEnum: 4 (`at_start`, `after_open_quote`, `after_h`,
  `after_complete_value`).
- Nullable: 5 (`boolean@start`, `integer@start`, `string@start`,
  `commit_to_inner`, `commit_to_null`).
- Nested-scalar: 3 (`integer_in_array_after_digit`,
  `string_in_array_after_open_quote`,
  `boolean_in_array_after_t`).
- ObjectKey: 3 (`empty_partial`, `after_v`, `after_complete_name`).
- Gate: 1 (`unsupported_schema_returns_none`).

Smoke command (CUDA-only):

```bash
cargo test -p ferrotorch-grammar --lib --features cuda gpu_dispatch:: 2>&1 | tail -3
```

When CUDA is not available, the tests are gated out and `cargo test
-p ferrotorch-grammar --lib` runs the non-CUDA suite. The
`compute_mask_gpu` function itself is compiled (just the test
harness gates on CUDA presence at runtime).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: private `struct CompiledDfa { transitions, char_classes, num_classes, start_state, reject_state, complete_states }` in `gpu_dispatch.rs`; non-test consumer: `fn run_dfa_on_gpu` reads every field to build `DfaMaskInputs::new(...)` in `gpu_dispatch.rs`, dispatched by `pub fn compute_mask_gpu`. |
| REQ-2 | SHIPPED | impl: 7 `fn compile_dfa_for_*` constructors (`boolean`, `null`, `integer`, `number`, `string`, `string_enum`, `object_key`, `nullable`) + `compile_linear_literal` + `compile_boolean_full` in `gpu_dispatch.rs`; non-test consumer: `pub fn compute_mask_gpu` invokes each via the emission-stage match chain. |
| REQ-3 | SHIPPED | impl: `fn add_terminators_to_states(dfa, terminators)` + `fn split_class_for_char(dfa, c)` in `gpu_dispatch.rs`; non-test consumer: `pub fn compute_mask_gpu` wraps every scalar DFA in `add_terminators_to_states` using `grammar.top_frame_parent_terminators()` in production. |
| REQ-4 | SHIPPED | impl: `fn merge_null_branch(inner)` + `fn compile_dfa_for_nullable(inner)` in `gpu_dispatch.rs`; non-test consumer: `pub fn compute_mask_gpu`'s `NullableEmissionStage::Start { inner }` arm dispatches to `compile_dfa_for_nullable(inner)?`. |
| REQ-5 | SHIPPED | impl: `pub struct PackedVocab { pub offsets, pub chars, pub max_token_len }` with `pub fn PackedVocab::pack(vocab: &[String]) -> Self` + manual `Debug` impl in `gpu_dispatch.rs`; non-test consumer: `pub fn compute_mask_gpu` takes `packed: &PackedVocab` and reads `packed.offsets`, `packed.chars`, `packed.max_token_len` to build `DfaMaskInputs`; the `pub use` in `lib.rs:24` makes it reachable as `ferrotorch_grammar::PackedVocab`. |
| REQ-6 | SHIPPED | impl: `pub fn compute_mask_gpu<R: Runtime>(processor, client, packed) -> Option<TokenMask>` in `gpu_dispatch.rs` with the emission-stage match chain; non-test consumer: the `pub use` in `lib.rs:24` makes it reachable as `ferrotorch_grammar::compute_mask_gpu`; grandfathered boundary public API per goal.md S5 via `ferrotorch-llama/src/lib.rs:156`. |
| REQ-7 | NOT-STARTED | Object/Array structural phases fall through to `None` in `compute_mask_gpu` (the final `else { return None }` arm) — pinned by test `unsupported_schema_returns_none` in `gpu_dispatch.rs`. Open prereq blocker #1492 — needs DFA shapes for `ObjectFreshOpen`, `ObjectExpectKey`, `ObjectAfterValue`, `ObjectColon`, `ArrayFreshOpen`, `ArrayAfterValue`. |
| REQ-8 | NOT-STARTED | `add_terminators_to_states` routes complete_state × terminator → popped sink, which then rejects any further char — under-allowing cross-boundary BPE tokens like `,"`. Documented in the doc-comment around the `popped` allocation in `gpu_dispatch.rs`. Open prereq blocker #1493 — needs cross-stack DFA composition or kernel-side parent-state walking. |
