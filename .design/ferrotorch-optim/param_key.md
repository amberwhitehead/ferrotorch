# ferrotorch-optim — `ParamKey` (typed per-parameter HashMap key)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/optim/optimizer.py
-->

## Summary

`ferrotorch-optim/src/param_key.rs` defines `ParamKey`, an 8-byte
`Copy` newtype over `(group: u32, param: u32)` that the in-tree
optimizers (`Adam`, `AdamW`, `Adamax`, `Asgd`, `RAdam`, `SparseAdam`,
...) use as the key in their per-parameter state HashMaps. It
replaces the pre-CL-1122 pattern of recomputing
`format!("g{group_idx}_p{param_idx}")` inside the hot inner step
loop — for a 7-billion-parameter model that meant a fresh `String`
heap allocation per parameter per step. The wire format
(`"g{g}_p{p}"`, used by `OptimizerState`) is preserved through
`Display` + `FromStr` impls so older checkpoints round-trip
unchanged.

PyTorch's analog is the use of the tensor itself as a `defaultdict`
key in `torch.optim.Optimizer.state` (`torch/optim/optimizer.py:395`
`self.state: defaultdict[torch.Tensor, Any] = defaultdict(dict)`).
Python's tensor identity (hash by `id`) makes that cheap; the
ferrotorch translation uses indices instead because `Parameter<T>`
identity is not a Rust-idiomatic hash key (R-DEV-4 deviation —
Rust eliminates Python's id-based identity hashing).

## Requirements

- REQ-1: `pub struct ParamKey { pub group: u32, pub param: u32 }`
  is `#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]`
  so it can be used as a HashMap key and sorted for deterministic
  serialization order. 8 bytes total.
- REQ-2: `pub const fn new(group: usize, param: usize) -> Self`
  constructor with `debug_assert!` bounds on the `u32` truncation
  (no realistic model holds 4 billion parameter groups or 4 billion
  parameters within a single group; the debug assertion catches
  programming errors).
- REQ-3: `impl fmt::Display` produces the legacy wire format
  `"g{group}_p{param}"`, matching the pre-CL-1122
  `format!("g{}_p{}", group_idx, param_idx)` string. Preserves
  checkpoint backward compatibility.
- REQ-4: `impl FromStr` parses the wire format back into a
  `ParamKey`. Bad formats return `FerrotorchError::InvalidArgument`
  with a descriptive message. Used by `load_state_dict` paths.
- REQ-5: `impl From<ParamKey> for String` and
  `impl TryFrom<&str> for ParamKey` for ergonomic conversions
  matching Rust idioms.

## Acceptance Criteria

- [x] AC-1: `ParamKey` is 8 bytes (two `u32` fields) and `Copy`.
- [x] AC-2: `ParamKey::new(g, p)` is `const fn` so it can appear in
  `const` contexts.
- [x] AC-3: `to_string()` returns exactly `"g{g}_p{p}"` matching
  the legacy format.
- [x] AC-4: `"g3_p17".parse::<ParamKey>()` returns
  `Ok(ParamKey { group: 3, param: 17 })`.
- [x] AC-5: Bad-format strings (`""`, `"0_0"`, `"g0p0"`, `"g_p"`,
  `"gx_p0"`, `"g0_py"`) all return `Err`.
- [x] AC-6: Used as a HashMap key by every state-keeping optimizer
  in the crate (`Adam`, `AdamW`, `Adamax`, `Asgd`, `RAdam`,
  `SparseAdam`, `Rprop`).

## Architecture

### Why `u32` × 2 not `usize` × 2?

The struct is `#[repr(Rust)]`, so on 64-bit targets `usize` × 2
would be 16 bytes; the `u32` × 2 packing is 8 bytes and fits in a
single register on 64-bit. Hashing 8 bytes is faster than hashing
16. The truncation is guarded by `debug_assert!` in `new`; no
realistic optimizer reaches `u32::MAX` group / param counts.

### Wire format compatibility (REQ-3, REQ-4)

The string `"g{group}_p{param}"` is the exact format the
pre-CL-1122 optimizers wrote into `OptimizerState`. Preserving it
via `Display` + `FromStr` means:
- Old checkpoints loaded by post-CL-1122 code → `FromStr` parses
  the legacy strings.
- New checkpoints saved by post-CL-1122 code → `Display` writes
  the same legacy strings, so old code can load them too.

### `FromStr` parser (REQ-4)

```text
"g3_p17"
→ strip "g" prefix
→ split on "_p"
→ parse "3" as u32, "17" as u32
→ ParamKey { group: 3, param: 17 }
```

Any deviation (missing `g` prefix, missing `_p` separator,
non-numeric components, leading whitespace, ...) returns
`FerrotorchError::InvalidArgument` with a message that quotes the
offending input. The error type is the same as the rest of the
crate's error vocabulary, so `load_state_dict` impls can use `?`
to propagate it.

### Non-test production consumers

- `ferrotorch-optim/src/adam.rs` — `state: HashMap<ParamKey, AdamParamState>`.
- `ferrotorch-optim/src/adamw.rs:163` `state: HashMap<ParamKey, AdamWParamState>` + line 166 `foreach_state: HashMap<ParamKey, AdamWForeachState<T>>`.
- `ferrotorch-optim/src/adamax.rs:134` — same pattern.
- `ferrotorch-optim/src/asgd.rs:151` — same.
- `ferrotorch-optim/src/radam.rs:139` — same.
- `ferrotorch-optim/src/sparse_adam.rs:88` — same.
- Each optimizer's `load_state_dict` calls `key.parse::<ParamKey>()?`
  to convert the wire-format string back to a typed key (e.g.
  `ferrotorch-optim/src/adamw.rs:571`, `radam.rs:503`).

## Parity contract

`parity_ops = []`. The newtype is an internal representation; it
does not surface in numerical computation. Edge cases owned:

- **`new(u32::MAX as usize + 1, ...)`**: `debug_assert!` panics in
  debug builds; truncates silently in release builds. Since no
  realistic optimizer reaches this, the choice is defensible.
- **Hash determinism**: `derive(Hash)` produces a key that is
  deterministic per run but NOT stable across Rust versions /
  platforms. This is fine for in-memory state; the wire format
  uses the `Display` string, not the hash, for cross-run stability.
- **Empty string parse**: returns `Err`.
- **Negative numbers in parse**: returns `Err` (`u32::parse`
  rejects them).

## Verification

Four unit tests in `mod tests` (param_key.rs lines 109-160):

- `display_matches_legacy_format` — wire-format round-trip.
- `from_str_round_trips_display` — `Display ∘ FromStr = id` over
  a sample grid.
- `from_str_rejects_bad_format` — five malformed inputs all
  return `Err`.
- `copy_and_hash_eq_consistent` — `HashMap<ParamKey, _>` lookup
  works as expected; `Copy` semantics preserved.
- `string_conversion` — `From<ParamKey> for String` and
  `TryFrom<&str> for ParamKey` round-trip.

Smoke command:

```bash
cargo test -p ferrotorch-optim --lib param_key:: 2>&1 | tail -3
```

Expected: `5 passed; 0 failed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct ParamKey { pub group: u32, pub param: u32 }` with the full derive list at `ferrotorch-optim/src/param_key.rs:43`; non-test consumer: `ferrotorch-optim/src/adamw.rs:163` `state: HashMap<ParamKey, AdamWParamState>` plus 5 other optimizer state-maps. |
| REQ-2 | SHIPPED | impl: `pub const fn new` at `ferrotorch-optim/src/param_key.rs:62` with `debug_assert!` bounds; non-test consumer: `ferrotorch-optim/src/adamw.rs:220` `ParamKey::new(group_idx, param_idx)` (called from `AdamW::param_key`), same pattern at `radam.rs:161`, `asgd.rs:181`, `adamax.rs:156`, `sparse_adam.rs:114`. |
| REQ-3 | SHIPPED | impl: `impl fmt::Display for ParamKey` at `ferrotorch-optim/src/param_key.rs:72` writing `"g{g}_p{p}"` matching `format!("g{}_p{}", g, p)`; non-test consumer: `ferrotorch-optim/src/adamw.rs:556` `// CL-1122: render typed ParamKey to the "g{}_p{}" wire format` (used in `state_dict` serialization); same pattern at `radam.rs:489`, `asgd.rs:441`, `adamax.rs:415`. |
| REQ-4 | SHIPPED | impl: `impl FromStr for ParamKey` at `ferrotorch-optim/src/param_key.rs:85` returning `FerrotorchError::InvalidArgument` on malformed input; non-test consumer: `ferrotorch-optim/src/adamw.rs:571` `let key: ParamKey = key.parse()?;` inside `AdamW::load_state_dict`; same pattern at `radam.rs:503`, `asgd.rs:456`, `adamax.rs:429`. |
| REQ-5 | SHIPPED | impl: `impl From<ParamKey> for String` at `ferrotorch-optim/src/param_key.rs:78` and `impl TryFrom<&str>` at line 100; non-test consumer: `From<ParamKey> for String` is used by the `to_string()` path inside every `state_dict` serializer (e.g. `adamw.rs:556`); `TryFrom<&str>` is the ergonomic conversion exposed at the crate boundary for external optimizer authors. |
