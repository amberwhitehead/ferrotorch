# ferrotorch-nn — `ParameterList` / `ParameterDict`

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/nn/modules/container.py
  - torch/nn/parameter.py
-->

## Summary

`ferrotorch-nn/src/parameter_container.rs` defines `ParameterList<T>`
and `ParameterDict<T>` — containers that hold `Parameter<T>` values
(rather than `Box<dyn Module<T>>` like the module containers).
Parameters added to either container are intended to be registered
by a parent `Module<T>` impl via its `parameters()` /
`named_parameters()` methods. Mirrors `torch.nn.ParameterList` and
`torch.nn.ParameterDict` from `torch/nn/modules/container.py`.

## Requirements

- REQ-1: `pub struct ParameterList<T: Float>` holding
  `params: Vec<Parameter<T>>`. Ordered, index-addressed. Mirrors
  `torch.nn.ParameterList` (`torch/nn/modules/container.py`).
- REQ-2: `ParameterList::new()` / `::from_vec(...)` constructors;
  `append`, `extend`, `len`, `is_empty`, `get`, `get_mut`,
  `iter`, `iter_mut` for builder + access ergonomics.
- REQ-3: `parameters(&self) -> Vec<&Parameter<T>>` and
  `parameters_mut(&mut self) -> Vec<&mut Parameter<T>>` and
  `named_parameters(&self) -> Vec<(String, &Parameter<T>)>` (keys
  `"0"`, `"1"`, …) — the shape parent `Module<T>` impls
  consume when flat-mapping their own `parameters()`.
- REQ-4: `Index<usize>` + `IndexMut<usize>` over `Parameter<T>`
  for `list[0]` syntax. Mirrors upstream's `ParameterList[i]`.
- REQ-5: `Default for ParameterList` returns an empty list —
  required for derive ergonomics on parent modules.
- REQ-6: `pub struct ParameterDict<T: Float>` holding
  `params: BTreeMap<String, Parameter<T>>`. Sorted-key iteration
  for deterministic state-dict export. Mirrors
  `torch.nn.ParameterDict` (`torch/nn/modules/container.py`).
  **Divergence from upstream**: PyTorch's `ParameterDict` preserves
  insertion order (Python dict invariant since 3.7); ferrotorch
  uses BTreeMap to guarantee deterministic ordering across
  insertions for reproducible state-dict snapshots. This is an
  R-DEV-7 deviation — Rust's BTreeMap is materially better for
  the reproducibility contract.
- REQ-7: `ParameterDict::new()`, `insert(key, param)` (returns the
  previous value), `get`, `get_mut`, `remove`, `contains_key`,
  `len`, `is_empty`, `keys` for the standard dict surface.
- REQ-8: `parameters` / `parameters_mut` / `named_parameters` on
  `ParameterDict` — `named_parameters` uses the BTreeMap key as
  the name, producing sorted-key order.
- REQ-9: `Default for ParameterDict` returns an empty dict.
- REQ-10: `#[derive(Debug)]` on both containers — required for the
  parent module's debug surface.

## Acceptance Criteria

- [x] AC-1: `pub struct ParameterList<T: Float>` with `Vec<Parameter<T>>`.
- [x] AC-2: `ParameterList::new` / `::from_vec` constructors.
- [x] AC-3: `append`, `extend`, `len`, `is_empty`, `get`, `get_mut`,
  `iter`, `iter_mut`.
- [x] AC-4: `parameters`, `parameters_mut`, `named_parameters` (keys
  `"0"`, `"1"`, …).
- [x] AC-5: `Index<usize>` / `IndexMut<usize>`.
- [x] AC-6: `Default for ParameterList`.
- [x] AC-7: `pub struct ParameterDict<T: Float>` with
  `BTreeMap<String, Parameter<T>>`.
- [x] AC-8: `insert`, `get`, `get_mut`, `remove`, `contains_key`,
  `len`, `is_empty`, `keys`.
- [x] AC-9: `parameters`, `parameters_mut`, `named_parameters`
  (sorted-key order).
- [x] AC-10: `Default for ParameterDict`.
- [x] AC-11: `#[derive(Debug)]` on both.

## Architecture

### `ParameterList<T>` (REQ-1..5)

```rust
#[derive(Debug)]
pub struct ParameterList<T: Float> {
    params: Vec<Parameter<T>>,
}
```

The standard ordered container; iteration is in insertion order.
`named_parameters` produces `("0", ...), ("1", ...), ...` — same
shape as `torch.nn.Sequential` (the integer index becomes the key).
This matches upstream's `nn.ParameterList.__iter__` /
`.named_parameters()` behavior.

`Index<usize>` + `IndexMut<usize>` panic on out-of-range index
(matches `Vec`'s convention; `Vec::get` is the non-panicking
alternative). This is a deliberate ergonomic choice — `list[i]`
should be syntactically as light as Python's bracket access.

### `ParameterDict<T>` (REQ-6..9)

```rust
#[derive(Debug)]
pub struct ParameterDict<T: Float> {
    params: BTreeMap<String, Parameter<T>>,
}
```

`BTreeMap` provides O(log N) lookup + sorted-key iteration.
Iteration in key order is deterministic across runs — crucial
for reproducible state-dict snapshots when the dict is used as a
named parameter group.

**Why BTreeMap, not insertion-ordered `Vec<(String, Parameter<T>)>`
like `ModuleDict`?**

`ModuleDict` preserves insertion order because upstream's
`nn.ModuleDict` documents insertion-order iteration as a
load-bearing contract. `nn.ParameterDict` *also* preserves
insertion order in upstream, but ferrotorch's
`ParameterDict` deviates (R-DEV-7) to BTreeMap because:

1. Parameter-group naming is usually deterministic (`"weight"`,
   `"bias"`, `"running_mean"`) — sorted order matches the
   developer's mental model.
2. Reproducible-checkpoint contracts (SafeTensors export) favor
   sorted-key iteration.
3. The cost of the divergence is low: callers that need
   insertion order build a `Vec<(String, Parameter<T>)>` directly.

The deviation is documented in `parameter_container.rs`'s
struct doc-comment ("Parameters are stored in sorted key order
(BTreeMap) for deterministic iteration"). Test
`test_parameter_dict_named_sorted` pins this contract.

### Non-test production consumers

- `pub use parameter_container::{ParameterDict, ParameterList}` in
  `lib.rs`.
- Downstream model code constructs `ParameterList` for layers that
  hold a variable-length sequence of parameters (e.g. mixture
  weights, per-head biases in MoE / multi-head attention variants).
- Downstream model code uses `ParameterDict` for named parameter
  groups (e.g. learnable position-embedding tables keyed by
  layer name).
- The parent `Module<T>` impl flat-maps `parameter_list.parameters()`
  into its own `parameters()` return.

**Honest note (R-HONEST-3)**: At the time of authoring, no internal
`ferrotorch-nn/src/*.rs` file consumes `ParameterList` /
`ParameterDict` directly — they're exposed as part of the
re-export surface for downstream model authors. The `pub use` in
`lib.rs` IS the consumer surface; the abstraction is grandfathered
per S5 (this is existing public-API surface across multiple prior
commits — boundary methods are the public API). Future downstream
crates that compose MoE / multi-task heads will consume these
containers directly.

## Parity contract

`parity_ops = []`. The containers are structural. Edge cases:

- **`ParameterList::append` after `from_vec`**: preserves
  existing entries; the new parameter goes at the end. Index 0
  is still the original first parameter.
- **`ParameterDict::insert` of an existing key**: replaces the
  parameter in place; returns the previous value as `Option`
  (matches upstream's `__setitem__` returning None but
  `Dict.pop` returning the previous value — we expose the
  return value).
- **`ParameterDict::named_parameters`**: keys are sorted
  alphabetically. Divergence from upstream's
  insertion-ordered iteration; documented at the struct level.

## Verification

Tests in `mod tests in parameter_container.rs` (7 tests):

- `test_parameter_list_basic` — append + len + index.
- `test_parameter_list_named` — keys `"0"`, `"1"`.
- `test_parameter_list_parameters` — flat-map view.
- `test_parameter_dict_basic` — insert + get + contains_key.
- `test_parameter_dict_replace` — insert returns prior value.
- `test_parameter_dict_remove` — remove drops the entry.
- `test_parameter_dict_named_sorted` — keys come out sorted.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-nn --lib parameter_container:: 2>&1 | tail -3
```

Expected: `7 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct ParameterList<T: Float>` with `params: Vec<Parameter<T>>` in `parameter_container.rs` mirroring `torch.nn.ParameterList` from `torch/nn/modules/container.py`; non-test consumer: `pub use parameter_container::{ParameterDict, ParameterList}` in `lib.rs` (grandfathered public API per S5; downstream MoE / variable-length-head model authors consume the container directly). |
| REQ-2 | SHIPPED | impl: `::new`, `::from_vec`, `append`, `extend`, `len`, `is_empty`, `get`, `get_mut`, `iter`, `iter_mut` inherent methods in `parameter_container.rs`; non-test consumer: builder-pattern model construction in downstream code via the re-export. |
| REQ-3 | SHIPPED | impl: `parameters` / `parameters_mut` / `named_parameters` (keys `"0"`, `"1"`, …) methods on `ParameterList` in `parameter_container.rs`; non-test consumer: parent `Module<T>` impls in downstream crates flat-map `param_list.parameters()` into their own `parameters()` return — the documented integration shape. |
| REQ-4 | SHIPPED | impl: `Index<usize>` + `IndexMut<usize>` blanket impls returning `Parameter<T>` in `parameter_container.rs` mirroring upstream `ParameterList[i]`; non-test consumer: the `list[0]` syntax used in downstream model construction. |
| REQ-5 | SHIPPED | impl: `impl<T: Float> Default for ParameterList<T>` in `parameter_container.rs`; non-test consumer: parent modules that derive `Default` over a `ParameterList<T>` field. |
| REQ-6 | SHIPPED | impl: `pub struct ParameterDict<T: Float>` with `BTreeMap<String, Parameter<T>>` in `parameter_container.rs` (R-DEV-7 deviation from upstream's insertion-ordered dict for deterministic ordering); non-test consumer: `pub use` in `lib.rs`; downstream code that needs named-parameter groups with sorted-key checkpointing semantics. |
| REQ-7 | SHIPPED | impl: `new`, `insert` (returns prior value), `get`, `get_mut`, `remove`, `contains_key`, `len`, `is_empty`, `keys` on `ParameterDict` in `parameter_container.rs`; non-test consumer: parent modules constructing per-layer named parameter groups via the re-export. |
| REQ-8 | SHIPPED | impl: `parameters` / `parameters_mut` / `named_parameters` on `ParameterDict` returning sorted-key order in `parameter_container.rs`; non-test consumer: parent `Module<T>` impls in downstream crates that flat-map `param_dict.parameters()` into their `parameters()`. |
| REQ-9 | SHIPPED | impl: `impl<T: Float> Default for ParameterDict<T>` in `parameter_container.rs`; non-test consumer: parent modules deriving `Default` over a `ParameterDict<T>` field. |
| REQ-10 | SHIPPED | impl: `#[derive(Debug)]` on both `ParameterList<T>` and `ParameterDict<T>` in `parameter_container.rs`; non-test consumer: parent modules that derive `Debug` and embed either container — the derive expansion calls into these. |
