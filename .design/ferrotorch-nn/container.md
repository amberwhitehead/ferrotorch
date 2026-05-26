# ferrotorch-nn — `Sequential` / `ModuleList` / `ModuleDict`

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/nn/modules/container.py
-->

## Summary

`ferrotorch-nn/src/container.rs` defines three container types that hold
sub-modules and propagate `parameters()`, `train()`/`eval()`, and
`state_dict()` to every child. `Sequential` chains layers in order
(forward feeds output of each layer to the next); `ModuleList` and
`ModuleDict` store layers without defining a forward pass (callers
iterate manually). All three mirror their PyTorch counterparts'
named-parameter conventions (`"0.weight"`, `"1.weight"`, …, or
`"encoder.weight"`, `"decoder.weight"`, …).

## Requirements

- REQ-1: `pub struct Sequential<T: Float>` holding
  `layers: Vec<Box<dyn Module<T>>>` and `training: bool`. Mirrors
  `torch.nn.Sequential` (`torch/nn/modules/container.py:59-333`).
- REQ-2: `Sequential::new(layers)` constructor accepting a
  `Vec<Box<dyn Module<T>>>`. Mirrors `nn.Sequential(*args)` /
  `nn.Sequential(OrderedDict)` (`container.py:108-122`); we accept a
  Vec because Rust lacks Python's variadic positional args and the
  OrderedDict construction is replaced by the explicit ordering of
  Vec entries.
- REQ-3: `Sequential::push` / `len` / `is_empty` for builder
  ergonomics. Mirrors `nn.Sequential.append`
  (`container.py:256-275`), `__len__`, and an empty-check idiom.
- REQ-4: `impl Module<T> for Sequential<T>`: `forward` chains each
  layer in order (errors on empty), `parameters` /
  `parameters_mut` flat-maps every layer's parameters,
  `named_parameters` prefixes each layer's named parameters with
  `"{i}.{name}"` matching upstream's PyTorch naming convention
  (`container.py:117-122`), `train` / `eval` propagate to all
  children. Mirrors `Sequential.forward(input)`
  (`container.py:248-254`).
- REQ-5: `pub struct ModuleList<T: Float>` holding `modules:
  Vec<Box<dyn Module<T>>>` and `training: bool`. Mirrors
  `torch.nn.ModuleList` (`container.py:335-502`).
- REQ-6: `ModuleList::new(modules)` / `::empty()` / `::get(i)` /
  `::get_mut(i)` / `::push(m)` / `::len()` / `::is_empty()`. Mirrors
  `nn.ModuleList.__init__`, `__getitem__`, `append`, `__len__`
  (`container.py:361-500`).
- REQ-7: `impl Module<T> for ModuleList<T>`: `forward` returns
  `InvalidArgument` (matches upstream's choice to inherit from
  `_forward_unimplemented` — see `container.py:502` comment "remove
  forward altogether to fallback on Module's
  `_forward_unimplemented`"). Parameter iteration, naming, and
  train/eval propagation mirror Sequential's behavior with index
  prefixes.
- REQ-8: `pub struct ModuleDict<T: Float>` holding `entries:
  Vec<(String, Box<dyn Module<T>>)>` and `training: bool`. The
  `Vec<(...)>` preserves insertion order without requiring an
  `IndexMap` dependency — matches upstream's documented insertion-order
  guarantee (`container.py:512-516`).
- REQ-9: `ModuleDict::new()` / `::insert(key, module)` (replaces
  existing entry in place if key matches) / `::get(&str)` /
  `::get_mut(&str)` / `::keys()` / `::len()` / `::is_empty()`.
  Mirrors `nn.ModuleDict.__init__`, `__setitem__`, `__getitem__`,
  `keys()`, `update()` (`container.py:548-700`).
- REQ-10: `impl Module<T> for ModuleDict<T>`: `forward` returns
  `InvalidArgument` (matches upstream's pattern); parameter iteration
  flat-maps; `named_parameters` prefixes with `"{key}.{name}"`;
  train/eval propagate.
- REQ-11: `Default for ModuleDict` returns an empty dict — required
  for `#[derive(Default)]` on parent modules that embed a ModuleDict
  via `derive(Default)`.
- REQ-12: `Display` impls for all three types render
  `Sequential(...)` / `ModuleList(...)` / `ModuleDict(...)` with
  one line per child. Mirrors upstream's `__repr__` shape (the
  `_addindent` helper in `container.py:34-44`).

## Acceptance Criteria

- [x] AC-1: `pub struct Sequential<T: Float>` with the described fields.
- [x] AC-2: `Sequential::new` / `push` / `len` / `is_empty`.
- [x] AC-3: `Sequential::forward` chains layers, errors on empty.
- [x] AC-4: `Sequential::named_parameters` produces
  `"0.weight"`, `"1.weight"`, ….
- [x] AC-5: `ModuleList` struct + `::new` / `::empty` / `::get` /
  `::get_mut` / `::push` / `::len` / `::is_empty`.
- [x] AC-6: `ModuleList::forward` returns `InvalidArgument`.
- [x] AC-7: `ModuleDict` struct preserving insertion order via
  `Vec<(String, Box<dyn Module<T>>)>`.
- [x] AC-8: `ModuleDict::insert` replaces existing entry in place.
- [x] AC-9: `ModuleDict::named_parameters` produces
  `"encoder.weight"`, `"decoder.weight"`, ….
- [x] AC-10: `Default for ModuleDict`.
- [x] AC-11: `Display` impls for all three.
- [x] AC-12: `test_containers_are_send_sync` asserts `Send + Sync`.

## Architecture

### `Sequential` (REQ-1..4)

```rust
pub struct Sequential<T: Float> {
    layers: Vec<Box<dyn Module<T>>>,
    training: bool,
}
```

Forward chains layers in order; if the layer list is empty,
`forward` returns `InvalidArgument` (upstream raises a runtime
error from `nn.Module._forward_unimplemented` on empty
`Sequential`). Named parameters use the index as prefix:
`"0.weight"`, `"0.bias"`, `"1.weight"`, ….

### `ModuleList` (REQ-5..7)

Trait-object Vec; deliberately does not define `forward` (returns
`InvalidArgument` if called). Users iterate manually:

```rust
let mut x = input.clone();
for i in 0..list.len() {
    let layer = list.get(i).ok_or_else(|| FerrotorchError::InvalidArgument { ... })?;
    x = layer.forward(&x)?;
}
```

The `forward` returning `InvalidArgument` matches upstream's
choice to fall back on `_forward_unimplemented`
(`container.py:502`) — both languages signal "you held the API
wrong" rather than picking an implicit semantic.

### `ModuleDict` (REQ-8..11)

Insertion-ordered via `Vec<(String, Box<dyn Module<T>>)>`; lookups
are O(N) linear scans. The trade-off: we avoid pulling in `IndexMap`
as a dependency while preserving upstream's documented
insertion-order guarantee. `insert` matches upstream's "replace in
place if key exists" semantics — the entry stays at its original
position rather than being moved to the end.

`Default for ModuleDict` returns an empty dict so parent modules
that derive `Default` and embed a ModuleDict work without manual
glue.

### `Display` (REQ-12)

Renders one line per child, indented two spaces:

```
Sequential(
  (0): <module>
  (1): <module>
)
```

Mirrors PyTorch's `__repr__` shape from `_addindent`
(`container.py:34-44`); we don't yet render the inner module's repr
(the `<module>` placeholder) because the `Module` trait doesn't
require a `Display` impl.

### Non-test production consumers

- `pub use container::{ModuleDict, ModuleList, Sequential}` in `lib.rs:195`.
- `ferrotorch-nn/src/lib.rs` prelude re-exports all three.
- Downstream model composition code (e.g. `ferrotorch-llama`, `ferrotorch-vision`'s ResNet construction, `ferrotorch-bert` transformer-block stacks) instantiates these containers to compose layer trees. The `Sequential`-of-`Linear`-plus-activation pattern is the canonical onboarding shape.
- `impl Module<T> for Sequential<T>` is itself consumed by every caller that treats the container as a single module (composes into a parent `Sequential`, passes to `optimizer.parameters()`, etc.).

## Parity contract

`parity_ops = []`. The containers are structural — their numerical
contract is entirely on the contained modules. Edge cases the
containers themselves own:

- **Empty Sequential**: `forward` returns `InvalidArgument`. PyTorch
  raises a runtime error from the inherited
  `_forward_unimplemented`; ferrotorch surfaces it as a `Result::Err`.
- **ModuleList / ModuleDict forward**: deliberately error
  (`InvalidArgument`) — these are storage containers, not chained
  composition.
- **`ModuleDict::insert` of an existing key**: replaces in place,
  preserving position. Matches upstream's `__setitem__` semantics.
- **State-dict round-trip**: parent path `"i.weight"` /
  `"key.weight"` matches upstream exactly so SafeTensors checkpoints
  load cleanly across PyTorch ↔ ferrotorch.

## Verification

Tests in `mod tests in container.rs` (22 tests):

- Sequential: `test_sequential_forward_chains_layers`,
  `_empty_forward_errors`, `_parameter_count`,
  `_named_parameters_keys`, `_train_eval_propagation`,
  `_state_dict_roundtrip`, `_push`.
- ModuleList: `test_module_list_forward_errors`, `_get`,
  `_get_mut`, `_push`, `_parameters`, `_train_eval`.
- ModuleDict: `test_module_dict_forward_errors`, `_insert_get`,
  `_insert_replaces`, `_keys_insertion_order`, `_get_mut`,
  `_named_parameters_prefixed_by_key`, `_train_eval`, `_default`,
  `_state_dict_roundtrip`.
- `test_containers_are_send_sync` — auto-trait assertion.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-nn --lib container:: 2>&1 | tail -3
```

Expected: `22 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct Sequential<T: Float>` with `layers: Vec<Box<dyn Module<T>>>` + `training: bool` in `container.rs` mirroring `torch/nn/modules/container.py:59-333`; non-test consumer: `pub use container::{ModuleDict, ModuleList, Sequential}` in `lib.rs:195`; downstream model crates compose `Sequential` for stacked layers (e.g. an MLP defined as `Sequential::new(vec![Linear, ReLU, Linear])`). |
| REQ-2 | SHIPPED | impl: `pub fn Sequential::new(layers)` constructor in `container.rs` mirroring `torch/nn/modules/container.py:108-122`; non-test consumer: every downstream MLP/CNN composition pattern. |
| REQ-3 | SHIPPED | impl: `push`, `len`, `is_empty` inherent methods in `container.rs` mirroring `nn.Sequential.append` (`container.py:256-275`); non-test consumer: builder-pattern model construction in downstream code. |
| REQ-4 | SHIPPED | impl: `impl<T: Float> Module<T> for Sequential<T>` with chained `forward`, flat-mapped `parameters`/`parameters_mut`, indexed `named_parameters` in `container.rs` mirroring `container.py:117-122, 248-254`; non-test consumer: `optimizer.parameters()` flows through this; `ferrotorch-nn/src/transformer.rs`'s `Transformer` composes multiple `Sequential` instances and consumes their `Module<T>` impl through `Box<dyn Module<T>>`. |
| REQ-5 | SHIPPED | impl: `pub struct ModuleList<T: Float>` with `modules: Vec<Box<dyn Module<T>>>` in `container.rs` mirroring `container.py:335-502`; non-test consumer: re-exported via `lib.rs:195`; downstream multi-branch architectures (e.g. mixture-of-experts heads) consume `ModuleList`. |
| REQ-6 | SHIPPED | impl: `::new`, `::empty`, `::get`, `::get_mut`, `::push`, `::len`, `::is_empty` inherent methods on `ModuleList` mirroring `container.py:361-500`; non-test consumer: every downstream callsite that iterates over a `ModuleList` (the canonical MoE pattern). |
| REQ-7 | SHIPPED | impl: `impl<T: Float> Module<T> for ModuleList<T>` with `forward` returning `InvalidArgument` (matches upstream `container.py:502` fallback to `_forward_unimplemented`); non-test consumer: `optimizer.parameters()` consumes `ModuleList::parameters()`; downstream callers that nest a `ModuleList` inside a `Sequential` consume the `Module<T>` impl. |
| REQ-8 | SHIPPED | impl: `pub struct ModuleDict<T: Float>` with insertion-ordered `Vec<(String, Box<dyn Module<T>>)>` in `container.rs` mirroring `container.py:505-700` (the documented insertion-order guarantee at `container.py:512-516`); non-test consumer: re-exported via `lib.rs:195`; downstream encoder/decoder-style architectures consume named module containers. |
| REQ-9 | SHIPPED | impl: `::new`, `::insert` (in-place replace), `::get`, `::get_mut`, `::keys`, `::len`, `::is_empty` on `ModuleDict` mirroring `container.py:548-700`; non-test consumer: dynamic-dispatch model heads that look up modules by name at the composition layer. |
| REQ-10 | SHIPPED | impl: `impl<T: Float> Module<T> for ModuleDict<T>` with `forward` returning `InvalidArgument`, key-prefixed `named_parameters`, train/eval propagation; non-test consumer: `optimizer.parameters()` and state-dict load/save through the `Module<T>` impl. |
| REQ-11 | SHIPPED | impl: `impl<T: Float> Default for ModuleDict<T>` returning empty dict in `container.rs`; non-test consumer: parent modules that `#[derive(Default)]` over a `ModuleDict` field. |
| REQ-12 | SHIPPED | impl: `impl<T: Float> std::fmt::Display for Sequential<T>` / `for ModuleList<T>` / `for ModuleDict<T>` in `container.rs` mirroring `container.py:34-44` `_addindent`; non-test consumer: logging / debug output in training drivers that print model topology via `{model}`. |
