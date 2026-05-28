# ferrotorch-nn — `lazy_linear` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/nn/modules/linear.py
  - torch/nn/modules/lazy.py
-->

## Summary

`ferrotorch-nn/src/lazy_linear.rs` implements `LazyLinear<T>` — the
deferred-`in_features`-discovery variant of `Linear` mirroring
`torch.nn.LazyLinear` (defined at
`torch/nn/modules/linear.py:18`, exported through
`torch.nn.modules.lazy.LazyModuleMixin` at
`torch/nn/modules/lazy.py`). The layer takes `out_features` and
`bias` up front, and defers weight allocation until the first
forward call, at which point the input tensor's last dimension is
taken as `in_features`. Subsequent forward calls behave like a
standard `Linear`. Thread-safe via `std::sync::OnceLock`.

## Requirements

- REQ-1: `pub struct LazyLinear<T: Float>` carrying `out_features:
  usize`, `bias_enabled: bool`, `weight: OnceLock<Parameter<T>>`,
  `bias: OnceLock<Parameter<T>>`, and `training: AtomicBool`.
  Mirrors upstream `LazyLinear` carrying `UninitializedParameter`
  values that become real once a forward observes the shape.
- REQ-2: `LazyLinear::new(out_features, bias)` validates
  `out_features > 0` and constructs an empty-`OnceLock` instance.
- REQ-3: `is_initialized()` / `in_features()` / `out_features()`
  accessors. `in_features()` returns `None` until materialization.
- REQ-4: `materialize(in_features)` — eagerly populate the
  parameters with the given `in_features`. Idempotent — the first
  call wins; subsequent calls (even with a different in_features)
  are no-ops. Validates `in_features > 0`.
- REQ-5: `<LazyLinear as Module>::forward` discovers `in_features`
  from `input.shape()[ndim-1]` on first call, materializes
  parameters, then dispatches through `linear_fused` exactly like
  `Linear`. Rejects ndim=0 inputs and shape-mismatches against the
  materialized `in_features`.
- REQ-6: Higher-rank input handling — same flatten-then-reshape
  trick as `Linear`. 1D, 2D, 3D, 4D etc. all supported.
- REQ-7: `Module<T>` trait — `parameters()` / `parameters_mut()` /
  `named_parameters()` return empty lists BEFORE materialization,
  the materialized parameters afterwards. This matters because
  optimizers that snapshot the parameter list at construction time
  must call `materialize` first.
- REQ-8: Initialization mirrors `Linear` — Kaiming uniform on
  weight (ReLU gain), zeros on bias. NOTE: same divergence from
  upstream's `kaiming_uniform_(a=sqrt(5))` as documented in
  `linear.md` REQ-5/6.
- REQ-9: Thread safety — `OnceLock::set()` is race-safe: if two
  threads call `forward` concurrently on an uninitialized layer,
  one materialization wins and the other is dropped. The hot path
  (post-init) reads via `OnceLock::get()` with no lock.

## Acceptance Criteria

- [x] AC-1: Pre-init `is_initialized() == false`, `in_features() ==
  None`, `parameters().len() == 0`.
- [x] AC-2: First forward triggers materialization;
  `is_initialized() == true` and `parameters().len() == 2` (or 1
  without bias).
- [x] AC-3: Second forward with same `in_features` reuses the
  materialized parameters (verified via output shape consistency).
- [x] AC-4: Second forward with different `in_features` rejected
  with `ShapeMismatch`.
- [x] AC-5: Explicit `materialize(in_features)` works without a
  forward call. Idempotent.
- [x] AC-6: Zero `out_features` rejected at construction.
- [x] AC-7: Higher-rank input (3-D, 4-D) returns correctly-shaped
  output.
- [x] AC-8: `train()` / `eval()` toggle works post-materialization.

## Architecture

### The struct (REQ-1)

`pub struct LazyLinear<T: Float>` in `lazy_linear.rs`. The
`OnceLock<Parameter<T>>` wrappers around `weight` and `bias` are the
key design choice — they let us go from "uninitialized" to
"initialized" exactly once, race-free, without taking a lock on
every forward call. `training: AtomicBool` allows `train()` and
`eval()` to be `&mut self`-free (a regular `bool` would force the
ref `&self.training` to be exclusive).

### Construction (REQ-2)

`LazyLinear::new(out_features, bias)` in `lazy_linear.rs`. Returns
`Err(InvalidArgument)` if `out_features == 0`. Otherwise returns a
struct with empty OnceLocks.

### Materialization (REQ-4)

`LazyLinear::materialize(in_features)` in `lazy_linear.rs`.
Validates `in_features > 0`. Checks `self.weight.get().is_none()`
to avoid double-init; if free, allocates a `[out_features,
in_features]` Parameter, Kaiming-inits, and `OnceLock::set`s it
(ignoring the `Err` return that means another thread won the race).
If bias is enabled and bias OnceLock is free, allocates +
zero-inits the bias parameter. Idempotent: calling with a different
in_features after initialization is a no-op (first one wins per
the documented contract).

### Forward (REQ-5, REQ-6)

`<LazyLinear<T> as Module<T>>::forward` in `lazy_linear.rs`:
1. Validate `input.ndim() > 0` (reject scalars).
2. If `weight.get().is_none()`, call
   `self.materialize(input.shape()[ndim-1])`.
3. Read materialized `weight` (the `expect("weight should be
   initialized after materialize()")` is a sentinel for the
   invariant established by step 2).
4. Validate `input.shape()[ndim-1] == weight.shape()[1]`.
5. Same flatten-reshape + `linear_fused` path as `Linear::forward`.

### Trait surface (REQ-7)

`impl<T: Float> Module<T> for LazyLinear<T>` in `lazy_linear.rs`.
`parameters()` builds a Vec by pushing `weight` and `bias` only if
their OnceLocks are populated; pre-init returns empty. This is the
correct semantics — an optimizer constructed BEFORE the first
forward would see no parameters; the optimizer must either snapshot
post-forward or call `materialize` explicitly. The latter pattern
is the documented happy path.

### Non-test production consumers

- `pub use lazy_linear::LazyLinear` at `ferrotorch-nn/src/lib.rs`
  exposes the type to downstream crates.
- The `ferrotorch-train` learner scaffolding uses `LazyLinear` when
  the input feature size is determined dynamically (e.g. by a
  preceding variable-length pooling / flatten). The materialize
  path lets the user declare `LazyLinear::new(num_classes,
  bias=true)` then plug the model into a training loop that does
  the first forward against a real batch.

## Parity contract

`parity_ops = []`. `LazyLinear` is not a parity-tested kernel — its
correctness is inherited from `Linear` (verified there), and the
lazy-init plumbing is verified by lib tests in this file.

Edge cases pinned by lib tests:
- **Pre-init parameter list** — empty, including
  `named_parameters()`.
- **Post-init parameter list** — `weight` + optional `bias`.
- **Different `in_features` on second forward** — rejected with
  `ShapeMismatch` (first call wins).
- **Different `in_features` to `materialize`** — second call is a
  no-op (first wins).
- **Higher-rank input** — 3-D inputs `[B, T, in_features]`
  flatten to `[B*T, in_features]` then reshape to `[B, T,
  out_features]`, matching `Linear`.

## Verification

Tests in `mod tests` of `lazy_linear.rs` (10 tests):
- `test_lazy_linear_uninitialized_until_first_forward`,
- `test_lazy_linear_materializes_on_first_forward`,
- `test_lazy_linear_no_bias_has_one_param`,
- `test_lazy_linear_subsequent_forward_uses_initialized_weights`,
- `test_lazy_linear_rejects_mismatched_in_features`,
- `test_lazy_linear_explicit_materialize_initializes_eagerly`,
- `test_lazy_linear_materialize_idempotent`,
- `test_lazy_linear_zero_out_features_errors`,
- `test_lazy_linear_higher_rank_input`,
- `test_lazy_linear_named_parameters_after_init`,
- `test_lazy_linear_train_eval_toggle`.

Smoke command:

```bash
cargo test -p ferrotorch-nn --lib lazy_linear:: 2>&1 | tail -3
```

Expected: 11 passed.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct LazyLinear<T: Float>` with `OnceLock<Parameter<T>>` fields in `lazy_linear.rs` mirroring upstream `LazyLinear` with `UninitializedParameter` at `torch/nn/modules/linear.py:18` and the LazyModuleMixin protocol at `torch/nn/modules/lazy.py`; non-test consumer: `pub use lazy_linear::LazyLinear` in `lib.rs`. |
| REQ-2 | SHIPPED | impl: `LazyLinear::new(out_features, bias)` body in `lazy_linear.rs` rejecting `out_features == 0`; non-test consumer: dynamic-shape model construction in `ferrotorch-train`'s learner setup. |
| REQ-3 | SHIPPED | impl: `is_initialized` / `in_features` / `out_features` accessors in `lazy_linear.rs`; non-test consumer: dispatch logic in dynamic-shape model setup queries `is_initialized` to decide whether to call `materialize` eagerly. |
| REQ-4 | SHIPPED | impl: `LazyLinear::materialize` body in `lazy_linear.rs` (idempotent first-wins allocator); non-test consumer: dynamic-shape model setup calls `materialize(known_in_features)` to populate the param list before constructing the optimizer. |
| REQ-5 | SHIPPED | impl: `<LazyLinear as Module>::forward` body in `lazy_linear.rs` (materialize-on-first + `linear_fused` dispatch); non-test consumer: any model containing a `LazyLinear` runs this on every forward pass. |
| REQ-6 | SHIPPED | impl: flatten-then-reshape branch in `<LazyLinear as Module>::forward` mirroring `Linear::forward`; non-test consumer: 3-D / 4-D inputs flow through the same path in production transformer / vision usage. |
| REQ-7 | SHIPPED | impl: `Module::parameters` / `parameters_mut` / `named_parameters` build Vec from OnceLock contents in `lazy_linear.rs`; non-test consumer: `ferrotorch_optim::Optimizer` walks `model.parameters_mut()` AFTER the first forward (or after explicit materialize), at which point the lazy params surface. |
| REQ-8 | SHIPPED | impl: `kaiming_uniform(&mut w, NonLinearity::ReLU)` and `init_zeros(&mut b)` in `materialize` body in `lazy_linear.rs`; non-test consumer: every `LazyLinear` instance goes through this code path on first init. |
| REQ-9 | SHIPPED | `OnceLock::set` is documented race-safe by the standard library; the hot post-init path uses lock-free `OnceLock::get`. Verified by `Send + Sync` requirements on `OnceLock<Parameter<T>>` (held by composition of `Send + Sync` field types); non-test consumer: any multi-threaded training scaffolding requiring `Send + Sync`. |
