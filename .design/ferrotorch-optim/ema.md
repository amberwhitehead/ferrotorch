# ferrotorch-optim — ExponentialMovingAverage (EMA / Polyak averaging)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/optim/swa_utils.py
-->

## Summary

`ferrotorch-optim/src/ema.rs` implements an exponential moving
average of model parameters (Polyak averaging for inference
stabilisation), mirroring the EMA pattern in
`torch.optim.swa_utils.AveragedModel` with `avg_fn=ema`
(`torch/optim/swa_utils.py:165-470`). PyTorch's API combines SWA
(Stochastic Weight Averaging) and EMA in one `AveragedModel`
class; ferrotorch separates them into `ExponentialMovingAverage`
(this file) and `crate::swa::AveragedModel` (`swa.rs`). The Rust
impl exposes the `ExponentialMovingAverage<T>` struct with
`update`, `apply_shadow`, `restore`, builder-style `with_foreach`
and `with_decay_warmup` flags.

## Requirements

- REQ-1: `pub struct ExponentialMovingAverage<T: Float>` carries
  `decay: f64`, `shadow_params: Vec<Vec<T>>` (CPU path) or
  `shadow_tensors: Vec<Tensor<T>>` (foreach path), `backup_params:
  Vec<Vec<T>>` (populated by `apply_shadow`),
  `num_updates: u64`, `use_decay_warmup: bool`, `foreach: bool`.
- REQ-2: `new(params, decay)` constructor: decay must be in
  `[0, 1]`, otherwise panics with the assertion message
  `"decay must be in [0, 1]"`. Initialises `shadow_params` by
  cloning each parameter's data via `data().unwrap_or_default()`.
- REQ-3: `update(&mut self, params)` performs the EMA step
  `shadow = decay * shadow + (1 - decay) * param` element-by-element
  on the CPU path. Increments `num_updates`. Panics on parameter-count
  mismatch.
- REQ-4: `apply_shadow(&mut self, params)` swaps the shadow values
  into the parameters (backing up the originals). `restore(&mut self,
  params)` swaps the backups back. Panics if `restore` is called
  without a prior `apply_shadow` (asserted message: `"restore()
  called without a prior apply_shadow"`).
- REQ-5: `with_foreach(&[Parameter<T>])` builder switches the impl
  to on-device tensor storage. Deep-copies each parameter into a
  fresh `Tensor` so the shadow storage is NOT aliased with the
  parameter storage (the SAFETY block at `apply_shadow`'s
  `update_storage` call relies on this).
- REQ-6: `with_decay_warmup(true)` enables the bias-correction
  warmup `effective_decay = min(decay, (1 + t) / (10 + t))` so the
  early EMA isn't dominated by the (often random) initial weights.
- REQ-7: `shadow_values(index)` returns the shadow values as a
  flat `Vec<T>` for both the CPU and foreach paths (CPU returns a
  clone; foreach reads via `Tensor::data_vec`).

## Acceptance Criteria

- [x] AC-1: `ExponentialMovingAverage::new(params, 0.999)`
  initialises `shadow_params` with deep copies of `params`.
- [x] AC-2: `new(params, 1.5)` panics with
  `"decay must be in [0, 1]"` (`test_ema_invalid_decay`).
- [x] AC-3: `test_ema_single_update` produces
  `shadow = 0.9 * [1, 2] + 0.1 * [3, 4] = [1.2, 2.2]` with
  `decay=0.9` after one update.
- [x] AC-4: `test_ema_two_updates` confirms the recurrence
  `shadow = 0.5 * shadow + 0.5 * param` over two steps.
- [x] AC-5: `test_ema_apply_and_restore` round-trips
  `[10, 20] -> [5, 10] (apply_shadow) -> [0, 0] (restore)`.
- [x] AC-6: `test_ema_decay_warmup` confirms
  `effective_decay = (1+t)/(10+t)` for `t=0`: shadow
  approaches the parameter quickly.
- [x] AC-7: `test_ema_decay_one_freezes_shadow` confirms
  `decay=1.0` produces zero update.
- [x] AC-8: Foreach-parity tests
  (`test_ema_foreach_basic_parity`,
  `test_ema_foreach_parity_with_decay_warmup`,
  `test_ema_foreach_apply_and_restore`) confirm CPU and foreach
  paths agree to within 1e-3 (warmup case) / 1e-5 (no warmup).
- [x] AC-9: `restore` without a prior `apply_shadow` panics
  (`test_ema_restore_without_apply_panics`).

## Architecture

### `ExponentialMovingAverage<T>` (REQ-1)

A `#[derive(Debug, Clone)]` struct with no autograd interaction
(shadow tensors are constructed `requires_grad=false`, and updates
run inside `no_grad`). Builder methods consume `self` and return
`Self`.

### `new` constructor (REQ-2)

```rust
pub fn new(params: &[Parameter<T>], decay: f64) -> Self {
    assert!((0.0..=1.0).contains(&decay), "decay must be in [0, 1], got {decay}");
    let shadow_params: Vec<Vec<T>> = params.iter()
        .map(|p| p.data().unwrap_or_default().to_vec())
        .collect();
    Self { decay, shadow_params, ..Default::default()-like-defaults }
}
```

The `data().unwrap_or_default()` is the standard read pattern;
since this is a builder method (not in a `Result` context),
parameter read failures degrade to an empty shadow rather than
propagate.

### `update` (REQ-3)

CPU path: a per-element scalar loop computing
`s = decay_t * s + (1 - decay_t) * p` where `decay_t = cast::<f64, T>(decay)`.
Parameter-count mismatch panics with a descriptive message.

Foreach path (`update_foreach`): uses tensor ops
`mul(shadow, scalar(decay))` + `mul(param, scalar(1 - decay))` +
`add(...)`, all inside `no_grad`. The shadow tensor is replaced
via `*shadow = new_shadow;` — no `update_storage` is needed
because the shadow tensor is owned by `self.shadow_tensors[i]`,
not aliased with the parameter.

### `apply_shadow` / `restore` (REQ-4)

`apply_shadow`:
1. Save current `params.data().unwrap_or_default().to_vec()`
   into `backup_params`.
2. Foreach path: clone each shadow tensor and `update_storage`
   the cloned storage into the parameter. The SAFETY block
   documents the four sole-writer invariants AND explicitly
   cites `ferrotorch-core/src/tensor.rs:716`'s
   `into_storage_and_shape` deep-copy guarantee at refcount>1.
3. CPU path: `unsafe { param.tensor().update_data(shadow) }`
   inside `no_grad`.

`restore`:
1. Assert `!backup_params.is_empty()` (panics if no prior
   `apply_shadow`).
2. Write `backup_params[i]` back into each parameter via
   `update_data` (unsafe; SAFETY block).
3. Clear `backup_params` so a second `restore` without a new
   `apply_shadow` panics.

### `with_foreach` (REQ-5)

Builder method that:
1. Asserts `params.len() == self.shadow_params.len()`.
2. Reads each `param.data_vec()` (handles GPU/non-contiguous via
   the centralised path).
3. Constructs a fresh `Tensor::from_storage(TensorStorage::cpu(data),
   shape, false)` and moves it to the parameter's device via
   `.to(device)`.
4. Clears `self.shadow_params` (now unused) and sets
   `self.foreach = true`.

The deep-copy is load-bearing: `apply_shadow`'s
`update_storage` SAFETY relies on the shadow tensor's storage
being disjoint from the parameter's storage. The deep-copy
through `data_vec() -> TensorStorage::cpu(data) -> to(device)`
ensures a fresh storage Arc.

### `with_decay_warmup` (REQ-6)

Sets `use_decay_warmup = true`. `effective_decay()` then returns
`self.decay.min((1.0 + t) / (10.0 + t))` instead of the raw
`self.decay`.

### Non-test production consumers

- `ferrotorch-optim/src/lib.rs:35` — `pub use
  ema::ExponentialMovingAverage;` is the crate-public surface.
- `ferrotorch/src/lib.rs:61` — `pub use ferrotorch_optim::*;`
  re-exports `ExponentialMovingAverage` through the umbrella
  crate's `optim` module.

There is NO direct consumer of `ExponentialMovingAverage` in
`ferrotorch-train` or in the umbrella `examples/`. The type is
exposed as public API for downstream user code (e.g. a training
loop that wants Polyak averaging at inference time). The
re-export chain `ferrotorch-optim/src/lib.rs:35 ->
ferrotorch/src/lib.rs:61` is the production-consumer wiring; the
type is part of the umbrella crate's stable surface.

## Parity contract

`parity_ops = []`. Edge-cases:

- **Builder construction panics**: `new(params, decay)` is the
  one construction path that can panic (on `decay` out of range).
  `with_foreach` panics on length mismatch. These are tested.
- **`apply_shadow` followed by `restore`**: idempotent in one
  direction (apply once, restore once). Calling `apply_shadow`
  twice without intervening `restore` overwrites the backup —
  the same as PyTorch's behaviour with `swap_tensors_with_load`
  semantics.
- **CPU/foreach interop on read**: `shadow_values(index)` works
  for both paths; `shadow_params()` only returns the CPU-path
  slice (foreach callers should use `shadow_values`).
- **`decay = 1.0` freezes the shadow**: confirmed by
  `test_ema_decay_one_freezes_shadow`.
- **`decay = 0.0`**: shadow = param (zero memory of history). Used
  in the `test_ema_multiple_params` test.

## Verification

Tests in `mod tests in ema.rs` (12 tests):

- Construction: `test_ema_construction`, `test_ema_invalid_decay`.
- Update: `test_ema_single_update`, `test_ema_two_updates`.
- Apply/restore: `test_ema_apply_and_restore`,
  `test_ema_restore_without_apply_panics`.
- Warmup: `test_ema_decay_warmup`.
- Multi-param: `test_ema_multiple_params`.
- Decay edge: `test_ema_decay_one_freezes_shadow`.
- Foreach: `test_ema_foreach_basic_parity`,
  `test_ema_foreach_parity_with_decay_warmup`,
  `test_ema_foreach_apply_and_restore`.

Smoke command:

```bash
cargo test -p ferrotorch-optim --lib ema:: 2>&1 | tail -3
```

Expected: `12 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct ExponentialMovingAverage<T: Float>` with the seven `decay, shadow_params, shadow_tensors, backup_params, num_updates, use_decay_warmup, foreach` fields in `ema.rs`; non-test consumer: `ferrotorch-optim/src/lib.rs:35` re-exports the type; `ferrotorch/src/lib.rs:61` re-exports the optim surface to umbrella-crate users. |
| REQ-2 | SHIPPED | impl: `pub fn new(params: &[Parameter<T>], decay: f64) -> Self` with `assert!((0.0..=1.0).contains(&decay), ...)` in `ema.rs`; non-test consumer: `ferrotorch/src/lib.rs:61` re-exports the type, making this the only production construction path. |
| REQ-3 | SHIPPED | impl: `pub fn update(&mut self, params: &[Parameter<T>]) -> FerrotorchResult<()>` method on `ExponentialMovingAverage` in `ema.rs` mirroring the EMA recurrence used by `torch/optim/swa_utils.py:267-310` (`avg_fn = ema`); non-test consumer: `ferrotorch/src/lib.rs:61` re-exports the type for downstream training code to call after each optimizer step. |
| REQ-4 | SHIPPED | impl: `apply_shadow` and `restore` methods on `ExponentialMovingAverage` in `ema.rs`; non-test consumer: same as REQ-3 — `ferrotorch/src/lib.rs:61` re-exports the public API, so downstream inference code calls `apply_shadow` before eval and `restore` after. |
| REQ-5 | SHIPPED | impl: `pub fn with_foreach(mut self, params: &[Parameter<T>]) -> Self` builder method in `ema.rs` performing deep-copy via `data_vec` + `Tensor::from_storage` + `to(device)`; non-test consumer: `ferrotorch/src/lib.rs:61` re-exports the type, making `.with_foreach(...)` reachable from downstream GPU-training code. |
| REQ-6 | SHIPPED | impl: `pub fn with_decay_warmup(mut self, use_warmup: bool) -> Self` builder method + `fn effective_decay(&self)` impl in `ema.rs`; non-test consumer: `ferrotorch/src/lib.rs:61` re-exports the type, making `.with_decay_warmup(true)` reachable. |
| REQ-7 | SHIPPED | impl: `pub fn shadow_values(&self, index: usize) -> FerrotorchResult<Vec<T>>` method in `ema.rs`; non-test consumer: `ferrotorch/src/lib.rs:61` re-exports the type, making this accessor reachable for downstream code reading the shadow into a checkpoint or log. |
