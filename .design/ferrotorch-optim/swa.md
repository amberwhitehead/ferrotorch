# `swa` — Stochastic Weight Averaging utilities

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/optim/swa_utils.py
-->

## Summary

`ferrotorch-optim/src/swa.rs` ships `AveragedModel<T>` and `Swalr` —
the SWA running-average tracker and the SWA-specific learning rate
scheduler. Mirrors `torch.optim.swa_utils.AveragedModel`
(`torch/optim/swa_utils.py:165`) and `torch.optim.swa_utils.SWALR`
(`torch/optim/swa_utils.py:430`). Reference: Izmailov et al.,
"Averaging Weights Leads to Wider Optima and Better Generalization"
(UAI 2018).

## Requirements

- REQ-1: `pub enum AveragingStrategy { Swa, Ema(f64) }` matching
  upstream's `get_swa_avg_fn()` and `get_ema_avg_fn(decay)`
  (`torch/optim/swa_utils.py:42-153`). The `Ema(decay)` variant must
  panic if `decay` is outside `[0, 1]`.
- REQ-2: `pub struct AveragedModel<T: Float>` with
  `new(params, strategy)` that captures the initial parameter snapshot
  into an internal averaged-vector store. Subsequent
  `update_parameters(&[Parameter<T>])` calls update the average per the
  strategy.
- REQ-3: SWA running mean update:
  `avg = avg + (param - avg) / (n + 1)` (numerically stable equal-weight
  average). Mirrors `get_swa_avg_fn()` at
  `torch/optim/swa_utils.py:153-164`.
- REQ-4: EMA update: `avg = decay * avg + (1 - decay) * param`. Mirrors
  `get_ema_avg_fn(decay)`.
- REQ-5: First update is always a snapshot copy regardless of strategy
  (matches upstream's `n_averaged == 0` branch at
  `torch/optim/swa_utils.py:265-285`).
- REQ-6: `apply_to(&[Parameter<T>])` copies the averaged values back
  into the supplied parameters; takes `&mut self` as a borrow-checker
  witness for the sole-writer contract on the underlying tensor
  storage. Documented in the SAFETY block at `swa.rs`.
- REQ-7: `with_foreach(params)` switches to the on-device (Tensor) path
  — averaged state lives as `Tensor<T>` on the parameter's device and
  the update composes via `add`/`mul`/`sub`/`div`. Used to avoid per-step
  CPU↔GPU round-trips (CL-497).
- REQ-8: `pub enum AnnealStrategy { Cosine, Linear }` and
  `pub struct Swalr` (LR scheduler). Cosine variant uses
  `alpha = (1 - cos(pi * t)) / 2`; linear uses `alpha = t`.
- REQ-9: `Swalr::new(swa_lr, anneal_epochs, anneal_strategy)`; on the
  first `step(&mut opt)` call, the scheduler captures the optimizer's
  current LR as `initial_lr`, then interpolates from `initial_lr` to
  `swa_lr` over `anneal_epochs` steps using the configured anneal
  factor. After `anneal_epochs`, the LR stays at `swa_lr`.
- REQ-10: `Swalr` implements `LrScheduler<T> for Swalr` (the trait from
  `crate::scheduler`), giving it the standard `step(&mut opt) /
  get_lr()` surface.

## Acceptance Criteria

- [x] AC-1: `AveragedModel::new` with `Ema(1.5)` panics with
  `"EMA decay must be in [0, 1]"` (`test_averaged_model_ema_invalid_decay`).
- [x] AC-2: First SWA update copies the parameter snapshot
  (`test_averaged_model_swa_first_update_copies`).
- [x] AC-3: SWA running mean over `[1, 3, 6]` reaches `10/3 ≈ 3.333`
  exactly (`test_averaged_model_swa_running_mean`).
- [x] AC-4: Averaging 4 identical values yields that value exactly
  (`test_averaged_model_swa_equal_weight_mean`).
- [x] AC-5: EMA with `decay=0.5` over `[10, 20]` yields `15.0`
  exactly (`test_averaged_model_ema`).
- [x] AC-6: `apply_to(&[p])` copies averaged values into the parameter
  (`test_averaged_model_apply_to`).
- [x] AC-7: Parameter count mismatch in `update_parameters` panics
  with `"parameter count mismatch"`
  (`test_averaged_model_param_count_mismatch`).
- [x] AC-8: Cosine `Swalr` reaches `swa_lr` exactly after `anneal_epochs`
  (`test_swalr_cosine_annealing`).
- [x] AC-9: Cosine midpoint hits exactly `(initial + swa_lr) / 2`
  (`test_swalr_cosine_midpoint`).
- [x] AC-10: LR stays at `swa_lr` after the anneal phase
  (`test_swalr_stays_at_swa_lr_after_anneal`).
- [x] AC-11: Linear annealing reaches `swa_lr` exactly at
  `anneal_epochs` and stays there (`test_swalr_linear_annealing`).
- [x] AC-12: `anneal_epochs == 0` ⇒ immediate switch to `swa_lr` on
  the first `step` (`test_swalr_immediate_switch`).
- [x] AC-13: `Swalr::get_lr()` returns the current interpolated LR
  (`test_swalr_get_lr`).
- [x] AC-14: Foreach path matches legacy CPU within `1e-4` for SWA and
  EMA (`test_averaged_model_swa_foreach_parity`,
  `test_averaged_model_ema_foreach_parity`).
- [x] AC-15: Foreach + `apply_to` writes the averaged tensor into the
  parameter on-device (`test_averaged_model_swa_foreach_apply_to`).
- [ ] AC-16: Module-cloning AveragedModel wrapper (deepcopy whole
  `nn::Module`, override `forward`, supply `update_bn()` over a
  DataLoader). Blocked by #1470 — current implementation operates on
  `&[Parameter<T>]` slices rather than wrapping a whole module.

## Architecture

### `AveragedModel<T>` — two storage paths

The CPU-vec path keeps `averaged_params: Vec<Vec<T>>` (one per
parameter) and runs the SWA / EMA algebra in scalar `T`. Initialized in
`new(params, strategy)` by snapshotting each parameter's
`p.data().unwrap_or_default().to_vec()`.

The foreach path (`with_foreach(params)`) replaces the vec store with
`averaged_tensors: Vec<Tensor<T>>` on the parameter's device. Buffers
are constructed via `Tensor::from_storage(TensorStorage::cpu(data), shape,
false)?.to(device)?` to guarantee a deep copy (no aliasing of
parameter storage).

### `update_parameters` & `apply_to`

`update_parameters(&[Parameter<T>])` dispatches to
`update_parameters_foreach` when the foreach path is active; otherwise
it runs the scalar SWA/EMA update under `no_grad`. On the first call,
both paths copy the snapshot.

`apply_to(&mut self, params)` takes `&mut self` as the borrow-checker
witness for sole-writer access to the parameter's storage; the SAFETY
blocks at `swa.rs` and `swa.rs` document the four
invariants for `update_storage` / `update_data`.

### `Swalr` interpolation

```rust
let t = (current_step / anneal_epochs).clamp(0.0, 1.0);
let alpha = match anneal_strategy {
    AnnealStrategy::Cosine => (1.0 - (PI * t).cos()) / 2.0,
    AnnealStrategy::Linear => t,
};
let lr = initial_lr * (1.0 - alpha) + swa_lr * alpha;
optimizer.set_lr(lr);
```

`initial_lr` is captured lazily on the first `step()` call so the
scheduler can be constructed before the optimizer's starting LR is
known. `anneal_epochs == 0` short-circuits to immediate switch.

### Non-test production consumers

`ferrotorch/src/lib.rs` `pub use ferrotorch_optim::*;` re-exports
`AveragedModel`, `AveragingStrategy`, `AnnealStrategy`, and `Swalr` as
`ferrotorch::optim::{AveragedModel, AveragingStrategy, AnnealStrategy,
Swalr}`.

`Swalr` implements `LrScheduler<T> for Swalr` (`swa.rs`), and
`crate::scheduler::LrScheduler` is consumed across the whole scheduler
family — any downstream training loop using a scheduler dispatch can
plug `Swalr` in.

## Parity contract

`parity_ops = []`. SWA-utilities parity is asserted via the unit-test
gauntlet, including the foreach-path cross-parity tests
(`test_averaged_model_*_foreach_parity`). The known divergence vs.
upstream is the module-cloning scope (#1470) — current ferrotorch
operates on parameter slices, not full `nn::Module` deepcopies, so
`update_bn()` and `AveragedModel.forward` are absent.

Edge cases the code owns:

- **First update** (`n_averaged == 0`) — both paths copy the parameter
  snapshot regardless of strategy.
- **EMA with `decay = 1.0`** — averaged values never update after the
  first snapshot.
- **EMA with `decay = 0.0`** — averaged values track the parameter
  exactly (no smoothing).
- **`Swalr` first call captures initial_lr** from the optimizer; later
  changes to the optimizer's `lr` between `step` calls do NOT update
  `initial_lr`.
- **`Swalr` linear annealing past `anneal_epochs`** — `t` is clamped to
  `1.0` so `lr` stays at `swa_lr`.

## Verification

Tests in `mod tests` of `swa.rs` (16 tests):

- `test_averaged_model_swa_first_update_copies`
- `test_averaged_model_swa_running_mean`
- `test_averaged_model_swa_equal_weight_mean`
- `test_averaged_model_ema`
- `test_averaged_model_apply_to`
- `test_swalr_cosine_annealing`
- `test_swalr_cosine_midpoint`
- `test_swalr_stays_at_swa_lr_after_anneal`
- `test_swalr_linear_annealing`
- `test_swalr_immediate_switch`
- `test_swalr_get_lr`
- `test_averaged_model_ema_invalid_decay` (should panic)
- `test_averaged_model_param_count_mismatch` (should panic)
- `test_averaged_model_swa_foreach_parity` (CL-497)
- `test_averaged_model_ema_foreach_parity` (CL-497)
- `test_averaged_model_swa_foreach_apply_to` (CL-497)

Smoke command:

```bash
cargo test -p ferrotorch-optim --lib swa:: 2>&1 | tail -3
```

Expected: `16 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub enum AveragingStrategy { Swa, Ema(f64) }` at `ferrotorch-optim/src/swa.rs` + decay-range panic at `ferrotorch-optim/src/swa.rs` mirroring `torch/optim/swa_utils.py:42-153`; non-test consumer: `ferrotorch/src/lib.rs` `pub use ferrotorch_optim::*;` re-export. |
| REQ-2 | SHIPPED | impl: `pub struct AveragedModel<T>` at `ferrotorch-optim/src/swa.rs` + `new` constructor at `ferrotorch-optim/src/swa.rs`; non-test consumer: `ferrotorch/src/lib.rs` re-export. |
| REQ-3 | SHIPPED | impl: SWA update at `ferrotorch-optim/src/swa.rs` (legacy path) + `ferrotorch-optim/src/swa.rs` (foreach path) mirroring `torch/optim/swa_utils.py:153-164`; non-test consumer: `ferrotorch/src/lib.rs` re-export. |
| REQ-4 | SHIPPED | impl: EMA update at `ferrotorch-optim/src/swa.rs` (legacy) + `ferrotorch-optim/src/swa.rs` (foreach); non-test consumer: `ferrotorch/src/lib.rs` re-export. |
| REQ-5 | SHIPPED | impl: `n_averaged == 0` snapshot branches at `ferrotorch-optim/src/swa.rs` and `ferrotorch-optim/src/swa.rs`; non-test consumer: `ferrotorch/src/lib.rs` re-export. |
| REQ-6 | SHIPPED | impl: `apply_to(&mut self, params)` at `ferrotorch-optim/src/swa.rs` with documented sole-writer SAFETY block; non-test consumer: `ferrotorch/src/lib.rs` re-export. |
| REQ-7 | SHIPPED | impl: `with_foreach(params)` at `ferrotorch-optim/src/swa.rs` + `update_parameters_foreach` at `ferrotorch-optim/src/swa.rs`; non-test consumer: `ferrotorch/src/lib.rs` re-export. |
| REQ-8 | SHIPPED | impl: `pub enum AnnealStrategy { Cosine, Linear }` at `ferrotorch-optim/src/swa.rs` + `anneal_factor` at `ferrotorch-optim/src/swa.rs` mirroring `torch/optim/swa_utils.py:430`; non-test consumer: `ferrotorch/src/lib.rs` re-export. |
| REQ-9 | SHIPPED | impl: `pub struct Swalr` at `ferrotorch-optim/src/swa.rs` + interpolation step at `ferrotorch-optim/src/swa.rs`; non-test consumer: `ferrotorch/src/lib.rs` re-export. |
| REQ-10 | SHIPPED | impl: `impl<T: Float> LrScheduler<T> for Swalr` at `ferrotorch-optim/src/swa.rs`; non-test consumer: `ferrotorch/src/lib.rs` re-export and the `LrScheduler` trait at `crate::scheduler` is consumed by every scheduler in the family. |
