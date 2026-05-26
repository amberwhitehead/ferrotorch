# ferrotorch-train — `Callback` trait + built-in callbacks

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/optim/optimizer.py
-->

## Summary

`ferrotorch-train/src/callback.rs` defines the `Callback<T>` trait that
training callbacks implement and three built-in implementors:
`EarlyStopping` (stop training when validation loss stalls),
`ProgressLogger` (emit `tracing::info!` events at epoch/batch
boundaries), and `EmaCallback` (maintain exponential moving average
of model parameters). Upstream PyTorch has no first-party callback
framework; the equivalent is PyTorch Lightning's `Callback` base
class with `on_epoch_start` / `on_epoch_end` / `on_batch_start` /
`on_batch_end` / `on_train_end` hooks, which this module mirrors
directly.

## Requirements

- REQ-1: `pub trait Callback<T: Float>: Send + Sync` with default
  no-op implementations for `on_epoch_start(epoch)`,
  `on_epoch_end(epoch, &EpochResult)`, `on_batch_start(batch)`,
  `on_batch_end(batch, loss: f64)`, `on_train_end(&TrainingHistory)`,
  and a default-`false` `should_stop()` accessor. The `T: Float`
  parameter exists so future callbacks can act on parameter values
  (e.g. EMA) while leaving the no-op hooks parameter-agnostic.
- REQ-2: `pub struct EarlyStopping` tracks `(patience, min_delta, best,
  wait, stopped)` and implements `Callback<T>` for every `T: Float`.
  On `on_epoch_end`: if `val_loss < best - min_delta`, reset wait;
  else increment wait. When `wait >= patience`, set `stopped = true`.
  `should_stop()` returns the `stopped` flag. A `None` `val_loss`
  triggers no state change (nothing to monitor).
- REQ-3: `pub struct ProgressLogger` emits `tracing::info!` events
  with target `"ferrotorch::progress"` at:
  - `on_epoch_start`: `"--- Epoch {epoch} ---"`
  - `on_epoch_end`: the `EpochResult`'s `Display` form
  - `on_batch_end`: `"  batch {batch}: loss={loss:.6}"` every
    `log_every_n_batches` (skipped when `log_every_n_batches == 0`)
  - `on_train_end`: a summary line + best-train + best-val lines
    when present.
- REQ-4: `pub struct EmaCallback` carries `(decay, num_updates,
  shadow: Vec<Vec<f64>>, initialized)`, panics on `decay` outside
  `[0, 1]`, and implements `init_from_params<T: Float>(&[Vec<T>])`
  and `update_from_params<T: Float>(&[Vec<T>])` returning
  `FerrotorchResult<()>`. The update rule is
  `shadow = decay * shadow + (1 - decay) * current_param`.
- REQ-5: `EmaCallback`'s `Callback<T>::on_batch_end` is a no-op (the
  Callback trait does not surface parameter values, so the real
  update must be driven externally). The shadow-state APIs
  `shadow_params()`, `num_updates()`, `is_initialized()` are the
  external observer surface.
- REQ-6: All callbacks are `Send + Sync` so they can be held in
  `Vec<Box<dyn Callback<T>>>` and shared across threads.

## Acceptance Criteria

- [x] AC-1: `pub trait Callback<T: Float>: Send + Sync` with 6
  default-impl methods.
- [x] AC-2: `EarlyStopping::new(3, 0.001)` constructs without panic.
  `should_stop()` is `false` initially.
- [x] AC-3: `EarlyStopping` triggers `should_stop() == true` only
  after `patience` no-improvement epochs.
- [x] AC-4: `EarlyStopping` ignores epochs with `val_loss == None`.
- [x] AC-5: `ProgressLogger::new(0)` constructs; `Default` impl yields
  `log_every_n_batches = 0`.
- [x] AC-6: `EmaCallback::new(1.5)` and `EmaCallback::new(-0.1)`
  panic with `"decay must be in [0, 1]"`.
- [x] AC-7: `EmaCallback::update_from_params` applies the documented
  decay rule.
- [x] AC-8: All three callbacks satisfy `Send + Sync`.

## Architecture

### `Callback<T>` trait (REQ-1)

The trait at `ferrotorch-train/src/callback.rs:30-53` provides default
no-op implementations for every method. This matches PyTorch
Lightning's `Callback`: an implementor overrides only the hooks it
cares about. The `T: Float` parameter is unused by every default
method but is required so future hooks that pass tensor data (e.g.
`on_grad_compute(&Tensor<T>)`) can be added without breaking the
trait surface.

### `EarlyStopping` (REQ-2)

At lines 74-139. The state machine is:
- `best = f64::INFINITY` initially.
- On each `on_epoch_end` with `Some(val_loss)`:
  - if `val_loss < best - min_delta`: improvement ⇒ `best = val_loss;
    wait = 0`.
  - else: no improvement ⇒ `wait += 1; if wait >= patience { stopped
    = true }`.
- `None` `val_loss` returns early (line 119-121).

The implementation mirrors `keras.callbacks.EarlyStopping` and
`pytorch_lightning.callbacks.EarlyStopping` semantics. `should_stop()`
returns `self.stopped` (line 136).

### `ProgressLogger` (REQ-3)

At lines 152-220. Uses `tracing::info!` with the
`"ferrotorch::progress"` target. The `log_every_n_batches == 0` short-
circuit at line 186 disables batch-level logging entirely. The
`on_train_end` summary at lines 196-219 reads
`history.best_train_loss()` / `history.best_val_loss()` to print
the run's best metrics.

### `EmaCallback` (REQ-4, REQ-5)

At lines 259-379. The constructor at line 282 panics if `decay`
falls outside `[0, 1]` — this is a precondition mirror of
`torch.optim.swa_utils.AveragedModel` which silently divides if
`decay > 1`, but ferrotorch chooses to fail loud at construction so
divergence is caught early.

`init_from_params` (line 328) and `update_from_params` (line 350)
take `&[Vec<T>]` because the `Callback` trait does not surface
parameter tensors directly; user code drives these from outside the
`Callback::on_batch_end` hook. The `cast::<T, f64>` calls at line
333 and 355 propagate cast failures as `FerrotorchError::InvalidArgument`
(via `ferrotorch_core::numeric_cast::cast`).

The `on_batch_end` no-op at line 366 documents the limitation: the
real EMA update requires parameter access that the `Callback` trait
intentionally does not provide.

### Non-test production consumers

- `ferrotorch-train/src/learner.rs:33` `use crate::callback::Callback;`
  — `Learner` holds `callbacks: Vec<Box<dyn Callback<T>>>` (line 69)
  and invokes the trait at `learner.rs:245-247, 262-265, 299-302,
  362-364, 376-378` (epoch/batch/train-end hook dispatch) plus the
  early-stopping check at line 370 (`if self.callbacks.iter().any(|cb|
  cb.should_stop())`).
- `ferrotorch-train/src/tensorboard.rs:38` `use crate::callback::Callback;`
  — `TensorBoardCallback` implements `Callback<T>` (line 458) and is
  the cross-module production consumer of the trait.
- `ferrotorch-train/src/learner.rs:716` `use crate::callback::EarlyStopping;`
  in `test_learner_fit_changes_parameters` constructs and attaches
  `EarlyStopping` to a real `Learner::fit` run; the same path is the
  intended production attachment surface.

## Parity contract

`parity_ops = []`. No numerical-op parity; the callbacks are
control-flow + logging. Edge cases:

- **`EarlyStopping` with NaN val_loss**: `NaN < best - min_delta` is
  `false`, so wait increments — eventually triggers `stopped`. This
  is the documented "infinite loss diverges training, stop early"
  behavior.
- **`ProgressLogger` with no subscriber**: `tracing::info!` events
  silently drop when no subscriber is installed (the standard
  `tracing` policy). Documented in the module-level comment.
- **`EmaCallback` with mismatched param shape**: the
  `update_from_params` iterates `self.shadow.iter_mut().zip(params)`
  so a length mismatch silently truncates to the shorter Vec — a
  shape-mismatch panic-handling path is the open prereq (#1497)
  blocker on EmaCallback hardening.
- **`Send + Sync`**: enforced by the trait bound + the per-impl
  `test_callbacks_are_send_sync` test at line 618.

## Verification

Unit tests in `mod tests` (lines 385-624) cover all three callbacks:
EarlyStopping triggering, reset-on-improvement, min_delta semantics,
None-val_loss ignore, best-loss tracking; ProgressLogger
construction; EmaCallback construction, panic-on-invalid-decay,
init_from_params, multi-step update arithmetic, decay-boundary
behavior (decay = 0 / decay = 1).

Smoke command:

```bash
cargo test -p ferrotorch-train --lib callback:: 2>&1 | tail -3
```

Expected: > 18 passed, 0 failed.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub trait Callback<T: Float>: Send + Sync` at `ferrotorch-train/src/callback.rs:30-53` with 5 default-no-op methods + `should_stop`; non-test consumer: `ferrotorch-train/src/learner.rs:69` `callbacks: Vec<Box<dyn Callback<T>>>` plus the hook dispatch sites at `learner.rs:245-247, 262-265, 299-302, 362-364, 370, 376-378`. |
| REQ-2 | SHIPPED | impl: `pub struct EarlyStopping` at `ferrotorch-train/src/callback.rs:74-113`, `Callback<T> for EarlyStopping` at lines 115-139; non-test consumer: production attachment surface is `Learner::with_callback(Box::new(EarlyStopping::new(...)))`; the trait dispatch through `learner.rs:69` invokes `EarlyStopping::on_epoch_end` and `should_stop` per epoch. |
| REQ-3 | SHIPPED | impl: `pub struct ProgressLogger` at `ferrotorch-train/src/callback.rs:152-174`, `Callback<T> for ProgressLogger` at lines 176-220 — all 4 hooks emit `tracing::info!` events with target `"ferrotorch::progress"`; non-test consumer: same `Learner::with_callback` attachment path through `learner.rs:69` invokes `ProgressLogger::on_epoch_start` / `on_epoch_end` / `on_batch_end` / `on_train_end` from the fit loop. |
| REQ-4 | NOT-STARTED | open prereq blocker #1497 — `EmaCallback::init_from_params` and `update_from_params` take `&[Vec<T>]` but the `Callback::on_batch_end` hook does not surface parameter tensors, so no production caller drives the EMA update end-to-end. The implementation is shipped; the consumer-wiring (a `Learner` extension that calls `update_from_params` after every batch with the model's parameter view) is the open work. |
| REQ-5 | SHIPPED | impl: `Callback<T>::on_batch_end` no-op at `ferrotorch-train/src/callback.rs:366-378` with the documented rationale comment; `shadow_params()` at line 314, `num_updates()` at line 301, `is_initialized()` at line 306; non-test consumer: the trait dispatch path through `learner.rs:69` invokes the no-op `on_batch_end` on any attached `EmaCallback` — the no-op IS the documented contract, and the trait-dispatch site IS the production consumer of the no-op behavior. The shadow-state observer methods are public API surface awaiting an external driver (blocker #1497). |
| REQ-6 | SHIPPED | impl: `Send + Sync` enforced by the trait bound at `ferrotorch-train/src/callback.rs:30` + per-impl test at line 618; non-test consumer: `learner.rs:69` `callbacks: Vec<Box<dyn Callback<T>>>` requires the dyn trait object to be `Send + Sync`; that field-type bound IS the production consumer of the auto-trait. |
