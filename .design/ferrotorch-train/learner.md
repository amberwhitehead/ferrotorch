# ferrotorch-train — `Learner` + `LossFn`

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/optim/optimizer.py
  - torch/amp/__init__.py
  - torch/utils/data/dataloader.py
-->

## Summary

`ferrotorch-train/src/learner.rs` defines the `Learner<M, T>` struct
that composes a model `M: Module<T>`, an `Optimizer<T>`, a `LossFn<T>`,
an optional `LrScheduler<T>`, an optional `GradScaler<T>` for AMP,
optional training/validation `Metric<Input = f64>` vectors, an
optional `Callback<T>` vector, and an optional `checkpoint_dir` into
a single object that drives the training loop via `fit(...)` and
evaluation via `evaluate(...)`. Mirrors the role of PyTorch
Lightning's `Trainer` and fastai's `Learner` — there is no first-party
`torch.Trainer`, so the design adapts the established third-party
patterns.

## Requirements

- REQ-1: `pub type LossFn<T> = Box<dyn Fn(&Tensor<T>, &Tensor<T>) ->
  FerrotorchResult<Tensor<T>> + Send + Sync>` — a boxed closure that
  takes `(prediction, target)` and returns a scalar loss tensor.
  Boxed rather than generic so `Learner<M, T>` is concrete on `T`
  alone and the user can swap loss functions without type churn.
- REQ-2: `pub struct Learner<M, T: Float>` holds the 11 fields
  documented in the module (`model`, `optimizer`, `loss_fn`,
  `scheduler`, `grad_scaler`, `train_metrics`, `val_metrics`,
  `callbacks`, `checkpoint_dir`, `epoch`, `step`, `skipped_steps`).
- REQ-3: `Learner::new(model, optimizer, loss_fn)` is the canonical
  constructor; `with_scheduler`, `with_grad_scaler`, `with_amp_context`,
  `with_train_metric`, `with_val_metric`, `with_callback`,
  `with_ema_callback`, `with_tensorboard_callback`, `with_grad_clip_norm`,
  `with_checkpointing` are builder-style consume-and-return-self
  mutators that match `Trainer.compile(...).fit(...)`-style chaining.
  The `with_tensorboard_callback` dedicated builder (closes #1504) is
  the analogue of `with_ema_callback` — it accepts a
  `TensorBoardCallback` by value and appends `Box::new(tb)` to the
  generic callback chain so `fit`'s standard `on_*` dispatch reaches
  it, giving the callback a non-vocab-only production attachment site.
- REQ-4: `Learner::fit(train_data, val_data, num_epochs) ->
  FerrotorchResult<TrainingHistory>` runs the documented training
  loop:
  - reset `skipped_steps`;
  - for each epoch:
    - dispatch `on_epoch_start`, reset training metrics, set model to
      `train()` mode;
    - for each batch: forward, loss, AMP-scaled backward + scaler-step
      OR standard backward + optimizer step + zero_grad, update step
      counter, update metrics, dispatch `on_batch_end`;
    - scheduler step (per-epoch);
    - validation pass if `val_data` is `Some`;
    - construct `EpochResult` via `EpochResult::new_with_defaults(epoch,
      train_loss, val_loss, lr)` then assign `metrics` and
      `duration_secs` (closes #1498 — gives the helper a non-test
      production caller); save checkpoint if `checkpoint_dir` is
      `Some`, dispatch `on_epoch_end`, push to history, increment
      epoch counter;
    - break if any callback returns `should_stop() == true`;
  - dispatch `on_train_end`, return history.
- REQ-5: `Learner::evaluate(val_data) -> FerrotorchResult<EvalResult>`
  resets validation metrics and runs `evaluate_iter`, which sets the
  model to `eval()` mode and runs forward + loss in
  `ferrotorch_core::no_grad(...)`. The model stays in eval mode after
  return.
- REQ-6: `Learner::load_checkpoint(path) -> FerrotorchResult<()>`
  reads a `TrainingCheckpoint<T>` from disk via
  `ferrotorch_serialize::load_checkpoint`, restores the model state
  with `strict = true`, restores the optimizer state, and restores
  the `epoch` + `step` counters.
- REQ-7: AMP path (`grad_scaler: Some(_)`) scales the loss before
  backward, calls `scaler.step(optimizer)` (which unscales and
  returns `false` on inf/NaN), increments `skipped_steps` when the
  step is skipped, calls `scaler.update()` to dynamically tune the
  scale, then `optimizer.zero_grad()`. Mirrors
  `torch.cuda.amp.GradScaler`'s documented step pattern at
  `torch/amp/__init__.py:1-50` + `torch/amp/grad_scaler.py`.
- REQ-8: `model()` / `model_mut()` / `epoch()` / `step()` /
  `skipped_steps()` accessors for external observation.

## Acceptance Criteria

- [x] AC-1: `pub type LossFn<T>` is `Box<dyn Fn(...) + Send + Sync>`.
- [x] AC-2: `Learner` has all 11 fields with the documented types.
- [x] AC-3: All 6 `with_*` builders return `Self`.
- [x] AC-4: `fit` decreases a real MSE loss across 5 epochs of SGD on
  a 1-D linear fixture (pinned by
  `test_learner_fit_with_metrics_decreases_loss`).
- [x] AC-5: `fit` visibly changes the model parameter (pinned by
  `test_learner_fit_changes_parameters` — the canonical sabotage
  probe).
- [x] AC-6: `evaluate` returns analytically correct MSE (pinned by
  `test_learner_evaluate_reports_meaningful_loss`).
- [x] AC-7: Constructed `Learner` has `grad_scaler == None`,
  `skipped_steps == 0`.
- [x] AC-8: `with_grad_scaler` attaches a scaler observable by reading
  the field through the unit test.

## Architecture

### `LossFn<T>` type alias (REQ-1)

At `ferrotorch-train/src/learner.rs:45-46`. The `Box<dyn Fn(...) + Send
+ Sync>` shape mirrors how PyTorch users pass loss functions: a
`callable(pred, target) -> scalar_tensor`. Boxing keeps `Learner<M, T>`
concrete on `T` rather than parameterising on the loss-fn type.

### `Learner` struct (REQ-2, REQ-3)

The struct at lines 57-77 owns:
- `model: M` — generic on the module type so monomorphisation gives
  zero dyn-dispatch on the forward path.
- `optimizer: Box<dyn Optimizer<T>>` — boxed because the user picks
  the optimizer at construction time.
- `loss_fn: LossFn<T>` — boxed closure.
- `scheduler: Option<Box<dyn LrScheduler<T>>>` — optional.
- `grad_scaler: Option<GradScaler<T>>` — optional AMP scaler (#595).
- `train_metrics`, `val_metrics: Vec<Box<dyn Metric<Input = f64>>>`
  — `f64` input restriction documented under `metric.md`.
- `callbacks: Vec<Box<dyn Callback<T>>>` — optional hook vector.
- `checkpoint_dir: Option<PathBuf>` — when set, fit saves
  `checkpoint_epoch_{epoch}.ftc` files each epoch.
- `epoch`, `step`, `skipped_steps: usize` — counters.

The builder methods at lines 105-161 follow the canonical "consume
self, mutate, return self" Rust pattern. None of them returns
`&mut Self` — they all return `Self`, which lets the
`Learner::new(...).with_X(...).with_Y(...)` chain produce the final
struct in one expression.

### `fit` loop (REQ-4, REQ-7)

At lines 227-381. The structure:
1. Reset `skipped_steps` (line 239).
2. For each epoch:
   - Dispatch `on_epoch_start` (line 245-247).
   - Reset training metrics (line 250-252).
   - `self.model.train()` (line 255).
   - Loop batches (line 259-303):
     - Dispatch `on_batch_start`.
     - `forward` → `loss` → cast to `f64`.
     - If AMP: `scale` → `backward` → `scaler.step(optimizer)` → bump
       `skipped_steps` if not stepped → `scaler.update()` →
       `zero_grad`.
     - Else: `backward` → `optimizer.step()` → `zero_grad`.
     - Track loss, update metrics, dispatch `on_batch_end`.
   - Scheduler step (line 306-308).
   - Validation phase if `val_data: Some`.
   - Construct `EpochResult` literal (line 352-359).
   - Save checkpoint if `checkpoint_dir: Some` (line 340-350).
   - Dispatch `on_epoch_end`, push to history, increment epoch.
   - Break if any `callback.should_stop()`.
3. Dispatch `on_train_end`.

The AMP scaffolding at line 272-281 mirrors PyTorch's documented
`scaler.scale(loss).backward(); scaler.step(opt); scaler.update();
opt.zero_grad()` recipe.

### `evaluate` / `evaluate_iter` (REQ-5)

At lines 394-445. `evaluate` resets validation metrics, then calls
`evaluate_iter`. `evaluate_iter` sets the model to `eval()` mode,
runs the forward + loss inside `ferrotorch_core::no_grad(...)`, and
returns `EvalResult { loss, metrics }`.

The model staying in eval mode after return is the documented
contract — `test_learner_evaluate_reports_meaningful_loss` asserts
this at line 804.

### `load_checkpoint` (REQ-6)

At lines 195-202. Reads `TrainingCheckpoint<T>` via
`ferrotorch_serialize::load_checkpoint`, then restores model + optimizer
state and the counters. The strict-load (`strict = true`) is a
deliberate departure from PyTorch's permissive `load_state_dict`
default — ferrotorch fails loud on parameter-name mismatches so a
silent shape divergence is caught immediately.

### Non-test production consumers

- `ferrotorch-train/examples/multi_epoch_train_dump.rs` — the
  multi-epoch training-trajectory dump binary (real-artifact parity
  harness, #1161) consumes the `Learner` API to run a 5-epoch Adam
  training loop on a 3-layer MLP and writes per-epoch state to disk.
- The internal `Learner` field types are themselves production
  consumers of their boxed trait objects: `Box<dyn Optimizer<T>>` is
  produced by `ferrotorch-optim`'s concrete optimizers; `Box<dyn
  LrScheduler<T>>` by `ferrotorch-optim/src/scheduler/*`; `Box<dyn
  Callback<T>>` and `Box<dyn Metric<Input = f64>>` by this crate's
  callback/metric impls.

## Parity contract

`parity_ops = []`. The `Learner` is structural composition; numerical
parity is owned by the underlying ops (mse_loss, optimizer step, etc.).
Edge cases the `Learner` itself owns:

- **Empty `train_data` iterator**: `train_batch_count == 0` ⇒
  `train_loss = 0.0` (line 310-314). Mirrors PyTorch's no-op-when-
  empty-DataLoader behavior.
- **AMP step skipped**: `stepped = false` from `scaler.step` ⇒
  `skipped_steps` increments, `scaler.update()` lowers the scale, the
  optimizer parameter is unchanged. `zero_grad` is still called so
  the next batch starts clean.
- **`load_checkpoint` with strict mismatch**: propagates the
  underlying `FerrotorchError` from `model.load_state_dict(_, true)`
  — the caller sees the strict-mode error message.
- **`evaluate` with empty `val_data`**: `val_batch_count == 0` ⇒
  `loss = 0.0`, empty metrics map. Documented in the module.
- **`fit` with `num_epochs = 0`**: returns an empty
  `TrainingHistory`; no model state changes.

## Verification

Unit tests in `mod tests` (lines 452-842):
- `test_learner_construction` (line 615) pins the constructor + epoch
  / step start at 0.
- `test_learner_with_checkpoint_dir` (line 626) pins the builder.
- `test_learner_fit_with_metrics_decreases_loss` (line 656) — real
  MSE + SGD path, 5 epochs, asserts loss halves and reaches < 0.5.
- `test_learner_fit_changes_parameters` (line 714) — canonical
  sabotage probe: asserts the weight moves toward `TRUE_W = 3.0` by
  ≥ 0.5 over 5 epochs.
- `test_learner_evaluate_reports_meaningful_loss` (line 771) — pins
  the analytic MSE expectation `21.875` for the linear fixture.
- `test_learner_grad_scaler_field_starts_none` /
  `test_learner_with_grad_scaler_attaches` /
  `test_learner_skipped_steps_counter_starts_zero` (lines 811-841)
  pin the AMP attachment surface.

Smoke command:

```bash
cargo test -p ferrotorch-train --lib learner:: 2>&1 | tail -3
```

Expected: > 7 passed, 0 failed (the real-loss tests are the slowest
but each completes in well under 1 second).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub type LossFn<T>` at `ferrotorch-train/src/learner.rs:45-46`; non-test consumer: `Learner::new(..., loss_fn: LossFn<T>)` at line 87 accepts the type; the `Learner.loss_fn` field at line 60 is invoked at `learner.rs:269, 420` as `(self.loss_fn)(&output, &target)?`. |
| REQ-2 | SHIPPED | impl: `pub struct Learner<M, T: Float>` at `ferrotorch-train/src/learner.rs:57-77` with all 11 fields; non-test consumer: every method body in `impl<M: Module<T>, T: Float> Learner<M, T>` (lines 79-446) reads/writes these fields directly. |
| REQ-3 | SHIPPED | impl: `Learner::new` at `ferrotorch-train/src/learner.rs:87-102`, `with_scheduler` at `:105`, `with_grad_scaler` at `:122`, `with_train_metric` at `:137`, `with_val_metric` at `:143`, `with_callback` at `:149`, `with_checkpointing` at `:158`; non-test consumer: `examples/multi_epoch_train_dump.rs` uses `Learner::new(...)` + builder chain for the real-artifact trajectory dump. |
| REQ-4 | SHIPPED | impl: `fit` at `ferrotorch-train/src/learner.rs:227-381` runs the documented epoch loop; non-test consumer: `examples/multi_epoch_train_dump.rs` invokes `Learner::fit` end-to-end on a 3-layer MLP + Adam to produce the 5-epoch state-dict trajectory for the parity harness. |
| REQ-5 | SHIPPED | impl: `evaluate` at `ferrotorch-train/src/learner.rs:394-404`, `evaluate_iter` at `:407-445` (model to eval mode at line 411, `no_grad` wrap at line 416); non-test consumer: `Learner::fit` calls `self.evaluate_iter(val_fn)` at line 323 whenever validation data is supplied. |
| REQ-6 | SHIPPED | impl: `load_checkpoint` at `ferrotorch-train/src/learner.rs:328-335` (calls `ferrotorch_serialize::load_checkpoint`, then `Module::load_state_dict(strict=true)` and `Optimizer::load_state_dict`, restoring epoch/step counters); non-test consumer: `ferrotorch-train/examples/multi_epoch_train_dump.rs:607` invokes `learner.load_checkpoint(resume_path)?` when `--resume <path>` is passed to the binary, restoring state before `Learner::fit`. (closes #1499) |
| REQ-7 | SHIPPED | impl: AMP fit-loop branches at `ferrotorch-train/src/learner.rs:421-434` (`AmpContext::backward_step` path) and `:435-444` (standalone `GradScaler` path) both honoured inside `Learner::fit`; non-test consumer: `ferrotorch-train/examples/multi_epoch_train_dump.rs:592-597` attaches an `AmpContext` via `Learner::with_amp_context` and runs `Learner::fit` end-to-end on every default invocation of the binary, exercising the AMP backward path (closes #1500). |
| REQ-8 | SHIPPED | impl: accessors `model()` at `ferrotorch-train/src/learner.rs:164`, `model_mut()` at `:169`, `epoch()` at `:174`, `step()` at `:179`, `skipped_steps()` at `:129`; non-test consumer: `examples/multi_epoch_train_dump.rs` reads `learner.model()` to snapshot parameter state after each epoch via the model's `state_dict()`. |

