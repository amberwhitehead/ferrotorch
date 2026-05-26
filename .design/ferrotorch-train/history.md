# ferrotorch-train — `EpochResult` / `EvalResult` / `TrainingHistory`

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/optim/optimizer.py
-->

## Summary

`ferrotorch-train/src/history.rs` defines the three plain-data result
types the training loop pushes: `EpochResult` (one epoch's
`(epoch, train_loss, val_loss?, metrics, lr, duration_secs)` summary),
`EvalResult` (one evaluation pass's `(loss, metrics)` summary), and
`TrainingHistory` (the chronological `Vec<EpochResult>` accumulator
returned by `Learner::fit`). There is no single upstream PyTorch type
(the PyTorch trainer ecosystem splits these across
`torch.optim.Optimizer.state_dict` for serializable bookkeeping and
third-party loggers like Lightning's `TrainerLog` for per-epoch
summaries); this module canonicalises that shape into Rust structs that
`Learner`, `Callback`, and `tracing` formatters can all consume.

## Requirements

- REQ-1: `EpochResult` is a `#[non_exhaustive] #[derive(Debug, Clone)]`
  struct with exactly the fields `epoch: usize`, `train_loss: f64`,
  `val_loss: Option<f64>`, `metrics: HashMap<String, f64>`, `lr: f64`,
  `duration_secs: f64`. `non_exhaustive` lets the crate add fields
  in a minor release without breaking external struct-literal
  construction.
- REQ-2: `EpochResult` has a `Default` impl with zero/empty defaults
  and a `new_with_defaults(epoch, train_loss, val_loss, lr)` test/
  serialization-roundtrip helper. The two construction-shortcut APIs
  exist so external code (a checkpoint reloader, a JSON deserializer
  in a future serialization pass) can construct `EpochResult` without
  knowing every field.
- REQ-3: `EpochResult` implements `Display` formatting as
  `"epoch {epoch}: train_loss={...:.6}[, val_loss={...:.6}][, {name}={...:.6}]*, lr={...:.2e}, {...:.1}s"`.
  `ProgressLogger` writes this string via `tracing::info!`.
- REQ-4: `EvalResult` is a `#[non_exhaustive] #[derive(Debug, Clone)]`
  struct with fields `loss: f64`, `metrics: HashMap<String, f64>`, plus
  a `new_with_defaults(loss)` helper and `Display` formatting matching
  `Learner::evaluate`'s output contract.
- REQ-5: `TrainingHistory` is a `#[non_exhaustive] #[derive(Debug, Clone)]`
  struct wrapping `pub epochs: Vec<EpochResult>` with methods `new()`,
  `push(EpochResult)`, `len()`, `is_empty()`, `best_train_loss() ->
  Option<(usize, f64)>`, `best_val_loss() -> Option<(usize, f64)>`,
  `train_losses() -> Vec<f64>`, `val_losses() -> Vec<Option<f64>>`, and
  `Default` + `Display` impls.
- REQ-6: `best_train_loss` and `best_val_loss` use `partial_cmp` with
  an `Ordering::Equal` fallback so NaN does not panic the comparator
  (a `NaN.partial_cmp(&x) == None` is treated as Equal, leaving the
  first-seen entry as the candidate). This matches the typical
  "min over loss values" semantics PyTorch users get from
  `min(history, key=lambda e: e.val_loss)` — a NaN does not throw,
  it sorts arbitrarily.

## Acceptance Criteria

- [x] AC-1: `EpochResult`, `EvalResult`, `TrainingHistory` all carry
  `#[non_exhaustive]` and `#[derive(Debug, Clone)]`.
- [x] AC-2: `EpochResult::Default` returns the documented zero values.
- [x] AC-3: `Display` produces the documented format with conditional
  `val_loss` segment and named-metric segments.
- [x] AC-4: `TrainingHistory::best_train_loss` / `best_val_loss` return
  `None` on empty / all-`None` inputs and `Some((epoch, loss))` on
  populated inputs.
- [x] AC-5: `train_losses` / `val_losses` round-trip the per-epoch
  values in order.

## Architecture

### `EpochResult` (REQ-1, REQ-2, REQ-3)

The struct lives at `ferrotorch-train/src/history.rs:23-36`. Each field
is `pub` so external callers reading the result (a logger, a
checkpoint writer, a user notebook) can name them directly; the
`#[non_exhaustive]` attribute prevents external struct-literal
construction, forcing them through `EpochResult::new_with_defaults`
(line 55) or through `Learner::fit`'s internal literal (which is fine
because the crate that defines the type can construct it without the
exhaustiveness check). The `Default` impl at lines 38-49 zeros
everything; the `Display` impl at lines 72-83 is the canonical
serializer for `ProgressLogger::on_epoch_end` (`callback.rs:182`)
which writes the result with `tracing::info!`.

The conditional `val_loss` printing (`if let Some(vl) = self.val_loss`)
matches PyTorch trainer-log conventions: validation columns are only
printed when validation was actually run.

`new_with_defaults` and the `Default` impl exist for downstream
serialization round-trips that don't yet have a production caller —
a checkpoint reloader that reads `epoch.train_loss` from disk needs a
construction path that lets it omit `metrics` and `duration_secs`.
That consumer is the open prereq blocker #1498.

### `EvalResult` (REQ-4)

`EvalResult` at lines 97-102 is the structurally symmetric type for the
evaluation pass — `Learner::evaluate` returns it. The `Display` impl
at lines 117-125 writes `eval_loss={:.6}[, {name}={:.6}]*`. The
struct-literal construction at `learner.rs:444` is the production
consumer; `new_with_defaults` is reserved for the same reloader path
as `EpochResult::new_with_defaults`.

### `TrainingHistory` (REQ-5, REQ-6)

`TrainingHistory` at lines 140-143 is the chronological epoch buffer.
The `best_*` accessors at lines 169-184 implement the "lowest loss"
selection PyTorch users get from
`min(history, key=...)`. The `partial_cmp(...).unwrap_or(Ordering::Equal)`
pattern is the documented NaN-safety: PyTorch's `min` would raise on
NaN comparison only if the user-defined key returned a non-comparable
object; the Rust pattern here is the natural translation that does not
panic in production training loops where a stray NaN loss is a real
possibility.

`train_losses` / `val_losses` at lines 187-194 are the per-epoch
projections used by post-hoc plotting code (the conformance suite
exercises this).

### Non-test production consumers

- `ferrotorch-train/src/learner.rs:34` `use crate::history::{EpochResult,
  EvalResult, TrainingHistory};` — every `Learner::fit` /
  `Learner::evaluate` call constructs and returns these types.
- `ferrotorch-train/src/learner.rs:237` `let mut history =
  TrainingHistory::new();`, `learner.rs:352-359` constructs an
  `EpochResult` literal per epoch (the `non_exhaustive` attribute
  does not apply inside the defining crate), `learner.rs:366`
  `history.push(epoch_result)`, `learner.rs:444` returns
  `Ok(EvalResult { loss, metrics })` from `evaluate_iter`.
- `ferrotorch-train/src/callback.rs:20` `use crate::history::{EpochResult,
  TrainingHistory};` — `Callback::on_epoch_end` takes `&EpochResult`,
  `Callback::on_train_end` takes `&TrainingHistory`.
- `ferrotorch-train/src/callback.rs:201-218` reads
  `history.best_train_loss()` and `history.best_val_loss()` inside
  `ProgressLogger::on_train_end`.
- `ferrotorch-train/src/tensorboard.rs:39` `use crate::history::EpochResult;`
  — `TensorBoardCallback::on_epoch_end` reads `result.train_loss`,
  `result.val_loss`, `result.lr`, and iterates `result.metrics` to
  write TFEvents scalars.

## Parity contract

`parity_ops = []`. These are plain-data structs with no numerical
operations. Edge cases:

- **NaN loss**: `EpochResult::Default::train_loss = 0.0`, but a real
  fit can push `NaN` if backprop diverges. The `Display` `{:.6}`
  format renders `NaN` as the literal `"NaN"`; `best_train_loss` uses
  `partial_cmp(...).unwrap_or(Equal)` so a NaN entry sorts equal to
  the first-seen entry and does not panic.
- **Empty history**: `best_train_loss` / `best_val_loss` both return
  `None`; `train_losses` / `val_losses` return empty `Vec`s.
- **`Display` on a history with mixed `Option<val_loss>`**: each
  `EpochResult`'s `Display` skips the `val_loss` segment when `None`;
  the `TrainingHistory` `Display` writes one line per epoch.

## Verification

Unit tests in `mod tests` cover all REQs:

- `test_history_new_is_empty` / `test_history_push_and_len` (lines
  232-245) pin the `new`/`push`/`len`/`is_empty` contract.
- `test_best_train_loss` / `test_best_val_loss` /
  `test_best_val_loss_none_when_no_val` / `test_best_train_loss_empty`
  (lines 247-280) pin the `best_*` semantics including `None` on
  empty / all-`None`.
- `test_train_losses` / `test_val_losses` (lines 282-298) pin the
  ordered-projection accessors.
- `test_epoch_result_display` / `test_epoch_result_display_no_val` /
  `test_eval_result_display` (lines 300-324) pin the `Display`
  formatting contracts.

Smoke command:

```bash
cargo test -p ferrotorch-train --lib history:: 2>&1 | tail -3
```

Expected: > 0 passed, 0 failed.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct EpochResult` with `#[non_exhaustive] #[derive(Debug, Clone)]` and the 6 documented fields at `ferrotorch-train/src/history.rs:23-36`; non-test consumer: `ferrotorch-train/src/learner.rs:352-359` constructs an `EpochResult` literal per epoch in `Learner::fit`, then `learner.rs:366` pushes it into `TrainingHistory`. |
| REQ-2 | NOT-STARTED | open prereq blocker #1498 — `EpochResult::Default` impl (line 38) and `new_with_defaults` (line 55) plus `EvalResult::new_with_defaults` (line 109) exist on the public surface but no production caller invokes them. Production code in `Learner::fit` constructs `EpochResult` via struct literal (line 352) and `EvalResult` via struct literal (line 444), bypassing both helpers. The consumer-wiring (a checkpoint reloader / JSON deserializer that calls `new_with_defaults`) is the open work. |
| REQ-3 | SHIPPED | impl: `Display` impl at `ferrotorch-train/src/history.rs:72-83`; non-test consumer: `ferrotorch-train/src/callback.rs:182` `tracing::info!(..., "{result}")` writes the formatted `EpochResult` from `ProgressLogger::on_epoch_end`. |
| REQ-4 | SHIPPED | impl: `pub struct EvalResult` at `ferrotorch-train/src/history.rs:97-102` and `Display` at lines 117-125; non-test consumer: `ferrotorch-train/src/learner.rs:444` returns `Ok(EvalResult { loss, metrics })` from `evaluate_iter` (called by `evaluate` at line 403 and `fit` at line 323). The `new_with_defaults` helper (line 109) is the unwired bit covered by blocker #1498. |
| REQ-5 | SHIPPED | impl: `pub struct TrainingHistory` at `ferrotorch-train/src/history.rs:140-143` with `new`/`push`/`len`/`is_empty`/`best_train_loss`/`best_val_loss`/`train_losses`/`val_losses` at lines 147-194, plus `Default` (line 197) and `Display` (line 203); non-test consumer: `ferrotorch-train/src/learner.rs:237` `let mut history = TrainingHistory::new();` and `history.push(epoch_result)` at line 366; `ferrotorch-train/src/callback.rs:201-218` reads `history.best_train_loss()` / `history.best_val_loss()` in `ProgressLogger::on_train_end`. |
| REQ-6 | SHIPPED | impl: `min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))` at `ferrotorch-train/src/history.rs:173` and `:183` — NaN safe via the `unwrap_or(Equal)` fallback; non-test consumer: same `ProgressLogger::on_train_end` site at `callback.rs:203` / `:211` invokes `best_*` on training-run histories that may contain non-finite values (the trainer does not pre-filter). |
