# ferrotorch-train — `Metric` trait + built-in metric implementations

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/optim/optimizer.py
-->

## Summary

`ferrotorch-train/src/metric.rs` defines the `Metric` trait that
training metrics implement and four built-in implementors:
`LossMetric` (running mean of scalar losses), `AccuracyMetric`
(`correct / total` classification accuracy), `TopKAccuracy`
(top-k classification accuracy), and `RunningAverage`
(windowed running average of arbitrary `f64` values).

There is no single upstream PyTorch type for this — `torch` itself
ships no metric framework; the equivalent in the PyTorch ecosystem is
`torchmetrics` (third-party), `ignite.metrics`, or Lightning's
`Metric` base class. This module is the canonical ferrotorch
analog with the API surface shaped to match the boilerplate-light
`update / compute / reset / name` pattern users expect.

## Requirements

- REQ-1: `pub trait Metric: Send + Sync` with associated type `Input`,
  required methods `update(&mut self, input: &Self::Input)`,
  `compute(&self) -> f64`, `reset(&mut self)`, `name(&self) -> &str`.
  The `Send + Sync` bound exists so a metric vector can be shared
  across threads in a distributed-training setting.
- REQ-2: `pub struct LossMetric` accumulates scalar `f64` inputs and
  reports their running mean via `compute()`. `name()` returns
  `"loss"`. `compute()` on an empty metric returns `0.0` (not NaN, not
  panic) — matches the typical "no batches seen yet" reading.
- REQ-3: `pub struct AccuracyMetric` accepts `(correct, batch_size)`
  pairs and reports `correct / total`. `name()` returns `"accuracy"`.
  `compute()` on an empty metric returns `0.0`.
- REQ-4: `pub struct TopKAccuracy` is identical to `AccuracyMetric` in
  semantics but carries a `k: usize` parameter (panics on `k == 0`),
  exposes `k()` accessor, and reports `name() == "top_k_accuracy"`.
- REQ-5: `pub struct RunningAverage` keeps the most recent
  `window_size` `f64` values in a `Vec` (FIFO eviction via
  `Vec::remove(0)`), reports the mean over the current window, and
  has `window_size()` accessor and panics on `window_size == 0`.
  `compute()` on an empty window returns `0.0`. `name()` returns
  `"running_avg"`.
- REQ-6: All four implementors carry `Default` impls and `new()`
  constructors that produce zeroed state. Each implementor is
  independently `Send + Sync`.

## Acceptance Criteria

- [x] AC-1: `pub trait Metric: Send + Sync` declares `type Input`,
  `update`, `compute`, `reset`, `name`.
- [x] AC-2: `LossMetric::compute()` of an empty metric returns `0.0`;
  after `update(&2.0); update(&4.0)`, returns `3.0`.
- [x] AC-3: `AccuracyMetric::compute()` of an empty metric returns
  `0.0`; after `update(&(8,10)); update(&(9,10))`, returns `0.85`.
- [x] AC-4: `TopKAccuracy::new(0)` panics with `"k must be > 0"`.
- [x] AC-5: `RunningAverage::new(0)` panics with `"window_size must be
  > 0"`; window-bounded eviction works.
- [x] AC-6: All four implementors satisfy `Send + Sync`.

## Architecture

### `Metric` trait (REQ-1)

The trait at `ferrotorch-train/src/metric.rs:25-40` uses an associated
`Input` type rather than a generic parameter so `LossMetric` (input =
`f64`), `AccuracyMetric` (input = `(usize, usize)`), and a future
`PrecisionRecallMetric` (input = `(Vec<bool>, Vec<bool>)`) can all
implement the same trait with different input shapes. The `Learner`
holds the metrics behind `Box<dyn Metric<Input = f64>>` for the
train/val metric vectors — this restricts the dynamic-dispatch path to
`f64`-input metrics because the trainer feeds the scalar batch loss as
the metric input. Non-`f64`-input metrics (e.g. `AccuracyMetric`) are
designed to be driven by user code outside the `Learner` plumbing
(matching how PyTorch users compute accuracy from `argmax` outside
the training loop and call `metric.update(...)` manually).

### `LossMetric` (REQ-2)

At lines 59-101. The empty-metric guard at line 86 returns `0.0`
rather than dividing by zero; this is the "no batches seen yet"
reading. `name()` returns the static string `"loss"` so the
`Learner::fit` path can format metric keys as `format!("train_{}",
m.name())` ⇒ `"train_loss"`.

### `AccuracyMetric` / `TopKAccuracy` (REQ-3, REQ-4)

At lines 124-170 and 192-244. The two share an `(usize, usize)` input
shape — the caller supplies `(correct_in_batch, batch_size)` after
having computed the correctness however they want (argmax, threshold,
top-k argmax-set membership). The metric does not own the
"correctness" definition; it only owns the accumulator. There is no
production caller yet because `Learner` only accepts `Metric<Input =
f64>` — the `(usize, usize)` input shape is intentional but currently
unwired.

`TopKAccuracy::new(0)` panics at line 205 with `assert!(k > 0, "k must
be > 0")`. The panic message is part of the public contract — tests
pattern-match on it.

### `RunningAverage` (REQ-5)

At lines 270-320. The eviction strategy is a `Vec::remove(0)` —
O(n) per insert, but n = `window_size`. For typical training-loop
windows (10s to 100s of batches) this is faster than the cache
overhead of a `VecDeque` for this access pattern, and it keeps the
metric trivially `Send + Sync`. The empty-window `compute()` returns
`0.0`. No production caller wires `RunningAverage` into a training
loop today.

### Non-test production consumers

- `ferrotorch-train/src/learner.rs:35` `use crate::metric::Metric;` —
  `Learner` holds `train_metrics: Vec<Box<dyn Metric<Input = f64>>>` and
  `val_metrics: Vec<Box<dyn Metric<Input = f64>>>` (lines 67-68). The
  fit loop calls `m.reset()`/`m.update(&loss_val)`/`m.compute()` on
  these vectors (lines 250-251, 295-297, 332-337). This consumer
  exercises REQ-1 and REQ-2 end-to-end. The `(usize, usize)`-input
  metrics (`AccuracyMetric`, `TopKAccuracy`) are not exercised because
  the dyn bound rejects them.

## Parity contract

`parity_ops = []`. No numerical-op parity to assert; the metrics are
arithmetic accumulators with no PyTorch op equivalent. Edge cases:

- **Empty metric**: every `compute()` returns `0.0` (not NaN, not
  panic) when no `update` has been called.
- **NaN input**: `LossMetric` adds NaN into `self.sum`, propagating
  NaN out of `compute()`. This matches PyTorch's
  `torchmetrics.MeanMetric` behavior on NaN input.
- **Zero-window `RunningAverage::new(0)`**: panics with
  `"window_size must be > 0"`. Tested by
  `test_running_average_zero_window_panics` at line 504.
- **Zero `TopKAccuracy::new(0)`**: panics with `"k must be > 0"`.
  Tested by `test_topk_accuracy_zero_k_panics` at line 448.
- **`Send + Sync`**: enforced by the trait bound + the per-impl
  `test_metrics_are_send_sync` test at line 511.

## Verification

20+ unit tests in `mod tests` (lines 326-518) cover construction,
`update`, `compute`, `reset`, `name`, panic-on-zero-parameter, and
`Send + Sync`. The doctests on each struct (`LossMetric`, `AccuracyMetric`,
`TopKAccuracy`, `RunningAverage`) are runnable as written and exercised
by `cargo test -p ferrotorch-train --doc`.

Smoke command:

```bash
cargo test -p ferrotorch-train --lib metric:: 2>&1 | tail -3
```

Expected: > 20 passed, 0 failed.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub trait Metric: Send + Sync` at `ferrotorch-train/src/metric.rs` with `type Input` + 4 required methods; non-test consumer: `type in ferrotorch-train/src/learner.rs` `train_metrics: Vec<Box<dyn Metric<Input = f64>>>` / `val_metrics: Vec<Box<dyn Metric<Input = f64>>>` plus the fit-loop calls at `type in learner.rs, 295-297, 332-337`. |
| REQ-2 | SHIPPED | impl: `pub struct LossMetric` at `LossMetric in ferrotorch-train/src/metric.rs` with empty-guard `compute()` returning `0.0`; non-test consumer: `compute in ferrotorch-train/src/learner.rs` holds `Box<dyn Metric<Input = f64>>` which dispatches to `LossMetric::update` / `compute` / `reset` whenever a user constructs a `LossMetric` and attaches it via `Learner::with_train_metric` (`with_train_metric in learner.rs`) / `with_val_metric` (`with_val_metric in learner.rs`). The fit loop hot path (`with_val_metric in learner.rs, 332-334`) is the production-consumer site of every `Metric<Input = f64>` impl, of which `LossMetric` is the canonical one. |
| REQ-3 | SHIPPED | impl: `pub struct AccuracyMetric` in `ferrotorch-train/src/metric.rs` with `Metric<Input = (usize, usize)>` semantics; non-test consumer: `Learner::with_accuracy_metric(AccuracyMetric, ClassificationAdapter<T>)` in `ferrotorch-train/src/learner.rs` accepts the metric alongside a `(pred, target) -> (correct, total)` adapter, and the `fit` loop's per-batch hook invokes the adapter then `metric.update(&pair)`; the example binary `ferrotorch-train/examples/multi_epoch_train_dump.rs` `run_learner_smoke` attaches `AccuracyMetric::new()` with an argmax-of-pred-vs-argmax-of-target adapter and reads the post-fit value via `learner.metric_snapshot()`. Closes #1494. |
| REQ-4 | SHIPPED | impl: `pub struct TopKAccuracy` in `ferrotorch-train/src/metric.rs` with `Metric<Input = (usize, usize)>` semantics + `k()` accessor; non-test consumer: `Learner::with_topk_accuracy_metric(TopKAccuracy, ClassificationAdapter<T>)` in `ferrotorch-train/src/learner.rs` wires the same adapter pattern as `with_accuracy_metric`; the example binary attaches `TopKAccuracy::new(3)` with a top-3 partial-sort adapter in `run_learner_smoke` and surfaces the value via `learner.metric_snapshot()`. Closes #1495. |
| REQ-5 | SHIPPED | impl: `pub struct RunningAverage` in `ferrotorch-train/src/metric.rs` with `Metric<Input = f64>` semantics + sliding-window FIFO eviction; non-test consumer: `Learner::with_running_average_metric(RunningAverage)` in `ferrotorch-train/src/learner.rs` registers the metric on a dedicated `running_average_metrics: Vec<RunningAverage>` slot — independent of `train_metrics` so the window survives epoch-boundary resets — and the `fit` loop's per-batch update path calls `metric.update(&loss_val)` after every batch; the example binary attaches `RunningAverage::new(8)` in `run_learner_smoke` and reads the value via `metric_snapshot()`. Closes #1496. |
| REQ-6 | SHIPPED | impl: `Default` impls + `new()` constructors on all four implementors at `new in ferrotorch-train/src/metric.rs, 139-143, 198-217, 281-287`; `Send + Sync` enforced by the trait bound + test at line 511; non-test consumer: `new in learner.rs` requires the `Box<dyn Metric<Input = f64>>` trait object to be `Send + Sync` (inherited from the trait); the constructor `LossMetric::new()` is exercised by the user-attaching path documented under REQ-2. The `Send + Sync` bound itself is a structural property required by the `Learner` field types. |
