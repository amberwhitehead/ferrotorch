# ferrotorch-train — crate root

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/utils/data/dataloader.py
  - torch/optim/optimizer.py
  - torch/amp/__init__.py
-->

## Summary

`ferrotorch-train/src/lib.rs` is the crate root for the training utilities
that compose `ferrotorch-core`, `ferrotorch-nn`, and `ferrotorch-optim`
into the high-level training loop a user actually drives. It declares the
8 submodules (`amp`, `callback`, `checkpoint`, `grad_utils`, `history`,
`learner`, `metric`, `tensorboard`) and re-exports the canonical public
surface (`Learner`, `LossFn`, `Callback`, `Metric`, `TrainingHistory`,
`TensorBoardWriter`, `clip_grad_norm_`, `clip_grad_value_`, etc.). It
mirrors the role of upstream `torch/__init__.py` together with
PyTorch-Lightning-style trainer composition: there is no single upstream
file because the trainer level in PyTorch is split across
`torch.optim.Optimizer`, `torch.utils.data.DataLoader`, `torch.amp`, and
the third-party Lightning / accelerate / fastai trainers users typically
import.

## Requirements

- REQ-1: Crate must expose exactly the modules `amp`, `callback`,
  `checkpoint`, `grad_utils`, `history`, `learner`, `metric`,
  `tensorboard` via `pub mod`. Each submodule must compile as a separate
  translation unit so downstream crates can `use ferrotorch_train::amp::*`
  without dragging the entire trainer in.
- REQ-2: Crate must re-export the user-facing API names at the crate
  root: `Callback`, `EarlyStopping`, `EmaCallback`, `ProgressLogger`,
  `checkpoint`, `checkpoint_sequential`, `clip_grad_norm_`,
  `clip_grad_value_`, `EpochResult`, `EvalResult`, `TrainingHistory`,
  `Learner`, `LossFn`, `AccuracyMetric`, `LossMetric`, `Metric`,
  `RunningAverage`, `TopKAccuracy`, `TensorBoardCallback`,
  `TensorBoardWriter`. This matches the `from torch.X import Y` flat
  import style PyTorch users expect.
- REQ-3: Lint baseline must mirror the workspace-standard pattern from
  sibling leaf crates (`ferrotorch-core`/`-nn`/`-distributed`/`-jit`/
  `-cubecl`/`-xpu`): `#![warn(clippy::all, clippy::pedantic)]` +
  `#![deny(rust_2018_idioms)]`, with documented `#![allow]`s for the
  pedantic lints that are noise rather than signal (one-line rationale
  per allow). `missing_docs` and `missing_debug_implementations` stay at
  `allow` to match precedent because public types hold trait objects
  (`Box<dyn Optimizer<T>>`, `Box<dyn LrScheduler<T>>`, `Box<dyn
  Callback<T>>`) whose `Debug` impls require careful hand-rolling.
- REQ-4: Doctests at the crate root must be runnable without a model or
  dataset (i.e. must not depend on GPU, network, or large in-memory
  fixtures). The `Learner::fit` example is `ignore`d because it requires
  a model + dataset; everything in the crate-root doctest must be
  executable as written.

## Acceptance Criteria

- [x] AC-1: `pub mod amp; pub mod callback; pub mod checkpoint; pub mod
  grad_utils; pub mod history; pub mod learner; pub mod metric; pub mod
  tensorboard;` declared in `lib.rs`.
- [x] AC-2: Re-exports in REQ-2 all resolve via the crate-root `use`
  ladder (`pub use callback::{Callback, EarlyStopping, EmaCallback,
  ProgressLogger};` etc.).
- [x] AC-3: `#![warn(clippy::all, clippy::pedantic)]` + workspace-aligned
  per-lint `#![allow]` list with rationale comments.
- [x] AC-4: `cargo test -p ferrotorch-train --doc` passes with the
  crate-root doctest constructing `LossMetric`, `EarlyStopping`,
  `ProgressLogger`, `EmaCallback`, `TrainingHistory` without panic.

## Architecture

### Module wiring (REQ-1)

The 8 `pub mod` declarations split the trainer surface along clean
seams. `learner.rs` owns the loop; `callback.rs` and `metric.rs` own the
extension hooks; `history.rs` owns the result types pushed by the loop;
`amp.rs` glues `ferrotorch_core::autograd::autocast` and
`ferrotorch_optim::GradScaler` into a single `AmpContext` object;
`checkpoint.rs` is a thin shell over `ferrotorch_core::autograd::
checkpoint::checkpoint` plus a sequential-segments helper;
`grad_utils.rs` re-exports `ferrotorch_nn::utils::clip_grad_*` so the
deduplication from `#1104` is preserved at the crate boundary; and
`tensorboard.rs` is a self-contained TFEvents writer + callback. The
boundary mirrors how PyTorch users typically reach for `torch.optim`,
`torch.utils.checkpoint`, `torch.utils.tensorboard`, and `torch.amp`
piecewise.

### Public surface (REQ-2)

The `pub use` ladder at the bottom of `lib.rs` (lines 177-183) is the
user contract. Any rename of `EarlyStopping` to `EarlyStoppingCallback`
or `Learner` to `Trainer` would be a breaking change to every
downstream notebook/example/README; the names match the in-tree
fixtures (`ferrotorch-train/examples/multi_epoch_train_dump.rs`) and
the public README quick-start blocks.

### Lint baseline (REQ-3)

The `#![allow]` block at lines 77-166 mirrors sibling leaf crates
verbatim. Each allow carries a one-line justification anchored in the
trainer-loop code patterns (`cast_*` lints for `usize -> f64` step/lr
arithmetic, `missing_errors_doc` because the focused doc-pass is
tracked separately, `too_many_lines` because `Learner::fit` is one
function mirroring `Trainer.fit` upstream, etc.). Diverging on the
allow list from a leaf crate would be Step 4 architectural
unilateralism — the precedent in the other ferrotorch leaf crates
governs.

### Crate-root doctest (REQ-4)

The fenced ```rust``` block at lines 26-51 of `lib.rs` constructs
`LossMetric`, `EarlyStopping`, `ProgressLogger`, `EmaCallback`, and
`TrainingHistory`. None of these require a model, dataset, GPU, or
network, so the doctest passes in any environment that compiles the
crate. The `Learner::fit` full example is `ignore`d in `learner.rs`
(line 10) because constructing a real model + dataset closure pair
would balloon the doctest budget.

### Non-test production consumers

- `ferrotorch-train/examples/multi_epoch_train_dump.rs` — the
  real-artifact training-trajectory dump binary uses the
  `Learner`/`LossFn` surface end-to-end on a 3-layer MLP + Adam.
- `ferrotorch-train/src/learner.rs` line 33-35
  (`use crate::callback::Callback; use crate::history::{EpochResult,
  EvalResult, TrainingHistory}; use crate::metric::Metric;`) consumes
  the submodule items directly through the lib.rs `pub mod`
  declarations.
- `ferrotorch-train/src/callback.rs` line 20
  (`use crate::history::{EpochResult, TrainingHistory};`) and
  `tensorboard.rs` lines 38-39
  (`use crate::callback::Callback; use crate::history::EpochResult;`)
  exercise the cross-submodule wiring `lib.rs` exposes.

The crate is not yet a dependency of any other in-tree ferrotorch crate
(no `Cargo.toml` lists `ferrotorch-train` as a dependency), so the
"consumer" is exclusively the example binary, the in-tree tests under
`ferrotorch-train/tests/`, and the sibling submodule wiring. The
downstream-crate consumer (an end-user model crate) is the open
prerequisite that gates further user-facing API ratchets.

## Parity contract

`parity_ops = []`. The crate root declares no numeric operations; the
contract here is structural (module set + re-export surface).
Edge cases the lib root itself owns:

- **Re-export drift**: a `pub use crate::foo::Bar` that points to a
  missing symbol fails the compile. The doctest in REQ-4 + the
  conformance surface coverage test
  (`ferrotorch-train/tests/conformance_surface_coverage.rs`) pin every
  re-export name to a concrete type.
- **Submodule visibility**: each `pub mod` is `pub`, not `pub(crate)`,
  so external code can name `ferrotorch_train::amp::AmpContext`
  directly. Downgrading to `pub(crate)` would break the
  `ferrotorch_train::amp::AmpContext` import in
  `examples/multi_epoch_train_dump.rs` (and in any downstream user
  code).

## Verification

The crate-root doctest is exercised by `cargo test -p ferrotorch-train
--doc`. The submodule presence + re-export integrity is exercised by
`ferrotorch-train/tests/conformance_surface_coverage.rs`. Submodule
unit tests run via `cargo test -p ferrotorch-train --lib`.

Smoke command:

```bash
cargo test -p ferrotorch-train --lib 2>&1 | tail -3
```

Expected: a passing test count > 0 with `0 failed`. The route's
`parity_ops` list is empty, so there is no `parity-sweep` invocation
at this level.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: 8 `pub mod` declarations at `ferrotorch-train/src/lib.rs:168-175`; non-test consumer: `ferrotorch-train/src/learner.rs:33-35` (`use crate::callback::Callback;` etc.), `ferrotorch-train/src/tensorboard.rs:38-39`, and `ferrotorch-train/src/callback.rs:20` — cross-submodule wiring exercises every module declaration. |
| REQ-2 | SHIPPED | impl: `pub use` ladder at `ferrotorch-train/src/lib.rs:177-183` covering `Callback`/`EarlyStopping`/`EmaCallback`/`ProgressLogger`/`checkpoint`/`checkpoint_sequential`/`clip_grad_norm_`/`clip_grad_value_`/`EpochResult`/`EvalResult`/`TrainingHistory`/`Learner`/`LossFn`/`AccuracyMetric`/`LossMetric`/`Metric`/`RunningAverage`/`TopKAccuracy`/`TensorBoardCallback`/`TensorBoardWriter`; non-test consumer: `ferrotorch-train/examples/multi_epoch_train_dump.rs` (the multi-epoch trajectory dump binary) uses `Learner`/`LossFn` end-to-end, and the crate-root doctest at `lib.rs:26-51` constructs `LossMetric`/`EarlyStopping`/`ProgressLogger`/`EmaCallback`/`TrainingHistory` through the re-exports. |
| REQ-3 | SHIPPED | impl: `#![warn(clippy::all, clippy::pedantic)]` + `#![deny(rust_2018_idioms)]` at `ferrotorch-train/src/lib.rs:62-63`, followed by per-lint `#![allow]` block at `lib.rs:77-166` with one-line rationale per allow; non-test consumer: every submodule under `ferrotorch-train/src/` inherits the baseline via the `mod` declarations — clippy clean at `-D warnings` is the integration test. |
| REQ-4 | SHIPPED | impl: fenced doctest at `ferrotorch-train/src/lib.rs:26-51` constructs `LossMetric`/`EarlyStopping`/`ProgressLogger`/`EmaCallback`/`TrainingHistory` with no model/dataset dependency; non-test consumer: the doctest is the consumer — it compiles and runs against the public surface as the user would, and is wired into `cargo test -p ferrotorch-train --doc`. |
