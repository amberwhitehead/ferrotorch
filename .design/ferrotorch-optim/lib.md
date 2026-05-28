# ferrotorch-optim — crate root (`lib.rs`)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/optim/__init__.py
  - torch/optim/optimizer.py
-->

## Summary

`ferrotorch-optim/src/lib.rs` is the crate-root manifest of the
optimizer family: it declares all 25 leaf modules (one per optimizer
family + scheduler tree + amp/grad-scaler / grad-accumulation /
differentiable / foreach-utils / param-key utility files) and
re-exports their public surface so downstream crates (`ferrotorch-train`,
`ferrotorch`, example/benchmark binaries) consume the optimizer family
through `ferrotorch_optim::*` without spelunking into submodule
paths. Mirrors `torch/optim/__init__.py`'s
"flat re-export of every optimizer class" pattern.

## Requirements

- REQ-1: Declare every leaf submodule in the crate as `pub mod` so
  every optimizer file is reachable both as `ferrotorch_optim::adam`
  (the per-file namespace) and through the crate-root re-exports.
- REQ-2: Re-export every optimizer config + struct pair at the crate
  root (`Adam`/`AdamConfig`, `AdamW`/`AdamWConfig`, `Sgd`/`SgdConfig`,
  ...) — mirrors `torch.optim.Adam`, `torch.optim.AdamW`, ... — so
  downstream user code writes `use ferrotorch_optim::Adam` rather
  than `use ferrotorch_optim::adam::Adam`.
- REQ-3: Re-export the cross-cutting traits + types — `Optimizer`,
  `OptimizerState`, `ParamGroup` — so downstream consumers can hold
  `Box<dyn Optimizer<T>>` without naming the `optimizer` submodule.
  Mirrors `torch.optim.Optimizer`.
- REQ-4: Re-export the LR-scheduler surface (`LrScheduler`,
  `StepLR`, `CosineAnnealingLR`, ...) at the crate root so the
  `learner.rs` consumer can plug a scheduler in without naming
  `ferrotorch_optim::scheduler`.
- REQ-5: Re-export the AMP / grad-scaling helpers (`GradScaler`,
  `GradScalerConfig`, `GradScalerState`) so `ferrotorch-train`'s
  `AmpContext` re-exports them in turn.
- REQ-6: Re-export the gradient-accumulation helper
  (`GradientAccumulator`) and the differentiable-step helpers
  (`diff_sgd_step`, `diff_sgd_momentum_step`) so meta-learning code
  + large-batch training loops do not need to know their submodule
  paths.
- REQ-7: Re-export the param-key newtype (`ParamKey`) for use by any
  custom optimizer that wants the same per-parameter HashMap key
  layout the in-tree optimizers use.

## Acceptance Criteria

- [x] AC-1: `pub mod` declarations cover all 25 submodules
  (adadelta, adafactor, adagrad, adam, adamax, adamw, asgd,
  differentiable, ema, foreach_utils, grad_accumulator, grad_scaler,
  lbfgs, muon, nadam, natural_gradient, optimizer, param_key, radam,
  rmsprop, rprop, scheduler, sgd, sparse_adam, swa).
- [x] AC-2: `pub use` re-exports every public optimizer type pair.
- [x] AC-3: `pub use optimizer::{Optimizer, OptimizerState, ParamGroup}`.
- [x] AC-4: `pub use scheduler::{...}` covers `LrScheduler`, every
  scheduler struct, and the `cosine_warmup_scheduler` factory.
- [x] AC-5: `pub use grad_scaler::{GradScaler, GradScalerConfig, GradScalerState}`.
- [x] AC-6: `pub use grad_accumulator::GradientAccumulator` and
  `pub use differentiable::{diff_sgd_step, diff_sgd_momentum_step}`.
- [x] AC-7: `pub use param_key::ParamKey`.

## Architecture

`lib.rs` is intentionally flat — no logic, only `pub mod` /
`pub use` lines. Each `pub mod` line is the entry point for the
translate-discipline hook (every routed `.rs` file is reachable via
exactly one `pub mod` here); each `pub use` line is the crate-root
public API the downstream crates consume.

### Module declarations (REQ-1)

Lines 1-25 of `lib.rs` declare every leaf submodule with `pub mod`.
The `scheduler` submodule is itself a directory module
(`ferrotorch-optim/src/scheduler/mod.rs` + the per-scheduler files);
all other submodules are single files.

### Optimizer family re-exports (REQ-2)

Lines 27-55 re-export each optimizer's two public structs (one
config, one optimizer). This matches PyTorch's
`torch.optim.__init__.py` which does
`from .adam import Adam`, `from .adamw import AdamW`, etc. The
configs (`AdamConfig`, `SgdConfig`, ...) are a Rust-idiom addition
(R-DEV-7) over PyTorch's flat-kwarg constructor — Rust users get
builder-style hyperparameter setup; the underlying numerical
contract is identical.

### Trait + utility re-exports (REQ-3..7)

- `pub use optimizer::{Optimizer, OptimizerState, ParamGroup}` — the
  trait every consumer holds dynamically.
- `pub use scheduler::{LrScheduler, ChainedScheduler, ...}` — flat
  re-export of every scheduler.
- `pub use grad_scaler::{GradScaler, GradScalerConfig, GradScalerState}`
  — AMP integration surface.
- `pub use grad_accumulator::GradientAccumulator` —
  large-effective-batch training helper.
- `pub use differentiable::{diff_sgd_step, diff_sgd_momentum_step}`
  — meta-learning inner-loop step helpers (MAML-style).
- `pub use param_key::ParamKey` — the typed per-parameter HashMap
  key used by every state-keeping optimizer in the crate.

### Non-test production consumers

- `ferrotorch-train/src/learner.rs` `use ferrotorch_optim::Optimizer; use ferrotorch_optim::grad_scaler::GradScaler; use ferrotorch_optim::scheduler::LrScheduler;` — the central training loop holds these as fields.
- `ferrotorch-train/src/amp.rs` `pub use ferrotorch_optim::{GradScaler, GradScalerConfig, GradScalerState}; use ferrotorch_optim::Optimizer;` — the AMP context re-exports the GradScaler family.
- `ferrotorch-train/examples/multi_epoch_train_dump.rs:63` `use ferrotorch_optim::{Adam, AdamConfig, Optimizer};` — example consumer.
- `benchmarks/ferrotorch_bench.rs` — benchmark consumer.

## Parity contract

`parity_ops = []`. `lib.rs` is a pure re-export manifest. Numerical
parity is owned by each consumed submodule's design doc.

## Verification

`lib.rs` itself ships no test code. The compile gate is its
verification: every routed `.rs` file in the crate must be reachable
through `lib.rs`, and the absence of any of the `pub use`
re-exports would break `ferrotorch-train`'s build.

Smoke command:

```bash
cargo check -p ferrotorch-optim 2>&1 | tail -3
cargo test -p ferrotorch-optim --lib 2>&1 | tail -3
```

Expected: `cargo check` clean; `327 passed; 0 failed` for the lib
tests across all 25 submodules.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: 25 `pub mod` lines at `ferrotorch-optim/src/lib.rs` mirroring `torch/optim/__init__.py:1-30` flat submodule layout; non-test consumer: the crate compiles — every leaf module is reachable from the crate root, exercised by `ferrotorch-train/src/learner.rs` `use ferrotorch_optim::Optimizer;`. |
| REQ-2 | SHIPPED | impl: `pub use adam::{Adam, AdamConfig}` etc. at `ferrotorch-optim/src/lib.rs:27-55` mirroring `torch/optim/__init__.py`'s flat re-exports; non-test consumer: `ferrotorch-train/examples/multi_epoch_train_dump.rs:63` `use ferrotorch_optim::{Adam, AdamConfig, Optimizer};` consumes the `Adam` / `AdamConfig` re-exports. |
| REQ-3 | SHIPPED | impl: `pub use optimizer::{Optimizer, OptimizerState, ParamGroup}` at `optimizer in ferrotorch-optim/src/lib.rs`; non-test consumer: `ferrotorch-train/src/learner.rs` `use ferrotorch_optim::Optimizer;` plus line 59 `optimizer: Box<dyn Optimizer<T>>` holds the trait dynamically. |
| REQ-4 | SHIPPED | impl: `pub use scheduler::{...}` block at `ferrotorch-optim/src/lib.rs`; non-test consumer: `ferrotorch-train/src/learner.rs` `use ferrotorch_optim::scheduler::LrScheduler;` — the learner consumes the scheduler surface to drive LR updates. |
| REQ-5 | SHIPPED | impl: `pub use grad_scaler::{GradScaler, GradScalerConfig, GradScalerState}` at `ferrotorch-optim/src/lib.rs`; non-test consumer: `AmpContext in ferrotorch-train/src/amp.rs` `pub use ferrotorch_optim::{GradScaler, GradScalerConfig, GradScalerState};` re-exports them through the `AmpContext` surface. |
| REQ-6 | SHIPPED | impl: `pub use grad_accumulator::GradientAccumulator` at `ferrotorch-optim/src/lib.rs` and `pub use differentiable::{diff_sgd_momentum_step, diff_sgd_step}` at line 34; non-test consumer: the public surface is consumed via the `pub use` chain — boundary-method per goal.md S5 ("Boundary methods ARE the public API; they don't need further downstream callers to be SHIPPED"). |
| REQ-7 | SHIPPED | impl: `pub use param_key::ParamKey` at `asgd in ferrotorch-optim/src/lib.rs`; non-test consumer: in-crate consumers `ferrotorch-optim/src/radam.rs`, `asgd.rs`, `adamax in adamax.rs`, `adamw.rs`, `sparse_adam in sparse_adam.rs` all `use crate::param_key::ParamKey;` (re-exporting via lib makes the same type available to external custom optimizers). |
