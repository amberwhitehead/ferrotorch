# `Rmsprop` — RMSprop optimizer with momentum and centered variant

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/optim/rmsprop.py
-->

## Summary

`ferrotorch-optim/src/rmsprop.rs` defines `Rmsprop<T>` and `RmspropConfig`,
mirroring `torch.optim.RMSprop` (`torch/optim/rmsprop.py:30`). Supports
optional momentum, centered variance normalization, L2 weight decay,
`maximize`, and an on-device "foreach" tensor-op step path.

## Requirements

- REQ-1: `pub struct RmspropConfig` with `lr` (0.01), `alpha` (0.99),
  `eps` (1e-8), `weight_decay` (0.0), `momentum` (0.0), `centered` (false),
  `maximize` (false), `foreach` (false). Mirrors upstream's `__init__`
  kwargs at `torch/optim/rmsprop.py:31-44`.
- REQ-2: `pub struct Rmsprop<T: Float>` with `new(params, config)` and
  `new_with_groups(groups, config)` constructors; implements
  `Optimizer<T>`.
- REQ-3: Per-parameter state `ParamState { square_avg, grad_avg,
  momentum_buf }` (the latter two `Option<Vec<T>>`, materialised only
  when the corresponding feature is enabled). Mirrors upstream's
  `state["square_avg"] / state["grad_avg"] / state["momentum_buffer"]`
  (`torch/optim/rmsprop.py:122-130`).
- REQ-4: Step update — `square_avg = alpha * square_avg + (1 - alpha) * g^2`;
  centered variant additionally tracks `grad_avg = alpha * grad_avg +
  (1 - alpha) * g`, then `avg = sqrt(square_avg - grad_avg^2) + eps`;
  non-centered: `avg = sqrt(square_avg) + eps`. Mirrors
  `_single_tensor_rmsprop` (`torch/optim/rmsprop.py:265-340`).
- REQ-5: **eps OUTSIDE the sqrt** (`sqrt(square_avg) + eps`, NOT
  `sqrt(square_avg + eps)`). This is upstream's contract — see the inline
  comment at `rmsprop.rs` citing the #1155 bug-fix and upstream's
  `avg.sqrt_(); avg.add_(eps)` two-step.
- REQ-6: Optional momentum: `momentum > 0` ⇒
  `buf = momentum * buf + g / avg; param -= lr * buf`,
  else `param -= lr * g / avg`. Mirrors `torch/optim/rmsprop.py:289-318`.
- REQ-7: L2 weight decay: `grad = grad + weight_decay * param`
  pre-square-avg. Mirrors `torch/optim/rmsprop.py:281-288`.
- REQ-8: `maximize: true` negates the gradient before applying the rest of
  the update (`g *= -1`). Mirrors upstream's `maximize=True` kwarg
  semantics.
- REQ-9: `foreach: true` switches to a fully on-device tensor-op step
  (`step_foreach`). Buffers live as `Tensor<T>` instead of `Vec<T>`,
  matching the same algorithm but composed via `add`/`mul`/`sqrt`/`sub`
  primitives. CL-497.
- REQ-10: `state_dict` / `load_state_dict` round-trip the legacy
  CPU-vec state under `"group{gi}_param{pi}"` keys (NOT the
  CL-1122 `ParamKey` scheme — Rmsprop predates the typed-key rollout).
  Round-trip preserves `square_avg`/`grad_avg`/`momentum_buf` exactly.

## Acceptance Criteria

- [x] AC-1: `RmspropConfig::default()` returns the documented defaults.
- [x] AC-2: Quadratic-loss convergence on `f(x) = ||x||^2` from
  `[3, -4, 5]` reaches `|x_i| < 0.1` within 2000 steps. Pinned by
  `test_rmsprop_quadratic_convergence`.
- [x] AC-3: Centered variant converges on the same quadratic
  (`test_rmsprop_centered_convergence`).
- [x] AC-4: Momentum variant converges (`test_rmsprop_momentum_convergence`).
- [x] AC-5: Weight-decay-only update (zero grad) shrinks the parameter
  (`test_rmsprop_weight_decay`).
- [x] AC-6: Single-step numerical check — for `p=[2.0]`, `lr=0.1`,
  `alpha=0.9`, `eps=1e-8`, gradient `4.0`, expected
  `2.0 - 0.1 * 4.0 / (sqrt(1.6) + 1e-8)`. Pinned by
  `test_rmsprop_single_step_numerics` (#1155 regression lock).
- [x] AC-7: `foreach: true` matches the legacy CPU path within `1e-4`
  on basic / momentum / centered / weight-decay configurations
  (`test_rmsprop_foreach_basic_parity`, `_with_momentum`, `_with_centered`,
  `_with_weight_decay`).
- [x] AC-8: `state_dict` round-trip preserves every state field within
  `1e-10` (`test_rmsprop_state_dict_roundtrip`).
- [x] AC-9: `zero_grad()` clears every parameter's gradient
  (`test_rmsprop_zero_grad`).
- [x] AC-10: `lr()`/`set_lr()` accessors propagate through every
  param_group (`test_rmsprop_lr_accessors`).

## Architecture

### Config

`#[derive(Debug, Clone)]` (not `Copy` — the field count plus boolean flags
push it above the inline cost). All `pub` fields with builder-style
`with_*` setters returning `Self`.

### Two step paths — legacy CPU and `foreach` on-device

The legacy CPU step (`step()` body when `config.foreach == false`)
reads `param.data_vec()` and `grad.data_vec()` into owned `Vec<T>`
buffers, runs the rmsprop algebra in scalar `T` arithmetic, then writes
back via `unsafe { param.tensor().update_data(&new_values) }` inside a
`no_grad` closure (`rmsprop.rs`). The SAFETY block documents the
sole-writer invariants.

The `foreach` path (`step_foreach`, `rmsprop.rs`) keeps
`square_avg`/`grad_avg`/`momentum_buf` as `Tensor<T>` on the parameter's
device and composes the entire update via `add`/`mul`/`sqrt`/`sub`/`neg`/`div`
primitives. Commit is via `into_storage_and_shape()` + `unsafe {
param_t.update_storage(storage) }` (the SAFETY block documents the four
sole-writer invariants). The two paths are kept in lock-step via the
`*_foreach_parity_*` tests.

### Numerical contract: eps OUTSIDE sqrt (#1155)

Both paths use `sqrt(square_avg) + eps`, not `sqrt(square_avg + eps)`.
The inline comment (`rmsprop.rs` and the matching comment in the
foreach path at `rmsprop.rs`) cites upstream's two-line
`_single_tensor_rmsprop`:

```text
avg = square_avg.sqrt()
avg = avg.add_(eps)
```

This was a real regression that surfaced via the Phase C.2 trajectory
harness; the test `test_rmsprop_single_step_numerics` pins it.

### Why `ParamKey` is *not* used (predates CL-1122)

Rmsprop is one of the older optimizers in the crate and uses a tuple
`type ParamKey = (usize, usize);` (`rmsprop.rs`) directly as the
HashMap key, plus a hand-rolled `"group{gi}_param{pi}"` wire format in
state_dict serialization. The CL-1122 typed `ParamKey` rollout
explicitly grandfathered this file. Newer optimizers (Adam, RAdam,
NAdam, Rprop, SparseAdam, …) use `crate::param_key::ParamKey`.

### Non-test production consumers

`ferrotorch/src/lib.rs` `pub use ferrotorch_optim::*;` re-exports
`Rmsprop` and `RmspropConfig` as `ferrotorch::optim::{Rmsprop,
RmspropConfig}`.

`ferrotorch-optim/examples/optimizer_trajectory_dump.rs`
constructs `Rmsprop::new(params, RmspropConfig::default().with_lr(lr))`
inside the trajectory-dump harness — a non-test production consumer
binary (examples are not gated by `#[cfg(test)]`).

## Parity contract

`parity_ops = []`. RMSprop's parity is asserted via the unit-test
gauntlet plus the cross-path parity tests (`test_rmsprop_foreach_*_parity`)
which keep the foreach and legacy CPU implementations in sync.

Edge cases the code owns:

- **`alpha == 0`** — `square_avg` is replaced every step (instantaneous
  squared-grad).
- **`alpha == 1`** — `square_avg` never updates after init; division by
  `eps` only.
- **`centered == true` with `square_avg < grad_avg^2`** — sqrt of a
  negative number; the code does NOT clamp (matches upstream — known
  numerical hazard when alpha very close to 1).
- **`maximize == true`** — gradient negated in-place before square-avg
  update.
- **`weight_decay > 0` + `maximize == true`** — the L2 penalty is added
  AFTER the negation, so the weight-decay term still pulls the parameter
  toward zero (matches upstream).

## Verification

Tests in `mod tests` of `rmsprop.rs` (12 tests):

- `test_rmsprop_quadratic_convergence`
- `test_rmsprop_centered_convergence`
- `test_rmsprop_momentum_convergence`
- `test_rmsprop_weight_decay`
- `test_rmsprop_single_step_numerics` (#1155 lock)
- `test_rmsprop_state_dict_roundtrip`
- `test_rmsprop_zero_grad`
- `test_rmsprop_lr_accessors`
- `test_rmsprop_skips_none_grad`
- `test_rmsprop_foreach_basic_parity` (CL-497 lock)
- `test_rmsprop_foreach_parity_with_momentum`
- `test_rmsprop_foreach_parity_with_centered`
- `test_rmsprop_foreach_parity_with_weight_decay`

Smoke command:

```bash
cargo test -p ferrotorch-optim --lib rmsprop:: 2>&1 | tail -3
```

Expected: `13 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct RmspropConfig` at `ferrotorch-optim/src/rmsprop.rs` mirroring `torch/optim/rmsprop.py:31`; non-test consumer: `ferrotorch/src/lib.rs` `pub use ferrotorch_optim::*;` re-export AND `ferrotorch-optim/examples/optimizer_trajectory_dump.rs` constructs `RmspropConfig::default().with_lr(lr)`. |
| REQ-2 | SHIPPED | impl: `pub struct Rmsprop<T>` at `ferrotorch-optim/src/rmsprop.rs` plus `impl<T: Float> Optimizer<T>` at `ferrotorch-optim/src/rmsprop.rs` mirroring `torch.optim.RMSprop`; non-test consumer: `ferrotorch-optim/examples/optimizer_trajectory_dump.rs` calls `Rmsprop::new(params, cfg)`. |
| REQ-3 | SHIPPED | impl: `struct ParamState<T> { square_avg, grad_avg, momentum_buf }` at `ferrotorch-optim/src/rmsprop.rs` mirroring `torch/optim/rmsprop.py:122`; non-test consumer: `ferrotorch-optim/examples/optimizer_trajectory_dump.rs` re-uses the optimizer end-to-end (state lives inside it). |
| REQ-4 | SHIPPED | impl: square_avg update at `ferrotorch-optim/src/rmsprop.rs` + centered/non-centered avg compute at `ferrotorch-optim/src/rmsprop.rs` mirroring `torch/optim/rmsprop.py:265-340`; non-test consumer: `ferrotorch-optim/examples/optimizer_trajectory_dump.rs` trajectory dump. |
| REQ-5 | SHIPPED | impl: `sq.sqrt() + eps_t` at `ferrotorch-optim/src/rmsprop.rs` and `add(&sq, &eps_t)?` at `ferrotorch-optim/src/rmsprop.rs` (foreach path) mirroring `torch/optim/rmsprop.py:_single_tensor_rmsprop`'s two-step `avg.sqrt_(); avg.add_(eps)`; non-test consumer: `ferrotorch-optim/examples/optimizer_trajectory_dump.rs`. |
| REQ-6 | SHIPPED | impl: momentum-buffered branch at `ferrotorch-optim/src/rmsprop.rs` mirroring `torch/optim/rmsprop.py:289-318`; non-test consumer: `ferrotorch/src/lib.rs` re-export. |
| REQ-7 | SHIPPED | impl: L2 wd at `ferrotorch-optim/src/rmsprop.rs` mirroring upstream's `_single_tensor_rmsprop` weight-decay branch; non-test consumer: `ferrotorch/src/lib.rs` re-export. |
| REQ-8 | SHIPPED | impl: `maximize` negation at `ferrotorch-optim/src/rmsprop.rs`; non-test consumer: `ferrotorch/src/lib.rs` re-export. |
| REQ-9 | SHIPPED | impl: `step_foreach` at `ferrotorch-optim/src/rmsprop.rs` (device-resident path) mirroring upstream's `_multi_tensor_rmsprop`; non-test consumer: `ferrotorch/src/lib.rs` re-export. |
| REQ-10 | SHIPPED | impl: `state_dict` at `ferrotorch-optim/src/rmsprop.rs` + `load_state_dict` at `ferrotorch-optim/src/rmsprop.rs`; non-test consumer: `ferrotorch/src/lib.rs` re-export. |
