# `Rprop` — Resilient Backpropagation optimizer

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/optim/rprop.py
-->

## Summary

`ferrotorch-optim/src/rprop.rs` defines `Rprop<T>` and `RpropConfig` — the
Riedmiller & Braun (1993) Resilient Backpropagation optimizer that adapts
per-element step sizes from the sign of consecutive gradients. Mirrors
`torch.optim.Rprop` (`torch/optim/rprop.py:30`).

## Requirements

- REQ-1: `pub struct RpropConfig` with `lr` (default `0.01`),
  `etas: (f64, f64)` (default `(0.5, 1.2)`), and
  `step_sizes: (f64, f64)` (default `(1e-6, 50.0)`). Field defaults
  mirror upstream's `__init__` signature (`torch/optim/rprop.py:31-43`).
- REQ-2: `pub struct Rprop<T: Float>` implementing `Optimizer<T>`
  with per-`ParamKey` state holding `step_count`, `prev_grad`, and
  `step_size` (1-D buffers of length `numel`). Mirrors upstream's
  `state["step"] / state["prev"] / state["step_size"]`
  (`torch/optim/rprop.py:223-260`).
- REQ-3: Element-wise sign-product adaptation:
  - `sign(g_prev[i] * g_t[i]) > 0` ⇒ `step_size[i] = min(step_size[i] * eta_plus, step_max)`
  - `sign(g_prev[i] * g_t[i]) < 0` ⇒ `step_size[i] = max(step_size[i] * eta_minus, step_min)` and `g_t[i] = 0`
  - `== 0` ⇒ step size unchanged.
  Then `p[i] -= sign(g_t[i]) * step_size[i]`.
  Mirrors `_single_tensor_rprop` at `torch/optim/rprop.py:223-292`.
- REQ-4: `prev_grad[i]` is set to `effective_grad` (zeroed on sign reversal)
  so the next step's sign-product is `0` rather than continuing to flip;
  matches upstream's "zero out the gradient" branch
  (`torch/optim/rprop.py:279-282`).
- REQ-5: `state_dict` / `load_state_dict` round-trip `step_count`,
  `prev_grad`, `step_size` keyed by `ParamKey::to_string()` (the
  `"g{}_p{}"` wire format).
- REQ-6: Builder-style setters `with_lr`, `with_etas`, `with_step_sizes`
  for ergonomic construction.
- REQ-7: `Optimizer<T>` trait methods: `step`, `zero_grad`, `lr`,
  `set_lr`, `param_groups`, `param_groups_mut`, `add_param_group`,
  `state_dict`, `load_state_dict`.
- REQ-8: CUDA fail-fast — when any parameter is on CUDA, `step()`
  returns `FerrotorchError::NotImplementedOnCuda { op: "Rprop" }`.
  Documented divergence from upstream (which supports CUDA); tracked
  by #1468.

## Acceptance Criteria

- [x] AC-1: `RpropConfig::default()` returns
  `{ lr: 1e-2, etas: (0.5, 1.2), step_sizes: (1e-6, 50.0) }`.
- [x] AC-2: After two consecutive same-sign gradients, the step size for
  that element multiplies by `eta_plus` (0.01 → 0.012). Pinned by
  `test_rprop_step_size_increase`.
- [x] AC-3: After a sign reversal, the step size multiplies by `eta_minus`
  (0.01 → 0.005) and the next-step gradient is treated as zero. Pinned by
  `test_rprop_step_size_decrease`.
- [x] AC-4: Step size clamps to `step_max` after many consistent steps.
  Pinned by `test_rprop_step_size_clamping`.
- [x] AC-5: Quadratic convergence: `f(x) = x^2` from `x = 5.0` converges
  to `|x| < 0.5` within 1000 steps. Pinned by
  `test_rprop_convergence_quadratic`.
- [x] AC-6: `state_dict` round-trip preserves all three state fields
  exactly. Pinned by `test_rprop_state_dict_roundtrip`.
- [x] AC-7: CUDA parameter routes to `NotImplementedOnCuda { op: "Rprop" }`
  (verified by the runtime CUDA-detect guard in `step()`).

## Architecture

### Config

`RpropConfig` is `#[derive(Debug, Clone, Copy)]` with `#[non_exhaustive]`
and three `pub` fields. The `with_*` setters return `Self` for fluent
construction. `lr` serves both as the initial step size and as the
fallback when `param_groups.first()` is empty.

### `Rprop<T>` and per-step body

The CPU step body (the only step path — see CL-1105 design note in the
file's module-level doc-comment) does the following for each
`(gi, pi)`:

1. Fail-fast on CUDA (`tensor.is_cuda()` → `NotImplementedOnCuda`).
2. Clone tensor + grad handles to release the immutable borrow on
   `self.param_groups` (CL-1125 — needed to mutate `self.state` and the
   workspaces in the same closure).
3. Fill three reusable workspaces via
   `fill_f64_workspace(&mut self.param_workspace, &tensor_handle)?`,
   `fill_f64_workspace(&mut self.grad_workspace, &grad_handle)?`,
   and `resize_typed_workspace(&mut self.new_values_workspace, numel)`
   (CL-1125 — eliminates per-step `Vec<f64>::new()` heap traffic).
4. Lazy-init the `RpropParamState` (filling `prev_grad = 0`,
   `step_size = init_lr`).
5. Element-wise compute the sign-product, adapt the step size, zero
   the effective gradient on sign reversal, then write
   `p[i] -= sign(g_t[i]) * step_size[i]` into `new_values_workspace`.
6. Commit with `unsafe { tensor_handle.update_data(...) }` inside
   `no_grad` (SAFETY block at `rprop.rs` documents the four
   sole-writer invariants).

### `state_dict` / `load_state_dict`

`state_dict()` produces `OptimizerState` rendering each `ParamKey`
through `Display` to the `"g{}_p{}"` wire format with three sub-entries:
`step_count`, `prev_grad`, `step_size`. `load_state_dict()` parses the key
back via `ParamKey::from_str` and rejects missing `prev_grad` or
`step_size` with `FerrotorchError::InvalidArgument`. This is the same
wire format every other CL-1122 optimizer uses.

### Why CUDA fail-fast (CL-1105 design note)

The rprop update uses an element-wise sign-product branch (`if g*prev > 0
{...} else if g*prev < 0 {...}`) and a conditional gradient zero-out on
sign reversal. ferrotorch-core does NOT currently expose a
device-resident `sign` or `where_bt`, so the only way to express rprop on
CUDA is either (a) silently round-trip to CPU (forbidden by goal.md
R-CODE-4) or (b) introduce arithmetic bias via abs-division tricks
(numerically unsound). The file joins SparseAdam and Adafactor in the
fail-fast group. Lift this when ferrotorch-core gains
`sign` + `where_bt`. Tracked by #1468.

### Non-test production consumers

`ferrotorch/src/lib.rs` `pub use ferrotorch_optim::*;` re-exports
`Rprop` and `RpropConfig` as `ferrotorch::optim::{Rprop, RpropConfig}`.
Every downstream training crate consumes the optimizer through the meta
crate's `optim` re-export.

## Parity contract

`parity_ops = []`. Rprop has no parity-sweep op because its sign-product
update is not a standard differentiable op; convergence parity is asserted
via the unit-test gauntlet against the analytic stepwise behavior.

Edge cases the code owns:

- **`g_prev[i] * g_t[i] == 0`** (boundary between `+` and `-`):
  step size unchanged, gradient kept as-is.
- **`step_size` saturates at `step_max`** — never exceeds `step_max + 1e-10`
  (clamping verified by `test_rprop_step_size_clamping`).
- **`step_size` underflows toward `step_min`** — never falls below
  `step_min`.
- **No gradient** (`grad().is_none()`) — `step()` skips that parameter
  (no init of state).
- **CUDA parameter** — `NotImplementedOnCuda` early-return; no demote.

## Verification

Tests in `mod tests` of `rprop.rs` (8 tests):

- `test_rprop_convergence_quadratic`
- `test_rprop_convergence_two_params`
- `test_rprop_step_size_increase`
- `test_rprop_step_size_decrease`
- `test_rprop_zero_grad`
- `test_rprop_state_dict_roundtrip`
- `test_rprop_lr_accessors`
- `test_rprop_default_config`
- `test_rprop_step_size_clamping`

Smoke command:

```bash
cargo test -p ferrotorch-optim --lib rprop:: 2>&1 | tail -3
```

Expected: `9 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct RpropConfig` with `lr`/`etas`/`step_sizes` at `ferrotorch-optim/src/rprop.rs` mirroring `torch/optim/rprop.py:31`; non-test consumer: `ferrotorch/src/lib.rs` `pub use ferrotorch_optim::*;` re-export. |
| REQ-2 | SHIPPED | impl: `pub struct Rprop<T>` at `ferrotorch-optim/src/rprop.rs` with `state: HashMap<ParamKey, RpropParamState>` and `impl<T: Float> Optimizer<T>` at `ferrotorch-optim/src/rprop.rs` mirroring `_single_tensor_rprop` at `torch/optim/rprop.py:223`; non-test consumer: `ferrotorch/src/lib.rs` re-export. |
| REQ-3 | SHIPPED | impl: sign-product adaptation at `ferrotorch-optim/src/rprop.rs` (sign_product → eta_plus/eta_minus clamp → effective_grad → write) mirroring `torch/optim/rprop.py:223-292`; non-test consumer: `ferrotorch/src/lib.rs` re-export. |
| REQ-4 | SHIPPED | impl: `state.prev_grad[i] = effective_grad` at `ferrotorch-optim/src/rprop.rs` (effective_grad zeroed on reversal) mirroring `torch/optim/rprop.py:279`; non-test consumer: `ferrotorch/src/lib.rs` re-export. |
| REQ-5 | SHIPPED | impl: `state_dict` at `ferrotorch-optim/src/rprop.rs` + `load_state_dict` at `ferrotorch-optim/src/rprop.rs` keyed by `ParamKey::to_string()` ("`g{}_p{}`"); non-test consumer: `ferrotorch/src/lib.rs` re-export. |
| REQ-6 | SHIPPED | impl: `with_lr`/`with_etas`/`with_step_sizes` at `ferrotorch-optim/src/rprop.rs`; non-test consumer: `ferrotorch/src/lib.rs` re-export. |
| REQ-7 | SHIPPED | impl: full `Optimizer<T>` trait impl at `ferrotorch-optim/src/rprop.rs`; non-test consumer: `ferrotorch/src/lib.rs` re-export which gates downstream `ferrotorch::optim::Rprop` use. |
| REQ-8 | SHIPPED | impl: `FerrotorchError::NotImplementedOnCuda { op: "Rprop" }` at `ferrotorch-optim/src/rprop.rs`; non-test consumer: `ferrotorch/src/lib.rs` re-export. Intentional divergence tracked by #1468 (lift when ferrotorch-core gains `sign` + `where_bt`). |
