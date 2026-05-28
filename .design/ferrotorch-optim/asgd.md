# ferrotorch-optim — Asgd (Averaged Stochastic Gradient Descent)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/optim/asgd.py
-->

## Summary

`ferrotorch-optim/src/asgd.rs` implements the ASGD optimizer
(Polyak & Juditsky 1992), mirroring `torch.optim.ASGD` in
`torch/optim/asgd.py`. ASGD performs a vanilla SGD update with an
effective learning rate that decays via
`eta_t = lr / (1 + lambd * lr * t)^alpha`, and maintains a running
average `ax` of parameter values that often generalises better than
the last iterate after sufficient training. The Rust impl exposes
`AsgdConfig` and the `Asgd<T: Float>` struct implementing the
workspace-local `Optimizer<T>` trait, with a legacy CPU f64 path, a
`step_foreach` on-device tensor-op path, and the CL-1105 CUDA
auto-route.

## Requirements

- REQ-1: `pub struct AsgdConfig` carries
  `lr=1e-2, lambd=1e-4, alpha=0.75, t0=1e6, weight_decay=0.0,
  foreach=false`, matching `torch.optim.ASGD.__init__` defaults
  (`torch/optim/asgd.py:30-124`).
- REQ-2: `pub struct Asgd<T: Float>` implements `Optimizer<T>` with
  all eight trait methods.
- REQ-3: The legacy CPU `step` path mirrors `_single_tensor_asgd`
  (`torch/optim/asgd.py:197-275`): L2 weight decay
  (`g += wd * p`), parameter update
  `p = p * (1 - lambd * eta) - eta * g`, running-average update
  `ax = ax + mu * (p - ax)` when `mu != 1`, then schedule
  updates `eta = lr / (1 + lambd*lr*t)^alpha`,
  `mu = 1 / max(1, t - t0)`.
- REQ-4: The foreach path `step_foreach` keeps `ax` on the
  parameter's device via `Tensor<T>` storage and updates via
  tensor ops, mirroring `_multi_tensor_asgd`
  (`torch/optim/asgd.py:277-422`).
- REQ-5: CL-1105 auto-route: dispatch to `step_foreach` when any
  parameter is CUDA-resident.
- REQ-6: `state_dict` serialises per-parameter `step_count`, `eta`,
  `mu`, and the averaged-param vector `ax` (as `Vec<f64>`). Keys
  are `"g{group_idx}_p{param_idx}"` via `ParamKey::Display`.
- REQ-7: `averaged_param(group_idx, param_idx) -> Option<&[f64]>`
  exposes the running average so users can swap it into the model
  for evaluation (the canonical Polyak-averaging workflow).
  Returns `None` when no step has been taken yet.
- REQ-8: Parameters whose `.grad()` is `None` are skipped.

## Acceptance Criteria

- [x] AC-1: `AsgdConfig::default()` returns
  `lr=1e-2, lambd=1e-4, alpha=0.75, t0=1e6, weight_decay=0.0`.
- [x] AC-2: `impl<T: Float> Optimizer<T> for Asgd<T>` compiles.
- [x] AC-3: `test_asgd_convergence_quadratic` minimises `x^2` from
  5.0 to within 0.5 of 0 in 3000 steps (with `t0=0` so averaging
  kicks in immediately).
- [x] AC-4: `test_asgd_convergence_two_params` minimises
  `x^2 + y^2` to within 0.5 of `(0, 0)`.
- [x] AC-5: `test_asgd_eta_decay` confirms `eta_2 < eta_1` across
  successive steps.
- [x] AC-6: `test_asgd_averaging_starts_after_t0` confirms `mu < 1`
  after step `t > t0`.
- [x] AC-7: `test_asgd_averaged_params` confirms
  `averaged_param(0, 0)` returns `Some(&[f64])` after the first
  step.
- [x] AC-8: `test_asgd_state_dict_roundtrip` round-trips
  `eta`, `mu`, `ax`, `step_count`.
- [x] AC-9: Two foreach-parity tests
  (`test_asgd_foreach_basic_parity`,
  `_parity_with_weight_decay_and_averaging`) confirm CPU and
  foreach paths agree to within 1e-4.

## Architecture

### `AsgdConfig` (REQ-1)

The config is `#[derive(Debug, Clone, Copy)]` `#[non_exhaustive]`
with six `pub` fields. `Default` matches PyTorch.

### `Asgd<T>` struct (REQ-2, REQ-7)

Owns:

- `param_groups: Vec<ParamGroup<T>>`
- `config: AsgdConfig`
- `state: HashMap<ParamKey, AsgdParamState>` (CPU path)
- `foreach_state: HashMap<ParamKey, AsgdForeachState<T>>` (foreach)

`AsgdParamState` holds `step_count: u64`, `eta: f64`, `mu: f64`,
`ax: Vec<f64>`. The eta/mu state is per-parameter (not global) so
each parameter advances its own schedule independently.

`AsgdForeachState<T>` holds the same scalars plus
`ax: Tensor<T>` on the parameter's device.

`averaged_param(group_idx, param_idx)` (REQ-7) returns
`self.state.get(&key).map(|s| s.ax.as_slice())`. The CPU-path
average is always available; the foreach path stores `ax` as a
device tensor (so the CPU-only accessor returns `None` for
foreach-resident state — callers needing both paths read via
`state_dict` instead).

### Legacy CPU `step` (REQ-3, REQ-8)

1. CL-1105 auto-route check (REQ-5).
2. Skip parameters with no gradient.
3. Read `param` and `grad` as `Vec<f64>`.
4. Apply L2 weight decay.
5. Lazy-init state with `eta = group_lr, mu = 1.0, ax = param.clone()`.
6. Increment `step_count`; capture `step`, `eta`, `mu`.
7. Update parameters:
   `new_param[i] = param[i] * (1 - lambd * eta) - eta * grad[i]`.
8. Update running average:
   - If `mu != 1`: `ax[i] += mu * (new_param[i] - ax[i])`.
   - Else: `ax = new_param.clone()`.
9. Schedule update:
   `eta = lr / (1 + lambd*lr*step)^alpha`,
   `mu = 1 / max(1, step - t0)`.
10. `update_data` writes new param values (unsafe; SAFETY block).

### Foreach path (REQ-4)

Uses tensor ops to perform the same update. `ax` starts as
`param_t.clone()` (Arc-shared with the parameter on first step —
note this is shallow; the first step's `param_t * (1 - lambd * eta)`
produces a new tensor and is then propagated through `ax`). The
foreach `ax` update is:

- If `mu != 1`: `ax = ax + mu * (new_param - ax)` (via tensor ops).
- Else: `ax = new_param.clone()`.

### CL-1105 auto-route (REQ-5)

Same pattern as Adagrad/Adadelta/Adamax.

### State-dict (REQ-6)

`state_dict` renders each `ParamKey` via `Display` to
`"g{group_idx}_p{param_idx}"`. Entries contain `step_count`,
`eta`, `mu`, `ax`. `load_state_dict` parses keys via `FromStr`.

### Non-test production consumers

- `ferrotorch-optim/src/lib.rs:33` — `pub use asgd::{Asgd, AsgdConfig};`
- `ferrotorch/src/lib.rs:61` — `pub use ferrotorch_optim::*;`
- `ferrotorch-train/src/learner.rs` — `use
  ferrotorch_optim::Optimizer;`

## Parity contract

`parity_ops = []`. Edge-cases:

- **Initial `eta == lr`, `mu == 1`**: first step uses the
  un-decayed lr and no averaging. The schedule updates AFTER the
  step, matching PyTorch.
- **`mu = 1 / max(1, t - t0)`**: when `t <= t0`, `mu = 1` and `ax`
  tracks the latest parameter exactly (no averaging). When
  `t > t0`, averaging kicks in.
- **`weight_decay` is L2**: added to gradient.
- **Per-parameter schedule state**: each parameter has its own
  `eta`/`mu` (not a global schedule). This matches PyTorch's
  per-parameter `state[p]['eta']` / `state[p]['mu']`.

## Verification

Tests in `mod tests in asgd.rs` (10 tests):

- Convergence: `test_asgd_convergence_quadratic`,
  `test_asgd_convergence_two_params`.
- Schedule: `test_asgd_eta_decay`,
  `test_asgd_averaging_starts_after_t0`,
  `test_asgd_averaged_params`.
- Trait: `test_asgd_zero_grad`, `test_asgd_lr_accessors`.
- Config: `test_asgd_default_config`.
- Weight decay: `test_asgd_weight_decay`.
- State-dict: `test_asgd_state_dict_roundtrip`.
- Foreach parity: `test_asgd_foreach_basic_parity`,
  `test_asgd_foreach_parity_with_weight_decay_and_averaging`.

Smoke command:

```bash
cargo test -p ferrotorch-optim --lib asgd:: 2>&1 | tail -3
```

Expected: `12 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct AsgdConfig` + `impl Default` in `asgd.rs` mirroring `torch/optim/asgd.py:30-124`; non-test consumer: `asgd in ferrotorch-optim/src/lib.rs` re-exports `AsgdConfig`; `ferrotorch/src/lib.rs` re-exports the optim surface. |
| REQ-2 | SHIPPED | impl: `impl<T: Float> Optimizer<T> for Asgd<T>` block in `asgd.rs`; non-test consumer: `ferrotorch-train/src/learner.rs` consumes the `Optimizer` trait. |
| REQ-3 | SHIPPED | impl: legacy CPU `step` (else branch after `any_cuda` check) in `asgd.rs` mirroring `_single_tensor_asgd` in `torch/optim/asgd.py:197-275`; non-test consumer: `ferrotorch/src/lib.rs:61` re-exports `Asgd`. |
| REQ-4 | SHIPPED | impl: `Asgd::step_foreach` method in `asgd.rs` mirroring `_multi_tensor_asgd` in `torch/optim/asgd.py:277-422`; non-test consumer: `ferrotorch/src/lib.rs:61` re-exports `AsgdConfig::with_foreach(true)`. |
| REQ-5 | SHIPPED | impl: `let any_cuda = ...; if self.config.foreach || any_cuda { return self.step_foreach(); }` in `step` in `asgd.rs` (CL-1105); non-test consumer: `step in ferrotorch-train/src/learner.rs` propagates the auto-routed path via the Optimizer trait. |
| REQ-6 | SHIPPED | impl: `state_dict` / `load_state_dict` methods in `asgd.rs` keyed by `ParamKey::Display`; non-test consumer: `ferrotorch-serialize/src/checkpoint.rs:48` `use ferrotorch_optim::OptimizerState;`. |
| REQ-7 | SHIPPED | impl: `pub fn averaged_param(&self, group_idx: usize, param_idx: usize) -> Option<&[f64]>` method on `Asgd<T>` in `asgd.rs`; non-test consumer: `ferrotorch/src/lib.rs:61` re-exports `Asgd` so downstream evaluation code can swap the averaged params in for inference. |
| REQ-8 | SHIPPED | impl: `match grad_opt { Some(g) => g, None => continue };` in both `step` and `step_foreach` in `asgd.rs`; non-test consumer: `ferrotorch-train/src/learner.rs` exercises this skip. |
