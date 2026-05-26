# ferrotorch-optim — AdamW (decoupled weight decay Adam)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/optim/adamw.py
  - torch/optim/adam.py
-->

## Summary

`ferrotorch-optim/src/adamw.rs` implements the AdamW optimizer
(Loshchilov & Hutter, ICLR 2019), mirroring `torch.optim.AdamW` in
`torch/optim/adamw.py`. AdamW differs from `Adam` in one place: weight
decay is applied to the parameter directly
(`p = p * (1 - lr * weight_decay)`) BEFORE the gradient update,
rather than being added to the gradient as L2 regularisation. This
decoupling produces better generalisation in practice and is the
default optimizer for most modern transformer training. The Rust
impl exposes `AdamWConfig` and the `AdamW<T: Float>` struct
implementing the workspace-local `Optimizer<T>` trait, with a legacy
CPU f64 path and a `step_foreach` on-device tensor-op path.

## Requirements

- REQ-1: `pub struct AdamWConfig` carries `lr=1e-3, betas=(0.9, 0.999),
  eps=1e-8, weight_decay=0.01, maximize=false, foreach=false`,
  matching `torch.optim.AdamW.__init__` defaults
  (`torch/optim/adamw.py:21-128`). Note `weight_decay=0.01` is the
  AdamW-specific default (Adam defaults to 0.0).
- REQ-2: `pub struct AdamW<T: Float>` implements `Optimizer<T>`
  with all eight trait methods.
- REQ-3: The legacy CPU `step` path applies decoupled weight decay
  via `decay_factor = 1 - lr * weight_decay; param = param * decay_factor`
  THEN performs the standard Adam moment + update math without
  L2-adding `wd * param` to the gradient, mirroring
  `_single_tensor_adamw` (`torch/optim/adamw.py:130-180`).
- REQ-4: The foreach path `step_foreach` keeps moments on the
  parameter's device and applies decoupled weight decay via
  `mul(param, scalar(1.0 - lr*wd))` before the gradient update.
- REQ-5: `state_dict` serialises per-parameter `step_count`,
  `exp_avg`, and `exp_avg_sq`. AdamW does NOT track `max_exp_avg_sq`
  by default (there is no AMSGrad flag on the AdamW config in
  this impl).
- REQ-6: Parameters whose `.grad()` is `None` are skipped, mirroring
  PyTorch's `if grad is None: continue`.
- REQ-7: Per-parameter-group weight decay: when `ParamGroup`
  carries its own `weight_decay`, that overrides the config-level
  default, matching PyTorch's `param_groups[i]['weight_decay']`
  override semantics.
- REQ-8: New optimizer construction supports both the single-group
  ergonomic constructor (`AdamW::new(params, config)`) and the
  multi-group constructor (`AdamW::new_with_groups(groups,
  config)`), matching PyTorch's `param_groups` flexibility.

## Acceptance Criteria

- [x] AC-1: `AdamWConfig::default()` returns
  `weight_decay=0.01` (the AdamW-specific default).
- [x] AC-2: `impl<T: Float> Optimizer<T> for AdamW<T>` compiles and
  exposes all eight trait methods.
- [x] AC-3: `test_weight_decay_with_zero_gradient` verifies that
  with `g=0`, only decoupled weight decay shrinks the parameter
  (`param *= (1 - lr*wd)`).
- [x] AC-4: `test_param_norm_decreases_with_zero_gradient` confirms
  the multiplicative decay ratio `(1 - lr*wd)^10` matches over 10
  steps within 1e-10.
- [x] AC-5: `test_convergence_quadratic` minimises `x^2` from 5.0 to
  ≤ 0.01 in 1000 steps.
- [x] AC-6: `test_convergence_rosenbrock` minimises Rosenbrock to
  within 0.1 of `(1, 1)` in 10000 steps.
- [x] AC-7: `test_multiple_param_groups` confirms per-group
  weight-decay override (`group2` decays, `group1` does not).
- [x] AC-8: `test_state_dict_round_trip` round-trips `exp_avg`,
  `exp_avg_sq`, `step_count`.
- [x] AC-9: Four foreach-parity tests
  (`test_adamw_foreach_basic_parity_no_decay`,
  `_parity_with_weight_decay`, `_multiple_steps_bias_correction`,
  `_skips_params_without_grad`, `_convergence_quadratic`,
  `_long_run_drives_to_zero_with_zero_grad`) all pass.

## Architecture

### `AdamWConfig` (REQ-1)

Same shape as `AdamConfig` minus the `amsgrad` flag. The
`weight_decay=0.01` default differs from Adam (Adam defaults to
0.0). Builder methods follow the `with_*` convention.

### `AdamW<T>` struct (REQ-2)

Owns:

- `param_groups: Vec<ParamGroup<T>>`
- `config: AdamWConfig`
- `state: HashMap<ParamKey, AdamWParamState>` (CPU path)
- `foreach_state: HashMap<ParamKey, AdamWForeachState<T>>` (foreach)
- `param_workspace`, `grad_workspace`, `exp_avg_new_workspace`,
  `exp_avg_sq_new_workspace`, `new_values_workspace` — the
  CL-1125 reusable buffers. AdamW additionally stages new moments
  through `exp_avg_new_workspace` / `exp_avg_sq_new_workspace` so
  that on partial failure (e.g. GPU upload error inside
  `update_data`) the optimizer state remains consistent and the
  step can be retried.

`AdamWParamState` holds CPU moments (`exp_avg`, `exp_avg_sq`,
`step_count`); no `max_exp_avg_sq` (AdamW has no AMSGrad variant
in this impl — PyTorch's AdamW class inherits from Adam and gets
AMSGrad via the parent's `amsgrad` flag; ferrotorch keeps the
types separate).

### Legacy CPU `step` (REQ-3, REQ-6, REQ-7)

1. Skip parameters with no gradient.
2. Fill `param_workspace` / `grad_workspace` via
   `fill_f64_workspace` (CL-1125).
3. Apply maximize negation.
4. Compute `decay_factor = 1 - group_lr * group_wd` — note
   `group_lr` and `group_wd` are pulled from the `ParamGroup`,
   not the config (REQ-7).
5. Lazy-init state on first step.
6. Stage new moments into the reusable workspaces:
   `exp_avg_new_workspace[i] = beta1 * state.exp_avg[i] + (1 - beta1) * g`
   `exp_avg_sq_new_workspace[i] = beta2 * state.exp_avg_sq[i] + (1 - beta2) * g^2`
7. Compute new parameter values: `m_hat = m_new / bc1`,
   `v_hat = v_sq_new / bc2`,
   `decayed = param * decay_factor`,
   `updated = decayed - lr * m_hat / (sqrt(v_hat) + eps)`.
8. `update_data` writes the new parameter values (unsafe; SAFETY block).
9. After successful write, `mem::swap` the staged buffers into
   `state.exp_avg` / `state.exp_avg_sq`. This makes the staged
   buffers become the "old" ones for the next step (zero-copy
   capacity retention).

### Foreach path (REQ-4)

Activated when `config.foreach == true`. Uses tensor-op kernels
to perform the moment update + decoupled weight decay + parameter
update entirely on the parameter's device. The decoupled-decay
step is `mul(param, scalar(1.0 - lr*wd))` — the gradient is NOT
modified before the moment update, distinguishing AdamW from Adam.

### State-dict (REQ-5)

`state_dict` renders each `ParamKey` via `Display` to
`"g{group_idx}_p{param_idx}"` (CL-1122). Per-parameter
entries contain `step_count`, `exp_avg`, `exp_avg_sq`.
`load_state_dict` parses keys via `FromStr`. AdamW does not
track `max_exp_avg_sq` so checkpoints are smaller than Adam's
AMSGrad checkpoints.

### Construction (REQ-8)

Two constructors:

- `AdamW::new(params: Vec<Parameter<T>>, config: AdamWConfig)` —
  single group, all params share `config.lr` and `config.weight_decay`.
- `AdamW::new_with_groups(groups: Vec<ParamGroup<T>>, config:
  AdamWConfig)` — multi-group, each `ParamGroup` carries its own
  `lr` and `weight_decay` (REQ-7).

### Non-test production consumers

- `ferrotorch-optim/src/lib.rs:32` — `pub use adamw::{AdamW, AdamWConfig};`
- `ferrotorch/src/lib.rs:51` — `pub use ferrotorch_optim::{Adam,
  AdamW, Optimizer, Sgd};` re-exports `AdamW` in the prelude.
- `ferrotorch-train/src/learner.rs:28` — `use
  ferrotorch_optim::Optimizer;` drives `AdamW::step` indirectly via
  the trait.
- `ferrotorch-train/examples/multi_epoch_train_dump.rs` — uses
  `Adam` directly; `AdamW` is the AdamW counterpart available via
  the same `Optimizer<T>` trait.

## Parity contract

`parity_ops = []`. Edge-cases the impl owns:

- **Decoupled decay vs L2**: `param *= (1 - lr * wd)` BEFORE the
  gradient update; the gradient is never modified.
- **Per-group override**: `ParamGroup::weight_decay` takes
  precedence over the config-level default. The
  `multiple_param_groups` test pins this.
- **State-staging on partial failure**: new moments are written
  into `exp_avg_new_workspace` / `exp_avg_sq_new_workspace` first,
  swapped into state only after the parameter write succeeds. A
  failure mid-step leaves state unchanged so the user can retry.
- **AdamWConfig has no `amsgrad`**: divergence from
  `torch.optim.AdamW` (which inherits from `Adam` and gets the
  `amsgrad` kwarg). If/when this matters, future work files a
  separate REQ; current impl does not expose the flag.

## Verification

Tests in `mod tests in adamw.rs` (16 tests):

- Config defaults: `test_default_config`.
- Basic step: `test_adamw_single_step`,
  `test_skip_params_without_grad`, `test_zero_grad`.
- Decoupled decay: `test_weight_decay_with_zero_gradient`,
  `test_param_norm_decreases_with_zero_gradient`.
- Convergence: `test_convergence_quadratic`,
  `test_convergence_rosenbrock`,
  `test_convergence_with_autograd`,
  `test_monotonic_loss_decrease`.
- Bias correction: `test_bias_correction_early_steps`.
- Multi-group: `test_multiple_param_groups`, `test_add_param_group`.
- LR: `test_lr_get_set`.
- State-dict: `test_state_dict_round_trip`.
- Foreach parity: `test_adamw_foreach_basic_parity_no_decay`,
  `test_adamw_foreach_parity_with_weight_decay`,
  `test_adamw_foreach_multiple_steps_bias_correction`,
  `test_adamw_foreach_skips_params_without_grad`,
  `test_adamw_foreach_convergence_quadratic`,
  `test_adamw_foreach_long_run_drives_to_zero_with_zero_grad`.

Smoke command:

```bash
cargo test -p ferrotorch-optim --lib adamw:: 2>&1 | tail -3
```

Expected: `22 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct AdamWConfig` + `impl Default { weight_decay: 0.01, ... }` in `adamw.rs` mirroring `torch/optim/adamw.py:21-128`; non-test consumer: `ferrotorch-optim/src/lib.rs:32` re-exports `AdamWConfig`; `ferrotorch/src/lib.rs:51` re-exports `AdamW` in the prelude. |
| REQ-2 | SHIPPED | impl: `impl<T: Float> Optimizer<T> for AdamW<T>` block in `adamw.rs`; non-test consumer: `ferrotorch-train/src/learner.rs:28` consumes the `Optimizer` trait. |
| REQ-3 | SHIPPED | impl: legacy CPU `step` method computes `decayed = param * (1 - lr*wd)` then `updated = decayed - lr * m_hat / (sqrt(v_hat) + eps)` in `adamw.rs` mirroring `torch/optim/adamw.py:130-180`; non-test consumer: `ferrotorch/src/lib.rs:51` `pub use ferrotorch_optim::{AdamW, ...};` exposes the type for use as the default transformer optimiser. |
| REQ-4 | SHIPPED | impl: `AdamW::step_foreach` method in `adamw.rs` performing decoupled decay via `mul(param, scalar(decay_factor))`; non-test consumer: `ferrotorch/src/lib.rs:61` re-exports `AdamWConfig::with_foreach(true)`. |
| REQ-5 | SHIPPED | impl: `state_dict` / `load_state_dict` methods in `adamw.rs` keying by `ParamKey::Display` to `"g{}_p{}"`; non-test consumer: `ferrotorch-serialize/src/checkpoint.rs:48` `use ferrotorch_optim::OptimizerState;` is the on-disk checkpoint reader/writer. |
| REQ-6 | SHIPPED | impl: `let grad_tensor = match tensor.grad()? { Some(g) => g, None => continue };` in both `step` and `step_foreach`; non-test consumer: `ferrotorch-train/src/learner.rs` consumes this skip path for frozen layers. |
| REQ-7 | SHIPPED | impl: `let group_wd = self.param_groups[gi].weight_decay;` inside `step` uses the per-group value (`adamw.rs`); non-test consumer: `ferrotorch/src/lib.rs:61` re-exports `ParamGroup`, which is consumed by external user code constructing multi-group optimisers. |
| REQ-8 | SHIPPED | impl: both `AdamW::new` and `AdamW::new_with_groups` constructors in `adamw.rs`; non-test consumer: `ferrotorch/src/lib.rs:51` `pub use ferrotorch_optim::AdamW;` makes both constructors reachable. |
