# ferrotorch-optim — Adamax (Adam variant using L-infinity norm)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/optim/adamax.py
-->

## Summary

`ferrotorch-optim/src/adamax.rs` implements the Adamax optimizer
(Kingma & Ba, ICLR 2015 Section 7), mirroring `torch.optim.Adamax`
in `torch/optim/adamax.py`. Adamax replaces Adam's L2-norm second
moment with the L-infinity norm
(`u_t = max(beta2 * u_{t-1}, |g_t| + eps)`), eliminating the need
for second-moment bias correction. The Rust impl exposes
`AdamaxConfig` and the `Adamax<T: Float>` struct implementing the
workspace-local `Optimizer<T>` trait, with a legacy CPU f64 path,
a `step_foreach` on-device tensor-op path, and the CL-1105 CUDA
auto-route.

## Requirements

- REQ-1: `pub struct AdamaxConfig` carries
  `lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
  foreach=false`, matching `torch.optim.Adamax.__init__` defaults
  (`torch/optim/adamax.py:29-121`).
- REQ-2: `pub struct Adamax<T: Float>` implements `Optimizer<T>`
  with all eight trait methods.
- REQ-3: The legacy CPU `step` path mirrors `_single_tensor_adamax`
  (`torch/optim/adamax.py:226-304`):
  `exp_avg = beta1 * exp_avg + (1 - beta1) * g`,
  `exp_inf = max(beta2 * exp_inf, |g| + eps)`,
  bias-correct only the first moment: `bc1 = 1 - beta1^t`,
  `clr = lr / bc1`, `param = param - clr * exp_avg / exp_inf`.
- REQ-4: The foreach path `step_foreach` keeps `exp_avg` and
  `exp_inf` on the parameter's device via `Tensor<T>` storage,
  mirroring `_multi_tensor_adamax`
  (`torch/optim/adamax.py:306-423`).
- REQ-5: CL-1105 auto-route: dispatch to `step_foreach` when any
  parameter is CUDA-resident even if `config.foreach == false`.
- REQ-6: `state_dict` serialises per-parameter `step_count`,
  `exp_avg`, `exp_inf`. Keys are `"g{group_idx}_p{param_idx}"` via
  `ParamKey::Display`.
- REQ-7: Parameters whose `.grad()` is `None` are skipped.

## Acceptance Criteria

- [x] AC-1: `AdamaxConfig::default()` returns the PyTorch defaults
  (`lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0`).
- [x] AC-2: `impl<T: Float> Optimizer<T> for Adamax<T>` compiles.
- [x] AC-3: `test_adamax_convergence_quadratic` minimises `x^2` from
  5.0 to within 0.1 of 0 in 3000 steps with `lr=0.01`.
- [x] AC-4: `test_adamax_convergence_two_params` minimises
  `x^2 + y^2` to within 0.1 of `(0, 0)` from `(3, -2)`.
- [x] AC-5: `test_adamax_weight_decay` confirms the L2 weight-decay
  branch produces a parameter movement from 5.0.
- [x] AC-6: `test_adamax_state_dict_roundtrip` round-trips
  `exp_avg`, `exp_inf`, `step_count`.
- [x] AC-7: Two foreach-parity tests
  (`test_adamax_foreach_basic_parity`,
  `test_adamax_foreach_parity_with_weight_decay`) confirm CPU and
  foreach paths agree to within 1e-4.

## Architecture

### `AdamaxConfig` (REQ-1)

The config is `#[derive(Debug, Clone, Copy)]` `#[non_exhaustive]`
with five `pub` fields. `Default` matches PyTorch (note
`lr=2e-3`, not `1e-3`).

### `Adamax<T>` struct (REQ-2)

Owns:

- `param_groups: Vec<ParamGroup<T>>`
- `config: AdamaxConfig`
- `state: HashMap<ParamKey, AdamaxParamState>` (CPU path)
- `foreach_state: HashMap<ParamKey, AdamaxForeachState<T>>` (foreach)

`AdamaxParamState` holds `step_count: u64`, `exp_avg: Vec<f64>`,
`exp_inf: Vec<f64>`.

`AdamaxForeachState<T>` holds the same fields with `exp_avg` and
`exp_inf` as `Tensor<T>`.

### Legacy CPU `step` (REQ-3, REQ-7)

1. CL-1105 auto-route check (REQ-5).
2. Skip parameters with no gradient.
3. Read `param` and `grad` as `Vec<f64>`.
4. Apply L2 weight decay.
5. Lazy-init state.
6. Update first moment:
   `exp_avg[i] = beta1 * exp_avg[i] + (1 - beta1) * g`.
7. Update infinity norm:
   `exp_inf[i] = max(beta2 * exp_inf[i], |g| + eps)`.
8. Bias correction for first moment only:
   `bc1 = 1 - beta1^step`, `clr = lr / bc1`.
9. Parameter update:
   `new_param[i] = param[i] - clr * exp_avg[i] / exp_inf[i]`.
10. `update_data` (unsafe; SAFETY block).

### Foreach path (REQ-4)

Uses tensor ops with `abs`, `add`, `mul`, `div`, `sub`. The
elementwise max for `exp_inf` uses
`elemwise_max(beta2_scaled_inf, abs_grad_plus_eps, device)`
from `crate::foreach_utils`.

### CL-1105 auto-route (REQ-5)

Same pattern as Adagrad/Adadelta/Asgd.

### State-dict (REQ-6)

`state_dict` keyed by `ParamKey::Display` to
`"g{group_idx}_p{param_idx}"`. Entries contain `step_count`,
`exp_avg`, `exp_inf`. `load_state_dict` parses keys via
`FromStr`.

### Non-test production consumers

- `ferrotorch-optim/src/lib.rs:31` — `pub use adamax::{Adamax,
  AdamaxConfig};`
- `ferrotorch/src/lib.rs:61` — `pub use ferrotorch_optim::*;`
- `ferrotorch-train/src/learner.rs:28` — `use
  ferrotorch_optim::Optimizer;`

## Parity contract

`parity_ops = []`. Edge-cases:

- **Initial `exp_inf` = 0**: when both `beta2 * exp_inf` and
  `|g|` are zero (e.g. `g == 0` on first step), the parameter
  update would divide by zero — `eps` is added inside the max:
  `max(beta2 * 0, 0 + eps) = eps`. Matches
  `torch/optim/adamax.py:268`.
- **No bias correction for second moment**: PyTorch's Adamax
  intentionally omits the second-moment bias correction (the
  paper's derivation makes this unnecessary). Ferrotorch follows.
- **`weight_decay` is L2**: added to gradient
  (`g += wd * p`), not decoupled.

## Verification

Tests in `mod tests in adamax.rs` (8 tests):

- Convergence: `test_adamax_convergence_quadratic`,
  `test_adamax_convergence_two_params`.
- Algorithm: `test_adamax_weight_decay`.
- Trait: `test_adamax_zero_grad`, `test_adamax_lr_accessors`.
- Config: `test_adamax_default_config`.
- State-dict: `test_adamax_state_dict_roundtrip`.
- Foreach parity: `test_adamax_foreach_basic_parity`,
  `test_adamax_foreach_parity_with_weight_decay`.

Smoke command:

```bash
cargo test -p ferrotorch-optim --lib adamax:: 2>&1 | tail -3
```

Expected: `9 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct AdamaxConfig` + `impl Default` in `adamax.rs` mirroring `torch/optim/adamax.py:29-121`; non-test consumer: `ferrotorch-optim/src/lib.rs:31` re-exports `AdamaxConfig`; `ferrotorch/src/lib.rs:61` re-exports the optim surface. |
| REQ-2 | SHIPPED | impl: `impl<T: Float> Optimizer<T> for Adamax<T>` block in `adamax.rs`; non-test consumer: `ferrotorch-train/src/learner.rs:28` consumes the `Optimizer` trait. |
| REQ-3 | SHIPPED | impl: legacy CPU `step` (else branch after `any_cuda` check) in `adamax.rs` mirroring `_single_tensor_adamax` in `torch/optim/adamax.py:226-304`; non-test consumer: `ferrotorch/src/lib.rs:61` re-exports `Adamax`. |
| REQ-4 | SHIPPED | impl: `Adamax::step_foreach` method in `adamax.rs` mirroring `_multi_tensor_adamax` in `torch/optim/adamax.py:306-423`; non-test consumer: `ferrotorch/src/lib.rs:61` re-exports `AdamaxConfig::with_foreach(true)`. |
| REQ-5 | SHIPPED | impl: `let any_cuda = ...; if self.config.foreach || any_cuda { return self.step_foreach(); }` at the top of `step()` in `adamax.rs` (CL-1105); non-test consumer: `ferrotorch-train/src/learner.rs:28` Optimizer trait drives the auto-routed path. |
| REQ-6 | SHIPPED | impl: `state_dict` / `load_state_dict` methods in `adamax.rs` keyed by `ParamKey::Display`; non-test consumer: `ferrotorch-serialize/src/checkpoint.rs:48` `use ferrotorch_optim::OptimizerState;`. |
| REQ-7 | SHIPPED | impl: `match grad_opt { Some(g) => g, None => continue };` in both `step` and `step_foreach` in `adamax.rs`; non-test consumer: `ferrotorch-train/src/learner.rs` exercises this skip for frozen layers. |
