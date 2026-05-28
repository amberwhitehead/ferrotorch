# ferrotorch-optim — Adagrad (adaptive subgradient)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/optim/adagrad.py
-->

## Summary

`ferrotorch-optim/src/adagrad.rs` implements the Adagrad optimizer
(Duchi, Hazan & Singer, JMLR 2011), mirroring `torch.optim.Adagrad`
in `torch/optim/adagrad.py`. Adagrad adapts the per-parameter
learning rate by dividing by the square root of the running sum of
squared gradients. The Rust impl exposes `AdagradConfig` and the
`Adagrad<T: Float>` struct implementing the workspace-local
`Optimizer<T>` trait. Two update paths are provided: a CPU
`data_vec()` loop and an on-device `step_foreach` path; the CL-1105
auto-route forces `step_foreach` when any parameter is CUDA-resident
even if `config.foreach == false`, preventing silent CPU demotion.

## Requirements

- REQ-1: `pub struct AdagradConfig` carries
  `lr=0.01, lr_decay=0.0, weight_decay=0.0,
  initial_accumulator_value=0.0, eps=1e-10, maximize=false,
  foreach=false`, matching `torch.optim.Adagrad.__init__` defaults
  (`torch/optim/adagrad.py:29-187`).
- REQ-2: `pub struct Adagrad<T: Float>` implements `Optimizer<T>`
  with all eight trait methods.
- REQ-3: The CPU `step` path mirrors `_single_tensor_adagrad`
  (`torch/optim/adagrad.py:364-431`): maximize negation, L2 weight
  decay (`g += wd * p`), accumulator update (`sum += g^2`),
  effective learning rate `clr = lr / (1 + (step-1) * lr_decay)`,
  parameter update `p = p - clr * g / (sqrt(sum) + eps)`.
- REQ-4: The accumulator (`AdagradParamState::sum`) is a `Tensor<T>`,
  not a `Vec<T>` or `Vec<f64>`. This lets the foreach path operate
  on it directly via tensor ops; the CPU path materialises via
  `data_vec()`. The `initial_accumulator_value` config field
  initialises the accumulator to a non-zero value (e.g. for warm
  starts in sparse training).
- REQ-5: The foreach path `step_foreach` keeps the accumulator on
  the parameter's device via `Tensor<T>` storage and updates via
  tensor ops, mirroring `_multi_tensor_adagrad`
  (`torch/optim/adagrad.py:432-555`).
- REQ-6: CL-1105 auto-route: `step()` checks `any_cuda` (any
  parameter is CUDA-resident) and dispatches to `step_foreach`
  unconditionally, even if `config.foreach == false`. Prevents the
  legacy `data_vec()` path from silently demoting a GPU step to
  CPU.
- REQ-7: `state_dict` serialises per-parameter `sum` (as `Vec<f64>`
  via `tensor_to_f64_vec`) and `step_count`. Keys are
  `"group{gi}_param{pi}"` strings (legacy format predating the
  `ParamKey` migration; Adagrad has not yet been migrated to
  `ParamKey`).
- REQ-8: Parameters whose `.grad()` is `None` are skipped, mirroring
  PyTorch.

## Acceptance Criteria

- [x] AC-1: `AdagradConfig::default()` returns
  `lr=0.01, lr_decay=0.0, weight_decay=0.0,
  initial_accumulator_value=0.0, eps=1e-10`.
- [x] AC-2: `impl<T: Float> Optimizer<T> for Adagrad<T>` compiles.
- [x] AC-3: `test_adagrad_convergence` minimises `x^2` from 5.0 to
  ≤ 1.0 in 100 steps with `lr=0.5`.
- [x] AC-4: `test_accumulator_grows_monotonically` confirms
  `sum_after_step2 >= sum_after_step1` elementwise.
- [x] AC-5: `test_lr_decay_reduces_effective_lr` verifies
  `clr_2 < clr_1` when `lr_decay > 0`.
- [x] AC-6: `test_weight_decay` confirms the effective gradient
  `g + wd * p` produces a larger accumulator
  `(1 + 0.5*5)^2 = 12.25` than the bare-gradient accumulator
  `1.0` after step 1.
- [x] AC-7: `test_state_dict_roundtrip` round-trips `sum` and
  `step_count`.
- [x] AC-8: Four foreach-parity tests
  (`test_adagrad_foreach_basic_parity`,
  `_parity_with_weight_decay`, `_parity_with_lr_decay`,
  `_parity_with_maximize`) confirm CPU and foreach paths agree to
  within 1e-5.
- [x] AC-9: `foreach_auto_routes_cuda_params_without_explicit_flag`
  (CUDA-gated, CL-1105 regression lock) confirms that with
  `foreach=false` but a CUDA-resident parameter, `step()`
  preserves CUDA residence.

## Architecture

### `AdagradConfig` (REQ-1)

The config is `#[derive(Debug, Clone)]` `#[non_exhaustive]` with
seven `pub` fields. `Default` matches PyTorch.

### `Adagrad<T>` struct (REQ-2, REQ-4)

Owns:

- `param_groups: Vec<ParamGroup<T>>`
- `config: AdagradConfig`
- `state: HashMap<(usize, usize), AdagradParamState<T>>` —
  per-parameter state keyed by `(group_idx, param_idx)`.

`AdagradParamState<T>` holds:

- `sum: Tensor<T>` — the accumulator, on the same device as the
  parameter once a step has run. Initialised on CPU at first state
  creation (`Tensor::from_storage` with `TensorStorage::cpu`),
  then moved to the parameter's device on the first foreach step
  if needed.
- `step_count: u64`.

### CPU `step` path (REQ-3, REQ-8)

1. Check `any_cuda`; if true OR `config.foreach`, dispatch to
   `step_foreach` (REQ-6).
2. Skip parameters with no gradient.
3. Read `grad.data_vec()` and `param.data_vec()` (CPU path; the
   `any_cuda` check above guarantees we never reach this for
   CUDA parameters).
4. Apply maximize.
5. Apply L2 weight decay: `g += wd * p`.
6. Initialise state if needed
   (`ensure_state(group_idx, param_idx, shape)`).
7. Increment `step_count`.
8. Compute `clr = lr / (1 + (step-1) * lr_decay)`.
9. Update accumulator: `sum[i] += g[i] * g[i]`.
10. Update parameter: `p[i] = p[i] - clr * g[i] / (sqrt(sum[i]) + eps)`.
11. Write `new_sum_data` back to `state.sum` via `update_data`
    (unsafe; SAFETY block).
12. Write `new_param_data` back to the parameter via `update_data`
    (unsafe; SAFETY block).

### Foreach `step_foreach` path (REQ-5)

Activated when `config.foreach == true` OR any param is CUDA
(REQ-6). Uses tensor ops on the parameter's device. The
accumulator is moved to the parameter's device on the first
step if necessary (the CPU-init path leaves it on CPU until
the first GPU step). State updates use `Entry::Vacant`-style
fallible init paths.

### State-dict (REQ-7)

`state_dict` serialises the accumulator to f64 via
`tensor_to_f64_vec` (`crate::optimizer::tensor_to_f64_vec`),
keyed by `"group{gi}_param{pi}"`. `load_state_dict` parses
the key by splitting on `_`, strips `"group"` / `"param"`
prefixes, and rebuilds the `Tensor` from the f64 vector via
`f64_vec_to_tensor`.

### CL-1105 auto-route (REQ-6)

```rust
let any_cuda = self.param_groups.iter()
    .any(|g| g.params.iter().any(|p| p.tensor().is_cuda()));
if self.config.foreach || any_cuda {
    return self.step_foreach();
}
```

This guard runs at the top of `step()` and prevents the legacy
`data_vec()` path from silently demoting CUDA-resident parameters
to CPU.

### Non-test production consumers

- `ferrotorch-optim/src/lib.rs:29` — `pub use adagrad::{Adagrad,
  AdagradConfig};` is the crate-public surface.
- `ferrotorch/src/lib.rs:61` — `pub use ferrotorch_optim::*;`
  exposes `Adagrad` and `AdagradConfig` through the umbrella
  crate's `optim` module.
- `ferrotorch-train/src/learner.rs` — `use
  ferrotorch_optim::Optimizer;` drives any `Optimizer<T>`
  implementor in the training loop (covers `Adagrad<f32>`).

## Parity contract

`parity_ops = []`. Edge-cases:

- **Accumulator initialisation**: `initial_accumulator_value`
  configurable (defaults to 0.0). PyTorch sets the same field
  (`torch/optim/adagrad.py:78`).
- **`lr_decay` step indexing**: the effective lr at step `t`
  (1-indexed) is `lr / (1 + (t-1) * lr_decay)`, mirroring
  PyTorch's `step_count` semantics
  (`torch/optim/adagrad.py:380-385`).
- **CUDA auto-route**: novel ferrotorch behaviour. PyTorch
  defaults `foreach=None` and decides per-tensor-list. Ferrotorch
  prefers the explicit `any_cuda` check because there is no
  per-tensor-list dispatch trampoline in our type system.

## Verification

Tests in `mod tests in adagrad.rs` (12 tests):

- Convergence: `test_adagrad_convergence`.
- Algorithm: `test_accumulator_grows_monotonically`,
  `test_lr_decay_reduces_effective_lr`, `test_weight_decay`.
- Trait surface: `test_zero_grad`, `test_skip_params_without_grad`.
- State-dict: `test_state_dict_roundtrip`.
- Config: `test_default_config`.
- Foreach parity: `test_adagrad_foreach_basic_parity`,
  `_parity_with_weight_decay`, `_parity_with_lr_decay`,
  `_parity_with_maximize`.
- CUDA auto-route (feature-gated):
  `foreach_auto_routes_cuda_params_without_explicit_flag`.

Smoke command:

```bash
cargo test -p ferrotorch-optim --lib adagrad:: 2>&1 | tail -3
```

Expected: `12 passed` (CUDA-gated test runs only with `--features cuda`).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct AdagradConfig` + `impl Default` in `adagrad.rs` mirroring `torch/optim/adagrad.py:29-187`; non-test consumer: `adagrad in ferrotorch-optim/src/lib.rs` re-exports `AdagradConfig`; `ferrotorch/src/lib.rs` re-exports the optim surface. |
| REQ-2 | SHIPPED | impl: `impl<T: Float> Optimizer<T> for Adagrad<T>` block in `adagrad.rs`; non-test consumer: `ferrotorch-train/src/learner.rs` consumes the `Optimizer` trait. |
| REQ-3 | SHIPPED | impl: legacy CPU `step` (else branch after `any_cuda` check) in `adagrad.rs` mirroring `_single_tensor_adagrad` in `torch/optim/adagrad.py:364-431`; non-test consumer: `ferrotorch/src/lib.rs:61` re-exports `Adagrad` so downstream training code can instantiate it. |
| REQ-4 | SHIPPED | impl: `struct AdagradParamState<T> { sum: Tensor<T>, step_count: u64 }` in `adagrad.rs`; non-test consumer: `ferrotorch-serialize/src/checkpoint.rs:48` reads/writes the resulting `OptimizerState` map which includes the `sum` Vec<f64>. |
| REQ-5 | SHIPPED | impl: `Adagrad::step_foreach` method in `adagrad.rs` mirroring `_multi_tensor_adagrad` in `torch/optim/adagrad.py:432-555`; non-test consumer: `ferrotorch/src/lib.rs:61` re-exports `AdagradConfig` so `with_foreach(true)` is reachable from downstream training code. |
| REQ-6 | SHIPPED | impl: `let any_cuda = ...; if self.config.foreach || any_cuda { return self.step_foreach(); }` at the top of `step()` in `adagrad.rs` (CL-1105); non-test consumer: `step in ferrotorch-train/src/learner.rs` propagates the GPU-residence preserving `step_foreach` choice via the `Optimizer::step` trait method. |
| REQ-7 | SHIPPED | impl: `state_dict` / `load_state_dict` methods in `adagrad.rs` keyed by `"group{gi}_param{pi}"`; non-test consumer: `ferrotorch-serialize/src/checkpoint.rs:48` `use ferrotorch_optim::OptimizerState;`. |
| REQ-8 | SHIPPED | impl: `let grad_tensor = match grad_opt { Some(g) => g, None => continue };` in both `step` and `step_foreach` in `adagrad.rs`; non-test consumer: training-loop callers in `ferrotorch-train/src/learner.rs` exercise this skip for frozen parameters. |
