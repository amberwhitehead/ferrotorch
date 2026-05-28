# ferrotorch-optim — Adadelta (adaptive learning rate without manual tuning)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/optim/adadelta.py
-->

## Summary

`ferrotorch-optim/src/adadelta.rs` implements the Adadelta optimizer
(Zeiler 2012), mirroring `torch.optim.Adadelta` in
`torch/optim/adadelta.py`. Adadelta adapts learning rates based on a
moving window of gradient updates, eliminating the need to set an
initial learning rate (the `lr` parameter defaults to `1.0`,
multiplying the auto-computed step). The Rust impl exposes
`AdadeltaConfig` and the `Adadelta<T: Float>` struct implementing
the workspace-local `Optimizer<T>` trait, with a legacy CPU f64 path
and a `step_foreach` on-device tensor-op path plus the CL-1105 CUDA
auto-route.

## Requirements

- REQ-1: `pub struct AdadeltaConfig` carries
  `lr=1.0, rho=0.9, eps=1e-6, weight_decay=0.0, foreach=false`,
  matching `torch.optim.Adadelta.__init__` defaults
  (`torch/optim/adadelta.py:29-126`). Note `lr=1.0` is the
  Adadelta-specific default.
- REQ-2: `pub struct Adadelta<T: Float>` implements `Optimizer<T>`
  with all eight trait methods.
- REQ-3: The legacy CPU `step` path mirrors `_single_tensor_adadelta`
  (`torch/optim/adadelta.py:245-303`):
  `square_avg = rho * square_avg + (1 - rho) * g^2`,
  `std = sqrt(square_avg + eps)`,
  `delta = sqrt(acc_delta + eps) / std * g`,
  `acc_delta = rho * acc_delta + (1 - rho) * delta^2`,
  `param = param - lr * delta`.
- REQ-4: The foreach path `step_foreach` keeps `square_avg` and
  `acc_delta` on the parameter's device via `Tensor<T>` storage,
  mirroring `_multi_tensor_adadelta`
  (`torch/optim/adadelta.py:305-410`).
- REQ-5: CL-1105 auto-route: `step()` checks `any_cuda` and
  dispatches to `step_foreach` when any parameter is CUDA-resident,
  even if `config.foreach == false`. Prevents silent CPU demotion.
- REQ-6: `state_dict` serialises per-parameter `step_count`,
  `square_avg`, `acc_delta`. Keys are
  `"g{group_idx}_p{param_idx}"` via `ParamKey::Display` (CL-1122).
- REQ-7: Parameters whose `.grad()` is `None` are skipped, mirroring
  PyTorch.

## Acceptance Criteria

- [x] AC-1: `AdadeltaConfig::default()` returns
  `lr=1.0, rho=0.9, eps=1e-6, weight_decay=0.0`.
- [x] AC-2: `impl<T: Float> Optimizer<T> for Adadelta<T>` compiles.
- [x] AC-3: `test_adadelta_convergence_quadratic` minimises `x^2`
  from 5.0 to within 0.5 of 0 in 3000 steps.
- [x] AC-4: `test_adadelta_convergence_two_params` minimises
  `x^2 + y^2` from `(3, -2)` to within 0.5 of `(0, 0)`.
- [x] AC-5: `test_adadelta_single_step_direction` confirms a
  positive gradient produces a parameter decrease.
- [x] AC-6: `test_adadelta_state_dict_roundtrip` round-trips
  `square_avg`, `acc_delta`, `step_count`.
- [x] AC-7: Foreach-parity tests
  (`test_adadelta_foreach_basic_parity`,
  `_parity_with_weight_decay`) confirm CPU and foreach paths
  agree to within 1e-4.

## Architecture

### `AdadeltaConfig` (REQ-1)

The config is `#[derive(Debug, Clone, Copy)]` `#[non_exhaustive]`
with five `pub` fields. `Default` matches PyTorch (note `lr=1.0`,
not the typical 1e-3).

### `Adadelta<T>` struct (REQ-2)

Owns:

- `param_groups: Vec<ParamGroup<T>>`
- `config: AdadeltaConfig`
- `state: HashMap<ParamKey, AdadeltaParamState>` (CPU path)
- `foreach_state: HashMap<ParamKey, AdadeltaForeachState<T>>` (foreach)

`AdadeltaParamState` holds `step_count: u64`, `square_avg: Vec<f64>`,
`acc_delta: Vec<f64>`. The CPU path computes in f64 and casts back
to T at the end.

`AdadeltaForeachState<T>` holds `step_count`, `square_avg: Tensor<T>`,
`acc_delta: Tensor<T>` on the parameter's device.

### Legacy CPU `step` (REQ-3, REQ-7)

1. CL-1105 auto-route check (REQ-5).
2. Skip parameters with no gradient.
3. Read `param` and `grad` as `Vec<f64>` (cast from T).
4. Apply L2 weight decay (`g += wd * p`) if enabled.
5. Lazy-init state if first step.
6. Per-element update loop:
   - `square_avg[i] = rho * square_avg[i] + (1 - rho) * g^2`
   - `std = sqrt(square_avg[i] + eps)`
   - `delta = sqrt(acc_delta[i] + eps) / std * g`
   - `acc_delta[i] = rho * acc_delta[i] + (1 - rho) * delta^2`
   - `new_param[i] = param[i] - lr * delta`
7. `update_data` writes new values (unsafe; SAFETY block).

### Foreach path (REQ-4)

Uses tensor ops to compute the same update entirely on the
parameter's device. `square_avg` and `acc_delta` start as
`zeros::<T>(param.shape())?.to(device)?` on first step (via
`Entry::Vacant` so the fallible allocations propagate `Err`).

### CL-1105 auto-route (REQ-5)

Same shape as Adagrad/Adamax/Asgd:

```rust
let any_cuda = self.param_groups.iter()
    .any(|g| g.params.iter().any(|p| p.tensor().is_cuda()));
if self.config.foreach || any_cuda {
    return self.step_foreach();
}
```

### State-dict (REQ-6)

`state_dict` renders each `ParamKey` via `Display` to
`"g{group_idx}_p{param_idx}"`. Entries contain `step_count`,
`square_avg`, `acc_delta`. `load_state_dict` parses keys via
`FromStr`.

### Non-test production consumers

- `ferrotorch-optim/src/lib.rs:27` — `pub use adadelta::{Adadelta,
  AdadeltaConfig};` is the crate-public surface.
- `ferrotorch/src/lib.rs:61` — `pub use ferrotorch_optim::*;`
  re-exports `Adadelta` and `AdadeltaConfig` through the umbrella
  crate's `optim` module.
- `ferrotorch-train/src/learner.rs` — `use
  ferrotorch_optim::Optimizer;` drives `Adadelta::step` via the
  trait.

## Parity contract

`parity_ops = []`. Edge-cases:

- **Both running averages start at zero**: `square_avg` and
  `acc_delta` both initialised to `vec![0.0; numel]`. The first
  step therefore produces `delta = sqrt(0 + eps) / sqrt(0 + (1-rho)*g^2 + eps) * g ≈ sqrt(eps) * g / |g|*sqrt(1-rho)`,
  which is small. Matches PyTorch (`torch/optim/adadelta.py:268-275`).
- **`weight_decay`** is L2 (added to gradient), not decoupled.
- **`rho` decay rate**: shared between `square_avg` and `acc_delta`.

## Verification

Tests in `mod tests in adadelta.rs` (8 tests):

- Convergence: `test_adadelta_convergence_quadratic`,
  `test_adadelta_convergence_two_params`.
- Direction sanity: `test_adadelta_single_step_direction`.
- Trait: `test_adadelta_zero_grad`, `test_adadelta_lr_accessors`.
- Config: `test_adadelta_default_config`.
- State-dict: `test_adadelta_state_dict_roundtrip`.
- Foreach parity: `test_adadelta_foreach_basic_parity`,
  `test_adadelta_foreach_parity_with_weight_decay`.

Smoke command:

```bash
cargo test -p ferrotorch-optim --lib adadelta:: 2>&1 | tail -3
```

Expected: `9 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct AdadeltaConfig` + `impl Default { lr: 1.0, rho: 0.9, eps: 1e-6, ... }` in `adadelta.rs` mirroring `torch/optim/adadelta.py:29-126`; non-test consumer: `adadelta in ferrotorch-optim/src/lib.rs` re-exports `AdadeltaConfig`; `ferrotorch/src/lib.rs` re-exports the optim surface. |
| REQ-2 | SHIPPED | impl: `impl<T: Float> Optimizer<T> for Adadelta<T>` block in `adadelta.rs`; non-test consumer: `ferrotorch-train/src/learner.rs` consumes the `Optimizer` trait. |
| REQ-3 | SHIPPED | impl: legacy CPU `step` (else branch after `any_cuda` check) in `adadelta.rs` mirroring `_single_tensor_adadelta` in `torch/optim/adadelta.py:245-303`; non-test consumer: `ferrotorch/src/lib.rs:61` re-exports `Adadelta` for downstream training code. |
| REQ-4 | SHIPPED | impl: `Adadelta::step_foreach` method in `adadelta.rs` mirroring `_multi_tensor_adadelta` in `torch/optim/adadelta.py:305-410`; non-test consumer: `ferrotorch/src/lib.rs:61` re-exports `AdadeltaConfig::with_foreach(true)`. |
| REQ-5 | SHIPPED | impl: `let any_cuda = ...; if self.config.foreach || any_cuda { return self.step_foreach(); }` at the top of `step()` in `adadelta.rs` (CL-1105); non-test consumer: `step in ferrotorch-train/src/learner.rs` propagates the GPU-residence preserving choice via the `Optimizer::step` trait. |
| REQ-6 | SHIPPED | impl: `state_dict` / `load_state_dict` methods in `adadelta.rs` keyed by `ParamKey::Display`; non-test consumer: `ferrotorch-serialize/src/checkpoint.rs:48` `use ferrotorch_optim::OptimizerState;`. |
| REQ-7 | SHIPPED | impl: `match grad_opt { Some(g) => g, None => continue };` in both `step` and `step_foreach` in `adadelta.rs`; non-test consumer: training-loop path in `ferrotorch-train/src/learner.rs` exercises this skip. |
