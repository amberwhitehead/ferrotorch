# `RAdam` — Rectified Adam

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/optim/radam.py
-->

## Summary

`ferrotorch-optim/src/radam.rs` defines `RAdam<T>` and `RAdamConfig`,
mirroring `torch.optim.RAdam` (`torch/optim/radam.py:31`) — Liu et al.
"On the Variance of the Adaptive Learning Rate and Beyond" (ICLR 2020).
RAdam switches between an adaptive (Adam-style) update and a non-adaptive
(SGD-with-momentum-style) update based on whether the approximated
simple-moving-average length `rho_t` exceeds 5.

## Requirements

- REQ-1: `pub struct RAdamConfig` with `lr` (1e-3), `betas` ((0.9, 0.999)),
  `eps` (1e-8), `weight_decay` (0.0), `decoupled_weight_decay` (false),
  `foreach` (false). Mirrors `torch/optim/radam.py:31-65`.
- REQ-2: `pub struct RAdam<T: Float>` implementing `Optimizer<T>` with
  per-`ParamKey` state holding `step_count`, `exp_avg`, `exp_avg_sq`.
  Mirrors `torch/optim/radam.py:256-360`.
- REQ-3: rho computation — `rho_inf = 2/(1 - beta2) - 1`, then per-step
  `rho_t = rho_inf - 2 * t * beta2^t / (1 - beta2^t)`. Mirrors
  `_single_tensor_radam` (`torch/optim/radam.py:256-360`).
- REQ-4: Branch on `rho_t > 5`: adaptive variance-rectified path uses
  ```text
  rect = sqrt((rho_t - 4)(rho_t - 2) * rho_inf /
              ((rho_inf - 4)(rho_inf - 2) * rho_t))
  adaptive_lr = sqrt(1 - beta2^t) / (sqrt(exp_avg_sq) + eps)
  theta -= lr * m_hat * rect * adaptive_lr
  ```
  Else fall back to `theta -= lr * m_hat`. Mirrors upstream
  `torch/optim/radam.py:300-340`.
- REQ-5: L2 weight decay (`grad += wd * param`) when
  `decoupled_weight_decay == false`, OR decoupled-AdamW-style
  (`param *= (1 - lr * wd)`) when `decoupled_weight_decay == true`.
- REQ-6: `foreach: true` switches to `step_foreach` (on-device tensor-op
  path). Buffers live as `Tensor<T>` and the update composes via
  `add`/`mul`/`sqrt`/`sub`/`div` primitives. CL-497.
- REQ-7: Auto-route to `step_foreach` when ANY parameter is on CUDA
  (CL-1105 — prevents silent CPU↔GPU demote). `step()` short-circuits to
  `step_foreach()` whenever `config.foreach || any_cuda`.
- REQ-8: `state_dict`/`load_state_dict` round-trip via `ParamKey`
  wire format. Preserves `step_count`, `exp_avg`, `exp_avg_sq`.
- REQ-9: Builder-style `with_*` setters.

## Acceptance Criteria

- [x] AC-1: `RAdamConfig::default()` returns
  `{ lr: 1e-3, betas: (0.9, 0.999), eps: 1e-8, weight_decay: 0.0,
  decoupled_weight_decay: false, foreach: false }`.
- [x] AC-2: Quadratic convergence from `x = 5.0` reaches `|x| < 0.1`
  within 3000 steps (`test_radam_convergence_quadratic`).
- [x] AC-3: Two-parameter quadratic convergence
  (`test_radam_convergence_two_params`).
- [x] AC-4: Adaptive (rectification) path actually fires after enough
  steps to push `rho_t > 5` (`test_radam_rectification_kicks_in`).
- [x] AC-5: Weight decay shrinks the parameter
  (`test_radam_weight_decay`).
- [x] AC-6: `zero_grad()` clears gradients (`test_radam_zero_grad`).
- [x] AC-7: `state_dict` round-trip preserves `step_count` and
  `exp_avg` exactly (`test_radam_state_dict_roundtrip`).
- [x] AC-8: `lr`/`set_lr` accessors (`test_radam_lr_accessors`).
- [x] AC-9: `foreach: true` matches the legacy CPU path within `1e-4`
  on default and decoupled-wd configurations
  (`test_radam_foreach_basic_parity`, `_with_decoupled_wd`).

## Architecture

### Config

Same pattern as NAdam — `#[derive(Debug, Clone, Copy)]`,
`#[non_exhaustive]`, six public fields, builder-style `with_*`.

### Two step paths

The legacy CPU path (`step()` body, `radam.rs`) reads
`param.data_vec()` / `grad.data_vec()` into `Vec<f64>`, runs the
rectification algebra in `f64`, casts back to `T`. The SAFETY block at
`radam.rs` documents the sole-writer invariants for the final
`update_data` write.

The foreach path (`step_foreach`, `radam.rs`) keeps moment
buffers as `Tensor<T>` on the parameter's device and composes the
update via `add`/`mul`/`sqrt`/`sub`/`div`. Commit via
`into_storage_and_shape()` + `unsafe { param_t.update_storage(storage) }`
inside `no_grad`.

### rho_t branching

The decisive line at `radam.rs` computes
`rho_t = rho_inf - 2.0 * (step as f64) * beta2.powi(step) / bc2`. For
the default `beta2 = 0.999`, `rho_inf ≈ 1999`, so `rho_t > 5` after
roughly 5-6 steps — the adaptive path is the steady-state behavior.
The early SGD-style fallback exists to keep the very first few updates
stable (which is the entire point of RAdam).

### `foreach_utils::f64_scalar_on`

The foreach path constructs scalar tensors via `f64_scalar_on::<T>(v,
device)` (an alias for `scalar(cast::<f64, T>(v)?)?.to(device)?`)
which yields a `Tensor<T>` carrying the scalar on the parameter's
device. Used wherever the algebra needs to broadcast a constant.

### Non-test production consumers

`ferrotorch/src/lib.rs` `pub use ferrotorch_optim::*;` re-exports
`RAdam` and `RAdamConfig` as `ferrotorch::optim::{RAdam, RAdamConfig}`.

## Parity contract

`parity_ops = []`. RAdam's parity is asserted via the unit-test gauntlet
plus the cross-path foreach-parity tests.

Edge cases:

- **`step_count == 1` to ~5** with `beta2 = 0.999`: `rho_t` is
  small (close to 0), so the SGD-fallback branch (non-rectified) is
  taken. After ~5-6 steps the adaptive branch kicks in.
- **`beta2 < 0.5`** ⇒ `rho_inf < 3` ⇒ adaptive branch never fires;
  rectification reduces to SGD-with-bias-correction. Matches upstream.
- **`weight_decay > 0 && decoupled_weight_decay == true`** —
  `decay_factor = 1 - lr*wd`; applied as `decayed = param * decay_factor`
  before the gradient-derived update.
- **`foreach == false && any_cuda == true`** — auto-routes to
  `step_foreach` (CL-1105).
- **No gradient** — skip that parameter.

## Verification

Tests in `mod tests` of `radam.rs` (9 tests):

- `test_radam_convergence_quadratic`
- `test_radam_convergence_two_params`
- `test_radam_zero_grad`
- `test_radam_state_dict_roundtrip`
- `test_radam_lr_accessors`
- `test_radam_rectification_kicks_in`
- `test_radam_weight_decay`
- `test_radam_foreach_basic_parity` (CL-497)
- `test_radam_foreach_parity_with_decoupled_wd`

Smoke command:

```bash
cargo test -p ferrotorch-optim --lib radam:: 2>&1 | tail -3
```

Expected: `9 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct RAdamConfig` at `ferrotorch-optim/src/radam.rs` mirroring `torch/optim/radam.py:31`; non-test consumer: `ferrotorch/src/lib.rs` `pub use ferrotorch_optim::*;` re-export. |
| REQ-2 | SHIPPED | impl: `pub struct RAdam<T>` at `ferrotorch-optim/src/radam.rs` + `impl<T: Float> Optimizer<T>` at `ferrotorch-optim/src/radam.rs` mirroring `torch/optim/radam.py:31`; non-test consumer: `ferrotorch/src/lib.rs` re-export. |
| REQ-3 | SHIPPED | impl: `rho_inf` at `ferrotorch-optim/src/radam.rs` + per-step `rho_t` at `ferrotorch-optim/src/radam.rs` mirroring `_single_tensor_radam` (`torch/optim/radam.py:256-360`); non-test consumer: `ferrotorch/src/lib.rs` re-export. |
| REQ-4 | SHIPPED | impl: `rho_t > 5.0` adaptive branch at `ferrotorch-optim/src/radam.rs` + fallback at `ferrotorch-optim/src/radam.rs` mirroring `torch/optim/radam.py:300-340`; non-test consumer: `ferrotorch/src/lib.rs` re-export. |
| REQ-5 | SHIPPED | impl: decoupled vs L2 branches at `ferrotorch-optim/src/radam.rs`; non-test consumer: `ferrotorch/src/lib.rs` re-export. |
| REQ-6 | SHIPPED | impl: `step_foreach` at `ferrotorch-optim/src/radam.rs`; non-test consumer: `ferrotorch/src/lib.rs` re-export. |
| REQ-7 | SHIPPED | impl: auto-route at `ferrotorch-optim/src/radam.rs`; non-test consumer: `ferrotorch/src/lib.rs` re-export. Partial-parity divergence from upstream `foreach=None` semantics tracked by #1471. |
| REQ-8 | SHIPPED | impl: `state_dict` at `ferrotorch-optim/src/radam.rs` + `load_state_dict` at `ferrotorch-optim/src/radam.rs`; non-test consumer: `ferrotorch/src/lib.rs` re-export. |
| REQ-9 | SHIPPED | impl: `with_*` setters at `ferrotorch-optim/src/radam.rs`; non-test consumer: `ferrotorch/src/lib.rs` re-export. |
