# `NAdam` — Nesterov-accelerated Adam

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/optim/nadam.py
-->

## Summary

`ferrotorch-optim/src/nadam.rs` defines `NAdam<T>` and `NAdamConfig`
mirroring `torch.optim.NAdam` (`torch/optim/nadam.py:32`) — Dozat's
"Incorporating Nesterov Momentum into Adam" (ICLR 2016 Workshop). The
parameter update uses a lookahead first-moment estimate combining current
gradient with next-step momentum, weighted by a schedule-aware `mu_t`.

## Requirements

- REQ-1: `pub struct NAdamConfig` with `lr` (2e-3), `betas` ((0.9, 0.999)),
  `eps` (1e-8), `weight_decay` (0.0), `momentum_decay` (4e-3),
  `decoupled_weight_decay` (false), `foreach` (false). Mirrors
  `torch/optim/nadam.py:32-65`.
- REQ-2: `pub struct NAdam<T: Float>` implementing `Optimizer<T>` with
  per-`ParamKey` state holding `step_count`, `mu_product` (cumulative
  product of mu schedule), `exp_avg`, `exp_avg_sq`. Mirrors upstream's
  per-parameter state at `torch/optim/nadam.py:281-380`.
- REQ-3: mu schedule — `mu_t = beta1 * (1 - 0.5 * 0.96^(t * psi))` where
  `psi = momentum_decay`. Both `mu` and `mu_next` are computed each step,
  and `mu_product *= mu` tracks the cumulative product. Mirrors
  `_single_tensor_nadam` (`torch/optim/nadam.py:281-380`).
- REQ-4: Lookahead update formula:
  ```text
  m_hat = (1 - mu)     * g_t            / (1 - mu_product)
        + mu_next      * exp_avg        / (1 - mu_product_next)
  v_hat = exp_avg_sq / (1 - beta2^t)
  theta -= lr * m_hat / (sqrt(v_hat) + eps)
  ```
- REQ-5: L2 weight decay (`grad += wd * param`) when
  `decoupled_weight_decay == false`, OR decoupled-AdamW-style
  (`param *= (1 - lr * wd)`) when `decoupled_weight_decay == true`.
  Mirrors upstream's `decoupled_weight_decay=True/False` kwarg.
- REQ-6: `foreach: true` switches to `step_foreach` (on-device tensor-op
  path). Buffers live as `Tensor<T>` and the update composes via
  `add`/`mul`/`sqrt`/`sub`/`div` primitives. CL-497.
- REQ-7: Auto-route to `step_foreach` when ANY parameter is on CUDA
  (CL-1105 — prevents silent CPU↔GPU demote). `step()` short-circuits to
  `step_foreach()` whenever `config.foreach || any_cuda`.
- REQ-8: `state_dict`/`load_state_dict` round-trip the four state fields
  keyed by `ParamKey::to_string()` (the `"g{}_p{}"` wire format).
  Round-trip preserves `step_count`, `mu_product`, `exp_avg`,
  `exp_avg_sq`.
- REQ-9: Builder-style setters `with_lr` / `with_betas` / `with_eps`
  / `with_weight_decay` / `with_momentum_decay` /
  `with_decoupled_weight_decay` / `with_foreach`.

## Acceptance Criteria

- [x] AC-1: `NAdamConfig::default()` returns
  `{ lr: 2e-3, betas: (0.9, 0.999), eps: 1e-8, weight_decay: 0.0,
  momentum_decay: 4e-3, decoupled_weight_decay: false, foreach: false }`.
- [x] AC-2: Quadratic-loss convergence on `f(x) = x^2` from `x = 5.0`
  reaches `|x| < 0.1` within 3000 steps. Pinned by
  `test_nadam_convergence_quadratic`.
- [x] AC-3: Two-parameter convergence (`test_nadam_convergence_two_params`).
- [x] AC-4: `zero_grad()` clears every parameter's gradient
  (`test_nadam_zero_grad`).
- [x] AC-5: `state_dict`/`load_state_dict` round-trip preserves
  `step_count` and `mu_product` exactly. Pinned by
  `test_nadam_state_dict_roundtrip`.
- [x] AC-6: `lr`/`set_lr` accessors (`test_nadam_lr_accessors`).
- [x] AC-7: `foreach: true` matches the legacy CPU path within `1e-4`
  on default and decoupled-wd configurations
  (`test_nadam_foreach_basic_parity`, `_with_decoupled_wd`).
- [ ] AC-8: `foreach=None` (the upstream default for auto-dispatch on
  CUDA-with-matching-dtype) is not implemented — ferrotorch dispatches
  to foreach only when `any_cuda` is true, not on dtype match. Blocked
  by #1471.

## Architecture

### Config

`#[derive(Debug, Clone, Copy)]` with `#[non_exhaustive]` and seven
public fields. The `with_*` builders return `Self` for fluent
construction.

### Two step paths

Like `Adam` and `RAdam`, NAdam has two paths:

- Legacy CPU path (`step()` body when `config.foreach == false &&
  !any_cuda`): casts `Vec<f64>`, runs the lookahead-Nesterov algebra in
  scalar `f64`, casts back to `T`, then writes via
  `unsafe { param.tensor().update_data(&new_values) }` inside a
  `no_grad` closure. SAFETY block at `nadam.rs` documents
  sole-writer invariants.
- `step_foreach` (always runs when CUDA, or when `foreach=true`):
  buffers live as `Tensor<T>` on the parameter's device, composed via
  `add`/`mul`/`sqrt`/`sub`/`div`. Commit via `update_storage`.

### mu schedule and mu_product

Each step:

```rust
let mu       = beta1 * (1.0 - 0.5 * 0.96_f64.powf(step       * psi));
let mu_next  = beta1 * (1.0 - 0.5 * 0.96_f64.powf((step + 1) * psi));
state.mu_product       *= mu;
let mu_product_next     = state.mu_product * mu_next;
```

The two products feed the bias-correction denominators of the lookahead
update. Mirror of upstream's
`_single_tensor_nadam`'s mu schedule (`torch/optim/nadam.py:281-380`).

### Decoupled vs L2 weight decay

When `decoupled_weight_decay == true`: `decay_factor = 1 - lr * wd`,
then `decayed = param * decay_factor`, then the lookahead update
subtracts the gradient term from `decayed`.

When `decoupled_weight_decay == false`: `grad += wd * param` BEFORE the
moment updates. Both branches match the upstream kwarg.

### Non-test production consumers

`ferrotorch/src/lib.rs` `pub use ferrotorch_optim::*;` re-exports
`NAdam` and `NAdamConfig` as `ferrotorch::optim::{NAdam, NAdamConfig}`.

## Parity contract

`parity_ops = []`. NAdam's parity is asserted via the unit-test gauntlet
plus `test_nadam_foreach_*_parity` tests keeping the two step paths in
lock-step.

Edge cases:

- **`step_count == 1`** — `mu_product = 1.0 * mu_1` (initial mu_product
  is 1.0 in `or_insert_with`), so the denominator `1 - mu_product` is
  `1 - mu_1`.
- **`decoupled_weight_decay == true && weight_decay == 0`** —
  `decay_factor = 1.0` ⇒ short-circuits to the un-decayed path.
- **`momentum_decay = 0`** — `mu_t = beta1 * (1 - 0.5) = 0.5 * beta1`
  constant; the schedule degenerates but the update is still valid.
- **`foreach == false && any_cuda == true`** — auto-routes to
  `step_foreach` (CL-1105 — no silent CPU demote).
- **No gradient** — that parameter is skipped (no state init).

## Verification

Tests in `mod tests` of `nadam.rs` (8 tests):

- `test_nadam_convergence_quadratic`
- `test_nadam_convergence_two_params`
- `test_nadam_zero_grad`
- `test_nadam_state_dict_roundtrip`
- `test_nadam_lr_accessors`
- `test_nadam_default_config`
- `test_nadam_foreach_basic_parity` (CL-497)
- `test_nadam_foreach_parity_with_decoupled_wd`

Smoke command:

```bash
cargo test -p ferrotorch-optim --lib nadam:: 2>&1 | tail -3
```

Expected: `8 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct NAdamConfig` at `ferrotorch-optim/src/nadam.rs` mirroring `torch/optim/nadam.py:32`; non-test consumer: `ferrotorch/src/lib.rs` `pub use ferrotorch_optim::*;` re-export. |
| REQ-2 | SHIPPED | impl: `pub struct NAdam<T>` at `ferrotorch-optim/src/nadam.rs` + `impl<T: Float> Optimizer<T>` at `ferrotorch-optim/src/nadam.rs` with per-`ParamKey` state at `ferrotorch-optim/src/nadam.rs` mirroring `torch/optim/nadam.py:281`; non-test consumer: `ferrotorch/src/lib.rs` re-export. |
| REQ-3 | SHIPPED | impl: mu schedule at `ferrotorch-optim/src/nadam.rs` (legacy path) and `ferrotorch-optim/src/nadam.rs` (foreach path) mirroring `torch/optim/nadam.py:281-380`; non-test consumer: `ferrotorch/src/lib.rs` re-export. |
| REQ-4 | SHIPPED | impl: lookahead update at `ferrotorch-optim/src/nadam.rs` (legacy) + `ferrotorch-optim/src/nadam.rs` (foreach); non-test consumer: `ferrotorch/src/lib.rs` re-export. |
| REQ-5 | SHIPPED | impl: decoupled vs L2 branches at `ferrotorch-optim/src/nadam.rs` (legacy) and `ferrotorch-optim/src/nadam.rs` (foreach); non-test consumer: `ferrotorch/src/lib.rs` re-export. |
| REQ-6 | SHIPPED | impl: `step_foreach` at `ferrotorch-optim/src/nadam.rs`; non-test consumer: `ferrotorch/src/lib.rs` re-export. |
| REQ-7 | SHIPPED | impl: auto-route logic at `ferrotorch-optim/src/nadam.rs` (`any_cuda` check before the CPU body); non-test consumer: `ferrotorch/src/lib.rs` re-export. Partial-parity divergence from upstream `foreach=None` semantics tracked by #1471. |
| REQ-8 | SHIPPED | impl: `state_dict` at `ferrotorch-optim/src/nadam.rs` + `load_state_dict` at `ferrotorch-optim/src/nadam.rs` keyed by `ParamKey::to_string()`; non-test consumer: `ferrotorch/src/lib.rs` re-export. |
| REQ-9 | SHIPPED | impl: `with_*` setters at `ferrotorch-optim/src/nadam.rs`; non-test consumer: `ferrotorch/src/lib.rs` re-export. |
