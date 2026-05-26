# ferrotorch-optim — Adafactor (sublinear-memory adaptive optimizer)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/optim/_adafactor.py
-->

## Summary

`ferrotorch-optim/src/adafactor.rs` implements the Adafactor optimizer
(Shazeer & Stern, ICML 2018), mirroring `torch.optim.Adafactor` in
`torch/optim/_adafactor.py`. Adafactor uses factored second-moment
estimation for 2D+ parameters (storing row and column factors of
shape `[rows]` and `[cols]` instead of the full `[rows*cols]`
matrix), reducing memory from O(mn) to O(m+n). Optional relative
step sizing eliminates the need for an explicit learning rate. The
Rust impl exposes `AdafactorConfig` and the `Adafactor<T: Float>`
struct implementing the workspace-local `Optimizer<T>` trait. CUDA
support is currently CPU-only with an explicit error.

## Requirements

- REQ-1: `pub struct AdafactorConfig` carries
  `lr=None, beta1=None, decay_rate=-0.8, eps_sq=1e-30, eps_rms=1e-3,
  weight_decay=0.0, relative_step=true, warmup_init=false`,
  matching `torch.optim.Adafactor.__init__` defaults
  (`torch/optim/_adafactor.py:24-120`). Note `lr=None` triggers
  relative-step computation, and `beta1=None` disables first-moment
  tracking (saving more memory).
- REQ-2: `pub struct Adafactor<T: Float>` implements `Optimizer<T>`
  with all eight trait methods.
- REQ-3: For 2D+ parameters, the second moment is FACTORED:
  per-step `grad_sq = g^2 + eps_sq` is averaged into a row-factor
  (`[rows]`) and column-factor (`[cols]`) running mean. The
  reconstructed second-moment estimate is
  `v_est[r,c] = row_factor[r] * col_factor[c] / mean(row_factor)`,
  mirroring `_single_tensor_adafactor`
  (`torch/optim/_adafactor.py:330-450`).
- REQ-4: For 1D parameters (or when factoring is off), the second
  moment is stored as the full `full_sq: Vec<f64>`. Standard
  EMA: `full_sq = rho * full_sq + (1 - rho) * grad_sq`.
- REQ-5: Optional first moment (`beta1`): when `Some(beta1)`,
  maintain `exp_avg = beta1 * exp_avg + (1 - beta1) * g` and use
  it in the update; when `None`, skip the first-moment buffer
  entirely (memory savings).
- REQ-6: Relative step sizing: when `relative_step == true`, the
  effective step at step `t` is `min(1e-6 * t, 1 / sqrt(t)) *
  max(eps_rms, RMS(param))` (or `1 / sqrt(t)` without warmup),
  mirroring `_compute_lr` in PyTorch
  (`torch/optim/_adafactor.py:200-220`).
- REQ-7: Decoupled weight decay: `param *= (1 - lr * weight_decay)`
  before the gradient update, matching PyTorch's behaviour for
  Adafactor (`torch/optim/_adafactor.py:411-414`).
- REQ-8: CUDA explicit error: when `param.is_cuda()`, return
  `Err(FerrotorchError::NotImplementedOnCuda { op: "Adafactor" })`.
  No foreach path; no silent CPU demotion.
- REQ-9: Parameters whose `.grad()` is `None` are skipped.

## Acceptance Criteria

- [x] AC-1: `AdafactorConfig::default()` returns the PyTorch
  defaults (`lr=None, beta1=None, decay_rate=-0.8, eps_sq=1e-30,
  eps_rms=1e-3, weight_decay=0.0, relative_step=true,
  warmup_init=false`).
- [x] AC-2: `impl<T: Float> Optimizer<T> for Adafactor<T>` compiles.
- [x] AC-3: `test_adafactor_1d_param_decreases_loss` confirms a
  1D parameter is updated correctly using `full_sq`.
- [x] AC-4: `test_adafactor_2d_factored` confirms a 2D parameter
  with shape `[3, 4]` uses the factored `row_factor[3]` /
  `col_factor[4]` path.
- [x] AC-5: `test_adafactor_with_beta1` exercises the first-moment
  branch when `beta1 = Some(0.9)`.
- [x] AC-6: `test_adafactor_relative_step` confirms relative-step
  sizing produces a parameter update.
- [x] AC-7: `test_adafactor_zero_grad` confirms the trait method.

## Architecture

### `AdafactorConfig` (REQ-1)

The config is `#[derive(Debug, Clone, Copy)]` `#[non_exhaustive]`
with eight `pub` fields. The `lr: Option<f64>` and `beta1:
Option<f64>` are the most distinctive Rust-isms: `None` for `lr`
triggers relative-step sizing, `None` for `beta1` disables first
moment.

### `Adafactor<T>` struct (REQ-2)

Owns:

- `param_groups: Vec<ParamGroup<T>>`
- `config: AdafactorConfig`
- `state: HashMap<ParamKey, AdafactorState>` keyed by `ParamKey`
  via `Display`.
- CL-1125 workspaces: `grad_workspace`, `param_workspace`,
  `new_param_workspace`, `grad_sq_workspace` — reusable per-step
  buffers. For 7B-param Adafactor these would otherwise allocate
  ~60 GB of transient memory per step.

`AdafactorState` holds:

- `step_count: u64`
- `row_factor: Vec<f64>` (used for 2D+ params)
- `col_factor: Vec<f64>` (used for 2D+ params)
- `full_sq: Vec<f64>` (used for 1D params or non-factored mode)
- `exp_avg: Vec<f64>` (only populated when `beta1.is_some()`)
- `rms: f64` (RMS of parameter values, used for relative step)

### `step` (REQ-3, REQ-4, REQ-5, REQ-6, REQ-7, REQ-8, REQ-9)

1. Return `Err(NotImplementedOnCuda)` when `tensor.is_cuda()` (REQ-8).
2. Skip parameters with no gradient (REQ-9).
3. Determine `use_factored = shape.len() >= 2` and extract
   `rows`/`cols` from the last two dimensions of `shape`.
4. Fill CL-1125 workspaces.
5. Lazy-init state (`Entry::Vacant` pattern); compute initial RMS
   from the parameter values.
6. Increment `step_count`.
7. Compute `rho = min(1 - step^decay_rate, 1 - 1e-8)`.
8. Compute effective `lr`:
   - If `relative_step`: `rel = step^(-0.5).min(1e-6 * step)` (with
     warmup) or `step^(-0.5)` (without); `lr = rel *
     max(eps_rms, state.rms)`.
   - Else: `lr = config.lr.unwrap_or(group_lr)`.
9. Apply decoupled weight decay if `weight_decay != 0` (REQ-7):
   `new_param_workspace[i] *= (1 - lr * weight_decay)`.
10. Compute `grad_sq_workspace[i] = g^2 + eps_sq`.
11. If `use_factored` (REQ-3):
    - Update row factor: row mean over rows of `grad_sq`,
      `row_factor[r] = rho * row_factor[r] + (1 - rho) * mean`.
    - Update col factor: col mean over cols, similar update.
    - Reconstruct `v_est[r,c] = row[r] * col[c] / mean(row)`.
    - If `beta1.is_some()`: update `exp_avg[i]`, use it / sqrt(v_est).
    - Else: update is `g / (sqrt(v_est) + 1e-30)`.
    - `new_param_workspace[i] -= lr * update`.
12. Else (REQ-4): standard `full_sq = rho * full_sq + (1 - rho) *
    grad_sq[i]`, then similar update logic.
13. Update `state.rms` for next step's relative-step sizing.
14. `update_data` writes new values (unsafe; SAFETY block).

### `state_dict` / `load_state_dict`

CURRENT IMPLEMENTATION: `state_dict` returns
`Ok(OptimizerState::default())` (empty) and `load_state_dict`
returns `Ok(())` (no-op). Adafactor state serialisation is NOT
currently implemented — checkpoints round-trip the parameter
values but lose the factored second-moment state. Resuming from
a checkpoint cold-starts the optimizer state.

### Non-test production consumers

- `ferrotorch-optim/src/lib.rs:28` — `pub use adafactor::{Adafactor,
  AdafactorConfig};`
- `ferrotorch/src/lib.rs:61` — `pub use ferrotorch_optim::*;`
  re-exports `Adafactor` through the umbrella crate's `optim`
  module.
- `ferrotorch-train/src/learner.rs:28` — `use
  ferrotorch_optim::Optimizer;` consumes the trait; Adafactor is
  drop-in for any `Optimizer<T>` slot.

## Parity contract

`parity_ops = []`. Edge-cases:

- **No state serialisation**: `state_dict` returns an empty map;
  `load_state_dict` is a no-op. This DIVERGES from PyTorch's
  Adafactor, which serialises `exp_avg_sq_row`, `exp_avg_sq_col`,
  and `exp_avg_sq` via the parent `Optimizer.state_dict`. A
  future REQ should add full serialisation; this is documented
  here as known shortfall rather than masked as SHIPPED. The
  state_dict method exists (REQ-2's trait surface is complete),
  but the wire-format does not preserve factored moments. This
  matters for resuming long-running training jobs; for now,
  consumers cold-start Adafactor on resume.
- **Memory savings without `beta1`**: when `beta1 = None`, no
  `exp_avg` buffer is allocated. This is the canonical Adafactor
  memory-saving knob.
- **`decay_rate = -0.8`**: PyTorch uses negative exponents
  (`step^(-0.8)`), so `rho = 1 - step^decay_rate` for
  `decay_rate < 0` is a sensible bounded function.
- **2D factoring shape convention**: rows = `shape[-2]`,
  cols = `shape[-1]`. Higher-rank tensors are flattened to
  `[batch, rows, cols]` and the row/col factors are computed via
  the `batch * rows * cols + r * cols + c` indexer.

## Verification

Tests in `mod tests in adafactor.rs` (5 tests):

- 1D: `test_adafactor_1d_param_decreases_loss`.
- 2D factored: `test_adafactor_2d_factored`.
- First moment: `test_adafactor_with_beta1`.
- Relative step: `test_adafactor_relative_step`.
- Trait surface: `test_adafactor_zero_grad`.

Smoke command:

```bash
cargo test -p ferrotorch-optim --lib adafactor:: 2>&1 | tail -3
```

Expected: `5 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct AdafactorConfig` + `impl Default` in `adafactor.rs` mirroring `torch/optim/_adafactor.py:24-120`; non-test consumer: `ferrotorch-optim/src/lib.rs:28` re-exports `AdafactorConfig`; `ferrotorch/src/lib.rs:61` re-exports the optim surface. |
| REQ-2 | SHIPPED | impl: `impl<T: Float> Optimizer<T> for Adafactor<T>` block in `adafactor.rs`; non-test consumer: `ferrotorch-train/src/learner.rs:28` consumes the `Optimizer` trait. |
| REQ-3 | SHIPPED | impl: `use_factored` branch in `step` updating `row_factor` and `col_factor` then reconstructing `v_est = row * col / mean(row)` in `adafactor.rs` mirroring `_single_tensor_adafactor` in `torch/optim/_adafactor.py:330-450`; non-test consumer: `ferrotorch/src/lib.rs:61` re-exports `Adafactor` for downstream large-model training code (where factored memory savings are required). |
| REQ-4 | SHIPPED | impl: `else` branch of `use_factored` in `step` updating `full_sq = rho * full_sq + (1 - rho) * grad_sq` in `adafactor.rs`; non-test consumer: `ferrotorch/src/lib.rs:61` re-exports `Adafactor`. |
| REQ-5 | SHIPPED | impl: optional first-moment branch `if let Some(beta1) = config.beta1 { state.exp_avg[i] = beta1 * state.exp_avg[i] + (1 - beta1) * g; ... }` in `adafactor.rs`; non-test consumer: `ferrotorch/src/lib.rs:61` re-exports `AdafactorConfig::with_beta1(Some(0.9))`. |
| REQ-6 | SHIPPED | impl: `if config.relative_step { let rel = step.powf(-0.5)... lr = rel * rms_val; }` branch in `step` in `adafactor.rs` mirroring `torch/optim/_adafactor.py:200-220`; non-test consumer: `ferrotorch/src/lib.rs:61` re-exports `AdafactorConfig::with_relative_step(true)`. |
| REQ-7 | SHIPPED | impl: `if config.weight_decay != 0.0 { *slot = cast::<f64, T>(p * (1.0 - lr * config.weight_decay))?; }` branch in `adafactor.rs` mirroring `torch/optim/_adafactor.py:411-414`; non-test consumer: `ferrotorch/src/lib.rs:61` re-exports `AdafactorConfig::with_weight_decay`. |
| REQ-8 | SHIPPED | impl: `if tensor.is_cuda() { return Err(FerrotorchError::NotImplementedOnCuda { op: "Adafactor" }); }` in `step` in `adafactor.rs`; non-test consumer: `ferrotorch-train/src/learner.rs:28` Optimizer-trait callers receive the `Err` and surface it to the user via the training loop's `?` propagation. |
| REQ-9 | SHIPPED | impl: `match tensor.grad()? { Some(g) => g, None => continue };` in `step` in `adafactor.rs`; non-test consumer: `ferrotorch-train/src/learner.rs` exercises the skip for frozen layers. |
