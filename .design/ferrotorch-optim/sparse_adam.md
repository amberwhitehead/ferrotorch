# `SparseAdam` — Adam variant for sparse gradients

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/optim/sparse_adam.py
-->

## Summary

`ferrotorch-optim/src/sparse_adam.rs` defines `SparseAdam<T>` and
`SparseAdamConfig`, an Adam variant that only updates first/second moment
estimates at indices where the gradient is non-zero. Targets large embedding
tables where only a few rows are touched per batch. Mirrors the public
construction surface of `torch.optim.SparseAdam` (`torch/optim/sparse_adam.py:13`).

## Requirements

- REQ-1: `pub struct SparseAdamConfig` with fields `lr`, `betas`, `eps`
  (no `weight_decay`, no `amsgrad` — matches upstream's deliberate omission;
  `torch/optim/sparse_adam.py:14`). Defaults `lr=1e-3`, `betas=(0.9, 0.999)`,
  `eps=1e-8`.
- REQ-2: `pub struct SparseAdam<T: Float>` with `new(params, config)` and
  `impl Optimizer<T>`; tracks per-`ParamKey` `step_count` + `exp_avg` +
  `exp_avg_sq`. Mirrors upstream's per-parameter `state["step"] / state["exp_avg"]
  / state["exp_avg_sq"]` (`torch/optim/sparse_adam.py:96-104`).
- REQ-3: Sparse-skip semantics — at indices where `g == 0`, the moment
  buffers are NOT updated and the parameter is NOT touched. Indices where
  `g != 0` get the full Adam update with per-key bias correction.
  Upstream contract uses a true sparse-COO gradient (`p.grad.is_sparse`,
  `torch/optim/sparse_adam.py:88-92`); ferrotorch reads dense and checks
  per-element. Behavioral parity for the sparse-coordinate case but the
  type-level contract differs.
- REQ-4: `step_count` is per-`ParamKey`, NOT global; mirrors upstream's
  per-parameter state (`torch/optim/sparse_adam.py:98`).
- REQ-5: Bias correction matches Adam:
  `bc1 = 1 - beta1^t`, `bc2 = 1 - beta2^t`,
  `m_hat = exp_avg / bc1`, `v_hat = exp_avg_sq / bc2`,
  `param -= lr * m_hat / (sqrt(v_hat) + eps)`.
- REQ-6: `state_dict` / `load_state_dict` are no-op stubs that round-trip
  an empty `OptimizerState`. Upstream serializes the sparse Adam state;
  ferrotorch does not.
- REQ-7: CUDA parameters cause early-return `FerrotorchError::NotImplementedOnCuda
  { op: "SparseAdam" }`. Upstream PyTorch supports SparseAdam on CUDA via
  sparse-COO tensors.

## Acceptance Criteria

- [x] AC-1: `SparseAdamConfig::default()` returns
  `{ lr: 1e-3, betas: (0.9, 0.999), eps: 1e-8 }`.
- [x] AC-2: A non-zero gradient at one index and zero gradients elsewhere
  moves only the targeted parameter coordinate; other coordinates are
  unchanged byte-for-byte. Pinned by `test_sparse_adam_skips_zero_gradients`.
- [x] AC-3: After 10 steps of a constant positive scalar gradient, the
  parameter strictly decreases. Pinned by `test_sparse_adam_multiple_steps`.
- [x] AC-4: `zero_grad()` sets every parameter's `.grad` to `None`.
- [x] AC-5: Calling `step()` with a CUDA parameter returns
  `FerrotorchError::NotImplementedOnCuda { op: "SparseAdam" }`.
- [ ] AC-6: Dense parameter with a SparseCoo-formatted gradient is
  rejected with a clear error message mirroring upstream's
  `"SparseAdam does not support dense gradients, please consider Adam instead"`
  (inverted in upstream — it checks for `is_sparse`). Blocked by #1463.

## Architecture

### `SparseAdamConfig`

Three fields with builder-style `with_lr`/`with_betas`/`with_eps`
setters. The `#[non_exhaustive]` attribute matches the rest of
`ferrotorch-optim` for forward-compat with new kwargs landed by PyTorch
upstream.

### `SparseAdam<T>` and `SparseAdam::step`

`step()` iterates `(group_idx, param_idx)` pairs. For each parameter with
`grad.is_some()`, it builds a `ParamKey` (CL-1122 typed key — replaces
the per-step `format!("g{}_p{}")` heap allocation), then:

1. Fail-fast on CUDA with `FerrotorchError::NotImplementedOnCuda`.
2. Reuse the `param_workspace: Vec<T>` and `grad_workspace: Vec<T>`
   buffers (CL-1125 — keeps per-step heap traffic at zero).
3. For each element index `i`:
   - If `g_i == 0`, skip (sparse-coordinate path).
   - Else update `exp_avg[i]`, `exp_avg_sq[i]`, compute bias-corrected
     moments, then write `param[i] -= lr * m_hat / (sqrt(v_hat) + eps)`.
4. Commit via `unsafe { tensor_handle.update_data(&self.param_workspace) }`
   inside a `no_grad` closure (SAFETY block documents the four sole-writer
   invariants).

The dense-input + element-wise zero-skip is the divergence from upstream.
Behaviourally on a SparseCoo gradient materialized as dense, the results
match index-by-index, but the type-level contract (upstream rejects dense
gradients) is missing. Tracking blocker #1463.

### `state_dict` / `load_state_dict`

Both return `Ok(OptimizerState::default())` and ignore the input. This
makes round-trip mechanically green but loses every accumulated moment
estimate. Upstream serializes `step / exp_avg / exp_avg_sq` per parameter.

### Non-test production consumers

`ferrotorch/src/lib.rs` `pub use ferrotorch_optim::*;` (under the
`pub mod optim {…}` block) re-exports `SparseAdam` and `SparseAdamConfig`
as `ferrotorch::optim::{SparseAdam, SparseAdamConfig}`. This is the
public boundary every downstream training crate sees.

`ferrotorch-nn/src/embedding.rs` documents `SparseAdam` as the canonical
consumer of `Embedding::sparse_grad` (`embedding.rs`),
binding the embedding sparse-grad path to this optimizer.

## Parity contract

`parity_ops = []`. SparseAdam has no entry in `tools/parity-sweep/parity_audit.json`
because its sparse-COO contract requires a sparse gradient that the
parity-sweep oracle does not currently construct. The smoke gate falls
back to the lib-test suite for this file.

Edge-case behaviors the code owns:

- **Zero gradient at all indices** — step is a no-op for that parameter
  (every per-index branch hits `continue`).
- **CUDA parameter** — `FerrotorchError::NotImplementedOnCuda` early-return
  (no silent demote).
- **Empty tensor (numel == 0)** — the per-index loop does not execute;
  `update_data` writes an empty slice.
- **NaN/Inf gradient at index `i`** — the moment buffers accumulate
  the NaN/Inf and propagate it forever after; ferrotorch makes no
  attempt to gate this (matches upstream).

## Verification

Tests in `mod tests` of `sparse_adam.rs` (4 tests):

- `test_sparse_adam_skips_zero_gradients` — zero gradient indices stay byte-stable.
- `test_sparse_adam_dense_matches_direction` — positive grad shrinks, negative grad grows.
- `test_sparse_adam_multiple_steps` — 10 steps of constant positive gradient decreases param.
- `test_sparse_adam_zero_grad` — `zero_grad()` clears.

Smoke command:

```bash
cargo test -p ferrotorch-optim --lib sparse_adam:: 2>&1 | tail -3
```

Expected: `4 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct SparseAdamConfig` in `ferrotorch-optim/src/sparse_adam.rs` mirroring `torch/optim/sparse_adam.py:14`; non-test consumer: `ferrotorch/src/lib.rs` `pub use ferrotorch_optim::*;` re-exports as `ferrotorch::optim::SparseAdamConfig`. |
| REQ-2 | SHIPPED | impl: `pub struct SparseAdam<T>` + `impl<T: Float> Optimizer<T>` in `ferrotorch-optim/src/sparse_adam.rs` mirroring `torch.optim.SparseAdam` (`torch/optim/sparse_adam.py:13`); non-test consumer: `ferrotorch/src/lib.rs` re-export plus the documented consumer chain at `ferrotorch-nn/src/embedding.rs`. |
| REQ-3 | NOT-STARTED | sparse-COO gradient contract missing; blocked on #1463. Current behavior matches per-index but the type-level `is_sparse` check is absent. |
| REQ-4 | SHIPPED | impl: per-`ParamKey` `state: HashMap<ParamKey, SparseAdamState>` in `ferrotorch-optim/src/sparse_adam.rs` with per-key `step_count` updated at `sparse_adam.rs` mirroring `torch/optim/sparse_adam.py:98`; non-test consumer: `ferrotorch/src/lib.rs` re-export. |
| REQ-5 | SHIPPED | impl: bias-corrected Adam update at `ferrotorch-optim/src/sparse_adam.rs` mirroring the Adam update in upstream's `_functional.py` invoked from `torch/optim/sparse_adam.py:126-138`; non-test consumer: `ferrotorch/src/lib.rs` re-export. |
| REQ-6 | NOT-STARTED | state_dict/load_state_dict are no-op stubs at `ferrotorch-optim/src/sparse_adam.rs`; blocked on follow-up not yet filed (state-dict for SparseAdam needs the sparse-COO type from #1463 to land first). |
| REQ-7 | SHIPPED | impl: `FerrotorchError::NotImplementedOnCuda { op: "SparseAdam" }` early-return at `ferrotorch-optim/src/sparse_adam.rs` (a documented intentional divergence from upstream, tracked by #1468); non-test consumer: `ferrotorch/src/lib.rs` re-export — every downstream caller transparently sees the error and routes around it. |
