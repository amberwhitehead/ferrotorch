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
- REQ-3: Sparse-COO gradient contract — `SparseAdam` REQUIRES a sparse
  gradient (a `ferrotorch_core::SparseGrad`, the analog of
  `p.grad.is_sparse == True`). A parameter with a DENSE `.grad` and no
  registered sparse grad is rejected with the verbatim upstream message
  `"SparseAdam does not support dense gradients, please consider Adam
  instead"` (`torch/optim/sparse_adam.py:88-92`). The masked update touches
  only the coalesced indices of the sparse grad; rows absent from the COO
  indices keep their moments at zero and their param byte-stable
  (`torch/optim/_functional.py:65-72`).
- REQ-4: `step_count` is per-`ParamKey`, NOT global; mirrors upstream's
  per-parameter state (`torch/optim/sparse_adam.py:98`).
- REQ-5: Bias-corrected sparse-Adam step (the TRUE masked formula, NOT dense
  Adam): `bc1 = 1 - beta1^t`, `bc2 = 1 - beta2^t`,
  `step_size = lr * sqrt(bc2) / bc1`,
  `param -= step_size * exp_avg / (sqrt(exp_avg_sq) + eps)`
  (`torch/optim/_functional.py:80-84`). Note eps enters the denominator
  differently from dense Adam's `m_hat/(sqrt(v_hat)+eps)`; the prior dense
  implementation used the Adam formula and was wrong for SparseAdam.
- REQ-6: `state_dict` / `load_state_dict` are no-op stubs that round-trip
  an empty `OptimizerState`. Upstream serializes the sparse Adam state;
  ferrotorch does not.
- REQ-7: CUDA parameters cause early-return `FerrotorchError::NotImplementedOnCuda
  { op: "SparseAdam" }`. Upstream PyTorch supports SparseAdam on CUDA via
  sparse-COO tensors.

## Acceptance Criteria

- [x] AC-1: `SparseAdamConfig::default()` returns
  `{ lr: 1e-3, betas: (0.9, 0.999), eps: 1e-8 }`.
- [x] AC-2: A sparse grad touching only some rows moves only those
  coordinates; rows absent from the COO indices are unchanged byte-for-byte.
  Pinned by `sparse_adam_leaves_untouched_rows_byte_stable` and
  `sparse_adam_skips_untouched_rows`.
- [x] AC-3: A multi-step constant sparse gradient drives the param along the
  torch trajectory. Pinned by `sparse_adam_matches_torch_oracle_multi_step`.
- [x] AC-4: `zero_grad()` sets every parameter's `.grad` to `None`.
- [x] AC-5: Calling `step()` with a CUDA parameter returns
  `FerrotorchError::NotImplementedOnCuda { op: "SparseAdam" }`.
- [x] AC-6: A parameter with a DENSE gradient and no registered sparse grad
  is rejected with the verbatim upstream message
  `"SparseAdam does not support dense gradients, please consider Adam instead"`
  (`torch/optim/sparse_adam.py:88-92`). Pinned by
  `sparse_adam_rejects_dense_grad_like_torch` (lib) and the live-torch oracle.
  Closes #1463.

## Architecture

### `SparseAdamConfig`

Three fields with builder-style `with_lr`/`with_betas`/`with_eps`
setters. The `#[non_exhaustive]` attribute matches the rest of
`ferrotorch-optim` for forward-compat with new kwargs landed by PyTorch
upstream.

### `SparseAdam<T>` and `SparseAdam::step`

The optimizer holds a `sparse_grads: HashMap<ParamKey, SparseGrad<T>>`
registry — the ferrotorch analog of `p.grad` being a sparse-COO tensor.
Callers register a parameter's sparse gradient via
`set_sparse_grad(group_idx, param_idx, grad)` (the boundary, like assigning
`p.grad = <sparse>`), or via the wired producer path
`collect_sparse_grad_from_embedding(emb, gi, pi)` which pulls
`Embedding::sparse_grad` and registers it.

`step()` iterates `(group_idx, param_idx)` pairs. For each:

1. Pop any registered `SparseGrad` for the `ParamKey` (CL-1122 typed key).
2. If none is registered but a DENSE `.grad` is set, REJECT with
   `"SparseAdam does not support dense gradients, please consider Adam
   instead"` (`torch/optim/sparse_adam.py:88-92`). If neither, skip (no grad).
3. Fail-fast on CUDA with `FerrotorchError::NotImplementedOnCuda` (#1468).
4. `sparse_step`: coalesce the grad (duplicate indices summed,
   `_functional.py:44`), increment the per-key step, then for each coalesced
   index `r` and slab element `j`: masked-EMA `exp_avg[r,j]` / `exp_avg_sq[r,j]`,
   compute `step_size = lr*sqrt(bc2)/bc1`, write
   `param[r,j] -= step_size * exp_avg[r,j] / (sqrt(exp_avg_sq[r,j]) + eps)`.
5. Commit via `unsafe { tensor.update_data(&data) }` (SAFETY block documents
   the sole-writer CPU-only invariant; mirrors torch's `@torch.no_grad()`).

Moment buffers and parameter elements OUTSIDE the coalesced indices are left
untouched (the sparse mask, `_functional.py:65-72`). Validated byte-for-byte
against live `torch.optim.SparseAdam` 2.11.0. Closes #1463.

### `state_dict` / `load_state_dict`

Both return `Ok(OptimizerState::default())` and ignore the input. This
makes round-trip mechanically green but loses every accumulated moment
estimate. Upstream serializes `step / exp_avg / exp_avg_sq` per parameter.

### Non-test production consumers

`ferrotorch/src/lib.rs` `pub use ferrotorch_optim::*;` (under the
`pub mod optim {…}` block) re-exports `SparseAdam` and `SparseAdamConfig`
as `ferrotorch::optim::{SparseAdam, SparseAdamConfig}`. This is the
public boundary every downstream training crate sees.

`ferrotorch-nn/src/embedding.rs`'s `Embedding::sparse_grad` is the sparse-grad
producer; `SparseAdam::collect_sparse_grad_from_embedding`
(`ferrotorch-optim/src/sparse_adam.rs`) is the wired non-test production
consumer — it calls `Embedding::sparse_grad`, registers the result via
`set_sparse_grad`, and `SparseAdam::step` applies the masked update. This is
the `nn.Embedding(sparse=True)` → `torch.optim.SparseAdam` flow
(`torch/nn/modules/sparse.py:34,48`).

## Parity contract

`parity_ops = []`. SparseAdam has no entry in `tools/parity-sweep/parity_audit.json`
because its sparse-COO contract requires a sparse gradient that the
parity-sweep oracle does not currently construct. The smoke gate falls
back to the lib-test suite for this file.

Edge-case behaviors the code owns:

- **Empty sparse grad (nnz == 0)** — step is a no-op for that parameter
  (`sparse_step` returns early before incrementing the step), matching
  torch's empty-grad skip (`torch/optim/_functional.py:47-49`).
- **Dense `.grad` with no registered sparse grad** — rejected with the
  verbatim upstream message (`torch/optim/sparse_adam.py:88-92`).
- **Index absent from the COO grad** — its moment buffers stay at zero and
  its param stays byte-stable (the sparse mask, `_functional.py:65-72`).
- **CUDA parameter** — `FerrotorchError::NotImplementedOnCuda` early-return
  (no silent demote).
- **NaN/Inf gradient value** — the moment buffers accumulate the NaN/Inf and
  propagate it forever after; ferrotorch makes no attempt to gate this
  (matches upstream).

## Verification

Tests in `mod tests` of `sparse_adam.rs` (6 tests), all grounded in a live
`torch.optim.SparseAdam` 2.11.0 oracle (R-CHAR-3):

- `sparse_adam_matches_torch_oracle_one_step_with_coalesce` — single step with
  a duplicate-index grad; byte-matches torch including the coalesce sum.
- `sparse_adam_matches_torch_oracle_multi_step` — 3 sequential steps match the
  torch trajectory.
- `sparse_adam_leaves_untouched_rows_byte_stable` — masked update touches only
  coalesced indices.
- `sparse_adam_rejects_dense_grad_like_torch` — dense grad → verbatim error.
- `sparse_adam_no_grad_is_noop` — no grad set → no-op.
- `sparse_adam_zero_grad_clears` — `zero_grad()` clears.

Conformance (`ferrotorch-optim/tests/conformance_optim_adam_family.rs`):
`sparse_adam_trajectory` drives the optimizer through a fully-materialised
sparse grad (the masked `step_size` formula, fixture validated byte-for-byte
vs live torch in `scripts/regenerate_optim_adam_fixtures.py`),
`sparse_adam_skips_untouched_rows`, and the end-to-end production chain
`sparse_adam_consumes_embedding_sparse_grad_end_to_end`
(`Embedding(sparse=true)` → `collect_sparse_grad_from_embedding` → `step`).

Smoke command:

```bash
cargo test -p ferrotorch-optim --lib sparse_adam:: 2>&1 | tail -3
```

Expected: `6 passed`. `parity_ops = []` per the parity contract below.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct SparseAdamConfig` in `ferrotorch-optim/src/sparse_adam.rs` mirroring `torch/optim/sparse_adam.py:14`; non-test consumer: `ferrotorch/src/lib.rs` `pub use ferrotorch_optim::*;` re-exports as `ferrotorch::optim::SparseAdamConfig`. |
| REQ-2 | SHIPPED | impl: `pub struct SparseAdam<T>` + `impl<T: Float> Optimizer<T>` in `ferrotorch-optim/src/sparse_adam.rs` mirroring `torch.optim.SparseAdam` (`torch/optim/sparse_adam.py:13`); non-test consumer: `ferrotorch/src/lib.rs` re-export plus the documented consumer chain at `ferrotorch-nn/src/embedding.rs`. |
| REQ-3 | SHIPPED | impl: `SparseAdam::step` requires a registered `ferrotorch_core::SparseGrad` and rejects a dense `.grad` with torch's verbatim message (`fn step` in `ferrotorch-optim/src/sparse_adam.rs`, mirroring `torch/optim/sparse_adam.py:88-92`); the masked step (`fn sparse_step`) mirrors `torch/optim/_functional.py:24-84`. Non-test production consumer: `SparseAdam::collect_sparse_grad_from_embedding` (`ferrotorch-optim/src/sparse_adam.rs`) consumes `Embedding::sparse_grad` (`ferrotorch-nn/src/embedding.rs`) and registers via `set_sparse_grad`, which `SparseAdam::step` reads. Closes #1463. |
| REQ-4 | SHIPPED | impl: per-`ParamKey` `state: HashMap<ParamKey, SparseAdamState>` in `ferrotorch-optim/src/sparse_adam.rs` with per-key `step_count` incremented in `fn sparse_step` mirroring `torch/optim/sparse_adam.py:98`; non-test consumer: `ferrotorch/src/lib.rs` re-export. |
| REQ-5 | SHIPPED | impl: bias-corrected sparse-Adam step (`step_size = lr*sqrt(bc2)/bc1`, `param -= step_size*numer/(sqrt(v)+eps)`) in `fn sparse_step` in `ferrotorch-optim/src/sparse_adam.rs` mirroring `torch/optim/_functional.py:80-84`; non-test consumer: `ferrotorch/src/lib.rs` re-export. |
| REQ-6 | NOT-STARTED | state_dict/load_state_dict are no-op stubs at `ferrotorch-optim/src/sparse_adam.rs`; serialisation of the masked moment buffers is unbuilt (no prereq blocker filed yet). |
| REQ-7 | SHIPPED | impl: `FerrotorchError::NotImplementedOnCuda { op: "SparseAdam" }` early-return in `fn step`/`fn sparse_step` at `ferrotorch-optim/src/sparse_adam.rs` (a documented intentional divergence from upstream, tracked by #1468); non-test consumer: `ferrotorch/src/lib.rs` re-export — every downstream caller transparently sees the error and routes around it. |
