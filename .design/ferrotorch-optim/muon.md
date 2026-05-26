# `Muon` — spectral-norm-aware SGD with Newton-Schulz orthogonalization

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/optim/_muon.py
-->

## Summary

`ferrotorch-optim/src/muon.rs` defines `Muon<T>` and `MuonConfig`,
mirroring `torch.optim.Muon` (`torch/optim/_muon.py:87`) — spectral-norm-aware
SGD with Newton-Schulz orthogonalization applied to 2-D weight matrices.
Reference: <https://arxiv.org/abs/2502.16982>.

## Requirements

- REQ-1: `pub struct MuonConfig` with `lr` (default `0.02`),
  `momentum` (`0.95`), `nesterov` (`true`), `ns_steps` (`5`),
  `weight_decay` (`0.0`), `maximize` (`false`). The upstream defaults are
  `lr=1e-3`, `weight_decay=0.1`, otherwise matching
  (`torch/optim/_muon.py:88-99`). The ferrotorch defaults are documented
  divergences for the demo-friendly LR; both should converge on the
  test surfaces.
- REQ-2: `pub struct Muon<T: Float>` with `new(params, config)` and
  `impl Optimizer<T>` providing `step` / `zero_grad` / `lr` / `set_lr` /
  `param_groups` / `param_groups_mut` / `add_param_group` / `state_dict`
  / `load_state_dict`.
- REQ-3: Newton-Schulz orthogonalization for 2-D parameters — given
  `G`, normalize via Frobenius norm then iterate
  `G = G @ (3*I - G^T @ G) / 2` for `ns_steps` iterations. The ferrotorch
  iteration uses the **cubic** Newton-Schulz `(3, -1, 0.5)`; the upstream
  uses a **quintic** with `ns_coefficients=(3.4445, -4.7750, 2.0315)`
  by default. Divergence tracked by #1465.
- REQ-4: For non-2D parameters, ferrotorch falls back to standard
  momentum SGD without orthogonalization. Upstream
  REJECTS non-2D parameters with `ValueError` at construction
  (`torch/optim/_muon.py:130-133`). Divergence tracked by #1464.
- REQ-5: Momentum buffer is stored as `Tensor<T>` on the parameter's
  device (CL-1105 Pattern B). On step `n`:
  - If `n == 0`: `buf = processed_grad.clone()` (initial state).
  - Else: `buf = momentum * buf + processed_grad`.
- REQ-6: When `nesterov == true`, the effective gradient is
  `processed_grad + momentum * buf` (lookahead); when `nesterov ==
  false`, it's `buf` alone.
- REQ-7: Weight decay uses **L2** style (`grad += wd * param`) BEFORE
  the orthogonalization. Upstream uses **decoupled** style (`theta_t -=
  lr * wd * theta_t` applied separately from the orthogonalized
  gradient term). Divergence tracked by #1466.
- REQ-8: `maximize: true` negates the gradient at the top of the step
  body (`grad = -grad_tensor`).
- REQ-9: CL-1105 Pattern B device residence — `Muon::step` keeps
  parameter tensors on their original device throughout. Newton-Schulz
  uses `tensor_matmul` (cuBLAS GEMM on CUDA) plus the device-aware
  `add`/`mul`/`sub`/`neg`. The pinning CUDA tests
  `muon_step_preserves_device_for_cuda_input` and
  `muon_step_matches_cpu_within_tolerance` are conditioned on
  `--features cuda`.
- REQ-10: `state_dict`/`load_state_dict` serialize momentum buffers
  (downloaded to CPU and cast to `f64`) plus per-key step counts,
  keyed by the string `"{group_idx}_{param_idx}"`.

## Acceptance Criteria

- [x] AC-1: `MuonConfig::default()` returns `MuonConfig::new(0.02)`
  with `momentum=0.95, nesterov=true, ns_steps=5, weight_decay=0.0,
  maximize=false`.
- [x] AC-2: Newton-Schulz on a non-orthogonal 2x2 produces a
  near-orthogonal matrix: `orth^T @ orth ≈ I` within `1e-4`. Pinned by
  `test_newton_schulz_produces_orthogonal`.
- [x] AC-3: NS on a zero matrix remains zero. Pinned by
  `test_newton_schulz_zero_grad`.
- [x] AC-4: 1-D parameter with momentum-0 + nesterov-off behaves like
  vanilla SGD (`p = 10 - 0.1 * 1.0 = 9.9`). Pinned by
  `test_muon_basic_step_1d`.
- [x] AC-5: 2-D parameter actually moves after a step
  (`test_muon_basic_step_2d`).
- [x] AC-6: Quadratic-norm convergence with nesterov-momentum reduces
  `||x||^2 < 0.01` within 200 steps (`test_muon_convergence_quadratic`).
- [x] AC-7: 2-D quadratic-norm convergence reduces `||W||_F^2 < 0.1`
  within 300 steps (`test_muon_convergence_2d_quadratic`).
- [x] AC-8: `lr()`/`set_lr()` propagate through param_groups
  (`test_muon_lr_get_set`).
- [x] AC-9: `state_dict` round-trip preserves momentum buffer numel
  (`test_muon_state_dict_roundtrip`).
- [x] AC-10: `zero_grad()` clears gradients (`test_muon_zero_grad`).
- [x] AC-11 (CUDA): Muon step preserves CUDA residence
  (`muon_step_preserves_device_for_cuda_input`, `--features cuda`).
- [x] AC-12 (CUDA): CUDA step matches CPU within `1e-6` for f64
  (`muon_step_matches_cpu_within_tolerance`, `--features cuda`).
- [ ] AC-13: Quintic NS coefficients `(3.4445, -4.7750, 2.0315)`
  configurable via `MuonConfig::ns_coefficients`. Blocked by #1465.
- [ ] AC-14: Strict non-2D parameter rejection at `Muon::new` matching
  upstream's `ValueError`. Blocked by #1464.
- [ ] AC-15: Decoupled weight decay path matching upstream
  (theta_t -= lr * wd * theta_t before NS-update apply). Blocked by
  #1466.
- [ ] AC-16: `adjust_lr_fn` ("original" / "match_rms_adamw") for
  RMS-aware LR scaling. Blocked by #1466.

## Architecture

### Newton-Schulz iteration

`newton_schulz_orthogonalize_tensor` (`muon.rs`) keeps the
matrix on the parameter's device throughout:

1. Compute `||G||_F = sqrt(sum(G * G))` as a scalar tensor.
2. Normalize: `g = G / (||G||_F + 1e-30)` (the epsilon guards the zero
   case).
3. Construct `I` (cols x cols) on the device via
   `creation::eye::<T>(cols)?.to(device)?`.
4. For `ns_steps` iterations: `G_{k+1} = G_k @ (3*I - G_k^T @ G_k) / 2`
   composed via `tensor_matmul` (cuBLAS GEMM on CUDA) + `add`/`mul`/`sub`.

The ferrotorch iteration is **cubic** with coefficients `(3, -1, 0.5)`.
Upstream's `_zeropower_via_newtonschulz` defaults to the **quintic**
`(3.4445, -4.7750, 2.0315)` which converges in fewer steps but is
mathematically distinct. Tracked by #1465.

A CPU-only reference `newton_schulz_orthogonalize` (`muon.rs`)
is kept under `#[cfg(test)]` for the `test_newton_schulz_*` unit tests
that verify the orthogonalization property.

### Per-step body

For each `(gi, pi)`:

1. Clone `param.tensor()` to release the borrow.
2. Skip parameters without gradients.
3. Inside `no_grad`:
   - Negate grad if `maximize`.
   - Apply L2 weight decay: `grad += wd * param`.
   - 2-D parameter ⇒ `processed_grad = newton_schulz_orthogonalize_tensor(grad, ns_steps)`;
     else `processed_grad = grad`.
   - Apply momentum: `step == 0` init copies, else
     `buf = momentum * buf + processed_grad`.
   - Apply nesterov: `effective_grad = processed_grad + momentum * buf`
     if `nesterov`, else `buf.clone()`.
   - `new_param = param - lr * effective_grad`.
   - Commit via `into_storage_and_shape()` + `unsafe { param_t.update_storage(storage) }`.

The SAFETY block at `muon.rs` documents the four sole-writer
invariants.

### Non-test production consumers

`ferrotorch/src/lib.rs` `pub use ferrotorch_optim::*;` re-exports
`Muon` and `MuonConfig` as `ferrotorch::optim::{Muon, MuonConfig}`.

## Parity contract

`parity_ops = []`. Muon's parity is asserted via the unit-test gauntlet
plus the CUDA-conditional tests gated on `--features cuda`. End-to-end
parity with upstream's quintic Newton-Schulz is intentionally NOT
asserted: ferrotorch ships the cubic variant and the parity divergence
is tracked by #1465.

Edge cases the code owns:

- **Zero gradient (Frobenius norm = 0)** — `eps_t = 1e-30` keeps the
  on-device division finite; NS iteration produces a zero output via the
  algorithmic fixed point.
- **Non-2D parameter** — falls back to vanilla momentum SGD without NS
  (divergence vs. upstream; #1464).
- **First step (step == 0)** — momentum buffer is initialized to the
  processed gradient, NOT zero, matching upstream's `_init_group`.
- **`momentum == 0`** — `effective_grad = processed_grad` (the entire
  momentum branch is skipped, no buffer created).
- **`maximize == true`** — `grad = -grad_tensor` before everything else.

## Verification

Tests in `mod tests` of `muon.rs`:

- `test_newton_schulz_produces_orthogonal`
- `test_newton_schulz_zero_grad`
- `test_muon_basic_step_1d`
- `test_muon_basic_step_2d`
- `test_muon_convergence_quadratic`
- `test_muon_convergence_2d_quadratic`
- `test_muon_lr_get_set`
- `test_muon_state_dict_roundtrip`
- `test_muon_zero_grad`
- `muon_step_preserves_device_for_cuda_input` (CUDA-conditional)
- `muon_step_matches_cpu_within_tolerance` (CUDA-conditional)

Smoke command:

```bash
cargo test -p ferrotorch-optim --lib muon:: 2>&1 | tail -3
```

Expected: `9 passed` without `--features cuda`, `11 passed` with.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct MuonConfig` at `ferrotorch-optim/src/muon.rs` mirroring `torch/optim/_muon.py:87`; non-test consumer: `ferrotorch/src/lib.rs` `pub use ferrotorch_optim::*;` re-export. (Lr default `0.02` vs upstream `1e-3` is a deliberate ferrotorch divergence.) |
| REQ-2 | SHIPPED | impl: `pub struct Muon<T>` at `ferrotorch-optim/src/muon.rs` + `impl<T: Float> Optimizer<T>` at `ferrotorch-optim/src/muon.rs`; non-test consumer: `ferrotorch/src/lib.rs` re-export. |
| REQ-3 | SHIPPED | impl: `newton_schulz_orthogonalize_tensor` at `ferrotorch-optim/src/muon.rs` mirroring upstream `_zeropower_via_newtonschulz` (`torch/optim/_muon.py:31`) structurally but with **cubic** coefficients vs upstream's quintic default. Parity divergence on coefficients tracked by #1465. Non-test consumer: `ferrotorch/src/lib.rs` re-export. |
| REQ-4 | NOT-STARTED | ferrotorch falls back to momentum-SGD for non-2D; upstream rejects with `ValueError`. Blocked by #1464. |
| REQ-5 | SHIPPED | impl: device-resident momentum buffer at `ferrotorch-optim/src/muon.rs` (`HashMap<String, Tensor<T>>`) + per-step update at `ferrotorch-optim/src/muon.rs`; non-test consumer: `ferrotorch/src/lib.rs` re-export. |
| REQ-6 | SHIPPED | impl: nesterov branch at `ferrotorch-optim/src/muon.rs` mirroring upstream's `nesterov` kwarg semantics; non-test consumer: `ferrotorch/src/lib.rs` re-export. |
| REQ-7 | NOT-STARTED | ferrotorch uses L2 wd at `ferrotorch-optim/src/muon.rs`; upstream uses decoupled wd. Blocked by #1466. |
| REQ-8 | SHIPPED | impl: `maximize` negation at `ferrotorch-optim/src/muon.rs`; non-test consumer: `ferrotorch/src/lib.rs` re-export. |
| REQ-9 | SHIPPED | impl: device-resident step body at `ferrotorch-optim/src/muon.rs` using `tensor_matmul` + device-aware arithmetic; non-test consumer: `ferrotorch/src/lib.rs` re-export. CUDA tests at `ferrotorch-optim/src/muon.rs` (`#[cfg(feature = "cuda")]`) verify residence + CPU/GPU agreement. |
| REQ-10 | SHIPPED | impl: `state_dict` at `ferrotorch-optim/src/muon.rs` + `load_state_dict` at `ferrotorch-optim/src/muon.rs`; non-test consumer: `ferrotorch/src/lib.rs` re-export. |
