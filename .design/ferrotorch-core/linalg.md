# Linear algebra (`torch.linalg.*`)

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - torch/linalg/__init__.py
  - aten/src/ATen/native/LinearAlgebra.cpp
  - aten/src/ATen/native/BatchLinearAlgebra.cpp
  - aten/src/ATen/native/cuda/linalg/CUDASolver.cpp
  - torch/_torch_docs.py
  - torch/overrides.py
-->

## Summary

`ferrotorch-core/src/linalg.rs` ships the `torch.linalg.*` free-function
namespace: decompositions (SVD, QR, Cholesky, LU, LDL), eigenproblems
(eigh/eigvalsh/eig/eigvals), solves (`solve`, `lstsq`,
`solve_triangular`, `tensorsolve`), inverses (`inv`, `pinv`,
`tensorinv`, `inv_ex`), norms (`matrix_norm`, `vector_norm`,
`matrix_rank`, `cond`), and other matrix operations (`det`,
`slogdet`, `matrix_power`, `matrix_exp`, `cross`, `multi_dot`,
`diagonal`, `householder_product`). The underlying numerics are
delegated to `ferray-linalg` (Rust LAPACK wrapper); on CUDA the
symmetric eigenproblems route through cuSOLVER `syevd`.

## Requirements

- REQ-1: `svd(input)` — reduced (thin) SVD `A = U @ diag(S) @ Vh`.
  CPU path uses `ferray_linalg::svd`. Mirrors
  `torch.linalg.svd(A, full_matrices=False)`.
- REQ-2: `solve(a, b)` — solve `A @ x = b` for square A and 1-D / 2-D
  b. CPU path uses `ferray_linalg::solve`. Mirrors
  `torch.linalg.solve`.
- REQ-3: `det(input)` — matrix determinant. Mirrors
  `torch.linalg.det`.
- REQ-4: `inv(input)` — matrix inverse. Mirrors
  `torch.linalg.inv`.
- REQ-5: `qr(input)` — reduced QR decomposition. Mirrors
  `torch.linalg.qr(A, mode='reduced')`.
- REQ-6: `cholesky(input)` — lower-triangular Cholesky factor.
  Mirrors `torch.linalg.cholesky`.
- REQ-7: `matrix_norm(input)` — Frobenius matrix norm (scalar
  output). Mirrors `torch.linalg.matrix_norm`.
- REQ-8: `pinv(input)` — Moore-Penrose pseudoinverse. Mirrors
  `torch.linalg.pinv`.
- REQ-9: `eigh(a)` / `eigvalsh(a)` — symmetric / Hermitian
  eigendecomposition. On CUDA via cuSOLVER `syevd` (with `jobz=NOVECTOR`
  for `eigvalsh`). Mirror `torch.linalg.eigh` / `torch.linalg.eigvalsh`.
- REQ-10: `eig(a)` / `eigvals(a)` — general eigendecomposition.
  Complex outputs encoded as tensors with a trailing dim of 2
  representing `[real, imag]`. CPU-only.
- REQ-11: `lu(a)` — full LU `A = P L U`. Mirrors
  `torch.linalg.lu`.
- REQ-12: `lu_factor(a)` — cuSOLVER-packed LU + pivots. Mirrors
  `torch.linalg.lu_factor`.
- REQ-13: `svdvals(a)` — singular values only. Mirrors
  `torch.linalg.svdvals`.
- REQ-14: `lstsq(a, b)` / `lstsq_solve(a, b)` — least-squares.
  Mirrors `torch.linalg.lstsq`.
- REQ-15: `matrix_power(a, n)` / `matrix_exp(a)` — matrix exponential
  / power.
- REQ-16: `tensorsolve(a, b)` / `tensorinv(a, ind)` — tensor solve /
  inverse. Mirror `torch.linalg.tensorsolve` / `tensorinv`.
- REQ-17: `vector_norm(input, ord)` — p-norm of a flattened tensor.
- REQ-18: `slogdet(a)` — sign + log-determinant.
- REQ-19: `matrix_rank(a, tol)` / `cond(a, p)` — rank / condition
  number.
- REQ-20: `cross(a, b, dim)` — 3-vector cross product.
- REQ-21: `multi_dot(matrices)` — optimal chained matmul.
- REQ-22: `diagonal(a, offset)` — matrix diagonal extraction.
- REQ-23: `solve_triangular(a, b, upper, transpose, unitriangular)` —
  triangular system solver.
- REQ-24: `ldl_factor(a)` / `ldl_solve(_, a, _)` — LDL factorisation.
- REQ-25: `householder_product(a, taus)` — Q from Householder reflectors.
- REQ-26: `cholesky_ex(_)` / `inv_ex(_)` / `solve_ex(_, _)` — "_ex"
  variants returning `(value, info)` where `info` is a status code
  rather than raising. Mirror `torch.linalg.{cholesky,inv,solve}_ex`.

## Acceptance Criteria

- [x] AC-1: `svd(A) → (U, S, Vh)` with `U @ diag(S) @ Vh ≈ A`.
- [x] AC-2: `solve(I, b) ≈ b`.
- [x] AC-3: `det(A * I) ≈ 1`.
- [x] AC-4: `cholesky(SPD) @ cholesky(SPD).T ≈ SPD`.
- [x] AC-5: `eigh(symm) → (w, Q)` with `Q @ diag(w) @ Q.T ≈ symm`.
- [x] AC-6: `cargo test -p ferrotorch-core --lib linalg` passes.

## Architecture

The file is ~3.1k LOC. Top-of-file helpers (`linalg.rs:20-112`):

- `tensor_to_arr2_f64` / `tensor_to_arr2_f32` — construct a
  `ferray::Array<f{32,64}, Ix2>` from a 2-D tensor's data, dispatched
  by type.
- `arr_to_vec_f64` / `arr_to_vec_f32` — convert `Array1` back to
  `Vec<T>` with the appropriate scalar cast.
- `is_f32` / `is_f64` — `TypeId`-based discriminators.
- `ensure_cpu_for_linalg(tensor)` — guard: linalg decompositions are
  CPU-only by default; this returns an explicit error for GPU tensors
  unless the op has a CUDA-aware override.

Each op (`svd` / `solve` / `det` / ...) follows the same pattern:

1. Guard with `ensure_cpu_for_linalg`.
2. Convert to `ferray::Array2<f{32,64}>` via the helper.
3. Call into `ferray_linalg::{svd, solve, cholesky, ...}`.
4. Marshal the result back into `Tensor<T>` via `arr_to_vec_*`.

CUDA-aware overrides (`eigh`, `eigvalsh`) route through the registered
`GpuBackend`'s `syevd_f32` / `syevd_f64` methods when the input is on
CUDA. The dispatch lives at `linalg.rs:569-668` and falls back to the
CPU path when the backend is unavailable.

Non-test production consumers:

- `ferrotorch-distributions/src/multivariate_normal.rs:44` imports
  `ferrotorch_core::linalg` and calls `linalg::cholesky` and
  `linalg::solve` to construct the multivariate normal distribution's
  Cholesky factor.
- `ferray_linalg::svd` callsite at `linalg.rs:156, 170` is the
  CPU dispatch for both SVD and `svdvals`.

## Parity contract

`parity_ops = []`. Decomposition outputs (U, V, L, Q, …) are
non-unique up to column sign / permutation, so direct byte-for-byte
parity sweeps would be misleading. The correctness contract is:

- Round-trip: reassembling the original matrix from the decomposition
  recovers it to within numerical tolerance (e.g. `Q @ R ≈ A`).
- Eigenvalues / singular values are sorted (descending for SVD,
  ascending for `eigh`) to a canonical order.

Future parity work should compare scalar invariants (singular
values, eigenvalues, determinant, condition number) against a live
PyTorch oracle.

## Verification

```bash
cargo test -p ferrotorch-core --lib linalg
```

Expected: round-trip tests for each decomposition pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `svd` at `ferrotorch-core/src/linalg.rs:121` delegating to `ferray_linalg::svd` (`:156, 170`); non-test consumer: re-exported in `lib.rs`; called by `pinv`, `svdvals`, `matrix_rank`, `cond` siblings within this same file (e.g. `cond` at `:1313` runs SVD). |
| REQ-2 | SHIPPED | impl: `solve` at `ferrotorch-core/src/linalg.rs:200`; non-test consumer: `ferrotorch-distributions/src/multivariate_normal.rs:44` imports `ferrotorch_core::linalg` and calls `linalg::solve`. |
| REQ-3 | SHIPPED | impl: `det` at `ferrotorch-core/src/linalg.rs:276`; non-test consumer: pub API; used by `slogdet` in the same file. |
| REQ-4 | SHIPPED | impl: `inv` at `ferrotorch-core/src/linalg.rs:310`; non-test consumer: `inv_ex` at `:2139` delegates to `inv` for the success path; production callsite via `lib.rs` re-export. |
| REQ-5 | SHIPPED | impl: `qr` at `ferrotorch-core/src/linalg.rs:348`; non-test consumer: pub API surface used by linear-regression / least-squares helpers downstream. |
| REQ-6 | SHIPPED | impl: `cholesky` at `ferrotorch-core/src/linalg.rs:419`; non-test consumer: `ferrotorch-distributions/src/multivariate_normal.rs` calls `linalg::cholesky` for the MVN covariance factor. |
| REQ-7 | SHIPPED | impl: `matrix_norm` at `ferrotorch-core/src/linalg.rs:471`; non-test consumer: pub API surface. |
| REQ-8 | SHIPPED | impl: `pinv` at `ferrotorch-core/src/linalg.rs:530`; non-test consumer: pub API; composes with `svd` (this file). |
| REQ-9 | SHIPPED | impl: `eigh` at `ferrotorch-core/src/linalg.rs:569`, `eigvalsh` at `:626`; non-test consumer: pub API; used by `matrix_norm` / `cond` for spectral computations on symmetric matrices. |
| REQ-10 | SHIPPED | impl: `eig` at `ferrotorch-core/src/linalg.rs:677`, `eigvals` at `:735`; non-test consumer: pub API. |
| REQ-11 | SHIPPED | impl: `lu` at `ferrotorch-core/src/linalg.rs:783`; non-test consumer: pub API. |
| REQ-12 | SHIPPED | impl: `lu_factor` at `ferrotorch-core/src/linalg.rs:833`; non-test consumer: pub API used by `solve` on CUDA dispatch and by `tensorsolve`. |
| REQ-13 | SHIPPED | impl: `svdvals` at `ferrotorch-core/src/linalg.rs:940`; non-test consumer: pub API; also called internally by `matrix_rank` and `cond`. |
| REQ-14 | SHIPPED | impl: `lstsq_solve` at `ferrotorch-core/src/linalg.rs:974`, `lstsq` at `:1023`; non-test consumer: pub API. |
| REQ-15 | SHIPPED | impl: `matrix_power` at `ferrotorch-core/src/linalg.rs:1106`, `matrix_exp` at `:1920`; non-test consumer: pub API; `matrix_exp` is used by ODE integrators in `ferrotorch-distributions` / continuous-time models. |
| REQ-16 | SHIPPED | impl: `tensorsolve` at `ferrotorch-core/src/linalg.rs:1135`, `tensorinv` at `:1163`; non-test consumer: pub API. |
| REQ-17 | SHIPPED | impl: `vector_norm` at `ferrotorch-core/src/linalg.rs:1194`; non-test consumer: pub API. |
| REQ-18 | SHIPPED | impl: `slogdet` at `ferrotorch-core/src/linalg.rs:1223`; non-test consumer: pub API; used by likelihood / log-prob computations in `ferrotorch-distributions`. |
| REQ-19 | SHIPPED | impl: `matrix_rank` at `ferrotorch-core/src/linalg.rs:1276`, `cond` at `:1313`; non-test consumer: pub API. |
| REQ-20 | SHIPPED | impl: `cross` at `ferrotorch-core/src/linalg.rs:1388`; non-test consumer: pub API. |
| REQ-21 | SHIPPED | impl: `multi_dot` at `ferrotorch-core/src/linalg.rs:1502`; non-test consumer: pub API. |
| REQ-22 | SHIPPED | impl: `diagonal` at `ferrotorch-core/src/linalg.rs:1545`; non-test consumer: pub API. |
| REQ-23 | SHIPPED | impl: `solve_triangular` at `ferrotorch-core/src/linalg.rs:1593`; non-test consumer: pub API; called by `cholesky_solve` paths. |
| REQ-24 | SHIPPED | impl: `ldl_factor` at `ferrotorch-core/src/linalg.rs:1713`, `ldl_solve` at `:1764`; non-test consumer: pub API. |
| REQ-25 | SHIPPED | impl: `householder_product` at `ferrotorch-core/src/linalg.rs:1835`; non-test consumer: pub API used by `qr` reconstruction. |
| REQ-26 | SHIPPED | impl: `cholesky_ex` at `ferrotorch-core/src/linalg.rs:2111`, `inv_ex` at `:2139`, `solve_ex` at `:2166`; non-test consumer: pub API. |
