# Linalg grad_fns

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158 (Revert "[dynamo] Implement nb_or/nb_inplace_or slot dispatch for | and |= operators (#181326)")
upstream-paths:
  - aten/src/ATen/native/LinearAlgebra.cpp
  - aten/src/ATen/native/BatchLinearAlgebra.cpp
  - torch/linalg/__init__.py
-->

## Summary

`ferrotorch-core/src/grad_fns/linalg.rs` is the autograd-tracking wrapper layer
for the linear-algebra family declared in `aten/src/ATen/native/LinearAlgebra.cpp`
and the structured-factorization family in
`aten/src/ATen/native/BatchLinearAlgebra.cpp`, exposed at the Python user
surface via `torch/linalg/__init__.py`. The file ships **fused `*Backward`
`GradFn` structs** for the four core matmul-family ops
(`mm`, `bmm`, `matmul`, `linalg.matmul`) plus three internal fused variants
(`mm_bt` = A @ B^T without materialising B^T, `linear_fused` = A @ W^T +
bias for the `nn::Linear` hot path, and `permute_0213` = the
attention-head reshape primitive).

Two-state distinction this doc maintains (per R-DOC-4): a REQ is **SHIPPED**
only when it has both a `*Backward` impl AND a non-test production consumer
(a grad-aware `pub fn` forward, a `Tensor` method, or another production
call site — the parity-sweep runner's dispatch table is a TEST-side
consumer and does NOT count). The matmul-family ops `mm`, `bmm`, `matmul`,
and `linalg.matmul` are SHIPPED end-to-end (impl + real production consumers
in `methods.rs`, `attention.rs`, `einsum.rs`, etc. + lib tests + parity smoke
`0 failed`); `matmul`/`linalg.matmul` closed #1347 (2026-05-26), routing the
CPU broadcast / 3D-x-2D / 4D paths and the bmm CPU fallback per-batch slabs
through the faer-backed `ops::linalg::mm_raw` workhorse. The runner's per-op
`tolerance_for` returns `rtol=1e-4` for matmul-family ops to acknowledge the
structural cross-BLAS-implementation f32 ULP variance (ferrotorch=faer vs
torch=MKL — see Parity-sweep status section); byte-for-byte parity vs MKL
requires the MKL/OpenBLAS FFI follow-up (future epic).

The tractable decomposition backwards `qr` (reduced, m≥n), `cholesky`
(Phi-symmetrisation VJP), and `slogdet` (real-case `g_abs * A^{-T}`)
shipped 2026-05-27 with `*Backward` `GradFn` structs in this file and
grad-aware forwards `pub fn qr` / `pub fn cholesky` / `pub fn slogdet` in
`ferrotorch-core/src/linalg.rs` delegating to them — these are SHIPPED
end-to-end (REQ-16/17/21). The well-conditioned factorizations
`linalg.solve` (`LinalgSolveBackward`), `linalg.inv` (`LinalgInvBackward`),
`linalg.det` (`LinalgDetBackward`) and the structural-autograd `trace`
(`TraceBackward`) / `outer` (`OuterBackward`) shipped 2026-05-27 (closing
#1583's solve/inv/det/trace/outer subset): their forwards `pub fn solve` /
`pub fn inv` / `pub fn det` / `pub fn trace` / `pub fn outer` in
`ferrotorch-core/src/linalg.rs` are now grad-aware and delegate to the
matching `*_differentiable` wrapper (which computes the forward under
`no_grad` to avoid re-entry), and each VJP is FD-verified by a
public-forward-driven test in this file's `#[cfg(test)] mod tests` — these
are SHIPPED end-to-end (REQ-10/18/20/29/35).

A set of `*Backward` structs still exists in this file BUT has no non-test
production consumer (the corresponding forward is not grad-aware and no
`Tensor` method or public forward delegates to it): the fused-affine
family (`AddmmBackward`, `AddbmmBackward`, `BaddbmmBackward`, `AddmvBackward`,
`AddrBackward`), the structural-autograd `KronBackward`, and the
diagonal-family `DiagonalBackward` / `DiagBackward` / `TriangularBackward`.
Per R-DOC-4 these are **NOT-STARTED end-to-end** and tracked by the
remaining consumer-wiring prereq blocker #1583 (the impl exists; the
grad-aware forward wiring — mirroring the qr/cholesky/slogdet and now
solve/inv/det/trace/outer pattern — does not).

The remaining `torch.linalg.*` factorizations (`svd`, `eig`, `eigh`,
`eigvals`, `eigvalsh`, `pinv`, `lstsq`, `norm`, `matrix_rank`, `cross`,
`householder_product`, `lu`, `lu_factor`) are **forward-only** in
`ferrotorch-core/src/linalg.rs` with no `*Backward` `GradFn` at all. Those
are NOT-STARTED and tracked by prereq blocker #1345; the research-grade
degenerate-eigenvalue / gauge-freedom subset
(`svd`/`eigh`/`eigvalsh`/`pinv`/`lstsq`/`lu`/`lu_factor`) is tracked under
sub-blocker #1577.

## Requirements

- REQ-1: `mm(A, B)` — 2D matrix multiply with fused VJP
  `dA = grad_C @ B^T`, `dB = A^T @ grad_C`. Mirrors
  `TORCH_IMPL_FUNC(mm_out_cpu)` at
  `aten/src/ATen/native/LinearAlgebra.cpp:1641` and the `torch.mm` Python
  surface. GPU fast path for f32/f64 via cuBLAS gemm; bf16/f16 routed via
  `dispatch_floating_dtype!` macro to `matmul_bf16_bf16` /
  `matmul_f16_f16` cuBLAS GemmEx kernels with f32 accumulator. Backward
  path is GPU-native (no CPU roundtrip) for f32/f64 via `mm_backward_gpu`.

- REQ-2: `bmm(A, B)` — 3D batched matrix multiply with fused VJP
  `dA[b] = grad_C[b] @ B[b]^T`, `dB[b] = A[b]^T @ grad_C[b]`. Mirrors
  `TORCH_IMPL_FUNC(bmm_out_cpu)` at
  `aten/src/ATen/native/LinearAlgebra.cpp:1894` and `torch.bmm`. GPU fast
  path for f32/f64 via cuBLAS `SgemmStridedBatched` /
  `DgemmStridedBatched`; f32 + autocast ReducedPrecision routes to
  `bmm_f16_f32` (Tensor Core path with f32 accumulator). Backward uses
  `batch_transpose` (permute + contiguous) so the transpose stays
  on-device.

- REQ-3: `matmul(A, B)` — general matmul dispatcher across all rank
  combinations (1D×1D = dot, 2D×1D = mv, 1D×2D = vm, 2D×2D = mm, 3D×3D =
  bmm, broadcast ≥3D = `broadcast_matmul_backward`). Mirrors
  `Tensor matmul(const Tensor & tensor1, const Tensor & tensor2)` at
  `aten/src/ATen/native/LinearAlgebra.cpp:2190`. GPU paths exist for
  1D×2D (cuBLAS gemv with OP_N transpose, `vm_f32`/`vm_f64`), 2D×2D
  (`matmul_f32`/`_f64`/`matmul_bf16_bf16`/`matmul_f16_f16` /
  `matmul_f16_f32` under autocast), and broadcast-bmm (4D bmm, 3D×2D,
  2D×3D, leading-dim broadcasts) via cuBLAS
  `gemmStridedBatched` with stride-0 on broadcasted axes
  (`broadcast_bmm_f32`/`_f64`). Backward dispatches via
  `MatmulBackward` to the rank-appropriate inner backward
  (`MmBackward` / `MvBackward` / `DotBackward` / inline vm / inline
  broadcast-bmm).

- REQ-4: `linalg.matmul(A, B)` — `torch.linalg.matmul` is an alias for
  `torch.matmul` per `Tensor linalg_matmul(const Tensor & tensor1,
  const Tensor & tensor2)` at
  `aten/src/ATen/native/LinearAlgebra.cpp:2206` (literally
  `return at::matmul(tensor1, tensor2)`). Documented at
  `torch/linalg/__init__.py:1651` (`matmul = _add_docstr(...)`).
  Satisfied by the same `matmul_differentiable` implementation as REQ-3.

- REQ-5: `addmm(self, mat1, mat2, beta=1, alpha=1) = beta * self + alpha *
  mat1 @ mat2`. Mirrors `TORCH_META_FUNC(addmm)` and
  `TORCH_IMPL_FUNC(addmm_out_cpu)` in
  `aten/src/ATen/native/LinearAlgebra.cpp`. **NOT-STARTED end-to-end**:
  `AddmmBackward` + `addmm_differentiable` now exist in
  `ferrotorch-core/src/grad_fns/linalg.rs`, but there is NO non-test
  production consumer (no public `addmm` forward delegating to the
  wrapper); only the parity-sweep runner + in-file tests call it. Open
  prereq blocker #1583 (consumer wiring).

- REQ-6: `addbmm(self, batch1, batch2, beta=1, alpha=1) = beta * self +
  alpha * sum_b(batch1[b] @ batch2[b])`. Mirrors `Tensor addbmm(...)` in
  `aten/src/ATen/native/LinearAlgebra.cpp`. **NOT-STARTED end-to-end**:
  `AddbmmBackward` + `addbmm_differentiable` exist in
  `ferrotorch-core/src/grad_fns/linalg.rs` but have no non-test
  production consumer. Open prereq blocker #1583.

- REQ-7: `baddbmm(self, batch1, batch2, beta=1, alpha=1) = beta * self +
  alpha * bmm(batch1, batch2)`. Mirrors `TORCH_META_FUNC(baddbmm)` and
  `TORCH_IMPL_FUNC(baddbmm_out_cpu)` in
  `aten/src/ATen/native/LinearAlgebra.cpp`. **NOT-STARTED end-to-end**:
  `BaddbmmBackward` + `baddbmm_differentiable` exist in
  `ferrotorch-core/src/grad_fns/linalg.rs` but have no non-test
  production consumer. Open prereq blocker #1583.

- REQ-8: `addmv(self, mat, vec, beta=1, alpha=1) = beta * self + alpha *
  mat @ vec`. Mirrors `TORCH_META_FUNC(addmv)` and
  `TORCH_IMPL_FUNC(addmv_out_cpu)` in `aten/src/ATen/native/Blas.cpp`.
  **NOT-STARTED end-to-end**: `AddmvBackward` + `addmv_differentiable`
  exist in `ferrotorch-core/src/grad_fns/linalg.rs` but have no non-test
  production consumer. Open prereq blocker #1583.

- REQ-9: `addr(self, vec1, vec2, beta=1, alpha=1) = beta * self + alpha *
  outer(vec1, vec2)`. Mirrors `Tensor addr(const Tensor& self, ...)` in
  `aten/src/ATen/native/LinearAlgebra.cpp`. **NOT-STARTED end-to-end**:
  `AddrBackward` + `addr_differentiable` exist in
  `ferrotorch-core/src/grad_fns/linalg.rs` but have no non-test
  production consumer. Open prereq blocker #1583.

- REQ-10: `linalg.solve(A, B)` — solve a square system `A @ X = B`.
  Mirrors `Tensor linalg_solve(const Tensor& A, ...)` in
  `aten/src/ATen/native/BatchLinearAlgebra.cpp` and documented in
  `torch/linalg/__init__.py`. **SHIPPED** (2026-05-27):
  `LinalgSolveBackward` + `solve_differentiable` in
  `ferrotorch-core/src/grad_fns/linalg.rs` implement the VJP
  `gB = A^{-T} @ gX`, `gA = -gB @ X^T` (vector RHS handled by
  column-promotion), grounded in `linalg_solve_backward` at upstream
  `torch/csrc/autograd/FunctionsManual.cpp:6160`. FD-verified by
  `fn solve_forward_is_grad_aware_and_matches_fd_matrix_rhs` and
  `fn solve_forward_is_grad_aware_and_matches_fd_vector_rhs` in the in-file
  `#[cfg(test)] mod tests`. Non-test production consumer: the grad-aware
  forward `pub fn solve` in `ferrotorch-core/src/linalg.rs` (the
  `torch.linalg.solve` public surface) delegates to `solve_differentiable`
  when `!a.is_cuda() && is_grad_enabled() && (a.requires_grad() ||
  b.requires_grad())`; the wrapper computes the forward inside `no_grad`
  (preventing re-entry).

- REQ-11: `linalg.svd(A, full_matrices=True)` — singular value
  decomposition `A = U @ diag(S) @ Vh`. Documented in
  `torch/linalg/__init__.py`. Forward-only impl `pub fn svd` in
  `ferrotorch-core/src/linalg.rs` (via `ferray_linalg::svd`).
  **NOT-STARTED in this file** — no `LinalgSvdBackward`. Open prereq
  blocker #1577 (research-grade degenerate-singular-value / gauge VJP).

- REQ-12: `linalg.eig(A)` — non-symmetric eigendecomposition. Documented
  in `torch/linalg/__init__.py`. Forward-only impl `pub fn eig` in
  `ferrotorch-core/src/linalg.rs` via `ferray_linalg::eig`.
  **NOT-STARTED in this file**. Open prereq blocker #1345.

- REQ-13: `linalg.eigh(A, UPLO='L')` — symmetric/Hermitian
  eigendecomposition. Documented in `torch/linalg/__init__.py`.
  Forward-only impl `pub fn eigh` in `ferrotorch-core/src/linalg.rs` via
  `ferray_linalg::eigh`. **NOT-STARTED in this file**. Open prereq
  blocker #1577 (research-grade degenerate-eigenvalue / gauge VJP).

- REQ-14: `linalg.eigvals(A)` — eigenvalues only (non-symmetric).
  Documented in `torch/linalg/__init__.py`. Forward-only impl
  `pub fn eigvals` in `ferrotorch-core/src/linalg.rs` via
  `ferray_linalg::eigvals`. **NOT-STARTED in this file**. Open prereq
  blocker #1345.

- REQ-15: `linalg.eigvalsh(A, UPLO='L')` — eigenvalues only
  (symmetric/Hermitian). Documented in `torch/linalg/__init__.py`.
  Forward-only impl `pub fn eigvalsh` in `ferrotorch-core/src/linalg.rs`
  via `ferray_linalg::eigvalsh`. **NOT-STARTED in this file**. Open
  prereq blocker #1577 (research-grade degenerate-eigenvalue subset).

- REQ-16: `linalg.qr(A, mode='reduced')` — QR factorization. Documented
  in `torch/linalg/__init__.py`, derivative `linalg_qr` in
  `tools/autograd/derivatives.yaml`. **SHIPPED** (2026-05-27):
  `QrBackwardQ`/`QrBackwardR` + `qr_differentiable` in
  `ferrotorch-core/src/grad_fns/linalg.rs` implement the real
  `linalg_qr_backward` VJP for the reduced, `m >= n` case
  (`gA = (Q @ syminvadj(triu(M)) + gQ) @ R^{-T}`,
  `M = gR @ R^T - Q^T @ gQ`), grounded in `linalg_qr_backward` in
  upstream `FunctionsManual.cpp`. The jointly-linear `(gQ, gR)` VJP is
  split across two single-output nodes (`QrBackwardQ` on `Q`,
  `QrBackwardR` on `R`) whose `A.grad` contributions the autograd engine
  accumulates. FD-verified by `fn qr_backward_matches_finite_difference_square`
  and `fn qr_backward_q_only_and_r_only` in the in-file
  `#[cfg(test)] mod tests` block. Non-test production consumer: the
  grad-aware forward `pub fn qr` in `ferrotorch-core/src/linalg.rs` (the
  `torch.linalg.qr` public surface) delegates to `qr_differentiable`
  when `is_grad_enabled() && input.requires_grad()`. The `m < n`
  (`trilImInvAdjSkew`) branch is the research-grade case tracked under
  sub-blocker #1577.

- REQ-17: `linalg.cholesky(A)` — Cholesky factorization for SPD matrices.
  Mirrors `Tensor linalg_cholesky(const Tensor& A, bool upper)` in
  `aten/src/ATen/native/BatchLinearAlgebra.cpp`, documented in
  `torch/linalg/__init__.py`, derivative `cholesky` in
  `tools/autograd/derivatives.yaml`. **SHIPPED** (2026-05-27):
  `CholeskyBackward` + `cholesky_differentiable` in
  `ferrotorch-core/src/grad_fns/linalg.rs` implement the
  Phi-symmetrisation VJP `gA = L^{-T} Φ(tril(L^T gL)) L^{-1}` (where
  `Φ(X) = 0.5*(tril(X) + tril(X,-1)^T)`), grounded in `cholesky_backward`
  in upstream `FunctionsManual.cpp` (real lower case). The two triangular
  solves reuse `pub fn solve_triangular` in
  `ferrotorch-core/src/linalg.rs`; the returned gradient is symmetric
  (PyTorch contract). FD-verified by
  `fn cholesky_backward_matches_finite_difference` (symmetric finite
  difference + symmetry assertion) in the in-file `#[cfg(test)] mod tests`
  block. Non-test production consumer: the grad-aware forward
  `pub fn cholesky` in `ferrotorch-core/src/linalg.rs` (the
  `torch.linalg.cholesky` public surface) delegates to
  `cholesky_differentiable` when grad is enabled.

- REQ-18: `linalg.inv(A)` — matrix inverse. Mirrors `Tensor linalg_inv(
  const Tensor& A)` in `aten/src/ATen/native/BatchLinearAlgebra.cpp` and
  documented in `torch/linalg/__init__.py`. **SHIPPED** (2026-05-27):
  `LinalgInvBackward` + `inv_differentiable` in
  `ferrotorch-core/src/grad_fns/linalg.rs` implement the VJP
  `dA = -Y^T @ grad @ Y^T`, `Y = A^{-1}`, per `linalg_inv_ex` at upstream
  `tools/autograd/derivatives.yaml:916`. FD-verified by
  `fn inv_forward_is_grad_aware_and_matches_fd` in the in-file
  `#[cfg(test)] mod tests`. Non-test production consumer: the grad-aware
  forward `pub fn inv` in `ferrotorch-core/src/linalg.rs` (the
  `torch.linalg.inv` public surface) delegates to `inv_differentiable`
  when `is_grad_enabled() && input.requires_grad()`; the wrapper computes
  the forward inside `no_grad` (preventing re-entry).

- REQ-19: `linalg.pinv(A, atol=None, rtol=None)` — Moore-Penrose
  pseudoinverse. Mirrors `Tensor linalg_pinv(...)` in
  `aten/src/ATen/native/LinearAlgebra.cpp`. Forward-only impl
  `pub fn pinv` in `ferrotorch-core/src/linalg.rs` via
  `ferray_linalg::pinv`. **NOT-STARTED in this file**. Open prereq
  blocker #1577 (research-grade pseudoinverse VJP).

- REQ-20: `linalg.det(A)` — determinant. Mirrors `Tensor linalg_det(const
  Tensor& A)` in `aten/src/ATen/native/LinearAlgebra.cpp` and documented
  in `torch/linalg/__init__.py`. **SHIPPED** (2026-05-27):
  `LinalgDetBackward` + `det_differentiable` in
  `ferrotorch-core/src/grad_fns/linalg.rs` implement the VJP
  `dA = det(A) * grad * inv(A)^T`, the invertible branch of
  `linalg_det_backward` at upstream
  `torch/csrc/autograd/FunctionsManual.cpp:4373`. FD-verified by
  `fn det_forward_is_grad_aware_and_matches_fd` in the in-file
  `#[cfg(test)] mod tests`. Non-test production consumer: the grad-aware
  forward `pub fn det` in `ferrotorch-core/src/linalg.rs` (the
  `torch.linalg.det` public surface) delegates to `det_differentiable`
  when `is_grad_enabled() && input.requires_grad()`; the wrapper computes
  the forward (and the VJP's internal `inv`) inside `no_grad` (preventing
  re-entry).

- REQ-21: `linalg.slogdet(A)` — sign and log-magnitude of the
  determinant. Documented in `torch/linalg/__init__.py`, derivative
  `_linalg_slogdet` in `tools/autograd/derivatives.yaml`
  (`output_differentiability: [True, True, False, False]`). **SHIPPED**
  (2026-05-27): `SlogdetBackward` + `slogdet_differentiable` in
  `ferrotorch-core/src/grad_fns/linalg.rs` attach the real-case VJP
  `dA = grad_logabsdet * inv(A)^T` to the differentiable `logabsdet`
  output (the `sign` output carries no real gradient — locally constant —
  so it is returned plain), grounded in `slogdet_backward` in upstream
  `FunctionsManual.cpp` (where the real case collapses
  `(g_abs - g_sign*sgn)*A^{-H}` to `g_abs * A^{-H}`). FD-verified by
  `fn slogdet_backward_matches_finite_difference` in the in-file
  `#[cfg(test)] mod tests` block. Non-test production consumer: the
  grad-aware forward `pub fn slogdet` in `ferrotorch-core/src/linalg.rs`
  (the `torch.linalg.slogdet` public surface) delegates to
  `slogdet_differentiable` when grad is enabled.

- REQ-22: `linalg.lstsq(A, B, rcond=None)` — least-squares solver.
  Documented in `torch/linalg/__init__.py`. Forward-only impl
  `pub fn lstsq` in `ferrotorch-core/src/linalg.rs` via
  `ferray_linalg::lstsq`. **NOT-STARTED in this file**. Open prereq
  blocker #1577 (least-squares VJP via QR; rank-deficient subset).

- REQ-23: `linalg.norm(A, ord=None, dim=None)` — generic norm (Frobenius
  for matrices, p-norm for vectors). Documented in
  `torch/linalg/__init__.py`. Forward-only impls `pub fn matrix_norm` and
  `pub fn vector_norm` in `ferrotorch-core/src/linalg.rs` via
  `ferray_linalg::norm`. **NOT-STARTED in this file**. Open prereq
  blocker #1345.

- REQ-24: `linalg.matrix_rank(A, tol=None)`. Mirrors `Tensor
  linalg_matrix_rank(...)` in `aten/src/ATen/native/LinearAlgebra.cpp`
  (overload family). Forward-only impl `pub fn matrix_rank` in
  `ferrotorch-core/src/linalg.rs`. **NOT-STARTED in this file**. Open
  prereq blocker #1345.

- REQ-25: `linalg.cross(A, B, dim=-1)` — vector cross product along
  `dim` (must equal 3). Forward-only impl `pub fn cross` in
  `ferrotorch-core/src/linalg.rs`. **NOT-STARTED in this file**. Open
  prereq blocker #1345.

- REQ-26: `linalg.householder_product(A, tau)`. Mirrors `Tensor
  linalg_householder_product(...)` in
  `aten/src/ATen/native/BatchLinearAlgebra.cpp` and documented in
  `torch/linalg/__init__.py`. Forward-only impl `pub fn
  householder_product` in `ferrotorch-core/src/linalg.rs`.
  **NOT-STARTED in this file**. Open prereq blocker #1345.

- REQ-27: `linalg.lu(A, pivot=True)` — LU factorization with pivoting.
  Documented in `torch/linalg/__init__.py`. Forward-only impl
  `pub fn lu` in `ferrotorch-core/src/linalg.rs` via
  `ferray_linalg::lu`. **NOT-STARTED in this file**. Open prereq blocker
  #1577 (LU pivoting / gauge-freedom VJP).

- REQ-28: `linalg.lu_factor(A)` — LU factorization without explicit
  unpacking. Documented in `torch/linalg/__init__.py`. Forward-only impl
  `pub fn lu_factor` in `ferrotorch-core/src/linalg.rs`. **NOT-STARTED
  in this file**. Open prereq blocker #1577.

- REQ-29: `trace(A)` — sum of the main diagonal. **SHIPPED** (2026-05-27):
  the grad-aware forward `pub fn trace` in `ferrotorch-core/src/linalg.rs`
  (sum of `A[i,i]`, scalar output) delegates to `trace_differentiable`
  (which attaches `TraceBackward`, VJP `dA = grad * I` per
  `trace_backward_symint` at upstream
  `tools/autograd/derivatives.yaml:1785`) when
  `is_grad_enabled() && a.requires_grad()`; the wrapper computes the
  forward inside `no_grad` (preventing re-entry). FD-verified by
  `fn trace_forward_is_grad_aware_and_matches_fd` in the in-file
  `#[cfg(test)] mod tests`. The grad-aware forward is the non-test
  production consumer.

- REQ-30: `diagonal(A, offset=0, dim1=0, dim2=1)`. Mirrors `Tensor
  linalg_diagonal(...)` in `aten/src/ATen/native/LinearAlgebra.cpp`.
  **NOT-STARTED end-to-end**: forward-only `pub fn diagonal` in
  `ferrotorch-core/src/linalg.rs` + `DiagonalBackward` /
  `diagonal_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs`
  exist, but the forward is NOT grad-aware and the wrapper has no non-test
  production consumer. Open prereq blocker #1583.

- REQ-31: `diag(A, diagonal=0)` — extract or construct a diagonal.
  **NOT-STARTED end-to-end**: forward-only `pub fn diag` in
  `ferrotorch-core/src/ops/tensor_ops.rs` + `DiagBackward` /
  `diag_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs`
  exist, but the forward is NOT grad-aware and the wrapper has no non-test
  production consumer. Open prereq blocker #1583.

- REQ-32: `tril(A, diagonal=0)` — lower triangular zeroing.
  **NOT-STARTED end-to-end**: forward-only `pub fn tril` in
  `ferrotorch-core/src/ops/tensor_ops.rs` + `TriangularBackward` /
  `tril_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs`
  exist, but the forward is NOT grad-aware and the wrapper has no non-test
  production consumer. Open prereq blocker #1583.

- REQ-33: `triu(A, diagonal=0)` — upper triangular zeroing.
  **NOT-STARTED end-to-end**: forward-only `pub fn triu` in
  `ferrotorch-core/src/ops/tensor_ops.rs` + `triu_differentiable`
  (sharing `TriangularBackward`) in
  `ferrotorch-core/src/grad_fns/linalg.rs` exist, but the forward is NOT
  grad-aware and the wrapper has no non-test production consumer. Open
  prereq blocker #1583.

- REQ-34: `kron(A, B)` — Kronecker product. Mirrors `Tensor kron(const
  Tensor& self, const Tensor& other)` in
  `aten/src/ATen/native/LinearAlgebra.cpp`. **NOT-STARTED end-to-end**:
  `KronBackward` + `kron_differentiable` exist in
  `ferrotorch-core/src/grad_fns/linalg.rs`, but there is NO public
  `pub fn kron` forward anywhere in ferrotorch-core src/, so no non-test
  production consumer. Open prereq blocker #1583.

- REQ-35: `outer(self, vec2)` — outer product. Mirrors `Tensor outer(
  const Tensor& self, const Tensor& vec2)` in
  `aten/src/ATen/native/LinearAlgebra.cpp` (which delegates to
  `self.reshape({-1, 1}) * vec2`). **SHIPPED** (2026-05-27): the
  grad-aware forward `pub fn outer` in `ferrotorch-core/src/linalg.rs`
  (`out[i,j] = a[i] * b[j]`, 1-D × 1-D) delegates to
  `outer_differentiable` (which attaches `OuterBackward`, VJP
  `da = grad_C @ b`, `db = grad_C^T @ a` per the `addr` vec1/vec2
  gradients at upstream `tools/autograd/derivatives.yaml:275-276`) when
  `is_grad_enabled() && (a.requires_grad() || b.requires_grad())`; the
  wrapper computes the forward inside `no_grad` (preventing re-entry).
  FD-verified by `fn outer_forward_is_grad_aware_and_matches_fd` in the
  in-file `#[cfg(test)] mod tests`. The grad-aware forward is the non-test
  production consumer.

## Acceptance Criteria

- [x] AC-1: `cargo test -p ferrotorch-core grad_fns::linalg` passes all
  forward and backward unit tests in the `#[cfg(test)] mod tests` block
  inside `ferrotorch-core/src/grad_fns/linalg.rs` (~690 LOC of tests).
- [x] AC-2: `mm` / `bmm` / `matmul` (all rank-combination dispatch
  branches) / `linalg.matmul` (alias) backward correctness verified by
  closed-form expected gradients computed in the test functions (`dA =
  grad_C @ B^T`, `dB = A^T @ grad_C`, etc.) at residual `< 1e-5` for the
  2x2/3x3 representative cases.
- [x] AC-3: `MatmulBackward` dispatches to the correct inner backward
  based on operand ranks at forward time (1D×1D → `DotBackward`, 2D×1D →
  `MvBackward`, 2D×2D → `MmBackward`, 3D×3D → `BmmBackward`, broadcast
  → `broadcast_matmul_backward`) — verified by
  `fn test_matmul_backward_dispatches_to_dot`,
  `fn test_matmul_backward_dispatches_to_mm`, and the bmm-dispatch test
  in `linalg.rs`.
- [x] AC-4: `mm_differentiable` GPU fast path for f32/f64/bf16/f16 routes
  through the dtype-aware `dispatch_floating_dtype!` macro at
  `ferrotorch-core/src/grad_fns/linalg.rs` (line range covered by `pub
  fn mm_differentiable`) — verified by GPU-side runs in
  `ferrotorch-core/tests/conformance_*` tests (live-GPU when CUDA is
  detected).
- [x] AC-5: `bmm_differentiable` backward uses `batch_transpose` (permute
  + contiguous) to keep the transpose on-device, avoiding the
  GPU→CPU→GPU roundtrip that dominated the pre-#796 path — see `fn
  batch_transpose` in `linalg.rs`.
- [x] AC-6: `dot_differentiable` on CUDA correctly handles the scalar
  output by extracting the 1-element grad via `.cpu()?.item()?` rather
  than the previously-broken `.data()?` path that returned
  `GpuTensorNotAccessible` — see the `if grad_output.is_cuda()` branch
  inside `impl<T: Float> GradFn<T> for DotBackward<T>` in `linalg.rs`.
- [x] AC-7: `linear_fused` saves bias as `Option<Tensor<T>>` and emits
  the correct gradient count from `inputs()` (2 if no bias, 3 if bias) —
  verified by `LinearFusedBackward::inputs` in `linalg.rs`.
- [x] AC-8: `broadcast_matmul_backward` correctly reduces gradients
  back to the original A/B shapes when batch dims were broadcast-expanded
  — verified by the `reduce_to_shape` helper and broadcast tests in
  `linalg.rs`.
- [x] AC-9: `no_grad` context disables grad-fn attachment for every
  differentiable forward in this file (the `is_grad_enabled() &&
  X.requires_grad()` guard at every `Tensor::from_operation` /
  `Tensor::from_storage` branch).
- [ ] AC-10: All 35 `parity_ops` from the route table return `N/N passed
  (0 skipped, 0 failed)` under `./target/release/parity-sweep sweep --op
  <op> --seeds 8`. **PARTIAL**: the four matmul-family ops all pass under
  the matmul-family `rtol=1e-4` tolerance contract (closes #1347):
  `mm 24/24 passed, 0 failed`; `bmm 8/8 passed, 0 failed`;
  `matmul 120/120 passed, 0 failed`; `linalg.matmul 120/120 passed,
  0 failed` (all verified 2026-05-26 at seeds=8). The GradFn-bearing
  ops `trace`, `outer`, `linalg.det`, `linalg.inv`, `linalg.solve` have
  a runner arm and pass for the well-conditioned op_db samples
  (`trace 8/8`, `outer 8/8`, `linalg.det 16/72 non-skipped`, `linalg.inv
  8/64 non-skipped`, `linalg.solve 24/192 non-skipped`, all `0 failed`;
  the det/inv/solve skips are op_db's batched / 0-sized samples — the
  faer forward is square-2-D-only) — but the runner is a TEST-side
  consumer, so this does NOT make those ops SHIPPED end-to-end (the
  production forwards are not grad-aware; see #1583). The decomposition
  ops `qr`, `cholesky`, `slogdet` (REQ-16/17/21) are FD-verified in
  `grad_fns/linalg.rs`'s `#[cfg(test)] mod tests` and consumed by the
  grad-aware `crate::linalg::{qr,cholesky,slogdet}` forwards, but have
  no parity-sweep runner arm yet (umbrella test-infra blocker #1344).
  The remaining linalg ops (svd/eig/eigh/eigvals/eigvalsh/pinv/lstsq/lu/
  lu_factor/householder_product/norm/matrix_rank, and the addmm family
  end-to-end) still need wiring; the research-grade degenerate /
  gauge-freedom set is tracked under #1577.
- [x] AC-11: `addmm` / `addbmm` / `baddbmm` / `addmv` / `addr` / `trace`
  / `diagonal` / `diag` / `tril` / `triu` / `kron` / `outer`
  `GradFn`-bearing fused implementations (the `*Backward` structs +
  `*_differentiable` wrappers) exist in
  `ferrotorch-core/src/grad_fns/linalg.rs`. (Note: the IMPLS land here,
  but none is wired to a non-test production consumer — that end-to-end
  wiring is AC-tracked separately and blocked on #1583.)
- [ ] AC-12: `linalg.solve` / `linalg.svd` / `linalg.eig` / `linalg.eigh`
  / `linalg.eigvals` / `linalg.eigvalsh` / `linalg.qr` / `linalg.cholesky`
  / `linalg.inv` / `linalg.pinv` / `linalg.det` / `linalg.slogdet` /
  `linalg.lstsq` / `linalg.norm` / `linalg.matrix_rank` / `linalg.cross`
  / `linalg.householder_product` / `linalg.lu` / `linalg.lu_factor` gain
  fused `*Backward` `GradFn` impls in this file AND a grad-aware forward
  that delegates to them. **PARTIAL**: `linalg.qr` (reduced m≥n),
  `linalg.cholesky`, `linalg.slogdet` (REQ-16/17/21) AND `linalg.solve`,
  `linalg.inv`, `linalg.det` (REQ-10/18/20, wired 2026-05-27) are wired
  end-to-end — their forwards `pub fn qr` / `pub fn cholesky` /
  `pub fn slogdet` / `pub fn solve` / `pub fn inv` / `pub fn det` in
  `ferrotorch-core/src/linalg.rs` delegate to the matching
  `*_differentiable` wrappers (forward computed under `no_grad`). The
  remaining factorizations
  (svd/eig/eigh/eigvals/eigvalsh/pinv/lstsq/norm/matrix_rank/cross/
  householder_product/lu/lu_factor) have no `GradFn` at all; the
  research-grade degenerate / gauge-freedom subset
  (svd/eigh/eigvalsh/pinv/lstsq/lu) is tracked under #1577, the rest
  under #1345.

## Architecture

### Module-level public surface

The file exposes 6 `pub struct *Backward` autograd nodes
(`MmBackward`, `MvBackward`, `DotBackward`, `BmmBackward`,
`MatmulBackward`, plus two crate-private fused variants `MmBtBackward`
and `LinearFusedBackward`), 8 `pub fn` differentiable forward wrappers
(`mm_differentiable`, `bmm_differentiable`, `matmul_differentiable`,
`mv_differentiable`, `dot_differentiable`, `mm_bt_differentiable`,
`linear_fused`, `bmm`), and one shape-utility `pub fn permute_0213`.
Every differentiable forward function follows the same scaffold:

1. **Device check**: `a.device() != b.device()` → `DeviceMismatch`.
2. **Materialize non-contiguous views**: `if !a.is_contiguous() {
   a.contiguous()? }`.
3. **GPU fast path**: if `a.is_cuda() && gpu_backend().is_some()`,
   dispatch to the dtype-appropriate cuBLAS kernel via
   `dispatch_floating_dtype!` (f32/f64/bf16/f16) and the
   `autocast_guard("op")` ReducedPrecision branch when applicable.
4. **CPU fallback**: dispatch to `ops::linalg::mm_raw` /
   `mm_raw_bt` / `mm_raw_at` (zero-copy raw-slice loops).
5. **Grad-fn attach**: if `is_grad_enabled() && (a.requires_grad() ||
   b.requires_grad())`, wrap the storage in `Tensor::from_operation`
   with the appropriate `*Backward` node; else `Tensor::from_storage`.

### REQ-1 mm

The `pub struct MmBackward` in `linalg.rs` saves `a: Tensor<T>` and
`b: Tensor<T>` (clones, so the backward graph survives even if the
caller drops the inputs). The VJP path branches:
- **GPU f32/f64**: `mm_backward_gpu` helper at the top of
  `linalg.rs` — `dA = grad_C @ B^T` via `backend.transpose_2d_*` +
  `backend.matmul_*`; `dB = A^T @ grad_C` similarly. All cuBLAS, no
  CPU roundtrip.
- **GPU non-f32/f64**: `Err(NotImplementedOnCuda { op: "MmBackward" })`.
- **CPU**: `crate::ops::linalg::mm_raw_bt` (for `dA`) and `mm_raw_at`
  (for `dB`) — direct zero-copy slice multiplications.

The forward `pub fn mm_differentiable` is the wrapper. It also handles
the autocast-`ReducedPrecision` branch where f32 inputs route to the
f16-accumulator cuBLAS path `matmul_f16_f32`. Non-test production
consumer: `pub fn mm` (the `Tensor::mm` method) in
`ferrotorch-core/src/methods.rs`; `use ferrotorch_core::grad_fns::linalg::mm_differentiable`
in `ferrotorch-nn/src/functional.rs` (the `pub fn linear` composite
path), `ferrotorch-nn/src/attention.rs` (Q/K/V projection
`mm_differentiable`), `ferrotorch-nn/src/lora.rs` (LoRA adapter
projections), `ferrotorch-nn/src/rnn.rs` (RNN gate computations:
`use ... mm_differentiable as mm`), and
`ferrotorch-core/src/grad_fns/shape.rs` (row-sums helper).

### REQ-2 bmm

The `pub struct BmmBackward` saves `a` and `b`. Backward computes
`grad_A = bmm(grad_C, batch_transpose(B))` and `grad_B = bmm(
batch_transpose(A), grad_C)` inside a `no_grad` block (we don't want
the backward computation itself to be tracked for second-order
gradients in this file's scope — second-order matmul wiring is a
future iteration). `batch_transpose` (at the top of `linalg.rs`) uses
`permute(&[0, 2, 1])? . contiguous()` so the transpose stays on-device
on CUDA.

`pub fn bmm` (the device-aware forward) and `pub fn bmm_differentiable`
(the autograd-attaching wrapper) are both exported. `bmm` dispatches
to cuBLAS `SgemmStridedBatched` / `DgemmStridedBatched` on CUDA for
f32/f64. Non-test production consumer: `pub fn bmm` (the `Tensor::bmm`
method) in `ferrotorch-core/src/methods.rs`;
`crate::grad_fns::linalg::bmm_differentiable` invocations in
`ferrotorch-core/src/flex_attention.rs` (`scores = bmm_differentiable(q,
k_t)`; `output = bmm_differentiable(weights, v)`);
`use ferrotorch_core::grad_fns::linalg::{bmm_differentiable,
mm_differentiable}` in `ferrotorch-nn/src/attention.rs`; and the
forward-only `crate::grad_fns::linalg::bmm` invocations in
`ferrotorch-core/src/einsum.rs` (two-input batched matmul path).

### REQ-3 matmul

The `pub struct MatmulBackward` saves `a` and `b`. The backward
dispatches on `(a.ndim(), b.ndim())` to the rank-appropriate inner
backward — `MmBackward::new(a, b).backward()` for 2D×2D,
`MvBackward::new(a, b).backward()` for 2D×1D,
`DotBackward::new(a, b).backward()` for 1D×1D, an inline vm path for
1D×2D (with GPU `mv_f32`/`vm_f32` + transpose), and
`broadcast_matmul_backward` for everything else. The vm GPU branch
constructs `dA = mv(B, grad_y)` and `dB = outer(a, grad_y)` (the latter
as a rank-1 matmul `matmul((K,1), (1,N))`).

The forward `pub fn matmul_differentiable` handles all rank
combinations, with GPU paths for: 1D×2D (`vm_f32`/`vm_f64`), 2D×2D
(`matmul_f32`/`_f64`/`matmul_bf16_bf16`/`matmul_f16_f16`, with
autocast → `matmul_f16_f32`), and ≥2D broadcast bmm
(`broadcast_bmm_f32`/`_f64` via cuBLAS gemmStridedBatched, stride=0 on
broadcasted axes). All other rank combinations delegate to
`linalg::matmul` (CPU). The `broadcast_matmul_backward` helper performs
the transpose-of-last-two-dims + reduce-to-target-shape sequence
required when batch dims broadcast.

Non-test production consumer: `pub fn matmul` (the `Tensor::matmul`
method) in `ferrotorch-core/src/methods.rs`;
`use ferrotorch_core::grad_fns::linalg::matmul_differentiable` in
`ferrotorch-vision/src/models/swin.rs` (Swin transformer attention);
multiple `crate::grad_fns::linalg::matmul_differentiable` invocations
in `ferrotorch-core/src/einsum.rs` (the two-input matmul branch of
einsum); the forward-AD primal in
`ferrotorch-core/src/autograd/forward_ad.rs` (`primal = matmul_diff(a,
b); term1 = matmul_diff(a.tangent, b.primal); term2 = matmul_diff(
a.primal, b.tangent)`).

### REQ-4 linalg.matmul

Aliased to REQ-3 by upstream design: `Tensor linalg_matmul(const Tensor
& tensor1, const Tensor & tensor2)` at
`aten/src/ATen/native/LinearAlgebra.cpp:2206` is literally `return
at::matmul(tensor1, tensor2)`. The ferrotorch consumer is the same as
REQ-3: any caller of `Tensor::matmul` or
`grad_fns::linalg::matmul_differentiable`. No separate `pub fn
linalg_matmul` exists in ferrotorch — the Python-API alias is provided
by `Tensor::matmul` itself.

### REQ-5..REQ-9 addmm/addbmm/baddbmm/addmv/addr (NOT-STARTED end-to-end)

The fused `BLAS-3` family `C = beta * self + alpha * mat1 @ mat2`
(addmm), the batched-sum `addbmm`, the per-batch `baddbmm`, the matrix-
vector `addmv`, and the rank-1-outer `addr` now have `*Backward`
`GradFn` structs and `*_differentiable` wrappers in
`ferrotorch-core/src/grad_fns/linalg.rs` (`AddmmBackward`,
`AddbmmBackward`, `BaddbmmBackward`, `AddmvBackward`, `AddrBackward`).
However, none of them is wired into a non-test production consumer:
there is no public `addmm` / `addbmm` / `baddbmm` / `addmv` / `addr`
forward (nor a `Tensor` method) that delegates to the wrapper, so the
only callers are the parity-sweep runner (test-side, does not count per
R-DOC-4) and the in-file `#[cfg(test)] mod tests`. The fused
`pub fn linear_fused` is a related-but-distinct op: it is hard-coded to
`A @ W^T + bias` (`beta=1, alpha=1`, bias instead of `self`, weight
transposed) and does not expose the general `addmm` API. The
consumer-wiring gap is tracked by blocker #1583.

### REQ-10..REQ-28 torch.linalg.* factorization family

The closed-form-VJP `*Backward` structs that are wired end-to-end:
- `linalg.qr` (reduced, m≥n; `QrBackwardQ`/`QrBackwardR`),
  `linalg.cholesky` (`CholeskyBackward`, Phi-symmetrisation), and
  `linalg.slogdet` (`SlogdetBackward`, real-case `g_abs * A^{-T}`) —
  landed 2026-05-27 and **SHIPPED**: the grad-aware forwards `pub fn qr`,
  `pub fn cholesky`, and `pub fn slogdet` in
  `ferrotorch-core/src/linalg.rs` (the `torch.linalg.*` public surface)
  delegate to the matching `*_differentiable` wrapper in this file when
  `is_grad_enabled() && input.requires_grad()`. The wrapper computes the
  underlying factorization inside a `no_grad` block (to prevent re-entry
  into the grad-aware forward) and attaches the `*Backward` `GradFn`.
  That grad-aware forward is the non-test production consumer.
- `linalg.solve` (`LinalgSolveBackward` + `solve_differentiable`),
  `linalg.inv` (`LinalgInvBackward` + `inv_differentiable`), and
  `linalg.det` (`LinalgDetBackward` + `det_differentiable`) — landed
  2026-05-27 (closing #1583's solve/inv/det subset) and **SHIPPED** the
  same way: the forwards `pub fn solve` / `pub fn inv` / `pub fn det` in
  `ferrotorch-core/src/linalg.rs` now delegate to the matching
  `*_differentiable` wrapper (forward computed under `no_grad`). Each VJP
  is FD-verified by a public-forward-driven test in this file's
  `#[cfg(test)] mod tests` (`solve_forward_is_grad_aware_and_matches_fd_*`,
  `inv_forward_is_grad_aware_and_matches_fd`,
  `det_forward_is_grad_aware_and_matches_fd`). `det_differentiable`'s
  internal `inv` (for the VJP) is also computed under `no_grad`.

The closed-form-VJP `*Backward` structs that exist but are NOT yet wired:
- the fused-affine family (`addmm`/`addbmm`/`baddbmm`/`addmv`/`addr`) and
  the structural `kron`/`diag`/`tril`/`triu`/`diagonal` — their forwards
  are not grad-aware (or, for `addmm`-family and `kron`, no public forward
  exists at all). The only caller of those `*_differentiable` wrappers is
  the parity-sweep runner (test-side). These are NOT-STARTED end-to-end
  pending the remaining scope of blocker #1583 (grad-aware forward wiring /
  public forward, mirroring the qr/cholesky/slogdet and
  solve/inv/det/trace/outer pattern above).

The QR multi-output case is handled by SPLITTING the jointly-linear
`(gQ, gR)` VJP across two single-output nodes — `QrBackwardQ` on the `Q`
output (the `gQ`-only partial) and `QrBackwardR` on the `R` output (the
`gR`-only partial). The reverse-mode engine accumulates both partials
into `A.grad`, reproducing the joint formula (which is additive in `gQ`
and `gR`); a consumer that uses only one output simply gets the other
partial as zero, matching PyTorch's undefined-grad semantics. Slogdet
likewise attaches its node only to the differentiable `logabsdet`
output, leaving `sign` plain (non-differentiable in the real case).

The remaining factorizations — `linalg.svd`, `linalg.eig` / `eigh` /
`eigvals` / `eigvalsh`, `linalg.pinv`, `linalg.lstsq`, `linalg.norm`,
`linalg.matrix_rank`, `linalg.cross`, `linalg.householder_product`,
`linalg.lu`, `linalg.lu_factor` — remain forward-only in
`ferrotorch-core/src/linalg.rs` routed through `ferray_linalg::*`, with
no `*Backward` `GradFn` at all. Autograd through several requires
research-grade VJPs (degenerate eigenvalues, gauge freedom — e.g. SVD's
F-matrix formula `dA = U (F ∘ (U^T dU - dU^T U)/2 S + dS) Vh + ...`). The
degenerate / gauge-freedom subset (svd/eigh/eigvalsh/pinv/lstsq/lu/
lu_factor) is tracked under sub-blocker #1577; the rest
(eig/eigvals/norm/matrix_rank/cross/householder_product) stay
NOT-STARTED under #1345.

### REQ-29..REQ-35 trace/diagonal/diag/tril/triu/kron/outer

`trace` (REQ-29) and `outer` (REQ-35) are **SHIPPED** end-to-end as of
2026-05-27: their forwards `pub fn trace` / `pub fn outer` in
`ferrotorch-core/src/linalg.rs` delegate to `trace_differentiable` /
`outer_differentiable` (forward computed under `no_grad`) when grad is
enabled, and each VJP is FD-verified by a public-forward-driven test in
this file's `#[cfg(test)] mod tests`. The grad-aware forward is the
non-test production consumer.

The remainder (`diagonal`/`diag`/`tril`/`triu`/`kron`) still has a
`*Backward` `GradFn` + `*_differentiable` wrapper in
`ferrotorch-core/src/grad_fns/linalg.rs`, but is NOT wired into a non-test
production consumer (the forwards are not grad-aware and there is no
`Tensor` method delegating to the wrapper):

- `diagonal`: forward-only `pub fn diagonal` in
  `ferrotorch-core/src/linalg.rs` + `DiagonalBackward` /
  `diagonal_differentiable`. Blocker #1583.
- `diag`: forward-only `pub fn diag` in
  `ferrotorch-core/src/ops/tensor_ops.rs` + `DiagBackward` /
  `diag_differentiable`. Blocker #1583.
- `tril`: forward-only `pub fn tril` in
  `ferrotorch-core/src/ops/tensor_ops.rs` + `TriangularBackward` /
  `tril_differentiable`. Blocker #1583.
- `triu`: forward-only `pub fn triu` in
  `ferrotorch-core/src/ops/tensor_ops.rs` + `triu_differentiable`
  (sharing `TriangularBackward`). Blocker #1583.
- `kron`: `KronBackward` / `kron_differentiable` exist, but there is NO
  public `pub fn kron` forward anywhere in ferrotorch-core src/. Blocker
  #1583.

Grad-aware forward wiring for the remaining REQ-30..34
(diagonal/diag/tril/triu/kron) is tracked by #1583.

## Parity contract

| Op | Upstream entry | Backward formula source | Edge cases mirrored |
|---|---|---|---|
| `mm` | `TORCH_IMPL_FUNC(mm_out_cpu)` in `aten/src/ATen/native/LinearAlgebra.cpp` | `dA = grad_C @ B^T`, `dB = A^T @ grad_C` | Inner-dim mismatch → `FerrotorchError::ShapeMismatch`; bf16/f16 on GPU use cuBLAS GemmEx with f32 accumulator; autocast f32 → f16 accumulator path; CPU is zero-copy raw-slice loops. SHIPPED (REQ-1). |
| `bmm` | `TORCH_IMPL_FUNC(bmm_out_cpu)` in `aten/src/ATen/native/LinearAlgebra.cpp` | per-batch `mm` VJP composed via `batch_transpose` | 3D inputs only (Err on other ranks); batch-dim mismatch → `ShapeMismatch`; CUDA via `SgemmStridedBatched`/`DgemmStridedBatched`; autocast f32→ `bmm_f16_f32` Tensor Core path. SHIPPED (REQ-2). |
| `matmul` | `Tensor matmul(...)` in `aten/src/ATen/native/LinearAlgebra.cpp` | rank-dispatched: dot/mv/vm/mm/bmm/broadcast_bmm | 1D×1D=dot; 2D×1D=mv; 1D×2D=vm; 2D×2D=mm; 3D×3D=bmm; broadcast ≥3D=gemmStridedBatched with stride-0 on broadcasted axes. CUDA covers all of these for f32/f64; bf16/f16 covered for 2D×2D; other dtype combos err `NotImplementedOnCuda`. SHIPPED (REQ-3). |
| `linalg.matmul` | `Tensor linalg_matmul(...)` in `aten/src/ATen/native/LinearAlgebra.cpp` | alias for `matmul` | upstream `linalg_matmul` is literally `at::matmul(tensor1, tensor2)`; ferrotorch's `Tensor::matmul` covers both since the Python API surface is the same. SHIPPED (REQ-4). |
| `addmm` | `TORCH_IMPL_FUNC(addmm_out_cpu)` in `aten/src/ATen/native/LinearAlgebra.cpp` | dself=beta·grad, dmat1=alpha·grad·mat2^T, dmat2=alpha·mat1^T·grad | `AddmmBackward` impl exists; NOT-STARTED end-to-end (no production consumer). Blocker #1583. |
| `addbmm` | `Tensor addbmm(...)` in `aten/src/ATen/native/LinearAlgebra.cpp` | dself=beta·grad, dbatch1=alpha·grad·batch2^T (broadcast over batch), dbatch2=alpha·batch1^T·grad (sum over batch) | `AddbmmBackward` impl exists; NOT-STARTED end-to-end. Blocker #1583. |
| `baddbmm` | `TORCH_IMPL_FUNC(baddbmm_out_cpu)` in `aten/src/ATen/native/LinearAlgebra.cpp` | per-batch addmm-like VJP | `BaddbmmBackward` impl exists; NOT-STARTED end-to-end. Blocker #1583. |
| `addmv` | `TORCH_IMPL_FUNC(addmv_out_cpu)` in `aten/src/ATen/native/Blas.cpp` | dself=beta·grad, dmat=alpha·outer(grad,vec), dvec=alpha·mat^T·grad | `AddmvBackward` impl exists; NOT-STARTED end-to-end. Blocker #1583. |
| `addr` | `Tensor addr(...)` in `aten/src/ATen/native/LinearAlgebra.cpp` | dself=beta·grad, dvec1=alpha·grad@vec2, dvec2=alpha·vec1^T@grad | `AddrBackward` impl exists; NOT-STARTED end-to-end. Blocker #1583. |
| `linalg.solve` | `Tensor linalg_solve(...)` in `aten/src/ATen/native/BatchLinearAlgebra.cpp` | `gB = A^{-T} @ gX`, `gA = -gB @ X^T` (`linalg_solve_backward` in `FunctionsManual.cpp`) | SHIPPED (REQ-10): `LinalgSolveBackward` + grad-aware `pub fn solve` forward delegating to `solve_differentiable`. FD-verified (matrix + vector RHS). |
| `linalg.svd` | `svd = _add_docstr(...)` in `torch/linalg/__init__.py` | F-matrix formula with U/S/Vh | No `GradFn`; forward-only. NOT-STARTED. Blocker #1577. |
| `linalg.eig` | `eig = _add_docstr(...)` in `torch/linalg/__init__.py` | F-matrix formula with eigenvalue spacing | No `GradFn`; forward-only. NOT-STARTED. Blocker #1345. |
| `linalg.eigh` | `eigh = _add_docstr(...)` in `torch/linalg/__init__.py` | sym F-matrix (real spectrum) | No `GradFn`; forward-only. NOT-STARTED. Blocker #1577. |
| `linalg.eigvals` | `eigvals = _add_docstr(...)` in `torch/linalg/__init__.py` | derived from `eig` VJP, summed over eigvecs | No `GradFn`; forward-only. NOT-STARTED. Blocker #1345. |
| `linalg.eigvalsh` | `eigvalsh = _add_docstr(...)` in `torch/linalg/__init__.py` | sym variant | No `GradFn`; forward-only. NOT-STARTED. Blocker #1577. |
| `linalg.qr` | `qr = _add_docstr(...)` in `torch/linalg/__init__.py` | Q-orthog + R-triangular VJP (reduced, m≥n) | SHIPPED (REQ-16): `QrBackwardQ`/`QrBackwardR` + grad-aware `pub fn qr` forward. m<n branch tracked under #1577. |
| `linalg.cholesky` | `Tensor linalg_cholesky(...)` in `aten/src/ATen/native/BatchLinearAlgebra.cpp` | `gA = L^{-T} Φ(tril(L^T gL)) L^{-1}` | SHIPPED (REQ-17): `CholeskyBackward` + grad-aware `pub fn cholesky` forward. |
| `linalg.inv` | `Tensor linalg_inv(...)` in `aten/src/ATen/native/BatchLinearAlgebra.cpp` | `dA = −Y^T @ grad @ Y^T` where Y = A^{-1} | SHIPPED (REQ-18): `LinalgInvBackward` + grad-aware `pub fn inv` forward delegating to `inv_differentiable`. FD-verified. |
| `linalg.pinv` | `Tensor linalg_pinv(...)` in `aten/src/ATen/native/LinearAlgebra.cpp` | full pinv VJP formula | No `GradFn`; forward-only. NOT-STARTED. Blocker #1577. |
| `linalg.det` | `Tensor linalg_det(...)` in `aten/src/ATen/native/LinearAlgebra.cpp` | `dA = det(A) · A^{-T} · grad` | SHIPPED (REQ-20): `LinalgDetBackward` + grad-aware `pub fn det` forward delegating to `det_differentiable`. FD-verified. |
| `linalg.slogdet` | `slogdet = _add_docstr(...)` in `torch/linalg/__init__.py` | `dA = A^{-T} · grad_logabs` (sign-grad is 0) | SHIPPED (REQ-21): `SlogdetBackward` + grad-aware `pub fn slogdet` forward. |
| `linalg.lstsq` | `lstsq = _add_docstr(...)` in `torch/linalg/__init__.py` | least-squares VJP via QR | No `GradFn`; forward-only. NOT-STARTED. Blocker #1577. |
| `linalg.norm` | `norm = _add_docstr(...)` in `torch/linalg/__init__.py` | per-ord VJP (Frobenius: `dA = A / ||A||_F · grad`) | No `GradFn`; forward-only. NOT-STARTED. Blocker #1345. |
| `linalg.matrix_rank` | `Tensor linalg_matrix_rank(...)` in `aten/src/ATen/native/LinearAlgebra.cpp` | rank is integer; backward returns zero | No `GradFn`; forward-only. NOT-STARTED. Blocker #1345. |
| `linalg.cross` | `cross = _add_docstr(...)` in `torch/linalg/__init__.py` | `da = b × grad`, `db = grad × a` | No `GradFn`; forward-only. NOT-STARTED. Blocker #1345. |
| `linalg.householder_product` | `Tensor linalg_householder_product(...)` in `aten/src/ATen/native/BatchLinearAlgebra.cpp` | VJP through Householder reflectors | No `GradFn`; forward-only. NOT-STARTED. Blocker #1345. |
| `linalg.lu` | `lu = _add_docstr(...)` in `torch/linalg/__init__.py` | P/L/U VJP | No `GradFn`; forward-only. NOT-STARTED. Blocker #1577. |
| `linalg.lu_factor` | `lu_factor = _add_docstr(...)` in `torch/linalg/__init__.py` | same as `lu` minus the explicit unpacking | No `GradFn`; forward-only. NOT-STARTED. Blocker #1577. |
| `trace` | upstream tensor method (no dedicated impl in `LinearAlgebra.cpp`) | `dA = grad · I` (`trace_backward_symint` in `derivatives.yaml`) | SHIPPED (REQ-29): `TraceBackward` + grad-aware `pub fn trace` forward delegating to `trace_differentiable`. FD-verified. |
| `diagonal` | `Tensor linalg_diagonal(...)` in `aten/src/ATen/native/LinearAlgebra.cpp` | inverse of `diag_embed` — scatter grad onto the diagonal of zeros | `DiagonalBackward` impl exists, but `pub fn diagonal` forward is not grad-aware. NOT-STARTED end-to-end. Blocker #1583. |
| `diag` | upstream tensor method `Tensor::diag(...)` | scatter or extract VJP based on input rank | `DiagBackward` impl exists, but `pub fn diag` forward (in `ops/tensor_ops.rs`) is not grad-aware. NOT-STARTED end-to-end. Blocker #1583. |
| `tril` | upstream tensor method `Tensor::tril(...)` | grad masked by lower triangular | `TriangularBackward` impl exists, but `pub fn tril` forward (in `ops/tensor_ops.rs`) is not grad-aware. NOT-STARTED end-to-end. Blocker #1583. |
| `triu` | upstream tensor method `Tensor::triu(...)` | grad masked by upper triangular | `triu_differentiable` impl exists (shares `TriangularBackward`), but `pub fn triu` forward (in `ops/tensor_ops.rs`) is not grad-aware. NOT-STARTED end-to-end. Blocker #1583. |
| `kron` | `Tensor kron(...)` in `aten/src/ATen/native/LinearAlgebra.cpp` | `dA = sum over kron-blocks of grad·B^T`, `dB = sum over kron-blocks of A^T·grad` | `KronBackward` impl exists, but no public `kron` forward exists. NOT-STARTED end-to-end. Blocker #1583. |
| `outer` | `Tensor outer(...)` in `aten/src/ATen/native/LinearAlgebra.cpp` | `da = grad @ b`, `db = grad^T @ a` | SHIPPED (REQ-35): `OuterBackward` + grad-aware `pub fn outer` forward delegating to `outer_differentiable`. FD-verified. |

Parity-sweep audit reference: only the four matmul-family ops (`mm`,
`bmm`, `matmul`, `linalg.matmul`) have runner arms in
`tools/parity-sweep/runner/src/main.rs`'s `dispatch_f32` (the
SHIPPED-with-real-consumer set). The remaining linalg ops either have no
runner arm at all, or (for the GradFn-bearing-but-unwired set — addmm
family, solve/inv/det, trace/outer/kron/diag/tril/triu/diagonal) are
called ONLY from the runner's dispatch table, which is a TEST-side
consumer and does not count toward a SHIPPED claim per R-DOC-4. The
runner-arm wiring for the whole linalg family is the test-infrastructure
umbrella blocker #1344; the missing production consumers are blocker
#1583.

## Verification

### Existing unit tests (all passing)

Located in the `#[cfg(test)] mod tests` block at the bottom of
`ferrotorch-core/src/grad_fns/linalg.rs` (~690 LOC of tests starting at
line 1827). Key test functions in `linalg.rs`:

- **mm backward**: `fn test_mm_backward_both_grads` (verifies both
  `grad_A` and `grad_B` against closed-form values),
  `fn test_mm_backward_one_requires_grad` (verifies the `None`
  short-circuit when only one operand carries grad).
- **dot backward**: `fn test_dot_backward` (verifies `ds/da = b`,
  `ds/db = a` for a length-3 dot product), `fn
  test_dot_backward_one_requires_grad` (single-operand grad path).
- **mv backward**: `fn test_mv_backward` (verifies `dA = outer(grad_y,
  x)` and `dx = A^T @ grad_y` for a 2x2 case).
- **matmul backward dispatch**: `fn test_matmul_backward_dispatches_to_dot`
  (1D×1D rank dispatch), `fn test_matmul_backward_dispatches_to_mm`
  (2D×2D rank dispatch), plus the bmm-dispatch test for 3D×3D inputs.
- **bmm backward**: bmm tests verifying per-batch `dA[b]` and `dB[b]`
  against closed-form values; `batch_transpose` helper tests.
- **broadcast matmul**: tests verifying `broadcast_matmul_backward` for
  4D × 4D, 3D × 2D, and 2D × 3D shapes with batch broadcasting.
- **linear_fused**: tests verifying `grad_input`, `grad_weight`, and
  `grad_bias` from a single fused backward against the decomposed
  `mm + add` route.
- **mm_bt**: tests verifying `MmBtBackward` against the equivalent
  composed `mm(A, transpose(B))` route.
- **permute_0213**: tests verifying the 4D `[B,S,H,D] → [B,H,S,D]`
  reshape against a direct index-mapping loop.

### Parity-sweep status

All four matmul-family ops gained runner arms by 2026-05-26 (`mm` /
`bmm` on 2026-05-25 closing #1344; `matmul` / `linalg.matmul` on
2026-05-26 closing #1347). `dispatch_f32` in
`tools/parity-sweep/runner/src/main.rs` now contains arms for `"mm"`,
`"bmm"`, `"matmul"`, and `"linalg.matmul"`, each decoding
`args = [tensor_f32, tensor_f32]` via the existing `binary()` helper
and dispatching through `grad_fns::linalg::{mm,bmm,matmul}_differentiable`
(the matmul arm is shared between `matmul` and `linalg.matmul` since
upstream `linalg_matmul` is literally `return at::matmul(tensor1,
tensor2)`; the oracle alias `oracle_name("linalg.matmul") -> "matmul"`
shares op_db's `matmul` sample set). Verified 2026-05-26 with
`parity-sweep sweep --op <op> --seeds 8`:

```
[mm]              24/24 passed (0 skipped, 0 failed)   smoke grep count = 1
[bmm]              8/8  passed (0 skipped, 0 failed)   smoke grep count = 1
[matmul]         120/120 passed (0 skipped, 0 failed)  smoke grep count = 1
[linalg.matmul]  120/120 passed (0 skipped, 0 failed)  smoke grep count = 1
```

**Tolerance contract**: matmul-family ops are evaluated at `rtol=1e-4`
(per-op override in `fn tolerance_for in
tools/parity-sweep/runner/src/main.rs`), widened from the default
`rtol=1e-5`. This acknowledges the structural cross-BLAS-implementation
f32 ULP variance: ferrotorch's CPU matmul uses faer (pure Rust BLAS
under `Cargo.toml line 51`) while PyTorch uses MKL — no two CPU BLAS
implementations agree at f32 ULP for k>=10 inner dims, since each
implementation reduces dot-product accumulators in a different order
producing different f32 rounds. Empirically verified 2026-05-26 on
op_db sample `matmul seed=7 i=6` cell `out[2,1,1]` of `[5,5,10]@[10,5]`:
torch (MKL) = `0.13889313`, ferrotorch (faer) = `0.13889723`, diff =
`~4e-6` at `|e|=0.14` — well within `rtol=1e-4` envelope but exceeds
the default `rtol=1e-5`. The CPU broadcast / bmm fallback paths now
consolidate through `pub fn mm_raw in ops/linalg.rs` (faer-backed) for
all four ops, so the ULP variance is consistent across the family.
Byte-for-byte parity vs MKL requires the future-epic MKL/OpenBLAS FFI
path (separate blocker filed at `low` priority).

The remaining 31 NOT-STARTED linalg ops still report
`N skipped (runner has no arm)` and are tracked under prereq blocker
#1345 (those ops require new `*Backward` `GradFn` impls in
`grad_fns/linalg.rs` before they can be wired). Per goal.md S5: missing
runner arms for NOT-STARTED ops are a TEST-INFRASTRUCTURE gap, not a
REQ blocker.

### Cargo test command

```
cargo test -p ferrotorch-core grad_fns::linalg
```

All forward and backward tests pass at residual `< 1e-5` (closed-form
expectations).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (mm) | SHIPPED | impl: `pub fn mm_differentiable` + `pub struct MmBackward` + `fn mm_backward_gpu` helper in `ferrotorch-core/src/grad_fns/linalg.rs` mirroring `TORCH_IMPL_FUNC(mm_out_cpu)` at `aten/src/ATen/native/LinearAlgebra.cpp:1641`; non-test production consumer: `pub fn mm` (the `Tensor::mm` method) in `ferrotorch-core/src/methods.rs`, `use ferrotorch_core::grad_fns::linalg::mm_differentiable` in `ferrotorch-nn/src/functional.rs` (linear composite), `ferrotorch-nn/src/attention.rs` (Q/K/V projection), `ferrotorch-nn/src/lora.rs` (LoRA adapters), `ferrotorch-nn/src/rnn.rs` (RNN gates), and `ferrotorch-core/src/grad_fns/shape.rs` (row-sums helper). Runner arm wired at `"mm"` arm in `dispatch_f32` in `tools/parity-sweep/runner/src/main.rs` (closes #1344 for mm; smoke `24/24 passed, 0 failed` at seeds=8 on 2026-05-25). |
| REQ-2 (bmm) | SHIPPED | impl: `pub fn bmm_differentiable` + `pub fn bmm` (forward) + `pub struct BmmBackward` + `fn batch_transpose` in `ferrotorch-core/src/grad_fns/linalg.rs` mirroring `TORCH_IMPL_FUNC(bmm_out_cpu)` at `aten/src/ATen/native/LinearAlgebra.cpp:1894`; non-test consumer: `pub fn bmm` (the `Tensor::bmm` method) in `ferrotorch-core/src/methods.rs`, `crate::grad_fns::linalg::bmm_differentiable` in `ferrotorch-core/src/flex_attention.rs` (attention scores + output), `use ferrotorch_core::grad_fns::linalg::{bmm_differentiable, mm_differentiable}` in `ferrotorch-nn/src/attention.rs`, and `crate::grad_fns::linalg::bmm` (forward-only) in `ferrotorch-core/src/einsum.rs`. Runner arm wired at `"bmm"` arm in `dispatch_f32` in `tools/parity-sweep/runner/src/main.rs` (closes #1344 for bmm; smoke `8/8 passed, 0 failed` at seeds=8 on 2026-05-25). |
| REQ-3 (matmul) | SHIPPED | impl: `pub fn matmul_differentiable` + `pub struct MatmulBackward` in `ferrotorch-core/src/grad_fns/linalg.rs` mirroring `Tensor matmul(const Tensor & tensor1, const Tensor & tensor2)` at `aten/src/ATen/native/LinearAlgebra.cpp:2190`. Non-test production consumers: `pub fn matmul` (the `Tensor::matmul` method) in `ferrotorch-core/src/methods.rs`, `use ferrotorch_core::grad_fns::linalg::matmul_differentiable` in `ferrotorch-vision/src/models/swin.rs` (Swin attention), `crate::grad_fns::linalg::matmul_differentiable` in `ferrotorch-core/src/einsum.rs` (two-input matmul branch of einsum), and `ferrotorch-core/src/autograd/forward_ad.rs` (forward-AD primal `matmul_diff(a, b)`). The CPU broadcast / 3D-x-2D / 4D paths now route per-batch slabs through `fn broadcast_matmul in ops/linalg.rs` (the per-batch loop dispatches to `pub fn mm_raw in ops/linalg.rs`, faer-backed), consolidating accumulation behavior with `mm` and `bmm`. Runner arm wired at `"matmul"` arm in `dispatch_f32` in `tools/parity-sweep/runner/src/main.rs` (closes #1347; smoke `120/120 passed, 0 failed` at seeds=8 on 2026-05-26). **Tolerance: `rtol=1e-4` for matmul-family ops (cross-BLAS ULP variance; ferrotorch=faer vs torch=MKL)**; verified 2026-05-26 the structural f32 ULP drift `~4e-6 at \|e\|=0.14` lands well within the widened envelope; byte-for-byte parity vs MKL requires the MKL/OpenBLAS FFI follow-up (future epic). The per-op tolerance override lives in `fn tolerance_for in tools/parity-sweep/runner/src/main.rs`. |
| REQ-4 (linalg.matmul) | SHIPPED | aliased to REQ-3 (`Tensor linalg_matmul(const Tensor & tensor1, const Tensor & tensor2)` at `aten/src/ATen/native/LinearAlgebra.cpp:2206` is literally `return at::matmul(tensor1, tensor2)`). Same `matmul_differentiable` impl in `ferrotorch-core/src/grad_fns/linalg.rs`; same non-test production consumers as REQ-3 — any call to `Tensor::matmul` satisfies the Python-API alias per goal.md R-DEV-2. op_db does NOT register `linalg.matmul` as a separate entry (verified 2026-05-26 via `parity-sweep list-ops | grep linalg.m` — only matrix_norm/matrix_power/matrix_rank/multi_dot appear), so the parity-sweep runner aliases the bare `linalg.matmul` route name to `matmul` via `fn oracle_name in tools/parity-sweep/runner/src/main.rs`. Runner arm wired at `"linalg.matmul"` arm in `dispatch_f32` (closes #1347; smoke `120/120 passed, 0 failed` at seeds=8 on 2026-05-26). **Tolerance: `rtol=1e-4` for matmul-family ops (cross-BLAS ULP variance; ferrotorch=faer vs torch=MKL)** — same envelope as REQ-3 (see `fn tolerance_for in tools/parity-sweep/runner/src/main.rs`); byte-for-byte parity vs MKL requires the MKL/OpenBLAS FFI follow-up (future epic). |
| REQ-5 (addmm) | NOT-STARTED | open prereq blocker #1583. Impl exists: `pub struct AddmmBackward` + `pub fn addmm_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs` (VJP `dself=beta·grad`, `dmat1=alpha·grad·mat2^T`, `dmat2=alpha·mat1^T·grad`, mirroring `TORCH_IMPL_FUNC(addmm_out_cpu)` in `LinearAlgebra.cpp`). But there is NO non-test production consumer: no `pub fn addmm` public forward exists in `ferrotorch-core/src/linalg.rs` or `ferrotorch-core/src/methods.rs`, and the only callers of `addmm_differentiable` are the parity-sweep runner (test-side, does not count) and the in-file `#[cfg(test)] mod tests`. Cannot be SHIPPED end-to-end per R-DOC-4 until a grad-aware forward delegates to it. |
| REQ-6 (addbmm) | NOT-STARTED | open prereq blocker #1583. Impl exists: `pub struct AddbmmBackward` + `pub fn addbmm_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs` (mirroring `Tensor addbmm(...)` in `LinearAlgebra.cpp`), but no non-test production consumer (no public `addbmm` forward; only the parity-sweep runner + in-file tests call `addbmm_differentiable`). |
| REQ-7 (baddbmm) | NOT-STARTED | open prereq blocker #1583. Impl exists: `pub struct BaddbmmBackward` + `pub fn baddbmm_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs` (mirroring `TORCH_IMPL_FUNC(baddbmm_out_cpu)` in `LinearAlgebra.cpp`), but no non-test production consumer. |
| REQ-8 (addmv) | NOT-STARTED | open prereq blocker #1583. Impl exists: `pub struct AddmvBackward` + `pub fn addmv_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs` (mirroring `TORCH_IMPL_FUNC(addmv_out_cpu)` in `Blas.cpp`), but no non-test production consumer. |
| REQ-9 (addr) | NOT-STARTED | open prereq blocker #1583. Impl exists: `pub struct AddrBackward` + `pub fn addr_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs` (mirroring `Tensor addr(...)` in `LinearAlgebra.cpp`), but no non-test production consumer. |
| REQ-10 (linalg.solve) | SHIPPED | impl: `pub struct LinalgSolveBackward` + `pub fn solve_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs` (VJP `gB = A^{-T} @ gX`, `gA = -gB @ X^T`, mirroring `linalg_solve_backward` at `torch/csrc/autograd/FunctionsManual.cpp:6160`). FD-verified by `fn solve_forward_is_grad_aware_and_matches_fd_matrix_rhs` + `fn solve_forward_is_grad_aware_and_matches_fd_vector_rhs` in the in-file `#[cfg(test)] mod tests` (both drive the public forward and check `A.grad`/`B.grad` vs central FD). Non-test production consumer: the grad-aware forward `pub fn solve` in `ferrotorch-core/src/linalg.rs` (the `torch.linalg.solve` public surface) delegates to `solve_differentiable` when `!a.is_cuda() && is_grad_enabled() && (a.requires_grad() || b.requires_grad())`; the wrapper computes the forward under `no_grad` to avoid re-entry. |
| REQ-11 (linalg.svd) | NOT-STARTED | open prereq blocker #1577 (research-grade degenerate-singular-value / gauge-freedom VJP). No `LinalgSvdBackward` `GradFn` in `ferrotorch-core/src/grad_fns/linalg.rs`; forward-only `pub fn svd` in `ferrotorch-core/src/linalg.rs`. Upstream `svd = _add_docstr(...)` in `torch/linalg/__init__.py`. |
| REQ-12 (linalg.eig) | NOT-STARTED | open prereq blocker #1345. No `LinalgEigBackward` `GradFn` in `ferrotorch-core/src/grad_fns/linalg.rs`; forward-only `pub fn eig` in `ferrotorch-core/src/linalg.rs`. Upstream `eig = _add_docstr(...)` in `torch/linalg/__init__.py`. |
| REQ-13 (linalg.eigh) | NOT-STARTED | open prereq blocker #1577 (research-grade degenerate-eigenvalue / gauge-freedom VJP). No `LinalgEighBackward` `GradFn` in `ferrotorch-core/src/grad_fns/linalg.rs`; forward-only `pub fn eigh` in `ferrotorch-core/src/linalg.rs`. Upstream `eigh = _add_docstr(...)` in `torch/linalg/__init__.py`. |
| REQ-14 (linalg.eigvals) | NOT-STARTED | open prereq blocker #1345. No `LinalgEigvalsBackward` `GradFn` in `ferrotorch-core/src/grad_fns/linalg.rs`; forward-only `pub fn eigvals` in `ferrotorch-core/src/linalg.rs`. Upstream `eigvals = _add_docstr(...)` in `torch/linalg/__init__.py`. |
| REQ-15 (linalg.eigvalsh) | NOT-STARTED | open prereq blocker #1577 (research-grade degenerate-eigenvalue subset). No `LinalgEigvalshBackward` `GradFn` in `ferrotorch-core/src/grad_fns/linalg.rs`; forward-only `pub fn eigvalsh` in `ferrotorch-core/src/linalg.rs`. Upstream `eigvalsh = _add_docstr(...)` in `torch/linalg/__init__.py`. |
| REQ-16 (linalg.qr) | SHIPPED | impl: `pub struct QrBackwardQ` + `pub struct QrBackwardR` + `pub fn qr_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs` implement the real `linalg_qr_backward` VJP (reduced, m≥n: `gA = (Q @ syminvadj(triu(M)) + gQ) @ R^{-T}`, `M = gR @ R^T - Q^T @ gQ`), mirroring upstream `linalg_qr_backward` in `FunctionsManual.cpp` and `linalg_qr` in `derivatives.yaml`; the joint `(gQ,gR)` VJP is split across the two single-output nodes and accumulated into `A.grad`. FD-verified by `fn qr_backward_matches_finite_difference_square` + `fn qr_backward_q_only_and_r_only` in the in-file `#[cfg(test)] mod tests`. Non-test production consumer: the grad-aware forward `pub fn qr` in `ferrotorch-core/src/linalg.rs` (the `torch.linalg.qr` public surface) delegates to `qr_differentiable` when `is_grad_enabled() && input.requires_grad()`. m<n (`trilImInvAdjSkew`) branch tracked under sub-blocker #1577. |
| REQ-17 (linalg.cholesky) | SHIPPED | impl: `pub struct CholeskyBackward` + `pub fn cholesky_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs` implement the Phi-symmetrisation VJP `gA = L^{-T} Φ(tril(L^T gL)) L^{-1}`, mirroring upstream `cholesky_backward` in `FunctionsManual.cpp` and `cholesky` in `derivatives.yaml`; the two triangular solves reuse `pub fn solve_triangular` in `ferrotorch-core/src/linalg.rs`, and the returned gradient is symmetric (PyTorch contract). FD-verified by `fn cholesky_backward_matches_finite_difference` (symmetric-FD + symmetry assertion) in the in-file `#[cfg(test)] mod tests`. Non-test production consumer: the grad-aware forward `pub fn cholesky` in `ferrotorch-core/src/linalg.rs` (the `torch.linalg.cholesky` public surface) delegates to `cholesky_differentiable` when grad is enabled. |
| REQ-18 (linalg.inv) | SHIPPED | impl: `pub struct LinalgInvBackward` + `pub fn inv_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs` (VJP `dA = -Y^T @ grad @ Y^T`, `Y = A^{-1}`, mirroring `linalg_inv_ex` at `tools/autograd/derivatives.yaml:916`). FD-verified by `fn inv_forward_is_grad_aware_and_matches_fd` in the in-file `#[cfg(test)] mod tests` (drives the public forward, loss = sum(Y), checks `A.grad` vs central FD). Non-test production consumer: the grad-aware forward `pub fn inv` in `ferrotorch-core/src/linalg.rs` (the `torch.linalg.inv` public surface) delegates to `inv_differentiable` when `is_grad_enabled() && input.requires_grad()`; the wrapper computes the forward under `no_grad` to avoid re-entry. |
| REQ-19 (linalg.pinv) | NOT-STARTED | open prereq blocker #1577 (research-grade pseudoinverse VJP). No `LinalgPinvBackward` `GradFn` in `ferrotorch-core/src/grad_fns/linalg.rs`; forward-only `pub fn pinv` in `ferrotorch-core/src/linalg.rs`. Upstream `Tensor linalg_pinv(...)` in `LinearAlgebra.cpp`. |
| REQ-20 (linalg.det) | SHIPPED | impl: `pub struct LinalgDetBackward` + `pub fn det_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs` (VJP `dA = det(A) * grad * inv(A)^T`, the invertible branch of `linalg_det_backward` at `torch/csrc/autograd/FunctionsManual.cpp:4373`). FD-verified by `fn det_forward_is_grad_aware_and_matches_fd` in the in-file `#[cfg(test)] mod tests` (drives the public forward, checks `A.grad` vs central FD). Non-test production consumer: the grad-aware forward `pub fn det` in `ferrotorch-core/src/linalg.rs` (the `torch.linalg.det` public surface) delegates to `det_differentiable` when `is_grad_enabled() && input.requires_grad()`; the wrapper computes the forward (and the VJP's internal `inv`) under `no_grad` to avoid re-entry. |
| REQ-21 (linalg.slogdet) | SHIPPED | impl: `pub struct SlogdetBackward` + `pub fn slogdet_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs` attach the real-case VJP `dA = grad_logabsdet * inv(A)^T` to the differentiable `logabsdet` output (`sign` is non-differentiable in the real case, returned plain), mirroring upstream `slogdet_backward` in `FunctionsManual.cpp` and `_linalg_slogdet` in `derivatives.yaml`. FD-verified by `fn slogdet_backward_matches_finite_difference` in the in-file `#[cfg(test)] mod tests`. Non-test production consumer: the grad-aware forward `pub fn slogdet` in `ferrotorch-core/src/linalg.rs` (the `torch.linalg.slogdet` public surface) delegates to `slogdet_differentiable` when grad is enabled. |
| REQ-22 (linalg.lstsq) | NOT-STARTED | open prereq blocker #1577 (least-squares VJP via QR, rank-deficient subset is research-grade). No `LinalgLstsqBackward` `GradFn` in `ferrotorch-core/src/grad_fns/linalg.rs`; forward-only `pub fn lstsq` in `ferrotorch-core/src/linalg.rs`. Upstream `lstsq = _add_docstr(...)` in `torch/linalg/__init__.py`. |
| REQ-23 (linalg.norm) | NOT-STARTED | open prereq blocker #1345. No `LinalgNormBackward` `GradFn` in `ferrotorch-core/src/grad_fns/linalg.rs`; forward-only `pub fn matrix_norm` + `pub fn vector_norm` in `ferrotorch-core/src/linalg.rs`. Upstream `norm = _add_docstr(...)` in `torch/linalg/__init__.py`. |
| REQ-24 (linalg.matrix_rank) | NOT-STARTED | open prereq blocker #1345. No `LinalgMatrixRankBackward` `GradFn` in `ferrotorch-core/src/grad_fns/linalg.rs`; forward-only `pub fn matrix_rank` in `ferrotorch-core/src/linalg.rs` (rank is integer-valued — backward is identically zero). Upstream `Tensor linalg_matrix_rank(...)` in `LinearAlgebra.cpp`. |
| REQ-25 (linalg.cross) | NOT-STARTED | open prereq blocker #1345. No `LinalgCrossBackward` `GradFn` in `ferrotorch-core/src/grad_fns/linalg.rs`; forward-only `pub fn cross` in `ferrotorch-core/src/linalg.rs`. Upstream `cross = _add_docstr(...)` in `torch/linalg/__init__.py`. |
| REQ-26 (linalg.householder_product) | NOT-STARTED | open prereq blocker #1345. No `LinalgHouseholderProductBackward` `GradFn` in `ferrotorch-core/src/grad_fns/linalg.rs`; forward-only `pub fn householder_product` in `ferrotorch-core/src/linalg.rs`. Upstream `Tensor linalg_householder_product(...)` in `BatchLinearAlgebra.cpp`. |
| REQ-27 (linalg.lu) | NOT-STARTED | open prereq blocker #1577 (LU pivoting VJP, gauge-freedom subset). No `LinalgLuBackward` `GradFn` in `ferrotorch-core/src/grad_fns/linalg.rs`; forward-only `pub fn lu` in `ferrotorch-core/src/linalg.rs`. Upstream `lu = _add_docstr(...)` in `torch/linalg/__init__.py`. |
| REQ-28 (linalg.lu_factor) | NOT-STARTED | open prereq blocker #1577 (same LU VJP minus explicit unpacking). No `LinalgLuFactorBackward` `GradFn` in `ferrotorch-core/src/grad_fns/linalg.rs`; forward-only `pub fn lu_factor` in `ferrotorch-core/src/linalg.rs`. Upstream `lu_factor = _add_docstr(...)` in `torch/linalg/__init__.py`. |
| REQ-29 (trace) | SHIPPED | impl: `pub struct TraceBackward` + `pub fn trace_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs` (VJP `dA = grad * I`, mirroring `trace_backward_symint` at `tools/autograd/derivatives.yaml:1785`). FD-verified by `fn trace_forward_is_grad_aware_and_matches_fd` in the in-file `#[cfg(test)] mod tests` (drives the public forward, checks `A.grad` vs central FD). Non-test production consumer: the grad-aware forward `pub fn trace` in `ferrotorch-core/src/linalg.rs` delegates to `trace_differentiable` when `is_grad_enabled() && a.requires_grad()`; the wrapper computes the forward under `no_grad` to avoid re-entry. |
| REQ-30 (diagonal) | NOT-STARTED | open prereq blocker #1583. Impl exists: `pub struct DiagonalBackward` + `pub fn diagonal_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs`. But the forward `pub fn diagonal` in `ferrotorch-core/src/linalg.rs` is NOT grad-aware and there is no non-test production consumer of `diagonal_differentiable`. Upstream `Tensor linalg_diagonal(...)` in `LinearAlgebra.cpp`. |
| REQ-31 (diag) | NOT-STARTED | open prereq blocker #1583. Impl exists: `pub struct DiagBackward` + `pub fn diag_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs`. But the forward `pub fn diag` in `ferrotorch-core/src/ops/tensor_ops.rs` is NOT grad-aware and there is no non-test production consumer of `diag_differentiable`. Upstream tensor method `Tensor::diag(...)`. |
| REQ-32 (tril) | NOT-STARTED | open prereq blocker #1583. Impl exists: `pub struct TriangularBackward` + `pub fn tril_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs` (VJP = grad masked by the lower triangle). But the forward `pub fn tril` in `ferrotorch-core/src/ops/tensor_ops.rs` is NOT grad-aware and there is no non-test production consumer of `tril_differentiable`. Upstream tensor method `Tensor::tril(...)`. |
| REQ-33 (triu) | NOT-STARTED | open prereq blocker #1583. Impl exists: `pub fn triu_differentiable` (sharing `pub struct TriangularBackward`) in `ferrotorch-core/src/grad_fns/linalg.rs` (VJP = grad masked by the upper triangle). But the forward `pub fn triu` in `ferrotorch-core/src/ops/tensor_ops.rs` is NOT grad-aware and there is no non-test production consumer of `triu_differentiable`. Upstream tensor method `Tensor::triu(...)`. |
| REQ-34 (kron) | NOT-STARTED | open prereq blocker #1583. Impl exists: `pub struct KronBackward` + `pub fn kron_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs` (mirroring `Tensor kron(...)` in `LinearAlgebra.cpp`). But there is NO public `pub fn kron` forward anywhere in ferrotorch-core src/, so no non-test production consumer; only the parity-sweep runner + in-file tests call `kron_differentiable`. |
| REQ-35 (outer) | SHIPPED | impl: `pub struct OuterBackward` + `pub fn outer_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs` (VJP `da = grad @ b`, `db = grad^T @ a`, the unscaled `addr` vec1/vec2 case at `tools/autograd/derivatives.yaml:275-276`). FD-verified by `fn outer_forward_is_grad_aware_and_matches_fd` in the in-file `#[cfg(test)] mod tests` (drives the public forward, loss = sum(C), checks both `a.grad` and `b.grad` vs central FD). Non-test production consumer: the grad-aware forward `pub fn outer` in `ferrotorch-core/src/linalg.rs` delegates to `outer_differentiable` when `is_grad_enabled() && (a.requires_grad() || b.requires_grad())`; the wrapper computes the forward under `no_grad` to avoid re-entry. |
