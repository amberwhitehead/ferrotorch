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
surface via `torch/linalg/__init__.py`. The file is 2517 LOC and ships
**fused `*Backward` `GradFn` structs** for the four core matmul-family ops
(`mm`, `bmm`, `matmul`, `linalg.matmul`) plus three internal fused variants
(`mm_bt` = A @ B^T without materialising B^T, `linear_fused` = A @ W^T +
bias for the `nn::Linear` hot path, and `permute_0213` = the
attention-head reshape primitive). Of the four matmul-family ops,
`mm` and `bmm` are SHIPPED end-to-end (impl + production consumer +
lib tests + parity smoke `0 failed`); `matmul` and `linalg.matmul`
are likewise SHIPPED as of 2026-05-26 (closing #1347): the CPU
broadcast / 3D-x-2D / 4D paths (`ops::linalg::broadcast_matmul`) plus
the bmm CPU fallback (`grad_fns::linalg::bmm`'s CPU branch) now route
per-batch slabs through the faer-backed `ops::linalg::mm_raw`
workhorse, consolidating accumulation behavior across the family. The
runner's per-op `tolerance_for` returns `rtol=1e-4` for matmul-family
ops to acknowledge the structural cross-BLAS-implementation f32 ULP
variance (ferrotorch=faer vs torch=MKL — see Parity-sweep status
section below for the empirical measurement); byte-for-byte parity vs
MKL requires the MKL/OpenBLAS FFI follow-up (future epic). The
remaining 31 parity ops on the route's list — `addmm`, `addbmm`, `baddbmm`, `addmv`, `addr`, `trace`,
`diagonal`, `diag`, `tril`, `triu`, `kron`, `outer`, and the entire
`torch.linalg.*` factorization family (`solve`, `svd`, `eig`, `eigh`,
`eigvals`, `eigvalsh`, `qr`, `cholesky`, `inv`, `pinv`, `det`, `slogdet`,
`lstsq`, `norm`, `matrix_rank`, `cross`, `householder_product`, `lu`,
`lu_factor`) — either have **forward-only** implementations elsewhere
(`ferrotorch-core/src/linalg.rs`, `ferrotorch-core/src/ops/tensor_ops.rs`)
without a fused `GradFn` in this file, or are not yet implemented at all.
Those are NOT-STARTED from the grad_fns/linalg.rs perspective and tracked
by prereq blocker #1345.

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
  mat1 @ mat2`. Mirrors `TORCH_META_FUNC(addmm)` at
  `aten/src/ATen/native/LinearAlgebra.cpp:194` and
  `TORCH_IMPL_FUNC(addmm_out_cpu)` at
  `aten/src/ATen/native/LinearAlgebra.cpp:1620`. **NOT-STARTED in this
  file** — no `AddmmBackward` or `pub fn addmm_differentiable` in
  `ferrotorch-core/src/grad_fns/linalg.rs`. The fused `linear_fused`
  forward implements an addmm-like pattern (A @ W^T + bias) but only for
  the specific `alpha=1, beta=1, B-transposed` shape, not the general
  `addmm(self, mat1, mat2, beta, alpha)` API. Open prereq blocker #1345.

- REQ-6: `addbmm(self, batch1, batch2, beta=1, alpha=1) = beta * self +
  alpha * sum_b(batch1[b] @ batch2[b])`. Mirrors
  `Tensor addbmm(const Tensor& self, const Tensor& batch1, const Tensor&
  batch2, const Scalar& beta, const Scalar& alpha)` at
  `aten/src/ATen/native/LinearAlgebra.cpp:1615`. **NOT-STARTED in this
  file**. Open prereq blocker #1345.

- REQ-7: `baddbmm(self, batch1, batch2, beta=1, alpha=1) = beta * self +
  alpha * bmm(batch1, batch2)`. Mirrors `TORCH_META_FUNC(baddbmm)` at
  `aten/src/ATen/native/LinearAlgebra.cpp:340` and
  `TORCH_IMPL_FUNC(baddbmm_out_cpu)` at
  `aten/src/ATen/native/LinearAlgebra.cpp:1886`. **NOT-STARTED in this
  file**. Open prereq blocker #1345.

- REQ-8: `addmv(self, mat, vec, beta=1, alpha=1) = beta * self + alpha *
  mat @ vec`. Mirrors `TORCH_META_FUNC(addmv)` at
  `aten/src/ATen/native/Blas.cpp:40` and `TORCH_IMPL_FUNC(addmv_out_cpu)`
  at `aten/src/ATen/native/Blas.cpp:72`. **NOT-STARTED in this file**.
  Open prereq blocker #1345.

- REQ-9: `addr(self, vec1, vec2, beta=1, alpha=1) = beta * self + alpha *
  outer(vec1, vec2)`. Mirrors `Tensor addr(const Tensor& self, ...)` at
  `aten/src/ATen/native/LinearAlgebra.cpp:1200`. **NOT-STARTED in this
  file**. Open prereq blocker #1345.

- REQ-10: `linalg.solve(A, B)` — solve a square system `A @ X = B`.
  Mirrors `Tensor linalg_solve(const Tensor& A, ...)` at
  `aten/src/ATen/native/BatchLinearAlgebra.cpp:2020` and documented at
  `torch/linalg/__init__.py:2218`. A forward-only implementation exists
  in `ferrotorch-core/src/linalg.rs` (the ops module — different file
  from this `grad_fns/linalg.rs`), routed through `ferray_linalg::solve`.
  **SHIPPED** (2026-05-27): `LinalgSolveBackward` +
  `solve_differentiable` at `ferrotorch-core/src/grad_fns/linalg.rs`
  attach the VJP `gB = A^{-T} @ gX` (computed as `solve(A^T, gX)`) and
  `gA = -gB @ X^T` (vector RHS promoted to a column matrix), grounded in
  `torch/csrc/autograd/FunctionsManual.cpp:6160 linalg_solve_backward`.
  Both gradient slots FD-verified at
  `ferrotorch-core/tests/divergence_linalg_grad_audit.rs`
  (`solve_backward_matrix_rhs_*` and `solve_backward_vector_rhs_*`).
  Non-test production consumer: the `"linalg.solve"` arm in
  `tools/parity-sweep/runner/src/main.rs` (`24/192` non-skipped samples
  pass, `0 failed`; batched/0-sized op_db samples legitimately skipped
  since the faer forward is square-2-D-only). Closes #1345 (this REQ).

- REQ-11: `linalg.svd(A, full_matrices=True)` — singular value
  decomposition `A = U @ diag(S) @ Vh`. Documented at
  `torch/linalg/__init__.py:1739`. Forward-only impl in
  `ferrotorch-core/src/linalg.rs` (via `ferray_linalg::svd`).
  **NOT-STARTED in this file** — no `LinalgSvdBackward`. Open prereq
  blocker #1345.

- REQ-12: `linalg.eig(A)` — non-symmetric eigendecomposition. Mirrors
  `torch/linalg/__init__.py:474`. Forward-only impl in
  `ferrotorch-core/src/linalg.rs` via `ferray_linalg::eig`. **NOT-STARTED
  in this file**. Open prereq blocker #1345.

- REQ-13: `linalg.eigh(A, UPLO='L')` — symmetric/Hermitian
  eigendecomposition. Mirrors `torch/linalg/__init__.py:642`.
  Forward-only impl in `ferrotorch-core/src/linalg.rs` via
  `ferray_linalg::eigh`. **NOT-STARTED in this file**. Open prereq
  blocker #1345.

- REQ-14: `linalg.eigvals(A)` — eigenvalues only (non-symmetric).
  Documented at `torch/linalg/__init__.py:584`. Forward-only impl in
  `ferrotorch-core/src/linalg.rs` via `ferray_linalg::eigvals`.
  **NOT-STARTED in this file**. Open prereq blocker #1345.

- REQ-15: `linalg.eigvalsh(A, UPLO='L')` — eigenvalues only
  (symmetric/Hermitian). Documented at `torch/linalg/__init__.py:765`.
  Forward-only impl in `ferrotorch-core/src/linalg.rs` via
  `ferray_linalg::eigvalsh`. **NOT-STARTED in this file**. Open prereq
  blocker #1345.

- REQ-16: `linalg.qr(A, mode='reduced')` — QR factorization. Documented
  at `torch/linalg/__init__.py:2823`. Forward-only impl in
  `ferrotorch-core/src/linalg.rs` via `ferray_linalg::qr`.
  **NOT-STARTED in this file**. Open prereq blocker #1345.

- REQ-17: `linalg.cholesky(A)` — Cholesky factorization for SPD matrices.
  Mirrors `Tensor linalg_cholesky(const Tensor& A, bool upper)` at
  `aten/src/ATen/native/BatchLinearAlgebra.cpp:1873` and documented at
  `torch/linalg/__init__.py:71`. Forward-only impl in
  `ferrotorch-core/src/linalg.rs` via `ferray_linalg::cholesky`.
  **NOT-STARTED in this file**. Open prereq blocker #1345.

- REQ-18: `linalg.inv(A)` — matrix inverse. Mirrors `Tensor linalg_inv(
  const Tensor& A)` at `aten/src/ATen/native/BatchLinearAlgebra.cpp:1683`
  and documented at `torch/linalg/__init__.py:214`. Forward-only impl in
  `ferrotorch-core/src/linalg.rs` via `ferray_linalg::inv`.
  **SHIPPED** (2026-05-27): `LinalgInvBackward` + `inv_differentiable`
  attach `dA = -Y^T @ grad @ Y^T` (`Y` = the retained inverse), grounded
  in `tools/autograd/derivatives.yaml:917` (`linalg_inv_ex`). FD-verified
  at `ferrotorch-core/tests/divergence_linalg_grad_audit.rs:inv_backward_matches_finite_difference`.
  Non-test consumer: the `"linalg.inv"` arm in
  `tools/parity-sweep/runner/src/main.rs` (`8/64` non-skipped pass,
  `0 failed`; batched/0-sized skipped). Closes #1345 (this REQ).

- REQ-19: `linalg.pinv(A, atol=None, rtol=None)` — Moore-Penrose
  pseudoinverse. Mirrors `Tensor linalg_pinv(...)` at
  `aten/src/ATen/native/LinearAlgebra.cpp:510`. Forward-only impl in
  `ferrotorch-core/src/linalg.rs` via `ferray_linalg::pinv`.
  **NOT-STARTED in this file**. Open prereq blocker #1345.

- REQ-20: `linalg.det(A)` — determinant. Mirrors `Tensor linalg_det(const
  Tensor& A)` at `aten/src/ATen/native/LinearAlgebra.cpp:378` and
  documented at `torch/linalg/__init__.py:390`. Forward-only impl in
  `ferrotorch-core/src/linalg.rs` via `ferray_linalg::det`.
  **SHIPPED** (2026-05-27): `LinalgDetBackward` + `det_differentiable`
  attach `dA = det(A) * grad * inv(A)^T` (the invertible branch of
  `torch/csrc/autograd/FunctionsManual.cpp:4373 linalg_det_backward`,
  which solves `A^T G = det * grad * I`). FD-verified at
  `ferrotorch-core/tests/divergence_linalg_grad_audit.rs:det_backward_matches_finite_difference`.
  Non-test consumer: the `"linalg.det"` arm in
  `tools/parity-sweep/runner/src/main.rs` (`16/72` non-skipped pass,
  `0 failed`; batched/0-sized skipped). Closes #1345 (this REQ).

- REQ-21: `linalg.slogdet(A)` — sign and log-magnitude of the
  determinant. Documented at `torch/linalg/__init__.py:424`.
  Forward-only impl in `ferrotorch-core/src/linalg.rs`. **NOT-STARTED in
  this file**. Open prereq blocker #1345.

- REQ-22: `linalg.lstsq(A, B, rcond=None)` — least-squares solver.
  Documented at `torch/linalg/__init__.py:1078`. Forward-only impl in
  `ferrotorch-core/src/linalg.rs` via `ferray_linalg::lstsq`.
  **NOT-STARTED in this file**. Open prereq blocker #1345.

- REQ-23: `linalg.norm(A, ord=None, dim=None)` — generic norm (Frobenius
  for matrices, p-norm for vectors). Documented at
  `torch/linalg/__init__.py:1353`. Forward-only impl in
  `ferrotorch-core/src/linalg.rs` via `ferray_linalg::norm`.
  **NOT-STARTED in this file**. Open prereq blocker #1345.

- REQ-24: `linalg.matrix_rank(A, tol=None)`. Mirrors `Tensor
  linalg_matrix_rank(...)` at
  `aten/src/ATen/native/LinearAlgebra.cpp:819-852` (overload family).
  Forward-only impl in `ferrotorch-core/src/linalg.rs` via
  `ferray_linalg::matrix_rank` style. **NOT-STARTED in this file**. Open
  prereq blocker #1345.

- REQ-25: `linalg.cross(A, B, dim=-1)` — vector cross product along
  `dim` (must equal 3). Forward-only impl in
  `ferrotorch-core/src/linalg.rs` (`pub fn cross`). **NOT-STARTED in this
  file**. Open prereq blocker #1345.

- REQ-26: `linalg.householder_product(A, tau)`. Mirrors `Tensor
  linalg_householder_product(...)` at
  `aten/src/ATen/native/BatchLinearAlgebra.cpp:2644` and documented at
  `torch/linalg/__init__.py:836`. Forward-only impl in
  `ferrotorch-core/src/linalg.rs` (`pub fn householder_product`).
  **NOT-STARTED in this file**. Open prereq blocker #1345.

- REQ-27: `linalg.lu(A, pivot=True)` — LU factorization with pivoting.
  Documented at `torch/linalg/__init__.py:2599`. Forward-only impl in
  `ferrotorch-core/src/linalg.rs` via `ferray_linalg::lu`. **NOT-STARTED
  in this file**. Open prereq blocker #1345.

- REQ-28: `linalg.lu_factor(A)` — LU factorization without explicit
  unpacking. Documented at `torch/linalg/__init__.py:2403`. Forward-only
  impl in `ferrotorch-core/src/linalg.rs`. **NOT-STARTED in this file**.
  Open prereq blocker #1345.

- REQ-29: `trace(A)` — sum of the main diagonal. **SHIPPED**
  (2026-05-27): forward `crate::linalg::trace` (sum of `A[i,i]`,
  scalar output) + `TraceBackward` / `trace_differentiable` in
  `ferrotorch-core/src/grad_fns/linalg.rs` attach the VJP `dA = grad * I`
  per `tools/autograd/derivatives.yaml:1785 trace_backward_symint`.
  FD-verified at
  `ferrotorch-core/tests/divergence_linalg_grad_audit.rs:trace_backward_matches_finite_difference`.
  Non-test consumer: the `"trace"` arm in
  `tools/parity-sweep/runner/src/main.rs` (parity `8/8`, `0 failed`).
  Closes #1345 (this REQ).

- REQ-30: `diagonal(A, offset=0, dim1=0, dim2=1)`. Mirrors `Tensor
  linalg_diagonal(const Tensor& A, int64_t offset, int64_t dim1, int64_t
  dim2)` at `aten/src/ATen/native/LinearAlgebra.cpp:2215`. Forward-only
  impl in `ferrotorch-core/src/linalg.rs` (`pub fn diagonal`).
  **NOT-STARTED in this file**. Open prereq blocker #1345.

- REQ-31: `diag(A, diagonal=0)` — extract or construct a diagonal.
  Forward-only impl in `ferrotorch-core/src/ops/tensor_ops.rs:98` (`pub
  fn diag`). **NOT-STARTED in this file**. Open prereq blocker #1345.

- REQ-32: `tril(A, diagonal=0)` — lower triangular zeroing.
  Forward-only impl in `ferrotorch-core/src/ops/tensor_ops.rs:62` (`pub
  fn tril`). **NOT-STARTED in this file**. Open prereq blocker #1345.

- REQ-33: `triu(A, diagonal=0)` — upper triangular zeroing.
  Forward-only impl in `ferrotorch-core/src/ops/tensor_ops.rs:28` (`pub
  fn triu`). **NOT-STARTED in this file**. Open prereq blocker #1345.

- REQ-34: `kron(A, B)` — Kronecker product. Mirrors `Tensor kron(const
  Tensor& self, const Tensor& other)` at
  `aten/src/ATen/native/LinearAlgebra.cpp:3530`. **NOT-STARTED** — no
  `pub fn kron` or `KronBackward` exists anywhere in ferrotorch-core
  src/. Open prereq blocker #1345.

- REQ-35: `outer(self, vec2)` — outer product. Mirrors `Tensor outer(
  const Tensor& self, const Tensor& vec2)` at
  `aten/src/ATen/native/LinearAlgebra.cpp:1337` (which delegates to
  `self.reshape({-1, 1}) * vec2`). **SHIPPED** (2026-05-27): forward
  `crate::linalg::outer` (`out[i,j] = a[i] * b[j]`, 1-D × 1-D) +
  `OuterBackward` / `outer_differentiable` in
  `ferrotorch-core/src/grad_fns/linalg.rs` attach `da = grad_C @ b`,
  `db = grad_C^T @ a` per `tools/autograd/derivatives.yaml:275-276` (the
  `addr` vec1/vec2 gradients, of which `outer` is the unscaled case).
  FD-verified at
  `ferrotorch-core/tests/divergence_linalg_grad_audit.rs:outer_backward_matches_finite_difference`.
  Non-test consumer: the `"outer"` arm in
  `tools/parity-sweep/runner/src/main.rs` (parity `8/8`, `0 failed`).
  Closes #1345 (this REQ).

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
  0 failed` (all verified 2026-05-26 at seeds=8). The tractable-VJP
  slice landed 2026-05-27 (closes the #1345 sub-slice): `trace 8/8,
  0 failed`; `outer 8/8, 0 failed`; `linalg.det 16/72 non-skipped,
  0 failed`; `linalg.inv 8/64 non-skipped, 0 failed`; `linalg.solve
  24/192 non-skipped, 0 failed` (the det/inv/solve skips are op_db's
  batched / 0-sized samples — the faer forward is square-2-D-only).
  The remaining linalg ops (svd/qr/cholesky/eigh/eig/pinv/lstsq/
  slogdet/norm/matrix_rank/lu/householder_product backward, and the
  fused add{mm,bmm,mv,r}/baddbmm/kron family, plus diagonal/diag/
  tril/triu autograd) still report `N skipped (runner has no arm)`
  and remain tracked under prereq blocker #1345 — those backwards are
  matrix-decomposition differentials (or fused-affine VJPs) that exceed
  the single-dispatch tractable scope and each need their own dispatch.
- [ ] AC-11: `addmm` / `addbmm` / `baddbmm` / `addmv` / `addr` / `trace`
  / `diag` / `tril` / `triu` / `kron` / `outer` `GradFn`-bearing fused
  implementations land in `ferrotorch-core/src/grad_fns/linalg.rs`.
  Tracked by blocker #1345.
- [ ] AC-12: `linalg.solve` / `linalg.svd` / `linalg.eig` / `linalg.eigh`
  / `linalg.eigvals` / `linalg.eigvalsh` / `linalg.qr` / `linalg.cholesky`
  / `linalg.inv` / `linalg.pinv` / `linalg.det` / `linalg.slogdet` /
  `linalg.lstsq` / `linalg.norm` / `linalg.matrix_rank` / `linalg.cross`
  / `linalg.householder_product` / `linalg.lu` / `linalg.lu_factor` gain
  fused `*Backward` `GradFn` impls in this file. Forward paths exist in
  `ferrotorch-core/src/linalg.rs` (the ops module) routed through
  `ferray_linalg`, but autograd is not yet wired. Tracked by blocker
  #1345.

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

### REQ-5..REQ-9 addmm/addbmm/baddbmm/addmv/addr (NOT-STARTED)

The fused `BLAS-3` family `C = beta * self + alpha * mat1 @ mat2`
(addmm), the batched-sum `addbmm`, the per-batch `baddbmm`, the matrix-
vector `addmv`, and the rank-1-outer `addr` are central building blocks
in PyTorch's BLAS surface. None of them are implemented in
`ferrotorch-core/src/grad_fns/linalg.rs`. The closest existing
implementation is `pub fn linear_fused` — but `linear_fused` is
hard-coded to `A @ W^T + bias` (`beta=1, alpha=1`, bias instead of `self`,
weight is transposed) so does not satisfy the general PyTorch addmm
API. Tracked by blocker #1345.

### REQ-10..REQ-28 torch.linalg.* factorization family (NOT-STARTED)

`linalg.solve`, `linalg.svd`, `linalg.eig` / `eigh` / `eigvals` /
`eigvalsh`, `linalg.qr`, `linalg.cholesky`, `linalg.inv`, `linalg.pinv`,
`linalg.det`, `linalg.slogdet`, `linalg.lstsq`, `linalg.norm`,
`linalg.matrix_rank`, `linalg.cross`, `linalg.householder_product`,
`linalg.lu`, `linalg.lu_factor` — all 19 ops have forward-only
implementations in `ferrotorch-core/src/linalg.rs` (the ops module,
NOT this `grad_fns/linalg.rs` file) routed through the
`ferray_linalg::*` LAPACK-backed crate. None of them carry an
autograd-aware fused `GradFn` in `grad_fns/linalg.rs`. Autograd through
these factorizations requires implementing the corresponding VJP — e.g.
SVD backward uses the F-matrix formula `dA = U (F ∘ (U^T dU - dU^T U) /
2) S + dS) Vh + ...`, which is its own substantial work item. Tracked
by blocker #1345.

### REQ-29..REQ-35 trace/diagonal/diag/tril/triu/kron/outer

- `trace`: **SHIPPED** — forward `crate::linalg::trace` + `TraceBackward`
  / `trace_differentiable` (VJP `dA = grad * I`). FD-verified.
- `diagonal`: forward-only `pub fn diagonal` exists at
  `ferrotorch-core/src/linalg.rs`. No autograd yet. Blocker #1345.
- `diag`: forward-only `pub fn diag` exists at
  `ferrotorch-core/src/ops/tensor_ops.rs:98`. No autograd. Blocker #1345.
- `tril`: forward-only `pub fn tril` exists at
  `ferrotorch-core/src/ops/tensor_ops.rs:62`. No autograd. Blocker #1345.
- `triu`: forward-only `pub fn triu` exists at
  `ferrotorch-core/src/ops/tensor_ops.rs:28`. No autograd. Blocker #1345.
- `kron`: no `pub fn kron` anywhere in ferrotorch-core. Blocker #1345.
- `outer`: **SHIPPED** — forward `crate::linalg::outer` + `OuterBackward`
  / `outer_differentiable` (VJP `da = grad @ b`, `db = grad^T @ a`).
  FD-verified.

Remaining (diagonal/diag/tril/triu/kron autograd) tracked by #1345.

## Parity contract

| Op | Upstream entry | Backward formula source | Edge cases mirrored |
|---|---|---|---|
| `mm` | `aten/src/ATen/native/LinearAlgebra.cpp:1641 TORCH_IMPL_FUNC(mm_out_cpu)` | `dA = grad_C @ B^T`, `dB = A^T @ grad_C` | Inner-dim mismatch → `FerrotorchError::ShapeMismatch`; bf16/f16 on GPU use cuBLAS GemmEx with f32 accumulator; autocast f32 → f16 accumulator path; CPU is zero-copy raw-slice loops. |
| `bmm` | `aten/src/ATen/native/LinearAlgebra.cpp:1894 TORCH_IMPL_FUNC(bmm_out_cpu)` | per-batch `mm` VJP composed via `batch_transpose` | 3D inputs only (Err on other ranks); batch-dim mismatch → `ShapeMismatch`; CUDA via `SgemmStridedBatched`/`DgemmStridedBatched`; autocast f32→ `bmm_f16_f32` Tensor Core path. |
| `matmul` | `aten/src/ATen/native/LinearAlgebra.cpp:2190 Tensor matmul(...)` | rank-dispatched: dot/mv/vm/mm/bmm/broadcast_bmm | 1D×1D=dot; 2D×1D=mv; 1D×2D=vm; 2D×2D=mm; 3D×3D=bmm; broadcast ≥3D=gemmStridedBatched with stride-0 on broadcasted axes. CUDA covers all of these for f32/f64; bf16/f16 covered for 2D×2D; other dtype combos err `NotImplementedOnCuda`. |
| `linalg.matmul` | `aten/src/ATen/native/LinearAlgebra.cpp:2206 Tensor linalg_matmul(...)` | alias for `matmul` | upstream `linalg_matmul` is literally `at::matmul(tensor1, tensor2)`; ferrotorch's `Tensor::matmul` covers both since the Python API surface is the same. |
| `addmm` | `aten/src/ATen/native/LinearAlgebra.cpp:1620 TORCH_IMPL_FUNC(addmm_out_cpu)` | dself=beta·grad, dmat1=alpha·grad·mat2^T, dmat2=alpha·mat1^T·grad | NOT-STARTED in this file. Blocker #1345. |
| `addbmm` | `aten/src/ATen/native/LinearAlgebra.cpp:1615 Tensor addbmm(...)` | dself=beta·grad, dbatch1=alpha·grad·batch2^T (broadcast over batch), dbatch2=alpha·batch1^T·grad (sum over batch) | NOT-STARTED. Blocker #1345. |
| `baddbmm` | `aten/src/ATen/native/LinearAlgebra.cpp:1886 TORCH_IMPL_FUNC(baddbmm_out_cpu)` | per-batch addmm-like VJP | NOT-STARTED. Blocker #1345. |
| `addmv` | `aten/src/ATen/native/Blas.cpp:72 TORCH_IMPL_FUNC(addmv_out_cpu)` | dself=beta·grad, dmat=alpha·outer(grad,vec), dvec=alpha·mat^T·grad | NOT-STARTED. Blocker #1345. |
| `addr` | `aten/src/ATen/native/LinearAlgebra.cpp:1200 Tensor addr(...)` | dself=beta·grad, dvec1=alpha·grad@vec2, dvec2=alpha·vec1^T@grad | NOT-STARTED. Blocker #1345. |
| `linalg.solve` | `aten/src/ATen/native/BatchLinearAlgebra.cpp:2020 Tensor linalg_solve(...)` | `gB = A^{-T} @ gX`, `gA = -gB @ X^T` (`FunctionsManual.cpp:6160`) | SHIPPED 2026-05-27 (`LinalgSolveBackward` + `solve_differentiable`; FD-verified; runner `"linalg.solve"` arm 24/192 non-skipped, 0 failed; batched/empty skipped). |
| `trace` | `aten/src/ATen/native/LinearAlgebra.cpp Tensor trace_cpu(...)` | `dA = grad * I` (`derivatives.yaml:1785`) | SHIPPED 2026-05-27 (`TraceBackward` + `trace_differentiable`; FD-verified; runner `"trace"` 8/8, 0 failed). |
| `outer` | `aten/src/ATen/native/LinearAlgebra.cpp:1337 Tensor outer(...)` | `da = grad @ b`, `db = grad^T @ a` (`derivatives.yaml:275-276`) | SHIPPED 2026-05-27 (`OuterBackward` + `outer_differentiable`; FD-verified; runner `"outer"` 8/8, 0 failed). |
| `linalg.det` | `aten/src/ATen/native/LinearAlgebra.cpp:378 Tensor linalg_det(...)` | `dA = det * grad * inv(A)^T` (`FunctionsManual.cpp:4373`) | SHIPPED 2026-05-27 (`LinalgDetBackward` + `det_differentiable`; FD-verified; runner `"linalg.det"` 16/72 non-skipped, 0 failed). |
| `linalg.inv` | `aten/src/ATen/native/BatchLinearAlgebra.cpp:1683 Tensor linalg_inv(...)` | `dA = -inv^T @ grad @ inv^T` (`derivatives.yaml:917`) | SHIPPED 2026-05-27 (`LinalgInvBackward` + `inv_differentiable`; FD-verified; runner `"linalg.inv"` 8/64 non-skipped, 0 failed). |
| `linalg.svd` | `torch/linalg/__init__.py:1739 svd = _add_docstr(...)` | F-matrix formula with U/S/Vh | NOT-STARTED in this file (forward exists in `ferrotorch-core/src/linalg.rs`). Blocker #1345. |
| `linalg.eig` | `torch/linalg/__init__.py:474 eig = _add_docstr(...)` | F-matrix formula with eigenvalue spacing | NOT-STARTED. Blocker #1345. |
| `linalg.eigh` | `torch/linalg/__init__.py:642 eigh = _add_docstr(...)` | sym F-matrix (real spectrum) | NOT-STARTED. Blocker #1345. |
| `linalg.eigvals` | `torch/linalg/__init__.py:584 eigvals = _add_docstr(...)` | derived from `eig` VJP, summed over eigvecs | NOT-STARTED. Blocker #1345. |
| `linalg.eigvalsh` | `torch/linalg/__init__.py:765 eigvalsh = _add_docstr(...)` | sym variant | NOT-STARTED. Blocker #1345. |
| `linalg.qr` | `torch/linalg/__init__.py:2823 qr = _add_docstr(...)` | Q-orthog + R-triangular VJP | NOT-STARTED. Blocker #1345. |
| `linalg.cholesky` | `aten/src/ATen/native/BatchLinearAlgebra.cpp:1873 Tensor linalg_cholesky(...)` | `dA = sym(tri(L^T @ grad)) @ L^{-T}` | NOT-STARTED. Blocker #1345. |
| `linalg.inv` | `aten/src/ATen/native/BatchLinearAlgebra.cpp:1683 Tensor linalg_inv(...)` | `dA = −X^T @ grad @ X^T` where X = A^{-1} | NOT-STARTED. Blocker #1345. |
| `linalg.pinv` | `aten/src/ATen/native/LinearAlgebra.cpp:510 Tensor linalg_pinv(...)` | full pinv VJP formula | NOT-STARTED. Blocker #1345. |
| `linalg.det` | `aten/src/ATen/native/LinearAlgebra.cpp:378 Tensor linalg_det(...)` | `dA = det(A) · A^{-T} · grad` | NOT-STARTED. Blocker #1345. |
| `linalg.slogdet` | `torch/linalg/__init__.py:424 slogdet = _add_docstr(...)` | `dA = A^{-T} · grad_logabs` (sign-grad is 0) | NOT-STARTED. Blocker #1345. |
| `linalg.lstsq` | `torch/linalg/__init__.py:1078 lstsq = _add_docstr(...)` | least-squares VJP via QR | NOT-STARTED. Blocker #1345. |
| `linalg.norm` | `torch/linalg/__init__.py:1353 norm = _add_docstr(...)` | per-ord VJP (Frobenius: `dA = A / ||A||_F · grad`) | NOT-STARTED. Blocker #1345. |
| `linalg.matrix_rank` | `aten/src/ATen/native/LinearAlgebra.cpp:819 Tensor linalg_matrix_rank(...)` | rank is integer; backward returns zero | NOT-STARTED. Blocker #1345. |
| `linalg.cross` | `torch/linalg/__init__.py: cross = _add_docstr(...)` at line 22 | `da = b × grad`, `db = grad × a` | NOT-STARTED. Blocker #1345. |
| `linalg.householder_product` | `aten/src/ATen/native/BatchLinearAlgebra.cpp:2644 Tensor linalg_householder_product(...)` | VJP through Householder reflectors | NOT-STARTED. Blocker #1345. |
| `linalg.lu` | `torch/linalg/__init__.py:2599 lu = _add_docstr(...)` | P/L/U VJP | NOT-STARTED. Blocker #1345. |
| `linalg.lu_factor` | `torch/linalg/__init__.py:2403 lu_factor = _add_docstr(...)` | same as `lu` minus the explicit unpacking | NOT-STARTED. Blocker #1345. |
| `trace` | upstream defined via tensor method (no dedicated .cpp impl in LinearAlgebra.cpp) | `dA = grad · I` (identity matrix scaled by grad scalar) | NOT-STARTED. Blocker #1345. |
| `diagonal` | `aten/src/ATen/native/LinearAlgebra.cpp:2215 Tensor linalg_diagonal(...)` | inverse of `diag_embed` — scatter grad onto the diagonal of zeros | NOT-STARTED in this file (forward exists in `ferrotorch-core/src/linalg.rs:1545`). Blocker #1345. |
| `diag` | upstream tensor method (`Tensor::diag(...)`) | scatter or extract VJP based on input rank | NOT-STARTED in this file (forward exists in `ferrotorch-core/src/ops/tensor_ops.rs:98`). Blocker #1345. |
| `tril` | upstream tensor method | grad masked by lower triangular | NOT-STARTED in this file (forward exists in `ferrotorch-core/src/ops/tensor_ops.rs:62`). Blocker #1345. |
| `triu` | upstream tensor method | grad masked by upper triangular | NOT-STARTED in this file (forward exists in `ferrotorch-core/src/ops/tensor_ops.rs:28`). Blocker #1345. |
| `kron` | `aten/src/ATen/native/LinearAlgebra.cpp:3530 Tensor kron(...)` | `dA = sum over kron-blocks of grad·B^T`, `dB = sum over kron-blocks of A^T·grad` | NOT-STARTED. Blocker #1345. |
| `outer` | `aten/src/ATen/native/LinearAlgebra.cpp:1337 Tensor outer(...)` | `da = grad @ vec2`, `dvec2 = a^T @ grad` | NOT-STARTED at public surface. Blocker #1345. |

Parity-sweep audit reference: all 35 ops are **MISSING** from
`tools/parity-sweep/parity_audit.json`. The runner's dispatch table in
`tools/parity-sweep/runner/src/main.rs` covers 76 elementwise / reduction
/ indexing ops, and none of the 35 linalg ops are listed there —
running `./target/release/parity-sweep sweep --op mm` returns
`[mm] 0/6 passed (6 skipped, 0 failed)` (all 6 randomized variants
skipped because the runner has no arm for `mm`). Per goal.md S5 this is
a test-infrastructure gap; tracked under umbrella runner-arm blocker
#1344.

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
| REQ-5 (addmm) | NOT-STARTED | open prereq blocker #1345. No `AddmmBackward` or `pub fn addmm_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs`. The fused `pub fn linear_fused` implements an addmm-like pattern but only for the specific `A @ W^T + bias` shape (`alpha=1, beta=1`, weight transposed), not the general `addmm(self, mat1, mat2, beta, alpha)` API. Upstream: `TORCH_IMPL_FUNC(addmm_out_cpu)` at `aten/src/ATen/native/LinearAlgebra.cpp:1620`. |
| REQ-6 (addbmm) | NOT-STARTED | open prereq blocker #1345. No `AddbmmBackward` in `ferrotorch-core/src/grad_fns/linalg.rs`. Upstream: `Tensor addbmm(...)` at `aten/src/ATen/native/LinearAlgebra.cpp:1615`. |
| REQ-7 (baddbmm) | NOT-STARTED | open prereq blocker #1345. No `BaddbmmBackward` in `ferrotorch-core/src/grad_fns/linalg.rs`. Upstream: `TORCH_IMPL_FUNC(baddbmm_out_cpu)` at `aten/src/ATen/native/LinearAlgebra.cpp:1886`. |
| REQ-8 (addmv) | NOT-STARTED | open prereq blocker #1345. No `AddmvBackward` in `ferrotorch-core/src/grad_fns/linalg.rs`. Upstream: `TORCH_IMPL_FUNC(addmv_out_cpu)` at `aten/src/ATen/native/Blas.cpp:72`. |
| REQ-9 (addr) | NOT-STARTED | open prereq blocker #1345. No `AddrBackward` in `ferrotorch-core/src/grad_fns/linalg.rs`. Upstream: `Tensor addr(...)` at `aten/src/ATen/native/LinearAlgebra.cpp:1200`. |
| REQ-10 (linalg.solve) | NOT-STARTED | open prereq blocker #1345. Forward-only `pub fn solve` exists in `ferrotorch-core/src/linalg.rs:200` (different file from this one) routed through `ferray_linalg::solve`. No `LinalgSolveBackward` `GradFn` in `ferrotorch-core/src/grad_fns/linalg.rs`. Upstream: `Tensor linalg_solve(...)` at `aten/src/ATen/native/BatchLinearAlgebra.cpp:2020`. |
| REQ-11 (linalg.svd) | NOT-STARTED | open prereq blocker #1345. Forward-only `pub fn svd` exists in `ferrotorch-core/src/linalg.rs:121` routed through `ferray_linalg::svd`. No `LinalgSvdBackward` `GradFn` in `ferrotorch-core/src/grad_fns/linalg.rs`. Upstream: `torch/linalg/__init__.py:1739`. |
| REQ-12 (linalg.eig) | NOT-STARTED | open prereq blocker #1345. Forward-only `pub fn eig` exists in `ferrotorch-core/src/linalg.rs:677` routed through `ferray_linalg::eig`. No `LinalgEigBackward` in `ferrotorch-core/src/grad_fns/linalg.rs`. Upstream: `torch/linalg/__init__.py:474`. |
| REQ-13 (linalg.eigh) | NOT-STARTED | open prereq blocker #1345. Forward-only `pub fn eigh` exists in `ferrotorch-core/src/linalg.rs:569` routed through `ferray_linalg::eigh`. No `LinalgEighBackward` in `ferrotorch-core/src/grad_fns/linalg.rs`. Upstream: `torch/linalg/__init__.py:642`. |
| REQ-14 (linalg.eigvals) | NOT-STARTED | open prereq blocker #1345. Forward-only `pub fn eigvals` exists in `ferrotorch-core/src/linalg.rs:735` routed through `ferray_linalg::eigvals`. No `LinalgEigvalsBackward` in `ferrotorch-core/src/grad_fns/linalg.rs`. Upstream: `torch/linalg/__init__.py:584`. |
| REQ-15 (linalg.eigvalsh) | NOT-STARTED | open prereq blocker #1345. Forward-only `pub fn eigvalsh` exists in `ferrotorch-core/src/linalg.rs:626` routed through `ferray_linalg::eigvalsh`. No `LinalgEigvalshBackward` in `ferrotorch-core/src/grad_fns/linalg.rs`. Upstream: `torch/linalg/__init__.py:765`. |
| REQ-16 (linalg.qr) | NOT-STARTED | open prereq blocker #1345. Forward-only `pub fn qr` exists in `ferrotorch-core/src/linalg.rs:348` routed through `ferray_linalg::qr`. No `LinalgQrBackward` in `ferrotorch-core/src/grad_fns/linalg.rs`. Upstream: `torch/linalg/__init__.py:2823`. |
| REQ-17 (linalg.cholesky) | NOT-STARTED | open prereq blocker #1345. Forward-only `pub fn cholesky` exists in `ferrotorch-core/src/linalg.rs:419` routed through `ferray_linalg::cholesky`. No `LinalgCholeskyBackward` in `ferrotorch-core/src/grad_fns/linalg.rs`. Upstream: `Tensor linalg_cholesky(...)` at `aten/src/ATen/native/BatchLinearAlgebra.cpp:1873`. |
| REQ-18 (linalg.inv) | NOT-STARTED | open prereq blocker #1345. Forward-only `pub fn inv` exists in `ferrotorch-core/src/linalg.rs:310` routed through `ferray_linalg::inv`. No `LinalgInvBackward` in `ferrotorch-core/src/grad_fns/linalg.rs`. Upstream: `Tensor linalg_inv(const Tensor& A)` at `aten/src/ATen/native/BatchLinearAlgebra.cpp:1683`. |
| REQ-19 (linalg.pinv) | NOT-STARTED | open prereq blocker #1345. Forward-only `pub fn pinv` exists in `ferrotorch-core/src/linalg.rs:530` routed through `ferray_linalg::pinv`. No `LinalgPinvBackward` in `ferrotorch-core/src/grad_fns/linalg.rs`. Upstream: `Tensor linalg_pinv(...)` at `aten/src/ATen/native/LinearAlgebra.cpp:510`. |
| REQ-20 (linalg.det) | NOT-STARTED | open prereq blocker #1345. Forward-only `pub fn det` exists in `ferrotorch-core/src/linalg.rs:276` routed through `ferray_linalg::det`. No `LinalgDetBackward` in `ferrotorch-core/src/grad_fns/linalg.rs`. Upstream: `Tensor linalg_det(const Tensor& A)` at `aten/src/ATen/native/LinearAlgebra.cpp:378`. |
| REQ-21 (linalg.slogdet) | NOT-STARTED | open prereq blocker #1345. Forward-only `pub fn slogdet` exists in `ferrotorch-core/src/linalg.rs:1223`. No `LinalgSlogdetBackward` in `ferrotorch-core/src/grad_fns/linalg.rs`. Upstream: `torch/linalg/__init__.py:424`. |
| REQ-22 (linalg.lstsq) | NOT-STARTED | open prereq blocker #1345. Forward-only `pub fn lstsq` exists in `ferrotorch-core/src/linalg.rs:1023` routed through `ferray_linalg::lstsq`. No `LinalgLstsqBackward` in `ferrotorch-core/src/grad_fns/linalg.rs`. Upstream: `torch/linalg/__init__.py:1078`. |
| REQ-23 (linalg.norm) | NOT-STARTED | open prereq blocker #1345. Forward-only `pub fn matrix_norm` exists in `ferrotorch-core/src/linalg.rs:471` and `pub fn vector_norm` at `ferrotorch-core/src/linalg.rs:1194` routed through `ferray_linalg::norm`. No `LinalgNormBackward` in `ferrotorch-core/src/grad_fns/linalg.rs`. Upstream: `torch/linalg/__init__.py:1353`. |
| REQ-24 (linalg.matrix_rank) | NOT-STARTED | open prereq blocker #1345. Forward-only `pub fn matrix_rank` exists in `ferrotorch-core/src/linalg.rs:1276`. No `LinalgMatrixRankBackward` in `ferrotorch-core/src/grad_fns/linalg.rs`. Upstream: `Tensor linalg_matrix_rank(...)` at `aten/src/ATen/native/LinearAlgebra.cpp:819`. |
| REQ-25 (linalg.cross) | NOT-STARTED | open prereq blocker #1345. Forward-only `pub fn cross` exists in `ferrotorch-core/src/linalg.rs:1388`. No `LinalgCrossBackward` in `ferrotorch-core/src/grad_fns/linalg.rs`. Upstream: `torch/linalg/__init__.py:22`. |
| REQ-26 (linalg.householder_product) | NOT-STARTED | open prereq blocker #1345. Forward-only `pub fn householder_product` exists in `ferrotorch-core/src/linalg.rs:1835`. No `LinalgHouseholderProductBackward` in `ferrotorch-core/src/grad_fns/linalg.rs`. Upstream: `Tensor linalg_householder_product(...)` at `aten/src/ATen/native/BatchLinearAlgebra.cpp:2644`. |
| REQ-27 (linalg.lu) | NOT-STARTED | open prereq blocker #1345. Forward-only `pub fn lu` exists in `ferrotorch-core/src/linalg.rs:783` routed through `ferray_linalg::lu`. No `LinalgLuBackward` in `ferrotorch-core/src/grad_fns/linalg.rs`. Upstream: `torch/linalg/__init__.py:2599`. |
| REQ-28 (linalg.lu_factor) | NOT-STARTED | open prereq blocker #1345. Forward-only `pub fn lu_factor` exists in `ferrotorch-core/src/linalg.rs:833`. No `LinalgLuFactorBackward` in `ferrotorch-core/src/grad_fns/linalg.rs`. Upstream: `torch/linalg/__init__.py:2403`. |
| REQ-29 (trace) | NOT-STARTED | open prereq blocker #1345. No `pub fn trace` exists anywhere in ferrotorch-core src/ as a linear-algebra op. (`autograd::anomaly::trace` at `ferrotorch-core/src/autograd/anomaly.rs:106` is an unrelated stack-trace function returning `&str`.) Upstream: tensor method; closed-form VJP is `dA = grad · I`. |
| REQ-30 (diagonal) | NOT-STARTED | open prereq blocker #1345. Forward-only `pub fn diagonal` exists in `ferrotorch-core/src/linalg.rs:1545`. No `DiagonalBackward` in `ferrotorch-core/src/grad_fns/linalg.rs`. Upstream: `Tensor linalg_diagonal(...)` at `aten/src/ATen/native/LinearAlgebra.cpp:2215`. |
| REQ-31 (diag) | NOT-STARTED | open prereq blocker #1345. Forward-only `pub fn diag` exists in `ferrotorch-core/src/ops/tensor_ops.rs:98`. No `DiagBackward` in `ferrotorch-core/src/grad_fns/linalg.rs`. Upstream: tensor method. |
| REQ-32 (tril) | NOT-STARTED | open prereq blocker #1345. Forward-only `pub fn tril` exists in `ferrotorch-core/src/ops/tensor_ops.rs:62`. No `TrilBackward` in `ferrotorch-core/src/grad_fns/linalg.rs`. Upstream: tensor method; VJP is grad masked by lower triangular. |
| REQ-33 (triu) | NOT-STARTED | open prereq blocker #1345. Forward-only `pub fn triu` exists in `ferrotorch-core/src/ops/tensor_ops.rs:28`. No `TriuBackward` in `ferrotorch-core/src/grad_fns/linalg.rs`. Upstream: tensor method; VJP is grad masked by upper triangular. |
| REQ-34 (kron) | NOT-STARTED | open prereq blocker #1345. No `pub fn kron` exists anywhere in ferrotorch-core src/. Upstream: `Tensor kron(const Tensor& self, const Tensor& other)` at `aten/src/ATen/native/LinearAlgebra.cpp:3530`. |
| REQ-35 (outer) | NOT-STARTED | open prereq blocker #1345. No `pub fn outer` at the public surface. The outer-product pattern is used internally inside `MvBackward` (`dA = outer(grad_y, x)`) and the vm branch of `MatmulBackward` as inline helpers, but not as a publicly callable op. Upstream: `Tensor outer(const Tensor& self, const Tensor& vec2)` at `aten/src/ATen/native/LinearAlgebra.cpp:1337`. |
