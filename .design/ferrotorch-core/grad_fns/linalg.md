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

The fused-affine family (`AddmmBackward`, `AddbmmBackward`,
`BaddbmmBackward`, `AddmvBackward`, `AddrBackward`), the structural-autograd
`KronBackward`, and the diagonal-family `DiagonalBackward` / `DiagBackward`
/ `TriangularBackward` shipped end-to-end 2026-05-27 (closing the remaining
scope of #1583): grad-aware public forwards `pub fn addmm` / `addbmm` /
`baddbmm` / `addmv` / `addr` / `kron` / `diagonal` in
`ferrotorch-core/src/linalg.rs` (the `torch.addmm`/.../`torch.kron`/
`torch.linalg.diagonal` public surfaces) and the now-grad-aware
`pub fn diag` / `tril` / `triu` in
`ferrotorch-core/src/ops/tensor_ops.rs` delegate to the matching
`*_differentiable` wrapper (the structural diag/tril/triu/diagonal wrappers
compute the forward under `no_grad` to avoid re-entry; the addmm-family and
kron wrappers compute the fused forward inline). Each VJP is FD-verified by
a public-forward-driven test in this file's `#[cfg(test)] mod tests`
(REQ-5/6/7/8/9/30/31/32/33/34). With these wired, **#1583 is fully
resolved** — all its ops are grad-aware end-to-end.

The research-grade decomposition backwards `eigvalsh`, `eigh` (real
symmetric F-matrix), `svd` (3-output reduced F-matrix VJP + rectangular
projectors), `pinv` (algebraic Moore-Penrose), `lstsq` (full-rank
solution VJP via `pinv_backward`), `lu` / `lu_factor` (square `m==n`), plus
the clean `cross` (bilinear) and `norm` (Frobenius / Euclidean `p=2`)
VJPs shipped 2026-05-27 with `*Backward` `GradFn` structs in this file and
grad-aware forwards in `ferrotorch-core/src/linalg.rs` delegating to them
(SHIPPED — REQ-11/13/15/19/22/23/25/27/28). `matrix_rank` is
non-differentiable and mirrors torch's no-grad contract (REQ-24, no
`GradFn` needed). With `svd` landed, **the differentiable scope of
sub-blocker #1577 is complete**. `householder_product` (REQ-26) shipped
2026-05-27 — `HouseholderProductBackward` + `householder_product_differentiable`
implement the real reflector-recursion VJP (`FunctionsManual.cpp:5544`),
consumed by the now-`[m,k]`-shaped grad-aware `pub fn householder_product`.

`eig` / `eigvals` (REQ-12/14) shipped their COMPLEX backward 2026-05-27,
closing the LAST #1345 linalg-autograd gap: `EigBackwardW`/`EigBackwardV`
(`eig`) and `EigvalsBackward` (`eigvals`) + the grad-aware `pub fn eig` /
`pub fn eigvals` forwards in `ferrotorch-core/src/linalg.rs` implement
`linalg_eig_backward` (`FunctionsManual.cpp:3820`, non-Hermitian branch) via
a private complex-linalg toolkit (`c_matmul`/`c_conj_transpose`/`c_inverse`/
`c_solve`/`c_econj_gap`) on the `[n,2]` / `[n,n,2]` real/imag encoding — the
real part of the complex `gA` (real `A`, `handle_r_to_c`,
`derivatives.yaml:1740`). With these wired, **#1345 is fully resolved** for
the differentiable scope — all its named ops are grad-aware end-to-end.

The remaining `torch.linalg.*` factorizations still **forward-only** in
`ferrotorch-core/src/linalg.rs` with no `*Backward` `GradFn` are the residual
#1577 follow-ups: the RECTANGULAR `m != n` LU and the rank-deficient `lstsq`
/ non-`p=2` `norm` branches. `eig`/`eigvals`/`householder_product` are no
longer forward-only.

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
  `aten/src/ATen/native/LinearAlgebra.cpp`. **SHIPPED** (2026-05-27,
  closing #1583): `AddmmBackward` + `addmm_differentiable` in
  `ferrotorch-core/src/grad_fns/linalg.rs`; non-test production consumer:
  the grad-aware forward `pub fn addmm` in `ferrotorch-core/src/linalg.rs`
  (the `torch.addmm` public surface) delegates to `addmm_differentiable`.
  FD-verified by `fn addmm_public_forward_is_grad_aware_and_matches_fd`.

- REQ-6: `addbmm(self, batch1, batch2, beta=1, alpha=1) = beta * self +
  alpha * sum_b(batch1[b] @ batch2[b])`. Mirrors `Tensor addbmm(...)` in
  `aten/src/ATen/native/LinearAlgebra.cpp`. **SHIPPED** (2026-05-27):
  `AddbmmBackward` + `addbmm_differentiable` in
  `ferrotorch-core/src/grad_fns/linalg.rs`; non-test production consumer:
  the grad-aware forward `pub fn addbmm` in `ferrotorch-core/src/linalg.rs`.
  FD-verified by `fn addbmm_public_forward_is_grad_aware_and_matches_fd`.

- REQ-7: `baddbmm(self, batch1, batch2, beta=1, alpha=1) = beta * self +
  alpha * bmm(batch1, batch2)`. Mirrors `TORCH_META_FUNC(baddbmm)` and
  `TORCH_IMPL_FUNC(baddbmm_out_cpu)` in
  `aten/src/ATen/native/LinearAlgebra.cpp`. **SHIPPED** (2026-05-27):
  `BaddbmmBackward` + `baddbmm_differentiable` in
  `ferrotorch-core/src/grad_fns/linalg.rs`; non-test production consumer:
  the grad-aware forward `pub fn baddbmm` in `ferrotorch-core/src/linalg.rs`.
  FD-verified by `fn baddbmm_public_forward_is_grad_aware_and_matches_fd`.

- REQ-8: `addmv(self, mat, vec, beta=1, alpha=1) = beta * self + alpha *
  mat @ vec`. Mirrors `TORCH_META_FUNC(addmv)` and
  `TORCH_IMPL_FUNC(addmv_out_cpu)` in `aten/src/ATen/native/Blas.cpp`.
  **SHIPPED** (2026-05-27): `AddmvBackward` + `addmv_differentiable` in
  `ferrotorch-core/src/grad_fns/linalg.rs`; non-test production consumer:
  the grad-aware forward `pub fn addmv` in `ferrotorch-core/src/linalg.rs`.
  FD-verified by `fn addmv_public_forward_is_grad_aware_and_matches_fd`.

- REQ-9: `addr(self, vec1, vec2, beta=1, alpha=1) = beta * self + alpha *
  outer(vec1, vec2)`. Mirrors `Tensor addr(const Tensor& self, ...)` in
  `aten/src/ATen/native/LinearAlgebra.cpp`. **SHIPPED** (2026-05-27):
  `AddrBackward` + `addr_differentiable` in
  `ferrotorch-core/src/grad_fns/linalg.rs`; non-test production consumer:
  the grad-aware forward `pub fn addr` in `ferrotorch-core/src/linalg.rs`.
  FD-verified by `fn addr_public_forward_is_grad_aware_and_matches_fd`.

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

- REQ-11: `linalg.svd(A, full_matrices=False)` — reduced singular value
  decomposition `A = U @ diag(S) @ Vh`. Documented in
  `torch/linalg/__init__.py`. **SHIPPED** (2026-05-27, closing #1577's svd
  subset): the split `SvdBackwardU` / `SvdBackwardS` / `SvdBackwardV`
  (real reduced-SVD F-matrix VJP, INCLUDING the rectangular `m != n`
  projector terms) + `svd_differentiable` in
  `ferrotorch-core/src/grad_fns/linalg.rs`, consumed by the grad-aware
  `pub fn svd` in `ferrotorch-core/src/linalg.rs`. Mirrors `svd_backward`
  at `torch/csrc/autograd/FunctionsManual.cpp:3605` (the real case): the
  singular-value-gap matrix `E[i,j] = S²[j] - S²[i]` off-diagonal / `1` on
  the diagonal (`FunctionsManual.cpp:3770-3777`), the symmetrized core
  `(UhgU·S + S·VhgV)/E + diag(gS)` (`3767-3797`), and the rectangular
  projectors `(I_m - UU^T) gU S⁻¹ V^T` for `m>n` and `U S⁻¹ gV^T (I_n -
  VV^T)` for `m<n` (`3799-3815`). The three outputs `(U,S,Vh)` are jointly
  linear in `gA`, so the joint VJP is SPLIT across the three single-output
  nodes (`SvdBackwardU` on the `U` output, `SvdBackwardS` on `S`,
  `SvdBackwardV` on `Vh`) accumulated into `A.grad` — the same split-node
  strategy QR (`QrBackwardQ`/`QrBackwardR`) and eigh
  (`EighBackwardW`/`EighBackwardV`) use. A consumer that uses only `S`
  (`gU=gVh=None`) gets only the `SvdBackwardS` contribution, matching
  torch's svdvals optimisation (`FunctionsManual.cpp:3738-3741`). Verified
  against LIVE `torch 2.11.0+cu130` float64 `A.grad` for SQUARE (3×3), TALL
  (4×3, `m>n`), WIDE (3×4, `m<n`), and S-only inputs by
  `grad_fns::linalg::tests::svd_backward_{square_3x3,tall_4x3,wide_3x4,
  s_only_square_3x3,s_only_tall_4x3}_matches_torch` at `1e-6`.
  **Eigenvector-gauge caveat (R-DEV-1, mirrors eigh #1584):** the reduced
  SVD is gauge-free — `(U,V)` and `(U·diag(±1), V·diag(±1))` are both valid
  decompositions (upstream `FunctionsManual.cpp:3682-3698`). ferray's
  faer-backed `svd` forward emits per-column signs that differ
  matrix-by-matrix from torch's LAPACK signs; the SINGULAR VALUES match
  torch exactly, and the VJP is sign-consistent, so for SIGN-INVARIANT
  losses (`sum((U*U)*M)`, reconstructions `U diag(f(S)) Vh`, PCA/whitening,
  and any `S`-only objective) `A.grad` matches torch byte-for-byte; the
  tests therefore assert a gauge-invariant loss
  `sum((U*U)*MU)+sum((Vh*Vh)*MV)+sum(S*c)` (verified maxdiff=0 under joint
  U/Vh column sign flips in the torch oracle). EXACT for inputs with
  DISTINCT singular values; on a degenerate input the `E` off-diagonal
  `1/(S²[j]-S²[i])` diverges exactly as torch's does (torch does not
  special-case degeneracy). The complex case (`Im(diag(...))` invariance
  terms at `FunctionsManual.cpp:3718-3725,3793-3795`) is out of scope — it
  needs complex-tensor autograd (#1345). The CUDA forward stays
  forward-only.

- REQ-12: `linalg.eig(A)` — non-symmetric eigendecomposition. Documented
  in `torch/linalg/__init__.py`, derivative `linalg_eig` in
  `tools/autograd/derivatives.yaml:1739-1740`. **SHIPPED** (2026-05-27,
  closing #1345): the split `EigBackwardW` / `EigBackwardV` +
  `eig_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs`
  implement the non-Hermitian COMPLEX VJP, grounded in `linalg_eig_backward`
  at `torch/csrc/autograd/FunctionsManual.cpp:3820` (the general,
  `is_hermitian=false` branch). For `A = V diag(L) V^{-1}`:
  `VhgV = V^H @ gV`; the unit-norm tangent projection
  `VhgV <- VhgV - V^H @ (V * real(diag(VhgV)))`
  (`FunctionsManual.cpp:3887-3889`); `Econj[i,j] = conj(L_j) - conj(L_i)`
  off-diagonal / `1` on the diagonal (`FunctionsManual.cpp:3893-3898`);
  `ret = VhgV / Econj` with the diagonal carrying `gL`; and the conjugation
  `gA = linalg_solve(V^H, ret @ V^H)` (`FunctionsManual.cpp:3919`). All
  arithmetic is COMPLEX, performed on the `[n,2]` / `[n,n,2]` real/imag
  encoding by the private complex-linalg toolkit (`c_matmul`,
  `c_conj_transpose`, `c_inverse` via Gauss-Jordan partial pivot, `c_solve`,
  `c_econj_gap`) in this file — bounded plumbing, NOT a complex-dtype
  subsystem. Because `A` is REAL the returned gradient is the REAL part
  (`handle_r_to_c`, `derivatives.yaml:1740`). The two outputs `(L, V)` are
  jointly linear in `gA`, so the joint VJP is SPLIT across two single-output
  nodes (`EigBackwardW` on the eigenvalues output, `EigBackwardV` on the
  eigenvectors output) accumulated into `A.grad` — the same split-node
  strategy `eigh`/`svd`/`qr` use; grad through `L` only (`gV=0`), `V` only
  (`gL=0`), or both is handled by the split. Non-test production consumer:
  the grad-aware forward `pub fn eig` in `ferrotorch-core/src/linalg.rs`
  delegates to `eig_differentiable` when
  `is_grad_enabled() && a.requires_grad()` (forward computed under
  `no_grad`); that forward also UNIT-NORM-normalizes ferray's faer-backed
  eigenvector columns (`normalize_complex_eigenvector_columns`) to match
  torch's documented norm-one contract (`FunctionsManual.cpp:3837-3839`), so
  the VJP's unit-norm projection is correct and `V` matches torch up to a
  per-column phase. Verified vs LIVE `torch 2.11.0` float64 `A.grad` for a
  3×3 with DISTINCT REAL eigenvalues (V real) AND a 2×2 with a
  COMPLEX-CONJUGATE eigenvalue pair (`[[1,-1],[1,1]]`, `L = 1 ± i`, V
  genuinely complex) by
  `grad_fns::linalg::tests::{eig_backward_real_3x3,
  eig_backward_complex_pair_2x2, eig_backward_v_only_complex_pair_2x2}_matches_torch`
  at `1e-6`. **Eigenvector-gauge caveat (R-DEV-1, mirrors eigh #1584; #1591):**
  eig eigenvectors are scale-free AND phase-free; torch normalizes to unit norm
  but the PHASE `V_j -> V_j e^{i phi}` is a genuine gauge freedom torch documents
  and asserts the loss must be invariant to (`FunctionsManual.cpp:3867-3879`).
  ferray's faer-backed `eig` emits per-column phases that differ matrix-by-matrix
  from torch's LAPACK `geev` gauge. To give a stable, REPRODUCIBLE contract,
  `canonicalize_complex_eigenvector_phase` in `ferrotorch-core/src/linalg.rs`
  rotates each eigenvector column by `e^{-i phi}` so its LARGEST-MAGNITUDE
  component becomes real-positive — a DETERMINISTIC ferrotorch convention (the
  complex analog of `canonicalize_eigenvector_signs` for `eigh`). This does NOT
  match torch's LAPACK gauge (matching its arbitrary `geev` phases would require
  replicating `geev`), but makes ferrotorch's eig output reproducible (call `eig`
  twice → identical `V`). For **PHASE-INVARIANT** losses (`sum(|V_ij|^2 * M)`,
  reconstructions, every well-posed objective) the gradient is gauge-FREE, so the
  canonicalization does not change `A.grad` and it matches torch exactly — the
  tests therefore use a phase-invariant loss. For **PHASE-DEPENDENT** losses the
  value/gradient is ILL-DEFINED (depends on the arbitrary phase); the
  `EigBackwardV` phase-invariance guard (`|imag(diag(V^H gV))| > 1e-2`, mirroring
  `FunctionsManual.cpp:3867-3879`) rejects grossly-phase-dependent losses, but its
  EXACT threshold is gauge-dependent and may differ from torch's LAPACK-gauge
  boundary — a window can exist where torch raises and ferrotorch's guard does
  not (or vice-versa). The losses in any such window are mathematically
  meaningless regardless of gauge; this is a property of the ill-posed objective,
  not a backward-formula bug. The reframed
  `divergence_eig_1590_phase_guard_boundary.rs` asserts the well-posed,
  gauge-robust quantities: phase-invariant grad vs LIVE torch, grossly
  phase-dependent loss errors, and eigenvector determinism. EXACT for
  DIAGONALIZABLE (distinct-eigenvalue) inputs; on a defective input `V` is
  singular and `c_inverse` returns `SingularMatrix`, and on a repeated eigenvalue
  the `Econj` off-diagonal diverges exactly as torch's does (torch does not
  special-case degeneracy). The CUDA forward stays forward-only.

- REQ-13: `linalg.eigh(A, UPLO='L')` — symmetric/Hermitian
  eigendecomposition. Documented in `torch/linalg/__init__.py`.
  **SHIPPED** (2026-05-27, closing #1577's eigh subset): the split
  `EighBackwardW`/`EighBackwardV` (F-matrix VJP with skew-symmetric
  projection) + `eigh_differentiable` in
  `ferrotorch-core/src/grad_fns/linalg.rs`, consumed by the grad-aware
  `pub fn eigh` in `ferrotorch-core/src/linalg.rs`. EXACT for distinct
  eigenvalues; degenerate inputs diverge through `1/(w_j-w_i)` exactly as
  upstream `linalg_eig_backward` (FunctionsManual.cpp:3882-3917).
  **Eigenvector-gauge caveat (R-DEV-1, #1584):** eigenvectors are defined
  only up to a per-column sign (real case) / `e^{i phi}` (complex), which
  upstream documents at `FunctionsManual.cpp:3877-3880` ("The eigenvectors
  ... are specified up to multiplication by e^{i phi}. The specified loss
  function depends on this quantity, so it is ill-defined."). ferray's
  `eigh` forward (faer-backed) emits column signs that differ matrix-by-
  matrix from LAPACK `syevd` (what `torch.linalg.eigh` returns);
  EIGENVALUES match torch exactly. To give a stable contract,
  `canonicalize_eigenvector_signs` in `ferrotorch-core/src/linalg.rs`
  forces the largest-absolute-value component of each eigenvector column
  non-negative — a DETERMINISTIC ferrotorch convention. This does NOT match
  torch (torch does not canonicalize; matching its arbitrary LAPACK signs
  would require replicating `syevd`). The `EighBackwardV` VJP is
  sign-consistent (flipping a column of `U` flips the same column of `gU`;
  the skew-projection + `U @ ret @ U^T` conjugation is invariant under that
  joint flip), so for **SIGN-INVARIANT** losses `L(U)=L(U·diag(±1))` —
  PCA, whitening, `U @ diag(f(w)) @ U^T` reconstructions, every
  mathematically well-posed objective on eigenvectors — `A.grad` matches
  torch byte-for-byte. For **gauge-DEPENDENT** losses (a raw `<W, U>` linear
  functional of eigenvector entries) the gradient is convention-dependent
  and fundamentally cannot match torch's arbitrary LAPACK signs; that is a
  property of the ill-posed objective, not a backward-formula bug. The
  reframed `divergence_eigh_1577_eigenvector_sign_convention.rs` asserts the
  well-posed sign-invariant gradient vs LIVE torch float64 + eigenvector
  determinism.

- REQ-14: `linalg.eigvals(A)` — eigenvalues only (non-symmetric).
  Documented in `torch/linalg/__init__.py` (no separate `derivatives.yaml`
  entry — eigvals backward is the eigenvalues-only specialization of
  `linalg_eig`). **SHIPPED** (2026-05-27, closing #1345): `EigvalsBackward`
  + `eigvals_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs`
  implement the `!gV.defined()`, non-Hermitian SHORTCUT of
  `linalg_eig_backward` at `torch/csrc/autograd/FunctionsManual.cpp:3857-3862`:
  `gA = linalg_solve(V^H, gL.unsqueeze(-1) * V^H)` (where
  `gL.unsqueeze(-1) * V^H == diag(gL) @ V^H`), i.e.
  `gA = real(V^{-H} @ diag(gL) @ V^H)`. The COMPLEX cotangent `gL` is
  reconstructed from the `[n,2]` real cotangent as `re + i*im` (torch's
  conjugate-Wirtinger convention for a real loss of a complex output —
  verified `L.grad == cr + i*ci`), and all arithmetic runs on the `[.,2]`
  encoding via the private complex toolkit (shared with REQ-12). Because `A`
  is REAL the returned gradient is the REAL part (`handle_r_to_c`,
  `derivatives.yaml:1740`). Non-test production consumer: the grad-aware
  forward `pub fn eigvals` in `ferrotorch-core/src/linalg.rs` delegates to
  `eigvals_differentiable` when `is_grad_enabled() && a.requires_grad()`
  (forward + the eigenvectors `V` the VJP needs both computed under
  `no_grad` via `crate::linalg::eig`). Verified vs LIVE `torch 2.11.0`
  float64 `A.grad` for a 3×3 with DISTINCT REAL eigenvalues AND a 2×2
  COMPLEX-CONJUGATE pair (`L = 1 ± i`) by
  `grad_fns::linalg::tests::{eigvals_backward_real_3x3,
  eigvals_backward_complex_pair_2x2}_matches_torch` at `1e-6`. EXACT for
  DIAGONALIZABLE inputs; defective inputs return `SingularMatrix` from
  `c_inverse` (torch likewise has no defined gradient there). The CUDA
  forward stays forward-only.

- REQ-15: `linalg.eigvalsh(A, UPLO='L')` — eigenvalues only
  (symmetric/Hermitian). Documented in `torch/linalg/__init__.py`.
  **SHIPPED** (2026-05-27, closing #1577's eigvalsh subset):
  `EigvalshBackward` (`gA = sym(U diag(gw) U^T)`) + `eigvalsh_differentiable`
  in `ferrotorch-core/src/grad_fns/linalg.rs`, consumed by the grad-aware
  `pub fn eigvalsh` in `ferrotorch-core/src/linalg.rs`. EXACT — symmetric
  eigenvalues are smooth in `A` (no gauge / degeneracy issue). Mirrors the
  Hermitian eigvals shortcut of `linalg_eig_backward`
  (FunctionsManual.cpp:3859).

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
  `aten/src/ATen/native/LinearAlgebra.cpp`. **SHIPPED** (2026-05-27,
  closing #1577's pinv subset): `PinvBackward` (algebraic full-rank VJP,
  both `m<=n` and `m>n` branches) + `pinv_differentiable` in
  `ferrotorch-core/src/grad_fns/linalg.rs`, consumed by the grad-aware
  `pub fn pinv` in `ferrotorch-core/src/linalg.rs`. No eigendecomposition,
  so exact for full-rank `A`. Mirrors `pinv_backward`
  (FunctionsManual.cpp:2175).

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
  Documented in `torch/linalg/__init__.py`. **SHIPPED** (2026-05-27,
  full-rank solution-output VJP): `LstsqBackward` (`gB = pinv(A)^T gX`,
  `gA = pinv_backward(gX B^T, pinv(A), A)`) + `lstsq_differentiable` in
  `ferrotorch-core/src/grad_fns/linalg.rs`, consumed by the grad-aware
  `pub fn lstsq` in `ferrotorch-core/src/linalg.rs` (node attached to the
  `solution` output only; residuals/rank/singular_values are
  non-differentiable). Mirrors `linalg_lstsq_backward`
  (FunctionsManual.cpp:4012). Rank-deficient inputs are a residual
  follow-up (pinv is full-rank).

- REQ-23: `linalg.norm(A, ord=None, dim=None)` — generic norm (Frobenius
  for matrices, p-norm for vectors). Documented in
  `torch/linalg/__init__.py`. **SHIPPED** (2026-05-27, the Frobenius /
  Euclidean `p=2` case): `NormBackward` (`dx = grad * x / norm`, with the
  zero-norm `masked_fill` guard) + `matrix_norm_differentiable` /
  `vector_norm_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs`,
  consumed by the grad-aware `pub fn matrix_norm` and `pub fn vector_norm`
  (`ord==2.0`) in `ferrotorch-core/src/linalg.rs`. Mirrors `norm_backward`
  `p==2` (FunctionsManual.cpp:341). The non-`p=2` `ord` branches
  (inf / 1 / p<1) are a residual follow-up.

- REQ-24: `linalg.matrix_rank(A, tol=None)`. Mirrors `Tensor
  linalg_matrix_rank(...)` in `aten/src/ATen/native/LinearAlgebra.cpp`
  (overload family). **SHIPPED** (non-differentiable, mirrors torch):
  `matrix_rank` is integer-valued and has NO `derivatives.yaml` entry —
  PyTorch returns no gradient. ferrotorch's `pub fn matrix_rank` in
  `ferrotorch-core/src/linalg.rs` attaches no `GradFn`, matching the
  upstream "non-differentiable leaf-stop" autograd contract byte-for-byte
  (R-DEV-1). No `*Backward` needed.

- REQ-25: `linalg.cross(A, B, dim=-1)` — vector cross product along
  `dim` (must equal 3). **SHIPPED** (2026-05-27): `CrossBackward`
  (bilinear VJP `da = cross(b, grad)`, `db = cross(grad, a)`) +
  `cross_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs`,
  consumed by the grad-aware `pub fn cross` in
  `ferrotorch-core/src/linalg.rs`. Mirrors `linalg_cross`
  (derivatives.yaml:516-518).

- REQ-26: `linalg.householder_product(A, tau)`. Mirrors `Tensor
  linalg_householder_product(...)` in
  `aten/src/ATen/native/BatchLinearAlgebra.cpp` and documented in
  `torch/linalg/__init__.py`. **SHIPPED** (2026-05-27): `HouseholderProductBackward`
  + `householder_product_differentiable` in
  `ferrotorch-core/src/grad_fns/linalg.rs` implement the real
  reflector-recursion VJP — `input = tril(V,-1)` with unit diagonal,
  `sigma_j = tau_j / (tau_j ||input[:,j]||^2 - 1)` so
  `H(sigma_j) = H(tau_j)^{-1}`, `K = Q_full @ grad^T`, then the
  `K <- H_0^{-1} K` priming + the per-`i` `update_grad`
  (`v_grad = -tau_i*(vHK^T) - tau_i*Kv`, `tau_grad = -(vHK[i:]@v)`) and
  `K <- H_{i+1}^{-1} K H_i` advance, with `grad_V = tril(grad_V,-1)` — grounded
  in `householder_product_backward` (real, `flip_order=false`) at upstream
  `torch/csrc/autograd/FunctionsManual.cpp:5544`. The grad-aware forward `pub fn
  householder_product` in `ferrotorch-core/src/linalg.rs` now matches torch's
  output shape `[m, k]` (the leading `k` columns of `Q`, NOT the full `[m, m]`
  matrix) and delegates to `householder_product_differentiable` when
  `is_grad_enabled() && (v.requires_grad() || tau.requires_grad())`; the new
  `pub fn householder_product_full` provides the full `[m, m]`
  reconstruction the backward needs for `K = Q_full @ grad^T` (computed under
  `no_grad`). Verified vs LIVE torch 2.11.0 float64 `V.grad`/`tau.grad` for
  SQUARE (3×3), TALL-full (4×3), and TALL-truncated (4×2, `k<m`) inputs by
  `grad_fns::linalg::tests::householder_product_backward_{square_3x3,tall_4x3,
  tall_4x2}_matches_torch` at `1e-9`, plus the tau-only single-input path by
  `householder_product_backward_single_input_grad`. The complex case (the
  `.mH()`/`.conj()` terms) is out of scope — it needs complex-tensor autograd
  (#1345). The residual eig/eigvals (complex eigenvectors) remain the only
  NOT-STARTED ops under #1345.

- REQ-27: `linalg.lu(A, pivot=True)` — LU factorization with pivoting.
  Documented in `torch/linalg/__init__.py`. **SHIPPED** (2026-05-27, the
  square `m == n` case): split `LuBackwardL`/`LuBackwardU` (`m==n` VJP with
  the right/left triangular solves + `P^T` adjoint) + `lu_differentiable`
  in `ferrotorch-core/src/grad_fns/linalg.rs`, consumed by the grad-aware
  `pub fn lu` in `ferrotorch-core/src/linalg.rs` (`P` non-differentiable).
  Mirrors `linalg_lu_backward` `m==n` branch (FunctionsManual.cpp:6854).
  The rectangular wide/tall block-solve branch is a residual follow-up
  under #1577.

- REQ-28: `linalg.lu_factor(A)` — LU factorization without explicit
  unpacking. Documented in `torch/linalg/__init__.py`. **SHIPPED**
  (2026-05-27, square `m == n`): `LuFactorBackward` (decomposes the packed
  `LU` grad into `gL`/`gU` and reuses `LuBackwardShared::grad_a_combined`)
  + `lu_factor_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs`,
  consumed by the grad-aware `pub fn lu_factor` in
  `ferrotorch-core/src/linalg.rs` (pivots non-differentiable; CUDA stays
  forward-only). Mirrors `lu_factor_ex_backward`
  (FunctionsManual.cpp:6960). Rectangular case is a residual follow-up
  under #1577.

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
  **SHIPPED** (2026-05-27, closing #1583): `DiagonalBackward` /
  `diagonal_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs`;
  non-test production consumer: the now-grad-aware forward `pub fn
  diagonal` in `ferrotorch-core/src/linalg.rs` delegates to
  `diagonal_differentiable` when grad is enabled (forward computed under
  `no_grad`). FD-verified by
  `fn diagonal_public_forward_is_grad_aware_and_matches_fd`.

- REQ-31: `diag(A, diagonal=0)` — extract or construct a diagonal.
  **SHIPPED** (2026-05-27): `DiagBackward` / `diag_differentiable` in
  `ferrotorch-core/src/grad_fns/linalg.rs`; non-test production consumer:
  the now-grad-aware forward `pub fn diag` in
  `ferrotorch-core/src/ops/tensor_ops.rs` delegates to
  `diag_differentiable` when grad is enabled (forward under `no_grad`).
  FD-verified by `fn diag_extract_public_forward_is_grad_aware_and_matches_fd`
  + `fn diag_construct_public_forward_is_grad_aware_and_matches_fd`.

- REQ-32: `tril(A, diagonal=0)` — lower triangular zeroing.
  **SHIPPED** (2026-05-27): `TriangularBackward` / `tril_differentiable`
  in `ferrotorch-core/src/grad_fns/linalg.rs`; non-test production
  consumer: the now-grad-aware forward `pub fn tril` in
  `ferrotorch-core/src/ops/tensor_ops.rs` delegates to
  `tril_differentiable` when grad is enabled (forward under `no_grad`).
  FD-verified by `fn tril_public_forward_is_grad_aware_and_matches_fd`.

- REQ-33: `triu(A, diagonal=0)` — upper triangular zeroing.
  **SHIPPED** (2026-05-27): `triu_differentiable` (sharing
  `TriangularBackward`) in `ferrotorch-core/src/grad_fns/linalg.rs`;
  non-test production consumer: the now-grad-aware forward `pub fn triu`
  in `ferrotorch-core/src/ops/tensor_ops.rs` delegates to
  `triu_differentiable` when grad is enabled (forward under `no_grad`).
  FD-verified by `fn triu_public_forward_is_grad_aware_and_matches_fd`.

- REQ-34: `kron(A, B)` — Kronecker product. Mirrors `Tensor kron(const
  Tensor& self, const Tensor& other)` in
  `aten/src/ATen/native/LinearAlgebra.cpp`. **SHIPPED** (2026-05-27):
  `KronBackward` + `kron_differentiable` in
  `ferrotorch-core/src/grad_fns/linalg.rs`; non-test production consumer:
  the new grad-aware forward `pub fn kron` in
  `ferrotorch-core/src/linalg.rs` (the `torch.kron` public surface)
  delegates to `kron_differentiable`. FD-verified by
  `fn kron_public_forward_is_grad_aware_and_matches_fd`.

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
  `ferrotorch-core/src/grad_fns/linalg.rs` AND every one is now wired to a
  non-test production consumer (the grad-aware public forwards
  `crate::linalg::{addmm,addbmm,baddbmm,addmv,addr,kron,diagonal,trace,outer}`
  and `crate::ops::tensor_ops::{diag,tril,triu}` delegate to them) —
  closing #1583's consumer-wiring scope (2026-05-27).
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
  research-grade degenerate / gauge-freedom subset
  (svd/eigh/eigvalsh/pinv/lstsq/lu/lu_factor + cross/norm) is now wired
  end-to-end (REQ-11/13/15/19/22/23/25/27/28), closing the differentiable
  scope of #1577. `linalg.eig` / `linalg.eigvals` (REQ-12/14) gained their
  COMPLEX `*Backward` `GradFn`s + grad-aware forwards 2026-05-27 (closing
  #1345). The remaining no-`GradFn` ops are `matrix_rank`
  (non-differentiable, mirrors torch) and the rectangular LU /
  rank-deficient lstsq / non-`p=2` norm branches (residual #1577).

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

### REQ-5..REQ-9 addmm/addbmm/baddbmm/addmv/addr (SHIPPED)

The fused `BLAS-3` family `C = beta * self + alpha * mat1 @ mat2`
(addmm), the batched-sum `addbmm`, the per-batch `baddbmm`, the matrix-
vector `addmv`, and the rank-1-outer `addr` have `*Backward`
`GradFn` structs and `*_differentiable` wrappers in
`ferrotorch-core/src/grad_fns/linalg.rs` (`AddmmBackward`,
`AddbmmBackward`, `BaddbmmBackward`, `AddmvBackward`, `AddrBackward`).
As of 2026-05-27 (closing #1583), each is wired into a non-test
production consumer: the grad-aware public forwards `pub fn addmm` /
`addbmm` / `baddbmm` / `addmv` / `addr` in
`ferrotorch-core/src/linalg.rs` (the `torch.addmm` / `torch.addbmm` /
`torch.baddbmm` / `torch.addmv` / `torch.addr` public surfaces) delegate
to the matching `*_differentiable` wrapper, which computes the fused
forward inline and attaches the `GradFn` when
`is_grad_enabled() && any-operand.requires_grad()` (no `no_grad` re-entry
guard is needed because the wrapper does not call back into the public
forward). The fused `pub fn linear_fused` remains a related-but-distinct
op (hard-coded `A @ W^T + bias`); the general fused-affine API is now the
`addmm`-family forwards.

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

The closed-form-VJP `*Backward` structs that shipped 2026-05-27 (closing
the remaining scope of #1583):
- the fused-affine family (`addmm`/`addbmm`/`baddbmm`/`addmv`/`addr`) and
  the structural `kron`/`diag`/`tril`/`triu`/`diagonal` — their grad-aware
  public forwards (`crate::linalg::{addmm,addbmm,baddbmm,addmv,addr,kron,
  diagonal}` and `crate::ops::tensor_ops::{diag,tril,triu}`) now delegate
  to the matching `*_differentiable` wrapper, mirroring the
  qr/cholesky/slogdet and solve/inv/det/trace/outer pattern above. With
  these wired, **#1583 is fully resolved**. The research-grade
  factorizations svd/eigh/eigvalsh/pinv/lstsq/lu/lu_factor (the #1577
  degenerate / gauge-freedom subset) plus cross/norm are now SHIPPED;
  only eig/eigvals/matrix_rank/householder_product (under #1345) and the
  rectangular-LU / rank-deficient-lstsq / non-`p=2`-norm residuals
  (#1577) remain forward-only.

The QR multi-output case is handled by SPLITTING the jointly-linear
`(gQ, gR)` VJP across two single-output nodes — `QrBackwardQ` on the `Q`
output (the `gQ`-only partial) and `QrBackwardR` on the `R` output (the
`gR`-only partial). The reverse-mode engine accumulates both partials
into `A.grad`, reproducing the joint formula (which is additive in `gQ`
and `gR`); a consumer that uses only one output simply gets the other
partial as zero, matching PyTorch's undefined-grad semantics. Slogdet
likewise attaches its node only to the differentiable `logabsdet`
output, leaving `sign` plain (non-differentiable in the real case).

`linalg.svd` is **SHIPPED** (2026-05-27, closing #1577's svd subset): the
split `SvdBackwardU`/`SvdBackwardS`/`SvdBackwardV` + `svd_differentiable`
implement the real reduced-SVD VJP — the F-matrix `E[i,j] = S²[j] - S²[i]`
off-diagonal / `1` on the diagonal, the symmetrized core
`(skew(U^T gU)·S + S·skew(Vh gVh^T))/E + diag(gS)`, AND the rectangular
projectors `(I_m - UU^T) gU S⁻¹ V^T` (for `m>n`) / `U S⁻¹ gV^T (I_n -
VV^T)` (for `m<n`), grounded in `svd_backward` at
`FunctionsManual.cpp:3605`. The 3-output joint VJP is split across the
`U`/`S`/`Vh` outputs (same strategy as QR's Q/R and eigh's W/V) and
accumulated into `A.grad`. Consumed by the grad-aware `pub fn svd` in
`ferrotorch-core/src/linalg.rs`; the complex case (the `Im(diag(...))`
invariance terms) is out of scope (#1345). With this landed the
differentiable scope of #1577 is complete.

`linalg.householder_product` is **SHIPPED** (2026-05-27, REQ-26):
`HouseholderProductBackward` + `householder_product_differentiable` implement
the real reflector-recursion VJP grounded in `householder_product_backward`
(real, `flip_order=false`) at `FunctionsManual.cpp:5544`. As part of this the
forward `pub fn householder_product` in `ferrotorch-core/src/linalg.rs` was
corrected to return torch's `[m, k]` shape (the leading `k` columns of `Q`,
not the full `[m, m]` matrix); the new `pub fn householder_product_full`
exposes the `[m, m]` reconstruction the backward needs. The grad-aware
forward is the non-test production consumer. Verified vs LIVE torch float64.

`linalg.eig` / `linalg.eigvals` are **SHIPPED** (2026-05-27, closing #1345,
REQ-12/14): `EigBackwardW`/`EigBackwardV` (eig) and `EigvalsBackward`
(eigvals) implement the non-Hermitian COMPLEX `linalg_eig_backward` VJP
(`FunctionsManual.cpp:3820`) on the `[.,2]` real/imag encoding via the
private complex-linalg toolkit, returning the real part for the real input
`A` (`handle_r_to_c`, `derivatives.yaml:1740`). The grad-aware `pub fn eig` /
`pub fn eigvals` forwards in `ferrotorch-core/src/linalg.rs` are the non-test
production consumers (the `eig` forward also unit-norm-normalizes ferray's
eigenvector columns to match torch's norm-one contract). EXACT for
diagonalizable A; the eig eigenvector PHASE gauge is handled by a
phase-invariant test loss (R-DEV-1). With these wired the differentiable
scope of #1345 is complete.

The factorizations that remain forward-only in
`ferrotorch-core/src/linalg.rs` (routed through `ferray_linalg::*`, no
`*Backward` `GradFn`): `linalg.matrix_rank` (non-differentiable, mirrors
torch). The rectangular `m != n` LU / rank-deficient `lstsq` / non-`p=2`
`norm` branches are residual follow-ups under #1577.

### REQ-29..REQ-35 trace/diagonal/diag/tril/triu/kron/outer

`trace` (REQ-29) and `outer` (REQ-35) are **SHIPPED** end-to-end as of
2026-05-27: their forwards `pub fn trace` / `pub fn outer` in
`ferrotorch-core/src/linalg.rs` delegate to `trace_differentiable` /
`outer_differentiable` (forward computed under `no_grad`) when grad is
enabled, and each VJP is FD-verified by a public-forward-driven test in
this file's `#[cfg(test)] mod tests`. The grad-aware forward is the
non-test production consumer.

The remainder (`diagonal`/`diag`/`tril`/`triu`/`kron`) shipped end-to-end
2026-05-27 (closing the last scope of #1583): each `*Backward` `GradFn` +
`*_differentiable` wrapper in `ferrotorch-core/src/grad_fns/linalg.rs` is
now consumed by a grad-aware public forward:

- `diagonal`: the now-grad-aware `pub fn diagonal` in
  `ferrotorch-core/src/linalg.rs` delegates to `diagonal_differentiable`
  (forward computed under `no_grad`). `DiagonalBackward`.
- `diag`: the now-grad-aware `pub fn diag` in
  `ferrotorch-core/src/ops/tensor_ops.rs` delegates to
  `diag_differentiable` (forward under `no_grad`). `DiagBackward`.
- `tril`: the now-grad-aware `pub fn tril` in
  `ferrotorch-core/src/ops/tensor_ops.rs` delegates to
  `tril_differentiable` (forward under `no_grad`). `TriangularBackward`.
- `triu`: the now-grad-aware `pub fn triu` in
  `ferrotorch-core/src/ops/tensor_ops.rs` delegates to
  `triu_differentiable` (sharing `TriangularBackward`; forward under
  `no_grad`).
- `kron`: the new grad-aware `pub fn kron` in
  `ferrotorch-core/src/linalg.rs` delegates to `kron_differentiable`.
  `KronBackward`.

With REQ-30..34 wired, **#1583 is fully resolved** — every op it covers is
grad-aware end-to-end.

## Parity contract

| Op | Upstream entry | Backward formula source | Edge cases mirrored |
|---|---|---|---|
| `mm` | `TORCH_IMPL_FUNC(mm_out_cpu)` in `aten/src/ATen/native/LinearAlgebra.cpp` | `dA = grad_C @ B^T`, `dB = A^T @ grad_C` | Inner-dim mismatch → `FerrotorchError::ShapeMismatch`; bf16/f16 on GPU use cuBLAS GemmEx with f32 accumulator; autocast f32 → f16 accumulator path; CPU is zero-copy raw-slice loops. SHIPPED (REQ-1). |
| `bmm` | `TORCH_IMPL_FUNC(bmm_out_cpu)` in `aten/src/ATen/native/LinearAlgebra.cpp` | per-batch `mm` VJP composed via `batch_transpose` | 3D inputs only (Err on other ranks); batch-dim mismatch → `ShapeMismatch`; CUDA via `SgemmStridedBatched`/`DgemmStridedBatched`; autocast f32→ `bmm_f16_f32` Tensor Core path. SHIPPED (REQ-2). |
| `matmul` | `Tensor matmul(...)` in `aten/src/ATen/native/LinearAlgebra.cpp` | rank-dispatched: dot/mv/vm/mm/bmm/broadcast_bmm | 1D×1D=dot; 2D×1D=mv; 1D×2D=vm; 2D×2D=mm; 3D×3D=bmm; broadcast ≥3D=gemmStridedBatched with stride-0 on broadcasted axes. CUDA covers all of these for f32/f64; bf16/f16 covered for 2D×2D; other dtype combos err `NotImplementedOnCuda`. SHIPPED (REQ-3). |
| `linalg.matmul` | `Tensor linalg_matmul(...)` in `aten/src/ATen/native/LinearAlgebra.cpp` | alias for `matmul` | upstream `linalg_matmul` is literally `at::matmul(tensor1, tensor2)`; ferrotorch's `Tensor::matmul` covers both since the Python API surface is the same. SHIPPED (REQ-4). |
| `addmm` | `TORCH_IMPL_FUNC(addmm_out_cpu)` in `aten/src/ATen/native/LinearAlgebra.cpp` | dself=beta·grad, dmat1=alpha·grad·mat2^T, dmat2=alpha·mat1^T·grad | SHIPPED (REQ-5): `AddmmBackward` + grad-aware `pub fn addmm` forward delegating to `addmm_differentiable`. FD-verified. |
| `addbmm` | `Tensor addbmm(...)` in `aten/src/ATen/native/LinearAlgebra.cpp` | dself=beta·grad, dbatch1=alpha·grad·batch2^T (broadcast over batch), dbatch2=alpha·batch1^T·grad (sum over batch) | SHIPPED (REQ-6): `AddbmmBackward` + grad-aware `pub fn addbmm` forward. FD-verified. |
| `baddbmm` | `TORCH_IMPL_FUNC(baddbmm_out_cpu)` in `aten/src/ATen/native/LinearAlgebra.cpp` | per-batch addmm-like VJP | SHIPPED (REQ-7): `BaddbmmBackward` + grad-aware `pub fn baddbmm` forward. FD-verified. |
| `addmv` | `TORCH_IMPL_FUNC(addmv_out_cpu)` in `aten/src/ATen/native/Blas.cpp` | dself=beta·grad, dmat=alpha·outer(grad,vec), dvec=alpha·mat^T·grad | SHIPPED (REQ-8): `AddmvBackward` + grad-aware `pub fn addmv` forward. FD-verified. |
| `addr` | `Tensor addr(...)` in `aten/src/ATen/native/LinearAlgebra.cpp` | dself=beta·grad, dvec1=alpha·grad@vec2, dvec2=alpha·vec1^T@grad | SHIPPED (REQ-9): `AddrBackward` + grad-aware `pub fn addr` forward. FD-verified. |
| `linalg.solve` | `Tensor linalg_solve(...)` in `aten/src/ATen/native/BatchLinearAlgebra.cpp` | `gB = A^{-T} @ gX`, `gA = -gB @ X^T` (`linalg_solve_backward` in `FunctionsManual.cpp`) | SHIPPED (REQ-10): `LinalgSolveBackward` + grad-aware `pub fn solve` forward delegating to `solve_differentiable`. FD-verified (matrix + vector RHS). |
| `linalg.svd` | `svd = _add_docstr(...)` in `torch/linalg/__init__.py` | 3-output reduced F-matrix + rectangular projectors | `SvdBackwardU`/`SvdBackwardS`/`SvdBackwardV` + `svd_differentiable`; grad-aware `crate::linalg::svd` consumer; verified vs LIVE torch float64 (square/tall/wide/S-only). SHIPPED (closes #1577 svd subset). |
| `linalg.eig` | `eig = _add_docstr(...)` in `torch/linalg/__init__.py` | COMPLEX F-matrix VJP | No `GradFn`; forward-only. NOT-STARTED. Blocker #1345 (needs complex-tensor autograd primitive — ferrotorch eig returns `[n,2]` real/imag). |
| `linalg.eigh` | `eigh = _add_docstr(...)` in `torch/linalg/__init__.py` | sym F-matrix (real spectrum) | SHIPPED (REQ-13): split `EighBackwardW`/`EighBackwardV` + grad-aware `pub fn eigh`. Exact for distinct eigenvalues; degenerate diverges as torch. FD-verified. |
| `linalg.eigvals` | `eigvals = _add_docstr(...)` in `torch/linalg/__init__.py` | COMPLEX eigvals shortcut of `eig` VJP | No `GradFn`; forward-only. NOT-STARTED. Blocker #1345 (same complex-tensor primitive as `eig`). |
| `linalg.eigvalsh` | `eigvalsh = _add_docstr(...)` in `torch/linalg/__init__.py` | sym variant `gA = sym(U diag(gw) U^T)` | SHIPPED (REQ-15): `EigvalshBackward` + grad-aware `pub fn eigvalsh`. Exact (no gauge/degeneracy for eigenvalues). FD-verified. |
| `linalg.qr` | `qr = _add_docstr(...)` in `torch/linalg/__init__.py` | Q-orthog + R-triangular VJP (reduced, m≥n) | SHIPPED (REQ-16): `QrBackwardQ`/`QrBackwardR` + grad-aware `pub fn qr` forward. m<n branch tracked under #1577. |
| `linalg.cholesky` | `Tensor linalg_cholesky(...)` in `aten/src/ATen/native/BatchLinearAlgebra.cpp` | `gA = L^{-T} Φ(tril(L^T gL)) L^{-1}` | SHIPPED (REQ-17): `CholeskyBackward` + grad-aware `pub fn cholesky` forward. |
| `linalg.inv` | `Tensor linalg_inv(...)` in `aten/src/ATen/native/BatchLinearAlgebra.cpp` | `dA = −Y^T @ grad @ Y^T` where Y = A^{-1} | SHIPPED (REQ-18): `LinalgInvBackward` + grad-aware `pub fn inv` forward delegating to `inv_differentiable`. FD-verified. |
| `linalg.pinv` | `Tensor linalg_pinv(...)` in `aten/src/ATen/native/LinearAlgebra.cpp` | algebraic pinv VJP (`pinv_backward`) | SHIPPED (REQ-19): `PinvBackward` (both `m<=n`/`m>n`) + grad-aware `pub fn pinv`. Exact full-rank. FD-verified tall+wide. |
| `linalg.det` | `Tensor linalg_det(...)` in `aten/src/ATen/native/LinearAlgebra.cpp` | `dA = det(A) · A^{-T} · grad` | SHIPPED (REQ-20): `LinalgDetBackward` + grad-aware `pub fn det` forward delegating to `det_differentiable`. FD-verified. |
| `linalg.slogdet` | `slogdet = _add_docstr(...)` in `torch/linalg/__init__.py` | `dA = A^{-T} · grad_logabs` (sign-grad is 0) | SHIPPED (REQ-21): `SlogdetBackward` + grad-aware `pub fn slogdet` forward. |
| `linalg.lstsq` | `lstsq = _add_docstr(...)` in `torch/linalg/__init__.py` | solution VJP via `pinv_backward` | SHIPPED (REQ-22, full-rank): `LstsqBackward` (solution output only) + grad-aware `pub fn lstsq`. FD-verified dA+dB. Rank-deficient residual. |
| `linalg.norm` | `norm = _add_docstr(...)` in `torch/linalg/__init__.py` | Frobenius/Euclidean `p=2`: `dx = grad · x / norm` | SHIPPED (REQ-23, `p=2`): `NormBackward` + grad-aware `matrix_norm`/`vector_norm`. FD-verified. Non-`p=2` ords residual. |
| `linalg.matrix_rank` | `Tensor linalg_matrix_rank(...)` in `aten/src/ATen/native/LinearAlgebra.cpp` | rank is integer; NON-DIFFERENTIABLE | SHIPPED (REQ-24): no `GradFn` — mirrors torch's no-`derivatives.yaml`-entry / no-grad contract (R-DEV-1). |
| `linalg.cross` | `cross = _add_docstr(...)` in `torch/linalg/__init__.py` | `da = cross(b, grad)`, `db = cross(grad, a)` | SHIPPED (REQ-25): `CrossBackward` (bilinear) + grad-aware `pub fn cross`. FD-verified dA+dB. |
| `linalg.householder_product` | `Tensor linalg_householder_product(...)` in `aten/src/ATen/native/BatchLinearAlgebra.cpp` | reflector-recursion VJP (`householder_product_backward` in `FunctionsManual.cpp`) | SHIPPED (REQ-26): `HouseholderProductBackward` + `householder_product_differentiable`; grad-aware `pub fn householder_product` (now `[m,k]`-shaped, matching torch) delegates here; `householder_product_full` gives the `[m,m]` reconstruction for the VJP. Verified vs LIVE torch float64 (square/tall-full/tall-truncated). Complex case under #1345. |
| `linalg.lu` | `lu = _add_docstr(...)` in `torch/linalg/__init__.py` | `m==n`: tri-solve VJP + `P^T` adjoint | SHIPPED (REQ-27, square): split `LuBackwardL`/`LuBackwardU` + grad-aware `pub fn lu`. FD-verified pivoted. Rectangular residual #1577. |
| `linalg.lu_factor` | `lu_factor = _add_docstr(...)` in `torch/linalg/__init__.py` | same as `lu` minus the explicit unpacking | SHIPPED (REQ-28, square): `LuFactorBackward` (packed) + grad-aware `pub fn lu_factor`. FD-verified pivoted. Rectangular residual #1577. |
| `trace` | upstream tensor method (no dedicated impl in `LinearAlgebra.cpp`) | `dA = grad · I` (`trace_backward_symint` in `derivatives.yaml`) | SHIPPED (REQ-29): `TraceBackward` + grad-aware `pub fn trace` forward delegating to `trace_differentiable`. FD-verified. |
| `diagonal` | `Tensor linalg_diagonal(...)` in `aten/src/ATen/native/LinearAlgebra.cpp` | inverse of `diag_embed` — scatter grad onto the diagonal of zeros | SHIPPED (REQ-30): `DiagonalBackward` + now-grad-aware `pub fn diagonal` forward (in `linalg.rs`) delegating to `diagonal_differentiable`. FD-verified. |
| `diag` | upstream tensor method `Tensor::diag(...)` | scatter or extract VJP based on input rank | SHIPPED (REQ-31): `DiagBackward` + now-grad-aware `pub fn diag` forward (in `ops/tensor_ops.rs`) delegating to `diag_differentiable`. FD-verified (extract + construct). |
| `tril` | upstream tensor method `Tensor::tril(...)` | grad masked by lower triangular | SHIPPED (REQ-32): `TriangularBackward` + now-grad-aware `pub fn tril` forward (in `ops/tensor_ops.rs`) delegating to `tril_differentiable`. FD-verified. |
| `triu` | upstream tensor method `Tensor::triu(...)` | grad masked by upper triangular | SHIPPED (REQ-33): `triu_differentiable` (shares `TriangularBackward`) + now-grad-aware `pub fn triu` forward (in `ops/tensor_ops.rs`). FD-verified. |
| `kron` | `Tensor kron(...)` in `aten/src/ATen/native/LinearAlgebra.cpp` | `dA = sum over kron-blocks of grad·B^T`, `dB = sum over kron-blocks of A^T·grad` | SHIPPED (REQ-34): `KronBackward` + new grad-aware `pub fn kron` forward (in `linalg.rs`) delegating to `kron_differentiable`. FD-verified. |
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
| REQ-5 (addmm) | SHIPPED | impl: `pub struct AddmmBackward` + `pub fn addmm_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs` (VJP `dself=beta·grad`, `dmat1=alpha·grad·mat2^T`, `dmat2=alpha·mat1^T·grad`, mirroring `TORCH_IMPL_FUNC(addmm_out_cpu)` at `aten/src/ATen/native/LinearAlgebra.cpp:1620` + `addmm` at `tools/autograd/derivatives.yaml:256`). FD-verified by `fn addmm_public_forward_is_grad_aware_and_matches_fd` in the in-file `#[cfg(test)] mod tests` (drives the public forward; checks `dself`/`dmat1`/`dmat2` vs central FD). Non-test production consumer: the grad-aware forward `pub fn addmm` in `ferrotorch-core/src/linalg.rs` (the `torch.addmm` public surface) delegates to `addmm_differentiable`; the wrapper attaches the `GradFn` when grad is enabled (the wrapper computes the fused-affine forward inline, so no re-entry guard is needed). |
| REQ-6 (addbmm) | SHIPPED | impl: `pub struct AddbmmBackward` + `pub fn addbmm_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs` (mirroring `Tensor addbmm(...)` at `aten/src/ATen/native/LinearAlgebra.cpp:1615` + `addbmm` at `tools/autograd/derivatives.yaml:238`). FD-verified by `fn addbmm_public_forward_is_grad_aware_and_matches_fd`. Non-test production consumer: the grad-aware forward `pub fn addbmm` in `ferrotorch-core/src/linalg.rs` delegates to `addbmm_differentiable`. |
| REQ-7 (baddbmm) | SHIPPED | impl: `pub struct BaddbmmBackward` + `pub fn baddbmm_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs` (mirroring `TORCH_IMPL_FUNC(baddbmm_out_cpu)` at `aten/src/ATen/native/LinearAlgebra.cpp:1886` + `baddbmm` at `tools/autograd/derivatives.yaml:359`). FD-verified by `fn baddbmm_public_forward_is_grad_aware_and_matches_fd`. Non-test production consumer: the grad-aware forward `pub fn baddbmm` in `ferrotorch-core/src/linalg.rs` delegates to `baddbmm_differentiable`. |
| REQ-8 (addmv) | SHIPPED | impl: `pub struct AddmvBackward` + `pub fn addmv_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs` (mirroring `TORCH_IMPL_FUNC(addmv_out_cpu)` at `aten/src/ATen/native/Blas.cpp:72` + `addmv` at `tools/autograd/derivatives.yaml:267`). FD-verified by `fn addmv_public_forward_is_grad_aware_and_matches_fd`. Non-test production consumer: the grad-aware forward `pub fn addmv` in `ferrotorch-core/src/linalg.rs` delegates to `addmv_differentiable`. |
| REQ-9 (addr) | SHIPPED | impl: `pub struct AddrBackward` + `pub fn addr_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs` (mirroring `Tensor addr(...)` at `aten/src/ATen/native/LinearAlgebra.cpp:1200` + `addr` at `tools/autograd/derivatives.yaml:273`). FD-verified by `fn addr_public_forward_is_grad_aware_and_matches_fd`. Non-test production consumer: the grad-aware forward `pub fn addr` in `ferrotorch-core/src/linalg.rs` delegates to `addr_differentiable`. |
| REQ-10 (linalg.solve) | SHIPPED | impl: `pub struct LinalgSolveBackward` + `pub fn solve_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs` (VJP `gB = A^{-T} @ gX`, `gA = -gB @ X^T`, mirroring `linalg_solve_backward` at `torch/csrc/autograd/FunctionsManual.cpp:6160`). FD-verified by `fn solve_forward_is_grad_aware_and_matches_fd_matrix_rhs` + `fn solve_forward_is_grad_aware_and_matches_fd_vector_rhs` in the in-file `#[cfg(test)] mod tests` (both drive the public forward and check `A.grad`/`B.grad` vs central FD). Non-test production consumer: the grad-aware forward `pub fn solve` in `ferrotorch-core/src/linalg.rs` (the `torch.linalg.solve` public surface) delegates to `solve_differentiable` when `!a.is_cuda() && is_grad_enabled() && (a.requires_grad() || b.requires_grad())`; the wrapper computes the forward under `no_grad` to avoid re-entry. |
| REQ-11 (linalg.svd) | SHIPPED | impl: split `struct SvdBackwardU` + `struct SvdBackwardS` + `struct SvdBackwardV` (sharing `SvdBackwardShared`) + `pub fn svd_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs` implement the real reduced-SVD VJP mirroring `svd_backward` at `torch/csrc/autograd/FunctionsManual.cpp:3605`: the F-matrix `E[i,j] = S²[j] - S²[i]` off-diagonal / `1` on the diagonal (`3770-3777`), the symmetrized core `(skew(U^T gU)·S + S·skew(Vh gVh^T))/E + diag(gS)` (`3767-3797`), AND the rectangular projectors `(I_m - UU^T) gU S⁻¹ V^T` for `m>n` / `U S⁻¹ gV^T (I_n - VV^T)` for `m<n` (`3799-3815`). The 3-output `(U,S,Vh)` jointly-linear VJP is split across the three single-output nodes (`SvdBackwardU` on `U`, `SvdBackwardS` on `S`, `SvdBackwardV` on `Vh`) and accumulated into `A.grad` — same split-node strategy as QR (Q/R) and eigh (W/V); an `S`-only consumer gets just the `SvdBackwardS` partial (torch's svdvals optimisation, `3738-3741`). Verified against LIVE `torch 2.11.0+cu130` float64 `A.grad` for SQUARE 3×3, TALL 4×3 (`m>n` projector), WIDE 3×4 (`m<n` projector), and S-only by `fn svd_backward_{square_3x3,tall_4x3,wide_3x4,s_only_square_3x3,s_only_tall_4x3}_matches_torch` at `1e-6`, using a gauge-invariant loss (gauge caveat mirrors eigh #1584: ferray=faer signs differ from torch=LAPACK, but the VJP is sign-consistent, so sign-invariant losses match exactly). Non-test production consumer: the grad-aware forward `pub fn svd` in `ferrotorch-core/src/linalg.rs` delegates to `svd_differentiable` when `!input.is_cuda() && is_grad_enabled() && input.requires_grad()` (forward under `no_grad`). The complex case (the `Im(diag(...))` invariance terms) is out of scope (#1345). Closes #1577 svd subset. |
| REQ-12 (linalg.eig) | NOT-STARTED | open prereq blocker #1345 — CONCRETE: needs complex-tensor autograd support. `linalg_eig_backward` (non-Hermitian, `torch/csrc/autograd/FunctionsManual.cpp:3820`) operates on COMPLEX eigenvalues/eigenvectors (`V.mH()`, `at::linalg_solve(V.mH(), ...)`, complex `Econj`). ferrotorch's `pub fn eig` returns eigenvalues/vectors as `[n,2]`/`[n,n,2]` real/imag-encoded tensors (no native `Complex<T>` `Tensor` with a complex autograd `GradFn`), so the conjugate-transpose VJP cannot be expressed without a complex-tensor primitive. Forward-only `pub fn eig` in `ferrotorch-core/src/linalg.rs`. |
| REQ-13 (linalg.eigh) | SHIPPED | impl: split `pub struct EighBackwardW` + `pub struct EighBackwardV` (sharing `EighBackwardShared`) + `pub fn eigh_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs` implement the real symmetric F-matrix VJP `gA = sym(U[(skew(U^T gU)/Econj) + diag(gw)]U^T)` with the skew-symmetric projection `0.5*(VhgV - VhgV^T)` and `Econj_{ij} = w_j - w_i` off-diagonal (1 on the diagonal), mirroring `linalg_eig_backward` (Hermitian branch, real case) at `torch/csrc/autograd/FunctionsManual.cpp:3882-3917`; the jointly-linear `(gw, gU)` VJP is split across the two single-output nodes and accumulated into `A.grad`. FD-verified on a distinct-eigenvalue well-conditioned symmetric 3x3 by `fn eigh_public_forward_is_grad_aware_and_matches_fd` (weighted sum over BOTH eigenvalues and eigenvectors, analytic-vs-FD both symmetrized, tol 2e-3). Non-test production consumer: the grad-aware forward `pub fn eigh` in `ferrotorch-core/src/linalg.rs` (the `torch.linalg.eigh` public surface) delegates to `eigh_differentiable` when `!a.is_cuda() && is_grad_enabled() && a.requires_grad()` (forward under `no_grad`). Degenerate-eigenvalue inputs diverge through `1/(w_j-w_i)` exactly as PyTorch's `linalg_eig_backward` does (no special-casing upstream). |
| REQ-14 (linalg.eigvals) | NOT-STARTED | open prereq blocker #1345 — same complex-tensor blocker as REQ-12. The eigvals-only shortcut of `linalg_eig_backward` (non-Hermitian, `at::linalg_solve(V.mH(), gL.unsqueeze(-1) * V.mH())`, `FunctionsManual.cpp:3861`) is still complex-valued (general eigenvalues are complex even for real `A`), so it needs the same complex-tensor autograd primitive REQ-12 lacks. Forward-only `pub fn eigvals` in `ferrotorch-core/src/linalg.rs`. |
| REQ-15 (linalg.eigvalsh) | SHIPPED | impl: `pub struct EigvalshBackward` + `pub fn eigvalsh_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs` implement the VJP `gA = sym(U @ diag(gw) @ U^T)`, the Hermitian eigenvalues-only shortcut of `linalg_eig_backward` (`at::matmul(V * gL.unsqueeze(-2), V.mH())`) at `torch/csrc/autograd/FunctionsManual.cpp:3859`. Symmetric eigenvalues are smooth functions of `A` with NO eigenvector-gauge / degenerate issue, so the VJP is EXACT; symmetrized to match torch's UPLO-triangle contract. FD-verified by `fn eigvalsh_public_forward_is_grad_aware_and_matches_fd` (distinct-eigenvalue symmetric 3x3, weighted-sum loss, symmetrized analytic-vs-FD, tol 1e-4). Non-test production consumer: the grad-aware forward `pub fn eigvalsh` in `ferrotorch-core/src/linalg.rs` (the `torch.linalg.eigvalsh` public surface) delegates to `eigvalsh_differentiable` when `!a.is_cuda() && is_grad_enabled() && a.requires_grad()` (forward + eigenvector solve under `no_grad`). |
| REQ-16 (linalg.qr) | SHIPPED | impl: `pub struct QrBackwardQ` + `pub struct QrBackwardR` + `pub fn qr_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs` implement the real `linalg_qr_backward` VJP (reduced, m≥n: `gA = (Q @ syminvadj(triu(M)) + gQ) @ R^{-T}`, `M = gR @ R^T - Q^T @ gQ`), mirroring upstream `linalg_qr_backward` in `FunctionsManual.cpp` and `linalg_qr` in `derivatives.yaml`; the joint `(gQ,gR)` VJP is split across the two single-output nodes and accumulated into `A.grad`. FD-verified by `fn qr_backward_matches_finite_difference_square` + `fn qr_backward_q_only_and_r_only` in the in-file `#[cfg(test)] mod tests`. Non-test production consumer: the grad-aware forward `pub fn qr` in `ferrotorch-core/src/linalg.rs` (the `torch.linalg.qr` public surface) delegates to `qr_differentiable` when `is_grad_enabled() && input.requires_grad()`. m<n (`trilImInvAdjSkew`) branch tracked under sub-blocker #1577. |
| REQ-17 (linalg.cholesky) | SHIPPED | impl: `pub struct CholeskyBackward` + `pub fn cholesky_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs` implement the Phi-symmetrisation VJP `gA = L^{-T} Φ(tril(L^T gL)) L^{-1}`, mirroring upstream `cholesky_backward` in `FunctionsManual.cpp` and `cholesky` in `derivatives.yaml`; the two triangular solves reuse `pub fn solve_triangular` in `ferrotorch-core/src/linalg.rs`, and the returned gradient is symmetric (PyTorch contract). FD-verified by `fn cholesky_backward_matches_finite_difference` (symmetric-FD + symmetry assertion) in the in-file `#[cfg(test)] mod tests`. Non-test production consumer: the grad-aware forward `pub fn cholesky` in `ferrotorch-core/src/linalg.rs` (the `torch.linalg.cholesky` public surface) delegates to `cholesky_differentiable` when grad is enabled. |
| REQ-18 (linalg.inv) | SHIPPED | impl: `pub struct LinalgInvBackward` + `pub fn inv_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs` (VJP `dA = -Y^T @ grad @ Y^T`, `Y = A^{-1}`, mirroring `linalg_inv_ex` at `tools/autograd/derivatives.yaml:916`). FD-verified by `fn inv_forward_is_grad_aware_and_matches_fd` in the in-file `#[cfg(test)] mod tests` (drives the public forward, loss = sum(Y), checks `A.grad` vs central FD). Non-test production consumer: the grad-aware forward `pub fn inv` in `ferrotorch-core/src/linalg.rs` (the `torch.linalg.inv` public surface) delegates to `inv_differentiable` when `is_grad_enabled() && input.requires_grad()`; the wrapper computes the forward under `no_grad` to avoid re-entry. |
| REQ-19 (linalg.pinv) | SHIPPED | impl: `pub struct PinvBackward` + `pub fn pinv_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs` implement the algebraic Moore-Penrose VJP (both `m <= n` and `m > n` branches expressed through `pinvA`, `grad`, `A`), mirroring `pinv_backward` at `torch/csrc/autograd/FunctionsManual.cpp:2175`. No eigendecomposition, so NO degenerate-singular-value issue — exact for full-rank `A`. FD-verified on a tall 4x2 (`fn pinv_public_forward_is_grad_aware_and_matches_fd_tall`) AND a wide 2x4 (`fn pinv_public_forward_is_grad_aware_and_matches_fd_wide`), both weighted-sum loss vs central FD at tol 1e-3, exercising both `m>n` and `m<=n` branches. Non-test production consumer: the grad-aware forward `pub fn pinv` in `ferrotorch-core/src/linalg.rs` (the `torch.linalg.pinv` public surface) delegates to `pinv_differentiable` when `is_grad_enabled() && input.requires_grad()` (forward under `no_grad`). |
| REQ-20 (linalg.det) | SHIPPED | impl: `pub struct LinalgDetBackward` + `pub fn det_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs` (VJP `dA = det(A) * grad * inv(A)^T`, the invertible branch of `linalg_det_backward` at `torch/csrc/autograd/FunctionsManual.cpp:4373`). FD-verified by `fn det_forward_is_grad_aware_and_matches_fd` in the in-file `#[cfg(test)] mod tests` (drives the public forward, checks `A.grad` vs central FD). Non-test production consumer: the grad-aware forward `pub fn det` in `ferrotorch-core/src/linalg.rs` (the `torch.linalg.det` public surface) delegates to `det_differentiable` when `is_grad_enabled() && input.requires_grad()`; the wrapper computes the forward (and the VJP's internal `inv`) under `no_grad` to avoid re-entry. |
| REQ-21 (linalg.slogdet) | SHIPPED | impl: `pub struct SlogdetBackward` + `pub fn slogdet_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs` attach the real-case VJP `dA = grad_logabsdet * inv(A)^T` to the differentiable `logabsdet` output (`sign` is non-differentiable in the real case, returned plain), mirroring upstream `slogdet_backward` in `FunctionsManual.cpp` and `_linalg_slogdet` in `derivatives.yaml`. FD-verified by `fn slogdet_backward_matches_finite_difference` in the in-file `#[cfg(test)] mod tests`. Non-test production consumer: the grad-aware forward `pub fn slogdet` in `ferrotorch-core/src/linalg.rs` (the `torch.linalg.slogdet` public surface) delegates to `slogdet_differentiable` when grad is enabled. |
| REQ-22 (linalg.lstsq) | SHIPPED | impl: `pub struct LstsqBackward` + `pub fn lstsq_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs` implement the full-rank solution-output VJP `gB = pinv(A)^T @ gX`, `gA = pinv_backward(gX @ B^T, pinv(A), A)` (reusing `PinvBackward::compute`), mirroring `linalg_lstsq_backward`'s solution-from-`gX` branch at `torch/csrc/autograd/FunctionsManual.cpp:4038-4050`. The `residuals`/`rank`/`singular_values` outputs are non-differentiable (`output_differentiability: [True, False, False, False]`, `derivatives.yaml:1056`) and returned plain; the node is attached to `solution` only. FD-verified (both `A.grad` and `B.grad`) on an overdetermined full-rank 4x2 system by `fn lstsq_public_forward_is_grad_aware_and_matches_fd`. Non-test production consumer: the grad-aware forward `pub fn lstsq` in `ferrotorch-core/src/linalg.rs` (the `torch.linalg.lstsq` public surface) delegates to `lstsq_differentiable` when `is_grad_enabled() && (a.requires_grad() || b.requires_grad())` (forward under `no_grad`). Rank-deficient inputs remain a residual follow-up (pinv is full-rank). |
| REQ-23 (linalg.norm) | SHIPPED | impl: `pub struct NormBackward` + `pub fn matrix_norm_differentiable` + `pub fn vector_norm_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs` implement the Frobenius/Euclidean (`p=2`) VJP `dx = grad * (x / norm)` with the `masked_fill_(norm == 0, 0)` zero-norm guard, mirroring `norm_backward`'s `p==2` branch at `torch/csrc/autograd/FunctionsManual.cpp:341` (via `linalg_vector_norm_backward` at `:523`). Frobenius matrix norm = 2-norm of flattened entries, so the same node serves both. FD-verified by `fn matrix_norm_public_forward_is_grad_aware_and_matches_fd` (2x3 matrix) and `fn vector_norm_public_forward_is_grad_aware_and_matches_fd` (length-4 vector, p=2), both vs central FD at tol 1e-5. Non-test production consumer: the grad-aware forwards `pub fn matrix_norm` (CPU) and `pub fn vector_norm` (`ord==2.0` branch) in `ferrotorch-core/src/linalg.rs` delegate to the matching `*_differentiable` wrapper when grad is enabled (forward under `no_grad`). Non-`p=2` `ord` values and the inf/0/p<1 branches of `norm_backward` are a residual follow-up. |
| REQ-24 (linalg.matrix_rank) | SHIPPED (non-differentiable, mirrors torch) | `matrix_rank` returns an integer-valued tensor and is NON-DIFFERENTIABLE in PyTorch (no `name: linalg_matrix_rank` entry in `tools/autograd/derivatives.yaml` → autograd returns no gradient). ferrotorch matches: `pub fn matrix_rank` in `ferrotorch-core/src/linalg.rs` attaches no `GradFn`, so a downstream `.backward()` through it contributes nothing — byte-for-byte the upstream contract (R-DEV-1: match the autograd `grad_fn` semantics, here "no grad_fn"). The grad-aware forwards of the other ops in this file ARE the production consumers exercising the autograd engine that correctly treats `matrix_rank` as a non-differentiable leaf-stop. No code change needed; the existing forward already mirrors torch. |
| REQ-25 (linalg.cross) | SHIPPED | impl: `pub struct CrossBackward` + `pub fn cross_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs` implement the bilinear VJP `da = cross(b, grad, dim)`, `db = cross(grad, a, dim)` (real case of `at::linalg_cross(other.conj(), grad)` / `at::linalg_cross(grad, self.conj())`), mirroring `linalg_cross` at `tools/autograd/derivatives.yaml:516-518`. The VJP reuses the `cross` forward itself (cross is bilinear). FD-verified (both `dA` and `dB`) by `fn cross_public_forward_is_grad_aware_and_matches_fd` (length-3 vectors, tol 1e-5). Non-test production consumer: the grad-aware forward `pub fn cross` in `ferrotorch-core/src/linalg.rs` (the `torch.linalg.cross` public surface) delegates to `cross_differentiable` when `is_grad_enabled() && (a.requires_grad() || b.requires_grad())` (forward under `no_grad`). |
| REQ-26 (linalg.householder_product) | NOT-STARTED | open prereq blocker #1345 — needs the `householder_product_backward` reflector-recursion VJP (`torch/csrc/autograd/FunctionsManual.cpp`), which accumulates gradients through the sequential Householder-reflector composition (a non-trivial recurrence distinct from the closed-form decomposition VJPs this commit shipped). Deferred as a separate residual; forward-only `pub fn householder_product` in `ferrotorch-core/src/linalg.rs`. |
| REQ-27 (linalg.lu) | SHIPPED (square m==n) | impl: split `struct LuBackwardL` + `struct LuBackwardU` (sharing `LuBackwardShared`) + `pub fn lu_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs` implement the `m == n` VJP `M = tril(L^T gL, -1) + triu(gU U^T)`, then `M <- M @ U^{-T}` (right tri-solve), `M <- L^{-T} M` (left unit tri-solve), `gA = P^T @ M`, mirroring `linalg_lu_backward`'s `m == n` branch at `torch/csrc/autograd/FunctionsManual.cpp:6854`. The `P^T` adjoint (vs torch's `P.matmul`) reflects ferray's `A = P L U` permutation convention (FD-verified). The jointly-linear `(gL, gU)` VJP is split across the two single-output nodes; `P` is non-differentiable (returned plain). FD-verified on a PIVOTED 3x3 (`fn lu_public_forward_is_grad_aware_and_matches_fd`, weighting L and U separately, tol 1e-3). Non-test production consumer: the grad-aware forward `pub fn lu` in `ferrotorch-core/src/linalg.rs` delegates to `lu_differentiable` when `is_grad_enabled() && a.requires_grad()` (forward under `no_grad`). Rectangular `m != n` (wide/tall block-solve branch) is rejected at grad-time and remains a residual follow-up under #1577. |
| REQ-28 (linalg.lu_factor) | SHIPPED (square m==n) | impl: `pub struct LuFactorBackward` + `pub fn lu_factor_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs` decompose the incoming packed-`LU` grad into `gL = tril(grad, -1)`, `gU = triu(grad)` and feed them jointly to `LuBackwardShared::grad_a_combined` (the same `m == n` formula), mirroring `lu_factor_ex_backward` at `torch/csrc/autograd/FunctionsManual.cpp:6960` (which unpacks `LU`/`pivs` and calls `linalg_lu_backward`). The pivot `Vec<i32>` is non-differentiable (returned plain). FD-verified on a pivoted 3x3 by `fn lu_factor_public_forward_is_grad_aware_and_matches_fd` (tol 1e-3). Non-test production consumer: the grad-aware forward `pub fn lu_factor` in `ferrotorch-core/src/linalg.rs` delegates to `lu_factor_differentiable` when `!a.is_cuda() && is_grad_enabled() && a.requires_grad()` (forward under `no_grad`; CUDA stays forward-only). Rectangular case is a residual follow-up under #1577. |
| REQ-29 (trace) | SHIPPED | impl: `pub struct TraceBackward` + `pub fn trace_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs` (VJP `dA = grad * I`, mirroring `trace_backward_symint` at `tools/autograd/derivatives.yaml:1785`). FD-verified by `fn trace_forward_is_grad_aware_and_matches_fd` in the in-file `#[cfg(test)] mod tests` (drives the public forward, checks `A.grad` vs central FD). Non-test production consumer: the grad-aware forward `pub fn trace` in `ferrotorch-core/src/linalg.rs` delegates to `trace_differentiable` when `is_grad_enabled() && a.requires_grad()`; the wrapper computes the forward under `no_grad` to avoid re-entry. |
| REQ-30 (diagonal) | SHIPPED | impl: `pub struct DiagonalBackward` + `pub fn diagonal_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs` (VJP scatters grad onto the offset-th diagonal of a zero matrix, per `diagonal_backward_symint` at `tools/autograd/derivatives.yaml:573`). FD-verified by `fn diagonal_public_forward_is_grad_aware_and_matches_fd`. Non-test production consumer: the now-grad-aware forward `pub fn diagonal` in `ferrotorch-core/src/linalg.rs` (the `torch.linalg.diagonal` public surface) delegates to `diagonal_differentiable` when `is_grad_enabled() && a.requires_grad()`; the wrapper computes the forward under `no_grad` (preventing re-entry). Upstream `Tensor linalg_diagonal(...)` at `aten/src/ATen/native/LinearAlgebra.cpp:2215`. |
| REQ-31 (diag) | SHIPPED | impl: `pub struct DiagBackward` + `pub fn diag_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs` (adjoint of the 0/1 selection — gather for the 1-D-construct forward, scatter for the 2-D-extract forward). FD-verified by `fn diag_extract_public_forward_is_grad_aware_and_matches_fd` + `fn diag_construct_public_forward_is_grad_aware_and_matches_fd`. Non-test production consumer: the now-grad-aware forward `pub fn diag` in `ferrotorch-core/src/ops/tensor_ops.rs` (the `torch.diag` public surface) delegates to `diag_differentiable` when grad is enabled; the wrapper computes the forward under `no_grad`. |
| REQ-32 (tril) | SHIPPED | impl: `pub struct TriangularBackward` + `pub fn tril_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs` (VJP = grad masked by the kept lower triangle, per `tril -> grad.tril_symint` at `tools/autograd/derivatives.yaml:1805`). FD-verified by `fn tril_public_forward_is_grad_aware_and_matches_fd`. Non-test production consumer: the now-grad-aware forward `pub fn tril` in `ferrotorch-core/src/ops/tensor_ops.rs` (the `torch.tril` public surface) delegates to `tril_differentiable` when grad is enabled; the wrapper computes the forward under `no_grad`. |
| REQ-33 (triu) | SHIPPED | impl: `pub fn triu_differentiable` (sharing `pub struct TriangularBackward`) in `ferrotorch-core/src/grad_fns/linalg.rs` (VJP = grad masked by the kept upper triangle, per `triu -> grad.triu_symint` at `tools/autograd/derivatives.yaml:1809`). FD-verified by `fn triu_public_forward_is_grad_aware_and_matches_fd`. Non-test production consumer: the now-grad-aware forward `pub fn triu` in `ferrotorch-core/src/ops/tensor_ops.rs` (the `torch.triu` public surface) delegates to `triu_differentiable` when grad is enabled; the wrapper computes the forward under `no_grad`. |
| REQ-34 (kron) | SHIPPED | impl: `pub struct KronBackward` + `pub fn kron_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs` (per-Kron-block VJP, mirroring `Tensor kron(...)` at `aten/src/ATen/native/LinearAlgebra.cpp:3530`). FD-verified by `fn kron_public_forward_is_grad_aware_and_matches_fd`. Non-test production consumer: the new grad-aware forward `pub fn kron` in `ferrotorch-core/src/linalg.rs` (the `torch.kron` public surface) delegates to `kron_differentiable`; the wrapper attaches the `GradFn` when grad is enabled (forward computed inline, no re-entry guard needed). |
| REQ-35 (outer) | SHIPPED | impl: `pub struct OuterBackward` + `pub fn outer_differentiable` in `ferrotorch-core/src/grad_fns/linalg.rs` (VJP `da = grad @ b`, `db = grad^T @ a`, the unscaled `addr` vec1/vec2 case at `tools/autograd/derivatives.yaml:275-276`). FD-verified by `fn outer_forward_is_grad_aware_and_matches_fd` in the in-file `#[cfg(test)] mod tests` (drives the public forward, loss = sum(C), checks both `a.grad` and `b.grad` vs central FD). Non-test production consumer: the grad-aware forward `pub fn outer` in `ferrotorch-core/src/linalg.rs` delegates to `outer_differentiable` when `is_grad_enabled() && (a.requires_grad() || b.requires_grad())`; the wrapper computes the forward under `no_grad` to avoid re-entry. |
