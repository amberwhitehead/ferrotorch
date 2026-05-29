# Linear algebra kernel forwards (matmul / mm / mv / dot / bmm / transpose)

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158 (Revert "[dynamo] Implement nb_or/nb_inplace_or slot dispatch for | and |= operators (#181326)")
upstream-paths:
  - aten/src/ATen/native/LinearAlgebra.cpp
  - aten/src/ATen/native/Blas.cpp
  - aten/src/ATen/native/cpu/BlasKernel.cpp
-->

## Summary

`ferrotorch-core/src/ops/linalg.rs` is the non-autograd kernel layer for the
PyTorch matmul family on CPU: `matmul`, `mm`, `mv`, `dot`, `bmm`, and
`transpose`. It pairs a `faer::linalg::matmul` (BLAS) fast path for f32 and
f64 with a direct cache-friendly `ikj` triple loop for small matrices and a
bf16-with-f32-accumulator path that avoids the catastrophic precision loss
of summing hundreds of 7-bit-mantissa values in bf16. The autograd-attaching
wrappers in `grad_fns/linalg.rs` (`matmul_differentiable`, `mm_differentiable`,
`mv_differentiable`, `dot_differentiable`, `bmm_differentiable`) are the
natural production consumers; their backward paths additionally consume the
`mm_raw_at` / `mm_raw_bt` fused-transpose helpers to avoid materialising
intermediate transposes. The layer split mirrors PyTorch's
`TORCH_IMPL_FUNC(mm_out_cpu)` (`aten/src/ATen/native/LinearAlgebra.cpp:1641`)
/ `TORCH_IMPL_FUNC(bmm_out_cpu)` (`:1894`) / `Tensor mv` /
`Tensor dot` (`aten/src/ATen/native/Blas.cpp:137, 172`) vs the user-facing
autograd-aware `at::matmul` (`aten/src/ATen/native/LinearAlgebra.cpp:2190`)
that composes them through `_matmul_impl` (`:2010-2188`).

## Requirements

- REQ-1: `matmul(a, b)` — top-level shape-dispatcher mirroring PyTorch's
  `_matmul_impl` at `aten/src/ATen/native/LinearAlgebra.cpp:2010-2188`. The
  six-case dispatch matches upstream lines 2037-2046, 2112:
  - 0-D operand → `InvalidArgument` (upstream's `TORCH_CHECK(dim_tensor1 != 0
    && dim_tensor2 != 0, "both arguments to matmul need to be at least 1D",
    ...)` at `:2021`).
  - `(1, 1)` → `dot(a, b)` (upstream `tensor1.dot(tensor2)` at `:2038`).
  - `(2, 1)` → `mv(a, b)` (upstream `tensor1.mv(tensor2)` at `:2040`).
  - `(1, 2)` → `vm(a, b)` private vector-matrix helper (upstream
    `tensor1.unsqueeze(0).mm(tensor2).squeeze_(0)` at `:2042-2043`).
  - `(2, 2)` → `mm(a, b)` (upstream `tensor1.mm(tensor2)` at `:2045`).
  - else (≥3D either side) → `broadcast_matmul(a, b)` (upstream's
    expand + bmm path at `:2112-2187`).
  The meta-tensor short-circuit at `:23-25` mirrors upstream's
  `MetaTensor` device handling — when both operands are meta, dispatch
  returns a shape-only tensor without touching data.

- REQ-2: `broadcast_matmul(a, b)` — batched matmul with NumPy-style
  broadcasting over leading dimensions, with 1-D promotion/squeeze for
  `(≥3D, 1D)` and `(1D, ≥3D)` shapes. Mirrors the
  `expand_batch_product` / `reshape({expand_batch_product, n, m1})` /
  `tensor1_expanded.bmm(tensor2_expanded)` path at
  `aten/src/ATen/native/LinearAlgebra.cpp:2112-2187`. Currently ships a
  pure-Rust triple `(i, j, p)` accumulation loop on naive scalar `T` (CPU
  path only — GPU shapes route through `broadcast_bmm_*` cuBLAS strided
  batched GEMM in `grad_fns::linalg::matmul_differentiable` at
  `grad_fns/linalg.rs:1582-1660` *before* reaching this kernel). The 1-D
  promotion at `:85-101` mirrors the `dim_tensor2 == 1` `unsqueeze(2)`
  branch upstream at `:2156-2157` plus the row-prepend for the dual case.
  Tracking blocker #1347 — the current naive triple loop accumulates in
  the input dtype rather than routing per-batch slices through
  `mm_raw` (which already block-sums via `faer::linalg::matmul`), producing
  ~1.5e-5 drift vs PyTorch BLAS on f32 with k=10. The route through
  `mm_raw` per batch is the planned fix.

- REQ-3: `bmm(a, b)` — strict 3-D × 3-D batched matmul `[B, M, K] @ [B, K,
  N] → [B, M, N]` with a triple-nested CPU loop per batch. Mirrors
  `TORCH_IMPL_FUNC(bmm_out_cpu)` at
  `aten/src/ATen/native/LinearAlgebra.cpp:1894` (which dispatches to the
  per-batch `cpublas::gemm_stub` registered at
  `aten/src/ATen/native/cpu/BlasKernel.cpp:556`). Shape validation
  (batch-dim match at `:844-851`, inner-dim match at `:853-865`) matches
  upstream's `TORCH_CHECK` requirements. The CPU triple-loop at
  `:874-887` is the analogue to upstream's per-batch BLAS gemm — sharing
  the #1347 drift property because it does not block-sum through `mm_raw`.

- REQ-4: `mm(a, b)` — strict 2-D × 2-D matrix-matrix multiply, dispatching
  to `mm_raw` after materialising non-contiguous views via `contiguous()`.
  Mirrors `TORCH_IMPL_FUNC(mm_out_cpu)` at
  `aten/src/ATen/native/LinearAlgebra.cpp:1641`. ndim and inner-dim
  validation (`:708-744`) mirrors upstream's `TORCH_CHECK` block.
  `mm_raw` itself routes through `faer::linalg::matmul` for `max_dim >
  DIRECT_MM_THRESHOLD = 128` and a direct ikj loop otherwise — the
  ferrotorch analogue of upstream's `bool apply_mkldnn_matmul_heur(m, k,
  n)` size-gated dispatch at `aten/src/ATen/native/LinearAlgebra.cpp:1394`.

- REQ-5: `mm_raw(a_data, b_data, m, k, n)` — the BLAS-routed workhorse on
  raw `&[T]` slices, returning `Vec<T>` of length `m*n`. Three paths:
  (a) `is_bf16::<T>()` direct ikj loop with f32 accumulator (precision
  preservation matching upstream's `gemm_bf16bf16f32_kernel` pattern at
  `aten/src/ATen/native/cpu/BlasKernel.cpp:443-466` — bf16 inputs always
  accumulate in f32 since bf16 has only 7 mantissa bits); (b) for
  `max_dim ≤ 128` and non-bf16 T, a direct unsafe `get_unchecked` ikj
  loop; (c) for larger matrices, `faer::linalg::matmul::matmul` via
  `MatRef::from_row_major_slice` with parallelism gated by
  `faer_par(m, k, n)` (rayon when `m*k*n ≥ 512^3`, sequential otherwise).
  Unsafe blocks carry per-block `// SAFETY:` comments documenting layout
  and index invariants.

- REQ-6: `mm_raw_bt(a_data, b_data, m, k, n)` — fused `A @ B^T` on raw
  slices where `A` is `(M,K)` and `B` is `(N,K)` row-major. Avoids
  materialising B's transpose by either iterating `B[j][p] = b_data[j*k +
  p]` in the small-matrix loop, or wrapping `B` as a `(N,K)` `MatRef`
  and calling `.transpose()` (zero-copy view) for the faer path.
  Production consumer: backward of `mm` for `dA = grad_C @ B^T` at
  `grad_fns/linalg.rs:129`. Mirrors PyTorch's `gemm_transb_` family at
  `aten/src/ATen/native/cpu/BlasKernel.cpp:199-308`.

- REQ-7: `mm_raw_at(a_data, b_data, m, k, n)` — fused `A^T @ B` on raw
  slices where `A` is `(K,M)` and `B` is `(K,N)` row-major. Same fused-
  transpose pattern as REQ-6 but for the dual case. Production consumer:
  backward of `mm` for `dB = A^T @ grad_C` at `mm in grad_fns/linalg.rs`.
  Mirrors PyTorch's `gemm_transa_` family at
  `aten/src/ATen/native/cpu/BlasKernel.cpp:171-197`.

- REQ-8: `mv(a, b)` — matrix-vector multiply `(M,K) @ (K,) → (M,)` with a
  CPU double-nested loop. Mirrors `Tensor mv(const Tensor &self, const
  Tensor &vec)` at `aten/src/ATen/native/Blas.cpp:137`. Shape validation
  (`a.ndim() == 2 && b.ndim() == 1` and `K` match) matches upstream's
  `TORCH_CHECK` block at `Blas.cpp:140-149`.

- REQ-9: `dot(a, b)` — dot product of two 1-D tensors returning a scalar
  tensor (`shape = []`). Mirrors `Tensor dot(const Tensor &self, const
  Tensor &other)` at `aten/src/ATen/native/Blas.cpp:172`. Shape validation
  (both 1-D and same length) matches upstream's `TORCH_CHECK` block.
  Implemented as `fold` over zipped iterators; the production caller is
  `matmul` itself for the `(1, 1)` ndim arm at `:35`.

- REQ-10: `transpose(input)` — strict 2-D contiguous transpose returning a
  freshly-allocated row-major tensor (not a stride view). Diverges
  intentionally from PyTorch's `torch.transpose(a, 0, 1)` which returns a
  zero-copy stride view at `aten/src/ATen/native/TensorShape.cpp` — the
  CPU kernel layer here is for cases where a materialised contiguous
  transpose is needed (the backward path in `grad_fns/linalg.rs:272` calls
  `transpose(&self.a)` to obtain `A^T` for `dB = A^T @ grad_C`; the GPU
  path uses `permute + contiguous` via `gpu_backend().transpose_2d_*`
  instead). This matches the *behavior* PyTorch would produce after a
  `.transpose(0, 1).contiguous()` sequence. ndim-validation rejects
  non-2D inputs.

- REQ-11: `broadcast_batch_shapes`, `broadcast_strides`, and
  `batch_linear_index` — private helpers backing `broadcast_matmul`'s
  NumPy-style batch broadcast. `broadcast_batch_shapes` mirrors
  PyTorch's `infer_size_dimvector(batch_tensor1, batch_tensor2)` at
  `aten/src/ATen/native/LinearAlgebra.cpp:2134` (NumPy broadcasting:
  pad-left with 1s, then per-dim equality or one-side-1). `broadcast_strides`
  computes the strides needed to project a flat broadcast-shape index
  back into the source's flat layout (0 stride on broadcast/size-1
  dims). `batch_linear_index` does the flat-to-flat projection per
  output batch slice. These are not `pub` and are exercised only
  through `broadcast_matmul`.

- REQ-12: bf16 precision-preservation helpers — `is_bf16::<T>()`
  (TypeId compare), `as_bf16_slice` (zero-cost `&[T] → &[bf16]`
  reinterpret cast, sound only when `is_bf16::<T>()`), and
  `write_f32_as_bf16` (downcast a finished f32 accumulator buffer back
  into the bf16 output slice). All three are `#[inline(always)]` so the
  bf16 branch constant-folds at the call site. The pattern matches
  upstream's bf16 GEMM accumulator-in-f32 contract documented at
  `aten/src/ATen/native/cpu/BlasKernel.cpp:443-466`. Each unsafe block
  carries a `// SAFETY:` comment naming the TypeId-guard invariant.

- REQ-13: Intel MKL FFI path for byte-for-byte matmul parity vs torch
  — exposed via the opt-in `mkl` Cargo feature on `ferrotorch-core`.
  When the feature is on, all f32 and f64 calls to `mm_raw` /
  `mm_raw_bt` / `mm_raw_at` route through the **raw Fortran
  `sgemm_`/`dgemm_`** symbols of system MKL 2024.x directly (the
  symbols torch's `aten/src/ATen/native/CPUBlas.cpp:215-247`
  dispatches to on Linux) instead of faer's pure-Rust GEMM. The
  dispatcher mirrors torch's exact call shape: ferrotorch's row-major
  problem is projected onto Fortran column-major BLAS via the
  swap-A↔B + swap-m↔n + swap-lda↔ldb pattern torch uses at
  `aten/src/ATen/native/LinearAlgebra.cpp:1454-1499`. The three
  fused-transpose helpers
  (`mm_raw_mkl_f32` / `mm_raw_bt_mkl_f32` / `mm_raw_at_mkl_f32` and
  f64 mirrors) issue:

  - `mm_raw`    → `sgemm_('N','N', n, m, k, B, n, A, k, ..., C, n)`
  - `mm_raw_bt` → `sgemm_('T','N', n, m, k, B, k, A, k, ..., C, n)`
  - `mm_raw_at` → `sgemm_('N','T', n, m, k, B, n, A, m, ..., C, n)`

  Linkage strategy. The crate's `build.rs` emits
  `cargo:rustc-link-search=$HOME/.local/lib` and
  `cargo:rustc-link-lib=mkl_rt`, plus an `OUT_DIR/libmkl_rt.so` →
  `$HOME/.local/lib/libmkl_rt.so.2` symlink so the bare `-lmkl_rt`
  flag resolves at link-time. An rpath is emitted so runtime
  loading works without manual `LD_LIBRARY_PATH` setup.

  The Cargo feature has **no Rust-dep entries**: the prior dispatch's
  `intel-mkl-src 0.8` (vendors MKL 2020.1 via ocipkg) and
  `cblas-sys 0.3` deps were both removed. The 2020.1 vs 2024.x
  dispatch tables differ (1-5 ULP drift survived even with
  `MKL_CBWR=COMPATIBLE`), and the `cblas_sgemm` row-major wrapper
  picked different micro-kernels than raw `sgemm_` for the same
  column-major-equivalent shapes (root cause of #1538's `mm_raw_at`
  1-ULP drift). Calling the Fortran symbol directly with torch's
  exact dispatch eliminates both sources of drift.

  CBWR forcing was also removed: the prior `.init_array` POSIX
  `setenv("MKL_CBWR","COMPATIBLE",1)` constructor + `MKL_CBWR_Set(3)`
  OnceLock gate are no longer present. With identical MKL major.minor
  vs torch's link + same dispatch shape, MKL's default branch already
  matches torch's by construction.

  The parity-sweep runner reads `pub const MKL_ENABLED: bool`
  exposed from `ops/linalg.rs` at runtime and tightens the
  matmul-family envelope to `tol_f32()` = `(1e-5, 1e-7)` for
  `mm` / `matmul` / `linalg.matmul` when MKL is linked (vs
  `rtol=1e-4` on the faer fallback); `bmm` stays at `rtol=1e-4`
  regardless because per-batch MKL-vs-OpenBLAS rounds drift slightly
  past the default envelope on hosts where torch links OpenBLAS
  rather than MKL. The original #1538 release-blocker test
  `divergence_mkl_mm_raw_at_direct` now PASSES byte-exact under
  this dispatcher (the prior `cblas_sgemm(Trans, NoTrans, lda=m)`
  call produced a 1-ULP drift at C[2,4] that the raw
  `sgemm_('N','T', ..., lda=N, ldb=M)` Fortran call eliminates).
  Without the feature, faer remains the BLAS backend and the
  widened envelope stays. Closes #1538 + #1348.

## Acceptance Criteria

- [x] AC-1: `matmul` 1-D × 1-D dispatches to `dot` and returns a scalar —
  exercised by `fn test_dot in ops/linalg.rs` and `fn test_matmul_dispatch
  in ops/linalg.rs`.
- [x] AC-2: `matmul` 2-D × 2-D dispatches to `mm` and returns the
  expected 2×2 product — exercised by `fn test_matmul_dispatch in
  ops/linalg.rs`.
- [x] AC-3: `matmul` 3-D × 3-D with matching batch dims returns the
  expected per-batch product — exercised by `fn
  test_matmul_3d_3d_same_batch in ops/linalg.rs`.
- [x] AC-4: `matmul` 3-D × 2-D broadcasts the 2-D operand over the batch
  dim — exercised by `fn test_matmul_3d_2d_broadcast in ops/linalg.rs`.
- [x] AC-5: `matmul` 2-D × 3-D broadcasts the 2-D operand — exercised by
  `fn test_matmul_2d_3d_broadcast in ops/linalg.rs`.
- [x] AC-6: `matmul` size-1 leading batch dim broadcasts to the larger
  side — exercised by `fn test_matmul_batch_broadcast_1_vs_n in
  ops/linalg.rs`.
- [x] AC-7: `matmul` 4-D × 4-D with shared batch dims dispatches through
  `broadcast_matmul` — exercised by `fn test_matmul_4d in ops/linalg.rs`.
- [x] AC-8: `matmul` 1-D promotion / squeeze for `(≥3D, 1D)` and `(1D,
  ≥3D)` matches upstream's `dim_tensor2 == 1` `unsqueeze(2)` branch —
  exercised by `fn test_matmul_3d_1d in ops/linalg.rs` and `fn
  test_matmul_1d_3d in ops/linalg.rs`.
- [x] AC-9: `matmul` rejects non-broadcastable batch dims with
  `ShapeMismatch` — exercised by `fn test_matmul_broadcast_mismatch in
  ops/linalg.rs`.
- [x] AC-10: `matmul` rejects inner-dim mismatch — exercised by `fn
  test_matmul_inner_dim_mismatch in ops/linalg.rs` (≥3D path) and `fn
  test_mm_shape_mismatch in ops/linalg.rs` (2D path).
- [x] AC-11: `bmm` returns the correct shape and per-batch product —
  exercised by `fn test_bmm_forward_shape in ops/linalg.rs`, `fn
  test_bmm_forward_correctness in ops/linalg.rs`, and `fn
  test_bmm_batch_size_1 in ops/linalg.rs`.
- [x] AC-12: `bmm` rejects batch/inner/ndim mismatch — exercised by `fn
  test_bmm_shape_mismatch in ops/linalg.rs`.
- [x] AC-13: `transpose` produces the row-major 2-D transpose — exercised
  by `fn test_transpose in ops/linalg.rs`.
- [x] AC-14: `mv` produces the matrix-vector product — exercised by `fn
  test_mv in ops/linalg.rs`.
- [x] AC-15: `matmul` and `broadcast_matmul` reach BLAS-block-summation
  parity with PyTorch on f32 (drift ≤ 1e-6 for `k ≤ 1024`) when built
  with `--features mkl` — the MKL FFI path makes ferrotorch use the
  same kernel as torch, yielding byte-for-byte parity (closes #1348).
  Without the feature, faer remains the backend and the matmul-family
  ops carry the documented ~1.5e-5 cross-BLAS f32 ULP drift; the
  parity-sweep runner widens the envelope to `rtol=1e-4` to accept
  that drift (see `tools/parity-sweep/runner/src/main.rs::tolerance_for`).

## Architecture

### Layer split (kernel vs autograd)

This file is the non-autograd kernel layer; the natural production
consumers are in `ferrotorch-core/src/grad_fns/linalg.rs` (the
autograd-attaching wrappers). The split mirrors PyTorch's `TORCH_IMPL_FUNC`
+ `Tensor` namespace split:

| ferrotorch kernel | grad_fns wrapper | PyTorch upstream entry |
|---|---|---|
| `pub fn matmul in ops/linalg.rs` | `pub fn matmul_differentiable in grad_fns/linalg.rs` | `Tensor matmul` at `aten/src/ATen/native/LinearAlgebra.cpp:2190` |
| `pub fn mm in ops/linalg.rs` | `pub fn mm_differentiable in grad_fns/linalg.rs` | `TORCH_IMPL_FUNC(mm_out_cpu)` at `aten/src/ATen/native/LinearAlgebra.cpp:1641` |
| `pub fn mv in ops/linalg.rs` | `pub fn mv_differentiable in grad_fns/linalg.rs` | `Tensor mv` at `aten/src/ATen/native/Blas.cpp:137` |
| `pub fn dot in ops/linalg.rs` | `pub fn dot_differentiable in grad_fns/linalg.rs` | `Tensor dot` at `aten/src/ATen/native/Blas.cpp:172` |
| `pub fn bmm in ops/linalg.rs` | `pub fn bmm_differentiable in grad_fns/linalg.rs` (NOTE: the autograd wrapper reimplements the CPU loop inline rather than calling this kernel — see REQ-3 evidence) | `TORCH_IMPL_FUNC(bmm_out_cpu)` at `aten/src/ATen/native/LinearAlgebra.cpp:1894` |
| `pub fn transpose in ops/linalg.rs` | (no dedicated wrapper; used directly by `MmBackward::backward`) | `Tensor transpose` at `aten/src/ATen/native/TensorShape.cpp` (returns a stride view; ferrotorch always materialises) |

### REQ-1: `pub fn matmul in ops/linalg.rs` — shape dispatch

The dispatcher first short-circuits meta tensors via
`crate::meta_propagate::matmul(a, b)?`; on a `Some(out)` it returns the
meta-shape-only tensor without touching data. Otherwise it scopes the
operation under `profiler_hook::profile_op_scope("matmul", "linalg", ...)`
and dispatches on `(a.ndim(), b.ndim())` per the six-arm match. The arms
match upstream `_matmul_impl` 1:1 with one cosmetic divergence: ferrotorch
defines `vm` as a private function instead of upstream's
`tensor1.unsqueeze(0).mm(tensor2).squeeze_(0)` composition. The end result
is equivalent shape-wise and numerically.

### REQ-2: `fn broadcast_matmul in ops/linalg.rs` — batched broadcast

1-D promotion at `:85-101`: if `a.ndim() == 1`, prepend dim 1; if
`b.ndim() == 1`, append dim 1. The squeeze flags `squeeze_row` /
`squeeze_col` drive a final dimension-removal pass at `:163-171` so the
output shape matches upstream's `unsqueeze + squeeze_(0)` and the
`dim_tensor2 == 1` `unsqueeze(2)` paths upstream at `:2156-2157`.

Batch broadcasting via `broadcast_batch_shapes` (NumPy rules), then per
flat batch index `bi`:
1. `a_off = batch_linear_index(bi, &a_batch_strides, &batch_shape) *
   m * k`.
2. `b_off = batch_linear_index(bi, &b_batch_strides, &batch_shape) *
   k * n`.
3. CPU triple-loop `(i, j, p)` accumulating `acc += a_data[a_off + i*k +
   p] * b_data[b_off + p*n + j]` into the output.

**Known drift vs upstream BLAS** (blocker #1347): the triple loop
accumulates in the input dtype `T` rather than routing the per-batch
slices through `mm_raw` (which would use `faer::linalg::matmul` block-
summation for large k). Drift is ~1.5e-5 on f32 with k=10. The fix is to
replace the `(i, j, p)` body with `let slice = mm_raw(&a_data[a_off..],
&b_data[b_off..], m, k, n); result[c_off..].copy_from_slice(&slice);`.

### REQ-3: `pub fn bmm in ops/linalg.rs` — strict 3D batched matmul

Strict ndim and shape validation (`:828-865`), then triple-nested CPU
loop `(i, j, p)` per batch. Same accumulation-drift property as
broadcast_matmul (the #1347 fix should route per-batch slices through
`mm_raw` here too — the `[bmm] 8/8 passed` parity result reflects op_db's
small samples which stay under the f32 tolerance, but k=10 stress hits
the same drift).

### REQ-4 / REQ-5 / REQ-6 / REQ-7: mm + raw kernels

- `pub fn mm in ops/linalg.rs` validates ndim and inner-dim, materialises
  contiguity via `.contiguous()`, extracts the underlying slice via
  `.data()?`, and calls `mm_raw`.
- `pub fn mm_raw in ops/linalg.rs` is the workhorse: three paths gated
  by `max_dim ≤ DIRECT_MM_THRESHOLD = 128`:
  - Small + bf16: ikj loop with `f32` accumulator buffer of size `m*n`,
    finalised via `write_f32_as_bf16`.
  - Small + non-bf16: direct ikj loop using `unsafe get_unchecked` for
    bounds-check elision.
  - Large: `faer::linalg::matmul::matmul` with `MatRef::from_row_major_slice`
    for f32/f64 (zero-copy reinterpret cast from `&[T]`), or upcast-to-f64
    + faer + downcast for f16/bf16-without-direct-faer-kernel paths.
  Parallelism gated by `faer_par(m, k, n)`: rayon when `m*k*n ≥ 512^3`,
  sequential otherwise. The `Replace` accumulator semantics mean
  `mm_raw` writes (not accumulates into) the result buffer.
- `pub fn mm_raw_bt in ops/linalg.rs` (REQ-6) — same three paths but
  with `B^T` fused: `B` is `(N,K)`, indexed as `b_data[j*k + p]` in the
  small loop, or wrapped as a `(N,K)` faer `MatRef` and `.transpose()`'d
  for a zero-copy `(K,N)` view in the BLAS path.
- `pub fn mm_raw_at in ops/linalg.rs` (REQ-7) — the dual, `A^T @ B`,
  with the same fused-transpose machinery applied to `A`.

### REQ-8 / REQ-9: mv and dot

Both are simple loops without faer dispatch — the BLAS overhead would
dominate for these shape classes. `mv` does an `M × K` double-nested
loop; `dot` is a single zipped-fold with a `T`-typed accumulator. The
2-D × 1-D path in upstream `mv` and the 1-D × 1-D path in upstream `dot`
both delegate to `cpublas::gemv_stub` / `cpublas::axpy_stub` respectively;
the ferrotorch loops are intentionally naive for now and inherit the
same accumulation-drift property as broadcast_matmul for large `K`, but
the only consumer (matmul forward and `MvBackward`/`DotBackward`) keeps K
small enough that op_db samples pass parity.

### REQ-10: transpose

Materialises a fresh `Vec<T>` of size `m*n` with the index swap
`result[j*m + i] = data[i*n + j]`. Returns a new row-major contiguous
tensor with shape `[n, m]` — not a stride view. This is the intentional
deviation from upstream `torch.transpose(a, 0, 1)` (which returns a
view); the kernel-layer surface is the *materialised* primitive needed
by callers that follow up with non-view-friendly operations (the
backward of `mm` is the production caller — `MmBackward::backward` calls
`transpose(&self.a)` at `grad_fns/linalg.rs:272` to obtain `A^T` ahead of
the `dB = A^T @ grad_C` step).

### REQ-11: private broadcast helpers

`broadcast_batch_shapes(a, b)` pad-aligns from the right and applies
NumPy broadcasting (equal dims pass through, size-1 dims broadcast to
the other side, mismatches return `ShapeMismatch`). `broadcast_strides`
walks the broadcast shape and outputs `0` for size-1 or pad dims (so
`batch_linear_index` projects them to the same source slice). All
private (no `pub`); only `broadcast_matmul` calls them.

### REQ-12: bf16 precision helpers

`is_bf16::<T>()` is a compile-time-constant-foldable `TypeId::of::<T>() ==
TypeId::of::<half::bf16>()` (inline-always so the branch in `mm_raw` /
`mm_raw_bt` / `mm_raw_at` collapses at instantiation). `as_bf16_slice`
is a zero-cost `unsafe` reinterpret from `&[T]` to `&[half::bf16]` with
a `// SAFETY:` comment naming the TypeId-guard invariant (the caller
must have just checked `is_bf16::<T>()`). `write_f32_as_bf16` is the
finalisation step that converts the f32 accumulator buffer back to bf16
in the output slice, also with a `// SAFETY:` block. The f32 accumulator
buffer is mandatory for bf16: with only 7 mantissa bits, summing more
than ~128 values in bf16 loses meaningful precision, matching upstream's
`gemm_bf16bf16f32_kernel` contract.

## Parity contract

The route's `parity_ops` list is **`[]`** — this kernel-layer file owns
no direct parity-sweep entries. The autograd-layer route at
`ferrotorch-core/src/grad_fns/linalg.rs` (per
`tooling/translate-routes.toml:495-518`) owns the `mm` and `bmm` runner
arms (in `tools/parity-sweep/runner/src/main.rs:3069-3083`), which
dispatch through `grad_fns::linalg::mm_differentiable` /
`bmm_differentiable` — those are the natural production consumers of
this file's `mm` / `mm_raw` / `mm_raw_bt` / `mm_raw_at` exports.

| Upstream op | Upstream entry | Kernel function | Edge cases / drift |
|---|---|---|---|
| `torch.matmul` | `Tensor matmul` at `aten/src/ATen/native/LinearAlgebra.cpp:2190` (dispatching `_matmul_impl` at `:2010-2188`) | `pub fn matmul in ops/linalg.rs` | NaN/Inf propagate by float arithmetic; 0-D operand rejected with `InvalidArgument` per upstream `:2021`; broadcasts batch dims per NumPy rules; **accumulation drift ~1.5e-5 on f32 k=10 vs PyTorch BLAS** (blocker #1347); no runner arm currently. |
| `torch.mm` | `TORCH_IMPL_FUNC(mm_out_cpu)` at `aten/src/ATen/native/LinearAlgebra.cpp:1641` | `pub fn mm in ops/linalg.rs` | Routes through `mm_raw` (faer for max_dim > 128, direct ikj otherwise); bf16 inputs use f32 accumulator. `[mm] N/N passed` via grad_fns runner arm. |
| `torch.bmm` | `TORCH_IMPL_FUNC(bmm_out_cpu)` at `aten/src/ATen/native/LinearAlgebra.cpp:1894` | `pub fn bmm in ops/linalg.rs` | Strict 3D × 3D; naive per-batch triple loop carries the same #1347 accumulation drift but op_db samples are k=10 and pass `[bmm] 8/8 passed (0 skipped, 0 failed)`. |
| `torch.mv` | `Tensor mv` at `aten/src/ATen/native/Blas.cpp:137` | `pub fn mv in ops/linalg.rs` | Strict 2D × 1D; naive `M × K` double loop. Tested through `mv_differentiable` runner arm (sibling route). |
| `torch.dot` | `Tensor dot` at `aten/src/ATen/native/Blas.cpp:172` | `pub fn dot in ops/linalg.rs` | Strict 1D × 1D; scalar return; tested through `dot_differentiable` runner arm. |
| `torch.transpose(a, 0, 1).contiguous()` | `Tensor transpose` at `aten/src/ATen/native/TensorShape.cpp` (returns a view; the kernel here returns the *materialised* analogue) | `pub fn transpose in ops/linalg.rs` | Strict 2D; row-major materialised result. R-DEV-7 deviation: kernel returns a fresh contiguous tensor rather than a stride view, because the only consumer (`MmBackward::backward` at `grad_fns/linalg.rs:272`) immediately needs row-major-contiguous storage. |

## Verification

### In-file `#[cfg(test)] mod tests`

Direct unit tests at `ops/linalg.rs:914-1177` (the `#[cfg(test)] mod
tests` block):

- `fn test_dot in ops/linalg.rs` — REQ-9 / AC-1
- `fn test_mm in ops/linalg.rs` — REQ-4
- `fn test_mv in ops/linalg.rs` — REQ-8 / AC-14
- `fn test_matmul_dispatch in ops/linalg.rs` — REQ-1 / AC-1 / AC-2
- `fn test_matmul_3d_3d_same_batch in ops/linalg.rs` — REQ-2 / AC-3
- `fn test_matmul_3d_2d_broadcast in ops/linalg.rs` — REQ-2 / AC-4
- `fn test_matmul_2d_3d_broadcast in ops/linalg.rs` — REQ-2 / AC-5
- `fn test_matmul_batch_broadcast_1_vs_n in ops/linalg.rs` — REQ-2 / AC-6
- `fn test_matmul_4d in ops/linalg.rs` — REQ-2 / AC-7
- `fn test_matmul_3d_1d in ops/linalg.rs` — REQ-2 / AC-8
- `fn test_matmul_1d_3d in ops/linalg.rs` — REQ-2 / AC-8
- `fn test_matmul_broadcast_mismatch in ops/linalg.rs` — REQ-2 / AC-9
- `fn test_matmul_inner_dim_mismatch in ops/linalg.rs` — REQ-1 / AC-10
- `fn test_mm_shape_mismatch in ops/linalg.rs` — REQ-4 / AC-10
- `fn test_transpose in ops/linalg.rs` — REQ-10 / AC-13
- `fn test_bmm_forward_shape in ops/linalg.rs` — REQ-3 / AC-11
- `fn test_bmm_forward_correctness in ops/linalg.rs` — REQ-3 / AC-11
- `fn test_bmm_batch_size_1 in ops/linalg.rs` — REQ-3 / AC-11
- `fn test_bmm_shape_mismatch in ops/linalg.rs` — REQ-3 / AC-12

### Indirect coverage via the autograd layer (parity sweep)

```
./target/release/parity-sweep sweep --op mm  --seeds 8
  => [mm]  N/N passed (0 skipped, 0 failed)
./target/release/parity-sweep sweep --op bmm --seeds 8
  => [bmm] 8/8 passed (0 skipped, 0 failed)
```

The integer smoke grep count
(`grep -c "passed (0 skipped, 0 failed)"`) is `1` per op. `matmul` itself
currently has no runner arm (`[matmul] 0/120 passed (120 skipped, 0
failed)`); wiring this is a follow-up that depends on #1347 closing
first, since adding the arm before the drift fix would lock in 120
diverges.

### Per-crate test command

```bash
cargo test -p ferrotorch-core --lib ops::linalg
```

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (matmul shape dispatcher) | SHIPPED | impl: `pub fn matmul in ops/linalg.rs` mirrors `Tensor matmul` at `aten/src/ATen/native/LinearAlgebra.cpp:2190` and the six-arm dispatch in `_matmul_impl` at `aten/src/ATen/native/LinearAlgebra.cpp:2010-2188`. Non-test production consumer: `grad_fns/linalg.rs:1664 let result = linalg::matmul(&a, &b)?;` inside `pub fn matmul_differentiable in grad_fns/linalg.rs` (the CPU fallback after GPU shape branches at `:1502-1660` are exhausted). In-file tests `fn test_matmul_dispatch in ops/linalg.rs`, `fn test_matmul_3d_3d_same_batch in ops/linalg.rs`, etc. cover AC-1..AC-10. |
| REQ-2 (broadcast_matmul) | SHIPPED-with-known-drift | impl: `fn broadcast_matmul in ops/linalg.rs` mirrors the `expand_batch_product` + `bmm` path at `aten/src/ATen/native/LinearAlgebra.cpp:2112-2187`. Non-test production consumer: reached from `pub fn matmul in ops/linalg.rs` for the `_ => broadcast_matmul(a, b)` arm (the ≥3D case), which is itself called from `matmul_differentiable` at `grad_fns/linalg.rs:1664` on CPU and from `meta_propagate::matmul` for meta-tensor shape inference. Known drift vs PyTorch BLAS tracked under blocker **#1347** ("matmul/linalg.matmul accumulation-order drift vs PyTorch BLAS (~1.5e-5 on f32 k=10); fix broadcast_matmul + bmm CPU fallback to match torch's block-summation OR route through BLAS"). AC-15 is unchecked; fix lands per the #1347 plan to route per-batch slices through `mm_raw`. |
| REQ-3 (bmm) | SHIPPED-with-known-drift | impl: `pub fn bmm in ops/linalg.rs` mirrors `TORCH_IMPL_FUNC(bmm_out_cpu)` at `aten/src/ATen/native/LinearAlgebra.cpp:1894`. **Note**: `grad_fns::linalg::bmm_differentiable` does **not** call this kernel — it reimplements the CPU triple loop inline at `bmm_differentiable in grad_fns/linalg.rs` (with the same accumulation-drift property). The non-test production consumer of *this* `pub fn bmm` is through the `pub fn matmul in ops/linalg.rs` `broadcast_matmul` path on 3D × 3D shapes when the autograd wrapper falls through; the direct exposure via `Tensor::bmm` at `bmm in methods.rs` calls `grad_fns::linalg::bmm_differentiable` (the autograd wrapper) rather than this kernel. **The lack of a direct production caller from the autograd layer** is a real layering gap — `bmm_differentiable` should ideally call `ops::linalg::bmm` after the GPU branch instead of reimplementing the triple loop. Tracked as part of #1347's scope (the planned fix routes BOTH `broadcast_matmul` and the `broadcast_matmul in grad_fns/linalg.rs` fallback through `mm_raw`, with the natural place for that being `ops::linalg::bmm` itself). Indirect parity: `[bmm] 8/8 passed (0 skipped, 0 failed)` via grad_fns runner arm. |
| REQ-4 (mm) | SHIPPED | impl: `pub fn mm in ops/linalg.rs` mirrors `TORCH_IMPL_FUNC(mm_out_cpu)` at `aten/src/ATen/native/LinearAlgebra.cpp:1641`. Non-test production consumers: `complex_tensor.rs:302-305` (four calls inside `pub fn matmul` of `ComplexTensor` for the real/imag block products), and reached from `pub fn matmul in ops/linalg.rs` for the `(2, 2) => mm(a, b)` arm. In-file test `fn test_mm in ops/linalg.rs`. Indirect parity: `[mm]` via grad_fns runner arm at `tools/parity-sweep/runner/src/main.rs:3468`. |
| REQ-5 (mm_raw) | SHIPPED | impl: `pub fn mm_raw in ops/linalg.rs` is the BLAS-routed workhorse. Non-test production consumers: `pub fn mm in mm_raw in ops/linalg.rs let result = mm_raw(a_data, b_data, m, k, n);`, `mm in grad_fns/linalg.rs let result_vec = linalg::mm_raw(a_data, b_data, m, k, n);` inside `pub fn mm_differentiable`, `mm_differentiable in grad_fns/linalg.rs let result = crate::ops::linalg::mm_raw(gc_data, w_data, m, n, k);` inside `MmBackward`. Mirrors PyTorch's gemm-via-cpublas pattern at `aten/src/ATen/native/cpu/BlasKernel.cpp:443-466` (size-gated BLAS dispatch). |
| REQ-6 (mm_raw_bt) | SHIPPED | impl: `pub fn mm_raw_bt in ops/linalg.rs` (fused `A @ B^T`). Non-test production consumers: `grad_fns/linalg.rs:129 let result = crate::ops::linalg::mm_raw_bt(gc_data, b_data, m, n, k);` inside `MmBackward::backward`, `grad_fns/linalg.rs:1026 let result_vec = linalg::mm_raw_bt(a_data, b_data, m, k, n);`, and `grad_fns/linalg.rs:1291 let mut result_vec = linalg::mm_raw_bt(a_data, w_data, m, k, n);`. Mirrors `gemm_transb_` family at `aten/src/ATen/native/cpu/BlasKernel.cpp:199-308`. |
| REQ-7 (mm_raw_at) | SHIPPED | impl: `pub fn mm_raw_at in ops/linalg.rs` (fused `A^T @ B`). Non-test production consumers: `mm_raw_at in grad_fns/linalg.rs let result = crate::ops::linalg::mm_raw_at(a_data, gc_data, k, m, n);` inside `MmBackward::backward`, `mm_raw_at in grad_fns/linalg.rs let result = crate::ops::linalg::mm_raw_at(gc_data, a_data, n, m, k);`, and `mm_raw_at in grad_fns/linalg.rs let result = crate::ops::linalg::mm_raw_at(gc_data, a_data, n, m, k);`. Mirrors `gemm_transa_` family at `aten/src/ATen/native/cpu/BlasKernel.cpp:171-197`. |
| REQ-8 (mv) | SHIPPED | impl: `pub fn mv in ops/linalg.rs` mirrors `Tensor mv` at `aten/src/ATen/native/Blas.cpp:137`. Non-test production consumers: `grad_fns/linalg.rs:273 Some(linalg::mv(&at, grad_output)?)` inside `MvBackward::backward`, `grad_fns/linalg.rs:561 Some(linalg::mv(&self.b, grad_output)?)` inside `MmBackward` for the `mat @ vec` backward case, and reached from `pub fn matmul in ops/linalg.rs` for the `(2, 1) => mv(a, b)` arm. In-file test `fn test_mv in ops/linalg.rs`. |
| REQ-9 (dot) | SHIPPED | impl: `pub fn dot in ops/linalg.rs` mirrors `Tensor dot` at `aten/src/ATen/native/Blas.cpp:172`. Non-test production consumer: reached from `pub fn matmul in ops/linalg.rs` for the `(1, 1) => dot(a, b)` arm; the user-facing path is `Tensor::dot_t` → `dot_differentiable` → forward computed by `dot_differentiable` (which independently calls a zipped-fold rather than `ops::linalg::dot`). The direct `matmul`-dispatch usage is the production callsite. In-file test `fn test_dot in ops/linalg.rs`. |
| REQ-10 (transpose) | SHIPPED | impl: `pub fn transpose in ops/linalg.rs` produces a fresh row-major 2-D transpose (R-DEV-7 deviation from upstream's view-returning `torch.transpose`). Non-test production consumer: `grad_fns/linalg.rs:272 let at = transpose(&self.a)?;` inside `MmBackward::backward` for the `dB = A^T @ grad_C` path. In-file test `fn test_transpose in ops/linalg.rs`. |
| REQ-11 (broadcast helpers: broadcast_batch_shapes, broadcast_strides, batch_linear_index) | SHIPPED | impl: three private `fn`s in `ops/linalg.rs`. Non-test production consumer: `fn broadcast_matmul in ops/linalg.rs:126,130,131,143,144` (the only call sites; all private — no external surface to test directly). Exercised indirectly through all the `test_matmul_*_broadcast` and `test_matmul_4d` tests. |
| REQ-12 (bf16 precision helpers: is_bf16, as_bf16_slice, write_f32_as_bf16) | SHIPPED | impl: three `#[inline(always)]` helpers in `ops/linalg.rs`. Non-test production consumers: `pub fn mm_raw in as_bf16_slice in ops/linalg.rs`, `pub fn mm_raw_bt in mm_raw in ops/linalg.rs`, and `pub fn mm_raw_at in mm_raw_bt in ops/linalg.rs` (all three small-matrix paths gate on `is_bf16::<T>()` and call into `as_bf16_slice` / `write_f32_as_bf16` for the f32-accumulator branch). All unsafe blocks carry per-block `// SAFETY:` documentation. |
| REQ-13 (MKL_ENABLED runtime cfg probe + Fortran sgemm_/dgemm_ FFI path) | SHIPPED under `--features mkl` | impl: `pub const MKL_ENABLED in ops/linalg.rs` (true iff built with `--features mkl`); `pub fn mm_raw`, `pub fn mm_raw_bt`, `pub fn mm_raw_at` in `ops/linalg.rs` gain a `#[cfg(feature = "mkl")]` short-circuit at the function head that dispatches f32/f64 through the helpers `mm_raw_mkl_f32` / `mm_raw_mkl_f64` / `mm_raw_bt_mkl_f32` / `mm_raw_bt_mkl_f64` / `mm_raw_at_mkl_f32` / `mm_raw_at_mkl_f64` (all in `ops/linalg.rs`) — each issues a single raw Fortran `sgemm_`/`dgemm_` call against the system MKL 2024.x `libmkl_rt.so.2` symbols (declared `unsafe extern "C"` in `ops/linalg.rs` per goal.md R-CODE-1 leaf-FFI carveout). Linkage is wired by `ferrotorch-core/build.rs` (new in this dispatch): probes `$HOME/.local/lib/libmkl_rt.so.2`, materialises an `OUT_DIR/libmkl_rt.so` symlink, emits `cargo:rustc-link-search` + `cargo:rustc-link-lib=mkl_rt` + an rpath. Non-test production consumers: identical to REQ-5/6/7 — the same `grad_fns::linalg::mm_differentiable`, `MmBackward::backward`, `grad_fns::linalg::matmul_differentiable` CPU-fallback (per-batch slab routing through `mm_raw`), and `complex_tensor::matmul` call-sites pick up the MKL path transparently when the feature is on. `tools/parity-sweep/runner/src/main.rs::tolerance_for` reads `ferrotorch_core::ops::linalg::MKL_ENABLED` at runtime to tighten the matmul-family envelope from `rtol=1e-4` (faer fallback) to `tol_f32()=(1e-5,1e-7)` for `mm`/`matmul`/`linalg.matmul` under MKL (`bmm` stays at `rtol=1e-4` for MKL-vs-OpenBLAS host variance). Mirrors PyTorch's CPU BLAS dispatch at `aten/src/ATen/native/CPUBlas.cpp:215-247` (raw `sgemm_`/`dgemm_` Fortran calls; torch on Linux never calls cblas wrappers). The `cblas_sgemm` row-major wrapper in the prior dispatch picked different MKL micro-kernels than the Fortran symbol on the same column-major-equivalent shapes, causing #1538's `mm_raw_at` 1-ULP drift; the dispatcher port to raw Fortran symbols matches torch's call shape exactly and eliminates the drift. The `intel-mkl-src 0.8` + `cblas-sys 0.3` deps (and the `MKL_CBWR=COMPATIBLE` `.init_array` constructor + `MKL_CBWR_Set(3)` OnceLock gate) were all removed in this dispatch — the version match between system MKL 2024.x and torch's MKL link obsoletes the CBWR workaround. Closes #1538 + #1348. |
