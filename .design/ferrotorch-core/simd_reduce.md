# SIMD reduction primitives (torch-matching f32 L2 norm)

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - aten/src/ATen/native/SharedReduceOps.h
  - aten/src/ATen/native/cpu/ReduceOpsKernel.cpp
  - aten/src/ATen/native/cpu/Reduce.h
  - aten/src/ATen/cpu/vec/functional_base.h
  - aten/src/ATen/cpu/vec/vec256/vec256_float.h
  - aten/src/ATen/OpMathType.h
-->

## Summary

`ferrotorch-core/src/simd_reduce.rs` ships the f32 L2-norm reduction PyTorch's
CPU kernel actually performs, so ferrotorch's `norm(p=2)` and
`embedding(max_norm=...)` renorm DECISIONS match torch byte-for-byte (modulo a
documented ~3% one-ULP residual). The naive scalar forms ferrotorch used before
(`Σ v.abs().powf(2.0)` then `.powf(0.5)`, and `Σ v*v` then `.sqrt()`) differ
from torch by one ULP on a meaningful fraction of f32 rows, which flips the
`norm > max_norm` boundary decision (#1612 / #1614).

Torch's f32 L2 path is NOT a scalar accumulation. For a contiguous last-dim f32
reduction it runs the vectorized kernel at
`aten/src/ATen/native/cpu/ReduceOpsKernel.cpp:222-255`: a width-8
`Vectorized<float>` (AVX2 — this host has avx2, no avx512) accumulates each
element's square into one of 8 lanes via `acc_fvec += data_fvec * data_fvec`
(`norm_two_reduce_step`, ReduceOpsKernel.cpp:176-178), the 8 lanes are stored to
a `buffer[8]` and folded into `buffer[0]` by a NAIVE LEFT-FOLD
(`for j in 1..8: buffer[0] += buffer[j]`), the `size % 8` remainder elements
accumulate into `buffer[0]` by a scalar `buffer[0] += data*data` loop, and the
final `std::sqrt(buffer[0])` is the L2 norm. The accumulator type is
`at::opmath_type<float> == float` (`OpMathType.h:16`), i.e. f32 — NOT f64.

The lane accumulate compiles to AVX2 `_mm256_mul_ps` + `_mm256_add_ps` (NOT a
fused multiply-add — `operator*` is `_mm256_mul_ps` at `vec256_float.h:564`),
while the scalar remainder's `buffer[0] += data*data` contracts to a hardware
FMA under PyTorch's build flags. `simd_reduce.rs` models exactly this split: a
scalar model of the 8 lanes with plain mul+add, a left-fold, then a `mul_add`
(FMA) tail. A scalar model is used (not `std::arch` intrinsics) for portability
and determinism — f32 `a + b*b` without contraction is bit-identical to the
AVX2 `_mm256_add_ps(a, _mm256_mul_ps(b, b))` lane-wise, so the scalar lane model
reproduces the AVX2 rounding exactly.

## Requirements

- REQ-1: `pub fn l2_norm_f32_torch(data: &[f32]) -> f32` — the torch-matching
  f32 L2 reduction. Models the width-8 lane accumulate + naive left-fold +
  scalar FMA remainder + `sqrt` of the vectorized last-dim L2 kernel
  (`ReduceOpsKernel.cpp:222-255`), with `NormTwoOps::reduce = acc + data*data`
  and `NormTwoOps::project = device_sqrt` (`SharedReduceOps.h:365-392`).
  Accumulator is f32 per `opmath_type<float>` (`OpMathType.h:16`). Must
  reproduce the #1612/#1614 boundary row `0x4201970d` and ~97% of a live-torch
  oracle sweep (vs ~79% for the prior `powf` scalar path), with the residual
  bounded to one ULP.

## Acceptance Criteria

- [x] AC-1: `l2_norm_f32_torch` reproduces live torch `at::norm(2.0)` f32 bits
  on the #1614 boundary row `[3.6006885, 18.799816, 0.4159323, -2.6984854,
  -4.786058, 25.550726]` == `0x4201970d`
  (`simd_reduce.rs` test `matches_torch_boundary_row_1614`).
- [x] AC-2: byte-exact on representative lengths 1, 6, 7, 8, 13, 16, 17 against
  live torch oracle bits (`matches_torch_len*` tests).
- [x] AC-3: a known-residual row (`matches one ULP below torch`) is pinned
  honestly with both the model value and the live-torch value recorded
  (`known_residual_one_ulp_below_torch`), so the ~3% residual is tracked, not
  hidden (R-HONEST-3).
- [x] AC-4: the empty slice norms to `0.0` (`empty_is_zero`).
- [x] AC-5: feeding the matched norm back as `max_norm` reproduces torch's
  "not greater → do not clip" decision (`boundary_decision_does_not_clip`),
  the exact `norm > max_norm` test `embedding_renorm_cpu_` makes
  (`Embedding.cpp:204`).

## REQ status

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (`l2_norm_f32_torch`) | SHIPPED | impl: `pub fn l2_norm_f32_torch` in `simd_reduce.rs` per upstream `ReduceOpsKernel.cpp:222-255` + `SharedReduceOps.h:365-392` + `OpMathType.h:16`. Non-test production consumers: `pub fn norm_with_dim` in `grad_fns/reduction.rs` (the `p==2.0`, `T==f32`, last-dim-contiguous slice path) and `fn renorm_weight_rows_in_place` in `ferrotorch-nn/src/embedding.rs` (the `norm_type==2.0`, `T==f32` renorm-decision path, itself consumed by `Embedding::forward` / `EmbeddingBag::forward_bag`). |

## Known residual (R-HONEST-3)

This primitive is NOT 100% byte-exact with torch. On a 400-row live-torch
oracle (this AVX2 host, lengths 1..65) it matches ~97% of rows; the prior
scalar `powf` path matched ~79%. The residual ~3% are one-ULP misses at certain
non-multiple-of-8 lengths where torch's compiled scalar-remainder FMA
contraction differs from a portable Rust loop. The #1612/#1614 boundary cases
are in the matching set, so the renorm decisions they pin match torch. The
parity-sweep envelope (atol 1e-7) tolerates the residual sub-ULP differences on
non-boundary rows; only adversarial boundary probes (the #1614 tests) are
sensitive to them. Pushing past 97% would require reproducing the exact
instruction-selection of PyTorch's compiled binary (which sub-expressions GCC
contracted to FMA), which is not achievable in portable Rust and is out of
scope for the boundary-decision contract this primitive exists to satisfy.
