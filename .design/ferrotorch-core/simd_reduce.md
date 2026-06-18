# SIMD reduction primitives (torch-matching CPU reductions)

<!--
tier: 3-component
status: draft
baseline-pytorch: local /home/doll/pytorch inspected 2026-06-18
upstream-paths:
  - aten/src/ATen/native/DispatchStub.cpp
  - aten/src/ATen/native/DispatchStub.h
  - aten/src/ATen/native/SharedReduceOps.h
  - aten/src/ATen/native/cpu/ReduceOpsKernel.cpp
  - aten/src/ATen/native/cpu/SumKernel.cpp
  - aten/src/ATen/AccumulateType.h
  - aten/src/ATen/cpu/vec/vec_base.h
  - aten/src/ATen/cpu/vec/vec256/vec256_float.h
  - aten/src/ATen/cpu/vec/vec512/vec512_float.h
  - aten/src/ATen/OpMathType.h
-->

## Summary

`ferrotorch-core/src/simd_reduce.rs` scalarizes PyTorch CPU reduction trees for
the contiguous rows that `grad_fns/reduction.rs` and `ferrotorch-nn` need:

- f32 L2 norm uses `ReduceOpsKernel.cpp`'s vectorized last-dim L2 tree:
  `Vectorized<float>` square accumulation, naive lane left-fold, scalar FMA
  tail, and f32 `sqrt`.
- f32/f64 sum uses current `SumKernel.cpp::cascade_sum`, not the older
  `Reduce.h` binary reducer. f32 sum uses f32 accumulation because
  `SumKernel.cpp` asks for `at::acc_type<float, true>`, which maps to float in
  `AccumulateType.h`; f64 sum accumulates in f64.
- Vector width follows PyTorch dispatch instead of one AVX2 machine:
  `ATEN_CPU_CAPABILITY` lower-case override first, then x86 AVX512/AVX2/default
  and the supported SVE256/VSX/ZVECTOR/default classes. Lane widths are
  `f32/f64 = 16/8` for AVX512, `8/4` for AVX2 and SVE256, `4/2` for VSX and
  ZVECTOR, and PyTorch default widths otherwise.

The original implementation and documentation were AVX2-host-shaped and the
sum path modeled an obsolete `Reduce.h` approximation. Issue #1793 corrected
that: the code now follows the local PyTorch source and live PyTorch oracle
rows for the ugly non-associative cases.

## Requirements

- REQ-1: `pub fn l2_norm_f32_torch(data: &[f32]) -> f32` models
  `ReduceOpsKernel.cpp:222-255` with PyTorch-selected
  `Vectorized<float>::size()`. Accumulator is f32 per `opmath_type<float>`.
  Must reproduce the #1612/#1614 embedding boundary row and keep the documented
  one-ULP scalar-tail residual pinned.

- REQ-2: `pub fn sum_f32(data: &[f32]) -> f32` / `pub fn sum_f64(data: &[f64])
  -> f64` model `SumKernel.cpp` contiguous inner `cascade_sum`: vector loads,
  `row_sum`, four-way `multi_row_sum`, scalar tail first, then vector partial
  lane fold. Must not fall back to sequential Rust iteration and must not
  hard-code AVX2 lane geometry.

- REQ-3: CPU capability selection follows `DispatchStub.cpp`: lower-case
  `ATEN_CPU_CAPABILITY` values only, then hardware dispatch. Upper-case values
  are intentionally invalid because PyTorch treats them as invalid.

## Acceptance Criteria

- [x] AC-1: L2 boundary row `[3.6006885, 18.799816, 0.4159323, -2.6984854,
  -4.786058, 25.550726]` produces live torch f32 bits `0x4201970d`
  (`matches_torch_boundary_row_1614`).
- [x] AC-2: L2 representative lengths 1, 6, 7, 8, 13, 16, 17 match live torch
  oracle bits; the known one-ULP residual row remains explicit
  (`known_residual_one_ulp_below_torch`).
- [x] AC-3: Dispatch-width tests prove AVX512, AVX2, SVE256, VSX, ZVECTOR, and
  default lane widths and PyTorch's case-sensitive environment override
  spelling.
- [x] AC-4: f32 sum proves PyTorch f32 accumulation semantics with
  `[16777216.0, 1.0, 1.0] -> 16777216.0`; an f64-accumulating implementation
  would return `16777218.0` and fail the test.
- [x] AC-5: f32 adversarial rows `[1e20, 1, -1e20, ...]` at lengths 31, 32, 33,
  64, and 65 match live PyTorch 2.11.0+cu130 AVX2 oracle bits from
  `reshape(1, n).sum(dim=1)`.
- [x] AC-6: f64 adversarial rows `[1e300, 1, -1e300, ...]` at lengths 31, 32,
  33, 64, and 65 match live PyTorch 2.11.0+cu130 AVX2 oracle bits from
  `reshape(1, n).sum(dim=1)`.
- [x] AC-7: AVX512 source-derived f32 sum geometry produces a different result
  than AVX2 on a non-associative row (`sum_f32_avx512_geometry_is_not_avx2_hardcoded`),
  catching future width-8 regressions on non-AVX2 hosts.

## REQ Status

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (`l2_norm_f32_torch`) | SHIPPED | Impl in `simd_reduce.rs` uses `torch_f32_lanes()` from PyTorch-style dispatch, f32 lane square accumulation, left-fold, FMA tail, and sqrt. Consumers: `grad_fns/reduction.rs::norm_with_dim` for f32 last-dim L2 and `ferrotorch-nn/src/embedding.rs::renorm_weight_rows_in_place` for `max_norm` boundary decisions. |
| REQ-2 (`sum_f32` / `sum_f64`) | SHIPPED | Impl in `simd_reduce.rs` mirrors `SumKernel.cpp` `row_sum` / `multi_row_sum` cascade and selected vector width. Consumers: `grad_fns/reduction.rs::reduce_axis_sum_contiguous`, reached by `sum_dim` and `mean_dim` last-dim CPU rows. |
| REQ-3 (dispatch width) | SHIPPED | `TorchCpuVectorClass` and `parse_torch_cpu_capability_override`; tests `cpu_capability_override_spelling_matches_pytorch`, `dispatch_classes_expose_pytorch_vector_widths`, and AVX512-vs-AVX2 adversarial sum geometry. |

## Known Residual

The f32 L2 primitive is not 100% byte-exact with torch for every row. On the
original 400-row live-torch oracle (AVX2 host, lengths 1..65), it matched about
97%; the prior scalar `powf` path matched about 79%. The remaining misses are
one-ULP cases caused by PyTorch's compiled scalar-remainder FMA contraction.
The boundary rows that drive `embedding(max_norm=...)` decisions are in the
matching set, and the residual row is pinned explicitly.

This residual does not apply to `sum_f32` / `sum_f64` as shipped in #1793. Those
now target the inspected `SumKernel.cpp` cascade and are probed against live
PyTorch adversarial rows plus source-derived non-AVX2 geometry.
