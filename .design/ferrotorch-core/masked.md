# Masked Tensors

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - aten/src/ATen/native/TensorAdvancedIndexing.cpp
  - torch/masked/__init__.py
-->

## Summary

`ferrotorch-core/src/masked.rs` implements `MaskedTensor` — a
data-tensor + boolean-mask pair mirroring `torch.masked.MaskedTensor`
(`torch/masked/__init__.py`). The torch convention is `mask[i] ==
true → VALID`; ferrotorch follows this convention internally and
inverts at the `ferray_ma` bridge boundary (where NumPy's "mask=true
means invalid" semantics apply). The module ships `masked_sum`,
`masked_mean`, `masked_min`, `masked_max`, `masked_count` reductions
+ `masked_where`/`masked_invalid`/`masked_equal` numpy-style
constructors + a `to_ferray` bridge.

## Requirements

- REQ-1: `MaskedTensor::new(data, mask)` — construct from data tensor
  + flat boolean mask of length `data.numel()`. Errors with
  `ShapeMismatch` on length mismatch. Mirrors
  `torch.masked.MaskedTensor.__init__`.
- REQ-2: `MaskedTensor::from_data(data)` — wrap data tensor with
  all-valid mask. Convenience constructor.
- REQ-3: `MaskedTensor::with_fill_value(value)` — override the fill
  value used by `filled()` / `to_tensor()`. Builder-style chained
  method. Defaults to zero. Mirrors `torch.masked.MaskedTensor.set_data`
  / `set_mask` indirectly (we keep one immutable constructor + a
  builder method).
- REQ-4: `filled()` / `to_tensor()` — materialise as a plain
  `Tensor<T>` with `fill_value` at masked-out positions. Mirrors
  `torch.masked.MaskedTensor.to_tensor`.
- REQ-5: `masked_sum` / `masked_mean` / `masked_min` / `masked_max` /
  `masked_count` — return 0-D tensors. GPU paths for `masked_sum` /
  `masked_mean` lower to `mul + reduce_sum` on f32/f64 (#597);
  `masked_min` / `masked_max` use the fused `masked_min_*` /
  `masked_max_*` PTX kernels (#627). Mirrors
  `torch.masked.{sum, mean, amin, amax}`.
- REQ-6: `masked_where(data, condition)` / `masked_invalid(data)` /
  `masked_equal(data, value)` — numpy-style constructors;
  `masked_where` inverts the condition to match torch convention.
  CPU-only (blocked on #1534 for GPU). Mirrors
  `numpy.ma.{masked_where, masked_invalid, masked_equal}`.
- REQ-7: `to_ferray::<U>(op)` — bridge to `ferray_ma::MaskedArray<U,
  IxDyn>`. Inverts the mask at the boundary (ferrotorch `true`=valid
  vs numpy `true`=invalid). Element type is generic over
  `U: ferray_core::Element + Copy + num_traits::Float`.

## Acceptance Criteria

- [x] AC-1: `cargo test -p ferrotorch-core --lib masked::tests` passes
  (covers construction, masked_where/invalid/equal, sum/mean/min/max/
  count reductions, filled, to_ferray round-trip).
- [x] AC-2: `MaskedTensor::new(d, mask)` errors when
  `mask.len() != d.numel()` (`new_rejects_mask_length_mismatch`).
- [x] AC-3: `masked_sum` of `[1,2,3,4,5]` with mask `[T,F,T,F,T]`
  returns `9.0` (`masked_sum_skips_masked_entries`).
- [x] AC-4: `masked_mean` returns NaN when every entry is masked
  (`masked_mean_all_masked_returns_nan`).
- [x] AC-5: `masked_invalid` of `[1.0, NaN, 3.0, +inf]` produces mask
  `[T, F, T, F]` (`masked_invalid_masks_nan`).
- [x] AC-6: `to_ferray` round-trip mean matches in-house mean
  (`to_ferray_round_trip_mean_matches_inhouse`).
- [x] AC-7: `masked_min` / `masked_max` return NaN when fully masked
  (`masked_min_max_all_masked_returns_nan`).
- [ ] AC-8: GPU paths for `masked_where` / `masked_invalid` /
  `masked_equal` constructors — NOT-STARTED, blocked on #1534.

## Architecture

`MaskedTensor<T: Float>` (`masked.rs:43-50`) holds `data: Tensor<T>`,
`mask: Vec<bool>` (flat, length `data.numel()`), and
`fill_value: T`. Constructors at `:52-85`. Accessors at `:87-118`.

`filled()` at `masked.rs:131` walks
`zip(data_vec, mask).map(|v, m| if m { v } else { fill })`. Always
CPU — masked tensors don't have a CUDA storage representation today
(`masked_tensor` itself can wrap a CUDA `data` field but the mask is
host-side).

`masked_sum` at `:200` branches on `is_cuda() && (is_f32 || is_f64)`
to the GPU lowering: `mask_as_float_tensor` lifts the bool mask onto
the device as a `[0/1]` float tensor, then `backend.mul_{f32,f64}` +
`backend.sum_{f32,f64}` produce the result on-device. CPU fallback
is a single-pass `zip` accumulator.

`masked_mean` at `:275` reuses `masked_sum_gpu` for the numerator;
the denominator is the host-side `count_valid()` (a `bool` vec walk).
The single scalar `sum / count` runs on host because the count is a
runtime-resolved integer, not a constant.

`masked_min` / `masked_max` at `:322,330` use the dedicated fused
`backend.masked_min_{f32,f64}` / `masked_max_{f32,f64}` PTX kernels
(#627) on CUDA. The kernel reads `(data, mask_f)` and folds the
sentinel-fill into the running min/max in a single launch — no
intermediate `prod` / `filled` buffers. CPU path walks data + mask
with an `Option<T>` accumulator.

`masked_count` at `:419` returns a 0-D tensor in `T` holding
`count_valid() as T`.

`masked_where` at `:435` inverts the condition (`!c` per element) to
match the torch convention. `masked_invalid` at `:453` and
`masked_equal` at `:472` walk the data in host memory and build the
mask. All three constructors reject CUDA inputs with
`NotImplementedOnCuda`, tracked by #1534.

The `to_ferray` bridge at `:165-186` inverts the mask
(`!v` per element) so the resulting `ferray_ma::MaskedArray` uses
NumPy convention; this lets ferrotorch callers reach the wider
ferray-ma op surface (`var`, `std`, masked sort, masked ufunc) by
casting through.

**Non-test consumer**: re-exported at `lib.rs:167-170` as the public
surface `ferrotorch_core::{MaskedTensor, masked_count, masked_equal,
masked_invalid, masked_max, masked_mean, masked_min, masked_sum,
masked_where}`. The boundary IS the public API per goal.md S5; users
construct `MaskedTensor` and call the reductions in-place.

## Parity contract

`parity_ops = []` (no `torch.masked.*` op_db entry maps directly).
The numeric contract is exact match with `torch.masked` semantics
under the "true=valid" convention: all-masked → NaN for mean / min /
max, mask-length-mismatch → ShapeMismatch error. Verified through
unit tests + the `to_ferray` round-trip vs `ferray_ma::mean()`.

## Verification

`cargo test -p ferrotorch-core --lib masked::tests` covers 18 tests
across constructors, reductions (CPU branch), and the ferray-ma
bridge. GPU `masked_sum` / `masked_mean` paths (#597) and
`masked_min` / `masked_max` paths (#627) are covered by integration
tests in `ferrotorch-core/tests/conformance_masked.rs` (not in this
unit file).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `MaskedTensor::new` at `masked.rs:60` mirrors `torch.masked.MaskedTensor.__init__` (`torch/masked/__init__.py`); non-test consumer: re-exported as `ferrotorch_core::MaskedTensor` at `lib.rs:167`. The constructor IS the entry-point public API |
| REQ-2 | SHIPPED | impl: `MaskedTensor::from_data` at `masked.rs:78`; non-test consumer: re-exported as `MaskedTensor::from_data` via `lib.rs:167` |
| REQ-3 | SHIPPED | impl: `with_fill_value` at `masked.rs:84`; non-test consumer: re-exported via `MaskedTensor` builder at `lib.rs:167` |
| REQ-4 | SHIPPED | impl: `filled` / `to_tensor` at `masked.rs:131,143`; non-test consumer: re-exported method on `MaskedTensor` at `lib.rs:167` |
| REQ-5 | SHIPPED | impl: `masked_sum`/`masked_mean`/`masked_min`/`masked_max`/`masked_count` at `masked.rs:200,275,322,330,419`; non-test consumer: re-exported at `lib.rs:167-170` |
| REQ-6 | SHIPPED | impl: `masked_where`/`masked_invalid`/`masked_equal` at `masked.rs:435,453,472`; non-test consumer: re-exported at `lib.rs:167-170`. GPU paths NOT-STARTED, blocked on #1534 — does NOT block CPU-path SHIPPED for the constructor itself |
| REQ-7 | SHIPPED | impl: `to_ferray` at `masked.rs:165`; non-test consumer: the bridge enables ferray-ma's wider op surface; `to_ferray_round_trip_mean_matches_inhouse` test pins the cross-check |
