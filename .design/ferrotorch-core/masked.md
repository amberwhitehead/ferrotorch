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
  `masked_invalid` / `masked_equal` compute their boolean predicate
  on-device for f32/f64 CUDA inputs (`GpuBackend::isfinite_mask` /
  `ne_scalar_mask`, #1545); the mask is read back once to the
  host-resident `Vec<bool>` (no value-data round trip). `masked_where`
  takes a host `&[bool]` and is device-agnostic. Mirrors
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
- [x] AC-8: GPU paths for `masked_invalid` / `masked_equal`
  constructors (f32/f64) — the predicate runs on-device and the input
  stays CUDA-resident. Covered by `gpu_masked_invalid_f32_matches_cpu_\
  and_is_on_device`, `gpu_masked_invalid_f64_matches_cpu`,
  `gpu_masked_equal_f32_matches_cpu_and_is_on_device`,
  `gpu_masked_equal_f64_nan_is_unequal` in
  `tests/conformance_masked.rs` (`--features gpu`, live CUDA).
  `masked_where` needs no GPU path (host `&[bool]` condition).

## Architecture

`MaskedTensor<T: Float>` (`MaskedTensor in masked.rs`) holds `data: Tensor<T>`,
`mask: Vec<bool>` (flat, length `data.numel()`), and
`fill_value: T`. Constructors at `:52-85`. Accessors at `:87-118`.

`fn filled in masked.rs` returns on the data tensor's device (#1759 /
CORE-065 device contract). On a CPU data tensor it walks
`zip(data_vec, mask).map(|v, m| if m { v } else { fill })` (logical
order, non-contiguous views included); on CUDA it routes through
`masked_fill_bt in grad_fns/indexing.rs` with a device-resident
`BoolTensor` mask → the dtype-generic `masked_fill_dt` kernel. Both
paths attach a `MaskedFillBackward` edge when the data tensor tracks
gradients (#1758 / CORE-064; torch `MaskedTensor.to_tensor` is
`grad_fn=MaskedFillBackward0` on live 2.11.0).

`masked_sum in masked.rs` branches on `is_cuda() && (is_f32 || is_f64)`
to the GPU lowering: `mask_as_float_tensor` lifts the bool mask onto
the device as a `[0/1]` float tensor, then `backend.mul_{f32,f64}` +
`backend.sum_{f32,f64}` produce the result on-device. CPU fallback
is a single-pass `zip` accumulator.

`fn masked_mean in masked.rs` reuses `masked_sum_gpu` for the
numerator; the denominator is the host-side `count_valid()` (a `bool`
vec walk). The division runs ON-DEVICE (`div_f32` / `div_f64` against
the uploaded count scalar — #1759): the GPU sum never crosses back to
the host and the result is a CUDA 0-d scalar. The all-masked NaN edge
is uploaded to the data device.

`fn masked_min in masked.rs` / `fn masked_max in masked.rs` use the
dedicated fused `backend.masked_min_{f32,f64}` /
`masked_max_{f32,f64}` PTX kernels (#627) on CUDA. The kernel reads
`(data, mask_f)` and folds the sentinel-fill into the running min/max
in a single launch — no intermediate `prod` / `filled` buffers. CPU
path walks data + mask with an `Option<T>` accumulator. The all-masked
NaN sentinel (#1924 pin) is returned on the data device (#1759).

Autograd (#1758): `masked_sum` / `masked_mean` / `masked_min` /
`masked_max` attach `MaskedSumBackward` / `MaskedMeanBackward` /
`MaskedExtremumBackward` nodes when the data tensor tracks gradients.
Gradient contracts are quoted from live torch 2.11.0+cu130 on each
node: sum routes the upstream gradient to valid positions; mean scales
by `1/count_valid`; extrema split the gradient EVENLY among valid
positions equal to the saved forward result (torch tie contract);
all-masked routes zero gradients (torch-probed) while the forward
value stays under the #1924 pin. Extremum tie detection on CUDA
f32/f64 reuses the `ne_scalar_mask` predicate readback (#1545 — mask
bytes only, value data stays on device).

`masked_count in masked.rs` returns a 0-D tensor in `T` holding
`count_valid() as T`, uploaded to the data device when the data is
CUDA-resident (#1759). It is non-differentiable (constant in the data
values) and stays `requires_grad = false`.

`masked_where in masked.rs` inverts the condition (`!c` per element)
to match the torch convention; it takes a host `&[bool]` and is
device-agnostic. `masked_invalid in masked.rs` and
`masked_equal in masked.rs` build the mask: on a CPU data tensor they
walk host memory; on an f32/f64 CUDA data tensor they run the
predicate on-device (`GpuBackend::isfinite_mask` for `isfinite`,
`GpuBackend::ne_scalar_mask` for `v != value`) and read the resulting
`DType::Bool` buffer back once via `predicate_mask_gpu in masked.rs`
into the host `Vec<bool>` — the value data never leaves and returns to
the device (#1545). bf16/f16 CUDA inputs still error
`NotImplementedOnCuda` (out of `MaskedTensor<T: Float>`'s GPU-lowered
dtype set).

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
| REQ-1 | SHIPPED | impl: `MaskedTensor::new` at `masked in masked.rs` mirrors `torch.masked.MaskedTensor.__init__` (`torch/masked/__init__.py`); non-test consumer: re-exported as `ferrotorch_core::MaskedTensor` at `masked in lib.rs`. The constructor IS the entry-point public API |
| REQ-2 | SHIPPED | impl: `MaskedTensor::from_data` at `masked.rs:78`; non-test consumer: re-exported as `MaskedTensor::from_data` via `lib.rs:167` |
| REQ-3 | SHIPPED | impl: `with_fill_value` at `masked in masked.rs`; non-test consumer: re-exported via `MaskedTensor` builder at `masked in lib.rs` |
| REQ-4 | SHIPPED | impl: `filled` / `to_tensor` at `masked in masked.rs,143`; non-test consumer: re-exported method on `MaskedTensor` at `masked in lib.rs` |
| REQ-5 | SHIPPED | impl: `masked_sum`/`masked_mean`/`masked_min`/`masked_max`/`masked_count` at `masked in masked.rs,275,322,330,419`; non-test consumer: re-exported at `masked in lib.rs` |
| REQ-6 | SHIPPED | impl: `masked_where`/`masked_invalid`/`masked_equal in masked.rs`; non-test consumer: re-exported at `lib.rs`. GPU predicate masks (f32/f64): `masked_invalid in masked.rs` consumes `GpuBackend::isfinite_mask`, `masked_equal in masked.rs` consumes `GpuBackend::ne_scalar_mask` (#1545) — the constructors' CUDA branches ARE the non-test production consumers of the new trait methods |
| REQ-7 | SHIPPED | impl: `to_ferray` at `masked in masked.rs`; non-test consumer: the bridge enables ferray-ma's wider op surface; `to_ferray_round_trip_mean_matches_inhouse` test pins the cross-check |
