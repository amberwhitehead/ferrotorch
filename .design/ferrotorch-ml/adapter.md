# ferrotorch-ml::adapter — `Tensor ↔ ndarray` bridge primitives

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/_tensor.py
  - torch/utils/_pytree.py
-->

## Summary

`ferrotorch-ml/src/adapter.rs` exposes six bridge functions that
materialise `ferrotorch_core::Tensor<T>` values as `ndarray::Array1<T>`
/ `Array2<T>` (and back). The bridge is a single `memcpy` per call:
the adapter routes the tensor through `Tensor::data_vec()`, which
already handles the CPU/GPU device crossing transparently. Upstream
PyTorch ships the equivalent via `Tensor.numpy()` / `numpy()` zero-copy
view (`torch/_tensor.py`); ferrotorch uses `ndarray` rather than NumPy
because ndarray is the Rust ecosystem's idiomatic array type
(R-DEV-7 — Rust ecosystem analog).

## Requirements

- REQ-1: `pub fn tensor_to_array1<T: Float + Clone>(t: &Tensor<T>) -> FerrotorchResult<Array1<T>>`
  materialises a 1-D tensor as an `ndarray::Array1<T>` via one
  contiguous allocation. Rejects non-1-D input with
  `FerrotorchError::ShapeMismatch`. GPU inputs are transparently
  moved to host memory via `Tensor::data_vec()`.
- REQ-2: `pub fn tensor_to_array2<T: Float + Clone>(t: &Tensor<T>) -> FerrotorchResult<Array2<T>>`
  materialises a 2-D tensor as an `ndarray::Array2<T>` in row-major
  layout. Rejects non-2-D input. Preserves row-major layout on both
  sides — the row-major `Tensor` data is fed straight into
  `Array2::from_shape_vec`.
- REQ-3: `pub fn array1_to_tensor<T: Float>(arr: Array1<T>) -> FerrotorchResult<Tensor<T>>`
  builds a 1-D CPU tensor from an `Array1<T>`. Takes the array by
  value to consume its owned storage without an extra clone.
- REQ-4: `pub fn array2_to_tensor<T: Float>(arr: Array2<T>) -> FerrotorchResult<Tensor<T>>`
  builds a 2-D CPU tensor. Handles non-contiguous (transposed/sliced)
  Array2 by collecting into row-major before constructing the tensor;
  contiguous arrays go through `into_raw_vec_and_offset` to avoid the
  copy.
- REQ-5: `pub fn array1_usize_to_tensor<T: Float>(arr: Array1<usize>) -> FerrotorchResult<Tensor<T>>`
  encodes integer class labels (sklearn convention: `Array1<usize>`)
  as floats inside a CPU tensor. Uses `ferrotorch_core::numeric_cast::cast`
  to detect overflow / NaN / non-finite issues during the
  `usize -> T` conversion.
- REQ-6: `pub fn tensor_to_array1_usize<T: Float>(t: &Tensor<T>) -> FerrotorchResult<Array1<usize>>`
  decodes float-encoded class labels back to `Array1<usize>` for
  sklearn classification metrics. Rejects non-1-D input. Rejects
  NaN, infinite, and negative elements with
  `FerrotorchError::InvalidArgument` before the `as usize` cast.

## Acceptance Criteria

- [x] AC-1: `tensor_to_array1` round-trips a 1-D `f64` tensor through
  `array1_to_tensor` without value loss.
- [x] AC-2: `tensor_to_array2` round-trips a 2-D `f64` tensor through
  `array2_to_tensor` without value loss.
- [x] AC-3: `array2_to_tensor` handles a transposed (non-contiguous)
  `Array2` by collecting into row-major.
- [x] AC-4: `tensor_to_array1` rejects 2-D input with
  `FerrotorchError::ShapeMismatch`.
- [x] AC-5: `tensor_to_array2` rejects 1-D input with
  `FerrotorchError::ShapeMismatch`.
- [x] AC-6: `array1_usize_to_tensor` + `tensor_to_array1_usize`
  round-trips `[0, 1, 2, 3]` exactly.
- [x] AC-7: `tensor_to_array1_usize` rejects negative elements with
  `FerrotorchError::InvalidArgument`.
- [x] AC-8: `tensor_to_array1_usize` rejects NaN elements with
  `FerrotorchError::InvalidArgument`.

## Architecture

### `tensor_to_array1` / `tensor_to_array2` (REQ-1, REQ-2)

Both functions delegate to `Tensor::data_vec()`, which is the
device-agnostic materialiser in `ferrotorch-core` — it already
performs the GPU→host transfer if needed. The 1-D case wraps the
result in `Array1::from(...)`; the 2-D case calls
`Array2::from_shape_vec((rows, cols), data)` and maps the
ndarray-side error into `FerrotorchError::ShapeMismatch` (the only
failure mode is row/col mismatch with the data length, which signals
a corrupted tensor).

Pre-flight shape check ensures the input has the expected ndim and
yields a precise error message naming the offending shape.

### `array1_to_tensor` / `array2_to_tensor` (REQ-3, REQ-4)

Owned-array consumption via `Array1::into_raw_vec_and_offset()` /
`Array2::into_raw_vec_and_offset()` extracts the backing `Vec<T>`
without copying when the array is contiguous (the standard case
for `Array1` and for un-transposed `Array2`).

The 2-D path checks `arr.is_standard_layout()` first — when the
caller passes a transposed/sliced view (`arr.t().to_owned()` in the
test fixture), the raw-vec extraction would return data in column-
major order; the function falls back to `arr.iter().copied().collect()`
to repack in row-major. This is the explicit fast/slow split documented
in the function body.

Both functions construct the resulting tensor via
`Tensor::from_storage(TensorStorage::cpu(data), shape, false)` — the
`requires_grad = false` argument matches the bridge's role as a data-
movement primitive (not a differentiable op).

### `array1_usize_to_tensor` (REQ-5)

Sklearn classification metrics return labels as `Array1<usize>`. To
feed those back into a tensor pipeline, the adapter casts each
`usize` to `T` via `ferrotorch_core::numeric_cast::cast::<f64, T>`,
collecting `FerrotorchResult<Vec<T>>` so a cast failure (e.g. label
too large for an `f32` mantissa to represent exactly) surfaces as
`FerrotorchError::InvalidArgument`. The intermediate `f64` is the
widest finite-precision domain `Float` types reliably round-trip.

### `tensor_to_array1_usize` (REQ-6)

The decode path is the inverse: each tensor element is converted to
`f64` via `Float::to_f64`, validated finite + non-negative, and cast
to `usize`. The explicit per-element check is mandatory because
`as usize` on a NaN or negative float is undefined-on-cast in older
Rust and `0`-mapped in current Rust — neither is the intended sklearn
semantics. The function emits a specific
`FerrotorchError::InvalidArgument` naming the offending index when
the check fails.

### Non-test production consumers

- `ferrotorch-ml/src/datasets.rs` —
  `use crate::adapter::{array1_to_tensor, array1_usize_to_tensor, array2_to_tensor}`
  consumes the inverse adapters to pack ferrolearn dataset output
  back into tensor pairs.
- `ferrotorch-ml/src/metrics.rs` —
  `use crate::adapter::{tensor_to_array1, tensor_to_array1_usize}`
  consumes the forward adapters in every metric wrapper.
- `ferrotorch-ml/src/metrics.rs:292` — the internal
  `tensor_to_array1_f64` helper delegates to `Tensor::data_vec()`
  (the same primitive the adapter uses) for f64-typed metric
  arguments.

### Upstream PyTorch mapping (R-DEV-7 deviation)

Upstream uses `Tensor.numpy()` (`torch/_tensor.py`'s `numpy()`
method) for the host-side bridge. ferrotorch substitutes `ndarray`
because it's the Rust ecosystem's idiomatic typed-array crate
(`ndarray::Array1` / `Array2` carry the dimensionality in the type,
matching ferrotorch's typed-shape posture). The API contract
preserved is the row-major layout and the single-memcpy cost; the
implementation is the Rust analog.

## Parity contract

`parity_ops = []`. The adapter is a structural bridge with no
numerical computation of its own. Edge-case parity:

- **NaN / Inf in float→usize**: `tensor_to_array1_usize` rejects
  with `FerrotorchError::InvalidArgument`. Float-side NaN/Inf cannot
  represent a valid class label and would silently produce
  `0` / `usize::MAX` under `as usize`.
- **Negative floats in float→usize**: same rejection (sklearn
  expects `usize` labels in `[0, n_classes)`).
- **Non-contiguous Array2**: `array2_to_tensor` repacks via `iter().copied()`
  so the resulting tensor's row-major layout is correct regardless of
  the input array's stride.
- **Empty tensor**: shape `[0]` → empty `Array1::from(vec![])`
  succeeds; shape `[0, K]` / `[N, 0]` round-trips via
  `Array2::from_shape_vec((0, K), vec![])`.
- **GPU input**: `Tensor::data_vec()` performs the host transfer
  transparently per the crate-level relaxation documented in
  `lib.rs`.

## Verification

Tests in `mod tests in adapter.rs` (7 tests):

- `array1_round_trip` — `Tensor → Array1 → Tensor` value preservation.
- `array2_round_trip` — same for 2-D, including shape and data.
- `array2_handles_transposed_view` — non-contiguous Array2 path.
- `tensor_to_array1_rejects_2d` — shape-check enforcement.
- `tensor_to_array2_rejects_1d` — same.
- `array1_usize_round_trip` — `Array1<usize> → Tensor → Array1<usize>`.
- `tensor_to_array1_usize_rejects_negative` — negative float
  rejection.
- `tensor_to_array1_usize_rejects_nan` — NaN rejection.

The integration test `ferrotorch-ml/tests/conformance_ml_adapter.rs`
exercises the public symbols through their fully-qualified import
paths to confirm crate-level reachability.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-ml --lib adapter:: 2>&1 | tail -3
```

Expected: `8 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn tensor_to_array1<T: Float + Clone>` in `tensor_to_array1 in ferrotorch-ml/src/adapter.rs` mirroring upstream `Tensor.numpy()` (`/home/doll/pytorch/torch/_tensor.py` `numpy` method); non-test consumer: `r2_score in ferrotorch-ml/src/metrics.rs` `let yt = tensor_to_array1(y_true)?` inside `r2_score` (consumed by every regression metric wrapper). |
| REQ-2 | SHIPPED | impl: `pub fn tensor_to_array2<T: Float + Clone>` in `tensor_to_array2 in ferrotorch-ml/src/adapter.rs` mirroring upstream `Tensor.numpy()` for the 2-D case; non-test consumer: `Ok in ferrotorch-ml/src/datasets.rs` `Ok((array2_to_tensor(x_arr)?, ...))` inside `pack_xy_classify` (consumed by every dataset generator) — reverse direction; forward direction available for downstream pipelines. |
| REQ-3 | SHIPPED | impl: `pub fn array1_to_tensor<T: Float>` in `array1_to_tensor in ferrotorch-ml/src/adapter.rs`; non-test consumer: `Ok in ferrotorch-ml/src/datasets.rs` `Ok((array2_to_tensor(x_arr)?, array1_to_tensor(y_arr)?))` inside `pack_xy_regress` (consumed by `make_regression`). |
| REQ-4 | SHIPPED | impl: `pub fn array2_to_tensor<T: Float>` in `array2_to_tensor in ferrotorch-ml/src/adapter.rs` with the contiguous-vs-non-contiguous branch; non-test consumer: `pack_xy_classify in ferrotorch-ml/src/datasets.rs` inside `pack_xy_classify` and `pack_xy_regress` packing ferrolearn dataset feature matrices into tensors. |
| REQ-5 | SHIPPED | impl: `pub fn array1_usize_to_tensor<T: Float>` in `ferrotorch-ml/src/adapter.rs:185` using `ferrotorch_core::numeric_cast::cast`; non-test consumer: `ferrotorch-ml/src/datasets.rs:49` inside `pack_xy_classify` converts the `Array1<usize>` class-label output of every classification generator (`make_classification`, `make_blobs`, `make_moons`, `make_circles`, `load_iris`, `load_wine`, `load_breast_cancer`) back into a float tensor. |
| REQ-6 | SHIPPED | impl: `pub fn tensor_to_array1_usize<T: Float>` in `ferrotorch-ml/src/adapter.rs:218` with the finite + non-negative check; non-test consumer: `ferrotorch-ml/src/metrics.rs:265` `let yt = tensor_to_array1_usize(y_true)?` inside `accuracy_score` (consumed by every classification metric: `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`, `hamming_loss`, `balanced_accuracy_score`, `matthews_corrcoef`, `cohen_kappa_score`, `zero_one_loss`, `top_k_accuracy_score`, `log_loss`, `brier_score_loss`, `d2_brier_score`, `average_precision_score`, `roc_auc_score`). |
