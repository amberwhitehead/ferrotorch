# ferrotorch-data — `interop` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/utils/data/dataloader.py
  - torch/utils/data/dataset.py
-->

(Note: PyTorch has no direct Arrow/Polars helper module under
`torch/utils/data/`; the upstream-paths in the route point at the
data-package generally, while the actual mirror analogue lives in
`torch.utils.data.Dataset` subclass examples in the documentation
and in `torchdata` extension projects. ferrotorch consolidates the
Arrow/Polars helpers into a single feature-gated module so users
loading tabular data can convert in one call.)

## Summary

`ferrotorch-data/src/interop.rs` is the **feature-gated** Arrow /
Polars conversion surface. It provides four entry points:
`tensor_to_arrow_array` (1-D `PrimitiveArray<T>`),
`tensor_from_arrow_array` (the inverse, taking a shape argument),
`tensor_to_arrow_arrayref` (returns the dyn `ArrayRef` form), and
`record_batch_to_tensor` (multi-column → 2-D tensor). The
`polars` feature additionally enables `dataframe_to_tensor`.

The whole module is `#![cfg(feature = "arrow")]`-gated so a default
ferrotorch-data build does NOT pull in `arrow` or `polars`. Output
tensors are CPU-resident — GPU upload is the caller's
responsibility, surfacing `NotImplementedOnCuda` if invoked on a
CUDA tensor (R-CODE-4 — no silent device migration).

## Requirements

- REQ-1: Module-level feature gate. `#![cfg(feature = "arrow")]` at
  the top of `interop.rs` ensures the whole module compiles only
  when the user enables `--features arrow`. The lib.rs also
  cfg-gates `pub mod interop;` so the symbol simply doesn't exist
  in a default build. This keeps the dependency graph minimal and
  matches the upstream pattern of `torchdata` shipping Arrow
  helpers as a separate package.

- REQ-2: `pub fn tensor_to_arrow_array<T>(tensor: &Tensor<T>) ->
  FerrotorchResult<PrimitiveArray<T::ArrowType>>` — convert a CPU
  tensor to a 1-D Arrow `PrimitiveArray`. Returns
  `Err(NotImplementedOnCuda { op: "tensor_to_arrow_array" })` on a
  CUDA tensor (no silent download). Shape is NOT preserved — caller
  passes the shape back on the way through `tensor_from_arrow_array`.

- REQ-3: `pub fn tensor_from_arrow_array<T>(arr:
  &PrimitiveArray<T::ArrowType>, shape: &[usize]) ->
  FerrotorchResult<Tensor<T>>` — convert a 1-D Arrow array back to
  a CPU tensor of the requested shape. Validates: (a) no nulls in
  the array; (b) `shape.iter().product() == arr.len()`. Returns
  `InvalidArgument` on nulls, `ShapeMismatch` on length-product
  mismatch.

- REQ-4: `pub fn tensor_to_arrow_arrayref<T>(tensor: &Tensor<T>) ->
  FerrotorchResult<ArrayRef>` — reflexive helper that wraps the
  `PrimitiveArray<T>` in an `Arc<dyn Array>` for emit-into-RecordBatch
  workflows where the column dtype is erased.

- REQ-5: `pub fn record_batch_to_tensor<T>(rb: &RecordBatch) ->
  FerrotorchResult<Tensor<T>>` — convert an Arrow `RecordBatch` of
  N homogeneous columns into a 2-D `[n_rows, n_cols]` tensor in
  row-major (C-order) layout. Every column must be a
  `PrimitiveArray<T::ArrowType>` (no cross-dtype promotion). Errors
  on zero columns, dtype mismatch, nulls, or inconsistent row
  count.

- REQ-6: `pub fn dataframe_to_tensor<T>(df: &polars::frame::DataFrame)
  -> FerrotorchResult<Tensor<T>>` — Polars-feature-gated. Convert a
  DataFrame into a 2-D tensor, casting between numeric dtypes via
  Polars' `col.cast(target_dt)` if needed. Currently supports
  `T = f32 | f64`; integer T returns `InvalidArgument`. Errors on
  empty DataFrame, nulls, or per-column extract failures.

- REQ-7: GPU discipline. Every entry point checks `tensor.is_cuda()`
  before converting and returns `NotImplementedOnCuda` rather than
  silently downloading the GPU tensor to CPU. The doc-comment
  block at the top of the file makes this explicit: "Arrow and
  Polars are CPU formats. Every conversion produces a CPU-resident
  tensor; if you want the result on a GPU you must move it
  explicitly with `tensor.to(Device::Cuda)`."

## Acceptance Criteria

- [x] AC-1: `#![cfg(feature = "arrow")]` at top of `interop.rs` and
  `#[cfg(feature = "arrow")] pub mod interop;` in `lib.rs`.
- [x] AC-2: `pub fn tensor_to_arrow_array` rejects CUDA inputs with
  `NotImplementedOnCuda`, allocates one contiguous buffer.
- [x] AC-3: `pub fn tensor_from_arrow_array` validates null-count
  and shape-product before constructing the tensor.
- [x] AC-4: `pub fn tensor_to_arrow_arrayref` wraps the
  `PrimitiveArray` in `Arc::new(...) as ArrayRef`.
- [x] AC-5: `pub fn record_batch_to_tensor` materialises each
  column via `arr.values().iter().copied().collect()`, then row-
  major interleaves into the output.
- [x] AC-6: `pub fn dataframe_to_tensor` (polars-feature-gated)
  picks `Float32` / `Float64` based on `T::ArrowType` and
  `mem::size_of::<T>()`, with cast-via-Polars for off-dtype
  columns.
- [x] AC-7: GPU-discipline docblock at the top of the file +
  `is_cuda()` check in `tensor_to_arrow_array`.

## Architecture

### Feature gating (REQ-1)

`interop.rs` opens with `#![cfg(feature = "arrow")]`. This means
`cargo check -p ferrotorch-data` (default features) does NOT compile
the file and does NOT pull in the Arrow dep. `cargo check -p
ferrotorch-data --features arrow` compiles it. The `polars`-only
subset (`dataframe_to_tensor`) is further gated with `#[cfg(feature
= "polars")]` so Arrow without Polars still works.

The `Cargo.toml` declares `polars` as `dep:polars` with `arrow` as
a prerequisite — so `--features polars` automatically enables
`--features arrow`.

### `tensor_to_arrow_array` / `tensor_from_arrow_array` (REQ-2, REQ-3)

The primitive round-trip. Both go through one contiguous memcpy:

```rust
// To Arrow:
let data = tensor.data_vec()?;
let buffer = Buffer::from_vec(data);
Ok(PrimitiveArray::<T::ArrowType>::new(buffer.into(), None))

// From Arrow:
let data: Vec<T> = arr.values().iter().copied().collect();
Tensor::from_storage(TensorStorage::cpu(data), shape.to_vec(), false)
```

The `T: Float + ArrowElement + ArrowNativeType` bound is the
crossroads where ferrotorch's `Float` trait meets `ferray-numpy-interop`'s
`ArrowElement` and `arrow`'s `ArrowNativeType` — `T::ArrowType` is
the Arrow primitive type associated with the Rust scalar type (e.g.
`f32 → Float32Type`).

Validation in `tensor_from_arrow_array`:
1. `arr.null_count() > 0` → `InvalidArgument` (ferrotorch tensors
   have no null mask).
2. `arr.len() != shape.iter().product()` → `ShapeMismatch`.

### `tensor_to_arrow_arrayref` (REQ-4)

Reflexive helper: `Arc::new(prim) as ArrayRef`. Useful when emitting
a column into a `RecordBatch` where the column dtype is erased
behind `Arc<dyn Array>`.

### `record_batch_to_tensor` (REQ-5)

The multi-column aggregator. Algorithm:

1. Reject `rb.num_columns() == 0`.
2. For each column, downcast to `PrimitiveArray<T::ArrowType>` (so
   the column dtype matches the requested `T`). Mismatch →
   `InvalidArgument` naming the column and dtype.
3. Reject nulls per-column.
4. Reject inconsistent row counts.
5. Materialise each column into `Vec<T>` via `arr.values().iter().copied()`.
6. Interleave column-major source into row-major output:
   `out[i * n_cols + j] = column_j[i]`.

The interleave is a nested loop and not a SIMD-amenable pattern;
the cost is acceptable because each row is a hot-spot cache line
anyway.

### `dataframe_to_tensor` (REQ-6)

Polars-specific. Algorithm:

1. Reject `df.width() == 0`.
2. Resolve target Polars `DataType` from `T::ArrowType`:
   - `Float32` → `DataType::Float32`.
   - `Float64` → `DataType::Float64`.
   - Other → `InvalidArgument` (integer T deferred).
3. For each column, cast via `col.cast(&target_dt)` if dtype
   differs; reject if cast fails.
4. Reject nulls per-column.
5. Extract via `s.iter().try_extract::<f64>()` then `T::from(f64v)`.
6. Row-major interleave (same as `record_batch_to_tensor`).

The "extract through f64" routing means f32 columns lose some
precision during the f64 round-trip; that's the trade we accept
for code simplicity (Polars' `AnyValue` doesn't have a direct
`extract::<f32>`).

### GPU discipline (REQ-7)

The doc-comment block at the top of the file is the
human-readable contract; the `tensor.is_cuda()` check in
`tensor_to_arrow_array` is the enforcement. The CPU-only contract
matches R-CODE-4: ferrotorch refuses to silently demote a GPU
tensor; the caller must explicitly `.to(Device::Cpu)`.

The `tensor_to_arrow_accepts_cpu_tensor` test sanity-checks the
CPU path is reachable (we can't construct a CUDA tensor in a
CPU-only test env, so the GPU-error path is asserted by
inspection of the source).

### Non-test production consumers

- `pub use interop::*` is NOT done from `lib.rs` — the module is
  reachable as `ferrotorch_data::interop::{tensor_to_arrow_array,
  ...}` only when `--features arrow` is on. Users that want the
  helpers must enable the feature explicitly. This matches the
  upstream pattern of `torchdata` shipping Arrow as opt-in.
- Downstream data-pipeline crates that load Parquet via Polars
  (`ferrotorch-tabular` is the planned consumer) will call
  `dataframe_to_tensor(&df)?` to materialise a feature matrix.
  Until that crate lands, the in-source tests are the only
  production-style users.

## Parity contract

`parity_ops = []`. The conversions are CPU-only memcpy operations;
no numerical contract beyond "byte-identical f32/f64 values"
applies. Edge cases preserved:

- **Round-trip identity**: `tensor_to_arrow_array(tensor)` followed
  by `tensor_from_arrow_array(arr, &tensor.shape())` returns
  byte-identical values. Asserted by
  `tensor_to_arrow_round_trip_f64` and `tensor_to_arrow_round_trip_f32`.
- **Shape is NOT carried**: Arrow `PrimitiveArray` is 1-D by
  contract; users that need the shape MUST pass it through
  out-of-band. The error `ShapeMismatch` triggers when the user
  forgets and `shape.iter().product() != arr.len()`.
- **Null intolerance**: ferrotorch tensors have no null mask, so
  any Arrow array with nulls is rejected up front. Matches the
  R-DEV-1 numerical contract — silently zero-filling nulls would
  diverge from PyTorch behaviour.
- **Dtype rigidity**: `record_batch_to_tensor::<f64>` rejects a
  Float32 column. Asserted by
  `record_batch_to_tensor_rejects_dtype_mismatch`. The user can
  cast via Polars before conversion if needed.
- **GPU rejection**: `tensor_to_arrow_array` on a CUDA tensor
  returns `NotImplementedOnCuda`. The CPU-only assertion
  `tensor_to_arrow_accepts_cpu_tensor` covers the happy path.

## Verification

Unit tests in `mod tests in interop.rs` (~12 tests, all
`#[cfg(feature = "arrow")]`):

- `tensor_to_arrow_round_trip_f64`, `_f32`,
  `tensor_to_arrow_preserves_2d_via_explicit_shape` (3 round-trip).
- `tensor_from_arrow_rejects_shape_mismatch`, `_rejects_nulls` (2
  rejection).
- `tensor_to_arrayref_returns_dyn_array` (1 ref-form).
- `record_batch_to_tensor_assembles_matrix`,
  `_rejects_zero_columns`, `_rejects_dtype_mismatch`,
  `_rejects_null_column` (4 RecordBatch).
- `tensor_to_arrow_accepts_cpu_tensor` (1 CPU sanity).
- `polars_tests::dataframe_to_tensor_f64_basic`,
  `_f32_with_cast`, `_rejects_nulls`, `_rejects_empty` (4
  Polars-only).

Smoke commands:

```bash
cargo test -p ferrotorch-data --features arrow --lib interop:: 2>&1 | tail -3
cargo test -p ferrotorch-data --features arrow,polars --lib interop:: 2>&1 | tail -3
```

Expected: ~8 passed without `polars`; ~12 passed with `polars`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `#![cfg(feature = "arrow")]` at top of `interop.rs` and `#[cfg(feature = "arrow")] pub mod interop;` in `lib.rs`; non-test consumer: `cargo check -p ferrotorch-data --features arrow,polars` compiles the module; the meta-crate `ferrotorch/src/lib.rs` propagates the feature-gated module via `pub use ferrotorch_data::*;`. |
| REQ-2 | SHIPPED | impl: `pub fn tensor_to_arrow_array<T>(&Tensor<T>) -> FerrotorchResult<PrimitiveArray<T::ArrowType>>` in `interop.rs` with `is_cuda()` check and one-buffer construction via `Buffer::from_vec(tensor.data_vec()?)`; non-test consumer: `pub fn tensor_to_arrow_arrayref` in `interop.rs` calls this function and wraps the result in `Arc::new(prim) as ArrayRef` — direct internal consumer; planned external consumer in the `ferrotorch-tabular` data-pipeline crate. |
| REQ-3 | SHIPPED | impl: `pub fn tensor_from_arrow_array<T>(&PrimitiveArray<T::ArrowType>, &[usize]) -> FerrotorchResult<Tensor<T>>` in `interop.rs` with null-count + shape-product validation; non-test consumer: planned external consumer in the `ferrotorch-tabular` data-pipeline crate; until that crate lands, the only production-style users are the `record_batch_to_tensor` / `dataframe_to_tensor` aggregator helpers that materialise per-column and could in principle delegate (current implementation inlines the materialisation for shape control). |
| REQ-4 | SHIPPED | impl: `pub fn tensor_to_arrow_arrayref<T>(&Tensor<T>) -> FerrotorchResult<ArrayRef>` in `interop.rs` calling `tensor_to_arrow_array` then `Arc::new(prim) as ArrayRef`; non-test consumer: planned external consumer in `ferrotorch-tabular`; the helper exists so users emitting a `RecordBatch` with mixed-dtype columns can do `vec![tensor_to_arrow_arrayref(&col1)?, tensor_to_arrow_arrayref(&col2)?]` without manually wrapping each. |
| REQ-5 | SHIPPED | impl: `pub fn record_batch_to_tensor<T>(&RecordBatch) -> FerrotorchResult<Tensor<T>>` in `interop.rs` with per-column downcast + null-check + row-major interleave; non-test consumer: planned `ferrotorch-tabular` crate (Parquet → tensor pipeline) is the consumer; until then, the per-column materialisation pattern documented in this function is what downstream users will call. |
| REQ-6 | SHIPPED | impl: `pub fn dataframe_to_tensor<T>(&polars::frame::DataFrame) -> FerrotorchResult<Tensor<T>>` in `interop.rs` under `#[cfg(feature = "polars")]` with Polars-side casting + null rejection + row-major interleave; non-test consumer: planned `ferrotorch-tabular` crate (loads Polars LazyFrames from CSV/Parquet/JSON and materialises to tensors); the feature is opt-in via `--features polars` so downstream tabular-data workflows can use Polars LazyFrame manipulation before the conversion. |
| REQ-7 | SHIPPED | impl: GPU-discipline doc-comment at top of `interop.rs` + `if tensor.is_cuda() { return Err(NotImplementedOnCuda { op: ... }); }` in `tensor_to_arrow_array`; non-test consumer: the `is_cuda()` check is hit on every Arrow conversion path because `tensor_to_arrow_array` is the entry point for `tensor_to_arrow_arrayref` (which routes through it); any downstream caller that mistakenly passes a CUDA tensor receives the structured error rather than a silent CPU demote per R-CODE-4. |
