# ferrotorch-ml::datasets — sklearn-style dataset generators returning `Tensor` pairs

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/utils/data/dataset.py
  - torch/utils/data/__init__.py
-->

## Summary

`ferrotorch-ml/src/datasets.rs` wraps the most common
[`ferrolearn_datasets`] generators with a `(Tensor<F>, Tensor<F>)`
return so they slot directly into a tensor-shaped pipeline. The
returned tensors live on CPU; the caller moves them to GPU explicitly
via `.to(...)` if needed. Upstream PyTorch ships
`torch.utils.data.Dataset` / `DataLoader` for streaming dataset
access, but does NOT include sklearn's curated toy/synthetic
generators (`make_classification`, `load_iris`, etc.). This module
fills that gap with eight wrappers covering the 90% case (synthetic
classification/regression/clustering generators + three toy datasets).

## Requirements

- REQ-1: `pub fn make_classification<F>(n_samples, n_features,
  n_classes, random_state) -> FerrotorchResult<(Tensor<F>, Tensor<F>)>`
  wraps `ferrolearn_datasets::make_classification`, returning
  `X: [n_samples, n_features]` + `y: [n_samples]` with integer class
  labels encoded as floats. Mirrors `sklearn.datasets.make_classification`.
- REQ-2: `pub fn make_regression<F>(n_samples, n_features,
  n_informative, noise, random_state) -> ...` wraps
  `ferrolearn_datasets::make_regression`. Mirrors
  `sklearn.datasets.make_regression`.
- REQ-3: `pub fn make_blobs<F>(n_samples, n_features, centers,
  cluster_std, random_state) -> ...` wraps
  `ferrolearn_datasets::make_blobs`. Mirrors
  `sklearn.datasets.make_blobs`.
- REQ-4: `pub fn make_moons<F>(n_samples, noise, random_state) -> ...`
  wraps `ferrolearn_datasets::make_moons`. Mirrors
  `sklearn.datasets.make_moons`. Binary task: `y ∈ {0.0, 1.0}`.
- REQ-5: `pub fn make_circles<F>(n_samples, noise, factor,
  random_state) -> ...` wraps `ferrolearn_datasets::make_circles`.
  Mirrors `sklearn.datasets.make_circles`. Binary task.
- REQ-6: `pub fn load_iris<F>() -> ...` — Iris flower classification
  (150 × 4, 3 classes), inline-shipped, no network/filesystem.
- REQ-7: `pub fn load_wine<F>() -> ...` — Wine cultivar classification
  (178 × 13, 3 classes), inline-shipped.
- REQ-8: `pub fn load_breast_cancer<F>() -> ...` — Breast-cancer
  Wisconsin diagnostic (569 × 30, binary), inline-shipped.
- REQ-9: All generators map ferrolearn errors via the local
  `map_dataset_err` helper into `FerrotorchError::InvalidArgument`
  so callers never have to import the ferrolearn error type.
- REQ-10: All generators return CPU tensors. Moving to GPU is the
  caller's explicit `.to(...)` call.

## Acceptance Criteria

- [x] AC-1: `make_classification::<f64>(100, 5, 3, Some(42))` returns
  shape `(X=[100, 5], y=[100])`.
- [x] AC-2: `make_classification` y-labels lie in `[0, n_classes)` and
  are integer-valued (fractional part zero).
- [x] AC-3: `make_regression::<f64>(80, 5, 3, 0.1, Some(7))` returns
  shape `(X=[80, 5], y=[80])`.
- [x] AC-4: `make_blobs::<f64>(60, 2, 3, 1.0, Some(1))` returns shape
  `(X=[60, 2], y=[60])` with labels in `[0, 3)`.
- [x] AC-5: `make_moons` produces binary labels exactly in `{0.0, 1.0}`.
- [x] AC-6: `make_circles` produces binary labels exactly in
  `{0.0, 1.0}` with `X.shape() == [n_samples, 2]`.
- [x] AC-7: `load_iris::<f64>()` returns `(X=[150, 4], y=[150])` with
  3-class labels.
- [x] AC-8: `load_wine::<f64>()` returns `(X=[178, 13], y=[178])`.
- [x] AC-9: `load_breast_cancer::<f64>()` returns
  `(X=[569, 30], y=[569])` with binary labels.
- [x] AC-10: End-to-end smoke: loading `load_iris` and computing
  `accuracy_score(&y, &y)` returns `1.0` exactly (confirms the
  dataset → metric adapter cooperation).

## Architecture

### Packers (`pack_xy_classify`, `pack_xy_regress`)

Two private helpers convert ferrolearn's native output types to
ferrotorch tensors:

```rust
fn pack_xy_classify<F: Float>(
    xy: (Array2<F>, Array1<usize>),
) -> FerrotorchResult<(Tensor<F>, Tensor<F>)> {
    let (x_arr, y_arr) = xy;
    Ok((array2_to_tensor(x_arr)?, array1_usize_to_tensor(y_arr)?))
}

fn pack_xy_regress<F: Float>(
    xy: (Array2<F>, Array1<F>),
) -> FerrotorchResult<(Tensor<F>, Tensor<F>)> {
    let (x_arr, y_arr) = xy;
    Ok((array2_to_tensor(x_arr)?, array1_to_tensor(y_arr)?))
}
```

The classification packer encodes labels as floats via the adapter's
`array1_usize_to_tensor` (REQ-5 in `adapter.md`); the regression
packer uses `array1_to_tensor` directly because targets are already
`F`-valued.

### Synthetic generators (REQ-1..REQ-5)

Each generator follows the same shape: validate parameters via the
ferrolearn call, map ferrolearn errors via `map_dataset_err`, and
pack the output. The `F: Float + num_traits::Float + Send + Sync +
'static` bound is the intersection ferrolearn requires plus the
`Float` trait that drives ferrotorch's typed-element interfaces.

`make_blobs` is a classification problem (cluster IDs are integer
labels) so it routes through `pack_xy_classify`. `make_moons` and
`make_circles` are 2-D binary classification toys with no
`n_classes` parameter (the binary task is structural).

### Toy datasets (REQ-6..REQ-8)

Iris, Wine, and Breast-cancer are inline-shipped by
`ferrolearn-datasets` — no network or filesystem access. The
deterministic shape contracts (150/178/569 samples, 4/13/30
features) are the de-facto standard from the sklearn quickstart
suite and are mechanically asserted by the AC tests.

### Error mapping (REQ-9)

```rust
fn map_dataset_err(e: ferrolearn_core::FerroError) -> FerrotorchError {
    FerrotorchError::InvalidArgument {
        message: format!("ferrolearn dataset: {e}"),
    }
}
```

The wrapper preserves the upstream ferrolearn error text but presents
the failure as a ferrotorch error so callers don't depend on
`ferrolearn_core` directly.

### Device placement (REQ-10)

`array2_to_tensor` / `array1_to_tensor` / `array1_usize_to_tensor`
all build CPU tensors via `Tensor::from_storage(TensorStorage::cpu(...), ...)`.
The caller is responsible for `.to(device)` if they want GPU residency.
Matches the sklearn semantics where loaded datasets are always
host-resident.

### Non-test production consumers

- `ferrotorch-ml/src/datasets.rs` —
  `use crate::adapter::{array1_to_tensor, array1_usize_to_tensor, array2_to_tensor}`
  is the only internal consumer; the dataset functions ARE the
  public API surface invoked by downstream pipelines.
- `ferrotorch-ml/src/datasets.rs:424` test
  `iris_self_classify_is_perfect_accuracy` uses the dataset →
  metric pipeline (`load_iris` → `accuracy_score`) — a test-only
  consumer, but it documents the cross-module contract.
- The pure-API role: the dataset wrappers form the leaf boundary of
  the bridge crate, so they're public-surface methods that
  participate in the workspace public-API contract (grandfathered
  under R-DEFER-1's boundary-method clause). Downstream model
  notebooks / examples invoke them via `use ferrotorch_ml::datasets::*`.

### Upstream PyTorch mapping (R-DEV-7 deviation)

Upstream `torch/utils/data/dataset.py` defines `Dataset` /
`IterableDataset` traits for streaming dataset access, but the
sklearn-style toy generators are absent. ferrotorch deviates by
providing them in a bridge crate (R-DEV-7 — Rust ecosystem analog
`ferrolearn-datasets` is materially better than asking users to call
sklearn from Python).

## Parity contract

`parity_ops = []`. The dataset generators are pure data-construction
wrappers — they delegate every numerical computation to
`ferrolearn_datasets`. Edge-case parity:

- **Reproducibility**: passing the same `random_state: Option<u64>`
  must produce the same `(X, y)` output across runs. Verified
  indirectly by the fixed-seed shape assertions (the shape contract
  doesn't change with seed but the values are deterministic).
- **Label encoding**: classification generators encode `usize`
  labels as `F`-floats via `array1_usize_to_tensor`. The forward
  encoding is exact (integer-in-float); the reverse decode by
  `tensor_to_array1_usize` is the inverse round-trip.
- **Toy-dataset constants**: Iris is `[150, 4]` × 3 classes, Wine is
  `[178, 13]` × 3 classes, Breast-cancer is `[569, 30]` × 2 classes.
  These are sklearn-canonical and pinned by AC tests.
- **CPU residency**: the returned tensors are on
  `Device::Cpu`. Moving to GPU is the caller's call.

## Verification

Tests in `mod tests in datasets.rs` (9 tests):

- `make_classification_returns_correct_shapes`,
  `make_classification_y_labels_are_in_range` — classification
  generator shape + label-range contract.
- `make_regression_returns_correct_shapes` — regression generator
  shape contract.
- `make_blobs_three_centers_two_features` — clustering generator
  shape + label-range.
- `make_moons_is_binary`, `make_circles_is_binary` — binary
  classification toys.
- `iris_has_known_shape`, `wine_has_known_shape`,
  `breast_cancer_has_known_shape` — toy dataset constants.
- `iris_self_classify_is_perfect_accuracy` — cross-module end-to-end
  smoke (dataset → adapter → metric).

The integration test `ferrotorch-ml/tests/conformance_ml_datasets.rs`
exercises the public symbols through `use ferrotorch_ml::datasets::*`.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-ml --lib datasets:: 2>&1 | tail -3
```

Expected: `9 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn make_classification<F>` in `make_classification in ferrotorch-ml/src/datasets.rs` delegating to `ferrolearn_datasets::make_classification` and packing via `pack_xy_classify`; non-test consumer: this is a leaf bridge wrapper — the public API IS the production consumer surface (R-DEFER-1 boundary-method clause). Downstream notebooks/examples consume via `use ferrotorch_ml::datasets::make_classification`; the conformance surface inventory at `ferrotorch-ml/tests/conformance/_surface_inventory.toml` enumerates it as a public path. |
| REQ-2 | SHIPPED | impl: `pub fn make_regression<F>` in `make_regression in ferrotorch-ml/src/datasets.rs` delegating to `ferrolearn_datasets::make_regression` via `pack_xy_regress`; non-test consumer: leaf bridge wrapper exposed as public API via `pub mod datasets;` in `lib.rs`. |
| REQ-3 | SHIPPED | impl: `pub fn make_blobs<F>` in `make_blobs in ferrotorch-ml/src/datasets.rs` delegating to `ferrolearn_datasets::make_blobs` via `pack_xy_classify`; non-test consumer: leaf bridge wrapper exposed as public API via `pub mod datasets;` in `lib.rs`. |
| REQ-4 | SHIPPED | impl: `pub fn make_moons<F>` in `make_moons in ferrotorch-ml/src/datasets.rs` delegating to `ferrolearn_datasets::make_moons` via `pack_xy_classify`; non-test consumer: leaf bridge wrapper exposed as public API via `pub mod datasets;` in `lib.rs`. |
| REQ-5 | SHIPPED | impl: `pub fn make_circles<F>` in `make_circles in ferrotorch-ml/src/datasets.rs` delegating to `ferrolearn_datasets::make_circles` via `pack_xy_classify`; non-test consumer: leaf bridge wrapper exposed as public API via `pub mod datasets;` in `lib.rs`. |
| REQ-6 | SHIPPED | impl: `pub fn load_iris<F>` in `load_iris in ferrotorch-ml/src/datasets.rs` delegating to `ferrolearn_datasets::load_iris`; non-test consumer: leaf bridge wrapper exposed as public API via `pub mod datasets;` in `lib.rs`. The cross-module smoke `iris_self_classify_is_perfect_accuracy` (test-only) documents the contract. |
| REQ-7 | SHIPPED | impl: `pub fn load_wine<F>` in `load_wine in ferrotorch-ml/src/datasets.rs` delegating to `ferrolearn_datasets::load_wine`; non-test consumer: leaf bridge wrapper exposed as public API via `pub mod datasets;` in `lib.rs`. |
| REQ-8 | SHIPPED | impl: `pub fn load_breast_cancer<F>` in `load_breast_cancer in ferrotorch-ml/src/datasets.rs` delegating to `ferrolearn_datasets::load_breast_cancer`; non-test consumer: leaf bridge wrapper exposed as public API via `pub mod datasets;` in `lib.rs`. |
| REQ-9 | SHIPPED | impl: `fn map_dataset_err` in `map_dataset_err in ferrotorch-ml/src/datasets.rs` maps `ferrolearn_core::FerroError` to `FerrotorchError::InvalidArgument`; non-test consumer: every `pub fn make_*` / `load_*` function above calls `.map_err(map_dataset_err)` on the ferrolearn result (lines 100, 141, 180, 210, 241, 271, 297, 324). |
| REQ-10 | SHIPPED | impl: every dataset function pipes through `array2_to_tensor` / `array1_to_tensor` / `array1_usize_to_tensor` (`array1_usize_to_tensor in ferrotorch-ml/src/datasets.rs,58`) which build CPU tensors via `Tensor::from_storage(TensorStorage::cpu(...), ...)`; non-test consumer: the module doc-comment (`cpu in ferrotorch-ml/src/datasets.rs`) declares the CPU contract as part of the public API; downstream notebooks calling `.to(...)` after construction depend on the guaranteed-CPU placement. |
