# ferrotorch-data — `dataset` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/utils/data/dataset.py
  - torch/utils/data/_utils/worker.py
-->

## Summary

`ferrotorch-data/src/dataset.rs` is the trait surface and core
dataset implementations mirroring PyTorch's `torch.utils.data`
dataset abstractions. It defines two traits — `Dataset` (map-style,
random access) and `IterableDataset` (streaming) — and provides five
concrete implementations: `VecDataset` (Vec-backed), `MappedDataset`
(transform composition), `TensorDataset` (slice-of-tensor),
`ConcatDataset` (end-to-end concatenation), and `ChainDataset`
(iterable chaining). The `WorkerInfo` struct supports
multi-worker partitioning for `IterableDataset`.

Mirrors `torch/utils/data/dataset.py:39-449` (the Dataset class
hierarchy) and `torch/utils/data/_utils/worker.py:79-99`
(`WorkerInfo` + `get_worker_info`).

## Requirements

- REQ-1: `pub trait Dataset: Send + Sync` with `type Sample: Send`,
  `fn len(&self) -> usize`, `fn is_empty(&self) -> bool { self.len()
  == 0 }`, and `fn get(&self, index: usize) ->
  FerrotorchResult<Self::Sample>`. Mirrors
  `torch/utils/data/dataset.py:39-71` `class Dataset(Generic[_T_co])`
  with its `__getitem__` / `__len__` contract. The `Send + Sync`
  bound is the Rust-side analog of PyTorch's "datasets must be
  picklable" requirement for multi-process loading (R-DEV-7: we use
  the typestate analog instead of a runtime check).

- REQ-2: `pub trait IterableDataset: Send + Sync` with `type Sample:
  Send` and `fn iter(&self, worker_info: Option<&WorkerInfo>) ->
  Box<dyn Iterator<Item = FerrotorchResult<Self::Sample>> + Send +
  '_>`. Mirrors `torch/utils/data/dataset.py:73-186` `class
  IterableDataset(Dataset[_T_co], Iterable[_T_co])`. The
  `WorkerInfo` parameter is how the dataset partitions its stream
  across multiple workers — same role as `get_worker_info()` upstream
  (`torch/utils/data/_utils/worker.py:99`).

- REQ-3: `pub struct WorkerInfo { pub worker_id: usize, pub
  num_workers: usize }` annotated `#[non_exhaustive]` so future
  fields (per-worker seed, rendezvous handle) can be added without a
  breaking change. Mirrors
  `torch/utils/data/_utils/worker.py:79-99` `class WorkerInfo` which
  carries `id`, `num_workers`, `seed`, `dataset` fields.

- REQ-4: `pub struct VecDataset<S: Send + Sync + Clone>` — in-memory
  dataset backed by `Vec<S>`. `impl<S: Send + Sync + Clone + 'static>
  Dataset for VecDataset<S>` returns `S` on `get` (cloned). The
  doc-comment notes "useful for testing and small datasets". Mirrors
  the `torch.utils.data.dataset.Dataset` use-pattern where the
  user's `__getitem__` returns from a Python list.

- REQ-5: `pub struct MappedDataset<D: Dataset, F>` — transform
  composition. Wraps a base `D: Dataset` and a closure `F:
  Fn(D::Sample) -> FerrotorchResult<O> + Send + Sync`. `impl
  Dataset` returns the transformed `O`. Mirrors PyTorch's
  `torch.utils.data.dataset.Dataset.__add__` / `map` ergonomics; the
  closer upstream analogue is the `torchvision.transforms.Compose`-
  wrapped dataset where the user writes `Dataset(transform=...)` in
  their `__init__`.

- REQ-6: `pub struct TensorDataset<T: Float>` — wraps a `Vec<Tensor<T>>`
  with the contract that all tensors share dim-0 size. `get(index)`
  returns `Vec<Tensor<T>>` with `select(t, 0, index)` applied to
  each stored tensor. Mirrors
  `torch/utils/data/dataset.py:189-209` `class TensorDataset`
  exactly: `__getitem__` returns `tuple(tensor[index] for tensor in
  self.tensors)`. Constructor validates all tensors are at least 1-D
  AND share the dim-0 size (matching upstream's
  `if any(tensors[0].size(0) != tensor.size(0) for tensor in
  tensors): raise AssertionError("Size mismatch between tensors")`).

- REQ-7: `pub struct ConcatDataset<D: Dataset>` — end-to-end
  concatenation of N datasets. Constructor caches the cumulative
  length array; `get(index)` does a binary search to locate which
  sub-dataset owns the index. Mirrors
  `torch/utils/data/dataset.py:299-355` `class ConcatDataset` with
  its `bisect.bisect_right(self.cumulative_sizes, idx)` index
  resolution.

- REQ-8: `pub struct ChainDataset<D: Dataset>` — implements BOTH
  `Dataset` (map-style, delegates to internal `ConcatDataset`) AND
  `IterableDataset` (streaming with worker-partitioned slicing).
  Mirrors `torch/utils/data/dataset.py:356-385` `class
  ChainDataset(IterableDataset)` but extends with the map-style
  `Dataset` impl since the internal `ConcatDataset` is already
  random-accessible.

## Acceptance Criteria

- [x] AC-1: `pub trait Dataset` in `dataset.rs` with the four
  required methods and `Send + Sync` super-trait.
- [x] AC-2: `pub trait IterableDataset` with the `iter(...)` method
  returning `Box<dyn Iterator<...> + Send + '_>`.
- [x] AC-3: `pub struct WorkerInfo` is `#[non_exhaustive]` with
  `pub worker_id: usize, pub num_workers: usize` and a `WorkerInfo::new`
  constructor.
- [x] AC-4: `pub struct VecDataset<S>` + `impl Dataset for
  VecDataset<S>` with `get(index)` returning a `cloned()` element or
  `IndexOutOfBounds`.
- [x] AC-5: `pub struct MappedDataset<D, F>` + `impl Dataset for
  MappedDataset<D, F>` with `get(index)` chaining through the
  transform closure.
- [x] AC-6: `pub struct TensorDataset<T: Float>` with `new(...)`
  validating dim-0-equality + at-least-1-D, `len()`/`is_empty()`/
  `get(...)` inherent methods, and `impl Dataset for TensorDataset<T>`.
- [x] AC-7: `pub struct ConcatDataset<D: Dataset>` with
  `new(...)` rejecting empty input, internal `cumulative` array, and
  `locate(...)` using `partition_point`.
- [x] AC-8: `pub struct ChainDataset<D: Dataset>` with `impl
  Dataset` AND `impl IterableDataset`; the iterable impl partitions
  via `WorkerInfo::worker_id / num_workers` with the
  `per_worker + (worker_id < remainder)` distribution.

## Architecture

### `Dataset` trait surface (REQ-1)

The trait is the Rust-side analog of `class Dataset(Generic[_T_co])`
upstream. The `Send + Sync` bound is mandatory for multi-worker
loading (workers are threads, not processes — R-DEV-7 — so the
Python pickling contract becomes a Rust thread-safety contract).
`type Sample: Send` is the associated type giving the dataset its
output type; this is what PyTorch leaves untyped (`_T_co` generic on
the class) but Rust needs explicit.

The default `is_empty(&self) -> bool { self.len() == 0 }` matches
the Python convention; users who can compute `is_empty` cheaper than
`len` can override.

### `IterableDataset` trait surface (REQ-2)

Returns a boxed `Iterator` so the dataset's iter-impl can choose any
concrete state machine (a `Range`, a closure-based generator, etc.).
The `+ Send` bound is required because workers may carry the iterator
across thread boundaries. The `+ '_` lifetime borrows `&self`,
matching the upstream "the dataset is held by the worker process for
the duration of iteration" contract.

### `WorkerInfo` (REQ-3)

`#[non_exhaustive]` is the load-bearing annotation: external code
cannot construct `WorkerInfo { worker_id, num_workers }` via struct
literal, only via `WorkerInfo::new(worker_id, num_workers)`. This
preserves room to add `seed: u64` / `rendezvous: Arc<...>` fields in
follow-up commits without breaking the public API. Mirrors
upstream's `WorkerInfo` which carries `id`, `num_workers`, `seed`,
`dataset` — ferrotorch ships just the first two for now (the others
are tracked as future-extension blockers in the `IterableDataset`
follow-up family but are NOT required for the current map-style
`DataLoader` pipeline).

### `VecDataset` (REQ-4)

The simplest dataset — `Vec<S>` storage, `get(i)` returns
`self.data.get(i).cloned()` mapped to `IndexOutOfBounds`. The
`S: Clone` bound is the cost of returning owned values from `get`;
users with non-Clone samples should use `MappedDataset` to wrap a
lazy-construction closure or write a custom `Dataset` impl.

### `MappedDataset` (REQ-5)

Generic over the base dataset `D` and the transform closure `F:
Fn(D::Sample) -> FerrotorchResult<O>`. The `O: Send + 'static` bound
on the `Dataset` impl matches `Send` from the trait + the `'static`
needed for boxed iterators in the loader pipeline. `get(i)` is:

```rust
let sample = self.inner.get(index)?;
(self.transform)(sample)
```

This is the typestate analog of PyTorch's "pass a callable
`transform` to the dataset's `__init__`" pattern, except composed
externally rather than baked into the dataset.

### `TensorDataset` (REQ-6)

`pub struct TensorDataset<T: Float>` stores `Vec<Tensor<T>>` plus a
cached `len: usize` (the shared dim-0 size). The constructor
enforces three invariants:

1. `tensors.is_empty()` → `InvalidArgument` (matches upstream's
   implicit failure when `tensors[0]` would IndexError).
2. Any tensor with `shape().is_empty()` (scalar) →
   `InvalidArgument`. PyTorch tolerates 0-D tensors here but the
   `tensor[index]` call would fail; ferrotorch rejects up front.
3. Any tensor with `shape[0] != len` → `ShapeMismatch`. Matches
   upstream's `AssertionError("Size mismatch between tensors")` at
   `dataset.py:202`.

`get(index)` returns `Vec<Tensor<T>>` by calling
`select(t, 0, index)` on each stored tensor — the same as
PyTorch's `tuple(tensor[index] for tensor in self.tensors)`. The
result has each tensor's dim-0 collapsed (e.g. `[3, 2]` becomes
`[2]`).

Both an inherent `len`/`is_empty`/`get` AND a `Dataset` trait impl
exist so callers can use the concrete struct directly OR through
the trait object. The `Dataset` impl forwards to the inherent
methods to avoid divergence.

### `ConcatDataset` (REQ-7)

`pub struct ConcatDataset<D: Dataset>` wraps `Vec<D>` plus a
`cumulative: Vec<usize>` cache where `cumulative[i] = sum of
len(datasets[0..=i])`. The constructor rejects an empty list.
`fn locate(&self, index: usize) -> (usize, usize)` in `dataset.rs`
uses `partition_point(|&cum| cum <= index)` to find the first
cumulative bucket strictly greater than `index`; the corresponding
local index is `index - cumulative[ds_idx - 1]` (or `index` itself
if the first dataset is the owner). This is the Rust translation of
upstream's `bisect.bisect_right(self.cumulative_sizes, idx)`.

### `ChainDataset` (REQ-8)

Layered on top of `ConcatDataset`: stores an internal
`ConcatDataset<D>` and forwards the map-style `Dataset` impl
straight through. The `IterableDataset` impl partitions the global
index range across workers using:

```rust
let per_worker = total / info.num_workers;
let remainder = total % info.num_workers;
let s = info.worker_id * per_worker + info.worker_id.min(remainder);
let extra = if info.worker_id < remainder { 1 } else { 0 };
(s, s + per_worker + extra)
```

This gives workers in `0..remainder` one extra sample, matching the
even-as-possible distribution PyTorch's `DistributedSampler` /
`get_worker_info` derives. The `bool_to_int_with_if` allow at the
crate root names this idiom explicitly (`if info.worker_id <
remainder { 1 } else { 0 }` is more legible than the bool-arith
rewrite).

### Non-test production consumers

- `DataLoader::new` and `DataLoader::iter` in `dataloader.rs` are
  generic over `D: Dataset` and call `self.dataset.get(idx)` /
  `self.dataset.len()` through this trait. This is the primary
  consumer for `Dataset`.
- The `iter()` body of `ChainDataset` is the production consumer of
  `IterableDataset` + `WorkerInfo`; the loader pipeline does not yet
  dispatch on `IterableDataset` (planned with the `_DatasetKind`
  follow-up).
- `pub use dataset::{...}` in `lib.rs` re-exports all eight items;
  the meta-crate glob propagates them as `ferrotorch::Dataset` etc.
- Downstream training-loop code in `ferrotorch-llama` /
  `ferrotorch-bert` constructs `TensorDataset::new(vec![inputs,
  targets])` for its supervised pipelines.

## Parity contract

`parity_ops = []`. The dataset abstractions are plumbing; the
numerical contract on the tensors they store is `ferrotorch-core`'s
responsibility. Edge cases preserved:

- **Out-of-bounds `get`**: every concrete impl returns
  `FerrotorchError::IndexOutOfBounds { index, axis: 0, size }`
  rather than panicking. Matches upstream's `IndexError` shape but
  with structured fields instead of a string message.
- **Empty dataset**: `len() == 0` is supported; `get` always returns
  OOB; the loader iterates zero batches. Upstream's behaviour
  matches.
- **Cumulative-length overflow**: `ConcatDataset::cumulative` uses
  `usize`, which is 64-bit on all supported platforms; overflow is
  practically impossible for real datasets. No explicit check.
- **`select` failure propagation**: `TensorDataset::get` propagates
  any `select` error verbatim; this catches device misalignment or
  shape edge cases that the constructor's checks didn't catch (e.g.
  if the user mutates a stored tensor's shape externally).
- **`Send + Sync` enforcement**: the test
  `test_dataset_is_send_sync` (in `mod tests`) statically asserts
  via `fn assert_send_sync<T: Send + Sync>() {}`. Catches accidental
  introduction of `Rc<RefCell<T>>` or `*mut`-bearing fields.
- **Worker partition correctness**: the `test_chain_dataset_iterable_with_workers`
  test exercises the `WorkerInfo` partition and asserts the two
  workers' outputs union back to the full range.

## Verification

Unit tests in `mod tests in dataset.rs` (~20 tests, grouped):

- `test_vec_dataset`, `test_vec_dataset_empty` — basic VecDataset
  contract.
- `test_mapped_dataset` — transform composition.
- `test_dataset_is_send_sync` — compile-time Send+Sync assertion.
- `test_tensor_dataset_basic`, `_single_tensor`, `_oob`,
  `_dim_mismatch`, `_empty_tensors`, `_scalar_rejected`,
  `_as_trait` — TensorDataset constructor + indexing contract.
- `test_concat_dataset_basic`, `_oob`, `_single`, `_empty_err`,
  `_boundary` — ConcatDataset index mapping.
- `test_chain_dataset_map_style`, `_iterable`,
  `_iterable_with_workers`, `_empty_err` — ChainDataset's dual
  trait impl.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-data --lib dataset:: 2>&1 | tail -3
```

Expected: ~20 passed.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub trait Dataset: Send + Sync` with `type Sample: Send` and four required methods in `dataset.rs`, mirroring `torch/utils/data/dataset.py:39-71`; non-test consumer: `pub use dataset::Dataset` in `lib.rs` and `DataLoader<D: Dataset>` in `dataloader.rs` is generic over this trait, calling `self.dataset.get(idx)` and `self.dataset.len()` at multiple sites in the iterator state machines. |
| REQ-2 | SHIPPED | impl: `pub trait IterableDataset: Send + Sync` with `iter(&self, worker_info: Option<&WorkerInfo>) -> Box<dyn Iterator<...> + Send + '_>` in `dataset.rs`, mirroring `torch/utils/data/dataset.py:73-186`; non-test consumer: `impl IterableDataset for ChainDataset<D> in dataset.rs` is the production user of the trait, and `pub use dataset::IterableDataset` in `lib.rs` re-exports the trait for the meta-crate surface. |
| REQ-3 | SHIPPED | impl: `pub struct WorkerInfo { pub worker_id: usize, pub num_workers: usize }` annotated `#[non_exhaustive]` with `WorkerInfo::new(...)` constructor in `dataset.rs`, mirroring `torch/utils/data/_utils/worker.py:79-99`; non-test consumer: `fn ChainDataset::iter` in `dataset.rs` accepts `Option<&WorkerInfo>` and reads `info.worker_id` / `info.num_workers` to partition its stream. |
| REQ-4 | SHIPPED | impl: `pub struct VecDataset<S: Send + Sync + Clone>` + `impl<S: ... + 'static> Dataset for VecDataset<S>` in `dataset.rs` with `get` returning `cloned()` or `IndexOutOfBounds`; non-test consumer: `pub use dataset::VecDataset` in `lib.rs` re-exports the type, and the loader's tests + downstream toy examples construct `VecDataset::new(vec![...])`. |
| REQ-5 | SHIPPED | impl: `pub struct MappedDataset<D: Dataset, F>` + `impl<D, F, O> Dataset for MappedDataset<D, F>` in `dataset.rs` calling the transform closure on each `get`; non-test consumer: `pub use dataset::MappedDataset` in `lib.rs` and `MappedDataset::new(ds, |x| Ok(x * 10))` composition in downstream code (and in `test_mapped_dataset` for the trait surface). |
| REQ-6 | SHIPPED | impl: `pub struct TensorDataset<T: Float>` + inherent `len`/`is_empty`/`get` + `impl<T: Float + 'static> Dataset for TensorDataset<T>` in `dataset.rs`, mirroring `torch/utils/data/dataset.py:189-209`; constructor validates dim-0 size + at-least-1-D; non-test consumer: `pub use dataset::TensorDataset` in `lib.rs` re-exports it, and downstream supervised-learning pipelines (in `ferrotorch-llama` / `-bert` training drivers) construct `TensorDataset::new(vec![inputs, targets])`. |
| REQ-7 | SHIPPED | impl: `pub struct ConcatDataset<D: Dataset>` with cached `cumulative: Vec<usize>` and `locate` via `partition_point` in `dataset.rs`, mirroring `torch/utils/data/dataset.py:299-355`; non-test consumer: `pub use dataset::ConcatDataset` in `lib.rs` and `ChainDataset<D>` internally wraps a `ConcatDataset<D>` so its map-style `Dataset` impl forwards to `ConcatDataset::get`. |
| REQ-8 | SHIPPED | impl: `pub struct ChainDataset<D: Dataset>` with both `impl Dataset` (delegating to internal `ConcatDataset`) and `impl IterableDataset` (worker-partitioned slicing) in `dataset.rs`, mirroring `torch/utils/data/dataset.py:356-385`; non-test consumer: `pub use dataset::ChainDataset` in `lib.rs` and the iterable trait impl is the production user of `WorkerInfo` (so REQ-2 + REQ-3 + REQ-8 all unblock together). |
