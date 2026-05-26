# ferrotorch-data — `dataloader` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/utils/data/dataloader.py
  - torch/utils/data/_utils/worker.py
  - torch/utils/data/_utils/collate.py
-->

## Summary

`ferrotorch-data/src/dataloader.rs` is the biggest single file in the
crate (2460 LOC) and the central consumer of every other module here.
It implements `DataLoader<D: Dataset>` — the builder-style entry
point users call as `DataLoader::new(Arc::new(ds), batch_size)` —
plus three iterator variants (`Sync`, `Prefetch`, `MultiWorker`)
hidden behind the `BatchIter` enum, a `CollatedIter` adapter, the
`ToDevice` trait + blanket impl for `Tensor<T>`, the `WorkerMode`
enum, and the internal `worker_loop` / `panic_payload_message` /
`compute_batch_count` helpers.

Mirrors `torch/utils/data/dataloader.py:142-617` `class DataLoader`
and its `_BaseDataLoaderIter` / `_SingleProcessDataLoaderIter` /
`_MultiProcessingDataLoaderIter` subclasses at lines 619-784. Uses
threads instead of processes (R-DEV-7: Rust has no GIL, so the
process-per-worker upstream pattern is unnecessary; threads share
the address space and have lower spawn cost).

## Requirements

- REQ-1: `pub struct DataLoader<D: Dataset>` with all builder
  fields: `dataset: Arc<D>, batch_size, shuffle, drop_last, seed,
  num_workers, prefetch_factor, worker_mode, device, pin_memory,
  custom_sampler, collate_fn, transfer_fn`. Constructor
  `DataLoader::new(dataset, batch_size) -> FerrotorchResult<Self>`
  rejects `batch_size == 0` with `InvalidArgument`. Builder methods
  consume-and-return for ergonomic chaining: `shuffle(bool)`,
  `drop_last(bool)`, `seed(u64)`, `num_workers(usize)`,
  `prefetch_factor(usize)`, `worker_mode(WorkerMode)`,
  `device(Device)`, `pin_memory(bool)`, `with_sampler(Box<dyn
  Sampler>)`, `with_collate(F)`. Mirrors
  `torch/utils/data/dataloader.py:248-268` `__init__` signature.

- REQ-2: `pub enum WorkerMode { IntraBatch, CrossBatch }` with
  `#[derive(Default)]` → `IntraBatch`. `IntraBatch` (default) uses
  rayon's work-stealing pool to load samples within one batch in
  parallel; `CrossBatch` spawns `num_workers` dedicated threads that
  produce independent batches in parallel with a reorder buffer for
  deterministic output ordering. The split mirrors PyTorch's
  `num_workers=0` (sync) vs `num_workers > 0` (multi-process) but
  recasts the upstream choice as a `WorkerMode` enum so both
  parallelism strategies are first-class.

- REQ-3: `pub trait ToDevice: Sized` with `to_device(&self, device:
  Device) -> FerrotorchResult<Self>` and `to_device_pinned(&self,
  device: Device) -> FerrotorchResult<Self>` (with a default that
  forwards to `to_device`). The trait is the typestate analogue of
  PyTorch's `pin_memory_device` plumbing: callers with custom batch
  types implement `ToDevice` so the loader can move whole samples
  to GPU. The blanket `impl<T: Float> ToDevice for Tensor<T>` makes
  raw tensors usable as Sample without a wrapper.

- REQ-4: `pub enum BatchIter<'a, D: Dataset>` with three variants:
  `Sync(DataLoaderIter<'a, D>)` for the calling-thread path,
  `Prefetch(PrefetchIter<D>)` for the single-background-thread
  prefetch path, `MultiWorker(MultiWorkerIter<D>)` for the
  `num_workers > 0 && WorkerMode::CrossBatch` path. Implements
  `Iterator<Item = FerrotorchResult<Vec<D::Sample>>>` and
  `ExactSizeIterator`. The dispatch in `DataLoader::iter(epoch)` is:
  - `WorkerMode::CrossBatch && num_workers > 0` → `MultiWorker`.
  - else if `prefetch_factor > 0` → `Prefetch`.
  - else → `Sync`.

- REQ-5: `pub struct DataLoaderIter<'a, D: Dataset>` — the
  synchronous variant. Holds `&'a D` (borrowed dataset), the
  pre-computed `indices: Vec<usize>` for one epoch, `batch_size`,
  `drop_last`, `num_workers` (for intra-batch rayon parallelism),
  optional `transfer_fn`, `pin_memory` flag, and the `pos` cursor.
  `next()` slices `indices[pos..pos+batch_size]`, loads via
  sequential or `par_iter()` based on `num_workers > 0`, applies
  the optional device transfer, and yields the batch.

- REQ-6: `pub struct PrefetchIter<D: Dataset>` — the single-
  background-thread prefetch variant. A bounded crossbeam channel
  of capacity `prefetch_factor` buffers batches ahead of the
  consumer. The background thread is wrapped in `catch_unwind` so
  panics inside `Dataset::get` or the transfer closure surface as
  `FerrotorchError::WorkerPanic` rather than silently closing the
  channel. `Drop::drop` first releases the receiver (to unblock a
  full-channel producer) THEN joins the handle — the ordering is
  critical to avoid deadlock.

- REQ-7: `pub struct MultiWorkerIter<D: Dataset>` — the `num_workers
  > 0` CrossBatch variant. `num_workers` worker threads pull
  `WorkItem`s (a `(seq, indices)` pair) from a shared bounded
  channel; each worker runs `worker_loop` to load the batch + apply
  the transfer + send back a `WorkResult`. The consumer reorders
  by sequence number via a `BinaryHeap<SeqEntry<S>>` so it sees
  output in sampler order despite out-of-order completion.

- REQ-8: Panic-safety + clean shutdown. `worker_loop` wraps each
  iteration in `catch_unwind` so a panicking `Dataset::get` or
  transfer closure surfaces as `FerrotorchError::WorkerPanic { message
  }` (extracted via `fn panic_payload_message`). `Drop` impls on
  `PrefetchIter` and `MultiWorkerIter` follow the canonical shutdown
  order: release receivers FIRST (so producers blocked on `send` see
  disconnect), THEN drop senders, THEN join handles. Workers exit
  cleanly when the work-channel sender is dropped.

- REQ-9: `pub struct CollatedIter<'a, D: Dataset>` — adapter that
  threads each `Vec<Sample>` from `BatchIter` through a collation
  closure to produce a single `Sample`. Returned by
  `DataLoader::iter_collated(epoch)`. The iter_collated method
  returns `Err(InvalidArgument)` if `with_collate` was never
  called, so misuse is observable rather than panicky. Mirrors
  upstream's automatic `collate_fn=default_collate` plumbing in
  `_BaseDataLoaderIter`.

- REQ-10: Reproducibility contract. Two `DataLoader::iter(epoch=0)`
  invocations with the same seed produce the same sequence of
  batches; different `epoch` values produce different shuffles.
  Asserted by `test_reproducible_with_same_seed_and_epoch` and
  `test_shuffle_different_epochs`. The effective seed is derived
  from `(base_seed, epoch)` via the sampler's seed-mixing — the
  loader itself doesn't manage the PRNG state, only forwards the
  epoch.

## Acceptance Criteria

- [x] AC-1: `pub struct DataLoader<D: Dataset>` with all 13 fields;
  `DataLoader::new` returns `Err` on `batch_size == 0`; all 10
  builder methods are present.
- [x] AC-2: `pub enum WorkerMode` derives `Default = IntraBatch`.
- [x] AC-3: `pub trait ToDevice` with `to_device` + `to_device_pinned`
  default; blanket `impl<T: Float> ToDevice for Tensor<T>`.
- [x] AC-4: `pub enum BatchIter` with three variants;
  `impl Iterator + ExactSizeIterator` dispatching to inner.
- [x] AC-5: `pub struct DataLoaderIter` with `pos` cursor + rayon
  par_iter on `num_workers > 0`.
- [x] AC-6: `pub struct PrefetchIter` with bounded crossbeam
  channel + `catch_unwind`-guarded producer + `Drop` releasing the
  receiver BEFORE joining the handle.
- [x] AC-7: `pub struct MultiWorkerIter` with `BinaryHeap`-backed
  reorder buffer + worker thread pool + bounded work / result
  channels.
- [x] AC-8: `fn worker_loop` wraps the per-item body in
  `catch_unwind`; `fn panic_payload_message` extracts the panic
  string; `Drop::drop` on both prefetched iters releases receivers
  first.
- [x] AC-9: `pub struct CollatedIter` + `DataLoader::iter_collated`
  returning `Err(InvalidArgument)` when no `collate_fn` is set.
- [x] AC-10: `test_reproducible_with_same_seed_and_epoch`
  asserts byte-identical batches across runs.

## Architecture

### `DataLoader` builder (REQ-1)

The fields enumerate the full upstream surface plus three Rust-
specific extensions (`worker_mode`, `pin_memory`, `transfer_fn`).
Builder methods all take `mut self` and return `Self` for
chaining:

```rust
let loader = DataLoader::new(Arc::new(ds), 32)?
    .shuffle(true)
    .seed(42)
    .num_workers(4)
    .worker_mode(WorkerMode::CrossBatch)
    .prefetch_factor(2)
    .device(Device::Cuda(0))
    .pin_memory(true);
```

`new` is the only fallible constructor — everything else is
infallible (configuration that can fail is rare). The
`Result<DataLoader>` return on `new` matches the Rust idiom of
"only fail at boundary calls"; the upstream `DataLoader.__init__`
raises `ValueError` on `num_workers < 0` etc, which we encode as
the boolean-typed `num_workers: usize` (negative impossible).

### `WorkerMode` enum (REQ-2)

The split between intra-batch parallelism (rayon work-stealing
inside one batch) and cross-batch parallelism (dedicated threads
each producing a full batch) is a Rust-side recasting of the
upstream "num_workers controls process count" — see the doc-
comment on `WorkerMode` for the rationale. Default is `IntraBatch`
because it has lower spawn cost and matches user expectations from
the upstream `num_workers > 0` behavior most closely.

### `ToDevice` trait + Tensor blanket impl (REQ-3)

The trait is two methods: `to_device(d)` and `to_device_pinned(d)`.
The pinned variant uses page-locked host memory + DMA for ~2×
faster CPU→GPU transfers; the default `to_device_pinned`
implementation forwards to `to_device` so existing impls keep
working.

```rust
impl<T: Float> ToDevice for Tensor<T> {
    fn to_device(&self, device: Device) -> FerrotorchResult<Self> {
        self.to(device)
    }
    fn to_device_pinned(&self, device: Device) -> FerrotorchResult<Self> {
        self.to_pinned(device)
    }
}
```

The `DataLoader::device(device)` builder accepts `D::Sample:
ToDevice` and installs a `transfer_fn` closure that calls
`to_device_pinned` if `pin_memory && device.is_cuda()` else
`to_device`. The boolean is threaded through the closure rather than
captured at construction so the `pin_memory(true)` builder method
works AFTER `device(...)` has been called.

### Three-variant `BatchIter` (REQ-4, REQ-5, REQ-6, REQ-7)

The dispatch in `DataLoader::iter(epoch)`:

```rust
if self.worker_mode == WorkerMode::CrossBatch && self.num_workers > 0 {
    return BatchIter::MultiWorker(MultiWorkerIter::new(...));
}
if self.prefetch_factor > 0 {
    BatchIter::Prefetch(PrefetchIter::new(...))
} else {
    BatchIter::Sync(DataLoaderIter { ... })
}
```

`DataLoaderIter` (Sync) is the simplest. The `pos` cursor advances
through `indices` by `batch_size`; `next()` returns `None` when
`pos >= indices.len()` or (if `drop_last`) when the remainder is
short. Intra-batch rayon parallelism kicks in if `num_workers > 0`.

`PrefetchIter` spawns ONE background thread that runs
`producer_loop`: load batch, apply transfer, send through the
bounded channel, repeat. Channel capacity = `prefetch_factor`. The
consumer's `next()` blocks on `rx.recv()`. Panic handling: the
spawn closure wraps `producer_loop` in `catch_unwind` and
forwards a `WorkerPanic` error before letting the thread unwind.

`MultiWorkerIter` is the most complex. `new()` plans every batch
up front (`batch_plans: Vec<Vec<usize>>` with one entry per batch),
then spawns N worker threads, each running `worker_loop`. The
dispatcher (the consumer thread itself) calls
`refill_work_queue()` to push `WorkItem { seq, indices }`s onto
the bounded work channel; workers receive, load, send back
`WorkResult { seq, batch }`. The consumer's `next()`:

1. Refills work queue.
2. Peeks the reorder heap. If `top.seq == next_yield_seq`, pop +
   return.
3. Else `result_rx.recv()` → push into heap → refill → loop.

The `BinaryHeap` is a max-heap by default; we implement `Ord` in
reverse so it pops smallest-seq first (see `impl Ord for SeqEntry`).

### Worker loop + panic safety (REQ-8)

`fn worker_loop` is the body of every cross-batch worker thread.
It loops:

```rust
loop {
    let item = work_rx.recv()?; // shutdown if disconnected
    let payload = catch_unwind(AssertUnwindSafe(|| {
        // Load batch sequentially + apply transfer.
        // Cross-batch already parallel; no intra-batch rayon.
    }));
    let result = payload.unwrap_or_else(|p| {
        Err(FerrotorchError::WorkerPanic { message: panic_payload_message(&*p) })
    });
    tx.send(WorkResult { seq, batch: result })?; // shutdown if no consumer
}
```

`fn panic_payload_message(payload: &(dyn Any + Send + 'static)) ->
String` downcasts to `&'static str` or `String`. The `'static`
bound is required for `Any::downcast_ref` to round-trip the
original payload type. Anything else (a panic with a custom
payload type) becomes the marker `"<non-string panic payload>"`.

`Drop` discipline on both prefetched iterators:

```rust
// PrefetchIter:
fn drop(&mut self) {
    self.receiver = None; // unblock producer if channel was full
    if let Some(handle) = self.handle.take() {
        let _ = handle.join();
    }
}

// MultiWorkerIter:
fn drop(&mut self) {
    self.work_tx = None;  // workers see Disconnect on recv()
    self.result_rx = None; // unblock workers blocked on send()
    for handle in self.worker_handles.drain(..) {
        let _ = handle.join();
    }
}
```

The ordering is the load-bearing invariant: dropping the receiver
BEFORE joining the handle is what prevents the deadlock where a
producer blocked on `send` to a full channel can never observe a
disconnect.

### `CollatedIter` (REQ-9)

A thin adapter that chains a collation closure onto each
`BatchIter` output. `next()` is:

```rust
let batch = self.inner.next()?;
Some(batch.and_then(|samples| (self.collate_fn)(samples)))
```

The `iter_collated(epoch)` method returns `Err(InvalidArgument)`
if `collate_fn` is `None` — making misuse observable. The
documentation directs the user to call `with_collate(...)` first.

### Reproducibility (REQ-10)

The loader does not own PRNG state; it forwards `epoch` to the
sampler (`RandomSampler::indices(epoch)` etc.). The same epoch +
same base seed produces the same index list, and the iterator
state machines are deterministic given an index list. Asserted by
`test_reproducible_with_same_seed_and_epoch`.

Caveat: the `num_workers > 0` + shuffle path is reproducible at
the BATCH level but not the WITHIN-BATCH level when rayon's work-
stealing reorders the parallel loads. Tests work around this by
unioning batches before checking content.

### Non-test production consumers

- `pub use dataloader::{...}` in `lib.rs` re-exports the seven
  public types to the crate surface; meta-crate glob propagates
  them as `ferrotorch::DataLoader` etc. — the primary consumer
  binding.
- Downstream training loops in `ferrotorch-llama`, `ferrotorch-bert`,
  `ferrotorch-whisper`, `ferrotorch-diffusion`, and the other model
  crates construct `DataLoader::new(...)` with `Arc::new(ds)`,
  builder-chain through `shuffle().num_workers().device()`, then
  iterate via `for batch in loader.iter(epoch) { ... }`.
- The `ToDevice` trait is implemented by user code in those same
  crates for custom batch types (e.g. `LlamaBatch { tokens, mask }`)
  so the loader can transfer the whole batch to GPU.

## Parity contract

`parity_ops = []`. The dataloader is plumbing; the numerical
contract is preserved by `Dataset::get` and the transfer-closure /
`Tensor::to_device`. Edge cases preserved:

- **Sampler order observance**: with a custom `Sampler`, the loader
  honours its index order even under multi-worker parallelism (via
  the reorder buffer in `MultiWorkerIter`). Asserted by
  `test_with_distributed_sampler`.
- **`drop_last` semantics**: when `drop_last=true`, the last batch
  is dropped if it would be smaller than `batch_size`. Asserted
  across the Sync / Prefetch / MultiWorker paths.
- **Worker panic surfacing**: a `Dataset::get` that panics surfaces
  as `FerrotorchError::WorkerPanic { message }` to the consumer's
  next `iter().next()`. Asserted by `worker_panic_surfaces_as_error`
  (in the test module).
- **Iterator drop cleanliness**: dropping the iterator mid-batch
  is safe — workers receive the disconnect signal and exit; no
  thread leak.
- **`size_hint` accuracy**: `BatchIter::size_hint()` returns
  `(remaining, Some(remaining))` so `ExactSizeIterator` works.
  Asserted by `test_size_hint_accurate`.
- **`with_sampler` precedence**: a custom sampler overrides the
  `shuffle` builder. Asserted by
  `test_with_sampler_overrides_shuffle`.
- **Device transfer in pipeline**: when `device(...)` is set, the
  transfer happens INSIDE the prefetch / worker pipeline so the
  consumer's wait is hidden behind the I/O overlap. Asserted by
  the `test_builder_chaining_with_device` test.
- **Pin-memory propagation**: `pin_memory(true)` is honored by the
  transfer closure's `to_device_pinned` branch on CUDA targets;
  has no effect on CPU targets. Direct test coverage gated on
  CUDA-available runners.

## Verification

Unit tests in `mod tests in dataloader.rs` (~50 tests across
groups):

- batch count: `_exact_division`, `_with_remainder`,
  `_single_element`, `_empty_dataset` (4).
- batch sizes / drop_last: `_exact`, `_with_partial_last`,
  `_all_samples_present_sequential`, `_drop_last_removes_partial_batch`,
  `_drop_last_exact_keeps_all`, `_drop_last_smaller_than_batch` (6).
- shuffle: `_produces_different_order`, `_contains_all_elements`,
  `_different_epochs` (3).
- reproducibility: `_reproducible_with_same_seed_and_epoch`,
  `_different_seeds_differ` (2).
- size_hint: `_accurate`, `_drop_last` (2).
- builder: `_zero_batch_size_returns_err`, `_chaining`,
  `_chaining_with_device` (3).
- collate_fn: `_with_collate_sum`, `_with_remainder`, `_with_drop_last`,
  `_fn_accessor`, `_iter_size_hint`, `_error_propagation`,
  `_iter_err_without_collate_fn`, `_uncollated_iter_unaffected` (8).
- num_workers: `_builder`, `_parallel_loads_all_samples`,
  `_parallel_batch_sizes`, `_parallel_drop_last`,
  `_parallel_with_shuffle`, `_zero_is_sequential` (6).
- custom sampler: `_overrides_shuffle`, `_distributed_sampler`,
  `_and_num_workers` (3).
- prefetch: `_produces_same_results_as_sync`, `_with_shuffle_same_elements`,
  `_with_drop_last`, `_empty_dataset`, `_single_element` (5).
- multi-worker (`WorkerMode::CrossBatch`): the test family covers
  the reorder buffer + worker-loop + panic-surfacing paths.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-data --lib dataloader:: 2>&1 | tail -3
```

Expected: ~50 passed.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct DataLoader<D: Dataset>` with 13 fields + `DataLoader::new` rejecting `batch_size == 0` + 10 builder methods in `dataloader.rs`, mirroring `torch/utils/data/dataloader.py:142-617`; non-test consumer: `pub use dataloader::DataLoader` in `lib.rs`, and the meta-crate glob propagates it as `ferrotorch::DataLoader`; downstream training-loop code in the model crates constructs `DataLoader::new(Arc::new(ds), batch_size)?.shuffle(true).num_workers(4)`. |
| REQ-2 | SHIPPED | impl: `pub enum WorkerMode { IntraBatch, CrossBatch }` with `#[derive(Default)]` → `IntraBatch` in `dataloader.rs`; non-test consumer: `fn DataLoader::iter in dataloader.rs` matches on `self.worker_mode == WorkerMode::CrossBatch && self.num_workers > 0` to dispatch to `MultiWorkerIter`; `pub use dataloader::WorkerMode` in `lib.rs`. |
| REQ-3 | SHIPPED | impl: `pub trait ToDevice: Sized` with `to_device + to_device_pinned (default)` + blanket `impl<T: Float> ToDevice for Tensor<T>` in `dataloader.rs`; non-test consumer: `fn DataLoader::device in dataloader.rs` is bounded by `D::Sample: ToDevice + 'static` and constructs a `transfer_fn` closure that calls `s.to_device_pinned(device)` or `s.to_device(device)`; the blanket impl makes `DataLoader<TensorDataset<f32>>` work out of the box. |
| REQ-4 | SHIPPED | impl: `pub enum BatchIter<'a, D: Dataset> { Sync, Prefetch, MultiWorker }` + `impl Iterator + ExactSizeIterator` dispatching to inner in `dataloader.rs`; non-test consumer: `fn DataLoader::iter` constructs each variant, `fn CollatedIter::next in dataloader.rs` calls `self.inner.next()` on the boxed `BatchIter`, and consumer code iterates via `for batch in loader.iter(epoch)`. |
| REQ-5 | SHIPPED | impl: `pub struct DataLoaderIter<'a, D: Dataset>` + `impl Iterator + ExactSizeIterator` with `pos` cursor + optional rayon `par_iter` in `dataloader.rs`; non-test consumer: `BatchIter::Sync(DataLoaderIter { ... })` constructed in `DataLoader::iter` when `prefetch_factor == 0`. |
| REQ-6 | SHIPPED | impl: `pub struct PrefetchIter<D: Dataset>` + `PrefetchIter::new` spawning the background thread inside `catch_unwind` + `Drop::drop` releasing the receiver before joining in `dataloader.rs`; non-test consumer: `BatchIter::Prefetch(PrefetchIter::new(...))` constructed in `DataLoader::iter` when `prefetch_factor > 0`. |
| REQ-7 | SHIPPED | impl: `pub struct MultiWorkerIter<D: Dataset>` + `BinaryHeap<SeqEntry<S>>` reorder buffer + N worker threads each running `worker_loop` + bounded crossbeam work/result channels in `dataloader.rs`; non-test consumer: `BatchIter::MultiWorker(MultiWorkerIter::new(...))` constructed in `DataLoader::iter` when `WorkerMode::CrossBatch && num_workers > 0`. |
| REQ-8 | SHIPPED | impl: `fn worker_loop in dataloader.rs` wraps the per-item body in `catch_unwind(AssertUnwindSafe(...))` and forwards `FerrotorchError::WorkerPanic { message: panic_payload_message(&*p) }` on panic; `fn panic_payload_message in dataloader.rs` does the downcast; both `Drop` impls follow the receiver-first ordering with `// SAFETY:`-style invariant comments; non-test consumer: every cross-batch worker thread in `MultiWorkerIter::new` runs through `worker_loop`, and the test `worker_panic_surfaces_as_error` (in the test module) asserts the panic→error pipeline. |
| REQ-9 | SHIPPED | impl: `pub struct CollatedIter<'a, D: Dataset>` + `DataLoader::iter_collated(epoch) -> FerrotorchResult<CollatedIter<'_, D>>` returning `Err(InvalidArgument)` when `collate_fn` is None in `dataloader.rs`; non-test consumer: `pub use dataloader::CollatedIter` in `lib.rs`, and downstream callers writing `loader.with_collate(default_collate).iter_collated(epoch)?` chain through this surface. |
| REQ-10 | SHIPPED | impl: `fn DataLoader::build_indices in dataloader.rs` forwards `epoch` to the sampler's `indices(epoch)` and the loader itself owns no PRNG state — determinism is inherited from the sampler; non-test consumer: every iterator construction path (Sync / Prefetch / MultiWorker) consumes the deterministic `indices: Vec<usize>` produced this way; `test_reproducible_with_same_seed_and_epoch` and `test_shuffle_different_epochs` in `mod tests` pin the contract. |
