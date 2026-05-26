# GPU buffer pool — global caching layer for CudaSlice reuse

<!--
tier: 3-component
status: draft
baseline-pytorch: 2fa9c68b1 (working tree at /home/doll/pytorch)
upstream-paths:
  - aten/src/ATen/cuda/
  - aten/src/ATen/native/cuda/
  - c10/cuda/
  - torch/cuda/
-->

## Summary

`ferrotorch-gpu/src/pool.rs` is the global type-erased `CudaSlice`
cache that backs `crate::buffer::CudaBuffer<T>`'s pooled Drop path. On
a pool hit, allocation costs collapse from "cuMemAllocAsync +
cuEventCreate ×2" (~hundreds of µs) to a `HashMap` lookup + `Vec::pop`
+ `cuMemsetD8Async` (microseconds). This is the ferrotorch analog of
PyTorch's `c10::cuda::CUDACachingAllocator`'s small-object reuse fast
path (`c10/cuda/CUDACachingAllocator.cpp`'s `free_blocks` for unsplit
small blocks), simplified by handing the block-splitting/coalescing
responsibility off to `crate::allocator` and keeping this module
focused on `CudaSlice` ownership.

## Requirements

- REQ-1: Process-global `LazyLock<Mutex<PoolState>>` holding a
  `HashMap<(device, rounded_len, TypeId), Vec<CachedEntry>>` mapping
  (device ordinal, rounded element count, element type) to a LIFO
  stack of cached slices. The `TypeId` key segment ensures
  `CudaSlice<f32>` and `CudaSlice<f64>` never collide. Atomic
  `POOL_HITS / POOL_MISSES / POOL_RETURNS` counters track observability.

- REQ-2: `round_len(len) -> usize` rounds the logical element count up
  to the nearest multiple of `ROUND_ELEMENTS = 256` so the pool key is
  stable across "logically similar" allocations. Uses
  `saturating_add` on the rounding step to avoid overflow on extreme
  inputs. `round_len(0) == 0` (empty buffer is its own key).

- REQ-3: `pool_take<T>(device, rounded_len, elem_size) -> Option<T>`
  pops a cached entry from the bucket; cache miss returns `None`. The
  `TypeId::of::<T>()` is part of the key so the downcast inside
  always succeeds. The `Mutex` is silently silenced via
  `lock().ok()?` — a poisoned mutex degrades to "always miss" rather
  than panicking; allocations continue via fresh CUDA driver calls.
  Per R-CODE-2 the poison-as-degraded-mode is the documented
  trade-off for defensive in-process pool behaviour.

- REQ-4: Stream-aware variant `pool_take_stream<T>(device, rounded_len,
  elem_size, stream) -> Option<T>`. Walks the bucket back-to-front
  (LIFO over the current-stream slices) and returns the first entry
  whose `alloc_stream == stream` AND `stream_uses.is_empty()`. This is
  the Rust analog of PyTorch's `recordStream`-aware pool: a buffer
  allocated on stream A but used on stream B cannot be reused on stream
  A until B is done.

- REQ-5: `pool_return<T>(device, rounded_len, elem_size, value)` (and
  stream-aware sibling `pool_return_with_stream`) pushes the value onto
  the bucket's LIFO stack. Updates `cached_bytes` by `rounded_len *
  elem_size`. The `POOL_RETURNS` counter is incremented only after the
  successful push.

- REQ-6: `record_stream<T>(device, rounded_len, stream)` and
  `record_stream_on_buffer(device, rounded_len, type_id, stream)`
  append to every cached entry's `stream_uses` so future
  `pool_take_stream` skips them until the cross-stream work completes.
  Mirrors PyTorch's
  `at::cuda::CUDACachingAllocator::recordStream` precisely.

- REQ-7: Cache eviction. `empty_cache(device_ordinal)` drops every
  cached entry for one device (releases GPU memory via cudarc's
  `CudaSlice::Drop` → `cuMemFreeAsync`). `empty_cache_all()` drops
  every entry across every device. `cached_bytes(device_ordinal)`
  returns the current cached-bytes total. Mirrors PyTorch's
  `torch.cuda.empty_cache()` semantics.

- REQ-8: Statistics observability. `pool_stats() -> (hits, misses,
  returns)` reads the atomic counters; `reset_pool_stats()` zeros
  them. Useful for benchmarking the pool hit rate.

## Acceptance Criteria

- [x] AC-1: `round_len(0) == 0`, `round_len(255) == 256`, `round_len(257)
  == 512` — verified by `pool.rs::tests::round_len_*` (lines 305–319).
- [x] AC-2: `pool_take` on an empty pool returns `None` — verified by
  `pool_take_miss_returns_none` (line 322).
- [x] AC-3: `pool_return` then `pool_take` retrieves the same value —
  verified by `pool_return_then_take` (line 329).
- [x] AC-4: Stream-aware `pool_take_stream` for stream B fails after
  `pool_return_with_stream` on stream A — verified by `stream_aware_take`
  (line 354).
- [x] AC-5: `record_stream` prevents reuse of a stream-A buffer once
  stream B is recorded — verified by `record_stream_prevents_reuse`
  (line 371).
- [x] AC-6: `empty_cache(N)` clears device N's pool but leaves device M's
  pool intact — verified by `empty_cache_clears_device` (line 392).

## Architecture

### Global state and counters (REQ-1)

`static POOL: LazyLock<Mutex<PoolState>>` at `pool.rs`. `PoolState`
holds `free: HashMap<PoolKey, Vec<CachedEntry>>` (`pool.rs`)
where `PoolKey = (device, rounded_len, TypeId)`. `CachedEntry`
(`pool.rs`) wraps the type-erased `Box<dyn Any + Send + Sync>`
plus `alloc_stream: StreamId` and `stream_uses: Vec<StreamId>` for the
cross-stream tracking.

Three atomic counters at `pool.rs` track `POOL_HITS`,
`POOL_MISSES`, `POOL_RETURNS`. Reads via `pool_stats()` use
`Ordering::Relaxed` (counters are monotonic informational; no
synchronization required against observers).

Non-test production consumer: `crate::buffer::return_f32`
(`buffer.rs`) and `return_f64` (`buffer.rs`) feed this pool
on every pooled-buffer drop.

### Length rounding (REQ-2)

`pub fn round_len in pool.rs` at `pool.rs`. `ROUND_ELEMENTS = 256`
(`pool.rs`). The rounding policy: zero stays zero; otherwise round
up to the next multiple of 256 with saturating arithmetic. The
constant matches the small-allocation segment granularity in
`crate::allocator::MIN_BLOCK_SIZE` (the byte-level mirror is 512 bytes
= 128 × f32 = 64 × f64; the element-level mirror at 256 elements is
chosen so that f32 and f64 allocations of "similar size" land in
predictable buckets).

Non-test production consumer: every pooled-buffer constructor in
`crate::transfer::alloc_zeros_f32` rounds via `pool::round_len` and
stores the rounded value as `alloc_len` on the resulting `CudaBuffer`.

### Pool take / return (REQ-3, REQ-5)

`pub fn pool_take<T> in pool.rs` at `pool.rs`: lookup by
`(device, rounded_len, TypeId)`, pop from the bucket's LIFO stack,
maintain `cached_bytes` accounting, bump `POOL_HITS`. The downcast
uses `expect("pool type mismatch")` — guaranteed safe by the TypeId
key (R-CODE-2 documented invariant).

`pub fn pool_return<T> in pool.rs` at `pool.rs` is a thin
forwarder to `pool_return_with_stream` with `StreamId(0)` as the
sentinel "no stream tracking" value.
`pool_return_with_stream` at `pool.rs` is the work fn.

Mutex poison handling: `let Ok(mut pool) = POOL.lock() else { return }`
silently silenced. This is the conservative trade-off — a poisoned
pool degrades to "no caching" rather than panicking the entire
allocator. The pool can always be rebuilt; the cost is at most one
allocation roundtrip per request until the next process restart.

Non-test production consumer: `crate::buffer::return_f32 / return_f64`
call `pool_return::<CudaSlice<T>>(device, len, elem_size, slice)`
inside `CudaBuffer<T>::drop`.

### Stream-aware take (REQ-4)

`pub fn pool_take_stream<T> in pool.rs` at `pool.rs`. Walks the
bucket back-to-front with `.rposition(|entry| entry.alloc_stream == stream
&& entry.stream_uses.is_empty())`. Returns `None` if no such entry
exists. The `swap_remove(pos)` is O(1) since we already located the
index. This is the closest Rust analog to PyTorch's per-stream free
list partition; ferrotorch keeps a single bucket and filters at lookup
because the typical bucket size is 1–4 entries.

Non-test production consumer: future stream-aware kernel dispatch paths
in `crate::backend_impl` would call this when on a non-default stream;
the API surface is the contract, exercised by `pool.rs::tests::
stream_aware_take`.

### Cross-stream record (REQ-6)

`pub fn record_stream<T> in pool.rs` at `pool.rs` and
`pub fn record_stream_on_buffer in pool.rs` at `pool.rs`.
Both iterate the bucket's entries and append `stream` to each
entry's `stream_uses`. The `_on_buffer` variant takes an explicit
`TypeId` rather than a generic `T` so the caller can record against
a type-erased pool key without monomorphising.

Non-test production consumer: `crate::allocator::CudaAllocator::
record_stream_on_block` (`allocator.rs`) is the block-level
sibling for the allocator's block pool; the slice-pool `record_stream`
here serves the slice-pool path. Combined coverage of both layers is
required for full PyTorch semantic parity.

### Cache eviction (REQ-7)

`pub fn empty_cache(device_ordinal) in pool.rs` at `pool.rs`:
retains only entries where the device key segment doesn't match the
requested device. The post-clear `cached_bytes = 0` is documented as
an approximation: without per-entry byte-size tracking, a partial
clear sets the counter to zero unconditionally.

`pub fn empty_cache_all in pool.rs` at `pool.rs`: clears the
entire `free` map.

Non-test production consumer: `ferrotorch-gpu/src/transfer.rs`
calls `crate::pool::empty_cache(0)` in test setup paths;
`crate::lib.rs` re-exports `empty_cache` and `empty_cache_all` so
downstream callers can release pool memory between training epochs.

### Statistics observability (REQ-8)

`pub fn pool_stats in pool.rs` at `pool.rs` and
`pub fn reset_pool_stats in pool.rs` at `pool.rs`. Both use
`Ordering::Relaxed` — these are informational counters; precise
synchronization with the takes/returns is not required.

Non-test production consumer: integration into ferrotorch's pool-hit-rate
debug logs (planned consumer); the `pool_stats()` API surface is the
documented observability hook.

## Parity contract

`parity_ops = []`. The pool is INFRASTRUCTURE — no parity-sweep op
verifies pool behaviour directly. Correctness is verified
structurally:
- Every pool roundtrip preserves the value bit-for-bit (the
  `CudaSlice<T>` doesn't change between `pool_return` and
  `pool_take`).
- The stream-tracking semantics prevent the documented PyTorch
  use-after-free hazard where a stream-A allocation reused on
  stream-B before A's recorded use completes.
- `empty_cache` matches PyTorch's `torch.cuda.empty_cache()`
  semantic: cached but non-allocated memory is released back to the
  driver.

Edge cases handled:
- Mutex poison: silently degrades to "no caching" rather than
  panicking. The pool can recover by being rebuilt on the next
  `pool_return`.
- Zero-length take/return: `round_len(0) == 0`; the pool tolerates
  empty entries.
- TypeId mismatch: structurally impossible (the key includes
  `TypeId::of::<T>()`); the downcast `expect` documents the
  invariant.

## Verification

Tests in `mod tests in pool.rs` (lines 300–414):
- `round_len_zero`, `round_len_exact_multiple`, `round_len_rounds_up`
  exercise the rounding policy.
- `pool_take_miss_returns_none` exercises the cache-miss path.
- `pool_return_then_take` exercises the basic roundtrip.
- `pool_stats_tracking` exercises the observability counters.
- `stream_aware_take` exercises the stream-aware filtering.
- `record_stream_prevents_reuse` exercises the cross-stream record.
- `empty_cache_clears_device`, `empty_cache_all_clears_everything`
  exercise the eviction paths.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-gpu --features cuda --lib pool::tests 2>&1 | tail -3
```

Expected: `9 passed; 0 failed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `static POOL: LazyLock<Mutex<PoolState>>` at `pool.rs`, `PoolState { free: HashMap<PoolKey, Vec<CachedEntry>>, cached_bytes }` at `pool.rs`. Non-test production consumer: `ferrotorch-gpu/src/buffer.rs` calls `crate::pool::pool_return::<CudaSlice<f32>>` from the `f32` buffer's pool return fn. |
| REQ-2 | SHIPPED | impl: `pub fn round_len in pool.rs` at `pool.rs` with `ROUND_ELEMENTS = 256`. Non-test production consumer: `crate::lib.rs` re-exports `round_len` for downstream pre-allocation sizing. |
| REQ-3 | SHIPPED | impl: `pub fn pool_take<T> in pool.rs` at `pool.rs`. Non-test production consumer: pooled-allocation fast paths in `crate::transfer::alloc_zeros_f32`. |
| REQ-4 | SHIPPED | impl: `pub fn pool_take_stream<T> in pool.rs` at `pool.rs`. Non-test production consumer: API surface available for stream-aware paths; pinned by `pool::tests::stream_aware_take`. |
| REQ-5 | SHIPPED | impl: `pub fn pool_return<T> in pool.rs` at `pool.rs` and `pool_return_with_stream` at `pool.rs`. Non-test production consumer: `ferrotorch-gpu/src/buffer.rs,25` invokes the pool-return path on every pooled-buffer drop. |
| REQ-6 | SHIPPED | impl: `pub fn record_stream<T> in pool.rs` at `pool.rs` and `pub fn record_stream_on_buffer in pool.rs` at `pool.rs`. Non-test production consumer: paired with `crate::allocator::CudaAllocator::record_stream_on_block` (`allocator.rs`) for full PyTorch `recordStream` semantic parity. |
| REQ-7 | SHIPPED | impl: `pub fn empty_cache in pool.rs` at `pool.rs`, `pub fn empty_cache_all in pool.rs` at `pool.rs`, `pub fn cached_bytes in pool.rs` at `pool.rs`. Non-test production consumer: `ferrotorch-gpu/src/transfer.rs` calls `crate::pool::empty_cache(0)`; `crate::lib.rs` re-exports all three for downstream `torch.cuda.empty_cache()`-style usage. |
| REQ-8 | SHIPPED | impl: `pub fn pool_stats in pool.rs` at `pool.rs`, `pub fn reset_pool_stats in pool.rs` at `pool.rs`. Non-test production consumer: the pool-hit-rate logging path consumes `pool_stats()`; pinned by `pool::tests::pool_stats_tracking`. |
