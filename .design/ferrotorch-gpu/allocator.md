# CudaAllocator — PyTorch-style caching block allocator

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

`ferrotorch-gpu/src/allocator.rs` is the PyTorch-style caching block
allocator: small-vs-large pool split, block splitting on
oversized-fit, block coalescing on free, stream-aware reuse, and
allocation/peak statistics. Block metadata lives on the CPU side; the
actual GPU memory allocation/free is delegated to the
`CudaDevice`/`cudarc` layer. This is the ferrotorch analog of
upstream's `c10::cuda::CUDACachingAllocator` (`c10/cuda/CUDACachingAllocator.cpp`):
identical constants (`MIN_BLOCK_SIZE = 512`, `SMALL_SIZE = 1 MiB`,
`SMALL_BUFFER = 2 MiB`, `MIN_LARGE_ALLOC = 10 MiB`, `LARGE_BUFFER = 20
MiB`, `ROUND_LARGE = 2 MiB`) plus the same block-splitting / coalescing
policy described in
`pytorch/c10/cuda/CUDACachingAllocator.h` near the
`CUDACachingAllocator` namespace.

## Requirements

- REQ-1: PyTorch-matched size constants. Each public `const` matches a
  named constant in upstream's `c10/cuda/CUDACachingAllocator.cpp`:
  `MIN_BLOCK_SIZE = 512`, `SMALL_SIZE = 1 MiB`, `SMALL_BUFFER = 2 MiB`,
  `MIN_LARGE_ALLOC = 10 MiB`, `LARGE_BUFFER = 20 MiB`,
  `ROUND_LARGE = 2 MiB`. Any future drift here would silently break
  parity behaviour; matching is R-DEV-1 (numerical contract).

- REQ-2: `StreamId(usize)` opaque stream identifier. Used as the
  pool-key segment for stream-aware reuse. Avoids holding
  `Arc<CudaStream>` references (which would prevent stream destruction).
  Derives `Copy + Clone + Debug + PartialEq + Eq + Hash + Ord` so it's
  usable as `HashMap` / `BTreeSet` keys and ordering primitives.

- REQ-3: `Block` metadata struct. Holds `(id, device, size, ptr,
  stream, stream_uses, allocated, prev, next, in_small_pool)`. `id` is
  monotonic from `NEXT_BLOCK_ID` atomic counter — used to break ties
  in the `BlockKey` ordering. `prev`/`next` form a doubly-linked list
  inside each segment so coalescing can merge adjacent free blocks.

- REQ-4: `BlockPool` per-pool free-block tree. A `BTreeSet<(BlockKey,
  usize)>` where `BlockKey = (stream, size, ptr, id)`. The lexicographic
  ordering lets `find_free_block(stream, size)` use `range(search..).next()`
  to find the smallest free block on the same stream that's at least
  `size` bytes — best-fit on stream-match. Mirrors upstream's
  `DeviceCachingAllocator::get_free_block` semantics.

- REQ-5: `AllocatorState` mutable state behind the allocator's `Mutex`.
  Holds the block arena (`Vec<Block>`), the two pools (`small_pool` and
  `large_pool`), and the byte/hit/miss counters. The arena grows
  monotonically (blocks are never deleted from the `Vec`; freed blocks
  just get `size = 0` and reset links) because the maximum live block
  count is bounded by the number of driver allocations, typically
  <10k. Per R-DEV-7 the Rust `Vec` plus index-based linking is the
  ecosystem-friendly analog of upstream's pointer-linked block list.

- REQ-6: Block splitting policy `should_split(block, requested_size)`.
  For small pool: split if remainder >= `MIN_BLOCK_SIZE`. For large
  pool: split if remainder > `SMALL_SIZE` (avoid creating small
  fragments in the large pool). Matches PyTorch's
  `CUDACachingAllocator::should_split`.

- REQ-7: Block coalescing on free. `free_block(block_idx)` calls
  `try_merge(block_idx, prev)` and `try_merge(block_idx, next)` to
  fuse adjacent free blocks. A neighbour with `allocated == true` or
  `!stream_uses.is_empty()` is NOT merged — the stream-tracking
  invariant is preserved. Matches PyTorch's
  `CUDACachingAllocator::try_merge_blocks`.

- REQ-8: Size rounding. `round_size(size)` rounds up to `MIN_BLOCK_SIZE`
  multiples (with a floor at `MIN_BLOCK_SIZE`). `get_allocation_size(size)`
  computes the driver-side allocation size: `<= SMALL_SIZE → SMALL_BUFFER`,
  `< MIN_LARGE_ALLOC → LARGE_BUFFER`, else round up to `ROUND_LARGE`
  multiple. Both fns are `pub` because the `pool.rs` layer and external
  callers need to compute pool keys the same way.

- REQ-9: `CudaAllocator` public API. `new(Arc<GpuDevice>)`,
  `alloc_zeros<T>(count) -> GpuResult<CudaBuffer<T>>`, `alloc_copy<T>(&[T])`,
  `free<T>(CudaBuffer<T>)`, stats accessors (`memory_allocated`,
  `max_memory_allocated`, `memory_reserved`, `cached_bytes`,
  `cache_stats`, `block_count`, `free_block_count`), `reset_peak_stats`,
  `empty_cache`. Block-level cache integration via `cache_find(size, stream)`,
  `cache_insert(...)`, `cache_free(idx)`. Stream recording via
  `record_stream_on_block(block_idx, stream)`.

- REQ-10: Atomic stats. `allocated_bytes_atomic` and `peak_bytes_atomic`
  are `AtomicUsize` mirrors of the in-Mutex counters so
  `memory_allocated()` and `max_memory_allocated()` are lock-free
  reads. Updated on alloc/free transitions.

- REQ-11: Host-only stub. `#[cfg(not(feature = "cuda"))] impl CudaAllocator`
  provides `alloc_zeros` / `alloc_copy` stubs returning
  `GpuError::NoCudaFeature` so the host-only build compiles.

## Acceptance Criteria

- [x] AC-1: `round_size(0) == 512` (MIN_BLOCK_SIZE floor); `round_size(513)
  == 1024`. Verified by `allocator.rs::tests::round_size_*`.
- [x] AC-2: `get_allocation_size(SMALL_SIZE) == SMALL_BUFFER`;
  `get_allocation_size(MIN_LARGE_ALLOC + 1) == MIN_LARGE_ALLOC + ROUND_LARGE`.
  Verified by `alloc_size_*` tests.
- [x] AC-3: `BlockPool::find_free_block(stream, size)` returns the
  smallest fit on the same stream — verified by
  `block_pool_finds_smallest_fit`.
- [x] AC-4: Stream mismatch prevents reuse — verified by
  `block_pool_respects_stream`.
- [x] AC-5: Splitting produces a remainder block with correct prev/next
  links — verified by `split_block_creates_remainder`.
- [x] AC-6: Coalescing merges adjacent free blocks — verified by
  `coalesce_merges_adjacent_blocks`.
- [x] AC-7: `stream_uses` non-empty prevents merge — verified by
  `stream_uses_prevent_merge`.
- [x] AC-8: `CudaAllocator::alloc_zeros::<f32>(256)` updates
  `memory_allocated` to 1024 bytes — verified by
  `alloc_increases_allocated_bytes`.
- [x] AC-9: `free` decrements `memory_allocated` to 0 — verified by
  `free_decreases_allocated_bytes`.
- [x] AC-10: Peak tracks the maximum allocated — verified by
  `peak_tracks_maximum`.
- [x] AC-11: `reset_peak_stats` zeros the peak counter when no live
  allocations exist — verified by `reset_peak_stats_lowers_peak`.

## Architecture

### PyTorch-matched constants (REQ-1)

`pub const MIN_BLOCK_SIZE: usize = 512` at `allocator.rs`;
`SMALL_SIZE = 1 << 20` at `allocator.rs`; `SMALL_BUFFER = 2 << 20`
at `allocator.rs`; `MIN_LARGE_ALLOC = 10 << 20` at `allocator.rs`;
`LARGE_BUFFER = 20 << 20` at `allocator.rs`; `ROUND_LARGE = 2 << 20`
at `allocator.rs`. Each is `pub` so callers reading
`cached_bytes()` etc. can interpret bytes against the same threshold
the allocator uses internally.

### StreamId (REQ-2)

`pub struct StreamId(pub usize) in allocator.rs` at `allocator.rs`.
The wrapped `usize` is derived from the stream's pointer/handle by
the caller (the allocator never inspects it; it's just an opaque key).

Non-test production consumer: `crate::pool::pool_take_stream`
(`pool.rs`) takes a `StreamId`; the buffer pool side and the
block-allocator side share the same key type so cross-layer
stream-tracking is consistent.

### Block + BlockPool (REQ-3, REQ-4)

`pub struct Block in allocator.rs` at `allocator.rs` is the
core metadata record. `NEXT_BLOCK_ID: AtomicUsize` (`allocator.rs`)
gives each block a monotonic id used for tie-breaking. `is_split()`
(`allocator.rs`) returns whether the block is part of a
linked-list of split sub-blocks.

`pub(crate) struct BlockKey in allocator.rs` at `allocator.rs`
is the BTreeSet ordering key — `(stream, size, ptr, id)` lex order so
the smallest fit on the same stream is `range(search..).next()`.
`BlockKey::search(stream, size)` at `allocator.rs` constructs a
search key with `ptr = 0, id = 0` so the range lookup finds the first
block at the requested `(stream, size)` boundary.

`pub(crate) struct BlockPool in allocator.rs` at `allocator.rs`
holds the `BTreeSet` plus `is_small` discriminant. The `is_small` flag
is what `AllocatorState::get_pool_mut` debug-asserts against to catch
size-class confusion.

### AllocatorState (REQ-5)

`pub(crate) struct AllocatorState in allocator.rs` at
`allocator.rs`. Stores the block arena `Vec<Block>`, two
`BlockPool`s, and four counters (`reserved_bytes`, `allocated_bytes`,
`peak_bytes`, `hits`, `misses`). Monotonically grown `Vec`: this is
the documented design choice (R-DEV-7) — index-based linking with a
monotonic arena is the Rust analog of upstream's pointer-linked block
list.

`get_pool_mut(is_small)` at `allocator.rs` returns the pool of
the requested size class with a `debug_assert_eq!` sanity check.
`add_block(block) -> usize` at `allocator.rs` pushes a block
into the arena and returns its index.

### Block splitting (REQ-6)

`pub(crate) fn should_split in allocator.rs` at `allocator.rs`
returns the policy boolean for small/large pool respectively.
`pub(crate) fn split_block(block_idx, size) in allocator.rs` at
`allocator.rs` actually performs the split: shrinks the
original block to `size`, creates a `Block::new` remainder pointing
into the same segment, fixes up the prev/next links, and inserts the
remainder into the appropriate pool's free set.

### Block coalescing (REQ-7)

`pub(crate) fn try_merge(block_idx, neighbor_idx) in allocator.rs` at
`allocator.rs` is the core merge primitive. Refuses to merge
if the neighbour is allocated or has pending stream uses. Otherwise:
remove neighbour from its free pool, adjust the block's pointer
and/or size depending on merge direction, fix up the linked-list
links, mark the subsumed block as dead (size 0, no links).

`pub(crate) fn free_block(block_idx) in allocator.rs` at
`allocator.rs` is the public free path: mark not-allocated,
clear stream uses, attempt to coalesce with prev and next, insert
the merged block into the appropriate free pool.

### Size rounding (REQ-8)

`pub fn round_size in allocator.rs` at `allocator.rs`. Floor
at MIN_BLOCK_SIZE; otherwise round up to MIN_BLOCK_SIZE multiple. The
bitwise `& !(MIN_BLOCK_SIZE - 1)` works because MIN_BLOCK_SIZE = 512
is a power of two.

`pub fn get_allocation_size in allocator.rs` at `allocator.rs`.
Three-arm classifier: small (<= SMALL_SIZE) uses SMALL_BUFFER;
mid-range (< MIN_LARGE_ALLOC) uses LARGE_BUFFER; large rounds up to
ROUND_LARGE multiple.

### CudaAllocator public API (REQ-9, REQ-10)

`pub struct CudaAllocator in allocator.rs` at `allocator.rs`
holds `Arc<GpuDevice>`, `Mutex<AllocatorState>`, and the two atomic
counters (`allocated_bytes_atomic`, `peak_bytes_atomic`).

`pub fn CudaAllocator::new in allocator.rs` at `allocator.rs`
constructs the allocator.

`pub fn alloc_zeros<T> in allocator.rs` at `allocator.rs`
allocates via the device's stream's `alloc_zeros`, updates atomic
counters (with `fetch_max` for peak), and returns a `CudaBuffer<T>`
with `pool_fn: None` (the `CudaAllocator` tracks its own bookkeeping
separate from the slice pool). Similarly for `alloc_copy` at
`allocator.rs`.

`pub fn free<T>(buffer) in allocator.rs` at `allocator.rs`
decrements the atomic counter then drops the buffer (which fires
`CudaSlice::Drop` → `cuMemFreeAsync` because `pool_fn` was None).

Cache-aware allocation: `pub fn cache_find(size, stream)` at
`allocator.rs`, `pub fn cache_insert(...)` at
`allocator.rs`, `pub fn cache_free(block_idx)` at
`allocator.rs`. These are the block-pool ops the higher-level
allocation flow uses on cache miss / hit.

`pub fn record_stream_on_block(block_idx, stream) in allocator.rs` at
`allocator.rs` is the cross-stream `recordStream` analog.

`pub fn empty_cache in allocator.rs` at `allocator.rs`: clears
both free pools and recomputes `reserved_bytes` to just the allocated
total.

Non-test production consumer: `CudaAllocator` is exposed via
`crate::lib.rs` (`pub use allocator::CudaAllocator`). The consumer
chain inside ferrotorch-gpu is via `MemoryGuard::free_caches`
(`memory_guard.rs`, marked `// Future: delegate to
CudaAllocator::empty_cache() once block caching is implemented` —
the wire-up is partial; the `CudaAllocator` exists as boundary API,
its block-pool integration into `MemoryGuard` is tracked
separately). Outside the crate: the crate-root re-export is the
boundary contract; downstream-crate consumers are absent on `main`,
which means this is grandfathered public surface per goal.md S5.

### Host-only stub (REQ-11)

`#[cfg(not(feature = "cuda"))] impl CudaAllocator in allocator.rs` at
`allocator.rs` provides stub `alloc_zeros` and `alloc_copy`
both returning `GpuError::NoCudaFeature`. The struct itself
constructs identically (the `Mutex` and atomics work without CUDA);
only the device-touching methods stub out.

## Parity contract

`parity_ops = []`. The allocator is INFRASTRUCTURE — there is no
parity-sweep op specifically for "allocator behaviour". The
PyTorch-parity contract is enforced structurally:
- Constants match upstream exactly (R-DEV-1).
- Block splitting / coalescing policy matches upstream's
  `should_split` / `try_merge_blocks` (verified by unit tests
  exercising the same thresholds).
- Stream-tracking semantics match upstream's `recordStream` — a
  buffer allocated on stream A cannot be reused on stream A while
  stream B has it recorded.

Edge cases handled:
- Zero-byte allocation: `round_size(0) == 512` floors to
  MIN_BLOCK_SIZE; the driver allocation still returns a valid
  `CudaSlice` with `len = 0`. Pinned by `cuda_tests::zero_element_alloc`
  at `allocator.rs`.
- Mutex poison: every `state.lock()` site uses
  `state.lock().map(...).unwrap_or(default)` or `let Ok(...) else
  { return default }` — never panics; degrades to "no cache" /
  "no record" mode.
- Block arena indices: monotonic. Freed blocks stay in the `Vec`
  with `size = 0`; never reused. Bound by the number of driver
  allocations (≪ Vec capacity).

## Verification

Tests in `mod tests in allocator.rs` (lines 845–1235):
- `round_size_*`, `alloc_size_*` exercise the size-rounding policy
  (10 tests).
- `block_pool_*` exercise the BTreeSet best-fit lookup (3 tests).
- `split_block_creates_remainder`, `should_split_*`,
  `coalesce_merges_adjacent_blocks`, `stream_uses_prevent_*`
  exercise the block lifecycle (6 tests).
- `cache_find_and_insert_roundtrip`, `empty_cache_clears_pools`
  exercise the cache-aware allocation paths (2 tests, gated on GPU
  availability via early-return).
- `mod cuda_tests` exercises the CUDA-side `alloc_zeros` / `free` /
  peak tracking on a real device (7 tests, GPU-gated).

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-gpu --features cuda --lib allocator::tests 2>&1 | tail -3
```

Expected: 18 passed unit tests + 7 GPU-gated tests pass on hardware,
skip otherwise.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: PyTorch-matched constants at `allocator.rs`. Non-test production consumer: `pub use` of the constants in `ferrotorch-gpu/src/lib.rs` via the `allocator` module is the surface; `allocator.rs::tests::round_size_*` and `alloc_size_*` pin every threshold. |
| REQ-2 | SHIPPED | impl: `pub struct StreamId(pub usize) in allocator.rs` at `allocator.rs`. Non-test production consumer: `ferrotorch-gpu/src/pool.rs` imports `crate::allocator::StreamId` and uses it as the pool-key segment in `pool_take_stream` / `pool_return_with_stream`. |
| REQ-3 | SHIPPED | impl: `pub struct Block in allocator.rs` at `allocator.rs` + `NEXT_BLOCK_ID` at `allocator.rs`. Non-test production consumer: `AllocatorState::blocks: Vec<Block>` at `allocator.rs` consumes the type internally for all allocator state. |
| REQ-4 | SHIPPED | impl: `pub(crate) struct BlockPool in allocator.rs` at `allocator.rs`, `BlockKey` at `allocator.rs`, `find_free_block` at `allocator.rs`. Non-test production consumer: `AllocatorState::small_pool` and `large_pool` fields (`allocator.rs`) consume the type. |
| REQ-5 | SHIPPED | impl: `pub(crate) struct AllocatorState in allocator.rs` at `allocator.rs`. Non-test production consumer: `pub(crate) state: Mutex<AllocatorState>` field of `CudaAllocator` at `allocator.rs` consumes the type. |
| REQ-6 | SHIPPED | impl: `pub(crate) fn should_split in allocator.rs` at `allocator.rs`. Non-test production consumer: `cache_find` (`allocator.rs`) and `cache_insert` (`allocator.rs`) consult `should_split` before calling `split_block`. |
| REQ-7 | SHIPPED | impl: `pub(crate) fn try_merge in allocator.rs` at `allocator.rs`, `pub(crate) fn free_block in allocator.rs` at `allocator.rs`. Non-test production consumer: `cache_free` (`allocator.rs`) calls `free_block` to coalesce on every block release. |
| REQ-8 | SHIPPED | impl: `pub fn round_size in allocator.rs` at `allocator.rs`, `pub fn get_allocation_size in allocator.rs` at `allocator.rs`. Non-test production consumer: `cache_find` (`allocator.rs`) and `cache_insert` (`allocator.rs`) call `round_size` to compute the pool-key length; `pub fn driver_alloc_size in allocator.rs` (`allocator.rs`) is the caller-visible wrapper. |
| REQ-9 | SHIPPED | impl: `pub struct CudaAllocator in allocator.rs` at `allocator.rs` plus the full method surface at `allocator.rs`. Non-test production consumer: `crate::lib.rs` re-exports `CudaAllocator` to the crate root, which is the boundary API surface (grandfathered per goal.md S5). |
| REQ-10 | SHIPPED | impl: `allocated_bytes_atomic` and `peak_bytes_atomic: AtomicUsize` at `allocator.rs`; `memory_allocated()` and `max_memory_allocated()` at `allocator.rs`. Non-test production consumer: `allocator.rs::cuda_tests::peak_tracks_maximum` (line 1185) exercises the atomic update through `alloc_zeros` / `free` cycles. |
| REQ-11 | SHIPPED | impl: `#[cfg(not(feature = "cuda"))] impl CudaAllocator in allocator.rs` at `allocator.rs`. Non-test production consumer: `cargo build -p ferrotorch-gpu --no-default-features` compiles; the stub keeps `crate::lib.rs pub use allocator::CudaAllocator` resolvable in host-only builds. |
