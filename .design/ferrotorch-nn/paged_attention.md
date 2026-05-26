# ferrotorch-nn — `paged_attention` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/nn/
  - aten/src/ATen/native/
-->

## Summary

`ferrotorch-nn/src/paged_attention.rs` implements *PagedAttention*
(Kwon et al., 2023 — *Efficient Memory Management for Large Language
Model Serving with PagedAttention*, SOSP 2023) — a memory-efficient
KV-cache layout for LLM serving. Manages the cache as fixed-size
*pages* drawn from a shared pool, so memory scales with the total
number of cached tokens across all concurrent sequences rather than
the worst-case per-sequence allocation.

No direct upstream `torch.nn` counterpart exists in mainline PyTorch
2.x — the canonical reference is the vLLM project and the SOSP 2023
paper. ferrotorch mirrors the data-structure design (pages + pool +
sequence-level cache + manager) but exposes it as a `pub mod
paged_attention` (not a `Module<T>` impl) because PagedAttention is
a serving-time abstraction, not a forward-pass building block.

## Requirements

- REQ-1: `pub struct KVPage<T: Float>` — a fixed-size page holding
  key/value data for up to `page_size` tokens, stored row-major
  `[page_size, num_heads, head_dim]`. Tracks `len: usize`
  (occupied tokens, `0..=page_size`). Allocation cost is paid at
  pool construction; the page itself is just an `&mut` view onto
  pre-allocated storage.

- REQ-2: `pub struct PagePool<T: Float>` — pre-allocates `num_pages`
  `KVPage<T>` instances at construction and manages a free-list of
  unused page IDs. `alloc_page() -> Option<usize>` returns a page
  ID in O(1) (or `None` when exhausted); `free_page(id)` returns a
  page to the pool in O(1) (clears its `len` to 0). Tracks
  `num_free()`, `num_used()`, `num_total()`.

- REQ-3: `pub struct PagedKVCache<T: Float>` — a single sequence's
  view onto pages held in a `PagePool`. Holds `page_ids: Vec<usize>`
  (ordered) plus `total_tokens`. The cache does NOT own the pages;
  it indexes into the pool by ID.

- REQ-4: `PagedKVCache::append(pool, key, value)` — append new
  tokens to the cache. Fills the last page first (using its
  remaining slots) then allocates fresh pages from the pool as
  needed. Errors when the pool is exhausted mid-append. Validates
  that `key.len() == value.len()` and is divisible by
  `num_heads * head_dim`.

- REQ-5: `PagedKVCache::get_keys(pool)` / `get_values(pool)` —
  gather all cached data into a contiguous tensor of shape
  `[total_tokens, num_heads, head_dim]`. Walks `page_ids` in
  order, extending a `Vec<T>` with each page's occupied slice.
  For empty caches, returns a `[0, num_heads, head_dim]` tensor.

- REQ-6: `PagedKVCache::free_all(pool)` — releases every page held
  by the cache back to the pool. Used at sequence completion.

- REQ-7: `pub struct PagedAttentionManager<T: Float>` — multi-sequence
  manager that owns the `PagePool` plus `Vec<Option<PagedKVCache<T>>>`
  slot vector. `add_sequence() -> usize` returns a sequence ID,
  reusing a previously-removed slot when available. `append_kv` /
  `get_kv` / `remove_sequence` operate by sequence ID and return
  `FerrotorchResult<_>` for invalid IDs or pool exhaustion.

- REQ-8: O(1) allocation / deallocation — `alloc_page` / `free_page`
  are constant-time. The append path fills the last page before
  allocating a new one, so an `n`-token append costs O(n /
  page_size) page allocations.

- REQ-9: `Default for PagedKVCache<T>` and `Debug for *` — derived
  via `derive(Debug, Clone)` for diagnostic logging. Cache state can
  be dumped for inspection in serving telemetry.

## Acceptance Criteria

- [x] AC-1: `PagedAttentionManager::new(64, 256, 8, 64)` constructs.
- [x] AC-2: `add_sequence` returns sequential IDs (`0, 1, 2, ...`).
- [x] AC-3: `append_kv(seq, key, value)` followed by `get_kv(seq)`
  returns tensors of shape `[num_new_tokens, 8, 64]`.
- [x] AC-4: Removing a sequence frees its pages
  (`pool.num_free()` increases by the page count).
- [x] AC-5: Pool exhaustion (more tokens than `num_pages *
  page_size`) returns an error rather than silently truncating.
- [x] AC-6: `append_kv` with `key.len() != value.len()` errors.
- [x] AC-7: `append_kv` with a length not divisible by
  `num_heads * head_dim` errors.

## Architecture

### KVPage (REQ-1)

`pub struct KVPage<T: Float>` at
`pub struct KVPage in paged_attention.rs` holds two `Vec<T>` of
capacity `page_size * num_heads * head_dim` (`key`, `value`) plus
a `len: usize`. The private `fn append in paged_attention.rs`
debug-asserts capacity then `copy_from_slice`s the source into
the page's tail.

### PagePool (REQ-2, REQ-8)

`pub struct PagePool<T: Float>` at
`pub struct PagePool in paged_attention.rs` pre-allocates the page
buffer at construction (`(0..num_pages).map(KVPage::new)`) and
maintains a `free_pages: Vec<usize>` stack pushed in reverse so the
lowest IDs are allocated first. `alloc_page` is a `Vec::pop`;
`free_page` clears the page and pushes the ID back onto the stack.

### PagedKVCache (REQ-3, REQ-4, REQ-5, REQ-6)

`pub struct PagedKVCache<T: Float>` at
`pub struct PagedKVCache in paged_attention.rs` carries
`page_ids: Vec<usize>` and `total_tokens: usize`. `append` runs in
a loop:

1. Check the remaining capacity of the last page (`remaining_in_last`).
2. If `remaining_in_last > 0`, write up to that many tokens into the
   last page.
3. Otherwise, allocate a fresh page from the pool, panic with a
   structured error if exhausted, and write up to `page_size`
   tokens.

`get_keys` / `get_values` use the private `fn gather_data in
paged_attention.rs` to extend a `Vec<T>` with each page's
`key_data()` / `value_data()` slice in order, then wrap the result
in a `Tensor<T>` of shape `[total_tokens, num_heads, head_dim]`.

`free_all` iterates `self.page_ids` and `pool.free_page(pid)`s each,
then clears the vec and resets `total_tokens`.

### PagedAttentionManager (REQ-7)

`pub struct PagedAttentionManager<T: Float>` at
`pub struct PagedAttentionManager in paged_attention.rs` owns the
pool plus a `Vec<Option<PagedKVCache<T>>>` indexed by sequence ID.
`add_sequence` scans for `None` slots to reuse before pushing a new
slot. `append_kv` / `get_kv` / `remove_sequence` all check the slot
is `Some(_)` and return a structured `FerrotorchError` on invalid
IDs.

### Non-test production consumers

- `pub use paged_attention::{KVPage, PagePool,
  PagedAttentionManager, PagedKVCache}` at
  `ferrotorch-nn/src/lib.rs:234` — grandfathered public API
  surface. Targets LLM-serving consumers (per the SOSP 2023
  motivation); the umbrella `pub use` exposes the four types for
  external composition.

## Parity contract

`parity_ops = []`. PagedAttention is a memory-management abstraction
rather than a numerical op, so it has no direct PyTorch oracle.
Edge cases:

- **Pool exhaustion** — `append` returns
  `FerrotorchError::InvalidArgument` with a message naming the
  remaining tokens that couldn't be stored. Matches vLLM's
  fail-fast behaviour.
- **Empty sequence** — `get_keys` / `get_values` return
  `[0, num_heads, head_dim]` tensors without invoking the pool.
- **Sequence-ID reuse after remove** — `add_sequence` reuses the
  oldest `None` slot, so sequence IDs are dense and stable for
  the lifetime of the manager.

## Verification

Tests in `mod tests in paged_attention.rs`. Highlights:

- `PagePool::new` / `alloc_page` / `free_page` round-trip.
- `PagedKVCache::append` with multi-page spillover.
- `PagedAttentionManager::add_sequence` /
  `remove_sequence` slot-reuse.
- Pool exhaustion returns an error.

No parity-sweep ops declared. Smoke command:

```bash
cargo test -p ferrotorch-nn --lib paged_attention:: 2>&1 | tail -3
```

Expected: all tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct KVPage<T: Float>` in `paged_attention.rs`; non-test consumer: re-export at `ferrotorch-nn/src/lib.rs:234`. |
| REQ-2 | SHIPPED | impl: `pub struct PagePool<T: Float>` with `alloc_page` / `free_page` / `num_free` / `num_used` / `num_total` in `paged_attention.rs`; non-test consumer: re-export at `lib.rs:234`. |
| REQ-3 | SHIPPED | impl: `pub struct PagedKVCache<T: Float>` in `paged_attention.rs`; non-test consumer: re-export at `lib.rs:234`. |
| REQ-4 | SHIPPED | impl: `pub fn append` on `PagedKVCache` in `paged_attention.rs` (length validation + iterative page-fill loop); non-test consumer: re-export at `lib.rs:234`. |
| REQ-5 | SHIPPED | impl: `pub fn get_keys` / `pub fn get_values` on `PagedKVCache` in `paged_attention.rs` (using the private `fn gather_data`); non-test consumer: re-export at `lib.rs:234`. |
| REQ-6 | SHIPPED | impl: `pub fn free_all` on `PagedKVCache` in `paged_attention.rs`; non-test consumer: re-export at `lib.rs:234`. |
| REQ-7 | SHIPPED | impl: `pub struct PagedAttentionManager<T: Float>` with `add_sequence` / `append_kv` / `get_kv` / `remove_sequence` in `paged_attention.rs`; non-test consumer: re-export at `lib.rs:234`. |
| REQ-8 | SHIPPED | impl: `Vec::pop` / `Vec::push` for `alloc_page` / `free_page` in `paged_attention.rs`; non-test consumer: re-export at `lib.rs:234`. |
| REQ-9 | SHIPPED | impl: `#[derive(Debug, Clone)]` on `KVPage`, `PagedKVCache` and `#[derive(Debug)]` on `PagePool`, `PagedAttentionManager` in `paged_attention.rs`; non-test consumer: re-export at `lib.rs:234`. |
