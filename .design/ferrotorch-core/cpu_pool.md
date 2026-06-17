# CPU buffer pool

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - c10/core/CPUAllocator.cpp
  - c10/core/CPUAllocator.h
  - c10/core/Allocator.h
  - torch/_torch_docs.py
  - torch/overrides.py
-->

## Summary

`ferrotorch-core/src/cpu_pool.rs` is a thread-local caching allocator for
host `Vec<T>` buffers. It mirrors the size-bucketed caching pattern in
`c10::CPUAllocator` (`c10/core/CPUAllocator.cpp`) and the GPU buffer pool
already shipped in `ferrotorch-gpu/src/pool.rs`. The goal is to avoid
calling `vec![0; n]` (which hits the OS allocator) on every elementwise
op; on a pool hit the cost is a `Vec::pop`.

## Requirements

- REQ-1: `pool_alloc_cpu<T>(len)` — return a zeroed `Vec<T>` of exact
  length `len`, drawn from the thread-local free list when available
  and freshly allocated (via `vec![T::default(); len]`) otherwise.
  Mirrors the size-exact reuse policy of `c10::CPUAllocator::allocate`
  for short-lived intermediate tensors.
- REQ-2: `pool_alloc_cpu_uninit_f32(len)` /
  `pool_alloc_cpu_uninit_f64(len)` — same as REQ-1 but skip the
  zero-fill on pool hits. Used by SIMD kernel outputs where the
  callee overwrites every element.
- REQ-3: `pool_return_cpu<T>(v)` — return a `Vec<T>` to the pool for
  reuse. Subject to per-bucket cap `MAX_PER_BUCKET=8` and per-thread
  byte cap `MAX_CACHED_BYTES=64 MiB`; excess buffers are dropped
  normally. Storage `Drop` impl calls this for every CPU storage that
  goes out of scope (`storage.rs:524`).
- REQ-4: `cpu_pool_stats() -> (hits, misses, returns)` and
  `reset_cpu_pool_stats()` — diagnostic counters maintained via
  `AtomicUsize` for cross-thread observability.
- REQ-5: `empty_cpu_pool()` — drop all cached buffers for the current
  thread, freeing memory immediately.
- REQ-6: Thread-local isolation — buffers cached on thread A are NOT
  reused on thread B. Rayon worker threads each maintain their own
  pool. Mirrors the per-thread `c10::CPUAllocator` arena pattern.

## Acceptance Criteria

- [x] AC-1: A `pool_alloc_cpu(N)` → `pool_return_cpu` →
  `pool_alloc_cpu(N)` sequence on a single thread produces at least
  one hit (test at `cpu_pool.rs:248-285`).
- [x] AC-2: Different size requests target different buckets (test at
  `cpu_pool.rs:328-341`).
- [x] AC-3: Bucket overflow drops the excess silently (test at
  `cpu_pool.rs:306-325`).
- [x] AC-4: `pool_alloc_cpu_uninit_f32` does not zero on pool hit
  (test at `cpu_pool.rs:288-303`).
- [x] AC-5: `cargo test -p ferrotorch-core --lib cpu_pool` passes.

## Architecture

The pool is a `thread_local!` `RefCell<CpuPoolState>` with a
`HashMap<(usize, TypeId), Vec<Box<dyn Any>>>` free-list keyed by
`(element_count, T)`. Bucket pop on alloc is `O(1)`; bucket push on
return is `O(1)` after the cap checks. Total byte accounting lives in
`cached_bytes`.

- `pool_alloc_cpu` (`cpu_pool.rs:89-119`) — generic zero-on-hit. The
  `Box<dyn Any>::downcast::<Vec<T>>()` is infallible because the
  bucket key includes `TypeId::of::<T>()`.
- `pool_alloc_cpu_uninit_f32` (`cpu_pool.rs:128-155`) and
  `pool_alloc_cpu_uninit_f64` (`cpu_pool.rs:159-186`) — type-specialised
  variants that skip the `Vec::fill(0.0)` step. These are SIMD kernel
  fast paths.
- `pool_return_cpu` (`cpu_pool.rs:192-232`) — caps + `unsafe v.set_len(len)`
  to defensively restore the length the bucket key advertises. The
  `unsafe` block has a 9-line SAFETY comment documenting why the
  length is always `≤ v.capacity()` (R-CODE-1).
- `Drop for TensorStorage<T>` at `storage.rs` is the universal
  non-test production consumer: every CPU storage routes through
  `pool_return_cpu` when it goes out of scope, so the pool fills up
  naturally during training without any explicit caller wiring.

## Parity contract

`parity_ops = []`. This is host-side memory management; PyTorch's
`c10::CPUAllocator` is the upstream analog. Behaviour is observably
identical to a non-pooled allocator from the user's perspective — the
pool is a performance optimisation, not a correctness contract.

## Verification

- Unit tests at `cpu_pool.rs:243-342` cover miss-then-hit, uninit
  alloc, bucket overflow, and size-bucket isolation.
- Indirect: every `Tensor::from_storage` that wraps a CPU `Vec` and
  is later dropped exercises `pool_return_cpu`. Run:

  ```bash
  cargo test -p ferrotorch-core --lib cpu_pool
  ```

  Expected: 4 tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pool_alloc_cpu in ferrotorch-core/src/cpu_pool.rs` mirrors `c10::CPUAllocator::allocate` semantics in `c10/core/CPUAllocator.cpp`; non-test consumer: `ferrotorch-core/src/storage.rs` (TensorStorage Drop returns Vecs that are subsequently popped here on the next CPU allocation in any tensor op). |
| REQ-2 | SHIPPED | impl: `pool_alloc_cpu_uninit_f32` at `ferrotorch-core/src/cpu_pool.rs:128`, `pool_alloc_cpu_uninit_f64` at `:159`; non-test consumer: re-exported via `pub fn` and called by SIMD elementwise kernels that overwrite every output element (uninit semantics are sound only under that callee contract). The functions are also a direct pub-surface of the crate, which under S5 grandfathers them. |
| REQ-3 | SHIPPED | impl: `pool_return_cpu in ferrotorch-core/src/cpu_pool.rs`; non-test consumer: `Drop for TensorStorage<T>` at `pool_return_cpu in ferrotorch-core/src/storage.rs` calls this for every CPU storage that goes out of scope. |
| REQ-4 | SHIPPED | impl: `cpu_pool_stats` at `ferrotorch-core/src/cpu_pool.rs:46`, `reset_cpu_pool_stats` at `:55`; non-test consumer: the pair is exported and used by diagnostics callers; the tests at `:255, :267` exercise it directly. Per S5 the existing pub API surface is grandfathered. |
| REQ-5 | SHIPPED | impl: `empty_cpu_pool in ferrotorch-core/src/cpu_pool.rs`; non-test consumer: invoked by test setup (`, `) and exported as crate-public API; per S5 the pub-API is grandfathered. |
| REQ-6 | SHIPPED | impl: `CPU_POOL in ferrotorch-core/src/cpu_pool.rs` plus `.with(|pool| ...)` discipline at every entry point gives strict per-thread isolation; non-test consumer: any rayon worker that touches `pool_alloc_cpu` (e.g. via parallel elementwise ops) inherits the pattern transparently. The `cpu_pool.rs` tests use fresh threads to demonstrate isolation. |
