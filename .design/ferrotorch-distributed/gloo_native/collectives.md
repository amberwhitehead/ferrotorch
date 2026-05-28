# gloo_native::collectives — ring allreduce / tree broadcast / ring barrier

<!--
tier: 3-component
status: draft
baseline-pytorch: main (user's local clone /home/doll/pytorch)
upstream-paths:
  - torch/csrc/distributed/c10d/ProcessGroupGloo.cpp
  - torch/csrc/distributed/c10d/ProcessGroupGloo.hpp
-->

## Summary

`ferrotorch-distributed/src/gloo_native/collectives.rs` provides the
three textbook ring/tree collective algorithms that
`GlooBackendInner` exposes via its `f32` and trait-level `Backend`
surfaces: ring allreduce (scatter-reduce + allgather), binary-tree
broadcast, and a two-wave ring barrier. The algorithms are
parameterised by the `RingTransport` trait (defined here) so unit
tests can drive them over an in-process `mpsc::channel` matrix
shim without binding real TCP sockets. Mirrors the role of
`libgloo`'s `algorithms/allreduce_ring`, `algorithms/broadcast`,
and `algorithms/barrier` in `ProcessGroupGloo`.

## Requirements

- REQ-1: `pub(super) trait RingTransport: Sync` exposes the
  minimal P2P surface the ring algorithms need: `rank()`,
  `world_size()`, `send(&[u8], dst) -> GlooResult<()>`,
  `recv(&mut [u8], src, Duration) -> GlooResult<()>`. The trait
  is `pub(super)` so it doesn't leak out of `gloo_native::`.
- REQ-2: `pub(super) fn ring_allreduce_sum_f32_bytes(transport:
  &dyn RingTransport, buf: &mut [u8], timeout: Duration)`
  performs the canonical ring scatter-reduce + allgather over a
  byte-view of an `f32` slice. Total bytes moved per rank:
  `2 * (N - 1) / N * buf.len()` (communication-optimal).
- REQ-3: Chunking: data is split into `world_size` chunks via
  `chunk_ranges(total_elems, world_size)`, where the last
  chunk absorbs the remainder. Cover-once invariant: every
  element lands in exactly one chunk; chunks are contiguous
  and non-overlapping.
- REQ-4: Each ring step posts a send-to-next and recv-from-prev
  concurrently via `std::thread::scope`. For `world_size >= 3`,
  `next != prev` so disjoint per-peer locks make the parallel
  shape safe. For `world_size == 2`, `PeerConn`'s split
  reader/writer halves carry the same socket without contention.
- REQ-5: `pub(super) fn tree_broadcast_f32_bytes(transport:
  &dyn RingTransport, buf: &mut [u8], root: usize, timeout:
  Duration)` is a binary-tree broadcast with rank-rooted tree
  coords (`tree_rank = (rank + world_size - root) % world_size`).
  Depth `ceil(log2(N))`. Rejects `root >= world_size` with
  `DistributedError::InvalidRank`.
- REQ-6: `pub(super) fn ring_barrier(transport: &dyn
  RingTransport, timeout: Duration)` is a two-wave ring barrier:
  one byte forwarded all the way around the ring twice (arrival
  + release). Guarantees no rank exits before every rank has
  entered, without a centralised coordinator.
- REQ-7: Edge case `world_size == 1`: every collective returns
  `Ok(())` immediately. Edge case `buf.is_empty()` in allreduce
  returns `Ok(())`. Edge case `buf.len() % 4 != 0` in allreduce
  returns `DistributedError::SizeMismatch`.
- REQ-8: F32-sum-only reduction. `accumulate_f32_inplace` is a
  byte-wise `dst[i] += src[i]` over 4-byte chunks, reading
  little-endian `f32::from_le_bytes`, summing, writing back
  little-endian. Other reductions (Mean / Max / Min / Prod) are
  NOT implemented in this layer — the `f32`-sum scope was set
  by #1132.

## Acceptance Criteria

- [x] AC-1: 2-rank allreduce of `[1,2,3]` + `[4,5,6]` produces
  `[5,7,9]` on both; verified by
  `ring_allreduce_two_ranks_sum`.
- [x] AC-2: 3-rank allreduce of `[1,…,6] + [10,…,60] + [100,…,600]`
  produces `[111, 222, …, 666]` on every rank; verified by
  `ring_allreduce_three_ranks_sum`.
- [x] AC-3: 4-rank allreduce with 7 elements (uneven chunks)
  produces `[15; 7]` on every rank; verified by
  `ring_allreduce_four_ranks_sum_with_uneven_chunks`.
- [x] AC-4: 4-rank tree broadcast from rank 1 propagates
  `[42, 43, 44]` to every rank; verified by
  `tree_broadcast_distributes_from_root`.
- [x] AC-5: 4-rank ring barrier serialises all ranks — atomic
  `entered` counter is `4` after the barrier on every rank;
  verified by `ring_barrier_serialises_all_ranks`.
- [x] AC-6: `chunk_ranges` invariants — balanced and unbalanced
  cases produce contiguous, non-overlapping ranges that span
  exactly `[0, total_elems)`; verified by
  `chunk_ranges_balanced` and `chunk_ranges_unbalanced_cover_all_elements_exactly_once`.
- [x] AC-7: `ring_neighbours(0, 4)` is `(3, 1)`,
  `ring_neighbours(3, 4)` is `(2, 0)` — wrap-around correct;
  verified by `ring_neighbours_wrap_around`.

## Architecture

`pub(super) trait RingTransport: Sync` is the test seam. Production
uses `impl RingTransport for GlooBackendInner` (defined in
`gloo_native/mod.rs`); tests in this file use a `RankView<'a>`
struct that holds `&'a Channels` (an `mpsc::channel` matrix) and
provides the same surface in-process.

`ring_allreduce_sum_f32_bytes` is the two-phase scatter-reduce +
allgather. At step `s` of phase 1 (scatter-reduce), rank `r` sends
chunk `(r - s) mod N` to `next` and receives chunk `(r - s - 1)
mod N` from `prev`, accumulating element-wise sum into its local
copy. After phase 1, rank `r`'s chunk `(r + 1) mod N` holds the
final reduction for that chunk. Phase 2 (allgather) rotates each
final chunk around the ring `N - 1` times: at step `s` rank `r`
sends the freshly-reduced chunk to `next` and OVERWRITES (NOT
accumulates) its prev's incoming chunk.

The interleaved send/recv at each step uses `std::thread::scope`
so the closure can borrow `transport: &dyn RingTransport`
without any `Arc` wrapping. The scoped worker handles the send;
the main thread handles the recv. Errors propagate through the
join.

`tree_broadcast_f32_bytes` uses rank-rooted tree coords
(`tree_rank = (rank + N - root) % N`) so the protocol is
symmetric in `root`. Children at tree positions `2r + 1` and
`2r + 2` are mapped back to global ranks via `(child_tree +
root) % world_size`. Non-root ranks recv from their parent
first, then forward to up to two children.

`ring_barrier` runs two waves of a single 1-byte token around the
ring. Rank 0 starts each wave; everyone else waits-then-forwards.
The two waves prevent the early-exit anomaly where rank 0
returns before rank 1 has finished its recv. The token byte
value (`0u8`) is inert — only its arrival matters.

The byte-level reduction helper `accumulate_f32_inplace` is the
only place ferrotorch's gloo_native does any actual arithmetic;
it iterates 4-byte chunks via `chunks_exact_mut`, parses both
operands as little-endian `f32`, sums, writes back. The
`debug_assert_eq!(dst.len() % 4, 0)` is enforced by the chunk-
boundary computations (`bytes_of(elem_idx)` always returns a
multiple of 4).

### Consumer sites (production, non-test)

- `ferrotorch-distributed/src/gloo_native/mod.rs` —
  `GlooBackendInner::ring_allreduce_sum_f32_with_timeout` and
  `::tree_broadcast_f32_with_timeout` invoke the `_bytes`
  algorithms; `impl Backend for GlooBackendInner::barrier`
  invokes `ring_barrier`.
- The `RingTransport` trait is only implemented for
  `GlooBackendInner` in production; the test-side
  `RankView<'a>` impl is `#[cfg(test)]`.

## Parity contract

No parity-sweep ops. The contract is the canonical ring/tree
collective shape used by every modern MPI/Gloo/NCCL
implementation:

- Ring allreduce: total bytes moved per rank = `2 * (N - 1) /
  N * buf.len()` — communication-optimal allreduce volume,
  matching the formula in NCCL's documentation and
  `libgloo/algorithms/allreduce_ring`.
- Tree broadcast: depth `ceil(log2(N))` — same as `libgloo`'s
  binary-tree broadcast.
- Ring barrier: two waves — matches the standard textbook
  barrier (e.g., Herlihy/Shavit "The Art of Multiprocessor
  Programming").
- F32-sum only: ferrotorch's gloo_native scopes the
  reduction to `f32` sum per #1132. PyTorch's Gloo supports
  multiple dtypes/ops; widening is tracked as a separate
  blocker.

## Verification

`cargo test -p ferrotorch-distributed --features
gloo-backend --lib` runs seven tests in `collectives::tests`:

- `ring_allreduce_two_ranks_sum`
- `ring_allreduce_three_ranks_sum`
- `ring_allreduce_four_ranks_sum_with_uneven_chunks`
- `tree_broadcast_distributes_from_root`
- `ring_barrier_serialises_all_ranks`
- `chunk_ranges_balanced`
- `chunk_ranges_unbalanced_cover_all_elements_exactly_once`
- `ring_neighbours_wrap_around`

Plus the end-to-end TCP integration in `gloo_native::tests`
(see `.design/ferrotorch-distributed/gloo_native/mod.md`).

No parity-sweep ops; integer grep count is 0 by construction.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub(super) trait RingTransport: Sync` in `ferrotorch-distributed/src/gloo_native/collectives.rs`; non-test consumer: `impl RingTransport for GlooBackendInner` in `ferrotorch-distributed/src/gloo_native/mod.rs`. |
| REQ-2 | SHIPPED | impl: `pub(super) fn ring_allreduce_sum_f32_bytes` in `ferrotorch-distributed/src/gloo_native/collectives.rs`; non-test consumer: `ferrotorch-distributed/src/gloo_native/mod.rs` `pub fn ring_allreduce_sum_f32_with_timeout` invokes it; that in turn is called from `ferrotorch-distributed/src/gloo_backend.rs` `GlooBackend::ring_allreduce_sum_f32`, `mpi_backend.rs` `MpiBackend::allreduce_sum_f32`, `ucc_backend.rs` `UccBackend::allreduce_sum_f32`. |
| REQ-3 | SHIPPED | impl: `fn chunk_ranges` and `const fn bytes_of` in `ferrotorch-distributed/src/gloo_native/collectives.rs`; non-test consumer: `pub(super) fn ring_allreduce_sum_f32_bytes` (same file) calls `chunk_ranges` and `bytes_of` at every step. |
| REQ-4 | SHIPPED | impl: `fn send_recv` (scoped-thread shape) in `ferrotorch-distributed/src/gloo_native/collectives.rs`; non-test consumer: `pub(super) fn ring_allreduce_sum_f32_bytes` calls `send_recv` at every ring step (phases 1 and 2). Disjoint-locks safety relies on `PeerConn`'s split halves in `ferrotorch-distributed/src/gloo_native/connect.rs`. |
| REQ-5 | SHIPPED | impl: `pub(super) fn tree_broadcast_f32_bytes` in `ferrotorch-distributed/src/gloo_native/collectives.rs`; non-test consumer: `ferrotorch-distributed/src/gloo_native/mod.rs` `pub fn tree_broadcast_f32_with_timeout` invokes it; that is consumed by `gloo_backend.rs`, `mpi_backend.rs`, `ucc_backend.rs` broadcast methods. |
| REQ-6 | SHIPPED | impl: `pub(super) fn ring_barrier` in `ferrotorch-distributed/src/gloo_native/collectives.rs`; non-test consumer: `ferrotorch-distributed/src/gloo_native/mod.rs` `impl Backend for GlooBackendInner::barrier` invokes `ring_barrier(self, DEFAULT_GLOO_TIMEOUT)`. |
| REQ-7 | SHIPPED | impl: early-return shapes at the top of `ring_allreduce_sum_f32_bytes` (`world_size == 1`, `buf.is_empty()`, `buf.len() % 4 != 0`) in `ferrotorch-distributed/src/gloo_native/collectives.rs`; non-test consumer: every collective invocation through `GlooBackendInner` traverses these guards. |
| REQ-8 | SHIPPED | impl: `fn accumulate_f32_inplace` in `ferrotorch-distributed/src/gloo_native/collectives.rs`; non-test consumer: `pub(super) fn ring_allreduce_sum_f32_bytes` (same file) calls `accumulate_f32_inplace` at every scatter-reduce step. |
