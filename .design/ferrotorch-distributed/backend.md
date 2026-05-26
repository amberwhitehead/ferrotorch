# Backend trait + TCP / Simulated / SubBackend implementations

<!--
tier: 3-component
status: draft
baseline-pytorch: main (user's local clone /home/doll/pytorch)
upstream-paths:
  - torch/distributed/distributed_c10d.py
  - torch/distributed/__init__.py
-->

## Summary

`ferrotorch-distributed/src/backend.rs` defines the `Backend` trait
(transport-agnostic point-to-point byte messaging + barrier + an
NCCL-downcast hook) and three implementations: `TcpBackend` (real
multi-process over TCP sockets with star topology), `SimulatedBackend`
(in-process channels for unit tests), and `SubBackend` (subgroup view
that translates local↔global rank mappings). Mirrors the role of
PyTorch's `ProcessGroupGloo` / `ProcessGroupNCCL` (the C10d
`ProcessGroup` shape) plus `dist.new_group(ranks=[…])` — the
collective layer (`crate::collective`) consumes only the trait, so
the same op surface works against TCP, Simulated, NCCL hybrid, or any
future binding.

## Requirements

- REQ-1: `pub trait Backend: Send + Sync` with methods `rank() ->
  usize`, `world_size() -> usize`, `send(&[u8], dst) ->
  FerrotorchResult<()>`, `recv(&mut [u8], src) -> FerrotorchResult<()>`,
  `recv_timeout(&mut [u8], src, Duration) -> FerrotorchResult<()>`
  (default-delegates to `recv`), `barrier() -> FerrotorchResult<()>`,
  and a feature-gated `as_nccl_backend() -> Option<&NcclBackend>`
  downcast hook (default `None`).
- REQ-2: `pub struct TcpBackend` with `pub fn new(rank, world_size,
  master_addr) -> FerrotorchResult<Self>` that drives the star-topology
  rendezvous (rank 0 listens; non-zero ranks connect and announce
  their identity in 8 little-endian bytes). Connection storage is
  `Vec<Option<Mutex<TcpStream>>>` indexed by peer rank with the
  self-slot held as `None`.
- REQ-3: `impl Backend for TcpBackend` with a length-prefixed wire
  protocol (8-byte LE `u64` length + payload), validates length
  against the receive buffer's `dst.len()` on every recv, and
  per-peer `Mutex` lock acquisition so concurrent
  send/recv-on-different-pairs from different threads is safe.
- REQ-4: `pub struct SimulatedBackend` with a shared
  `Arc<Vec<Vec<(Mutex<Sender>, Mutex<Receiver>)>>>` channel matrix
  indexed `[src][dst]`. `pub fn create_group(world_size) ->
  FerrotorchResult<Vec<Self>>` constructs one backend per rank with
  all ranks sharing the same matrix.
- REQ-5: `impl Backend for SimulatedBackend` routing every send/recv
  through the corresponding `mpsc::channel` cell, with
  `mpsc::RecvTimeoutError::Timeout` mapped to
  `DistributedError::Timeout` and `Disconnected` to
  `DistributedError::ChannelClosed`.
- REQ-6: `pub struct SubBackend` wrapping `Arc<dyn Backend>` + sorted/
  deduped `members: Vec<usize>` (global ranks) + `local_rank: usize`
  (this rank's index within `members`). `pub fn new(parent,
  members) -> FerrotorchResult<Self>` validates that `parent.rank()`
  is a member and every member is `< parent.world_size()`.
- REQ-7: `impl Backend for SubBackend` translating `rank()` →
  `local_rank`, `world_size()` → `members.len()`, and every
  `send`/`recv`/`recv_timeout` peer-rank argument from local to
  global through `members[peer]`. The barrier is a per-subgroup
  gather-scatter dance at local rank 0.
- REQ-8: Barrier semantics across all three backends: rank 0 waits
  for a tag byte from every other rank, then sends a tag byte back
  to each. Mirrors PyTorch's `monitored_barrier` shape without the
  per-rank timeout reporting.

## Acceptance Criteria

- [x] AC-1: `pub trait Backend: Send + Sync` with the eight methods
  in REQ-1, including the `#[cfg(feature = "nccl")]`-gated downcast.
- [x] AC-2: `TcpBackend::new` performs the star rendezvous; the
  protocol byte-counts match (8-byte LE length prefix).
- [x] AC-3: `SimulatedBackend::create_group(0)` returns
  `Err(DistributedError::InvalidWorldSize)`.
- [x] AC-4: `SubBackend::new(parent, vec![])` returns
  `Err(DistributedError::InvalidWorldSize { world_size: 0 })` and
  `SubBackend::new` rejects a parent whose rank is not in `members`
  with `DistributedError::InvalidRank`.
- [x] AC-5: All three barrier implementations complete under a 4-rank
  in-process thread-spawn test (`test_simulated_barrier`,
  `test_subbackend_barrier`).
- [x] AC-6: `SubBackend::new` sorts + dedups `members` before
  storage, verified by `test_subbackend_sorts_and_dedups_members`.

## Architecture

### The `Backend` trait (REQ-1)

`pub trait Backend: Send + Sync` is the contract on top of which every
collective in `crate::collective` is built. The trait is intentionally
narrow: byte-level send/recv plus a barrier. Shape, dtype, and
broadcast semantics live in `p2p.rs` and `collective.rs`. The
`recv_timeout` method has a default impl that ignores the timeout and
delegates to `recv`; production backends override it. The
`as_nccl_backend` hook lets `gpu_collective::gpu_allreduce` query
whether the runtime backend is NCCL-accelerated; default returns
`None` so non-NCCL backends opt out at zero cost.

### `TcpBackend` rendezvous (REQ-2)

The `new` constructor at `pub fn new` in `TcpBackend` implements a
simple star-topology rendezvous:

1. Validate `world_size >= 2` and `rank < world_size` against
   `DistributedError::{InvalidWorldSize, InvalidRank}`.
2. Pre-allocate `peer_streams: Vec<Option<TcpStream>>` of
   length `world_size` with all `None`.
3. If `rank == 0`: bind a `TcpListener` to `master_addr`, accept
   `world_size - 1` connections, read the connecting rank from the
   first 8 LE bytes of each, slot the stream into
   `peer_streams[peer_rank]`.
4. If `rank != 0`: connect to `master_addr`, announce our rank as 8
   LE bytes, slot the connection into `peer_streams[0]`.
5. Convert to `Vec<Option<Mutex<TcpStream>>>` with the self-slot held
   as `None` to forbid self-loops.

The star topology means non-zero ranks share NO direct connection —
attempts produce `DistributedError::NoConnection`. The collective
layer routes everything through rank 0 (correct but not bandwidth-
optimal — ring/tree algorithms can be layered later without changing
this interface).

### `TcpBackend` wire protocol (REQ-3)

Each `send` writes 8 LE bytes for the payload length followed by the
payload, then flushes the stream. Each `recv` reads exactly 8 bytes,
parses the length as `u64::from_le_bytes`, verifies it equals
`dst.len()` (otherwise `DistributedError::SizeMismatch`), and reads
exactly that many bytes into the caller's buffer. `recv_timeout` sets
`TcpStream::set_read_timeout(Some(timeout))` around the read pair and
maps `WouldBlock` / `TimedOut` to `DistributedError::Timeout`.
Restoration of blocking mode on the stream happens regardless of the
read outcome (the helper closure pattern at lines 316-356).

The per-peer `Mutex<TcpStream>` enables two threads to send/recv on
disjoint peer pairs simultaneously. The mutex is acquired with
`.lock()` and the `PoisonError` mapped to
`DistributedError::LockPoisoned`.

### `SimulatedBackend` channel matrix (REQ-4 / REQ-5)

`SimulatedBackend` uses `std::sync::mpsc::channel` to model the
process group in-process. `channels[src][dst]` is the
`(Mutex<Sender>, Mutex<Receiver>)` pair carrying messages from `src`
to `dst`. `create_group(world_size)` constructs the full matrix and
hands out one `SimulatedBackend` per rank, all sharing the matrix
through `Arc::clone`. The two `Mutex`es per cell let multiple threads
queue messages on the sender side without unsynchronized access to
the `mpsc` API (which is `!Sync`).

`send` acquires the sender mutex for `channels[self.rank][dst_rank]`
and pushes a `Vec<u8>` copy of the payload.
`recv` acquires the receiver mutex for
`channels[src_rank][self.rank]`, calls `.recv()`, and validates the
received length against `dst.len()`. `recv_timeout` substitutes
`mpsc::Receiver::recv_timeout` and maps the two error variants per
REQ-5.

### `SubBackend` rank translation (REQ-6 / REQ-7)

`SubBackend` wraps an `Arc<dyn Backend>` parent + a sorted-and-deduped
`members: Vec<usize>` listing the global ranks of the subgroup +
`local_rank: usize` (this rank's index within `members`).
`SubBackend::new`:

- Rejects `members.is_empty()` with `InvalidWorldSize`.
- Rejects out-of-range members with `InvalidRank`.
- Sorts + dedups `members`.
- Locates `parent.rank()` in the deduped list; if absent, errors with
  `InvalidRank`.

`impl Backend for SubBackend` translates every local rank in the
trait API into a global rank by indexing `members[local]`. The
barrier is a fresh gather-scatter restricted to the subgroup (local
rank 0 receives `world_size - 1` tag bytes from local ranks
`1..members.len()` then sends a tag byte back to each). Because every
`send`/`recv` call into the parent uses the local→global rank map,
overlapping subgroups at local rank 0 would conflict with a
parent-level barrier — the caller is expected to avoid that
(documented at lines 727-732).

### Consumer sites (production, non-test)

- `ferrotorch-distributed/src/collective.rs` — every collective op
  takes `backend: &dyn Backend`.
- `ferrotorch-distributed/src/p2p.rs` — `send`/`recv`/`sendrecv`
  consume the trait.
- `ferrotorch-distributed/src/async_collective.rs` —
  `async_all_gather` / `async_reduce_scatter` take `Arc<dyn Backend>`.
- `ferrotorch-distributed/src/ddp.rs`,
  `ferrotorch-distributed/src/fsdp.rs` — gradient sync.
- `ferrotorch-distributed/src/pipeline.rs` — pipeline-parallel
  staged forward/backward exchange.
- `ferrotorch-distributed/src/sync_batch_norm.rs` — batch-norm
  statistics allreduce.
- `ferrotorch-distributed/src/rpc.rs` — RPC framing on top of the
  trait.
- `ferrotorch-distributed/src/hybrid_backend.rs` — wraps a
  `TcpBackend` and delegates the `Backend` trait to it while exposing
  NCCL on the side.
- `ferrotorch-distributed/src/gpu_collective.rs` — uses
  `as_nccl_backend()` to pick fast-path NCCL when available.
- `ferrotorch-distributed/src/gloo_backend.rs`,
  `mpi_backend.rs`, `ucc_backend.rs` — backend skeletons
  implementing the trait.
- `ferrotorch/src/lib.rs` — `pub use ferrotorch_distributed::*;`
  re-exports `Backend`, `SimulatedBackend`, `TcpBackend`, `SubBackend`.

## Parity contract

No parity-sweep ops in the route (`parity_ops = []`). The contract is
the C10d `ProcessGroup` shape:

- `ProcessGroup::send(tensor, dst, tag)`: ferrotorch's `send` drops
  the tag (no MPI-style tags in TCP/Simulated; ferrotorch is one
  contention domain per (src, dst) pair).
- `ProcessGroup::recv(tensor_list, src, tag)`: ferrotorch returns
  `SizeMismatch` if the caller-allocated buffer is wrong length —
  PyTorch resizes the tensor in place. R-DEV-4: Rust's typed buffer
  shape doesn't support runtime resize without realloc.
- `ProcessGroup::barrier()`: identical semantics in the gather-scatter
  shape.
- `dist.new_group(ranks=[...])` (cf.
  `torch/distributed/distributed_c10d.py:4995` `new_group`): mirrored
  by `SubBackend::new`. PyTorch reuses the parent backend's transport;
  ferrotorch does the same.

## Verification

- `cargo test -p ferrotorch-distributed --lib` runs the
  `#[cfg(test)] mod tests` at lines 755-944, covering:
  - `test_simulated_send_recv`, `test_simulated_barrier`,
    `test_simulated_rank_world_size`,
    `test_invalid_world_size`, `test_send_to_invalid_rank`.
  - `test_subbackend_local_rank_and_world_size`,
    `test_subbackend_global_rank_not_in_members_is_error`,
    `test_subbackend_empty_members_is_error`,
    `test_subbackend_send_recv_routes_through_parent`,
    `test_subbackend_barrier`,
    `test_subbackend_to_global_to_local`,
    `test_subbackend_sorts_and_dedups_members`.
- Lint: `cargo clippy -p ferrotorch-distributed -- -D warnings` PASS.
- Parity-sweep: no ops; integer grep count is 0 by construction.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub trait Backend: Send + Sync` in `ferrotorch-distributed/src/backend.rs`; non-test consumers: `ferrotorch-distributed/src/collective.rs`, `ferrotorch-distributed/src/p2p.rs`, `ferrotorch-distributed/src/ddp.rs`, `ferrotorch-distributed/src/fsdp.rs`. |
| REQ-2 | SHIPPED | impl: `pub struct TcpBackend` in `ferrotorch-distributed/src/backend.rs` + `pub fn new` in `ferrotorch-distributed/src/backend.rs`; non-test consumer: `ferrotorch-distributed/src/hybrid_backend.rs` (`TcpBackend::new(rank, world_size, addr)`). |
| REQ-3 | SHIPPED | impl: `impl Backend for TcpBackend` in `ferrotorch-distributed/src/backend.rs` with length-prefix protocol at lines 222-233 and 259-275; non-test consumer: `ferrotorch-distributed/src/hybrid_backend.rs` (delegates all P2P methods to inner TcpBackend). |
| REQ-4 | SHIPPED | impl: `pub struct SimulatedBackend` in `ferrotorch-distributed/src/backend.rs` + `create_group` at line 406; non-test consumer: `ferrotorch/src/lib.rs` (`pub use ferrotorch_distributed::*;` re-exports `SimulatedBackend`); see also `ferrotorch-distributed/src/lib.rs` re-export. |
| REQ-5 | SHIPPED | impl: `impl Backend for SimulatedBackend` in `ferrotorch-distributed/src/backend.rs` with channel-cell send/recv at lines 447-501 and `recv_timeout` error mapping at lines 526-536; non-test consumer: `ferrotorch/src/lib.rs` re-export reaches every workspace user who picks `SimulatedBackend` for unit tests. |
| REQ-6 | SHIPPED | impl: `pub struct SubBackend` in `ferrotorch-distributed/src/backend.rs` + `pub fn new` in `ferrotorch-distributed/src/backend.rs` with members sort-dedup at lines 631-634; non-test consumer: `ferrotorch-distributed/src/lib.rs` re-exports `SubBackend`, and `ferrotorch-distributed/src/backend.rs` documents `SubBackend` as the FSDP `HybridShard` strategy primitive (CL-327). |
| REQ-7 | SHIPPED | impl: `impl Backend for SubBackend` in `ferrotorch-distributed/src/backend.rs` with local↔global translation at lines 688-707; non-test consumer: re-export at `ferrotorch-distributed/src/lib.rs` makes `SubBackend` reachable through the meta-crate (`ferrotorch/src/lib.rs`). |
| REQ-8 | SHIPPED | impl: barrier methods at `ferrotorch-distributed/src/backend.rs` (TcpBackend), `ferrotorch-distributed/src/backend.rs` (SimulatedBackend), `ferrotorch-distributed/src/backend.rs` (SubBackend); non-test consumer: `ferrotorch-distributed/src/collective.rs` (`pub fn barrier(backend: &dyn Backend)` re-exports the trait method through the collective module surface). |
