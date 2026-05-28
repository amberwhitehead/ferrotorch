# gloo_native::connect — rendezvous + full-mesh TCP setup

<!--
tier: 3-component
status: draft
baseline-pytorch: main (user's local clone /home/doll/pytorch)
upstream-paths:
  - torch/csrc/distributed/c10d/ProcessGroupGloo.cpp
  - torch/csrc/distributed/c10d/ProcessGroupGloo.hpp
  - torch/csrc/distributed/c10d/TCPStore.cpp
  - torch/distributed/distributed_c10d.py
-->

## Summary

`ferrotorch-distributed/src/gloo_native/connect.rs` drives the
3-step rendezvous protocol that establishes the full-mesh TCP
topology underpinning the native Gloo backend. Mirrors the
`init_process_group(backend="gloo")` env-var handshake PyTorch
exposes to users — `MASTER_ADDR`, `MASTER_PORT`, `RANK`, `WORLD_SIZE`
— and the per-peer connection layout `ProcessGroupGloo` uses
internally (`libgloo`'s `Context` builds a mesh; this module is
the Rust analog).

## Requirements

- REQ-1: `pub struct RendezvousConfig` carries `master_addr:
  String`, `rank: usize`, `world_size: usize`, `bind_addr:
  SocketAddr`. `bind_addr` defaults to `127.0.0.1:0` (kernel-
  assigned port) for the local peer listener.
- REQ-2: `pub fn RendezvousConfig::from_env() -> GlooResult<Self>`
  reads `MASTER_ADDR`, `MASTER_PORT`, `RANK`, `WORLD_SIZE` from
  process env, producing
  `master_addr = format!("{MASTER_ADDR}:{MASTER_PORT}")` and
  defaulting `bind_addr` to `127.0.0.1:0`. Missing env vars map
  to `DistributedError::Io` with a discriminating message.
- REQ-3: `pub(super) struct PeerConn` wraps one TCP socket as
  two `Mutex<TcpStream>` halves (`writer` and `reader`), obtained
  via `TcpStream::try_clone`. The split lets concurrent `send` /
  `recv` on the same peer take disjoint locks — required for
  2-rank ring algorithms where `next == prev` and both directions
  share one socket.
- REQ-4: `pub(super) type PeerStreams = Vec<Option<PeerConn>>` is
  the per-rank connection slot vector. The self-slot is held as
  `None`; out-of-range or self-targeted sends produce
  `DistributedError::{InvalidRank, SelfSend}` upstream.
- REQ-5: `pub(super) fn rendezvous(cfg: &RendezvousConfig) ->
  GlooResult<PeerStreams>` validates `world_size >= 2` and
  `rank < world_size`, binds the local peer listener (capturing
  the kernel-assigned port), then runs either `run_master` (rank
  0) or `run_worker` (others) and finally calls `form_full_mesh`.
- REQ-6: 3-step protocol implemented exactly:
  1. Rank 0 binds `MASTER_ADDR:MASTER_PORT`, accepts `world_size
     - 1` connections, reads `(u64 LE peer rank, 6-byte peer ad
     [4-byte IPv4 + 2-byte LE u16 port])` from each.
  2. Rank 0 broadcasts the assembled `(rank → addr)` table back
     to every non-zero rank as `world_size * 6` flat bytes.
  3. Each rank then accepts from peers `> self` on its local
     peer listener AND connects to peers `< self` at the
     advertised addresses, populating its `PeerStreams[peer]`
     slot. Self-slot stays `None`.
- REQ-7: Non-zero ranks retry the rank-0 connect for up to 30
  seconds (`RENDEZVOUS_RETRY_TIMEOUT`, 50ms `RENDEZVOUS_RETRY_INTERVAL`)
  to tolerate the launch-time skew where rank 0 hasn't quite
  bound its listener yet.
- REQ-8: IPv4-only — the peer-advertisement format is 4 bytes
  IPv4 octets + 2 bytes LE u16 port. IPv6 `bind_addr` is rejected
  with `DistributedError::Io { message: "IPv6 bind_addr is not
  supported" }`.

## Acceptance Criteria

- [x] AC-1: 2-rank rendezvous populates `conns[0][1].is_some()`
  and `conns[1][0].is_some()`, with both self-slots `None`;
  verified by `rendezvous_full_mesh_n2`.
- [x] AC-2: 4-rank rendezvous fills every non-self slot
  with `Some(PeerConn)`; verified by
  `rendezvous_full_mesh_n4_all_slots_filled`.
- [x] AC-3: `world_size < 2` is rejected with
  `InvalidWorldSize`; verified by
  `rendezvous_rejects_world_size_below_two`.
- [x] AC-4: `rank >= world_size` is rejected with
  `InvalidRank`; verified by
  `rendezvous_rejects_rank_out_of_range`.
- [x] AC-5: Peer-ad encode/decode round-trips; verified by
  `peer_ad_round_trip`.

## Architecture

`pub struct RendezvousConfig` carries the four fields plus
`bind_addr`. `Clone + Debug` is derived. The `from_env`
constructor reads `MASTER_ADDR`/`MASTER_PORT` separately
(PyTorch uses two env vars; we re-format to one
`host:port` string internally). `rank` and `world_size` are
parsed via `usize::parse` with `Io`-mapped errors.

`pub(super) struct PeerConn` is the per-peer connection
shape. The `writer` and `reader` are two `Mutex<TcpStream>`
fields wrapping CLONES of the same underlying OS socket (via
`TcpStream::try_clone`). Splitting like this is the
canonical full-duplex TCP pattern; without it a 2-rank ring
deadlocks. `PeerConn::from_stream(stream)` clones the
reader half and wraps both in `Mutex`es.

`pub(super) fn rendezvous(cfg)` is the entry point. It
validates `world_size >= 2` (else `InvalidWorldSize`), then
`rank < world_size` (else `InvalidRank`). It binds the local
peer listener (`TcpListener::bind(cfg.bind_addr)`), captures
the assigned address with `local_addr()`, encodes the
6-byte peer-ad with `encode_peer_ad`, and dispatches to
either `run_master` (rank 0) or `run_worker` (others) to
obtain the peer-ad table. It ends with `form_full_mesh`.

`run_master` accepts `world_size - 1` connections in a
loop. For each: read 8 LE bytes as peer rank
(validating `0 < peer_rank < world_size`, else
`InvalidRank`), read 6 LE bytes as peer ad, slot into
`peer_table[peer_rank]`. Stores connecting streams in a
`Vec<(rank, TcpStream)>` so they can be re-used for the
broadcast phase. After all connections: flatten the table
into `world_size * 6` bytes, `write_all` + `flush` to every
stored stream.

`run_worker` connects to rank 0 with bounded retry
(`std::time::Instant::now() < deadline` loop with
`thread::sleep(RENDEZVOUS_RETRY_INTERVAL)`), then writes
`(rank u64 LE, peer_ad [u8; 6])`, then reads back the flat
`world_size * 6` table, then parses it into
`Vec<[u8; 6]>`.

`form_full_mesh` is the symmetric step that turns the
peer-ad table into actual TCP connections. Each rank
accepts from peers `> self` (we expect `world_size - 1 -
rank` inbound connections) and initiates the connect to
peers `< self`. The connecting side writes its rank as 8
LE bytes so the accepting side can slot the stream into
the right `streams[peer_rank]` index. Finally, each
populated slot is wrapped into `PeerConn::from_stream` to
get the split reader/writer halves.

The encoding is IPv4-only (6 bytes per peer: 4 octets + 2
port bytes). IPv6 `bind_addr` is explicitly rejected by
`encode_peer_ad` with an `Io` error.

### Consumer sites (production, non-test)

- `ferrotorch-distributed/src/gloo_native/mod.rs` —
  `GlooBackendInner::new` invokes `connect::rendezvous(cfg)`.
- `ferrotorch-distributed/src/gloo_native/mod.rs` re-exports
  `pub use connect::RendezvousConfig as GlooRendezvousConfig`.
- `ferrotorch-distributed/src/mpi_backend.rs` —
  `MpiBackend::new` constructs a `GlooRendezvousConfig` and
  feeds it to `GlooBackendInner::new`.
- `ferrotorch-distributed/src/ucc_backend.rs` —
  `UccBackend::new` does the same.
- `ferrotorch-distributed/src/gloo_backend.rs` —
  `GlooBackend::new` and `from_env` both construct via
  `GlooRendezvousConfig`.

## Parity contract

No parity-sweep ops. The contract is PyTorch's
`init_process_group(backend="gloo")` env-var convention:

- `MASTER_ADDR` / `MASTER_PORT` ↔ `master_addr`
  composition (`from_env`).
- `RANK` / `WORLD_SIZE` ↔ `rank` / `world_size`.
- Full-mesh topology ↔ `ProcessGroupGloo` uses Gloo's
  internal `Context` mesh; ferrotorch's `PeerStreams` is
  the Rust analog.
- Note: ferrotorch's rendezvous is NOT wire-compatible
  with PyTorch's Gloo — `libgloo` uses its own
  proprietary handshake. Users mixing ferrotorch and
  PyTorch processes in the same world-group is not
  supported.

## Verification

`cargo test -p ferrotorch-distributed --features
gloo-backend --lib` runs five tests in `connect::tests`:

- `rendezvous_full_mesh_n2` — 2-rank slot population.
- `rendezvous_full_mesh_n4_all_slots_filled` — 4-rank
  cover-everywhere invariant.
- `rendezvous_rejects_world_size_below_two`.
- `rendezvous_rejects_rank_out_of_range`.
- `peer_ad_round_trip` — `encode_peer_ad` /
  `decode_peer_ad` round-trip with a known port.

No parity-sweep ops; integer grep count is 0 by
construction.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct RendezvousConfig` in `ferrotorch-distributed/src/gloo_native/connect.rs`; non-test consumer: `ferrotorch-distributed/src/gloo_native/mod.rs` re-exports as `GlooRendezvousConfig`; constructed in `ferrotorch-distributed/src/gloo_backend.rs` `GlooBackend::new`, `mpi_backend.rs` `MpiBackend::new`, `ucc_backend.rs` `UccBackend::new`. |
| REQ-2 | SHIPPED | impl: `pub fn RendezvousConfig::from_env` in `ferrotorch-distributed/src/gloo_native/connect.rs`; non-test consumer: `ferrotorch-distributed/src/gloo_backend.rs` `GlooBackend::from_env` invokes `GlooRendezvousConfig::from_env()`. |
| REQ-3 | SHIPPED | impl: `pub(super) struct PeerConn` in `ferrotorch-distributed/src/gloo_native/connect.rs`; non-test consumer: `ferrotorch-distributed/src/gloo_native/mod.rs` `fn conn` returns `&PeerConn`, used by `send_inner` / `recv_inner` to acquire the writer/reader halves. |
| REQ-4 | SHIPPED | impl: `pub(super) type PeerStreams` in `ferrotorch-distributed/src/gloo_native/connect.rs`; non-test consumer: `ferrotorch-distributed/src/gloo_native/mod.rs` `GlooBackendInner.connections: PeerStreams`. |
| REQ-5 | SHIPPED | impl: `pub(super) fn rendezvous` in `ferrotorch-distributed/src/gloo_native/connect.rs`; non-test consumer: `ferrotorch-distributed/src/gloo_native/mod.rs` `GlooBackendInner::new` calls `rendezvous(cfg)?`. |
| REQ-6 | SHIPPED | impl: `fn run_master`, `fn run_worker`, `fn form_full_mesh` in `ferrotorch-distributed/src/gloo_native/connect.rs`; non-test consumer: `pub(super) fn rendezvous` (same file) dispatches to them; tests `rendezvous_full_mesh_n2` and `_n4_all_slots_filled` exercise both ends of the protocol. |
| REQ-7 | SHIPPED | impl: `RENDEZVOUS_RETRY_TIMEOUT` / `RENDEZVOUS_RETRY_INTERVAL` constants and the retry loop in `fn run_worker` in `ferrotorch-distributed/src/gloo_native/connect.rs`; non-test consumer: every multi-rank construction path through `GlooBackendInner::new` exercises it (race-tolerant by design). |
| REQ-8 | SHIPPED | impl: `fn encode_peer_ad` matches `SocketAddr::V4` / rejects `V6` with `Io` error in `ferrotorch-distributed/src/gloo_native/connect.rs`; non-test consumer: `pub(super) fn rendezvous` invokes it; verified by `peer_ad_round_trip` test. |
