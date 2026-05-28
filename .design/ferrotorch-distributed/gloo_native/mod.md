# gloo_native — native-Rust Gloo backend root module

<!--
tier: 3-component
status: draft
baseline-pytorch: main (user's local clone /home/doll/pytorch)
upstream-paths:
  - torch/csrc/distributed/c10d/ProcessGroupGloo.cpp
  - torch/csrc/distributed/c10d/ProcessGroupGloo.hpp
  - torch/distributed/distributed_c10d.py
-->

## Summary

`ferrotorch-distributed/src/gloo_native/mod.rs` is the root of the
native-Rust Gloo backend (#1132). It wires the three sub-modules
(`transport`, `connect`, `collectives`) together into the
`GlooBackendInner` type that the public `gloo_backend::GlooBackend`
handle delegates to. The module replaces the C++ `libgloo` dependency
PyTorch's `ProcessGroupGloo` links against with pure-Rust TCP transport,
length-prefixed framing, full-mesh topology, and textbook ring
allreduce / tree broadcast / ring barrier algorithms.

## Requirements

- REQ-1: `pub struct GlooBackendInner` owns `rank: usize`,
  `world_size: usize`, and `connections: connect::PeerStreams`
  (one `PeerConn` per peer, self-slot held as `None`). The
  inner is constructed by driving the rendezvous handshake.
- REQ-2: `pub fn GlooBackendInner::new(cfg: &RendezvousConfig) ->
  GlooResult<Self>` invokes `connect::rendezvous(cfg)` to set up the
  full mesh, then captures rank/world_size from the config.
- REQ-3: `pub const DEFAULT_GLOO_TIMEOUT: Duration =
  Duration::from_secs(60)` matches `collective::DEFAULT_COLLECTIVE_TIMEOUT`
  so every `Backend::recv` / barrier call has a default upper bound.
- REQ-4: `pub fn ring_allreduce_sum_f32(&self, data: &mut [f32])`
  reinterprets the `f32` slice as `&mut [u8]` (via a `SAFETY`-justified
  raw-pointer cast) and forwards to
  `collectives::ring_allreduce_sum_f32_bytes`. A
  `_with_timeout` overload accepts a custom per-step recv timeout.
- REQ-5: `pub fn tree_broadcast_f32(&self, data: &mut [f32], root:
  usize)` mirrors REQ-4 but delegates to
  `collectives::tree_broadcast_f32_bytes`. A `_with_timeout`
  overload exists.
- REQ-6: `impl RingTransport for GlooBackendInner` exposes the
  `rank` / `world_size` / `send` / `recv` minimal surface the ring
  collectives in `collectives` need (a type-erased trait so tests
  can drive the same algorithms over an in-process channel matrix).
- REQ-7: `impl Backend for GlooBackendInner` provides the workspace-
  wide `Backend` trait surface: `rank` / `world_size` / `send` /
  `recv` / `recv_timeout` / `barrier`. Default `recv` uses
  `DEFAULT_GLOO_TIMEOUT` to avoid hangs. `barrier` calls
  `collectives::ring_barrier`.
- REQ-8: `pub use connect::RendezvousConfig as GlooRendezvousConfig`
  exposes the rendezvous config struct under a Gloo-specific name so
  `mpi_backend.rs` and `ucc_backend.rs` can build it without depending
  on the internal `connect` module path.
- REQ-9: The sub-modules are declared `pub(super) mod` so they're
  reachable from `gloo_backend` but not from outside the crate;
  module is `pub(crate) mod gloo_native` in `lib.rs`, gated on
  `#[cfg(feature = "gloo-backend")]`.

## Acceptance Criteria

- [x] AC-1: `GlooBackendInner::new` performs the full rendezvous
  handshake and returns a struct with `world_size` populated
  `Option<PeerConn>` slots (self-slot `None`); verified by
  `rendezvous_full_mesh_n4_all_slots_filled` in
  `connect::tests`.
- [x] AC-2: 2-rank ring allreduce over real TCP completes with
  the elementwise sum; verified by
  `ring_allreduce_over_real_tcp_two_ranks` in `mod::tests`.
- [x] AC-3: 4-rank ring allreduce with uneven chunks (13
  elements / 4 ranks) completes correctly — verified by
  `ring_allreduce_over_real_tcp_four_ranks`.
- [x] AC-4: 4-rank tree broadcast over real TCP correctly
  propagates from a non-zero root; verified by
  `tree_broadcast_over_real_tcp_four_ranks`.
- [x] AC-5: 3-rank ring barrier completes; verified by
  `barrier_over_real_tcp_three_ranks`.
- [x] AC-6: `RendezvousConfig::from_env` reads the four
  PyTorch-standard env vars; verified by
  `rendezvous_config_from_env_reads_pytorch_vars`.

## Architecture

The module file declares three private sub-modules (`collectives`,
`connect`, `error`, `transport`) and re-exports the symbols its
top-level types use: `ring_allreduce_sum_f32_bytes`,
`ring_barrier`, `tree_broadcast_f32_bytes`, `RingTransport`,
`PeerConn`, `PeerStreams`, `RendezvousConfig`, `rendezvous`,
`GlooResult`, `recv_msg_into`, `send_msg`, `with_read_timeout`.
The only `pub use` re-export is `RendezvousConfig as
GlooRendezvousConfig` (REQ-8).

`pub struct GlooBackendInner` in `gloo_native/mod.rs` owns
the three fields described in REQ-1. The `Debug` derive is
mechanical (no opaque pointers to hide).

`pub fn GlooBackendInner::new` is the only constructor; it
invokes `connect::rendezvous(cfg)` and reads `cfg.rank` /
`cfg.world_size` to populate the struct. `fn conn(&self, peer)`
is an internal helper that bounds-checks the peer rank against
`world_size`, rejects self-targeted ops with `SelfSend`, and
returns `&PeerConn` (or `NoConnection` if the slot is `None`).

The two byte-level inner methods `send_inner` and `recv_inner`
lock the appropriate half (`writer.lock()` for send,
`reader.lock()` for recv) of the per-peer `PeerConn` and forward
to `transport::send_msg` / `transport::recv_msg_into`. Lock
poisoning maps to `DistributedError::LockPoisoned` with a
formatted message identifying the rank pair.

The two `f32` collective entry points (`ring_allreduce_sum_f32`,
`tree_broadcast_f32`) reinterpret the caller's `&mut [f32]` as
`&mut [u8]` via a raw-pointer cast (the `SAFETY:` comment
documents that `f32` is plain old data, no padding, and the
view is held only for the call's duration). The byte slice goes
to the corresponding `_bytes` algorithm in `collectives`.

`impl RingTransport for GlooBackendInner` is the test-friendly
trait that the ring algorithms in `collectives` take as
`&dyn RingTransport`; the production path always uses the real
backend, but the tests use an in-process channel-matrix shim
of the same shape.

`impl Backend for GlooBackendInner` makes the inner backend
directly usable through the workspace `Backend` trait — useful
in `mpi_backend.rs` and `ucc_backend.rs` where the public
`MpiBackend` / `UccBackend` types delegate to it.

### Consumer sites (production, non-test)

- `ferrotorch-distributed/src/gloo_backend.rs` — `pub struct
  GlooBackend` wraps `GlooBackendInner` (feature-on) and forwards
  every `Backend` method.
- `ferrotorch-distributed/src/mpi_backend.rs` — `pub struct
  MpiBackend` wraps `GlooBackendInner` and forwards the
  collective entry points.
- `ferrotorch-distributed/src/ucc_backend.rs` — `pub struct
  UccBackend` holds a `cpu_inner: GlooBackendInner` and uses
  it for CPU-tensor collectives.

## Parity contract

No parity-sweep ops. The contract is the C10d `ProcessGroupGloo`
shape:

- `ProcessGroupGloo::Options` (`torch/csrc/distributed/c10d/ProcessGroupGloo.hpp`)
  ↔ `RendezvousConfig` (re-exported as `GlooRendezvousConfig`).
- `ProcessGroupGloo::allreduce` (CUDA/CPU ring algorithm) ↔
  `ring_allreduce_sum_f32` (f32 sum only — Gloo on PyTorch
  ships multiple dtypes/ops; ferrotorch's gloo_native is f32-sum
  scoped per #1132).
- `ProcessGroupGloo::broadcast` ↔ `tree_broadcast_f32`.
- `ProcessGroupGloo::barrier` ↔ `barrier` (ring barrier, two-wave).

## Verification

`cargo test -p ferrotorch-distributed --features gloo-backend
--lib` runs six in-process tests at `gloo_native/mod.rs`:

- `full_mesh_send_recv_two_ranks` — verifies the trait `send` /
  `recv` round-trip over real TCP for 2 ranks.
- `ring_allreduce_over_real_tcp_two_ranks` — 2-rank ring
  allreduce of `[1,2,3,4]` + `[10,20,30,40]` = `[11,22,33,44]`.
- `ring_allreduce_over_real_tcp_four_ranks` — 4-rank
  allreduce with 13 elements (uneven chunks).
- `tree_broadcast_over_real_tcp_four_ranks` — 4-rank tree
  broadcast from a non-zero root.
- `barrier_over_real_tcp_three_ranks` — 3-rank barrier.
- `rendezvous_config_from_env_reads_pytorch_vars` —
  `RendezvousConfig::from_env` env-var parsing.

Without `--features=gloo-backend`, the module is `cfg`-gated out
entirely; no tests run for this file.

No parity ops; integer grep count is 0 by construction.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct GlooBackendInner` in `ferrotorch-distributed/src/gloo_native/mod.rs`; non-test consumer: `ferrotorch-distributed/src/gloo_backend.rs` (`GlooBackend.inner: native::GlooBackendInner`), `ferrotorch-distributed/src/mpi_backend.rs` (`MpiBackend.inner`), `ferrotorch-distributed/src/ucc_backend.rs` (`UccBackend.cpu_inner`). |
| REQ-2 | SHIPPED | impl: `pub fn GlooBackendInner::new` in `ferrotorch-distributed/src/gloo_native/mod.rs`; non-test consumer: `ferrotorch-distributed/src/gloo_backend.rs` `GlooBackend::new` and `from_env` call it; mirrored in `mpi_backend.rs` `MpiBackend::new` and `ucc_backend.rs` `UccBackend::new`. |
| REQ-3 | SHIPPED | impl: `pub const DEFAULT_GLOO_TIMEOUT` in `ferrotorch-distributed/src/gloo_native/mod.rs`; non-test consumer: `impl Backend for GlooBackendInner::recv` and `::barrier` (same file) reference it as the default timeout for production `Backend::recv` / `Backend::barrier` calls. |
| REQ-4 | SHIPPED | impl: `pub fn ring_allreduce_sum_f32` and `pub fn ring_allreduce_sum_f32_with_timeout` in `ferrotorch-distributed/src/gloo_native/mod.rs`; non-test consumer: `ferrotorch-distributed/src/gloo_backend.rs` (`GlooBackend::ring_allreduce_sum_f32` forwards), `ferrotorch-distributed/src/mpi_backend.rs` (`MpiBackend::allreduce_sum_f32` forwards), `ferrotorch-distributed/src/ucc_backend.rs` (`UccBackend::allreduce_sum_f32` forwards). |
| REQ-5 | SHIPPED | impl: `pub fn tree_broadcast_f32` and `pub fn tree_broadcast_f32_with_timeout` in `ferrotorch-distributed/src/gloo_native/mod.rs`; non-test consumer: `ferrotorch-distributed/src/gloo_backend.rs` (`GlooBackend::tree_broadcast_f32` forwards), `ferrotorch-distributed/src/mpi_backend.rs` (`MpiBackend::broadcast_f32` forwards), `ferrotorch-distributed/src/ucc_backend.rs` (`UccBackend::broadcast_f32` forwards). |
| REQ-6 | SHIPPED | impl: `impl RingTransport for GlooBackendInner` in `ferrotorch-distributed/src/gloo_native/mod.rs`; non-test consumer: the `_with_timeout` collective entry points (same file) take `self: &Self` and pass `self` as `&dyn RingTransport` into `collectives::ring_allreduce_sum_f32_bytes` / `tree_broadcast_f32_bytes` / `ring_barrier`. |
| REQ-7 | SHIPPED | impl: `impl Backend for GlooBackendInner` in `ferrotorch-distributed/src/gloo_native/mod.rs`; non-test consumer: `ferrotorch-distributed/src/gloo_backend.rs` `impl Backend for GlooBackend` forwards every method to the inner; same for `mpi_backend.rs` and `ucc_backend.rs`. |
| REQ-8 | SHIPPED | impl: `pub use self::connect::RendezvousConfig as GlooRendezvousConfig` in `ferrotorch-distributed/src/gloo_native/mod.rs`; non-test consumer: `ferrotorch-distributed/src/mpi_backend.rs` imports `crate::gloo_native::GlooRendezvousConfig`, `ferrotorch-distributed/src/ucc_backend.rs` does the same. |
| REQ-9 | SHIPPED | impl: `pub(crate) mod gloo_native` gated by `#[cfg(feature = "gloo-backend")]` in `ferrotorch-distributed/src/lib.rs`; non-test consumer: `mpi_backend.rs` / `ucc_backend.rs` import via `crate::gloo_native::...` paths that resolve under the feature. |
