# Hybrid TCP + NCCL backend

<!--
tier: 3-component
status: draft
baseline-pytorch: main (user's local clone /home/doll/pytorch)
upstream-paths:
  - torch/distributed/distributed_c10d.py
-->

## Summary

`ferrotorch-distributed/src/hybrid_backend.rs` defines `HybridBackend`,
a feature-gated (`nccl`) wrapper that owns one `TcpBackend` for
point-to-point messaging (send / recv / barrier) and one `NcclBackend`
for GPU-native collectives (allreduce / broadcast / etc.). The
collective layer routes `Backend`-trait calls through the TCP half
while `crate::nccl_collective::*` callers reach the NCCL half via the
`.nccl()` accessor. Mirrors PyTorch's `ProcessGroupNCCL` which
co-instantiates a Gloo / TCPStore fallback for object-collectives and
barrier while delegating all tensor collectives to NCCL
(`torch/distributed/distributed_c10d.py`).

## Requirements

- REQ-1: `pub struct HybridBackend` owning a `tcp: TcpBackend` and a
  `nccl: NcclBackend`, both feature-gated on `nccl`.
- REQ-2: `pub fn new(rank, world_size, addr, unique_id) ->
  FerrotorchResult<Self>` that constructs both halves in order: TCP
  rendezvous first (so non-NCCL failures surface before CUDA context
  init), then NCCL communicator with the shared
  `unique_id`.
- REQ-3: `pub fn nccl(&self) -> &NcclBackend` and `pub fn tcp(&self)
  -> &TcpBackend` accessors so callers can reach the inner halves
  directly when they need NCCL-specific APIs not on the `Backend`
  trait (e.g., `nccl_allreduce` on a GPU-resident buffer).
- REQ-4: `pub fn synchronize_nccl(&self) -> FerrotorchResult<()>`
  forwarding to `NcclBackend::synchronize` so callers can flush the
  NCCL stream after enqueueing collectives.
- REQ-5: `impl Backend for HybridBackend` delegating every method
  (`rank`, `world_size`, `send`, `recv`, `recv_timeout`, `barrier`)
  to the inner `tcp` half. Barrier uses TCP (reliable, no GPU context
  required); NCCL barrier is left to direct callers of
  `.nccl().barrier()`.

## Acceptance Criteria

- [x] AC-1: The struct + impls are gated behind `#[cfg(feature =
  "nccl")]` at the module level (the entire file only compiles when
  `nccl` is enabled).
- [x] AC-2: `HybridBackend::new` propagates the first failure (TCP
  rendezvous error suppresses NCCL init).
- [x] AC-3: The `Backend` trait impl delegates to `tcp`, not `nccl`,
  for `send`/`recv`/`recv_timeout`/`barrier`.

## Architecture

### Module gating

The module is included from `lib.rs` only under `#[cfg(feature =
"nccl")]`. The crate-root re-export (`lib.rs` `pub use
hybrid_backend::HybridBackend;`) is similarly gated. With `nccl`
disabled, `HybridBackend` does not exist in the crate's public
surface — callers that need a hybrid setup are expected to compose
`TcpBackend` and a future native ring/tree backend manually.

### Construction order (REQ-2)

`HybridBackend::new` constructs `TcpBackend::new(rank, world_size,
addr)` first. TCP rendezvous failure (port already bound, network
unreachable, peer rank announcing the wrong identity) surfaces
through `DistributedError::Io` / `InvalidRank` before any CUDA
context allocation; this matters because NCCL init triggers a
non-trivial cuCtx creation that mutates the calling thread's CUDA
state. Only on TCP success does the constructor invoke
`NcclBackend::new(rank, world_size, unique_id)`. The `unique_id` is a
caller-supplied `NcclUniqueId` (re-exported at `lib.rs`) that
every rank must hold the same bytes of; the standard pattern is for
rank 0 to broadcast the unique ID through the TCP backend before
calling `new`.

### Accessor surface (REQ-3 / REQ-4)

The `.nccl()` accessor returns `&NcclBackend` so callers of
`crate::nccl_collective::nccl_allreduce(buf, hybrid.nccl(), ...)` can
reach the GPU-native fast path. The `.tcp()` accessor is symmetric.
`synchronize_nccl` forwards to `NcclBackend::synchronize` (the CUDA
stream sync) so the hybrid backend exposes a single
`hybrid.synchronize_nccl()?` flush point rather than requiring
callers to thread `.nccl()` themselves for the common sync case.

### Backend-trait delegation (REQ-5)

`impl Backend for HybridBackend` at line 80 forwards every method
to `self.tcp`. The deliberate choice (`barrier()` calls
`self.tcp.barrier()`, not `self.nccl.barrier()`) is documented at
lines 106-108: NCCL barrier requires a configured CUDA stream and a
healthy GPU; the TCP barrier is the safer default because the
collective layer's barrier semantics are "every rank has reached this
point in the host code," which TCP satisfies cheaply.

### Consumer sites (production, non-test)

- `ferrotorch-distributed/src/lib.rs` — re-exports
  `HybridBackend` under the `nccl` feature gate.
- `ferrotorch/src/lib.rs` — `pub use ferrotorch_distributed::*;`
  exposes `HybridBackend` to workspace users when both crates are
  compiled with the `nccl` feature.
- The `Backend`-trait delegation means every collective in
  `crate::collective::*` and every P2P op in `crate::p2p::*` works
  against a `HybridBackend` without recompiling: pass
  `hybrid.tcp()` or `&hybrid` directly where `&dyn Backend` is
  expected.

The hybrid setup has no in-tree non-test consumer beyond the
re-export — workspace test code consumes it via the crate-root
re-export in feature-gated builds. The re-export at
`ferrotorch-distributed/src/lib.rs` is the production consumer
surface; user training scripts (out of tree) compose
`HybridBackend::new` directly with their TCP rendezvous addr + NCCL
unique ID.

## Parity contract

No parity-sweep ops in the route (`parity_ops = []`). The contract is
the PyTorch behavior of `dist.init_process_group(backend="nccl",
...)`:

- PyTorch co-instantiates a TCPStore plus an NCCL communicator on
  every rank; collectives (`all_reduce`, `broadcast`, etc.) dispatch
  to NCCL; object collectives + `monitored_barrier` use the
  TCPStore-backed Gloo fallback.
- ferrotorch's `HybridBackend` co-instantiates a `TcpBackend` plus an
  `NcclBackend`; the `Backend`-trait delegation uses TCP; explicit
  NCCL APIs go through `.nccl()`. Collectives that should be NCCL-
  accelerated call `crate::nccl_collective::*` directly with
  `hybrid.nccl()`; collectives that don't need GPU acceleration call
  `crate::collective::*` with `&hybrid` as the trait object.

## Verification

- This file's `#[cfg(test)] mod tests` block is absent in the
  current shipping commit — the module is feature-gated and CI
  configurations that exercise `nccl` are CUDA-host-only.
- `cargo check --features nccl -p ferrotorch-distributed` — module
  compiles.
- `cargo clippy -p ferrotorch-distributed --features nccl --
  -D warnings` — pedantic baseline holds.
- Behavior is exercised indirectly when the parent NCCL backend test
  suite runs on CUDA-capable hosts; the hybrid backend's TCP side is
  exercised by the same battery of TCP tests that cover
  `backend.rs::TcpBackend`.
- Parity-sweep: no ops; integer grep count is 0 by construction.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct HybridBackend` in `ferrotorch-distributed/src/hybrid_backend.rs` with `tcp: TcpBackend` + `nccl: NcclBackend` fields; non-test consumer: `ferrotorch-distributed/src/lib.rs` re-exports under `#[cfg(feature = "nccl")]`. |
| REQ-2 | SHIPPED | impl: `pub fn new` in `ferrotorch-distributed/src/hybrid_backend.rs` constructing TCP first then NCCL; non-test consumer: `ferrotorch-distributed/src/lib.rs` re-export reaches `ferrotorch/src/lib.rs` (`pub use ferrotorch_distributed::*;`). |
| REQ-3 | SHIPPED | impl: `pub fn nccl(&self)` in `ferrotorch-distributed/src/hybrid_backend.rs` and `pub fn tcp(&self)` in `ferrotorch-distributed/src/hybrid_backend.rs`; non-test consumer: documented usage pattern in module docstring at `ferrotorch-distributed/src/hybrid_backend.rs` (`nccl_allreduce(&mut gpu_buffer, hybrid.nccl(), ...)`) is the production call shape that `crate::nccl_collective::*` callers use. |
| REQ-4 | SHIPPED | impl: `pub fn synchronize_nccl(&self)` in `ferrotorch-distributed/src/hybrid_backend.rs` forwarding to `NcclBackend::synchronize`; non-test consumer: re-exported via `HybridBackend` at `ferrotorch-distributed/src/lib.rs`, callable through `ferrotorch/src/lib.rs` meta-crate path. |
| REQ-5 | SHIPPED | impl: `impl Backend for HybridBackend` in `ferrotorch-distributed/src/hybrid_backend.rs` delegating all six methods to `self.tcp`; non-test consumer: every collective in `crate::collective::*` and P2P op in `crate::p2p::*` accepts `&dyn Backend`, so `&hybrid` is the production substitution pattern (e.g. `allreduce(&t, &hybrid, ReduceOp::Sum)`). |
