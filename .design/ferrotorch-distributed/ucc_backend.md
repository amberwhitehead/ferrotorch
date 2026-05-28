# UCC backend (native-Rust router to gloo_native + NCCL)

<!--
tier: 3-component
status: draft
baseline-pytorch: main (user's local clone /home/doll/pytorch)
upstream-paths:
  - torch/distributed/distributed_c10d.py
  - torch/csrc/distributed/c10d/ProcessGroupUCC.cpp
  - torch/csrc/distributed/c10d/ProcessGroupUCC.hpp
-->

## Summary

`ferrotorch-distributed/src/ucc_backend.rs` is the feature-gated
public handle for the native-Rust UCC backend (#1134). UCC is
openucx's unified-collective layer that routes between multiple
transports based on tensor location; ferrotorch's UCC is a
**pure-Rust router** that delegates CPU collectives to
`gloo_native::GlooBackendInner` and GPU collectives (when both
`ucc-native-gpu` and `with_nccl` are wired) to
`gpu_collective::gpu_allreduce` / `gpu_broadcast` via an attached
NCCL communicator. No `libucc.so` link, no `ucc-sys`. Without
the feature, every constructor returns
`DistributedError::BackendUnavailable { backend: "ucc" }`.

## Requirements

- REQ-1: `pub fn is_ucc_available() -> bool` returns
  `cfg!(any(feature = "ucc-native", feature = "ucc-backend"))`
  — the historical `ucc-backend` alias resolves to the same
  code path as `ucc-native`.
- REQ-2: `pub struct UccBackend` carries `cpu_inner:
  gloo_native::GlooBackendInner` (feature-on) and `gpu_inner:
  Mutex<Option<Arc<NcclBackend>>>` (only with the `nccl`
  feature chain). Hand-rolled `impl fmt::Debug` avoids leaking
  the opaque NCCL pointer; it reports rank / world_size /
  `nccl_attached: bool` only.
- REQ-3: `pub fn UccBackend::new(rank, world_size, master_addr)
  -> FerrotorchResult<Self>` builds a `GlooRendezvousConfig`
  (with `bind_addr` defaulted to `127.0.0.1:0`) and invokes
  `GlooBackendInner::new`. Feature-off returns
  `BackendUnavailable { backend: "ucc" }`.
- REQ-4: `pub fn UccBackend::from_env() -> FerrotorchResult<Self>`
  reads PyTorch's standard env vars
  (`MASTER_ADDR`/`MASTER_PORT`/`RANK`/`WORLD_SIZE`) via
  `GlooRendezvousConfig::from_env`.
- REQ-5: `pub fn UccBackend::with_nccl(&self, nccl: Arc<NcclBackend>)
  -> FerrotorchResult<()>` validates that the NCCL backend's rank
  / world_size match the CPU backend's, then stores in
  `gpu_inner`. Compiled under the `nccl` feature.
- REQ-6: Feature-on inherent CPU methods `allreduce_sum_f32` /
  `broadcast_f32` forward to the gloo_native primitives.
- REQ-7: Feature-on inherent GPU methods `gpu_allreduce` /
  `gpu_broadcast` (under `#[cfg(feature = "gpu")]`) check the
  `nccl` feature: if on AND the communicator is attached,
  dispatch through `gpu_collective::gpu_allreduce` /
  `gpu_broadcast`; otherwise return `UnsupportedOp` with a
  message naming the `ucc-native-gpu` upgrade or the missing
  `with_nccl` call.
- REQ-8: `impl Backend for UccBackend` delegates every method
  to `cpu_inner` on feature-on builds. Returns
  `BackendUnavailable` for `send`/`recv`/`barrier` on
  feature-off; returns `0` for `rank`/`world_size`.
- REQ-9: Feature-off non-vacuous discrimination: error message
  contains ``"`ucc`"`` (NOT `"gloo"` / `"mpi"`).

## Acceptance Criteria

- [x] AC-1: `is_ucc_available()` matches `cfg!(any(feature =
  "ucc-native", feature = "ucc-backend"))`.
- [x] AC-2: `UccBackend::new(0, 2, "127.0.0.1:0")` on a default
  (feature-off) build returns `InvalidArgument` containing
  ``"`ucc`"`` (verified by `ucc_unavailable_without_feature`).
- [x] AC-3: `UccBackend::from_env()` on feature-off has the
  same discrimination (`ucc_from_env_unavailable_without_feature`).
- [x] AC-4: 2-rank CPU allreduce over real TCP (with
  `--features=ucc-native`) returns elementwise sum (verified
  by `ucc_native_cpu_allreduce_via_gloo_two_ranks`).
- [x] AC-5: 3-rank CPU broadcast + barrier sequence (with
  `--features=ucc-native`) verified by
  `ucc_native_cpu_broadcast_and_barrier_three_ranks`.
- [x] AC-6: GPU entry points on `--features=ucc-native,gpu`
  (without nccl) return `UnsupportedOp` naming
  `ucc-native-gpu`; verified by
  `ucc_native_gpu_routing_returns_error_without_nccl_feature`.

## Architecture

The lint-baseline is the same as the rest of the crate. The
`UccBackend` struct has a hand-rolled `Debug` (rather than
derived) because the optional `gpu_inner: Mutex<Option<Arc<NcclBackend>>>`
field holds an opaque NCCL pointer that should NOT surface in
formatted output. The hand-rolled impl reports `rank`,
`world_size`, and a `nccl_attached: bool` flag from probing the
mutex.

`pub fn UccBackend::new` constructs the rendezvous config
inline (with `127.0.0.1:0` default `bind_addr`) and invokes
`GlooBackendInner::new`. On `--features=nccl`, the `gpu_inner`
slot is initialised to `Mutex::new(None)` — the caller is
expected to call `with_nccl` later if they want the GPU fast
path.

`pub fn UccBackend::with_nccl` is the only mutator on
`UccBackend`. It validates rank consistency (`cpu_rank ==
nccl_rank`, errors with `InvalidRank` otherwise) and
world_size consistency, then takes the mutex and stores
`Some(nccl_arc)`. The mutex protects against concurrent
`gpu_*` calls also probing the slot.

`pub fn UccBackend::gpu_allreduce` / `gpu_broadcast` are
gated by `#[cfg(feature = "gpu")]` (they take a `GpuTensor`
parameter that only exists with the `gpu` feature). Inside,
the `#[cfg(feature = "nccl")]` branch probes the mutex and
returns `UnsupportedOp` if the slot is `None` (message names
`with_nccl`); otherwise it dispatches through
`gpu_collective::gpu_allreduce` / `gpu_broadcast` using the
attached NCCL backend. The `#[cfg(not(feature = "nccl"))]`
branch returns `UnsupportedOp` with a message naming the
`ucc-native-gpu` upgrade.

`impl Backend for UccBackend` forwards every method to
`cpu_inner` on feature-on builds. The `as_nccl_backend` hook
is NOT overridden — `UccBackend` returns `None` from the
default impl. The rationale: even when an NCCL communicator
is attached via `with_nccl`, the `Backend` trait is CPU-
oriented (byte-level send/recv) and the NCCL communicator
provides GPU collectives. Callers wanting the NCCL fast path
explicitly invoke `UccBackend::gpu_allreduce` rather than the
trait surface.

### Consumer sites (production, non-test)

- `ferrotorch-distributed/src/lib.rs` — `pub use
  ucc_backend::{UccBackend, is_ucc_available};` at line 235.
- `ferrotorch/src/lib.rs` — meta-crate re-export reaches
  every workspace user.

## Parity contract

No parity-sweep ops. The contract is the C10d
`ProcessGroupUCC` shape:

- `init_process_group(backend="ucc")` ↔ `UccBackend::from_env`.
- `ProcessGroupUCC::allreduce` on a CPU tensor ↔
  `UccBackend::allreduce_sum_f32` (routes through gloo_native
  ring allreduce).
- `ProcessGroupUCC::allreduce` on a GPU tensor ↔
  `UccBackend::gpu_allreduce` (routes through NCCL when
  `ucc-native-gpu` is on + `with_nccl` was called).
- `ProcessGroupUCC::broadcast` (CPU) ↔
  `UccBackend::broadcast_f32`.
- `ProcessGroupUCC::broadcast` (GPU) ↔
  `UccBackend::gpu_broadcast`.
- `ProcessGroupUCC::barrier` ↔ `Backend::barrier` (delegates
  to gloo_native ring barrier).
- `Backend.send / recv` (byte-level CPU P2P) ↔
  `Backend::send` / `recv` (delegates to gloo_native
  full-mesh TCP).

## Verification

`cargo test -p ferrotorch-distributed --lib` runs three lib
tests on default builds:

- `ucc_unavailable_without_feature` — discriminates the
  feature-off error.
- `ucc_from_env_unavailable_without_feature` — same for
  `from_env`.
- `is_ucc_available_default_off` — predicate is false.

With `--features=ucc-native`, two additional tests run:

- `ucc_native_cpu_allreduce_via_gloo_two_ranks` — 2-rank
  CPU allreduce sum.
- `ucc_native_cpu_broadcast_and_barrier_three_ranks` —
  3-rank broadcast + barrier.

With `--features=ucc-native,gpu` (without nccl) the
`ucc_native_gpu_routing_returns_error_without_nccl_feature`
test runs (uses real `GpuTensor` construction but only
verifies the error path).

With `--features=ucc-native-gpu` (which implies nccl), the
hardware-gated `ucc_native_gpu_allreduce_via_nccl_single_rank`
test is `#[ignore]`'d.

No parity-sweep ops; integer grep count is 0 by construction.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn is_ucc_available` in `ferrotorch-distributed/src/ucc_backend.rs`; non-test consumer: `pub use ucc_backend::{UccBackend, is_ucc_available};` in `ferrotorch-distributed/src/lib.rs` line 235. |
| REQ-2 | SHIPPED | impl: `pub struct UccBackend` + hand-rolled `impl Debug` in `ferrotorch-distributed/src/ucc_backend.rs`; non-test consumer: re-export at `ferrotorch-distributed/src/lib.rs` line 235. |
| REQ-3 | SHIPPED | impl: `pub fn UccBackend::new` in `ferrotorch-distributed/src/ucc_backend.rs`; non-test consumer: re-export at `ferrotorch-distributed/src/lib.rs` line 235. |
| REQ-4 | SHIPPED | impl: `pub fn UccBackend::from_env` in `ferrotorch-distributed/src/ucc_backend.rs`; non-test consumer: re-export at `ferrotorch-distributed/src/lib.rs` line 235. |
| REQ-5 | SHIPPED | impl: `pub fn UccBackend::with_nccl` (under `#[cfg(feature = "nccl")]`) in `ferrotorch-distributed/src/ucc_backend.rs`; non-test consumer: the inherent `gpu_allreduce` / `gpu_broadcast` methods (same file) read the slot it populates; reachable through the `lib.rs` re-export by any consumer building with `ucc-native-gpu`. |
| REQ-6 | SHIPPED | impl: `pub fn UccBackend::allreduce_sum_f32` and `pub fn UccBackend::broadcast_f32` in `ferrotorch-distributed/src/ucc_backend.rs`; non-test consumer: feature-on builds reach them through the `lib.rs` re-export. |
| REQ-7 | SHIPPED | impl: `pub fn UccBackend::gpu_allreduce` and `pub fn UccBackend::gpu_broadcast` in `ferrotorch-distributed/src/ucc_backend.rs`; non-test consumer: feature-on builds reach them through the `lib.rs` re-export; the methods invoke `gpu_collective::gpu_allreduce` / `gpu_broadcast` from `ferrotorch-distributed/src/gpu_collective.rs`. |
| REQ-8 | SHIPPED | impl: `impl Backend for UccBackend` in `ferrotorch-distributed/src/ucc_backend.rs`; non-test consumer: every `&dyn Backend`-accepting function in `ferrotorch-distributed/src/collective.rs` / `p2p.rs` can accept `UccBackend`. |
| REQ-9 | SHIPPED | impl: `BackendUnavailable { backend: "ucc" }` raised in every feature-off branch in `ferrotorch-distributed/src/ucc_backend.rs`; non-test consumer: `_surface.json` conformance fixture plus `ucc_unavailable_without_feature` (in-file). |
