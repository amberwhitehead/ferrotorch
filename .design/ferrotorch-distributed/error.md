# Distributed error taxonomy

<!--
tier: 3-component
status: draft
baseline-pytorch: main (user's local clone /home/doll/pytorch)
upstream-paths:
  - torch/distributed/__init__.py
  - torch/distributed/distributed_c10d.py
-->

## Summary

`ferrotorch-distributed/src/error.rs` defines `DistributedError`, the
non-exhaustive enum every distributed subsystem function returns when
it fails. The enum captures the subset of failure modes that upstream
PyTorch raises through Python exceptions (`ValueError`, `RuntimeError`,
`TimeoutError`, `DistBackendError`) and the C10d C++ layer surfaces
through `c10::Error` subclasses. The conversion `From<DistributedError>
for FerrotorchError` projects every variant into the workspace-wide
error taxonomy so distributed errors flow through `FerrotorchResult<T>`
returns without callers having to match on the inner shape.

## Requirements

- REQ-1: `pub enum DistributedError` with `#[non_exhaustive]` and
  `thiserror::Error` derive, covering at minimum the rank / world-size
  / send-to-self / size-mismatch / I/O / lock-poison / channel-closed
  / unsupported-reduce-op / timeout / no-connection / backend-unavailable
  failure modes that the rest of the crate constructs.
- REQ-2: Every variant carries the diagnostic fields a user needs to
  triage the failure without a debugger (offending rank, observed vs
  expected byte counts, timeout seconds, originating backend name).
- REQ-3: `impl From<DistributedError> for FerrotorchError` projecting
  every variant into `FerrotorchError::InvalidArgument` with the
  `Display`-formatted message, so distributed code can return
  `FerrotorchResult<T>` and `?`-propagate through the workspace error
  type.
- REQ-4: `BackendUnavailable { backend: &'static str }` variant
  returned when a cargo-feature-gated backend (`gloo-backend`,
  `mpi-native`, `ucc-backend`) is invoked without the underlying C
  library available. Mirrors PyTorch's
  `distributed.DistBackendError("Backend X is not available")` shape
  raised in `torch/distributed/distributed_c10d.py` when a missing
  backend is requested.

## Acceptance Criteria

- [x] AC-1: The enum is `#[non_exhaustive]` and derives `Debug` and
  `thiserror::Error`; new variants can be added without breaking
  match-exhaustiveness in downstream code.
- [x] AC-2: Each variant's `#[error("…")]` format string includes the
  carried fields by name (no opaque "operation failed" messages).
- [x] AC-3: The `From<DistributedError> for FerrotorchError`
  conversion compiles and is used by every fallible path in
  `backend.rs`, `collective.rs`, and the backend-impl modules.
- [x] AC-4: Variant set covers every distributed-subsystem failure
  mode actually constructed elsewhere in the crate (grep'd against
  `DistributedError::` constructor sites).

## Architecture

`pub enum DistributedError` is declared in `enum DistributedError` in
`error.rs`. The variants and their roles:

- `InvalidWorldSize { world_size }` — backend constructors (`TcpBackend::new`,
  `SimulatedBackend::create_group`, `SubBackend::new`) reject zero or
  out-of-range world sizes. Mirrors the
  `torch/distributed/distributed_c10d.py` `_check_default_pg` shape.
- `InvalidRank { rank, world_size }` — caller passed a rank outside
  `[0, world_size)`, or a peer rank announced an out-of-range identity
  during the TCP rendezvous.
- `SelfSend { rank }` — explicit guard against `send`/`recv` to the
  same rank. PyTorch's `dist.send` raises `RuntimeError` here.
- `SizeMismatch { expected, got }` — fixed-length protocol fail. Used
  by `Backend::recv` length-prefix check and by `all_gather` /
  `reduce_scatter` divisibility / per-peer length validation.
- `Io { message }` — wraps `std::io::Error` strings; produced by
  `TcpBackend` rendezvous, send/recv, and the gloo-native transport
  layer.
- `LockPoisoned { message }` — wraps `std::sync::PoisonError`
  display from the per-peer `Mutex<TcpStream>` / `Mutex<Sender>` /
  `Mutex<Receiver>` channel-matrix guards. R-CODE-2-safe alternative
  to `.unwrap()` on lock acquisition.
- `ChannelClosed { message }` — wraps `mpsc::SendError` /
  `mpsc::RecvError` from `SimulatedBackend`.
- `UnsupportedOp { message }` — used by `gpu_collective::gpu_allreduce`
  when called without `nccl` feature and without
  `FERROTORCH_ENABLE_GPU_FALLBACK=1`. Also used by NCCL backends for
  reductions NCCL itself does not support.
- `Timeout { seconds }` — every `recv_timeout` path on every backend
  returns this when a `std::io::ErrorKind::WouldBlock` /
  `mpsc::RecvTimeoutError::Timeout` surfaces.
- `NoConnection { rank }` — TcpBackend star topology: non-zero ranks
  do not have direct sockets to each other. Returned when a non-rank-0
  process attempts to `send`/`recv` to/from a peer that isn't rank 0.
- `BackendUnavailable { backend: &'static str }` — feature-gate
  failure (gloo / MPI / UCC / CUDA backend requested but binding not
  compiled in). The `'static str` argument is the documented backend
  name from `Backend` (cf. `torch/distributed/distributed_c10d.py`
  `Backend.GLOO`, `Backend.MPI`, `Backend.NCCL`, `Backend.UCC`).

The `From<DistributedError> for FerrotorchError` conversion at the
bottom of the file maps every variant into
`FerrotorchError::InvalidArgument { message: e.to_string() }`. This
intentionally lossy projection keeps the workspace-wide error type
flat: callers can pattern-match on the original `DistributedError` if
they catch the error before `?`-propagation, but the typical path is
that distributed errors flow through `FerrotorchResult<T>` and surface
as `InvalidArgument` text at the boundary.

### Consumer sites (production, non-test)

- `backend.rs` — `use crate::error::DistributedError;` constructs
  every variant of the enum in the TCP / Simulated / SubBackend impls.
- `collective.rs` — uses `InvalidRank`, `SizeMismatch`, `Timeout`.
- `gloo_backend.rs` — uses `BackendUnavailable`.
- `mpi_backend.rs` — uses `BackendUnavailable`.
- `ucc_backend.rs` — uses `BackendUnavailable`.
- `nccl_backend.rs` — feature-gated; uses `BackendUnavailable`,
  `Io`, `Timeout`.
- `gpu_collective.rs` — uses `UnsupportedOp`, `InvalidRank`.
- `gloo_native/{mod,connect,error,transport,collectives}.rs` — wrap
  rendezvous and ring-allreduce errors as `DistributedError`.
- `ferrotorch/src/lib.rs` — re-exports the enum via the workspace
  meta-crate (`pub use ferrotorch_distributed::*;`) so user code can
  match on the variants.

## Parity contract

This module is a vocabulary file — no parity-sweep ops are declared
in the route (`parity_ops = []`). The contract it implements is the
shape PyTorch's `dist` module uses for exception messages, sourced
from `torch/distributed/distributed_c10d.py`:

- Rank-out-of-range: PyTorch raises `RuntimeError("Invalid rank ...")`.
  ferrotorch returns `DistributedError::InvalidRank`.
- Backend unavailable: PyTorch raises
  `RuntimeError("Distributed package doesn't have <backend> built in")`.
  ferrotorch returns `DistributedError::BackendUnavailable`.
- Timeout: PyTorch raises `DistError("Timed out")`. ferrotorch returns
  `DistributedError::Timeout`.

The user-visible shape is similar; ferrotorch deviates from PyTorch
only in that the return-vs-raise vocabulary is the Rust-native
`Result`. R-DEV-4 / R-DEV-7 justify the deviation.

## Verification

- Tests: this file has no `#[cfg(test)]` block of its own. Each
  variant is exercised by the consumer module's tests:
  - `InvalidWorldSize` / `InvalidRank`: `backend.rs` tests
    `test_invalid_world_size`, `test_send_to_invalid_rank`,
    `test_subbackend_global_rank_not_in_members_is_error`.
  - `Timeout`: covered by `collective.rs` recv-timeout edge tests.
  - `SizeMismatch`: covered by `test_reduce_scatter_indivisible`
    and `test_all_to_all_rejects_uneven_numel`.
  - `BackendUnavailable`: covered by `gloo_backend.rs`,
    `mpi_backend.rs`, `ucc_backend.rs` feature-off tests.
- Lint: `cargo clippy -p ferrotorch-distributed -- -D warnings`
  passes with the crate-level `clippy::pedantic` opt-in (no
  `#[allow]` overrides specific to this file).
- Parity-sweep: no ops; the integer grep count is 0 by construction.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub enum DistributedError` in `ferrotorch-distributed/src/error.rs` with 11 variants; non-test consumers: `ferrotorch-distributed/src/backend.rs`, `ferrotorch-distributed/src/collective.rs`, `ferrotorch-distributed/src/gloo_backend.rs`. |
| REQ-2 | SHIPPED | every variant in `ferrotorch-distributed/src/error.rs` carries diagnostic fields rendered in the `#[error("…")]` format strings; the named-field shape is verified by consumer tests (`backend.rs` `test_invalid_world_size`, `test_send_to_invalid_rank`). |
| REQ-3 | SHIPPED | impl: `impl From<DistributedError> for FerrotorchError` in `ferrotorch-distributed/src/error.rs`; non-test consumer: every `.into()` in `ferrotorch-distributed/src/backend.rs` (e.g. `backend.rs`, `backend.rs`, `backend.rs`, `backend.rs`) and `ferrotorch-distributed/src/collective.rs`. |
| REQ-4 | SHIPPED | impl: `BackendUnavailable { backend: &'static str }` variant in `ferrotorch-distributed/src/error.rs`; non-test consumers: `ferrotorch-distributed/src/gloo_backend.rs` (feature-off construction), `ferrotorch-distributed/src/mpi_backend.rs`, `ferrotorch-distributed/src/ucc_backend.rs`. |
