# ferrotorch-distributed crate surface

<!--
tier: 3-component
status: draft
baseline-pytorch: main (user's local clone /home/doll/pytorch)
upstream-paths:
  - torch/distributed/__init__.py
  - torch/distributed/distributed_c10d.py
-->

## Summary

`ferrotorch-distributed/src/lib.rs` declares the crate-wide lint
baseline, lists every public module, and re-exports the
crate-root convenience surface. It is the single entry point through
which `ferrotorch/src/lib.rs` ingests the distributed subsystem and
through which user code reaches `Backend`, `allreduce`, `DDP`, `FSDP`,
`DTensor`, `DeviceMesh`, `Pipeline`, and the feature-gated NCCL /
GLOO / MPI / UCC bindings. Mirrors the role of
`torch/distributed/__init__.py`, which exports the same high-level
names.

## Requirements

- REQ-1: Workspace-standard lint configuration: `clippy::all` +
  `clippy::pedantic` warn, `rust_2018_idioms` deny, with per-lint
  `#[allow(..., reason = "...")]` overrides for FFI-pattern lints
  (`cast_possible_truncation`, `cast_sign_loss`, …) and rustdoc-pass
  lints held at `allow` while the workspace-wide rustdoc backfill is
  tracked separately. The baseline mirrors
  `ferrotorch-gpu`/`-jit`/`-cubecl`/`-xpu` lib.rs.
- REQ-2: Public module surface declaring every routed file:
  `async_collective`, `backend`, `checkpoint`, `collective`, `ddp`,
  `device_mesh`, `dtensor`, `error`, `fsdp`, `gloo_backend`,
  `mpi_backend`, `p2p`, `pipeline`, `rpc`, `sync_batch_norm`,
  `ucc_backend`, plus the feature-gated `gpu_collective` (`gpu`),
  `hybrid_backend` / `nccl_backend` / `nccl_collective` / `nccl_sys`
  (`nccl`), and the crate-private `gloo_native` (`gloo-backend`).
- REQ-3: Crate-root convenience re-exports mirroring
  `torch.distributed.*`: `Backend`, `SimulatedBackend`, `TcpBackend`,
  `SubBackend`, `allreduce`, `broadcast`, `all_gather`,
  `reduce_scatter`, `reduce_scatter_tensor`, `all_to_all`,
  `all_to_all_single_uneven`, `barrier`, `*_with_timeout` variants,
  `ReduceOp`, `DEFAULT_COLLECTIVE_TIMEOUT`, `DDP`, `FSDP`, `DeviceMesh`,
  `DTensor`, `Placement`, `send`, `recv`, `recv_with_timeout`,
  `recv_into`, `recv_into_with_timeout`, `sendrecv`, `RpcAgent`,
  `RpcError`, `TcpRpcBackend`, `Pipeline`, `PipelineSchedule`,
  `PendingCollective`, `async_all_gather`, `async_reduce_scatter`,
  `SyncBatchNorm2d`, `DistributedError`, and feature-gated NCCL /
  GLOO / MPI / UCC handles.
- REQ-4: Crate-level documentation in the top-of-file `//!` block
  pointing the reader at each subsystem's entry point. Used by
  `cargo doc` to produce the crate landing page that
  `ferrotorch/src/lib.rs` users hit when they click through.

## Acceptance Criteria

- [x] AC-1: `#![warn(clippy::all, clippy::pedantic)]` and
  `#![deny(rust_2018_idioms)]` are set at the crate root with all
  pedantic allows carrying inline rationale comments.
- [x] AC-2: Every routed `.rs` file under
  `ferrotorch-distributed/src/` (excluding `gloo_native` submodule
  files, which are gated behind `pub(crate) mod gloo_native`) appears
  as a `pub mod` declaration matched to its cargo feature gate.
- [x] AC-3: `pub use` re-exports cover the crate-root convenience
  surface listed in REQ-3.
- [x] AC-4: `cargo doc -p ferrotorch-distributed` builds without
  warnings (the `missing_docs` / `missing_debug_implementations`
  pair is held at `allow` per the lint baseline; documentation
  expansion is tracked separately).

## Architecture

### Lint baseline (REQ-1)

Lines 1-93 set the crate-wide warn/deny configuration. The header
documents `unsafe_code` deliberately not being denied: this crate calls
into NCCL via raw FFI (`nccl_sys`), uses `dlopen`/`dlsym`/`transmute`
to load CUDA stream symbols without a compile-time CUDA dependency
(`nccl_backend`), and performs byte-reinterpret tensor I/O
(`checkpoint`, `pipeline`). Per-block `// SAFETY:` substantiation lives
at each `unsafe { ... }` site. The pedantic allow-list at lines 22-93
gives a one-line justification for each suppressed lint (cast hygiene
in FFI rank/world-size juggling; `must_use_candidate` churn; manual
`Debug` impls that intentionally omit non-`Debug` fields like
`Mutex<NcclComm>` and `Arc<dyn Backend>`; etc.). Mirrors the established
ferrotorch-gpu / ferrotorch-jit pattern — diverging unilaterally for a
leaf crate would be a Step-4 architectural change.

### Public module surface (REQ-2)

Lines 172-201 declare every `pub mod` matched to its feature gate:

- Unconditional modules (16): `async_collective`, `backend`,
  `checkpoint`, `collective`, `ddp`, `device_mesh`, `dtensor`, `error`,
  `fsdp`, `gloo_backend`, `mpi_backend`, `p2p`, `pipeline`, `rpc`,
  `sync_batch_norm`, `ucc_backend`.
- `gpu_collective` — gated on `gpu`.
- `hybrid_backend`, `nccl_backend`, `nccl_collective`, `nccl_sys` —
  gated on `nccl`.
- `gloo_native` is `pub(crate)` and gated on `gloo-backend`; it's the
  ring-allreduce / tree-broadcast primitive set the public
  `gloo_backend` delegates to.

### Crate-root convenience surface (REQ-3)

Lines 204-238 expose the cherry-picked names through `pub use ...`:

- Backends: `Backend`, `SimulatedBackend`, `TcpBackend`, `SubBackend`.
- Sync collectives: `ReduceOp`, `DEFAULT_COLLECTIVE_TIMEOUT`,
  `allreduce`, `allreduce_with_timeout`, `all_gather`,
  `all_gather_with_timeout`, `reduce_scatter`,
  `reduce_scatter_with_timeout`, `reduce_scatter_tensor`,
  `broadcast`, `barrier`, `all_to_all`, `all_to_all_with_timeout`,
  `all_to_all_single_uneven`.
- Async collectives: `PendingCollective`, `async_all_gather`,
  `async_reduce_scatter`.
- High-level wrappers: `DDP`, `FSDP`, `DeviceMesh`, `DTensor`,
  `Placement`, `Pipeline`, `PipelineSchedule`, `SyncBatchNorm2d`.
- P2P: `send`, `recv`, `recv_with_timeout`, `recv_into`,
  `recv_into_with_timeout`, `sendrecv`.
- RPC: `RpcAgent`, `RpcError`, `TcpRpcBackend`.
- Checkpointing: `AsyncCheckpointer`, `CheckpointFuture`,
  `DistCheckpointError`, `DistributedCheckpoint`, `ShardMetadata`,
  `TensorShardSpec`, `flat_shard_metadata`, `load_distributed`,
  `reshard`, `save_distributed`.
- Backend skeletons: `GlooBackend`, `is_gloo_available`,
  `MpiBackend`, `is_mpi_available`, `UccBackend`, `is_ucc_available`.
- Errors: `DistributedError`.
- Feature-gated: `HybridBackend` (`nccl`), `NcclBackend` /
  `is_nccl_available` / `nccl_*` collective fns / `NcclUniqueId`
  (`nccl`), `gpu_allreduce` / `gpu_broadcast` (`gpu`).

### Crate-level documentation (REQ-4)

Lines 95-170 are the `//!` block introducing each subsystem with a
one-paragraph blurb and a `Quick start` doctest stub. Mirrors the
PyTorch-side `torch/distributed/__init__.py` docstring at lines 1-50.

### Consumer sites (production, non-test)

- `ferrotorch/src/lib.rs` — `pub use ferrotorch_distributed::*;`
  is the workspace meta-crate's flat re-export. Every user-facing
  ferrotorch project pulls the distributed surface via this path.
- Internal to the crate: every `pub mod` declared at lines 172-201 is
  loaded by sibling modules (`crate::backend`, `crate::collective`,
  etc.) and indirectly by every test file in `tests/`.

## Parity contract

This file is a vocabulary / surface declaration — no parity-sweep ops
are declared in the route (`parity_ops = []`). The contract it
implements is "every name `torch.distributed` exposes through its
`__init__.py` has a ferrotorch-side equivalent reachable through this
crate-root re-export." Where ferrotorch lacks a PyTorch name, the gap
is tracked at the corresponding module's design doc:

- `dist.init_process_group` — ferrotorch uses backend-specific
  constructors (`TcpBackend::new`, `SimulatedBackend::create_group`)
  rather than a unified init. R-DEV-4 deviation (no global mutable
  process-group state; explicit ownership).
- `dist.gather` / `dist.scatter` (root-only) — not implemented;
  `all_gather` + slice on root is the workaround. See
  `collective.md`.
- `dist.monitored_barrier`, `dist.send_object_list`,
  `dist.recv_object_list` — Python-pickle-based object collectives,
  out of scope.

## Verification

- This file has no `#[cfg(test)] mod tests` block; its
  acceptance is structural (the module list is the contract).
- `cargo check -p ferrotorch-distributed` — every `pub mod`
  declaration resolves.
- `cargo clippy -p ferrotorch-distributed -- -D warnings` —
  pedantic baseline holds.
- `cargo test -p ferrotorch-distributed --lib` — every submodule's
  unit tests link against this crate root.
- Parity-sweep: no ops; integer grep count is 0 by construction.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: lint header in `ferrotorch-distributed/src/lib.rs` mirroring `ferrotorch-gpu/src/lib.rs` baseline; non-test consumer: every other `.rs` file in the crate inherits the lint policy, and `cargo clippy -p ferrotorch-distributed -- -D warnings` is green. |
| REQ-2 | SHIPPED | impl: `pub mod` block in `ferrotorch-distributed/src/lib.rs` declares all 16 unconditional modules and the 5 feature-gated modules; non-test consumer: every sibling `crate::<mod>::` path in `backend.rs`, `collective.rs`, `fsdp.rs`, etc. resolves through this surface. |
| REQ-3 | SHIPPED | impl: `pub use` block in `ferrotorch-distributed/src/lib.rs` (re-exports 35 symbols from 10 submodules); non-test consumer: `ferrotorch/src/lib.rs` `pub use ferrotorch_distributed::*;` re-exports the entire surface through the workspace meta-crate. |
| REQ-4 | SHIPPED | impl: top-of-file `//!` block in `ferrotorch-distributed/src/lib.rs` enumerating every subsystem with rustdoc links; non-test consumer: `cargo doc -p ferrotorch-distributed` renders the landing page; the docstring is the reference the meta-crate's user-facing doc inherits via `pub use ferrotorch_distributed::*;`. |
