# MPI-subset backend (native-Rust router to gloo_native)

<!--
tier: 3-component
status: draft
baseline-pytorch: main (user's local clone /home/doll/pytorch)
upstream-paths:
  - torch/distributed/distributed_c10d.py
  - torch/csrc/distributed/c10d/ProcessGroupMPI.cpp
  - torch/csrc/distributed/c10d/ProcessGroupMPI.hpp
-->

## Summary

`ferrotorch-distributed/src/mpi_backend.rs` is the feature-gated
public handle for the native-Rust MPI-subset backend (#1133). Under
`--features=mpi-native`, it delegates every MPI-style collective
(allreduce, broadcast, barrier, send, recv) to
`gloo_native::GlooBackendInner` — no C MPI library link, no `mpicc`,
no `libmpi.so` runtime dependency. The rendezvous accepts the
typical MPI-launcher env vars (`OMPI_COMM_WORLD_RANK`/`_SIZE` for
Open MPI, `PMI_RANK`/`_SIZE` for MPICH+Hydra+SLURM) in priority
order, falling back to PyTorch's `RANK`/`WORLD_SIZE`. Without the
feature, every constructor returns
`DistributedError::BackendUnavailable { backend: "mpi" }`.

## Requirements

- REQ-1: `pub fn is_mpi_available() -> bool` returns
  `cfg!(feature = "mpi-native")`. The legacy `mpi-backend` feature
  name is an alias for `mpi-native` (Cargo dependency feature) so
  either spelling toggles the same code.
- REQ-2: `pub struct MpiBackend` carries either `inner:
  gloo_native::GlooBackendInner` (feature on) or `_phantom:
  PhantomData<()>` (feature off). Mirrors `GlooBackend`'s
  layout.
- REQ-3: `pub fn MpiBackend::new(rank, world_size, master_addr)
  -> FerrotorchResult<Self>` builds a `gloo_native::GlooRendezvousConfig`
  with `bind_addr` defaulted to `127.0.0.1:0` and invokes
  `GlooBackendInner::new`. Feature-off returns
  `BackendUnavailable`.
- REQ-4: `pub fn MpiBackend::from_env() -> FerrotorchResult<Self>`
  reads MPI-launcher env vars in priority order (Open MPI →
  PMI → PyTorch fallback) for `(rank, world_size)`. Always
  reads `MASTER_ADDR` / `MASTER_PORT` for the rendezvous endpoint
  (MPI itself doesn't standardise an out-of-band rendezvous
  address; we ride PyTorch's convention for the native TCP
  rendezvous).
- REQ-5: `impl Backend for MpiBackend` forwards every method to
  the inner gloo_native backend. The `as_nccl_backend` hook is
  NOT overridden (returns `None` from the default impl —
  consistent with `GlooBackend`).
- REQ-6: Feature-on exposes inherent methods `allreduce_sum_f32`
  / `broadcast_f32` that mirror the `MPI_Allreduce` /
  `MPI_Bcast` API shape and forward to
  `gloo_native::GlooBackendInner`'s `ring_allreduce_sum_f32` /
  `tree_broadcast_f32`.
- REQ-7: Feature-off non-vacuous discrimination: every
  constructor returns `BackendUnavailable { backend: "mpi" }`,
  which converts to `FerrotorchError::InvalidArgument { message }`
  whose `message` contains ``"`mpi`"`` (NOT `"gloo"` or `"ucc"`)
  for the `is_mpi_available_matches_fixture` conformance fixture.

## Acceptance Criteria

- [x] AC-1: `is_mpi_available()` matches `cfg!(feature =
  "mpi-native")`.
- [x] AC-2: `MpiBackend::new(0, 2, "127.0.0.1:0")` on a default
  (feature-off) workspace build returns `InvalidArgument`
  whose `message` contains ``"`mpi`"`` (verified by
  `mpi_unavailable_without_feature`).
- [x] AC-3: `MpiBackend::from_env()` on feature-off returns
  the same discrimination (`mpi_from_env_unavailable_without_feature`).
- [x] AC-4: With `--features=mpi-native`, the 2-rank
  `mpi_native_e2e_allreduce_two_ranks` test completes the
  allreduce and verifies the elementwise sum.
- [x] AC-5: With `--features=mpi-native`, the 3-rank
  `mpi_native_e2e_broadcast_and_barrier_three_ranks` test
  exercises both broadcast and barrier in sequence.
- [x] AC-6: The env-var resolution order is OMPI →
  PMI → PyTorch — verified by
  `mpi_native_from_env_prefers_ompi_then_pmi_then_pytorch`.

## Architecture

`pub fn is_mpi_available` queries `cfg!(feature = "mpi-native")`
in `mpi_backend.rs`. Note that the `mpi-backend` alias resolves
to the same `mpi-native` flag at the workspace Cargo feature
graph, so a single `cfg!` covers both spellings.

`pub struct MpiBackend` has the same `cfg`-switched layout as
`GlooBackend` (REQ-2). The choice to keep `inner` as a single
field rather than `Option<GlooBackendInner>` matches the
`GlooBackend` convention — feature-off builds have no
constructor that can reach the `inner`-bearing branch.

`pub fn MpiBackend::new` constructs the rendezvous config
inline (with the `127.0.0.1:0` default `bind_addr`) and passes
it to `GlooBackendInner::new`. The full TCP rendezvous from
`gloo_native::connect::rendezvous` runs.

`pub fn MpiBackend::from_env` is more elaborate: the private
`native::mpi_rendezvous_from_env` helper tries `(OMPI_COMM_WORLD_RANK,
OMPI_COMM_WORLD_SIZE)` first, then `(PMI_RANK, PMI_SIZE)`, then
`(RANK, WORLD_SIZE)` for the `(rank, world_size)` pair. The
priority order matters when an `mpirun` job inherits a stale
`RANK` env var from a previous run — OMPI's vars win.
`MASTER_ADDR` / `MASTER_PORT` are always read separately, since
MPI launchers don't standardise an out-of-band rendezvous
address.

`impl Backend for MpiBackend` delegates `rank` / `world_size` /
`send` / `recv` / `recv_timeout` / `barrier` to
`self.inner.*` on feature-on. Feature-off returns
`BackendUnavailable` for `send`/`recv`/`barrier`, returns `0`
for `rank`/`world_size` (unreachable values, but the trait
must be implementable so type-erased paths compile).

The inherent `allreduce_sum_f32` / `broadcast_f32` methods are
documented under the `MPI_Allreduce` / `MPI_Bcast` names in the
module doc — the API shape mirrors the MPI standard surface
PyTorch's MPI backend exposes.

### Consumer sites (production, non-test)

- `ferrotorch-distributed/src/lib.rs` — `pub use
  mpi_backend::{MpiBackend, is_mpi_available};`.
- `ferrotorch/src/lib.rs` — meta-crate re-export.

## Parity contract

No parity-sweep ops. The contract is the C10d
`ProcessGroupMPI` shape:

- `MPI_Allreduce(buf, op=SUM, dtype=FLOAT)` ↔
  `MpiBackend::allreduce_sum_f32`. Only f32 sum is wired;
  other dtypes/ops would widen the gloo_native primitive.
- `MPI_Bcast(buf, root, dtype=FLOAT)` ↔ `MpiBackend::broadcast_f32`.
- `MPI_Barrier` ↔ `Backend::barrier` (delegates to
  `gloo_native`'s ring barrier).
- `MPI_Send` / `MPI_Recv` ↔ `Backend::send` /
  `Backend::recv` (delegates to gloo_native full-mesh).
- `mpirun -n` rank assignment ↔ `OMPI_COMM_WORLD_RANK`
  / `OMPI_COMM_WORLD_SIZE` env vars.
- `srun --mpi=pmix` rank assignment ↔ `PMI_RANK` / `PMI_SIZE`
  env vars.

## Verification

`cargo test -p ferrotorch-distributed --lib` runs three lib
tests on default builds:

- `mpi_unavailable_without_feature` — discriminates the
  feature-off error variant.
- `mpi_from_env_unavailable_without_feature` — same for
  `from_env`.
- `is_mpi_available_default_off` — predicate is `false`
  on default workspace builds.

With `--features=mpi-native`, three additional tests run:

- `mpi_native_e2e_allreduce_two_ranks` — real TCP, 2 ranks,
  allreduce sum.
- `mpi_native_e2e_broadcast_and_barrier_three_ranks` —
  real TCP, 3 ranks, broadcast + barrier.
- `mpi_native_from_env_prefers_ompi_then_pmi_then_pytorch` —
  env-var resolution order.

No parity-sweep ops; integer grep count is 0 by construction.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn is_mpi_available` in `ferrotorch-distributed/src/mpi_backend.rs`; non-test consumer: `pub use mpi_backend::{MpiBackend, is_mpi_available};` in `ferrotorch-distributed/src/lib.rs` line 230. |
| REQ-2 | SHIPPED | impl: `pub struct MpiBackend` in `ferrotorch-distributed/src/mpi_backend.rs`; non-test consumer: re-export at `ferrotorch-distributed/src/lib.rs` line 230 reaches `ferrotorch/src/lib.rs`. |
| REQ-3 | SHIPPED | impl: `pub fn MpiBackend::new` in `ferrotorch-distributed/src/mpi_backend.rs`; non-test consumer: re-export in `ferrotorch-distributed/src/lib.rs` line 230; feature-on builds reach `gloo_native::GlooBackendInner::new` through it. |
| REQ-4 | SHIPPED | impl: `pub fn MpiBackend::from_env` and `mod native::mpi_rendezvous_from_env` in `ferrotorch-distributed/src/mpi_backend.rs`; non-test consumer: re-export in `ferrotorch-distributed/src/lib.rs` line 230. |
| REQ-5 | SHIPPED | impl: `impl Backend for MpiBackend` in `ferrotorch-distributed/src/mpi_backend.rs`; non-test consumer: every `&dyn Backend`-accepting function in `ferrotorch-distributed/src/collective.rs` / `p2p.rs` can accept `MpiBackend`. |
| REQ-6 | SHIPPED | impl: feature-gated `pub fn MpiBackend::allreduce_sum_f32` and `pub fn MpiBackend::broadcast_f32` in `ferrotorch-distributed/src/mpi_backend.rs`; non-test consumer: feature-on builds use the inherent methods reachable through the `lib.rs` re-export. |
| REQ-7 | SHIPPED | impl: `BackendUnavailable { backend: "mpi" }` raised in every feature-off branch of `pub fn MpiBackend::new`, `from_env`, `send`, `recv`, `recv_timeout`, `barrier` in `ferrotorch-distributed/src/mpi_backend.rs`; non-test consumer: `_surface.json` conformance fixture, plus `mpi_unavailable_without_feature` test (in-file). |
