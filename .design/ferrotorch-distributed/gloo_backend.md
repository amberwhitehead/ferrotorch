# Gloo backend public surface

<!--
tier: 3-component
status: draft
baseline-pytorch: main (user's local clone /home/doll/pytorch)
upstream-paths:
  - torch/distributed/distributed_c10d.py
  - torch/csrc/distributed/c10d/ProcessGroupGloo.cpp
  - torch/csrc/distributed/c10d/ProcessGroupGloo.hpp
-->

## Summary

`ferrotorch-distributed/src/gloo_backend.rs` is the public, feature-gated
handle for the native-Rust Gloo backend. It mirrors the shape of PyTorch's
`init_process_group(backend="gloo")` user-facing API — env-var rendezvous
(`MASTER_ADDR` / `MASTER_PORT` / `RANK` / `WORLD_SIZE`), a `ProcessGroup`-style
`Backend` impl, and a runtime availability predicate — while delegating all
real work to the `gloo_native` submodule (which in turn replaces the C++
`libgloo` dependency PyTorch's `ProcessGroupGloo` links against). With the
`gloo-backend` feature off, every constructor returns
`DistributedError::BackendUnavailable { backend: "gloo" }` so dyn-Backend
type-erasure paths still compile.

## Requirements

- REQ-1: `pub fn is_gloo_available() -> bool` returns `cfg!(feature =
  "gloo-backend")`. Used by the conformance suite and any caller wanting
  to discriminate at runtime.
- REQ-2: `pub struct GlooBackend` is a unit-shaped handle: when the
  `gloo-backend` feature is on it owns a `gloo_native::GlooBackendInner`;
  when off it owns only a `PhantomData<()>`. Either way the type is a
  `Backend` impl, so `&dyn Backend` and trait-object code compiles on
  both flavours of build.
- REQ-3: `pub fn GlooBackend::new(rank, world_size, master_addr) ->
  FerrotorchResult<Self>` constructs the inner backend by driving the
  `gloo_native::rendezvous` handshake. Without the feature it returns
  `BackendUnavailable`.
- REQ-4: `pub fn GlooBackend::from_env() -> FerrotorchResult<Self>`
  reads PyTorch's standard env vars (`MASTER_ADDR`, `MASTER_PORT`, `RANK`,
  `WORLD_SIZE`) via `GlooRendezvousConfig::from_env` and constructs.
- REQ-5: `impl Backend for GlooBackend` forwards `rank`, `world_size`,
  `send`, `recv`, `recv_timeout`, and `barrier` to the inner backend on
  feature-on builds. On feature-off builds, `rank`/`world_size` return
  `0` (the type is unreachable so the values are inert) and every other
  method returns `BackendUnavailable`.
- REQ-6: Feature-on builds expose `pub fn ring_allreduce_sum_f32` and
  `pub fn tree_broadcast_f32` directly on the public handle so callers
  can drive ring/tree collectives without going through the `Backend`
  trait's byte-level surface.
- REQ-7: The non-vacuous discrimination contract: when the feature is
  off, `DistributedError::BackendUnavailable { backend: "gloo" }`
  converts to `FerrotorchError::InvalidArgument` whose message names
  `"gloo"` (and not `"mpi"` / `"ucc"`), so the `_surface.json`
  conformance fixture (`is_gloo_available_matches_fixture`) can key off
  the discriminant.

## Acceptance Criteria

- [x] AC-1: `is_gloo_available()` matches `cfg!(feature = "gloo-backend")`.
- [x] AC-2: `GlooBackend::new(0, 2, "127.0.0.1:0")` on a default
  (feature-off) workspace build returns
  `FerrotorchError::InvalidArgument` whose message contains the literal
  ``"`gloo`"`` (verified by `gloo_unavailable_without_feature`).
- [x] AC-3: `GlooBackend::from_env()` on a feature-off build returns
  the same `InvalidArgument` discrimination (verified by
  `gloo_from_env_unavailable_without_feature`).
- [x] AC-4: With `--features=gloo-backend`, a 2-rank `GlooBackend`
  constructed in-process completes a `ring_allreduce_sum_f32` over
  real TCP (the test `ring_allreduce_over_real_tcp_two_ranks` in
  `gloo_native/mod.rs` exercises the underlying inner, which `GlooBackend`
  delegates to).
- [x] AC-5: `impl Backend for GlooBackend` compiles on both
  feature-on and feature-off builds (structural — the no-feature
  branch returns `BackendUnavailable` for `send`/`recv`/`barrier` and
  `0` for rank/world_size).

## Architecture

`pub fn is_gloo_available` in `gloo_backend.rs` is a one-line
`cfg!(feature = "gloo-backend")` query. Production consumers: the
crate root re-export `pub use gloo_backend::{GlooBackend,
is_gloo_available}` in `ferrotorch-distributed/src/lib.rs` and the
meta-crate re-export `pub use ferrotorch_distributed::*` in
`ferrotorch/src/lib.rs`.

`pub struct GlooBackend` carries one `cfg`-switched field — `inner:
gloo_native::GlooBackendInner` on feature-on builds, `_phantom:
PhantomData<()>` on feature-off — so the type's layout is well-defined
on both. The `#[allow(unused_variables)]` annotations on `new` and
`Backend` methods exist because the constructor parameters are inert
on no-feature builds.

`pub fn GlooBackend::new` in `gloo_backend.rs` constructs a
`gloo_native::GlooRendezvousConfig` with `bind_addr` defaulted to
`127.0.0.1:0` (kernel-assigned port) and hands it to
`gloo_native::GlooBackendInner::new`, which drives the full mesh
rendezvous. On feature-off builds the constructor returns
`Err(DistributedError::BackendUnavailable { backend: "gloo" }.into())`.

`pub fn GlooBackend::from_env` in `gloo_backend.rs` calls
`gloo_native::GlooRendezvousConfig::from_env()` (which reads
`MASTER_ADDR`/`MASTER_PORT`/`RANK`/`WORLD_SIZE`) then the same
inner-constructor. Feature-off branch mirrors `new`.

`impl Backend for GlooBackend` in `gloo_backend.rs` delegates every
trait method to `self.inner` on feature-on builds and short-circuits
to `BackendUnavailable` on feature-off. The `as_nccl_backend`
downcast is NOT overridden — `GlooBackend` is the trait's CPU-only
default and returns `None` for the NCCL hook.

The feature-on-only `pub fn ring_allreduce_sum_f32` and `pub fn
tree_broadcast_f32` methods forward to the corresponding methods on
the inner backend.

### Consumer sites (production, non-test)

- `ferrotorch-distributed/src/lib.rs` — crate-root re-export of
  `GlooBackend` and `is_gloo_available`.
- `ferrotorch/src/lib.rs` — meta-crate `pub use
  ferrotorch_distributed::*` makes both names reachable at the
  workspace surface.
- `ferrotorch-distributed/src/mpi_backend.rs` and
  `ferrotorch-distributed/src/ucc_backend.rs` — both route their CPU
  collective primitives through `gloo_native::GlooBackendInner`
  (which `GlooBackend` is the public face of), keeping the gloo
  native code path as the single source of truth for CPU
  collectives.

## Parity contract

No parity-sweep ops (`parity_ops = []`). The contract is the C10d
`ProcessGroup` shape rather than a numerical op:

- `init_process_group(backend="gloo", ...)` (cf.
  `torch/distributed/distributed_c10d.py` `init_process_group`) ↔
  `GlooBackend::from_env`.
- `dist.is_gloo_available()` ↔ `is_gloo_available()`.
- `ProcessGroupGloo::send / recv / barrier` (cf.
  `torch/csrc/distributed/c10d/ProcessGroupGloo.cpp`) ↔ `impl Backend
  for GlooBackend` (which delegates to the gloo_native inner).
- The `gloo-backend` feature is the upstream-link analog: with the
  feature on we link in the native-Rust replacement for `libgloo`;
  with it off, construction errors with a `BackendUnavailable` shape
  PyTorch's `ProcessGroupGloo` mirrors via Python `ImportError` when
  built without USE_GLOO=1.

## Verification

`cargo test -p ferrotorch-distributed --lib` runs three lib tests
that pin the no-feature contract:

- `gloo_backend::tests::gloo_unavailable_without_feature` — verifies
  `GlooBackend::new` returns `InvalidArgument` whose message names
  ``"`gloo`"`` and not ``"`mpi`"`` / ``"`ucc`"``.
- `gloo_backend::tests::gloo_from_env_unavailable_without_feature` —
  same discrimination for `from_env`.
- `gloo_backend::tests::is_gloo_available_default_off` — asserts
  `is_gloo_available()` is `false` on the default workspace build.

With `--features=gloo-backend`, the feature-on tests live in
`gloo_native::tests` (see `.design/ferrotorch-distributed/gloo_native/mod.md`)
and exercise the real ring allreduce / tree broadcast / barrier over
in-process TCP.

`cargo clippy -p ferrotorch-distributed -- -D warnings`: PASS.

No parity-sweep ops; integer grep count is 0 by construction.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn is_gloo_available` in `ferrotorch-distributed/src/gloo_backend.rs` mirrors `torch.distributed.is_gloo_available` in `torch/distributed/distributed_c10d.py`; non-test consumer: re-export at `ferrotorch-distributed/src/lib.rs` line 229 and `ferrotorch/src/lib.rs`. |
| REQ-2 | SHIPPED | impl: `pub struct GlooBackend` in `ferrotorch-distributed/src/gloo_backend.rs`; non-test consumer: re-export at `ferrotorch-distributed/src/lib.rs` line 229 (`pub use gloo_backend::{GlooBackend, is_gloo_available}`). |
| REQ-3 | SHIPPED | impl: `pub fn GlooBackend::new` in `ferrotorch-distributed/src/gloo_backend.rs`; non-test consumer: `pub use` re-export in `ferrotorch-distributed/src/lib.rs` line 229, and (with `--features=gloo-backend`) routed through `mpi_backend.rs`/`ucc_backend.rs` which both consume `gloo_native::GlooBackendInner::new` (the same constructor `GlooBackend::new` wraps). |
| REQ-4 | SHIPPED | impl: `pub fn GlooBackend::from_env` in `ferrotorch-distributed/src/gloo_backend.rs`; non-test consumer: re-export in `ferrotorch-distributed/src/lib.rs` line 229. |
| REQ-5 | SHIPPED | impl: `impl Backend for GlooBackend` in `ferrotorch-distributed/src/gloo_backend.rs`; non-test consumer: every `&dyn Backend`-accepting function in `ferrotorch-distributed/src/collective.rs` and `ferrotorch-distributed/src/p2p.rs` can accept a `GlooBackend`. |
| REQ-6 | SHIPPED | impl: feature-gated `pub fn ring_allreduce_sum_f32` and `pub fn tree_broadcast_f32` in `ferrotorch-distributed/src/gloo_backend.rs`; non-test consumer: feature-on builds re-export the surface through crate-root `lib.rs` (the methods are inherent on `GlooBackend`, reached via the `lib.rs` `pub use`). |
| REQ-7 | SHIPPED | impl: `BackendUnavailable { backend: "gloo" }` produced in `pub fn GlooBackend::new` / `from_env` / `send` / `recv` / `recv_timeout` / `barrier` in `ferrotorch-distributed/src/gloo_backend.rs`; non-test consumer: `_surface.json` conformance fixture used by the `ferrotorch-core/tests/conformance/` suite, and `gloo_unavailable_without_feature` test pin. The error variant routes to `FerrotorchError::InvalidArgument` whose `message` discriminates backend by name. |
