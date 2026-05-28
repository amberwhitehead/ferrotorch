# gloo_native::error — internal Result alias

<!--
tier: 3-component
status: draft
baseline-pytorch: main (user's local clone /home/doll/pytorch)
upstream-paths:
  - torch/csrc/distributed/c10d/ProcessGroupGloo.hpp
  - torch/csrc/distributed/c10d/exception.h
-->

## Summary

`ferrotorch-distributed/src/gloo_native/error.rs` is a 17-line utility
module that introduces a single internal type alias — `GlooResult<T> =
Result<T, DistributedError>` — used throughout the `gloo_native::`
sub-modules. It deliberately re-uses the workspace-wide
`DistributedError` taxonomy rather than introducing a parallel
`GlooError` enum: every failure mode the native backend can surface
(`Io`, `SizeMismatch`, `InvalidRank`, `LockPoisoned`, `Timeout`,
`NoConnection`, `SelfSend`, `InvalidWorldSize`) already has a variant
on `DistributedError`, and a parallel taxonomy would force callers
to write match arms twice.

## Requirements

- REQ-1: `pub(super) type GlooResult<T> = Result<T, DistributedError>`
  is declared at module scope. The visibility is `pub(super)` so the
  `gloo_native::` sub-modules (`transport`, `connect`, `collectives`,
  and the root `mod.rs`) can use it but external crates cannot.
- REQ-2: The module re-uses `crate::error::DistributedError` (does
  NOT introduce a `GlooError` enum). This is a deliberate design
  decision documented in the module's `//!` doc-comment.

## Acceptance Criteria

- [x] AC-1: `pub(super) type GlooResult<T>` is declared in
  `gloo_native/error.rs` and imported as `use self::error::GlooResult`
  from `gloo_native/mod.rs`.
- [x] AC-2: No `GlooError` enum exists anywhere in the crate
  (`grep -rn "enum GlooError" ferrotorch-distributed/src/`
  returns no matches).
- [x] AC-3: Module compiles with `--features=gloo-backend` (it's
  inside the `pub(crate) mod gloo_native` that's `cfg`-gated on
  the feature) and `cargo clippy -- -D warnings` is clean.

## Architecture

The file is a 5-line type declaration plus 12 lines of doc comment.
There is no function body, no struct, no enum. The only export is
the `pub(super)` type alias.

`pub(super) type GlooResult<T> = Result<T, DistributedError>` is
the single source of truth for the `gloo_native::` sub-modules'
result type. It exists solely so the rest of the module can read
`-> GlooResult<T>` instead of the longer
`Result<T, DistributedError>`. Saving 27 characters per signature
is significant in the `connect.rs` (~25 signatures) and
`transport.rs` (~5 signatures) and `collectives.rs` (~10
signatures) functions.

The visibility is `pub(super)`, not `pub` — exactly the right
shape for an internal type that needs to cross sub-module
boundaries but not leak to outside crates.

### Consumer sites (production, non-test)

- `ferrotorch-distributed/src/gloo_native/transport.rs` — every
  helper signature returns `GlooResult<_>`.
- `ferrotorch-distributed/src/gloo_native/connect.rs` — every
  rendezvous helper returns `GlooResult<_>`.
- `ferrotorch-distributed/src/gloo_native/collectives.rs` — the
  `RingTransport` trait methods and `ring_allreduce_sum_f32_bytes`
  / `tree_broadcast_f32_bytes` / `ring_barrier` all return
  `GlooResult<_>`.
- `ferrotorch-distributed/src/gloo_native/mod.rs` — `pub fn
  GlooBackendInner::new` returns `GlooResult<Self>`; the private
  `send_inner` / `recv_inner` helpers return `GlooResult<()>`.

## Parity contract

No parity-sweep ops. The contract is the design decision that
`gloo_native::` uses the workspace `DistributedError` taxonomy
rather than a Gloo-private enum — this matches PyTorch's
`ProcessGroupGloo`, which throws C++ exceptions of the same
`torch::distributed::c10d::Error` type that `ProcessGroupNCCL` /
`ProcessGroupMPI` / `ProcessGroupUCC` throw. There is no separate
`GlooError`-shaped C++ class in upstream.

## Verification

No unit tests in this file (a type alias has nothing to test).
The alias is exercised structurally by every test in the other
`gloo_native::` sub-modules; if the alias were missing or
ill-typed, the whole `gloo_native::` module would fail to compile.

`cargo clippy -p ferrotorch-distributed --features gloo-backend
-- -D warnings`: PASS.

No parity-sweep ops; integer grep count is 0 by construction.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub(super) type GlooResult<T> = Result<T, DistributedError>` in `ferrotorch-distributed/src/gloo_native/error.rs`; non-test consumer: `ferrotorch-distributed/src/gloo_native/mod.rs` (`use self::error::GlooResult;` + every internal helper returns the alias), `gloo_native/transport.rs`, `gloo_native/connect.rs`, `gloo_native/collectives.rs`. |
| REQ-2 | SHIPPED | impl: the module file's doc comment in `ferrotorch-distributed/src/gloo_native/error.rs` documents the re-use decision; no `GlooError` enum exists in `ferrotorch-distributed/src/`; non-test consumer: every `gloo_native::*` error path produces a `DistributedError` variant directly (e.g., `DistributedError::Io`, `DistributedError::SizeMismatch`, `DistributedError::LockPoisoned` in `gloo_native/transport.rs` / `connect.rs` / `mod.rs`). |
