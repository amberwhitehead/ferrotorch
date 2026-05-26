# DTensor — distributed tensor over a DeviceMesh

<!--
tier: 3-component
status: draft
baseline-pytorch: main (user's local clone /home/doll/pytorch)
upstream-paths:
  - torch/distributed/tensor/_api.py
  - torch/distributed/tensor/placement_types.py
  - torch/distributed/tensor/_redistribute.py
-->

## Summary

`ferrotorch-distributed/src/dtensor.rs` defines `DTensor<T>` and
`Placement`, mirroring `torch.distributed.tensor.DTensor` plus the
`Replicate` / `Shard` / `Partial` placement types. A `DTensor`
represents a logical tensor whose physical storage is sharded or
replicated across the ranks of a `DeviceMesh`; each mesh dimension
carries one `Placement`. The module ships the placement spec, the
local-shard accessor / constructor surface, and a shape-checked
`redistribute` API contract that records the intended target layout.
Cross-rank communication for the sharded↔replicated transitions is
performed by the lower-level `crate::collective::*` ops before/after
`redistribute` lands.

## Requirements

- REQ-1: `pub enum Placement` with variants `Replicate`, `Shard(usize)`,
  `Partial(ReduceOp)`. Derives `Debug + Clone + Copy + PartialEq`.
- REQ-2: Placement predicates: `pub fn is_replicate()`, `is_shard()`,
  `is_partial()`, plus `shard_dim() -> Option<usize>` returning the
  sharded tensor dim for `Shard(d)` and `None` otherwise.
- REQ-3: `pub struct DTensor<T: Float>` carrying `local_tensor:
  Tensor<T>`, `placements: Vec<Placement>` (length matches
  `mesh.ndim()`), `global_shape: Vec<usize>`, and `mesh: DeviceMesh`.
  Derives `Debug + Clone`.
- REQ-4: `pub fn from_local(local, mesh, placements, global_shape)`
  validating `placements.len() == mesh.ndim()` and every
  `Shard(d).d < global_shape.len()`. Both validation failures return
  `FerrotorchError::{ShapeMismatch, InvalidArgument}` with messages.
- REQ-5: `pub fn from_local_replicated(local, mesh)` convenience
  constructor that defaults `placements` to
  `vec![Replicate; mesh.ndim()]` and `global_shape` to
  `local.shape()`.
- REQ-6: Accessors: `pub fn to_local() -> &Tensor<T>`, `pub fn
  shape() -> &[usize]` (returns global shape), `pub fn placements()
  -> &[Placement]`, `pub fn mesh() -> &DeviceMesh`, `pub fn numel()
  -> usize` (global).
- REQ-7: `pub fn redistribute(target_placements)` shape-checking
  API: validates `target.len() == mesh.ndim()` and each
  `Shard(d).d < global_shape.len()`, then updates the
  `placements` field. Documents the supported transitions
  (Replicate↔Replicate, Shard↔Shard same dim, Replicate→Shard,
  Shard→Replicate, Partial→Replicate, Shard(d)→Shard(e)) and the
  collective op the caller should run alongside.

## Acceptance Criteria

- [x] AC-1: `Placement::Replicate.is_replicate()` is `true`;
  `Placement::Shard(2).shard_dim()` is `Some(2)`.
- [x] AC-2: `DTensor::from_local_replicated` produces a DTensor whose
  global shape equals the local shape and every placement is
  `Replicate`.
- [x] AC-3: `from_local` with `placements.len() != mesh.ndim()`
  returns `Err(ShapeMismatch)`.
- [x] AC-4: `from_local` with `Shard(d)` where `d >=
  global_shape.len()` returns `Err(InvalidArgument)`.
- [x] AC-5: `redistribute(vec![Replicate])` from a `Shard(0)` start
  state changes the placement vec accordingly.
- [x] AC-6: `redistribute` with the wrong number of target
  placements returns `Err(ShapeMismatch)`.
- [x] AC-7: `numel()` reports the GLOBAL shape product (not local).

## Architecture

### `Placement` enum (REQ-1 / REQ-2)

`pub enum Placement` carries:

- `Replicate`: every rank in the mesh dim holds a full copy.
- `Shard(usize)`: tensor is split along tensor-dim `d` across mesh
  ranks. The caller is responsible for ensuring even divisibility
  (no auto-padding).
- `Partial(ReduceOp)`: each rank holds an unreduced contribution; a
  pending reduction with `op` will collapse it to `Replicate`.

The three predicates (`is_*`) + `shard_dim()` are convenience
accessors mirroring `Placement.is_replicate()` / `Placement.is_shard()`
/ `Placement.dim` on the upstream `torch.distributed.tensor.Placement`
hierarchy. `Copy` + `PartialEq` are derived so a `&[Placement]` can be
trivially compared between actual and target layouts during
redistribute planning.

### `DTensor<T>` shape (REQ-3 / REQ-6)

`pub struct DTensor<T: Float>` holds the per-rank `local_tensor`
storage + the logical `global_shape` describing the full tensor every
rank conceptually shares + the per-mesh-dim `placements` annotation +
the `mesh` itself. The four accessor methods (`to_local`, `shape`,
`placements`, `mesh`) project the fields read-only; `numel()`
computes the global element count via `global_shape.iter().product()`
with a `.max(1)` floor matching PyTorch's empty-shape convention.

### Constructors (REQ-4 / REQ-5)

`from_local` validates two structural invariants:

1. `placements.len() == mesh.ndim()` (one placement per mesh dim).
2. Every `Shard(d)` placement's `d < global_shape.len()` (the
   sharded tensor dim must exist).

Both checks produce typed errors (`ShapeMismatch` / `InvalidArgument`)
without panicking. `from_local_replicated` is the common case helper:
construct a DTensor whose global shape equals the local shape and
every placement is `Replicate`. Used when a caller has a tensor and
just wants to wrap it as a DTensor over an arbitrary mesh.

### `redistribute` shape-checking contract (REQ-7)

`redistribute(target_placements)` is the API contract for layout
transitions. The method:

1. Validates `target.len() == mesh.ndim()`.
2. Validates each `Shard(d)` in the target has `d <
   global_shape.len()`.
3. Updates the `placements` field.

What `redistribute` does NOT do today:

- It does NOT perform cross-rank communication. The lower-level
  `crate::collective::*` ops (`all_gather` for Shard→Replicate,
  `allreduce` for Partial→Replicate, `all_to_all` for Shard(d)→Shard(e))
  are run by the caller before or after `redistribute`, with the
  `local_tensor` field rebuilt accordingly.
- It does NOT scatter on Replicate→Shard; the caller picks the
  local shard from the full tensor.

This split (placement bookkeeping vs collective dispatch) keeps the
DTensor API testable without real multi-process launches. The
module docstring at lines 16-29 documents the contract explicitly.

### Consumer sites (production, non-test)

- `ferrotorch-distributed/src/lib.rs` — `pub use dtensor::{DTensor,
  Placement};` re-exports both types at the crate root.
- `ferrotorch/src/lib.rs` — meta-crate `pub use
  ferrotorch_distributed::*;` exposes `DTensor` and `Placement` to
  user code.
- `ferrotorch-distributed/src/dtensor.rs` — `use
  crate::collective::ReduceOp;` is the in-tree dependency that lets
  `Placement::Partial(ReduceOp)` reuse the collective layer's
  reduce-op enum.

No in-tree non-test consumer constructs a DTensor outside of the
unit tests in the same file — the type is the boundary API for user
training-script code (and for the planned tensor-parallel module
crates that compose DTensors). R-DEFER-1 grandfathers existing pub
API surface as the public API.

## Parity contract

No parity-sweep ops in the route (`parity_ops = []`). The contract is
the PyTorch `torch.distributed.tensor.DTensor` shape:

- `DTensor.placements` is `tuple[Placement, ...]`: ferrotorch returns
  `&[Placement]`.
- `DTensor.shape` is the global shape: ferrotorch's `shape()` returns
  the same.
- `DTensor.to_local()` returns the per-rank local tensor: matches
  ferrotorch's `to_local()`.
- `DTensor.from_local(local_tensor, device_mesh, placements,
  run_check=False)` matches ferrotorch's
  `from_local(local, mesh, placements, global_shape)`. The deviation:
  PyTorch takes the local tensor's shape and infers the global shape
  by multiplying sharded dims by the corresponding mesh dim size;
  ferrotorch requires the caller to pass `global_shape` explicitly.
  R-DEV-4 deviation: the inference logic depends on
  `torch._C._distributed.Shard.compute_global_shape`, which is C++ and
  not yet ferrotorch's runtime; explicit `global_shape` avoids
  hidden mis-inference.
- `DTensor.redistribute(device_mesh, placements)` matches
  ferrotorch's `redistribute(target_placements)` for the SHAPE side,
  but PyTorch's version also fires the collective. The
  collective-dispatch side of ferrotorch's redistribute is left to
  explicit `crate::collective::*` calls.

## Verification

- `cargo test -p ferrotorch-distributed --lib` runs the
  `#[cfg(test)] mod tests` at lines 220-320 covering:
  - `placement_predicates`,
    `from_local_replicated_uses_local_shape`,
    `from_local_rejects_placement_count_mismatch`,
    `from_local_rejects_oob_shard_dim`,
    `redistribute_updates_placements`,
    `redistribute_rejects_target_count_mismatch`,
    `redistribute_rejects_oob_shard`,
    `numel_uses_global_shape`.
- Conformance fixtures: `ferrotorch-distributed/tests/conformance/fixtures.json`
  pins `DTensor_from_local_valid` and
  `DTensor_from_local_placement_mismatch_error`.
- Lint: `cargo clippy -p ferrotorch-distributed -- -D warnings` PASS.
- Parity-sweep: no ops; integer grep count is 0 by construction.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub enum Placement` in `ferrotorch-distributed/src/dtensor.rs`; non-test consumer: `ferrotorch-distributed/src/dtensor.rs` stores `placements: Vec<Placement>` as a `DTensor` field, and `ferrotorch-distributed/src/lib.rs` re-exports the enum at the crate root. |
| REQ-2 | SHIPPED | impl: `is_replicate` / `is_shard` / `is_partial` / `shard_dim` methods in `ferrotorch-distributed/src/dtensor.rs`; non-test consumer: re-exported via `Placement` at `ferrotorch-distributed/src/lib.rs`, used by `DTensor` callers to inspect layout. |
| REQ-3 | SHIPPED | impl: `pub struct DTensor<T: Float>` in `ferrotorch-distributed/src/dtensor.rs` with `local_tensor` / `placements` / `global_shape` / `mesh` fields; non-test consumer: `ferrotorch-distributed/src/lib.rs` re-exports `DTensor` at the crate root, reached through `ferrotorch/src/lib.rs`. |
| REQ-4 | SHIPPED | impl: `pub fn from_local` in `ferrotorch-distributed/src/dtensor.rs` with placement-count and shard-dim validation at lines 109-131; non-test consumer: `ferrotorch-distributed/src/dtensor.rs` (`from_local_replicated` invokes `Self::from_local`), and the crate-root re-export at `ferrotorch-distributed/src/lib.rs` reaches user training scripts. |
| REQ-5 | SHIPPED | impl: `pub fn from_local_replicated` in `ferrotorch-distributed/src/dtensor.rs`; non-test consumer: crate-root re-export at `ferrotorch-distributed/src/lib.rs` — boundary API for user code wanting a fully-replicated DTensor over a mesh. |
| REQ-6 | SHIPPED | impl: `pub fn to_local` (line 150), `pub fn shape` (line 155), `pub fn placements` (line 160), `pub fn mesh` (line 165), `pub fn numel` (line 170) all in `ferrotorch-distributed/src/dtensor.rs`; non-test consumer: `ferrotorch-distributed/src/dtensor.rs` (`numel` uses `self.global_shape.iter().product`), accessor surface re-exported via `DTensor` at `ferrotorch-distributed/src/lib.rs`. |
| REQ-7 | SHIPPED | impl: `pub fn redistribute` in `ferrotorch-distributed/src/dtensor.rs` with target-count and shard-dim validation at lines 194-215; non-test consumer: crate-root re-export via `DTensor` at `ferrotorch-distributed/src/lib.rs` — boundary API for user code orchestrating layout transitions alongside `crate::collective::*` calls. |
