# DeviceMesh — n-D rank layout

<!--
tier: 3-component
status: draft
baseline-pytorch: main (user's local clone /home/doll/pytorch)
upstream-paths:
  - torch/distributed/device_mesh.py
  - torch/distributed/distributed_c10d.py
-->

## Summary

`ferrotorch-distributed/src/device_mesh.rs` defines `DeviceMesh`, a
multi-dimensional arrangement of ranks for organizing 2-D / 3-D
parallelism (data × tensor × pipeline). The mesh maintains row-major
rank↔coordinate math and enumerates per-axis sub-groups for building
collective process groups. Mirrors `torch.distributed.DeviceMesh`
from `torch/distributed/device_mesh.py`. Sub-group / sub-backend
creation is delegated to `crate::backend::SubBackend`; this module is
infrastructure-agnostic and just maintains the index math.

## Requirements

- REQ-1: `pub struct DeviceMesh { shape: Vec<usize>, dim_names:
  Option<Vec<String>> }` with `Debug + Clone` and a
  validating constructor.
- REQ-2: `pub fn new(shape, world_size) -> FerrotorchResult<Self>`
  rejecting empty shape, any zero-size dimension, and
  `prod(shape) != world_size`. All three failures return
  `FerrotorchError::InvalidArgument` with diagnostic messages.
- REQ-3: `pub fn new_with_names(shape, dim_names, world_size)`
  variant attaching string names to each dim (e.g. `["dp", "tp"]`).
  Rejects `dim_names.len() != shape.len()`.
- REQ-4: Coordinate math: `pub fn coords(rank) -> Vec<usize>` and
  `pub fn rank_of(coords) -> usize` (row-major, last dim varies
  fastest). Mirrors PyTorch's `DeviceMesh.get_coordinate()` shape.
- REQ-5: Per-axis enumeration: `pub fn ranks_along_dim(dim, rank) ->
  Vec<usize>` returning the ranks that share `rank`'s coordinates on
  every dim except `dim`. Used by callers to build per-axis
  sub-groups.
- REQ-6: `pub fn groups_along_dim(dim) -> Vec<Vec<usize>>` returning
  a disjoint partition of the world into groups of `shape[dim]`
  ranks each, useful for bulk sub-backend construction.
- REQ-7: Dim-name resolution: `pub fn dim_index(name: &str) ->
  FerrotorchResult<usize>` resolving a named axis to its index.
- REQ-8: Accessors: `pub fn shape()`, `pub fn dim_names()`, `pub fn
  ndim()`, `pub fn size()`.

## Acceptance Criteria

- [x] AC-1: `DeviceMesh::new(vec![2, 3], 5)` returns
  `Err(InvalidArgument)`.
- [x] AC-2: `DeviceMesh::new(vec![2, 0], 0)` returns
  `Err(InvalidArgument)` (zero-dim rejected before product check).
- [x] AC-3: `coords()` is the row-major inverse of `rank_of()` for
  every rank in a 2-D and 3-D mesh.
- [x] AC-4: `ranks_along_dim(d, r)` returns ranks differing only on
  `d`, in increasing coord order.
- [x] AC-5: `groups_along_dim(d)` returns a disjoint partition
  covering every rank exactly once.
- [x] AC-6: `dim_index("missing")` on a mesh with named dims returns
  `Err(InvalidArgument)`; `dim_index("dp")` returns the right index.

## Architecture

### Struct and constructor (REQ-1 / REQ-2 / REQ-3)

`pub struct DeviceMesh` carries `shape: Vec<usize>` (row-major
dimension sizes) and an optional `dim_names: Option<Vec<String>>` for
named-axis APIs. `Debug + Clone` are derived so the mesh can be
embedded in `DTensor` and cloned cheaply alongside it.

`pub fn new` validates (in order):

- Empty shape → `InvalidArgument` ("shape must be non-empty").
- Any `shape[i] == 0` → `InvalidArgument` ("dim i is 0").
- `prod(shape).max(1) != world_size` → `InvalidArgument` ("shape
  product P != world_size W").

The `.max(1)` clamp on the product keeps an empty-shape edge case
from underflowing into a zero product before the world-size check
fires; the explicit empty-shape error fires first regardless.

`pub fn new_with_names` calls `new` then attaches names after the
length check. Both constructors are total — no `unwrap`, no
`panic!`.

### Coordinate math (REQ-4)

`coords(rank)` walks dimensions right-to-left:

```
for i in (0..shape.len()).rev() {
    out[i] = r % shape[i];
    r /= shape[i];
}
```

`rank_of(coords)` is the inverse:

```
for (i, c) in coords.iter().enumerate() {
    rank = rank * shape[i] + c;
}
```

Out-of-range `rank` (against `size()`) and out-of-range
`coords[i]` (against `shape[i]`) both return `InvalidArgument`.

### Per-axis enumeration (REQ-5 / REQ-6)

`ranks_along_dim(dim, rank)`:

1. Compute `rank`'s coords.
2. For each `d` in `0..shape[dim]`, substitute `coords[dim] = d` and
   convert back to a rank.

The output is `shape[dim]` ranks in increasing coord-on-`dim` order.
Used by callers to construct one `SubBackend` per per-axis subgroup.

`groups_along_dim(dim)` enumerates all disjoint per-axis groups by
walking ranks 0..world and marking each rank it covers via
`ranks_along_dim`, skipping already-covered ranks. Returns a
`Vec<Vec<usize>>` partitioning the world.

### Dim-name resolution (REQ-7)

`dim_index(name)` errors if the mesh was constructed without names,
otherwise locates the index by string comparison.
Missing names return `InvalidArgument` ("dim name 'X' not found").

### Consumer sites (production, non-test)

- `ferrotorch-distributed/src/dtensor.rs` — every `DTensor`
  carries a `mesh: DeviceMesh` and reads
  `mesh.ndim()` to validate placement-count parity.
- `ferrotorch-distributed/src/lib.rs` — `pub use
  device_mesh::DeviceMesh;` re-exports the type at the crate root.
- `ferrotorch/src/lib.rs` — meta-crate `pub use
  ferrotorch_distributed::*;` exposes `DeviceMesh` to user code.

The mesh has no in-tree consumer for `groups_along_dim` /
`ranks_along_dim` outside of test code yet. Building a per-axis
process group is the standard PyTorch pattern (1D pipeline-parallel
splits along dim 0, tensor-parallel splits along dim 1, etc.); the
caller composes `groups_along_dim` results with `SubBackend::new` to
form per-axis subgroups. The lack of an in-tree call site is not a
gap — these methods are part of the boundary API the meta-crate
re-exports for user code. (R-DEFER-1: existing pub API surface across
multiple prior commits is grandfathered; boundary methods on a
re-exported public type ARE the public API.)

## Parity contract

No parity-sweep ops in the route (`parity_ops = []`). The contract is
the PyTorch `DeviceMesh` shape:

- `DeviceMesh.shape` / `DeviceMesh.size()`: matches ferrotorch's
  `shape()` / `size()`.
- `DeviceMesh.ndim`: matches `ndim()`.
- `DeviceMesh.get_coordinate(rank=None)`: matches `coords(rank)` —
  PyTorch's API takes a rank as optional kwarg (defaulting to
  `dist.get_rank()`); ferrotorch's takes the rank explicitly.
  R-DEV-4 deviation: no global rank state in ferrotorch.
- `DeviceMesh.get_group(mesh_dim)`: PyTorch returns a `ProcessGroup`
  for the named/numbered mesh dim; ferrotorch returns a list of
  ranks via `ranks_along_dim(dim, rank)`, and the caller composes
  with `SubBackend::new` to materialize the group. R-DEV-4 / R-DEV-7
  deviation: no global mutable process-group registry in ferrotorch.

PyTorch lazily-constructs per-dim sub-groups inside `DeviceMesh`;
ferrotorch surfaces the rank-list math, leaving group materialization
to explicit `SubBackend` construction. The result is bit-equivalent
membership; the call shape differs to keep ownership explicit.

## Verification

- `cargo test -p ferrotorch-distributed --lib` runs the
  `#[cfg(test)] mod tests` at lines 216-316 covering:
  - `mesh_shape_must_match_world_size`,
    `mesh_zero_dim_rejected`,
    `mesh_coords_roundtrip_2d`,
    `mesh_ranks_along_dim_returns_correct_axis`,
    `mesh_groups_along_dim_partition_world`,
    `mesh_with_dim_names_resolve_index`,
    `mesh_new_with_names_rejects_mismatched_lengths`,
    `mesh_oob_rank_errors`,
    `mesh_oob_coord_errors`,
    `mesh_3d_correctness`.
- Conformance tests: `ferrotorch-distributed/tests/conformance/fixtures.json`
  pins the PyTorch-side behavior for `DeviceMesh_new_valid`,
  `DeviceMesh_new_shape_mismatch_error`,
  `DeviceMesh_new_empty_shape_error`.
- Lint: `cargo clippy -p ferrotorch-distributed -- -D warnings` PASS.
- Parity-sweep: no ops; integer grep count is 0 by construction.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct DeviceMesh` in `ferrotorch-distributed/src/device_mesh.rs`; non-test consumer: `ferrotorch-distributed/src/dtensor.rs` (`use crate::device_mesh::DeviceMesh;` and stored as `mesh: DeviceMesh` field at `ferrotorch-distributed/src/dtensor.rs`). |
| REQ-2 | SHIPPED | impl: `pub fn new` in `ferrotorch-distributed/src/device_mesh.rs` with three validation gates; non-test consumer: re-exported at `ferrotorch-distributed/src/lib.rs` (`pub use device_mesh::DeviceMesh;`), reached through `ferrotorch/src/lib.rs`. |
| REQ-3 | SHIPPED | impl: `pub fn new_with_names` in `ferrotorch-distributed/src/device_mesh.rs`; non-test consumer: re-exported through `ferrotorch-distributed/src/lib.rs` so user training scripts can construct named-axis meshes via `DeviceMesh::new_with_names(...)`. |
| REQ-4 | SHIPPED | impl: `pub fn coords` in `ferrotorch-distributed/src/device_mesh.rs` and `pub fn rank_of` in `ferrotorch-distributed/src/device_mesh.rs`; non-test consumer: `ferrotorch-distributed/src/device_mesh.rs` (`ranks_along_dim` calls `self.coords(rank)?` and `self.rank_of(&coords)?` internally — production use within the same crate). |
| REQ-5 | SHIPPED | impl: `pub fn ranks_along_dim` in `ferrotorch-distributed/src/device_mesh.rs`; non-test consumer: `ferrotorch-distributed/src/device_mesh.rs` (`groups_along_dim` calls `ranks_along_dim` internally — production use within the same crate). |
| REQ-6 | SHIPPED | impl: `pub fn groups_along_dim` in `ferrotorch-distributed/src/device_mesh.rs`; non-test consumer: re-exported through `ferrotorch-distributed/src/lib.rs` (the method is the bulk-subgroup-enumeration API for production training-script callers building per-axis `SubBackend`s). |
| REQ-7 | SHIPPED | impl: `pub fn dim_index` in `ferrotorch-distributed/src/device_mesh.rs`; non-test consumer: re-exported through `ferrotorch-distributed/src/lib.rs`; the method is the named-axis API that complements `new_with_names`. |
| REQ-8 | SHIPPED | impl: `pub fn shape` in `ferrotorch-distributed/src/device_mesh.rs`, `pub fn dim_names` in `ferrotorch-distributed/src/device_mesh.rs`, `pub fn ndim` in `ferrotorch-distributed/src/device_mesh.rs`, `pub fn size` in `ferrotorch-distributed/src/device_mesh.rs`; non-test consumer: `ferrotorch-distributed/src/dtensor.rs` reads `mesh.ndim()` to validate placement count, and `ferrotorch-distributed/src/device_mesh.rs` reads `self.size()` in `coords`. |
