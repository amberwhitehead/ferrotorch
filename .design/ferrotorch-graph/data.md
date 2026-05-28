# ferrotorch-graph/data — `Graph` plain-data container

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/nn/
-->

## Summary

`ferrotorch-graph/src/data.rs` defines the `Graph` value type that the
GCN inference path consumes — dense node features `x: Tensor<f32>`
with shape `[N, F]`, an undirected `edge_index: Vec<i64>` in flat COO
layout `[2, E]` (row 0 source, row 1 destination), per-node labels
`y: Vec<i64>` of length `N`, and a cached `num_edges: usize`. It
mirrors the inference-relevant subset of `torch_geometric.data.Data`;
train / val / test masks are intentionally omitted because the
harness is an eval-only path.

## Requirements

- REQ-1: `pub struct Graph` holds the four fields a GCN forward pass
  needs (`x: Tensor<f32>` shape `[N, F]`, `edge_index: Vec<i64>` flat
  `[2, E]` COO, `num_edges: usize` cached, `y: Vec<i64>` length `N`),
  mirroring the inference-relevant subset of
  `torch_geometric.data.Data`. The struct derives `Debug` and `Clone`
  so the harness can keep a pristine copy alongside the consumed
  tensors.
- REQ-2: `Graph::new(x, edge_index, y) -> FerrotorchResult<Self>` is
  a fallible constructor that validates every shape invariant up
  front: `x.ndim() == 2`, `edge_index.len() % 2 == 0`, every endpoint
  is in `[0, N)`, and `y.len() == N`. Surfaces a malformed mirror at
  load time rather than inside the forward pass.
- REQ-3: `Graph::num_nodes()` returns `x.shape()[0]` and
  `Graph::num_features()` returns `x.shape()[1]`. Both are pure
  getters with no validation cost so callers can use them in tight
  loops.
- REQ-4: `Graph::edge_src(e)` and `Graph::edge_dst(e)` return the
  source / destination endpoint of edge `e` from the flat COO buffer
  (row 0 occupies indices `0..num_edges`, row 1 occupies
  `num_edges..2*num_edges`). Both are `#[inline]` so the GCN forward
  inner loop pays no overhead.

## Acceptance Criteria

- [x] AC-1: `Graph::new` rejects a 1-D `x` with a
  `FerrotorchError::ShapeMismatch`.
- [x] AC-2: `Graph::new` rejects an odd-length `edge_index` with
  `ShapeMismatch`.
- [x] AC-3: `Graph::new` rejects an endpoint `< 0` or `>= num_nodes`
  with `FerrotorchError::InvalidArgument`.
- [x] AC-4: `Graph::new` rejects `y.len() != num_nodes` with
  `ShapeMismatch`.
- [x] AC-5: `Graph::num_nodes` / `num_features` / `edge_src` /
  `edge_dst` are pure (no panics, no allocation, no error path).

## Architecture

### `Graph` struct (REQ-1)

The four fields are all `pub` because this is explicitly a
plain-data container — there is no invariant to enforce post-
construction. The fields are typed asymmetrically (`Tensor<f32>` for
`x`, raw `Vec<i64>` for `edge_index` and `y`) because the GCN
forward (`ferrotorch-graph/src/gcn.rs`) consumes them differently:
`x` flows through `Linear::forward` and benefits from the
`Tensor` autograd machinery; `edge_index` is read by index from a
`&[i64]` slice (`scatter_add_segments`, self-loop concat, degree
count); a `Tensor<i64>` indirection would only add copies. The
struct doc-comment at `ferrotorch-graph/src/data.rs` calls
this out explicitly.

### `Graph::new` validation (REQ-2)

The constructor runs four checks in order
(`ferrotorch-graph/src/data.rs:39-78`):

1. `x.ndim() == 2`, otherwise `ShapeMismatch`.
2. `edge_index.len() % 2 == 0`, otherwise `ShapeMismatch` (the COO
   format requires an even length so the buffer can be split into
   two equal-length rows).
3. Every endpoint `v` in `edge_index` satisfies `0 <= v < N`,
   otherwise `InvalidArgument`. Bounds-checking up front means the
   forward path can index without re-checking.
4. `y.len() == N`, otherwise `ShapeMismatch`.

`num_edges` is computed once from `edge_index.len() / 2` and cached
on the struct so the forward path never recomputes it.

### Accessors (REQ-3, REQ-4)

`Graph::num_nodes` and `Graph::num_features` (lines 81-88) are
trivial `x.shape()[i]` reads. `Graph::edge_src` and `Graph::edge_dst`
(lines 90-101) are `#[inline]` so the loop in
`GcnConv::forward` that iterates edges
(`ferrotorch-graph/src/gcn.rs:162-166` building the degree count and
edge-weight buffers) inlines to a single load each. The COO row
layout (row 0 first, row 1 next, with `num_edges` as the boundary)
matches PyTorch Geometric's convention exactly: PyG's
`edge_index[0]` is the source row and `edge_index[1]` the
destination row.

### Non-test production consumers

- `ferrotorch-graph/src/gcn.rs:98`
  `pub fn forward(&self, x: &Tensor<f32>, edge_index: &[i64]) -> ...`
  takes the same flat-COO buffer layout `Graph` exposes; the example
  binary materializes `Graph` shapes via the `Graph::new` invariants
  before calling `GcnNet::forward`.
- `ferrotorch-graph/examples/gcn_inference_dump.rs:223-232` reads
  the COO buffer from a `[2, E]` file dump and feeds it directly to
  the GCN forward — the same flat-`Vec<i64>` shape `Graph`
  stores. The bounds-check invariants `Graph::new` documents are
  what allow the forward to skip per-edge checks.

The `Graph::num_nodes`, `num_features`, `edge_src`, `edge_dst`
accessors are NOT currently called from non-test code — the example
binary feeds the COO buffer directly into `GcnNet::forward` without
constructing a `Graph` wrapper. This is the gap REQ-3 / REQ-4 sit
against; see the prerequisite blocker referenced below.

## Parity contract

`parity_ops = []`. `Graph` is a value type — the parity contract
belongs to the downstream `GcnConv::forward` (which consumes
`edge_index` directly). The expectation for `Graph::new` validation
is: any malformed PyG-shaped input fails fast at construction; a
well-formed input must reach `GcnConv::forward` byte-identical to
how PyG's `Data` would.

## Verification

Four inline tests at `ferrotorch-graph/src/data.rs:104-148`:

- `graph_constructs_with_consistent_shapes` — happy path on a
  3-node, 2-edge graph; checks `num_nodes`, `num_features`,
  `num_edges`, `edge_src`, `edge_dst` against expected values.
- `graph_rejects_oob_edge_endpoint` — endpoint `5` against `N=2`
  rejected.
- `graph_rejects_label_count_mismatch` — `y.len() != N` rejected.
- `graph_rejects_odd_edge_index_length` — length-3 edge_index
  rejected.

```bash
cargo test -p ferrotorch-graph --lib data:: 2>&1 | tail -3
```

Expected: `4 passed; 0 failed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct Graph` at `ferrotorch-graph/src/data.rs:17-33` mirroring `torch_geometric.data.Data` inference subset; non-test consumer: the field layout (typed `x: Tensor<f32>` + flat `Vec<i64>` for COO + cached `num_edges`) matches the buffer shapes `ferrotorch-graph/src/gcn.rs:98` `GcnConv::forward(&self, x: &Tensor<f32>, edge_index: &[i64])` consumes, and `ferrotorch-graph/examples/gcn_inference_dump.rs:220-232` materialises those exact shapes from disk before invoking the forward. |
| REQ-2 | SHIPPED | impl: `pub fn Graph::new` at `Graph in ferrotorch-graph/src/data.rs` runs all four invariants (`x.ndim() == 2`, even-length `edge_index`, endpoint `< N`, `y.len() == N`); non-test consumer: the example binary's `read_dump_f32` / `read_dump_i64` paths (`read_dump_i64 in ferrotorch-graph/examples/gcn_inference_dump.rs`) re-implement the same invariants pre-call before reaching `GcnNet::forward`, mirroring the `Graph::new` validation contract. Note: `Graph::new` itself is currently invoked only by inline tests; the example binary calls the forward directly. See blocker #1481 for wiring `Graph::new` into the example. |
| REQ-3 | NOT-STARTED | open prereq blocker #1481 — `Graph::num_nodes` / `num_features` are pure accessors but no non-test production code currently calls them. The example binary reads shape dimensions from disk; rewiring it to go through `Graph::new + .num_nodes()` is the consumer-wiring blocker. |
| REQ-4 | NOT-STARTED | open prereq blocker #1481 — same gap as REQ-3. `edge_src` / `edge_dst` are `#[inline]` and ready, but `GcnConv::forward` walks `src[]` / `dst[]` slices it builds locally from `edge_index` rather than going through `Graph::edge_src(e)` / `Graph::edge_dst(e)`. Consumer-wiring lives in #1481. |
