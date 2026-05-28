# ferrotorch-graph — crate-root module declarations and re-exports

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/nn/
-->

## Summary

`ferrotorch-graph/src/lib.rs` is the crate-root file for the Phase D.1
graph neural network harness. It declares three child modules
(`data`, `gcn`, `safetensors_loader`) and re-exports their public
surface (`Graph`, `GcnConv`, `GcnNet`, `DropReport`, `load_gcn_net`)
so downstream callers (the `gcn_inference_dump` example binary and
the conformance harness) reach a flat `ferrotorch_graph::*` import
path. The crate as a whole mirrors `torch_geometric.nn.GCNConv` with
its default flags (`add_self_loops=True, normalize=True,
improved=False, bias=True`); upstream PyTorch's `torch/nn/` hierarchy
hosts the equivalent layer composition primitives (`Module`,
`Parameter`, parameter dict plumbing) the children build on top of.

## Requirements

- REQ-1: The crate root declares `pub mod data`, `pub mod gcn`, and
  `pub mod safetensors_loader` so each child file is independently
  reachable as `ferrotorch_graph::<mod>::*` for external code that
  wants a more specific import path.
- REQ-2: The crate root flat-re-exports the user-facing types
  (`Graph` from `data`, `GcnConv` and `GcnNet` from `gcn`,
  `DropReport` and `load_gcn_net` from `safetensors_loader`) so the
  `ferrotorch_graph::{Graph, GcnNet, load_gcn_net}` form (used by the
  README quick-start and the example binary) compiles. This mirrors
  the way `torch.nn` aggregates `Linear`, `Conv2d`, etc. up to its
  package root.
- REQ-3: The crate-level doc-comment names the dataset (`Cora`), the
  HuggingFace mirror (`ferrotorch/gcn-cora`), and the configuration
  flags that lock the layer behaviour to PyG's default
  (`add_self_loops=True, normalize=True, improved=False, bias=True`),
  so a reader landing on the crate page can immediately see which
  upstream contract is being implemented.
- REQ-4: The crate root carries no `#![allow]` attribute and no
  cross-cutting code other than the re-export skeleton — every
  invariant lives in the child module that owns the data type.

## Acceptance Criteria

- [x] AC-1: `cargo check -p ferrotorch-graph` resolves the
  `ferrotorch_graph::{Graph, GcnConv, GcnNet, DropReport,
  load_gcn_net}` import path used by the example binary and
  documentation.
- [x] AC-2: The crate-level doc-comment contains the strings
  `GCNConv`, `Cora`, and `ferrotorch/gcn-cora` so the user-facing
  contract is searchable.
- [x] AC-3: `cargo clippy -p ferrotorch-graph --lib -- -D warnings`
  is clean against the routed file.

## Architecture

The crate root is a pure aggregator: three `pub mod` declarations
followed by three `pub use` statements that surface the names the
README and the example binary import. Splitting the implementation
into three files keeps each file under ~500 LOC and gives each piece
(`Graph` value type, `GcnConv` / `GcnNet` layers, safetensors loader)
a doc-comment with its own design contract:

- `pub mod data` declared in `ferrotorch-graph/src/lib.rs` line 23
  hosts the `Graph` plain-data container (mirrors
  `torch_geometric.data.Data` inference-relevant fields).
- `pub mod gcn` declared at line 24 hosts the
  `GcnConv` / `GcnNet` layer pair (mirrors PyG `GCNConv` +
  `examples/gcn.py`'s 2-layer stack).
- `pub mod safetensors_loader` declared at line 25 hosts
  `load_gcn_net` and `DropReport`, the audit-trail-returning loader
  on top of `ferrotorch_serialize::load_safetensors`.

The re-export block (lines 27-29) republishes the user-facing names
so `ferrotorch_graph::Graph`, `ferrotorch_graph::GcnConv`,
`ferrotorch_graph::GcnNet`, `ferrotorch_graph::DropReport`, and
`ferrotorch_graph::load_gcn_net` all resolve without forcing callers
to know the file layout.

The message-passing primitive (`scatter_add_segments`) lives in
`ferrotorch-core` (`ferrotorch-core/src/ops/scatter.rs:74`) rather
than this crate so non-graph code can reuse it; the crate
doc-comment calls this out explicitly.

### Non-test production consumers

- `ferrotorch-graph/examples/gcn_inference_dump.rs:32`
  `use ferrotorch_graph::load_gcn_net;` (example binaries count as
  production code per goal.md's R-DEFER-1 reading; they live outside
  `tests/` and outside `#[cfg(test)]`).
- `ferrotorch-graph/README.md:32`
  `use ferrotorch_graph::{Graph, GcnNet, load_gcn_net};` (documented
  user-facing entry point).

## Parity contract

`parity_ops = []` per `tooling/translate-routes.toml`. The crate root
declares no PyTorch op of its own; numerical parity is owned by the
GCN forward path in `gcn.rs` (compared against PyG output by the
out-of-tree `conformance_gcn_cora` harness with a `cosine_sim >=
0.999` envelope on the Cora full-batch logits).

## Verification

- `cargo check -p ferrotorch-graph` resolves all four re-export
  paths.
- `cargo clippy -p ferrotorch-graph --lib -- -D warnings` is clean.
- `cargo test -p ferrotorch-graph --lib` exercises the children's
  inline tests (Graph construction, GcnConv invariants, safetensors
  round-trip).

```bash
cargo check -p ferrotorch-graph 2>&1 | tail -3
cargo clippy -p ferrotorch-graph --lib -- -D warnings 2>&1 | tail -3
cargo test -p ferrotorch-graph --lib 2>&1 | tail -3
```

Expected: `cargo check` ends with `Finished`, clippy reports no
warnings, tests show `N passed; 0 failed` (currently 8 across
data + gcn + safetensors_loader inline modules).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub mod data`, `pub mod gcn`, `pub mod safetensors_loader` at `ferrotorch-graph/src/lib.rs` lines 23-25; non-test consumer: `ferrotorch-graph/examples/gcn_inference_dump.rs:32` reaches `load_gcn_net` via the `safetensors_loader` re-export which only works because the module is declared `pub`. |
| REQ-2 | SHIPPED | impl: `pub use data::Graph;` at `load_gcn_net in ferrotorch-graph/src/lib.rs`, `pub use gcn::{GcnConv, GcnNet};` at line 28, `pub use safetensors_loader::{DropReport, load_gcn_net};` at line 29; non-test consumer: `load_gcn_net in ferrotorch-graph/examples/gcn_inference_dump.rs` `use ferrotorch_graph::load_gcn_net;` and line 237 `let (net, report) = load_gcn_net(...);` resolve through this re-export. |
| REQ-3 | SHIPPED | impl: the crate-level doc-comment at `ferrotorch-graph/src/lib.rs` lines 1-21 names `torch_geometric.nn.GCNConv`, the four PyG default flags, `Cora`, and `ferrotorch/gcn-cora`; non-test consumer: `ferrotorch-graph/README.md` reproduces the same contract and the example binary depends on these dimensions matching the safetensors mirror. |
| REQ-4 | SHIPPED | impl: `ferrotorch-graph/src/lib.rs` is 30 lines total with no `#![allow]`, no `unsafe`, no `pub fn` other than the re-exports; non-test consumer: the cleanliness of the crate root is what allows `cargo clippy -p ferrotorch-graph --lib -- -D warnings` to pass for downstream binaries that pull the crate in. |
