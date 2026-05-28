# ferrotorch-graph

Graph neural network composition for ferrotorch.

The first pinned reference is a 2-layer GCN trained for 200 epochs on
the Cora node-classification benchmark (1433-dim features, 2708 nodes,
7 classes), mirrored to `ferrotorch/gcn-cora` and registered in
[`ferrotorch-hub`](../ferrotorch-hub) (#1157).

## What it provides

- **`data::Graph`** — plain-data container holding node features
  `x: [N, F]`, the COO `edge_index` (flat `Vec<i64>`, `[2, E]`),
  cached `num_edges`, and per-node integer labels `y`. The
  symmetric-normalized adjacency `Â = D^(-1/2) (A + I) D^(-1/2)` is
  computed inside the convolution forward, not stored on the graph.
- **`GcnConv`** — one Kipf-and-Welling graph convolution layer
  (`Â · X · W + b`). Forward: `(x: &Tensor<f32>, edge_index: &[i64])`
  maps `[N, in_features]` → `[N, out_features]`.
- **`GcnNet`** — two-layer GCN classifier (`conv1 -> ReLU -> conv2`,
  dropout is `Identity` at eval) with helpers `in_features()`,
  `hidden()`, `num_classes()`.
- **`load_gcn_net`** — SafeTensors loader for the pinned upstream
  PyTorch-Geometric checkpoint; constructs the `GcnNet` and returns
  `(GcnNet, DropReport)` (#1141 silent-drop-bug guard).

The message-passing primitive (segmented `scatter_add`) lives in
`ferrotorch-core::ops::scatter::scatter_add_segments` so non-graph
crates can reuse it.

## Quick start

```rust
use ferrotorch_graph::{Graph, load_gcn_net};

// Cora: 2708 nodes, 1433 features, 7 classes
let graph = Graph::new(x, edge_index, labels)?; // x: [N, F], edge_index: [2, E] flat
let (net, _drop) = load_gcn_net(
    Path::new("/path/to/gcn-cora.safetensors"),
    /*in*/ 1433, /*hidden*/ 16, /*classes*/ 7, /*strict*/ true,
)?;

let logits = net.forward(&graph.x, &graph.edge_index)?;
// logits: [2708, 7] — argmax along axis 1 to get predicted class
```

## Real-artifact parity

`scripts/verify_gnn_inference.py` compares this crate's full-graph
forward against the upstream
[`torch_geometric==2.7.0`](https://pyg.org)
`GCNConv`-with-self-loops reference on the pinned Cora checkpoint
(Phase D.1 of real-artifact-driven development; #1157). PASS floor:
`cosine_sim >= 0.999, max_abs <= 0.5`.

## Part of ferrotorch

This crate is one component of the
[ferrotorch](https://github.com/dollspace-gay/ferrotorch) workspace.
See the workspace README for full documentation.

## License

MIT OR Apache-2.0
