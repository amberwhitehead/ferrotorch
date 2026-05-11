# ferrotorch-graph

Graph neural network composition for ferrotorch.

The first pinned reference is a 2-layer GCN trained for 200 epochs on the
Cora node-classification benchmark (1433-dim features, 2708 nodes, 7
classes), mirrored to `ferrotorch/gcn-cora` and registered in
`ferrotorch-hub`. The `examples/gcn_inference_dump.rs` binary +
`scripts/verify_gnn_inference.py` harness compare full-graph forward
logits against the upstream `torch_geometric==2.7.0`
`GCNConv`-with-self-loops reference (Phase D.1 of real-artifact-driven
development).

The message-passing primitive used here (segmented `scatter_add`) lives
in `ferrotorch-core::ops::scatter::scatter_add_segments` so non-graph
crates can reuse it.
