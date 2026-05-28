# ferrotorch-graph/gcn — `GcnConv` + `GcnNet` (PyG `GCNConv` default config)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/nn/
-->

## Summary

`ferrotorch-graph/src/gcn.rs` defines the single-layer `GcnConv` and
the 2-layer `GcnNet` stack the Phase D.1 harness pins against PyG's
reference `examples/gcn.py`. Each `GcnConv` runs the standard
symmetric-normalized message-passing recipe: add self-loops, compute
`deg_inv_sqrt[u] * deg_inv_sqrt[v]` edge weights, project with
`x @ W^T` (no bias inside the linear), scatter-add the weighted
messages onto destination nodes, then add the post-aggregation bias.
`GcnNet` chains `conv1 -> ReLU -> conv2` matching PyG's example
verbatim; dropout is `Identity` at eval and the harness runs in eval
mode. Both layers implement `ferrotorch_nn::Module` so the standard
`state_dict` / `load_state_dict` plumbing applies.

## Requirements

- REQ-1: `pub struct GcnConv` holds a `Linear<f32>` (no bias inside —
  see invariant below), a separate `Parameter<f32>` bias of shape
  `[out_features]`, the `in_features` / `out_features` dims, and a
  `training: bool` flag. The split-bias layout is load-bearing for
  parity: PyG's `GCNConv` adds the bias *after* aggregation, so
  folding it into the linear would scale it by the normalization
  weights and diverge from upstream.
- REQ-2: `GcnConv::new(in_features, out_features) -> FerrotorchResult<Self>`
  constructs with zero-initialized parameters. Init is not load-
  bearing because the safetensors loader overwrites every parameter
  before the first forward.
- REQ-3: `GcnConv::forward(&self, x: &Tensor<f32>, edge_index: &[i64])
  -> FerrotorchResult<Tensor<f32>>` runs the five-step PyG recipe:
  (1) self-loop concat, (2) symmetric-normalized edge weights with
  `inf -> 0` substitution for `deg=0` nodes (matches PyG's
  `gcn_norm`), (3) `x @ W^T` projection via the internal linear,
  (4) per-edge `msg = h[src] * w` build + `scatter_add_segments`
  onto destinations, (5) post-aggregation bias add. Returns the
  `[N, out_features]` aggregated tensor.
- REQ-4: `impl Module<f32> for GcnConv` exposes the standard
  `parameters` / `parameters_mut` / `named_parameters` / `train` /
  `eval` / `is_training` machinery so the safetensors loader's
  generic `load_state_dict` path consumes it. The `Module::forward`
  arm is implemented but refuses the call (it lacks the
  `edge_index` argument the graph-aware variant needs) — calls go
  through `GcnConv::forward(x, edge_index)` directly. The `Module`
  trait impl exists so the inherited `state_dict` /
  `load_state_dict` default methods work.
- REQ-5: `named_parameters` returns `[("lin.weight", _), ("bias", _)]`
  matching PyG `GCNConv.state_dict()` keys verbatim. This is what
  lets the pinned `ferrotorch/gcn-cora` safetensors load
  zero-copy.
- REQ-6: `pub struct GcnNet` holds two `GcnConv` layers
  (`conv1: in_features -> hidden`, `conv2: hidden -> num_classes`)
  plus a `training: bool` flag.
- REQ-7: `GcnNet::forward(&self, x, edge_index)` runs the canonical
  `conv1 -> ReLU -> conv2 -> raw logits` recipe matching PyG
  `examples/gcn.py`. Dropout is omitted at eval per the same
  example.
- REQ-8: `impl Module<f32> for GcnNet` aggregates both convs'
  parameters and produces `["conv1.bias", "conv1.lin.weight",
  "conv2.bias", "conv2.lin.weight"]` for `named_parameters` —
  matches PyG's `Sequential([GCNConv, GCNConv]).state_dict()`
  key prefix.

## Acceptance Criteria

- [x] AC-1: `GcnConv` separates the linear (no bias) from the
  post-aggregation `bias: Parameter<f32>` field.
- [x] AC-2: `GcnConv::forward` adds `N` self-loops to the edge list
  before computing weights.
- [x] AC-3: Disconnected nodes get `deg=1` after self-loops and the
  self-loop weight is `1.0 / sqrt(1) * 1.0 / sqrt(1) == 1.0`
  (verified by `gcn_conv_self_loops_disconnected_node_is_identity`).
- [x] AC-4: With `W = I`, `b = 0`, and an undirected 2-node graph
  (edges `(0,1)` and `(1,0)`), each output row is the symmetric mean
  of input rows (`out[v] = 0.5 * x[u] + 0.5 * x[v]`) — verified by
  `gcn_conv_two_node_chain_aggregates_neighbor`.
- [x] AC-5: `named_parameters` of `GcnConv` returns exactly
  `{"lin.weight", "bias"}` (sorted set comparison —
  `gcn_conv_named_parameters_match_pyg_layout`).
- [x] AC-6: `named_parameters` of `GcnNet` returns exactly
  `{"conv1.bias", "conv1.lin.weight", "conv2.bias",
  "conv2.lin.weight"}` —
  `gcn_net_named_parameters_match_pyg_layout`.
- [x] AC-7: `GcnNet::forward` on a 3-node line graph produces an
  `[N, num_classes]` tensor with all finite values
  (`gcn_net_forward_two_layer_chain`).
- [x] AC-8: `Module::forward(&self, _input)` on either layer
  returns `FerrotorchError::InvalidArgument` (so accidental
  `Module`-trait-driven callers get a clear diagnostic).

## Architecture

### Bias split (REQ-1)

The struct doc-comment at `doc in ferrotorch-graph/src/gcn.rs`
documents the bias-split rationale: PyG adds bias *after*
aggregation. The `lin: Linear<f32>` is constructed with
`bias = false` (`ferrotorch-graph/src/gcn.rs:82`) so the linear pass
is bias-free; the post-aggregation bias lives in
`bias: Parameter<f32>` (line 83).

### Forward path (REQ-3)

`GcnConv::forward` (`ferrotorch-graph/src/gcn.rs:98-216`) runs the
five PyG steps inline. Numbered comment blocks tie each block to the
step in the module doc-comment:

1. Self-loop concat (lines 125-139): build `src` and `dst` vectors
   of capacity `e_in + N`; copy the original COO rows; push
   `(v, v)` for `v in 0..N`.
2. Edge weights (lines 141-166): scan `dst` to build `deg[v]`; cast
   to `f32` and compute `deg_inv_sqrt[v] = 1.0 / sqrt(deg[v])` with
   the `deg=0 -> 0.0` guard PyG uses (the guard never fires once
   self-loops are added — every node has `deg >= 1` — but it stays
   in for parity with PyG's `gcn_norm` source).
3. Projection (lines 168-176): `h = self.lin.forward(x)?` — this is
   the `x @ W^T` call into `ferrotorch-nn`. Bias is `false` inside
   the linear, so `h` carries no bias.
4. Aggregation (lines 178-197): build `msg: [E_aug, out_features]`
   by row-multiplying `h[src[e]]` with `edge_w[e]`, then call
   `scatter_add_segments(msg, dst, N)` from
   `ferrotorch-core/src/ops/scatter.rs:74` to accumulate per-dst.
5. Bias add (lines 199-215): materialize `out = aggregated +
   bias` directly into a fresh `[N, out_features]` buffer; bypass
   the broadcasting add machinery because (a) inference is
   `requires_grad=false` so autograd recording would be wasted, and
   (b) writing one tight loop avoids the dispatch round-trip.

### `Module<f32>` trait impl (REQ-4)

`impl Module<f32> for GcnConv` (lines 219-274) is the standard
parameter / training-mode aggregation. The `forward(&self, _input)`
arm explicitly returns
`FerrotorchError::InvalidArgument` because the trait signature
lacks `edge_index`; routing through a thread-local edge index or
a wrapper input tensor was rejected as worse than the typed
inherent `GcnConv::forward(x, edge_index)`. The doc-comment at
`ferrotorch-graph/src/gcn.rs:220-232` records this decision.

### `named_parameters` PyG key parity (REQ-5)

`GcnConv::named_parameters` (lines 247-259) prefixes the inner
linear's `weight` with `"lin."` and appends `"bias"`. The PyG
`GCNConv.state_dict()` keys are exactly `["bias", "lin.weight"]` —
the set comparison test `gcn_conv_named_parameters_match_pyg_layout`
(line 401) pins this.

### `GcnNet` two-layer stack (REQ-6, REQ-7, REQ-8)

`GcnNet::new(in_features, hidden, num_classes)` constructs the two
convs sharing the inner Linear / Parameter wiring; `GcnNet::forward`
(lines 327-339) chains `conv1 -> ReLU -> conv2`. The ReLU is
materialised inline (loop over `h_data` writing
`if v > 0.0 { v } else { 0.0 }`) because the inference path is
`requires_grad=false` and going through `ferrotorch-core`'s
autograd-tracked ReLU would only add overhead.

`impl Module<f32> for GcnNet::named_parameters` (lines 360-369)
prefixes each conv's outputs with `"conv1."` / `"conv2."`,
producing the four PyG-canonical keys.

### Non-test production consumers

- `ferrotorch-graph/src/safetensors_loader.rs:72`
  `let mut net = GcnNet::new(in_features, hidden, num_classes)?;`
  (the loader constructs a fresh net before populating it from the
  safetensors).
- `ferrotorch-graph/src/safetensors_loader.rs:73-75`
  `net.named_parameters().into_iter().map(|(n, _)| n).collect()` —
  the loader walks `named_parameters` to filter unmapped upstream
  keys for the `DropReport`.
- `DropReport in ferrotorch-graph/src/safetensors_loader.rs`
  `net.load_state_dict(&filtered, true)?` — drives the
  `Module::load_state_dict` machinery whose backing
  `named_parameters` / `parameters_mut` impls live in this file.
- `ferrotorch-graph/examples/gcn_inference_dump.rs`
  `let (net, report) = load_gcn_net(...)` followed by
  `ferrotorch-graph/examples/gcn_inference_dump.rs:250`
  `let logits = net.forward(&x, &edge_index)?;` — the example
  binary calls the graph-aware forward directly.
- The `Module::forward` arm that returns the "call
  `GcnConv::forward(x, edge_index)` instead" error is a
  diagnostic-only consumer; the trait method is required so the
  inherited `Module::state_dict` default works.

## Parity contract

`parity_ops = []`. The numerical parity contract for `GcnConv` /
`GcnNet` is owned by the out-of-tree
`scripts/verify_gnn_inference.py` harness comparing ferrotorch
logits to PyG's `examples/gcn.py` output. Expected envelope:
`cosine_sim >= 0.999`, `max_abs <= 0.5` on the 2708-node Cora
full-batch logits (per the module doc-comment at
`ferrotorch-graph/src/gcn.rs`).

Per-op edge-case expectations:

- **Disconnected node** (no edges): self-loop gives `deg=1`,
  `w(v,v) = 1`, so the output row is `W x_v + b` — verified by
  `gcn_conv_self_loops_disconnected_node_is_identity`.
- **Undirected 2-node graph**: every edge weight becomes `0.5`, so
  the output is the symmetric mean — verified by
  `gcn_conv_two_node_chain_aggregates_neighbor`.
- **Empty edge_index** (length 0): self-loops still added; behaves
  as the disconnected case for every node.
- **NaN / Inf in features**: not specifically guarded; PyG also
  doesn't guard. Propagates through the linear and aggregation.
- **dtype promotion**: forward is `f32`-only by the type signature.
  No dtype handling needed.
- **Non-contiguous input**: `x.data_vec()?` materializes to row-
  major contiguous data, matching what `Linear::forward` and the
  inline ReLU loop expect.

## Verification

Five inline tests at `ferrotorch-graph/src/gcn.rs:392-505`:

- `gcn_conv_named_parameters_match_pyg_layout` — set equality with
  PyG `GCNConv.state_dict()` keys (line 401).
- `gcn_net_named_parameters_match_pyg_layout` — set equality with
  PyG 2-layer keys (line 417).
- `gcn_conv_self_loops_disconnected_node_is_identity` — REQ-3
  numerical check with `W=I, b=0` (line 434).
- `gcn_conv_two_node_chain_aggregates_neighbor` — REQ-3 numerical
  check with symmetric-mean expectation (line 456).
- `gcn_net_forward_two_layer_chain` — REQ-7 shape + finite-value
  smoke (line 483).

The crate also carries an `#[ignored]` integration test in
`ferrotorch-graph/tests/conformance_gcn_cora.rs` that compares
against a recorded PyG reference (gated on the pinned safetensors
mirror being downloadable).

```bash
cargo test -p ferrotorch-graph --lib gcn:: 2>&1 | tail -3
```

Expected: `5 passed; 0 failed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct GcnConv` at `GcnConv in ferrotorch-graph/src/gcn.rs` with the `Linear<f32>` (`bias = false`) + separate `Parameter<f32>` bias split, mirroring PyG's bias-after-aggregation contract; non-test consumer: `load_gcn_net in ferrotorch-graph/src/safetensors_loader.rs` constructs a `GcnNet` (each layer is a `GcnConv`) and the example binary's `load_gcn_net in gcn_inference_dump.rs` reaches this struct via `load_gcn_net`. |
| REQ-2 | SHIPPED | impl: `pub fn GcnConv::new(in_features, out_features) -> FerrotorchResult<Self>` at `ferrotorch-graph/src/gcn.rs:81-91`; non-test consumer: `ferrotorch-graph/src/gcn.rs:298` `GcnNet::new` invokes `GcnConv::new` twice (once per layer), and `ferrotorch-graph/src/safetensors_loader.rs:72` invokes `GcnNet::new` which transitively calls `GcnConv::new`. |
| REQ-3 | SHIPPED | impl: `pub fn GcnConv::forward` at `GcnConv in ferrotorch-graph/src/gcn.rs` runs the five-step PyG recipe; non-test consumer: `forward in ferrotorch-graph/src/gcn.rs` `GcnNet::forward` calls `self.conv1.forward(x, edge_index)?` and line 337 calls `self.conv2.forward(&h_relu, edge_index)`; the example binary at `forward in ferrotorch-graph/examples/gcn_inference_dump.rs` invokes `net.forward(&x, &edge_index)` which transitively reaches this. |
| REQ-4 | SHIPPED | impl: `impl Module<f32> for GcnConv` at `ferrotorch-graph/src/gcn.rs:219-274` exposes `parameters` / `parameters_mut` / `named_parameters` / `train` / `eval` / `is_training` and explicitly refuses the `Module::forward(_input)` arm; non-test consumer: `ferrotorch-graph/src/safetensors_loader.rs:95` `net.load_state_dict(&filtered, true)?` drives the `Module` machinery on every `GcnConv` instance inside the loaded `GcnNet`. |
| REQ-5 | SHIPPED | impl: `GcnConv::named_parameters` at `ferrotorch-graph/src/gcn.rs:247-259` prefixes the linear's `weight` with `"lin."` and appends `"bias"`, matching PyG `GCNConv.state_dict()` keys verbatim; non-test consumer: `ferrotorch-graph/src/safetensors_loader.rs:73-74` walks `net.named_parameters().into_iter().map(|(n, _)| n).collect()` to compute the expected key set against the safetensors header. |
| REQ-6 | SHIPPED | impl: `pub struct GcnNet` at `GcnNet in ferrotorch-graph/src/gcn.rs` holds two `GcnConv` layers + training flag; non-test consumer: `ferrotorch-graph/src/safetensors_loader.rs` returns `(GcnNet, DropReport)` and the example binary at `gcn_inference_dump.rs` destructures it. |
| REQ-7 | SHIPPED | impl: `GcnNet::forward` at `ferrotorch-graph/src/gcn.rs:327-339` runs `conv1 -> ReLU -> conv2`; non-test consumer: `ferrotorch-graph/examples/gcn_inference_dump.rs:250` `let logits = net.forward(&x, &edge_index)?;` is the end-to-end inference entry point used by the harness binary. |
| REQ-8 | SHIPPED | impl: `impl Module<f32> for GcnNet::named_parameters` at `named_parameters in ferrotorch-graph/src/gcn.rs` produces the four PyG-canonical prefixed keys; non-test consumer: `named_parameters in ferrotorch-graph/src/safetensors_loader.rs` consumes those keys to compute the expected-key set against the safetensors header, and `named_parameters in safetensors_loader.rs` calls `net.load_state_dict(&filtered, true)?` which mutates parameters indexed by those keys. |
