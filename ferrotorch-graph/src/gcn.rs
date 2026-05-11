//! 2-layer Graph Convolutional Network — matches
//! `torch_geometric.nn.GCNConv` with the default
//! `add_self_loops=True, normalize=True, improved=False, bias=True`
//! configuration.
//!
//! # Per-layer math (matches PyG's `gcn_norm`)
//!
//! Given an edge list `(src, dst)` over `N` nodes:
//!
//! 1. **Add self-loops.** Append `(v, v)` for every `v in 0..N` to the
//!    edge list. (`improved=False`, so the self-loop weight is 1.)
//! 2. **Compute symmetric-normalized edge weights.** For each edge
//!    `(u, v)` with self-loops included:
//!    ```text
//!    deg[v]   = number of edges with v as the destination (incl. self-loop)
//!    w(u, v)  = 1 / (sqrt(deg[u]) * sqrt(deg[v]))
//!    ```
//!    PyG actually computes `deg_inv_sqrt[u] * deg_inv_sqrt[v]` after
//!    setting `deg_inv_sqrt = deg.pow(-0.5)` (and replacing infinities
//!    with 0 — a guard against deg=0 nodes that cannot occur once
//!    self-loops are added). The forward path here follows that same
//!    formula.
//! 3. **Linear transform.** `h = x @ W^T` (no bias on the propagated
//!    half; PyG's `GCNConv` adds the bias *after* aggregation).
//! 4. **Propagate.** For each (self-loop-augmented) edge `(u, v)`:
//!    ```text
//!    msg(u, v) = h[u] * w(u, v)
//!    out[v]   += msg(u, v)
//!    ```
//!    The accumulation is a `scatter_add_segments` over the `v` row of
//!    the augmented edge_index.
//! 5. **Bias.** Add `b` (broadcast over the `N` rows).
//!
//! # On dropout / activation
//!
//! The reference PyG `examples/gcn.py` wraps the first layer with
//! `F.relu` and the second with neither (raw logits). Dropout is the
//! identity at eval. `GcnNet::forward` follows that recipe.
//!
//! # Numerical fidelity
//!
//! All intermediate tensors are `f32`. The harness compares against a
//! PyG forward also run in `f32` (no `bfloat16` or `tf32` modes), so the
//! only divergence comes from accumulation order — small enough to fit
//! a `cosine_sim >= 0.999` / `max_abs <= 0.5` envelope on the 2708-node
//! Cora full-batch logits.

use ferrotorch_core::{
    FerrotorchError, FerrotorchResult, Tensor, TensorStorage, scatter_add_segments,
};
use ferrotorch_nn::linear::Linear;
use ferrotorch_nn::module::Module;
use ferrotorch_nn::parameter::Parameter;

/// One Graph Convolutional layer (PyG `GCNConv` default config).
///
/// Holds an internal `Linear` (no bias) for the `x @ W^T` projection
/// plus a separate post-aggregation `bias` parameter. Splitting the
/// bias off `Linear` matches PyG's runtime behavior: in PyG the bias
/// is added *after* the aggregation, not inside it. Folding the bias
/// into the linear pre-multiply would scale it by the normalization
/// weights — wrong, and a known footgun called out in PyG's own
/// docstring.
#[derive(Debug)]
pub struct GcnConv {
    /// `lin.weight: [out_features, in_features]`. No bias parameter
    /// inside the linear — see struct doc.
    pub lin: Linear<f32>,
    /// Post-aggregation bias, shape `[out_features]`.
    pub bias: Parameter<f32>,
    in_features: usize,
    out_features: usize,
    training: bool,
}

impl GcnConv {
    /// Construct a fresh `GcnConv` with zero-initialized parameters.
    /// Initialization is not load-bearing for the harness — the loader
    /// overwrites every parameter from the pinned safetensors before
    /// the first forward.
    pub fn new(in_features: usize, out_features: usize) -> FerrotorchResult<Self> {
        let lin = Linear::<f32>::new(in_features, out_features, /* bias = */ false)?;
        let bias = Parameter::zeros(&[out_features])?;
        Ok(Self {
            lin,
            bias,
            in_features,
            out_features,
            training: false,
        })
    }

    /// Forward over one graph snapshot.
    ///
    /// `edge_index` is a flat `[2, E]` COO buffer — row 0 holds the
    /// source endpoint and row 1 the destination endpoint of each
    /// edge.
    pub fn forward(
        &self,
        x: &Tensor<f32>,
        edge_index: &[i64],
    ) -> FerrotorchResult<Tensor<f32>> {
        let shape = x.shape();
        if shape.len() != 2 {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "GcnConv::forward: x must be 2-D [N, F], got shape {shape:?}"
                ),
            });
        }
        let n = shape[0];
        let in_f = shape[1];
        if in_f != self.in_features {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "GcnConv::forward: x.shape()[1] = {in_f} != in_features = {}",
                    self.in_features
                ),
            });
        }
        if edge_index.len() % 2 != 0 {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "GcnConv::forward: edge_index length {} not divisible by 2",
                    edge_index.len()
                ),
            });
        }
        let e_in = edge_index.len() / 2;

        // ---------------------------------------------------------------
        // 1. Add self-loops: append `(v, v)` for v in 0..N to the COO
        //    edge_index. PyG inserts them at the *end* of the buffer.
        // ---------------------------------------------------------------
        let e_aug = e_in + n;
        let mut src: Vec<i64> = Vec::with_capacity(e_aug);
        let mut dst: Vec<i64> = Vec::with_capacity(e_aug);
        // Original edges first (rows 0 then 1 of the [2, E] buffer).
        src.extend_from_slice(&edge_index[..e_in]);
        dst.extend_from_slice(&edge_index[e_in..]);
        // Then one self-loop per node.
        for v in 0..n {
            src.push(v as i64);
            dst.push(v as i64);
        }

        // ---------------------------------------------------------------
        // 2. Symmetric-normalized edge weights.
        //    deg[v] = #edges with dst == v (in the augmented edge list).
        //    deg_inv_sqrt[v] = 1 / sqrt(deg[v]) (or 0 if deg[v] == 0).
        //    w(u, v) = deg_inv_sqrt[u] * deg_inv_sqrt[v].
        // ---------------------------------------------------------------
        let mut deg = vec![0u32; n];
        for &v in &dst {
            // Endpoint range was validated in Graph::new for the original
            // edges, and self-loop entries are constructed in-range, so
            // both halves are safe.
            deg[v as usize] += 1;
        }
        let mut deg_inv_sqrt = vec![0.0_f32; n];
        for (i, &d) in deg.iter().enumerate() {
            if d > 0 {
                deg_inv_sqrt[i] = 1.0_f32 / (d as f32).sqrt();
            }
            // else stays 0.0, matching PyG's `inf -> 0` substitution.
        }
        let mut edge_w = Vec::with_capacity(e_aug);
        for e in 0..e_aug {
            let u = src[e] as usize;
            let v = dst[e] as usize;
            edge_w.push(deg_inv_sqrt[u] * deg_inv_sqrt[v]);
        }

        // ---------------------------------------------------------------
        // 3. Linear transform: h = x @ W^T (NO bias — bias is post-agg).
        //    `Linear::forward` accepts [N, F] -> [N, out_features].
        // ---------------------------------------------------------------
        let h = self.lin.forward(x)?;
        let h_data = h.data_vec()?;
        let out_f = self.out_features;
        debug_assert_eq!(h_data.len(), n * out_f);

        // ---------------------------------------------------------------
        // 4. Build the per-edge message buffer:
        //      msg[e, :] = h[src[e], :] * edge_w[e]
        //    then aggregate it with scatter_add_segments onto dst.
        // ---------------------------------------------------------------
        let mut msg = vec![0.0_f32; e_aug * out_f];
        for e in 0..e_aug {
            let u = src[e] as usize;
            let w = edge_w[e];
            let h_row = &h_data[u * out_f..(u + 1) * out_f];
            let m_row = &mut msg[e * out_f..(e + 1) * out_f];
            for (m, &hv) in m_row.iter_mut().zip(h_row.iter()) {
                *m = hv * w;
            }
        }
        let msg_tensor = Tensor::<f32>::from_storage(
            TensorStorage::cpu(msg),
            vec![e_aug, out_f],
            /* requires_grad = */ false,
        )?;
        let aggregated = scatter_add_segments(&msg_tensor, &dst, n)?;

        // ---------------------------------------------------------------
        // 5. Post-aggregation bias add. We materialize directly into a
        //    fresh buffer rather than calling a broadcasting `add`,
        //    because the bias here is a `[out_features]` vector and an
        //    `[N, out_features]` tensor — and the goal is to match PyG
        //    elementwise, not to drag autograd into the inference path.
        // ---------------------------------------------------------------
        let agg_data = aggregated.data_vec()?;
        let bias_data = self.bias.tensor().data_vec()?;
        debug_assert_eq!(bias_data.len(), out_f);
        let mut out = vec![0.0_f32; n * out_f];
        for v in 0..n {
            for f in 0..out_f {
                out[v * out_f + f] = agg_data[v * out_f + f] + bias_data[f];
            }
        }
        Tensor::<f32>::from_storage(TensorStorage::cpu(out), vec![n, out_f], false)
    }
}

impl Module<f32> for GcnConv {
    fn forward(&self, _input: &Tensor<f32>) -> FerrotorchResult<Tensor<f32>> {
        // The graph-aware forward needs the edge list, which a plain
        // `Module::forward` signature does not carry. Routing through
        // the `Module` trait would force a thread-local edge_index or a
        // wrapper input tensor — both worse than a typed inherent
        // `forward(&self, x, edge_index)`. Keep the trait impl so the
        // standard `state_dict / load_state_dict` plumbing works, but
        // refuse the call.
        Err(FerrotorchError::InvalidArgument {
            message:
                "GcnConv::Module::forward: call GcnConv::forward(x, edge_index) instead — \
                 the graph-aware variant needs the edge list".into(),
        })
    }

    fn parameters(&self) -> Vec<&Parameter<f32>> {
        let mut out = self.lin.parameters();
        out.push(&self.bias);
        out
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<f32>> {
        let mut out = self.lin.parameters_mut();
        out.push(&mut self.bias);
        out
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<f32>)> {
        // Matches PyG GCNConv.state_dict():
        //   "lin.weight" (from Linear, which only has `weight`)
        //   "bias"
        let mut out: Vec<(String, &Parameter<f32>)> = self
            .lin
            .named_parameters()
            .into_iter()
            .map(|(n, p)| (format!("lin.{n}"), p))
            .collect();
        out.push(("bias".to_string(), &self.bias));
        out
    }

    fn train(&mut self) {
        self.training = true;
        self.lin.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.lin.eval();
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ---------------------------------------------------------------------------
// GcnNet — the 2-layer GCN that the harness compares against.
// ---------------------------------------------------------------------------

/// Two-layer GCN matching PyG's `examples/gcn.py`:
/// `conv1 -> ReLU -> conv2 -> raw logits`. Dropout is `Identity` at
/// eval and the inference path runs in eval mode.
#[derive(Debug)]
pub struct GcnNet {
    pub conv1: GcnConv,
    pub conv2: GcnConv,
    in_features: usize,
    hidden: usize,
    num_classes: usize,
    training: bool,
}

impl GcnNet {
    /// Construct a `GcnNet` with zero-initialized parameters. The
    /// loader replaces them with the pinned safetensors values before
    /// the first forward.
    pub fn new(
        in_features: usize,
        hidden: usize,
        num_classes: usize,
    ) -> FerrotorchResult<Self> {
        let conv1 = GcnConv::new(in_features, hidden)?;
        let conv2 = GcnConv::new(hidden, num_classes)?;
        Ok(Self {
            conv1,
            conv2,
            in_features,
            hidden,
            num_classes,
            training: false,
        })
    }

    /// `in_features` the model was constructed with.
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// `hidden` the model was constructed with.
    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// `num_classes` the model was constructed with.
    pub fn num_classes(&self) -> usize {
        self.num_classes
    }

    /// Inference forward: `[N, in_features] -> [N, num_classes]` logits.
    /// Matches PyG `examples/gcn.py` at eval (no dropout).
    pub fn forward(
        &self,
        x: &Tensor<f32>,
        edge_index: &[i64],
    ) -> FerrotorchResult<Tensor<f32>> {
        let h = self.conv1.forward(x, edge_index)?;
        // ReLU after conv1.
        let h_data = h.data_vec()?;
        let mut relu = vec![0.0_f32; h_data.len()];
        for (out_v, &in_v) in relu.iter_mut().zip(h_data.iter()) {
            *out_v = if in_v > 0.0 { in_v } else { 0.0 };
        }
        let h_relu = Tensor::<f32>::from_storage(
            TensorStorage::cpu(relu),
            h.shape().to_vec(),
            false,
        )?;
        self.conv2.forward(&h_relu, edge_index)
    }
}

impl Module<f32> for GcnNet {
    fn forward(&self, _input: &Tensor<f32>) -> FerrotorchResult<Tensor<f32>> {
        Err(FerrotorchError::InvalidArgument {
            message:
                "GcnNet::Module::forward: call GcnNet::forward(x, edge_index) instead"
                    .into(),
        })
    }

    fn parameters(&self) -> Vec<&Parameter<f32>> {
        let mut out = self.conv1.parameters();
        out.extend(self.conv2.parameters());
        out
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<f32>> {
        let mut out = self.conv1.parameters_mut();
        out.extend(self.conv2.parameters_mut());
        out
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<f32>)> {
        let mut out = Vec::new();
        for (n, p) in self.conv1.named_parameters() {
            out.push((format!("conv1.{n}"), p));
        }
        for (n, p) in self.conv2.named_parameters() {
            out.push((format!("conv2.{n}"), p));
        }
        out
    }

    fn train(&mut self) {
        self.training = true;
        self.conv1.train();
        self.conv2.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.conv1.eval();
        self.conv2.eval();
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn t(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
    }

    #[test]
    fn gcn_conv_named_parameters_match_pyg_layout() {
        let conv = GcnConv::new(8, 4).unwrap();
        let names: Vec<String> = conv
            .named_parameters()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        // PyG `GCNConv.state_dict()`: ["bias", "lin.weight"] (insertion order).
        // We only require the *set* to match — order is irrelevant for
        // the loader, which reads by key.
        let mut have = names;
        have.sort();
        assert_eq!(have, vec!["bias", "lin.weight"]);
    }

    #[test]
    fn gcn_net_named_parameters_match_pyg_layout() {
        let net = GcnNet::new(8, 4, 3).unwrap();
        let names: Vec<String> = net
            .named_parameters()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        let mut have = names;
        have.sort();
        assert_eq!(
            have,
            vec![
                "conv1.bias",
                "conv1.lin.weight",
                "conv2.bias",
                "conv2.lin.weight"
            ]
        );
    }

    #[test]
    fn gcn_conv_self_loops_disconnected_node_is_identity() {
        // 2 nodes, no edges. Self-loops give each node deg=1, so
        // w(v,v) = 1. The convolution becomes h = (W x) + b for each
        // row independently.
        let mut conv = GcnConv::new(2, 2).unwrap();
        // Set W = I, b = 0 (parameters are owned by Parameter; cheat
        // by reaching through `set_data`).
        let w = t(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        conv.lin.weight.set_data(w);
        let b = t(&[0.0, 0.0], &[2]);
        conv.bias.set_data(b);
        let x = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let out = conv.forward(&x, &[]).unwrap();
        let got = out.data_vec().unwrap();
        // With W = I and no edges, out == x (after self-loop normalization
        // which is 1/sqrt(1)*1/sqrt(1) = 1).
        for (g, e) in got.iter().zip([1.0, 2.0, 3.0, 4.0].iter()) {
            assert!((g - e).abs() < 1e-5, "got={g} expected={e}");
        }
    }

    #[test]
    fn gcn_conv_two_node_chain_aggregates_neighbor() {
        // 2 nodes, one undirected edge 0<->1 (i.e. two directed edges).
        // edge_index = [[0, 1], [1, 0]] in COO. Self-loops add (0,0) and (1,1).
        // Augmented edges: (0,1) (1,0) (0,0) (1,1).
        // dst counts: deg[0]=2 (edges into 0: (1,0) and (0,0)), deg[1]=2.
        // deg_inv_sqrt = [1/sqrt(2), 1/sqrt(2)] for both nodes.
        // All four edge weights w(u,v) = (1/sqrt(2))^2 = 0.5.
        // With W=I, b=0:
        //   out[0] = 0.5 * x[1] + 0.5 * x[0]  (from edges (1,0) and (0,0))
        //   out[1] = 0.5 * x[0] + 0.5 * x[1]  (from edges (0,1) and (1,1))
        let mut conv = GcnConv::new(2, 2).unwrap();
        conv.lin.weight.set_data(t(&[1.0, 0.0, 0.0, 1.0], &[2, 2]));
        conv.bias.set_data(t(&[0.0, 0.0], &[2]));
        let x = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        // [src; dst] flat: src=[0,1] dst=[1,0]
        let edge_index = vec![0_i64, 1, 1, 0];
        let out = conv.forward(&x, &edge_index).unwrap();
        let got = out.data_vec().unwrap();
        // out[0] = 0.5*[3,4] + 0.5*[1,2] = [2, 3]
        // out[1] = 0.5*[1,2] + 0.5*[3,4] = [2, 3]
        let expect = [2.0, 3.0, 2.0, 3.0];
        for (g, e) in got.iter().zip(expect.iter()) {
            assert!((g - e).abs() < 1e-5, "got={g} expected={e}");
        }
    }

    #[test]
    fn gcn_net_forward_two_layer_chain() {
        // Smoke test: 2-layer GCN on a 3-node line graph; make sure
        // forward returns the right shape and is finite.
        let mut net = GcnNet::new(2, 4, 3).unwrap();
        for p in net.parameters_mut() {
            // Replace any default init with explicit small values so
            // outputs stay bounded.
            let shape = p.shape().to_vec();
            let n: usize = shape.iter().product();
            let data: Vec<f32> = (0..n).map(|i| 0.01_f32 * (i as f32 + 1.0)).collect();
            p.set_data(Tensor::from_storage(TensorStorage::cpu(data), shape, false).unwrap());
        }
        // 3 nodes, 2 directed edges: 0->1 and 1->2.
        let x = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let edge_index = vec![0_i64, 1, 1, 2];
        let out = net.forward(&x, &edge_index).unwrap();
        assert_eq!(out.shape(), &[3, 3]);
        let d = out.data_vec().unwrap();
        for v in d {
            assert!(v.is_finite(), "non-finite logit: {v}");
        }
    }
}
