//! Plain-data container for one graph in the GCN harness.
//!
//! Mirrors the subset of `torch_geometric.data.Data` that `GcnNet`'s
//! inference path needs: dense node features `x: [N, F]`, an undirected
//! `edge_index: [2, E]` adjacency list (`i64`, COO layout), and per-node
//! labels `y: [N]` (`i64`). Train / val / test masks are not part of the
//! inference path and are intentionally omitted.

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Tensor};

/// One static graph plus its node labels.
///
/// The harness loads exactly one snapshot of the Cora dataset, so this
/// is a value type rather than a `Dataset` trait. Larger graph datasets
/// (Citeseer, PubMed) fit the same shape and would slot in unchanged.
#[derive(Debug, Clone)]
pub struct Graph {
    /// Dense node features, shape `[num_nodes, num_features]`.
    pub x: Tensor<f32>,
    /// COO edge list, shape `[2, num_edges]`. Row 0 = source, row 1 =
    /// destination. Stored as flat `Vec<i64>` (not `Tensor<i64>`)
    /// because all graph-side ops that consume it (`scatter_add_segments`,
    /// self-loop concat, degree count) want a `&[i64]` directly and a
    /// detour through `IntTensor` would only add copies.
    pub edge_index: Vec<i64>,
    /// Number of edges (i.e. `edge_index.len() / 2`). Cached so the
    /// forward path does not have to re-derive it.
    pub num_edges: usize,
    /// Per-node integer labels, shape `[num_nodes]`. Not consumed by
    /// the forward pass but carried alongside so the harness can
    /// surface `argmax(logits) vs y` accuracy at the verdict line.
    pub y: Vec<i64>,
}

impl Graph {
    /// Construct a `Graph` from raw component buffers, validating
    /// shapes up front so a malformed mirror surfaces at load time
    /// rather than inside the forward pass.
    pub fn new(x: Tensor<f32>, edge_index: Vec<i64>, y: Vec<i64>) -> FerrotorchResult<Self> {
        if x.ndim() != 2 {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "Graph::new: x must be 2-D [N, F], got shape {:?}",
                    x.shape()
                ),
            });
        }
        let n = x.shape()[0];
        if edge_index.len() % 2 != 0 {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "Graph::new: edge_index length {} not divisible by 2 (must be [2, E])",
                    edge_index.len()
                ),
            });
        }
        let num_edges = edge_index.len() / 2;
        // Bounds-check every edge endpoint against num_nodes — saves
        // having to repeat the check inside the forward path.
        for (i, &v) in edge_index.iter().enumerate() {
            if v < 0 || (v as usize) >= n {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!("Graph::new: edge_index[{i}] = {v} out of range [0, {n})"),
                });
            }
        }
        if y.len() != n {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!("Graph::new: y length {} != num_nodes {n}", y.len()),
            });
        }
        Ok(Self {
            x,
            edge_index,
            num_edges,
            y,
        })
    }

    /// Number of nodes (`x.shape()[0]`).
    pub fn num_nodes(&self) -> usize {
        self.x.shape()[0]
    }

    /// Feature dimension (`x.shape()[1]`).
    pub fn num_features(&self) -> usize {
        self.x.shape()[1]
    }

    /// Source endpoint of edge `e` (row 0 of the COO edge_index).
    #[inline]
    pub fn edge_src(&self, e: usize) -> i64 {
        // Row-major: row 0 occupies indices `0..num_edges`.
        self.edge_index[e]
    }

    /// Destination endpoint of edge `e` (row 1 of the COO edge_index).
    #[inline]
    pub fn edge_dst(&self, e: usize) -> i64 {
        self.edge_index[self.num_edges + e]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::TensorStorage;

    fn x_tensor(rows: usize, cols: usize) -> Tensor<f32> {
        let data: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
        Tensor::from_storage(TensorStorage::cpu(data), vec![rows, cols], false).unwrap()
    }

    #[test]
    fn graph_constructs_with_consistent_shapes() {
        // 3 nodes, 2 features each; 2 directed edges: 0->1, 1->2.
        let g = Graph::new(
            x_tensor(3, 2),
            vec![0, 1, /* src row done */ 1, 2 /* dst row */],
            vec![0, 1, 2],
        )
        .unwrap();
        assert_eq!(g.num_nodes(), 3);
        assert_eq!(g.num_features(), 2);
        assert_eq!(g.num_edges, 2);
        assert_eq!(g.edge_src(0), 0);
        assert_eq!(g.edge_dst(0), 1);
        assert_eq!(g.edge_src(1), 1);
        assert_eq!(g.edge_dst(1), 2);
    }

    #[test]
    fn graph_rejects_oob_edge_endpoint() {
        let err = Graph::new(x_tensor(2, 1), vec![0, 5, 1, 0], vec![0, 1]);
        assert!(err.is_err(), "expected oob rejection, got {err:?}");
    }

    #[test]
    fn graph_rejects_label_count_mismatch() {
        let err = Graph::new(x_tensor(2, 1), vec![0, 1, 1, 0], vec![0]);
        assert!(err.is_err());
    }

    #[test]
    fn graph_rejects_odd_edge_index_length() {
        let err = Graph::new(x_tensor(2, 1), vec![0, 1, 1], vec![0, 1]);
        assert!(err.is_err());
    }
}
