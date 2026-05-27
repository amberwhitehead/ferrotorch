//! Wave-E audit (#1542): #1481 Graph::new + accessor wiring in the GCN
//! inference example. Verifies the production example actually goes
//! through the validated `Graph` path rather than only importing the type.

#![allow(clippy::approx_constant)]

use std::fs;
use std::path::PathBuf;

use ferrotorch_core::{Tensor, TensorStorage};
use ferrotorch_graph::Graph;

fn x_tensor(rows: usize, cols: usize) -> Tensor<f32> {
    let data: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
    Tensor::from_storage(TensorStorage::cpu(data), vec![rows, cols], false).unwrap()
}

/// `Graph::new` is callable from the crate root and returns a populated
/// Graph for valid inputs.
#[test]
fn audit_1481_graph_new_callable_from_crate_root() {
    let g = Graph::new(x_tensor(3, 2), vec![0, 1, 1, 2], vec![0, 1, 2])
        .expect("valid Graph::new must succeed");
    assert_eq!(g.num_nodes(), 3);
    assert_eq!(g.num_features(), 2);
    assert_eq!(g.num_edges, 2);
}

/// `edge_src` / `edge_dst` return COO row-major endpoints.
#[test]
fn audit_1481_edge_accessors_match_coo_rowmajor() {
    let g = Graph::new(x_tensor(3, 2), vec![0, 1, 1, 2], vec![0, 1, 2]).unwrap();
    assert_eq!(g.edge_src(0), 0);
    assert_eq!(g.edge_src(1), 1);
    assert_eq!(g.edge_dst(0), 1);
    assert_eq!(g.edge_dst(1), 2);
}

/// `Graph::new` rejects out-of-range endpoints, even-length violations,
/// and y-length mismatch — full invariant matrix.
#[test]
fn audit_1481_graph_new_rejects_invalid_inputs() {
    // out-of-range endpoint
    assert!(Graph::new(x_tensor(2, 1), vec![0, 5, 1, 0], vec![0, 1]).is_err());
    // odd edge_index length
    assert!(Graph::new(x_tensor(2, 1), vec![0, 1, 1], vec![0, 1]).is_err());
    // y length mismatch
    assert!(Graph::new(x_tensor(2, 1), vec![0, 1, 1, 0], vec![0]).is_err());
}

/// The example consumer must actually CALL Graph::new (not just import the
/// type). Textual audit of the production example.
#[test]
fn audit_1481_example_actually_calls_graph_new() {
    let example_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples/gcn_inference_dump.rs");
    let src = fs::read_to_string(&example_path)
        .unwrap_or_else(|e| panic!("read {example_path:?}: {e}"));

    assert!(
        src.contains("Graph::new("),
        "gcn_inference_dump.rs must call `Graph::new(...)` — found no \
         such call. The #1481 closure is vocab-only without this."
    );
    assert!(
        src.contains("graph.num_nodes()") || src.contains(".num_nodes()"),
        "example must read `graph.num_nodes()` to exercise the REQ-3 \
         accessor"
    );
    assert!(
        src.contains("graph.num_features()") || src.contains(".num_features()"),
        "example must read `graph.num_features()` to exercise the REQ-3 \
         accessor"
    );
    assert!(
        src.contains("graph.edge_src(") || src.contains(".edge_src("),
        "example must invoke `graph.edge_src(...)` to exercise the REQ-4 \
         accessor"
    );
    assert!(
        src.contains("graph.edge_dst(") || src.contains(".edge_dst("),
        "example must invoke `graph.edge_dst(...)` to exercise the REQ-4 \
         accessor"
    );
}

/// The example's forward path must consume the Graph's validated buffers,
/// not the loose tensors built before Graph::new — otherwise the
/// validation is bypassed.
#[test]
fn audit_1481_example_forward_uses_graph_buffers() {
    let example_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples/gcn_inference_dump.rs");
    let src = fs::read_to_string(&example_path).unwrap();
    assert!(
        src.contains("net.forward(&graph.x, &graph.edge_index)"),
        "example must drive forward through Graph fields, not the loose \
         tensors. Current source must contain \
         `net.forward(&graph.x, &graph.edge_index)`."
    );
}
