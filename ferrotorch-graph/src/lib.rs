//! Graph neural network composition for ferrotorch.
//!
//! Phase D.1 of real-artifact-driven development: a 2-layer GCN
//! matching `torch_geometric.nn.GCNConv` with `add_self_loops=True,
//! normalize=True, improved=False, bias=True`, trained on the Cora
//! node-classification benchmark and pinned to `ferrotorch/gcn-cora`
//! on the HuggingFace Hub.
//!
//! Public surface:
//!
//! * [`Graph`] — plain-data container for one graph snapshot (node
//!   features, COO edge index, integer labels).
//! * [`GcnConv`] / [`GcnNet`] — the layer + 2-layer stack matching
//!   PyG's reference `examples/gcn.py`.
//! * [`load_gcn_net`] — pulls a pinned safetensors mirror into a
//!   ready-to-forward `GcnNet`, returning a [`DropReport`] for the
//!   state-dict audit rail.
//!
//! The message-passing primitive these modules call into
//! ([`ferrotorch_core::scatter_add_segments`]) lives in
//! `ferrotorch-core` so non-graph crates can reuse it.
//!
//! ## REQ status (per `.design/ferrotorch-graph/lib.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | impl: `pub mod data`, `pub mod gcn`, `pub mod safetensors_loader` in `ferrotorch-graph/src/lib.rs`; non-test consumer: `ferrotorch-graph/examples/gcn_inference_dump.rs` reaches `load_gcn_net` via the `safetensors_loader` re-export. |
//! | REQ-2 | SHIPPED | impl: `pub use data::Graph;`, `pub use gcn::{GcnConv, GcnNet};`, `pub use safetensors_loader::{DropReport, load_gcn_net};`; non-test consumer: `ferrotorch-graph/examples/gcn_inference_dump.rs` `use ferrotorch_graph::load_gcn_net;` and `let (net, report) = load_gcn_net(...);` resolve through these re-exports. |
//! | REQ-3 | SHIPPED | impl: the crate-level doc-comment above names `torch_geometric.nn.GCNConv`, the four PyG default flags, `Cora`, and `ferrotorch/gcn-cora`; non-test consumer: `ferrotorch-graph/README.md` reproduces the same contract and the example binary depends on these dimensions matching the safetensors mirror. |
//! | REQ-4 | SHIPPED | impl: this file is the re-export skeleton only — no `#![allow]`, no `unsafe`, no `pub fn` other than the re-exports; non-test consumer: the crate-root cleanliness is what allows `cargo clippy -p ferrotorch-graph --lib -- -D warnings` to pass for downstream binaries that pull the crate in. |

pub mod data;
pub mod gcn;
pub mod safetensors_loader;

pub use data::Graph;
pub use gcn::{GcnConv, GcnNet};
pub use safetensors_loader::{DropReport, load_gcn_net};
