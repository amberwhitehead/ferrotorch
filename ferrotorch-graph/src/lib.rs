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

pub mod data;
pub mod gcn;
pub mod safetensors_loader;

pub use data::Graph;
pub use gcn::{GcnConv, GcnNet};
pub use safetensors_loader::{DropReport, load_gcn_net};
