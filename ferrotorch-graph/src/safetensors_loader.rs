//! Helpers that load a [`GcnNet`] from a `model.safetensors` mirror.
//!
//! The pinned `ferrotorch/gcn-cora` mirror stores the upstream PyG
//! `state_dict()` verbatim — the keys are exactly:
//!
//! ```text
//! conv1.bias
//! conv1.lin.weight
//! conv2.bias
//! conv2.lin.weight
//! ```
//!
//! These are the same keys [`GcnNet::named_parameters`] produces, so
//! the loader is a pure pass-through into the standard
//! `Module::load_state_dict` machinery. The wrapper here exists for
//! parity with the other crate-level `load_*` entry points and to
//! return a [`DropReport`] documenting upstream keys that were
//! intentionally not consumed (per the #1141 audit rail: every key
//! must either land in a parameter or appear in the report).

use std::path::Path;

use ferrotorch_core::{FerrotorchError, FerrotorchResult};
use ferrotorch_nn::module::Module;
use ferrotorch_serialize::load_safetensors;

use crate::gcn::GcnNet;

/// Audit trail returned by [`load_gcn_net`].
///
/// `unmapped` lists every upstream safetensors key that did NOT match
/// a parameter on the ferrotorch `GcnNet`. For the canonical PyG
/// `GCNConv`-pair state dict this is always empty — any non-empty
/// entry on a real pin is a state-dict-drop bug (the #1141 class of
/// failure) and the loader propagates it loudly when `strict=true`.
#[derive(Debug, Default, Clone)]
pub struct DropReport {
    /// Upstream keys present in the safetensors but not mapped to a
    /// parameter on `GcnNet`. Empty for a clean pin.
    pub unmapped: Vec<String>,
}

/// Load a [`GcnNet`] from `weights_path` (a `model.safetensors` file)
/// using `in_features`, `hidden`, `num_classes` to size the model.
///
/// Returns the loaded model plus a [`DropReport`] for the audit rail.
///
/// `strict=true` errors loudly if any upstream key cannot be mapped;
/// `strict=false` records the unmapped keys in the report and
/// continues. Either way, all *expected* parameter keys must be
/// present in the state dict — a missing key is always fatal.
///
/// # Errors
///
/// Forwards safetensors parse errors, `GcnNet` construction errors,
/// and any per-key shape mismatch from `Module::load_state_dict`.
pub fn load_gcn_net(
    weights_path: &Path,
    in_features: usize,
    hidden: usize,
    num_classes: usize,
    strict: bool,
) -> FerrotorchResult<(GcnNet, DropReport)> {
    let state =
        load_safetensors::<f32>(weights_path).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!(
                "load_gcn_net: failed to decode safetensors {}: {e}",
                weights_path.display()
            ),
        })?;

    let mut net = GcnNet::new(in_features, hidden, num_classes)?;
    let expected: std::collections::HashSet<String> =
        net.named_parameters().into_iter().map(|(n, _)| n).collect();
    let mut unmapped: Vec<String> = Vec::new();
    for k in state.keys() {
        if !expected.contains(k) {
            unmapped.push(k.clone());
        }
    }
    unmapped.sort();
    if strict && !unmapped.is_empty() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("load_gcn_net: unmapped upstream keys (strict mode): {unmapped:?}"),
        });
    }

    // Filter to only the keys `Module::load_state_dict` knows about
    // (otherwise it would itself reject extras in strict mode and
    // bypass our richer DropReport).
    let filtered: std::collections::HashMap<String, ferrotorch_core::Tensor<f32>> = state
        .into_iter()
        .filter(|(k, _)| expected.contains(k))
        .collect();
    net.load_state_dict(&filtered, /* strict = */ true)?;
    Ok((net, DropReport { unmapped }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_serialize::save_safetensors;

    #[test]
    fn round_trip_into_gcn_net() {
        // Build a tiny GcnNet, dump its state_dict, load it back, and
        // confirm the named_parameters' tensor values match exactly.
        let src = GcnNet::new(4, 3, 2).unwrap();
        // Snapshot expected (name, data) before consuming src.
        let expected: Vec<(String, Vec<f32>)> = src
            .named_parameters()
            .into_iter()
            .map(|(n, p)| (n, p.tensor().data_vec().unwrap()))
            .collect();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        save_safetensors(&src.state_dict(), &path).unwrap();
        let (dst, report) = load_gcn_net(&path, 4, 3, 2, /* strict = */ true).unwrap();
        assert!(report.unmapped.is_empty(), "report = {report:?}");
        let dst_params: std::collections::HashMap<String, Vec<f32>> = dst
            .named_parameters()
            .into_iter()
            .map(|(n, p)| (n, p.tensor().data_vec().unwrap()))
            .collect();
        for (k, vexp) in &expected {
            let v = &dst_params[k];
            assert_eq!(v.len(), vexp.len(), "len mismatch for {k}");
            for (a, b) in v.iter().zip(vexp.iter()) {
                assert!((a - b).abs() < 1e-7, "value mismatch in {k}: {a} vs {b}");
            }
        }
    }
}
