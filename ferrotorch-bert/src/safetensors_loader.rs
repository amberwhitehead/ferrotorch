//! Helpers that turn a path-to-safetensors into a loaded
//! [`BertModel`] / [`SentenceTransformer`].
//!
//! The shared key remapper is the only non-trivial piece: HuggingFace
//! BERT checkpoints use `encoder.layer.{i}.attention.{self,output}.{...}`,
//! which is exactly the layout `BertModel::named_parameters()` already
//! exposes. The remap is therefore identity — but [`load_bert_model`]
//! keeps the indirection so any future name drift (e.g. dropping the
//! `encoder.` prefix, or absorbing the pooler) stays localised to one
//! function.

use std::path::Path;

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float};
use ferrotorch_nn::module::StateDict;
use ferrotorch_serialize::load_safetensors;

use crate::config::BertConfig;
use crate::model::{BertModel, DropReport, SentenceTransformer};

/// Upstream HF BERT key prefixes the loader does NOT consume (they are
/// either non-parameter buffers or pooler-only and would be dropped by
/// [`BertModel::load_hf_state_dict`] anyway).
///
/// Loading them through `ferrotorch_serialize::load_safetensors::<f32>`
/// would fail loudly because `embeddings.position_ids` is stored as
/// `I64` upstream (not representable as `f32`). We filter those out
/// before the decode and let the model's loader still see the
/// `pooler.*` keys so its drop-report API is exercised end-to-end.
fn key_is_skippable_at_decode(key: &str) -> bool {
    // Skip ONLY the int64 buffer at decode time. The pooler keys are
    // f32 and are dropped by `BertModel::load_hf_state_dict` after they
    // appear in the state dict (and so end up in the DropReport, which
    // is the audit-trail surface we keep alive).
    key == "embeddings.position_ids"
}

/// Read a single non-sharded safetensors file into a typed `StateDict`,
/// dropping any keys flagged by [`key_is_skippable_at_decode`] BEFORE
/// the generic-`T` decode (so a non-`T`-decodable int64 buffer cannot
/// poison the `load_safetensors::<f32>` cast).
fn load_safetensors_filtered<T: Float>(weights_path: &Path) -> FerrotorchResult<StateDict<T>> {
    use safetensors::SafeTensors;

    let bytes = std::fs::read(weights_path).map_err(|e| FerrotorchError::InvalidArgument {
        message: format!(
            "load_safetensors_filtered: failed to read {}: {e}",
            weights_path.display()
        ),
    })?;
    let st =
        SafeTensors::deserialize(&bytes).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!(
                "load_safetensors_filtered: failed to parse {}: {e}",
                weights_path.display()
            ),
        })?;
    let mut keep: Vec<String> = Vec::new();
    for k in st.names() {
        let s: &str = k.as_str();
        if !key_is_skippable_at_decode(s) {
            keep.push(String::from(s));
        }
    }
    // Re-serialize only the kept tensors into an in-memory safetensors
    // blob and feed that to `load_safetensors::<T>`. This pays one extra
    // pass but reuses the audited generic decoder rather than
    // re-implementing dtype dispatch here.
    let mut subset: Vec<(String, safetensors::tensor::TensorView<'_>)> =
        Vec::with_capacity(keep.len());
    for k in &keep {
        let v = st.tensor(k).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!(
                "load_safetensors_filtered: missing tensor {k:?} after filter: {e}"
            ),
        })?;
        subset.push((k.clone(), v));
    }
    let serialized = safetensors::serialize(subset, &None).map_err(|e| {
        FerrotorchError::InvalidArgument {
            message: format!("load_safetensors_filtered: re-serialize failed: {e}"),
        }
    })?;
    let tmp = tempfile::NamedTempFile::new().map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("load_safetensors_filtered: tempfile: {e}"),
    })?;
    std::fs::write(tmp.path(), &serialized).map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("load_safetensors_filtered: tempfile write: {e}"),
    })?;
    load_safetensors::<T>(tmp.path())
}

/// Load a [`BertModel`] from a `model.safetensors` file plus a
/// pre-parsed config.
///
/// Returns the populated model and the [`DropReport`] documenting
/// which upstream HF keys were intentionally not consumed
/// (`embeddings.position_ids` and any `pooler.*` keys present in the
/// checkpoint). Pass `strict=true` to fail loudly if the upstream
/// safetensors carries `pooler.*` keys.
///
/// # Errors
///
/// Propagates safetensors parse errors, [`BertModel`] construction
/// errors, and any per-key shape / strict-mode mismatch from the
/// underlying load.
pub fn load_bert_model<T: Float>(
    weights_path: &Path,
    cfg: BertConfig,
    strict: bool,
) -> FerrotorchResult<(BertModel<T>, DropReport)> {
    // Probe the raw safetensors to learn which upstream keys are
    // present (so the DropReport reflects the upstream checkpoint, not
    // the post-filter view).
    let raw_bytes =
        std::fs::read(weights_path).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!(
                "load_bert_model: failed to read safetensors {}: {e}",
                weights_path.display()
            ),
        })?;
    let raw_st = safetensors::SafeTensors::deserialize(&raw_bytes).map_err(|e| {
        FerrotorchError::InvalidArgument {
            message: format!(
                "load_bert_model: failed to parse safetensors {}: {e}",
                weights_path.display()
            ),
        }
    })?;
    let upstream_keys: Vec<String> = raw_st
        .names()
        .iter()
        .map(|s| String::from(s.as_str()))
        .collect();
    drop(raw_st);
    drop(raw_bytes);

    // Load only the f32-decodable parameters (drops int64 position_ids
    // at decode time — that key is a buffer, not a parameter).
    let mut state = load_safetensors_filtered::<T>(weights_path).map_err(|e| {
        FerrotorchError::InvalidArgument {
            message: format!(
                "load_bert_model: failed to decode safetensors {}: {e}",
                weights_path.display()
            ),
        }
    })?;

    // Re-add a placeholder entry for `embeddings.position_ids` (if it
    // was present in the upstream file) so the model's DropReport
    // captures it as an intentionally-dropped upstream key. The
    // placeholder tensor is never consumed — the loader drops it
    // before reaching any parameter slot.
    if upstream_keys.iter().any(|k| k == "embeddings.position_ids") {
        state.insert(
            "embeddings.position_ids".to_string(),
            ferrotorch_core::zeros::<T>(&[1])?,
        );
    }

    let mut model = BertModel::<T>::new(cfg)?;
    let report = model.load_hf_state_dict(&state, strict)?;
    Ok((model, report))
}

/// Load a [`SentenceTransformer`] wrapping a [`BertModel`] from
/// `model.safetensors` + config + the pooling normalize flag.
///
/// `normalize` is the value of `pooling_mode_mean_tokens` cross-checked
/// against the optional `2_Normalize` module — for
/// `sentence-transformers/all-MiniLM-L6-v2` the caller passes `true`.
///
/// # Errors
///
/// See [`load_bert_model`].
pub fn load_sentence_transformer<T: Float>(
    weights_path: &Path,
    cfg: BertConfig,
    normalize: bool,
    strict: bool,
) -> FerrotorchResult<(SentenceTransformer<T>, DropReport)> {
    let (bert, report) = load_bert_model::<T>(weights_path, cfg, strict)?;
    Ok((SentenceTransformer { bert, normalize }, report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_nn::module::Module;
    use ferrotorch_serialize::save_safetensors;
    use std::path::PathBuf;

    fn tiny_cfg() -> BertConfig {
        BertConfig {
            vocab_size: 32,
            type_vocab_size: 2,
            max_position_embeddings: 16,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
        }
    }

    fn tmp_safetensors_from(model: &BertModel<f32>) -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        save_safetensors(&model.state_dict(), &path).unwrap();
        (dir, path)
    }

    #[test]
    fn round_trip_safetensors_into_bert_model() {
        let src = BertModel::<f32>::new(tiny_cfg()).unwrap();
        let (_d, p) = tmp_safetensors_from(&src);
        let (dst, report) = load_bert_model::<f32>(&p, tiny_cfg(), true).unwrap();
        assert!(!report.dropped_position_ids);
        assert!(report.dropped_pooler.is_empty());
        // Same forward.
        let ids = vec![1u32, 5, 7];
        let a = src.forward_from_ids(&ids, None).unwrap();
        let b = dst.forward_from_ids(&ids, None).unwrap();
        for (x, y) in a.data().unwrap().iter().zip(b.data().unwrap().iter()) {
            assert!((x - y).abs() < 1e-6);
        }
    }

    #[test]
    fn round_trip_into_sentence_transformer() {
        let src = BertModel::<f32>::new(tiny_cfg()).unwrap();
        let (_d, p) = tmp_safetensors_from(&src);
        let (st, _r) = load_sentence_transformer::<f32>(&p, tiny_cfg(), true, true).unwrap();
        let ids = vec![1u32, 2, 3];
        let out = st.encode(&ids, None, None).unwrap();
        assert_eq!(out.shape(), &[1, 8]);
        let sq: f32 = out.data().unwrap().iter().map(|v| v * v).sum();
        assert!((sq.sqrt() - 1.0).abs() < 1e-5);
    }
}
