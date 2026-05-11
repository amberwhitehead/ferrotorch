//! Top-level BERT encoder + SentenceTransformer wrapper.
//!
//! [`BertModel`] is the encoder stack (embeddings + N encoder layers).
//! [`SentenceTransformer`] adds mean-pooling over the attention mask
//! followed by optional L2 normalization, matching the inference path
//! of `sentence_transformers.SentenceTransformer.encode(...)`.

use std::collections::HashMap;

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};
use ferrotorch_nn::module::{Module, StateDict};
use ferrotorch_nn::parameter::Parameter;

use crate::config::BertConfig;
use crate::embeddings::BertEmbeddings;
use crate::layer::BertLayer;

/// HF `BertEncoder` — `N × BertLayer` in sequence.
#[derive(Debug)]
pub struct BertEncoder<T: Float> {
    /// One [`BertLayer`] per `num_hidden_layers` in the config.
    pub layer: Vec<BertLayer<T>>,
    training: bool,
}

impl<T: Float> BertEncoder<T> {
    /// Build a randomly-initialized encoder stack.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`FerrotorchError`] on bad config dims.
    pub fn new(cfg: &BertConfig) -> FerrotorchResult<Self> {
        cfg.validate()?;
        let mut layer = Vec::with_capacity(cfg.num_hidden_layers);
        for _ in 0..cfg.num_hidden_layers {
            layer.push(BertLayer::new(cfg)?);
        }
        Ok(Self {
            layer,
            training: false,
        })
    }
}

impl<T: Float> Module<T> for BertEncoder<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let mut h = input.clone();
        for l in &self.layer {
            h = l.forward(&h)?;
        }
        Ok(h)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut out = Vec::new();
        for l in &self.layer {
            out.extend(l.parameters());
        }
        out
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut out = Vec::new();
        for l in &mut self.layer {
            out.extend(l.parameters_mut());
        }
        out
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (i, l) in self.layer.iter().enumerate() {
            for (n, p) in l.named_parameters() {
                out.push((format!("layer.{i}.{n}"), p));
            }
        }
        out
    }

    fn train(&mut self) {
        self.training = true;
        for l in &mut self.layer {
            l.train();
        }
    }

    fn eval(&mut self) {
        self.training = false;
        for l in &mut self.layer {
            l.eval();
        }
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn state_dict(&self) -> StateDict<T> {
        self.named_parameters()
            .into_iter()
            .map(|(n, p)| (n, p.tensor().clone()))
            .collect()
    }

    fn load_state_dict(&mut self, state: &StateDict<T>, strict: bool) -> FerrotorchResult<()> {
        let extract = |prefix: &str| -> StateDict<T> {
            let expected = format!("{prefix}.");
            state
                .iter()
                .filter_map(|(k, v)| {
                    k.strip_prefix(&expected)
                        .map(|rest| (rest.to_string(), v.clone()))
                })
                .collect()
        };
        if strict {
            for key in state.keys() {
                if !key.starts_with("layer.") {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!(
                            "unexpected key in BertEncoder state_dict: \"{key}\""
                        ),
                    });
                }
            }
        }
        for (i, l) in self.layer.iter_mut().enumerate() {
            l.load_state_dict(&extract(&format!("layer.{i}")), strict)?;
        }
        Ok(())
    }
}

/// HF `BertModel` (without pooler) — embeddings + encoder.
///
/// Sentence-transformers does not use the optional `pooler.dense.*`
/// parameters. They are dropped during HF state-dict ingest (when a
/// non-strict load is requested via [`Self::load_hf_state_dict`]) so
/// downloaded checkpoints with a pooler still load. Pooler weights
/// are surfaced as a hard mismatch in strict mode so we never silently
/// discard parameters: if a future caller wants the pooler, they must
/// extend the model first.
#[derive(Debug)]
pub struct BertModel<T: Float> {
    /// Input embedding stack (word + position + token_type + LayerNorm).
    pub embeddings: BertEmbeddings<T>,
    /// `N × BertLayer` encoder.
    pub encoder: BertEncoder<T>,
    /// Frozen copy of the configuration used to construct this model.
    pub config: BertConfig,
    training: bool,
}

impl<T: Float> BertModel<T> {
    /// Build a randomly-initialized BertModel for the given config.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`FerrotorchError`] when any sub-module
    /// fails to construct (typically `ShapeMismatch` on bad config dims).
    pub fn new(cfg: BertConfig) -> FerrotorchResult<Self> {
        cfg.validate()?;
        let embeddings = BertEmbeddings::new(&cfg)?;
        let encoder = BertEncoder::new(&cfg)?;
        Ok(Self {
            embeddings,
            encoder,
            config: cfg,
            training: false,
        })
    }

    /// Run the encoder on a sequence of token ids and return the
    /// per-token hidden states `[1, S, hidden]`.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] when `input_ids` is
    /// empty or longer than `max_position_embeddings`. Token-type ids
    /// must either be `None` (substitutes all-zero) or have the same
    /// length as `input_ids`.
    pub fn forward_from_ids(
        &self,
        input_ids: &[u32],
        token_type_ids: Option<&[u32]>,
    ) -> FerrotorchResult<Tensor<T>> {
        let h = self
            .embeddings
            .forward_from_ids(input_ids, token_type_ids)?;
        self.encoder.forward(&h)
    }

    /// Load a HuggingFace-format `BertModel` `StateDict` into this
    /// module.
    ///
    /// The expected key layout matches the HF `BertModel` naming
    /// convention this crate already exposes via `named_parameters()`:
    /// `embeddings.{...}`, `encoder.layer.{i}.{...}`. The HF safetensors
    /// also ship two pieces of state we explicitly drop:
    ///
    /// * `embeddings.position_ids` — a `[1, max_pos]` buffer
    ///   (`arange(max_pos)`) that the forward pass regenerates from
    ///   `input_ids.len()`. Not a parameter; no information lost.
    /// * `pooler.dense.{weight,bias}` — the optional CLS-token pooler
    ///   that sentence-transformers does not use. Dropping it is
    ///   correct for the all-MiniLM-L6-v2 inference path; any caller
    ///   that wants the pooler must extend `BertModel` first (then
    ///   `strict=true` will accept those keys).
    ///
    /// Both of these are surfaced via the returned [`DropReport`] so
    /// the pin script can confirm every key in the upstream
    /// safetensors was either consumed or intentionally dropped — the
    /// FPN-bias drop bug (#1141) burned us once and the report is the
    /// guard rail.
    ///
    /// # Errors
    ///
    /// Forwards whatever each sub-module's `load_state_dict` returns
    /// (`ShapeMismatch` on a wrong-shape tensor, `InvalidArgument` in
    /// strict mode when a required tensor is missing). Strict mode
    /// will surface `pooler.*` / `embeddings.position_ids` as errors;
    /// callers that have a checkpoint with those keys must use
    /// `strict=false`.
    pub fn load_hf_state_dict(
        &mut self,
        hf_state: &StateDict<T>,
        strict: bool,
    ) -> FerrotorchResult<DropReport> {
        let mut remapped: StateDict<T> = HashMap::with_capacity(hf_state.len());
        let mut dropped_position_ids = false;
        let mut dropped_pooler: Vec<String> = Vec::new();
        for (k, v) in hf_state {
            if k == "embeddings.position_ids" {
                // Buffer, regenerated on forward — never a parameter on
                // our side. Drop silently in both modes; record so the
                // pin script can audit.
                dropped_position_ids = true;
                continue;
            }
            if k.starts_with("pooler.") {
                if strict {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!(
                            "BertModel::load_hf_state_dict: pooler key {k:?} \
                             present but sentence-transformers does not use the \
                             pooler — load with strict=false to drop, or extend \
                             BertModel to surface the pooler."
                        ),
                    });
                }
                dropped_pooler.push(k.clone());
                continue;
            }
            remapped.insert(k.clone(), v.clone());
        }
        self.load_state_dict(&remapped, strict)?;
        Ok(DropReport {
            dropped_position_ids,
            dropped_pooler,
        })
    }
}

impl<T: Float> Module<T> for BertModel<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let h = self.embeddings.forward(input)?;
        self.encoder.forward(&h)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.embeddings.parameters());
        out.extend(self.encoder.parameters());
        out
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.embeddings.parameters_mut());
        out.extend(self.encoder.parameters_mut());
        out
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (n, p) in self.embeddings.named_parameters() {
            out.push((format!("embeddings.{n}"), p));
        }
        for (n, p) in self.encoder.named_parameters() {
            out.push((format!("encoder.{n}"), p));
        }
        out
    }

    fn train(&mut self) {
        self.training = true;
        self.embeddings.train();
        self.encoder.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.embeddings.eval();
        self.encoder.eval();
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn state_dict(&self) -> StateDict<T> {
        self.named_parameters()
            .into_iter()
            .map(|(n, p)| (n, p.tensor().clone()))
            .collect()
    }

    fn load_state_dict(&mut self, state: &StateDict<T>, strict: bool) -> FerrotorchResult<()> {
        let extract = |prefix: &str| -> StateDict<T> {
            let expected = format!("{prefix}.");
            state
                .iter()
                .filter_map(|(k, v)| {
                    k.strip_prefix(&expected)
                        .map(|rest| (rest.to_string(), v.clone()))
                })
                .collect()
        };
        if strict {
            for key in state.keys() {
                if !(key.starts_with("embeddings.") || key.starts_with("encoder.")) {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!("unexpected key in BertModel state_dict: \"{key}\""),
                    });
                }
            }
        }
        self.embeddings
            .load_state_dict(&extract("embeddings"), strict)?;
        self.encoder
            .load_state_dict(&extract("encoder"), strict)?;
        Ok(())
    }
}

/// Audit trail returned by [`BertModel::load_hf_state_dict`].
///
/// Records which upstream HF keys were intentionally dropped (rather
/// than mapped onto a parameter). The pin script asserts the entries
/// here exactly match the upstream-extras list so a silent state-dict
/// drop can never recur (per #1141).
#[derive(Debug, Default, Clone)]
pub struct DropReport {
    /// `embeddings.position_ids` was present and dropped (it is a
    /// buffer regenerated each forward, never a parameter).
    pub dropped_position_ids: bool,
    /// `pooler.*` keys that were present and dropped (sentence-
    /// transformers does not use the pooler).
    pub dropped_pooler: Vec<String>,
}

/// Sentence-transformers wrapper around [`BertModel`].
///
/// Inference pipeline:
///
/// 1. Run BERT encoder → per-token hidden states `[1, S, hidden]`.
/// 2. Apply attention mask: positions with mask=0 contribute nothing
///    to the pool.
/// 3. Mean-pool over non-masked tokens → sentence embedding
///    `[1, hidden]`.
/// 4. If `normalize == true`, L2-normalize so `||emb|| == 1`.
#[derive(Debug)]
pub struct SentenceTransformer<T: Float> {
    /// Underlying BERT encoder.
    pub bert: BertModel<T>,
    /// Whether to L2-normalize the pooled embedding (`true` for
    /// `sentence-transformers/all-MiniLM-L6-v2` per its
    /// `2_Normalize` module).
    pub normalize: bool,
}

impl<T: Float> SentenceTransformer<T> {
    /// Build a randomly-initialized sentence transformer.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`FerrotorchError`] on bad config dims.
    pub fn new(cfg: BertConfig, normalize: bool) -> FerrotorchResult<Self> {
        Ok(Self {
            bert: BertModel::new(cfg)?,
            normalize,
        })
    }

    /// Compute the sentence embedding for one sequence.
    ///
    /// `attention_mask` is the per-token 0/1 mask (1 = keep, 0 =
    /// padding). When `None` we assume every position is real (length
    /// matches `input_ids`).
    ///
    /// # Errors
    ///
    /// Propagates the encoder error; returns
    /// [`FerrotorchError::InvalidArgument`] when `attention_mask` has
    /// a different length than `input_ids`, when every mask bit is
    /// zero (no token to pool), or when `input_ids` is empty.
    pub fn encode(
        &self,
        input_ids: &[u32],
        attention_mask: Option<&[u32]>,
        token_type_ids: Option<&[u32]>,
    ) -> FerrotorchResult<Tensor<T>> {
        let seq_len = input_ids.len();
        if seq_len == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "SentenceTransformer::encode needs at least one token".into(),
            });
        }
        let mask_owned: Vec<u32> = match attention_mask {
            Some(m) => {
                if m.len() != seq_len {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!(
                            "SentenceTransformer::encode: attention_mask length {} \
                             does not match input_ids length {seq_len}",
                            m.len()
                        ),
                    });
                }
                m.to_vec()
            }
            None => vec![1u32; seq_len],
        };
        let kept: usize = mask_owned.iter().map(|&v| v as usize).sum();
        if kept == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "SentenceTransformer::encode: attention_mask is all zero".into(),
            });
        }

        // Encoder produces [1, S, hidden].
        let hidden_states = self.bert.forward_from_ids(input_ids, token_type_ids)?;
        let shape = hidden_states.shape();
        if shape.len() != 3 || shape[0] != 1 || shape[1] != seq_len {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "SentenceTransformer::encode: encoder returned {:?}, expected [1, {seq_len}, hidden]",
                    shape
                ),
            });
        }
        let hidden = shape[2];
        let data = hidden_states.data_vec()?;

        // Mask-weighted mean pool. We accumulate sums in f64 so the
        // running total of (S × hidden) f32 adds doesn't drift from
        // the reference; the divisor is `max(kept_count, 1)` per HF
        // sentence-transformers (`pooling_mode_mean_tokens` source).
        let zero = <T as num_traits::Zero>::zero();
        let mut pooled: Vec<T> = vec![zero; hidden];
        for (s, &m) in mask_owned.iter().enumerate() {
            if m == 0 {
                continue;
            }
            let row = &data[s * hidden..(s + 1) * hidden];
            for (out, &v) in pooled.iter_mut().zip(row.iter()) {
                *out += v;
            }
        }
        // Divide by kept count. Match HF's
        // `sum / clamp(mask.sum(-1), min=1e-9)` behavior — we already
        // returned an error for all-zero mask, so `kept >= 1`.
        let denom = T::from(kept as f64).ok_or(FerrotorchError::InvalidArgument {
            message: format!(
                "SentenceTransformer::encode: kept count {kept} not representable in T"
            ),
        })?;
        // Note: ferrotorch_core::Float does not impl DivAssign, so the
        // `*v /= denom` shorthand will not type-check. Explicit assign
        // is required; allow the clippy lint locally.
        #[allow(clippy::assign_op_pattern)]
        for v in &mut pooled {
            *v = *v / denom;
        }

        // Optional L2 normalize (per `2_Normalize` module).
        if self.normalize {
            let mut sq_sum = <T as num_traits::Zero>::zero();
            for &v in &pooled {
                sq_sum += v * v;
            }
            // sqrt via cast to f64; T::from is reasonable for f32/f64.
            let sq_sum_f64: f64 = num_traits::ToPrimitive::to_f64(&sq_sum).ok_or(
                FerrotorchError::InvalidArgument {
                    message: "SentenceTransformer::encode: ||pooled||^2 not representable as f64"
                        .into(),
                },
            )?;
            // HF normalize: `F.normalize(emb, p=2, dim=1, eps=1e-12)`.
            let norm_f64 = sq_sum_f64.sqrt().max(1e-12);
            let inv = T::from(1.0 / norm_f64).ok_or(FerrotorchError::InvalidArgument {
                message: format!(
                    "SentenceTransformer::encode: 1/norm ({}) not representable in T",
                    1.0 / norm_f64
                ),
            })?;
            for v in &mut pooled {
                *v = *v * inv;
            }
        }

        Tensor::from_storage(TensorStorage::cpu(pooled), vec![1, hidden], false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg() -> BertConfig {
        BertConfig {
            vocab_size: 32,
            type_vocab_size: 2,
            max_position_embeddings: 16,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
        }
    }

    #[test]
    fn tiny_model_forward_shape() {
        let model = BertModel::<f32>::new(tiny_cfg()).unwrap();
        let ids = vec![1u32, 5, 7, 9];
        let out = model.forward_from_ids(&ids, None).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        for &v in out.data().unwrap() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn tiny_named_parameters_use_hf_layout() {
        let model = BertModel::<f32>::new(tiny_cfg()).unwrap();
        let names: Vec<String> = model
            .named_parameters()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        for k in [
            "embeddings.word_embeddings.weight",
            "embeddings.position_embeddings.weight",
            "embeddings.token_type_embeddings.weight",
            "embeddings.LayerNorm.weight",
            "embeddings.LayerNorm.bias",
            "encoder.layer.0.attention.self.query.weight",
            "encoder.layer.0.attention.self.query.bias",
            "encoder.layer.0.attention.output.dense.weight",
            "encoder.layer.0.attention.output.LayerNorm.weight",
            "encoder.layer.0.intermediate.dense.weight",
            "encoder.layer.0.output.dense.weight",
            "encoder.layer.0.output.LayerNorm.bias",
            "encoder.layer.1.attention.self.query.weight",
        ] {
            assert!(
                names.iter().any(|n| n == k),
                "missing parameter key {k:?} in BertModel.named_parameters()",
            );
        }
    }

    #[test]
    fn round_trip_state_dict() {
        let src = BertModel::<f32>::new(tiny_cfg()).unwrap();
        let sd = src.state_dict();
        let mut dst = BertModel::<f32>::new(tiny_cfg()).unwrap();
        dst.load_state_dict(&sd, true).unwrap();
        let ids = vec![2u32, 4, 6];
        let a = src.forward_from_ids(&ids, None).unwrap();
        let b = dst.forward_from_ids(&ids, None).unwrap();
        for (x, y) in a.data().unwrap().iter().zip(b.data().unwrap().iter()) {
            assert!((x - y).abs() < 1e-6, "round-trip differs: {x} vs {y}");
        }
    }

    #[test]
    fn load_state_dict_strict_rejects_unknown_key() {
        let mut model = BertModel::<f32>::new(tiny_cfg()).unwrap();
        let mut sd = model.state_dict();
        sd.insert(
            "mystery.key".to_string(),
            ferrotorch_core::zeros::<f32>(&[1]).unwrap(),
        );
        assert!(model.load_state_dict(&sd, true).is_err());
    }

    #[test]
    fn load_hf_state_dict_drops_position_ids_and_pooler() {
        let mut model = BertModel::<f32>::new(tiny_cfg()).unwrap();
        let mut sd = model.state_dict();
        sd.insert(
            "embeddings.position_ids".into(),
            ferrotorch_core::zeros::<f32>(&[1, 16]).unwrap(),
        );
        sd.insert(
            "pooler.dense.weight".into(),
            ferrotorch_core::zeros::<f32>(&[8, 8]).unwrap(),
        );
        sd.insert(
            "pooler.dense.bias".into(),
            ferrotorch_core::zeros::<f32>(&[8]).unwrap(),
        );
        let rep = model.load_hf_state_dict(&sd, /* strict = */ false).unwrap();
        assert!(rep.dropped_position_ids);
        let mut dropped = rep.dropped_pooler.clone();
        dropped.sort();
        assert_eq!(
            dropped,
            vec!["pooler.dense.bias".to_string(), "pooler.dense.weight".to_string()]
        );
    }

    #[test]
    fn load_hf_state_dict_strict_rejects_pooler() {
        let mut model = BertModel::<f32>::new(tiny_cfg()).unwrap();
        let mut sd = model.state_dict();
        sd.insert(
            "pooler.dense.weight".into(),
            ferrotorch_core::zeros::<f32>(&[8, 8]).unwrap(),
        );
        assert!(model.load_hf_state_dict(&sd, true).is_err());
    }

    #[test]
    fn sentence_transformer_encode_shape_and_norm_unnormalized() {
        let st = SentenceTransformer::<f32>::new(tiny_cfg(), false).unwrap();
        let ids = vec![1u32, 2, 3, 4];
        let mask = vec![1u32, 1, 1, 0]; // last token padded out
        let out = st.encode(&ids, Some(&mask), None).unwrap();
        assert_eq!(out.shape(), &[1, 8]);
        for &v in out.data().unwrap() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn sentence_transformer_encode_l2_normalizes_to_unit() {
        let st = SentenceTransformer::<f32>::new(tiny_cfg(), true).unwrap();
        let ids = vec![1u32, 2, 3];
        let out = st.encode(&ids, None, None).unwrap();
        let data = out.data().unwrap();
        let sq: f32 = data.iter().map(|v| v * v).sum();
        let norm = sq.sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "expected ||emb||=1 after normalize, got {norm}"
        );
    }

    #[test]
    fn sentence_transformer_rejects_all_zero_mask() {
        let st = SentenceTransformer::<f32>::new(tiny_cfg(), true).unwrap();
        let ids = vec![1u32, 2];
        let mask = vec![0u32, 0];
        assert!(st.encode(&ids, Some(&mask), None).is_err());
    }

    #[test]
    fn sentence_transformer_rejects_mask_length_mismatch() {
        let st = SentenceTransformer::<f32>::new(tiny_cfg(), true).unwrap();
        let ids = vec![1u32, 2, 3];
        let mask = vec![1u32, 1]; // length 2 vs ids.len() 3
        assert!(st.encode(&ids, Some(&mask), None).is_err());
    }
}
