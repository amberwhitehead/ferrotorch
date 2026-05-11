//! BERT input embeddings.
//!
//! Sums three lookups (word + position + token_type) and applies a
//! final LayerNorm. Mirrors `transformers.models.bert.modeling_bert.BertEmbeddings`,
//! minus the inference-time-noop Dropout.

use ferrotorch_core::grad_fns::arithmetic::add;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};
use ferrotorch_nn::module::{Module, StateDict};
use ferrotorch_nn::parameter::Parameter;
use ferrotorch_nn::{Embedding, LayerNorm};

use crate::config::BertConfig;

/// BERT input embedding layer.
///
/// Computes
/// `LayerNorm(word(ids) + position(0..S) + token_type(type_ids))`.
/// At inference dropout is the identity, so the module forwards the
/// LayerNorm output directly. The `embeddings.position_ids` buffer
/// HuggingFace ships in `model.safetensors` is a `[1, max_pos]` constant
/// (just `[0, 1, ..., max_pos-1]`) we do not need to store — the forward
/// pass regenerates the positional indices from the input length.
#[derive(Debug)]
pub struct BertEmbeddings<T: Float> {
    /// Token lookup table — shape `[vocab_size, hidden_size]`.
    pub word_embeddings: Embedding<T>,
    /// Learned positional embedding table — shape
    /// `[max_position_embeddings, hidden_size]`.
    pub position_embeddings: Embedding<T>,
    /// Token-type / segment embedding table — shape
    /// `[type_vocab_size, hidden_size]`.
    pub token_type_embeddings: Embedding<T>,
    /// Post-sum LayerNorm.
    pub layer_norm: LayerNorm<T>,
    hidden_size: usize,
    max_position_embeddings: usize,
    type_vocab_size: usize,
    training: bool,
}

impl<T: Float> BertEmbeddings<T> {
    /// Build randomly-initialized embeddings for the given config.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`FerrotorchError`] if any sub-module
    /// fails to construct.
    pub fn new(cfg: &BertConfig) -> FerrotorchResult<Self> {
        cfg.validate()?;
        Ok(Self {
            word_embeddings: Embedding::new(cfg.vocab_size, cfg.hidden_size, None)?,
            position_embeddings: Embedding::new(
                cfg.max_position_embeddings,
                cfg.hidden_size,
                None,
            )?,
            token_type_embeddings: Embedding::new(cfg.type_vocab_size, cfg.hidden_size, None)?,
            layer_norm: LayerNorm::new(
                vec![cfg.hidden_size],
                cfg.layer_norm_eps,
                /* elementwise_affine = */ true,
            )?,
            hidden_size: cfg.hidden_size,
            max_position_embeddings: cfg.max_position_embeddings,
            type_vocab_size: cfg.type_vocab_size,
            training: false,
        })
    }

    /// Run the embedding stack on a sequence of token ids.
    ///
    /// `input_ids` is the verbatim token-id vector (length `S`).
    /// `token_type_ids` is the per-token segment id; if `None` we
    /// substitute all-zero (the sentence-transformers default — only a
    /// single sentence). The output has shape `[1, S, hidden]`.
    ///
    /// # Errors
    ///
    /// * [`FerrotorchError::InvalidArgument`] if `input_ids` is empty,
    ///   exceeds `max_position_embeddings`, or `token_type_ids` is
    ///   provided with a mismatched length / out-of-range value.
    /// * Propagates downstream embedding-lookup / LayerNorm errors.
    pub fn forward_from_ids(
        &self,
        input_ids: &[u32],
        token_type_ids: Option<&[u32]>,
    ) -> FerrotorchResult<Tensor<T>> {
        if input_ids.is_empty() {
            return Err(FerrotorchError::InvalidArgument {
                message: "BertEmbeddings::forward_from_ids needs at least one token".into(),
            });
        }
        let seq_len = input_ids.len();
        if seq_len > self.max_position_embeddings {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "BertEmbeddings: sequence length {seq_len} exceeds \
                     max_position_embeddings {}",
                    self.max_position_embeddings
                ),
            });
        }

        // -- word_embeddings(input_ids) ------------------------------------
        let word_idx = float_index_tensor::<T>(input_ids)?;
        let word_2d = self.word_embeddings.forward(&word_idx)?; // [S, hidden]

        // -- position_embeddings(0..seq_len) -------------------------------
        let pos_ids: Vec<u32> = (0..seq_len as u32).collect();
        let pos_idx = float_index_tensor::<T>(&pos_ids)?;
        let pos_2d = self.position_embeddings.forward(&pos_idx)?;

        // -- token_type_embeddings(type_ids) -------------------------------
        let type_ids_owned: Vec<u32> = match token_type_ids {
            Some(t) => {
                if t.len() != seq_len {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!(
                            "BertEmbeddings: token_type_ids length {} does not match \
                             input_ids length {seq_len}",
                            t.len()
                        ),
                    });
                }
                for &v in t {
                    if (v as usize) >= self.type_vocab_size {
                        return Err(FerrotorchError::InvalidArgument {
                            message: format!(
                                "BertEmbeddings: token_type_id {v} out of range \
                                 for type_vocab_size {}",
                                self.type_vocab_size
                            ),
                        });
                    }
                }
                t.to_vec()
            }
            None => vec![0u32; seq_len],
        };
        let type_idx = float_index_tensor::<T>(&type_ids_owned)?;
        let type_2d = self.token_type_embeddings.forward(&type_idx)?;

        // -- sum the three 2-D embedding tables. ---------------------------
        let summed = add(&add(&word_2d, &pos_2d)?, &type_2d)?;

        // -- promote to [1, S, hidden] for LayerNorm. ----------------------
        let summed_3d = reshape_to_3d(&summed, 1, seq_len, self.hidden_size)?;

        // -- LayerNorm over the last axis. ---------------------------------
        self.layer_norm.forward(&summed_3d)
    }
}

impl<T: Float> Module<T> for BertEmbeddings<T> {
    /// Treats `input` as already-summed embeddings `[1, S, hidden]` and
    /// only applies the LayerNorm. Sentence-embedding callers should use
    /// [`Self::forward_from_ids`] instead.
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        self.layer_norm.forward(input)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.word_embeddings.parameters());
        out.extend(self.position_embeddings.parameters());
        out.extend(self.token_type_embeddings.parameters());
        out.extend(self.layer_norm.parameters());
        out
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.word_embeddings.parameters_mut());
        out.extend(self.position_embeddings.parameters_mut());
        out.extend(self.token_type_embeddings.parameters_mut());
        out.extend(self.layer_norm.parameters_mut());
        out
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (n, p) in self.word_embeddings.named_parameters() {
            out.push((format!("word_embeddings.{n}"), p));
        }
        for (n, p) in self.position_embeddings.named_parameters() {
            out.push((format!("position_embeddings.{n}"), p));
        }
        for (n, p) in self.token_type_embeddings.named_parameters() {
            out.push((format!("token_type_embeddings.{n}"), p));
        }
        for (n, p) in self.layer_norm.named_parameters() {
            out.push((format!("LayerNorm.{n}"), p));
        }
        out
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
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
            let prefixes = [
                "word_embeddings",
                "position_embeddings",
                "token_type_embeddings",
                "LayerNorm",
            ];
            for key in state.keys() {
                if !prefixes.iter().any(|p| key.starts_with(&format!("{p}."))) {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!(
                            "unexpected key in BertEmbeddings state_dict: \"{key}\""
                        ),
                    });
                }
            }
        }

        self.word_embeddings
            .load_state_dict(&extract("word_embeddings"), strict)?;
        self.position_embeddings
            .load_state_dict(&extract("position_embeddings"), strict)?;
        self.token_type_embeddings
            .load_state_dict(&extract("token_type_embeddings"), strict)?;
        self.layer_norm
            .load_state_dict(&extract("LayerNorm"), strict)?;
        Ok(())
    }
}

/// Build a 1-D float-encoded index tensor from u32 token ids.
fn float_index_tensor<T: Float>(ids: &[u32]) -> FerrotorchResult<Tensor<T>> {
    let data: Vec<T> = ids
        .iter()
        .map(|&i| ferrotorch_core::numeric_cast::cast::<u32, T>(i))
        .collect::<FerrotorchResult<Vec<T>>>()?;
    let n = data.len();
    Tensor::from_storage(TensorStorage::cpu(data), vec![n], false)
}

/// Reshape `[a*b, c]` to `[a, b, c]` (we own the data; no view tricks).
fn reshape_to_3d<T: Float>(
    t: &Tensor<T>,
    a: usize,
    b: usize,
    c: usize,
) -> FerrotorchResult<Tensor<T>> {
    let data = t.data_vec()?;
    Tensor::from_storage(TensorStorage::cpu(data), vec![a, b, c], t.requires_grad())
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
            num_hidden_layers: 1,
            num_attention_heads: 2,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
        }
    }

    #[test]
    fn forward_from_ids_produces_correct_shape() {
        let emb = BertEmbeddings::<f32>::new(&tiny_cfg()).unwrap();
        let ids = [1u32, 5, 7, 9];
        let out = emb.forward_from_ids(&ids, None).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        for &v in out.data().unwrap() {
            assert!(v.is_finite(), "embedding non-finite: {v}");
        }
    }

    #[test]
    fn forward_from_ids_rejects_too_long_sequence() {
        let emb = BertEmbeddings::<f32>::new(&tiny_cfg()).unwrap();
        let ids: Vec<u32> = (0..17).collect();
        assert!(emb.forward_from_ids(&ids, None).is_err());
    }

    #[test]
    fn forward_from_ids_rejects_bad_token_type_length() {
        let emb = BertEmbeddings::<f32>::new(&tiny_cfg()).unwrap();
        let ids = [1u32, 2];
        let bad_types = [0u32]; // length 1 vs ids.len() 2
        assert!(emb.forward_from_ids(&ids, Some(&bad_types)).is_err());
    }

    #[test]
    fn named_parameters_use_hf_layout() {
        let emb = BertEmbeddings::<f32>::new(&tiny_cfg()).unwrap();
        let names: Vec<String> = emb.named_parameters().into_iter().map(|(n, _)| n).collect();
        assert!(names.contains(&"word_embeddings.weight".to_string()));
        assert!(names.contains(&"position_embeddings.weight".to_string()));
        assert!(names.contains(&"token_type_embeddings.weight".to_string()));
        assert!(names.contains(&"LayerNorm.weight".to_string()));
        assert!(names.contains(&"LayerNorm.bias".to_string()));
    }
}
