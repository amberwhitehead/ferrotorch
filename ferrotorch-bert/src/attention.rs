//! BERT self-attention block.
//!
//! Bidirectional (non-causal) multi-head attention. Matches HuggingFace
//! `BertSelfAttention` + `BertSelfOutput`:
//!
//! ```text
//! q, k, v = Linear(input)           // 3 × [hidden -> hidden], with bias
//! ctx     = softmax(Q K^T / √d) V   // 12 heads, no causal mask
//! out     = Linear(ctx)             // [hidden -> hidden], with bias
//! return    LayerNorm(input + out)  // POST-NORM residual
//! ```

use ferrotorch_core::grad_fns::arithmetic::add;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};
use ferrotorch_nn::module::{Module, StateDict};
use ferrotorch_nn::parameter::Parameter;
use ferrotorch_nn::{
    LayerNorm, Linear, reshape_to_heads, standard_attention, transpose_heads_to_2d,
};

use crate::config::BertConfig;

/// HF `BertSelfAttention`: q / k / v projections + scaled dot-product
/// attention. Returns the context tensor `[1, S, hidden]` (no residual
/// add — that's done in [`BertAttention`] / `BertSelfOutput`).
#[derive(Debug)]
pub struct BertSelfAttention<T: Float> {
    /// Query projection — `[hidden -> hidden]`, with bias.
    pub query: Linear<T>,
    /// Key projection — `[hidden -> hidden]`, with bias.
    pub key: Linear<T>,
    /// Value projection — `[hidden -> hidden]`, with bias.
    pub value: Linear<T>,
    num_heads: usize,
    head_dim: usize,
    hidden: usize,
    training: bool,
}

impl<T: Float> BertSelfAttention<T> {
    /// Build randomly-initialized self-attention projections.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`FerrotorchError`] on bad config dims.
    pub fn new(cfg: &BertConfig) -> FerrotorchResult<Self> {
        cfg.validate()?;
        Ok(Self {
            query: Linear::new(cfg.hidden_size, cfg.hidden_size, true)?,
            key: Linear::new(cfg.hidden_size, cfg.hidden_size, true)?,
            value: Linear::new(cfg.hidden_size, cfg.hidden_size, true)?,
            num_heads: cfg.num_attention_heads,
            head_dim: cfg.head_dim(),
            hidden: cfg.hidden_size,
            training: false,
        })
    }
}

impl<T: Float> Module<T> for BertSelfAttention<T> {
    /// Forward pass.
    ///
    /// # Input
    ///
    /// `[1, seq_len, hidden]`.
    ///
    /// # Output
    ///
    /// `[1, seq_len, hidden]` — the attention context, ready to be fed
    /// to [`BertSelfOutput`] for the post-norm residual.
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let shape = input.shape();
        if shape.len() != 3 || shape[0] != 1 || shape[2] != self.hidden {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "BertSelfAttention expects [1, S, {}], got {:?}",
                    self.hidden, shape,
                ),
            });
        }
        let seq_len = shape[1];

        // Project QKV. Linear handles any rank; output is [1, S, hidden].
        let q = self.query.forward(input)?;
        let k = self.key.forward(input)?;
        let v = self.value.forward(input)?;

        // Drop the batch=1 leading dim for reshape_to_heads (2-D in).
        let q2 = reshape_2d(&q, seq_len, self.hidden)?;
        let k2 = reshape_2d(&k, seq_len, self.hidden)?;
        let v2 = reshape_2d(&v, seq_len, self.hidden)?;

        // Split into heads. [S, H*d] → [H, S, d].
        let q_h = reshape_to_heads(&q2, self.num_heads, seq_len, self.head_dim)?;
        let k_h = reshape_to_heads(&k2, self.num_heads, seq_len, self.head_dim)?;
        let v_h = reshape_to_heads(&v2, self.num_heads, seq_len, self.head_dim)?;

        // Scaled dot-product attention, non-causal (full bidirectional).
        let ctx = standard_attention(&q_h, &k_h, &v_h, /* causal = */ false)?;

        // Merge heads back to [S, H*d] then promote to [1, S, hidden].
        let ctx2 = transpose_heads_to_2d(&ctx, self.num_heads, seq_len, self.head_dim)?;
        reshape_3d(&ctx2, 1, seq_len, self.hidden)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.query.parameters());
        out.extend(self.key.parameters());
        out.extend(self.value.parameters());
        out
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.query.parameters_mut());
        out.extend(self.key.parameters_mut());
        out.extend(self.value.parameters_mut());
        out
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (n, p) in self.query.named_parameters() {
            out.push((format!("query.{n}"), p));
        }
        for (n, p) in self.key.named_parameters() {
            out.push((format!("key.{n}"), p));
        }
        for (n, p) in self.value.named_parameters() {
            out.push((format!("value.{n}"), p));
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
            let prefixes = ["query", "key", "value"];
            for key in state.keys() {
                if !prefixes.iter().any(|p| key.starts_with(&format!("{p}."))) {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!(
                            "unexpected key in BertSelfAttention state_dict: \"{key}\""
                        ),
                    });
                }
            }
        }
        self.query.load_state_dict(&extract("query"), strict)?;
        self.key.load_state_dict(&extract("key"), strict)?;
        self.value.load_state_dict(&extract("value"), strict)?;
        Ok(())
    }
}

/// HF `BertSelfOutput`: post-attention output projection + post-norm
/// residual add. Forward is `LayerNorm(input + Linear(ctx))`.
///
/// Distinct from [`BertSelfAttention`] in HF's tree under
/// `attention.output.*` (vs `attention.self.*`).
#[derive(Debug)]
pub struct BertSelfOutput<T: Float> {
    /// Output projection — `[hidden -> hidden]`, with bias.
    pub dense: Linear<T>,
    /// LayerNorm applied to the (input + dense(context)) sum.
    pub layer_norm: LayerNorm<T>,
    training: bool,
}

impl<T: Float> BertSelfOutput<T> {
    /// Build randomly-initialized output dense + LayerNorm.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`FerrotorchError`] on bad config dims.
    pub fn new(cfg: &BertConfig) -> FerrotorchResult<Self> {
        cfg.validate()?;
        Ok(Self {
            dense: Linear::new(cfg.hidden_size, cfg.hidden_size, true)?,
            layer_norm: LayerNorm::new(vec![cfg.hidden_size], cfg.layer_norm_eps, true)?,
            training: false,
        })
    }

    /// Run the residual + LayerNorm pipeline. `attn` is the context
    /// produced by [`BertSelfAttention::forward`]; `input` is the
    /// pre-attention tensor (the residual input).
    ///
    /// # Errors
    ///
    /// Propagates any downstream Linear / LayerNorm error.
    pub fn forward_residual(
        &self,
        attn: &Tensor<T>,
        input: &Tensor<T>,
    ) -> FerrotorchResult<Tensor<T>> {
        let projected = self.dense.forward(attn)?;
        let summed = add(input, &projected)?;
        self.layer_norm.forward(&summed)
    }
}

impl<T: Float> Module<T> for BertSelfOutput<T> {
    /// Convenience: when called via `Module::forward` we ONLY run the
    /// dense layer. The post-norm residual needs the un-attended input;
    /// use [`Self::forward_residual`] for the full sub-block.
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        self.dense.forward(input)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.dense.parameters());
        out.extend(self.layer_norm.parameters());
        out
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.dense.parameters_mut());
        out.extend(self.layer_norm.parameters_mut());
        out
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (n, p) in self.dense.named_parameters() {
            out.push((format!("dense.{n}"), p));
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
            let prefixes = ["dense", "LayerNorm"];
            for key in state.keys() {
                if !prefixes.iter().any(|p| key.starts_with(&format!("{p}."))) {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!("unexpected key in BertSelfOutput state_dict: \"{key}\""),
                    });
                }
            }
        }
        self.dense.load_state_dict(&extract("dense"), strict)?;
        self.layer_norm
            .load_state_dict(&extract("LayerNorm"), strict)?;
        Ok(())
    }
}

/// HF `BertAttention` = `BertSelfAttention` + `BertSelfOutput` combined.
///
/// Forward: `LayerNorm(input + dense(self_attn(input)))`.
#[derive(Debug)]
pub struct BertAttention<T: Float> {
    /// `attention.self.*` in HF — bare Q / K / V + SDPA.
    pub self_attn: BertSelfAttention<T>,
    /// `attention.output.*` in HF — output projection + post-norm.
    pub output: BertSelfOutput<T>,
    training: bool,
}

impl<T: Float> BertAttention<T> {
    /// Build a randomly-initialized attention sub-block.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`FerrotorchError`] on bad config dims.
    pub fn new(cfg: &BertConfig) -> FerrotorchResult<Self> {
        Ok(Self {
            self_attn: BertSelfAttention::new(cfg)?,
            output: BertSelfOutput::new(cfg)?,
            training: false,
        })
    }
}

impl<T: Float> Module<T> for BertAttention<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let ctx = self.self_attn.forward(input)?;
        self.output.forward_residual(&ctx, input)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.self_attn.parameters());
        out.extend(self.output.parameters());
        out
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.self_attn.parameters_mut());
        out.extend(self.output.parameters_mut());
        out
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (n, p) in self.self_attn.named_parameters() {
            out.push((format!("self.{n}"), p));
        }
        for (n, p) in self.output.named_parameters() {
            out.push((format!("output.{n}"), p));
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
            let prefixes = ["self", "output"];
            for key in state.keys() {
                if !prefixes.iter().any(|p| key.starts_with(&format!("{p}."))) {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!("unexpected key in BertAttention state_dict: \"{key}\""),
                    });
                }
            }
        }
        self.self_attn.load_state_dict(&extract("self"), strict)?;
        self.output.load_state_dict(&extract("output"), strict)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// 2-D / 3-D reshape helpers (own the data; no view trick).
// ---------------------------------------------------------------------------

fn reshape_2d<T: Float>(t: &Tensor<T>, a: usize, b: usize) -> FerrotorchResult<Tensor<T>> {
    let data = t.data_vec()?;
    Tensor::from_storage(TensorStorage::cpu(data), vec![a, b], t.requires_grad())
}

fn reshape_3d<T: Float>(
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
    fn self_attention_shape() {
        let attn = BertSelfAttention::<f32>::new(&tiny_cfg()).unwrap();
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![0.5f32; 4 * 8]),
            vec![1, 4, 8],
            false,
        )
        .unwrap();
        let out = attn.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn full_attention_block_shape() {
        let attn = BertAttention::<f32>::new(&tiny_cfg()).unwrap();
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![0.25f32; 3 * 8]),
            vec![1, 3, 8],
            false,
        )
        .unwrap();
        let out = attn.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 3, 8]);
    }

    #[test]
    fn named_parameters_match_hf_layout() {
        let attn = BertAttention::<f32>::new(&tiny_cfg()).unwrap();
        let names: Vec<String> = attn
            .named_parameters()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        assert!(names.contains(&"self.query.weight".to_string()));
        assert!(names.contains(&"self.query.bias".to_string()));
        assert!(names.contains(&"self.key.weight".to_string()));
        assert!(names.contains(&"self.value.weight".to_string()));
        assert!(names.contains(&"output.dense.weight".to_string()));
        assert!(names.contains(&"output.dense.bias".to_string()));
        assert!(names.contains(&"output.LayerNorm.weight".to_string()));
        assert!(names.contains(&"output.LayerNorm.bias".to_string()));
    }
}
