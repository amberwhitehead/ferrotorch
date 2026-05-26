//! Single BERT encoder layer.
//!
//! ```text
//! attn_out = LayerNorm(input + dense(self_attn(input)))   // POST-NORM
//! ffn_out  = GELU(Linear_inter(attn_out))                  // intermediate
//! layer_out= LayerNorm(attn_out + Linear_out(ffn_out))    // POST-NORM
//! ```
//!
//! ## REQ status (per `.design/<area>/<file>.md`)
//!
//! | REQ | Status | Evidence |
//! | --- | --- | --- |
//! | REQ-1 | SHIPPED | impl: `pub struct BertIntermediate<T: Float>` + its `Module<T>` impl in `layer.rs`; non-test consumer: field `intermediate` of `pub struct BertLayer` in `layer.rs`; `BertLayer::Module::forward` in `layer.rs` invokes `self.intermediate.forward(&attn_out)`. |
//! | REQ-2 | SHIPPED | impl: `BertOutput::forward_residual` in `layer.rs`; non-test consumer: `BertLayer::Module::forward` in `layer.rs` invokes `self.output.forward_residual(&inter, &attn_out)`. |
//! | REQ-3 | SHIPPED | impl: `pub struct BertLayer<T: Float>` + its `Module<T>` impl in `layer.rs`; non-test consumer: element of `pub layer: Vec<BertLayer<T>>` at `ferrotorch-bert/src/model.rs:22`; `BertEncoder::new` at `ferrotorch-bert/src/model.rs:35` constructs them; `BertEncoder::Module::forward` at `ferrotorch-bert/src/model.rs:46` iterates them. |
//! | REQ-4 | SHIPPED | impl: `BertOutput::Module::forward` (dense-only) in `layer.rs`; non-test consumer: re-export at `ferrotorch-bert/src/lib.rs:89`. |
//! | REQ-5 | SHIPPED | impl: `named_parameters` / `load_state_dict` for `BertLayer` in `layer.rs`; non-test consumer: `BertEncoder::load_state_dict` at `ferrotorch-bert/src/model.rs:105` recurses through `layer.{i}.*`. |
//! | REQ-6 | SHIPPED | impl: `BertIntermediate::new` constructs `GELU::new()` (exact-erf) in `layer.rs`; non-test consumer: the `gelu` activation choice is enforced upstream by `HfBertConfig::validate` in `ferrotorch-bert/src/config.rs` which rejects any other activation name on load. |

use ferrotorch_core::grad_fns::arithmetic::add;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor};
use ferrotorch_nn::module::{Module, StateDict};
use ferrotorch_nn::parameter::Parameter;
use ferrotorch_nn::{GELU, LayerNorm, Linear};

use crate::attention::BertAttention;
use crate::config::BertConfig;

/// HF `BertIntermediate` — `Linear(hidden -> intermediate) + GELU`.
#[derive(Debug)]
pub struct BertIntermediate<T: Float> {
    /// Expansion projection — `[hidden -> intermediate]`, with bias.
    pub dense: Linear<T>,
    activation: GELU,
    training: bool,
}

impl<T: Float> BertIntermediate<T> {
    /// Build randomly-initialized intermediate FFN.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`FerrotorchError`] on bad config dims.
    pub fn new(cfg: &BertConfig) -> FerrotorchResult<Self> {
        cfg.validate()?;
        Ok(Self {
            dense: Linear::new(cfg.hidden_size, cfg.intermediate_size, true)?,
            // HF BertIntermediate uses `gelu` (i.e. exact erf-based by default;
            // the HF kernel matches `torch.nn.functional.gelu(approximate="none")`).
            activation: GELU::new(),
            training: false,
        })
    }
}

impl<T: Float> Module<T> for BertIntermediate<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let projected = self.dense.forward(input)?;
        self.activation.forward(&projected)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        self.dense.parameters()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        self.dense.parameters_mut()
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (n, p) in self.dense.named_parameters() {
            out.push((format!("dense.{n}"), p));
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
            for key in state.keys() {
                if !key.starts_with("dense.") {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!(
                            "unexpected key in BertIntermediate state_dict: \"{key}\""
                        ),
                    });
                }
            }
        }
        self.dense.load_state_dict(&extract("dense"), strict)
    }
}

/// HF `BertOutput` — `Linear(intermediate -> hidden)` + LayerNorm
/// over the residual sum.
#[derive(Debug)]
pub struct BertOutput<T: Float> {
    /// Reduction projection — `[intermediate -> hidden]`, with bias.
    pub dense: Linear<T>,
    /// LayerNorm applied to the (attn_out + dense(intermediate)) sum.
    pub layer_norm: LayerNorm<T>,
    training: bool,
}

impl<T: Float> BertOutput<T> {
    /// Build randomly-initialized output dense + LayerNorm.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`FerrotorchError`] on bad config dims.
    pub fn new(cfg: &BertConfig) -> FerrotorchResult<Self> {
        cfg.validate()?;
        Ok(Self {
            dense: Linear::new(cfg.intermediate_size, cfg.hidden_size, true)?,
            layer_norm: LayerNorm::new(vec![cfg.hidden_size], cfg.layer_norm_eps, true)?,
            training: false,
        })
    }

    /// Apply the dense projection then the post-norm residual.
    ///
    /// `intermediate` is the GELU-activated `BertIntermediate` output
    /// `[1, S, intermediate]`. `attn_out` is the post-attention,
    /// post-norm tensor `[1, S, hidden]` that feeds the residual.
    ///
    /// # Errors
    ///
    /// Propagates any downstream Linear / LayerNorm error.
    pub fn forward_residual(
        &self,
        intermediate: &Tensor<T>,
        attn_out: &Tensor<T>,
    ) -> FerrotorchResult<Tensor<T>> {
        let projected = self.dense.forward(intermediate)?;
        let summed = add(attn_out, &projected)?;
        self.layer_norm.forward(&summed)
    }
}

impl<T: Float> Module<T> for BertOutput<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // The residual partner is unavailable here. Module::forward only
        // applies the dense projection (used by tooling that walks
        // sub-module forwards individually). The full BertOutput
        // sub-block is invoked through [`Self::forward_residual`].
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
                        message: format!("unexpected key in BertOutput state_dict: \"{key}\""),
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

/// HF `BertLayer` — one full encoder block.
#[derive(Debug)]
pub struct BertLayer<T: Float> {
    /// Self-attention + post-norm output sub-block.
    pub attention: BertAttention<T>,
    /// FFN expansion + GELU activation.
    pub intermediate: BertIntermediate<T>,
    /// FFN reduction + post-norm residual.
    pub output: BertOutput<T>,
    training: bool,
}

impl<T: Float> BertLayer<T> {
    /// Build a randomly-initialized encoder layer.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`FerrotorchError`] on bad config dims.
    pub fn new(cfg: &BertConfig) -> FerrotorchResult<Self> {
        Ok(Self {
            attention: BertAttention::new(cfg)?,
            intermediate: BertIntermediate::new(cfg)?,
            output: BertOutput::new(cfg)?,
            training: false,
        })
    }
}

impl<T: Float> Module<T> for BertLayer<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let attn_out = self.attention.forward(input)?;
        let inter = self.intermediate.forward(&attn_out)?;
        self.output.forward_residual(&inter, &attn_out)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.attention.parameters());
        out.extend(self.intermediate.parameters());
        out.extend(self.output.parameters());
        out
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.attention.parameters_mut());
        out.extend(self.intermediate.parameters_mut());
        out.extend(self.output.parameters_mut());
        out
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (n, p) in self.attention.named_parameters() {
            out.push((format!("attention.{n}"), p));
        }
        for (n, p) in self.intermediate.named_parameters() {
            out.push((format!("intermediate.{n}"), p));
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
            let prefixes = ["attention", "intermediate", "output"];
            for key in state.keys() {
                if !prefixes.iter().any(|p| key.starts_with(&format!("{p}."))) {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!("unexpected key in BertLayer state_dict: \"{key}\""),
                    });
                }
            }
        }
        self.attention
            .load_state_dict(&extract("attention"), strict)?;
        self.intermediate
            .load_state_dict(&extract("intermediate"), strict)?;
        self.output.load_state_dict(&extract("output"), strict)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::TensorStorage;

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
    fn layer_forward_shape() {
        let layer = BertLayer::<f32>::new(&tiny_cfg()).unwrap();
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![0.1f32; 5 * 8]),
            vec![1, 5, 8],
            false,
        )
        .unwrap();
        let out = layer.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 5, 8]);
        for &v in out.data().unwrap() {
            assert!(v.is_finite(), "layer output non-finite: {v}");
        }
    }

    #[test]
    fn layer_named_parameters_match_hf_layout() {
        let layer = BertLayer::<f32>::new(&tiny_cfg()).unwrap();
        let names: Vec<String> = layer
            .named_parameters()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        // HF: encoder.layer.{i}.attention.self.{q,k,v}.{weight,bias}
        //     encoder.layer.{i}.attention.output.dense.{weight,bias}
        //     encoder.layer.{i}.attention.output.LayerNorm.{weight,bias}
        //     encoder.layer.{i}.intermediate.dense.{weight,bias}
        //     encoder.layer.{i}.output.dense.{weight,bias}
        //     encoder.layer.{i}.output.LayerNorm.{weight,bias}
        for k in [
            "attention.self.query.weight",
            "attention.self.query.bias",
            "attention.self.key.weight",
            "attention.self.value.weight",
            "attention.output.dense.weight",
            "attention.output.LayerNorm.weight",
            "intermediate.dense.weight",
            "intermediate.dense.bias",
            "output.dense.weight",
            "output.LayerNorm.weight",
        ] {
            assert!(
                names.iter().any(|n| n == k),
                "missing parameter key {k:?} in {names:?}"
            );
        }
    }
}
