//! Typed BERT configuration.
//!
//! [`BertConfig`] is the flat, self-contained struct the model layer
//! works against. Construct it directly for synthetic tests, or via
//! [`BertConfig::from_hf`] from a HuggingFace BERT `config.json`.
//!
//! ## REQ status (per `.design/<area>/<file>.md`)
//!
//! | REQ | Status | Evidence |
//! | --- | --- | --- |
//! | REQ-1 | SHIPPED | impl: `pub struct BertConfig` in `config.rs`; non-test consumer: `pub use` at `ferrotorch-bert/src/lib.rs:87` plus `BertEmbeddings::new` at `ferrotorch-bert/src/embeddings.rs:49`. |
//! | REQ-2 | SHIPPED | impl: `pub struct HfBertConfig` in `config.rs`; non-test consumer: `pub use` at `ferrotorch-bert/src/lib.rs:87`. |
//! | REQ-3 | SHIPPED | impl: `BertConfig::validate` in `config.rs`; non-test consumer: invoked by `BertEmbeddings::new` at `ferrotorch-bert/src/embeddings.rs:49` and `BertSelfAttention::new` at `ferrotorch-bert/src/attention.rs:46`. |
//! | REQ-4 | SHIPPED | impl: `HfBertConfig::validate` in `config.rs`; non-test consumer: invoked by `BertConfig::from_hf` in `config.rs`. |
//! | REQ-5 | SHIPPED | impl: `BertConfig::from_hf` in `config.rs`; non-test consumer: `pub use` at `ferrotorch-bert/src/lib.rs:87` (called by Hub-load helpers). |
//! | REQ-6 | SHIPPED | impl: `BertConfig::all_minilm_l6_v2` in `config.rs`; non-test consumer: `pub use` at `ferrotorch-bert/src/lib.rs:87`. |
//! | REQ-7 | SHIPPED | impl: `HfBertConfig::from_file` / `from_json_str` in `config.rs`; non-test consumer: `pub use` at `ferrotorch-bert/src/lib.rs:87`. |
//! | REQ-8 | SHIPPED | impl: `BertConfig::head_dim` in `config.rs`; non-test consumer: `BertSelfAttention::new` at `ferrotorch-bert/src/attention.rs:53` reads `cfg.head_dim()`. |

use std::path::Path;

use ferrotorch_core::{FerrotorchError, FerrotorchResult};
use serde::Deserialize;

/// BERT model hyperparameters.
///
/// Mirrors the union of fields a HuggingFace `BertConfig` exposes that
/// the encoder-only forward pass consumes. Marked `#[non_exhaustive]`
/// so the schema can grow (e.g. positional-embedding variants beyond
/// `absolute`) without breaking external callers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BertConfig {
    /// Vocabulary size (word-embedding rows).
    pub vocab_size: usize,
    /// Token-type vocabulary size (typically `2` — sentence A / B).
    pub type_vocab_size: usize,
    /// Maximum positional context the model was trained with.
    pub max_position_embeddings: usize,
    /// Embedding / hidden dimension (`d_model`).
    pub hidden_size: usize,
    /// FFN inner dimension (also known as `d_ff`).
    pub intermediate_size: usize,
    /// Number of transformer encoder layers.
    pub num_hidden_layers: usize,
    /// Number of attention heads per layer.
    pub num_attention_heads: usize,
    /// LayerNorm epsilon. HF BERT default: `1e-12`.
    pub layer_norm_eps: f64,
    /// Pad-token id (used to skip rows from the mean pool when no
    /// attention mask is supplied — typically `0`).
    pub pad_token_id: usize,
}

impl BertConfig {
    /// The published `sentence-transformers/all-MiniLM-L6-v2` config.
    pub fn all_minilm_l6_v2() -> Self {
        Self {
            vocab_size: 30_522,
            type_vocab_size: 2,
            max_position_embeddings: 512,
            hidden_size: 384,
            intermediate_size: 1_536,
            num_hidden_layers: 6,
            num_attention_heads: 12,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
        }
    }

    /// Build a [`BertConfig`] from a parsed HuggingFace `bert/config.json`.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] if validation fails
    /// or the activation function is not `gelu` (the only activation
    /// the ferrotorch-bert forward pass implements).
    pub fn from_hf(hf: &HfBertConfig) -> FerrotorchResult<Self> {
        hf.validate()?;
        Ok(Self {
            vocab_size: hf.vocab_size,
            type_vocab_size: hf.type_vocab_size,
            max_position_embeddings: hf.max_position_embeddings,
            hidden_size: hf.hidden_size,
            intermediate_size: hf.intermediate_size,
            num_hidden_layers: hf.num_hidden_layers,
            num_attention_heads: hf.num_attention_heads,
            layer_norm_eps: hf.layer_norm_eps,
            pad_token_id: hf.pad_token_id,
        })
    }

    /// Per-head feature dimension: `hidden_size / num_attention_heads`.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Enforce invariants the model layer relies on. Called by every
    /// `BertConfig`-consuming constructor.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] when any size is
    /// zero or when `hidden_size % num_attention_heads != 0`.
    pub fn validate(&self) -> FerrotorchResult<()> {
        if self.vocab_size == 0
            || self.type_vocab_size == 0
            || self.max_position_embeddings == 0
            || self.hidden_size == 0
            || self.intermediate_size == 0
            || self.num_hidden_layers == 0
            || self.num_attention_heads == 0
        {
            return Err(FerrotorchError::InvalidArgument {
                message: "BertConfig: vocab_size / type_vocab_size / hidden_size / \
                          intermediate_size / num_hidden_layers / num_attention_heads / \
                          max_position_embeddings must all be positive"
                    .into(),
            });
        }
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "BertConfig: hidden_size ({}) must be divisible by num_attention_heads ({})",
                    self.hidden_size, self.num_attention_heads
                ),
            });
        }
        Ok(())
    }
}

/// Parsed contents of a HuggingFace BERT `config.json`.
///
/// `#[non_exhaustive]` so the schema can grow without breaking callers.
#[derive(Debug, Clone, Deserialize)]
#[non_exhaustive]
pub struct HfBertConfig {
    /// Embedding / hidden dimension (`d_model`).
    pub hidden_size: usize,
    /// Number of transformer encoder layers.
    pub num_hidden_layers: usize,
    /// Number of attention heads per layer.
    pub num_attention_heads: usize,
    /// FFN inner dimension.
    pub intermediate_size: usize,
    /// LayerNorm epsilon. HF default for BERT: `1e-12`.
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f64,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Token-type vocabulary size (sentence A / B).
    #[serde(default = "default_type_vocab_size")]
    pub type_vocab_size: usize,
    /// Maximum positional context length.
    pub max_position_embeddings: usize,
    /// Activation inside the FFN. BERT uses `"gelu"`.
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    /// Pad-token id. Most BERT checkpoints set this to `0`.
    #[serde(default)]
    pub pad_token_id: usize,
    /// Architectures declared in the config (e.g. `["BertModel"]`).
    #[serde(default)]
    pub architectures: Vec<String>,
    /// Model type tag (e.g. `"bert"`).
    #[serde(default)]
    pub model_type: Option<String>,
    /// Positional-embedding variant. Only `"absolute"` is supported.
    #[serde(default = "default_position_embedding_type")]
    pub position_embedding_type: String,
}

fn default_layer_norm_eps() -> f64 {
    1e-12
}

fn default_type_vocab_size() -> usize {
    2
}

fn default_hidden_act() -> String {
    "gelu".to_string()
}

fn default_position_embedding_type() -> String {
    "absolute".to_string()
}

impl HfBertConfig {
    /// Parse from a JSON string.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] if the JSON is not
    /// a valid HF BERT config (missing required fields, malformed).
    pub fn from_json_str(s: &str) -> FerrotorchResult<Self> {
        serde_json::from_str(s).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("failed to parse BERT config JSON: {e}"),
        })
    }

    /// Parse from a `config.json` file on disk.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] if the file cannot
    /// be read or its contents are not a valid HF BERT config.
    pub fn from_file(path: impl AsRef<Path>) -> FerrotorchResult<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("failed to read config file {}: {e}", path.display()),
        })?;
        let s = std::str::from_utf8(&bytes).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("config file {} is not valid UTF-8: {e}", path.display()),
        })?;
        Self::from_json_str(s)
    }

    /// Validate invariants downstream code relies on.
    ///
    /// Checks:
    /// - all counts are positive
    /// - `hidden_size % num_attention_heads == 0`
    /// - `hidden_act == "gelu"` (only activation implemented)
    /// - `position_embedding_type == "absolute"` (only variant implemented)
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] on any failed check.
    pub fn validate(&self) -> FerrotorchResult<()> {
        if self.hidden_size == 0
            || self.num_attention_heads == 0
            || self.num_hidden_layers == 0
            || self.intermediate_size == 0
            || self.vocab_size == 0
            || self.type_vocab_size == 0
            || self.max_position_embeddings == 0
        {
            return Err(FerrotorchError::InvalidArgument {
                message: "HfBertConfig: hidden_size / num_attention_heads / \
                          num_hidden_layers / intermediate_size / vocab_size / \
                          type_vocab_size / max_position_embeddings must all be positive"
                    .into(),
            });
        }
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "HfBertConfig: hidden_size ({}) must be divisible by \
                     num_attention_heads ({})",
                    self.hidden_size, self.num_attention_heads
                ),
            });
        }
        if self.hidden_act != "gelu" {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "HfBertConfig: unsupported hidden_act {:?} \
                     (ferrotorch-bert implements only \"gelu\")",
                    self.hidden_act
                ),
            });
        }
        if self.position_embedding_type != "absolute" {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "HfBertConfig: unsupported position_embedding_type {:?} \
                     (ferrotorch-bert implements only \"absolute\")",
                    self.position_embedding_type
                ),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINILM_L6_CONFIG: &str = r#"{
        "architectures": ["BertModel"],
        "attention_probs_dropout_prob": 0.1,
        "gradient_checkpointing": false,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 384,
        "initializer_range": 0.02,
        "intermediate_size": 1536,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "model_type": "bert",
        "num_attention_heads": 12,
        "num_hidden_layers": 6,
        "pad_token_id": 0,
        "position_embedding_type": "absolute",
        "transformers_version": "4.8.2",
        "type_vocab_size": 2,
        "use_cache": true,
        "vocab_size": 30522
    }"#;

    #[test]
    fn parses_minilm_l6_config() {
        let cfg = HfBertConfig::from_json_str(MINILM_L6_CONFIG).unwrap();
        assert_eq!(cfg.hidden_size, 384);
        assert_eq!(cfg.num_hidden_layers, 6);
        assert_eq!(cfg.num_attention_heads, 12);
        assert_eq!(cfg.intermediate_size, 1536);
        assert!((cfg.layer_norm_eps - 1e-12).abs() < 1e-15);
        assert_eq!(cfg.vocab_size, 30_522);
        assert_eq!(cfg.type_vocab_size, 2);
        assert_eq!(cfg.max_position_embeddings, 512);
        assert_eq!(cfg.hidden_act, "gelu");
        assert_eq!(cfg.position_embedding_type, "absolute");
        assert_eq!(cfg.pad_token_id, 0);
    }

    #[test]
    fn from_hf_round_trips_minilm_l6() {
        let hf = HfBertConfig::from_json_str(MINILM_L6_CONFIG).unwrap();
        let cfg = BertConfig::from_hf(&hf).unwrap();
        assert_eq!(cfg, BertConfig::all_minilm_l6_v2());
    }

    #[test]
    fn all_minilm_l6_v2_is_valid() {
        let cfg = BertConfig::all_minilm_l6_v2();
        cfg.validate().unwrap();
        assert_eq!(cfg.head_dim(), 32); // 384 / 12
    }

    #[test]
    fn validate_rejects_zero_fields() {
        let mut cfg = BertConfig::all_minilm_l6_v2();
        cfg.hidden_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_non_divisible_heads() {
        let mut cfg = BertConfig::all_minilm_l6_v2();
        cfg.num_attention_heads = 11; // 384 % 11 != 0
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_unsupported_activation() {
        let json = r#"{
            "hidden_size": 384, "num_hidden_layers": 6,
            "num_attention_heads": 12, "intermediate_size": 1536,
            "max_position_embeddings": 512, "vocab_size": 30522,
            "hidden_act": "relu"
        }"#;
        let hf = HfBertConfig::from_json_str(json).unwrap();
        assert!(hf.validate().is_err());
    }

    #[test]
    fn validate_rejects_unsupported_position_embedding_type() {
        let json = r#"{
            "hidden_size": 384, "num_hidden_layers": 6,
            "num_attention_heads": 12, "intermediate_size": 1536,
            "max_position_embeddings": 512, "vocab_size": 30522,
            "position_embedding_type": "relative_key"
        }"#;
        let hf = HfBertConfig::from_json_str(json).unwrap();
        assert!(hf.validate().is_err());
    }

    #[test]
    fn unknown_fields_ignored() {
        let json = r#"{
            "hidden_size": 384, "num_hidden_layers": 6,
            "num_attention_heads": 12, "intermediate_size": 1536,
            "max_position_embeddings": 512, "vocab_size": 30522,
            "some_brand_new_field": "ignored"
        }"#;
        let cfg = HfBertConfig::from_json_str(json).unwrap();
        cfg.validate().unwrap();
    }
}
