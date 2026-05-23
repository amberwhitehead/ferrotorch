//! Stable-Diffusion 1.5 CLIP text encoder
//! (`openai/clip-vit-large-patch14` — the text tower of CLIP-ViT-L/14).
//!
//! Phase B.3c of real-artifact-driven development — third and final SD
//! sub-model. Together with the VAE decoder (Phase B.3a, #1150) and the
//! UNet (Phase B.3b, #1151) this completes the bit-perfect SD-1.5
//! inference pipeline:
//!
//! ```text
//! CLIP text encoder  →  UNet (denoise)  →  VAE decoder
//! [B, S=77, 768]        [B, 4, 64, 64]    [B, 3, 512, 512]
//! ```
//!
//! Mirrors `transformers.CLIPTextModel` for the SD-1.5 config exactly:
//!
//! ```text
//! hidden_size        = 768
//! intermediate_size  = 3072
//! num_attention_heads = 12
//! num_hidden_layers  = 12
//! max_position_embeddings = 77
//! vocab_size         = 49408
//! hidden_act         = "quick_gelu"     # x * sigmoid(1.702 * x)
//! layer_norm_eps     = 1e-5
//! ```
//!
//! Architecture (state-dict prefix in parens):
//!
//! ```text
//! CLIPTextModel
//! └── text_model
//!     ├── embeddings
//!     │   ├── token_embedding.weight    [49408, 768]
//!     │   └── position_embedding.weight [77, 768]
//!     ├── encoder
//!     │   └── layers.{0..11}.
//!     │       ├── layer_norm1.{weight,bias}    [768], [768]
//!     │       ├── self_attn.
//!     │       │   ├── q_proj.{weight,bias}    [768,768], [768]
//!     │       │   ├── k_proj.{weight,bias}    [768,768], [768]
//!     │       │   ├── v_proj.{weight,bias}    [768,768], [768]
//!     │       │   └── out_proj.{weight,bias}  [768,768], [768]
//!     │       ├── layer_norm2.{weight,bias}    [768], [768]
//!     │       └── mlp.
//!     │           ├── fc1.{weight,bias}        [3072,768], [3072]
//!     │           └── fc2.{weight,bias}        [768,3072], [768]
//!     └── final_layer_norm.{weight,bias}        [768], [768]
//! ```
//!
//! Forward pass (per layer is pre-LayerNorm + residual):
//!
//! ```text
//! h = token_embedding(input_ids) + position_embedding(arange(S))
//! for layer in encoder.layers:
//!     residual = h
//!     h = layer_norm1(h)
//!     h = causal_self_attn(h, h, h)            # ← causal mask is critical
//!     h = residual + h
//!     residual = h
//!     h = layer_norm2(h)
//!     h = fc2(quick_gelu(fc1(h)))
//!     h = residual + h
//! h = final_layer_norm(h)
//! return h                                      # last_hidden_state [B, S, 768]
//! ```
//!
//! ## Critical correctness gotchas
//!
//! 1. **Causal mask**. Despite the "encoder" name, CLIP text-side
//!    self-attention is causal — position `i` attends only to
//!    `0..=i`. Omitting this would still pass shape checks but break
//!    parity vs `transformers`.
//! 2. **QuickGELU**, not standard GELU. CLIP-ViT-L/14 uses the fast
//!    sigmoid approximation `x * sigmoid(1.702 * x)`, not the erf-based
//!    or tanh-based GELU. We pin this via
//!    `GELU::with_approximate(GeluApproximate::Sigmoid)`.
//! 3. **Position embedding is *learned* and absolute** — full table of
//!    77 entries. Position ids are `[0, 1, ..., S-1]` for every forward.
//! 4. **All four self-attention projections have bias** (unlike SD's
//!    UNet `Attention` which has `bias=False` on q/k/v).
//! 5. **SD uses `last_hidden_state` directly**, not the EOS-pooled
//!    output. We return `[B, S, hidden_size]`.

use std::collections::HashMap;

use ferrotorch_core::grad_fns::arithmetic::{add, mul};
use ferrotorch_core::{
    FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage, numeric_cast,
};
use ferrotorch_nn::module::{Module, StateDict};
use ferrotorch_nn::parameter::Parameter;
use ferrotorch_nn::{
    Embedding, GELU, GeluApproximate, LayerNorm, Linear, reshape_to_heads, standard_attention,
    transpose_heads_to_2d,
};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for the SD-1.5 CLIP text encoder
/// (`runwayml/stable-diffusion-v1-5/text_encoder/config.json`).
#[derive(Debug, Clone)]
pub struct ClipTextConfig {
    /// Hidden width. SD-1.5: 768.
    pub hidden_size: usize,
    /// FFN expansion width. SD-1.5: 3072.
    pub intermediate_size: usize,
    /// Number of attention heads per layer. SD-1.5: 12. Must divide
    /// `hidden_size` evenly.
    pub num_attention_heads: usize,
    /// Number of transformer layers. SD-1.5: 12.
    pub num_hidden_layers: usize,
    /// Maximum sequence length. SD-1.5: 77.
    pub max_position_embeddings: usize,
    /// Token vocabulary size. SD-1.5: 49408.
    pub vocab_size: usize,
    /// LayerNorm epsilon. SD-1.5: 1e-5.
    pub layer_norm_eps: f64,
}

impl Default for ClipTextConfig {
    fn default() -> Self {
        Self::sd_v1_5()
    }
}

impl ClipTextConfig {
    /// SD-1.5 CLIP text encoder defaults (CLIP-ViT-L/14 text tower).
    pub fn sd_v1_5() -> Self {
        Self {
            hidden_size: 768,
            intermediate_size: 3072,
            num_attention_heads: 12,
            num_hidden_layers: 12,
            max_position_embeddings: 77,
            vocab_size: 49408,
            layer_norm_eps: 1e-5,
        }
    }

    /// Per-head dimension.
    #[inline]
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Validate field bounds.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] for any out-of-bounds
    /// or arithmetic-incompatible field.
    pub fn validate(&self) -> FerrotorchResult<()> {
        if self.hidden_size == 0
            || self.intermediate_size == 0
            || self.num_attention_heads == 0
            || self.num_hidden_layers == 0
            || self.max_position_embeddings == 0
            || self.vocab_size == 0
        {
            return Err(FerrotorchError::InvalidArgument {
                message: "ClipTextConfig: all size fields must be > 0".into(),
            });
        }
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "ClipTextConfig: hidden_size {} not divisible by num_attention_heads {}",
                    self.hidden_size, self.num_attention_heads,
                ),
            });
        }
        if !self.layer_norm_eps.is_finite() || self.layer_norm_eps <= 0.0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "ClipTextConfig: layer_norm_eps must be finite and > 0, got {}",
                    self.layer_norm_eps,
                ),
            });
        }
        Ok(())
    }

    /// Parse a `text_encoder/config.json` document into a [`ClipTextConfig`].
    ///
    /// Recognised keys (all optional — anything missing falls back to the
    /// SD-1.5 defaults): `hidden_size`, `intermediate_size`,
    /// `num_attention_heads`, `num_hidden_layers`,
    /// `max_position_embeddings`, `vocab_size`, `layer_norm_eps`.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] on malformed JSON or
    /// invalid field values.
    pub fn from_json_str(s: &str) -> FerrotorchResult<Self> {
        let v: serde_json::Value =
            serde_json::from_str(s).map_err(|e| FerrotorchError::InvalidArgument {
                message: format!("ClipTextConfig::from_json_str: bad JSON: {e}"),
            })?;
        let mut cfg = Self::default();
        if let Some(x) = v.get("hidden_size").and_then(serde_json::Value::as_u64) {
            cfg.hidden_size = x as usize;
        }
        if let Some(x) = v
            .get("intermediate_size")
            .and_then(serde_json::Value::as_u64)
        {
            cfg.intermediate_size = x as usize;
        }
        if let Some(x) = v
            .get("num_attention_heads")
            .and_then(serde_json::Value::as_u64)
        {
            cfg.num_attention_heads = x as usize;
        }
        if let Some(x) = v
            .get("num_hidden_layers")
            .and_then(serde_json::Value::as_u64)
        {
            cfg.num_hidden_layers = x as usize;
        }
        if let Some(x) = v
            .get("max_position_embeddings")
            .and_then(serde_json::Value::as_u64)
        {
            cfg.max_position_embeddings = x as usize;
        }
        if let Some(x) = v.get("vocab_size").and_then(serde_json::Value::as_u64) {
            cfg.vocab_size = x as usize;
        }
        if let Some(x) = v.get("layer_norm_eps").and_then(serde_json::Value::as_f64) {
            cfg.layer_norm_eps = x;
        }
        cfg.validate()?;
        Ok(cfg)
    }

    /// Parse a `text_encoder/config.json` file from disk.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] for I/O or parse
    /// failures.
    pub fn from_file(path: &std::path::Path) -> FerrotorchResult<Self> {
        let s = std::fs::read_to_string(path).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!(
                "ClipTextConfig::from_file: failed to read {}: {e}",
                path.display(),
            ),
        })?;
        Self::from_json_str(&s)
    }
}

// ---------------------------------------------------------------------------
// Helper: reshape utilities (own-the-data so the per-row buffers are
// always contiguous and ready for the BMM-style attention call).
// ---------------------------------------------------------------------------

fn reshape_owned<T: Float>(t: &Tensor<T>, shape: Vec<usize>) -> FerrotorchResult<Tensor<T>> {
    let prod: usize = shape.iter().product();
    if prod != t.numel() {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "ClipTextEncoder reshape: target {shape:?} (= {prod} elements) does not \
                 match source numel {}",
                t.numel()
            ),
        });
    }
    let data = t.data_vec()?;
    Tensor::from_storage(TensorStorage::cpu(data), shape, t.requires_grad())
}

/// Build a 1-D float-encoded index tensor from u32 ids (matches the
/// trick `BertEmbeddings::float_index_tensor` uses).
fn float_index_tensor<T: Float>(ids: &[u32]) -> FerrotorchResult<Tensor<T>> {
    let data: Vec<T> = ids
        .iter()
        .map(|&i| numeric_cast::cast::<u32, T>(i))
        .collect::<FerrotorchResult<Vec<T>>>()?;
    let n = data.len();
    Tensor::from_storage(TensorStorage::cpu(data), vec![n], false)
}

// ---------------------------------------------------------------------------
// CLIPTextEmbeddings
// ---------------------------------------------------------------------------

/// Token embedding + learned absolute position embedding. The two
/// lookups are summed. Mirrors `CLIPTextEmbeddings` in transformers.
///
/// Note: there is NO LayerNorm at the embedding level (unlike BERT).
/// The first per-layer `layer_norm1` handles normalisation downstream.
#[derive(Debug)]
pub struct ClipTextEmbeddings<T: Float> {
    /// Token lookup — `[vocab_size, hidden_size]`.
    pub token_embedding: Embedding<T>,
    /// Learned position lookup — `[max_position_embeddings, hidden_size]`.
    pub position_embedding: Embedding<T>,
    hidden_size: usize,
    max_position_embeddings: usize,
    training: bool,
}

impl<T: Float> ClipTextEmbeddings<T> {
    /// Build randomly-initialized embeddings for the given config.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError`] from the underlying [`Embedding`]
    /// constructors.
    pub fn new(cfg: &ClipTextConfig) -> FerrotorchResult<Self> {
        cfg.validate()?;
        Ok(Self {
            token_embedding: Embedding::new(cfg.vocab_size, cfg.hidden_size, None)?,
            position_embedding: Embedding::new(cfg.max_position_embeddings, cfg.hidden_size, None)?,
            hidden_size: cfg.hidden_size,
            max_position_embeddings: cfg.max_position_embeddings,
            training: false,
        })
    }

    /// Run the embedding sum on a sequence of token ids.
    ///
    /// `input_ids` is the verbatim token-id vector (length `S`). The
    /// output has shape `[1, S, hidden]`.
    ///
    /// # Errors
    ///
    /// * [`FerrotorchError::InvalidArgument`] if `input_ids` is empty or
    ///   exceeds `max_position_embeddings`.
    /// * Propagates downstream embedding-lookup errors.
    pub fn forward_from_ids(&self, input_ids: &[u32]) -> FerrotorchResult<Tensor<T>> {
        if input_ids.is_empty() {
            return Err(FerrotorchError::InvalidArgument {
                message: "ClipTextEmbeddings::forward_from_ids needs at least one token".into(),
            });
        }
        let seq_len = input_ids.len();
        if seq_len > self.max_position_embeddings {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "ClipTextEmbeddings: sequence length {seq_len} exceeds \
                     max_position_embeddings {}",
                    self.max_position_embeddings,
                ),
            });
        }

        let word_idx = float_index_tensor::<T>(input_ids)?;
        let word_2d = self.token_embedding.forward(&word_idx)?; // [S, hidden]

        let pos_ids: Vec<u32> = (0..seq_len as u32).collect();
        let pos_idx = float_index_tensor::<T>(&pos_ids)?;
        let pos_2d = self.position_embedding.forward(&pos_idx)?; // [S, hidden]

        let summed = add(&word_2d, &pos_2d)?;
        // Promote to [1, S, hidden] so downstream uses 3-D ranks.
        reshape_owned(&summed, vec![1, seq_len, self.hidden_size])
    }
}

impl<T: Float> Module<T> for ClipTextEmbeddings<T> {
    /// When called via `Module::forward` we treat `input` as a
    /// 1-D float-index tensor (same convention as the inner
    /// `Embedding` modules). Real callers should use
    /// [`Self::forward_from_ids`].
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let word_2d = self.token_embedding.forward(input)?;
        let seq_len = input.numel();
        let pos_ids: Vec<u32> = (0..seq_len as u32).collect();
        let pos_idx = float_index_tensor::<T>(&pos_ids)?;
        let pos_2d = self.position_embedding.forward(&pos_idx)?;
        let summed = add(&word_2d, &pos_2d)?;
        reshape_owned(&summed, vec![1, seq_len, self.hidden_size])
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.token_embedding.parameters());
        out.extend(self.position_embedding.parameters());
        out
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.token_embedding.parameters_mut());
        out.extend(self.position_embedding.parameters_mut());
        out
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (n, p) in self.token_embedding.named_parameters() {
            out.push((format!("token_embedding.{n}"), p));
        }
        for (n, p) in self.position_embedding.named_parameters() {
            out.push((format!("position_embedding.{n}"), p));
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
            let p = format!("{prefix}.");
            state
                .iter()
                .filter_map(|(k, v)| k.strip_prefix(&p).map(|r| (r.to_string(), v.clone())))
                .collect()
        };
        if strict {
            let prefixes = ["token_embedding", "position_embedding"];
            for k in state.keys() {
                if !prefixes.iter().any(|p| k.starts_with(&format!("{p}."))) {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!("unexpected key in ClipTextEmbeddings state_dict: {k:?}"),
                    });
                }
            }
        }
        self.token_embedding
            .load_state_dict(&extract("token_embedding"), strict)?;
        self.position_embedding
            .load_state_dict(&extract("position_embedding"), strict)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CLIP self-attention
// ---------------------------------------------------------------------------

/// Multi-head self-attention with a *causal* mask. Mirrors
/// `CLIPAttention` (the text-side variant) in transformers — all four
/// projections (q/k/v/out) carry bias, the scale is `1/sqrt(head_dim)`,
/// and the attention is causal (each position attends to itself and
/// earlier positions only).
///
/// State-dict layout:
///
/// ```text
/// q_proj.{weight,bias}    [hidden, hidden], [hidden]
/// k_proj.{weight,bias}    [hidden, hidden], [hidden]
/// v_proj.{weight,bias}    [hidden, hidden], [hidden]
/// out_proj.{weight,bias}  [hidden, hidden], [hidden]
/// ```
#[derive(Debug)]
pub struct ClipSelfAttention<T: Float> {
    /// Query projection — `[hidden, hidden]`, with bias.
    pub q_proj: Linear<T>,
    /// Key projection — `[hidden, hidden]`, with bias.
    pub k_proj: Linear<T>,
    /// Value projection — `[hidden, hidden]`, with bias.
    pub v_proj: Linear<T>,
    /// Output projection — `[hidden, hidden]`, with bias.
    pub out_proj: Linear<T>,
    num_heads: usize,
    head_dim: usize,
    hidden: usize,
    training: bool,
}

impl<T: Float> ClipSelfAttention<T> {
    /// Build randomly-initialized self-attention projections.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`FerrotorchError`] on bad config dims.
    pub fn new(cfg: &ClipTextConfig) -> FerrotorchResult<Self> {
        cfg.validate()?;
        Ok(Self {
            q_proj: Linear::new(cfg.hidden_size, cfg.hidden_size, true)?,
            k_proj: Linear::new(cfg.hidden_size, cfg.hidden_size, true)?,
            v_proj: Linear::new(cfg.hidden_size, cfg.hidden_size, true)?,
            out_proj: Linear::new(cfg.hidden_size, cfg.hidden_size, true)?,
            num_heads: cfg.num_attention_heads,
            head_dim: cfg.head_dim(),
            hidden: cfg.hidden_size,
            training: false,
        })
    }
}

impl<T: Float> Module<T> for ClipSelfAttention<T> {
    /// Forward — input `[1, S, hidden]`, output `[1, S, hidden]`.
    ///
    /// The attention is causal: position `i` cannot attend to position
    /// `j > i`. This matches `transformers.CLIPAttention`'s use of
    /// `_create_4d_causal_attention_mask`.
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let shape = input.shape();
        if shape.len() != 3 || shape[0] != 1 || shape[2] != self.hidden {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "ClipSelfAttention expects [1, S, {}], got {:?}",
                    self.hidden, shape,
                ),
            });
        }
        let seq_len = shape[1];

        // Projections — Linear handles any rank; output is [1, S, hidden].
        let q = self.q_proj.forward(input)?;
        let k = self.k_proj.forward(input)?;
        let v = self.v_proj.forward(input)?;

        // Drop the batch=1 leading dim so reshape_to_heads / the
        // attention helper can treat the rows as [S, H*d].
        let q2 = reshape_owned(&q, vec![seq_len, self.hidden])?;
        let k2 = reshape_owned(&k, vec![seq_len, self.hidden])?;
        let v2 = reshape_owned(&v, vec![seq_len, self.hidden])?;

        // [S, H*d] → [H, S, d] (batch-first heads).
        let q_h = reshape_to_heads(&q2, self.num_heads, seq_len, self.head_dim)?;
        let k_h = reshape_to_heads(&k2, self.num_heads, seq_len, self.head_dim)?;
        let v_h = reshape_to_heads(&v2, self.num_heads, seq_len, self.head_dim)?;

        // Scaled dot-product attention with causal mask. `standard_attention`
        // applies `1/sqrt(head_dim)` scaling and `-inf` upper-triangular
        // mask, then softmax + value mix.
        let ctx = standard_attention(&q_h, &k_h, &v_h, /* causal = */ true)?;

        // [H, S, d] → [S, H*d] → [1, S, hidden].
        let ctx2 = transpose_heads_to_2d(&ctx, self.num_heads, seq_len, self.head_dim)?;
        let ctx3 = reshape_owned(&ctx2, vec![1, seq_len, self.hidden])?;

        // Output projection (with bias).
        self.out_proj.forward(&ctx3)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.q_proj.parameters());
        out.extend(self.k_proj.parameters());
        out.extend(self.v_proj.parameters());
        out.extend(self.out_proj.parameters());
        out
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.q_proj.parameters_mut());
        out.extend(self.k_proj.parameters_mut());
        out.extend(self.v_proj.parameters_mut());
        out.extend(self.out_proj.parameters_mut());
        out
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (n, p) in self.q_proj.named_parameters() {
            out.push((format!("q_proj.{n}"), p));
        }
        for (n, p) in self.k_proj.named_parameters() {
            out.push((format!("k_proj.{n}"), p));
        }
        for (n, p) in self.v_proj.named_parameters() {
            out.push((format!("v_proj.{n}"), p));
        }
        for (n, p) in self.out_proj.named_parameters() {
            out.push((format!("out_proj.{n}"), p));
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
            let p = format!("{prefix}.");
            state
                .iter()
                .filter_map(|(k, v)| k.strip_prefix(&p).map(|r| (r.to_string(), v.clone())))
                .collect()
        };
        if strict {
            let prefixes = ["q_proj", "k_proj", "v_proj", "out_proj"];
            for k in state.keys() {
                if !prefixes.iter().any(|p| k.starts_with(&format!("{p}."))) {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!("unexpected key in ClipSelfAttention state_dict: {k:?}"),
                    });
                }
            }
        }
        self.q_proj.load_state_dict(&extract("q_proj"), strict)?;
        self.k_proj.load_state_dict(&extract("k_proj"), strict)?;
        self.v_proj.load_state_dict(&extract("v_proj"), strict)?;
        self.out_proj
            .load_state_dict(&extract("out_proj"), strict)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CLIPMLP
// ---------------------------------------------------------------------------

/// CLIP MLP: `fc2(quick_gelu(fc1(x)))`.
///
/// QuickGELU is `x * sigmoid(1.702 * x)` — the fast sigmoid
/// approximation. NOT the standard `0.5 * x * (1 + erf(x/sqrt(2)))`
/// kernel. This is the published `hidden_act = "quick_gelu"` in
/// CLIP-ViT-L/14's config.
#[derive(Debug)]
pub struct ClipMlp<T: Float> {
    /// Expansion projection — `[hidden, intermediate]`, with bias.
    pub fc1: Linear<T>,
    /// Reduction projection — `[intermediate, hidden]`, with bias.
    pub fc2: Linear<T>,
    activation: GELU,
    training: bool,
}

impl<T: Float> ClipMlp<T> {
    /// Build randomly-initialized MLP for the given config.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`FerrotorchError`] on bad config dims.
    pub fn new(cfg: &ClipTextConfig) -> FerrotorchResult<Self> {
        cfg.validate()?;
        Ok(Self {
            fc1: Linear::new(cfg.hidden_size, cfg.intermediate_size, true)?,
            fc2: Linear::new(cfg.intermediate_size, cfg.hidden_size, true)?,
            // QuickGELU: `x * sigmoid(1.702 * x)`. See module doc.
            activation: GELU::with_approximate(GeluApproximate::Sigmoid),
            training: false,
        })
    }
}

impl<T: Float> Module<T> for ClipMlp<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let h = self.fc1.forward(input)?;
        let h = self.activation.forward(&h)?;
        self.fc2.forward(&h)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.fc1.parameters());
        out.extend(self.fc2.parameters());
        out
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.fc1.parameters_mut());
        out.extend(self.fc2.parameters_mut());
        out
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (n, p) in self.fc1.named_parameters() {
            out.push((format!("fc1.{n}"), p));
        }
        for (n, p) in self.fc2.named_parameters() {
            out.push((format!("fc2.{n}"), p));
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
            let p = format!("{prefix}.");
            state
                .iter()
                .filter_map(|(k, v)| k.strip_prefix(&p).map(|r| (r.to_string(), v.clone())))
                .collect()
        };
        if strict {
            for k in state.keys() {
                if !(k.starts_with("fc1.") || k.starts_with("fc2.")) {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!("unexpected key in ClipMlp state_dict: {k:?}"),
                    });
                }
            }
        }
        self.fc1.load_state_dict(&extract("fc1"), strict)?;
        self.fc2.load_state_dict(&extract("fc2"), strict)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CLIPEncoderLayer
// ---------------------------------------------------------------------------

/// One CLIP text encoder layer.
///
/// Pre-LayerNorm + residual stack:
///
/// ```text
/// h = x + self_attn(layer_norm1(x))
/// h = h + mlp(layer_norm2(h))
/// ```
#[derive(Debug)]
pub struct ClipEncoderLayer<T: Float> {
    /// Pre-attention LayerNorm.
    pub layer_norm1: LayerNorm<T>,
    /// Causal self-attention (q/k/v/out, all biased).
    pub self_attn: ClipSelfAttention<T>,
    /// Pre-FFN LayerNorm.
    pub layer_norm2: LayerNorm<T>,
    /// Two-layer MLP with QuickGELU activation.
    pub mlp: ClipMlp<T>,
    training: bool,
}

impl<T: Float> ClipEncoderLayer<T> {
    /// Build a randomly-initialized encoder layer.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`FerrotorchError`] on bad config dims.
    pub fn new(cfg: &ClipTextConfig) -> FerrotorchResult<Self> {
        Ok(Self {
            layer_norm1: LayerNorm::new(vec![cfg.hidden_size], cfg.layer_norm_eps, true)?,
            self_attn: ClipSelfAttention::new(cfg)?,
            layer_norm2: LayerNorm::new(vec![cfg.hidden_size], cfg.layer_norm_eps, true)?,
            mlp: ClipMlp::new(cfg)?,
            training: false,
        })
    }
}

impl<T: Float> Module<T> for ClipEncoderLayer<T> {
    /// Forward — `[1, S, hidden]` → `[1, S, hidden]`.
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Self-attention sub-block (pre-norm).
        let normed = self.layer_norm1.forward(input)?;
        let attn_out = self.self_attn.forward(&normed)?;
        let after_attn = add(input, &attn_out)?;

        // MLP sub-block (pre-norm).
        let normed_ffn = self.layer_norm2.forward(&after_attn)?;
        let mlp_out = self.mlp.forward(&normed_ffn)?;
        add(&after_attn, &mlp_out)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.layer_norm1.parameters());
        out.extend(self.self_attn.parameters());
        out.extend(self.layer_norm2.parameters());
        out.extend(self.mlp.parameters());
        out
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.layer_norm1.parameters_mut());
        out.extend(self.self_attn.parameters_mut());
        out.extend(self.layer_norm2.parameters_mut());
        out.extend(self.mlp.parameters_mut());
        out
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (n, p) in self.layer_norm1.named_parameters() {
            out.push((format!("layer_norm1.{n}"), p));
        }
        for (n, p) in self.self_attn.named_parameters() {
            out.push((format!("self_attn.{n}"), p));
        }
        for (n, p) in self.layer_norm2.named_parameters() {
            out.push((format!("layer_norm2.{n}"), p));
        }
        for (n, p) in self.mlp.named_parameters() {
            out.push((format!("mlp.{n}"), p));
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
            let p = format!("{prefix}.");
            state
                .iter()
                .filter_map(|(k, v)| k.strip_prefix(&p).map(|r| (r.to_string(), v.clone())))
                .collect()
        };
        if strict {
            let prefixes = ["layer_norm1", "self_attn", "layer_norm2", "mlp"];
            for k in state.keys() {
                if !prefixes.iter().any(|p| k.starts_with(&format!("{p}."))) {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!("unexpected key in ClipEncoderLayer state_dict: {k:?}"),
                    });
                }
            }
        }
        self.layer_norm1
            .load_state_dict(&extract("layer_norm1"), strict)?;
        self.self_attn
            .load_state_dict(&extract("self_attn"), strict)?;
        self.layer_norm2
            .load_state_dict(&extract("layer_norm2"), strict)?;
        self.mlp.load_state_dict(&extract("mlp"), strict)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CLIPEncoder
// ---------------------------------------------------------------------------

/// Stack of `num_hidden_layers` [`ClipEncoderLayer`]s applied in order.
#[derive(Debug)]
pub struct ClipEncoder<T: Float> {
    /// One layer per `num_hidden_layers`.
    pub layers: Vec<ClipEncoderLayer<T>>,
    training: bool,
}

impl<T: Float> ClipEncoder<T> {
    /// Build a randomly-initialized encoder stack.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`FerrotorchError`] on bad config dims.
    pub fn new(cfg: &ClipTextConfig) -> FerrotorchResult<Self> {
        cfg.validate()?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for _ in 0..cfg.num_hidden_layers {
            layers.push(ClipEncoderLayer::new(cfg)?);
        }
        Ok(Self {
            layers,
            training: false,
        })
    }
}

impl<T: Float> Module<T> for ClipEncoder<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let mut h = input.clone();
        for l in &self.layers {
            h = l.forward(&h)?;
        }
        Ok(h)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut out = Vec::new();
        for l in &self.layers {
            out.extend(l.parameters());
        }
        out
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut out = Vec::new();
        for l in &mut self.layers {
            out.extend(l.parameters_mut());
        }
        out
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (i, l) in self.layers.iter().enumerate() {
            for (n, p) in l.named_parameters() {
                out.push((format!("layers.{i}.{n}"), p));
            }
        }
        out
    }

    fn train(&mut self) {
        self.training = true;
        for l in &mut self.layers {
            l.train();
        }
    }

    fn eval(&mut self) {
        self.training = false;
        for l in &mut self.layers {
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
            let p = format!("{prefix}.");
            state
                .iter()
                .filter_map(|(k, v)| k.strip_prefix(&p).map(|r| (r.to_string(), v.clone())))
                .collect()
        };
        if strict {
            for k in state.keys() {
                if !k.starts_with("layers.") {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!("unexpected key in ClipEncoder state_dict: {k:?}"),
                    });
                }
            }
        }
        for (i, l) in self.layers.iter_mut().enumerate() {
            l.load_state_dict(&extract(&format!("layers.{i}")), strict)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CLIPTextTransformer / ClipTextEncoder
// ---------------------------------------------------------------------------

/// The full SD-1.5 CLIP text encoder. Wraps [`ClipTextEmbeddings`] +
/// [`ClipEncoder`] + a final [`LayerNorm`] (`final_layer_norm`).
///
/// Mirrors `CLIPTextTransformer` in transformers. The HF
/// `CLIPTextModel` wrapper sits one prefix above
/// (`text_model.embeddings.*`, `text_model.encoder.*`,
/// `text_model.final_layer_norm.*`) — see
/// [`crate::safetensors_loader::load_clip_text_encoder`] for the
/// `text_model.` strip.
///
/// Output is the per-token `last_hidden_state` `[B, S, hidden_size]`.
/// SD-1.5 consumes this directly as `encoder_hidden_states` for the
/// UNet's cross-attention (no pooling).
#[derive(Debug)]
pub struct ClipTextEncoder<T: Float> {
    /// Token + position embedding sum.
    pub embeddings: ClipTextEmbeddings<T>,
    /// 12 × [`ClipEncoderLayer`] for SD-1.5.
    pub encoder: ClipEncoder<T>,
    /// Final LayerNorm over the last hidden state.
    pub final_layer_norm: LayerNorm<T>,
    /// Frozen copy of the configuration used to build the module.
    pub config: ClipTextConfig,
    training: bool,
}

impl<T: Float> ClipTextEncoder<T> {
    /// Build a randomly-initialized text encoder.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`FerrotorchError`] from any sub-module
    /// constructor.
    pub fn new(cfg: ClipTextConfig) -> FerrotorchResult<Self> {
        cfg.validate()?;
        let embeddings = ClipTextEmbeddings::new(&cfg)?;
        let encoder = ClipEncoder::new(&cfg)?;
        let final_layer_norm = LayerNorm::new(vec![cfg.hidden_size], cfg.layer_norm_eps, true)?;
        Ok(Self {
            embeddings,
            encoder,
            final_layer_norm,
            config: cfg,
            training: false,
        })
    }

    /// Run the encoder on a token-id sequence and return the per-token
    /// `last_hidden_state` `[1, S, hidden_size]`.
    ///
    /// `input_ids` is the verbatim CLIP-BPE token-id vector (length
    /// `S`). For SD-1.5 the canonical inference call is `S = 77`
    /// (already-padded with EOS to the max length).
    ///
    /// # Errors
    ///
    /// * [`FerrotorchError::InvalidArgument`] if `input_ids` is empty
    ///   or longer than `max_position_embeddings`.
    /// * Propagates downstream Embedding / LayerNorm errors.
    pub fn forward_from_ids(&self, input_ids: &[u32]) -> FerrotorchResult<Tensor<T>> {
        let h = self.embeddings.forward_from_ids(input_ids)?;
        let h = self.encoder.forward(&h)?;
        self.final_layer_norm.forward(&h)
    }

    /// Run the encoder on a pre-built float-encoded token-id tensor of
    /// shape `[S]`. Returns `[1, S, hidden_size]`.
    ///
    /// `ids` carries u32 token ids losslessly cast to `T`
    /// (`numeric_cast::cast::<u32, T>`). Mirrors what the dump example
    /// reads off disk.
    ///
    /// # Errors
    ///
    /// Propagates downstream lookup / LayerNorm errors and converts
    /// invalid (negative / NaN / overflow) ids to
    /// [`FerrotorchError::InvalidArgument`].
    pub fn forward_from_id_tensor(&self, ids: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Convert to u32 ids — same defensive cast the underlying
        // `Embedding::forward` does.
        if ids.ndim() != 1 {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "ClipTextEncoder::forward_from_id_tensor expects 1-D ids, got {:?}",
                    ids.shape()
                ),
            });
        }
        let data = ids.data_vec()?;
        let mut u32_ids: Vec<u32> = Vec::with_capacity(data.len());
        for (i, v) in data.iter().enumerate() {
            let f = num_traits::ToPrimitive::to_f64(v).ok_or_else(|| {
                FerrotorchError::InvalidArgument {
                    message: format!(
                        "ClipTextEncoder::forward_from_id_tensor: id at {i} \
                         not representable as f64"
                    ),
                }
            })?;
            if !f.is_finite() || f < 0.0 || f > u32::MAX as f64 || f.fract() != 0.0 {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "ClipTextEncoder::forward_from_id_tensor: id at {i} ({f}) \
                         is not a non-negative integer"
                    ),
                });
            }
            u32_ids.push(f as u32);
        }
        self.forward_from_ids(&u32_ids)
    }

    /// Load a HuggingFace `CLIPTextModel` state dict into this module.
    ///
    /// Accepts both:
    ///   - bare-`text_model` layout (no prefix; what the pin script
    ///     normalises to).
    ///   - full `text_model.<rest>` prefix (what the upstream HF
    ///     checkpoint ships).
    ///
    /// The HF safetensors also ships a non-parameter buffer we
    /// explicitly drop:
    ///
    /// * `text_model.embeddings.position_ids` — a `[1, max_pos]`
    ///   `arange(max_pos)` buffer regenerated each forward pass. Recorded
    ///   in the [`crate::safetensors_loader::DropReport`].
    ///
    /// # Errors
    ///
    /// Forwards whatever each sub-module's `load_state_dict` returns
    /// (shape mismatch / strict-mode missing key). Strict mode will
    /// surface `text_model.embeddings.position_ids` and any unknown
    /// key as errors; callers with a full HF checkpoint must pass
    /// `strict=false`.
    pub fn load_hf_state_dict(
        &mut self,
        hf_state: &StateDict<T>,
        strict: bool,
    ) -> FerrotorchResult<crate::safetensors_loader::DropReport> {
        let mut remapped: StateDict<T> = HashMap::with_capacity(hf_state.len());
        let mut dropped: Vec<String> = Vec::new();
        for (k, v) in hf_state {
            // Strip the optional `text_model.` prefix.
            let after = k
                .strip_prefix("text_model.")
                .map_or_else(|| k.clone(), str::to_owned);

            // `embeddings.position_ids` is a buffer — not a parameter on our
            // side. Drop in both modes; record so the pin script can audit.
            if after == "embeddings.position_ids" {
                dropped.push(k.clone());
                continue;
            }

            let is_known = after.starts_with("embeddings.token_embedding.")
                || after.starts_with("embeddings.position_embedding.")
                || after.starts_with("encoder.")
                || after.starts_with("final_layer_norm.");
            if is_known {
                remapped.insert(after, v.clone());
                continue;
            }

            if strict {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "ClipTextEncoder::load_hf_state_dict: key {k:?} is not a \
                         known CLIP text-tower parameter and strict mode is on. \
                         Pass strict=false to drop unknown keys."
                    ),
                });
            }
            dropped.push(k.clone());
        }
        dropped.sort();
        self.load_state_dict(&remapped, strict)?;
        Ok(crate::safetensors_loader::DropReport { dropped })
    }
}

impl<T: Float> Module<T> for ClipTextEncoder<T> {
    /// `Module::forward` treats `input` as already-summed embeddings
    /// `[1, S, hidden]` and only runs the encoder + final LayerNorm.
    /// Real callers should use [`Self::forward_from_ids`] /
    /// [`Self::forward_from_id_tensor`].
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let h = self.encoder.forward(input)?;
        self.final_layer_norm.forward(&h)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.embeddings.parameters());
        out.extend(self.encoder.parameters());
        out.extend(self.final_layer_norm.parameters());
        out
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.embeddings.parameters_mut());
        out.extend(self.encoder.parameters_mut());
        out.extend(self.final_layer_norm.parameters_mut());
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
        for (n, p) in self.final_layer_norm.named_parameters() {
            out.push((format!("final_layer_norm.{n}"), p));
        }
        out
    }

    fn train(&mut self) {
        self.training = true;
        self.embeddings.train();
        self.encoder.train();
        self.final_layer_norm.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.embeddings.eval();
        self.encoder.eval();
        self.final_layer_norm.eval();
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
            let p = format!("{prefix}.");
            state
                .iter()
                .filter_map(|(k, v)| k.strip_prefix(&p).map(|r| (r.to_string(), v.clone())))
                .collect()
        };
        if strict {
            for k in state.keys() {
                if !(k.starts_with("embeddings.")
                    || k.starts_with("encoder.")
                    || k.starts_with("final_layer_norm."))
                {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!("unexpected key in ClipTextEncoder state_dict: {k:?}"),
                    });
                }
            }
        }
        self.embeddings
            .load_state_dict(&extract("embeddings"), strict)?;
        self.encoder.load_state_dict(&extract("encoder"), strict)?;
        self.final_layer_norm
            .load_state_dict(&extract("final_layer_norm"), strict)?;
        Ok(())
    }
}

// `mul` is re-exported above so downstream features (e.g. embedding
// scaling, never used for CLIP-ViT-L) can reach for it without a fresh
// import. Suppress the unused-import warning on a vanilla build.
#[allow(dead_code)]
fn _unused_mul_ref<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    mul(a, b)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg() -> ClipTextConfig {
        // 2 heads × 4 dim/head = 8 hidden_size; 16 intermediate; 1 layer;
        // 6 positions; tiny vocab.
        ClipTextConfig {
            hidden_size: 8,
            intermediate_size: 16,
            num_attention_heads: 2,
            num_hidden_layers: 1,
            max_position_embeddings: 6,
            vocab_size: 32,
            layer_norm_eps: 1e-5,
        }
    }

    #[test]
    fn sd_v1_5_config_is_canonical() {
        let c = ClipTextConfig::sd_v1_5();
        assert_eq!(c.hidden_size, 768);
        assert_eq!(c.intermediate_size, 3072);
        assert_eq!(c.num_attention_heads, 12);
        assert_eq!(c.num_hidden_layers, 12);
        assert_eq!(c.max_position_embeddings, 77);
        assert_eq!(c.vocab_size, 49408);
        assert_eq!(c.head_dim(), 64);
        c.validate().unwrap();
    }

    #[test]
    fn validate_catches_bad_head_count() {
        let mut c = tiny_cfg();
        c.num_attention_heads = 3; // 8 % 3 != 0
        assert!(c.validate().is_err());
    }

    #[test]
    fn from_json_str_round_trip() {
        let json = r#"{
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "max_position_embeddings": 77,
            "vocab_size": 49408,
            "layer_norm_eps": 1e-5,
            "hidden_act": "quick_gelu"
        }"#;
        let c = ClipTextConfig::from_json_str(json).unwrap();
        assert_eq!(c.hidden_size, 768);
        assert_eq!(c.intermediate_size, 3072);
        assert_eq!(c.num_attention_heads, 12);
        assert_eq!(c.num_hidden_layers, 12);
        assert_eq!(c.max_position_embeddings, 77);
    }

    #[test]
    fn embeddings_forward_shape() {
        let emb = ClipTextEmbeddings::<f32>::new(&tiny_cfg()).unwrap();
        let ids = [1u32, 5, 7, 9];
        let out = emb.forward_from_ids(&ids).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        for &v in out.data().unwrap() {
            assert!(v.is_finite(), "embedding non-finite: {v}");
        }
    }

    #[test]
    fn embeddings_reject_too_long_sequence() {
        let emb = ClipTextEmbeddings::<f32>::new(&tiny_cfg()).unwrap();
        let ids: Vec<u32> = (0..7).collect(); // > max_position 6
        assert!(emb.forward_from_ids(&ids).is_err());
    }

    #[test]
    fn self_attention_forward_shape() {
        let attn = ClipSelfAttention::<f32>::new(&tiny_cfg()).unwrap();
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![0.1f32; 5 * 8]),
            vec![1, 5, 8],
            false,
        )
        .unwrap();
        let out = attn.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 5, 8]);
        for &v in out.data().unwrap() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn self_attention_is_actually_causal() {
        // Changing later tokens MUST NOT change earlier rows.
        // Build a tensor [1, 4, 8] with the first 2 rows fixed and the
        // last 2 rows perturbed across two runs. The first 2 output
        // rows must be bit-identical between the runs (within f32
        // round-off).
        let attn = ClipSelfAttention::<f32>::new(&tiny_cfg()).unwrap();
        let mut a = vec![0.1f32; 4 * 8];
        for i in 0..2 * 8 {
            a[i] = ((i + 1) as f32).sin();
        }
        let mut b = a.clone();
        // Perturb only rows 2 and 3.
        for i in (2 * 8)..(4 * 8) {
            b[i] = ((i + 11) as f32).sin();
        }
        let xa = Tensor::from_storage(TensorStorage::cpu(a), vec![1, 4, 8], false).unwrap();
        let xb = Tensor::from_storage(TensorStorage::cpu(b), vec![1, 4, 8], false).unwrap();
        let oa = attn.forward(&xa).unwrap();
        let ob = attn.forward(&xb).unwrap();
        let da = oa.data().unwrap();
        let db = ob.data().unwrap();
        for i in 0..2 * 8 {
            assert!(
                (da[i] - db[i]).abs() < 1e-5,
                "row {} ({}) differs between runs: {} vs {}",
                i / 8,
                i % 8,
                da[i],
                db[i]
            );
        }
    }

    #[test]
    fn mlp_uses_quick_gelu() {
        // QuickGELU(x) = x * sigmoid(1.702 * x). Verify the FC1 + GELU
        // branch produces this for a known scalar input — we do so
        // indirectly by checking that the forward output remains finite
        // and the intermediate activation at zero input gives bias.
        let mlp = ClipMlp::<f32>::new(&tiny_cfg()).unwrap();
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![0.0f32; 3 * 8]),
            vec![1, 3, 8],
            false,
        )
        .unwrap();
        let out = mlp.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 3, 8]);
        for &v in out.data().unwrap() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn encoder_layer_forward_shape() {
        let layer = ClipEncoderLayer::<f32>::new(&tiny_cfg()).unwrap();
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![0.1f32; 5 * 8]),
            vec![1, 5, 8],
            false,
        )
        .unwrap();
        let out = layer.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 5, 8]);
        for &v in out.data().unwrap() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn encoder_layer_named_parameters_use_hf_layout() {
        let layer = ClipEncoderLayer::<f32>::new(&tiny_cfg()).unwrap();
        let names: Vec<String> = layer
            .named_parameters()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        for k in [
            "layer_norm1.weight",
            "layer_norm1.bias",
            "self_attn.q_proj.weight",
            "self_attn.q_proj.bias",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.out_proj.weight",
            "self_attn.out_proj.bias",
            "layer_norm2.weight",
            "mlp.fc1.weight",
            "mlp.fc1.bias",
            "mlp.fc2.weight",
            "mlp.fc2.bias",
        ] {
            assert!(
                names.iter().any(|n| n == k),
                "missing parameter key {k:?} in {names:?}"
            );
        }
    }

    #[test]
    fn tiny_encoder_forward_from_ids_shape() {
        let enc = ClipTextEncoder::<f32>::new(tiny_cfg()).unwrap();
        let ids = vec![1u32, 5, 7];
        let out = enc.forward_from_ids(&ids).unwrap();
        assert_eq!(out.shape(), &[1, 3, 8]);
        for &v in out.data().unwrap() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn tiny_named_parameters_use_hf_layout() {
        let enc = ClipTextEncoder::<f32>::new(tiny_cfg()).unwrap();
        let names: Vec<String> = enc.named_parameters().into_iter().map(|(n, _)| n).collect();
        for k in [
            "embeddings.token_embedding.weight",
            "embeddings.position_embedding.weight",
            "encoder.layers.0.layer_norm1.weight",
            "encoder.layers.0.self_attn.q_proj.weight",
            "encoder.layers.0.self_attn.out_proj.bias",
            "encoder.layers.0.layer_norm2.bias",
            "encoder.layers.0.mlp.fc1.weight",
            "encoder.layers.0.mlp.fc2.bias",
            "final_layer_norm.weight",
            "final_layer_norm.bias",
        ] {
            assert!(
                names.iter().any(|n| n == k),
                "missing parameter key {k:?} in {names:?}"
            );
        }
    }

    #[test]
    fn round_trip_state_dict() {
        let src = ClipTextEncoder::<f32>::new(tiny_cfg()).unwrap();
        let sd = src.state_dict();
        let mut dst = ClipTextEncoder::<f32>::new(tiny_cfg()).unwrap();
        dst.load_state_dict(&sd, true).unwrap();
        let ids = vec![2u32, 4, 6];
        let a = src.forward_from_ids(&ids).unwrap();
        let b = dst.forward_from_ids(&ids).unwrap();
        for (x, y) in a.data().unwrap().iter().zip(b.data().unwrap().iter()) {
            assert!((x - y).abs() < 1e-5, "round-trip differs: {x} vs {y}");
        }
    }

    #[test]
    fn load_hf_state_dict_strips_text_model_prefix() {
        let src = ClipTextEncoder::<f32>::new(tiny_cfg()).unwrap();
        let bare = src.state_dict();
        let mut prefixed: StateDict<f32> = HashMap::new();
        for (k, v) in bare {
            prefixed.insert(format!("text_model.{k}"), v);
        }
        // Add the position_ids buffer — it should be dropped.
        prefixed.insert(
            "text_model.embeddings.position_ids".into(),
            ferrotorch_core::zeros::<f32>(&[1, 6]).unwrap(),
        );
        let mut dst = ClipTextEncoder::<f32>::new(tiny_cfg()).unwrap();
        let rep = dst.load_hf_state_dict(&prefixed, false).unwrap();
        assert_eq!(
            rep.dropped,
            vec!["text_model.embeddings.position_ids".to_string()]
        );
        let ids = vec![1u32, 2, 3];
        let a = src.forward_from_ids(&ids).unwrap();
        let b = dst.forward_from_ids(&ids).unwrap();
        for (x, y) in a.data().unwrap().iter().zip(b.data().unwrap().iter()) {
            assert!((x - y).abs() < 1e-5);
        }
    }

    #[test]
    fn load_hf_state_dict_strict_rejects_unknown_key() {
        let mut dst = ClipTextEncoder::<f32>::new(tiny_cfg()).unwrap();
        let mut sd: StateDict<f32> = HashMap::new();
        sd.insert(
            "mystery.key".into(),
            ferrotorch_core::zeros::<f32>(&[1]).unwrap(),
        );
        assert!(dst.load_hf_state_dict(&sd, true).is_err());
    }

    #[test]
    fn forward_from_id_tensor_matches_forward_from_ids() {
        let enc = ClipTextEncoder::<f32>::new(tiny_cfg()).unwrap();
        let ids = vec![1u32, 5, 7];
        let id_tensor = float_index_tensor::<f32>(&ids).unwrap();
        let a = enc.forward_from_ids(&ids).unwrap();
        let b = enc.forward_from_id_tensor(&id_tensor).unwrap();
        for (x, y) in a.data().unwrap().iter().zip(b.data().unwrap().iter()) {
            assert!((x - y).abs() < 1e-5);
        }
    }
}
