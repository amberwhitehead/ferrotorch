// Crate-level lint baseline. Mirrors the ferrotorch-llama posture:
// deny correctness / idiom / Debug / docs problems; warn pedantic
// stylistic issues. Specific pedantic lints are allowed crate-wide
// where the lint is consistently wrong for ML/numeric kernel code.

#![deny(unsafe_code)]
#![deny(rust_2018_idioms)]
#![deny(missing_debug_implementations)]
#![deny(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
// Casts: dimension math (`as usize`, `as f32`, `as u32`) is intrinsic
// to tensor indexing — every kernel call would otherwise need a
// per-call allow.
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_lossless)]
// Builder-style accessors don't all need `#[must_use]`.
#![allow(clippy::must_use_candidate)]
// Identifiers like `bf16`, `f32`, `LayerNorm`, `BERT` are flagged as
// missing backticks even when they appear in code-fenced text.
#![allow(clippy::doc_markdown)]
// `needless_pass_by_value` would force `&BertConfig` signatures
// throughout, hiding intent in the API.
#![allow(clippy::needless_pass_by_value)]
// `unnecessary_wraps` flags `Result`-returning helpers that today
// always succeed but are part of an extensible API surface.
#![allow(clippy::unnecessary_wraps)]
// `uninlined_format_args` flags `format!("x={}", x)` vs
// `format!("x={x}")`. Both are equally clear; the fixup churn is
// high.
#![allow(clippy::uninlined_format_args)]
// `many_single_char_names` flags conventional ML kernel locals
// (`q`, `k`, `v`, `h`).
#![allow(clippy::many_single_char_names)]
// `similar_names` flags variable pairs that are intentionally similar
// (e.g. `q2` / `q_h`).
#![allow(clippy::similar_names)]

//! BERT-family encoder-only model composition for ferrotorch.
//!
//! Assembles the standard BERT encoder stack from ferrotorch primitives:
//!
//! ```text
//! SentenceTransformer
//! └── BertModel
//!     ├── BertEmbeddings
//!     │   ├── Embedding (word_embeddings)
//!     │   ├── Embedding (position_embeddings, learned absolute)
//!     │   ├── Embedding (token_type_embeddings)
//!     │   └── LayerNorm
//!     └── BertEncoder
//!         └── BertLayer × N
//!             ├── BertAttention
//!             │   ├── BertSelfAttention (q / k / v Linear, no causal mask)
//!             │   └── BertSelfOutput   (Linear + LayerNorm(input + .))    ← post-norm
//!             ├── BertIntermediate     (Linear → GELU)
//!             └── BertOutput           (Linear + LayerNorm(attn_out + .)) ← post-norm
//! ```
//!
//! # Loading real weights
//!
//! [`BertModel::load_hf_state_dict`] accepts a `StateDict` whose keys
//! use the HuggingFace `BertModel` naming convention and rewrites them
//! to match the ferrotorch parameter paths before delegating to
//! [`Module::load_state_dict`](ferrotorch_nn::module::Module::load_state_dict).
//! It returns a [`DropReport`] documenting any upstream keys it
//! intentionally did not consume — currently `embeddings.position_ids`
//! (a `[1, max_pos]` buffer regenerated each forward) and any
//! `pooler.*` keys (sentence-transformers does not use the pooler).
//!
//! Combined with `ferrotorch_serialize::load_safetensors` and the
//! [`load_sentence_transformer`] helper this gives a direct path from
//! a downloaded `sentence-transformers/all-MiniLM-L6-v2` checkpoint
//! to a loaded model ready to compute sentence embeddings.

pub mod attention;
pub mod config;
pub mod embeddings;
pub mod layer;
pub mod model;
pub mod safetensors_loader;

pub use attention::{BertAttention, BertSelfAttention, BertSelfOutput};
pub use config::{BertConfig, HfBertConfig};
pub use embeddings::BertEmbeddings;
pub use layer::{BertIntermediate, BertLayer, BertOutput};
pub use model::{BertEncoder, BertModel, DropReport, SentenceTransformer};
pub use safetensors_loader::{load_bert_model, load_sentence_transformer};
