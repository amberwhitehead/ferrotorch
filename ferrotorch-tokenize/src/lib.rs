//! Text tokenization for ferrotorch models.
//!
//! ## REQ status (per `.design/ferrotorch-tokenize/lib.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (lint baseline) | SHIPPED | `#![warn]`/`#![deny]`/`#![allow]` block at top of `lib.rs`; verified by `cargo clippy -p ferrotorch-tokenize -- -D warnings`. |
//! | REQ-2 (I/O vs parse error split) | SHIPPED | `pub fn load_tokenizer` + `pub fn load_chat_template` in `lib.rs` separate `Internal` (I/O) from `InvalidArgument` (parse) per #734. |
//! | REQ-3 (encode/decode/vocab surface) | SHIPPED | the `load_tokenizer`/`encode`/`encode_batch`/`decode`/`vocab_size`/`token_to_id`/`id_to_token` entries in `lib.rs` mirror the upstream `tokenizers::Tokenizer` API; consumers: `ferrotorch-llama/examples/{llama3_8b,llama3_8b_gpu,llama3_70b_gpu,llm_inference_dump}.rs`. |
//! | REQ-4 (`apply_chat_template` via minijinja) | SHIPPED | the `apply_chat_template` + `_to_ids` entries in `lib.rs` mirror `transformers/tokenization_utils_base.py:1547`; meta-crate `pub use` boundary per S5. |
//! | REQ-5 (`ChatMessage` with flattened extras) | SHIPPED | `#[non_exhaustive] pub struct ChatMessage` in `lib.rs` with `#[serde(flatten)] extra: BTreeMap<...>`. |
//! | REQ-6 (`load_chat_template` from `tokenizer_config.json`) | SHIPPED | the `load_chat_template` entry in `lib.rs` handles String / Array / None shapes. |
//!
//! This crate is a thin wrapper around the `HuggingFace`
//! [`tokenizers`] crate — the same library powering Python's
//! `transformers.AutoTokenizer` — with an API shaped for ferrotorch
//! idioms (`Vec<u32>` token ids, `FerrotorchResult` errors).
//!
//! # Quick start
//!
//! ```no_run
//! use ferrotorch_tokenize::{load_tokenizer, encode, decode};
//!
//! // Llama 3 ships a `tokenizer.json` alongside its weights.
//! let tok = load_tokenizer("/path/to/tokenizer.json")?;
//! let ids = encode(&tok, "Hello, world!", /* add_special_tokens = */ true)?;
//! let text = decode(&tok, &ids, /* skip_special_tokens = */ false)?;
//! # Ok::<(), ferrotorch_core::FerrotorchError>(())
//! ```
//!
//! # Scope
//!
//! The wrapper covers the path the Llama 3 8B `PoC` needs:
//! - Load a `tokenizer.json` file into a [`Tokenizer`].
//! - Encode / decode single strings and batches.
//! - Query vocab size and special-token ids.
//!
//! More advanced features (chat templates, truncation strategies,
//! added-token manipulation) are available by calling the re-exported
//! [`tokenizers`] API directly on the returned [`Tokenizer`].
//!
//! # REQ status (per `.design/ferrotorch-tokenize/lib.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | impl: lint baseline at top of `lib.rs` — `#![warn(clippy::all, clippy::pedantic)]` + `#![deny(rust_2018_idioms, missing_debug_implementations)]` + documented `#![allow(missing_docs)]`; non-test consumer: the meta-crate `pub use ferrotorch_tokenize::*` re-export in `ferrotorch/src/lib.rs` `pub use ferrotorch_tokenize::*;` |
//! | REQ-2 | SHIPPED | impl: `pub fn load_tokenizer in lib.rs` splits `std::fs::read_to_string` (`Internal`) from `Tokenizer::from_str` (`InvalidArgument`); `pub fn load_chat_template in lib.rs` splits `std::fs::read` from `serde_json::from_slice` identically; non-test consumer: `ferrotorch-llama/examples/llama3_8b.rs` (greedy-decode `load_tokenizer` call site) calls `load_tokenizer(&tok_path).expect("failed to load tokenizer.json")`; verified by `loader_rejects_missing_file` + `loader_rejects_malformed_json` + `load_chat_template_categorizes_missing_file_as_internal` + `load_chat_template_categorizes_malformed_json_as_invalid_argument` (issue #734) |
//! | REQ-3 | SHIPPED | impl: `pub fn load_tokenizer / encode / encode_batch / decode / vocab_size / token_to_id / id_to_token in lib.rs` mirror the identically-named methods on `tokenizers::Tokenizer` (upstream crate registry `tokenizers-0.22.2/src/tokenizer/mod.rs`, methods `from_file` / `get_vocab_size` / `token_to_id` / `id_to_token` / `encode` / `decode` / `encode_batch`); non-test consumers: `ferrotorch-llama/examples/llama3_8b.rs`, `llama3_8b_gpu.rs`, `llama3_70b_gpu.rs`, `llm_inference_dump.rs` (each `use ferrotorch_tokenize::{decode, encode, load_tokenizer};` plus the load/encode/decode call sites in their greedy-decode loops) (all `[[example]]` Cargo binary targets, non-test); meta-crate boundary at the meta-crate `pub use ferrotorch_tokenize::*` re-export in `ferrotorch/src/lib.rs` |
//! | REQ-4 | SHIPPED | impl: `pub fn apply_chat_template in lib.rs` registers `raise_exception` + `strftime_now` Jinja helpers and renders via `minijinja::Environment`, mirroring `transformers/tokenization_utils_base.py:1547`; `pub fn apply_chat_template_to_ids in lib.rs` chains it through `encode`; non-test consumer: the meta-crate `pub use ferrotorch_tokenize::*` re-export in `ferrotorch/src/lib.rs` re-exports to `ferrotorch::tokenize::apply_chat_template` (existing pub API across #588 + #734, grandfathered per goal.md S5 R-DEFER-1); verified by `chat_template_renders_simple_two_turn`, `chat_template_appends_generation_prompt_when_requested`, `chat_template_trims_whitespace_in_content`, `chat_template_passes_bos_token`, `chat_template_rejects_invalid_template`, `chat_template_strftime_now_renders_wall_clock_date`, `chat_template_strftime_now_rejects_malformed_format_string`, `chat_template_raise_exception_function_propagates_error` |
//! | REQ-5 | SHIPPED | impl: `#[non_exhaustive] pub struct ChatMessage in lib.rs` with `#[serde(flatten)] extra: BTreeMap<String, serde_json::Value>` and `impl ChatMessage::new in lib.rs`; non-test consumer: the meta-crate `pub use ferrotorch_tokenize::*` re-export in `ferrotorch/src/lib.rs` re-export propagates `ChatMessage` to `ferrotorch::tokenize::ChatMessage` (existing pub API #588, grandfathered); verified by `chat_template_propagates_extra_fields` + `chat_message_roundtrip_through_serde` |
//! | REQ-6 | SHIPPED | impl: `pub fn load_chat_template in lib.rs` reads bytes (Internal on io error), parses JSON (InvalidArgument on parse error), dispatches on `chat_template` field shape (None / String / Array of `{template: string}` / other → InvalidArgument); non-test consumer: the meta-crate `pub use ferrotorch_tokenize::*` re-export in `ferrotorch/src/lib.rs` re-exports to `ferrotorch::tokenize::load_chat_template` (existing pub API #588 + #734, grandfathered); verified by `load_chat_template_extracts_string_field` + `load_chat_template_handles_array_form` + `load_chat_template_returns_none_when_missing` + `conformance_encode_decode::load_chat_template_round_trips_through_disk` |

#![warn(clippy::all, clippy::pedantic)]
#![deny(rust_2018_idioms, missing_debug_implementations)]
// Workspace-wide rustdoc completeness pass is tracked separately; this crate's
// public items are documented but `missing_docs` is not yet enforced workspace-
// wide, so don't gate compilation on it here.
#![allow(missing_docs)]

use std::path::Path;
use std::str::FromStr;

use ferrotorch_core::{FerrotorchError, FerrotorchResult};

pub use tokenizers::Tokenizer;

/// Load a tokenizer from a `HuggingFace` `tokenizer.json` file.
///
/// This accepts any format that `tokenizers::Tokenizer::from_file`
/// supports — which is the full HF tokenizer format including `BPE`,
/// `WordPiece`, Unigram, pre/post processors, and added tokens.
///
/// # Errors
///
/// Returns [`FerrotorchError::Internal`] if the file cannot be read
/// (not found, permission denied, other `std::io::Error`).
///
/// Returns [`FerrotorchError::InvalidArgument`] if the file content is
/// not a valid `HuggingFace` tokenizer JSON (parse error, unknown model
/// type, structurally invalid serialization).
pub fn load_tokenizer(path: impl AsRef<Path>) -> FerrotorchResult<Tokenizer> {
    // I/O and parse errors are categorically different: missing file or
    // permission denied is an environment problem (Internal), while a
    // malformed tokenizer.json is genuinely a bad parameter value
    // (InvalidArgument). Audit issue #734 — split the error categories
    // by doing the I/O ourselves so the typed `std::io::Error` is
    // available at the boundary, then parse via `Tokenizer: FromStr`.
    let path = path.as_ref();
    let content = std::fs::read_to_string(path).map_err(|e| FerrotorchError::Internal {
        message: format!("failed to read tokenizer {}: {e:?}", path.display()),
    })?;
    Tokenizer::from_str(&content).map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("failed to parse tokenizer {}: {e}", path.display()),
    })
}

/// Encode a single text into its token ids.
///
/// `add_special_tokens` controls whether BOS / EOS and other
/// template-defined special tokens are inserted (Llama 3 prepends
/// `<|begin_of_text|>` / `128000` when true).
///
/// # Errors
///
/// Returns [`FerrotorchError::InvalidArgument`] if the tokenizer's underlying
/// model or post-processor rejects the input.
pub fn encode(
    tokenizer: &Tokenizer,
    text: &str,
    add_special_tokens: bool,
) -> FerrotorchResult<Vec<u32>> {
    let encoding = tokenizer.encode(text, add_special_tokens).map_err(|e| {
        FerrotorchError::InvalidArgument {
            message: format!("tokenizer encode failed: {e}"),
        }
    })?;
    Ok(encoding.get_ids().to_vec())
}

/// Encode a batch of texts in parallel.
///
/// # Errors
///
/// Returns [`FerrotorchError::InvalidArgument`] if the underlying tokenizer
/// rejects any text in the batch.
pub fn encode_batch(
    tokenizer: &Tokenizer,
    texts: &[&str],
    add_special_tokens: bool,
) -> FerrotorchResult<Vec<Vec<u32>>> {
    // `tokenizers::Tokenizer::encode_batch` accepts any `E: Into<EncodeInput>`.
    // `&str` satisfies that chain via `InputSequence`, so we pass the slice
    // entries directly and avoid a `Vec<String>` intermediate allocation.
    let encodings = tokenizer
        .encode_batch(texts.to_vec(), add_special_tokens)
        .map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("tokenizer encode_batch failed: {e}"),
        })?;
    Ok(encodings
        .into_iter()
        .map(|e| e.get_ids().to_vec())
        .collect())
}

/// Decode a sequence of token ids back to text.
///
/// `skip_special_tokens` drops BOS / EOS / pad tokens from the output.
///
/// # Errors
///
/// Returns [`FerrotorchError::InvalidArgument`] if the tokenizer's decoder
/// rejects the id sequence (e.g. out-of-range ids on some tokenizer types).
pub fn decode(
    tokenizer: &Tokenizer,
    ids: &[u32],
    skip_special_tokens: bool,
) -> FerrotorchResult<String> {
    tokenizer
        .decode(ids, skip_special_tokens)
        .map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("tokenizer decode failed: {e}"),
        })
}

/// Vocabulary size the tokenizer was trained with, including any
/// special / added tokens (Llama 3: `128_256`).
#[must_use]
pub fn vocab_size(tokenizer: &Tokenizer, with_added_tokens: bool) -> usize {
    tokenizer.get_vocab_size(with_added_tokens)
}

/// Resolve a token string to its numeric id, if present in the vocab
/// (including added/special tokens).
#[must_use]
pub fn token_to_id(tokenizer: &Tokenizer, token: &str) -> Option<u32> {
    tokenizer.token_to_id(token)
}

/// Resolve a token id to its string form.
#[must_use]
pub fn id_to_token(tokenizer: &Tokenizer, id: u32) -> Option<String> {
    tokenizer.id_to_token(id)
}

// ===========================================================================
// Chat-template rendering (#588)
// ===========================================================================

/// One message in a chat-completion conversation.
///
/// Mirrors the `OpenAI` / `HuggingFace` `messages` list structure that
/// `LLM` chat templates expect: `{ role, content }` plus optional structured
/// fields the template may reference (`name`, `tool_calls`, `tool_call_id`).
/// We model the optional fields as a free-form `serde_json::Value` map so
/// the renderer can pass any extra keys straight to Jinja.
///
/// Fields are intentionally `pub` to allow direct construction from parsed
/// JSON or deserialized data. The struct is `#[non_exhaustive]` so that
/// adding new fields (e.g. `tool_call_id`) is a non-breaking change;
/// use [`ChatMessage::new`] as the canonical constructor.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct ChatMessage {
    /// Conventional roles: `"system"`, `"user"`, `"assistant"`, `"tool"`.
    pub role: String,
    /// Text content of the message.
    pub content: String,
    /// Extra fields propagated to the Jinja template (e.g. `name`,
    /// `tool_calls`). Any JSON value is allowed.
    #[serde(flatten)]
    pub extra: std::collections::BTreeMap<String, serde_json::Value>,
}

impl ChatMessage {
    /// Convenience constructor for the common `{ role, content }` case.
    #[must_use]
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            extra: std::collections::BTreeMap::new(),
        }
    }
}

/// Apply a HuggingFace-style Jinja2 chat template to a list of messages.
///
/// Renders to the same string that Python's `tokenizer.apply_chat_template`
/// would produce. The template usually lives in ``tokenizer_config.json``
/// under the key `chat_template` and references:
/// - `messages` — the list of `ChatMessage` records (passed through here)
/// - `add_generation_prompt` — bool, whether to append the assistant turn header
/// - `bos_token`, `eos_token` — special tokens (passed via the eponymous args)
///
/// Pass `bos_token` / `eos_token` as `None` if the template doesn't reference
/// them; otherwise pass the literal token text the template expects (e.g.
/// `"<|begin_of_text|>"` for Llama 3).
///
/// Use [`apply_chat_template_to_ids`] when you also want to tokenize the
/// rendered string in one call.
///
/// # Errors
///
/// Returns [`FerrotorchError::InvalidArgument`] if the template string is not
/// valid Jinja2, if a template variable is missing, or if the template calls
/// `raise_exception(msg)` (propagated as an error with `msg` in the message).
pub fn apply_chat_template(
    template: &str,
    messages: &[ChatMessage],
    add_generation_prompt: bool,
    bos_token: Option<&str>,
    eos_token: Option<&str>,
) -> FerrotorchResult<String> {
    let mut env = minijinja::Environment::new();
    // `raise_exception` is referenced by some HF templates (Mistral et al.).
    // Implement it as a no-arg helper that just panics with a message.
    env.add_function(
        "raise_exception",
        |msg: String| -> Result<String, minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                msg,
            ))
        },
    );
    // `strftime_now` is referenced by some templates (Llama 3.1 system prompt
    // includes e.g. `{{ strftime_now('%d %b %Y') }}`). Format the current local
    // wall-clock time using chrono's strftime-compatible `format_with_items`.
    //
    // `chrono::format::StrftimeItems` parses the format string once and emits
    // `Item::Error` for invalid specifiers; we surface that as a template
    // error rather than silently producing a malformed string.
    env.add_function(
        "strftime_now",
        |fmt: String| -> Result<String, minijinja::Error> {
            use std::fmt::Write as _;
            let items: Vec<chrono::format::Item<'_>> =
                chrono::format::StrftimeItems::new(&fmt).collect();
            if items
                .iter()
                .any(|item| matches!(item, chrono::format::Item::Error))
            {
                return Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    format!("strftime_now: invalid format string {fmt:?}"),
                ));
            }
            let now = chrono::Local::now();
            let mut buf = String::new();
            write!(buf, "{}", now.format_with_items(items.iter())).map_err(|e| {
                minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    format!("strftime_now: formatting failed: {e}"),
                )
            })?;
            Ok(buf)
        },
    );

    env.add_template("chat", template)
        .map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("invalid chat template: {e}"),
        })?;
    let tmpl = env
        .get_template("chat")
        .map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("chat template lookup failed: {e}"),
        })?;

    let context = minijinja::context! {
        messages => messages,
        add_generation_prompt => add_generation_prompt,
        bos_token => bos_token.unwrap_or(""),
        eos_token => eos_token.unwrap_or(""),
    };
    tmpl.render(context)
        .map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("chat template render failed: {e}"),
        })
}

/// Apply a chat template and tokenize the result.
///
/// `add_special_tokens` is forwarded to [`encode`] — usually `false` here
/// because chat templates already embed the BOS / role headers literally.
///
/// Returns the rendered string and the encoded token ids so the caller can
/// log / inspect the prompt.
///
/// # Errors
///
/// Returns [`FerrotorchError::InvalidArgument`] if the template fails to
/// render (see [`apply_chat_template`]) or if the rendered string fails to
/// encode (see [`encode`]).
pub fn apply_chat_template_to_ids(
    tokenizer: &Tokenizer,
    template: &str,
    messages: &[ChatMessage],
    add_generation_prompt: bool,
    bos_token: Option<&str>,
    eos_token: Option<&str>,
    add_special_tokens: bool,
) -> FerrotorchResult<(String, Vec<u32>)> {
    let prompt = apply_chat_template(
        template,
        messages,
        add_generation_prompt,
        bos_token,
        eos_token,
    )?;
    let ids = encode(tokenizer, &prompt, add_special_tokens)?;
    Ok((prompt, ids))
}

/// Read the `chat_template` field out of a ``tokenizer_config.json`` file.
///
/// Returns `None` if the file exists but doesn't define `chat_template`,
/// or an error if the file can't be read or parsed. Some configs ship the
/// template as `chat_template: [{name, template}]` (multiple templates,
/// keyed by name); this loader returns the first one in that case to
/// match what `transformers.AutoTokenizer` does by default.
///
/// # Errors
///
/// Returns [`FerrotorchError::Internal`] if the file cannot be read
/// (not found, permission denied, other `std::io::Error`).
///
/// Returns [`FerrotorchError::InvalidArgument`] if the file content is
/// not valid JSON, or if the `chat_template` field exists but is neither
/// a string nor an array of `{name, template}` objects.
pub fn load_chat_template(
    tokenizer_config_path: impl AsRef<Path>,
) -> FerrotorchResult<Option<String>> {
    // Audit issue #734 — categorize I/O failure (Internal) vs JSON
    // parse / structural failure (InvalidArgument). The original code
    // collapsed every error category into InvalidArgument.
    let path = tokenizer_config_path.as_ref();
    let bytes = std::fs::read(path).map_err(|e| FerrotorchError::Internal {
        message: format!("failed to read tokenizer_config {}: {e:?}", path.display()),
    })?;
    let value: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("failed to parse tokenizer_config {}: {e}", path.display()),
        })?;
    match value.get("chat_template") {
        None => Ok(None),
        Some(serde_json::Value::String(s)) => Ok(Some(s.clone())),
        Some(serde_json::Value::Array(arr)) => {
            // Multi-template form: pick the first entry's `template` field.
            for entry in arr {
                if let Some(t) = entry.get("template").and_then(|v| v.as_str()) {
                    return Ok(Some(t.to_string()));
                }
            }
            Ok(None)
        }
        _ => Err(FerrotorchError::InvalidArgument {
            message: format!(
                "tokenizer_config {} has chat_template of unexpected type",
                path.display()
            ),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Resolve the HF cache directory for a gated model.
    fn hf_cache_snapshot(repo_slug: &str) -> Option<std::path::PathBuf> {
        let home = std::env::var_os("HOME").map(std::path::PathBuf::from)?;
        let base = home
            .join(".cache/huggingface/hub")
            .join(format!("models--{}", repo_slug.replace('/', "--")))
            .join("snapshots");
        std::fs::read_dir(&base)
            .ok()?
            .next()?
            .ok()
            .map(|e| e.path())
    }

    #[test]
    fn loader_rejects_missing_file() {
        // Audit issue #734: missing file is an environment / I/O failure,
        // categorized as `Internal` rather than `InvalidArgument`.
        let r = load_tokenizer("/nonexistent/tokenizer.json");
        match r {
            Err(FerrotorchError::Internal { .. }) => {}
            other => panic!("expected Internal for missing file, got {other:?}"),
        }
    }

    #[test]
    fn loader_rejects_malformed_json() {
        // Audit issue #734: malformed content is a parse failure of a
        // user-supplied parameter, categorized as `InvalidArgument`.
        let tmp = std::env::temp_dir().join("ferrotorch_tok_malformed.json");
        std::fs::write(&tmp, "{ not valid").unwrap();
        let r = load_tokenizer(&tmp);
        let outcome = match r {
            Err(FerrotorchError::InvalidArgument { .. }) => Ok(()),
            other => Err(format!(
                "expected InvalidArgument for malformed JSON, got {other:?}"
            )),
        };
        let _ = std::fs::remove_file(&tmp);
        outcome.unwrap();
    }

    #[test]
    fn load_chat_template_categorizes_missing_file_as_internal() {
        // Audit issue #734: same I/O-vs-parse split for the chat-template
        // loader.
        let r = load_chat_template("/nonexistent/`tokenizer_config.json`");
        match r {
            Err(FerrotorchError::Internal { .. }) => {}
            other => panic!("expected Internal for missing file, got {other:?}"),
        }
    }

    #[test]
    fn load_chat_template_categorizes_malformed_json_as_invalid_argument() {
        let tmp = std::env::temp_dir().join("ferrotorch_tok_chat_cfg_malformed.json");
        std::fs::write(&tmp, "{ not valid json").unwrap();
        let r = load_chat_template(&tmp);
        let outcome = match r {
            Err(FerrotorchError::InvalidArgument { .. }) => Ok(()),
            other => Err(format!(
                "expected InvalidArgument for malformed JSON, got {other:?}"
            )),
        };
        let _ = std::fs::remove_file(&tmp);
        outcome.unwrap();
    }

    /// End-to-end: load the real Llama 3 tokenizer.json from the HF
    /// cache and verify the basic surface works.
    /// Ignored by default so CI without the gated model skips it.
    #[test]
    #[ignore = "requires Meta-Llama-3-8B tokenizer.json in the HF cache"]
    fn llama3_tokenizer_loads_and_round_trips() {
        let snapshot = hf_cache_snapshot("meta-llama/Meta-Llama-3-8B")
            .expect("Meta-Llama-3-8B snapshot missing from HF cache");
        let tok_path = snapshot.join("tokenizer.json");
        let tok = load_tokenizer(&tok_path).unwrap();

        // Llama 3 vocab.
        assert_eq!(vocab_size(&tok, true), 128_256);

        // Special tokens the Llama 3 chat template uses.
        assert_eq!(token_to_id(&tok, "<|begin_of_text|>"), Some(128_000));
        assert_eq!(token_to_id(&tok, "<|end_of_text|>"), Some(128_001));

        // Encode with special tokens — BOS should prepend 128000.
        let ids = encode(&tok, "Hello, world!", true).unwrap();
        assert!(!ids.is_empty());
        assert_eq!(ids[0], 128_000, "BOS not prepended: {ids:?}");

        // Without add_special_tokens, BOS is not prepended.
        let ids_bare = encode(&tok, "Hello, world!", false).unwrap();
        assert_ne!(ids_bare[0], 128_000);

        // Round-trip via decode.
        let text = decode(&tok, &ids_bare, false).unwrap();
        assert!(
            text.contains("Hello") && text.contains("world"),
            "decoded text unexpected: {text:?}"
        );

        // encode_batch returns one vec per input.
        let batch = encode_batch(&tok, &["hi", "bye"], false).unwrap();
        assert_eq!(batch.len(), 2);
        assert!(!batch[0].is_empty());
        assert!(!batch[1].is_empty());
    }

    // -----------------------------------------------------------------------
    // Chat-template rendering (#588)
    // -----------------------------------------------------------------------

    /// A simplified Llama-3-style template for testing — captures the same
    /// shape (per-message header + EOT, optional generation prompt) without
    /// the full template's quirks.
    const SIMPLE_LLAMA3_LIKE: &str = "{% for m in messages %}\
<|start_header_id|>{{ m.role }}<|end_header_id|>\n\n{{ m.content | trim }}<|eot_id|>\
{% endfor %}\
{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}";

    #[test]
    fn chat_template_renders_simple_two_turn() {
        let messages = vec![
            ChatMessage::new("user", "hi"),
            ChatMessage::new("assistant", "hello there"),
        ];
        let s = apply_chat_template(SIMPLE_LLAMA3_LIKE, &messages, false, None, None).unwrap();
        assert!(s.contains("<|start_header_id|>user<|end_header_id|>"));
        assert!(s.contains("hi<|eot_id|>"));
        assert!(s.contains("<|start_header_id|>assistant<|end_header_id|>"));
        assert!(s.contains("hello there<|eot_id|>"));
        // No generation prompt requested → string ends after the last EOT.
        assert!(s.ends_with("<|eot_id|>"));
    }

    #[test]
    fn chat_template_appends_generation_prompt_when_requested() {
        let messages = vec![ChatMessage::new("user", "hi")];
        let s = apply_chat_template(SIMPLE_LLAMA3_LIKE, &messages, true, None, None).unwrap();
        assert!(s.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn chat_template_trims_whitespace_in_content() {
        let messages = vec![ChatMessage::new("user", "   hi   ")];
        let s = apply_chat_template(SIMPLE_LLAMA3_LIKE, &messages, false, None, None).unwrap();
        // The | trim filter strips outer whitespace.
        assert!(s.contains("hi<|eot_id|>"));
        assert!(!s.contains("   hi"));
    }

    #[test]
    fn chat_template_passes_bos_token() {
        let template = "{{ bos_token }}{% for m in messages %}{{ m.content }}{% endfor %}";
        let messages = vec![ChatMessage::new("user", "hi")];
        let s = apply_chat_template(template, &messages, false, Some("<|begin_of_text|>"), None)
            .unwrap();
        assert_eq!(s, "<|begin_of_text|>hi");
    }

    #[test]
    fn chat_template_propagates_extra_fields() {
        let template = "{% for m in messages %}{{ m.role }}:{{ m.name }}{% endfor %}";
        let mut msg = ChatMessage::new("tool", "result");
        msg.extra.insert(
            "name".to_string(),
            serde_json::Value::String("my_tool".to_string()),
        );
        let s = apply_chat_template(template, &[msg], false, None, None).unwrap();
        assert_eq!(s, "tool:my_tool");
    }

    #[test]
    fn chat_template_rejects_invalid_template() {
        let messages = vec![ChatMessage::new("user", "hi")];
        // Unclosed braces.
        let s = apply_chat_template(
            "{% for m in messages %}{{ m.role",
            &messages,
            false,
            None,
            None,
        );
        assert!(s.is_err());
    }

    #[test]
    fn chat_template_strftime_now_renders_wall_clock_date() {
        // Recognized English 3-letter month abbreviations chrono's `%b` emits.
        const MONTHS: &[&str] = &[
            "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ];
        // Llama 3.1's official system template references
        // `{{ strftime_now('%d %b %Y') }}` inside the system header. We
        // assert structural properties (non-empty, 4-digit year, valid
        // 3-letter month abbreviation) so the test is wall-clock-driven
        // without being date-of-execution-fragile.
        let template = "{{ strftime_now('%d %b %Y') }}";
        let s = apply_chat_template(template, &[], false, None, None).unwrap();
        assert!(!s.is_empty(), "strftime_now produced empty string: {s:?}");

        // `%d %b %Y` → e.g. "12 May 2026". Split: day | month | year.
        let parts: Vec<&str> = s.split_whitespace().collect();
        assert_eq!(parts.len(), 3, "expected `DD Mon YYYY`, got {s:?}");

        // Day: 1-2 digits, 01-31.
        let day: u32 = parts[0]
            .parse()
            .unwrap_or_else(|_| panic!("day not numeric in {s:?}"));
        assert!((1..=31).contains(&day), "day out of range: {s:?}");

        assert!(
            MONTHS.contains(&parts[1]),
            "month abbreviation not recognized in {s:?}"
        );

        // Year: 4 digits, plausibly in this century.
        let year: u32 = parts[2]
            .parse()
            .unwrap_or_else(|_| panic!("year not numeric in {s:?}"));
        assert_eq!(parts[2].len(), 4, "expected 4-digit year in {s:?}");
        assert!(
            (2000..=2999).contains(&year),
            "year out of plausible range: {s:?}"
        );
    }

    #[test]
    fn chat_template_strftime_now_rejects_malformed_format_string() {
        // A trailing `%` with no specifier is an `Item::Error` in chrono's
        // strftime parser — the function must surface a template error
        // rather than rendering an empty / garbled string.
        let template = "{{ strftime_now('%') }}";
        let err = apply_chat_template(template, &[], false, None, None).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("strftime_now") || msg.contains("invalid"),
            "expected strftime_now error to surface: got {msg:?}"
        );
    }

    #[test]
    fn chat_template_raise_exception_function_propagates_error() {
        let template = "{% if messages | length == 0 %}\
{{ raise_exception(\"no messages\") }}\
{% else %}{{ messages[0].content }}{% endif %}";
        let empty: Vec<ChatMessage> = Vec::new();
        let err = apply_chat_template(template, &empty, false, None, None).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("no messages") || msg.contains("invalid operation"),
            "expected raise_exception to surface: got {msg:?}"
        );
    }

    #[test]
    fn load_chat_template_extracts_string_field() {
        let tmp = std::env::temp_dir().join("ferrotorch_tok_chat_cfg.json");
        let body = serde_json::json!({
            "chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}"
        });
        std::fs::write(&tmp, serde_json::to_vec_pretty(&body).unwrap()).unwrap();
        let t = load_chat_template(&tmp).unwrap().unwrap();
        assert!(t.contains("messages"));
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn load_chat_template_handles_array_form() {
        let tmp = std::env::temp_dir().join("ferrotorch_tok_chat_cfg_arr.json");
        let body = serde_json::json!({
            "chat_template": [
                {"name": "default", "template": "ARRAY_TEMPLATE"},
                {"name": "tool_use", "template": "OTHER"},
            ]
        });
        std::fs::write(&tmp, serde_json::to_vec_pretty(&body).unwrap()).unwrap();
        let t = load_chat_template(&tmp).unwrap().unwrap();
        assert_eq!(t, "ARRAY_TEMPLATE");
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn load_chat_template_returns_none_when_missing() {
        let tmp = std::env::temp_dir().join("ferrotorch_tok_chat_cfg_no_field.json");
        std::fs::write(&tmp, serde_json::json!({"foo": "bar"}).to_string()).unwrap();
        let t = load_chat_template(&tmp).unwrap();
        assert!(t.is_none());
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn chat_message_roundtrip_through_serde() {
        let mut msg = ChatMessage::new("assistant", "hello");
        msg.extra
            .insert("name".to_string(), serde_json::json!("alice"));
        let s = serde_json::to_string(&msg).unwrap();
        // Both top-level fields and the flattened extras must show up.
        assert!(s.contains("\"role\":\"assistant\""));
        assert!(s.contains("\"content\":\"hello\""));
        assert!(s.contains("\"name\":\"alice\""));
    }
}
