# ferrotorch-hub — `hf_config` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/hub.py
-->

## Summary

`ferrotorch-hub/src/hf_config.rs` parses the HuggingFace transformer
`config.json` published alongside contemporary decoder-only LLM
weights (Llama 3, Mistral, Gemma, Qwen, Falcon, …). It produces a
flat `HfTransformerConfig` struct that downstream model crates
(`ferrotorch-llama`, `ferrotorch-bert`, etc.) map into their own
model-specific configs (`LlamaConfig::from_hf(&hf)`). There is no
direct `torch.hub` counterpart — `torch.hub` doesn't know about HF
configs at all; this module mirrors the responsibility of the
HuggingFace Python `transformers.AutoConfig` dispatch surface but
keeps the deserialiser to a single flat struct because ferrotorch
crates handle the AutoConfig-style polymorphism themselves.

## Requirements

- REQ-1: `pub struct HfTransformerConfig` is the deserialisation
  target for HF `config.json` files. Marked `#[non_exhaustive]`
  because the HF schema gains fields between versions; external
  callers cannot struct-literal-construct (only deserialise) so the
  field list is free to grow.
- REQ-2: Required fields (must be present in the JSON): `hidden_size`,
  `num_hidden_layers`, `num_attention_heads`, `intermediate_size`,
  `max_position_embeddings`, `vocab_size`.
- REQ-3: Optional fields with defaults: `num_key_value_heads`
  (defaults to `None` → resolves to `num_attention_heads` for
  classical MHA), `rms_norm_eps` (defaults to `1e-6` — the HF default
  for the Llama family), `rope_theta` (defaults to `10_000.0` —
  original RoPE), `tie_word_embeddings` (defaults to `false`),
  `hidden_act` (defaults to `"silu"`), `hidden_act_param`
  (defaults to `None`), `torch_dtype` (defaults to `None`),
  `architectures` (defaults to empty `Vec`), `rope_scaling`
  (defaults to `None`), `model_type` (defaults to `None`).
- REQ-4: `pub fn from_json_str(s: &str) -> FerrotorchResult<Self>`
  parses from a JSON string. `pub fn from_file(path: impl AsRef<Path>)
  -> FerrotorchResult<Self>` reads + UTF-8 decodes + parses.
  Both wrap `serde_json` errors and I/O errors in
  `FerrotorchError::InvalidArgument` with the path or parse error
  embedded in the message.
- REQ-5: Unknown JSON fields are silently ignored (HF regularly adds
  new fields to `config.json` — `attention_dropout`, `pretraining_tp`,
  `transformers_version`, `use_cache`, etc. — and older callers must
  still parse).
- REQ-6: Derived accessors: `pub fn num_key_value_heads(&self)
  -> usize` applies the MHA default; `pub fn head_dim(&self) -> usize`
  is `hidden_size / num_attention_heads`; `pub fn is_gqa(&self)
  -> bool` is `num_key_value_heads() < num_attention_heads`.
- REQ-7: `pub fn validate(&self) -> FerrotorchResult<()>` enforces
  the invariants downstream model code relies on: all counts > 0;
  `hidden_size % num_attention_heads == 0`;
  `num_attention_heads % num_key_value_heads == 0`;
  `hidden_act` matches one of the supported names
  (`silu`/`swish`/`gelu`/`relu`/`gelu_new`/`fatrelu`).

## Acceptance Criteria

- [x] AC-1: The published `meta-llama/Meta-Llama-3-8B/config.json`
  parses with all expected fields
  (`parses_llama_3_8b_config`).
- [x] AC-2: Derived accessors return expected values for Llama 3 8B:
  `head_dim() == 128`, `is_gqa() == true`
  (`derived_values_for_llama_3_8b`).
- [x] AC-3: `validate()` passes on Llama 3 8B
  (`validate_passes_on_llama_3_8b`).
- [x] AC-4: Classical MHA (no `num_key_value_heads` in JSON) →
  `num_key_value_heads()` falls back to `num_attention_heads`
  (`num_kv_heads_defaults_to_num_heads_when_absent`).
- [x] AC-5: All field defaults apply when absent
  (`defaults_applied_when_fields_absent`).
- [x] AC-6: Unknown fields ignored
  (`unknown_fields_are_ignored`).
- [x] AC-7: `validate()` rejects non-divisible head splits
  (`validate_rejects_non_divisible_heads`,
  `validate_rejects_non_divisible_kv_heads`).
- [x] AC-8: `validate()` rejects zero counts
  (`validate_rejects_zero_counts`).
- [x] AC-9: `validate()` rejects unsupported activations
  (`validate_rejects_unsupported_activation`).
- [x] AC-10: `from_file` reads a config from disk
  (`from_file_reads_config_json`).
- [x] AC-11: `from_file` reports missing files
  (`from_file_reports_missing_file`).
- [x] AC-12: `from_json_str` reports malformed JSON
  (`from_json_str_reports_bad_json`).
- [x] AC-13: `from_json_str` reports missing required fields
  (`from_json_str_reports_missing_required_field`).

## Architecture

### Struct (REQ-1, REQ-2, REQ-3)

`HfTransformerConfig` is a flat `serde::Deserialize` struct with
`#[non_exhaustive]` so HF schema evolution can add fields in a minor
version. The required fields are bare; each optional field has either
`#[serde(default)]` (None or empty Vec) or
`#[serde(default = "default_<name>")]` pointing at a free fn returning
the canonical default (`1e-6` for RMSNorm eps; `10_000.0` for RoPE
theta; `"silu"` for hidden act).

The struct derives `Debug` and `Clone` so downstream model configs
can take it by reference for `from_hf` and clone it when needed.

### Parsing entry points (REQ-4)

`from_json_str(s)` wraps `serde_json::from_str(s)` and re-maps the
error to `FerrotorchError::InvalidArgument` with the parser's message
embedded.

`from_file(path)` reads the bytes via `std::fs::read`, UTF-8 decodes
via `std::str::from_utf8`, then calls `from_json_str`. Three error
arms (I/O, UTF-8, parse) all map to `InvalidArgument` with the path
in the message for diagnosis.

### Unknown-field handling (REQ-5)

Serde's default `deny_unknown_fields = false` behaviour is what we
want — every HF version adds new fields and a parser that rejects
unknown keys would fail on every fresh config. The behaviour is
documented in the struct comment and exercised by
`unknown_fields_are_ignored`.

### Derived accessors (REQ-6)

- `num_key_value_heads(&self)`:
  `self.num_key_value_heads.unwrap_or(self.num_attention_heads)`.
  Applies the classical-MHA default at call time so the on-disk
  representation can faithfully preserve the JSON's `Option<usize>`
  shape.
- `head_dim(&self)`: `self.hidden_size / self.num_attention_heads`.
  Integer division — caller is responsible for checking divisibility
  via `validate()` before computing the head dim.
- `is_gqa(&self)`: `self.num_key_value_heads() < self.num_attention_heads`.
  Returns `false` for MHA (kv == q) and for atypical multi-query
  attention where kv == 1 still happens via the same predicate.

### `validate()` (REQ-7)

The invariant gate. Five arms:

1. Any of `hidden_size`/`num_attention_heads`/`num_hidden_layers`/
   `intermediate_size`/`vocab_size` is zero → `InvalidArgument`.
2. `hidden_size % num_attention_heads != 0` → `InvalidArgument`.
3. Resolved `kv_heads == 0` → `InvalidArgument`.
4. `num_attention_heads % kv_heads != 0` → `InvalidArgument`.
5. `hidden_act` not in {silu, swish, gelu, relu, gelu_new, fatrelu}
   → `InvalidArgument`.

All five guard the downstream `LlamaConfig::from_hf` from feeding
non-sensical shapes into the transformer block constructors.

### Default-value free functions

`fn default_rms_norm_eps()`, `fn default_rope_theta()`,
`fn default_hidden_act()` are private free functions returning the
canonical defaults. They are pointed at by `#[serde(default = "...")]`
attributes; serde requires a free fn (not a closure) for the default
factory.

### Non-test production consumers

- `pub use hf_config::HfTransformerConfig` in `lib.rs` is the
  flattened entry point.
- `ferrotorch-llama/src/config.rs` (line 8: `use ferrotorch_hub::
  HfTransformerConfig;`) calls `HfTransformerConfig::from_json_str`
  + `HfTransformerConfig::from_file` in its `LlamaConfig::from_hf(hf:
  &HfTransformerConfig) -> FerrotorchResult<Self>` constructor.
  `from_hf` calls `hf.validate()?` first, then dispatches `hf.hidden_act`
  to a `LlamaActivation` enum, and populates every `LlamaConfig` field
  from the `HfTransformerConfig` accessor surface.
- `ferrotorch-llama/examples/{llama3_8b, llama3_8b_gpu, llama3_70b_gpu,
  prosparse_7b_gpu, llm_inference_dump}.rs` all `use
  ferrotorch_hub::HfTransformerConfig;` and call
  `HfTransformerConfig::from_file(cache.path("config.json"))` to
  feed the parsed config into `LlamaConfig::from_hf`.

## Parity contract

`parity_ops = []`. The contract is the on-disk HF `config.json`
format, which is an external wire format (R-DEV-3: match upstream).
The canonical reference is the file published by HF for each model;
the test fixture is the verbatim `meta-llama/Meta-Llama-3-8B/config.json`.

Edge cases:

- Field absent + has a `#[serde(default)]` → uses the documented
  default.
- Field absent + no default → JSON parse error (e.g. missing
  `hidden_size`).
- Unknown field present → silently ignored (HF schema evolution).
- `num_key_value_heads` absent → resolves to `num_attention_heads`
  at accessor time (classical MHA).
- `hidden_act == "fatrelu"` with `hidden_act_param: Some(0.005)` →
  ProSparse-style FATReLU with non-standard threshold; downstream
  `LlamaConfig::from_hf` extracts the threshold from
  `hf.hidden_act_param`.
- `torch_dtype: "bfloat16"` → carried as `Some("bfloat16")`;
  downstream loaders use it to pick the right cast.
- `rope_scaling: Some(serde_json::Value::Object(...))` → carried
  verbatim; model-specific wrappers parse the structure they need
  (NTK / linear / dynamic scaling).

## Verification

Tests in `mod tests in hf_config.rs` (12 tests):

- Real-world parse: `parses_llama_3_8b_config` (verbatim Llama 3 8B
  JSON), `derived_values_for_llama_3_8b`,
  `validate_passes_on_llama_3_8b`.
- Defaults: `num_kv_heads_defaults_to_num_heads_when_absent`,
  `defaults_applied_when_fields_absent`.
- Unknown fields: `unknown_fields_are_ignored`.
- `validate()` rejections: `validate_rejects_non_divisible_heads`,
  `validate_rejects_non_divisible_kv_heads`,
  `validate_rejects_zero_counts`,
  `validate_rejects_unsupported_activation`.
- I/O entry points: `from_file_reads_config_json`,
  `from_file_reports_missing_file`,
  `from_json_str_reports_bad_json`,
  `from_json_str_reports_missing_required_field`.

Smoke command:

```bash
cargo test -p ferrotorch-hub --lib hf_config:: 2>&1 | tail -3
```

Expected: 14 passed.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `#[non_exhaustive] pub struct HfTransformerConfig` + `#[derive(Debug, Clone, Deserialize)]` in `hf_config.rs`; non-test consumer: `ferrotorch-llama/src/config.rs` imports `HfTransformerConfig` (line 8) and `LlamaConfig::from_hf(hf: &HfTransformerConfig)` at line 126 takes a reference to the struct, dispatching its fields to build `LlamaConfig`. |
| REQ-2 | SHIPPED | impl: required fields (`hidden_size`, `num_hidden_layers`, `num_attention_heads`, `intermediate_size`, `max_position_embeddings`, `vocab_size`) declared without `#[serde(default)]` in `hf_config.rs`; non-test consumer: `ferrotorch-llama/src/config.rs::LlamaConfig::from_hf` reads `hf.vocab_size`, `hf.hidden_size`, `hf.intermediate_size`, `hf.num_hidden_layers`, `hf.num_attention_heads`, `hf.num_key_value_heads()` directly into the Llama config (lines 142-147). |
| REQ-3 | SHIPPED | impl: every optional field marked `#[serde(default)]` or `#[serde(default = "default_<name>")]` in `hf_config.rs` plus the three default fns (`default_rms_norm_eps`, `default_rope_theta`, `default_hidden_act`); non-test consumer: `ferrotorch-llama/src/config.rs::LlamaConfig::from_hf` reads `hf.rms_norm_eps`, `hf.rope_theta`, `hf.tie_word_embeddings`, `hf.hidden_act`, `hf.hidden_act_param` — every accessor depends on the default surface so a real-world `config.json` with absent fields still produces a valid `LlamaConfig`. |
| REQ-4 | SHIPPED | impl: `pub fn from_json_str` + `pub fn from_file` (`impl HfTransformerConfig`) in `hf_config.rs` with the I/O / UTF-8 / parse error arms; non-test consumer: `ferrotorch-llama/examples/llama3_8b.rs`, `llama3_8b_gpu.rs`, `llama3_70b_gpu.rs`, `prosparse_7b_gpu.rs`, `llm_inference_dump.rs` all call `HfTransformerConfig::from_file(cache.path("config.json"))` after `hf_download_model` returns. |
| REQ-5 | SHIPPED | impl: serde default `deny_unknown_fields = false` (no struct-level annotation overriding it) on `HfTransformerConfig` in `hf_config.rs`; non-test consumer: every real-world HF config carries fields ferrotorch doesn't model (`attention_bias`, `bos_token_id`, `pretraining_tp`, `transformers_version`, `use_cache`, …); the parse-then-`LlamaConfig::from_hf` chain in `ferrotorch-llama/src/config.rs` only succeeds because the unknown-field tolerance is the default. |
| REQ-6 | SHIPPED | impl: `pub fn num_key_value_heads`, `pub fn head_dim`, `pub fn is_gqa` (`impl HfTransformerConfig`) in `hf_config.rs`; non-test consumer: `ferrotorch-llama/src/config.rs::LlamaConfig::from_hf` calls `hf.num_key_value_heads()` (line 147) and downstream Llama code uses `hf.head_dim()` and `hf.is_gqa()` to pick attention kernels. |
| REQ-7 | SHIPPED | impl: `pub fn validate` (`impl HfTransformerConfig`) in `hf_config.rs` with the five guard arms; non-test consumer: `ferrotorch-llama/src/config.rs::LlamaConfig::from_hf` calls `hf.validate()?` as its very first step (line 127) before reading any field — every production model load fires this gate. |
