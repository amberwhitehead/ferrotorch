# ferrotorch-llama — `config` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - HuggingFace transformers/models/llama/configuration_llama.py
    (LlamaConfig: vocab_size, hidden_size, intermediate_size,
    num_hidden_layers, num_attention_heads, num_key_value_heads,
    rms_norm_eps, rope_theta, max_position_embeddings, hidden_act,
    tie_word_embeddings)
  - Meta-Llama-3-8B config.json (canonical 8B preset values)
  - Meta-Llama-3.3-70B-Instruct config.json (canonical 70B preset
    values)
-->

## Summary

`ferrotorch-llama/src/config.rs` defines `LlamaConfig` — a flat,
`Copy` hyperparameters struct mirroring the relevant subset of
HuggingFace's `LlamaConfig` — plus the `LlamaActivation` enum
covering the three FFN activations real Llama checkpoints use
(SiLU/SwiGLU, ReluLLaMA, ProSparse FATReLU). Canonical presets
(`llama3_8b`, `llama2_7b`, `llama3_3_70b_instruct`, `prosparse_7b`)
provide ready-made configurations matching the corresponding
`config.json` files on the Hub.

## Requirements

- REQ-1: `pub struct LlamaConfig` carries every hyperparameter the
  rest of the crate needs to construct the decoder stack:
  `vocab_size`, `hidden_size`, `intermediate_size`,
  `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`,
  `rms_norm_eps`, `rope_theta`, `max_position_embeddings`,
  `tie_word_embeddings`, `hidden_act`. The struct derives
  `Debug, Clone, Copy, PartialEq` so it can be passed by value to
  per-layer constructors and stored frozen on the model.
- REQ-2: `pub enum LlamaActivation { Silu, FatRelu(f64), Relu }`
  covers the three activations real Llama-family checkpoints use.
  FATReLU carries its threshold (default 0.01 per the ProSparse
  paper) inside the variant.
- REQ-3: Canonical presets `LlamaConfig::llama3_8b`,
  `LlamaConfig::llama2_7b`, `LlamaConfig::prosparse_7b`,
  `LlamaConfig::llama3_3_70b_instruct` produce
  byte-for-byte-matching configurations against the corresponding
  HuggingFace `config.json` values.
- REQ-4: `LlamaConfig::from_hf(hf: &HfTransformerConfig)` validates
  the parsed HF config, maps `hidden_act` strings (`"silu"`,
  `"swish"`, `"relu"`, `"fatrelu"`) onto the enum, and rejects any
  other activation name with `FerrotorchError::InvalidArgument`.
- REQ-5: `LlamaConfig::validate` enforces the structural invariants
  the rest of the crate relies on: every size positive,
  `hidden_size % num_attention_heads == 0`, `num_attention_heads %
  num_key_value_heads == 0`. Returns
  `FerrotorchError::InvalidArgument` with a descriptive message on
  any violation.
- REQ-6: Convenience accessors `head_dim()` (=
  `hidden_size / num_attention_heads`) and `kv_group_size()` (=
  `num_attention_heads / num_key_value_heads`) so consumers don't
  re-derive these constants and risk mismatching the invariants.

## Acceptance Criteria

- [x] AC-1: `LlamaConfig::llama3_8b().validate()` succeeds; the
  resulting `head_dim()` is 128 and `kv_group_size()` is 4.
- [x] AC-2: `LlamaConfig::llama2_7b().validate()` succeeds;
  `kv_group_size()` is 1 (Llama 2 uses MHA, not GQA).
- [x] AC-3: `LlamaConfig::prosparse_7b()` has
  `hidden_act == LlamaActivation::FatRelu(0.01)`.
- [x] AC-4: `LlamaConfig::llama3_3_70b_instruct().validate()`
  succeeds; `num_hidden_layers == 80`, `hidden_size == 8192`,
  `num_attention_heads == 64`, `num_key_value_heads == 8`,
  `head_dim() == 128`.
- [x] AC-5: `LlamaConfig::from_hf` round-trips an authentic
  `config.json` byte sequence into the matching preset.
- [x] AC-6: `validate()` rejects `hidden_size = 0` and
  `num_attention_heads = 7` (non-divisor).

## Architecture

`pub struct LlamaConfig` in `config.rs` carries the 11 hyperparameters
the decoder stack consumes. `Copy` is intentional — every
sub-module's constructor takes `cfg: &LlamaConfig` (for read access)
or `cfg: LlamaConfig` (for storage), and `Copy` removes the
ergonomic friction without runtime cost.

`pub enum LlamaActivation` carries the activation variant on the
config struct; the MLP block (`LlamaMLP::activate` in `mlp.rs`)
matches on it to dispatch the activation function. FATReLU's
threshold is data on the variant, not a separate field on
`LlamaConfig`, so the activation contract is self-contained.

`pub fn LlamaConfig::from_hf` in `config.rs` builds a `LlamaConfig`
from `ferrotorch_hub::HfTransformerConfig` (the parsed HF
`config.json`). It calls `hf.validate()` first to surface structural
problems with the JSON itself, then maps the lowercase
`hidden_act` string onto the enum. Unknown strings produce
`InvalidArgument`. The HF config stores `rms_norm_eps` /
`rope_theta` as f32; we widen to f64 for the model layer's
numerical-stability needs.

`pub fn LlamaConfig::validate` in `config.rs` is the structural
gate every constructor calls before allocating tensors: it rejects
zero sizes and non-divisible head counts. The two divisibility
checks (`hidden_size % num_attention_heads`,
`num_attention_heads % num_key_value_heads`) are what `head_dim()`
and `kv_group_size()` depend on; calling `validate` before either
accessor is the contract.

### Non-test production consumers

- `LlamaConfig::from_hf` is called at
  `ferrotorch-llama/examples/llama3_8b.rs:76`, at
  `ferrotorch-llama/examples/llama3_8b_gpu.rs:66`, at
  `ferrotorch-llama/examples/prosparse_7b_gpu.rs:47`, and at
  `ferrotorch-llama/examples/llm_inference_dump.rs:311`.
- `LlamaConfig::validate` is called at the top of every
  sub-module constructor: `LlamaModel::new` in `model.rs`,
  `LlamaForCausalLM::new` in `model.rs`, `LlamaDecoderLayer::new` in
  `layer.rs`, `LlamaAttention::new` in `attention.rs`,
  `LlamaMLP::new` in `mlp.rs`, `LlamaGpuInferencer::new` in `gpu.rs`.
- `LlamaActivation` is matched in `LlamaMLP::activate` in `mlp.rs`
  and in the GPU MLP block at `gpu.rs` (the `forward_core` match
  on `cfg.hidden_act` selecting between `gpu_silu_bf16`,
  `gpu_relu_bf16`, `gpu_fatrelu_bf16`).
- The `config` field is held on `LlamaForCausalLM` in `model.rs`,
  on `LlamaGpuInferencer` in `gpu.rs`, and threaded into every
  layer / mlp / attention block constructor.

## Parity contract

`parity_ops = []`. `LlamaConfig` is a hyperparameters struct, not an
op. The parity contract it carries is structural:

- Field names match HF's `LlamaConfig` Python attribute names
  byte-for-byte (`num_key_value_heads`, `rms_norm_eps`,
  `rope_theta`, `tie_word_embeddings`, `max_position_embeddings`,
  `hidden_act`). A caller can read an HF `config.json`, parse it
  with `ferrotorch_hub`, and reach a `LlamaConfig` without rename.
- The preset values for `llama3_8b` (`vocab_size = 128_256`,
  `hidden_size = 4096`, `intermediate_size = 14_336`,
  `num_hidden_layers = 32`, `num_attention_heads = 32`,
  `num_key_value_heads = 8`, `rope_theta = 500_000.0`,
  `max_position_embeddings = 8192`) and for
  `llama3_3_70b_instruct` (`hidden_size = 8192`,
  `intermediate_size = 28_672`, `num_hidden_layers = 80`,
  `num_attention_heads = 64`, `num_key_value_heads = 8`,
  `rope_theta = 500_000.0`, `max_position_embeddings = 131_072`)
  match the corresponding HF `config.json` files exactly.
- The `hidden_act` string mapping mirrors HF
  `ACT2FN[config.hidden_act]` semantics: `"silu" | "swish" → SiLU`,
  `"relu" → ReLU`, `"fatrelu" → FATReLU(threshold)`.

## Verification

Tests in `mod tests in config.rs`:

- `llama3_8b_is_valid`
- `llama2_7b_is_valid`
- `prosparse_7b_is_valid`
- `llama3_3_70b_instruct_is_valid`
- `from_hf_round_trips_70b_config`
- `from_hf_round_trips_llama3_config`
- `validate_rejects_zero_fields`
- `validate_rejects_non_divisible_heads`
- `validate_rejects_non_divisible_kv_heads`

No parity-sweep ops. Smoke command:

```bash
cargo test -p ferrotorch-llama --lib config:: 2>&1 | tail -3
```

Expected: all tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct LlamaConfig` definition in `config.rs`; non-test consumer: stored as the `config` field on `LlamaForCausalLM` in `model.rs` and on `LlamaGpuInferencer` in `gpu.rs`; consumed by every constructor in the crate. |
| REQ-2 | SHIPPED | impl: `pub enum LlamaActivation` in `config.rs`; non-test consumer: matched in `LlamaMLP::activate` in `mlp.rs` and in the GPU activation dispatch in `gpu.rs` (`forward_core` match on `cfg.hidden_act`). |
| REQ-3 | SHIPPED | impl: `LlamaConfig::llama3_8b` / `llama2_7b` / `prosparse_7b` / `llama3_3_70b_instruct` in `config.rs`; non-test consumer: examples use these as fallbacks when no HF config is present (`llama3_8b` shape used in `ferrotorch-llama/examples/llama3_8b.rs:76` after `from_hf`). |
| REQ-4 | SHIPPED | impl: `LlamaConfig::from_hf` in `config.rs`; non-test consumer: `from_hf in ferrotorch-llama/examples/llama3_8b.rs`, `from_hf in llama3_8b_gpu.rs`, `from_hf in prosparse_7b_gpu.rs`, `from_hf in llm_inference_dump.rs`. |
| REQ-5 | SHIPPED | impl: `LlamaConfig::validate` in `config.rs`; non-test consumer: called at the top of every sub-module constructor — `LlamaModel::new`, `LlamaForCausalLM::new`, `LlamaDecoderLayer::new`, `LlamaAttention::new`, `LlamaMLP::new` (all in their respective `.rs` files), and `LlamaGpuInferencer::new` in `gpu.rs`. |
| REQ-6 | SHIPPED | impl: `LlamaConfig::head_dim` and `LlamaConfig::kv_group_size` in `config.rs`; non-test consumer: `head_dim` invoked in `LlamaAttention::new` in `attention.rs` to size the K/V projections; `kv_group_size` invoked in the GPU forward at `gpu.rs` (`forward_core` computing `group = cfg.kv_group_size()`). |
