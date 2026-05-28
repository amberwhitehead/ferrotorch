# ferrotorch-llama — `gguf_remap` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - llama.cpp GGUF tensor naming convention
    (token_embd.weight, output_norm.weight, output.weight,
    blk.{i}.attn_q.weight, blk.{i}.attn_k.weight,
    blk.{i}.attn_v.weight, blk.{i}.attn_output.weight,
    blk.{i}.attn_norm.weight, blk.{i}.ffn_norm.weight,
    blk.{i}.ffn_gate.weight, blk.{i}.ffn_up.weight,
    blk.{i}.ffn_down.weight)
  - HuggingFace transformers/models/llama/modeling_llama.py (target
    naming convention: model.embed_tokens.weight,
    model.layers.{i}.self_attn.q_proj.weight, ...)
-->

## Summary

`ferrotorch-llama/src/gguf_remap.rs` translates GGUF (llama.cpp)
tensor names into the HuggingFace transformers naming convention
the rest of the crate expects. After a GGUF file is dequantized
via `ferrotorch_serialize::gguf::load_gguf_state_dict`, the
resulting `StateDict<f32>` is keyed in the GGUF convention; this
module remaps it onto HF keys so `LlamaForCausalLM::load_hf_state_dict`
accepts it unchanged. This is the single-host fit path for
70B-class checkpoints that ship only in GGUF.

## Requirements

- REQ-1: `pub fn gguf_key_to_hf(gguf_name)` maps a single GGUF
  tensor name to its HF equivalent for Llama-architecture models.
  Returns `None` for names that don't match any known Llama
  pattern (caller decides whether to drop or pass through).
- REQ-2: Top-level keys: `token_embd.weight` →
  `model.embed_tokens.weight`, `output_norm.weight` →
  `model.norm.weight`, `output.weight` → `lm_head.weight`.
- REQ-3: Per-layer keys: parse `blk.{i}.<suffix>` and map each
  suffix:
  - `attn_norm.weight` → `input_layernorm.weight`
  - `attn_q.weight` → `self_attn.q_proj.weight`
  - `attn_k.weight` → `self_attn.k_proj.weight`
  - `attn_v.weight` → `self_attn.v_proj.weight`
  - `attn_output.weight` → `self_attn.o_proj.weight`
  - `ffn_norm.weight` → `post_attention_layernorm.weight`
  - `ffn_gate.weight` → `mlp.gate_proj.weight`
  - `ffn_up.weight` → `mlp.up_proj.weight`
  - `ffn_down.weight` → `mlp.down_proj.weight`
- REQ-4: Malformed layer indices (e.g. `blk.x.attn_q.weight`,
  `blk..attn_q.weight`) return `None` (parser failure).
- REQ-5: `pub fn gguf_to_hf_state_dict(state, strict)` walks the
  whole state dict and produces a new HF-keyed `StateDict<T>`. In
  non-strict mode, unrecognised keys are dropped (matching the HF
  semantic that state dicts can carry unrelated metadata). In
  strict mode, unrecognised keys produce `InvalidArgument`.
- REQ-6: The function is generic over `T: Float + Clone` so it
  works for both `f32` (the common dequantized GGUF dtype) and
  `bf16` (for direct upload to `LlamaGpuInferencer`).

## Acceptance Criteria

- [x] AC-1: All three top-level keys (`token_embd.weight`,
  `output_norm.weight`, `output.weight`) map correctly.
- [x] AC-2: Every per-layer suffix maps correctly for arbitrary
  layer indices.
- [x] AC-3: Unknown top-level keys (e.g. `rope_freqs.weight`) and
  unknown suffixes (e.g. `blk.0.unknown.weight`) return `None`.
- [x] AC-4: Malformed indices (`blk.x.attn_q.weight`,
  `blk..attn_q.weight`) return `None`.
- [x] AC-5: A full 70B keyset (80 layers × 9 suffixes + 3
  top-level = 723 tensors) translates completely in strict mode
  without dropping anything.
- [x] AC-6: Non-strict mode drops `rope_freqs.weight` silently.
- [x] AC-7: Strict mode produces `InvalidArgument` on
  `rope_freqs.weight`.

## Architecture

`pub fn gguf_key_to_hf` in `gguf_remap.rs` is a flat `match`
covering the three top-level keys with explicit returns and
delegating per-layer keys to `fn translate_layer_key`.

`fn translate_layer_key` in `gguf_remap.rs` strips the `blk.`
prefix, finds the first `.` to split off the layer index, parses
the index as `usize`, and matches the remaining suffix against the
nine known patterns. Any parse failure or unknown suffix returns
`None`. The output format is
`model.layers.{layer_idx}.{mapped_suffix}`.

`pub fn gguf_to_hf_state_dict<T: Float + Clone>` in
`gguf_remap.rs` walks the input state dict and, for each key:

- If `gguf_key_to_hf` returns `Some(hf_key)`, insert
  `(hf_key, tensor.clone())` into the output.
- If `None` and `strict == true`, return
  `FerrotorchError::InvalidArgument` with the unknown key in the
  message.
- If `None` and `strict == false`, continue (drop silently).

The function pre-allocates the output `HashMap` with
`state.len()` capacity so the common case avoids rehashing.

### Non-test production consumers

- `pub use gguf_remap::{gguf_key_to_hf, gguf_to_hf_state_dict}` at
  `ferrotorch-llama/src/lib.rs:170`.
- The `ferrotorch::llama` umbrella re-export at
  `ferrotorch/src/lib.rs:155` exposes both functions as
  `ferrotorch::llama::gguf_key_to_hf` / `gguf_to_hf_state_dict`
  for any downstream user of the meta-crate.

The current crate ships no in-crate `.rs` file that invokes
`gguf_to_hf_state_dict` as a non-test caller (the consumers are
the example drivers and downstream applications that bridge
`ferrotorch_serialize::gguf::load_gguf_state_dict` into
`LlamaForCausalLM::load_hf_state_dict`). The re-export at
`ferrotorch::llama::gguf_to_hf_state_dict` is the public API
surface; per goal.md R-DEFER-1 / S5, existing pub API surface in
prior commits is grandfathered as the boundary method that IS the
public API.

## Parity contract

`parity_ops = []`. The remap is a pure name-translation function;
its correctness criterion is "for every name a real GGUF Llama
checkpoint exports, produce the matching HF name". The 70B
keyset test (`full_70b_layer_set_translates_completely`) exercises
the cartesian product of {80 layers} × {9 suffixes} plus the 3
top-level keys to confirm complete coverage.

Behavioral contract:

- **HF-side target shape**: matches `LlamaForCausalLM`'s
  `named_parameters` output exactly. The whole point of the
  remap is to produce keys that `load_hf_state_dict` accepts in
  strict mode without further rewriting.
- **GGUF-side source shape**: matches llama.cpp's
  `gguf-py/gguf/constants.py` `TENSOR_NAMES[MODEL_TENSOR.X]`
  values for the LLAMA arch.
- **Unknown keys dropped by default**: matches the HF
  load-state-dict default that tolerates extra keys (e.g.
  `rope_freqs.weight` is a llama.cpp-specific buffer that doesn't
  exist in the HF model). Strict mode is opt-in.

## Verification

Tests in `mod tests in gguf_remap.rs`:

- `embedding_and_norms`
- `per_layer_attention_projections`
- `per_layer_norms_and_mlp`
- `unknown_keys_return_none`
- `malformed_layer_indices_return_none`
- `state_dict_translation_drops_unknown_by_default`
- `strict_mode_rejects_unknown_keys`
- `full_70b_layer_set_translates_completely`

No parity-sweep ops. Smoke command:

```bash
cargo test -p ferrotorch-llama --lib gguf_remap:: 2>&1 | tail -3
```

Expected: all tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn gguf_key_to_hf` in `gguf_remap.rs`; non-test consumer: re-exported at `gguf_remap in ferrotorch-llama/src/lib.rs`, reachable from the meta-crate as `ferrotorch::llama::gguf_key_to_hf` via the umbrella re-export at `ferrotorch/src/lib.rs`. The function is also invoked internally by `gguf_to_hf_state_dict` in `gguf_remap.rs` — a non-test production call site. |
| REQ-2 | SHIPPED | impl: the three explicit `match` arms at the top of `gguf_key_to_hf` in `gguf_remap.rs`; non-test consumer: same re-export surface as REQ-1; `gguf_to_hf_state_dict` calls `gguf_key_to_hf` on every input key. |
| REQ-3 | SHIPPED | impl: the nine-arm `match suffix` block in `fn translate_layer_key` in `gguf_remap.rs`; non-test consumer: same chain as REQ-1/2 (`gguf_to_hf_state_dict` → `gguf_key_to_hf` → `translate_layer_key`). |
| REQ-4 | SHIPPED | impl: the `parse::<usize>().ok()?` early return in `fn translate_layer_key` in `gguf_remap.rs`; non-test consumer: same chain — malformed inputs surface as the `None` branch of `gguf_to_hf_state_dict`. |
| REQ-5 | SHIPPED | impl: `pub fn gguf_to_hf_state_dict` (with the strict-mode branch returning `InvalidArgument`) in `gguf_remap.rs`; non-test consumer: re-exported at `ferrotorch-llama/src/lib.rs:170`, reachable from the meta-crate. |
| REQ-6 | SHIPPED | impl: `pub fn gguf_to_hf_state_dict<T: Float + Clone>` generic bound in `gguf_remap.rs`; non-test consumer: same re-export surface — downstream consumers that load GGUF as `f32` (the common dequantized dtype) instantiate `gguf_to_hf_state_dict::<f32>`; consumers feeding the GPU bf16 path instantiate `gguf_to_hf_state_dict::<bf16>`. |
