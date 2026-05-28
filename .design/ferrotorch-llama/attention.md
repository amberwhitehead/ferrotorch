# ferrotorch-llama — `attention` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - HuggingFace transformers/models/llama/modeling_llama.py
    (LlamaAttention:197-265, rotate_half:109-113, apply_rotary_pos_emb:116-140,
    repeat_kv:159-168, eager_attention_forward:171-194)
-->

## Summary

`ferrotorch-llama/src/attention.rs` ships `LlamaAttention`, the
grouped-query attention block with RoPE applied to Q/K
post-projection, pre-attention. It composes the four linear
projections (Q/K/V/O), a half-rotation `RotaryPositionEmbedding`
keyed off the config, and the `standard_attention` primitive from
`ferrotorch-nn`. Both full-prefix `Module::forward` (any seq) and
the per-token `forward_with_cache` paths are implemented.

## Requirements

- REQ-1: `pub struct LlamaAttention<T: Float>` carries four `Linear<T>`
  projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, all
  `bias = false`) plus a `rope: RotaryPositionEmbedding<T>` table
  precomputed from `cfg.max_position_embeddings` and `cfg.rope_theta`.
- REQ-2: Grouped-query attention support: K and V project to
  `num_kv_heads * head_dim` (`< hidden_size` when GQA is active).
  Q projects to the full `hidden_size = num_heads * head_dim`.
- REQ-3: `Module::forward` runs the standard attention:
  `Q,K,V = proj(input)` → `RoPE(Q), RoPE(K)` → `K,V = repeat_kv(K),
  repeat_kv(V)` to match Q's head count → `standard_attention(Q, K, V,
  causal=true)` → `o_proj(merge_heads(ctx))`. Input shape
  `[1, seq, hidden]` is required (batch=1 only). Any other shape
  yields `InvalidArgument` / `ShapeMismatch`.
- REQ-4: `forward_with_cache(input, cache, seq_offset)` is the
  incremental path: input `[1, 1, hidden]`. Q gets RoPE applied at
  `seq_offset`; K gets RoPE applied at `seq_offset` and appended to
  the cache (post-RoPE keys are cached). V is appended raw
  (positional-invariant). The full cache is then expanded via
  `repeat_kv`, and `standard_attention` runs with
  `causal = false` (`N_q = 1` against `N_k = seq_offset + 1` — there
  is no future to mask).
- REQ-5: Cache shape contract: post-RoPE K and raw V are stored as
  `[num_kv_heads, seq, head_dim]`. `LayerKvCache::from_single_token`
  seeds the cache on first call; `LayerKvCache::append` extends it
  thereafter.
- REQ-6: First-token discipline: `cache = None` requires
  `seq_offset = 0`; otherwise `InvalidArgument` (the only correct
  starting state).
- REQ-7: `named_parameters` produces HF-compatible keys:
  `q_proj.weight`, `k_proj.weight`, `v_proj.weight`, `o_proj.weight`.

## Acceptance Criteria

- [x] AC-1: `LlamaAttention::<f32>::new(&cfg)` constructs for every
  preset configuration.
- [x] AC-2: Full-prefix `forward` returns `[1, S, hidden]` for input
  `[1, S, hidden]` and produces finite values (exercised via the
  model-level `tiny_model_forward_from_ids_produces_correct_shape`).
- [x] AC-3: `forward_with_cache` on `[1, 1, hidden]` returns the
  same-shape output and a `LayerKvCache` with `seq_len = 1` (first
  token) / `seq_len = old + 1` (subsequent).
- [x] AC-4: Full-prefix and per-token paths agree numerically to
  `< 1e-4` on the tiny model
  (`forward_one_with_cache_matches_full_prefix_forward` in
  `mod tests in model.rs`).
- [x] AC-5: GQA broadcast: when `num_attention_heads = 32` and
  `num_key_value_heads = 8`, the `repeat_kv` expansion produces the
  expected `[32, S, d]` shape internally before `standard_attention`.
- [x] AC-6: HF-keyed state-dict round-trip on `q_proj.weight`,
  `k_proj.weight`, `v_proj.weight`, `o_proj.weight`.

## Architecture

`pub struct LlamaAttention<T: Float>` in `attention.rs` carries the
four projections, the RoPE table (with
`RoPEConvention::HalfRotation` to match Meta/HF's half-rotation
convention — `rotate_half` at HF `modeling_llama.py:109`), and the
cached head dimensions for forward-time math.

`Module::forward` in `attention.rs`:

1. Shape-validate: reject anything but `[1, S, hidden]`.
2. Project Q (`[1, S, H*d]`), K/V (`[1, S, Hkv*d]`).
3. Squeeze batch: reshape `[1, S, x]` → `[S, x]`.
4. Split heads: `[S, H*d]` → `[H, S, d]` via
   `reshape_to_heads` for each of Q/K/V.
5. RoPE Q and K at position offset 0 (full-prefix).
6. `repeat_kv` to expand `[Hkv, S, d]` → `[H, S, d]` (GQA broadcast).
7. `standard_attention(Q, K, V, causal=true)`.
8. `transpose_heads_to_2d` to merge heads: `[H, S, d]` → `[S, H*d]`.
9. Restore batch dim: `[S, hidden]` → `[1, S, hidden]`.
10. `o_proj`.

`pub fn forward_with_cache` in `attention.rs` mirrors steps 2-9 for
a single token (`S = 1`):

1. Shape-validate `[1, 1, hidden]`.
2. Project Q (`[1, 1, H*d]`), K/V (`[1, 1, Hkv*d]`).
3. Reshape Q `[1, H*d]` → `[H, 1, d]`, K `[1, Hkv*d]` → `[Hkv, 1, d]`,
   V same.
4. Apply RoPE at `seq_offset` to Q and K only.
5. Append the post-RoPE K and raw V to the cache via
   `LayerKvCache::append` (or seed via
   `LayerKvCache::from_single_token` when `cache = None` and
   `seq_offset = 0`).
6. Expand the full cached K/V via `repeat_kv` to `[H, seq, d]`.
7. `standard_attention(Q, K_full, V_full, causal=false)`. The
   `causal=false` choice is deliberate: with `N_q = 1` the single
   query position is the LAST row of the conceptual full sequence
   and can attend to all history; the `standard_attention` causal
   mask `j > i` would mask everything past position 0 in this
   `N_q = 1` shape, which is wrong for incremental decoding.
8. Merge heads, restore batch dim, `o_proj`.

The strict-mode `load_state_dict` path validates the four projection
prefixes (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and rejects keys
outside that set.

### Non-test production consumers

- `pub use attention::LlamaAttention` at
  `ferrotorch-llama/src/lib.rs` exposes the type.
- `pub self_attn: LlamaAttention<T>` field of `LlamaDecoderLayer` in
  `layer.rs` is the canonical consumer. `LlamaDecoderLayer::new`
  constructs `LlamaAttention::new(cfg)?`.
- `LlamaDecoderLayer::forward` in `layer.rs` calls
  `self.self_attn.forward(&h)?` after `input_layernorm`.
- `LlamaDecoderLayer::forward_with_cache` in `layer.rs` calls
  `self.self_attn.forward_with_cache(&h, cache, seq_offset)?`.

## Parity contract

`parity_ops = []`. Attention composes `Linear`,
`RotaryPositionEmbedding`, `standard_attention`, `repeat_kv`,
`reshape_to_heads`, `transpose_heads_to_2d` — all owned by
`ferrotorch-nn` for parity. Behavioral contract:

- **RoPE convention**: `HalfRotation` (HF's `rotate_half` at
  `modeling_llama.py:109` splits the last dim in half and rotates
  `[-x2, x1]`). Differs from the "interleave" convention used by
  some other RoPE implementations.
- **No QKV bias / no output-projection bias**: `bias = false` on all
  four projections, matching HF's `attention_bias = False` default.
- **GQA via repeat_kv before attention**: HF's
  `eager_attention_forward` at `modeling_llama.py:181` calls
  `repeat_kv(key, module.num_key_value_groups)`. Ferrotorch matches:
  the GQA broadcast happens before the score matmul, not inside
  it. Numerically equivalent; matches HF's reference path.
- **Scale = 1 / sqrt(head_dim)**: `standard_attention` applies the
  scale internally. Matches HF's `self.scaling = self.head_dim**-0.5`
  at `modeling_llama.py:206`.
- **Causal mask on full prefix; no mask on single-token forward**:
  full-prefix `forward(causal=true)` matches HF's
  `is_causal = True` default. The `forward_with_cache` `causal=false`
  choice is correct for the `N_q = 1` shape (see Architecture step 7
  above).

## Verification

`attention.rs` has no in-file `#[cfg(test)] mod tests`. Its
behavior is exercised transitively via the model-level tests:

- `tiny_model_forward_from_ids_produces_correct_shape` — full-prefix
  forward on the tiny stack.
- `forward_one_with_cache_matches_full_prefix_forward` — pins the
  `forward_with_cache` numerical equivalence to the full-prefix
  path.
- `conformance_pretrained_causal_lm.rs` (integration test) drives
  the full attention block against real Llama-format weights.

No parity-sweep ops. Smoke command:

```bash
cargo test -p ferrotorch-llama --lib model::tests 2>&1 | tail -3
```

Expected: all tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct LlamaAttention<T: Float>` + `LlamaAttention::new` in `attention.rs` (constructs four `Linear` projections with `bias = false` and the RoPE table); non-test consumer: `pub self_attn: LlamaAttention<T>` field of `LlamaDecoderLayer` in `layer.rs`. |
| REQ-2 | SHIPPED | impl: `LlamaAttention::new` sizes K/V as `cfg.num_key_value_heads * head_dim` in `attention.rs`; non-test consumer: same `LlamaDecoderLayer::new` path; the GPU MLP path in `gpu.rs` independently sizes the K/V projections to `n_kv * head_dim` per the same config. |
| REQ-3 | SHIPPED | impl: `Module::forward` for `LlamaAttention` in `attention.rs`; non-test consumer: `LlamaDecoderLayer::forward` in `layer.rs` calls `self.self_attn.forward(&h)?`. |
| REQ-4 | SHIPPED | impl: `pub fn forward_with_cache` in `attention.rs`; non-test consumer: `LlamaDecoderLayer::forward_with_cache` in `layer.rs` calls `self.self_attn.forward_with_cache(&h, cache, seq_offset)?`. |
| REQ-5 | SHIPPED | impl: cache write through `LayerKvCache::from_single_token` / `LayerKvCache::append` in `forward_with_cache` in `attention.rs`; non-test consumer: the cache type is held in `LlamaKvCache::layers` in `kv_cache.rs`, threaded by `LlamaForCausalLM::forward_one_with_cache` in `model.rs`. |
| REQ-6 | SHIPPED | impl: the `if seq_offset != 0` guard in the `cache = None` branch in `forward_with_cache` in `attention.rs`; non-test consumer: same call path as REQ-4 — the `LlamaForCausalLM::forward_one_with_cache` driver in `model.rs` always seeds with `seq_offset = cache.len()`, so the gate trips only on misuse. |
| REQ-7 | SHIPPED | impl: `Module::named_parameters` for `LlamaAttention` in `attention.rs`; non-test consumer: `LlamaDecoderLayer::named_parameters` in `layer.rs` walks the attention's named parameters and prefixes with `self_attn.`. |
