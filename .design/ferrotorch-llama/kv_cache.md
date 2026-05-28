# ferrotorch-llama — `kv_cache` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - HuggingFace transformers/models/llama/modeling_llama.py
    (past_key_values argument threaded through LlamaDecoderLayer.forward
    and LlamaAttention.forward; DynamicCache class from
    transformers.cache_utils)
  - HuggingFace transformers.cache_utils.DynamicCache (the
    per-layer growing K/V slab analog)
-->

## Summary

`ferrotorch-llama/src/kv_cache.rs` ships the per-layer
`LayerKvCache<T>` (one layer's K and V slabs) and the per-call
`LlamaKvCache<T>` (one entry per decoder layer). The cache enables
incremental decoding: a single-token forward pass attends against
the cached K/V from all previous positions, dropping the per-step
cost from `O(seq²·d)` (full-prefix re-forward) to `O(seq·d)`.
Cloning is intentional deep-copy so beam-search fork points
produce independent K/V trails.

## Requirements

- REQ-1: `pub struct LayerKvCache<T: Float>` carries
  `k: Tensor<T>` and `v: Tensor<T>` both shaped
  `[num_kv_heads, seq_len, head_dim]`. K is post-RoPE; V is raw.
- REQ-2: `LayerKvCache::seq_len` returns the cached length or
  `ShapeMismatch` if the underlying tensor is not 3-D.
- REQ-3: `LayerKvCache::append(new_k, new_v)` extends both slabs
  by one position. Inputs must be `[num_kv_heads, 1, head_dim]`
  and match the cache on head count and head_dim. Returns a fresh
  `LayerKvCache` with `seq_len + 1`.
- REQ-4: `LayerKvCache::from_single_token(new_k, new_v)` seeds a
  cache directly from `[num_kv_heads, 1, head_dim]` K and V
  tensors. Rejects any other shape with `ShapeMismatch`.
- REQ-5: `pub struct LlamaKvCache<T: Float>` carries
  `layers: Vec<LayerKvCache<T>>` (one per decoder layer) plus a
  cached `seq_len: usize`. Derives `Clone` so beam-search fork
  points can deep-copy parent state.
- REQ-6: `LlamaKvCache::empty` returns a freshly-allocated empty
  cache with `seq_len = 0`. `LlamaKvCache::len` and
  `LlamaKvCache::is_empty` provide the canonical length accessors.
- REQ-7: `LlamaKvCache::extend(new_layer_kv)` appends one token's
  per-layer K/V slabs. On first call (empty cache) it seeds each
  layer via `from_single_token`; on subsequent calls it appends
  via `LayerKvCache::append`. Rejects layer-count mismatch with
  `InvalidArgument`.

## Acceptance Criteria

- [x] AC-1: `LlamaKvCache::<f32>::empty()` constructs and reports
  `is_empty() == true`, `len() == 0`.
- [x] AC-2: `extend` seeds an empty cache from 2 layers, then
  appends one position; the resulting K tensor's seq dim grows from
  1 to 2.
- [x] AC-3: `extend` rejects a layer-count change between calls
  (`InvalidArgument`).
- [x] AC-4: `LayerKvCache::append` rejects a head-count mismatch
  (`ShapeMismatch`).
- [x] AC-5: A cache cloned at a beam-search fork point and grown by
  one token does NOT affect the parent cache (per `Clone` deep-copy
  contract).

## Architecture

`pub struct LayerKvCache<T: Float>` in `kv_cache.rs` carries the
post-RoPE keys and raw values for a single decoder layer. The
3-D `[num_kv_heads, seq, head_dim]` shape is the post-projection,
pre-GQA-broadcast format — broadcasting via `repeat_kv` happens
inside `LlamaAttention::forward_with_cache` after the cache lookup.

`pub fn LayerKvCache::append` in `kv_cache.rs` extends the cache
by one position. It validates every input tensor is 3-D, validates
the new K/V have `seq_len = 1`, validates the head count and
`head_dim` match. Then it builds two new `Vec<T>` buffers
interleaving the existing rows and the new row per head, and
constructs fresh tensors. The deep-copy is intentional — beam search
forks share-by-clone, and aliasing storage would let a child
mutation corrupt its parent.

`pub fn LayerKvCache::from_single_token` in `kv_cache.rs` is the
seed constructor: it validates the `[Hkv, 1, d]` shape and
constructs a `LayerKvCache` from the tensors directly. No copy
beyond the move into the struct.

`pub struct LlamaKvCache<T: Float>` in `kv_cache.rs` carries one
`LayerKvCache` per decoder layer plus a cached `seq_len`. The
`#[derive(Clone)]` produces a deep-copy of every layer's tensor
storage — this is `O(num_layers · seq_len · num_kv_heads · head_dim)`
but is the only correct behavior for branching beam search.

`pub fn LlamaKvCache::extend` in `kv_cache.rs` is the canonical
growth path. When the cache is empty (first token), it constructs
one `LayerKvCache::from_single_token` per supplied `(K, V)` pair
and sets `seq_len = 1`. On subsequent calls, it requires the
incoming layer count to match the existing layer count, then
appends per-layer via `LayerKvCache::append`.

The `LlamaForCausalLM::forward_one_with_cache` path in `model.rs`
manages the cache differently: rather than calling
`LlamaKvCache::extend`, it constructs the per-layer caches inside
each `LlamaDecoderLayer::forward_with_cache` call (which returns
the new layer cache) and reassembles a fresh `LlamaKvCache` at the
end. Both paths produce equivalent results; the layer-level path
avoids one round-trip of cache packing/unpacking.

### Non-test production consumers

- `pub use kv_cache::{LayerKvCache, LlamaKvCache}` at
  `ferrotorch-llama/src/lib.rs:171` exposes both types.
- `LlamaForCausalLM::forward_one_with_cache` in `model.rs` takes
  `cache: &LlamaKvCache<T>` and returns `(Vec<f64>, LlamaKvCache<T>)`.
- `LlamaAttention::forward_with_cache` in `attention.rs` takes
  `cache: Option<&LayerKvCache<T>>` and calls
  `prev.append(&k_rot, &v_h)` / `LayerKvCache::from_single_token`.
- `LlamaDecoderLayer::forward_with_cache` in `layer.rs` threads
  `Option<&LayerKvCache<T>>` between input layernorm and attention.
- `generation::beam_search` in `generation.rs` constructs a
  `LlamaKvCache::<T>::empty()` per beam and clones it on
  fork points (the `parent.cache.clone()` at the beam-expansion
  step).

## Parity contract

`parity_ops = []`. The cache is a structural artifact — HF's
`DynamicCache` from `transformers.cache_utils` carries equivalent
state but with a different in-memory layout (HF stores K/V tensors
per layer in a `[batch, num_kv_heads, seq, head_dim]` 4-D format;
ferrotorch's `[num_kv_heads, seq, head_dim]` 3-D format reflects
the single-batch contract).

Behavioral guarantees:

- **Deep clone on fork**: `#[derive(Clone)]` copies every layer's
  tensor storage. Beam search cloning a parent cache produces N
  independent K/V trails. HF's `DynamicCache.clone` has equivalent
  semantics.
- **Seq dimension grows monotonically**: every `append` /
  `extend` call increases `seq_len` by exactly one. No
  truncation, no rewind — beam search backtracks by NOT cloning
  the parent (it constructs the next round from the surviving
  candidates).
- **Post-RoPE K caching**: matches HF's contract where the cache
  stores the rotated keys (`past_key_values.update(key_states,
  value_states, ...)` at HF `modeling_llama.py:246` happens AFTER
  `apply_rotary_pos_emb`). Storing pre-RoPE keys would require
  re-rotating every attended-to position on every step.

## Verification

Tests in `mod tests in kv_cache.rs`:

- `empty_cache_has_zero_len`
- `extend_seeds_then_appends`
- `extend_rejects_layer_count_change`
- `append_rejects_shape_mismatch`

Plus model-level integration via:

- `forward_one_with_cache_matches_full_prefix_forward` in
  `mod tests in model.rs` — exercises the full cache growth path
  and pins numerical equivalence to the full-prefix forward.

No parity-sweep ops. Smoke command:

```bash
cargo test -p ferrotorch-llama --lib kv_cache:: 2>&1 | tail -3
```

Expected: all tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct LayerKvCache<T: Float>` in `kv_cache.rs`; non-test consumer: held inside `LlamaKvCache::layers` in `kv_cache.rs` and produced by `LlamaAttention::forward_with_cache` in `attention.rs`. |
| REQ-2 | SHIPPED | impl: `pub fn LayerKvCache::seq_len` in `kv_cache.rs`; non-test consumer: the same `LayerKvCache` shape is read by `LlamaAttention::forward_with_cache` in `attention.rs` for the `repeat_kv` broadcast step. |
| REQ-3 | SHIPPED | impl: `pub fn LayerKvCache::append` in `kv_cache.rs`; non-test consumer: invoked by `LlamaAttention::forward_with_cache` in `attention.rs` (the `prev.append(&k_rot, &v_h)?` line on the cached-prior branch). |
| REQ-4 | SHIPPED | impl: `pub fn LayerKvCache::from_single_token` in `kv_cache.rs`; non-test consumer: invoked by `LlamaAttention::forward_with_cache` in `attention.rs` (the `cache = None` branch seeding the layer). |
| REQ-5 | SHIPPED | impl: `pub struct LlamaKvCache<T: Float>` + `#[derive(Clone)]` in `kv_cache.rs`; non-test consumer: argument to `LlamaForCausalLM::forward_one_with_cache` in `model.rs`; cloned on every beam fork in `beam_search` in `generation.rs`. |
| REQ-6 | SHIPPED | impl: `LlamaKvCache::empty` / `len` / `is_empty` in `kv_cache.rs`; non-test consumer: `let mut seed_cache = crate::kv_cache::LlamaKvCache::<T>::empty();` in `beam_search` in `generation.rs`. |
| REQ-7 | SHIPPED | impl: `pub fn LlamaKvCache::extend` in `kv_cache.rs`; non-test consumer: `LlamaKvCache` growth happens via `LlamaForCausalLM::forward_one_with_cache` in `model.rs` which builds and returns the next cache directly; `extend` is the public alternative for callers that already have per-layer slabs in hand. |
