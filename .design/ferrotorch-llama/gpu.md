# ferrotorch-llama — `gpu` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - HuggingFace transformers/models/llama/modeling_llama.py (the
    structural reference for the bf16 GPU forward pass: same
    embedding → N × (RMSNorm + GQA + RoPE + SwiGLU + residual) →
    final RMSNorm → lm_head pipeline)
  - llama.cpp (the architectural inspiration: weights resident on
    device as raw u16 bf16 bit slabs, every op a kernel launch
    against CudaSlice<u16>)
-->

## Summary

`ferrotorch-llama/src/gpu.rs` ships `LlamaGpuInferencer`, the
GPU-resident inference path: every weight uploaded as
`CudaSlice<u16>` (bf16 bit layout), every op a direct call into the
`ferrotorch-gpu` hand-written bf16 PTX kernels. No generic
`Tensor<T>` dispatch, no CPU round-trips between layers, no
external toolchain. This is the canonical downstream consumer of
the `ferrotorch-gpu` bf16 transformer primitives
(`gpu_embedding_gather_bf16`, `gpu_rmsnorm_bf16`,
`gpu_matmul_bf16_bf16_nt`, `gpu_rope_half_bf16`,
`gpu_repeat_kv_bf16`, `gpu_causal_mask_bf16`, `gpu_softmax_bf16`,
`gpu_silu_bf16`, `gpu_add_bf16`, `gpu_mul_bf16`,
`gpu_transpose_to_heads_bf16`, `gpu_transpose_from_heads_bf16`,
`gpu_block_reduce_max_abs_bf16`,
`gpu_matmul_bf16_bf16_strided_batched`,
`gpu_matmul_bf16_bf16_strided_batched_nt`).

This module is enabled by the `cuda` feature on
`ferrotorch-llama`.

## Requirements

- REQ-1: `pub struct LlamaGpuInferencer` owns the frozen
  `LlamaConfig`, the `GpuDevice` handle, and every weight as a
  `CudaSlice<u16>` (`embed_tokens`, `norm`, `lm_head`, per-layer
  `LlamaGpuLayer`, plus precomputed `cos_cache` / `sin_cache` for
  half-rotation RoPE).
- REQ-2: `pub struct LlamaGpuLayer` carries the nine per-layer
  weights: `input_norm`, `q_proj`, `k_proj`, `v_proj`, `o_proj`,
  `post_attn_norm`, `gate_proj`, `up_proj`, `down_proj` — all
  `CudaSlice<u16>`.
- REQ-3: `LlamaGpuInferencer::new(config, state, device)` uploads
  every required tensor from a `StateDict<bf16>` to VRAM, drains
  the host state as each tensor uploads, validates expected shapes
  against the config, and precomputes the RoPE cos/sin caches.
  Handles tied embeddings by copying `model.embed_tokens.weight`
  into `lm_head.weight` when the latter is absent and
  `config.tie_word_embeddings == true`.
- REQ-4: `LlamaGpuInferencer::forward_from_ids(&ids)` runs the full
  bf16 forward over `[seq, hidden]` and returns last-token logits
  as `Vec<f32>` of length `vocab_size`. Empty ids and
  `seq > max_position_embeddings` return `InvalidArgument`.
- REQ-5: `LlamaGpuInferencer::forward_logits_from_ids_all(&ids)` is
  the harness-oriented sibling that downloads logits for every
  position (not just the last row). Same forward pass; only the
  host-side download window differs.
- REQ-6: `LlamaGpuInferencer::forward_from_ids_profiled(...)` runs
  the forward with per-(layer, token, head) attention tap
  magnitudes and per-(layer, token, block) MLP tap magnitudes for
  the paged-weight sparsity profiler.
- REQ-7: `LlamaGpuInferencer::forward_from_ids_profiled_with_bootstrap(...)`
  extends the profiler with an optional bootstrap hidden-state tap
  at layer `bootstrap_k - 1` (snapshotted on-device, downloaded
  to host on return). When `plainact_local` is set, the MLP tap
  uses the local plainact surrogate
  `|gated[i] * (y_full @ down_proj)[i]|` instead of raw activation
  magnitude.
- REQ-8: Activation dispatch in the GPU MLP block matches
  `LlamaActivation`: `Silu → gpu_silu_bf16`, `Relu → gpu_relu_bf16`,
  `FatRelu(θ) → gpu_fatrelu_bf16(θ as f32)`.
- REQ-9: Error mapping: `ferrotorch_gpu::GpuError` and
  `cudarc::driver::DriverError` route through `FerrotorchError`
  variants — `ShapeMismatch` / `LengthMismatch` /
  `NotImplementedOnCuda` keep their categorical mappings;
  everything else (driver, cublas, oom, etc.) goes through
  `FerrotorchError::Gpu { source }` for downcast recovery
  (tracks #699).

## Acceptance Criteria

- [x] AC-1: `LlamaGpuInferencer::new(cfg, state, device)` returns
  a valid inferencer on a known-good `StateDict<bf16>` for the
  llama3_8b configuration (exercised by the example
  `llama3_8b_gpu.rs`).
- [x] AC-2: Missing required tensor (`model.embed_tokens.weight`)
  returns `InvalidArgument`.
- [x] AC-3: Wrong shape on a known tensor returns `ShapeMismatch`.
- [x] AC-4: Tied embeddings (`tie_word_embeddings = true`,
  `lm_head.weight` absent) copy `embed_tokens.weight` into
  `lm_head.weight` before per-tensor upload.
- [x] AC-5: `forward_from_ids` on an empty `ids` slice returns
  `InvalidArgument`.
- [x] AC-6: `forward_from_ids` on `seq > max_position_embeddings`
  returns `InvalidArgument`.
- [x] AC-7: Profiled forward with `bootstrap_k = 0` or
  `bootstrap_k > num_hidden_layers` returns `InvalidArgument`.

## Architecture

`pub struct LlamaGpuInferencer` in `gpu.rs` holds the full model
weights as `CudaSlice<u16>` (bf16 bit pattern) plus the device
handle and the precomputed RoPE caches. The manual `Debug` impl
(cudarc's `CudaSlice<u16>` doesn't derive `Debug`) prints elem
counts for each buffer.

`pub fn LlamaGpuInferencer::new` in `gpu.rs`:

1. `config.validate()?`.
2. If `tie_word_embeddings && !state.contains_key("lm_head.weight")`:
   copy `model.embed_tokens.weight` into `lm_head.weight` BEFORE
   the per-tensor uploads (which drain the state dict).
3. Upload `embed_tokens` via `upload_bf16_tensor`.
4. For each layer index, call `upload_layer` to construct one
   `LlamaGpuLayer` from the nine HF-keyed tensors.
5. Upload `norm` and `lm_head`.
6. Build `cos_cache` / `sin_cache` via `build_rope_caches`.

`fn upload_bf16_tensor` in `gpu.rs` `remove`s the named tensor from
the state dict, validates the expected shape, casts the
`Vec<bf16>` into `Vec<u16>` bits via `bytemuck::cast_slice` (bf16
is `repr(transparent)` over `u16`), and uploads via
`device.stream().clone_htod`.

`fn build_rope_caches` in `gpu.rs` computes the cos/sin tables for
half-rotation RoPE in f64 (for numerical stability), casts to
bf16 bits, and uploads. Output shape: each cache is
`[max_seq, head_dim / 2]`.

`pub fn LlamaGpuInferencer::forward_from_ids` is the
last-token-logits entry point. It calls `forward_core(ids, None)`
to produce the final post-RMSNorm hidden state, projects to logits
via `gpu_matmul_bf16_bf16_nt`, then `clone_dtoh`s the last-token
row (vocab elements) and converts bf16 bits to f32.

`fn forward_core` in `gpu.rs` is the shared per-layer loop. For
each layer:

1. `gpu_rmsnorm_bf16` (input layernorm).
2. `gpu_matmul_bf16_bf16_nt` for Q, K, V projections.
3. `gpu_transpose_to_heads_bf16` to split heads.
4. `gpu_rope_half_bf16` on Q and K only.
5. `gpu_repeat_kv_bf16` to broadcast K/V over the GQA group.
6. `gpu_matmul_bf16_bf16_strided_batched_nt` for attention scores
   (with the `1/sqrt(head_dim)` scale).
7. `gpu_causal_mask_bf16` to apply the lower-triangular mask.
8. `gpu_softmax_bf16` row-wise.
9. `gpu_matmul_bf16_bf16_strided_batched` for the scores @ V matmul.
10. (optional attention tap)
11. `gpu_transpose_from_heads_bf16` then `gpu_matmul_bf16_bf16_nt`
    for `o_proj`.
12. `gpu_add_bf16` for the attention residual.
13. `gpu_rmsnorm_bf16` (post-attention layernorm).
14. `gpu_matmul_bf16_bf16_nt` for gate / up projections.
15. Activation dispatch by `cfg.hidden_act`:
    `Silu → gpu_silu_bf16`, `Relu → gpu_relu_bf16`,
    `FatRelu(θ) → gpu_fatrelu_bf16(θ as f32)`.
16. `gpu_mul_bf16(activated_gate, up)` for the SwiGLU product.
17. (optional default MLP tap via `gpu_block_reduce_max_abs_bf16`)
18. `gpu_matmul_bf16_bf16_nt` for `down_proj`.
19. `gpu_add_bf16` for the MLP residual.
20. (optional local-plainact MLP tap via one extra
    `gpu_matmul_bf16_bf16` against the residual)
21. (optional bootstrap snapshot via `memcpy_dtod`)

After the layer loop, a final `gpu_rmsnorm_bf16` against
`self.norm` produces the post-stack hidden state.

The profiled and bootstrap variants (REQ-6 / REQ-7) feed the
shared `forward_core` with `Some(&mut ForwardTaps)` so the
per-(layer, token, head) and per-(layer, token, block) magnitudes
collect into pre-allocated CUDA buffers. After the forward, the
host download reassembles the magnitudes in token-major order
(`[t, l, h]` for attention, `[t, l, b]` for MLP).

`fn map_gpu_err` and `fn map_driver_err` in `gpu.rs` are the
categorical error mappers (#699). Shape/length errors map to
`FerrotorchError::ShapeMismatch`. Unsupported (op, dtype) maps to
`FerrotorchError::NotImplementedOnCuda`. Driver / cuBLAS / cuSOLVER
/ cuFFT / state errors route through `FerrotorchError::Gpu { source }`
so callers can downcast to the original `GpuError` via
`std::error::Error::source`.

### Non-test production consumers

- `#[cfg(feature = "cuda")] pub use gpu::{LlamaGpuInferencer,
  LlamaGpuLayer, ProfiledForwardResult}` at
  `ferrotorch-llama/src/lib.rs:181`.
- `LlamaGpuInferencer::new(cfg, state, device)` at
  `ferrotorch-llama/examples/llama3_8b_gpu.rs:100`.
- `inferencer.forward_from_ids(&tokens)` at
  `ferrotorch-llama/examples/llama3_8b_gpu.rs:121`.
- `LlamaGpuInferencer::new(cfg, state, device)` at
  `ferrotorch-llama/examples/llama3_70b_gpu.rs` (construction site)
  and `inferencer.forward_from_ids(&tokens)` at line 135.
- `LlamaGpuInferencer::new(cfg, state, device)` at
  `ferrotorch-llama/examples/prosparse_7b_gpu.rs:75`; the same
  example calls
  `inferencer.forward_from_ids_profiled_with_bootstrap(&ids, 1,
  ffn, None, false)` at line 94.
- `LlamaGpuInferencer::new(cfg, state, device)` at
  `ferrotorch-llama/examples/llm_inference_dump.rs:255` (inside
  the `--feature cuda` branch).
- The `ferrotorch::llama` umbrella re-export at
  `ferrotorch/src/lib.rs:155` makes `LlamaGpuInferencer` reachable
  for any downstream user that depends on the meta-crate with the
  `llama-cuda` feature.

## Parity contract

`parity_ops = []` (the GPU module composes per-op kernels owned by
`ferrotorch-gpu`). Structural / numerical guarantees:

- **End-to-end bf16**: every intermediate tensor between layers
  lives in VRAM as bf16. The only f32 traffic is the post-attention
  / post-MLP magnitude taps (used for sparsity profiling, not the
  forward path) and the final logits download.
- **No CPU round-trips between layers** (R-CODE-4): every op is a
  direct kernel launch; the hidden state never leaves the device
  during the layer loop.
- **GQA via `gpu_repeat_kv_bf16` before scores**: matches HF's
  reference path which broadcasts K/V before the `Q @ K^T` matmul.
- **RoPE half-rotation cos/sin caches built once at upload**:
  matches HF's `LlamaRotaryEmbedding` which precomputes
  `inv_freq` then per-position cos/sin on each forward; ferrotorch
  precomputes the full `[max_seq, head_dim/2]` table at upload time
  so the per-forward cost is just an index lookup.
- **Causal mask via `gpu_causal_mask_bf16`**: lower-triangular
  mask added before softmax. Matches HF's
  `create_causal_mask` at `modeling_llama.py:382`.
- **bf16 RoPE caches built in f64 then cast**: matches the
  numerical-stability discipline of HF's
  `with torch.autocast(device_type=device_type, enabled=False):
  freqs = ... ; cos = emb.cos() ...` at
  `modeling_llama.py:100-104`.

## Verification

`gpu.rs` has no in-file `#[cfg(test)] mod tests`. CUDA-gated
testing happens in:

- `ferrotorch-llama/tests/gpu_smoke.rs` — minimal GPU forward
  smoke test.
- `ferrotorch-llama/tests/conformance_pretrained_causal_lm_gpu.rs`
  — full-checkpoint GPU parity against HF.
- The example drivers `llama3_8b_gpu.rs`, `llama3_70b_gpu.rs`,
  `prosparse_7b_gpu.rs`, `llm_inference_dump.rs` (which double as
  smoke tests).

No parity-sweep ops in this module's route (the per-op parity is
owned by `ferrotorch-gpu` op routes). Smoke command:

```bash
cargo test -p ferrotorch-llama --features cuda --test gpu_smoke 2>&1 | tail -3
```

Expected: passes on a CUDA-capable host.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct LlamaGpuInferencer` definition in `gpu.rs`; non-test consumer: constructed by `LlamaGpuInferencer::new` at `new in ferrotorch-llama/examples/llama3_8b_gpu.rs` and `new in prosparse_7b_gpu.rs`. |
| REQ-2 | SHIPPED | impl: `pub struct LlamaGpuLayer` definition in `gpu.rs`; non-test consumer: stored as `pub layers: Vec<LlamaGpuLayer>` field on `LlamaGpuInferencer` in `gpu.rs` (populated by `upload_layer` once per `cfg.num_hidden_layers`). |
| REQ-3 | SHIPPED | impl: `pub fn LlamaGpuInferencer::new` in `gpu.rs` (with the tied-embeddings remap branch and the `upload_bf16_tensor` / `upload_layer` / `build_rope_caches` calls); non-test consumer: `LlamaGpuInferencer::new(cfg, state, device)` at `ferrotorch-llama/examples/llama3_8b_gpu.rs:100`. |
| REQ-4 | SHIPPED | impl: `pub fn LlamaGpuInferencer::forward_from_ids` in `gpu.rs`; non-test consumer: `inferencer.forward_from_ids(&tokens)` at `forward_from_ids in ferrotorch-llama/examples/llama3_8b_gpu.rs` and at `forward_from_ids in llama3_70b_gpu.rs`. |
| REQ-5 | SHIPPED | impl: `pub fn LlamaGpuInferencer::forward_logits_from_ids_all` in `gpu.rs`; non-test consumer: the parity-harness driver in `ferrotorch-llama/examples/llm_inference_dump.rs` (the `--feature cuda` branch downloads full-prefix logits via this method to compute top-1 argmax-agreement). |
| REQ-6 | SHIPPED | impl: `pub fn LlamaGpuInferencer::forward_from_ids_profiled` in `gpu.rs` (delegates to `_with_bootstrap` with `bootstrap_k = None`, `plainact_local = false`); non-test consumer: `inferencer.forward_from_ids_profiled_with_bootstrap(&ids, 1, ffn, None, false)` at `ferrotorch-llama/examples/prosparse_7b_gpu.rs:94`. |
| REQ-7 | SHIPPED | impl: `pub fn LlamaGpuInferencer::forward_from_ids_profiled_with_bootstrap` in `gpu.rs` (includes `bootstrap_k` validation, `bootstrap_hidden` snapshot via `memcpy_dtod`, and the optional `plainact_local` MLP-tap branch); non-test consumer: same `forward_from_ids_profiled_with_bootstrap in prosparse_7b_gpu.rs` call site directly invokes this variant. |
| REQ-8 | SHIPPED | impl: the `match cfg.hidden_act` block in `fn forward_core` in `gpu.rs` dispatching to `gpu_silu_bf16` / `gpu_relu_bf16` / `gpu_fatrelu_bf16`; non-test consumer: every example forward (CPU and GPU) drives a config whose `hidden_act` selects the matching branch. |
| REQ-9 | SHIPPED | impl: `fn map_gpu_err` and `fn map_driver_err` in `gpu.rs`; non-test consumer: every `.map_err(map_gpu_err)?` / `.map_err(map_driver_err)?` call inside `forward_core`, `forward_from_ids`, and the upload helpers is a production path that converts kernel-level errors into the `FerrotorchError` taxonomy. |
