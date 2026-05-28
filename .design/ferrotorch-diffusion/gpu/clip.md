# GpuClipTextEncoder

<!--
tier: 3-component
status: draft
baseline-pytorch: /home/doll/pytorch (HEAD)
upstream-paths:
  - transformers/src/transformers/models/clip/modeling_clip.py
-->

## Summary

GPU twin of [`ClipTextEncoder`] for SD-1.5
(`openai/clip-vit-large-patch14` — the text tower). VRAM-resident
forward path; mirrors the CPU module op-for-op via
`ferrotorch-gpu` kernels (`gpu_embed_lookup_batch`,
`gpu_layernorm`, `gpu_matmul_f32`, `gpu_bmm_f32`, `gpu_softmax`,
`gpu_gelu` (QuickGELU), `gpu_add`, `gpu_broadcast_add`,
`gpu_scale`). One-shot weight upload at construction; downloads
only the final `last_hidden_state` `[1, S, hidden_size]` at
`encode`. Causal-masked self-attention; QuickGELU MLP.

## Requirements

- REQ-1: `GpuClipTextEncoder::new(config, state, device)` validates
  the config, uploads every parameter tensor exactly once into
  VRAM, transposes Linear weights on host to store `W^T` so each
  forward Linear is a single `matmul` without per-call transpose.
  Pre-computes the `[max_pos, max_pos]` causal mask buffer on the
  device (0.0 at/below diagonal, `-INF` strictly above). Returns
  `(Self, DropReport)`.
- REQ-2: `GpuClipTextEncoder::from_module(cpu, device)` is the
  convenience constructor: extracts `cpu.state_dict()` and calls
  `Self::new`.
- REQ-3: `GpuClipTextEncoder::encode(input_ids: &[u32])` runs the
  forward on a single padded sequence and returns
  `[1, S, hidden_size]` as a host `Tensor<f32>`. Validates `S > 0`,
  `S ≤ max_position_embeddings`, every id `< vocab_size`.
- REQ-4: Per-layer forward is pre-LN: residual ← h; h ← LN1(h); h ←
  causal-self-attn(h) (the pre-computed causal mask is added to the
  per-head scores before softmax); h ← residual + h; residual ← h;
  h ← LN2(h); h ← fc2(QuickGELU(fc1(h))); h ← residual + h. Final
  `final_layer_norm` after the 12-layer stack.
- REQ-5: All four self-attention projections (q/k/v/out) carry
  bias. QuickGELU (not standard GELU, not tanh-approx): `x *
  sigmoid(1.702 * x)` via `ferrotorch_gpu::kernels::gpu_gelu`.

## Acceptance Criteria

- [x] AC-1: `gpu_clip_matches_cpu` (lib test, `feature = "cuda"`)
  loads a tiny CPU encoder, builds the GPU twin via `from_module`,
  runs both on the same input ids, asserts cosine similarity
  ≥ 0.999 on the last_hidden_state.
- [x] AC-2: Causal mask is correctly broadcast — `attn[i, j]` is
  unchanged for `j ≤ i` and dampened by `-INF` for `j > i` before
  softmax. Pinned by the same lib test (any leak of bidirectional
  attention would crater the cosine similarity).
- [x] AC-3: All twelve encoder layers + final layer norm produce
  finite outputs on the tiny config.

## Architecture

- Per-component bundles at `gpu in gpu/clip.rs`:
  `GpuLayerNorm`, `GpuLinearT` (stores `[in, out]` row-major =
  transpose of PyTorch `[out, in]`), `GpuClipAttn`, `GpuClipMlp`,
  `GpuClipLayer`.
- `GpuClipTextEncoder in gpu/clip.rs` holds
  `token_embedding`, `position_embedding`, twelve `GpuClipLayer`,
  `final_layer_norm`, pre-computed `causal_mask_full`, config,
  and a `GpuDevice` handle.
- `new` at `gpu in gpu/clip.rs` pops every required state-dict key
  by name (matching the CPU `ClipTextEncoder` layout:
  `embeddings.{token,position}_embedding.weight`,
  `encoder.layers.{i}.{layer_norm1,self_attn,layer_norm2,mlp}.*`,
  `final_layer_norm.{weight,bias}`); enforces per-key length;
  uploads via `cpu_to_gpu`. Linear weights are transposed on host
  before upload (the `to_transposed_*` helper).
- `from_module` at `gpu in gpu/clip.rs` extracts the CPU
  module's `state_dict()` and delegates to `new`.
- `encode` at `gpu in gpu/clip.rs` runs the seven-stage forward.
  Token + position embeddings via `gpu_embed_lookup_batch` (with
  a `[0..S)` index tensor for positions). Causal-mask slice is the
  full pre-computed buffer when `S == max_pos`; otherwise downloads,
  slices, and re-uploads (cold path; SD-1.5 inference uses
  `S == 77 == max_pos`). Twelve layers run pre-LN + Q/K/V projections
  + heads reshape + bmm + scale + causal-mask add + softmax +
  weighted bmm + out-proj + residual; then LN2 + MLP + residual.
  Final layer norm on the result. Download via `gpu_to_cpu`.

Non-test production consumers:

- `ferrotorch-diffusion/src/gpu/mod.rs,38` re-exports
  `GpuClipTextEncoder`; the prelude is `use
  ferrotorch_diffusion::gpu::GpuClipTextEncoder`.
- `ferrotorch-diffusion/src/gpu/pipeline.rs,64,87` uses
  `GpuClipTextEncoder` as the `text_encoder` field of
  `GpuStableDiffusionPipeline`; `encode_prompt` at
  `gpu/pipeline.rs` calls `self.text_encoder.encode(...)`.
- `ferrotorch-diffusion/examples/clip_text_encode_dump.rs:362,370`
  imports and constructs `GpuClipTextEncoder` for the SD-1.5 CLIP
  inference-dump binary.
- `ferrotorch-diffusion/examples/sd_pipeline_dump.rs:482` imports
  it for the end-to-end SD-1.5 GPU dump.

## Parity contract

`parity_ops = []`. The encoder is a composition of ferrotorch-gpu
kernels (each separately parity-checked). End-to-end vs Python:
the `conformance_pretrained_diffusion.rs` integration test runs the
encoder on the pinned SD-1.5 mirror and asserts cosine similarity
≥ 0.99 against the HF reference output.

Critical-not-to-regress invariants:

- **Causal mask** (mandatory): `[S, S]` buffer with `0.0` at/below
  diagonal and `-INF` strictly above, broadcast-added to per-head
  attention scores BEFORE the softmax. Module rustdoc at
  `gpu in gpu/clip.rs` states this explicitly.
- **QuickGELU not GELU**: `x * sigmoid(1.702 * x)`. The
  `ferrotorch_gpu::kernels::gpu_gelu` kernel is the QuickGELU
  variant (a separate `gpu_gelu_erf` exists for the UNet GEGLU).
- **All four projections have bias** (q/k/v/out).
- **`W^T` storage** (`[in, out]` row-major): every Linear forward is
  `matmul(x, W_t) + b`, no per-call transpose.

## Verification

Lib tests at the end of `gpu/clip.rs` (`#[cfg(test)]
mod tests` under `feature = "cuda"`). Cargo invocation:

```text
cargo test -p ferrotorch-diffusion --lib --features cuda gpu::clip
```

End-to-end: `tests/conformance_pretrained_clip.rs` and
`tests/conformance_pretrained_diffusion.rs` exercise the encoder
against the pinned SD-1.5 mirror.

No parity-sweep ops apply (composition module; per-kernel parity is
checked at the op layer).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `GpuClipTextEncoder::new` at `ferrotorch-diffusion/src/gpu/clip.rs:193..`; non-test consumer: `ferrotorch-diffusion/src/gpu/clip.rs:348` `from_module` calls `Self::new(cpu.config.clone(), state, device.clone())`; production binary `ferrotorch-diffusion/examples/clip_text_encode_dump.rs:370` constructs the encoder via `from_module` |
| REQ-2 | SHIPPED | impl: `from_module` at `ferrotorch-diffusion/src/gpu/clip.rs:343..349`; non-test consumer: `ferrotorch-diffusion/examples/clip_text_encode_dump.rs:370` `GpuClipTextEncoder::from_module(encoder, &device)?`; `ferrotorch-diffusion/examples/sd_pipeline_dump.rs` similarly invokes it for the pipeline build |
| REQ-3 | SHIPPED | impl: `encode` at `ferrotorch-diffusion/src/gpu/clip.rs:363..`; non-test consumer: `ferrotorch-diffusion/src/gpu/pipeline.rs:108` `self.text_encoder.encode(input_ids)` is the canonical text-tower call in the SD-1.5 GPU pipeline; `encode_prompt` is itself called from the dump example |
| REQ-4 | SHIPPED | impl: pre-LN layer body at `ferrotorch-diffusion/src/gpu/clip.rs:447..505` (LN1, Q/K/V, heads, bmm, scale, causal mask add, softmax, weighted bmm, out-proj, residual; LN2 + MLP + residual; final `final_layer_norm`); non-test consumer: `encode` at `ferrotorch-diffusion/src/gpu/clip.rs:447` iterates over `self.layers` invoking the full body for every forward |
| REQ-5 | SHIPPED | impl: q/k/v/out bias upload (`GpuClipAttn` four `GpuLinearT` fields at `GpuLinearT in gpu/clip.rs`, each carrying `bias: CudaBuffer<f32>` per `gpu in gpu/clip.rs`); QuickGELU dispatched via `ferrotorch_gpu::kernels::gpu_gelu` import at `gpu_gelu in ferrotorch-diffusion/src/gpu/clip.rs`; non-test consumer: `encode`'s per-layer forward uses both surfaces (`linear_forward` for each proj at `gpu in gpu/clip.rs,491..492` and `gpu_gelu` for the MLP at `gpu in gpu/clip.rs`) |
