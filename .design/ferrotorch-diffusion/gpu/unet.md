# GpuUNet2DConditional

<!--
tier: 3-component
status: draft
baseline-pytorch: /home/doll/pytorch (HEAD)
upstream-paths:
  - diffusers/src/diffusers/models/unets/unet_2d_condition.py
  - diffusers/src/diffusers/models/unets/unet_2d_blocks.py
  - diffusers/src/diffusers/models/attention.py
-->

## Summary

GPU twin of [`UNet2DConditionModel`] for SD-1.5. VRAM-resident
forward path mirroring the CPU module op-for-op via
`ferrotorch-gpu` kernels (`gpu_conv2d_f32`, `gpu_group_norm_f32`,
`gpu_layernorm`, `gpu_matmul_f32`, `gpu_bmm_f32`, `gpu_softmax`,
`gpu_silu`, `gpu_gelu_erf` (for the GEGLU FF — NOT QuickGELU),
`gpu_nearest_upsample2x_f32`, `gpu_scale`, `gpu_add`,
`gpu_broadcast_add`). Host-side traffic limited to the one-shot
weight upload, the parameter-free `Timesteps` sinusoidal encoding
(small `[B, bocs[0]]` tensor), the up-block skip-cat bounce
(buffer counts are tiny vs. the rest of the forward), and the
final result download.

## Requirements

- REQ-1: `GpuUNet2DConditional::new(config, state, device)`
  validates the config, uploads every parameter tensor once into
  VRAM, transposes Linear weights on host (`W^T` storage) so each
  forward matmul is single-pass. Returns `(Self, DropReport)` with
  any state-dict keys not consumed.
- REQ-2: `GpuUNet2DConditional::from_module(cpu, device)` is the
  convenience constructor: extracts `cpu.state_dict()` and calls
  `Self::new`.
- REQ-3: `forward(sample, timesteps, encoder_hidden_states)` runs
  the SD-1.5 UNet topology end-to-end:
  - `time_proj` (parameter-free sinusoidal encoding on host via
    `Timesteps::forward_t`);
  - upload + `time_embedding` MLP (Linear → SiLU → Linear);
  - `conv_in`;
  - push `conv_in` output as the initial skip;
  - 3× `CrossAttnDownBlock` + 1× `DownBlock` (with per-block skips
    appended);
  - mid-block (`resnet → transformer → resnet`);
  - 1× `UpBlock` + 3× `CrossAttnUpBlock` (each popping its trailing
    `num_resnets` skips, concatenating along the channel axis
    before each resnet);
  - `conv_norm_out → SiLU → conv_out`;
  - download to host `[B, out_channels, H, W]`.
- REQ-4: Diffusers footgun preserved: `attention_head_dim` is the
  COUNT of heads (per-head dim = `out_channels / num_heads`); the
  module-level rustdoc + `new` body document this explicitly.
- REQ-5: Two distinct GELU variants: `gpu_silu` for resnet
  activations and the time-embedding MLP; `gpu_gelu_erf` (exact erf
  GELU, NOT QuickGELU) for the GEGLU `FeedForward` inside each
  `BasicTransformerBlock`. The contrast with `clip.rs` (which uses
  the QuickGELU `gpu_gelu`) is deliberate.
- REQ-6: Shape validation: `sample.ndim() == 4` and
  `sample.shape()[1] == config.in_channels`; `timesteps.ndim() ==
  1`; `encoder_hidden_states.ndim() == 3` and
  `encoder_hidden_states.shape()[2] == config.cross_attention_dim`.
  Mismatches raise `FerrotorchError::ShapeMismatch`.

## Acceptance Criteria

- [x] AC-1: `gpu_unet_matches_cpu` (lib test, `feature = "cuda"`)
  loads a tiny CPU UNet, builds the GPU twin via `from_module`,
  runs both on a fixed seed sample, asserts cosine similarity
  ≥ 0.999 on the noise prediction.
- [x] AC-2: All 64 + multi-head bmm + 12+ layer-norm + 4
  upsample paths produce finite outputs on the tiny config.
- [x] AC-3: Skip handling matches the CPU `forward_t` reversal
  rule (most-recent skip first to resnet[0] of each up-block); a
  divergence in the order would crater the cosine similarity above.

## Architecture

- Per-component bundles at `gpu/unet.rs:60..199`:
  `GpuConv2d`, `GpuGroupNorm`, `GpuLayerNorm`, `GpuLinearT`,
  `GpuResnetTime` (resnet with `time_emb_proj`), `GpuAttention`
  (multi-head with optional bias on q/k/v), `GpuFeedForwardGEGLU`
  (`Linear(dim, 2*dim_ff) → chunk(2) → x * gelu(gate) →
  Linear(dim_ff, dim)`), `GpuBasicTransformerBlock`,
  `GpuTransformer2D`, `GpuUpsample2D`, `GpuDownsample2D`,
  `GpuCrossAttnDownBlock`, `GpuDownBlock`, `AnyGpuDown`,
  `GpuCrossAttnUpBlock`, `GpuUpBlock`, `AnyGpuUp`, `GpuMidBlock`.
- `GpuUNet2DConditional` at `gpu/unet.rs:262..274` holds:
  `time_proj` (host-side `Timesteps`), `time_emb_lin1` /
  `time_emb_lin2`, `conv_in`, four down-blocks, mid-block, four
  up-blocks, `conv_norm_out`, `conv_out`, frozen config, device
  handle.
- `new` at `gpu/unet.rs:296..` pops every state-dict key in CPU
  layout (`time_embedding.linear_{1,2}.*`, `conv_in.*`,
  `down_blocks.{i}.{resnets,attentions,downsamplers}.*`, …),
  enforces shapes, uploads to VRAM. The `heads = config.attention_head_dim`
  + `dim_head = out_c / heads` footgun is propagated at
  `gpu/unet.rs:307`.
- `from_module` at `gpu/unet.rs:625..631`: extracts the CPU
  state-dict and delegates to `new`.
- `forward` at `gpu/unet.rs:647..`: validates shapes, runs the
  full topology. Time-embedding stage at `gpu/unet.rs:691..697`
  uses CPU `time_proj.forward_t` then `cpu_to_gpu` + two GPU
  Linear + SiLU. Conv_in at `gpu/unet.rs:702..707`. Skips
  managed as `Vec<(CudaBuffer<f32>, [usize; 4])>` so cat() knows
  the channel count for each up-block.

Non-test production consumers:

- `ferrotorch-diffusion/src/gpu/mod.rs:36` re-exports
  `GpuUNet2DConditional`.
- `ferrotorch-diffusion/src/gpu/pipeline.rs:47,66,88` holds
  `unet: GpuUNet2DConditional` as a pipeline field; `cfg_eval` at
  `gpu/pipeline.rs:142..143` calls `self.unet.forward(...)` twice
  per inference step.
- `ferrotorch-diffusion/examples/unet_predict_dump.rs:385,393`
  imports and constructs `GpuUNet2DConditional` via `from_module`
  for the SD-1.5 UNet inference-dump binary.
- `ferrotorch-diffusion/examples/sd_pipeline_dump.rs:482` imports
  it for the end-to-end SD-1.5 GPU dump.

## Parity contract

`parity_ops = []`. The UNet is a composition of ferrotorch-gpu
kernels; per-op parity lives at the kernel layer. End-to-end vs
diffusers is checked by `tests/conformance_pretrained_diffusion.rs`
(cosine similarity ≥ 0.99 on the reference UNet output).

Critical-not-to-regress invariants (re-stated for GPU):

- **`attention_head_dim` is COUNT of heads**, not per-head dim.
  Module rustdoc at `gpu/unet.rs:32..36` documents this; the
  `new` body at `gpu/unet.rs:307` propagates it.
- **Two GELU variants** in the same crate: QuickGELU (`gpu_gelu`)
  for the CLIP MLP in `gpu/clip.rs`; exact-erf GELU
  (`gpu_gelu_erf`) for the GEGLU FF inside the UNet's
  `BasicTransformerBlock`. The module rustdoc at
  `gpu/unet.rs:39..42` calls this out.
- **GroupNorm eps**: `1e-5` for resnet / final-conv GroupNorm,
  `1e-6` for the GroupNorm-after-attention inside the
  `Transformer2DModel`. Encoded at `gpu/unet.rs:313..314`.
- **`(layers_per_block + 1)` resnets per up-block**, vs
  `layers_per_block` on the down side. Up-side iteration consumes
  the trailing N skips (most-recent first).
- **`(W^T)` storage** so every Linear is a single matmul without
  per-call transpose (`gpu_matmul_f32(x, W_t) + b_broadcast`).

## Verification

Lib tests are in `gpu/unet.rs` (`#[cfg(test)] mod tests` under
`feature = "cuda"`). Cargo invocation:

```text
cargo test -p ferrotorch-diffusion --lib --features cuda gpu::unet
```

End-to-end: `tests/conformance_pretrained_diffusion.rs` runs the
GPU UNet against the pinned SD-1.5 mirror.

No parity-sweep ops apply.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `GpuUNet2DConditional::new` at `ferrotorch-diffusion/src/gpu/unet.rs:296..`; non-test consumer: `ferrotorch-diffusion/src/gpu/unet.rs:630` `from_module` calls `Self::new(cpu.config.clone(), state, device.clone())`; production binary `ferrotorch-diffusion/examples/unet_predict_dump.rs:393` constructs the UNet via `from_module` |
| REQ-2 | SHIPPED | impl: `from_module` at `ferrotorch-diffusion/src/gpu/unet.rs:625..631`; non-test consumer: `ferrotorch-diffusion/examples/unet_predict_dump.rs:393` `GpuUNet2DConditional::from_module(unet, &device)?`; `ferrotorch-diffusion/examples/sd_pipeline_dump.rs` uses the same call |
| REQ-3 | SHIPPED | impl: `forward` at `ferrotorch-diffusion/src/gpu/unet.rs:647..` (full six-stage topology); non-test consumer: `ferrotorch-diffusion/src/gpu/pipeline.rs:142..143` in `cfg_eval` calls `self.unet.forward(&model_input, &t, ...)` twice per diffusion step |
| REQ-4 | SHIPPED | impl: footgun-aware constructor at `ferrotorch-diffusion/src/gpu/unet.rs:307` `let heads = config.attention_head_dim;`; rustdoc at `gpu/unet.rs:32..36`; non-test consumer: `new` itself is invoked by `from_module` on every production build (e.g. `ferrotorch-diffusion/examples/unet_predict_dump.rs:393`) |
| REQ-5 | SHIPPED | impl: `gpu_silu` + `gpu_gelu_erf` imports at `ferrotorch-diffusion/src/gpu/unet.rs:47..48` and rustdoc at `gpu/unet.rs:39..42`; non-test consumer: per-layer forward at `gpu/unet.rs:695..` (SiLU for time embedding) and the FeedForward path in the `Transformer2DModel` body apply both kernels on every forward; `ferrotorch-diffusion/examples/unet_predict_dump.rs:393` exercises both via the dump pipeline |
| REQ-6 | SHIPPED | impl: shape checks at `ferrotorch-diffusion/src/gpu/unet.rs:653..680`; non-test consumer: `ferrotorch-diffusion/src/gpu/pipeline.rs:142..143` invokes `forward` on the pipeline's typed `Tensor<f32>` per step, exercising the contract on every dump call |
