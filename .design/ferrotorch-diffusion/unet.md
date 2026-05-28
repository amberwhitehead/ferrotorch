# UNet2DConditionModel (CPU)

<!--
tier: 3-component
status: draft
baseline-pytorch: /home/doll/pytorch (HEAD)
upstream-paths:
  - diffusers/src/diffusers/models/unets/unet_2d_condition.py
  - diffusers/src/diffusers/models/unets/unet_2d_blocks.py
-->

## Summary

CPU SD-1.5 UNet2DConditionModel forward path. Mirrors
`diffusers.models.unets.UNet2DConditionModel.forward(sample,
timestep, encoder_hidden_states).sample` for
`runwayml/stable-diffusion-v1-5` — the `[CrossAttn × 3,
DownBlock2D]` / `[UpBlock2D, CrossAttn × 3]` / `UNetMidBlock2DCrossAttn`
topology with sinusoidal time projection + 2-layer MLP, ResnetBlock2DTime
+ Transformer2DModel cross-attention blocks, and four-level
down/up sampling.

## Requirements

- REQ-1: `CrossAttnDownBlock2D<T>` composes
  `layers_per_block × (ResnetBlock2DTime + Transformer2DModel)` plus
  an optional `Downsample2D`. `forward_t(x, temb, ehs)` returns
  `(output, skips[])` where every resnet+attn pair emits one skip,
  and the downsampler (if present) emits a trailing skip. Diffusers
  footgun: `attention_head_dim` config is the *number of heads*, not
  the per-head dimension (`dim_head = out_channels / num_heads`).
- REQ-2: `DownBlock2D<T>` is the attn-free variant:
  `layers_per_block × ResnetBlock2DTime` + optional `Downsample2D`,
  same skip protocol.
- REQ-3: `UNetMidBlock2DCrossAttn<T>` is `resnet0 →
  layers_per_block × (Transformer2DModel + ResnetBlock2DTime)`
  matching `UNetMidBlock2DCrossAttn`.
- REQ-4: `CrossAttnUpBlock2D<T>` composes
  `(layers_per_block + 1) × (ResnetBlock2DTime + Transformer2DModel)`
  + optional `Upsample2D`. `forward_t(x, skips, temb, ehs)` pops
  one skip per resnet, concatenates along the channel axis BEFORE
  each resnet, and validates `skips.len() == num_resnets()`.
- REQ-5: `UpBlock2D<T>` is the attn-free variant:
  `(layers_per_block + 1) × ResnetBlock2DTime` + optional
  `Upsample2D`, same skip-pop protocol.
- REQ-6: `AnyDownBlock<T>` / `AnyUpBlock<T>` enums dispatch to the
  cross-attn or plain variants based on
  `config.down_block_has_attn[i]` / `config.up_block_has_attn[i]`.
- REQ-7: `UNet2DConditionModel<T>::forward_t(sample, timesteps,
  encoder_hidden_states)` runs the full forward: `time_proj →
  time_embedding → conv_in → push initial skip → down_blocks →
  mid_block → up_blocks (each popping its trailing skips, reversed
  to feed resnet[0] the most-recent skip first) → conv_norm_out →
  SiLU → conv_out`. Returns `[B, out_channels, H, W]`.
- REQ-8: `Module<T>::load_state_dict` accepts the canonical
  diffusers layout with the seven prefixes (`time_embedding.`,
  `conv_in.`, `down_blocks.`, `mid_block.`, `up_blocks.`,
  `conv_norm_out.`, `conv_out.`); strict mode rejects others.

## Acceptance Criteria

- [x] AC-1: `unet_forward_shape` runs a tiny config (`bocs=[16, 32,
  64, 64]`, `S=7`, `cross_attention_dim=24`) end-to-end on a
  `[1, 4, 8, 8]` sample and gets `[1, 4, 8, 8]` back
  (`unet in unet.rs`).
- [x] AC-2: `unet_named_parameters_includes_canonical_keys`
  enumerates eleven canonical key patterns
  (`time_embedding.linear_1.weight`,
  `down_blocks.0.resnets.0.norm1.weight`,
  `down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight`,
  `mid_block.attentions.0.transformer_blocks.0.attn2.to_v.weight`,
  `up_blocks.1.attentions.0.transformer_blocks.0.ff.net.0.proj.weight`,
  `conv_out.bias`, …) (`unet in unet.rs`).

## Architecture

- `CrossAttnDownBlock2D<T>` at `CrossAttnDownBlock2D in unet.rs`, `new` at
  `unet in unet.rs` (heads=`attention_head_dim`, `dim_head =
  out_channels / heads`), `forward_t` at `unet.rs:147..165`.
- `DownBlock2D<T>` at `DownBlock2D in unet.rs`, `new` at
  `unet in unet.rs`, `forward_t in unet.rs`.
- `UNetMidBlock2DCrossAttn<T>` at `UNetMidBlock2DCrossAttn in unet.rs`, `new` at
  `unet in unet.rs`, `forward_t in unet.rs`.
- `CrossAttnUpBlock2D<T>` at `CrossAttnUpBlock2D in unet.rs`, `new` at
  `unet in unet.rs`, `forward_t in unet.rs`.
- `UpBlock2D<T>` at `UpBlock2D in unet.rs`, `new` at
  `unet in unet.rs`, `forward_t in unet.rs`.
- `AnyDownBlock<T>` at `AnyDownBlock in unet.rs` (enum + `forward_t` /
  parameters / named_parameters / load dispatch).
- `AnyUpBlock<T>` at `AnyUpBlock in unet.rs` (mirror enum).
- `UNet2DConditionModel<T>` at `UNet2DConditionModel in unet.rs`. `new` at
  `unet in unet.rs` builds:
  - `Timesteps::new(bocs[0], flip_sin_to_cos, freq_shift)` (the
    parameter-free sinusoidal encoding);
  - `TimestepEmbedding::<T>::new(bocs[0], temb_channels)` (the
    SiLU+2×Linear MLP);
  - `conv_in (in_channels → bocs[0])`;
  - four down-blocks (CrossAttn or Plain per
    `cfg.down_block_has_attn[i]`, `add_downsample` on all but the
    last);
  - mid-block at the deepest channels (`bocs[-1]`);
  - four up-blocks iterated over `reversed = bocs.reverse()`
    (`prev_up = mid_channels` initially, `in_c =
    reversed[(i+1).min(N-1)]`, `add_upsample = !is_final`,
    `layers_per_block + 1` resnets per block);
  - `conv_norm_out (GroupNorm(groups, bocs[0], eps=1e-5))` and
    `conv_out (bocs[0] → out_channels)`.
- `forward_t in unet.rs` runs the six-stage forward.
  Skip handling: `skips = [conv_in_output]`; each down-block
  appends; each up-block pops its trailing N resnet count, reverses
  the popped list, and hands them to `block.forward_t`. The reversal
  is the diffusers convention (most-recent skip first).
- `Module<T>::forward` at `unet in unet.rs` returns an
  `InvalidArgument` error pointing callers at `forward_t`; the
  multi-argument signature is the only valid forward path.
- `load_state_dict` at `unet in unet.rs` splits by the seven
  diffusers prefixes; strict mode rejects others.

Non-test production consumers:

- `ferrotorch-diffusion/src/safetensors_loader.rs:175`
  `load_unet` calls `UNet2DConditionModel::<T>::new(cfg)` and
  `unet.load_hf_state_dict(&state, strict)`.
- `ferrotorch-diffusion/src/pipeline.rs:73` `StableDiffusionPipeline`
  holds a `UNet2DConditionModel<T>` field; the per-step `cfg_eval`
  calls `unet.forward_t(...)` on each timestep.
- `ferrotorch-diffusion/src/gpu/unet.rs:625..631`
  `GpuUNet2DConditional::from_module(cpu: &UNet2DConditionModel<f32>,
  ...)` consumes the CPU module's `state_dict()` to build the GPU
  twin.
- `ferrotorch-diffusion/examples/unet_predict_dump.rs:292` loads via
  `load_unet` and dumps per-step UNet outputs.

## Parity contract

`parity_ops = []`. The UNet is a composition; per-op parity is
exercised at the kernel layer (Conv2d, GroupNorm, LayerNorm,
matmul, softmax, …). End-to-end numerical parity vs diffusers is
checked by `tests/conformance_pretrained_diffusion.rs` (cosine
similarity ≥ 0.99 against the pinned reference UNet output).

The diffusers footguns that must NOT regress:

- `attention_head_dim` is the COUNT of heads, not the dim — the
  comment at `unet in unet.rs` documents this. The actual per-head
  dim is `out_channels / heads`.
- Up-block skip order is most-recent-first (the loop at
  `unet in unet.rs` reverses `popped` before handing to
  `block.forward_t`).
- `(layers_per_block + 1)` resnets per up-block (vs
  `layers_per_block` on the down side) — explicit at
  `unet in unet.rs`, `unet in unet.rs`.
- Mid-block channels = `bocs[-1]` (deepest).
- `flip_sin_to_cos = true` and `freq_shift = 0` are SD-1.5 defaults
  consumed at `unet in unet.rs`.

## Verification

Lib tests at `unet in unet.rs`:

- `unet_forward_shape` — full forward through the four-block stack
  (`unet in unet.rs`).
- `unet_named_parameters_includes_canonical_keys` — eleven canonical
  diffusers keys exposed (`unet in unet.rs`).

Integration: `tests/conformance_pretrained_diffusion.rs` runs the
pinned SD-1.5 UNet against the reference outputs.

No parity-sweep ops apply (composition module).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `CrossAttnDownBlock2D<T>` at `CrossAttnDownBlock2D in ferrotorch-diffusion/src/unet.rs` and `forward_t in ferrotorch-diffusion/src/unet.rs`; non-test consumer: `AnyDownBlock::CrossAttn` variant at `CrossAttn in ferrotorch-diffusion/src/unet.rs` is built by `UNet2DConditionModel::new` for every attn down-block |
| REQ-2 | SHIPPED | impl: `DownBlock2D<T>` at `DownBlock2D in ferrotorch-diffusion/src/unet.rs` and `forward_t in ferrotorch-diffusion/src/unet.rs`; non-test consumer: `AnyDownBlock::Plain` variant at `Plain in ferrotorch-diffusion/src/unet.rs` is built by `UNet2DConditionModel::new` for the final attn-free down-block |
| REQ-3 | SHIPPED | impl: `UNetMidBlock2DCrossAttn<T>` at `UNetMidBlock2DCrossAttn in ferrotorch-diffusion/src/unet.rs` and `forward_t in ferrotorch-diffusion/src/unet.rs`; non-test consumer: `UNet2DConditionModel::new` at `new in ferrotorch-diffusion/src/unet.rs` constructs the mid block; `forward_t in unet.rs` invokes it per forward |
| REQ-4 | SHIPPED | impl: `CrossAttnUpBlock2D<T>` at `CrossAttnUpBlock2D in ferrotorch-diffusion/src/unet.rs` and `forward_t in ferrotorch-diffusion/src/unet.rs`; non-test consumer: `AnyUpBlock::CrossAttn` variant at `CrossAttn in ferrotorch-diffusion/src/unet.rs` is built by `UNet2DConditionModel::new` for every attn up-block |
| REQ-5 | SHIPPED | impl: `UpBlock2D<T>` at `UpBlock2D in ferrotorch-diffusion/src/unet.rs` and `forward_t in ferrotorch-diffusion/src/unet.rs`; non-test consumer: `AnyUpBlock::Plain` variant at `Plain in ferrotorch-diffusion/src/unet.rs` is built for the first up-block (attn-free) |
| REQ-6 | SHIPPED | impl: `AnyDownBlock<T>` at `AnyDownBlock in ferrotorch-diffusion/src/unet.rs` and `AnyUpBlock<T>` at `AnyUpBlock in ferrotorch-diffusion/src/unet.rs`; non-test consumer: `UNet2DConditionModel::new` at `new in ferrotorch-diffusion/src/unet.rs,1216` dispatches by `cfg.down_block_has_attn[i]` / `cfg.up_block_has_attn[i]` |
| REQ-7 | SHIPPED | impl: `UNet2DConditionModel::forward_t` at `forward_t in ferrotorch-diffusion/src/unet.rs`; non-test consumer: `forward_t in ferrotorch-diffusion/src/pipeline.rs` in `cfg_eval` calls `self.unet.forward_t(&model_input, &t, ...)` twice per diffusion step |
| REQ-8 | SHIPPED | impl: `Module<T>::load_state_dict` at `ferrotorch-diffusion/src/unet.rs:1444..1486`; non-test consumer: `ferrotorch-diffusion/src/safetensors_loader.rs:146` `UNet2DConditionModel::load_hf_state_dict` calls `self.load_state_dict(&remapped, strict)` after stripping the `unet.` prefix |
