# GpuStableDiffusionPipeline

<!--
tier: 3-component
status: draft
baseline-pytorch: /home/doll/pytorch (HEAD)
upstream-paths:
  - diffusers/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
  - diffusers/src/diffusers/schedulers/scheduling_ddim.py
-->

## Summary

End-to-end SD-1.5 text-to-image generation pipeline composing three
VRAM-resident sub-models ([`GpuClipTextEncoder`],
[`GpuUNet2DConditional`], [`GpuVaeDecoder`]) with the CPU
[`DDIMScheduler`]. Mirrors the CPU `StableDiffusionPipeline`
op-for-op for the no-img2img / classifier-free-guidance DDIM path.
The scheduler stays on host because its math operates on the small
`[1, 4, 64, 64]` latent and round-tripping through VRAM would add
latency without changing arithmetic precision.

## Requirements

- REQ-1: `GpuStableDiffusionPipeline::new(text_encoder, unet, vae,
  scheduler, device)` composes the three GPU sub-models and the
  host scheduler into a pipeline. Returns `FerrotorchResult<Self>`;
  currently infallible but `Result`-shaped for forward-compatible
  validation.
- REQ-2: `encode_prompt(input_ids)` thinly delegates to
  `text_encoder.encode(input_ids)`, returning `[1, S, hidden_size]`.
- REQ-3: `generate(cond_embeds, uncond_embeds, init_latent,
  num_inference_steps, guidance_scale)` runs the full diffusion
  loop: `scheduler.set_timesteps` → for each step:
  `scheduler.scale_model_input` (identity for DDIM), two UNet
  forwards (uncond + cond), CFG blend `guided = uncond + scale *
  (cond - uncond)` (host f32 because UNet returns host tensors),
  `scheduler.step`. After the loop: `vae.decode(latent)` (which
  applies `1 / scaling_factor` internally). Returns
  `(final_image [1, 3, 512, 512], per_step_dumps[])`.
- REQ-4: Shape validation: `init_latent.ndim() == 4`,
  `cond_embeds.shape() == uncond_embeds.shape()`; mismatches raise
  `FerrotorchError::ShapeMismatch`.
- REQ-5: `PipelineStepDump` (re-exported from
  `crate::pipeline::PipelineStepDump`) records per-step
  diagnostics: `step`, `timestep`, `noise_pred_uncond`,
  `noise_pred_cond`, `guided_noise`, `latent_after_step`. One dump
  per inference step; emitted in iteration order.

## Acceptance Criteria

- [x] AC-1: CFG blend uses
  `ferrotorch_core::grad_fns::arithmetic::{add, mul, sub}` for the
  host arithmetic (`gpu/pipeline.rs`).
- [x] AC-2: `init_latent` is multiplied by
  `scheduler.init_noise_sigma()` before the first step (kept
  explicit even though DDIM SD-1.5 sigma is 1.0) at
  `gpu/pipeline.rs`.
- [x] AC-3: VAE decode is `self.vae.decode(&latent)` (decode
  applies the `1 / scaling_factor` divide internally) at
  `gpu/pipeline.rs`.

## Architecture

- `GpuStableDiffusionPipeline in gpu/pipeline.rs` holds
  `text_encoder: GpuClipTextEncoder`, `unet: GpuUNet2DConditional`,
  `vae: GpuVaeDecoder`, `scheduler: DDIMScheduler`, `_device:
  GpuDevice` (kept for completeness; per-call ops route through the
  sub-models' own device clones).
- `new in gpu/pipeline.rs` is a field-pack returning
  `FerrotorchResult<Self>`.
- `encode_prompt in gpu/pipeline.rs` delegates to
  `text_encoder.encode`.
- `timestep_tensor in gpu/pipeline.rs` builds the
  `[B]` host f32 timestep tensor the UNet sinusoidal projection
  consumes.
- `cfg_eval in gpu/pipeline.rs` is the per-step inner
  loop: builds the timestep tensor, calls
  `scheduler.scale_model_input` (identity for DDIM), runs two GPU
  UNet forwards, then computes `guided = uncond + gs * (cond -
  uncond)` via `add` / `sub` / `mul` from
  `grad_fns::arithmetic`. Returns the three host tensors.
- `generate in gpu/pipeline.rs` is the outer loop:
  validates shapes, snapshots `set_timesteps(...)` to a `Vec`
  (so the loop body can call `&mut self` for `step`), multiplies
  `init_latent` by `scheduler.init_noise_sigma()`, iterates the
  timesteps running `cfg_eval` + `scheduler.step` per step,
  collects the per-step dumps. After the loop: `vae.decode`.
- Module docstring at `gpu/pipeline.rs` documents
  determinism (init_latent comes from the caller; rust's PRNG ≠
  `torch.Generator`), the host-side scheduler choice, and the
  download-once-per-forward residency model of the sub-models.

Non-test production consumers:

- `ferrotorch-diffusion/src/gpu/mod.rs:35` re-exports
  `GpuStableDiffusionPipeline`.
- `ferrotorch-diffusion/examples/sd_pipeline_dump.rs` imports
  `GpuClipTextEncoder, GpuStableDiffusionPipeline,
  GpuUNet2DConditional, GpuVaeDecoder` and constructs the pipeline
  for the end-to-end SD-1.5 GPU dump binary.

## Parity contract

`parity_ops = []`. The pipeline is composition only. The per-step
CFG arithmetic uses ferrotorch-core's `add` / `sub` / `mul`
(already parity-checked); the three GPU sub-models each carry their
own forward parity. End-to-end vs diffusers'
`StableDiffusionPipeline` is checked by
`tests/conformance_sd_pipeline.rs` (and indirectly by
`tests/conformance_pretrained_diffusion.rs`).

Determinism caveat: rust's `StdRng` does NOT match
`torch.Generator(device='cpu').manual_seed(seed)`. The pipeline
takes `init_latent` from the caller, so deterministic comparisons
against Python require the caller to feed the SAME starting
latent — typically read from the pinned
`ferrotorch/sd-v1-5-generation-trajectory` mirror.

The CFG blend formula `guided = uncond + scale * (cond - uncond)`
is the standard diffusers convention; the module docstring at
`gpu/pipeline.rs` documents it explicitly.

## Verification

Lib tests are in the GPU sub-models; this composition module has no
unit tests of its own (it has no novel arithmetic).

End-to-end: `tests/conformance_sd_pipeline.rs` runs the full
pipeline on the pinned 4-step DDIM recipe and compares per-step
latents + final image against the diffusers reference within
tolerance.

No parity-sweep ops apply.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `GpuStableDiffusionPipeline::new` at `ferrotorch-diffusion/src/gpu/pipeline.rs:86..100`; non-test consumer: `ferrotorch-diffusion/examples/sd_pipeline_dump.rs` constructs the pipeline via this constructor for the dump binary |
| REQ-2 | SHIPPED | impl: `encode_prompt` at `ferrotorch-diffusion/src/gpu/pipeline.rs:107..109`; non-test consumer: same example calls `pipeline.encode_prompt(...)` before invoking `generate` |
| REQ-3 | SHIPPED | impl: `generate in ferrotorch-diffusion/src/gpu/pipeline.rs` (full loop) and `cfg_eval in gpu/pipeline.rs`; non-test consumer: `ferrotorch-diffusion/examples/sd_pipeline_dump.rs` invokes `generate(...)` to produce the dump artifact |
| REQ-4 | SHIPPED | impl: shape checks at `generate in ferrotorch-diffusion/src/gpu/pipeline.rs`; non-test consumer: same `generate` consumes them on every dump call; the validation contract mirrors the CPU `StableDiffusionPipeline::generate` shape check at `pipeline in pipeline.rs` |
| REQ-5 | SHIPPED | impl: `PipelineStepDump` is constructed at `ferrotorch-diffusion/src/gpu/pipeline.rs:215..222`; non-test consumer: the dump example writes each `PipelineStepDump` to disk for diffusion-trajectory audit |
