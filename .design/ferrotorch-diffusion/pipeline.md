# StableDiffusionPipeline

<!--
tier: 3-component
status: draft
baseline-pytorch: /home/doll/pytorch (HEAD)
upstream-paths:
  - diffusers/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
  - diffusers/src/diffusers/schedulers/scheduling_ddim.py
-->

## Summary

End-to-end fixed-seed text-to-image generation pipeline that composes
the four SD-1.5 sub-models (`ClipTextEncoder`, `UNet2DConditionModel`,
`VaeDecoder`, `DDIMScheduler`). Mirrors
`diffusers.StableDiffusionPipeline.__call__` for the deterministic
DDIM path with classifier-free guidance, but takes the initial latent
from the caller rather than seeding a Rust-side PRNG (the noise
sequence is not reproducible between Rust's `StdRng` and
`torch.Generator(device='cpu').manual_seed(seed)`).

## Requirements

- REQ-1: `StableDiffusionPipeline::new` constructs a pipeline from
  the four sub-models (text encoder, UNet, VAE, scheduler).
- REQ-2: `encode_prompt(input_ids)` runs the CLIP text encoder on a
  padded `[S]` token-id sequence and returns `[1, S, hidden_size]`.
- REQ-3: `generate(cond_embeds, uncond_embeds, init_latent,
  num_inference_steps, guidance_scale)` runs the full diffusion loop:
  scheduler.set_timesteps -> for each step: scale_model_input, two
  UNet forwards (uncond + cond), CFG blend, scheduler.step -> VAE
  decode_with_scaling.
- REQ-4: `PipelineStepDump` captures per-step diagnostics (the two
  noise predictions, the CFG-guided noise, the post-step latent) so
  the dump example can pinpoint divergence at each step.
- REQ-5: Input shape validation: `init_latent` must be rank-4,
  `cond_embeds.shape() == uncond_embeds.shape()`; mismatches raise
  `FerrotorchError::ShapeMismatch`.

## Acceptance Criteria

- [x] AC-1: `pipeline_constructs` test builds a tiny pipeline end to
  end (`pipeline in pipeline.rs`).
- [x] AC-2: Per-step CFG math `guided = uncond + scale * (cond -
  uncond)` is expressed via `add`/`sub`/`mul` from
  `grad_fns::arithmetic` (`pipeline in pipeline.rs`).
- [x] AC-3: VAE decode applies the scaling factor by calling
  `vae.decode_with_scaling(latent)` (`pipeline in pipeline.rs`).

## Architecture

Composition only — no new tensor math beyond the CFG blend.

- `StableDiffusionPipeline<T: Float>` at `StableDiffusionPipeline in pipeline.rs` is a
  plain struct holding the four sub-models. `text_encoder`, `unet`,
  `vae` are `&self`-callable; `scheduler` requires `&mut self`
  because `set_timesteps` mutates the cached timesteps vector.
- `PipelineStepDump<T: Float>` at `PipelineStepDump in pipeline.rs` is the
  per-step diagnostic record.
- `new` at `pipeline in pipeline.rs` is currently a plain field-pack;
  per-scheduler validation is deferred to the call sites that need
  it (epsilon prediction is implicit because `DDIMScheduler` only
  ships that path today).
- `encode_prompt` at `pipeline in pipeline.rs` thinly delegates to
  `ClipTextEncoder::forward_from_ids`.
- `cfg_eval` at `pipeline in pipeline.rs` is the per-step inner loop:
  builds the `[B]` timestep tensor, calls
  `scheduler.scale_model_input` (identity for DDIM), runs two UNet
  forwards, computes `guided = uncond + gs * (cond - uncond)` via
  `add`/`sub`/`mul`.
- `generate` at `pipeline in pipeline.rs` is the outer loop:
  validates shapes, scales `init_latent` by
  `scheduler.init_noise_sigma()` (1.0 for SD-1.5 DDIM, but the
  multiplication is kept for forward-compat), iterates the timesteps,
  runs `cfg_eval` + `scheduler.step` per step, then calls
  `vae.decode_with_scaling`.

Non-test production consumers:

- `ferrotorch-diffusion/examples/sd_pipeline_dump.rs` constructs the
  pipeline and calls `generate`.
- `ferrotorch-diffusion/src/gpu/pipeline.rs` imports
  `PipelineStepDump` for the GPU mirror's dump compatibility.
- `ferrotorch-diffusion/tests/conformance_sd_pipeline.rs` exercises
  the full pipeline against pinned reference dumps (test, but the
  GPU mirror reuses `PipelineStepDump` in production).

## Parity contract

`parity_ops = []`. The pipeline is a composition of already-verified
sub-models. The numerical contract (CFG blend, `scale_model_input`
identity for DDIM, `vae.decode_with_scaling` applying `1 /
scaling_factor`) follows diffusers' `pipeline_stable_diffusion.py`
exactly for the no-img2img / deterministic-DDIM path.

## Verification

- `pipeline_constructs` (`pipeline in pipeline.rs`) — builds a tiny
  pipeline end-to-end on a `sample_size=8` config.
- `conformance_sd_pipeline.rs` integration test runs the pipeline on
  the pinned 4-step recipe and compares the per-step latent + final
  image against the reference HF dump within tolerance.

No parity-sweep ops (composition-only).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `StableDiffusionPipeline::new` at `new in ferrotorch-diffusion/src/pipeline.rs`; non-test consumer: `new in ferrotorch-diffusion/src/gpu/pipeline.rs` re-uses the `PipelineStepDump` produced by this constructor's `generate` method |
| REQ-2 | SHIPPED | impl: `encode_prompt` at `ferrotorch-diffusion/src/pipeline.rs:101..103`; non-test consumer: `ferrotorch-diffusion/examples/sd_pipeline_dump.rs` invokes it as the first stage of the dump pipeline |
| REQ-3 | SHIPPED | impl: `generate` at `ferrotorch-diffusion/src/pipeline.rs:167..229`; non-test consumer: `ferrotorch-diffusion/examples/sd_pipeline_dump.rs` calls `generate(...)` to produce the dump artifact |
| REQ-4 | SHIPPED | impl: `PipelineStepDump<T>` at `PipelineStepDump in ferrotorch-diffusion/src/pipeline.rs`; non-test consumer: `ferrotorch-diffusion/src/gpu/pipeline.rs` constructs `PipelineStepDump` values for the GPU dump |
| REQ-5 | SHIPPED | impl: shape checks at `ferrotorch-diffusion/src/pipeline.rs:175..191`; non-test consumer: `ferrotorch-diffusion/src/gpu/pipeline.rs:181..193` mirrors the same shape contract for the GPU pipeline |
