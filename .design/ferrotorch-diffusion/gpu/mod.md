# gpu module (SD-1.5 GPU forward paths)

<!--
tier: 3-component
status: draft
baseline-pytorch: /home/doll/pytorch (HEAD)
upstream-paths:
  - diffusers/src/diffusers/models/unets/unet_2d_condition.py
  - diffusers/src/diffusers/models/autoencoders/autoencoder_kl.py
  - diffusers/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
  - transformers/src/transformers/models/clip/modeling_clip.py
-->

## Summary

`feature = "cuda"`-gated GPU twin of the CPU SD-1.5 sub-model
families. Hosts five sub-modules (`clip`, `unet`, `vae`,
`vae_encoder`, `pipeline`) and re-exports their top-level public types
so callers can `use ferrotorch_diffusion::gpu::{...};` without
referring to per-sub-module paths.

## Requirements

- REQ-1: `pub mod` declarations expose the five GPU sub-modules
  (`clip`, `pipeline`, `unet`, `vae`, `vae_encoder`); they are
  conditionally compiled under `feature = "cuda"` via the parent
  `#[cfg(feature = "cuda")] pub mod gpu;` gate in `lib.rs`.
- REQ-2: `pub use` re-exports five top-level types
  (`GpuClipTextEncoder`, `GpuStableDiffusionPipeline`,
  `GpuUNet2DConditional`, `GpuVaeDecoder`, `GpuVaeEncoder`) so
  external callers do not need to depend on per-sub-module paths.
- REQ-3: Module-level docstring describes the architecture: every
  sub-model holds weights in VRAM and downloads only the final
  forward result to host f32, with one-shot weight uploads at
  construction.

## Acceptance Criteria

- [x] AC-1: `cargo doc -p ferrotorch-diffusion --features cuda`
  renders the gpu module page with all five sub-types linked.
- [x] AC-2: `use ferrotorch_diffusion::gpu::GpuStableDiffusionPipeline;`
  resolves at the crate root.

## Architecture

This is a one-liner-per-sub-module re-export hub. No tensor math.

- `#![cfg(feature = "cuda")]` at `gpu/mod.rs` gates the entire
  module so CPU-only builds drop the GPU code at compile time.
- `pub mod` declarations at `gpu/mod.rs` register the five
  sub-modules (alphabetical).
- `pub use` block at `gpu/mod.rs` re-exports each
  sub-module's top-level public type.
- Module rustdoc at `gpu/mod.rs` describes per-sub-model
  scope, gating, and the in-VRAM-residency contract.

Non-test production consumers:

- `ferrotorch-diffusion/src/gpu/pipeline.rs` imports
  `GpuClipTextEncoder`, `GpuUNet2DConditional`, and `GpuVaeDecoder`
  through these re-exports.
- `ferrotorch-diffusion/examples/sd_pipeline_dump.rs:482` imports
  `GpuClipTextEncoder, GpuStableDiffusionPipeline,
  GpuUNet2DConditional, GpuVaeDecoder` for the end-to-end SD-1.5
  GPU dump binary.
- `ferrotorch-diffusion/examples/unet_predict_dump.rs:385` imports
  `GpuUNet2DConditional` through the same path.
- `ferrotorch-diffusion/examples/vae_decode_dump.rs:307` and
  `ferrotorch-diffusion/examples/clip_text_encode_dump.rs:362`
  consume their respective re-exports.

## Parity contract

`parity_ops = []`. The module is composition only; the GPU
sub-models each carry their own (op-by-op) parity contract via
the kernels they dispatch through (`ferrotorch-gpu::gpu_conv2d_f32`,
`gpu_layernorm`, `gpu_softmax`, `gpu_bmm_f32`, `gpu_matmul_f32`,
etc.). The `gpu/mod.rs` file itself has no tensor surface to test.

## Verification

No unit tests in this file (it has no executable code beyond
declarations). The five sub-modules each carry their own
`#[cfg(test)] mod tests`, exercised by `cargo test -p
ferrotorch-diffusion --lib --features cuda`. The end-to-end
`conformance_pretrained_diffusion.rs` integration test exercises
the full re-export surface against the pinned SD-1.5 mirror.

No parity-sweep ops apply.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub mod` block at `ferrotorch-diffusion/src/gpu/mod.rs`; non-test consumer: `ferrotorch-diffusion/src/gpu/pipeline.rs` resolves `crate::gpu::clip::GpuClipTextEncoder` / `unet::GpuUNet2DConditional` / `vae::GpuVaeDecoder` through these declarations |
| REQ-2 | SHIPPED | impl: `pub use` block at `ferrotorch-diffusion/src/gpu/mod.rs`; non-test consumer: `ferrotorch-diffusion/examples/sd_pipeline_dump.rs` uses the re-exports to construct the full GPU pipeline |
| REQ-3 | SHIPPED | impl: module rustdoc at `ferrotorch-diffusion/src/gpu/mod.rs`; non-test consumer: `cargo doc -p ferrotorch-diffusion --features cuda` renders the module landing page and the five sub-modules link back through this rustdoc |
