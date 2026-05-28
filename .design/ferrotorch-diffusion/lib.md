# ferrotorch-diffusion crate root

<!--
tier: 3-component
status: draft
baseline-pytorch: /home/doll/pytorch (HEAD)
upstream-paths:
  - diffusers/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
  - diffusers/src/diffusers/models/autoencoders/autoencoder_kl.py
  - diffusers/src/diffusers/models/unets/unet_2d_condition.py
  - transformers/src/transformers/models/clip/modeling_clip.py
-->

## Summary

Crate-level integration surface for the Stable-Diffusion v1.5 model
family. Re-exports the CLIP text encoder, the UNet2DConditionModel, the
VAE encoder/decoder, the DDIM scheduler, and the end-to-end
`StableDiffusionPipeline`. The crate composes upstream
HuggingFace `diffusers` + `transformers` SD-1.5 components in Rust;
the underlying ops (`Conv2d`, `Linear`, `GroupNorm`, `LayerNorm`,
`SiLU`, `GELU`, `Embedding`, multi-head attention) come from
`ferrotorch-core` and `ferrotorch-nn`. There is no direct PyTorch C++
counterpart for the crate root; the upstream contract is the diffusers
Python pipeline.

## Requirements

- REQ-1: The crate exposes a single top-level `pub mod` per SD-1.5
  sub-system: `attention`, `blocks`, `clip_text_encoder`, `config`,
  `pipeline`, `resnet_block_time`, `safetensors_loader`, `scheduler`,
  `time_embedding`, `unet`, `unet_config`, `vae`, `vae_encoder`, with
  optional `gpu` behind `feature = "cuda"`.
- REQ-2: A user-facing `pub use` set re-exports the principal types
  (`StableDiffusionPipeline`, `UNet2DConditionModel`, `VaeDecoder`,
  `VaeEncoder`, `ClipTextEncoder`, `DDIMScheduler`, configs, and the
  block primitives) at the crate root so downstream callers don't
  need module paths.
- REQ-3: Crate-level lint baseline matches the rest of ferrotorch:
  `deny(unsafe_code)`, `deny(missing_docs)`, `warn(clippy::pedantic)`,
  with the same per-lint `#![allow(..)]` exceptions
  (`cast_*`, `module_name_repetitions`, `too_many_lines`,
  `too_many_arguments`) the other model crates use.
- REQ-4: The crate-level `//!` doc-comment documents the two
  shipped pipelines (VAE decoder + UNet2DConditionModel) and the
  ResnetBlock2DTime + Transformer2DModel forward recipes that the
  composing modules implement.

## Acceptance Criteria

- [x] AC-1: `cargo check -p ferrotorch-diffusion` compiles.
- [x] AC-2: `pub use` set at `lib.rs` lines 116..139 includes every
  type listed in REQ-2.
- [x] AC-3: Lint baseline at `lib.rs` lines 6..59 matches
  `ferrotorch-bert` / `ferrotorch-whisper`.
- [x] AC-4: Module-level doc-comment (`lib.rs` lines 61..98) documents
  VAE decoder, UNet2DConditionModel, ResnetBlock2DTime, and
  Transformer2DModel forward recipes.

## Architecture

The crate is a thin composition layer. `lib.rs` is responsible for:

- **Module wiring** at `lib.rs` lines 100..114: declares every public
  sub-module. `gpu` is gated behind `#[cfg(feature = "cuda")]` so the
  default CPU-only build stays small.
- **Re-export surface** at `lib.rs` lines 116..139: the curated
  `pub use` set hoists the top-level types so external callers can
  write `ferrotorch_diffusion::StableDiffusionPipeline` rather than
  `ferrotorch_diffusion::pipeline::StableDiffusionPipeline`.
- **Lint baseline** at `lib.rs` lines 6..59: explicitly enumerated
  allows are sourced from the same template used by
  `ferrotorch-bert/src/lib.rs` and `ferrotorch-whisper/src/lib.rs`,
  keeping the rule set unified across model crates.

Non-test production consumers:

- `ferrotorch-diffusion/examples/sd_pipeline_dump.rs` and
  `examples/unet_predict_dump.rs` consume the top-level re-exports.
- `ferrotorch-hub/src/registry.rs` references `ClipTextEncoder` via
  the crate-level re-export.
- `ferrotorch-diffusion/src/gpu/pipeline.rs` consumes
  `pipeline::PipelineStepDump` through the crate re-export.

## Parity contract

No direct parity ops — this file is the wiring layer. Per-op parity
is covered by the leaf modules (e.g. `silu`, `gelu`, `softmax`,
`conv2d`) in `ferrotorch-core` and `ferrotorch-nn`.

## Verification

`cargo build -p ferrotorch-diffusion` builds the module graph; the
re-export set is exercised by every `examples/*_dump.rs` binary in
this crate. There are no unit tests in `lib.rs` itself — verification
is by `cargo check` + downstream crate compilation.

Smoke: `cargo check -p ferrotorch-diffusion 2>&1 | tail -3` returns
`Finished` (no parity-sweep applies — `parity_ops = []`).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub mod` block at `ferrotorch-diffusion/src/lib.rs`; non-test consumer: `ferrotorch-diffusion/src/safetensors_loader.rs` imports six of the modules to wire the loaders |
| REQ-2 | SHIPPED | impl: `pub use` re-export block at `ferrotorch-diffusion/src/lib.rs:116..139`; non-test consumer: `ferrotorch-hub/src/registry.rs` references `ClipTextEncoder` through the re-export |
| REQ-3 | SHIPPED | impl: `#![deny(...)]` and `#![allow(...)]` attribute block at `ferrotorch-diffusion/src/lib.rs:6..59`; non-test consumer: `cargo clippy -p ferrotorch-diffusion --lib -- -D warnings` enforces it on every build of every downstream crate |
| REQ-4 | SHIPPED | impl: `//!` doc-comment at `ferrotorch-diffusion/src/lib.rs`; non-test consumer: `cargo doc -p ferrotorch-diffusion` renders this docstring as the published crate landing page |
