# Timesteps + TimestepEmbedding

<!--
tier: 3-component
status: draft
baseline-pytorch: /home/doll/pytorch (HEAD)
upstream-paths:
  - diffusers/src/diffusers/models/embeddings.py
-->

## Summary

Two modules that match `diffusers.models.embeddings.{Timesteps,
TimestepEmbedding}` for SD-1.5's settings
(`flip_sin_to_cos = true`, `downscale_freq_shift = 0`,
`max_period = 10000`). `Timesteps` is the parameter-free sinusoidal
positional encoding from a scalar timestep; `TimestepEmbedding` is
the `Linear → SiLU → Linear` MLP that follows it inside the UNet.

## Requirements

- REQ-1: `Timesteps::new(num_channels, flip_sin_to_cos,
  downscale_freq_shift)` constructs a parameter-free module that
  produces `[B, num_channels]` from a `[B]` timestep input.
  `num_channels` must be a positive even integer.
- REQ-2: `Timesteps::forward_t` implements the diffusers recipe
  `freqs = exp(-ln(max_period) * arange(half) / (half - shift))`;
  `args = t * freqs`; `out = cat([cos(args), sin(args)], dim=-1)` for
  `flip_sin_to_cos = true` (SD default), else `cat([sin, cos])`.
- REQ-3: `TimestepEmbedding::new(in_channels, time_emb_dim)`
  constructs the MLP `Linear(in_channels, time_emb_dim) → SiLU →
  Linear(time_emb_dim, time_emb_dim)` (both biases on, matching
  diffusers).
- REQ-4: `TimestepEmbedding`'s `Module::named_parameters` produces
  the diffusers state-dict layout `linear_1.{weight,bias}` and
  `linear_2.{weight,bias}`.
- REQ-5: `Timesteps` is parameter-free: `parameters()` is empty;
  `train`/`eval` are no-ops; `load_state_dict` accepts (and ignores)
  any state.

## Acceptance Criteria

- [x] AC-1: `timesteps_shape_flip_true` returns `[3, 8]` for input
  `[3]` and at `t=0` the cos-half is all ones, sin-half all zeros
  (`time_embedding.rs:298..317`).
- [x] AC-2: `timesteps_rejects_odd_channels` returns Err for
  `num_channels = 7` (`time_embedding.rs:319..322`).
- [x] AC-3: `timestep_embedding_shapes` produces `[1, 16]` from
  `[1, 8]` (`time_embedding.rs:324..330`).
- [x] AC-4: `timestep_embedding_named_parameters` lists all four
  parameter names with the diffusers layout
  (`time_embedding.rs:332..344`).

## Architecture

- `Timesteps` (`time_embedding.rs:37..48`) — public-field struct
  carrying `num_channels`, `flip_sin_to_cos`, `downscale_freq_shift`,
  `max_period`. No parameters.
- `Timesteps::new` (`time_embedding.rs:57..75`) validates that
  `num_channels` is positive and even, then hard-codes
  `max_period = 10000.0`.
- `Timesteps::forward_t` (`time_embedding.rs:87..152`) operates
  entirely in `f64` for the per-frequency products, then casts back
  to `T` via `T::from(...)`. The output is allocated as a flat
  `Vec<T>` of size `B * num_channels` and shaped into a CPU tensor
  via `Tensor::from_storage`.
- `Module<T>` impl (`time_embedding.rs:156..186`) routes `forward`
  through `forward_t`, returns empty parameter lists, and
  `is_training` is always `false` (the encoding is deterministic).
- `TimestepEmbedding<T: Float>` (`time_embedding.rs:201..209`)
  carries two `Linear<T>` layers + a `SiLU` activation.
- `TimestepEmbedding::new` (`time_embedding.rs:217..226`) calls
  `Linear::<T>::new` twice with bias = true.
- `Module<T>` impl for `TimestepEmbedding` (`time_embedding.rs:229..291`)
  composes the forward pass, exposes the two-linear parameter set
  under the diffusers prefixes (`linear_1.*`, `linear_2.*`), and
  routes `load_state_dict` through a per-prefix extract helper.

Non-test production consumers:

- `ferrotorch-diffusion/src/unet.rs:50` imports both
  `TimestepEmbedding` and `Timesteps`. The
  `UNet2DConditionModel::new` constructor instantiates them
  (`unet.rs` near line 1100+) and the `forward_t` path runs them
  on every UNet call.

## Parity contract

`parity_ops = []`. The contract is byte-equivalence with
`diffusers.models.embeddings.{Timesteps, TimestepEmbedding}` for
the SD-1.5 settings. Edge cases:

- `num_channels = 0` or odd → `InvalidArgument`.
- `downscale_freq_shift >= half` → `InvalidArgument` (degenerate
  denominator).
- `t = 0` → cos(0) = 1, sin(0) = 0 — verified in `timesteps_shape_flip_true`.

## Verification

Four lib tests in `time_embedding.rs:294..346`:

- `timesteps_shape_flip_true` — shape `[3, 8]`, `t=0` boundary check.
- `timesteps_rejects_odd_channels` — `num_channels = 7` is rejected.
- `timestep_embedding_shapes` — MLP shape `[1, 8] → [1, 16]`.
- `timestep_embedding_named_parameters` — layout of the diffusers
  prefixes.

No parity-sweep ops apply.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `Timesteps::new` at `ferrotorch-diffusion/src/time_embedding.rs:57..75`; non-test consumer: `ferrotorch-diffusion/src/unet.rs:50` imports `Timesteps` and instantiates it inside `UNet2DConditionModel::new` |
| REQ-2 | SHIPPED | impl: `Timesteps::forward_t` at `ferrotorch-diffusion/src/time_embedding.rs:87..152`; non-test consumer: `ferrotorch-diffusion/src/unet.rs` calls `self.time_proj.forward_t(timesteps)` inside the UNet forward path |
| REQ-3 | SHIPPED | impl: `TimestepEmbedding::new` at `ferrotorch-diffusion/src/time_embedding.rs:217..226`; non-test consumer: `ferrotorch-diffusion/src/unet.rs:50` constructs a `TimestepEmbedding` field inside `UNet2DConditionModel::new` |
| REQ-4 | SHIPPED | impl: `TimestepEmbedding::named_parameters` at `ferrotorch-diffusion/src/time_embedding.rs:248..257`; non-test consumer: `ferrotorch-diffusion/src/safetensors_loader.rs` (via `UNet2DConditionModel::load_state_dict`) routes HF checkpoint keys through this layout |
| REQ-5 | SHIPPED | impl: `Module<T> for Timesteps` empty-parameter and no-op `train`/`eval` at `ferrotorch-diffusion/src/time_embedding.rs:156..186`; non-test consumer: `ferrotorch-diffusion/src/unet.rs` includes `Timesteps` in its module graph and `parameters()` enumeration without any parameter contribution |
