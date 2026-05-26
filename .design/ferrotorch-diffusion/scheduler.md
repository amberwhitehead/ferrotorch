# DDIM noise scheduler

<!--
tier: 3-component
status: draft
baseline-pytorch: /home/doll/pytorch (HEAD)
upstream-paths:
  - diffusers/src/diffusers/schedulers/scheduling_ddim.py
  - diffusers/src/diffusers/schedulers/scheduling_ddpm.py
-->

## Summary

Deterministic DDIM noise scheduler (`eta = 0`) matching
`diffusers.schedulers.DDIMScheduler` for the SD-1.5 sampling defaults:
`scaled_linear` beta schedule, `epsilon` prediction, `leading`
timestep spacing, `clip_sample = false`, `set_alpha_to_one = false`,
`init_noise_sigma = 1.0`. Pre-computes the betas / alphas /
`alphas_cumprod` grid at construction time; `set_timesteps` selects
the inference subset and `step` performs one denoising step.

## Requirements

- REQ-1: `DDIMScheduler::new` precomputes `betas`, `alphas`, and
  `alphas_cumprod` over the full `num_train_timesteps` grid, plus the
  `final_alpha_cumprod` value used when the last step would walk past
  index 0.
- REQ-2: `set_timesteps(N)` returns the inference timestep schedule
  matching diffusers's `leading` spacing exactly. For SD-1.5 with
  `num_train_timesteps=1000`, `steps_offset=1`, `N=4`, this returns
  `[751, 501, 251, 1]`.
- REQ-3: `step(model_output, t, sample)` performs one DDIM denoising
  step using the standard η=0 recipe:
  `pred_x0 = (sample - sqrt(1 - alpha_t) * eps) / sqrt(alpha_t)`;
  `pred_dir = sqrt(1 - alpha_t_prev) * eps`;
  `prev_sample = sqrt(alpha_t_prev) * pred_x0 + pred_dir`.
- REQ-4: `init_noise_sigma()` returns 1.0 — the SD-1.5 DDIM contract.
  `scale_model_input(sample, t)` is the identity (DDIM doesn't
  rescale model input).
- REQ-5: `BetaSchedule::ScaledLinear` computes
  `betas = linspace(sqrt(beta_start), sqrt(beta_end), N)^2`.
- REQ-6: Only `PredictionType::Epsilon` is supported; any other
  configuration returns `FerrotorchError::InvalidArgument` from
  `set_timesteps`.

## Acceptance Criteria

- [x] AC-1: `beta_schedule_scaled_linear_matches_diffusers_sd15`
  (`scheduler.rs:422..445`).
- [x] AC-2: `timesteps_leading_4_steps_sd15` returns
  `[751, 501, 251, 1]` (`scheduler.rs:465..472`).
- [x] AC-3: `alphas_cumprod_is_monotone_decreasing` and the final
  value lies in `[0.001, 0.01]` for SD-1.5 (`scheduler.rs:447..462`).
- [x] AC-4: `step_recovers_zero_for_identity_noise` preserves shape
  and yields only finite values (`scheduler.rs:496..524`).
- [x] AC-5: `init_noise_sigma_is_one` passes (`scheduler.rs:484..488`).

## Architecture

Pure Rust state machine; tensor work goes through
`ferrotorch_core::grad_fns::arithmetic::{add, mul, sub}` and
`ferrotorch_core::scalar` for broadcastable scalar multiplies.

- `BetaSchedule` (`scheduler.rs:24..30`) — enum with
  `ScaledLinear` (SD default) and `Linear` variants.
- `TimestepSpacing` (`scheduler.rs:33..42`) — enum with `Leading`
  (SD default) and `Linspace` variants.
- `PredictionType` (`scheduler.rs:73..77`) — `Epsilon` only today;
  the enum exists so future `VPrediction` work can land additively.
- `DDIMConfig` (`scheduler.rs:45..70`) carries every diffusers
  configuration knob that affects forward math. `Default` (SD-1.5
  defaults) at `scheduler.rs:79..93`.
- `DDIMScheduler` (`scheduler.rs:108..121`) holds the precomputed
  `alphas_cumprod`, the `final_alpha_cumprod`, and the inference
  timesteps cache.
- `new` (`scheduler.rs:131..167`) precomputes the betas via
  `compute_betas` (`scheduler.rs:366..395`), then accumulates
  `alphas_cumprod`. The `final_alpha_cumprod` falls back to
  `alphas_cumprod[0]` when `set_alpha_to_one=false`.
- `set_timesteps` (`scheduler.rs:191..248`) implements the
  diffusers `leading` and `linspace` paths; for SD-1.5 with
  `steps_offset=1` and `N=4` this returns `[751, 501, 251, 1]`.
- `step` (`scheduler.rs:295..361`) is the η=0 DDIM update. Builds
  four broadcast-scalar tensors (`sqrt(beta_t)`,
  `1/sqrt(alpha_t)`, `sqrt(1-alpha_t_prev)`, `sqrt(alpha_t_prev)`)
  and combines them with `mul` / `sub` / `add`. Optional
  `clip_sample` clamps `pred_x0` to `[-1, 1]` (off for SD-1.5).
- `scale_model_input` (`scheduler.rs:265..271`) is the identity for
  DDIM, kept for forward-compat with non-DDIM schedulers.

Non-test production consumers:

- `ferrotorch-diffusion/src/pipeline.rs:30` imports
  `DDIMScheduler`; `pipeline.rs:132` calls `scale_model_input`,
  `pipeline.rs:194` calls `set_timesteps`, `pipeline.rs:199` calls
  `init_noise_sigma`, `pipeline.rs:212` calls `step`.
- `ferrotorch-diffusion/src/gpu/pipeline.rs` mirrors the same call
  sequence for the GPU pipeline.

## Parity contract

`parity_ops = []`. The contract is a numerical match against
`diffusers.schedulers.DDIMScheduler` with the SD-1.5 defaults: the
inference timesteps must equal `[751, 501, 251, 1]` for `N=4`, and
the step recipe must compute `prev_sample` byte-equivalent to
diffusers's float32 reference for the deterministic η=0 path. Edge
cases:

- `prev_timestep < 0` → use `final_alpha_cumprod` (the very last
  denoising step).
- `timestep >= num_train_timesteps` → `InvalidArgument`.
- `model_output.shape() != sample.shape()` → `ShapeMismatch`.

## Verification

Six lib tests in `scheduler.rs:417..525`:

- `beta_schedule_scaled_linear_matches_diffusers_sd15` — spot-checks
  `betas[0]`, `betas[999]`, and the midpoint within `5e-3`.
- `alphas_cumprod_is_monotone_decreasing` — verifies strictly
  decreasing positive values, with the final value in
  `[0.001, 0.01]` (SD-1.5 expects ~0.0047).
- `timesteps_leading_4_steps_sd15` — exact equality with
  `[751, 501, 251, 1]`.
- `timesteps_leading_50_steps_sd15_head` — first/last entries
  (`981`, `1`) at `N=50`.
- `init_noise_sigma_is_one` — exact 1.0.
- `final_alpha_cumprod_is_alphas_cumprod_zero_when_set_alpha_to_one_false`
  — verifies the SD-1.5 fallback.
- `step_recovers_zero_for_identity_noise` — shape preservation +
  finiteness on the zero-noise case.

No parity-sweep ops apply — the scheduler is a numerical recipe, not
a wrapped PyTorch op.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `DDIMScheduler::new` at `ferrotorch-diffusion/src/scheduler.rs:131..167`; non-test consumer: `ferrotorch-diffusion/src/pipeline.rs:255` (test helper) and `examples/sd_pipeline_dump.rs` construct the scheduler from a `DDIMConfig::sd_v1_5()` |
| REQ-2 | SHIPPED | impl: `set_timesteps` at `ferrotorch-diffusion/src/scheduler.rs:191..248`; non-test consumer: `ferrotorch-diffusion/src/pipeline.rs:194` calls `self.scheduler.set_timesteps(num_inference_steps)?` |
| REQ-3 | SHIPPED | impl: `step` at `ferrotorch-diffusion/src/scheduler.rs:295..361`; non-test consumer: `ferrotorch-diffusion/src/pipeline.rs:212` calls `self.scheduler.step(&guided, t, &latent)?` |
| REQ-4 | SHIPPED | impl: `init_noise_sigma` at `ferrotorch-diffusion/src/scheduler.rs:177..179` and `scale_model_input` at `ferrotorch-diffusion/src/scheduler.rs:265..271`; non-test consumer: `ferrotorch-diffusion/src/pipeline.rs:199` and `pipeline.rs:132` invoke both |
| REQ-5 | SHIPPED | impl: `compute_betas` (`ScaledLinear` arm) at `ferrotorch-diffusion/src/scheduler.rs:383..392`; non-test consumer: `DDIMScheduler::new` at `ferrotorch-diffusion/src/scheduler.rs:146` calls `compute_betas(...)` during construction |
| REQ-6 | SHIPPED | impl: prediction-type guard at `ferrotorch-diffusion/src/scheduler.rs:206..212`; non-test consumer: `set_timesteps` is called from `ferrotorch-diffusion/src/pipeline.rs:194` and surfaces this error before the diffusion loop runs |
