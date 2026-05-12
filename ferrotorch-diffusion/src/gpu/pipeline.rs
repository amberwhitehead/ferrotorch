#![cfg(feature = "cuda")]
//! GPU-resident Stable-Diffusion 1.5 end-to-end text-to-image generation
//! pipeline (Phase F.4 of the SD GPU sequence, #1163).
//!
//! Mirrors [`crate::pipeline::StableDiffusionPipeline`] op-for-op but
//! composes the three VRAM-resident sub-models
//! ([`crate::gpu::clip::GpuClipTextEncoder`],
//! [`crate::gpu::unet::GpuUNet2DConditional`],
//! [`crate::gpu::vae::GpuVaeDecoder`]) plus the existing CPU-side
//! [`crate::scheduler::DDIMScheduler`] into a single
//! `generate` call. The scheduler stays on the host: it is a few KB of
//! `alphas_cumprod` math operating on the small `[1, 4, 64, 64]` latent,
//! and round-tripping through VRAM would add latency without changing
//! arithmetic precision.
//!
//! ```text
//! prompt_ids       → GpuClipTextEncoder → cond_embeds   [1, 77, 768]   (host f32)
//! negative_ids     → GpuClipTextEncoder → uncond_embeds [1, 77, 768]   (host f32)
//! init_latent      [1, 4, 64, 64]                                       (host f32)
//! for step in 0..N:
//!   t = scheduler.timesteps[step]
//!   uncond_noise = GpuUNet(latent, t, uncond_embeds)
//!   cond_noise   = GpuUNet(latent, t, cond_embeds)
//!   noise_pred   = uncond_noise + guidance_scale * (cond_noise - uncond_noise)
//!   latent       = scheduler.step(noise_pred, t, latent)              (host f32)
//! image = GpuVaeDecoder.decode(latent / 0.18215) → [1, 3, 512, 512]    (host f32)
//! ```
//!
//! Determinism: rust's PRNG does NOT match
//! `torch.Generator(device='cpu')`. The pipeline therefore takes
//! `init_latent` from the caller (the dump example reads it from the
//! pinned `ferrotorch/sd-v1-5-generation-trajectory` mirror).
//!
//! Note on device residency: all three sub-models hold weights in VRAM
//! and download every forward result to host f32 (matching their
//! existing `encode` / `forward` / `decode` contracts). Per-step
//! intermediates are therefore host tensors — perfectly sized for the
//! CFG-blend + DDIM-step arithmetic in this module — and the dump
//! example can persist them with the same `Tensor<f32>` machinery the
//! CPU path uses.

use ferrotorch_core::grad_fns::arithmetic::{add, mul, sub};
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Tensor, TensorStorage};
use ferrotorch_gpu::GpuDevice;

use crate::gpu::clip::GpuClipTextEncoder;
use crate::gpu::unet::GpuUNet2DConditional;
use crate::gpu::vae::GpuVaeDecoder;
use crate::pipeline::PipelineStepDump;
use crate::scheduler::DDIMScheduler;

/// VRAM-resident SD-1.5 pipeline.
///
/// Constructed from already-loaded GPU sub-models on a single
/// [`GpuDevice`] (the caller is responsible for ensuring all three live
/// on the same device — `GpuDevice` is `Clone` and the dump example
/// hands the same handle to every constructor).
///
/// `scheduler` is a CPU [`DDIMScheduler`]: the per-step scheduler math
/// is cheap and benefits from f64 internal computation.
#[derive(Debug)]
pub struct GpuStableDiffusionPipeline {
    /// CLIP text tower (token-ids → `[1, 77, 768]` last_hidden_state).
    pub text_encoder: GpuClipTextEncoder,
    /// UNet noise predictor.
    pub unet: GpuUNet2DConditional,
    /// VAE decoder (latent → `[1, 3, 512, 512]` image in `[-1, 1]`).
    pub vae: GpuVaeDecoder,
    /// DDIM scheduler. Lives host-side — `&mut self` is needed because
    /// `set_timesteps` caches per-call state.
    pub scheduler: DDIMScheduler,
    /// Device handle the three sub-models share. Kept for completeness
    /// even though every per-call GPU op routes through the sub-models'
    /// own copies of the handle.
    _device: GpuDevice,
}

impl GpuStableDiffusionPipeline {
    /// Compose three GPU sub-models + a CPU scheduler into a pipeline.
    ///
    /// # Errors
    ///
    /// Currently infallible; returned for forward-compat with future
    /// validations (e.g. cross-checking config consistency between the
    /// sub-models).
    pub fn new(
        text_encoder: GpuClipTextEncoder,
        unet: GpuUNet2DConditional,
        vae: GpuVaeDecoder,
        scheduler: DDIMScheduler,
        device: GpuDevice,
    ) -> FerrotorchResult<Self> {
        Ok(Self {
            text_encoder,
            unet,
            vae,
            scheduler,
            _device: device,
        })
    }

    /// Encode a single token-id sequence into `[1, S, hidden_size]`.
    ///
    /// # Errors
    ///
    /// Propagates any error from [`GpuClipTextEncoder::encode`].
    pub fn encode_prompt(&self, input_ids: &[u32]) -> FerrotorchResult<Tensor<f32>> {
        self.text_encoder.encode(input_ids)
    }

    /// Build the `[B]` host f32 timestep tensor the UNet's
    /// [`GpuUNet2DConditional::forward`] consumes (the GPU UNet reads it
    /// as f32 host-side for the sinusoidal projection).
    fn timestep_tensor(timestep: usize, batch: usize) -> FerrotorchResult<Tensor<f32>> {
        Tensor::<f32>::from_storage(
            TensorStorage::cpu(vec![timestep as f32; batch]),
            vec![batch],
            false,
        )
    }

    /// One CFG-guided UNet evaluation: two GPU forwards (uncond + cond)
    /// blended on host f32. Mirrors
    /// [`crate::pipeline::StableDiffusionPipeline::cfg_eval`] one-to-one
    /// — the CFG arithmetic stays host-side because the per-step
    /// noise-prediction tensors are already host f32 once the UNet
    /// returns (see module-level note).
    fn cfg_eval(
        &self,
        latent: &Tensor<f32>,
        timestep: usize,
        cond_embeds: &Tensor<f32>,
        uncond_embeds: &Tensor<f32>,
        guidance_scale: f32,
    ) -> FerrotorchResult<(Tensor<f32>, Tensor<f32>, Tensor<f32>)> {
        let batch = latent.shape()[0];
        let t = Self::timestep_tensor(timestep, batch)?;
        // DDIM `scale_model_input` is identity at SD-1.5 defaults; the
        // call keeps the pipeline forward-compatible with non-DDIM
        // schedulers.
        let model_input = self.scheduler.scale_model_input(latent, timestep)?;
        let noise_uncond = self.unet.forward(&model_input, &t, uncond_embeds)?;
        let noise_cond = self.unet.forward(&model_input, &t, cond_embeds)?;
        // guided = uncond + scale * (cond - uncond).
        let gs_scalar = ferrotorch_core::scalar::<f32>(guidance_scale)?;
        let diff = sub(&noise_cond, &noise_uncond)?;
        let scaled = mul(&diff, &gs_scalar)?;
        let guided = add(&noise_uncond, &scaled)?;
        Ok((noise_uncond, noise_cond, guided))
    }

    /// Run the full diffusion loop starting from `init_latent`. Returns
    /// the final decoded image `[1, 3, 512, 512]` plus the per-step
    /// diagnostic dumps in iteration order.
    ///
    /// The caller pre-encodes `cond_embeds` / `uncond_embeds`
    /// (`[1, S, cross_attention_dim]`) via [`Self::encode_prompt`] and
    /// supplies `init_latent` (`[1, in_channels, H, W]`).
    /// `init_latent` is scaled by `scheduler.init_noise_sigma()` (1.0
    /// for SD-1.5 DDIM) — kept explicit so future scheduler swaps keep
    /// working.
    ///
    /// `num_inference_steps` and `guidance_scale` follow the
    /// diffusers convention.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::ShapeMismatch`] on any rank/shape
    /// mismatch and forwards every sub-model error.
    pub fn generate(
        &mut self,
        cond_embeds: &Tensor<f32>,
        uncond_embeds: &Tensor<f32>,
        init_latent: &Tensor<f32>,
        num_inference_steps: usize,
        guidance_scale: f32,
    ) -> FerrotorchResult<(Tensor<f32>, Vec<PipelineStepDump<f32>>)> {
        if init_latent.ndim() != 4 {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "GpuStableDiffusionPipeline::generate: expected init_latent \
                     [B, 4, H, W], got {:?}",
                    init_latent.shape()
                ),
            });
        }
        if cond_embeds.shape() != uncond_embeds.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "GpuStableDiffusionPipeline::generate: cond_embeds shape {:?} \
                     != uncond_embeds {:?}",
                    cond_embeds.shape(),
                    uncond_embeds.shape()
                ),
            });
        }

        // Configure the scheduler and snapshot timesteps up front (set_timesteps
        // returns a borrow; collect so the loop body can still call &mut self
        // for `step`).
        let timesteps: Vec<usize> = self
            .scheduler
            .set_timesteps(num_inference_steps)?
            .to_vec();

        // latent = init_latent * scheduler.init_noise_sigma. Identity for
        // SD-1.5 DDIM but keep the multiplication explicit so future
        // scheduler swaps (LMS, Euler) keep working.
        let sigma = self.scheduler.init_noise_sigma() as f32;
        let sigma_scalar = ferrotorch_core::scalar::<f32>(sigma)?;
        let mut latent = mul(init_latent, &sigma_scalar)?;

        let mut dumps: Vec<PipelineStepDump<f32>> = Vec::with_capacity(num_inference_steps);
        for (i, &t) in timesteps.iter().enumerate() {
            let (noise_uncond, noise_cond, guided) =
                self.cfg_eval(&latent, t, cond_embeds, uncond_embeds, guidance_scale)?;
            let latent_after = self.scheduler.step(&guided, t, &latent)?;
            dumps.push(PipelineStepDump {
                step: i,
                timestep: t,
                noise_pred_uncond: noise_uncond,
                noise_pred_cond: noise_cond,
                guided_noise: guided,
                latent_after_step: latent_after.clone(),
            });
            latent = latent_after;
        }

        // VAE decode: `image = vae.decode(latent / scaling_factor)`. The
        // existing `GpuVaeDecoder::decode` applies the `1/scaling_factor`
        // divide internally (mirroring `decode_with_scaling` on CPU and
        // `diffusers.AutoencoderKL.decode(z / 0.18215).sample`).
        let image = self.vae.decode(&latent)?;
        Ok((image, dumps))
    }
}
