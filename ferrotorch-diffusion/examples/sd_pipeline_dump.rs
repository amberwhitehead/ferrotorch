//! Stable-Diffusion 1.5 end-to-end pipeline dump binary for the Phase F
//! real-artifact harness (#1163).
//!
//! Companion to `scripts/verify_sd_pipeline_inference.py`. Loads the
//! three pinned SD sub-model mirrors
//! (`ferrotorch/sd-v1-5-clip-text-encoder`, `ferrotorch/sd-v1-5-unet`,
//! `ferrotorch/sd-v1-5-vae-decoder`) plus the generation-trajectory
//! mirror (`ferrotorch/sd-v1-5-generation-trajectory`), composes them
//! into a `StableDiffusionPipeline` with a rust `DDIMScheduler`, runs
//! the same 4-step CFG denoising loop the pin script used, and dumps
//! every intermediate to disk in the standard
//! `[u32 ndim][u32 × ndim shape][f32 data]` little-endian format.
//!
//! Critical: the rust pipeline reads `init_latent.bin` from the
//! trajectory mirror rather than generating its own Gaussian noise —
//! rust's PRNG does not match `torch.Generator(device='cpu')`, so the
//! only way to get a byte-identical initial state is to consume the
//! reference noise.
//!
//! Likewise, the prompt-tokenizer is left python-side. The pipeline
//! reads pre-tokenized `prompt_input_ids.bin` /
//! `uncond_input_ids.bin` from the trajectory mirror and feeds them
//! into the rust CLIP encoder.
//!
//! Usage:
//! ```text
//! cargo run -p ferrotorch-diffusion --release --example sd_pipeline_dump -- \
//!     --output-dir /tmp/rust_sd_dumps \
//!     --trajectory-dir /tmp/trajectory_inputs \
//!     [--device cpu|gpu] [--seed 42] [--steps 4] [--guidance 7.5]
//! ```
//!
//! `--device gpu` requires the `cuda` cargo feature; without it the
//! example errors out at arg-parse time so a missing-feature build
//! cannot silently fall back to CPU.
//!
//! Output: every per-step `.bin` (matching the python pin layout
//! one-for-one) plus a final `final_image.bin`. The harness verifies
//! each.

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use ferrotorch_core::{FerrotorchResult, Tensor, TensorStorage};
use ferrotorch_diffusion::{
    load_clip_text_encoder, load_unet, load_vae_decoder, ClipTextConfig, DDIMConfig,
    DDIMScheduler, PipelineStepDump, StableDiffusionPipeline, UNet2DConditionConfig,
    VaeDecoderConfig,
};
use ferrotorch_hub::{hf_download_model, HubCache};

/// Target device for the forward passes.
///
/// `--device gpu` requires the `cuda` cargo feature; without it the
/// example errors out at arg-parse time so a missing-feature build
/// cannot silently fall back to CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Device {
    Cpu,
    Gpu,
}

#[derive(Debug)]
struct Args {
    output_dir: PathBuf,
    steps: usize,
    guidance: f32,
    device: Device,
    /// Directory containing the pinned trajectory fixtures
    /// (`init_latent.bin`, `prompt_input_ids.bin`, etc). Required because
    /// `hf_download_model` only resolves config + safetensors, not the
    /// fixture-bundle files we need; the python harness pre-downloads
    /// them via `hf_hub_download` and points us at the cache directory.
    trajectory_dir: PathBuf,
}

fn parse_args() -> Result<Args, String> {
    let mut output_dir: Option<PathBuf> = None;
    let mut steps: usize = 4;
    let mut guidance: f32 = 7.5;
    let mut trajectory_dir: Option<PathBuf> = None;
    let mut device = Device::Cpu;
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1usize;
    while i < argv.len() {
        match argv[i].as_str() {
            "--output-dir" => {
                output_dir = Some(PathBuf::from(
                    argv.get(i + 1).ok_or("--output-dir needs a value")?,
                ));
                i += 2;
            }
            "--steps" => {
                steps = argv
                    .get(i + 1)
                    .ok_or("--steps needs a value")?
                    .parse::<usize>()
                    .map_err(|e| format!("--steps must be a positive integer: {e}"))?;
                i += 2;
            }
            "--guidance" => {
                guidance = argv
                    .get(i + 1)
                    .ok_or("--guidance needs a value")?
                    .parse::<f32>()
                    .map_err(|e| format!("--guidance must be a float: {e}"))?;
                i += 2;
            }
            "--trajectory-dir" => {
                trajectory_dir = Some(PathBuf::from(
                    argv.get(i + 1).ok_or("--trajectory-dir needs a value")?,
                ));
                i += 2;
            }
            "--device" => {
                let v = argv.get(i + 1).ok_or("--device needs a value (cpu|gpu)")?;
                device = match v.as_str() {
                    "cpu" => Device::Cpu,
                    "gpu" => Device::Gpu,
                    other => return Err(format!("--device must be cpu|gpu, got {other:?}")),
                };
                i += 2;
            }
            // Accept `--seed`/`--prompt` for ergonomic parity with the
            // python script even though the rust side consumes the
            // pre-tokenized ids + init_latent from the mirror (so these
            // arguments do not actually change anything). Documented in
            // the module-level doc comment.
            "--seed" | "--prompt" | "--negative-prompt" => {
                if argv.get(i + 1).is_none() {
                    return Err(format!("{} needs a value", argv[i]));
                }
                i += 2;
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    Ok(Args {
        output_dir: output_dir.ok_or("--output-dir is required")?,
        steps,
        guidance,
        device,
        trajectory_dir: trajectory_dir.ok_or(
            "--trajectory-dir is required (points at directory holding init_latent.bin, \
             prompt_input_ids.bin, uncond_input_ids.bin — pulled via hf_hub_download by \
             the python harness)",
        )?,
    })
}

fn read_dump_f32(path: &Path) -> Result<(Vec<usize>, Vec<f32>), String> {
    let mut f = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut header4 = [0u8; 4];
    f.read_exact(&mut header4)
        .map_err(|e| format!("read header from {}: {e}", path.display()))?;
    let ndim = u32::from_le_bytes(header4) as usize;
    let mut shape = vec![0usize; ndim];
    for entry in &mut shape {
        f.read_exact(&mut header4)
            .map_err(|e| format!("read shape entry from {}: {e}", path.display()))?;
        *entry = u32::from_le_bytes(header4) as usize;
    }
    let count: usize = shape.iter().product();
    let mut buf = vec![0u8; count * 4];
    f.read_exact(&mut buf)
        .map_err(|e| format!("read data from {}: {e}", path.display()))?;
    let mut data = Vec::with_capacity(count);
    for chunk in buf.chunks_exact(4) {
        data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok((shape, data))
}

fn write_dump_f32(path: &Path, shape: &[usize], data: &[f32]) -> std::io::Result<()> {
    let expected: usize = shape.iter().product();
    assert_eq!(
        data.len(),
        expected,
        "data length {} disagrees with shape product {}",
        data.len(),
        expected
    );
    let mut f = File::create(path)?;
    f.write_all(&(shape.len() as u32).to_le_bytes())?;
    for &d in shape {
        f.write_all(&(d as u32).to_le_bytes())?;
    }
    let mut buf = Vec::with_capacity(data.len() * 4);
    for &v in data {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    f.write_all(&buf)
}

fn dump_tensor_to(path: &Path, t: &Tensor<f32>) -> Result<(), String> {
    let data = t
        .data()
        .map_err(|e| format!("tensor.data() failed: {e}"))?;
    write_dump_f32(path, t.shape(), data).map_err(|e| format!("write {}: {e}", path.display()))
}

fn locate_weights(dir: &Path) -> FerrotorchResult<PathBuf> {
    for name in ["model.safetensors", "diffusion_pytorch_model.safetensors"] {
        let p = dir.join(name);
        if p.is_file() {
            return Ok(p);
        }
    }
    Err(ferrotorch_core::FerrotorchError::InvalidArgument {
        message: format!(
            "neither model.safetensors nor diffusion_pytorch_model.safetensors found in {}",
            dir.display()
        ),
    })
}

/// Read the `[1, S]` (or `[S]`) f32-stored tokenizer ids back into a u32 vec.
fn read_input_ids(path: &Path) -> Result<Vec<u32>, String> {
    let (shape, data) = read_dump_f32(path)?;
    if !(shape.len() == 1 || (shape.len() == 2 && shape[0] == 1)) {
        return Err(format!(
            "expected input_ids shape [S] or [1, S], got {shape:?}"
        ));
    }
    let mut out: Vec<u32> = Vec::with_capacity(data.len());
    for (i, &v) in data.iter().enumerate() {
        if !v.is_finite() || v < 0.0 || v.fract() != 0.0 || v > u32::MAX as f32 {
            return Err(format!(
                "input_ids entry {i} ({v}) is not a non-negative integer"
            ));
        }
        out.push(v as u32);
    }
    Ok(out)
}

fn run() -> FerrotorchResult<()> {
    let args = parse_args().map_err(|m| ferrotorch_core::FerrotorchError::InvalidArgument {
        message: m,
    })?;
    std::fs::create_dir_all(&args.output_dir).map_err(|e| {
        ferrotorch_core::FerrotorchError::InvalidArgument {
            message: format!(
                "could not create output dir {}: {e}",
                args.output_dir.display()
            ),
        }
    })?;

    let cache = HubCache::with_default_dir();

    // ---- 1. Pull the three sub-model mirrors. -----------------------------
    eprintln!("[sd_pipeline_dump] downloading SD-1.5 sub-models from HF...");
    let clip_dir = hf_download_model("ferrotorch/sd-v1-5-clip-text-encoder", "main", &cache)?;
    let unet_dir = hf_download_model("ferrotorch/sd-v1-5-unet", "main", &cache)?;
    let vae_dir = hf_download_model("ferrotorch/sd-v1-5-vae-decoder", "main", &cache)?;
    let traj_dir = args.trajectory_dir.clone();
    eprintln!("[sd_pipeline_dump] clip:        {}", clip_dir.display());
    eprintln!("[sd_pipeline_dump] unet:        {}", unet_dir.display());
    eprintln!("[sd_pipeline_dump] vae:         {}", vae_dir.display());
    eprintln!("[sd_pipeline_dump] trajectory:  {}", traj_dir.display());

    // ---- 2. Build the three sub-models. -----------------------------------
    //
    // For `--device gpu` we load the CPU sub-models, hand their
    // state-dicts to the matching `Gpu*::from_module` constructor, then
    // drop the CPU copies before the diffusion loop runs. That keeps
    // peak host RAM down (the original #1163 attempt OOMed on CPU when
    // the three CPU sub-models + Python diffusers reference all stayed
    // resident simultaneously).
    let clip_cfg = ClipTextConfig::from_file(&clip_dir.join("config.json"))?;
    let (clip, clip_drop) =
        load_clip_text_encoder::<f32>(&locate_weights(&clip_dir)?, clip_cfg.clone(), false)?;
    eprintln!(
        "[sd_pipeline_dump] CLIP loaded (dropped {} buffer keys)",
        clip_drop.dropped.len()
    );

    let unet_cfg = UNet2DConditionConfig::from_file(&unet_dir.join("config.json"))?;
    let (unet, unet_drop) = load_unet::<f32>(&locate_weights(&unet_dir)?, unet_cfg.clone(), false)?;
    eprintln!(
        "[sd_pipeline_dump] UNet loaded (dropped {} keys)",
        unet_drop.dropped.len()
    );

    let vae_cfg = VaeDecoderConfig::from_file(&vae_dir.join("config.json"))?;
    let (vae, vae_drop) = load_vae_decoder::<f32>(&locate_weights(&vae_dir)?, vae_cfg.clone(), false)?;
    eprintln!(
        "[sd_pipeline_dump] VAE loaded (dropped {} keys)",
        vae_drop.dropped.len()
    );

    let scheduler = DDIMScheduler::new(DDIMConfig::sd_v1_5())?;
    eprintln!("[sd_pipeline_dump] DDIM scheduler built (SD-1.5 defaults)");

    // ---- 3. Read the pinned tokenized inputs + init_latent. ---------------
    let prompt_ids = read_input_ids(&traj_dir.join("prompt_input_ids.bin"))
        .map_err(|e| ferrotorch_core::FerrotorchError::InvalidArgument {
            message: format!("read prompt_input_ids.bin: {e}"),
        })?;
    let uncond_ids = read_input_ids(&traj_dir.join("uncond_input_ids.bin"))
        .map_err(|e| ferrotorch_core::FerrotorchError::InvalidArgument {
            message: format!("read uncond_input_ids.bin: {e}"),
        })?;
    eprintln!(
        "[sd_pipeline_dump] prompt_ids: {} entries; uncond_ids: {} entries",
        prompt_ids.len(),
        uncond_ids.len(),
    );

    // ---- 4. Load init_latent (rust PRNG != torch PRNG, so read it). -------
    let (latent_shape, latent_data) = read_dump_f32(&traj_dir.join("init_latent.bin"))
        .map_err(|e| ferrotorch_core::FerrotorchError::InvalidArgument {
            message: format!("read init_latent.bin: {e}"),
        })?;
    let init_latent = Tensor::<f32>::from_storage(
        TensorStorage::cpu(latent_data),
        latent_shape.clone(),
        false,
    )?;
    eprintln!(
        "[sd_pipeline_dump] loaded init_latent shape={latent_shape:?}",
    );

    // ---- 5. Run encoder + diffusion loop on the chosen device. ------------
    eprintln!(
        "[sd_pipeline_dump] generating: device={:?} steps={} guidance={}",
        args.device, args.steps, args.guidance
    );
    let (cond_embeds, uncond_embeds, image, dumps) = match args.device {
        Device::Cpu => {
            eprintln!("[sd_pipeline_dump] device = cpu");
            let mut pipeline = StableDiffusionPipeline::new(clip, unet, vae, scheduler)?;
            let cond_embeds = pipeline.encode_prompt(&prompt_ids)?;
            let uncond_embeds = pipeline.encode_prompt(&uncond_ids)?;
            eprintln!(
                "[sd_pipeline_dump] encoded: cond_embeds shape={:?}, uncond_embeds shape={:?}",
                cond_embeds.shape(),
                uncond_embeds.shape()
            );
            let (image, dumps) = pipeline.generate(
                &cond_embeds,
                &uncond_embeds,
                &init_latent,
                args.steps,
                args.guidance,
            )?;
            (cond_embeds, uncond_embeds, image, dumps)
        }
        Device::Gpu => run_gpu(
            &clip,
            &unet,
            &vae,
            scheduler,
            &prompt_ids,
            &uncond_ids,
            &init_latent,
            args.steps,
            args.guidance,
        )?,
    };
    // Free the CPU sub-models before we touch the heavy dumps in GPU
    // mode (in CPU mode they live until the pipeline is consumed
    // above; the `drop` here is a no-op in that branch).
    drop((clip_drop, unet_drop, vae_drop));

    dump_tensor_to(&args.output_dir.join("cond_embeds.bin"), &cond_embeds).map_err(|e| {
        ferrotorch_core::FerrotorchError::InvalidArgument { message: e }
    })?;
    dump_tensor_to(&args.output_dir.join("uncond_embeds.bin"), &uncond_embeds).map_err(|e| {
        ferrotorch_core::FerrotorchError::InvalidArgument { message: e }
    })?;

    // Persist a copy of init_latent in the dump dir so the harness can
    // confirm byte-exact passthrough (the trivial "init_latent: PASS,
    // exact match" stage).
    dump_tensor_to(&args.output_dir.join("init_latent.bin"), &init_latent).map_err(|e| {
        ferrotorch_core::FerrotorchError::InvalidArgument { message: e }
    })?;

    for step in &dumps {
        eprintln!(
            "[sd_pipeline_dump] step {} (t={}): |uncond|={:.3} |cond|={:.3} \
             |guided|={:.3} |latent|={:.3}",
            step.step,
            step.timestep,
            l2_norm(&step.noise_pred_uncond),
            l2_norm(&step.noise_pred_cond),
            l2_norm(&step.guided_noise),
            l2_norm(&step.latent_after_step),
        );
        let i = step.step;
        dump_tensor_to(
            &args.output_dir.join(format!("step_{i}_noise_pred_uncond.bin")),
            &step.noise_pred_uncond,
        )
        .map_err(|e| ferrotorch_core::FerrotorchError::InvalidArgument { message: e })?;
        dump_tensor_to(
            &args.output_dir.join(format!("step_{i}_noise_pred_cond.bin")),
            &step.noise_pred_cond,
        )
        .map_err(|e| ferrotorch_core::FerrotorchError::InvalidArgument { message: e })?;
        dump_tensor_to(
            &args.output_dir.join(format!("step_{i}_guided_noise.bin")),
            &step.guided_noise,
        )
        .map_err(|e| ferrotorch_core::FerrotorchError::InvalidArgument { message: e })?;
        dump_tensor_to(
            &args.output_dir.join(format!("step_{i}_latent_after.bin")),
            &step.latent_after_step,
        )
        .map_err(|e| ferrotorch_core::FerrotorchError::InvalidArgument { message: e })?;
    }

    dump_tensor_to(&args.output_dir.join("final_image.bin"), &image).map_err(|e| {
        ferrotorch_core::FerrotorchError::InvalidArgument { message: e }
    })?;
    eprintln!(
        "[sd_pipeline_dump] final image shape={:?} |image|={:.3}",
        image.shape(),
        l2_norm(&image)
    );

    // JSON verdict for the harness.
    let mut s = String::new();
    s.push('{');
    s.push_str(&format!("\"steps\":{},", args.steps));
    s.push_str(&format!("\"guidance\":{},", args.guidance));
    s.push_str(&format!(
        "\"image_shape\":[{},{},{},{}],",
        image.shape()[0],
        image.shape()[1],
        image.shape()[2],
        image.shape()[3]
    ));
    s.push_str(&format!("\"output_dir\":\"{}\"", args.output_dir.display()));
    s.push('}');
    println!("{s}");
    Ok(())
}

fn l2_norm(t: &Tensor<f32>) -> f32 {
    match t.data() {
        Ok(d) => {
            let s: f32 = d.iter().map(|x| x * x).sum();
            s.sqrt()
        }
        Err(_) => f32::NAN,
    }
}

/// Bundle returned from a single end-to-end run on either device:
/// `(cond_embeds, uncond_embeds, final_image, per_step_dumps)`.
type RunBundle = (
    Tensor<f32>,
    Tensor<f32>,
    Tensor<f32>,
    Vec<PipelineStepDump<f32>>,
);

/// GPU forward path. Builds the three `Gpu*` sub-models from the
/// already-loaded CPU state-dicts, composes them with the CPU
/// `DDIMScheduler` into a [`GpuStableDiffusionPipeline`], encodes both
/// prompts on the GPU, runs the diffusion loop, decodes the final
/// image, and returns everything the CPU branch returns so the caller
/// can dump the results uniformly.
///
/// Without the `cuda` cargo feature this is a hard error — the example
/// must refuse to silently fall back to CPU when the harness asked for
/// GPU.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn run_gpu(
    clip: &ferrotorch_diffusion::ClipTextEncoder<f32>,
    unet: &ferrotorch_diffusion::UNet2DConditionModel<f32>,
    vae: &ferrotorch_diffusion::VaeDecoder<f32>,
    scheduler: DDIMScheduler,
    prompt_ids: &[u32],
    uncond_ids: &[u32],
    init_latent: &Tensor<f32>,
    steps: usize,
    guidance: f32,
) -> FerrotorchResult<RunBundle> {
    use ferrotorch_diffusion::gpu::{
        GpuClipTextEncoder, GpuStableDiffusionPipeline, GpuUNet2DConditional, GpuVaeDecoder,
    };
    use ferrotorch_gpu::GpuDevice;

    eprintln!("[sd_pipeline_dump] device = gpu");
    let device =
        GpuDevice::new(0).map_err(|e| ferrotorch_core::FerrotorchError::InvalidArgument {
            message: format!("GpuDevice::new(0) failed: {e}"),
        })?;
    let (gpu_clip, clip_report) = GpuClipTextEncoder::from_module(clip, &device)?;
    eprintln!(
        "[sd_pipeline_dump] gpu CLIP: dropped_keys={}",
        clip_report.dropped.len(),
    );
    let (gpu_unet, unet_report) = GpuUNet2DConditional::from_module(unet, &device)?;
    eprintln!(
        "[sd_pipeline_dump] gpu UNet: dropped_keys={}",
        unet_report.dropped.len(),
    );
    let (gpu_vae, vae_report) = GpuVaeDecoder::from_module(vae, &device)?;
    eprintln!(
        "[sd_pipeline_dump] gpu VAE: dropped_keys={}",
        vae_report.dropped.len(),
    );

    let mut pipeline =
        GpuStableDiffusionPipeline::new(gpu_clip, gpu_unet, gpu_vae, scheduler, device)?;
    let cond_embeds = pipeline.encode_prompt(prompt_ids)?;
    let uncond_embeds = pipeline.encode_prompt(uncond_ids)?;
    eprintln!(
        "[sd_pipeline_dump] encoded: cond_embeds shape={:?}, uncond_embeds shape={:?}",
        cond_embeds.shape(),
        uncond_embeds.shape()
    );
    let (image, dumps) =
        pipeline.generate(&cond_embeds, &uncond_embeds, init_latent, steps, guidance)?;
    Ok((cond_embeds, uncond_embeds, image, dumps))
}

#[cfg(not(feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
fn run_gpu(
    _clip: &ferrotorch_diffusion::ClipTextEncoder<f32>,
    _unet: &ferrotorch_diffusion::UNet2DConditionModel<f32>,
    _vae: &ferrotorch_diffusion::VaeDecoder<f32>,
    _scheduler: DDIMScheduler,
    _prompt_ids: &[u32],
    _uncond_ids: &[u32],
    _init_latent: &Tensor<f32>,
    _steps: usize,
    _guidance: f32,
) -> FerrotorchResult<(Tensor<f32>, Tensor<f32>, Tensor<f32>, Vec<PipelineStepDump<f32>>)> {
    Err(ferrotorch_core::FerrotorchError::InvalidArgument {
        message: "--device gpu requires the `cuda` cargo feature \
                  (build with `--features=cuda`)"
            .into(),
    })
}

fn main() {
    if let Err(e) = run() {
        eprintln!("[sd_pipeline_dump] error: {e}");
        std::process::exit(1);
    }
}
