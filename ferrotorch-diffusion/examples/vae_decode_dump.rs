//! VAE-decoder inference-dump binary for the SD-1.5 real-artifact harness.
//!
//! Companion to `scripts/verify_diffusion_inference.py`. Loads the
//! pinned `ferrotorch/sd-v1-5-vae-decoder` mirror from HF, runs the
//! decoder forward pass on the parity-probe latent (or a
//! caller-supplied `.bin` latent file), and dumps the resulting image
//! `[1, 3, 512, 512]` to disk in the standard
//! `[u32 ndim][u32 × ndim shape][f32 data]` little-endian format used
//! across vision / causal-LM / text-embedding / audio dumps.
//!
//! Usage (network required for first-touch; subsequent runs use the
//! local hub cache):
//! ```text
//! cargo run -p ferrotorch-diffusion --release --example vae_decode_dump -- \
//!     --model sd-v1-5-vae-decoder \
//!     --latent /tmp/parity_latent.bin \
//!     --output /tmp/rust_image.bin
//! ```
//!
//! `--latent` is required and points at a `[u32 ndim][u32 shape][f32]`
//! little-endian dump of the reference latent. For SD 1.5 it has shape
//! `[1, 4, 64, 64]`. The mirror also ships `_value_parity_latent.bin`
//! so the harness can fall back to it if `--latent` is omitted.
//!
//! The latent on disk has already been multiplied by `scaling_factor`
//! (i.e. it matches the latent the Python pipeline feeds to
//! `vae.decode`). The Rust forward path divides by `scaling_factor`
//! internally — same as `AutoencoderKL.decode`.
//!
//! Output:
//!   * `--output <path>`: image tensor `[1, 3, 512, 512]` in the
//!     standard dump format.
//!   * stdout: one JSON line
//!     `{"shape":[1,3,512,512],"latent_shape":[1,4,64,64],"dropped_keys":N}`
//!     so the Python harness can parse the verdict.

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use ferrotorch_core::{FerrotorchResult, Tensor, TensorStorage};
use ferrotorch_diffusion::{load_vae_decoder, VaeDecoderConfig};
use ferrotorch_hub::{hf_download_model, HubCache};

#[derive(Debug)]
struct Args {
    model: String,
    output: PathBuf,
    latent: Option<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let mut model: Option<String> = None;
    let mut output: Option<PathBuf> = None;
    let mut latent: Option<PathBuf> = None;
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1usize;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => {
                model = Some(argv.get(i + 1).ok_or("--model needs a value")?.clone());
                i += 2;
            }
            "--output" => {
                output = Some(PathBuf::from(
                    argv.get(i + 1).ok_or("--output needs a value")?,
                ));
                i += 2;
            }
            "--latent" => {
                latent = Some(PathBuf::from(
                    argv.get(i + 1).ok_or("--latent needs a value")?,
                ));
                i += 2;
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    Ok(Args {
        model: model.ok_or("--model is required (e.g. --model sd-v1-5-vae-decoder)")?,
        output: output.ok_or("--output is required (path to decoded image .bin)")?,
        latent,
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

fn run() -> FerrotorchResult<()> {
    let args = parse_args().map_err(|m| ferrotorch_core::FerrotorchError::InvalidArgument {
        message: m,
    })?;

    let repo = format!("ferrotorch/{}", args.model);
    eprintln!("[vae_decode_dump] repo = {repo}");

    // -- 1. Download the bundle into the hub cache. ---------------------
    let cache = HubCache::with_default_dir();
    let repo_dir = hf_download_model(&repo, "main", &cache)?;
    eprintln!(
        "[vae_decode_dump] cached at {} ({} files)",
        repo_dir.display(),
        std::fs::read_dir(&repo_dir).map(|r| r.count()).unwrap_or(0)
    );

    // -- 2. Parse config. -----------------------------------------------
    let cfg_path = repo_dir.join("config.json");
    let cfg = VaeDecoderConfig::from_file(&cfg_path)?;
    eprintln!(
        "[vae_decode_dump] cfg: block_out_channels={:?} layers_per_block={} \
         norm_num_groups={} latent_channels={} sample_size={} scaling_factor={}",
        cfg.block_out_channels,
        cfg.layers_per_block,
        cfg.norm_num_groups,
        cfg.latent_channels,
        cfg.sample_size,
        cfg.scaling_factor,
    );

    // -- 3. Resolve latent input. The harness can supply --latent;
    //       otherwise fall back to the frozen `_value_parity_latent.bin`
    //       shipped by the mirror. ---------------------------------------
    let latent_path = if let Some(p) = args.latent.clone() {
        p
    } else {
        let parity = repo_dir.join("_value_parity_latent.bin");
        if !parity.is_file() {
            return Err(ferrotorch_core::FerrotorchError::InvalidArgument {
                message: format!(
                    "neither --latent passed nor parity-probe latent found at {}",
                    parity.display(),
                ),
            });
        }
        parity
    };
    let (lat_shape, lat_data) =
        read_dump_f32(&latent_path).map_err(|e| ferrotorch_core::FerrotorchError::InvalidArgument {
            message: format!(
                "failed to read latent input from {}: {e}",
                latent_path.display()
            ),
        })?;
    eprintln!(
        "[vae_decode_dump] latent: shape={lat_shape:?} from {}",
        latent_path.display(),
    );
    let latent = Tensor::from_storage(TensorStorage::cpu(lat_data), lat_shape.clone(), false)?;

    // -- 4. Load weights and build decoder. -----------------------------
    let weights_path = locate_weights(&repo_dir)?;
    eprintln!(
        "[vae_decode_dump] weights file: {}",
        weights_path.display()
    );
    let (decoder, drop_report) =
        load_vae_decoder::<f32>(&weights_path, cfg, /* strict = */ false)?;
    eprintln!(
        "[vae_decode_dump] loaded weights: dropped_keys={}",
        drop_report.dropped.len(),
    );

    // -- 5. Forward + dump. --------------------------------------------
    // The on-disk latent is already pre-multiplied by `scaling_factor`
    // (matching the SD pipeline convention): `decode_with_scaling`
    // divides on the way in.
    let out = decoder.decode_with_scaling(&latent)?;
    let out_shape = out.shape();
    let out_data = out.data()?;
    assert_eq!(
        out_shape.len(),
        4,
        "decoder output must be [B, 3, H, W], got {out_shape:?}",
    );

    write_dump_f32(&args.output, out_shape, out_data).map_err(|e| {
        ferrotorch_core::FerrotorchError::InvalidArgument {
            message: format!(
                "failed writing decoder output to {}: {e}",
                args.output.display()
            ),
        }
    })?;
    eprintln!(
        "[vae_decode_dump] wrote {} ({} bytes, shape={out_shape:?})",
        args.output.display(),
        std::fs::metadata(&args.output)
            .map(|m| m.len())
            .unwrap_or(0)
    );

    // -- 6. JSON verdict line. -----------------------------------------
    let mut s = String::new();
    s.push('{');
    s.push_str(&format!(
        "\"shape\":[{},{},{},{}],",
        out_shape[0], out_shape[1], out_shape[2], out_shape[3]
    ));
    s.push_str(&format!(
        "\"latent_shape\":[{},{},{},{}],",
        lat_shape[0], lat_shape[1], lat_shape[2], lat_shape[3]
    ));
    s.push_str(&format!("\"dropped_keys\":{}", drop_report.dropped.len()));
    s.push('}');
    println!("{s}");

    Ok(())
}

/// Locate the weights file inside the mirror directory. The pin script
/// uploads as `model.safetensors`; some upstreams use
/// `diffusion_pytorch_model.safetensors`.
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

fn main() {
    if let Err(e) = run() {
        eprintln!("[vae_decode_dump] error: {e}");
        std::process::exit(1);
    }
}
