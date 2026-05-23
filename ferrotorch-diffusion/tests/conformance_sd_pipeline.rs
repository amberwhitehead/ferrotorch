//! Real-artifact end-to-end SD-1.5 pipeline parity test, gated on
//! network access.
//!
//! Runs `scripts/verify_sd_pipeline_inference.py` end-to-end against
//! the pinned `ferrotorch/sd-v1-5-generation-trajectory` mirror (Phase
//! F, #1163). Composes
//! `ferrotorch/sd-v1-5-clip-text-encoder` (#1152) +
//! `ferrotorch/sd-v1-5-unet` (#1151) +
//! `ferrotorch/sd-v1-5-vae-decoder` (#1150) + the new
//! `DDIMScheduler` into a single `StableDiffusionPipeline` and
//! verifies the rust pipeline output stage-by-stage against the frozen
//! `diffusers.StableDiffusionPipeline` trajectory.
//!
//! Marked `#[ignore]` since the harness:
//!   * touches HuggingFace (downloads the four SD mirrors, ~4 GB total
//!     including the UNet),
//!   * runs the rust UNet 8 times (4 steps × CFG) + VAE on GPU,
//!   * runs the python diffusers reference once during the pin (not
//!     each verification — the trajectory is replayed from the
//!     mirror), then compares.
//!
//! Defaults to `--device gpu` (requires the `cuda` cargo feature). The
//! CPU path OOMs at end-to-end pipeline scale on standard 32GB-RAM
//! machines because all three SD sub-models + diffusers reference must
//! live simultaneously (see #1163 dispatch history).
//!
//! Enable via:
//!
//! ```text
//! cargo test -p ferrotorch-diffusion --features=cuda \
//!     --test conformance_sd_pipeline -- --ignored
//! ```

use std::path::PathBuf;
use std::process::Command;

/// Resolve the workspace root from this crate's `CARGO_MANIFEST_DIR`.
fn workspace_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .expect("ferrotorch-diffusion manifest must have a parent (the workspace root)")
        .to_path_buf()
}

/// Real-artifact end-to-end SD-1.5 pipeline parity smoke (#1163).
///
/// Drives the python harness which:
///  1. Pulls the trajectory mirror.
///  2. Runs `cargo run --example sd_pipeline_dump`.
///  3. Compares every stage (text embeddings, init_latent passthrough,
///     per-step UNet/CFG/scheduler outputs, final VAE-decoded image)
///     against the diffusers reference at the per-stage tolerance
///     baked into the harness.
///  4. Prints `sd-v1-5-end-to-end-pipeline: PASS` on success.
#[test]
#[ignore = "Requires network access — enable with --ignored"]
fn pretrained_sd_v1_5_pipeline_parity_smoke() {
    let root = workspace_root();
    let harness = root.join("scripts").join("verify_sd_pipeline_inference.py");
    assert!(
        harness.is_file(),
        "harness missing at {}",
        harness.display()
    );

    // Default to GPU; CPU path OOMs at end-to-end pipeline scale.
    let device = if cfg!(feature = "cuda") { "gpu" } else { "cpu" };

    let output = Command::new("python3")
        .arg(&harness)
        .args(["--quiet", "--device", device])
        .current_dir(&root)
        .output()
        .expect("failed to launch verify_sd_pipeline_inference.py");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        output.status.success(),
        "harness exited non-zero ({:?}).\nstdout:\n{stdout}\nstderr:\n{stderr}",
        output.status,
    );
    assert!(
        stdout.contains("sd-v1-5-end-to-end-pipeline: PASS"),
        "expected 'sd-v1-5-end-to-end-pipeline: PASS' in stdout but got:\n{stdout}\n\nstderr:\n{stderr}"
    );
    assert!(
        !stdout.contains(" FAIL"),
        "stdout contains a FAIL verdict line:\n{stdout}\n\nstderr:\n{stderr}"
    );
}
