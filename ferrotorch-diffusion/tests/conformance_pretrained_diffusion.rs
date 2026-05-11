//! Real-artifact pretrained SD-1.5 VAE-decoder parity test, gated on network.
//!
//! Runs `scripts/verify_diffusion_inference.py` end-to-end against the
//! pinned `ferrotorch/sd-v1-5-vae-decoder` HF mirror. Marked `#[ignore]`
//! since it requires network access (to first-touch the HF mirror) and
//! a Python environment with `diffusers`, `transformers`, `torch`,
//! `huggingface_hub`, `numpy` installed.
//!
//! Enable via:
//!
//! ```text
//! cargo test -p ferrotorch-diffusion --test conformance_pretrained_diffusion \
//!     -- --ignored
//! ```
//!
//! Mirrors the Whisper-side `conformance_pretrained_whisper_encoder`
//! and the BERT-side `conformance_pretrained_text_embedding` cargo
//! tests.

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

#[test]
#[ignore = "Requires network access — enable with --ignored"]
fn pretrained_sd_v1_5_vae_decoder_parity_smoke() {
    let root = workspace_root();
    let harness = root.join("scripts").join("verify_diffusion_inference.py");
    assert!(
        harness.is_file(),
        "harness missing at {}",
        harness.display()
    );

    let output = Command::new("python3")
        .arg(&harness)
        .args(["--models", "sd-v1-5-vae-decoder", "--quiet"])
        .current_dir(&root)
        .output()
        .expect("failed to launch verify_diffusion_inference.py");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        output.status.success(),
        "harness exited non-zero ({:?}).\nstdout:\n{stdout}\nstderr:\n{stderr}",
        output.status,
    );

    assert!(
        stdout.contains("sd-v1-5-vae-decoder: PASS"),
        "expected 'sd-v1-5-vae-decoder: PASS' in stdout but got:\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        !stdout.contains(" FAIL"),
        "stdout contains a FAIL verdict line:\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
}
