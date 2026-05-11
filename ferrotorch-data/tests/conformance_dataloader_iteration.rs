//! Real-artifact DataLoader iteration parity tests, gated on network.
//!
//! Runs `scripts/verify_dataloader_inference.py` end-to-end against the
//! pinned `ferrotorch/dataloader-batches-v1` HF mirror (Phase C.3 of
//! real-artifact-driven development, #1156). Marked `#[ignore]` since it
//! requires network access (to first-touch the HF mirror) and a Python
//! environment with `huggingface_hub`, `numpy`, and `torch` installed.
//!
//! Enable via:
//!
//! ```text
//! cargo test -p ferrotorch-data --test conformance_dataloader_iteration \
//!     -- --ignored
//! ```
//!
//! Mirrors the diffusion / Whisper / BERT / SmolLM / optimizer
//! real-artifact conformance test wrappers in shape: shell out to the
//! Python harness, assert on its PASS verdict lines.

use std::path::PathBuf;
use std::process::Command;

/// Resolve the workspace root from this crate's `CARGO_MANIFEST_DIR`.
fn workspace_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .expect("ferrotorch-data manifest must have a parent (the workspace root)")
        .to_path_buf()
}

/// Every config the harness verifies. Asserting on `<name>: PASS` rather
/// than `PASS` alone catches a future regression that silently skips a
/// config and reports the (now smaller) remaining set as all-PASS. The
/// naming matches the configs in
/// `scripts/pin_pretrained_dataloader_batches.py`.
const EXPECTED_PASS_LINES: &[&str] = &[
    "sequential: PASS",
    "sequential_droplast: PASS",
    "shuffled_seeded: PASS",
    "shuffled_droplast: PASS",
    "batch_size_3: PASS",
];

#[test]
#[ignore = "Requires network access — enable with --ignored"]
fn pretrained_dataloader_iteration_parity_smoke() {
    let root = workspace_root();
    let harness = root.join("scripts").join("verify_dataloader_inference.py");
    assert!(
        harness.is_file(),
        "harness missing at {}",
        harness.display()
    );

    let output = Command::new("python3")
        .arg(&harness)
        .arg("--quiet")
        .current_dir(&root)
        .output()
        .expect("failed to launch verify_dataloader_inference.py");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        output.status.success(),
        "harness exited non-zero ({:?}).\nstdout:\n{stdout}\nstderr:\n{stderr}",
        output.status,
    );

    for expected in EXPECTED_PASS_LINES {
        assert!(
            stdout.contains(expected),
            "expected '{expected}' in stdout but got:\nstdout:\n{stdout}\nstderr:\n{stderr}"
        );
    }
    assert!(
        !stdout.contains(" FAIL"),
        "stdout contains a FAIL verdict line:\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
}
