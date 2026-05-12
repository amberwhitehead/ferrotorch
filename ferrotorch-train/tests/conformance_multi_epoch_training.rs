//! Real-artifact multi-epoch training-trajectory parity test, gated on
//! network access.
//!
//! Runs `scripts/verify_training_trajectory.py` end-to-end against the
//! pinned `ferrotorch/training-trajectory-v1` HF mirror (#1161). The
//! Python harness:
//!
//!   1. Pulls `initial_state.safetensors`, `X_full.bin`, `y_full.bin`,
//!      `meta.json`, and `epoch_{0..5}_state.safetensors` from HF.
//!   2. Invokes `multi_epoch_train_dump` (this crate's example) to run
//!      the live ferrotorch training loop (forward + MSE + LIVE
//!      autograd backward + Adam.step()) for 5 epochs.
//!   3. Compares per-epoch state_dicts and per-epoch mean losses to
//!      the torch reference under
//!      `max_abs <= 1e-4` AND `cosine_sim >= 0.9999`.
//!
//! Marked `#[ignore]` since it requires network access (to first-touch
//! the HF mirror) and a Python environment with `huggingface_hub`,
//! `numpy`, `safetensors` installed.
//!
//! Enable via:
//!
//! ```text
//! cargo test -p ferrotorch-train --test conformance_multi_epoch_training \
//!     -- --ignored
//! ```
//!
//! Mirrors the optimizer / diffusion / Whisper / BERT / SmolLM
//! real-artifact conformance test wrappers in shape: shell out to the
//! Python harness, assert on its `PASS` verdict line for every model.

use std::path::PathBuf;
use std::process::Command;

/// Resolve the workspace root from this crate's `CARGO_MANIFEST_DIR`.
fn workspace_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .expect("ferrotorch-train manifest must have a parent (the workspace root)")
        .to_path_buf()
}

/// Every model the harness verifies. Asserting on a specific
/// `<name>: PASS` line (rather than just `PASS`) catches a future
/// regression that silently skips a model and reports the (now smaller)
/// remaining set as all-PASS. The naming matches the `MODELS` map in
/// `scripts/verify_training_trajectory.py`.
const EXPECTED_PASS_LINES: &[&str] = &["training-trajectory-v1: PASS"];

#[test]
#[ignore = "Requires network access — enable with --ignored"]
fn pretrained_multi_epoch_training_parity_smoke() {
    let root = workspace_root();
    let harness = root.join("scripts").join("verify_training_trajectory.py");
    assert!(
        harness.is_file(),
        "harness missing at {}",
        harness.display()
    );

    let output = Command::new("python3")
        .arg(&harness)
        .arg("--models")
        .arg("training-trajectory-v1")
        .arg("--quiet")
        .current_dir(&root)
        .output()
        .expect("failed to launch verify_training_trajectory.py");

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
