//! Real-artifact conformance test for ferrotorch-core autograd
//! backward parity vs torch.autograd (Phase G.5, #1171).
//!
//! Calls `scripts/verify_autograd_inference.py` which:
//!
//!   1. Resolves the `ferrotorch/autograd-parity-v1` fixture tree
//!      (either from the pin script's WORK_DIR, or via
//!      `huggingface_hub.hf_hub_download`).
//!   2. For each of the 25 (op, config) fixtures, drives the
//!      `cargo run --example autograd_dump` binary to replay the
//!      forward + backward through `ferrotorch-core`'s differentiable
//!      surface and dump the rust gradients.
//!   3. Compares each rust gradient + forward output against the
//!      frozen torch.autograd reference with the gradcheck tolerances
//!      `max_abs <= 1e-4` AND `cosine_sim >= 0.9999`.
//!
//! Gated behind `#[ignore]` so the test is network-aware and only
//! runs on operator request:
//!
//! ```text
//! cargo test --test conformance_autograd_parity -- --ignored
//! ```

#![allow(clippy::missing_panics_doc)]

use std::path::PathBuf;
use std::process::Command;

/// Repository root resolved from `CARGO_MANIFEST_DIR`. The manifest
/// lives at `ferrotorch-core/Cargo.toml`, so the parent is the
/// workspace root.
fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("ferrotorch-core/Cargo.toml must have a parent (workspace root)")
        .to_path_buf()
}

#[test]
#[ignore = "network-aware real-artifact harness; run with --ignored"]
fn autograd_backward_parity_via_python_harness() {
    let script = repo_root().join("scripts/verify_autograd_inference.py");
    assert!(
        script.exists(),
        "verify_autograd_inference.py not found at {}",
        script.display()
    );

    let output = Command::new("python3")
        .arg(&script)
        .current_dir(repo_root())
        .output()
        .expect("failed to spawn python3");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    eprintln!("---- verify_autograd_inference.py stdout ----\n{stdout}");
    if !stderr.is_empty() {
        eprintln!("---- verify_autograd_inference.py stderr ----\n{stderr}");
    }

    assert!(
        output.status.success(),
        "verify_autograd_inference.py exited with {:?}; PASS line not produced",
        output.status
    );
    // Belt-and-braces: the script's last line should be `OVERALL: PASS`.
    assert!(
        stdout.contains("OVERALL: PASS"),
        "verify_autograd_inference.py did not report OVERALL: PASS"
    );
}
