//! Real-artifact pretrained text-embedding parity test, gated on network.
//!
//! Runs `scripts/verify_text_embedding_inference.py` end-to-end against
//! the pinned `ferrotorch/all-MiniLM-L6-v2` HF mirror. Marked `#[ignore]`
//! since it requires network access (to first-touch the HF mirror) and a
//! Python environment with `sentence_transformers`, `transformers`,
//! `torch`, `huggingface_hub`, `numpy` installed.
//!
//! Enable via:
//!
//! ```text
//! cargo test -p ferrotorch-bert --test conformance_pretrained_text_embedding \
//!     -- --ignored
//! ```
//!
//! Mirrors the vision-side `conformance_pretrained_inference` and the
//! causal-LM `conformance_pretrained_causal_lm` cargo tests.

use std::path::PathBuf;
use std::process::Command;

/// Resolve the workspace root from this crate's `CARGO_MANIFEST_DIR`.
fn workspace_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .expect("ferrotorch-bert manifest must have a parent (the workspace root)")
        .to_path_buf()
}

#[test]
#[ignore = "Requires network access — enable with --ignored"]
fn pretrained_text_embedding_parity_smoke() {
    let root = workspace_root();
    let harness = root
        .join("scripts")
        .join("verify_text_embedding_inference.py");
    assert!(
        harness.is_file(),
        "harness missing at {}",
        harness.display()
    );

    let output = Command::new("python3")
        .arg(&harness)
        .args(["--models", "all-MiniLM-L6-v2", "--quiet"])
        .current_dir(&root)
        .output()
        .expect("failed to launch verify_text_embedding_inference.py");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        output.status.success(),
        "harness exited non-zero ({:?}).\nstdout:\n{stdout}\nstderr:\n{stderr}",
        output.status,
    );

    assert!(
        stdout.contains("all-MiniLM-L6-v2: PASS"),
        "expected 'all-MiniLM-L6-v2: PASS' in stdout but got:\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        !stdout.contains(" FAIL"),
        "stdout contains a FAIL verdict line:\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
}
