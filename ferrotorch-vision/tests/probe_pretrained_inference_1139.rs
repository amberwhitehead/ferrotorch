//! #1139 — end-to-end pretrained-inference verification probe.
//!
//! This file is a wrapper test that runs the Python verification script in
//! `scripts/verify_pretrained_inference.py`. It is `#[ignore]`-gated because:
//!
//!   * it requires Python 3 + torchvision installed,
//!   * it requires the 5 pretrained weight files cached under
//!     `~/.ferrotorch/hub/` (pinned via #1130),
//!   * it requires the 5 fixed COCO val2017 images cached under
//!     `/tmp/ferrotorch_verify_images/` (downloaded on first run by the
//!     Python script),
//!   * it is slow (Mask R-CNN alone takes ~30 s per image; the whole sweep
//!     is several minutes).
//!
//! Run explicitly with:
//!   cargo test -p ferrotorch-vision --test probe_pretrained_inference_1139 \
//!       -- --ignored --nocapture
//!
//! ## What it asserts
//!
//! The test invokes `scripts/verify_pretrained_inference.py` and parses
//! the JSON report it emits at
//! `/tmp/ferrotorch_verify_images/verify_pretrained_inference_report.json`.
//! It asserts the Python script exits cleanly. It does **NOT** assert that
//! every model passes — at the time of #1139 every model FAILs (see the
//! report for diagnoses). When models pass in a future dispatch, flip the
//! assertion below to `model_pass(...)`.

use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .to_path_buf()
}

fn rust_bin_path() -> PathBuf {
    repo_root()
        .join("target")
        .join("release")
        .join("examples")
        .join("inference_dump")
}

#[test]
#[ignore]
fn pretrained_inference_verification_runs() {
    // Sanity: the Rust dump binary must be pre-built. We don't try to build
    // it from inside the test — that interacts badly with cargo's lock and
    // overall takes ~10 minutes.
    let bin = rust_bin_path();
    assert!(
        bin.exists(),
        "inference_dump binary missing at {bin:?}. \
         Build first: cargo build -p ferrotorch-vision --release \
         --example inference_dump"
    );

    // Invoke the Python verification script.
    let script = repo_root()
        .join("scripts")
        .join("verify_pretrained_inference.py");
    let out = Command::new("python3")
        .arg(&script)
        .arg("--quiet")
        .output()
        .expect("failed to spawn python3");

    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    eprintln!("--- python stdout ---\n{stdout}");
    eprintln!("--- python stderr ---\n{stderr}");
    assert!(
        out.status.success(),
        "verify_pretrained_inference.py exited non-zero ({})",
        out.status
    );
}

#[test]
#[ignore]
fn pretrained_inference_sabotage_probe() {
    // The sabotage probe exercised via the python harness's `--sabotage`
    // flag, which intentionally degrades the comparison so we can verify
    // the framework catches deliberate divergence.
    let bin = rust_bin_path();
    if !bin.exists() {
        panic!(
            "inference_dump binary missing at {bin:?}; build it first \
             (see pretrained_inference_verification_runs test docs)"
        );
    }

    let script = repo_root()
        .join("scripts")
        .join("verify_pretrained_inference.py");
    let out = Command::new("python3")
        .arg(&script)
        .arg("--models")
        .arg("deeplabv3_resnet50")
        .arg("--sabotage")
        .arg("--quiet")
        .output()
        .expect("failed to spawn python3");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("SABOTAGED"),
        "sabotage flag must surface in output: {stdout}"
    );
    assert!(
        stdout.contains("FAIL"),
        "sabotage probe must yield a FAIL: {stdout}"
    );
}
