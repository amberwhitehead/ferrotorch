//! Regression audit of commit `b247d7dbc`.
//!
//! The commit message claims:
//!
//! ```text
//! VERIFICATION:
//!   divergence_cite_drift_generic: 4 passed, 0 failed
//! ```
//!
//! The original divergence was that the generic cite-drift test did not pass
//! after the cite resolver was tightened. This regression keeps the claimed
//! gate honest: the full generic cite sweep must pass at HEAD.

use std::path::PathBuf;
use std::process::Command;

fn workspace_root() -> PathBuf {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    PathBuf::from(manifest_dir).parent().unwrap().to_path_buf()
}

/// Re-run the generic cite-drift test the commit message advertises as
/// "4 passed, 0 failed" and assert it actually passes.
#[test]
fn divergence_b247_generic_test_passes_4_of_4_at_head() {
    let root = workspace_root();
    let out = Command::new("cargo")
        .args([
            "test",
            "-p",
            "ferrotorch-core",
            "--test",
            "divergence_cite_drift_generic",
            "--no-fail-fast",
        ])
        .current_dir(&root)
        .output()
        .expect("cargo test invocation failed");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let combined = format!("{stdout}\n{stderr}");
    assert!(
        out.status.success(),
        "Commit b247d7dbc claims `divergence_cite_drift_generic: 4 passed, 0 failed`, \
         so the generic cite-drift test must stay green at HEAD.\n\n\
         Generic-test output:\n{combined}",
    );
}
