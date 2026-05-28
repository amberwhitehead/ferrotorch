//! Divergence audit of commit `b247d7dbc`.
//!
//! The commit message claims:
//!
//! ```text
//! VERIFICATION:
//!   divergence_cite_drift_generic: 4 passed, 0 failed
//! ```
//!
//! This is FALSE at HEAD. The same commit added a `## DIVERGENCE-1269-GAP-C-PROBE`
//! section to `.design/ferrotorch-core/grad_fns/arithmetic.md` containing two
//! deliberately-typo'd `.rs` cites (`arithmatic.rs:1565` and
//! `gradfns/arithmetic.rs:1565`). The same commit ALSO sharpened
//! `resolve_cite_path` to return `CitePath::Unresolved` (treated as a FAIL)
//! instead of silently skipping — the very Gap-C fix the audit-gap test
//! tracks under #1271.
//!
//! Combining the two changes makes `arithmetic_md_cites_resolve_at_head`
//! fail at HEAD with two unresolvable-cite errors. Confirmed by running
//! the test directly:
//!
//! ```text
//! arithmetic.md has 2 stale cite(s) (R-CITE-2):
//!  .design/ferrotorch-core/grad_fns/arithmetic.md:922 cites
//!    unresolvable path `arithmatic.rs:1565-1565` ...
//!  .design/ferrotorch-core/grad_fns/arithmetic.md:923 cites
//!    unresolvable path `gradfns/arithmetic.rs:1565-1565` ...
//! ```
//!
//! The probe section was committed as a permanent fixture in arithmetic.md;
//! it should have been (a) confined to a tmpfile inside the audit-gap test
//! that owns it, (b) gated behind a `#[ignore]`d test, or (c) removed once
//! the resolver was sharpened to fail on it.
//!
//! Generator must fix: either delete the probe block from
//! `.design/ferrotorch-core/grad_fns/arithmetic.md` (lines 919-924 in HEAD)
//! OR amend the commit message to retract the "4 passed, 0 failed" claim.

use std::path::PathBuf;
use std::process::Command;

fn workspace_root() -> PathBuf {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    PathBuf::from(manifest_dir).parent().unwrap().to_path_buf()
}

/// Re-run the generic cite-drift test the commit message advertises as
/// "4 passed, 0 failed" and assert it actually passes. At HEAD it does
/// not — the `arithmetic_md_cites_resolve_at_head` arm fails because of
/// the self-inflicted Gap-C probe section in arithmetic.md.
#[test]
fn divergence_b247_generic_test_does_not_pass_4_of_4_at_head() {
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
        "Commit b247d7dbc claims `divergence_cite_drift_generic: 4 passed, 0 failed` \
         but the test suite fails at HEAD because the same commit appended a \
         `## DIVERGENCE-1269-GAP-C-PROBE` section to arithmetic.md containing two \
         typo'd cites (`arithmatic.rs`, `gradfns/arithmetic.rs`) that the same \
         commit's resolver-sharpening now correctly rejects.\n\n\
         Generic-test output:\n{combined}",
    );
}
