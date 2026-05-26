//! Divergence: the audit-infrastructure gap that allowed #1274 to ship as
//! a manual cite-refresh remains unresolved. `divergence_cite_drift_generic`
//! is scoped to `arithmetic.md` + `cumulative.md`; `indexing.md` is NOT
//! scanned. That is precisely why the stale REQ-8 cites in indexing.md
//! escaped pre-merge detection and required #1274 as a one-off fix.
//!
//! The #1274 commit fixed the symptom (3 specific stale cites in REQ-8)
//! but did NOT extend the generic audit to cover indexing.md — meaning the
//! NEXT time someone shifts a line in indexing.rs, the generic test will
//! still be blind and another one-off #1274-style fix will be required.
//!
//! Per goal.md S3 ("Symbol anchors in design-doc cites, NEVER line
//! numbers... Line numbers in `.design/` cites are forbidden — they spawn
//! cite-drift fixer dispatches every commit"), the audit infrastructure
//! must enforce the discipline workspace-wide, not on a hand-picked subset
//! of files. The generic-cite-drift test is the durable contract; that
//! contract must cover EVERY `.design/**/*.md` file.
//!
//! As of HEAD, indexing.md contains 51 line-number cites pointing into
//! `indexing.rs` / `methods.rs`, every one of which is a future cite-drift
//! exposure. Closing this gap means either:
//!   (a) extending divergence_cite_drift_generic.rs to scan indexing.md
//!       (and ideally every .design/**/*.md file) — same scanner, broader
//!       file list, OR
//!   (b) converting every line-number cite in indexing.md to a symbol
//!       anchor per the S3 discipline so the file has no line cites left
//!       to drift.
//!
//! Either fix closes the divergence; this test fails until one of them
//! lands.
//!
//! Tracking: blocker (filed by acto-critic).

use std::fs;
use std::path::PathBuf;

fn workspace_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if !p.join(".design").exists() {
        p.pop();
    }
    p
}

/// Either the generic audit covers indexing.md, OR indexing.md has been
/// fully converted to symbol anchors. Currently neither is true.
#[test]
fn indexing_md_must_be_covered_by_generic_audit_or_use_symbol_anchors() {
    let root = workspace_root();

    // Path 1: does the generic audit scan indexing.md?
    let generic = root.join("ferrotorch-core/tests/divergence_cite_drift_generic.rs");
    let generic_text = fs::read_to_string(&generic).expect("read generic cite-drift test");
    let covered_by_generic = generic_text.contains("indexing.md");

    // Path 2: is indexing.md free of `indexing.rs:NNN` / `methods.rs:NNN`
    // / `cumulative.rs:NNN` line-number cites (i.e. fully symbol-anchored)?
    let md = root.join(".design/ferrotorch-core/grad_fns/indexing.md");
    let md_text = fs::read_to_string(&md).expect("read indexing.md");
    let line_num_cite_re = regex_lite_count(&md_text);

    assert!(
        covered_by_generic || line_num_cite_re == 0,
        "indexing.md has {} workspace-internal line-number cite(s) AND is \
         NOT covered by divergence_cite_drift_generic.rs. Per goal.md S3 \
         the generic audit must catch this drift category. Fix by EITHER \
         extending the generic audit to scan indexing.md OR converting \
         all line-number cites in indexing.md to symbol anchors.",
        line_num_cite_re
    );
}

/// Count `<rs-file>.rs:<digits>` cites in the design doc text, restricted to
/// the workspace-internal files (indexing.rs / methods.rs / ops/indexing.rs)
/// — NOT upstream `aten/src/...:NNN` cites which S3 explicitly permits
/// ("Upstream cites (read-only) still use `file:line`").
fn regex_lite_count(text: &str) -> usize {
    // Match `[indexing|methods].rs:<digit>` in the document. We don't
    // attempt to exclude false positives — the design doc shouldn't have
    // any of these substrings outside cites anyway.
    let needles = ["indexing.rs:", "methods.rs:"];
    let mut count = 0;
    for needle in needles {
        let mut start = 0;
        while let Some(pos) = text[start..].find(needle) {
            let after = start + pos + needle.len();
            if after < text.len() && text.as_bytes()[after].is_ascii_digit() {
                count += 1;
            }
            start = after;
        }
    }
    count
}
