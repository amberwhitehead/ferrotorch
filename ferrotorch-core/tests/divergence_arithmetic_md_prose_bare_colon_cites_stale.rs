//! Audit divergence (commit `91ad29360`): `.design/ferrotorch-core/grad_fns/arithmetic.md`
//! BARE-COLON CONTINUATION CITES (e.g. ``RsqrtBackward struct at `:1540` ``)
//! in the REQ-N prose body rows and AC-N acceptance-criterion prose rows are
//! STILL pointing at PRE-`//!` HEADER-INSERTION line numbers, even though the
//! commit message claims "REQ-table impl cites refreshed" with the post-shift
//! line numbers.
//!
//! The fixer refreshed:
//!   - The REQ status table rows at the BOTTOM of the file (lines 902-917)
//!   - One block of `*Backward (arithmetic.rs:N-M)` cites in the architecture
//!     section (lines 428-707 — see commit diff `+`-lines)
//!
//! The fixer DID NOT refresh, but is required to per the commit-message claim
//! `Per R-CITE-2 every cite now resolves to the named symbol at HEAD`:
//!
//!   Line 115: ``RsqrtBackward` struct at `:1540` ``       — actual struct at :1565
//!   Line 128: ``ReciprocalBackward` struct at `:1702` ``  — actual at :1727
//!   Line 152: ``FloorDivideBackward` struct at `:2459` `` — actual at :2484
//!   Line 178: ``RemainderBackward` struct at `:1865` ``   — actual at :1890
//!   Line 204: ``FmodBackward` struct at `:2168` ``        — actual at :2193
//!   Line 229: ``AddcmulBackward` struct at `:2820` ``     — actual at :2845
//!   Line 253: ``AddcdivBackward` at `:3116` ``            — actual at :3141
//!
//! And the same seven cites repeated again in the AC-N rows:
//!
//!   Line 319: ``RsqrtBackward` (`:1540`) ``
//!   Line 327: ``ReciprocalBackward` (`:1702`) ``
//!   Line 335: ``RemainderBackward` (`:1865`) ``
//!   Line 343: ``FmodBackward` (`:2168`) ``
//!   Line 351: ``FloorDivideBackward` (`:2459`) ``
//!   Line 359: ``AddcmulBackward` (`:2820`) ``
//!   Line 369: ``AddcdivBackward` (`:3116`) ``
//!
//! 14 stale bare-colon cites in arithmetic.md REQ-N and AC-N prose. The
//! commit-fixer's own pinned tests only walk the REQ-status table at the
//! bottom of the file, so they miss every prose-body cite — the existing
//! audit shape is BLIND to this drift class.
//!
//! Per goal.md R-CITE-2 the bare-colon continuation cite `:NNN` still
//! resolves against the most recently mentioned file path on its line
//! (`grad_fns/arithmetic.rs` for the REQ-N prose rows; `arithmetic.rs` for
//! the AC-N rows). The "named symbol" the cite claims is `*Backward` —
//! `RsqrtBackward`, `ReciprocalBackward`, etc. — and the assertion is that
//! `grad_fns/arithmetic.rs:NNN` contains a `(pub )?struct *Backward` line.
//! It does NOT, because the rs-file is at HEAD and the doc is stale.
//!
//! Tracking: filed via crosslink (see audit report).

#![allow(clippy::missing_panics_doc)]

use std::fs;
use std::path::PathBuf;

fn workspace_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if !p.join(".design").exists() {
        p.pop();
    }
    p
}

/// Returns the line at `line_no` (1-indexed) of `file` if it exists.
fn line_at(file: &PathBuf, line_no: usize) -> Option<String> {
    fs::read_to_string(file)
        .ok()
        .and_then(|s| s.lines().nth(line_no - 1).map(str::to_string))
}

/// Each (doc_line_no, struct_name, cited_rs_line) the doc claims.
fn stale_backward_cites() -> Vec<(usize, &'static str, usize)> {
    vec![
        // REQ-N prose body cites (lines 100-280 of arithmetic.md).
        (115, "RsqrtBackward", 1540),
        (128, "ReciprocalBackward", 1702),
        (152, "FloorDivideBackward", 2459),
        (178, "RemainderBackward", 1865),
        (204, "FmodBackward", 2168),
        (229, "AddcmulBackward", 2820),
        (253, "AddcdivBackward", 3116),
        // AC-N prose cites (lines 316-370 of arithmetic.md).
        (319, "RsqrtBackward", 1540),
        (327, "ReciprocalBackward", 1702),
        (335, "RemainderBackward", 1865),
        (343, "FmodBackward", 2168),
        (351, "FloorDivideBackward", 2459),
        (359, "AddcmulBackward", 2820),
        (369, "AddcdivBackward", 3116),
    ]
}

#[test]
fn divergence_arithmetic_md_bare_colon_backward_cites_resolve_at_head() {
    let root = workspace_root();
    let md = root.join(".design/ferrotorch-core/grad_fns/arithmetic.md");
    let rs = root.join("ferrotorch-core/src/grad_fns/arithmetic.rs");

    let md_text = fs::read_to_string(&md).expect("read arithmetic.md");
    let md_lines: Vec<&str> = md_text.lines().collect();

    let mut errors: Vec<String> = Vec::new();

    for (doc_line_no, struct_name, cited_rs_line) in stale_backward_cites() {
        // 1. Confirm the doc-line really cites struct_name :cited_rs_line.
        let doc_line = match md_lines.get(doc_line_no - 1) {
            Some(s) => *s,
            None => {
                errors.push(format!(
                    "arithmetic.md has no line {doc_line_no} (file shorter than expected)"
                ));
                continue;
            }
        };
        let cite_needle1 = format!("`{struct_name}` struct at `:{cited_rs_line}`");
        let cite_needle2 = format!("`{struct_name}` (`:{cited_rs_line}`)");
        let cite_needle3 = format!("`{struct_name}` at `:{cited_rs_line}`");
        if !doc_line.contains(&cite_needle1)
            && !doc_line.contains(&cite_needle2)
            && !doc_line.contains(&cite_needle3)
        {
            errors.push(format!(
                "arithmetic.md:{doc_line_no} does not contain the expected stale cite\n  expected one of:\n    `{cite_needle1}`\n    `{cite_needle2}`\n    `{cite_needle3}`\n  actual line: {doc_line}"
            ));
            continue;
        }

        // 2. Resolve the cite against arithmetic.rs at HEAD: does line
        // `cited_rs_line` contain `(pub )?struct <struct_name>`?
        let rs_line = line_at(&rs, cited_rs_line).unwrap_or_default();
        let pub_struct = format!("pub struct {struct_name}");
        let plain_struct = format!("struct {struct_name}");
        if !rs_line.contains(&pub_struct) && !rs_line.contains(&plain_struct) {
            errors.push(format!(
                "arithmetic.md:{doc_line_no} cites `{struct_name}` at `grad_fns/arithmetic.rs:{cited_rs_line}` but that line at HEAD is:\n    `{rs_line}`\n  (expected to contain `struct {struct_name}` — the prose-body / AC-row cite was NOT refreshed by commit 91ad29360 even though the REQ-status-table row WAS)"
            ));
        }
    }

    assert!(
        errors.is_empty(),
        "arithmetic.md has stale bare-colon continuation cites in the REQ-N prose body and AC-N rows (R-CITE-2 violation — commit 91ad29360 refreshed the bottom REQ status table but left the prose duplicates pointing at pre-shift line numbers):\n\n{}\n\n14 stale cites total. The fixer's own pinned tests only walk `| REQ-N (...) | SHIPPED |` rows so they miss every prose-body duplicate cite.",
        errors.join("\n\n")
    );
}
