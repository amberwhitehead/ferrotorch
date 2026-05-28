//! Post-#1215 divergence: `.design/ferrotorch-core/grad_fns/arithmetic.md`
//! prose still describes a separate `SubBackward` struct and a `sub_inner`
//! private helper, both of which were REMOVED in commit `d0fd83f1a` when
//! `arithmetic::sub` was changed to a one-line delegation `sub_scaled(a, b,
//! 1.0)` → `add_scaled(a, b, -1.0)`.
//!
//! ## The drift
//!
//! As of commit `2f792bfc5`:
//!
//! - `arithmetic.rs` contains NO `SubBackward` struct (only `AddScaledBackward`
//!   handles the VJP for the now-delegated `sub`). Verifiable via
//!   `grep '^struct SubBackward\|^pub struct SubBackward' ferrotorch-core/src/grad_fns/arithmetic.rs`
//!   → zero hits.
//! - `arithmetic.rs` contains NO `sub_inner` private function. Same grep with
//!   `^fn sub_inner` → zero hits.
//!
//! But the design doc at `.design/ferrotorch-core/grad_fns/arithmetic.md`
//! still claims:
//!
//! - Line ~56: "`sub_scaled` matches that contract byte-for-byte by
//!   delegating to `add_scaled(a, b, -alpha)`; the `SubBackward` VJP
//!   `(grad, -grad)` for the plain `sub` path is preserved separately"
//!   (preserves `SubBackward` as a distinct backward node, which it is NOT
//!   anymore — `sub` now goes through `AddScaledBackward`).
//! - Lines ~270-273: "`SubBackward` (`arithmetic.rs:790-813`) saves `a`/`b`;
//!   backward returns ... The forward `pub fn sub` (`arithmetic.rs:816-831`)
//!   and `sub_inner` (`arithmetic.rs:833-922`) follow the same shape as
//!   `add`" — three artifacts that no longer exist at those line ranges or
//!   anywhere in the file.
//! - Line ~330: "recursive use inside `SubBackward` / `DivBackward` here" —
//!   `SubBackward` is gone; `DivBackward` is the only one of that pair left.
//!
//! ## Why this is a divergence (R-CITE-2 / R-HONEST-4)
//!
//! goal.md R-CITE-2 mandates that every PyTorch citation carry a `file:line`
//! and R-HONEST-4 mandates that an audit-revealed wrong original commit be
//! corrected in code AND documented in the supplemental commit's body. The
//! delegation commit `d0fd83f1a` updated the REQ-2 STATUS-TABLE row but did
//! not update the design doc's architecture-section prose that describes the
//! same code. A reader looking at the doc to understand `sub`'s mechanics
//! will be told `SubBackward` exists, will grep for it, and will find it
//! present only in `ferrotorch-jit/src/{trace,graph_break}.rs` (where the JIT
//! tracer's known-op table maps a now-impossible name) — i.e., the doc is
//! actively misleading.
//!
//! Tracking: filed via crosslink (see report).

#![allow(clippy::missing_panics_doc)]

use std::fs;
use std::path::PathBuf;

fn read_arith_design_doc() -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crate dir has parent")
        .join(".design/ferrotorch-core/grad_fns/arithmetic.md");
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"))
}

fn read_arith_src() -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/grad_fns/arithmetic.rs");
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"))
}

/// Production assertion: `SubBackward` struct + `sub_inner` private fn were
/// removed in commit `d0fd83f1a`. Confirm via the source itself.
#[test]
fn baseline_sub_backward_and_sub_inner_truly_removed_from_src() {
    let src = read_arith_src();
    // Match the actual definition patterns; permit references inside
    // doc-comments / Cat-G mention in the conformance test docstring.
    assert!(
        !src.contains("struct SubBackward"),
        "arithmetic.rs MUST NOT contain `struct SubBackward` definition \
         (commit d0fd83f1a removed it)"
    );
    assert!(
        !src.contains("fn sub_inner"),
        "arithmetic.rs MUST NOT contain `fn sub_inner` definition \
         (commit d0fd83f1a removed it)"
    );
}

/// Divergence: the design-doc prose in `arithmetic.md` still mentions
/// `SubBackward` and `sub_inner` as if they exist. Per goal.md R-HONEST-4,
/// the supplemental fix should have updated the prose, not only the
/// status-table row. This test fails until the prose is reconciled.
#[test]
fn divergence_arith_design_doc_mentions_removed_subbackward_struct() {
    let doc = read_arith_design_doc();

    // Count occurrences of the removed symbol names in the doc.
    let subbackward_hits: Vec<usize> = doc
        .lines()
        .enumerate()
        .filter(|(_, l)| l.contains("SubBackward"))
        .map(|(i, _)| i + 1)
        .collect();
    let sub_inner_hits: Vec<usize> = doc
        .lines()
        .enumerate()
        .filter(|(_, l)| l.contains("sub_inner"))
        .map(|(i, _)| i + 1)
        .collect();

    assert!(
        subbackward_hits.is_empty(),
        "design doc still mentions removed `SubBackward` struct at lines \
         {subbackward_hits:?}; commit d0fd83f1a removed the struct from \
         arithmetic.rs but did not update the doc prose. Either remove the \
         mentions OR add a note explicitly stating `SubBackward` was \
         eliminated by the add_scaled delegation."
    );
    assert!(
        sub_inner_hits.is_empty(),
        "design doc still mentions removed `sub_inner` helper at lines \
         {sub_inner_hits:?}; same fix path as above."
    );
}

/// Divergence: REQ-2 prose at the architecture section (around lines 268-295)
/// still cites `arithmetic.rs:790-813`, `arithmetic.rs:816-831`,
/// `arithmetic.rs:833-922`, `arithmetic.rs:923-936` — but `sub` is now at
/// `arithmetic.rs:786-788` (a one-line delegation) and `sub_scaled` is at
/// `arithmetic.rs:815-824`. The four old line-ranges no longer point at the
/// content the doc claims. Per goal.md R-CITE-2: a file path without an
/// accurate line is not a citation.
#[test]
fn divergence_arith_design_doc_cites_stale_line_ranges_for_sub() {
    let doc = read_arith_design_doc();
    let src = read_arith_src();

    // The doc's REQ-2 architecture section claims SubBackward is at
    // arithmetic.rs:790-813. Verify the source at those lines.
    let src_lines: Vec<&str> = src.lines().collect();
    let stale_ranges = [
        ("790-813", 790_usize, 813_usize, "SubBackward"),
        ("816-831", 816_usize, 831_usize, "pub fn sub"),
        ("833-922", 833_usize, 922_usize, "fn sub_inner"),
    ];

    // Each (start, end, claimed_content) is mentioned in the doc. We assert
    // that for each cited range, the source code in that range does NOT
    // contain the claimed_content — i.e., the doc's citation is stale.
    let mut stale_findings: Vec<String> = Vec::new();
    for (label, start, end, claimed) in stale_ranges {
        let doc_mentions_range = doc.contains(label);
        if !doc_mentions_range {
            continue;
        }
        let s = start.saturating_sub(1);
        let e = end.min(src_lines.len());
        let body = src_lines[s..e].join("\n");
        if !body.contains(claimed) {
            stale_findings.push(format!(
                "doc cites `arithmetic.rs:{label}` for `{claimed}` but the \
                 source at those lines no longer contains `{claimed}`"
            ));
        }
    }

    assert!(
        stale_findings.is_empty(),
        "arithmetic.md REQ-2 architecture section cites stale line ranges \
         that no longer point at the claimed code (per goal.md R-CITE-2):\n  {}",
        stale_findings.join("\n  ")
    );
}
