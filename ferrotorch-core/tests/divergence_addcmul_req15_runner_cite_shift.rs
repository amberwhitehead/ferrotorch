//! Divergence test for commit `9036936a0` (addcmul REQ-15 audit, blocker
//! #1200).
//!
//! The addcmul commit inserted two new helpers in
//! `tools/parity-sweep/runner/src/main.rs` — `ternary()` (lines 230-247) and
//! `value_kwarg()` (lines 263-270) — in the body of `dispatch_f32`, BEFORE
//! the existing op-arm `match` block. Each helper adds ~17 lines and ~8
//! lines respectively, shifting EVERY downstream op-arm's line number by ~44.
//!
//! However, the commit did NOT update the REQ-9..REQ-14 SHIPPED rows in
//! `.design/ferrotorch-core/grad_fns/arithmetic.md`, which still cite the
//! pre-commit line numbers for those op-arms. This is the identical
//! "citation-theater" anti-pattern that #1203 documented for REQ-8 and that
//! `divergence_rsub_req9_stale_cites.rs` was written to catch — except now
//! the shift affects 6 additional REQs at once.
//!
//! ## Actual op-arm lines at HEAD (post-commit 9036936a0)
//!
//! Verified empirically via `grep -n '"<op>" =>'
//! tools/parity-sweep/runner/src/main.rs` on 2026-05-25:
//!
//! ```text
//!   "rsub"         => actual 297   (cited as 253 in arithmetic.md REQ-9)
//!   "rsqrt"        => actual 320   (cited as 276 in REQ-10)
//!   "reciprocal"   => actual 328   (cited as 284 in REQ-11)
//!   "remainder"    => actual 341   (cited as 297 in REQ-13)
//!   "fmod"         => actual 355   (cited as 311 in REQ-14)
//!   "floor_divide" => actual 374   (cited as 330 in REQ-12)
//!   "pow"          => actual 407   (cited as 232 in REQ-8)
//!   "addcmul"      => actual 390   (cited as 390 in REQ-15)  <- only this one is fresh
//! ```
//!
//! Every cite except the just-added REQ-15 row is stale by 18-175 lines.
//!
//! ## Why this is a regression beyond the prior REQ-9 test
//!
//! The prior `divergence_rsub_req9_stale_cites.rs` was a one-off for the
//! REQ-9 row when its 10-line drift was introduced by the original rsub
//! shipper. This audit test is broader: it pins the convention that
//! ANY commit that inserts code above the runner-arm `match` block MUST
//! update every downstream SHIPPED-REQ row's cite — and a single forgotten
//! cite trips the test. The test is parameterized over (REQ-N, op-name)
//! pairs, so adding REQ-16 (addcdiv) on the next commit is one new tuple.
//!
//! Per goal.md:
//!   - R-CITE-2 (cite with file:line, not just file)
//!   - R-HONEST-2 (SHIPPED requires impl AND cited production consumer)
//!   - R-DEFER-3 (no "acceptable drift" — every divergence is real work)
//!
//! Per R-CHAR-3 the expected anchor (e.g. `"rsub" =>`) is constructed at
//! test time from the op-name, NOT copy-pasted from arithmetic.md. The test
//! reads main.rs to discover where the anchor ACTUALLY lives and compares
//! against the cite line in the doc row.
//!
//! Tracking: file a `crosslink quick` blocker per the new divergence;
//! discriminator reports the failing test path.

use std::fs;
use std::path::PathBuf;

fn locate_design_doc() -> PathBuf {
    let candidates = [
        "../.design/ferrotorch-core/grad_fns/arithmetic.md",
        ".design/ferrotorch-core/grad_fns/arithmetic.md",
    ];
    for c in candidates {
        let p = PathBuf::from(c);
        if p.exists() {
            return p;
        }
    }
    panic!("could not locate arithmetic.md");
}

fn locate_runner_main_rs() -> PathBuf {
    let candidates = [
        "../tools/parity-sweep/runner/src/main.rs",
        "tools/parity-sweep/runner/src/main.rs",
    ];
    for c in candidates {
        let p = PathBuf::from(c);
        if p.exists() {
            return p;
        }
    }
    panic!("could not locate tools/parity-sweep/runner/src/main.rs");
}

/// Find the 1-indexed line of `anchor` in `text`. Returns the first match.
fn find_anchor_line(text: &str, anchor: &str) -> Option<usize> {
    for (i, line) in text.lines().enumerate() {
        if line.contains(anchor) {
            return Some(i + 1);
        }
    }
    None
}

/// Extract the LONGEST line of the design doc that begins (after optional
/// whitespace) with `| REQ-<n>` — the canonical SHIPPED-status table row.
fn extract_req_row(doc: &str, req_n: u32) -> Option<String> {
    let want = format!("REQ-{req_n}");
    let mut best: Option<String> = None;
    for line in doc.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix('|') {
            let cell0 = rest.trim_start();
            // Must match REQ-N exactly (not REQ-N0/etc); require the next
            // char to be space, `)`, `|`, or end.
            if cell0.starts_with(&want) {
                let after = &cell0[want.len()..];
                let ok = after
                    .chars()
                    .next()
                    .map(|c| c == ' ' || c == ')' || c == '|' || c == '\t')
                    .unwrap_or(true);
                if ok {
                    let s = line.to_string();
                    if best.as_ref().is_none_or(|b| s.len() > b.len()) {
                        best = Some(s);
                    }
                }
            }
        }
    }
    best
}

/// Find the cite line of the form
/// `tools/parity-sweep/runner/src/main.rs:<DIGITS>` in `row`. Returns the
/// first integer following the `.rs:` suffix.
fn extract_runner_cite_line(row: &str) -> Option<usize> {
    let key = "tools/parity-sweep/runner/src/main.rs:";
    let pos = row.find(key)?;
    let after = &row[pos + key.len()..];
    let mut end = 0;
    for c in after.chars() {
        if c.is_ascii_digit() {
            end += c.len_utf8();
        } else {
            break;
        }
    }
    if end == 0 {
        return None;
    }
    after[..end].parse::<usize>().ok()
}

/// (REQ-N, op-name, runner-arm anchor) tuples for every SHIPPED row in
/// arithmetic.md that cites a parity-sweep runner arm. Adding a new REQ
/// (e.g. REQ-16 addcdiv) is one new tuple here.
fn audit_targets() -> Vec<(u32, &'static str, String)> {
    vec![
        (8, "pow", "\"pow\" =>".to_string()),
        (9, "rsub", "\"rsub\" =>".to_string()),
        (10, "rsqrt", "\"rsqrt\" =>".to_string()),
        (11, "reciprocal", "\"reciprocal\" =>".to_string()),
        (12, "floor_divide", "\"floor_divide\" =>".to_string()),
        (13, "remainder", "\"remainder\" =>".to_string()),
        (14, "fmod", "\"fmod\" =>".to_string()),
        (15, "addcmul", "\"addcmul\" =>".to_string()),
    ]
}

#[test]
fn divergence_addcmul_req15_runner_arm_cites_resolve_post_helper_insert() {
    let doc_path = locate_design_doc();
    let runner_path = locate_runner_main_rs();

    let doc = fs::read_to_string(&doc_path)
        .unwrap_or_else(|e| panic!("could not read {}: {}", doc_path.display(), e));
    let runner = fs::read_to_string(&runner_path)
        .unwrap_or_else(|e| panic!("could not read {}: {}", runner_path.display(), e));

    let targets = audit_targets();
    let mut violations: Vec<String> = Vec::new();
    let mut audited: Vec<String> = Vec::new();

    for (req_n, op, anchor) in &targets {
        let Some(row) = extract_req_row(&doc, *req_n) else {
            // Row doesn't exist (e.g. REQ-12 might use a different format);
            // skip rather than fault.
            continue;
        };
        let Some(cite_line) = extract_runner_cite_line(&row) else {
            // Row exists but does not cite the runner. Not all SHIPPED rows
            // need a runner cite; skip.
            continue;
        };
        // Discover the actual line of the runner arm.
        let Some(actual_line) = find_anchor_line(&runner, anchor) else {
            violations.push(format!(
                "REQ-{req_n} ({op}): anchor `{anchor}` not found anywhere in runner main.rs; \
                 either the arm was removed or the anchor format changed"
            ));
            continue;
        };

        audited.push(format!(
            "REQ-{req_n} ({op}): cite={cite_line} actual={actual_line}"
        ));

        // Tolerance window: 4 lines (matches the rsub stale-cite test).
        let diff = (actual_line as i64 - cite_line as i64).abs();
        if diff > 4 {
            violations.push(format!(
                "REQ-{req_n} ({op}): arithmetic.md cites runner arm at \
                 tools/parity-sweep/runner/src/main.rs:{cite_line} but the \
                 anchor `{anchor}` is actually at line {actual_line} \
                 (drift = {diff} lines, exceeds 4-line tolerance)"
            ));
        }
    }

    assert!(
        !audited.is_empty(),
        "test audited zero REQ rows — either the design doc was reformatted \
         or the row extractor is broken"
    );

    let doc_display = doc_path.display().to_string();
    assert!(
        violations.is_empty(),
        "REQ rows in {} carry STALE tools/parity-sweep/runner/src/main.rs \
         cites — commit `9036936a0` (addcmul) inserted two new helpers in \
         `dispatch_f32` (`ternary()` at L230-247 + `value_kwarg()` at \
         L263-270) which shifted every downstream op-arm by ~44 lines, but \
         the commit did not update the prior REQ rows' runner cites.\n\n\
         Stale cite(s):\n  - {}\n\n\
         Audited:\n  - {}\n\n\
         Per goal.md R-CITE-2 (cite with file:line, not just file) and \
         R-HONEST-2 (SHIPPED requires cited production consumer), the doc \
         must be updated when the cites move. Fix: update each REQ-N row's \
         `tools/parity-sweep/runner/src/main.rs:<L>` cite to the current \
         line of the arm.",
        doc_display,
        violations.join("\n  - "),
        audited.join("\n  - "),
    );
}
