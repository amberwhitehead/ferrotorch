//! Audit divergence (commit `91ad29360`):
//! `.design/ferrotorch-core/grad_fns/cumulative.md` PROSE SECTIONS still cite
//! `cumulative.rs:NNN` line numbers from before the file-shift, AND the
//! REQ-6 / REQ-7 status-table rows (which the existing audit's pinned test
//! does NOT validate — that test walks `pub fn <op>` cites only, not
//! call-site tuple cites or test-fn-range cites) ALSO still carry the
//! pre-shift values.
//!
//! Commit message claim:
//!
//!   Prose architecture-section ranges (~lines 41-189 in cumulative.md)
//!   shifted to keep `cumulative.rs:N-M` cites pointing at the right code.
//!
//! The REFRESH at the REQ-status table rows REQ-1..REQ-5 landed. REQ-6
//! and REQ-7 did NOT, and the architecture-section prose still cites
//! pre-shift values for forwards, backwards, normalize_axis call sites,
//! reverse_cumsum call sites, and test-fn locations.
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

fn line_at(path: &PathBuf, line_no: usize) -> Option<String> {
    fs::read_to_string(path)
        .ok()
        .and_then(|s| s.lines().nth(line_no - 1).map(str::to_string))
}

#[test]
fn divergence_cumulative_md_prose_pub_fn_cites_resolve_at_head() {
    let root = workspace_root();
    let rs = root.join("ferrotorch-core/src/grad_fns/cumulative.rs");

    // Architecture-section / AC-row cites of the form
    //   `pub fn <op>` at `cumulative.rs:NNN(-MMM)` (or `:NNN(-MMM)` continuation).
    // Refreshed to HEAD line numbers post-#1267 (was: 322/524/72/202 pre-shift).
    // The tuple shape remains `(doc_line_no, cited_rs_line, op_name)` — both
    // the .md cite and this fixture now point at HEAD, so the test is
    // permanent regression coverage against future drift.
    let pub_fn_cites = vec![
        // (doc-line, cited rs line, op_name)
        (154, 995, "logcumsumexp"), // "ferrotorch implements forward via `logcumsumexp` at `cumulative.rs:995-1018`"
        (226, 995, "logcumsumexp"), // AC-5: "`cumulative.rs:995-1018 pub fn logcumsumexp`"
        (288, 105, "cumsum"),       // "`pub fn cumsum` at `:105-122`"
        (299, 400, "cumprod"),      // "`pub fn cumprod` at `:400-418`"
        (327, 995, "logcumsumexp"), // "`pub fn logcumsumexp` at `:995-1018`"
    ];

    let mut errors: Vec<String> = Vec::new();
    for (doc_line, rs_line, op) in pub_fn_cites {
        let actual = line_at(&rs, rs_line).unwrap_or_default();
        let needle = format!("pub fn {op}");
        if !actual.contains(&needle) {
            errors.push(format!(
                "cumulative.md:{doc_line} cites `pub fn {op}` at `cumulative.rs:{rs_line}` but that line at HEAD is:\n    `{actual}`\n  (expected to contain `pub fn {op}`)"
            ));
        }
    }

    assert!(
        errors.is_empty(),
        "cumulative.md prose `pub fn` cites do not resolve at HEAD (R-CITE-2 — the commit-msg `~10 prose ranges` claim leaves out these architecture-section + AC-row cites):\n\n{}",
        errors.join("\n\n")
    );
}

#[test]
fn divergence_cumulative_md_prose_struct_backward_cites_resolve_at_head() {
    let root = workspace_root();
    let rs = root.join("ferrotorch-core/src/grad_fns/cumulative.rs");

    // Refreshed to HEAD line numbers post-#1267 (was: 103/264 pre-shift).
    let struct_cites = vec![
        // (doc-line, cited rs line, struct)
        (72, 283, "CumprodBackward"), // "ferrotorch implements this as `CumprodBackward` at `cumulative.rs:283-388`"
        (158, 882, "LogcumsumexpBackward"), // "Backward is `LogcumsumexpBackward` at `cumulative.rs:882-927`"
        (295, 283, "CumprodBackward"),      // "`CumprodBackward<T>` at `:283-287` saves"
        (317, 882, "LogcumsumexpBackward"), // "`LogcumsumexpBackward<T>` at `:882-886` saves"
    ];

    let mut errors: Vec<String> = Vec::new();
    for (doc_line, rs_line, st) in struct_cites {
        let actual = line_at(&rs, rs_line).unwrap_or_default();
        let needle = format!("struct {st}");
        if !actual.contains(&needle) {
            errors.push(format!(
                "cumulative.md:{doc_line} cites `{st}` at `cumulative.rs:{rs_line}` but that line at HEAD is:\n    `{actual}`\n  (expected to contain `struct {st}`)"
            ));
        }
    }

    assert!(
        errors.is_empty(),
        "cumulative.md prose `*Backward` struct cites do not resolve at HEAD (R-CITE-2):\n\n{}",
        errors.join("\n\n")
    );
}

#[test]
fn divergence_cumulative_md_req6_normalize_axis_tuple_cite_stale() {
    // REQ-6 status table row at cumulative.md:442 (and architecture-section
    // sentences at :177 / :332) cites `cumulative.rs:73, :203, :231, :241,
    // :323` as the normalize_axis call sites. At HEAD the actual sites are
    // `:109, :404, :675, :716, :1004` (this is what the cumulative.rs //!
    // doc-comment correctly reflects, refreshed by commit 6cfaeb115). Commit
    // 91ad29360 was advertised as refreshing cumulative.md prose ranges but
    // missed the REQ-6 row + the two duplicate citations in prose.
    let root = workspace_root();
    let rs = root.join("ferrotorch-core/src/grad_fns/cumulative.rs");
    let md = root.join(".design/ferrotorch-core/grad_fns/cumulative.md");
    let rs_text = fs::read_to_string(&rs).expect("read cumulative.rs");
    let md_text = fs::read_to_string(&md).expect("read cumulative.md");
    let md_lines: Vec<&str> = md_text.lines().collect();

    // 1. confirm the stale tuple (substring `:73, :203, :231, :241, :323`) is
    // still present somewhere in cumulative.md.
    let stale_substr = ":73, :203, :231, :241, :323";
    let mut stale_hits: Vec<usize> = Vec::new();
    for (i, line) in md_lines.iter().enumerate() {
        if line.contains(stale_substr) {
            stale_hits.push(i + 1);
        }
    }
    let stale_substr_partial = ":73, :203, :231, :241,"; // line 332 wraps the tuple
    for (i, line) in md_lines.iter().enumerate() {
        if line.contains(stale_substr_partial) && !stale_hits.contains(&(i + 1)) {
            stale_hits.push(i + 1);
        }
    }

    // 2. confirm the stale-cited rs lines :73 / :203 / :231 / :241 / :323 do
    // NOT contain `normalize_axis(`.
    let stale_sites: [usize; 5] = [73, 203, 231, 241, 323];
    let mut wrongly_hits: Vec<usize> = Vec::new();
    for site in stale_sites {
        let line = rs_text.lines().nth(site - 1).unwrap_or("");
        if line.contains("normalize_axis(") {
            wrongly_hits.push(site);
        }
    }

    // 3. confirm the actual sites :109, :404, :675, :716, :1004 DO contain it.
    let actual_sites: [usize; 5] = [109, 404, 675, 716, 1004];
    let mut missing: Vec<usize> = Vec::new();
    for site in actual_sites {
        let line = rs_text.lines().nth(site - 1).unwrap_or("");
        if !line.contains("normalize_axis(") {
            missing.push(site);
        }
    }

    assert!(
        stale_hits.is_empty() && wrongly_hits.is_empty() && missing.is_empty(),
        "cumulative.md REQ-6 normalize_axis tuple-cite is stale (R-CITE-2):\n  - stale cite (subseq `{stale_substr}` or `{stale_substr_partial}`) still in cumulative.md at lines: {stale_hits:?}\n  - stale-cited rs lines that DON'T contain the call: {wrongly_hits:?} (good — confirms cite is stale)\n  - actual normalize_axis sites at HEAD :109/:404/:675/:716/:1004 — any missing: {missing:?}\n  cumulative.rs's own //!-header REQ table (refreshed by commit 6cfaeb115 #1266) correctly cites the new sites, but cumulative.md was NOT updated by commit 91ad29360."
    );
}

#[test]
fn divergence_cumulative_md_req7_reverse_cumsum_consumer_cite_stale() {
    // REQ-7 used to cite two stale consumers (:50 and :291). HEAD has one
    // real consumer: CumsumBackward::backward at :77. LogcumsumexpBackward now
    // uses the signed log-space implementation and must not be listed here.
    let root = workspace_root();
    let rs = root.join("ferrotorch-core/src/grad_fns/cumulative.rs");
    let md = root.join(".design/ferrotorch-core/grad_fns/cumulative.md");
    let rs_text = fs::read_to_string(&rs).expect("read cumulative.rs");
    let md_text = fs::read_to_string(&md).expect("read cumulative.md");

    let line77 = rs_text.lines().nth(76).unwrap_or("");
    let line889 = rs_text.lines().nth(888).unwrap_or("");

    assert!(
        md_text.contains("cumulative.rs:77"),
        "cumulative.md REQ-7 must cite the current CumsumBackward reverse_cumsum consumer at cumulative.rs:77"
    );
    assert!(
        !md_text.contains("`cumulative.rs:50`")
            && !md_text.contains("`cumulative.rs:291`")
            && !md_text.contains("`cumulative.rs:676`"),
        "cumulative.md REQ-7 must not retain stale reverse_cumsum consumer cites :50/:291/:676"
    );
    let logcumsumexp_reverse_claim = md_text.lines().any(|line| {
        line.contains("LogcumsumexpBackward::backward") && line.contains("reverse_cumsum")
    });
    assert!(
        !logcumsumexp_reverse_claim,
        "cumulative.md REQ-7 must not claim LogcumsumexpBackward consumes reverse_cumsum"
    );
    assert!(
        line77.contains("reverse_cumsum("),
        "expected cumulative.rs:77 to remain the reverse_cumsum call, got `{line77}`"
    );
    assert!(
        !line889.contains("reverse_cumsum("),
        "LogcumsumexpBackward::backward should not be the reverse_cumsum consumer, got `{line889}`"
    );
}
