//! Divergence test for #1203: REQ-8 (pow) design-doc row cites a test-block line.
//!
//! `.design/ferrotorch-core/grad_fns/arithmetic.md` carries a `## REQ status`
//! table whose REQ-8 (pow) SHIPPED row enumerates "non-test production
//! consumers" of `arithmetic::pow`. Per goal.md every cite in that row must
//! be a real production caller:
//!
//!   - goal.md:73 — "SHIPPED ... fn `<name>` at `<file>:<L>` per upstream
//!     `<pytorch-file>:<L>` (consumer at `<caller-file>:<L>`)"
//!   - goal.md:150 — R-HONEST-2: "SHIPPED requires both implementation AND
//!     a non-test production consumer cited."
//!   - goal.md (R-DEFER-1) — "Test-only callers don't count."
//!
//! The closing commit 2debdcf9e listed three cite-targets in the REQ-8 row.
//! One of them — `ferrotorch-core/src/autograd/graph.rs:876` — points INSIDE
//! the `#[cfg(test)] mod tests` block of `graph.rs` (the test module opens at
//! `graph.rs:653` immediately after `#[cfg(test)]` at line 652; the cited
//! line is the body of `test_backward_one_element_through_pow_and_add` at
//! `graph.rs:870`). Citing a test-block line as a "non-test consumer" is the
//! citation-theater shape R-HONEST-2 forbids.
//!
//! ## What this test does
//!
//! Rather than hardcoding `876`/`graph.rs` (which would make the assertion
//! tautological per R-CHAR-3 — see acto-critic.md §"R-CHAR-3 — no
//! tautological tests"), the test:
//!
//!   1. Reads `.design/ferrotorch-core/grad_fns/arithmetic.md` and locates
//!      the row whose first cell begins with `| REQ-8`.
//!   2. Parses every cite of the form `ferrotorch-*/src/<path>.rs:<L>` (with
//!      `:L1,L2` comma-separated multi-line variants treated as multiple
//!      cites — the existing row uses `grad_penalty.rs:111,118`).
//!   3. For each cite, opens the source file at the workspace root and
//!      locates the FIRST `#[cfg(test)]` annotation followed (on the next
//!      non-blank line) by `mod tests`.
//!   4. Asserts that the cited line number precedes that test-mod opening
//!      line — i.e. the cite points at production code, not test code.
//!
//! The test FAILS as long as any REQ-8 cite resolves into a `#[cfg(test)]`
//! block. It PASSES once acto-fixer drops the offending cite (or replaces it
//! with a real non-test caller of `arithmetic::pow`).
//!
//! ## Why nothing here is hardcoded
//!
//! No literal `876` or `graph.rs` appears in the assertion body. The
//! expected-behaviour predicate ("cited line < `#[cfg(test)]` line in that
//! file") is derived entirely from goal.md:73 + goal.md:150 plus what the
//! design-doc row says. The test stays correct against future REQ-8 row
//! edits — drop one cite, add another, the test re-derives whether every
//! cite is production-side from first principles.
//!
//! Tracking: blocker #1203.

use std::fs;
use std::path::{Path, PathBuf};

/// Path to the design doc relative to either the workspace root or the
/// ferrotorch-core crate root (cargo test's cwd varies). Returns the first
/// existing path.
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
    panic!(
        "could not locate .design/ferrotorch-core/grad_fns/arithmetic.md from cwd; tried: {candidates:?}"
    );
}

/// Resolve a `ferrotorch-<crate>/src/...` path against either the workspace
/// root (cwd = workspace) or the ferrotorch-core crate root
/// (cwd = ferrotorch-core/). Returns the first existing path.
fn locate_source(rel: &str) -> PathBuf {
    let candidates = [PathBuf::from("..").join(rel), PathBuf::from(rel)];
    for p in candidates.iter() {
        if p.exists() {
            return p.clone();
        }
    }
    panic!("could not locate source file `{rel}` from cwd; tried: {candidates:?}");
}

/// A parsed consumer cite from a design-doc row.
#[derive(Debug, Clone)]
struct Cite {
    /// e.g. `ferrotorch-core/src/autograd/graph.rs`
    src_path: String,
    /// 1-indexed source line claimed to host the consumer
    line: usize,
}

/// Locate the REQ-8 SHIPPED row in the design doc. Returns the row's full
/// text (one line, including the leading `|`).
fn extract_req8_row(doc: &str) -> String {
    for line in doc.lines() {
        let trimmed = line.trim_start();
        // Match any row whose first cell starts with "REQ-8" (e.g.
        // "| REQ-8 (pow) | SHIPPED | ..."). The leading `|` and any
        // whitespace before "REQ-8" are skipped.
        if let Some(rest) = trimmed.strip_prefix('|') {
            let cell0 = rest.trim_start();
            if cell0.starts_with("REQ-8") {
                return line.to_string();
            }
        }
    }
    panic!("could not find a `| REQ-8` row in the design doc");
}

/// Parse all consumer cites of the form `ferrotorch-<crate>/src/<path>.rs:L`
/// out of a design-doc row. `:L1,L2,...` is expanded into one Cite per line
/// number (mirroring how `grad_penalty.rs:111,118` lists two distinct
/// consumer sites under one path).
fn parse_cites(row: &str) -> Vec<Cite> {
    // Strategy: find every occurrence of "ferrotorch-" in the row that is
    // followed by a /src/.../<file>.rs:<digits>(,digits)* sequence. We scan
    // byte by byte so we can deal with backticks, parentheses, etc.
    let mut out = Vec::new();
    let bytes = row.as_bytes();
    let n = bytes.len();
    let mut i = 0;
    while i < n {
        // Find next "ferrotorch-" anchor.
        let Some(start) = row[i..].find("ferrotorch-") else {
            break;
        };
        let abs_start = i + start;
        // Walk forward until we see `.rs:` or hit something that can't be a
        // path char (`)`, ` `, ` ``, `(`, etc.).
        let mut j = abs_start;
        while j < n {
            let c = bytes[j] as char;
            // Allowed in a `ferrotorch-foo/src/bar/baz.rs` path: alnum, `-`,
            // `_`, `/`, `.`.
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '/' || c == '.' {
                j += 1;
            } else {
                break;
            }
        }
        let path_str = &row[abs_start..j];
        // Must end in `.rs` AND be followed by `:`.
        if path_str.ends_with(".rs") && j < n && bytes[j] == b':' {
            // Collect digits + commas after the colon.
            let mut k = j + 1;
            while k < n {
                let c = bytes[k] as char;
                if c.is_ascii_digit() || c == ',' {
                    k += 1;
                } else {
                    break;
                }
            }
            let line_spec = &row[j + 1..k];
            for tok in line_spec.split(',') {
                if let Ok(ln) = tok.parse::<usize>() {
                    out.push(Cite {
                        src_path: path_str.to_string(),
                        line: ln,
                    });
                }
            }
            i = k;
        } else {
            // Not a real consumer cite (e.g. just `ferrotorch-core/src` in
            // prose); skip past this anchor.
            i = abs_start + "ferrotorch-".len();
        }
    }
    out
}

/// Scan a source file and return the 1-indexed line on which `mod tests`
/// opens immediately after a `#[cfg(test)]` annotation, if any. The pattern
/// matched is:
///
/// ```text
///     #[cfg(test)]
///     mod tests {
/// ```
///
/// (whitespace-tolerant; allows the `pub` keyword before `mod`). Returns
/// `None` if no such opening exists, which means every line is production.
fn find_cfg_test_mod_open(src_path: &Path) -> Option<usize> {
    let text = fs::read_to_string(src_path)
        .unwrap_or_else(|e| panic!("could not read source file {}: {}", src_path.display(), e));
    let mut prev_was_cfg = false;
    for (idx0, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if prev_was_cfg {
            // Allow `mod tests` or `pub mod tests`.
            let after_pub = trimmed.strip_prefix("pub ").unwrap_or(trimmed);
            if after_pub.starts_with("mod tests") {
                return Some(idx0 + 1); // 1-indexed
            }
        }
        prev_was_cfg = trimmed == "#[cfg(test)]";
    }
    None
}

/// Predicate codified from goal.md:73 + goal.md:150 (R-HONEST-2): a
/// SHIPPED-REQ consumer cite must live in production code, i.e. on a line
/// that strictly precedes the `#[cfg(test)] mod tests` opening of its file.
/// Files with no `#[cfg(test)]` block are entirely production — any cite
/// into such a file passes the predicate trivially.
fn cite_is_production(cite: &Cite) -> Result<(), String> {
    let src = locate_source(&cite.src_path);
    match find_cfg_test_mod_open(&src) {
        None => Ok(()),
        Some(test_mod_open) => {
            if cite.line < test_mod_open {
                Ok(())
            } else {
                Err(format!(
                    "{}:{} is inside `#[cfg(test)]` block opening at {}:{}",
                    cite.src_path, cite.line, cite.src_path, test_mod_open
                ))
            }
        }
    }
}

#[test]
fn divergence_pow_req8_consumer_cites_are_production_only() {
    let doc_path = locate_design_doc();
    let doc = fs::read_to_string(&doc_path)
        .unwrap_or_else(|e| panic!("could not read {}: {}", doc_path.display(), e));

    let row = extract_req8_row(&doc);
    let cites = parse_cites(&row);

    assert!(
        !cites.is_empty(),
        "REQ-8 row contained zero parseable `ferrotorch-*/src/<path>.rs:<line>` cites; \
         the parser is wrong or the row degenerated. Row text: {row}"
    );

    let mut violations: Vec<String> = Vec::new();
    for cite in &cites {
        if let Err(msg) = cite_is_production(cite) {
            violations.push(msg);
        }
    }

    assert!(
        violations.is_empty(),
        "design-doc REQ-8 row ({}) cites at least one test-block line as a \
         \"non-test production consumer\", violating goal.md:73 + goal.md:150 \
         (R-HONEST-2 \"Test-only callers don't count\"). Offending cite(s):\n  - {}\n\n\
         Fix: drop the offending cite from the REQ-8 row in {} (or replace it \
         with a real non-test caller of `arithmetic::pow`). Cites parsed from row: \
         {:?}",
        doc_path.display(),
        violations.join("\n  - "),
        doc_path.display(),
        cites
            .iter()
            .map(|c| format!("{}:{}", c.src_path, c.line))
            .collect::<Vec<_>>()
    );
}

// -- self-tests for the parser ---------------------------------------------
//
// These guard against the parser regressing into a state where it silently
// finds zero cites (which would make the main test vacuously pass). They use
// synthetic inputs constructed inline — they do NOT mirror the design doc.

#[cfg(test)]
mod parser_self_tests {
    use super::{Cite, parse_cites};

    fn cite_strs(cs: &[Cite]) -> Vec<String> {
        cs.iter()
            .map(|c| format!("{}:{}", c.src_path, c.line))
            .collect()
    }

    #[test]
    fn parses_single_cite() {
        let row = "| REQ-X | foo `ferrotorch-core/src/methods.rs:35` bar |";
        let got = cite_strs(&parse_cites(row));
        assert_eq!(got, vec!["ferrotorch-core/src/methods.rs:35"]);
    }

    #[test]
    fn parses_comma_separated_lines() {
        let row =
            "consumer at `ferrotorch-core/src/autograd/grad_penalty.rs:111,118` (norm-then-fac)";
        let got = cite_strs(&parse_cites(row));
        assert_eq!(
            got,
            vec![
                "ferrotorch-core/src/autograd/grad_penalty.rs:111",
                "ferrotorch-core/src/autograd/grad_penalty.rs:118",
            ]
        );
    }

    #[test]
    fn parses_multiple_distinct_files() {
        let row = "`ferrotorch-core/src/methods.rs:35` and `ferrotorch-core/src/autograd/graph.rs:876` and `ferrotorch-nn/src/functional.rs:979,983`";
        let got = cite_strs(&parse_cites(row));
        assert_eq!(
            got,
            vec![
                "ferrotorch-core/src/methods.rs:35",
                "ferrotorch-core/src/autograd/graph.rs:876",
                "ferrotorch-nn/src/functional.rs:979",
                "ferrotorch-nn/src/functional.rs:983",
            ]
        );
    }

    #[test]
    fn ignores_bare_ferrotorch_prefix_with_no_rs_line() {
        // Prose mention without `.rs:<line>` should NOT be picked up.
        let row = "see ferrotorch-core for context (no cite here)";
        assert!(parse_cites(row).is_empty());
    }
}
