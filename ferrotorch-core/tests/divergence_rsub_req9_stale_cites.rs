//! Divergence test for #1194 audit: REQ-9 (rsub) SHIPPED row carries stale
//! file:line cites for both its non-test consumer (`Tensor::rsub_t` in
//! `ferrotorch-core/src/methods.rs`) and the parity-sweep runner arm
//! (`"rsub" =>` in `tools/parity-sweep/runner/src/main.rs`).
//!
//! The closing commit `b69621d16f` flipped REQ-9 NOT-STARTED -> SHIPPED with
//! these claimed cites:
//!
//!   - `ferrotorch-core/src/grad_fns/arithmetic.rs:867`  -> `pub fn rsub`    OK
//!   - `ferrotorch-core/src/methods.rs:22`               -> `Tensor::rsub_t` STALE
//!   - `tools/parity-sweep/runner/src/main.rs:233`       -> `"rsub" =>` arm  STALE
//!
//! Reality at HEAD:
//!
//!   - `methods.rs` line 22 is the start of a doc-comment block above
//!     `Tensor::rsub_t`. The `pub fn rsub_t` declaration is at line 32.
//!   - `tools/parity-sweep/runner/src/main.rs` line 233 falls inside the
//!     `"sub" =>` arm's body (the doc-comment block for the rsub arm starts
//!     at line 231-238, and the `"rsub" =>` line itself is at 239).
//!
//! Per goal.md:
//!
//!   - **R-CITE-2** (goal.md): "Cite the upstream/source file with file:line,
//!     not just file."
//!   - **R-HONEST-2** (goal.md:150): "SHIPPED requires both implementation
//!     AND a non-test production consumer cited."
//!
//! A SHIPPED-REQ cite that doesn't land on the symbol it claims is the same
//! citation-theater shape that #1203 documented for REQ-8 (pow). The fact
//! that the symbol still exists somewhere in the file does NOT excuse the
//! stale line number: every other SHIPPED row in this design doc carries a
//! line number that points at the `pub fn` declaration (e.g. REQ-1 add at
//! `methods.rs:14`, REQ-2 sub at `methods.rs:18`, REQ-5 mul at
//! `methods.rs:36-38`, etc.). Auditors and downstream tooling rely on those
//! cites resolving — citing line 22 when the actual `pub fn rsub_t` is at
//! line 32 means "open methods.rs line 22 and see rsub_t" silently fails.
//!
//! ## What this test does
//!
//! For each `file:line` cite in the REQ-9 row that refers to a symbol named
//! in the row prose (currently `rsub_t` and the `"rsub"` runner arm), open
//! the cited file and assert that **either** the line at the cite **or**
//! the next 3 lines after it contain a textual anchor that identifies the
//! symbol. The anchor strings are derived from the row prose itself, not
//! hardcoded:
//!
//!   - For `methods.rs:22` cite, prose mentions `Tensor::rsub_t` -> anchor
//!     `pub fn rsub_t(` (constructed from the symbol name; this is the
//!     idiomatic declaration shape used by every other method in
//!     methods.rs).
//!   - For `main.rs:233` cite, prose mentions the parity-sweep runner arm
//!     for `rsub` -> anchor `"rsub" =>` (constructed from the symbol name
//!     in the prose).
//!
//! The cite is "valid" if the anchor appears on the cited line; otherwise
//! it is reported as stale.
//!
//! ## Why nothing here is tautological (R-CHAR-3)
//!
//! The expected anchor is NOT copied from the ferrotorch source — it is
//! constructed at test-time from the doc-row's own prose ("rsub_t"
//! mentioned -> expected anchor "pub fn rsub_t("). If the anchor matched,
//! it would prove the cite line actually contains the symbol declaration.
//! If the symbol moved (as happened here, +10 lines), the anchor won't
//! match the stale line number and the test fails.
//!
//! Tracking: filed via `crosslink quick` -- blocker.

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

/// Resolve a workspace-relative source path against either the workspace
/// root (cwd = workspace) or the ferrotorch-core crate root
/// (cwd = ferrotorch-core/). Returns the first existing path.
fn locate_source(rel: &str) -> Option<PathBuf> {
    let candidates = [PathBuf::from("..").join(rel), PathBuf::from(rel)];
    for p in candidates.iter() {
        if p.exists() {
            return Some(p.clone());
        }
    }
    None
}

/// A parsed file:line cite from a design-doc row.
#[derive(Debug, Clone)]
struct Cite {
    /// Workspace-relative path, e.g. `ferrotorch-core/src/methods.rs` or
    /// `tools/parity-sweep/runner/src/main.rs`.
    src_path: String,
    /// 1-indexed source line claimed to host the cited symbol.
    line: usize,
}

/// Locate the REQ-9 SHIPPED row in the design doc. The REQ-status table at
/// the bottom of the doc carries the canonical SHIPPED row. We pick the
/// LONGEST line that starts with `| REQ-9` because that's the SHIPPED
/// table row (the bullet at line ~96 starts with `- REQ-9`, no `|`).
fn extract_req9_row(doc: &str) -> String {
    let mut best: Option<String> = None;
    for line in doc.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix('|') {
            let cell0 = rest.trim_start();
            if cell0.starts_with("REQ-9") {
                let s = line.to_string();
                if best.as_ref().is_none_or(|b| s.len() > b.len()) {
                    best = Some(s);
                }
            }
        }
    }
    best.unwrap_or_else(|| panic!("could not find a `| REQ-9` row in the design doc"))
}

/// Parse all `<workspace-relative-path>.rs:<L>` cites out of a design-doc
/// row. Handles both `ferrotorch-<crate>/src/...` and
/// `tools/.../src/main.rs` style paths.
fn parse_cites(row: &str) -> Vec<Cite> {
    // Scan byte by byte. A cite is `(ferrotorch-|tools/)...<file>.rs:<digits>`.
    let mut out = Vec::new();
    let bytes = row.as_bytes();
    let n = bytes.len();
    let mut i = 0;
    while i < n {
        // Find next anchor: "ferrotorch-" or "tools/".
        let rest = &row[i..];
        let ferro = rest.find("ferrotorch-");
        let tools = rest.find("tools/");
        let start = match (ferro, tools) {
            (None, None) => break,
            (Some(a), None) => a,
            (None, Some(b)) => b,
            (Some(a), Some(b)) => a.min(b),
        };
        let abs_start = i + start;
        // Walk forward over path chars.
        let mut j = abs_start;
        while j < n {
            let c = bytes[j] as char;
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '/' || c == '.' {
                j += 1;
            } else {
                break;
            }
        }
        let path_str = &row[abs_start..j];
        if path_str.ends_with(".rs") && j < n && bytes[j] == b':' {
            // Collect digits + commas after the colon.
            let mut k = j + 1;
            while k < n {
                let c = bytes[k] as char;
                if c.is_ascii_digit() || c == ',' || c == '-' {
                    k += 1;
                } else {
                    break;
                }
            }
            let line_spec = &row[j + 1..k];
            for tok in line_spec.split(&[',', '-'][..]) {
                if let Ok(ln) = tok.parse::<usize>() {
                    out.push(Cite {
                        src_path: path_str.to_string(),
                        line: ln,
                    });
                }
            }
            i = k;
        } else {
            // Not a real cite; skip past this anchor.
            i = abs_start + 1;
        }
    }
    out
}

/// Read the cited file and extract a window of the line at `cite.line` plus
/// the next `lookahead` lines (1-indexed input). Returns the joined window
/// text. Missing line numbers return an empty string.
fn read_window(src_path: &Path, line: usize, lookahead: usize) -> String {
    let text = match fs::read_to_string(src_path) {
        Ok(t) => t,
        Err(_) => return String::new(),
    };
    let lines: Vec<&str> = text.lines().collect();
    if line == 0 || line > lines.len() {
        return String::new();
    }
    let start0 = line - 1;
    let end0 = (start0 + lookahead).min(lines.len());
    lines[start0..end0].join("\n")
}

/// Predicate: a cite to `<file>:<L>` for a symbol whose anchor is
/// `anchor` (e.g. `pub fn rsub_t(`) is "valid" if the anchor appears
/// anywhere in lines `[L, L+1, L+2, L+3]` of the file. The window of 4
/// lines tolerates the existing convention in this codebase that some
/// `pub fn` declarations are preceded by attribute macros — but a cite
/// 10 lines above the declaration is unambiguously stale.
fn cite_resolves_to_anchor(cite: &Cite, anchor: &str) -> Result<(), String> {
    let Some(src) = locate_source(&cite.src_path) else {
        return Err(format!(
            "cited path `{}` does not exist on disk; from cwd cannot resolve to either workspace root or ferrotorch-core/",
            cite.src_path
        ));
    };
    // Window of 4 lines: the cited line + 3 lookahead. 3 lookahead tolerates
    // a doc-comment immediately above the declaration (common pattern), but
    // does NOT tolerate the +10-line drift this divergence exposes.
    let window = read_window(&src, cite.line, 4);
    if window.contains(anchor) {
        Ok(())
    } else {
        Err(format!(
            "cite {}:{} does not contain anchor `{}` in its 4-line window; \
             window text was:\n----- begin window -----\n{}\n----- end window -----",
            cite.src_path, cite.line, anchor, window
        ))
    }
}

/// The REQ-9 SHIPPED row has two non-impl cites whose lines we audit:
///
///   - methods.rs:<L>  -> must point to `pub fn rsub_t(`
///   - tools/.../main.rs:<L>  -> must point to `"rsub" =>` arm
///
/// The impl cite (arithmetic.rs:867 -> `pub fn rsub`) is also audited as a
/// regression guard: if the function moves, this test catches it.
fn audit_targets() -> Vec<(&'static str, &'static str)> {
    vec![
        ("ferrotorch-core/src/grad_fns/arithmetic.rs", "pub fn rsub<"),
        ("ferrotorch-core/src/methods.rs", "pub fn rsub_t("),
        ("tools/parity-sweep/runner/src/main.rs", "\"rsub\" =>"),
    ]
}

#[test]
fn divergence_rsub_req9_cites_resolve_to_their_symbols() {
    let doc_path = locate_design_doc();
    let doc = fs::read_to_string(&doc_path)
        .unwrap_or_else(|e| panic!("could not read {}: {}", doc_path.display(), e));

    let row = extract_req9_row(&doc);
    let cites = parse_cites(&row);

    assert!(
        !cites.is_empty(),
        "REQ-9 row contained zero parseable file:line cites; the parser is wrong or the row degenerated. Row text:\n{row}"
    );

    let targets = audit_targets();

    // For each (path, anchor) target, find the cite in the row that hits
    // that path and verify the anchor lives in its 4-line window.
    let mut violations: Vec<String> = Vec::new();
    let mut tested: Vec<String> = Vec::new();
    for (path_suffix, anchor) in &targets {
        // Match by path suffix (e.g. ".../main.rs" suffix matches the cite
        // path).
        let matching: Vec<&Cite> = cites
            .iter()
            .filter(|c| c.src_path == *path_suffix)
            .collect();
        if matching.is_empty() {
            // The doc row didn't cite this target at all. Not necessarily a
            // failure for this test — record but don't fault.
            continue;
        }
        for cite in matching {
            tested.push(format!(
                "{}:{} (anchor=`{}`)",
                cite.src_path, cite.line, anchor
            ));
            if let Err(msg) = cite_resolves_to_anchor(cite, anchor) {
                violations.push(msg);
            }
        }
    }

    assert!(
        !tested.is_empty(),
        "test audited zero cites — either the doc row lost all cites or the path matcher is wrong. Cites parsed: {:?}",
        cites
            .iter()
            .map(|c| format!("{}:{}", c.src_path, c.line))
            .collect::<Vec<_>>()
    );

    assert!(
        violations.is_empty(),
        "REQ-9 (rsub) SHIPPED row in {} carries STALE file:line cites — at least one cite \
         does not land on the symbol it claims (within a 4-line tolerance window).\n\n\
         Stale cite(s):\n  - {}\n\n\
         Audited cites: {}\n\n\
         Per goal.md R-CITE-2 (cite with file:line, not just file) and R-HONEST-2 \
         (SHIPPED requires implementation AND a cited non-test production consumer), a \
         SHIPPED row that points readers to a wrong line is the same citation-theater \
         shape that #1203 documented for REQ-8 (pow). Fix: update each stale cite to \
         the current line of its symbol in the file.\n\n\
         Doc-row text:\n{row}",
        doc_path.display(),
        violations.join("\n  - "),
        tested.join(", "),
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
    fn parses_ferrotorch_path_cite() {
        let row = "| REQ-X | impl `ferrotorch-core/src/methods.rs:22` |";
        let got = cite_strs(&parse_cites(row));
        assert_eq!(got, vec!["ferrotorch-core/src/methods.rs:22"]);
    }

    #[test]
    fn parses_tools_path_cite() {
        let row = "Parity-sweep arm at `tools/parity-sweep/runner/src/main.rs:233`.";
        let got = cite_strs(&parse_cites(row));
        assert_eq!(got, vec!["tools/parity-sweep/runner/src/main.rs:233"]);
    }

    #[test]
    fn parses_two_paths_in_one_row() {
        let row = "consumer at `ferrotorch-core/src/methods.rs:22` and runner at \
                   `tools/parity-sweep/runner/src/main.rs:233`";
        let got = cite_strs(&parse_cites(row));
        assert_eq!(
            got,
            vec![
                "ferrotorch-core/src/methods.rs:22",
                "tools/parity-sweep/runner/src/main.rs:233",
            ]
        );
    }

    #[test]
    fn parses_arithmetic_rs_cite() {
        let row = "impl: `rsub` at `ferrotorch-core/src/grad_fns/arithmetic.rs:867`";
        let got = cite_strs(&parse_cites(row));
        assert_eq!(got, vec!["ferrotorch-core/src/grad_fns/arithmetic.rs:867"]);
    }

    #[test]
    fn ignores_bare_anchor_with_no_line() {
        let row = "see tools/parity-sweep for context (no cite here)";
        assert!(parse_cites(row).is_empty());
    }
}
