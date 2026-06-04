//! Divergence test for #1192 audit: REQ-2 (sub, sub_scaled) is R-DEFER-1
//! vocabulary-only-shipping — `arithmetic::sub_scaled` has ZERO non-test
//! production consumers.
//!
//! `.design/ferrotorch-core/grad_fns/arithmetic.md` REQ-2 SHIPPED row in
//! commit `1527b3bd1` claims:
//!
//!   "Non-test consumer: ferrotorch-core/src/inplace.rs:265
//!     (`Tensor::sub_scaled_`) calls `Tensor::add_scaled_(other, -alpha)`,
//!     ... also ferrotorch-core/src/methods.rs:18 (`Tensor::sub_t`) calls
//!     `arithmetic::sub` and ferrotorch-core/src/autograd/forward_ad.rs:97
//!     (dual-number forward subtraction primal) calls `arithmetic::sub`."
//!
//! All three cited "consumers" call OTHER functions:
//!
//!   - `inplace.rs:266 sub_scaled_` calls `self.add_scaled_(other, -alpha)`
//!     — that is a call to `Tensor::add_scaled_`, NOT `arithmetic::sub_scaled`.
//!   - `methods.rs:18 sub_t` calls `arithmetic::sub(self, other)`
//!     — that is `arithmetic::sub`, NOT `arithmetic::sub_scaled`.
//!   - `forward_ad.rs:97 dual_sub` calls `arithmetic::sub(&a.primal, &b.primal)`
//!     — that is `arithmetic::sub`, NOT `arithmetic::sub_scaled`.
//!
//! Per goal.md:
//!
//!   - **R-DEFER-1** (goal.md:186): "A commit that adds a public API surface
//!     (new `pub fn` ...) MUST also add a non-test production consumer in
//!     the same commit. Test-only callers don't count. The parity-sweep
//!     runner's dispatch table is a test-side consumer; it does NOT count
//!     as a production consumer."
//!   - **R-HONEST-2** (goal.md:150): "SHIPPED requires both implementation
//!     AND a non-test production consumer cited."
//!   - **R-DEFER-2** (goal.md:188): "SHIPPED or NOT-STARTED. SHIPPED means
//!     end-to-end functional with non-test production consumer + tests +
//!     parity-sweep smoke >=1."
//!
//! REQ-2 is a joint claim for `(sub, sub_scaled)`. The `sub` half is
//! correctly cited. The `sub_scaled` half is vocabulary-only — it is a
//! public symbol that no production code calls. The only non-test code
//! that mentions `sub_scaled` are doc-comments referencing it, plus its
//! own definition at `grad_fns/arithmetic.rs:938`.
//!
//! ## What this test does
//!
//! Asserts a derived property: for every `pub fn` whose name appears in the
//! parenthetical of the REQ-2 row header (i.e. `sub` and `sub_scaled`),
//! the workspace contains at least one non-test call site inside a routed
//! production source file.
//!
//! Programmatic, not tautological (R-CHAR-3):
//!
//!   1. Reads `.design/ferrotorch-core/grad_fns/arithmetic.md` and locates
//!      the row whose first cell begins with `| REQ-2`.
//!   2. Parses the symbol names from the parenthetical
//!      `(sub, sub_scaled)` — names are derived from the doc, not hardcoded.
//!   3. For each name, greps the workspace's `ferrotorch-*/src/` tree for
//!      call sites of the form `arithmetic::<name>(` or `::<name>(`
//!      (matching both `crate::grad_fns::arithmetic::sub_scaled(...)` and
//!      `arithmetic::sub_scaled(...)`).
//!   4. Excludes:
//!      - The symbol's own definition (the line in `arithmetic.rs` that
//!        declares `pub fn <name>` — that's not a consumer).
//!      - Lines inside any `#[cfg(test)]` block.
//!      - Test files (`/tests/*.rs`).
//!      - Doc-comments (`//!` and `///` lines).
//!      - The parity-sweep runner crate (`tools/parity-sweep/runner/`)
//!        per R-DEFER-1's explicit carve-out.
//!   5. Asserts the remaining call-site count is >= 1.
//!
//! The test FAILS for `sub_scaled` (zero remaining call sites). It passes
//! for `sub` (sub_t at methods.rs:19, dual_sub at forward_ad.rs:97-98).
//!
//! Tracking: blocker (filed via crosslink quick).

use std::fs;
use std::path::{Path, PathBuf};

/// Path to the design doc relative to either the workspace root or the
/// ferrotorch-core crate root (cargo test's cwd varies).
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

/// Locate workspace root (the dir containing `ferrotorch-core/` and
/// `.design/`). Test cwd is either workspace root or `ferrotorch-core/`.
fn locate_workspace_root() -> PathBuf {
    let candidates = [PathBuf::from(".."), PathBuf::from(".")];
    for c in candidates {
        if c.join(".design").exists() && c.join("ferrotorch-core").exists() {
            return c;
        }
    }
    panic!("could not locate workspace root from cwd");
}

/// Locate the REQ-2 SHIPPED row in the design doc and return its full text.
fn extract_req2_row(doc: &str) -> String {
    for line in doc.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix('|') {
            let cell0 = rest.trim_start();
            if cell0.starts_with("REQ-2") {
                return line.to_string();
            }
        }
    }
    panic!("could not find a `| REQ-2` row in the design doc");
}

/// Parse the symbol-name list from the `(sub, sub_scaled)` parenthetical
/// in the row header. We look for the first balanced `(...)` in the row's
/// header cell and split on commas. Names are alphanumeric + `_`.
fn parse_req_symbols(row: &str) -> Vec<String> {
    // Header cell is between the first two `|`. e.g.
    //   "| REQ-2 (sub, sub_scaled) | SHIPPED | ..."
    let bytes = row.as_bytes();
    let mut pipes: Vec<usize> = Vec::new();
    for (i, &b) in bytes.iter().enumerate() {
        if b == b'|' {
            pipes.push(i);
        }
        if pipes.len() >= 2 {
            break;
        }
    }
    assert!(
        pipes.len() >= 2,
        "REQ-2 row malformed (need at least two `|`): {row}"
    );
    let header = &row[pipes[0] + 1..pipes[1]];

    // Find first `(...)`.
    let open = header.find('(').unwrap_or_else(|| {
        panic!("REQ-2 row header `{header}` has no `(symbol-list)` parenthetical")
    });
    let close = header[open..]
        .find(')')
        .map(|c| open + c)
        .unwrap_or_else(|| panic!("REQ-2 row header `{header}` has unclosed paren"));
    let inner = &header[open + 1..close];
    inner
        .split(',')
        .map(|tok| {
            tok.chars()
                .filter(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect::<String>()
        })
        .filter(|s| !s.is_empty())
        .collect()
}

/// 1-indexed line on which `mod tests` opens immediately after a
/// `#[cfg(test)]` annotation, if any. Returns `None` if the whole file is
/// production.
fn find_cfg_test_mod_open(text: &str) -> Option<usize> {
    let mut prev_was_cfg = false;
    for (idx0, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if prev_was_cfg {
            let after_pub = trimmed.strip_prefix("pub ").unwrap_or(trimmed);
            if after_pub.starts_with("mod tests") {
                return Some(idx0 + 1);
            }
        }
        prev_was_cfg = trimmed == "#[cfg(test)]";
    }
    None
}

/// Walk a directory and collect all `.rs` files (recursively).
fn walk_rs(root: &Path, out: &mut Vec<PathBuf>) {
    if !root.is_dir() {
        return;
    }
    let entries = match fs::read_dir(root) {
        Ok(e) => e,
        Err(_) => return,
    };
    for ent in entries.flatten() {
        let p = ent.path();
        if p.is_dir() {
            // Skip target/ and any vendor/build dirs.
            let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if name == "target" || name.starts_with('.') || name == "node_modules" {
                continue;
            }
            walk_rs(&p, out);
        } else if p.extension().and_then(|s| s.to_str()) == Some("rs") {
            out.push(p);
        }
    }
}

/// Is a given source-file path considered production for R-DEFER-1?
///
/// Production = ferrotorch-*/src/**/*.rs, NOT inside /tests/, NOT under
/// tools/parity-sweep/ (test infra per R-DEFER-1's explicit carve-out).
fn is_production_path(p: &Path) -> bool {
    let s = p.to_string_lossy().replace('\\', "/");
    // Must be under a ferrotorch-* crate's src/.
    let in_ferrotorch_src = s.contains("/ferrotorch-") && s.contains("/src/");
    if !in_ferrotorch_src {
        return false;
    }
    // Exclude integration-test dirs.
    if s.contains("/tests/") {
        return false;
    }
    // Exclude benches.
    if s.contains("/benches/") {
        return false;
    }
    // Exclude examples.
    if s.contains("/examples/") {
        return false;
    }
    true
}

/// Count non-test call sites of `<symbol>(` in a single source file.
/// A line is a "call site" if it contains `<symbol>(` AND is not:
///   - the line defining `pub fn <symbol>` (or `fn <symbol>`)
///   - inside the `#[cfg(test)]` block
///   - a doc-comment (line starts with `//!` or `///` after trim)
///   - a regular `//` comment line
///
/// Returns matched (file_relative_path_string, 1-indexed-line) pairs.
fn find_call_sites(src_path: &Path, symbol: &str) -> Vec<(String, usize)> {
    let text = match fs::read_to_string(src_path) {
        Ok(t) => t,
        Err(_) => return Vec::new(),
    };
    let test_open = find_cfg_test_mod_open(&text);
    let needle = format!("{symbol}(");
    let mut out = Vec::new();
    for (idx0, line) in text.lines().enumerate() {
        let lineno = idx0 + 1;
        if let Some(open) = test_open
            && lineno >= open
        {
            break; // Past the test boundary; rest is test code.
        }
        if !line.contains(&needle) {
            continue;
        }
        let trimmed = line.trim_start();
        // Skip doc-comments and comments.
        if trimmed.starts_with("//") {
            continue;
        }
        // Skip the symbol's own definition line.
        // Pattern: `pub fn <symbol><` or `pub fn <symbol>(` or `fn <symbol><`.
        let def_pat_pub = format!("pub fn {symbol}");
        let def_pat_priv = format!("fn {symbol}");
        // We want true definition lines — `pub fn sub_scaled<T: Float>` or
        // `fn sub_inner<T: Float>`. Distinguish from `arithmetic::sub_scaled(` calls.
        if trimmed.starts_with(&def_pat_pub) || trimmed.starts_with(&def_pat_priv) {
            continue;
        }
        out.push((src_path.to_string_lossy().to_string(), lineno));
    }
    out
}

#[test]
fn divergence_sub_req2_sub_scaled_has_no_production_consumer() {
    let doc_path = locate_design_doc();
    let doc = fs::read_to_string(&doc_path)
        .unwrap_or_else(|e| panic!("could not read {}: {}", doc_path.display(), e));

    let row = extract_req2_row(&doc);
    let symbols = parse_req_symbols(&row);
    assert!(
        !symbols.is_empty(),
        "REQ-2 row header has no parseable symbol list. Row: {row}"
    );

    let ws = locate_workspace_root();
    // Walk every ferrotorch-* crate under the workspace.
    let mut prod_files: Vec<PathBuf> = Vec::new();
    let entries = fs::read_dir(&ws).expect("read workspace root");
    for ent in entries.flatten() {
        let p = ent.path();
        if p.is_dir() {
            let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if name.starts_with("ferrotorch-") {
                let src = p.join("src");
                if src.is_dir() {
                    walk_rs(&src, &mut prod_files);
                }
            }
        }
    }
    let prod_files: Vec<PathBuf> = prod_files
        .into_iter()
        .filter(|p| is_production_path(p))
        .collect();
    assert!(
        !prod_files.is_empty(),
        "walked workspace at {} and found zero production .rs files; walker is broken",
        ws.display()
    );

    // Per symbol, count call sites.
    let mut findings: Vec<(String, Vec<(String, usize)>)> = Vec::new();
    for sym in &symbols {
        let mut sites: Vec<(String, usize)> = Vec::new();
        for f in &prod_files {
            sites.extend(find_call_sites(f, sym));
        }
        findings.push((sym.clone(), sites));
    }

    // The R-DEFER-1 predicate: every symbol named in the REQ row must have
    // >= 1 non-test production caller.
    let mut violations: Vec<String> = Vec::new();
    for (sym, sites) in &findings {
        if sites.is_empty() {
            violations.push(format!(
                "`arithmetic::{sym}` has ZERO non-test production callers anywhere \
                 in ferrotorch-*/src/. R-DEFER-1: \"A commit that adds a public API \
                 surface MUST also add a non-test production consumer in the same \
                 commit. Test-only callers don't count. The parity-sweep runner's \
                 dispatch table is a test-side consumer; it does NOT count as a \
                 production consumer.\""
            ));
        }
    }

    assert!(
        violations.is_empty(),
        "REQ-2 ({}) is R-DEFER-1 vocabulary-only-shipping. The design-doc row \
         claims SHIPPED for all symbols in its header, but at least one symbol \
         has no non-test production consumer:\n  - {}\n\n\
         Findings per symbol (file:line of non-test call sites, excluding the \
         symbol's own `pub fn` definition):\n{}\n\n\
         Doc-row text: {}",
        doc_path.display(),
        violations.join("\n  - "),
        findings
            .iter()
            .map(|(s, sites)| format!(
                "  {} -> {} site(s){}",
                s,
                sites.len(),
                if sites.is_empty() {
                    String::new()
                } else {
                    format!(
                        "\n      {}",
                        sites
                            .iter()
                            .map(|(f, l)| format!("{f}:{l}"))
                            .collect::<Vec<_>>()
                            .join("\n      ")
                    )
                }
            ))
            .collect::<Vec<_>>()
            .join("\n"),
        row,
    );
}

// -- self-tests for the parsers --------------------------------------------

#[cfg(test)]
mod parser_self_tests {
    use super::{find_cfg_test_mod_open, parse_req_symbols};

    #[test]
    fn parses_two_symbols() {
        let row = "| REQ-2 (sub, sub_scaled) | SHIPPED | impl: ... |";
        let got = parse_req_symbols(row);
        assert_eq!(got, vec!["sub".to_string(), "sub_scaled".to_string()]);
    }

    #[test]
    fn parses_single_symbol() {
        let row = "| REQ-5 (neg) | SHIPPED | impl ... |";
        let got = parse_req_symbols(row);
        assert_eq!(got, vec!["neg".to_string()]);
    }

    #[test]
    fn parses_four_symbols_with_underscores() {
        let row = "| REQ-1 (add, add_scaled, add_out, add_scaled_out) | SHIPPED | impl ... |";
        let got = parse_req_symbols(row);
        assert_eq!(
            got,
            vec![
                "add".to_string(),
                "add_scaled".to_string(),
                "add_out".to_string(),
                "add_scaled_out".to_string(),
            ]
        );
    }

    #[test]
    fn cfg_test_finder_basic() {
        // Synthetic source: a production fn (body has one statement so the
        // anti-pattern scanner does NOT flag this as a stub), then a
        // `#[cfg(test)]` annotation, then `mod tests` opening at line 4.
        let src = "pub fn foo() { 1 + 1; }\n\n#[cfg(test)]\nmod tests {\n    use super::*;\n}\n";
        assert_eq!(find_cfg_test_mod_open(src), Some(4));
    }

    #[test]
    fn cfg_test_finder_no_block() {
        // Two production functions, neither followed by `#[cfg(test)]`.
        // Bodies contain a statement so the anti-pattern scanner does not
        // see them as stubs.
        let src = "pub fn foo() { 1 + 1; }\npub fn bar() { 2 + 2; }\n";
        assert_eq!(find_cfg_test_mod_open(src), None);
    }

    #[test]
    fn cfg_test_finder_with_pub() {
        // `pub mod tests` after `#[cfg(test)]` should also match.
        let src = "fn foo() { let _ = 1; }\n#[cfg(test)]\npub mod tests {\n    fn t() { let _ = 1; }\n}\n";
        assert_eq!(find_cfg_test_mod_open(src), Some(3));
    }
}
