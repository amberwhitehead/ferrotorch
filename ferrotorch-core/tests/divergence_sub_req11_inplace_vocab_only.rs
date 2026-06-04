//! Divergence test for #1192 audit: REQ-11 (sub_scaled_) in
//! `.design/ferrotorch-core/inplace.md` is R-DEFER-1 vocabulary-only-
//! shipping — `Tensor::sub_scaled_` has ZERO non-test production consumers.
//!
//! The REQ-11 row in commit `1527b3bd1` claims:
//!
//!   "Non-test production consumer: ferrotorch-core/src/grad_fns/arithmetic.rs:923-936
//!    pub fn sub_scaled IS the out-of-place sibling and itself delegates
//!    to add_scaled(a, b, -alpha)"
//!
//! That citation is nonsense: `arithmetic::sub_scaled` does NOT call
//! `Tensor::sub_scaled_`. It calls `add_scaled`. Two functions that share
//! a `sub_*` prefix but never invoke one another are not in a
//! producer/consumer relationship.
//!
//! Per goal.md:
//!
//!   - **R-DEFER-1** (goal.md:186): "A commit that adds a public API surface
//!     (new pub fn ...) MUST also add a non-test production consumer in
//!     the same commit. Test-only callers don't count."
//!   - **R-HONEST-2** (goal.md:150): "SHIPPED requires both implementation
//!     AND a non-test production consumer cited."
//!
//! ## What this test does
//!
//! For every method-name parsed from the REQ-11 row's parenthetical
//! `(sub_scaled_)`, assert there is at least one non-test, non-self-recursive
//! production call site of `.<method>(` (or `Tensor::<method>(`) in
//! `ferrotorch-*/src/`. The method's own `pub fn <method>` definition is
//! excluded, doc-comments are excluded, and lines inside `#[cfg(test)]`
//! are excluded. The parity-sweep runner crate is excluded per R-DEFER-1's
//! explicit carve-out.
//!
//! Symbol names are derived from the doc row, not hardcoded (R-CHAR-3 —
//! no tautological tests).
//!
//! Tracking: blocker (filed via crosslink quick).

use std::fs;
use std::path::{Path, PathBuf};

fn locate_design_doc() -> PathBuf {
    let candidates = [
        "../.design/ferrotorch-core/inplace.md",
        ".design/ferrotorch-core/inplace.md",
    ];
    for c in candidates {
        let p = PathBuf::from(c);
        if p.exists() {
            return p;
        }
    }
    panic!("could not locate .design/ferrotorch-core/inplace.md from cwd; tried: {candidates:?}");
}

fn locate_workspace_root() -> PathBuf {
    let candidates = [PathBuf::from(".."), PathBuf::from(".")];
    for c in candidates {
        if c.join(".design").exists() && c.join("ferrotorch-core").exists() {
            return c;
        }
    }
    panic!("could not locate workspace root from cwd");
}

/// Locate the REQ-11 SHIPPED row.
fn extract_req11_row(doc: &str) -> String {
    for line in doc.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix('|') {
            let cell0 = rest.trim_start();
            if cell0.starts_with("REQ-11") {
                return line.to_string();
            }
        }
    }
    panic!("could not find a `| REQ-11` row in the design doc");
}

fn parse_req_symbols(row: &str) -> Vec<String> {
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
        "REQ-11 row malformed (need at least two `|`): {row}"
    );
    let header = &row[pipes[0] + 1..pipes[1]];

    let open = header.find('(').unwrap_or_else(|| {
        panic!("REQ-11 row header `{header}` has no `(symbol-list)` parenthetical")
    });
    let close = header[open..]
        .find(')')
        .map(|c| open + c)
        .unwrap_or_else(|| panic!("REQ-11 row header `{header}` has unclosed paren"));
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

fn is_production_path(p: &Path) -> bool {
    let s = p.to_string_lossy().replace('\\', "/");
    let in_ferrotorch_src = s.contains("/ferrotorch-") && s.contains("/src/");
    if !in_ferrotorch_src {
        return false;
    }
    if s.contains("/tests/") || s.contains("/benches/") || s.contains("/examples/") {
        return false;
    }
    true
}

/// Find non-test, non-definition call sites of a `Tensor::<method>` (or
/// method-style `.method(`) within a single source file.
///
/// For `sub_scaled_` we look for `.sub_scaled_(` patterns OR
/// `Tensor::sub_scaled_(` patterns. The method's own `pub fn sub_scaled_`
/// definition is excluded.
fn find_method_call_sites(src_path: &Path, method: &str) -> Vec<(String, usize)> {
    let text = match fs::read_to_string(src_path) {
        Ok(t) => t,
        Err(_) => return Vec::new(),
    };
    let test_open = find_cfg_test_mod_open(&text);
    let dot_needle = format!(".{method}(");
    let tensor_needle = format!("Tensor::{method}(");
    let mut out = Vec::new();
    for (idx0, line) in text.lines().enumerate() {
        let lineno = idx0 + 1;
        if let Some(open) = test_open
            && lineno >= open
        {
            break;
        }
        if !line.contains(&dot_needle) && !line.contains(&tensor_needle) {
            continue;
        }
        let trimmed = line.trim_start();
        if trimmed.starts_with("//") {
            continue;
        }
        // Skip the method's own definition line.
        let def_pat_pub = format!("pub fn {method}");
        let def_pat_priv = format!("fn {method}");
        if trimmed.starts_with(&def_pat_pub) || trimmed.starts_with(&def_pat_priv) {
            continue;
        }
        out.push((src_path.to_string_lossy().to_string(), lineno));
    }
    out
}

#[test]
fn divergence_inplace_req11_sub_scaled_inplace_has_no_production_consumer() {
    let doc_path = locate_design_doc();
    let doc = fs::read_to_string(&doc_path)
        .unwrap_or_else(|e| panic!("could not read {}: {}", doc_path.display(), e));

    let row = extract_req11_row(&doc);
    let symbols = parse_req_symbols(&row);
    assert!(
        !symbols.is_empty(),
        "REQ-11 row header has no parseable symbol list. Row: {row}"
    );

    let ws = locate_workspace_root();
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
        "walked workspace at {} and found zero production .rs files",
        ws.display()
    );

    let mut findings: Vec<(String, Vec<(String, usize)>)> = Vec::new();
    for sym in &symbols {
        let mut sites: Vec<(String, usize)> = Vec::new();
        for f in &prod_files {
            sites.extend(find_method_call_sites(f, sym));
        }
        findings.push((sym.clone(), sites));
    }

    let mut violations: Vec<String> = Vec::new();
    for (sym, sites) in &findings {
        if sites.is_empty() {
            violations.push(format!(
                "`Tensor::{sym}` has ZERO non-test production callers anywhere \
                 in ferrotorch-*/src/. R-DEFER-1 vocabulary-only-shipping. \
                 The REQ-11 row's cited \"consumer\" (`arithmetic::sub_scaled`) \
                 does NOT call this method — it calls `add_scaled`."
            ));
        }
    }

    assert!(
        violations.is_empty(),
        "REQ-11 ({}) is R-DEFER-1 vocabulary-only-shipping:\n  - {}\n\n\
         Findings per symbol:\n{}\n\nDoc-row text: {}",
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
