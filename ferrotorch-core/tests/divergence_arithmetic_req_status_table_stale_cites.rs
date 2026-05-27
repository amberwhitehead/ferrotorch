//! Divergence: arithmetic.md REQ status table cites are stale across multiple
//! REQ rows.
//!
//! Per goal.md R-CITE-2, every `file:line` cite in a design doc must point at
//! the symbol it claims. The REQ status table at
//! `.design/ferrotorch-core/grad_fns/arithmetic.md:891-906` cites
//! `ferrotorch-core/src/grad_fns/arithmetic.rs:NNN` (and a few
//! `ferrotorch-core/src/methods.rs:NNN`, `ferrotorch-core/src/inplace.rs:NNN`)
//! for each REQ's `impl: <op> at ...` anchor.
//!
//! As of commit 19fb7b9ff (addcdiv REQ-16 ship), the post-build file shifts
//! left at least the following arithmetic.rs impl-line cites stale:
//!
//!   REQ-2 sub          cites :805  actual :786
//!   REQ-3 mul          cites :991  actual :941
//!   REQ-4 div          cites :1151 actual :1101
//!   REQ-5 neg          cites :1293 actual :1243
//!   REQ-6 abs          cites :1646 actual :3482   (the wildest miss)
//!   REQ-7 sqrt         cites :1525 actual :1475
//!   REQ-8 pow          cites :1423 actual :1373
//!
//! And in methods.rs (cite pattern `<path>` (`Tensor::<op>_t`)):
//!
//!   REQ-1 add_t        cites :15   actual :14
//!   REQ-3 mul_t        cites :23   actual :36
//!   REQ-4 div_t        cites :27   actual :40
//!   REQ-5 neg_t        cites :31   actual :44
//!   REQ-6 abs_t        cites :43   actual :82
//!   REQ-7 sqrt_t       cites :39   actual :52
//!   REQ-8 pow_t        cites :35   actual :48
//!
//! And in inplace.rs:
//!
//!   REQ-1 add_scaled_  cites :213  actual :167
//!   REQ-2 sub_scaled_  cites :265  actual :266
//!
//! The builder for commit 19fb7b9ff flagged only REQ-6 abs and REQ-8 pow in
//! the "SPILLOVER FINDINGS" section of the commit message ("leaving for a
//! follow-up audit dispatch") — but the failure is much wider. Every REQ
//! landed prior to the REQ-9 (rsub) cite-refresh wave (commit d7ff0d0ed) has
//! drifted because subsequent op insertions shifted every following line.
//!
//! This is a META-divergence across the REQ status table — filed as a SINGLE
//! blocker (#1228) per the audit-shape instruction not to fragment per-REQ.
//!
//! Tracking: #1228

use std::fs;
use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if !p.join(".design").exists() {
        p.pop();
    }
    p
}

fn read_arithmetic_md() -> String {
    let root = workspace_root();
    let p = root.join(".design/ferrotorch-core/grad_fns/arithmetic.md");
    fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {}: {}", p.display(), e))
}

fn req_op_names() -> Vec<(u32, Vec<&'static str>)> {
    vec![
        (1, vec!["add", "add_scaled", "add_out", "add_scaled_out"]),
        (2, vec!["sub", "sub_scaled"]),
        (3, vec!["mul"]),
        (4, vec!["div"]),
        (5, vec!["neg"]),
        (6, vec!["abs"]),
        (7, vec!["sqrt"]),
        (8, vec!["pow"]),
        (9, vec!["rsub"]),
        (10, vec!["rsqrt"]),
        (11, vec!["reciprocal"]),
        (12, vec!["floor_divide"]),
        (13, vec!["remainder"]),
        (14, vec!["fmod"]),
        (15, vec!["addcmul"]),
        (16, vec!["addcdiv"]),
    ]
}

/// Pattern A: `` `<op>` at `<path>:<line>` `` (impl cites).
/// Continuation form: `` `<op>` at `:<line>` `` reuses prior path within row.
fn parse_pattern_a(row: &str, expected_ops: &[&str]) -> Vec<(String, String, usize)> {
    let mut out = Vec::new();
    for &op in expected_ops {
        let needle = format!("`{op}` at `");
        let mut pos = 0usize;
        let mut last_path: Option<String> = None;
        while let Some(found) = row[pos..].find(&needle) {
            let start = pos + found + needle.len();
            let rest = &row[start..];
            let end_bt = match rest.find('`') {
                Some(e) => e,
                None => break,
            };
            let cite = &rest[..end_bt];
            if let Some(colon) = cite.rfind(':') {
                let line_str = &cite[colon + 1..];
                if let Ok(line_num) = line_str.parse::<usize>() {
                    let path_part = &cite[..colon];
                    let path = if path_part.is_empty() {
                        match last_path.as_ref() {
                            Some(p) => p.clone(),
                            None => {
                                pos = start + end_bt + 1;
                                continue;
                            }
                        }
                    } else {
                        path_part.to_string()
                    };
                    last_path = Some(path.clone());
                    out.push((op.to_string(), path, line_num));
                }
            }
            pos = start + end_bt + 1;
        }
    }
    out
}

/// Pattern B: `` `<path>:<line>` (`<op>`) `` (non-test-consumer cites, in
/// particular methods.rs / inplace.rs entries that use the path-first order).
/// Returns `(op, path, line)`.
fn parse_pattern_b_path_then_op(row: &str) -> Vec<(String, String, usize)> {
    let mut out = Vec::new();
    // Look for `<path>:<line>` followed by ` (` `<op>` `)`.
    // We sweep through every backtick-delimited segment in the row.
    let bytes = row.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] != b'`' {
            i += 1;
            continue;
        }
        let start = i + 1;
        let mut j = start;
        while j < bytes.len() && bytes[j] != b'`' {
            j += 1;
        }
        if j >= bytes.len() {
            break;
        }
        let inside = &row[start..j];
        // Does `inside` look like a `<path>:<line>`?
        if let Some(colon) = inside.rfind(':') {
            let (p, l) = (&inside[..colon], &inside[colon + 1..]);
            if (p.ends_with(".rs") || p.contains("/src/")) && l.parse::<usize>().is_ok() {
                // Look ahead for ` (` then a backtick-quoted op name.
                let after = &row[j + 1..];
                if let Some(stripped) = after.strip_prefix(" (`") {
                    if let Some(end_op) = stripped.find('`') {
                        let op = &stripped[..end_op];
                        if !op.is_empty() && !op.contains(' ') {
                            out.push((op.to_string(), p.to_string(), l.parse().unwrap()));
                        }
                    }
                }
            }
        }
        i = j + 1;
    }
    out
}

fn find_req_row(doc: &str, req: u32) -> Option<String> {
    let prefix = format!("| REQ-{req} (");
    for line in doc.lines() {
        if line.starts_with(&prefix) {
            return Some(line.to_string());
        }
    }
    None
}

/// Check whether the given file at `line_no` contains a `pub fn <op>(` or
/// `struct <Op>Backward` definition (or any `pub fn <op>...`).
fn line_declares_op(file_path: &Path, line_no: usize, op: &str) -> Result<bool, String> {
    let src = fs::read_to_string(file_path)
        .map_err(|e| format!("read {}: {}", file_path.display(), e))?;
    let lines: Vec<&str> = src.lines().collect();
    if line_no == 0 || line_no > lines.len() {
        return Ok(false);
    }
    let line = lines[line_no - 1];
    // Accept `pub fn <op>` (handles `pub fn add<T:Float>`, `pub fn add_scaled(`,
    // method `pub fn add_t(&self,`, in-place `pub fn add_scaled_(&self,` ...).
    let pub_fn = format!("pub fn {op}");
    if line.contains(&pub_fn) {
        return Ok(true);
    }
    // Backward struct: e.g. "struct AddcdivBackward<T: Float>" for op "addcdiv".
    let pascal = to_pascal_case(op);
    let backward = format!("struct {pascal}Backward");
    if line.contains(&backward) {
        return Ok(true);
    }
    Ok(false)
}

fn to_pascal_case(s: &str) -> String {
    let mut out = String::new();
    let mut up = true;
    for c in s.chars() {
        if c == '_' {
            up = true;
        } else if up {
            out.extend(c.to_uppercase());
            up = false;
        } else {
            out.push(c);
        }
    }
    out
}

fn read_actual(abs: &Path, line: usize) -> String {
    fs::read_to_string(abs)
        .ok()
        .and_then(|s| s.lines().nth(line - 1).map(|l| l.to_string()))
        .unwrap_or_default()
}

#[test]
fn divergence_arithmetic_req_status_table_arithmetic_rs_cites_resolve() {
    let doc = read_arithmetic_md();
    let root = workspace_root();
    let mut errors: Vec<String> = Vec::new();

    for (req, ops) in req_op_names() {
        let row = match find_req_row(&doc, req) {
            Some(r) => r,
            None => {
                errors.push(format!("REQ-{req} row not found in status table"));
                continue;
            }
        };
        let cites = parse_pattern_a(&row, &ops);
        let mut saw_arithmetic_cite = false;
        for (op, path, line) in cites {
            if !path.ends_with("grad_fns/arithmetic.rs") {
                continue;
            }
            saw_arithmetic_cite = true;
            let abs = root.join(&path);
            match line_declares_op(&abs, line, &op) {
                Ok(true) => {}
                Ok(false) => {
                    let actual = read_actual(&abs, line);
                    errors.push(format!(
                        "REQ-{req}: arithmetic.md cites `{op}` at `{path}:{line}` but that line is:\n    `{actual}`\n  (expected to contain `pub fn {op}(` or `struct {pascal}Backward`)",
                        pascal = to_pascal_case(&op),
                    ));
                }
                Err(e) => errors.push(format!("REQ-{req}: {e}")),
            }
        }
        if !saw_arithmetic_cite {
            errors.push(format!(
                "REQ-{req}: status-table row has no arithmetic.rs impl cite — could not audit"
            ));
        }
    }

    assert!(
        errors.is_empty(),
        "arithmetic.md REQ status table has stale impl-line cites that no longer resolve to their named symbols (R-CITE-2):\n\n{}\n\nTracking: #1228 — this is a META-divergence across the REQ status table; the builder for commit 19fb7b9ff flagged only REQ-6 abs and REQ-8 pow as spillover but the actual failure is wider.",
        errors.join("\n\n")
    );
}

#[test]
fn divergence_arithmetic_req_status_table_consumer_cites_resolve() {
    let doc = read_arithmetic_md();
    let root = workspace_root();
    let mut errors: Vec<String> = Vec::new();

    // Expected consumer-symbol names by REQ. These are `Tensor::<op>_t`
    // methods.rs entries, plus `add_scaled_` / `sub_scaled_` inplace.rs
    // entries for REQ-1 and REQ-2.
    let expected: Vec<(u32, Vec<&str>)> = vec![
        (1, vec!["add_t", "add_scaled_"]),
        (2, vec!["sub_t", "sub_scaled_"]),
        (3, vec!["mul_t"]),
        (4, vec!["div_t"]),
        (5, vec!["neg_t"]),
        (6, vec!["abs_t"]),
        (7, vec!["sqrt_t"]),
        (8, vec!["pow_t"]),
        (9, vec!["rsub_t"]),
        (10, vec!["rsqrt_t"]),
        (11, vec!["reciprocal_t"]),
        (12, vec!["floor_divide_t"]),
        (13, vec!["remainder_t"]),
        (14, vec!["fmod_t"]),
        (15, vec!["addcmul_t"]),
        (16, vec!["addcdiv_t"]),
    ];

    for (req, expected_ops) in expected {
        let row = match find_req_row(&doc, req) {
            Some(r) => r,
            None => continue,
        };
        // Use the path-then-op pattern (`<path>:<N>` (`Tensor::<op>`)).
        let cites = parse_pattern_b_path_then_op(&row);
        for (op_raw, path, line) in cites {
            // Strip `Tensor::` prefix.
            let op = op_raw.strip_prefix("Tensor::").unwrap_or(&op_raw);
            // Skip ops not expected for this REQ (the row may cite many
            // unrelated symbols like `arithmetic::add` which is fine).
            if !expected_ops.iter().any(|&e| e == op) {
                continue;
            }
            // Only audit methods.rs / inplace.rs / forward_ad.rs / einsum.rs etc.
            if !(path.ends_with("methods.rs") || path.ends_with("inplace.rs")) {
                continue;
            }
            let abs = root.join(&path);
            match line_declares_op(&abs, line, op) {
                Ok(true) => {}
                Ok(false) => {
                    let actual = read_actual(&abs, line);
                    errors.push(format!(
                        "REQ-{req}: arithmetic.md cites `{path}:{line}` (`Tensor::{op}` or `{op}`) but that line is:\n    `{actual}`\n  (expected to contain `pub fn {op}(`)"
                    ));
                }
                Err(e) => errors.push(format!("REQ-{req}: {e}")),
            }
        }
    }

    assert!(
        errors.is_empty(),
        "arithmetic.md REQ status table has stale consumer cites in methods.rs / inplace.rs (R-CITE-2):\n\n{}\n\nTracking: #1228.",
        errors.join("\n\n")
    );
}
