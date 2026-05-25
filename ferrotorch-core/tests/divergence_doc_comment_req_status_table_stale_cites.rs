//! Divergence: the `## REQ status` doc-comment tables added by commit
//! `a00574f36` (closes #1236) to `ferrotorch-core/src/grad_fns/arithmetic.rs`
//! and `ferrotorch-core/src/grad_fns/cumulative.rs` cite STALE impl line
//! numbers for the symbols they name.
//!
//! Per goal.md R-CITE-2, every `file:line` cite must resolve to the named
//! symbol. The previously-filed META-divergence #1228 (test
//! `divergence_arithmetic_req_status_table_stale_cites`) caught the same
//! drift in `.design/ferrotorch-core/grad_fns/arithmetic.md`. Commit
//! `a00574f36` propagated those stale cites verbatim into the
//! `//!` doc-comment of `arithmetic.rs` AND additionally introduced
//! parallel-but-newly-stale cites into the `//!` doc-comment of
//! `cumulative.rs` for `cummax`/`cummin`/`logcumsumexp` whose actual
//! file positions have shifted dramatically since `cumulative.md` was
//! authored (e.g. `logcumsumexp` claimed at `:531`, actual at `:712`).
//!
//! This test parses the `//!` doc-comment REQ status tables out of the two
//! `.rs` files and verifies each `pub fn <op>` / `struct <Op>Backward`
//! impl cite resolves to a line that DECLARES that symbol. The
//! `pytorch/.../*.cpp:LLL` and `derivatives.yaml:LLL` upstream cites are
//! NOT audited here (they live under `/home/doll/pytorch` and are not in
//! this workspace).
//!
//! Expected behavior (per upstream goal.md Step 6 template): every `file:L`
//! in the synopsis row points at the named symbol's declaration line in
//! the post-insertion file (NOT the pre-insertion file).
//!
//! Actual behavior: the doc-comment author wrote cites against the
//! pre-insertion line numbers (and/or copied stale `.design/` cites
//! verbatim). After the 25-line insertion into arithmetic.rs and the
//! parallel 16-line insertion into cumulative.rs, every `arithmetic.rs:L`
//! cite is shifted by ~25 from the actual symbol; every cumulative.rs
//! impl cite for the post-#1231 surface (cummax/cummin/logcumsumexp) is
//! shifted by 57-181 lines.
//!
//! Tracking: file a new blocker (cross-link issue) once orchestrator
//! dispatches a refresher.

use std::fs;
use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if !p.join(".design").exists() {
        p.pop();
    }
    p
}

fn read_file(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {}", path.display(), e))
}

/// Extract the `//! | REQ-N ...` rows from the top-of-file `//!` doc-comment.
/// Returns only lines starting with `//! | REQ-`.
fn extract_doc_comment_req_rows(src: &str) -> Vec<String> {
    src.lines()
        .take_while(|l| l.starts_with("//!") || l.is_empty() || l.starts_with("//"))
        .filter(|l| l.starts_with("//! | REQ-"))
        .map(|l| l.trim_start_matches("//! ").to_string())
        .collect()
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

/// Look in `file` at `line_no` (1-based) for a line that contains either
/// `pub fn <op>(` / `pub fn <op><` (generic) / `pub struct <Op>Backward`
/// / `impl<T: Float> GradFn<T> for <Op>Backward`. Permissive on whether
/// it's `pub fn` or `fn` for the impl line (matches the existing #1228
/// test which only checks `pub fn` / `struct`).
fn line_declares_op_or_backward(src: &str, line_no: usize, op: &str) -> bool {
    let lines: Vec<&str> = src.lines().collect();
    if line_no == 0 || line_no > lines.len() {
        return false;
    }
    let line = lines[line_no - 1];
    let pub_fn = format!("pub fn {op}");
    if line.contains(&pub_fn) {
        return true;
    }
    let pascal = to_pascal_case(op);
    let backward_struct = format!("struct {pascal}Backward");
    if line.contains(&backward_struct) {
        return true;
    }
    let backward_impl = format!("GradFn<T> for {pascal}Backward");
    if line.contains(&backward_impl) {
        return true;
    }
    false
}

fn read_actual_line(src: &str, line_no: usize) -> String {
    src.lines()
        .nth(line_no.saturating_sub(1))
        .unwrap_or("")
        .to_string()
}

/// Parse all `<symbol>` at `<path>:<line>` / `<symbol>` at `:<line>` cites
/// from a row. Returns `(symbol, path_or_empty, line)`. When `path` is
/// empty the cite is "continuation" (same row reuses previous path).
fn parse_at_cites(row: &str, op_universe: &[&str]) -> Vec<(String, String, usize)> {
    let mut out = Vec::new();
    for &op in op_universe {
        let needle = format!("`{op}` at `");
        let mut pos = 0usize;
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
                    out.push((op.to_string(), path_part.to_string(), line_num));
                }
            }
            pos = start + end_bt + 1;
        }
    }
    out
}

#[test]
fn divergence_arithmetic_rs_doc_comment_req_table_impl_cites_resolve() {
    let root = workspace_root();
    let path = root.join("ferrotorch-core/src/grad_fns/arithmetic.rs");
    let src = read_file(&path);

    let rows = extract_doc_comment_req_rows(&src);
    assert!(
        !rows.is_empty(),
        "no `//! | REQ-N ...` rows found in arithmetic.rs doc-comment — was the \
         status table removed? Commit a00574f36 added 16 rows."
    );

    // Op universe for arithmetic.rs cites (impl + backward struct names
    // mentioned across all 16 REQ rows).
    let ops = vec![
        "add",
        "add_scaled",
        "add_out",
        "add_scaled_out",
        "sub",
        "sub_scaled",
        "mul",
        "div",
        "neg",
        "abs",
        "sqrt",
        "pow",
        "rsub",
        "rsqrt",
        "reciprocal",
        "floor_divide",
        "remainder",
        "fmod",
        "addcmul",
        "addcdiv",
    ];

    let mut errors: Vec<String> = Vec::new();

    for row in &rows {
        let cites = parse_at_cites(row, &ops);
        // Track the current path within this row (continuation form `:<line>`
        // reuses the most-recent path within the same row).
        let mut current_path: Option<String> = None;
        for (op, path, line) in cites {
            let path_to_use = if path.is_empty() {
                match current_path.as_ref() {
                    Some(p) => p.clone(),
                    None => continue,
                }
            } else {
                current_path = Some(path.clone());
                path.clone()
            };
            // Only audit cites pointing at arithmetic.rs itself.
            if !path_to_use.ends_with("arithmetic.rs") {
                continue;
            }
            if !line_declares_op_or_backward(&src, line, &op) {
                let actual = read_actual_line(&src, line);
                errors.push(format!(
                    "doc-comment row `{row_first40}...` cites `{op}` at `{path_to_use}:{line}` \
                     but that line is:\n      `{actual}`\n      (expected `pub fn {op}(` or \
                     `struct {pascal}Backward` or `impl ... GradFn<T> for {pascal}Backward`)",
                    row_first40 = &row.chars().take(40).collect::<String>(),
                    pascal = to_pascal_case(&op),
                ));
            }
        }
    }

    assert!(
        errors.is_empty(),
        "arithmetic.rs `//! ## REQ status` doc-comment table has stale impl-line \
         cites that do not resolve to the named `pub fn`/`struct ...Backward` \
         (R-CITE-2; goal.md Step 6 template requires a current `<file>:<L>` cite \
         on every row). Commit a00574f36 inserted the doc-comment table but \
         appears to have copied the stale `.design/ferrotorch-core/grad_fns/arithmetic.md` \
         cites (already tracked under #1228) without re-validating them \
         against the post-insertion file. The 25-line doc-comment block ALSO \
         shifted every following `arithmetic.rs:L` by +25.\n\n{}\n",
        errors.join("\n\n")
    );
}

#[test]
fn divergence_cumulative_rs_doc_comment_req_table_impl_cites_resolve() {
    let root = workspace_root();
    let path = root.join("ferrotorch-core/src/grad_fns/cumulative.rs");
    let src = read_file(&path);

    let rows = extract_doc_comment_req_rows(&src);
    assert!(
        !rows.is_empty(),
        "no `//! | REQ-N ...` rows found in cumulative.rs doc-comment — was the \
         status table removed? Commit a00574f36 added 7 rows."
    );

    let ops = vec![
        "cumsum",
        "cumprod",
        "cummax",
        "cummin",
        "logcumsumexp",
        "reverse_cumsum",
        "cummaxmin_backward_impl",
    ];

    let mut errors: Vec<String> = Vec::new();

    for row in &rows {
        let cites = parse_at_cites(row, &ops);
        let mut current_path: Option<String> = None;
        for (op, path, line) in cites {
            let path_to_use = if path.is_empty() {
                match current_path.as_ref() {
                    Some(p) => p.clone(),
                    None => continue,
                }
            } else {
                current_path = Some(path.clone());
                path.clone()
            };
            // Only audit cites pointing at cumulative.rs itself.
            if !path_to_use.ends_with("cumulative.rs") || path_to_use.ends_with("ops/cumulative.rs")
            {
                continue;
            }
            if !line_declares_op_or_backward(&src, line, &op) {
                let actual = read_actual_line(&src, line);
                errors.push(format!(
                    "doc-comment row `{row_first40}...` cites `{op}` at `{path_to_use}:{line}` \
                     but that line is:\n      `{actual}`\n      (expected `pub fn {op}(` or \
                     `struct {pascal}Backward` or `impl ... GradFn<T> for {pascal}Backward`)",
                    row_first40 = &row.chars().take(40).collect::<String>(),
                    pascal = to_pascal_case(&op),
                ));
            }
        }
    }

    assert!(
        errors.is_empty(),
        "cumulative.rs `//! ## REQ status` doc-comment table has stale impl-line \
         cites that do not resolve to the named `pub fn`/`struct ...Backward` \
         (R-CITE-2). The cites appear to be copied from \
         `.design/ferrotorch-core/grad_fns/cumulative.md` which itself predates \
         the recent #1231 / #1232 / #1233 cumulative-op landings. For example, \
         `logcumsumexp` is cited at `cumulative.rs:531` but the actual `pub fn \
         logcumsumexp` is at line ~712.\n\n{}\n",
        errors.join("\n\n")
    );
}

/// Parse the `normalize_axis(...)` call-site cite list out of the REQ-6 row
/// of the cumulative.rs `//!` doc-comment.
///
/// The row contains a fragment of the form:
///
/// ```text
/// `normalize_axis(dim as isize, ndim)` calls at `cumulative.rs:108, :358, :528, :560, :721` ...
/// ```
///
/// We locate the literal `` `cumulative.rs: `` prefix (it begins the
/// backtick-quoted span), then read everything up to the closing backtick.
/// The body is a comma-separated list whose first token is `LINE` and
/// whose subsequent tokens are `:LINE` (continuation form, reusing the
/// path). The parser returns the numeric line list as cited.
///
/// Returning a `Vec<usize>` (rather than the hard-coded array the previous
/// version of this test used) makes the assertion compare doc-comment text
/// to source-code reality, so the test FAILS while the cites are stale
/// and PASSES once the doc-comment is refreshed. The previous formulation
/// (`let cited: [usize; 5] = [73, 203, 231, 241, 323];`) was tautologically
/// unfixable: no production-code edit could ever satisfy it because the
/// "expected" side was a frozen literal.
fn parse_normalize_axis_cited_lines(src: &str) -> Vec<usize> {
    let rows = extract_doc_comment_req_rows(src);
    // Find the REQ-6 row.
    let row = rows
        .iter()
        .find(|r| r.starts_with("| REQ-6 "))
        .expect("REQ-6 row missing from cumulative.rs //! doc-comment");

    // Pattern: ...calls at `cumulative.rs:NNN, :NNN, :NNN, ...`...
    let prefix = "`cumulative.rs:";
    let start = row
        .find(prefix)
        .unwrap_or_else(|| panic!("REQ-6 row missing `cumulative.rs:` cite prefix: {row}"))
        + prefix.len();
    let rest = &row[start..];
    let end = rest
        .find('`')
        .unwrap_or_else(|| panic!("REQ-6 row missing closing backtick after cite prefix: {row}"));
    let body = &rest[..end];

    // Body is "NNN, :NNN, :NNN, :NNN, :NNN" — split on comma, trim, strip
    // leading colon, parse.
    body.split(',')
        .map(|tok| {
            let t = tok.trim();
            let t = t.strip_prefix(':').unwrap_or(t);
            t.parse::<usize>()
                .unwrap_or_else(|_| panic!("non-numeric token in REQ-6 cite list: {t:?}"))
        })
        .collect()
}

/// Verify the `normalize_axis(...)` call-site cites in REQ-6 of the
/// cumulative.rs doc-comment match the actual call lines.
///
/// PARSER-BASED (not literal-hardcoded): both `cited` and `actual` are
/// derived from the file content at test time, so this test FAILS while
/// the doc-comment carries stale line numbers and PASSES once the
/// doc-comment is refreshed by the fixer.
#[test]
fn divergence_cumulative_rs_doc_comment_normalize_axis_call_lines() {
    let root = workspace_root();
    let path = root.join("ferrotorch-core/src/grad_fns/cumulative.rs");
    let src = read_file(&path);

    // Cited lines parsed out of the REQ-6 row of the //! doc-comment.
    let cited: Vec<usize> = parse_normalize_axis_cited_lines(&src);

    // Actual lines where `normalize_axis(` appears as a function-CALL
    // (not as an import line `use crate::shape::normalize_axis;` and not
    // as a citation inside the //! doc-comment itself).
    let actual: Vec<usize> = src
        .lines()
        .enumerate()
        .filter_map(|(i, l)| {
            if l.contains("normalize_axis(")
                && !l.trim_start().starts_with("use ")
                && !l.starts_with("//!")
            {
                Some(i + 1)
            } else {
                None
            }
        })
        .collect();

    assert_eq!(
        actual, cited,
        "cumulative.rs doc-comment REQ-6 row cites `normalize_axis(...)` calls at \
         lines {cited:?} but the actual call lines are {actual:?}. The doc-comment \
         cites are stale (R-CITE-2)."
    );
}
