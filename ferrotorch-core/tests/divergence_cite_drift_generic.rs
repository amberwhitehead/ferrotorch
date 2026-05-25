//! Generic cite-drift audit (closes #1268).
//!
//! Parses every backtick-quoted `<filename>.rs:<N>` or
//! `<filename>.rs:<N>-<M>` substring (and the bare-colon continuation form
//! `:<N>` / `:<N>-<M>` that reuses the most-recently-mentioned file WITHIN
//! THE SAME backtick-quoted span) from each design doc in scope, plus the
//! `//!` doc-comment of each covered `.rs` source file, and asserts that
//! every cite resolves to substantive content in the target file at HEAD.
//!
//! When the prose adjacent to the cite names a recognizable symbol (e.g.
//! `pub fn <name>`, `struct <Name>Backward`, `fn <test_name>`,
//! `normalize_axis(`, `reverse_cumsum(`), the test additionally verifies
//! that the cited line OR a line within +/-3 lines actually declares that
//! symbol. This is the structural anti-drift contract: a `file:line` cite
//! that lands on a closing brace or a blank line is treated as stale.
//!
//! Replaces the per-category narrow audit tests
//! `divergence_cite_drift_uncaught_test_fn_and_section_headers.rs`,
//! `divergence_arithmetic_md_prose_bare_colon_cites_stale.rs`,
//! `divergence_cumulative_md_prose_cites_stale.rs`, and
//! `divergence_doc_comment_req_status_table_stale_cites.rs`'s broader cite
//! cross-check, whose per-pattern fixture lists were brittle. The narrow
//! REQ-status-table test (`divergence_arithmetic_req_status_table_stale_cites.rs`)
//! and the runner-side cite-shift test
//! (`divergence_addcmul_req15_runner_cite_shift.rs`) survive — they audit
//! different domains (REQ-row parsing logic and
//! `tools/parity-sweep/runner/src/main.rs`, neither of which the generic
//! audit covers).
//!
//! Per goal.md R-CITE-2 every cite carries a line number; per R-CHAR-3 the
//! expected resolution is computed at test time from the actual file
//! contents, not hard-coded.

#![allow(clippy::missing_panics_doc)]

use std::fs;
use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if !p.join(".design").exists() {
        p.pop();
    }
    p
}

/// A parsed cite: (file path or basename, line_start, line_end_inclusive,
/// optional named symbol the prose claims this cite points at).
#[derive(Debug, Clone)]
struct Cite {
    /// As written in the doc — may include a prefix like `ops/` or
    /// `grad_fns/`. We preserve that to resolve `ops/cumulative.rs`
    /// distinctly from `grad_fns/cumulative.rs`.
    file_as_written: String,
    line_start: usize,
    line_end: usize,
    symbol_hint: Option<String>,
}

/// Scanner state that walks the doc and tracks the most-recent `.rs`
/// filename mentioned (whether inside or outside a backtick span). The
/// bare-colon continuation form `:<N>` and the bare-parens form `(<N>)` /
/// `(<N>-<M>)` resolve against this context. Whenever an unsupported
/// non-`.rs` cite (e.g. `ReduceOps.cpp:506`, `derivatives.yaml:529-531`)
/// is seen, the context is INVALIDATED so subsequent bare cites can't
/// silently inherit the wrong file.
struct CiteContext<'a> {
    /// Most-recent `.rs` file mentioned (path-as-written, e.g.
    /// `cumulative.rs` or `ops/cumulative.rs`). `None` if we've never seen
    /// one or if the most recent cite was a non-`.rs` cite (which
    /// invalidates).
    last_rs_file: Option<String>,
    out: &'a mut Vec<Cite>,
    /// `prefix + current_line` text used for symbol-hint extraction (the
    /// prefix is the tail of the previous line, so a cite on this line can
    /// see a symbol that wrapped from the prior line).
    line_for_hints: &'a str,
    /// Length of the prefix portion of `line_for_hints` — cite positions
    /// (from `scan_line_inner`) are offsets into the CURRENT line, so to
    /// translate into `line_for_hints` we add `prefix_len`.
    prefix_len: usize,
}

impl<'a> CiteContext<'a> {
    fn new_with_prefix(line_for_hints: &'a str, out: &'a mut Vec<Cite>, prefix_len: usize) -> Self {
        CiteContext {
            last_rs_file: None,
            out,
            line_for_hints,
            prefix_len,
        }
    }

    fn emit(&mut self, file_as_written: String, lo: usize, hi: usize, cite_start_in_line: usize) {
        let cite_start_in_hint_line = cite_start_in_line + self.prefix_len;
        let symbol_hint = extract_symbol_hint(self.line_for_hints, cite_start_in_hint_line);
        self.out.push(Cite {
            file_as_written,
            line_start: lo,
            line_end: hi,
            symbol_hint,
        });
    }
}

/// Scan one doc line. Handles:
///   1. backtick-quoted `<name>.rs:<N>(-<M>)?` cites
///   2. bare-colon continuation `:<N>(-<M>)?` inside backticks
///   3. bare-parens `(<N>)` / `(<N>-<M>)` cites OUTSIDE backticks (resolve
///      against the doc-wide most-recent `.rs` filename context)
fn scan_line_inner<'a>(line: &'a str, ctx: &mut CiteContext<'a>) {
    // Two-pass: backticks first (they're unambiguous), then bare-parens
    // cites in the residue between backtick spans (so a `(...)` that
    // happens to wrap a backtick span doesn't confuse the paren parser).
    let bytes = line.as_bytes();
    // Pass 1: collect backtick span ranges + process them in order.
    let mut backtick_ranges: Vec<(usize, usize)> = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'`' {
            let start = i + 1;
            let end = match line[start..].find('`') {
                Some(e) => start + e,
                None => break,
            };
            backtick_ranges.push((start, end));
            // Process this span's cites BEFORE looking for bare-parens.
            let span = &line[start..end];
            scan_span_inner(span, ctx, start);
            i = end + 1;
        } else {
            i += 1;
        }
    }
    // Pass 2: scan for bare-parens cites OUTSIDE any backtick span. The
    // content inside `(...)` must NOT overlap a backtick range — that way a
    // prose `(...)` wrapping a real `.rs` cite (e.g. `(in-file ...
    // `arithmetic.rs:3549`)`) is NOT treated as a bare-parens cite — the
    // real cite already got picked up by the backtick pass.
    let mut j = 0usize;
    while j < bytes.len() {
        if bytes[j] == b'(' && !in_backtick_range(j, &backtick_ranges) {
            let start = j + 1;
            let end = match find_matching_close_paren(line, start) {
                Some(e) => e,
                None => {
                    j += 1;
                    continue;
                }
            };
            // Reject if the paren content overlaps any backtick span.
            let mut overlaps = false;
            for &(bts, bte) in &backtick_ranges {
                if !(bte < start || bts > end) {
                    overlaps = true;
                    break;
                }
            }
            if overlaps {
                j = end + 1;
                continue;
            }
            let span = &line[start..end];
            let trimmed = span.trim();
            let (lo_hi_opt, had_colon) = parse_paren_cite(trimmed);
            if let Some((lo, hi)) = lo_hi_opt {
                if (had_colon || lo >= 100) && ctx.last_rs_file.is_some() {
                    let file = ctx.last_rs_file.clone().unwrap();
                    ctx.emit(file, lo, hi, start);
                }
            }
            j = end + 1;
        } else {
            j += 1;
        }
    }
}

fn in_backtick_range(pos: usize, ranges: &[(usize, usize)]) -> bool {
    ranges.iter().any(|&(s, e)| pos >= s && pos < e)
}

/// Find the position of the matching `)` for an opening `(` (paren depth
/// tracked). Returns None if unmatched.
fn find_matching_close_paren(line: &str, start_after_open: usize) -> Option<usize> {
    let bytes = line.as_bytes();
    let mut depth = 1i32;
    let mut k = start_after_open;
    while k < bytes.len() {
        match bytes[k] {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(k);
                }
            }
            _ => {}
        }
        k += 1;
    }
    None
}

/// Try to parse the contents of a `(...)` as a line-number cite. Returns
/// the (lo, hi) if it matches `NNN`, `NNN-MMM`, `:NNN`, or `:NNN-MMM`.
/// The returned bool indicates whether we stripped a leading `:`.
fn parse_paren_cite(s: &str) -> (Option<(usize, usize)>, bool) {
    let (stripped, had_colon) = match s.strip_prefix(':') {
        Some(rest) => (rest, true),
        None => (s, false),
    };
    // Must be purely digits + optional dash + digits.
    let valid = !stripped.is_empty() && stripped.chars().all(|c| c.is_ascii_digit() || c == '-');
    if !valid {
        return (None, had_colon);
    }
    (parse_line_range(stripped), had_colon)
}

/// Extract cites from a backtick-quoted span. Updates the line-level
/// context for continuation cites.
fn scan_span_inner<'a>(span: &'a str, ctx: &mut CiteContext<'a>, span_offset: usize) {
    // Span-local filename context (for `:N` continuations inside this span).
    let mut span_local_file: Option<String> = ctx.last_rs_file.clone();
    for tok_raw in span.split([',', ' ']) {
        let tok = tok_raw.trim();
        if tok.is_empty() {
            continue;
        }
        // Try a named cite (`<dir/>?<name>.rs:<N>(-<M>)?` or non-`.rs`).
        if let Some((file_or_none, lo, hi)) = parse_any_named_cite(tok) {
            match file_or_none {
                Some(file_as_written) => {
                    span_local_file = Some(file_as_written.clone());
                    ctx.last_rs_file = Some(file_as_written.clone());
                    ctx.emit(file_as_written, lo, hi, span_offset);
                }
                None => {
                    // Non-`.rs` cite — invalidate context so subsequent
                    // bare-colon cites can't inherit the wrong file.
                    span_local_file = None;
                    ctx.last_rs_file = None;
                }
            }
            continue;
        }
        // Bare-colon continuation.
        if let Some((lo, hi)) = parse_bare_colon_cite(tok) {
            if let Some(file_as_written) = span_local_file.clone() {
                ctx.emit(file_as_written, lo, hi, span_offset);
            }
        }
    }
}

/// Parse a `<dir/>?<name>.<ext>:<N>(-<M>)?` token. Returns:
///   - `Some((Some(file), lo, hi))` for a `.rs` cite (we want to audit it)
///   - `Some((None, lo, hi))` for a non-`.rs` cite (we want to record that
///     a cite was here, to invalidate the bare-colon context).
///   - `None` if not a file-line cite at all.
fn parse_any_named_cite(tok: &str) -> Option<(Option<String>, usize, usize)> {
    let colon = tok.find(':')?;
    let file_part = &tok[..colon];
    // Must look like `<basename>.<ext>` where ext is rs / cpp / py / yaml /
    // h / hpp / md / toml — i.e. has a dot and the part after the dot is
    // 1-5 ascii_lowercase. Anything else isn't a file-line cite.
    let dot = file_part.rfind('.')?;
    let ext = &file_part[dot + 1..];
    if ext.is_empty() || ext.len() > 5 || !ext.chars().all(|c| c.is_ascii_lowercase()) {
        return None;
    }
    // Validate basename stem is identifier-shaped. Accept uppercase (PyTorch
    // C++ files use PascalCase like `ReduceOps.cpp`) AND lowercase (Rust
    // crates use snake_case like `arithmetic.rs`). We deliberately allow
    // both so that a `ReduceOps.cpp:506` cite is RECOGNIZED as a cite (then
    // ignored because ext != "rs") rather than silently mis-parsed and
    // letting subsequent bare-colon `:NNN` tokens inherit the wrong file.
    let basename = file_part.rsplit('/').next().unwrap_or(file_part);
    let stem = &basename[..basename.len() - (ext.len() + 1)];
    if stem.is_empty() || !stem.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return None;
    }
    let after = &tok[colon + 1..];
    let (lo, hi) = parse_line_range(after)?;
    let path = if ext == "rs" {
        Some(file_part.to_string())
    } else {
        None
    };
    Some((path, lo, hi))
}

/// Parse a `:<N>(-<M>)?` continuation token.
fn parse_bare_colon_cite(tok: &str) -> Option<(usize, usize)> {
    let after = tok.strip_prefix(':')?;
    parse_line_range(after)
}

/// Parse a numeric line range `<N>(-<M>)?`, stopping at the first
/// non-digit/non-dash character. Returns None if the leading token isn't
/// numeric at all.
fn parse_line_range(s: &str) -> Option<(usize, usize)> {
    let mut chars = s.chars().peekable();
    let mut lo_str = String::new();
    while let Some(&c) = chars.peek() {
        if c.is_ascii_digit() {
            lo_str.push(c);
            chars.next();
        } else {
            break;
        }
    }
    if lo_str.is_empty() {
        return None;
    }
    let lo: usize = lo_str.parse().ok()?;
    if chars.peek() == Some(&'-') {
        chars.next();
        let mut hi_str = String::new();
        while let Some(&c) = chars.peek() {
            if c.is_ascii_digit() {
                hi_str.push(c);
                chars.next();
            } else {
                break;
            }
        }
        if let Ok(hi) = hi_str.parse::<usize>() {
            if hi >= lo {
                return Some((lo, hi));
            }
        }
        return Some((lo, lo));
    }
    Some((lo, lo))
}

/// Words that should NOT be treated as symbol hints when found between
/// prose and a backtick cite ("`X` at `:NNN`" — "at" is a prep, not a
/// symbol). The list is conservative; missing entries just mean a hint
/// gets discarded for being not-a-symbol later in `validate_cite`.
const STOPWORD_HINTS: &[&str] = &[
    "at",
    "and",
    "via",
    "by",
    "in",
    "of",
    "for",
    "the",
    "is",
    "are",
    "to",
    "on",
    "with",
    "from",
    "as",
    "or",
    "an",
    "a",
    "calls",
    "cite",
    "cites",
    "see",
    "per",
    "consumer",
    "consumers",
    "uses",
    "used",
    "into",
    "after",
    "before",
    "between",
    "tests",
    "test",
    "row",
    "rows",
    "verified",
    "implementation",
    "impl",
    "section",
    "block",
    "lines",
    "line",
];

/// Look at the prose immediately preceding `cite_start_in_line` (offset in
/// the original line) for a recognizable symbol hint to validate against.
/// Returns the symbol name (without backticks) if found and non-stopword.
///
/// Heuristic ranked: explicit `pub fn <name>` / `pub struct <Name>` / `fn
/// <name>` / `struct <Name>` directly preceding wins over a bare
/// backtick-quoted identifier. Stopwords like "at", "and", "via" are
/// rejected; only identifiers matching a recognizable shape
/// (`*Backward`, `test_*`, `pub fn <name>`, etc.) are kept.
fn extract_symbol_hint(line: &str, cite_start_in_line: usize) -> Option<String> {
    // Window: enough chars to capture a long symbol-hint like
    // `test_cumsum_no_grad_fn_when_not_requires_grad` (45 chars) plus the
    // intervening ` ` (` chars before the cite. We additionally REQUIRE the
    // chosen hint's closing-backtick to be within ~15 chars of cite-start
    // (see close_to_cite below) so distant unrelated mentions don't bleed
    // through.
    let window_start = cite_start_in_line.saturating_sub(80);
    let context = line.get(window_start..cite_start_in_line)?;

    // Pattern 1: explicit declaration form, e.g. `pub fn add_scaled` at `:733`.
    for marker in &["pub fn ", "fn ", "pub struct ", "struct "] {
        if let Some(pos) = context.rfind(marker) {
            let after = &context[pos + marker.len()..];
            let ident: String = after
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            if !ident.is_empty() && !STOPWORD_HINTS.contains(&ident.as_str()) {
                return Some(ident);
            }
        }
    }

    // Pattern 2: walk the context left-to-right, parsing complete backtick
    // pairs in order. Track the latest acceptable symbol-hint candidate.
    // We pair-from-the-left so an UNCLOSED trailing backtick (e.g. for
    // `` (` `` at the end of context — the opening of the cite's own span)
    // is correctly ignored.
    // Walk the context and classify each backtick as either an OPEN or
    // CLOSE based on the char immediately before it. An open backtick is
    // preceded by start-of-context, whitespace, `(`, `,`, `;`, `=`, `-`, or
    // `/`; a close backtick is preceded by an alphanumeric / `_` / `>` /
    // `)` (i.e. ends an identifier or expression). Pair consecutive
    // opens with the next close.
    #[derive(Debug)]
    enum BtKind {
        Open,
        Close,
    }
    let mut classified: Vec<(usize, BtKind)> = Vec::new();
    for (i, c) in context.char_indices() {
        if c != '`' {
            continue;
        }
        let prev_char = if i == 0 {
            ' '
        } else {
            // Walk back to find the previous char (UTF-8 safe by char_indices iteration).
            let mut p = i - 1;
            while !context.is_char_boundary(p) && p > 0 {
                p -= 1;
            }
            context[p..i].chars().next().unwrap_or(' ')
        };
        let kind = match prev_char {
            ' ' | '(' | ',' | ';' | '=' | '-' | '/' | '\t' | '\n' | '*' | '[' | '{' => BtKind::Open,
            _ => BtKind::Close,
        };
        classified.push((i, kind));
    }
    // Build pairs: each Open followed by the next Close.
    let mut backticks: Vec<usize> = Vec::new();
    let mut pending_open: Option<usize> = None;
    for (pos, kind) in classified {
        match kind {
            BtKind::Open => {
                if pending_open.is_none() {
                    pending_open = Some(pos);
                }
                // else: nested-quote oddity; drop the prior open silently.
            }
            BtKind::Close => {
                if let Some(open) = pending_open.take() {
                    backticks.push(open);
                    backticks.push(pos);
                }
                // else: unmatched close; ignore.
            }
        }
    }
    let mut last_accepted: Option<String> = None;
    // Pair from the LEFT, but track only the LAST pair (the one closest to
    // the cite). We further require that the closing-backtick of the chosen
    // pair is within ~15 chars of the cite-start — beyond that, the
    // "preceding symbol" is most likely an unrelated mention earlier in the
    // sentence (e.g. ``Tensor::sub_t`) calls `arithmetic::sub` and ... ` at
    // `forward_ad.rs:97` (...)`` — `Tensor::sub_t` is irrelevant to the cite).
    let context_len = context.len();
    let mut k = 0usize;
    while k + 1 < backticks.len() {
        let open = backticks[k];
        let close = backticks[k + 1];
        let bt = &context[open + 1..close];
        // For `Module::Path::symbol` forms, take the LAST identifier
        // component — `Tensor::sub_t` should hint at `sub_t`, not `Tensor`.
        let last_segment = bt.rsplit("::").next().unwrap_or(bt);
        let ident: String = last_segment
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        // Only consider this pair as a symbol-hint candidate if its closing
        // backtick is within 15 chars of the cite-start.
        let dist_from_cite = context_len.saturating_sub(close);
        let close_to_cite = dist_from_cite <= 15;
        if !ident.is_empty() && !STOPWORD_HINTS.contains(&ident.as_str()) {
            let starts_upper = ident.chars().next().is_some_and(|c| c.is_ascii_uppercase());
            let is_test_fn = ident.starts_with("test_");
            let is_known_helper = matches!(
                ident.as_str(),
                "normalize_axis"
                    | "reverse_cumsum"
                    | "cummax_forward"
                    | "cummin_forward"
                    | "cumsum_forward"
                    | "cumprod_forward"
                    | "logcumsumexp_forward"
                    | "cummaxmin_backward_impl"
                    | "cumulative_scalar_identity"
                    | "cumextreme_scalar_identity"
            );
            let ends_in_backward = ident.ends_with("Backward");
            // Pure-type names too common to be a useful hint:
            let too_generic_type = matches!(
                ident.as_str(),
                "Tensor"
                    | "String"
                    | "Result"
                    | "Vec"
                    | "Option"
                    | "Float"
                    | "FerrotorchResult"
                    | "FerrotorchError"
                    | "GradFn"
                    | "Self"
                    | "Path"
                    | "PathBuf"
                    | "REQ"
                    | "AC"
            );
            // `*_t` chainable-method shape (e.g. `addcmul_t`, `sub_t`).
            let is_method_t =
                ident.ends_with("_t") && ident.chars().all(|c| c.is_ascii_lowercase() || c == '_');
            if close_to_cite
                && !too_generic_type
                && (is_test_fn
                    || is_known_helper
                    || ends_in_backward
                    || is_method_t
                    || starts_upper)
            {
                last_accepted = Some(ident);
            }
        }
        k += 2;
    }
    last_accepted
}

/// Resolve a cite's `file_as_written` to an absolute path under one of the
/// known source roots. Returns `None` if no candidate exists — those cites
/// are silently skipped (they likely point to files outside the audit's
/// scope, e.g. `tools/parity-sweep/runner/src/main.rs` covered by a
/// different test).
fn resolve_cite_path(root: &Path, file_as_written: &str) -> Option<PathBuf> {
    if file_as_written.contains('/') {
        // Doc wrote a path with one or more directory components. Try the
        // path verbatim from workspace root first (handles
        // `ferrotorch-core/src/grad_fns/cumulative.rs`), then with the
        // `ferrotorch-core/src/` prefix prepended (handles `ops/foo.rs`,
        // `grad_fns/foo.rs`).
        let candidates = [
            file_as_written.to_string(),
            format!("ferrotorch-core/src/{file_as_written}"),
            format!("ferrotorch-core/{file_as_written}"),
        ];
        for c in &candidates {
            let p = root.join(c);
            if p.exists() {
                return Some(p);
            }
        }
        return None;
    }
    // Plain basename — try the known directories in priority order.
    let basename = file_as_written;
    let candidates = [
        format!("ferrotorch-core/src/grad_fns/{basename}"),
        format!("ferrotorch-core/src/ops/{basename}"),
        format!("ferrotorch-core/src/{basename}"),
    ];
    for c in &candidates {
        let p = root.join(c);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// Is `trimmed` a line we consider "non-substantive" — i.e. a cite landing
/// on this line tells the reader nothing about the named code surface?
///
/// We treat as non-substantive:
/// - blank / whitespace-only
/// - pure block-comment punctuation: `}`, `{`, `},`, `});`
/// - inner `//!` doc-comment (those document the file/module, not a symbol)
/// - inline `//` line-comment alone
/// - block-comment middle / closer lines (`*` , `*/`) — only when alone
///
/// We DO treat as substantive:
/// - any line containing actual code (`let x = ...`, `pub fn ...`, etc.)
/// - `///` outer doc-comments (they document the NEXT item, which is the
///   intent of cites pointing at this kind of line)
fn is_substantive(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return false;
    }
    // Punctuation-only braces.
    if matches!(trimmed, "}" | "{" | "}," | "});" | "});," | "})") {
        return false;
    }
    // Pure `//!` (inner) doc-comments — documenting the module, not a sym.
    if trimmed.starts_with("//!") {
        return false;
    }
    // Pure `//` line comments (NOT `///`).
    if trimmed.starts_with("//") && !trimmed.starts_with("///") {
        return false;
    }
    // Block-comment middle/end lines that are ONLY `*` or `*/`.
    if trimmed == "*" || trimmed == "*/" || trimmed.starts_with("*/") {
        return false;
    }
    // Otherwise substantive (`///` doc-comments, code lines, etc.)
    true
}

/// Validate a single cite against the file at HEAD. Returns Some(error_msg)
/// on failure or None on success / skip-because-unresolvable.
fn validate_cite(cite: &Cite, root: &Path, doc_label: &str, doc_line_no: usize) -> Option<String> {
    let target = resolve_cite_path(root, &cite.file_as_written)?;
    let src = match fs::read_to_string(&target) {
        Ok(s) => s,
        Err(_) => return None,
    };
    let src_lines: Vec<&str> = src.lines().collect();
    let total = src_lines.len();

    if cite.line_start == 0 || cite.line_end > total {
        return Some(format!(
            "{doc_label}:{doc_line_no} cites `{file}:{lo}-{hi}` but file has only {total} lines",
            file = cite.file_as_written,
            lo = cite.line_start,
            hi = cite.line_end,
        ));
    }

    // Range vs point cite. Range requires at least one substantive line;
    // point cite requires the cited line itself to be substantive (or one
    // adjacent line within +/-1 — handles cases like cite of `///` doc-
    // comment line immediately preceding a `pub fn`).
    let is_range = cite.line_end > cite.line_start;
    let mut any_substantive = false;
    for line_num in cite.line_start..=cite.line_end {
        if is_substantive(src_lines[line_num - 1]) {
            any_substantive = true;
            break;
        }
    }
    if !any_substantive {
        // Try a +/-1 window for point cites only.
        if !is_range {
            let lo = cite.line_start.saturating_sub(1).max(1);
            let hi = (cite.line_start + 1).min(total);
            for line_num in lo..=hi {
                if is_substantive(src_lines[line_num - 1]) {
                    any_substantive = true;
                    break;
                }
            }
        }
    }
    if !any_substantive {
        let actual = if is_range {
            format!(":{}-{}", cite.line_start, cite.line_end)
        } else {
            format!(":{} `{}`", cite.line_start, src_lines[cite.line_start - 1])
        };
        return Some(format!(
            "{doc_label}:{doc_line_no} cites `{file}{actual}` which has no substantive content at HEAD",
            file = cite.file_as_written,
        ));
    }

    // Symbol-hint validation: only run for `test_*` symbols, where the cite
    // is expected to point AT the fn declaration. For `*Backward` /
    // helper / `*_t` symbols, the cite often points inside the symbol's
    // BODY (e.g. ``CumsumBackward` (`:76`)` means "the reverse_cumsum call
    // inside CumsumBackward::backward at line :76", not "the declaration of
    // CumsumBackward at :76") — these can't be cleanly resolved by
    // window-around-declaration logic, so skip.
    let validate_hint = cite
        .symbol_hint
        .as_deref()
        .is_some_and(|s| s.starts_with("test_"));
    if let (true, Some(symbol)) = (validate_hint, &cite.symbol_hint) {
        let needles = build_symbol_needles(symbol);
        let window_lo = cite.line_start.saturating_sub(3).max(1);
        let window_hi = (cite.line_end + 3).min(total);
        let mut found = false;
        for line_num in window_lo..=window_hi {
            let line = src_lines[line_num - 1];
            if needles.iter().any(|n| line.contains(n.as_str())) {
                found = true;
                break;
            }
        }
        if !found {
            let actual = src_lines[cite.line_start - 1];
            return Some(format!(
                "{doc_label}:{doc_line_no} cites `{file}:{lo}` (with symbol hint `{symbol}`) but neither :{lo} nor any line within +/-3 declares it; actual :{lo} is: `{actual}`",
                file = cite.file_as_written,
                lo = cite.line_start,
            ));
        }
    }

    None
}

/// Build the needles list for a symbol hint: every form that line-contains
/// could see (`pub fn <s>`, `fn <s>`, `pub struct <s>`, `struct <s>`,
/// `<s>(`).
fn build_symbol_needles(symbol: &str) -> Vec<String> {
    let mut out = vec![
        format!("pub fn {symbol}"),
        format!("fn {symbol}"),
        format!("pub struct {symbol}"),
        format!("struct {symbol}"),
        format!("{symbol}("),
        format!("pub fn {symbol}<"),
    ];
    // `Tensor::<op>_t` chainable-method consumers delegate to
    // `crate::grad_fns::arithmetic::<op>` (with the `_t` suffix stripped).
    // Cites of the form ``Tensor::<op>_t` at `forward_ad.rs:<N>`` are
    // expected to land on a `arithmetic::<op>(` CALL — accept the un-`_t`
    // form as a valid resolution.
    if let Some(stem) = symbol.strip_suffix("_t") {
        out.push(format!("{stem}("));
        out.push(format!("::{stem}("));
        // Function-pointer pass form: `::<stem>,` or `::<stem> ` (e.g.
        // `vmap(crate::grad_fns::arithmetic::neg, 0, 0)`).
        out.push(format!("::{stem},"));
        out.push(format!("::{stem} "));
        out.push(format!("::{stem})"));
    }
    out
}

fn audit_doc(doc_label: &str, doc_text: &str, root: &Path) -> Vec<String> {
    let mut failures = Vec::new();
    // Doc-wide context: bare-parens `(NNNN)` and bare-colon `:NNN` on a later
    // line reuse the most recent `.rs` filename seen anywhere above. A
    // non-`.rs` cite invalidates the context so we don't carry the wrong
    // filename across (e.g. `ReduceOps.cpp:622` would NOT make subsequent
    // bare cites think they target the previous .rs file).
    let mut doc_wide_last_rs: Option<String> = None;
    // The most-recent prior-line tail (~80 chars). Used to look up symbol
    // hints when the current line starts with a continuation cite like
    // `  (:465-484)` whose name lives at the end of the previous line.
    let mut prev_line_tail: String = String::new();
    let lines: Vec<&str> = doc_text.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        let doc_line_no = i + 1;
        let mut line_cites = Vec::new();
        {
            // Synthesize a search-context line that PREPENDS the prior line's
            // tail so symbol-hint extraction across line wraps works. The
            // file-scan still uses the real `line` (cite positions are
            // line-local to `line`), but the symbol-hint extractor sees the
            // joined context.
            let prefix_len = prev_line_tail.len();
            let joined = format!("{prev_line_tail}{line}");
            let mut ctx = CiteContext::new_with_prefix(&joined, &mut line_cites, prefix_len);
            ctx.last_rs_file = doc_wide_last_rs.clone();
            scan_line_inner(line, &mut ctx);
            doc_wide_last_rs = ctx.last_rs_file.clone();
        }
        for cite in line_cites {
            if let Some(err) = validate_cite(&cite, root, doc_label, doc_line_no) {
                failures.push(err);
            }
        }
        // Update prev_line_tail to the last 80 chars of this line.
        if line.len() <= 80 {
            prev_line_tail = (*line).to_string();
        } else {
            // SAFE: 80 chars >= 80 bytes is a sufficient buffer; we use char
            // boundaries by walking back from end.
            let mut start = line.len() - 80;
            while !line.is_char_boundary(start) {
                start -= 1;
            }
            prev_line_tail = line[start..].to_string();
        }
    }
    failures
}

/// Audit one design doc file.
fn audit_design_doc(rel_path: &str) -> Vec<String> {
    let root = workspace_root();
    let doc_path = root.join(rel_path);
    let text = fs::read_to_string(&doc_path)
        .unwrap_or_else(|e| panic!("read {}: {}", doc_path.display(), e));
    audit_doc(rel_path, &text, &root)
}

/// Audit the `//!`-prefixed top-of-file doc-comment block of one source file.
fn audit_source_doc_comment(rel_path: &str) -> Vec<String> {
    let root = workspace_root();
    let src_path = root.join(rel_path);
    let src = fs::read_to_string(&src_path)
        .unwrap_or_else(|e| panic!("read {}: {}", src_path.display(), e));
    // Take leading `//!` and blank-comment lines as the header block.
    let mut header = String::new();
    for line in src.lines() {
        if line.starts_with("//!") || line.is_empty() || line.starts_with("//") {
            header.push_str(line);
            header.push('\n');
        } else {
            break;
        }
    }
    audit_doc(rel_path, &header, &root)
}

#[test]
fn arithmetic_md_cites_resolve_at_head() {
    let failures = audit_design_doc(".design/ferrotorch-core/grad_fns/arithmetic.md");
    assert!(
        failures.is_empty(),
        "arithmetic.md has {} stale cite(s) (R-CITE-2):\n\n{}",
        failures.len(),
        failures.join("\n\n")
    );
}

#[test]
fn cumulative_md_cites_resolve_at_head() {
    let failures = audit_design_doc(".design/ferrotorch-core/grad_fns/cumulative.md");
    assert!(
        failures.is_empty(),
        "cumulative.md has {} stale cite(s) (R-CITE-2):\n\n{}",
        failures.len(),
        failures.join("\n\n")
    );
}

#[test]
fn arithmetic_rs_doc_comment_cites_resolve_at_head() {
    let failures = audit_source_doc_comment("ferrotorch-core/src/grad_fns/arithmetic.rs");
    assert!(
        failures.is_empty(),
        "arithmetic.rs `//!` doc-comment has {} stale cite(s) (R-CITE-2):\n\n{}",
        failures.len(),
        failures.join("\n\n")
    );
}

#[test]
fn cumulative_rs_doc_comment_cites_resolve_at_head() {
    let failures = audit_source_doc_comment("ferrotorch-core/src/grad_fns/cumulative.rs");
    assert!(
        failures.is_empty(),
        "cumulative.rs `//!` doc-comment has {} stale cite(s) (R-CITE-2):\n\n{}",
        failures.len(),
        failures.join("\n\n")
    );
}
