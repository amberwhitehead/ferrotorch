//! Generic cite-drift audit (closes #1268, scope-broadened by #1279).
//!
//! Parses every backtick-quoted `<filename>.rs:<N>` or
//! `<filename>.rs:<N>-<M>` substring (and the bare-colon continuation form
//! `:<N>` / `:<N>-<M>` that reuses the most-recently-mentioned file WITHIN
//! THE SAME backtick-quoted span) from EVERY `.md` file under `.design/`
//! (walked recursively via `std::fs::read_dir`), plus the `//!` doc-comment
//! of each covered `.rs` source file, and asserts that every cite resolves
//! to substantive content in the target file at HEAD.
//!
//! The walker-based primary test `all_design_docs_cites_resolve_at_head`
//! superseded the previous per-doc (`arithmetic_md`, `cumulative_md`) tests
//! — those were structurally blind to drift in any doc the test author
//! hadn't hand-listed, which is exactly how the indexing.md drift loop
//! (#1274) survived to ship. With the walker test in place, every NEW
//! `.design/**/*.md` file is automatically under audit the moment it's
//! committed.
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
    /// Byte offset in the doc line just AFTER the cite's closing
    /// backtick (or 0 if not tracked, e.g. bare-parens cites). Used by
    /// the post-cite hint promoter to find a trailing identifier-shaped
    /// backtick span like `` `cumulative.rs:449-484` (`test_cumsum_backward_*`) ``
    /// where the symbol hint appears AFTER the cite.
    end_pos_in_line: usize,
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
        self.emit_with_end(file_as_written, lo, hi, cite_start_in_line, 0)
    }

    fn emit_with_end(
        &mut self,
        file_as_written: String,
        lo: usize,
        hi: usize,
        cite_start_in_line: usize,
        end_pos_in_line: usize,
    ) {
        let cite_start_in_hint_line = cite_start_in_line + self.prefix_len;
        let symbol_hint = extract_symbol_hint(self.line_for_hints, cite_start_in_hint_line);
        self.out.push(Cite {
            file_as_written,
            line_start: lo,
            line_end: hi,
            symbol_hint,
            end_pos_in_line,
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
            scan_span_inner(span, ctx, start, end);
            i = end + 1;
        } else {
            i += 1;
        }
    }
    // Cross-span post-cite hint promotion (#1270 — pattern
    // `` `cumulative.rs:449-484` (`test_cumsum_backward_*`) ``):
    // For each cite emitted on this line, if a following backtick span
    // appears within ~10 chars of the cite-end and contains an
    // identifier-shaped token (test_*, *Backward, *_t, PascalCase), promote
    // it to the cite's symbol_hint. We preserve any existing more-specific
    // hint per the same priority rules as the in-span promoter.
    promote_cross_span_hints(line, &backtick_ranges, ctx.out);
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

/// For each cite emitted on this line, see if a backtick span starts
/// within ~20 chars after the cite's closing backtick and contains an
/// identifier-shaped token; if so, promote it to the cite's symbol_hint.
///
/// Catches the cumulative.md REQ-7 row pattern where the cite is in one
/// backtick span and the qualifying test name is in the next:
///
/// ```text
/// unit tests at `cumulative.rs:449-484` (`test_cumsum_backward_*`)
/// ```
///
/// The intervening `(`, `,`, ` ` etc. are skipped over (max 20 chars
/// distance to keep the cross-span heuristic tight — we don't want to
/// pick up the test name from an unrelated sentence further down).
fn promote_cross_span_hints(line: &str, backtick_ranges: &[(usize, usize)], cites: &mut [Cite]) {
    // The cites emitted on THIS line all carry end_pos_in_line > 0 (set
    // by emit_with_end). Cites carried over from prior lines have
    // end_pos_in_line == 0; skip those (we'd need their own line context).
    for cite in cites.iter_mut() {
        if cite.end_pos_in_line == 0 {
            continue;
        }
        let cite_end = cite.end_pos_in_line;
        // Find the first backtick span starting after cite_end + within 20
        // chars of cite_end.
        let mut chosen: Option<(usize, usize)> = None;
        for &(bs, be) in backtick_ranges {
            // The span at bs..be — its OPEN backtick is at bs-1; OPEN must
            // be after cite_end and within 20 chars.
            if bs == 0 {
                continue;
            }
            let open_pos = bs - 1;
            if open_pos <= cite_end {
                continue;
            }
            if open_pos - cite_end > 20 {
                continue;
            }
            chosen = Some((bs, be));
            break;
        }
        let Some((bs, be)) = chosen else {
            continue;
        };
        let candidate_span = &line[bs..be];
        // The span itself should NOT contain a cite-shape (`:` or `.`).
        if candidate_span.contains(':') || candidate_span.contains('.') {
            continue;
        }
        // Take the FIRST identifier-shaped token (whitespace or `,`
        // separated). Allow trailing `*` (e.g. `test_cumsum_backward_*`)
        // since that's how globbed test-family hints are written.
        let first = candidate_span
            .split([',', ' '])
            .map(str::trim)
            .find(|s| !s.is_empty())
            .unwrap_or("");
        let cleaned: String = first
            .trim_matches(|c: char| c == '*' || c == '(' || c == ')' || c == ';' || c == '.')
            .to_string();
        if cleaned.is_empty()
            || !cleaned
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_')
        {
            continue;
        }
        // STRICTLY restrict to test_* identifiers — see in-span promoter
        // above for the rationale (generic prose lists `(cite, symbol)`
        // pairs that would otherwise clobber pre-cite hints).
        if !cleaned.starts_with("test_") {
            continue;
        }
        if STOPWORD_HINTS.contains(&cleaned.as_str()) {
            continue;
        }
        let promote = match cite.symbol_hint.as_deref() {
            None => true,
            Some(prev) => !prev.starts_with("test_"),
        };
        if promote {
            cite.symbol_hint = Some(cleaned);
        }
    }
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
///
/// In addition to the pre-cite symbol hint extraction handled by
/// [`extract_symbol_hint`] (which looks at prose preceding the backtick
/// span), this function ALSO recognizes a post-cite hint: when a cite
/// token is immediately followed within the same span by an
/// identifier-shaped token, that token is treated as the cite's symbol
/// hint. This closes #1270's root cause: in `cumulative.md` the REQ-6/7
/// table rows write things like
/// `` `cumulative.rs:420-428 test_cumsum_negative_dim` `` where the test
/// name lives AFTER the cite, so the prose-preceding extractor cannot
/// see it.
fn scan_span_inner<'a>(
    span: &'a str,
    ctx: &mut CiteContext<'a>,
    span_offset: usize,
    span_end: usize,
) {
    // Span-local filename context (for `:N` continuations inside this span).
    let mut span_local_file: Option<String> = ctx.last_rs_file.clone();
    // Collect tokens so we can peek ahead for post-cite symbol hints.
    let tokens: Vec<&str> = span
        .split([',', ' '])
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    let mut emitted_indices: Vec<usize> = Vec::new();
    for (idx, tok) in tokens.iter().enumerate() {
        // Try a named cite (`<dir/>?<name>.rs:<N>(-<M>)?` or non-`.rs`).
        if let Some((file_or_none, lo, hi)) = parse_any_named_cite(tok) {
            match file_or_none {
                Some(file_as_written) => {
                    span_local_file = Some(file_as_written.clone());
                    ctx.last_rs_file = Some(file_as_written.clone());
                    ctx.emit_with_end(file_as_written, lo, hi, span_offset, span_end);
                    emitted_indices.push(idx);
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
                ctx.emit_with_end(file_as_written, lo, hi, span_offset, span_end);
                emitted_indices.push(idx);
            }
        }
    }
    // Post-cite hint pass (#1270): for each emitted cite, if the next token
    // in the span is a `test_*` identifier, promote it to the cite's
    // symbol_hint UNLESS the cite already has a `test_*` hint. This is
    // intentionally NARROW — restricted to `test_*` candidates — because
    // generic prose like `arithmetic::floor_divide at arithmetic.rs:2641,
    // FloorDivideBackward at :2484` would otherwise promote
    // `FloorDivideBackward` over the correct pre-cite hint `floor_divide`.
    // The actual #1270 pattern (`cumulative.rs:420-428
    // test_cumsum_negative_dim`) has a `test_*` token following the cite;
    // only that narrow shape is promoted here.
    if !emitted_indices.is_empty() {
        let cite_count = emitted_indices.len();
        let start_slot = ctx.out.len() - cite_count;
        for (slot_off, &tok_idx) in emitted_indices.iter().enumerate() {
            let next_idx = tok_idx + 1;
            if next_idx >= tokens.len() {
                continue;
            }
            let next_tok = tokens[next_idx];
            if next_tok.contains(':') || next_tok.contains('.') {
                continue;
            }
            let cleaned: String = next_tok
                .trim_matches(|c: char| c == '(' || c == ')' || c == ';' || c == '.')
                .to_string();
            if cleaned.is_empty()
                || !cleaned
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_')
            {
                continue;
            }
            // STRICTLY restrict to test_* identifiers to avoid clobbering
            // the correct pre-cite hint on lines that simply list multiple
            // (cite, symbol) pairs separated by `,` or `+`.
            if !cleaned.starts_with("test_") {
                continue;
            }
            if STOPWORD_HINTS.contains(&cleaned.as_str()) {
                continue;
            }
            let slot = start_slot + slot_off;
            let existing = ctx.out[slot].symbol_hint.clone();
            let promote = match existing.as_deref() {
                None => true,
                Some(prev) => !prev.starts_with("test_"),
            };
            if promote {
                ctx.out[slot].symbol_hint = Some(cleaned);
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

/// Result of resolving a cite path:
/// - `Resolved(PathBuf)`: file exists under a known crate root; audit it.
/// - `AllowedExternal`: path is in the documented external-cite allow-list
///   (upstream `/home/doll/pytorch/*`); do NOT audit, treat as a pass.
/// - `Unresolved`: path is neither in a known crate root nor in the allow-
///   list — this is a stale or typo'd cite and MUST fail the audit
///   (closes #1269 Gap C).
enum CitePath {
    Resolved(PathBuf),
    AllowedExternal,
    Unresolved,
}

/// Resolve a cite's `file_as_written` against the workspace.
///
/// Resolution order for paths containing `/`:
///   1. verbatim from workspace root
///   2. with `ferrotorch-core/src/` prepended (handles bare `ops/foo.rs`)
///   3. with `ferrotorch-core/` prepended (handles `src/...`)
///
/// Resolution order for plain basenames (priority high → low):
///   ferrotorch-core/src/grad_fns/, src/ops/, src/, src/autograd/,
///   ferrotorch-nn/src/, ferrotorch-vision/src/,
///   tools/parity-sweep/runner/src/
///
/// Allow-list (treated as `AllowedExternal`, not audited):
///   - any path starting with `/home/doll/pytorch/`
fn resolve_cite_path(root: &Path, file_as_written: &str) -> CitePath {
    // Upstream PyTorch cites — we don't audit those (would require pinning
    // the user's local clone). Treat as a pass.
    if file_as_written.starts_with("/home/doll/pytorch/") {
        return CitePath::AllowedExternal;
    }
    if file_as_written.contains('/') {
        let candidates = [
            file_as_written.to_string(),
            format!("ferrotorch-core/src/{file_as_written}"),
            format!("ferrotorch-core/{file_as_written}"),
        ];
        for c in &candidates {
            let p = root.join(c);
            if p.exists() {
                return CitePath::Resolved(p);
            }
        }
        return CitePath::Unresolved;
    }
    let basename = file_as_written;
    let candidates = [
        format!("ferrotorch-core/src/grad_fns/{basename}"),
        format!("ferrotorch-core/src/ops/{basename}"),
        format!("ferrotorch-core/src/{basename}"),
        format!("ferrotorch-core/src/autograd/{basename}"),
        format!("ferrotorch-nn/src/{basename}"),
        format!("ferrotorch-vision/src/{basename}"),
        format!("tools/parity-sweep/runner/src/{basename}"),
    ];
    for c in &candidates {
        let p = root.join(c);
        if p.exists() {
            return CitePath::Resolved(p);
        }
    }
    CitePath::Unresolved
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
/// on failure or None on success.
///
/// Failure modes:
///   1. Unresolvable cite path (Gap C / #1269) — not under a known crate
///      root AND not in the upstream allow-list. Catches typos like
///      `arithmatic.rs` and `gradfns/arithmetic.rs`.
///   2. Out-of-file line range.
///   3. No substantive content at the cited line(s).
///   4. Symbol-hint mismatch — the cite carries a named symbol hint
///      (`pub fn <name>`, `struct <Name>Backward`, `test_<name>`, a
///      method like `Tensor::<sym>_t`, etc.) but the named symbol is not
///      declared at the cited line(s). For POINT cites the window is ±0
///      (the cited line MUST exactly declare the symbol — closes #1269
///      Gap B); for RANGE cites the window is ±3 around the range.
fn validate_cite(cite: &Cite, root: &Path, doc_label: &str, doc_line_no: usize) -> Option<String> {
    let target = match resolve_cite_path(root, &cite.file_as_written) {
        CitePath::Resolved(p) => p,
        CitePath::AllowedExternal => return None,
        CitePath::Unresolved => {
            return Some(format!(
                "{doc_label}:{doc_line_no} cites unresolvable path `{file}:{lo}-{hi}` (not under any known crate root: ferrotorch-core/src/, ferrotorch-core/src/grad_fns/, ferrotorch-core/src/ops/, ferrotorch-core/src/autograd/, ferrotorch-nn/src/, ferrotorch-vision/src/, tools/parity-sweep/runner/src/; not on the `/home/doll/pytorch/` upstream allow-list). Either fix the typo or add the file to the resolver's candidate list.",
                file = cite.file_as_written,
                lo = cite.line_start,
                hi = cite.line_end,
            ));
        }
    };
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

    // Symbol-hint validation. Closes #1269 Gap A: the prior implementation
    // only validated `test_*` hints, so a `*Backward` cite drifting to a
    // wrong line silently passed. Now every recognizable symbol hint is
    // validated.
    //
    // Window strategy (closes #1269 Gap B):
    //   - POINT cite (lo == hi): window is ±0 — the cited line MUST
    //     literally contain the symbol declaration. A +1 line shift in
    //     the source surfaces as a failure.
    //   - RANGE cite (lo < hi): window is ±3 around the range — ranges
    //     are inherently approximate (a doc author may write `:733-806`
    //     for a fn whose decl starts at :733 even after small edits
    //     inside the body), so some tolerance is preserved.
    if let Some(symbol) = &cite.symbol_hint {
        let needles = build_symbol_needles(symbol);
        let (window_lo, window_hi) = if is_range {
            (
                cite.line_start.saturating_sub(3).max(1),
                (cite.line_end + 3).min(total),
            )
        } else {
            (cite.line_start, cite.line_start)
        };
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
            let window_desc = if is_range {
                format!("any line within :{window_lo}-{window_hi}")
            } else {
                format!(":{} exactly", cite.line_start)
            };
            return Some(format!(
                "{doc_label}:{doc_line_no} cites `{file}:{lo}{hi_disp}` (with symbol hint `{symbol}`) but {window_desc} does not declare it (needles: {needles:?}); actual :{lo} is: `{actual}`",
                file = cite.file_as_written,
                lo = cite.line_start,
                hi_disp = if is_range {
                    format!("-{}", cite.line_end)
                } else {
                    String::new()
                },
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
#[allow(
    dead_code,
    reason = "retained for backwards compatibility — the walker-based `all_design_docs_cites_resolve_at_head` test is now the primary audit surface, but this helper remains for any future per-doc invocation"
)]
fn audit_design_doc(rel_path: &str) -> Vec<String> {
    let root = workspace_root();
    let doc_path = root.join(rel_path);
    let text = fs::read_to_string(&doc_path)
        .unwrap_or_else(|e| panic!("read {}: {}", doc_path.display(), e));
    audit_doc(rel_path, &text, &root)
}

/// Walk `.design/` recursively and collect every `.md` file's path
/// relative to the workspace root. Skips non-`.md` files. Uses
/// `std::fs::read_dir` (no external crate) per dispatch constraints.
fn collect_design_docs(root: &Path) -> Vec<String> {
    let mut out = Vec::new();
    let design_dir = root.join(".design");
    if !design_dir.exists() {
        return out;
    }
    let mut stack: Vec<PathBuf> = vec![design_dir];
    while let Some(dir) = stack.pop() {
        let entries = match fs::read_dir(&dir) {
            Ok(it) => it,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let file_type = match entry.file_type() {
                Ok(ft) => ft,
                Err(_) => continue,
            };
            if file_type.is_dir() {
                stack.push(path);
            } else if file_type.is_file() && path.extension().and_then(|e| e.to_str()) == Some("md")
            {
                // Store path relative to workspace root for stable labels.
                if let Ok(rel) = path.strip_prefix(root) {
                    if let Some(s) = rel.to_str() {
                        out.push(s.to_string());
                    }
                }
            }
        }
    }
    out.sort();
    out
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

/// Workspace-wide cite-drift audit (closes #1279).
///
/// Walks every `.md` file under `.design/` (recursively), runs the
/// shared cite-extraction + cite-resolution logic on each, and emits a
/// per-file failure summary. This replaces the prior per-doc tests
/// (`arithmetic_md_cites_resolve_at_head`, `cumulative_md_cites_resolve_at_head`)
/// which were structurally blind to drift in `indexing.md`, `methods.md`,
/// `inplace.md`, `quantize_grad.md`, `ops/cumulative.md`, and any future
/// design doc the workspace grows. Per goal.md S3, the durable contract
/// is that NO `.design/**/*.md` cite drifts silently — that contract has
/// to cover every doc, not a hand-picked subset.
///
/// Also subsumes `divergence_indexing_md_uncovered_by_generic_cite_drift_audit.rs`
/// (which existed to pin the scope-gap that this walker closes).
#[test]
fn all_design_docs_cites_resolve_at_head() {
    let root = workspace_root();
    let docs = collect_design_docs(&root);
    assert!(
        !docs.is_empty(),
        "expected at least one .md file under .design/, found none"
    );
    let mut per_doc_failures: Vec<(String, Vec<String>)> = Vec::new();
    let mut total_failures = 0usize;
    for doc in &docs {
        let doc_path = root.join(doc);
        let text = match fs::read_to_string(&doc_path) {
            Ok(t) => t,
            Err(e) => {
                per_doc_failures.push((doc.clone(), vec![format!("read error: {e}")]));
                total_failures += 1;
                continue;
            }
        };
        let failures = audit_doc(doc, &text, &root);
        if !failures.is_empty() {
            total_failures += failures.len();
            per_doc_failures.push((doc.clone(), failures));
        }
    }
    assert!(
        per_doc_failures.is_empty(),
        "{n_docs_scanned} design doc(s) scanned, {n_failed_docs} doc(s) have stale cite(s) ({total_failures} total stale cite(s)) (R-CITE-2 + goal.md S3):\n\n{summary}",
        n_docs_scanned = docs.len(),
        n_failed_docs = per_doc_failures.len(),
        total_failures = total_failures,
        summary = per_doc_failures
            .iter()
            .map(|(doc, fails)| format!(
                "=== {doc} ({n} stale cite(s)) ===\n{body}",
                n = fails.len(),
                body = fails.join("\n\n")
            ))
            .collect::<Vec<_>>()
            .join("\n\n"),
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
