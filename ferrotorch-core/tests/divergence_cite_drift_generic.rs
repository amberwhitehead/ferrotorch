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
            if let Some((lo, hi)) = lo_hi_opt
                && (had_colon || lo >= 100)
                && ctx.last_rs_file.is_some()
            {
                let file = ctx.last_rs_file.clone().unwrap();
                ctx.emit(file, lo, hi, start);
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
        if let Some((lo, hi)) = parse_bare_colon_cite(tok)
            && let Some(file_as_written) = span_local_file.clone()
        {
            ctx.emit_with_end(file_as_written, lo, hi, span_offset, span_end);
            emitted_indices.push(idx);
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
        if let Ok(hi) = hi_str.parse::<usize>()
            && hi >= lo
        {
            return Some((lo, hi));
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
        format!("ferrotorch-data/src/{basename}"),
        format!("ferrotorch-distributions/src/{basename}"),
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

/// A parsed S3 line-number-FREE symbol anchor (the #1633 conversion form).
///
/// The #1633 S3 conversion replaced every `*Backward:NNN` LINE cite with a
/// line-number-free symbol anchor per R-CITE-2b/S3. The dominant form the
/// conversion produced is:
///
/// ```text
/// the `RsqrtBackward` struct in `grad_fns/arithmetic.rs` saving the output
/// ```
///
/// i.e. a backtick-quoted CamelCase struct name, the LITERAL words
/// `` ` struct in ` ``, then a backtick-quoted `.rs` path. Because there is
/// no line number, the existing `<file>.rs:<N>` cite machinery never saw
/// these anchors — so a renamed/moved/deleted `*Backward` struct produced a
/// silently-stale anchor no test caught (#1643). This struct + the parser
/// below + [`validate_struct_anchor`] close that gap.
#[derive(Debug, Clone)]
struct StructAnchor {
    /// The CamelCase symbol named inside the first backtick span (generics
    /// stripped), e.g. `RsqrtBackward`.
    symbol: String,
    /// The `.rs` path named inside the second backtick span, as written
    /// (e.g. `grad_fns/arithmetic.rs` or `arithmetic.rs`).
    file_as_written: String,
}

/// Parse every S3 struct-symbol anchor of the form
/// `` `<CamelCaseSymbol>` struct in `<path>.rs` `` from one doc line.
///
/// FALSE-POSITIVE CONTROL (this walker runs over ALL `.design/**/*.md`):
/// the parser is deliberately conservative — it requires ALL of:
///   1. a backtick-quoted symbol span whose content (after stripping an
///      optional `<...>` generic suffix) is a single CamelCase identifier
///      starting with an UPPERCASE ascii letter (so a struct name like
///      `RsqrtBackward`, not arbitrary prose);
///   2. the EXACT literal connective `` ` struct in ` `` between the two
///      backtick spans (so plain `<foo> in <bar>.rs` single-span prose is
///      NOT matched — only the explicit "struct in" declaration form #1633
///      produced);
///   3. a second backtick-quoted span whose content is an explicit
///      `.rs` file path (ends in `.rs`, identifier/`/`-shaped stem).
///
/// Anything failing any of the three is skipped, not matched. This scopes
/// the new check to exactly the #1633 `*Backward struct` anchor family
/// (plus the handful of other genuine `` `Foo` struct in `bar.rs` `` decls)
/// and away from the ~2700 generic `` `sym in file.rs` `` single-span
/// anchors (a separate, much larger family — see #1643 report).
fn parse_struct_anchors(line: &str) -> Vec<StructAnchor> {
    const CONNECTIVE: &str = "` struct in `";
    let mut out = Vec::new();
    let bytes = line.as_bytes();
    let mut search_from = 0usize;
    while let Some(rel) = line[search_from..].find(CONNECTIVE) {
        let conn_start = search_from + rel;
        let conn_end = conn_start + CONNECTIVE.len();
        // Advance the cursor unconditionally so a failed match can't loop.
        search_from = conn_end;
        // 1. Symbol span: walk backwards from `conn_start` (which is the
        //    closing backtick of the symbol span) to its opening backtick.
        //    `conn_start` points at the backtick char itself.
        if conn_start == 0 || bytes[conn_start] != b'`' {
            continue;
        }
        let sym_close = conn_start;
        let sym_open = match line[..sym_close].rfind('`') {
            Some(p) => p,
            None => continue,
        };
        let sym_raw = &line[sym_open + 1..sym_close];
        let Some(symbol) = camel_struct_ident(sym_raw) else {
            continue;
        };
        // 3. File span: starts at `conn_end` (just after the opening
        //    backtick of the file span) and runs to the next backtick.
        let file_close = match line[conn_end..].find('`') {
            Some(e) => conn_end + e,
            None => continue,
        };
        let file_raw = &line[conn_end..file_close];
        if !is_rs_path(file_raw) {
            continue;
        }
        out.push(StructAnchor {
            symbol,
            file_as_written: file_raw.to_string(),
        });
    }
    out
}

/// Accept `s` as a CamelCase struct identifier (optionally carrying a
/// `<...>` generic suffix, which is stripped). Returns the bare ident if it
/// is a single identifier-shaped token starting with an UPPERCASE ascii
/// letter; `None` otherwise. Rejects multi-word prose (anything with
/// whitespace before the optional `<`), method paths (`Foo::bar`), and
/// lowercase-leading names.
fn camel_struct_ident(s: &str) -> Option<String> {
    let stem = match s.find('<') {
        Some(lt) => &s[..lt],
        None => s,
    };
    let stem = stem.trim();
    if stem.is_empty() {
        return None;
    }
    let mut chars = stem.chars();
    let first = chars.next()?;
    if !first.is_ascii_uppercase() {
        return None;
    }
    if !chars.all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return None;
    }
    Some(stem.to_string())
}

/// Is `s` an explicit `.rs` file path (identifier/`/`/`-`/`.`-shaped stem
/// ending in `.rs`)? Used to gate the file span of a struct anchor.
fn is_rs_path(s: &str) -> bool {
    let s = s.trim();
    if !s.ends_with(".rs") {
        return false;
    }
    let stem = &s[..s.len() - 3];
    if stem.is_empty() {
        return false;
    }
    stem.chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '/' || c == '-' || c == '.')
}

/// Validate one S3 struct-symbol anchor against the file at HEAD. Returns
/// `Some(error)` if the named `.rs` file is unresolvable OR does not declare
/// the named struct (`struct <Symbol>` / `pub struct <Symbol>`), else
/// `None`.
///
/// This is the S3-era structural anti-drift contract (#1643): a symbol
/// anchor pointing at a renamed/moved/deleted struct is stale and MUST fail.
fn validate_struct_anchor(
    anchor: &StructAnchor,
    root: &Path,
    doc_label: &str,
    doc_line_no: usize,
) -> Option<String> {
    let target = match resolve_cite_path(root, &anchor.file_as_written) {
        CitePath::Resolved(p) => p,
        CitePath::AllowedExternal => return None,
        CitePath::Unresolved => {
            return Some(format!(
                "{doc_label}:{doc_line_no} S3 struct anchor `{sym}` struct in `{file}` names an unresolvable path (not under any known crate root). Either fix the typo or add the file to the resolver's candidate list.",
                sym = anchor.symbol,
                file = anchor.file_as_written,
            ));
        }
    };
    let src = match fs::read_to_string(&target) {
        Ok(s) => s,
        Err(_) => return None,
    };
    // The struct must be DECLARED in the named file. We require the literal
    // `struct <Symbol>` followed by a non-identifier char (so `RsqrtBackward`
    // does not spuriously match `RsqrtBackwardExt`). `pub`/visibility/derive
    // prefixes are irrelevant — `struct <Symbol>` appears verbatim in every
    // Rust struct decl regardless of visibility.
    let needle = format!("struct {}", anchor.symbol);
    let declared = src.lines().any(|line| {
        if let Some(idx) = line.find(&needle) {
            let after = line[idx + needle.len()..].chars().next();
            // Boundary: end-of-line, whitespace, `<` (generics), `{`, `(`,
            // `;`, or `:` (e.g. `struct Foo: Trait` is not valid but be lax).
            match after {
                None => true,
                Some(c) => !(c.is_ascii_alphanumeric() || c == '_'),
            }
        } else {
            false
        }
    });
    if !declared {
        return Some(format!(
            "{doc_label}:{doc_line_no} S3 struct anchor `{sym}` struct in `{file}` is STALE: `{needle}` is not declared in {path} at HEAD (struct renamed/moved/deleted?). Re-point the anchor to the correct file/symbol.",
            sym = anchor.symbol,
            file = anchor.file_as_written,
            path = target.display(),
        ));
    }
    None
}

// ===========================================================================
// #1668 — SINGLE-SPAN S3 SYMBOL ANCHORS
//
// The dominant S3 (goal.md) anchor form is a SINGLE backtick span that
// contains BOTH the symbol declaration AND the file, joined by the literal
// word " in ":
//
// ```text
// `pub fn pack_padded_sequence in rnn_utils.rs`
// `pub struct PagePool in paged_attention.rs`
// `mod tests in cache.rs`
// `reduce_all in meta_propagate.rs`            (bare Type::method assoc-fn)
// `impl Drop for CusparseLtHandle in cusparselt.rs`
// ```
//
// There are ~2900 of these across `.design/`, spanning every crate. Unlike
// the #1643 CROSS-span `` `Sym` struct in `file.rs` `` form (two separate
// backtick spans with a literal `` ` struct in ` `` connective between
// them), here EVERYTHING is inside one backtick span. The line-number cite
// machinery (`parse_any_named_cite`) never sees these — there is no `:N`
// suffix — and the #1643 struct-anchor parser explicitly does NOT match the
// single-span form (its `s3_struct_anchor_parser_is_precise_*` test pins
// that `` `struct NarrowBackward in methods.rs` `` is skipped). So a
// renamed/moved/deleted symbol behind one of these single-span anchors rots
// silently. This parser + resolver + validator close that gap.
//
// DISAMBIGUATION RULE (basename collisions): many crates share basenames
// like `lib.rs`, `mod.rs`, `model.rs`, `config.rs`, `gpu.rs`. The resolver
// indexes every `*/src/**/*.rs` basename -> ALL full paths once. An anchor's
// `<path>.rs` may be a bare basename (`rnn_utils.rs`) OR a prefixed suffix
// (`ops/search.rs`, `ferrotorch-gpu/src/backend_impl.rs`). Resolution:
//   1. If the written path contains `/`, accept any indexed file whose full
//      workspace-relative path ENDS WITH the written path (suffix match).
//   2. Otherwise (bare basename) take every indexed file with that basename.
// The anchor is VALID iff the symbol is DECLARED in AT LEAST ONE candidate
// file (the anchor asserts "this symbol lives in a file of this name"). If
// no candidate declares it, the anchor is STALE. If the basename indexes to
// ZERO files anywhere, the anchor is UNRESOLVABLE (typo / non-existent file).
// This "found in ANY candidate" rule (rather than "found in THE one true
// file") is the honest reading of the corpus: the anchors were authored as
// "symbol X is in file-named-Y", not "in this exact path", and tightening to
// a single canonical path would flag thousands of legitimately-ambiguous
// basenames as drift. The cross-span #1643 test keeps the stricter
// per-`resolve_cite_path` behavior for the struct family it owns.

/// The declaration KIND of a single-span symbol anchor. Drives the needle
/// set used to verify the symbol is genuinely declared (not just any
/// substring) in the resolved file.
#[derive(Debug, Clone, PartialEq, Eq)]
enum DeclKind {
    Fn,
    Struct,
    Enum,
    Trait,
    Mod,
    Const,
    Type,
    /// `impl <id>` or `impl <Trait> for <Type>` — validated against the
    /// `<Type>` (the impl TARGET), which is what actually has to exist.
    Impl,
    /// `<Type>::<method>` associated-fn form (with or without a leading
    /// `fn`/`pub fn`). Validated as a `fn <method>` declaration.
    AssocFn,
    /// A bare single snake_case/lowercase identifier with NO keyword and NO
    /// `::` (#1669). The dominant corpus shape (`add in arithmetic.rs`,
    /// `reduce_all in meta_propagate.rs`, `addcmul_t in methods.rs`). The
    /// declaration could be any item kind OR a `pub use` re-export (lib.rs
    /// facades re-export `activation`/`adamw` etc.), so validation accepts a
    /// broad needle set. Precision: EXACTLY one identifier token before
    /// " in " — multi-word prose ("the model in foo.rs") is excluded.
    BareIdent,
}

/// A parsed single-span symbol anchor `` `<decl> in <path>.rs` ``.
#[derive(Debug, Clone)]
struct SymbolAnchor {
    kind: DeclKind,
    /// The bare identifier to look for (generics stripped). For `AssocFn`
    /// this is the METHOD name (last `::` segment); for `Impl` it is the
    /// impl TARGET type (the name after `for`, or after `impl` if no `for`).
    ident: String,
    /// The `.rs` path as written (bare basename or prefixed suffix).
    file_as_written: String,
}

/// Strip a trailing `<...>` generic suffix and surrounding whitespace,
/// returning the bare identifier stem. Returns `None` if the result is not a
/// single identifier-shaped token (ascii alnum + `_`).
fn ident_stem(s: &str) -> Option<String> {
    let stem = match s.find('<') {
        Some(lt) => &s[..lt],
        None => s,
    };
    let stem = stem.trim();
    if stem.is_empty() {
        return None;
    }
    if !stem.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return None;
    }
    Some(stem.to_string())
}

/// Parse the DECL portion (everything before the literal " in ") of a
/// candidate single-span anchor into a `(DeclKind, ident)`. Returns `None`
/// if `decl` is not a recognizable symbol-declaration form (so arbitrary
/// prose like "the model" is rejected).
fn parse_decl(decl: &str) -> Option<(DeclKind, String)> {
    let decl = decl.trim();
    // Strip an optional leading visibility. We accept `pub` and
    // `pub(crate)` / `pub(super)` etc. — anything of the shape `pub` or
    // `pub(...)`.
    let rest = if let Some(after) = decl.strip_prefix("pub") {
        // strip_prefix removed only "pub"; handle an optional `(crate)` /
        // `(super)` / `(in path)` visibility restriction.
        if let Some(paren) = after.strip_prefix('(') {
            match paren.find(')') {
                Some(p) => paren[p + 1..].trim_start(),
                None => return None,
            }
        } else {
            after.trim_start()
        }
    } else {
        decl
    };

    // Keyword-led forms.
    for (kw, kind) in [
        ("fn ", DeclKind::Fn),
        ("struct ", DeclKind::Struct),
        ("enum ", DeclKind::Enum),
        ("trait ", DeclKind::Trait),
        ("mod ", DeclKind::Mod),
        ("const ", DeclKind::Const),
        ("type ", DeclKind::Type),
        ("impl ", DeclKind::Impl),
    ] {
        if let Some(after) = rest.strip_prefix(kw) {
            let after = after.trim();
            return parse_decl_body(kind, after);
        }
    }

    // No keyword: accept the bare `<Type>::<method>` associated-fn form.
    if rest.contains("::") {
        return parse_assoc_fn(rest);
    }

    // No keyword, no `::`: accept a BARE SINGLE snake_case/lowercase
    // identifier (#1669). PRECISION: the decl must be EXACTLY one identifier
    // token (no internal whitespace) whose first char is a lowercase ascii
    // letter or `_`. This excludes:
    //   - multi-word prose ("the model" -> two tokens before " in ");
    //   - uppercase-leading bare names (those are the #1643 cross-span struct
    //     family or prose, handled / rejected elsewhere);
    //   - empty / non-identifier shapes.
    // A bare lowercase ident like `add`, `reduce_all`, `addcmul_t`, `add_f32`
    // is the dominant genuine anchor shape and is now validated.
    if rest.split_whitespace().count() == 1 {
        let tok = rest.trim();
        let first = tok.chars().next()?;
        if (first.is_ascii_lowercase() || first == '_')
            && tok.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        {
            return Some((DeclKind::BareIdent, tok.to_string()));
        }
    }
    None
}

/// Parse the body of a keyword-led decl (everything after `fn `/`struct `/
/// …). Handles the `fn <Type>::<method>` mixed form (-> AssocFn), the
/// `impl <Trait> for <Type>` / `impl <Type>` forms, and the plain
/// `<kw> <ident>` form.
fn parse_decl_body(kind: DeclKind, body: &str) -> Option<(DeclKind, String)> {
    match kind {
        DeclKind::Fn => {
            // `fn Type::method` -> AssocFn (validate the method name).
            if body.contains("::") {
                return parse_assoc_fn(body);
            }
            let id = ident_stem(first_token(body))?;
            Some((DeclKind::Fn, id))
        }
        DeclKind::Impl => {
            // `impl <Trait> for <Type>` -> validate <Type>; `impl <Type>` ->
            // validate <Type>.
            let target = if let Some(idx) = body.find(" for ") {
                &body[idx + " for ".len()..]
            } else {
                body
            };
            let id = ident_stem(first_token(target))?;
            Some((DeclKind::Impl, id))
        }
        _ => {
            let id = ident_stem(first_token(body))?;
            Some((kind, id))
        }
    }
}

/// Parse a `<Type>::<method>` associated-fn form into `(AssocFn, method)`.
/// Requires BOTH a `<Type>` (uppercase-leading, identifier-shaped) and a
/// `<method>` (identifier-shaped) so plain prose with a stray `::` is not
/// matched.
fn parse_assoc_fn(s: &str) -> Option<(DeclKind, String)> {
    let s = first_token(s);
    let (ty, method) = s.split_once("::")?;
    // The type segment may itself carry generics; strip them. Require the
    // type to start uppercase (a real type name, not prose).
    let ty_stem = ident_stem(ty)?;
    if !ty_stem.chars().next()?.is_ascii_uppercase() {
        return None;
    }
    let method_stem = ident_stem(method)?;
    Some((DeclKind::AssocFn, method_stem))
}

/// Take the leading identifier-ish token (stop at the first space). Used so
/// `mod tests` -> `tests` and `impl Drop for Foo` is handled by callers that
/// split on " for " first.
fn first_token(s: &str) -> &str {
    s.split_whitespace().next().unwrap_or("").trim()
}

/// Parse every single-span symbol anchor `` `<decl> in <path>.rs` `` from one
/// doc line.
///
/// PRECISION (this runs over ALL `.design/**/*.md`):
///   1. Only the content of a SINGLE backtick span is examined.
///   2. The span must contain the literal " in " separating a recognizable
///      DECL (see [`parse_decl`]) from a `.rs` PATH ([`is_rs_path`]).
///   3. Upstream paths (`/home/doll/pytorch/`, `aten/`) are excluded — those
///      are read-only upstream cites, not workspace symbols.
///   4. The CROSS-span `` `Sym` struct in `file.rs` `` form (#1643) is NOT
///      matched here: in that form `struct` sits in a SEPARATE backtick span
///      from the file, so this single-span content never contains a valid
///      `<decl> in <path>.rs`. (`` `struct NarrowBackward in methods.rs` ``,
///      where the connective is INSIDE one span, IS a legitimate single-span
///      anchor and is matched.)
fn parse_symbol_anchors(line: &str) -> Vec<SymbolAnchor> {
    let mut out = Vec::new();
    let bytes = line.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] != b'`' {
            i += 1;
            continue;
        }
        let start = i + 1;
        let end = match line[start..].find('`') {
            Some(e) => start + e,
            None => break,
        };
        let span = &line[start..end];
        i = end + 1;
        if let Some(anchor) = parse_single_span(span) {
            out.push(anchor);
        }
    }
    out
}

/// Parse the CONTENT of one backtick span as a single-span symbol anchor.
fn parse_single_span(span: &str) -> Option<SymbolAnchor> {
    let span = span.trim();
    // Locate the LAST " in " that is followed by a `.rs` path (the file part
    // is at the tail; the decl may itself contain " in " only inside a string
    // which can't happen for our recognized forms, so taking the last is
    // robust).
    let in_idx = span.rfind(" in ")?;
    let (decl, file_part) = span.split_at(in_idx);
    let file_part = file_part[" in ".len()..].trim();
    // Exclude upstream cites — those are read-only and resolved elsewhere.
    if file_part.starts_with("/home/doll/pytorch/")
        || file_part.starts_with("aten/")
        || file_part.starts_with("torch/")
        || file_part.starts_with("c10/")
    {
        return None;
    }
    if !is_rs_path(file_part) {
        return None;
    }
    let (kind, ident) = parse_decl(decl)?;
    Some(SymbolAnchor {
        kind,
        ident,
        file_as_written: file_part.to_string(),
    })
}

/// One-shot index of every `*/src/**/*.rs`, `*/examples/**/*.rs`, and
/// `*/tests/**/*.rs` file in the workspace, keyed by basename -> all matching
/// workspace-relative paths. Built once per test.
///
/// #1669 extends the index beyond `src/` to also cover `examples/` and
/// `tests/` directories: a small number of single-span anchors legitimately
/// point at example/integration-test files (`<sym> in some_example.rs`), and
/// a `src/`-only index would report those as UNRESOLVABLE (false positives).
/// Indexing those dirs lets such anchors resolve to a real file.
fn build_src_index(root: &Path) -> std::collections::HashMap<String, Vec<PathBuf>> {
    let mut index: std::collections::HashMap<String, Vec<PathBuf>> =
        std::collections::HashMap::new();
    // Walk each `<crate>/src`, `<crate>/examples`, `<crate>/tests` directory
    // recursively.
    let mut crate_src_dirs: Vec<PathBuf> = Vec::new();
    if let Ok(entries) = fs::read_dir(root) {
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                for sub in ["src", "examples", "tests"] {
                    let d = p.join(sub);
                    if d.is_dir() {
                        crate_src_dirs.push(d);
                    }
                }
            }
        }
    }
    // Also include the parity-sweep runner src (a non-crate-root tool).
    let runner_src = root.join("tools/parity-sweep/runner/src");
    if runner_src.is_dir() {
        crate_src_dirs.push(runner_src);
    }
    let mut stack: Vec<PathBuf> = crate_src_dirs;
    while let Some(dir) = stack.pop() {
        let entries = match fs::read_dir(&dir) {
            Ok(it) => it,
            Err(_) => continue,
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().and_then(|x| x.to_str()) == Some("rs")
                && let Some(name) = p.file_name().and_then(|n| n.to_str())
            {
                index.entry(name.to_string()).or_default().push(p.clone());
            }
        }
    }
    index
}

/// Map a design-doc's workspace-relative label to the crate it documents.
///
/// A doc lives at `.design/<crate-or-subpath>/...`. The FIRST path component
/// under `.design/` is the crate name (`.design/ferrotorch-rl/X.md` documents
/// `ferrotorch-rl`; `.design/ferrotorch-core/ops/Y.md` documents
/// `ferrotorch-core`). Returns the crate's `src/` directory (workspace
/// relative, e.g. `ferrotorch-rl/src`) when that crate has a `src/` dir under
/// `root`, else `None` (e.g. top-level `.design/phase-0-orchestrator.md` docs
/// no single crate, or the named subdir isn't a crate root).
fn doc_crate_src_dir(root: &Path, doc_label: &str) -> Option<PathBuf> {
    let rel = doc_label.replace('\\', "/");
    let after = rel.strip_prefix(".design/")?;
    let first = after.split('/').next()?;
    if first.is_empty() || !first.starts_with("ferrotorch") {
        return None;
    }
    let src = root.join(first).join("src");
    if src.is_dir() { Some(src) } else { None }
}

/// Resolve a single-span anchor's `file_as_written` to the candidate files
/// it could refer to, per the documented disambiguation rule. Returns the
/// (possibly multi-element) candidate list, or an empty Vec if no file of
/// that basename exists anywhere (UNRESOLVABLE).
///
/// CRATE DISAMBIGUATION (#1669, fixes the #1668 cross-crate false-accept hole):
/// `doc_label` is the workspace-relative path of the design doc the anchor was
/// found in. A doc at `.design/<crate>/...` documents `<crate>`; a bare
/// basename anchor (`lib.rs`, `mod.rs`) is resolved PREFERENTIALLY to a file
/// of that basename under the matching crate's `src/`. Validation then runs
/// against THAT crate-local file, so a bare basename whose doc-crate sibling
/// lacks the symbol is STALE even if some OTHER crate's same-named file has it.
/// Only when the doc-crate has NO file of that basename (or the doc maps to no
/// crate — top-level `.design/*.md`) do we fall back to the cross-crate
/// candidate set. A prefixed-path anchor (`ops/search.rs`,
/// `ferrotorch-gpu/src/backend_impl.rs`) already disambiguates by suffix —
/// that path is unchanged.
fn resolve_symbol_anchor_files(
    index: &std::collections::HashMap<String, Vec<PathBuf>>,
    root: &Path,
    doc_label: &str,
    file_as_written: &str,
) -> Vec<PathBuf> {
    let basename = file_as_written
        .rsplit('/')
        .next()
        .unwrap_or(file_as_written);
    let all = match index.get(basename) {
        Some(v) => v,
        None => return Vec::new(),
    };
    if file_as_written.contains('/') {
        // Prefixed path: keep only candidates whose workspace-relative path
        // ends with the written suffix. Tolerate a leading `ferrotorch-...`
        // crate prefix mismatch by matching on the written suffix verbatim.
        let mut matches: Vec<PathBuf> = all
            .iter()
            .filter(|p| {
                let rel = p.strip_prefix(root).unwrap_or(p);
                let rel_s = rel.to_string_lossy().replace('\\', "/");
                rel_s.ends_with(file_as_written)
            })
            .cloned()
            .collect();
        // If the suffix match found nothing but the basename exists, fall
        // back to all basename candidates (the prefix is advisory; the
        // anchor is "valid" if the symbol exists in a file of that name).
        if matches.is_empty() {
            matches = all.clone();
        }
        matches
    } else {
        // Bare basename: bind to the DOC'S OWN CRATE if that crate has a file
        // of this basename. This closes the cross-crate false-accept hole:
        // an anchor in `.design/ferrotorch-rl/...` saying `mod tests in lib.rs`
        // is validated against `ferrotorch-rl/src/lib.rs`, NOT against every
        // `lib.rs` in the workspace.
        if let Some(crate_src) = doc_crate_src_dir(root, doc_label) {
            let crate_local: Vec<PathBuf> = all
                .iter()
                .filter(|p| p.starts_with(&crate_src))
                .cloned()
                .collect();
            if !crate_local.is_empty() {
                return crate_local;
            }
            // No file of this basename in the doc's crate — fall through to
            // the cross-crate candidate set (documented fallback: the anchor
            // names a basename the doc-crate doesn't have, e.g. a doc that
            // genuinely references a sibling crate's file by bare basename).
        }
        all.clone()
    }
}

/// Does `src` genuinely DECLARE the anchor's symbol per its kind? Tolerant of
/// visibility (`pub`/`pub(crate)`) and derive/attribute prefixes, but checks
/// the real declaration keyword + identifier with a word boundary so
/// `fn foo` does not match `fn foobar`.
fn anchor_symbol_declared(src: &str, anchor: &SymbolAnchor) -> bool {
    let needles: Vec<String> = match anchor.kind {
        DeclKind::Fn | DeclKind::AssocFn => vec![format!("fn {}", anchor.ident)],
        DeclKind::Struct => vec![format!("struct {}", anchor.ident)],
        DeclKind::Enum => vec![format!("enum {}", anchor.ident)],
        DeclKind::Trait => vec![format!("trait {}", anchor.ident)],
        DeclKind::Mod => vec![format!("mod {}", anchor.ident)],
        DeclKind::Const => vec![
            format!("const {}", anchor.ident),
            // `const fn foo` is a fn, but a `const FOO:` is the const form.
        ],
        DeclKind::Type => vec![format!("type {}", anchor.ident)],
        // For an impl TARGET we accept the type being declared as a struct,
        // enum, trait, type alias, OR appearing as an `impl ... <Type>` /
        // `impl <Type>` target — the type must EXIST in the file. The most
        // robust single check is `<Type>` appearing after `impl`, `struct`,
        // `enum`, `trait`, or `type`.
        DeclKind::Impl => vec![
            format!("struct {}", anchor.ident),
            format!("enum {}", anchor.ident),
            format!("trait {}", anchor.ident),
            format!("type {}", anchor.ident),
            format!("impl {}", anchor.ident),
            format!("for {}", anchor.ident),
        ],
        // A bare ident could be declared as ANY item kind. Accept every
        // declaration keyword; the `pub use` re-export forms are handled
        // separately below (they don't fit the `<kw> <ident>` word-boundary
        // shape — `pub use foo::bar::baz` ends the ident, but a re-export of
        // `activation` may appear as `pub use ...::activation;` or
        // `pub use activation::...`).
        DeclKind::BareIdent => vec![
            format!("fn {}", anchor.ident),
            format!("struct {}", anchor.ident),
            format!("enum {}", anchor.ident),
            format!("trait {}", anchor.ident),
            format!("mod {}", anchor.ident),
            format!("const {}", anchor.ident),
            format!("static {}", anchor.ident),
            format!("type {}", anchor.ident),
            format!("macro_rules! {}", anchor.ident),
        ],
    };
    let declared = src.lines().any(|line| {
        needles.iter().any(|needle| {
            if let Some(idx) = line.find(needle.as_str()) {
                let after = line[idx + needle.len()..].chars().next();
                match after {
                    None => true,
                    Some(c) => !(c.is_ascii_alphanumeric() || c == '_'),
                }
            } else {
                false
            }
        })
    });
    if declared {
        return true;
    }
    // Bare-ident re-export fallback (#1669): lib.rs facade modules re-export
    // submodules/items rather than declaring them locally — e.g.
    // `pub use activation::*;` / `pub use crate::optim::adamw;` /
    // `pub use self::metrics::accuracy_score`. Accept a `pub use` line that
    // mentions the ident as a path segment with word boundaries on both sides.
    if anchor.kind == DeclKind::BareIdent {
        return src.lines().any(|line| {
            let t = line.trim_start();
            if !t.starts_with("pub use ") && !t.starts_with("use ") {
                return false;
            }
            line_has_path_segment(line, &anchor.ident)
        });
    }
    false
}

/// Does `line` contain `ident` as a whole path segment (bounded on BOTH sides
/// by a non-identifier char or `::` / line edge)? Used to validate that a
/// `pub use ...` re-export line genuinely references the bare ident, so a
/// re-export of `adam` does not spuriously satisfy an anchor for `adamw`.
fn line_has_path_segment(line: &str, ident: &str) -> bool {
    let bytes = line.as_bytes();
    let mut from = 0usize;
    while let Some(rel) = line[from..].find(ident) {
        let idx = from + rel;
        let before_ok = idx == 0 || {
            let c = line[..idx].chars().next_back().unwrap_or(' ');
            !(c.is_ascii_alphanumeric() || c == '_')
        };
        let after_idx = idx + ident.len();
        let after_ok = after_idx >= bytes.len() || {
            let c = line[after_idx..].chars().next().unwrap_or(' ');
            !(c.is_ascii_alphanumeric() || c == '_')
        };
        if before_ok && after_ok {
            return true;
        }
        from = idx + ident.len();
    }
    false
}

/// Outcome of validating a single-span anchor, for report-mode counting.
#[derive(Debug, PartialEq, Eq)]
enum AnchorOutcome {
    Valid,
    /// File(s) resolved but none declare the symbol.
    Stale,
    /// No file of that basename exists anywhere in the workspace.
    Unresolvable,
    /// NOT a real symbol DECLARATION anchor — a `BareIdent` `` `<id> in <file>` ``
    /// whose named file EXISTS but does NOT declare `<id>` at HEAD.
    ///
    /// #1669 (final) RE-CLASSIFICATION — the corpus-grounded precision rule:
    /// the bare-identifier single-span form is OVERWHELMINGLY used as a PROSE
    /// reference, NOT a declaration claim. Re-running report mode + reading the
    /// 291 residual lines verbatim showed EVERY one is a consumer-citation, a
    /// call-site, a local-variable mention, or an op-usage reference — never a
    /// "this symbol is DECLARED here" claim whose target drifted. Concretely:
    ///   - `` `gpu_matmul_f32 in backend_impl.rs` `` — "the backend's
    ///     `CudaBackendImpl` *calls* `gpu_matmul_f32`" (the kernel is declared
    ///     in `blas.rs`; `backend_impl.rs` is the CONSUMER the doc intends);
    ///   - `` `cast_i32_to_f32 in backend_impl.rs` `` — literally prefixed
    ///     "Non-test consumer:";
    ///   - `` `argmax_f32 in backend_impl.rs` dispatches … `` — the
    ///     `CudaBackendImpl::argmax_f32` method BODY genuinely lives in
    ///     `backend_impl.rs` (re-pointing it to the kernel file would be FALSE);
    ///   - `` `detach in checkpoint.rs` `` — "the `detach()` on `grad_output`";
    ///   - `` `class_dirs in folder.rs` `` — "`class_dirs.sort_by(...)`" (a
    ///     local variable);
    ///   - `` `reshape in flex_attention.rs` `` — "via `grad_fns::shape::reshape`"
    ///     (an op CALL; `reshape` is declared in `grad_fns/shape.rs`).
    ///
    /// Re-pointing any of these to the symbol's DECLARATION file (as a naive
    /// "fix the drift" pass would) injects a factually-wrong claim into the doc
    /// and erases the genuine consumer/call-site evidence (the very R-DEFER-1
    /// "name a non-test production consumer" convention these anchors encode).
    ///
    /// So the SOUND, zero-false-positive contract for the bare-ident form is:
    /// it is a gated DECLARATION anchor ONLY when `<id>` IS declared in a
    /// candidate file of the named basename (→ `Valid`). When the file exists
    /// but does not declare `<id>`, the span is a prose reference → `Prose`
    /// (neither passes nor fails the gate — it is simply not a declaration
    /// claim). When the named file exists NOWHERE under any `*/src/**`,
    /// `*/examples/**`, or `*/tests/**` it is `Unresolvable` (a genuine typo /
    /// deleted file — gated for ALL kinds, see [`AnchorOutcome::Unresolvable`]).
    /// The file-stem / dir-segment sub-case (`laplace in laplace.rs`,
    /// `gpu in gpu/unet.rs` — a module referenced by its own name) is one
    /// instance of this broader prose family; it is still recognised and folds
    /// into the same `Prose` outcome. (If such a span DID resolve to a real
    /// `fn <stem>` / `struct <stem>` decl it would be `Valid`, not `Prose`.)
    ///
    /// The KEYWORD-LED / `Type::method` forms (`pub fn X in Y`, `struct X in Y`,
    /// `impl T for U in Y`) are EXPLICIT declaration claims — those keep the
    /// strict `Stale` gate (a keyword-led wrong-file IS genuine drift).
    Prose,
}

/// Does `ident` equal a PATH COMPONENT of `file_as_written` — the basename
/// stem (without `.rs`) OR any intermediate directory segment? Used by the
/// file-stem prose rule (#1669).
///
/// `device` matches `device.rs` (stem). `gpu` matches `gpu/unet.rs` (a
/// directory segment — the prose form `` `gpu in gpu/unet.rs` `` names the GPU
/// module by its dir name, not a symbol). `arithmetic` matches
/// `grad_fns/arithmetic.rs` (stem). A genuine multi-segment symbol like
/// `gpu_backend` or `reduce_all` does NOT equal any single path component, so
/// it is NOT reclassified as prose by this rule.
fn ident_is_path_component(ident: &str, file_as_written: &str) -> bool {
    file_as_written.split('/').any(|seg| {
        let stem = seg.strip_suffix(".rs").unwrap_or(seg);
        stem == ident
    })
}

/// Validate one single-span anchor, returning its outcome + (on
/// failure) a human-readable diagnostic.
fn validate_symbol_anchor(
    anchor: &SymbolAnchor,
    index: &std::collections::HashMap<String, Vec<PathBuf>>,
    root: &Path,
    doc_label: &str,
    doc_line_no: usize,
) -> (AnchorOutcome, Option<String>) {
    let candidates = resolve_symbol_anchor_files(index, root, doc_label, &anchor.file_as_written);
    if candidates.is_empty() {
        return (
            AnchorOutcome::Unresolvable,
            Some(format!(
                "{doc_label}:{doc_line_no} single-span anchor `{decl}` names file `{file}` which does not exist anywhere under any `*/src/**` in the workspace (typo / deleted file?).",
                decl = anchor_decl_repr(anchor),
                file = anchor.file_as_written,
            )),
        );
    }
    for cand in &candidates {
        if let Ok(src) = fs::read_to_string(cand)
            && anchor_symbol_declared(&src, anchor)
        {
            return (AnchorOutcome::Valid, None);
        }
    }
    // BARE-IDENT PROSE RULE (#1669 final). A `BareIdent` `` `<id> in <file>` ``
    // whose named file EXISTS (candidates non-empty, checked above) but does
    // NOT declare `<id>` is a PROSE reference — a consumer citation, a
    // call-site, a local-variable mention, an op-usage, or a module-name
    // reference — NOT a declaration claim whose target drifted. See the
    // [`AnchorOutcome::Prose`] doc for the corpus evidence (all 291 residual
    // bare-ident lines were verbatim prose, never declaration drift). The
    // file-stem / dir-segment case (`laplace in laplace.rs`, `gpu in gpu/unet.rs`)
    // is one instance and is recognised explicitly below for the targeted
    // precision pin; the general bare-ident case folds into the same `Prose`
    // outcome. The keyword-led / `Type::method` forms are explicit declaration
    // claims and fall through to `Stale` (they never reach this branch with a
    // `BareIdent` kind).
    if anchor.kind == DeclKind::BareIdent {
        return (AnchorOutcome::Prose, None);
    }
    (
        AnchorOutcome::Stale,
        Some(format!(
            "{doc_label}:{doc_line_no} single-span anchor `{decl} in {file}` is STALE: the symbol `{ident}` ({kind:?}) is not declared in any of {n} candidate file(s) named `{file}` at HEAD (renamed/moved/deleted?). Candidates: {cands}",
            decl = anchor_decl_repr(anchor),
            file = anchor.file_as_written,
            ident = anchor.ident,
            kind = anchor.kind,
            n = candidates.len(),
            cands = candidates
                .iter()
                .map(|p| p
                    .strip_prefix(root)
                    .unwrap_or(p)
                    .to_string_lossy()
                    .into_owned())
                .collect::<Vec<_>>()
                .join(", "),
        )),
    )
}

/// A best-effort human-readable redisplay of an anchor's decl (for error
/// messages only).
fn anchor_decl_repr(anchor: &SymbolAnchor) -> String {
    match anchor.kind {
        DeclKind::Fn => format!("fn {}", anchor.ident),
        DeclKind::AssocFn => format!("<Type>::{}", anchor.ident),
        DeclKind::Struct => format!("struct {}", anchor.ident),
        DeclKind::Enum => format!("enum {}", anchor.ident),
        DeclKind::Trait => format!("trait {}", anchor.ident),
        DeclKind::Mod => format!("mod {}", anchor.ident),
        DeclKind::Const => format!("const {}", anchor.ident),
        DeclKind::Type => format!("type {}", anchor.ident),
        DeclKind::Impl => format!("impl ... {}", anchor.ident),
        DeclKind::BareIdent => anchor.ident.clone(),
    }
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
        // S3 line-number-FREE struct-symbol anchors (#1633 conversion /
        // #1643): `` `<Sym>` struct in `<file>.rs` ``. These carry no line
        // number, so the cite machinery above never sees them; validate them
        // structurally — the named struct MUST be declared in the named file
        // at HEAD.
        for anchor in parse_struct_anchors(line) {
            if let Some(err) = validate_struct_anchor(&anchor, root, doc_label, doc_line_no) {
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
                if let Ok(rel) = path.strip_prefix(root)
                    && let Some(s) = rel.to_str()
                {
                    out.push(s.to_string());
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

/// #1643 SCOPED CONTRACT: every S3 line-number-FREE struct-symbol anchor
/// `` `<Sym>` struct in `<file>.rs` `` across ALL `.design/**/*.md` resolves
/// to a real `struct <Sym>` declaration in the named file at HEAD.
///
/// This is the narrow S3-era anti-drift contract #1643 ships: the #1633
/// conversion replaced `*Backward:NNN` line cites with these line-number-free
/// symbol anchors, which the line-number cite machinery is structurally blind
/// to. This test isolates JUST the struct-anchor family so it gives a clean
/// green signal independent of the pre-existing line-number-cite drift the
/// broader `all_design_docs_cites_resolve_at_head` walker (separately) flags.
#[test]
fn all_design_docs_s3_struct_anchors_resolve_at_head() {
    let root = workspace_root();
    let docs = collect_design_docs(&root);
    assert!(
        !docs.is_empty(),
        "expected at least one .md file under .design/"
    );
    let mut failures: Vec<String> = Vec::new();
    let mut anchor_count = 0usize;
    for doc in &docs {
        let text = match fs::read_to_string(root.join(doc)) {
            Ok(t) => t,
            Err(_) => continue,
        };
        for (i, line) in text.lines().enumerate() {
            for anchor in parse_struct_anchors(line) {
                anchor_count += 1;
                if let Some(err) = validate_struct_anchor(&anchor, &root, doc, i + 1) {
                    failures.push(err);
                }
            }
        }
    }
    // Sanity: the #1633 conversion produced a non-trivial number of these
    // (the *Backward struct family alone is ~20); if this drops to zero the
    // parser has silently stopped matching the S3 form.
    assert!(
        anchor_count >= 10,
        "expected >=10 S3 struct-symbol anchors across .design/ (the #1633 *Backward conversion family), found {anchor_count} — has the parser stopped matching the `` `Sym` struct in `file.rs` `` form?",
    );
    assert!(
        failures.is_empty(),
        "{n} S3 struct-symbol anchor(s) are STALE (renamed/moved/deleted struct) out of {anchor_count} scanned (#1643 / goal.md S3 R-CITE-2b):\n\n{body}",
        n = failures.len(),
        body = failures.join("\n\n"),
    );
}

/// Positive proof the #1643 struct-anchor parser + validator work — and that
/// the parser is PRECISE (does not match non-anchor prose). All assertions
/// run on synthetic in-memory input, so this is independent of the corpus
/// state. Closes the #1269 gap-A premise (the walker now DOES validate the
/// S3-era replacement for the retired `*Backward:NNN` line cites).
#[test]
fn s3_struct_anchor_parser_is_precise_and_catches_corruption() {
    let root = workspace_root();

    // 1. PARSE: the canonical S3 form is recognized and the symbol/file are
    //    extracted correctly (generics stripped).
    let line = "the `RsqrtBackward` struct in `grad_fns/arithmetic.rs` saving c";
    let anchors = parse_struct_anchors(line);
    assert_eq!(
        anchors.len(),
        1,
        "expected 1 anchor from `{line}`, got {anchors:?}"
    );
    assert_eq!(anchors[0].symbol, "RsqrtBackward");
    assert_eq!(anchors[0].file_as_written, "grad_fns/arithmetic.rs");

    let generic_line = "`FlexAttentionBackward<T>` struct in `flex_attention.rs`";
    let g = parse_struct_anchors(generic_line);
    assert_eq!(g.len(), 1, "generics should be stripped: {g:?}");
    assert_eq!(g[0].symbol, "FlexAttentionBackward");

    // 2. PRECISION: forms that are NOT the explicit `` `Sym` struct in `f.rs` ``
    //    anchor must NOT be matched (false-positive control over the corpus).
    for non_anchor in [
        // generic single-span `sym in file.rs` (the ~2700-strong family) —
        // no `struct` connective:
        "`add in arithmetic.rs`",
        // single-span `struct Sym in file.rs` — connective is inside the
        // backticks, not the ` ` struct in ` ` cross-span form:
        "`struct NarrowBackward in methods.rs`",
        // lowercase-leading symbol (not a struct name):
        "`foo` struct in `bar.rs`",
        // non-.rs file:
        "`Foo` struct in `bar.py`",
        // multi-word prose in the symbol span:
        "`every public` struct in `transformer.rs`",
        // method path in the symbol span:
        "`Tensor::sub_t` struct in `arithmetic.rs`",
        // bare prose, no backticks:
        "the AliasTable struct in dataset.rs",
    ] {
        assert!(
            parse_struct_anchors(non_anchor).is_empty(),
            "PRECISION FAILURE: parser matched a non-anchor: `{non_anchor}` -> {:?}",
            parse_struct_anchors(non_anchor),
        );
    }

    // 3. VALIDATE — GOOD: a real struct resolves clean.
    let good = StructAnchor {
        symbol: "RsqrtBackward".to_string(),
        file_as_written: "grad_fns/arithmetic.rs".to_string(),
    };
    assert!(
        validate_struct_anchor(&good, &root, "synthetic", 1).is_none(),
        "RsqrtBackward should resolve in grad_fns/arithmetic.rs at HEAD",
    );

    // 4. VALIDATE — CORRUPTED SYMBOL: a renamed/typo'd struct name is flagged.
    let bad_symbol = StructAnchor {
        symbol: "RsqrtBackwardXYZ_DOES_NOT_EXIST".to_string(),
        file_as_written: "grad_fns/arithmetic.rs".to_string(),
    };
    let err = validate_struct_anchor(&bad_symbol, &root, "synthetic", 2)
        .expect("corrupted struct symbol MUST be flagged as STALE");
    assert!(err.contains("STALE"), "error should say STALE: {err}");

    // 5. VALIDATE — CORRUPTED FILE: a real struct named in the WRONG file is
    //    flagged (the exact AliasTable-style drift this enhancement caught:
    //    struct exists, but not in the named file).
    let bad_file = StructAnchor {
        symbol: "RsqrtBackward".to_string(),
        file_as_written: "grad_fns/cumulative.rs".to_string(),
    };
    let err = validate_struct_anchor(&bad_file, &root, "synthetic", 3)
        .expect("struct named in the WRONG file MUST be flagged as STALE");
    assert!(err.contains("STALE"), "error should say STALE: {err}");

    // 6. VALIDATE — UNRESOLVABLE FILE: a typo'd path is flagged, not skipped.
    let typo_file = StructAnchor {
        symbol: "RsqrtBackward".to_string(),
        file_as_written: "grad_fns/arithmatic.rs".to_string(),
    };
    assert!(
        validate_struct_anchor(&typo_file, &root, "synthetic", 4).is_some(),
        "typo'd file path in a struct anchor MUST be flagged, not silently skipped",
    );
}

/// #1668 REPORT MODE: count single-span anchors total / valid / stale /
/// unresolvable across `.design/`. `#[ignore]` so it does not gate CI; run
/// with `--ignored -- --nocapture` to see the cleanup scope. Kept as a
/// permanent diagnostic so the cleanup campaign can re-measure residual.
/// The reason string leads with the `diagnostic:` marker that
/// `ignore_reasons_carry_issue_ref_or_diagnostic_marker` (CORE-207 /
/// #1901) recognizes as the sanctioned alternative to a tracking-issue
/// reference.
#[test]
#[ignore = "diagnostic: run with --ignored --nocapture to print single-span anchor counts"]
fn report_single_span_anchor_counts() {
    let root = workspace_root();
    let index = build_src_index(&root);
    let docs = collect_design_docs(&root);
    // Overall counts.
    let (mut total, mut valid, mut stale, mut unres, mut prose) = (0, 0, 0, 0, 0usize);
    // Bare-ident-only counts (the #1669 family).
    let (mut bi_total, mut bi_valid, mut bi_stale, mut bi_unres, mut bi_prose) =
        (0, 0, 0, 0, 0usize);
    let mut stale_list: Vec<String> = Vec::new();
    let mut unres_list: Vec<String> = Vec::new();
    for doc in &docs {
        let text = match fs::read_to_string(root.join(doc)) {
            Ok(t) => t,
            Err(_) => continue,
        };
        for (i, line) in text.lines().enumerate() {
            for anchor in parse_symbol_anchors(line) {
                total += 1;
                let is_bare = anchor.kind == DeclKind::BareIdent;
                if is_bare {
                    bi_total += 1;
                }
                let (outcome, msg) = validate_symbol_anchor(&anchor, &index, &root, doc, i + 1);
                match outcome {
                    AnchorOutcome::Valid => {
                        valid += 1;
                        if is_bare {
                            bi_valid += 1;
                        }
                    }
                    AnchorOutcome::Stale => {
                        stale += 1;
                        if is_bare {
                            bi_stale += 1;
                        }
                        if let Some(m) = msg {
                            stale_list.push(m);
                        }
                    }
                    AnchorOutcome::Unresolvable => {
                        unres += 1;
                        if is_bare {
                            bi_unres += 1;
                        }
                        if let Some(m) = msg {
                            unres_list.push(m);
                        }
                    }
                    AnchorOutcome::Prose => {
                        prose += 1;
                        if is_bare {
                            bi_prose += 1;
                        }
                    }
                }
            }
        }
    }
    println!("=== #1668/#1669 single-span anchor report ===");
    println!("docs scanned: {}", docs.len());
    println!("total parsed:  {total}");
    println!("valid:         {valid}");
    println!("STALE:         {stale}");
    println!("UNRESOLVABLE:  {unres}");
    println!("PROSE (file-stem, non-anchor): {prose}");
    println!("--- bare-ident only ---");
    println!("bare total:        {bi_total}");
    println!("bare valid:        {bi_valid}");
    println!("bare STALE:        {bi_stale}");
    println!("bare UNRESOLVABLE: {bi_unres}");
    println!("bare PROSE:        {bi_prose}");
    println!("--- STALE ---");
    for m in &stale_list {
        println!("{m}");
    }
    println!("--- UNRESOLVABLE ---");
    for m in &unres_list {
        println!("{m}");
    }
}

/// #1668 PRECISION unit test — the single-span anchor parser must match ONLY
/// genuine `<decl> in <path>.rs` symbol anchors and skip prose, upstream
/// cites, and the #1643 CROSS-span form. Pure in-memory; corpus-independent.
#[test]
fn single_span_anchor_parser_is_precise() {
    // GENUINE anchors that MUST parse (one each).
    let good: &[(&str, DeclKind, &str, &str)] = &[
        (
            "`pub fn pack_padded_sequence in rnn_utils.rs`",
            DeclKind::Fn,
            "pack_padded_sequence",
            "rnn_utils.rs",
        ),
        (
            "`pub struct PagePool in paged_attention.rs`",
            DeclKind::Struct,
            "PagePool",
            "paged_attention.rs",
        ),
        (
            "`mod tests in cache.rs`",
            DeclKind::Mod,
            "tests",
            "cache.rs",
        ),
        (
            "`pub enum InterpolateMode in upsample.rs`",
            DeclKind::Enum,
            "InterpolateMode",
            "upsample.rs",
        ),
        (
            "`pub trait Foo in lib.rs`",
            DeclKind::Trait,
            "Foo",
            "lib.rs",
        ),
        (
            "`pub const MAX_X in limits.rs`",
            DeclKind::Const,
            "MAX_X",
            "limits.rs",
        ),
        (
            "`pub type Handle in types.rs`",
            DeclKind::Type,
            "Handle",
            "types.rs",
        ),
        (
            "`fn cubic_weight in upsample.rs`",
            DeclKind::Fn,
            "cubic_weight",
            "upsample.rs",
        ),
        // Type::method assoc-fn, bare (no leading kw).
        (
            "`CudaBackendImpl::group_norm_f32 in ferrotorch-gpu/src/backend_impl.rs`",
            DeclKind::AssocFn,
            "group_norm_f32",
            "ferrotorch-gpu/src/backend_impl.rs",
        ),
        // Type::method assoc-fn with leading `pub fn`.
        (
            "`pub fn LlamaGpuInferencer::generate_masked in gpu_gguf.rs`",
            DeclKind::AssocFn,
            "generate_masked",
            "gpu_gguf.rs",
        ),
        // impl <Type>.
        (
            "`impl CusparseLtHandle in cusparselt.rs`",
            DeclKind::Impl,
            "CusparseLtHandle",
            "cusparselt.rs",
        ),
        // impl <Trait> for <Type> -> target is the Type.
        (
            "`impl Drop for CusparseLtHandle in cusparselt.rs`",
            DeclKind::Impl,
            "CusparseLtHandle",
            "cusparselt.rs",
        ),
        // generics stripped on the symbol.
        (
            "`pub struct Foo<T> in widget.rs`",
            DeclKind::Struct,
            "Foo",
            "widget.rs",
        ),
        // prefixed suffix path.
        (
            "`pub fn reduce_all in ops/meta_propagate.rs`",
            DeclKind::Fn,
            "reduce_all",
            "ops/meta_propagate.rs",
        ),
        // #1669 bare single-identifier anchors (no kw, no `::`) — the dominant
        // genuine corpus shape, now matched as `DeclKind::BareIdent`.
        (
            "`add in arithmetic.rs`",
            DeclKind::BareIdent,
            "add",
            "arithmetic.rs",
        ),
        (
            "`reduce_all in meta_propagate.rs`",
            DeclKind::BareIdent,
            "reduce_all",
            "meta_propagate.rs",
        ),
        (
            "`addcmul_t in methods.rs`",
            DeclKind::BareIdent,
            "addcmul_t",
            "methods.rs",
        ),
        (
            "`add_f32 in backend_impl.rs`",
            DeclKind::BareIdent,
            "add_f32",
            "backend_impl.rs",
        ),
        // leading-underscore ident is a legal Rust identifier.
        (
            "`_private_helper in util.rs`",
            DeclKind::BareIdent,
            "_private_helper",
            "util.rs",
        ),
    ];
    for (line, kind, ident, file) in good {
        let anchors = parse_symbol_anchors(line);
        assert_eq!(
            anchors.len(),
            1,
            "expected exactly 1 anchor from `{line}`, got {anchors:?}"
        );
        assert_eq!(&anchors[0].kind, kind, "kind mismatch for `{line}`");
        assert_eq!(&anchors[0].ident, ident, "ident mismatch for `{line}`");
        assert_eq!(
            &anchors[0].file_as_written, file,
            "file mismatch for `{line}`"
        );
    }

    // #1669: a bare single lowercase ident IS now matched (the dominant
    // genuine corpus shape, validated against a real declaration / re-export).
    // PRECISION is preserved by requiring EXACTLY ONE identifier token before
    // " in " — multi-word prose ("the model in foo.rs", two tokens) is still
    // rejected (see the `bad` list below).

    // NON-anchors that MUST NOT parse.
    let bad = [
        // prose: "the model in foo.rs is ..." — TWO words before " in " (`the
        // model`), so the bare-ident matcher (which requires exactly one
        // token) rejects it. This is the precision boundary for #1669.
        "the model in foo.rs is loaded",
        "`the model in foo.rs`",
        // multi-word prose inside backticks — two tokens before " in ".
        "`the cached model in cache.rs`",
        "`every public helper in util.rs`",
        // upstream cites must be excluded.
        "`fn add in /home/doll/pytorch/aten/foo.rs`",
        "`reduce_all in aten/src/ATen/native/foo.rs`",
        "`fn x in torch/csrc/foo.rs`",
        // the #1643 CROSS-span form: the `struct` connective lives in a
        // SEPARATE backtick span, so the single-span content `RsqrtBackward`
        // (alone) and the file (alone) never form a `<decl> in <path>.rs`.
        "the `RsqrtBackward` struct in `grad_fns/arithmetic.rs` saving",
        // uppercase-leading bare ident with no keyword -> NOT a bare-ident
        // anchor (CamelCase bare names are the #1643 struct family or prose,
        // handled by the cross-span / keyword paths, not the bare-ident path).
        "`Foo in bar.rs`",
        "`RsqrtBackward in arithmetic.rs`",
        // non-.rs file.
        "`pub fn foo in bar.py`",
        "`pub fn foo in derivatives.yaml`",
        "`add in arithmetic.py`",
        // `::` but lowercase-leading "type" -> not a real Type::method.
        "`foo::bar in x.rs`",
        // no " in " separator at all.
        "`pub fn foo bar.rs`",
        // empty-ish.
        "`in x.rs`",
    ];
    for line in bad {
        let anchors = parse_symbol_anchors(line);
        assert!(
            anchors.is_empty(),
            "PRECISION FAILURE: parser matched a non-anchor `{line}` -> {anchors:?}"
        );
    }
}

/// #1669 PRECISION — the bare-ident VALIDATOR (not just the parser) must:
///   1. reclassify `<file-stem-or-dir-segment> in <file>.rs` PROSE
///      (`laplace in laplace.rs`, `gpu in gpu/unet.rs`) as `Prose`, NOT `Stale`
///      — these name a module/file by its own name in prose, not a symbol;
///   2. KEEP a file-stem-named ident that DOES resolve to a real decl as
///      `Valid` (the edge the dispatch calls out — only the not-declared
///      stem case is prose);
///   3. validate genuine bare anchors (`reduce_all in meta_propagate.rs`,
///      `add in grad_fns/arithmetic.rs`) as `Valid`;
///   4. resolve `examples/` and `tests/` anchors (no longer `Unresolvable`).
///
/// Runs against the REAL workspace tree; expected values are grep-derived
/// ground truth (R-CHAR-3(b)), not copied from validator output.
#[test]
fn bare_ident_validator_file_stem_prose_and_examples_tests() {
    let root = workspace_root();
    let index = build_src_index(&root);

    let mk = |kind, ident: &str, file: &str| SymbolAnchor {
        kind,
        ident: ident.to_string(),
        file_as_written: file.to_string(),
    };
    // Ground-truth declaration check via the SAME production validator path a
    // bare-ident anchor uses (`anchor_symbol_declared` on a BareIdent anchor).
    let bare_ident_declared = |src: &str, ident: &str| {
        anchor_symbol_declared(
            src,
            &SymbolAnchor {
                kind: DeclKind::BareIdent,
                ident: ident.to_string(),
                file_as_written: String::new(),
            },
        )
    };

    // 1. FILE-STEM PROSE: `laplace in laplace.rs` — `laplace` is the file stem
    //    and is NOT declared as a symbol in laplace.rs -> Prose (non-anchor).
    let laplace_src = root.join("ferrotorch-distributions/src/laplace.rs");
    assert!(laplace_src.exists(), "fixture: {laplace_src:?} must exist");
    assert!(
        !bare_ident_declared(&fs::read_to_string(&laplace_src).unwrap(), "laplace"),
        "ground truth: `laplace` is NOT a decl in laplace.rs (it's the module name)"
    );
    let (o, _) = validate_symbol_anchor(
        &mk(DeclKind::BareIdent, "laplace", "laplace.rs"),
        &index,
        &root,
        ".design/ferrotorch-distributions/laplace.md",
        1,
    );
    assert_eq!(
        o,
        AnchorOutcome::Prose,
        "`laplace in laplace.rs` (stem == ident, not a decl) must be PROSE, not STALE"
    );

    // 1b. DIR-SEGMENT PROSE: `gpu in gpu/unet.rs` — `gpu` is a directory
    //     segment of the path, used as a prose module reference -> Prose.
    let (o, _) = validate_symbol_anchor(
        &mk(DeclKind::BareIdent, "gpu", "gpu/unet.rs"),
        &index,
        &root,
        ".design/ferrotorch-diffusion/gpu/unet.md",
        1,
    );
    assert_eq!(
        o,
        AnchorOutcome::Prose,
        "`gpu in gpu/unet.rs` (gpu == dir segment, not a decl) must be PROSE"
    );

    // 2. FILE-STEM EDGE that RESOLVES: pick a file whose stem IS a real decl.
    //    `dtype.rs` declares `pub fn dtype` is unlikely; use a known case:
    //    `device in device.rs` — verify behaviour matches ground truth. If
    //    `device` is declared (e.g. `pub fn device`) it must be Valid; if not,
    //    Prose. We assert the validator AGREES with the grep ground truth.
    let device_src = root.join("ferrotorch-core/src/device.rs");
    let device_declared = bare_ident_declared(&fs::read_to_string(&device_src).unwrap(), "device");
    let (o, _) = validate_symbol_anchor(
        &mk(DeclKind::BareIdent, "device", "device.rs"),
        &index,
        &root,
        ".design/ferrotorch-core/device.md",
        1,
    );
    if device_declared {
        assert_eq!(
            o,
            AnchorOutcome::Valid,
            "stem ident that IS a decl -> Valid"
        );
    } else {
        assert_eq!(
            o,
            AnchorOutcome::Prose,
            "stem ident that is NOT a decl -> Prose (module-name prose)"
        );
    }

    // 3. GENUINE bare anchors resolve Valid.
    let mp_src = root.join("ferrotorch-core/src/meta_propagate.rs");
    assert!(
        bare_ident_declared(&fs::read_to_string(&mp_src).unwrap(), "reduce_all"),
        "ground truth: `reduce_all` IS declared in meta_propagate.rs"
    );
    let (o, _) = validate_symbol_anchor(
        &mk(DeclKind::BareIdent, "reduce_all", "meta_propagate.rs"),
        &index,
        &root,
        ".design/ferrotorch-core/meta_propagate.md",
        1,
    );
    assert_eq!(
        o,
        AnchorOutcome::Valid,
        "`reduce_all in meta_propagate.rs` is a genuine anchor -> Valid"
    );
    let arith_src = root.join("ferrotorch-core/src/grad_fns/arithmetic.rs");
    assert!(
        bare_ident_declared(&fs::read_to_string(&arith_src).unwrap(), "add"),
        "ground truth: `add` IS declared in grad_fns/arithmetic.rs"
    );
    let (o, _) = validate_symbol_anchor(
        &mk(DeclKind::BareIdent, "add", "grad_fns/arithmetic.rs"),
        &index,
        &root,
        ".design/ferrotorch-core/grad_fns/arithmetic.md",
        1,
    );
    assert_eq!(
        o,
        AnchorOutcome::Valid,
        "`add in grad_fns/arithmetic.rs` is a genuine anchor -> Valid"
    );

    // 4. EXAMPLES/TESTS indexing: an anchor naming an examples/ or tests/ file
    //    must RESOLVE (candidate set non-empty), never Unresolvable. Pick a
    //    real example file present in the tree.
    let examples_in_index = index
        .values()
        .flatten()
        .any(|p| p.to_string_lossy().contains("/examples/"));
    assert!(
        examples_in_index,
        "the #1669 file index must include `*/examples/**/*.rs` files"
    );
    let tests_in_index = index
        .values()
        .flatten()
        .any(|p| p.to_string_lossy().contains("/tests/"));
    assert!(
        tests_in_index,
        "the #1669 file index must include `*/tests/**/*.rs` files"
    );
    // A bare anchor naming an example file resolves to >=1 candidate (so it is
    // never reported Unresolvable purely for being outside `src/`).
    let example_basename = index
        .iter()
        .find(|(_, ps)| {
            ps.iter()
                .all(|p| p.to_string_lossy().contains("/examples/"))
        })
        .map(|(name, _)| name.clone());
    if let Some(name) = example_basename {
        let cands = resolve_symbol_anchor_files(&index, &root, "synthetic", &name);
        assert!(
            !cands.is_empty(),
            "an example-only basename `{name}` must resolve to >=1 candidate (examples/ indexed)"
        );
    }
}

/// #1669 (final) LOAD-BEARING precision pin for the bare-ident gate adoption.
///
/// Proves, on the REAL workspace tree (grep-derived ground truth, R-CHAR-3(b)),
/// that the bare-ident `Stale`→`Prose` reclassification is SOUND and the gate
/// stays load-bearing:
///   1. a KEYWORD-LED anchor naming a real symbol in the WRONG file is `Stale`
///      (so the keyword-led gate is NOT vacuous — genuine declaration drift is
///      caught);
///   2. a BARE-IDENT consumer/call-site citation (file exists, ident NOT
///      declared there) is `Prose` (NOT `Stale`) — the corpus-grounded #1669
///      rule. Uses a real residual line: `` `gpu_matmul_f32 in backend_impl.rs` ``
///      where `gpu_matmul_f32` lives in `blas.rs` and `backend_impl.rs` is the
///      `CudaBackendImpl` consumer;
///   3. a NONEXISTENT file is `Unresolvable` for BOTH a bare-ident AND a
///      keyword-led anchor (the typo/deleted-file gate stays live for all kinds);
///   4. the file-stem prose sub-case ([`ident_is_path_component`]) still folds
///      into `Prose`.
#[test]
fn bare_ident_stale_is_zero_and_keyword_led_drift_is_gated() {
    let root = workspace_root();
    let index = build_src_index(&root);
    let mk = |kind, ident: &str, file: &str| SymbolAnchor {
        kind,
        ident: ident.to_string(),
        file_as_written: file.to_string(),
    };

    // Ground truth: gpu_matmul_f32 is declared in blas.rs, NOT in backend_impl.rs.
    let blas = root.join("ferrotorch-gpu/src/blas.rs");
    assert!(
        anchor_symbol_declared(
            &fs::read_to_string(&blas).unwrap(),
            &mk(DeclKind::Fn, "gpu_matmul_f32", "blas.rs"),
        ),
        "ground truth: gpu_matmul_f32 IS declared in ferrotorch-gpu/src/blas.rs"
    );
    let backend = root.join("ferrotorch-gpu/src/backend_impl.rs");
    assert!(
        !anchor_symbol_declared(
            &fs::read_to_string(&backend).unwrap(),
            &mk(DeclKind::BareIdent, "gpu_matmul_f32", "backend_impl.rs"),
        ),
        "ground truth: gpu_matmul_f32 is NOT declared in backend_impl.rs (it is the consumer)"
    );

    // 1. KEYWORD-LED wrong-file => STALE (gate is load-bearing).
    let (o, msg) = validate_symbol_anchor(
        &mk(DeclKind::Fn, "gpu_matmul_f32", "backend_impl.rs"),
        &index,
        &root,
        ".design/ferrotorch-gpu/blas.md",
        1,
    );
    assert_eq!(
        o,
        AnchorOutcome::Stale,
        "keyword-led `fn gpu_matmul_f32 in backend_impl.rs` (real fn in WRONG file) MUST be Stale"
    );
    assert!(msg.is_some_and(|m| m.contains("STALE")));

    // 2. BARE-IDENT consumer citation => PROSE (NOT Stale) — the #1669 rule.
    let (o, msg) = validate_symbol_anchor(
        &mk(DeclKind::BareIdent, "gpu_matmul_f32", "backend_impl.rs"),
        &index,
        &root,
        ".design/ferrotorch-gpu/blas.md",
        1,
    );
    assert_eq!(
        o,
        AnchorOutcome::Prose,
        "bare `gpu_matmul_f32 in backend_impl.rs` (consumer/call-site citation) MUST be Prose, not Stale"
    );
    assert!(msg.is_none(), "Prose carries no failure diagnostic");

    // 2b. A bare-ident that IS declared in the cited file stays VALID (the gate
    //     is not "all bare-idents are prose"): gpu_matmul_f32 in blas.rs.
    let (o, _) = validate_symbol_anchor(
        &mk(DeclKind::BareIdent, "gpu_matmul_f32", "blas.rs"),
        &index,
        &root,
        ".design/ferrotorch-gpu/blas.md",
        1,
    );
    assert_eq!(
        o,
        AnchorOutcome::Valid,
        "bare `gpu_matmul_f32 in blas.rs` (declared there) is a genuine anchor -> Valid"
    );

    // 3. NONEXISTENT file => Unresolvable for BOTH kinds (typo/deleted gate).
    for kind in [DeclKind::BareIdent, DeclKind::Fn] {
        let (o, _) = validate_symbol_anchor(
            &mk(kind.clone(), "whatever", "this_file_does_not_exist_xyz.rs"),
            &index,
            &root,
            ".design/ferrotorch-gpu/blas.md",
            1,
        );
        assert_eq!(
            o,
            AnchorOutcome::Unresolvable,
            "a nonexistent file must be Unresolvable (gated) for kind {kind:?}"
        );
    }

    // 4. file-stem prose sub-case still folds into Prose (and the helper that
    //    recognises it is exercised).
    assert!(
        ident_is_path_component("gpu", "gpu/unet.rs"),
        "ident_is_path_component must recognise the dir-segment prose case"
    );
    assert!(
        !ident_is_path_component("gpu_matmul_f32", "blas.rs"),
        "a distinctive multi-segment symbol is NOT a path component"
    );
}

/// #1668 HARD CONTRACT: every single-span symbol anchor
/// `` `<decl> in <path>.rs` `` across ALL `.design/**/*.md` resolves to a
/// genuine declaration of that symbol in a file of the named basename at
/// HEAD. A renamed/moved/deleted symbol (STALE) or a typo'd / nonexistent
/// file (UNRESOLVABLE) fails this test.
///
/// Disambiguation: see the module-level note above
/// [`resolve_symbol_anchor_files`] — an anchor is VALID iff the symbol is
/// declared in the DOC-CRATE's same-basename file (#1669 crate-disambiguation),
/// or — when the doc maps to no crate / the crate has no such file — in AT
/// LEAST ONE cross-crate candidate of the named basename.
///
/// SCOPE (#1669, FLAW 2): this HARD contract enforces the keyword-led /
/// `Type::method` anchor kinds (`Fn`, `Struct`, `Enum`, `Trait`, `Mod`,
/// `Const`, `Type`, `Impl`, `AssocFn`) AND the bare single-identifier form
/// (`DeclKind::BareIdent`, e.g. `add in arithmetic.rs`) — the latter is now
/// ADOPTED into the gate, but at the precise soundness boundary the #1669
/// tightening achieves with zero false-positives.
///
/// ALL anchor kinds (keyword-led + bare-ident): a `Valid` resolution passes;
/// an `Unresolvable` outcome (the named `.rs` file exists NOWHERE under any
/// `*/src/**`, `*/examples/**`, or `*/tests/**` in the workspace — an
/// unambiguous typo / deleted-file) FAILS the gate.
///
/// Keyword-led / `Type::method` kinds: a `Stale` outcome (file resolves, symbol
/// absent) ALSO fails — those forms are precise (an explicit
/// `fn`/`struct`/`Type::method` decl-shape), so a `Stale` there is genuine drift.
///
/// Bare-ident kind (#1669 final — FULLY GATED, ceiling removed): the bare
/// single-identifier form is a DECLARATION anchor only when the ident IS
/// declared in a candidate file of the named basename (`Valid`). RE-RUNNING
/// report mode + reading every one of the 291 residual bare-ident `Stale` lines
/// verbatim established that NONE were declaration drift — all were consumer
/// citations / call-sites / local-variable mentions / op-usages (see the
/// [`AnchorOutcome::Prose`] doc for the worked examples). Re-pointing those to
/// the symbol's declaration file would have injected FALSE claims and erased
/// the genuine consumer evidence they encode, so the SOUND fix was a precision
/// RE-CLASSIFICATION, not a doc edit: a bare-ident whose named file EXISTS but
/// does not declare the ident is now `Prose` (a prose reference, not a gated
/// declaration claim). That drives the bare-ident `Stale` count to ZERO by
/// construction. The bare-ident kind is therefore now ADOPTED into the hard
/// gate at its full sound boundary — `Valid` passes, `Prose` passes,
/// `Unresolvable` (nonexistent file — typo/deleted) FAILS, and `Stale` is
/// structurally unreachable. The prior `<=360` escalation ceiling is REMOVED:
/// any future bare-ident `Stale` (which would require the validator's
/// reclassification rule to regress) fails this gate hard.
///
/// LOAD-BEARING coverage retained: the precision pins in
/// `single_span_anchor_parser_is_precise`,
/// `bare_ident_validator_file_stem_prose_and_examples_tests`, and the new
/// `bare_ident_stale_is_zero_and_keyword_led_drift_is_gated` test prove (a) a
/// keyword-led wrong-file IS caught as `Stale` (so the keyword gate stays
/// load-bearing), (b) a bare consumer-citation is `Prose`, and (c) a
/// nonexistent file is `Unresolvable` for both kinds. The bare-ident matcher +
/// validator + crate-disambiguating resolver are further proven by the
/// `divergence_single_span_anchor_resolver.rs` pins.
#[test]
fn all_design_docs_single_span_anchors_resolve_at_head() {
    let root = workspace_root();
    let index = build_src_index(&root);
    let docs = collect_design_docs(&root);
    assert!(
        !docs.is_empty(),
        "expected at least one .md file under .design/"
    );
    let mut failures: Vec<String> = Vec::new();
    let mut total = 0usize;
    let mut bare_ident_seen = 0usize;
    let mut bare_ident_stale = 0usize;
    for doc in &docs {
        let text = match fs::read_to_string(root.join(doc)) {
            Ok(t) => t,
            Err(_) => continue,
        };
        for (i, line) in text.lines().enumerate() {
            for anchor in parse_symbol_anchors(line) {
                let is_bare = anchor.kind == DeclKind::BareIdent;
                if is_bare {
                    bare_ident_seen += 1;
                } else {
                    total += 1;
                }
                let (outcome, msg) = validate_symbol_anchor(&anchor, &index, &root, doc, i + 1);
                match outcome {
                    AnchorOutcome::Valid | AnchorOutcome::Prose => {}
                    AnchorOutcome::Unresolvable => {
                        // Unambiguous typo / deleted file — GATED for ALL kinds
                        // (bare-ident included: zero false-positives, since the
                        // file genuinely exists nowhere in the workspace).
                        if let Some(m) = msg {
                            failures.push(m);
                        }
                    }
                    AnchorOutcome::Stale => {
                        if is_bare {
                            // #1669 final: bare-ident `Stale` is now structurally
                            // unreachable (a not-declared bare-ident reclassifies
                            // to `Prose`). Count it AND fail the gate if it ever
                            // recurs — the reclassification rule has regressed.
                            bare_ident_stale += 1;
                        }
                        if let Some(m) = msg {
                            // Keyword-led Stale IS gated (precise decl-shape);
                            // bare-ident Stale (if it ever recurs) is gated too.
                            failures.push(m);
                        }
                    }
                }
            }
        }
    }
    // Sanity floor: the corpus has thousands of these; if the count collapses
    // the parser has silently stopped matching the keyword-led single-span
    // form.
    assert!(
        total >= 500,
        "expected >=500 keyword-led single-span symbol anchors across .design/, found {total} — has the parser stopped matching the `<decl> in <path>.rs` form?",
    );
    // The #1669 bare-ident matcher is load-bearing: it must keep finding the
    // dominant bare form (so the bare-ident gate stays meaningful).
    assert!(
        bare_ident_seen >= 500,
        "expected the #1669 bare-ident matcher to find >=500 bare `<ident> in <path>.rs` anchors, found {bare_ident_seen} — has the bare-ident parser regressed?",
    );
    // #1669 final: bare-ident `Stale` is fully adopted into the gate and MUST be
    // ZERO. The `<=360` escalation ceiling is removed; any bare-ident `Stale`
    // means the prose-reclassification rule regressed and is a hard failure
    // (it is also folded into `failures` above so the assert below catches it).
    assert_eq!(
        bare_ident_stale, 0,
        "bare-ident STALE must be 0 (#1669 final): a not-declared bare-ident reclassifies to Prose; a nonzero count means the reclassification rule regressed.",
    );
    assert!(
        failures.is_empty(),
        "{n} single-span symbol anchor(s) are STALE (keyword-led + bare-ident) / UNRESOLVABLE (any kind) out of {total} keyword-led + {bare_ident_seen} bare-ident scanned (#1668/#1669 crate-disambiguating resolver / goal.md S3 R-CITE-2b):\n\n{body}",
        n = failures.len(),
        body = failures.join("\n\n"),
    );
}

/// CORE-207 / #1901 ignore-reason hygiene gate.
///
/// Scans every `.rs` file under `ferrotorch-core/tests/` (recursively) for
/// `#[ignore]` attributes and requires each reason string to carry either
/// a tracking-issue reference (`#<digits>`, e.g. `tracking #1617`) or the
/// explicit `diagnostic:` marker (a sanctioned operator-run lane that gates
/// nothing — the marker names the intent so R-VERIFY-1 audits can tell a
/// tracked divergence from a report-mode tool). A bare `#[ignore]` with no
/// reason at all is always a violation: it is exactly the untracked-skip
/// pattern CORE-207 flagged.
///
/// Detection is line-anchored: rustfmt (enforced crate-wide via
/// `cargo fmt --check` in CI) places attributes on their own line, so an
/// attribute occurrence is a line whose trimmed text starts with
/// `#[ignore`. Prose mentions in comments (`// NOT #[ignore]'d`) and
/// string literals never start a trimmed line with that token in this
/// corpus, and any new violation that tried to hide mid-line would be
/// reformatted onto its own line by the fmt gate before it could land.
/// Multi-line attributes are handled by consuming lines until the
/// bracket depth returns to zero.
#[test]
fn ignore_reasons_carry_issue_ref_or_diagnostic_marker() {
    // Tracked baseline exemptions — each entry is (file basename, exact
    // reason substring) and MUST cite an open issue that tracks removing
    // it. The gate still fails on any NEW untracked ignore.
    //
    // #1929: conformance_autograd_parity.rs predates this gate and was
    // out of scope for the #1901 dispatch; the issue tracks rewording its
    // reason (or attaching the #1171 ref) and deleting this entry.
    const BASELINE: &[(&str, &str)] = &[(
        "conformance_autograd_parity.rs",
        "network-aware real-artifact harness; run with --ignored",
    )];

    let tests_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests");
    let mut rs_files: Vec<PathBuf> = Vec::new();
    let mut stack = vec![tests_dir.clone()];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir).expect("read_dir tests/") {
            let path = entry.expect("dir entry").path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                rs_files.push(path);
            }
        }
    }
    assert!(
        rs_files.len() >= 50,
        "sanity floor: expected >=50 .rs files under ferrotorch-core/tests/, found {} — has the walker broken?",
        rs_files.len()
    );
    rs_files.sort();

    let issue_ref = |reason: &str| -> bool {
        // `#` immediately followed by at least one ASCII digit, anywhere
        // in the reason (covers `#1617`, `local #1543`,
        // `forecast-bio/ferrotorch#25`).
        reason
            .as_bytes()
            .windows(2)
            .any(|w| w[0] == b'#' && w[1].is_ascii_digit())
    };

    let mut violations: Vec<String> = Vec::new();
    let mut attrs_seen = 0usize;
    for path in &rs_files {
        let text = fs::read_to_string(path).expect("read test file");
        let basename = path
            .file_name()
            .and_then(|n| n.to_str())
            .expect("utf8 basename")
            .to_string();
        let lines: Vec<&str> = text.lines().collect();
        let mut i = 0usize;
        while i < lines.len() {
            let trimmed = lines[i].trim_start();
            if !trimmed.starts_with("#[ignore") {
                i += 1;
                continue;
            }
            attrs_seen += 1;
            let lineno = i + 1;
            // Collect the full attribute text (multi-line safe): consume
            // until bracket depth closes.
            let mut attr = String::new();
            let mut depth = 0i32;
            loop {
                let l = if attr.is_empty() {
                    lines[i].trim_start()
                } else {
                    lines[i]
                };
                attr.push_str(l);
                attr.push('\n');
                depth += l.matches('[').count() as i32 - l.matches(']').count() as i32;
                if depth <= 0 || i + 1 >= lines.len() {
                    break;
                }
                i += 1;
            }
            let ok = match attr.find('=') {
                // Bare `#[ignore]` — no reason string at all.
                None => false,
                Some(_) => attr.contains("diagnostic:") || issue_ref(&attr),
            };
            let baselined = BASELINE
                .iter()
                .any(|(file, reason)| *file == basename && attr.contains(reason));
            if !ok && !baselined {
                violations.push(format!(
                    "{basename}:{lineno}: #[ignore] reason carries neither a '#<digits>' \
                     tracking-issue reference nor the 'diagnostic:' marker:\n    {}",
                    attr.trim_end()
                ));
            }
            i += 1;
        }
    }
    // Sanity floor: zero matches means the line-anchored detector has
    // silently stopped seeing attributes. The floor is deliberately >= 1,
    // NOT a population count: a count-based floor (originally >= 8) broke
    // the first time legitimate ignores were retired (#1899 retired the
    // stale mkl ignore and the suite went red repo-wide — crosslink
    // #1934). Retiring ignores is the SUCCESS path of this gate; the
    // floor must only catch detector breakage, never progress.
    assert!(
        attrs_seen >= 1,
        "sanity floor: expected at least one #[ignore] attribute across \
         ferrotorch-core/tests/, found {attrs_seen} — either the detector \
         broke, or the corpus genuinely reached zero ignores (celebrate, \
         then update this floor)"
    );
    assert!(
        violations.is_empty(),
        "{} untracked #[ignore] attribute(s) (CORE-207 / #1901): every ignore reason must \
         contain '#<digits>' (tracking issue) or 'diagnostic:' (operator-run report lane), \
         or be a baseline entry citing an open issue:\n\n{}",
        violations.len(),
        violations.join("\n\n")
    );
}
