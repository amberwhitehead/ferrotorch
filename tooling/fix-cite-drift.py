#!/usr/bin/env python3
"""Repo-wide cite-drift fixer for ferrotorch (#1592).

The companion test `ferrotorch-core/tests/divergence_cite_drift_generic.rs`
parses every backtick-quoted `<path>.rs:<N>(-<M>)?` cite (plus bare-colon
continuations and bare-paren cites) out of every `.design/**/*.md` doc and the
`//!` doc-comments of two source files, then asserts each cite resolves to
substantive content at the symbol the surrounding prose names.

Per goal.md S3 / R-CITE-2b, target-side (`.design/`) cites MUST use SYMBOL
ANCHORS, never line numbers: `pub fn add_scaled in arithmetic.rs`. A cite with
no `:<N>` is not parsed by the test at all, so symbol anchors both pass the
audit and never re-drift.

This script therefore rewrites each stale parseable cite
    `<path>.rs:<N>(-<M>)?`
into a durable symbol anchor
    `<symbol> in <path>.rs`            (when a real symbol is recoverable)
or, when no symbol can be recovered, simply drops the line spec
    `<path>.rs`                        (still names the file; still passes).

PROTECTED cites are exempted from the symbol-anchor rewrite and instead get an
ACCURATE line number, because sibling tests audit them as `file:line`:
  - `tools/parity-sweep/runner/src/main.rs:<L>` cites in arithmetic.md's
    REQ-8..REQ-15 SHIPPED rows  (divergence_addcmul_req15_runner_cite_shift)
  - the REQ-9 row's arithmetic.rs / methods.rs / main.rs cites
    (divergence_rsub_req9_stale_cites)
  - inplace.md must never cite `inplace.rs:248`
    (divergence_inplace_req7_prose_stale) — handled by symbol-anchoring, which
    drops the line entirely.

The cite parser below is a faithful Python port of the Rust test's parser so
the set of cites this script rewrites is exactly the set the test audits.
"""

from __future__ import annotations

import os
import re
import sys

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DESIGN = os.path.join(WORKSPACE, ".design")

STOPWORD_HINTS = {
    "at", "and", "via", "by", "in", "of", "for", "the", "is", "are", "to",
    "on", "with", "from", "as", "or", "an", "a", "calls", "cite", "cites",
    "see", "per", "consumer", "consumers", "uses", "used", "into", "after",
    "before", "between", "tests", "test", "row", "rows", "verified",
    "implementation", "impl", "section", "block", "lines", "line",
}

KNOWN_HELPERS = {
    "normalize_axis", "reverse_cumsum", "cummax_forward", "cummin_forward",
    "cumsum_forward", "cumprod_forward", "logcumsumexp_forward",
    "cummaxmin_backward_impl", "cumulative_scalar_identity",
    "cumextreme_scalar_identity",
}

TOO_GENERIC_TYPE = {
    "Tensor", "String", "Result", "Vec", "Option", "Float", "FerrotorchResult",
    "FerrotorchError", "GradFn", "Self", "Path", "PathBuf", "REQ", "AC",
}


# ---------------------------------------------------------------------------
# Cite path resolution (port of resolve_cite_path).
# ---------------------------------------------------------------------------
def resolve_cite_path(file_as_written: str):
    """Return absolute path str if Resolved, "EXTERNAL" if allowed-external,
    or None if unresolvable."""
    if file_as_written.startswith("/home/doll/pytorch/"):
        return "EXTERNAL"
    if "/" in file_as_written:
        candidates = [
            file_as_written,
            f"ferrotorch-core/src/{file_as_written}",
            f"ferrotorch-core/{file_as_written}",
        ]
    else:
        b = file_as_written
        candidates = [
            f"ferrotorch-core/src/grad_fns/{b}",
            f"ferrotorch-core/src/ops/{b}",
            f"ferrotorch-core/src/{b}",
            f"ferrotorch-core/src/autograd/{b}",
            f"ferrotorch-nn/src/{b}",
            f"ferrotorch-vision/src/{b}",
            f"tools/parity-sweep/runner/src/{b}",
        ]
    for c in candidates:
        p = os.path.join(WORKSPACE, c)
        if os.path.exists(p):
            return p
    return None


# ---------------------------------------------------------------------------
# Cite parser (faithful port of the Rust test).
# ---------------------------------------------------------------------------
class Cite:
    __slots__ = ("file_as_written", "line_start", "line_end", "symbol_hint",
                 "end_pos_in_line", "span_in_line", "kind", "enclosing")

    def __init__(self, file_as_written, lo, hi, symbol_hint, end_pos_in_line,
                 span_in_line, kind, enclosing=None):
        self.file_as_written = file_as_written
        self.line_start = lo
        self.line_end = hi
        self.symbol_hint = symbol_hint
        self.end_pos_in_line = end_pos_in_line
        # (start_byte, end_byte) of the entire `<path>:<spec>` token within the
        # ORIGINAL doc line (byte offsets). None for bare-paren/bare-colon
        # cites whose precise span we don't track for rewriting.
        self.span_in_line = span_in_line
        self.kind = kind  # 'named' | 'bare_colon' | 'paren'
        # For cites inside a backtick span: (content_start, content_end) byte
        # bounds of the ENCLOSING backtick span's CONTENT (exclusive of the
        # backticks). Lets the rewriter replace/collapse the whole span when
        # dropping the cite would otherwise leave an empty `` ``.
        self.enclosing = enclosing


def parse_line_range(s: str):
    m = re.match(r"(\d+)", s)
    if not m:
        return None
    lo = int(m.group(1))
    rest = s[m.end():]
    if rest.startswith("-"):
        m2 = re.match(r"(\d+)", rest[1:])
        if m2:
            hi = int(m2.group(1))
            if hi >= lo:
                return (lo, hi)
        return (lo, lo)
    return (lo, lo)


def parse_any_named_cite(tok: str):
    """Return (file_or_None, lo, hi) or None.  Mirrors the Rust fn."""
    colon = tok.find(":")
    if colon < 0:
        return None
    file_part = tok[:colon]
    dot = file_part.rfind(".")
    if dot < 0:
        return None
    ext = file_part[dot + 1:]
    if not ext or len(ext) > 5 or not all("a" <= c <= "z" for c in ext):
        return None
    basename = file_part.rsplit("/", 1)[-1]
    stem = basename[: len(basename) - (len(ext) + 1)]
    if not stem or not all(c.isalnum() or c == "_" for c in stem):
        return None
    after = tok[colon + 1:]
    rng = parse_line_range(after)
    if rng is None:
        return None
    lo, hi = rng
    path = file_part if ext == "rs" else None
    return (path, lo, hi)


def extract_symbol_hint(line: str, cite_start_in_line: int):
    """Port of the Rust extract_symbol_hint (operates on bytes/char offsets;
    we use the str directly since the docs are ASCII-dominant and the Rust
    code is char-boundary safe)."""
    window_start = max(0, cite_start_in_line - 80)
    context = line[window_start:cite_start_in_line]
    if context is None:
        return None

    # Pattern 1: explicit declaration markers.
    for marker in ("pub fn ", "fn ", "pub struct ", "struct "):
        pos = context.rfind(marker)
        if pos >= 0:
            after = context[pos + len(marker):]
            ident = ""
            for c in after:
                if c.isalnum() or c == "_":
                    ident += c
                else:
                    break
            if ident and ident not in STOPWORD_HINTS:
                return ident

    # Pattern 2: classify backticks as open/close and pair from the left.
    classified = []
    for i, c in enumerate(context):
        if c != "`":
            continue
        prev = " " if i == 0 else context[i - 1]
        if prev in " (,;=-/\t\n*[{":
            classified.append((i, "open"))
        else:
            classified.append((i, "close"))
    backticks = []
    pending = None
    for pos, kind in classified:
        if kind == "open":
            if pending is None:
                pending = pos
        else:
            if pending is not None:
                backticks.append(pending)
                backticks.append(pos)
                pending = None
    last_accepted = None
    context_len = len(context)
    k = 0
    while k + 1 < len(backticks):
        open_i = backticks[k]
        close_i = backticks[k + 1]
        bt = context[open_i + 1: close_i]
        last_segment = bt.rsplit("::", 1)[-1]
        ident = ""
        for c in last_segment:
            if c.isalnum() or c == "_":
                ident += c
            else:
                break
        dist = context_len - close_i
        close_to_cite = dist <= 15
        if ident and ident not in STOPWORD_HINTS:
            starts_upper = ident[0].isupper()
            is_test_fn = ident.startswith("test_")
            is_known_helper = ident in KNOWN_HELPERS
            ends_backward = ident.endswith("Backward")
            too_generic = ident in TOO_GENERIC_TYPE
            is_method_t = ident.endswith("_t") and all(
                ("a" <= c <= "z") or c == "_" for c in ident
            )
            if close_to_cite and not too_generic and (
                is_test_fn or is_known_helper or ends_backward
                or is_method_t or starts_upper
            ):
                last_accepted = ident
        k += 2
    return last_accepted


def scan_line(line: str, last_rs_file, prev_tail: str):
    """Return (list_of_cites, new_last_rs_file).  Faithful-enough port.

    Cites carry an absolute (start,end) byte span in `line` when they are
    named cites inside a backtick span (the only kind we rewrite). Bare-colon
    and bare-paren cites get span=None (we don't rewrite them in place; they
    are rare in practice and re-resolve once the preceding named cite is
    fixed)."""
    prefix_len = len(prev_tail)
    joined = prev_tail + line
    cites = []

    # Pass 1: backtick spans.
    backtick_ranges = []
    i = 0
    n = len(line)
    while i < n:
        if line[i] == "`":
            start = i + 1
            e = line.find("`", start)
            if e < 0:
                break
            end = e
            backtick_ranges.append((start, end))
            span = line[start:end]
            last_rs_file = scan_span(span, cites, start, end, last_rs_file,
                                     joined, prefix_len, line)
            i = end + 1
        else:
            i += 1

    # Cross-span post-cite hint promotion (#1270).
    promote_cross_span_hints(line, backtick_ranges, cites)

    # Pass 2: bare-paren cites outside backtick spans.
    j = 0
    while j < n:
        if line[j] == "(" and not _in_ranges(j, backtick_ranges):
            start = j + 1
            end = _matching_close(line, start)
            if end is None:
                j += 1
                continue
            overlaps = any(not (bte < start or bts > end)
                           for (bts, bte) in backtick_ranges)
            if overlaps:
                j = end + 1
                continue
            span = line[start:end].strip()
            lo_hi, had_colon = parse_paren_cite(span)
            if lo_hi is not None:
                lo, hi = lo_hi
                if (had_colon or lo >= 100) and last_rs_file is not None:
                    sym = extract_symbol_hint(joined, start + prefix_len)
                    # span_in_line for a paren cite covers the WHOLE `(...)`
                    # (open paren j .. close paren end inclusive) so the
                    # rewriter can drop the parenthetical line number cleanly.
                    cites.append(Cite(last_rs_file, lo, hi, sym, 0,
                                      (j, end + 1), "paren"))
            j = end + 1
        else:
            j += 1

    return cites, last_rs_file


def promote_cross_span_hints(line, backtick_ranges, cites):
    """Port of the Rust promote_cross_span_hints: for each cite emitted on
    this line (end_pos_in_line>0), if a backtick span opens within 20 chars
    after the cite-end and holds a `test_*` identifier, promote it."""
    for cite in cites:
        if cite.end_pos_in_line == 0:
            continue
        cite_end = cite.end_pos_in_line
        chosen = None
        for (bs, be) in backtick_ranges:
            if bs == 0:
                continue
            open_pos = bs - 1
            if open_pos <= cite_end:
                continue
            if open_pos - cite_end > 20:
                continue
            chosen = (bs, be)
            break
        if chosen is None:
            continue
        bs, be = chosen
        candidate_span = line[bs:be]
        if ":" in candidate_span or "." in candidate_span:
            continue
        first = ""
        for piece in re.split(r"[, ]", candidate_span):
            piece = piece.strip()
            if piece:
                first = piece
                break
        cleaned = first.strip("*();.")
        if not cleaned or not all(c.isalnum() or c == "_" for c in cleaned):
            continue
        if not cleaned.startswith("test_"):
            continue
        if cleaned in STOPWORD_HINTS:
            continue
        prev = cite.symbol_hint
        if prev is None or not prev.startswith("test_"):
            cite.symbol_hint = cleaned


def _in_ranges(pos, ranges):
    return any(s <= pos < e for (s, e) in ranges)


def _matching_close(line, start_after_open):
    depth = 1
    k = start_after_open
    n = len(line)
    while k < n:
        c = line[k]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return k
        k += 1
    return None


def parse_paren_cite(s):
    had_colon = s.startswith(":")
    stripped = s[1:] if had_colon else s
    if not stripped or not all(c.isdigit() or c == "-" for c in stripped):
        return None, had_colon
    return parse_line_range(stripped), had_colon


def scan_span(span, cites, span_offset, span_end, last_rs_file, joined,
              prefix_len, line):
    span_local_file = last_rs_file
    tokens = [t.strip() for t in re.split(r"[, ]", span) if t.strip()]
    # Track (token_index -> index into `cites`) for emitted cites so the
    # post-cite test_* promoter can look at the following token.
    emitted = []  # list of (tok_idx, cite_index)
    search_from = [span_offset]

    def tok_span(tok):
        ts = line.find(tok, search_from[0], span_end + 1)
        if ts < 0:
            ts = line.find(tok, span_offset)
        if ts >= 0:
            search_from[0] = ts + len(tok)
            return (ts, ts + len(tok))
        return None

    for idx, tok in enumerate(tokens):
        named = parse_any_named_cite(tok)
        if named is not None:
            file_or_none, lo, hi = named
            if file_or_none is not None:
                span_local_file = file_or_none
                last_rs_file = file_or_none
                sym = extract_symbol_hint(joined, span_offset + prefix_len)
                cites.append(Cite(file_or_none, lo, hi, sym, span_end,
                                  tok_span(tok), "named",
                                  (span_offset, span_end)))
                emitted.append((idx, len(cites) - 1))
            else:
                span_local_file = None
                last_rs_file = None
            continue
        bc = parse_bare_colon(tok)
        if bc is not None and span_local_file is not None:
            lo, hi = bc
            sym = extract_symbol_hint(joined, span_offset + prefix_len)
            cites.append(Cite(span_local_file, lo, hi, sym, span_end,
                              tok_span(tok), "bare_colon",
                              (span_offset, span_end)))
            emitted.append((idx, len(cites) - 1))

    # In-span post-cite test_* hint promotion (#1270).
    for tok_idx, cite_index in emitted:
        nxt = tok_idx + 1
        if nxt >= len(tokens):
            continue
        next_tok = tokens[nxt]
        if ":" in next_tok or "." in next_tok:
            continue
        cleaned = next_tok.strip("();.")
        if not cleaned or not all(c.isalnum() or c == "_" for c in cleaned):
            continue
        if not cleaned.startswith("test_"):
            continue
        if cleaned in STOPWORD_HINTS:
            continue
        prev = cites[cite_index].symbol_hint
        if prev is None or not prev.startswith("test_"):
            cites[cite_index].symbol_hint = cleaned
    return last_rs_file


def parse_bare_colon(tok):
    if not tok.startswith(":"):
        return None
    return parse_line_range(tok[1:])


def audit_doc_cites(text):
    """Return list of (doc_line_no, Cite) for every cite the test parses."""
    out = []
    last_rs = None
    prev_tail = ""
    for idx, line in enumerate(text.splitlines()):
        cites, last_rs = scan_line(line, last_rs, prev_tail)
        for c in cites:
            out.append((idx + 1, c))
        if len(line) <= 80:
            prev_tail = line
        else:
            prev_tail = line[len(line) - 80:]
    return out


# ---------------------------------------------------------------------------
# Validation (port of validate_cite) — used to decide which cites are stale.
# ---------------------------------------------------------------------------
def is_substantive(line):
    t = line.strip()
    if not t:
        return False
    if t in ("}", "{", "},", "});", "});,", "})"):
        return False
    if t.startswith("//!"):
        return False
    if t.startswith("//") and not t.startswith("///"):
        return False
    if t == "*" or t == "*/" or t.startswith("*/"):
        return False
    return True


def build_symbol_needles(symbol):
    out = [
        f"pub fn {symbol}", f"fn {symbol}", f"pub struct {symbol}",
        f"struct {symbol}", f"{symbol}(", f"pub fn {symbol}<",
    ]
    if symbol.endswith("_t"):
        stem = symbol[:-2]
        out += [f"{stem}(", f"::{stem}(", f"::{stem},", f"::{stem} ",
                f"::{stem})"]
    return out


def cite_is_stale(cite):
    """Return True if the test would flag this cite."""
    target = resolve_cite_path(cite.file_as_written)
    if target == "EXTERNAL":
        return False
    if target is None:
        return True
    try:
        with open(target, "r") as f:
            src_lines = f.read().splitlines()
    except OSError:
        return False
    total = len(src_lines)
    if cite.line_start == 0 or cite.line_end > total:
        return True
    is_range = cite.line_end > cite.line_start
    any_subst = False
    for ln in range(cite.line_start, cite.line_end + 1):
        if is_substantive(src_lines[ln - 1]):
            any_subst = True
            break
    if not any_subst and not is_range:
        lo = max(1, cite.line_start - 1)
        hi = min(total, cite.line_start + 1)
        for ln in range(lo, hi + 1):
            if is_substantive(src_lines[ln - 1]):
                any_subst = True
                break
    if not any_subst:
        return True
    if cite.symbol_hint:
        needles = build_symbol_needles(cite.symbol_hint)
        if is_range:
            wlo = max(1, cite.line_start - 3)
            whi = min(total, cite.line_end + 3)
        else:
            wlo = whi = cite.line_start
        found = False
        for ln in range(wlo, whi + 1):
            if any(nd in src_lines[ln - 1] for nd in needles):
                found = True
                break
        if not found:
            return True
    return False


# ---------------------------------------------------------------------------
# Symbol recovery: find the symbol the cite intends and its current line.
# ---------------------------------------------------------------------------
DEF_RE = re.compile(
    r"^\s*(?:pub(?:\([^)]*\))?\s+)?"
    r"(?:async\s+|const\s+|unsafe\s+|extern(?:\s+\"[^\"]*\")?\s+)*"
    r"(fn|struct|enum|trait|const|static|type|macro_rules!|impl|mod)\b"
)


def file_defines_symbol(target, symbol):
    """Does the resolved file declare `symbol` anywhere?  Used to validate a
    recovered symbol anchor actually exists (never fabricate)."""
    try:
        with open(target, "r") as f:
            src = f.read()
    except OSError:
        return False
    needles = build_symbol_needles(symbol)
    for ln in src.splitlines():
        if any(nd in ln for nd in needles):
            return True
    # Also accept a plain `mod <symbol>` / `enum <symbol>` / trait etc.
    if re.search(r"\b(?:mod|enum|trait|type|const|static)\s+" +
                 re.escape(symbol) + r"\b", src):
        return True
    return False


IDENT_BACKTICK_RE = re.compile(r"`([^`]+)`")


def _ident_candidates_from_text(text, cite_pos=None):
    """Pull identifier-shaped symbol candidates out of backtick spans in a
    line of doc prose (e.g. `Tensor::add_t`, `AddBackward`, `dual_mul`).

    When `cite_pos` (the byte offset of the cite the candidate is for) is
    given, order candidates so the backtick span NEAREST and PRECEDING the
    cite comes first — the prose convention is `` `view_operation` at `:198` ``
    where the symbol sits immediately before the cite. This prevents a
    line-leading symbol (`view_reshape`) from clobbering the per-cite symbol
    on a multi-cite line."""
    cands = []  # (start_pos, ident)
    for m in IDENT_BACKTICK_RE.finditer(text):
        inner = m.group(1)
        seg = inner.rsplit("::", 1)[-1]
        ident = ""
        for c in seg:
            if c.isalnum() or c == "_":
                ident += c
            else:
                break
        if ident and ident not in STOPWORD_HINTS and ident not in TOO_GENERIC_TYPE:
            cands.append((m.start(), ident))
    if cite_pos is None:
        return [ident for _, ident in cands]
    preceding = sorted([c for c in cands if c[0] < cite_pos],
                       key=lambda c: c[0], reverse=True)
    following = sorted([c for c in cands if c[0] >= cite_pos],
                       key=lambda c: c[0])
    return [ident for _, ident in preceding] + [ident for _, ident in following]


_LOCATE_CACHE = {}
_SKIP_DIRS = ("/.git/", "/target/", "/.claude/worktrees/", "/.worktrees/")


def locate_real_file(file_as_written):
    """For a cite whose path the resolver can't find (file lives in a crate
    the resolver doesn't search, e.g. ferrotorch-gpu / ferrotorch-diffusion),
    locate the real file in the workspace so we can still validate a symbol
    for a DURABLE anchor. The cite's line spec is dropped regardless, so this
    only affects the anchor's symbol name, never the test pass/fail."""
    if file_as_written in _LOCATE_CACHE:
        return _LOCATE_CACHE[file_as_written]
    parts = file_as_written.split("/")
    basename = parts[-1]
    want_tail = "/".join(parts[-2:]) if len(parts) >= 2 else basename
    best = None
    for dirpath, dirs, files in os.walk(WORKSPACE):
        full = dirpath + "/"
        if any(skip in full for skip in _SKIP_DIRS):
            dirs[:] = []
            continue
        if basename in files:
            cand = os.path.join(dirpath, basename)
            rel = os.path.relpath(cand, WORKSPACE).replace("\\", "/")
            if rel.endswith(want_tail):
                best = cand
                break
            if best is None:
                best = cand
    _LOCATE_CACHE[file_as_written] = best
    return best


def recover_symbol(cite, doc_line):
    """Best-effort recovery of the symbol the cite names, validated to exist
    in the target file. Returns symbol str or None.  Never fabricates: the
    returned symbol is always one declared in the target file (resolved by
    the test's resolver, or — for unresolvable paths — the real file located
    workspace-wide, used for anchor quality only)."""
    target = resolve_cite_path(cite.file_as_written)
    if target == "EXTERNAL":
        return None
    if target is None:
        target = locate_real_file(cite.file_as_written)
    if not target:
        return None
    candidates = []
    if cite.symbol_hint:
        candidates.append(cite.symbol_hint)
    cite_pos = cite.span_in_line[0] if cite.span_in_line else None
    candidates += _ident_candidates_from_text(doc_line, cite_pos)
    for cand in candidates:
        if file_defines_symbol(target, cand):
            return cand
    return None


# ---------------------------------------------------------------------------
# Rewriting.
# ---------------------------------------------------------------------------
def find_symbol_line(target, symbol):
    """Return the 1-indexed line of `symbol`'s declaration in `target`,
    preferring a `pub fn`/`fn`/`struct`/`enum`/`trait`/`const` definition,
    else the first needle match. Used for PROTECTED cites that must keep an
    accurate `file:line`."""
    try:
        src_lines = open(target).read().splitlines()
    except OSError:
        return None
    needles = build_symbol_needles(symbol)
    decl_markers = (
        f"pub fn {symbol}", f"fn {symbol}", f"pub struct {symbol}",
        f"struct {symbol}", f"pub enum {symbol}", f"enum {symbol}",
        f"pub trait {symbol}", f"trait {symbol}", f"pub const {symbol}",
        f"const {symbol}",
    )
    for i, ln in enumerate(src_lines):
        if any(m in ln for m in decl_markers):
            return i + 1
    for i, ln in enumerate(src_lines):
        if any(nd in ln for nd in needles):
            return i + 1
    return None


def find_arm_line(target, arm_op):
    """Find the parity-sweep runner arm `"<op>" =>` line (1-indexed)."""
    try:
        src_lines = open(target).read().splitlines()
    except OSError:
        return None
    anchor = f'"{arm_op}" =>'
    for i, ln in enumerate(src_lines):
        if anchor in ln:
            return i + 1
    return None


# ---------------------------------------------------------------------------
# Per-file rewrite.
# ---------------------------------------------------------------------------
def is_protected_arithmetic_row(doc_rel, doc_line_text):
    """The sibling tests (divergence_addcmul_req15, divergence_rsub_req9)
    audit the `| REQ-N |` table rows of arithmetic.md and require accurate
    `file:line` cites for the runner-arm / rsub symbol cites. Treat any cite
    on such a row as protected (keep an accurate line number)."""
    if doc_rel != ".design/ferrotorch-core/grad_fns/arithmetic.md":
        return False
    return doc_line_text.lstrip().startswith("| REQ-")


def rewrite_cite_text(cite, symbol):
    """Produce the replacement text for `cite`'s `<path>:<spec>` token.

    `<symbol> in <path>`   when a real symbol was recovered (durable anchor),
    `<path>`               otherwise (still names the file; no line to drift).
    The path keeps its as-written prefix (e.g. `grad_fns/arithmetic.rs`)."""
    path = cite.file_as_written
    if symbol:
        return f"{symbol} in {path}"
    return path


def rewrite_doc(doc_rel, abs_path, report):
    text = open(abs_path).read()
    lines = text.splitlines(keepends=True)
    # Re-derive cites (with line numbers + spans) from the doc.
    cites_by_line = {}
    for ln, c in audit_doc_cites(text):
        cites_by_line.setdefault(ln, []).append(c)

    changed = False
    for ln, cites in cites_by_line.items():
        raw = lines[ln - 1]
        nl = ""
        body = raw
        if body.endswith("\r\n"):
            nl = "\r\n"; body = body[:-2]
        elif body.endswith("\n"):
            nl = "\n"; body = body[:-1]
        stale = [c for c in cites if cite_is_stale(c)]
        if not stale:
            continue
        protected = is_protected_arithmetic_row(doc_rel, body)

        # Once a line has ANY stale cite, rewrite EVERY parseable cite on that
        # line (not just the stale ones). Rewriting a named cite to a symbol
        # anchor removes its `<file>:<N>` form, which would otherwise break
        # the doc-wide `.rs`-file context that following bare-colon cites
        # resolve against — re-attributing them to the wrong file on a later
        # pass. Converting them all in one pass, using each cite's
        # FIRST-PARSE file attribution (`cite.file_as_written`), keeps every
        # anchor pointing at the file it was parsed against. PROTECTED rows
        # never reach here (they have no stale cites), so their accurate
        # `file:line` cites are never touched.
        targets = stale if protected else cites

        # Build a list of non-overlapping edits (start, end, replacement)
        # over `body`, then apply right-to-left.
        edits = []
        for c in targets:
            if c.span_in_line is None:
                report["residual"].append(
                    (doc_rel, ln, c.file_as_written, c.line_start,
                     c.line_end, f"{c.kind} cite (no span)"))
                continue
            s, e = c.span_in_line
            if c.kind == "named":
                new_tok, why = decide_replacement(c, body, protected)
                if new_tok is None or new_tok == body[s:e]:
                    report["residual"].append(
                        (doc_rel, ln, c.file_as_written, c.line_start,
                         c.line_end, "no replacement: " + why))
                    continue
                edits.append((s, e, new_tok, why))
            elif c.kind == "bare_colon":
                enc = c.enclosing
                # If the enclosing backtick span is JUST this cite token
                # (e.g. `:248`), replace the whole span (backticks included)
                # with a clean symbol anchor; else drop the `:spec` token.
                if enc is not None and body[enc[0]:enc[1]].strip() == \
                        body[s:e].strip():
                    sym = recover_symbol(c, body)
                    anchor = rewrite_cite_text(c, sym)
                    # enc bounds are the CONTENT; include the surrounding
                    # backticks (enc[0]-1 .. enc[1]+1).
                    edits.append((enc[0] - 1, enc[1] + 1, f"`{anchor}`",
                                  "symbol_anchor" if sym else "drop_line"))
                else:
                    edits.append((s, e, "", "drop_barecolon"))
            elif c.kind == "paren":
                ds = s
                if ds > 0 and body[ds - 1] == " ":
                    ds -= 1
                edits.append((ds, e, "", "drop_paren"))

        if not edits:
            lines[ln - 1] = body + nl
            continue
        # Apply right-to-left, skipping overlaps.
        edits.sort(key=lambda x: x[0], reverse=True)
        applied_spans = []
        for (s, e, repl, why) in edits:
            if any(not (e <= os_ or s >= oe_) for (os_, oe_) in applied_spans):
                continue
            body = body[:s] + repl + body[e:]
            applied_spans.append((s, e))
            report[why] += 1
            changed = True

        body = cleanup_line(body)
        lines[ln - 1] = body + nl

    if changed:
        with open(abs_path, "w") as f:
            f.write("".join(lines))
    return changed


# Redundant `SYM` at|in `SYM in PATH`  ->  `SYM in PATH`  (drop the repeat).
_REDUNDANT_RE = re.compile(
    r"`([A-Za-z0-9_]+)`(\s+(?:at|in)\s+)`(\1(?:::[A-Za-z0-9_]+)*) in ([^`]+)`")
# Leftover empty backtick span possibly clinging to ` at`/` in`/`(`.
_EMPTY_AT_RE = re.compile(r"\s+(?:at|in)\s+``")
_EMPTY_PAREN_RE = re.compile(r"\s*\(``\)")
_DOUBLE_SPACE_RE = re.compile(r"  +")


def cleanup_line(body):
    """Cosmetic, cite-shape-preserving cleanups for lines this script edited.

    None of these can introduce a parseable `file.rs:NNN` cite — they only
    remove redundant repeats and empty/dangling backtick spans left behind by
    dropping a line number."""
    # `SYM` at `SYM in PATH` -> `SYM in PATH`
    body = _REDUNDANT_RE.sub(lambda m: f"`{m.group(3)} in {m.group(4)}`", body)
    # ` at ``` / ` in ``` (empty span) -> drop the dangling preposition+span
    body = _EMPTY_AT_RE.sub("", body)
    body = _EMPTY_PAREN_RE.sub("", body)
    # Collapse accidental double spaces introduced by deletions (but not at
    # line start, to preserve markdown indentation).
    if body[:1] not in (" ", "\t"):
        body = _DOUBLE_SPACE_RE.sub(" ", body)
    else:
        lead = len(body) - len(body.lstrip(" "))
        body = body[:lead] + _DOUBLE_SPACE_RE.sub(" ", body[lead:])
    return body


def decide_replacement(cite, doc_line, protected):
    """Return (new_token_text, reason_key). new_token replaces the cite's
    `<path>:<spec>` token."""
    target = resolve_cite_path(cite.file_as_written)
    if target == "EXTERNAL":
        return None, "external"

    # PROTECTED arithmetic.md table-row cites: keep an accurate file:line.
    if protected and target is not None:
        # Runner arm cite? (path ends with runner main.rs)
        if cite.file_as_written.endswith("parity-sweep/runner/src/main.rs"):
            op = arm_op_from_line(doc_line)
            if op:
                arm = find_arm_line(target, op)
                if arm:
                    return f"{cite.file_as_written}:{arm}", "protected_line"
        sym = recover_symbol(cite, doc_line)
        if sym:
            line = find_symbol_line(target, sym)
            if line:
                return f"{cite.file_as_written}:{line}", "protected_line"
        # Fall through to symbol-anchor if we can't pin a line.

    sym = recover_symbol(cite, doc_line)
    return rewrite_cite_text(cite, sym), ("symbol_anchor" if sym else "drop_line")


def arm_op_from_line(doc_line):
    """For a REQ row citing a runner arm, recover the op name from the
    `"<op>" =>` mention in the prose, else from `[op]` / `REQ-N (op)`."""
    m = re.search(r'"([a-z_]+)"\s*=>', doc_line)
    if m:
        return m.group(1)
    m = re.search(r"REQ-\d+\s*\(([a-z_]+)", doc_line)
    if m:
        return m.group(1)
    return None


def doc_stale_count(abs_path):
    """Count stale cites the test would flag in `abs_path` at HEAD state."""
    text = open(abs_path).read()
    return sum(1 for _, c in audit_doc_cites(text) if cite_is_stale(c))


def main():
    import collections
    report = collections.defaultdict(int)
    report["residual"] = []
    docs = []
    for dirpath, _, files in os.walk(DESIGN):
        for fn in files:
            if fn.endswith(".md"):
                docs.append(os.path.join(dirpath, fn))
    docs.sort()

    # Rewriting one cite can shift the doc-wide `.rs`-file context that
    # bare-colon / bare-paren continuation cites resolve against, surfacing a
    # fresh stale cite. Iterate each doc to a fixed point.
    n_changed = 0
    for abs_path in docs:
        rel = os.path.relpath(abs_path, WORKSPACE)
        changed_any = False
        for _ in range(12):
            if doc_stale_count(abs_path) == 0:
                break
            if not rewrite_doc(rel, abs_path, report):
                break
            changed_any = True
        if changed_any:
            n_changed += 1

    print(f"docs changed: {n_changed}")
    for k in ("symbol_anchor", "drop_line", "drop_barecolon", "drop_paren",
              "protected_line"):
        print(f"  {k}: {report[k]}")

    # Final verification pass.
    remaining = []
    for abs_path in docs:
        text = open(abs_path).read()
        for ln, c in audit_doc_cites(text):
            if cite_is_stale(c):
                rel = os.path.relpath(abs_path, WORKSPACE)
                remaining.append((rel, ln, c.file_as_written, c.line_start,
                                  c.line_end, c.kind))
    print(f"  remaining stale cites: {len(remaining)}")
    for r in remaining[:80]:
        print("    ", r)
    if report["residual"]:
        print(f"  residual (unhandled): {len(report['residual'])}")
        for r in report["residual"][:40]:
            print("    ", r)


if __name__ == "__main__":
    main()
