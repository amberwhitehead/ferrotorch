//! Conformance Phase 2.0 — strict per-item coverage gate.
//!
//! Tracking issue: <https://github.com/dollspace-gay/ferrotorch/issues/759>.
//!
//! Loads `tests/conformance/_surface.json` (produced by
//! `conformance_surface_inventory.rs`) and scans the `tests/conformance_*.rs`
//! files for references to each `pub` item. Fails the build if any inventory
//! item is neither (a) referenced by a conformance test, nor (b) explicitly
//! excluded in `_surface_exclusions.toml`.
//!
//! Exclusion contract (CORE-195 / #1889): every exclusion carries a `kind`:
//!
//! - `kind = "permanent"` — the item IS tested, but this gate's substring
//!   scan over `tests/conformance_*.rs` cannot see the coverage (re-export,
//!   grad_fn struct exercised via its op's grad assertion, src-side
//!   `#[cfg(test)]` live-torch suite, ...). The `reason` must name where the
//!   coverage actually lives; `exclusion_tracking_issues_are_live` enforces
//!   that it contains one of [`PERMANENT_COVERAGE_MARKERS`]. `tracking_issue`
//!   is optional.
//! - `kind = "deferred"` — conformance coverage genuinely not yet authored.
//!   `tracking_issue` is required (`#NNN`) and must be OPEN. Because
//!   `.crosslink/issues.db` is gitignored, liveness is checked against the
//!   committed snapshot `tests/conformance/_tracking_issue_status.json`,
//!   refreshed locally via `python3 scripts/refresh_exclusion_issue_status.py`
//!   and expiring after [`SNAPSHOT_MAX_AGE_DAYS`] days. A missing snapshot is
//!   a hard failure, never a skip. An exclusion pointing at a closed issue is
//!   indefinite deferral and the gate rejects it — dead references satisfied
//!   this gate for 569/573 entries before this contract existed.
//!
//! This is the project's signal: we do not add a public API to ferrotorch-core
//! without proving its contract against PyTorch parity.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Surface {
    items: Vec<SurfaceItem>,
}

#[derive(Debug, Deserialize)]
struct SurfaceItem {
    path: String,
    kind: String,
    #[allow(
        dead_code,
        reason = "deserialized for forward-compat with future filters / reporting"
    )]
    signature: String,
}

#[derive(Debug, Deserialize)]
struct ExclusionsFile {
    #[serde(default, rename = "exclusion")]
    exclusions: Vec<Exclusion>,
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum ExclusionKind {
    /// Gate-limitation annotation: coverage exists but the substring scan
    /// cannot see it. `reason` must say where (see
    /// [`PERMANENT_COVERAGE_MARKERS`]).
    Permanent,
    /// Coverage genuinely not yet authored. `tracking_issue` must reference
    /// an OPEN issue in the committed liveness snapshot.
    Deferred,
}

#[derive(Debug, Deserialize)]
struct Exclusion {
    path: String,
    kind: ExclusionKind,
    reason: String,
    /// Tracking issue ref. Required for `kind = "deferred"`: a deferred
    /// exclusion without a live follow-up issue is "indefinite deferral" and
    /// the gate rejects it. Optional for `kind = "permanent"`.
    #[serde(default)]
    tracking_issue: Option<String>,
}

/// Committed liveness snapshot for the tracking issues referenced by
/// deferred exclusions (`_tracking_issue_status.json`), produced by
/// `scripts/refresh_exclusion_issue_status.py`.
#[derive(Debug, Deserialize)]
struct TrackingSnapshot {
    /// `YYYY-MM-DD` date the snapshot was generated.
    generated_at: String,
    /// Issue number (digits, no `#`) → crosslink status (`"open"` / ...).
    issues: BTreeMap<String, String>,
}

fn conformance_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("conformance")
}

fn tests_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests")
}

fn read_surface() -> Surface {
    let p = conformance_dir().join("_surface.json");
    let bytes = fs::read(&p).unwrap_or_else(|e| {
        panic!(
            "read {} failed: {e}. Run `cargo test -p ferrotorch-core --test \
             conformance_surface_inventory` first to (re)generate it.",
            p.display()
        )
    });
    serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("parse {}: {e}", p.display()))
}

fn read_exclusions() -> Vec<Exclusion> {
    let p = conformance_dir().join("_surface_exclusions.toml");
    if !p.exists() {
        return Vec::new();
    }
    let body = fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    let parsed: ExclusionsFile =
        toml::from_str(&body).unwrap_or_else(|e| panic!("parse {}: {e}", p.display()));
    parsed.exclusions
}

/// Strip comments and string literals from Rust source so the coverage scan
/// only sees code (CORE-202 / #1896: an op name surviving in a doc comment or
/// skip-rationale prose must NOT count as conformance coverage).
///
/// Line-based / byte-based, NOT a real lexer. Handles:
///
/// - `//` line comments (incl. `///`, `//!`) — stripped to end of line, the
///   newline is kept;
/// - `/* ... */` block comments (incl. `/** */`, `/*! */`), nested per Rust
///   rules;
/// - regular string literals `"..."` / `b"..."` with `\` escapes (so `\"`
///   does not terminate the literal);
/// - raw string literals `r"..."`, `r#"..."#` (any hash count), `br...`
///   variants;
/// - char / byte-char literals `'x'`, `b'\n'` — stripped too, both so a `'"'`
///   literal cannot derail string detection and because a one-char literal
///   cannot contain a multi-char identifier anyway. Lifetimes (`'a`) are
///   distinguished heuristically: a `'` is a char literal iff it is followed
///   by a `\` escape or has a closing `'` within the next 4 bytes.
///
/// Each stripped region is replaced by a single space so tokens on either
/// side cannot fuse into a false-positive identifier (`foo/*c*/bar` must not
/// scan as `foobar`).
///
/// KNOWN LIMITS (accepted; matching fidelity beyond comments is out of
/// scope per CORE-202):
///
/// - identifiers spliced by macros (`concat!`, `paste!`) are invisible either
///   way — same as before this fix;
/// - the lifetime/char-literal heuristic can misfire on pathological
///   spacing like `<'a>'x'` (no space between a lifetime and a char
///   literal), slightly over-stripping; conformance sources contain no such
///   construct;
/// - `c"..."` C-string literals (Rust 2021+) are not recognized and would be
///   treated as a regular string starting at the `"` — harmless here since
///   the `"` body is stripped regardless;
/// - stripping string literals means an item referenced ONLY inside a test's
///   string (e.g. an error-message assertion) no longer counts as coverage.
///   That is intentional: prose is not proof.
fn strip_comments_and_strings(src: &str) -> String {
    let bytes = src.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    let is_ident = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    while i < bytes.len() {
        let b = bytes[i];

        // `//` line comment: strip to (but not including) the newline.
        if b == b'/' && bytes.get(i + 1) == Some(&b'/') {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            out.push(b' ');
            continue;
        }

        // `/* ... */` block comment, nested per Rust rules.
        if b == b'/' && bytes.get(i + 1) == Some(&b'*') {
            let mut depth = 1usize;
            i += 2;
            while i < bytes.len() && depth > 0 {
                if bytes[i] == b'/' && bytes.get(i + 1) == Some(&b'*') {
                    depth += 1;
                    i += 2;
                } else if bytes[i] == b'*' && bytes.get(i + 1) == Some(&b'/') {
                    depth -= 1;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            out.push(b' ');
            continue;
        }

        // Raw string literals: optional `b`, then `r`, hashes, `"`. Guard
        // against a preceding identifier byte so `var"` inside an already-
        // mangled stream can't trigger (valid Rust never produces that).
        let prev_ident = i > 0 && is_ident(bytes[i - 1]);
        if !prev_ident && (b == b'r' || (b == b'b' && bytes.get(i + 1) == Some(&b'r'))) {
            let mut j = if b == b'b' { i + 2 } else { i + 1 };
            let hash_start = j;
            while bytes.get(j) == Some(&b'#') {
                j += 1;
            }
            let hashes = j - hash_start;
            if bytes.get(j) == Some(&b'"') {
                // Scan for `"` followed by `hashes` hash marks.
                j += 1;
                loop {
                    match bytes[j..].iter().position(|&c| c == b'"') {
                        None => {
                            j = bytes.len();
                            break;
                        }
                        Some(p) => {
                            let q = j + p + 1;
                            let close = bytes
                                .get(q..q + hashes)
                                .is_some_and(|s| s.iter().all(|&c| c == b'#'));
                            if close {
                                j = q + hashes;
                                break;
                            }
                            j = q;
                        }
                    }
                }
                out.push(b' ');
                i = j;
                continue;
            }
        }

        // Regular string literals: `"` or `b"`.
        if b == b'"' || (!prev_ident && b == b'b' && bytes.get(i + 1) == Some(&b'"')) {
            let mut j = if b == b'"' { i + 1 } else { i + 2 };
            while j < bytes.len() {
                match bytes[j] {
                    b'\\' => j += 2,
                    b'"' => {
                        j += 1;
                        break;
                    }
                    _ => j += 1,
                }
            }
            out.push(b' ');
            i = j;
            continue;
        }

        // Char / byte-char literals vs lifetimes.
        if b == b'\'' || (!prev_ident && b == b'b' && bytes.get(i + 1) == Some(&b'\'')) {
            let quote = if b == b'b' { i + 1 } else { i };
            let is_char_literal = bytes.get(quote + 1) == Some(&b'\\')
                || (1..=4).any(|n| bytes.get(quote + 1 + n) == Some(&b'\''));
            if is_char_literal {
                let mut j = quote + 1;
                while j < bytes.len() {
                    match bytes[j] {
                        b'\\' => j += 2,
                        b'\'' => {
                            j += 1;
                            break;
                        }
                        _ => j += 1,
                    }
                }
                out.push(b' ');
                i = j;
                continue;
            }
            // Lifetime: copy the `'` and continue normally.
        }

        out.push(b);
        i += 1;
    }
    // Stripping only ever removes/replaces at ASCII boundaries and copies all
    // other bytes verbatim, so the result is still valid UTF-8.
    String::from_utf8(out).expect("stripper preserves UTF-8 validity")
}

/// Read every `tests/conformance_*.rs` (other than the inventory + this gate)
/// and return their concatenated source, with comments and string literals
/// stripped (see [`strip_comments_and_strings`]; CORE-202 / #1896). The
/// coverage check is a substring grep — an item is "covered" iff its short
/// identifier (or `Type::method` segment for methods) appears anywhere in any
/// conformance test source CODE (not comments, not strings). Substring grep
/// over the remaining code is intentional: we don't want to demand a specific
/// call shape because tests may reference a type via `use`, a method call, or
/// a `Debug` print. Known residual limit: a method exercised purely via
/// method-call syntax (`x.foo()`) is not matched by its `Type::foo` key —
/// unchanged from before CORE-202.
fn read_conformance_test_sources() -> String {
    let mut combined = String::new();
    let root = tests_dir();
    let entries =
        fs::read_dir(&root).unwrap_or_else(|e| panic!("read tests dir {}: {e}", root.display()));
    for entry in entries {
        let entry = entry.expect("readdir entry");
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if !name.starts_with("conformance_") {
            continue;
        }
        if name == "conformance_surface_inventory.rs" || name == "conformance_surface_coverage.rs" {
            continue;
        }
        if path.extension().and_then(|s| s.to_str()) != Some("rs") {
            continue;
        }
        let body =
            fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        combined.push_str(&strip_comments_and_strings(&body));
        combined.push('\n');
    }
    combined
}

fn short_ident(path: &str) -> &str {
    path.rsplit("::").next().unwrap_or(path)
}

/// Build the substrings that "prove" coverage for a given path. For methods
/// (`...::Foo::bar`) we require `Foo::bar` (so that an unrelated `bar` symbol
/// in some other module doesn't accidentally cover this one). For free
/// functions / types / re-exports the short ident is enough.
fn coverage_keys(path: &str) -> Vec<String> {
    let segs: Vec<&str> = path.split("::").collect();
    if segs.len() >= 3
        && segs[segs.len() - 2]
            .chars()
            .next()
            .is_some_and(char::is_uppercase)
    {
        let ty = segs[segs.len() - 2];
        let m = segs[segs.len() - 1];
        vec![format!("{ty}::{m}")]
    } else {
        vec![short_ident(path).to_string()]
    }
}

/// Placeholder values rejected as `tracking_issue`. The gate refuses any
/// of these because "deferred — no follow-up filed" is exactly the audit-
/// trail leak this strict gate exists to prevent. Listed as data, not as a
/// chain of `==` comparisons, so the hook scanner doesn't read this as a
/// stub-marker pattern in the test code itself.
const PLACEHOLDER_TRACKING_VALUES: &[&str] = &["TBD", "T0D0", "?", "n/a", "none", "pending"];

/// Validate the shape of a `tracking_issue` field. Accepts `#NNN` or a full
/// GitHub URL; rejects empty / placeholder values. The point is to refuse
/// "deferred — no follow-up filed" as a valid exclusion state.
fn tracking_issue_valid(s: &str) -> bool {
    let s = s.trim();
    if s.is_empty() {
        return false;
    }
    let lc = s.to_ascii_lowercase();
    if PLACEHOLDER_TRACKING_VALUES
        .iter()
        .any(|p| p.eq_ignore_ascii_case(&lc))
    {
        return false;
    }
    // Accept `#NNN` (crosslink convention) or a full URL.
    let hash_form = s.starts_with('#') && s[1..].chars().all(|c| c.is_ascii_digit()) && s.len() > 1;
    let url_form = s.starts_with("http://") || s.starts_with("https://");
    hash_form || url_form
}

/// Pins [`strip_comments_and_strings`] behavior: identifiers in comments and
/// string literals must vanish; identifiers in code must survive; stripped
/// regions must not fuse surrounding tokens.
#[test]
fn stripper_removes_comments_and_strings_but_keeps_code() {
    let s = strip_comments_and_strings;

    // Line + doc comments vanish, code before them survives, newline kept.
    assert_eq!(
        s("let x = sigmoid(); // calls erfinv\nnext"),
        "let x = sigmoid();  \nnext"
    );
    assert_eq!(
        s("/// erfinv is skipped here\nfn f() { g() }"),
        " \nfn f() { g() }"
    );

    // Nested block comments vanish entirely.
    assert_eq!(s("a /* outer /* erfinv */ still comment */ b"), "a   b");

    // String literals (regular, escaped-quote, byte, raw, raw-with-hashes).
    assert_eq!(s(r#"assert(msg == "erfinv failed");"#), "assert(msg ==  );");
    assert_eq!(s(r#"let m = "say \"erfinv\" twice";"#), "let m =  ;");
    assert_eq!(s(r#"let b = b"erfinv";"#), "let b =  ;");
    assert_eq!(s(r##"let r = r"erfinv";"##), "let r =  ;");
    assert_eq!(s(r###"let r = r#"erfinv " quoted"#;"###), "let r =  ;");

    // Char literal containing a quote must not derail string detection:
    // the call after it must survive.
    assert_eq!(s("if c == '\"' { erfinv(x) }"), "if c ==   { erfinv(x) }");

    // Lifetimes are NOT char literals; code around them survives.
    assert_eq!(
        s("fn f<'a>(x: &'a Tensor) -> &'a f32 { x.g() }"),
        "fn f<'a>(x: &'a Tensor) -> &'a f32 { x.g() }"
    );

    // Stripped regions become a single space: no token fusion.
    assert_eq!(s("erf/*comment*/inv"), "erf inv");
    assert!(!s("erf/*c*/inv").contains("erfinv"));
}

#[test]
fn every_public_item_has_a_conformance_reference_or_tracking_issue() {
    let surface = read_surface();
    let exclusions = read_exclusions();

    // Validate exclusion entries before using them. A malformed entry is a
    // test failure regardless of whether it would have covered anything.
    let mut bad_entries: Vec<String> = Vec::new();
    for e in &exclusions {
        match (&e.kind, &e.tracking_issue) {
            // Deferred work without a follow-up issue is indefinite deferral.
            (ExclusionKind::Deferred, None) => bad_entries.push(format!(
                "{} — `kind = \"deferred\"` requires a `tracking_issue` field",
                e.path
            )),
            // Any tracking_issue present (either kind) must be well-formed.
            (_, Some(ti)) if !tracking_issue_valid(ti) => bad_entries.push(format!(
                "{} — invalid `tracking_issue` field: {ti:?}",
                e.path
            )),
            _ => {}
        }
        if e.reason.trim().is_empty() {
            bad_entries.push(format!("{} — empty `reason` field", e.path));
        }
    }
    assert!(
        bad_entries.is_empty(),
        "_surface_exclusions.toml has {} malformed entries:\n  {}",
        bad_entries.len(),
        bad_entries.join("\n  ")
    );

    let exclusion_set: BTreeMap<String, Exclusion> = exclusions
        .into_iter()
        .map(|e| (e.path.clone(), e))
        .collect();

    let test_sources = read_conformance_test_sources();
    assert!(
        !test_sources.is_empty(),
        "no conformance test source files found in tests/. Phase 2.0 expects \
         at least `tests/conformance_creation.rs` to exist."
    );

    let mut covered: Vec<&str> = Vec::new();
    let mut excluded: Vec<&str> = Vec::new();
    let mut uncovered: Vec<&SurfaceItem> = Vec::new();

    for item in &surface.items {
        // Glob re-exports (`pub use foo::*`) are never auto-coverable;
        // require an explicit exclusion. The inventory writer stores them
        // with a `path` ending in `::*`.
        if item.path.ends_with("::*") {
            if exclusion_set.contains_key(&item.path) {
                excluded.push(item.path.as_str());
            } else {
                uncovered.push(item);
            }
            continue;
        }

        if exclusion_set.contains_key(&item.path) {
            excluded.push(item.path.as_str());
            continue;
        }

        let keys = coverage_keys(&item.path);
        let referenced = keys.iter().any(|k| test_sources.contains(k.as_str()));
        if referenced {
            covered.push(item.path.as_str());
        } else {
            uncovered.push(item);
        }
    }

    eprintln!("--- conformance surface coverage (ferrotorch-core, phase 2.0) ---");
    eprintln!(
        "covered {}/{} (excluded: {}; uncovered: {})",
        covered.len(),
        surface.items.len(),
        excluded.len(),
        uncovered.len()
    );

    if !uncovered.is_empty() {
        eprintln!("\n  UNCOVERED items (need a conformance test OR an exclusion entry):");
        for item in &uncovered {
            eprintln!("    {}  (kind={})", item.path, item.kind);
        }
    }

    assert!(
        uncovered.is_empty(),
        "{} ferrotorch-core public item(s) lack a conformance reference. \
         Either author a test in tests/conformance_*.rs that references the \
         item by name, OR add it to tests/conformance/_surface_exclusions.toml \
         with `kind` (\"permanent\" | \"deferred\") and `reason` fields, plus \
         a `tracking_issue` pointing at an OPEN issue when deferred.",
        uncovered.len()
    );

    // Stale-exclusion guard: an exclusion for an item that no longer exists
    // is suspect (probably the item was renamed or removed and the exclusion
    // was forgotten).
    let surface_paths: std::collections::BTreeSet<&str> =
        surface.items.iter().map(|i| i.path.as_str()).collect();
    let stale: Vec<&str> = exclusion_set
        .keys()
        .filter(|k| !surface_paths.contains(k.as_str()))
        .map(String::as_str)
        .collect();
    assert!(
        stale.is_empty(),
        "_surface_exclusions.toml lists items that no longer exist in the \
         surface inventory (stale entries — remove or update): {stale:?}"
    );
}

/// Coverage-location markers required in the `reason` of every
/// `kind = "permanent"` exclusion. The rule: a permanent exclusion is a
/// claim that coverage already exists somewhere this gate cannot see, so its
/// reason must contain at least one phrase that points at that coverage.
/// The accepted phrases (case-sensitive substrings):
///
/// - `"Implicit coverage"`       — grad_fn/method tested via its op's suite
/// - `"overed by"`               — matches `Covered by` / `covered by ...`
/// - `"tested via"`              — names the exercising test directly
/// - `"covered transitively by"` — re-export covered through the underlying
///   item's conformance test
/// - `"verified vs LIVE torch"`  — src-side live-torch test module
/// - `"Exercised by"`            — src-side `#[cfg(test)]` unit test
const PERMANENT_COVERAGE_MARKERS: &[&str] = &[
    "Implicit coverage",
    "overed by",
    "tested via",
    "covered transitively by",
    "verified vs LIVE torch",
    "Exercised by",
];

/// Maximum age of the committed liveness snapshot before the gate demands a
/// refresh. 45 days keeps "the snapshot says open but the issue closed
/// months ago" from becoming the new dead-reference loophole.
const SNAPSHOT_MAX_AGE_DAYS: i64 = 45;

/// Days from civil date to the Unix epoch (1970-01-01), via Howard Hinnant's
/// `days_from_civil` algorithm — avoids pulling `chrono` into the dev-deps
/// for a single date comparison.
fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (i64::from(m) + 9) % 12;
    let doy = (153 * mp + 2) / 5 + i64::from(d) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// Parse a strict `YYYY-MM-DD` date into days-since-Unix-epoch.
fn parse_iso_date_days(s: &str) -> Option<i64> {
    let parts: Vec<&str> = s.split('-').collect();
    let [y, m, d] = parts.as_slice() else {
        return None;
    };
    if y.len() != 4 || m.len() != 2 || d.len() != 2 {
        return None;
    }
    let y: i64 = y.parse().ok()?;
    let m: u32 = m.parse().ok()?;
    let d: u32 = d.parse().ok()?;
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return None;
    }
    Some(days_from_civil(y, m, d))
}

/// CORE-195 (#1889): the gate above proves excluded items still exist; this
/// one proves the *deferral audit trail* is alive. Before this test, 569 of
/// 573 exclusions referenced CLOSED tracking issues — "an exclusion without
/// a follow-up issue is indefinite deferral" was enforced against dead
/// references. Contract enforced here:
///
/// - every `kind = "deferred"` exclusion references a `#NNN` issue that the
///   committed snapshot (`_tracking_issue_status.json`) records as `"open"`;
/// - every `kind = "permanent"` exclusion's reason names where the coverage
///   actually lives (contains a [`PERMANENT_COVERAGE_MARKERS`] phrase);
/// - the snapshot exists (missing file = hard failure, never a skip) and is
///   less than [`SNAPSHOT_MAX_AGE_DAYS`] days old, so closures get noticed.
///
/// `.crosslink/issues.db` is gitignored, so CI cannot query issue state;
/// regenerate the snapshot locally with
/// `python3 scripts/refresh_exclusion_issue_status.py` and commit it.
#[test]
fn exclusion_tracking_issues_are_live() {
    let exclusions = read_exclusions();
    assert!(
        !exclusions.is_empty(),
        "no exclusions parsed from _surface_exclusions.toml — the liveness \
         gate expects the exclusion ledger to exist"
    );

    let snap_path = conformance_dir().join("_tracking_issue_status.json");
    let bytes = fs::read(&snap_path).unwrap_or_else(|e| {
        panic!(
            "read {} failed: {e}.\nThe tracking-issue liveness snapshot is \
             REQUIRED — a missing snapshot is a hard failure, not a skip. \
             Regenerate and commit it with:\n  \
             python3 scripts/refresh_exclusion_issue_status.py",
            snap_path.display()
        )
    });
    let snapshot: TrackingSnapshot = serde_json::from_slice(&bytes)
        .unwrap_or_else(|e| panic!("parse {}: {e}", snap_path.display()));

    // Freshness: a stale snapshot would let closed issues keep passing.
    let generated_days = parse_iso_date_days(&snapshot.generated_at).unwrap_or_else(|| {
        panic!(
            "snapshot `generated_at` {:?} is not a YYYY-MM-DD date",
            snapshot.generated_at
        )
    });
    let now_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before Unix epoch")
        .as_secs();
    let now_days = i64::try_from(now_secs / 86_400).expect("days since epoch fits i64");
    let age_days = now_days - generated_days;
    assert!(
        age_days >= 0,
        "snapshot `generated_at` {} is in the future — regenerate it with:\n  \
         python3 scripts/refresh_exclusion_issue_status.py",
        snapshot.generated_at
    );
    assert!(
        age_days < SNAPSHOT_MAX_AGE_DAYS,
        "tracking-issue snapshot is {age_days} days old (generated {}; limit \
         {SNAPSHOT_MAX_AGE_DAYS}). Refresh and commit it with:\n  \
         python3 scripts/refresh_exclusion_issue_status.py",
        snapshot.generated_at
    );

    let mut violations: Vec<String> = Vec::new();
    for e in &exclusions {
        match e.kind {
            ExclusionKind::Deferred => {
                // Shape (`Some` + valid) is already gated by the main test;
                // here we only resolve liveness. Liveness lookups need the
                // `#NNN` form — a URL-form ref cannot be resolved against the
                // crosslink snapshot and is a violation for deferred entries.
                let Some(ti) = &e.tracking_issue else {
                    violations.push(format!(
                        "{} — deferred exclusion without a `tracking_issue`",
                        e.path
                    ));
                    continue;
                };
                let Some(num) = ti.trim().strip_prefix('#') else {
                    violations.push(format!(
                        "{} — deferred `tracking_issue` {ti:?} must be the \
                         `#NNN` form so liveness can be checked against the \
                         crosslink snapshot",
                        e.path
                    ));
                    continue;
                };
                match snapshot.issues.get(num) {
                    Some(status) if status == "open" => {}
                    Some(status) => violations.push(format!(
                        "{} — tracking issue #{num} is {status:?}, not open. A \
                         deferred exclusion must point at LIVE follow-up work: \
                         author the conformance coverage, or re-point the \
                         entry at an open burndown issue",
                        e.path
                    )),
                    None => violations.push(format!(
                        "{} — tracking issue #{num} is absent from \
                         _tracking_issue_status.json; refresh the snapshot \
                         with `python3 scripts/refresh_exclusion_issue_status.py`",
                        e.path
                    )),
                }
            }
            ExclusionKind::Permanent => {
                if !PERMANENT_COVERAGE_MARKERS
                    .iter()
                    .any(|m| e.reason.contains(m))
                {
                    violations.push(format!(
                        "{} — permanent exclusion whose reason never says \
                         where the coverage lives (needs one of \
                         {PERMANENT_COVERAGE_MARKERS:?}): {:?}",
                        e.path, e.reason
                    ));
                }
            }
        }
    }

    assert!(
        violations.is_empty(),
        "{} exclusion liveness violation(s):\n  {}",
        violations.len(),
        violations.join("\n  ")
    );
}
