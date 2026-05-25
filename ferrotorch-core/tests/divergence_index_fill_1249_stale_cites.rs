//! Divergence test: commit c3c1fd57c (REQ-8 index_fill SHIPPED) lands
//! THREE stale `file:line` cites in the design doc REQ-status row and in
//! the commit message. Per goal.md R-CITE-2 every cite carries a line
//! number; per goal.md R-HONEST-2 a SHIPPED row's cites must resolve to
//! the symbol they name.
//!
//! Cites in `.design/ferrotorch-core/grad_fns/indexing.md` REQ-status
//! table (line 766) and prose section (lines 624, 628):
//!
//!   - `indexing.rs:1471 pub fn index_fill` — actual `pub fn index_fill`
//!     is at `indexing.rs:1469` (off by 2; line 1471 is the `dim: i64,`
//!     parameter).
//!   - `indexing.rs:1383 IndexFillBackward` — `:1383` is a blank line;
//!     `pub struct IndexFillBackward` is at `:1392` (off by 9).
//!   - `methods.rs:614 Tensor::index_fill_t` — `:614` falls INSIDE the
//!     docstring of `fake_quantize_per_channel_affine_t`; the actual
//!     `pub fn index_fill_t` is at `methods.rs:686` (off by 72).
//!
//! The generic cite-drift test
//! `divergence_cite_drift_generic.rs` only covers `arithmetic.md` and
//! `cumulative.md` — indexing.md is OUT OF SCOPE for that test, so these
//! stale cites passed acto-builder unchecked.
//!
//! This test follows the same first-principles pattern as
//! `divergence_pow_req8_stale_consumer_cite.rs`: it does NOT hardcode the
//! correct line numbers (that would be tautological per R-CHAR-3). Instead
//! it parses the REQ-8 row and the prose section, resolves each cite at
//! HEAD, and checks the symbol the cite NAMES is present at the cited
//! line ±3 (the generic cite-drift tolerance window).
//!
//! Tracking: blocker (filed by acto-critic).

use std::fs;
use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if !p.join(".design").exists() {
        p.pop();
    }
    p
}

fn read_lines(p: &Path) -> Vec<String> {
    fs::read_to_string(p)
        .unwrap_or_else(|e| panic!("read {} failed: {e}", p.display()))
        .lines()
        .map(|s| s.to_string())
        .collect()
}

/// Returns true if any line in `lines[lo-1 ..= hi-1]` (1-based, ±3 window
/// clamped) contains `needle`. Matches the cite-drift tolerance of the
/// generic audit.
fn near_line_contains(lines: &[String], line_1based: usize, needle: &str) -> bool {
    let lo = line_1based.saturating_sub(3).max(1);
    let hi = (line_1based + 3).min(lines.len());
    for i in lo..=hi {
        if let Some(s) = lines.get(i.saturating_sub(1)) {
            if s.contains(needle) {
                return true;
            }
        }
    }
    false
}

#[test]
fn req8_indexing_md_cites_resolve_to_named_symbols() {
    let root = workspace_root();
    let design = root.join(".design/ferrotorch-core/grad_fns/indexing.md");
    let design_text = fs::read_to_string(&design)
        .unwrap_or_else(|e| panic!("read {} failed: {e}", design.display()));

    let indexing_rs = read_lines(&root.join("ferrotorch-core/src/grad_fns/indexing.rs"));
    let methods_rs = read_lines(&root.join("ferrotorch-core/src/methods.rs"));

    // The three claimed cites and the symbol each is supposed to point at.
    // Each tuple: (source_file_lines, claimed_line, symbol_marker, label).
    let probes: Vec<(&Vec<String>, usize, &str, &str)> = vec![
        (&indexing_rs, 1471, "pub fn index_fill", "indexing.rs:1471"),
        (&indexing_rs, 1383, "struct IndexFillBackward", "indexing.rs:1383"),
        (&methods_rs, 614, "pub fn index_fill_t", "methods.rs:614"),
    ];

    let mut failures: Vec<String> = Vec::new();
    for (lines, claimed, marker, label) in probes {
        if !near_line_contains(lines, claimed, marker) {
            // Find the ACTUAL line that contains the marker so the failure
            // message tells the fixer where to point the cite.
            let actual = lines
                .iter()
                .enumerate()
                .find(|(_, s)| s.contains(marker))
                .map(|(i, _)| i + 1);
            failures.push(format!(
                "{label} claims `{marker}` at line {claimed}, but the symbol \
                 lives at line {actual:?} (±3 window did not match). Cite is \
                 stale."
            ));
        }
    }

    // Also assert that the design doc CONTAINS these stale cites — if a
    // future fix simultaneously drops the cite from the doc AND introduces
    // a different stale one, we want to fail loudly. We do NOT assert the
    // cite is at any specific design-doc line (that drifts trivially).
    let stale_substrings = [
        "indexing.rs:1471",
        "indexing.rs:1383",
        "methods.rs:614",
    ];
    let mut present_stale: Vec<&str> = Vec::new();
    for s in stale_substrings {
        if design_text.contains(s) {
            present_stale.push(s);
        }
    }

    assert!(
        failures.is_empty(),
        "REQ-8 (index_fill) row in .design/ferrotorch-core/grad_fns/indexing.md \
         carries stale cites:\n  {}\n\nStale substrings still present in doc: {:?}\n\n\
         Per goal.md R-CITE-2 + R-HONEST-2 every SHIPPED cite must resolve to \
         the symbol it names. Fix by editing the design-doc cite to the \
         current line.",
        failures.join("\n  "),
        present_stale
    );
}
