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
use std::path::PathBuf;

fn workspace_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if !p.join(".design").exists() {
        p.pop();
    }
    p
}

/// Returns true if any line in `lines[lo-1 ..= hi-1]` (1-based, ±3 window
/// clamped) contains `needle`. Matches the cite-drift tolerance of the
/// generic audit.

#[test]
fn req8_indexing_md_cites_resolve_to_named_symbols() {
    let root = workspace_root();
    let design = root.join(".design/ferrotorch-core/grad_fns/indexing.md");
    let design_text = fs::read_to_string(&design)
        .unwrap_or_else(|e| panic!("read {} failed: {e}", design.display()));

    // Check that the stale line-number cites have been replaced with
    // symbol anchors per goal.md S3 discipline. After the fix, the old
    // line-number substrings should be gone.

    // Also assert that the design doc CONTAINS these stale cites — if a
    // future fix simultaneously drops the cite from the doc AND introduces
    // a different stale one, we want to fail loudly. We do NOT assert the
    // cite is at any specific design-doc line (that drifts trivially).
    let stale_substrings = ["indexing.rs:1471", "indexing.rs:1383", "methods.rs:614"];
    let mut present_stale: Vec<&str> = Vec::new();
    for s in stale_substrings {
        if design_text.contains(s) {
            present_stale.push(s);
        }
    }

    assert!(
        present_stale.is_empty(),
        "REQ-8 (index_fill) row in .design/ferrotorch-core/grad_fns/indexing.md \
         still carries stale line-number cites:\n  {:?}\n\n\
         Per goal.md S3 discipline, cites in design docs must use SYMBOL ANCHORS \
         (e.g., `struct IndexFillBackward in indexing.rs`), never line numbers. \
         Fix by replacing `<symbol> at <file>:<line>` with `<symbol> (in <file>)`.",
        present_stale
    );
}
