//! Post-#1216 divergence: `.design/ferrotorch-core/inplace.md` prose still
//! describes `Tensor::sub_` as a hand-rolled, shape-strict, no-broadcast,
//! `sub_f32`-fast-path implementation — but commit `2f792bfc5` replaced that
//! 40-line body with a one-line delegation `self.sub_scaled_(other, 1.0)`
//! which inherits broadcasting + GPU dispatch from `add_scaled_`.
//!
//! ## The drift
//!
//! The actual code at `ferrotorch-core/src/inplace.rs:295-297` is now:
//!
//! ```ignore
//! pub fn sub_(&self, other: &Tensor<T>) -> FerrotorchResult<&Self> {
//!     self.sub_scaled_(other, 1.0)
//! }
//! ```
//!
//! But the design doc claims:
//!
//! - Line 104 (REQ-7 prose): "ferrotorch's current `sub_` at `inplace.rs:248`"
//!   — wrong line (sub_ is at :295 after the delegation).
//! - Lines 280-288 (architecture section): "`sub_(other)` (`:278`) — checks
//!   `self.shape() == other.shape()` (shape-strict, no broadcasting), GPU
//!   f32 fast path via `sub_f32`, CPU fallback. REQ-7 NOT-STARTED — two
//!   gaps: missing `alpha` kwarg ... AND no non-test caller." All three
//!   architectural claims (shape-strict, sub_f32 fast path, CPU fallback)
//!   are FALSE post-delegation. The delegation inherits broadcasting from
//!   `add_scaled_`.
//! - Line 415 (REQ status table): "impl at `ferrotorch-core/src/inplace.rs:248`
//!   mirrors only the `alpha=1, same-shape` slice ... missing `alpha` kwarg,
//!   missing broadcasting, no `sub_scaled_` sibling." Multiple stale claims:
//!   (a) wrong line, (b) NO LONGER same-shape only (delegation inherits
//!   broadcasting from `add_scaled_`), (c) `sub_scaled_` sibling EXISTS at
//!   `inplace.rs:266` per the SAME doc's REQ-11 row.
//!
//! ## Why this is a divergence (R-CITE-2)
//!
//! Per goal.md R-CITE-2: a file path without an accurate line is not a
//! citation. Per R-HONEST-4: when an audit reveals a wrong original commit,
//! correct the code AND document the correction. The delegation in
//! `2f792bfc5` updated REQ-11's row but not REQ-7's row nor the architecture
//! section that describes `sub_`'s body.
//!
//! Tracking: filed via crosslink (see report).

#![allow(clippy::missing_panics_doc)]

use std::fs;
use std::path::PathBuf;

fn read_inplace_design_doc() -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crate dir has parent")
        .join(".design/ferrotorch-core/inplace.md");
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"))
}

fn read_inplace_src() -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/inplace.rs");
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"))
}

/// Baseline: `pub fn sub_` body in `inplace.rs` is a one-line delegation
/// (verifies the production code state post-2f792bfc5).
#[test]
fn baseline_sub_in_inplace_is_delegation() {
    let src = read_inplace_src();
    assert!(
        src.contains("self.sub_scaled_(other, 1.0)"),
        "inplace.rs MUST contain the one-line delegation \
         `self.sub_scaled_(other, 1.0)` (per commit 2f792bfc5)"
    );
}

/// Divergence: `inplace.md` REQ-7 prose at line ~104 still cites
/// `inplace.rs:248` for `sub_`'s impl, but post-delegation `sub_` is at line
/// 295 (one-line body delegating to `sub_scaled_`). Per R-CITE-2: a file
/// path without an accurate line is not a citation.
#[test]
fn divergence_inplace_design_doc_cites_stale_sub_line() {
    let doc = read_inplace_design_doc();
    let src = read_inplace_src();

    // Find which line in the source actually has `pub fn sub_(&self`.
    let actual_sub_line = src
        .lines()
        .position(|l| l.contains("pub fn sub_(&self"))
        .map(|idx| idx + 1)
        .expect("inplace.rs must define `pub fn sub_(&self ...)`");

    // The doc cites :248 in two places (REQ-7 prose at ~104, REQ status row
    // at ~415). Check that the doc no longer cites :248 for sub_'s impl.
    let cites_248 = doc.lines().filter(|l| l.contains("inplace.rs:248")).count();
    assert_eq!(
        cites_248, 0,
        "inplace.md still cites `inplace.rs:248` ({cites_248} times) for \
         `sub_`'s impl, but the actual `pub fn sub_(&self ...)` is at line \
         {actual_sub_line}. Per goal.md R-CITE-2 a stale line is not a \
         citation; the delegation commit (2f792bfc5) should have updated \
         these doc citations."
    );
}

/// Divergence: `inplace.md` architecture section at lines 280-288 describes
/// `sub_(other)` as: (a) shape-strict, no broadcasting; (b) GPU f32 fast path
/// via `sub_f32`; (c) CPU fallback. Post-delegation NONE of these claims
/// hold — `sub_` is a one-line delegation through `sub_scaled_` →
/// `add_scaled_` which inherits broadcasting + `add_scaled_f32`'s GPU path.
///
/// This test asserts the doc no longer contains the stale architectural
/// claims; it fails until the prose is reconciled.
#[test]
fn divergence_inplace_design_doc_claims_sub_is_shape_strict_no_broadcast() {
    let doc = read_inplace_design_doc();
    let doc_lines: Vec<&str> = doc.lines().collect();

    let mut sub_arch_section_is_stale = false;
    let mut sub_f32_in_sub_arch = false;
    for (i, l) in doc_lines.iter().enumerate() {
        if l.contains("sub_(other)")
            && (l.contains("shape-strict") || l.contains("sub_f32") || l.contains("same-shape"))
        {
            sub_arch_section_is_stale = true;
        }
        if l.contains("sub_(other)") {
            let end = (i + 15).min(doc_lines.len());
            for ll in &doc_lines[i..end] {
                if ll.contains("sub_f32") {
                    sub_f32_in_sub_arch = true;
                }
            }
        }
    }

    let mut failures: Vec<String> = Vec::new();
    if sub_arch_section_is_stale {
        failures.push(
            "doc claims sub_(other) is `shape-strict, no broadcasting` / \
             `same-shape` only — FALSE post-delegation: `sub_` delegates \
             to `sub_scaled_` which delegates to `add_scaled_` and \
             inherits its broadcasting"
                .to_string(),
        );
    }
    if sub_f32_in_sub_arch {
        failures.push(
            "doc claims sub_(other) uses a `sub_f32` GPU fast path — \
             FALSE post-delegation: the GPU path now flows through \
             `add_scaled_f32` (sub_f32 may be orphaned for this call site)"
                .to_string(),
        );
    }
    // Check the REQ-7 row claim "no sub_scaled_ sibling".
    if doc.contains("no `sub_scaled_` sibling") || doc.contains("no sub_scaled_ sibling") {
        failures.push(
            "doc REQ-7 row claims `no sub_scaled_ sibling`, but \
             inplace.rs:266 has `pub fn sub_scaled_` — verified by REQ-11 \
             row of the same doc. The two rows contradict each other."
                .to_string(),
        );
    }
    // The REQ-7 row also claims "missing broadcasting" — but the delegation
    // gives sub_ broadcasting via add_scaled_.
    if doc.contains("missing broadcasting, no `sub_scaled_` sibling")
        || doc.contains("missing broadcasting, no sub_scaled_ sibling")
    {
        failures.push(
            "doc REQ-7 row claims sub_ has `missing broadcasting` — FALSE \
             post-delegation"
                .to_string(),
        );
    }

    assert!(
        failures.is_empty(),
        "inplace.md REQ-7 architecture / status prose contains stale claims \
         contradicted by commit 2f792bfc5's delegation:\n  - {}",
        failures.join("\n  - ")
    );
}
