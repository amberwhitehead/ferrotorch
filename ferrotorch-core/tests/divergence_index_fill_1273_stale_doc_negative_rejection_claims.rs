//! Divergence: commit `3d82aec1e` (#1273) flipped the impl of `index_fill`
//! to WRAP negative indices per upstream, but THREE doc-sites still claim
//! "negative indices are rejected" — TWO in the design doc and ONE in the
//! `Tensor::index_fill_t` rustdoc that users actually read.
//!
//! Stale claims at HEAD (post-#1273):
//!
//!   1. `ferrotorch-core/src/methods.rs:680-681` (Tensor::index_fill_t
//!      rustdoc):
//!        "Negative index values are rejected (R-DEV-1 narrower contract,
//!         matching the rest of ferrotorch's `IntTensor` index validation)."
//!      This is the PUBLIC API doc. A user reading it would write code
//!      assuming negative indices error, then be surprised when they
//!      silently wrap.
//!
//!   2. `.design/ferrotorch-core/grad_fns/indexing.md:201` (REQ-8 prose):
//!        "Negative index values rejected (ferrotorch narrower contract
//!         shared with the rest of the IntTensor index family)."
//!
//!   3. `.design/ferrotorch-core/grad_fns/indexing.md:653` (Parity contract
//!      table for `index_fill`):
//!        "Negative index values: ferrotorch rejects (narrower contract
//!         shared with the rest of the IntTensor index family); upstream
//!         accepts via wrap."
//!      AND on the same line:
//!        "0-d input: upstream unsqueezes to 1-d at `:1917`; ferrotorch
//!         rejects (#1256 narrower-contract gap)."
//!      Both halves of this sentence are stale post-#1272 and post-#1273.
//!
//! Per goal.md R-CITE-2 + R-HONEST-2 every behavioral claim in a SHIPPED
//! row must reflect the actual impl. The #1274 Haiku fixer that converted
//! REQ-8 line-number cites to symbol anchors did NOT sweep the
//! corresponding semantic claims in the prose / contract table / public
//! API rustdoc, even though they sit in the same REQ-8 region and the
//! same file (indexing.md) the Haiku fixer was working in.
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

#[test]
fn methods_rs_index_fill_t_rustdoc_must_not_claim_negative_rejection() {
    let root = workspace_root();
    let p = root.join("ferrotorch-core/src/methods.rs");
    let text = fs::read_to_string(&p).expect("read methods.rs");

    // Per goal.md R-CHAR-3 we don't hardcode the symbol's line. We scan
    // for the divergent claim near the index_fill_t fn declaration.
    let fn_pos = text
        .find("pub fn index_fill_t")
        .expect("Tensor::index_fill_t must exist in methods.rs");
    // Walk backward to the start of the preceding doc-comment block (look
    // for the first non-doc/non-attr line) — bounded by 200 lines of /// .
    let prefix = &text[..fn_pos];
    let doc_start = prefix.rfind("/// `torch.Tensor.index_fill").unwrap_or(0);
    let docblock = &text[doc_start..fn_pos];

    let stale_phrases = [
        "Negative index values are rejected",
        "narrower contract",
    ];
    let found: Vec<&str> = stale_phrases
        .iter()
        .filter(|s| docblock.contains(*s))
        .copied()
        .collect();

    assert!(
        found.is_empty(),
        "Tensor::index_fill_t rustdoc in methods.rs still carries stale \
         claims that contradict the #1273 impl (which now wraps negative \
         indices per upstream IndexKernel.cpp:224-229):\n  {:?}\n\n\
         Public API doc must match impl. Fix by replacing the rustdoc to \
         say negative indices wrap (in range) and OOB raises IndexError.",
        found
    );
}

#[test]
fn indexing_md_must_not_claim_negative_index_rejection() {
    let root = workspace_root();
    let p = root.join(".design/ferrotorch-core/grad_fns/indexing.md");
    let text = fs::read_to_string(&p).expect("read indexing.md");

    // Two stale claims in the same file at different sites.
    let stale_phrases = [
        // REQ-8 prose section
        "Negative index values rejected (ferrotorch narrower contract",
        // Parity contract table row for index_fill
        "Negative index values: ferrotorch rejects",
    ];
    let found: Vec<&str> = stale_phrases
        .iter()
        .filter(|s| text.contains(*s))
        .copied()
        .collect();

    assert!(
        found.is_empty(),
        "indexing.md still carries stale REJECTS claims post-#1273:\n  {:?}\n\n\
         Per R-HONEST-2 every SHIPPED row's behavior claims must reflect \
         the impl. Fix the prose at REQ-8 + the parity-contract row to \
         describe the wrap semantics.",
        found
    );
}

#[test]
fn indexing_md_must_not_claim_zero_d_rejection() {
    let root = workspace_root();
    let p = root.join(".design/ferrotorch-core/grad_fns/indexing.md");
    let text = fs::read_to_string(&p).expect("read indexing.md");

    // Parity contract table claim that 0-d is rejected — stale post-#1272.
    let stale = "ferrotorch rejects (#1256 narrower-contract gap)";
    assert!(
        !text.contains(stale),
        "indexing.md parity-contract row for index_fill still claims 0-d \
         input is rejected, but #1272 SHIPPED the unsqueeze path. The \
         entire `0-d input: ... ferrotorch rejects (#1256 ...)` sentence \
         is stale and must be updated to describe the SHIPPED behavior."
    );
}
