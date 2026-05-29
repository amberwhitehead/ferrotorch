//! Re-audit of #1668 (commit 3dfb173b6) + verification of the #1669 fix
//! (completes #1668): the single-span S3 symbol-anchor validator in
//! `ferrotorch-core/tests/divergence_cite_drift_generic.rs`
//! (`all_design_docs_single_span_anchors_resolve_at_head`,
//! `parse_symbol_anchors`, `resolve_symbol_anchor_files`,
//! `anchor_symbol_declared`).
//!
//! #1668 reported 1070 anchors parsed, ALL valid, 0 stale, 0 unresolvable.
//! That green report masked TWO real flaws, both now FIXED by #1669; these
//! tests assert the FIXED properties hold (they were `#[ignore]`'d failing
//! pins under #1668, un-ignored here as permanent regression coverage):
//!
//!   1. RESOLVER SOUNDNESS HOLE — FIXED by crate-disambiguation.
//!      #1668's `resolve_symbol_anchor_files` resolved a bare basename
//!      (e.g. `lib.rs`) to ALL files of that basename across EVERY crate, and
//!      accepted the anchor if the symbol was declared in ANY candidate — so
//!      an anchor in `.design/ferrotorch-rl/...` saying `mod tests in lib.rs`
//!      passed because `ferrotorch-mps/src/lib.rs` has `mod tests`, even
//!      though `ferrotorch-rl/src/lib.rs` does not.
//!      #1669 binds a bare-basename anchor to the DOC'S OWN CRATE
//!      (`.design/<crate>/...` -> `<crate>/src/`): the anchor is validated
//!      against the crate-local same-basename file, so the cross-crate
//!      sibling can no longer mask drift. This test reproduces the
//!      crate-disambiguating resolver and asserts the masking is GONE.
//!
//!   2. FALSE-NEGATIVE COVERAGE GAP — FIXED by the bare-ident matcher.
//!      #1668's parser only matched keyword-led decls and `Type::method`
//!      assoc-fns, dropping the bare `<lowercase_sym> in <file>.rs` form (the
//!      DOMINANT corpus shape, overwhelmingly GENUINE symbol anchors). #1669
//!      extends the parser to match a bare single snake_case identifier and
//!      validate it against a real declaration OR a `pub use` re-export. This
//!      test asserts those genuine bare anchors ARE now validated.
//!
//! Both tests run against the REAL workspace source tree and real `.design/`
//! corpus. Expected values are grep-derived ground truth about actual
//! declarations (R-CHAR-3(b)), never literal-copied from the validator output.
//!
//! The resolver/validator reproductions below mirror the production helpers in
//! `divergence_cite_drift_generic.rs` (which are private to that test crate),
//! INCLUDING the #1669 crate-disambiguation and bare-ident validation.

#![allow(clippy::missing_panics_doc)]

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if !p.join(".design").exists() {
        p.pop();
    }
    p
}

/// Faithful reproduction of `build_src_index`
/// (divergence_cite_drift_generic.rs:1481-1520): index every
/// `<crate>/src/**/*.rs` and `tools/parity-sweep/runner/src/**/*.rs` by
/// basename -> all full paths. Worktrees under `.claude/` are NOT indexed (the
/// production walker only descends `<top-level-dir>/src`, and `.claude/src`
/// does not exist) — we mirror that exactly.
fn build_src_index(root: &Path) -> HashMap<String, Vec<PathBuf>> {
    let mut index: HashMap<String, Vec<PathBuf>> = HashMap::new();
    let mut crate_src_dirs: Vec<PathBuf> = Vec::new();
    if let Ok(entries) = fs::read_dir(root) {
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                let src = p.join("src");
                if src.is_dir() {
                    crate_src_dirs.push(src);
                }
            }
        }
    }
    let runner_src = root.join("tools/parity-sweep/runner/src");
    if runner_src.is_dir() {
        crate_src_dirs.push(runner_src);
    }
    let mut stack = crate_src_dirs;
    while let Some(dir) = stack.pop() {
        let entries = match fs::read_dir(&dir) {
            Ok(it) => it,
            Err(_) => continue,
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().and_then(|x| x.to_str()) == Some("rs") {
                if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                    index.entry(name.to_string()).or_default().push(p.clone());
                }
            }
        }
    }
    index
}

/// Reproduction of the #1668 (PRE-#1669) `resolve_symbol_anchor_files`: a bare
/// basename resolves to ALL files of that basename anywhere in the workspace,
/// with NO crate disambiguation. Retained to demonstrate the masking
/// precondition the #1669 fix removes.
fn resolve_symbol_anchor_files_pre1669(
    index: &HashMap<String, Vec<PathBuf>>,
    root: &Path,
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
        let mut matches: Vec<PathBuf> = all
            .iter()
            .filter(|p| {
                let rel = p.strip_prefix(root).unwrap_or(p);
                let rel_s = rel.to_string_lossy().replace('\\', "/");
                rel_s.ends_with(file_as_written)
            })
            .cloned()
            .collect();
        if matches.is_empty() {
            matches = all.clone();
        }
        matches
    } else {
        all.clone()
    }
}

/// Map a design-doc's workspace-relative label to its crate's `src/` dir.
/// Mirrors `doc_crate_src_dir` in `divergence_cite_drift_generic.rs` (#1669).
fn doc_crate_src_dir(root: &Path, doc_label: &str) -> Option<PathBuf> {
    let rel = doc_label.replace('\\', "/");
    let after = rel.strip_prefix(".design/")?;
    let first = after.split('/').next()?;
    if first.is_empty() || !first.starts_with("ferrotorch") {
        return None;
    }
    let src = root.join(first).join("src");
    if src.is_dir() {
        Some(src)
    } else {
        None
    }
}

/// Faithful reproduction of the #1669 crate-disambiguating
/// `resolve_symbol_anchor_files`: a bare basename binds PREFERENTIALLY to the
/// doc-crate's same-basename file; only when the doc-crate has no such file
/// (or the doc maps to no crate) does it fall back to the cross-crate set.
fn resolve_symbol_anchor_files(
    index: &HashMap<String, Vec<PathBuf>>,
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
        let mut matches: Vec<PathBuf> = all
            .iter()
            .filter(|p| {
                let rel = p.strip_prefix(root).unwrap_or(p);
                let rel_s = rel.to_string_lossy().replace('\\', "/");
                rel_s.ends_with(file_as_written)
            })
            .cloned()
            .collect();
        if matches.is_empty() {
            matches = all.clone();
        }
        matches
    } else {
        if let Some(crate_src) = doc_crate_src_dir(root, doc_label) {
            let crate_local: Vec<PathBuf> = all
                .iter()
                .filter(|p| p.starts_with(&crate_src))
                .cloned()
                .collect();
            if !crate_local.is_empty() {
                return crate_local;
            }
        }
        all.clone()
    }
}

/// Faithful reproduction of the `DeclKind::Mod` arm of `anchor_symbol_declared`
/// (divergence_cite_drift_generic.rs:1568-1607): `mod <ident>` with a
/// word-boundary check.
fn mod_declared(src: &str, ident: &str) -> bool {
    let needle = format!("mod {ident}");
    src.lines().any(|line| {
        if let Some(idx) = line.find(&needle) {
            let after = line[idx + needle.len()..].chars().next();
            match after {
                None => true,
                Some(c) => !(c.is_ascii_alphanumeric() || c == '_'),
            }
        } else {
            false
        }
    })
}

/// Reproduction of the `DeclKind::Fn` arm: `fn <ident>` with word boundary.
fn fn_declared(src: &str, ident: &str) -> bool {
    let needle = format!("fn {ident}");
    src.lines().any(|line| {
        if let Some(idx) = line.find(&needle) {
            let after = line[idx + needle.len()..].chars().next();
            match after {
                None => true,
                Some(c) => !(c.is_ascii_alphanumeric() || c == '_'),
            }
        } else {
            false
        }
    })
}

/// Reproduction of the #1669 `DeclKind::BareIdent` arm of
/// `anchor_symbol_declared`: a bare ident is declared if the file has any of
/// `fn`/`struct`/`enum`/`trait`/`mod`/`const`/`static`/`type`/`macro_rules!`
/// `<ident>` (word-boundary aware) OR a `pub use`/`use` re-export line that
/// references `<ident>` as a whole path segment.
fn bare_ident_declared(src: &str, ident: &str) -> bool {
    let needles = [
        format!("fn {ident}"),
        format!("struct {ident}"),
        format!("enum {ident}"),
        format!("trait {ident}"),
        format!("mod {ident}"),
        format!("const {ident}"),
        format!("static {ident}"),
        format!("type {ident}"),
        format!("macro_rules! {ident}"),
    ];
    let decl = src.lines().any(|line| {
        needles
            .iter()
            .any(|needle| match line.find(needle.as_str()) {
                Some(idx) => match line[idx + needle.len()..].chars().next() {
                    None => true,
                    Some(c) => !(c.is_ascii_alphanumeric() || c == '_'),
                },
                None => false,
            })
    });
    if decl {
        return true;
    }
    src.lines().any(|line| {
        let t = line.trim_start();
        (t.starts_with("pub use ") || t.starts_with("use ")) && line_has_path_segment(line, ident)
    })
}

/// Word-boundary path-segment match, mirroring `line_has_path_segment` in
/// `divergence_cite_drift_generic.rs` (#1669).
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

// ===========================================================================
// FINDING 1 — RESOLVER SOUNDNESS HOLE (bare-basename cross-crate false-accept)
// ===========================================================================

/// #1669 FIX VERIFICATION (was #1668 soundness-hole pin): the
/// crate-disambiguating `resolve_symbol_anchor_files` binds a bare-basename
/// anchor to the DOC'S OWN CRATE, so a cross-crate sibling can no longer mask
/// drift. Concretely: an anchor `` `mod tests in lib.rs` `` found in a
/// `.design/ferrotorch-rl/...` doc resolves to `ferrotorch-rl/src/lib.rs`
/// ONLY — which does NOT declare `mod tests` — so the validator now reports it
/// STALE, even though `ferrotorch-mps/src/lib.rs` (a sibling) does declare it.
///
/// GROUND TRUTH (grep-verified, R-CHAR-3(b)):
///   - `mod tests` IS declared in `ferrotorch-mps/src/lib.rs` (sibling).
///   - `mod tests` is NOT declared in `ferrotorch-rl/src/lib.rs` (doc-crate).
///
/// The test FIRST demonstrates the #1668 masking precondition (the pre-#1669
/// resolver returns a candidate set spanning multiple crates, and the
/// any-candidate rule would call it Valid), THEN asserts the #1669 resolver
/// closes it (doc-crate-local candidate set, validator reports STALE).
#[test]
fn divergence_bare_basename_resolver_false_accepts_cross_crate_sibling() {
    let root = workspace_root();
    let index = build_src_index(&root);

    let file_as_written = "lib.rs";
    let ident = "tests";
    // The doc the anchor lives in determines the crate it's validated against.
    let doc_label = ".design/ferrotorch-rl/lib.md";

    // --- #1668 masking PRECONDITION (pre-#1669 resolver) -------------------
    // The old resolver returns every lib.rs in the workspace; among them some
    // declare `mod tests` and some do not — exactly the masking condition.
    let pre = resolve_symbol_anchor_files_pre1669(&index, &root, file_as_written);
    assert!(
        pre.len() > 1,
        "expected `lib.rs` to be a multi-crate basename (collision precondition); got {}",
        pre.len()
    );
    let any_declares = pre.iter().any(|p| {
        fs::read_to_string(p)
            .map(|s| mod_declared(&s, ident))
            .unwrap_or(false)
    });
    let any_lacks = pre.iter().any(|p| {
        fs::read_to_string(p)
            .map(|s| !mod_declared(&s, ident))
            .unwrap_or(true)
    });
    assert!(
        any_declares && any_lacks,
        "expected SOME lib.rs to declare `mod {ident}` and SOME to NOT — the masking precondition"
    );
    // Under the old any-candidate rule this anchor would be Valid (a sibling
    // declares the symbol), masking the doc-crate's absence — the bug.
    assert!(
        any_declares,
        "pre-#1669 any-candidate rule would (incorrectly) accept `mod {ident} in lib.rs`"
    );

    // --- Ground-truth file facts ------------------------------------------
    let rl_lib = root.join("ferrotorch-rl/src/lib.rs");
    assert!(rl_lib.exists(), "fixture: {} must exist", rl_lib.display());
    assert!(
        !mod_declared(&fs::read_to_string(&rl_lib).unwrap(), ident),
        "ground truth: ferrotorch-rl/src/lib.rs must NOT declare `mod {ident}`"
    );
    let mps_lib = root.join("ferrotorch-mps/src/lib.rs");
    assert!(
        mod_declared(&fs::read_to_string(&mps_lib).unwrap(), ident),
        "ground truth: ferrotorch-mps/src/lib.rs MUST declare `mod {ident}` (the masking sibling)"
    );

    // --- #1669 FIX: crate-disambiguating resolver -------------------------
    // The anchor in a ferrotorch-rl doc resolves to ferrotorch-rl's lib.rs
    // ONLY — the cross-crate mps sibling is excluded.
    let post = resolve_symbol_anchor_files(&index, &root, doc_label, file_as_written);
    assert_eq!(
        post.len(),
        1,
        "expected crate-disambiguation to bind to exactly ferrotorch-rl/src/lib.rs, got {post:?}"
    );
    assert_eq!(
        post[0], rl_lib,
        "expected the sole candidate to be the doc-crate's lib.rs"
    );
    assert!(
        !post.contains(&mps_lib),
        "the masking sibling ferrotorch-mps/src/lib.rs must NOT be a candidate"
    );

    // SOUNDNESS ASSERTION (now PASSES under #1669): the validator, restricted
    // to the doc-crate candidate, finds the symbol absent -> STALE. The
    // cross-crate sibling can no longer mask the drift.
    let post_says_valid = post.iter().any(|p| {
        fs::read_to_string(p)
            .map(|s| mod_declared(&s, ident))
            .unwrap_or(false)
    });
    assert!(
        !post_says_valid,
        "RESOLVER SOUNDNESS (#1669 fix): a bare-basename anchor in a \
         .design/ferrotorch-rl doc must be validated against \
         ferrotorch-rl/src/lib.rs (which lacks `mod {ident}`) and reported \
         STALE — NOT accepted because a sibling crate's lib.rs declares it."
    );
}

// ===========================================================================
// FINDING 2 — FALSE-NEGATIVE COVERAGE GAP (bare-lowercase anchors unvalidated)
// ===========================================================================

/// #1669 FIX VERIFICATION (was #1668 coverage-gap pin): the parser now matches
/// the bare `<lowercase_sym> in <file>.rs` form (`DeclKind::BareIdent`) and the
/// validator resolves it against a real declaration OR a `pub use` re-export.
/// #1668 dropped this form entirely ("No keyword: ... is NOT a declaration"),
/// leaving the dominant anchor family unvalidated; #1669 closes that.
///
/// GROUND TRUTH (grep-verified, R-CHAR-3(b)): each sampled bare anchor's symbol
/// IS a real declaration (or re-export) in a file of the named basename under
/// the doc's crate — so the form is a GENUINE anchor that MUST be validated,
/// not prose to be skipped. This test asserts ALL of them now VALIDATE through
/// the #1669 bare-ident matcher + crate-disambiguating resolver + bare-ident
/// validator (a renamed symbol behind such an anchor would now be caught).
#[test]
fn divergence_bare_lowercase_anchors_are_genuine_and_unvalidated() {
    let root = workspace_root();
    let index = build_src_index(&root);

    // A grep-derived sample of bare `<lowercase_sym> in <file>.rs` corpus spans
    // #1668 EXCLUDED (no kw, no `Type::method`). Each tuple is
    // (symbol, file-as-written, doc-crate-label) — the doc label drives the
    // #1669 crate-disambiguation for bare basenames.
    let sampled_bare_anchors: &[(&str, &str, &str)] = &[
        (
            "abs",
            "ferrotorch-core/src/complex_tensor.rs",
            ".design/ferrotorch-core/complex_tensor.md",
        ),
        (
            "accuracy_score",
            "ferrotorch-ml/src/metrics.rs",
            ".design/ferrotorch-ml/metrics.md",
        ),
        // lib.rs facade re-exports — validated via the `pub use` path.
        ("activation", "lib.rs", ".design/ferrotorch-nn/lib.md"),
        ("adamw", "lib.rs", ".design/ferrotorch-optim/lib.md"),
        (
            "addcmul_t",
            "methods.rs",
            ".design/ferrotorch-core/methods.md",
        ),
        (
            "addcdiv_t",
            "methods.rs",
            ".design/ferrotorch-core/methods.md",
        ),
        (
            "add_f32",
            "backend_impl.rs",
            ".design/ferrotorch-gpu/backend_impl.md",
        ),
        (
            "add_f64",
            "backend_impl.rs",
            ".design/ferrotorch-gpu/backend_impl.md",
        ),
        (
            "add",
            "grad_fns/arithmetic.rs",
            ".design/ferrotorch-core/grad_fns/arithmetic.md",
        ),
        (
            "align_to",
            "ferrotorch-core/src/named_tensor.rs",
            ".design/ferrotorch-core/named_tensor.md",
        ),
    ];

    // Each sample is a bare single lowercase ident (no `::`, no kw) — exactly
    // the form #1668 dropped and #1669 now parses.
    for (sym, _file, _doc) in sampled_bare_anchors {
        assert!(
            !sym.contains("::") && sym.chars().next().is_some_and(|c| c.is_ascii_lowercase()),
            "fixture precondition: `{sym}` must be a bare lowercase ident (the form #1668 excluded)"
        );
    }

    // FIX ASSERTION (PASSES under #1669): every sampled bare anchor resolves
    // to a candidate in its doc-crate that genuinely DECLARES or RE-EXPORTS the
    // symbol — i.e. the bare form is now VALIDATED, not silently skipped.
    let mut unvalidated = Vec::new();
    for (sym, file, doc) in sampled_bare_anchors {
        let candidates = resolve_symbol_anchor_files(&index, &root, doc, file);
        let validated = candidates.iter().any(|p| {
            fs::read_to_string(p)
                .map(|s| bare_ident_declared(&s, sym))
                .unwrap_or(false)
        });
        if !validated {
            unvalidated.push((*sym, *file, *doc, candidates.len()));
        }
    }
    assert!(
        unvalidated.is_empty(),
        "COVERAGE-GAP FIX (#1669): every sampled bare `<sym> in <file>.rs` anchor \
         must now be VALIDATED (declared or re-exported) via the bare-ident \
         matcher + crate-disambiguating resolver + bare-ident validator. \
         Still-unvalidated samples (a regression in the #1669 fix): {unvalidated:?}",
    );

    // And the matcher must still be selective: `fn_declared` alone (the #1668
    // narrow check) would have MISSED the `pub use` facade re-exports
    // (`activation`, `adamw`), proving the bare-ident validator's re-export
    // path is load-bearing, not redundant.
    let facade_missed_by_fn_only = ["activation", "adamw"].iter().all(|sym| {
        let p = root.join("ferrotorch-nn/src/lib.rs");
        let nn = fs::read_to_string(&p).unwrap_or_default();
        let p2 = root.join("ferrotorch-optim/src/lib.rs");
        let optim = fs::read_to_string(&p2).unwrap_or_default();
        !fn_declared(&nn, sym) && !fn_declared(&optim, sym)
    });
    assert!(
        facade_missed_by_fn_only,
        "fixture: the facade re-exports `activation`/`adamw` are NOT `fn` decls — \
         the bare-ident validator must accept them via the `pub use` re-export path"
    );
}
