//! Adversarial re-audit of #1668 (commit 3dfb173b6): the single-span S3
//! symbol-anchor validator added to
//! `ferrotorch-core/tests/divergence_cite_drift_generic.rs`
//! (`all_design_docs_single_span_anchors_resolve_at_head`,
//! `parse_symbol_anchors`, `resolve_symbol_anchor_files`,
//! `anchor_symbol_declared`).
//!
//! The builder reported 1070 anchors parsed, ALL valid, 0 stale, 0
//! unresolvable. This file pins TWO real flaws in that validator that the
//! green report masks:
//!
//!   1. RESOLVER SOUNDNESS HOLE (bare-basename cross-crate false-accept).
//!      The shipped `resolve_symbol_anchor_files` resolves a bare basename
//!      (e.g. `lib.rs`) to ALL files of that basename across EVERY crate, and
//!      `validate_symbol_anchor` accepts the anchor if the symbol is declared
//!      in ANY candidate. So an anchor whose INTENDED file does not declare
//!      the symbol still passes when a DIFFERENT crate's same-basename file
//!      happens to declare it — masking genuine drift.
//!      (cite: the resolver, divergence_cite_drift_generic.rs:1526-1562; the
//!      "found in ANY candidate" rule, :1639-1645.)
//!
//!   2. FALSE-NEGATIVE COVERAGE GAP. The parser
//!      (`parse_decl`, divergence_cite_drift_generic.rs:1310-1354) only matches
//!      keyword-led decls and `Type::method` assoc-fns; it explicitly drops the
//!      bare `<lowercase_sym> in <file>.rs` form (:1347-1353,
//!      "No keyword: ... Anything else ... is NOT a declaration"). Those bare
//!      forms are the DOMINANT corpus shape and are overwhelmingly GENUINE
//!      symbol anchors (e.g. `` `abs in complex_tensor.rs` ``,
//!      `` `addcmul_t in methods.rs` ``) that now rot unvalidated.
//!
//! Both tests are written against the REAL workspace source tree and real
//! `.design/` corpus. The expected values are grep-derived ground truth about
//! actual declarations (R-CHAR-3(b)), never literal-copied from the ferrotorch
//! validator's output.
//!
//! These tests reproduce the EXACT documented resolver rule from #1668 (a bare
//! basename resolves to all same-basename files; valid iff declared in any),
//! because the production helpers are private to the cite-drift test crate
//! file. Each reproduction is annotated with the upstream line it mirrors.

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

/// Faithful reproduction of `resolve_symbol_anchor_files`
/// (divergence_cite_drift_generic.rs:1526-1562): a bare basename resolves to
/// ALL files of that basename anywhere in the workspace.
fn resolve_symbol_anchor_files(
    index: &HashMap<String, Vec<PathBuf>>,
    root: &Path,
    file_as_written: &str,
) -> Vec<PathBuf> {
    let basename = file_as_written.rsplit('/').next().unwrap_or(file_as_written);
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

// ===========================================================================
// FINDING 1 — RESOLVER SOUNDNESS HOLE (bare-basename cross-crate false-accept)
// ===========================================================================

/// Divergence: #1668's `validate_symbol_anchor`
/// (`divergence_cite_drift_generic.rs:1639-1645`, "found in ANY candidate")
/// green-lights a single-span anchor `` `mod tests in lib.rs` `` whose INTENDED
/// crate file does NOT declare the symbol, because a DIFFERENT crate's
/// same-basename `lib.rs` does. A genuinely-stale anchor passes — the validator
/// cannot detect drift in any anchor that names a basename present in >= 2
/// crates (`lib.rs` x201, `mod.rs` x90, `error.rs` x41, ...).
///
/// GROUND TRUTH (grep-verified, R-CHAR-3(b)):
///   - `mod tests` IS declared in `ferrotorch-mps/src/lib.rs` (sibling).
///   - `mod tests` is NOT declared in `ferrotorch-rl/src/lib.rs` (intended).
/// So an anchor intending ferrotorch-rl's lib.rs is STALE-relative-to-intent
/// yet the shipped resolver returns Valid.
///
/// This test asserts the SOUND property the validator should hold: a bare
/// basename anchor must be validatable against the INTENDED file, not merely
/// "some file of that name somewhere". It FAILS because the shipped resolver
/// returns a candidate list that includes a sibling crate whose declaration
/// masks the intended file's absence.
/// Tracking: #1669 (blocker).
#[test]
#[ignore = "divergence: #1668 resolver soundness hole — bare-basename cross-crate false-accept masks drift; tracking #1669"]
fn divergence_bare_basename_resolver_false_accepts_cross_crate_sibling() {
    let root = workspace_root();
    let index = build_src_index(&root);

    // The anchor as it appears (parsed) in the corpus: `mod tests in lib.rs`.
    // (Real occurrence: `mod tests in lib.rs` is in the #1668-parsed universe.)
    let file_as_written = "lib.rs";
    let ident = "tests";

    // Resolve exactly as #1668 does: every lib.rs in the workspace.
    let candidates = resolve_symbol_anchor_files(&index, &root, file_as_written);
    assert!(
        candidates.len() > 1,
        "expected `lib.rs` to be a multi-crate basename (the collision precondition); \
         got {} candidate(s)",
        candidates.len()
    );

    // Ground truth: at least one sibling lib.rs declares `mod tests`, and at
    // least one OTHER lib.rs does not. This is precisely the masking condition.
    let declaring: Vec<&PathBuf> = candidates
        .iter()
        .filter(|p| fs::read_to_string(p).map(|s| mod_declared(&s, ident)).unwrap_or(false))
        .collect();
    let not_declaring: Vec<&PathBuf> = candidates
        .iter()
        .filter(|p| fs::read_to_string(p).map(|s| !mod_declared(&s, ident)).unwrap_or(false))
        .collect();
    assert!(
        !declaring.is_empty() && !not_declaring.is_empty(),
        "expected SOME lib.rs to declare `mod {ident}` and SOME to NOT — the \
         masking precondition. declaring={declaring:?} not_declaring(len)={}",
        not_declaring.len()
    );

    // The shipped validator's verdict: Valid (declared in ANY candidate).
    let shipped_validator_says_valid = candidates
        .iter()
        .any(|p| fs::read_to_string(p).map(|s| mod_declared(&s, ident)).unwrap_or(false));

    // Now model an anchor that INTENDED a specific crate whose lib.rs lacks
    // the symbol (ferrotorch-rl). A sound validator must report this intended
    // anchor as drift; #1668's any-candidate rule reports Valid.
    let intended = root.join("ferrotorch-rl/src/lib.rs");
    let intended_declares = fs::read_to_string(&intended)
        .map(|s| mod_declared(&s, ident))
        .unwrap_or(false);
    assert!(
        intended.exists(),
        "test fixture precondition: {} must exist",
        intended.display()
    );
    assert!(
        !intended_declares,
        "ground-truth precondition: ferrotorch-rl/src/lib.rs must NOT declare `mod {ident}`"
    );

    // SOUNDNESS ASSERTION (FAILS under #1668): a validator that cannot be
    // fooled by a cross-crate sibling would NOT call this Valid when the
    // intended file lacks the symbol. The shipped any-candidate rule does.
    assert!(
        !shipped_validator_says_valid,
        "RESOLVER SOUNDNESS HOLE (#1668): the single-span validator green-lights \
         `mod {ident} in {file_as_written}` because a SIBLING crate's lib.rs \
         declares it, even though the intended file \
         ferrotorch-rl/src/lib.rs does NOT. Drift in any anchor naming a \
         multi-crate basename ({} lib.rs candidates here) is undetectable. \
         A sound validator must bind the anchor to its intended file (e.g. \
         require a crate-qualified path, or fail on multi-candidate basenames \
         where not all candidates agree).",
        candidates.len(),
    );
}

// ===========================================================================
// FINDING 2 — FALSE-NEGATIVE COVERAGE GAP (bare-lowercase anchors unvalidated)
// ===========================================================================

/// Divergence: #1668's parser (`parse_decl`,
/// `divergence_cite_drift_generic.rs:1347-1353`) drops the bare
/// `<lowercase_sym> in <file>.rs` form ("No keyword: only accept the bare
/// `<Type>::<method>` ... Anything else ... is NOT a declaration"). The builder
/// reports 1070 parsed, but the non-upstream `` `<x> in <file>.rs` `` corpus is
/// ~3879 spans — the bare forms are the dominant remainder and are
/// OVERWHELMINGLY genuine symbol anchors, not prose. They now rot unvalidated:
/// a renamed `abs`, `accuracy_score`, `addcmul_t`, `add_f32` etc. behind a bare
/// anchor passes silently. #1668 closes only ~1/3 of the hole it claims.
///
/// GROUND TRUTH (grep-verified, R-CHAR-3(b)): each sampled bare anchor's symbol
/// IS a real declaration in a file of the named basename — proving the form is
/// a GENUINE anchor that SHOULD be validated, not prose to be safely skipped.
///
/// This test FAILS by asserting that the genuine-anchor share of the excluded
/// bare forms is small (i.e. that exclusion is safe). It is not: the share is
/// high, so the assertion that "the excluded forms are mostly prose" is false.
/// Tracking: #1669 (blocker).
#[test]
#[ignore = "divergence: #1668 false-negative coverage gap — bare-lowercase anchors (~2/3 of corpus) unvalidated; tracking #1669"]
fn divergence_bare_lowercase_anchors_are_genuine_and_unvalidated() {
    let root = workspace_root();
    let index = build_src_index(&root);

    // A grep-derived sample of bare `<lowercase_sym> in <file>.rs` corpus spans
    // that #1668's parser EXCLUDES (no kw, no `Type::method`). Each pair is the
    // (symbol, file-as-written) exactly as it appears in `.design/`.
    let sampled_bare_anchors: &[(&str, &str)] = &[
        ("abs", "ferrotorch-core/src/complex_tensor.rs"),
        ("accuracy_score", "ferrotorch-ml/src/metrics.rs"),
        ("activation", "ferrotorch-nn/src/lib.rs"),
        ("adamw", "ferrotorch-optim/src/lib.rs"),
        ("addcmul_t", "methods.rs"),
        ("addcdiv_t", "methods.rs"),
        ("add_f32", "backend_impl.rs"),
        ("add_f64", "backend_impl.rs"),
        ("add", "grad_fns/arithmetic.rs"),
        ("align_to", "ferrotorch-core/src/named_tensor.rs"),
    ];

    // Confirm none of these would have been parsed by #1668 (they are bare,
    // lowercase, no keyword, no `::`) — i.e. they are genuinely in the EXCLUDED
    // set, not double-counted in the 1070.
    for (sym, _file) in sampled_bare_anchors {
        assert!(
            !sym.contains("::")
                && sym.chars().next().is_some_and(|c| c.is_ascii_lowercase()),
            "fixture precondition: `{sym}` must be bare-lowercase (excluded by #1668's parser)"
        );
    }

    // How many of the sampled bare anchors are GENUINE (the symbol is really
    // declared in a file of that basename)? If exclusion were safe, almost all
    // would be prose (genuine count ~ 0).
    let mut genuine = 0usize;
    let mut prose = Vec::new();
    for (sym, file) in sampled_bare_anchors {
        let candidates = resolve_symbol_anchor_files(&index, &root, file);
        let is_genuine = candidates.iter().any(|p| {
            fs::read_to_string(p)
                .map(|s| fn_declared(&s, sym) || s.contains(&format!("struct {sym}")) )
                .unwrap_or(false)
        });
        if is_genuine {
            genuine += 1;
        } else {
            prose.push((*sym, *file));
        }
    }

    // COVERAGE ASSERTION (FAILS under #1668): if dropping the bare form were
    // safe, the excluded spans would be mostly prose. They are not — the vast
    // majority are real declarations. We assert the (false) "mostly prose"
    // premise so the test fails loudly, pinning the coverage gap.
    let n = sampled_bare_anchors.len();
    assert!(
        genuine <= n / 4,
        "FALSE-NEGATIVE COVERAGE GAP (#1668): {genuine}/{n} sampled bare \
         `<sym> in <file>.rs` anchors are GENUINE symbol declarations (not \
         prose), so #1668's exclusion of the bare form silently un-validates \
         the dominant anchor family (~2700 of ~3879 non-upstream spans; only \
         1070 are parsed). A renamed bare-anchored symbol rots undetected. The \
         parser must extend to the bare `<lowercase_sym> in <path>.rs` form \
         (resolvable against a real declaration) instead of dropping it. \
         Non-genuine (truly prose) sample: {prose:?}",
    );
}
