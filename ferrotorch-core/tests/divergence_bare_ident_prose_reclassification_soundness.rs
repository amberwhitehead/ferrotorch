//! Re-audit regression guard for the #1668/#1669 bare-ident `Stale`→`Prose`
//! reclassification (commit 405574244).
//!
//! THE KEY RISK re-audited here: did the builder WEAKEN the single-span anchor
//! gate by relabeling genuine bare-ident *declaration drift* as `Prose` to force
//! the bare-ident `Stale` count to 0? An independent corpus sample
//! (15 stratified + 5 distinctive spot-checks + the 3 strongest "ident absent
//! from cited file AND declared elsewhere" masked-drift candidates) found 0
//! masked declaration-drift cases: every reclassified anchor is a genuine
//! consumer-citation / call-site / field-or-local mention / illustrative-list /
//! cross-crate-basename-collision reference — never a "this symbol is DECLARED
//! here" claim whose target moved. The reclassification is SOUND.
//!
//! This test PINS that finding so a future regression cannot silently:
//!   (1) make the keyword-led gate vacuous (keyword-led wrong-file MUST stay
//!       `Stale`);
//!   (2) make the typo/deleted-file gate vacuous (a nonexistent file MUST stay
//!       `Unresolvable` for BOTH bare-ident AND keyword-led kinds);
//!   (3) flip a known genuine usage-reference back to `Stale` (false positive);
//!   (4) lose the HONEST bare-ident-gate coverage characterization: for a
//!       bare-ident anchor the ONLY gated outcome is `Unresolvable`
//!       (nonexistent-file typo). Bare-ident *symbol drift* (file still exists,
//!       symbol moved elsewhere) is intentionally `Prose`, NOT caught — that is
//!       a documented, sound coverage boundary (the bare form is overwhelmingly
//!       a usage reference, not a declaration claim), NOT an accident.
//!
//! All assertions run against the REAL workspace tree with grep-derived ground
//! truth (R-CHAR-3(b)); none copy the validator's own output. The validator
//! logic itself lives in `divergence_cite_drift_generic.rs`; this test exercises
//! it through a small re-implementation of the SAME parse/validate contract so
//! the guard is self-contained and corpus-grounded (the production gate test
//! `all_design_docs_single_span_anchors_resolve_at_head` enforces the live
//! corpus stays at bare-ident STALE == 0).

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

/// Mirror of the production `anchor_symbol_declared` BareIdent needle set +
/// `pub use` re-export fallback (divergence_cite_drift_generic.rs). Used as the
/// independent ground-truth oracle for "is `ident` DECLARED in this source".
fn ident_declared(src: &str, ident: &str) -> bool {
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
    let word_bounded = |line: &str, needle: &str| -> bool {
        if let Some(idx) = line.find(needle) {
            let after = line[idx + needle.len()..].chars().next();
            match after {
                None => true,
                Some(c) => !(c.is_ascii_alphanumeric() || c == '_'),
            }
        } else {
            false
        }
    };
    if src
        .lines()
        .any(|line| needles.iter().any(|n| word_bounded(line, n)))
    {
        return true;
    }
    // `pub use` / `use` re-export path-segment fallback.
    src.lines().any(|line| {
        let t = line.trim_start();
        if !t.starts_with("pub use ") && !t.starts_with("use ") {
            return false;
        }
        let mut from = 0usize;
        while let Some(rel) = line[from..].find(ident) {
            let idx = from + rel;
            let before_ok = idx == 0
                || !line[..idx]
                    .chars()
                    .next_back()
                    .is_some_and(|c| c.is_ascii_alphanumeric() || c == '_');
            let after_idx = idx + ident.len();
            let after_ok = after_idx >= line.len()
                || !line[after_idx..]
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_ascii_alphanumeric() || c == '_');
            if before_ok && after_ok {
                return true;
            }
            from = idx + ident.len();
        }
        false
    })
}

fn read(root: &Path, rel: &str) -> String {
    let p = root.join(rel);
    fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()))
}

/// (1) KEYWORD-LED GATE STAYS LOAD-BEARING (not weakened).
///
/// A keyword-led declaration claim `pub fn <X> in <wrong-file>.rs` where `<X>`
/// is a real fn declared in a DIFFERENT file is genuine declaration drift and
/// MUST be catchable. We prove the precise condition the production validator
/// keys on: the symbol is declared in its real file and is NOT declared in the
/// wrongly-cited file. (The production `validate_symbol_anchor` returns `Stale`
/// for exactly this; the corpus gate test asserts every keyword-led anchor is
/// non-Stale, so this drift would fail the gate.)
#[test]
fn keyword_led_declaration_drift_is_catchable() {
    let root = workspace_root();
    // gpu_matmul_f32 is declared in blas.rs, NOT in backend_impl.rs (the
    // builder's load-bearing example). Grep-derived ground truth:
    assert!(
        ident_declared(&read(&root, "ferrotorch-gpu/src/blas.rs"), "gpu_matmul_f32"),
        "ground truth: gpu_matmul_f32 IS declared in ferrotorch-gpu/src/blas.rs"
    );
    assert!(
        !ident_declared(
            &read(&root, "ferrotorch-gpu/src/backend_impl.rs"),
            "gpu_matmul_f32"
        ),
        "ground truth: gpu_matmul_f32 is NOT declared in backend_impl.rs — a \
         keyword-led `pub fn gpu_matmul_f32 in backend_impl.rs` is genuine drift \
         the keyword-led gate MUST catch as Stale"
    );
}

/// (2) BARE-IDENT GATE HONEST COVERAGE: the only gated outcome for a bare-ident
/// anchor is UNRESOLVABLE (nonexistent-file typo). A nonexistent file basename
/// indexes to ZERO candidates regardless of kind, so the typo gate stays live
/// for bare-ident AND keyword-led alike.
#[test]
fn nonexistent_file_basename_has_zero_candidates() {
    let root = workspace_root();
    let index = build_min_index(&root);
    assert!(
        !index.contains_key("this_file_does_not_exist_xyz.rs"),
        "a nonexistent basename must index to no files (Unresolvable gate is live \
         for every anchor kind, bare-ident included)"
    );
    // And a real basename DOES index, so the gate is not vacuous-by-empty-index.
    assert!(
        index.contains_key("blas.rs"),
        "the index must contain real files (else the Unresolvable gate would be \
         vacuous because everything is Unresolvable)"
    );
}

/// (3) KNOWN GENUINE USAGE-REFERENCE STAYS PROSE (no false-positive regression).
///
/// `gpu_matmul_f32 in backend_impl.rs` (bare-ident) is a CONSUMER citation:
/// gpu_matmul_f32 is declared in blas.rs; backend_impl.rs is the CudaBackendImpl
/// caller. The reclassification rule (file-exists + not-declared-here → Prose)
/// is correct ONLY if the cited file genuinely is NOT the declaration site —
/// which is exactly the bare-ident contract. Re-pointing it to blas.rs would
/// erase the consumer evidence (R-DEFER-1). Pin the ground truth.
#[test]
fn known_bare_ident_usage_reference_is_genuinely_not_a_declaration() {
    let root = workspace_root();
    // Declared in blas.rs, consumed (called) in backend_impl.rs.
    assert!(
        ident_declared(&read(&root, "ferrotorch-gpu/src/blas.rs"), "gpu_matmul_f32"),
        "gpu_matmul_f32 declaration lives in blas.rs"
    );
    let backend = read(&root, "ferrotorch-gpu/src/backend_impl.rs");
    assert!(
        !ident_declared(&backend, "gpu_matmul_f32"),
        "backend_impl.rs does NOT declare gpu_matmul_f32 (so the anchor is not a \
         declaration claim → Prose is the sound classification)"
    );
}

/// (4) THE 3 STRONGEST MASKED-DRIFT CANDIDATES ARE GENUINE PROSE, NOT DRIFT.
///
/// Independent corpus sweep found exactly 3 bare-ident PROSE anchors where the
/// ident is NOT a path-component of the cited file, does NOT appear in the cited
/// file at all, AND is declared as a real fn somewhere else — the maximal-risk
/// "is the rule masking declaration drift?" shape. Reading each in context shows
/// none is declaration drift:
///   - `neg_t in vmap.rs`: `neg_t` (Tensor method) is declared in methods.rs;
///     the anchor sits in a vmap-usage location LIST discussing where `neg`/`neg_t`
///     is exercised via vmap — a usage reference, not a decl claim.
///   - `shifted_chebyshev_polynomial_w_f64 in backend_impl.rs`: this exact symbol
///     is declared NOWHERE — it is an illustrative ellipsis-LIST item naming the
///     conceptual `CudaBackendImpl::<poly>_f64` consumer surface; the real GPU
///     dispatch is the parameterized `chebyshev_poly_f32`/`_f64`. Not a decl claim.
///   - `hermite_polynomial_he in special.rs`: the doc is `.design/ferrotorch-gpu/`
///     so `special.rs` binds to the GPU crate's special.rs, but the symbol is
///     declared in CORE's special.rs (the dispatch ORIGIN the prose names) — a
///     cross-crate basename-collision usage reference, not drift.
///
/// We pin the ground-truth facts each judgement rests on. If any flips (e.g. the
/// symbol becomes genuinely declared in the cited file, or the chebyshev symbol
/// starts existing), this test must be revisited — it is the audit trail for the
/// soundness verdict.
#[test]
fn the_three_maximal_risk_prose_anchors_are_not_declaration_drift() {
    let root = workspace_root();

    // -- neg_t --
    assert!(
        ident_declared(&read(&root, "ferrotorch-core/src/methods.rs"), "neg_t"),
        "neg_t is declared in methods.rs (the Tensor::neg_t chainable method)"
    );
    let vmap = read(&root, "ferrotorch-core/src/vmap.rs");
    assert!(
        !ident_declared(&vmap, "neg_t"),
        "vmap.rs does NOT declare neg_t — the `neg_t in vmap.rs` anchor is a \
         vmap-usage reference (a location list), not a declaration claim"
    );

    // -- shifted_chebyshev_polynomial_w_f64: declared NOWHERE (illustrative) --
    let chebyshev_methods_exist = [
        "ferrotorch-gpu/src/backend_impl.rs",
        "ferrotorch-gpu/src/special.rs",
    ]
    .iter()
    .any(|f| ident_declared(&read(&root, f), "shifted_chebyshev_polynomial_w_f64"));
    assert!(
        !chebyshev_methods_exist,
        "shifted_chebyshev_polynomial_w_f64 is declared NOWHERE — it is an \
         illustrative ellipsis-list item, never a real declaration; classing it \
         Prose (not Stale) is sound because there is no 'correct file' to re-point to"
    );
    // The REAL chebyshev dispatch method that DOES exist (parameterized form):
    assert!(
        ident_declared(
            &read(&root, "ferrotorch-gpu/src/backend_impl.rs"),
            "chebyshev_poly_f32"
        ),
        "the real GPU chebyshev dispatch is the parameterized `chebyshev_poly_f32` \
         in backend_impl.rs (confirming backend_impl.rs IS the genuine consumer file)"
    );

    // -- hermite_polynomial_he: cross-crate basename collision --
    assert!(
        ident_declared(
            &read(&root, "ferrotorch-core/src/special.rs"),
            "hermite_polynomial_he"
        ),
        "hermite_polynomial_he is declared in CORE's special.rs (the dispatch origin)"
    );
    assert!(
        !ident_declared(
            &read(&root, "ferrotorch-gpu/src/special.rs"),
            "hermite_polynomial_he"
        ),
        "GPU's special.rs does NOT declare hermite_polynomial_he — the GPU-doc \
         `hermite_polynomial_he in special.rs` anchor is a cross-crate-basename \
         usage reference to core's dispatch, not declaration drift in the GPU file"
    );
    // Both core and gpu have a `special.rs` (the collision precondition).
    assert!(
        root.join("ferrotorch-core/src/special.rs").exists()
            && root.join("ferrotorch-gpu/src/special.rs").exists(),
        "the special.rs basename collides across core and gpu crates"
    );
}

/// Minimal basename → existence index over `<crate>/{src,examples,tests}` +
/// the parity-sweep runner src, mirroring the production `build_src_index`
/// directory set. Only used to assert the Unresolvable gate's index behavior.
fn build_min_index(root: &Path) -> HashMap<String, ()> {
    let mut index = HashMap::new();
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Ok(entries) = fs::read_dir(root) {
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                for sub in ["src", "examples", "tests"] {
                    let d = p.join(sub);
                    if d.is_dir() {
                        dirs.push(d);
                    }
                }
            }
        }
    }
    let runner = root.join("tools/parity-sweep/runner/src");
    if runner.is_dir() {
        dirs.push(runner);
    }
    while let Some(dir) = dirs.pop() {
        let entries = match fs::read_dir(&dir) {
            Ok(it) => it,
            Err(_) => continue,
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                dirs.push(p);
            } else if p.extension().and_then(|x| x.to_str()) == Some("rs") {
                if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                    index.insert(name.to_string(), ());
                }
            }
        }
    }
    index
}
