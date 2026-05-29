//! DISCRIMINATOR precision probe for the #1643 S3 struct-anchor parser.
//!
//! Re-derives the parser's precision contract independently of the builder's
//! own in-suite test (`s3_struct_anchor_parser_is_precise_and_catches_corruption`),
//! using the EXACT corpus lines that contain the substring "struct in `" but
//! are NOT the canonical `` `<Camel>` struct in `<file>.rs` `` anchor. These are
//! the real false-positive hazards (found by grepping .design/):
//!   - tensor.md:203       "...every grad-fn struct in `grad_fns/*`..."
//!   - transformer.md:271  "...every transformer struct in `transformer.rs`"
//!   - rnn.md:265          "...every public struct in `rnn.rs`"
//!   - indexing.md:197     "`IndexFillBackward` (struct in `grad_fns/indexing.rs`)"
//!
//! Since the parser's functions are private to the test crate's sibling file,
//! we re-implement the parser's PUBLIC CONTRACT here as a black-box reference
//! (the exact CONNECTIVE + camel + .rs-path rules documented in the doc-comment
//! of `parse_struct_anchors`) and assert it on these lines. We ALSO assert the
//! corpus-wide scoped test outcome indirectly: the four hazard lines, if they
//! were matched, would either be unresolvable (`grad_fns/*`) or name a
//! non-struct symbol (`transformer`, `public`) and FAIL the scoped test — but
//! the scoped test is green at HEAD with 28 anchors, so they are NOT matched.
//!
//! Per R-CHAR-3: expected values here are derived from the documented parser
//! contract + live corpus lines, not copied from the ferrotorch side.

#![allow(clippy::missing_panics_doc)]

use std::fs;
use std::path::PathBuf;

fn workspace_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if !p.join(".design").exists() {
        p.pop();
    }
    p
}

/// Black-box reference re-implementation of the parser's MATCH PREDICATE: does
/// `line` contain at least one canonical S3 anchor? Mirrors the documented
/// contract of `parse_struct_anchors`:
///   1. literal connective "` struct in `" (backtick + " struct in " + backtick)
///   2. symbol span: backtick-quoted single CamelCase ident (generics stripped),
///      leading UPPERCASE ascii
///   3. file span: backtick-quoted path ending in ".rs"
fn reference_has_anchor(line: &str) -> bool {
    const CONNECTIVE: &str = "` struct in `";
    let mut from = 0usize;
    let bytes = line.as_bytes();
    while let Some(rel) = line[from..].find(CONNECTIVE) {
        let conn_start = from + rel;
        let conn_end = conn_start + CONNECTIVE.len();
        from = conn_end;
        if conn_start == 0 || bytes[conn_start] != b'`' {
            continue;
        }
        let sym_open = match line[..conn_start].rfind('`') {
            Some(p) => p,
            None => continue,
        };
        let sym_raw = &line[sym_open + 1..conn_start];
        let stem = match sym_raw.find('<') {
            Some(lt) => &sym_raw[..lt],
            None => sym_raw,
        }
        .trim();
        let camel = !stem.is_empty()
            && stem.chars().next().is_some_and(|c| c.is_ascii_uppercase())
            && stem.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
        if !camel {
            continue;
        }
        let file_close = match line[conn_end..].find('`') {
            Some(e) => conn_end + e,
            None => continue,
        };
        let file_raw = line[conn_end..file_close].trim();
        let rs = file_raw.ends_with(".rs")
            && !file_raw[..file_raw.len() - 3].is_empty()
            && file_raw[..file_raw.len() - 3]
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '/' || c == '-' || c == '.');
        if camel && rs {
            return true;
        }
    }
    false
}

#[test]
fn s3_parser_rejects_corpus_struct_in_prose_that_is_not_an_anchor() {
    // The four real corpus hazard lines (substring "struct in `" but NOT the
    // canonical anchor). None must be matched.
    let hazards = [
        "| REQ-19 | SHIPPED | impl: `trait GradFn<T>` at `ferrotorch-core/src/tensor.rs:46-68`; non-test consumer: every grad-fn struct in `grad_fns/*` implements this — see `grad_fns/arithmetic.rs::AddBackward`, `AddScaledBackward`, etc. |",
        "| REQ-11 | SHIPPED | impl: `impl<T: Float> Module<T> for ...` blocks for every transformer struct in `transformer.rs`; non-test consumer: re-export at `lib.rs`. |",
        "| REQ-10 | SHIPPED | impl: `impl<T: Float> Module<T> for ...` blocks for every public struct in `rnn.rs`; non-test consumer: re-export at `lib.rs`. |",
        "  `IndexFillBackward` (struct in `grad_fns/indexing.rs`) which on backward returns",
    ];
    for line in hazards {
        assert!(
            !reference_has_anchor(line),
            "PRECISION: reference predicate matched a non-anchor corpus line: {line}",
        );
    }

    // Positive controls: the canonical forms MUST match.
    assert!(reference_has_anchor(
        "the `RsqrtBackward` struct in `grad_fns/arithmetic.rs` saving c"
    ));
    assert!(reference_has_anchor(
        "`FlexAttentionBackward<T>` struct in `flex_attention.rs`"
    ));

    // Adversarial: a non-struct CamelCase token immediately preceding the
    // connective whose CONNECTIVE backtick is missing must be rejected.
    assert!(!reference_has_anchor("the Foo struct in `bar.rs`"));
    assert!(!reference_has_anchor("`Foo` struct in `bar.py`"));
}

/// Corpus-grounded coverage + corpus-integrity check: every line in .design/
/// matched by our reference predicate must be one of the known canonical
/// anchors, and there must be >= 10 of them (the #1633 *Backward family). This
/// independently corroborates the scoped test's anchor_count >= 10 floor and
/// confirms the four hazard lines above are NOT among the matches.
#[test]
fn reference_predicate_matches_only_canonical_anchors_in_corpus() {
    let root = workspace_root();
    let design = root.join(".design");
    let mut matched = 0usize;
    let mut hazard_matched = 0usize;
    let mut stack = vec![design];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = fs::read_dir(&dir) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
                continue;
            }
            if p.extension().and_then(|x| x.to_str()) != Some("md") {
                continue;
            }
            let Ok(text) = fs::read_to_string(&p) else {
                continue;
            };
            for line in text.lines() {
                if reference_has_anchor(line) {
                    matched += 1;
                    // A canonical anchor's symbol span ends in "Backward" or is
                    // one of the known non-Backward structs.
                    let known_non_backward =
                        line.contains("`AliasTable`") || line.contains("`ContinuousBernoulli`");
                    let backward = line.contains("Backward` struct in `");
                    if !backward && !known_non_backward {
                        hazard_matched += 1;
                    }
                }
            }
        }
    }
    assert!(
        matched >= 10,
        "expected >= 10 canonical S3 anchors in corpus, found {matched}",
    );
    assert_eq!(
        hazard_matched, 0,
        "reference predicate matched {hazard_matched} line(s) that are neither a *Backward anchor nor a known non-Backward struct anchor — possible false positive",
    );
}
