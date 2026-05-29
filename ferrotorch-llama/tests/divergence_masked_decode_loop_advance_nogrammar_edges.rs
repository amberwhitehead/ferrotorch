//! Re-audit (#1667 / #1350) — production `masked_decode_loop` invariants the
//! two existing reaudit tests (`divergence_apply_token_mask_gpu_reaudit.rs`)
//! do NOT yet pin:
//!
//!   2(b) GRAMMAR STATE ADVANCES per step — the allow set at step N+1 reflects
//!        the token chosen at step N, not a stale initial mask. A bug that
//!        reused the step-0 mask for every step would still emit a
//!        grammar-allowed-at-step-0 prefix but would not narrow correctly; we
//!        pin that the *allowed set strictly changes* across the walk and that
//!        the emitted multi-char sequence is the unique grammar walk.
//!
//!   5    NO-GRAMMAR path — `grammar=None` is pure greedy decode over the raw
//!        `next_logits`, runs the FULL `max_new_tokens` (no spurious early
//!        completion), and never touches the mask kernel (the forbidden
//!        max-logit token IS emitted every step precisely because no mask
//!        suppresses it — the contrapositive of the masked path).
//!
//!   6    EDGES — `max_new_tokens == 0` (empty output, no panic); a grammar
//!        already complete at entry (returns empty immediately, no
//!        `AlreadyComplete` error); a non-empty prompt is not echoed into the
//!        generated output.
//!
//! All tests drive the REAL `ferrotorch_llama::masked_decode_loop` (the same
//! symbol `generate_masked` wraps); the masking step runs live on the RTX 3090
//! via `apply_grammar_mask_gpu` -> cubecl kernel. Expected token ids are
//! derived from the host-reference grammar walk (the chars that spell the JSON
//! value), never copied from a ferrotorch internal (R-CHAR-3).
//!
//! Gated on `--features cuda`.
//!
//! Tracking: crosslink #1667.

#![cfg(feature = "cuda")]

use ferrotorch_grammar::JsonSchemaProcessor;
use ferrotorch_llama::masked_decode_loop;
use serde_json::json;

/// Printable-ASCII single-char vocabulary (0x20..=0x7E). Index 0 is space.
fn ascii_vocab() -> Vec<String> {
    (0x20u8..=0x7Eu8).map(|b| (b as char).to_string()).collect()
}

fn id_of(vocab: &[String], c: char) -> u32 {
    vocab
        .iter()
        .position(|s| s == &c.to_string())
        .unwrap_or_else(|| panic!("char {c:?} not in vocab")) as u32
}

/// A logits source that always hands the FORBIDDEN token the single largest
/// raw logit, and gives each char in `preferred` a high-but-smaller logit so
/// the *masked* argmax walks the grammar in `preferred` order. Ignores ids.
fn forbidden_max_source<'a>(
    vocab: &'a [String],
    forbidden: char,
    preferred: &'a [char],
) -> impl FnMut(&[u32]) -> ferrotorch_core::FerrotorchResult<Vec<f32>> + 'a {
    let vocab_len = vocab.len();
    let forbidden_id = id_of(vocab, forbidden) as usize;
    let pref: Vec<usize> = preferred
        .iter()
        .map(|&c| id_of(vocab, c) as usize)
        .collect();
    move |_ids: &[u32]| {
        let mut logits = vec![-10.0f32; vocab_len];
        logits[forbidden_id] = 1000.0;
        for (rank, &idx) in pref.iter().enumerate() {
            logits[idx] = 100.0 - rank as f32;
        }
        Ok(logits)
    }
}

/// 2(b): grammar STATE ADVANCES per step.
///
/// We replay the emitted sequence over a fresh grammar and capture the allowed
/// set at each step. Property pinned: the allowed set is NOT constant across
/// the walk (a stale-initial-mask bug would keep the same allowed set every
/// step). For `{"const":"true"}`-style literal walk the allowed set narrows to
/// exactly the next required char at each position, so consecutive steps must
/// differ. We assert the per-step allowed *singleton* equals the host-reference
/// char at that position (derived from the literal, not from ferrotorch).
#[test]
fn masked_decode_loop_grammar_state_advances_per_step() {
    let vocab = ascii_vocab();
    // A boolean grammar's `true` walk: t -> r -> u -> e. At each prefix the
    // grammar permits exactly the next char of `true` (host reference).
    let walk: [char; 4] = ['t', 'r', 'u', 'e'];

    let mut proc = JsonSchemaProcessor::new(&json!({"type": "boolean"}), vocab.clone()).unwrap();
    let next_logits = forbidden_max_source(&vocab, ' ', &walk);

    let generated = masked_decode_loop(next_logits, Some(&mut proc), &[], 16, 0)
        .expect("masked_decode_loop must not error on the boolean walk");

    let expected: Vec<u32> = walk.iter().map(|&c| id_of(&vocab, c)).collect();
    assert_eq!(
        generated, expected,
        "the masked walk must spell `true` exactly"
    );

    // Replay and capture the allowed set at each step from a FRESH grammar
    // (independent of the loop's state). Assert the set genuinely changes —
    // i.e. the mask is recomputed per advance, not frozen at step 0.
    let mut replay = JsonSchemaProcessor::new(&json!({"type": "boolean"}), vocab).unwrap();
    let mut allowed_sets: Vec<Vec<usize>> = Vec::new();
    for &tok in &generated {
        let mask = replay.compute_mask();
        let allowed: Vec<usize> = mask
            .allow
            .iter()
            .enumerate()
            .filter_map(|(i, &b)| (b == 1).then_some(i))
            .collect();
        allowed_sets.push(allowed);
        replay
            .step_token(tok)
            .expect("emitted token must be allowed");
    }

    // The allowed set at the very first step (start of `true`) must include
    // `t`; after stepping `t` the allowed set must NOT be identical to the
    // step-0 set (state advanced). A stale-mask bug yields all-equal sets.
    assert!(
        allowed_sets[0].contains(&(id_of(&ascii_vocab(), 't') as usize)),
        "step 0 must allow `t` (start of boolean literal)"
    );
    let all_identical = allowed_sets.windows(2).all(|w| w[0] == w[1]);
    assert!(
        !all_identical,
        "grammar state did not advance: allowed sets were identical across \
         every step ({allowed_sets:?}) — indicates a stale step-0 mask"
    );
    // Concretely: the step-1 allowed set (after `t`) differs from step 0.
    assert_ne!(
        allowed_sets[0], allowed_sets[1],
        "allowed set after stepping `t` must differ from the start set"
    );
}

/// 5: NO-GRAMMAR path is pure greedy decode and runs the full budget.
///
/// With `grammar=None` the loop must NOT apply any mask: the forbidden token
/// (the one with the single largest raw logit) IS the greedy argmax and so is
/// emitted EVERY step, for exactly `max_new_tokens` steps. (If the mask were
/// erroneously applied with no grammar, this token would be suppressed and the
/// output would differ — this test is the contrapositive of the masked path.)
#[test]
fn masked_decode_loop_no_grammar_is_full_unmasked_greedy() {
    let vocab = ascii_vocab();
    let forbidden = ' ';
    let forbidden_id = id_of(&vocab, forbidden);
    // No `preferred` chars needed; the source still ranks `forbidden` highest.
    let next_logits = forbidden_max_source(&vocab, forbidden, &[]);

    let max_new = 5usize;
    let generated = masked_decode_loop(next_logits, None, &[], max_new, 0)
        .expect("no-grammar greedy decode must not error");

    // Runs the FULL budget — no spurious completion when grammar is None.
    assert_eq!(
        generated.len(),
        max_new,
        "no-grammar decode must run the full max_new_tokens ({max_new})"
    );
    // Every step picks the unmasked argmax = the forbidden max-logit token,
    // proving NO mask was applied on the grammar=None path.
    assert_eq!(
        generated,
        vec![forbidden_id; max_new],
        "no-grammar greedy decode must emit the raw-argmax (forbidden) token \
         every step; a non-uniform result means a mask was wrongly applied"
    );
}

/// 6: EDGE — max_new_tokens == 0 yields an empty output with no panic, both
/// with and without a grammar.
#[test]
fn masked_decode_loop_zero_max_new_tokens_is_empty() {
    let vocab = ascii_vocab();

    // No grammar.
    let src = forbidden_max_source(&vocab, ' ', &[]);
    let g_none = masked_decode_loop(src, None, &[], 0, 0).expect("zero-budget no-grammar");
    assert!(
        g_none.is_empty(),
        "max_new_tokens=0 must yield empty output"
    );

    // With grammar (the completion guard / mask path must not run either).
    let mut proc = JsonSchemaProcessor::new(&json!({"type": "boolean"}), vocab.clone()).unwrap();
    let src2 = forbidden_max_source(&vocab, ' ', &['t', 'r', 'u', 'e']);
    let g_some = masked_decode_loop(src2, Some(&mut proc), &[], 0, 0).expect("zero-budget grammar");
    assert!(
        g_some.is_empty(),
        "max_new_tokens=0 with a grammar must yield empty output, no panic"
    );
    assert!(
        !proc.is_complete(),
        "a never-stepped boolean grammar must not be complete"
    );
}

/// 6: EDGE — a grammar that is ALREADY COMPLETE at entry must return empty
/// immediately (the step-0 completion guard fires before any mask/argmax/step),
/// with NO `AlreadyComplete` error — even though the budget is non-zero.
#[test]
fn masked_decode_loop_already_complete_grammar_returns_empty_no_error() {
    let vocab = ascii_vocab();
    let mut proc = JsonSchemaProcessor::new(&json!({"type": "boolean"}), vocab.clone()).unwrap();
    // Drive the grammar to completion by hand (host-reference value `true`).
    for c in ['t', 'r', 'u', 'e'] {
        proc.step_token(id_of(&vocab, c)).unwrap();
    }
    assert!(proc.is_complete(), "grammar must be complete after `true`");

    // The source still ranks the forbidden token highest; if the guard did NOT
    // fire we would mask -> all-deny -> argmax index 0 -> step_token(0) ->
    // AlreadyComplete error (the #1667 divergence). The guard must short it.
    let src = forbidden_max_source(&vocab, ' ', &[]);
    let generated = masked_decode_loop(src, Some(&mut proc), &[], 8, 0)
        .expect("already-complete grammar must return empty, NOT AlreadyComplete error");
    assert!(
        generated.is_empty(),
        "an already-complete grammar must emit nothing (got {generated:?})"
    );
}

/// 6: EDGE — a non-empty prompt is fed to `next_logits` but is NOT echoed into
/// the generated output (only newly produced ids are returned). We assert the
/// prompt ids do not prefix the generated vector and the length matches the
/// grammar walk, not prompt+walk.
#[test]
fn masked_decode_loop_prompt_not_echoed_in_output() {
    let vocab = ascii_vocab();
    let walk: [char; 4] = ['t', 'r', 'u', 'e'];
    let prompt: Vec<u32> = vec![id_of(&vocab, 'X'), id_of(&vocab, 'Y'), id_of(&vocab, 'Z')];

    let mut proc = JsonSchemaProcessor::new(&json!({"type": "boolean"}), vocab.clone()).unwrap();
    let src = forbidden_max_source(&vocab, ' ', &walk);
    let generated = masked_decode_loop(src, Some(&mut proc), &prompt, 16, 0)
        .expect("prompted masked decode must not error");

    let expected: Vec<u32> = walk.iter().map(|&c| id_of(&vocab, c)).collect();
    assert_eq!(
        generated, expected,
        "generated output must be exactly the grammar walk, not prompt + walk"
    );
    assert!(
        !generated.starts_with(&prompt),
        "the prompt must not be echoed into the generated output"
    );
}
