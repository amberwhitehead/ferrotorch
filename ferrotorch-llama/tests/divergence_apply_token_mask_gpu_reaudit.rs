//! Re-audit of #1350 second half (commit 05f8a7a0): the
//! `apply_grammar_mask_gpu` / `LlamaGpuInferencer::generate_masked` GPU
//! grammar-constrained decode path in `ferrotorch-llama/src/gpu_gguf.rs`.
//!
//! The builder's four tests in `divergence_apply_token_mask_gpu.rs` pin
//! the *single-step* masking semantics (allowed == bit-exact, disallowed
//! == `f32::MIN`, length mismatch == clean error, forbidden-max-logit
//! never wins one masked argmax). They do **not** exercise the loop
//! invariant that the autoregressive decoder must hold across steps:
//! generation must STOP when the grammar reaches a complete value.
//!
//! DIVERGENCE (loop completion was not honored) — #1667
//! ----------------------------------------------------
//! `generate_masked` originally looped `for _ in 0..max_new_tokens` with
//! NO `if proc.is_complete() { break }` guard. Once the grammar emits a
//! complete value (e.g. `true` for `{"type":"boolean"}`), `compute_mask()`
//! returns an ALL-ZERO (all-deny) mask (`json_schema.rs:211-216`). The GPU
//! kernel then forces every logit to `f32::MIN`, `argmax_f32`
//! (init `NEG_INFINITY`) returns index 0 — a *forbidden* token (the space
//! char ` ` in an ASCII vocab) — and `step_token(0)` returns
//! `Err(StepError::AlreadyComplete)`, which `generate_masked` wrapped into
//! `FerrotorchError::InvalidArgument` and propagated. Any caller that
//! requested more `max_new_tokens` than the grammar's completion length
//! got a hard ERROR instead of the completed sequence.
//!
//! THE FIX drives REAL production code, not a duplicate
//! ----------------------------------------------------
//! These tests call the production [`ferrotorch_llama::masked_decode_loop`]
//! directly with an INJECTED synthetic-logits closure (so no real GGUF
//! model is needed — the cubecl GPU mask kernel still runs live on the
//! RTX 3090). `generate_masked` is now a thin wrapper over exactly this
//! function (`|ids| self.forward_from_ids(ids)` is the only difference),
//! so the completion guard, the GPU masking, and the grammar advance these
//! tests exercise are the SAME code the production decoder runs. The guard
//! is load-bearing: deleting `if proc.is_complete() { break }` from
//! `masked_decode_loop` fails both tests below.
//!
//! The injected closure hands the FORBIDDEN max-logit token the single
//! largest raw logit at every step (so an unmasked argmax would emit it),
//! proving the GPU mask + the completion guard together prevent ever
//! emitting it. Expected token ids are derived from the grammar's vocab
//! (the chars of the host-reference value `true`), not copied from any
//! ferrotorch internal (R-CHAR-3).
//!
//! Gated on `--features cuda`; the masking step runs on the live RTX 3090.
//!
//! Tracking: crosslink #1667 (release-blocker: tests left un-#[ignore]d intentionally)

#![cfg(feature = "cuda")]

use ferrotorch_grammar::JsonSchemaProcessor;
use ferrotorch_llama::masked_decode_loop;
use serde_json::json;

/// The printable-ASCII char vocabulary the boolean-grammar tests use:
/// each printable ASCII byte (0x20..=0x7E) is its own single-char token.
/// Index 0 is the space char ` ` — the token `argmax_f32` lands on under
/// an all-deny (post-completion) mask, hence the "forbidden index 0".
fn ascii_vocab() -> Vec<String> {
    (0x20u8..=0x7Eu8).map(|b| (b as char).to_string()).collect()
}

/// Position of a single-char token in the vocab.
fn id_of(vocab: &[String], c: char) -> u32 {
    vocab
        .iter()
        .position(|s| s == &c.to_string())
        .unwrap_or_else(|| panic!("char {c:?} not in vocab")) as u32
}

/// Build a synthetic per-step logits source for `masked_decode_loop`.
///
/// This stands in for `forward_from_ids`: it ignores the id history and
/// returns a fixed logit field where (a) the FORBIDDEN max-logit token
/// gets the single strictly-largest raw logit (so an *unmasked* greedy
/// argmax would emit it — proving the GPU mask is load-bearing), and (b)
/// the chars that spell `true` get high-but-smaller logits, ranked
/// `t > r > u > e`, so the masked argmax walks the boolean grammar to
/// completion. The closure is `FnMut` per the production signature; it
/// holds no state here.
fn synthetic_logits_source(
    vocab: &[String],
    forbidden: char,
) -> impl FnMut(&[u32]) -> ferrotorch_core::FerrotorchResult<Vec<f32>> + '_ {
    let vocab_len = vocab.len();
    let forbidden_id = id_of(vocab, forbidden) as usize;
    let preferred: Vec<usize> = ['t', 'r', 'u', 'e']
        .iter()
        .map(|&c| id_of(vocab, c) as usize)
        .collect();
    move |_ids: &[u32]| {
        let mut logits = vec![-10.0f32; vocab_len];
        // The forbidden token has the strictly-largest raw logit.
        logits[forbidden_id] = 1000.0;
        // The `true` chars get high (but smaller) logits, t > r > u > e.
        for (rank, &idx) in preferred.iter().enumerate() {
            logits[idx] = 100.0 - rank as f32;
        }
        Ok(logits)
    }
}

/// DIVERGENCE (fixed by #1667): with the completion guard in the
/// production `masked_decode_loop`, a boolean grammar driven for MORE
/// steps than it can emit halts cleanly at completion after `t r u e`
/// (4 tokens) with NO error — instead of running past completion, picking
/// the forbidden all-deny index-0 token, and erroring `AlreadyComplete`.
///
/// This drives the REAL production helper (`generate_masked` is a thin
/// wrapper over it). Removing the `is_complete()` break from production
/// reintroduces the error and fails this test.
#[test]
fn masked_decode_loop_stops_at_grammar_completion_not_error() {
    let vocab = ascii_vocab();
    let mut proc = JsonSchemaProcessor::new(&json!({"type": "boolean"}), vocab.clone()).unwrap();

    // Forbidden token: index 0 (space). It gets the max raw logit; the
    // mask + guard must prevent it ever being emitted.
    let next_logits = synthetic_logits_source(&vocab, ' ');

    // Expected sequence from the host-reference value `true`'s chars (NOT
    // copied from any ferrotorch internal): the grammar's allowed walk.
    let expected: Vec<u32> = ['t', 'r', 'u', 'e']
        .iter()
        .map(|&c| id_of(&vocab, c))
        .collect();

    // Ask for MORE tokens than the grammar can ever emit.
    let generated = masked_decode_loop(next_logits, Some(&mut proc), &[], 8, 0)
        .expect("masked_decode_loop must stop at grammar completion, not error");

    // Stops at completion (length == grammar completion length, not 8).
    assert_eq!(
        generated,
        expected,
        "masked_decode_loop must yield exactly the completed boolean value `true`, \
         stopping at completion (len {}) not max_new_tokens (8)",
        expected.len()
    );
    // And the grammar is indeed complete after the run.
    assert!(
        proc.is_complete(),
        "boolean grammar must be complete after `true`"
    );
}

/// DIVERGENCE (corollary, fixed by #1667): across the WHOLE production run
/// every emitted token must be grammar-allowed — in particular the loop
/// must NEVER emit the forbidden max-logit token, and must stop at
/// completion rather than emitting the post-completion all-deny index-0
/// token. We replay the grammar over the emitted ids and assert each step
/// was allowed by the mask in force at that step.
#[test]
fn masked_decode_loop_emits_only_grammar_allowed_tokens() {
    let vocab = ascii_vocab();
    let forbidden = ' '; // index 0 — the all-deny argmax landing spot.
    let forbidden_id = id_of(&vocab, forbidden);

    // Run the production loop with the forbidden token holding the max
    // raw logit at every step.
    let generated = {
        let mut proc =
            JsonSchemaProcessor::new(&json!({"type": "boolean"}), vocab.clone()).unwrap();
        let next_logits = synthetic_logits_source(&vocab, forbidden);
        masked_decode_loop(next_logits, Some(&mut proc), &[], 8, 0)
            .expect("masked_decode_loop must not error")
    };

    // (a) The loop stopped at completion, not max_new_tokens.
    assert!(
        generated.len() < 8,
        "masked_decode_loop must stop at completion (got {} tokens, max was 8)",
        generated.len()
    );
    assert!(
        !generated.is_empty(),
        "masked_decode_loop must have emitted the boolean value"
    );

    // (b) Replay the grammar over the emitted ids; each emitted token must
    // have been ALLOWED by the mask at that step (re-derived from a fresh
    // grammar, independent of the loop's internal state). In particular
    // none is the forbidden index-0 token.
    let mut replay = JsonSchemaProcessor::new(&json!({"type": "boolean"}), vocab.clone()).unwrap();
    for &tok in &generated {
        assert_ne!(
            tok, forbidden_id,
            "masked_decode_loop emitted the FORBIDDEN max-logit token {tok} \
             (char {:?})",
            vocab[tok as usize]
        );
        let mask = replay.compute_mask();
        assert_eq!(
            mask.allow[tok as usize], 1,
            "masked_decode_loop emitted token {tok} (char {:?}) that the grammar \
             forbade at that step",
            vocab[tok as usize]
        );
        replay
            .step_token(tok)
            .expect("re-stepping an emitted token must be accepted");
    }
    assert!(
        replay.is_complete(),
        "the emitted sequence must complete the boolean grammar"
    );
}
