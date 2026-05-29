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
//! DIVERGENCE (loop completion is not honored)
//! -------------------------------------------
//! `generate_masked` (gpu_gguf.rs:514) is:
//!
//! ```ignore
//! for _ in 0..max_new_tokens {
//!     let logits = self.forward_from_ids(&ids)?;
//!     let mask   = proc.compute_mask();                 // step 2
//!     let masked = apply_grammar_mask_gpu(&logits, ..)?;// step 3 (GPU)
//!     let next   = argmax_f32(&masked);                 // step 4
//!     proc.step_token(next)?;                           // step 5
//!     ...
//! }
//! ```
//!
//! There is NO `if proc.is_complete() { break }` guard. Once the grammar
//! emits a complete value (e.g. `true` for `{"type":"boolean"}`),
//! `compute_mask()` returns an ALL-ZERO (all-deny) mask
//! (`json_schema.rs:211-216`). The GPU kernel then forces every logit to
//! `f32::MIN`, `argmax_f32` (gpu_gguf.rs:544, init `NEG_INFINITY`) returns
//! index 0 — a *forbidden* token (the space char ` ` in an ASCII vocab) —
//! and `proc.step_token(0)` returns `Err(StepError::AlreadyComplete)`
//! (state.rs:58-59), which `generate_masked` wraps into
//! `FerrotorchError::InvalidArgument` and propagates.
//!
//! Net effect: any caller that requests more `max_new_tokens` than the
//! grammar's completion length gets a hard ERROR instead of the completed
//! token sequence. A correct constrained decoder stops at completion. The
//! builder's "every emitted token is grammar-allowed" claim is also
//! violated at the post-completion step, where the all-deny argmax yields
//! index 0 (a forbidden token) before the step errors.
//!
//! This test reproduces `generate_masked`'s documented inner composition
//! (compute_mask -> apply_grammar_mask_gpu [LIVE GPU] -> argmax_f32 ->
//! step_token) without needing a loaded GGUF model, and asserts the
//! behavior a correct decoder MUST have: the loop yields the completed
//! `t r u e` sequence and the post-completion all-deny step is NOT a
//! forbidden token / error. It FAILS against commit 05f8a7a0 because the
//! production loop has no completion guard.
//!
//! Gated on `--features cuda`; the masking step runs on the live RTX 3090.
//!
//! Tracking: crosslink #1667 (release-blocker: tests left un-#[ignore]d intentionally)

#![cfg(feature = "cuda")]

use ferrotorch_grammar::JsonSchemaProcessor;
use ferrotorch_llama::apply_grammar_mask_gpu;
use serde_json::json;

/// Replicate `argmax_f32` from gpu_gguf.rs:544 exactly (init NEG_INFINITY,
/// ties -> lowest index, empty -> 0) so the test exercises the same
/// selection rule the production loop uses.
fn argmax_f32(logits: &[f32]) -> u32 {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx as u32
}

/// Drive the documented inner composition of `generate_masked`
/// (compute_mask -> apply_grammar_mask_gpu[GPU] -> argmax_f32 ->
/// step_token) for `max_new_tokens` steps, stopping ONLY when the loop
/// errors. Returns the generated token-id sequence and an optional error
/// string. The grammar's raw logits are stubbed so that the allowed token
/// with the largest logit advances the value toward `true`.
fn drive_masked_loop(
    proc: &mut JsonSchemaProcessor,
    vocab: &[String],
    max_new_tokens: usize,
) -> (Vec<u32>, Option<String>) {
    let vocab_len = vocab.len();
    // Logit field: prefer the letters that spell `true` so the masked
    // argmax walks the boolean grammar to completion. Every other token
    // gets a smaller logit. (This stands in for forward_from_ids; the
    // GPU mask + argmax + grammar step are the real code under audit.)
    let preferred: Vec<usize> = ['t', 'r', 'u', 'e']
        .iter()
        .map(|&c| vocab.iter().position(|s| s == &c.to_string()).unwrap())
        .collect();

    let mut generated = Vec::new();
    for _ in 0..max_new_tokens {
        // Build per-step logits: preferred chars get a high logit.
        let mut logits = vec![-10.0f32; vocab_len];
        for (rank, &idx) in preferred.iter().enumerate() {
            logits[idx] = 100.0 - rank as f32; // t > r > u > e, all high
        }

        // ---- exactly generate_masked's grammar branch ----
        let mask = proc.compute_mask();
        let masked = match apply_grammar_mask_gpu(&logits, &mask.allow, 0) {
            Ok(m) => m,
            Err(e) => return (generated, Some(format!("{e}"))),
        };
        let next = argmax_f32(&masked);
        if let Err(e) = proc.step_token(next) {
            // generate_masked wraps this into InvalidArgument and returns.
            return (generated, Some(format!("{e}")));
        }
        generated.push(next);
    }
    (generated, None)
}

/// DIVERGENCE: `generate_masked` has no `is_complete()` break, so once the
/// boolean value `true` is fully emitted (4 tokens), the 5th step's
/// all-deny mask forces an all-`f32::MIN` logit vector, the argmax lands
/// on the forbidden index 0, and `step_token` errors with
/// `AlreadyComplete`. A correct constrained decoder stops at completion
/// and returns the completed sequence.
///
/// Expected (correct decoder): drive 8 steps over a boolean grammar,
/// generation halts at completion after `t r u e` (4 tokens), NO error.
///
/// Actual (commit 05f8a7a0): the loop runs past completion, the
/// post-completion all-deny argmax selects a forbidden token, and the run
/// terminates with an `AlreadyComplete` error (surfaced as
/// `InvalidArgument`).
#[test]
fn generate_masked_stops_at_grammar_completion_not_error() {
    let vocab: Vec<String> = (0x20u8..=0x7Eu8).map(|b| (b as char).to_string()).collect();
    let mut proc = JsonSchemaProcessor::new(&json!({"type": "boolean"}), vocab.clone()).unwrap();

    let t_id = vocab.iter().position(|s| s == "t").unwrap() as u32;
    let r_id = vocab.iter().position(|s| s == "r").unwrap() as u32;
    let u_id = vocab.iter().position(|s| s == "u").unwrap() as u32;
    let e_id = vocab.iter().position(|s| s == "e").unwrap() as u32;

    // Ask for MORE tokens than the grammar can ever emit.
    let (generated, err) = drive_masked_loop(&mut proc, &vocab, 8);

    // A correct constrained decoder: no error, stops cleanly at completion.
    assert!(
        err.is_none(),
        "generate_masked must stop at grammar completion, not error; \
         got error after generating {:?}: {:?}",
        generated,
        err
    );
    // ...having produced exactly the completed value `t r u e`.
    assert_eq!(
        generated,
        vec![t_id, r_id, u_id, e_id],
        "generate_masked must yield exactly the completed boolean value"
    );
}

/// DIVERGENCE (corollary): the post-completion step's all-deny argmax is a
/// FORBIDDEN token (index 0 == the space char ` `), directly contradicting
/// the builder's "every emitted token is grammar-allowed / forbidden token
/// never emitted" claim. We isolate the single post-completion step on the
/// live GPU: complete the grammar, then mask + argmax once more.
#[test]
fn post_completion_all_deny_argmax_is_a_forbidden_token() {
    let vocab: Vec<String> = (0x20u8..=0x7Eu8).map(|b| (b as char).to_string()).collect();
    let mut proc = JsonSchemaProcessor::new(&json!({"type": "boolean"}), vocab.clone()).unwrap();

    // Walk the grammar to completion: emit `t r u e`.
    for ch in ['t', 'r', 'u', 'e'] {
        let id = vocab.iter().position(|s| s == &ch.to_string()).unwrap() as u32;
        proc.step_token(id).expect("emitting `true` must be accepted");
    }
    assert!(proc.is_complete(), "boolean grammar must be complete after `true`");

    // Post-completion mask is all-deny.
    let mask = proc.compute_mask();
    let allowed: u32 = mask.allow.iter().copied().sum();
    assert_eq!(allowed, 0, "completed grammar must allow zero tokens");

    // The production loop would now GPU-mask arbitrary logits and argmax.
    let logits: Vec<f32> = (0..vocab.len()).map(|i| i as f32).collect();
    let masked = apply_grammar_mask_gpu(&logits, &mask.allow, 0).unwrap();

    // generate_masked's argmax over an all-`f32::MIN` vector -> index 0.
    let chosen = argmax_f32(&masked);

    // The chosen token must NOT be forbidden. With an all-deny mask EVERY
    // token is forbidden, so a correct decoder must never reach here (it
    // must have stopped). This assert pins the divergence: index 0 is
    // forbidden, yet generate_masked would feed it to step_token.
    assert_eq!(
        mask.allow[chosen as usize], 1,
        "generate_masked's post-completion argmax selected forbidden token \
         id {chosen} (char {:?}); a correct decoder must stop before this step",
        vocab[chosen as usize]
    );
}
