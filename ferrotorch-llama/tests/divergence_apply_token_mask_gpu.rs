//! Live-GPU tests for the grammar-constrained on-device logit masking
//! path (#1350, REQ-8 of `.design/ferrotorch-llama/gpu_gguf_loader.md`).
//!
//! These tests prove that
//! [`ferrotorch_llama::apply_grammar_mask_gpu`] — the production consumer
//! of [`ferrotorch_cubecl::quant::apply_token_mask_to_gpu`] — runs the
//! `kernel_apply_token_mask` `#[cube]` kernel **on the GPU** (cubecl
//! `CudaRuntime` client, real JIT + device dispatch, no host shortcut)
//! and that the masking has the load-bearing property the
//! grammar-constrained decoder relies on:
//!
//!   * allowed positions (`mask[i] != 0`) pass through **bit-exact**;
//!   * disallowed positions (`mask[i] == 0`) become `f32::MIN`
//!     (the kernel's `F::min_value()`), which can never win a greedy
//!     argmax against any finite logit and underflows to probability
//!     zero under softmax.
//!
//! The `generate_masked` loop composes exactly
//! `forward_from_ids → apply_grammar_mask_gpu → argmax → step_token`;
//! the argmax-after-mask test below pins the "a forbidden max-logit
//! token is never emitted" invariant at that composition's boundary,
//! which is the property the full GPU generate loop inherits.
//!
//! Gated on `--features cuda`; requires a CUDA device at test time
//! (validated on a live RTX 3090). No `#[ignore]`, no CPU fallback.

#![cfg(feature = "cuda")]

use ferrotorch_grammar::JsonSchemaProcessor;
use ferrotorch_llama::apply_grammar_mask_gpu;
use serde_json::json;

/// Host reference for the GPU kernel: `mask[i] != 0 ? logits[i] : f32::MIN`.
/// `f32::MIN` is the most-negative finite f32 (≈ -3.4e38), which is what
/// cubecl's `F::min_value()` resolves to for the `f32` element type.
fn host_mask_reference(logits: &[f32], mask: &[u32]) -> Vec<f32> {
    logits
        .iter()
        .zip(mask.iter())
        .map(|(&v, &m)| if m != 0 { v } else { f32::MIN })
        .collect()
}

/// CORRECTNESS: on the GPU, allowed positions are bit-exact and
/// disallowed positions are exactly `f32::MIN`.
#[test]
fn gpu_mask_allowed_bit_exact_disallowed_is_f32_min() {
    // Deterministic mixed-sign logits + an allow mask with both 0s and 1s.
    let logits: Vec<f32> = vec![
        1.5, -2.25, 0.0, 100.0, -100.0, 3.5, 42.125, -0.001, 7.0, -7.5, 0.5, -0.5,
    ];
    let mask: Vec<u32> = vec![1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0];
    assert_eq!(logits.len(), mask.len());

    let gpu = apply_grammar_mask_gpu(&logits, &mask, 0).expect("GPU mask kernel must run");
    let reference = host_mask_reference(&logits, &mask);

    assert_eq!(gpu.len(), logits.len());
    for i in 0..logits.len() {
        if mask[i] != 0 {
            // Allowed: must be bit-exact equal to the original logit.
            assert_eq!(
                gpu[i].to_bits(),
                logits[i].to_bits(),
                "allowed position {i} must pass through bit-exact: got {} want {}",
                gpu[i],
                logits[i]
            );
        } else {
            // Disallowed: must be exactly f32::MIN (kernel's min_value()).
            assert_eq!(
                gpu[i].to_bits(),
                f32::MIN.to_bits(),
                "disallowed position {i} must be f32::MIN: got {} (bits {:#x})",
                gpu[i],
                gpu[i].to_bits()
            );
        }
        // And it must match the host reference exactly, bit for bit.
        assert_eq!(
            gpu[i].to_bits(),
            reference[i].to_bits(),
            "position {i} diverges from host reference",
        );
    }
}

/// ON-GPU sanity: an all-allow mask is the identity (every logit passes
/// through bit-exact), and an all-mask collapses everything to f32::MIN.
/// Proves the kernel actually consumed the mask buffer (not a no-op
/// passthrough nor a constant fill).
#[test]
fn gpu_mask_all_allow_is_identity_all_deny_is_min() {
    let logits: Vec<f32> = (0..64).map(|i| (i as f32) * 0.5 - 16.0).collect();

    let all_allow = vec![1u32; logits.len()];
    let id = apply_grammar_mask_gpu(&logits, &all_allow, 0).unwrap();
    for (i, (&g, &l)) in id.iter().zip(logits.iter()).enumerate() {
        assert_eq!(g.to_bits(), l.to_bits(), "identity mask diverged at {i}");
    }

    let all_deny = vec![0u32; logits.len()];
    let collapsed = apply_grammar_mask_gpu(&logits, &all_deny, 0).unwrap();
    for (i, &g) in collapsed.iter().enumerate() {
        assert_eq!(
            g.to_bits(),
            f32::MIN.to_bits(),
            "all-deny must force f32::MIN at {i}"
        );
    }
}

/// Length mismatch is a clean `InvalidArgument`, not a panic / debug
/// assert / silent truncation.
#[test]
fn gpu_mask_length_mismatch_is_clean_error() {
    let logits = vec![1.0f32, 2.0, 3.0];
    let mask = vec![1u32, 0]; // shorter than logits
    let err = apply_grammar_mask_gpu(&logits, &mask, 0).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("logits.len()") && msg.contains("allow_mask.len()"),
        "unexpected error message: {msg}"
    );
}

/// FORBIDDEN-TOKEN-NOT-EMITTED: this is the invariant `generate_masked`
/// inherits. Construct logits where the token with the **maximum raw
/// logit** is one the grammar forbids; after GPU masking, a greedy argmax
/// must NOT pick it.
///
/// We use a real `JsonSchemaProcessor` so the mask is produced by the
/// same `compute_mask` the generate loop calls. With a `{"type":
/// "boolean"}` schema at the start state only `t` and `f` are allowed, so
/// any other vocab entry is forbidden — we hand the forbidden entry the
/// single largest logit and confirm the masked argmax lands on an allowed
/// (`t`/`f`) token instead.
#[test]
fn gpu_masked_argmax_never_selects_forbidden_max_logit_token() {
    // Printable-ASCII char vocab: each char is its own "token".
    let vocab: Vec<String> = (0x20u8..=0x7Eu8).map(|b| (b as char).to_string()).collect();
    let processor = JsonSchemaProcessor::new(&json!({"type": "boolean"}), vocab.clone()).unwrap();

    let mask = processor.compute_mask();
    assert_eq!(mask.allow.len(), vocab.len());

    let t_id = vocab.iter().position(|s| s == "t").unwrap();
    let f_id = vocab.iter().position(|s| s == "f").unwrap();
    let x_id = vocab.iter().position(|s| s == "x").unwrap();
    // Sanity: the grammar allows t/f and forbids x at the start state.
    assert_eq!(mask.allow[t_id], 1, "boolean start state must allow 't'");
    assert_eq!(mask.allow[f_id], 1, "boolean start state must allow 'f'");
    assert_eq!(mask.allow[x_id], 0, "boolean start state must forbid 'x'");

    // Give the FORBIDDEN token 'x' the strictly-largest raw logit, and a
    // smaller (but still finite, positive) logit to an allowed token.
    let mut logits = vec![-1.0f32; vocab.len()];
    logits[x_id] = 1000.0; // forbidden token has the max raw logit
    logits[t_id] = 5.0; // allowed token, the intended winner
    logits[f_id] = 1.0; // allowed token, runner-up

    // Raw argmax would pick the forbidden token — confirm the setup.
    let raw_argmax = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    assert_eq!(
        raw_argmax, x_id,
        "test setup: raw argmax must be the forbidden token"
    );

    // GPU-mask, then greedy argmax (exactly generate_masked's inner step).
    let masked = apply_grammar_mask_gpu(&logits, &mask.allow, 0).unwrap();
    let chosen = masked
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
            if v > bv { (i, v) } else { (bi, bv) }
        })
        .0;

    assert_ne!(
        chosen, x_id,
        "masked greedy argmax must NEVER select the forbidden token"
    );
    assert_eq!(
        chosen, t_id,
        "masked greedy argmax must select the highest-logit ALLOWED token ('t')"
    );
    // The forbidden token's masked logit is f32::MIN on device.
    assert_eq!(masked[x_id].to_bits(), f32::MIN.to_bits());
}
