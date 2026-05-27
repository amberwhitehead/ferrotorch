//! #1594 re-audit (R-FIX-4): the GPU number DFA in
//! `ferrotorch-grammar/src/gpu_dispatch.rs::compile_dfa_for_number` must
//! produce a token-allow mask byte-identical to the CPU number DFA
//! (`state.rs` `apply_step` + `valid_next_chars_for` `Schema::Number`
//! arm) at *every* position of the JSON-number grammar — including the
//! exponent edge cases the fixer added (states 6/7/8:
//! AfterExponentMarker / AfterExponentSign / AfterExponentDigits).
//!
//! R-CHAR-3: the CPU DFA is the ORACLE. Every expected value here is the
//! live `JsonSchemaProcessor::compute_mask()` output (the CPU path), NOT
//! a literal copied from the ferrotorch GPU side. The assertion is the
//! byte-for-byte equality `cpu.allow == gpu.allow`, plus a small set of
//! per-char sanity probes whose expected truth is derived from the CPU
//! `valid_next_chars_for` source (cited inline), not from the GPU table.
//!
//! These tests require a CUDA device (RTX 3090 present) and the `cuda`
//! feature; without it the module is empty.
//!
//! Walks (top-level and array-nested numbers, the latter exercising the
//! completion/terminator gate that distinguishes states 6/7 from 8):
//!
//! 1. `1e5` — 'e' right after an integer digit (no decimal point).
//! 2. `1.e5` — 'e' right after the decimal POINT (mid-decimal); CPU
//!    REJECTS 'e' here (`had_digits && !mid_decimal` is false).
//! 3. `1e` — exponent marker then END; NOT complete (state 6).
//! 4. `1e+` / `1e-` — marker + sign then END; NOT complete (state 7).
//! 5. `1e+5` / `1E-3` / `1.5E10` — sign + digits; complete (state 8).
//! 6. `1e5e5` — second 'e' must be REJECTED inside the exponent.
//! 7. `1e05` — leading-zero exponent; CPU allows it (no JSON
//!    leading-zero restriction in the exponent).
//! 8. `+1e5` (rejected: JSON forbids leading '+') / `-1.5e-10`
//!    (signed mantissa + signed exponent).

#![cfg(feature = "cuda")]

use cubecl::prelude::Runtime;
use cubecl_cuda::{CudaDevice, CudaRuntime};
use ferrotorch_grammar::{JsonSchemaProcessor, PackedVocab, TokenMask, compute_mask_gpu};
use serde_json::json;

fn ascii_vocab() -> Vec<String> {
    (0x20u8..=0x7Eu8).map(|b| (b as char).to_string()).collect()
}

fn client() -> cubecl::prelude::ComputeClient<CudaRuntime> {
    CudaRuntime::client(&CudaDevice { index: 0 })
}

/// Build a top-level `{"type":"number"}` processor and step `prefix`
/// (each char is one single-char token in the ASCII vocab).
fn number_after(prefix: &str, vocab: &[String]) -> JsonSchemaProcessor {
    let mut p = JsonSchemaProcessor::new(&json!({"type": "number"}), vocab.to_vec()).unwrap();
    for c in prefix.chars() {
        let id = vocab.iter().position(|t| t == &c.to_string()).unwrap() as u32;
        p.step_token(id).unwrap();
    }
    p
}

/// Build an `{"type":"array","items":{"type":"number"}}` processor,
/// open the array with `[`, then step `prefix` so the number frame has a
/// non-empty parent-terminator set (`,` and `]`). This is what surfaces
/// the completion gate (states 6/7 NOT complete, state 8 complete) in the
/// token mask — the terminator is allowed iff the CPU number is complete.
fn array_number_after(prefix: &str, vocab: &[String]) -> JsonSchemaProcessor {
    let schema = json!({"type": "array", "items": {"type": "number"}});
    let mut p = JsonSchemaProcessor::new(&schema, vocab.to_vec()).unwrap();
    let step = |p: &mut JsonSchemaProcessor, c: char| {
        let id = vocab.iter().position(|t| t == &c.to_string()).unwrap() as u32;
        p.step_token(id).unwrap();
    };
    step(&mut p, '[');
    for c in prefix.chars() {
        step(&mut p, c);
    }
    p
}

/// CPU oracle mask. R-CHAR-3: this is the expected value, derived live
/// from the CPU DFA — never a literal copy of the GPU side.
fn cpu_oracle(p: &JsonSchemaProcessor) -> TokenMask {
    p.compute_mask()
}

/// GPU mask under test. Panics if the state is not DFA-compilable: the
/// whole point of #1594 is that number-exponent states ARE compilable,
/// so a `None` here is itself a divergence (the fix didn't take).
fn gpu_under_test(p: &JsonSchemaProcessor, vocab: &[String]) -> TokenMask {
    let cl = client();
    let packed = PackedVocab::pack(vocab);
    compute_mask_gpu::<CudaRuntime>(p, &cl, &packed)
        .expect("number-exponent stage must be GPU-DFA-compilable after #1594")
}

/// Assert the GPU mask equals the CPU oracle byte-for-byte, returning the
/// (identical) mask for further per-char sanity probes. The `label`
/// names the grammar position so a failure pins which transition diverged.
fn assert_gpu_eq_cpu(p: &JsonSchemaProcessor, vocab: &[String], label: &str) -> TokenMask {
    let cpu = cpu_oracle(p);
    let gpu = gpu_under_test(p, vocab);
    assert_eq!(
        cpu.allow, gpu.allow,
        "GPU number DFA mask must equal CPU oracle byte-for-byte at: {label}"
    );
    cpu
}

fn idx(vocab: &[String], c: char) -> usize {
    vocab.iter().position(|t| t == &c.to_string()).unwrap()
}

// =====================================================================
// Top-level: byte-for-byte GPU==CPU through every exponent position.
// =====================================================================

/// Edge 1 + 6 + 7: `1e5e5` walked one char at a time. Pins:
///   - 'e' allowed right after an integer digit (no decimal),
///   - second 'e' REJECTED inside the exponent (state 8),
///   - leading-zero exponent is implicitly covered by `1e05` below.
#[test]
fn edge1_e_after_integer_digit_and_edge6_double_e() {
    let vocab = ascii_vocab();

    // "1" — AfterDigitsNoDecimal. 'e'/'E' must be allowed.
    let p = number_after("1", &vocab);
    let m = assert_gpu_eq_cpu(&p, &vocab, "after \"1\" (AfterDigitsNoDecimal)");
    // CPU `valid_next_chars_for` Schema::Number: `had_digits && !mid_decimal`
    // pushes 'e' and 'E' (state.rs ~1802-1805).
    assert_eq!(
        m.allow[idx(&vocab, 'e')],
        1,
        "'e' allowed after integer digit"
    );
    assert_eq!(
        m.allow[idx(&vocab, 'E')],
        1,
        "'E' allowed after integer digit"
    );

    // "1e" — AfterExponentMarker. '+','-',digits allowed; '.','e' rejected.
    let p = number_after("1e", &vocab);
    let m = assert_gpu_eq_cpu(&p, &vocab, "after \"1e\" (AfterExponentMarker)");
    assert_eq!(m.allow[idx(&vocab, '+')], 1, "'+' allowed right after 'e'");
    assert_eq!(m.allow[idx(&vocab, '-')], 1, "'-' allowed right after 'e'");
    assert_eq!(
        m.allow[idx(&vocab, '0')],
        1,
        "digit allowed right after 'e'"
    );
    assert_eq!(m.allow[idx(&vocab, '.')], 0, "'.' rejected inside exponent");
    assert_eq!(m.allow[idx(&vocab, 'e')], 0, "'e' rejected right after 'e'");

    // "1e5" — AfterExponentDigits. digits allowed; '+','-','e' rejected.
    let p = number_after("1e5", &vocab);
    let m = assert_gpu_eq_cpu(&p, &vocab, "after \"1e5\" (AfterExponentDigits)");
    assert_eq!(m.allow[idx(&vocab, '5')], 1, "more exponent digits allowed");
    assert_eq!(
        m.allow[idx(&vocab, '+')],
        0,
        "'+' rejected after exponent digit"
    );
    assert_eq!(
        m.allow[idx(&vocab, '-')],
        0,
        "'-' rejected after exponent digit"
    );
    // Edge 6: the SECOND 'e' (double exponent) must be rejected.
    assert_eq!(
        m.allow[idx(&vocab, 'e')],
        0,
        "second 'e' (double exponent) rejected"
    );
    assert_eq!(
        m.allow[idx(&vocab, 'E')],
        0,
        "second 'E' (double exponent) rejected"
    );
}

/// Edge 2 (HIGH RISK): 'e' right after the decimal POINT. CPU's
/// `mid_decimal = had_decimal && !had_fractional_digit` gate makes
/// `had_digits && !mid_decimal` false, so 'e'/'E' is NOT emitted
/// (state.rs ~1791,1802). Contrast `1.5e5` where the fractional digit
/// clears mid_decimal and 'e' IS allowed.
#[test]
fn edge2_e_after_decimal_point_rejected_but_allowed_after_frac_digit() {
    let vocab = ascii_vocab();

    // "1." — AfterDecimalNoFrac (mid_decimal). Only digits; NO 'e'/'E'.
    let p = number_after("1.", &vocab);
    let m = assert_gpu_eq_cpu(&p, &vocab, "after \"1.\" (AfterDecimalNoFrac, mid_decimal)");
    assert_eq!(m.allow[idx(&vocab, '0')], 1, "digit allowed mid-decimal");
    assert_eq!(
        m.allow[idx(&vocab, 'e')],
        0,
        "'e' REJECTED right after '.' (mid_decimal)"
    );
    assert_eq!(
        m.allow[idx(&vocab, 'E')],
        0,
        "'E' REJECTED right after '.' (mid_decimal)"
    );
    assert_eq!(m.allow[idx(&vocab, '.')], 0, "second '.' rejected");

    // "1.5" — AfterFractionalDigits. NOW 'e'/'E' allowed.
    let p = number_after("1.5", &vocab);
    let m = assert_gpu_eq_cpu(&p, &vocab, "after \"1.5\" (AfterFractionalDigits)");
    assert_eq!(
        m.allow[idx(&vocab, 'e')],
        1,
        "'e' allowed after a fractional digit"
    );
    assert_eq!(
        m.allow[idx(&vocab, 'E')],
        1,
        "'E' allowed after a fractional digit"
    );

    // "1.5e10" walked to completion — GPU==CPU all the way.
    for prefix in ["1.5e", "1.5e1", "1.5e10"] {
        let p = number_after(prefix, &vocab);
        assert_gpu_eq_cpu(&p, &vocab, prefix);
    }
    // Edge 5: "1.5E10" with capital E.
    for prefix in ["1.5E", "1.5E1", "1.5E10"] {
        let p = number_after(prefix, &vocab);
        assert_gpu_eq_cpu(&p, &vocab, prefix);
    }
}

/// Edge 5 + 8: signed exponent and signed mantissa. `1e+5`, `1E-3`,
/// `-1.5e-10`. Also pins that `+` as a LEADING char is rejected (JSON
/// forbids a leading '+'; CPU Phase::Start only emits digits and '-').
#[test]
fn edge5_signed_exponent_and_edge8_signed_mantissa() {
    let vocab = ascii_vocab();

    // Leading char of a number: digits or '-' only, NEVER '+'.
    let p = JsonSchemaProcessor::new(&json!({"type": "number"}), vocab.clone()).unwrap();
    let m = assert_gpu_eq_cpu(&p, &vocab, "Number @ Phase::Start");
    assert_eq!(m.allow[idx(&vocab, '-')], 1, "'-' allowed as leading sign");
    assert_eq!(
        m.allow[idx(&vocab, '+')],
        0,
        "'+' NOT allowed as leading sign (JSON)"
    );

    // "1e+" then "1e+5" — sign then digits.
    for prefix in ["1e+", "1e+5"] {
        let p = number_after(prefix, &vocab);
        assert_gpu_eq_cpu(&p, &vocab, prefix);
    }
    // "1E-" then "1E-3".
    for prefix in ["1E-", "1E-3"] {
        let p = number_after(prefix, &vocab);
        assert_gpu_eq_cpu(&p, &vocab, prefix);
    }

    // After "1e+", AfterExponentSign: only digits valid; a second sign or
    // 'e' rejected (state.rs: `!had_exponent_sign && !had_exponent_digit`
    // gate on the sign branch; the marker branch needs `!had_exponent_marker`).
    let p = number_after("1e+", &vocab);
    let m = assert_gpu_eq_cpu(&p, &vocab, "after \"1e+\" (AfterExponentSign)");
    assert_eq!(
        m.allow[idx(&vocab, '5')],
        1,
        "digit allowed after exponent sign"
    );
    assert_eq!(
        m.allow[idx(&vocab, '+')],
        0,
        "second '+' rejected after exponent sign"
    );
    assert_eq!(
        m.allow[idx(&vocab, '-')],
        0,
        "'-' rejected after exponent sign"
    );
    assert_eq!(
        m.allow[idx(&vocab, 'e')],
        0,
        "'e' rejected after exponent sign"
    );

    // Full signed-both: "-1.5e-10" walked to completion.
    for prefix in [
        "-", "-1", "-1.", "-1.5", "-1.5e", "-1.5e-", "-1.5e-1", "-1.5e-10",
    ] {
        let p = number_after(prefix, &vocab);
        assert_gpu_eq_cpu(&p, &vocab, prefix);
    }
}

/// Edge 7: leading-zero exponent `1e05`. JSON places no leading-zero
/// restriction inside the exponent, so the CPU allows `0` right after the
/// marker and another digit after that. GPU must match.
#[test]
fn edge7_leading_zero_exponent() {
    let vocab = ascii_vocab();
    for prefix in ["1e0", "1e05"] {
        let p = number_after(prefix, &vocab);
        let m = assert_gpu_eq_cpu(&p, &vocab, prefix);
        // After "1e0" (exponent digit present), more digits remain valid.
        assert_eq!(
            m.allow[idx(&vocab, '5')],
            1,
            "exponent leading zero then more digits"
        );
    }
}

// =====================================================================
// Array-nested: the completion gate (states 6/7 NOT complete, 8 complete)
// becomes observable because the number frame now has parent terminators
// ',' and ']'. Edge 3 / 4 live here. THESE are the highest-risk cases.
// =====================================================================

/// Edge 3 (HIGH RISK): `[1e` — exponent marker then the parent
/// terminator `]`. CPU emits parent terminators only once
/// `had_exponent_digit` (state.rs ~1786). At the bare marker the number
/// is NOT complete, so `]` and `,` must be REJECTED. GPU's
/// `complete_states` excludes state 6, so `add_terminators_to_states`
/// must NOT have added the terminator there.
#[test]
fn edge3_incomplete_at_exponent_marker_terminator_rejected() {
    let vocab = ascii_vocab();

    // "[1" — number complete, terminators ']' and ',' allowed.
    let p = array_number_after("1", &vocab);
    let m = assert_gpu_eq_cpu(&p, &vocab, "array after \"[1\" (number complete)");
    assert_eq!(
        m.allow[idx(&vocab, ']')],
        1,
        "']' allowed: integer is complete"
    );

    // "[1e" — AfterExponentMarker, NOT complete. ']' and ',' rejected.
    let p = array_number_after("1e", &vocab);
    let m = assert_gpu_eq_cpu(
        &p,
        &vocab,
        "array after \"[1e\" (AfterExponentMarker, NOT complete)",
    );
    assert_eq!(
        m.allow[idx(&vocab, ']')],
        0,
        "']' REJECTED: bare exponent marker is NOT complete"
    );
    assert_eq!(
        m.allow[idx(&vocab, ',')],
        0,
        "',' REJECTED: bare exponent marker is NOT complete"
    );
    assert_eq!(m.allow[idx(&vocab, '0')], 1, "exponent digit still allowed");
    assert_eq!(m.allow[idx(&vocab, '+')], 1, "'+' still allowed at marker");
}

/// Edge 4 (HIGH RISK): `[1e+` / `[1e-` — marker + sign then the parent
/// terminator. AfterExponentSign is NOT complete; `]` / `,` rejected.
/// Then `[1e+5` completes and the terminator is allowed.
#[test]
fn edge4_incomplete_at_exponent_sign_terminator_rejected() {
    let vocab = ascii_vocab();

    for prefix in ["1e+", "1e-"] {
        let p = array_number_after(prefix, &vocab);
        let m = assert_gpu_eq_cpu(
            &p,
            &vocab,
            &format!("array after \"[{prefix}\" (AfterExponentSign)"),
        );
        assert_eq!(
            m.allow[idx(&vocab, ']')],
            0,
            "']' REJECTED: exponent sign is NOT complete"
        );
        assert_eq!(
            m.allow[idx(&vocab, ',')],
            0,
            "',' REJECTED: exponent sign is NOT complete"
        );
        assert_eq!(
            m.allow[idx(&vocab, '5')],
            1,
            "digit allowed after exponent sign"
        );
    }

    // "[1e+5" — AfterExponentDigits, complete. Terminator now allowed.
    let p = array_number_after("1e+5", &vocab);
    let m = assert_gpu_eq_cpu(
        &p,
        &vocab,
        "array after \"[1e+5\" (AfterExponentDigits, complete)",
    );
    assert_eq!(
        m.allow[idx(&vocab, ']')],
        1,
        "']' allowed: exponent has a digit, complete"
    );
    assert_eq!(
        m.allow[idx(&vocab, ',')],
        1,
        "',' allowed: exponent has a digit, complete"
    );
    assert_eq!(m.allow[idx(&vocab, '5')], 1, "more exponent digits allowed");
}

/// Cross-check the nested walk end-to-end for `[1.5e-10]`, asserting
/// GPU==CPU at each position including the mid-decimal mid-exponent gates.
#[test]
fn nested_full_walk_byte_identical() {
    let vocab = ascii_vocab();
    for prefix in ["1", "1.", "1.5", "1.5e", "1.5e-", "1.5e-1", "1.5e-10"] {
        let p = array_number_after(prefix, &vocab);
        assert_gpu_eq_cpu(&p, &vocab, &format!("array after \"[{prefix}\""));
    }
}
