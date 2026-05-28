//! #1596 re-audit (R-FIX-4): the GPU string ESCAPE DFA in
//! `ferrotorch-grammar/src/gpu_dispatch.rs::compile_dfa_for_string` must
//! produce a token-allow mask byte-identical to the CPU string+escape DFA
//! (`state.rs` `apply_step` + `valid_next_chars_for` for
//! `(Schema::String, Phase::StringChars)` and `(Schema::String,
//! Phase::StringEscape)`) for every escape sequence.
//!
//! R-CHAR-3: the CPU DFA is the ORACLE. Every expected value here is the
//! live `JsonSchemaProcessor::compute_mask()` output (the CPU path), NOT a
//! literal copied from the GPU side. The primary assertion is the
//! byte-for-byte equality `cpu.allow == gpu.allow`. A few sanity probes
//! also anchor the CPU oracle's truth directly so a failure pins which
//! escape transition diverged.
//!
//! Why MULTI-CHAR tokens? The single-char ASCII vocab only exercises the
//! per-position mask, and `JsonGrammar::string_emission_stage_top` returns
//! `None` for `Phase::StringEscape` (state.rs:658-662), so a mid-escape
//! processor state would make `compute_mask_gpu` fall back to `None` and
//! never run the GPU escape DFA at all. To exercise the GPU
//! `compile_dfa_for_string` escape walk (states 3/4/5/6) we hand it
//! multi-char tokens whose chars are walked *inside* the kernel from the
//! `InBody` start state — exactly mirroring how the CPU `compute_mask`
//! walks each token's chars via `step_char` from the same `InBody` state.
//!
//! HIGHEST-RISK AXIS — the `\uXXXX` hex-digit count boundary:
//!   CPU oracle (`state.rs` apply_step `Phase::StringEscape`, lines
//!   1118-1163): after `\u` we set `hex_digits=1`, then accept a hex digit
//!   at hex_digits = 1, 2, 3 AND 4 (the 4th is the `hex_digits==4`
//!   `valid_next_chars_for` branch at state.rs:1819-1826; apply_step's
//!   `new_n == 5` arm at line 1154 resolves to body). So the CPU accepts
//!   EXACTLY 4 hex digits after `\u`.
//!
//!   GPU DFA (`gpu_dispatch.rs` compile_dfa_for_string, lines 332-374):
//!   states are escape_start(3) --u--> hex1(4) --hex--> hex2(5) --hex-->
//!   hex3(6) --hex--> body(1). That is escape_start + only THREE hex-digit
//!   transitions (4->5, 5->6, 6->1) before returning to body. So the GPU
//!   accepts EXACTLY 3 hex digits after `\u`, then treats the 4th char as
//!   ordinary body content.
//!
//!   => `\u123"` (3 hex digits then closing quote):
//!        CPU: at hex_digits=4, valid set is hex-only -> '"' REJECTED ->
//!             the whole token is rejected.
//!        GPU: after `\u123` already back in body(1) -> '"' closes(2) ->
//!             the whole token is ACCEPTED.
//!      The GPU OVER-accepts a malformed 3-hex-digit escape.
//!
//! Requires a CUDA device (RTX 3090 present) and the `cuda` feature;
//! without it the module is empty.

#![cfg(feature = "cuda")]

use cubecl::prelude::Runtime;
use cubecl_cuda::{CudaDevice, CudaRuntime};
use ferrotorch_grammar::{JsonSchemaProcessor, PackedVocab, TokenMask, compute_mask_gpu};
use serde_json::json;

fn client() -> cubecl::prelude::ComputeClient<CudaRuntime> {
    CudaRuntime::client(&CudaDevice { index: 0 })
}

/// A vocab of all single ASCII chars (so the structural opening `"` token
/// exists) plus the multi-char escape-probe tokens we want to mask. The
/// multi-char tokens are appended at the end; their indices are looked up
/// by string value so order is irrelevant.
fn escape_probe_vocab() -> Vec<String> {
    let mut v: Vec<String> = (0x20u8..=0x7Eu8).map(|b| (b as char).to_string()).collect();
    for probe in PROBE_TOKENS {
        v.push((*probe).to_string());
    }
    v
}

/// The multi-char escape tokens under test. Each is walked from `InBody`.
const PROBE_TOKENS: &[&str] = &[
    // --- \uXXXX hex-digit-count boundary (highest risk) ---
    "\\u123\"",  // 3 hex digits then close: CPU REJECTS (mid-escape), GPU accepts
    "\\u12\"",   // 2 hex digits then close: CPU REJECTS, GPU rejects (control)
    "\\u1234\"", // 4 hex digits then close: CPU ACCEPTS (control — both should accept)
    "\\u12345",  // 5 hex chars: CPU treats the 5th as body content (accept), GPU too
    "\\u123g",   // 3 hex digits then non-hex 'g': CPU REJECTS (mid-escape needs hex)
    // --- short escapes ---
    "\\n",  // valid short escape
    "\\t",  // valid short escape
    "\\/",  // forward-slash short escape (JSON-legal)
    "\\\"", // escaped quote
    "\\\\", // escaped backslash
    "\\b",  // 'b' short escape (also a hex digit)
    "\\f",  // 'f' short escape (also a hex digit)
    // --- invalid escapes ---
    "\\x", // 'x' is NOT a valid escape char -> CPU REJECTS
    "\\z", // 'z' invalid escape -> CPU REJECTS
    "\\1", // digit after backslash is NOT a valid escape -> CPU REJECTS
    // --- escape then body then close ---
    "\\nab\"", // short escape, body chars, close: CPU ACCEPTS
];

/// Build a top-level `{"type":"string"}` processor and step the opening
/// `"` so the grammar is at `InBody` (Phase::StringChars). The escape
/// probe tokens are then walked from this state by both CPU and GPU.
fn string_in_body(vocab: &[String]) -> JsonSchemaProcessor {
    let mut p = JsonSchemaProcessor::new(&json!({"type": "string"}), vocab.to_vec()).unwrap();
    let dq = vocab.iter().position(|t| t == "\"").unwrap() as u32;
    p.step_token(dq).unwrap();
    p
}

/// Same, but the string is nested inside an array (`["` stepped). Axis 6:
/// the escape DFA must behave identically when the string frame has a
/// parent (the `,`/`]` terminators only fire at completion states).
fn nested_string_in_body(vocab: &[String]) -> JsonSchemaProcessor {
    let schema = json!({"type": "array", "items": {"type": "string"}});
    let mut p = JsonSchemaProcessor::new(&schema, vocab.to_vec()).unwrap();
    let step = |p: &mut JsonSchemaProcessor, s: &str| {
        let id = vocab.iter().position(|t| t == s).unwrap() as u32;
        p.step_token(id).unwrap();
    };
    step(&mut p, "[");
    step(&mut p, "\"");
    p
}

fn gpu_mask(p: &JsonSchemaProcessor, vocab: &[String]) -> TokenMask {
    let cl = client();
    let packed = PackedVocab::pack(vocab);
    compute_mask_gpu::<CudaRuntime>(p, &cl, &packed)
        .expect("String InBody must be GPU-DFA-compilable")
}

fn tok(vocab: &[String], s: &str) -> usize {
    vocab
        .iter()
        .position(|t| t == s)
        .unwrap_or_else(|| panic!("probe token {s:?} not in vocab"))
}

// =====================================================================
// Byte-for-byte parity: the canonical CPU<->GPU invariant for #1596.
// =====================================================================

/// The full mask over the escape-probe vocab MUST be byte-identical
/// CPU vs GPU at `InBody`. This single assertion catches every escape
/// divergence at once; the per-token tests below pin the specific
/// transitions for the report.
#[test]
fn string_escape_gpu_mask_matches_cpu_byte_for_byte() {
    let vocab = escape_probe_vocab();
    let p = string_in_body(&vocab);
    let cpu = p.compute_mask();
    let gpu = gpu_mask(&p, &vocab);
    assert_eq!(
        cpu.allow, gpu.allow,
        "GPU string-escape DFA must equal CPU oracle byte-for-byte at InBody"
    );
}

/// Axis 6: same byte-for-byte invariant when the string is nested in an
/// array (the fixer's `nested_string_in_array_after_open_quote` covers
/// the body, this re-verifies the ESCAPE walk in the nested case).
#[test]
fn string_escape_gpu_mask_matches_cpu_byte_for_byte_nested() {
    let vocab = escape_probe_vocab();
    let p = nested_string_in_body(&vocab);
    let cpu = p.compute_mask();
    let gpu = gpu_mask(&p, &vocab);
    assert_eq!(
        cpu.allow, gpu.allow,
        "GPU string-escape DFA must equal CPU oracle byte-for-byte at nested InBody"
    );
}

// =====================================================================
// Axis 2 (HIGHEST RISK): \uXXXX hex-digit count is EXACTLY 4.
// =====================================================================

/// `\u123"` — exactly 3 hex digits then a closing quote.
///
/// CPU oracle (state.rs apply_step Phase::StringEscape): after `\u123`
/// we are at `hex_digits=4`, whose `valid_next_chars_for`
/// (state.rs:1819-1826) is hex-digits-ONLY. A `"` is therefore rejected
/// mid-escape, so the whole token `\u123"` is rejected.
///
/// GPU DFA (gpu_dispatch.rs:371): after only 3 hex transitions the walk
/// is already back in body(1), so `"` closes the string and the token is
/// accepted. This is the 3-vs-4 hex-count boundary divergence.
#[test]
fn divergence_u_escape_three_hex_then_quote_over_accepted_by_gpu() {
    let vocab = escape_probe_vocab();
    let p = string_in_body(&vocab);

    // CPU oracle anchor (NOT copied from the GPU side): the CPU rejects
    // `\u123"` because after 3 hex digits the escape is mid-`\uXXXX`.
    let cpu = p.compute_mask();
    let i = tok(&vocab, "\\u123\"");
    assert_eq!(
        cpu.allow[i], 0,
        "CPU oracle REJECTS `\\u123\\\"` — only 3 of the required 4 hex digits \
         before the closing quote (state.rs:1819 hex_digits==4 needs another hex)"
    );

    // The GPU mask must match the CPU oracle. The current GPU DFA accepts
    // this token (3-hex-digit escape resolves to body early), so this
    // FAILS — the divergence is pinned.
    let gpu = gpu_mask(&p, &vocab);
    assert_eq!(
        gpu.allow[i], cpu.allow[i],
        "GPU must REJECT `\\u123\\\"` to match the CPU oracle (4 hex digits required, \
         GPU only requires 3)"
    );
}

/// `\u123g` — exactly 3 hex digits then a non-hex char `g`.
///
/// CPU oracle: after `\u123` at `hex_digits=4`, only a hex digit is valid;
/// `g` is not a hex digit -> the token is rejected. GPU: after 3 hex
/// digits already in body, `g` is ordinary body content -> accepted.
#[test]
fn divergence_u_escape_three_hex_then_nonhex_over_accepted_by_gpu() {
    let vocab = escape_probe_vocab();
    let p = string_in_body(&vocab);

    let cpu = p.compute_mask();
    let i = tok(&vocab, "\\u123g");
    assert_eq!(
        cpu.allow[i], 0,
        "CPU oracle REJECTS `\\u123g` — `g` arrives at hex_digits==4 which \
         requires a 4th hex digit, not arbitrary body content"
    );

    let gpu = gpu_mask(&p, &vocab);
    assert_eq!(
        gpu.allow[i], cpu.allow[i],
        "GPU must REJECT `\\u123g` to match the CPU oracle"
    );
}

/// Control: `ሴ"` — exactly 4 hex digits then close. BOTH sides
/// should ACCEPT (the legal `\uXXXX` form). If this fails too, the GPU
/// is off-by-more-than-one rather than exactly one short.
#[test]
fn control_u_escape_four_hex_then_quote_accepted_by_both() {
    let vocab = escape_probe_vocab();
    let p = string_in_body(&vocab);
    let cpu = p.compute_mask();
    let i = tok(&vocab, "\\u1234\"");
    assert_eq!(
        cpu.allow[i], 1,
        "CPU oracle ACCEPTS the legal `\\u1234\\\"` (4 hex digits then close)"
    );
    let gpu = gpu_mask(&p, &vocab);
    assert_eq!(gpu.allow[i], cpu.allow[i], "GPU must accept `\\u1234\\\"`");
}

// =====================================================================
// Axes 1/3/5: short escapes, invalid escapes, escape-then-body.
// These are expected to AGREE (the fixer modeled them); they're here to
// pin the exact post-`\` valid set and to prove the divergence is the
// hex-count boundary specifically, not the short-escape table.
// =====================================================================

/// Axis 1: every legal short escape is accepted by the CPU oracle and the
/// GPU must match. Axis 3: every illegal escape char is rejected.
#[test]
fn short_and_invalid_escapes_match_cpu_oracle() {
    let vocab = escape_probe_vocab();
    let p = string_in_body(&vocab);
    let cpu = p.compute_mask();
    let gpu = gpu_mask(&p, &vocab);

    // Legal short escapes (state.rs:1812 valid set " \ / b f n r t u).
    for s in ["\\n", "\\t", "\\/", "\\\"", "\\\\", "\\b", "\\f"] {
        let i = tok(&vocab, s);
        assert_eq!(cpu.allow[i], 1, "CPU oracle ACCEPTS short escape {s:?}");
        assert_eq!(gpu.allow[i], cpu.allow[i], "GPU must match CPU for {s:?}");
    }
    // Illegal escape chars (not in the post-`\` valid set).
    for s in ["\\x", "\\z", "\\1"] {
        let i = tok(&vocab, s);
        assert_eq!(
            cpu.allow[i], 0,
            "CPU oracle REJECTS invalid escape {s:?} (not in `\" \\ / b f n r t u`)"
        );
        assert_eq!(gpu.allow[i], cpu.allow[i], "GPU must match CPU for {s:?}");
    }
}

/// Axis 5: a completed short escape followed by body chars and a closing
/// quote (`\nab"`) is a complete legal string. Both sides accept.
#[test]
fn escape_then_body_then_close_matches_cpu_oracle() {
    let vocab = escape_probe_vocab();
    let p = string_in_body(&vocab);
    let cpu = p.compute_mask();
    let gpu = gpu_mask(&p, &vocab);
    let i = tok(&vocab, "\\nab\"");
    assert_eq!(
        cpu.allow[i], 1,
        "CPU oracle ACCEPTS `\\nab\\\"` (escape, body, close)"
    );
    assert_eq!(
        gpu.allow[i], cpu.allow[i],
        "GPU must match CPU for `\\nab\\\"`"
    );
}
