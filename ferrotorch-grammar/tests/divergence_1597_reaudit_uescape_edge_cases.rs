//! #1597 RE-AUDIT (adversarial): the GPU string DFA in
//! `ferrotorch-grammar/src/gpu_dispatch.rs::compile_dfa_for_string` (after
//! the `e64fd7cc8` 3-hex -> 4-hex fix) must STILL produce a token-allow
//! mask byte-identical to the CPU oracle (`state.rs` `apply_step` +
//! `valid_next_chars_for`, driven through `compute_mask`) across the FULL
//! `\uXXXX` edge-case surface — not just the `\u123"` boundary the fix
//! targeted.
//!
//! R-CHAR-3: the CPU `compute_mask()` is the ORACLE. Every expected value
//! is the live CPU mask (`p.compute_mask()`), NOT copied from the GPU side.
//! The master assertion is `cpu.allow == gpu.allow` byte-for-byte over a
//! large probe vocab; the per-axis anchors assert the CPU oracle's truth
//! independently so a failure pins WHICH transition still diverges.
//!
//! Edge axes (each a distinct way the post-fix DFA could still diverge):
//!   A. Non-hex char at EACH hex position: `\uG..`, `\u1G.`, `\u12G`,
//!      `\u123G` (the original bug position), and after a complete escape.
//!   B. Truncated escapes ending mid-walk: `\u`, `\u1`, `\u12`, `\u123`
//!      (CPU accepts as a valid prefix — token steps without error).
//!   C. Uppercase / lowercase / mixed / all-digit hex bodies.
//!   D. JSON surrogate "pairs" `𐀀` — JSON does NOT validate
//!      surrogate pairing, so the oracle treats them as two independent
//!      `\uXXXX` escapes; the GPU must match whatever the oracle does.
//!   E. Multiple escapes in one token (`ሴ噸`, `\n\t`, mixed).
//!   F. Regression: the other short escapes and a plain string.
//!
//! Requires a CUDA device (RTX 3090 present) and the `cuda` feature.

#![cfg(feature = "cuda")]

use cubecl::prelude::Runtime;
use cubecl_cuda::{CudaDevice, CudaRuntime};
use ferrotorch_grammar::{JsonSchemaProcessor, PackedVocab, TokenMask, compute_mask_gpu};
use serde_json::json;

fn client() -> cubecl::prelude::ComputeClient<CudaRuntime> {
    CudaRuntime::client(&CudaDevice { index: 0 })
}

/// Every adversarial probe token. Mixed with the full single-ASCII vocab
/// so the structural opening `"` token exists and the masks are wide.
const PROBE_TOKENS: &[&str] = &[
    // ---- Axis A: non-hex at each position ----
    "\\uG123",   // non-hex at position 1
    "\\u1G23",   // non-hex at position 2
    "\\u12G3",   // non-hex at position 3
    "\\u123G",   // non-hex at position 4 (the original #1597 bug position)
    "\\u123\"",  // 3 hex then close quote (the #1597 boundary)
    "\\u123g",   // 3 hex then non-hex 'g'
    "\\u1234G",  // 4 hex (complete) then non-hex 'G' as body content
    "\\u1234\"", // 4 hex (complete) then close: legal
    // ---- Axis B: truncated escapes (valid prefixes) ----
    "\\u",     // \u with no hex digits yet
    "\\u1",    // 1 hex digit
    "\\u12",   // 2 hex digits
    "\\u123",  // 3 hex digits (the original over-accept ended here)
    "\\u1234", // 4 hex digits (escape complete, prefix)
    // ---- Axis C: uppercase / lowercase / mixed / all-digit ----
    "\\uABCD",   // all-uppercase hex
    "\\uabcd",   // all-lowercase hex
    "\\uAbCd",   // mixed-case hex
    "\\u0000",   // all-digit hex (zeros)
    "\\u9999",   // all-digit hex (nines)
    "\\uFfFf\"", // mixed-case then close: legal complete escape
    "\\ubeef",   // 'b' and 'f' are BOTH short-escapes and hex digits
    "\\uBEEF",   // uppercase variant
    "\\udead",   // hex with 'd','e','a' (all hex)
    // ---- Axis D: surrogate "pairs" (JSON does not validate pairing) ----
    "\\uD800\\uDC00",   // high+low surrogate, both well-formed escapes
    "\\uD800",          // lone high surrogate (well-formed \uXXXX)
    "\\uDC00",          // lone low surrogate (well-formed \uXXXX)
    "\\uD800\\uDC00\"", // surrogate pair then close
    // ---- Axis E: multiple escapes in one token ----
    "\\u1234\\u5678",   // two complete hex escapes
    "\\u1234\\u5678\"", // two escapes then close
    "\\n\\t",           // two short escapes
    "\\u1234\\n",       // hex escape then short escape
    "\\n\\u1234",       // short escape then hex escape
    "\\u1234ab\"",      // hex escape, body chars, close
    "\\u1234\\uGGGG",   // good escape then bad escape (non-hex)
    "\\u123\\u1234",    // 3-hex (incomplete) then another \u -> mid-escape '\\' invalid
    // ---- Axis F: regression — short escapes + plain ----
    "\\n",
    "\\t",
    "\\r",
    "\\b",
    "\\f",
    "\\\"",
    "\\\\",
    "\\/",
    "hello\"",   // plain string body then close
    "ab\\ncd\"", // body, short escape, body, close
];

fn escape_probe_vocab() -> Vec<String> {
    let mut v: Vec<String> = (0x20u8..=0x7Eu8).map(|b| (b as char).to_string()).collect();
    for probe in PROBE_TOKENS {
        v.push((*probe).to_string());
    }
    v
}

/// Top-level `{"type":"string"}` stepped past the opening `"` so the
/// grammar is at `InBody` (Phase::StringChars) — exactly the GPU
/// `StringEmissionStage::InBody` start state.
fn string_in_body(vocab: &[String]) -> JsonSchemaProcessor {
    let mut p = JsonSchemaProcessor::new(&json!({"type": "string"}), vocab.to_vec()).unwrap();
    let dq = vocab.iter().position(|t| t == "\"").unwrap() as u32;
    p.step_token(dq).unwrap();
    p
}

/// Same, but nested inside an array so the string frame has a parent
/// (parent terminators only fire at completion states; the escape walk
/// must be identical).
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
// MASTER: full byte-for-byte parity over the whole adversarial vocab.
// If ANY edge case diverges, this fails and the diff printer below
// names every divergent token.
// =====================================================================

#[test]
fn reaudit_uescape_full_mask_matches_cpu_byte_for_byte() {
    let vocab = escape_probe_vocab();
    let p = string_in_body(&vocab);
    let cpu = p.compute_mask();
    let gpu = gpu_mask(&p, &vocab);

    if cpu.allow != gpu.allow {
        let mut diffs = Vec::new();
        for (i, t) in vocab.iter().enumerate() {
            if cpu.allow[i] != gpu.allow[i] {
                diffs.push(format!(
                    "  token {t:?}: CPU oracle={} GPU={}",
                    cpu.allow[i], gpu.allow[i]
                ));
            }
        }
        panic!(
            "GPU string DFA diverges from CPU oracle on {} token(s):\n{}",
            diffs.len(),
            diffs.join("\n")
        );
    }
}

#[test]
fn reaudit_uescape_full_mask_matches_cpu_byte_for_byte_nested() {
    let vocab = escape_probe_vocab();
    let p = nested_string_in_body(&vocab);
    let cpu = p.compute_mask();
    let gpu = gpu_mask(&p, &vocab);

    if cpu.allow != gpu.allow {
        let mut diffs = Vec::new();
        for (i, t) in vocab.iter().enumerate() {
            if cpu.allow[i] != gpu.allow[i] {
                diffs.push(format!(
                    "  token {t:?}: CPU oracle={} GPU={}",
                    cpu.allow[i], gpu.allow[i]
                ));
            }
        }
        panic!(
            "GPU string DFA diverges from CPU oracle (nested) on {} token(s):\n{}",
            diffs.len(),
            diffs.join("\n")
        );
    }
}

// =====================================================================
// Axis A: non-hex at each hex position. The CPU oracle requires a hex
// digit at positions 1..=4; a non-hex char before the 4th hex digit is
// rejected. The GPU must match at EVERY position, not just position 4.
// =====================================================================

#[test]
fn reaudit_axis_a_nonhex_each_position_matches_oracle() {
    let vocab = escape_probe_vocab();
    let p = string_in_body(&vocab);
    let cpu = p.compute_mask();
    let gpu = gpu_mask(&p, &vocab);
    // CPU oracle anchors (independent of GPU): a non-hex char anywhere in
    // positions 1..=4 makes the escape malformed -> token rejected.
    for s in [
        "\\uG123", "\\u1G23", "\\u12G3", "\\u123G", "\\u123\"", "\\u123g",
    ] {
        let i = tok(&vocab, s);
        assert_eq!(
            cpu.allow[i], 0,
            "CPU oracle REJECTS {s:?} (non-hex / `\"` inside the 4-hex \\uXXXX walk)"
        );
        assert_eq!(
            gpu.allow[i], cpu.allow[i],
            "GPU must match CPU oracle for {s:?}"
        );
    }
    // After a COMPLETE 4-hex escape, the next char is ordinary body.
    for (s, expect) in [("\\u1234G", 1u32), ("\\u1234\"", 1u32)] {
        let i = tok(&vocab, s);
        assert_eq!(cpu.allow[i], expect, "CPU oracle anchor for {s:?}");
        assert_eq!(
            gpu.allow[i], cpu.allow[i],
            "GPU must match CPU oracle for {s:?}"
        );
    }
}

// =====================================================================
// Axis B: truncated escapes are valid PREFIXES — the CPU `compute_mask`
// accepts a token whose chars all step without error, even mid-escape.
// The GPU (no completion requirement either) must match.
// =====================================================================

#[test]
fn reaudit_axis_b_truncated_prefixes_match_oracle() {
    let vocab = escape_probe_vocab();
    let p = string_in_body(&vocab);
    let cpu = p.compute_mask();
    let gpu = gpu_mask(&p, &vocab);
    for s in ["\\u", "\\u1", "\\u12", "\\u123", "\\u1234"] {
        let i = tok(&vocab, s);
        assert_eq!(
            cpu.allow[i], 1,
            "CPU oracle ACCEPTS truncated prefix {s:?} (steps without error mid-walk)"
        );
        assert_eq!(
            gpu.allow[i], cpu.allow[i],
            "GPU must match CPU oracle for {s:?}"
        );
    }
}

// =====================================================================
// Axis C: hex case-folding + all-digit + b/f (short-escape AND hex).
// =====================================================================

#[test]
fn reaudit_axis_c_hex_case_and_digits_match_oracle() {
    let vocab = escape_probe_vocab();
    let p = string_in_body(&vocab);
    let cpu = p.compute_mask();
    let gpu = gpu_mask(&p, &vocab);
    for s in [
        "\\uABCD",
        "\\uabcd",
        "\\uAbCd",
        "\\u0000",
        "\\u9999",
        "\\uFfFf\"",
        "\\ubeef",
        "\\uBEEF",
        "\\udead",
    ] {
        let i = tok(&vocab, s);
        assert_eq!(
            cpu.allow[i], 1,
            "CPU oracle ACCEPTS hex body {s:?} (all chars are valid hex digits)"
        );
        assert_eq!(
            gpu.allow[i], cpu.allow[i],
            "GPU must match CPU oracle for {s:?}"
        );
    }
}

// =====================================================================
// Axis D: surrogate "pairs" are two independent \uXXXX escapes (JSON does
// not validate pairing). Whatever the CPU oracle decides, GPU must match.
// =====================================================================

#[test]
fn reaudit_axis_d_surrogate_pairs_match_oracle() {
    let vocab = escape_probe_vocab();
    let p = string_in_body(&vocab);
    let cpu = p.compute_mask();
    let gpu = gpu_mask(&p, &vocab);
    for s in ["\\uD800\\uDC00", "\\uD800", "\\uDC00", "\\uD800\\uDC00\""] {
        let i = tok(&vocab, s);
        // Anchor the oracle's actual decision (no GPU copy); each of these
        // is a well-formed \uXXXX sequence so the oracle ACCEPTS.
        assert_eq!(
            cpu.allow[i], 1,
            "CPU oracle treats {s:?} as well-formed independent \\uXXXX escapes"
        );
        assert_eq!(
            gpu.allow[i], cpu.allow[i],
            "GPU must match CPU oracle for {s:?}"
        );
    }
}

// =====================================================================
// Axis E: multiple escapes in one token. Exercises the body<->escape
// re-entry after a complete escape AND the mid-escape rejection.
// =====================================================================

#[test]
fn reaudit_axis_e_multiple_escapes_match_oracle() {
    let vocab = escape_probe_vocab();
    let p = string_in_body(&vocab);
    let cpu = p.compute_mask();
    let gpu = gpu_mask(&p, &vocab);
    // These re-enter the escape walk after returning to body — all legal.
    for s in [
        "\\u1234\\u5678",
        "\\u1234\\u5678\"",
        "\\n\\t",
        "\\u1234\\n",
        "\\n\\u1234",
        "\\u1234ab\"",
    ] {
        let i = tok(&vocab, s);
        assert_eq!(cpu.allow[i], 1, "CPU oracle ACCEPTS {s:?}");
        assert_eq!(
            gpu.allow[i], cpu.allow[i],
            "GPU must match CPU oracle for {s:?}"
        );
    }
    // Malformed second escape / incomplete-then-reescape -> rejected.
    for s in ["\\u1234\\uGGGG", "\\u123\\u1234"] {
        let i = tok(&vocab, s);
        assert_eq!(cpu.allow[i], 0, "CPU oracle REJECTS {s:?}");
        assert_eq!(
            gpu.allow[i], cpu.allow[i],
            "GPU must match CPU oracle for {s:?}"
        );
    }
}

// =====================================================================
// Axis F: regression — the other short escapes and plain bodies must be
// unperturbed by the hex-count fix.
// =====================================================================

#[test]
fn reaudit_axis_f_regression_short_escapes_and_plain_match_oracle() {
    let vocab = escape_probe_vocab();
    let p = string_in_body(&vocab);
    let cpu = p.compute_mask();
    let gpu = gpu_mask(&p, &vocab);
    for s in [
        "\\n",
        "\\t",
        "\\r",
        "\\b",
        "\\f",
        "\\\"",
        "\\\\",
        "\\/",
        "hello\"",
        "ab\\ncd\"",
    ] {
        let i = tok(&vocab, s);
        assert_eq!(cpu.allow[i], 1, "CPU oracle ACCEPTS regression token {s:?}");
        assert_eq!(
            gpu.allow[i], cpu.allow[i],
            "GPU must match CPU oracle for {s:?}"
        );
    }
}
