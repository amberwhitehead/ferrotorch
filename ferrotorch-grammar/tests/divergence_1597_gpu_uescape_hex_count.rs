//! #1597 — GPU `\uXXXX` hex-digit COUNT divergence.
//!
//! The GPU string DFA in
//! `ferrotorch-grammar/src/gpu_dispatch.rs::compile_dfa_for_string`
//! accepts EXACTLY **3** hex digits after `\u`, then returns to the body
//! and treats the 4th character as ordinary body content. The CPU oracle
//! (`ferrotorch-grammar/src/state.rs`, the
//! `(Schema::String, Phase::StringEscape)` arm) requires EXACTLY **4** hex
//! digits — the JSON spec form `\uXXXX`.
//!
//! R-CHAR-3: the CPU DFA is the ORACLE here (this is a GPU↔CPU-oracle
//! divergence, NOT a torch call). Every expected value below is taken
//! from the CPU oracle's own `JsonSchemaProcessor::compute_mask()` output,
//! never literal-copied from the GPU side. The headline invariant is
//! `gpu.allow[i] == cpu.allow[i]` for each `\uXXXX` probe; the CPU-only
//! anchor assertions document WHY the oracle says what it says, citing the
//! exact `state.rs` lines, so the fixer can read the precise rule.
//!
//! Precise off-by-one (the fixer's spec):
//!
//!   CPU oracle — `state.rs`:
//!     * `valid_next_chars_for` `(Schema::String, Phase::StringEscape)`
//!       (state.rs:1809-1826): with `hex_digits == 0` the valid set is the
//!       short escapes + `u`; with `hex_digits` in 1..=4 the valid set is
//!       hex-only. The `hex_digits == 4` branch (state.rs:1819-1826) means
//!       a 4th hex digit is REQUIRED — a non-hex char (incl. the closing
//!       `"`) at `hex_digits == 4` is REJECTED.
//!     * `apply_step` `(Schema::String, Phase::StringEscape)`
//!       (state.rs:1142-1163): `new_n = hex_digits + 1`; only when
//!       `new_n == 5` (i.e. the 4th hex digit, seen at `hex_digits == 4`)
//!       does the walk push the resolved codepoint and return to body
//!       (`Phase::StringChars`). So the CPU accepts EXACTLY 4 hex digits.
//!
//!   GPU DFA — `gpu_dispatch.rs::compile_dfa_for_string`:
//!     * states are escape_start(3) --u--> hex1(4) --hex--> hex2(5)
//!       --hex--> hex3(6) --hex--> body(1). The hex-walk transition table
//!       at `gpu_dispatch.rs:371` is exactly
//!       `&[(hex1, hex2), (hex2, hex3), (hex3, 1u32)]` — only THREE
//!       hex-digit transitions (4→5, 5→6, 6→1). There is NO `hex4` state.
//!       So after `\u` + 3 hex digits the walk is already back in body(1),
//!       and the 4th char is matched as ordinary body content.
//!
//!   => GPU accepts EXACTLY 3 hex digits; CPU oracle requires EXACTLY 4.
//!
//! The fix is at `gpu_dispatch.rs:371` (and `num_states` at line 332,
//! plus the state-numbering doc at lines 288-296): add a 4th hex state
//! `hex4` so the walk is hex1→hex2→hex3→hex4→body, i.e. FOUR hex
//! transitions before returning to body(1).
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

/// The `\uXXXX`-count probe tokens. Each is walked from `InBody` by both
/// the CPU oracle and the GPU DFA. Naming uses the hex-digit count and
/// the trailing char so the report reads cleanly.
const PROBE_TOKENS: &[&str] = &[
    // n=3 hex then a TERMINATOR — the core off-by-one. CPU rejects
    // (mid-escape at hex_digits==4 needs a hex), GPU accepts (already in
    // body, `"` closes). This proves the GPU "accepts 3 and stops".
    "\\u123\"",
    // n=3 hex then a NON-HEX, NON-TERMINATOR body char. CPU rejects
    // (hex_digits==4 needs a hex), GPU accepts (`g` is body content).
    // This distinguishes "accepts 3 and stops" from "accepts 3 then
    // strictly needs a 4th": the GPU does NOT require a 4th — it admits
    // arbitrary body content as the 4th char.
    "\\u123g",
    // n=3 hex then a 4th HEX then close — the legal `\uXXXX`. CPU
    // ACCEPTS. GPU also accepts (its 4th char `4` is body content, then
    // `"` closes). This is the control: it shows the GPU accepts the
    // legal form too, so the bug is OVER-acceptance of the 3-digit form,
    // not UNDER-acceptance of the 4-digit form.
    "\\u1234\"",
    // n=2 hex then close — too short for BOTH. CPU rejects (hex_digits==3
    // needs hex), GPU rejects (state hex2(5) needs hex, `"`→reject).
    // Lower control: confirms the GPU isn't simply accepting everything.
    "\\u12\"",
    // n=1 hex then close — too short for BOTH. CPU rejects, GPU rejects.
    "\\u1\"",
    // n=0 hex (just `\u`) then close — CPU rejects (hex_digits==1 needs
    // hex), GPU rejects (hex1(4) needs hex). Both reject.
    "\\u\"",
];

/// Vocab of every single printable ASCII char (so the structural opening
/// `"` token exists and body content chars are present) plus the
/// multi-char `\uXXXX` probe tokens appended at the end. Probe indices are
/// looked up by string value, so append order is irrelevant.
fn uescape_vocab() -> Vec<String> {
    let mut v: Vec<String> = (0x20u8..=0x7Eu8).map(|b| (b as char).to_string()).collect();
    for probe in PROBE_TOKENS {
        v.push((*probe).to_string());
    }
    v
}

/// Build a top-level `{"type":"string"}` processor and step the opening
/// `"` so the grammar is at `InBody` (Phase::StringChars). The probe
/// tokens are then walked from this state by both CPU and GPU (mirroring
/// the #1596 harness so the two tests share an identical setup).
fn string_in_body(vocab: &[String]) -> JsonSchemaProcessor {
    let mut p = JsonSchemaProcessor::new(&json!({"type": "string"}), vocab.to_vec()).unwrap();
    let dq = vocab.iter().position(|t| t == "\"").unwrap() as u32;
    p.step_token(dq).unwrap();
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
// Headline: the GPU `\uXXXX` mask must equal the CPU oracle for every
// hex-count probe. FAILS on current code (GPU over-accepts 3-hex forms).
// =====================================================================

/// Byte-for-byte over the `\uXXXX`-count probes only. The CPU oracle is
/// the authority; the GPU must match it. The current GPU DFA accepts the
/// 3-hex forms `\u123"` and `\u123g` that the CPU rejects, so this fails.
#[test]
fn divergence_1597_uescape_hex_count_mask_matches_cpu_oracle() {
    let vocab = uescape_vocab();
    let p = string_in_body(&vocab);
    let cpu = p.compute_mask();
    let gpu = gpu_mask(&p, &vocab);

    for &probe in PROBE_TOKENS {
        let i = tok(&vocab, probe);
        assert_eq!(
            gpu.allow[i], cpu.allow[i],
            "GPU vs CPU-oracle mismatch for {probe:?}: GPU={}, CPU oracle={} \
             (GPU accepts 3 hex digits after `\\u`; CPU oracle requires 4 — \
             state.rs:1819 / apply_step new_n==5)",
            gpu.allow[i], cpu.allow[i],
        );
    }
}

// =====================================================================
// Pinning the PRECISE off-by-one so the fixer knows exactly what changes.
// =====================================================================

/// `\u123"` — 3 hex digits then the closing quote.
///
/// CPU oracle ANCHOR (R-CHAR-3, NOT copied from GPU): after `\u123` the
/// state is `Phase::StringEscape { hex_digits: 4 }`; its valid set
/// (state.rs:1819-1826) is hex-only, so `"` is rejected mid-escape and
/// the whole token is rejected.
///
/// GPU: after only 3 hex transitions the walk is back in body(1), so `"`
/// closes the string and the token is ACCEPTED. This proves the GPU
/// "accepts 3 hex digits and returns to body" (NOT "accepts 3 then
/// strictly requires a 4th").
#[test]
fn divergence_1597_three_hex_then_quote_gpu_closes_early() {
    let vocab = uescape_vocab();
    let p = string_in_body(&vocab);

    // CPU oracle anchor: the canonical truth is that `\u123"` is rejected.
    let cpu = p.compute_mask();
    let i = tok(&vocab, "\\u123\"");
    assert_eq!(
        cpu.allow[i], 0,
        "CPU oracle REJECTS `\\u123\\\"` — only 3 of the required 4 hex digits \
         before the closing quote (state.rs:1819 hex_digits==4 needs a 4th hex; \
         apply_step resolves to body only at new_n==5, state.rs:1154)"
    );

    // The GPU must match the oracle. It does not (it closes early), so
    // this FAILS — the over-acceptance is pinned.
    let gpu = gpu_mask(&p, &vocab);
    assert_eq!(
        gpu.allow[i], cpu.allow[i],
        "GPU must REJECT `\\u123\\\"` to match the CPU oracle. GPU accepts it \
         because gpu_dispatch.rs:371 wires only 3 hex transitions \
         (hex1→hex2→hex3→body), so `\"` after 3 hex digits closes the string."
    );
}

/// `\u123g` — 3 hex digits then a non-hex, non-terminator body char `g`.
///
/// This distinguishes the two possible GPU bugs:
///   (a) "accepts 3 hex then strictly requires a hex 4th" — would REJECT
///       `g` (matching the CPU), OR
///   (b) "accepts 3 hex then is back in body, any content allowed" — would
///       ACCEPT `g`.
/// The CPU oracle is (a)'s expectation; if the GPU accepts `g`, the GPU is
/// behavior (b): it does NOT enforce a hex 4th at all — it is purely
/// 3-hex-then-body.
#[test]
fn divergence_1597_three_hex_then_nonhex_pins_body_fallthrough() {
    let vocab = uescape_vocab();
    let p = string_in_body(&vocab);

    let cpu = p.compute_mask();
    let i = tok(&vocab, "\\u123g");
    assert_eq!(
        cpu.allow[i], 0,
        "CPU oracle REJECTS `\\u123g` — at hex_digits==4 the valid set is \
         hex-only (state.rs:1819-1826); `g` is not a hex digit, so the escape \
         cannot complete"
    );

    let gpu = gpu_mask(&p, &vocab);
    assert_eq!(
        gpu.allow[i], cpu.allow[i],
        "GPU must REJECT `\\u123g`. It accepts it because after 3 hex \
         transitions the GPU is in body(1) where `g` (class_content) is valid \
         (gpu_dispatch.rs:371). This pins the GPU bug as 3-hex-then-BODY, not \
         3-hex-then-strict-4th-hex."
    );
}

// =====================================================================
// Controls — both sides must AGREE here. If a control fails, the GPU is
// off by something other than exactly one hex digit.
// =====================================================================

/// `ሴ"` — the legal 4-hex form. CPU ACCEPTS; GPU also accepts. The
/// bug is OVER-acceptance of the 3-hex form, not UNDER-acceptance of the
/// legal 4-hex form, so this control must pass on both sides.
#[test]
fn control_1597_four_hex_then_quote_accepted_by_both() {
    let vocab = uescape_vocab();
    let p = string_in_body(&vocab);
    let cpu = p.compute_mask();
    let i = tok(&vocab, "\\u1234\"");
    assert_eq!(
        cpu.allow[i], 1,
        "CPU oracle ACCEPTS the legal `\\u1234\\\"` (4 hex digits then close)"
    );
    let gpu = gpu_mask(&p, &vocab);
    assert_eq!(
        gpu.allow[i], cpu.allow[i],
        "GPU must accept the legal `\\u1234\\\"` (control)"
    );
}

/// `\u12"`, `\u1"`, `\u"` — fewer than 3 hex digits then close. BOTH the
/// CPU oracle and the GPU reject these (the GPU's hex1/hex2 states require
/// a hex digit and `"` falls to reject). Confirms the GPU is short by
/// EXACTLY one digit (it accepts 3, not 2 or fewer), so the fix is "+1 hex
/// state", not "the whole hex walk is wrong".
#[test]
fn control_1597_fewer_than_three_hex_rejected_by_both() {
    let vocab = uescape_vocab();
    let p = string_in_body(&vocab);
    let cpu = p.compute_mask();
    let gpu = gpu_mask(&p, &vocab);
    for s in ["\\u12\"", "\\u1\"", "\\u\""] {
        let i = tok(&vocab, s);
        assert_eq!(
            cpu.allow[i], 0,
            "CPU oracle REJECTS {s:?} — fewer than the required 4 hex digits, \
             closing quote arrives mid-escape (state.rs:1813 needs a hex)"
        );
        assert_eq!(
            gpu.allow[i], cpu.allow[i],
            "GPU must REJECT {s:?} (control: GPU is short by exactly one hex \
             digit — it accepts 3, rejects fewer)"
        );
    }
}
