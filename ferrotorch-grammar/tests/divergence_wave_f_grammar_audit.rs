//! Wave-F regression coverage for ferrotorch-grammar feature gaps
//! (umbrella #1542):
//!
//! - #1486 (PARTIAL): `oneOf` / `anyOf` composition keywords compile
//!   into `Schema::OneOf` / `Schema::AnyOf` and the `JsonGrammar`
//!   state machine commits to a branch on the first emitted char.
//!   `allOf` remains `SchemaError::Unsupported("allOf")` — pinned by
//!   `audit_1486_allof_still_unsupported_pinned` in
//!   `divergence_jsonschema_features_audit.rs`.
//! - #1490 (SHIPPED): whitespace-permissive mode pops the number
//!   frame on whitespace once at least one digit has been emitted,
//!   so `{ "a" : 1 }` (with spaces around the integer) validates
//!   instead of rejecting at the trailing space.
//! - #1493 (PARTIAL SHIPPED): cross-boundary BPE tokens that lead
//!   with a parent terminator (`,"` is the canonical case) are
//!   recovered by `cross_boundary_post_pass` in `gpu_dispatch.rs`.
//!   That helper is `--features cuda`-gated; this test file covers
//!   the CPU side of the equivalence (the CPU path was already
//!   correct; the new test pins the CPU semantics so a future
//!   regression there would surface here even without a GPU
//!   available).
//!
//! Tests in this file are NOT tautological — every assertion either
//! exercises the state machine via `step_char` / `step_str` chains
//! built from named typed bits, or compares two grammar walks
//! against each other. None of them assert ferrotorch values
//! against a self-mirrored fixture.

use ferrotorch_grammar::{JsonGrammar, JsonSchemaProcessor, Schema};
use serde_json::json;

fn ascii_vocab() -> Vec<String> {
    (0x20u8..=0x7Eu8).map(|b| (b as char).to_string()).collect()
}

// ---------------------------------------------------------------------------
// #1486 oneOf / anyOf — state machine behaviour
// ---------------------------------------------------------------------------

/// `oneOf:[{type:"string"},{type:"number"}]` accepts `"foo"` (commits
/// to the string branch on the leading `"`).
#[test]
fn req_1486_oneof_string_or_number_accepts_string_value() {
    let schema =
        Schema::from_json_schema(&json!({"oneOf":[{"type":"string"},{"type":"number"}]})).unwrap();
    let mut g = JsonGrammar::new(schema);
    g.step_str("\"foo\"").unwrap();
    assert!(g.is_complete(), "string branch must reach a complete value");
}

/// Same schema accepts `42` (commits to the number branch on the
/// leading digit).
#[test]
fn req_1486_oneof_string_or_number_accepts_number_value() {
    let schema =
        Schema::from_json_schema(&json!({"oneOf":[{"type":"string"},{"type":"number"}]})).unwrap();
    let mut g = JsonGrammar::new(schema);
    g.step_char('4').unwrap();
    g.step_char('2').unwrap();
    // Number completes implicitly once at least one digit has been
    // emitted at the top level — `is_complete` doesn't flip without
    // a terminator at top level, but the next-chars must include
    // more digits and no other branches.
    let nx = g.valid_next_chars();
    assert!(nx.contains(&'0'));
    assert!(!nx.contains(&'"'), "string branch must not leak");
}

/// Negative path: a char accepted by NEITHER branch rejects.
#[test]
fn req_1486_oneof_rejects_first_char_outside_branch_starts() {
    let schema =
        Schema::from_json_schema(&json!({"oneOf":[{"type":"string"},{"type":"number"}]})).unwrap();
    let mut g = JsonGrammar::new(schema);
    // `t` is the boolean literal head — neither string nor number
    // accepts it.
    let err = g.step_char('t').unwrap_err();
    assert!(
        matches!(
            err,
            ferrotorch_grammar::state::StepError::UnexpectedChar { .. }
        ),
        "uncovered char must reject; got {err:?}"
    );
}

/// `anyOf` behaves the same shape as `oneOf` for prefix-grammar
/// acceptance.
#[test]
fn req_1486_anyof_boolean_or_null_accepts_both_branches() {
    let schema =
        Schema::from_json_schema(&json!({"anyOf":[{"type":"boolean"},{"type":"null"}]})).unwrap();

    let mut g_true = JsonGrammar::new(schema.clone());
    g_true.step_str("true").unwrap();
    assert!(g_true.is_complete());

    let mut g_null = JsonGrammar::new(schema);
    g_null.step_str("null").unwrap();
    assert!(g_null.is_complete());
}

/// `oneOf` inside `properties` lets a field hold either branch.
#[test]
fn req_1486_oneof_in_object_property() {
    let schema = Schema::from_json_schema(&json!({
        "type": "object",
        "properties": {
            "v": {"oneOf": [{"type": "string"}, {"type": "integer"}]}
        },
        "required": ["v"]
    }))
    .unwrap();

    let mut g_str = JsonGrammar::new(schema.clone());
    g_str.step_str("{\"v\":\"hi\"}").unwrap();
    assert!(g_str.is_complete());

    let mut g_int = JsonGrammar::new(schema);
    g_int.step_str("{\"v\":7}").unwrap();
    assert!(g_int.is_complete());
}

// ---------------------------------------------------------------------------
// #1490 whitespace-permissive — number-frame pop on whitespace
// ---------------------------------------------------------------------------

/// `{ "a" : 1 }` (every structural boundary AND the post-digit
/// boundary padded with a space) validates under
/// `whitespace_permissive`.
#[test]
fn req_1490_whitespace_after_integer_pops_number_frame() {
    let schema = Schema::from_json_schema(&json!({
        "type": "object",
        "properties": {"a": {"type": "integer"}},
        "required": ["a"]
    }))
    .unwrap();
    let mut g = JsonGrammar::new(schema).with_whitespace_permissive(true);
    g.step_str("{ \"a\" : 1 }").unwrap();
    assert!(g.is_complete());
}

/// Same shape for a number with decimal + exponent: `1.5e10` then a
/// space must pop the number frame.
#[test]
fn req_1490_whitespace_after_full_number_pops_frame() {
    let schema = Schema::from_json_schema(&json!({
        "type": "object",
        "properties": {"x": {"type": "number"}},
        "required": ["x"]
    }))
    .unwrap();
    let mut g = JsonGrammar::new(schema).with_whitespace_permissive(true);
    g.step_str("{ \"x\" : 1.5e10 }").unwrap();
    assert!(g.is_complete());
}

/// Whitespace MID-NUMBER (between digits) must still reject — the
/// pop is gated on `had_digits=true` AND not mid-decimal /
/// mid-exponent-without-digit.
#[test]
fn req_1490_whitespace_mid_decimal_still_rejects() {
    let mut g = JsonGrammar::new(Schema::Number).with_whitespace_permissive(true);
    g.step_str("1.").unwrap();
    // Mid-decimal: only digits valid, NOT whitespace.
    let err = g.step_char(' ').unwrap_err();
    assert!(matches!(
        err,
        ferrotorch_grammar::state::StepError::UnexpectedChar { .. }
    ));
}

/// Whitespace right after a bare `-` (no digit yet) must reject —
/// `had_digits` is false at that point.
#[test]
fn req_1490_whitespace_after_sign_only_still_rejects() {
    let mut g = JsonGrammar::new(Schema::Number).with_whitespace_permissive(true);
    g.step_char('-').unwrap();
    let err = g.step_char(' ').unwrap_err();
    assert!(matches!(
        err,
        ferrotorch_grammar::state::StepError::UnexpectedChar { .. }
    ));
}

/// Mid-exponent without exponent digit (`1e` then space) must reject.
#[test]
fn req_1490_whitespace_mid_exponent_still_rejects() {
    let mut g = JsonGrammar::new(Schema::Number).with_whitespace_permissive(true);
    g.step_str("1e").unwrap();
    let err = g.step_char(' ').unwrap_err();
    assert!(matches!(
        err,
        ferrotorch_grammar::state::StepError::UnexpectedChar { .. }
    ));
}

/// Default mode (whitespace-permissive OFF) still rejects whitespace
/// after a digit — regression guard for the strict mode.
#[test]
fn req_1490_strict_mode_still_rejects_post_digit_whitespace() {
    let mut g = JsonGrammar::new(Schema::Integer);
    g.step_char('1').unwrap();
    let err = g.step_char(' ').unwrap_err();
    assert!(matches!(
        err,
        ferrotorch_grammar::state::StepError::UnexpectedChar { .. }
    ));
}

// ---------------------------------------------------------------------------
// #1493 cross-boundary BPE — CPU semantics
// ---------------------------------------------------------------------------

/// The CPU path accepts the canonical `,"` cross-boundary BPE token
/// at `ObjectAfterValue`. This pinned behaviour is what the GPU-side
/// `cross_boundary_post_pass` recovers.
#[test]
fn req_1493_cpu_accepts_comma_quote_at_object_after_value() {
    let schema = Schema::from_json_schema(&json!({
        "type": "object",
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "integer"}
        },
        "required": ["a", "b"]
    }))
    .unwrap();
    let mut g = JsonGrammar::new(schema);
    g.step_str("{\"a\":1").unwrap();
    // The full `,"` cross-boundary token: terminator then open-key
    // quote. CPU semantics MUST accept this stream verbatim.
    g.step_str(",\"").unwrap();
    // The grammar is now in ObjectKey ready to walk `b`.
    g.step_str("b\":2}").unwrap();
    assert!(g.is_complete());
}

/// Same shape for an array: `,1` (comma followed by next-element
/// digit) is acceptable mid-array.
#[test]
fn req_1493_cpu_accepts_comma_digit_in_array() {
    let schema = Schema::from_json_schema(&json!({
        "type": "array",
        "items": {"type": "integer"}
    }))
    .unwrap();
    let mut g = JsonGrammar::new(schema);
    g.step_str("[1").unwrap();
    // `,1` is the cross-boundary `terminator + next-element-head`
    // token; CPU must accept.
    g.step_str(",1").unwrap();
    g.step_char(']').unwrap();
    assert!(g.is_complete());
}

/// CPU `compute_mask` for `JsonSchemaProcessor` accepts every
/// single-char terminator after an integer-in-object — sanity guard
/// that the post-pass we wired in gpu_dispatch.rs has parity with
/// the CPU behaviour we're matching.
#[test]
fn req_1493_cpu_mask_lights_terminators_after_integer_in_object() {
    let vocab = ascii_vocab();
    let mut processor = JsonSchemaProcessor::new(
        &json!({
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"}
            },
            "required": ["a", "b"]
        }),
        vocab.clone(),
    )
    .unwrap();
    for s in ["{", "\"", "a", "\"", ":", "1"] {
        let id = vocab.iter().position(|t| t == s).unwrap() as u32;
        processor.step_token(id).unwrap();
    }
    let mask = processor.compute_mask();
    let comma = vocab.iter().position(|s| s == ",").unwrap();
    assert_eq!(
        mask.allow[comma], 1,
        "CPU mask must light `,` at ObjectAfterValue with 'b' still unseen"
    );
}
