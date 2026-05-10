//! Conformance Phase 1 — bit-identical encode/decode against Python.
//!
//! Tracking issue: <https://github.com/<owner>/ferrotorch/issues/758>.
//!
//! Asserts that `ferrotorch_tokenize::{encode, encode_batch, decode,
//! vocab_size, token_to_id, id_to_token}` produce **bit-identical** output
//! to Python's `tokenizers.Tokenizer` for the same input on the same
//! `tokenizer.json` from a real Llama-3 release.
//!
//! Tokenization is integer-domain: there is no float tolerance, the
//! assertions are `assert_eq!`. If a single token id differs, the wrapper
//! lies about its conformance claim.
//!
//! Fixture provenance: `scripts/regenerate_tokenize_fixtures.py` runs the
//! Python `tokenizers.Tokenizer` against a curated corpus and writes
//! `tests/conformance/fixtures/llama3.json`. Re-run that script when the
//! Rust workspace bumps its `tokenizers` crate version, when the corpus
//! changes, or when the asset tokenizer file changes.

use std::path::PathBuf;

use ferrotorch_tokenize::{
    ChatMessage, apply_chat_template, apply_chat_template_to_ids, decode, encode, encode_batch,
    id_to_token, load_chat_template, load_tokenizer, token_to_id, vocab_size,
};
use serde::Deserialize;

/// Metadata block from `fixtures/llama3.json`. Fields prefixed with
/// `_` are deserialized only so the JSON shape is documented and so a
/// future test can read provenance — the dead-code allow is narrow.
#[derive(Debug, Deserialize)]
#[allow(
    dead_code,
    reason = "metadata fields document fixture provenance; consumed at debug print time only"
)]
struct FixtureMetadata {
    tokenizer_repo: String,
    tokenizer_path: String,
    tokenizers_version: String,
    #[serde(default)]
    transformers_version: String,
    #[serde(default)]
    generated_at: String,
    num_test_cases: usize,
    #[serde(default)]
    schema_version: u32,
}

#[derive(Debug, Deserialize)]
struct TestCase {
    input: String,
    encode_with_special: Vec<u32>,
    encode_no_special: Vec<u32>,
    decode_with_special_keep: String,
    decode_with_special_skip: String,
    decode_no_special: String,
}

#[derive(Debug, Deserialize)]
struct Fixture {
    metadata: FixtureMetadata,
    vocab_size_with_added_tokens: usize,
    vocab_size_no_added_tokens: usize,
    test_cases: Vec<TestCase>,
}

fn conformance_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("conformance")
}

fn load_fixture() -> Fixture {
    let p = conformance_dir().join("fixtures").join("llama3.json");
    let bytes = std::fs::read(&p).unwrap_or_else(|e| panic!("read fixture {}: {e}", p.display()));
    serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("parse fixture {}: {e}", p.display()))
}

fn load_asset_tokenizer(meta: &FixtureMetadata) -> ferrotorch_tokenize::Tokenizer {
    // The fixture stores the tokenizer.json path relative to the crate
    // root (`tests/conformance/assets/...`); resolve it back here.
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(&meta.tokenizer_path);
    load_tokenizer(&p).unwrap_or_else(|e| {
        panic!(
            "failed to load tokenizer asset {} (repo={}, tokenizers_version={}): {e:?}",
            p.display(),
            meta.tokenizer_repo,
            meta.tokenizers_version,
        )
    })
}

/// Compare the Rust workspace's `tokenizers` crate version against the
/// version that produced the fixture. A mismatch is *not* a hard failure
/// — minor-version drifts often produce identical output — but it's a
/// signal worth surfacing if a downstream test ever fails.
fn warn_on_tokenizers_version_drift(fixture_version: &str) {
    // The Rust workspace pins `tokenizers = "0.22"`. Ground-truth source
    // is the workspace Cargo.toml; we record this here as a literal so a
    // dep-bump that forgets to refresh fixtures yells at us.
    const RUST_TOKENIZERS_MAJOR_MINOR: &str = "0.22";
    if !fixture_version.starts_with(RUST_TOKENIZERS_MAJOR_MINOR) {
        eprintln!(
            "warning: fixture was generated with tokenizers={fixture_version}, but Rust \
             workspace uses tokenizers={RUST_TOKENIZERS_MAJOR_MINOR}.x. Re-run \
             scripts/regenerate_tokenize_fixtures.py if conformance tests fail."
        );
    }
}

#[test]
fn vocab_size_matches_python_reference() {
    let fixture = load_fixture();
    warn_on_tokenizers_version_drift(&fixture.metadata.tokenizers_version);
    let tok = load_asset_tokenizer(&fixture.metadata);

    let with_added = vocab_size(&tok, true);
    let no_added = vocab_size(&tok, false);

    assert_eq!(
        with_added, fixture.vocab_size_with_added_tokens,
        "vocab_size(true) drifted: rust={with_added}, python={}",
        fixture.vocab_size_with_added_tokens
    );
    assert_eq!(
        no_added, fixture.vocab_size_no_added_tokens,
        "vocab_size(false) drifted: rust={no_added}, python={}",
        fixture.vocab_size_no_added_tokens
    );
}

#[test]
fn encode_matches_python_reference_per_case() {
    let fixture = load_fixture();
    warn_on_tokenizers_version_drift(&fixture.metadata.tokenizers_version);
    let tok = load_asset_tokenizer(&fixture.metadata);

    assert_eq!(
        fixture.test_cases.len(),
        fixture.metadata.num_test_cases,
        "fixture metadata num_test_cases inconsistent with test_cases.len()"
    );

    for (i, case) in fixture.test_cases.iter().enumerate() {
        let prefix: String = case.input.chars().take(40).collect();
        let with_special = encode(&tok, &case.input, true).unwrap_or_else(|e| {
            panic!("case[{i}] encode(true) errored on input prefix={prefix:?}: {e:?}")
        });
        assert_eq!(
            with_special,
            case.encode_with_special,
            "case[{i}] encode(add_special_tokens=true) mismatch \
             on input prefix={prefix:?} (len={})",
            case.input.len()
        );

        let no_special = encode(&tok, &case.input, false).unwrap_or_else(|e| {
            panic!("case[{i}] encode(false) errored on input prefix={prefix:?}: {e:?}")
        });
        assert_eq!(
            no_special, case.encode_no_special,
            "case[{i}] encode(add_special_tokens=false) mismatch \
             on input prefix={prefix:?}",
        );
    }
}

#[test]
fn decode_matches_python_reference_per_case() {
    let fixture = load_fixture();
    warn_on_tokenizers_version_drift(&fixture.metadata.tokenizers_version);
    let tok = load_asset_tokenizer(&fixture.metadata);

    for (i, case) in fixture.test_cases.iter().enumerate() {
        let prefix: String = case.input.chars().take(40).collect();
        let dec_keep = decode(&tok, &case.encode_with_special, false).unwrap_or_else(|e| {
            panic!("case[{i}] decode(skip=false) errored prefix={prefix:?}: {e:?}")
        });
        assert_eq!(
            dec_keep, case.decode_with_special_keep,
            "case[{i}] decode(skip_special_tokens=false) mismatch \
             on input prefix={prefix:?}",
        );

        let dec_skip = decode(&tok, &case.encode_with_special, true).unwrap_or_else(|e| {
            panic!("case[{i}] decode(skip=true) errored prefix={prefix:?}: {e:?}")
        });
        assert_eq!(
            dec_skip, case.decode_with_special_skip,
            "case[{i}] decode(skip_special_tokens=true) mismatch \
             on input prefix={prefix:?}",
        );

        let dec_no = decode(&tok, &case.encode_no_special, false).unwrap_or_else(|e| {
            panic!("case[{i}] decode(no-special, skip=false) errored prefix={prefix:?}: {e:?}")
        });
        assert_eq!(
            dec_no, case.decode_no_special,
            "case[{i}] decode(no-special) mismatch on input prefix={prefix:?}",
        );
    }
}

#[test]
fn encode_batch_matches_per_input_encode() {
    // `encode_batch` is exercised against its single-input equivalent
    // (`encode(add_special_tokens=false)`) for every fixture case — i.e.
    // batched and unbatched paths must agree, and both must agree with
    // the Python reference.
    let fixture = load_fixture();
    warn_on_tokenizers_version_drift(&fixture.metadata.tokenizers_version);
    let tok = load_asset_tokenizer(&fixture.metadata);

    let inputs: Vec<&str> = fixture
        .test_cases
        .iter()
        .map(|c| c.input.as_str())
        .collect();
    let batched = encode_batch(&tok, &inputs, false).expect("encode_batch failed");

    assert_eq!(
        batched.len(),
        fixture.test_cases.len(),
        "encode_batch returned wrong number of results"
    );
    for (i, case) in fixture.test_cases.iter().enumerate() {
        assert_eq!(
            batched[i],
            case.encode_no_special,
            "case[{i}] encode_batch result diverged from per-input encode \
             on input prefix={:?}",
            case.input.chars().take(40).collect::<String>()
        );
    }
}

#[test]
fn token_to_id_resolves_known_special_tokens() {
    let fixture = load_fixture();
    let tok = load_asset_tokenizer(&fixture.metadata);

    // Llama 3 vocabulary: `<|begin_of_text|>` is at id 128000 in every
    // released variant of the tokenizer, regardless of whether it's the
    // 8B/70B/3.2-1B distribution. This is the most stable cross-check
    // available without re-deriving Python's vocabulary in Rust.
    let bos_id = token_to_id(&tok, "<|begin_of_text|>");
    assert_eq!(
        bos_id,
        Some(128_000),
        "Llama 3 BOS id drifted; fixture from {}",
        fixture.metadata.tokenizer_repo
    );

    // Round-trip: id_to_token(token_to_id(x)) == Some(x) for special tokens.
    let bos_back = id_to_token(&tok, 128_000);
    assert_eq!(
        bos_back.as_deref(),
        Some("<|begin_of_text|>"),
        "id_to_token(128000) round-trip failed"
    );

    // Unknown token: must return None, not panic.
    let unknown = token_to_id(&tok, "ZZZZ_definitely_not_in_llama_vocab_ZZZZ");
    assert!(
        unknown.is_none(),
        "token_to_id returned Some({unknown:?}) for an obviously-absent string"
    );

    // id_to_token on an out-of-range id: must return None, not panic.
    let way_out = id_to_token(&tok, u32::MAX);
    assert!(
        way_out.is_none(),
        "id_to_token(u32::MAX) returned Some({way_out:?})"
    );
}

#[test]
fn chat_template_round_trip_matches_minijinja() {
    // `apply_chat_template` and `apply_chat_template_to_ids` are
    // covered by the in-source tests already (see `lib.rs::tests`). The
    // conformance test here confirms the tokenize half of
    // `apply_chat_template_to_ids`: rendering + tokenizing in one call
    // must produce exactly the same ids as the two-step (render →
    // encode) path on the same inputs.
    let fixture = load_fixture();
    let tok = load_asset_tokenizer(&fixture.metadata);

    // A Llama-3-style template, spelled inline so the test does not
    // depend on a tokenizer_config.json being present alongside the
    // tokenizer.json asset.
    const SIMPLE_TPL: &str = "{% for m in messages %}\
<|start_header_id|>{{ m.role }}<|end_header_id|>\n\n{{ m.content }}<|eot_id|>\
{% endfor %}";

    let messages = vec![
        ChatMessage::new("system", "You are helpful."),
        ChatMessage::new("user", "Hi."),
    ];

    let prompt = apply_chat_template(SIMPLE_TPL, &messages, false, None, None).expect("render");
    let direct_ids = encode(&tok, &prompt, false).expect("encode");

    // Hand-computed expected output of `SIMPLE_TPL` against `messages`,
    // computed by walking the minijinja template manually (the `\n`
    // characters below are literal newlines, not escape sequences).
    // This is the external reference the test was missing — without it,
    // the prior assertions only proved `apply_chat_template` agrees with
    // itself. Now we anchor the renderer to a fixed string.
    const EXPECTED_RENDERED: &str = "<|start_header_id|>system<|end_header_id|>\n\nYou are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHi.<|eot_id|>";
    assert_eq!(
        prompt, EXPECTED_RENDERED,
        "minijinja render must match the hand-computed expected string"
    );

    let (combined_prompt, combined_ids) =
        apply_chat_template_to_ids(&tok, SIMPLE_TPL, &messages, false, None, None, false)
            .expect("apply_chat_template_to_ids");

    assert_eq!(combined_prompt, prompt, "render-only prompt mismatch");
    assert_eq!(
        combined_ids, direct_ids,
        "apply_chat_template_to_ids ids diverged from manual render+encode"
    );
}

#[test]
fn load_chat_template_round_trips_through_disk() {
    // `load_chat_template` reads `tokenizer_config.json`. We don't ship
    // a tokenizer_config.json asset (the Llama 3 chat template is a
    // separate concern from the conformance corpus), so this test
    // synthesizes one in `temp_dir` and verifies the loader reads back
    // exactly what we wrote — i.e. proves the I/O contract without
    // depending on a specific upstream template.
    let dir = std::env::temp_dir();
    let path = dir.join("ferrotorch_tokenize_conformance_chat_template.json");
    let template = "{% for m in messages %}{{ m.content }}{% endfor %}";
    let body = serde_json::json!({ "chat_template": template });
    std::fs::write(&path, serde_json::to_vec_pretty(&body).unwrap())
        .expect("write tokenizer_config.json fixture");

    let loaded = load_chat_template(&path)
        .expect("load_chat_template error")
        .expect("chat_template field absent");
    assert_eq!(
        loaded, template,
        "load_chat_template returned a drifted body"
    );

    let _ = std::fs::remove_file(&path);
}
