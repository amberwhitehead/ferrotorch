//! Integration tests for GPTQ / AWQ / HQQ quantized checkpoint loading.
//!
//! ## Purpose
//!
//! These tests verify end-to-end parity between ferrotorch-llama's quantized
//! weight loaders and the HuggingFace `transformers` + `auto_gptq` / `autoawq` /
//! `hqq` reference pipeline for the **same prompt and checkpoint**.
//!
//! Each test:
//! 1. Downloads a small quantized checkpoint from the HF Hub via
//!    [`ferrotorch_hub::hf_download_model`].
//! 2. Loads the packed weight tiles through the corresponding
//!    ferrotorch-llama dequantizer (`dequantize_gptq_q4`, `dequantize_awq_q4`,
//!    `dequantize_hqq_q4_axis1`). For HQQ the test drives the production
//!    `LlamaForCausalLM::load_hqq_state_dict` path directly (#1172).
//! 3. Injects the dequantized weights into a [`LlamaForCausalLM`] and runs
//!    `forward_from_ids` on a fixed token sequence.
//! 4. Compares the last-token logits against the expected values captured by
//!    `scripts/regenerate_quant_loader_fixtures.py` within
//!    `F32_TRANSCENDENTAL = 1e-5` absolute tolerance.
//!
//! ## Running
//!
//! Tests are [`#[ignore]`]-gated so they are **skipped in CI** by default:
//!
//! ```bash
//! # Run all three quant-loader smoke tests (requires network + HF Hub access):
//! cargo test -p ferrotorch-llama -- --ignored quant_loader_smoke
//!
//! # Or individually:
//! cargo test -p ferrotorch-llama -- --ignored gptq_smoke
//! cargo test -p ferrotorch-llama -- --ignored awq_smoke
//! cargo test -p ferrotorch-llama -- --ignored hqq_smoke
//! ```
//!
//! ## Regenerating fixtures
//!
//! When the model checkpoints change or the dequantization logic is updated,
//! regenerate the expected outputs:
//!
//! ```bash
//! pip install --user transformers==4.50.3 auto-gptq==0.7.1 autoawq==0.2.6 hqq==0.2.1 torch
//! python3 scripts/regenerate_quant_loader_fixtures.py
//! ```
//!
//! The script writes `ferrotorch-llama/tests/fixtures/quant_loader_expected.json`.
//!
//! ## No-network note
//!
//! If the HF Hub is unreachable or the checkpoint is not cached,
//! `hf_download_model` returns an error and the test fails immediately with
//! a message explaining which checkpoint to pre-populate in the hub cache.

use std::collections::HashMap;
use std::path::Path;

use ferrotorch_core::{Tensor, TensorStorage};
use ferrotorch_hub::{HubCache, hf_download_model};
use ferrotorch_llama::{
    AwqQ4, GptqQ4, HqqQ4Axis1, LlamaActivation, LlamaConfig, LlamaForCausalLM, dequantize_awq_q4,
    dequantize_gptq_q4, dequantize_hqq_q4_axis1, hqq_state_dict_to_dense,
};
use ferrotorch_nn::module::StateDict;
use ferrotorch_serialize::load_safetensors_auto;
use serde_json::Value;

// ---------------------------------------------------------------------------
// Tolerance
// ---------------------------------------------------------------------------

/// Absolute tolerance for comparing ferrotorch logits against the Python
/// reference.  1e-5 = F32_TRANSCENDENTAL as specified in the sprint brief.
/// Dequantized weights introduce rounding vs. the reference pipeline's
/// native GPTQ/AWQ/HQQ CUDA kernels; this tolerance accounts for the
/// CPU-f32 accumulation path used here.
const F32_TRANSCENDENTAL: f32 = 1e-5;

// ---------------------------------------------------------------------------
// HF checkpoint coordinates
// ---------------------------------------------------------------------------

/// A small GPTQ-quantized TinyLlama checkpoint (~200 MB).
/// TheBloke/TinyLlama-1.1B-Chat-v0.3-GPTQ is the canonical small GPTQ
/// reference on the Hub.
const GPTQ_REPO: &str = "TheBloke/TinyLlama-1.1B-Chat-v0.3-GPTQ";
const GPTQ_REVISION: &str = "main";

/// A small AWQ-quantized TinyLlama checkpoint.
/// TheBloke/TinyLlama-1.1B-Chat-v0.3-AWQ is the canonical small AWQ
/// reference on the Hub.
const AWQ_REPO: &str = "TheBloke/TinyLlama-1.1B-Chat-v0.3-AWQ";
const AWQ_REVISION: &str = "main";

/// A small HQQ-quantized TinyLlama checkpoint.
/// mobiuslabsgmbh/TinyLlama-1.1B-Chat-v1.0-HQQ is the canonical small HQQ
/// reference on the Hub.
const HQQ_REPO: &str = "mobiuslabsgmbh/TinyLlama-1.1B-Chat-v1.0-HQQ";
const HQQ_REVISION: &str = "main";

// ---------------------------------------------------------------------------
// Fixed prompt token ids
// ---------------------------------------------------------------------------

/// Token-id sequence for the fixed prompt "Hello, world!" tokenized with
/// the TinyLlama tokenizer (BOS=1, space-encoded with SentencePiece).
/// These ids must match what `regenerate_quant_loader_fixtures.py` uses.
const PROMPT_IDS: &[u32] = &[1, 15043, 29892, 3186, 29991];

// ---------------------------------------------------------------------------
// Fixture helpers
// ---------------------------------------------------------------------------

/// Load `ferrotorch-llama/tests/fixtures/quant_loader_expected.json`.
fn load_expected_fixture() -> Value {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let path =
        std::path::PathBuf::from(manifest_dir).join("tests/fixtures/quant_loader_expected.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "Cannot read quant_loader expected fixture at {}: {e}\n\
             Run `python3 scripts/regenerate_quant_loader_fixtures.py` to generate it.",
            path.display()
        )
    });
    serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("Fixture JSON parse error at {}: {e}", path.display()))
}

/// Parse a flat JSON `f32` array.
fn parse_f32_vec(v: &Value) -> Vec<f32> {
    v.as_array()
        .expect("expected JSON array of f32")
        .iter()
        .map(|x| x.as_f64().expect("expected numeric element") as f32)
        .collect()
}

/// Assert element-wise absolute error within `tol`.
fn assert_allclose(actual: &[f32], expected: &[f32], tol: f32, context: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{context}: length mismatch (got {}, expected {})",
        actual.len(),
        expected.len()
    );
    let mut worst_abs = 0.0_f32;
    let mut worst_idx = 0;
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        if diff > worst_abs {
            worst_abs = diff;
            worst_idx = i;
        }
    }
    assert!(
        worst_abs <= tol,
        "{context}: max absolute error {worst_abs:.3e} > tol {tol:.1e} \
         at index {worst_idx} (got={:.6}, expected={:.6})",
        actual[worst_idx],
        expected[worst_idx],
    );
}

// ---------------------------------------------------------------------------
// Shared hub cache
// ---------------------------------------------------------------------------

/// Return a [`HubCache`] backed by a temporary directory.
/// Each test uses its own tempdir so parallel test runs don't collide.
fn temp_hub_cache() -> (tempfile::TempDir, HubCache) {
    let dir = tempfile::tempdir().expect("failed to create tempdir for hub cache");
    let cache = HubCache::new(dir.path());
    (dir, cache)
}

// ---------------------------------------------------------------------------
// GPTQ smoke test
// ---------------------------------------------------------------------------

/// Download `TheBloke/TinyLlama-1.1B-Chat-v0.3-GPTQ`, dequantize via
/// [`dequantize_gptq_q4`], run a forward pass, and compare logits against
/// the Python reference fixture within `F32_TRANSCENDENTAL = 1e-5`.
///
/// # Running
///
/// ```bash
/// cargo test -p ferrotorch-llama -- --ignored gptq_smoke
/// ```
#[test]
#[ignore = "requires network access to HF Hub; gated for offline CI (Sprint C.9 #633)"]
fn gptq_smoke() {
    let (_tmpdir, cache) = temp_hub_cache();

    // 1. Download the checkpoint.
    let model_dir = hf_download_model(GPTQ_REPO, GPTQ_REVISION, &cache).unwrap_or_else(|e| {
        panic!(
            "gptq_smoke: failed to download {GPTQ_REPO}@{GPTQ_REVISION}: {e}\n\
             Ensure network access and a valid HF token (HF_TOKEN env var) if the \
             checkpoint is gated."
        )
    });

    // 2. Load the config.json and build a matching LlamaConfig.
    let cfg = load_llama_config_from_dir(&model_dir);

    // 3. Load + dequantize packed GPTQ weights.
    //    GPTQ checkpoints store the quantized tensors in the safetensors shard.
    //    We load them as-is (f32 reinterpretation of int32 bytes is handled
    //    by the load path; GPTQ int32 data arrives as `i32`-backed f32
    //    buffers — see note in quant_loaders.rs).
    let raw_sd = load_safetensors_auto::<f32>(&model_dir)
        .unwrap_or_else(|e| panic!("gptq_smoke: safetensors load failed: {e}"));

    let dequant_sd = dequantize_gptq_state_dict(&raw_sd, &cfg);

    // 4. Build model and run forward pass.
    let logits = run_forward(cfg, dequant_sd);

    // 5. Compare against reference fixture.
    let fixture = load_expected_fixture();
    let expected = parse_f32_vec(&fixture["gptq"]["last_token_logits"]);
    let vocab = logits.shape()[2];
    let logits_data = logits.data_vec().expect("gptq_smoke: data_vec failed");
    let seq_len = PROMPT_IDS.len();
    let last_logits = &logits_data[(seq_len - 1) * vocab..seq_len * vocab];

    assert_allclose(
        last_logits,
        &expected,
        F32_TRANSCENDENTAL,
        "gptq_smoke/last_logits",
    );
}

// ---------------------------------------------------------------------------
// AWQ smoke test
// ---------------------------------------------------------------------------

/// Download `TheBloke/TinyLlama-1.1B-Chat-v0.3-AWQ`, dequantize via
/// [`dequantize_awq_q4`], run a forward pass, and compare logits against
/// the Python reference fixture within `F32_TRANSCENDENTAL = 1e-5`.
///
/// # Running
///
/// ```bash
/// cargo test -p ferrotorch-llama -- --ignored awq_smoke
/// ```
#[test]
#[ignore = "requires network access to HF Hub; gated for offline CI (Sprint C.9 #633)"]
fn awq_smoke() {
    let (_tmpdir, cache) = temp_hub_cache();

    // 1. Download the checkpoint.
    let model_dir = hf_download_model(AWQ_REPO, AWQ_REVISION, &cache).unwrap_or_else(|e| {
        panic!(
            "awq_smoke: failed to download {AWQ_REPO}@{AWQ_REVISION}: {e}\n\
             Ensure network access and a valid HF token (HF_TOKEN env var) if the \
             checkpoint is gated."
        )
    });

    // 2. Build config.
    let cfg = load_llama_config_from_dir(&model_dir);

    // 3. Load + dequantize packed AWQ weights.
    let raw_sd = load_safetensors_auto::<f32>(&model_dir)
        .unwrap_or_else(|e| panic!("awq_smoke: safetensors load failed: {e}"));

    let dequant_sd = dequantize_awq_state_dict(&raw_sd, &cfg);

    // 4. Build model and run forward pass.
    let logits = run_forward(cfg, dequant_sd);

    // 5. Compare against reference fixture.
    let fixture = load_expected_fixture();
    let expected = parse_f32_vec(&fixture["awq"]["last_token_logits"]);
    let vocab = logits.shape()[2];
    let logits_data = logits.data_vec().expect("awq_smoke: data_vec failed");
    let seq_len = PROMPT_IDS.len();
    let last_logits = &logits_data[(seq_len - 1) * vocab..seq_len * vocab];

    assert_allclose(
        last_logits,
        &expected,
        F32_TRANSCENDENTAL,
        "awq_smoke/last_logits",
    );
}

// ---------------------------------------------------------------------------
// HQQ smoke test
// ---------------------------------------------------------------------------

/// Download `mobiuslabsgmbh/TinyLlama-1.1B-Chat-v1.0-HQQ`, dequantize via
/// [`dequantize_hqq`], run a forward pass, and compare logits against the
/// Python reference fixture within `F32_TRANSCENDENTAL = 1e-5`.
///
/// # Running
///
/// ```bash
/// cargo test -p ferrotorch-llama -- --ignored hqq_smoke
/// ```
#[test]
#[ignore = "requires network access to HF Hub; gated for offline CI (Sprint C.9 #633)"]
fn hqq_smoke() {
    let (_tmpdir, cache) = temp_hub_cache();

    // 1. Download the checkpoint.
    let model_dir = hf_download_model(HQQ_REPO, HQQ_REVISION, &cache).unwrap_or_else(|e| {
        panic!(
            "hqq_smoke: failed to download {HQQ_REPO}@{HQQ_REVISION}: {e}\n\
             Ensure network access and a valid HF token (HF_TOKEN env var) if the \
             checkpoint is gated."
        )
    });

    // 2. Build config.
    let cfg = load_llama_config_from_dir(&model_dir);

    // 3. Load packed HQQ weights and run them through the *production*
    //    consumer: LlamaForCausalLM::load_hqq_state_dict (#1172). This
    //    exercises the same path a real application would, rather than a
    //    test-local dequantizer.
    let raw_sd = load_safetensors_auto::<f32>(&model_dir)
        .unwrap_or_else(|e| panic!("hqq_smoke: safetensors load failed: {e}"));

    // 4. Build model and run forward pass via the production HQQ loader.
    let logits = run_forward_hqq(cfg, raw_sd);

    // 5. Compare against reference fixture.
    let fixture = load_expected_fixture();
    let expected = parse_f32_vec(&fixture["hqq"]["last_token_logits"]);
    let vocab = logits.shape()[2];
    let logits_data = logits.data_vec().expect("hqq_smoke: data_vec failed");
    let seq_len = PROMPT_IDS.len();
    let last_logits = &logits_data[(seq_len - 1) * vocab..seq_len * vocab];

    assert_allclose(
        last_logits,
        &expected,
        F32_TRANSCENDENTAL,
        "hqq_smoke/last_logits",
    );
}

// ---------------------------------------------------------------------------
// Shared helpers — config loading
// ---------------------------------------------------------------------------

/// Parse `config.json` from the downloaded model directory into a
/// [`LlamaConfig`].  Panics if the file is absent or the required fields
/// are missing.
fn load_llama_config_from_dir(model_dir: &Path) -> LlamaConfig {
    let config_path = model_dir.join("config.json");
    let text = std::fs::read_to_string(&config_path).unwrap_or_else(|e| {
        panic!(
            "load_llama_config_from_dir: cannot read config.json at {}: {e}",
            config_path.display()
        )
    });
    let v: Value = serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("load_llama_config_from_dir: JSON parse error: {e}"));

    let get_usize = |key: &str| -> usize {
        v[key]
            .as_u64()
            .unwrap_or_else(|| panic!("load_llama_config_from_dir: missing/invalid field {key}"))
            as usize
    };

    let num_attention_heads = get_usize("num_attention_heads");
    let num_key_value_heads = v["num_key_value_heads"]
        .as_u64()
        .map(|n| n as usize)
        .unwrap_or(num_attention_heads);

    LlamaConfig {
        vocab_size: get_usize("vocab_size"),
        hidden_size: get_usize("hidden_size"),
        intermediate_size: get_usize("intermediate_size"),
        num_hidden_layers: get_usize("num_hidden_layers"),
        num_attention_heads,
        num_key_value_heads,
        rms_norm_eps: v["rms_norm_eps"].as_f64().unwrap_or(1e-5),
        rope_theta: v["rope_theta"].as_f64().unwrap_or(10_000.0),
        max_position_embeddings: v["max_position_embeddings"]
            .as_u64()
            .map(|n| n as usize)
            .unwrap_or(2048),
        tie_word_embeddings: v["tie_word_embeddings"].as_bool().unwrap_or(false),
        hidden_act: LlamaActivation::Silu,
    }
}

// ---------------------------------------------------------------------------
// Shared helpers — model forward
// ---------------------------------------------------------------------------

/// Build a [`LlamaForCausalLM`], inject `state_dict`, run `forward_from_ids`
/// on [`PROMPT_IDS`], and return the logit tensor `[1, seq_len, vocab_size]`.
fn run_forward(cfg: LlamaConfig, state_dict: StateDict<f32>) -> Tensor<f32> {
    let mut model =
        LlamaForCausalLM::<f32>::new(cfg).expect("run_forward: LlamaForCausalLM::new failed");
    model
        .load_hf_state_dict(&state_dict, false)
        .expect("run_forward: load_hf_state_dict failed");
    model
        .forward_from_ids(PROMPT_IDS)
        .expect("run_forward: forward_from_ids failed")
}

/// Build a [`LlamaForCausalLM`] and load a *raw HQQ-format* state dict
/// through the production [`LlamaForCausalLM::load_hqq_state_dict`] path,
/// then run `forward_from_ids` on [`PROMPT_IDS`]. (#1172)
fn run_forward_hqq(cfg: LlamaConfig, raw_hqq_sd: StateDict<f32>) -> Tensor<f32> {
    let mut model =
        LlamaForCausalLM::<f32>::new(cfg).expect("run_forward_hqq: LlamaForCausalLM::new failed");
    model
        .load_hqq_state_dict(&raw_hqq_sd, false)
        .expect("run_forward_hqq: load_hqq_state_dict failed");
    model
        .forward_from_ids(PROMPT_IDS)
        .expect("run_forward_hqq: forward_from_ids failed")
}

// ---------------------------------------------------------------------------
// GPTQ state-dict dequantizer
// ---------------------------------------------------------------------------

/// Walk every `*.qweight` key in `raw_sd`, reconstruct the packed [`GptqQ4`]
/// tile from the parallel `*.qzeros`, `*.scales`, and optional `*.g_idx`
/// tensors, dequantize to f32, and emit standard `*.weight` keys in a fresh
/// state dict.  Non-quantized tensors (norms, embeddings, lm_head) are
/// passed through unchanged.
#[allow(clippy::manual_checked_ops)]
fn dequantize_gptq_state_dict(raw_sd: &StateDict<f32>, cfg: &LlamaConfig) -> StateDict<f32> {
    let mut out: StateDict<f32> = HashMap::new();

    // Collect unique weight prefixes that have a `qweight` tensor.
    let prefixes: std::collections::BTreeSet<String> = raw_sd
        .keys()
        .filter_map(|k| k.strip_suffix(".qweight").map(String::from))
        .collect();

    for prefix in &prefixes {
        let qweight_key = format!("{prefix}.qweight");
        let qzeros_key = format!("{prefix}.qzeros");
        let scales_key = format!("{prefix}.scales");
        let g_idx_key = format!("{prefix}.g_idx");

        let qweight_t = raw_sd
            .get(&qweight_key)
            .unwrap_or_else(|| panic!("dequantize_gptq: missing {qweight_key}"));
        let qzeros_t = raw_sd
            .get(&qzeros_key)
            .unwrap_or_else(|| panic!("dequantize_gptq: missing {qzeros_key}"));
        let scales_t = raw_sd
            .get(&scales_key)
            .unwrap_or_else(|| panic!("dequantize_gptq: missing {scales_key}"));

        // qweight shape: [K/8, N] — reinterpret the f32 bytes back to i32.
        let qweight_shape = qweight_t.shape().to_vec();
        let qzeros_shape = qzeros_t.shape().to_vec();

        let qweight_raw = qweight_t
            .data_vec()
            .expect("dequantize_gptq: qweight data_vec");
        let qzeros_raw = qzeros_t
            .data_vec()
            .expect("dequantize_gptq: qzeros data_vec");
        let scales_raw = scales_t
            .data_vec()
            .expect("dequantize_gptq: scales data_vec");

        // Reinterpret f32 storage as i32 (packed int4 tiles are stored as
        // raw bytes in safetensors; the f32 loader preserves the bit pattern).
        let qweight_i32: Vec<i32> = qweight_raw
            .iter()
            .map(|&f| f32::to_bits(f) as i32)
            .collect();
        let qzeros_i32: Vec<i32> = qzeros_raw.iter().map(|&f| f32::to_bits(f) as i32).collect();

        let g_idx: Option<Vec<i32>> = raw_sd.get(&g_idx_key).map(|t| {
            t.data_vec()
                .expect("dequantize_gptq: g_idx data_vec")
                .iter()
                .map(|&f| f32::to_bits(f) as i32)
                .collect()
        });

        // qweight shape [K/8, N]: in_features = (K/8)*8, out_features = N.
        let k_over_8 = qweight_shape[0];
        let n = qweight_shape[1];
        let in_features = k_over_8 * 8;
        let out_features = n;
        // group_size = in_features / num_groups; num_groups = qzeros_shape[0].
        let num_groups = qzeros_shape[0];
        let group_size = if num_groups > 0 {
            in_features / num_groups
        } else {
            in_features
        };

        let packed = GptqQ4::new(
            qweight_i32,
            qzeros_i32,
            scales_raw,
            g_idx,
            in_features,
            out_features,
            group_size,
        );

        let dequant = dequantize_gptq_q4(&packed)
            .unwrap_or_else(|e| panic!("dequantize_gptq: {prefix} dequant failed: {e}"));

        // Reconstruct as [out_features, in_features] tensor.
        let weight_tensor = make_tensor_f32(dequant, &[out_features, in_features]);
        out.insert(format!("{prefix}.weight"), weight_tensor);
    }

    // Pass through all non-quantized tensors.
    for (k, v) in raw_sd {
        if !k.ends_with(".qweight")
            && !k.ends_with(".qzeros")
            && !k.ends_with(".scales")
            && !k.ends_with(".g_idx")
        {
            out.insert(k.clone(), v.clone());
        }
    }

    let _ = cfg; // cfg available for future use
    out
}

// ---------------------------------------------------------------------------
// AWQ state-dict dequantizer
// ---------------------------------------------------------------------------

/// Walk every `*.qweight` key, reconstruct the packed [`AwqQ4`] tile from
/// the parallel `*.qzeros` and `*.scales` tensors, dequantize to f32, and
/// emit standard `*.weight` keys.  Non-quantized tensors pass through.
#[allow(clippy::manual_checked_ops)]
fn dequantize_awq_state_dict(raw_sd: &StateDict<f32>, cfg: &LlamaConfig) -> StateDict<f32> {
    let mut out: StateDict<f32> = HashMap::new();

    let prefixes: std::collections::BTreeSet<String> = raw_sd
        .keys()
        .filter_map(|k| k.strip_suffix(".qweight").map(String::from))
        .collect();

    for prefix in &prefixes {
        let qweight_key = format!("{prefix}.qweight");
        let qzeros_key = format!("{prefix}.qzeros");
        let scales_key = format!("{prefix}.scales");

        let qweight_t = raw_sd
            .get(&qweight_key)
            .unwrap_or_else(|| panic!("dequantize_awq: missing {qweight_key}"));
        let qzeros_t = raw_sd
            .get(&qzeros_key)
            .unwrap_or_else(|| panic!("dequantize_awq: missing {qzeros_key}"));
        let scales_t = raw_sd
            .get(&scales_key)
            .unwrap_or_else(|| panic!("dequantize_awq: missing {scales_key}"));

        // AWQ qweight shape: [in_features, out_features / 8].
        let qweight_shape = qweight_t.shape().to_vec();
        let qzeros_shape = qzeros_t.shape().to_vec();

        let qweight_raw = qweight_t
            .data_vec()
            .expect("dequantize_awq: qweight data_vec");
        let qzeros_raw = qzeros_t
            .data_vec()
            .expect("dequantize_awq: qzeros data_vec");
        let scales_raw = scales_t
            .data_vec()
            .expect("dequantize_awq: scales data_vec");

        let qweight_i32: Vec<i32> = qweight_raw
            .iter()
            .map(|&f| f32::to_bits(f) as i32)
            .collect();
        let qzeros_i32: Vec<i32> = qzeros_raw.iter().map(|&f| f32::to_bits(f) as i32).collect();

        // AWQ qweight: [K, N/8] → in_features=K, out_features=N/8*8.
        let in_features = qweight_shape[0];
        let n_packed = qweight_shape[1];
        let out_features = n_packed * 8;
        let num_groups = qzeros_shape[0];
        let group_size = if num_groups > 0 {
            in_features / num_groups
        } else {
            in_features
        };

        let packed = AwqQ4::new(
            qweight_i32,
            qzeros_i32,
            scales_raw,
            in_features,
            out_features,
            group_size,
        );

        let dequant = dequantize_awq_q4(&packed)
            .unwrap_or_else(|e| panic!("dequantize_awq: {prefix} dequant failed: {e}"));

        let weight_tensor = make_tensor_f32(dequant, &[out_features, in_features]);
        out.insert(format!("{prefix}.weight"), weight_tensor);
    }

    // Pass through non-quantized tensors.
    for (k, v) in raw_sd {
        if !k.ends_with(".qweight") && !k.ends_with(".qzeros") && !k.ends_with(".scales") {
            out.insert(k.clone(), v.clone());
        }
    }

    let _ = cfg;
    out
}

// ---------------------------------------------------------------------------
// HQQ axis=1 grouped Q4 — offline golden-oracle integration test (#1172)
//
// Network-free: builds a raw HQQ-format state dict from oracle bytes
// produced by the HQQ reference (mobiusml/hqq v0.2.1) and runs it through
// the production `hqq_state_dict_to_dense` + `load_hqq_state_dict` path.
// R-CHAR-3: the dense weight is checked against the reference dequant,
// not against ferrotorch's own output.
// ---------------------------------------------------------------------------

/// Oracle (gs=4, in=8, out=4, axis=1): `W = (arange(32).reshape(4,8) -
/// 16.0) * 0.25` quantized by `Quantizer.quantize(nbits=4, group_size=4,
/// axis=1, bitpack=True)`. Two groups per output row — the per-row HQQ
/// model cannot represent this. Verifies the production state-dict path
/// reproduces `Quantizer.dequantize` byte-for-byte (within f32 tol).
#[test]
fn hqq_axis1_state_dict_to_dense_matches_reference_oracle() {
    let mk = |d: Vec<f32>, shape: Vec<usize>| {
        Tensor::from_storage(TensorStorage::cpu(d), shape, false).unwrap()
    };
    let mut raw: StateDict<f32> = HashMap::new();
    // pack_4bit_u8 output, shape [num_groups/2=4, group_size=4].
    raw.insert(
        "lm_head.W_q".to_string(),
        mk(
            vec![
                0.0, 85.0, 170.0, 255.0, 0.0, 85.0, 170.0, 255.0, 0.0, 85.0, 170.0, 255.0, 0.0,
                85.0, 170.0, 255.0,
            ],
            vec![4, 4],
        ),
    );
    raw.insert("lm_head.scale".to_string(), mk(vec![0.05; 8], vec![8, 1]));
    raw.insert(
        "lm_head.zero".to_string(),
        mk(
            vec![80.0, 60.0, 40.0, 20.0, 0.0, -20.0, -40.0, -60.0],
            vec![8, 1],
        ),
    );
    raw.insert("lm_head.nbits".to_string(), mk(vec![4.0], vec![1]));
    raw.insert("lm_head.group_size".to_string(), mk(vec![4.0], vec![1]));
    raw.insert("lm_head.shape".to_string(), mk(vec![4.0, 8.0], vec![2]));

    let dense = hqq_state_dict_to_dense(&raw).expect("hqq_state_dict_to_dense");
    let w = dense
        .get("lm_head.weight")
        .expect("dense state dict missing lm_head.weight");
    assert_eq!(w.shape(), &[4, 8]);

    let expected: [f32; 32] = [
        -4.0, -3.75, -3.5, -3.25, -3.0, -2.75, -2.5, -2.25, -2.0, -1.75, -1.5, -1.25, -1.0, -0.75,
        -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25,
        3.5, 3.75,
    ];
    let got = w.data_vec().unwrap();
    assert_allclose(
        &got,
        &expected,
        F32_TRANSCENDENTAL,
        "hqq_axis1/lm_head.weight",
    );
}

/// Direct check of [`dequantize_hqq_q4_axis1`] against the gs=8 reference
/// oracle (one group per row) — the `out=4, in=8` case from
/// `Quantizer.dequantize`.
#[test]
fn hqq_q4_axis1_dequant_matches_reference_oracle_gs8() {
    let tile = HqqQ4Axis1::new(
        vec![
            0, 34, 68, 102, 153, 187, 221, 255, 0, 34, 68, 102, 153, 187, 221, 255,
        ],
        vec![0.046_666_67, 0.046_666_67, 0.046_666_66, 0.046_666_67],
        vec![21.428_572, 4.285_714, -12.857_144, -30.0],
        8,
        4,
        8,
    );
    let out = dequantize_hqq_q4_axis1(&tile).expect("dequantize_hqq_q4_axis1");
    let expected: [f32; 32] = [
        -1.0, -0.906_667, -0.813_333, -0.72, -0.58, -0.486_667, -0.393_333, -0.3, -0.2, -0.106_667,
        -0.013_333, 0.08, 0.22, 0.313_333, 0.406_667, 0.5, 0.6, 0.693_333, 0.786_667, 0.88, 1.02,
        1.113_333, 1.206_667, 1.3, 1.4, 1.493_333, 1.586_667, 1.68, 1.82, 1.913_334, 2.006_667,
        2.1,
    ];
    assert_allclose(&out, &expected, 1e-4, "hqq_axis1/gs8");
}

// ---------------------------------------------------------------------------
// Tensor construction helper
// ---------------------------------------------------------------------------

/// Build a CPU [`Tensor<f32>`] from a flat `Vec<f32>` and explicit shape.
fn make_tensor_f32(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data), shape.to_vec(), false)
        .expect("make_tensor_f32: Tensor::from_storage failed")
}
