//! Live GPU multi-step test for [`GraphedDecoder`] (#1595 / #1355 / #1358).
//!
//! This is the verification that the prior builder could NOT pass: the
//! captured single-token decode forward, replayed N>=3 times in a row,
//! must produce logits BIT-IDENTICAL to the eager oracle, AND the private
//! graph-capture mempool must isolate the captured allocations so that
//! interleaved eager forwards after a replay still succeed (the bug that
//! caused #1595 was the shared default async pool corrupting on replay,
//! failing the 2nd decode_step and every subsequent eager forward with
//! `CUDA_ERROR_INVALID_VALUE`).
//!
//! R-CHAR-3: the eager `forward_from_ids(&[token])` is the oracle. The
//! graphed logits are compared bit-for-bit against the eager logits the
//! same model produces for the same token — not against any hard-coded
//! constant.
//!
//! Run with `cargo test -p ferrotorch-llama --features cuda`.

#![cfg(feature = "cuda")]

use std::collections::HashMap;

use ferrotorch_core::{Tensor, TensorStorage};
use ferrotorch_gpu::GpuDevice;
use ferrotorch_llama::{GraphedDecoder, LlamaConfig, LlamaGpuInferencer};
use half::bf16;

/// Same deterministic synthetic fill as `gpu_smoke.rs`: a small non-zero
/// pattern so RMSNorm doesn't divide by zero and matmuls aren't trivially
/// zero.
fn fill_tensor(shape: Vec<usize>) -> Tensor<bf16> {
    let numel: usize = shape.iter().product();
    let data: Vec<bf16> = (0..numel)
        .map(|i| bf16::from_f32(0.01 * ((i % 13) as f32 + 1.0)))
        .collect();
    Tensor::from_storage(TensorStorage::cpu(data), shape, false)
        .expect("synthetic tensor construction must succeed for valid shapes")
}

fn tiny_cfg() -> LlamaConfig {
    LlamaConfig {
        vocab_size: 32,
        hidden_size: 16,
        intermediate_size: 32,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        num_key_value_heads: 2,
        rms_norm_eps: 1e-5,
        rope_theta: 10_000.0,
        max_position_embeddings: 16,
        tie_word_embeddings: false,
        hidden_act: ferrotorch_llama::LlamaActivation::Silu,
    }
}

fn build_state(cfg: &LlamaConfig) -> HashMap<String, Tensor<bf16>> {
    let head_dim = cfg.head_dim();
    let kv_dim = cfg.num_key_value_heads * head_dim;
    let mut state: HashMap<String, Tensor<bf16>> = HashMap::new();
    state.insert(
        "model.embed_tokens.weight".to_string(),
        fill_tensor(vec![cfg.vocab_size, cfg.hidden_size]),
    );
    state.insert(
        "model.norm.weight".to_string(),
        fill_tensor(vec![cfg.hidden_size]),
    );
    state.insert(
        "lm_head.weight".to_string(),
        fill_tensor(vec![cfg.vocab_size, cfg.hidden_size]),
    );
    for i in 0..cfg.num_hidden_layers {
        state.insert(
            format!("model.layers.{i}.input_layernorm.weight"),
            fill_tensor(vec![cfg.hidden_size]),
        );
        state.insert(
            format!("model.layers.{i}.post_attention_layernorm.weight"),
            fill_tensor(vec![cfg.hidden_size]),
        );
        state.insert(
            format!("model.layers.{i}.self_attn.q_proj.weight"),
            fill_tensor(vec![cfg.hidden_size, cfg.hidden_size]),
        );
        state.insert(
            format!("model.layers.{i}.self_attn.k_proj.weight"),
            fill_tensor(vec![kv_dim, cfg.hidden_size]),
        );
        state.insert(
            format!("model.layers.{i}.self_attn.v_proj.weight"),
            fill_tensor(vec![kv_dim, cfg.hidden_size]),
        );
        state.insert(
            format!("model.layers.{i}.self_attn.o_proj.weight"),
            fill_tensor(vec![cfg.hidden_size, cfg.hidden_size]),
        );
        state.insert(
            format!("model.layers.{i}.mlp.gate_proj.weight"),
            fill_tensor(vec![cfg.intermediate_size, cfg.hidden_size]),
        );
        state.insert(
            format!("model.layers.{i}.mlp.up_proj.weight"),
            fill_tensor(vec![cfg.intermediate_size, cfg.hidden_size]),
        );
        state.insert(
            format!("model.layers.{i}.mlp.down_proj.weight"),
            fill_tensor(vec![cfg.hidden_size, cfg.intermediate_size]),
        );
    }
    state
}

fn build_inferencer() -> LlamaGpuInferencer {
    let cfg = tiny_cfg();
    cfg.validate().expect("cfg must validate");
    let state = build_state(&cfg);
    let device =
        GpuDevice::new(0).expect("CUDA device 0 must be available for the graphed-decoder test");
    LlamaGpuInferencer::new(cfg, state, device)
        .expect("LlamaGpuInferencer::new must succeed with a synthetic StateDict")
}

/// Compare two logit vectors bit-exactly (after the bf16 round trip both
/// are f32-from-bf16-bits, so exact equality is the right test — the
/// graph replays the identical kernel sequence the eager path runs).
fn assert_bit_identical(graphed: &[f32], eager: &[f32], step: usize, token: u32) {
    assert_eq!(
        graphed.len(),
        eager.len(),
        "step {step} token {token}: length mismatch graphed={} eager={}",
        graphed.len(),
        eager.len()
    );
    for (i, (&g, &e)) in graphed.iter().zip(eager.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "step {step} token {token}: logit[{i}] graphed={g} (0x{:08x}) != eager={e} (0x{:08x})",
            g.to_bits(),
            e.to_bits()
        );
    }
}

/// N>=3 sequential graphed decode_steps each bit-identical to the eager
/// oracle. The 2nd+ replays are exactly what failed in #1595 before the
/// private mempool — if the shared pool were still being used, the 2nd
/// decode_step would error with CUDA_ERROR_INVALID_VALUE.
#[test]
fn graphed_decode_multi_step_matches_eager_oracle() {
    let model = build_inferencer();

    // Capture the single-token decode graph (warm-up token 1).
    let mut decoder =
        GraphedDecoder::capture(&model, 1u32).expect("GraphedDecoder::capture must succeed");

    // 5 sequential replays, each a different token, each compared to the
    // eager forward of that same single token.
    let tokens: [u32; 5] = [1, 7, 13, 2, 31];
    for (step, &token) in tokens.iter().enumerate() {
        let graphed = decoder
            .decode_step(token)
            .unwrap_or_else(|e| panic!("step {step} token {token}: decode_step failed: {e:?}"));
        let eager = model
            .forward_from_ids(&[token])
            .unwrap_or_else(|e| panic!("step {step} token {token}: eager oracle failed: {e:?}"));
        assert_bit_identical(&graphed, &eager, step, token);
    }

    assert_eq!(
        decoder.num_replays(),
        tokens.len() as u64,
        "expected one replay per decode_step"
    );
}

/// Interleave graphed replay with eager forwards: graphed -> eager ->
/// graphed -> eager -> graphed. This is the direct proof that the private
/// mempool isolates the captured allocations from the eager path — under
/// #1595's bug, the first eager forward AFTER a graphed replay failed with
/// CUDA_ERROR_INVALID_VALUE because the shared default pool was corrupted.
#[test]
fn graphed_decode_interleaved_with_eager_forwards() {
    let model = build_inferencer();
    let mut decoder =
        GraphedDecoder::capture(&model, 1u32).expect("GraphedDecoder::capture must succeed");

    let tokens: [u32; 3] = [5, 11, 20];
    for (step, &token) in tokens.iter().enumerate() {
        // Graphed replay.
        let graphed = decoder
            .decode_step(token)
            .unwrap_or_else(|e| panic!("step {step}: graphed decode_step failed: {e:?}"));

        // Interleaved eager forward of the SAME token — must succeed
        // (proves the eager default pool is intact after the replay) and
        // must match the graphed result bit-for-bit.
        let eager = model
            .forward_from_ids(&[token])
            .unwrap_or_else(|e| panic!("step {step}: interleaved eager forward failed: {e:?}"));
        assert_bit_identical(&graphed, &eager, step, token);

        // An additional multi-token eager forward to stress the shared
        // pool with a different allocation size after the replay.
        let multi = model
            .forward_from_ids(&[token, token, token])
            .unwrap_or_else(|e| panic!("step {step}: post-replay multi-token eager failed: {e:?}"));
        assert_eq!(multi.len(), model.config.vocab_size);
        for &v in &multi {
            assert!(
                v.is_finite(),
                "step {step}: post-replay eager produced non-finite logit"
            );
        }
    }
}
