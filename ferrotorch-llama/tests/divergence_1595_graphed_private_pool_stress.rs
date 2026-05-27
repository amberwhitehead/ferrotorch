//! Adversarial stress audit of the private CUDA graph-capture mempool +
//! `GraphedDecoder` shipped in commit aa2483075 (#1595 / #1355 / #1358).
//!
//! The builder's `graphed_decoder_live.rs` verifies only 5 sequential
//! graphed steps and a single `graphed -> eager -> graphed` interleaving.
//! This file stresses the failure class the user warns about: a GPU thing
//! that compiles + passes the happy path but corrupts state on an UNTESTED
//! pattern. Every assertion uses the eager `forward_from_ids` as the oracle
//! (R-CHAR-3) — no hard-coded constants, no graphed value copied into the
//! expected side.
//!
//! Concerns mapped from the audit brief:
//!   1. Multi-step correctness at depth (>= 8 sequential replays).
//!   2. Interleavings the builder did NOT test (graphed->eager->eager->
//!      graphed; big eager alloc between replays; two decoders alive).
//!   3. RAII restore correctness (eager-heavy workload AFTER a capture must
//!      draw from the restored DEFAULT pool, not the private one).
//!   4. Address stability across replays (token T's logits reflect token T,
//!      not a stale buffer — proven by alternating tokens whose eager logits
//!      differ).
//!
//! Run: cargo test -p ferrotorch-llama --features cuda \
//!        --test divergence_1595_graphed_private_pool_stress

#![cfg(feature = "cuda")]

use std::collections::HashMap;

use ferrotorch_core::{Tensor, TensorStorage};
use ferrotorch_gpu::GpuDevice;
use ferrotorch_llama::{GraphedDecoder, LlamaConfig, LlamaGpuInferencer};
use half::bf16;

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

/// Bit-exact comparison; the eager path is the oracle (R-CHAR-3).
fn assert_bit_identical(graphed: &[f32], eager: &[f32], ctx: &str) {
    assert_eq!(
        graphed.len(),
        eager.len(),
        "{ctx}: length mismatch graphed={} eager={}",
        graphed.len(),
        eager.len()
    );
    for (i, (&g, &e)) in graphed.iter().zip(eager.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "{ctx}: logit[{i}] graphed={g} (0x{:08x}) != eager={e} (0x{:08x})",
            g.to_bits(),
            e.to_bits()
        );
    }
}

// ---------------------------------------------------------------------------
// Concern 1 — depth. 12 sequential graphed replays, each vs the eager oracle.
// A pool-recycling / address-reuse bug may only surface after several
// replays past the builder's 5.
// ---------------------------------------------------------------------------
#[test]
fn graphed_decode_twelve_sequential_steps_match_eager() {
    let model = build_inferencer();
    let mut decoder =
        GraphedDecoder::capture(&model, 1u32).expect("GraphedDecoder::capture must succeed");

    // 12 steps cycling through distinct tokens (depth >> builder's 5).
    let tokens: [u32; 12] = [1, 7, 13, 2, 31, 0, 19, 4, 28, 9, 30, 3];
    for (step, &token) in tokens.iter().enumerate() {
        let graphed = decoder
            .decode_step(token)
            .unwrap_or_else(|e| panic!("step {step} token {token}: decode_step failed: {e:?}"));
        let eager = model
            .forward_from_ids(&[token])
            .unwrap_or_else(|e| panic!("step {step} token {token}: eager oracle failed: {e:?}"));
        assert_bit_identical(
            &graphed,
            &eager,
            &format!("depth step {step} token {token}"),
        );
    }
    assert_eq!(decoder.num_replays(), tokens.len() as u64);
}

// ---------------------------------------------------------------------------
// Concern 4 — address stability. Alternate between two tokens whose eager
// logits DIFFER, so a stale-buffer bug (graph reading the previous replay's
// input/output buffer) would surface as token A's logits leaking into token
// B's slot. We first assert the two tokens' eager logits actually differ
// (else the test is vacuous), then drive the graph A,B,A,B,... checking each.
// ---------------------------------------------------------------------------
#[test]
fn graphed_decode_alternating_tokens_no_stale_buffer() {
    let model = build_inferencer();
    let (tok_a, tok_b) = (3u32, 27u32);

    let eager_a = model.forward_from_ids(&[tok_a]).expect("eager A");
    let eager_b = model.forward_from_ids(&[tok_b]).expect("eager B");
    // Guard against a vacuous test: the two oracles must differ somewhere.
    let differ = eager_a
        .iter()
        .zip(eager_b.iter())
        .any(|(&a, &b)| a.to_bits() != b.to_bits());
    assert!(
        differ,
        "test precondition: tokens {tok_a} and {tok_b} must produce different eager logits"
    );

    let mut decoder = GraphedDecoder::capture(&model, 1u32).expect("capture");
    for round in 0..6 {
        let g_a = decoder.decode_step(tok_a).expect("decode A");
        assert_bit_identical(&g_a, &eager_a, &format!("round {round} token A={tok_a}"));
        let g_b = decoder.decode_step(tok_b).expect("decode B");
        assert_bit_identical(&g_b, &eager_b, &format!("round {round} token B={tok_b}"));
    }
}

// ---------------------------------------------------------------------------
// Concern 2 — the interleaving the builder did NOT test:
// graphed -> eager -> eager -> graphed, with a LARGE eager allocation
// (multi-token prefill) wedged between two replays. If the RAII MemPoolScope
// restore is wrong, the big eager alloc would draw from the private pool and
// either OOM-fragment it or alias a captured buffer -> the next replay would
// diverge from the oracle.
// ---------------------------------------------------------------------------
#[test]
fn graphed_eager_eager_graphed_with_big_eager_alloc_between() {
    let model = build_inferencer();
    let mut decoder = GraphedDecoder::capture(&model, 1u32).expect("capture");

    let token = 11u32;
    let eager_single = model.forward_from_ids(&[token]).expect("eager single");

    for cycle in 0..4 {
        // graphed
        let g1 = decoder.decode_step(token).expect("graphed pre");
        assert_bit_identical(&g1, &eager_single, &format!("cycle {cycle} graphed pre"));

        // eager (single)
        let e1 = model.forward_from_ids(&[token]).expect("eager 1");
        assert_bit_identical(&g1, &e1, &format!("cycle {cycle} eager1 vs graphed"));

        // eager (BIG: a max-length prefill drawing many large intermediates
        // from the DEFAULT pool — this is the allocation that would land in
        // the private pool if the scope restore is broken).
        let big_ids: Vec<u32> = (0..model.config.max_position_embeddings as u32)
            .map(|i| i % model.config.vocab_size as u32)
            .collect();
        let e_big = model
            .forward_logits_from_ids_all(&big_ids)
            .expect("big eager prefill must succeed (default pool intact)");
        assert_eq!(e_big.len(), big_ids.len() * model.config.vocab_size);
        for &v in &e_big {
            assert!(
                v.is_finite(),
                "cycle {cycle}: big eager produced non-finite logit"
            );
        }

        // graphed again — must STILL match the oracle after the big eager
        // allocation churned the default pool.
        let g2 = decoder.decode_step(token).expect("graphed post");
        assert_bit_identical(
            &g2,
            &eager_single,
            &format!("cycle {cycle} graphed post big-eager-alloc"),
        );
    }
}

// ---------------------------------------------------------------------------
// Concern 3 — RAII restore: an eager-HEAVY workload run entirely AFTER a
// capture (decoder already constructed, scope dropped). If MemPoolScope::drop
// failed to restore the device default mempool, these async eager allocations
// would silently draw from the private pool and eventually corrupt the
// captured graph's resident buffers; the subsequent replay would then diverge.
// ---------------------------------------------------------------------------
#[test]
fn eager_heavy_workload_after_capture_then_replay_still_correct() {
    let model = build_inferencer();
    let decoder = GraphedDecoder::capture(&model, 1u32).expect("capture");
    // Hold the decoder (and its private pool) alive; do NOT replay yet.
    let mut decoder = decoder;

    let token = 5u32;
    let eager_single = model.forward_from_ids(&[token]).expect("oracle");

    // 40 eager forwards of varying sizes — heavy churn on the (restored)
    // default async pool.
    for n in 0..40usize {
        let len = (n % model.config.max_position_embeddings) + 1;
        let ids: Vec<u32> = (0..len as u32)
            .map(|i| (i + n as u32) % model.config.vocab_size as u32)
            .collect();
        let out = model
            .forward_from_ids(&ids)
            .unwrap_or_else(|e| panic!("eager forward n={n} len={len} failed: {e:?}"));
        assert_eq!(out.len(), model.config.vocab_size);
        for &v in &out {
            assert!(v.is_finite(), "eager n={n}: non-finite logit");
        }
    }

    // Now replay — the captured graph's resident private-pool buffers must
    // be untouched by all the eager churn above.
    let graphed = decoder.decode_step(token).expect("post-churn replay");
    assert_bit_identical(&graphed, &eager_single, "replay after 40 eager forwards");
}

// ---------------------------------------------------------------------------
// Concern 2 (cont.) — TWO GraphedDecoders alive at once, each with its own
// private pool. Interleave their replays. A double-mempool-swap / global
// event-tracking-state bug (the disable/enable is process-wide, not scoped to
// one pool) would corrupt one decoder when the other captures or replays.
// ---------------------------------------------------------------------------
#[test]
fn two_graphed_decoders_alive_interleaved_replays() {
    let model = build_inferencer();

    let mut dec1 = GraphedDecoder::capture(&model, 1u32).expect("capture dec1");
    // Capturing dec2 happens AFTER dec1 exists: this is the second
    // device-mempool swap + second event-tracking toggle while dec1's
    // captured graph + private pool are already live.
    let mut dec2 = GraphedDecoder::capture(&model, 2u32).expect("capture dec2");

    let tokens: [u32; 6] = [4, 17, 9, 22, 1, 30];
    for (step, &token) in tokens.iter().enumerate() {
        let eager = model
            .forward_from_ids(&[token])
            .unwrap_or_else(|e| panic!("step {step}: eager oracle failed: {e:?}"));

        let g1 = dec1
            .decode_step(token)
            .unwrap_or_else(|e| panic!("step {step}: dec1 decode failed: {e:?}"));
        assert_bit_identical(
            &g1,
            &eager,
            &format!("two-decoder step {step} dec1 token {token}"),
        );

        let g2 = dec2
            .decode_step(token)
            .unwrap_or_else(|e| panic!("step {step}: dec2 decode failed: {e:?}"));
        assert_bit_identical(
            &g2,
            &eager,
            &format!("two-decoder step {step} dec2 token {token}"),
        );
    }
}

// ---------------------------------------------------------------------------
// Concern 6 — leak / cleanup. Drop a GraphedDecoder (PrivateMemPool::drop ->
// cuMemPoolDestroy, MemPoolScope already dropped), then construct a fresh one
// and replay. A double-free / use-after-destroy of the private pool would
// surface here as a capture or replay failure / divergence.
// ---------------------------------------------------------------------------
#[test]
fn drop_decoder_then_recreate_and_replay() {
    let model = build_inferencer();

    let token = 13u32;
    let eager = model.forward_from_ids(&[token]).expect("oracle");

    // First decoder: capture, replay once, then DROP it.
    {
        let mut dec = GraphedDecoder::capture(&model, 1u32).expect("capture dec#1");
        let g = dec.decode_step(token).expect("dec#1 replay");
        assert_bit_identical(&g, &eager, "dec#1 replay");
    } // dec dropped here -> PrivateMemPool::drop -> cuMemPoolDestroy

    // Eager forward in between to use the default pool after the destroy.
    let e_mid = model.forward_from_ids(&[token]).expect("mid eager");
    assert_bit_identical(&e_mid, &eager, "eager after dec#1 drop");

    // Second decoder: fresh capture + private pool after the first was
    // destroyed. Must capture and replay correctly (no use-after-destroy).
    let mut dec2 = GraphedDecoder::capture(&model, 1u32).expect("capture dec#2 after drop");
    for round in 0..3 {
        let g = dec2.decode_step(token).expect("dec#2 replay");
        assert_bit_identical(&g, &eager, &format!("dec#2 replay round {round}"));
    }
}
