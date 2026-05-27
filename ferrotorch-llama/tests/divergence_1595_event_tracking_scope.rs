//! Concern #5 of the aa2483075 audit: the cudarc per-CudaSlice event
//! tracking that `capture_into_private_pool` disables for the capture window
//! must be RE-ENABLED (scoped) after capture. If it stayed off globally,
//! later eager ops on other streams would silently lose their stream-ordering
//! safety.
//!
//! cudarc's `CudaContext` defaults `event_tracking = true` (core.rs ctor).
//! `EventTrackingRestore::drop` (graph.rs:704-712) re-enables it on every exit
//! path of `capture_into_private_pool`. This test pins that the context the
//! GraphedDecoder captured on has event tracking ON again after `capture`
//! returns — i.e. the disable was scoped to the capture window, not leaked.
//!
//! The oracle is cudarc's own `CudaContext::is_event_tracking()` (a named
//! symbolic bit, not a value copied from ferrotorch) — R-CHAR-3 compliant.
//!
//! Run: cargo test -p ferrotorch-llama --features cuda \
//!        --test divergence_1595_event_tracking_scope

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

/// After `GraphedDecoder::capture` returns, the context's event tracking
/// must be back ON (the capture-window disable was scoped). The GraphedDecoder
/// forks a capture device sharing the model's `CudaContext`, so the model
/// device's context is the same one whose flag was toggled.
#[test]
fn event_tracking_restored_after_capture() {
    let model = build_inferencer();

    // Precondition: a freshly built inferencer's context has event tracking
    // ON (cudarc default). This is the named symbolic bit we restore to.
    assert!(
        model.device.context().is_event_tracking(),
        "precondition: cudarc CudaContext defaults event_tracking=true"
    );

    let _decoder = GraphedDecoder::capture(&model, 1u32).expect("capture must succeed");

    // After capture, the disable issued inside capture_into_private_pool must
    // have been undone by EventTrackingRestore::drop on the way out.
    assert!(
        model.device.context().is_event_tracking(),
        "event tracking must be RE-ENABLED after GraphedDecoder::capture returns \
         (the capture-window disable must be scoped, not leaked globally)"
    );
}

/// Two sequential captures must each leave tracking ON afterwards (the
/// disable/enable toggle is process-wide and not refcounted, so we pin that
/// back-to-back captures don't leave it stuck off).
#[test]
fn event_tracking_restored_after_two_captures() {
    let model = build_inferencer();
    assert!(model.device.context().is_event_tracking());

    let d1 = GraphedDecoder::capture(&model, 1u32).expect("capture 1");
    assert!(
        model.device.context().is_event_tracking(),
        "tracking on after capture 1"
    );
    let d2 = GraphedDecoder::capture(&model, 2u32).expect("capture 2");
    assert!(
        model.device.context().is_event_tracking(),
        "tracking on after capture 2"
    );
    drop(d1);
    drop(d2);
    assert!(
        model.device.context().is_event_tracking(),
        "tracking still on after both decoders dropped"
    );
}
