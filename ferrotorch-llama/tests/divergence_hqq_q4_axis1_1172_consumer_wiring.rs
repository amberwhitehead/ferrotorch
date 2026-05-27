//! End-to-end consumer pin: `LlamaForCausalLM::load_hqq_state_dict` must
//! actually wire the HQQ-dequantized weight into the model parameter (not a
//! stub), bit-for-bit matching the `mobiusml/hqq` reference dequant (#1172,
//! scrutiny item 5).
//!
//! Builds a 1-layer Llama, hands `load_hqq_state_dict` a raw HQQ-format state
//! dict whose `self_attn.q_proj` is HQQ Q4 (axis=1, gs=8) and reads the
//! loaded `q_proj.weight` back out of `model.state_dict()`. The expected
//! values are the live `Quantizer.dequantize` output (R-CHAR-3), so this
//! fails if the consumer drops the dequant, loads zeros, or mismatches keys.

use ferrotorch_core::{Tensor, TensorStorage};
use ferrotorch_llama::{LlamaActivation, LlamaConfig, LlamaForCausalLM};
use ferrotorch_nn::module::{Module, StateDict};

fn t(d: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(d), shape, false).unwrap()
}

/// HQQ reference dequant of the [8,8] `q_proj.weight` (live oracle).
const Q_PROJ_DEQUANT: [f32; 64] = [
    -0.09532061, 0.33364725, -0.7387724, 0.11916332, 0.7626151, -0.30980456, 1.6205509,
    -1.5967082, -0.062150154, -0.3059757, 0.26295057, -0.22470051, 0.913152, 0.18167539,
    0.58805126, 0.42550093, 0.3266392, 0.8661768, -0.57259017, -1.6516653, 1.0460227, 0.50648504,
    -0.2128984, -0.033052538, 0.49952698, 2.1942883, 1.9521796, -1.1952343, 0.98374444, 0.49952698,
    2.436397, -0.22679928, -0.43573517, -0.2627815, -0.34925833, 0.34255627, -0.003351033,
    -0.08982786, 0.4290331, -0.8681193, -0.24719663, 0.10835528, 0.10835528, 0.7009418,
    -0.9583004, 0.10835528, 0.7009418, 0.81945908, -0.5103969, 2.2124186, -0.5103969, -0.20786187,
    0.69974333, -2.3256073, 1.3048134, -0.20786187, 0.9742818, -1.2448393, -1.6483159, 0.77254355,
    -0.8413627, -0.43788618, 1.3777584, 1.3777584,
];

fn tiny_config() -> LlamaConfig {
    LlamaConfig {
        vocab_size: 16,
        hidden_size: 8,
        intermediate_size: 16,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        num_key_value_heads: 2,
        rms_norm_eps: 1e-5,
        rope_theta: 10_000.0,
        max_position_embeddings: 32,
        tie_word_embeddings: false,
        hidden_act: LlamaActivation::Silu,
    }
}

#[test]
fn divergence_hqq_consumer_loads_dequantized_q_proj_weight() {
    let mut raw: StateDict<f32> = StateDict::new();

    // q_proj: HQQ Q4 axis=1 gs=8, original shape [8,8]. Live-oracle buffers.
    let w_q: Vec<f32> = vec![
        117., 151., 70., 142., 186., 105., 255., 0., 54., 9., 121., 30., 240., 105., 190., 159.,
        182., 239., 102., 7., 250., 192., 140., 151., 125., 226., 208., 12., 148., 118., 255., 79.,
    ];
    let scale: Vec<f32> = vec![
        0.21448393, 0.08127518, 0.17984587, 0.24210875, 0.086476825, 0.1185173, 0.30253506,
        0.20173828,
    ];
    let zero: Vec<f32> = vec![
        7.4444184, 3.764688, 9.183783, 4.936766, 10.038751, 8.085743, 7.687067, 8.170566,
    ];
    let p = "model.layers.0.self_attn.q_proj";
    raw.insert(format!("{p}.W_q"), t(w_q, vec![4, 8]));
    raw.insert(format!("{p}.scale"), t(scale, vec![8, 1]));
    raw.insert(format!("{p}.zero"), t(zero, vec![8, 1]));
    raw.insert(format!("{p}.nbits"), t(vec![4.0], vec![1]));
    raw.insert(format!("{p}.group_size"), t(vec![8.0], vec![1]));
    raw.insert(format!("{p}.shape"), t(vec![8.0, 8.0], vec![2]));

    let mut model = LlamaForCausalLM::<f32>::new(tiny_config()).unwrap();
    model
        .load_hqq_state_dict(&raw, false)
        .expect("load_hqq_state_dict should succeed");

    // Read the loaded parameter back out and compare to the HQQ reference.
    let sd = model.state_dict();
    let key = "model.layers.0.self_attn.q_proj.weight";
    let w = sd
        .get(key)
        .unwrap_or_else(|| panic!("model has no {key} after load_hqq_state_dict"));
    assert_eq!(w.shape(), &[8, 8], "q_proj.weight shape");
    let got = w.data_vec().unwrap();
    for (i, (g, e)) in got.iter().zip(Q_PROJ_DEQUANT.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-4,
            "q_proj.weight[{i}]: model holds {g}, HQQ reference dequant is {e} \
             (consumer failed to wire the dequantized weight)"
        );
    }
}
