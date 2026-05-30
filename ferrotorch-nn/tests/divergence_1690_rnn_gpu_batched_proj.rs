//! #1690 GPU probe — RNN/GRU/LSTM forward on CUDA after batched input
//! projection (`#![cfg(feature = "cuda")]`, no-op on non-CUDA hosts).
//!
//! The builder noted the GPU GRU per-step `xw` is a `narrow` VIEW into the
//! batched-projection buffer, and the fused GRU kernel is offset-unaware, so
//! the per-step slice is `.contiguous()`-materialized before its handle
//! reaches `fused_gru_cell_f32`. If that `contiguous()` does NOT fire (a
//! narrowed view reaching the kernel at offset 0), every timestep after t=0
//! would read the wrong block and the forward would be corrupt.
//!
//! Distinct-per-timestep input (x[b,t,:] has a large per-t signature) makes a
//! wrong-offset read visible: t>0 would otherwise read t=0's projection.
//!
//! R-CHAR-3: expected values are LIVE torch.cuda 2.11.0+cu130 (native,
//! cudnn-disabled — the non-cuDNN CUDA RNN path, matching ferrotorch's own
//! non-cuDNN fused-cell composition) from `/tmp/rnn_1690_gpu_oracle.py`.
//! Constants in `rnn_1690_gpu_refs.rs.inc`.

#![cfg(feature = "cuda")]
#![allow(clippy::excessive_precision)]
#![allow(clippy::approx_constant)]

use ferrotorch_core::{Device, Tensor, TensorStorage};
use ferrotorch_nn::module::Module as _;
use ferrotorch_nn::{GRU, LSTM};
use std::collections::HashMap;

include!("rnn_1690_gpu_refs.rs.inc");

fn cuda_ready() -> bool {
    ferrotorch_gpu::init_cuda_backend().is_ok()
}

fn cpu_tensor(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

fn ramp(n: usize, scale: f32, offset: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 * 0.137 + offset).sin()) * scale)
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32], ctx: &str) {
    assert_eq!(actual.len(), expected.len(), "{ctx}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= 1e-3 + 1e-3 * e.abs(),
            "{ctx}: GPU diverges from torch.cuda at {i}: ferrotorch={a} torch={e}"
        );
    }
}

fn sd1(hs: usize, insz: usize, gate: usize) -> HashMap<String, Tensor<f32>> {
    let mut sd = HashMap::new();
    sd.insert(
        "layers.0.weight_ih".to_string(),
        cpu_tensor(&ramp(gate * hs * insz, 0.3, 0.0), &[gate * hs, insz]),
    );
    sd.insert(
        "layers.0.weight_hh".to_string(),
        cpu_tensor(&ramp(gate * hs * hs, 0.3, 1.7), &[gate * hs, hs]),
    );
    sd.insert(
        "layers.0.bias_ih".to_string(),
        cpu_tensor(&ramp(gate * hs, 0.1, 0.5), &[gate * hs]),
    );
    sd.insert(
        "layers.0.bias_hh".to_string(),
        cpu_tensor(&ramp(gate * hs, 0.1, 2.3), &[gate * hs]),
    );
    sd
}

#[test]
fn gru_gpu_batched_proj_forward_vs_torch_cuda() {
    if !cuda_ready() {
        return;
    }
    let (batch, seq, insz, hs) = (3usize, 5, 4, 6);
    let mut gru = GRU::<f32>::new(insz, hs).unwrap();
    gru.load_state_dict(&sd1(hs, insz, 3), true).unwrap();
    gru.to_device(Device::Cuda(0)).unwrap();

    let x = cpu_tensor(GRU_GPU_XIN, &[batch, seq, insz])
        .to(Device::Cuda(0))
        .unwrap();
    let h0 = cpu_tensor(&ramp(batch * hs, 0.2, 3.1), &[1, batch, hs])
        .to(Device::Cuda(0))
        .unwrap();

    let (out, hn) = gru.forward(&x, Some(&h0)).unwrap();
    assert!(out.is_cuda(), "GPU GRU output must stay on CUDA");
    assert_close(&out.data_vec().unwrap(), GRU_GPU_OUT, "GRU GPU output");
    assert_close(&hn.data_vec().unwrap(), GRU_GPU_HN, "GRU GPU h_n");
}

#[test]
fn lstm_gpu_batched_proj_forward_vs_torch_cuda() {
    if !cuda_ready() {
        return;
    }
    let (batch, seq, insz, hs) = (3usize, 5, 4, 6);
    let mut lstm = LSTM::<f32>::new(insz, hs, 1).unwrap();
    lstm.load_state_dict(&sd1(hs, insz, 4), true).unwrap();
    lstm.to_device(Device::Cuda(0)).unwrap();

    let x = cpu_tensor(LSTM_GPU_XIN, &[batch, seq, insz])
        .to(Device::Cuda(0))
        .unwrap();
    let h0 = cpu_tensor(&ramp(batch * hs, 0.2, 3.1), &[1, batch, hs])
        .to(Device::Cuda(0))
        .unwrap();
    let c0 = cpu_tensor(&ramp(batch * hs, 0.2, 4.4), &[1, batch, hs])
        .to(Device::Cuda(0))
        .unwrap();

    let (out, (hn, cn)) = lstm.forward_with_state(&x, Some((&h0, &c0))).unwrap();
    assert!(out.is_cuda(), "GPU LSTM output must stay on CUDA");
    assert_close(&out.data_vec().unwrap(), LSTM_GPU_OUT, "LSTM GPU output");
    assert_close(&hn.data_vec().unwrap(), LSTM_GPU_HN, "LSTM GPU h_n");
    assert_close(&cn.data_vec().unwrap(), LSTM_GPU_CN, "LSTM GPU c_n");
}
