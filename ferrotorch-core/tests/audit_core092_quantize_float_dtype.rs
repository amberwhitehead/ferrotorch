//! CORE-092 (#1786): public quantize/dequantize must follow PyTorch's
//! floating dtype contract instead of silently narrowing f64/f16/bf16 through
//! f32.
//!
//! Local PyTorch 2.11.0+cu130 oracles:
//!
//! ```python
//! torch.quantize_per_tensor(torch.tensor([1.], dtype=torch.float64),
//!                           0.1, 0, torch.qint8)
//! # RuntimeError: Quantize only works on Float Tensor, got Double
//!
//! torch.quantize_per_channel(torch.ones(2, 2, dtype=torch.float16),
//!                            torch.tensor([0.1, 0.1]),
//!                            torch.tensor([0, 0]), 0, torch.qint8)
//! # RuntimeError: quantize_tensor_per_channel_affine expects a Float Tensor, got Half
//!
//! q = torch.quantize_per_tensor(torch.tensor([1.], dtype=torch.float32),
//!                               0.1, 0, torch.qint8)
//! q.dequantize().dtype
//! # torch.float32
//! ```

use ferrotorch_core::quantize::{QuantDtype, QuantScheme, dequantize, quantize};
use ferrotorch_core::{FerrotorchError, Tensor, TensorStorage};

fn cpu_tensor<T: ferrotorch_core::Float>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
    Tensor::from_storage(TensorStorage::cpu(data), shape, false).expect("cpu tensor")
}

fn invalid_message(result: FerrotorchError) -> String {
    match result {
        FerrotorchError::InvalidArgument { message } => message,
        other => panic!("expected InvalidArgument, got {other:?}"),
    }
}

#[test]
fn quantize_rejects_f64_before_any_precision_or_range_narrowing() {
    let huge = cpu_tensor(
        vec![1.0e39_f64, 1.0e39_f64 + 1.0e25_f64, -1.0e39_f64],
        vec![3],
    );
    let err = quantize(&huge, QuantScheme::PerTensor, QuantDtype::Int8)
        .expect_err("f64 quantize must reject like PyTorch");
    let message = invalid_message(err);

    assert!(
        message.contains("Quantize only works on Float Tensor, got Double"),
        "unexpected f64 per-tensor error: {message}"
    );
}

#[test]
fn quantize_per_channel_rejects_f64_like_pytorch() {
    let x = cpu_tensor(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]);
    let err = quantize(&x, QuantScheme::PerChannel(0), QuantDtype::Int8)
        .expect_err("f64 per-channel quantize must reject like PyTorch");
    let message = invalid_message(err);

    assert!(
        message.contains("quantize_tensor_per_channel_affine expects a Float Tensor, got Double"),
        "unexpected f64 per-channel error: {message}"
    );
}

#[test]
fn quantize_rejects_half_and_bfloat16_like_pytorch() {
    let f16 = cpu_tensor(
        vec![half::f16::from_f32(1.0), half::f16::from_f32(2.0)],
        vec![2],
    );
    let f16_err = quantize(&f16, QuantScheme::PerTensor, QuantDtype::Int8)
        .expect_err("f16 quantize must reject like PyTorch");
    let f16_message = invalid_message(f16_err);
    assert!(
        f16_message.contains("Quantize only works on Float Tensor, got Half"),
        "unexpected f16 error: {f16_message}"
    );

    let bf16 = cpu_tensor(
        vec![half::bf16::from_f32(1.0), half::bf16::from_f32(2.0)],
        vec![2],
    );
    let bf16_err = quantize(&bf16, QuantScheme::PerTensor, QuantDtype::Int8)
        .expect_err("bf16 quantize must reject like PyTorch");
    let bf16_message = invalid_message(bf16_err);
    assert!(
        bf16_message.contains("Quantize only works on Float Tensor, got BFloat16"),
        "unexpected bf16 error: {bf16_message}"
    );
}

#[test]
fn dequantize_only_returns_f32_like_pytorch() {
    let x = cpu_tensor(vec![1.0_f32, 2.0, 3.0, 4.0], vec![4]);
    let q = quantize(&x, QuantScheme::PerTensor, QuantDtype::Int8).expect("f32 quantize");
    let y: Tensor<f32> = dequantize(&q).expect("f32 dequantize");
    assert_eq!(y.shape(), &[4]);

    let err = dequantize::<f64>(&q).expect_err("f64 dequantize must reject");
    let message = invalid_message(err);
    assert!(
        message.contains("dequantize returns a Float Tensor in PyTorch; requested Double"),
        "unexpected dequantize f64 error: {message}"
    );
}
