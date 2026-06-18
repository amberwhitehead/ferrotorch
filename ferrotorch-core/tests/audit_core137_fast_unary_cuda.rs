#![cfg(feature = "gpu")]

//! CORE-137: direct public `ops::elementwise::fast_*` calls must not fall
//! through to CPU-only `.data()` on CUDA tensors.
//!
//! Oracle: local PyTorch 2.11.0+cu130 CUDA probes on 2026-06-18 for
//! `torch.exp`, `torch.log`, `torch.sigmoid`, and `torch.tanh` on f32, f64,
//! f16, and bf16. These tests call the exported ferrotorch-core fast helpers
//! directly, not the higher-level grad_fns wrappers.

use std::sync::Once;

use ferrotorch_core::creation::from_vec;
use ferrotorch_core::device::Device;
use ferrotorch_core::ops::elementwise::{fast_exp, fast_log, fast_sigmoid, fast_tanh};
use ferrotorch_core::tensor::Tensor;
use half::{bf16, f16};

static GPU_INIT: Once = Once::new();
const LN2_F32: f32 = std::f32::consts::LN_2;
const LN2_F64: f64 = std::f64::consts::LN_2;

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for CORE-137 fast unary tests");
    });
}

fn cuda_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    from_vec::<f32>(data.to_vec(), shape)
        .expect("f32 CPU tensor")
        .to(Device::Cuda(0))
        .expect("upload f32")
}

fn cuda_f64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    from_vec::<f64>(data.to_vec(), shape)
        .expect("f64 CPU tensor")
        .to(Device::Cuda(0))
        .expect("upload f64")
}

fn cuda_f16(data: &[f32], shape: &[usize]) -> Tensor<f16> {
    let values = data.iter().copied().map(f16::from_f32).collect();
    from_vec::<f16>(values, shape)
        .expect("f16 CPU tensor")
        .to(Device::Cuda(0))
        .expect("upload f16")
}

fn cuda_bf16(data: &[f32], shape: &[usize]) -> Tensor<bf16> {
    let values = data.iter().copied().map(bf16::from_f32).collect();
    from_vec::<bf16>(values, shape)
        .expect("bf16 CPU tensor")
        .to(Device::Cuda(0))
        .expect("upload bf16")
}

fn host_f32(t: &Tensor<f32>, label: &str) -> Vec<f32> {
    assert_eq!(t.device(), Device::Cuda(0), "{label}: output device");
    t.data_vec()
        .unwrap_or_else(|e| panic!("{label}: explicit readback failed: {e}"))
}

fn host_f64(t: &Tensor<f64>, label: &str) -> Vec<f64> {
    assert_eq!(t.device(), Device::Cuda(0), "{label}: output device");
    t.data_vec()
        .unwrap_or_else(|e| panic!("{label}: explicit readback failed: {e}"))
}

fn host_f16(t: &Tensor<f16>, label: &str) -> Vec<f32> {
    assert_eq!(t.device(), Device::Cuda(0), "{label}: output device");
    t.cpu()
        .unwrap_or_else(|e| panic!("{label}: f16 D2H failed: {e}"))
        .data()
        .unwrap_or_else(|e| panic!("{label}: f16 host data failed: {e}"))
        .iter()
        .map(|v| v.to_f32())
        .collect()
}

fn host_bf16(t: &Tensor<bf16>, label: &str) -> Vec<f32> {
    assert_eq!(t.device(), Device::Cuda(0), "{label}: output device");
    t.cpu()
        .unwrap_or_else(|e| panic!("{label}: bf16 D2H failed: {e}"))
        .data()
        .unwrap_or_else(|e| panic!("{label}: bf16 host data failed: {e}"))
        .iter()
        .map(|v| v.to_f32())
        .collect()
}

fn assert_close_f32(actual: &[f32], expected: &[f32], rtol: f32, atol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        let limit = atol + rtol * want.abs();
        assert!(
            (got - want).abs() <= limit,
            "{label}[{i}] got {got}, want {want}, limit {limit}"
        );
    }
}

fn assert_close_f64(actual: &[f64], expected: &[f64], rtol: f64, atol: f64, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        let limit = atol + rtol * want.abs();
        assert!(
            (got - want).abs() <= limit,
            "{label}[{i}] got {got}, want {want}, limit {limit}"
        );
    }
}

#[test]
fn direct_fast_unary_cuda_f32_matches_pytorch_and_stays_resident() {
    ensure_cuda_backend();

    let signed = cuda_f32(&[-2.0, -0.5, 0.0, 0.5, 2.0], &[5]);
    let positive = cuda_f32(&[0.25, 0.5, 1.0, 2.0, 4.0], &[5]);

    assert_close_f32(
        &host_f32(
            &fast_exp(&signed).expect("fast_exp f32 CUDA"),
            "fast_exp f32",
        ),
        &[0.13533528, 0.60653067, 1.0, 1.6487212, 7.389056],
        1e-6,
        1e-6,
        "fast_exp f32",
    );
    assert_close_f32(
        &host_f32(
            &fast_log(&positive).expect("fast_log f32 CUDA"),
            "fast_log f32",
        ),
        &[-2.0 * LN2_F32, -LN2_F32, 0.0, LN2_F32, 2.0 * LN2_F32],
        1e-6,
        1e-6,
        "fast_log f32",
    );
    assert_close_f32(
        &host_f32(
            &fast_sigmoid(&signed).expect("fast_sigmoid f32 CUDA"),
            "fast_sigmoid f32",
        ),
        &[0.11920292, 0.37754068, 0.5, 0.62245935, 0.880797],
        1e-6,
        1e-6,
        "fast_sigmoid f32",
    );
    assert_close_f32(
        &host_f32(
            &fast_tanh(&signed).expect("fast_tanh f32 CUDA"),
            "fast_tanh f32",
        ),
        &[-0.9640276, -0.4621172, 0.0, 0.4621172, 0.9640276],
        1e-6,
        1e-6,
        "fast_tanh f32",
    );
}

#[test]
fn direct_fast_unary_cuda_f64_matches_pytorch_and_stays_resident() {
    ensure_cuda_backend();

    let signed = cuda_f64(&[-2.0, -0.5, 0.0, 0.5, 2.0], &[5]);
    let positive = cuda_f64(&[0.25, 0.5, 1.0, 2.0, 4.0], &[5]);

    assert_close_f64(
        &host_f64(
            &fast_exp(&signed).expect("fast_exp f64 CUDA"),
            "fast_exp f64",
        ),
        &[
            0.1353352832366127,
            0.6065306597126334,
            1.0,
            1.6487212707001282,
            7.38905609893065,
        ],
        1e-12,
        1e-12,
        "fast_exp f64",
    );
    assert_close_f64(
        &host_f64(
            &fast_log(&positive).expect("fast_log f64 CUDA"),
            "fast_log f64",
        ),
        &[-2.0 * LN2_F64, -LN2_F64, 0.0, LN2_F64, 2.0 * LN2_F64],
        1e-12,
        1e-12,
        "fast_log f64",
    );
    assert_close_f64(
        &host_f64(
            &fast_sigmoid(&signed).expect("fast_sigmoid f64 CUDA"),
            "fast_sigmoid f64",
        ),
        &[
            0.11920292202211755,
            0.3775406687981454,
            0.5,
            0.6224593312018546,
            0.8807970779778823,
        ],
        1e-12,
        1e-12,
        "fast_sigmoid f64",
    );
    assert_close_f64(
        &host_f64(
            &fast_tanh(&signed).expect("fast_tanh f64 CUDA"),
            "fast_tanh f64",
        ),
        &[
            -0.9640275800758169,
            -0.4621171572600098,
            0.0,
            0.4621171572600098,
            0.9640275800758169,
        ],
        1e-12,
        1e-12,
        "fast_tanh f64",
    );
}

#[test]
fn direct_fast_unary_cuda_reduced_dtypes_match_pytorch_rounding() {
    ensure_cuda_backend();

    let signed_f16 = cuda_f16(&[-2.0, -0.5, 0.0, 0.5, 2.0], &[5]);
    let positive_f16 = cuda_f16(&[0.25, 0.5, 1.0, 2.0, 4.0], &[5]);
    assert_close_f32(
        &host_f16(
            &fast_exp(&signed_f16).expect("fast_exp f16 CUDA"),
            "fast_exp f16",
        ),
        &[0.13537598, 0.6064453, 1.0, 1.6484375, 7.390625],
        0.0,
        0.0,
        "fast_exp f16",
    );
    assert_close_f32(
        &host_f16(
            &fast_log(&positive_f16).expect("fast_log f16 CUDA"),
            "fast_log f16",
        ),
        &[-1.3867188, -0.6933594, 0.0, 0.6933594, 1.3867188],
        0.0,
        0.0,
        "fast_log f16",
    );
    assert_close_f32(
        &host_f16(
            &fast_sigmoid(&signed_f16).expect("fast_sigmoid f16 CUDA"),
            "fast_sigmoid f16",
        ),
        &[0.11920166, 0.3774414, 0.5, 0.6225586, 0.8808594],
        0.0,
        0.0,
        "fast_sigmoid f16",
    );
    assert_close_f32(
        &host_f16(
            &fast_tanh(&signed_f16).expect("fast_tanh f16 CUDA"),
            "fast_tanh f16",
        ),
        &[-0.9638672, -0.4621582, 0.0, 0.4621582, 0.9638672],
        0.0,
        0.0,
        "fast_tanh f16",
    );

    let signed_bf16 = cuda_bf16(&[-2.0, -0.5, 0.0, 0.5, 2.0], &[5]);
    let positive_bf16 = cuda_bf16(&[0.25, 0.5, 1.0, 2.0, 4.0], &[5]);
    assert_close_f32(
        &host_bf16(
            &fast_exp(&signed_bf16).expect("fast_exp bf16 CUDA"),
            "fast_exp bf16",
        ),
        &[0.13574219, 0.60546875, 1.0, 1.6484375, 7.375],
        0.0,
        0.0,
        "fast_exp bf16",
    );
    assert_close_f32(
        &host_bf16(
            &fast_log(&positive_bf16).expect("fast_log bf16 CUDA"),
            "fast_log bf16",
        ),
        &[-1.3828125, -0.69140625, 0.0, 0.69140625, 1.3828125],
        0.0,
        0.0,
        "fast_log bf16",
    );
    assert_close_f32(
        &host_bf16(
            &fast_sigmoid(&signed_bf16).expect("fast_sigmoid bf16 CUDA"),
            "fast_sigmoid bf16",
        ),
        &[0.119140625, 0.376_953_13, 0.5, 0.62109375, 0.87890625],
        0.0,
        0.0,
        "fast_sigmoid bf16",
    );
    assert_close_f32(
        &host_bf16(
            &fast_tanh(&signed_bf16).expect("fast_tanh bf16 CUDA"),
            "fast_tanh bf16",
        ),
        &[-0.96484375, -0.462_890_63, 0.0, 0.462_890_63, 0.96484375],
        0.0,
        0.0,
        "fast_tanh bf16",
    );
}

#[test]
fn direct_fast_unary_cuda_accepts_noncontiguous_views() {
    ensure_cuda_backend();

    let base = cuda_f32(&[0.25, 0.5, 1.0, 2.0, 4.0, 8.0], &[2, 3]);
    let view = base.transpose(0, 1).expect("transpose CUDA view");
    assert!(!view.is_contiguous(), "test setup must be non-contiguous");

    let out = fast_log(&view).expect("fast_log non-contiguous CUDA view");
    assert_eq!(out.device(), Device::Cuda(0), "view output device");
    assert_eq!(out.shape(), &[3, 2], "view output shape");
    assert_close_f32(
        &host_f32(&out, "fast_log non-contiguous view"),
        &[
            -2.0 * LN2_F32,
            LN2_F32,
            -LN2_F32,
            2.0 * LN2_F32,
            0.0,
            3.0 * LN2_F32,
        ],
        1e-6,
        1e-6,
        "fast_log non-contiguous view",
    );
}
