#![cfg(feature = "gpu")]

//! CUDA `signbit` parity.
//!
//! Live PyTorch 2.11.0+cu130 oracle on this machine:
//! - f32/f64 CUDA inspect raw sign bits, including negative NaN -> true.
//! - bf16 CUDA inspects raw sign bits, including negative NaN -> true.
//! - f16 CUDA returns false for every NaN payload, including raw negative NaN,
//!   while still returning true for `-0.0`, finite negatives, and `-inf`.

use std::sync::Once;

use ferrotorch_core::bool_tensor::BoolTensor;
use ferrotorch_core::creation::from_vec;
use ferrotorch_core::device::Device;
use ferrotorch_core::grad_fns::transcendental::signbit;
use ferrotorch_core::tensor::Tensor;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for signbit tests");
    });
}

fn bools(mask: &BoolTensor) -> Vec<bool> {
    assert_eq!(mask.device(), Device::Cuda(0), "mask must stay on CUDA");
    mask.to(Device::Cpu)
        .expect("bool D2H")
        .data()
        .expect("bool data")
        .to_vec()
}

fn cuda<T: ferrotorch_core::dtype::Float>(values: Vec<T>, shape: &[usize]) -> Tensor<T> {
    from_vec::<T>(values, shape)
        .expect("CPU tensor")
        .to(Device::Cuda(0))
        .expect("upload tensor")
}

#[test]
fn signbit_cuda_matches_torch_for_f32_f64_and_reduced_precision() {
    ensure_cuda_backend();

    let f32_values = vec![
        -0.0_f32,
        0.0,
        -1.0,
        1.0,
        f32::NEG_INFINITY,
        f32::INFINITY,
        f32::from_bits(0x7FC0_0000),
        f32::from_bits(0xFFC0_0000),
    ];
    let expected_full = [true, false, true, false, true, false, false, true];
    let got = signbit(&cuda(f32_values, &[8])).expect("signbit f32 CUDA");
    assert_eq!(got.shape(), &[8]);
    assert_eq!(bools(&got), expected_full);

    let f64_values = vec![
        -0.0_f64,
        0.0,
        -1.0,
        1.0,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::from_bits(0x7FF8_0000_0000_0000),
        f64::from_bits(0xFFF8_0000_0000_0000),
    ];
    let got = signbit(&cuda(f64_values, &[8])).expect("signbit f64 CUDA");
    assert_eq!(bools(&got), expected_full);

    let f16_values = vec![
        half::f16::from_bits(0x8000),
        half::f16::from_bits(0x0000),
        half::f16::from_bits(0xBC00),
        half::f16::from_bits(0x3C00),
        half::f16::from_bits(0xFC00),
        half::f16::from_bits(0x7C00),
        half::f16::from_bits(0x7E00),
        half::f16::from_bits(0xFE00),
    ];
    let got = signbit(&cuda(f16_values, &[8])).expect("signbit f16 CUDA");
    assert_eq!(
        bools(&got),
        [true, false, true, false, true, false, false, false]
    );

    let bf16_values = vec![
        half::bf16::from_bits(0x8000),
        half::bf16::from_bits(0x0000),
        half::bf16::from_bits(0xBF80),
        half::bf16::from_bits(0x3F80),
        half::bf16::from_bits(0xFF80),
        half::bf16::from_bits(0x7F80),
        half::bf16::from_bits(0x7FC0),
        half::bf16::from_bits(0xFFC0),
    ];
    let got = signbit(&cuda(bf16_values, &[8])).expect("signbit bf16 CUDA");
    assert_eq!(bools(&got), expected_full);
}

#[test]
fn signbit_cuda_preserves_shape_empty_scalar_and_logical_view_order() {
    ensure_cuda_backend();

    let empty = cuda::<half::f16>(vec![], &[0]);
    let got = signbit(&empty).expect("signbit empty f16 CUDA");
    assert_eq!(got.shape(), &[0]);
    assert_eq!(bools(&got), Vec::<bool>::new());

    let scalar = cuda::<f32>(vec![-0.0], &[]);
    let got = signbit(&scalar).expect("signbit scalar CUDA");
    assert_eq!(got.shape(), &[]);
    assert_eq!(bools(&got), [true]);

    let base = cuda::<f32>(
        vec![
            0.0,
            -1.0,
            -0.0,
            2.0,
            f32::NEG_INFINITY,
            f32::from_bits(0xFFC0_0000),
        ],
        &[2, 3],
    );
    let view = base.transpose(0, 1).expect("transpose view");
    let got = signbit(&view).expect("signbit transposed f32 CUDA");
    assert_eq!(got.shape(), &[3, 2]);
    assert_eq!(bools(&got), [false, false, true, true, true, true]);
}

#[test]
fn signbit_cpu_uses_logical_view_order() {
    let base = from_vec::<f32>(vec![0.0, -1.0, -0.0, 2.0, f32::NEG_INFINITY, 3.0], &[2, 3])
        .expect("CPU tensor");
    let view = base.transpose(0, 1).expect("transpose view");

    let got = signbit(&view).expect("signbit transposed f32 CPU");
    assert_eq!(got.device(), Device::Cpu);
    assert_eq!(got.shape(), &[3, 2]);
    assert_eq!(
        got.data().expect("CPU bool data"),
        [false, false, true, true, true, false]
    );
}
