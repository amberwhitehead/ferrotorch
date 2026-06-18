#![cfg(feature = "gpu")]

//! CUDA `copysign` parity.
//!
//! Live PyTorch 2.11.0+cu130 oracle on this machine:
//! - f32/f64 copy the raw sign bit and preserve magnitude NaN payloads.
//! - f16 CUDA canonicalizes NaN magnitude outputs to positive `0x7fff` and
//!   ignores NaN sign operands for finite magnitudes.
//! - bf16 CUDA canonicalizes NaN magnitude outputs to positive `0x7fff` and
//!   honors the raw sign bit for finite magnitudes, including NaN sign operands.
//! - backward is magnitude-only; zero magnitudes get zero gradient.

use std::sync::Once;

use ferrotorch_core::creation::from_vec;
use ferrotorch_core::device::Device;
use ferrotorch_core::grad_fns::reduction::sum;
use ferrotorch_core::grad_fns::transcendental::copysign;
use ferrotorch_core::tensor::Tensor;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for copysign CUDA tests");
    });
}

fn cuda<T: ferrotorch_core::dtype::Float>(
    values: Vec<T>,
    shape: &[usize],
    requires_grad: bool,
) -> Tensor<T> {
    from_vec::<T>(values, shape)
        .expect("CPU tensor")
        .to(Device::Cuda(0))
        .expect("upload tensor")
        .requires_grad_(requires_grad)
}

fn bits_f32(t: &Tensor<f32>) -> Vec<u32> {
    assert_eq!(t.device(), Device::Cuda(0), "f32 tensor must stay CUDA");
    t.cpu()
        .expect("D2H f32")
        .data()
        .expect("f32 data")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn bits_f64(t: &Tensor<f64>) -> Vec<u64> {
    assert_eq!(t.device(), Device::Cuda(0), "f64 tensor must stay CUDA");
    t.cpu()
        .expect("D2H f64")
        .data()
        .expect("f64 data")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn bits_f16(t: &Tensor<half::f16>) -> Vec<u16> {
    assert_eq!(t.device(), Device::Cuda(0), "f16 tensor must stay CUDA");
    t.cpu()
        .expect("D2H f16")
        .data()
        .expect("f16 data")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn bits_bf16(t: &Tensor<half::bf16>) -> Vec<u16> {
    assert_eq!(t.device(), Device::Cuda(0), "bf16 tensor must stay CUDA");
    t.cpu()
        .expect("D2H bf16")
        .data()
        .expect("bf16 data")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

#[test]
fn copysign_cuda_full_precision_matches_raw_torch_bits() {
    ensure_cuda_backend();

    let mag = cuda(
        vec![
            f32::from_bits(0x4040_0000),
            f32::from_bits(0xC040_0000),
            f32::from_bits(0x0000_0000),
            f32::from_bits(0x8000_0000),
            f32::from_bits(0x7FC1_2345),
            f32::from_bits(0xFFC5_4321),
        ],
        &[2, 3],
        false,
    );
    let sign = cuda(
        vec![
            f32::from_bits(0xBF80_0000),
            f32::from_bits(0x3F80_0000),
            f32::from_bits(0x8000_0000),
            f32::from_bits(0x0000_0000),
            f32::from_bits(0x7FC1_1111),
            f32::from_bits(0xFFC2_2222),
        ],
        &[2, 3],
        false,
    );
    let out = copysign(&mag, &sign).expect("f32 CUDA copysign");
    assert_eq!(
        bits_f32(&out),
        vec![
            0xC040_0000,
            0x4040_0000,
            0x8000_0000,
            0x0000_0000,
            0x7FC1_2345,
            0xFFC5_4321,
        ]
    );

    let mag = cuda(
        vec![
            f64::from_bits(0x4008_0000_0000_0000),
            f64::from_bits(0xC008_0000_0000_0000),
            f64::from_bits(0x0000_0000_0000_0000),
            f64::from_bits(0x8000_0000_0000_0000),
            f64::from_bits(0x7FF8_1234_5678_9ABC),
            f64::from_bits(0xFFF8_5432_1234_5678),
        ],
        &[6],
        false,
    );
    let sign = cuda(
        vec![
            f64::from_bits(0xBFF0_0000_0000_0000),
            f64::from_bits(0x3FF0_0000_0000_0000),
            f64::from_bits(0x8000_0000_0000_0000),
            f64::from_bits(0x0000_0000_0000_0000),
            f64::from_bits(0x7FF8_1111_1111_1111),
            f64::from_bits(0xFFF8_2222_2222_2222),
        ],
        &[6],
        false,
    );
    let out = copysign(&mag, &sign).expect("f64 CUDA copysign");
    assert_eq!(
        bits_f64(&out),
        vec![
            0xC008_0000_0000_0000,
            0x4008_0000_0000_0000,
            0x8000_0000_0000_0000,
            0x0000_0000_0000_0000,
            0x7FF8_1234_5678_9ABC,
            0xFFF8_5432_1234_5678,
        ]
    );
}

#[test]
fn copysign_cuda_reduced_precision_matches_torch_nan_rules() {
    ensure_cuda_backend();

    let mag = cuda(
        vec![
            half::f16::from_bits(0x4200),
            half::f16::from_bits(0xC200),
            half::f16::from_bits(0x0000),
            half::f16::from_bits(0x8000),
            half::f16::from_bits(0x7E01),
            half::f16::from_bits(0xFE01),
            half::f16::from_bits(0x4200),
            half::f16::from_bits(0x4200),
        ],
        &[8],
        false,
    );
    let sign = cuda(
        vec![
            half::f16::from_bits(0xBC00),
            half::f16::from_bits(0x3C00),
            half::f16::from_bits(0x8000),
            half::f16::from_bits(0x0000),
            half::f16::from_bits(0x7E11),
            half::f16::from_bits(0xFE22),
            half::f16::from_bits(0x7E11),
            half::f16::from_bits(0xFE22),
        ],
        &[8],
        false,
    );
    let out = copysign(&mag, &sign).expect("f16 CUDA copysign");
    assert_eq!(
        bits_f16(&out),
        vec![
            0xC200, 0x4200, 0x8000, 0x0000, 0x7FFF, 0x7FFF, 0x4200, 0x4200
        ]
    );

    let mag = cuda(
        vec![
            half::bf16::from_bits(0x4040),
            half::bf16::from_bits(0xC040),
            half::bf16::from_bits(0x0000),
            half::bf16::from_bits(0x8000),
            half::bf16::from_bits(0x7FC1),
            half::bf16::from_bits(0xFFC5),
            half::bf16::from_bits(0x4040),
            half::bf16::from_bits(0x4040),
        ],
        &[8],
        false,
    );
    let sign = cuda(
        vec![
            half::bf16::from_bits(0xBF80),
            half::bf16::from_bits(0x3F80),
            half::bf16::from_bits(0x8000),
            half::bf16::from_bits(0x0000),
            half::bf16::from_bits(0x7FC1),
            half::bf16::from_bits(0xFFC2),
            half::bf16::from_bits(0x7FC1),
            half::bf16::from_bits(0xFFC2),
        ],
        &[8],
        false,
    );
    let out = copysign(&mag, &sign).expect("bf16 CUDA copysign");
    assert_eq!(
        bits_bf16(&out),
        vec![
            0xC040, 0x4040, 0x8000, 0x0000, 0x7FFF, 0x7FFF, 0x4040, 0xC040
        ]
    );
}

#[test]
fn copysign_cuda_backward_broadcasts_and_keeps_grads_on_device() {
    ensure_cuda_backend();

    let mag = cuda(vec![3.0_f32, -3.0], &[2, 1], true);
    let sign = cuda(vec![-1.0_f32, 1.0, -0.0], &[1, 3], true);
    let out = copysign(&mag, &sign).expect("f32 broadcast copysign");
    assert_eq!(out.shape(), &[2, 3]);
    assert_eq!(
        bits_f32(&out),
        vec![
            (-3.0_f32).to_bits(),
            3.0_f32.to_bits(),
            (-3.0_f32).to_bits(),
            (-3.0_f32).to_bits(),
            3.0_f32.to_bits(),
            (-3.0_f32).to_bits(),
        ]
    );

    sum(&out).expect("sum").backward().expect("backward");
    let mag_grad = mag
        .grad()
        .expect("mag grad slot")
        .expect("mag grad must exist");
    let sign_grad = sign
        .grad()
        .expect("sign grad slot")
        .expect("sign grad must exist");
    assert_eq!(mag_grad.device(), Device::Cuda(0));
    assert_eq!(sign_grad.device(), Device::Cuda(0));
    assert_eq!(mag_grad.shape(), &[2, 1]);
    assert_eq!(sign_grad.shape(), &[1, 3]);
    assert_eq!(
        bits_f32(&mag_grad),
        vec![(-1.0_f32).to_bits(), 1.0_f32.to_bits()]
    );
    assert_eq!(bits_f32(&sign_grad), vec![0.0_f32.to_bits(); 3]);
}

#[test]
fn copysign_cuda_half_backward_nan_and_zero_rules_match_torch() {
    ensure_cuda_backend();

    let mag = cuda(
        vec![
            half::f16::from_bits(0x7E01),
            half::f16::from_bits(0xFE01),
            half::f16::from_bits(0x3C00),
            half::f16::from_bits(0x3C00),
            half::f16::from_bits(0x0000),
            half::f16::from_bits(0x8000),
        ],
        &[6],
        true,
    );
    let sign = cuda(
        vec![
            half::f16::from_bits(0x3C00),
            half::f16::from_bits(0x3C00),
            half::f16::from_bits(0xFE22),
            half::f16::from_bits(0x8000),
            half::f16::from_bits(0xBC00),
            half::f16::from_bits(0x3C00),
        ],
        &[6],
        true,
    );
    let out = copysign(&mag, &sign).expect("f16 copysign backward fixture");
    assert_eq!(
        bits_f16(&out),
        vec![0x7FFF, 0x7FFF, 0x3C00, 0xBC00, 0x8000, 0x0000]
    );
    sum(&out)
        .expect("sum f16")
        .backward()
        .expect("backward f16");
    let grad = mag
        .grad()
        .expect("f16 grad slot")
        .expect("f16 grad must exist");
    assert_eq!(grad.device(), Device::Cuda(0));
    assert_eq!(
        bits_f16(&grad),
        vec![0x7FFF, 0x7FFF, 0x3C00, 0xBC00, 0x0000, 0x0000]
    );

    let mag = cuda(
        vec![
            half::bf16::from_bits(0x7FC1),
            half::bf16::from_bits(0xFFC1),
            half::bf16::from_bits(0x3F80),
            half::bf16::from_bits(0x3F80),
            half::bf16::from_bits(0x0000),
            half::bf16::from_bits(0x8000),
        ],
        &[6],
        true,
    );
    let sign = cuda(
        vec![
            half::bf16::from_bits(0x3F80),
            half::bf16::from_bits(0x3F80),
            half::bf16::from_bits(0xFFC2),
            half::bf16::from_bits(0x8000),
            half::bf16::from_bits(0xBF80),
            half::bf16::from_bits(0x3F80),
        ],
        &[6],
        true,
    );
    let out = copysign(&mag, &sign).expect("bf16 copysign backward fixture");
    assert_eq!(
        bits_bf16(&out),
        vec![0x7FFF, 0x7FFF, 0xBF80, 0xBF80, 0x8000, 0x0000]
    );
    sum(&out)
        .expect("sum bf16")
        .backward()
        .expect("backward bf16");
    let grad = mag
        .grad()
        .expect("bf16 grad slot")
        .expect("bf16 grad must exist");
    assert_eq!(grad.device(), Device::Cuda(0));
    assert_eq!(
        bits_bf16(&grad),
        vec![0x7FFF, 0x7FFF, 0xBF80, 0xBF80, 0x0000, 0x0000]
    );
}

#[test]
fn copysign_cuda_accepts_empty_scalar_and_noncontiguous_views() {
    ensure_cuda_backend();

    let empty = cuda::<f32>(vec![], &[0], false);
    let out = copysign(&empty, &empty).expect("empty copysign CUDA");
    assert_eq!(out.shape(), &[0]);
    assert_eq!(bits_f32(&out), Vec::<u32>::new());

    let scalar_mag = cuda::<f32>(vec![-3.0], &[], false);
    let scalar_sign = cuda::<f32>(vec![-0.0], &[], false);
    let out = copysign(&scalar_mag, &scalar_sign).expect("scalar copysign CUDA");
    assert_eq!(out.shape(), &[]);
    assert_eq!(bits_f32(&out), vec![(-3.0_f32).to_bits()]);

    let base = cuda::<f32>(vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0], &[2, 3], false);
    let mag_view = base.transpose(0, 1).expect("transpose CUDA view");
    let sign = cuda::<f32>(vec![-1.0], &[], false);
    let out = copysign(&mag_view, &sign).expect("non-contiguous magnitude copysign");
    assert_eq!(out.shape(), &[3, 2]);
    assert_eq!(
        bits_f32(&out),
        vec![
            (-1.0_f32).to_bits(),
            (-4.0_f32).to_bits(),
            (-2.0_f32).to_bits(),
            (-5.0_f32).to_bits(),
            (-3.0_f32).to_bits(),
            (-6.0_f32).to_bits(),
        ]
    );

    let base = cuda::<f32>(vec![1.0, 2.0, 3.0, 4.0], &[4], false);
    let offset_view = base
        .as_strided(&[2], &[2], Some(1))
        .expect("positive-stride offset CUDA view");
    let sign = cuda::<f32>(vec![-1.0], &[], false);
    let out = copysign(&offset_view, &sign).expect("offset-strided copysign");
    assert_eq!(
        bits_f32(&out),
        vec![(-2.0_f32).to_bits(), (-4.0_f32).to_bits()]
    );

    let base = cuda::<f32>(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], false);
    let rank9 = base
        .as_strided(
            &[2, 1, 1, 1, 1, 1, 1, 1, 2],
            &[1, 0, 0, 0, 0, 0, 0, 0, 2],
            Some(0),
        )
        .expect("rank-9 CUDA view");
    let sign = cuda::<f32>(vec![-1.0], &[], false);
    let out = copysign(&rank9, &sign).expect("rank-9 non-contiguous copysign");
    assert_eq!(out.shape(), &[2, 1, 1, 1, 1, 1, 1, 1, 2]);
    assert_eq!(
        bits_f32(&out),
        vec![
            (-1.0_f32).to_bits(),
            (-3.0_f32).to_bits(),
            (-2.0_f32).to_bits(),
            (-4.0_f32).to_bits(),
        ]
    );
}
