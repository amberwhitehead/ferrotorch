#![cfg(feature = "gpu")]

//! CUDA `atan2` parity for crosslink #2010.
//!
//! Live local PyTorch 2.11.0+cu130 oracle:
//! - CUDA supports f32/f64/f16/bf16 `torch.atan2` forward and autograd.
//! - f32 NaN outputs canonicalize to `0x7fffffff`; f64 preserves the first
//!   NaN operand payload/sign (`y` before `x`); f16/bf16 NaNs canonicalize to
//!   `0x7fff`.
//! - signed-zero quadrants produce `±0` or `±pi`; infinities produce exact
//!   quadrant constants; backward masks `(y, x) == (0, 0)` to zero.
//! - Results and gradients must stay CUDA-resident until explicit readback.

use std::sync::Once;

use ferrotorch_core::creation::from_vec;
use ferrotorch_core::device::Device;
use ferrotorch_core::grad_fns::reduction::sum;
use ferrotorch_core::grad_fns::transcendental::atan2;
use ferrotorch_core::tensor::Tensor;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for atan2 CUDA tests");
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
    assert_eq!(t.device(), Device::Cuda(0), "f32 tensor left CUDA");
    t.cpu()
        .expect("D2H f32")
        .data()
        .expect("f32 data")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn bits_f64(t: &Tensor<f64>) -> Vec<u64> {
    assert_eq!(t.device(), Device::Cuda(0), "f64 tensor left CUDA");
    t.cpu()
        .expect("D2H f64")
        .data()
        .expect("f64 data")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn bits_f16(t: &Tensor<half::f16>) -> Vec<u16> {
    assert_eq!(t.device(), Device::Cuda(0), "f16 tensor left CUDA");
    t.cpu()
        .expect("D2H f16")
        .data()
        .expect("f16 data")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn bits_bf16(t: &Tensor<half::bf16>) -> Vec<u16> {
    assert_eq!(t.device(), Device::Cuda(0), "bf16 tensor left CUDA");
    t.cpu()
        .expect("D2H bf16")
        .data()
        .expect("bf16 data")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn vec_f32(t: &Tensor<f32>) -> Vec<f32> {
    assert_eq!(t.device(), Device::Cuda(0), "f32 tensor left CUDA");
    t.cpu().expect("D2H f32").data_vec().expect("f32 data")
}

fn vec_f64(t: &Tensor<f64>) -> Vec<f64> {
    assert_eq!(t.device(), Device::Cuda(0), "f64 tensor left CUDA");
    t.cpu().expect("D2H f64").data_vec().expect("f64 data")
}

fn cpu_vec_f64(t: &Tensor<f64>) -> Vec<f64> {
    t.data_vec().expect("CPU f64 data")
}

fn assert_close_f32(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let allowed = 2e-5_f32 * e.abs().max(1.0);
        let diff = (a - e).abs();
        assert!(
            diff <= allowed,
            "{label}: index {i} diff {diff} exceeds {allowed}; actual={a} expected={e}"
        );
    }
}

fn assert_close_f64(actual: &[f64], expected: &[f64], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let allowed = 2e-10_f64 * e.abs().max(1.0);
        let diff = (a - e).abs();
        assert!(
            diff <= allowed,
            "{label}: index {i} diff {diff} exceeds {allowed}; actual={a} expected={e}"
        );
    }
}

#[test]
fn atan2_cuda_full_precision_matches_torch_edge_bits() {
    ensure_cuda_backend();

    let y = cuda(
        vec![
            f32::from_bits(0x0000_0000),
            f32::from_bits(0x8000_0000),
            f32::from_bits(0x0000_0000),
            f32::from_bits(0x8000_0000),
            1.0,
            -1.0,
            1.0,
            -1.0,
            f32::INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::from_bits(0x7FC1_2345),
            1.0,
        ],
        &[14],
        false,
    );
    let x = cuda(
        vec![
            1.0,
            1.0,
            -1.0,
            -1.0,
            0.0,
            0.0,
            -0.0,
            -0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
            1.0,
            f32::from_bits(0xFFC2_2222),
        ],
        &[14],
        false,
    );
    let out = atan2(&y, &x).expect("f32 CUDA atan2");
    assert_eq!(
        bits_f32(&out),
        vec![
            0x0000_0000,
            0x8000_0000,
            0x4049_0FDB,
            0xC049_0FDB,
            0x3FC9_0FDB,
            0xBFC9_0FDB,
            0x3FC9_0FDB,
            0xBFC9_0FDB,
            0x3F49_0FDB,
            0x4016_CBE4,
            0xBF49_0FDB,
            0xC016_CBE4,
            0x7FFF_FFFF,
            0x7FFF_FFFF,
        ]
    );

    let y = cuda(
        vec![
            f64::from_bits(0x0000_0000_0000_0000),
            f64::from_bits(0x8000_0000_0000_0000),
            f64::from_bits(0x0000_0000_0000_0000),
            f64::from_bits(0x8000_0000_0000_0000),
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
            f64::from_bits(0x7FF8_1234_5678_9ABC),
            1.0,
            f64::from_bits(0xFFF8_1234_5678_9ABC),
        ],
        &[11],
        false,
    );
    let x = cuda(
        vec![
            1.0,
            1.0,
            -1.0,
            -1.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            1.0,
            f64::from_bits(0x7FF8_2222_2222_2222),
            f64::from_bits(0x7FF8_2222_2222_2222),
        ],
        &[11],
        false,
    );
    let out = atan2(&y, &x).expect("f64 CUDA atan2");
    assert_eq!(
        bits_f64(&out),
        vec![
            0x0000_0000_0000_0000,
            0x8000_0000_0000_0000,
            0x4009_21FB_5444_2D18,
            0xC009_21FB_5444_2D18,
            0x3FE9_21FB_5444_2D18,
            0x4002_D97C_7F33_21D2,
            0xBFE9_21FB_5444_2D18,
            0xC002_D97C_7F33_21D2,
            0x7FF8_1234_5678_9ABC,
            0x7FF8_2222_2222_2222,
            0xFFF8_1234_5678_9ABC,
        ]
    );
}

#[test]
fn atan2_cuda_reduced_precision_matches_torch_edge_bits() {
    ensure_cuda_backend();

    let y = cuda(
        vec![
            half::f16::from_bits(0x0000),
            half::f16::from_bits(0x8000),
            half::f16::from_f32(0.0),
            half::f16::from_f32(-0.0),
            half::f16::from_f32(1.0),
            half::f16::from_f32(-1.0),
            half::f16::from_f32(f32::INFINITY),
            half::f16::from_f32(f32::INFINITY),
            half::f16::from_f32(f32::NEG_INFINITY),
            half::f16::from_f32(f32::NEG_INFINITY),
            half::f16::from_bits(0x7E01),
        ],
        &[11],
        false,
    );
    let x = cuda(
        vec![
            half::f16::from_f32(1.0),
            half::f16::from_f32(1.0),
            half::f16::from_f32(-1.0),
            half::f16::from_f32(-1.0),
            half::f16::from_f32(-0.0),
            half::f16::from_f32(-0.0),
            half::f16::from_f32(f32::INFINITY),
            half::f16::from_f32(f32::NEG_INFINITY),
            half::f16::from_f32(f32::INFINITY),
            half::f16::from_f32(f32::NEG_INFINITY),
            half::f16::from_f32(1.0),
        ],
        &[11],
        false,
    );
    let out = atan2(&y, &x).expect("f16 CUDA atan2");
    assert_eq!(
        bits_f16(&out),
        vec![
            0x0000, 0x8000, 0x4248, 0xC248, 0x3E48, 0xBE48, 0x3A48, 0x40B6, 0xBA48, 0xC0B6, 0x7FFF,
        ]
    );

    let y = cuda(
        vec![
            half::bf16::from_bits(0x0000),
            half::bf16::from_bits(0x8000),
            half::bf16::from_f32(0.0),
            half::bf16::from_f32(-0.0),
            half::bf16::from_f32(1.0),
            half::bf16::from_f32(-1.0),
            half::bf16::from_f32(f32::INFINITY),
            half::bf16::from_f32(f32::INFINITY),
            half::bf16::from_f32(f32::NEG_INFINITY),
            half::bf16::from_f32(f32::NEG_INFINITY),
            half::bf16::from_bits(0x7FC1),
        ],
        &[11],
        false,
    );
    let x = cuda(
        vec![
            half::bf16::from_f32(1.0),
            half::bf16::from_f32(1.0),
            half::bf16::from_f32(-1.0),
            half::bf16::from_f32(-1.0),
            half::bf16::from_f32(-0.0),
            half::bf16::from_f32(-0.0),
            half::bf16::from_f32(f32::INFINITY),
            half::bf16::from_f32(f32::NEG_INFINITY),
            half::bf16::from_f32(f32::INFINITY),
            half::bf16::from_f32(f32::NEG_INFINITY),
            half::bf16::from_f32(1.0),
        ],
        &[11],
        false,
    );
    let out = atan2(&y, &x).expect("bf16 CUDA atan2");
    assert_eq!(
        bits_bf16(&out),
        vec![
            0x0000, 0x8000, 0x4049, 0xC049, 0x3FC9, 0xBFC9, 0x3F49, 0x4017, 0xBF49, 0xC017, 0x7FFF,
        ]
    );
}

#[test]
fn atan2_cuda_backward_broadcasts_and_masks_origin() {
    ensure_cuda_backend();

    let y = cuda(vec![0.0_f32, 3.0], &[2, 1], true);
    let x = cuda(vec![0.0_f32, 4.0, -4.0], &[1, 3], true);
    let out = atan2(&y, &x).expect("f32 broadcast atan2");
    assert_eq!(out.shape(), &[2, 3]);
    assert_eq!(out.device(), Device::Cuda(0));
    sum(&out).expect("sum").backward().expect("backward");
    let gy = y.grad().expect("y grad slot").expect("y grad");
    let gx = x.grad().expect("x grad slot").expect("x grad");
    assert_eq!(gy.device(), Device::Cuda(0));
    assert_eq!(gx.device(), Device::Cuda(0));
    assert_close_f32(&vec_f32(&gy), &[0.0, 0.0], "f32 grad_y");
    assert_close_f32(&vec_f32(&gx), &[-0.333_333_34, -0.12, -0.12], "f32 grad_x");

    let y = cuda(vec![0.0_f64, 3.0], &[2, 1], true);
    let x = cuda(vec![0.0_f64, 4.0, -4.0], &[1, 3], true);
    let out = atan2(&y, &x).expect("f64 broadcast atan2");
    assert_eq!(out.shape(), &[2, 3]);
    assert_eq!(out.device(), Device::Cuda(0));
    sum(&out).expect("sum").backward().expect("backward");
    let gy = y.grad().expect("y grad slot").expect("y grad");
    let gx = x.grad().expect("x grad slot").expect("x grad");
    assert_eq!(gy.device(), Device::Cuda(0));
    assert_eq!(gx.device(), Device::Cuda(0));
    assert_close_f64(&vec_f64(&gy), &[0.0, 0.0], "f64 grad_y");
    assert_close_f64(&vec_f64(&gx), &[-1.0 / 3.0, -0.12, -0.12], "f64 grad_x");
}

#[test]
fn atan2_cuda_reduced_precision_backward_matches_torch_bits() {
    ensure_cuda_backend();

    let y = cuda(
        vec![half::f16::from_f32(0.0), half::f16::from_f32(3.0)],
        &[2, 1],
        true,
    );
    let x = cuda(
        vec![
            half::f16::from_f32(0.0),
            half::f16::from_f32(4.0),
            half::f16::from_f32(-4.0),
        ],
        &[1, 3],
        true,
    );
    sum(&atan2(&y, &x).expect("f16 atan2"))
        .expect("sum")
        .backward()
        .expect("f16 backward");
    let gy = y.grad().expect("f16 y grad slot").expect("f16 y grad");
    let gx = x.grad().expect("f16 x grad slot").expect("f16 x grad");
    assert_eq!(bits_f16(&gy), vec![0x0000, 0x0000]);
    assert_eq!(bits_f16(&gx), vec![0xB555, 0xAFAE, 0xAFAE]);

    let y = cuda(
        vec![half::bf16::from_f32(0.0), half::bf16::from_f32(3.0)],
        &[2, 1],
        true,
    );
    let x = cuda(
        vec![
            half::bf16::from_f32(0.0),
            half::bf16::from_f32(4.0),
            half::bf16::from_f32(-4.0),
        ],
        &[1, 3],
        true,
    );
    sum(&atan2(&y, &x).expect("bf16 atan2"))
        .expect("sum")
        .backward()
        .expect("bf16 backward");
    let gy = y.grad().expect("bf16 y grad slot").expect("bf16 y grad");
    let gx = x.grad().expect("bf16 x grad slot").expect("bf16 x grad");
    assert_eq!(bits_bf16(&gy), vec![0x0000, 0x0000]);
    assert_eq!(bits_bf16(&gx), vec![0xBEAB, 0xBDF6, 0xBDF6]);
}

#[test]
fn atan2_cuda_noncontiguous_broadcast_forward_backward_matches_cpu_reference() {
    ensure_cuda_backend();

    let base_cpu = from_vec::<f64>(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .expect("cpu base")
        .requires_grad_(true);
    let x_cpu = from_vec::<f64>(vec![1.0, -2.0, 0.5], &[3, 1])
        .expect("cpu x")
        .requires_grad_(true);
    let y_cpu = base_cpu.transpose(0, 1).expect("cpu transpose view");
    let out_cpu = atan2(&y_cpu, &x_cpu).expect("cpu atan2 view");
    let out_expected = cpu_vec_f64(&out_cpu);
    sum(&out_cpu)
        .expect("cpu sum")
        .backward()
        .expect("cpu backward");
    let base_grad_expected =
        cpu_vec_f64(&base_cpu.grad().expect("base grad slot").expect("base grad"));
    let x_grad_expected = cpu_vec_f64(&x_cpu.grad().expect("x grad slot").expect("x grad"));

    let base_gpu = from_vec::<f64>(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .expect("gpu base source")
        .to(Device::Cuda(0))
        .expect("upload base")
        .requires_grad_(true);
    let x_gpu = from_vec::<f64>(vec![1.0, -2.0, 0.5], &[3, 1])
        .expect("gpu x source")
        .to(Device::Cuda(0))
        .expect("upload x")
        .requires_grad_(true);
    let y_gpu = base_gpu.transpose(0, 1).expect("gpu transpose view");
    assert!(
        !y_gpu.is_contiguous(),
        "test must exercise non-contiguous CUDA y"
    );
    let out_gpu = atan2(&y_gpu, &x_gpu).expect("gpu atan2 view");
    assert_eq!(out_gpu.device(), Device::Cuda(0));
    assert_close_f64(&vec_f64(&out_gpu), &out_expected, "noncontig fwd");
    sum(&out_gpu)
        .expect("gpu sum")
        .backward()
        .expect("gpu backward");
    let base_grad = base_gpu
        .grad()
        .expect("gpu base grad slot")
        .expect("gpu base grad");
    let x_grad = x_gpu.grad().expect("gpu x grad slot").expect("gpu x grad");
    assert_eq!(base_grad.device(), Device::Cuda(0));
    assert_eq!(x_grad.device(), Device::Cuda(0));
    assert_close_f64(&vec_f64(&base_grad), &base_grad_expected, "base grad");
    assert_close_f64(&vec_f64(&x_grad), &x_grad_expected, "x grad");
}
