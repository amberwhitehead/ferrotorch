//! CORE-041 (#1735, CLASS-S High) regression battery: differentiable
//! fake-quantization (`fake_quantize_per_tensor_affine`, the tensor-qparams
//! overload, and `fake_quantize_per_channel_affine`) must
//!
//!   1. preserve device residency — a CUDA input yields a CUDA output
//!      (documented host round trip + re-upload is acceptable, R-LOUD-2);
//!   2. deliver the STE backward gradient on the saved input's device
//!      (R-ORACLE-3);
//!   3. keep the QAT values bit-identical to live torch on the way through.
//!
//! Pre-fix observed behavior (R-AHON-1 probe at HEAD, pasted in #1735):
//! both forwards read CUDA inputs through `data_vec` (host download) and
//! unconditionally constructed `TensorStorage::cpu` outputs — `is_cuda()`
//! was false on every output; both backward nodes likewise returned CPU
//! gradients for CUDA leaves.
//!
//! All numerical expectations are pasted from a LIVE `torch==2.11.0+cu130`
//! session on the same device class (RTX 3090) — snippets quoted per test
//! (R-ORACLE-1(b)). Comparisons are exact (`==`): every sample uses a
//! dyadic scale (0.25 / 0.5 / 1.0) or a value whose dequantized result is
//! the f32 cast of an exact f64 product (e.g. `-128 * 0.01` →
//! `-1.2799999713897705`), so the expected bit patterns are reachable
//! without rounding ambiguity in both ferrotorch and torch.

#![cfg(feature = "gpu")]

use ferrotorch_core::autograd::graph::backward_with_grad;
use ferrotorch_core::grad_fns::quantize_grad::{
    fake_quantize_per_channel_affine, fake_quantize_per_tensor_affine,
    fake_quantize_per_tensor_affine_tensor_qparams,
};
use ferrotorch_core::{Device, FerrotorchError, Float, IntTensor, Tensor, TensorStorage};
use std::sync::Once;

static GPU_INIT: Once = Once::new();
fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for the CORE-041 GPU pins");
    });
}

/// Build a CUDA-resident float tensor from f64 literals (exactly
/// representable in both dtypes for every value used in this suite).
fn t_cuda<T: Float>(data: &[f64], shape: &[usize], rg: bool) -> Tensor<T> {
    let cast: Vec<T> = data
        .iter()
        .map(|&v| <T as num_traits::NumCast>::from(v).unwrap())
        .collect();
    Tensor::from_storage(TensorStorage::cpu(cast), shape.to_vec(), false)
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap()
        .requires_grad_(rg)
}

fn t_cpu<T: Float>(data: &[f64], shape: &[usize], rg: bool) -> Tensor<T> {
    let cast: Vec<T> = data
        .iter()
        .map(|&v| <T as num_traits::NumCast>::from(v).unwrap())
        .collect();
    Tensor::from_storage(TensorStorage::cpu(cast), shape.to_vec(), rg).unwrap()
}

fn zp_cpu(data: &[i64], shape: &[usize]) -> IntTensor<i64> {
    IntTensor::from_vec(data.to_vec(), shape.to_vec()).unwrap()
}

fn zp_cuda(data: &[i64], shape: &[usize]) -> IntTensor<i64> {
    zp_cpu(data, shape).to(Device::Cuda(0)).unwrap()
}

/// Read any-device tensor back as exact f64 values for oracle comparison.
fn host_f64<T: Float>(t: &Tensor<T>) -> Vec<f64> {
    t.data_vec()
        .unwrap()
        .into_iter()
        .map(|v| <f64 as num_traits::NumCast>::from(v).unwrap())
        .collect()
}

fn assert_cuda_with<T: Float>(t: &Tensor<T>, what: &str, expected: &[f64]) {
    assert!(
        t.is_cuda(),
        "{what} must be CUDA-resident (CORE-041: silent CPU demotion), got {:?}",
        t.device()
    );
    assert_eq!(host_f64(t), expected, "{what} values vs live torch oracle");
}

fn grad_of<T: Float>(leaf: &Tensor<T>) -> Tensor<T> {
    leaf.grad()
        .unwrap()
        .expect("grad must reach the leaf (R-ORACLE-3)")
}

fn assert_device_mismatch<T>(
    result: Result<T, FerrotorchError>,
    expected: Device,
    got: Device,
    label: &str,
) {
    match result {
        Err(FerrotorchError::DeviceMismatch {
            expected: e,
            got: g,
        }) => {
            assert_eq!(e, expected, "{label}: expected-device field");
            assert_eq!(g, got, "{label}: got-device field");
        }
        Err(other) => panic!("{label}: expected DeviceMismatch, got {other:?}"),
        Ok(_) => panic!("{label}: expected DeviceMismatch, got Ok"),
    }
}

// ---------------------------------------------------------------------------
// per-tensor, scalar qparams
// ---------------------------------------------------------------------------

/// Live torch 2.11.0+cu130 (RTX 3090), f32 AND f64 (torch's kernel computes
/// the dequant tail at f32 for both dtypes; every expected value here is a
/// dyadic rational exactly representable in f32):
///   x = torch.tensor([-50., -1., .1, 1., 5., 100.], device='cuda',
///                    requires_grad=True)            # (dtype f32 / f64)
///   out = torch.fake_quantize_per_tensor_affine(x, 0.25, 0, -128, 127)
///   out.backward(torch.tensor([1.,2.,3.,4.,5.,6.], device='cuda'))
///   -> out  [-32.0, -1.0, 0.0, 1.0, 5.0, 31.75]  on cuda:0
///   -> grad [0.0, 2.0, 3.0, 4.0, 5.0, 0.0]       on cuda:0
///   (-50 → q=-200 clamps to -128, mask 0; 100 → q=400 clamps to 127,
///    mask 0; 0.1 → round_ties_even(0.4)=0 → 0.0, mask 1.)
fn per_tensor_cuda_resident<T: Float>() {
    ensure_cuda_backend();
    let x = t_cuda::<T>(&[-50.0, -1.0, 0.1, 1.0, 5.0, 100.0], &[6], true);
    let out = fake_quantize_per_tensor_affine(&x, 0.25, 0, -128, 127).unwrap();
    assert_cuda_with(
        &out,
        "fake_quantize_per_tensor_affine output",
        &[-32.0, -1.0, 0.0, 1.0, 5.0, 31.75],
    );

    let seed = t_cuda::<T>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6], false);
    backward_with_grad(&out, Some(&seed)).unwrap();
    let gx = grad_of(&x);
    assert_cuda_with(
        &gx,
        "fake_quantize_per_tensor_affine grad_input",
        &[0.0, 2.0, 3.0, 4.0, 5.0, 0.0],
    );
}

#[test]
fn fake_quantize_per_tensor_cuda_resident_f32() {
    per_tensor_cuda_resident::<f32>();
}

#[test]
fn fake_quantize_per_tensor_cuda_resident_f64() {
    per_tensor_cuda_resident::<f64>();
}

/// Non-dyadic scale + non-zero zero-point, f32. Live torch 2.11.0+cu130:
///   x = torch.tensor([-2., -1., 0., .25, 1., 23., 30.], device='cuda',
///                    requires_grad=True)
///   out = torch.fake_quantize_per_tensor_affine(x, 0.1, 10, 0, 255)
///   out.backward(torch.tensor([1.,2.,3.,4.,5.,6.,7.], device='cuda'))
///   -> out  [-1.0, -1.0, 0.0, 0.20000000298023224, 1.0, 23.0, 24.5] cuda:0
///   -> grad [0., 2., 3., 4., 5., 6., 0.] cuda:0
///   (-2 → q=10-20=-10 clamps to 0 → (0-10)*0.1=-1.0, mask 0;
///    30 → q=10+300=310 clamps to 255 → 24.5, mask 0.)
/// `0.20000000298023224` is the exact f64 widening of f32 `0.2`.
#[test]
fn fake_quantize_per_tensor_asymmetric_cuda_resident_f32() {
    ensure_cuda_backend();
    let x = t_cuda::<f32>(&[-2.0, -1.0, 0.0, 0.25, 1.0, 23.0, 30.0], &[7], true);
    let out = fake_quantize_per_tensor_affine(&x, 0.1, 10, 0, 255).unwrap();
    assert_cuda_with(
        &out,
        "fake_quantize_per_tensor_affine (asym) output",
        &[-1.0, -1.0, 0.0, 0.200_000_002_980_232_24, 1.0, 23.0, 24.5],
    );

    let seed = t_cuda::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], &[7], false);
    backward_with_grad(&out, Some(&seed)).unwrap();
    let gx = grad_of(&x);
    assert_cuda_with(
        &gx,
        "fake_quantize_per_tensor_affine (asym) grad_input",
        &[0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0],
    );
}

/// No-grad path must also preserve residency (the demotion was in the
/// forward, before the grad_fn attach decision).
#[test]
fn fake_quantize_per_tensor_cuda_resident_no_grad_f32() {
    ensure_cuda_backend();
    let x = t_cuda::<f32>(&[-50.0, 1.0], &[2], false);
    let out = fake_quantize_per_tensor_affine(&x, 0.25, 0, -128, 127).unwrap();
    assert_cuda_with(
        &out,
        "fake_quantize_per_tensor_affine no-grad output",
        &[-32.0, 1.0],
    );
    assert!(out.grad_fn().is_none());
}

// ---------------------------------------------------------------------------
// per-tensor, tensor qparams
// ---------------------------------------------------------------------------

/// Tensor-qparams overload, all operands CUDA. Live torch 2.11.0+cu130:
///   x = torch.tensor([-50., -1., .1, 1., 5., 100.], device='cuda',
///                    requires_grad=True)
///   out = torch.fake_quantize_per_tensor_affine(
///       x, torch.tensor([0.25], device='cuda'),
///       torch.tensor([0], dtype=torch.int32, device='cuda'), -128, 127)
///   out.backward(torch.tensor([1.,2.,3.,4.,5.,6.], device='cuda'))
///   -> out  [-32.0, -1.0, 0.0, 1.0, 5.0, 31.75] cuda:0
///   -> grad [0.0, 2.0, 3.0, 4.0, 5.0, 0.0]      cuda:0
/// (ferrotorch's zero_point carrier is `IntTensor<i64>` per the documented
/// R-DEV deviation on the function; values are dtype-independent here.)
#[test]
fn fake_quantize_tensor_qparams_cuda_resident_f32() {
    ensure_cuda_backend();
    let x = t_cuda::<f32>(&[-50.0, -1.0, 0.1, 1.0, 5.0, 100.0], &[6], true);
    let scale = t_cuda::<f32>(&[0.25], &[1], false);
    let zp = IntTensor::from_vec(vec![0_i64], vec![1])
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();
    let out = fake_quantize_per_tensor_affine_tensor_qparams(&x, &scale, &zp, -128, 127).unwrap();
    assert_cuda_with(
        &out,
        "fake_quantize tensor-qparams output",
        &[-32.0, -1.0, 0.0, 1.0, 5.0, 31.75],
    );

    let seed = t_cuda::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6], false);
    backward_with_grad(&out, Some(&seed)).unwrap();
    let gx = grad_of(&x);
    assert_cuda_with(
        &gx,
        "fake_quantize tensor-qparams grad_input",
        &[0.0, 2.0, 3.0, 4.0, 5.0, 0.0],
    );
}

/// Live torch 2.11.0+cu130 rejects every mixed-device tensor-qparams operand
/// set at the wrapper boundary before reading qparam data:
///   x cuda, scale cpu, zp cpu  -> scale is on cpu, other tensors cuda:0
///   x cuda, scale cuda, zp cpu -> zero_point is on cpu, other tensors cuda:0
///   x cpu,  scale cuda, zp cuda -> scale is on cuda:0, other tensors cpu
#[test]
fn fake_quantize_tensor_qparams_rejects_mixed_devices() {
    ensure_cuda_backend();
    let x_cuda = t_cuda::<f32>(&[1.0, 2.0], &[2], false);
    let x_cpu = t_cpu::<f32>(&[1.0, 2.0], &[2], false);
    let scale_cuda = t_cuda::<f32>(&[0.25], &[1], false);
    let scale_cpu = t_cpu::<f32>(&[0.25], &[1], false);
    let zp_cuda = zp_cuda(&[0], &[1]);
    let zp_cpu = zp_cpu(&[0], &[1]);

    assert_device_mismatch(
        fake_quantize_per_tensor_affine_tensor_qparams(&x_cuda, &scale_cpu, &zp_cpu, -128, 127),
        Device::Cuda(0),
        Device::Cpu,
        "x cuda, scale cpu, zp cpu",
    );
    assert_device_mismatch(
        fake_quantize_per_tensor_affine_tensor_qparams(&x_cuda, &scale_cuda, &zp_cpu, -128, 127),
        Device::Cuda(0),
        Device::Cpu,
        "x cuda, scale cuda, zp cpu",
    );
    assert_device_mismatch(
        fake_quantize_per_tensor_affine_tensor_qparams(&x_cuda, &scale_cpu, &zp_cuda, -128, 127),
        Device::Cuda(0),
        Device::Cpu,
        "x cuda, scale cpu, zp cuda",
    );
    assert_device_mismatch(
        fake_quantize_per_tensor_affine_tensor_qparams(&x_cpu, &scale_cuda, &zp_cuda, -128, 127),
        Device::Cpu,
        Device::Cuda(0),
        "x cpu, scale cuda, zp cuda",
    );
}

// ---------------------------------------------------------------------------
// per-channel
// ---------------------------------------------------------------------------

/// Per-channel forward + STE backward, all operands CUDA, f32 (upstream
/// admits only Float/BFloat16 scales — `FakeQuantPerChannelAffine.cpp:51-52`).
/// Live torch 2.11.0+cu130:
///   x  = torch.tensor([[-50., .1, 1.], [2., -.3, .7]], device='cuda',
///                     requires_grad=True)
///   sc = torch.tensor([0.25, 0.5], device='cuda')
///   zp = torch.tensor([0, 1], dtype=torch.int32, device='cuda')
///   out = torch.fake_quantize_per_channel_affine(x, sc, zp, 0, -128, 127)
///   out.backward(torch.tensor([[1.,2.,3.],[4.,5.,6.]], device='cuda'))
///   -> out  [[-32.0, 0.0, 1.0], [2.0, -0.5, 0.5]] cuda:0
///   -> grad [[0.0, 2.0, 3.0], [4.0, 5.0, 6.0]]    cuda:0
///   (row 0, scale 0.25 zp 0: -50 → q=-200 clamps to -128 → -32.0, mask 0;
///    row 1, scale 0.5 zp 1: -0.3 → q=1+round_ties_even(-0.6)=0 → -0.5,
///    mask 1.)
#[test]
fn fake_quantize_per_channel_cuda_resident_f32() {
    ensure_cuda_backend();
    let x = t_cuda::<f32>(&[-50.0, 0.1, 1.0, 2.0, -0.3, 0.7], &[2, 3], true);
    let scale = t_cuda::<f32>(&[0.25, 0.5], &[2], false);
    let zp = IntTensor::from_vec(vec![0_i64, 1], vec![2])
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();
    let out = fake_quantize_per_channel_affine(&x, &scale, &zp, 0, -128, 127).unwrap();
    assert_cuda_with(
        &out,
        "fake_quantize_per_channel_affine output",
        &[-32.0, 0.0, 1.0, 2.0, -0.5, 0.5],
    );
    assert_eq!(out.shape(), &[2, 3]);

    let seed = t_cuda::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
    backward_with_grad(&out, Some(&seed)).unwrap();
    let gx = grad_of(&x);
    assert_cuda_with(
        &gx,
        "fake_quantize_per_channel_affine grad_input",
        &[0.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    );
}

/// No-grad per-channel path must also preserve residency.
#[test]
fn fake_quantize_per_channel_cuda_resident_no_grad_f32() {
    ensure_cuda_backend();
    let x = t_cuda::<f32>(&[-50.0, 0.1, 1.0, 2.0, -0.3, 0.7], &[2, 3], false);
    let scale = t_cuda::<f32>(&[0.25, 0.5], &[2], false);
    let zp = IntTensor::from_vec(vec![0_i64, 1], vec![2])
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();
    let out = fake_quantize_per_channel_affine(&x, &scale, &zp, 0, -128, 127).unwrap();
    assert_cuda_with(
        &out,
        "fake_quantize_per_channel_affine no-grad output",
        &[-32.0, 0.0, 1.0, 2.0, -0.5, 0.5],
    );
    assert!(out.grad_fn().is_none());
}

/// Live torch 2.11.0+cu130 rejects every mixed-device per-channel qparam set
/// at the wrapper/TensorIterator boundary; no qparam host readback is allowed
/// to make invalid mixed placement appear to work.
#[test]
fn fake_quantize_per_channel_rejects_mixed_devices() {
    ensure_cuda_backend();
    let x_cuda = t_cuda::<f32>(&[1.0, 2.0], &[1, 2], false);
    let x_cpu = t_cpu::<f32>(&[1.0, 2.0], &[1, 2], false);
    let scale_cuda = t_cuda::<f32>(&[0.25, 0.5], &[2], false);
    let scale_cpu = t_cpu::<f32>(&[0.25, 0.5], &[2], false);
    let zp_cuda = zp_cuda(&[0, 1], &[2]);
    let zp_cpu = zp_cpu(&[0, 1], &[2]);

    assert_device_mismatch(
        fake_quantize_per_channel_affine(&x_cuda, &scale_cpu, &zp_cpu, 1, -128, 127),
        Device::Cuda(0),
        Device::Cpu,
        "x cuda, scale cpu, zp cpu",
    );
    assert_device_mismatch(
        fake_quantize_per_channel_affine(&x_cuda, &scale_cuda, &zp_cpu, 1, -128, 127),
        Device::Cuda(0),
        Device::Cpu,
        "x cuda, scale cuda, zp cpu",
    );
    assert_device_mismatch(
        fake_quantize_per_channel_affine(&x_cuda, &scale_cpu, &zp_cuda, 1, -128, 127),
        Device::Cuda(0),
        Device::Cpu,
        "x cuda, scale cpu, zp cuda",
    );
    assert_device_mismatch(
        fake_quantize_per_channel_affine(&x_cpu, &scale_cuda, &zp_cuda, 1, -128, 127),
        Device::Cpu,
        Device::Cuda(0),
        "x cpu, scale cuda, zp cuda",
    );
}
