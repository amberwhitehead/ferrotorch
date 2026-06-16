//! CUDA-resident parity probes for `addcmul` and `addcdiv`.
//!
//! PyTorch routes these through native ternary CUDA kernels:
//!
//! - `aten/src/ATen/native/cuda/PointwiseOpsKernel.cu`
//! - `aten/src/ATen/native/cuda/DeviceAddCmulCdiv.cuh`
//!
//! The probes below pin CUDA residency, broadcast gradients, IEEE edges
//! around zero divisors / infinities / NaNs / signed zeros, and the reduced
//! dtype f32-opmath scalar behavior that PyTorch uses for f16/bf16.

#![cfg(feature = "cuda")]

use std::sync::Once;

use ferrotorch_core::grad_fns::{arithmetic, reduction};
use ferrotorch_core::{Device, Float, Tensor, TensorStorage};
use ferrotorch_gpu::init_cuda_backend;

fn ensure_cuda() -> bool {
    static INIT: Once = Once::new();
    static mut OK: bool = false;
    if ferrotorch_gpu::device::GpuDevice::new(0).is_err() {
        return false;
    }
    INIT.call_once(|| {
        init_cuda_backend().expect("CUDA backend init");
        unsafe { OK = true }
    });
    unsafe { OK }
}

fn cast<T: Float>(x: f32) -> T {
    T::from(x).expect("test value representable in dtype")
}

fn cpu_tensor<T: Float>(data: &[T], shape: &[usize], requires_grad: bool) -> Tensor<T> {
    Tensor::from_storage(
        TensorStorage::cpu(data.to_vec()),
        shape.to_vec(),
        requires_grad,
    )
    .expect("cpu tensor")
}

fn cuda_tensor<T: Float>(data: &[T], shape: &[usize], requires_grad: bool) -> Tensor<T> {
    cpu_tensor(data, shape, false)
        .to(Device::Cuda(0))
        .expect("to cuda")
        .requires_grad_(requires_grad)
}

fn cpu_values_f32<T: Float>(t: &Tensor<T>) -> Vec<f32> {
    t.cpu()
        .expect("to cpu")
        .data()
        .expect("cpu data")
        .iter()
        .map(|v| v.to_f32().expect("value representable as f32"))
        .collect()
}

fn grad_values_f32<T: Float>(t: &Tensor<T>) -> (Device, Vec<f32>) {
    let grad = t.grad().expect("grad access").expect("grad present");
    (grad.device(), cpu_values_f32(&grad))
}

#[derive(Clone, Copy)]
enum Want {
    Num(f32),
    PosZero,
    NegZero,
    PosInf,
    NegInf,
    NaN,
}

fn assert_pattern(got: &[f32], want: &[Want], name: &str) {
    assert_eq!(got.len(), want.len(), "{name}: length mismatch");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        match w {
            Want::Num(v) => assert!((g - v).abs() <= 1e-3, "{name}[{i}] expected {v}, got {g}"),
            Want::PosZero => assert!(
                g == 0.0 && !g.is_sign_negative(),
                "{name}[{i}] expected +0, got {g:?}"
            ),
            Want::NegZero => assert!(
                g == 0.0 && g.is_sign_negative(),
                "{name}[{i}] expected -0, got {g:?}"
            ),
            Want::PosInf => assert!(g == f32::INFINITY, "{name}[{i}] expected +inf, got {g:?}"),
            Want::NegInf => assert!(
                g == f32::NEG_INFINITY,
                "{name}[{i}] expected -inf, got {g:?}"
            ),
            Want::NaN => assert!(g.is_nan(), "{name}[{i}] expected NaN, got {g:?}"),
        }
    }
}

fn run_forward_edges<T: Float>(dtype_name: &str) {
    let input = cuda_tensor(
        &[
            cast::<T>(1.0),
            cast::<T>(1.0),
            cast::<T>(1.0),
            cast::<T>(-0.0),
        ],
        &[4],
        false,
    );
    let tensor1 = cuda_tensor(
        &[
            cast::<T>(2.0),
            cast::<T>(0.0),
            cast::<T>(f32::INFINITY),
            cast::<T>(1.0),
        ],
        &[4],
        false,
    );
    let tensor2 = cuda_tensor(
        &[
            cast::<T>(0.0),
            cast::<T>(0.0),
            cast::<T>(2.0),
            cast::<T>(-0.0),
        ],
        &[4],
        false,
    );

    let addcmul_zero =
        arithmetic::addcmul(&input, &tensor1, &tensor2, 0.0).expect("addcmul value=0");
    let addcmul_neg =
        arithmetic::addcmul(&input, &tensor1, &tensor2, -1.0).expect("addcmul value=-1");
    let addcmul_inf =
        arithmetic::addcmul(&input, &tensor1, &tensor2, f64::INFINITY).expect("addcmul value=inf");
    let addcdiv_zero =
        arithmetic::addcdiv(&input, &tensor1, &tensor2, 0.0).expect("addcdiv value=0");
    let addcdiv_inf =
        arithmetic::addcdiv(&input, &tensor1, &tensor2, f64::INFINITY).expect("addcdiv value=inf");

    for (name, out) in [
        ("addcmul_zero", &addcmul_zero),
        ("addcmul_neg", &addcmul_neg),
        ("addcmul_inf", &addcmul_inf),
        ("addcdiv_zero", &addcdiv_zero),
        ("addcdiv_inf", &addcdiv_inf),
    ] {
        assert_eq!(
            out.device(),
            Device::Cuda(0),
            "{dtype_name} {name} must remain CUDA"
        );
    }

    assert_pattern(
        &cpu_values_f32(&addcmul_zero),
        &[Want::Num(1.0), Want::Num(1.0), Want::NaN, Want::NegZero],
        &format!("{dtype_name} addcmul value=0"),
    );
    assert_pattern(
        &cpu_values_f32(&addcmul_neg),
        &[Want::Num(1.0), Want::Num(1.0), Want::NegInf, Want::PosZero],
        &format!("{dtype_name} addcmul value=-1"),
    );
    assert_pattern(
        &cpu_values_f32(&addcmul_inf),
        &[Want::NaN, Want::NaN, Want::PosInf, Want::NaN],
        &format!("{dtype_name} addcmul value=inf"),
    );
    assert_pattern(
        &cpu_values_f32(&addcdiv_zero),
        &[Want::NaN, Want::NaN, Want::NaN, Want::NaN],
        &format!("{dtype_name} addcdiv value=0"),
    );
    assert_pattern(
        &cpu_values_f32(&addcdiv_inf),
        &[Want::PosInf, Want::NaN, Want::PosInf, Want::NegInf],
        &format!("{dtype_name} addcdiv value=inf"),
    );
}

#[test]
fn cuda_addcmul_addcdiv_forward_edges_all_float_dtypes() {
    if !ensure_cuda() {
        eprintln!("no CUDA device; skipping");
        return;
    }

    run_forward_edges::<f32>("f32");
    run_forward_edges::<f64>("f64");
    run_forward_edges::<half::f16>("f16");
    run_forward_edges::<half::bf16>("bf16");
}

#[test]
fn addcmul_addcdiv_reduced_dtypes_use_f32_scalar_opmath() {
    if !ensure_cuda() {
        eprintln!("no CUDA device; skipping");
        return;
    }

    let f16_one = [half::f16::from_f32(1.0)];
    let bf16_one = [half::bf16::from_f32(1.0)];

    for (label, got, want) in [
        (
            "cpu f16 addcmul",
            arithmetic::addcmul(
                &cpu_tensor(&f16_one, &[1], false),
                &cpu_tensor(&f16_one, &[1], false),
                &cpu_tensor(&f16_one, &[1], false),
                0.1,
            )
            .expect("cpu f16 addcmul"),
            1.0996094,
        ),
        (
            "cuda f16 addcmul",
            arithmetic::addcmul(
                &cuda_tensor(&f16_one, &[1], false),
                &cuda_tensor(&f16_one, &[1], false),
                &cuda_tensor(&f16_one, &[1], false),
                0.1,
            )
            .expect("cuda f16 addcmul"),
            1.0996094,
        ),
    ] {
        let got = cpu_values_f32(&got);
        assert_eq!(got.len(), 1, "{label}: scalar output length");
        assert!(
            (got[0] - want).abs() <= f32::EPSILON,
            "{label}: expected {want}, got {}",
            got[0]
        );
    }

    for (label, got, want) in [
        (
            "cpu bf16 addcdiv",
            arithmetic::addcdiv(
                &cpu_tensor(&bf16_one, &[1], false),
                &cpu_tensor(&bf16_one, &[1], false),
                &cpu_tensor(&bf16_one, &[1], false),
                0.1,
            )
            .expect("cpu bf16 addcdiv"),
            1.1015625,
        ),
        (
            "cuda bf16 addcdiv",
            arithmetic::addcdiv(
                &cuda_tensor(&bf16_one, &[1], false),
                &cuda_tensor(&bf16_one, &[1], false),
                &cuda_tensor(&bf16_one, &[1], false),
                0.1,
            )
            .expect("cuda bf16 addcdiv"),
            1.1015625,
        ),
    ] {
        let got = cpu_values_f32(&got);
        assert_eq!(got.len(), 1, "{label}: scalar output length");
        assert!(
            (got[0] - want).abs() <= f32::EPSILON,
            "{label}: expected {want}, got {}",
            got[0]
        );
    }
}

fn assert_close(got: &[f32], want: &[f32], name: &str) {
    assert_eq!(got.len(), want.len(), "{name}: length mismatch");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= 1e-3, "{name}[{i}] expected {w}, got {g}");
    }
}

fn run_backward_residency<T: Float>(dtype_name: &str) {
    let input = cuda_tensor(
        &[cast::<T>(1.0), cast::<T>(2.0), cast::<T>(3.0)],
        &[3],
        true,
    );
    let tensor1 = cuda_tensor(
        &[
            cast::<T>(2.0),
            cast::<T>(4.0),
            cast::<T>(6.0),
            cast::<T>(8.0),
            cast::<T>(10.0),
            cast::<T>(12.0),
        ],
        &[2, 3],
        true,
    );
    let tensor2 = cuda_tensor(
        &[cast::<T>(1.0), cast::<T>(2.0), cast::<T>(4.0)],
        &[1, 3],
        true,
    );

    let out = arithmetic::addcmul(&input, &tensor1, &tensor2, 0.5).expect("addcmul");
    assert_eq!(out.shape(), &[2, 3], "{dtype_name} addcmul shape");
    assert!(out.is_cuda(), "{dtype_name} addcmul output must stay CUDA");
    reduction::sum(&out)
        .expect("sum")
        .backward()
        .expect("backward");

    let (dev, grad) = grad_values_f32(&input);
    assert_eq!(
        dev,
        Device::Cuda(0),
        "{dtype_name} addcmul input grad device"
    );
    assert_close(
        &grad,
        &[2.0, 2.0, 2.0],
        &format!("{dtype_name} addcmul input grad"),
    );
    let (dev, grad) = grad_values_f32(&tensor1);
    assert_eq!(
        dev,
        Device::Cuda(0),
        "{dtype_name} addcmul tensor1 grad device"
    );
    assert_close(
        &grad,
        &[0.5, 1.0, 2.0, 0.5, 1.0, 2.0],
        &format!("{dtype_name} addcmul tensor1 grad"),
    );
    let (dev, grad) = grad_values_f32(&tensor2);
    assert_eq!(
        dev,
        Device::Cuda(0),
        "{dtype_name} addcmul tensor2 grad device"
    );
    assert_close(
        &grad,
        &[5.0, 7.0, 9.0],
        &format!("{dtype_name} addcmul tensor2 grad"),
    );

    let input = cuda_tensor(
        &[cast::<T>(1.0), cast::<T>(2.0), cast::<T>(3.0)],
        &[3],
        true,
    );
    let tensor1 = cuda_tensor(
        &[
            cast::<T>(2.0),
            cast::<T>(4.0),
            cast::<T>(6.0),
            cast::<T>(8.0),
            cast::<T>(10.0),
            cast::<T>(12.0),
        ],
        &[2, 3],
        true,
    );
    let tensor2 = cuda_tensor(
        &[cast::<T>(1.0), cast::<T>(2.0), cast::<T>(4.0)],
        &[1, 3],
        true,
    );

    let out = arithmetic::addcdiv(&input, &tensor1, &tensor2, 0.5).expect("addcdiv");
    assert_eq!(out.shape(), &[2, 3], "{dtype_name} addcdiv shape");
    assert!(out.is_cuda(), "{dtype_name} addcdiv output must stay CUDA");
    reduction::sum(&out)
        .expect("sum")
        .backward()
        .expect("backward");

    let (dev, grad) = grad_values_f32(&input);
    assert_eq!(
        dev,
        Device::Cuda(0),
        "{dtype_name} addcdiv input grad device"
    );
    assert_close(
        &grad,
        &[2.0, 2.0, 2.0],
        &format!("{dtype_name} addcdiv input grad"),
    );
    let (dev, grad) = grad_values_f32(&tensor1);
    assert_eq!(
        dev,
        Device::Cuda(0),
        "{dtype_name} addcdiv tensor1 grad device"
    );
    assert_close(
        &grad,
        &[0.5, 0.25, 0.125, 0.5, 0.25, 0.125],
        &format!("{dtype_name} addcdiv tensor1 grad"),
    );
    let (dev, grad) = grad_values_f32(&tensor2);
    assert_eq!(
        dev,
        Device::Cuda(0),
        "{dtype_name} addcdiv tensor2 grad device"
    );
    assert_close(
        &grad,
        &[-5.0, -1.75, -0.5625],
        &format!("{dtype_name} addcdiv tensor2 grad"),
    );
}

#[test]
fn cuda_addcmul_addcdiv_backward_broadcast_grads_stay_cuda() {
    if !ensure_cuda() {
        eprintln!("no CUDA device; skipping");
        return;
    }

    run_backward_residency::<f32>("f32");
    run_backward_residency::<f64>("f64");
    run_backward_residency::<half::f16>("f16");
    run_backward_residency::<half::bf16>("bf16");
}
