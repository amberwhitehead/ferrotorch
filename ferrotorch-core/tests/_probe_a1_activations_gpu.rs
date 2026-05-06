//! Permanent regression sentinel for #796: sin / cos / leaky_relu /
//! softplus autograd-on-CUDA broken — backward grad_fns saved a CPU vec
//! via `.data()?`, which fails with `GpuTensorNotAccessible` when the
//! saved tensor lives in VRAM.
//!
//! Pre-fix observable failure for each of the four ops:
//!   1. Construct a CUDA tensor with `requires_grad = true`.
//!   2. Call the op (forward attaches the backward grad_fn).
//!   3. Call `.sum().backward()`.
//!   4. The grad_fn calls `self.input.data()?` on a CUDA `Tensor`, which
//!      returns `Err(GpuTensorNotAccessible)`. The whole `backward()`
//!      bubbles that error up.
//!
//! This probe documents the working post-fix sequence: each step prints
//! its outcome, so the captured stdout is the spec. Post-fix, every op
//! returns a finite gradient on the GPU side that matches the CPU
//! reference within the same dtype tolerance.

#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::autograd::graph::backward;
use ferrotorch_core::creation::from_vec;
use ferrotorch_core::grad_fns::activation::{leaky_relu, softplus};
use ferrotorch_core::grad_fns::transcendental::{cos, sin};
use ferrotorch_core::Device;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for the GPU probe suite");
    });
}

/// Sin autograd on CUDA f32 — pre-fix fails at the saved-state `.data()?`.
#[test]
fn sin_autograd_cuda_f32() {
    ensure_cuda_backend();
    println!("step 1: build CUDA f32 input with requires_grad=true");
    let cpu = from_vec::<f32>(vec![0.1, 0.5, 1.0, -0.7], &[4]).expect("cpu tensor");
    let gpu = cpu
        .to(Device::Cuda(0))
        .expect("cpu->gpu")
        .requires_grad_(true);

    println!("step 2: forward sin on CUDA tensor");
    let y = sin(&gpu).expect("sin forward (post-fix; pre-fix fast_sin .data()? trap)");
    assert!(y.is_cuda(), "sin output must remain on CUDA device");

    println!("step 3: y.sum_all().backward()");
    let s = y.sum_all().expect("sum");
    backward(&s).expect("sin.backward (post-fix; pre-fix #796 GpuTensorNotAccessible)");

    println!("step 4: grad of input is finite and matches cos(x)");
    let g = gpu.grad().expect("grad lookup").expect("grad attached");
    let g_cpu = g.cpu().expect("grad gpu->cpu");
    let g_data = g_cpu.data().expect("grad data");
    // sum() backward produces ones, so dx = cos(x). Tolerance generous
    // for the fast-trig approximation used on GPU.
    let expected: Vec<f32> = vec![0.1f32.cos(), 0.5f32.cos(), 1.0f32.cos(), (-0.7f32).cos()];
    for (i, (actual, want)) in g_data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - want).abs() < 1e-4,
            "sin grad mismatch at {i}: got {actual}, want {want}"
        );
    }
}

/// Cos autograd on CUDA f32 — same shape as `sin_autograd_cuda_f32`.
#[test]
fn cos_autograd_cuda_f32() {
    ensure_cuda_backend();
    println!("step 1: build CUDA f32 input with requires_grad=true");
    let cpu = from_vec::<f32>(vec![0.1, 0.5, 1.0, -0.7], &[4]).expect("cpu tensor");
    let gpu = cpu
        .to(Device::Cuda(0))
        .expect("cpu->gpu")
        .requires_grad_(true);

    println!("step 2: forward cos on CUDA tensor");
    let y = cos(&gpu).expect("cos forward (post-fix; pre-fix fast_cos .data()? trap)");
    assert!(y.is_cuda(), "cos output must remain on CUDA device");

    println!("step 3: y.sum_all().backward()");
    let s = y.sum_all().expect("sum");
    backward(&s).expect("cos.backward (post-fix; pre-fix #796 GpuTensorNotAccessible)");

    println!("step 4: grad of input is finite and matches -sin(x)");
    let g = gpu.grad().expect("grad lookup").expect("grad attached");
    let g_cpu = g.cpu().expect("grad gpu->cpu");
    let g_data = g_cpu.data().expect("grad data");
    let expected: Vec<f32> =
        vec![-(0.1f32.sin()), -(0.5f32.sin()), -(1.0f32.sin()), -((-0.7f32).sin())];
    for (i, (actual, want)) in g_data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - want).abs() < 1e-4,
            "cos grad mismatch at {i}: got {actual}, want {want}"
        );
    }
}

/// Leaky-ReLU autograd on CUDA f32 — pre-fix had no GPU dispatch in
/// `LeakyReluBackward`, hitting `self.input.data()?`.
#[test]
fn leaky_relu_autograd_cuda_f32() {
    ensure_cuda_backend();
    println!("step 1: build CUDA f32 input straddling zero");
    let cpu = from_vec::<f32>(vec![-2.0, -0.5, 0.5, 2.0], &[4]).expect("cpu tensor");
    let gpu = cpu
        .to(Device::Cuda(0))
        .expect("cpu->gpu")
        .requires_grad_(true);

    println!("step 2: forward leaky_relu on CUDA tensor (slope=0.1)");
    let y = leaky_relu(&gpu, 0.1).expect("leaky_relu forward");
    assert!(
        y.is_cuda(),
        "leaky_relu output must remain on CUDA device (pre-fix forward stored CPU storage)"
    );

    println!("step 3: y.sum_all().backward()");
    let s = y.sum_all().expect("sum");
    backward(&s).expect("leaky_relu.backward (post-fix; pre-fix #796)");

    println!("step 4: grad is slope where x<0, 1 where x>0");
    let g = gpu.grad().expect("grad lookup").expect("grad attached");
    let g_cpu = g.cpu().expect("grad gpu->cpu");
    let g_data = g_cpu.data().expect("grad data");
    // x = [-2, -0.5, 0.5, 2] → grad = [0.1, 0.1, 1.0, 1.0]
    assert!((g_data[0] - 0.1).abs() < 1e-6);
    assert!((g_data[1] - 0.1).abs() < 1e-6);
    assert!((g_data[2] - 1.0).abs() < 1e-6);
    assert!((g_data[3] - 1.0).abs() < 1e-6);
}

/// Softplus autograd on CUDA f32 — pre-fix forward boxed CPU storage even
/// for CUDA inputs (`output.data()?.to_vec()`), so backward saved a CPU
/// tensor and the reverse pass succeeded only because the input was
/// already CPU. With a real CUDA input the saved input fails `.data()?`.
#[test]
fn softplus_autograd_cuda_f32() {
    ensure_cuda_backend();
    println!("step 1: build CUDA f32 input around zero");
    let cpu = from_vec::<f32>(vec![-1.0, 0.0, 1.0, 2.0], &[4]).expect("cpu tensor");
    let gpu = cpu
        .to(Device::Cuda(0))
        .expect("cpu->gpu")
        .requires_grad_(true);

    println!("step 2: forward softplus on CUDA tensor (beta=1.0, threshold=20.0)");
    let y = softplus(&gpu, 1.0, 20.0).expect("softplus forward");
    assert!(
        y.is_cuda(),
        "softplus output must remain on CUDA device (pre-fix forward forced CPU storage)"
    );

    println!("step 3: y.sum_all().backward()");
    let s = y.sum_all().expect("sum");
    backward(&s).expect("softplus.backward (post-fix; pre-fix #796)");

    println!("step 4: grad equals sigmoid(beta * x)");
    let g = gpu.grad().expect("grad lookup").expect("grad attached");
    let g_cpu = g.cpu().expect("grad gpu->cpu");
    let g_data = g_cpu.data().expect("grad data");
    let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
    let expected = [sigmoid(-1.0), sigmoid(0.0), sigmoid(1.0), sigmoid(2.0)];
    for (i, (actual, want)) in g_data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - want).abs() < 1e-4,
            "softplus grad mismatch at {i}: got {actual}, want {want}"
        );
    }
}

// ---------------------------------------------------------------------------
// f64 dtype coverage — PyTorch parity requires both f32 and f64 GPU paths.
// ---------------------------------------------------------------------------

#[test]
fn sin_autograd_cuda_f64() {
    ensure_cuda_backend();
    let cpu = from_vec::<f64>(vec![0.1, 0.5, 1.0, -0.7], &[4]).expect("cpu tensor");
    let gpu = cpu
        .to(Device::Cuda(0))
        .expect("cpu->gpu")
        .requires_grad_(true);

    let y = sin(&gpu).expect("sin forward f64");
    assert!(y.is_cuda());
    let s = y.sum_all().expect("sum");
    backward(&s).expect("sin.backward f64");

    let g = gpu.grad().expect("grad lookup").expect("grad attached");
    let g_cpu = g.cpu().expect("grad gpu->cpu");
    let g_data = g_cpu.data().expect("grad data");
    let expected: Vec<f64> = vec![0.1f64.cos(), 0.5f64.cos(), 1.0f64.cos(), (-0.7f64).cos()];
    for (i, (actual, want)) in g_data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - want).abs() < 1e-9,
            "sin f64 grad mismatch at {i}: got {actual}, want {want}"
        );
    }
}

#[test]
fn cos_autograd_cuda_f64() {
    ensure_cuda_backend();
    let cpu = from_vec::<f64>(vec![0.1, 0.5, 1.0, -0.7], &[4]).expect("cpu tensor");
    let gpu = cpu
        .to(Device::Cuda(0))
        .expect("cpu->gpu")
        .requires_grad_(true);

    let y = cos(&gpu).expect("cos forward f64");
    assert!(y.is_cuda());
    let s = y.sum_all().expect("sum");
    backward(&s).expect("cos.backward f64");

    let g = gpu.grad().expect("grad lookup").expect("grad attached");
    let g_cpu = g.cpu().expect("grad gpu->cpu");
    let g_data = g_cpu.data().expect("grad data");
    let expected: Vec<f64> =
        vec![-(0.1f64.sin()), -(0.5f64.sin()), -(1.0f64.sin()), -((-0.7f64).sin())];
    for (i, (actual, want)) in g_data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - want).abs() < 1e-9,
            "cos f64 grad mismatch at {i}: got {actual}, want {want}"
        );
    }
}

#[test]
fn leaky_relu_autograd_cuda_f64() {
    ensure_cuda_backend();
    let cpu = from_vec::<f64>(vec![-2.0, -0.5, 0.5, 2.0], &[4]).expect("cpu tensor");
    let gpu = cpu
        .to(Device::Cuda(0))
        .expect("cpu->gpu")
        .requires_grad_(true);

    let y = leaky_relu(&gpu, 0.1).expect("leaky_relu f64 forward");
    assert!(y.is_cuda());
    let s = y.sum_all().expect("sum");
    backward(&s).expect("leaky_relu f64 backward");

    let g = gpu.grad().expect("grad lookup").expect("grad attached");
    let g_cpu = g.cpu().expect("grad gpu->cpu");
    let g_data = g_cpu.data().expect("grad data");
    assert!((g_data[0] - 0.1).abs() < 1e-12);
    assert!((g_data[1] - 0.1).abs() < 1e-12);
    assert!((g_data[2] - 1.0).abs() < 1e-12);
    assert!((g_data[3] - 1.0).abs() < 1e-12);
}

#[test]
fn softplus_autograd_cuda_f64() {
    ensure_cuda_backend();
    let cpu = from_vec::<f64>(vec![-1.0, 0.0, 1.0, 2.0], &[4]).expect("cpu tensor");
    let gpu = cpu
        .to(Device::Cuda(0))
        .expect("cpu->gpu")
        .requires_grad_(true);

    let y = softplus(&gpu, 1.0, 20.0).expect("softplus f64 forward");
    assert!(y.is_cuda());
    let s = y.sum_all().expect("sum");
    backward(&s).expect("softplus f64 backward");

    let g = gpu.grad().expect("grad lookup").expect("grad attached");
    let g_cpu = g.cpu().expect("grad gpu->cpu");
    let g_data = g_cpu.data().expect("grad data");
    let sigmoid = |x: f64| 1.0 / (1.0 + (-x).exp());
    let expected = [sigmoid(-1.0), sigmoid(0.0), sigmoid(1.0), sigmoid(2.0)];
    for (i, (actual, want)) in g_data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - want).abs() < 1e-9,
            "softplus f64 grad mismatch at {i}: got {actual}, want {want}"
        );
    }
}
