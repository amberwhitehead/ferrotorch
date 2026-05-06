//! Permanent regression sentinel for #820: `log_softmax_f64` GPU **backward**
//! gradient is wildly wrong (delta ~4.0 absolute on a (1,4) probe).
//!
//! This is the f64 sibling of #798 (f32, closed in commit 2fbb23d8). The
//! f64 forward path was previously blocked by #797 (EXP_F64_PTX JIT
//! failure). With #797 closed in Batch 3, the f64 forward now runs live
//! — and the backward kernel surfaces the same algebraic bug as the f32
//! kernel did:
//!
//! `LOG_SOFTMAX_BACKWARD_F64_PTX` reads `output[j]` and runs an inline
//! f64 exp polynomial to "recover" the softmax probability. But the Rust
//! host (`grad_fns/activation.rs::log_softmax_inner`) already saved
//! `softmax = exp(log_softmax)` at forward time and passes those
//! probabilities in as `output_ptr`. The kernel's inline exp therefore
//! double-exp's the buffer, yielding a fixed-shape multiplicative error.
//!
//! Math reference (PyTorch parity):
//!   log_softmax(x).backward(grad_out)
//!     = grad_out − softmax(x) · sum(grad_out, dim=-1, keepdim=True)
//!
//! For x = [1, 2, 3, 4] (last-dim row), upstream grad = ones (from sum_all):
//!   softmax(x) ≈ [0.0321, 0.0871, 0.2369, 0.6439]
//!   sum(grad)  = 4
//!   grad_in[j] = 1 − softmax[j] * 4  ≈ [0.8716, 0.6517, 0.0526, −1.5759]
//!
//! Post-fix the GPU result must match this CPU reference within
//! `F64_TRANSCENDENTAL = 1e-10`.

#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::Device;
use ferrotorch_core::autograd::graph::backward;
use ferrotorch_core::creation::from_vec;
use ferrotorch_core::grad_fns::activation::log_softmax;

const F64_TRANSCENDENTAL: f64 = 1e-10;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for the GPU probe suite");
    });
}

/// CPU reference: `grad_in = grad_out - softmax(x) * sum(grad_out, dim=-1)`
/// over rows of length `cols`.
fn cpu_reference_grad(x: &[f64], grad_out: &[f64], cols: usize) -> Vec<f64> {
    assert_eq!(x.len(), grad_out.len());
    let rows = x.len() / cols;
    let mut out = vec![0.0f64; x.len()];
    for r in 0..rows {
        let base = r * cols;
        let row = &x[base..base + cols];
        let mut max_v = row[0];
        for &v in &row[1..] {
            if v > max_v {
                max_v = v;
            }
        }
        let mut sum_e = 0.0f64;
        let mut sm = vec![0.0f64; cols];
        for (j, &v) in row.iter().enumerate() {
            sm[j] = (v - max_v).exp();
            sum_e += sm[j];
        }
        for s in &mut sm {
            *s /= sum_e;
        }
        let sum_g: f64 = grad_out[base..base + cols].iter().sum();
        for j in 0..cols {
            out[base + j] = grad_out[base + j] - sm[j] * sum_g;
        }
    }
    out
}

/// `log_softmax_f64` backward agreement probe — the (1,4) f64 minimum-shape
/// fixture from issue #820. Mirrors the f32 probe in
/// `_probe_a2_log_softmax_grad.rs`.
#[test]
fn log_softmax_grad_cuda_f64_minimum_shape() {
    ensure_cuda_backend();
    let raw = vec![1.0f64, 2.0, 3.0, 4.0];

    // Forward agreement on GPU vs CPU first — #797 closed the f64 forward,
    // so this should already pass within F64_TRANSCENDENTAL.
    println!("step 1: forward log_softmax_f64 on CPU and CUDA, compare");
    let cpu_x = from_vec::<f64>(raw.clone(), &[1, 4]).expect("cpu input");
    let cpu_y = log_softmax(&cpu_x).expect("cpu forward");
    let cpu_y_data = cpu_y.data().expect("cpu fwd data").to_vec();

    let gpu_x = cpu_x
        .to(Device::Cuda(0))
        .expect("cpu->gpu")
        .requires_grad_(true);
    let gpu_y = log_softmax(&gpu_x).expect("gpu forward");
    assert!(gpu_y.is_cuda(), "log_softmax output must remain on CUDA");
    let gpu_y_cpu = gpu_y.cpu().expect("gpu->cpu fwd");
    let gpu_y_data = gpu_y_cpu.data().expect("gpu fwd data").to_vec();

    println!("    cpu fwd: {cpu_y_data:?}");
    println!("    gpu fwd: {gpu_y_data:?}");
    for (i, (g, c)) in gpu_y_data.iter().zip(cpu_y_data.iter()).enumerate() {
        assert!(
            (g - c).abs() < F64_TRANSCENDENTAL,
            "forward divergence at {i}: gpu={g}, cpu={c}"
        );
    }

    // Backward — sum_all then backward, compare grad to CPU reference.
    println!("step 2: backward log_softmax_f64 on CUDA via sum_all().backward()");
    let s = gpu_y.sum_all().expect("sum");
    backward(&s).expect("backward");
    let g_gpu = gpu_x
        .grad()
        .expect("grad lookup")
        .expect("grad attached")
        .cpu()
        .expect("grad gpu->cpu");
    let g_gpu_data = g_gpu.data().expect("grad data").to_vec();

    let grad_out_ones = vec![1.0f64; 4];
    let expected = cpu_reference_grad(&raw, &grad_out_ones, 4);

    println!("    gpu grad: {g_gpu_data:?}");
    println!("    cpu ref : {expected:?}");

    let mut max_delta = 0.0f64;
    for (i, (a, b)) in g_gpu_data.iter().zip(expected.iter()).enumerate() {
        let d = (a - b).abs();
        if d > max_delta {
            max_delta = d;
        }
        println!("    [{i}] gpu={a:.12} ref={b:.12} delta={d:.3e}");
    }
    println!("    max_delta = {max_delta:.3e}");

    assert!(
        max_delta < F64_TRANSCENDENTAL,
        "log_softmax f64 GPU backward delta {max_delta} exceeds F64_TRANSCENDENTAL=1e-10; \
         pre-fix #820 reported delta ~4.0 (kernel did exp() of an already-exp'd softmax buffer)"
    );
}

/// Wider (2, 8) f64 row coverage so the post-fix sentinel exercises the
/// per-row reduction more than once.
#[test]
fn log_softmax_grad_cuda_f64_two_rows() {
    ensure_cuda_backend();
    let raw: Vec<f64> = (0..16).map(|i| (i as f64) * 0.25 - 1.0).collect();
    let cpu_x = from_vec::<f64>(raw.clone(), &[2, 8]).expect("cpu input");

    let gpu_x = cpu_x
        .to(Device::Cuda(0))
        .expect("cpu->gpu")
        .requires_grad_(true);
    let gpu_y = log_softmax(&gpu_x).expect("gpu forward");
    let s = gpu_y.sum_all().expect("sum");
    backward(&s).expect("backward");
    let g_gpu = gpu_x
        .grad()
        .expect("grad lookup")
        .expect("grad attached")
        .cpu()
        .expect("grad gpu->cpu");
    let g_gpu_data = g_gpu.data().expect("grad data").to_vec();

    let grad_out_ones = vec![1.0f64; 16];
    let expected = cpu_reference_grad(&raw, &grad_out_ones, 8);

    for (i, (a, b)) in g_gpu_data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - b).abs() < F64_TRANSCENDENTAL,
            "log_softmax (2,8) f64 grad mismatch at {i}: gpu={a}, cpu={b}"
        );
    }
}
