//! Permanent regression sentinel for #798: `log_softmax_f32` GPU **backward**
//! gradient is wildly wrong (delta ~4.0 absolute on a (1,4) probe).
//!
//! Forward log_softmax is fine on GPU; only the backward path is broken.
//!
//! Pre-fix observable failure (root cause discovered while writing this probe):
//! the PTX backward kernel `log_softmax_backward_kernel` reads `output[j]` and
//! computes `exp(output[j])` to recover the softmax probability, on the
//! assumption that `output` is the **log-softmax** forward output. But the
//! Rust host (`grad_fns/activation.rs::log_softmax_inner`) pre-computes
//! `softmax = exp(log_softmax)` on GPU and passes that **already-exp'd**
//! buffer in as `output`. The kernel then takes `exp` of the softmax
//! probability, so the subtracted term becomes `exp(softmax[j]) * sum_grad`
//! instead of `softmax[j] * sum_grad`. For uniform input on (1,4) every
//! softmax[j] ≈ 0.25 → exp(0.25) ≈ 1.284 → grad off by a factor of ~5×
//! per element with a fixed wrong sign — the conformance probe's ~4.0
//! absolute delta.
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
//! Post-fix, the GPU result must match this CPU reference within an f32
//! tolerance of 1e-5 across the row.

#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::autograd::graph::backward;
use ferrotorch_core::creation::from_vec;
use ferrotorch_core::grad_fns::activation::log_softmax;
use ferrotorch_core::Device;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for the GPU probe suite");
    });
}

/// CPU reference: `grad_in = grad_out - softmax(x) * sum(grad_out, dim=-1)`
/// over rows of length `cols`.
fn cpu_reference_grad(x: &[f32], grad_out: &[f32], cols: usize) -> Vec<f32> {
    assert_eq!(x.len(), grad_out.len());
    let rows = x.len() / cols;
    let mut out = vec![0.0f32; x.len()];
    for r in 0..rows {
        let base = r * cols;
        let row = &x[base..base + cols];
        let mut max_v = row[0];
        for &v in &row[1..] {
            if v > max_v {
                max_v = v;
            }
        }
        let mut sum_e = 0.0f32;
        let mut sm = vec![0.0f32; cols];
        for (j, &v) in row.iter().enumerate() {
            sm[j] = (v - max_v).exp();
            sum_e += sm[j];
        }
        for s in &mut sm {
            *s /= sum_e;
        }
        let sum_g: f32 = grad_out[base..base + cols].iter().sum();
        for j in 0..cols {
            out[base + j] = grad_out[base + j] - sm[j] * sum_g;
        }
    }
    out
}

/// `log_softmax_f32` backward agreement probe — the (1,4) f32 minimum-shape
/// fixture from issue #798.
#[test]
fn log_softmax_grad_cuda_f32_minimum_shape() {
    ensure_cuda_backend();
    let raw = vec![1.0f32, 2.0, 3.0, 4.0];

    // Forward agreement on GPU vs CPU first (forward already passes per #798;
    // we capture the values so the diagnostic prints both).
    println!("step 1: forward log_softmax on CPU and CUDA, compare");
    let cpu_x = from_vec::<f32>(raw.clone(), &[1, 4]).expect("cpu input");
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
            (g - c).abs() < 1e-5,
            "forward divergence at {i}: gpu={g}, cpu={c}"
        );
    }

    // Backward — sum_all then backward, compare grad to CPU reference.
    println!("step 2: backward log_softmax on CUDA via sum_all().backward()");
    let s = gpu_y.sum_all().expect("sum");
    backward(&s).expect("backward");
    let g_gpu = gpu_x
        .grad()
        .expect("grad lookup")
        .expect("grad attached")
        .cpu()
        .expect("grad gpu->cpu");
    let g_gpu_data = g_gpu.data().expect("grad data").to_vec();

    let grad_out_ones = vec![1.0f32; 4];
    let expected = cpu_reference_grad(&raw, &grad_out_ones, 4);

    println!("    gpu grad: {g_gpu_data:?}");
    println!("    cpu ref : {expected:?}");

    let mut max_delta = 0.0f32;
    for (i, (a, b)) in g_gpu_data.iter().zip(expected.iter()).enumerate() {
        let d = (a - b).abs();
        if d > max_delta {
            max_delta = d;
        }
        println!("    [{i}] gpu={a:.6} ref={b:.6} delta={d:.6}");
    }
    println!("    max_delta = {max_delta:.6}");

    assert!(
        max_delta < 1e-5,
        "log_softmax f32 GPU backward delta {max_delta} exceeds 1e-5; \
         pre-fix #798 reported delta ~4.0 (kernel did exp() of an already-exp'd softmax buffer)"
    );
}

/// Wider (2, 8) f32 row coverage so the post-fix sentinel exercises the
/// per-row reduction more than once.
#[test]
fn log_softmax_grad_cuda_f32_two_rows() {
    ensure_cuda_backend();
    let raw: Vec<f32> = (0..16).map(|i| (i as f32) * 0.25 - 1.0).collect();
    let cpu_x = from_vec::<f32>(raw.clone(), &[2, 8]).expect("cpu input");

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

    let grad_out_ones = vec![1.0f32; 16];
    let expected = cpu_reference_grad(&raw, &grad_out_ones, 8);

    for (i, (a, b)) in g_gpu_data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "log_softmax (2,8) f32 grad mismatch at {i}: gpu={a}, cpu={b}"
        );
    }
}
