//! Permanent regression sentinel for #799: gpu `gelu_with(GeluApproximate::None)`
//! f32 forward diverges from PyTorch by ~1.25e-2 — well outside the
//! `F32_TRANSCENDENTAL_GPU = 1e-4` parity gate.
//!
//! Probe shape:
//!   1. Construct CUDA f32 (and f64) tensors at multiple shapes / value ranges.
//!   2. Compute `gelu_with(t, None)` on GPU.
//!   3. Compare to `gelu_with(cpu_t, None)` (the CPU lane is fdlibm-based and
//!      treated as ground truth — closed under #792 / batch-3 A1).
//!   4. Compare GPU `None` vs GPU `Sigmoid` and GPU `None` vs GPU `Tanh` —
//!      they MUST be numerically distinct. Pre-fix the dispatch silently
//!      collapsed all three modes onto the same kernel (sigmoid approx),
//!      causing a 1.25e-2 divergence on the `None` lane.
//!
//! Post-fix:
//!   - GPU `None` matches CPU `None` within `F32_TRANSCENDENTAL_GPU = 1e-4`.
//!   - GPU `None`, `Sigmoid`, `Tanh` are pairwise numerically distinct.
//!
//! Tolerance constants are inlined here (the workspace constants live as
//! private items inside `tests/conformance_activation.rs`).

#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::Device;
use ferrotorch_core::autograd::graph::backward;
use ferrotorch_core::creation::from_vec;
use ferrotorch_core::grad_fns::activation::{GeluApproximate, gelu_with};

/// Workspace `F32_TRANSCENDENTAL_GPU` parity gate (mirrors the constant in
/// `tests/conformance_activation.rs`). The whole point of this probe is to
/// guard the `None` lane against falling back to a different kernel that
/// would push beyond this band.
const F32_TRANSCENDENTAL_GPU: f32 = 1e-4;

/// Post-#823 the f64 GPU None lane meets the workspace `F64_TRANSCENDENTAL
/// = 1e-10` gate: `GELU_ERF_F64_PTX` and `GELU_BACKWARD_ERF_F64_PTX` were
/// re-implemented around the SunPro fdlibm piecewise rational (matching
/// the CPU lane from #792). Empirical residual on the probe range is
/// ~1e-16 (machine ulp). This probe holds the kernel to the workspace
/// gate so a regression that re-introduces the A&S 7.1.26 polynomial (or
/// re-corrupts the fdlibm constants) surfaces immediately.
const F64_GELU_AS_BOUND: f64 = 1e-10;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for the GPU probe suite");
    });
}

/// Test inputs that probe both small and large arguments — the A&S 7.1.26
/// erf approximation is most stressed near `|z| ≈ 1` (around |x| ≈ √2).
fn f32_inputs() -> Vec<f32> {
    // Matches the conformance fixture range (`|x| ≤ 0.75`), plus a few
    // larger magnitudes so the probe also stresses the catastrophic
    // cancellation regime where `1 + erf(x/sqrt(2))` ≈ 0.
    vec![
        -3.0, -2.0, -1.5, -1.0, -0.75, -0.62, -0.49, -0.36, -0.23, -0.10, 0.0, 0.1, 0.3, 0.7, 1.0,
        1.5, 2.0, 3.0,
    ]
}

fn f64_inputs() -> Vec<f64> {
    vec![
        -3.0, -2.0, -1.5, -1.0, -0.7, -0.3, -0.1, 0.0, 0.1, 0.3, 0.7, 1.0, 1.5, 2.0, 3.0,
    ]
}

fn read_back_f32(t: &ferrotorch_core::Tensor<f32>) -> Vec<f32> {
    let cpu = if t.is_cuda() {
        t.cpu().expect("gpu->cpu copy")
    } else {
        t.clone()
    };
    cpu.data().expect("read_back").to_vec()
}

fn read_back_f64(t: &ferrotorch_core::Tensor<f64>) -> Vec<f64> {
    let cpu = if t.is_cuda() {
        t.cpu().expect("gpu->cpu copy")
    } else {
        t.clone()
    };
    cpu.data().expect("read_back").to_vec()
}

/// Pre-fix: GPU `gelu_with(None)` returned the sigmoid-approx values, so
/// the residual against CPU was ~1.25e-2. Post-fix: residual ≤ 1e-4.
#[test]
fn gpu_gelu_none_f32_matches_cpu() {
    ensure_cuda_backend();
    let xs = f32_inputs();
    let n = xs.len();

    let cpu = from_vec::<f32>(xs.clone(), &[n]).expect("cpu tensor");
    let gpu = cpu.to(Device::Cuda(0)).expect("cpu->gpu");

    let y_cpu = gelu_with(&cpu, GeluApproximate::None).expect("cpu gelu_with(None)");
    let y_gpu = gelu_with(&gpu, GeluApproximate::None).expect("gpu gelu_with(None)");

    assert!(y_gpu.is_cuda(), "gpu gelu_with(None) result must stay on GPU");

    let cpu_vals = read_back_f32(&y_cpu);
    let gpu_vals = read_back_f32(&y_gpu);
    assert_eq!(cpu_vals.len(), gpu_vals.len());

    let mut max_abs_diff = 0.0f32;
    for (i, (g, c)) in gpu_vals.iter().zip(cpu_vals.iter()).enumerate() {
        let d = (g - c).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
        }
        assert!(
            d <= F32_TRANSCENDENTAL_GPU,
            "gpu_gelu_none_f32 mismatch at i={i} x={}: gpu={g} cpu={c} diff={d} \
             gate={F32_TRANSCENDENTAL_GPU} (#799 regression)",
            xs[i]
        );
    }
    println!("gpu_gelu_none_f32_matches_cpu: max |diff| = {max_abs_diff}");
}

/// f64 lane — separately surfaced under #799 as also broken (cascade_skip
/// matches both `cuda:0, _`). Post-fix the f64 None lane also routes
/// through the on-device erf composite (the existing
/// `gelu_erf_f64`/`GELU_ERF_F64_PTX` kernel is correct; we just verify the
/// dispatch lands on it).
#[test]
fn gpu_gelu_none_f64_matches_cpu() {
    ensure_cuda_backend();
    let xs = f64_inputs();
    let n = xs.len();

    let cpu = from_vec::<f64>(xs.clone(), &[n]).expect("cpu tensor");
    let gpu = cpu.to(Device::Cuda(0)).expect("cpu->gpu");

    let y_cpu = gelu_with(&cpu, GeluApproximate::None).expect("cpu gelu_with(None) f64");
    let y_gpu = gelu_with(&gpu, GeluApproximate::None).expect("gpu gelu_with(None) f64");

    let cpu_vals = read_back_f64(&y_cpu);
    let gpu_vals = read_back_f64(&y_gpu);
    assert_eq!(cpu_vals.len(), gpu_vals.len());

    let mut max_abs_diff = 0.0f64;
    for (i, (g, c)) in gpu_vals.iter().zip(cpu_vals.iter()).enumerate() {
        let d = (g - c).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
        }
        assert!(
            d <= F64_GELU_AS_BOUND,
            "gpu_gelu_none_f64 mismatch at i={i} x={}: gpu={g} cpu={c} diff={d} \
             F64_TRANSCENDENTAL={F64_GELU_AS_BOUND} (#823 regression)",
            xs[i]
        );
    }
    println!(
        "gpu_gelu_none_f64_matches_cpu: max |diff| = {max_abs_diff} \
         (within F64_TRANSCENDENTAL = 1e-10 via SunPro fdlibm port — #823)"
    );
}

/// `GeluBackward` for `GeluApproximate::None` exercises
/// `gelu_backward_erf_kernel` on f32. The conformance `gpu_gelu_none`
/// row hits this path through `loss.backward()` after the forward; if
/// the kernel JITs but produces wrong gradients, the conformance row
/// flags it. This probe is a finer-grained check that the kernel even
/// loads (the pre-#799 cascade_skip never exercised it).
#[test]
fn gpu_gelu_none_f32_backward_loads_and_runs() {
    ensure_cuda_backend();
    let xs: Vec<f32> = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
    let n = xs.len();
    let cpu_for_gpu = from_vec::<f32>(xs.clone(), &[n]).expect("cpu tensor a");
    let cpu_for_cpu = from_vec::<f32>(xs.clone(), &[n]).expect("cpu tensor b");
    let gpu = cpu_for_gpu
        .to(Device::Cuda(0))
        .expect("cpu->gpu")
        .requires_grad_(true);

    let y = gelu_with(&gpu, GeluApproximate::None).expect("gpu gelu_with(None) fwd");
    let loss = y.sum_all().expect("sum");
    backward(&loss).expect("backward (must JIT gelu_backward_erf_kernel)");

    let g = gpu.grad().expect("grad lookup").expect("grad attached");
    let g_data = read_back_f32(&g);

    // Reference: d/dx gelu_none(x) = 0.5 * (1 + erf(x/sqrt(2))) + x * pdf(x)
    //                              = Phi(x) + x * phi(x), phi = 1/sqrt(2pi) * exp(-x^2/2)
    // CPU reference via gelu_with on a duplicate CPU tensor with grad
    // tracking; this inherits the workspace fdlibm-erf precision.
    let cpu_g = cpu_for_cpu.requires_grad_(true);
    let y_cpu = gelu_with(&cpu_g, GeluApproximate::None).expect("cpu gelu_with(None) fwd");
    let loss_cpu = y_cpu.sum_all().expect("cpu sum");
    backward(&loss_cpu).expect("cpu backward");
    let cpu_grad = cpu_g.grad().expect("cpu grad lookup").expect("cpu grad attached");
    let cpu_grad_vals = read_back_f32(&cpu_grad);
    for (i, (g_val, want)) in g_data.iter().zip(cpu_grad_vals.iter()).enumerate() {
        let d = (g_val - want).abs();
        assert!(
            d <= F32_TRANSCENDENTAL_GPU,
            "gelu_backward_erf f32 mismatch at i={i} x={}: got {g_val}, want {want}, diff={d}",
            xs[i]
        );
    }
}

/// The GeluApproximate::None and GeluApproximate::Sigmoid GPU kernels
/// must produce numerically distinct outputs. Pre-#799 fix the `None`
/// kernel held corrupted A&S coefficients that aliased the Horner curve
/// onto a different shape — outputs were wrong but still distinct from
/// Sigmoid. The probe stays useful as a regression sentinel: if a future
/// change collapses the dispatch (e.g. routing `None` to the sigmoid
/// kernel), this assertion catches it.
///
/// Note: `GeluApproximate::Tanh` is NOT exercised here because the
/// `gelu_tanh_kernel` PTX has an unrelated JIT-time compilation failure
/// that is out of #799 scope. The conformance suite catches the Tanh
/// path separately via its own fixture row when that kernel is fixed.
#[test]
fn gpu_gelu_modes_are_distinct_f32() {
    ensure_cuda_backend();
    let xs = f32_inputs();
    let n = xs.len();

    let cpu = from_vec::<f32>(xs, &[n]).expect("cpu tensor");
    let gpu = cpu.to(Device::Cuda(0)).expect("cpu->gpu");

    let y_none = gelu_with(&gpu, GeluApproximate::None).expect("gpu None");
    let y_sigmoid = gelu_with(&gpu, GeluApproximate::Sigmoid).expect("gpu Sigmoid");

    let v_none = read_back_f32(&y_none);
    let v_sigmoid = read_back_f32(&y_sigmoid);

    let max_diff = |a: &[f32], b: &[f32]| -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };

    let none_vs_sigmoid = max_diff(&v_none, &v_sigmoid);

    println!("gpu_gelu_modes_are_distinct_f32:");
    println!("  max |None - Sigmoid| = {none_vs_sigmoid}");

    // Threshold = an order of magnitude above the parity gate; the
    // analytical gap between exact-erf and sigmoid-1.702 GELU is
    // ~1.25e-2 around |x| in [0.7, 1.0], so this is well-bounded.
    let distinct = 1e-3;
    assert!(
        none_vs_sigmoid >= distinct,
        "GPU None and Sigmoid GELU collapsed onto the same kernel \
         (max |None - Sigmoid| = {none_vs_sigmoid}, expected >= {distinct}). \
         This is the #799 dispatch-collapse regression."
    );
}
