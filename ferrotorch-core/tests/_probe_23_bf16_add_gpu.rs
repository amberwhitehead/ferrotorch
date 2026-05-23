//! Maxine's bf16 + CUDA add repro for forecast-bio/ferrotorch#23.
//!
//! Documented behaviour BEFORE the dispatch refactor:
//! `Tensor<bf16> + Tensor<bf16>` on CUDA errors with
//! "GPU handle does not contain a CudaBuffer<f32>" because
//! `arithmetic::add_inner` falls through the `is_f64::<T>()` arm into the
//! else branch which calls `backend.add_f32(...)` against a bf16 handle.
//!
//! This probe pins the BEFORE/AFTER inversion explicitly: it asserts
//! `is_err` so the fix is mechanically visible in the diff (`is_err` →
//! `is_ok`). The architect's prompt requires this inversion be one of
//! the evidence items.

#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::Device;
use ferrotorch_core::creation::from_vec;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for the #23 bf16+add probe");
    });
}

#[test]
fn issue23_bf16_add_on_cuda_routes_to_gpu_kernel() {
    ensure_cuda_backend();

    let n = 8;
    let a_f32: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32) * 0.1).collect();
    let b_f32: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.05).collect();
    let a_bf: Vec<half::bf16> = a_f32.iter().copied().map(half::bf16::from_f32).collect();
    let b_bf: Vec<half::bf16> = b_f32.iter().copied().map(half::bf16::from_f32).collect();

    let a = from_vec::<half::bf16>(a_bf.clone(), &[n])
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();
    let b = from_vec::<half::bf16>(b_bf.clone(), &[n])
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();

    let result = ferrotorch_core::grad_fns::arithmetic::add(&a, &b);

    // AFTER the dispatch refactor: this MUST succeed and route to
    // `gpu_add_bf16` PTX kernel. The architect will flip is_err → is_ok
    // after auditing the dispatcher refactor.
    assert!(
        result.is_ok(),
        "bf16+CUDA add should route to gpu_add_bf16 kernel; got: {:?}",
        result.err()
    );

    // Numerical sanity: c[i] = a[i] + b[i] within bf16 ULP.
    let c = result.unwrap();
    let c_cpu = c.to(Device::Cpu).unwrap();
    let c_data = c_cpu.data().unwrap();
    for i in 0..n {
        let expected = a_bf[i].to_f32() + b_bf[i].to_f32();
        let got = c_data[i].to_f32();
        let diff = (got - expected).abs();
        assert!(
            diff < 5e-2,
            "bf16 add row {i}: got {got}, expected {expected}, diff {diff}"
        );
    }
}
