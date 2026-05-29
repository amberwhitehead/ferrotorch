//! Re-audit companion for commit 7c8b87118: the shifted-Chebyshev family
//! `shifted_chebyshev_polynomial_{t,u,v,w}` had its domain shift rewritten from
//! the f64 literal form `2.0 * x - 1.0` to the native-`T` form `x + x - one`
//! inside the `elementwise_native` closure. This pins that the shift is still
//! numerically correct vs LIVE torch on the [0,1] domain, CPU == GPU == torch.
//!
//! R-CHAR-3: expected values are LIVE `torch 2.11.0+cu130` outputs
//! (`torch.special.shifted_chebyshev_polynomial_*`, dtype=float64), NOT copied
//! from ferrotorch. PASSING positive control.

#![cfg(feature = "cuda")]

use std::sync::Once;

use ferrotorch_core::{Device, Tensor, TensorStorage, special};
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

fn cpu_f64(data: &[f64]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false).unwrap()
}

#[test]
fn reaudit_shifted_chebyshev_native_t_shift_cpu_gpu_torch_f64() {
    if !ensure_cuda() {
        return;
    }
    type Fn64 = fn(&Tensor<f64>, usize) -> ferrotorch_core::FerrotorchResult<Tensor<f64>>;
    // LIVE torch 2.11.0+cu130, dtype=float64, on the [0,1] shifted domain.
    let cases: [(&str, Fn64, f64, usize, f64); 4] = [
        (
            "sh_t",
            special::shifted_chebyshev_polynomial_t,
            0.6,
            10,
            0.42845562879999977,
        ),
        (
            "sh_u",
            special::shifted_chebyshev_polynomial_u,
            0.6,
            10,
            0.6128946175999997,
        ),
        (
            "sh_v",
            special::shifted_chebyshev_polynomial_v,
            0.4,
            9,
            -0.6781783039999996,
        ),
        (
            "sh_w",
            special::shifted_chebyshev_polynomial_w,
            0.4,
            9,
            -1.166211584,
        ),
    ];
    for (name, f, x, n, torch_val) in cases {
        let cpu = f(&cpu_f64(&[x]), n).unwrap().data().unwrap()[0];
        let g = cpu_f64(&[x]).to(Device::Cuda(0)).expect("to GPU");
        let gpu = f(&g, n).unwrap().to(Device::Cpu).unwrap().data().unwrap()[0];
        assert!(
            (cpu - torch_val).abs() <= 1e-12 && (gpu - torch_val).abs() <= 1e-12,
            "{name}(x={x}, n={n}) f64: torch={torch_val}, CPU={cpu} (d={:.2e}), \
             GPU={gpu} (d={:.2e}) — native-T shift regressed",
            (cpu - torch_val).abs(),
            (gpu - torch_val).abs()
        );
    }
}
