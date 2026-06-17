//! Regression coverage for CORE-051/#1745 and #1932: global `amax`/`amin`
//! must match PyTorch for empty inputs and NaN propagation.
//!
//! Live oracle, torch 2.11.0+cu130:
//! ```python
//! torch.amax(torch.tensor([]))
//! # RuntimeError: amax(): Expected reduction dim to be specified for input.numel() == 0.
//!
//! x = torch.tensor([1., float("nan"), 3.], requires_grad=True)
//! y = torch.amax(x)  # tensor(nan)
//! y.backward()
//! x.grad             # tensor([nan, nan, nan])
//! ```
//! `torch.amin` has the same empty-input and NaN-gradient behavior. CUDA
//! returns the same values with outputs and gradients on the input device.

use ferrotorch_core::autograd::backward;
use ferrotorch_core::creation::from_vec;
use ferrotorch_core::grad_fns::reduction::{amax, amin};
use ferrotorch_core::{FerrotorchError, Tensor};

fn expect_empty_extreme_error<T: std::fmt::Debug>(label: &str, result: Result<T, FerrotorchError>) {
    match result {
        Err(FerrotorchError::InvalidArgument { message }) => assert!(
            message.contains("requires an explicit dim"),
            "{label}: expected PyTorch empty-reduction error, got {message:?}"
        ),
        other => panic!("{label}: expected InvalidArgument, got {other:?}"),
    }
}

fn assert_all_nan_f32(label: &str, values: &[f32]) {
    assert!(
        values.iter().all(|v| v.is_nan()),
        "{label}: expected all NaN, got {values:?}"
    );
}

fn assert_all_nan_f64(label: &str, values: &[f64]) {
    assert!(
        values.iter().all(|v| v.is_nan()),
        "{label}: expected all NaN, got {values:?}"
    );
}

fn cpu_f32(values: Vec<f32>, shape: &[usize], requires_grad: bool) -> Tensor<f32> {
    from_vec(values, shape)
        .expect("f32 tensor")
        .requires_grad_(requires_grad)
}

fn cpu_f64(values: Vec<f64>, shape: &[usize], requires_grad: bool) -> Tensor<f64> {
    from_vec(values, shape)
        .expect("f64 tensor")
        .requires_grad_(requires_grad)
}

#[test]
fn cpu_global_amax_amin_reject_empty_like_torch() {
    let empty32 = cpu_f32(vec![], &[0], false);
    expect_empty_extreme_error("cpu f32 amax empty", amax(&empty32));
    expect_empty_extreme_error("cpu f32 amin empty", amin(&empty32));

    let empty64 = cpu_f64(vec![], &[0], false);
    expect_empty_extreme_error("cpu f64 amax empty", amax(&empty64));
    expect_empty_extreme_error("cpu f64 amin empty", amin(&empty64));
}

#[test]
fn cpu_global_amax_amin_nan_forward_backward_match_torch() {
    for op_name in ["amax", "amin"] {
        let x = cpu_f32(vec![1.0, f32::NAN, 3.0], &[3], true);
        let y = if op_name == "amax" {
            amax(&x).expect("amax f32")
        } else {
            amin(&x).expect("amin f32")
        };
        assert_all_nan_f32(
            &format!("cpu f32 {op_name} forward"),
            &y.data_vec().unwrap(),
        );
        backward(&y).expect("backward f32");
        let grad = x.grad().unwrap().expect("f32 grad");
        assert_all_nan_f32(
            &format!("cpu f32 {op_name} grad"),
            &grad.data_vec().unwrap(),
        );

        let x = cpu_f64(vec![1.0, f64::NAN, 3.0], &[3], true);
        let y = if op_name == "amax" {
            amax(&x).expect("amax f64")
        } else {
            amin(&x).expect("amin f64")
        };
        assert_all_nan_f64(
            &format!("cpu f64 {op_name} forward"),
            &y.data_vec().unwrap(),
        );
        backward(&y).expect("backward f64");
        let grad = x.grad().unwrap().expect("f64 grad");
        assert_all_nan_f64(
            &format!("cpu f64 {op_name} grad"),
            &grad.data_vec().unwrap(),
        );
    }
}

#[cfg(feature = "gpu")]
mod cuda {
    use super::*;
    use ferrotorch_core::Device;
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for extrema probes");
        });
    }

    fn cuda_f32(values: Vec<f32>, shape: &[usize], requires_grad: bool) -> Tensor<f32> {
        from_vec(values, shape)
            .expect("f32 CPU tensor")
            .to(Device::Cuda(0))
            .expect("upload f32")
            .requires_grad_(requires_grad)
    }

    fn cuda_f64(values: Vec<f64>, shape: &[usize], requires_grad: bool) -> Tensor<f64> {
        from_vec(values, shape)
            .expect("f64 CPU tensor")
            .to(Device::Cuda(0))
            .expect("upload f64")
            .requires_grad_(requires_grad)
    }

    fn host_f32(t: &Tensor<f32>) -> Vec<f32> {
        assert_eq!(
            t.device(),
            Device::Cuda(0),
            "tensor must stay CUDA-resident until explicit readback"
        );
        t.cpu().expect("D2H f32").data_vec().expect("f32 data")
    }

    fn host_f64(t: &Tensor<f64>) -> Vec<f64> {
        assert_eq!(
            t.device(),
            Device::Cuda(0),
            "tensor must stay CUDA-resident until explicit readback"
        );
        t.cpu().expect("D2H f64").data_vec().expect("f64 data")
    }

    #[test]
    fn cuda_global_amax_amin_reject_empty_before_backend_dispatch() {
        ensure_cuda_backend();

        let empty32 = cuda_f32(vec![], &[0], false);
        expect_empty_extreme_error("cuda f32 amax empty", amax(&empty32));
        expect_empty_extreme_error("cuda f32 amin empty", amin(&empty32));

        let empty64 = cuda_f64(vec![], &[0], false);
        expect_empty_extreme_error("cuda f64 amax empty", amax(&empty64));
        expect_empty_extreme_error("cuda f64 amin empty", amin(&empty64));
    }

    #[test]
    fn cuda_global_amax_amin_nan_forward_backward_match_torch_and_stay_resident() {
        ensure_cuda_backend();

        for op_name in ["amax", "amin"] {
            let x = cuda_f32(vec![1.0, f32::NAN, 3.0], &[3], true);
            let y = if op_name == "amax" {
                amax(&x).expect("cuda amax f32")
            } else {
                amin(&x).expect("cuda amin f32")
            };
            assert_all_nan_f32(&format!("cuda f32 {op_name} forward"), &host_f32(&y));
            backward(&y).expect("cuda backward f32");
            let grad = x.grad().unwrap().expect("cuda f32 grad");
            assert_all_nan_f32(&format!("cuda f32 {op_name} grad"), &host_f32(&grad));

            let x = cuda_f64(vec![1.0, f64::NAN, 3.0], &[3], true);
            let y = if op_name == "amax" {
                amax(&x).expect("cuda amax f64")
            } else {
                amin(&x).expect("cuda amin f64")
            };
            assert_all_nan_f64(&format!("cuda f64 {op_name} forward"), &host_f64(&y));
            backward(&y).expect("cuda backward f64");
            let grad = x.grad().unwrap().expect("cuda f64 grad");
            assert_all_nan_f64(&format!("cuda f64 {op_name} grad"), &host_f64(&grad));
        }
    }
}
