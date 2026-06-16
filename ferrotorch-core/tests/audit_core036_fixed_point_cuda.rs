#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::Tensor;
use ferrotorch_core::autograd::fixed_point::fixed_point;
use ferrotorch_core::creation::full_like;
use ferrotorch_core::device::Device;
use ferrotorch_core::grad_fns::arithmetic::{add, mul};
use ferrotorch_core::storage::TensorStorage;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for CORE-036 fixed_point CUDA probes");
    });
}

fn leaf_scalar(val: f32, requires_grad: bool) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(vec![val]), vec![], requires_grad).unwrap()
}

#[test]
fn tracked_fixed_point_result_stays_cuda() {
    ensure_cuda_backend();

    let x0 = leaf_scalar(0.0, false).to(Device::Cuda(0)).unwrap();
    let b = leaf_scalar(1.0, true)
        .to(Device::Cuda(0))
        .unwrap()
        .requires_grad_(true);

    let x_star = fixed_point(
        |x, params| {
            let half = full_like(x, 0.5f32)?;
            let half_x = mul(x, &half)?;
            add(&half_x, params[0])
        },
        &x0,
        &[&b],
        1000,
        1e-5,
    )
    .unwrap();

    assert_eq!(x_star.device(), Device::Cuda(0));
    let value = x_star.cpu().unwrap().item().unwrap();
    assert!(
        (value - 2.0).abs() < 1e-3,
        "tracked CUDA fixed point should be near 2, got {value}"
    );
}
