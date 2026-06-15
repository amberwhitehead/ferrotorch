//! End-to-end CUDA coverage for dim-aware `topk`.
//!
//! These tests exercise the public `ferrotorch_core::topk_dim` path, not just
//! the raw PTX launcher: CUDA tensors dispatch through `GpuBackend::topk_nd`,
//! values remain GPU-resident, and only the int64 indices are decoded to the
//! public `Vec<usize>`.

#![cfg(feature = "cuda")]

use ferrotorch_core::{Device, Tensor, TensorStorage, topk_dim};
use ferrotorch_gpu::init_cuda_backend;

fn ensure_cuda() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        init_cuda_backend().expect("CUDA backend init");
    });
}

fn cpu_f32(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
    Tensor::from_storage(
        TensorStorage::cpu(data.to_vec()),
        shape.to_vec(),
        requires_grad,
    )
    .unwrap()
}

#[test]
fn topk_dim_f32_cuda_middle_axis_values_resident_and_indices_match_torch() {
    ensure_cuda();
    // Live torch 2.11.0+cu130:
    // torch.topk(x.reshape(2,3,4), 2, dim=1, largest=True, sorted=True)
    // values flatten:
    // [8,7,6,9, 4,5,3,6, 5,10,11,11, 2,5,5,3]
    // indices flatten:
    // [2,2,2,1, 1,0,0,2, 1,0,2,2, 0,1,1,0]
    let data = [
        1.0_f32, 5.0, 3.0, 2.0, 4.0, 4.0, 0.0, 9.0, 8.0, 7.0, 6.0, 6.0, 2.0, 10.0, -1.0, 3.0, 5.0,
        5.0, 5.0, 1.0, 0.0, -2.0, 11.0, 11.0,
    ];
    let input = cpu_f32(&data, &[2, 3, 4], false)
        .to(Device::Cuda(0))
        .expect("upload f32");

    let (values, indices) = topk_dim(&input, 2, 1, true).expect("topk_dim f32 cuda");

    assert_eq!(values.device(), Device::Cuda(0));
    assert_eq!(values.shape(), &[2, 2, 4]);
    let host = values
        .to(Device::Cpu)
        .expect("read values")
        .data()
        .expect("host values")
        .to_vec();
    assert_eq!(
        host,
        vec![
            8.0, 7.0, 6.0, 9.0, 4.0, 5.0, 3.0, 6.0, 5.0, 10.0, 11.0, 11.0, 2.0, 5.0, 5.0, 3.0,
        ]
    );
    assert_eq!(
        indices,
        vec![2, 2, 2, 1, 1, 0, 0, 2, 1, 0, 2, 2, 0, 1, 1, 0]
    );
}

#[test]
fn topk_dim_f32_cuda_scalar_k_zero_matches_torch_special_case() {
    ensure_cuda();
    // Live torch 2.11.0+cu130:
    // torch.topk(torch.tensor(7., device='cuda'), k=0, dim=0) returns scalar
    // value 7 and scalar index 0, not an empty tensor.
    let input = cpu_f32(&[7.0], &[], true)
        .to(Device::Cuda(0))
        .expect("upload scalar")
        .detach()
        .requires_grad_(true);

    let (values, indices) = topk_dim(&input, 0, 0, true).expect("topk_dim scalar cuda");

    assert_eq!(values.device(), Device::Cuda(0));
    assert_eq!(values.shape(), &[] as &[usize]);
    assert!(
        values.requires_grad(),
        "scalar topk values must keep the torch TopkBackward-style edge"
    );
    let host = values
        .to(Device::Cpu)
        .expect("read value")
        .data()
        .expect("host scalar")
        .to_vec();
    assert_eq!(host, vec![7.0]);
    assert_eq!(indices, vec![0]);
}
