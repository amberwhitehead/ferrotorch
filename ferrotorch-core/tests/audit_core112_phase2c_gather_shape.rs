//! CORE-112 (#1806, CLASS-V) regression suite for `ops::phase2c::gather`.
//!
//! Live oracle, torch 2.11.0+cu130:
//! ```text
//! >>> x = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
//! >>> torch.gather(x, 1, torch.tensor([[1, 0]]))
//! tensor([[2., 1.]])
//! >>> xi = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)
//! >>> torch.gather(xi, 1, torch.tensor([[1, 0]]))
//! tensor([[2, 1]])
//! >>> torch.gather(x, 0, torch.tensor([[1], [0]]))
//! tensor([[4.],
//!         [1.]])
//! ```
//! PyTorch allows the index to be smaller on non-gather axes; see
//! `aten/src/ATen/native/ScatterGatherChecks.h:41-58`.

use ferrotorch_core::int_tensor::IntTensor;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

#[cfg(feature = "gpu")]
fn ensure_cuda_backend() {
    ferrotorch_gpu::init_cuda_backend().expect("CUDA backend init for CORE-112 suite");
}

#[test]
fn phase2c_tensor_gather_allows_smaller_non_axis_dim() {
    let input = Tensor::from_storage(
        TensorStorage::cpu(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]),
        vec![2, 3],
        false,
    )
    .unwrap();
    let index = IntTensor::from_vec(vec![1_i64, 0], vec![1, 2]).unwrap();

    let out = input
        .gather(1, &index)
        .expect("torch.gather accepts smaller non-axis index dimensions");

    assert_eq!(out.shape(), &[1, 2]);
    assert_eq!(out.data().unwrap(), &[2.0, 1.0]);
}

#[test]
fn phase2c_inttensor_gather_allows_smaller_non_axis_dim() {
    let input = IntTensor::from_vec(vec![1_i64, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
    let index = IntTensor::from_vec(vec![1_i64, 0], vec![1, 2]).unwrap();

    let out = input
        .gather(1, &index)
        .expect("torch.gather accepts smaller non-axis index dimensions");

    assert_eq!(out.shape(), &[1, 2]);
    assert_eq!(out.data().unwrap(), &[2, 1]);
}

#[test]
fn phase2c_tensor_gather_allows_smaller_trailing_non_axis_dim() {
    let input = Tensor::from_storage(
        TensorStorage::cpu(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]),
        vec![2, 3],
        false,
    )
    .unwrap();
    let index = IntTensor::from_vec(vec![1_i64, 0], vec![2, 1]).unwrap();

    let out = input
        .gather(0, &index)
        .expect("torch.gather accepts smaller non-axis index dimensions");

    assert_eq!(out.shape(), &[2, 1]);
    assert_eq!(out.data().unwrap(), &[4.0, 1.0]);
}

#[cfg(feature = "gpu")]
#[test]
fn phase2c_tensor_gather_cuda_smaller_non_axis_dim_returns_to_cuda() {
    use ferrotorch_core::device::Device;

    ensure_cuda_backend();
    let input = Tensor::from_storage(
        TensorStorage::cpu(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]),
        vec![2, 3],
        false,
    )
    .unwrap()
    .to(Device::Cuda(0))
    .unwrap();
    let index = IntTensor::from_vec(vec![1_i64, 0], vec![1, 2])
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();

    let out = input
        .gather(1, &index)
        .expect("CUDA gather must handle smaller non-axis index dimensions");

    assert!(out.is_cuda(), "gather result must stay CUDA-resident");
    assert_eq!(out.shape(), &[1, 2]);
    assert_eq!(out.to(Device::Cpu).unwrap().data().unwrap(), &[2.0, 1.0]);
}

#[cfg(feature = "gpu")]
#[test]
fn phase2c_inttensor_gather_cuda_smaller_non_axis_dim_returns_to_cuda() {
    use ferrotorch_core::device::Device;

    ensure_cuda_backend();
    let input = IntTensor::from_vec(vec![1_i64, 2, 3, 4, 5, 6], vec![2, 3])
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();
    let index = IntTensor::from_vec(vec![1_i64, 0], vec![1, 2])
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();

    let out = input
        .gather(1, &index)
        .expect("CUDA IntTensor gather must handle smaller non-axis dimensions");

    assert!(out.is_cuda(), "gather result must stay CUDA-resident");
    assert_eq!(out.shape(), &[1, 2]);
    assert_eq!(out.to(Device::Cpu).unwrap().data().unwrap(), &[2, 1]);
}
