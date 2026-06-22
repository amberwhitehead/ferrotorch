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

#[test]
fn phase2c_tensor_gather_empty_index_skips_shape_checks_like_torch() {
    let input = Tensor::from_storage(
        TensorStorage::cpu(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]),
        vec![2, 3],
        false,
    )
    .unwrap();
    let index = IntTensor::<i64>::from_vec(Vec::new(), vec![999, 0]).unwrap();

    let out = input
        .gather(1, &index)
        .expect("torch.gather returns empty output before shape checks");

    assert_eq!(out.shape(), &[999, 0]);
    assert!(out.data().unwrap().is_empty());
}

#[test]
fn phase2c_inttensor_gather_empty_index_skips_shape_checks_like_torch() {
    let input = IntTensor::from_vec(vec![1_i64, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
    let index = IntTensor::<i64>::from_vec(Vec::new(), vec![999, 0]).unwrap();

    let out = input
        .gather(1, &index)
        .expect("torch.gather returns empty output before shape checks");

    assert_eq!(out.shape(), &[999, 0]);
    assert!(out.data().unwrap().is_empty());
}

#[test]
fn phase2c_tensor_gather_scalar_input_uses_nonempty_dim_contract() {
    let input = Tensor::from_storage(TensorStorage::cpu(vec![5.0_f32]), vec![], false).unwrap();
    let scalar_index = IntTensor::from_vec(vec![0_i64], vec![]).unwrap();
    let vector_index = IntTensor::from_vec(vec![0_i64, 0, 0], vec![3]).unwrap();

    let scalar = input.gather(0, &scalar_index).unwrap();
    let vector = input.gather(-1, &vector_index).unwrap();

    assert_eq!(scalar.shape(), &[] as &[usize]);
    assert_eq!(scalar.data().unwrap(), &[5.0]);
    assert_eq!(vector.shape(), &[3]);
    assert_eq!(vector.data().unwrap(), &[5.0, 5.0, 5.0]);
}

#[test]
fn phase2c_tensor_index_select_scalar_input_matches_torch_contract() {
    use ferrotorch_core::autograd::graph::backward;

    let input = Tensor::from_storage(TensorStorage::cpu(vec![5.0_f32]), vec![], false)
        .unwrap()
        .requires_grad_(true);
    let scalar_index = IntTensor::from_vec(vec![0_i64], vec![]).unwrap();
    let one_element_index = IntTensor::from_vec(vec![0_i64], vec![1]).unwrap();

    let scalar = input.index_select(0, &scalar_index).unwrap();
    let one_element = input.index_select(-1, &one_element_index).unwrap();

    assert_eq!(scalar.shape(), &[] as &[usize]);
    assert_eq!(scalar.data().unwrap(), &[5.0]);
    assert_eq!(one_element.shape(), &[] as &[usize]);
    assert_eq!(one_element.data().unwrap(), &[5.0]);

    backward(&one_element).unwrap();
    let grad = input
        .grad()
        .unwrap()
        .expect("scalar index_select must accumulate scalar grad");
    assert_eq!(grad.shape(), &[] as &[usize]);
    assert_eq!(grad.data().unwrap(), &[1.0]);
}

#[test]
fn phase2c_inttensor_index_select_scalar_input_matches_torch_contract() {
    let input = IntTensor::from_vec(vec![5_i64], vec![]).unwrap();
    let scalar_index = IntTensor::from_vec(vec![0_i64], vec![]).unwrap();
    let one_element_index = IntTensor::from_vec(vec![0_i64], vec![1]).unwrap();

    let scalar = input.index_select(0, &scalar_index).unwrap();
    let one_element = input.index_select(-1, &one_element_index).unwrap();

    assert_eq!(scalar.shape(), &[] as &[usize]);
    assert_eq!(scalar.data().unwrap(), &[5]);
    assert_eq!(one_element.shape(), &[] as &[usize]);
    assert_eq!(one_element.data().unwrap(), &[5]);
}

#[test]
fn phase2c_index_select_scalar_input_rejects_non_singleton_index_like_torch() {
    let input = Tensor::from_storage(TensorStorage::cpu(vec![5.0_f32]), vec![], false).unwrap();
    let int_input = IntTensor::from_vec(vec![5_i64], vec![]).unwrap();
    let empty = IntTensor::<i64>::from_vec(Vec::new(), vec![0]).unwrap();
    let two = IntTensor::from_vec(vec![0_i64, 0], vec![2]).unwrap();

    assert!(input.index_select(0, &empty).is_err());
    assert!(input.index_select(0, &two).is_err());
    assert!(int_input.index_select(0, &empty).is_err());
    assert!(int_input.index_select(0, &two).is_err());
}

#[test]
fn phase2c_tensor_gather_empty_tracked_backward_is_zero() {
    use ferrotorch_core::autograd::graph::backward;

    let input = Tensor::from_storage(
        TensorStorage::cpu(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]),
        vec![2, 3],
        false,
    )
    .unwrap()
    .requires_grad_(true);
    let index = IntTensor::<i64>::from_vec(Vec::new(), vec![999, 0]).unwrap();
    let out = input.gather(1, &index).unwrap();

    backward(&out.sum_all().unwrap()).unwrap();

    let grad = input.grad().unwrap().expect("empty gather still has a VJP");
    assert_eq!(grad.shape(), &[2, 3]);
    assert_eq!(grad.data().unwrap(), &[0.0; 6]);
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
fn phase2c_tensor_gather_cuda_empty_index_returns_empty_cuda() {
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
    let index = IntTensor::<i64>::from_vec(Vec::new(), vec![999, 0])
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();

    let out = input
        .gather(1, &index)
        .expect("CUDA empty gather must mirror torch's early return");

    assert!(out.is_cuda(), "empty gather result must stay CUDA-resident");
    assert_eq!(out.shape(), &[999, 0]);
    assert!(out.to(Device::Cpu).unwrap().data().unwrap().is_empty());
}

#[cfg(feature = "gpu")]
#[test]
fn phase2c_tensor_index_select_cuda_scalar_input_stays_resident() {
    use ferrotorch_core::autograd::graph::backward;
    use ferrotorch_core::device::Device;

    ensure_cuda_backend();
    let input = Tensor::from_storage(TensorStorage::cpu(vec![5.0_f32]), vec![], false)
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap()
        .requires_grad_(true);
    let index = IntTensor::from_vec(vec![0_i64], vec![1])
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();

    let out = input
        .index_select(0, &index)
        .expect("CUDA scalar index_select must match torch");

    assert!(
        out.is_cuda(),
        "scalar index_select output must stay CUDA-resident"
    );
    assert_eq!(out.shape(), &[] as &[usize]);
    assert_eq!(out.to(Device::Cpu).unwrap().data().unwrap(), &[5.0]);

    backward(&out).unwrap();
    let grad = input.grad().unwrap().expect("scalar CUDA grad");
    assert!(
        grad.is_cuda(),
        "scalar index_select backward must stay CUDA-resident"
    );
    assert_eq!(grad.shape(), &[] as &[usize]);
    assert_eq!(grad.to(Device::Cpu).unwrap().data().unwrap(), &[1.0]);
}

#[cfg(feature = "gpu")]
#[test]
fn phase2c_inttensor_index_select_cuda_scalar_input_stays_resident() {
    use ferrotorch_core::device::Device;

    ensure_cuda_backend();
    let input = IntTensor::from_vec(vec![5_i64], vec![])
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();
    let index = IntTensor::from_vec(vec![0_i64], vec![1])
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();

    let out = input
        .index_select(0, &index)
        .expect("CUDA IntTensor scalar index_select must match torch");

    assert!(
        out.is_cuda(),
        "IntTensor scalar index_select output must stay CUDA-resident"
    );
    assert_eq!(out.shape(), &[] as &[usize]);
    assert_eq!(out.to(Device::Cpu).unwrap().data().unwrap(), &[5]);
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

#[cfg(feature = "gpu")]
#[test]
fn phase2c_inttensor_gather_cuda_empty_index_returns_empty_cuda() {
    use ferrotorch_core::device::Device;

    ensure_cuda_backend();
    let input = IntTensor::from_vec(vec![1_i64, 2, 3, 4, 5, 6], vec![2, 3])
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();
    let index = IntTensor::<i64>::from_vec(Vec::new(), vec![999, 0])
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();

    let out = input
        .gather(1, &index)
        .expect("CUDA IntTensor empty gather must mirror torch's early return");

    assert!(out.is_cuda(), "empty gather result must stay CUDA-resident");
    assert_eq!(out.shape(), &[999, 0]);
    assert!(out.to(Device::Cpu).unwrap().data().unwrap().is_empty());
}
