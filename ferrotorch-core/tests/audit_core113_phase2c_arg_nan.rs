//! CORE-113 (#1807): direct phase2c argmax/argmin must use PyTorch's
//! `GreaterOrNan` / `LessOrNan` contract and the same dimensional edge cases.
//!
//! Live oracle, torch 2.11.0+cu130:
//! ```text
//! torch.argmax(torch.tensor([1., nan, 3.])) == 1
//! torch.argmin(torch.tensor([1., nan, 3.])) == 1
//! torch.argmax(torch.tensor([1., nan, nan, 5.])) == 1
//! torch.argmax(torch.empty((0, 2)), dim=1).shape == torch.Size([0])
//! torch.argmax(torch.empty((2, 0, 3)), dim=1) raises IndexError
//! torch.argmax(torch.tensor(5.), dim=0) == 0
//! ```
//! Upstream source: `aten/src/ATen/native/SharedReduceOps.h` defines
//! `GreaterOrNan` / `LessOrNan`; `ReduceOps.cpp::argmax_argmin_impl` only
//! errors when the selected reduction dimension is empty.

use ferrotorch_core::error::FerrotorchError;
use ferrotorch_core::int_tensor::IntTensor;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

#[cfg(feature = "gpu")]
use ferrotorch_core::device::Device;

#[cfg(feature = "gpu")]
fn ensure_cuda_backend() {
    ferrotorch_gpu::init_cuda_backend().expect("CUDA backend init for CORE-113 suite");
}

fn f32_tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data), shape.to_vec(), false).unwrap()
}

fn read_i64(t: &IntTensor<i64>) -> Vec<i64> {
    t.data().unwrap().to_vec()
}

#[track_caller]
fn assert_invalid<T: std::fmt::Debug>(r: Result<T, FerrotorchError>, needle: &str) {
    match r {
        Err(FerrotorchError::InvalidArgument { message }) => {
            assert!(
                message.contains(needle),
                "error message {message:?} did not contain {needle:?}"
            );
        }
        Err(other) => panic!("expected InvalidArgument containing {needle:?}, got {other:?}"),
        Ok(v) => panic!("expected InvalidArgument containing {needle:?}, got Ok({v:?})"),
    }
}

#[test]
fn direct_tensor_argmax_argmin_cpu_first_nan_is_sticky() {
    let x = f32_tensor(vec![1.0, f32::NAN, f32::NAN, 5.0], &[4]);

    assert_eq!(read_i64(&x.argmax(None).unwrap()), vec![1]);
    assert_eq!(read_i64(&x.argmin(None).unwrap()), vec![1]);
}

#[test]
fn direct_tensor_argmax_argmin_cpu_dim_nan_per_slice() {
    let x = f32_tensor(
        vec![1.0, f32::NAN, 3.0, f32::NAN, 5.0, f32::NAN, 4.0, 2.0, 6.0],
        &[3, 3],
    );

    assert_eq!(read_i64(&x.argmax(Some(1)).unwrap()), vec![1, 0, 2]);
    assert_eq!(read_i64(&x.argmin(Some(1)).unwrap()), vec![1, 0, 1]);
    assert_eq!(read_i64(&x.argmax(Some(0)).unwrap()), vec![1, 0, 1]);
    assert_eq!(read_i64(&x.argmin(Some(0)).unwrap()), vec![1, 0, 1]);
}

#[test]
fn direct_tensor_argmax_argmin_cpu_reduced_precision_nan() {
    let f16 = Tensor::from_storage(
        TensorStorage::cpu(vec![
            half::f16::from_f32(1.0),
            half::f16::from_f32(f32::NAN),
            half::f16::from_f32(3.0),
        ]),
        vec![3],
        false,
    )
    .unwrap();
    let bf16 = Tensor::from_storage(
        TensorStorage::cpu(vec![
            half::bf16::from_f32(1.0),
            half::bf16::from_f32(f32::NAN),
            half::bf16::from_f32(3.0),
        ]),
        vec![3],
        false,
    )
    .unwrap();

    assert_eq!(read_i64(&f16.argmax(None).unwrap()), vec![1]);
    assert_eq!(read_i64(&f16.argmin(None).unwrap()), vec![1]);
    assert_eq!(read_i64(&bf16.argmax(None).unwrap()), vec![1]);
    assert_eq!(read_i64(&bf16.argmin(None).unwrap()), vec![1]);
}

#[test]
fn direct_tensor_arg_dim_scalar_and_zero_outer_match_torch() {
    let scalar = f32_tensor(vec![5.0], &[]);
    let zero_outer = f32_tensor(Vec::new(), &[0, 2]);

    let scalar_dim0 = scalar.argmax(Some(0)).unwrap();
    let scalar_dim_neg = scalar.argmin(Some(-1)).unwrap();
    assert_eq!(scalar_dim0.shape(), &[] as &[usize]);
    assert_eq!(scalar_dim_neg.shape(), &[] as &[usize]);
    assert_eq!(read_i64(&scalar_dim0), vec![0]);
    assert_eq!(read_i64(&scalar_dim_neg), vec![0]);

    let empty = zero_outer.argmax(Some(1)).unwrap();
    assert_eq!(empty.shape(), &[0]);
    assert!(read_i64(&empty).is_empty());
}

#[test]
fn direct_tensor_arg_dim_selected_zero_axis_errors() {
    let x = f32_tensor(Vec::new(), &[2, 0, 3]);

    assert_invalid(
        x.argmax(Some(1)).map(|t| read_i64(&t)),
        "Expected reduction dim 1 to have non-zero size",
    );
    assert_invalid(
        x.argmin(Some(1)).map(|t| read_i64(&t)),
        "Expected reduction dim 1 to have non-zero size",
    );
}

#[test]
fn direct_tensor_arg_global_empty_and_scalar_bad_dim_errors_match_torch() {
    let empty = f32_tensor(Vec::new(), &[0]);
    let scalar = f32_tensor(vec![5.0], &[]);

    assert_invalid(
        empty.argmax(None).map(|t| read_i64(&t)),
        "Expected reduction dim to be specified for input.numel() == 0",
    );
    assert_invalid(
        scalar.argmin(Some(1)).map(|t| read_i64(&t)),
        "Dimension out of range (expected to be in range of [-1, 0], but got 1)",
    );
}

#[test]
fn direct_inttensor_arg_dim_scalar_and_zero_outer_match_torch() {
    let scalar = IntTensor::from_vec(vec![7_i64], vec![]).unwrap();
    let zero_outer = IntTensor::from_vec(Vec::<i64>::new(), vec![0, 2]).unwrap();

    let scalar_dim0 = scalar.argmax(Some(0)).unwrap();
    let scalar_dim_neg = scalar.argmin(Some(-1)).unwrap();
    assert_eq!(scalar_dim0.shape(), &[] as &[usize]);
    assert_eq!(scalar_dim_neg.shape(), &[] as &[usize]);
    assert_eq!(read_i64(&scalar_dim0), vec![0]);
    assert_eq!(read_i64(&scalar_dim_neg), vec![0]);

    let empty = zero_outer.argmax(Some(1)).unwrap();
    assert_eq!(empty.shape(), &[0]);
    assert!(read_i64(&empty).is_empty());
}

#[test]
fn direct_inttensor_arg_global_empty_and_scalar_bad_dim_errors_match_torch() {
    let empty = IntTensor::from_vec(Vec::<i64>::new(), vec![0]).unwrap();
    let scalar = IntTensor::from_vec(vec![7_i64], vec![]).unwrap();

    assert_invalid(
        empty.argmin(None).map(|t| read_i64(&t)),
        "Expected reduction dim to be specified for input.numel() == 0",
    );
    assert_invalid(
        scalar.argmax(Some(-2)).map(|t| read_i64(&t)),
        "Dimension out of range (expected to be in range of [-1, 0], but got -2)",
    );
}

#[cfg(feature = "gpu")]
#[test]
fn direct_tensor_argmax_argmin_cuda_nan_outputs_stay_resident() {
    ensure_cuda_backend();
    let x = f32_tensor(vec![1.0, f32::NAN, f32::NAN, 5.0], &[4])
        .to(Device::Cuda(0))
        .unwrap();

    let max = x.argmax(None).unwrap();
    let min = x.argmin(None).unwrap();

    assert!(max.is_cuda(), "argmax result must stay CUDA-resident");
    assert!(min.is_cuda(), "argmin result must stay CUDA-resident");
    assert_eq!(read_i64(&max.to(Device::Cpu).unwrap()), vec![1]);
    assert_eq!(read_i64(&min.to(Device::Cpu).unwrap()), vec![1]);
}

#[cfg(feature = "gpu")]
#[test]
fn direct_tensor_argmax_argmin_cuda_dim_nan_per_slice() {
    ensure_cuda_backend();
    let x = f32_tensor(
        vec![1.0, f32::NAN, 3.0, f32::NAN, 5.0, f32::NAN, 4.0, 2.0, 6.0],
        &[3, 3],
    )
    .to(Device::Cuda(0))
    .unwrap();

    let max = x.argmax(Some(1)).unwrap();
    let min = x.argmin(Some(1)).unwrap();

    assert!(max.is_cuda(), "argmax(dim) result must stay CUDA-resident");
    assert!(min.is_cuda(), "argmin(dim) result must stay CUDA-resident");
    assert_eq!(read_i64(&max.to(Device::Cpu).unwrap()), vec![1, 0, 2]);
    assert_eq!(read_i64(&min.to(Device::Cpu).unwrap()), vec![1, 0, 1]);
}

#[cfg(feature = "gpu")]
#[test]
fn direct_tensor_argmax_argmin_cuda_reduced_precision_nan() {
    ensure_cuda_backend();
    let f16 = Tensor::from_storage(
        TensorStorage::cpu(vec![
            half::f16::from_f32(1.0),
            half::f16::from_f32(f32::NAN),
            half::f16::from_f32(3.0),
        ]),
        vec![3],
        false,
    )
    .unwrap()
    .to(Device::Cuda(0))
    .unwrap();
    let bf16 = Tensor::from_storage(
        TensorStorage::cpu(vec![
            half::bf16::from_f32(1.0),
            half::bf16::from_f32(f32::NAN),
            half::bf16::from_f32(3.0),
        ]),
        vec![3],
        false,
    )
    .unwrap()
    .to(Device::Cuda(0))
    .unwrap();

    assert_eq!(
        read_i64(&f16.argmax(None).unwrap().to(Device::Cpu).unwrap()),
        vec![1]
    );
    assert_eq!(
        read_i64(&f16.argmin(None).unwrap().to(Device::Cpu).unwrap()),
        vec![1]
    );
    assert_eq!(
        read_i64(&bf16.argmax(None).unwrap().to(Device::Cpu).unwrap()),
        vec![1]
    );
    assert_eq!(
        read_i64(&bf16.argmin(None).unwrap().to(Device::Cpu).unwrap()),
        vec![1]
    );
}

#[cfg(feature = "gpu")]
#[test]
fn direct_tensor_arg_dim_cuda_scalar_and_zero_outer_stay_resident() {
    ensure_cuda_backend();
    let scalar = f32_tensor(vec![5.0], &[]).to(Device::Cuda(0)).unwrap();
    let zero_outer = f32_tensor(Vec::new(), &[0, 2]).to(Device::Cuda(0)).unwrap();

    let scalar_dim0 = scalar.argmax(Some(0)).unwrap();
    let empty = zero_outer.argmax(Some(1)).unwrap();

    assert!(
        scalar_dim0.is_cuda(),
        "scalar dim result must stay CUDA-resident"
    );
    assert!(empty.is_cuda(), "empty dim result must stay CUDA-resident");
    assert_eq!(scalar_dim0.shape(), &[] as &[usize]);
    assert_eq!(read_i64(&scalar_dim0.to(Device::Cpu).unwrap()), vec![0]);
    assert_eq!(empty.shape(), &[0]);
    assert!(read_i64(&empty.to(Device::Cpu).unwrap()).is_empty());
}
