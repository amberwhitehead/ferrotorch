//! CORE-130 (#1824): public `masked_select` must use PyTorch broadcast
//! semantics, not equal-numel flat pairing.
//!
//! PyTorch source contract:
//! - CPU: `aten/src/ATen/native/TensorAdvancedIndexing.cpp` calls
//!   `expand_outplace(mask, self)` before compaction.
//! - CUDA: `aten/src/ATen/native/cuda/IndexKernel.cpp` also expands both
//!   operands before `index_out`.
//! - Backward: `masked_select_backward` scatters into
//!   `zeros_like(input.expand(infer_size(input.sizes(), mask.sizes())))`, so
//!   gradients reduce through broadcasted input dimensions.

use ferrotorch_core::autograd::graph::backward;
use ferrotorch_core::{BoolTensor, FerrotorchError, Tensor, TensorStorage, masked_select};

fn cpu_f32(data: &[f32], shape: &[usize], rg: bool) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), rg).unwrap()
}

#[track_caller]
fn assert_shape_err<T: std::fmt::Debug>(r: Result<T, FerrotorchError>, label: &str) {
    match r {
        Err(FerrotorchError::ShapeMismatch { .. } | FerrotorchError::InvalidArgument { .. }) => {}
        Err(other) => panic!("{label}: expected shape error, got {other:?}"),
        Ok(value) => panic!("{label}: expected shape error, got Ok({value:?})"),
    }
}

#[test]
fn core130_public_masked_select_rejects_equal_numel_incompatible_shapes() {
    let input = cpu_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], false);
    let mask = BoolTensor::from_vec(vec![true, false, true, false], vec![4]).unwrap();

    assert_shape_err(masked_select(&input, &mask), "free function");
    assert_shape_err(input.masked_select(&mask), "tensor method");
}

#[test]
fn core130_public_masked_select_broadcasts_mask_against_input() {
    let input = cpu_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], false);
    let mask = BoolTensor::from_vec(vec![false, true], vec![1, 2]).unwrap();

    let out = masked_select(&input, &mask).unwrap();
    assert_eq!(out.shape(), &[2]);
    assert_eq!(out.data_vec().unwrap(), vec![2.0, 4.0]);

    let method_out = input.masked_select(&mask).unwrap();
    assert_eq!(method_out.shape(), &[2]);
    assert_eq!(method_out.data_vec().unwrap(), vec![2.0, 4.0]);
}

#[test]
fn core130_public_masked_select_broadcasted_input_backward_reduces() {
    let input = cpu_f32(&[10.0, 20.0], &[1, 2], true);
    let mask = BoolTensor::from_vec(vec![true, false, true, true], vec![2, 2]).unwrap();

    let out = input.masked_select(&mask).unwrap();
    assert_eq!(out.shape(), &[3]);
    assert_eq!(out.data_vec().unwrap(), vec![10.0, 10.0, 20.0]);

    backward(&out.sum_all().unwrap()).unwrap();
    let grad = input
        .grad()
        .unwrap()
        .expect("grad must reach original input");
    assert_eq!(grad.shape(), &[1, 2]);
    assert_eq!(grad.data_vec().unwrap(), vec![2.0, 1.0]);
}

#[cfg(feature = "gpu")]
mod cuda {
    use super::*;
    use ferrotorch_core::Device;
    use ferrotorch_core::autograd::graph::backward_with_grad;
    use half::{bf16, f16};
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for CORE-130 probes");
        });
    }

    fn cuda_f32(data: &[f32], shape: &[usize], rg: bool) -> Tensor<f32> {
        cpu_f32(data, shape, false)
            .to(Device::Cuda(0))
            .unwrap()
            .requires_grad_(rg)
    }

    fn cuda_mask(data: Vec<bool>, shape: &[usize]) -> BoolTensor {
        BoolTensor::from_vec(data, shape.to_vec())
            .unwrap()
            .to(Device::Cuda(0))
            .unwrap()
    }

    fn cuda_f16(data: &[f32], shape: &[usize], rg: bool) -> Tensor<f16> {
        let values: Vec<f16> = data.iter().copied().map(f16::from_f32).collect();
        Tensor::from_storage(TensorStorage::cpu(values), shape.to_vec(), false)
            .unwrap()
            .to(Device::Cuda(0))
            .unwrap()
            .requires_grad_(rg)
    }

    fn cuda_bf16(data: &[f32], shape: &[usize], rg: bool) -> Tensor<bf16> {
        let values: Vec<bf16> = data.iter().copied().map(bf16::from_f32).collect();
        Tensor::from_storage(TensorStorage::cpu(values), shape.to_vec(), false)
            .unwrap()
            .to(Device::Cuda(0))
            .unwrap()
            .requires_grad_(rg)
    }

    fn cuda_f16_ones(shape: &[usize]) -> Tensor<f16> {
        let numel: usize = shape.iter().product();
        Tensor::from_storage(
            TensorStorage::cpu(vec![f16::from_f32(1.0); numel]),
            shape.to_vec(),
            false,
        )
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap()
    }

    fn cuda_bf16_ones(shape: &[usize]) -> Tensor<bf16> {
        let numel: usize = shape.iter().product();
        Tensor::from_storage(
            TensorStorage::cpu(vec![bf16::from_f32(1.0); numel]),
            shape.to_vec(),
            false,
        )
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap()
    }

    #[test]
    fn core130_cuda_masked_select_broadcasts_and_stays_resident() {
        ensure_cuda_backend();
        let input = cuda_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], false);
        let mask = cuda_mask(vec![false, true], &[1, 2]);

        let out = input.masked_select(&mask).unwrap();
        assert!(out.is_cuda(), "forward output must stay CUDA-resident");
        assert_eq!(out.cpu().unwrap().data_vec().unwrap(), vec![2.0, 4.0]);
    }

    #[test]
    fn core130_cuda_masked_select_backward_reduces_broadcasted_input() {
        ensure_cuda_backend();
        let input = cuda_f32(&[10.0, 20.0], &[1, 2], true);
        let mask = cuda_mask(vec![true, false, true, true], &[2, 2]);

        let out = input.masked_select(&mask).unwrap();
        assert!(out.is_cuda(), "forward output must stay CUDA-resident");
        assert_eq!(
            out.cpu().unwrap().data_vec().unwrap(),
            vec![10.0, 10.0, 20.0]
        );

        backward(&out.sum_all().unwrap()).unwrap();
        let grad = input.grad().unwrap().expect("grad must reach input");
        assert!(grad.is_cuda(), "gradient must stay CUDA-resident");
        assert_eq!(grad.cpu().unwrap().data_vec().unwrap(), vec![2.0, 1.0]);
    }

    #[test]
    fn core130_cuda_masked_select_rejects_equal_numel_incompatible_shapes() {
        ensure_cuda_backend();
        let input = cuda_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], false);
        let mask = cuda_mask(vec![true, false, true, false], &[4]);

        assert_shape_err(input.masked_select(&mask), "cuda equal-numel mismatch");
    }

    #[test]
    fn core130_cuda_reduced_precision_broadcasts_without_cpu_fallback() {
        ensure_cuda_backend();
        let mask = cuda_mask(vec![true, false, true, true], &[2, 2]);

        let input_f16 = cuda_f16(&[10.0, 20.0], &[1, 2], true);
        let out_f16 = input_f16.masked_select(&mask).unwrap();
        assert!(out_f16.is_cuda(), "f16 output must stay CUDA-resident");
        let got_f16: Vec<f32> = out_f16
            .cpu()
            .unwrap()
            .data_vec()
            .unwrap()
            .into_iter()
            .map(f32::from)
            .collect();
        assert_eq!(got_f16, vec![10.0, 10.0, 20.0]);
        let grad_seed_f16 = cuda_f16_ones(out_f16.shape());
        backward_with_grad(&out_f16, Some(&grad_seed_f16)).unwrap();
        let grad_f16: Vec<f32> = input_f16
            .grad()
            .unwrap()
            .expect("f16 grad must reach input")
            .cpu()
            .unwrap()
            .data_vec()
            .unwrap()
            .into_iter()
            .map(f32::from)
            .collect();
        assert_eq!(grad_f16, vec![2.0, 1.0]);

        let input_bf16 = cuda_bf16(&[10.0, 20.0], &[1, 2], true);
        let out_bf16 = input_bf16.masked_select(&mask).unwrap();
        assert!(out_bf16.is_cuda(), "bf16 output must stay CUDA-resident");
        let got_bf16: Vec<f32> = out_bf16
            .cpu()
            .unwrap()
            .data_vec()
            .unwrap()
            .into_iter()
            .map(f32::from)
            .collect();
        assert_eq!(got_bf16, vec![10.0, 10.0, 20.0]);
        let grad_seed_bf16 = cuda_bf16_ones(out_bf16.shape());
        backward_with_grad(&out_bf16, Some(&grad_seed_bf16)).unwrap();
        let grad_bf16: Vec<f32> = input_bf16
            .grad()
            .unwrap()
            .expect("bf16 grad must reach input")
            .cpu()
            .unwrap()
            .data_vec()
            .unwrap()
            .into_iter()
            .map(f32::from)
            .collect();
        assert_eq!(grad_bf16, vec![2.0, 1.0]);
    }
}
