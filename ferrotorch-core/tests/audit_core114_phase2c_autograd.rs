//! CORE-114 (#1808): public `Tensor::index_select` / `Tensor::gather` must not
//! return detached tensors when the input requires grad.
//!
//! PyTorch oracle:
//! - `x.reshape(3,2).index_select(0, tensor([2,0,2])).sum().backward()`
//!   accumulates row gradients `[1,1, 0,0, 2,2]`.
//! - `x.reshape(2,3).gather(1, tensor([[0,2],[2,2]])).sum().backward()`
//!   accumulates duplicate column gradients `[1,0,1, 0,0,2]`.

use ferrotorch_core::Tensor;
use ferrotorch_core::autograd::graph::backward;
use ferrotorch_core::creation::from_vec;
use ferrotorch_core::int_tensor::IntTensor;

fn idx(vals: &[i64], shape: &[usize]) -> IntTensor<i64> {
    IntTensor::<i64>::from_vec(vals.to_vec(), shape.to_vec()).unwrap()
}

fn grad_data(x: &Tensor<f32>) -> Vec<f32> {
    x.grad()
        .unwrap()
        .expect("leaf gradient must be populated")
        .cpu()
        .unwrap()
        .data_vec()
        .unwrap()
}

#[test]
fn tensor_index_select_method_keeps_cpu_autograd() {
    let x = from_vec::<f32>((0..6).map(|v| v as f32).collect(), &[3, 2])
        .unwrap()
        .requires_grad_(true);
    let out = x.index_select(0, &idx(&[2, 0, 2], &[3])).unwrap();

    backward(&out.sum_all().unwrap()).unwrap();

    assert_eq!(grad_data(&x), vec![1.0, 1.0, 0.0, 0.0, 2.0, 2.0]);
}

#[test]
fn tensor_gather_method_keeps_cpu_autograd() {
    let x = from_vec::<f32>((0..6).map(|v| v as f32).collect(), &[2, 3])
        .unwrap()
        .requires_grad_(true);
    let out = x.gather(1, &idx(&[0, 2, 2, 2], &[2, 2])).unwrap();

    backward(&out.sum_all().unwrap()).unwrap();

    assert_eq!(grad_data(&x), vec![1.0, 0.0, 1.0, 0.0, 0.0, 2.0]);
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
                .expect("CUDA backend must initialize for CORE-114 probes");
        });
    }

    fn cuda_f32(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
        from_vec::<f32>(data, shape)
            .unwrap()
            .to(Device::Cuda(0))
            .unwrap()
            .requires_grad_(true)
    }

    fn cuda_idx(vals: &[i64], shape: &[usize]) -> IntTensor<i64> {
        idx(vals, shape).to(Device::Cuda(0)).unwrap()
    }

    #[test]
    fn tensor_index_select_method_keeps_cuda_autograd() {
        ensure_cuda_backend();
        let x = cuda_f32((0..6).map(|v| v as f32).collect(), &[3, 2]);
        let out = x.index_select(0, &cuda_idx(&[2, 0, 2], &[3])).unwrap();
        assert!(out.is_cuda(), "forward output must stay CUDA-resident");

        backward(&out.sum_all().unwrap()).unwrap();

        let g = x.grad().unwrap().expect("leaf gradient must be populated");
        assert!(g.is_cuda(), "gradient must stay CUDA-resident");
        assert_eq!(
            g.cpu().unwrap().data_vec().unwrap(),
            vec![1.0, 1.0, 0.0, 0.0, 2.0, 2.0],
        );
    }

    #[test]
    fn tensor_gather_method_keeps_cuda_autograd() {
        ensure_cuda_backend();
        let x = cuda_f32((0..6).map(|v| v as f32).collect(), &[2, 3]);
        let out = x.gather(1, &cuda_idx(&[0, 2, 2, 2], &[2, 2])).unwrap();
        assert!(out.is_cuda(), "forward output must stay CUDA-resident");

        backward(&out.sum_all().unwrap()).unwrap();

        let g = x.grad().unwrap().expect("leaf gradient must be populated");
        assert!(g.is_cuda(), "gradient must stay CUDA-resident");
        assert_eq!(
            g.cpu().unwrap().data_vec().unwrap(),
            vec![1.0, 0.0, 1.0, 0.0, 0.0, 2.0],
        );
    }
}
