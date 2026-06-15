//! `torch.outer` parity probes.
//!
//! PyTorch implements outer as:
//! `self.reshape({self.size(0), 1}) * vec2`
//! (`aten/src/ATen/native/LinearAlgebra.cpp:1337-1342`). These probes ensure
//! ferrotorch follows that composite behavior on CPU and CUDA instead of
//! routing through CPU-only data access or a CPU-only custom VJP.

use ferrotorch_core::autograd::graph::backward;
use ferrotorch_core::creation::from_vec;
use ferrotorch_core::linalg::outer;

#[test]
fn outer_cpu_strided_view_forward_backward_matches_composite() {
    let base = from_vec::<f32>(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6])
        .expect("base")
        .requires_grad_(true);
    let a = base
        .as_strided(&[3], &[2], Some(0))
        .expect("strided 1-D view");
    assert!(
        !a.is_contiguous(),
        "probe must exercise reshape materialization"
    );
    let b = from_vec::<f32>(vec![10.0, 20.0], &[2])
        .expect("b")
        .requires_grad_(true);

    let out = outer(&a, &b).expect("outer forward");
    assert_eq!(out.shape(), &[3, 2]);
    assert_eq!(
        out.data_vec().expect("outer values"),
        vec![10.0, 20.0, 30.0, 60.0, 50.0, 100.0]
    );

    backward(&out.sum_all().expect("loss")).expect("outer backward");
    let base_grad = base.grad().expect("base grad handle").expect("base grad");
    assert_eq!(
        base_grad.data_vec().expect("base grad values"),
        vec![30.0, 0.0, 30.0, 0.0, 30.0, 0.0]
    );
    let b_grad = b.grad().expect("b grad handle").expect("b grad");
    assert_eq!(b_grad.data_vec().expect("b grad values"), vec![9.0, 9.0]);
}

#[cfg(feature = "gpu")]
mod cuda {
    use super::*;
    use std::sync::Once;

    use ferrotorch_core::device::Device;
    use ferrotorch_core::tensor::Tensor;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for outer probes");
        });
    }

    fn cuda_f32(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
        from_vec::<f32>(data.to_vec(), shape)
            .expect("f32 CPU tensor")
            .to(Device::Cuda(0))
            .expect("upload f32")
            .requires_grad_(requires_grad)
    }

    fn cuda_f16(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<half::f16> {
        let values = data.iter().copied().map(half::f16::from_f32).collect();
        from_vec::<half::f16>(values, shape)
            .expect("f16 CPU tensor")
            .to(Device::Cuda(0))
            .expect("upload f16")
            .requires_grad_(requires_grad)
    }

    fn cuda_bf16(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<half::bf16> {
        let values = data.iter().copied().map(half::bf16::from_f32).collect();
        from_vec::<half::bf16>(values, shape)
            .expect("bf16 CPU tensor")
            .to(Device::Cuda(0))
            .expect("upload bf16")
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

    fn host_f16(t: &Tensor<half::f16>) -> Vec<f32> {
        assert_eq!(
            t.device(),
            Device::Cuda(0),
            "tensor must stay CUDA-resident until explicit readback"
        );
        t.cpu()
            .expect("D2H f16")
            .data_vec()
            .expect("f16 data")
            .iter()
            .map(|v| v.to_f32())
            .collect()
    }

    fn host_bf16(t: &Tensor<half::bf16>) -> Vec<f32> {
        assert_eq!(
            t.device(),
            Device::Cuda(0),
            "tensor must stay CUDA-resident until explicit readback"
        );
        t.cpu()
            .expect("D2H bf16")
            .data_vec()
            .expect("bf16 data")
            .iter()
            .map(|v| v.to_f32())
            .collect()
    }

    #[test]
    fn outer_cuda_f32_strided_view_forward_backward_resident() {
        ensure_cuda_backend();
        let base = cuda_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6], true);
        let a = base
            .as_strided(&[3], &[2], Some(0))
            .expect("CUDA strided 1-D view");
        assert!(
            !a.is_contiguous(),
            "probe must exercise CUDA reshape materialization"
        );
        let b = cuda_f32(&[10.0, 20.0], &[2], true);

        let out = outer(&a, &b).expect("CUDA f32 outer");
        assert!(out.is_cuda(), "outer output must stay CUDA-resident");
        assert_eq!(out.shape(), &[3, 2]);
        assert_eq!(host_f32(&out), vec![10.0, 20.0, 30.0, 60.0, 50.0, 100.0]);

        backward(&out.sum_all().expect("loss")).expect("CUDA f32 outer backward");
        let base_grad = base.grad().expect("base grad handle").expect("base grad");
        assert!(base_grad.is_cuda(), "base grad must stay CUDA-resident");
        assert_eq!(host_f32(&base_grad), vec![30.0, 0.0, 30.0, 0.0, 30.0, 0.0]);
        let b_grad = b.grad().expect("b grad handle").expect("b grad");
        assert!(b_grad.is_cuda(), "b grad must stay CUDA-resident");
        assert_eq!(host_f32(&b_grad), vec![9.0, 9.0]);
    }

    #[test]
    fn outer_cuda_bf16_forward_backward_resident() {
        ensure_cuda_backend();
        let a = cuda_bf16(&[1.0, 2.0], &[2], true);
        let b = cuda_bf16(&[3.0, 4.0, 5.0], &[3], true);

        let out = outer(&a, &b).expect("CUDA bf16 outer");
        assert!(out.is_cuda(), "outer output must stay CUDA-resident");
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(host_bf16(&out), vec![3.0, 4.0, 5.0, 6.0, 8.0, 10.0]);

        backward(&out.sum_all().expect("loss")).expect("CUDA bf16 outer backward");
        let a_grad = a.grad().expect("a grad handle").expect("a grad");
        assert!(a_grad.is_cuda(), "a grad must stay CUDA-resident");
        assert_eq!(host_bf16(&a_grad), vec![12.0, 12.0]);
        let b_grad = b.grad().expect("b grad handle").expect("b grad");
        assert!(b_grad.is_cuda(), "b grad must stay CUDA-resident");
        assert_eq!(host_bf16(&b_grad), vec![3.0, 3.0, 3.0]);
    }

    #[test]
    fn outer_cuda_f16_empty_forward_backward_resident() {
        ensure_cuda_backend();
        let a = cuda_f16(&[], &[0], true);
        let b = cuda_f16(&[1.0, 2.0], &[2], true);

        let out = outer(&a, &b).expect("CUDA f16 empty outer");
        assert!(out.is_cuda(), "empty outer output must stay CUDA-resident");
        assert_eq!(out.shape(), &[0, 2]);
        assert!(host_f16(&out).is_empty());

        backward(&out.sum_all().expect("loss")).expect("CUDA f16 empty outer backward");
        let a_grad = a.grad().expect("a grad handle").expect("a grad");
        assert!(a_grad.is_cuda(), "empty a grad must stay CUDA-resident");
        assert_eq!(a_grad.shape(), &[0]);
        assert!(host_f16(&a_grad).is_empty());
        let b_grad = b.grad().expect("b grad handle").expect("b grad");
        assert!(b_grad.is_cuda(), "b grad must stay CUDA-resident");
        assert_eq!(host_f16(&b_grad), vec![0.0, 0.0]);
    }
}
