//! Trace parity probes.
//!
//! PyTorch oracle for this segment:
//! - CPU `torch.trace` works on non-contiguous 2-D views.
//! - CUDA `torch.trace` is implemented as `self.diagonal().sum()`.
//! - Backward scatters the scalar cotangent onto the input's main diagonal.

use ferrotorch_core::autograd::graph::backward;
use ferrotorch_core::creation::from_vec;
use ferrotorch_core::linalg::trace;

#[test]
fn trace_cpu_noncontiguous_transpose_forward_backward() {
    let base = from_vec::<f32>(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .expect("base")
        .requires_grad_(true);
    let view = base.transpose(0, 1).expect("transpose view");
    assert!(
        !view.is_contiguous(),
        "probe must exercise non-contiguous trace"
    );

    let out = trace(&view).expect("trace of transposed view");
    assert_eq!(out.shape(), &[] as &[usize]);
    assert_eq!(out.data_vec().expect("trace value"), vec![6.0]);

    backward(&out).expect("trace backward");
    let grad = base.grad().expect("grad handle").expect("base grad");
    assert_eq!(grad.shape(), &[2, 3]);
    assert_eq!(
        grad.data_vec().expect("base grad values"),
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    );
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
                .expect("CUDA backend must initialize for trace probes");
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
    fn trace_cuda_f32_forward_backward_resident() {
        ensure_cuda_backend();
        let x = cuda_f32(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            &[3, 4],
            true,
        );

        let out = trace(&x).expect("CUDA f32 trace");
        assert!(out.is_cuda(), "trace output must stay CUDA-resident");
        assert_eq!(out.shape(), &[] as &[usize]);
        assert_eq!(host_f32(&out), vec![18.0]);

        backward(&out).expect("CUDA f32 trace backward");
        let grad = x.grad().expect("grad handle").expect("x grad");
        assert!(grad.is_cuda(), "trace grad must stay CUDA-resident");
        assert_eq!(
            host_f32(&grad),
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        );
    }

    #[test]
    fn trace_cuda_bf16_noncontiguous_transpose_forward_backward_resident() {
        ensure_cuda_backend();
        let base = cuda_bf16(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
        let view = base.transpose(0, 1).expect("CUDA transpose view");
        assert!(
            !view.is_contiguous(),
            "probe must exercise non-contiguous CUDA trace"
        );

        let out = trace(&view).expect("CUDA bf16 trace of transposed view");
        assert!(out.is_cuda(), "trace output must stay CUDA-resident");
        assert_eq!(out.shape(), &[] as &[usize]);
        assert_eq!(host_bf16(&out), vec![6.0]);

        backward(&out).expect("CUDA bf16 trace backward");
        let grad = base.grad().expect("grad handle").expect("base grad");
        assert!(grad.is_cuda(), "trace grad must stay CUDA-resident");
        assert_eq!(host_bf16(&grad), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn trace_cuda_f16_empty_matrix_backward_zero_resident() {
        ensure_cuda_backend();
        let x = cuda_f16(&[], &[0, 4], true);

        let out = trace(&x).expect("CUDA f16 empty trace");
        assert!(out.is_cuda(), "empty trace output must stay CUDA-resident");
        assert_eq!(out.shape(), &[] as &[usize]);
        assert_eq!(host_f16(&out), vec![0.0]);

        backward(&out).expect("CUDA f16 empty trace backward");
        let grad = x.grad().expect("grad handle").expect("x grad");
        assert!(grad.is_cuda(), "empty trace grad must stay CUDA-resident");
        assert_eq!(grad.shape(), &[0, 4]);
        assert!(host_f16(&grad).is_empty());
    }
}
