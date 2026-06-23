//! Direct public-surface probes for `OuterBackward` and `outer_differentiable`.
//!
//! PyTorch implements `torch.outer` as
//! `self.reshape({self.size(0), 1}) * vec2`
//! (`aten/src/ATen/native/LinearAlgebra.cpp:1337-1342`). The exported
//! `OuterBackward` node must therefore compute its VJP through resident
//! broadcast/reduction primitives, not through CPU-only matrix-vector helpers.

use ferrotorch_core::autograd::graph::backward;
use ferrotorch_core::creation::from_vec;
use ferrotorch_core::grad_fns::linalg::{OuterBackward, outer_differentiable};
use ferrotorch_core::tensor::GradFn;

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (idx, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let diff = (a - e).abs();
        assert!(
            diff <= 1.0e-6,
            "value {idx}: actual {a}, expected {e}, diff {diff}"
        );
    }
}

#[test]
fn outer_backward_public_node_matches_pytorch_vjp_cpu_strided_grad() {
    let a = from_vec::<f32>(vec![2.0, -1.0], &[2])
        .expect("a")
        .requires_grad_(true);
    let b = from_vec::<f32>(vec![3.0, 5.0, 7.0], &[3])
        .expect("b")
        .requires_grad_(true);
    let grad_base =
        from_vec::<f32>(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).expect("grad base");
    let grad_output = grad_base.transpose(0, 1).expect("strided grad_output");
    assert!(
        !grad_output.is_contiguous(),
        "probe must exercise a non-contiguous upstream gradient"
    );

    let grads = OuterBackward::new(a, b)
        .backward(&grad_output)
        .expect("outer backward");
    let grad_a = grads[0].as_ref().expect("grad a");
    let grad_b = grads[1].as_ref().expect("grad b");

    assert_eq!(grad_a.shape(), &[2]);
    assert_eq!(grad_b.shape(), &[3]);
    assert_close(&grad_a.data_vec().expect("grad a data"), &[53.0, 68.0]);
    assert_close(&grad_b.data_vec().expect("grad b data"), &[0.0, 2.0, 4.0]);
}

#[test]
fn outer_differentiable_weighted_backward_matches_composite_vjp_cpu() {
    let a = from_vec::<f32>(vec![2.0, -1.0], &[2])
        .expect("a")
        .requires_grad_(true);
    let b = from_vec::<f32>(vec![3.0, 5.0, 7.0], &[3])
        .expect("b")
        .requires_grad_(true);
    let weights = from_vec::<f32>(vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0], &[2, 3]).expect("weights");

    let out = outer_differentiable(&a, &b).expect("outer differentiable");
    let loss = out
        .mul_t(&weights)
        .expect("weighted output")
        .sum_all()
        .expect("loss");
    backward(&loss).expect("outer differentiable backward");

    let grad_a = a.grad().expect("a grad handle").expect("a grad");
    let grad_b = b.grad().expect("b grad handle").expect("b grad");
    assert_close(&grad_a.data_vec().expect("grad a data"), &[53.0, 68.0]);
    assert_close(&grad_b.data_vec().expect("grad b data"), &[0.0, 2.0, 4.0]);
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
                .expect("CUDA backend must initialize for outer backward probes");
        });
    }

    fn cuda_f32(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
        from_vec::<f32>(data.to_vec(), shape)
            .expect("CPU tensor")
            .to(Device::Cuda(0))
            .expect("upload")
            .requires_grad_(requires_grad)
    }

    fn host_f32(t: &Tensor<f32>) -> Vec<f32> {
        assert_eq!(
            t.device(),
            Device::Cuda(0),
            "tensor must stay CUDA-resident until explicit readback"
        );
        t.cpu().expect("D2H").data_vec().expect("data")
    }

    #[test]
    fn outer_backward_public_node_matches_pytorch_vjp_cuda_strided_grad() {
        ensure_cuda_backend();
        let a = cuda_f32(&[2.0, -1.0], &[2], true);
        let b = cuda_f32(&[3.0, 5.0, 7.0], &[3], true);
        let grad_base = cuda_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], false);
        let grad_output = grad_base.transpose(0, 1).expect("strided CUDA grad_output");
        assert!(
            !grad_output.is_contiguous(),
            "probe must exercise a non-contiguous CUDA upstream gradient"
        );

        let grads = OuterBackward::new(a, b)
            .backward(&grad_output)
            .expect("CUDA outer backward");
        let grad_a = grads[0].as_ref().expect("grad a");
        let grad_b = grads[1].as_ref().expect("grad b");

        assert!(grad_a.is_cuda(), "grad a must remain CUDA-resident");
        assert!(grad_b.is_cuda(), "grad b must remain CUDA-resident");
        assert_eq!(grad_a.shape(), &[2]);
        assert_eq!(grad_b.shape(), &[3]);
        assert_close(&host_f32(grad_a), &[53.0, 68.0]);
        assert_close(&host_f32(grad_b), &[0.0, 2.0, 4.0]);
    }

    #[test]
    fn outer_backward_public_node_handles_empty_cuda_without_host_roundtrip() {
        ensure_cuda_backend();
        let a = cuda_f32(&[], &[0], true);
        let b = cuda_f32(&[3.0, 5.0, 7.0], &[3], true);
        let grad_output = cuda_f32(&[], &[0, 3], false);

        let grads = OuterBackward::new(a, b)
            .backward(&grad_output)
            .expect("empty CUDA outer backward");
        let grad_a = grads[0].as_ref().expect("grad a");
        let grad_b = grads[1].as_ref().expect("grad b");

        assert!(grad_a.is_cuda(), "empty grad a must remain CUDA-resident");
        assert!(grad_b.is_cuda(), "grad b must remain CUDA-resident");
        assert_eq!(grad_a.shape(), &[0]);
        assert_eq!(grad_b.shape(), &[3]);
        assert!(host_f32(grad_a).is_empty());
        assert_close(&host_f32(grad_b), &[0.0, 0.0, 0.0]);
    }
}
