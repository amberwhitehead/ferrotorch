//! `scatter_add_segments` autograd parity probes.
//!
//! PyTorch's scatter-add VJP is:
//! `src: grad.gather(dim, index)`
//! (`tools/autograd/derivatives.yaml:1519-1523`). For the segmented row
//! primitive, that becomes `grad_src[e, :] = grad_out[index[e], :]`.
//! These probes pin CPU and CUDA behavior, including duplicate segments,
//! empty output rows, non-contiguous source views, and bf16 CUDA residency.

use ferrotorch_core::autograd::graph::backward;
use ferrotorch_core::creation::from_vec;
use ferrotorch_core::grad_fns::arithmetic::mul;
use ferrotorch_core::scatter_add_segments;

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= 1e-4,
            "mismatch at {i}: actual {a}, expected {e}"
        );
    }
}

#[test]
fn scatter_add_segments_cpu_backward_gathers_weighted_rows_into_strided_source() {
    let base = from_vec::<f32>(
        vec![
            1.0, 99.0, 2.0, 3.0, 99.0, 4.0, 5.0, 99.0, 6.0, 7.0, 99.0, 8.0,
        ],
        &[4, 3],
    )
    .expect("base")
    .requires_grad_(true);
    let src = base
        .as_strided(&[4, 2], &[3, 1], Some(0))
        .expect("2-column strided source view");
    assert!(
        !src.is_contiguous(),
        "probe must route backward through a source view"
    );

    let index = [1_i64, 0, 1, 2];
    let out = scatter_add_segments(&src, &index, 4).expect("scatter forward");
    assert_eq!(out.shape(), &[4, 2]);
    assert_eq!(
        out.data_vec().expect("out data"),
        vec![3.0, 99.0, 6.0, 198.0, 7.0, 99.0, 0.0, 0.0]
    );

    let coeffs = from_vec::<f32>(
        vec![10.0, 11.0, 20.0, 21.0, 30.0, 31.0, 40.0, 41.0],
        &[4, 2],
    )
    .expect("coeffs");
    let loss = mul(&out, &coeffs)
        .expect("weighted output")
        .sum_all()
        .expect("loss");
    backward(&loss).expect("backward");

    let grad = base.grad().expect("base grad handle").expect("base grad");
    assert_eq!(grad.shape(), &[4, 3]);
    assert_close(
        &grad.data_vec().expect("base grad values"),
        &[
            20.0, 21.0, 0.0, // edge 0 -> output row 1
            10.0, 11.0, 0.0, // edge 1 -> output row 0
            20.0, 21.0, 0.0, // edge 2 -> output row 1
            30.0, 31.0, 0.0, // edge 3 -> output row 2
        ],
    );
}

#[test]
fn scatter_add_segments_cpu_backward_handles_empty_source() {
    let src = from_vec::<f32>(vec![], &[0, 2])
        .expect("empty src")
        .requires_grad_(true);
    let out = scatter_add_segments(&src, &[], 3).expect("empty scatter forward");
    assert_eq!(out.shape(), &[3, 2]);
    assert_eq!(out.data_vec().expect("empty-source output"), vec![0.0; 6]);

    let coeffs = from_vec::<f32>(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).expect("coeffs");
    let loss = mul(&out, &coeffs)
        .expect("weighted empty-source output")
        .sum_all()
        .expect("loss");
    backward(&loss).expect("backward");

    let grad = src.grad().expect("src grad handle").expect("src grad");
    assert_eq!(grad.shape(), &[0, 2]);
    assert!(
        grad.data_vec().expect("empty src grad values").is_empty(),
        "empty source gradient must stay empty"
    );
}

#[cfg(feature = "gpu")]
mod cuda {
    use super::*;
    use std::sync::Once;

    use ferrotorch_core::Device;
    use ferrotorch_core::tensor::Tensor;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for scatter-add probes");
        });
    }

    fn cuda_f32(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
        from_vec::<f32>(data.to_vec(), shape)
            .expect("f32 CPU tensor")
            .to(Device::Cuda(0))
            .expect("upload f32")
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

    fn cuda_f16(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<half::f16> {
        let values = data.iter().copied().map(half::f16::from_f32).collect();
        from_vec::<half::f16>(values, shape)
            .expect("f16 CPU tensor")
            .to(Device::Cuda(0))
            .expect("upload f16")
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

    #[test]
    fn scatter_add_segments_cuda_f32_backward_gathers_rows_into_strided_source() {
        ensure_cuda_backend();
        let base = cuda_f32(
            &[
                1.0, 99.0, 2.0, 3.0, 99.0, 4.0, 5.0, 99.0, 6.0, 7.0, 99.0, 8.0,
            ],
            &[4, 3],
            true,
        );
        let src = base
            .as_strided(&[4, 2], &[3, 1], Some(0))
            .expect("CUDA strided source view");
        assert!(
            !src.is_contiguous(),
            "probe must exercise CUDA source materialization and view backward"
        );

        let out =
            scatter_add_segments(&src, &[1_i64, 0, 1, 2], 4).expect("CUDA f32 scatter forward");
        assert!(out.is_cuda(), "forward output must stay CUDA-resident");
        assert_eq!(out.shape(), &[4, 2]);
        assert_close(
            &host_f32(&out),
            &[3.0, 99.0, 6.0, 198.0, 7.0, 99.0, 0.0, 0.0],
        );

        let coeffs = cuda_f32(
            &[10.0, 11.0, 20.0, 21.0, 30.0, 31.0, 40.0, 41.0],
            &[4, 2],
            false,
        );
        let loss = mul(&out, &coeffs)
            .expect("weighted CUDA output")
            .sum_all()
            .expect("CUDA loss");
        backward(&loss).expect("CUDA f32 backward");

        let grad = base.grad().expect("base grad handle").expect("base grad");
        assert!(grad.is_cuda(), "base grad must stay CUDA-resident");
        assert_eq!(grad.shape(), &[4, 3]);
        assert_close(
            &host_f32(&grad),
            &[
                20.0, 21.0, 0.0, 10.0, 11.0, 0.0, 20.0, 21.0, 0.0, 30.0, 31.0, 0.0,
            ],
        );
    }

    #[test]
    fn scatter_add_segments_cuda_bf16_backward_gathers_rows_resident() {
        ensure_cuda_backend();
        let src = cuda_bf16(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], true);
        let out = scatter_add_segments(&src, &[2_i64, 2, 0], 4).expect("CUDA bf16 scatter forward");
        assert!(out.is_cuda(), "bf16 forward output must stay CUDA-resident");
        assert_eq!(out.shape(), &[4, 2]);
        assert_close(&host_bf16(&out), &[5.0, 6.0, 0.0, 0.0, 4.0, 6.0, 0.0, 0.0]);

        let coeffs = cuda_bf16(
            &[10.0, 11.0, 20.0, 21.0, 30.0, 31.0, 40.0, 41.0],
            &[4, 2],
            false,
        );
        let loss = mul(&out, &coeffs)
            .expect("weighted bf16 CUDA output")
            .sum_all()
            .expect("bf16 CUDA loss");
        backward(&loss).expect("CUDA bf16 backward");

        let grad = src.grad().expect("src grad handle").expect("src grad");
        assert!(grad.is_cuda(), "bf16 src grad must stay CUDA-resident");
        assert_eq!(grad.shape(), &[3, 2]);
        assert_close(&host_bf16(&grad), &[30.0, 31.0, 30.0, 31.0, 10.0, 11.0]);
    }

    #[test]
    fn scatter_add_segments_cuda_f16_backward_handles_empty_source_resident() {
        ensure_cuda_backend();
        let src = cuda_f16(&[], &[0, 2], true);
        let out = scatter_add_segments(&src, &[], 3).expect("CUDA f16 empty scatter forward");
        assert!(
            out.is_cuda(),
            "empty-source forward output must stay CUDA-resident"
        );
        assert_eq!(out.shape(), &[3, 2]);
        assert_eq!(host_f16(&out), vec![0.0; 6]);

        let coeffs = cuda_f16(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], false);
        let loss = mul(&out, &coeffs)
            .expect("weighted f16 empty-source output")
            .sum_all()
            .expect("f16 empty-source loss");
        backward(&loss).expect("CUDA f16 empty-source backward");

        let grad = src.grad().expect("src grad handle").expect("src grad");
        assert!(grad.is_cuda(), "empty-source grad must stay CUDA-resident");
        assert_eq!(grad.shape(), &[0, 2]);
        assert!(host_f16(&grad).is_empty());
    }
}
