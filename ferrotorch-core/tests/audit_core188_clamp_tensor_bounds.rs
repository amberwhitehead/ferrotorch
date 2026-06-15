use ferrotorch_core::grad_fns::reduction::sum;
use ferrotorch_core::grad_fns::transcendental::{clamp_min_tensor, clamp_tensor, clip_tensor};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

fn cpu_f32(values: Vec<f32>, shape: &[usize], requires_grad: bool) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(values), shape.to_vec(), requires_grad)
        .expect("cpu tensor")
}

fn assert_f32_close_nan(got: &[f32], expected: &[f32]) {
    assert_eq!(got.len(), expected.len());
    for (idx, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        if e.is_nan() {
            assert!(g.is_nan(), "idx {idx}: expected NaN, got {g:?}");
        } else {
            assert!(
                (g - e).abs() <= 1e-6,
                "idx {idx}: expected {e:?}, got {g:?}"
            );
        }
    }
}

#[test]
fn cpu_clamp_tensor_broadcast_backward_matches_pytorch_masks() {
    let x = cpu_f32(vec![-1.0, 0.0, 2.0, 4.0, 1.0, f32::NAN], &[2, 3], true);
    let min = cpu_f32(vec![0.0, 0.0, 3.0], &[3], true);
    let max = cpu_f32(vec![1.0, 2.0], &[2, 1], true);

    let y = clamp_tensor(&x, Some(&min), Some(&max)).expect("tensor clamp");
    assert_eq!(y.shape(), &[2, 3]);
    assert_f32_close_nan(&y.data_vec().unwrap(), &[0.0, 0.0, 1.0, 2.0, 1.0, f32::NAN]);

    sum(&y).expect("sum").backward().expect("backward");
    assert_f32_close_nan(
        &x.grad().unwrap().unwrap().data_vec().unwrap(),
        &[0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
    );
    assert_f32_close_nan(
        &min.grad().unwrap().unwrap().data_vec().unwrap(),
        &[1.0, 0.0, 0.0],
    );
    assert_f32_close_nan(
        &max.grad().unwrap().unwrap().data_vec().unwrap(),
        &[1.0, 2.0],
    );
}

#[test]
fn cpu_clamp_tensor_one_sided_nan_and_methods_match_pytorch() {
    let x = cpu_f32(vec![f32::NAN, -1.0, 0.0, 2.0], &[4], true);
    let min = cpu_f32(vec![0.0, 0.0, 0.0, f32::NAN], &[4], true);
    let y = clamp_min_tensor(&x, &min).expect("clamp_min tensor");
    assert_f32_close_nan(&y.data_vec().unwrap(), &[f32::NAN, 0.0, 0.0, f32::NAN]);
    sum(&y).expect("sum").backward().expect("backward");
    assert_f32_close_nan(
        &x.grad().unwrap().unwrap().data_vec().unwrap(),
        &[0.0, 0.0, 1.0, 0.0],
    );
    assert_f32_close_nan(
        &min.grad().unwrap().unwrap().data_vec().unwrap(),
        &[0.0, 1.0, 0.0, 0.0],
    );

    let z = cpu_f32(vec![-1.0, 0.0, 2.0], &[3], false);
    let max = cpu_f32(vec![0.0], &[1], false);
    assert_eq!(
        z.clamp_max_tensor_t(&max)
            .expect("method clamp_max tensor")
            .data_vec()
            .unwrap(),
        vec![-1.0, 0.0, 0.0]
    );
    assert_eq!(
        clip_tensor(&z, Some(&max), None)
            .expect("clip tensor alias")
            .data_vec()
            .unwrap(),
        vec![0.0, 0.0, 2.0]
    );
    assert!(z.clamp_tensor_t(None, None).is_err());
}

#[cfg(feature = "gpu")]
mod cuda {
    use super::*;
    use ferrotorch_core::creation::from_vec;
    use ferrotorch_core::device::Device;
    use ferrotorch_core::grad_fns::transcendental::clamp_max_tensor;
    use half::{bf16, f16};
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend().expect("CUDA backend must initialize");
        });
    }

    fn cuda_f32(values: Vec<f32>, shape: &[usize], requires_grad: bool) -> Tensor<f32> {
        from_vec(values, shape)
            .expect("cpu tensor")
            .to(Device::Cuda(0))
            .expect("upload tensor")
            .requires_grad_(requires_grad)
    }

    fn cuda_f64(values: Vec<f64>, shape: &[usize], requires_grad: bool) -> Tensor<f64> {
        from_vec(values, shape)
            .expect("cpu tensor")
            .to(Device::Cuda(0))
            .expect("upload tensor")
            .requires_grad_(requires_grad)
    }

    fn cuda_half(values: Vec<f16>, shape: &[usize], requires_grad: bool) -> Tensor<f16> {
        from_vec(values, shape)
            .expect("cpu tensor")
            .to(Device::Cuda(0))
            .expect("upload tensor")
            .requires_grad_(requires_grad)
    }

    fn cuda_bf16(values: Vec<bf16>, shape: &[usize], requires_grad: bool) -> Tensor<bf16> {
        from_vec(values, shape)
            .expect("cpu tensor")
            .to(Device::Cuda(0))
            .expect("upload tensor")
            .requires_grad_(requires_grad)
    }

    fn cpu_data<T: ferrotorch_core::dtype::Float>(tensor: &Tensor<T>) -> Vec<T> {
        assert_eq!(tensor.device(), Device::Cuda(0), "tensor must stay CUDA");
        tensor
            .to(Device::Cpu)
            .expect("download tensor")
            .data_vec()
            .expect("cpu data")
    }

    #[test]
    fn cuda_clamp_tensor_broadcast_backward_stays_resident() {
        ensure_cuda_backend();
        let x = cuda_f32(vec![-1.0, 0.0, 2.0, 4.0, 1.0, f32::NAN], &[2, 3], true);
        let min = cuda_f32(vec![0.0, 0.0, 3.0], &[3], true);
        let max = cuda_f32(vec![1.0, 2.0], &[2, 1], true);

        let y = clamp_tensor(&x, Some(&min), Some(&max)).expect("tensor clamp cuda");
        assert_eq!(y.device(), Device::Cuda(0));
        assert_f32_close_nan(&cpu_data(&y), &[0.0, 0.0, 1.0, 2.0, 1.0, f32::NAN]);

        sum(&y).expect("sum").backward().expect("backward");
        assert_f32_close_nan(
            &cpu_data(&x.grad().unwrap().unwrap()),
            &[0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        );
        assert_f32_close_nan(&cpu_data(&min.grad().unwrap().unwrap()), &[1.0, 0.0, 0.0]);
        assert_f32_close_nan(&cpu_data(&max.grad().unwrap().unwrap()), &[1.0, 2.0]);

        let xd = cuda_f64(vec![-1.0, 0.0, 2.0], &[3], true);
        let mind = cuda_f64(vec![0.0], &[1], true);
        let maxd = cuda_f64(vec![1.0], &[1], true);
        let yd = clamp_tensor(&xd, Some(&mind), Some(&maxd)).expect("f64 tensor clamp cuda");
        assert_eq!(yd.device(), Device::Cuda(0));
        assert_eq!(cpu_data(&yd), vec![0.0, 0.0, 1.0]);
        sum(&yd).expect("sum f64").backward().expect("backward f64");
        assert_eq!(cpu_data(&xd.grad().unwrap().unwrap()), vec![0.0, 1.0, 0.0]);
        assert_eq!(cpu_data(&mind.grad().unwrap().unwrap()), vec![1.0]);
        assert_eq!(cpu_data(&maxd.grad().unwrap().unwrap()), vec![1.0]);
    }

    #[test]
    fn cuda_clamp_tensor_signed_zero_matches_torch_cuda() {
        ensure_cuda_backend();
        let neg_zero = -0.0_f32;
        let pos_zero = 0.0_f32;
        let x = cuda_f32(vec![neg_zero, pos_zero], &[2], false);
        let bound = cuda_f32(vec![pos_zero, neg_zero], &[2], false);

        let maxed = clamp_min_tensor(&x, &bound).expect("cuda clamp_min tensor zeros");
        assert_eq!(
            cpu_data(&maxed)
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            vec![0, 0]
        );

        let mined = clamp_max_tensor(&x, &bound).expect("cuda clamp_max tensor zeros");
        assert_eq!(
            cpu_data(&mined)
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            vec![0x8000_0000, 0x8000_0000]
        );
    }

    #[test]
    fn cuda_clamp_tensor_half_and_bfloat_backward_stays_resident() {
        ensure_cuda_backend();
        let h = cuda_half(
            vec![f16::from_f32(-1.0), f16::from_f32(0.0), f16::from_f32(2.0)],
            &[3],
            true,
        );
        let h_min = cuda_half(vec![f16::from_f32(0.0)], &[1], true);
        let h_out = clamp_min_tensor(&h, &h_min).expect("f16 tensor clamp_min");
        assert_eq!(h_out.device(), Device::Cuda(0));
        assert_eq!(
            cpu_data(&h_out)
                .iter()
                .map(|v| v.to_f32())
                .collect::<Vec<_>>(),
            vec![0.0, 0.0, 2.0]
        );
        sum(&h_out)
            .expect("sum f16")
            .backward()
            .expect("backward f16");
        assert_eq!(
            cpu_data(&h.grad().unwrap().unwrap())
                .iter()
                .map(|v| v.to_f32())
                .collect::<Vec<_>>(),
            vec![0.0, 1.0, 1.0]
        );
        assert_eq!(
            cpu_data(&h_min.grad().unwrap().unwrap())
                .iter()
                .map(|v| v.to_f32())
                .collect::<Vec<_>>(),
            vec![1.0]
        );

        let b = cuda_bf16(
            vec![
                bf16::from_f32(-1.0),
                bf16::from_f32(0.0),
                bf16::from_f32(2.0),
            ],
            &[3],
            true,
        );
        let b_max = cuda_bf16(vec![bf16::from_f32(0.0)], &[1], true);
        let b_out = clamp_max_tensor(&b, &b_max).expect("bf16 tensor clamp_max");
        assert_eq!(b_out.device(), Device::Cuda(0));
        assert_eq!(
            cpu_data(&b_out)
                .iter()
                .map(|v| v.to_f32())
                .collect::<Vec<_>>(),
            vec![-1.0, 0.0, 0.0]
        );
        sum(&b_out)
            .expect("sum bf16")
            .backward()
            .expect("backward bf16");
        assert_eq!(
            cpu_data(&b.grad().unwrap().unwrap())
                .iter()
                .map(|v| v.to_f32())
                .collect::<Vec<_>>(),
            vec![1.0, 1.0, 0.0]
        );
        assert_eq!(
            cpu_data(&b_max.grad().unwrap().unwrap())
                .iter()
                .map(|v| v.to_f32())
                .collect::<Vec<_>>(),
            vec![1.0]
        );
    }
}
