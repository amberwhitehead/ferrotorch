use ferrotorch_core::creation::from_vec;
use ferrotorch_core::tensor::Tensor;

fn tensor<T: ferrotorch_core::Float>(values: Vec<T>, shape: &[usize]) -> Tensor<T> {
    from_vec(values, shape).expect("tensor")
}

#[test]
fn scalar_flip_accepts_zero_and_negative_one_like_pytorch() {
    let x = tensor(vec![7.0_f32], &[]);

    assert_eq!(x.flip_t(&[]).expect("flip []").data_vec().unwrap(), [7.0]);
    assert_eq!(x.flip_t(&[0]).expect("flip [0]").data_vec().unwrap(), [7.0]);
    assert_eq!(
        x.flip_t(&[-1]).expect("flip [-1]").data_vec().unwrap(),
        [7.0]
    );

    let duplicate = x.flip_t(&[0, -1]).expect_err("duplicate scalar dim");
    assert!(
        duplicate.to_string().contains("appears multiple times"),
        "{duplicate}"
    );

    let out_of_range = x.flip_t(&[1]).expect_err("scalar dim out of range");
    assert!(
        out_of_range.to_string().contains("out of range"),
        "{out_of_range}"
    );
}

#[cfg(feature = "gpu")]
mod cuda {
    use super::*;
    use ferrotorch_core::device::Device;
    use half::{bf16, f16};
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend().expect("CUDA backend must initialize");
        });
    }

    fn cuda_tensor<T: ferrotorch_core::Float>(values: Vec<T>, shape: &[usize]) -> Tensor<T> {
        tensor(values, shape).to(Device::Cuda(0)).expect("upload")
    }

    fn cuda_data<T: ferrotorch_core::Float>(t: &Tensor<T>) -> Vec<T> {
        assert_eq!(t.device(), Device::Cuda(0), "tensor must stay CUDA");
        t.to(Device::Cpu)
            .expect("download")
            .data_vec()
            .expect("data")
    }

    fn assert_close_f32(got: &[f32], expected: &[f32], tol: f32) {
        assert_eq!(got.len(), expected.len());
        for (idx, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() <= tol,
                "idx {idx}: expected {e:?}, got {g:?}, tol {tol}"
            );
        }
    }

    #[test]
    fn cuda_flip_f32_f64_dims_and_scalar_stay_resident() {
        ensure_cuda_backend();

        let x = cuda_tensor(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let y = x.flip_t(&[0, 1]).expect("flip f32");
        assert_eq!(y.device(), Device::Cuda(0));
        assert_eq!(cuda_data(&y), vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);

        let xd = cuda_tensor(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let yd = xd.flip_t(&[-1]).expect("flip f64");
        assert_eq!(yd.device(), Device::Cuda(0));
        assert_eq!(cuda_data(&yd), vec![3.0, 2.0, 1.0, 6.0, 5.0, 4.0]);

        let s = cuda_tensor(vec![7.0_f32], &[]);
        let flipped = s.flip_t(&[0]).expect("scalar flip [0]");
        assert_eq!(flipped.device(), Device::Cuda(0));
        assert_eq!(cuda_data(&flipped), vec![7.0]);
    }

    #[test]
    fn cuda_flip_materializes_noncontiguous_input_on_device() {
        ensure_cuda_backend();

        let x = cuda_tensor(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let xt = x.transpose(0, 1).expect("transpose view");
        assert_eq!(xt.shape(), &[3, 2]);
        assert_eq!(xt.device(), Device::Cuda(0));

        let y = xt.flip_t(&[0]).expect("flip non-contiguous CUDA view");
        assert_eq!(y.device(), Device::Cuda(0));
        assert_eq!(y.shape(), &[3, 2]);
        assert_eq!(cuda_data(&y), vec![3.0, 6.0, 2.0, 5.0, 1.0, 4.0]);
    }

    #[test]
    fn cuda_flip_zero_size_and_half_family() {
        ensure_cuda_backend();

        let empty = cuda_tensor(Vec::<f32>::new(), &[0, 3]);
        let empty_flipped = empty.flip_t(&[0, 1]).expect("flip empty");
        assert_eq!(empty_flipped.device(), Device::Cuda(0));
        assert_eq!(empty_flipped.shape(), &[0, 3]);
        assert!(cuda_data(&empty_flipped).is_empty());

        let xh = cuda_tensor(
            vec![
                f16::from_f32(1.0),
                f16::from_f32(2.0),
                f16::from_f32(3.0),
                f16::from_f32(4.0),
            ],
            &[2, 2],
        );
        let yh = xh.flip_t(&[1]).expect("flip f16");
        let got_h: Vec<f32> = cuda_data(&yh).iter().map(|v| v.to_f32()).collect();
        assert_close_f32(&got_h, &[2.0, 1.0, 4.0, 3.0], 0.0);

        let xb = cuda_tensor(
            vec![
                bf16::from_f32(1.0),
                bf16::from_f32(2.0),
                bf16::from_f32(3.0),
                bf16::from_f32(4.0),
            ],
            &[2, 2],
        );
        let yb = xb.flip_t(&[0]).expect("flip bf16");
        let got_b: Vec<f32> = cuda_data(&yb).iter().map(|v| v.to_f32()).collect();
        assert_close_f32(&got_b, &[3.0, 4.0, 1.0, 2.0], 0.0);
    }

    #[test]
    fn cuda_flip_backward_is_resident_self_inverse() {
        ensure_cuda_backend();

        let x = cuda_tensor(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).requires_grad_(true);
        let y = x.flip_t(&[0, 1]).expect("flip");
        let grad = cuda_tensor(vec![10.0_f32, 20.0, 30.0, 40.0, 50.0, 60.0], &[2, 3]);
        let grad_input = y
            .grad_fn()
            .expect("flip grad fn")
            .backward(&grad)
            .expect("backward")[0]
            .clone()
            .expect("grad input");
        assert_eq!(grad_input.device(), Device::Cuda(0));
        assert_eq!(
            cuda_data(&grad_input),
            vec![60.0, 50.0, 40.0, 30.0, 20.0, 10.0]
        );
    }
}
