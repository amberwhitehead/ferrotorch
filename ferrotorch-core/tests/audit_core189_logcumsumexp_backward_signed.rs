use ferrotorch_core::grad_fns::cumulative::logcumsumexp;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

fn cpu_tensor<T: ferrotorch_core::Float>(
    values: Vec<T>,
    shape: &[usize],
    requires_grad: bool,
) -> Tensor<T> {
    Tensor::from_storage(TensorStorage::cpu(values), shape.to_vec(), requires_grad)
        .expect("cpu tensor")
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

fn assert_close_f64(got: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(got.len(), expected.len());
    for (idx, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() <= tol,
            "idx {idx}: expected {e:?}, got {g:?}, tol {tol}"
        );
    }
}

#[test]
fn cpu_logcumsumexp_backward_signed_and_extreme_matches_pytorch() {
    let x = cpu_tensor(vec![1000.0_f64, 999.0, -1000.0, 1001.0], &[4], true);
    let y = logcumsumexp(&x, 0).expect("logcumsumexp");
    let grad = cpu_tensor(vec![1.0_f64, -2.0, 0.0, 3.0], &[4], false);
    let grads = y
        .grad_fn()
        .expect("logcumsumexp grad fn")
        .backward(&grad)
        .expect("backward");
    let got = grads[0].as_ref().expect("grad input").data_vec().unwrap();
    assert_close_f64(
        &got,
        &[
            0.2720682559043739,
            -0.26779112322882637,
            0.0,
            1.9957228673244782,
        ],
        1e-12,
    );
}

#[test]
fn cpu_logcumsumexp_backward_ignores_nan_upstream_like_pytorch_where_masks() {
    let x = cpu_tensor(vec![0.0_f32, 1.0, -1.0, 2.0], &[4], true);
    let y = logcumsumexp(&x, 0).expect("logcumsumexp");
    let grad = cpu_tensor(vec![f32::NAN, 1.0, -1.0, 0.0], &[4], false);
    let grads = y
        .grad_fn()
        .expect("logcumsumexp grad fn")
        .backward(&grad)
        .expect("backward");
    let got = grads[0].as_ref().expect("grad input").data_vec().unwrap();
    assert_close_f32(&got, &[0.024212942, 0.065817535, -0.09003056, 0.0], 1e-6);
}

#[cfg(feature = "gpu")]
mod cuda {
    use super::*;
    use ferrotorch_core::creation::from_vec;
    use ferrotorch_core::device::Device;
    use half::{bf16, f16};
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend().expect("CUDA backend must initialize");
        });
    }

    fn cuda_tensor<T: ferrotorch_core::Float>(
        values: Vec<T>,
        shape: &[usize],
        requires_grad: bool,
    ) -> Tensor<T> {
        from_vec(values, shape)
            .expect("cpu tensor")
            .to(Device::Cuda(0))
            .expect("upload tensor")
            .requires_grad_(requires_grad)
    }

    fn cpu_data<T: ferrotorch_core::Float>(tensor: &Tensor<T>) -> Vec<T> {
        assert_eq!(tensor.device(), Device::Cuda(0), "tensor must stay CUDA");
        tensor
            .to(Device::Cpu)
            .expect("download tensor")
            .data_vec()
            .expect("cpu data")
    }

    #[test]
    fn cuda_logcumsumexp_backward_signed_extreme_f32_f64_stays_resident() {
        ensure_cuda_backend();

        let xf = cuda_tensor(vec![1000.0_f32, 999.0, -1000.0, 1001.0], &[4], true);
        let yf = logcumsumexp(&xf, 0).expect("f32 logcumsumexp");
        let gf = cuda_tensor(vec![1.0_f32, -2.0, 0.0, 3.0], &[4], false);
        let gradf = yf
            .grad_fn()
            .expect("f32 grad fn")
            .backward(&gf)
            .expect("f32 backward")[0]
            .clone()
            .expect("f32 grad input");
        assert_eq!(gradf.device(), Device::Cuda(0));
        assert_close_f32(
            &cpu_data(&gradf),
            &[0.27198184, -0.26781338, 0.0, 1.9957902],
            5e-3,
        );

        let xd = cuda_tensor(vec![1000.0_f64, 999.0, -1000.0, 1001.0], &[4], true);
        let yd = logcumsumexp(&xd, 0).expect("f64 logcumsumexp");
        let gd = cuda_tensor(vec![1.0_f64, -2.0, 0.0, 3.0], &[4], false);
        let gradd = yd
            .grad_fn()
            .expect("f64 grad fn")
            .backward(&gd)
            .expect("f64 backward")[0]
            .clone()
            .expect("f64 grad input");
        assert_eq!(gradd.device(), Device::Cuda(0));
        assert_close_f64(
            &cpu_data(&gradd),
            &[
                0.2720682559043739,
                -0.26779112322882637,
                0.0,
                1.9957228673244782,
            ],
            5e-3,
        );
    }

    #[test]
    fn cuda_logcumsumexp_backward_half_family_signed_matches_pytorch() {
        ensure_cuda_backend();

        let xh = cuda_tensor(
            vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)],
            &[3],
            true,
        );
        let yh = logcumsumexp(&xh, 0).expect("f16 logcumsumexp");
        let gh = cuda_tensor(
            vec![f16::from_f32(1.0), f16::from_f32(-2.0), f16::from_f32(3.0)],
            &[3],
            false,
        );
        let gradh = yh
            .grad_fn()
            .expect("f16 grad fn")
            .backward(&gh)
            .expect("f16 backward")[0]
            .clone()
            .expect("f16 grad input");
        assert_eq!(gradh.device(), Device::Cuda(0));
        let got_h: Vec<f32> = cpu_data(&gradh).iter().map(|v| v.to_f32()).collect();
        assert_close_f32(&got_h, &[0.7319336, -0.7294922, 1.9960938], 8e-3);

        let xb = cuda_tensor(
            vec![
                bf16::from_f32(1.0),
                bf16::from_f32(2.0),
                bf16::from_f32(3.0),
            ],
            &[3],
            true,
        );
        let yb = logcumsumexp(&xb, 0).expect("bf16 logcumsumexp");
        let gb = cuda_tensor(
            vec![
                bf16::from_f32(1.0),
                bf16::from_f32(-2.0),
                bf16::from_f32(3.0),
            ],
            &[3],
            false,
        );
        let gradb = yb
            .grad_fn()
            .expect("bf16 grad fn")
            .backward(&gb)
            .expect("bf16 backward")[0]
            .clone()
            .expect("bf16 grad input");
        assert_eq!(gradb.device(), Device::Cuda(0));
        let got_b: Vec<f32> = cpu_data(&gradb).iter().map(|v| v.to_f32()).collect();
        assert_close_f32(&got_b, &[0.73046875, -0.72265625, 1.9921875], 2.5e-2);
    }
}
