use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::{Device, DualTensor, FerrotorchError, Tensor, dual_mul, jacfwd};

fn cpu_tensor(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .expect("construct cpu tensor")
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch, actual={actual:?}, expected={expected:?}"
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "{label}[{i}]: expected {e}, got {a}; actual={actual:?}"
        );
    }
}

#[test]
fn dual_tensor_rejects_mismatched_primal_tangent_devices() {
    let primal = cpu_tensor(&[1.0, 2.0], &[2]);
    let tangent = cpu_tensor(&[1.0, 1.0], &[2])
        .to(Device::Meta)
        .expect("metadata-only transfer to meta");

    let err = DualTensor::new(primal, tangent).expect_err("device mismatch must be rejected");
    assert!(
        matches!(err, FerrotorchError::DeviceMismatch { .. }),
        "expected DeviceMismatch, got {err:?}"
    );
}

#[test]
fn jacfwd_empty_input_returns_output_shape_plus_empty_input_axis() {
    let input = cpu_tensor(&[], &[0]);
    let jac = jacfwd(
        |x| {
            let out = cpu_tensor(&[2.0, 3.0, 5.0], &[3]).to(x.primal.device())?;
            DualTensor::constant(out)
        },
        &input,
    )
    .expect("jacfwd empty input");

    assert_eq!(jac.shape(), &[3, 0]);
    assert_eq!(jac.device(), Device::Cpu);
    assert_eq!(jac.numel(), 0);
}

#[test]
fn jacfwd_scalar_output_matches_torch_shape_convention() {
    let input = cpu_tensor(&[2.0, 3.0], &[2]);
    let jac = jacfwd(
        |x| {
            let y = dual_mul(&x, &x)?;
            let primal = y.primal.sum_all()?;
            let tangent = y.tangent.sum_all()?;
            DualTensor::new(primal, tangent)
        },
        &input,
    )
    .expect("jacfwd scalar output");

    assert_eq!(jac.shape(), &[2]);
    assert_close(&jac.data_vec().expect("jac data"), &[4.0, 6.0], 1e-5, "jac");
}

#[cfg(feature = "gpu")]
mod cuda {
    use super::{assert_close, cpu_tensor};
    use ferrotorch_core::creation::{full_like, ones_like, rand_like, randn_like, zeros_like};
    use ferrotorch_core::{
        Device, DualTensor, Tensor, dual_cos, dual_exp, dual_log, dual_relu, dual_sigmoid,
        dual_sin, dual_tanh, jacfwd, jvp_exact,
    };
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for CORE-031 tests");
        });
    }

    fn cuda_tensor(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        cpu_tensor(data, shape)
            .to(Device::Cuda(0))
            .expect("upload to cuda")
    }

    fn read_cuda(t: &Tensor<f32>) -> Vec<f32> {
        assert_eq!(
            t.device(),
            Device::Cuda(0),
            "tensor must stay CUDA-resident"
        );
        t.cpu().expect("readback").data_vec().expect("host data")
    }

    #[test]
    fn like_factories_preserve_cuda_device_and_values() {
        ensure_cuda_backend();
        let base = cuda_tensor(&[2.0, 4.0, 8.0], &[3]);

        let zeros = zeros_like(&base).expect("zeros_like cuda");
        assert_eq!(zeros.device(), Device::Cuda(0));
        assert_close(&read_cuda(&zeros), &[0.0, 0.0, 0.0], 0.0, "zeros_like");

        let ones = ones_like(&base).expect("ones_like cuda");
        assert_eq!(ones.device(), Device::Cuda(0));
        assert_close(&read_cuda(&ones), &[1.0, 1.0, 1.0], 0.0, "ones_like");

        let full = full_like(&base, -3.5).expect("full_like cuda");
        assert_eq!(full.device(), Device::Cuda(0));
        assert_close(&read_cuda(&full), &[-3.5, -3.5, -3.5], 0.0, "full_like");

        let rand = rand_like(&base).expect("rand_like cuda");
        assert_eq!(rand.device(), Device::Cuda(0));
        let rand_vals = read_cuda(&rand);
        assert!(
            rand_vals.iter().all(|v| (0.0..1.0).contains(v)),
            "rand_like values out of [0, 1): {rand_vals:?}"
        );

        let randn = randn_like(&base).expect("randn_like cuda");
        assert_eq!(randn.device(), Device::Cuda(0));
        assert_eq!(randn.shape(), &[3]);
    }

    #[test]
    fn dual_constant_allocates_zero_tangent_on_cuda() {
        ensure_cuda_backend();
        let primal = cuda_tensor(&[1.0, 2.0, 3.0], &[3]);
        let dual = DualTensor::constant(primal).expect("constant dual");

        assert_eq!(dual.primal.device(), Device::Cuda(0));
        assert_eq!(dual.tangent.device(), Device::Cuda(0));
        assert_close(
            &read_cuda(&dual.tangent),
            &[0.0, 0.0, 0.0],
            0.0,
            "constant tangent",
        );
    }

    #[test]
    fn elementary_forward_rules_keep_cuda_tangent_resident() {
        ensure_cuda_backend();
        let primal = cuda_tensor(&[-0.5, 0.25, 1.0], &[3]);
        let tangent = cuda_tensor(&[1.0, 2.0, 3.0], &[3]);
        let x = DualTensor::new(primal, tangent).expect("dual input");

        let y = dual_relu(&x).and_then(|v| dual_sigmoid(&v));
        let y = y
            .and_then(|v| dual_tanh(&v))
            .and_then(|v| dual_exp(&v))
            .and_then(|v| dual_log(&v))
            .and_then(|v| dual_sin(&v))
            .and_then(|v| dual_cos(&v))
            .expect("cuda elementary chain");

        assert_eq!(y.primal.device(), Device::Cuda(0));
        assert_eq!(y.tangent.device(), Device::Cuda(0));
        assert_eq!(y.tangent.shape(), &[3]);
    }

    #[test]
    fn jvp_exact_rejects_mixed_input_tangent_devices_like_torch() {
        ensure_cuda_backend();
        let input = cuda_tensor(&[1.0, 2.0], &[2]);
        let tangent = cpu_tensor(&[1.0, 1.0], &[2]);

        let err = jvp_exact(Ok, &input, &tangent).expect_err("mixed devices must fail");
        assert!(
            matches!(err, ferrotorch_core::FerrotorchError::DeviceMismatch { .. }),
            "expected DeviceMismatch, got {err:?}"
        );
    }

    #[test]
    fn jacfwd_cuda_nonlinear_chain_returns_cuda_jacobian() {
        ensure_cuda_backend();
        let input = cuda_tensor(&[0.25, 0.5, 0.75], &[3]);

        let jac = jacfwd(
            |x| {
                let e = dual_exp(&x)?;
                dual_sin(&e)
            },
            &input,
        )
        .expect("jacfwd cuda");

        assert_eq!(jac.device(), Device::Cuda(0));
        assert_eq!(jac.shape(), &[3, 3]);
        let expected_diag: Vec<f32> = [0.25_f32, 0.5, 0.75]
            .iter()
            .map(|&x| x.exp() * x.exp().cos())
            .collect();
        let expected = [
            expected_diag[0],
            0.0,
            0.0,
            0.0,
            expected_diag[1],
            0.0,
            0.0,
            0.0,
            expected_diag[2],
        ];
        assert_close(&read_cuda(&jac), &expected, 2e-4, "jacfwd cuda");
    }

    #[test]
    fn jacfwd_cuda_empty_input_keeps_empty_jacobian_on_cuda() {
        ensure_cuda_backend();
        let input = cuda_tensor(&[], &[0]);

        let jac = jacfwd(
            |x| {
                let out = cuda_tensor(&[7.0, 11.0], &[2]).to(x.primal.device())?;
                DualTensor::constant(out)
            },
            &input,
        )
        .expect("jacfwd cuda empty input");

        assert_eq!(jac.device(), Device::Cuda(0));
        assert_eq!(jac.shape(), &[2, 0]);
        assert_eq!(jac.numel(), 0);
    }
}
