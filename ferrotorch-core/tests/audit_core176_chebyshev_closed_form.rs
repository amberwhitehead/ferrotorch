//! CORE-176: Chebyshev T/U/V/W must use PyTorch's high-order closed-form
//! trigonometric branch for interior inputs.
//!
//! Oracle: local PyTorch source at `/home/doll/pytorch/aten/src/ATen/native/Math.h`
//! and live local PyTorch CPU probes on 2026-06-18. The branch thresholds are
//! part of the contract: unshifted T uses `n > 6`, unshifted U/V/W use `n > 8`,
//! shifted T/U/V use `n > 6`, and shifted W uses `n > 4`.

use ferrotorch_core::special;
use ferrotorch_core::{Tensor, TensorStorage};

fn tensor<T: ferrotorch_core::Float>(data: Vec<T>) -> Tensor<T> {
    Tensor::from_storage(TensorStorage::cpu(data), vec![4], false).unwrap()
}

fn assert_close(actual: f64, expected: f64, rtol: f64, atol: f64, label: &str) {
    let limit = atol + rtol * expected.abs();
    let diff = (actual - expected).abs();
    assert!(
        diff <= limit,
        "{label}: actual={actual:?} expected={expected:?} diff={diff:?} limit={limit:?}"
    );
}

fn assert_close_vec(actual: &[f64], expected: &[f64], rtol: f64, atol: f64, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_close(actual, expected, rtol, atol, &format!("{label}[{i}]"));
    }
}

#[test]
fn cpu_f32_unshifted_chebyshev_high_order_matches_pytorch_closed_forms() {
    let x = tensor(vec![0.9999_f32, -0.75, -1.0, 1.0]);

    let t = special::chebyshev_polynomial_t(&x, 100).unwrap();
    assert_close_vec(
        &t.data()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .collect::<Vec<_>>(),
        &[0.1558161973953247, -0.9998588562011719, 1.0, 1.0],
        1e-5,
        1e-6,
        "chebyshev_t f32 n=100",
    );

    let u = special::chebyshev_polynomial_u(&x, 100).unwrap();
    assert_close(
        f64::from(u.data().unwrap()[0]),
        69.99180603027344,
        1e-6,
        2e-5,
        "chebyshev_u f32 n=100 x=0.9999",
    );

    let v = special::chebyshev_polynomial_v(&x, 20).unwrap();
    assert_close_vec(
        &v.data()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .collect::<Vec<_>>(),
        &[0.9582849144935608, 2.201282262802124, 41.0, 1.0],
        1e-5,
        1e-6,
        "chebyshev_v f32 n=20",
    );

    let w = special::chebyshev_polynomial_w(&x, 100).unwrap();
    assert_close(
        f64::from(w.data().unwrap()[0]),
        139.83477783203125,
        1e-6,
        3e-5,
        "chebyshev_w f32 n=100 x=0.9999",
    );
}

#[test]
fn cpu_f64_unshifted_chebyshev_high_order_matches_pytorch_closed_forms() {
    let x = tensor(vec![0.9999_f64, -0.75, -1.0, 1.0]);

    let t = special::chebyshev_polynomial_t(&x, 100).unwrap();
    assert_close_vec(
        t.data().unwrap(),
        &[0.15593205355938233, -0.9998589883151928, 1.0, 1.0],
        1e-12,
        1e-12,
        "chebyshev_t f64 n=100",
    );

    let u = special::chebyshev_polynomial_u(&x, 100).unwrap();
    assert_close(
        u.data().unwrap()[0],
        69.99642332950172,
        1e-12,
        1e-12,
        "chebyshev_u f64 n=100 x=0.9999",
    );

    let w = special::chebyshev_polynomial_w(&x, 100).unwrap();
    assert_close(
        w.data().unwrap()[0],
        139.8438993530464,
        1e-12,
        1e-12,
        "chebyshev_w f64 n=100 x=0.9999",
    );
}

#[test]
fn cpu_f32_shifted_chebyshev_uses_shifted_pytorch_thresholds() {
    let x = tensor(vec![0.9999_f32, 0.25, 0.0, 1.0]);

    let t = special::shifted_chebyshev_polynomial_t(&x, 100).unwrap();
    assert_close_vec(
        &t.data()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .collect::<Vec<_>>(),
        &[-0.4163280725479126, -0.5000033974647522, 1.0, 1.0],
        1e-5,
        1e-6,
        "shifted_chebyshev_t f32 n=100",
    );

    let u = special::shifted_chebyshev_polynomial_u(&x, 7).unwrap();
    assert_close_vec(
        &u.data()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .collect::<Vec<_>>(),
        &[7.966434955596924, -1.000000238418579, -8.0, 8.0],
        1e-5,
        1e-6,
        "shifted_chebyshev_u f32 n=7",
    );

    let v = special::shifted_chebyshev_polynomial_v(&x, 7).unwrap();
    assert_close_vec(
        &v.data()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .collect::<Vec<_>>(),
        &[0.9888182878494263, -2.000000238418579, -15.0, 1.0],
        1e-5,
        1e-6,
        "shifted_chebyshev_v f32 n=7",
    );

    let w = special::shifted_chebyshev_polynomial_w(&x, 5).unwrap();
    assert_close_vec(
        &w.data()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .collect::<Vec<_>>(),
        &[10.978008270263672, -0.9999996423721313, -1.0, 11.0],
        1e-5,
        1e-6,
        "shifted_chebyshev_w f32 n=5",
    );
}

#[test]
fn cpu_f64_shifted_chebyshev_uses_shifted_pytorch_thresholds() {
    let x = tensor(vec![0.9999_f64, 0.25, 0.0, 1.0]);

    let t = special::shifted_chebyshev_polynomial_t(&x, 100).unwrap();
    assert_close_vec(
        t.data().unwrap(),
        &[-0.41617714759407787, -0.5000000000000093, 1.0, 1.0],
        1e-12,
        1e-12,
        "shifted_chebyshev_t f64 n=100",
    );

    let u = special::shifted_chebyshev_polynomial_u(&x, 7).unwrap();
    assert_close_vec(
        u.data().unwrap(),
        &[7.966440298885635, -1.000000000000001, -8.0, 8.0],
        1e-12,
        1e-12,
        "shifted_chebyshev_u f64 n=7",
    );

    let v = special::shifted_chebyshev_polynomial_v(&x, 7).unwrap();
    assert_close_vec(
        v.data().unwrap(),
        &[0.9888201465642246, -2.0000000000000004, -15.0, 1.0],
        1e-12,
        1e-12,
        "shifted_chebyshev_v f64 n=7",
    );

    let w = special::shifted_chebyshev_polynomial_w(&x, 5).unwrap();
    assert_close_vec(
        w.data().unwrap(),
        &[10.978012317184284, -0.9999999999999994, -1.0, 11.0],
        1e-12,
        1e-12,
        "shifted_chebyshev_w f64 n=5",
    );
}

#[cfg(feature = "gpu")]
mod cuda {
    use super::{assert_close, assert_close_vec, special};
    use ferrotorch_core::Device;
    use ferrotorch_core::creation::from_vec;
    use ferrotorch_core::{Tensor, TensorStorage};

    fn ensure_cuda_backend() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            ferrotorch_gpu::init_cuda_backend().expect("CUDA backend must initialise");
        });
    }

    fn cuda_f32(data: &[f32]) -> ferrotorch_core::Tensor<f32> {
        from_vec(data.to_vec(), &[data.len()])
            .expect("construct cpu tensor")
            .to(Device::Cuda(0))
            .expect("H2D")
    }

    fn cuda_f64(data: &[f64]) -> ferrotorch_core::Tensor<f64> {
        from_vec(data.to_vec(), &[data.len()])
            .expect("construct cpu tensor")
            .to(Device::Cuda(0))
            .expect("H2D")
    }

    #[test]
    fn cuda_f32_public_chebyshev_closed_forms_stay_resident() {
        ensure_cuda_backend();
        let x = cuda_f32(&[0.9999]);

        let u = special::chebyshev_polynomial_u(&x, 100).unwrap();
        assert_eq!(u.device(), Device::Cuda(0), "U output must stay on cuda:0");
        let u_host = u
            .data_vec()
            .expect("explicit readback after residency check");
        assert_close(
            f64::from(u_host[0]),
            69.99180603027344,
            2e-6,
            2e-4,
            "cuda chebyshev_u f32 n=100 x=0.9999",
        );

        let w = special::chebyshev_polynomial_w(&x, 100).unwrap();
        assert_eq!(w.device(), Device::Cuda(0), "W output must stay on cuda:0");
        let w_host = w
            .data_vec()
            .expect("explicit readback after residency check");
        assert_close(
            f64::from(w_host[0]),
            139.83477783203125,
            2e-6,
            3e-4,
            "cuda chebyshev_w f32 n=100 x=0.9999",
        );

        let shifted = special::shifted_chebyshev_polynomial_t(&x, 100).unwrap();
        assert_eq!(
            shifted.device(),
            Device::Cuda(0),
            "shifted T output must stay on cuda:0"
        );
        let shifted_host = shifted
            .data_vec()
            .expect("explicit readback after residency check");
        assert_close(
            f64::from(shifted_host[0]),
            -0.4163280725479126,
            2e-6,
            2e-4,
            "cuda shifted_chebyshev_t f32 n=100 x=0.9999",
        );
    }

    #[test]
    fn cuda_f32_closed_form_masks_outside_domain_to_recurrence() {
        ensure_cuda_backend();
        let x = cuda_f32(&[2.0, -2.0, -1.0, 1.0, 0.0]);

        let t = special::chebyshev_polynomial_t(&x, 7).unwrap();
        assert_eq!(t.device(), Device::Cuda(0), "T output must stay on cuda:0");
        let host = t
            .data_vec()
            .expect("explicit readback after residency check");
        let actual = host.iter().map(|&v| f64::from(v)).collect::<Vec<_>>();
        assert_close_vec(
            &actual,
            &[5042.0, -5042.0, -1.0, 1.0, 6.636075795540819e-7],
            1e-5,
            2e-4,
            "cuda chebyshev_t f32 n=7 mixed-domain",
        );
    }

    #[test]
    fn cuda_chebyshev_is_forward_only_like_pytorch_special() {
        ensure_cuda_backend();
        let x = Tensor::from_storage(TensorStorage::cpu(vec![0.9999_f32]), vec![1], true)
            .unwrap()
            .to(Device::Cuda(0))
            .expect("H2D");
        assert!(
            x.requires_grad(),
            "test setup must use a gradient-tracked input"
        );

        let t = special::chebyshev_polynomial_t(&x, 100).unwrap();
        assert_eq!(t.device(), Device::Cuda(0), "T output must stay on cuda:0");
        assert!(
            !t.requires_grad(),
            "torch.special Chebyshev polynomial ops are forward-only"
        );
    }

    #[test]
    fn cuda_f64_public_chebyshev_closed_forms_stay_resident() {
        ensure_cuda_backend();
        let x = cuda_f64(&[0.9999]);

        let u = special::chebyshev_polynomial_u(&x, 100).unwrap();
        assert_eq!(u.device(), Device::Cuda(0), "U output must stay on cuda:0");
        let u_host = u
            .data_vec()
            .expect("explicit readback after residency check");
        assert_close(
            u_host[0],
            69.99642332950172,
            1e-10,
            1e-10,
            "cuda chebyshev_u f64 n=100 x=0.9999",
        );

        let w = special::chebyshev_polynomial_w(&x, 100).unwrap();
        assert_eq!(w.device(), Device::Cuda(0), "W output must stay on cuda:0");
        let w_host = w
            .data_vec()
            .expect("explicit readback after residency check");
        assert_close(
            w_host[0],
            139.8438993530464,
            1e-10,
            1e-10,
            "cuda chebyshev_w f64 n=100 x=0.9999",
        );

        let shifted = special::shifted_chebyshev_polynomial_t(&x, 100).unwrap();
        assert_eq!(
            shifted.device(),
            Device::Cuda(0),
            "shifted T output must stay on cuda:0"
        );
        let shifted_host = shifted
            .data_vec()
            .expect("explicit readback after residency check");
        assert_close(
            shifted_host[0],
            -0.41617714759407787,
            1e-10,
            1e-10,
            "cuda shifted_chebyshev_t f64 n=100 x=0.9999",
        );
    }
}
