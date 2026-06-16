//! CORE-170: `special.rs` must not silently detach PyTorch-differentiable ops.
//!
//! Live PyTorch 2.11.0+cu130 probes used for the numeric oracles:
//! - `erf/erfc/erfinv/lgamma/digamma/log1p/expm1/sinc`
//! - `special.entr/ndtr/ndtri/i0/i0e/i1/i1e`
//! - `xlogy`, `zeta`, `igamma`/`igammac`, `mvlgamma`
//!
//! PyTorch does *not* autograd-track the polynomial, K-Bessel, spherical
//! Bessel, or Airy families in this build, so those remain non-grad here.

use ferrotorch_core::error::FerrotorchError;
use ferrotorch_core::special;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

fn leaf(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}

fn plain(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

fn grad_after_sum<F>(data: &[f64], forward: F) -> Vec<f64>
where
    F: FnOnce(&Tensor<f64>) -> Tensor<f64>,
{
    let x = leaf(data, &[data.len()]);
    let out = forward(&x);
    assert!(
        out.requires_grad(),
        "PyTorch-differentiable special op must attach a grad_fn"
    );
    out.sum_all()
        .expect("scalar loss")
        .backward()
        .expect("backward");
    x.grad()
        .expect("grad slot")
        .expect("x grad")
        .data_vec()
        .expect("grad data")
}

fn assert_close(got: &[f64], want: &[f64], tol: f64, label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        if w.is_nan() {
            assert!(g.is_nan(), "{label}[{i}]: got {g}, want NaN");
        } else if w.is_infinite() {
            assert_eq!(g, w, "{label}[{i}]");
        } else {
            assert!(
                (g - w).abs() <= tol,
                "{label}[{i}]: got {g}, want {w}, tol {tol}"
            );
        }
    }
}

#[test]
fn unary_special_cpu_gradients_match_pytorch_formulas() {
    let xs = [0.2, -0.4, 1.1];
    let sqrt_pi = std::f64::consts::PI.sqrt();
    assert_close(
        &grad_after_sum(&xs, |x| special::erf(x).expect("erf")),
        &xs.map(|x| 2.0 / sqrt_pi * (-(x * x)).exp()),
        1e-12,
        "erf grad",
    );
    assert_close(
        &grad_after_sum(&xs, |x| special::erfc(x).expect("erfc")),
        &xs.map(|x| -2.0 / sqrt_pi * (-(x * x)).exp()),
        1e-12,
        "erfc grad",
    );

    assert_close(
        &grad_after_sum(&[-0.6, 0.0, 0.6], |x| special::erfinv(x).expect("erfinv")),
        &[1.2628624281411842, 0.8862269254527579, 1.2628624281411842],
        1e-12,
        "erfinv grad",
    );
    assert_close(
        &grad_after_sum(&[0.3, 1.2, 3.0], |x| special::lgamma(x).expect("lgamma")),
        &[-3.502524222200133, -0.28903989659218843, 0.922784335098467],
        1e-12,
        "lgamma grad",
    );
    assert_close(
        &grad_after_sum(&[0.3, 1.2, 3.0], |x| special::digamma(x).expect("digamma")),
        &[12.245364544940365, 1.2673772060383834, 0.39493406693194577],
        1e-12,
        "digamma grad",
    );

    let xs = [-0.4, 0.0, 1.5];
    assert_close(
        &grad_after_sum(&xs, |x| special::log1p(x).expect("log1p")),
        &[1.6666666666666667, 1.0, 0.4],
        1e-12,
        "log1p grad",
    );
    assert_close(
        &grad_after_sum(&xs, |x| special::expm1(x).expect("expm1")),
        &xs.map(f64::exp),
        1e-12,
        "expm1 grad",
    );
    assert_close(
        &grad_after_sum(&[0.0, 0.25, 1.25], |x| special::sinc(x).expect("sinc")),
        &[0.0, -0.7728381398822339, -0.4216348143641011],
        1e-12,
        "sinc grad",
    );
}

#[test]
fn distribution_and_bessel_i_cpu_gradients_match_pytorch_probe() {
    assert_close(
        &grad_after_sum(&[0.0, 0.2, 1.5], |x| special::entr(x).expect("entr")),
        &[f64::INFINITY, 0.6094379124341003, -1.4054651081081644],
        1e-12,
        "entr grad",
    );
    assert_close(
        &grad_after_sum(&[-1.0, 0.0, 1.0], |x| special::ndtr(x).expect("ndtr")),
        &[0.2419707245191433, 0.3989422804014327, 0.2419707245191433],
        1e-12,
        "ndtr grad",
    );
    assert_close(
        &grad_after_sum(&[0.1, 0.5, 0.9], |x| special::ndtri(x).expect("ndtri")),
        &[5.698059856117003, 2.5066282746310002, 5.698059856117003],
        1e-12,
        "ndtri grad",
    );
    assert_close(
        &grad_after_sum(&[-1.0, 0.0, 1.0], |x| special::i0(x).expect("i0")),
        &[-0.5651591039924851, 0.0, 0.5651591039924851],
        1e-12,
        "i0 grad",
    );
    assert_close(
        &grad_after_sum(&[-1.0, 0.0, 1.0], |x| special::i0e(x).expect("i0e")),
        &[0.25784919224393194, 0.0, -0.25784919224393194],
        1e-12,
        "i0e grad",
    );
    assert_close(
        &grad_after_sum(&[-1.0, 0.0, 1.0], |x| special::i1(x).expect("i1")),
        &[0.700906773759523, 0.5, 0.700906773759523],
        1e-12,
        "i1 grad",
    );
    assert_close(
        &grad_after_sum(&[-1.0, 0.0, 1.0], |x| special::i1e(x).expect("i1e")),
        &[0.049938776894223436, 0.5, 0.049938776894223436],
        1e-12,
        "i1e grad",
    );
}

#[test]
fn xlogy_backward_matches_pytorch_mask_convention() {
    let x = leaf(&[0.0, 1.5, -2.0], &[3]);
    let y = leaf(&[0.0, 2.0, 3.0], &[3]);
    let out = special::xlogy(&x, &y).expect("xlogy");
    assert!(out.requires_grad(), "xlogy must attach a grad_fn");
    out.sum_all().expect("loss").backward().expect("backward");

    assert_close(
        &x.grad()
            .expect("x grad slot")
            .expect("x grad")
            .data_vec()
            .expect("x grad data"),
        &[0.0, std::f64::consts::LN_2, 1.0986122886681098],
        1e-12,
        "xlogy x grad",
    );
    assert_close(
        &y.grad()
            .expect("y grad slot")
            .expect("y grad")
            .data_vec()
            .expect("y grad data"),
        &[f64::NAN, 0.75, -0.6666666666666666],
        1e-12,
        "xlogy y grad",
    );
}

#[test]
fn incomplete_gamma_backward_matches_pytorch_partial_semantics() {
    let a = plain(&[0.5, 1.5, 3.0], &[3]);
    let x = leaf(&[0.2, 2.0, 4.0], &[3]);
    let out = special::gammainc(&a, &x).expect("gammainc");
    assert!(out.requires_grad(), "gammainc must track differentiable x");
    out.sum_all().expect("loss").backward().expect("backward");
    assert_close(
        &x.grad()
            .expect("x grad slot")
            .expect("x grad")
            .data_vec()
            .expect("x grad data"),
        &[1.0328830949345562, 0.21596386605275225, 0.14652511110987348],
        1e-12,
        "gammainc x grad",
    );

    let a = plain(&[0.5, 1.5, 3.0], &[3]);
    let x = leaf(&[0.2, 2.0, 4.0], &[3]);
    special::gammaincc(&a, &x)
        .expect("gammaincc")
        .sum_all()
        .expect("loss")
        .backward()
        .expect("gammaincc backward");
    assert_close(
        &x.grad()
            .expect("x grad slot")
            .expect("x grad")
            .data_vec()
            .expect("x grad data"),
        &[
            -1.0328830949345562,
            -0.21596386605275225,
            -0.14652511110987348,
        ],
        1e-12,
        "gammaincc x grad",
    );

    let a = leaf(&[0.5, 1.5], &[2]);
    let x = plain(&[0.2, 2.0], &[2]);
    let err = special::gammainc(&a, &x)
        .expect("tracked a forward")
        .sum_all()
        .expect("loss")
        .backward()
        .expect_err("PyTorch raises for derivative wrt igamma input");
    assert!(
        matches!(err, FerrotorchError::InvalidArgument { ref message } if message.contains("igamma: input")),
        "unexpected error: {err:?}"
    );
}

#[test]
fn beta_and_mvlgamma_specials_are_not_detached() {
    let a = leaf(&[0.7, 2.5], &[2]);
    let b = leaf(&[1.2, 3.5], &[2]);
    special::log_beta(&a, &b)
        .expect("log_beta")
        .sum_all()
        .expect("loss")
        .backward()
        .expect("log_beta backward");
    assert_close(
        &a.grad()
            .expect("a grad slot")
            .expect("a grad")
            .data_vec()
            .expect("a grad data"),
        &[-1.5762077148619942, -1.002961027786557],
        1e-12,
        "log_beta a grad",
    );
    assert_close(
        &b.grad()
            .expect("b grad slot")
            .expect("b grad")
            .data_vec()
            .expect("b grad data"),
        &[-0.645224057756248, -0.6029610277865571],
        1e-12,
        "log_beta b grad",
    );

    let x = leaf(&[2.0, 3.0, 4.0], &[3]);
    let out = special::mvlgamma(&x, 3).expect("mvlgamma");
    assert!(out.requires_grad(), "mvlgamma must attach a grad_fn");
    out.sum_all()
        .expect("loss")
        .backward()
        .expect("mvlgamma backward");
    assert_close(
        &x.grad()
            .expect("x grad slot")
            .expect("x grad")
            .data_vec()
            .expect("x grad data"),
        &[-0.1179413558244895, 2.048725310842177, 3.2820586441755104],
        1e-12,
        "mvlgamma grad",
    );
}

#[test]
fn zeta_backward_matches_pytorch_partial_semantics() {
    let x = plain(&[2.0, 3.0, 4.0], &[3]);
    let q = leaf(&[1.0, 1.5, 2.0], &[3]);
    let out = special::zeta(&x, &q).expect("zeta");
    assert!(out.requires_grad(), "zeta must track differentiable q");
    out.sum_all().expect("loss").backward().expect("backward");
    assert_close(
        &q.grad()
            .expect("q grad slot")
            .expect("q grad")
            .data_vec()
            .expect("q grad data"),
        &[
            -2.404113806319188,
            -0.7045455170012184,
            -0.14771102057347976,
        ],
        1e-12,
        "zeta q grad",
    );

    let x = plain(&[2.0, 3.0], &[2, 1]);
    let q = leaf(&[1.0, 1.5, 2.0], &[1, 3]);
    special::zeta(&x, &q)
        .expect("broadcast zeta")
        .sum_all()
        .expect("loss")
        .backward()
        .expect("broadcast backward");
    assert_close(
        &q.grad()
            .expect("broadcast q grad slot")
            .expect("broadcast q grad")
            .data_vec()
            .expect("broadcast q grad data"),
        &[
            -5.6510835074526025,
            -1.5333421612355385,
            -0.6510835074526032,
        ],
        1e-12,
        "zeta broadcast q grad",
    );

    let x = leaf(&[2.0, 3.0], &[2]);
    let q = plain(&[1.0, 1.5], &[2]);
    let out = special::zeta(&x, &q).expect("tracked exponent zeta");
    assert!(
        out.requires_grad(),
        "PyTorch zeta creates a grad_fn when the exponent requires grad"
    );
    let err = out
        .sum_all()
        .expect("loss")
        .backward()
        .expect_err("zeta derivative wrt exponent is not implemented");
    assert!(
        matches!(err, FerrotorchError::InvalidArgument { ref message } if message.contains("zeta")),
        "unexpected error: {err:?}"
    );
}

#[test]
fn pytorch_non_differentiable_special_families_stay_untracked() {
    let x = leaf(&[0.5, 1.0, 2.0], &[3]);
    let non_grad = [
        special::spherical_bessel_j0(&x).expect("spherical_bessel_j0"),
        special::modified_bessel_k0(&x).expect("modified_bessel_k0"),
        special::modified_bessel_k1(&x).expect("modified_bessel_k1"),
        special::scaled_modified_bessel_k0(&x).expect("scaled_modified_bessel_k0"),
        special::scaled_modified_bessel_k1(&x).expect("scaled_modified_bessel_k1"),
        special::airy_ai(&x).expect("airy_ai"),
        special::chebyshev_polynomial_t(&x, 3).expect("chebyshev t"),
        special::chebyshev_polynomial_u(&x, 3).expect("chebyshev u"),
        special::hermite_polynomial_h(&x, 3).expect("hermite h"),
        special::laguerre_polynomial_l(&x, 3).expect("laguerre"),
        special::legendre_polynomial_p(&x, 3).expect("legendre"),
    ];
    for out in non_grad {
        assert!(
            !out.requires_grad(),
            "this family is non-differentiable in live PyTorch and must not invent a grad_fn"
        );
    }
}

#[cfg(feature = "gpu")]
mod cuda {
    use super::*;
    use std::sync::Once;

    use ferrotorch_core::creation::from_vec;
    use ferrotorch_core::device::Device;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for CORE-170 probes");
        });
    }

    fn cuda(data: &[f32]) -> Tensor<f32> {
        from_vec::<f32>(data.to_vec(), &[data.len()])
            .expect("cpu tensor")
            .to(Device::Cuda(0))
            .expect("upload")
            .requires_grad_(true)
    }

    fn host(t: &Tensor<f32>) -> Vec<f32> {
        assert_eq!(t.device(), Device::Cuda(0), "tensor must remain CUDA");
        t.cpu().expect("readback").data_vec().expect("host data")
    }

    fn assert_close_f32(got: &[f32], want: &[f32], tol: f32, label: &str) {
        assert_eq!(got.len(), want.len(), "{label}: length");
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (g - w).abs() <= tol,
                "{label}[{i}]: got {g}, want {w}, tol {tol}"
            );
        }
    }

    #[test]
    fn cuda_resident_special_backward_stays_on_device() {
        ensure_cuda_backend();

        let x = cuda(&[-1.0, 0.0, 1.0]);
        let out = special::ndtr(&x).expect("ndtr cuda");
        assert!(out.is_cuda(), "ndtr output must stay resident");
        assert!(out.requires_grad(), "ndtr cuda output must attach grad_fn");
        out.sum_all().expect("loss").backward().expect("backward");
        let grad = x.grad().expect("grad slot").expect("grad");
        assert!(grad.is_cuda(), "ndtr grad must stay resident");
        assert_close_f32(
            &host(&grad),
            &[0.24197075, 0.3989423, 0.24197075],
            2e-5,
            "ndtr cuda grad",
        );

        let x = cuda(&[-1.0, 0.0, 1.0]);
        let out = special::i1(&x).expect("i1 cuda");
        assert!(out.is_cuda(), "i1 output must stay resident");
        assert!(out.requires_grad(), "i1 cuda output must attach grad_fn");
        out.sum_all().expect("loss").backward().expect("backward");
        let grad = x.grad().expect("grad slot").expect("grad");
        assert!(grad.is_cuda(), "i1 grad must stay resident");
        assert_close_f32(
            &host(&grad),
            &[0.70090675, 0.5, 0.70090675],
            2e-5,
            "i1 cuda grad",
        );

        let x = cuda(&[-1.0, 0.0, 1.0]);
        let out = special::i0e(&x).expect("i0e cuda");
        assert!(out.is_cuda(), "i0e output must stay resident");
        assert!(out.requires_grad(), "i0e cuda output must attach grad_fn");
        out.sum_all().expect("loss").backward().expect("backward");
        let grad = x.grad().expect("grad slot").expect("grad");
        assert!(grad.is_cuda(), "i0e grad must stay resident");
        assert_close_f32(
            &host(&grad),
            &[0.2578492, 0.0, -0.2578492],
            2e-5,
            "i0e cuda grad",
        );
    }
}
