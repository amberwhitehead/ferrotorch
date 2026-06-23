//! CORE-146 linalg autograd regression tests.
//!
//! These tests pin two contracts:
//! - ops that can be composed from existing differentiable primitives must
//!   actually propagate gradients, including CUDA where the primitives are
//!   resident;
//! - remaining forward-only paths must fail loudly on tracked inputs instead
//!   of returning plausible detached tensors.

use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::linalg;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

fn leaf(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}

fn plain(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

fn grad_data(t: &Tensor<f64>) -> Vec<f64> {
    t.grad()
        .expect("grad lookup")
        .expect("grad must be present")
        .data_vec()
        .expect("grad data")
}

fn assert_close(actual: &[f64], expected: &[f64], tol: f64, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch {} vs {}",
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "{label}[{i}]: actual={a}, expected={e}, diff={}",
            (a - e).abs()
        );
    }
}

fn fd_grad<F>(data: &[f64], shape: &[usize], eps: f64, f: F) -> Vec<f64>
where
    F: Fn(&Tensor<f64>) -> f64,
{
    let mut out = vec![0.0; data.len()];
    for i in 0..data.len() {
        let mut plus = data.to_vec();
        let mut minus = data.to_vec();
        plus[i] += eps;
        minus[i] -= eps;
        out[i] = (f(&plain(&plus, shape)) - f(&plain(&minus, shape))) / (2.0 * eps);
    }
    out
}

fn weighted_sum(t: &Tensor<f64>, weights: &[f64]) -> f64 {
    let data = t.data_vec().expect("weighted tensor data");
    assert_eq!(data.len(), weights.len(), "weighted_sum shape mismatch");
    data.iter().zip(weights).map(|(&x, &w)| x * w).sum()
}

fn assert_forward_only_autograd_error<T>(result: FerrotorchResult<T>, op: &str) {
    let err = match result {
        Ok(_) => panic!("tracked forward-only path must error"),
        Err(err) => err,
    };
    assert!(
        matches!(
            &err,
            FerrotorchError::InvalidArgument { message }
                if message.contains(op) && message.contains("refusing to return a detached tensor")
        ),
        "expected explicit detached-autograd refusal for {op}, got {err:?}"
    );
}

#[test]
fn svdvals_cpu_tracked_matches_finite_difference() {
    let data = [2.0, 0.3, -0.4, 1.7, 0.2, 0.8];
    let shape = [3, 2];
    let a = leaf(&data, &shape);
    let loss = linalg::svdvals(&a).unwrap().sum_all().unwrap();
    assert!(loss.requires_grad());
    loss.backward().unwrap();
    let analytic = grad_data(&a);

    let numeric = fd_grad(&data, &shape, 1e-6, |x| {
        linalg::svdvals(x)
            .unwrap()
            .sum_all()
            .unwrap()
            .item()
            .unwrap()
    });
    assert_close(&analytic, &numeric, 2e-5, "svdvals dA");
}

#[test]
fn matrix_power_negative_cpu_tracked_matches_finite_difference() {
    let data = [2.0, 0.3, 0.1, 1.5];
    let shape = [2, 2];
    let a = leaf(&data, &shape);
    let loss = linalg::matrix_power(&a, -2).unwrap().sum_all().unwrap();
    assert!(loss.requires_grad());
    loss.backward().unwrap();
    let analytic = grad_data(&a);

    let numeric = fd_grad(&data, &shape, 1e-6, |x| {
        linalg::matrix_power(x, -2)
            .unwrap()
            .sum_all()
            .unwrap()
            .item()
            .unwrap()
    });
    assert_close(&analytic, &numeric, 2e-5, "matrix_power(-2) dA");
}

#[test]
fn matrix_power_zero_tracks_zero_gradient_like_torch_copy_backward() {
    let data = [2.0, 0.3, 0.1, 1.5];
    let a = leaf(&data, &[2, 2]);
    let y = linalg::matrix_power(&a, 0).unwrap();
    assert!(
        y.requires_grad(),
        "torch returns a grad-tracking identity copy"
    );
    assert_close(y.data().unwrap(), &[1.0, 0.0, 0.0, 1.0], 0.0, "A^0");
    y.sum_all().unwrap().backward().unwrap();
    assert_close(&grad_data(&a), &[0.0, 0.0, 0.0, 0.0], 0.0, "A^0 grad");
}

#[test]
fn tensorsolve_cpu_tracked_matches_finite_difference_for_b() {
    let a_data = [
        2.0, 0.1, 0.0, 0.0, //
        0.0, 1.5, 0.2, 0.0, //
        0.0, 0.0, 1.7, 0.3, //
        0.1, 0.0, 0.0, 1.9,
    ];
    let b_data = [1.0, -0.5, 0.25, 2.0];
    let a = leaf(&a_data, &[2, 2, 2, 2]);
    let b = leaf(&b_data, &[2, 2]);
    let loss = linalg::tensorsolve(&a, &b).unwrap().sum_all().unwrap();
    assert!(loss.requires_grad());
    loss.backward().unwrap();
    let gb = grad_data(&b);

    let numeric_b = fd_grad(&b_data, &[2, 2], 1e-6, |bb| {
        let aa = plain(&a_data, &[2, 2, 2, 2]);
        linalg::tensorsolve(&aa, bb)
            .unwrap()
            .sum_all()
            .unwrap()
            .item()
            .unwrap()
    });
    assert_close(&gb, &numeric_b, 2e-5, "tensorsolve dB");
    assert!(
        a.grad().unwrap().is_some(),
        "tensorsolve must also propagate into A"
    );
}

#[test]
fn tensorinv_cpu_tracked_matches_finite_difference() {
    let data = [
        2.0, 0.1, 0.0, 0.0, //
        0.0, 1.5, 0.2, 0.0, //
        0.0, 0.0, 1.7, 0.3, //
        0.1, 0.0, 0.0, 1.9,
    ];
    let a = leaf(&data, &[2, 2, 2, 2]);
    let loss = linalg::tensorinv(&a, 2).unwrap().sum_all().unwrap();
    assert!(loss.requires_grad());
    loss.backward().unwrap();
    let analytic = grad_data(&a);

    let numeric = fd_grad(&data, &[2, 2, 2, 2], 1e-6, |x| {
        linalg::tensorinv(x, 2)
            .unwrap()
            .sum_all()
            .unwrap()
            .item()
            .unwrap()
    });
    assert_close(&analytic, &numeric, 3e-5, "tensorinv dA");
}

#[test]
fn cond_p2_cpu_tracked_matches_finite_difference() {
    let data = [2.0, 0.3, 0.1, 1.5];
    let a = leaf(&data, &[2, 2]);
    let c = linalg::cond(&a, 2.0).unwrap();
    assert!(c.requires_grad());
    c.backward().unwrap();
    let analytic = grad_data(&a);

    let numeric = fd_grad(&data, &[2, 2], 1e-6, |x| {
        linalg::cond(x, 2.0).unwrap().item().unwrap()
    });
    assert_close(&analytic, &numeric, 2e-5, "cond p=2 dA");
}

#[test]
fn solve_triangular_cpu_tracked_transpose_unit_matches_finite_difference() {
    let a_data = [
        3.0, 8.0, -7.0, //
        0.2, 4.0, 6.0, //
        -0.1, 0.3, 5.0,
    ];
    let b_data = [1.0, -0.5, 0.3, 2.0, -1.2, 0.7];
    let weights = [0.7, -0.2, 1.3, 0.5, -0.9, 0.4];
    let a = leaf(&a_data, &[3, 3]);
    let b = leaf(&b_data, &[3, 2]);
    let w = plain(&weights, &[3, 2]);
    let y = linalg::solve_triangular(&a, &b, false, true, true).unwrap();
    assert!(y.requires_grad());
    y.mul_t(&w).unwrap().sum_all().unwrap().backward().unwrap();
    let ga = grad_data(&a);
    let gb = grad_data(&b);

    let numeric_a = fd_grad(&a_data, &[3, 3], 1e-6, |aa| {
        let bb = plain(&b_data, &[3, 2]);
        let out = linalg::solve_triangular(aa, &bb, false, true, true).unwrap();
        weighted_sum(&out, &weights)
    });
    let numeric_b = fd_grad(&b_data, &[3, 2], 1e-6, |bb| {
        let aa = plain(&a_data, &[3, 3]);
        let out = linalg::solve_triangular(&aa, bb, false, true, true).unwrap();
        weighted_sum(&out, &weights)
    });
    assert_close(&ga, &numeric_a, 2e-5, "solve_triangular dA");
    assert_close(&gb, &numeric_b, 2e-5, "solve_triangular dB");
}

#[test]
fn matrix_exp_cpu_tracked_matches_finite_difference() {
    let data = [0.2, -0.3, 0.4, 0.1];
    let weights = [1.0, -0.25, 0.5, 1.5];
    let a = leaf(&data, &[2, 2]);
    let w = plain(&weights, &[2, 2]);
    let y = linalg::matrix_exp(&a).unwrap();
    assert!(y.requires_grad());
    y.mul_t(&w).unwrap().sum_all().unwrap().backward().unwrap();
    let analytic = grad_data(&a);

    let numeric = fd_grad(&data, &[2, 2], 1e-6, |x| {
        let out = linalg::matrix_exp(x).unwrap();
        weighted_sum(&out, &weights)
    });
    assert_close(&analytic, &numeric, 3e-5, "matrix_exp dA");
}

#[test]
fn lu_wide_cpu_tracked_matches_finite_difference() {
    let data = [4.0, 0.2, -0.1, 0.3, 3.0, 0.5];
    let l_weights = [0.7, -0.2, 1.1, 0.4];
    let u_weights = [0.5, -0.8, 0.3, 1.2, -0.4, 0.9];
    let a = leaf(&data, &[2, 3]);
    let (_p, l, u) = linalg::lu(&a).unwrap();
    assert!(l.requires_grad());
    assert!(u.requires_grad());
    let wl = plain(&l_weights, &[2, 2]);
    let wu = plain(&u_weights, &[2, 3]);
    l.mul_t(&wl)
        .unwrap()
        .sum_all()
        .unwrap()
        .add_t(&u.mul_t(&wu).unwrap().sum_all().unwrap())
        .unwrap()
        .backward()
        .unwrap();
    let analytic = grad_data(&a);

    let numeric = fd_grad(&data, &[2, 3], 1e-6, |x| {
        let (_p, l, u) = linalg::lu(x).unwrap();
        weighted_sum(&l, &l_weights) + weighted_sum(&u, &u_weights)
    });
    assert_close(&analytic, &numeric, 2e-4, "lu wide dA");
}

#[test]
fn lu_factor_tall_cpu_tracked_matches_finite_difference() {
    let data = [4.0, 0.2, 0.3, 3.0, 0.1, 0.4];
    let weights = [0.5, -0.8, 0.3, 1.2, -0.4, 0.9];
    let a = leaf(&data, &[3, 2]);
    let (lu, pivots) = linalg::lu_factor(&a).unwrap();
    assert_eq!(lu.shape(), &[3, 2]);
    assert_eq!(pivots.numel(), 2);
    assert_eq!(pivots.shape(), &[2]);
    assert!(lu.requires_grad());
    let w = plain(&weights, &[3, 2]);
    lu.mul_t(&w).unwrap().sum_all().unwrap().backward().unwrap();
    let analytic = grad_data(&a);

    let numeric = fd_grad(&data, &[3, 2], 1e-6, |x| {
        let (lu, _pivots) = linalg::lu_factor(x).unwrap();
        weighted_sum(&lu, &weights)
    });
    assert_close(&analytic, &numeric, 2e-4, "lu_factor tall dA");
}

#[test]
fn cond_rejects_non_torch_numeric_selector() {
    let a = plain(&[2.0, 0.3, 0.1, 1.5], &[2, 2]);
    let err = linalg::cond(&a, 3.0).unwrap_err();
    assert!(
        matches!(
            &err,
            FerrotorchError::InvalidArgument { message }
                if message.contains("linalg.cond got an invalid norm type: 3")
        ),
        "expected torch-style invalid cond selector error, got {err:?}"
    );
}

#[test]
fn forward_only_tracked_paths_refuse_to_detach() {
    let a = leaf(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    assert_forward_only_autograd_error(linalg::ldl_factor(&a), "ldl_factor");
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use std::sync::Once;

    use ferrotorch_core::Device;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for CORE-146 GPU tests");
        });
    }

    fn cuda_leaf(data: &[f64], shape: &[usize]) -> Tensor<f64> {
        ensure_cuda_backend();
        plain(data, shape)
            .to(Device::Cuda(0))
            .expect("upload")
            .requires_grad_(true)
    }

    fn cpu_leaf(data: &[f64], shape: &[usize]) -> Tensor<f64> {
        leaf(data, shape)
    }

    fn grad_vec(t: &Tensor<f64>) -> Vec<f64> {
        t.grad()
            .expect("grad lookup")
            .expect("grad present")
            .data_vec()
            .expect("grad data")
    }

    #[test]
    fn solve_cuda_backward_stays_on_device_and_matches_cpu() {
        let a_data = [3.0, 0.5, 0.25, 2.0];
        let b_data = [1.0, -0.5];

        let a_gpu = cuda_leaf(&a_data, &[2, 2]);
        let b_gpu = cuda_leaf(&b_data, &[2]);
        let loss_gpu = linalg::solve(&a_gpu, &b_gpu).unwrap().sum_all().unwrap();
        assert!(loss_gpu.requires_grad());
        loss_gpu.backward().unwrap();
        let ga_gpu = a_gpu.grad().unwrap().unwrap();
        let gb_gpu = b_gpu.grad().unwrap().unwrap();
        assert_eq!(ga_gpu.device(), Device::Cuda(0));
        assert_eq!(gb_gpu.device(), Device::Cuda(0));

        let a_cpu = cpu_leaf(&a_data, &[2, 2]);
        let b_cpu = cpu_leaf(&b_data, &[2]);
        linalg::solve(&a_cpu, &b_cpu)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();

        assert_close(
            &ga_gpu.data_vec().unwrap(),
            &grad_vec(&a_cpu),
            2e-8,
            "solve dA",
        );
        assert_close(
            &gb_gpu.data_vec().unwrap(),
            &grad_vec(&b_cpu),
            2e-8,
            "solve dB",
        );
    }

    #[test]
    fn solve_triangular_cuda_transpose_unit_backward_matches_cpu() {
        let a_data = [
            3.0, 8.0, -7.0, //
            0.2, 4.0, 6.0, //
            -0.1, 0.3, 5.0,
        ];
        let b_data = [1.0, -0.5, 0.3, 2.0, -1.2, 0.7];
        let weights = [0.7, -0.2, 1.3, 0.5, -0.9, 0.4];

        let a_gpu = cuda_leaf(&a_data, &[3, 3]);
        let b_gpu = cuda_leaf(&b_data, &[3, 2]);
        let w_gpu = plain(&weights, &[3, 2]).to(Device::Cuda(0)).unwrap();
        linalg::solve_triangular(&a_gpu, &b_gpu, false, true, true)
            .unwrap()
            .mul_t(&w_gpu)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        let ga_gpu = a_gpu.grad().unwrap().unwrap();
        let gb_gpu = b_gpu.grad().unwrap().unwrap();
        assert_eq!(ga_gpu.device(), Device::Cuda(0));
        assert_eq!(gb_gpu.device(), Device::Cuda(0));

        let a_cpu = cpu_leaf(&a_data, &[3, 3]);
        let b_cpu = cpu_leaf(&b_data, &[3, 2]);
        let w_cpu = plain(&weights, &[3, 2]);
        linalg::solve_triangular(&a_cpu, &b_cpu, false, true, true)
            .unwrap()
            .mul_t(&w_cpu)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();

        assert_close(
            &ga_gpu.data_vec().unwrap(),
            &grad_vec(&a_cpu),
            2e-8,
            "solve_triangular CUDA dA",
        );
        assert_close(
            &gb_gpu.data_vec().unwrap(),
            &grad_vec(&b_cpu),
            2e-8,
            "solve_triangular CUDA dB",
        );
    }

    #[test]
    fn matrix_exp_cuda_backward_stays_on_device_and_matches_cpu() {
        let data = [0.1, 0.2, -0.3, 0.4];
        let weights = [0.7, -0.2, 1.3, 0.5];

        let a_gpu = cuda_leaf(&data, &[2, 2]);
        let w_gpu = plain(&weights, &[2, 2]).to(Device::Cuda(0)).unwrap();
        let y_gpu = linalg::matrix_exp(&a_gpu).unwrap();
        assert_eq!(y_gpu.device(), Device::Cuda(0));
        y_gpu
            .mul_t(&w_gpu)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        let ga_gpu = a_gpu.grad().unwrap().unwrap();
        assert_eq!(ga_gpu.device(), Device::Cuda(0));

        let a_cpu = cpu_leaf(&data, &[2, 2]);
        let w_cpu = plain(&weights, &[2, 2]);
        let y_cpu = linalg::matrix_exp(&a_cpu).unwrap();
        y_cpu
            .mul_t(&w_cpu)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();

        assert_close(
            &y_gpu.data_vec().unwrap(),
            y_cpu.data_vec().unwrap().as_slice(),
            1e-8,
            "matrix_exp CUDA forward",
        );
        assert_close(
            &ga_gpu.data_vec().unwrap(),
            &grad_vec(&a_cpu),
            2e-7,
            "matrix_exp CUDA dA",
        );
    }

    #[test]
    fn matrix_norm_cuda_backward_stays_on_device_and_matches_cpu() {
        let data = [1.0, -2.0, 0.5, 3.0];
        let x_gpu = cuda_leaf(&data, &[2, 2]);
        let loss_gpu = linalg::matrix_norm(&x_gpu).unwrap();
        assert!(loss_gpu.requires_grad());
        loss_gpu.backward().unwrap();
        let gx_gpu = x_gpu.grad().unwrap().unwrap();
        assert_eq!(gx_gpu.device(), Device::Cuda(0));

        let x_cpu = cpu_leaf(&data, &[2, 2]);
        linalg::matrix_norm(&x_cpu).unwrap().backward().unwrap();
        assert_close(
            &gx_gpu.data_vec().unwrap(),
            &grad_vec(&x_cpu),
            2e-8,
            "matrix_norm dX",
        );
    }

    #[test]
    fn matrix_power_cuda_negative_backward_stays_on_device_and_matches_cpu() {
        let data = [2.0, 0.3, 0.1, 1.5];
        let a_gpu = cuda_leaf(&data, &[2, 2]);
        linalg::matrix_power(&a_gpu, -2)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        let ga_gpu = a_gpu.grad().unwrap().unwrap();
        assert_eq!(ga_gpu.device(), Device::Cuda(0));

        let a_cpu = cpu_leaf(&data, &[2, 2]);
        linalg::matrix_power(&a_cpu, -2)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        assert_close(
            &ga_gpu.data_vec().unwrap(),
            &grad_vec(&a_cpu),
            2e-8,
            "matrix_power CUDA dA",
        );
    }

    #[test]
    fn cholesky_cuda_backward_stays_on_device_and_matches_cpu() {
        let data = [4.0, 0.8, 0.8, 3.0];
        let a_gpu = cuda_leaf(&data, &[2, 2]);
        linalg::cholesky(&a_gpu)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        let ga_gpu = a_gpu.grad().unwrap().unwrap();
        assert_eq!(ga_gpu.device(), Device::Cuda(0));

        let a_cpu = cpu_leaf(&data, &[2, 2]);
        linalg::cholesky(&a_cpu)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        assert_close(
            &ga_gpu.data_vec().unwrap(),
            &grad_vec(&a_cpu),
            2e-8,
            "cholesky CUDA dA",
        );
    }

    #[test]
    fn qr_cuda_backward_stays_on_device_and_matches_cpu() {
        let data = [2.0, 0.3, -0.4, 1.7, 0.2, 0.8];
        let a_gpu = cuda_leaf(&data, &[3, 2]);
        let (q_gpu, r_gpu) = linalg::qr(&a_gpu).unwrap();
        q_gpu
            .sum_all()
            .unwrap()
            .add_t(&r_gpu.sum_all().unwrap())
            .unwrap()
            .backward()
            .unwrap();
        let ga_gpu = a_gpu.grad().unwrap().unwrap();
        assert_eq!(ga_gpu.device(), Device::Cuda(0));

        let a_cpu = cpu_leaf(&data, &[3, 2]);
        let (q_cpu, r_cpu) = linalg::qr(&a_cpu).unwrap();
        q_cpu
            .sum_all()
            .unwrap()
            .add_t(&r_cpu.sum_all().unwrap())
            .unwrap()
            .backward()
            .unwrap();
        assert_close(
            &ga_gpu.data_vec().unwrap(),
            &grad_vec(&a_cpu),
            2e-8,
            "qr CUDA dA",
        );
    }

    #[test]
    fn eigvalsh_cuda_backward_stays_on_device_and_matches_cpu() {
        let data = [2.0, 0.3, 0.3, 1.5];
        let weights = [0.7, -1.3];
        let a_gpu = cuda_leaf(&data, &[2, 2]);
        let w_gpu = plain(&weights, &[2]).to(Device::Cuda(0)).unwrap();
        linalg::eigvalsh(&a_gpu)
            .unwrap()
            .mul_t(&w_gpu)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        let ga_gpu = a_gpu.grad().unwrap().unwrap();
        assert_eq!(ga_gpu.device(), Device::Cuda(0));

        let a_cpu = cpu_leaf(&data, &[2, 2]);
        let w_cpu = plain(&weights, &[2]);
        linalg::eigvalsh(&a_cpu)
            .unwrap()
            .mul_t(&w_cpu)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        assert_close(
            &ga_gpu.data_vec().unwrap(),
            &grad_vec(&a_cpu),
            2e-8,
            "eigvalsh CUDA dA",
        );
    }

    #[test]
    fn eigh_cuda_backward_stays_on_device_and_matches_cpu() {
        let data = [2.0, 0.3, 0.3, 1.5];
        let vec_weights = [0.2, -0.7, 1.1, 0.4];
        let a_gpu = cuda_leaf(&data, &[2, 2]);
        let weights_gpu = plain(&vec_weights, &[2, 2]).to(Device::Cuda(0)).unwrap();
        let (w_gpu, u_gpu) = linalg::eigh(&a_gpu).unwrap();
        let loss_gpu = w_gpu
            .sum_all()
            .unwrap()
            .add_t(
                &u_gpu
                    .mul_t(&u_gpu)
                    .unwrap()
                    .mul_t(&weights_gpu)
                    .unwrap()
                    .sum_all()
                    .unwrap(),
            )
            .unwrap();
        loss_gpu.backward().unwrap();
        let ga_gpu = a_gpu.grad().unwrap().unwrap();
        assert_eq!(ga_gpu.device(), Device::Cuda(0));

        let a_cpu = cpu_leaf(&data, &[2, 2]);
        let weights_cpu = plain(&vec_weights, &[2, 2]);
        let (w_cpu, u_cpu) = linalg::eigh(&a_cpu).unwrap();
        let loss_cpu = w_cpu
            .sum_all()
            .unwrap()
            .add_t(
                &u_cpu
                    .mul_t(&u_cpu)
                    .unwrap()
                    .mul_t(&weights_cpu)
                    .unwrap()
                    .sum_all()
                    .unwrap(),
            )
            .unwrap();
        loss_cpu.backward().unwrap();
        assert_close(
            &ga_gpu.data_vec().unwrap(),
            &grad_vec(&a_cpu),
            2e-8,
            "eigh CUDA dA",
        );
    }

    fn svd_loss(u: &Tensor<f64>, s: &Tensor<f64>, vh: &Tensor<f64>, device: Device) -> Tensor<f64> {
        let u_pattern = [0.2, -0.7, 1.1, 0.4, -0.3, 0.9];
        let vh_pattern = [0.6, -0.5, 0.8, -1.1, 0.3, 0.2];
        let u_weights_data: Vec<f64> = (0..u.numel())
            .map(|i| u_pattern[i % u_pattern.len()])
            .collect();
        let vh_weights_data: Vec<f64> = (0..vh.numel())
            .map(|i| vh_pattern[i % vh_pattern.len()])
            .collect();
        let u_weights = plain(&u_weights_data, u.shape()).to(device).unwrap();
        let vh_weights = plain(&vh_weights_data, vh.shape()).to(device).unwrap();
        let s_weights = plain(&[0.9, -0.4], s.shape()).to(device).unwrap();
        s.mul_t(&s_weights)
            .unwrap()
            .sum_all()
            .unwrap()
            .add_t(
                &u.mul_t(u)
                    .unwrap()
                    .mul_t(&u_weights)
                    .unwrap()
                    .sum_all()
                    .unwrap(),
            )
            .unwrap()
            .add_t(
                &vh.mul_t(vh)
                    .unwrap()
                    .mul_t(&vh_weights)
                    .unwrap()
                    .sum_all()
                    .unwrap(),
            )
            .unwrap()
    }

    #[test]
    fn svdvals_cuda_backward_stays_on_device_and_matches_cpu() {
        let data = [2.0, 0.3, -0.4, 1.7, 0.2, 0.8];
        let weights = [0.9, -0.4];
        let a_gpu = cuda_leaf(&data, &[3, 2]);
        let w_gpu = plain(&weights, &[2]).to(Device::Cuda(0)).unwrap();
        linalg::svdvals(&a_gpu)
            .unwrap()
            .mul_t(&w_gpu)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        let ga_gpu = a_gpu.grad().unwrap().unwrap();
        assert_eq!(ga_gpu.device(), Device::Cuda(0));

        let a_cpu = cpu_leaf(&data, &[3, 2]);
        let w_cpu = plain(&weights, &[2]);
        linalg::svdvals(&a_cpu)
            .unwrap()
            .mul_t(&w_cpu)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        assert_close(
            &ga_gpu.data_vec().unwrap(),
            &grad_vec(&a_cpu),
            2e-8,
            "svdvals CUDA dA",
        );
    }

    #[test]
    fn svd_cuda_tall_backward_stays_on_device_and_matches_cpu() {
        let data = [2.0, 0.3, -0.4, 1.7, 0.2, 0.8];
        let a_gpu = cuda_leaf(&data, &[3, 2]);
        let (u_gpu, s_gpu, vh_gpu) = linalg::svd(&a_gpu).unwrap();
        svd_loss(&u_gpu, &s_gpu, &vh_gpu, Device::Cuda(0))
            .backward()
            .unwrap();
        let ga_gpu = a_gpu.grad().unwrap().unwrap();
        assert_eq!(ga_gpu.device(), Device::Cuda(0));

        let a_cpu = cpu_leaf(&data, &[3, 2]);
        let (u_cpu, s_cpu, vh_cpu) = linalg::svd(&a_cpu).unwrap();
        svd_loss(&u_cpu, &s_cpu, &vh_cpu, Device::Cpu)
            .backward()
            .unwrap();
        assert_close(
            &ga_gpu.data_vec().unwrap(),
            &grad_vec(&a_cpu),
            2e-8,
            "svd tall CUDA dA",
        );
    }

    #[test]
    fn svd_cuda_wide_backward_stays_on_device_and_matches_cpu() {
        let data = [2.0, 0.3, -0.4, 1.7, 0.2, 0.8];
        let a_gpu = cuda_leaf(&data, &[2, 3]);
        let (u_gpu, s_gpu, vh_gpu) = linalg::svd(&a_gpu).unwrap();
        svd_loss(&u_gpu, &s_gpu, &vh_gpu, Device::Cuda(0))
            .backward()
            .unwrap();
        let ga_gpu = a_gpu.grad().unwrap().unwrap();
        assert_eq!(ga_gpu.device(), Device::Cuda(0));

        let a_cpu = cpu_leaf(&data, &[2, 3]);
        let (u_cpu, s_cpu, vh_cpu) = linalg::svd(&a_cpu).unwrap();
        svd_loss(&u_cpu, &s_cpu, &vh_cpu, Device::Cpu)
            .backward()
            .unwrap();
        assert_close(
            &ga_gpu.data_vec().unwrap(),
            &grad_vec(&a_cpu),
            2e-8,
            "svd wide CUDA dA",
        );
    }

    #[test]
    fn pinv_cuda_backward_stays_on_device_and_matches_cpu() {
        let data = [2.0, 0.3, -0.4, 1.7, 0.2, 0.8];
        let weights = [0.7, -0.2, 1.1, 0.4, -0.8, 0.3];

        let a_gpu = cuda_leaf(&data, &[3, 2]);
        let w_gpu = plain(&weights, &[2, 3]).to(Device::Cuda(0)).unwrap();
        let p_gpu = linalg::pinv(&a_gpu).unwrap();
        assert_eq!(p_gpu.device(), Device::Cuda(0));
        assert_eq!(p_gpu.shape(), &[2, 3]);
        p_gpu
            .mul_t(&w_gpu)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        let ga_gpu = a_gpu.grad().unwrap().unwrap();
        assert_eq!(ga_gpu.device(), Device::Cuda(0));

        let a_cpu = cpu_leaf(&data, &[3, 2]);
        let w_cpu = plain(&weights, &[2, 3]);
        linalg::pinv(&a_cpu)
            .unwrap()
            .mul_t(&w_cpu)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        assert_close(
            &ga_gpu.data_vec().unwrap(),
            &grad_vec(&a_cpu),
            2e-8,
            "pinv CUDA dA",
        );
    }

    #[test]
    fn lstsq_solve_cuda_backward_stays_on_device_and_matches_cpu() {
        let a_data = [1.0, 0.5, 2.0, -1.0, 0.3, 1.5, -0.7, 2.0];
        let b_data = [1.0, -0.5, 0.8, 1.2, -0.3, 0.6, 2.0, -1.0];
        let weights = [0.7, -1.1, 0.4, 1.3];

        let a_gpu = cuda_leaf(&a_data, &[4, 2]);
        let b_gpu = cuda_leaf(&b_data, &[4, 2]);
        let w_gpu = plain(&weights, &[2, 2]).to(Device::Cuda(0)).unwrap();
        let x_gpu = linalg::lstsq_solve(&a_gpu, &b_gpu).unwrap();
        assert_eq!(x_gpu.device(), Device::Cuda(0));
        assert!(x_gpu.requires_grad());
        x_gpu
            .mul_t(&w_gpu)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        let ga_gpu = a_gpu.grad().unwrap().unwrap();
        let gb_gpu = b_gpu.grad().unwrap().unwrap();
        assert_eq!(ga_gpu.device(), Device::Cuda(0));
        assert_eq!(gb_gpu.device(), Device::Cuda(0));

        let a_cpu = cpu_leaf(&a_data, &[4, 2]);
        let b_cpu = cpu_leaf(&b_data, &[4, 2]);
        let w_cpu = plain(&weights, &[2, 2]);
        linalg::lstsq_solve(&a_cpu, &b_cpu)
            .unwrap()
            .mul_t(&w_cpu)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();

        assert_close(
            &ga_gpu.data_vec().unwrap(),
            &grad_vec(&a_cpu),
            5e-8,
            "lstsq_solve CUDA dA",
        );
        assert_close(
            &gb_gpu.data_vec().unwrap(),
            &grad_vec(&b_cpu),
            5e-8,
            "lstsq_solve CUDA dB",
        );
    }

    fn manual_lstsq_residual_loss(a: &Tensor<f64>, b: &Tensor<f64>, weights: &[f64]) -> f64 {
        let x = linalg::lstsq_solve(a, b).unwrap();
        let r = a.mm(&x).unwrap().sub_t(b).unwrap();
        let r2 = r.mul_t(&r).unwrap();
        let residuals = ferrotorch_core::grad_fns::reduction::sum_dim(&r2, 0, false).unwrap();
        weighted_sum(&residuals, weights)
    }

    #[test]
    fn lstsq_cuda_residuals_backward_stays_on_device_and_matches_finite_difference() {
        let a_data = [1.0, 0.5, 2.0, -1.0, 0.3, 1.5, -0.7, 2.0];
        let b_data = [1.0, -0.5, 0.8, 1.2, -0.3, 0.6, 2.0, -1.0];
        let weights = [0.6, -0.4];

        let a_gpu = cuda_leaf(&a_data, &[4, 2]);
        let b_gpu = cuda_leaf(&b_data, &[4, 2]);
        let w_gpu = plain(&weights, &[2]).to(Device::Cuda(0)).unwrap();
        let (_x, residuals_gpu, rank_gpu, sv_gpu) = linalg::lstsq(&a_gpu, &b_gpu, None).unwrap();
        assert_eq!(residuals_gpu.device(), Device::Cuda(0));
        assert_eq!(residuals_gpu.shape(), &[2]);
        assert!(residuals_gpu.requires_grad());
        assert_eq!(rank_gpu.shape(), &[0]);
        assert_eq!(sv_gpu.shape(), &[0]);
        residuals_gpu
            .mul_t(&w_gpu)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        let ga_gpu = a_gpu.grad().unwrap().unwrap();
        let gb_gpu = b_gpu.grad().unwrap().unwrap();
        assert_eq!(ga_gpu.device(), Device::Cuda(0));
        assert_eq!(gb_gpu.device(), Device::Cuda(0));

        let numeric_a = fd_grad(&a_data, &[4, 2], 1e-6, |aa| {
            let bb = plain(&b_data, &[4, 2]);
            manual_lstsq_residual_loss(aa, &bb, &weights)
        });
        let numeric_b = fd_grad(&b_data, &[4, 2], 1e-6, |bb| {
            let aa = plain(&a_data, &[4, 2]);
            manual_lstsq_residual_loss(&aa, bb, &weights)
        });

        assert_close(
            &ga_gpu.data_vec().unwrap(),
            &numeric_a,
            2e-5,
            "lstsq residual CUDA dA",
        );
        assert_close(
            &gb_gpu.data_vec().unwrap(),
            &numeric_b,
            2e-5,
            "lstsq residual CUDA dB",
        );
    }

    #[test]
    fn lu_factor_cuda_tall_backward_stays_on_device_and_matches_cpu() {
        let data = [4.0, 0.2, 0.3, 3.0, 0.1, 0.4];
        let weights = [0.5, -0.8, 0.3, 1.2, -0.4, 0.9];

        let a_gpu = cuda_leaf(&data, &[3, 2]);
        let w_gpu = plain(&weights, &[3, 2]).to(Device::Cuda(0)).unwrap();
        let (lu_gpu, pivots_gpu) = linalg::lu_factor(&a_gpu).unwrap();
        assert_eq!(lu_gpu.device(), Device::Cuda(0));
        assert_eq!(lu_gpu.shape(), &[3, 2]);
        assert_eq!(pivots_gpu.device(), Device::Cuda(0));
        assert_eq!(pivots_gpu.numel(), 2);
        lu_gpu
            .mul_t(&w_gpu)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        let ga_gpu = a_gpu.grad().unwrap().unwrap();
        assert_eq!(ga_gpu.device(), Device::Cuda(0));

        let a_cpu = cpu_leaf(&data, &[3, 2]);
        let w_cpu = plain(&weights, &[3, 2]);
        let (lu_cpu, pivots_cpu) = linalg::lu_factor(&a_cpu).unwrap();
        let pivots_gpu_cpu = pivots_gpu.to(Device::Cpu).unwrap();
        assert_eq!(pivots_cpu.data().unwrap(), pivots_gpu_cpu.data().unwrap());
        lu_cpu
            .mul_t(&w_cpu)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        assert_close(
            &ga_gpu.data_vec().unwrap(),
            &grad_vec(&a_cpu),
            2e-8,
            "lu_factor CUDA tall dA",
        );
    }

    #[test]
    fn lu_cuda_wide_backward_stays_on_device_and_matches_cpu() {
        let data = [4.0, 0.2, -0.1, 0.3, 3.0, 0.5];
        let l_weights = [0.7, -0.2, 1.1, 0.4];
        let u_weights = [0.5, -0.8, 0.3, 1.2, -0.4, 0.9];

        let a_gpu = cuda_leaf(&data, &[2, 3]);
        let wl_gpu = plain(&l_weights, &[2, 2]).to(Device::Cuda(0)).unwrap();
        let wu_gpu = plain(&u_weights, &[2, 3]).to(Device::Cuda(0)).unwrap();
        let (p_gpu, l_gpu, u_gpu) = linalg::lu(&a_gpu).unwrap();
        assert_eq!(p_gpu.device(), Device::Cuda(0));
        assert_eq!(l_gpu.device(), Device::Cuda(0));
        assert_eq!(u_gpu.device(), Device::Cuda(0));
        l_gpu
            .mul_t(&wl_gpu)
            .unwrap()
            .sum_all()
            .unwrap()
            .add_t(&u_gpu.mul_t(&wu_gpu).unwrap().sum_all().unwrap())
            .unwrap()
            .backward()
            .unwrap();
        let ga_gpu = a_gpu.grad().unwrap().unwrap();
        assert_eq!(ga_gpu.device(), Device::Cuda(0));

        let a_cpu = cpu_leaf(&data, &[2, 3]);
        let wl_cpu = plain(&l_weights, &[2, 2]);
        let wu_cpu = plain(&u_weights, &[2, 3]);
        let (_p_cpu, l_cpu, u_cpu) = linalg::lu(&a_cpu).unwrap();
        l_cpu
            .mul_t(&wl_cpu)
            .unwrap()
            .sum_all()
            .unwrap()
            .add_t(&u_cpu.mul_t(&wu_cpu).unwrap().sum_all().unwrap())
            .unwrap()
            .backward()
            .unwrap();
        assert_close(
            &ga_gpu.data_vec().unwrap(),
            &grad_vec(&a_cpu),
            2e-8,
            "lu CUDA wide dA",
        );
    }
}
