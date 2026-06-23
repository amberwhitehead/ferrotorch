//! CORE-1989: linalg norm/rank/cond parity.
//!
//! PyTorch references:
//! - `/home/doll/pytorch/aten/src/ATen/native/LinearAlgebra.cpp`
//!   `linalg_vector_norm_out`, `matrix_rank_impl`, `linalg_cond`
//! - `/home/doll/pytorch/tools/autograd/derivatives.yaml`
//!   `linalg_vector_norm`

use ferrotorch_core::linalg::{
    MatrixNormOrder, cond, matrix_norm, matrix_norm_ord, matrix_rank, vector_norm,
};
use ferrotorch_core::{Device, Tensor, TensorStorage};
use half::{bf16, f16};

fn tensor_f32(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
    Tensor::from_storage(
        TensorStorage::cpu(data.to_vec()),
        shape.to_vec(),
        requires_grad,
    )
    .unwrap()
}

fn tensor_f64(data: &[f64], shape: &[usize], requires_grad: bool) -> Tensor<f64> {
    Tensor::from_storage(
        TensorStorage::cpu(data.to_vec()),
        shape.to_vec(),
        requires_grad,
    )
    .unwrap()
}

fn tensor_f16(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f16> {
    Tensor::from_storage(
        TensorStorage::cpu(data.iter().copied().map(f16::from_f32).collect()),
        shape.to_vec(),
        requires_grad,
    )
    .unwrap()
}

fn tensor_bf16(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<bf16> {
    Tensor::from_storage(
        TensorStorage::cpu(data.iter().copied().map(bf16::from_f32).collect()),
        shape.to_vec(),
        requires_grad,
    )
    .unwrap()
}

fn assert_close_f32(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "{label}[{i}] actual={a} expected={e} diff={}",
            (a - e).abs()
        );
    }
}

fn assert_close_f64(actual: &[f64], expected: &[f64], tol: f64, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "{label}[{i}] actual={a} expected={e} diff={}",
            (a - e).abs()
        );
    }
}

#[test]
fn matrix_norm_ord_cpu_matches_torch_order_set() {
    // Live torch 2.11 oracle for:
    // A = torch.tensor([[1., -2., 3.], [4., -5., 6.]], dtype=torch.float64)
    let a = tensor_f64(&[1.0, -2.0, 3.0, 4.0, -5.0, 6.0], &[2, 3], false);
    let cases = [
        (MatrixNormOrder::Fro, 9.539_392_014_169_456),
        (MatrixNormOrder::Nuclear, 10.280_901_636_369_208),
        (MatrixNormOrder::One, 9.0),
        (MatrixNormOrder::NegOne, 5.0),
        (MatrixNormOrder::Two, 9.508_032_000_695_724),
        (MatrixNormOrder::NegTwo, 0.772_869_635_673_484_5),
        (MatrixNormOrder::Inf, 15.0),
        (MatrixNormOrder::NegInf, 6.0),
    ];
    for (ord, expected) in cases {
        let got = matrix_norm_ord(&a, ord).unwrap();
        assert_eq!(got.shape(), &[], "ord={ord:?} shape");
        assert_close_f64(
            got.data().unwrap(),
            &[expected],
            1e-10,
            &format!("ord={ord:?}"),
        );
    }

    assert_close_f64(
        matrix_norm(&a).unwrap().data().unwrap(),
        &[9.539_392_014_169_456],
        1e-10,
        "default fro",
    );
}

#[test]
fn matrix_norm_ord_cpu_empty_edges_match_torch() {
    let zero_by_three = tensor_f64(&[], &[0, 3], false);
    assert_eq!(
        matrix_norm_ord(&zero_by_three, MatrixNormOrder::Fro)
            .unwrap()
            .data()
            .unwrap(),
        &[0.0]
    );
    assert_eq!(
        matrix_norm_ord(&zero_by_three, MatrixNormOrder::Nuclear)
            .unwrap()
            .data()
            .unwrap(),
        &[0.0]
    );
    assert_eq!(
        matrix_norm_ord(&zero_by_three, MatrixNormOrder::One)
            .unwrap()
            .data()
            .unwrap(),
        &[0.0]
    );
    assert_eq!(
        matrix_norm_ord(&zero_by_three, MatrixNormOrder::NegOne)
            .unwrap()
            .data()
            .unwrap(),
        &[0.0]
    );
    assert_eq!(
        matrix_norm_ord(&zero_by_three, MatrixNormOrder::Two)
            .unwrap()
            .data()
            .unwrap(),
        &[0.0]
    );
    assert!(
        matrix_norm_ord(&zero_by_three, MatrixNormOrder::NegTwo)
            .unwrap_err()
            .to_string()
            .contains("non-zero size")
    );
    assert_eq!(
        matrix_norm_ord(&zero_by_three, MatrixNormOrder::Inf)
            .unwrap()
            .data()
            .unwrap(),
        &[0.0]
    );
    assert!(
        matrix_norm_ord(&zero_by_three, MatrixNormOrder::NegInf)
            .unwrap_err()
            .to_string()
            .contains("non-zero size")
    );

    let three_by_zero = tensor_f64(&[], &[3, 0], false);
    assert_eq!(
        matrix_norm_ord(&three_by_zero, MatrixNormOrder::One)
            .unwrap()
            .data()
            .unwrap(),
        &[0.0]
    );
    assert!(
        matrix_norm_ord(&three_by_zero, MatrixNormOrder::NegOne)
            .unwrap_err()
            .to_string()
            .contains("non-zero size")
    );
    assert_eq!(
        matrix_norm_ord(&three_by_zero, MatrixNormOrder::Inf)
            .unwrap()
            .data()
            .unwrap(),
        &[0.0]
    );
    assert_eq!(
        matrix_norm_ord(&three_by_zero, MatrixNormOrder::NegInf)
            .unwrap()
            .data()
            .unwrap(),
        &[0.0]
    );
}

#[test]
fn matrix_norm_ord_low_precision_accepts_only_torch_supported_orders() {
    let x16 = tensor_f16(&[1.0, 2.0, 3.0, 4.0], &[2, 2], false);
    assert_eq!(
        matrix_norm_ord(&x16, MatrixNormOrder::One)
            .unwrap()
            .data()
            .unwrap()[0]
            .to_f32(),
        6.0
    );
    assert_eq!(
        matrix_norm_ord(&x16, MatrixNormOrder::Inf)
            .unwrap()
            .data()
            .unwrap()[0]
            .to_f32(),
        7.0
    );
    assert!(
        matrix_norm_ord(&x16, MatrixNormOrder::Two)
            .unwrap_err()
            .to_string()
            .contains("Low precision dtypes not supported")
    );
    assert!(
        matrix_norm_ord(&x16, MatrixNormOrder::Nuclear)
            .unwrap_err()
            .to_string()
            .contains("Low precision dtypes not supported")
    );

    let xb = tensor_bf16(&[1.0, 2.0, 3.0, 4.0], &[2, 2], false);
    assert_eq!(
        matrix_norm_ord(&xb, MatrixNormOrder::One)
            .unwrap()
            .data()
            .unwrap()[0]
            .to_f32(),
        6.0
    );
    assert!(
        matrix_norm_ord(&xb, MatrixNormOrder::NegTwo)
            .unwrap_err()
            .to_string()
            .contains("Low precision dtypes not supported")
    );
}

#[test]
fn matrix_norm_ord_one_backward_uses_reduction_subgradient() {
    let a = tensor_f64(&[1.0, -2.0, 3.0, 4.0, -5.0, 6.0], &[2, 3], true);
    let n = matrix_norm_ord(&a, MatrixNormOrder::One).unwrap();
    assert_eq!(n.data().unwrap(), &[9.0]);
    n.backward().unwrap();
    let grad = a.grad().unwrap().unwrap();
    assert_close_f64(
        grad.data().unwrap(),
        &[0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        0.0,
        "ord=1 backward",
    );
}

#[test]
fn vector_norm_cpu_supports_half_and_bfloat_orders_like_torch() {
    let x16 = tensor_f16(&[1.0, -2.0, 0.0, 4.0], &[4], false);
    assert_eq!(
        vector_norm(&x16, 0.0).unwrap().data().unwrap()[0].to_f32(),
        3.0
    );
    assert_eq!(
        vector_norm(&x16, 3.5).unwrap().data().unwrap()[0].to_f32(),
        4.105_468_8
    );
    assert_eq!(
        vector_norm(&x16, f64::INFINITY).unwrap().data().unwrap()[0].to_f32(),
        4.0
    );

    let xb = tensor_bf16(&[1.0, -2.0, 0.0, 4.0], &[4], false);
    assert_eq!(
        vector_norm(&xb, 0.0).unwrap().data().unwrap()[0].to_f32(),
        3.0
    );
    assert_eq!(
        vector_norm(&xb, 3.5).unwrap().data().unwrap()[0].to_f32(),
        4.09375
    );
}

#[test]
fn vector_norm_empty_nan_order_returns_zero_like_torch() {
    // Live torch 2.11 oracle:
    // torch.linalg.vector_norm(torch.tensor([], dtype=torch.float64), ord=float("nan")) -> 0.
    let x = tensor_f64(&[], &[0], false);
    let n = vector_norm(&x, f64::NAN).unwrap();
    assert_eq!(n.shape(), &[]);
    assert_eq!(n.data().unwrap(), &[0.0]);
}

#[test]
fn vector_norm_cpu_accepts_noncontiguous_views() {
    // PyTorch accepts strided inputs. The transposed view's logical sequence is
    // [1, 4, 2, 5, 3, 6], so L1 is 21 and inf is 6.
    let base = tensor_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
    let view = base.transpose(0, 1).unwrap();
    assert!(!view.is_contiguous(), "test must exercise the view path");

    assert_eq!(vector_norm(&view, 1.0).unwrap().data().unwrap(), &[21.0]);
    assert_eq!(
        vector_norm(&view, f64::INFINITY).unwrap().data().unwrap(),
        &[6.0]
    );
}

#[test]
fn vector_norm_cpu_inf_nan_poison_is_not_erased_by_later_finite_values() {
    // torch.linalg.vector_norm([nan, 3, 4], ord=±inf) returns nan. This guards
    // against ordinary max/min folds that overwrite a prior NaN with a later
    // finite value.
    let x = tensor_f64(&[f64::NAN, 3.0, 4.0], &[3], false);
    assert!(vector_norm(&x, f64::INFINITY).unwrap().data().unwrap()[0].is_nan());
    assert!(vector_norm(&x, f64::NEG_INFINITY).unwrap().data().unwrap()[0].is_nan());
}

#[test]
fn vector_norm_ord_zero_tracks_but_leaves_gradient_undefined() {
    let x = tensor_f32(&[1.0, 0.0, -2.0], &[3], true);
    let n = vector_norm(&x, 0.0).unwrap();
    assert!(n.requires_grad(), "torch attaches a grad_fn for ord=0");
    assert_eq!(n.data().unwrap()[0], 2.0);

    n.backward().unwrap();

    assert!(
        x.grad().unwrap().is_none(),
        "torch norm_backward returns undefined gradient for ord=0"
    );
}

#[test]
fn matrix_rank_returns_int64_scalar_not_float_encoded_rank() {
    let full = tensor_f64(&[1.0, 2.0, 3.0, 5.0], &[2, 2], false);
    let rank = matrix_rank(&full, None).unwrap();
    assert_eq!(rank.shape(), &[]);
    assert_eq!(rank.device(), Device::Cpu);
    assert_eq!(rank.data().unwrap(), &[2]);

    let singular = tensor_f64(&[1.0, 2.0, 2.0, 4.0], &[2, 2], false);
    assert_eq!(matrix_rank(&singular, None).unwrap().data().unwrap(), &[1]);

    let tol = tensor_f64(&[5.0, 0.0, 0.0, 1.0], &[2, 2], false);
    assert_eq!(matrix_rank(&tol, Some(2.0)).unwrap().data().unwrap(), &[1]);
}

#[test]
fn matrix_rank_and_cond_reject_low_precision_like_torch() {
    let x16 = tensor_f16(&[1.0, 0.0, 0.0, 1.0], &[2, 2], false);
    assert!(
        matrix_rank(&x16, None)
            .unwrap_err()
            .to_string()
            .contains("requires f32 or f64")
    );
    assert!(
        cond(&x16, 2.0)
            .unwrap_err()
            .to_string()
            .contains("requires f32 or f64")
    );

    let xb = tensor_bf16(&[1.0, 0.0, 0.0, 1.0], &[2, 2], false);
    assert!(
        matrix_rank(&xb, None)
            .unwrap_err()
            .to_string()
            .contains("requires f32 or f64")
    );
    assert!(
        cond(&xb, 2.0)
            .unwrap_err()
            .to_string()
            .contains("requires f32 or f64")
    );
}

#[test]
fn cond_cpu_tracked_inverse_norm_order_has_real_backward() {
    let a = tensor_f32(&[2.0, 0.0, 0.0, 0.5], &[2, 2], true);
    let c = cond(&a, 1.0).unwrap();
    assert_eq!(c.device(), Device::Cpu);
    assert_close_f32(c.data().unwrap(), &[4.0], 1e-6, "cond p=1 forward");

    c.backward().unwrap();

    let grad = a.grad().unwrap().unwrap();
    assert_eq!(grad.device(), Device::Cpu);
    assert_close_f32(
        grad.data().unwrap(),
        &[2.0, 0.0, 0.0, -8.0],
        1e-5,
        "cond p=1 backward",
    );
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use std::sync::Once;

    static INIT: Once = Once::new();

    fn ensure_cuda() {
        INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend().expect("CUDA backend must initialize");
        });
    }

    fn cuda_f32(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
        ensure_cuda();
        tensor_f32(data, shape, false)
            .to(Device::Cuda(0))
            .expect("upload")
            .requires_grad_(requires_grad)
    }

    fn cuda_f16(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f16> {
        ensure_cuda();
        tensor_f16(data, shape, false)
            .to(Device::Cuda(0))
            .expect("upload")
            .requires_grad_(requires_grad)
    }

    fn cuda_bf16(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<bf16> {
        ensure_cuda();
        tensor_bf16(data, shape, false)
            .to(Device::Cuda(0))
            .expect("upload")
            .requires_grad_(requires_grad)
    }

    #[test]
    fn vector_norm_cuda_half_bfloat_forward_stays_resident() {
        let x16 = cuda_f16(&[1.0, -2.0, 0.0, 4.0], &[4], false);
        let n16 = vector_norm(&x16, 3.5).unwrap();
        assert_eq!(n16.device(), Device::Cuda(0));
        assert_eq!(n16.data_vec().unwrap()[0].to_f32(), 4.105_468_8);

        let xb = cuda_bf16(&[1.0, -2.0, 0.0, 4.0], &[4], false);
        let nb = vector_norm(&xb, 3.5).unwrap();
        assert_eq!(nb.device(), Device::Cuda(0));
        assert_eq!(nb.data_vec().unwrap()[0].to_f32(), 4.09375);
    }

    #[test]
    fn vector_norm_cuda_backward_handles_zero_ugly_cases() {
        let x = cuda_f32(&[0.0, 1.0, 4.0], &[3], true);
        vector_norm(&x, 0.5).unwrap().backward().unwrap();
        let grad = x.grad().unwrap().unwrap();
        assert_eq!(grad.device(), Device::Cuda(0));
        assert_close_f32(
            &grad.data_vec().unwrap(),
            &[0.0, 3.0, 1.5],
            2e-5,
            "p<1 grad",
        );

        let z = cuda_f32(&[0.0, 0.0, 0.0], &[3], true);
        vector_norm(&z, 3.0).unwrap().backward().unwrap();
        let zgrad = z.grad().unwrap().unwrap();
        assert_eq!(zgrad.device(), Device::Cuda(0));
        assert_close_f32(
            &zgrad.data_vec().unwrap(),
            &[0.0, 0.0, 0.0],
            0.0,
            "zero p>2 grad",
        );
    }

    #[test]
    fn vector_norm_cuda_nan_backward_matches_torch_sign_rules() {
        let x1 = cuda_f32(&[f32::NAN, 3.0, -3.0, 0.0], &[4], true);
        let n1 = vector_norm(&x1, 1.0).unwrap();
        assert_eq!(n1.device(), Device::Cuda(0));
        assert!(n1.data_vec().unwrap()[0].is_nan());
        n1.backward().unwrap();
        let g1 = x1.grad().unwrap().unwrap();
        assert_eq!(g1.device(), Device::Cuda(0));
        assert_close_f32(
            &g1.data_vec().unwrap(),
            &[0.0, 1.0, -1.0, 0.0],
            0.0,
            "cuda p=1 NaN sign",
        );

        let xi = cuda_f32(&[f32::NAN, 3.0, -3.0, 0.0], &[4], true);
        let ni = vector_norm(&xi, f64::INFINITY).unwrap();
        assert_eq!(ni.device(), Device::Cuda(0));
        assert!(ni.data_vec().unwrap()[0].is_nan());
        ni.backward().unwrap();
        let gi = xi.grad().unwrap().unwrap();
        assert_eq!(gi.device(), Device::Cuda(0));
        assert_close_f32(
            &gi.data_vec().unwrap(),
            &[0.0, 0.0, 0.0, 0.0],
            0.0,
            "cuda p=inf NaN sign",
        );
    }

    #[test]
    fn vector_norm_cuda_empty_nan_order_returns_zero_resident() {
        let x = cuda_f32(&[], &[0], false);
        let n = vector_norm(&x, f64::NAN).unwrap();
        assert_eq!(n.device(), Device::Cuda(0));
        assert_eq!(n.data_vec().unwrap(), &[0.0]);
    }

    #[test]
    fn matrix_norm_ord_cuda_matches_torch_and_stays_resident() {
        // Live torch 2.11 f32 oracle for [[1,2],[3,4]].
        let a = cuda_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], false);
        let cases = [
            (MatrixNormOrder::Fro, 5.477_226),
            (MatrixNormOrder::Nuclear, 5.830_951_7),
            (MatrixNormOrder::One, 6.0),
            (MatrixNormOrder::NegOne, 4.0),
            (MatrixNormOrder::Two, 5.464_986),
            (MatrixNormOrder::NegTwo, 0.365_966),
            (MatrixNormOrder::Inf, 7.0),
            (MatrixNormOrder::NegInf, 3.0),
        ];
        for (ord, expected) in cases {
            let got = matrix_norm_ord(&a, ord).unwrap();
            assert_eq!(got.device(), Device::Cuda(0), "ord={ord:?} device");
            assert_close_f32(
                &got.data_vec().unwrap(),
                &[expected],
                2e-5,
                &format!("cuda ord={ord:?}"),
            );
        }
    }

    #[test]
    fn matrix_rank_cuda_returns_int64_on_cuda() {
        let a = cuda_f32(&[1.0, 2.0, 2.0, 4.0], &[2, 2], false);
        let rank = matrix_rank(&a, None).unwrap();
        assert_eq!(rank.device(), Device::Cuda(0));
        assert_eq!(rank.shape(), &[]);
        assert_eq!(rank.to(Device::Cpu).unwrap().data().unwrap(), &[1]);
    }

    #[test]
    fn cond_cuda_supports_svd_and_inverse_norm_orders_resident() {
        let a = cuda_f32(&[2.0, 0.0, 0.0, 0.5], &[2, 2], false);

        let c2 = cond(&a, 2.0).unwrap();
        assert_eq!(c2.device(), Device::Cuda(0));
        assert_close_f32(&c2.data_vec().unwrap(), &[4.0], 1e-5, "cond p=2");

        let cm2 = cond(&a, -2.0).unwrap();
        assert_eq!(cm2.device(), Device::Cuda(0));
        assert_close_f32(&cm2.data_vec().unwrap(), &[0.25], 1e-5, "cond p=-2");

        let c1 = cond(&a, 1.0).unwrap();
        assert_eq!(c1.device(), Device::Cuda(0));
        assert_close_f32(&c1.data_vec().unwrap(), &[4.0], 1e-5, "cond p=1");

        let cinf = cond(&a, f64::INFINITY).unwrap();
        assert_eq!(cinf.device(), Device::Cuda(0));
        assert_close_f32(&cinf.data_vec().unwrap(), &[4.0], 1e-5, "cond p=inf");
    }

    #[test]
    fn inv_cuda_forward_and_backward_stay_resident() {
        let a = cuda_f32(&[2.0, 0.0, 0.0, 0.5], &[2, 2], true);

        let inv = ferrotorch_core::linalg::inv(&a).unwrap();
        assert_eq!(inv.device(), Device::Cuda(0));
        assert_close_f32(
            &inv.data_vec().unwrap(),
            &[0.5, 0.0, 0.0, 2.0],
            1e-6,
            "inv forward",
        );

        inv.sum_all().unwrap().backward().unwrap();
        let grad = a.grad().unwrap().unwrap();
        assert_eq!(grad.device(), Device::Cuda(0));
        assert_close_f32(
            &grad.data_vec().unwrap(),
            &[-0.25, -1.0, -1.0, -4.0],
            2e-5,
            "inv backward",
        );
    }
}
