//! CORE-046 (#1740) regression: `var_dim` / `std_dim` must differentiate.
//!
//! At HEAD before the fix both functions constructed `requires_grad=false`
//! outputs ("forward-only" comments) and silently severed every training
//! graph that used a dim-keyed variance/std — while torch attaches
//! `VarBackward0` / `StdBackward0` (`derivatives.yaml` `var.correction` /
//! `std.correction` → `var_backward` / `std_backward` in
//! `torch/csrc/autograd/FunctionsManual.cpp`).
//!
//! Every numerical expectation below is a live torch 2.11.0+cu130 oracle;
//! the generating snippet is quoted in a comment above each case
//! (R-ORACLE-1b). All assertions check gradient VALUES on the original
//! leaf (gradient flow per R-ORACLE-3), never `requires_grad` flags alone.
//!
//! Device contract: the forwards are CPU-only by explicit
//! `NotImplementedOnCuda` error, so the backward inherits the CPU contract
//! (no silent-demotion lane to assert).

#![allow(
    clippy::excessive_precision,
    reason = "float literals quote live-torch oracle printouts verbatim (R-ORACLE-1b \
              traceability); clippy's shortest-representation rewrite would obscure \
              the quoted oracle digits"
)]

use ferrotorch_core::grad_fns::reduction::{std_dim, var_dim};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

fn leaf_f64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}

fn leaf_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}

fn plain_f64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

fn plain_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

/// f64 two-pass reduction over <=6 elements: torch's oracle digits are exact
/// to the printed precision; 1e-12 absorbs only the (x-mean)/denom rounding
/// difference between torch's Welford kernel and the two-pass form (both
/// f64; divergence is <=4 ulps on these tiny slices).
const TOL_F64: f64 = 1e-12;
/// f32 oracle values here are exact small rationals (probed values print as
/// exact halves); 1e-6 ~ 8 ulps at magnitude 5 covers the two-pass vs
/// Welford difference at f32.
const TOL_F32: f32 = 1e-6;

fn assert_close_f64(actual: &[f64], expected: &[f64], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        if e.is_nan() {
            assert!(a.is_nan(), "{label}[{i}]: expected NaN, got {a}");
        } else {
            assert!(
                (a - e).abs() <= TOL_F64,
                "{label}[{i}]: got {a}, torch oracle {e}"
            );
        }
    }
}

fn assert_close_f32(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        if e.is_nan() {
            assert!(a.is_nan(), "{label}[{i}]: expected NaN, got {a}");
        } else {
            assert!(
                (a - e).abs() <= TOL_F32,
                "{label}[{i}]: got {a}, torch oracle {e}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// var_dim
// ---------------------------------------------------------------------------

/// Live oracle:
/// ```python
/// x = torch.tensor([[1.,2.,4.],[3.,7.,8.]], dtype=torch.float64, requires_grad=True)
/// y = torch.var(x, dim=1, correction=1, keepdim=False)   # [2.33333..., 7.0]
/// y.backward(torch.tensor([1.,2.], dtype=torch.float64))
/// x.grad  # [[-1.33333333333333348, -0.33333333333333348, 1.66666666666666652],
///         #  [-6.0, 2.0, 4.0]]
/// ```
#[test]
fn var_dim_f64_corr1_grad_values() {
    let x = leaf_f64(&[1.0, 2.0, 4.0, 3.0, 7.0, 8.0], &[2, 3]);
    let y = var_dim(&x, 1, 1.0, false).unwrap();
    assert_close_f64(
        y.data().unwrap(),
        &[2.333_333_333_333_333, 7.0],
        "var_dim fwd",
    );
    let go = plain_f64(&[1.0, 2.0], &[2]);
    y.backward_with_gradient(&go).unwrap();
    let g = x
        .grad()
        .unwrap()
        .expect("var_dim: no grad reached the leaf");
    assert_eq!(g.shape(), &[2, 3], "var_dim grad shape");
    assert_close_f64(
        g.data().unwrap(),
        &[
            -1.333_333_333_333_333_5,
            -0.333_333_333_333_333_5,
            1.666_666_666_666_666_5,
            -6.0,
            2.0,
            4.0,
        ],
        "var_dim grad",
    );
}

/// 3-D middle-dim reduction (exercises outer*inner indexing). Live oracle:
/// ```python
/// x = (torch.arange(12, dtype=torch.float64).reshape(2,3,2)**2 * 0.25 + 1.0
///      ).detach().requires_grad_(True)
/// y = torch.var(x, dim=1, correction=1)   # [[4.3333..., 9.3333...],
///                                         #  [64.3333..., 81.3333...]]
/// y.backward(torch.tensor([[1.,2.],[3.,4.]], dtype=torch.float64))
/// x.grad  # [[[-1.66666666666666652, -5.33333333333333304],
///         #   [-0.66666666666666652, -1.33333333333333304],
///         #   [ 2.33333333333333348,  6.66666666666666696]],
///         #  [[-23.00000000000000355, -34.66666666666667140],
///         #   [ -2.00000000000000355,  -2.66666666666667140],
///         #   [ 24.99999999999999645,  37.33333333333332860]]]
/// ```
#[test]
fn var_dim_f64_middle_dim_3d_grad_values() {
    #[allow(
        clippy::cast_precision_loss,
        reason = "i in 0..12 is exactly representable in f64"
    )]
    let data: Vec<f64> = (0..12).map(|i| (i * i) as f64 * 0.25 + 1.0).collect();
    let x = leaf_f64(&data, &[2, 3, 2]);
    let y = var_dim(&x, 1, 1.0, false).unwrap();
    assert_close_f64(
        y.data().unwrap(),
        &[
            4.333_333_333_333_333,
            9.333_333_333_333_332,
            64.333_333_333_333_33,
            81.333_333_333_333_33,
        ],
        "var_dim 3d fwd",
    );
    let go = plain_f64(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    y.backward_with_gradient(&go).unwrap();
    let g = x
        .grad()
        .unwrap()
        .expect("var_dim 3d: no grad reached the leaf");
    assert_eq!(g.shape(), &[2, 3, 2], "var_dim 3d grad shape");
    assert_close_f64(
        g.data().unwrap(),
        &[
            -1.666_666_666_666_666_5,
            -5.333_333_333_333_333,
            -0.666_666_666_666_666_5,
            -1.333_333_333_333_333,
            2.333_333_333_333_333_5,
            6.666_666_666_666_667,
            -23.000_000_000_000_004,
            -34.666_666_666_666_67,
            -2.000_000_000_000_003_6,
            -2.666_666_666_666_671_4,
            24.999_999_999_999_996,
            37.333_333_333_333_33,
        ],
        "var_dim 3d grad",
    );
}

/// f32 + dim=0 + keepdim=false. Live oracle:
/// ```python
/// x = torch.tensor([[1.,2.,4.],[3.,7.,8.]], dtype=torch.float32, requires_grad=True)
/// y = torch.var(x, dim=0, correction=1)   # [2., 12.5, 8.]
/// y.backward(torch.ones(3))
/// x.grad  # [[-2., -5., -4.], [2., 5., 4.]]
/// ```
#[test]
fn var_dim_f32_dim0_grad_values() {
    let x = leaf_f32(&[1.0, 2.0, 4.0, 3.0, 7.0, 8.0], &[2, 3]);
    let y = var_dim(&x, 0, 1.0, false).unwrap();
    assert_close_f32(y.data().unwrap(), &[2.0, 12.5, 8.0], "var_dim f32 fwd");
    let go = plain_f32(&[1.0, 1.0, 1.0], &[3]);
    y.backward_with_gradient(&go).unwrap();
    let g = x
        .grad()
        .unwrap()
        .expect("var_dim f32: no grad reached the leaf");
    assert_close_f32(
        g.data().unwrap(),
        &[-2.0, -5.0, -4.0, 2.0, 5.0, 4.0],
        "var_dim f32 grad",
    );
}

/// Singleton reduced dim. Live oracles:
/// ```python
/// x = torch.tensor([[3.],[4.]], dtype=torch.float64, requires_grad=True)
/// torch.var(x, dim=1, correction=1)  # [nan, nan]; .backward -> x.grad [[nan],[nan]]
/// torch.var(x, dim=1, correction=0)  # [0., 0.];  .backward -> x.grad [[0.],[0.]]
/// ```
#[test]
fn var_dim_singleton_corr1_nan_and_corr0_zero() {
    // correction=1 over a length-1 slice: denom 0 -> fwd NaN, grad NaN.
    let x = leaf_f64(&[3.0, 4.0], &[2, 1]);
    let y = var_dim(&x, 1, 1.0, false).unwrap();
    assert_close_f64(y.data().unwrap(), &[f64::NAN, f64::NAN], "singleton fwd");
    let go = plain_f64(&[1.0, 1.0], &[2]);
    y.backward_with_gradient(&go).unwrap();
    let g = x
        .grad()
        .unwrap()
        .expect("var_dim singleton: no grad reached the leaf");
    assert_close_f64(
        g.data().unwrap(),
        &[f64::NAN, f64::NAN],
        "singleton corr=1 grad",
    );

    // correction=0: fwd 0, grad 0.
    let x0 = leaf_f64(&[3.0, 4.0], &[2, 1]);
    let y0 = var_dim(&x0, 1, 0.0, false).unwrap();
    assert_close_f64(y0.data().unwrap(), &[0.0, 0.0], "singleton corr=0 fwd");
    let go0 = plain_f64(&[1.0, 1.0], &[2]);
    y0.backward_with_gradient(&go0).unwrap();
    let g0 = x0
        .grad()
        .unwrap()
        .expect("var_dim singleton corr=0: no grad reached the leaf");
    assert_close_f64(g0.data().unwrap(), &[0.0, 0.0], "singleton corr=0 grad");
}

/// Empty reduced slice. Live oracle:
/// ```python
/// x = torch.zeros((2,0), dtype=torch.float64, requires_grad=True)
/// y = torch.var(x, dim=1, correction=1)   # [nan, nan]
/// y.backward(torch.ones_like(y))
/// x.grad   # tensor([], size=(2, 0))
/// ```
#[test]
fn var_dim_empty_slice_nan_forward_empty_grad() {
    let x = leaf_f64(&[], &[2, 0]);
    let y = var_dim(&x, 1, 1.0, false).unwrap();
    assert_close_f64(y.data().unwrap(), &[f64::NAN, f64::NAN], "empty fwd");
    let go = plain_f64(&[1.0, 1.0], &[2]);
    y.backward_with_gradient(&go).unwrap();
    let g = x
        .grad()
        .unwrap()
        .expect("var_dim empty: no grad reached the leaf");
    assert_eq!(g.shape(), &[2, 0], "empty grad shape");
    assert_eq!(g.data().unwrap().len(), 0, "empty grad numel");
}

// ---------------------------------------------------------------------------
// std_dim
// ---------------------------------------------------------------------------

/// keepdim=true + correction=0. Live oracle:
/// ```python
/// x = torch.tensor([[1.,2.,4.],[3.,7.,8.]], dtype=torch.float64, requires_grad=True)
/// y = torch.std(x, dim=1, correction=0, keepdim=True)
/// # [[1.24721912892464704], [2.16024689946928694]]
/// y.backward(torch.tensor([[1.],[2.]], dtype=torch.float64))
/// x.grad  # [[-0.35634832254989923, -0.08908708063747484, 0.44543540318737396],
///         #  [-0.92582009977255131,  0.30860669992418377, 0.61721339984836754]]
/// ```
#[test]
fn std_dim_f64_keepdim_corr0_grad_values() {
    let x = leaf_f64(&[1.0, 2.0, 4.0, 3.0, 7.0, 8.0], &[2, 3]);
    let y = std_dim(&x, 1, 0.0, true).unwrap();
    assert_eq!(y.shape(), &[2, 1], "std_dim keepdim shape");
    assert_close_f64(
        y.data().unwrap(),
        &[1.247_219_128_924_647, 2.160_246_899_469_287],
        "std_dim fwd",
    );
    let go = plain_f64(&[1.0, 2.0], &[2, 1]);
    y.backward_with_gradient(&go).unwrap();
    let g = x
        .grad()
        .unwrap()
        .expect("std_dim: no grad reached the leaf");
    assert_eq!(g.shape(), &[2, 3], "std_dim grad shape");
    assert_close_f64(
        g.data().unwrap(),
        &[
            -0.356_348_322_549_899_23,
            -0.089_087_080_637_474_84,
            0.445_435_403_187_373_96,
            -0.925_820_099_772_551_31,
            0.308_606_699_924_183_77,
            0.617_213_399_848_367_54,
        ],
        "std_dim grad",
    );
}

/// Zero-variance slice: std == 0 -> zero gradient for that slice
/// (`derivatives.yaml:1676` `masked_fill_(result == 0, 0)`). Live oracle:
/// ```python
/// x = torch.tensor([[5.,5.,5.],[1.,2.,3.]], dtype=torch.float64, requires_grad=True)
/// y = torch.std(x, dim=1, correction=1)   # [0., 1.]
/// y.backward(torch.tensor([1.,1.], dtype=torch.float64))
/// x.grad  # [[0., 0., 0.], [-0.5, 0., 0.5]]
/// ```
#[test]
fn std_dim_zero_std_slice_zero_grad() {
    let x = leaf_f64(&[5.0, 5.0, 5.0, 1.0, 2.0, 3.0], &[2, 3]);
    let y = std_dim(&x, 1, 1.0, false).unwrap();
    assert_close_f64(y.data().unwrap(), &[0.0, 1.0], "std zero-slice fwd");
    let go = plain_f64(&[1.0, 1.0], &[2]);
    y.backward_with_gradient(&go).unwrap();
    let g = x
        .grad()
        .unwrap()
        .expect("std_dim zero-slice: no grad reached the leaf");
    assert_close_f64(
        g.data().unwrap(),
        &[0.0, 0.0, 0.0, -0.5, 0.0, 0.5],
        "std zero-slice grad",
    );
}

/// Negative dim. Live oracle:
/// ```python
/// x = torch.tensor([[1.,2.],[3.,5.]], dtype=torch.float64, requires_grad=True)
/// y = torch.std(x, dim=-1, correction=1)
/// # [0.70710678118654757, 1.41421356237309515]
/// y.backward(torch.tensor([1.,3.], dtype=torch.float64))
/// x.grad  # [[-0.70710678118654746, 0.70710678118654746],
///         #  [-2.12132034355964239, 2.12132034355964239]]
/// ```
#[test]
#[allow(
    clippy::approx_constant,
    reason = "the oracle values happen to be sqrt(2)/2 and sqrt(2); they are \
              quoted live-torch printouts, not hand-typed constants"
)]
fn std_dim_negative_dim_grad_values() {
    let x = leaf_f64(&[1.0, 2.0, 3.0, 5.0], &[2, 2]);
    let y = std_dim(&x, -1, 1.0, false).unwrap();
    assert_close_f64(
        y.data().unwrap(),
        &[0.707_106_781_186_547_57, 1.414_213_562_373_095_15],
        "std dim=-1 fwd",
    );
    let go = plain_f64(&[1.0, 3.0], &[2]);
    y.backward_with_gradient(&go).unwrap();
    let g = x
        .grad()
        .unwrap()
        .expect("std_dim dim=-1: no grad reached the leaf");
    assert_close_f64(
        g.data().unwrap(),
        &[
            -0.707_106_781_186_547_46,
            0.707_106_781_186_547_46,
            -2.121_320_343_559_642_39,
            2.121_320_343_559_642_39,
        ],
        "std dim=-1 grad",
    );
}

/// f32 + dim=0 + keepdim=true + correction=0 + non-unit grad_output.
/// Live oracle:
/// ```python
/// x = torch.tensor([[1.,2.,4.],[3.,7.,8.]], dtype=torch.float32, requires_grad=True)
/// y = torch.std(x, dim=0, correction=0, keepdim=True)   # [[1., 2.5, 2.]]
/// y.backward(torch.tensor([[2.,1.,1.]]))
/// x.grad  # [[-1., -0.5, -0.5], [1., 0.5, 0.5]]
/// ```
#[test]
fn std_dim_f32_dim0_keepdim_grad_values() {
    let x = leaf_f32(&[1.0, 2.0, 4.0, 3.0, 7.0, 8.0], &[2, 3]);
    let y = std_dim(&x, 0, 0.0, true).unwrap();
    assert_eq!(y.shape(), &[1, 3], "std f32 keepdim shape");
    assert_close_f32(y.data().unwrap(), &[1.0, 2.5, 2.0], "std f32 fwd");
    let go = plain_f32(&[2.0, 1.0, 1.0], &[1, 3]);
    y.backward_with_gradient(&go).unwrap();
    let g = x
        .grad()
        .unwrap()
        .expect("std_dim f32: no grad reached the leaf");
    assert_close_f32(
        g.data().unwrap(),
        &[-1.0, -0.5, -0.5, 1.0, 0.5, 0.5],
        "std f32 grad",
    );
}

/// Singleton slice, correction=1: NaN forward AND NaN gradient (torch
/// propagates the 0/0; live oracle: `torch.std([[3.],[4.]], dim=1,
/// correction=1)` -> `[nan, nan]`, grad `[[nan],[nan]]`).
#[test]
fn std_dim_singleton_corr1_nan_grad() {
    let x = leaf_f64(&[3.0, 4.0], &[2, 1]);
    let y = std_dim(&x, 1, 1.0, false).unwrap();
    assert_close_f64(
        y.data().unwrap(),
        &[f64::NAN, f64::NAN],
        "std singleton fwd",
    );
    let go = plain_f64(&[1.0, 1.0], &[2]);
    y.backward_with_gradient(&go).unwrap();
    let g = x
        .grad()
        .unwrap()
        .expect("std_dim singleton: no grad reached the leaf");
    assert_close_f64(
        g.data().unwrap(),
        &[f64::NAN, f64::NAN],
        "std singleton grad",
    );
}

/// No-grad contract: a `requires_grad=false` input stays detached (torch:
/// no grad_fn on the output of a non-tracking input).
#[test]
fn var_std_dim_no_grad_input_stays_detached() {
    let x = plain_f64(&[1.0, 2.0, 4.0, 3.0, 7.0, 8.0], &[2, 3]);
    let y = var_dim(&x, 1, 1.0, false).unwrap();
    assert!(!y.requires_grad(), "var_dim: non-tracking input tracked");
    assert!(y.grad_fn().is_none(), "var_dim: grad_fn on non-tracking");
    let z = std_dim(&x, 1, 1.0, false).unwrap();
    assert!(!z.requires_grad(), "std_dim: non-tracking input tracked");
    assert!(z.grad_fn().is_none(), "std_dim: grad_fn on non-tracking");
}
