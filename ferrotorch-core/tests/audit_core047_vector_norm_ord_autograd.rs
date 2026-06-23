//! CORE-047 (#1741) regression: `vector_norm` must differentiate EVERY
//! accepted `ord`, not only `ord == 2`.
//!
//! At HEAD before the fix `vector_norm_differentiable` computed every ord
//! under `no_grad` but attached `NormBackward` only for `ord == 2.0` — any
//! other order silently returned a detached output from a tracking input,
//! while torch attaches `LinalgVectorNormBackward0` for all of them
//! (`norm_backward` in `torch/csrc/autograd/FunctionsManual.cpp`).
//!
//! Every numerical expectation below is a live torch 2.11.0+cu130 oracle;
//! the generating snippet is quoted above each case (R-ORACLE-1b). All
//! assertions check gradient VALUES reaching the original leaf
//! (R-ORACLE-3). Forward is CPU-only (`require_cpu`), so there is no CUDA
//! lane for these backwards.

#![allow(
    clippy::excessive_precision,
    reason = "float literals quote live-torch oracle printouts verbatim (R-ORACLE-1b \
              traceability); clippy's shortest-representation rewrite would obscure \
              the quoted oracle digits"
)]

use ferrotorch_core::linalg::vector_norm;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

fn leaf_f64(data: &[f64]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], true).unwrap()
}

fn leaf_f32(data: &[f32]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], true).unwrap()
}

/// Forward + backward are closed-form over <=5 elements in f64; torch's
/// printed oracle digits are exact to 17 significant digits, so 1e-12
/// absorbs only ulp-level libm differences (pow/sqrt) between platforms.
const TOL_F64: f64 = 1e-12;
/// f32 lane: one pow + one divide per element at magnitude <=1; 1e-6 is
/// ~8 ulps at 0.5 and covers the f32 libm pow difference.
const TOL_F32: f32 = 1e-6;

fn check_grad_f64(x: &Tensor<f64>, expected: &[f64], label: &str) {
    let g = x
        .grad()
        .unwrap()
        .unwrap_or_else(|| panic!("{label}: no grad reached the leaf"));
    let got = g.data().unwrap();
    assert_eq!(got.len(), expected.len(), "{label}: grad length");
    for (i, (a, e)) in got.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= TOL_F64,
            "{label}[{i}]: got {a}, torch oracle {e}"
        );
    }
}

fn check_grad_f64_nan(x: &Tensor<f64>, label: &str) {
    let g = x
        .grad()
        .unwrap()
        .unwrap_or_else(|| panic!("{label}: no grad reached the leaf"));
    let got = g.data().unwrap();
    assert!(
        got.iter().all(|v| v.is_nan()),
        "{label}: torch oracle is all NaN, got {got:?}"
    );
}

fn fwd_close(n: &Tensor<f64>, expected: f64, label: &str) {
    let v = n.data().unwrap()[0];
    assert!(
        (v - expected).abs() <= TOL_F64,
        "{label} fwd: got {v}, torch oracle {expected}"
    );
}

/// ord=1 — `dx = sgn(x) * g`, `sgn(0) = 0`. Live oracle:
/// ```python
/// x = torch.tensor([1.,-2.,3.,-3.,0.], dtype=torch.float64, requires_grad=True)
/// n = torch.linalg.vector_norm(x, 1.0)   # 9.0
/// n.backward()
/// x.grad   # [1., -1., 1., -1., 0.]
/// ```
#[test]
fn ord_1_sign_grad() {
    let x = leaf_f64(&[1.0, -2.0, 3.0, -3.0, 0.0]);
    let n = vector_norm(&x, 1.0).unwrap();
    fwd_close(&n, 9.0, "ord=1");
    n.backward().unwrap();
    check_grad_f64(&x, &[1.0, -1.0, 1.0, -1.0, 0.0], "ord=1 grad");
}

/// Real-valued torch.sign maps NaN to zero, so p=1 does not propagate a NaN
/// through the sign VJP. Live oracle:
/// ```python
/// x = torch.tensor([nan, 3., -3., 0.], dtype=torch.float64, requires_grad=True)
/// torch.linalg.vector_norm(x, 1.0).backward()
/// x.grad   # [0., 1., -1., 0.]
/// ```
#[test]
fn ord_1_nan_sign_matches_torch() {
    let x = leaf_f64(&[f64::NAN, 3.0, -3.0, 0.0]);
    let n = vector_norm(&x, 1.0).unwrap();
    assert!(n.data().unwrap()[0].is_nan(), "ord=1 NaN forward");
    n.backward().unwrap();
    check_grad_f64(&x, &[0.0, 1.0, -1.0, 0.0], "ord=1 NaN sign grad");
}

/// ord=3 (general p>2 branch) — `dx = x*|x|^(p-2) * g / norm^(p-1)`.
/// Live oracle:
/// ```python
/// x = torch.tensor([1.,-2.,3.,-3.,0.], dtype=torch.float64, requires_grad=True)
/// n = torch.linalg.vector_norm(x, 3.0)   # 3.9790572078963917
/// n.backward()
/// x.grad   # [0.06315963822057766, -0.25263855288231063, 0.56843674398519894,
///          #  -0.56843674398519894, 0.0]
/// ```
#[test]
fn ord_3_general_p_grad() {
    let x = leaf_f64(&[1.0, -2.0, 3.0, -3.0, 0.0]);
    let n = vector_norm(&x, 3.0).unwrap();
    fwd_close(&n, 3.979_057_207_896_391_7, "ord=3");
    n.backward().unwrap();
    check_grad_f64(
        &x,
        &[
            0.063_159_638_220_577_66,
            -0.252_638_552_882_310_63,
            0.568_436_743_985_198_94,
            -0.568_436_743_985_198_94,
            0.0,
        ],
        "ord=3 grad",
    );
}

/// ord=0.5 (p<1 branch) — `dx = sgn(x)*|x|^(p-1) * g * norm^(1-p)` with
/// the `x == 0 → 0` subgradient mask. Live oracle:
/// ```python
/// x = torch.tensor([1.,-2.,3.,-3.,0.], dtype=torch.float64, requires_grad=True)
/// n = torch.linalg.vector_norm(x, 0.5)   # 34.55458932615441
/// n.backward()
/// x.grad   # [5.87831517751084931, -4.15659652396972490, 3.39384685011735199,
///          #  -3.39384685011735199, 0.0]
/// ```
#[test]
fn ord_half_sub_one_p_grad() {
    let x = leaf_f64(&[1.0, -2.0, 3.0, -3.0, 0.0]);
    let n = vector_norm(&x, 0.5).unwrap();
    fwd_close(&n, 34.554_589_326_154_41, "ord=0.5");
    n.backward().unwrap();
    check_grad_f64(
        &x,
        &[
            5.878_315_177_510_849_31,
            -4.156_596_523_969_724_9,
            3.393_846_850_117_352,
            -3.393_846_850_117_352,
            0.0,
        ],
        "ord=0.5 grad",
    );
}

/// ord=0.5, zero element does NOT poison the rest. Live oracle:
/// ```python
/// x = torch.tensor([0.,1.,4.], dtype=torch.float64, requires_grad=True)
/// n = torch.linalg.vector_norm(x, 0.5)   # 9.0
/// n.backward()
/// x.grad   # [0., 3., 1.5]
/// ```
#[test]
fn ord_half_zero_element_masked() {
    let x = leaf_f64(&[0.0, 1.0, 4.0]);
    let n = vector_norm(&x, 0.5).unwrap();
    fwd_close(&n, 9.0, "ord=0.5 zero-elem");
    n.backward().unwrap();
    check_grad_f64(&x, &[0.0, 3.0, 1.5], "ord=0.5 zero-elem grad");
}

/// ord=1.5 (1<p<2 branch) — `dx = sgn(x)*|x|^(p-1) * g / norm^(p-1)`;
/// `|0|^(p-1) = 0` for p>1, no mask needed. Live oracle:
/// ```python
/// x = torch.tensor([1.,-2.,0.,4.], dtype=torch.float64, requires_grad=True)
/// n = torch.linalg.vector_norm(x, 1.5)   # 5.191402066808782
/// n.backward()
/// x.grad   # [0.43889200246648841, -0.62068702230519379, 0., 0.87778400493297681]
/// ```
#[test]
fn ord_1p5_between_one_and_two_grad() {
    let x = leaf_f64(&[1.0, -2.0, 0.0, 4.0]);
    let n = vector_norm(&x, 1.5).unwrap();
    fwd_close(&n, 5.191_402_066_808_782, "ord=1.5");
    n.backward().unwrap();
    check_grad_f64(
        &x,
        &[
            0.438_892_002_466_488_41,
            -0.620_687_022_305_193_79,
            0.0,
            0.877_784_004_932_976_81,
        ],
        "ord=1.5 grad",
    );
}

/// ord=3 at a zero element (p>=2 branch, `0*|0|^(p-2) = 0`). Live oracle:
/// ```python
/// x = torch.tensor([0.,1.,-2.], dtype=torch.float64, requires_grad=True)
/// n = torch.linalg.vector_norm(x, 3.0)   # 2.080083823051904
/// n.backward()
/// x.grad   # [0., 0.23112042478354489, -0.92448169913417955]
/// ```
#[test]
fn ord_3_zero_element_grad() {
    let x = leaf_f64(&[0.0, 1.0, -2.0]);
    let n = vector_norm(&x, 3.0).unwrap();
    fwd_close(&n, 2.080_083_823_051_904, "ord=3 zero-elem");
    n.backward().unwrap();
    check_grad_f64(
        &x,
        &[0.0, 0.231_120_424_783_544_89, -0.924_481_699_134_179_55],
        "ord=3 zero-elem grad",
    );
}

/// ord=inf — gradient routed to max-|x| elements, ties split EVENLY
/// (`grad / count_nonzero(|x| == norm)` per `norm_backward`'s isinf
/// branch). Live oracle:
/// ```python
/// x = torch.tensor([3.,-3.,1.], dtype=torch.float64, requires_grad=True)
/// n = torch.linalg.vector_norm(x, float('inf'))   # 3.0
/// n.backward()
/// x.grad   # [0.5, -0.5, 0.]
/// ```
#[test]
fn ord_inf_tie_split() {
    let x = leaf_f64(&[3.0, -3.0, 1.0]);
    let n = vector_norm(&x, f64::INFINITY).unwrap();
    fwd_close(&n, 3.0, "ord=inf");
    n.backward().unwrap();
    check_grad_f64(&x, &[0.5, -0.5, 0.0], "ord=inf tie grad");
}

/// Upstream's isinf branch includes `self_abs.isnan()` in the tie mask, but
/// then multiplies by `torch.sign`, which maps real NaNs to zero. Live oracle:
/// ```python
/// x = torch.tensor([nan, 3., -3., 0.], dtype=torch.float64, requires_grad=True)
/// torch.linalg.vector_norm(x, float('inf')).backward()
/// x.grad   # [0., 0., -0., 0.]
/// ```
#[test]
fn ord_inf_nan_tie_has_zero_sign_contribution() {
    let x = leaf_f64(&[f64::NAN, 3.0, -3.0, 0.0]);
    let n = vector_norm(&x, f64::INFINITY).unwrap();
    assert!(n.data().unwrap()[0].is_nan(), "ord=inf NaN forward");
    n.backward().unwrap();
    check_grad_f64(&x, &[0.0, 0.0, 0.0, 0.0], "ord=inf NaN grad");
}

/// ord=inf with a non-unit upstream gradient — the tie split scales.
/// Live oracle:
/// ```python
/// x = torch.tensor([3.,-3.,1.], dtype=torch.float64, requires_grad=True)
/// n = torch.linalg.vector_norm(x, float('inf'))
/// n.backward(torch.tensor(2.0, dtype=torch.float64))
/// x.grad   # [1., -1., 0.]
/// ```
#[test]
fn ord_inf_nonunit_grad_output() {
    let x = leaf_f64(&[3.0, -3.0, 1.0]);
    let n = vector_norm(&x, f64::INFINITY).unwrap();
    let go = Tensor::from_storage(TensorStorage::cpu(vec![2.0]), vec![], false).unwrap();
    n.backward_with_gradient(&go).unwrap();
    check_grad_f64(&x, &[1.0, -1.0, 0.0], "ord=inf go=2 grad");
}

/// ord=-inf — gradient routed to min-|x| elements, same tie split.
/// Live oracle:
/// ```python
/// x = torch.tensor([1.,-1.,5.], dtype=torch.float64, requires_grad=True)
/// n = torch.linalg.vector_norm(x, float('-inf'))   # 1.0
/// n.backward()
/// x.grad   # [0.5, -0.5, 0.]
/// ```
#[test]
fn ord_neg_inf_tie_split() {
    let x = leaf_f64(&[1.0, -1.0, 5.0]);
    let n = vector_norm(&x, f64::NEG_INFINITY).unwrap();
    fwd_close(&n, 1.0, "ord=-inf");
    n.backward().unwrap();
    check_grad_f64(&x, &[0.5, -0.5, 0.0], "ord=-inf tie grad");
}

/// ord=0 — count of nonzeros; torch attaches a grad_fn, backward SUCCEEDS,
/// and the gradient is UNDEFINED: the leaf's `.grad` stays `None`
/// (`norm_backward` `p == 0` returns an undefined Tensor). Live oracle:
/// ```python
/// x = torch.tensor([1.,0.,-2.], dtype=torch.float64, requires_grad=True)
/// n = torch.linalg.vector_norm(x, 0.0)   # 2.0; n.grad_fn is set
/// n.backward()
/// x.grad   # None
/// # and mixed it accumulates as zero:
/// x2.grad after (norm0(x2) + x2.sum()).backward()  # [1., 1., 1.]
/// ```
#[test]
fn ord_0_backward_succeeds_grad_is_none() {
    let x = leaf_f64(&[1.0, 0.0, -2.0]);
    let n = vector_norm(&x, 0.0).unwrap();
    fwd_close(&n, 2.0, "ord=0");
    assert!(
        n.requires_grad(),
        "ord=0: torch attaches a grad_fn; output must track"
    );
    n.backward().unwrap();
    assert!(
        x.grad().unwrap().is_none(),
        "ord=0: torch leaves leaf .grad None (undefined gradient)"
    );
}

/// ord=-1 (negative p, p<1 branch). Live oracle:
/// ```python
/// x = torch.tensor([1.,-2.,4.], dtype=torch.float64, requires_grad=True)
/// n = torch.linalg.vector_norm(x, -1.0)   # 0.5714285714285714
/// n.backward()
/// x.grad   # [0.32653061224489793, -0.08163265306122448, 0.02040816326530612]
/// ```
#[test]
fn ord_neg_1_grad() {
    let x = leaf_f64(&[1.0, -2.0, 4.0]);
    let n = vector_norm(&x, -1.0).unwrap();
    fwd_close(&n, 0.571_428_571_428_571_4, "ord=-1");
    n.backward().unwrap();
    check_grad_f64(
        &x,
        &[
            0.326_530_612_244_897_93,
            -0.081_632_653_061_224_48,
            0.020_408_163_265_306_12,
        ],
        "ord=-1 grad",
    );
}

/// ord=-2 (negative p). Live oracle:
/// ```python
/// x = torch.tensor([1.,-2.,4.], dtype=torch.float64, requires_grad=True)
/// n = torch.linalg.vector_norm(x, -2.0)   # 0.8728715609439696
/// n.backward()
/// x.grad   # [0.66504499881445300, -0.08313062485180663, 0.01039132810647583]
/// ```
#[test]
fn ord_neg_2_grad() {
    let x = leaf_f64(&[1.0, -2.0, 4.0]);
    let n = vector_norm(&x, -2.0).unwrap();
    fwd_close(&n, 0.872_871_560_943_969_6, "ord=-2");
    n.backward().unwrap();
    check_grad_f64(
        &x,
        &[
            0.665_044_998_814_453,
            -0.083_130_624_851_806_63,
            0.010_391_328_106_475_83,
        ],
        "ord=-2 grad",
    );
}

/// ord=-1 with a zero element: forward collapses to 0 and the gradient is
/// zero everywhere (`norm^(1-p) = 0` scale; `x == 0` masked). Live oracle:
/// ```python
/// x = torch.tensor([1.,-2.,0.], dtype=torch.float64, requires_grad=True)
/// n = torch.linalg.vector_norm(x, -1.0)   # 0.0
/// n.backward()
/// x.grad   # [0., -0., 0.]
/// ```
#[test]
fn ord_neg_1_zero_element_zero_grad() {
    let x = leaf_f64(&[1.0, -2.0, 0.0]);
    let n = vector_norm(&x, -1.0).unwrap();
    fwd_close(&n, 0.0, "ord=-1 zero-elem");
    n.backward().unwrap();
    check_grad_f64(&x, &[0.0, 0.0, 0.0], "ord=-1 zero-elem grad");
}

/// All-zero input: zero gradient for p=1 / p=3 / inf (torch returns exact
/// zeros, never NaN). Live oracles:
/// ```python
/// x = torch.zeros(2, dtype=torch.float64, requires_grad=True)
/// torch.linalg.vector_norm(x, p).backward()  ->  x.grad [0., 0.]
/// # probed for p in {1.0, 3.0, inf}; forward 0.0 in all three
/// ```
#[test]
fn all_zero_input_zero_grad() {
    for ord in [1.0, 3.0, f64::INFINITY] {
        let x = leaf_f64(&[0.0, 0.0]);
        let n = vector_norm(&x, ord).unwrap();
        fwd_close(&n, 0.0, &format!("ord={ord} all-zero"));
        n.backward().unwrap();
        check_grad_f64(&x, &[0.0, 0.0], &format!("ord={ord} all-zero grad"));
    }
}

/// ord=2 regression sanity — the pre-existing Euclidean branch keeps its
/// behavior (`dx = g * x / norm`). Live oracle:
/// ```python
/// x = torch.tensor([3.,4.], dtype=torch.float64, requires_grad=True)
/// n = torch.linalg.vector_norm(x, 2.0)   # 5.0
/// n.backward()
/// x.grad   # [0.6, 0.8]
/// ```
#[test]
fn ord_2_unchanged() {
    let x = leaf_f64(&[3.0, 4.0]);
    let n = vector_norm(&x, 2.0).unwrap();
    fwd_close(&n, 5.0, "ord=2");
    n.backward().unwrap();
    check_grad_f64(&x, &[0.6, 0.8], "ord=2 grad");
}

/// For p=2 a NaN norm propagates through the whole VJP. This guards the
/// p=2 branch against over-applying the zero-norm mask. Live oracle:
/// ```python
/// x = torch.tensor([nan, 3., -3., 0.], dtype=torch.float64, requires_grad=True)
/// torch.linalg.vector_norm(x, 2.0).backward()
/// x.grad   # [nan, nan, nan, nan]
/// ```
#[test]
fn ord_2_nan_norm_propagates_to_all_grad_entries() {
    let x = leaf_f64(&[f64::NAN, 3.0, -3.0, 0.0]);
    let n = vector_norm(&x, 2.0).unwrap();
    assert!(n.data().unwrap()[0].is_nan(), "ord=2 NaN forward");
    n.backward().unwrap();
    check_grad_f64_nan(&x, "ord=2 NaN grad");
}

/// f32 lane spot check, ord=3. Live oracle:
/// ```python
/// x = torch.tensor([1.,-2.,3.,-3.,0.], dtype=torch.float32, requires_grad=True)
/// n = torch.linalg.vector_norm(x, 3.0)   # 3.9790573120117188
/// n.backward()
/// x.grad   # [0.06315963715314865, -0.25263854861259460, 0.56843674182891846,
///          #  -0.56843674182891846, 0.0]
/// ```
#[test]
fn ord_3_f32_grad() {
    let x = leaf_f32(&[1.0, -2.0, 3.0, -3.0, 0.0]);
    let n = vector_norm(&x, 3.0).unwrap();
    let v = n.data().unwrap()[0];
    assert!((v - 3.979_057_3).abs() <= TOL_F32, "ord=3 f32 fwd: got {v}");
    n.backward().unwrap();
    let g = x
        .grad()
        .unwrap()
        .expect("ord=3 f32: no grad reached the leaf");
    let got = g.data().unwrap();
    let expected: [f32; 5] = [
        0.063_159_637,
        -0.252_638_55,
        0.568_436_74,
        -0.568_436_74,
        0.0,
    ];
    for (i, (a, e)) in got.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= TOL_F32,
            "ord=3 f32 grad[{i}]: got {a}, torch oracle {e}"
        );
    }
}

/// No-grad contract: a `requires_grad=false` input stays detached for
/// every ord (torch: no grad_fn without a tracking input).
#[test]
fn no_grad_input_stays_detached_every_ord() {
    for ord in [
        1.0,
        3.0,
        0.5,
        1.5,
        2.0,
        0.0,
        -1.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ] {
        let x =
            Tensor::from_storage(TensorStorage::cpu(vec![1.0, -2.0, 3.0]), vec![3], false).unwrap();
        let n = vector_norm(&x, ord).unwrap();
        assert!(
            !n.requires_grad() && n.grad_fn().is_none(),
            "ord={ord}: non-tracking input produced a tracking output"
        );
    }
}
