//! ACToR critic re-audit of `svd_backward` (#1577, commit `c9830de42`).
//!
//! The builder verified `A.grad` vs LIVE torch float64 for the gauge-invariant
//! loss `sum((U*U)*MU) + sum((Vh*Vh)*MV) + sum(S*c)` on five cases:
//! square 3x3, tall 4x3, wide 3x4, S-only square, S-only tall. This file
//! attacks the SPACE THE BUILDER LEFT UNTESTED:
//!
//!   1. CLOSE-but-distinct singular values (3x3, S = [2.0, 1.95, 1.0]) — the
//!      F-matrix `1/(S^2[j]-S^2[i])` is large there (gap S^2 ≈ 0.195), so a
//!      sign error or i<->j swap in the gap term shows as a O(5x) blow-up.
//!   2. HIGHER aspect-ratio rectangular: TALL 5x2 (m/n = 2.5) and WIDE 2x5 —
//!      a wrong `(I-UUᵀ)gU S⁻¹Vᵀ` / `U S⁻¹ gVᵀ (I-VVᵀ)` projector or S⁻¹
//!      scaling is amplified at high aspect ratio.
//!   3. U-ONLY and V-ONLY partials (gS=gVh=None / gS=gU=None) — the builder
//!      tested S-only but NOT the gU-only or gVh-only None-handling branches,
//!      including the projector branches exercised through gU alone (tall) and
//!      gVh alone (wide).
//!
//! Every `TORCH_*` constant below is a LIVE `torch 2.11.0+cu130 float64`
//! result from `torch.linalg.svd(A, full_matrices=False)` (R-CHAR-3 (a),
//! reproduced by /tmp/svd_oracle.py 2026-05-27). NONE is copied from
//! ferrotorch. The loss is gauge-invariant (each `U_ij^2` / `Vh_ij^2` is
//! unchanged under a column sign flip; `S` is gauge-free), so torch and
//! ferrotorch must agree regardless of their differing LAPACK/faer sign
//! conventions — exactly the well-posedness the builder relied on.
//!
//!     A  = torch.tensor(<a>).reshape(<shape>).clone().requires_grad_(True)
//!     U,S,Vh = torch.linalg.svd(A, full_matrices=False)
//!     L = ((U*U)*MU).sum() + ((Vh*Vh)*MV).sum() + (S*c).sum()   # or U-only / V-only
//!     L.backward(); A.grad.reshape(-1)

use ferrotorch_core::Tensor;
use ferrotorch_core::grad_fns::arithmetic::mul;
use ferrotorch_core::grad_fns::arithmetic::add;
use ferrotorch_core::grad_fns::reduction::sum as reduce_sum;
use ferrotorch_core::linalg::svd;
use ferrotorch_core::storage::TensorStorage;

fn leaf(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}
fn nograd(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

fn assert_close(actual: &[f64], torch: &[f64], tol: f64, label: &str) {
    assert_eq!(
        actual.len(),
        torch.len(),
        "{label}: length mismatch {} vs {}",
        actual.len(),
        torch.len()
    );
    let mut maxdiff = 0.0_f64;
    for (i, (&a, &t)) in actual.iter().zip(torch.iter()).enumerate() {
        let d = (a - t).abs();
        if d > maxdiff {
            maxdiff = d;
        }
        assert!(
            d < tol,
            "{label} grad[{i}]: ferrotorch={a}, torch={t}, diff={d} (maxdiff so far {maxdiff})"
        );
    }
}

/// Full gauge-invariant loss `sum((U*U)*MU)+sum((Vh*Vh)*MV)+sum(S*c)` through
/// the PUBLIC grad-aware `ferrotorch_core::linalg::svd`.
fn gauge_grad(a_data: &[f64], shape: &[usize], mu: &[f64], mv: &[f64], c: &[f64]) -> Vec<f64> {
    let a = leaf(a_data, shape);
    let (u, s, vh) = svd(&a).unwrap();
    assert!(
        u.grad_fn().is_some() && s.grad_fn().is_some() && vh.grad_fn().is_some(),
        "svd must attach grad_fns on all three outputs"
    );
    let mu_t = nograd(mu, u.shape());
    let mv_t = nograd(mv, vh.shape());
    let c_t = nograd(c, s.shape());
    let lu = reduce_sum(&mul(&mul(&u, &u).unwrap(), &mu_t).unwrap()).unwrap();
    let lv = reduce_sum(&mul(&mul(&vh, &vh).unwrap(), &mv_t).unwrap()).unwrap();
    let ls = reduce_sum(&mul(&s, &c_t).unwrap()).unwrap();
    let loss = add(&add(&lu, &lv).unwrap(), &ls).unwrap();
    loss.backward().unwrap();
    a.grad().unwrap().unwrap().data().unwrap().to_vec()
}

/// U-only gauge-invariant loss `sum((U*U)*MU)` — gS=gVh=None branch.
fn u_only_grad(a_data: &[f64], shape: &[usize], mu: &[f64]) -> Vec<f64> {
    let a = leaf(a_data, shape);
    let (u, _s, _vh) = svd(&a).unwrap();
    let mu_t = nograd(mu, u.shape());
    let loss = reduce_sum(&mul(&mul(&u, &u).unwrap(), &mu_t).unwrap()).unwrap();
    loss.backward().unwrap();
    a.grad().unwrap().unwrap().data().unwrap().to_vec()
}

/// V-only gauge-invariant loss `sum((Vh*Vh)*MV)` — gS=gU=None branch.
fn v_only_grad(a_data: &[f64], shape: &[usize], mv: &[f64]) -> Vec<f64> {
    let a = leaf(a_data, shape);
    let (_u, _s, vh) = svd(&a).unwrap();
    let mv_t = nograd(mv, vh.shape());
    let loss = reduce_sum(&mul(&mul(&vh, &vh).unwrap(), &mv_t).unwrap()).unwrap();
    loss.backward().unwrap();
    a.grad().unwrap().unwrap().data().unwrap().to_vec()
}

// ---------------------------------------------------------------------------
// 1. CLOSE-but-distinct singular values: F-matrix sign/i<->j stress.
//    A = Q1 diag([2.0, 1.95, 1.0]) Q2^T. Gap S^2[0]-S^2[1] ≈ 0.195, so the
//    off-diag 1/E term is ~5x larger than a well-separated case — a sign or
//    transpose error in the gap matrix is unmissable.
// ---------------------------------------------------------------------------
#[test]
fn svd_backward_close_singular_3x3_matches_torch() {
    let a = [
        1.865891085855741, -0.22477159228786844, 0.6699877601755171,
        0.26350482720595897, 1.8318447213084508, -0.007985525067841984,
        -0.3008831850485233, 0.5818901899786847, 0.9834958354990592,
    ];
    let mu = [0.2, -0.5, 0.7, 0.3, 0.1, -0.4, -0.6, 0.8, 0.25];
    let mv = [0.4, 0.1, -0.3, -0.2, 0.5, 0.6, 0.15, -0.7, 0.3];
    let c = [1.3, -0.7, 0.9];
    let g = gauge_grad(&a, &[3, 3], &mu, &mv, &c);
    let torch = [
        5.298138800580567, -6.798003887894706, 0.40678673583201497,
        -5.338415759354935, -4.033298670401751, -3.3632259826905604,
        -2.987867481407046, -1.811690553316089, -0.6384026018512374,
    ];
    assert_close(&g, &torch, 1e-6, "svd close-singular 3x3 A.grad vs torch");
}

// ---------------------------------------------------------------------------
// 2a. TALL 5x2 (m>n, aspect 2.5) — high-aspect `m>n` projector via gU.
// ---------------------------------------------------------------------------
#[test]
fn svd_backward_tall_5x2_matches_torch() {
    let a = [3.0, 0.4, 0.2, 2.2, 0.25, 0.1, 0.6, 0.35, 0.15, 0.45];
    let mu = [0.2, -0.5, 0.7, 0.3, 0.1, -0.4, -0.6, 0.8, 0.25, -0.3];
    let mv = [0.4, 0.1, -0.3, 0.5];
    let c = [1.1, -0.6];
    let g = gauge_grad(&a, &[5, 2], &mu, &mv, &c);
    let torch = [
        1.180791073676472, 0.1973190136761945, 0.2363760681294639,
        -0.6464319397779605, 0.1033522661077754, -0.02647910256740578,
        0.1544404513021659, -0.11048566286732203, 0.1136887266900019,
        -0.22414756687155052,
    ];
    assert_close(&g, &torch, 1e-6, "svd tall 5x2 A.grad vs torch");
}

// ---------------------------------------------------------------------------
// 2b. WIDE 2x5 (m<n) — high-aspect `m<n` projector via gVh.
// ---------------------------------------------------------------------------
#[test]
fn svd_backward_wide_2x5_matches_torch() {
    let a = [3.0, 0.4, 0.2, 0.5, 0.3, 0.1, 2.2, 0.3, 0.15, 0.25];
    let mu = [0.2, -0.5, 0.7, 0.3];
    let mv = [0.4, 0.1, -0.3, 0.2, -0.2, 0.5, 0.6, -0.1, 0.15, -0.7];
    let c = [1.2, -0.5];
    let g = gauge_grad(&a, &[2, 5], &mu, &mv, &c);
    let torch = [
        1.1354631669600612, 0.3292984222198734, 0.08975686949557016,
        0.18513087837884779, 0.12289751390307416, 0.33014111107295263,
        -0.43154383654023915, -0.11970685591834646, 0.015023170874864052,
        -0.11748151852887387,
    ];
    assert_close(&g, &torch, 1e-6, "svd wide 2x5 A.grad vs torch");
}

// ---------------------------------------------------------------------------
// 3a. U-ONLY square 3x3 (gS=gVh=None). Loss = sum((U*U)*MU).
// ---------------------------------------------------------------------------
#[test]
fn svd_backward_u_only_square_3x3_matches_torch() {
    let a = [4.0, 0.5, 0.3, 0.2, 2.5, 0.1, 0.3, 0.15, 1.2];
    let mu = [0.2, -0.5, 0.7, 0.3, 0.1, -0.4, -0.6, 0.8, 0.25];
    let g = u_only_grad(&a, &[3, 3], &mu);
    let torch = [
        0.029328629434962557, -0.0363669197957946, -0.004298238590405856,
        -0.07008240645441464, -0.02394609269870782, -0.02574434665386807,
        -0.02179163753115973, -0.04517849775103137, -0.006725977199824413,
    ];
    assert_close(&g, &torch, 1e-6, "svd U-only 3x3 A.grad vs torch");
}

// ---------------------------------------------------------------------------
// 3b. V-ONLY square 3x3 (gS=gU=None). Loss = sum((Vh*Vh)*MV).
// ---------------------------------------------------------------------------
#[test]
fn svd_backward_v_only_square_3x3_matches_torch() {
    let a = [4.0, 0.5, 0.3, 0.2, 2.5, 0.1, 0.3, 0.15, 1.2];
    let mv = [0.4, 0.1, -0.3, -0.2, 0.5, 0.6, 0.15, -0.7, 0.3];
    let g = v_only_grad(&a, &[3, 3], &mv);
    let torch = [
        0.06761228704293544, -0.15755303560986564, -0.03986653293497512,
        -0.08248076570493339, -0.05071826858106933, -0.09688195933102081,
        -0.004441682277739582, -0.05853479636876305, -0.0138499381840984,
    ];
    assert_close(&g, &torch, 1e-6, "svd V-only 3x3 A.grad vs torch");
}

// ---------------------------------------------------------------------------
// 3c. U-ONLY TALL 4x3 (gS=gVh=None, m>n) — projector exercised through gU
//     ALONE. If the m>n branch only fires when gU is part of the joint VJP
//     this catches it.
// ---------------------------------------------------------------------------
#[test]
fn svd_backward_u_only_tall_4x3_matches_torch() {
    let a = [3.0, 0.4, 0.2, 0.1, 2.2, 0.3, 0.25, 0.1, 1.5, 0.6, 0.35, 0.4];
    let mu = [0.2, -0.5, 0.7, 0.3, 0.1, -0.4, -0.6, 0.8, 0.25, 0.5, -0.2, 0.9];
    let g = u_only_grad(&a, &[4, 3], &mu);
    let torch = [
        0.10487491466826758, -0.050713429097242725, -0.030740342581945464,
        -0.1535357612428205, -0.019278319606425554, -0.15765724772379358,
        -0.05959266551900041, -0.18256232012065507, -0.11136503821712838,
        0.058257840578831925, -0.13614838655612158, 0.07442734097722879,
    ];
    assert_close(&g, &torch, 1e-6, "svd U-only tall 4x3 A.grad vs torch");
}

// ---------------------------------------------------------------------------
// 3d. V-ONLY WIDE 3x4 (gS=gU=None, m<n) — projector exercised through gVh
//     ALONE.
// ---------------------------------------------------------------------------
#[test]
fn svd_backward_v_only_wide_3x4_matches_torch() {
    let a = [3.0, 0.4, 0.2, 0.5, 0.1, 2.2, 0.3, 0.15, 0.25, 0.1, 1.5, 0.35];
    let mv = [0.4, 0.1, -0.3, 0.2, -0.2, 0.5, 0.6, -0.1, 0.15, -0.7, 0.3, 0.45];
    let g = v_only_grad(&a, &[3, 4], &mv);
    let torch = [
        0.20440326414224666, -0.2743774079847026, -0.07868321403099147,
        0.012127584327848455, -0.14266064499916262, -0.006401972350948143,
        -0.4549763965330436, -0.16687558893553028, 0.031093203089195262,
        -0.3281229372812002, -0.17821254169683234, -0.033010034620775904,
    ];
    assert_close(&g, &torch, 1e-6, "svd V-only wide 3x4 A.grad vs torch");
}
