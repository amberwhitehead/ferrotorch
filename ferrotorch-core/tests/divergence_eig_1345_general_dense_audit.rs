//! R-BUILD-4 adversarial re-audit of commit `97ebfdf16` (#1345 REQ-12/REQ-14:
//! eig/eigvals COMPLEX backward — the private complex toolkit + EigvalsBackward
//! + EigBackwardW/EigBackwardV).
//!
//! The builder's own 6 lib tests cover only TWO matrix shapes:
//!   - upper-triangular real 3x3 ([[2,.5,.3],[0,3,.4],[0,0,5]]) where the
//!     eigenvectors are special (triangular -> simple V structure), and
//!   - the single complex-pair 2x2 [[1,-1],[1,1]] (eigenvalues 1±i), a highly
//!     symmetric structure.
//!
//! Neither is (a) a GENERAL dense non-symmetric matrix with a non-orthogonal
//! well-conditioned V, nor (b) a MIXED matrix with one complex-conjugate
//! eigenvalue pair AND one real eigenvalue (V has both real and complex columns,
//! Econj is a genuinely mixed complex matrix), nor (c) an ASYMMETRIC complex
//! pair. Those exercise the complex solve on a non-trivial `V^H`, the full
//! `ret @ V^H` conjugation, the Econj index orientation and per-column
//! phase-gauge handling in regimes the builder's narrow inputs mask. CORE-189
//! later replaced the explicit inverse in that solve with a direct LU solve, so
//! these cases still pin the production VJP without relying on special matrix
//! structure.
//!
//! Per R-CHAR-3 every expected value below is the `.grad` of a LIVE torch
//! float64 run (torch 2.11.0+cu130). Reproduction:
//!   import torch; torch.set_default_dtype(torch.float64)
//!   A = torch.tensor(<a>).reshape(n,n).clone().requires_grad_(True)
//!   # eigvals: L=torch.linalg.eigvals(A);
//!   #          ((L.real*cr).sum()+(L.imag*ci).sum()).backward()
//!   # eig:     L,V=torch.linalg.eig(A);
//!   #          (((V.real**2+V.imag**2)*MR).sum()+(L.real*cr).sum()
//!   #           +(L.imag*ci).sum()).backward()
//!   A.grad.reshape(-1)
//! The ferrotorch side drives the now-grad-aware PUBLIC `linalg::eigvals` /
//! `linalg::eig` through the IDENTICAL loss (the same construction the builder's
//! own lib helpers use), so a divergence is in the production VJP.

use ferrotorch_core::Tensor;
use ferrotorch_core::grad_fns::arithmetic::mul;
use ferrotorch_core::grad_fns::reduction::sum as reduce_sum;
use ferrotorch_core::linalg as linalg_fwd;
use ferrotorch_core::storage::TensorStorage;

fn leaf(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}

fn leaf32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}

fn no_grad_leaf(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

fn no_grad_leaf32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

fn assert_close(actual: &[f64], expected: &[f64], tol: f64, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch {} vs {}",
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < tol,
            "{label}[{i}]: ferrotorch={a}, torch={e}, diff={}",
            (a - e).abs()
        );
    }
}

fn assert_close32(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < tol,
            "{label}[{i}]: ferrotorch={a}, torch={e}, diff={}",
            (a - e).abs()
        );
    }
}

/// `sum_k (re(w_k)*cr_k + im(w_k)*ci_k)` on the complex `[n,2]` eigenvalues.
fn eigval_linear_loss(w: &Tensor<f64>, cr: &[f64], ci: &[f64]) -> Tensor<f64> {
    let n = cr.len();
    let mut wt = vec![0.0; n * 2];
    for k in 0..n {
        wt[2 * k] = cr[k];
        wt[2 * k + 1] = ci[k];
    }
    let wts = no_grad_leaf(&wt, &[n, 2]);
    reduce_sum(&mul(w, &wts).unwrap()).unwrap()
}

fn eigval_linear_loss32(w: &Tensor<f32>, cr: &[f32], ci: &[f32]) -> Tensor<f32> {
    let n = cr.len();
    let mut wt = vec![0.0; n * 2];
    for k in 0..n {
        wt[2 * k] = cr[k];
        wt[2 * k + 1] = ci[k];
    }
    let wts = no_grad_leaf32(&wt, &[n, 2]);
    reduce_sum(&mul(w, &wts).unwrap()).unwrap()
}

/// `sum((re^2+im^2)*MR[i,j])` on the complex `[n,n,2]` eigenvectors (phase-inv).
fn eigvec_phase_invariant_loss(v: &Tensor<f64>, mr: &[f64], n: usize) -> Tensor<f64> {
    let mut wt = vec![0.0; n * n * 2];
    for idx in 0..n * n {
        wt[2 * idx] = mr[idx];
        wt[2 * idx + 1] = mr[idx];
    }
    let wts = no_grad_leaf(&wt, &[n, n, 2]);
    let vsq = mul(v, v).unwrap();
    reduce_sum(&mul(&vsq, &wts).unwrap()).unwrap()
}

fn eigvals_grad(a_data: &[f64], n: usize, cr: &[f64], ci: &[f64]) -> Vec<f64> {
    let a = leaf(a_data, &[n, n]);
    let w = linalg_fwd::eigvals(&a).unwrap();
    assert!(w.grad_fn().is_some(), "eigvals must attach grad_fn");
    let loss = eigval_linear_loss(&w, cr, ci);
    loss.backward().unwrap();
    a.grad().unwrap().unwrap().data().unwrap().to_vec()
}

fn eigvals_grad32(a_data: &[f32], n: usize, cr: &[f32], ci: &[f32]) -> Vec<f32> {
    let a = leaf32(a_data, &[n, n]);
    let w = linalg_fwd::eigvals(&a).unwrap();
    assert!(w.grad_fn().is_some(), "eigvals must attach grad_fn");
    let loss = eigval_linear_loss32(&w, cr, ci);
    loss.backward().unwrap();
    a.grad().unwrap().unwrap().data().unwrap().to_vec()
}

fn eig_grad(a_data: &[f64], n: usize, mr: &[f64], cr: &[f64], ci: &[f64]) -> Vec<f64> {
    let a = leaf(a_data, &[n, n]);
    let (w, v) = linalg_fwd::eig(&a).unwrap();
    assert!(
        w.grad_fn().is_some() && v.grad_fn().is_some(),
        "eig must attach grad_fns on both outputs"
    );
    let lv = eigvec_phase_invariant_loss(&v, mr, n);
    let lw = eigval_linear_loss(&w, cr, ci);
    let loss = lv.add_t(&lw).unwrap();
    loss.backward().unwrap();
    a.grad().unwrap().unwrap().data().unwrap().to_vec()
}

fn eig_vonly(a_data: &[f64], n: usize, mr: &[f64]) -> Vec<f64> {
    let a = leaf(a_data, &[n, n]);
    let (_w, v) = linalg_fwd::eig(&a).unwrap();
    let loss = eigvec_phase_invariant_loss(&v, mr, n);
    loss.backward().unwrap();
    a.grad().unwrap().unwrap().data().unwrap().to_vec()
}

// ===========================================================================
// (1) General dense non-symmetric 3x3 with DISTINCT REAL eigenvalues
//     (~4.5547, 2.6499, 0.7954). V is real but genuinely non-orthogonal.
// ===========================================================================
const A_GEN: [f64; 9] = [4.0, 1.0, 2.0, 0.5, 3.0, 1.0, 0.2, 0.3, 1.0];

#[test]
fn eigvals_backward_general_dense_3x3_matches_torch() {
    let g = eigvals_grad(&A_GEN, 3, &[1.3, -0.7, 0.9], &[0.4, 0.6, -0.2]);
    let torch = [
        0.794_436_955_898_141_9,
        0.574_691_900_455_072_3,
        0.061_973_468_283_093_17,
        1.251_143_371_757_985_6,
        -0.202_307_488_740_244_43,
        -0.072_399_100_203_100_11,
        0.365_335_755_711_332_53,
        -0.071_731_049_263_934_16,
        0.907_870_532_842_102_7,
    ];
    assert_close(
        &g,
        &torch,
        1e-6,
        "eigvals general dense 3x3 A.grad vs torch",
    );
}

#[test]
fn eig_backward_general_dense_3x3_matches_torch() {
    let mr = [0.2, -0.5, 0.7, 0.3, 0.1, -0.4, -0.6, 0.8, 0.25];
    let g = eig_grad(&A_GEN, 3, &mr, &[1.3, -0.7, 0.9], &[0.4, 0.6, -0.2]);
    let torch = [
        0.866_380_177_816_953_8,
        0.355_745_731_766_169_36,
        0.178_059_862_364_260_75,
        1.498_728_808_679_781_9,
        -0.250_366_688_949_083_87,
        -0.289_580_559_527_914_95,
        0.536_967_820_821_379,
        -0.317_982_905_831_539_93,
        0.883_986_511_132_130_2,
    ];
    assert_close(&g, &torch, 1e-6, "eig general dense 3x3 A.grad vs torch");
}

// ===========================================================================
// (2) MIXED 3x3: one complex-conjugate eigenvalue PAIR (0.9871 ± 0.9839i) plus
//     one REAL eigenvalue (4.0259). The essential genuinely-complex case at
//     n=3 — V has two complex (conjugate) columns and one real column; Econj is
//     a fully complex 3x3 with a degenerate-diagonal structure (1 on diag).
//     Stresses the complex solve on V^H and the Econj index orientation.
// ===========================================================================
const A_MIX: [f64; 9] = [1.0, -1.0, 0.3, 1.0, 1.0, 0.5, 0.2, 0.1, 4.0];

#[test]
fn eigvals_backward_mixed_complex_real_3x3_matches_torch() {
    let g = eigvals_grad(&A_MIX, 3, &[1.3, -0.7, 0.9], &[0.4, 0.6, -0.2]);
    let torch = [
        0.302_123_160_038_934_14,
        0.105_224_178_954_166_57,
        0.040_450_526_597_206_5,
        -0.101_514_267_476_477_67,
        0.301_837_782_232_957_8,
        0.013_090_818_149_201_284,
        0.042_126_232_507_364_81,
        0.102_553_205_522_896_72,
        0.896_039_057_728_108,
    ];
    assert_close(
        &g,
        &torch,
        1e-6,
        "eigvals mixed complex+real 3x3 A.grad vs torch",
    );
}

#[test]
fn eig_backward_mixed_complex_real_3x3_matches_torch() {
    let mr = [0.2, -0.5, 0.7, 0.3, 0.1, -0.4, -0.6, 0.8, 0.25];
    let g = eig_grad(&A_MIX, 3, &mr, &[1.3, -0.7, 0.9], &[0.4, 0.6, -0.2]);
    let torch = [
        0.300_499_046_263_303_87,
        0.275_707_548_352_676_6,
        0.029_084_407_374_960_44,
        0.076_137_999_433_845_2,
        0.289_348_129_705_253_75,
        -0.067_806_619_385_886_88,
        0.024_057_235_493_125_252,
        0.082_174_780_675_396_1,
        0.910_152_824_031_442_3,
    ];
    assert_close(
        &g,
        &torch,
        1e-6,
        "eig mixed complex+real 3x3 A.grad vs torch",
    );
}

#[test]
fn eig_backward_v_only_mixed_complex_real_3x3_matches_torch() {
    let mr = [0.2, -0.5, 0.7, 0.3, 0.1, -0.4, -0.6, 0.8, 0.25];
    let g = eig_vonly(&A_MIX, 3, &mr);
    // V-only (gL=0): isolates the EigBackwardV path with a genuinely complex V
    // and Econj. This is where the unit-norm tangent projection + Econj divide
    // dominate.
    let torch = [
        -0.001_624_113_775_630_330_5,
        0.170_483_369_398_51,
        -0.011_366_119_222_246_044,
        0.177_652_266_910_322_76,
        -0.012_489_652_527_704_09,
        -0.080_897_437_535_088_18,
        -0.018_068_997_014_239_547,
        -0.020_378_424_847_500_66,
        0.014_113_766_303_334_25,
    ];
    assert_close(
        &g,
        &torch,
        1e-6,
        "eig V-only mixed complex+real 3x3 A.grad vs torch",
    );
}

// ===========================================================================
// (3) ASYMMETRIC 2x2 complex pair [[2,-3],[1,0]] -> eigenvalues 1 ± i*sqrt(2).
//     Not the symmetric [[1,-1],[1,1]] the builder tested; eigenvectors are
//     not "balanced", stressing the solve pivoting + normalization.
// ===========================================================================
const A_ASYM: [f64; 4] = [2.0, -3.0, 1.0, 0.0];

#[test]
fn eigvals_backward_asymmetric_complex_2x2_matches_torch() {
    let g = eigvals_grad(&A_ASYM, 2, &[1.3, -0.7], &[0.4, 0.6]);
    let torch = [
        0.370_710_678_118_654_93,
        0.070_710_678_118_654_72,
        -0.212_132_034_355_964_2,
        0.229_289_321_881_345_35,
    ];
    assert_close(
        &g,
        &torch,
        1e-6,
        "eigvals asymmetric complex 2x2 A.grad vs torch",
    );
}

#[test]
fn eig_backward_asymmetric_complex_2x2_matches_torch() {
    let mr = [0.5, -0.3, 0.2, 0.8];
    let g = eig_grad(&A_ASYM, 2, &mr, &[1.3, -0.7], &[0.4, 0.6]);
    let torch = [
        0.370_710_678_118_654_8,
        0.120_710_678_118_654_77,
        -0.062_132_034_355_964_08,
        0.229_289_321_881_345_35,
    ];
    assert_close(
        &g,
        &torch,
        1e-6,
        "eig asymmetric complex 2x2 A.grad vs torch",
    );
}

// ===========================================================================
// (4) ILL-CONDITIONED eigenvectors: upper-triangular 2x2 with eigenvalues
//     separated by 1e-4. The eigenvector matrix columns are nearly collinear,
//     so `solve(V^H, rhs)` amplifies rounding. This is the CORE-189 regime that
//     explicit inverse formation handles poorly.
// ===========================================================================
const A_ILL: [f64; 4] = [1.0, 1.0, 0.0, 1.0001];
const A_ILL32: [f32; 4] = [1.0, 1.0, 0.0, 1.0001];

#[test]
fn eigvals_backward_ill_conditioned_v_matches_torch() {
    let g = eigvals_grad(&A_ILL, 2, &[1.3, -0.7], &[0.4, 0.6]);
    let torch = [1.3, 0.0, -20_000.000_000_002_197, -0.7];
    assert_close(
        &g,
        &torch,
        1e-5,
        "eigvals ill-conditioned V A.grad vs torch",
    );
}

#[test]
fn eigvals_backward_ill_conditioned_v_f32_matches_torch() {
    let g = eigvals_grad32(&A_ILL32, 2, &[1.3, -0.7], &[0.4, 0.6]);
    let torch: [f32; 4] = [1.3, 0.0, -19_996.682, -0.7];
    assert_close32(
        &g,
        &torch,
        5e-2,
        "eigvals ill-conditioned V f32 A.grad vs torch",
    );
}

#[test]
fn eig_backward_ill_conditioned_v_matches_torch() {
    let mr = [0.5, -0.3, 0.2, 0.8];
    let g = eig_grad(&A_ILL, 2, &mr, &[1.3, -0.7], &[0.4, 0.6]);
    let torch = [
        1.299_780_000_003_991_4,
        -0.000_000_021_999_999_600_864_42,
        -19_997.800_000_042_113,
        -0.699_780_000_003_991_2,
    ];
    assert_close(&g, &torch, 1e-5, "eig ill-conditioned V A.grad vs torch");
}
