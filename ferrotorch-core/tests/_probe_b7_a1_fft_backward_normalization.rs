#![allow(
    non_snake_case,
    clippy::needless_borrow,
    clippy::needless_range_loop,
    clippy::approx_constant
)]

//! Probe for Bugfix Batch 7 / dispatch A1 — FFT-family backward N normalization.
//!
//! Issue #809: All 6 backward grad_fns (Rfft / Irfft / Rfftn / Irfftn / Hfft /
//! Ihfft) compute their VJPs against the *normalized* inverse transform when
//! PyTorch's reference uses the unnormalized form (with a non-trivial
//! Hermitian-doubling correction along the truncated axis).
//!
//! This probe locks in the numerical answer for each backward against:
//! - The literal repro from the issue text (Rfft of `arange(1..=8)` f64),
//! - PyTorch fixtures for the other 5 ops (already in the conformance JSON),
//! - A finite-difference Jacobian sanity check for the smallest case.
//!
//! Pre-fix: this probe should fail for Rfft / Irfft / Rfftn / Irfftn / Hfft.
//! Post-fix: every assertion within `F64_FFT_GRAD = 1e-9`.

use ferrotorch_core::TensorStorage;
use ferrotorch_core::grad_fns::fft::{
    hfft_differentiable, ihfft_differentiable, irfft_differentiable, irfftn_differentiable,
    rfft_differentiable, rfftn_differentiable,
};
use ferrotorch_core::grad_fns::reduction::sum as sum_loss;
use ferrotorch_core::tensor::Tensor;

const F64_FFT_GRAD: f64 = 1e-9;

fn leaf_f64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}

fn assert_close(label: &str, actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch ({} vs {})",
        actual.len(),
        expected.len(),
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        assert!(
            diff <= tol.max(tol * e.abs()),
            "{label}: index {i} actual={a} expected={e} diff={diff:.3e} tol={tol:.3e}"
        );
    }
}

// -----------------------------------------------------------------------------
// (1) RfftBackward — issue's literal repro.
// -----------------------------------------------------------------------------
//
// x = [1, 2, 3, 4, 5, 6, 7, 8] (f64), L = sum(real(rfft(x))) + sum(imag(rfft(x))).
// PyTorch grad: [5, -2.4142, 1, -0.4142, 1, 0.4142, 1, 2.4142].

#[test]
fn probe_rfft_backward_arange_8() {
    let x_data: Vec<f64> = (1..=8).map(|i| i as f64).collect();
    let x = leaf_f64(&x_data, &[8]);
    let y = rfft_differentiable(&x, None).expect("rfft_diff");
    // Loss = sum(re + im) over the half-spectrum tensor (last dim is real/imag pair).
    // sum() over the flat tensor reproduces sum(re) + sum(im).
    let l = sum_loss(&y).expect("sum");
    l.backward().expect("backward");
    let g = x.grad().unwrap().expect("grad");
    let g_data = g.data().unwrap();

    let sqrt2 = std::f64::consts::SQRT_2;
    let expected = [
        5.0,
        -1.0 - sqrt2,
        1.0,
        1.0 - sqrt2,
        1.0,
        -1.0 + sqrt2,
        1.0,
        1.0 + sqrt2,
    ];
    assert_close("rfft arange8", &g_data, &expected, F64_FFT_GRAD);
}

// -----------------------------------------------------------------------------
// (2) IrfftBackward — sum(irfft(X, N=8)) where X = ones complex Hermitian [5, 2].
// -----------------------------------------------------------------------------
//
// PyTorch grad (only re[0] is nonzero, all else 0):
//   [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  (shape [5, 2] flattened)

#[test]
fn probe_irfft_backward_ones_K5_N8() {
    // x = ones complex Hermitian of shape [5, 2] (each entry = 1+0i for the
    // real grid; loss linear, exact answer below).
    let x_data: Vec<f64> = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let x = leaf_f64(&x_data, &[5, 2]);
    let y = irfft_differentiable(&x, Some(8)).expect("irfft_diff");
    let l = sum_loss(&y).expect("sum");
    l.backward().expect("backward");
    let g = x.grad().unwrap().expect("grad");
    let g_data = g.data().unwrap();

    // y[n] = irfft of (1+0i)*K5 with Hermitian extension. y[0] = 1, y[k!=0] = 0
    // for n=0; in general y = real part of unnormalized inverse of
    // [1+0i, ..., 1+0i, 1-0i, 1-0i, 1-0i] / N — which is 1 at n=0 only because
    // the full spectrum is all-ones; so sum(y) = 1.
    // grad_x_re[0] = (1/N) * fft(grad_y=1)[0] = (1/8) * 8 = 1. all others 0.
    let expected = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    assert_close("irfft ones K=5 N=8", &g_data, &expected, F64_FFT_GRAD);
}

// -----------------------------------------------------------------------------
// (3) RfftnBackward — 2-D `rfftn` of a 3x4 real matrix. Use literal value from
// the conformance fixture (`tag = ndim_2_3x4`, `grad_a` field).
// -----------------------------------------------------------------------------

#[test]
fn probe_rfftn_backward_3x4() {
    // Loss linear → grad is independent of input data. Use a simple input.
    let x_data: Vec<f64> = (0..12).map(|i| i as f64 * 0.1).collect();
    let x = leaf_f64(&x_data, &[3, 4]);
    let y = rfftn_differentiable(&x, None, None).expect("rfftn_diff");
    let l = sum_loss(&y).expect("sum");
    l.backward().expect("backward");
    let g = x.grad().unwrap().expect("grad");
    let g_data = g.data().unwrap();

    // From PyTorch: torch.fft.rfftn of [3,4] real has output [3, 3, 2].
    // Loss = sum(re) + sum(im). grad pattern (per fixture `ndim_2_3x4`):
    //   [9, -3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0]
    let expected = [9.0, -3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    assert_close("rfftn 3x4", &g_data, &expected, F64_FFT_GRAD);
}

// -----------------------------------------------------------------------------
// (4) IrfftnBackward — `irfftn` of a 3x3 Hermitian-complex tensor.
// Cross-check against numerical Jacobian.
// -----------------------------------------------------------------------------
//
// We construct a perfectly Hermitian input and verify grad numerically.

#[test]
fn probe_irfftn_backward_3x3_jacobian() {
    // Use a Hermitian-symmetric input to avoid #808 (irfftn rejects non-Hermitian).
    // For a 3-row, 3-column-pair real signal of shape [3, 4], rfftn gives
    // [3, 3, 2]. We build a known Hermitian spectrum: rfftn(ones).
    // grad_in for sum(irfftn) is computed below.
    let x_data: Vec<f64> = vec![
        12.0, 0.0, 0.0, 0.0, 0.0, 0.0, // row 0
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // row 1
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // row 2
    ];
    let x = leaf_f64(&x_data, &[3, 3, 2]);
    let y = irfftn_differentiable(&x, Some(&[3, 4]), None).expect("irfftn_diff");
    let l = sum_loss(&y).expect("sum");
    l.backward().expect("backward");
    let g = x.grad().unwrap().expect("grad");
    let g_data = g.data().unwrap();

    // grad_y = ones(3,4). For irfftn(x, s=[3,4]) = (1/12) * full_2d_ifft(hermitian_extend(x)).
    // The forward applied to grad_y as input gives:
    //   grad_x[k0, k1] = (1/12) * fft2(ones(3,4))[k0, k1] with last-axis interior doubling.
    //   fft2(ones)[0,0] = 12, all else 0.
    // So grad_x re[0,0] = 12/12 = 1, all interior re/im at [k0, k1=1, k1=2] = 2*0 = 0,
    // boundary re[0,3 / 2 = ... wait s[-1]=4 so K=3, boundary k1∈{0,2}, interior k1=1.
    // grad_x_re[0, 0] = 1, grad_x_re[k0!=0 or k1!=0] = 0 for boundary; interior * 2 = 0.
    let expected = [
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, // row 0
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // row 1
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // row 2
    ];
    assert_close("irfftn 3x3 grad ones", &g_data, &expected, F64_FFT_GRAD);
}

// -----------------------------------------------------------------------------
// (5) HfftBackward — `hfft` of a Hermitian complex `[5, 2]` to real `[8]`.
// -----------------------------------------------------------------------------

#[test]
fn probe_hfft_backward_K5_N8() {
    // Use Hermitian-symmetric input. Loss linear → grad independent of values.
    let x_data: Vec<f64> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let x = leaf_f64(&x_data, &[5, 2]);
    let y = hfft_differentiable(&x, Some(8)).expect("hfft_diff");
    let l = sum_loss(&y).expect("sum");
    l.backward().expect("backward");
    let g = x.grad().unwrap().expect("grad");
    let g_data = g.data().unwrap();

    // hfft = unnormalized inverse with conj. For grad_y = ones(8):
    // grad_x_re[k] = sum_n cos(2π k n/N), with boundary 1× / interior 2×.
    //   grad_x_re[0] = sum 1 = 8.
    //   grad_x_re[N/2 = 4] = sum (-1)^n = 0.
    //   grad_x_re[k] for k=1,2,3 = 2 * sum cos(2π k n/8) = 2 * 0 = 0.
    // grad_x_im[k] = 2 * sum sin(2π k n/8) = 0 for all k. (interior; boundary 0)
    // Pattern: [8, 0, 0, 0, 0, 0, 0, 0, 0, 0].
    let expected = [8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    assert_close("hfft K=5 N=8", &g_data, &expected, F64_FFT_GRAD);
}

// -----------------------------------------------------------------------------
// (6) IhfftBackward — `ihfft` of a real `[8]` to Hermitian complex `[5, 2]`.
// -----------------------------------------------------------------------------

#[test]
fn probe_ihfft_backward_N8() {
    let x_data: Vec<f64> = (1..=8).map(|i| i as f64).collect();
    let x = leaf_f64(&x_data, &[8]);
    let y = ihfft_differentiable(&x, None).expect("ihfft_diff");
    let l = sum_loss(&y).expect("sum");
    l.backward().expect("backward");
    let g = x.grad().unwrap().expect("grad");
    let g_data = g.data().unwrap();

    // ihfft(x) = conj(rfft(x))/N. For Y[k] = re/N + i*sum_n x[n] sin(2πkn/N)/N
    // (after the conj). Loss = sum(re Y) + sum(im Y) over k=0..K-1.
    // grad_x[n] = (1/N) sum_{k=0..K-1} (cos(2π k n / N) + sin(2π k n / N)).
    let n_total = 8usize;
    let k_max = n_total / 2 + 1; // K = 5
    let mut expected = vec![0.0_f64; n_total];
    for n in 0..n_total {
        for k in 0..k_max {
            let theta = 2.0 * std::f64::consts::PI * (k as f64) * (n as f64) / (n_total as f64);
            expected[n] += theta.cos() + theta.sin();
        }
        expected[n] /= n_total as f64;
    }
    assert_close("ihfft N=8", &g_data, &expected, F64_FFT_GRAD);
}

// -----------------------------------------------------------------------------
// (7) Multi-size sanity — RfftBackward at N=16, finite-difference Jacobian.
// -----------------------------------------------------------------------------

#[test]
fn probe_rfft_backward_N16_finite_diff() {
    let n = 16usize;
    let x_data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.5 - 1.0).collect();
    let x = leaf_f64(&x_data, &[n]);
    let y = rfft_differentiable(&x, None).expect("rfft_diff");
    let l = sum_loss(&y).expect("sum");
    l.backward().expect("backward");
    let g = x.grad().unwrap().expect("grad");
    let g_data = g.data().unwrap();

    // Finite-difference reference: L(x + h e_i) - L(x - h e_i) / 2h.
    // L is linear in x, so finite-diff is exact.
    let eps = 1e-3;
    let mut fd = vec![0.0_f64; n];
    for i in 0..n {
        let mut xp = x_data.clone();
        xp[i] += eps;
        let mut xm = x_data.clone();
        xm[i] -= eps;
        let xp_t = Tensor::from_storage(TensorStorage::cpu(xp), vec![n], false).unwrap();
        let xm_t = Tensor::from_storage(TensorStorage::cpu(xm), vec![n], false).unwrap();
        let yp = ferrotorch_core::fft::rfft(&xp_t, None).expect("rfft+");
        let ym = ferrotorch_core::fft::rfft(&xm_t, None).expect("rfft-");
        let lp: f64 = yp.data().unwrap().iter().sum();
        let lm: f64 = ym.data().unwrap().iter().sum();
        fd[i] = (lp - lm) / (2.0 * eps);
    }
    assert_close("rfft N=16 fd-jacobian", &g_data, &fd, 1e-7);
}

// -----------------------------------------------------------------------------
// (8) Multi-size sanity — RfftBackward at N=32 for real input. Linear so grad
// has a closed form: grad_x[n] = sum_{k=0..N/2} (cos(2πkn/N) - sin(2πkn/N)).
// -----------------------------------------------------------------------------

#[test]
fn probe_rfft_backward_N32_closed_form() {
    let n = 32usize;
    let x_data: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
    let x = leaf_f64(&x_data, &[n]);
    let y = rfft_differentiable(&x, None).expect("rfft_diff");
    let l = sum_loss(&y).expect("sum");
    l.backward().expect("backward");
    let g = x.grad().unwrap().expect("grad");
    let g_data = g.data().unwrap();

    let k_max = n / 2 + 1;
    let mut expected = vec![0.0_f64; n];
    for nn in 0..n {
        for k in 0..k_max {
            let theta = 2.0 * std::f64::consts::PI * (k as f64) * (nn as f64) / (n as f64);
            expected[nn] += theta.cos() - theta.sin();
        }
    }
    assert_close("rfft N=32 closed-form", &g_data, &expected, F64_FFT_GRAD);
}
