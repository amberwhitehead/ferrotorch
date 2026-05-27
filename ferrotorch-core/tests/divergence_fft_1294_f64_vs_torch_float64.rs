//! Divergence audit for commit cffd5117d (FFT norm/dim/s, #1294).
//!
//! The parity-sweep harness only ever exercises the **f32** path
//! (`dispatch_fft` does `wire.to_f32()`, the comparator is
//! `assert_close_f32` at the FFT-family `tolerance_for` rtol=1e-4 —
//! `tools/parity-sweep/runner/src/main.rs:4831,6396-6400`). The builder's
//! justification for rtol=1e-4 is "ferrotorch computes the transform in f64
//! via ferray_fft ... casts the result back ... values agree to 4-5
//! significant figures" (`main.rs:6381-6395`).
//!
//! This file tests the claim the harness CANNOT: it drives ferrotorch's
//! **f64** FFT path directly and compares against torch's **float64 /
//! complex128** output at 1e-9. If the f64 path matches torch float64 to
//! ~1e-9, the f32 rtol=1e-4 is genuine cast noise (legitimate). If the f64
//! path ALSO needs ~1e-4 to pass, rtol=1e-4 is masking an algorithmic
//! divergence.
//!
//! Expected values are torch float64/complex128 outputs produced by
//! `python3 -c "import torch; torch.fft.*"` (torch 2.11.0), NOT copied from
//! ferrotorch (R-CHAR-3). Each expected block names the exact torch call.

use ferrotorch_core::{FftNorm, fft_norm, fftn_norm, from_vec, hfft_norm, rfft_norm};

/// Tight tolerance: a pure-f64 butterfly should match torch float64 to
/// near machine epsilon. 1e-9 is ~6 orders of magnitude tighter than the
/// runner's masked f32 rtol=1e-4.
const F64_TOL: f64 = 1e-9;

fn assert_close_f64(actual: &[f64], expected: &[f64], tol: f64, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch {} vs {}",
        actual.len(),
        expected.len()
    );
    let mut worst = 0.0f64;
    let mut worst_i = 0usize;
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let d = (a - e).abs();
        if d > worst {
            worst = d;
            worst_i = i;
        }
    }
    assert!(
        worst < tol,
        "{label}: worst abs diff {worst:.3e} at index {worst_i} \
         (ferrotorch={}, torch_float64={}) exceeds tol {tol:.1e}",
        actual[worst_i],
        expected[worst_i],
    );
}

/// Complex tensor `[n, 2]` from interleaved re/im pairs.
fn complex_tensor(re: &[f64], im: &[f64]) -> ferrotorch_core::Tensor<f64> {
    assert_eq!(re.len(), im.len());
    let mut data = Vec::with_capacity(re.len() * 2);
    for i in 0..re.len() {
        data.push(re[i]);
        data.push(im[i]);
    }
    from_vec(data, &[re.len(), 2]).unwrap()
}

/// Divergence probe: ferrotorch's f64 `fft_norm` (backward) vs
/// `torch.fft.fft(c, norm="backward")` at float64/complex128.
/// torch oracle: `torch.fft.fft(complex([1,-1,3,0,-2.5,0.7],[2,0.5,-1,0,1.5,-0.3]))`.
#[test]
fn divergence_fft_f64_backward_matches_torch_float64() {
    let input = complex_tensor(
        &[1.0, -1.0, 3.0, 0.0, -2.5, 0.7],
        &[2.0, 0.5, -1.0, 0.0, 1.5, -0.3],
    );
    let out = fft_norm(&input, None, None, FftNorm::Backward).unwrap();
    // torch.fft.fft(c, norm="backward"), float64, interleaved re/im.
    let expected = [
        1.2,
        2.7,
        -0.8722431864335455,
        -1.4408965343808666,
        3.7578838324886474,
        7.885382907247958,
        1.8,
        2.3,
        -1.9578838324886472,
        -4.585382907247958,
        2.0722431864335453,
        5.140896534380866,
    ];
    assert_close_f64(out.data().unwrap(), &expected, F64_TOL, "fft backward f64");
}

/// Divergence probe: ortho-norm scale must be EXACT 1/sqrt(n), not approx.
/// torch oracle: `torch.fft.fft(c, norm="ortho")` (same input).
#[test]
fn divergence_fft_f64_ortho_matches_torch_float64() {
    let input = complex_tensor(
        &[1.0, -1.0, 3.0, 0.0, -2.5, 0.7],
        &[2.0, 0.5, -1.0, 0.0, 1.5, -0.3],
    );
    let out = fft_norm(&input, None, None, FftNorm::Ortho).unwrap();
    let expected = [
        0.48989794855663577,
        1.1022703842524304,
        -0.3560917897302476,
        -0.5882435468962937,
        1.53414965037528,
        3.2191940915369455,
        0.7348469228349535,
        0.938971068066885,
        -0.7993027275403266,
        -1.871974733006197,
        0.8459897382868834,
        2.0987622216125867,
    ];
    assert_close_f64(out.data().unwrap(), &expected, F64_TOL, "fft ortho f64");
}

/// Divergence probe: forward-norm = backward/n exactly.
/// torch oracle: `torch.fft.fft(c, norm="forward")` (same input).
#[test]
fn divergence_fft_f64_forward_matches_torch_float64() {
    let input = complex_tensor(
        &[1.0, -1.0, 3.0, 0.0, -2.5, 0.7],
        &[2.0, 0.5, -1.0, 0.0, 1.5, -0.3],
    );
    let out = fft_norm(&input, None, None, FftNorm::Forward).unwrap();
    let expected = [
        0.19999999999999998,
        0.45,
        -0.1453738644055909,
        -0.24014942239681109,
        0.6263139720814412,
        1.3142304845413264,
        0.3,
        0.3833333333333333,
        -0.3263139720814412,
        -0.7642304845413262,
        0.34537386440559087,
        0.8568160890634777,
    ];
    assert_close_f64(out.data().unwrap(), &expected, F64_TOL, "fft forward f64");
}

/// Divergence probe: fft along dim=0 of a 2-D [3,2] complex tensor.
/// torch oracle: `torch.fft.fft(complex([[1,2],[3,4],[5,6]], 0), dim=0)`.
/// Input is a [3, 2, 2] ferrotorch tensor (3 rows, 2 cols, complex pair).
#[test]
fn divergence_fft_f64_dim0_matches_torch_float64() {
    // 3 rows x 2 cols, all imag=0. Layout [3,2,2] interleaved.
    let m = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut data = Vec::new();
    for &v in &m {
        data.push(v);
        data.push(0.0);
    }
    let input = from_vec(data, &[3, 2, 2]).unwrap();
    let out = fft_norm(&input, None, Some(0), FftNorm::Backward).unwrap();
    assert_eq!(out.shape(), &[3, 2, 2]);
    // torch.fft.fft(mc, dim=0): per-column length-3 DFT.
    let expected = [
        9.0,
        0.0,
        12.0,
        0.0,
        -3.0,
        1.7320508075688772,
        -3.0,
        1.7320508075688772,
        -3.0,
        -1.7320508075688772,
        -3.0,
        -1.7320508075688772,
    ];
    assert_close_f64(out.data().unwrap(), &expected, F64_TOL, "fft dim0 f64");
}

/// Divergence probe: rfft ortho — Hermitian half-spectrum with 1/sqrt(n).
/// torch oracle: `torch.fft.rfft([1..8], norm="ortho")`, float64.
#[test]
#[allow(
    clippy::approx_constant,
    reason = "oracle f64 outputs from live torch.fft.rfft; values that happen to equal sqrt(2)/1-sqrt(2) are torch results, not hand-written math constants — replacing with consts::SQRT_2 would lose the file:line oracle traceability"
)]
fn divergence_rfft_f64_ortho_matches_torch_float64() {
    let input = from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[8]).unwrap();
    let out = rfft_norm(&input, None, None, FftNorm::Ortho).unwrap();
    assert_eq!(out.shape(), &[5, 2]);
    let expected = [
        12.727922061357855,
        0.0,
        -1.414213562373095,
        3.4142135623730945,
        -1.414213562373095,
        1.414213562373095,
        -1.414213562373095,
        0.5857864376269051,
        -1.414213562373095,
        0.0,
    ];
    assert_close_f64(out.data().unwrap(), &expected, F64_TOL, "rfft ortho f64");
}

/// Divergence probe: hfft ortho — c2r normalization must use the REAL
/// output size n (not the half-spectrum size) for the 1/sqrt(n) scale.
/// torch oracle: `torch.fft.hfft(hs, n=8, norm="ortho")`, complex128 input.
#[test]
fn divergence_hfft_f64_ortho_matches_torch_float64() {
    let input = complex_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0], &[0.0, 0.5, -0.5, 0.3, 0.0]);
    let out = hfft_norm(&input, Some(8), None, FftNorm::Ortho).unwrap();
    assert_eq!(out.shape(), &[8]);
    let expected = [
        8.48528137423857,
        -2.3677669529663685,
        0.1414213562373095,
        0.3393398282201788,
        0.0,
        -1.1677669529663686,
        -0.1414213562373095,
        -2.4606601717798213,
    ];
    assert_close_f64(out.data().unwrap(), &expected, F64_TOL, "hfft ortho f64");
}

/// Divergence probe: 2-D fftn ortho. N-D ortho applies 1/sqrt(n1*n2);
/// a per-axis vs whole-transform scale bug would show here.
/// torch oracle: `torch.fft.fftn(complex(arange(6).reshape(2,3),0), norm="ortho")`.
#[test]
#[allow(
    clippy::approx_constant,
    reason = "oracle f64 outputs from live torch.fft.fftn; values that happen to equal sqrt(6)/2 or 1/sqrt(2) are torch results, not hand-written math constants — replacing with consts would lose the file:line oracle traceability"
)]
fn divergence_fftn_f64_ortho_2d_matches_torch_float64() {
    // 2x3 complex, imag=0, layout [2,3,2].
    let vals: Vec<f64> = (0..6).map(|x| x as f64).collect();
    let mut data = Vec::new();
    for &v in &vals {
        data.push(v);
        data.push(0.0);
    }
    let input = from_vec(data, &[2, 3, 2]).unwrap();
    let out = fftn_norm(&input, None, None, FftNorm::Ortho).unwrap();
    assert_eq!(out.shape(), &[2, 3, 2]);
    let expected = [
        6.123724356957946,
        0.0,
        -1.2247448713915892,
        0.7071067811865476,
        -1.2247448713915892,
        -0.7071067811865476,
        -3.6742346141747677,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ];
    assert_close_f64(out.data().unwrap(), &expected, F64_TOL, "fftn ortho 2d f64");
}
