//! Discriminator probes for the #1687 GPU non-symmetric eig fix (complex-A
//! datatype contract). The builder's own e2e suite
//! (`eig_complex_transpose_e2e.rs`) covers only: 2x2 rotation, 3x3 block,
//! diag(3,5,-2), and a 2x2 triangular. This file hunts divergence in the cases
//! it did NOT test:
//!
//!   - larger n (8x8) random non-symmetric, eigenvalues as a multiset vs torch
//!   - companion matrix with multiple complex-conjugate pairs (8th roots)
//!   - defective / non-diagonalizable matrix [[1,1],[0,1]]
//!   - symmetric real-spectrum matrix (imag must be ~0)
//!   - degenerate / repeated eigenvalues
//!   - n=1 trivial, n=0 empty
//!   - the gpu_real_to_complex_f32/f64 kernel exactness at odd/large sizes
//!   - EVERY eigenpair residual A v = lambda v (not just the first)
//!
//! GROUND TRUTH: every expected eigenvalue here was produced by LIVE
//! torch.linalg.eig (torch 2.11.0+cu130, RTX 3090) and is reproduced in the
//! per-test doc comment. NO expected value is copied from the ferrotorch side
//! (R-CHAR-3).
#![cfg(feature = "cuda")]

use ferrotorch_gpu::cusolver::{gpu_eig_f32_dev, gpu_eig_f64_dev};
use ferrotorch_gpu::kernels::{gpu_real_to_complex_f32, gpu_real_to_complex_f64};
use ferrotorch_gpu::transfer::{cpu_to_gpu, gpu_to_cpu};
use ferrotorch_gpu::{GpuDevice, init_cuda_backend};

fn ensure_init() {
    if !ferrotorch_core::gpu_dispatch::has_gpu_backend() {
        init_cuda_backend().expect("init_cuda_backend");
    }
}

/// Match every computed eigenvalue to a not-yet-claimed expected one within
/// `tol` (greedy nearest, complex distance). Order-independent multiset check.
fn assert_eigvals_set(w: &[f64], n: usize, expected: &[(f64, f64)], tol: f64) {
    assert_eq!(expected.len(), n, "expected set size must equal n");
    let got: Vec<(f64, f64)> = (0..n).map(|j| (w[2 * j], w[2 * j + 1])).collect();
    let mut claimed = vec![false; expected.len()];
    for &g in &got {
        let mut best: Option<(usize, f64)> = None;
        for (i, &e) in expected.iter().enumerate() {
            if claimed[i] {
                continue;
            }
            let d = ((g.0 - e.0).powi(2) + (g.1 - e.1).powi(2)).sqrt();
            if best.is_none() || d < best.unwrap().1 {
                best = Some((i, d));
            }
        }
        match best {
            Some((i, d)) if d < tol => claimed[i] = true,
            _ => panic!(
                "eigenvalue {g:?} has no match within {tol:e} in expected {expected:?} (got {got:?})"
            ),
        }
    }
}

/// Verify A v_j = lambda_j v_j for EVERY column j of the row-major
/// interleaved-complex VR buffer returned by the eig path. `a_row` is the real
/// n x n row-major matrix.
fn check_eigenpairs(a_row: &[f64], w: &[f64], vr_row: &[f64], n: usize, tol: f64) {
    for j in 0..n {
        let lam_re = w[2 * j];
        let lam_im = w[2 * j + 1];
        for r in 0..n {
            let mut av_re = 0.0;
            let mut av_im = 0.0;
            for k in 0..n {
                let a = a_row[r * n + k];
                av_re += a * vr_row[(k * n + j) * 2];
                av_im += a * vr_row[(k * n + j) * 2 + 1];
            }
            let v_re = vr_row[(r * n + j) * 2];
            let v_im = vr_row[(r * n + j) * 2 + 1];
            let lv_re = lam_re * v_re - lam_im * v_im;
            let lv_im = lam_re * v_im + lam_im * v_re;
            let err = ((av_re - lv_re).powi(2) + (av_im - lv_im).powi(2)).sqrt();
            assert!(
                err < tol,
                "A v = lambda v violated: col {j} row {r}: err={err:.3e} (lam=({lam_re},{lam_im}))"
            );
        }
        let norm2: f64 = (0..n)
            .map(|k| {
                let re = vr_row[(k * n + j) * 2];
                let im = vr_row[(k * n + j) * 2 + 1];
                re * re + im * im
            })
            .sum();
        assert!(norm2 > 1e-6, "eigenvector col {j} is ~zero");
    }
}

fn run_f64(a_row: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    ensure_init();
    let dev = GpuDevice::new(0).expect("device");
    let d_a = cpu_to_gpu(a_row, &dev).expect("upload A");
    let (d_w, d_vr) = gpu_eig_f64_dev(&d_a, n, &dev).expect("gpu_eig_f64_dev must run");
    let w = gpu_to_cpu(&d_w, &dev).expect("download W");
    let vr = gpu_to_cpu(&d_vr, &dev).expect("download VR");
    (w, vr)
}

// ===========================================================================
// PROBE 1 — larger n=8 random non-symmetric, eigenvalues as a multiset.
// torch.linalg.eig (f64, seed 1687) multiset:
//   (-3.702216,0) (-1.756766,0) (-1.01131,-0.213558) (-1.01131,0.213558)
//   (0.084655,-2.314675) (0.084655,2.314675) (1.343042,0) (2.368522,0)
// ===========================================================================
#[test]
fn eig_f64_dev_rand_8x8_eigvals_match_torch() {
    let a_row: Vec<f64> = vec![
        -0.9442729465697248,
        2.1088314558929464,
        1.0883842432738466,
        0.3996407777515308,
        0.5747094125731214,
        -0.8877392149005505,
        0.08327009207878158,
        -0.8734896557467873,
        -0.16960265491641555,
        0.1685789813154307,
        0.4021887171478154,
        0.5912459271083189,
        -0.9539979204593794,
        -1.9696565671285429,
        -0.5643267420460211,
        1.2033349696299804,
        1.310959994430523,
        0.606963513920415,
        0.11296840203937493,
        -0.019701610309299847,
        0.7272387248252795,
        -1.0601602914405572,
        -0.7301679384338858,
        -0.05434584252726653,
        0.39364203620568217,
        0.41712264701160506,
        1.2364540003268585,
        -1.464719847571328,
        -1.8875063742605234,
        0.9997484692397808,
        -0.724448170641284,
        1.4746544833968724,
        1.6240308302956872,
        1.1993394860254922,
        0.8564163399506219,
        -0.7594775712868722,
        -2.4488563349552255,
        1.891219961633063,
        -0.12874337044649875,
        -0.5023774038510367,
        -0.785846636418831,
        0.9511176453858454,
        -0.5598018763019493,
        -1.0730471611497474,
        0.13111997036085302,
        0.7924334423409372,
        -0.7803413864092041,
        0.5055361236487617,
        1.2715567385450148,
        -0.4106576691455445,
        -0.26777109869501675,
        1.130909158050243,
        -1.1281994720335422,
        -0.464825487755811,
        0.5191529814618616,
        -1.9675839104268127,
        -0.9172197415914208,
        -0.8299182573178825,
        1.024755776862596,
        0.5156683115481856,
        1.2011941626901288,
        -0.6240451596490983,
        -0.3788991195207488,
        -0.3360117533227261,
    ];
    let n = 8;
    let (w, vr) = run_f64(&a_row, n);
    assert_eigvals_set(
        &w,
        n,
        &[
            (-3.702216, 0.0),
            (-1.756766, 0.0),
            (-1.01131, -0.213558),
            (-1.01131, 0.213558),
            (0.084655, -2.314675),
            (0.084655, 2.314675),
            (1.343042, 0.0),
            (2.368522, 0.0),
        ],
        1e-4,
    );
    check_eigenpairs(&a_row, &w, &vr, n, 1e-7);
}

// ===========================================================================
// PROBE 2 — companion matrix of x^4 + 1 = 0 (4 distinct complex-conjugate
// pairs of primitive 8th roots of unity).
// torch.linalg.eig (f64):
//   (-0.70710678, +-0.70710678), (0.70710678, +-0.70710678)
// Also verify eigvectors of conjugate eigenvalues are conjugate.
// ===========================================================================
#[test]
fn eig_f64_dev_companion_x4p1_complex_pairs_match_torch() {
    // companion of x^4 + 1: row-major
    // [[0,0,0,-1],[1,0,0,0],[0,1,0,0],[0,0,1,0]]
    let a_row = vec![
        0.0, 0.0, 0.0, -1.0, //
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0,
    ];
    let n = 4;
    let (w, vr) = run_f64(&a_row, n);
    let s = std::f64::consts::FRAC_1_SQRT_2; // 0.70710678
    assert_eigvals_set(&w, n, &[(-s, s), (-s, -s), (s, s), (s, -s)], 1e-7);
    check_eigenpairs(&a_row, &w, &vr, n, 1e-7);
}

// ===========================================================================
// PROBE 3 — symmetric 3x3 [[2,1,0],[1,2,1],[0,1,2]], real spectrum.
// torch.linalg.eig (f64): {2+sqrt2, 2, 2-sqrt2} all imag = 0 exactly.
//   = (3.41421356,0), (2.0,0), (0.58578644,0)
// Probe: imag parts must be ~0 (not tiny-nonzero garbage from promotion).
// ===========================================================================
#[test]
fn eig_f64_dev_symmetric_3x3_real_spectrum_match_torch() {
    let a_row = vec![2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0];
    let n = 3;
    let (w, vr) = run_f64(&a_row, n);
    let sq2 = std::f64::consts::SQRT_2;
    assert_eigvals_set(
        &w,
        n,
        &[(2.0 + sq2, 0.0), (2.0, 0.0), (2.0 - sq2, 0.0)],
        1e-7,
    );
    for j in 0..n {
        assert!(
            w[2 * j + 1].abs() < 1e-9,
            "imag of eigval {j} nonzero: {}",
            w[2 * j + 1]
        );
    }
    check_eigenpairs(&a_row, &w, &vr, n, 1e-7);
}

// ===========================================================================
// PROBE 4 — degenerate/repeated diagonalizable eigenvalues diag(2,2,5).
// torch.linalg.eig (f64): {(2,0),(2,0),(5,0)}.
// ===========================================================================
#[test]
fn eig_f64_dev_repeated_eigvals_diag_match_torch() {
    let a_row = vec![2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 5.0];
    let n = 3;
    let (w, vr) = run_f64(&a_row, n);
    assert_eigvals_set(&w, n, &[(2.0, 0.0), (2.0, 0.0), (5.0, 0.0)], 1e-9);
    check_eigenpairs(&a_row, &w, &vr, n, 1e-9);
}

// ===========================================================================
// PROBE 5 — defective / non-diagonalizable [[1,1],[0,1]], repeated eigval 1
// with a single eigenvector (geometric multiplicity 1). torch.linalg.eig:
//   eigvals {(1,0),(1,0)}. torch returns the genuine eigenvector(s); the
//   Av=lv residual on the RETURNED vectors must hold. Pin: eigenvalues match
//   torch, and whatever vectors are returned must satisfy Av=lv (a defective
//   matrix should not crash or emit NaN/garbage that breaks the residual).
// ===========================================================================
#[test]
fn eig_f64_dev_defective_2x2_match_torch() {
    let a_row = vec![1.0, 1.0, 0.0, 1.0];
    let n = 2;
    let (w, vr) = run_f64(&a_row, n);
    assert_eigvals_set(&w, n, &[(1.0, 0.0), (1.0, 0.0)], 1e-7);
    // Returned eigenvectors must be finite and satisfy Av = lv.
    for &x in &vr {
        assert!(x.is_finite(), "VR contains non-finite value: {x}");
    }
    check_eigenpairs(&a_row, &w, &vr, n, 1e-7);
}

// ===========================================================================
// PROBE 6 — n=1 trivial: [[7]] -> eigval 7, eigvec [1].
// torch.linalg.eig([[7.0]]) = {(7,0)}.
// ===========================================================================
#[test]
fn eig_f64_dev_n1_scalar_match_torch() {
    let a_row = vec![7.0];
    let n = 1;
    let (w, vr) = run_f64(&a_row, n);
    assert_eigvals_set(&w, n, &[(7.0, 0.0)], 1e-12);
    check_eigenpairs(&a_row, &w, &vr, n, 1e-12);
}

// ===========================================================================
// PROBE 7 — n=0 empty path must return empty W and VR.
// ===========================================================================
#[test]
fn eig_f64_dev_n0_empty() {
    ensure_init();
    let dev = GpuDevice::new(0).expect("device");
    let a_row: Vec<f64> = vec![];
    let d_a = cpu_to_gpu(&a_row, &dev).expect("upload A");
    let (d_w, d_vr) = gpu_eig_f64_dev(&d_a, 0, &dev).expect("gpu_eig_f64_dev n=0");
    let w = gpu_to_cpu(&d_w, &dev).expect("W");
    let vr = gpu_to_cpu(&d_vr, &dev).expect("VR");
    assert_eq!(w.len(), 0, "n=0 W must be empty");
    assert_eq!(vr.len(), 0, "n=0 VR must be empty");
}

// ===========================================================================
// PROBE 8 — f32 path on the 8x8 random matrix (multiset, generous tol).
// ===========================================================================
#[test]
fn eig_f32_dev_rand_8x8_eigvals_match_torch() {
    ensure_init();
    let dev = GpuDevice::new(0).expect("device");
    let a_row: Vec<f32> = vec![
        -0.9442729465697248,
        2.1088314558929464,
        1.0883842432738466,
        0.3996407777515308,
        0.5747094125731214,
        -0.8877392149005505,
        0.08327009207878158,
        -0.8734896557467873,
        -0.16960265491641555,
        0.1685789813154307,
        0.4021887171478154,
        0.5912459271083189,
        -0.9539979204593794,
        -1.9696565671285429,
        -0.5643267420460211,
        1.2033349696299804,
        1.310959994430523,
        0.606963513920415,
        0.11296840203937493,
        -0.019701610309299847,
        0.7272387248252795,
        -1.0601602914405572,
        -0.7301679384338858,
        -0.05434584252726653,
        0.39364203620568217,
        0.41712264701160506,
        1.2364540003268585,
        -1.464719847571328,
        -1.8875063742605234,
        0.9997484692397808,
        -0.724448170641284,
        1.4746544833968724,
        1.6240308302956872,
        1.1993394860254922,
        0.8564163399506219,
        -0.7594775712868722,
        -2.4488563349552255,
        1.891219961633063,
        -0.12874337044649875,
        -0.5023774038510367,
        -0.785846636418831,
        0.9511176453858454,
        -0.5598018763019493,
        -1.0730471611497474,
        0.13111997036085302,
        0.7924334423409372,
        -0.7803413864092041,
        0.5055361236487617,
        1.2715567385450148,
        -0.4106576691455445,
        -0.26777109869501675,
        1.130909158050243,
        -1.1281994720335422,
        -0.464825487755811,
        0.5191529814618616,
        -1.9675839104268127,
        -0.9172197415914208,
        -0.8299182573178825,
        1.024755776862596,
        0.5156683115481856,
        1.2011941626901288,
        -0.6240451596490983,
        -0.3788991195207488,
        -0.3360117533227261,
    ]
    .iter()
    .map(|&x: &f64| x as f32)
    .collect();
    let n = 8;
    let d_a = cpu_to_gpu(&a_row, &dev).expect("upload A");
    let (d_w, d_vr) = gpu_eig_f32_dev(&d_a, n, &dev).expect("gpu_eig_f32_dev must run");
    let w: Vec<f64> = gpu_to_cpu(&d_w, &dev)
        .expect("W")
        .iter()
        .map(|&x| x as f64)
        .collect();
    let vr: Vec<f64> = gpu_to_cpu(&d_vr, &dev)
        .expect("VR")
        .iter()
        .map(|&x| x as f64)
        .collect();
    let a64: Vec<f64> = a_row.iter().map(|&x| x as f64).collect();
    assert_eigvals_set(
        &w,
        n,
        &[
            (-3.702216, 0.0),
            (-1.756766, 0.0),
            (-1.01131, -0.213558),
            (-1.01131, 0.213558),
            (0.084655, -2.314675),
            (0.084655, 2.314675),
            (1.343042, 0.0),
            (2.368522, 0.0),
        ],
        2e-3,
    );
    check_eigenpairs(&a64, &w, &vr, n, 1e-3);
}

// ===========================================================================
// PROBE 9 — gpu_real_to_complex_f32 exactness at an ODD, large size.
// out[2i] = real[i], out[2i+1] = 0 EXACTLY. (#1685 was an f64 stride bug in a
// sibling kernel; pin this kernel's interleave byte-for-byte.)
// ===========================================================================
#[test]
fn real_to_complex_f32_exact_odd_large() {
    ensure_init();
    let dev = GpuDevice::new(0).expect("device");
    let len = 1031usize; // odd, > 1024 (spans >1 block)
    let input: Vec<f32> = (0..len).map(|i| (i as f32) * 0.5 - 3.0).collect();
    let d_in = cpu_to_gpu(&input, &dev).expect("upload");
    let d_out = gpu_real_to_complex_f32(&d_in, len, &dev).expect("promote");
    let out = gpu_to_cpu(&d_out, &dev).expect("download");
    assert_eq!(out.len(), 2 * len);
    for i in 0..len {
        assert_eq!(out[2 * i], input[i], "real mismatch at {i}");
        assert_eq!(out[2 * i + 1], 0.0f32, "imag not zero at {i}");
    }
}

// ===========================================================================
// PROBE 10 — gpu_real_to_complex_f64 exactness at an ODD, large size.
// Guards the #1685 f64-stride class of bug directly on the new f64 kernel.
// ===========================================================================
#[test]
fn real_to_complex_f64_exact_odd_large() {
    ensure_init();
    let dev = GpuDevice::new(0).expect("device");
    let len = 1031usize;
    let input: Vec<f64> = (0..len).map(|i| (i as f64) * 0.5 - 3.0).collect();
    let d_in = cpu_to_gpu(&input, &dev).expect("upload");
    let d_out = gpu_real_to_complex_f64(&d_in, len, &dev).expect("promote");
    let out = gpu_to_cpu(&d_out, &dev).expect("download");
    assert_eq!(out.len(), 2 * len);
    for i in 0..len {
        assert_eq!(out[2 * i], input[i], "real mismatch at {i}");
        assert_eq!(out[2 * i + 1], 0.0f64, "imag not zero at {i}");
    }
}
