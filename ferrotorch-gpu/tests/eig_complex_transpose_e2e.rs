//! End-to-end semantic verification of the GPU non-symmetric eig path
//! (`gpu_eig_f32_dev` / `gpu_eig_f64_dev` in cusolver.rs), whose VR eigenvector
//! col-major -> row-major repack is performed by `gpu_transpose_complex_f32/f64`
//! (kernels.rs), the kernel fixed in #1685.
//!
//! WHY THIS EXISTS: the builder's own test (transpose_complex_kernel.rs) asserts
//! `out[k] == in[(k%n)*n + (k/n)]`, a formula DERIVED FROM the kernel's own index
//! math — it can only confirm the kernel matches itself, not that the repack
//! produces the layout `torch.linalg.eig` expects. This test asserts the
//! semantic ground truth: the eigenvector COLUMNS returned THROUGH THE KERNEL by
//! the real consumer (the eig path) must satisfy `A v = lambda v`, with the
//! eigenvalues independently pinned from `torch.linalg.eig`.
//!
//! RESOLUTION (#1687): the eig path's `cusolverDnXgeev` call returned
//! CUSOLVER_STATUS_INVALID_VALUE because it passed `dataTypeA = CUDA_R_32F` /
//! `computeType = CUDA_R_32F` with a real A buffer. Xgeev requires a
//! HOMOGENEOUS datatype set — torch.linalg.eig always promotes the real input
//! to a complex buffer (imag = 0) and passes all-CUDA_C_32F/64F (verified vs
//! `aten/src/ATen/native/cuda/linalg/CUDASolver.cpp:1865-1931`). The fix
//! promotes the real col-major A to complex col-major via
//! `gpu_real_to_complex_f32/f64` (kernels.rs) and sets every datatype to
//! CUDA_C_*. With the path now functional, these tests verify the #1685
//! transpose repack end-to-end via its real consumer: the eigenvector COLUMNS
//! returned THROUGH THE KERNEL must satisfy `A v = lambda v`, eigenvalues
//! independently pinned from `torch.linalg.eig`.
#![cfg(feature = "cuda")]

use ferrotorch_gpu::cusolver::{gpu_eig_f32, gpu_eig_f32_dev, gpu_eig_f64, gpu_eig_f64_dev};
use ferrotorch_gpu::transfer::{cpu_to_gpu, gpu_to_cpu};
use ferrotorch_gpu::{GpuDevice, init_cuda_backend};

fn ensure_init() {
    if !ferrotorch_core::gpu_dispatch::has_gpu_backend() {
        init_cuda_backend().expect("init_cuda_backend");
    }
}

/// Verify A v_j = lambda_j v_j for every column j of the row-major
/// interleaved-complex VR buffer (length 2*n*n, logical [n, n, 2]) returned by
/// the eig path, where W (length 2n, [n, 2]) holds the eigenvalues.
/// `a_row` is the real n x n row-major input matrix.
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
                "A v = lambda v violated: col {j} row {r}: err={err:.3e}"
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

/// 2x2 rotation [[0,-1],[1,0]].
/// torch.linalg.eig (torch 2.11.0+cu130, RTX 3090): eigvals = {1j, -1j};
/// eigvecs cols = [[0.7071, 0.7071], [-0.7071j, 0.7071j]], each verified
/// max|Av-lv| = 0.
///
/// Fixed in #1687: with complex A promotion + all-CUDA_C_64F datatypes,
/// gpu_eig_f64_dev now matches torch.linalg.eig and reaches the #1685
/// complex-transpose repack, so A v = lambda v holds.
#[test]
fn eig_f64_dev_rotation_2x2_satisfies_av_eq_lv() {
    ensure_init();
    let dev = GpuDevice::new(0).expect("device");
    let n = 2;
    let a_row = vec![0.0, -1.0, 1.0, 0.0];
    let d_a = cpu_to_gpu(&a_row, &dev).expect("upload A");
    let (d_w, d_vr) = gpu_eig_f64_dev(&d_a, n, &dev).expect("gpu_eig_f64_dev must run");
    let w = gpu_to_cpu(&d_w, &dev).expect("download W");
    let vr = gpu_to_cpu(&d_vr, &dev).expect("download VR");

    // eigenvalues are +-i (torch ground truth); order-independent set check
    let mut got: Vec<(f64, f64)> = (0..n).map(|j| (w[2 * j], w[2 * j + 1])).collect();
    got.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    assert!(
        (got[0].0).abs() < 1e-9 && (got[0].1 + 1.0).abs() < 1e-9,
        "got {got:?}"
    );
    assert!(
        (got[1].0).abs() < 1e-9 && (got[1].1 - 1.0).abs() < 1e-9,
        "got {got:?}"
    );

    check_eigenpairs(&a_row, &w, &vr, n, 1e-9);
}

/// 3x3 block [[1,-1,0],[1,1,0],[0,0,2]].
/// torch.linalg.eig: eigvals = {(1+1j), (1-1j), (2+0j)}, each verified A v = l v.
/// Fixed in #1687 (complex A promotion + all-CUDA_C_64F datatypes).
#[test]
fn eig_f64_dev_block_3x3_satisfies_av_eq_lv() {
    ensure_init();
    let dev = GpuDevice::new(0).expect("device");
    let n = 3;
    let a_row = vec![1.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 2.0];
    let d_a = cpu_to_gpu(&a_row, &dev).expect("upload A");
    let (d_w, d_vr) = gpu_eig_f64_dev(&d_a, n, &dev).expect("gpu_eig_f64_dev must run");
    let w = gpu_to_cpu(&d_w, &dev).expect("download W");
    let vr = gpu_to_cpu(&d_vr, &dev).expect("download VR");
    check_eigenpairs(&a_row, &w, &vr, n, 1e-9);
}

/// f32 path on the same 3x3 block. Fixed in #1687 (complex A promotion +
/// all-CUDA_C_32F datatypes).
#[test]
fn eig_f32_dev_block_3x3_satisfies_av_eq_lv() {
    ensure_init();
    let dev = GpuDevice::new(0).expect("device");
    let n = 3;
    let a_row: Vec<f32> = vec![1.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 2.0];
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
    check_eigenpairs(&a64, &w, &vr, n, 1e-4);
}

/// Assert the eigenvalue multiset matches `expected` order-independently.
/// Eigenvalue order is implementation-defined (cuSOLVER vs torch may differ),
/// and near-degenerate real parts make a positional (re, im) sort fragile, so
/// match each computed eigenvalue to a not-yet-claimed expected one within
/// `tol` (greedy nearest, complex distance).
fn assert_eigvals_set(w: &[f64], n: usize, expected: &[(f64, f64)], tol: f64) {
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
                "eigenvalue {g:?} has no match within {tol:e} in expected {expected:?} (got set {got:?})"
            ),
        }
    }
}

/// HOST-BOUNCE variant (`gpu_eig_f64`, the production consumer at
/// backend_impl.rs:5324). Same 2x2 rotation; torch eigvals {+-i}.
/// Confirms the complex-A datatype fix applies to the host-bounce path too.
#[test]
fn eig_f64_host_rotation_2x2_satisfies_av_eq_lv() {
    ensure_init();
    let dev = GpuDevice::new(0).expect("device");
    let n = 2;
    let a_row = vec![0.0, -1.0, 1.0, 0.0];
    let d_a = cpu_to_gpu(&a_row, &dev).expect("upload A");
    let (d_w, d_vr) = gpu_eig_f64(&d_a, n, &dev).expect("gpu_eig_f64 must run");
    let w = gpu_to_cpu(&d_w, &dev).expect("download W");
    let vr = gpu_to_cpu(&d_vr, &dev).expect("download VR");
    // torch.linalg.eig([[0,-1],[1,0]]) eigvals = {(0,-1),(0,1)}.
    assert_eigvals_set(&w, n, &[(0.0, -1.0), (0.0, 1.0)], 1e-9);
    check_eigenpairs(&a_row, &w, &vr, n, 1e-9);
}

/// HOST-BOUNCE variant (`gpu_eig_f32`, production consumer at
/// backend_impl.rs:5312) on the 3x3 block. torch eigvals {(1,1),(1,-1),(2,0)}.
#[test]
fn eig_f32_host_block_3x3_satisfies_av_eq_lv() {
    ensure_init();
    let dev = GpuDevice::new(0).expect("device");
    let n = 3;
    let a_row: Vec<f32> = vec![1.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 2.0];
    let d_a = cpu_to_gpu(&a_row, &dev).expect("upload A");
    let (d_w, d_vr) = gpu_eig_f32(&d_a, n, &dev).expect("gpu_eig_f32 must run");
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
    assert_eigvals_set(&w, n, &[(1.0, 1.0), (1.0, -1.0), (2.0, 0.0)], 1e-4);
    check_eigenpairs(&a64, &w, &vr, n, 1e-4);
}

/// REAL-SPECTRUM regression: a diagonal matrix has purely real eigenvalues.
/// torch.linalg.eig(diag(3,5,-2)) = {(3,0),(5,0),(-2,0)} (imag parts exactly 0).
/// Confirms the real->complex promotion (imag = 0) did not corrupt the
/// real-eigenvalue case. (#1687)
#[test]
fn eig_f64_dev_diag_real_spectrum() {
    ensure_init();
    let dev = GpuDevice::new(0).expect("device");
    let n = 3;
    let a_row = vec![3.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, -2.0];
    let d_a = cpu_to_gpu(&a_row, &dev).expect("upload A");
    let (d_w, d_vr) = gpu_eig_f64_dev(&d_a, n, &dev).expect("gpu_eig_f64_dev must run");
    let w = gpu_to_cpu(&d_w, &dev).expect("download W");
    let vr = gpu_to_cpu(&d_vr, &dev).expect("download VR");
    assert_eigvals_set(&w, n, &[(3.0, 0.0), (5.0, 0.0), (-2.0, 0.0)], 1e-9);
    // Imaginary parts must be exactly the zero we promoted in.
    for j in 0..n {
        assert!(w[2 * j + 1].abs() < 1e-12, "imag of eigval {j} nonzero");
    }
    check_eigenpairs(&a_row, &w, &vr, n, 1e-9);
}

/// REAL eigenvalues with ASYMMETRIC storage: a triangular matrix
/// [[2,7],[0,3]] is non-symmetric but has real eigenvalues {2,3}.
/// torch.linalg.eig gives eigvals {(2,0),(3,0)}. Confirms the column-major
/// promotion + repack handles non-symmetric real-spectrum storage. (#1687)
#[test]
fn eig_f64_dev_triangular_asymmetric_real_spectrum() {
    ensure_init();
    let dev = GpuDevice::new(0).expect("device");
    let n = 2;
    let a_row = vec![2.0, 7.0, 0.0, 3.0];
    let d_a = cpu_to_gpu(&a_row, &dev).expect("upload A");
    let (d_w, d_vr) = gpu_eig_f64_dev(&d_a, n, &dev).expect("gpu_eig_f64_dev must run");
    let w = gpu_to_cpu(&d_w, &dev).expect("download W");
    let vr = gpu_to_cpu(&d_vr, &dev).expect("download VR");
    assert_eigvals_set(&w, n, &[(2.0, 0.0), (3.0, 0.0)], 1e-9);
    check_eigenpairs(&a_row, &w, &vr, n, 1e-9);
}
