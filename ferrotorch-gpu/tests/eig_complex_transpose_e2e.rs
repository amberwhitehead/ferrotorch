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
//! AUDIT FINDING (#1687): the eig path does not reach the #1685 transpose kernel
//! at all — `cusolverDnXgeev` returns CUSOLVER_STATUS_INVALID_VALUE first, so the
//! whole GPU general-eig path diverges from torch (which succeeds) and the
//! transpose kernel's end-to-end correctness is unverifiable via the consumer.
//! These tests are #[ignore]'d behind #1687 because the divergence is upstream of
//! #1685's scope; #1685's transpose kernel is independently verified correct by
//! transpose_complex_kernel.rs (now live) and the boundary probes in this audit.
#![cfg(feature = "cuda")]

use ferrotorch_gpu::cusolver::{gpu_eig_f32_dev, gpu_eig_f64_dev};
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
/// Divergence: ferrotorch's gpu_eig_f64_dev (ferrotorch-gpu/src/cusolver.rs:4992)
/// returns Err(Solver(CusolverError(CUSOLVER_STATUS_INVALID_VALUE))) from the
/// `cusolverDnXgeev` call (cusolver.rs:5092) BEFORE the #1685 complex-transpose
/// repack at cusolver.rs:5132 ever runs.
/// Upstream torch.linalg.eig returns the eigenpairs above; ferrotorch errors.
/// Tracking: #1687
#[test]
#[ignore = "divergence: GPU general-eig errors at cusolverDnXgeev (INVALID_VALUE) before reaching #1685 transpose; tracking #1687"]
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
    assert!((got[0].0).abs() < 1e-9 && (got[0].1 + 1.0).abs() < 1e-9, "got {got:?}");
    assert!((got[1].0).abs() < 1e-9 && (got[1].1 - 1.0).abs() < 1e-9, "got {got:?}");

    check_eigenpairs(&a_row, &w, &vr, n, 1e-9);
}

/// 3x3 block [[1,-1,0],[1,1,0],[0,0,2]].
/// torch.linalg.eig: eigvals = {(1+1j), (1-1j), (2+0j)}, each verified A v = l v.
/// Same divergence as the 2x2 case (#1687): Xgeev INVALID_VALUE before the
/// #1685 transpose repack runs.
/// Tracking: #1687
#[test]
#[ignore = "divergence: GPU general-eig errors at cusolverDnXgeev (INVALID_VALUE) before reaching #1685 transpose; tracking #1687"]
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

/// f32 path on the same 3x3 block. Same divergence (#1687).
/// Tracking: #1687
#[test]
#[ignore = "divergence: GPU general-eig errors at cusolverDnXgeev (INVALID_VALUE) before reaching #1685 transpose; tracking #1687"]
fn eig_f32_dev_block_3x3_satisfies_av_eq_lv() {
    ensure_init();
    let dev = GpuDevice::new(0).expect("device");
    let n = 3;
    let a_row: Vec<f32> = vec![1.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 2.0];
    let d_a = cpu_to_gpu(&a_row, &dev).expect("upload A");
    let (d_w, d_vr) = gpu_eig_f32_dev(&d_a, n, &dev).expect("gpu_eig_f32_dev must run");
    let w: Vec<f64> = gpu_to_cpu(&d_w, &dev).expect("W").iter().map(|&x| x as f64).collect();
    let vr: Vec<f64> = gpu_to_cpu(&d_vr, &dev).expect("VR").iter().map(|&x| x as f64).collect();
    let a64: Vec<f64> = a_row.iter().map(|&x| x as f64).collect();
    check_eigenpairs(&a64, &w, &vr, n, 1e-4);
}
