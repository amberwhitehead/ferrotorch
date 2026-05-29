//! ADVERSARIAL RE-AUDIT of #1679 — `gpu_matmul_f32_nt` / `gpu_matmul_f64_nt`
//! (cuBLAS SGEMM/DGEMM with the weight transpose folded into the cuBLAS
//! transpose flag, computing `C[m,n] = A[m,k] @ B[n,k]^T`).
//!
//! The generator's own test (`divergence_matmul_nt_linear_fused_1679.rs`)
//! anchors `matmul_*_nt` against the OLD `transpose_2d + matmul` GPU path.
//! That is a *self-consistency* check: if both the new fold and the old
//! transpose shared the same column-major dim-swap bug, they would AGREE and
//! the test would pass while the math is wrong. The only true torch-parity
//! check there is a single 2x3x2 named-bits case.
//!
//! This re-audit closes the charter gaps the orchestrator flagged:
//!
//!  - **Asymmetric `m != k != n`, distinct/non-symmetric data, vs an
//!    INDEPENDENT CPU f64 ground truth.** A transposed-dim or wrong-`lda`
//!    bug only surfaces when m, k, n differ AND the operands are not
//!    symmetric. The reference is computed in f64 directly from the closed
//!    form `out[i,j] = Σ_p A[i,p] * B[j,p]` — i.e. `A @ B^T`, the exact shape
//!    PyTorch's `at::linear` lowers to (`aten/src/ATen/native/Linear.cpp:108`
//!    `at::addmm(*bias, input, weight.t())`). This is NOT a ferrotorch
//!    self-call (R-CHAR-3 option (b): symbolic closed form traceable to a
//!    PyTorch file:line).
//!
//!  - **Degenerate dims: m=1 (vector x matrix^T), k=1 (rank-1), n=1
//!    (matrix x vector).** These are exactly where a leading-dimension or
//!    dim-swap bug produces plausible-but-wrong output.
//!
//!  - **`F.linear` with bias** end-to-end (nt + bias broadcast) vs the f64
//!    ground truth, on a non-square shape with distinct weights.
//!
//! If clean, this is a PASSING regression guard. If any case diverges, the
//! `assert!` fails and the generator must fix the nt indexing — a critical
//! bug, since `linear_fused` feeds EVERY GPU Linear forward.

#![cfg(feature = "cuda")]

use std::sync::Once;

use ferrotorch_gpu::{GpuDevice, blas, init_cuda_backend, transfer};

static INIT: Once = Once::new();

fn ensure_cuda() {
    INIT.call_once(|| {
        init_cuda_backend().expect("CUDA backend init");
    });
}

/// Deterministic ASYMMETRIC f32 fill. Unlike all-ones or symmetric data, a
/// distinct value per (row, col) means any dim-swap / wrong-lda permutation of
/// the operand changes the result, so the bug cannot hide. Values span a
/// signed range with non-trivial fractional parts.
fn asym_f32(rows: usize, cols: usize, base: f32) -> Vec<f32> {
    let mut v = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            // Each element a distinct, non-symmetric function of (i, j).
            let x = base + (i as f32) * 0.5 - (j as f32) * 0.25 + (i as f32) * (j as f32) * 0.03125;
            v.push(x);
        }
    }
    v
}

fn asym_f64(rows: usize, cols: usize, base: f64) -> Vec<f64> {
    let mut v = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            let x = base + (i as f64) * 0.5 - (j as f64) * 0.25 + (i as f64) * (j as f64) * 0.03125;
            v.push(x);
        }
    }
    v
}

/// Independent CPU ground truth: `C[m,n] = A[m,k] @ B[n,k]^T`, i.e.
/// `C[i,j] = Σ_p A[i,p] * B[j,p]`. Accumulated in f64 for f32 inputs so it is
/// authoritative regardless of GPU reduction order. This is the closed form of
/// PyTorch `at::linear` without bias (`input @ weight.t()`).
fn matmul_nt_ref_f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for p in 0..k {
                acc += f64::from(a[i * k + p]) * f64::from(b[j * k + p]);
            }
            c[i * n + j] = acc as f32;
        }
    }
    c
}

fn matmul_nt_ref_f64(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for p in 0..k {
                acc += a[i * k + p] * b[j * k + p];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

fn max_rel_diff_f32(got: &[f32], reference: &[f32]) -> f32 {
    assert_eq!(got.len(), reference.len());
    got.iter()
        .zip(reference.iter())
        .map(|(&g, &r)| (g - r).abs() / (r.abs().max(1.0)))
        .fold(0.0f32, f32::max)
}

fn max_rel_diff_f64(got: &[f64], reference: &[f64]) -> f64 {
    assert_eq!(got.len(), reference.len());
    got.iter()
        .zip(reference.iter())
        .map(|(&g, &r)| (g - r).abs() / (r.abs().max(1.0)))
        .fold(0.0f64, f64::max)
}

/// **The key check.** `gpu_matmul_f32_nt` on ASYMMETRIC `m != k != n` shapes
/// with distinct/non-symmetric operand data must equal the INDEPENDENT CPU f64
/// `A @ B^T` ground truth element-wise. A transposed-dim / wrong-lda bug in the
/// column-major fold shows ONLY here (the old-path equivalence test cannot
/// catch a shared-convention bug, and symmetric/square data masks it).
#[test]
fn matmul_f32_nt_asymmetric_nonsquare_vs_cpu_f64_truth() {
    ensure_cuda();
    let device = GpuDevice::new(0).expect("GpuDevice::new");

    // All shapes have m, k, n pairwise distinct.
    for &(m, k, n) in &[
        (3usize, 5usize, 7usize),
        (7, 3, 5),
        (5, 7, 3),
        (32, 784, 256),
        (32, 256, 10),
        (4, 256, 10),
        (17, 33, 5),
    ] {
        let a_host = asym_f32(m, k, 0.3);
        let b_host = asym_f32(n, k, -0.7); // [n, k] = [out, in]

        let a = transfer::cpu_to_gpu(&a_host, &device).expect("upload A");
        let b = transfer::cpu_to_gpu(&b_host, &device).expect("upload B");

        let c = blas::gpu_matmul_f32_nt(&a, &b, m, k, n, &device).expect("matmul_f32_nt");
        let got = transfer::gpu_to_cpu(&c, &device).expect("download");

        assert_eq!(
            got.len(),
            m * n,
            "({m}x{k}) @ ({n}x{k})^T -> [{m},{n}] length"
        );

        let reference = matmul_nt_ref_f32(&a_host, &b_host, m, k, n);
        let rel = max_rel_diff_f32(&got, &reference);
        assert!(
            rel <= 2e-4,
            "matmul_f32_nt diverges from CPU f64 A@B^T truth for asymmetric \
             m={m} k={k} n={n}: max rel|Δ|={rel:.3e}\n got[..min(8)]={:?}\n ref[..min(8)]={:?}",
            &got[..got.len().min(8)],
            &reference[..reference.len().min(8)],
        );
    }
}

/// f64 counterpart of the asymmetric truth check, with a tight tolerance.
#[test]
fn matmul_f64_nt_asymmetric_nonsquare_vs_cpu_f64_truth() {
    ensure_cuda();
    let device = GpuDevice::new(0).expect("GpuDevice::new");

    for &(m, k, n) in &[
        (3usize, 5usize, 7usize),
        (7, 3, 5),
        (5, 7, 3),
        (32, 256, 64),
        (8, 100, 10),
        (17, 33, 5),
    ] {
        let a_host = asym_f64(m, k, 0.3);
        let b_host = asym_f64(n, k, -0.7);

        let a = transfer::cpu_to_gpu(&a_host, &device).expect("upload A");
        let b = transfer::cpu_to_gpu(&b_host, &device).expect("upload B");

        let c = blas::gpu_matmul_f64_nt(&a, &b, m, k, n, &device).expect("matmul_f64_nt");
        let got = transfer::gpu_to_cpu(&c, &device).expect("download");

        assert_eq!(got.len(), m * n);

        let reference = matmul_nt_ref_f64(&a_host, &b_host, m, k, n);
        let rel = max_rel_diff_f64(&got, &reference);
        assert!(
            rel <= 1e-12,
            "matmul_f64_nt diverges from CPU f64 A@B^T truth for asymmetric \
             m={m} k={k} n={n}: max rel|Δ|={rel:.3e}",
        );
    }
}

/// Degenerate dims: m=1 (row vector x B^T), k=1 (rank-1 outer-product-like),
/// n=1 (A x column vector). Each is where a leading-dim / dim-swap bug is most
/// likely to slip a square test yet flip a vector case. f32 and f64.
#[test]
fn matmul_nt_degenerate_dims_vs_cpu_f64_truth() {
    ensure_cuda();
    let device = GpuDevice::new(0).expect("GpuDevice::new");

    for &(m, k, n) in &[
        (1usize, 5usize, 7usize), // m = 1
        (3, 1, 7),                // k = 1
        (3, 5, 1),                // n = 1
        (1, 1, 7),
        (1, 5, 1),
        (3, 1, 1),
        (1, 1, 1),
    ] {
        // f32.
        let a32 = asym_f32(m, k, 0.41);
        let b32 = asym_f32(n, k, -0.59);
        let ga = transfer::cpu_to_gpu(&a32, &device).expect("upA32");
        let gb = transfer::cpu_to_gpu(&b32, &device).expect("upB32");
        let gc = blas::gpu_matmul_f32_nt(&ga, &gb, m, k, n, &device).expect("f32_nt");
        let got32 = transfer::gpu_to_cpu(&gc, &device).expect("dl32");
        let ref32 = matmul_nt_ref_f32(&a32, &b32, m, k, n);
        let rel32 = max_rel_diff_f32(&got32, &ref32);
        assert!(
            rel32 <= 2e-4,
            "matmul_f32_nt degenerate m={m} k={k} n={n}: rel|Δ|={rel32:.3e}, \
             got={got32:?} ref={ref32:?}"
        );

        // f64.
        let a64 = asym_f64(m, k, 0.41);
        let b64 = asym_f64(n, k, -0.59);
        let ga6 = transfer::cpu_to_gpu(&a64, &device).expect("upA64");
        let gb6 = transfer::cpu_to_gpu(&b64, &device).expect("upB64");
        let gc6 = blas::gpu_matmul_f64_nt(&ga6, &gb6, m, k, n, &device).expect("f64_nt");
        let got64 = transfer::gpu_to_cpu(&gc6, &device).expect("dl64");
        let ref64 = matmul_nt_ref_f64(&a64, &b64, m, k, n);
        let rel64 = max_rel_diff_f64(&got64, &ref64);
        assert!(
            rel64 <= 1e-12,
            "matmul_f64_nt degenerate m={m} k={k} n={n}: rel|Δ|={rel64:.3e}, \
             got={got64:?} ref={ref64:?}"
        );
    }
}

/// End-to-end `F.linear(input, weight, bias) = input @ weight^T + bias`
/// (PyTorch `aten/src/ATen/native/Linear.cpp:108`) via the nt path + an
/// explicit bias broadcast (exactly what `linear_fused` does after the matmul),
/// vs the f64 ground truth. Non-square (in != out), distinct asymmetric
/// weights, WITH and WITHOUT bias.
#[test]
fn f_linear_with_and_without_bias_nt_vs_cpu_f64_truth() {
    ensure_cuda();
    let device = GpuDevice::new(0).expect("GpuDevice::new");

    // (batch m, in_features k, out_features n).
    for &(m, k, n) in &[(32usize, 784usize, 256usize), (4, 256, 10), (5, 7, 3)] {
        let x = asym_f32(m, k, 0.2); // input [m, k]
        let w = asym_f32(n, k, -0.45); // weight [out, in] = [n, k]
        let bias: Vec<f32> = (0..n).map(|j| 0.1 * (j as f32) - 0.3).collect();

        let xb = transfer::cpu_to_gpu(&x, &device).expect("up x");
        let wb = transfer::cpu_to_gpu(&w, &device).expect("up w");
        let cb = blas::gpu_matmul_f32_nt(&xb, &wb, m, k, n, &device).expect("nt");
        let mut got = transfer::gpu_to_cpu(&cb, &device).expect("dl");

        // --- without bias ---
        let ref_nobias = matmul_nt_ref_f32(&x, &w, m, k, n);
        let rel_nb = max_rel_diff_f32(&got, &ref_nobias);
        assert!(
            rel_nb <= 2e-4,
            "F.linear (no bias) nt vs f64 truth m={m} k={k} n={n}: rel|Δ|={rel_nb:.3e}"
        );

        // --- with bias broadcast (as linear_fused does) ---
        for i in 0..m {
            for j in 0..n {
                got[i * n + j] += bias[j];
            }
        }
        let mut ref_bias = ref_nobias;
        for i in 0..m {
            for j in 0..n {
                ref_bias[i * n + j] += bias[j];
            }
        }
        let rel_b = max_rel_diff_f32(&got, &ref_bias);
        assert!(
            rel_b <= 2e-4,
            "F.linear (with bias) nt vs f64 truth m={m} k={k} n={n}: rel|Δ|={rel_b:.3e}"
        );
    }
}
