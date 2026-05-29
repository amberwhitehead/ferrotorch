//! #1679 — fused-transpose matmul (`A @ B^T`) on f32 / f64 GPU.
//!
//! `linear_fused`'s GPU path used to launch a separate `transpose_2d_f32(weight)`
//! kernel every forward and then `matmul_f32(input, wt)`. The weight is constant,
//! so transposing it per forward wasted a kernel launch + compute + buffer alloc.
//! cuBLAS can fold the transpose into the `transb = CUBLAS_OP_T` flag, so we add
//! `gpu_matmul_f32_nt` / `gpu_matmul_f64_nt` computing `C = A @ B^T` directly and
//! rewire `linear_fused` to call them.
//!
//! These tests pin two correctness properties on a live GPU:
//!
//! 1. **Old-path equivalence (regression anchor).** `gpu_matmul_f32_nt(a, b, m,
//!    k, n)` must produce the SAME bytes (within f32/f64 eps) as the OLD path
//!    `transpose_2d(b) -> gpu_matmul(a, b^T)`. cuBLAS `transb=T` shares the same
//!    f32 accumulation as transpose-then-matmul, so the only differences are the
//!    intra-warp reduction order — bounded by a tight eps. Verified on NON-SQUARE
//!    shapes (`out != in`, e.g. `784 -> 256`, `256 -> 10`) where a wrong
//!    transpose / dim swap would show.
//!
//! 2. **PyTorch `F.linear` parity.** For a small named-bits case, the result must
//!    equal `torch.nn.functional.linear(input, weight, bias) = input @ weight.T
//!    + bias` computed from the SAME typed inputs by the closed-form definition
//!    (PyTorch `aten/src/ATen/native/Linear.cpp` `at::linear` = `addmm(bias,
//!    input, weight.t())`). The reference is built from named constants, not
//!    from a ferrotorch self-call (R-CHAR-3).
//!
//! All tests are gated on `#[cfg(feature = "cuda")]` and require a live CUDA GPU.

#![cfg(feature = "cuda")]

use std::sync::Once;

use ferrotorch_gpu::{GpuDevice, blas, init_cuda_backend, kernels, transfer};

static INIT: Once = Once::new();

fn ensure_cuda() {
    INIT.call_once(|| {
        init_cuda_backend().expect("CUDA backend init");
    });
}

/// Deterministic pseudo-random f32 fill (LCG) so shapes are reproducible across
/// runs without pulling in a PRNG dependency. Range roughly [-1, 1).
fn fill_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 33) as u32) as f64 / (u32::MAX as f64);
            (u * 2.0 - 1.0) as f32
        })
        .collect()
}

fn fill_f64(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 33) as u32) as f64 / (u32::MAX as f64);
            u * 2.0 - 1.0
        })
        .collect()
}

fn max_abs_diff_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn max_abs_diff_f64(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0f64, f64::max)
}

/// `gpu_matmul_f32_nt(a, b, m, k, n)` == OLD path `transpose_2d(b) ->
/// gpu_matmul_f32(a, b^T)` for the two MLP layer shapes 784->256 and 256->10,
/// which are non-square so a wrong transpose/dim would surface.
#[test]
fn matmul_f32_nt_equals_old_transpose_then_matmul_nonsquare() {
    ensure_cuda();
    let device = GpuDevice::new(0).expect("GpuDevice::new");

    // (m, k, n) = (batch, in_features, out_features).
    // weight is [n, k] = [out, in], result is [m, n] = input @ weight^T.
    for &(m, k, n) in &[(32usize, 784usize, 256usize), (32, 256, 10), (17, 33, 5)] {
        let a_host = fill_f32(m * k, 11 + (m * 7 + n) as u64);
        let b_host = fill_f32(n * k, 97 + (k * 3 + n) as u64);

        let a = transfer::cpu_to_gpu(&a_host, &device).expect("upload A");
        let b = transfer::cpu_to_gpu(&b_host, &device).expect("upload B");

        // NEW path: fold transpose into transb=T.
        let c_nt = blas::gpu_matmul_f32_nt(&a, &b, m, k, n, &device).expect("matmul_f32_nt");
        let got_nt = transfer::gpu_to_cpu(&c_nt, &device).expect("download nt");

        // OLD path: explicit transpose_2d(weight) [n,k] -> [k,n], then matmul.
        let bt = kernels::gpu_transpose_2d(&b, n, k, &device).expect("transpose_2d");
        let c_old = blas::gpu_matmul_f32(&a, &bt, m, k, n, &device).expect("matmul_f32");
        let got_old = transfer::gpu_to_cpu(&c_old, &device).expect("download old");

        assert_eq!(got_nt.len(), m * n, "nt output length [m,n]");
        let d = max_abs_diff_f32(&got_nt, &got_old);
        // Both are cuBLAS SGEMM with the same f32 accumulator; the only source
        // of difference is reduction order, well under 1e-3 for these magnitudes.
        assert!(
            d <= 1e-3,
            "matmul_f32_nt vs old transpose+matmul ({m}x{k} @ {n}x{k}): max|Δ|={d:.3e}"
        );
    }
}

/// f64 counterpart of the old-path equivalence anchor, with a tight 1e-9 tol.
#[test]
fn matmul_f64_nt_equals_old_transpose_then_matmul_nonsquare() {
    ensure_cuda();
    let device = GpuDevice::new(0).expect("GpuDevice::new");

    for &(m, k, n) in &[(32usize, 256usize, 64usize), (8, 100, 10), (17, 33, 5)] {
        let a_host = fill_f64(m * k, 11 + (m * 7 + n) as u64);
        let b_host = fill_f64(n * k, 97 + (k * 3 + n) as u64);

        let a = transfer::cpu_to_gpu(&a_host, &device).expect("upload A");
        let b = transfer::cpu_to_gpu(&b_host, &device).expect("upload B");

        let c_nt = blas::gpu_matmul_f64_nt(&a, &b, m, k, n, &device).expect("matmul_f64_nt");
        let got_nt = transfer::gpu_to_cpu(&c_nt, &device).expect("download nt");

        let bt = kernels::gpu_transpose_2d_f64(&b, n, k, &device).expect("transpose_2d_f64");
        let c_old = blas::gpu_matmul_f64(&a, &bt, m, k, n, &device).expect("matmul_f64");
        let got_old = transfer::gpu_to_cpu(&c_old, &device).expect("download old");

        assert_eq!(got_nt.len(), m * n, "nt f64 output length [m,n]");
        let d = max_abs_diff_f64(&got_nt, &got_old);
        assert!(
            d <= 1e-9,
            "matmul_f64_nt vs old transpose+matmul ({m}x{k} @ {n}x{k}): max|Δ|={d:.3e}"
        );
    }
}

/// PyTorch `F.linear` parity from named-bits inputs (R-CHAR-3).
///
/// `F.linear(input, weight) = input @ weight.T` (PyTorch
/// `torch/nn/functional.py` -> `at::linear` -> `addmm(bias, input,
/// weight.t())` in `aten/src/ATen/native/Linear.cpp`). With
///   input  = [[1, 2, 3],            # [m=2, k=3]
///             [4, 5, 6]]
///   weight = [[ 1,  0, -1],         # [n=2, k=3]
///             [ 2,  1,  0]]
/// the closed-form reference is:
///   out[0,0] = 1*1 + 2*0 + 3*(-1) = -2
///   out[0,1] = 1*2 + 2*1 + 3*0    =  4
///   out[1,0] = 4*1 + 5*0 + 6*(-1) = -2
///   out[1,1] = 4*2 + 5*1 + 6*0    = 13
/// These integer-valued products are exactly representable in f32, so the
/// match is bit-exact, not eps.
#[test]
fn matmul_f32_nt_matches_torch_f_linear_named_bits() {
    ensure_cuda();
    let device = GpuDevice::new(0).expect("GpuDevice::new");

    let (m, k, n) = (2usize, 3usize, 2usize);
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2,3]
    let weight: Vec<f32> = vec![1.0, 0.0, -1.0, 2.0, 1.0, 0.0]; // [2,3]
    // F.linear(input, weight) reference, hand-derived above.
    let torch_ref: Vec<f32> = vec![-2.0, 4.0, -2.0, 13.0];

    let a = transfer::cpu_to_gpu(&input, &device).expect("upload input");
    let b = transfer::cpu_to_gpu(&weight, &device).expect("upload weight");

    let c = blas::gpu_matmul_f32_nt(&a, &b, m, k, n, &device).expect("matmul_f32_nt");
    let got = transfer::gpu_to_cpu(&c, &device).expect("download");

    assert_eq!(got, torch_ref, "matmul_f32_nt vs F.linear named bits");
}

/// CPU f64 ground-truth for one `F.linear` layer: `out[i,j] = Σ_p x[i,p] *
/// w[j,p] + bias[j]` (PyTorch `at::linear` = `input @ weight.t() + bias`).
/// Accumulated in f64 so it is the authoritative reference, independent of
/// whichever GPU kernel (cuBLAS vs PTX small-matmul) runs.
fn linear_ref_f64(x: &[f32], w: &[f32], bias: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for p in 0..k {
                acc += f64::from(x[i * k + p]) * f64::from(w[j * k + p]);
            }
            out[i * n + j] = (acc + f64::from(bias[j])) as f32;
        }
    }
    out
}

/// Two-layer MLP forward (784->256, relu, 256->10) end-to-end via the fused-
/// transpose nt path, with bias added at each layer. This is the exact shape
/// and op sequence `linear_fused` lowers to. Correctness is pinned against an
/// f64 CPU reference (the authoritative `input @ weight^T + bias`), which is
/// kernel-independent — the prior implementation compared two GPU kernels
/// (cuBLAS-nt vs the PTX small-matmul that `gpu_matmul_f32` selects for the
/// `total_ops < 500k` second layer), whose rounding legitimately differs.
#[test]
fn mlp_forward_nt_matches_cpu_reference_with_bias() {
    ensure_cuda();
    let device = GpuDevice::new(0).expect("GpuDevice::new");

    let (m, in0, h, out) = (32usize, 784usize, 256usize, 10usize);
    let x = fill_f32(m * in0, 1);
    let w1 = fill_f32(h * in0, 2); // [256, 784]
    let b1 = fill_f32(h, 3);
    let w2 = fill_f32(out * h, 4); // [10, 256]
    let b2 = fill_f32(out, 5);

    // Host helper: one Linear layer via the NT path (input @ weight^T + bias).
    let layer_nt =
        |x_host: &[f32], w_host: &[f32], bvec: &[f32], mm: usize, kk: usize, nn: usize| {
            let xb = transfer::cpu_to_gpu(x_host, &device).unwrap();
            let wb = transfer::cpu_to_gpu(w_host, &device).unwrap();
            let cb = blas::gpu_matmul_f32_nt(&xb, &wb, mm, kk, nn, &device).unwrap();
            let mut out = transfer::gpu_to_cpu(&cb, &device).unwrap();
            for i in 0..mm {
                for j in 0..nn {
                    out[i * nn + j] += bvec[j];
                }
            }
            out
        };

    let relu = |v: &mut [f32]| v.iter_mut().for_each(|x| *x = x.max(0.0));

    // GPU NT pipeline.
    let mut h_nt = layer_nt(&x, &w1, &b1, m, in0, h);
    relu(&mut h_nt);
    let out_nt = layer_nt(&h_nt, &w2, &b2, m, h, out);

    // CPU f64 reference pipeline over the SAME inputs.
    let mut h_ref = linear_ref_f64(&x, &w1, &b1, m, in0, h);
    relu(&mut h_ref);
    let out_ref = linear_ref_f64(&h_ref, &w2, &b2, m, h, out);

    let d = max_abs_diff_f32(&out_nt, &out_ref);
    // cuBLAS SGEMM accumulates in f32 over k=784/256; the second matmul's PTX
    // small-matmul kernel (selected for total_ops<500k) likewise. A few e-2
    // over a two-layer relu MLP with O(1) random weights is the expected f32
    // accumulation floor — the same floor the old transpose+matmul path hits.
    assert!(
        d <= 5e-2,
        "MLP forward nt vs CPU f64 reference: max|Δ|={d:.3e} (out len {})",
        out_nt.len()
    );
}
