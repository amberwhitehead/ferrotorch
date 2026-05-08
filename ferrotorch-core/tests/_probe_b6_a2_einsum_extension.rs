//! Permanent regression sentinel for #821 + #822 — einsum GPU coverage
//! extensions (Bugfix Batch 6 / Dispatch A2).
//!
//! ## #821 — repeated-index einsum on GPU
//!
//! Pre-fix: `einsum("ii->", a)`, `einsum("ii->i", a)`, `einsum("ii", a)`
//! return `Err(NotImplementedOnCuda { op: "einsum_repeated_index" })` for
//! CUDA inputs (see `einsum.rs:213-217`). PyTorch supports these on CUDA.
//!
//! Post-fix: each pattern produces the correct value on-device via an
//! `as_strided`-based diagonal extraction (shape [N], stride [N+1])
//! routed through the existing `strided_copy_f{32,64}` kernel, then
//! `sum_dim` for the trace case. `is_cuda()` is true.
//!
//! ## #822 — multi-axis / permuted 2-input contractions on GPU
//!
//! Pre-fix: `einsum("ijk,jkl->il", a, b)` returns
//! `Err(NotImplementedOnCuda { op: "einsum_general" })` because the existing
//! decomposition only covers single-axis 2D matmul, 3D bmm, and a small
//! handful of vector / Hadamard / outer patterns.
//!
//! Post-fix: a general permute+reshape+matmul/bmm decomposition handles
//! multi-axis contractions and permuted variants.
//!
//! Tolerance constants mirror `tests/conformance_einops.rs::tolerance`.

#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::Device;
use ferrotorch_core::einsum::einsum;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

const F32_MATMUL_GPU: f32 = 1e-3;
const F64_MATMUL_GPU: f64 = 1e-9;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for the GPU probe suite");
    });
}

fn t_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}
fn t_f64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

fn upload_f32(t: Tensor<f32>) -> Tensor<f32> {
    t.to(Device::Cuda(0)).expect("upload f32")
}
fn upload_f64(t: Tensor<f64>) -> Tensor<f64> {
    t.to(Device::Cuda(0)).expect("upload f64")
}

fn read_f32(t: &Tensor<f32>) -> Vec<f32> {
    let cpu = if t.is_cuda() {
        t.cpu().unwrap()
    } else {
        t.clone()
    };
    cpu.data().unwrap().to_vec()
}
fn read_f64(t: &Tensor<f64>) -> Vec<f64> {
    let cpu = if t.is_cuda() {
        t.cpu().unwrap()
    } else {
        t.clone()
    };
    cpu.data().unwrap().to_vec()
}

fn assert_close_f32(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "{label}: index {i}: {a} vs {e} (diff {})",
            (a - e).abs()
        );
    }
}
fn assert_close_f64(actual: &[f64], expected: &[f64], tol: f64, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "{label}: index {i}: {a} vs {e} (diff {})",
            (a - e).abs()
        );
    }
}

// ---------------------------------------------------------------------------
// #821 — repeated-index einsum (trace, diagonal)
// ---------------------------------------------------------------------------

// Build a deterministic 4x4 matrix with distinct values.
fn mat4x4_f32() -> Vec<f32> {
    (1..=16).map(|i| i as f32).collect()
}
fn mat4x4_f64() -> Vec<f64> {
    (1..=16).map(|i| i as f64).collect()
}

// Diagonal of `mat4x4_*` in row-major (indices 0, 5, 10, 15).
fn diag4_f32() -> Vec<f32> {
    vec![1.0, 6.0, 11.0, 16.0]
}
fn diag4_f64() -> Vec<f64> {
    vec![1.0, 6.0, 11.0, 16.0]
}

#[test]
fn gpu_einsum_trace_2d_f32_works() {
    ensure_cuda_backend();
    let a = upload_f32(t_f32(&mat4x4_f32(), &[4, 4]));
    let r = einsum("ii->", &[&a]).expect("einsum trace f32 on cuda");
    assert!(
        r.is_cuda(),
        "trace f32 result must stay on device, got {:?}",
        r.device()
    );
    let host = read_f32(&r);
    assert_close_f32(
        &host,
        &[1.0 + 6.0 + 11.0 + 16.0],
        F32_MATMUL_GPU,
        "trace f32",
    );
}

#[test]
fn gpu_einsum_trace_2d_f64_works() {
    ensure_cuda_backend();
    let a = upload_f64(t_f64(&mat4x4_f64(), &[4, 4]));
    let r = einsum("ii->", &[&a]).expect("einsum trace f64 on cuda");
    assert!(
        r.is_cuda(),
        "trace f64 result must stay on device, got {:?}",
        r.device()
    );
    let host = read_f64(&r);
    assert_close_f64(
        &host,
        &[1.0 + 6.0 + 11.0 + 16.0],
        F64_MATMUL_GPU,
        "trace f64",
    );
}

#[test]
fn gpu_einsum_trace_implicit_f32_works() {
    // "ii" with no explicit output is equivalent to "ii->" (trace).
    ensure_cuda_backend();
    let a = upload_f32(t_f32(&mat4x4_f32(), &[4, 4]));
    let r = einsum("ii", &[&a]).expect("einsum implicit trace f32");
    assert!(r.is_cuda(), "implicit trace must stay on device");
    let host = read_f32(&r);
    assert_close_f32(&host, &[34.0], F32_MATMUL_GPU, "implicit trace f32");
}

#[test]
fn gpu_einsum_diagonal_2d_f32_works() {
    ensure_cuda_backend();
    let a = upload_f32(t_f32(&mat4x4_f32(), &[4, 4]));
    let r = einsum("ii->i", &[&a]).expect("einsum diagonal f32 on cuda");
    assert!(r.is_cuda(), "diagonal f32 result must stay on device");
    assert_eq!(r.shape(), &[4]);
    let host = read_f32(&r);
    assert_close_f32(&host, &diag4_f32(), F32_MATMUL_GPU, "diagonal f32");
}

#[test]
fn gpu_einsum_diagonal_2d_f64_works() {
    ensure_cuda_backend();
    let a = upload_f64(t_f64(&mat4x4_f64(), &[4, 4]));
    let r = einsum("ii->i", &[&a]).expect("einsum diagonal f64 on cuda");
    assert!(r.is_cuda(), "diagonal f64 result must stay on device");
    assert_eq!(r.shape(), &[4]);
    let host = read_f64(&r);
    assert_close_f64(&host, &diag4_f64(), F64_MATMUL_GPU, "diagonal f64");
}

// ---------------------------------------------------------------------------
// #822 — multi-axis / permuted 2-input contractions on GPU
// ---------------------------------------------------------------------------

// CPU reference: einsum("ijk,jkl->il", a:[2,3,4], b:[3,4,5]) -> [2,5]
fn ref_ijk_jkl_il_f64(
    a: &[f64],
    b: &[f64],
    i_n: usize,
    j_n: usize,
    k_n: usize,
    l_n: usize,
) -> Vec<f64> {
    let mut out = vec![0.0_f64; i_n * l_n];
    for i in 0..i_n {
        for l in 0..l_n {
            let mut acc = 0.0_f64;
            for j in 0..j_n {
                for k in 0..k_n {
                    let a_idx = i * (j_n * k_n) + j * k_n + k;
                    let b_idx = j * (k_n * l_n) + k * l_n + l;
                    acc += a[a_idx] * b[b_idx];
                }
            }
            out[i * l_n + l] = acc;
        }
    }
    out
}

fn ramp_f32(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32) * 0.01 - 1.0).collect()
}
fn ramp_f64(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i as f64) * 0.01 - 1.0).collect()
}

#[test]
fn gpu_einsum_multi_axis_ijk_jkl_il_f32_works() {
    ensure_cuda_backend();
    // a:[2,3,4], b:[3,4,5] -> [2,5], contracting j AND k.
    let a_data = ramp_f32(2 * 3 * 4);
    let b_data = ramp_f32(3 * 4 * 5);
    let a_cpu = t_f32(&a_data, &[2, 3, 4]);
    let b_cpu = t_f32(&b_data, &[3, 4, 5]);
    let a = upload_f32(a_cpu);
    let b = upload_f32(b_cpu);
    let r = einsum("ijk,jkl->il", &[&a, &b]).expect("einsum multi-axis f32 on cuda");
    assert!(r.is_cuda(), "multi-axis f32 result must stay on device");
    assert_eq!(r.shape(), &[2, 5]);

    // Compute reference in f64.
    let a_d: Vec<f64> = a_data.iter().map(|&x| x as f64).collect();
    let b_d: Vec<f64> = b_data.iter().map(|&x| x as f64).collect();
    let expected_d = ref_ijk_jkl_il_f64(&a_d, &b_d, 2, 3, 4, 5);
    let expected: Vec<f32> = expected_d.iter().map(|&x| x as f32).collect();
    let host = read_f32(&r);
    assert_close_f32(&host, &expected, F32_MATMUL_GPU, "ijk,jkl->il f32");
}

#[test]
fn gpu_einsum_multi_axis_ijk_jkl_il_f64_works() {
    ensure_cuda_backend();
    let a_data = ramp_f64(2 * 3 * 4);
    let b_data = ramp_f64(3 * 4 * 5);
    let a = upload_f64(t_f64(&a_data, &[2, 3, 4]));
    let b = upload_f64(t_f64(&b_data, &[3, 4, 5]));
    let r = einsum("ijk,jkl->il", &[&a, &b]).expect("einsum multi-axis f64 on cuda");
    assert!(r.is_cuda(), "multi-axis f64 result must stay on device");
    assert_eq!(r.shape(), &[2, 5]);

    let expected = ref_ijk_jkl_il_f64(&a_data, &b_data, 2, 3, 4, 5);
    let host = read_f64(&r);
    assert_close_f64(&host, &expected, F64_MATMUL_GPU, "ijk,jkl->il f64");
}

#[test]
fn gpu_einsum_batch_multi_axis_bijk_bjkl_bil_f32_works() {
    // Batched multi-axis. a:[B,I,J,K], b:[B,J,K,L] -> [B,I,L].
    ensure_cuda_backend();
    let (b_n, i_n, j_n, k_n, l_n) = (2usize, 2usize, 3usize, 4usize, 5usize);
    let a_data = ramp_f32(b_n * i_n * j_n * k_n);
    let b_data = ramp_f32(b_n * j_n * k_n * l_n);
    let a = upload_f32(t_f32(&a_data, &[b_n, i_n, j_n, k_n]));
    let b = upload_f32(t_f32(&b_data, &[b_n, j_n, k_n, l_n]));
    let r = einsum("bijk,bjkl->bil", &[&a, &b]).expect("einsum batch multi-axis f32");
    assert!(r.is_cuda(), "bijk,bjkl->bil result must stay on device");
    assert_eq!(r.shape(), &[b_n, i_n, l_n]);

    // Reference:
    let a_d: Vec<f64> = a_data.iter().map(|&x| x as f64).collect();
    let b_d: Vec<f64> = b_data.iter().map(|&x| x as f64).collect();
    let mut expected_d = vec![0.0_f64; b_n * i_n * l_n];
    for bi in 0..b_n {
        for ii in 0..i_n {
            for ll in 0..l_n {
                let mut acc = 0.0_f64;
                for jj in 0..j_n {
                    for kk in 0..k_n {
                        let a_idx = bi * (i_n * j_n * k_n) + ii * (j_n * k_n) + jj * k_n + kk;
                        let b_idx = bi * (j_n * k_n * l_n) + jj * (k_n * l_n) + kk * l_n + ll;
                        acc += a_d[a_idx] * b_d[b_idx];
                    }
                }
                expected_d[bi * (i_n * l_n) + ii * l_n + ll] = acc;
            }
        }
    }
    let expected: Vec<f32> = expected_d.iter().map(|&x| x as f32).collect();
    let host = read_f32(&r);
    assert_close_f32(&host, &expected, F32_MATMUL_GPU, "bijk,bjkl->bil f32");
}

#[test]
fn gpu_einsum_permuted_a_ikj_jkl_il_f32_works() {
    // a is laid out [i,k,j] (so [I,K,J]) but contracted axes are j,k.
    // Equivalent to ijk,jkl->il with A permuted axes 1<->2.
    ensure_cuda_backend();
    let (i_n, j_n, k_n, l_n) = (2usize, 3usize, 4usize, 5usize);
    // Build a in [I,K,J] order.
    let mut a_data = vec![0.0_f32; i_n * k_n * j_n];
    let base: Vec<f32> = ramp_f32(i_n * j_n * k_n); // base in [I,J,K] order
    for ii in 0..i_n {
        for kk in 0..k_n {
            for jj in 0..j_n {
                let dst = ii * (k_n * j_n) + kk * j_n + jj;
                let src = ii * (j_n * k_n) + jj * k_n + kk;
                a_data[dst] = base[src];
            }
        }
    }
    let b_data = ramp_f32(j_n * k_n * l_n);
    let a = upload_f32(t_f32(&a_data, &[i_n, k_n, j_n]));
    let b = upload_f32(t_f32(&b_data, &[j_n, k_n, l_n]));
    let r = einsum("ikj,jkl->il", &[&a, &b]).expect("einsum permuted f32");
    assert!(r.is_cuda(), "permuted result must stay on device");
    assert_eq!(r.shape(), &[i_n, l_n]);

    // Reference uses base in [I,J,K] order with the canonical equation.
    let base_d: Vec<f64> = base.iter().map(|&x| x as f64).collect();
    let b_d: Vec<f64> = b_data.iter().map(|&x| x as f64).collect();
    let expected_d = ref_ijk_jkl_il_f64(&base_d, &b_d, i_n, j_n, k_n, l_n);
    let expected: Vec<f32> = expected_d.iter().map(|&x| x as f32).collect();
    let host = read_f32(&r);
    assert_close_f32(&host, &expected, F32_MATMUL_GPU, "ikj,jkl->il f32");
}

#[test]
fn gpu_einsum_permuted_out_ijk_jkl_li_f32_works() {
    // Output is permuted: instead of "il", request "li".
    ensure_cuda_backend();
    let (i_n, j_n, k_n, l_n) = (2usize, 3usize, 4usize, 5usize);
    let a_data = ramp_f32(i_n * j_n * k_n);
    let b_data = ramp_f32(j_n * k_n * l_n);
    let a = upload_f32(t_f32(&a_data, &[i_n, j_n, k_n]));
    let b = upload_f32(t_f32(&b_data, &[j_n, k_n, l_n]));
    let r = einsum("ijk,jkl->li", &[&a, &b]).expect("einsum perm-out f32");
    assert!(r.is_cuda(), "perm-out result must stay on device");
    assert_eq!(r.shape(), &[l_n, i_n]);

    let a_d: Vec<f64> = a_data.iter().map(|&x| x as f64).collect();
    let b_d: Vec<f64> = b_data.iter().map(|&x| x as f64).collect();
    let il_d = ref_ijk_jkl_il_f64(&a_d, &b_d, i_n, j_n, k_n, l_n);
    // Transpose il -> li.
    let mut expected_d = vec![0.0_f64; l_n * i_n];
    for ii in 0..i_n {
        for ll in 0..l_n {
            expected_d[ll * i_n + ii] = il_d[ii * l_n + ll];
        }
    }
    let expected: Vec<f32> = expected_d.iter().map(|&x| x as f32).collect();
    let host = read_f32(&r);
    assert_close_f32(&host, &expected, F32_MATMUL_GPU, "ijk,jkl->li f32");
}
