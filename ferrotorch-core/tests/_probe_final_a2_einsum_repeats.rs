//! Permanent regression sentinel for #824 + #825 — einsum mixed/operand
//! repeats on GPU (Final mop-up A2 / Bugfix Batch 6 dispatch A2 follow-up).
//!
//! ## #824 — single-input mixed repeated/free indices
//!
//! Pre-fix: `einsum("iij->j", a)` on CUDA returns
//! `Err(NotImplementedOnCuda { op: "einsum_repeated_index_mixed" })`. PyTorch
//! handles this on CUDA: extract the i-diagonal under each j independently.
//!
//! Post-fix: `einsum_single_repeated_gpu` extends to mixed cases by
//! constructing per-axis as_strided views that walk the diagonal across
//! the repeated chars while preserving free-index strides. The free-index
//! axes are then sum-reduced (or kept) to match `out_subs`.
//!
//! ## #825 — 2-input with operand repeats
//!
//! Pre-fix: `einsum("ii,j->j", a, b)` on CUDA returns
//! `Err(NotImplementedOnCuda { op: "einsum_repeated_index" })`. PyTorch
//! handles these by diagonalising the offending operand first, then
//! dispatching the standard contraction.
//!
//! Post-fix: a pre-pass in `einsum_two_gpu` detects repeated input chars
//! and applies the same as_strided-driven diagonalisation as #824 to
//! reduce each operand to distinct chars before falling into the general
//! 2-input decomposition.

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

// CPU reference: takes a value-source closure and walks the einsum
// definition over user-provided index ranges. Used to compute the
// expected value without re-implementing einsum on the host.
fn ramp_f32(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32) * 0.1 - 1.0).collect()
}
fn ramp_f64(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i as f64) * 0.1 - 1.0).collect()
}

// ---------------------------------------------------------------------------
// #824 — single-input mixed repeated/free indices
// ---------------------------------------------------------------------------

#[test]
fn gpu_einsum_iij_to_j_f32_works() {
    // a: [N, N, M], result[j] = sum_i a[i, i, j].
    ensure_cuda_backend();
    let n = 4usize;
    let m = 3usize;
    let a_data = ramp_f32(n * n * m);
    let a = upload_f32(t_f32(&a_data, &[n, n, m]));
    let r = einsum("iij->j", &[&a]).expect("einsum iij->j f32 on cuda");
    assert!(r.is_cuda(), "iij->j f32 result must stay on device");
    assert_eq!(r.shape(), &[m]);

    let mut expected = vec![0.0_f32; m];
    for j in 0..m {
        let mut acc = 0.0_f32;
        for i in 0..n {
            acc += a_data[i * n * m + i * m + j];
        }
        expected[j] = acc;
    }
    let host = read_f32(&r);
    assert_close_f32(&host, &expected, F32_MATMUL_GPU, "iij->j f32");
}

#[test]
fn gpu_einsum_iij_to_j_f64_works() {
    ensure_cuda_backend();
    let n = 4usize;
    let m = 3usize;
    let a_data = ramp_f64(n * n * m);
    let a = upload_f64(t_f64(&a_data, &[n, n, m]));
    let r = einsum("iij->j", &[&a]).expect("einsum iij->j f64 on cuda");
    assert!(r.is_cuda());
    assert_eq!(r.shape(), &[m]);

    let mut expected = vec![0.0_f64; m];
    for j in 0..m {
        let mut acc = 0.0_f64;
        for i in 0..n {
            acc += a_data[i * n * m + i * m + j];
        }
        expected[j] = acc;
    }
    let host = read_f64(&r);
    assert_close_f64(&host, &expected, F64_MATMUL_GPU, "iij->j f64");
}

#[test]
fn gpu_einsum_iji_to_j_f32_works() {
    // a: [N, M, N], result[j] = sum_i a[i, j, i]. j sits between repeats.
    ensure_cuda_backend();
    let n = 4usize;
    let m = 3usize;
    let a_data = ramp_f32(n * m * n);
    let a = upload_f32(t_f32(&a_data, &[n, m, n]));
    let r = einsum("iji->j", &[&a]).expect("einsum iji->j f32 on cuda");
    assert!(r.is_cuda());
    assert_eq!(r.shape(), &[m]);

    let mut expected = vec![0.0_f32; m];
    for j in 0..m {
        let mut acc = 0.0_f32;
        for i in 0..n {
            acc += a_data[i * m * n + j * n + i];
        }
        expected[j] = acc;
    }
    let host = read_f32(&r);
    assert_close_f32(&host, &expected, F32_MATMUL_GPU, "iji->j f32");
}

#[test]
fn gpu_einsum_iijk_to_jk_f32_works() {
    // a: [N, N, M, P], result[j,k] = sum_i a[i,i,j,k]. Multi-free with repeats.
    ensure_cuda_backend();
    let n = 3usize;
    let m = 2usize;
    let p = 4usize;
    let a_data = ramp_f32(n * n * m * p);
    let a = upload_f32(t_f32(&a_data, &[n, n, m, p]));
    let r = einsum("iijk->jk", &[&a]).expect("einsum iijk->jk f32 on cuda");
    assert!(r.is_cuda());
    assert_eq!(r.shape(), &[m, p]);

    let mut expected = vec![0.0_f32; m * p];
    for j in 0..m {
        for k in 0..p {
            let mut acc = 0.0_f32;
            for i in 0..n {
                acc += a_data[i * n * m * p + i * m * p + j * p + k];
            }
            expected[j * p + k] = acc;
        }
    }
    let host = read_f32(&r);
    assert_close_f32(&host, &expected, F32_MATMUL_GPU, "iijk->jk f32");
}

#[test]
fn gpu_einsum_iii_to_i_regression_f32_works() {
    // Homogeneous-repeat regression sentinel from #821: must keep working.
    ensure_cuda_backend();
    let n = 4usize;
    let a_data = ramp_f32(n * n * n);
    let a = upload_f32(t_f32(&a_data, &[n, n, n]));
    let r = einsum("iii->i", &[&a]).expect("einsum iii->i f32 on cuda");
    assert!(r.is_cuda());
    assert_eq!(r.shape(), &[n]);

    let mut expected = vec![0.0_f32; n];
    for i in 0..n {
        expected[i] = a_data[i * n * n + i * n + i];
    }
    let host = read_f32(&r);
    assert_close_f32(&host, &expected, F32_MATMUL_GPU, "iii->i regression f32");
}

#[test]
fn gpu_einsum_iij_to_ij_kept_f32_works() {
    // a: [N, N, M], request "iij->ij": diagonal kept under both i and j.
    // result[i,j] = a[i,i,j] (no reduction over i).
    ensure_cuda_backend();
    let n = 4usize;
    let m = 3usize;
    let a_data = ramp_f32(n * n * m);
    let a = upload_f32(t_f32(&a_data, &[n, n, m]));
    let r = einsum("iij->ij", &[&a]).expect("einsum iij->ij f32 on cuda");
    assert!(r.is_cuda());
    assert_eq!(r.shape(), &[n, m]);

    let mut expected = vec![0.0_f32; n * m];
    for i in 0..n {
        for j in 0..m {
            expected[i * m + j] = a_data[i * n * m + i * m + j];
        }
    }
    let host = read_f32(&r);
    assert_close_f32(&host, &expected, F32_MATMUL_GPU, "iij->ij f32");
}

// ---------------------------------------------------------------------------
// #825 — 2-input with operand repeats
// ---------------------------------------------------------------------------

#[test]
fn gpu_einsum_ii_comma_j_to_j_f32_works() {
    // a:[N,N], b:[M] -> [M]. Operand-repeat in A (no contraction with B).
    // result[j] = (sum_i a[i,i]) * b[j]  (effectively trace(A) * B).
    ensure_cuda_backend();
    let n = 4usize;
    let m = 3usize;
    let a_data = ramp_f32(n * n);
    let b_data = ramp_f32(m);
    let a = upload_f32(t_f32(&a_data, &[n, n]));
    let b = upload_f32(t_f32(&b_data, &[m]));
    let r = einsum("ii,j->j", &[&a, &b]).expect("einsum ii,j->j f32 on cuda");
    assert!(r.is_cuda());
    assert_eq!(r.shape(), &[m]);

    let mut tr = 0.0_f32;
    for i in 0..n {
        tr += a_data[i * n + i];
    }
    let expected: Vec<f32> = b_data.iter().map(|&x| tr * x).collect();
    let host = read_f32(&r);
    assert_close_f32(&host, &expected, F32_MATMUL_GPU, "ii,j->j f32");
}

#[test]
fn gpu_einsum_ii_comma_j_to_j_f64_works() {
    ensure_cuda_backend();
    let n = 4usize;
    let m = 3usize;
    let a_data = ramp_f64(n * n);
    let b_data = ramp_f64(m);
    let a = upload_f64(t_f64(&a_data, &[n, n]));
    let b = upload_f64(t_f64(&b_data, &[m]));
    let r = einsum("ii,j->j", &[&a, &b]).expect("einsum ii,j->j f64 on cuda");
    assert!(r.is_cuda());
    assert_eq!(r.shape(), &[m]);

    let mut tr = 0.0_f64;
    for i in 0..n {
        tr += a_data[i * n + i];
    }
    let expected: Vec<f64> = b_data.iter().map(|&x| tr * x).collect();
    let host = read_f64(&r);
    assert_close_f64(&host, &expected, F64_MATMUL_GPU, "ii,j->j f64");
}

#[test]
fn gpu_einsum_ij_comma_jj_to_i_f32_works() {
    // a:[I,J], b:[J,J] -> [I]. Repeats in B; j is the contracting char.
    // result[i] = sum_j a[i,j] * b[j,j]
    //          = sum_j a[i,j] * diag(B)[j]
    ensure_cuda_backend();
    let i_n = 3usize;
    let j_n = 4usize;
    let a_data = ramp_f32(i_n * j_n);
    let b_data = ramp_f32(j_n * j_n);
    let a = upload_f32(t_f32(&a_data, &[i_n, j_n]));
    let b = upload_f32(t_f32(&b_data, &[j_n, j_n]));
    let r = einsum("ij,jj->i", &[&a, &b]).expect("einsum ij,jj->i f32 on cuda");
    assert!(r.is_cuda());
    assert_eq!(r.shape(), &[i_n]);

    let diag_b: Vec<f32> = (0..j_n).map(|j| b_data[j * j_n + j]).collect();
    let mut expected = vec![0.0_f32; i_n];
    for i in 0..i_n {
        let mut acc = 0.0_f32;
        for j in 0..j_n {
            acc += a_data[i * j_n + j] * diag_b[j];
        }
        expected[i] = acc;
    }
    let host = read_f32(&r);
    assert_close_f32(&host, &expected, F32_MATMUL_GPU, "ij,jj->i f32");
}

#[test]
fn gpu_einsum_ii_comma_jk_to_jk_f32_works() {
    // a:[N,N], b:[M,P] -> [M,P]. A contributes a scalar (trace) factor.
    // result[j,k] = (sum_i a[i,i]) * b[j,k].
    ensure_cuda_backend();
    let n = 3usize;
    let m = 2usize;
    let p = 4usize;
    let a_data = ramp_f32(n * n);
    let b_data = ramp_f32(m * p);
    let a = upload_f32(t_f32(&a_data, &[n, n]));
    let b = upload_f32(t_f32(&b_data, &[m, p]));
    let r = einsum("ii,jk->jk", &[&a, &b]).expect("einsum ii,jk->jk f32 on cuda");
    assert!(r.is_cuda());
    assert_eq!(r.shape(), &[m, p]);

    let mut tr = 0.0_f32;
    for i in 0..n {
        tr += a_data[i * n + i];
    }
    let expected: Vec<f32> = b_data.iter().map(|&x| tr * x).collect();
    let host = read_f32(&r);
    assert_close_f32(&host, &expected, F32_MATMUL_GPU, "ii,jk->jk f32");
}

#[test]
fn gpu_einsum_no_repeats_regression_ij_jk_to_ik_f32_works() {
    // Pure no-repeat regression sentinel for #822 — must keep working.
    ensure_cuda_backend();
    let i_n = 2usize;
    let j_n = 3usize;
    let k_n = 4usize;
    let a_data = ramp_f32(i_n * j_n);
    let b_data = ramp_f32(j_n * k_n);
    let a = upload_f32(t_f32(&a_data, &[i_n, j_n]));
    let b = upload_f32(t_f32(&b_data, &[j_n, k_n]));
    let r = einsum("ij,jk->ik", &[&a, &b]).expect("einsum ij,jk->ik f32 on cuda");
    assert!(r.is_cuda());
    assert_eq!(r.shape(), &[i_n, k_n]);

    let mut expected = vec![0.0_f32; i_n * k_n];
    for i in 0..i_n {
        for k in 0..k_n {
            let mut acc = 0.0_f32;
            for j in 0..j_n {
                acc += a_data[i * j_n + j] * b_data[j * k_n + k];
            }
            expected[i * k_n + k] = acc;
        }
    }
    let host = read_f32(&r);
    assert_close_f32(&host, &expected, F32_MATMUL_GPU, "ij,jk->ik regression f32");
}
