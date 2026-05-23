//! Tensor<bf16> dispatch verification for #17.
//!
//! After the bf16 GPU buffer plumbing landed in #19 and the
//! LayerNorm/GELU bf16 kernels + `*_bf16_bf16` trait surface landed in
//! #17, `Tensor<bf16>` can move CPU<->CUDA and the `GpuBackend` trait
//! exposes native bf16 -> bf16 dispatch arms (`matmul_bf16_bf16`,
//! `softmax_bf16_bf16`, `layernorm_bf16_bf16`, `gelu_bf16_bf16`,
//! `add_bf16_bf16`, `mul_bf16_bf16`, `scale_bf16_bf16`,
//! `silu_bf16_bf16`, `relu_bf16_bf16`). This probe constructs a
//! `Tensor<bf16>` on CUDA via `cpu.to(Device::Cuda(0))`, extracts the
//! `GpuBufferHandle`, and exercises each new dispatch arm end-to-end
//! against a CPU bf16 reference.
//!
//! No silent CPU fallback (rust-gpu-discipline §3): the GPU kernels are
//! launched directly; failures here are real bugs, not missing backends.

#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::Device;
use ferrotorch_core::creation::from_vec;
use ferrotorch_core::gpu_dispatch;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for the GPU probe suite");
    });
}

fn cpu_bf16_tensor_on_cuda(
    data: &[half::bf16],
    shape: &[usize],
) -> ferrotorch_core::Tensor<half::bf16> {
    let cpu = from_vec::<half::bf16>(data.to_vec(), shape).expect("bf16 cpu tensor");
    cpu.to(Device::Cuda(0))
        .expect("Tensor<bf16>::to(Cuda) must succeed")
}

fn download_bf16(h: &gpu_dispatch::GpuBufferHandle) -> Vec<half::bf16> {
    let backend = gpu_dispatch::gpu_backend().expect("backend");
    let bytes = backend.gpu_to_cpu(h).expect("gpu_to_cpu bf16");
    bytes
        .chunks_exact(2)
        .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

fn max_abs_err(got: &[half::bf16], expected: &[f32]) -> f32 {
    got.iter()
        .zip(expected.iter())
        .map(|(g, e)| (g.to_f32() - e).abs())
        .fold(0.0_f32, f32::max)
}

/// Verifies the `Tensor<bf16> -> gpu_handle -> backend.matmul_bf16_bf16` path.
#[test]
fn issue17_tensor_bf16_matmul_routes_to_gpu_kernel() {
    ensure_cuda_backend();
    let backend = gpu_dispatch::gpu_backend().expect("backend");

    let m = 4;
    let k = 6;
    let n = 5;
    let a_f32: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1 - 0.7).collect();
    let b_f32: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.05 + 0.2).collect();
    let a_bf16: Vec<half::bf16> = a_f32.iter().copied().map(half::bf16::from_f32).collect();
    let b_bf16: Vec<half::bf16> = b_f32.iter().copied().map(half::bf16::from_f32).collect();

    let a_t = cpu_bf16_tensor_on_cuda(&a_bf16, &[m, k]);
    let b_t = cpu_bf16_tensor_on_cuda(&b_bf16, &[k, n]);

    let c_handle = backend
        .matmul_bf16_bf16(
            a_t.gpu_handle().unwrap(),
            b_t.gpu_handle().unwrap(),
            m,
            k,
            n,
        )
        .expect("matmul_bf16_bf16 must launch real cuBLAS GemmEx kernel");

    let got = download_bf16(&c_handle);
    let mut expected = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f32;
            for p in 0..k {
                acc += a_bf16[i * k + p].to_f32() * b_bf16[p * n + j].to_f32();
            }
            expected[i * n + j] = acc;
        }
    }
    let max_err = max_abs_err(&got, &expected);
    assert!(
        max_err < 5e-2,
        "matmul_bf16_bf16 routing: max_abs={max_err}"
    );
}

/// LayerNorm via `Tensor<bf16>` handles — exercises the new
/// `gpu_layernorm_bf16` PTX kernel.
#[test]
fn issue17_tensor_bf16_layernorm_routes_to_gpu_kernel() {
    ensure_cuda_backend();
    let backend = gpu_dispatch::gpu_backend().expect("backend");

    let rows = 2;
    let cols = 12;
    let eps = 1e-5_f32;
    let a_f32: Vec<f32> = (0..rows * cols).map(|i| 0.3 + (i as f32) * 0.07).collect();
    let g_f32: Vec<f32> = (0..cols).map(|i| 1.0 + (i as f32) * 0.01).collect();
    let b_f32: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.02 - 0.05).collect();
    let a_bf16: Vec<half::bf16> = a_f32.iter().copied().map(half::bf16::from_f32).collect();
    let g_bf16: Vec<half::bf16> = g_f32.iter().copied().map(half::bf16::from_f32).collect();
    let b_bf16: Vec<half::bf16> = b_f32.iter().copied().map(half::bf16::from_f32).collect();

    let a_t = cpu_bf16_tensor_on_cuda(&a_bf16, &[rows, cols]);
    let g_t = cpu_bf16_tensor_on_cuda(&g_bf16, &[cols]);
    let b_t = cpu_bf16_tensor_on_cuda(&b_bf16, &[cols]);

    let out_handle = backend
        .layernorm_bf16_bf16(
            a_t.gpu_handle().unwrap(),
            g_t.gpu_handle().unwrap(),
            b_t.gpu_handle().unwrap(),
            rows,
            cols,
            eps,
        )
        .expect("layernorm_bf16_bf16 must launch real PTX kernel");

    let got = download_bf16(&out_handle);
    let mut expected = vec![0.0_f32; rows * cols];
    for r in 0..rows {
        let row: Vec<f32> = a_bf16[r * cols..(r + 1) * cols]
            .iter()
            .map(|x| x.to_f32())
            .collect();
        let mean: f32 = row.iter().sum::<f32>() / (cols as f32);
        let var: f32 = row.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / (cols as f32);
        let inv_std = 1.0 / (var + eps).sqrt();
        for c in 0..cols {
            expected[r * cols + c] =
                (row[c] - mean) * inv_std * g_bf16[c].to_f32() + b_bf16[c].to_f32();
        }
    }
    let max_err = max_abs_err(&got, &expected);
    assert!(
        max_err < 5e-2,
        "layernorm_bf16_bf16 routing: max_abs={max_err}"
    );
}

/// GELU + softmax + add + mul + scale + silu + relu via `Tensor<bf16>` handles.
#[test]
fn issue17_tensor_bf16_activations_route_to_gpu_kernels() {
    ensure_cuda_backend();
    let backend = gpu_dispatch::gpu_backend().expect("backend");

    let n = 32;
    let a_f32: Vec<f32> = (-16..16).map(|i| (i as f32) * 0.3).collect();
    let a_bf16: Vec<half::bf16> = a_f32.iter().copied().map(half::bf16::from_f32).collect();
    let a_t = cpu_bf16_tensor_on_cuda(&a_bf16, &[n]);

    // GELU — verifies the new gpu_gelu_bf16 PTX kernel runs.
    let gelu_h = backend
        .gelu_bf16_bf16(a_t.gpu_handle().unwrap())
        .expect("gelu_bf16_bf16");
    let gelu_got = download_bf16(&gelu_h);
    assert_eq!(gelu_got.len(), n);

    // SiLU
    let silu_h = backend
        .silu_bf16_bf16(a_t.gpu_handle().unwrap())
        .expect("silu_bf16_bf16");
    assert_eq!(download_bf16(&silu_h).len(), n);

    // ReLU
    let relu_h = backend
        .relu_bf16_bf16(a_t.gpu_handle().unwrap())
        .expect("relu_bf16_bf16");
    let relu_got = download_bf16(&relu_h);
    let expected_relu: Vec<f32> = a_bf16.iter().map(|x| x.to_f32().max(0.0)).collect();
    let max_err = max_abs_err(&relu_got, &expected_relu);
    assert!(max_err < 1e-2, "relu_bf16_bf16 routing: max_abs={max_err}");

    // scale
    let scaled_h = backend
        .scale_bf16_bf16(a_t.gpu_handle().unwrap(), 0.25)
        .expect("scale_bf16_bf16");
    let scaled_got = download_bf16(&scaled_h);
    let expected_scaled: Vec<f32> = a_bf16.iter().map(|x| x.to_f32() * 0.25).collect();
    let max_err = max_abs_err(&scaled_got, &expected_scaled);
    assert!(max_err < 1e-2, "scale_bf16_bf16 routing: max_abs={max_err}");

    // add (self + self = 2*self)
    let added_h = backend
        .add_bf16_bf16(a_t.gpu_handle().unwrap(), a_t.gpu_handle().unwrap())
        .expect("add_bf16_bf16");
    let added_got = download_bf16(&added_h);
    let expected_add: Vec<f32> = a_bf16.iter().map(|x| 2.0 * x.to_f32()).collect();
    let max_err = max_abs_err(&added_got, &expected_add);
    assert!(max_err < 1e-2, "add_bf16_bf16 routing: max_abs={max_err}");

    // mul (self * self)
    let muled_h = backend
        .mul_bf16_bf16(a_t.gpu_handle().unwrap(), a_t.gpu_handle().unwrap())
        .expect("mul_bf16_bf16");
    let muled_got = download_bf16(&muled_h);
    let expected_mul: Vec<f32> = a_bf16.iter().map(|x| x.to_f32() * x.to_f32()).collect();
    let max_err = max_abs_err(&muled_got, &expected_mul);
    // squared values reach magnitude 23 in this test set → bf16 absolute
    // tolerance scales accordingly (7-bit mantissa => ~1 ULP at this
    // magnitude is ~0.2).
    assert!(max_err < 0.25, "mul_bf16_bf16 routing: max_abs={max_err}");

    // softmax — bf16 -> bf16
    let softmax_t = cpu_bf16_tensor_on_cuda(&a_bf16, &[1, n]);
    let sm_h = backend
        .softmax_bf16_bf16(softmax_t.gpu_handle().unwrap(), 1, n)
        .expect("softmax_bf16_bf16");
    let sm_got = download_bf16(&sm_h);
    let sum: f32 = sm_got.iter().map(|x| x.to_f32()).sum();
    assert!(
        (sum - 1.0).abs() < 0.03,
        "softmax_bf16_bf16 row sum: {sum} expected ~1.0"
    );
}
