#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::device::Device;
use ferrotorch_core::dtype::DType;
use ferrotorch_core::grad_fns::linalg::{
    bmm_differentiable, dot_differentiable, matmul_differentiable, mm_differentiable,
    mv_differentiable,
};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for CORE-188 CUDA parity tests");
    });
}

fn f16_leaf(data: &[f32], shape: &[usize]) -> Tensor<half::f16> {
    let values: Vec<half::f16> = data.iter().map(|&v| half::f16::from_f32(v)).collect();
    Tensor::from_storage(TensorStorage::cpu(values), shape.to_vec(), false)
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap()
        .requires_grad_(true)
}

fn bf16_leaf(data: &[f32], shape: &[usize]) -> Tensor<half::bf16> {
    let values: Vec<half::bf16> = data.iter().map(|&v| half::bf16::from_f32(v)).collect();
    Tensor::from_storage(TensorStorage::cpu(values), shape.to_vec(), false)
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap()
        .requires_grad_(true)
}

fn to_f32_f16(t: &Tensor<half::f16>) -> Vec<f32> {
    let cpu = if t.is_cuda() {
        t.cpu().unwrap()
    } else {
        t.clone()
    };
    cpu.data().unwrap().iter().map(|v| v.to_f32()).collect()
}

fn to_f32_bf16(t: &Tensor<half::bf16>) -> Vec<f32> {
    let cpu = if t.is_cuda() {
        t.cpu().unwrap()
    } else {
        t.clone()
    };
    cpu.data().unwrap().iter().map(|v| v.to_f32()).collect()
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    for (i, (&got, &want)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() <= tol,
            "{label}: index {i} got {got}, expected {want} within {tol}"
        );
    }
}

fn assert_cuda_dtype_f16(t: &Tensor<half::f16>, label: &str) {
    assert_eq!(t.device(), Device::Cuda(0), "{label}: device");
    assert_eq!(
        t.gpu_handle().unwrap().dtype(),
        DType::F16,
        "{label}: dtype"
    );
}

fn assert_cuda_dtype_bf16(t: &Tensor<half::bf16>, label: &str) {
    assert_eq!(t.device(), Device::Cuda(0), "{label}: device");
    assert_eq!(
        t.gpu_handle().unwrap().dtype(),
        DType::BF16,
        "{label}: dtype"
    );
}

fn assert_grad_f16(leaf: &Tensor<half::f16>, expected: &[f32], label: &str) {
    let grad = leaf
        .grad()
        .unwrap()
        .unwrap_or_else(|| panic!("{label}: missing grad"));
    assert_cuda_dtype_f16(&grad, label);
    assert_close(&to_f32_f16(&grad), expected, 0.0, label);
}

fn assert_grad_bf16(leaf: &Tensor<half::bf16>, expected: &[f32], label: &str) {
    let grad = leaf
        .grad()
        .unwrap()
        .unwrap_or_else(|| panic!("{label}: missing grad"));
    assert_cuda_dtype_bf16(&grad, label);
    assert_close(&to_f32_bf16(&grad), expected, 0.0, label);
}

#[test]
fn f16_cuda_mm_backward_matches_torch_dtype_and_device() {
    ensure_cuda_backend();
    let a = f16_leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = f16_leaf(&[0.5, -1.0, 1.5, 2.0, -0.5, 3.0], &[3, 2]);

    let y = mm_differentiable(&a, &b).expect("f16 CUDA mm forward");
    assert_cuda_dtype_f16(&y, "f16 mm output");
    assert_close(
        &to_f32_f16(&y),
        &[2.0, 12.0, 6.5, 24.0],
        0.0,
        "f16 mm output",
    );

    y.sum_all().unwrap().backward().expect("f16 mm backward");
    assert_grad_f16(&a, &[-0.5, 3.5, 2.5, -0.5, 3.5, 2.5], "f16 mm grad a");
    assert_grad_f16(&b, &[5.0, 5.0, 7.0, 7.0, 9.0, 9.0], "f16 mm grad b");
}

#[test]
fn bf16_cuda_mm_backward_matches_torch_dtype_and_device() {
    ensure_cuda_backend();
    let a = bf16_leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = bf16_leaf(&[0.5, -1.0, 1.5, 2.0, -0.5, 3.0], &[3, 2]);

    let y = mm_differentiable(&a, &b).expect("bf16 CUDA mm forward");
    assert_cuda_dtype_bf16(&y, "bf16 mm output");
    assert_close(
        &to_f32_bf16(&y),
        &[2.0, 12.0, 6.5, 24.0],
        0.0,
        "bf16 mm output",
    );

    y.sum_all().unwrap().backward().expect("bf16 mm backward");
    assert_grad_bf16(&a, &[-0.5, 3.5, 2.5, -0.5, 3.5, 2.5], "bf16 mm grad a");
    assert_grad_bf16(&b, &[5.0, 5.0, 7.0, 7.0, 9.0, 9.0], "bf16 mm grad b");
}

#[test]
fn f16_cuda_bmm_backward_matches_torch_dtype_and_device() {
    ensure_cuda_backend();
    let a = f16_leaf(&(1..=12).map(|v| v as f32).collect::<Vec<_>>(), &[2, 2, 3]);
    let b = f16_leaf(&(1..=12).map(|v| v as f32).collect::<Vec<_>>(), &[2, 3, 2]);

    let y = bmm_differentiable(&a, &b).expect("f16 CUDA bmm forward");
    assert_cuda_dtype_f16(&y, "f16 bmm output");
    assert_close(
        &to_f32_f16(&y),
        &[22.0, 28.0, 49.0, 64.0, 220.0, 244.0, 301.0, 334.0],
        0.0,
        "f16 bmm output",
    );

    y.sum_all().unwrap().backward().expect("f16 bmm backward");
    assert_grad_f16(
        &a,
        &[
            3.0, 7.0, 11.0, 3.0, 7.0, 11.0, 15.0, 19.0, 23.0, 15.0, 19.0, 23.0,
        ],
        "f16 bmm grad a",
    );
    assert_grad_f16(
        &b,
        &[
            5.0, 5.0, 7.0, 7.0, 9.0, 9.0, 17.0, 17.0, 19.0, 19.0, 21.0, 21.0,
        ],
        "f16 bmm grad b",
    );
}

#[test]
fn bf16_cuda_bmm_backward_matches_torch_dtype_and_device() {
    ensure_cuda_backend();
    let a = bf16_leaf(&(1..=12).map(|v| v as f32).collect::<Vec<_>>(), &[2, 2, 3]);
    let b = bf16_leaf(&(1..=12).map(|v| v as f32).collect::<Vec<_>>(), &[2, 3, 2]);

    let y = bmm_differentiable(&a, &b).expect("bf16 CUDA bmm forward");
    assert_cuda_dtype_bf16(&y, "bf16 bmm output");
    assert_close(
        &to_f32_bf16(&y),
        &[22.0, 28.0, 49.0, 64.0, 220.0, 244.0, 300.0, 334.0],
        0.0,
        "bf16 bmm output",
    );

    y.sum_all().unwrap().backward().expect("bf16 bmm backward");
    assert_grad_bf16(
        &a,
        &[
            3.0, 7.0, 11.0, 3.0, 7.0, 11.0, 15.0, 19.0, 23.0, 15.0, 19.0, 23.0,
        ],
        "bf16 bmm grad a",
    );
    assert_grad_bf16(
        &b,
        &[
            5.0, 5.0, 7.0, 7.0, 9.0, 9.0, 17.0, 17.0, 19.0, 19.0, 21.0, 21.0,
        ],
        "bf16 bmm grad b",
    );
}

#[test]
fn f16_bf16_cuda_broadcast_matmul_backward_matches_torch_sum_to() {
    ensure_cuda_backend();

    let a = f16_leaf(&(1..=12).map(|v| v as f32).collect::<Vec<_>>(), &[2, 2, 3]);
    let b = f16_leaf(&[1.0, -2.0, 3.0, 4.0, -1.0, 2.0], &[3, 2]);
    let y = matmul_differentiable(&a, &b).expect("f16 CUDA broadcast matmul forward");
    assert_cuda_dtype_f16(&y, "f16 broadcast output");
    assert_close(
        &to_f32_f16(&y),
        &[4.0, 12.0, 13.0, 24.0, 22.0, 36.0, 31.0, 48.0],
        0.0,
        "f16 broadcast output",
    );
    y.sum_all()
        .unwrap()
        .backward()
        .expect("f16 broadcast backward");
    assert_grad_f16(
        &a,
        &[
            -1.0, 7.0, 1.0, -1.0, 7.0, 1.0, -1.0, 7.0, 1.0, -1.0, 7.0, 1.0,
        ],
        "f16 broadcast grad a",
    );
    assert_grad_f16(
        &b,
        &[22.0, 22.0, 26.0, 26.0, 30.0, 30.0],
        "f16 broadcast grad b",
    );

    let a = bf16_leaf(&(1..=12).map(|v| v as f32).collect::<Vec<_>>(), &[2, 2, 3]);
    let b = bf16_leaf(&[1.0, -2.0, 3.0, 4.0, -1.0, 2.0], &[3, 2]);
    let y = matmul_differentiable(&a, &b).expect("bf16 CUDA broadcast matmul forward");
    assert_cuda_dtype_bf16(&y, "bf16 broadcast output");
    assert_close(
        &to_f32_bf16(&y),
        &[4.0, 12.0, 13.0, 24.0, 22.0, 36.0, 31.0, 48.0],
        0.0,
        "bf16 broadcast output",
    );
    y.sum_all()
        .unwrap()
        .backward()
        .expect("bf16 broadcast backward");
    assert_grad_bf16(
        &a,
        &[
            -1.0, 7.0, 1.0, -1.0, 7.0, 1.0, -1.0, 7.0, 1.0, -1.0, 7.0, 1.0,
        ],
        "bf16 broadcast grad a",
    );
    assert_grad_bf16(
        &b,
        &[22.0, 22.0, 26.0, 26.0, 30.0, 30.0],
        "bf16 broadcast grad b",
    );
}

#[test]
fn f16_bf16_cuda_non_uniform_broadcast_matmul_stays_resident() {
    ensure_cuda_backend();
    let a_ones = vec![1.0; 2 * 2 * 3];
    let b_ones = vec![1.0; 4 * 3 * 2];

    let a = f16_leaf(&a_ones, &[2, 1, 2, 3]);
    let b = f16_leaf(&b_ones, &[1, 4, 3, 2]);
    let y = matmul_differentiable(&a, &b).expect("f16 non-uniform broadcast forward");
    assert_cuda_dtype_f16(&y, "f16 non-uniform output");
    assert_eq!(y.shape(), &[2, 4, 2, 2]);
    assert_close(
        &to_f32_f16(&y),
        &[3.0; 2 * 4 * 2 * 2],
        0.0,
        "f16 non-uniform output",
    );
    y.sum_all()
        .unwrap()
        .backward()
        .expect("f16 non-uniform backward");
    assert_grad_f16(&a, &[8.0; 2 * 2 * 3], "f16 non-uniform grad a");
    assert_grad_f16(&b, &[4.0; 4 * 3 * 2], "f16 non-uniform grad b");

    let a = bf16_leaf(&a_ones, &[2, 1, 2, 3]);
    let b = bf16_leaf(&b_ones, &[1, 4, 3, 2]);
    let y = matmul_differentiable(&a, &b).expect("bf16 non-uniform broadcast forward");
    assert_cuda_dtype_bf16(&y, "bf16 non-uniform output");
    assert_eq!(y.shape(), &[2, 4, 2, 2]);
    assert_close(
        &to_f32_bf16(&y),
        &[3.0; 2 * 4 * 2 * 2],
        0.0,
        "bf16 non-uniform output",
    );
    y.sum_all()
        .unwrap()
        .backward()
        .expect("bf16 non-uniform backward");
    assert_grad_bf16(&a, &[8.0; 2 * 2 * 3], "bf16 non-uniform grad a");
    assert_grad_bf16(&b, &[4.0; 4 * 3 * 2], "bf16 non-uniform grad b");
}

#[test]
fn f16_bf16_cuda_vector_matmul_cases_match_torch() {
    ensure_cuda_backend();

    let x = f16_leaf(&[1.0, 2.0, 3.0], &[3]);
    let b = f16_leaf(&[0.5, -1.0, 1.5, 2.0, -0.5, 3.0], &[3, 2]);
    let y = matmul_differentiable(&x, &b).expect("f16 vm forward");
    assert_cuda_dtype_f16(&y, "f16 vm output");
    assert_close(&to_f32_f16(&y), &[2.0, 12.0], 0.0, "f16 vm output");
    y.sum_all().unwrap().backward().expect("f16 vm backward");
    assert_grad_f16(&x, &[-0.5, 3.5, 2.5], "f16 vm grad x");
    assert_grad_f16(&b, &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0], "f16 vm grad b");

    let x = bf16_leaf(&[1.0, 2.0, 3.0], &[3]);
    let b = bf16_leaf(&[0.5, -1.0, 1.5, 2.0, -0.5, 3.0], &[3, 2]);
    let y = matmul_differentiable(&x, &b).expect("bf16 vm forward");
    assert_cuda_dtype_bf16(&y, "bf16 vm output");
    assert_close(&to_f32_bf16(&y), &[2.0, 12.0], 0.0, "bf16 vm output");
    y.sum_all().unwrap().backward().expect("bf16 vm backward");
    assert_grad_bf16(&x, &[-0.5, 3.5, 2.5], "bf16 vm grad x");
    assert_grad_bf16(&b, &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0], "bf16 vm grad b");

    let a = f16_leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let x = f16_leaf(&[0.5, 1.5, -0.5], &[3]);
    let y = mv_differentiable(&a, &x).expect("f16 mv forward");
    assert_cuda_dtype_f16(&y, "f16 mv output");
    assert_close(&to_f32_f16(&y), &[2.0, 6.5], 0.0, "f16 mv output");
    y.sum_all().unwrap().backward().expect("f16 mv backward");
    assert_grad_f16(&a, &[0.5, 1.5, -0.5, 0.5, 1.5, -0.5], "f16 mv grad a");
    assert_grad_f16(&x, &[5.0, 7.0, 9.0], "f16 mv grad x");

    let a = bf16_leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let x = bf16_leaf(&[0.5, 1.5, -0.5], &[3]);
    let y = mv_differentiable(&a, &x).expect("bf16 mv forward");
    assert_cuda_dtype_bf16(&y, "bf16 mv output");
    assert_close(&to_f32_bf16(&y), &[2.0, 6.5], 0.0, "bf16 mv output");
    y.sum_all().unwrap().backward().expect("bf16 mv backward");
    assert_grad_bf16(&a, &[0.5, 1.5, -0.5, 0.5, 1.5, -0.5], "bf16 mv grad a");
    assert_grad_bf16(&x, &[5.0, 7.0, 9.0], "bf16 mv grad x");

    let x = f16_leaf(&[1.0, 2.0, 3.0], &[3]);
    let yv = f16_leaf(&[0.5, 1.5, -0.5], &[3]);
    let y = dot_differentiable(&x, &yv).expect("f16 dot forward");
    assert_cuda_dtype_f16(&y, "f16 dot output");
    assert_close(&to_f32_f16(&y), &[2.0], 0.0, "f16 dot output");
    y.backward().expect("f16 dot backward");
    assert_grad_f16(&x, &[0.5, 1.5, -0.5], "f16 dot grad x");
    assert_grad_f16(&yv, &[1.0, 2.0, 3.0], "f16 dot grad y");

    let x = bf16_leaf(&[1.0, 2.0, 3.0], &[3]);
    let yv = bf16_leaf(&[0.5, 1.5, -0.5], &[3]);
    let y = dot_differentiable(&x, &yv).expect("bf16 dot forward");
    assert_cuda_dtype_bf16(&y, "bf16 dot output");
    assert_close(&to_f32_bf16(&y), &[2.0], 0.0, "bf16 dot output");
    y.backward().expect("bf16 dot backward");
    assert_grad_bf16(&x, &[0.5, 1.5, -0.5], "bf16 dot grad x");
    assert_grad_bf16(&yv, &[1.0, 2.0, 3.0], "bf16 dot grad y");
}
