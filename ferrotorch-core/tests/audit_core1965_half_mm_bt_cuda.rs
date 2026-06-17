//! Regression coverage for crosslink #1965: CUDA `mm_bt` must support f16 and
//! bf16 end-to-end because nested attention's composite fallback is
//! `Q @ K^T -> scale -> softmax -> @ V`, matching PyTorch's math SDPA path.
//!
//! PyTorch parity anchors:
//! - `torch.nn.functional.scaled_dot_product_attention` accepts CUDA
//!   `torch.float16` and `torch.bfloat16` and returns the same dtype.
//! - `torch.matmul(a, b.T)` and `torch.nn.functional.linear` on CUDA preserve
//!   half/bfloat storage for outputs and parameter gradients.

#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::grad_fns::linalg::{linear_fused, mm_bt_differentiable};
use ferrotorch_core::nested::{NestedTensor, nested_scaled_dot_product_attention};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::{Device, Tensor};

static GPU_INIT: Once = Once::new();

const A: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
const B: [f32; 12] = [
    0.5, -1.0, 2.0, -2.0, 0.25, 1.5, 3.0, -0.5, -1.0, 0.0, 2.0, -0.25,
];
const BIAS: [f32; 4] = [0.125, -0.5, 1.25, -1.0];

const MM_BT_OUT: [f32; 8] = [4.5, 3.0, -1.0, 3.25, 9.0, 2.25, 3.5, 8.5];
const LINEAR_OUT: [f32; 8] = [4.625, 2.5, 0.25, 2.25, 9.125, 1.75, 4.75, 7.5];
const GRAD_A: [f32; 6] = [1.5, 0.75, 2.25, 1.5, 0.75, 2.25];
const GRAD_B: [f32; 12] = [5.0, 7.0, 9.0, 5.0, 7.0, 9.0, 5.0, 7.0, 9.0, 5.0, 7.0, 9.0];
const GRAD_BIAS: [f32; 4] = [2.0, 2.0, 2.0, 2.0];

const ATTN_F16_D129: [f32; 8] = [
    0.014_404_297,
    0.011_222_839,
    -0.025_619_507,
    0.014_404_297,
    -0.014_404_297,
    0.025_619_507,
    -0.011_222_839,
    -0.014_404_297,
];
const ATTN_BF16_D129: [f32; 8] = [
    0.014_404_297,
    0.011_230_469,
    -0.025_634_766,
    0.014_404_297,
    -0.014_404_297,
    0.025_634_766,
    -0.011_230_469,
    -0.014_404_297,
];

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for crosslink #1965 tests");
    });
}

fn tensor_f16(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<half::f16> {
    let values = data.iter().copied().map(half::f16::from_f32).collect();
    Tensor::from_storage(TensorStorage::cpu(values), shape.to_vec(), requires_grad)
        .expect("make f16 tensor")
}

fn tensor_bf16(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<half::bf16> {
    let values = data.iter().copied().map(half::bf16::from_f32).collect();
    Tensor::from_storage(TensorStorage::cpu(values), shape.to_vec(), requires_grad)
        .expect("make bf16 tensor")
}

fn upload_f16(t: Tensor<half::f16>) -> Tensor<half::f16> {
    let track = t.requires_grad();
    t.detach()
        .to(Device::Cuda(0))
        .expect("upload f16 to CUDA")
        .requires_grad_(track)
}

fn upload_bf16(t: Tensor<half::bf16>) -> Tensor<half::bf16> {
    let track = t.requires_grad();
    t.detach()
        .to(Device::Cuda(0))
        .expect("upload bf16 to CUDA")
        .requires_grad_(track)
}

fn read_f16(t: &Tensor<half::f16>) -> Vec<f32> {
    assert_eq!(
        t.device(),
        Device::Cuda(0),
        "result must stay CUDA-resident"
    );
    t.cpu()
        .expect("read f16 CUDA tensor")
        .data()
        .expect("f16 cpu data")
        .iter()
        .map(|v| v.to_f32())
        .collect()
}

fn read_bf16(t: &Tensor<half::bf16>) -> Vec<f32> {
    assert_eq!(
        t.device(),
        Device::Cuda(0),
        "result must stay CUDA-resident"
    );
    t.cpu()
        .expect("read bf16 CUDA tensor")
        .data()
        .expect("bf16 cpu data")
        .iter()
        .map(|v| v.to_f32())
        .collect()
}

fn assert_close(label: &str, got: &[f32], want: &[f32], atol: f32) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() <= atol,
            "{label}[{i}]: got {g}, expected {w}, diff {} > {atol}",
            (g - w).abs()
        );
    }
}

#[test]
fn cuda_mm_bt_f16_forward_and_backward_match_torch() {
    ensure_cuda_backend();
    let a = upload_f16(tensor_f16(&A, &[2, 3], true));
    let b = upload_f16(tensor_f16(&B, &[4, 3], true));

    let out = mm_bt_differentiable(&a, &b).expect("f16 CUDA mm_bt");
    assert_eq!(out.shape(), &[2, 4]);
    assert_close("mm_bt f16 forward", &read_f16(&out), &MM_BT_OUT, 0.01);

    let loss = ferrotorch_core::grad_fns::reduction::sum(&out).expect("sum f16 mm_bt");
    loss.backward().expect("backward f16 mm_bt");
    assert_close(
        "mm_bt f16 grad_a",
        &read_f16(&a.grad().unwrap().expect("grad a")),
        &GRAD_A,
        0.01,
    );
    assert_close(
        "mm_bt f16 grad_b",
        &read_f16(&b.grad().unwrap().expect("grad b")),
        &GRAD_B,
        0.01,
    );
}

#[test]
fn cuda_mm_bt_bf16_forward_and_backward_match_torch() {
    ensure_cuda_backend();
    let a = upload_bf16(tensor_bf16(&A, &[2, 3], true));
    let b = upload_bf16(tensor_bf16(&B, &[4, 3], true));

    let out = mm_bt_differentiable(&a, &b).expect("bf16 CUDA mm_bt");
    assert_eq!(out.shape(), &[2, 4]);
    assert_close("mm_bt bf16 forward", &read_bf16(&out), &MM_BT_OUT, 0.02);

    let loss = ferrotorch_core::grad_fns::reduction::sum(&out).expect("sum bf16 mm_bt");
    loss.backward().expect("backward bf16 mm_bt");
    assert_close(
        "mm_bt bf16 grad_a",
        &read_bf16(&a.grad().unwrap().expect("grad a")),
        &GRAD_A,
        0.02,
    );
    assert_close(
        "mm_bt bf16 grad_b",
        &read_bf16(&b.grad().unwrap().expect("grad b")),
        &GRAD_B,
        0.02,
    );
}

#[test]
fn cuda_linear_fused_half_and_bfloat_forward_backward_match_torch() {
    ensure_cuda_backend();

    let a16 = upload_f16(tensor_f16(&A, &[2, 3], true));
    let w16 = upload_f16(tensor_f16(&B, &[4, 3], true));
    let b16 = upload_f16(tensor_f16(&BIAS, &[4], true));
    let y16 = linear_fused(&a16, &w16, Some(&b16)).expect("f16 linear_fused");
    assert_close(
        "linear_fused f16 forward",
        &read_f16(&y16),
        &LINEAR_OUT,
        0.01,
    );
    ferrotorch_core::grad_fns::reduction::sum(&y16)
        .expect("sum f16 linear")
        .backward()
        .expect("backward f16 linear");
    assert_close(
        "linear_fused f16 grad_a",
        &read_f16(&a16.grad().unwrap().expect("grad input")),
        &GRAD_A,
        0.01,
    );
    assert_close(
        "linear_fused f16 grad_w",
        &read_f16(&w16.grad().unwrap().expect("grad weight")),
        &GRAD_B,
        0.01,
    );
    assert_close(
        "linear_fused f16 grad_bias",
        &read_f16(&b16.grad().unwrap().expect("grad bias")),
        &GRAD_BIAS,
        0.01,
    );

    let ab = upload_bf16(tensor_bf16(&A, &[2, 3], true));
    let wb = upload_bf16(tensor_bf16(&B, &[4, 3], true));
    let bb = upload_bf16(tensor_bf16(&BIAS, &[4], true));
    let yb = linear_fused(&ab, &wb, Some(&bb)).expect("bf16 linear_fused");
    assert_close(
        "linear_fused bf16 forward",
        &read_bf16(&yb),
        &LINEAR_OUT,
        0.02,
    );
    ferrotorch_core::grad_fns::reduction::sum(&yb)
        .expect("sum bf16 linear")
        .backward()
        .expect("backward bf16 linear");
    assert_close(
        "linear_fused bf16 grad_a",
        &read_bf16(&ab.grad().unwrap().expect("grad input")),
        &GRAD_A,
        0.02,
    );
    assert_close(
        "linear_fused bf16 grad_w",
        &read_bf16(&wb.grad().unwrap().expect("grad weight")),
        &GRAD_B,
        0.02,
    );
    assert_close(
        "linear_fused bf16 grad_bias",
        &read_bf16(&bb.grad().unwrap().expect("grad bias")),
        &GRAD_BIAS,
        0.02,
    );
}

#[test]
fn cuda_nested_attention_composite_accepts_f16_and_bf16() {
    ensure_cuda_backend();
    let q: Vec<f32> = (0..2 * 129).map(|i| (i % 7) as f32 * 0.25 - 0.5).collect();
    let k: Vec<f32> = (0..3 * 129).map(|i| (i % 5) as f32 * 0.5 - 1.0).collect();
    let v: Vec<f32> = (0..3 * 4).map(|i| (i % 3) as f32 - 1.0).collect();

    let q16 = NestedTensor::new(vec![upload_f16(tensor_f16(&q, &[2, 129], false))], 0).unwrap();
    let k16 = NestedTensor::new(vec![upload_f16(tensor_f16(&k, &[3, 129], false))], 0).unwrap();
    let v16 = NestedTensor::new(vec![upload_f16(tensor_f16(&v, &[3, 4], false))], 0).unwrap();
    let y16 = nested_scaled_dot_product_attention(&q16, &k16, &v16)
        .expect("f16 CUDA nested attention composite");
    let y16_0 = &y16.tensors()[0];
    assert_eq!(y16_0.shape(), &[2, 4]);
    assert_close(
        "nested attention f16",
        &read_f16(y16_0),
        &ATTN_F16_D129,
        0.002,
    );

    let qb = NestedTensor::new(vec![upload_bf16(tensor_bf16(&q, &[2, 129], false))], 0).unwrap();
    let kb = NestedTensor::new(vec![upload_bf16(tensor_bf16(&k, &[3, 129], false))], 0).unwrap();
    let vb = NestedTensor::new(vec![upload_bf16(tensor_bf16(&v, &[3, 4], false))], 0).unwrap();
    let yb = nested_scaled_dot_product_attention(&qb, &kb, &vb)
        .expect("bf16 CUDA nested attention composite");
    let yb_0 = &yb.tensors()[0];
    assert_eq!(yb_0.shape(), &[2, 4]);
    assert_close(
        "nested attention bf16",
        &read_bf16(yb_0),
        &ATTN_BF16_D129,
        0.004,
    );
}
