//! Regression coverage for #1973: `ferrotorch_nn::functional::linear` must
//! accept PyTorch-valid 1D/ND inputs without detaching bias or round-tripping
//! CUDA tensors through CPU storage.
//!
//! Oracle values are from torch 2.11 CUDA semantics for:
//! `F.linear(arange(24).reshape(2,4,3), [[1,2,3],[4,5,6]], [10,20])`.

#![cfg(feature = "cuda")]

use ferrotorch_core::grad_fns::reduction::sum;
use ferrotorch_core::{Device, Tensor, TensorStorage};
use ferrotorch_nn::functional::linear;

fn cuda_ready() -> bool {
    ferrotorch_gpu::init_cuda_backend().is_ok()
}

fn cuda_tensor(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap()
        .detach()
        .requires_grad_(requires_grad)
}

fn host(t: &Tensor<f32>, label: &str) -> Vec<f32> {
    assert_eq!(t.device(), Device::Cuda(0), "{label} left CUDA device");
    t.cpu().unwrap().data().unwrap().to_vec()
}

fn assert_close(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert!((g - w).abs() <= 1e-5, "{label}[{i}]: got {g}, expected {w}");
    }
}

#[test]
fn functional_linear_cuda_nd_forward_and_backward_match_torch() {
    if !cuda_ready() {
        return;
    }

    let input_data: Vec<f32> = (0..24).map(|v| v as f32).collect();
    let input = cuda_tensor(&input_data, &[2, 4, 3], true);
    let weight = cuda_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
    let bias = cuda_tensor(&[10.0, 20.0], &[2], true);

    let output = linear(&input, &weight, Some(&bias)).unwrap();
    assert_eq!(output.shape(), &[2, 4, 2]);
    assert_close(
        "forward",
        &host(&output, "output"),
        &[
            18.0, 37.0, 36.0, 82.0, 54.0, 127.0, 72.0, 172.0, 90.0, 217.0, 108.0, 262.0, 126.0,
            307.0, 144.0, 352.0,
        ],
    );

    sum(&output).unwrap().backward().unwrap();

    assert_close(
        "input grad",
        &host(
            &input
                .grad()
                .unwrap()
                .expect("input gradient should be populated"),
            "input grad",
        ),
        &[
            5.0, 7.0, 9.0, 5.0, 7.0, 9.0, 5.0, 7.0, 9.0, 5.0, 7.0, 9.0, 5.0, 7.0, 9.0, 5.0, 7.0,
            9.0, 5.0, 7.0, 9.0, 5.0, 7.0, 9.0,
        ],
    );
    assert_close(
        "weight grad",
        &host(
            &weight
                .grad()
                .unwrap()
                .expect("weight gradient should be populated"),
            "weight grad",
        ),
        &[84.0, 92.0, 100.0, 84.0, 92.0, 100.0],
    );
    assert_close(
        "bias grad",
        &host(
            &bias
                .grad()
                .unwrap()
                .expect("bias gradient should be populated"),
            "bias grad",
        ),
        &[8.0, 8.0],
    );
}

#[test]
fn functional_linear_cuda_1d_input_with_matrix_broadcast_bias_matches_torch() {
    if !cuda_ready() {
        return;
    }

    let input = cuda_tensor(&[1.0, 2.0, 3.0], &[3], false);
    let weight = cuda_tensor(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3], false);
    let bias = cuda_tensor(&[10.0, 20.0], &[1, 2], false);

    let output = linear(&input, &weight, Some(&bias)).unwrap();
    assert_eq!(output.shape(), &[2]);
    assert_close("1d input", &host(&output, "1d output"), &[11.0, 22.0]);
}
