#![cfg(feature = "gpu")]

use ferrotorch_core::creation::from_vec;
use ferrotorch_core::device::Device;
use ferrotorch_core::tensor::Tensor;
use half::{bf16, f16};
use std::sync::Once;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend().expect("CUDA backend must initialize");
    });
}

fn tensor<T: ferrotorch_core::Float>(values: Vec<T>, shape: &[usize]) -> Tensor<T> {
    from_vec(values, shape).expect("tensor")
}

fn cuda_tensor<T: ferrotorch_core::Float>(values: Vec<T>, shape: &[usize]) -> Tensor<T> {
    tensor(values, shape).to(Device::Cuda(0)).expect("upload")
}

fn cuda_data<T: ferrotorch_core::Float>(t: &Tensor<T>) -> Vec<T> {
    assert_eq!(t.device(), Device::Cuda(0), "tensor must stay CUDA");
    t.to(Device::Cpu)
        .expect("download")
        .data_vec()
        .expect("data")
}

fn assert_close_f32(got: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(got.len(), expected.len());
    for (idx, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() <= tol,
            "idx {idx}: expected {e:?}, got {g:?}, tol {tol}"
        );
    }
}

#[test]
fn cuda_repeat_interleave_f32_f64_forward_backward_stay_resident() {
    ensure_cuda_backend();

    let x = cuda_tensor(vec![0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3]).requires_grad_(true);
    let y = x.repeat_interleave_t(2, 1).expect("repeat_interleave f32");
    assert_eq!(y.device(), Device::Cuda(0));
    assert_eq!(y.shape(), &[2, 6]);
    assert_eq!(
        cuda_data(&y),
        vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0]
    );
    let grad = cuda_tensor(
        vec![
            1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[2, 6],
    );
    let gx = y
        .grad_fn()
        .expect("grad fn")
        .backward(&grad)
        .expect("backward")[0]
        .clone()
        .expect("grad input");
    assert_eq!(gx.device(), Device::Cuda(0));
    assert_eq!(cuda_data(&gx), vec![3.0, 7.0, 11.0, 15.0, 19.0, 23.0]);

    let xd = cuda_tensor(vec![0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3]).requires_grad_(true);
    let yd = xd.repeat_interleave_t(2, 1).expect("repeat_interleave f64");
    assert_eq!(yd.device(), Device::Cuda(0));
    assert_eq!(
        cuda_data(&yd),
        vec![
            0.0_f64, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0
        ]
    );
    let gradd = cuda_tensor(
        vec![
            1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[2, 6],
    );
    let gxd = yd
        .grad_fn()
        .expect("grad fn")
        .backward(&gradd)
        .expect("backward")[0]
        .clone()
        .expect("grad input");
    assert_eq!(gxd.device(), Device::Cuda(0));
    assert_eq!(cuda_data(&gxd), vec![3.0_f64, 7.0, 11.0, 15.0, 19.0, 23.0]);
}

#[test]
fn cuda_repeat_interleave_noncontiguous_and_zero_repeat() {
    ensure_cuda_backend();

    let x = cuda_tensor(vec![0.0_f32, 1.0, 2.0, 3.0], &[2, 2]);
    let xt = x.transpose(0, 1).expect("transpose view");
    let y = xt
        .repeat_interleave_t(3, 0)
        .expect("repeat_interleave non-contiguous");
    assert_eq!(y.device(), Device::Cuda(0));
    assert_eq!(y.shape(), &[6, 2]);
    assert_eq!(
        cuda_data(&y),
        vec![0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0]
    );

    let z = cuda_tensor(vec![0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3]).requires_grad_(true);
    let repeated = z
        .repeat_interleave_t(0, 1)
        .expect("repeat_interleave zero repeats");
    assert_eq!(repeated.device(), Device::Cuda(0));
    assert_eq!(repeated.shape(), &[2, 0]);
    assert!(cuda_data(&repeated).is_empty());
    let empty_grad = cuda_tensor(Vec::<f32>::new(), &[2, 0]);
    let gz = repeated
        .grad_fn()
        .expect("grad fn")
        .backward(&empty_grad)
        .expect("backward")[0]
        .clone()
        .expect("grad input");
    assert_eq!(gz.device(), Device::Cuda(0));
    assert_eq!(cuda_data(&gz), vec![0.0; 6]);
}

#[test]
fn cuda_repeat_interleave_half_family_forward_backward() {
    ensure_cuda_backend();

    let xh = cuda_tensor(
        vec![f16::from_f32(0.0), f16::from_f32(1.0), f16::from_f32(2.0)],
        &[3],
    )
    .requires_grad_(true);
    let yh = xh.repeat_interleave_t(2, 0).expect("repeat_interleave f16");
    let got_h: Vec<f32> = cuda_data(&yh).iter().map(|v| v.to_f32()).collect();
    assert_close_f32(&got_h, &[0.0, 0.0, 1.0, 1.0, 2.0, 2.0], 0.0);
    let gh = cuda_tensor(
        vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
            f16::from_f32(5.0),
            f16::from_f32(6.0),
        ],
        &[6],
    );
    let gxh = yh
        .grad_fn()
        .expect("grad fn")
        .backward(&gh)
        .expect("backward")[0]
        .clone()
        .expect("grad input");
    assert_eq!(gxh.device(), Device::Cuda(0));
    let got_gh: Vec<f32> = cuda_data(&gxh).iter().map(|v| v.to_f32()).collect();
    assert_close_f32(&got_gh, &[3.0, 7.0, 11.0], 0.0);

    let xb = cuda_tensor(
        vec![
            bf16::from_f32(0.0),
            bf16::from_f32(1.0),
            bf16::from_f32(2.0),
        ],
        &[3],
    )
    .requires_grad_(true);
    let yb = xb
        .repeat_interleave_t(2, 0)
        .expect("repeat_interleave bf16");
    let got_b: Vec<f32> = cuda_data(&yb).iter().map(|v| v.to_f32()).collect();
    assert_close_f32(&got_b, &[0.0, 0.0, 1.0, 1.0, 2.0, 2.0], 0.0);
    let gb = cuda_tensor(
        vec![
            bf16::from_f32(1.0),
            bf16::from_f32(2.0),
            bf16::from_f32(3.0),
            bf16::from_f32(4.0),
            bf16::from_f32(5.0),
            bf16::from_f32(6.0),
        ],
        &[6],
    );
    let gxb = yb
        .grad_fn()
        .expect("grad fn")
        .backward(&gb)
        .expect("backward")[0]
        .clone()
        .expect("grad input");
    assert_eq!(gxb.device(), Device::Cuda(0));
    let got_gb: Vec<f32> = cuda_data(&gxb).iter().map(|v| v.to_f32()).collect();
    assert_close_f32(&got_gb, &[3.0, 7.0, 11.0], 0.0);
}
