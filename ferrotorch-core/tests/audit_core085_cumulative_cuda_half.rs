#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::creation::from_vec;
use ferrotorch_core::device::Device;
use ferrotorch_core::grad_fns::cumulative::{cummax, cummin, cumprod, cumsum, logcumsumexp};
use ferrotorch_core::grad_fns::reduction::sum;
use ferrotorch_core::int_tensor::IntTensor;
use ferrotorch_core::tensor::Tensor;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for cumulative half CUDA tests");
    });
}

fn f16_cuda(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<half::f16> {
    let h: Vec<half::f16> = data.iter().copied().map(half::f16::from_f32).collect();
    from_vec::<half::f16>(h, shape)
        .expect("f16 cpu tensor")
        .to(Device::Cuda(0))
        .expect("upload f16")
        .requires_grad_(requires_grad)
}

fn bf16_cuda(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<half::bf16> {
    let h: Vec<half::bf16> = data.iter().copied().map(half::bf16::from_f32).collect();
    from_vec::<half::bf16>(h, shape)
        .expect("bf16 cpu tensor")
        .to(Device::Cuda(0))
        .expect("upload bf16")
        .requires_grad_(requires_grad)
}

fn f32_cuda(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
    from_vec::<f32>(data.to_vec(), shape)
        .expect("f32 cpu tensor")
        .to(Device::Cuda(0))
        .expect("upload f32")
        .requires_grad_(requires_grad)
}

fn f64_cuda(data: &[f64], shape: &[usize], requires_grad: bool) -> Tensor<f64> {
    from_vec::<f64>(data.to_vec(), shape)
        .expect("f64 cpu tensor")
        .to(Device::Cuda(0))
        .expect("upload f64")
        .requires_grad_(requires_grad)
}

fn host_f16(t: &Tensor<half::f16>) -> Vec<f32> {
    assert_eq!(
        t.device(),
        Device::Cuda(0),
        "result must stay CUDA-resident"
    );
    t.cpu()
        .expect("D2H f16")
        .data()
        .expect("f16 cpu data")
        .iter()
        .map(|v| v.to_f32())
        .collect()
}

fn host_bf16(t: &Tensor<half::bf16>) -> Vec<f32> {
    assert_eq!(
        t.device(),
        Device::Cuda(0),
        "result must stay CUDA-resident"
    );
    t.cpu()
        .expect("D2H bf16")
        .data()
        .expect("bf16 cpu data")
        .iter()
        .map(|v| v.to_f32())
        .collect()
}

fn host_f32(t: &Tensor<f32>) -> Vec<f32> {
    assert_eq!(
        t.device(),
        Device::Cuda(0),
        "result must stay CUDA-resident"
    );
    t.cpu()
        .expect("D2H f32")
        .data()
        .expect("f32 cpu data")
        .to_vec()
}

fn host_f64(t: &Tensor<f64>) -> Vec<f64> {
    assert_eq!(
        t.device(),
        Device::Cuda(0),
        "result must stay CUDA-resident"
    );
    t.cpu()
        .expect("D2H f64")
        .data()
        .expect("f64 cpu data")
        .to_vec()
}

fn host_indices(t: &IntTensor<i64>) -> Vec<i64> {
    assert_eq!(
        t.device(),
        Device::Cuda(0),
        "indices tensor must stay CUDA-resident"
    );
    t.to(Device::Cpu)
        .expect("D2H indices")
        .data()
        .expect("indices cpu data")
        .to_vec()
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    for (i, (&got, &want)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() <= tol,
            "{label}[{i}] got {got}, want {want}, tol {tol}"
        );
    }
}

fn assert_close_f64(actual: &[f64], expected: &[f64], tol: f64, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    for (i, (&got, &want)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() <= tol,
            "{label}[{i}] got {got}, want {want}, tol {tol}"
        );
    }
}

#[test]
fn f16_cumulative_cuda_forward_and_backward() {
    ensure_cuda_backend();
    let data = [1.0, -2.0, 3.0, 0.5, 4.0, 4.0, 2.0, 5.0];
    let x = f16_cuda(&data, &[2, 4], false);

    assert_close(
        &host_f16(&cumsum(&x, 1).expect("cumsum f16")),
        &[1.0, -1.0, 2.0, 2.5, 4.0, 8.0, 10.0, 15.0],
        0.0,
        "f16 cumsum",
    );
    assert_close(
        &host_f16(&cumprod(&x, 1).expect("cumprod f16")),
        &[1.0, -2.0, -6.0, -3.0, 4.0, 16.0, 32.0, 160.0],
        0.0,
        "f16 cumprod",
    );
    assert_close(
        &host_f16(&logcumsumexp(&x, 1).expect("logcumsumexp f16")),
        &[1.0, 1.0488, 3.1328, 3.2012, 4.0, 4.6914, 4.7578, 5.5781],
        0.02,
        "f16 logcumsumexp",
    );

    let cmx = cummax(&x, 1).expect("cummax f16");
    assert_close(
        &host_f16(&cmx.values),
        &[1.0, 1.0, 3.0, 3.0, 4.0, 4.0, 4.0, 5.0],
        0.0,
        "f16 cummax values",
    );
    assert_eq!(host_indices(&cmx.indices_tensor), &[0, 0, 2, 2, 0, 1, 1, 3]);
    assert!(
        cmx.indices.is_empty(),
        "CUDA cummax must not populate a host indices cache"
    );

    let cmn = cummin(&x, 1).expect("cummin f16");
    assert_close(
        &host_f16(&cmn.values),
        &[1.0, -2.0, -2.0, -2.0, 4.0, 4.0, 2.0, 2.0],
        0.0,
        "f16 cummin values",
    );
    assert_eq!(host_indices(&cmn.indices_tensor), &[0, 1, 1, 1, 0, 1, 2, 2]);
    assert!(
        cmn.indices.is_empty(),
        "CUDA cummin must not populate a host indices cache"
    );

    let x = f16_cuda(&data, &[2, 4], true);
    sum(&cumsum(&x, 1).expect("tracked cumsum f16"))
        .expect("sum")
        .backward()
        .expect("cumsum backward");
    assert_close(
        &host_f16(&x.grad().expect("grad slot").expect("cumsum grad")),
        &[4.0, 3.0, 2.0, 1.0, 4.0, 3.0, 2.0, 1.0],
        0.0,
        "f16 cumsum grad",
    );

    let x = f16_cuda(&data, &[2, 4], true);
    sum(&cumprod(&x, 1).expect("tracked cumprod f16"))
        .expect("sum")
        .backward()
        .expect("cumprod backward");
    assert_close(
        &host_f16(&x.grad().expect("grad slot").expect("cumprod grad")),
        &[-10.0, 5.5, -3.0, -6.0, 53.0, 52.0, 96.0, 32.0],
        0.0,
        "f16 cumprod grad",
    );

    let x = f16_cuda(&data, &[2, 4], true);
    sum(&logcumsumexp(&x, 1).expect("tracked logcumsumexp f16"))
        .expect("sum")
        .backward()
        .expect("logcumsumexp backward");
    assert_close(
        &host_f16(&x.grad().expect("grad slot").expect("logcumsumexp grad")),
        &[
            2.1816, 0.0589, 1.6943, 0.0671, 2.1758, 1.1758, 0.0912, 0.5610,
        ],
        0.04,
        "f16 logcumsumexp grad",
    );

    let x = f16_cuda(&data, &[2, 4], true);
    sum(&cummax(&x, 1).expect("tracked cummax f16").values)
        .expect("sum")
        .backward()
        .expect("cummax backward");
    assert_close(
        &host_f16(&x.grad().expect("grad slot").expect("cummax grad")),
        &[2.0, 0.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0],
        0.0,
        "f16 cummax grad",
    );

    let x = f16_cuda(&data, &[2, 4], true);
    sum(&cummin(&x, 1).expect("tracked cummin f16").values)
        .expect("sum")
        .backward()
        .expect("cummin backward");
    assert_close(
        &host_f16(&x.grad().expect("grad slot").expect("cummin grad")),
        &[1.0, 3.0, 0.0, 0.0, 1.0, 1.0, 2.0, 0.0],
        0.0,
        "f16 cummin grad",
    );
}

#[test]
fn bf16_cumulative_cuda_forward_and_backward() {
    ensure_cuda_backend();
    let data = [1.0, -2.0, 3.0, 0.5, 4.0, 4.0, 2.0, 5.0];
    let x = bf16_cuda(&data, &[2, 4], false);

    assert_close(
        &host_bf16(&cumsum(&x, 1).expect("cumsum bf16")),
        &[1.0, -1.0, 2.0, 2.5, 4.0, 8.0, 10.0, 15.0],
        0.0,
        "bf16 cumsum",
    );
    assert_close(
        &host_bf16(&cumprod(&x, 1).expect("cumprod bf16")),
        &[1.0, -2.0, -6.0, -3.0, 4.0, 16.0, 32.0, 160.0],
        0.0,
        "bf16 cumprod",
    );
    assert_close(
        &host_bf16(&logcumsumexp(&x, 1).expect("logcumsumexp bf16")),
        &[1.0, 1.0469, 3.125, 3.2031, 4.0, 4.6875, 4.75, 5.5938],
        0.05,
        "bf16 logcumsumexp",
    );

    let cmx = cummax(&x, 1).expect("cummax bf16");
    assert_close(
        &host_bf16(&cmx.values),
        &[1.0, 1.0, 3.0, 3.0, 4.0, 4.0, 4.0, 5.0],
        0.0,
        "bf16 cummax values",
    );
    assert_eq!(host_indices(&cmx.indices_tensor), &[0, 0, 2, 2, 0, 1, 1, 3]);
    assert!(
        cmx.indices.is_empty(),
        "CUDA cummax must not populate a host indices cache"
    );

    let cmn = cummin(&x, 1).expect("cummin bf16");
    assert_close(
        &host_bf16(&cmn.values),
        &[1.0, -2.0, -2.0, -2.0, 4.0, 4.0, 2.0, 2.0],
        0.0,
        "bf16 cummin values",
    );
    assert_eq!(host_indices(&cmn.indices_tensor), &[0, 1, 1, 1, 0, 1, 2, 2]);
    assert!(
        cmn.indices.is_empty(),
        "CUDA cummin must not populate a host indices cache"
    );

    let x = bf16_cuda(&data, &[2, 4], true);
    sum(&cumsum(&x, 1).expect("tracked cumsum bf16"))
        .expect("sum")
        .backward()
        .expect("cumsum backward");
    assert_close(
        &host_bf16(&x.grad().expect("grad slot").expect("cumsum grad")),
        &[4.0, 3.0, 2.0, 1.0, 4.0, 3.0, 2.0, 1.0],
        0.0,
        "bf16 cumsum grad",
    );

    let x = bf16_cuda(&data, &[2, 4], true);
    sum(&cumprod(&x, 1).expect("tracked cumprod bf16"))
        .expect("sum")
        .backward()
        .expect("cumprod backward");
    assert_close(
        &host_bf16(&x.grad().expect("grad slot").expect("cumprod grad")),
        &[-10.0, 5.5, -3.0, -6.0, 53.0, 52.0, 96.0, 32.0],
        0.0,
        "bf16 cumprod grad",
    );

    let x = bf16_cuda(&data, &[2, 4], true);
    sum(&logcumsumexp(&x, 1).expect("tracked logcumsumexp bf16"))
        .expect("sum")
        .backward()
        .expect("logcumsumexp backward");
    assert_close(
        &host_bf16(&x.grad().expect("grad slot").expect("logcumsumexp grad")),
        &[
            2.1875, 0.0591, 1.7031, 0.0669, 2.1875, 1.1719, 0.0903, 0.5508,
        ],
        0.08,
        "bf16 logcumsumexp grad",
    );

    let x = bf16_cuda(&data, &[2, 4], true);
    sum(&cummax(&x, 1).expect("tracked cummax bf16").values)
        .expect("sum")
        .backward()
        .expect("cummax backward");
    assert_close(
        &host_bf16(&x.grad().expect("grad slot").expect("cummax grad")),
        &[2.0, 0.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0],
        0.0,
        "bf16 cummax grad",
    );

    let x = bf16_cuda(&data, &[2, 4], true);
    sum(&cummin(&x, 1).expect("tracked cummin bf16").values)
        .expect("sum")
        .backward()
        .expect("cummin backward");
    assert_close(
        &host_bf16(&x.grad().expect("grad slot").expect("cummin grad")),
        &[1.0, 3.0, 0.0, 0.0, 1.0, 1.0, 2.0, 0.0],
        0.0,
        "bf16 cummin grad",
    );
}

#[test]
fn logcumsumexp_cuda_equal_infinities_match_pytorch() {
    ensure_cuda_backend();
    let f32_x = from_vec::<f32>(vec![f32::NEG_INFINITY, f32::NEG_INFINITY, 0.0], &[3])
        .expect("f32")
        .to(Device::Cuda(0))
        .expect("upload f32");
    let f32_y = logcumsumexp(&f32_x, 0).expect("f32 logcumsumexp");
    assert_eq!(
        f32_y.cpu().unwrap().data().unwrap(),
        &[f32::NEG_INFINITY, f32::NEG_INFINITY, 0.0]
    );
    let f32_pos_inf = from_vec::<f32>(vec![0.0, f32::INFINITY], &[2])
        .expect("f32")
        .to(Device::Cuda(0))
        .expect("upload f32");
    assert_eq!(
        logcumsumexp(&f32_pos_inf, 0)
            .expect("f32 pos-inf logcumsumexp")
            .cpu()
            .unwrap()
            .data()
            .unwrap(),
        &[0.0, f32::INFINITY]
    );

    let f64_x = from_vec::<f64>(vec![f64::INFINITY, f64::INFINITY], &[2])
        .expect("f64")
        .to(Device::Cuda(0))
        .expect("upload f64");
    let f64_y = logcumsumexp(&f64_x, 0).expect("f64 logcumsumexp");
    assert_eq!(
        f64_y.cpu().unwrap().data().unwrap(),
        &[f64::INFINITY, f64::INFINITY]
    );
    let f64_neg_inf = from_vec::<f64>(vec![f64::NEG_INFINITY, 0.0], &[2])
        .expect("f64")
        .to(Device::Cuda(0))
        .expect("upload f64");
    assert_eq!(
        logcumsumexp(&f64_neg_inf, 0)
            .expect("f64 neg-inf logcumsumexp")
            .cpu()
            .unwrap()
            .data()
            .unwrap(),
        &[f64::NEG_INFINITY, 0.0]
    );

    let f16_x = from_vec::<half::f16>(
        vec![
            half::f16::NEG_INFINITY,
            half::f16::NEG_INFINITY,
            half::f16::from_f32(0.0),
        ],
        &[3],
    )
    .expect("f16")
    .to(Device::Cuda(0))
    .expect("upload f16");
    assert_eq!(
        host_f16(&logcumsumexp(&f16_x, 0).expect("f16 logcumsumexp")),
        &[f32::NEG_INFINITY, f32::NEG_INFINITY, 0.0]
    );
    let f16_pos_inf =
        from_vec::<half::f16>(vec![half::f16::from_f32(0.0), half::f16::INFINITY], &[2])
            .expect("f16")
            .to(Device::Cuda(0))
            .expect("upload f16");
    assert_eq!(
        host_f16(&logcumsumexp(&f16_pos_inf, 0).expect("f16 pos-inf logcumsumexp")),
        &[0.0, f32::INFINITY]
    );

    let bf16_x = from_vec::<half::bf16>(
        vec![
            half::bf16::NEG_INFINITY,
            half::bf16::NEG_INFINITY,
            half::bf16::from_f32(0.0),
        ],
        &[3],
    )
    .expect("bf16")
    .to(Device::Cuda(0))
    .expect("upload bf16");
    assert_eq!(
        host_bf16(&logcumsumexp(&bf16_x, 0).expect("bf16 logcumsumexp")),
        &[f32::NEG_INFINITY, f32::NEG_INFINITY, 0.0]
    );
    let bf16_pos_inf =
        from_vec::<half::bf16>(vec![half::bf16::from_f32(0.0), half::bf16::INFINITY], &[2])
            .expect("bf16")
            .to(Device::Cuda(0))
            .expect("upload bf16");
    assert_eq!(
        host_bf16(&logcumsumexp(&bf16_pos_inf, 0).expect("bf16 pos-inf logcumsumexp")),
        &[0.0, f32::INFINITY]
    );
}

#[test]
fn cumprod_cuda_backward_zero_segments_match_pytorch() {
    ensure_cuda_backend();
    let data_f32 = [2.0, 0.0, 3.0, 0.0, 5.0, 6.0];
    let expected_f32 = [1.0, 8.0, 0.0, 36.0, 0.0, 0.0];

    let x = f32_cuda(&data_f32, &[2, 3], true);
    sum(&cumprod(&x, 1).expect("cumprod f32 zero"))
        .expect("sum")
        .backward()
        .expect("cumprod f32 zero backward");
    assert_close(
        &host_f32(&x.grad().expect("grad slot").expect("f32 zero grad")),
        &expected_f32,
        0.0,
        "f32 cumprod zero grad",
    );

    let data_f64 = [2.0, 0.0, 3.0, 0.0, 5.0, 6.0];
    let expected_f64 = [1.0, 8.0, 0.0, 36.0, 0.0, 0.0];
    let x = f64_cuda(&data_f64, &[2, 3], true);
    sum(&cumprod(&x, 1).expect("cumprod f64 zero"))
        .expect("sum")
        .backward()
        .expect("cumprod f64 zero backward");
    assert_close_f64(
        &host_f64(&x.grad().expect("grad slot").expect("f64 zero grad")),
        &expected_f64,
        0.0,
        "f64 cumprod zero grad",
    );

    let x = f16_cuda(&data_f32, &[2, 3], true);
    sum(&cumprod(&x, 1).expect("cumprod f16 zero"))
        .expect("sum")
        .backward()
        .expect("cumprod f16 zero backward");
    assert_close(
        &host_f16(&x.grad().expect("grad slot").expect("f16 zero grad")),
        &expected_f32,
        0.0,
        "f16 cumprod zero grad",
    );

    let x = bf16_cuda(&data_f32, &[2, 3], true);
    sum(&cumprod(&x, 1).expect("cumprod bf16 zero"))
        .expect("sum")
        .backward()
        .expect("cumprod bf16 zero backward");
    assert_close(
        &host_bf16(&x.grad().expect("grad slot").expect("bf16 zero grad")),
        &expected_f32,
        0.0,
        "bf16 cumprod zero grad",
    );
}
