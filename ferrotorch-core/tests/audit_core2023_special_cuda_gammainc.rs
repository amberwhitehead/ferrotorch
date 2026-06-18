#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::error::FerrotorchError;
use ferrotorch_core::grad_fns::reduction::sum as reduce_sum;
use ferrotorch_core::{Device, Tensor, TensorStorage, gammainc, gammaincc};

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for CORE-2023 gammainc tests");
    });
}

fn cpu_f32(data: Vec<f32>, shape: &[usize], requires_grad: bool) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data), shape.to_vec(), requires_grad)
        .expect("cpu f32 tensor")
}

fn cuda_f32(data: Vec<f32>, shape: &[usize], requires_grad: bool) -> Tensor<f32> {
    cpu_f32(data, shape, false)
        .to(Device::Cuda(0))
        .expect("upload f32")
        .requires_grad_(requires_grad)
}

fn cpu_f64(data: Vec<f64>, shape: &[usize], requires_grad: bool) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data), shape.to_vec(), requires_grad)
        .expect("cpu f64 tensor")
}

fn cuda_f64(data: Vec<f64>, shape: &[usize], requires_grad: bool) -> Tensor<f64> {
    cpu_f64(data, shape, false)
        .to(Device::Cuda(0))
        .expect("upload f64")
        .requires_grad_(requires_grad)
}

fn cuda_f16(data: &[f32], shape: &[usize]) -> Tensor<half::f16> {
    let values: Vec<half::f16> = data.iter().copied().map(half::f16::from_f32).collect();
    Tensor::from_storage(TensorStorage::cpu(values), shape.to_vec(), false)
        .expect("cpu f16 tensor")
        .to(Device::Cuda(0))
        .expect("upload f16")
}

fn cuda_bf16(data: &[f32], shape: &[usize]) -> Tensor<half::bf16> {
    let values: Vec<half::bf16> = data.iter().copied().map(half::bf16::from_f32).collect();
    Tensor::from_storage(TensorStorage::cpu(values), shape.to_vec(), false)
        .expect("cpu bf16 tensor")
        .to(Device::Cuda(0))
        .expect("upload bf16")
}

fn read_cuda_f32(t: &Tensor<f32>, label: &str) -> Vec<f32> {
    assert_eq!(
        t.device(),
        Device::Cuda(0),
        "{label}: expected CUDA-resident tensor, got {:?}",
        t.device()
    );
    t.to(Device::Cpu)
        .expect("download")
        .data_vec()
        .expect("logical data")
}

fn read_cuda_f64(t: &Tensor<f64>, label: &str) -> Vec<f64> {
    assert_eq!(
        t.device(),
        Device::Cuda(0),
        "{label}: expected CUDA-resident tensor, got {:?}",
        t.device()
    );
    t.to(Device::Cpu)
        .expect("download")
        .data_vec()
        .expect("logical data")
}

fn assert_close_or_special_f32(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        if e.is_nan() {
            assert!(a.is_nan(), "{label}[{i}]: expected NaN, got {a:?}");
        } else if e.is_infinite() {
            assert_eq!(a, e, "{label}[{i}]: expected {e:?}, got {a:?}");
        } else {
            let diff = (a - e).abs();
            assert!(
                diff <= tol,
                "{label}[{i}]: expected {e:?}, got {a:?}, diff={diff:?}"
            );
        }
    }
}

fn assert_close_or_special_f64(actual: &[f64], expected: &[f64], tol: f64, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        if e.is_nan() {
            assert!(a.is_nan(), "{label}[{i}]: expected NaN, got {a:?}");
        } else if e.is_infinite() {
            assert_eq!(a, e, "{label}[{i}]: expected {e:?}, got {a:?}");
        } else {
            let diff = (a - e).abs();
            assert!(
                diff <= tol,
                "{label}[{i}]: expected {e:?}, got {a:?}, diff={diff:?}"
            );
        }
    }
}

fn assert_not_implemented_on_cuda<T>(result: Result<T, FerrotorchError>, op: &'static str) {
    match result {
        Err(FerrotorchError::NotImplementedOnCuda { op: got }) => assert_eq!(got, op),
        Err(other) => panic!("expected NotImplementedOnCuda({op}), got {other:?}"),
        Ok(_) => panic!("expected NotImplementedOnCuda({op}), got Ok"),
    }
}

#[test]
fn cuda_gammainc_f32_broadcasts_and_matches_torch_regions() {
    ensure_cuda_backend();

    let a = cuda_f32(vec![2.0, 7.5], &[2, 1], false);
    let x = cuda_f32(vec![1.5, 3.0, 10.0], &[1, 3], false);

    let p = gammainc(&a, &x).expect("gammainc f32");
    let q = gammaincc(&a, &x).expect("gammaincc f32");
    assert_eq!(p.shape(), &[2, 3]);
    assert_eq!(q.shape(), &[2, 3]);

    // Local PyTorch 2.11 CUDA/CPU oracle, dtype=torch.float32, 2026-06-18.
    assert_close_or_special_f32(
        &read_cuda_f32(&p, "gammainc f32"),
        &[
            0.442_174_6,
            0.800_851_7,
            0.999_500_6,
            0.000_402_198_6,
            0.020_252_265,
            0.828_067_4,
        ],
        7e-4,
        "gammainc f32",
    );
    assert_close_or_special_f32(
        &read_cuda_f32(&q, "gammaincc f32"),
        &[
            0.557_825_4,
            0.199_148_28,
            0.000_499_399_26,
            0.999_597_8,
            0.979_747_7,
            0.171_932_64,
        ],
        7e-4,
        "gammaincc f32",
    );
}

#[test]
fn cuda_gammainc_rejects_half_and_bfloat_like_torch() {
    ensure_cuda_backend();

    // Local PyTorch 2.11 CUDA oracle, 2026-06-18:
    // torch.special.gammainc/gammaincc raise NotImplementedError for Half and
    // BFloat16 ("igamma_cuda"/"igammac_cuda" not implemented). Ferrotorch must
    // reject these dtypes cleanly instead of silently routing through CPU.
    let a_f16 = cuda_f16(&[0.5, 2.0], &[2]);
    let x_f16 = cuda_f16(&[0.5, 1.5], &[2]);
    assert_not_implemented_on_cuda(gammainc(&a_f16, &x_f16), "gammainc");
    assert_not_implemented_on_cuda(gammaincc(&a_f16, &x_f16), "gammaincc");

    let a_bf16 = cuda_bf16(&[0.5, 2.0], &[2]);
    let x_bf16 = cuda_bf16(&[0.5, 1.5], &[2]);
    assert_not_implemented_on_cuda(gammainc(&a_bf16, &x_bf16), "gammainc");
    assert_not_implemented_on_cuda(gammaincc(&a_bf16, &x_bf16), "gammaincc");
}

#[test]
fn cuda_gammainc_f64_asymptotic_and_boundaries_match_torch() {
    ensure_cuda_backend();

    let a = cuda_f64(vec![50.0, 250.0], &[2], false);
    let x = cuda_f64(vec![50.0, 260.0], &[2], false);
    assert_close_or_special_f64(
        &read_cuda_f64(
            &gammainc(&a, &x).expect("gammainc asym f64"),
            "gammainc f64",
        ),
        &[0.518_808_315_210_348_8, 0.740_610_517_296_988_6],
        3e-3,
        "gammainc asym f64",
    );
    assert_close_or_special_f64(
        &read_cuda_f64(
            &gammaincc(&a, &x).expect("gammaincc asym f64"),
            "gammaincc f64",
        ),
        &[0.481_191_684_789_651_24, 0.259_389_482_703_011_4],
        3e-3,
        "gammaincc asym f64",
    );

    let a = cuda_f64(
        vec![
            0.0,
            2.0,
            -1.0,
            2.0,
            0.0,
            f64::INFINITY,
            f64::INFINITY,
            2.0,
            f64::NAN,
        ],
        &[9],
        false,
    );
    let x = cuda_f64(
        vec![
            2.0,
            0.0,
            2.0,
            -1.0,
            0.0,
            2.0,
            f64::INFINITY,
            f64::INFINITY,
            2.0,
        ],
        &[9],
        false,
    );
    assert_close_or_special_f64(
        &read_cuda_f64(
            &gammainc(&a, &x).expect("gammainc boundary f64"),
            "gammainc boundary",
        ),
        &[
            1.0,
            0.0,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            0.0,
            f64::NAN,
            1.0,
            f64::NAN,
        ],
        0.0,
        "gammainc boundary",
    );
    assert_close_or_special_f64(
        &read_cuda_f64(
            &gammaincc(&a, &x).expect("gammaincc boundary f64"),
            "gammaincc boundary",
        ),
        &[
            0.0,
            1.0,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            1.0,
            f64::NAN,
            0.0,
            f64::NAN,
        ],
        0.0,
        "gammaincc boundary",
    );
}

#[test]
fn cuda_gammainc_backward_matches_pytorch_x_grad_and_rejects_a_grad() {
    ensure_cuda_backend();

    let a = cuda_f32(vec![0.5, 2.0, 4.0], &[3], false);
    let x = cuda_f32(vec![0.5, 1.5, 3.0], &[3], true);
    reduce_sum(&gammainc(&a, &x).expect("gammainc backward f32"))
        .expect("sum gammainc")
        .backward()
        .expect("gammainc backward");
    let gx = x.grad().expect("x grad lookup").expect("x grad");
    assert_close_or_special_f32(
        &read_cuda_f32(&gx, "gammainc x grad"),
        &[0.483_941_47, 0.334_695_25, 0.224_041_82],
        5e-4,
        "gammainc x grad",
    );

    let a = cuda_f64(vec![0.5, 2.0, 4.0], &[3], false);
    let x = cuda_f64(vec![0.5, 1.5, 3.0], &[3], true);
    reduce_sum(&gammaincc(&a, &x).expect("gammaincc backward f64"))
        .expect("sum gammaincc")
        .backward()
        .expect("gammaincc backward");
    let gx = x.grad().expect("x grad lookup").expect("x grad");
    assert_close_or_special_f64(
        &read_cuda_f64(&gx, "gammaincc x grad"),
        &[
            -0.483_941_449_038_286_7,
            -0.334_695_240_222_644_74,
            -0.224_041_807_655_387_75,
        ],
        1e-6,
        "gammaincc x grad",
    );

    let a = cuda_f32(vec![2.0], &[1], true);
    let x = cuda_f32(vec![1.5], &[1], false);
    let err = reduce_sum(&gammainc(&a, &x).expect("gammainc a grad forward"))
        .expect("sum gammainc a")
        .backward()
        .expect_err("PyTorch does not implement igamma gradient for input/a");
    match err {
        FerrotorchError::InvalidArgument { message } => {
            assert!(
                message.contains("igamma: input"),
                "wrong a-gradient error message: {message}"
            );
        }
        other => panic!("expected InvalidArgument for igamma input grad, got {other:?}"),
    }
}
