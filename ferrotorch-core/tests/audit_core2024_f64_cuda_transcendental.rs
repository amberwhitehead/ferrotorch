#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::{Device, Tensor, TensorStorage, log, xlogy};

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for CORE-2024 tests");
    });
}

fn cpu_f64(data: Vec<f64>, shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data), shape.to_vec(), false).expect("cpu f64 tensor")
}

fn cuda_f64(data: Vec<f64>, shape: &[usize]) -> Tensor<f64> {
    cpu_f64(data, shape)
        .to(Device::Cuda(0))
        .expect("upload f64")
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

fn assert_close_or_special(actual: &[f64], expected: &[f64], tol: f64, label: &str) {
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

#[test]
fn cuda_log_f64_matches_torch_cuda_specials_and_subnormals() {
    ensure_cuda_backend();

    let input = cuda_f64(
        vec![
            0.0,
            -0.0,
            -1.0,
            1e-320,
            1e-300,
            1.0,
            f64::from_bits(0x3ff0000000000001),
            f64::from_bits(0x3fefffffffffffff),
            2.0,
            f64::INFINITY,
            f64::NAN,
        ],
        &[11],
    );
    let out = log(&input).expect("cuda f64 log");
    let got = read_cuda_f64(&out, "log f64");

    // PyTorch 2.11 CUDA f64 oracle on the same inputs.
    let expected = vec![
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        -736.8272408909739,
        -690.7755278982137,
        0.0,
        2.2204460492503128e-16,
        -1.1102230246251565e-16,
        std::f64::consts::LN_2,
        f64::INFINITY,
        f64::NAN,
    ];
    assert_close_or_special(&got, &expected, 2e-13, "log f64");
}

#[test]
fn cuda_xlogy_f64_uses_double_precision_log_and_torch_branch_order() {
    ensure_cuda_backend();

    let x = cuda_f64(vec![1.0, 2.0, 0.0, 0.0, 3.0, 1.0, 1.0], &[7]);
    let y = cuda_f64(
        vec![
            f64::from_bits(0x3ff0000000000001),
            f64::from_bits(0x3ff0000000000001),
            f64::NAN,
            0.0,
            -1.0,
            f64::INFINITY,
            1e-300,
        ],
        &[7],
    );
    let out = xlogy(&x, &y).expect("cuda f64 xlogy");
    let got = read_cuda_f64(&out, "xlogy f64");

    // PyTorch 2.11 CUDA f64 oracle on the same inputs.
    let expected = vec![
        2.2204460492503128e-16,
        4.4408920985006257e-16,
        f64::NAN,
        0.0,
        f64::NAN,
        f64::INFINITY,
        -690.7755278982137,
    ];
    assert_close_or_special(&got, &expected, 2e-13, "xlogy f64");
}
