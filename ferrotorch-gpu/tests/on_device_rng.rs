//! On-device `rand_on_device` / `randn_on_device` verification (#1682).
//!
//! These tests confirm that `ferrotorch_core::rand_on_device` /
//! `randn_on_device` for `Device::Cuda` generate values DIRECTLY on the GPU
//! (the result is an `is_cuda()` tensor produced by the Philox kernel, NOT a
//! CPU generate-then-upload), that the distributions are correct, and that
//! the output is reproducible after `manual_seed`.
//!
//! PyTorch parity anchor: `torch.rand(size, device='cuda')` =
//! `at::empty(size, options).uniform_(0, 1)`
//! (`aten/src/ATen/native/TensorFactories.cpp:1075-1076`); the tensor is
//! created on the CUDA device and filled by the on-device Philox
//! kernel. `torch.manual_seed` seeds BOTH the CPU and all CUDA generators
//! (`torch/random.py:67` -> `torch.cuda.manual_seed_all`,
//! `torch/cuda/random.py:112`).

#[cfg(feature = "cuda")]
use ferrotorch_core::randn_on_device;
use ferrotorch_core::{Device, manual_seed, rand_on_device};
#[cfg(feature = "cuda")]
use ferrotorch_gpu::init_cuda_backend;
#[cfg(feature = "cuda")]
use half::{bf16, f16};

#[cfg(feature = "cuda")]
fn ensure_init() {
    if !ferrotorch_core::gpu_dispatch::has_gpu_backend() {
        init_cuda_backend().expect("init_cuda_backend");
    }
}

/// Move a CUDA tensor to CPU and read its values for statistical checks.
#[cfg(feature = "cuda")]
fn to_host(t: &ferrotorch_core::Tensor<f32>) -> Vec<f32> {
    let cpu = t.to(Device::Cpu).expect("tensor.to(Cpu)");
    cpu.data().expect("cpu data").to_vec()
}

#[cfg(feature = "cuda")]
fn to_host_f64<T: ferrotorch_core::dtype::Float>(t: &ferrotorch_core::Tensor<T>) -> Vec<f64> {
    let cpu = t.to(Device::Cpu).expect("tensor.to(Cpu)");
    cpu.data()
        .expect("cpu data")
        .iter()
        .map(|x| x.to_f64().expect("floating value converts to f64"))
        .collect()
}

#[cfg(feature = "cuda")]
#[test]
fn rand_on_device_cuda_is_on_device() {
    ensure_init();
    let t = rand_on_device::<f32>(&[256], Device::Cuda(0)).expect("rand_on_device");
    assert!(
        t.is_cuda(),
        "rand_on_device(Cuda) must return an is_cuda() tensor (on-device buffer)"
    );
    assert_eq!(t.shape(), &[256]);
    assert_eq!(t.device(), Device::Cuda(0));
}

#[cfg(feature = "cuda")]
#[test]
fn randn_on_device_cuda_is_on_device() {
    ensure_init();
    let t = randn_on_device::<f32>(&[256], Device::Cuda(0)).expect("randn_on_device");
    assert!(
        t.is_cuda(),
        "randn_on_device(Cuda) must return an is_cuda() tensor"
    );
    assert_eq!(t.shape(), &[256]);
}

#[cfg(feature = "cuda")]
#[test]
fn rand_on_device_cuda_non_f32_dtypes_are_on_device() {
    ensure_init();

    let f64_t = rand_on_device::<f64>(&[257], Device::Cuda(0)).expect("rand f64 cuda");
    assert!(f64_t.is_cuda(), "f64 rand must stay on CUDA");
    assert_eq!(f64_t.device(), Device::Cuda(0));

    let f16_t = rand_on_device::<f16>(&[257], Device::Cuda(0)).expect("rand f16 cuda");
    assert!(f16_t.is_cuda(), "f16 rand must stay on CUDA");
    assert_eq!(f16_t.device(), Device::Cuda(0));

    let bf16_t = rand_on_device::<bf16>(&[257], Device::Cuda(0)).expect("rand bf16 cuda");
    assert!(bf16_t.is_cuda(), "bf16 rand must stay on CUDA");
    assert_eq!(bf16_t.device(), Device::Cuda(0));
}

#[cfg(feature = "cuda")]
#[test]
fn randn_on_device_cuda_non_f32_dtypes_are_on_device() {
    ensure_init();

    let f64_t = randn_on_device::<f64>(&[257], Device::Cuda(0)).expect("randn f64 cuda");
    assert!(f64_t.is_cuda(), "f64 randn must stay on CUDA");
    assert_eq!(f64_t.device(), Device::Cuda(0));

    let f16_t = randn_on_device::<f16>(&[257], Device::Cuda(0)).expect("randn f16 cuda");
    assert!(f16_t.is_cuda(), "f16 randn must stay on CUDA");
    assert_eq!(f16_t.device(), Device::Cuda(0));

    let bf16_t = randn_on_device::<bf16>(&[257], Device::Cuda(0)).expect("randn bf16 cuda");
    assert!(bf16_t.is_cuda(), "bf16 randn must stay on CUDA");
    assert_eq!(bf16_t.device(), Device::Cuda(0));
}

#[cfg(feature = "cuda")]
#[test]
fn rand_on_device_uniform_distribution() {
    ensure_init();
    // 1M samples: uniform [0,1) => mean ~= 0.5, every value in [0,1).
    let n = 1_000_000usize;
    let t = rand_on_device::<f32>(&[n], Device::Cuda(0)).expect("rand_on_device");
    let v = to_host(&t);
    assert_eq!(v.len(), n);

    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    for &x in &v {
        assert!(
            (0.0..1.0).contains(&x),
            "uniform value {x} out of [0,1) range"
        );
        min = min.min(x);
        max = max.max(x);
        sum += x as f64;
    }
    let mean = sum / n as f64;
    assert!(
        (mean - 0.5).abs() < 0.01,
        "uniform mean {mean} should be ~= 0.5"
    );
    // With 1M samples we expect min near 0 and max near 1.
    assert!(min < 0.01, "uniform min {min} should be near 0");
    assert!(max > 0.99, "uniform max {max} should be near 1");
}

#[cfg(feature = "cuda")]
#[test]
fn rand_on_device_non_f32_uniform_distribution() {
    ensure_init();
    let n = 262_144usize;

    for (name, values) in [
        (
            "f64",
            to_host_f64(&rand_on_device::<f64>(&[n], Device::Cuda(0)).expect("rand f64")),
        ),
        (
            "f16",
            to_host_f64(&rand_on_device::<f16>(&[n], Device::Cuda(0)).expect("rand f16")),
        ),
        (
            "bf16",
            to_host_f64(&rand_on_device::<bf16>(&[n], Device::Cuda(0)).expect("rand bf16")),
        ),
    ] {
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        let mut sum = 0.0f64;
        for &x in &values {
            assert!(
                (0.0..1.0).contains(&x),
                "{name} uniform value {x} out of [0,1)"
            );
            min = min.min(x);
            max = max.max(x);
            sum += x;
        }
        let mean = sum / n as f64;
        assert!(
            (mean - 0.5).abs() < 0.02,
            "{name} uniform mean {mean} should be ~= 0.5"
        );
        assert!(min < 0.02, "{name} uniform min {min} should be near 0");
        assert!(max > 0.98, "{name} uniform max {max} should be near 1");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn randn_on_device_normal_distribution() {
    ensure_init();
    // 1M samples: standard normal => mean ~= 0, std ~= 1.
    let n = 1_000_000usize;
    let t = randn_on_device::<f32>(&[n], Device::Cuda(0)).expect("randn_on_device");
    let v = to_host(&t);
    assert_eq!(v.len(), n);

    let mut sum = 0.0f64;
    for &x in &v {
        assert!(x.is_finite(), "normal value must be finite, got {x}");
        sum += x as f64;
    }
    let mean = sum / n as f64;
    let mut var = 0.0f64;
    for &x in &v {
        let d = x as f64 - mean;
        var += d * d;
    }
    let std = (var / n as f64).sqrt();
    assert!((mean).abs() < 0.02, "normal mean {mean} should be ~= 0");
    assert!((std - 1.0).abs() < 0.05, "normal std {std} should be ~= 1");
}

#[cfg(feature = "cuda")]
#[test]
fn randn_on_device_non_f32_normal_distribution() {
    ensure_init();
    let n = 262_144usize;

    for (name, values) in [
        (
            "f64",
            to_host_f64(&randn_on_device::<f64>(&[n], Device::Cuda(0)).expect("randn f64")),
        ),
        (
            "f16",
            to_host_f64(&randn_on_device::<f16>(&[n], Device::Cuda(0)).expect("randn f16")),
        ),
        (
            "bf16",
            to_host_f64(&randn_on_device::<bf16>(&[n], Device::Cuda(0)).expect("randn bf16")),
        ),
    ] {
        let sum = values.iter().copied().sum::<f64>();
        for &x in &values {
            assert!(x.is_finite(), "{name} normal value must be finite, got {x}");
        }
        let mean = sum / n as f64;
        let var = values
            .iter()
            .map(|&x| {
                let d = x - mean;
                d * d
            })
            .sum::<f64>()
            / n as f64;
        let std = var.sqrt();
        assert!(
            mean.abs() < 0.03,
            "{name} normal mean {mean} should be ~= 0"
        );
        assert!(
            (std - 1.0).abs() < 0.06,
            "{name} normal std {std} should be ~= 1"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn f64_rand_then_randn_uses_distinct_cuda_kernels() {
    ensure_init();
    let n = 65_536usize;
    let _ = rand_on_device::<f64>(&[n], Device::Cuda(0)).expect("rand f64 warms uniform kernel");
    let normal =
        to_host_f64(&randn_on_device::<f64>(&[n], Device::Cuda(0)).expect("randn f64 cuda"));
    let mean = normal.iter().copied().sum::<f64>() / n as f64;
    assert!(
        mean.abs() < 0.04,
        "f64 randn after f64 rand must launch the normal kernel, got mean {mean}"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn rand_on_device_reproducible_same_seed() {
    ensure_init();
    let n = 4096usize;

    manual_seed(42);
    let a = to_host(&rand_on_device::<f32>(&[n], Device::Cuda(0)).expect("rand a"));

    manual_seed(42);
    let b = to_host(&rand_on_device::<f32>(&[n], Device::Cuda(0)).expect("rand b"));

    assert_eq!(
        a, b,
        "manual_seed(42) before rand_on_device(Cuda) must be reproducible bit-for-bit"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn rand_on_device_differs_with_different_seed() {
    ensure_init();
    let n = 4096usize;

    manual_seed(42);
    let a = to_host(&rand_on_device::<f32>(&[n], Device::Cuda(0)).expect("rand a"));

    manual_seed(1337);
    let b = to_host(&rand_on_device::<f32>(&[n], Device::Cuda(0)).expect("rand b"));

    assert_ne!(
        a, b,
        "different seeds must produce different on-device random streams"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn randn_on_device_reproducible_same_seed() {
    ensure_init();
    let n = 4096usize;

    manual_seed(7);
    let a = to_host(&randn_on_device::<f32>(&[n], Device::Cuda(0)).expect("randn a"));

    manual_seed(7);
    let b = to_host(&randn_on_device::<f32>(&[n], Device::Cuda(0)).expect("randn b"));

    assert_eq!(
        a, b,
        "manual_seed(7) before randn_on_device(Cuda) must be reproducible"
    );
}

#[test]
fn rand_on_device_cpu_matches_plain_rand() {
    // For Device::Cpu, rand_on_device must be identical to the existing
    // byte-exact-with-torch CPU `rand` (no behaviour change on the CPU path).
    manual_seed(99);
    let a = rand_on_device::<f32>(&[1000], Device::Cpu).expect("rand_on_device cpu");
    manual_seed(99);
    let b = ferrotorch_core::rand::<f32>(&[1000]).expect("rand cpu");
    assert!(!a.is_cuda());
    assert_eq!(
        a.data().unwrap().to_vec(),
        b.data().unwrap().to_vec(),
        "rand_on_device(Cpu) must equal the existing CPU rand path"
    );
}
