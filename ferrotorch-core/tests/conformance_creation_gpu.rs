//! CUDA conformance for creation factories whose contract is device-resident.
//!
//! The CPU creation suite covers shape and CPU oracle behavior; this file pins
//! the PyTorch CUDA factory contract for `rand_on_device` / `randn_on_device`:
//! the tensor is allocated on CUDA and filled by GPU RNG kernels, not by a CPU
//! generate-then-upload fallback.

#![cfg(feature = "gpu")]

use ferrotorch_core::creation;
use ferrotorch_core::dtype::Float;
use ferrotorch_core::{Device, Tensor, manual_seed, rand_on_device, randn_on_device};
use half::{bf16, f16};
use std::sync::{Mutex, MutexGuard, Once};

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for creation GPU conformance");
    });
}

fn default_rng_test_lock() -> MutexGuard<'static, ()> {
    static TEST_LOCK: Mutex<()> = Mutex::new(());
    TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn host_values<T: Float>(tensor: &Tensor<T>) -> Vec<f64> {
    tensor
        .to(Device::Cpu)
        .expect("CUDA tensor should copy to CPU for inspection")
        .data_vec()
        .expect("CPU data should be readable")
        .into_iter()
        .map(|x| x.to_f64().expect("float value should convert to f64"))
        .collect()
}

fn assert_cuda_uniform<T: Float>(name: &str, tensor: Tensor<T>, shape: &[usize]) {
    assert_eq!(tensor.device(), Device::Cuda(0), "{name} device");
    assert!(tensor.is_cuda(), "{name} must stay CUDA-resident");
    assert_eq!(tensor.shape(), shape, "{name} shape");

    let values = host_values(&tensor);
    assert_eq!(values.len(), shape.iter().product::<usize>(), "{name} len");
    assert!(
        values.iter().all(|&x| (0.0..1.0).contains(&x)),
        "{name} values must be in [0, 1): {values:?}"
    );
}

fn assert_cuda_normal<T: Float>(name: &str, tensor: Tensor<T>, shape: &[usize]) {
    assert_eq!(tensor.device(), Device::Cuda(0), "{name} device");
    assert!(tensor.is_cuda(), "{name} must stay CUDA-resident");
    assert_eq!(tensor.shape(), shape, "{name} shape");

    let values = host_values(&tensor);
    assert_eq!(values.len(), shape.iter().product::<usize>(), "{name} len");
    assert!(
        values.iter().all(|x| x.is_finite()),
        "{name} values must be finite: {values:?}"
    );
    assert!(
        values.iter().any(|&x| x != 0.0),
        "{name} should not be a fabricated zero fill"
    );
}

#[test]
fn creation_rand_on_device_cuda_float_dtypes_are_resident() {
    let _guard = default_rng_test_lock();
    ensure_cuda_backend();
    manual_seed(1682);
    let shape = [17, 3];

    assert_cuda_uniform(
        "creation::rand_on_device f32",
        creation::rand_on_device::<f32>(&shape, Device::Cuda(0)).expect("rand f32"),
        &shape,
    );
    assert_cuda_uniform(
        "creation::rand_on_device f64",
        creation::rand_on_device::<f64>(&shape, Device::Cuda(0)).expect("rand f64"),
        &shape,
    );
    assert_cuda_uniform(
        "creation::rand_on_device f16",
        creation::rand_on_device::<f16>(&shape, Device::Cuda(0)).expect("rand f16"),
        &shape,
    );
    assert_cuda_uniform(
        "creation::rand_on_device bf16",
        creation::rand_on_device::<bf16>(&shape, Device::Cuda(0)).expect("rand bf16"),
        &shape,
    );
}

#[test]
fn creation_randn_on_device_cuda_float_dtypes_are_resident() {
    let _guard = default_rng_test_lock();
    ensure_cuda_backend();
    manual_seed(1683);
    let shape = [19, 2];

    assert_cuda_normal(
        "creation::randn_on_device f32",
        creation::randn_on_device::<f32>(&shape, Device::Cuda(0)).expect("randn f32"),
        &shape,
    );
    assert_cuda_normal(
        "creation::randn_on_device f64",
        creation::randn_on_device::<f64>(&shape, Device::Cuda(0)).expect("randn f64"),
        &shape,
    );
    assert_cuda_normal(
        "creation::randn_on_device f16",
        creation::randn_on_device::<f16>(&shape, Device::Cuda(0)).expect("randn f16"),
        &shape,
    );
    assert_cuda_normal(
        "creation::randn_on_device bf16",
        creation::randn_on_device::<bf16>(&shape, Device::Cuda(0)).expect("randn bf16"),
        &shape,
    );
}

#[test]
fn top_level_rand_reexports_preserve_cuda_residency() {
    let _guard = default_rng_test_lock();
    ensure_cuda_backend();
    manual_seed(1684);

    assert_cuda_uniform(
        "rand_on_device re-export",
        rand_on_device::<f32>(&[11], Device::Cuda(0)).expect("rand re-export"),
        &[11],
    );
    assert_cuda_normal(
        "randn_on_device re-export",
        randn_on_device::<f32>(&[12], Device::Cuda(0)).expect("randn re-export"),
        &[12],
    );
}
