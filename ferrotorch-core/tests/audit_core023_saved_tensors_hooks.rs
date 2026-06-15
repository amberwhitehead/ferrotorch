use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

#[cfg(feature = "gpu")]
use ferrotorch_core::Device;
use ferrotorch_core::autograd::saved_tensors::saved_tensors_hooks;
use ferrotorch_core::error::FerrotorchError;
use ferrotorch_core::grad_fns::arithmetic::mul;
use ferrotorch_core::grad_fns::reduction::sum;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;
use half::bf16;

#[cfg(feature = "gpu")]
static GPU_INIT: std::sync::Once = std::sync::Once::new();

#[cfg(feature = "gpu")]
fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for saved tensor hook CUDA probes");
    });
}

#[test]
fn production_autograd_saved_tensors_use_pack_and_unpack_hooks() {
    let pack_count = Arc::new(AtomicUsize::new(0));
    let unpack_count = Arc::new(AtomicUsize::new(0));

    let x = Tensor::from_storage(TensorStorage::cpu(vec![2.0f32, 3.0]), vec![2], true).unwrap();
    let y = Tensor::from_storage(TensorStorage::cpu(vec![5.0f32, 7.0]), vec![2], true).unwrap();

    let pack_seen = Arc::clone(&pack_count);
    let unpack_seen = Arc::clone(&unpack_count);
    let product = saved_tensors_hooks(
        move |tensor: Tensor<f32>| {
            pack_seen.fetch_add(1, Ordering::SeqCst);
            Ok(tensor)
        },
        move |tensor: Tensor<f32>| {
            unpack_seen.fetch_add(1, Ordering::SeqCst);
            Ok(tensor)
        },
        || mul(&x, &y),
    )
    .unwrap();

    assert_eq!(
        pack_count.load(Ordering::SeqCst),
        2,
        "MulBackward must pack both saved forward inputs while hooks are active"
    );
    assert_eq!(
        unpack_count.load(Ordering::SeqCst),
        0,
        "saved tensors should not unpack until backward reads them"
    );

    let loss = sum(&product).unwrap();
    loss.backward().unwrap();

    assert_eq!(
        unpack_count.load(Ordering::SeqCst),
        2,
        "MulBackward must unpack both saved tensors during backward after the hook scope exits"
    );
    assert_eq!(
        x.grad().unwrap().unwrap().data_vec().unwrap(),
        vec![5.0, 7.0]
    );
    assert_eq!(
        y.grad().unwrap().unwrap().data_vec().unwrap(),
        vec![2.0, 3.0]
    );
}

#[test]
fn detached_pack_payload_preserves_original_autograd_metadata() {
    let pack_count = Arc::new(AtomicUsize::new(0));
    let unpack_count = Arc::new(AtomicUsize::new(0));

    let x = Tensor::from_storage(TensorStorage::cpu(vec![2.0f32, 3.0]), vec![2], true).unwrap();
    let y = Tensor::from_storage(TensorStorage::cpu(vec![5.0f32, 7.0]), vec![2], true).unwrap();

    let pack_seen = Arc::clone(&pack_count);
    let unpack_seen = Arc::clone(&unpack_count);
    let product = saved_tensors_hooks(
        move |tensor: Tensor<f32>| {
            pack_seen.fetch_add(1, Ordering::SeqCst);
            Ok(tensor.detach())
        },
        move |tensor: Tensor<f32>| {
            unpack_seen.fetch_add(1, Ordering::SeqCst);
            Ok(tensor)
        },
        || mul(&x, &y),
    )
    .unwrap();

    let loss = sum(&product).unwrap();
    loss.backward().unwrap();

    assert_eq!(pack_count.load(Ordering::SeqCst), 2);
    assert_eq!(unpack_count.load(Ordering::SeqCst), 2);
    assert_eq!(
        x.grad().unwrap().unwrap().data_vec().unwrap(),
        vec![5.0, 7.0],
        "packing a detached payload must not make the saved original look non-differentiable"
    );
    assert_eq!(
        y.grad().unwrap().unwrap().data_vec().unwrap(),
        vec![2.0, 3.0]
    );
}

#[test]
fn pack_hook_mutating_saved_input_errors_in_forward() {
    let x = Tensor::from_storage(TensorStorage::cpu(vec![2.0f32, 3.0]), vec![2], true).unwrap();
    let y = Tensor::from_storage(TensorStorage::cpu(vec![5.0f32, 7.0]), vec![2], true).unwrap();

    let result = saved_tensors_hooks(
        |tensor: Tensor<f32>| {
            let values = unsafe { tensor.data_mut()? };
            values[0] = 99.0;
            Ok(tensor)
        },
        |tensor: Tensor<f32>| Ok(tensor),
        || mul(&x, &y),
    );

    let Err(FerrotorchError::InvalidArgument { message }) = result else {
        panic!("mutating a tensor inside pack_hook must fail during forward");
    };
    assert!(
        message.contains("pack hook modified its input in place"),
        "unexpected error: {message}"
    );
}

#[test]
fn production_bf16_saved_tensors_use_hooks() {
    let pack_count = Arc::new(AtomicUsize::new(0));
    let unpack_count = Arc::new(AtomicUsize::new(0));

    let x = Tensor::from_storage(
        TensorStorage::cpu(vec![bf16::from_f32(2.0), bf16::from_f32(3.0)]),
        vec![2],
        true,
    )
    .unwrap();
    let y = Tensor::from_storage(
        TensorStorage::cpu(vec![bf16::from_f32(5.0), bf16::from_f32(7.0)]),
        vec![2],
        true,
    )
    .unwrap();

    let pack_seen = Arc::clone(&pack_count);
    let unpack_seen = Arc::clone(&unpack_count);
    let product = saved_tensors_hooks(
        move |tensor: Tensor<bf16>| {
            pack_seen.fetch_add(1, Ordering::SeqCst);
            Ok(tensor.detach())
        },
        move |tensor: Tensor<bf16>| {
            unpack_seen.fetch_add(1, Ordering::SeqCst);
            Ok(tensor)
        },
        || mul(&x, &y),
    )
    .unwrap();

    let loss = sum(&product).unwrap();
    loss.backward().unwrap();

    assert_eq!(pack_count.load(Ordering::SeqCst), 2);
    assert_eq!(unpack_count.load(Ordering::SeqCst), 2);
    let grad_x: Vec<f32> = x
        .grad()
        .unwrap()
        .unwrap()
        .data_vec()
        .unwrap()
        .into_iter()
        .map(f32::from)
        .collect();
    assert_eq!(grad_x, vec![5.0, 7.0]);
}

#[test]
fn pack_hook_runs_with_grad_disabled() {
    let grad_enabled_seen = Arc::new(AtomicBool::new(true));

    let x = Tensor::from_storage(TensorStorage::cpu(vec![2.0f32, 3.0]), vec![2], true).unwrap();
    let y = Tensor::from_storage(TensorStorage::cpu(vec![5.0f32, 7.0]), vec![2], true).unwrap();

    let seen = Arc::clone(&grad_enabled_seen);
    let product = saved_tensors_hooks(
        move |tensor: Tensor<f32>| {
            seen.store(ferrotorch_core::is_grad_enabled(), Ordering::SeqCst);
            Ok(tensor.detach())
        },
        |tensor: Tensor<f32>| Ok(tensor),
        || mul(&x, &y),
    )
    .unwrap();

    assert!(
        !grad_enabled_seen.load(Ordering::SeqCst),
        "pack_hook must run under no_grad so offload/compression does not build autograd edges"
    );
    sum(&product).unwrap().backward().unwrap();
    assert_eq!(
        x.grad().unwrap().unwrap().data_vec().unwrap(),
        vec![5.0, 7.0]
    );
}

#[cfg(feature = "gpu")]
#[test]
fn cuda_saved_tensor_hooks_offload_and_restore_without_cpu_grad_fallback() {
    ensure_cuda_backend();

    let pack_count = Arc::new(AtomicUsize::new(0));
    let unpack_count = Arc::new(AtomicUsize::new(0));

    let x = Tensor::from_storage(TensorStorage::cpu(vec![2.0f32, 3.0]), vec![2], false)
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap()
        .requires_grad_(true);
    let y = Tensor::from_storage(TensorStorage::cpu(vec![5.0f32, 7.0]), vec![2], false)
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap()
        .requires_grad_(true);

    let pack_seen = Arc::clone(&pack_count);
    let unpack_seen = Arc::clone(&unpack_count);
    let product = saved_tensors_hooks(
        move |tensor: Tensor<f32>| {
            pack_seen.fetch_add(1, Ordering::SeqCst);
            assert_eq!(tensor.device(), Device::Cuda(0));
            tensor.to(Device::Cpu)
        },
        move |tensor: Tensor<f32>| {
            unpack_seen.fetch_add(1, Ordering::SeqCst);
            assert_eq!(tensor.device(), Device::Cpu);
            tensor.to(Device::Cuda(0))
        },
        || mul(&x, &y),
    )
    .unwrap();

    assert_eq!(pack_count.load(Ordering::SeqCst), 2);
    assert_eq!(product.device(), Device::Cuda(0));

    let loss = sum(&product).unwrap();
    loss.backward().unwrap();

    assert_eq!(unpack_count.load(Ordering::SeqCst), 2);
    let x_grad = x.grad().unwrap().unwrap();
    let y_grad = y.grad().unwrap().unwrap();
    assert_eq!(
        x_grad.device(),
        Device::Cuda(0),
        "gradient must remain CUDA-resident; CPU fallback is not acceptable"
    );
    assert_eq!(y_grad.device(), Device::Cuda(0));
    assert_eq!(x_grad.cpu().unwrap().data_vec().unwrap(), vec![5.0, 7.0]);
    assert_eq!(y_grad.cpu().unwrap().data_vec().unwrap(), vec![2.0, 3.0]);
}
