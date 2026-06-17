//! Hardening sentinel (GPU dtype-parity epic, crosslink #1185, task #36):
//! `TensorStorage::try_clone` performs a device-to-device copy with no host
//! staging, for every dtype that core can upload to CUDA.
//!
//! Tensor handle `clone()` is intentionally shallow and aliases storage through
//! `Arc<TensorStorage<_>>`, matching PyTorch's Tensor/TensorImpl handle model.
//! This probe exercises the explicit fallible deep-copy API instead:
//!
//! 1. build CUDA storage directly from typed CPU values;
//! 2. call `TensorStorage::try_clone`;
//! 3. verify the clone stays CUDA-resident with the same authoritative dtype tag;
//! 4. verify source and clone have distinct CUDA device pointers; and
//! 5. read back the clone and compare logical bytes exactly.
//!
//! Requires the `gpu` feature plus a real CUDA device.

#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::device::Device;
use ferrotorch_core::dtype::Element;
use ferrotorch_core::gpu_dispatch::gpu_backend;
use ferrotorch_core::storage::TensorStorage;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialise for the storage clone probe");
    });
}

fn record(label: &str, ok: bool, detail: &str, pass: &mut usize, fail: &mut usize) {
    if ok {
        *pass += 1;
        println!("PASS  {label:<32} {detail}");
    } else {
        *fail += 1;
        println!("FAIL  {label:<32} {detail}");
    }
}

fn logical_bytes<T>(values: &[T]) -> Vec<u8> {
    let byte_len = std::mem::size_of_val(values);
    // SAFETY: `values` is a live typed slice. We copy its initialized object
    // representation into a new Vec<u8> for byte-exact comparison only.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), byte_len).to_vec() }
}

fn check_try_clone<T>(label: &str, values: &[T], pass: &mut usize, fail: &mut usize)
where
    T: Element + Copy + std::fmt::Debug,
{
    let result = (|| {
        let backend = gpu_backend().ok_or_else(|| "CUDA backend was not registered".to_string())?;
        let source = TensorStorage::on_device(values.to_vec(), Device::Cuda(0))
            .map_err(|e| format!("upload failed: {e}"))?;
        let cloned = source
            .try_clone()
            .map_err(|e| format!("try_clone failed: {e}"))?;

        let source_handle = source
            .gpu_handle()
            .ok_or_else(|| "source storage is not CUDA-resident".to_string())?;
        let clone_handle = cloned
            .gpu_handle()
            .ok_or_else(|| "cloned storage is not CUDA-resident".to_string())?;

        let source_ptr = backend.raw_device_ptr(source_handle);
        let clone_ptr = backend.raw_device_ptr(clone_handle);
        let resident = cloned.is_gpu() && cloned.device() == Device::Cuda(0);
        let tag = clone_handle.dtype() == T::dtype();
        let len = clone_handle.len() == values.len();
        let distinct_ptrs =
            !source_ptr.is_null() && !clone_ptr.is_null() && source_ptr != clone_ptr;

        let mut got = backend
            .gpu_to_cpu(clone_handle)
            .map_err(|e| format!("readback failed: {e}"))?;
        let logical_len = values
            .len()
            .checked_mul(T::dtype().size_of())
            .ok_or_else(|| "logical byte length overflowed".to_string())?;
        if got.len() < logical_len {
            return Err(format!(
                "readback returned {} bytes, expected at least {logical_len}",
                got.len()
            ));
        }
        got.truncate(logical_len);
        let bytes_match = got == logical_bytes(values);

        Ok((
            resident && tag && len && distinct_ptrs && bytes_match,
            format!(
                "resident={resident} tag={tag} len={len} \
                 distinct_ptrs={distinct_ptrs} bytes_match={bytes_match}"
            ),
        ))
    })();

    match result {
        Ok((ok, detail)) => record(label, ok, &detail, pass, fail),
        Err(detail) => record(label, false, &detail, pass, fail),
    }
}

#[test]
fn probe_phase_hardening_storage_try_clone() {
    ensure_cuda_backend();

    let mut pass = 0usize;
    let mut fail = 0usize;

    check_try_clone(
        "f32 storage try_clone",
        &[1.0f32, -2.5, f32::MIN, f32::MAX, -0.0],
        &mut pass,
        &mut fail,
    );
    check_try_clone(
        "f64 storage try_clone",
        &[1.0f64, -2.5, f64::MIN, f64::MAX, -0.0],
        &mut pass,
        &mut fail,
    );

    let bf16: Vec<half::bf16> = [1.0f32, -2.0, 3.5, -5.25, 0.0]
        .into_iter()
        .map(half::bf16::from_f32)
        .collect();
    check_try_clone("bf16 storage try_clone", &bf16, &mut pass, &mut fail);

    let f16: Vec<half::f16> = [1.0f32, -2.0, 3.5, -5.25, 0.0]
        .into_iter()
        .map(half::f16::from_f32)
        .collect();
    check_try_clone("f16 storage try_clone", &f16, &mut pass, &mut fail);

    check_try_clone(
        "i16 storage try_clone",
        &[1i16, -2, i16::MIN, i16::MAX, 0],
        &mut pass,
        &mut fail,
    );
    check_try_clone(
        "i32 storage try_clone",
        &[1i32, -2, i32::MIN, i32::MAX, 0],
        &mut pass,
        &mut fail,
    );
    check_try_clone(
        "i64 storage try_clone",
        &[1i64, -2, i64::MIN, i64::MAX, 0],
        &mut pass,
        &mut fail,
    );
    check_try_clone(
        "bool storage try_clone",
        &[true, false, true, true, false],
        &mut pass,
        &mut fail,
    );

    println!("PASS: {pass}, FAIL: {fail}");
    assert_eq!(fail, 0, "storage clone hardening probe had {fail} failures");
}
