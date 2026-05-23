//! Hardening sentinel (GPU dtype-parity epic, crosslink #1185, task #36):
//! `CudaBackendImpl::clone_buffer` performs a DEVICE-TO-DEVICE copy with NO
//! host round trip, for EVERY supported dtype.
//!
//! Cloning a GPU tensor is a universal hot op — every `.clone()` of a
//! `Tensor` / `IntTensor` / `BoolTensor` resident on CUDA routes through
//! `TensorStorage::clone` -> `backend.clone_buffer`. The old implementation
//! did a silent GPU->CPU->GPU round trip (host RAM crossed twice). The fix
//! uses cudarc's `CudaSlice::try_clone` (alloc + `memcpy_dtod_async`), so the
//! bytes never leave VRAM.
//!
//! What this probe asserts, for each of f32 / f64 / bf16 / f16 / i32 / i64 /
//! bool:
//!   1. A tensor built on CUDA(0), then `.clone()`d, is itself CUDA-resident
//!      (`is_cuda()` + `device() == Cuda(0)` — no silent CPU detour).
//!   2. The clone carries the SAME `DType` tag as the source (PyTorch parity:
//!      a clone preserves the ScalarType, never re-infers it).
//!   3. The clone is a faithful DEEP COPY: reading it back to host yields the
//!      source values bit-exact (f16/bf16 compared on their stored bit
//!      pattern).
//!   4. INDEPENDENT ALLOCATION (observable, dtype-agnostic): after cloning,
//!      the SOURCE is dropped and several "poison" tensors of the same dtype
//!      and length are allocated on CUDA. If the clone had aliased the
//!      source's storage, the dropped source's slot would be reused by a
//!      poison tensor and corrupt the clone. The clone is read back AFTER all
//!      this and must STILL equal the source values — proving the clone owns a
//!      separate device allocation.
//!
//! Prints a PASS/FAIL table ending `PASS: N, FAIL: 0`. Requires the `gpu`
//! feature + a real CUDA device (run on the host RTX 3090).

#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::bool_tensor::BoolTensor;
use ferrotorch_core::creation::from_vec;
use ferrotorch_core::device::Device;
use ferrotorch_core::int_tensor::IntTensor;
use ferrotorch_core::DType;

static GPU_INIT: Once = Once::new();

/// How many same-size "poison" tensors to allocate after dropping the source,
/// to maximise the chance a reused slot would clobber an aliasing clone.
const POISON_ALLOCS: usize = 8;

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialise for the clone-hardening probe");
    });
}

fn record(label: &str, ok: bool, detail: &str, pass: &mut usize, fail: &mut usize) {
    if ok {
        *pass += 1;
        println!("PASS  {label:<28} {detail}");
    } else {
        *fail += 1;
        println!("FAIL  {label:<28} {detail}");
    }
}

fn main_check(pass: &mut usize, fail: &mut usize) {
    // -- f32 -----------------------------------------------------------------
    {
        let src_vals = vec![1.0f32, -2.5, 3.25, 4.0, f32::MIN, f32::MAX, 0.0, -0.0];
        let n = src_vals.len();
        let src = from_vec::<f32>(src_vals.clone(), &[n])
            .expect("f32 cpu")
            .to(Device::Cuda(0))
            .expect("f32 to cuda");
        let cloned = src.clone();

        let resident = cloned.is_cuda() && matches!(cloned.device(), Device::Cuda(0));
        let tag = cloned.gpu_handle().expect("f32 clone handle").dtype() == DType::F32;

        // Independent allocation: drop source, allocate poison, read clone last.
        drop(src);
        let mut poison = Vec::new();
        for _ in 0..POISON_ALLOCS {
            poison.push(
                from_vec::<f32>(vec![999.0f32; n], &[n])
                    .expect("f32 poison")
                    .to(Device::Cuda(0))
                    .expect("f32 poison to cuda"),
            );
        }
        let clone_host = cloned
            .to(Device::Cpu)
            .expect("f32 clone to cpu")
            .data()
            .expect("f32 clone data")
            .to_vec();
        drop(poison);
        let deep_copy_and_independent = clone_host == src_vals;

        record(
            "f32 clone",
            resident && tag && deep_copy_and_independent,
            &format!("resident={resident} tag={tag} deep_copy_independent={deep_copy_and_independent}"),
            pass,
            fail,
        );
    }

    // -- f64 -----------------------------------------------------------------
    {
        let src_vals = vec![1.0f64, -2.5, 3.25, f64::MIN, f64::MAX, 0.0];
        let n = src_vals.len();
        let src = from_vec::<f64>(src_vals.clone(), &[n])
            .expect("f64 cpu")
            .to(Device::Cuda(0))
            .expect("f64 to cuda");
        let cloned = src.clone();

        let resident = cloned.is_cuda() && matches!(cloned.device(), Device::Cuda(0));
        let tag = cloned.gpu_handle().expect("f64 clone handle").dtype() == DType::F64;

        drop(src);
        let mut poison = Vec::new();
        for _ in 0..POISON_ALLOCS {
            poison.push(
                from_vec::<f64>(vec![999.0f64; n], &[n])
                    .expect("f64 poison")
                    .to(Device::Cuda(0))
                    .expect("f64 poison to cuda"),
            );
        }
        let clone_host = cloned
            .to(Device::Cpu)
            .expect("f64 clone to cpu")
            .data()
            .expect("f64 clone data")
            .to_vec();
        drop(poison);
        let deep_copy_and_independent = clone_host == src_vals;

        record(
            "f64 clone",
            resident && tag && deep_copy_and_independent,
            &format!("resident={resident} tag={tag} deep_copy_independent={deep_copy_and_independent}"),
            pass,
            fail,
        );
    }

    // -- bf16 ----------------------------------------------------------------
    {
        let src_f32 = [1.0f32, -2.0, 3.5, 4.0, -5.25, 0.0];
        let src_vals: Vec<half::bf16> = src_f32.iter().copied().map(half::bf16::from_f32).collect();
        let n = src_vals.len();
        let src = from_vec::<half::bf16>(src_vals.clone(), &[n])
            .expect("bf16 cpu")
            .to(Device::Cuda(0))
            .expect("bf16 to cuda");
        let cloned = src.clone();

        let resident = cloned.is_cuda() && matches!(cloned.device(), Device::Cuda(0));
        let tag = cloned.gpu_handle().expect("bf16 clone handle").dtype() == DType::BF16;

        drop(src);
        let mut poison = Vec::new();
        let poison_vals: Vec<half::bf16> = vec![half::bf16::from_f32(999.0); n];
        for _ in 0..POISON_ALLOCS {
            poison.push(
                from_vec::<half::bf16>(poison_vals.clone(), &[n])
                    .expect("bf16 poison")
                    .to(Device::Cuda(0))
                    .expect("bf16 poison to cuda"),
            );
        }
        let clone_host: Vec<half::bf16> = cloned
            .to(Device::Cpu)
            .expect("bf16 clone to cpu")
            .data()
            .expect("bf16 clone data")
            .to_vec();
        drop(poison);
        // bit-exact: cloning never changes the stored bit pattern.
        let deep_copy_and_independent = clone_host
            .iter()
            .zip(&src_vals)
            .all(|(a, b)| a.to_bits() == b.to_bits());

        record(
            "bf16 clone",
            resident && tag && deep_copy_and_independent,
            &format!("resident={resident} tag={tag} deep_copy_independent={deep_copy_and_independent}"),
            pass,
            fail,
        );
    }

    // -- f16 -----------------------------------------------------------------
    {
        let src_f32 = [1.0f32, -2.0, 3.5, 4.0, -5.25, 0.0, 100.0];
        let src_vals: Vec<half::f16> = src_f32.iter().copied().map(half::f16::from_f32).collect();
        let n = src_vals.len();
        let src = from_vec::<half::f16>(src_vals.clone(), &[n])
            .expect("f16 cpu")
            .to(Device::Cuda(0))
            .expect("f16 to cuda");
        let cloned = src.clone();

        let resident = cloned.is_cuda() && matches!(cloned.device(), Device::Cuda(0));
        let tag = cloned.gpu_handle().expect("f16 clone handle").dtype() == DType::F16;

        drop(src);
        let mut poison = Vec::new();
        let poison_vals: Vec<half::f16> = vec![half::f16::from_f32(999.0); n];
        for _ in 0..POISON_ALLOCS {
            poison.push(
                from_vec::<half::f16>(poison_vals.clone(), &[n])
                    .expect("f16 poison")
                    .to(Device::Cuda(0))
                    .expect("f16 poison to cuda"),
            );
        }
        let clone_host: Vec<half::f16> = cloned
            .to(Device::Cpu)
            .expect("f16 clone to cpu")
            .data()
            .expect("f16 clone data")
            .to_vec();
        drop(poison);
        let deep_copy_and_independent = clone_host
            .iter()
            .zip(&src_vals)
            .all(|(a, b)| a.to_bits() == b.to_bits());

        record(
            "f16 clone",
            resident && tag && deep_copy_and_independent,
            &format!("resident={resident} tag={tag} deep_copy_independent={deep_copy_and_independent}"),
            pass,
            fail,
        );
    }

    // -- i32 -----------------------------------------------------------------
    {
        let src_vals: Vec<i32> = vec![1, -2, 3, i32::MIN, i32::MAX, 0];
        let n = src_vals.len();
        let src = IntTensor::<i32>::from_vec(src_vals.clone(), vec![n])
            .expect("i32 cpu")
            .to(Device::Cuda(0))
            .expect("i32 to cuda");
        let cloned = src.clone();

        let resident = cloned.is_cuda() && matches!(cloned.device(), Device::Cuda(0));
        let tag = cloned.gpu_handle().expect("i32 clone handle").dtype() == DType::I32;

        drop(src);
        let mut poison = Vec::new();
        for _ in 0..POISON_ALLOCS {
            poison.push(
                IntTensor::<i32>::from_vec(vec![999i32; n], vec![n])
                    .expect("i32 poison")
                    .to(Device::Cuda(0))
                    .expect("i32 poison to cuda"),
            );
        }
        let clone_host = cloned
            .to(Device::Cpu)
            .expect("i32 clone to cpu")
            .data()
            .expect("i32 clone data")
            .to_vec();
        drop(poison);
        let deep_copy_and_independent = clone_host == src_vals;

        record(
            "i32 clone",
            resident && tag && deep_copy_and_independent,
            &format!("resident={resident} tag={tag} deep_copy_independent={deep_copy_and_independent}"),
            pass,
            fail,
        );
    }

    // -- i64 -----------------------------------------------------------------
    {
        let src_vals: Vec<i64> = vec![1, -2, 3, i64::MIN, i64::MAX, 0, 1_000_000_000_000];
        let n = src_vals.len();
        let src = IntTensor::<i64>::from_vec(src_vals.clone(), vec![n])
            .expect("i64 cpu")
            .to(Device::Cuda(0))
            .expect("i64 to cuda");
        let cloned = src.clone();

        let resident = cloned.is_cuda() && matches!(cloned.device(), Device::Cuda(0));
        let tag = cloned.gpu_handle().expect("i64 clone handle").dtype() == DType::I64;

        drop(src);
        let mut poison = Vec::new();
        for _ in 0..POISON_ALLOCS {
            poison.push(
                IntTensor::<i64>::from_vec(vec![999i64; n], vec![n])
                    .expect("i64 poison")
                    .to(Device::Cuda(0))
                    .expect("i64 poison to cuda"),
            );
        }
        let clone_host = cloned
            .to(Device::Cpu)
            .expect("i64 clone to cpu")
            .data()
            .expect("i64 clone data")
            .to_vec();
        drop(poison);
        let deep_copy_and_independent = clone_host == src_vals;

        record(
            "i64 clone",
            resident && tag && deep_copy_and_independent,
            &format!("resident={resident} tag={tag} deep_copy_independent={deep_copy_and_independent}"),
            pass,
            fail,
        );
    }

    // -- bool ----------------------------------------------------------------
    {
        let src_vals = vec![true, false, true, true, false, false, true, false];
        let n = src_vals.len();
        let src = BoolTensor::from_vec(src_vals.clone(), vec![n])
            .expect("bool cpu")
            .to(Device::Cuda(0))
            .expect("bool to cuda");
        let cloned = src.clone();

        let resident = cloned.is_cuda() && matches!(cloned.device(), Device::Cuda(0));
        let tag = cloned.gpu_handle().expect("bool clone handle").dtype() == DType::Bool;

        drop(src);
        let mut poison = Vec::new();
        for _ in 0..POISON_ALLOCS {
            poison.push(
                BoolTensor::from_vec(vec![true; n], vec![n])
                    .expect("bool poison")
                    .to(Device::Cuda(0))
                    .expect("bool poison to cuda"),
            );
        }
        let clone_host = cloned
            .to(Device::Cpu)
            .expect("bool clone to cpu")
            .data()
            .expect("bool clone data")
            .to_vec();
        drop(poison);
        let deep_copy_and_independent = clone_host == src_vals;

        record(
            "bool clone",
            resident && tag && deep_copy_and_independent,
            &format!("resident={resident} tag={tag} deep_copy_independent={deep_copy_and_independent}"),
            pass,
            fail,
        );
    }
}

#[test]
fn probe_phase_hardening_clone() {
    ensure_cuda_backend();

    let mut pass = 0usize;
    let mut fail = 0usize;
    main_check(&mut pass, &mut fail);

    println!("PASS: {pass}, FAIL: {fail}");
    assert_eq!(fail, 0, "clone-hardening probe had {fail} failures");
}
