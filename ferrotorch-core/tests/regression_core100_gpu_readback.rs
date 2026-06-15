//! Regression tests for CORE-100 (#1794) — CLASS-U, Critical.
//!
//! `GpuBackend::gpu_to_cpu` returns an ordinary `Vec<u8>`. Before the fix,
//! `Tensor::to(Cpu)` and `IntTensor::to(Cpu)` rebuilt a typed `Vec<T>` over
//! that byte vector's allocation with `Vec::from_raw_parts` — which is
//! undefined behavior unless the backend's `Vec<u8>` happens to be
//! T-aligned AND allocation-layout compatible (capacity in whole elements,
//! allocated under `Layout::array::<T>`). Nothing in the trait promises
//! either. The bundled CUDA backend coincidentally satisfies both (it builds
//! its `Vec<u8>` by reinterpreting a typed vector); any conforming foreign
//! backend that returns a *normally allocated* `Vec<u8>` made the readback
//! UB: misaligned reads + dealloc with the wrong layout.
//!
//! These tests register a mock `GpuBackend` whose `gpu_to_cpu` returns
//! exactly such a normally allocated `Vec<u8>` (built with
//! `Vec::with_capacity` + `extend_from_slice`, i.e. `Layout::array::<u8>`),
//! and pin the post-fix contract: the readback decodes BY COPY into a
//! freshly allocated `Vec<T>`, so values round-trip correctly with no
//! assumption about the byte allocation's alignment, capacity, or allocator.
//!
//! What this mock can and cannot force:
//! - CAN force a `Vec<u8>` allocated under `Layout::array::<u8>(n)` (align 1)
//!   with a capacity that is NOT a multiple of `size_of::<T>()` — under the
//!   pre-fix code that is a guaranteed dealloc-layout mismatch (UB).
//! - CANNOT force a numerically misaligned pointer on real hardware (the
//!   global allocator returns ≥16-byte-aligned blocks in practice), so the
//!   plain `cargo test` run of the round-trip tests passed even pre-fix.
//!   Under MIRI, however, BOTH halves of the UB class are deterministic:
//!   MIRI tracks the byte allocation's true align-1 layout symbolically, so
//!   the pre-fix code failed `readback_from_normally_allocated_byte_vec_f32`
//!   with "constructing invalid value of type &[f32]: encountered an
//!   unaligned reference (required 4 byte alignment but found 1)" before
//!   even reaching the dealloc-layout mismatch. These tests are
//!   MIRI-compatible (no FFI, no real CUDA); run them under
//!   `cargo +nightly miri test -p ferrotorch-core --test
//!   regression_core100_gpu_readback` to re-verify the soundness half.
//!
//! This file is its own integration-test binary (own process), so
//! registering the mock in ferrotorch-core's process-global backend slot
//! cannot interfere with any other test binary.

use ferrotorch_core::gpu_dispatch::{
    GpuBackend, GpuBufferHandle, GpuRngState, register_gpu_backend,
};
use ferrotorch_core::{
    DType, Device, FerrotorchError, FerrotorchResult, IntTensor, Tensor, TensorStorage,
};

/// The device-side payload the mock stores inside a `GpuBufferHandle`.
struct MockBuf {
    /// Raw element bytes (what a D2H copy would produce).
    bytes: Vec<u8>,
    /// Extra capacity (beyond `bytes.len()`) that `gpu_to_cpu` reserves on
    /// the returned `Vec<u8>` — used to force a capacity that is not a
    /// multiple of the element size.
    extra_capacity: usize,
    /// Stray bytes `gpu_to_cpu` appends to the readback — used to force a
    /// byte length that is not a multiple of the element size (a malformed
    /// backend reply that must surface as a structured `Err`).
    pad_bytes: usize,
}

/// Uploads addressed to this (fake) ordinal get one stray byte appended on
/// readback — the malformed-length backend behavior under test. Reachable
/// through the fully public `to(Device::Cuda(CORRUPT_ORDINAL))` API.
const CORRUPT_ORDINAL: usize = 7;

/// A conforming `GpuBackend` that honors the documented `gpu_to_cpu`
/// contract — "return the buffer's bytes as an ordinary `Vec<u8>`" — with a
/// NORMALLY allocated byte vector (no typed-vector reinterpretation, no
/// alignment guarantee, `Layout::array::<u8>` allocation).
struct MockByteBackend;

impl GpuBackend for MockByteBackend {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn cpu_to_gpu(
        &self,
        data: &[u8],
        dtype: DType,
        device: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let elem = dtype.size_of().max(1);
        // SAFETY: `MockBuf` is the concrete allocation type this fake backend
        // owns; len is computed in logical elements for the supplied dtype,
        // and `device` is the fake ordinal recorded in the handle.
        Ok(unsafe {
            GpuBufferHandle::new(
                Box::new(MockBuf {
                    bytes: data.to_vec(),
                    extra_capacity: 0,
                    pad_bytes: usize::from(device == CORRUPT_ORDINAL),
                }),
                device,
                data.len() / elem,
                dtype,
            )
        })
    }

    fn gpu_to_cpu(&self, handle: &GpuBufferHandle) -> FerrotorchResult<Vec<u8>> {
        let buf = handle
            .downcast_ref::<MockBuf>()
            .ok_or(FerrotorchError::InvalidArgument {
                message: "MockByteBackend: foreign handle".into(),
            })?;
        // A NORMALLY allocated Vec<u8>: allocated under Layout::array::<u8>
        // (alignment 1), filled by memcpy. `extra_capacity` lets a test force
        // `capacity % size_of::<T>() != 0`, which the pre-fix from_raw_parts
        // reinterpretation turned into a dealloc-layout mismatch. `pad_bytes`
        // appends stray bytes so the byte LENGTH is not an element multiple.
        let mut out = Vec::with_capacity(buf.bytes.len() + buf.pad_bytes + buf.extra_capacity);
        out.extend_from_slice(&buf.bytes);
        out.extend(std::iter::repeat_n(0xAB_u8, buf.pad_bytes));
        Ok(out)
    }

    fn clone_buffer(&self, handle: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let buf = handle
            .downcast_ref::<MockBuf>()
            .ok_or(FerrotorchError::InvalidArgument {
                message: "MockByteBackend: foreign handle".into(),
            })?;
        // SAFETY: cloning preserves the mock allocation type, logical length,
        // dtype tag, and fake device ordinal from the source handle.
        Ok(unsafe {
            GpuBufferHandle::new(
                Box::new(MockBuf {
                    bytes: buf.bytes.clone(),
                    extra_capacity: buf.extra_capacity,
                    pad_bytes: buf.pad_bytes,
                }),
                handle.device_ordinal(),
                handle.len(),
                handle.dtype(),
            )
        })
    }

    fn alloc_zeros(
        &self,
        len: usize,
        dtype: DType,
        device: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        // SAFETY: the mock buffer owns exactly `len * itemsize` bytes for the
        // requested dtype and fake device ordinal.
        Ok(unsafe {
            GpuBufferHandle::new(
                Box::new(MockBuf {
                    bytes: vec![0u8; len * dtype.size_of().max(1)],
                    extra_capacity: 0,
                    pad_bytes: 0,
                }),
                device,
                len,
                dtype,
            )
        })
    }

    // ------------------------------------------------------------------
    // Required kernel slots the readback tests never touch. Honest Err per
    // R-LOUD-1 — a mock that cannot compute returns a structured error,
    // never a plausible value.
    // ------------------------------------------------------------------

    fn add_f32(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "mock add_f32" })
    }
    fn sub_f32(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "mock sub_f32" })
    }
    fn mul_f32(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "mock mul_f32" })
    }
    fn neg_f32(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "mock neg_f32" })
    }
    fn relu_f32(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "mock relu_f32",
        })
    }
    fn matmul_f32(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "mock matmul_f32",
        })
    }
    fn sum_f32(&self, _a: &GpuBufferHandle, _len: usize) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "mock sum_f32" })
    }
    fn broadcast_add_f32(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _a_shape: &[usize],
        _b_shape: &[usize],
        _out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "mock broadcast_add_f32",
        })
    }
    fn broadcast_sub_f32(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _a_shape: &[usize],
        _b_shape: &[usize],
        _out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "mock broadcast_sub_f32",
        })
    }
    fn broadcast_mul_f32(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _a_shape: &[usize],
        _b_shape: &[usize],
        _out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "mock broadcast_mul_f32",
        })
    }
    fn broadcast_div_f32(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _a_shape: &[usize],
        _b_shape: &[usize],
        _out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "mock broadcast_div_f32",
        })
    }
    fn softmax_f32(
        &self,
        _a: &GpuBufferHandle,
        _rows: usize,
        _cols: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "mock softmax_f32",
        })
    }
    fn dropout_f32(
        &self,
        _a: &GpuBufferHandle,
        _threshold: u32,
        _scale: f32,
        _seed: u32,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "mock dropout_f32",
        })
    }
    fn transpose_2d_f32(
        &self,
        _a: &GpuBufferHandle,
        _m: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "mock transpose_2d_f32",
        })
    }
    fn permute_0213_f32(
        &self,
        _a: &GpuBufferHandle,
        _d0: usize,
        _d1: usize,
        _d2: usize,
        _d3: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "mock permute_0213_f32",
        })
    }
    fn bmm_f32(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _batch: usize,
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "mock bmm_f32" })
    }
    fn gelu_f32(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "mock gelu_f32",
        })
    }
    fn layernorm_f32(
        &self,
        _input: &GpuBufferHandle,
        _weight: &GpuBufferHandle,
        _bias: &GpuBufferHandle,
        _rows: usize,
        _cols: usize,
        _eps: f32,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "mock layernorm_f32",
        })
    }
    fn slice_write_f32(
        &self,
        _src: &GpuBufferHandle,
        _dst: &mut GpuBufferHandle,
        _n_batch: usize,
        _d: usize,
        _max_len: usize,
        _pos: usize,
    ) -> FerrotorchResult<()> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "mock slice_write_f32",
        })
    }
    fn slice_read_f32(
        &self,
        _src: &GpuBufferHandle,
        _n_batch: usize,
        _d: usize,
        _len: usize,
        _max_len: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "mock slice_read_f32",
        })
    }
    fn embed_lookup_f32(
        &self,
        _idx: &GpuBufferHandle,
        _weight: &GpuBufferHandle,
        _d: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "mock embed_lookup_f32",
        })
    }
    fn embed_lookup_batch_f32(
        &self,
        _indices: &GpuBufferHandle,
        _weight: &GpuBufferHandle,
        _n: usize,
        _d: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "mock embed_lookup_batch_f32",
        })
    }
    fn scatter_add_rows_f32(
        &self,
        _grad_output: &GpuBufferHandle,
        _indices: &GpuBufferHandle,
        _num_embeddings: usize,
        _d: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "mock scatter_add_rows_f32",
        })
    }
    fn scale_f32(&self, _a: &GpuBufferHandle, _scalar: f32) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "mock scale_f32",
        })
    }
    fn relu_backward_f32(
        &self,
        _grad: &GpuBufferHandle,
        _input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "mock relu_backward_f32",
        })
    }
    fn gelu_backward_f32(
        &self,
        _grad: &GpuBufferHandle,
        _input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "mock gelu_backward_f32",
        })
    }
    fn gelu_backward_erf_f32(
        &self,
        _grad: &GpuBufferHandle,
        _input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "mock gelu_backward_erf_f32",
        })
    }
    fn index_select_1d_f32(
        &self,
        _input: &GpuBufferHandle,
        _indices: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "mock index_select_1d_f32",
        })
    }
    fn scatter_add_1d_f32(
        &self,
        _grad_output: &GpuBufferHandle,
        _indices: &GpuBufferHandle,
        _input_len: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "mock scatter_add_1d_f32",
        })
    }
    fn index_select_dim_f32(
        &self,
        _input: &GpuBufferHandle,
        _indices: &GpuBufferHandle,
        _outer: usize,
        _in_dim_size: usize,
        _out_dim_size: usize,
        _inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "mock index_select_dim_f32",
        })
    }
    fn masked_fill_f32(
        &self,
        _input: &GpuBufferHandle,
        _mask: &GpuBufferHandle,
        _value: f32,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "mock masked_fill_f32",
        })
    }
    fn masked_zero_f32(
        &self,
        _grad: &GpuBufferHandle,
        _mask: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "mock masked_zero_f32",
        })
    }
    fn has_inf_nan_f32(&self, _a: &GpuBufferHandle) -> FerrotorchResult<bool> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "mock has_inf_nan_f32",
        })
    }
}

/// Register the mock exactly once for this test binary's process.
fn ensure_mock() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        // Err means a backend was already registered — impossible in this
        // binary (nothing else registers one), but harmless either way.
        let _ = register_gpu_backend(Box::new(MockByteBackend));
    });
    // GpuRngState is referenced so the import stays honest if the trait
    // gains required RNG methods; silence the unused-import path.
    let _ = std::mem::size_of::<GpuRngState>();
}

/// Build a "CUDA-resident" Tensor<f32> directly from a mock handle.
fn mock_gpu_tensor_f32(values: &[f32], shape: &[usize], extra_capacity: usize) -> Tensor<f32> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    // SAFETY: the mock buffer bytes are the little-endian representation of
    // `values`, with one logical F32 element per value and fake ordinal 0.
    let handle = unsafe {
        GpuBufferHandle::new(
            Box::new(MockBuf {
                bytes,
                extra_capacity,
                pad_bytes: 0,
            }),
            0,
            values.len(),
            DType::F32,
        )
    };
    Tensor::from_storage(TensorStorage::gpu(handle), shape.to_vec(), false)
        .expect("mock gpu tensor")
}

// ---------------------------------------------------------------------------
// Round-trip through the documented (alignment-free) gpu_to_cpu contract
// ---------------------------------------------------------------------------

#[test]
fn readback_from_normally_allocated_byte_vec_f32() {
    ensure_mock();
    let vals = [1.0f32, -2.5, 3.25, 0.0, 5.5, -6.75];
    let gpu = Tensor::<f32>::from_storage(TensorStorage::cpu(vals.to_vec()), vec![2, 3], false)
        .expect("cpu tensor")
        .to(Device::Cuda(0))
        .expect("mock upload");
    assert!(gpu.is_cuda(), "mock upload must report CUDA residency");
    let cpu = gpu.to(Device::Cpu).expect("readback through plain Vec<u8>");
    assert_eq!(cpu.device(), Device::Cpu);
    assert_eq!(cpu.shape(), &[2, 3]);
    // Bit-exact transport: D2H readback is a copy, not a computation
    // (upstream `.cpu()` semantics — TensorBody.h `toBackend` copies).
    assert_eq!(cpu.data().expect("cpu data"), &vals[..]);
}

#[test]
fn readback_from_normally_allocated_byte_vec_f64() {
    ensure_mock();
    let vals = [f64::MIN_POSITIVE, -1.0, 2.0_f64.powi(60), 0.125];
    let gpu = Tensor::<f64>::from_storage(TensorStorage::cpu(vals.to_vec()), vec![4], false)
        .expect("cpu tensor")
        .to(Device::Cuda(0))
        .expect("mock upload");
    let cpu = gpu.to(Device::Cpu).expect("readback through plain Vec<u8>");
    assert_eq!(cpu.device(), Device::Cpu);
    assert_eq!(cpu.data().expect("cpu data"), &vals[..]);
}

#[test]
fn readback_from_normally_allocated_byte_vec_i32_and_i64() {
    ensure_mock();
    let v32 = [i32::MIN, -1, 0, 1, i32::MAX, 42];
    let t32 = IntTensor::<i32>::from_vec(v32.to_vec(), vec![2, 3])
        .expect("int tensor")
        .to(Device::Cuda(0))
        .expect("mock upload");
    let r32 = t32.to(Device::Cpu).expect("i32 readback");
    assert_eq!(r32.device(), Device::Cpu);
    assert_eq!(r32.data().expect("i32 data"), &v32[..]);

    let v64 = [i64::MIN, i64::MAX, 0, -7];
    let t64 = IntTensor::<i64>::from_vec(v64.to_vec(), vec![4])
        .expect("int tensor")
        .to(Device::Cuda(0))
        .expect("mock upload");
    let r64 = t64.to(Device::Cpu).expect("i64 readback");
    assert_eq!(r64.data().expect("i64 data"), &v64[..]);
}

/// Capacity of the returned `Vec<u8>` is NOT a multiple of `size_of::<T>()`
/// (24 bytes used, capacity 25). Pre-fix this was the by-construction UB
/// case: `from_raw_parts(ptr, 6, 25/4=6)` then dealloc under
/// `Layout::array::<f32>(6)` (24 bytes, align 4) for an allocation made
/// under `Layout::array::<u8>(25)` (25 bytes, align 1) — caught by MIRI;
/// often silently "fine" at runtime. Post-fix the bytes are copied out and
/// the `Vec<u8>` is dropped under its own true layout.
#[test]
fn readback_with_non_element_multiple_capacity_is_sound() {
    ensure_mock();
    let vals = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0];
    let gpu = mock_gpu_tensor_f32(&vals, &[2, 3], 1); // capacity 25, len 24
    let cpu = gpu.to(Device::Cpu).expect("readback");
    assert_eq!(cpu.data().expect("cpu data"), &vals[..]);
}

// ---------------------------------------------------------------------------
// R-RED-2: malformed byte length → structured Err, never a plausible value
// ---------------------------------------------------------------------------

/// The backend returns 25 bytes for a 6-element f32 tensor (not a multiple
/// of 4). Pre-fix, `Tensor::to(Cpu)` performed no divisibility validation:
/// it silently floor-divided (len 25/4 = 6) and returned a plausible tensor
/// (plus the dealloc-layout UB). Post-fix this is a structured
/// `InvalidArgument`.
#[test]
fn truncated_byte_count_is_structured_err_f32() {
    ensure_mock();
    let mut bytes = Vec::with_capacity(25);
    for v in [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0] {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes.push(0xAB); // 25 bytes: NOT a multiple of size_of::<f32>()
    // SAFETY: intentionally malformed byte length for a metadata-consistent
    // mock F32 handle; the fake backend readback path is what rejects it.
    let handle = unsafe {
        GpuBufferHandle::new(
            Box::new(MockBuf {
                bytes,
                extra_capacity: 0,
                pad_bytes: 0,
            }),
            0,
            6,
            DType::F32,
        )
    };
    let gpu = Tensor::<f32>::from_storage(TensorStorage::gpu(handle), vec![2, 3], false)
        .expect("mock gpu tensor");
    match gpu.to(Device::Cpu) {
        Err(FerrotorchError::InvalidArgument { message }) => {
            assert!(
                message.contains("not a multiple"),
                "error must name the divisibility violation, got: {message}"
            );
        }
        Err(other) => panic!("expected InvalidArgument, got different error: {other:?}"),
        Ok(_) => panic!(
            "expected structured Err on a 25-byte readback for 6 f32 elements, \
             got Ok (pre-CORE-100 silent floor-division behavior)"
        ),
    }
}

/// Same malformed-length contract on the IntTensor path: an upload routed to
/// `CORRUPT_ORDINAL` reads back `2 * 4 + 1 = 9` bytes for 2 i32 elements.
/// This arm's divisibility check predates CORE-100 (it was already a
/// structured `Err` at HEAD); the test pins it against regression while the
/// decode underneath moves from `from_raw_parts` to copy.
#[test]
fn truncated_byte_count_is_structured_err_i32() {
    ensure_mock();
    let gpu = IntTensor::<i32>::from_vec(vec![1, 2], vec![2])
        .expect("int tensor")
        .to(Device::Cuda(CORRUPT_ORDINAL))
        .expect("mock upload");
    match gpu.to(Device::Cpu) {
        Err(FerrotorchError::InvalidArgument { message }) => {
            assert!(
                message.contains("not a multiple"),
                "error must name the divisibility violation, got: {message}"
            );
        }
        Err(other) => panic!("expected InvalidArgument, got different error: {other:?}"),
        Ok(_) => panic!("expected structured Err on a 9-byte readback for 2 i32 elements"),
    }
}
