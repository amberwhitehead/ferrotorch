//! Regression test for the secondary half of CORE-151 (#1845):
//! `split_t` must not depend on a CUDA split kernel or demote to a CPU
//! materialization fallback.
//!
//! PyTorch's `Tensor.split` returns views. Ferrotorch should therefore
//! be able to split a CUDA tensor even on a backend with no
//! `strided_split_f32` implementation, because forward split/chunk is
//! metadata-only: same storage, adjusted shape/offset/strides, resident
//! on the original device. Reading a non-zero-offset result back to CPU
//! may still require `strided_copy`, but the split itself must not call a
//! split kernel or perform a hidden host round trip.
//!
//! This test registers a mock `GpuBackend` whose data plumbing
//! (`cpu_to_gpu` / `gpu_to_cpu` / `clone_buffer` / `alloc_zeros`) works
//! but whose kernels return structured errors, exactly like a conforming
//! foreign backend that has not implemented compute kernels.
//!
//! Oracle (R-ORACLE-1(b)): live torch 2.11.0+cu130, 2026-06-11:
//!
//! ```python
//! x = torch.arange(24., dtype=torch.float32).reshape(4, 6)
//! parts = x.split([2, 2], dim=0)
//! # parts[0].flatten() -> [0..12); parts[1].flatten() -> [12..24)
//! ```
//!
//! This file is its own integration-test binary (own process), so
//! registering the mock in the process-global backend slot cannot
//! interfere with any other test binary.

use ferrotorch_core::creation::from_vec;
use ferrotorch_core::gpu_dispatch::{GpuBackend, GpuBufferHandle, register_gpu_backend};
use ferrotorch_core::{DType, Device, FerrotorchError, FerrotorchResult};
use std::sync::Arc;

/// Device-side payload: raw element bytes, exactly what a D2H copy returns.
struct MockBuf {
    bytes: Vec<u8>,
}

/// A conforming backend with working memory plumbing and NO compute
/// kernels — every kernel slot (including the defaulted
/// `strided_split_f32`) returns a structured error per R-LOUD-1.
struct MockNoKernelBackend;

impl GpuBackend for MockNoKernelBackend {
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
                message: "MockNoKernelBackend: foreign handle".into(),
            })?;
        Ok(buf.bytes.clone())
    }

    fn clone_buffer(&self, handle: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let buf = handle
            .downcast_ref::<MockBuf>()
            .ok_or(FerrotorchError::InvalidArgument {
                message: "MockNoKernelBackend: foreign handle".into(),
            })?;
        // SAFETY: cloning preserves the mock allocation type, logical length,
        // dtype tag, and fake device ordinal from the source handle.
        Ok(unsafe {
            GpuBufferHandle::new(
                Box::new(MockBuf {
                    bytes: buf.bytes.clone(),
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
                }),
                device,
                len,
                dtype,
            )
        })
    }

    // ------------------------------------------------------------------
    // Required kernel slots this test never computes with. Honest Err per
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
        let _ = register_gpu_backend(Box::new(MockNoKernelBackend));
    });
}

/// `split` of a contiguous CUDA f32 tensor on a backend whose
/// `strided_split_f32` errors must still succeed as metadata-only views.
#[test]
fn split_returns_cuda_views_without_split_kernel_or_cpu_fallback() {
    ensure_mock();

    let data: Vec<f32> = (0..24).map(|v| v as f32).collect();
    let x = from_vec(data, &[4, 6])
        .expect("construct cpu tensor")
        .to(Device::Cuda(0))
        .expect("cpu->mock-gpu upload");

    let parts = x
        .split(&[2, 2], 0)
        .expect("split must be metadata-only and not require a CUDA split kernel");
    assert_eq!(parts.len(), 2);

    for (i, part) in parts.iter().enumerate() {
        assert!(part.is_cuda(), "split view {i} must stay on CUDA");
        assert_eq!(part.shape(), &[2, 6], "chunk {i} shape");
        assert!(
            Arc::ptr_eq(x.inner_storage_arc(), part.inner_storage_arc()),
            "split view {i} must share storage with the input like PyTorch"
        );
    }
    assert_eq!(parts[0].storage_offset(), 0);
    assert_eq!(parts[1].storage_offset(), 12);
}
