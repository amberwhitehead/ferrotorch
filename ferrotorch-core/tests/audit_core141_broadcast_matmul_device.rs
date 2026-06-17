//! CORE-141 / crosslink #1835: raw `ops::linalg::matmul` must not implement
//! broadcast matmul by downloading CUDA operands, computing on CPU, and
//! uploading the result again.
//!
//! PyTorch source anchor:
//! `/home/doll/pytorch/aten/src/ATen/native/LinearAlgebra.cpp:_matmul_impl`
//! expands/folds leading dimensions, then calls `bmm` on the input device.

use std::sync::{
    Mutex,
    atomic::{AtomicUsize, Ordering},
};

use ferrotorch_core::dtype::DType;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::gpu_dispatch::{
    GpuBackend, GpuBufferHandle, GpuRngState, register_gpu_backend,
};
use ferrotorch_core::ops::linalg::matmul;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::{Device, Tensor};

static CPU_TO_GPU_CALLS: AtomicUsize = AtomicUsize::new(0);
static GPU_TO_CPU_CALLS: AtomicUsize = AtomicUsize::new(0);
static BROADCAST_BMM_CALLS: AtomicUsize = AtomicUsize::new(0);
static TEST_LOCK: Mutex<()> = Mutex::new(());

#[derive(Debug)]
struct MockBroadcastBackend;

fn fake_handle(len: usize, dtype: DType, device: usize) -> GpuBufferHandle {
    // SAFETY: this test backend never exposes a raw pointer and never lets
    // callers read the dummy allocation back. The handle metadata is the
    // behavior under test: device ordinal, dtype tag, and logical length.
    unsafe { GpuBufferHandle::new(Box::new(Vec::<u8>::new()), device, len, dtype) }
}

fn register_mock_backend() {
    let _ = register_gpu_backend(Box::new(MockBroadcastBackend));
}

fn reset_counts() {
    CPU_TO_GPU_CALLS.store(0, Ordering::SeqCst);
    GPU_TO_CPU_CALLS.store(0, Ordering::SeqCst);
    BROADCAST_BMM_CALLS.store(0, Ordering::SeqCst);
}

fn cpu_tensor(data_len: usize, shape: &[usize]) -> Tensor<f32> {
    let data: Vec<f32> = (0..data_len).map(|v| v as f32).collect();
    Tensor::from_storage(TensorStorage::cpu(data), shape.to_vec(), false).unwrap()
}

macro_rules! not_used_handle {
    ($name:ident ( $($arg:ident : $ty:ty),* $(,)? )) => {
        fn $name(&self, $($arg: $ty),*) -> FerrotorchResult<GpuBufferHandle> {
            $(let _ = $arg;)*
            Err(FerrotorchError::NotImplementedOnCuda { op: stringify!($name) })
        }
    };
}

impl GpuBackend for MockBroadcastBackend {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn cpu_to_gpu(
        &self,
        data: &[u8],
        dtype: DType,
        device: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        CPU_TO_GPU_CALLS.fetch_add(1, Ordering::SeqCst);
        Ok(fake_handle(data.len() / dtype.size_of(), dtype, device))
    }

    fn gpu_to_cpu(&self, handle: &GpuBufferHandle) -> FerrotorchResult<Vec<u8>> {
        let _ = handle;
        GPU_TO_CPU_CALLS.fetch_add(1, Ordering::SeqCst);
        Err(FerrotorchError::InvalidArgument {
            message: "mock backend forbids D2H in broadcast matmul".into(),
        })
    }

    fn clone_buffer(&self, handle: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Ok(fake_handle(
            handle.len(),
            handle.dtype(),
            handle.device_ordinal(),
        ))
    }

    fn alloc_zeros(
        &self,
        len: usize,
        dtype: DType,
        device: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Ok(fake_handle(len, dtype, device))
    }

    not_used_handle!(add_f32(a: &GpuBufferHandle, b: &GpuBufferHandle));
    not_used_handle!(sub_f32(a: &GpuBufferHandle, b: &GpuBufferHandle));
    not_used_handle!(mul_f32(a: &GpuBufferHandle, b: &GpuBufferHandle));
    not_used_handle!(neg_f32(a: &GpuBufferHandle));
    not_used_handle!(relu_f32(a: &GpuBufferHandle));
    not_used_handle!(matmul_f32(a: &GpuBufferHandle, b: &GpuBufferHandle, m: usize, k: usize, n: usize));
    not_used_handle!(sum_f32(a: &GpuBufferHandle, len: usize));
    not_used_handle!(broadcast_add_f32(a: &GpuBufferHandle, b: &GpuBufferHandle, a_shape: &[usize], b_shape: &[usize], out_shape: &[usize]));
    not_used_handle!(broadcast_sub_f32(a: &GpuBufferHandle, b: &GpuBufferHandle, a_shape: &[usize], b_shape: &[usize], out_shape: &[usize]));
    not_used_handle!(broadcast_mul_f32(a: &GpuBufferHandle, b: &GpuBufferHandle, a_shape: &[usize], b_shape: &[usize], out_shape: &[usize]));
    not_used_handle!(broadcast_div_f32(a: &GpuBufferHandle, b: &GpuBufferHandle, a_shape: &[usize], b_shape: &[usize], out_shape: &[usize]));
    not_used_handle!(softmax_f32(a: &GpuBufferHandle, rows: usize, cols: usize));
    not_used_handle!(dropout_f32(a: &GpuBufferHandle, threshold: u32, scale: f32, seed: u32));
    not_used_handle!(transpose_2d_f32(a: &GpuBufferHandle, m: usize, n: usize));
    not_used_handle!(permute_0213_f32(a: &GpuBufferHandle, d0: usize, d1: usize, d2: usize, d3: usize));
    not_used_handle!(bmm_f32(a: &GpuBufferHandle, b: &GpuBufferHandle, batch: usize, m: usize, k: usize, n: usize));
    not_used_handle!(gelu_f32(a: &GpuBufferHandle));
    not_used_handle!(layernorm_f32(input: &GpuBufferHandle, weight: &GpuBufferHandle, bias: &GpuBufferHandle, rows: usize, cols: usize, eps: f32));
    fn slice_write_f32(
        &self,
        src: &GpuBufferHandle,
        dst: &mut GpuBufferHandle,
        n_batch: usize,
        d: usize,
        max_len: usize,
        pos: usize,
    ) -> FerrotorchResult<()> {
        let _ = (src, dst, n_batch, d, max_len, pos);
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "slice_write_f32",
        })
    }
    not_used_handle!(slice_read_f32(src: &GpuBufferHandle, n_batch: usize, d: usize, len: usize, max_len: usize));
    not_used_handle!(embed_lookup_f32(idx: &GpuBufferHandle, weight: &GpuBufferHandle, d: usize));
    not_used_handle!(embed_lookup_batch_f32(indices: &GpuBufferHandle, weight: &GpuBufferHandle, n: usize, d: usize));
    not_used_handle!(scatter_add_rows_f32(grad_output: &GpuBufferHandle, indices: &GpuBufferHandle, num_embeddings: usize, d: usize));
    not_used_handle!(scale_f32(a: &GpuBufferHandle, scale: f32));
    not_used_handle!(relu_backward_f32(grad: &GpuBufferHandle, input: &GpuBufferHandle));
    not_used_handle!(gelu_backward_f32(grad: &GpuBufferHandle, input: &GpuBufferHandle));
    not_used_handle!(gelu_backward_erf_f32(grad: &GpuBufferHandle, input: &GpuBufferHandle));
    not_used_handle!(index_select_1d_f32(input: &GpuBufferHandle, indices: &GpuBufferHandle));
    not_used_handle!(scatter_add_1d_f32(grad_output: &GpuBufferHandle, indices: &GpuBufferHandle, input_len: usize));
    not_used_handle!(index_select_dim_f32(input: &GpuBufferHandle, indices: &GpuBufferHandle, outer: usize, in_dim_size: usize, out_dim_size: usize, inner: usize));
    not_used_handle!(masked_fill_f32(input: &GpuBufferHandle, mask: &GpuBufferHandle, value: f32));
    not_used_handle!(masked_zero_f32(input: &GpuBufferHandle, mask: &GpuBufferHandle));

    fn has_inf_nan_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<bool> {
        let _ = a;
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "has_inf_nan_f32",
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn maxpool2d_f32(
        &self,
        input: &GpuBufferHandle,
        batch: usize,
        channels: usize,
        h_in: usize,
        w_in: usize,
        kh: usize,
        kw: usize,
        sh: usize,
        sw: usize,
        ph: usize,
        pw: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, [usize; 4])> {
        let _ = (input, batch, channels, h_in, w_in, kh, kw, sh, sw, ph, pw);
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "maxpool2d_f32",
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn maxpool2d_f64(
        &self,
        input: &GpuBufferHandle,
        batch: usize,
        channels: usize,
        h_in: usize,
        w_in: usize,
        kh: usize,
        kw: usize,
        sh: usize,
        sw: usize,
        ph: usize,
        pw: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, [usize; 4])> {
        let _ = (input, batch, channels, h_in, w_in, kh, kw, sh, sw, ph, pw);
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "maxpool2d_f64",
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn avgpool2d_f32(
        &self,
        input: &GpuBufferHandle,
        batch: usize,
        channels: usize,
        h_in: usize,
        w_in: usize,
        kh: usize,
        kw: usize,
        sh: usize,
        sw: usize,
        ph: usize,
        pw: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, [usize; 4])> {
        let _ = (input, batch, channels, h_in, w_in, kh, kw, sh, sw, ph, pw);
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "avgpool2d_f32",
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn avgpool2d_f64(
        &self,
        input: &GpuBufferHandle,
        batch: usize,
        channels: usize,
        h_in: usize,
        w_in: usize,
        kh: usize,
        kw: usize,
        sh: usize,
        sw: usize,
        ph: usize,
        pw: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, [usize; 4])> {
        let _ = (input, batch, channels, h_in, w_in, kh, kw, sh, sw, ph, pw);
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "avgpool2d_f64",
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn conv2d_f32(
        &self,
        input: &GpuBufferHandle,
        weight: &GpuBufferHandle,
        bias: Option<&GpuBufferHandle>,
        input_shape: [usize; 4],
        weight_shape: [usize; 4],
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, [usize; 4])> {
        let _ = (
            input,
            weight,
            bias,
            input_shape,
            weight_shape,
            stride,
            padding,
            dilation,
            groups,
        );
        Err(FerrotorchError::NotImplementedOnCuda { op: "conv2d_f32" })
    }

    #[allow(clippy::too_many_arguments)]
    fn conv2d_f64(
        &self,
        input: &GpuBufferHandle,
        weight: &GpuBufferHandle,
        bias: Option<&GpuBufferHandle>,
        input_shape: [usize; 4],
        weight_shape: [usize; 4],
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, [usize; 4])> {
        let _ = (
            input,
            weight,
            bias,
            input_shape,
            weight_shape,
            stride,
            padding,
            dilation,
            groups,
        );
        Err(FerrotorchError::NotImplementedOnCuda { op: "conv2d_f64" })
    }

    fn broadcast_bmm_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        a_lead: &[usize],
        b_lead: &[usize],
        out_lead: &[usize],
        m: usize,
        k: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        assert_eq!(a.device_ordinal(), b.device_ordinal());
        assert_eq!(a.dtype(), DType::F32);
        assert_eq!(b.dtype(), DType::F32);
        let a_batch: usize = a_lead.iter().product();
        let b_batch: usize = b_lead.iter().product();
        assert_eq!(a.len(), a_batch.max(1) * m * k);
        assert_eq!(b.len(), b_batch.max(1) * k * n);
        BROADCAST_BMM_CALLS.fetch_add(1, Ordering::SeqCst);
        let out_batch: usize = out_lead.iter().product();
        Ok(fake_handle(
            out_batch * m * n,
            DType::F32,
            a.device_ordinal(),
        ))
    }

    fn save_rng_state(&self, device: usize) -> FerrotorchResult<GpuRngState> {
        Ok(GpuRngState::new(0, 0, 0, device))
    }

    fn restore_rng_state(&self, state: GpuRngState) -> FerrotorchResult<()> {
        let _ = state;
        Ok(())
    }
}

#[test]
fn ops_matmul_cuda_broadcast_uses_backend_not_readback() {
    let _guard = TEST_LOCK.lock().expect("mock backend test lock");
    register_mock_backend();
    reset_counts();

    let a = cpu_tensor(2 * 3 * 4, &[2, 1, 3, 4])
        .to(Device::Cuda(0))
        .expect("upload A");
    let b = cpu_tensor(2 * 5 * 4 * 6, &[2, 5, 4, 6])
        .to(Device::Cuda(0))
        .expect("upload B");

    let out = matmul(&a, &b).expect("CUDA broadcast matmul through backend");

    assert_eq!(out.device(), Device::Cuda(0));
    assert_eq!(out.shape(), &[2, 5, 3, 6]);
    assert_eq!(BROADCAST_BMM_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(
        GPU_TO_CPU_CALLS.load(Ordering::SeqCst),
        0,
        "broadcast matmul must not download CUDA operands"
    );
}

#[test]
fn ops_matmul_cuda_vector_batched_promotion_uses_backend() {
    let _guard = TEST_LOCK.lock().expect("mock backend test lock");
    register_mock_backend();
    reset_counts();

    let vector = cpu_tensor(4, &[4])
        .to(Device::Cuda(0))
        .expect("upload vector");
    let rhs = cpu_tensor(2 * 3 * 4 * 5, &[2, 3, 4, 5])
        .to(Device::Cuda(0))
        .expect("upload rhs");

    let out = matmul(&vector, &rhs).expect("CUDA 1D @ batched RHS");

    assert_eq!(out.device(), Device::Cuda(0));
    assert_eq!(out.shape(), &[2, 3, 5]);
    assert_eq!(BROADCAST_BMM_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(
        GPU_TO_CPU_CALLS.load(Ordering::SeqCst),
        0,
        "1D promotion must still use resident broadcast_bmm"
    );
}

#[test]
fn ops_matmul_rejects_mixed_cpu_cuda_before_readback() {
    let _guard = TEST_LOCK.lock().expect("mock backend test lock");
    register_mock_backend();
    reset_counts();

    let a = cpu_tensor(2 * 3 * 4, &[2, 3, 4]);
    let b = cpu_tensor(4 * 5, &[4, 5])
        .to(Device::Cuda(0))
        .expect("upload B");

    let err = matmul(&a, &b).expect_err("mixed CPU/CUDA matmul must reject");

    assert!(matches!(err, FerrotorchError::DeviceMismatch { .. }));
    assert_eq!(BROADCAST_BMM_CALLS.load(Ordering::SeqCst), 0);
    assert_eq!(GPU_TO_CPU_CALLS.load(Ordering::SeqCst), 0);
}
