use ferrotorch_core::dtype::DType;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::gpu_dispatch::{GpuBackend, GpuBufferHandle, register_gpu_backend};
use ferrotorch_core::{Generator, manual_seed, rand};

#[derive(Debug)]
struct FailingSeedBackend;

macro_rules! not_used_handle {
    ($name:ident ( $($arg:ident : $ty:ty),* $(,)? )) => {
        fn $name(&self, $($arg: $ty),*) -> FerrotorchResult<GpuBufferHandle> {
            $(let _ = $arg;)*
            Err(FerrotorchError::NotImplementedOnCuda { op: stringify!($name) })
        }
    };
}

impl GpuBackend for FailingSeedBackend {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn cpu_to_gpu(
        &self,
        data: &[u8],
        dtype: DType,
        device: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let _ = (data, dtype, device);
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "test cpu_to_gpu",
        })
    }

    fn gpu_to_cpu(&self, handle: &GpuBufferHandle) -> FerrotorchResult<Vec<u8>> {
        let _ = handle;
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "test gpu_to_cpu",
        })
    }

    not_used_handle!(clone_buffer(handle: &GpuBufferHandle));
    not_used_handle!(alloc_zeros(len: usize, dtype: DType, device: usize));
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

    fn manual_seed_gpu(&self, seed: u64) -> FerrotorchResult<()> {
        Err(FerrotorchError::InvalidArgument {
            message: format!("mock backend rejected GPU seed {seed}"),
        })
    }
}

#[test]
fn manual_seed_propagates_registered_gpu_backend_failure() {
    manual_seed(123).expect("baseline CPU manual_seed must succeed before registering backend");
    let expected_after_failed_gpu_seed: Vec<u32> = {
        let mut generator = Generator::new(123);
        (0..4)
            .map(|_| generator.next_uniform_f32().to_bits())
            .collect()
    };

    let _ = register_gpu_backend(Box::new(FailingSeedBackend));

    let err = manual_seed(0x5eed).expect_err(
        "manual_seed must propagate registered backend seed failures instead of discarding them",
    );

    match err {
        FerrotorchError::InvalidArgument { message } => {
            assert!(
                message.contains("mock backend rejected GPU seed 24301"),
                "unexpected manual_seed error: {message}"
            );
        }
        other => panic!("expected backend InvalidArgument to propagate, got {other:?}"),
    }

    let after = rand::<f32>(&[4]).expect("CPU rand should still work after failed GPU seeding");
    let after_bits: Vec<u32> = after
        .data()
        .expect("rand output should be a contiguous CPU tensor")
        .iter()
        .map(|value| value.to_bits())
        .collect();
    assert_eq!(
        after_bits, expected_after_failed_gpu_seed,
        "a registered GPU seed failure must not advance or reseed the CPU default RNG"
    );
}
