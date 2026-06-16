//! CORE-050 regression coverage: differentiable linalg wrappers must validate
//! rank, dimensions, and devices before indexing shapes, raw CPU slices, or CUDA
//! backend handles.

use std::panic::{AssertUnwindSafe, catch_unwind};

use ferrotorch_core::creation::{from_vec, scalar};
use ferrotorch_core::grad_fns::linalg::{
    dot_differentiable, linear_fused, mm_bt_differentiable, mm_differentiable, mv_differentiable,
};
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Tensor};

fn t(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    from_vec(data.to_vec(), shape).expect("test tensor")
}

fn assert_shape_err<F>(label: &str, f: F)
where
    F: FnOnce() -> FerrotorchResult<Tensor<f32>>,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Err(FerrotorchError::ShapeMismatch { .. })) => {}
        Ok(Err(other)) => panic!("{label}: expected ShapeMismatch, got {other:?}"),
        Ok(Ok(value)) => panic!("{label}: expected ShapeMismatch, got Ok({value:?})"),
        Err(_) => panic!("{label}: panicked instead of returning ShapeMismatch"),
    }
}

#[cfg(feature = "gpu")]
fn assert_shape_err_f64<F>(label: &str, f: F)
where
    F: FnOnce() -> FerrotorchResult<Tensor<f64>>,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Err(FerrotorchError::ShapeMismatch { .. })) => {}
        Ok(Err(other)) => panic!("{label}: expected ShapeMismatch, got {other:?}"),
        Ok(Ok(value)) => panic!("{label}: expected ShapeMismatch, got Ok({value:?})"),
        Err(_) => panic!("{label}: panicked instead of returning ShapeMismatch"),
    }
}

#[cfg(feature = "gpu")]
fn assert_device_err<F>(label: &str, f: F)
where
    F: FnOnce() -> FerrotorchResult<Tensor<f32>>,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Err(FerrotorchError::DeviceMismatch { .. })) => {}
        Ok(Err(other)) => panic!("{label}: expected DeviceMismatch, got {other:?}"),
        Ok(Ok(value)) => panic!("{label}: expected DeviceMismatch, got Ok({value:?})"),
        Err(_) => panic!("{label}: panicked instead of returning DeviceMismatch"),
    }
}

#[test]
fn mm_variants_reject_rank_and_inner_mismatches_without_panic() {
    let scalar = scalar(1.0f32).expect("scalar");
    let vec3 = t(&[1.0, 2.0, 3.0], &[3]);
    let a23 = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let a24 = t(&[1.0; 8], &[2, 4]);
    let b35 = t(&[1.0; 15], &[3, 5]);
    let b54 = t(&[1.0; 20], &[5, 4]);

    assert_shape_err("mm scalar lhs", || mm_differentiable(&scalar, &b35));
    assert_shape_err("mm vector lhs", || mm_differentiable(&vec3, &b35));
    assert_shape_err("mm vector rhs", || mm_differentiable(&a23, &vec3));
    assert_shape_err("mm inner mismatch", || mm_differentiable(&a24, &b35));

    assert_shape_err("mm_bt scalar lhs", || mm_bt_differentiable(&scalar, &b35));
    assert_shape_err("mm_bt vector rhs", || mm_bt_differentiable(&a23, &vec3));
    assert_shape_err("mm_bt inner mismatch", || mm_bt_differentiable(&a23, &b54));
}

#[test]
fn mv_rejects_rank_and_vector_length_mismatches_without_panic() {
    let scalar = scalar(1.0f32).expect("scalar");
    let a23 = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let v2 = t(&[1.0, 2.0], &[2]);
    let v4 = t(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let m23 = t(&[1.0; 6], &[2, 3]);

    assert_shape_err("mv scalar matrix", || mv_differentiable(&scalar, &v2));
    assert_shape_err("mv scalar vector", || mv_differentiable(&a23, &scalar));
    assert_shape_err("mv 2d vector", || mv_differentiable(&a23, &m23));
    assert_shape_err("mv short vector", || mv_differentiable(&a23, &v2));
    assert_shape_err("mv long vector", || mv_differentiable(&a23, &v4));
}

#[test]
fn dot_rejects_rank_and_length_mismatches_without_panic() {
    let scalar = scalar(1.0f32).expect("scalar");
    let v2 = t(&[1.0, 2.0], &[2]);
    let v3 = t(&[1.0, 2.0, 3.0], &[3]);
    let m23 = t(&[1.0; 6], &[2, 3]);

    assert_shape_err("dot scalar scalar", || dot_differentiable(&scalar, &scalar));
    assert_shape_err("dot matrix lhs", || dot_differentiable(&m23, &v3));
    assert_shape_err("dot matrix rhs", || dot_differentiable(&v3, &m23));
    assert_shape_err("dot length mismatch", || dot_differentiable(&v2, &v3));
}

#[test]
fn linear_fused_rejects_invalid_weight_bias_contract_without_panic() {
    let scalar = scalar(1.0f32).expect("scalar");
    let input = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let input_bad_inner = t(&[1.0; 8], &[2, 4]);
    let weight = t(&[1.0; 12], &[4, 3]);
    let weight_bad_rank = t(&[1.0, 2.0, 3.0], &[3]);
    let bias_short = t(&[1.0, 2.0, 3.0], &[3]);
    let bias_long = t(&[1.0; 5], &[5]);

    assert_shape_err("linear scalar input", || {
        linear_fused(&scalar, &weight, None)
    });
    assert_shape_err("linear scalar weight", || {
        linear_fused(&input, &scalar, None)
    });
    assert_shape_err("linear vector weight", || {
        linear_fused(&input, &weight_bad_rank, None)
    });
    assert_shape_err("linear inner mismatch", || {
        linear_fused(&input_bad_inner, &weight, None)
    });
    assert_shape_err("linear short bias", || {
        linear_fused(&input, &weight, Some(&bias_short))
    });
    assert_shape_err("linear long bias", || {
        linear_fused(&input, &weight, Some(&bias_long))
    });
}

#[cfg(feature = "gpu")]
mod cuda {
    use std::sync::Once;

    use ferrotorch_core::creation::from_vec;
    use ferrotorch_core::grad_fns::linalg::{
        dot_differentiable, linear_fused, matmul_differentiable, mm_bt_differentiable,
        mm_differentiable, mv_differentiable,
    };
    use ferrotorch_core::{Device, Tensor};

    use super::{assert_device_err, assert_shape_err, assert_shape_err_f64, t};

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for CORE-050 GPU tests");
        });
    }

    fn gpu(tensor: &Tensor<f32>) -> Tensor<f32> {
        tensor.to(Device::Cuda(0)).expect("upload f32 tensor")
    }

    fn gpu64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
        from_vec(data.to_vec(), shape)
            .expect("cpu f64 tensor")
            .to(Device::Cuda(0))
            .expect("upload f64 tensor")
    }

    #[test]
    fn cuda_2d_matmul_paths_reject_inner_mismatch_before_dispatch() {
        ensure_cuda_backend();

        let a24 = gpu(&t(&[1.0; 8], &[2, 4]));
        let b35 = gpu(&t(&[1.0; 15], &[3, 5]));

        assert_shape_err("cuda mm inner mismatch", || mm_differentiable(&a24, &b35));
        assert_shape_err("cuda matmul 2d inner mismatch", || {
            matmul_differentiable(&a24, &b35)
        });

        let a24_f64 = gpu64(&[1.0; 8], &[2, 4]);
        let b35_f64 = gpu64(&[1.0; 15], &[3, 5]);
        assert_shape_err_f64("cuda f64 matmul 2d inner mismatch", || {
            matmul_differentiable(&a24_f64, &b35_f64)
        });
    }

    #[test]
    fn cuda_vector_linalg_paths_reject_length_mismatch_before_dispatch() {
        ensure_cuda_backend();

        let a23 = gpu(&t(&[1.0; 6], &[2, 3]));
        let v2 = gpu(&t(&[1.0, 2.0], &[2]));
        let v3 = gpu(&t(&[1.0, 2.0, 3.0], &[3]));

        assert_shape_err("cuda mv short vector", || mv_differentiable(&a23, &v2));
        assert_shape_err("cuda dot length mismatch", || dot_differentiable(&v2, &v3));
    }

    #[test]
    fn cuda_core_linalg_wrappers_reject_mixed_devices_at_boundary() {
        ensure_cuda_backend();

        let cpu_a23 = t(&[1.0; 6], &[2, 3]);
        let cpu_b34 = t(&[1.0; 12], &[3, 4]);
        let cpu_w43 = t(&[1.0; 12], &[4, 3]);
        let cpu_bias4 = t(&[1.0; 4], &[4]);
        let cpu_v3 = t(&[1.0, 2.0, 3.0], &[3]);

        let gpu_a23 = gpu(&cpu_a23);
        let gpu_b34 = gpu(&cpu_b34);
        let gpu_w43 = gpu(&cpu_w43);
        let gpu_v3 = gpu(&cpu_v3);

        assert_device_err("mm cpu/cuda", || mm_differentiable(&cpu_a23, &gpu_b34));
        assert_device_err("mm cuda/cpu", || mm_differentiable(&gpu_a23, &cpu_b34));
        assert_device_err("mm_bt cpu/cuda", || {
            mm_bt_differentiable(&cpu_a23, &gpu_w43)
        });
        assert_device_err("mv cuda/cpu", || mv_differentiable(&gpu_a23, &cpu_v3));
        assert_device_err("dot cpu/cuda", || dot_differentiable(&cpu_v3, &gpu_v3));
        assert_device_err("linear mixed bias", || {
            linear_fused(&gpu_a23, &gpu_w43, Some(&cpu_bias4))
        });
    }
}
