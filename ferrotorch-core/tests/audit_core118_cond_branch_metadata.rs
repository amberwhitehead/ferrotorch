use ferrotorch_core::Tensor;
use ferrotorch_core::grad_fns::arithmetic::{add, mul};
use ferrotorch_core::ops::higher_order::{cond, validate_cond_branches};
use ferrotorch_core::storage::TensorStorage;

fn cpu_tensor(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
    Tensor::from_storage(
        TensorStorage::cpu(data.to_vec()),
        shape.to_vec(),
        requires_grad,
    )
    .expect("cpu tensor")
}

fn assert_error_contains(
    result: ferrotorch_core::FerrotorchResult<Vec<Tensor<f32>>>,
    needle: &str,
) {
    let err = result.expect_err("expected cond metadata validation failure");
    let msg = err.to_string();
    assert!(
        msg.contains(needle),
        "expected error containing {needle:?}, got {msg:?}"
    );
}

#[test]
fn cond_rejects_output_count_mismatch_even_when_true_branch_is_selected() {
    // PyTorch 2.11 oracle: torch.cond rejects differing branch pytrees before
    // using the predicate, so the untaken false branch is still validated.
    let pred = cpu_tensor(&[1.0], &[], false);
    let x = cpu_tensor(&[1.0, 2.0], &[2], false);

    let result = cond(
        &pred,
        |ops| vec![add(&ops[0], &ops[0]).expect("add")],
        |ops| {
            vec![
                add(&ops[0], &ops[0]).expect("add"),
                mul(&ops[0], &ops[0]).expect("mul"),
            ]
        },
        std::slice::from_ref(&x),
    );

    assert_error_contains(
        result,
        "true branch returns 1 tensors but false branch returns 2",
    );
}

#[test]
fn cond_rejects_output_count_mismatch_even_when_false_branch_is_selected() {
    let pred = cpu_tensor(&[0.0], &[], false);
    let x = cpu_tensor(&[1.0, 2.0], &[2], false);

    let result = cond(
        &pred,
        |ops| vec![add(&ops[0], &ops[0]).expect("add")],
        |ops| {
            vec![
                add(&ops[0], &ops[0]).expect("add"),
                mul(&ops[0], &ops[0]).expect("mul"),
            ]
        },
        std::slice::from_ref(&x),
    );

    assert_error_contains(
        result,
        "true branch returns 1 tensors but false branch returns 2",
    );
}

#[test]
fn cond_rejects_static_shape_mismatch_from_untaken_branch() {
    // Ferrotorch has static sizes, so unlike PyTorch's symbolic shape merge,
    // concrete branch size differences are rejected directly.
    let pred = cpu_tensor(&[1.0], &[], false);
    let x = cpu_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);

    let result = cond(
        &pred,
        |ops| vec![ops[0].as_strided(&[6], &[1], Some(0)).expect("flat view")],
        |ops| vec![ops[0].clone()],
        std::slice::from_ref(&x),
    );

    assert_error_contains(result, "shape mismatch");
}

#[test]
fn cond_rejects_same_shape_stride_mismatch() {
    let pred = cpu_tensor(&[1.0], &[], false);
    let x = cpu_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);

    let result = cond(
        &pred,
        |ops| vec![ops[0].clone()],
        |ops| {
            let transposed = ops[0].t().expect("transpose");
            let materialized = transposed.contiguous().expect("contiguous");
            vec![materialized.t().expect("transpose back")]
        },
        std::slice::from_ref(&x),
    );

    assert_error_contains(result, "stride mismatch");
}

#[test]
fn cond_rejects_same_shape_stride_and_device_but_different_storage_offset() {
    let pred = cpu_tensor(&[1.0], &[], false);
    let x = cpu_tensor(&[1.0, 2.0, 3.0, 4.0], &[4], false);

    let result = cond(
        &pred,
        |ops| vec![ops[0].as_strided(&[2], &[1], Some(0)).expect("offset 0")],
        |ops| vec![ops[0].as_strided(&[2], &[1], Some(1)).expect("offset 1")],
        std::slice::from_ref(&x),
    );

    assert_error_contains(result, "storage_offset mismatch");
}

#[test]
fn validate_cond_branches_accepts_matching_cpu_metadata() {
    let cpu = cpu_tensor(&[1.0, 2.0], &[2], false);
    let other_cpu = cpu_tensor(&[3.0, 4.0], &[2], false);
    validate_cond_branches(std::slice::from_ref(&cpu), std::slice::from_ref(&other_cpu))
        .expect("matching CPU metadata");
}

#[cfg(feature = "gpu")]
mod cuda {
    use super::{Tensor, assert_error_contains, cond, cpu_tensor};
    use ferrotorch_core::device::Device;
    use ferrotorch_core::grad_fns::arithmetic::add;
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for CORE-118 probes");
        });
    }

    fn cuda_tensor(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
        cpu_tensor(data, shape, false)
            .to(Device::Cuda(0))
            .expect("to cuda")
            .requires_grad_(requires_grad)
    }

    #[test]
    fn cond_rejects_cpu_cuda_branch_device_mismatch_before_selection() {
        ensure_cuda_backend();
        let pred = cpu_tensor(&[1.0], &[], false)
            .to(Device::Cuda(0))
            .expect("predicate to cuda");
        let x = cuda_tensor(&[1.0, 2.0], &[2], false);

        let result = cond(
            &pred,
            |ops| vec![add(&ops[0], &ops[0]).expect("cuda add")],
            |_| vec![cpu_tensor(&[2.0, 4.0], &[2], false)],
            std::slice::from_ref(&x),
        );

        assert_error_contains(result, "device mismatch");
    }

    #[test]
    fn cond_accepts_matching_cuda_branch_metadata_and_keeps_selected_output_cuda() {
        ensure_cuda_backend();
        let pred = cpu_tensor(&[0.0], &[], false)
            .to(Device::Cuda(0))
            .expect("predicate to cuda");
        let x = cuda_tensor(&[1.0, 2.0], &[2], false);

        let out = cond(
            &pred,
            |ops| vec![add(&ops[0], &ops[0]).expect("true cuda add")],
            |ops| vec![add(&ops[0], &ops[0]).expect("false cuda add")],
            std::slice::from_ref(&x),
        )
        .expect("matching cuda metadata");

        assert_eq!(out.len(), 1);
        assert!(out[0].is_cuda(), "selected cond output must remain CUDA");
    }
}
