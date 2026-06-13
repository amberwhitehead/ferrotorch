//! CORE-126 (#1820, CLASS-V High): public gather/scatter APIs must enforce
//! PyTorch's non-dim index-shape rule and must execute valid smaller non-dim
//! CUDA layouts on device.
//!
//! Upstream source contract:
//! `aten/src/ATen/native/ScatterGatherChecks.h::gather_shape_check` requires
//! `index.size(d) <= self.size(d)` for every `d != dim`; `scatter_shape_check`
//! applies the same self-side rule and additionally checks `src`.
//! The scatter axis itself is governed by per-value index bounds.
//!
//! Pre-fix code audit:
//! `ops/indexing.rs::validate_gather_shapes` only checked rank, index-slice
//! length, and index values. CPU coordinate loops therefore could index past
//! input/output for larger non-dim axes. CUDA forward/backward used dim-aware
//! `[outer, axis, inner]` kernels factored from the input shape, rejecting
//! valid compact non-dim index shapes with "until #1820" errors.

use ferrotorch_core::{FerrotorchError, Tensor, TensorStorage, gather, scatter, scatter_add};

fn cpu_f32(data: &[f32], shape: &[usize], rg: bool) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), rg).unwrap()
}

fn input_2x3(rg: bool) -> Tensor<f32> {
    cpu_f32(&[10.0, 11.0, 12.0, 20.0, 21.0, 22.0], &[2, 3], rg)
}

fn zeros_2x3(rg: bool) -> Tensor<f32> {
    cpu_f32(&[0.0; 6], &[2, 3], rg)
}

fn src_1x2(rg: bool) -> Tensor<f32> {
    cpu_f32(&[5.0, 6.0], &[1, 2], rg)
}

#[track_caller]
fn assert_shape_err<T: std::fmt::Debug>(r: Result<T, FerrotorchError>, label: &str) {
    match r {
        Err(FerrotorchError::ShapeMismatch { .. } | FerrotorchError::InvalidArgument { .. }) => {}
        Err(other) => panic!("{label}: expected shape/argument error, got {other:?}"),
        Ok(v) => panic!("{label}: expected error, got Ok({v:?})"),
    }
}

#[test]
fn core126_cpu_gather_rejects_larger_non_dim_axis() {
    let index = [0usize, 1, 2, 0, 1, 2];
    let r = gather(&input_2x3(false), 1, &index, &[3, 2]);
    assert_shape_err(
        r.map(|t| t.data_vec()),
        "gather index rows exceed input rows",
    );
}

#[test]
fn core126_cpu_scatter_family_rejects_larger_non_dim_axis() {
    let index = [2usize, 0, 1, 2, 0, 1];
    let src = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], false);

    assert_shape_err(
        scatter(&zeros_2x3(false), 1, &index, &[3, 2], &src).map(|t| t.data_vec()),
        "scatter index rows exceed input rows",
    );
    assert_shape_err(
        ferrotorch_core::ops::indexing::scatter_value(&zeros_2x3(false), 1, &index, &[3, 2], 9.0)
            .map(|t| t.data_vec()),
        "scatter_value index rows exceed input rows",
    );
    assert_shape_err(
        scatter_add(&zeros_2x3(false), 1, &index, &[3, 2], &src).map(|t| t.data_vec()),
        "scatter_add index rows exceed input rows",
    );
}

#[test]
fn core126_cpu_smaller_non_dim_axis_matches_torch_contract() {
    let idx = [2usize, 0];
    let gathered = gather(&input_2x3(false), 1, &idx, &[1, 2]).unwrap();
    assert_eq!(gathered.shape(), &[1, 2]);
    assert_eq!(gathered.data_vec().unwrap(), vec![12.0, 10.0]);

    let scattered = scatter(&zeros_2x3(false), 1, &idx, &[1, 2], &src_1x2(false)).unwrap();
    assert_eq!(
        scattered.data_vec().unwrap(),
        vec![6.0, 0.0, 5.0, 0.0, 0.0, 0.0]
    );

    let added = scatter_add(&zeros_2x3(false), 1, &idx, &[1, 2], &src_1x2(false)).unwrap();
    assert_eq!(
        added.data_vec().unwrap(),
        vec![6.0, 0.0, 5.0, 0.0, 0.0, 0.0]
    );

    let valued =
        ferrotorch_core::ops::indexing::scatter_value(&zeros_2x3(false), 1, &idx, &[1, 2], 9.0)
            .unwrap();
    assert_eq!(
        valued.data_vec().unwrap(),
        vec![9.0, 0.0, 9.0, 0.0, 0.0, 0.0]
    );
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use ferrotorch_core::Device;
    use ferrotorch_core::autograd::graph::backward;
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for CORE-126 probes");
        });
    }

    fn cuda(t: Tensor<f32>, rg: bool) -> Tensor<f32> {
        t.to(Device::Cuda(0)).unwrap().requires_grad_(rg)
    }

    #[test]
    fn core126_cuda_smaller_non_dim_forward_stays_cuda() {
        ensure_cuda_backend();
        let idx = [2usize, 0];
        let input = cuda(input_2x3(false), false);
        let zeros = cuda(zeros_2x3(false), false);
        let src = cuda(src_1x2(false), false);

        let gathered = gather(&input, 1, &idx, &[1, 2]).unwrap();
        assert!(gathered.is_cuda(), "gather must stay CUDA-resident");
        assert_eq!(
            gathered.cpu().unwrap().data_vec().unwrap(),
            vec![12.0, 10.0]
        );

        let scattered = scatter(&zeros, 1, &idx, &[1, 2], &src).unwrap();
        assert!(scattered.is_cuda(), "scatter must stay CUDA-resident");
        assert_eq!(
            scattered.cpu().unwrap().data_vec().unwrap(),
            vec![6.0, 0.0, 5.0, 0.0, 0.0, 0.0],
        );

        let added = scatter_add(&zeros, 1, &idx, &[1, 2], &src).unwrap();
        assert!(added.is_cuda(), "scatter_add must stay CUDA-resident");
        assert_eq!(
            added.cpu().unwrap().data_vec().unwrap(),
            vec![6.0, 0.0, 5.0, 0.0, 0.0, 0.0],
        );

        let valued =
            ferrotorch_core::ops::indexing::scatter_value(&zeros, 1, &idx, &[1, 2], 9.0).unwrap();
        assert!(valued.is_cuda(), "scatter_value must stay CUDA-resident");
        assert_eq!(
            valued.cpu().unwrap().data_vec().unwrap(),
            vec![9.0, 0.0, 9.0, 0.0, 0.0, 0.0],
        );
    }

    #[test]
    fn core126_cuda_gather_backward_smaller_non_dim_stays_cuda() {
        ensure_cuda_backend();
        let idx = [2usize, 0];
        let input = cuda(input_2x3(false), true);
        let out = gather(&input, 1, &idx, &[1, 2]).unwrap();
        backward(&out.sum_all().unwrap()).unwrap();

        let grad = input.grad().unwrap().expect("grad must reach input");
        assert!(grad.is_cuda(), "gather grad must stay CUDA-resident");
        assert_eq!(
            grad.cpu().unwrap().data_vec().unwrap(),
            vec![1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        );
    }

    #[test]
    fn core126_cuda_scatter_backward_smaller_non_dim_stays_cuda() {
        ensure_cuda_backend();
        let idx = [2usize, 0];
        let input = cuda(zeros_2x3(false), true);
        let src = cuda(src_1x2(false), true);
        let out = scatter(&input, 1, &idx, &[1, 2], &src).unwrap();
        backward(&out.sum_all().unwrap()).unwrap();

        let gi = input.grad().unwrap().expect("grad must reach input");
        let gs = src.grad().unwrap().expect("grad must reach src");
        assert!(
            gi.is_cuda() && gs.is_cuda(),
            "scatter grads must stay CUDA-resident"
        );
        assert_eq!(
            gi.cpu().unwrap().data_vec().unwrap(),
            vec![0.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        );
        assert_eq!(gs.cpu().unwrap().data_vec().unwrap(), vec![1.0, 1.0]);
    }

    #[test]
    fn core126_cuda_scatter_add_backward_smaller_non_dim_stays_cuda() {
        ensure_cuda_backend();
        let idx = [2usize, 0];
        let input = cuda(zeros_2x3(false), true);
        let src = cuda(src_1x2(false), true);
        let out = scatter_add(&input, 1, &idx, &[1, 2], &src).unwrap();
        backward(&out.sum_all().unwrap()).unwrap();

        let gi = input.grad().unwrap().expect("grad must reach input");
        let gs = src.grad().unwrap().expect("grad must reach src");
        assert!(
            gi.is_cuda() && gs.is_cuda(),
            "scatter_add grads must stay CUDA-resident"
        );
        assert_eq!(gi.cpu().unwrap().data_vec().unwrap(), vec![1.0; 6]);
        assert_eq!(gs.cpu().unwrap().data_vec().unwrap(), vec![1.0, 1.0]);
    }
}
