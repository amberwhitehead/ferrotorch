//! Regression coverage for CORE-053/#1747: broadcast arithmetic must not have
//! a fixed 16-dimension coordinate ceiling.
//!
//! Live oracle, torch 2.11.0+cu130:
//! ```python
//! shape = [1] * 32 + [3]
//! scalar = [1] * 33
//! a = torch.tensor([5.5, -5.5, 7.25]).reshape(shape)
//! b = torch.tensor([2.0]).reshape(scalar)
//! torch.remainder(a, b).reshape(-1)     # [1.5, 0.5, 1.25]
//! torch.fmod(a, b).reshape(-1)          # [1.5, -1.5, 1.25]
//! torch.floor_divide(a, b).reshape(-1)  # [2.0, -3.0, 3.0]
//! ```
//! `torch.addcmul` and `torch.addcdiv` use the same TensorIterator broadcast
//! metadata path and accept ranks 16, 17, and 33 on CPU and CUDA.

use ferrotorch_core::Tensor;
use ferrotorch_core::autograd::backward;
use ferrotorch_core::creation::from_vec;
use ferrotorch_core::grad_fns::arithmetic::{addcdiv, addcmul, floor_divide, fmod, mul, remainder};
use ferrotorch_core::grad_fns::reduction::sum;

fn tail_shape(rank: usize, tail: usize) -> Vec<usize> {
    assert!(rank > 0);
    let mut shape = vec![1; rank - 1];
    shape.push(tail);
    shape
}

fn scalar_shape(rank: usize) -> Vec<usize> {
    vec![1; rank]
}

fn cpu_f64(values: Vec<f64>, shape: &[usize], requires_grad: bool) -> Tensor<f64> {
    from_vec(values, shape)
        .expect("f64 tensor")
        .requires_grad_(requires_grad)
}

fn assert_close_f64(label: &str, actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch {actual:?} vs {expected:?}"
    );
    for (idx, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (got - want).abs() <= tol,
            "{label}[{idx}]: got {got}, expected {want}, actual={actual:?}"
        );
    }
}

fn assert_shape_and_values(label: &str, tensor: &Tensor<f64>, shape: &[usize], expected: &[f64]) {
    assert_eq!(tensor.shape(), shape, "{label}: shape mismatch");
    assert_close_f64(
        label,
        &tensor.data_vec().expect("cpu data"),
        expected,
        1e-12,
    );
}

#[test]
fn cpu_rank_16_17_33_forward_broadcast_ops_match_torch() {
    for rank in [16, 17, 33] {
        let out_shape = tail_shape(rank, 3);
        let rhs_shape = scalar_shape(rank);
        let a = cpu_f64(vec![5.5, -5.5, 7.25], &out_shape, false);
        let b = cpu_f64(vec![2.0], &rhs_shape, false);

        assert_shape_and_values(
            &format!("rank {rank} remainder"),
            &remainder(&a, &b).expect("remainder"),
            &out_shape,
            &[1.5, 0.5, 1.25],
        );
        assert_shape_and_values(
            &format!("rank {rank} fmod"),
            &fmod(&a, &b).expect("fmod"),
            &out_shape,
            &[1.5, -1.5, 1.25],
        );
        assert_shape_and_values(
            &format!("rank {rank} floor_divide"),
            &floor_divide(&a, &b).expect("floor_divide"),
            &out_shape,
            &[2.0, -3.0, 3.0],
        );

        let input = cpu_f64(vec![1.0], &rhs_shape, false);
        let t1 = cpu_f64(vec![2.0, 4.0, 6.0], &out_shape, false);
        let t2 = cpu_f64(vec![10.0, 20.0, 30.0], &out_shape, false);
        assert_shape_and_values(
            &format!("rank {rank} addcmul"),
            &addcmul(&input, &t1, &t2, 0.5).expect("addcmul"),
            &out_shape,
            &[11.0, 41.0, 91.0],
        );
        assert_shape_and_values(
            &format!("rank {rank} addcdiv"),
            &addcdiv(&input, &t1, &t2, 0.5).expect("addcdiv"),
            &out_shape,
            &[1.1, 1.1, 1.1],
        );
    }
}

#[test]
fn cpu_rank_33_broadcast_grad_reducer_matches_torch() {
    let rank = 33;
    let out_shape = tail_shape(rank, 3);
    let lhs_shape = scalar_shape(rank);
    let lhs = cpu_f64(vec![2.0], &lhs_shape, true);
    let rhs = cpu_f64(vec![1.0, 2.0, 3.0], &out_shape, true);

    let y = mul(&lhs, &rhs).expect("mul");
    let loss = sum(&y).expect("sum");
    backward(&loss).expect("backward");

    let lhs_grad = lhs.grad().expect("lhs grad result").expect("lhs grad");
    assert_eq!(lhs_grad.shape(), lhs_shape.as_slice());
    assert_close_f64(
        "rank 33 scalar-side broadcast grad",
        &lhs_grad.data_vec().expect("lhs grad data"),
        &[6.0],
        1e-12,
    );

    let rhs_grad = rhs.grad().expect("rhs grad result").expect("rhs grad");
    assert_eq!(rhs_grad.shape(), out_shape.as_slice());
    assert_close_f64(
        "rank 33 tail-side broadcast grad",
        &rhs_grad.data_vec().expect("rhs grad data"),
        &[2.0, 2.0, 2.0],
        1e-12,
    );
}

#[test]
fn cpu_rank_33_addcmul_addcdiv_backward_match_torch() {
    let rank = 33;
    let out_shape = tail_shape(rank, 3);
    let input_shape = scalar_shape(rank);

    let input = cpu_f64(vec![1.0], &input_shape, true);
    let t1 = cpu_f64(vec![2.0, 4.0, 6.0], &out_shape, true);
    let t2 = cpu_f64(vec![10.0, 20.0, 30.0], &out_shape, true);
    let y = addcmul(&input, &t1, &t2, 0.5).expect("addcmul");
    backward(&sum(&y).expect("sum addcmul")).expect("backward addcmul");

    assert_close_f64(
        "addcmul input grad",
        &input
            .grad()
            .expect("addcmul input grad result")
            .expect("addcmul input grad")
            .data_vec()
            .expect("addcmul input grad data"),
        &[3.0],
        1e-12,
    );
    assert_close_f64(
        "addcmul tensor1 grad",
        &t1.grad()
            .expect("addcmul t1 grad result")
            .expect("addcmul t1 grad")
            .data_vec()
            .expect("addcmul t1 grad data"),
        &[5.0, 10.0, 15.0],
        1e-12,
    );
    assert_close_f64(
        "addcmul tensor2 grad",
        &t2.grad()
            .expect("addcmul t2 grad result")
            .expect("addcmul t2 grad")
            .data_vec()
            .expect("addcmul t2 grad data"),
        &[1.0, 2.0, 3.0],
        1e-12,
    );

    let input = cpu_f64(vec![1.0], &input_shape, true);
    let t1 = cpu_f64(vec![2.0, 4.0, 6.0], &out_shape, true);
    let t2 = cpu_f64(vec![10.0, 20.0, 30.0], &out_shape, true);
    let y = addcdiv(&input, &t1, &t2, 0.5).expect("addcdiv");
    backward(&sum(&y).expect("sum addcdiv")).expect("backward addcdiv");

    assert_close_f64(
        "addcdiv input grad",
        &input
            .grad()
            .expect("addcdiv input grad result")
            .expect("addcdiv input grad")
            .data_vec()
            .expect("addcdiv input grad data"),
        &[3.0],
        1e-12,
    );
    assert_close_f64(
        "addcdiv tensor1 grad",
        &t1.grad()
            .expect("addcdiv t1 grad result")
            .expect("addcdiv t1 grad")
            .data_vec()
            .expect("addcdiv t1 grad data"),
        &[0.05, 0.025, 1.0 / 60.0],
        1e-12,
    );
    assert_close_f64(
        "addcdiv tensor2 grad",
        &t2.grad()
            .expect("addcdiv t2 grad result")
            .expect("addcdiv t2 grad")
            .data_vec()
            .expect("addcdiv t2 grad data"),
        &[-0.01, -0.005, -1.0 / 300.0],
        1e-12,
    );
}

#[cfg(feature = "gpu")]
mod cuda {
    use super::*;
    use ferrotorch_core::Device;
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for rank broadcast probes");
        });
    }

    fn cuda_f32(values: Vec<f32>, shape: &[usize], requires_grad: bool) -> Tensor<f32> {
        from_vec(values, shape)
            .expect("f32 CPU tensor")
            .to(Device::Cuda(0))
            .expect("upload f32")
            .requires_grad_(requires_grad)
    }

    fn cuda_f64(values: Vec<f64>, shape: &[usize], requires_grad: bool) -> Tensor<f64> {
        from_vec(values, shape)
            .expect("f64 CPU tensor")
            .to(Device::Cuda(0))
            .expect("upload f64")
            .requires_grad_(requires_grad)
    }

    fn host_f32(label: &str, t: &Tensor<f32>) -> Vec<f32> {
        assert_eq!(
            t.device(),
            Device::Cuda(0),
            "{label}: tensor must remain CUDA-resident before explicit readback"
        );
        t.cpu().expect("D2H f32").data_vec().expect("f32 data")
    }

    fn host_f64(label: &str, t: &Tensor<f64>) -> Vec<f64> {
        assert_eq!(
            t.device(),
            Device::Cuda(0),
            "{label}: tensor must remain CUDA-resident before explicit readback"
        );
        t.cpu().expect("D2H f64").data_vec().expect("f64 data")
    }

    fn assert_close_f32(label: &str, actual: &[f32], expected: &[f32], tol: f32) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{label}: length mismatch {actual:?} vs {expected:?}"
        );
        for (idx, (&got, &want)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (got - want).abs() <= tol,
                "{label}[{idx}]: got {got}, expected {want}, actual={actual:?}"
            );
        }
    }

    #[test]
    fn cuda_rank_17_and_33_forward_broadcast_ops_match_torch_and_stay_resident() {
        ensure_cuda_backend();

        for rank in [17, 33] {
            let out_shape = tail_shape(rank, 3);
            let rhs_shape = scalar_shape(rank);

            let a = cuda_f32(vec![5.5, -5.5, 7.25], &out_shape, false);
            let b = cuda_f32(vec![2.0], &rhs_shape, false);
            assert_close_f32(
                &format!("cuda f32 rank {rank} remainder"),
                &host_f32(
                    "cuda f32 remainder",
                    &remainder(&a, &b).expect("cuda f32 remainder"),
                ),
                &[1.5, 0.5, 1.25],
                1e-6,
            );
            assert_close_f32(
                &format!("cuda f32 rank {rank} fmod"),
                &host_f32("cuda f32 fmod", &fmod(&a, &b).expect("cuda f32 fmod")),
                &[1.5, -1.5, 1.25],
                1e-6,
            );
            assert_close_f32(
                &format!("cuda f32 rank {rank} floor_divide"),
                &host_f32(
                    "cuda f32 floor_divide",
                    &floor_divide(&a, &b).expect("cuda f32 floor_divide"),
                ),
                &[2.0, -3.0, 3.0],
                1e-6,
            );

            let input = cuda_f32(vec![1.0], &rhs_shape, false);
            let t1 = cuda_f32(vec![2.0, 4.0, 6.0], &out_shape, false);
            let t2 = cuda_f32(vec![10.0, 20.0, 30.0], &out_shape, false);
            assert_close_f32(
                &format!("cuda f32 rank {rank} addcmul"),
                &host_f32(
                    "cuda f32 addcmul",
                    &addcmul(&input, &t1, &t2, 0.5).expect("cuda f32 addcmul"),
                ),
                &[11.0, 41.0, 91.0],
                1e-5,
            );
            assert_close_f32(
                &format!("cuda f32 rank {rank} addcdiv"),
                &host_f32(
                    "cuda f32 addcdiv",
                    &addcdiv(&input, &t1, &t2, 0.5).expect("cuda f32 addcdiv"),
                ),
                &[1.1, 1.1, 1.1],
                1e-6,
            );

            let a = cuda_f64(vec![5.5, -5.5, 7.25], &out_shape, false);
            let b = cuda_f64(vec![2.0], &rhs_shape, false);
            assert_close_f64(
                &format!("cuda f64 rank {rank} remainder"),
                &host_f64(
                    "cuda f64 remainder",
                    &remainder(&a, &b).expect("cuda f64 remainder"),
                ),
                &[1.5, 0.5, 1.25],
                1e-12,
            );
            assert_close_f64(
                &format!("cuda f64 rank {rank} floor_divide"),
                &host_f64(
                    "cuda f64 floor_divide",
                    &floor_divide(&a, &b).expect("cuda f64 floor_divide"),
                ),
                &[2.0, -3.0, 3.0],
                1e-12,
            );
        }
    }

    #[test]
    fn cuda_rank_33_addcmul_backward_reduces_broadcast_axes_on_device() {
        ensure_cuda_backend();

        let rank = 33;
        let out_shape = tail_shape(rank, 3);
        let input_shape = scalar_shape(rank);
        let input = cuda_f32(vec![1.0], &input_shape, true);
        let t1 = cuda_f32(vec![2.0, 4.0, 6.0], &out_shape, true);
        let t2 = cuda_f32(vec![10.0, 20.0, 30.0], &out_shape, true);

        let y = addcmul(&input, &t1, &t2, 0.5).expect("cuda addcmul");
        backward(&sum(&y).expect("cuda sum addcmul")).expect("cuda backward addcmul");

        assert_close_f32(
            "cuda addcmul input grad",
            &host_f32(
                "cuda addcmul input grad",
                &input
                    .grad()
                    .expect("input grad result")
                    .expect("input grad"),
            ),
            &[3.0],
            1e-6,
        );
        assert_close_f32(
            "cuda addcmul tensor1 grad",
            &host_f32(
                "cuda addcmul tensor1 grad",
                &t1.grad()
                    .expect("tensor1 grad result")
                    .expect("tensor1 grad"),
            ),
            &[5.0, 10.0, 15.0],
            1e-5,
        );
        assert_close_f32(
            "cuda addcmul tensor2 grad",
            &host_f32(
                "cuda addcmul tensor2 grad",
                &t2.grad()
                    .expect("tensor2 grad result")
                    .expect("tensor2 grad"),
            ),
            &[1.0, 2.0, 3.0],
            1e-6,
        );
    }
}
