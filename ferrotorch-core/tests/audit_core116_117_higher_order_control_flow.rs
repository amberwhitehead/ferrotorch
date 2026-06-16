use ferrotorch_core::Tensor;
use ferrotorch_core::creation::ones_like;
use ferrotorch_core::grad_fns::arithmetic::mul;
use ferrotorch_core::grad_fns::reduction::sum;
use ferrotorch_core::ops::higher_order::{cond, scan};
use ferrotorch_core::storage::TensorStorage;

fn cpu_tensor(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
    Tensor::from_storage(
        TensorStorage::cpu(data.to_vec()),
        shape.to_vec(),
        requires_grad,
    )
    .expect("cpu tensor")
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch, actual={actual:?}, expected={expected:?}"
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "{label}[{i}]: expected {e}, got {a}; actual={actual:?}, expected={expected:?}"
        );
    }
}

#[test]
fn cond_detached_branch_output_matches_torch_zero_grad_contract() {
    // PyTorch 2.11 oracle:
    // torch.cond(pred, lambda x: torch.ones_like(x).detach(), ..., (x,))
    // returns requires_grad=True, and x.grad is zeros after backward.
    let pred = cpu_tensor(&[1.0], &[], false);
    let x = cpu_tensor(&[1.0, 2.0, 3.0], &[3], true);

    let out = cond(
        &pred,
        |ops| vec![ones_like(&ops[0]).expect("ones_like").detach()],
        |ops| vec![mul(&ops[0], &ops[0]).expect("mul")],
        std::slice::from_ref(&x),
    )
    .expect("cond");

    assert_eq!(out.len(), 1);
    assert!(out[0].requires_grad());
    assert_eq!(out[0].device(), x.device());
    assert_close(
        &out[0].data_vec().expect("out"),
        &[1.0, 1.0, 1.0],
        0.0,
        "cond detached output",
    );

    sum(&out[0]).expect("sum").backward().expect("backward");
    let gx = x.grad().expect("grad lookup").expect("x grad");
    assert_close(
        &gx.data_vec().expect("x grad"),
        &[0.0, 0.0, 0.0],
        0.0,
        "cond detached x.grad",
    );
}

#[test]
fn scan_detached_step_output_matches_torch_zero_grad_contract() {
    // PyTorch 2.11 scan emits ScanAutogradOp outputs requiring grad when scan
    // inputs require grad; detached step outputs backpropagate zero tensors.
    let init = cpu_tensor(&[0.0], &[1], true);
    let x0 = cpu_tensor(&[1.0], &[1], true);
    let x1 = cpu_tensor(&[2.0], &[1], true);
    let xs = vec![x0.clone(), x1.clone()];

    let (final_carry, outputs) = scan(
        |carry, x| {
            (
                ones_like(carry).expect("carry ones").detach(),
                ones_like(x).expect("output ones").detach(),
            )
        },
        &init,
        &xs,
    )
    .expect("scan");

    assert!(final_carry.requires_grad());
    assert_eq!(final_carry.device(), init.device());
    assert_eq!(outputs.len(), 2);
    assert!(outputs.iter().all(Tensor::requires_grad));

    sum(&outputs[0])
        .expect("sum output")
        .backward()
        .expect("backward");
    let ginit = init.grad().expect("init grad lookup").expect("init grad");
    let gx0 = x0.grad().expect("x0 grad lookup").expect("x0 grad");
    let gx1 = x1.grad().expect("x1 grad lookup").expect("x1 grad");
    assert_close(
        &ginit.data_vec().expect("init grad"),
        &[0.0],
        0.0,
        "init.grad",
    );
    assert_close(&gx0.data_vec().expect("x0 grad"), &[0.0], 0.0, "x0.grad");
    assert_close(&gx1.data_vec().expect("x1 grad"), &[0.0], 0.0, "x1.grad");
}

#[cfg(feature = "gpu")]
mod cuda {
    use super::{Tensor, assert_close, cond, cpu_tensor, mul, ones_like, scan, sum};
    use ferrotorch_core::device::Device;
    use ferrotorch_core::grad_fns::arithmetic::add;
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for CORE-116/117 probes");
        });
    }

    fn cuda_leaf(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        cpu_tensor(data, shape, false)
            .to(Device::Cuda(0))
            .expect("to cuda")
            .requires_grad_(true)
    }

    fn read_cuda(t: &Tensor<f32>, label: &str) -> Vec<f32> {
        assert!(t.is_cuda(), "{label}: tensor must be CUDA-resident");
        t.cpu()
            .expect("cuda to cpu")
            .data_vec()
            .expect("read cuda tensor")
    }

    #[test]
    fn cond_connected_output_and_grad_stay_cuda() {
        ensure_cuda_backend();
        let pred = cpu_tensor(&[1.0], &[], false);
        let x = cuda_leaf(&[1.0, 2.0, 3.0], &[3]);

        let out = cond(
            &pred,
            |ops| vec![mul(&ops[0], &ops[0]).expect("mul")],
            |ops| vec![add(&ops[0], &ops[0]).expect("add")],
            std::slice::from_ref(&x),
        )
        .expect("cond");

        assert!(out[0].is_cuda(), "cond output must stay CUDA-resident");
        assert!(out[0].requires_grad());
        assert_close(
            &read_cuda(&out[0], "cond connected output"),
            &[1.0, 4.0, 9.0],
            1e-6,
            "cond connected output",
        );

        sum(&out[0]).expect("sum").backward().expect("backward");
        let gx = x.grad().expect("grad lookup").expect("x grad");
        assert!(gx.is_cuda(), "cond input grad must stay CUDA-resident");
        assert_close(
            &read_cuda(&gx, "cond connected grad"),
            &[2.0, 4.0, 6.0],
            1e-6,
            "cond connected grad",
        );
    }

    #[test]
    fn cond_detached_output_and_zero_grad_stay_cuda() {
        ensure_cuda_backend();
        let pred = cpu_tensor(&[1.0], &[], false);
        let x = cuda_leaf(&[1.0, 2.0, 3.0], &[3]);

        let out = cond(
            &pred,
            |ops| vec![ones_like(&ops[0]).expect("ones_like").detach()],
            |ops| vec![mul(&ops[0], &ops[0]).expect("mul")],
            std::slice::from_ref(&x),
        )
        .expect("cond");

        assert!(out[0].is_cuda(), "detached cond output must stay CUDA");
        assert!(out[0].requires_grad());
        assert_close(
            &read_cuda(&out[0], "cond detached output"),
            &[1.0, 1.0, 1.0],
            0.0,
            "cond detached output",
        );

        sum(&out[0]).expect("sum").backward().expect("backward");
        let gx = x.grad().expect("grad lookup").expect("x grad");
        assert!(gx.is_cuda(), "zero grad must stay CUDA-resident");
        assert_close(
            &read_cuda(&gx, "cond detached grad"),
            &[0.0, 0.0, 0.0],
            0.0,
            "cond detached grad",
        );
    }

    #[test]
    fn cond_noncontiguous_branch_output_preserves_cuda_view() {
        ensure_cuda_backend();
        let pred = cpu_tensor(&[1.0], &[], false);
        let x = cuda_leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

        let out = cond(
            &pred,
            |ops| vec![ops[0].t().expect("transpose")],
            |ops| vec![ops[0].clone()],
            std::slice::from_ref(&x),
        )
        .expect("cond");

        assert!(
            out[0].is_cuda(),
            "non-contiguous cond output must stay CUDA"
        );
        assert!(
            !out[0].is_contiguous(),
            "cond wrapper must preserve non-contiguous view metadata"
        );
        assert_eq!(out[0].shape(), &[3, 2]);
        assert_close(
            &read_cuda(&out[0], "cond noncontig output"),
            &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
            1e-6,
            "cond noncontig output",
        );

        sum(&out[0]).expect("sum").backward().expect("backward");
        let gx = x.grad().expect("grad lookup").expect("x grad");
        assert!(gx.is_cuda(), "non-contiguous cond grad must stay CUDA");
        assert_close(
            &read_cuda(&gx, "cond noncontig grad"),
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            1e-6,
            "cond noncontig grad",
        );
    }

    #[test]
    fn scan_connected_carry_output_and_grad_stay_cuda() {
        ensure_cuda_backend();
        let init = cuda_leaf(&[0.0], &[1]);
        let x0 = cuda_leaf(&[2.0], &[1]);
        let x1 = cuda_leaf(&[3.0], &[1]);
        let xs = vec![x0.clone(), x1.clone()];

        let (final_carry, outputs) = scan(
            |carry, x| {
                let next = add(carry, x).expect("add");
                let output = mul(&next, x).expect("mul");
                (next, output)
            },
            &init,
            &xs,
        )
        .expect("scan");

        assert!(final_carry.is_cuda(), "scan final carry must stay CUDA");
        assert!(outputs.iter().all(Tensor::is_cuda));
        assert_close(
            &read_cuda(&final_carry, "scan final carry"),
            &[5.0],
            1e-6,
            "scan final carry",
        );
        assert_close(
            &read_cuda(&outputs[0], "scan output 0"),
            &[4.0],
            1e-6,
            "scan output 0",
        );
        assert_close(
            &read_cuda(&outputs[1], "scan output 1"),
            &[15.0],
            1e-6,
            "scan output 1",
        );

        sum(&final_carry)
            .expect("sum")
            .backward()
            .expect("backward");
        for (label, tensor) in [("init", &init), ("x0", &x0), ("x1", &x1)] {
            let grad = tensor
                .grad()
                .expect("grad lookup")
                .unwrap_or_else(|| panic!("{label} grad"));
            assert!(grad.is_cuda(), "{label} grad must stay CUDA-resident");
            assert_close(
                &read_cuda(&grad, label),
                &[1.0],
                1e-6,
                &format!("{label}.grad"),
            );
        }
    }

    #[test]
    fn scan_detached_output_zero_grads_stay_cuda() {
        ensure_cuda_backend();
        let init = cuda_leaf(&[0.0], &[1]);
        let x0 = cuda_leaf(&[2.0], &[1]);
        let x1 = cuda_leaf(&[3.0], &[1]);
        let xs = vec![x0.clone(), x1.clone()];

        let (final_carry, outputs) = scan(
            |carry, x| {
                (
                    ones_like(carry).expect("carry ones").detach(),
                    ones_like(x).expect("output ones").detach(),
                )
            },
            &init,
            &xs,
        )
        .expect("scan");

        assert!(final_carry.is_cuda(), "detached scan carry must stay CUDA");
        assert!(outputs.iter().all(Tensor::is_cuda));
        assert!(final_carry.requires_grad());
        assert!(outputs.iter().all(Tensor::requires_grad));

        sum(&outputs[0])
            .expect("sum output")
            .backward()
            .expect("backward");
        for (label, tensor) in [("init", &init), ("x0", &x0), ("x1", &x1)] {
            let grad = tensor
                .grad()
                .expect("grad lookup")
                .unwrap_or_else(|| panic!("{label} grad"));
            assert!(grad.is_cuda(), "{label} zero grad must stay CUDA-resident");
            assert_close(
                &read_cuda(&grad, label),
                &[0.0],
                0.0,
                &format!("{label}.zero_grad"),
            );
        }
    }
}
