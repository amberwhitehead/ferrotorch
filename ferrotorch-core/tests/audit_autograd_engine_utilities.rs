use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::{env, process::Command, thread};

use ferrotorch_core::autograd::gradcheck::gradcheck;
use ferrotorch_core::autograd::graph::backward_parallel;
use ferrotorch_core::autograd::higher_order::grad;
use ferrotorch_core::autograd::hooks::HookHandle;
use ferrotorch_core::grad_fns::activation::relu;
use ferrotorch_core::grad_fns::arithmetic::{add, mul};
use ferrotorch_core::grad_fns::reduction::{mean, mean_dim, sum, sum_dim};
use ferrotorch_core::tensor::GradFn;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Tensor, TensorStorage};

fn leaf(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}

fn constant(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

#[derive(Debug)]
struct PassThroughBackward {
    input: Tensor<f32>,
}

impl GradFn<f32> for PassThroughBackward {
    fn backward(&self, grad_output: &Tensor<f32>) -> FerrotorchResult<Vec<Option<Tensor<f32>>>> {
        Ok(vec![Some(grad_output.clone())])
    }

    fn inputs(&self) -> Vec<&Tensor<f32>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "PassThroughBackward"
    }
}

#[derive(Debug)]
struct RequireGradOutputShape {
    inputs: Vec<Tensor<f32>>,
    expected_shape: Vec<usize>,
}

impl GradFn<f32> for RequireGradOutputShape {
    fn backward(&self, grad_output: &Tensor<f32>) -> FerrotorchResult<Vec<Option<Tensor<f32>>>> {
        if grad_output.shape() != self.expected_shape.as_slice() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "expected upstream gradient shape {:?}, got {:?}",
                    self.expected_shape,
                    grad_output.shape()
                ),
            });
        }

        let mut grads = Vec::with_capacity(self.inputs.len());
        grads.push(Some(grad_output.clone()));
        grads.resize_with(self.inputs.len(), || None);
        Ok(grads)
    }

    fn inputs(&self) -> Vec<&Tensor<f32>> {
        self.inputs.iter().collect()
    }

    fn name(&self) -> &'static str {
        "RequireGradOutputShape"
    }
}

#[derive(Debug)]
struct IntentionalFailBackward {
    inputs: Vec<Tensor<f32>>,
}

impl GradFn<f32> for IntentionalFailBackward {
    fn backward(&self, _grad_output: &Tensor<f32>) -> FerrotorchResult<Vec<Option<Tensor<f32>>>> {
        Err(FerrotorchError::InvalidArgument {
            message: "CORE-021 intentional backward failure".into(),
        })
    }

    fn inputs(&self) -> Vec<&Tensor<f32>> {
        self.inputs.iter().collect()
    }

    fn name(&self) -> &'static str {
        "IntentionalFailBackward"
    }
}

fn parallel_failure_graph() -> Tensor<f32> {
    let leaves: Vec<_> = (0..8).map(|idx| leaf(&[idx as f32], &[1])).collect();
    let failing = Tensor::from_operation(
        TensorStorage::cpu(vec![0.0]),
        vec![1],
        Arc::new(IntentionalFailBackward { inputs: leaves }),
    )
    .unwrap();
    Tensor::from_operation(
        TensorStorage::cpu(vec![0.0]),
        vec![1],
        Arc::new(PassThroughBackward { input: failing }),
    )
    .unwrap()
}

fn run_exact_child_before_timeout(test_name: &str, env_key: &str, failure_message: &str) {
    let exe = env::current_exe().expect("current test binary");
    let mut child = Command::new(exe)
        .arg("--exact")
        .arg(test_name)
        .arg("--nocapture")
        .env(env_key, "1")
        .spawn()
        .expect("spawn child test");

    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(status) = child.try_wait().expect("poll child") {
            assert!(
                status.success(),
                "child test {test_name} exited with {status}"
            );
            return;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("{failure_message}");
        }
        thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn backward_parallel_implicit_seed_preserves_singleton_root_shape() {
    let x = leaf(&[2.0], &[1]);
    let mut y = x.clone();
    for _ in 0..8 {
        y = add(&y, &x).expect("deep add");
    }

    backward_parallel(&y, None, 2).expect("backward_parallel");

    let grad = x.grad().expect("grad lookup").expect("x grad");
    assert_eq!(grad.shape(), &[1]);
    assert_eq!(grad.data().expect("grad data"), &[9.0]);
}

#[test]
fn backward_parallel_implicit_seed_preserves_2d_singleton_root_shape() {
    let x = leaf(&[2.0], &[1, 1]);
    let mut inputs = vec![x.clone()];
    for idx in 0..8 {
        inputs.push(constant(&[idx as f32], &[1]));
    }
    let y = Tensor::from_operation(
        TensorStorage::cpu(vec![2.0]),
        vec![1, 1],
        Arc::new(RequireGradOutputShape {
            inputs,
            expected_shape: vec![1, 1],
        }),
    )
    .expect("shape-checking root");

    backward_parallel(&y, None, 2).expect("backward_parallel");

    let grad = x.grad().expect("grad lookup").expect("x grad");
    assert_eq!(grad.shape(), &[1, 1]);
    assert_eq!(grad.data().expect("grad data"), &[1.0]);
}

#[test]
fn backward_parallel_error_path_returns_before_timeout() {
    run_exact_child_before_timeout(
        "backward_parallel_error_path_child",
        "FERROTORCH_CORE021_CHILD",
        "backward_parallel did not return after a worker failure",
    );
}

#[test]
fn backward_parallel_error_path_child() {
    if env::var_os("FERROTORCH_CORE021_CHILD").is_none() {
        return;
    }

    let root = parallel_failure_graph();
    let err = backward_parallel(&root, None, 4).expect_err("parallel backward must return failure");
    let FerrotorchError::InvalidArgument { message } = err else {
        panic!("expected InvalidArgument, got {err:?}");
    };
    assert!(
        message.contains("CORE-021 intentional backward failure"),
        "unexpected error message: {message}"
    );
}

#[test]
fn higher_order_grad_implicit_seed_preserves_singleton_output_shape() {
    let x = leaf(&[2.0], &[1]);
    let y = mul(&x, &x).expect("mul");

    let grads = grad(&y, &[&x], false, false).expect("grad");

    let gx = grads[0].as_ref().expect("x grad");
    assert_eq!(gx.shape(), &[1]);
    assert_eq!(gx.data().expect("gx data"), &[4.0]);
}

#[test]
fn create_graph_constant_add_gradient_is_not_fake_leaf() {
    let x = leaf(&[2.0], &[1]);
    let y = add(&x, &x).expect("add");
    let s = sum(&y).expect("sum");

    let grads = grad(&s, &[&x], true, true).expect("create_graph grad");
    let gx = grads[0].as_ref().expect("x grad");

    assert_eq!(gx.shape(), &[1]);
    assert_eq!(gx.data().expect("gx data"), &[2.0]);
    assert!(
        !gx.requires_grad(),
        "PyTorch leaves constant create_graph gradients detached; ferrotorch must not wrap them as fake leaves"
    );
    assert!(
        gx.grad_fn().is_none(),
        "constant gradients must not advertise disconnected higher-order history"
    );
}

#[test]
fn create_graph_sum_square_gradient_is_connected_to_input() {
    let x = leaf(&[2.0, 3.0], &[2]);
    let squared = mul(&x, &x).expect("mul");
    let y = sum(&squared).expect("sum");

    let grads = grad(&y, &[&x], true, true).expect("first grad");
    let gx = grads[0].as_ref().expect("x grad");

    assert_eq!(gx.shape(), &[2]);
    assert_eq!(gx.data().expect("gx data"), &[4.0, 6.0]);
    assert!(
        gx.requires_grad(),
        "sum(x * x) first derivative depends on x and must carry real history"
    );
    assert!(
        gx.grad_fn().is_some(),
        "connected first derivative must have an autograd node, not a fake leaf"
    );

    let gx_sum = sum(gx).expect("sum first grad");
    let grads2 = grad(&gx_sum, &[&x], false, false).expect("second grad");
    let g2 = grads2[0].as_ref().expect("second derivative");
    assert_eq!(g2.shape(), &[2]);
    assert_eq!(g2.data().expect("g2 data"), &[2.0, 2.0]);
}

#[test]
fn create_graph_mean_square_gradient_is_connected_to_input() {
    let x = leaf(&[2.0, 4.0], &[2]);
    let squared = mul(&x, &x).expect("mul");
    let y = mean(&squared).expect("mean");

    let grads = grad(&y, &[&x], true, true).expect("first grad");
    let gx = grads[0].as_ref().expect("x grad");

    assert_eq!(gx.shape(), &[2]);
    assert_eq!(gx.data().expect("gx data"), &[2.0, 4.0]);
    assert!(
        gx.requires_grad(),
        "mean(x * x) first derivative depends on x and must carry real history"
    );

    let gx_sum = sum(gx).expect("sum first grad");
    let grads2 = grad(&gx_sum, &[&x], false, false).expect("second grad");
    let g2 = grads2[0].as_ref().expect("second derivative");
    assert_eq!(g2.shape(), &[2]);
    assert_eq!(g2.data().expect("g2 data"), &[1.0, 1.0]);
}

#[test]
fn create_graph_broadcast_reduction_gradient_keeps_mixed_second_derivative() {
    let x = leaf(&[2.0, 3.0], &[2, 1]);
    let y = leaf(&[5.0, 7.0, 11.0, 13.0, 17.0, 19.0], &[2, 3]);
    let prod = mul(&x, &y).expect("mul broadcast");
    let loss = sum(&prod).expect("sum");

    let grads = grad(&loss, &[&x], true, true).expect("first grad wrt x");
    let gx = grads[0].as_ref().expect("x grad");

    assert_eq!(gx.shape(), &[2, 1]);
    assert_eq!(gx.data().expect("gx data"), &[23.0, 49.0]);
    assert!(
        gx.requires_grad(),
        "broadcast-reduced gradient depends on the broadcast partner"
    );

    let gx_sum = sum(gx).expect("sum gx");
    let mixed = grad(&gx_sum, &[&y], false, false).expect("mixed grad");
    let gy = mixed[0].as_ref().expect("mixed derivative wrt y");
    assert_eq!(gy.shape(), &[2, 3]);
    assert_eq!(gy.data().expect("gy data"), &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn create_graph_sum_dim_square_gradient_is_connected_to_input() {
    let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let squared = mul(&x, &x).expect("mul");
    let rows = sum_dim(&squared, 1, false).expect("sum_dim");
    let loss = sum(&rows).expect("sum rows");

    let grads = grad(&loss, &[&x], true, true).expect("first grad");
    let gx = grads[0].as_ref().expect("x grad");
    assert_eq!(gx.shape(), &[2, 3]);
    assert_eq!(
        gx.data().expect("gx data"),
        &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    );
    assert!(gx.requires_grad());

    let gx_sum = sum(gx).expect("sum first grad");
    let grads2 = grad(&gx_sum, &[&x], false, false).expect("second grad");
    let g2 = grads2[0].as_ref().expect("second derivative");
    assert_eq!(g2.shape(), &[2, 3]);
    assert_eq!(g2.data().expect("g2 data"), &[2.0, 2.0, 2.0, 2.0, 2.0, 2.0]);
}

#[test]
fn create_graph_mean_dim_square_gradient_is_connected_to_input() {
    let x = leaf(&[1.5, 3.0, 4.5, 6.0, 7.5, 9.0], &[2, 3]);
    let squared = mul(&x, &x).expect("mul");
    let rows = mean_dim(&squared, 1, false).expect("mean_dim");
    let loss = sum(&rows).expect("sum rows");

    let grads = grad(&loss, &[&x], true, true).expect("first grad");
    let gx = grads[0].as_ref().expect("x grad");
    assert_eq!(gx.shape(), &[2, 3]);
    assert_eq!(gx.data().expect("gx data"), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert!(gx.requires_grad());

    let gx_sum = sum(gx).expect("sum first grad");
    let grads2 = grad(&gx_sum, &[&x], false, false).expect("second grad");
    let g2 = grads2[0].as_ref().expect("second derivative");
    assert_eq!(g2.shape(), &[2, 3]);
    for &v in g2.data().expect("g2 data") {
        assert!((v - 2.0 / 3.0).abs() < 1e-6);
    }
}

#[test]
fn create_graph_relu_second_derivative_is_zero_not_disconnected() {
    let x = leaf(&[-1.0, 0.0, 2.0], &[3]);
    let y = sum(&relu(&x).expect("relu")).expect("sum relu");

    let grads = grad(&y, &[&x], true, true).expect("first grad");
    let gx = grads[0].as_ref().expect("x grad");
    assert_eq!(gx.shape(), &[3]);
    assert_eq!(gx.data().expect("gx data"), &[0.0, 0.0, 1.0]);
    assert!(
        gx.requires_grad(),
        "PyTorch keeps ReLU first derivatives connected so gradgrad returns zeros"
    );

    let gx_sum = sum(gx).expect("sum relu grad");
    let grads2 = grad(&gx_sum, &[&x], false, false).expect("second grad");
    let g2 = grads2[0].as_ref().expect("relu second derivative");
    assert_eq!(g2.shape(), &[3]);
    assert_eq!(g2.data().expect("g2 data"), &[0.0, 0.0, 0.0]);
}

#[cfg(feature = "gpu")]
mod cuda_create_graph {
    use super::*;
    use ferrotorch_core::Device;
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for create_graph regressions");
        });
    }

    fn cuda_leaf(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        ensure_cuda_backend();
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
            .expect("cpu tensor")
            .to(Device::Cuda(0))
            .expect("upload to cuda:0")
            .requires_grad_(true)
    }

    fn read_cuda(t: &Tensor<f32>, label: &str) -> Vec<f32> {
        assert_eq!(
            t.device(),
            Device::Cuda(0),
            "{label}: expected CUDA-resident tensor, got {:?}",
            t.device()
        );
        t.cpu()
            .expect("D2H readback")
            .data()
            .expect("cpu data")
            .to_vec()
    }

    #[test]
    fn cuda_create_graph_constant_add_gradient_is_not_fake_leaf() {
        let x = cuda_leaf(&[2.0], &[1]);
        let y = add(&x, &x).expect("add");
        let s = sum(&y).expect("sum");

        let grads = grad(&s, &[&x], true, true).expect("create_graph grad");
        let gx = grads[0].as_ref().expect("x grad");

        assert_eq!(gx.shape(), &[1]);
        assert_eq!(read_cuda(gx, "constant add grad"), &[2.0]);
        assert!(!gx.requires_grad());
        assert!(gx.grad_fn().is_none());
    }

    #[test]
    fn cuda_create_graph_sum_square_second_derivative_stays_resident() {
        let x = cuda_leaf(&[2.0, 3.0], &[2]);
        let squared = mul(&x, &x).expect("mul");
        let y = sum(&squared).expect("sum");

        let grads = grad(&y, &[&x], true, true).expect("first grad");
        let gx = grads[0].as_ref().expect("x grad");
        assert_eq!(read_cuda(gx, "sum square first grad"), &[4.0, 6.0]);
        assert!(gx.requires_grad());

        let gx_sum = sum(gx).expect("sum first grad");
        let grads2 = grad(&gx_sum, &[&x], false, false).expect("second grad");
        let g2 = grads2[0].as_ref().expect("second derivative");
        assert_eq!(read_cuda(g2, "sum square second grad"), &[2.0, 2.0]);
    }

    #[test]
    fn cuda_create_graph_broadcast_mixed_second_derivative_stays_resident() {
        let x = cuda_leaf(&[2.0, 3.0], &[2, 1]);
        let y = cuda_leaf(&[5.0, 7.0, 11.0, 13.0, 17.0, 19.0], &[2, 3]);
        let prod = mul(&x, &y).expect("mul broadcast");
        let loss = sum(&prod).expect("sum");

        let grads = grad(&loss, &[&x], true, true).expect("first grad wrt x");
        let gx = grads[0].as_ref().expect("x grad");
        assert_eq!(read_cuda(gx, "broadcast first grad"), &[23.0, 49.0]);
        assert!(gx.requires_grad());

        let gx_sum = sum(gx).expect("sum gx");
        let mixed = grad(&gx_sum, &[&y], false, false).expect("mixed grad");
        let gy = mixed[0].as_ref().expect("mixed derivative wrt y");
        assert_eq!(
            read_cuda(gy, "broadcast mixed derivative"),
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        );
    }

    #[test]
    fn cuda_create_graph_relu_second_derivative_is_zero_and_resident() {
        let x = cuda_leaf(&[-1.0, 0.0, 2.0], &[3]);
        let y = sum(&relu(&x).expect("relu")).expect("sum relu");

        let grads = grad(&y, &[&x], true, true).expect("first grad");
        let gx = grads[0].as_ref().expect("x grad");
        assert_eq!(read_cuda(gx, "relu first grad"), &[0.0, 0.0, 1.0]);
        assert!(gx.requires_grad());

        let gx_sum = sum(gx).expect("sum relu grad");
        let grads2 = grad(&gx_sum, &[&x], false, false).expect("second grad");
        let g2 = grads2[0].as_ref().expect("relu second derivative");
        assert_eq!(read_cuda(g2, "relu second derivative"), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn gradcheck_cuda_finite_difference_inputs_preserve_device() {
        let x = cuda_leaf(&[1.5, 2.5], &[2]);
        let seen_devices = Arc::new(Mutex::new(Vec::new()));
        let seen = Arc::clone(&seen_devices);
        let func = move |inputs: &[Tensor<f32>]| -> FerrotorchResult<Tensor<f32>> {
            seen.lock().expect("record device").push(inputs[0].device());
            let squared = mul(&inputs[0], &inputs[0])?;
            sum(&squared)
        };

        assert!(
            gradcheck(
                func,
                std::slice::from_ref(&x),
                Some(1.0e-2),
                Some(1.0e-2),
                Some(1.0e-2)
            )
            .expect("CUDA gradcheck")
        );
        let devices = seen_devices.lock().expect("devices");
        assert!(
            devices.len() >= 5,
            "expected analytical plus finite-difference calls, got {devices:?}"
        );
        assert!(
            devices.iter().all(|&device| device == Device::Cuda(0)),
            "gradcheck must preserve CUDA inputs for every function call, got {devices:?}"
        );
    }

    #[test]
    fn gradcheck_cuda_preserves_preexisting_leaf_grad() {
        let x = cuda_leaf(&[1.5, 2.5], &[2]);
        let prior_grad =
            Tensor::from_storage(TensorStorage::cpu(vec![13.0, -17.0]), vec![2], false)
                .expect("cpu prior grad")
                .to(Device::Cuda(0))
                .expect("upload prior grad");
        x.set_grad(Some(prior_grad)).expect("set existing grad");

        let func = |inputs: &[Tensor<f32>]| -> FerrotorchResult<Tensor<f32>> {
            let squared = mul(&inputs[0], &inputs[0])?;
            sum(&squared)
        };

        assert!(
            gradcheck(
                func,
                std::slice::from_ref(&x),
                Some(1.0e-2),
                Some(1.0e-2),
                Some(1.0e-2)
            )
            .expect("CUDA gradcheck")
        );

        let preserved_grad = x
            .grad()
            .expect("grad lookup")
            .expect("preexisting grad must remain set");
        assert_eq!(preserved_grad.shape(), &[2]);
        assert_eq!(
            read_cuda(&preserved_grad, "preserved CUDA grad"),
            &[13.0, -17.0]
        );
    }
}

#[test]
fn gradcheck_is_functional_and_repeatable() {
    let x = leaf(&[1.0, 2.0, 3.0], &[3]);
    let func = |inputs: &[Tensor<f32>]| -> FerrotorchResult<Tensor<f32>> {
        let squared = mul(&inputs[0], &inputs[0])?;
        sum(&squared)
    };

    assert!(gradcheck(func, std::slice::from_ref(&x), None, None, None).expect("gradcheck 1"));
    assert!(
        x.grad().expect("grad lookup").is_none(),
        "gradcheck must not accumulate into caller .grad state"
    );
    assert!(gradcheck(func, std::slice::from_ref(&x), None, None, None).expect("gradcheck 2"));
    assert!(
        x.grad().expect("grad lookup").is_none(),
        "repeated gradcheck must remain side-effect free"
    );
}

#[test]
fn gradcheck_preserves_preexisting_leaf_grad() {
    let x = leaf(&[1.0, 2.0, 3.0], &[3]);
    x.set_grad(Some(constant(&[13.0, -17.0, 19.0], &[3])))
        .expect("set existing grad");
    let func = |inputs: &[Tensor<f32>]| -> FerrotorchResult<Tensor<f32>> {
        let squared = mul(&inputs[0], &inputs[0])?;
        sum(&squared)
    };

    assert!(gradcheck(func, std::slice::from_ref(&x), None, None, None).expect("gradcheck"));

    let preserved_grad = x
        .grad()
        .expect("grad lookup")
        .expect("preexisting grad must remain set");
    assert_eq!(preserved_grad.shape(), &[3]);
    assert_eq!(
        preserved_grad.data().expect("preserved grad data"),
        &[13.0, -17.0, 19.0]
    );
}

#[test]
fn gradient_hook_rejects_wrong_shape_replacement() {
    let x = leaf(&[3.0], &[1]);
    let w = constant(&[2.0], &[1]);
    x.register_hook(|_g| Some(constant(&[1.0, 1.0], &[2])))
        .expect("register_hook");

    let y = sum(&mul(&x, &w).expect("mul")).expect("sum");
    let err = y.backward().expect_err("wrong-shape hook must fail");

    assert!(
        matches!(err, FerrotorchError::ShapeMismatch { .. }),
        "expected shape mismatch, got {err:?}"
    );
}

#[test]
fn post_accumulate_hook_rejects_non_leaf_tensor() {
    let x = leaf(&[3.0], &[1]);
    let w = constant(&[2.0], &[1]);
    let y = mul(&x, &w).expect("mul");

    let err = y
        .register_post_accumulate_grad_hook(|_t| {})
        .expect_err("post-accumulate hooks are leaf-only");

    assert!(
        matches!(err, FerrotorchError::InvalidArgument { .. }),
        "expected invalid argument, got {err:?}"
    );
}

#[test]
fn post_accumulate_hook_can_remove_itself_during_callback() {
    run_exact_child_before_timeout(
        "post_accumulate_hook_can_remove_itself_during_callback_child",
        "FERROTORCH_CORE025_POST_REMOVE_SELF_CHILD",
        "post-accumulate hook self-removal deadlocked the backward pass",
    );
}

#[test]
fn post_accumulate_hook_can_remove_itself_during_callback_child() {
    if env::var_os("FERROTORCH_CORE025_POST_REMOVE_SELF_CHILD").is_none() {
        return;
    }

    let x = leaf(&[3.0], &[1]);
    let w = constant(&[2.0], &[1]);
    let calls = Arc::new(AtomicUsize::new(0));
    let handle_slot: Arc<Mutex<Option<HookHandle>>> = Arc::new(Mutex::new(None));
    let calls_in_hook = Arc::clone(&calls);
    let slot_in_hook = Arc::clone(&handle_slot);

    let handle = x
        .register_post_accumulate_grad_hook(move |t| {
            calls_in_hook.fetch_add(1, Ordering::SeqCst);
            let handle = slot_in_hook
                .lock()
                .expect("slot lock")
                .expect("hook handle populated");
            assert!(t.remove_hook(handle).expect("remove hook"));
        })
        .expect("register hook");
    *handle_slot.lock().expect("slot lock") = Some(handle);

    sum(&mul(&x, &w).expect("mul 1"))
        .expect("sum 1")
        .backward()
        .expect("backward 1");
    x.zero_grad().expect("zero grad");
    sum(&mul(&x, &w).expect("mul 2"))
        .expect("sum 2")
        .backward()
        .expect("backward 2");

    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn gradient_hook_can_mutate_hook_list_during_callback() {
    run_exact_child_before_timeout(
        "gradient_hook_can_mutate_hook_list_during_callback_child",
        "FERROTORCH_CORE025_GRAD_MUTATE_CHILD",
        "gradient hook list mutation deadlocked the backward pass",
    );
}

#[test]
fn gradient_hook_can_mutate_hook_list_during_callback_child() {
    if env::var_os("FERROTORCH_CORE025_GRAD_MUTATE_CHILD").is_none() {
        return;
    }

    let x = leaf(&[3.0], &[1]);
    let w = constant(&[2.0], &[1]);
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_in_hook = Arc::clone(&calls);
    let x_in_hook = x.clone();

    x.register_hook(move |_g| {
        calls_in_hook.fetch_add(1, Ordering::SeqCst);
        let handle = x_in_hook
            .register_hook(|_g| None)
            .expect("reentrant register_hook");
        assert!(
            x_in_hook
                .remove_hook(handle)
                .expect("reentrant remove_hook")
        );
        None
    })
    .expect("register_hook");

    sum(&mul(&x, &w).expect("mul"))
        .expect("sum")
        .backward()
        .expect("backward");

    let grad = x.grad().expect("grad lookup").expect("x grad");
    assert_eq!(grad.data().expect("grad data"), &[2.0]);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn post_accumulate_hook_can_mutate_hook_list_during_callback() {
    run_exact_child_before_timeout(
        "post_accumulate_hook_can_mutate_hook_list_during_callback_child",
        "FERROTORCH_CORE025_POST_MUTATE_CHILD",
        "post-accumulate hook list mutation deadlocked the backward pass",
    );
}

#[test]
fn post_accumulate_hook_can_mutate_hook_list_during_callback_child() {
    if env::var_os("FERROTORCH_CORE025_POST_MUTATE_CHILD").is_none() {
        return;
    }

    let x = leaf(&[3.0], &[1]);
    let w = constant(&[2.0], &[1]);
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_in_hook = Arc::clone(&calls);

    x.register_post_accumulate_grad_hook(move |t| {
        calls_in_hook.fetch_add(1, Ordering::SeqCst);
        let handle = t
            .register_post_accumulate_grad_hook(|_t| {})
            .expect("reentrant register_post_accumulate_grad_hook");
        assert!(t.remove_hook(handle).expect("reentrant remove_hook"));
    })
    .expect("register_post_accumulate_grad_hook");

    sum(&mul(&x, &w).expect("mul"))
        .expect("sum")
        .backward()
        .expect("backward");

    let grad = x.grad().expect("grad lookup").expect("x grad");
    assert_eq!(grad.data().expect("grad data"), &[2.0]);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn hook_list_mutations_affect_next_backward_only() {
    run_exact_child_before_timeout(
        "hook_list_mutations_affect_next_backward_only_child",
        "FERROTORCH_CORE025_SNAPSHOT_CHILD",
        "hook-list snapshot mutation semantics deadlocked the backward pass",
    );
}

#[test]
fn hook_list_mutations_affect_next_backward_only_child() {
    if env::var_os("FERROTORCH_CORE025_SNAPSHOT_CHILD").is_none() {
        return;
    }

    let x = leaf(&[3.0], &[1]);
    let w = constant(&[2.0], &[1]);
    let calls = Arc::new(Mutex::new(Vec::new()));
    let second_handle: Arc<Mutex<Option<HookHandle>>> = Arc::new(Mutex::new(None));
    let x_for_first_hook = x.clone();
    let calls_for_first_hook = Arc::clone(&calls);
    let second_handle_for_first_hook = Arc::clone(&second_handle);

    x.register_hook(move |_g| {
        calls_for_first_hook.lock().expect("calls lock").push("g1");
        if let Some(handle) = second_handle_for_first_hook
            .lock()
            .expect("handle lock")
            .take()
        {
            assert!(x_for_first_hook.remove_hook(handle).expect("remove g2"));
        }
        None
    })
    .expect("register g1");
    let calls_for_second_hook = Arc::clone(&calls);
    let handle = x
        .register_hook(move |_g| {
            calls_for_second_hook.lock().expect("calls lock").push("g2");
            None
        })
        .expect("register g2");
    *second_handle.lock().expect("handle lock") = Some(handle);

    sum(&mul(&x, &w).expect("mul 1"))
        .expect("sum 1")
        .backward()
        .expect("backward 1");
    x.zero_grad().expect("zero grad 1");
    sum(&mul(&x, &w).expect("mul 2"))
        .expect("sum 2")
        .backward()
        .expect("backward 2");
    assert_eq!(&*calls.lock().expect("calls lock"), &["g1", "g2", "g1"]);

    let y = leaf(&[4.0], &[1]);
    let w = constant(&[5.0], &[1]);
    let calls = Arc::new(Mutex::new(Vec::new()));
    let registered = Arc::new(Mutex::new(false));
    let y_for_first_hook = y.clone();
    let calls_for_first_hook = Arc::clone(&calls);
    let registered_for_first_hook = Arc::clone(&registered);

    y.register_hook(move |_g| {
        calls_for_first_hook.lock().expect("calls lock").push("ga");
        let mut registered = registered_for_first_hook.lock().expect("registered lock");
        if !*registered {
            let calls_for_new_hook = Arc::clone(&calls_for_first_hook);
            y_for_first_hook
                .register_hook(move |_g| {
                    calls_for_new_hook.lock().expect("calls lock").push("gb");
                    None
                })
                .expect("register gb");
            *registered = true;
        }
        None
    })
    .expect("register ga");

    sum(&mul(&y, &w).expect("mul 3"))
        .expect("sum 3")
        .backward()
        .expect("backward 3");
    y.zero_grad().expect("zero grad 2");
    sum(&mul(&y, &w).expect("mul 4"))
        .expect("sum 4")
        .backward()
        .expect("backward 4");
    assert_eq!(&*calls.lock().expect("calls lock"), &["ga", "ga", "gb"]);

    let z = leaf(&[3.0], &[1]);
    let w = constant(&[2.0], &[1]);
    let calls = Arc::new(Mutex::new(Vec::new()));
    let second_handle: Arc<Mutex<Option<HookHandle>>> = Arc::new(Mutex::new(None));
    let calls_for_first_hook = Arc::clone(&calls);
    let second_handle_for_first_hook = Arc::clone(&second_handle);

    z.register_post_accumulate_grad_hook(move |t| {
        calls_for_first_hook.lock().expect("calls lock").push("p1");
        if let Some(handle) = second_handle_for_first_hook
            .lock()
            .expect("handle lock")
            .take()
        {
            assert!(t.remove_hook(handle).expect("remove p2"));
        }
    })
    .expect("register p1");
    let calls_for_second_hook = Arc::clone(&calls);
    let handle = z
        .register_post_accumulate_grad_hook(move |_t| {
            calls_for_second_hook.lock().expect("calls lock").push("p2");
        })
        .expect("register p2");
    *second_handle.lock().expect("handle lock") = Some(handle);

    sum(&mul(&z, &w).expect("mul 5"))
        .expect("sum 5")
        .backward()
        .expect("backward 5");
    z.zero_grad().expect("zero grad 3");
    sum(&mul(&z, &w).expect("mul 6"))
        .expect("sum 6")
        .backward()
        .expect("backward 6");
    assert_eq!(&*calls.lock().expect("calls lock"), &["p1", "p2", "p1"]);

    let q = leaf(&[4.0], &[1]);
    let w = constant(&[5.0], &[1]);
    let calls = Arc::new(Mutex::new(Vec::new()));
    let registered = Arc::new(Mutex::new(false));
    let calls_for_first_hook = Arc::clone(&calls);
    let registered_for_first_hook = Arc::clone(&registered);

    q.register_post_accumulate_grad_hook(move |t| {
        calls_for_first_hook.lock().expect("calls lock").push("pa");
        let mut registered = registered_for_first_hook.lock().expect("registered lock");
        if !*registered {
            let calls_for_new_hook = Arc::clone(&calls_for_first_hook);
            t.register_post_accumulate_grad_hook(move |_t| {
                calls_for_new_hook.lock().expect("calls lock").push("pb");
            })
            .expect("register pb");
            *registered = true;
        }
    })
    .expect("register pa");

    sum(&mul(&q, &w).expect("mul 7"))
        .expect("sum 7")
        .backward()
        .expect("backward 7");
    q.zero_grad().expect("zero grad 4");
    sum(&mul(&q, &w).expect("mul 8"))
        .expect("sum 8")
        .backward()
        .expect("backward 8");
    assert_eq!(&*calls.lock().expect("calls lock"), &["pa", "pa", "pb"]);
}
