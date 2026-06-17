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
    let exe = env::current_exe().expect("current test binary");
    let mut child = Command::new(exe)
        .arg("--exact")
        .arg("backward_parallel_error_path_child")
        .arg("--nocapture")
        .env("FERROTORCH_CORE021_CHILD", "1")
        .spawn()
        .expect("spawn CORE-021 child test");

    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(status) = child.try_wait().expect("poll child") {
            assert!(status.success(), "CORE-021 child exited with {status}");
            return;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("backward_parallel did not return after a worker failure");
        }
        thread::sleep(Duration::from_millis(10));
    }
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
