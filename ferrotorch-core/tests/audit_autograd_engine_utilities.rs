use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use ferrotorch_core::autograd::gradcheck::gradcheck;
use ferrotorch_core::autograd::graph::backward_parallel;
use ferrotorch_core::autograd::higher_order::grad;
use ferrotorch_core::autograd::hooks::HookHandle;
use ferrotorch_core::grad_fns::arithmetic::{add, mul};
use ferrotorch_core::grad_fns::reduction::sum;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Tensor, TensorStorage};

fn leaf(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}

fn constant(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
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
fn higher_order_grad_implicit_seed_preserves_singleton_output_shape() {
    let x = leaf(&[2.0], &[1]);
    let y = mul(&x, &x).expect("mul");

    let grads = grad(&y, &[&x], false, false).expect("grad");

    let gx = grads[0].as_ref().expect("x grad");
    assert_eq!(gx.shape(), &[1]);
    assert_eq!(gx.data().expect("gx data"), &[4.0]);
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
