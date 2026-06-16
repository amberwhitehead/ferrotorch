use ferrotorch_core::autograd::checkpoint::{checkpoint, checkpoint_multi};
use ferrotorch_core::autograd::no_grad::{is_grad_enabled, no_grad};
use ferrotorch_core::grad_fns::arithmetic::{add, mul};
use ferrotorch_core::grad_fns::reduction::sum;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::{FerrotorchResult, Tensor};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

fn leaf_grad(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true)
        .expect("construct leaf")
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch actual={actual:?} expected={expected:?}"
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "{label}[{i}]: expected {e}, got {a}; actual={actual:?}"
        );
    }
}

#[test]
fn checkpoint_backward_inside_no_grad_recomputes_with_grad_enabled() {
    let calls = Arc::new(AtomicUsize::new(0));
    let forward_grad_enabled = Arc::new(AtomicBool::new(true));
    let recompute_grad_enabled = Arc::new(AtomicBool::new(false));

    let calls_for_fn = Arc::clone(&calls);
    let forward_for_fn = Arc::clone(&forward_grad_enabled);
    let recompute_for_fn = Arc::clone(&recompute_grad_enabled);

    let x = leaf_grad(&[1.0, 2.0, 3.0], &[3]);
    let y = checkpoint(
        move |t: &Tensor<f32>| -> FerrotorchResult<Tensor<f32>> {
            match calls_for_fn.fetch_add(1, Ordering::SeqCst) {
                0 => forward_for_fn.store(is_grad_enabled(), Ordering::SeqCst),
                1 => recompute_for_fn.store(is_grad_enabled(), Ordering::SeqCst),
                call => panic!("checkpoint function called more than twice: {call}"),
            }
            let sq = mul(t, t)?;
            add(&sq, t)
        },
        &x,
    )
    .expect("checkpoint");

    let loss = sum(&y).expect("loss");
    no_grad(|| {
        assert!(
            !is_grad_enabled(),
            "test must enter no_grad before backward"
        );
        loss.backward().expect("checkpoint backward inside no_grad");
        assert!(
            !is_grad_enabled(),
            "checkpoint backward must restore the caller's no_grad state"
        );
    });
    assert!(is_grad_enabled(), "outer grad mode should be restored");

    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "checkpoint should run forward once and recompute once"
    );
    assert!(
        !forward_grad_enabled.load(Ordering::SeqCst),
        "reentrant checkpoint forward should run under no_grad"
    );
    assert!(
        recompute_grad_enabled.load(Ordering::SeqCst),
        "checkpoint recomputation must force grad mode on"
    );

    let grad = x.grad().expect("grad lookup").expect("x grad");
    assert_close(
        grad.data().expect("grad data"),
        &[3.0, 5.0, 7.0],
        1e-5,
        "single checkpoint grad",
    );
}

#[test]
fn checkpoint_multi_backward_inside_no_grad_recomputes_with_grad_enabled() {
    let calls = Arc::new(AtomicUsize::new(0));
    let forward_grad_enabled = Arc::new(AtomicBool::new(true));
    let recompute_grad_enabled = Arc::new(AtomicBool::new(false));

    let calls_for_fn = Arc::clone(&calls);
    let forward_for_fn = Arc::clone(&forward_grad_enabled);
    let recompute_for_fn = Arc::clone(&recompute_grad_enabled);

    let a = leaf_grad(&[1.0, 2.0, 3.0], &[3]);
    let b = leaf_grad(&[4.0, 5.0, 6.0], &[3]);
    let y = checkpoint_multi(
        move |ts: &[Tensor<f32>]| -> FerrotorchResult<Tensor<f32>> {
            match calls_for_fn.fetch_add(1, Ordering::SeqCst) {
                0 => forward_for_fn.store(is_grad_enabled(), Ordering::SeqCst),
                1 => recompute_for_fn.store(is_grad_enabled(), Ordering::SeqCst),
                call => panic!("checkpoint_multi function called more than twice: {call}"),
            }
            let prod = mul(&ts[0], &ts[1])?;
            add(&prod, &ts[0])
        },
        &[a.clone(), b.clone()],
    )
    .expect("checkpoint_multi");

    let loss = sum(&y).expect("loss");
    no_grad(|| {
        assert!(
            !is_grad_enabled(),
            "test must enter no_grad before backward"
        );
        loss.backward()
            .expect("checkpoint_multi backward inside no_grad");
        assert!(
            !is_grad_enabled(),
            "checkpoint_multi backward must restore the caller's no_grad state"
        );
    });
    assert!(is_grad_enabled(), "outer grad mode should be restored");

    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "checkpoint_multi should run forward once and recompute once"
    );
    assert!(
        !forward_grad_enabled.load(Ordering::SeqCst),
        "reentrant checkpoint_multi forward should run under no_grad"
    );
    assert!(
        recompute_grad_enabled.load(Ordering::SeqCst),
        "checkpoint_multi recomputation must force grad mode on"
    );

    let grad_a = a.grad().expect("a grad lookup").expect("a grad");
    assert_close(
        grad_a.data().expect("grad-a data"),
        &[5.0, 6.0, 7.0],
        1e-5,
        "checkpoint_multi grad-a",
    );
    let grad_b = b.grad().expect("b grad lookup").expect("b grad");
    assert_close(
        grad_b.data().expect("grad-b data"),
        &[1.0, 2.0, 3.0],
        1e-5,
        "checkpoint_multi grad-b",
    );
}
