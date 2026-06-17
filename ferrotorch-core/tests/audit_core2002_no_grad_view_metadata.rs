use ferrotorch_core::autograd::no_grad::{inference_mode, no_grad};
use ferrotorch_core::grad_fns::arithmetic::mul;
use ferrotorch_core::grad_fns::reduction::sum;
use ferrotorch_core::{FerrotorchError, Tensor, TensorStorage, view_t};

fn tensor(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
    Tensor::from_storage(
        TensorStorage::cpu(data.to_vec()),
        shape.to_vec(),
        requires_grad,
    )
    .unwrap()
}

fn assert_slice_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (idx, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= 1.0e-6,
            "mismatch at {idx}: got {a}, expected {e}"
        );
    }
}

fn assert_no_grad_view_error(err: FerrotorchError) {
    let msg = err.to_string();
    assert!(
        msg.contains("view was created in no_grad mode")
            && msg.contains("modified inplace with grad mode enabled"),
        "unexpected no_grad view error: {msg}"
    );
}

fn assert_inference_view_error(err: FerrotorchError) {
    let msg = err.to_string();
    assert!(
        msg.contains("view was created in inference mode")
            && msg.contains("modified inplace in normal mode"),
        "unexpected inference view error: {msg}"
    );
}

#[test]
fn no_grad_view_of_grad_tensor_requires_grad_but_has_no_backward_edge() {
    let base = tensor(&[1.0, 2.0, 3.0, 4.0], &[4], true);
    let view = no_grad(|| view_t(&base, &[2, 2]).unwrap());

    assert!(view.requires_grad());
    assert!(view.is_leaf());
    assert!(view.grad_fn().is_none());
    assert!(!view.is_inference());

    let scale = tensor(&[2.0, 2.0, 2.0, 2.0], &[2, 2], false);
    let product = mul(&view, &scale).unwrap();
    assert!(product.requires_grad());
    assert!(product.grad_fn().is_some());

    sum(&product).unwrap().backward().unwrap();
    assert!(base.grad().unwrap().is_none());
    assert!(view.grad().unwrap().is_none());
}

#[test]
fn no_grad_view_is_constant_for_base_but_other_operands_still_receive_grad() {
    let base = tensor(&[0.0, 1.0, 2.0, 3.0], &[4], true);
    let view = no_grad(|| view_t(&base, &[4]).unwrap());
    let other = tensor(&[10.0, 20.0, 30.0, 40.0], &[4], true);

    let product = mul(&view, &other).unwrap();
    assert!(product.requires_grad());
    assert!(product.grad_fn().is_some());

    sum(&product).unwrap().backward().unwrap();
    assert!(base.grad().unwrap().is_none());
    assert!(view.grad().unwrap().is_none());
    assert_slice_close(
        other.grad().unwrap().unwrap().data().unwrap(),
        &[0.0, 1.0, 2.0, 3.0],
    );
}

#[test]
fn nested_view_from_no_grad_alias_keeps_creation_guard_and_stops_at_alias() {
    let base = tensor(&[0.0, 1.0, 2.0, 3.0], &[4], true);
    let view = no_grad(|| view_t(&base, &[2, 2]).unwrap());
    let nested = view_t(&view, &[4]).unwrap();

    assert!(nested.requires_grad());
    assert!(!nested.is_leaf());
    assert!(nested.grad_fn().is_some());

    let other = tensor(&[10.0, 20.0, 30.0, 40.0], &[4], true);
    sum(&mul(&nested, &other).unwrap())
        .unwrap()
        .backward()
        .unwrap();
    assert!(base.grad().unwrap().is_none());
    assert!(view.grad().unwrap().is_none());
    assert_slice_close(
        other.grad().unwrap().unwrap().data().unwrap(),
        &[0.0, 1.0, 2.0, 3.0],
    );

    let err = nested
        .fill_(0.0)
        .expect_err("PyTorch rejects later views of a no_grad-created view");
    assert_no_grad_view_error(err);
}

#[test]
fn no_grad_view_requires_grad_setter_preserves_public_flag_separately_from_edge() {
    let base = tensor(&[1.0, 2.0, 3.0, 4.0], &[4], true);
    let view = no_grad(|| view_t(&base, &[4]).unwrap());

    let disabled_alias = view.clone().try_requires_grad_(false).unwrap();
    assert!(disabled_alias.requires_grad());
    assert!(disabled_alias.is_leaf());
    assert!(disabled_alias.grad_fn().is_none());

    sum(&mul(&disabled_alias, &tensor(&[2.0, 2.0, 2.0, 2.0], &[4], false)).unwrap())
        .unwrap()
        .backward()
        .unwrap();
    assert!(base.grad().unwrap().is_none());
    assert!(disabled_alias.grad().unwrap().is_none());

    let enabled_alias = disabled_alias.try_requires_grad_(true).unwrap();
    sum(&mul(&enabled_alias, &tensor(&[3.0, 3.0, 3.0, 3.0], &[4], false)).unwrap())
        .unwrap()
        .backward()
        .unwrap();
    assert!(base.grad().unwrap().is_none());
    assert_slice_close(
        enabled_alias.grad().unwrap().unwrap().data().unwrap(),
        &[3.0, 3.0, 3.0, 3.0],
    );

    let err = enabled_alias
        .fill_(0.0)
        .expect_err("requires_grad_ does not clear no_grad view creation metadata");
    assert_no_grad_view_error(err);
}

#[test]
fn no_grad_view_of_plain_base_requires_grad_setter_can_clear_local_edge() {
    let base = tensor(&[1.0, 2.0, 3.0, 4.0], &[4], false);
    let view = no_grad(|| view_t(&base, &[4]).unwrap());
    assert!(!view.requires_grad());

    view.fill_(5.0)
        .expect("PyTorch allows mutation while the alias does not require grad");
    assert_eq!(base.data().unwrap(), &[5.0, 5.0, 5.0, 5.0]);

    let enabled_alias = view.try_requires_grad_(true).unwrap();
    assert!(enabled_alias.requires_grad());
    let err = enabled_alias
        .fill_(0.0)
        .expect_err("PyTorch applies the no_grad view guard once the alias requires grad");
    assert_no_grad_view_error(err);

    let disabled_alias = enabled_alias.try_requires_grad_(false).unwrap();
    assert!(!disabled_alias.requires_grad());
    disabled_alias
        .fill_(6.0)
        .expect("clearing the local grad edge removes the write guard when the base is plain");
    assert_eq!(base.data().unwrap(), &[6.0, 6.0, 6.0, 6.0]);

    let product = mul(&disabled_alias, &tensor(&[2.0, 2.0, 2.0, 2.0], &[4], false)).unwrap();
    assert!(!product.requires_grad());
    assert!(product.grad_fn().is_none());
}

#[test]
fn backward_on_no_grad_view_leaf_errors_like_non_differentiable_root() {
    let base = tensor(&[3.0], &[1], true);
    let view = no_grad(|| view_t(&base, &[]).unwrap());

    assert!(view.requires_grad());
    assert!(view.is_leaf());
    assert!(view.grad_fn().is_none());

    let err = view
        .backward()
        .expect_err("PyTorch reports no grad edge for a no_grad-created view leaf");
    let msg = err.to_string();
    assert!(
        msg.contains("does not require grad") && msg.contains("does not have a grad_fn"),
        "unexpected backward error: {msg}"
    );
}

#[test]
fn no_grad_view_inplace_errors_in_grad_mode_but_is_allowed_under_no_grad() {
    let base = tensor(&[1.0, 2.0, 3.0, 4.0], &[4], true);
    let view = no_grad(|| view_t(&base, &[2, 2]).unwrap());

    let err = view
        .fill_(0.0)
        .expect_err("PyTorch rejects no_grad-created view mutation in grad mode");
    assert_no_grad_view_error(err);

    no_grad(|| view.fill_(9.0)).unwrap();
    assert_eq!(base.data().unwrap(), &[9.0, 9.0, 9.0, 9.0]);
}

#[test]
fn inference_mode_view_of_normal_tensor_is_not_inference_but_keeps_creation_guard() {
    let base = tensor(&[1.0, 2.0, 3.0, 4.0], &[4], true);
    let view = inference_mode(|| view_t(&base, &[2, 2]).unwrap());

    assert!(view.requires_grad());
    assert!(view.is_leaf());
    assert!(!view.is_inference());
    assert!(view.grad_fn().is_none());

    let err = view
        .fill_(0.0)
        .expect_err("PyTorch rejects inference-created view mutation in normal mode");
    assert_inference_view_error(err);

    inference_mode(|| view.fill_(7.0)).unwrap();
    assert_eq!(base.data().unwrap(), &[7.0, 7.0, 7.0, 7.0]);
}
