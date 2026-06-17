use ferrotorch_core::autograd::no_grad::{inference_mode, no_grad};
use ferrotorch_core::grad_fns::arithmetic::{add, mul};
use ferrotorch_core::grad_fns::reduction::sum;
use ferrotorch_core::{FerrotorchError, Tensor, TensorStorage, permute_t, view_t};

fn tensor(data: &[f32], requires_grad: bool) -> Tensor<f32> {
    Tensor::from_storage(
        TensorStorage::cpu(data.to_vec()),
        vec![data.len()],
        requires_grad,
    )
    .unwrap()
}

fn assert_inference_requires_grad_error(err: FerrotorchError) {
    let msg = err.to_string();
    assert!(
        msg.contains("Setting requires_grad=True on inference tensor outside InferenceMode"),
        "unexpected requires_grad error: {msg}"
    );
}

fn assert_saved_inference_error(err: FerrotorchError) {
    let msg = err.to_string();
    assert!(
        msg.contains("Inference tensors cannot be saved for backward")
            && msg.contains("created in inference mode"),
        "unexpected saved-tensor error: {msg}"
    );
}

#[test]
fn inference_tensor_flag_persists_and_blocks_late_requires_grad_enable() {
    let t = inference_mode(|| tensor(&[1.0, 2.0], false));

    assert!(t.is_inference());
    assert!(!t.requires_grad());
    assert!(t.is_leaf());
    assert!(t.grad_fn().is_none());

    let err = t
        .clone()
        .try_requires_grad_(true)
        .expect_err("PyTorch rejects setting requires_grad=True outside inference mode");
    assert_inference_requires_grad_error(err);

    let still_inference = t.try_requires_grad_(false).unwrap();
    assert!(still_inference.is_inference());
    assert!(!still_inference.requires_grad());
}

#[test]
fn inference_mode_allows_requires_grad_enable_inside_scope_only() {
    let t = inference_mode(|| tensor(&[1.0, 2.0], false).try_requires_grad_(true).unwrap());

    assert!(t.is_inference());
    assert!(t.requires_grad());

    let err = t
        .clone()
        .try_requires_grad_(true)
        .expect_err("already-true inference tensor still cannot be re-enabled outside scope");
    assert_inference_requires_grad_error(err);
}

#[test]
fn no_grad_tensor_is_not_inference_and_can_require_grad_later() {
    let t = no_grad(|| tensor(&[1.0, 2.0], false));

    assert!(!t.is_inference());
    let t = t.try_requires_grad_(true).unwrap();
    assert!(t.requires_grad());
    assert!(!t.is_inference());
}

#[test]
fn inference_aliases_preserve_inference_flag_without_autograd_edge() {
    let base = inference_mode(|| tensor(&[1.0, 2.0, 3.0, 4.0], true));
    assert!(base.is_inference());
    assert!(base.requires_grad());

    let view = view_t(&base, &[2, 2]).unwrap();
    assert!(view.is_inference());
    assert!(!view.requires_grad());
    assert!(view.is_leaf());
    assert!(view.grad_fn().is_none());
    assert_eq!(view.data().unwrap(), &[1.0, 2.0, 3.0, 4.0]);

    let strided = base.try_stride_view(vec![2], vec![1], 1).unwrap();
    assert!(strided.is_inference());
    assert!(!strided.requires_grad());
    assert!(strided.is_leaf());
    assert_eq!(strided.data_vec().unwrap(), vec![2.0, 3.0]);

    let detached = base.detach();
    assert!(detached.is_inference());
    assert!(!detached.requires_grad());
    assert!(detached.is_leaf());
}

#[test]
fn noncontiguous_inference_reshape_copy_outside_scope_is_normal_tensor() {
    let base = inference_mode(|| {
        Tensor::from_storage(
            TensorStorage::cpu(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            vec![2, 3],
            true,
        )
        .unwrap()
    });
    let transposed = permute_t(&base, &[1, 0]).unwrap();
    assert!(transposed.is_inference());
    assert!(!transposed.is_contiguous());

    let reshaped = transposed.reshape_t(&[6]).unwrap();

    assert!(!reshaped.is_inference());
    assert!(!reshaped.requires_grad());
    assert!(reshaped.is_leaf());
    assert!(reshaped.grad_fn().is_none());
    assert_eq!(reshaped.data().unwrap(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn operation_result_created_inside_inference_mode_is_inference_tensor() {
    let tracked = tensor(&[1.0, 2.0], true);
    let constant = tensor(&[10.0, 20.0], false);

    let out = inference_mode(|| add(&tracked, &constant).unwrap());

    assert!(out.is_inference());
    assert!(!out.requires_grad());
    assert!(out.is_leaf());
    assert!(out.grad_fn().is_none());
    assert_eq!(out.data().unwrap(), &[11.0, 22.0]);
}

#[test]
fn inference_constants_can_route_graph_edges_but_cannot_be_saved_for_backward() {
    let inference_constant = inference_mode(|| tensor(&[10.0, 20.0], false));

    let add_input = tensor(&[1.0, 2.0], true);
    let added = add(&add_input, &inference_constant)
        .expect("add backward does not save input values, matching PyTorch permissiveness");
    assert!(!added.is_inference());
    sum(&added).unwrap().backward().unwrap();
    let add_grad = add_input.grad().unwrap().unwrap();
    assert_eq!(add_grad.data().unwrap(), &[1.0, 1.0]);

    let mul_input = tensor(&[1.0, 2.0], true);
    let err = mul(&mul_input, &inference_constant)
        .expect_err("mul backward needs saved operand data and must reject inference tensors");
    assert_saved_inference_error(err);
}
