use ferrotorch_core::grad_fns::{activation, arithmetic, reduction};
use ferrotorch_core::{Tensor, creation, no_grad};

fn meta_leaf(shape: &[usize]) -> Tensor<f32> {
    creation::zeros_meta(shape).unwrap().requires_grad_(true)
}

fn assert_meta_nonleaf(t: &Tensor<f32>, shape: &[usize], grad_fn: &str) {
    assert!(t.is_meta(), "{grad_fn} output should stay on meta");
    assert_eq!(t.shape(), shape);
    assert!(t.requires_grad(), "{grad_fn} output should require grad");
    assert!(!t.is_leaf(), "{grad_fn} output should be a non-leaf");
    assert_eq!(t.grad_fn().as_ref().map(|f| f.name()), Some(grad_fn));
}

fn assert_meta_grad(t: &Tensor<f32>, shape: &[usize]) {
    let grad = t.grad().unwrap().expect("leaf should receive grad");
    assert!(grad.is_meta(), "leaf grad should stay on meta");
    assert_eq!(grad.shape(), shape);
    assert!(!grad.requires_grad(), "first-order leaf grad is detached");
}

#[test]
fn meta_relu_sum_backward_keeps_graph_and_meta_grad() {
    let x = meta_leaf(&[2, 3]);
    let y = activation::relu(&x).unwrap();
    assert_meta_nonleaf(&y, &[2, 3], "ReluBackward");

    let loss = reduction::sum(&y).unwrap();
    assert_meta_nonleaf(&loss, &[], "SumBackward");
    loss.backward().unwrap();

    assert_meta_grad(&x, &[2, 3]);
}

#[test]
fn meta_add_broadcast_backward_reduces_to_original_input_shapes() {
    let a = meta_leaf(&[2, 3]);
    let b = meta_leaf(&[1, 3]);

    let y = arithmetic::add(&a, &b).unwrap();
    assert_meta_nonleaf(&y, &[2, 3], "AddBackward");

    reduction::sum(&y).unwrap().backward().unwrap();

    assert_meta_grad(&a, &[2, 3]);
    assert_meta_grad(&b, &[1, 3]);
}

#[test]
fn meta_add_scaled_non_identity_backward_stays_shape_only() {
    let a = meta_leaf(&[2, 1]);
    let b = meta_leaf(&[1, 3]);

    let y = arithmetic::add_scaled(&a, &b, 2.5).unwrap();
    assert_meta_nonleaf(&y, &[2, 3], "AddScaledBackward");

    reduction::sum(&y).unwrap().backward().unwrap();

    assert_meta_grad(&a, &[2, 1]);
    assert_meta_grad(&b, &[1, 3]);
}

#[test]
fn meta_saved_output_unary_backward_does_not_read_data() {
    let x = meta_leaf(&[4]);

    let y = arithmetic::sqrt(&x).unwrap();
    assert_meta_nonleaf(&y, &[4], "SqrtBackward");

    reduction::sum(&y).unwrap().backward().unwrap();

    assert_meta_grad(&x, &[4]);
}

#[test]
fn meta_sum_dim_backward_reexpands_to_input_shape() {
    let x = meta_leaf(&[2, 3, 4]);
    let y = reduction::sum_dim(&x, -1, false).unwrap();
    assert_meta_nonleaf(&y, &[2, 3], "SumDimBackward");

    reduction::sum(&y).unwrap().backward().unwrap();

    assert_meta_grad(&x, &[2, 3, 4]);
}

#[test]
fn meta_value_dependent_reductions_attach_backward_without_data_reads() {
    let x = meta_leaf(&[2, 3]);

    let logsumexp = reduction::logsumexp(&x).unwrap();
    assert_meta_nonleaf(&logsumexp, &[], "LogsumexpBackward");
    logsumexp.backward().unwrap();
    assert_meta_grad(&x, &[2, 3]);

    let v = meta_leaf(&[2, 3]);
    let var = reduction::var(&v, false).unwrap();
    assert_meta_nonleaf(&var, &[], "VarBackward");
    var.backward().unwrap();
    assert_meta_grad(&v, &[2, 3]);
}

#[test]
fn meta_no_grad_result_stays_detached_leaf() {
    let x = meta_leaf(&[2, 3]);

    let y = no_grad(|| activation::relu(&x)).unwrap();

    assert!(y.is_meta());
    assert_eq!(y.shape(), &[2, 3]);
    assert!(!y.requires_grad());
    assert!(y.is_leaf());
    assert!(y.grad_fn().is_none());
}
