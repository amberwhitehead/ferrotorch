use ferrotorch_core::grad_fns::arithmetic::mul;
use ferrotorch_core::grad_fns::indexing::masked_fill;
use ferrotorch_core::grad_fns::reduction::sum;
use ferrotorch_core::{Tensor, TensorStorage};

fn tensor(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
    Tensor::from_storage(
        TensorStorage::cpu(data.to_vec()),
        shape.to_vec(),
        requires_grad,
    )
    .expect("tensor")
}

#[test]
fn masked_fill_host_mask_accepts_noncontiguous_cpu_view_and_backpropagates() {
    let base = tensor(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3], true);
    let view = base.transpose(0, 1).expect("transpose");
    assert_eq!(view.shape(), &[3, 2]);
    assert!(
        !view.is_contiguous(),
        "test must exercise logical-order view handling"
    );

    let mask = [true, false, false, true, true, false];
    let filled = masked_fill(&view, &mask, -1.0).expect("masked_fill");

    assert_eq!(filled.shape(), &[3, 2]);
    assert_eq!(
        filled.data_vec().expect("filled data"),
        vec![-1.0, 3.0, 1.0, -1.0, -1.0, 5.0]
    );

    let weights = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], false);
    let weighted = mul(&filled, &weights).expect("mul");
    sum(&weighted).expect("sum").backward().expect("backward");

    let grad = base.grad().expect("grad lookup").expect("base grad");
    assert_eq!(grad.shape(), &[2, 3]);
    assert_eq!(
        grad.data_vec().expect("grad data"),
        vec![0.0, 3.0, 0.0, 2.0, 0.0, 6.0]
    );
}
