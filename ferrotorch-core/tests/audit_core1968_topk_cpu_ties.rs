//! CORE-1968 / #1968: CPU `topk` tie ordering must follow PyTorch's concrete
//! CPU selection order, not a stable ascending-index sort.
//!
//! Live torch 2.11.0+cu130 CPU oracles:
//!   torch.topk(torch.tensor([1.,1.,1.,1.]), 1, largest=False).indices -> [2]
//!   torch.topk(torch.tensor([1.,1.,1.,1.]), 2, largest=False).indices -> [2, 3]
//!   torch.topk(torch.ones(100), 2, largest=False).indices -> [67, 66]
//!   torch.topk(torch.tensor([[1.,1.,1.,1.],[4.,3.,3.,4.]]), 2,
//!              largest=True).indices -> [[2, 3], [3, 0]]

use ferrotorch_core::{Tensor, TensorStorage, topk};

fn t(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

#[test]
fn topk_cpu_equal_ties_follow_torch_nth_element_order() {
    let x = t(&[1.0, 1.0, 1.0, 1.0], &[4]);

    let (v1, i1) = topk(&x, 1, false).unwrap();
    assert_eq!(v1.data().unwrap(), &[1.0]);
    assert_eq!(i1, vec![2]);

    let (v2, i2) = topk(&x, 2, false).unwrap();
    assert_eq!(v2.data().unwrap(), &[1.0, 1.0]);
    assert_eq!(i2, vec![2, 3]);
}

#[test]
fn topk_cpu_large_equal_ties_follow_torch_nth_element_order() {
    let x = t(&vec![1.0; 100], &[100]);

    let (values, indices) = topk(&x, 2, false).unwrap();

    assert_eq!(values.data().unwrap(), &[1.0, 1.0]);
    assert_eq!(indices, vec![67, 66]);
}

#[test]
fn topk_cpu_2d_ties_are_resolved_per_row_like_torch() {
    let x = t(&[1.0, 1.0, 1.0, 1.0, 4.0, 3.0, 3.0, 4.0], &[2, 4]);

    let (values, indices) = topk(&x, 2, true).unwrap();

    assert_eq!(values.shape(), &[2, 2]);
    assert_eq!(values.data().unwrap(), &[1.0, 1.0, 4.0, 4.0]);
    assert_eq!(indices, vec![2, 3, 3, 0]);
}
