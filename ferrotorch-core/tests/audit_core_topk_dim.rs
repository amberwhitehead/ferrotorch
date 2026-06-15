//! PyTorch parity audit for dim-aware `topk`.
//!
//! Live torch 2.11.0+cu130 probes used here:
//! - `torch.topk(x.reshape(2,3,4), 2, dim=1)` values/indices below.
//! - `torch.topk(torch.tensor(7.), k=0|1, dim=0)` returns scalar value 7,
//!   scalar index 0; `k=2` errors.
//! - Weighted backward for `dim=1` scatters cotangents to the selected input
//!   positions only.

use ferrotorch_core::{Tensor, TensorStorage, topk_dim, topk_dim_sorted};

fn tensor_3d(requires_grad: bool) -> Tensor<f32> {
    Tensor::from_storage(
        TensorStorage::cpu(vec![
            1.0_f32, 5.0, 3.0, 2.0, 4.0, 4.0, 0.0, 9.0, 8.0, 7.0, 6.0, 6.0, 2.0, 10.0, -1.0, 3.0,
            5.0, 5.0, 5.0, 1.0, 0.0, -2.0, 11.0, 11.0,
        ]),
        vec![2, 3, 4],
        requires_grad,
    )
    .unwrap()
}

#[test]
fn topk_dim_middle_axis_matches_torch_values_and_indices() {
    let input = tensor_3d(false);

    let (values, indices) = topk_dim(&input, 2, 1, true).unwrap();

    assert_eq!(values.shape(), &[2, 2, 4]);
    assert_eq!(
        values.data().unwrap(),
        &[
            8.0, 7.0, 6.0, 9.0, 4.0, 5.0, 3.0, 6.0, 5.0, 10.0, 11.0, 11.0, 2.0, 5.0, 5.0, 3.0,
        ]
    );
    assert_eq!(
        indices,
        vec![2, 2, 2, 1, 1, 0, 0, 2, 1, 0, 2, 2, 0, 1, 1, 0]
    );
}

#[test]
fn topk_dim_sorted_false_matches_torch_cpu_selection_order() {
    // Live torch 2.11.0+cu130 CPU:
    // torch.topk([[3,1,4,2],[0,9,-1,8]], 3, dim=-1, largest=True, sorted=False)
    // values [[4,3,2],[8,9,0]], indices [[2,0,3],[3,1,0]].
    let input = Tensor::from_storage(
        TensorStorage::cpu(vec![3.0_f32, 1.0, 4.0, 2.0, 0.0, 9.0, -1.0, 8.0]),
        vec![2, 4],
        false,
    )
    .unwrap();

    let (values, indices) = topk_dim_sorted(&input, 3, -1, true, false).unwrap();

    assert_eq!(values.shape(), &[2, 3]);
    assert_eq!(values.data().unwrap(), &[4.0, 3.0, 2.0, 8.0, 9.0, 0.0]);
    assert_eq!(indices, vec![2, 0, 3, 3, 1, 0]);
}

#[test]
fn topk_dim_scalar_matches_torch_special_case() {
    let input = Tensor::from_storage(TensorStorage::cpu(vec![7.0_f32]), vec![], true).unwrap();

    for k in [0, 1] {
        let (values, indices) = topk_dim(&input, k, 0, true).unwrap();
        assert_eq!(values.shape(), &[] as &[usize]);
        assert_eq!(values.data().unwrap(), &[7.0]);
        assert_eq!(indices, vec![0]);
    }

    assert!(topk_dim(&input, 2, 0, true).is_err());
}

#[test]
fn topk_dim_backward_scatters_cotangent_along_selected_axis() {
    let input = tensor_3d(true);
    let (values, _indices) = topk_dim(&input, 2, 1, true).unwrap();
    let cotangent = Tensor::from_storage(
        TensorStorage::cpu((1..=16).map(|v| v as f32).collect()),
        vec![2, 2, 4],
        false,
    )
    .unwrap();

    values.backward_with_gradient(&cotangent).unwrap();

    let grad = input.grad().unwrap().unwrap();
    assert_eq!(
        grad.data().unwrap(),
        &[
            0.0, 6.0, 7.0, 0.0, 5.0, 0.0, 0.0, 4.0, 1.0, 2.0, 3.0, 8.0, 13.0, 10.0, 0.0, 16.0, 9.0,
            14.0, 15.0, 0.0, 0.0, 0.0, 11.0, 12.0,
        ]
    );
}
