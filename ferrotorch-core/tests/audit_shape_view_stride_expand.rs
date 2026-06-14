//! Shape/view/stride regression probes for `expand`.
//!
//! PyTorch parity pinned here:
//! - `expand` is a metadata-only view, not a materialized broadcast copy;
//! - actually expanded dimensions use stride 0;
//! - synthetic leading size-1 dimensions keep contiguous-style strides;
//! - zero-sized expanded outputs are valid and still metadata-only;
//! - backward reduces broadcast multiplicity back to the original shape.

use ferrotorch_core::grad_fns::shape::expand;
use ferrotorch_core::{Tensor, TensorStorage, backward};

fn leaf(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
    Tensor::from_storage(
        TensorStorage::cpu(data.to_vec()),
        shape.to_vec(),
        requires_grad,
    )
    .expect("cpu tensor")
}

#[test]
fn expand_2d_is_zero_stride_view_not_materialized_copy() {
    let x = leaf(&[1.0, 2.0, 3.0], &[1, 3], false);
    let y = expand(&x, &[2, 3]).expect("expand");

    assert_eq!(y.shape(), &[2, 3]);
    assert_eq!(y.strides(), &[0isize, 1]);
    assert_eq!(y.storage_len(), x.storage_len());
    assert_eq!(y.storage_offset(), x.storage_offset());
    assert_eq!(
        y.data_vec().expect("logical data"),
        &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    );
}

#[test]
fn expand_leading_size_one_dim_matches_torch_stride() {
    let x = leaf(&[1.0, 2.0, 3.0], &[3], false);
    let y = expand(&x, &[1, 3]).expect("expand");

    // torch.tensor([1,2,3]).expand(1, 3).stride() == (3, 1)
    assert_eq!(y.shape(), &[1, 3]);
    assert_eq!(y.strides(), &[3isize, 1]);
    assert_eq!(y.storage_len(), x.storage_len());
    assert_eq!(y.data_vec().expect("logical data"), &[1.0, 2.0, 3.0]);
}

#[test]
fn expand_scalar_uses_zero_strides_for_all_expanded_dims() {
    let x = leaf(&[7.0], &[], false);
    let y = expand(&x, &[2, 3]).expect("expand scalar");

    // torch.tensor(7.).expand(2, 3).stride() == (0, 0)
    assert_eq!(y.shape(), &[2, 3]);
    assert_eq!(y.strides(), &[0isize, 0]);
    assert_eq!(y.storage_len(), x.storage_len());
    assert_eq!(
        y.data_vec().expect("logical data"),
        &[7.0, 7.0, 7.0, 7.0, 7.0, 7.0]
    );
}

#[test]
fn expand_to_zero_extent_is_metadata_only_and_empty() {
    let x = leaf(&[5.0], &[1], false);
    let y = expand(&x, &[2, 0]).expect("expand to zero");

    // torch.tensor([5.]).expand(2, 0).stride() == (0, 0)
    assert_eq!(y.shape(), &[2, 0]);
    assert_eq!(y.strides(), &[0isize, 0]);
    assert_eq!(y.storage_len(), x.storage_len());
    assert!(y.data_vec().expect("logical data").is_empty());
}

#[test]
fn expand_backward_sums_broadcast_multiplicity() {
    let x = leaf(&[1.0, 2.0, 3.0], &[1, 3], true);
    let y = expand(&x, &[2, 3]).expect("expand");
    let loss = y
        .contiguous()
        .expect("materialize view")
        .sum_all()
        .expect("sum");

    backward(&loss).expect("backward");
    let grad = x.grad().expect("grad access").expect("leaf grad");
    assert_eq!(grad.data_vec().expect("grad data"), &[2.0, 2.0, 2.0]);
}
