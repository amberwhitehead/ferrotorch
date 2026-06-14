//! PyTorch 2.11.0+cu130 oracle for `torch.masked.amin/amax` extrema:
//!
//! ```python
//! torch.masked.amin(torch.tensor([1., 2.]), mask=torch.tensor([False, False]))
//! # tensor(inf)
//! torch.masked.amax(torch.tensor([1., 2.]), mask=torch.tensor([False, False]))
//! # tensor(-inf)
//! torch.masked.amin(torch.empty(0), mask=torch.empty(0, dtype=torch.bool))
//! # IndexError: amin(): Expected reduction dim 0 to have non-zero size.
//! x = torch.tensor([1., 2.], requires_grad=True)
//! torch.masked.amax(x, mask=torch.tensor([False, False])).backward()
//! x.grad
//! # tensor([0., 0.])
//! ```

use ferrotorch_core::masked::{MaskedTensor, masked_max, masked_min};
use ferrotorch_core::{Tensor, TensorStorage};

fn tensor_f32(data: &[f32], requires_grad: bool) -> Tensor<f32> {
    Tensor::from_storage(
        TensorStorage::cpu(data.to_vec()),
        vec![data.len()],
        requires_grad,
    )
    .expect("tensor")
}

fn tensor_f64(data: &[f64], requires_grad: bool) -> Tensor<f64> {
    Tensor::from_storage(
        TensorStorage::cpu(data.to_vec()),
        vec![data.len()],
        requires_grad,
    )
    .expect("tensor")
}

#[test]
fn all_masked_nonempty_extrema_return_torch_identity_payloads() {
    let x32 = tensor_f32(&[1.0, 2.0], false);
    let mt32 = MaskedTensor::new(x32, vec![false, false]).expect("masked tensor");
    assert_eq!(
        masked_min(&mt32).expect("amin").data().unwrap(),
        &[f32::INFINITY]
    );
    assert_eq!(
        masked_max(&mt32).expect("amax").data().unwrap(),
        &[f32::NEG_INFINITY]
    );

    let x64 = tensor_f64(&[1.0, 2.0], false);
    let mt64 = MaskedTensor::new(x64, vec![false, false]).expect("masked tensor");
    assert_eq!(
        masked_min(&mt64).expect("amin").data().unwrap(),
        &[f64::INFINITY]
    );
    assert_eq!(
        masked_max(&mt64).expect("amax").data().unwrap(),
        &[f64::NEG_INFINITY]
    );
}

#[test]
fn empty_extrema_error_like_torch() {
    let x = tensor_f32(&[], false);
    let mt = MaskedTensor::new(x, vec![]).expect("empty masked tensor");

    assert!(
        masked_min(&mt).is_err(),
        "torch.masked.amin raises on empty input"
    );
    assert!(
        masked_max(&mt).is_err(),
        "torch.masked.amax raises on empty input"
    );
}

#[test]
fn all_masked_extrema_backward_reaches_leaf_with_zero_grad() {
    let x = tensor_f32(&[1.0, 2.0], true);
    let mt = MaskedTensor::new(x.clone(), vec![false, false]).expect("masked tensor");
    let y = masked_max(&mt).expect("amax");
    assert!(y.requires_grad(), "all-masked amax must still carry graph");
    assert_eq!(y.data().unwrap(), &[f32::NEG_INFINITY]);

    y.backward().expect("backward");
    let grad = x.grad().expect("grad access").expect("leaf grad");
    assert_eq!(grad.device(), x.device());
    assert_eq!(grad.data().unwrap(), &[0.0, 0.0]);
}
