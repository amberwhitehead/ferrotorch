//! Probe for #805: BoolTensor / IntTensor empty-shape numel divergence.
//!
//! PyTorch parity table:
//!   shape=[]      -> numel == 1   (0-d scalar)
//!   shape=[0]     -> numel == 0   (1-D empty)
//!   shape=[2,0,3] -> numel == 0   (any zero-axis collapses to empty)
//!
//! Pre-fix expectation: `BoolTensor::zeros(&[0]).numel() == 1` (BUG).
//! Post-fix expectation: all rows of the table above hold.

use ferrotorch_core::{BoolTensor, IntTensor};

#[test]
fn bool_zeros_empty_1d_has_numel_zero() {
    let t = BoolTensor::zeros(&[0]);
    assert_eq!(t.shape(), &[0_usize]);
    assert_eq!(t.numel(), 0, "BoolTensor::zeros(&[0]).numel() must be 0 (PyTorch parity)");
}

#[test]
fn bool_zeros_scalar_has_numel_one() {
    let t = BoolTensor::zeros(&[]);
    assert_eq!(t.shape(), &[] as &[usize]);
    assert_eq!(t.numel(), 1, "BoolTensor::zeros(&[]).numel() must be 1 (0-d scalar)");
}

#[test]
fn int_zeros_empty_1d_has_numel_zero() {
    let t = IntTensor::<i64>::zeros(&[0]);
    assert_eq!(t.shape(), &[0_usize]);
    assert_eq!(t.numel(), 0, "IntTensor::zeros(&[0]).numel() must be 0 (PyTorch parity)");
}

#[test]
fn int_zeros_scalar_has_numel_one() {
    let t = IntTensor::<i64>::zeros(&[]);
    assert_eq!(t.shape(), &[] as &[usize]);
    assert_eq!(t.numel(), 1, "IntTensor::zeros(&[]).numel() must be 1 (0-d scalar)");
}

#[test]
fn int_arange_zero_returns_empty_1d() {
    let t = IntTensor::<i64>::arange(0).expect("arange(0) must succeed and return empty 1-D");
    assert_eq!(t.shape(), &[0_usize]);
    assert_eq!(t.numel(), 0);
}

#[test]
fn bool_from_vec_empty_1d() {
    let t = BoolTensor::from_vec(vec![], vec![0])
        .expect("from_vec(vec![], vec![0]) must succeed for shape=[0]");
    assert_eq!(t.shape(), &[0_usize]);
    assert_eq!(t.numel(), 0);
}

#[test]
fn int_from_vec_empty_1d() {
    let t = IntTensor::<i64>::from_vec(vec![], vec![0])
        .expect("from_vec(vec![], vec![0]) must succeed for shape=[0]");
    assert_eq!(t.shape(), &[0_usize]);
    assert_eq!(t.numel(), 0);
}

#[test]
fn bool_zeros_multi_dim_with_zero_axis() {
    let t = BoolTensor::zeros(&[2, 0, 3]);
    assert_eq!(t.shape(), &[2_usize, 0, 3]);
    assert_eq!(t.numel(), 0, "any-dim shape with a zero axis is empty");
}

#[test]
fn int_zeros_multi_dim_with_zero_axis() {
    let t = IntTensor::<i32>::zeros(&[2, 0, 3]);
    assert_eq!(t.shape(), &[2_usize, 0, 3]);
    assert_eq!(t.numel(), 0, "any-dim shape with a zero axis is empty");
}

#[test]
fn bool_ones_empty_1d_has_numel_zero() {
    let t = BoolTensor::ones(&[0]);
    assert_eq!(t.shape(), &[0_usize]);
    assert_eq!(t.numel(), 0);
}

#[test]
fn bool_reshape_to_empty_preserves_numel_zero() {
    // Build an empty tensor via from_vec, reshape to another empty layout.
    let t = BoolTensor::from_vec(vec![], vec![0]).expect("empty");
    let r = t.reshape(&[2, 0, 3]).expect("reshape empty -> empty must succeed");
    assert_eq!(r.shape(), &[2_usize, 0, 3]);
    assert_eq!(r.numel(), 0);
}

#[test]
fn int_reshape_to_empty_preserves_numel_zero() {
    let t = IntTensor::<i32>::from_vec(vec![], vec![0]).expect("empty");
    let r = t.reshape(&[0, 5]).expect("reshape empty -> empty must succeed");
    assert_eq!(r.shape(), &[0_usize, 5]);
    assert_eq!(r.numel(), 0);
}
