//! Pin: HQQ Q4 axis=1 loader must ERROR on unsupported configs rather than
//! silently produce wrong weights (#1172, scrutiny item 4).
//!
//! ferrotorch only handles `nbits=4`, `axis=1`, 2-D shape. Other configs
//! (nbits!=4 nested/2-bit/3-bit, axis=0 per-input-channel grouping, missing
//! shape) are NOT implemented; the contract is that they return an `Err`,
//! never an Ok dense weight computed under the wrong assumption.
//!
//! These assert the rejection contract. They are regression coverage that
//! the guard rails hold; if a future change makes any of these return Ok
//! with wrong-but-plausible weights, the test catches it.

use ferrotorch_core::{Tensor, TensorStorage};
use ferrotorch_llama::hqq_q4_axis1_to_dense;

fn t(d: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(d), shape, false).unwrap()
}

/// nbits=3 (a real HQQ packing) must be rejected, not dequantized as 4-bit.
#[test]
fn hqq_q4_axis1_rejects_nbits_3() {
    let w_q = t(vec![0.0; 8], vec![2, 4]);
    let scale = t(vec![1.0; 4], vec![4, 1]);
    let zero = t(vec![0.0; 4], vec![4, 1]);
    let r = hqq_q4_axis1_to_dense(&w_q, &scale, &zero, 3, 4, 1, &[4, 4]);
    assert!(
        r.is_err(),
        "nbits=3 must be rejected, got {:?}",
        r.map(|_| ())
    );
}

/// axis=0 (per-input-channel grouping, the HQQ default for some configs)
/// must be rejected — the byte layout and reshape differ from axis=1.
#[test]
fn hqq_q4_axis1_rejects_axis_0() {
    let w_q = t(vec![0.0; 8], vec![2, 4]);
    let scale = t(vec![1.0; 4], vec![4, 1]);
    let zero = t(vec![0.0; 4], vec![4, 1]);
    let r = hqq_q4_axis1_to_dense(&w_q, &scale, &zero, 4, 4, 0, &[4, 4]);
    assert!(
        r.is_err(),
        "axis=0 must be rejected, got {:?}",
        r.map(|_| ())
    );
}

/// A non-2-D shape must be rejected.
#[test]
fn hqq_q4_axis1_rejects_non_2d_shape() {
    let w_q = t(vec![0.0; 8], vec![2, 4]);
    let scale = t(vec![1.0; 4], vec![4, 1]);
    let zero = t(vec![0.0; 4], vec![4, 1]);
    let r = hqq_q4_axis1_to_dense(&w_q, &scale, &zero, 4, 4, 1, &[2, 2, 4]);
    assert!(
        r.is_err(),
        "3-D shape must be rejected, got {:?}",
        r.map(|_| ())
    );
}
