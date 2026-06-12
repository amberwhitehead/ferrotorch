//! Red-then-green regression test for audit finding CORE-071 (crosslink
//! #1765, CLASS-V): `PackedNestedTensor::mean_per_component` returned a
//! fabricated finite `0` for empty components. An arithmetic mean over
//! zero elements is undefined; torch's floating reductions return NaN.
//!
//! torch oracle (live session, torch 2.11.0+cu130):
//!
//! ```python
//! >>> torch.tensor([]).mean().item()
//! nan
//! ```
//!
//! The pre-fix in-module test `packed_mean_handles_empty_component_as_zero`
//! locked in the divergence; it is replaced by the NaN contract.

use ferrotorch_core::nested::PackedNestedTensor;

#[test]
// reason: the non-empty component mean (1.5) is exact in binary (3.0 / 2.0),
// so bitwise equality is the right assertion for it.
#[allow(clippy::float_cmp)]
fn core071_empty_component_mean_is_nan_f32() {
    let p =
        PackedNestedTensor::from_sequences(vec![vec![1.0f32, 2.0], vec![]], &[2, 0], &[]).unwrap();
    let means = p.mean_per_component();
    assert_eq!(means[0], 1.5, "non-empty component mean unchanged");
    assert!(
        means[1].is_nan(),
        "torch oracle: torch.tensor([]).mean() is nan; got {} — a plausible \
         finite 0 silently biases downstream aggregation",
        means[1]
    );
}

#[test]
fn core071_empty_component_mean_is_nan_f64() {
    let p = PackedNestedTensor::from_sequences(
        vec![vec![], vec![4.0f64, 6.0], vec![]],
        &[0, 2, 0],
        &[],
    )
    .unwrap();
    let means = p.mean_per_component();
    assert!(means[0].is_nan(), "empty leading component must be NaN");
    assert!(
        (means[1] - 5.0).abs() < f64::EPSILON,
        "non-empty mean unchanged"
    );
    assert!(means[2].is_nan(), "empty trailing component must be NaN");
}

/// A zero-NUMEL component arising from a zero-containing tail (`[L, 0]`)
/// is also an empty reduction — NaN per the same torch contract.
#[test]
fn core071_zero_tail_component_mean_is_nan() {
    let p = PackedNestedTensor::<f32>::from_sequences(vec![vec![]], &[3], &[0]).unwrap();
    let means = p.mean_per_component();
    assert!(
        means[0].is_nan(),
        "mean over a [3, 0] component (0 elements) must be NaN, got {}",
        means[0]
    );
}
